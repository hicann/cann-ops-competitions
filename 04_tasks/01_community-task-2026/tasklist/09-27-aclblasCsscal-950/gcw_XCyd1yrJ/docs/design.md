# aclblasCsscal 算子设计文档

本次修订关联已合并的[设计PR #1457](https://gitcode.com/cann/cann-ops-competitions/pull/1457)，按当前保留实现更新设计。代码仓为[2401_87688128/ops-blas](https://gitcode.com/2401_87688128/ops-blas)，代码PR为[#477](https://gitcode.com/cann/ops-blas/pull/477)，当前提交为3a314e97710e28ab33c386bb3d3931bdb6cb1640，实测提交为b7cb689a16c60421b33e1af85b3f4f6c6f90f07e；当前提交仅对Host启动配置、Kernel输入载入和测试辅助函数做等价拆分，Kernel、tiling、计时路径、官方CSV输入和精度判定逻辑保持一致。本次修订的评审状态以本PR为准，不沿用旧PR的通过结论。

# 需求背景（required）

## 需求来源

CANN训练营东南大学社区任务“aclblasCsscal算子开发(950)”。依据任务书§2功能、§3验收、§4交付及§5合入要求，在ops-blas公共接口框架中为Ascend 950PR提供Ascend C直调实现。

模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md 。模板中的Addcdiv/TBE为示例，本设计替换为CSScal实际接口和cuBLAS/Netlib参照，不引入TBE对比项。

## 背景介绍

### aclblasCsscal算子实现

CSScal是BLAS Level-1复数向量乘实数标量操作。complex64由连续的实部float32和虚部float32构成，两分量分别乘alpha，不进行复数交叉乘法。更新直接写回x。

### 现有工程与接口分析

公开声明已在include/cann_ops_blas.h中存在，复用仓内aclblasComplex及句柄结构。实现文件为blas/scal/arch35/csscal_host.cpp、csscal_kernel.cpp、csscal_tiling_data.h；测试位于test/scal/csscal/，基于仓内CSV、填充器、GTest及cblas封装。连续访存使用AIV向量乘，正步长使用SIMT。

### 算子功能分析

| 参数 | 含义 | 类型/位置 | 形状与约束 |
|---|---|---|---|
| handle | 携带绑定stream的句柄 | aclblasHandle_t，Host | 非空，否则返回HANDLE_IS_NULLPTR |
| n | 复数元素个数 | int，Host | n≤0合法no-op |
| alpha | 实数乘数 | const float*，Host | 实际计算时指针可读且非空 |
| x | 原地更新向量 | aclblasComplex*，Device | 逻辑[n]；n>0、incx>0时物理长度1+(n−1)incx |
| incx | 复数元素步长 | int，Host | incx≤0合法no-op；正步长执行 |

无额外leading dimension、广播、输出视图或动态shape框架需求；n仍为运行时参数。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasCsscal(
    aclblasHandle_t handle, int n, const float* alpha,
    aclblasComplex* x, int incx);
```

支持complex64输入输出与float32实数标量，遵循任务书规定的参数校验、no-op和alpha=0置零语义。沿handle绑定stream异步提交，不定义950PR私有平行接口。

## 需求拆解

1. 功能：按incx更新逻辑元素，两分量独立计算；间隙元素不修改。
2. 精度：cblas golden；实部、虚部分别按FLOAT32标准验证。alpha=±0另验bit-exact正零。
3. 性能：10次预热后100次有效调用；三个标杆和全部200条ratio要求均检查。按官方回复以msprof采集设备Kernel耗时，保守上界、Host端到端读数分别保留。
4. 工程：使用官方固定上游7eae232的build.sh、顶层CMake与共享测试框架；不添加私有优化开关，不替换官方CSV输入。
5. 可复现性：记录实测SHA、实际编译命令、加载库、产物哈希和完整日志；用独立附加测试补分布及指针偏移覆盖。

# 详细设计（required）

## 算子分析

### 数学公式

对i=0,…,n−1，j=i×incx：

```text
x[j].real = alpha * x[j].real
x[j].imag = alpha * x[j].imag
```

特殊路径：n≤0或incx≤0不更新；alpha=1不更新；alpha=0（包括−0）直接写+0，不使用0×Inf/NaN计算。后者按任务书明确语义处理。

### 支持数据类型

alpha为FLOAT32，x为COMPLEX64（两分量FLOAT32），n/incx为接口规定的int。既有接口签名和类型定义不变。

### 支持形状

逻辑一维[n]。连续路径把n个复数解释为F=2n个float。正步长路径按复数元素计算物理偏移，调用者须提供实际可访问的Device存储；不承诺所有数学尺寸均可实际分配。

## 算子实现

### 实现方案

| 条件 | 路径 | 计算方式 |
|---|---|---|
| n≤0、incx≤0或alpha=1 | Host快速返回 | 不启动Kernel |
| incx=1 | csscal_aiv_kernel | GM→UB、Muls/Duplicate、UB→GM |
| incx>1 | csscal_simt_kernel | 每线程按步长读写复数两分量 |

#### 3.2.1 host侧设计：

校验顺序为：handle判空 → n/incx合法no-op → alpha判空 → x判空 → alpha=1快速返回 → 查询AIV核数。空handle返回ACLBLAS_STATUS_HANDLE_IS_NULLPTR；计算路径空alpha/x返回ACLBLAS_STATUS_INVALID_VALUE；核数查询为0返回ACLBLAS_STATUS_EXECUTION_FAILED。

##### 1. 分核策略：

令C=GetAivCoreCount()。连续路径先以uint64_t计算F=2n，确认可放入uint32_t后缩窄：

```text
B = min(ceil(F/8), C)
若 n <= 131072：B = min(B, max(1, floor(n/2048)))
p = floor(floor(F/B)/8)*8
L = F - p*B
e = floor(L/8)
t = L % 8
offset(i) = i*p + min(i,e)*8
count(i) = p + (i<e ? 8 : 0) + (i==B-1 ? t : 0)
```

单位为float；每核起点32B相对对齐，区间无重叠地覆盖[0,F)。F为偶数，因此不拆分复数分量对。核数按设备查询，不硬编码56；56是本次设备观测值。

正步长路径U=min(ceil(n/128),C)，按复数元素分配：q=floor(n/U)，r=n%U；第i核处理q+(i<r)个元素，起点i*q+min(i,r)。线程数min(ceilAlign(ceil(n/U),128),2048)，每线程循环步长blockDim.x。

##### 2. 数据分块和内存优化策略：

UB_SIZE使用仓内常量248×1024=253952字节，不是运行时设备容量探测。连续路径tileSize=floor(UB_SIZE/(4×sizeof(float))/8)×8=15872个float。输入和输出TQue类型深度均为2。

单核实际处理量count≤tileSize时，输入、输出各初始化一个物理缓冲；超过tileSize时各初始化两个。每缓冲容量为ceil(min(count,tileSize)/8)×8×4字节。非零alpha多tile最多4×15872×4=253952字节；单tile最多126976字节。alpha=0不初始化输入队列，减少读取及分配。以上为源码UB预算，不等同进程设备内存。

同一输入/输出GM地址原地更新，无额外GM workspace，启动包装传workspace=nullptr。正常非零操作的算法读写量约16n字节，不据此推断物理HBM流量或缓存命中率。

##### 3. tilingkey规划策略：

没有独立TilingKey注册。Host启动包装根据incx选择AIV/SIMT入口；AIV内部根据count和alpha选择单tile、多tile、置零或乘法分支。Tiling结构为：

```cpp
struct CsscalTilingData {
    uint32_t totalN;          // AIV: float数；SIMT: 复数数
    uint32_t perCoreN;
    uint32_t extraBlockCores;
    uint32_t tailElements;
    uint32_t tileSize;
    float alpha;
    int64_t incx;
    uint32_t useCoreNum;
    uint32_t nthreads;
};
```

#### 3.2.2 kernel侧设计：

**初始化与单tile。** TPipe位于Kernel入口，CsscalAIV解析tiling并计算本核offset/count。空核返回；count≤tileSize直接调用SingleIteration，不进入预取循环。非零alpha搬入、输入EnQue/DeQue、Muls、FreeTensor；alpha=0使用Duplicate(out,0.0f,count)。输出EnQue/DeQue后写回并释放，依赖队列管理流水同步，不引入手工事件或单UB原地计算。

**多tile。** 非零alpha先LoadNext首块。每次循环取得当前输入，提交Muls，释放当前输入，再搬入下一块；之后输出EnQue/DeQue并写回当前块。最后一块只计算有效长度；仅存在下一块时计算remaining，避免无符号回绕。该结构提供搬运重叠机会，不声称所有阶段完全重叠。最终保留的是“当前计算后提交下一块搬入”的顺序。

**对齐与尾块。** x基址32B对齐且块长度为8个float的倍数时使用DataCopy；否则使用一次DataCopyPad搬运该块全部有效字节，并在UB末尾补零至32B。写回仅写count×4字节，不写padding。基址允许有效complex元素偏移，例如实测8/16/24字节偏移；不把32B基址对齐作为额外API限制。

**SIMT。** 复数索引为(uint64_t(startOffset)+i)×uint64_t(incx)，float索引为其两倍。非零alpha分别读乘写实虚部；alpha=±0分别写0.0f。只访问逻辑元素，步长间隙保持不变。

两入口均使用KERNEL_TYPE_AIV_ONLY，通过handle->stream异步提交。算子内部不增加Host同步；测试在读回前同步，性能fixture逐次同步绑定的stream。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Ascend 950PR（arch35），CANN 9.1.0 | √ |

本次不据950PR结果推断950DT或其他平台已验证。

## 算子约束限制

无广播、无额外leading dimension、无额外输出张量；x为Device指针，alpha为Host指针。正步长支持由incx表达的非连续访问；非正步长是任务书规定的no-op。调用者负责缓冲区有效性和生命周期。

build.sh和顶层CMake保持固定官方上游7eae232原文，使用默认Release；本次实际bisheng命令含-O3 -DNDEBUG。没有csscal私有-O2、baseline/phase2开关或fast-math选项。新增测试CMake仅调用仓库已有测试构建函数。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | 实部/虚部分别按FLOAT32验证：rtol=2^-10、atol=2^-16、matched_ratio≥0.99；绝对误差限制按官方验证器的max(1e-2,32×ULP)检查。零标量另验正零位模式 | 任务书§3.2及官方测试框架 |
| 三标杆 | n=1048576/2097152/4194304，incx=1，平均单次耗时≤13.57/21.05/43.02μs | 任务书§3.3 |
| 全量性能 | 200个官方性能case，GPU_ms×1000/平均Kernel_us≥0.4，预热后有效采样>50 | 配套基线及官方回复 |
| 内存 | §3.4无性能阈值；§4自测报告仍记录内存占用 | 任务书§3.4、§4 |

**覆盖。** 官方CSV1200条原样保留：1000精度、200性能。主精度集合1000+14独立功能测试=1014项；另有SupplementalDistributionCoverage和SupplementalDevicePointerOffsets共2项。前者32组组合中alpha/x各16组均匀、16组正态，实虚部独立生成；后者覆盖8/16/24B指针偏移、incx=1/3及保护区。官方CSV本身不宣称50%正态覆盖。

**输入和测量。** 性能每次调用前恢复官方CSV指定输入，避免连续缩放溢出；不使用旧版小幅值替代输入。每case预热10次、有效100次，最终输出全量分量golden校验。Host端到端avg_us含一次API及stream同步；官方6KB脚本读取GTest整用例毫秒上界；按官方回复另用msprof设备Task Duration(us)计算Kernel平均耗时。这三个口径分别存档，不改写失败或NO_REF。

**本次实测。** 2026-09-18对b7cb689归档产物完成1014+2项精度，通过。全量profile为22000条同stream目标Kernel记录，结合实际GTest顺序、固定源码每次一个Kernel、每case110次、核数及无重叠核验映射；剔除每case前10次后对100次取算术平均，保留全部慢样本。200/200按Kernel口径通过，最低ratio=0.6371388646。

| case | 平均Kernel（μs） | ratio | 任务书上限（μs） |
|---|---:|---:|---:|
| TC_PF_1001 | 7.59993 | 0.7144803 | 13.57 |
| TC_PF_1002 | 12.70686 | 0.6625555 | 21.05 |
| TC_PF_1003 | 23.05570 | 0.7463664 | 43.02 |

独立内存补测复用同一产物：设备全局HBM采样最大5559MB、进程设备占用363MB、Host RSS最大232952KiB（约227.5MiB）。它们分别属于整卡共享、进程设备及Host内存，不相加，不表示每case独占或连续真实峰值。采样循环休眠1秒，运行期实际相邻记录约3.22～4.57秒。结果原始记录见同交付包自测报告及内存目录。

**复现与维护。** 使用仓库test/scal/csscal/README.md和交付件“自测用例及测试代码/README.md”。官方原件不变；官方GTest整项结果与msprof设备Kernel结果分开记录。构建日志、加载路径、二进制哈希及原始PROF一同溯源；解析异常不能产生达标结论。

## 兼容性分析

公共接口签名、aclblasComplex定义和stream使用方式保持不变。架构专属实现位于arch35，不改其他架构的算子代码及共享测试框架。内部Tiling只用于Host/Kernel成对构建，不暴露为公共ABI。此次交付无私有平行接口；README仅声明Ascend 950PR支持。

文档描述的是保留实现，不包括已回退的SIMT小尺寸切换、单UB手工同步、独立单tile入口、64-float分区或提前预取候选。历史设计以PR #1457原文为准；本次更新申请重新评审当前实现设计。

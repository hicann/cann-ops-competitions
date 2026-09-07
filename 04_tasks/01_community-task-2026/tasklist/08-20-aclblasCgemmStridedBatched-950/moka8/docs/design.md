# 需求背景（required）

## 需求来源

《8月社区任务—aclblasCgemmStridedBatched 算子开发（950）》任务书，目标环境为 Ascend 950PR、CANN 9.1.0；对应[官方任务列表](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/README.md)第 20 项。本文遵循[社区任务提交说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)，采用[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/f3f5a5c282a4d290763436d821943a0e443e161e/04_tasks/01_community-task-2026/resources/design_template.md)的章节结构。

提交账号及个人目录：`moka8`。实现参考为 [moka8/ops-blas](https://gitcode.com/moka8/ops-blas)，分支 `master`，固定提交 [`c8beaa40c0e4ce76632968c7508404a5faca3ee8`](https://gitcode.com/moka8/ops-blas/commit/c8beaa40c0e4ce76632968c7508404a5faca3ee8)。本文描述该版本的接口、算法与自验设计。

## 背景介绍

### 上游实现现状

开发基线为 ops-blas 提交 `2760346765b92e40579d16d8fffdb9f0988c35df`。上游已提供句柄、stream、workspace 管理，以及实数 `aclblasSgemmStridedBatched` 的 Host/Ascend C/GTest 工程。本实现复用这些工程能力，扩展公共复数接口和独立复数测试。complex64 使用公共 `aclblasComplex`，实部、虚部各为 float32。

### 功能分析

批次 i 执行 `C_i = alpha * op(A_i) * op(B_i) + beta * C_i`。A/B/C 为列主序矩阵，alpha/beta 为全批共用的 Host 复数标量，输出原地写入 C。`op` 支持原矩阵 N、转置 T、共轭转置 C，组合共九种。

# 需求分析（required）

## 需求描述

在公共头文件 `include/cann_ops_blas.h` 新增与 cuBLAS 参数顺序一致的 `aclblasCgemmStridedBatched`。代码位于 `blas/gemm_strided_batched/arch35/`，通过句柄绑定的 stream 直接发射 Ascend C kernel。测试扩展仓库 CSV/GTest 框架，并逐批调用 Netlib `cblas_cgemm` 生成 Golden。

### 公共接口与参数约束

```cpp
aclblasStatus_t aclblasCgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, int64_t strideA,
    const aclblasComplex* B, int ldb, int64_t strideB,
    const aclblasComplex* beta, aclblasComplex* C, int ldc,
    int64_t strideC, int batchCount);
```

| 参数 | 含义与约束 |
| --- | --- |
| `handle` | 有效的库句柄，携带执行 stream；空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。 |
| `transA/transB` | 分别取 `ACLBLAS_OP_N/T/C`，非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| `m/n/k/batchCount` | 非负整数；负数返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| `alpha/beta` | 非空 Host 指针，指向全批次共用的 `aclblasComplex` 标量。 |
| `A/B` | 只读 Device 地址；非零 m/n/batch 且 k>0 时不能为空，即使 alpha=0 也保持此项参数校验。 |
| `lda` | transA=N 时至少为 `max(1,m)`，transA=T/C 时至少为 `max(1,k)`。 |
| `ldb` | transB=N 时至少为 `max(1,k)`，transB=T/C 时至少为 `max(1,n)`。 |
| `C/ldc` | C 为 Device 原地输出，`ldc >= max(1,m)`；beta=0 可不初始化旧值，实际写回仍需要有效输出存储。 |
| `strideA/B/C` | 有符号 64 位复数元素偏移；A/B 的零步长广播，C 批次必须互不重叠；由调用者保证完整地址范围有效。 |

正常执行或合法 no-op 返回 `ACLBLAS_STATUS_SUCCESS`；其余状态沿用公共状态码定义。接口位于公共 C ABI 中，声明、定义与任务书参数顺序一致。

## 需求拆解

1. 支持运行时 m/n/k/batchCount、全部 N/T/C 组合、leading dimension padding、地址偏移和显式 stride。
2. stride 以复数元素为单位；A/B 的 stride=0 表示广播。负 stride 按调用者提供的有效基址定位。输出各批次不得重叠。
3. 校验空句柄、非法枚举、负维度、非法 leading dimension 和必要指针；保持仓库状态码。
4. 零 m/n/batchCount 为 no-op；k=0 或 alpha=0 执行 beta*C；beta=0 不读取旧 C。
5. 实部与虚部独立满足任务书精度标准，alpha=0 的缩放使用 EXACT 校验。
6. 完整 API 经过预热后采样 60 次；核对三个任务性能上限以及随包参考性能。
7. 自验覆盖均匀/正态各 50% 的随机矩阵及随机复数标量，另外覆盖固定值、NaN/Inf、边界和异常参数。

# 详细设计（required）

## 算子分析

### 数学公式

对于逻辑输出 `(r,j)`：

`C_i(r,j) = alpha * sum[p=0..k-1](op(A_i)(r,p) * op(B_i)(p,j)) + beta * C_i(r,j)`。

复数乘法为 `(ar*br-ai*bi, ar*bi+ai*br)`。共轭转置在转置地址映射后取虚部相反数。N 路径矩阵元素地址为 `base + batch*stride + column*ld + row`。

### 支持数据类型

输入、输出及 alpha/beta 为 complex64。公开 ABI 复用 `aclblasComplex`，不增加产品私有类型或平行接口。维度为 int，stride 为 int64_t。

### 支持形状

A 的逻辑形状为 m×k，B 为 k×n，C 为 m×n。N/T/C 决定物理行列数与最小 leading dimension。A/B 只读，只写 C 的逻辑区域，保留行 padding、批次间隙及保护区。

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

先校验 handle、枚举、维度、leading dimension 及标量指针，再处理零维 no-op。非空计算且 k>0 时 A/B 必须有效；实际写输出需要有效 C 存储。k=0 或 alpha=0 进入缩放路径，beta=1 时直接返回。分核数量通过平台接口获取，避免硬编码计算核数量。

Tiling 记录逻辑尺寸、对齐尺寸、ld/stride、转置属性、复数标量及各 workspace 平面偏移。所有批次定位和 workspace 偏移采用 64 位运算。workspace 在申请前检查容量，按 512 字节对齐，遵循上游句柄的 2 GiB workspace 限制。极瘦长矩阵若因补齐导致 workspace 预算超限，则使用无需临时空间的直接计算路径。空间由句柄拥有；本接口不独立释放共享 workspace。

按尺寸、批次数、标量及数值路径需要选择：缩放/SIMT、小矩阵本地 Cube 流水线、通用打包 Cube 路径。输入数据相关的数值保护在 Device 内完成，不读取测试名称、随机 seed 或 Golden。kernel 在 handle 的 stream 上排队，调用者在读取结果前同步。

#### 3.2.2 Kernel 侧设计

**缩放和极小矩阵。** AIV SIMT 将逻辑输出分配给不同线程。缩放路径单独处理 beta=0/1，固定复数中间运算顺序，支持 alpha=0 的位精确检查。极小矩阵通过一线程多输出复用数据，减少 kernel 调度开销。

**小矩阵本地流水线。** 将复数乘法展开为实数乘法，在 L1 中组织 `[Ar,Ai]`、`[-Ai,Ar]` 与 B 实/虚数据。输入拆为 FP16 高位和 FP16 残差，低×高、高×低、高×高以 FP32 累加，Fixpipe 写回交错复数输出。容量允许时采用双槽，ready/release 事件保护槽位；异常标志使用独立 32 字节区域，并在读取前使缓存失效。低批次矩阵沿行切为最多 32 行的子块，增加可并行 tile 数；两个 L0B 区域保留高位与残差操作数，第三次乘法复用第二次的 L0A 和第一次的 L0B，省去重复搬运。每次覆盖 L0A 前等待前次乘法完成。异常输入回到对应批次的原始 FP32 计算。

**单位标量 Cube。** alpha=1、beta=0 时，按列批量打包转置/共轭后的实、虚平面，广播矩阵只打包一次。无需累加顺序保护时可使用三乘形式：`P0=Br*Ar`、`P1=Bi*Ai`、`P2=(Br+Bi)*(Ar+Ai)`，再合成 `(P0-P1, P2-P0-P1)`。受保护分量的前缀分别计算所需实数乘积；两个分量均需保护时计算四个乘积。长 K 路径采用 FP16 高位/残差补偿和 FP32 累加。中等矩阵使用 MIX kernel 与核间屏障，大矩阵按同一 stream 执行打包、必要的格式规范化、Cube/有序累加以及结果合成。一般复数标量使用下面的直接有序计算路径。

小矩阵批分组结合核数、容量和总批次选择：可在一轮内完成的至多 32 阶矩阵允许更大分组，跨多轮时限制分组大小以减少尾轮负载不均。至多 128 阶的单步 K 乘法可复用前两个补偿乘积留下的高位 L0A/L0B；最后一次乘法占用两个槽位，完成事件同时保护两者。输出合成根据 UB 容量选择分列或整块处理，整块路径使用单输出缓冲，最大申请量为 250880 字节。

**累加顺序保护。** 一般复数标量保留 Netlib 的缩放与累加顺序，避免先求和后缩放产生不同 FP32 舍入。长 K 单位标量路径同时统计 A/B 分量均值，根据相干累加幅值分别选择实部、虚部或两者的数值保护。受保护分量由 Cube 计算有限长度的前缀，再从该前缀结果开始，使用 FP32 SIMD 依次累加剩余 K 项。前缀长度仅依据 K 决定，并限制最大长度以控制相干累加误差；全部 K 项均参与计算。

**布局整理与并行执行。** 有序路径以 64 行 tile 复用 A，B 按输出列分组，K 方向采用双缓冲。转置、共轭及非对齐输入先通过独立 SIMD kernel 整理为带零填充的 NN 布局，广播输入只整理一次。Cube 前缀完成后发出跨核事件；AIV 计算剩余 K 项的同时，AIC 可计算不需要有序累加的另一分量。跨核屏障保证前缀可见，后续独立合成 kernel 将本次调用已经生成的分量与 Cube 分量合并。该合成读取的是本次调用的中间输出，不依赖调用前旧 C 的值。

**资源隔离。** 有序计算与布局整理 kernel 仅包含 SIMD 实现，避免手动 UB 分区与 SIMT 运行时资源共用。SIMT 异常值处理位于独立调度路径。含队列的打包/合成对象由调用者传入 TPipe，TPipe 不作为类成员。输出尾块通过带长度的 DMA 写回，保护逻辑矩阵外的行 padding 和批次间隙。

**非有限值。** 非有限输入及超出半精度安全范围的输入从原始 FP32 值计算，避免半精度打包改变 NaN/Inf 传播。beta=0 的各路径均避免读旧 C。批次局部异常不得污染其他批次。

## 支持硬件

Ascend 950PR，CANN 9.1.0。arch35 注册项在 asc-devkit 版本不满足 9.1 要求时跳过算子和相应测试。公共接口保持产品共用。

## 算子约束限制

调用者保证给定 shape、ld、stride、偏移所涉及的 Device 地址有效；A/B 只读，C 各批次不重叠。`strideC=0 && batchCount>1` 的行为未定义，不构造此类正确性测试。支持范围以 ld/stride 所表达的列主序存储为限，不增加任意 Tensor stride 接口。任务未要求确定性，也未要求其他硬件产品线的实现。

# 可维可测分析

## 精度标准/性能标准

每批 Golden 使用明确加载的 Netlib BLAS `cblas_cgemm`，CPU Golden 完成后再执行验证与计时。实部、虚部分别按 `atol=2^-16`、`rtol=2^-10` 判断，matched ratio 至少 0.99；有限分量同时检查绝对误差不超过 `max(0.01, 32*ULP(golden))`。alpha=0 的结果比较位模式。NaN/Inf、C padding、批次间隙及分配保护区独立检查。

七组 CSV 共 3289 项：任务包 1200、补充 145、单点非有限值 72、流水线边界 183、正态 1024 场景 1、分布配对 1580、beta=0 的旧 C 污染检查 108。原始任务文件和随包 GPU 参考文件保持不变。配对生成器记录源文件 SHA256 和每行对应关系，均匀与正态的实部/虚部独立采样，正态 μ∈[-5,5]、σ∈[0.1,2]。随机 alpha/beta 的均匀与正态组各占一半，特殊标量另行覆盖。

性能使用完整 API 的 ACL Event 区间，预热 10 次、有效采样 60 次；输入生成、拷贝、Golden 和比对在计时区间外。三个任务上限如下；原包其余参考性能项按随包规则 `GPU_time / NPU_time >= 0.4` 检查。

| m=n=k | batchCount | 平均单次 API 耗时上限（us） |
| --- | --- | --- |
| 256 | 32 | 227.17 |
| 512 | 16 | 841.87 |
| 1024 | 8 | 3269.74 |

对应参数的均匀/正态变体沿用同一上限，所有 402 项性能观察共同决定报告中的性能结论。

内存报告记录 A/B/C Device 分配字节数及句柄 workspace 分配字节数，不将其表述为外部测量的进程峰值 HBM；任务书第 3.4 节不设内存比例上限。复现步骤见算子和测试 README。最终测试结论依据具体版本的 CSV/XML/log 与 `summary.json`，本文不替代测试报告。

## 兼容性分析

新增公共复数接口，不修改现有实数 API 签名。实数 Cube 流水线的共享提取需与上游基线对照，复数精度测试使用独立 Golden。版本门禁覆盖不满足 9.1 的编译跳过场景。所有验收材料关联同一组源文件及二进制校验值。


## 源码与复现入口

以下链接固定到本设计对应的实现提交，避免后续分支更新改变评审依据：

- [公共 API 声明](https://gitcode.com/moka8/ops-blas/blob/c8beaa40c0e4ce76632968c7508404a5faca3ee8/include/cann_ops_blas.h)
- [Host 校验、workspace 与调度](https://gitcode.com/moka8/ops-blas/blob/c8beaa40c0e4ce76632968c7508404a5faca3ee8/blas/gemm_strided_batched/arch35/cgemm_strided_batched_host.cpp)
- [计算流水线与 L0 复用](https://gitcode.com/moka8/ops-blas/blob/c8beaa40c0e4ce76632968c7508404a5faca3ee8/blas/gemm_strided_batched/arch35/cgemm_strided_batched_compute.h)
- [数值保护策略](https://gitcode.com/moka8/ops-blas/blob/c8beaa40c0e4ce76632968c7508404a5faca3ee8/blas/gemm_strided_batched/arch35/cgemm_strided_batched_policy.h)
- [算子说明与调用约束](https://gitcode.com/moka8/ops-blas/blob/c8beaa40c0e4ce76632968c7508404a5faca3ee8/blas/gemm_strided_batched/README_cgemm.md)
- [测试依赖、构建与完整自验步骤](https://gitcode.com/moka8/ops-blas/blob/c8beaa40c0e4ce76632968c7508404a5faca3ee8/test/gemm_strided_batched/cgemm_strided_batched/README.md)

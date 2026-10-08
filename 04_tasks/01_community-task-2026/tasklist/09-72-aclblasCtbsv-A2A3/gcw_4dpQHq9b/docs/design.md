# 需求背景（required）

## 需求来源

CANN 社区任务 **09-72：aclblasCtbsv-A2A3**。依据官方09-72任务书；结构对应官方 `04_tasks/01_community-task-2026/resources/design_template.md`（核对提交 `ef9062d88855f4a693bc5f9acaee207b6cfe8c17`）。本文记录实现与开发者自验证结果，不代表社区或华为已经验收。提交目录为 `04_tasks/01_community-task-2026/tasklist/09-72-aclblasCtbsv-A2A3/gcw_4dpQHq9b/docs/design.md`。

- 代码仓库：[gcw_4dpQHq9b/ops-blas](https://gitcode.com/gcw_4dpQHq9b/ops-blas)。
- 代码分支：`community/aclblas-ctbsv-a2a3`。
- 验收分支：`acceptance/aclblas-ctbsv-a2a3`。
- 数值证据子集位于本包 `task_submission/evidence/public-safe-evidence/`；开发者自验证不表示已通过官方验收。

本文源码路径按代码仓库组织；证据包中的冻结源码位于 `task_submission/evidence/public-safe-evidence/source/ops-blas/`，与仓库根目录的同名相对路径对应。

## 背景介绍

### aclblasCtbsv 算子实现优化

在 ops-blas 中增加 Ascend C kernel 直调接口，原地求解 complex64 三角带状线性方程，覆盖 Atlas A2/A3。生产代码位于 `blas/tbsv/arch22/`，公共声明位于 `include/cann_ops_blas.h`。

### aclblasCtbsv 算子实现现状分析

本任务补充 BLAS 直调算子，不是替换已验证的 TBE 实现。以 Netlib CTBSV 为数值参考、任务提供的 GPU 时延为性能基线；两种参考编译语义分别验证。

### aclblasCtbsv 算子功能分析

| 参数 | 含义与约束 |
| --- | --- |
| handle | 有效 BLAS handle；异步使用其 stream |
| uplo / trans / diag | 上/下三角；N/T/C；非单位/单位对角 |
| n / k / lda | 阶数、带宽、带存储列距；n、k 非负，lda ≥ k+1 |
| A | complex64 列主序带存储，分配范围 lda×n，不修改 |
| x / incx | complex64 原地输入输出；正或负非零步长，物理跨度为 (n−1)×abs(incx)+1 |

不涉及广播。UNIT 不读取对角元素；n=0 无操作，k=0 退化为对角求解。

# 需求分析（required）

## 需求描述

实现与任务规定 API、布局、异常返回和精度要求一致的 `aclblasCtbsv`；通过 A2、A3 实机验证并保留可复算原始证据。官方速度比下限为0.8；A2/A3各200例均达到该标准。

## 需求拆解

1. 校验枚举、尺寸、步长、指针跨度及 A/x 不重叠，维持异步接口语义。
2. 覆盖三角方向、转置/共轭、单位对角、负步长及特殊浮点输入。
3. 利用 UB 缓存与向量归约降低带矩阵访问成本，同时保留数值风险回退。
4. 默认与显式 FMA 兼容配置独立验证；完整记录 200 个性能案例与前后冻结输入。

# 详细设计（required）

## 算子分析

### 数学公式

$$\operatorname{op}(A)x=b,\qquad \operatorname{op}(A)\in\{A,A^T,A^H\}.$$

上三角带存储中，逻辑元素位于 $A_{\mathrm{band}}[k+i-j+j\,lda]$；下三角位于 $A_{\mathrm{band}}[i-j+j\,lda]$。仅使用对应三角带内元素。输入 x 保存 b，返回后保存解。

### 支持数据类型

`aclblasComplex`：实部、虚部均 float32（complex64）；不做低精度转换。

### 支持形状

运行时 n、k、lda、incx 由 host 生成 tiling；不依赖固定编译形状。受有效分配范围、整数地址计算和平台内存限制；不承诺任意巨大尺寸均可分配。负步长按 BLAS 约定从逻辑首元素开始访问。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

##### 1. 分核策略：

三角求解存在逐行/逐列依赖，使用一个 AIV 核；在核内并行搬运与乘积归约，不强行跨核拆解有依赖的求解序列。

##### 2. 数据分块和内存优化策略：

查询运行时 UB 容量；对齐后的 x 不超过 UB 一半时缓存 x。按剩余预算与指令长度限制选择带宽 tile，预留对齐余量。NONUNIT 在 n≤4095 且预算允许时缓存对角，否则回退普通读取。预算同时计入 x、五组对齐 tile 缓冲和每元素 32 bytes 的对角缓冲。

实测 UB 容量为 196352 bytes；k=1024 时对角预算边界 n=3852/3853 已覆盖，并另外覆盖 k=8 的 n=4094/4095/4096/4097。该容量不是内存峰值。kernel 不申请临时 GM；公共 handle workspace 不计作零占用证明。

##### 3. tilingkey规划策略：

使用直调 tiling 结构传递尺寸、枚举、缓存长度与地址；通过 kernel 内部条件选择缓存/标量、N/T/C、UNIT/NONUNIT 分支，无额外图模式 tilingkey 注册。

数据检测：先校验 handle、枚举与整数参数；n=0 直接成功。n>0 校验 A/x 非空，使用 64 位跨度计算拒绝地址回绕和重叠；拒绝 incx=INT_MIN。无效输入返回任务指定状态，不启动 kernel。k=0 且 UNIT 在完成必要校验后直接成功。

#### 3.2.2 kernel侧设计：

1. 按 uplo/trans 确定顺序求解方向；C 路径对系数共轭，正确处理负 incx。
2. 通过 DataCopyPad/Gather 分块访问系数与缓存 x。UNIT 使用 PairReduceSum 形成复乘积后按参考顺序扣减；NONUNIT 在安全范围内使用 ReduceSum。
3. 在风险归约前检查输入量级与解的增长边界。界限保守时回退标量顺序计算；动态风险回退从原 RHS 重算，避免已更新数据被重复使用。NaN 通过整数位模式参与风险判定，避免浮点最大值归约吞掉 NaN。
4. 带外 padding 不得产生影响输出的零乘 NaN/Inf；保守风险扫描可能因 padding 极值进入慢路径，这是效率取舍。N 路径保持零 RHS 的参考跳过语义。
5. 复除保留常规直接计算；仅对实际中间溢出、次正规比例/乘积及比值溢出采用保护与重排。不声称全 float32 范围正确舍入。
6. `CTBSV_SCALAR_FMA=0` 默认对齐源码 Netlib 的 no-contract 参考；`=1` 显式 FMA 路径对齐对应 aarch64 Netlib。二者是参考兼容配置，不是全局逐位等价保证。FMA 实测使用包含同一生产源的独立 CMake 入口。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 实测环境 |
| --- | --- | --- |
| Atlas A2 系列 | √ | Ascend910B3，logical0/physical2，私有 CANN9.1 |
| Atlas A3 系列 | √ | Ascend910_9382，CANN9.1 |

上述为芯片实测范围，不代表逐一测试所有服务器整机型号。

## 算子约束限制

A/x 不允许重叠；调用后须同步 handle stream 才能在 host 读取结果。任务未要求确定性模式或广播。奇异矩阵不增加未规定的 host 内容扫描。精度结论限于所述测试及参考配置；FMA 配置没有本轮正式性能结论。A2 profiler 为明确记录的官方组件组合变体，并非原装 CANN9.1 profiler。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度 | rtol=atol=2⁻¹³，合格比例≥99%，最大绝对误差按任务规定的 0.01/32 ULP 规则；特殊值分类检查 | 任务书 §3.2 及测试实现 |
| 性能 | GPU/NPU≥0.8；每例 5 warmup 后 11 次有效样本，取均值；阈值为 GPU 时延/0.8 | 任务书官方 0.8 下限 |
| 内存 | 任务“不涉及”；A2/A3 均未测峰值，不虚报为零 | 任务书 §3.4 |

当前 kernel SHA256：`045a7170491272647832e8df4a5e016e441b3973a40560070b0444508ecc2d73`。

| 平台 | 默认精度 | FMA 精度 | 默认正式性能 | 最小 GPU/NPU |
| --- | --- | --- | --- | --- |
| A3 | 2299/2299 | 2299/2299 | 200/200，3200 原始 CSV | 0.932274603 |
| A2 | 2299/2299 | 2299/2299 | 200/200，3200 原始 CSV | 0.933645863 |

2299 项由原 CSV 1200 项、新增参数化 1092 项和 7 项 fixture 构成；fixture 内子场景不重复计数。覆盖 A 不变、x 间隙不变、UNIT 毒化对角、参数错误、缓存/分块边界、已知解/残差、NaN/Inf、极值复除与归约溢出。XML 中部分误差 metadata 仅保留六位小数，0.000000 不代表数学零；判定使用未舍入 double。

A3 正式性能四卡分片；A2 单卡四批顺序采集，每批 50 例、800 CSV，均保留全部样本。A2 使用 CANN9.1 编译/runtime、9.1 profiler frontend 加官方未修改 msopcom `c3d890d765275811ba39188a02688300e01c2162` 注入组件；原 SDK 不修改。

A2/A3 精度与性能结论来自实机记录；数值证据子集与复算入口见本包 `task_submission/evidence/public-safe-evidence/README.md`。完整来源冻结不在公开包中；公开子集只能复算所含数值与文件完整性，不能重新证明全部环境来源和逐次运行归属。设备快照不构成持续独占证明。

## 兼容性分析

新增公共 API，不改变其他算子接口。A2 Ubuntu20.04/glibc2.31 的 harness 增加 `-ldl -pthread`，生产算子源与 A3 相同；SDK 重定位和 profiler 组件差异独立记录。

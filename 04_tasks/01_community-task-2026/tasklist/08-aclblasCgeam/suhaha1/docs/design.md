# aclblasCgeam 算子设计文档

本文定义 <code>aclblasCgeam</code> 在 Ascend 950PR 上的接口语义、Host
校验、Ascend C Kernel、数据搬运、性能优化和验收方案。设计以任务书、
ops-blas 公共接口和 cuBLAS <code>cublasCgeam</code> 为准，目标是让实现
可以直接合入 ops-blas，而不引入 950PR 私有接口。

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 8 月社区任务：aclblasCgeam 算子开发（950） |
| 算子接口 | <code>aclblasCgeam</code> |
| 目标硬件 | Ascend 950PR，<code>arch35</code> |
| 目标软件 | CANN 9.1.0 |
| 数据类型 | <code>complex64</code>，实部和虚部均为 <code>float32</code> |
| 参考接口 | cuBLAS <code>cublasCgeam</code> |
| 算子目录 | <code>blas/geam/arch35/</code> |
| 测试目录 | <code>test/geam/cgeam/arch35/</code> |
| 文档版本 | V1.0 |
| 提交作者 | 待提交人补充 |
| 设计日期 | 2026 年 9 月 8 日 |

## 一、需求描述

本节说明算子的来源、目标、边界和用户可观察的行为。所有公开行为均以
任务书第 2 节为准。

### 1.1 需求来源

<code>aclblasCgeam</code> 是面向 Ascend 950PR 的 BLAS 扩展算子。算子
对两个单精度复数矩阵执行缩放、转置或共轭转置后相加，提供与 cuBLAS
<code>cublasCgeam</code> 对齐的句柄式接口。Netlib BLAS 和 CBLAS 没有
对应的 <code>cgeam</code> 实现，因此精度参考使用 ops-blas 测试目录中的
CPU Golden。

任务要求采用 Ascend C Kernel 直调模式实现，Host 接口通过 handle 取得
调用 stream，并在该 stream 上异步发起 NPU Kernel。正式实现完成后，算子
代码和测试代码分别提交至 ops-blas 的 geam 目录。

### 1.2 功能目标

算子的核心目标是对所有合法输入计算下式：

~~~text
C = alpha * op(A) + beta * op(B)
~~~

其中 <code>op(X)</code> 根据转置属性定义如下：

| 属性 | 语义 | 复数处理 |
| --- | --- | --- |
| <code>ACLBLAS_OP_N</code> | <code>op(X) = X</code> | 不改变 |
| <code>ACLBLAS_OP_T</code> | <code>op(X) = X^T</code> | 不取共轭 |
| <code>ACLBLAS_OP_C</code> | <code>op(X) = X^H</code> | 实部不变，虚部取反 |

设计必须满足以下功能约束：

- <code>op(A)</code>、<code>op(B)</code> 和 <code>C</code> 的逻辑形状均为
  <code>m × n</code>。
- A、B、C 均采用列主序，元素 <code>X[i, j]</code> 的地址为
  <code>X[i + j * ldX]</code>。
- <code>alpha</code>、<code>beta</code>、A、B、C 使用
  <code>complex64</code>；实部和虚部按 <code>float32</code> 运算。
- <code>m = 0</code> 或 <code>n = 0</code> 是合法 no-op，返回
  <code>ACLBLAS_STATUS_SUCCESS</code>，不发起计算 Kernel。
- 当 <code>alpha = (0, 0)</code> 时不读取 A，A 可以为
  <code>nullptr</code>；当 <code>beta = (0, 0)</code> 时不读取 B，B 可以为
  <code>nullptr</code>。
- 当 <code>alpha = beta = (0, 0)</code> 时，仅将 C 的逻辑区域清零。
- 支持 <code>C == A</code> 和 <code>C == B</code> 的原地操作，但分别要求
  对应输入未转置，且输入和输出前导维相等。
- 不提供广播、超出 leading dimension 语义的非连续访问或动态批处理。

### 1.3 非目标范围

本设计只覆盖任务书声明的单矩阵、单精度复数和 Ascend 950PR
<code>arch35</code> 场景。以下能力不在本次交付范围内：

- <code>complex128</code>、实数 dtype、批量矩阵或广播；
- 任意重叠但不满足精确指针相等的部分原地写入；
- 依赖额外 workspace 的全矩阵临时副本；
- 通过 CPU 或 GPU fallback 代替 NPU Kernel；
- 950PR 之外产品的专属私有 API。

### 1.4 公开接口

公开接口直接复用 ops-blas <code>include/cann_ops_blas.h</code> 中已有
声明，设计不增加新的 950PR 平行接口。

~~~cpp
aclblasStatus_t aclblasCgeam(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* beta, const aclblasComplex* B, int ldb,
    aclblasComplex* C, int ldc);
~~~

### 1.5 参数契约

下表定义 Host 校验和 Kernel 使用的参数契约。标量和属性在 Host
内存中，矩阵数据在 Device 内存中。

| 参数 | 方向 | 类型和位置 | 语义与布局 | 合法性及异常 |
| --- | --- | --- | --- | --- |
| <code>handle</code> | 输入 | <code>aclblasHandle_t</code>，Host | 库上下文，携带执行 stream | 空句柄返回 <code>ACLBLAS_STATUS_HANDLE_IS_NULLPTR</code> |
| <code>transa</code> | 输入 | <code>aclblasOperation_t</code>，Host | A 的 N、T 或 C 操作 | 非法枚举返回 <code>ACLBLAS_STATUS_INVALID_ENUM</code> |
| <code>transb</code> | 输入 | <code>aclblasOperation_t</code>，Host | B 的 N、T 或 C 操作 | 非法枚举返回 <code>ACLBLAS_STATUS_INVALID_ENUM</code> |
| <code>m</code> | 输入 | <code>int</code>，Host | C 和 <code>op(A)</code> 的行数 | <code>m &lt; 0</code> 返回 <code>ACLBLAS_STATUS_INVALID_VALUE</code> |
| <code>n</code> | 输入 | <code>int</code>，Host | C 和 <code>op(B)</code> 的列数 | <code>n &lt; 0</code> 返回 <code>ACLBLAS_STATUS_INVALID_VALUE</code> |
| <code>alpha</code> | 输入 | <code>const aclblasComplex*</code>，Host | A 的复数缩放因子 | 空指针返回 <code>ACLBLAS_STATUS_INVALID_VALUE</code> |
| A | 输入 | <code>const aclblasComplex*</code>，Device | 列主序输入矩阵；N 时为 <code>lda × n</code>，T/C 时为 <code>lda × m</code> | <code>alpha</code> 非零时必须非空 |
| <code>lda</code> | 输入 | <code>int</code>，Host | A 的前导维 | N 时 <code>lda &gt;= max(1,m)</code>；T/C 时 <code>lda &gt;= max(1,n)</code> |
| <code>beta</code> | 输入 | <code>const aclblasComplex*</code>，Host | B 的复数缩放因子 | 空指针返回 <code>ACLBLAS_STATUS_INVALID_VALUE</code> |
| B | 输入 | <code>const aclblasComplex*</code>，Device | 列主序输入矩阵；N 时为 <code>ldb × n</code>，T/C 时为 <code>ldb × m</code> | <code>beta</code> 非零时必须非空 |
| <code>ldb</code> | 输入 | <code>int</code>，Host | B 的前导维 | N 时 <code>ldb &gt;= max(1,m)</code>；T/C 时 <code>ldb &gt;= max(1,n)</code> |
| C | 输出 | <code>aclblasComplex*</code>，Device | <code>m × n</code> 列主序输出矩阵 | <code>m &gt; 0</code> 且 <code>n &gt; 0</code> 时必须非空 |
| <code>ldc</code> | 输入 | <code>int</code>，Host | C 的前导维 | <code>ldc &gt;= max(1,m)</code>，否则返回参数错误 |

矩阵元素访问只使用有效逻辑区域。C 的 padding 区域不是输出的一部分，
Kernel 不读取也不修改该区域。

### 1.6 状态码和校验顺序

Host 端采用固定的控制参数校验顺序，使异常行为稳定并避免在参数无效时
访问 Device 地址。校验流程如下：

1. 检查 <code>handle</code>，空句柄立即返回
   <code>ACLBLAS_STATUS_HANDLE_IS_NULLPTR</code>。
2. 检查 <code>transa</code> 和 <code>transb</code>，只接受 N、T、C 三个
   枚举值。
3. 检查 <code>m</code> 和 <code>n</code>，负维度返回
   <code>ACLBLAS_STATUS_INVALID_VALUE</code>。
4. 检查 <code>alpha</code> 和 <code>beta</code>，标量指针为空返回
   <code>ACLBLAS_STATUS_INVALID_VALUE</code>。
5. 当 <code>m == 0</code> 或 <code>n == 0</code> 时返回成功，不解引用
   A、B、C，不读取 Device 内存，也不发起 Kernel。
6. 根据转置属性校验 <code>lda</code>、<code>ldb</code> 和 <code>ldc</code>
   的最小前导维。
7. 读取 Host 标量并判断短路条件。只有实际需要的输入矩阵才检查对应
   Device 指针。
8. 检查 C 指针以及精确指针相等情况下的原地约束，最后生成 tiling
   参数并在 handle 绑定的 stream 上发起 Kernel。

<code>alpha</code> 和 <code>beta</code> 的合法性检查始终执行。短路只跳过
不参与计算的 A 或 B 数据访问，不改变前导维和输出 C 的接口契约。

### 1.7 原地和短路语义

原地语义的关键是保证一个输出 tile 在被覆盖前已经完成了所需输入的读取。
接口只承诺精确的基地址相等，不承诺任意部分重叠。

| 场景 | 合法条件 | 实现行为 |
| --- | --- | --- |
| <code>C == A</code> | <code>transa == N</code> 且 <code>lda == ldc</code> | 先将 A tile 搬入 UB，再写回 C |
| <code>C == B</code> | <code>transb == N</code> 且 <code>ldb == ldc</code> | 先将 B tile 搬入 UB，再写回 C |
| <code>C == A == B</code> | 同时满足上面两组条件 | 在写回前完成 A、B 两个逻辑输入的读取 |
| <code>alpha == 0</code> | A 可为 <code>nullptr</code> | 不读取 A，不执行 A 的复数乘法 |
| <code>beta == 0</code> | B 可为 <code>nullptr</code> | 不读取 B，不执行 B 的复数乘法 |
| <code>alpha == beta == 0</code> | C 必须有效 | 仅清零 C 的 <code>m × n</code> 逻辑区域 |

当 <code>C == A</code> 但 A 使用 T/C，或 <code>C == B</code> 但 B 使用
T/C，或者相应 leading dimension 不相等时，Host 返回
<code>ACLBLAS_STATUS_INVALID_VALUE</code>，不发起 Kernel。

## 二、现状与方案调研

本节对公共接口、参考实现和可选工程方案进行比较，明确本设计的取舍
依据。

### 2.1 ops-blas 现状

任务书要求的公共头文件已经包含 <code>aclblasCgeam</code> 声明。实现
需要放入 <code>blas/geam/arch35/</code>，测试需要放入
<code>test/geam/cgeam/arch35/</code>，并沿用 ops-blas 的
<code>build.sh</code>、Handle、stream 和 GTest 组织方式。

本设计不修改函数签名，不创建 <code>aclblasCgeam950</code> 或其他产品
私有入口。Host 层只承担参数解析、合法性检查、tiling 生成和 Kernel
发起；计算和 Device 数据访问全部位于 Ascend C Kernel。

### 2.2 cuBLAS 参考语义

cuBLAS <code>cublasCgeam</code> 给出了本算子的外部调用模型：句柄位于
首参数，随后依次传入两个转置属性、输出形状、两个复数缩放因子、A/B/C
及其 leading dimension。实现采用列主序，并支持 N、T、C 三类操作。

本设计沿用该参数顺序和矩阵语义，因此应用迁移不需要额外参数映射。
对于 Netlib BLAS 没有覆盖的 <code>geam</code> 能力，使用仓内 CPU Golden
对实部和虚部分别校验。

### 2.3 可选实现方案对比

候选方案的比较结果如下，最终选择“Host 校验加 Ascend C 直调 Kernel”
并在 Kernel 内共享通用计算骨架、区分数据搬运路径。

| 方案 | 优点 | 风险或代价 | 结论 |
| --- | --- | --- | --- |
| CPU 计算后搬回 Device | 实现简单，容易生成参考结果 | 违反 NPU 直算要求，搬运开销大，无法通过验收 | 不采用 |
| 通用图算子或额外私有 API | 可复用部分运行时能力 | 接口不与 ops-blas 对齐，增加公共维护面 | 不采用 |
| 一个 Kernel 使用全局内存逐元素访问 | 代码短，适配所有转置 | 访存不规则，难以达到 950PR 性能目标 | 仅作为正确性基线 |
| tile 搬运加 UB 向量计算 | 访问合并，适合复数逐元素运算 | 需要处理转置、尾块和 UB 容量 | 采用 |
| 为 N、T/C 分别实现完全独立 Kernel | 便于局部优化 | 代码重复，修复语义问题容易分叉 | 采用共享骨架加模式分支 |
| 全矩阵 Device 临时副本 | 转置实现简单 | 额外显存、额外搬运，原地场景更复杂 | 不采用 |

### 2.4 设计原则

方案遵循以下优先级：

1. 先保证 cuBLAS 语义和任务书异常行为一致，再做性能特化。
2. 以有效逻辑区域为边界，任何 tile 都不访问 padding 之外的地址。
3. 让短路和特殊标量在 Host 端形成明确 Kernel 模式，减少无效访存。
4. 不申请与 <code>m × n</code> 成比例的额外 workspace，避免内存峰值随
   输出放大。
5. 将可调参数集中到 tiling 数据，便于在 950PR 上按性能指标调优 tile。

## 三、总体设计与实现方案

本节给出 Host、tiling、Kernel 和测试工程之间的模块边界，以及一次调用
从 API 到 Device 完成的完整路径。

### 3.1 总体架构

一次正常调用经过以下模块。Host 负责控制面，Kernel 负责数据面。

~~~mermaid
flowchart LR
    U[调用者] --> H[aclblasCgeam]
    H --> V[Handle/枚举/维度/指针校验]
    V --> F{短路或 no-op}
    F -->|m=0 或 n=0| R1[返回 SUCCESS]
    F -->|双零| Z[Zero Kernel]
    F -->|单输入或双输入| T[Tiling 生成]
    T --> L[Handle 绑定 Stream]
    L --> K[Ascend C cgeam Kernel]
    K --> G[GM A/B 读取]
    G --> U1[UB 搬运与转置]
    U1 --> C[复数缩放与加法]
    C --> W[GM C 写回]
    Z --> R2[返回异步提交状态]
    W --> R2
~~~

Kernel 提交后接口按照 ops-blas 的异步模型返回。调用者在读取 C 之前
必须按照现有库约定同步同一 stream；设计不在 Host 端额外插入全局同步。

### 3.2 Host 层模块设计

Host 层放在 <code>blas/geam/arch35/</code> 的公开接口实现和内部辅助
模块中，建议拆分为参数校验、tiling 生成和 Kernel 启动三个职责。

| 模块 | 主要职责 | 设计要点 |
| --- | --- | --- |
| API 入口 | 接收公共接口参数 | 保持 <code>cann_ops_blas.h</code> 的声明和参数顺序 |
| 参数校验 | 检查枚举、维度、leading dimension、指针和 alias | 不读取无效 Device 指针；状态码稳定 |
| 标量解析 | 读取 Host 上的 <code>alpha</code>、<code>beta</code> | 复制实部和虚部到 tiling，避免 Kernel 反复读取 Host |
| 模式选择 | 选择 no-op、zero、single-input 或 general 模式 | 结合 N/T/C 和 <code>1+0i</code> 等特殊标量生成 flags |
| tiling 生成 | 计算 tile 数、block 数和尾块信息 | 不申请全矩阵 workspace |
| Kernel 启动 | 绑定 handle stream 并发起 arch35 Kernel | 不调用同步 API，不做 CPU fallback |

建议的内部 tiling 结构如下。该结构是 Host 与 Kernel 的内部契约，不对
用户公开。

~~~cpp
struct CgeamTilingData {
    int32_t m;
    int32_t n;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t transa;
    int32_t transb;
    int32_t tile_m;
    int32_t tile_n;
    int32_t tiles_m;
    int32_t tiles_n;
    int32_t block_count;
    uint32_t mode;
    float alpha_real;
    float alpha_imag;
    float beta_real;
    float beta_imag;
};
~~~

<code>mode</code> 至少包含以下位：

- <code>kSkipA</code>：<code>alpha</code> 为精确零，不读取 A；
- <code>kSkipB</code>：<code>beta</code> 为精确零，不读取 B；
- <code>kZeroOutput</code>：两个标量均为精确零；
- <code>kAlphaIdentity</code>：<code>alpha</code> 为
  <code>1+0i</code>；
- <code>kBetaIdentity</code>：<code>beta</code> 为
  <code>1+0i</code>；
- <code>kConjA</code> 和 <code>kConjB</code>：对应输入需要执行共轭；
- <code>kTransposeA</code> 和 <code>kTransposeB</code>：对应输入需要执行
  转置搬运。

这些 flags 只影响内部路径，不改变公开语义。非零的极小复数不按零
处理，必须走普通复数乘法。

### 3.3 Kernel 任务划分

Kernel 将输出逻辑矩阵按二维 tile 划分。每个输出 tile 负责一个连续的
逻辑区域，block 通过线性 tile 编号领取任务，避免为每个 shape 生成单独
的 Kernel。

设 tile 起点为 <code>(i0, j0)</code>，tile 有效大小为
<code>tm × tn</code>。输出地址为：

~~~text
C[i0 + i + (j0 + j) * ldc]
~~~

输入地址按照转置属性计算：

~~~text
N: X[i0 + i + (j0 + j) * ldX]
T: X[j0 + j + (i0 + i) * ldX]
C: conjugate(X[j0 + j + (i0 + i) * ldX])
~~~

当 A 或 B 是转置/共轭转置时，源 tile 的物理形状为
<code>tn × tm</code>。Kernel 先以列主序将源 tile 搬入 UB，再在 UB 内
完成转置；C 模式在转置过程中对虚部取负。这样可以将不规则的跨列访问
限制在 tile 搬运阶段，后续复数乘加使用连续的 UB 数据。

### 3.4 Kernel 数据流

普通双输入路径采用 MTE2、Vector 和 MTE3 的流水：

~~~mermaid
flowchart LR
    A[GM A tile] --> A1[MTE2]
    B[GM B tile] --> B1[MTE2]
    A1 --> UB1[UB ping/pong]
    B1 --> UB2[UB ping/pong]
    UB1 --> X[转置/共轭]
    UB2 --> Y[转置/共轭]
    X --> V[复数向量乘法]
    Y --> V
    V --> S[复数加法]
    S --> O[输出 UB]
    O --> MTE3[MTE3]
    MTE3 --> C[GM C tile]
~~~

每个复数元素使用两个 <code>float32</code> 分量。对输入
<code>x = xr + i*xi</code> 和复数标量 <code>a = ar + i*ai</code>，乘法
按下式执行：

~~~text
real(a*x) = ar*xr - ai*xi
imag(a*x) = ar*xi + ai*xr
~~~

Kernel 保持 <code>float32</code> 分量计算，不把中间结果提升为
<code>float64</code>。T 模式只转置，C 模式在输入规范化阶段将虚部取负，
之后统一复用同一套复数乘加路径。

### 3.5 tile 和 block 配置

tile 取值需要结合目标硬件的 UB 容量、向量宽度和任务书性能指标确定。
以下配置作为实现和调优起点：

| 路径 | 初始 <code>tile_m × tile_n</code> | 选择依据 |
| --- | --- | --- |
| N/N，连续输入 | <code>128 × 64</code> | A、B、C 的有效片段连续，优先提高向量吞吐 |
| 含 T/C 的路径 | <code>64 × 64</code> | 降低转置临时 tile 的 UB 占用 |
| 单输入路径 | <code>128 × 64</code> | 少搬运一个输入，可扩大有效计算 tile |
| 小 shape | 一个 tile 覆盖有效区域 | 减少 block 和流水启动开销 |

每个 tile 的实际 UB 占用按输入、输出、转置临时区和双缓冲总量计算，
必须小于目标 SOC 的可用 UB 容量。若初始 tile 不能满足容量，优先降低
<code>tile_m</code>，其次降低 <code>tile_n</code>，不改变接口语义。

block 数计算为：

~~~text
tiles_m = ceil(m / tile_m)
tiles_n = ceil(n / tile_n)
block_count = min(tiles_m * tiles_n, available_core_count)
~~~

当 tile 数少于可用核数时只启动需要的 block，避免空 block 的调度开销。
当 tile 数较多时，block 以线性 tile 编号循环领取任务，使宽矩阵和窄矩阵
均能覆盖足够的并行度。

### 3.6 尾块和边界保护

M、N 方向尾块使用 <code>valid_m</code> 和 <code>valid_n</code> 计算有效
元素数。MTE2、Vector 和 MTE3 的每一步均以有效区域为边界，尾块之外的
UB 内容不参与运算，也不写入 C。

输入的 padding 只由 <code>ldX</code> 跳过，不作为逻辑数据。输出
<code>ldc</code> 大于 <code>max(1,m)</code> 时，Kernel 仅写每列的前
<code>m</code> 个元素，padding 保持不变。

### 3.7 特殊路径

Host 根据标量和 alias 条件选择以下内部路径：

| 路径 | 触发条件 | Kernel 行为 |
| --- | --- | --- |
| No-op | <code>m == 0</code> 或 <code>n == 0</code> | Host 直接返回，不发起 Kernel |
| Zero | <code>alpha == 0</code> 且 <code>beta == 0</code> | 仅生成 C 的零值 tile |
| B-only | <code>alpha == 0</code> | 只读取 B，执行 <code>beta * op(B)</code> |
| A-only | <code>beta == 0</code> | 只读取 A，执行 <code>alpha * op(A)</code> |
| Identity | 标量为 <code>1+0i</code> | 跳过对应复数乘法 |
| General | 其他情况 | 读取需要的输入并执行完整复数乘加 |

Zero 路径仍需检查 C 和 <code>ldc</code>，并且只清零逻辑矩阵，不清零
padding。短路路径不读取被跳过的 A 或 B，因此可以支持任务书规定的空
指针用例。

### 3.8 原地写入实现

原地场景不使用额外全矩阵副本。每个 tile 的执行顺序如下：

1. 根据 <code>transa</code> 和 <code>transb</code> 从 GM 将所有需要的
   输入片段搬入 UB。
2. 在 UB 内完成转置、共轭和复数缩放。
3. 在所有输入读取完成后写回 C。

由于 Host 拒绝转置原地和不相等 leading dimension，本 tile 写回
不会破坏后续 tile 的未读输入。对于 A、B、C 三者同址的场景，Kernel
先分别完成两个输入的读取，再执行任何写回。

### 3.9 异步执行和生命周期

Kernel 使用 <code>aclblasSetStream</code> 绑定到 handle 的 stream。接口
返回时只表示命令已经提交或提交失败，不表示 C 已经可以在 Host 侧读取。
调用者必须在读取 C 前同步该 stream。

Host 侧 tiling 数据和 Kernel 参数在提交前完成构造，并由运行时按现有
ops-blas 生命周期管理。Kernel 不保存 <code>alpha</code>、<code>beta</code>、
A、B 或 C 的 Host 指针；只使用已经编码到参数区的标量和 Device 地址。

## 四、性能优化方案

本节将性能目标拆成访存、计算、并行和调度四个方面。优化不改变
<code>aclblasCgeam</code> 的数值和异常语义。

### 4.1 性能目标

性能测试在 Ascend 950PR 上进行。每个 case 先 warmup，再使用超过 50
次有效采样计算平均单次耗时，单位为微秒。

| Case | <code>m</code> | <code>n</code> | <code>transa</code> | <code>transb</code> | 平均耗时上限 |
| --- | ---: | ---: | --- | --- | ---: |
| 1 | 1024 | 1024 | N | N | 12.41 us |
| 2 | 2048 | 2048 | N | N | 63.62 us |
| 3 | 2048 | 2048 | T | C | 65.33 us |

上述 case 使用 <code>alpha = 1+0i</code>、<code>beta = 1+0i</code> 和
紧凑 leading dimension。任务书中的直接微秒上限是正式门禁；
<code>gpu_baseline.csv</code> 回填结果只用于参考比率和性能诊断，不替代
任务书上限。

### 4.2 访存优化

访存优化按以下顺序实施：

- N/N 路径按列连续搬运 A、B 和 C 的有效片段，减少跨列访存。
- T/C 路径以源矩形 tile 为单位搬运，在 UB 内转置，避免对每个元素
  发起独立全局内存访问。
- 使用 ping-pong UB，让下一 tile 的 MTE2 搬运与本 tile 的 Vector
  计算重叠。
- MTE3 只写有效输出元素，跳过 padding 和无效尾块。
- alpha 或 beta 为零时跳过对应输入的搬运，避免无效带宽。

设计不使用全矩阵转置缓冲区，因此额外内存访问量与 tile 数相关，而不
会增加一个 <code>m × n</code> 的临时副本。

### 4.3 计算优化

复数乘法拆分为四次实数乘法和两次加减。对于 <code>1+0i</code>、
<code>0+0i</code> 和单输入路径，Host 通过 flags 让 Kernel 删除不必要
的计算：

- <code>1+0i</code> 只做分量拷贝或加法；
- <code>0+0i</code> 不读取对应输入；
- 双零路径只做向量清零；
- T/C 转换与复数分量处理在同一个 UB 流程内完成，减少中间写回。

Vector 计算采用连续复数分量布局。尾部不足一个向量宽度时使用 mask，
确保不读取或写入越界元素。

### 4.4 并行和调度优化

输出 tile 是独立任务，优先按输出 tile 数映射到多个 AI Core。调度策略
保持简单的线性分块，以降低任务分配开销并覆盖宽、窄、方形矩阵：

- tile 数小于核数时只启用有效 block；
- tile 数较大时每个 block 处理多个 tile；
- 线性编号按 N 方向连续推进，使同一列附近的访问具有较好的局部性；
- 1×1 到几十行的小 shape 关闭不必要的多阶段流水，降低启动延迟。

### 4.5 性能调优顺序

实现阶段按以下顺序调优，并在每次调整后执行精度回归：

1. 先确认 N/N 的 <code>128 × 64</code> 初始 tile 能通过功能和 3 个
   性能 case。
2. 对 <code>2048 × 2048</code> 的 T/C case 调整转置 tile，观察 MTE2、
   Vector 和 MTE3 的占比。
3. 比较单缓冲与双缓冲，只有流水重叠能够抵消同步成本时才保留双缓冲。
4. 在 <code>1024 × 1024</code>、<code>2048 × 2048</code> 和 4096 级别
   矩阵上分别调整 block 数和尾块策略。
5. 使用 <code>msprof</code> 或等效工具定位瓶颈后再做局部改动，避免仅依据
   Host 端 GTest 总耗时判断 Kernel 性能。

## 五、接口影响和工程改动

本节列出设计对应的工程文件、公开接口和对其他框架环节的影响。

### 5.1 代码目录

正式代码按 ops-blas 目录规范组织：

| 位置 | 内容 |
| --- | --- |
| <code>include/cann_ops_blas.h</code> | 复用已有 <code>aclblasCgeam</code> 声明，不新增私有签名 |
| <code>blas/geam/arch35/</code> | Host API、参数校验、tiling、Kernel、构建配置 |
| <code>test/geam/cgeam/arch35/</code> | C++ GTest、CSV 解析、Device 调用封装 |
| <code>test/geam/geam_golden.h</code> | 复用仓内 complex64 CPU Golden |
| <code>test/geam/cgeam/arch35/cgeam_test.csv</code> | 任务用例安装后的正式回归 CSV |

任务包中的设计文档和测试辅助脚本位于：

- [任务书](../aclblasCgeam_Atlas950PR_task_doc.md)
- [测试指导](../test_cases/README.md)
- [用例生成脚本](../test_cases/gen_csv.py)
- [精度验收脚本](../test_cases/verify_accuracy.py)
- [性能验收脚本](../test_cases/verify_performance.py)

### 5.2 公开接口变化

本次不新增公开接口签名。现有 <code>aclblasCgeam</code> 声明作为唯一
调用入口，实现只补齐 arch35 的 Host、Kernel、构建和测试内容。

### 5.3 对框架和应用的影响

该算子是 ops-blas 句柄式 C 接口实现，不修改图编译器、网络定义、
分布式执行、模型保存或推理部署格式。调用者需要注意两点：

- A、B、C 是列主序，不能直接按行主序数组解释；
- 调用返回后 C 可能仍在 stream 上执行，Host 读取前必须同步。

## 六、测试与验收设计

本节给出精度、异常、性能、内存和可复现性测试方案，并定义统一判定标准。

### 6.1 测试环境

测试在以下目标环境中执行：

- Ascend 950PR，<code>arch35</code>；
- CANN 9.1.0；
- ops-blas 对应提交版本；
- 单用例 Host 侧输入输出预算不超过 512 MB；
- CPU Golden 使用 ops-blas 仓内 <code>std::complex&lt;float&gt;</code>
  参考实现。

### 6.2 精度判定

输出 C 的 <code>m × n</code> 全逻辑矩阵逐元素比较，实部和虚部分别按
<code>float32</code> 标准判定：

| 数据类型 | <code>rtol</code> | <code>atol</code> | 匹配率 | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| <code>complex64</code> 分量 | <code>2^-10</code>，约 <code>9.77e-4</code> | <code>2^-16</code>，约 <code>1.53e-5</code> | 至少 0.99 | <code>1e-2</code> 或 <code>32 × ULP</code> |

单个分量满足下式时视为通过：

~~~text
abs(actual - golden) <= atol + rtol * abs(golden)
~~~

一个用例同时满足匹配率和最大误差约束时，判定该用例通过。Padding
不参与逻辑输出比较，但测试会用 guard 或预填充值检查 Kernel 没有越界
写入。

### 6.3 用例覆盖

默认 CSV 由 <code>gen_csv.py</code> 生成 1,000 条精度用例和 200 条
性能/内存用例。精度用例分类如下：

| 类别 | 前缀 | 数量 | 覆盖内容 |
| --- | --- | ---: | --- |
| 基础组合 | <code>TC_L0</code> | 18 | 9 种 N/T/C 组合和 2、4 小 shape |
| 尺寸扫描 | <code>TC_SQ</code> | 92 | 1 到 2048 的边界、奇数和非对齐尺寸 |
| 标量组合 | <code>TC_AB</code> | 32 | 零、单位、负数、复数、纯虚数和双零 |
| 宽矩阵 | <code>TC_WS</code> | 12 | <code>n</code> 远大于 <code>m</code> 的矩形 |
| 高矩阵 | <code>TC_TH</code> | 12 | <code>m</code> 远大于 <code>n</code> 的矩形 |
| Leading dimension | <code>TC_LD</code> | 6 | 最小约束值和 padding |
| 填充模式 | <code>TC_FL</code> | 6 | 随机、全零、交替、极值、Inf、NaN |
| 覆盖扩展 | <code>TC_CV</code> | 32 | 中等尺寸和代表性转置组合 |
| 边界与负向 | <code>TC_ED</code> | 30 | 零维、空指针、非法枚举、负维度和原地约束 |
| 扩展精度 | <code>TC_EX</code> | 760 | 尺寸、转置、标量和 padding 的确定性采样 |
| 性能/内存 | <code>TC_PF</code> | 200 | 任务书 case、小尺寸、方阵和矩形扫描 |

边界测试必须至少覆盖以下行为：

- <code>m == 0</code>、<code>n == 0</code> 和两者同时为零；
- <code>alpha == 0</code> 时 <code>A == nullptr</code>，
  <code>beta == 0</code> 时 <code>B == nullptr</code>；
- 双零清零和 C 为空指针的非法场景；
- A、B、C 必需指针为空；
- 非法 <code>transa</code>、<code>transb</code>；
- <code>lda</code>、<code>ldb</code>、<code>ldc</code> 小于最小约束；
- 负维度；
- 合法和非法 C==A、C==B 原地组合；
- Inf、NaN、正负零和特殊复数标量。

精度测试输入必须覆盖均匀分布和正态分布两类，实部和虚部独立采样。
测试工程使用 <code>--dist mixed</code> 或等效方式生成混合分布，并在
用例参数中记录分布类型。

### 6.4 精度测试步骤

在已获取 ops-blas 仓库和 Ascend 950PR 环境的前提下，执行以下步骤：

1. 在任务包目录运行 <code>python gen_csv.py</code>，生成
   <code>cgeam_test.csv</code>。
2. 使用 <code>verify_accuracy.py --repo &lt;ops-blas路径&gt; --soc
   ascend950 --csv ./cgeam_test.csv</code> 将 CSV 安装到 arch35 测试
   目录并编译。
3. 等待 GTest 完成全部非 <code>TC_PF</code> 用例，确认汇总中
   <code>FAIL=0</code>。
4. 使用 <code>--filter TC_L0</code>、<code>--filter TC_ED</code> 等参数
   进行定向回归，并按照精度阈值判定每个用例。

示例命令如下：

~~~bash
cd 8月社区任务-aclblasCgeam算子开发（950）/test_cases
python verify_accuracy.py --repo /path/to/ops-blas \
  --soc ascend950 --csv ./cgeam_test.csv --timeout 3600
~~~

### 6.5 性能测试步骤

性能测试只使用 <code>TC_PF</code>，且性能 CSV 的 <code>lda</code>、
<code>ldb</code>、<code>ldc</code> 留空，由测试工程推导紧凑 leading
dimension。执行流程如下：

1. 准备 arch35 测试二进制，并使用 <code>verify_performance.py</code>
   执行性能用例。
2. 对每个 case 先 warmup，再采集至少 51 个有效样本。
3. 统计每个 case 的 warmup 次数、有效样本数和平均耗时。
4. 检查 1024×1024 NN、2048×2048 NN、2048×2048 TC 三个 case 的平均
   耗时分别不超过 12.41、63.62、65.33 微秒。
5. 使用 <code>msprof</code> 或等效工具分析 Kernel 级耗时及 MTE/Vector
   占比，用于定位性能瓶颈。

示例命令如下：

~~~bash
cd 8月社区任务-aclblasCgeam算子开发（950）/test_cases
python verify_performance.py --repo /path/to/ops-blas \
  --soc ascend950 --timeout 3600 --output-dir ./results
~~~

性能脚本会把超出任务书 3 个典型 case 的其他性能用例标为
<code>OUT_OF_SCOPE</code>，这些用例用于发现规模回退和内存问题，不替代
正式门禁。

### 6.6 内存测试

任务书没有单独的算子内存上限，但实现必须避免全矩阵临时副本。内存
测试采集以下指标：

- 输入 A、B 和输出 C 的字节数；
- Kernel 运行期间的峰值 Device 内存；
- UB 之外的额外 workspace，预期为零或固定大小；
- 512 MB Host 侧预算下的最大性能 case；
- padding 和 guard 区的越界写检查。

## 七、风险、取舍和待确认项

本节列出实施中需要通过代码验证或设备 Profiling 关闭的风险。

### 7.1 风险清单

下表列出实现、性能、精度和测试复现中需要重点关注的风险，以及对应的
缓解措施和关闭条件。

| 风险 | 影响 | 缓解措施 | 关闭条件 |
| --- | --- | --- | --- |
| T/C 访存不规则 | 2048×2048 TC 性能不达标 | 源 tile 搬运加 UB 转置，单独调节 T/C tile | 典型 TC case 达到 65.33 us |
| 复数分量布局不匹配 | 精度或向量吞吐下降 | 统一 real/imag 访问和 Golden 对照 | 全量精度用例通过 |
| 原地写回覆盖输入 | C==A/B 结果错误 | Host 限制 N 且 ld 相等，tile 先读后写 | 原地和 alias 回归通过 |
| 短路指针误访问 | 空指针用例崩溃 | Host 生成 skip flags，Kernel 不搬运对应输入 | nullA/nullB 用例通过 |
| 尾块写越界 | 破坏 padding 或引起异常 | 有效尺寸 mask 和 guard 检查 | LD、奇数和矩形用例通过 |
| 小 shape 启动开销 | 延迟用例性能下降 | 小 shape 采用单 tile 简化路径 | TC_PF 小尺寸稳定 |
| 测试分布不完整 | 自测报告覆盖不足 | 正式验收前完成 mixed 分布支持或补充用例 | 报告可追溯到实际输入分布 |
| 测试设备版本差异 | 结果无法复现 | 记录 SOC、驱动、固件、CANN 和提交号 | 验收环境信息完整 |

### 7.2 方案取舍

本设计用 UB tile 换取更规则的 Device 访存和更少的全局访问次数，用
少量 Host 模式分支换取零标量、单位标量和单输入场景的计算削减。代价
是需要维护 N/N 与 T/C 的数据搬运逻辑，并对尾块、共轭和 alias 进行
完整回归。

相较于全矩阵转置副本，该方案的额外内存固定在 tile 级，适合任务书
要求的 2048 和 4096 级矩阵。相较于逐元素 GM 访问，该方案增加了
tiling 和 UB 管理复杂度，但更有机会满足 950PR 的微秒级性能上限。

### 7.3 待确认项

以下内容在正式实现阶段以目标 SOC 的编译器和运行时接口为准：

- 950PR 可用 UB 容量和推荐 Vector 复数数据布局；
- Ascend C arch35 对复数分量转置、mask 和双缓冲的最佳 API 组合；
- 初始 tile 的实际可用核数、MTE2/Vector/MTE3 重叠效果；
- 测试工程对 mixed 正态/均匀分布、Inf/NaN 比较的最终支持情况。

上述待确认项只影响内部实现和测试工具，不改变公开接口契约。

## 八、名词解释

下表统一本文和代码评审中的术语。

| 名词 | 解释 |
| --- | --- |
| Cgeam | Complex GEneral matrix Addition，复数矩阵加/缩放/转置扩展接口 |
| <code>op(X)</code> | 根据转置属性对 X 执行不转置、转置或共轭转置 |
| Leading dimension | 列主序矩阵相邻列之间的物理步长，本文记为 <code>ldX</code> |
| N/T/C | No transpose、Transpose、Conjugate transpose |
| GM | Device Global Memory，Device 全局内存 |
| UB | Unified Buffer，Kernel 使用的片上统一缓冲区 |
| Tile | Kernel 一次搬运和计算的二维逻辑数据块 |
| no-op | 合法但不执行计算的调用，本文对应 <code>m == 0</code> 或 <code>n == 0</code> |

## 九、参考资料

参考资料和工作区中的验证入口如下：

1. [aclblasCgeam 任务书](../aclblasCgeam_Atlas950PR_task_doc.md)。
2. [cgeam 测试用例及测试指导](../test_cases/README.md)。
3. [Ascend C 算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)。
4. [Ascend C 算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)。
5. [ops-blas 开源仓](https://gitcode.com/cann/ops-blas)。
6. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。
7. [cuBLAS cublasCgeam 参考文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-geam)。
8. [社区任务设计文档模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。

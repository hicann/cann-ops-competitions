# aclblasCsscal Ascend 950PR 算子设计文档

# 需求背景（required）

## 需求来源

本需求来源于 CANN 训练营及昇腾算子社区任务，目标是在 Ascend 950PR 上，基于 CANN 9.1.0 和 ops-blas 工程，使用 Ascend C 编程语言实现单精度复数向量乘实数标量的 `aclblasCsscal` 算子。

算子完成设计、开发和自验证后，计划将算子实现、测试工程和配套说明提交至 CANN `ops-blas` 开源仓库。其中算子实现位于 `blas/scal/arch35/`，测试实现位于 `test/scal/csscal/arch35/`。

## 背景介绍

### aclblasCsscal 算子功能

`aclblasCsscal` 是 BLAS Level 1 向量缩放接口，用于将单精度复数向量中的每个逻辑元素乘以一个单精度实数标量，并将结果原地写回输入向量。其功能语义与 cuBLAS `cublasCsscal` 以及 Netlib BLAS `csscal` 对齐。

对于逻辑元素下标 `i = 0, 1, ..., n-1`，计算公式为：

```text
x[i * incx].real = alpha * x[i * incx].real
x[i * incx].imag = alpha * x[i * incx].imag
```

其中，`alpha` 为 Host 侧 `float` 标量，`x` 为 Device 侧 `complex64` 向量，`incx` 以复数元素为单位表示相邻逻辑元素之间的物理步长。

### ops-blas 同类算子实现现状

ops-blas 的 `scal` 算子族已包含 `aclblasSscal` 和 `aclblasCscal` 等接口，可为句柄管理、Stream 获取、Host 侧参数检查、Kernel 直调和测试工程组织方式提供参考。

`aclblasCsscal` 的公共接口已在 `include/cann_ops_blas.h` 中声明。本需求不增加 Ascend 950PR 私有接口，而是在现有公共接口下补充 `arch35` 实现，并复用 ops-blas 的句柄、日志、构建和测试基础设施。

接口定义如下：

```cpp
aclblasStatus_t aclblasCsscal(
    aclblasHandle_t handle,
    int n,
    const float* alpha,
    aclblasComplex* x,
    int incx);
```

### aclblasCsscal 能力分析

| 参数 | 参数含义 | 参数位置 | 数据类型 | 约束 |
| --- | --- | --- | --- | --- |
| handle | ops-blas 上下文句柄，携带执行 Stream | Host | `aclblasHandle_t` | 不得为空 |
| n | 复数向量逻辑元素数量 | Host | `int` | `n <= 0` 时为合法 no-op |
| alpha | 实数缩放系数 | Host | `const float*` | 正常计算时不得为空 |
| x | 输入及输出复数向量 | Device | `aclblasComplex*` | 正常计算时不得为空，原地更新 |
| incx | 相邻逻辑元素的复数步长 | Host | `int` | `incx <= 0` 时为合法 no-op，正值支持任意步长 |

`aclblasComplex` 由两个连续的 `float` 分量构成，分别表示实部和虚部。算子不涉及广播、不改变输入形状，也不产生额外输出张量。

### 参考实现与约束

Netlib `csscal` 的 no-op 条件为 `N <= 0`、`INCX <= 0` 或 `SA == 1`。本实现遵循以下处理顺序：

1. `handle == nullptr` 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n <= 0` 或 `incx <= 0` 时返回 `ACLBLAS_STATUS_SUCCESS`，且不访问 `alpha` 和 `x`。
3. 正常计算时，`alpha == nullptr` 或 `x == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `alpha == 1.0f` 时直接返回成功，以保留输入的全部位模式。
5. `alpha == 0.0f` 时按照任务书要求将所有选中元素的实部和虚部写为正零，而不是作为 no-op 处理。

任务书同时引用 Netlib golden，并要求零标量结果 bit-exact。对于非有限输入，IEEE 浮点乘法中的 `0 * Inf/NaN` 与强制写正零存在差异。本设计以任务书明确的置零要求为当前实现依据，并在测试 golden 中单独实现该分支；正式验收时需要保留该规格裁决记录。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Ascend 950PR 上实现 `aclblasCsscal`，支持 complex64 向量与 float32 实数标量的原地缩放，支持任意正步长 `incx`，并满足任务书规定的接口语义、精度标准和性能指标。

## 需求拆解

1. 复用 `include/cann_ops_blas.h` 中已有的 `aclblasCsscal` 公共接口。
2. 在 `blas/scal/arch35/` 中实现 Host 侧参数检查、Tiling 计算和 Kernel 下发。
3. 使用 Ascend C AIV Kernel 优化连续且对齐的输入。
4. 支持 `incx > 1` 的非连续向量以及未按 32 字节对齐的 Device 视图。
5. 正确处理 `n <= 0`、`incx <= 0`、空指针、`alpha=0`、`alpha=1`、Inf、NaN 和尾块等边界场景。
6. Kernel 在 handle 绑定的 Stream 上异步执行，公共接口内部不进行 Stream 同步。
7. 精度满足 complex64 实部和虚部分别按 FLOAT32 判定的要求。
8. 在三项指定大规模连续访存场景下满足 13.57 us、21.05 us 和 43.02 us 的平均耗时目标。
9. 建立可复现的 CSV 驱动 GTest、设备事件计时和反馈收集流程。

# 详细设计（required）

## 算子分析

### 数学公式

设输入向量为 `x`，实数标量为 `alpha`，逻辑向量长度为 `n`，步长为 `incx`，则：

```text
for i in [0, n):
    j = i * incx
    x[j] = alpha * x[j]
```

将复数拆分为实部和虚部后：

```text
x[j].real_out = alpha * x[j].real_in
x[j].imag_out = alpha * x[j].imag_in
```

算子原地更新 `x`，因此输出与输入共享同一段 Device 内存。对于 `incx > 1`，不属于逻辑向量的间隙元素必须保持不变。

### 支持数据类型

| 对象 | 数据类型 | 说明 |
| --- | --- | --- |
| alpha | FLOAT32 | Host 侧实数标量 |
| x | COMPLEX64 | Device 侧复数向量，每个元素由两个 FLOAT32 分量组成 |
| n、incx | INT32 | Host 侧运行时参数 |

本需求不包含复数标量乘复数向量的 `aclblasCscal`，也不扩展 float16、bfloat16 或其他数据类型。

### 支持形状与数据排布

算子逻辑输入为一维向量 `[n]`，采用 ND 排布。`incx > 0` 时，调用者至少需要提供以下数量的复数元素：

```text
physical_length = 1 + (n - 1) * incx
```

`n` 和 `incx` 均为运行时参数，不要求静态 Shape。算子不涉及广播、转置或 Leading Dimension，但支持通过 `incx` 表达的非连续向量视图。

### 数据量与性能特征

每个 complex64 元素包含 8 字节输入和 8 字节原地写回，理论逻辑访存量约为每元素 16 字节。该算子计算量低、访存占比高，属于典型带宽敏感型向量算子。

任务书中的三个主要性能规模分别产生约 16 MiB、32 MiB 和 64 MiB 的逻辑读写流量。对应性能上限要求约等价于 1.24 TB/s、1.59 TB/s 和 1.56 TB/s 的有效带宽，因此设计重点是减少 Host 下发开销、提高连续搬运效率并重叠搬运与计算。

## 算子实现

### 整体实现方案

实现采用 Ascend C Kernel 直调模式。Host 侧完成参数校验、执行路径选择、分核和 Tiling 数据构造，通过 handle 中绑定的 Stream 异步下发 Kernel。Device 侧根据输入连续性和每核工作量选择以下三条路径：

| 路径 | 适用条件 | 实现方式 |
| --- | --- | --- |
| 单块 SIMD | `incx == 1`、地址 32 字节对齐，且每核数据不超过 16384 个 float | 单个 TBuf，整块搬入、Muls、整块搬出 |
| 流水 SIMD | `incx == 1`、地址 32 字节对齐，且每核数据较大 | 输入输出双缓冲，DataCopy 与向量计算流水重叠 |
| SIMT | `incx > 1` 或起始地址未按 32 字节对齐 | 每个线程按逻辑下标精确处理复数元素 |

这样既保证连续大向量的带宽利用率，又避免非连续步长或非对齐视图发生越界、覆盖间隙或对相邻内存进行填充写入。

### Host 侧设计

#### 参数校验

Host 侧按照“句柄检查、合法 no-op、数据指针检查、标量快速返回、物理长度检查”的顺序执行：

1. 检查 handle，空句柄立即返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 检查 `n` 和 `incx`，任一不大于零时直接成功返回。
3. 检查 `alpha` 和 `x`，正常计算时空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 读取 Host 标量；`alpha == 1.0f` 时直接成功返回。
5. 使用 `uint64_t` 计算物理跨度，并检查转换为字节数时是否溢出 `size_t`。
6. 查询 AIV 核数量；查询失败或结果为零时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。

no-op 检查先于数据指针解引用，保证 `n <= 0` 或 `incx <= 0` 时即使 `alpha`、`x` 为空也不会访问它们。

#### AIV 核数缓存

查询平台 AIV 核数存在 Host 开销。v3 使用线程局部缓存保存最近一次设备 ID 及其 AIV 核数：

- 每次真实计算通过 `aclrtGetDevice` 获取当前设备 ID。
- 当前设备发生变化或缓存无效时重新查询 AIV 核数。
- 缓存只保存能力值，不保存 handle、Stream、上下文或 Device 地址。
- 线程局部存储避免线程间竞争，也避免将第一张设备的核数错误复用于其他设备。

#### 执行路径选择

当 `incx == 1` 且 `x` 的起始地址按 32 字节对齐时选择 SIMD，否则选择 SIMT。非对齐输入不采用带 Padding 的块写路径，从而避免写入视图之前或之后的相邻数据。

SIMD 路径以每核约 2048 个复数元素作为初始工作量目标；SIMT 路径以每个 Block 约 256 个逻辑元素作为分核目标。实际 Block 数计算为：

```text
wanted_blocks = ceil(n / work_per_core)
block_count = min(aiv_core_count, wanted_blocks)
```

#### Tiling 数据

Host 侧构造 24 字节的 `CsscalTilingData`，按值传递给 Kernel：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| n | `uint32_t` | 复数逻辑元素数量 |
| incx | `uint32_t` | 复数元素步长 |
| simd | `uint32_t` | 0 为 SIMT，1 为流水 SIMD，2 为单块 SIMD |
| alpha | `float` | 缩放系数 |
| unitsPerCore | `uint32_t` | 每核完整 32 字节单元数量 |
| extraUnits | `uint32_t` | 不能均分的完整单元数量 |

Host 侧预先完成完整 32 字节单元的除法和取余，Device 各核只需要根据 Block ID 计算自身偏移与长度。

#### 异步语义

Host 接口不申请额外 Device Workspace，不将 Tiling 拷贝到独立 GM，不执行 Host/Device 数据复制，也不在接口内部同步 Stream。Kernel 直接下发到 `handle->stream`，调用者在读回结果前负责同步 Stream。

### Kernel 侧设计

#### SIMD 分核策略

连续数据被视为 `2 * n` 个连续 float 分量。一个 32 字节单元包含 8 个 float，即 4 个 complex64 元素。

完整 32 字节单元在所有 Block 间均匀分配，不能均分的单元依次分配给前面的 Block。仅最后一个 Block 负责不足 32 字节的尾部。各 Block 处理区间互不重叠，不会产生跨核写冲突。

#### 单块 SIMD 路径

当所有 Block 的数据量都不超过 `CSSCAL_SINGLE_TILE_FLOATS = 16384` 个 float 时，选择独立的单块 Kernel 入口：

1. 按实际长度向上对齐到 32 字节。
2. 申请一个最大 64 KiB 的 `TBuf<VECCALC>`。
3. 将数据从 GM 搬入 UB。
4. 使用 `Muls` 对全部 float 分量乘以 `alpha`。
5. 将结果从 UB 原地写回 GM。

该入口仅创建 `TPipe` 和一个 `TBuf`，避免小中型任务初始化通用双队列带来的额外开销。

#### 流水 SIMD 路径

当每核数据超过 16384 个 float 时，采用分 Tile 双缓冲流水：

- 单 Tile 为 `CSSCAL_TILE_FLOATS = 8192` 个 float，即 32 KiB。
- 输入队列包含两个 32 KiB 缓冲区。
- 输出队列包含两个 32 KiB 缓冲区。
- 输入和输出缓冲区合计占用 128 KiB UB。

流水过程先预取两个输入 Tile，随后交错执行下一 Tile 搬入、当前 Tile 计算和上一 Tile 写回，以重叠 MTE2、Vector 和 MTE3 阶段。流水结束时排空剩余输出，确保所有结果写回完成。

#### 搬运与尾块处理

对完整对齐的数据块使用 `DataCopy`。当最后一个 Tile 或最后一个 Block 的数据量不是 32 字节整数倍时，使用 `DataCopyPad` 按有效字节数精确搬入和写回。

UB 申请长度可以向上对齐，但 GM 写回仅覆盖有效分量，不能写入尾部 Padding，也不能修改向量外的相邻内存。

#### alpha 等于零的优化

当 `alpha == 0.0f` 时，不需要读取原输入数据，直接使用 `Duplicate` 在 UB 中生成正零并写回所有选中分量。该路径同时减少 GM 读取流量，并满足任务书对正零 bit-exact 的要求。

#### 事件同步

单块路径通过 MTE2 到 Vector、Vector 到 MTE3 的硬事件保证搬运、计算与写回顺序，最后使用流水线屏障排空操作。双缓冲路径通过 TQue 的入队、出队和缓冲区生命周期管理保证流水依赖。

### SIMT 路径设计

当 `incx > 1` 或 `x` 未按 32 字节对齐时，使用 SIMT 路径：

1. 将 `n` 个逻辑复数元素均匀分配给多个 Block。
2. 每个 Block 启动 256 个线程。
3. 每个线程以线程步长循环处理本 Block 的逻辑元素。
4. 使用 64 位整数计算 GM 分量偏移：

```text
float_offset = logical_index * incx * 2
```

5. 分别读取和写回实部、虚部。

SIMT 路径只写逻辑向量选中的两个 float 分量，因此天然支持任意正步长，并且不会修改步长间隙。`alpha == 0.0f` 时同样直接写正零。

### 正确性与边界分析

#### 数值范围

`n` 为正 `int` 时，连续路径的分量数量 `2 * n` 最大为 `2^32 - 2`，可以由 `uint32_t` 表示。物理跨度使用 `uint64_t` 计算，避免 `(n - 1) * incx` 在 32 位整数中溢出。

#### 内存安全

- SIMD 路径按完整 32 字节单元分核，尾部只由最后一个 Block 处理。
- 尾块写回只使用有效字节数。
- 非对齐地址和非单位步长转入逐元素 SIMT 路径。
- 测试在向量两侧布置保护区，并逐位检查保护区、步长间隙和 no-op 输入。

#### 特殊浮点值

- `alpha == 1.0f` 不执行乘法，从而保留 NaN payload 和有符号零等位模式。
- `alpha == 0.0f` 根据任务书强制输出正零。
- 其他标量使用普通 FLOAT32 乘法；Inf 和 NaN 的分类、无穷符号及数值误差由测试分别检查。

## 测试设计

### 测试工程

测试工程位于 `test/scal/csscal/`，仅在 `SOC_VERSION=ascend950` 时注册。测试使用 ops-blas 的 CSV 加载和 GTest 框架，CPU golden 对普通缩放调用 `cblas_csscal`，对任务书规定的零标量行为使用显式置零分支。

### 功能与精度覆盖

任务原始 CSV 包含 1000 条精度用例和 200 条性能用例。当前测试保留全部原始用例，并补充 226 条精度回归用例。完整精度阶段还包含 3 条固定接口测试，总计 1229 项检查。

测试覆盖以下场景：

1. `n=0`、负 `n`、`incx=0` 和负 `incx` 的 no-op。
2. 空 handle、空 alpha 和空 x。
3. `alpha=0`、`alpha=1`、负数、小数和较大标量。
4. `incx=1/2/3/5/7` 等连续与非连续访问。
5. 小尺寸、质数尺寸、2 的幂及其相邻尺寸、大尺寸和非对齐尺寸。
6. 起始地址偏移、32 字节尾块、单块与流水路径边界。
7. 均匀分布、正态分布、全零、交替值、极值、Inf 和 NaN。
8. 向量前后保护区、步长间隙和 no-op 缓冲区不被修改。
9. `INT_MAX` 规模与步长组合的物理地址容量溢出检查。

实部和虚部分别统计 matched ratio、最大绝对误差和最大 ULP。精度判定使用：

```text
atol = 2^-16
rtol = 2^-10
matched_ratio >= 0.99
max_abs_error <= 1e-2 或 max_ulp <= 32
```

### 性能测试

性能阶段执行原始 200 条连续访存用例。每条用例先验证数值结果，再预热 10 次并采集 100 次有效样本。每次正式采样前在计时区间外恢复输入，防止重复缩放导致溢出。

计时使用带时间线的 ACL Device Event，记录一次真实 `aclblasCsscal` 调用在 Stream 上的耗时，并输出平均值、最小值、最大值及全部原始样本。GTest 断言、输入恢复和结果读回不放入 Device Event 区间。

三项任务书指定用例直接使用 13.57 us、21.05 us 和 43.02 us 判定；其余用例依据 GPU 基线和 0.4 倍率要求生成上限。性能结论使用全部有效样本的平均值，不删除慢样本，也不使用 GTest 总运行时间代替设备事件耗时。

Device Event 区间仍可能受到 Host 提交空隙和设备调度影响。接近阈值或出现离群慢样本时，可使用 msprof 进行诊断，但 profiler 数据不直接替换正常验收计时结果。

### 当前验证状态

v2 远程反馈显示十轮精度均通过，性能测试有 8/10 轮整体通过，剩余超限集中于中小尺寸及偶发慢样本。v3 针对 Host 查询、单块 Kernel 初始化和计时区间中的测试代码进行了优化。

当前 v3 为待远程验证的候选版本。本地已完成源码审阅、Python 语法和交付完整性检查，但尚未在 Ascend 950PR + CANN 9.1.0 环境完成 v3 编译、1229 项完整精度和十轮性能回归。因此，最终验收数据、自测报告截图及实测内存数据应以远程执行结果为准，不沿用 v2 结果作为 v3 通过结论。

# 可维可测分析

## 可维护性

1. Host、Kernel 和 Tiling 数据分别放置在独立源文件中，职责清晰。
2. 连续 SIMD、流水 SIMD 和非连续 SIMT 路径通过明确模式字段选择，便于独立调优。
3. Tile 大小、单块容量、对齐单位和 SIMT 线程数使用具名常量统一管理。
4. 实现复用 ops-blas 的公共句柄、日志、CMake 和测试设施，不引入私有平行 API。
5. 设计、远程测试、性能反馈和本地验证记录随代码保存，便于后续定位版本差异。

## 可测试性

1. CSV 用例具有固定随机种子，可重复生成输入。
2. GTest XML 记录每个用例状态及精度、性能属性。
3. Runner 校验预期测试集合，缺失、重复、跳过、超时和失败均不能判定通过。
4. 反馈包记录环境、源码哈希、构建指纹、测试日志和性能样本，降低代码与结果错配风险。
5. 单块、流水、SIMT、尾块、偏移、no-op 和异常参数均有针对性覆盖。

## 可观测性

1. Host 参数错误通过 ops-blas 日志输出具体原因。
2. 性能测试记录 Device Event 样本和 Host 提交耗时，便于区分稳定耗时与调度抖动。
3. 反馈中记录设备、CANN 版本、二进制和动态库指纹。
4. 对性能异常保留完整原始样本，可进一步结合 msprof 分析。

# 兼容性分析

1. 接口签名沿用 `include/cann_ops_blas.h` 的既有声明，对调用方不引入新的 API。
2. 实现仅位于 `arch35`，通过现有 CMake 架构选择机制参与 Ascend 950PR 构建，不改变 `arch22` 等其他架构实现。
3. 算子使用 handle 绑定的 Stream，遵循 ops-blas 现有异步执行模型。
4. no-op 与正步长语义对齐 Netlib `csscal`；负步长不反向遍历。
5. `alpha=0` 对非有限输入强制写正零是任务书特例，需要在正式验收中确认其优先级并保持实现、golden 和文档一致。

# 风险分析

| 风险 | 影响 | 应对措施 |
| --- | --- | --- |
| 小尺寸 Host 下发或事件开销占比较高 | 性能均值接近或超过阈值 | 缓存设备能力、使用轻量单块 Kernel、保留原始样本并进行绑核对照 |
| 非对齐尾块发生越界写 | 破坏相邻数据 | 使用 DataCopyPad 精确写回，并通过保护区逐位检查 |
| 大规模步长计算溢出 | 地址计算错误或越界 | Host 使用 uint64_t 计算跨度并检查 size_t，SIMT 使用 64 位分量偏移 |
| alpha=0 与 Netlib 非有限值语义冲突 | golden 或验收结论不一致 | 保留规格裁决记录，统一修改 Kernel、golden 和测试预期 |
| 旧构建产物与新源码混用 | 产生不可信测试结果 | 使用源码、二进制和动态库指纹校验，源码变化后强制重编译 |
| 缺少 950PR 真机验证 | 无法确认编译兼容性和真实性能 | 在 CANN 9.1.0 + Ascend 950PR 上依次执行冒烟、完整精度和十轮性能测试 |

# 参考资料

1. CANN ops-blas 开源仓库：`https://gitcode.com/cann/ops-blas`
2. Netlib BLAS csscal：`https://www.netlib.org/blas/csscal.f`
3. NVIDIA cuBLAS Scal 接口说明：`https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-scal`
4. 昇腾 Ascend C 算子开发文档：`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html`
5. 生态算子开源精度标准：`https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md`

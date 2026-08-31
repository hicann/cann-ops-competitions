# 需求背景（required）

## 需求来源

参考 cuBLAS `cublasCgeru` 与 Netlib BLAS `cgeru` 的接口语义，在
Ascend 950PR 上使用 Ascend C 实现单精度复数无共轭秩一更新算子
`aclblasCgeru`，并提交至 ops-blas 开源仓。

目标接口为 ops-blas 公共 BLAS 接口。接口声明新增至
`include/cann_ops_blas.h`，供不同产品线共用，不定义 Ascend 950PR
私有平行接口。

## 背景介绍

### Cgeru 算子功能

`aclblasCgeru` 是 BLAS Level-2 算子，完成如下原地矩阵更新：

```text
A = alpha * x * y^T + A
```

其中：

- `alpha` 为 Host 侧单精度复数标量。
- `x` 为包含 `m` 个逻辑元素的 Device 侧复数向量。
- `y` 为包含 `n` 个逻辑元素的 Device 侧复数向量。
- `A` 为 Device 侧 `m x n` 列主序复数矩阵，并原地写回结果。
- `y^T` 只执行转置，不对 `y` 取共轭。

逐元素语义为：

```text
A(row, col) = alpha * x(row) * y(col) + A(row, col)
```

其中 `0 <= row < m`、`0 <= col < n`。

### 现状分析

ops-blas 已有同族 `aclblasCgerc` 接口与工程框架，但 `aclblasCgeru`
公共接口声明、Ascend 950PR 对应 arch35 实现和测试用例需要补齐。

`Cgeru` 与 `Cgerc` 的关键差异是 `Cgeru` 对 `y` 不取共轭。实现和
golden 均需按无共轭语义计算，不能直接复用带共轭的乘法路径。

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上实现 `aclblasCgeru`，要求：

1. 与 cuBLAS `cublasCgeru` 的核心功能和参数语义对齐。
2. 支持 `aclblasComplex` / complex64 数据类型。
3. 矩阵采用 Column-Major 存储，支持 `lda >= max(1, m)`。
4. `incx`、`incy` 支持正负步长，且不得为零。
5. 支持 Netlib `cgeru` 的零维和 `alpha=(0,0)` quick return。
6. 通过 handle 绑定的 stream 异步直调 Ascend C Kernel。
7. 核心计算全部在 Device 侧完成，不使用 CPU fallback。
8. 正确性、性能和工程质量满足任务书验收标准。

## 接口定义

```cpp
aclblasStatus_t aclblasCgeru(
    aclblasHandle_t handle,
    int m,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x,
    int incx,
    const aclblasComplex* y,
    int incy,
    aclblasComplex* A,
    int lda);
```

## 参数语义

| 参数 | 位置 | 语义与约束 | 异常行为 |
| --- | --- | --- | --- |
| `handle` | Host | 有效 ops-blas 句柄，携带 stream | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `m` | Host | `A` 的行数，`m >= 0` | `m < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n` | Host | `A` 的列数，`n >= 0` | `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha` | Host | complex64 标量指针，非空 | 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `x` | Device | complex64 只读向量，逻辑长度 `m` | 非 quick return 场景为空时返回非法值 |
| `incx` | Host | `x` 的元素步长，支持正负值 | 等于 0 时返回非法值 |
| `y` | Device | complex64 只读向量，逻辑长度 `n`，不取共轭 | 非 quick return 场景为空时返回非法值 |
| `incy` | Host | `y` 的元素步长，支持正负值 | 等于 0 时返回非法值 |
| `A` | Device | complex64 列主序输入输出矩阵 | 非 quick return 场景为空时返回非法值 |
| `lda` | Host | `A` 的前导维，`lda >= max(1, m)` | 不满足约束时返回非法值 |

向量物理存储元素数至少为：

```text
xStorage = 1 + (m - 1) * abs(incx)
yStorage = 1 + (n - 1) * abs(incy)
```

## 需求拆解

1. 在公共头文件增加 `aclblasCgeru` 声明。
2. 在 `blas/ger/arch35/` 增加 Host、Kernel 和 tiling 实现。
3. Host 侧实现参数校验、quick return、tiling 和异步 Kernel launch。
4. Kernel 侧实现 complex64 无共轭乘法和矩阵原地更新。
5. 支持单位步长、非单位步长、负步长和带 padding 的 `lda`。
6. 为连续大 shape 设计 UB 数据复用路径，为通用 stride 保留泛化路径。
7. 在 `test/ger/cgeru/arch35/` 增加 CSV、GTest、NPU wrapper 和 golden。
8. 补充算子 README、产品支持表和复现步骤。

# 详细设计（required）

## 算子分析

### 数学公式

对每个矩阵元素：

```text
A[row + col * lda] =
    alpha * x[xBase + row * incx] * y[yBase + col * incy]
    + A[row + col * lda]
```

设复数 `a=(ar, ai)`、`b=(br, bi)`，复数乘法为：

```text
real(a*b) = ar*br - ai*bi
imag(a*b) = ar*bi + ai*br
```

`y` 直接参与乘法，不执行虚部取反。

### 支持数据类型

| 输入或输出 | 数据类型 |
| --- | --- |
| `alpha` | `aclblasComplex` / complex64 |
| `x` | `aclblasComplex` / complex64 |
| `y` | `aclblasComplex` / complex64 |
| `A` | `aclblasComplex` / complex64 |

`aclblasComplex` 的实部和虚部均为 float32，内存布局以
`include/cann_ops_blas_common.h` 定义为准。

### 支持形状与步长

- `m`、`n` 为运行时参数。
- `A` 的存储形状为 `[lda, n]`，只更新前 `m x n` 区域。
- `incx`、`incy` 支持正数和负数。
- 负步长逻辑起点为：

```text
xBase = (incx < 0) ? (m - 1) * (-incx) : 0
yBase = (incy < 0) ? (n - 1) * (-incy) : 0
```

## 算子实现

### Host 侧设计

#### 参数检查和 quick return

Host 侧按以下顺序执行：

1. `handle == nullptr` 时返回句柄空指针错误。
2. `m < 0` 或 `n < 0` 时返回非法值。
3. `alpha == nullptr` 时返回非法值。
4. 校验 `incx != 0`、`incy != 0` 和 `lda >= max(1, m)`。
5. `m == 0` 或 `n == 0` 时返回成功，不访问 Device 数据。
6. `alpha == (0,0)` 时返回成功，不访问或写入 `x`、`y`、`A`。
7. 常规计算场景校验 `x`、`y` 和 `A` 非空。
8. 生成 tiling 数据，并在 handle 绑定的 stream 上异步启动 Kernel。

#### Tiling 数据

Host 侧向 Kernel 传递：

| 字段 | 说明 |
| --- | --- |
| `m`、`n`、`lda` | 矩阵形状和前导维 |
| `incx`、`incy` | 向量步长 |
| `xBase`、`yBase` | 负步长逻辑起点 |
| `alphaReal`、`alphaImag` | 标量实部和虚部 |
| `totalElements` | 有效输出元素数量 |
| `coreNum` | 实际参与计算的 AIV 数量 |
| `tileSize` | 每次处理的数据规模 |
| `computeMode` | 通用路径或连续优化路径 |

#### 分核策略

通用路径将 `m * n` 个输出元素按线性索引切分到多个 AIV。Host 根据
输出规模和平台核数动态选择 `coreNum`，前若干核处理余数元素，保证任务
分配均衡。

连续路径按列分核，使每个 AIV 处理若干完整或部分矩阵列，保持
Column-Major 下的连续访问，并提高 `x` 的复用率。

#### Workspace 和异步语义

算子不需要额外 workspace。Kernel 使用调用方通过 handle 设置的 stream
异步执行，不创建私有 stream，也不在接口内部执行 Host 侧强制同步。

### Kernel 侧设计

#### 通用 stride 路径

通用路径将二维输出展平。每个任务根据线性索引恢复 `row`、`col`，计算
`x`、`y` 和 `A` 的偏移，完成一次复数乘加并原地写回。每个矩阵元素仅由
一个任务写入，不需要原子操作或跨核同步。

该路径覆盖：

- 正负 `incx`、`incy`
- 非单位步长
- `lda > m` 的 padding
- 非方阵和非对齐 shape

#### 连续访问优化路径

当 `incx == 1`、`incy == 1` 且 shape 达到优化阈值时，使用列分块路径：

1. 每个 AIV 处理一组连续矩阵列。
2. 将一段 `x` 缓存到 UB，供多列重复使用。
3. 每列读取一次 `y[col]`，预计算 `alpha * y[col]`。
4. 分块读入 `A`，执行复数乘加并原地写回。
5. 尾块根据有效元素数单独处理，避免越界访问。

该路径减少对 `x` 的重复 GM 读取，并使 `A` 访问保持列主序连续。

#### FP32 特殊值处理

Kernel 保持与参考实现一致的复数运算顺序，覆盖普通有限值、零、次正规
数、Inf 和 NaN。避免使用会改变 IEEE 特殊值传播行为的代数重排。

### 内存访问和同步

- `x`、`y` 只读，`A` 原地读写。
- 单个输出元素只由一个任务更新，不存在写冲突。
- 连续路径使用 UB 分块和对齐搬运提高数据复用。
- Kernel 不分配 GM 临时缓冲区。
- 同一 stream 上的数据依赖由 stream 顺序保证。

## 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | --- |
| Ascend 950PR | 是 |

## 算子约束限制

- 仅支持 `aclblasComplex` / complex64。
- `A` 使用 Column-Major 存储。
- `m >= 0`、`n >= 0`。
- `incx != 0`、`incy != 0`。
- `lda >= max(1, m)`。
- 不支持超出 BLAS `incx`、`incy`、`lda` 语义的任意 Tensor view。
- `A` 原地更新，不返回视图。
- `m == 0`、`n == 0` 或 `alpha == (0,0)` 时为合法 no-op。
- 依赖 `aclblasSetStream` 绑定 stream，读取结果前由调用方同步。

# 可维可测分析

## 正确性测试方案

使用 ops-blas 测试框架加载官方 CSV，通过 GTest 调用
`aclblasCgeru`。golden 使用 cblas / Netlib `cgeru` 生成，`y` 不取共轭，
并对输出矩阵有效区域的实部和虚部分别比较。

测试覆盖：

- 0、1、小质数、2 的幂、2 的幂相邻值和非对齐尺寸。
- 方阵、宽矩阵和高矩阵。
- `incx`、`incy` 的 `+/-1`、`+/-2`、`+/-3` 组合。
- 紧凑 `lda` 和多种 padding。
- `alpha=(0,0)`、`(1,0)`、纯实数、纯虚数和大值。
- 均匀分布、正态分布、全零、交替值、极端值、Inf 和 NaN。
- 零维、空指针、负维度、零步长和非法 `lda`。
- 非默认 stream 上的异步执行和下游数据依赖。
- 重复运行、内存越界和资源泄漏检查。

## 精度标准

| 数据类型 | rtol | atol | required matched ratio | max absolute error |
| --- | --- | --- | --- | --- |
| complex64（实部/虚部分别按 FP32） | `2^-10` | `2^-16` | `0.99` | `1e-2` 或 `32 * ULP` |

逐元素通过条件为：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

用例同时满足 matched ratio 和最大绝对误差要求时判定通过。

## 性能测试方案

在 Ascend 950PR 和任务书指定 CANN 9.1.0 环境中，先执行 warmup，再进行
超过 50 次有效采样并计算平均单次耗时。使用官方测试框架或 Device event
计时，并以任务书标杆为验收上限：

| m | n | incx | incy | 平均耗时上限（us） |
| --- | --- | --- | --- | --- |
| 512 | 512 | 1 | 1 | 1.38 |
| 1024 | 1024 | 1 | 1 | 2.05 |
| 2048 | 2048 | 1 | 1 | 7.12 |

使用 Profiler 分析 Kernel 启动、GM 带宽、AIV 利用率和流水等待，重点
优化小 shape 调度开销、连续路径 `x` 的 UB 复用以及矩阵搬运效率。

## 构建和工程质量测试方案

- X86 与 ARM Host 编译。
- Ascend 950PR 对应 arch35 构建目标。
- 仓库 precommit、codecheck、codestyle 和安全扫描。
- CANN 9.1.0 环境下完整正确性和性能复验。
- README 产品支持表标注 Ascend 950PR 支持。
- 测试 README 提供可复现的构建和运行步骤。

## 兼容性分析

`aclblasCgeru` 为新增公共接口，不修改已有接口语义。实现遵循 ops-blas
句柄、stream、状态码和工程目录约定，对现有算子无行为影响。

## 风险与后续计划

- 复数特殊值可能受编译器融合和表达式重排影响，需要保留专项回归。
- 小 shape 主要受 Kernel 启动开销影响，需要结合 Profiler 调整分核策略。
- 大 shape 主要受 GM 带宽影响，需要持续优化 UB 复用和矩阵分块。
- 负步长和大 `lda` 场景需要重点检查地址计算溢出和访存边界。
- 设计评审通过后继续完成全量官方测试、性能调优、Profiler 证据和
  CANN 9.1.0 最终复验。

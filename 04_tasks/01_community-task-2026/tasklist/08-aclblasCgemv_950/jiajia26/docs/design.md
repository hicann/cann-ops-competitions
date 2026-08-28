# 需求背景（required）

## 需求来源

参考 cuBLAS `cublasCgemv` 和 Netlib BLAS `cgemv` 的接口语义，在 Ascend 950PR 上基于 Ascend C 编程语言实现单精度复数矩阵-向量乘算子 `aclblasCgemv`，并提交至 ops-blas 开源仓。

目标接口为 ops-blas 仓 `include/cann_ops_blas.h` 中已有的公共 BLAS 接口，禁止新增 950PR 私有接口：

```cpp
aclblasStatus_t aclblasCgemv(
    aclblasHandle_t handle,
    aclblasOperation_t trans,
    int m,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A,
    int lda,
    const aclblasComplex* x,
    int incx,
    const aclblasComplex* beta,
    aclblasComplex* y,
    int incy);
```

## 背景介绍

### CGEMV 算子功能

GEMV 是 BLAS Level-2 算子，完成稠密矩阵与向量乘加。`aclblasCgemv` 面向 complex64 数据类型，计算公式为：

```text
y = alpha * op(A) * x + beta * y
```

其中：

- `A` 为 `m x n` 复数矩阵，按 BLAS 约定采用 Column-Major 存储。
- `x`、`y` 为复数向量，支持正负步长。
- `alpha`、`beta` 为 Host 侧复数标量。
- `op(A)` 支持不转置、转置和共轭转置。

`op(A)` 的具体语义如下：

| trans | op(A) | x 逻辑长度 | y 逻辑长度 |
| --- | --- | --- | --- |
| `ACLBLAS_OP_N` | `A` | `n` | `m` |
| `ACLBLAS_OP_T` | `A^T` | `m` | `n` |
| `ACLBLAS_OP_C` | `A^H` | `m` | `n` |

复数场景下 `ACLBLAS_OP_C` 必须执行共轭转置，即读取矩阵元素时对虚部取反；不能退化为普通转置。

### 现状分析

ops-blas 仓已有公共头文件声明 `aclblasCgemv`，但 Ascend 950PR 对应 arch35 实现和测试用例仍需补齐。本任务需要在 `blas/gemv/arch35/` 中实现句柄式 BLAS API，通过 `aclblasHandle_t` 绑定的 stream 直调 Ascend C kernel。

同时需要在 `blas/gemv/README.md` 中补充 `aclblasCgemv` 的参数说明和产品支持表，保证接口文档与实现一致。

# 需求分析（required）

## 需求描述

实现 `aclblasCgemv` complex64 矩阵-向量乘加算子，功能与 cuBLAS `cublasCgemv` 核心语义对齐，覆盖：

1. `ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C` 三种矩阵操作。
2. Column-Major 矩阵访问，支持 `lda >= max(1, m)`。
3. `incx`、`incy` 支持正负步长，且 `incx != 0`、`incy != 0`。
4. complex64 复数乘法、复数加法、复数标量缩放。
5. Netlib `cgemv` quick return 和退化计算语义。
6. 参数非法时返回 ops-blas 约定状态码。
7. Ascend 950PR 性能场景达到任务书标杆要求。

## 参数语义

| 参数 | 语义 | 约束 | 异常行为 |
| --- | --- | --- | --- |
| `handle` | ops-blas 句柄，携带 stream | 非空且已创建 | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` | 矩阵操作枚举 | `ACLBLAS_OP_N/T/C` | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `m` | A 行数 | `m >= 0` | `m < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n` | A 列数 | `n >= 0` | `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha` | Host 侧复数标量 | 非空 | 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | Device 侧复数矩阵 | 需要读 A 时非空 | 非法返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | Column-Major 前导维 | `lda >= max(1, m)` | 非法返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `x` | Device 侧输入向量 | 需要读 x 时非空 | 非法返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | x 元素步长 | `incx != 0` | 非法返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `beta` | Host 侧复数标量 | 非空 | 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `y` | Device 侧输入输出向量 | 需要写 y 时非空 | 非法返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incy` | y 元素步长 | `incy != 0` | 非法返回 `ACLBLAS_STATUS_INVALID_VALUE` |

## 需求拆解

1. Host 侧实现参数校验、quick return 判断、tiling 计算、kernel launch。
2. Kernel 侧实现 `trans=N` 行方向归约，每个输出元素计算一行点积。
3. Kernel 侧实现 `trans=T` 列方向归约，每个输出元素计算一列点积。
4. Kernel 侧实现 `trans=C` 列方向归约并对 A 元素取共轭。
5. 实现 `alpha=(0,0)` 时的 `y=beta*y` 缩放分支，避免不必要读取 A 和 x。
6. 支持 `beta=(0,0)` 时不读取 y 的输入值，直接写入 `alpha * dot`。
7. 支持正负 `incx`、`incy`，负步长按 Netlib 语义从反向起点遍历。
8. 补充 README 参数说明、产品支持表和测试用例。

# 详细设计（required）

## 算子分析

### 数学公式

当 `trans == ACLBLAS_OP_N` 时：

```text
y_i = alpha * sum(j=0..n-1, A[i + j * lda] * x[jx]) + beta * y[iy]
```

其中 `i in [0, m)`。

当 `trans == ACLBLAS_OP_T` 时：

```text
y_j = alpha * sum(i=0..m-1, A[i + j * lda] * x[ix]) + beta * y[jy]
```

其中 `j in [0, n)`。

当 `trans == ACLBLAS_OP_C` 时：

```text
y_j = alpha * sum(i=0..m-1, conj(A[i + j * lda]) * x[ix]) + beta * y[jy]
```

其中 `j in [0, n)`。

复数乘法采用：

```text
(a + bi) * (c + di) = (ac - bd) + (ad + bc)i
```

复数加法采用分量相加。共轭操作为：

```text
conj(a + bi) = a - bi
```

### 支持数据类型

| 输入/输出 | 数据类型 |
| --- | --- |
| `A` | `aclblasComplex` / complex64 |
| `x` | `aclblasComplex` / complex64 |
| `y` | `aclblasComplex` / complex64 |
| `alpha` | `aclblasComplex` / complex64 |
| `beta` | `aclblasComplex` / complex64 |

`aclblasComplex` 的内存布局以 ops-blas 仓 `include/cann_ops_blas_common.h` 定义为准，实部和虚部均为 float32。

### 支持形状

| 场景 | A 有效形状 | A 存储形状 | x 逻辑长度 | y 逻辑长度 |
| --- | --- | --- | --- | --- |
| `trans=N` | `[m, n]` | `[lda, n]` | `n` | `m` |
| `trans=T/C` | `[m, n]` | `[lda, n]` | `m` | `n` |

向量实际存储元素数至少为：

```text
1 + (len - 1) * abs(inc)
```

当 `incx < 0` 时，逻辑第 0 个元素对应物理偏移：

```text
(lenx - 1) * (-incx)
```

当 `incy < 0` 时，逻辑第 0 个元素对应物理偏移：

```text
(leny - 1) * (-incy)
```

## Host 侧设计

### 参数检查流程

Host 侧按以下顺序检查，确保错误码稳定：

1. `handle == nullptr`，返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `trans` 不属于 `ACLBLAS_OP_N/T/C`，返回 `ACLBLAS_STATUS_INVALID_ENUM`。
3. `m < 0`、`n < 0`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `alpha == nullptr` 或 `beta == nullptr`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. `lda < max(1, m)`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. `incx == 0` 或 `incy == 0`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
7. 根据 quick return 和退化分支判断是否需要检查 `A`、`x`、`y`。

quick return 和退化分支：

| 条件 | 行为 |
| --- | --- |
| `m == 0 || n == 0` | 返回成功，不 launch kernel |
| `alpha == (0,0) && beta == (1,0)` | 返回成功，不读写 y |
| `alpha == (0,0) && beta != (1,0)` | 仅执行 `y = beta * y` |
| `beta == (0,0)` | 计算输出时不读取 y 输入值 |

指针检查策略：

| 分支 | A | x | y |
| --- | --- | --- | --- |
| quick return | 不检查 | 不检查 | 不检查 |
| `alpha == 0 && beta != 1` | 不检查 | 不检查 | 需非空 |
| 常规 GEMV | 需非空 | 需非空 | 需非空 |

### Tiling 参数

Host 侧生成以下 tiling 信息并传入 kernel：

| 字段 | 说明 |
| --- | --- |
| `m`、`n`、`lda` | 矩阵形状和前导维 |
| `incx`、`incy` | 向量步长 |
| `lenx`、`leny` | x/y 逻辑长度 |
| `xBaseOffset`、`yBaseOffset` | 负步长时的逻辑起点偏移 |
| `transMode` | `N/T/C` 分支标记 |
| `computeMode` | 常规 GEMV 或仅缩放 y |
| `coreNum` | 实际参与计算的 AI Core 数 |
| `tileRows` / `tileCols` | 每核输出元素范围 |

`lenx`、`leny` 计算：

```text
if trans == N:
    lenx = n
    leny = m
else:
    lenx = m
    leny = n
```

负步长起点：

```text
xBaseOffset = (incx < 0) ? (lenx - 1) * (-incx) : 0
yBaseOffset = (incy < 0) ? (leny - 1) * (-incy) : 0
```

### 分核策略

GEMV 输出向量各元素相互独立，采用按输出元素维度切分的并行策略：

- `trans=N`：输出长度为 `m`，按行切分，每个 core 负责若干行的点积。
- `trans=T/C`：输出长度为 `n`，按列切分，每个 core 负责若干列的点积。

Host 侧根据输出长度 `leny` 和平台 AI Core 数确定实际 core 数：

```text
coreNum = min(platformCoreNum, leny)
```

将 `leny` 均分到 `coreNum`：

```text
base = leny / coreNum
remain = leny % coreNum
```

前 `remain` 个 core 每核多处理 1 个输出元素，保证负载均衡。

### TilingKey 规划

| TilingKey | 场景 |
| --- | --- |
| `0` | `trans=N` 常规 GEMV |
| `1` | `trans=T` 常规 GEMV |
| `2` | `trans=C` 常规 GEMV |
| `3` | `alpha=(0,0)` 且 `beta!=(1,0)`，仅缩放 y |

Host 侧根据 `trans` 和 `alpha/beta` 特殊值选择 TilingKey，kernel 侧使用不同路径减少分支判断。

## Kernel 侧设计

Kernel 采用 `Init` + `Process` 结构。`Process` 中按输出元素循环执行 `CopyIn`、`Compute`、`CopyOut`。

### 数据访问

`A` 为 Column-Major：

```text
A(i, j) = A[i + j * lda]
```

向量访问：

```text
x(k) = x[xBaseOffset + k * incx]
y(k) = y[yBaseOffset + k * incy]
```

当 `incx` 或 `incy` 为负数时，base offset 保证逻辑顺序与 Netlib 一致。

### `trans=N` 计算路径

每个 core 负责连续的输出行 `i`。对每一行执行：

```text
sum = 0
for j in [0, n):
    a = A[i + j * lda]
    xv = x[xBaseOffset + j * incx]
    sum += a * xv

out = alpha * sum
if beta != 0:
    out += beta * y[yBaseOffset + i * incy]
y[yBaseOffset + i * incy] = out
```

该路径访问 A 的 stride 为 `lda`，适合中小矩阵和任务给定性能基线。实现时优先保证单输出元素归约正确性，再结合 UB 分块减少重复访存。

### `trans=T` 计算路径

每个 core 负责连续的输出列 `j`。对每一列执行：

```text
sum = 0
for i in [0, m):
    a = A[i + j * lda]
    xv = x[xBaseOffset + i * incx]
    sum += a * xv

out = alpha * sum
if beta != 0:
    out += beta * y[yBaseOffset + j * incy]
y[yBaseOffset + j * incy] = out
```

该路径对 A 的一列连续读取，访存局部性较好。`x` 在多个输出列之间复用，后续优化可按 UB 容量缓存 x 分块。

### `trans=C` 计算路径

共轭转置与 `trans=T` 的输出维度一致，仅矩阵元素读取后执行共轭：

```text
sum = 0
for i in [0, m):
    a = A[i + j * lda]
    a.imag = -a.imag
    xv = x[xBaseOffset + i * incx]
    sum += a * xv
```

其余 `alpha`、`beta` 缩放与写回逻辑与 `trans=T` 一致。

### 仅缩放 y 路径

当 `alpha == (0,0)` 且 `beta != (1,0)` 时不读取 A 和 x，只对 y 执行：

```text
for k in assigned output elements:
    y[k] = beta * y[k]
```

当 `beta == (0,0)` 时可直接写零；常规 GEMV 中也不读取 y 输入值。

### UB 与流水

由于 CGEMV 属于带归约的 BLAS Level-2 算子，单个输出元素需要遍历一个输入维度。实现上采用输出元素并行 + 归约维度分块：

1. 按输出元素分配 core。
2. 每个 core 在归约维度上按 tile 搬运 A 和 x。
3. 在 UB 中完成复数乘加累积。
4. 一个输出元素计算完成后再执行 beta 融合和写回。

对 `trans=T/C`，A 的列方向连续，可优先采用较大的归约 tile；对 `trans=N`，A 按列跨步读取同一行，首版以正确性为主，后续可进一步采用多行并行 tile，提高 A 列块复用。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

1. 仅支持 complex64，即 `aclblasComplex`。
2. A 按 BLAS Column-Major 存储，不支持 Row-Major 语义。
3. 不支持除 `lda/incx/incy` 外的任意非连续 Tensor 视图。
4. 不涉及 broadcast。
5. 不涉及 dynamic shape 编译期泛化，`m/n/lda/incx/incy` 为运行时 Host 入参。
6. y 为原地输入输出，结果覆盖原 y。
7. `m=0` 或 `n=0` 为合法 quick return。
8. `beta=(0,0)` 时 y 的输入值不参与计算。
9. 计算结果允许与 CPU/Netlib golden 存在浮点舍入差异，按任务精度标准判定。

# 可维可测分析

## 精度标准

golden 由 Netlib BLAS `cgemv` 或等价 cblas 复数实现生成，实部、虚部分别按 FLOAT32 标准比对：

| 数据类型 | rtol | atol | matched ratio | max abs error |
| --- | --- | --- | --- | --- |
| complex64 分量 | `2^-10` | `2^-16` | `>= 0.99` | `<= 1e-2` 或 `<= 32 * ULP` |

逐元素判定：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

实部和虚部均满足阈值时，该 complex64 元素判定通过。

## 性能标准

在 Ascend 950PR 上执行性能测试，warmup 后有效采样大于 50 次，平均单次耗时不高于任务书标杆：

| case | trans | m | n | 标杆耗时 Avg time(us) |
| --- | --- | --- | --- | --- |
| 1 | `ACLBLAS_OP_N` | 512 | 512 | 2.18 |
| 2 | `ACLBLAS_OP_N` | 2048 | 2048 | 3.86 |
| 3 | `ACLBLAS_OP_T` | 2048 | 2048 | 3.51 |

## 自测用例设计

测试用例使用 CSV 驱动，覆盖以下类别：

| 类别 | 覆盖内容 |
| --- | --- |
| 基础功能 | `trans=N/T/C`，小尺寸和常规方阵 |
| 尺寸扫描 | `1`、小质数、2 的幂、2 的幂±1、非对齐值、大尺寸 |
| 非方阵 | `m < n` 宽矩阵、`m > n` 窄矩阵 |
| 标量特殊值 | `alpha/beta` 为 `0`、`1`、`-1`、纯虚数、一般复数 |
| 前导维 | `lda=max(1,m)` 和 `lda>m` padding |
| 步长 | `incx/incy` 为 `±1`、`±2`、`±3` |
| quick return | `m=0`、`n=0`、`alpha=0 && beta=1` |
| 退化计算 | `alpha=0 && beta!=1`、`beta=0` |
| 负向用例 | 空指针、非法枚举、负维度、非法 lda、零步长 |
| 特殊值 | A/x/y 中包含 Inf、NaN 的传播场景 |
| 性能用例 | 任务书指定性能 case 和扩展规模 |

负向用例期望错误码：

| 场景 | 期望 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` 非法 | `ACLBLAS_STATUS_INVALID_ENUM` |
| `m < 0` 或 `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha == nullptr` 或 `beta == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda < max(1,m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx == 0` 或 `incy == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 常规计算时 `A/x/y == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

## 可维护性分析

1. 复数基础操作封装为内联函数，避免三种转置路径重复实现实部/虚部计算。
2. Host 侧将维度、步长、base offset、分核结果统一写入 tiling，kernel 侧只消费 tiling 数据。
3. `trans=N/T/C` 使用独立 TilingKey，便于后续针对不同访存模式做性能优化。
4. README、测试 CSV、GTest 入口与实现代码同步提交，保证验收人可复现。

## 兼容性分析

本任务使用 ops-blas 已有 `aclblasCgemv` 公共接口，不改变现有 ABI。新增实现位于 `blas/gemv/arch35/`，仅增加 Ascend 950PR 产品支持，不影响其他产品线已有接口声明和实现。

# aclblasCgeru 算子设计文档

# 需求背景（required）

## 需求来源

本任务来自 CANN 社区算子实操工坊广州站 A2/A3 算子开发任务，要求在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3 系列产品）上使用 Ascend C/CATLASS 编程语言开发单精度复数无共轭秩 1 更新算子 `aclblasCgeru`。算子完成设计、开发和测试后，目标合入 `ops-blas` 开源仓。

对标接口为 cuBLAS `cublasCgeru`，语义参考 Netlib BLAS `cgeru`：

```text
A = alpha * x * y^T + A
```

其中 `alpha` 为 complex64 复数标量，`x` 为长度 `m` 的复数向量，`y` 为长度 `n` 的复数向量，`A` 为 `m x n` 复数矩阵，矩阵按 Column-Major 列主序存储。

## 背景介绍

### aclblasCgeru 算子功能

`ger` 是 BLAS Level-2 中的秩 1 更新操作，常用于矩阵迭代更新、低秩修正和线性代数基础例程。`aclblasCgeru` 面向 complex64 类型，执行无共轭版本的复数外积更新。

`geru` 与 `gerc` 的核心区别在于 `y` 是否取共轭：

| 算子 | 计算公式 | y 处理 |
| --- | --- | --- |
| Cgeru | `A = alpha * x * y^T + A` | 不取共轭 |
| Cgerc | `A = alpha * x * y^H + A` | 取共轭 |

因此本算子的实现和 golden 比对均不得对 `y` 的虚部取反。

### aclblasCgeru 算子实现优化

ops-blas 仓当前采用句柄式 BLAS 接口，通过 `aclblasHandle_t` 绑定 stream 并直调 NPU kernel。本任务需要在 `include/cann_ops_blas.h` 中新增 `aclblasCgeru` 声明，并在 `blas/ger/` 下实现 A2/A3 架构对应的 kernel 路径。

相比通过通用 Tensor 算子组合实现，直调 BLAS kernel 可以避免临时外积矩阵和额外访存，直接在 `A` 的列主序布局上完成原地读改写。

### aclblasCgeru 算子现状分析

`aclblasCgeru` 与 cuBLAS `cublasCgeru` 参数语义一致，输入输出能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 句柄 | `aclblasHandle_t` | - | 非空 | 标量 |
| m | A 的行数 | int | int32 | `m >= 0` | 标量 |
| n | A 的列数 | int | int32 | `n >= 0` | 标量 |
| alpha | 复数标量 | pointer | complex64 | 非空；`alpha=(0,0)` 时 no-op | 标量 |
| x | 输入向量 | pointer | complex64 | 逻辑长度为 `m`，`incx != 0` | `1 + (m - 1) * abs(incx)` |
| incx | x 步长 | int | int32 | 支持正负步长，不能为 0 | 标量 |
| y | 输入向量 | pointer | complex64 | 逻辑长度为 `n`，`incy != 0`，不取共轭 | `1 + (n - 1) * abs(incy)` |
| incy | y 步长 | int | int32 | 支持正负步长，不能为 0 | 标量 |
| A | 输入输出矩阵 | pointer | complex64 | Column-Major，原地更新 | `lda x n` |
| lda | A 前导维 | int | int32 | `lda >= max(1, m)` | 标量 |

# 需求分析（required）

## 需求描述

使用 Ascend C/CATLASS 实现 `aclblasCgeru`，支持 complex64 类型、正负步长、Column-Major 矩阵访问和 no-op 快速返回。接口声明放入 `include/cann_ops_blas.h`，与同族 `aclblasCgerc` 共享 BLAS 风格声明方式，禁止定义产品私有平行接口。

函数签名如下：

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

## 需求拆解

1. 支持 complex64 输入输出，实部和虚部均为 float32。
2. 实现 `A = alpha * x * y^T + A`，保证 `y` 不取共轭。
3. 支持 `m >= 0`、`n >= 0`，其中 `m=0` 或 `n=0` 为合法 no-op。
4. 支持 `alpha=(0,0)` no-op，直接返回成功且不写 `A`。
5. 支持 `incx`、`incy` 为正步长和负步长，步长不能为 0。
6. 支持 `lda >= max(1, m)`，只更新 `A` 的前 `m x n` 有效区域。
7. Host 侧完成参数校验、tiling 计算和 kernel launch。
8. Kernel 侧完成复数乘加和 `A` 原地读改写。
9. 精度与 cblas/Netlib `cgeru` golden 对齐。
10. 性能满足任务书给出的 A2/A3 性能门限。

# 详细设计（required）

## 算子分析

### 数学公式

逻辑矩阵元素更新如下：

```text
A[i, j] = alpha * x[i] * y[j] + A[i, j]
```

考虑 Column-Major 存储和步长后，实际索引为：

```text
A[i + j * lda] = alpha * x[x_index(i)] * y[y_index(j)] + A[i + j * lda]
```

正负步长索引：

```text
x_index(i) = i * incx,                 incx > 0
x_index(i) = (m - 1 - i) * abs(incx),  incx < 0

y_index(j) = j * incy,                 incy > 0
y_index(j) = (n - 1 - j) * abs(incy),  incy < 0
```

复数乘法展开：

```text
ay_real = alpha_real * y_real - alpha_imag * y_imag
ay_imag = alpha_real * y_imag + alpha_imag * y_real

out_real = A_real + ay_real * x_real - ay_imag * x_imag
out_imag = A_imag + ay_real * x_imag + ay_imag * x_real
```

该公式为 geru 无共轭语义，`y_imag` 不取反。

### 支持数据类型

| 数据 | 类型 | 说明 |
| --- | --- | --- |
| alpha | complex64 | Host 侧复数标量 |
| x | complex64 | Device 侧输入向量 |
| y | complex64 | Device 侧输入向量 |
| A | complex64 | Device 侧输入输出矩阵 |
| m/n/incx/incy/lda | int32 | Host 侧参数 |

### 支持形状

```text
m >= 0
n >= 0
lda >= max(1, m)
incx != 0
incy != 0

x physical length = 1 + (m - 1) * abs(incx)
y physical length = 1 + (n - 1) * abs(incy)
A physical shape  = lda x n
```

## 算子实现

### 实现方案

整体采用 ops-blas 句柄式 BLAS 接口和 Ascend C kernel 直调方式实现：

| 模块 | 文件/目录 | 说明 |
| --- | --- | --- |
| API 声明 | `include/cann_ops_blas.h` | 新增 `aclblasCgeru` 声明 |
| Host 实现 | `blas/ger/arch22/` | 参数校验、tiling、kernel launch |
| Kernel 实现 | `blas/ger/arch22/` | complex64 geru 核心计算 |
| 测试 | `test/ger/cgeru/arch22/` | CSV 用例、GTest、golden 比对 |

### 3.2.1 Host 侧设计

Host 侧主要职责如下：

1. 校验 `handle` 是否为空。
2. 校验 `m`、`n` 非负。
3. 校验 `alpha` 是否为空。
4. 校验 no-op 场景：`m=0`、`n=0` 或 `alpha=(0,0)` 时直接返回成功。
5. 校验 `x`、`y`、`A` 指针非空。
6. 校验 `incx`、`incy` 不为 0。
7. 校验 `lda >= max(1, m)`。
8. 根据 `m`、`n`、`incx`、`incy`、`lda` 和硬件 core 数生成 tiling 数据。
9. 从 handle 获取 stream 并发射 kernel。

参数校验顺序建议如下：

| 步骤 | 校验项 | 非法返回 |
| --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `m < 0` 或 `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `alpha == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `m == 0` 或 `n == 0` | `ACLBLAS_STATUS_SUCCESS` |
| 5 | `alpha == (0,0)` | `ACLBLAS_STATUS_SUCCESS` |
| 6 | `x == nullptr` 或 `y == nullptr` 或 `A == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `incx == 0` 或 `incy == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `lda < max(1, m)` | `ACLBLAS_STATUS_INVALID_VALUE` |

Tiling 数据包含：

| 字段 | 说明 |
| --- | --- |
| m, n, lda | 矩阵规模与前导维 |
| incx, incy | 向量步长 |
| alphaReal, alphaImag | alpha 实部、虚部 |
| blockNum | 使用的 core 数 |
| colsPerBlock | 每个 core 处理的列数 |
| xBase, yBase | 负步长时的起始偏移 |

### 3.2.2 Kernel 侧设计

Kernel 侧以列为主要并行粒度。每个 core 处理若干列，每列先读取 `y[j]` 并计算 `alpha * y[j]`，再沿行方向遍历 `x[i]` 和 `A[i + j * lda]` 完成复数乘加。

基本流程如下：

1. 根据 block id 计算当前 core 负责的列区间。
2. 对每个列 `j`：
   - 按 `incy` 读取 `y[j]`。
   - 计算 `ay = alpha * y[j]`。
3. 对每个行 `i`：
   - 按 `incx` 读取 `x[i]`。
   - 按 Column-Major 读取 `A[i + j * lda]`。
   - 执行 complex64 乘加。
   - 写回更新后的 `A[i + j * lda]`。

由于每个输出元素 `A[i, j]` 仅被一个 core 更新，不存在跨 core 写冲突，计算结果具备确定性。

### 3.2.3 性能优化策略

1. 列并行：按 `n` 维分核，使各 core 处理相近数量的列。
2. 复用 `alpha * y[j]`：每列只计算一次 `alpha * y[j]`，减少内层重复复数乘法。
3. 连续步长优化：当 `incx=1` 时，`x` 访问连续，可在 UB 中分块缓存以减少重复 GM 读取。
4. lda padding 支持：使用 `i + j * lda` 访问矩阵，保证 padded leading dimension 下仅更新有效 `m x n` 区域。
5. no-op 快速返回：`m=0`、`n=0`、`alpha=(0,0)` 时不发射 kernel，降低无效开销。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I A2 | √ |
| Atlas 800I A3 | √ |

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| dtype | 仅支持 complex64 |
| 存储格式 | A 为 Column-Major |
| x/y 步长 | 支持正负步长，不支持 0 步长 |
| lda | 必须满足 `lda >= max(1, m)` |
| broadcast | 不涉及 |
| 反向 | 不涉及，仅 BLAS 前向计算 |
| 原地语义 | A 原地读改写 |
| 共轭 | geru 对 y 不取共轭 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 与 cblas/Netlib `cgeru` golden 对齐，实部和虚部分别按 float32 标准比对 | 任务要求 |
| 性能标准 | 在 Atlas 800I A2/A3 上，平均单次耗时不高于任务书给出的 case 门限 | 任务要求 |

complex64 比对时，实部和虚部分别按 float32 处理：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

性能验收用例包括：

| case | m | n | incx | incy |
| --- | ---: | ---: | ---: | ---: |
| 1 | 512 | 512 | 1 | 1 |
| 2 | 1024 | 1024 | 1 | 1 |
| 3 | 2048 | 2048 | 1 | 1 |
| 4 | 4096 | 4096 | 1 | 1 |

对应性能参考门限如下：

| case | 参考平均耗时（us） |
| --- | ---: |
| 1 | 5.70 |
| 2 | 10.10 |
| 3 | 55.71 |
| 4 | 219.35 |

性能测试先执行 warmup，再进行超过 50 次有效采样，取平均单次耗时。测试过程使用 handle 绑定的 stream，并在读取 Device 侧结果前完成 stream 同步，避免异步执行影响计时结果。

## 测试设计

测试覆盖以下场景：

| 编号 | 场景 | 说明 |
| --- | --- | --- |
| TC-01 | 基础小 shape | 校验 geru 无共轭复数乘加 |
| TC-02 | 方阵规模扫描 | 覆盖 512、1024、2048、4096 等性能规模 |
| TC-03 | 矩形矩阵 | 覆盖 `m != n` 场景 |
| TC-04 | 正负步长 | 覆盖 `incx/incy` 为正数和负数 |
| TC-05 | lda padding | 覆盖 `lda > m` 的列主序 padding |
| TC-06 | alpha 特殊值 | 覆盖 `(0,0)`、`(1,0)`、纯实数、纯虚数 |
| TC-07 | no-op | 覆盖 `m=0`、`n=0`、`alpha=(0,0)` |
| TC-08 | 非法参数 | 覆盖空 handle、负维度、0 步长、非法 lda、空指针 |
| TC-09 | Inf/NaN | 覆盖规格允许的特殊浮点传播行为 |

golden 使用 cblas/Netlib `cgeru` 生成，严格采用无共轭公式，分别对输出矩阵实部和虚部进行误差判断。测试用例还应检查 `lda` padding 区域不被修改、no-op 场景不写 `A`，以及负步长下从向量尾部开始反向取数。

## 兼容性分析

该算子为 ops-blas 新增接口，不改变已有 BLAS 接口行为。接口签名与 cuBLAS `cublasCgeru` 对齐，并与同族 `aclblasCgerc` 保持一致的调用风格。非法输入通过 Host 侧参数校验返回错误码，合法 no-op 场景返回成功且不写 `A`。

# 参考资料

- ops-blas 开源仓：<https://gitcode.com/cann/ops-blas>
- cuBLAS `cublasCgeru`：<https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-ger>
- Netlib BLAS `cgeru`：<https://www.netlib.org/blas/cgeru.f>
- 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>

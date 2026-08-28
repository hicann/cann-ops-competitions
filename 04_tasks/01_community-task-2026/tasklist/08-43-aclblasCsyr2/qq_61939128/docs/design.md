# aclblasCsyr2 算子设计文档

# 需求背景（required）

## 需求来源

任务要求基于 ops-blas 开源仓，在 Atlas A2/A3 系列产品上使用 Ascend C 开发单精度复数对称秩 2 更新算子 `aclblasCsyr2`，接口语义对齐 cuBLAS `cublasCsyr2`，并按社区任务要求完成算子设计、开发、测试与验收交付。

开发代码需合入 ops-blas 仓 `blas/syr2/arch22/` 目录；测试代码需合入 `test/syr2/csyr2/arch22/` 目录。`aclblasCsyr2` API 声明需新增到公共头文件 `include/cann_ops_blas.h`，与其他产品线共用同一接口，不定义产品私有平行接口。

## 背景介绍

### aclblasCsyr2 算子功能

`syr2` 为 BLAS Level 2 中的 symmetric rank-2 update。`aclblasCsyr2` 面向单精度复数 `complex64`，实现如下矩阵原地更新：

```text
A = alpha * x * y^T + alpha * y * x^T + A
```

其中：

- `alpha` 为单精度复数标量。
- `x`、`y` 为包含 `n` 个逻辑元素的单精度复数向量。
- `A` 为 `n x n` 单精度复数对称矩阵，列主序存储。
- `uplo` 指定仅引用并更新 `A` 的上三角或下三角。
- 对称矩阵满足 `A = A^T`，这里是普通转置，不是共轭转置。

本算子与 Hermitian 族算子不同：计算中不对 `x`、`y` 或 `A` 做共轭处理，对角元素虚部也没有特殊约束，应按普通复数参与乘加。

### 对标接口与参考语义

`aclblasCsyr2` 对标 cuBLAS `cublasCsyr2` 的参数语义，计算语义参考 Netlib BLAS `ssyr2`。由于 Netlib/cblas 未提供复数 `csyr2` 实现，测试 golden 需要按 `ssyr2` 的三角更新、列主序与步长语义自实现复数版本。

需要特别遵循以下行为：

- `n = 0` 为合法 quick return，返回 `ACLBLAS_STATUS_SUCCESS`，不更新 `A`。
- `alpha = (0, 0)` 为合法 quick return，返回 `ACLBLAS_STATUS_SUCCESS`，不更新 `A`。
- `incx`、`incy` 支持负步长；逻辑第 `i` 个元素按照 Netlib 规则从反向起点读取。
- 仅更新 `uplo` 指定三角，未指定三角不读不写。
- Host 侧不做 stream 同步，调用方在读取结果前自行同步。

### 现有仓库基础

ops-blas 仓中已有实数同族算子 `aclblasSsyr2`：

- API 声明位于 `include/cann_ops_blas.h`。
- 算子文档位于 `blas/syr2/README.md`。
- A2/A3 架构实现位于 `blas/syr2/arch22/`。
- 测试工程位于 `test/syr2/ssyr2/`。

`aclblasCsyr2` 应复用同族目录组织、handle/stream 直调 kernel 模式、公共状态码与参数校验风格，在 `syr2` 算子族下新增 complex64 实现。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas A2/A3 系列产品上实现 `aclblasCsyr2`，支持 single complex 类型 `aclblasComplex`，完成对称秩 2 原地更新。算子需满足 cuBLAS `cublasCsyr2` 参数语义与任务书验收要求，覆盖上三角/下三角、正负步长、leading dimension padding、quick return、非法参数返回等场景。

函数原型如下：

```cpp
aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* A, int lda);
```

## 需求拆解

1. 在 `include/cann_ops_blas.h` 新增 `aclblasCsyr2` 公共 API 声明。
2. 在 `blas/syr2/arch22/` 新增 Ascend C host 侧与 kernel 侧实现。
3. 支持 `aclblasComplex`，即实部、虚部均为 `float32` 的 interleaved complex64。
4. 实现 `A = alpha*x*y^T + alpha*y*x^T + A`，只更新 `uplo` 指定三角。
5. 保证复数 symmetric 语义：普通转置，不共轭。
6. 支持 `incx`、`incy` 为正或负，拒绝 0 步长。
7. 支持 `lda >= max(1, n)` 的列主序矩阵存储，包括 padding 场景。
8. 支持 `n = 0` 或 `alpha = (0,0)` quick return。
9. 对非法参数返回任务书约定状态码。
10. 新增 `test/syr2/csyr2/arch22/` 测试代码与 CSV 用例，覆盖精度、性能、边界与负向用例。
11. 更新 `blas/syr2/README.md`，补充 `aclblasCsyr2` 产品支持、接口说明、约束和调用示例。

# 详细设计（required）

## 算子分析

### 数学公式

对 `A` 的被引用三角区域逐元素更新：

```text
A(i, j) = A(i, j) + alpha * x(i) * y(j) + alpha * y(i) * x(j)
```

其中 `0 <= i < n`，`0 <= j < n`，列主序地址为：

```text
A(i, j) => A[j * lda + i]
```

当 `uplo = ACLBLAS_UPPER` 时，只处理 `i <= j`；当 `uplo = ACLBLAS_LOWER` 时，只处理 `i >= j`。

复数乘法按常规复数运算展开：

```text
(a + bi) * (c + di) = (ac - bd) + (ad + bc)i
```

kernel 中可预计算每行的：

```text
alpha_x_i = alpha * x(i)
alpha_y_i = alpha * y(i)
```

则单个矩阵元素更新为：

```text
A(i, j) += alpha_x_i * y(j) + alpha_y_i * x(j)
```

对角线元素自然满足：

```text
A(i, i) += 2 * alpha * x(i) * y(i)
```

该式不对复数做共轭，对角虚部按正常复数乘加产生和保留。

### 支持数据类型

| 参数 | 数据类型 | 说明 |
| --- | --- | --- |
| alpha | `aclblasComplex` | Host 侧单精度复数标量 |
| x | `aclblasComplex` | Device 侧单精度复数输入向量 |
| y | `aclblasComplex` | Device 侧单精度复数输入向量 |
| A | `aclblasComplex` | Device 侧单精度复数矩阵，原地更新 |

`aclblasComplex` 由 `include/cann_ops_blas_common.h` 定义：

```cpp
typedef struct aclblasComplex {
    float real;
    float imag;
} aclblasComplex;
```

### 支持形状与数据排布

| 对象 | 逻辑 shape | 物理存储要求 |
| --- | --- | --- |
| x | `[n]` | 至少包含 `1 + (n - 1) * abs(incx)` 个 complex64 元素 |
| y | `[n]` | 至少包含 `1 + (n - 1) * abs(incy)` 个 complex64 元素 |
| A | `[n, n]` | 列主序存储，物理大小至少为 `lda * n` 个 complex64 元素 |

逻辑向量元素到物理地址的映射为：

```text
x_index(i) = (incx > 0) ? i * incx : (n - 1 - i) * (-incx)
y_index(i) = (incy > 0) ? i * incy : (n - 1 - i) * (-incy)
```

该映射等价于 Netlib 的负步长起点规则。

### 参数与异常行为

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | dtype 类型 | 数据排布 | shape | 值域范围 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| handle | 输入 | ops-blas 句柄，携带 stream | scalar | - | - | - | 有效句柄 | `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | 输入 | 指定上/下三角 | attr | enum | - | - | `ACLBLAS_UPPER` / `ACLBLAS_LOWER` | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| n | 输入 | 矩阵阶数与向量逻辑长度 | scalar | int | - | - | `n >= 0` | `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha | 输入 | 复数缩放系数，Host 内存 | scalar | complex64 | - | - | float32 复数全集 | `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| x | 输入 | 输入向量 x，Device 内存 | tensor | complex64 | ND | `[1 + (n-1)*abs(incx)]` | float32 复数全集 | `n > 0` 且 `alpha != 0` 时为 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | 输入 | x 步长 | scalar | int | - | - | `incx != 0` | `0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| y | 输入 | 输入向量 y，Device 内存 | tensor | complex64 | ND | `[1 + (n-1)*abs(incy)]` | float32 复数全集 | `n > 0` 且 `alpha != 0` 时为 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| incy | 输入 | y 步长 | scalar | int | - | - | `incy != 0` | `0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| A | 输入/输出 | 对称矩阵 A，Device 内存，原地更新 | tensor | complex64 | ND，列主序 | `[lda, n]` | float32 复数全集 | `n > 0` 且 `alpha != 0` 时为 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| lda | 输入 | A 的 leading dimension | scalar | int | - | - | `lda >= max(1, n)` | 不满足返回 `ACLBLAS_STATUS_INVALID_VALUE` |

建议 Host 侧校验顺序如下：

1. 校验 `handle != nullptr`。
2. 校验 `uplo` 是否为 `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`。
3. 校验 `n >= 0`。
4. 校验 `alpha != nullptr`。
5. 校验 `incx != 0`、`incy != 0`。
6. 校验 `lda >= max(1, n)`。
7. 若 `n == 0` 或 `*alpha == (0,0)`，返回 `ACLBLAS_STATUS_SUCCESS`。
8. 校验 `x`、`y`、`A` 非空。
9. 计算 tiling 并启动 kernel。

该顺序保证 quick return 不掩盖明确的 Host 参数错误，同时避免 `alpha = 0` 时不必要地访问 Device 指针。

## 算子实现

### 总体实现方案

`aclblasCsyr2` 采用 Ascend C kernel 直调模式：

1. Host 侧接收 BLAS 句柄式接口参数。
2. Host 侧完成参数合法性检查、quick return 判断、core 数获取与 tiling 数据计算。
3. Host 侧将 `x`、`y`、`A` 的 device 地址和 tiling 数据传入 AIV kernel。
4. Kernel 侧按 block/thread 对矩阵三角区域进行划分，每个元素只由一个执行单元负责，完成 `A(i,j)` 读、复数乘加和写回。
5. Host 侧不做 `aclrtSynchronizeStream`，保持 ops-blas 异步执行语义。

推荐新增文件：

| 文件 | 作用 |
| --- | --- |
| `include/cann_ops_blas.h` | 新增 `aclblasCsyr2` API 声明 |
| `blas/syr2/arch22/csyr2_tiling_data.h` | 定义 `Csyr2TilingData` |
| `blas/syr2/arch22/csyr2_host.cpp` | Host 参数校验、tiling、kernel launch |
| `blas/syr2/arch22/csyr2_kernel.cpp` | Ascend C kernel 实现 |
| `blas/syr2/README.md` | 增补 `aclblasCsyr2` 文档 |
| `test/syr2/csyr2/` | 新增 complex64 测试工程与 CSV 用例 |

### Host 侧设计

#### 参数校验

Host 侧实现 `ValidateCsyr2Params`，按照任务书约束返回对应状态码：

- `handle == nullptr`：`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- `uplo` 非 `ACLBLAS_UPPER` / `ACLBLAS_LOWER`：`ACLBLAS_STATUS_INVALID_ENUM`
- `n < 0`：`ACLBLAS_STATUS_INVALID_VALUE`
- `alpha == nullptr`：`ACLBLAS_STATUS_INVALID_VALUE`
- `incx == 0` 或 `incy == 0`：`ACLBLAS_STATUS_INVALID_VALUE`
- `lda < max(1, n)`：`ACLBLAS_STATUS_INVALID_VALUE`
- quick return 之后，若需要执行 kernel，则 `x` / `y` / `A` 为 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`

#### Quick Return

满足以下任一条件时直接返回成功，不启动 kernel，不读写 `x`、`y`、`A`：

```text
n == 0
alpha.real == 0.0f && alpha.imag == 0.0f
```

#### Tiling 数据结构

Host 侧生成 `Csyr2TilingData`，建议字段如下：

```cpp
struct Csyr2TilingData {
    uint32_t numThreads;
    uint32_t rowsPerBlock;
    uint32_t n;
    uint32_t lda;
    uint32_t uplo;
    float alphaReal;
    float alphaImag;
    int64_t incx;
    int64_t incy;
};
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `numThreads` | 每个 block 的 SIMT thread 数，按 `n` 与硬件约束计算 |
| `rowsPerBlock` | 每个 block 处理的逻辑行数，`ceil(n / numBlocks)` |
| `n` | 矩阵阶数 |
| `lda` | A 的 leading dimension |
| `uplo` | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| `alphaReal` / `alphaImag` | 复数 alpha 的实部和虚部 |
| `incx` / `incy` | x/y 逻辑步长，支持负值 |

#### 分核策略

`aclblasCsyr2` 的总计算量约为 `n * (n + 1) / 2` 个 complex64 矩阵元素更新。Host 侧按行粒度将三角区域分配到多个 AIV core：

```text
aivCoreNum = GetAivCoreCount()
numBlocks = min(ceil(n / SIMT_MIN_THREAD_NUM), aivCoreNum)
rowsPerBlock = ceil(n / numBlocks)
numThreads = min(ceil_align(ceil(n / numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)
```

每个 block 处理连续行区间：

```text
rowStart = blockIdx * rowsPerBlock
rowEnd = min(rowStart + rowsPerBlock, n)
```

在每一行内根据 `uplo` 选择列范围：

```text
UPPER: colStart = row, colEnd = n
LOWER: colStart = 0,   colEnd = row + 1
```

这种划分使每个三角元素只被一个 thread 负责，避免 atomic add，并保证未指定三角不被访问。

#### Tiling Key 规划

本算子输入类型固定为 complex64，广播与 dynamic shape 分支不涉及，功能上不需要多 tiling key。可在 kernel 内根据 tiling 字段走两个执行路径：

- GM 通用路径：处理任意 `incx`、`incy`、`lda`，包括负步长与小尺寸。
- UB 优化路径：处理 `incx == 1 && incy == 1` 且向量片段可放入 UB 的性能热点场景。

该分支可由 kernel 内判断完成，无需额外公开 tiling key。

### Kernel 侧设计

#### 数据访问

Kernel 将 `GM_ADDR` 转换为 `__gm__ aclblasComplex*` 或等价的 interleaved float pair 视图。逻辑向量读取函数为：

```cpp
__aicore__ inline aclblasComplex LoadX(uint32_t i)
{
    int64_t idx = (incx > 0) ? static_cast<int64_t>(i) * incx
                             : static_cast<int64_t>(n - 1 - i) * (-incx);
    return xGm[idx];
}
```

`LoadY` 同理。矩阵元素地址为：

```cpp
uint64_t aIdx = static_cast<uint64_t>(col) * lda + row;
```

#### 复数基础运算

Kernel 内实现内联复数乘法与加法：

```cpp
Complex Mul(Complex a, Complex b)
{
    return {a.real * b.real - a.imag * b.imag,
            a.real * b.imag + a.imag * b.real};
}

Complex Add(Complex a, Complex b)
{
    return {a.real + b.real, a.imag + b.imag};
}
```

更新过程：

```text
xRow = LoadX(row)
yRow = LoadY(row)
alphaXRow = alpha * xRow
alphaYRow = alpha * yRow

for col in selected triangle columns:
    xCol = LoadX(col)
    yCol = LoadY(col)
    out = A(row, col)
    out += alphaXRow * yCol + alphaYRow * xCol
    A(row, col) = out
```

#### GM 通用路径

GM 通用路径用于完整功能覆盖：

- 支持 `incx` / `incy` 的正负步长。
- 支持任意合法 `lda`。
- 支持小尺寸与不适合 UB 缓存的大尺寸分片。
- 每个线程循环处理多行，直接从 GM 读取 `x`、`y`、`A` 并写回 `A`。

伪代码如下：

```text
for row = globalThreadId; row < n; row += gridDim * blockDim:
    xRow = load_x(row)
    yRow = load_y(row)
    alphaXRow = alpha * xRow
    alphaYRow = alpha * yRow
    [colStart, colEnd) = triangle_range(row, uplo)
    for col in [colStart, colEnd):
        xCol = load_x(col)
        yCol = load_y(col)
        A[col * lda + row] += alphaXRow * yCol + alphaYRow * xCol
```

该路径每个 `A(i,j)` 只写一次，不需要原子加。

#### UB 优化路径

性能热点用例多为 `incx = incy = 1`、`lda = n` 或带少量 padding 的连续向量场景。对这类 case，可按 block 的行区间将需要访问的 `x/y` 连续片段搬入 UB：

- `UPPER`：block 处理 `rowStart..rowEnd`，列范围可能访问 `rowStart..n-1`，因此缓存 `[rowStart, n)`。
- `LOWER`：列范围可能访问 `[0, rowEnd)`，因此缓存 `[0, rowEnd)`。

当片段长度不超过 UB 预算时，kernel 使用 UB 中的 `x/y` 进行列循环，减少 GM 重复读取。若片段超过 UB 预算，则回退 GM 通用路径。

UB 预算需同时容纳 x/y 两个 complex64 片段及必要临时变量。可按 complex 元素数配置，例如：

```text
UB_COMPLEX_CAPACITY = floor(可用 UB 字节数 / (2 * sizeof(aclblasComplex)))
```

为降低实现复杂度和避免溢出，首版可采用保守固定上限；后续根据 profiling 调整。

#### 上三角与下三角分支

Kernel 可使用模板参数或普通分支区分 `uplo`：

```text
UPPER: row <= col < n
LOWER: 0 <= col <= row
```

上三角和下三角分支均使用同一复数更新公式，不对另一三角读写。

#### 异步执行

Host 侧调用 kernel launch 后立即返回 `ACLBLAS_STATUS_SUCCESS`。算子执行绑定到 `handle->stream`，调用方负责后续 stream 同步和 Device 结果读回。

## 支持硬件

| 支持的芯片版本 | 支持情况 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 支持 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 支持 |
| Atlas 800I A2 / Atlas 800T A2（910B3） | 支持，作为任务性能测试设备 |

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 数据类型 | 仅支持 `aclblasComplex` / complex64 |
| 数据排布 | `A` 为列主序，`x/y` 为按步长访问的一维向量 |
| 三角区域 | 仅更新 `uplo` 指定三角，另一三角不读不写 |
| 复数语义 | symmetric，不共轭；不同于 Hermitian/her2 |
| n | `n >= 0`；`n = 0` quick return |
| alpha | `alpha != nullptr`；`alpha = (0,0)` quick return |
| incx/incy | 必须非 0，支持负步长 |
| lda | `lda >= max(1, n)` |
| broadcast | 不涉及，不支持广播 |
| dynamic shape | 不涉及高维 dynamic shape；`n` 为运行时标量 |
| 原地语义 | `A` 原地覆写 |
| 确定性 | 单元素单写，无 atomic 时结果稳定；浮点乘加不要求 bit-exact |
| 同步语义 | Host 不同步 stream |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | complex64 的实部、虚部分别按 float32 标准判定；逐元素满足 `abs(actual - golden) <= atol + rtol * abs(golden)`；用例需满足 matched ratio 与最大误差阈值 | 社区任务书与生态算子开源精度标准 |
| 性能标准 | 性能 case 平均单次耗时不高于任务书标杆；需 warmup 后有效采样超过 50 次取平均 | 社区任务书与测试用例目录 |

精度阈值：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64（实部/虚部分别按 FLOAT32） | `2^-10`，约 `9.77e-4` | `2^-16`，约 `1.53e-5` | `0.99` | `1e-2` 或 `32 * ULP` |

任务书性能典型 case：

| case | uplo | n | alpha | incx | incy | lda | 标杆耗时 Avg time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | UPPER | 512 | `(1,0)` | 1 | 1 | 512 | 4.82 us |
| 2 | LOWER | 1024 | `(1,0)` | 1 | 1 | 1024 | 6.22 us |
| 3 | UPPER | 2048 | `(1,0)` | 1 | 1 | 2048 | 12.19 us |

随附 `gpu_baseline.csv` 还包含 `n=4096` 以及更多尺寸扫描性能参考，用于自测阶段采集与对比。

## 测试设计

测试工程按 ops-blas CSV 驱动 GTest 方式组织，建议新增：

```text
test/syr2/csyr2/
├── CMakeLists.txt
├── csyr2_param.h
├── csyr2_golden.h
└── arch22/
    ├── csyr2_test.cpp
    ├── csyr2_npu_wrapper.h
    └── csyr2_test.csv
```

golden 参考实现按 Netlib `ssyr2` 语义实现复数版本：

- 列主序。
- 只验证 `uplo` 指定三角。
- 不共轭。
- 支持负步长。
- `n = 0` 或 `alpha = (0,0)` 不更新。

随任务提供的 `syr2_test.csv` 共 1200 条用例，分类如下：

| 类别 | 前缀 | 条数 | 覆盖内容 |
| --- | --- | --- | --- |
| L0 基础 | `TC_L0` | 4 | 上/下三角，小尺寸基础功能 |
| L1 尺寸 | `TC_SQ` | 46 | 1 到 2048 的尺寸扫描，含奇数、边界和非对齐值 |
| L2 标量 | `TC_AB` | 16 | `alpha=(1,0)`、`(0,0)`、纯虚、负值、大值等 |
| L4 leading dimension | `TC_LD` | 6 | `lda = n + 4` padding |
| L4 步长 | `TC_INC` | 36 | `incx/incy` 覆盖 `±1/±2/±3` |
| L5 填充 | `TC_FL` | 18 | 随机、全零、交替、极端值、Inf、NaN |
| L5b 覆盖 | `TC_CV` | 16 | 中等尺寸组合覆盖 |
| L6 边界 | `TC_ED` | 13 | quick return、空指针、非法枚举、非法 lda、负 n、零步长 |
| EX 扩展 | `TC_EX` | 845 | 尺寸、uplo、alpha、步长、padding 的确定性采样 |
| PF 性能 | `TC_PF` | 200 | 连续访存性能用例与任务书典型 case |

## 自验方法

精度自验：

```bash
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --csv ./syr2_test.csv
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --skip-build --device 1
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --filter TC_L0 --timeout 3600
```

性能自验：

```bash
python verify_performance.py --repo /path/to/ops-blas --soc ascend910b3 --timeout 3600
python verify_performance.py --repo /path/to/ops-blas --soc ascend910b3 --skip-build --device 1
```

性能测试需先 warmup，再有效采样超过 50 次取平均。读回 Device 结果前由测试框架同步 stream。

## 兼容性分析

`aclblasCsyr2` 为新增 API，不改变已有 `aclblasSsyr2` 行为。新增声明位于公共头文件 `include/cann_ops_blas.h`，参数序列与 cuBLAS `cublasCsyr2` 对齐，便于用户从 cuBLAS 迁移。

本算子与 `aclblasSsyr2` 同属 `syr2` 算子族：

- 共享 `aclblasHandle_t`、`aclblasFillMode_t`、`aclblasStatus_t` 等公共类型。
- 共享 ops-blas 异步 stream 执行模型。
- README 与测试目录按现有 BLAS 算子组织方式扩展。

## 风险与规避

| 风险 | 影响 | 规避方案 |
| --- | --- | --- |
| 将 complex symmetric 误写为 Hermitian 语义 | 数值错误，尤其虚部与对角线错误 | 明确不做共轭；golden 覆盖纯虚 alpha、复数 x/y、对角虚部 |
| 负步长起点处理错误 | `incx/incy < 0` 用例失败 | 使用 Netlib 等价映射 `(n - 1 - i) * (-inc)` |
| `alpha = 0` quick return 仍访问空指针 | 负向/边界用例失败 | quick return 前只校验 Host 参数，quick return 后再校验 Device 指针 |
| `uplo` 非法返回码不一致 | 接口行为不符合任务书 | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 只按 `lda = n` 实现 | padding 用例失败 | 矩阵地址统一使用 `col * lda + row` |
| 使用 atomic 分两路累加导致非确定或性能波动 | 性能与可测性风险 | 单线程一次计算两个外积项并写回目标元素 |
| UB 优化片段越界 | 精度或稳定性问题 | 超过 UB 预算自动回退 GM 通用路径 |

## 交付件

| 序号 | 交付件 | 内容 |
| --- | --- | --- |
| 1 | 算子设计文档 | 本文档，按社区模板说明需求、设计、约束、测试和验收 |
| 2 | 算子实现代码 | `include/cann_ops_blas.h` 声明，`blas/syr2/arch22/` host/kernel/tiling 实现 |
| 3 | README 文档 | `blas/syr2/README.md` 新增 `aclblasCsyr2` 产品支持、接口说明、调用示例 |
| 4 | 测试代码与用例 | `test/syr2/csyr2/arch22/` GTest、golden、CSV 用例 |
| 5 | 自测报告 | 精度、性能、内存及截图，覆盖任务书要求 |


# aclblasCtrsm 算子设计文档

# 需求背景（required）

## 需求来源

本任务来自 CANN 社区 2026 年 8 月 BLAS 算子开发任务，要求在 `ops-blas` 仓中实现单精度复数三角矩阵方程求解算子 `aclblasCtrsm`。任务适配硬件为 Ascend 950PR，CANN 版本为 9.1.0，算子实现目录为 `blas/trsm/arch35/`。

`aclblasCtrsm` 对标 cuBLAS `cublasCtrsm` 与 Netlib BLAS `ctrsm`，用于求解多右端三角线性系统。输入矩阵均采用列主序存储，输出结果原地覆写输入右端矩阵 `B`。

## 背景介绍

### aclblasCtrsm 算子实现优化

TRSM 是 BLAS Level 3 中的三角矩阵求解接口。复数版本 `ctrsm` 的输入输出为 complex64，其中每个元素由 float32 实部和 float32 虚部组成。与 GEMM 类算子不同，TRSM 在三角维度上存在前代或回代依赖，计算顺序由 `side`、`uplo`、`trans`、`diag` 共同决定。

本任务要求新增 `aclblasCtrsm` 句柄式 BLAS API，通过 `aclblasHandle_t` 绑定 stream，Host 侧完成参数合法性校验和 tiling 参数生成，Device 侧使用 Ascend C kernel 直调完成复数三角求解。接口声明需放入 `include/cann_ops_blas.h`，与其他产品线共用同一 API，禁止定义 Ascend 950PR 私有平行接口。

### cuBLAS/Netlib 实现现状分析

对标接口语义如下：

```cpp
aclblasStatus_t aclblasCtrsm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    aclblasComplex* B, int ldb);
```

输入输出规格如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 上下文句柄 | scalar | aclblasHandle_t | 不可为空，携带 stream | - |
| side | A 位于方程左侧或右侧 | attr | aclblasSideMode_t | LEFT 或 RIGHT | - |
| uplo | A 使用上三角或下三角 | attr | aclblasFillMode_t | UPPER 或 LOWER | - |
| trans | op(A) 类型 | attr | aclblasOperation_t | N、T、C | - |
| diag | A 对角类型 | attr | aclblasDiagType_t | NON_UNIT 或 UNIT | - |
| m | B 行数 | scalar | int | `m >= 0` | - |
| n | B 列数 | scalar | int | `n >= 0` | - |
| alpha | 复数标量乘子 | scalar pointer | aclblasComplex | Host 指针，不可为空 | - |
| A | 三角矩阵 | tensor pointer | aclblasComplex | Device 指针，alpha 非零时不可为空 | LEFT: `lda x m`; RIGHT: `lda x n` |
| lda | A 前导维 | scalar | int | LEFT 时 `lda >= max(1,m)`；RIGHT 时 `lda >= max(1,n)` | - |
| B | 右端矩阵和输出解 | tensor pointer | aclblasComplex | Device 指针，非零规模时不可为空 | `ldb x n` |
| ldb | B 前导维 | scalar | int | `ldb >= max(1,m)` | - |

### aclblasCtrsm 算子功能分析

算子求解如下方程：

1. `side = ACLBLAS_SIDE_LEFT` 时：

```text
op(A) * X = alpha * B
```

2. `side = ACLBLAS_SIDE_RIGHT` 时：

```text
X * op(A) = alpha * B
```

其中 `op(A)` 由 `trans` 决定：

| trans | op(A) |
| --- | --- |
| `ACLBLAS_OP_N` | `A` |
| `ACLBLAS_OP_T` | `A^T` |
| `ACLBLAS_OP_C` | `A^H`，即共轭转置 |

`A` 仅引用 `uplo` 指定的三角部分；`diag = ACLBLAS_UNIT` 时 A 的对角元素不读取，计算时按 1 处理；`diag = ACLBLAS_NON_UNIT` 时从 A 读取对角元素，但本算子不做奇异性或近奇异性检测。

# 需求分析（required）

## 需求描述

在 `ops-blas` 仓中新增 `aclblasCtrsm`，实现 Ascend 950PR 上 complex64 三角矩阵求解能力。函数行为需与 cuBLAS `cublasCtrsm` 和 Netlib `ctrsm` 核心语义保持一致，满足任务书给出的精度、性能、异常处理和自验要求。

## 需求拆解

1. 新增 `include/cann_ops_blas.h` 中的 `aclblasCtrsm` API 声明。
2. 支持 `aclblasComplex` complex64 输入输出，实部和虚部均为 float32。
3. 支持 `side = LEFT / RIGHT` 两类方程。
4. 支持 `uplo = UPPER / LOWER` 两种三角存储。
5. 支持 `trans = N / T / C`，其中 `C` 需要处理复数共轭。
6. 支持 `diag = NON_UNIT / UNIT`，UNIT 场景不读取 A 对角元素。
7. 支持列主序存储和 `lda` / `ldb` 前导维 padding。
8. 支持 `m = 0` 或 `n = 0` 合法 no-op，直接返回成功。
9. 支持 `alpha = (0,0)` 快速路径：A 不被引用，B 原地置零后返回。
10. 非法枚举、负维度、非法前导维、空指针等场景返回任务书规定的状态码。
11. 通过 CSV 驱动的 C++ GTest 完成精度、边界、负向和性能测试。

# 详细设计（required）

## 算子分析

### 数学公式

令 `A` 为三角矩阵，`B` 为输入右端矩阵，`X` 为输出解矩阵，`alpha` 为复数标量。

LEFT 场景：

```text
op(A) * X = alpha * B
```

按列独立求解，每一列 `j` 满足：

```text
op(A) * X[:, j] = alpha * B[:, j]
```

RIGHT 场景：

```text
X * op(A) = alpha * B
```

按行独立求解，每一行 `i` 满足：

```text
X[i, :] * op(A) = alpha * B[i, :]
```

复数乘法、除法定义如下：

```text
(a + bi) * (c + di) = (ac - bd) + (ad + bc)i
(a + bi) / (c + di) = ((a + bi) * (c - di)) / (c^2 + d^2)
```

当 `trans = ACLBLAS_OP_C` 时，读取 `A` 后需对参与计算的元素取共轭：

```text
conj(a + bi) = a - bi
```

### 支持数据类型

| 数据 | 类型 | 路径 |
| --- | --- | --- |
| alpha | aclblasComplex | Host 输入 |
| A | aclblasComplex | Device 输入，只读 |
| B | aclblasComplex | Device 输入/输出，原地覆写 |
| m/n/lda/ldb | int | Host 输入 |
| side/uplo/trans/diag | enum | Host 输入 |

### 支持形状

```text
side = LEFT:
    A: lda x m, 有效三角矩阵阶数为 m
    B: ldb x n, 有效矩阵为 m x n

side = RIGHT:
    A: lda x n, 有效三角矩阵阶数为 n
    B: ldb x n, 有效矩阵为 m x n
```

其中 `m >= 0`，`n >= 0`。列主序索引为：

```text
A(row, col) = A[row + col * lda]
B(row, col) = B[row + col * ldb]
```

## 算子实现

### 实现方案

计划实现按 `ops-blas` 仓同族 TRSM 算子组织，新增或修改文件如下：

| 模块 | 新增/修改 | 说明 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | 修改 | 新增 `aclblasCtrsm` 对外 API 声明 |
| `blas/trsm/arch35/ctrsm.cpp` | 新增 | Host 侧入口、参数校验、tiling 生成、kernel launch |
| `blas/trsm/arch35/ctrsm_kernel.cpp` | 新增 | Ascend C complex64 三角求解 kernel |
| `blas/trsm/arch35/ctrsm_tiling.h` | 新增 | tiling 参数结构体定义 |
| `blas/trsm/README.md` | 修改 | 补充 `aclblasCtrsm` 接口说明和支持表 |
| `test/trsm/ctrsm/arch35/` | 新增 | CSV 用例、GTest 测试代码、golden 比对和 README |

Host 侧根据 `side/uplo/trans/diag/m/n/lda/ldb/alpha` 生成 tiling 信息，将 stream 从 `handle` 中取出后下发 Ascend C kernel。Device 侧直接读取列主序 A/B，按三角求解依赖顺序原地更新 B。

### 3.2.1 host 侧设计

Host 侧完成以下校验和分发：

1. `handle == nullptr` 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `side` 不属于 `ACLBLAS_SIDE_LEFT` / `ACLBLAS_SIDE_RIGHT` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `uplo` 不属于 `ACLBLAS_UPPER` / `ACLBLAS_LOWER` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `trans` 不属于 `ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. `diag` 不属于 `ACLBLAS_NON_UNIT` / `ACLBLAS_UNIT` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. `m < 0` 或 `n < 0` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
7. `alpha == nullptr` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
8. 前导维校验：
   - LEFT: `lda >= max(1, m)`；
   - RIGHT: `lda >= max(1, n)`；
   - `ldb >= max(1, m)`。
9. `m == 0 || n == 0` 时直接返回 `ACLBLAS_STATUS_SUCCESS`，不下发 kernel。
10. `B == nullptr` 且 `m > 0 && n > 0` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
11. `alpha != (0,0)` 且 `A == nullptr` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
12. `alpha == (0,0)` 时不引用 A，走 B 置零 kernel 或设备 memset 路径后返回成功。

参数校验顺序固定为：先检查 `handle`、枚举值、`m/n`、`alpha` 和前导维，再处理零维 quick return，随后检查非零规模下的 `B` 指针，最后根据 alpha 是否为零决定是否检查 `A` 指针。这样可以保证 `alpha=(0,0)` 时不引用 A，同时不放宽其他非法参数。

Tiling 参数包括：

| 字段 | 说明 |
| --- | --- |
| `m` / `n` | B 有效矩阵尺寸 |
| `lda` / `ldb` | A/B 前导维 |
| `side` / `uplo` / `trans` / `diag` | 计算分支枚举 |
| `alpha_real` / `alpha_imag` | Host 侧读取后的复数标量 |
| `a_order` | A 有效阶数，LEFT 为 m，RIGHT 为 n |
| `total_elements` | B 有效元素数，用于 alpha 为零置零路径 |

### 3.2.2 kernel 侧设计

Kernel 按 `side` 分为 LEFT 和 RIGHT 两类路径。

LEFT 路径以 B 的列为独立求解单元。每一列内部沿三角维度存在依赖，需按前代/回代顺序串行推进；不同列之间可以分核或分 block 并行。

RIGHT 路径以 B 的行为独立求解单元。每一行内部沿三角维度存在依赖；不同 B 行之间可以并行。由于 B 为列主序，RIGHT 路径按行访问时跨列步长为 `ldb`，需要通过分块缓存 B 行片段降低非连续访存开销。

三角遍历方向由实际参与求解的 `op(A)` 决定：

| side | trans | uplo | 求解方向 |
| --- | --- | --- | --- |
| LEFT | N | UPPER | `k = m-1 -> 0` 回代 |
| LEFT | N | LOWER | `k = 0 -> m-1` 前代 |
| LEFT | T/C | UPPER | `k = 0 -> m-1` 前代 |
| LEFT | T/C | LOWER | `k = m-1 -> 0` 回代 |
| RIGHT | N | UPPER | `k = 0 -> n-1` 前代式更新 |
| RIGHT | N | LOWER | `k = n-1 -> 0` 回代式更新 |
| RIGHT | T/C | UPPER | `k = n-1 -> 0` 回代式更新 |
| RIGHT | T/C | LOWER | `k = 0 -> n-1` 前代式更新 |

核心计算流程：

1. 若 `alpha != (1,0)`，先对 B 有效区域执行复数标量乘：`B = alpha * B`。
2. 对每个独立 RHS 向量按三角方向遍历主维 `k`。
3. `diag = NON_UNIT` 时读取有效对角并执行复数除法；`diag = UNIT` 时跳过对角读取和除法。
4. 使用当前解元素更新尚未求解的元素。
5. `trans = C` 时所有从 A 读取的参与计算元素均取共轭。
6. 只读取 `uplo` 指定的三角区域，不访问另一半矩阵。

### 3.2.3 分核与并行策略

TRSM 在三角维度上有严格数据依赖，因此主求解维度不做破坏语义的并行化。并行粒度选择如下：

| 场景 | 并行粒度 | 说明 |
| --- | --- | --- |
| LEFT | B 的列 | 每列独立，列内按 k 顺序求解 |
| RIGHT | B 的行或行分组 | 每行独立，行内按 k 顺序求解 |
| alpha 为零 | B 有效元素 | 直接并行置零 |

性能重点优化方向：

1. 对 LEFT 大矩阵场景，按列分配多个 RHS，保证列内连续访存。
2. 对 RIGHT 场景，使用 UB 暂存 B 行片段，减少列主序跨步访问造成的重复 GM 读写。
3. 对 `diag = UNIT` 路径跳过对角读取和复数除法。
4. 对 `alpha = (1,0)` 路径跳过初始标量乘。
5. 对 `trans = N/T/C` 编译期或运行期分支拆分，减少内层循环判断。

### 3.2.4 复数基础运算

Device 侧统一使用 complex64 的实部/虚部 float32 计算：

```text
complex_mul(x, y):
    real = x.real * y.real - x.imag * y.imag
    imag = x.real * y.imag + x.imag * y.real

complex_div(x, y):
    denom = y.real * y.real + y.imag * y.imag
    real = (x.real * y.real + x.imag * y.imag) / denom
    imag = (x.imag * y.real - x.real * y.imag) / denom
```

本算子不检测 `denom == 0`。当 `diag = NON_UNIT` 且 A 对角为零或近零时，结果遵循浮点计算自然行为，输入合法性由调用方和测试数据生成策略保证。

### 3.2.5 alpha 特殊路径

`alpha = (0,0)` 时遵循 Netlib `ctrsm` 语义：

1. A 不被引用，允许 `A == nullptr`。
2. B 有效区域全部置为 `(0,0)`。
3. 不执行三角求解。

该路径仍要求 `B` 在非零规模下不可为空，且 `lda/ldb`、枚举、维度等参数先完成合法性校验。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 数据类型 | 仅支持 complex64，即 `aclblasComplex` |
| 存储格式 | A/B 均为列主序 ND 存储 |
| 前导维 | 支持 `lda` / `ldb` padding，不支持超出 BLAS 前导维语义的任意非连续视图 |
| side | 支持 LEFT、RIGHT |
| uplo | 支持 UPPER、LOWER，仅引用指定三角 |
| trans | 支持 N、T、C，C 路径执行共轭转置 |
| diag | 支持 NON_UNIT、UNIT；UNIT 不读取 A 对角 |
| 零维 | `m = 0` 或 `n = 0` 合法 no-op |
| alpha 为零 | A 不被引用，B 原地置零 |
| 奇异性检测 | 不支持，与 cuBLAS/Netlib 一致 |
| 原地语义 | B 输入右端矩阵，输出时被解 X 覆写 |
| 异步执行 | 通过 handle 绑定 stream，下发后由调用方按需同步 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas/Netlib `ctrsm` 生成，输出 B 的实部和虚部分别按 FLOAT32 标准比对 | 任务书、生态算子开源精度标准 |
| 性能标准 | Ascend 950PR 平均单次耗时不高于任务书和 `gpu_baseline.csv` 给出的标杆阈值 | 任务书、测试用例目录 |

精度判定按 complex64 的实部、虚部分别执行：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64 分量 FLOAT32 | `2^-10` | `2^-16` | `0.99` | `1e-2` 或 `32 * ULP` |

逐元素通过条件：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

用例整体同时满足 matched_ratio 和 max_abs_error_limit 时判定通过。

任务书给出的典型性能 case 如下：

| case | m | n | side | uplo | trans | diag | 标杆耗时 Avg time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 256 | LEFT | UPPER | N | NON_UNIT | 42.57 us |
| 2 | 512 | 512 | LEFT | LOWER | N | NON_UNIT | 89.38 us |
| 3 | 1024 | 1024 | RIGHT | UPPER | T | NON_UNIT | 238.19 us |

测试包 `gpu_baseline.csv` 还补充了 200 条性能基线，其中前 5 条为典型 case：

| id | m | n | side | uplo | trans | diag | gpu_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ctrsm-base-000 | 256 | 256 | LEFT | UPPER | N | NON_UNIT | 0.106424 |
| ctrsm-base-001 | 512 | 512 | LEFT | LOWER | N | NON_UNIT | 0.223440 |
| ctrsm-base-002 | 1024 | 1024 | RIGHT | UPPER | T | NON_UNIT | 0.595487 |
| ctrsm-base-003 | 2048 | 2048 | LEFT | LOWER | C | NON_UNIT | 1.919005 |
| ctrsm-base-004 | 2048 | 2048 | RIGHT | LOWER | N | UNIT | 1.852380 |

其中任务书三条性能 case 为验收强约束，单位为 us；`gpu_baseline.csv` 为随测试包提供的扩展性能参考，单位为 ms。两组数据采集环境和统计口径不同，文档中分别列出，正式验收以任务书和评审侧测试口径为准。

## 测试设计

测试使用 `ops-blas` 仓 C++ GTest 框架，通过 CSV 描述参数并生成输入数据。精度 golden 由 cblas/Netlib BLAS 复数 `ctrsm` 生成，不导入被测实现，避免自比较。

测试目录设计如下：

| 文件 | 说明 |
| --- | --- |
| `test/trsm/ctrsm/arch35/ctrsm_test.csv` | CSV 驱动用例，覆盖精度、边界、负向和性能场景 |
| `test/trsm/ctrsm/arch35/ctrsm_test.cpp` | GTest 主测试入口，调用 `aclblasCtrsm` |
| `test/trsm/ctrsm/arch35/ctrsm_golden.h` | cblas/Netlib golden 封装 |
| `test/trsm/ctrsm/arch35/ctrsm_param.h` | CSV 参数解析和输入生成 |
| `test/trsm/ctrsm/arch35/README.md` | 测试步骤、CSV 字段、复现方法说明 |

随任务提供的 `ctrsm_test.csv` 共 1200 条用例，分类如下：

| 类别 | 前缀 | 条数 | 覆盖内容 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 24 | `side/uplo/trans/diag` 24 组枚举全组合，小尺寸 8x8 |
| L1 尺寸 | TC_SQ | 23 | 1 到 2048 的尺寸扫描，含奇数、边界和非对齐值 |
| L2 标量 | TC_AB | 24 | alpha 特殊值，含 `(0,0)`、`(1,0)`、`(-1,0)`、纯虚数和大值 |
| L3 非方阵 | TC_WS/TC_TH | 24 | 宽矩阵、窄矩阵，覆盖 LEFT/RIGHT 不同 A 阶数 |
| L4 前导维 | TC_LD | 12 | `lda/ldb` 最小约束和 padding |
| L5 填充 | TC_FL | 11 | A/B 多种填充，含 Inf/NaN 特殊值 |
| L5b 覆盖 | TC_CV | 24 | 中等尺寸枚举组合补充 |
| L6 边界 | TC_ED | 18 | 零维、空指针、非法枚举、非法前导维、负维度 |
| EX 扩展 | TC_EX | 838 | 扩展精度组合 |
| PF 性能 | TC_PF | 200 | 典型性能 case、小尺寸、规模扫描、宽/窄网格 |

输入生成规则遵循任务书要求：`side/uplo/trans/diag` 覆盖 24 组枚举组合；`m/n` 覆盖 0、1、小质数、2 的幂、2 的幂附近值、非对齐值和大规模值；`alpha` 覆盖 `(0,0)`、`(1,0)`、`(-1,0)`、纯虚数、大值以及随机均匀/正态分布；A 仅填充 `uplo` 指定三角，`diag=NON_UNIT` 时对角加符号保持偏移 `boost=max(5,A阶数)` 以避免病态系统；B 覆盖随机、全零、交替、极端值、Inf 和 NaN；`lda/ldb` 覆盖最小合法前导维和 padding 场景。性能测试执行 warmup 后有效采样超过 50 次，取平均耗时。

负向和边界用例期望如下：

| 场景 | 期望返回值 |
| --- | --- |
| `alpha == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `A == nullptr` 且 alpha 非零 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `B == nullptr` 且非零规模 | `ACLBLAS_STATUS_INVALID_VALUE` |
| side/uplo/trans/diag 非法枚举 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` 不满足约束 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldb` 不满足约束 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `m < 0` 或 `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `m = 0` 或 `n = 0` | `ACLBLAS_STATUS_SUCCESS` |
| `alpha = (0,0)` 且 A 为空 | `ACLBLAS_STATUS_SUCCESS` |

## 兼容性分析

该算子为 `ops-blas` 新增 API，不改变已有 `aclblasStrsm` 或其他 BLAS 接口行为。新增声明放入公共头文件 `include/cann_ops_blas.h`，状态码沿用 `include/cann_ops_blas_common.h` 中既有定义。

非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`，与 cuBLAS `CUBLAS_STATUS_INVALID_VALUE` 口径以及仓内同族 `strsm` 测试期望保持一致。

## 风险分析

| 风险项 | 影响 | 应对策略 |
| --- | --- | --- |
| 三角维度串行依赖 | 主求解维无法完全并行，可能影响大阶数性能 | LEFT 按列并行、RIGHT 按行或行分组并行，并对 alpha 特殊路径和 UNIT 对角路径做分支优化 |
| RIGHT 路径列主序跨步访问 | 按行求解时 B 访问不连续，可能造成 GM 访存效率下降 | 使用 UB 暂存行片段，按 tile 组织读写，减少重复跨步访问 |
| 复数除法数值放大 | NON_UNIT 且对角接近零时误差可能扩大 | 与 cuBLAS/Netlib 保持一致，不做奇异性检测；测试生成时通过对角 boost 保证对角远离零 |
| trans=C 共轭处理遗漏 | 共轭转置路径结果错误 | 将 OP_T 和 OP_C 分支显式区分，内层读取 A 后统一经过共轭开关处理 |
| alpha 为零路径误引用 A | `A==nullptr` 的合法场景误报或异常访问 | Host 侧固定校验顺序，alpha 为零时跳过 A 指针检查和所有 A 访问 |
| 前导维 padding 覆盖不足 | 非紧凑列主序场景可能出现索引错误 | 精度用例覆盖 `lda/ldb` 最小合法值和多个 padding 组合 |

# 参考资料

- `aclblasCtrsm_Atlas950PR_task_doc.md`
- `test_cases/README.md`
- `test_cases/ctrsm_test.csv`
- `test_cases/gpu_baseline.csv`
- ops-blas 仓库：<https://gitcode.com/cann/ops-blas>
- cann-competitions 设计文档模板：<https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>
- Netlib BLAS `ctrsm`：<https://www.netlib.org/blas/ctrsm.f>
- cuBLAS `cublasCtrsm`：<https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsm>
- 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>

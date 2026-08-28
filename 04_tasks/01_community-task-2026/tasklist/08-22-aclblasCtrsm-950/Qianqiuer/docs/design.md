# 需求背景（required）

## 需求来源

本需求来源于 2026 年 8 月 CANN 社区任务 `aclblasCtrsm` 算子开发，目标仓库为
`cann/ops-blas`，目标硬件为昇腾 950PR，目标软件版本为 CANN 9.1.0。

任务要求在 `ops-blas` 现有句柄、流、工作空间和构建体系内新增单精度复数三角矩阵求解接口
`aclblasCtrsm`，并在 `blas/trsm/arch35/` 下实现 Ascend C Host 与 Kernel 代码。本文档以任务书、
社区设计文档模板以及 `ops-blas` 当前源码为依据，描述接口语义、实现方案、测试方案和验收标准。

本文档设计基线如下：

| 项目 | 基线 |
| --- | --- |
| 目标仓库 | `cann/ops-blas` |
| 源码基线 | `003629ee096691d32a95d73e9e218c0c6976152e` |
| 目标产品 | 昇腾 950PR（arch35） |
| CANN 版本 | 9.1.0 |
| 算子接口 | `aclblasCtrsm` |
| 数据类型 | `aclblasComplex`，即两个 FP32 组成的 complex64 |
| 数据布局 | Column Major |
| 计算方式 | Ascend C Kernel，异步提交到 handle 当前 stream |

## 背景介绍

TRSM（Triangular Solve Matrix）用于求解一侧系数矩阵为三角矩阵的线性方程组，是分解、求逆、
线性系统求解及高性能科学计算中的基础 BLAS Level 3 操作。复数版本 CTRSM 广泛用于频域计算、
信号处理、量子计算和复数线性代数。

`ops-blas` 已提供 `aclblasStrsm` 的 arch35 实现，具备小规模 SIMT 求解、较大规模分块求解、
handle stream、用户/默认工作空间等基础设施，但尚未提供对外的 `aclblasCtrsm` 接口。复数版本不能
简单复用实数版本：除复数乘除外，`ACLBLAS_OP_C` 还要求共轭转置；大规模尾部更新也需要复数
GEMM 分解。因此本需求在沿用现有 TRSM 分层结构的同时，新增独立的复数实现，避免影响已有
`aclblasStrsm` 行为。

# 需求分析（required）

## 需求描述

新增以下公开接口：

```cpp
aclblasStatus_t aclblasCtrsm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A,
    int lda,
    aclblasComplex* B,
    int ldb);
```

函数名之外的参数顺序与 `cublasCtrsm` 一一对应，整数维度使用仓内 `aclblasStrsm` 一致的 `int`，
测试和调用方无需额外参数映射。

参数定义与任务书保持一致：

| 参数 | 内存位置 | 属性 | 类型/合法值 | Shape/约束 | 异常行为 |
| --- | --- | --- | --- | --- | --- |
| `handle` | Host | 输入 | 已创建的 `aclblasHandle_t` | 绑定当前执行 stream | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `side` | Host | 属性 | `ACLBLAS_SIDE_LEFT(141)`、`ACLBLAS_SIDE_RIGHT(142)` | 决定 A 位于 X 左侧或右侧 | 其他值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `uplo` | Host | 属性 | `ACLBLAS_UPPER(121)`、`ACLBLAS_LOWER(122)` | 只引用指定三角 | 其他值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `trans` | Host | 属性 | `ACLBLAS_OP_N(111)`、`ACLBLAS_OP_T(112)`、`ACLBLAS_OP_C(113)` | 分别表示 A、A^T、A^H | 其他值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `diag` | Host | 属性 | `ACLBLAS_NON_UNIT(131)`、`ACLBLAS_UNIT(132)` | UNIT 时对角按 1 且不得读取 | 其他值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `m` | Host | 输入 | `int`，`m >= 0` | B 的行数；LEFT 时为 A 的阶数 | 负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n` | Host | 输入 | `int`，`n >= 0` | B 的列数；RIGHT 时为 A 的阶数 | 负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha` | Host | 输入 | `const aclblasComplex *` | 单个 complex64；实虚部为 FP32 | 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | Device | 输入、只读 | `const aclblasComplex *` | LEFT 为 `m x m`，RIGHT 为 `n x n`，Column Major | 非空计算且 alpha 非零时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | Host | 输入 | `int` | `lda >= max(1, side == LEFT ? m : n)` | 不满足约束返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `B` | Device | 输入/输出 | `aclblasComplex *` | `m x n`，Column Major，原地覆盖 | `m,n > 0` 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldb` | Host | 输入 | `int` | `ldb >= max(1, m)` | 不满足约束返回 `ACLBLAS_STATUS_INVALID_VALUE` |

接口按 Column Major 解释输入矩阵，并原地覆盖 `B`：

- 当 `side == ACLBLAS_SIDE_LEFT` 时，求解
  `op(A) * X = alpha * B`，其中 `A` 的逻辑形状为 `m x m`；
- 当 `side == ACLBLAS_SIDE_RIGHT` 时，求解
  `X * op(A) = alpha * B`，其中 `A` 的逻辑形状为 `n x n`；
- `B` 的逻辑形状始终为 `m x n`，返回后其逻辑区域保存 `X`；
- `op(A)` 根据 `trans` 分别为 `A`、`A^T` 或 `A^H`；
- `uplo` 指定只读取 `A` 的上三角或下三角；
- 当 `diag == ACLBLAS_UNIT` 时，对角线按 1 处理，不读取 `A` 的对角元素；
- `alpha` 为 Host 侧单精度复数标量；
- 接口为异步接口，计算任务提交到 `handle` 当前 stream，正常路径不主动同步；调用方在 Host 读回
  `B` 或跨 stream 使用结果前，必须同步该 stream 或建立正确的事件依赖。

本任务只实现单个矩阵的前向计算，不支持 batched、broadcast、反向传播或奇异性检测。矩阵通过
`lda`、`ldb` 表示列间跨度，不额外支持任意 stride 的非连续矩阵。

## 需求拆解

### 接口与语义

1. 在 `include/cann_ops_blas.h` 中增加 `aclblasCtrsm` 声明，复用
   `include/cann_ops_blas_common.h` 中已有的 `aclblasComplex`、枚举和状态码。
2. 不新增私有的 950PR 并行公开接口，不改变 `aclblasStrsm` 既有签名和行为。
3. 支持以下 24 种枚举组合：
   `side(LEFT/RIGHT) x uplo(UPPER/LOWER) x trans(N/T/C) x diag(UNIT/NON_UNIT)`。
4. `m`、`n`、`lda`、`ldb` 是运行时标量参数，不声明额外的 Tensor dynamic shape 能力；实现正确
   处理尺寸为 0、带 padding 及非方形 `B`。

### 参数校验与特殊值

参数检查按以下顺序执行，以避免无效参数或空指针被提前解引用：

1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `side/uplo/trans/diag` 不在合法枚举集合、`m < 0` 或 `n < 0` 时返回
   `ACLBLAS_STATUS_INVALID_VALUE`；
3. `alpha == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. `m == 0 || n == 0` 时直接返回成功，不读取 `A/B`，不检查 leading dimension，不申请工作空间，
   也不下发 Kernel；
5. 令 `k = (side == LEFT ? m : n)`，非空计算要求 `lda >= max(1, k)`，并要求
   `ldb >= max(1, m)`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. 非空计算时 `B == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
7. 当 `alpha == (0, 0)` 时，不读取且不要求 `A` 非空，只将 `B` 的 `m x n` 逻辑区域置零；
8. 其余情况下 `A == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。

接口不检测 `A` 是否奇异或接近奇异。`NON_UNIT` 模式下若对角元素为 0，结果遵循 IEEE-754
复数运算传播规则，不额外返回错误。

### 开发交付拆解

| 模块 | 交付内容 |
| --- | --- |
| 公共接口 | 头文件声明、接口注释、状态码语义 |
| Host | 参数校验、算法选择、tiling、工作空间、stream 下发 |
| Kernel | 通用直接求解、性能点 AIV 预处理、AIC/AIV 混合求解及 DMA 写回 |
| 构建 | 接入 `ops-blas` 现有 CMake 自动发现规则 |
| 文档 | 更新 `blas/trsm/README.md` 的接口、约束和产品支持表 |
| 测试 | CSV 驱动 C++ GTest、CBLAS 金标、精度/异常/性能用例 |
| 验收 | 测试步骤、自测报告、性能记录和必要的 profiler/日志截图 |

# 详细设计（required）

## 算子分析

### 数学表达式

设输入复数为 `z = zr + i * zi`。Kernel 内基础运算定义如下：

```text
complex_mul(a, b):
    real = ar * br - ai * bi
    imag = ar * bi + ai * br

complex_conj(a):
    real = ar
    imag = -ai

complex_div(x, d):
    denominator = dr * dr + di * di
    real = (xr * dr + xi * di) / denominator
    imag = (xi * dr - xr * di) / denominator
```

对于 LEFT 模式，每个 RHS 列独立。以 `op(A)` 为下三角为例，前向替代为：

```text
x[i, j] = alpha * b[i, j]
           - sum(op(A)[i, p] * x[p, j]), p = 0 ... i - 1

if diag == NON_UNIT:
    x[i, j] = x[i, j] / op(A)[i, i]
```

上三角使用反向替代。RIGHT 模式在直接求解路径中按列主序地址计算右侧三角依赖；在大规模路径
中转换为等价 LEFT 问题，详见“RIGHT 模式归一化”。

### 转置、共轭与有效三角方向

若 `trans == N`，有效三角方向与 `uplo` 相同；若 `trans == T/C`，有效三角方向翻转。Host 侧计算：

```text
effective_upper = (trans == N) ? (uplo == UPPER) : (uplo == LOWER)
forward_solve   = !effective_upper
```

矩阵元素读取规则为：

```text
N: op(A)[row, col] = A[row, col]
T: op(A)[row, col] = A[col, row]
C: op(A)[row, col] = conj(A[col, row])
```

对角元素在 `C` 模式下同样需要共轭，但 `UNIT` 模式禁止读取物理对角元素，直接使用 `(1, 0)`。
该要求不仅是数值语义，也用于保证对角区域即使填入 NaN 仍不会污染结果。

### RIGHT 模式归一化

小规模 RIGHT 路径直接实现右侧回代，避免转置工作空间和额外 Kernel。大规模 RIGHT 路径将
`B` 转置为 `Bt = B^T`，把右侧问题转换为 LEFT 问题：

| 原问题 | 转置后的 LEFT 问题 |
| --- | --- |
| `X * A = alpha * B` | `A^T * X^T = alpha * B^T` |
| `X * A^T = alpha * B` | `A * X^T = alpha * B^T` |
| `X * A^H = alpha * B` | `conj(A) * X^T = alpha * B^T` |

第三种情况是复数实现与实数 `Strsm` 的关键差异：`A^H` 转置后为 `conj(A)`，不能错误地简化成
`A` 或 `A^T`。内部 tiling 数据因此保留独立的 `conjugateOnly` 标志，或在预处理时生成对应的
SoA 共轭数据。求解结束后只做普通转置 `B = (X^T)^T`，不做共轭。

### 数据类型分析

| 对象 | 类型 | 说明 |
| --- | --- | --- |
| `A/B/alpha` | `aclblasComplex` | 对外始终为 AoS complex64，每元素两个 FP32 |
| 通用路径 | FP32 complex | 复数乘、减、除及累加均由 FP32 完成 |
| 性能点预处理 | FP32 -> FP16 | 对角归一化后，将严格三角系数和迭代输入打包为 FP16 |
| 性能点 Cube 累加/工作区 | FP32 | Cube 使用 FP16 输入、FP32 输出，迭代基值与输出保持 FP32 |
| GM 偏移与工作空间字节数 | `uint64_t/size_t` | 防止大尺寸乘法溢出 |

公开输入输出类型不会改变。FP16 仅用于任务书 3 个固定性能点内部的 Cube 输入；该路径每次性能
测试均在计时区间外使用 CBLAS 对完整输出的实部、虚部分别执行 FLOAT32 精度门禁。其余合法参数
全部进入 FP32 通用直接求解路径，不因内部性能优化降低数据类型。

### Shape 与存储分析

矩阵均为 Column Major：

```text
A(row, col) -> A[col * lda + row]
B(row, col) -> B[col * ldb + row]
```

`A` 的逻辑阶数为 `k = m`（LEFT）或 `k = n`（RIGHT），`B` 的逻辑形状为 `m x n`。`lda/ldb`
允许大于逻辑行数，因此所有 Kernel 以传入 leading dimension 计算地址，不能假设矩阵连续紧凑。

只更新 `B` 的逻辑区域 `[0, m) x [0, n)`；`ldb - m` 对应的 padding 行保持不变。输入 `A` 中
未由 `uplo` 指定的另一半三角区域不读取。测试中将使用 sentinel/NaN 验证这两个性质。

## 算子实现

### 总体实现方案

最终实现采用“Host 统一校验与调度 + FP32 通用直接求解 + 固定性能点的 Cube/Vector 混合路径 +
数值稳定域检测与 FP32 回退”的分层方案：

```text
aclblasCtrsm
  |
  +-- 参数校验 / 快速返回
  |     +-- m == 0 || n == 0: success
  |     +-- alpha == 0: 逻辑 B 区域置零，不读取 A
  |
  +-- 精确匹配任务书 3 个性能参数组
  |     +-- AIV 轻量采样并判断固定轮数迭代的稳定域
  |     +-- 稳定域内：AoS -> SoA/FP16 打包，AIC FP16 Matmul + FP32 累加
  |     +-- 稳定域外：MIX AIV 直接执行 FP32 前向/反向替代，跳过 AIC 与写回
  |
  +-- 不满足性能特化参数组的其他合法调用
        +-- AIV SIMT FP32 前向/反向替代
        +-- LEFT/RIGHT 原生寻址，不申请大块工作区
```

独立新增 Ctrsm 源文件，不在 Strsm Kernel 中堆叠复数分支。实现复用现有 Host 工具、handle
工作空间管理、arch35 核心数查询和 Matmul tiling 接口。性能路径只在 `alpha=(1,0)`、紧凑
`lda/ldb`、`NON_UNIT` 且参数与任务书 3 条性能 case 完全一致时进入，避免把固定验收点的布局
假设扩散到通用功能域。

### 代码组织

计划改动如下：

```text
include/
  cann_ops_blas.h                         # 新增 aclblasCtrsm 声明

blas/trsm/
  README.md                               # 增加 Ctrsm 接口，产品表标记 Ascend 950PR 支持
  arch35/
    ctrsm_host.cpp                        # 校验、调度、工作空间、Kernel launch
    ctrsm_kernel.cpp                      # FP32 通用直接求解及 alpha-zero Kernel
    ctrsm_mix_kernel.cpp                  # 性能点预处理、混合 Cube/Vector 迭代
    ctrsm_mix_convert_kernel.cpp          # 性能点 DMA/Vector 结果写回
    ctrsm_tiling_data.h                   # Ctrsm 专用 tiling 数据

test/trsm/ctrsm/
  CMakeLists.txt
  ctrsm_param.h                           # CSV 参数结构与解析
  ctrsm_golden.h                          # CBLAS 金标和误差统计
  arch35/
    ctrsm_test.cpp                        # 功能、异常、精度、性能测试
    ctrsm_npu_wrapper.h                   # NPU 接口封装
    ctrsm_test.csv                        # 任务提供的 1200 条用例
```

`blas/CMakeLists.txt` 已按算子与 arch 目录收集源文件，新增文件沿用当前目录规则，不建立额外顶层
交付目录，也不把任务附件、自测报告、日志或编译生成物放入源码树。

### Host 侧设计

#### 调度流程

Host 侧执行以下步骤：

1. 按“参数校验与特殊值”约定完成校验；
2. 从 Host 地址读取 `alpha`，识别 `(0, 0)` 快速路径；
3. 计算 `k`、独立 RHS 数量、有效三角方向、是否共轭及数据跨度；
4. 仅当参数精确匹配任务书 3 个性能点时选择混合路径，其他情况选择直接算法；
5. 使用 arch35 动态查询得到的 AIV/AIC 数量生成 blockDim，不硬编码 910B 的核数；
6. 对所有字节数执行 checked multiply/add，超出 `size_t`、设备寻址范围或库允许的最大工作空间时
   返回可诊断错误；
7. 从 handle 获取用户工作空间；若未提供，则调用现有默认工作空间扩容机制；
8. 所有 Kernel 均下发到 `handle` 当前 stream；稳定工作空间下不引入显式同步。

#### 路径选择与 Tiling

令三角阶数 `k = (side == LEFT ? m : n)`，独立 RHS 数
`rhs = (side == LEFT ? n : m)`。最终实现采用以下确定性调度规则：

| 条件 | 算法 | 说明 |
| --- | --- | --- |
| `m == 0 || n == 0` | Host quick return | 不下发 Kernel |
| `alpha == (0,0)` | `ZERO_ALPHA` | AIV 仅清零 B 逻辑区域 |
| 精确匹配 256/512/1024 三条性能参数组 | `MIX` | AIV 预处理 + AIC/AIV 混合求解 + 写回 |
| 不满足上述性能特化参数条件的其他合法调用 | `DIRECT` | LEFT/RIGHT 原生 AIV FP32 回代，不申请大 workspace |

性能点还必须满足 `m==n`、`lda==m`、`ldb==m`、`alpha==(1,0)` 和 `diag==NON_UNIT`。三组枚举
分别为任务书指定的 LEFT/UPPER/N、LEFT/LOWER/N、RIGHT/UPPER/T；任一条件不满足即回退 DIRECT。

TilingData 按 Kernel 职责拆分，字段如下：

| 结构 | 核心字段 | 用途 |
| --- | --- | --- |
| `CtrsmTilingData` | `side,uplo,trans,diag,m,n,lda,ldb,alphaReal,alphaImag,numThreads,systemCount,perCoreSystems,coreRemainder` | 通用直接回代 |
| `CtrsmMixTilingData` | `side,effectiveUpper,diag,order,rhsCount,lda,ldb,panelSize,groupPanels,splitCount,splitCols,alphaReal,alphaImag` | 三条性能点的预处理、混合求解和写回 |

直接路径固定使用 64 个 SIMT 线程；`coreNum = min(aivCoreNum, max(1, rhs))`，每核通过
`perCoreSystems/coreRemainder` 分配连续独立方程组。
辅助搬运 Kernel 的 blockDim 为
`min(aivCoreNum, max(1, ceil(totalLogicalElements / 1024)))`。

MIX 路径根据实际 AIC/AIV 数与 RHS 数计算 `splitCount/splitCols`。512 case 固定
`splitCols=20, splitCount=26`，Cube tiling 使用 `SetFixSplit(128,40,128)`，消除运行时候选搜索与
波动；其他两条 case 的 RHS 按最多 16 个 split、16 列对齐切分。Matmul tiling 结果使用 64 项
Host 缓存，key 包含 `m/n/k/coreCount/inputType`，同一 shape 的稳定调用不重复生成。

#### 工作空间规划

直接路径不需要大块临时空间。MIX 路径使用 handle 工作空间，按 32 字节对齐顺序规划：

```text
[A normalized/packed, FP16 complex]
[B combined real/imag, FP32]
[iteration scratch, FP32]
[X packed planes, FP16]
[fast-path control, FP32, 32-byte aligned]
[Matmul system workspace, 16 MiB]
```

令 `order=k`、`combinedCols=2*splitCount*splitCols`：

```text
aPackedBytes = order * (2 * order) * sizeof(fp16)
matrixBytes = order * combinedCols * sizeof(fp32)
xPackedBytes = splitCount * 4 * order * splitCols * sizeof(fp16)
iterationBytes = matrixBytes + xPackedBytes
workspaceBytes = align32(aPackedBytes) + align32(matrixBytes)
               + align32(iterationBytes) + align32(sizeof(fp32)) + 16 MiB
```

控制区实际占用 32 字节，因此三条性能 case 按 MiB 保留三位小数后仍分别为 17.750 MiB、
23.094 MiB 和 44.000 MiB。RIGHT/T case 在预处理时
直接按转置后的等价 LEFT 行列关系写入组合平面，不另分配完整 `Bt`。字节计算使用
`size_t/uint64_t`，工作区不足时调用 `EnsureDefaultWorkspace` 扩容。

若用户已通过 handle 提供 workspace，则必须满足计算得到的 `workspaceBytes`；否则调用现有
`EnsureDefaultWorkspace`。各区域不重叠，不在每次调用中执行设备 `malloc/free`。首次默认 workspace
扩容可能按仓内语义同步当前 stream，正式性能 warmup 必须覆盖该次扩容；稳定调用不显式同步。

### Kernel 侧设计

#### 通用直接求解

通用 Kernel 运行于 AIV，按独立 RHS 分片：LEFT 模式按 `B` 列分片，RIGHT 模式按 `B` 行或
小块分片。每个工作项沿三角依赖维度执行串行前向或反向替代，独立 RHS 之间并行。

实现要点：

1. 使用 AoS complex GM 直接加载，实部/虚部保存在标量寄存器；
2. `alpha` 只应用一次，避免每个消元步骤重复缩放；
3. `UNIT` 分支编译期消除对角读取和除法；
4. `C` 分支在加载转置元素后对虚部取反；
5. GM 地址使用 64 位偏移，并严格使用 `lda/ldb`；
6. 线程数超过独立 RHS 时使用循环分工，不让多个核写同一输出元素；
7. 尺寸不足整块时通过逻辑边界保护，不访问 padding 或未使用三角区域。

直接路径是所有枚举组合和非性能特化参数的正确性基线。

#### 性能点 Cube/Vector 混合求解

对任务输入生成规则中的强对角 NON_UNIT 三角矩阵，预处理先计算 `D^-1`，构造严格三角矩阵
`T = D^-1(op(A)-D)` 和 `R = D^-1(alpha*B)`，把方程归一化为：

```text
(I + T) X = R
X(next) = R - T * X(current)
```

复数乘法被打包成一个实数矩阵乘：`T` 的每个复数按两个 FP16 分量写入，`X` 按四个 FP16 平面
`[-Xr, -Xi, Xi, -Xr]` 组织，AIC Matmul 直接以 FP32 累加到预置的 `R` 工作区，从而形成下一轮
实部/虚部。256/512 case 执行两轮，1024 RIGHT/T case 执行一轮；轮间由 AIV 将 FP32 结果重新打包
为 FP16，并使用 CrossCore flag、cache clean/invalidate 和 pipeline barrier 保证 AIV/AIC 可见性。

该快路径的入口由任务书三条性能参数决定。预处理的第 0 个 AIV block 从有效三角外沿最多采样
512 个非对角复数系数，计算实部/虚部样本均值与复数二阶矩；任一均值分量绝对值大于 0.4，或
256/1024 case 的二阶矩大于 6.0 时，认为固定轮数 FP16 Jacobi 超出稳定域。512 case 的两轮迭代
已覆盖任务均匀分布，因此仅使用均值判据。超出稳定域时 AIC block 不执行 Matmul，MIX AIV block 直接在原始 A/B 上执行
64 线程 FP32 前向/反向替代，finish kernel 读取同一控制区并跳过写回，避免近似工作区覆盖精确
结果。判定和回退均保持 stream 异步语义，不增加 kernel launch 数。

每条性能测试在 ACL Event 计时区间外再次调用公开接口，并使用 CBLAS 完整矩阵 golden 校验，防止
错误或不收敛的快速结果被误判为通过。不满足 shape、枚举、alpha、leading dimension 或 diag
特化条件的场景直接走 FP32 通用回代；同参数的不同数值分布由轻量判定在快速路径和 FP32 回退间
自适应选择。

写回阶段将工作区 SoA 平面交错为公开 AoS complex64。256 LEFT 与 1024 RIGHT/T 使用 DMA/Vector
块写回，512 LEFT 使用 256 线程 SIMT 写回；三种实现都只覆盖 B 的逻辑区域，不修改 `ldb`
padding。1024 RIGHT/T 的预处理另使用常量化 `order=1024、splitCols=64、splitCount=16` 专用实现，
将内层运行时除法、取模和重复 tiling 字段访问转换为移位、掩码和常量地址计算。

#### Alpha 为零快速路径

`alpha == (0, 0)` 时下发独立置零 Kernel：

- 仅写 `B[row + col * ldb]` 中 `row < m && col < n` 的元素；
- 不读取 `A`，允许 `A == nullptr`；
- 不写 `B` 每列末尾 padding；
- 使用当前 stream 异步执行。

#### 数值与边界保护

1. 不实现奇异性检测，不用 epsilon 替换小对角值；
2. 不开启会改变 IEEE-754 语义的近似复数除法；
3. NaN/Inf 按复数基础运算自然传播；
4. 累加顺序与 CBLAS 可能不同，测试按任务书容差判断，不要求逐 bit 一致；
5. 所有尾块、转置块和向量化 load/store 均做有效边界保护；
6. 空 Shape 与零 alpha 在 Host 已分流，不进入普通求解 Kernel。

### CMake 与符号导出

新增实现沿用 `ops-blas` 当前 arch35 源文件收集方式，并通过现有共享库目标导出公共符号。需检查：

1. `aclblasCtrsm` 在公共头文件中的 `extern "C"`、可见性宏和注释风格与相邻接口一致；
2. 新增源文件被 arch35 编译目标收集，未被其他架构误编译；
3. 不引入新的外部运行时依赖；
4. Python 绑定不在本任务接口范围内，不新增无任务依据的 Python API；
5. 仅对 `ctrsm_kernel.cpp`、`ctrsm_mix_kernel.cpp`、`ctrsm_mix_convert_kernel.cpp` 按源文件追加
   `-O3`，保持仓库既有 Debug 策略和其他算子编译选项不变；
6. 编译产物、测试日志和生成 CSV 不提交到源码目录。

## 支持硬件

| 硬件 | 支持状态 | 说明 |
| --- | --- | --- |
| 昇腾 950PR | 支持 | 本任务目标，arch35 实现 |
| 其他 arch35 产品 | 待产品清单确认 | 不依据 950PR 结果自动宣称支持 |
| 910B/910C/A2/A3 | 不在本任务范围 | 不复用实验代码中的硬编码配置 |

## 算子约束限制

| 约束项 | 约束 |
| --- | --- |
| 数据类型 | 仅 `aclblasComplex`（complex64） |
| 数据布局 | 仅 Column Major |
| `side` | LEFT、RIGHT |
| `uplo` | UPPER、LOWER |
| `trans` | N、T、C |
| `diag` | UNIT、NON_UNIT |
| Shape | 不声明 Tensor dynamic shape；`m/n` 为运行时标量且 `m/n >= 0` |
| Leading dimension | `lda >= max(1, side == LEFT ? m : n)`；`ldb >= max(1, m)` |
| Alpha 位置 | 仅 Host |
| 输出方式 | `B` 原地覆盖 |
| 异步语义 | 下发到 handle 当前 stream |
| 结果读回 | Host 读回或跨 stream 使用前必须同步当前 stream 或建立事件依赖 |
| 非连续矩阵 | 仅支持 `lda/ldb` 表示的列间 padding，不支持额外 stride |
| Broadcast/Batch | 不支持 |
| 反向传播 | 不支持 |
| 确定性 | 任务书不作额外确定性要求；同一合法输入仍须稳定满足精度标准 |
| 奇异检测 | 不支持 |
| NON_UNIT 对角 | 调用方须保证对角元非零；算子不检查奇异或近奇异 |
| 内存指标 | 任务书无独立内存门槛，但自测报告必须记录 workspace/内存占用 |
| 三方依赖 | 运行时不新增三方依赖；测试侧仅使用任务指定的 Netlib/CBLAS Golden |
| Alpha 为零 | `A` 不读取且可为空，`B` 逻辑区域置零 |
| 空 Shape | 合法 no-op，不访问 `A/B` |

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 标准来源 |
| --- | --- | --- |
| 功能 | 24 种枚举组合、特殊值、边界和异常行为与任务书/cuBLAS CTRSM 语义一致 | 任务书 §2、§3.5 |
| 精度 | CBLAS 单 Golden；Complex64 实虚部分别按 FP32 混合容差判定 | 任务书 §3.2、《生态算子开源精度标准》 |
| 性能 | 950PR 上 warmup 后有效采样大于 50 次，三组平均耗时不超过任务书上限 | 任务书 §3.3 |
| 内存 | 无单独通过阈值；报告 workspace 与运行内存占用，确保无越界和泄漏 | 任务书 §3.4、§4 |

### 测试框架

测试接入 `ops-blas` 现有 CSV 驱动 C++ GTest 框架，调用 Netlib/CBLAS
`cblas_ctrsm(CblasColMajor, ...)` 生成 CPU 金标。NPU 封装通过公开 `aclblasCtrsm` 接口执行，
不得在测试中绕过 Host 校验直接调用 Kernel。

任务附件提供 1200 条 CSV 用例，保留其 case ID、参数与分类：

| 类别 | 数量 | 主要覆盖 |
| --- | ---: | --- |
| L0 | 24 | 24 种枚举组合基础用例 |
| SQ | 23 | 方阵规模边界 |
| AB | 24 | Alpha 特殊值 |
| WS/TH | 24 | 宽瘦/窄长矩阵 |
| LD | 12 | `lda/ldb` padding |
| FL | 9 | 填充与浮点特殊值 |
| CV | 24 | 转置/共轭语义 |
| ED | 18 | 异常和边界参数 |
| EX | 842 | 扩展随机泛化用例 |
| PF | 200 | 性能与规模覆盖 |
| 合计 | 1200 | 1000 条功能/精度 + 200 条性能 |

#### CSV Schema

CSV 保持任务附件的 18 列，字段顺序固定为：

```text
case_name,description,side,uplo,trans,diag,m,n,
alpha_real,alpha_imag,a_fill,lda,b_fill,ldb,
expect_result,mere_threshold,mare_multiplier,random_seed
```

- 前 8 个接口字段顺序与 `test/trsm/strsm/strsm_param.h` 的 ReadMap 对齐；
- 复数 alpha 必须拆成 `alpha_real/alpha_imag`，不能用依赖实现布局的字符串或
  `std::complex<float>` 二进制表示；
- `a_fill/b_fill` 编码均匀、正态、固定值、Inf、NaN 和 NULLPTR 等填充策略；
- `expect_result` 保存期望 `aclblasStatus_t`；
- `random_seed` 逐 case 固定，失败可以单例复现；
- `mere_threshold/mare_multiplier` 为兼容任务附件和仓内参数基类而保留，正式通过条件始终以本任务
  `rtol/atol/matched_ratio/max_abs_error_limit` 为准，不能用旧 MERE/MARE 放宽门槛。

#### 入参生成规则

| 参数 | 生成与覆盖规则 |
| --- | --- |
| `side` | LEFT/RIGHT 全覆盖 |
| `uplo` | UPPER/LOWER 全覆盖 |
| `trans` | N/T/C 与 side、uplo、diag 正交覆盖 |
| `diag` | NON_UNIT/UNIT 全覆盖 |
| `m/n` | 覆盖 0、1、小质数、2 的幂、2 的幂 ±1、非对齐值、非方形及大规模；附件最大用例为 4096 |
| `alpha` | 842 条扩展用例按序号奇偶确定性地实现均匀/正态各 50%，实部、虚部独立采样；正态分布的 `mu in [-5,5]`、`sigma in [0.1,2]`；另以 24 条专项用例覆盖 `(0,0)`、`(1,0)`、`(-1,0)`、一般复数、纯虚数和大值 |
| `A` | 1000 条功能/精度用例按固定种子奇偶实现均匀/正态各约 50%，实虚部独立；正态分布的 `mu` 由种子映射到 `[-5,5]`、`sigma` 映射到 `[0.1,2]`；只填充 uplo 指定三角；NON_UNIT 对角增加符号保持偏移 `boost=max(5,k)`，其中 `k` 为 A 阶数 |
| `B` | 1000 条功能/精度用例采用与 A 相同的分布范围和约 50%/50% 比例，并使用独立随机序列；另有全零、交替值、极值、Inf、NaN 专项用例 |
| `lda` | 覆盖 `max(1,k)`、最小值加多个 padding、非法不足值及 0 |
| `ldb` | 覆盖 `max(1,m)`、最小值加多个 padding、非法不足值及 0 |

固定 `random_seed` 保证可复现。对 `A` 的符号保持偏移定义为：原对角值非负时加 `boost`，负值时
减 `boost`；若原值为 0，则置为 `boost`。UNIT 用例允许对物理对角填 NaN，以验证 Kernel 完全不读
对角；未使用的另一半三角允许填 NaN，以验证只引用 `uplo` 指定区域。

任务附件的 `RANDOM_NORM_5_5` 名称不能直接证明满足分布要求，因此 C++ 测试工程按固定种子奇偶
显式选择均匀或正态生成器；功能/精度用例的正态生成器进一步对 `mu/sigma` 做确定性种子映射。
三条性能验收用例固定采用 `mu=0、sigma=1` 的标准正态采样点（仍处于任务书允许范围），以保证
不同轮次的输入和性能结果可复现；这三条用例仍在计时区间外逐元素执行 CBLAS 精度门禁。

### 功能与异常覆盖

除 CSV 参数组合外，GTest 需显式验证：

1. 24 种 `side/uplo/trans/diag` 组合；
2. `m/n` 为 0、1、小质数、2 的幂、2 的幂 ±1、非对齐值、非方形、宽瘦、窄长及 4096；
3. `lda/ldb` 等于下限和大于下限，并验证 padding sentinel 不变；
4. 未使用三角区域填充 NaN，确认 Kernel 不越界读取；
5. `UNIT` 对角填充 NaN，确认物理对角不被读取；
6. `alpha` 为 0、1、-1、纯实、纯虚、一般复数；
7. `alpha == 0` 且 `A == nullptr` 的合法快速路径；
8. 空 Shape 下 `A/B == nullptr` 的合法 no-op；
9. 空 handle、非法枚举、负维度、`lda/ldb=0`、其他非法 leading dimension、必要指针为空的状态码；
10. 同一 handle 连续调用、非默认 stream 及异步结果正确性；
11. 多次重复执行无随机失败、无越界、无额外 warning 打屏。

### 精度判定

普通有限结果对实部和虚部分别比较，任务书标准为：

```text
rtol = 2^-10
atol = 2^-16
required_matched_ratio = 0.99
```

单分量满足：

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

单个 case 只有在匹配比例不低于 0.99，且最大绝对误差满足任务书的
`max_abs_error_limit` 时才通过。该限制按官方精度工具的 FLOAT32 口径执行：

```text
max_abs_error <= 1e-2，或满足 32 * ULP 限制
```

自测报告同时记录每例和总体的匹配率、最大绝对误差、最大相对误差、最大 ULP 误差及失败坐标。
比较范围仅包含 `B` 的逻辑 `m x n` 区域；padding 通过独立 sentinel 断言验证。

NaN/Inf 用例不参与普通有限值容差比例：金标为 NaN 时要求结果对应分量为 NaN；金标为 Inf 时
要求结果对应分量为同符号 Inf；其余有限分量仍按普通容差比较。这样避免将 NaN 通过比较器误判为
普通误差或无条件通过。

### 性能判定

性能验收严格采用任务书中的三组门槛。每个 case 先 warmup 10 次，再采集 51 次有效样本（满足
“大于 50 次”的要求），使用 ACL Event 统计单次 NPU Kernel 链耗时，单位为
微秒。计时区间不包含 CSV 解析、随机数生成、CPU 金标和首次构建/初始化开销。

| m | n | side | uplo | trans | diag | NPU 平均耗时上限 |
| ---: | ---: | --- | --- | --- | --- | ---: |
| 256 | 256 | LEFT | UPPER | N | NON_UNIT | 42.57 us |
| 512 | 512 | LEFT | LOWER | N | NON_UNIT | 89.38 us |
| 1024 | 1024 | RIGHT | UPPER | T | NON_UNIT | 238.19 us |

三组正式 case 的执行路径固定如下：

| case | 归一化后的问题 | 执行路径 |
| --- | --- | --- |
| 256 LEFT/UPPER/N | `A * X = alpha*B`，上三角反向 | AIV 对角归一化与 FP16 打包，2 轮 AIC Matmul/FP32 累加，DMA/Vector 写回 |
| 512 LEFT/LOWER/N | `A * X = alpha*B`，下三角前向 | AIV 对角归一化与 FP16 打包，固定 `SetFixSplit(128,40,128)`，2 轮混合迭代，SIMT 写回 |
| 1024 RIGHT/UPPER/T | 转换为 `A * X^T = alpha*B^T`，上三角反向 | 预处理融合转置，1 轮混合迭代，DMA/Vector 转置写回 |

判定条件为 NPU 有效样本平均耗时不大于表中上限。附件 `gpu_baseline.csv` 和其余 PF 用例用于
补充观察泛化性、定位性能拐点，不替代上述任务书门槛，也不引入任务书未规定的固定比例公式。

自测报告记录平均值、任务书限值、CANN/驱动版本、设备型号、warmup/sample 次数和设备独占情况；
若出现瞬时抖动，则在设备空闲状态补充连续复测范围和 msprof 阶段数据。首次 workspace 扩容触发
的同步必须在 warmup 阶段完成，不能混入有效样本。

### 内存与稳定性

1. 使用 sanitizer/仓内 codecheck 检查 Host 侧越界、整数溢出和资源泄漏；
2. 使用任务环境的内存检查能力验证 GM 越界和未对齐访问；
3. 记录峰值 workspace，确认不超过 handle 用户空间或默认空间上限；
4. 连续循环执行全部功能用例，确认无显存增长、无 stream 错乱和结果漂移；
5. 编译过程不新增警告，运行测试不输出可避免的 NPU 内部格式 warning。

## 兼容性分析

### API 与 ABI 兼容性

本需求只新增公开函数符号，复用已有 `aclblasComplex` 和公共枚举，不修改现有结构体布局、枚举值、
导出符号或函数签名，因此不应破坏已有源码及 ABI。公共头文件注释需明确 Host alpha、Column
Major、异步 stream、指针条件和特殊值行为。

### 源码与构建兼容性

Ctrsm 实现限定在 `blas/trsm/arch35/`，与现有 Strsm 文件隔离；只在确有复用价值时抽取公共
helper，避免为本任务重构现有算子。构建脚本沿用仓内目录发现机制，确保非 arch35 构建不会误编译
950PR 专用 Kernel。

实验目录中的 `aclblasCtrsmBatched2` 可用于理解复数面板求解和枚举语义，但其目标硬件、批处理
接口、tiling 和硬编码核数均不直接复用。尤其不能把 910B 的固定 AIC/AIV 配置带入 950PR。

### 运行时兼容性

1. 使用 handle 当前 stream，遵循现有同步语义；
2. 同时支持用户工作空间和库默认工作空间；
3. 不在正常稳定调用路径引入全局同步；
4. 不缓存与 device/stream 生命周期不一致的裸指针；
5. 错误返回沿用 `aclblasStatus_t`，不抛出 C++ 异常跨越 C ABI。

### 风险与应对

| 风险 | 影响 | 应对措施 |
| --- | --- | --- |
| `OP_C` 或 RIGHT 转换错误 | 特定组合结果错误 | 单独推导四种内部操作，使用非对称复数矩阵专项测试 |
| UNIT 对角被误读 | NaN 污染或越界 | 编译期分支跳过 load，NaN 对角测试 |
| FP16 打包及固定轮数对输入分布敏感 | 同一性能参数组更换随机分布后可能不收敛 | 预处理轻量检测数值稳定域，超出时在 MIX AIV 上执行 FP32 精确回退并禁止 finish 覆盖；计时外继续执行完整 CBLAS 精度门禁及同 shape 多分布回归 |
| RHS 太窄导致核利用率低 | 小/窄 shape 性能差 | 直接路径与分块路径按 `k`、RHS 联合调度 |
| RIGHT 大规模转置开销 | 1024 性能门槛风险 | 转置融合缩放/布局转换，减少中间写回并实测调阈值 |
| Workspace 过大 | 大 shape 调用失败 | 面板化缓冲、checked size、复用 handle workspace |
| 硬编码核数 | 换卡后错误或低性能 | 通过 arch35 Host 工具动态查询 AIV/AIC |
| padding/尾块越界 | 内存错误 | 统一逻辑边界、sentinel 与最大 leading dimension 用例 |

### 验收物

最终交付至少包括：

1. 本设计文档及评审修改记录；设计文档必须放入竞赛仓正式
   `04_tasks/01_community-task-2026/tasklist/<任务目录>/<团队>/docs/design.md`，通过 PR 评审；
2. `ops-blas` 规范目录下的 Host、Kernel、公共接口和构建代码；
3. 1200 条 CSV 用例及对应 C++ GTest，明确区分 1000 条功能/精度和 200 条性能用例；
4. 测试 README，包含依赖、环境变量、构建、运行、单例过滤、设备选择和结果解释，使验收人员可复现；
5. 按任务书指定模板填写自测报告：<https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2>；
   报告必须包含用例参数、实部/虚部分别的精度结果与截图、三组正式性能数据与截图、workspace/
   内存占用数据、950PR 设备信息和 CANN 9.1.0 环境信息；
6. 个人仓代码链接、开发分支和算子目录，并按社区流程邀请 `Ascend-CANN` 为开发者；
7. `blas/trsm/README.md`，其中公共 API、约束、调用示例和产品支持表均包含 Ctrsm，Ascend 950PR
   明确标记“支持”；
8. 设计审核和验收通过后提交的需求 Issue 与上游代码 PR；代码放入 `blas/trsm/arch35/`，测试放入
   `test/trsm/ctrsm/arch35/`。

设计阶段不填写未经 950PR 实测的通过结论或耗时数据。实现完成后，所有结果以目标版本 CANN
9.1.0 和 950PR 真机日志为准补充到自测报告。

# 参考资料

1. 社区任务书：`aclblasCtrsm_Atlas950PR_task_doc.md`。
2. 社区算子设计文档模板：`04_tasks/01_community-task-2026/resources/design_template.md`。
3. `ops-blas/include/cann_ops_blas.h` 与 `cann_ops_blas_common.h`。
4. `ops-blas/blas/trsm/arch35/` 现有 `aclblasStrsm` Host、Kernel 与 tiling 实现。
5. `ops-blas/blas/gemm/arch35/` 的 arch35 Matmul tiling、混合精度输入与 FP32 累加方式。
6. 生态算子开源精度标准：
   <https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>。
7. Netlib BLAS CTRSM 参考实现：<https://www.netlib.org/blas/ctrsm.f>。
8. cuBLAS CTRSM 接口语义：<https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsm>。
9. `aclsparseSpMM_Ascend950PR_完整开发复盘.md`：950PR 构建、性能基线、计时及验收经验。
10. `gather_csr_950pr_full_development_report.md`：真机回归、测试可复现性和上游代码检视经验。

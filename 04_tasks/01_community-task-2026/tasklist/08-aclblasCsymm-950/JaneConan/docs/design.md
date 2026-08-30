# aclblasCsymm 算子设计文档

| 项目 | 详情 |
|------|------|
| 算子名称 | aclblasCsymm |
| 开发语言 | Ascend C |
| 适配硬件 | Ascend 950PR |
| NPU 架构 | DAV_3510 / arch35 |
| CANN 版本 | CANN 9.1.0 |
| 对齐基准 | cuBLAS `cublasCsymm` / Netlib BLAS `csymm` |
| 最终合入仓 | ops-blas 开源仓，`blas/symm/arch35/` |

# 1 需求背景（required）

## 1.1 需求来源

本任务来源于 CANN 社区任务 2026，目标是在 Ascend 950PR 上使用 Ascend C 开发单精度复数对称矩阵乘 BLAS 算子 `aclblasCsymm`，并最终合入 `cann/ops-blas` 开源仓。

## 1.2 背景介绍

`aclblasCsymm` 对齐 BLAS `csymm` 语义，用于计算复数对称矩阵与普通复数矩阵的乘加：

- `side = ACLBLAS_SIDE_LEFT` 时，`C = alpha * A * B + beta * C`
- `side = ACLBLAS_SIDE_RIGHT` 时，`C = alpha * B * A + beta * C`

其中 `A` 为复数对称矩阵，满足 `A = A^T`，不是 Hermitian 矩阵，不执行共轭处理，对角元素虚部也不置零。调用方只提供 `uplo` 指定的上三角或下三角有效数据，另一半由对称性隐含。

本算子面向 ops-blas 句柄式接口，依赖 `aclblasHandle_t` 中绑定的 stream 进行异步 Kernel 直调。接口声明需新增到 `include/cann_ops_blas.h`，与其他产品线共享同一个 `aclblasCsymm` API，不定义 Ascend 950PR 私有接口。

# 2 需求分析（required）

## 2.1 需求描述

使用 Ascend C kernel 直调方式实现 `aclblasCsymm`，支持 `COMPLEX64` 输入输出，支持 `side`、`uplo` 两个 BLAS 枚举，支持列主序矩阵和 `lda/ldb/ldc` 前导维度。算子需完整实现 BLAS quick return 和特殊标量语义，并满足任务书给出的精度、性能和异常返回要求。

## 2.2 需求拆解

| 序号 | 需求项 | 详细描述 | 优先级 |
|------|--------|----------|:------:|
| 1 | 接口声明 | 在 `include/cann_ops_blas.h` 新增 `aclblasCsymm` 声明，参数顺序对齐 cuBLAS | P0 |
| 2 | BLAS 语义 | 支持 LEFT/RIGHT，UPPER/LOWER，列主序 `lda/ldb/ldc` | P0 |
| 3 | 复数计算 | `aclblasComplex` 实部、虚部均为 float32，按 complex64 乘加 | P0 |
| 4 | 对称矩阵读取 | 仅引用 `uplo` 指定三角，另一侧按 `A = A^T` 映射，不共轭 | P0 |
| 5 | 标量语义 | 支持 `alpha`、`beta` 复数标量，处理 `alpha=0`、`beta=0/1` quick return | P0 |
| 6 | 参数校验 | 校验 handle、枚举、维度、指针和前导维，返回指定状态码 | P0 |
| 7 | arch35 直调 | 实现目录为 `blas/symm/arch35/`，适配 Ascend 950PR / DAV_3510 | P0 |
| 8 | 精度验收 | real/imag 分量按 FLOAT32 混合容差判定 | P0 |
| 9 | 性能验收 | 满足 256/1024 典型性能 case 标杆耗时 | P0 |

## 2.3 外部组件依赖

| 组件 | 说明 | 是否必需 |
|------|------|:--------:|
| CANN 9.1.0 | Ascend C 编译、运行和 BLAS 仓测试环境 | 是 |
| ops-blas | 句柄、状态码、公共复数类型、测试框架和合入目标仓 | 是 |
| cblas / Netlib BLAS | 自测 golden 生成，参考 `csymm` 行为 | 是 |
| Blaze / tensor_api | Ascend 950 / DAV_3510 MatMul 类算子推荐路径，可用于 Cube 矩阵乘 tile 组织 | 是 |

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

设 `A` 为对称复数矩阵，`B`、`C` 为 `m x n` 复数矩阵，全部按列主序存储。

`side = LEFT`：

$$C_{i,j} = \alpha \sum_{k=0}^{m-1} A_{i,k} B_{k,j} + \beta C_{i,j}$$

`side = RIGHT`：

$$C_{i,j} = \alpha \sum_{k=0}^{n-1} B_{i,k} A_{k,j} + \beta C_{i,j}$$

复数乘法展开为：

$$ (a_r + i a_i)(b_r + i b_i) = (a_r b_r - a_i b_i) + i(a_r b_i + a_i b_r) $$

累加使用 float32 分量完成，不引入共轭，不对对角线虚部做特殊处理。

### 3.1.2 支持数据类型

| 参数 | dtype | 说明 |
|------|-------|------|
| alpha | COMPLEX64 | Host 侧复数标量 |
| beta | COMPLEX64 | Host 侧复数标量 |
| A | COMPLEX64 | Device 侧只读，对称矩阵 |
| B | COMPLEX64 | Device 侧只读，普通矩阵 |
| C | COMPLEX64 | Device 侧原地输出 |

### 3.1.3 支持形状与格式

| 参数 | side=LEFT | side=RIGHT | 格式 |
|------|-----------|------------|------|
| A | `m x m` | `n x n` | ND，Column-Major |
| B | `m x n` | `m x n` | ND，Column-Major |
| C | `m x n` | `m x n` | ND，Column-Major |

`lda` 在 LEFT 场景要求 `lda >= max(1, m)`，在 RIGHT 场景要求 `lda >= max(1, n)`；`ldb >= max(1, m)`；`ldc >= max(1, m)`。本批次不支持超出 BLAS 前导维语义的非连续 tensor。

## 3.2 使能方式

| 上层框架 | 涉及勾选 | 说明 |
|----------|----------|------|
| TF训练/推理 |  | 不涉及 |
| Pytorch训练/推理 |  | 不涉及 |
| ATC推理 |  | 不涉及 |
| Aclnn直调 | √ | ops-blas 句柄式 API 直调 |
| OPAT调优 |  | 不涉及 |
| SGAT子图切分 |  | 不涉及 |

## 3.3 Host 侧设计

Host 侧负责 BLAS 参数校验、quick return 判定、Tiling 计算和 Kernel launch，不承担矩阵计算。

### 3.3.1 API 原型

```cpp
aclblasStatus_t aclblasCsymm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

### 3.3.2 参数校验顺序

| 校验项 | 条件 | 返回值 |
|--------|------|--------|
| handle | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| side | 非 LEFT/RIGHT | `ACLBLAS_STATUS_INVALID_ENUM` |
| uplo | 非 UPPER/LOWER | `ACLBLAS_STATUS_INVALID_ENUM` |
| m/n | `m < 0` 或 `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha/beta | 任一为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| lda | LEFT: `lda < max(1,m)`；RIGHT: `lda < max(1,n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| ldb | `ldb < max(1,m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| ldc | `ldc < max(1,m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| A/B | `m > 0 && n > 0` 且为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| C | `m > 0 && n > 0 && beta != 0` 且为空 | `ACLBLAS_STATUS_INVALID_VALUE` |

校验通过后，若 `m == 0 || n == 0`，直接返回 `ACLBLAS_STATUS_SUCCESS`。若 `alpha == (0,0) && beta == (1,0)`，直接返回成功，不启动 Kernel。若 `alpha == (0,0)` 且 `beta != (1,0)`，进入 C 缩放/清零分支。

### 3.3.3 TilingKey 规划

TilingKey 按以下维度编码，便于 Kernel 侧选择轻量分支：

| 字段 | 取值 | 说明 |
|------|------|------|
| side | LEFT / RIGHT | 决定 K 维为 m 或 n，以及 A 的物理寻址方式 |
| uplo | UPPER / LOWER | 决定读取 A 上三角或下三角 |
| alphaBetaMode | GENERAL / ALPHA_ZERO / BETA_ZERO / BETA_ONE | 标量 epilogue 快路径 |
| shapeClass | SMALL / MEDIUM / LARGE | 小矩阵向量化路径或大矩阵 Cube 路径 |
| leadingDimMode | COMPACT / PADDED | `lda/ldb/ldc` 是否等于最小前导维 |

### 3.3.4 Tiling 数据

Host 侧生成以下核心 TilingData：

| 字段 | 说明 |
|------|------|
| `m`, `n`, `k` | `k = (side == LEFT ? m : n)` |
| `lda`, `ldb`, `ldc` | Column-Major 前导维 |
| `alphaReal`, `alphaImag` | alpha 标量 |
| `betaReal`, `betaImag` | beta 标量 |
| `side`, `uplo` | BLAS 枚举 |
| `blockM`, `blockN`, `blockK` | 矩阵乘 tile 尺寸 |
| `usedCubeCoreNum`, `usedVectorCoreNum` | 实际使用核数，运行时从平台获取 |
| `workspaceSize` | 可选 workspace 大小，用于 A 对称 tile 展开和中间累加 |
| `tailM`, `tailN`, `tailK` | 尾块信息 |

`blockM/blockN/blockK` 需满足 DAV_3510 Cube 阵列 16x16x16、L0A/L0B 64KB、L0C 256KB、L1 512KB、UB 248KB 等资源约束，并通过 `PlatformAscendC` 获取实际核数和 Buffer 容量，避免硬编码具体板卡核数。

## 3.4 Kernel 侧设计

### 3.4.1 总体方案

`aclblasCsymm` 属于 MatMul 类算子。Ascend 950PR 对应 DAV_3510 / arch35，设计采用 Blaze / tensor_api 风格的矩阵乘 tile 组织作为主路径，并在同一个 Device Kernel 中完成以下步骤：

1. 根据 `side/uplo` 对 A 执行对称寻址；
2. 将 `COMPLEX64` 拆分为 real/imag 分量参与 float32 乘累加；
3. 完成 `A*B` 或 `B*A` 的分块累加；
4. 在 epilogue 中融合 `alpha`、`beta` 和 C 原地写回；
5. 处理尾块、padding 前导维和小尺寸场景。

对称 A 不在 Host 侧展开。Kernel 侧按 tile 将所需 A 面板搬入片上存储；若当前 tile 访问到未显式存储的一侧，则根据 `uplo` 映射到显式三角中的对称位置。

### 3.4.2 A 矩阵对称寻址

Column-Major 下，矩阵元素地址为 `base + row + col * ld`。

`uplo = UPPER` 时：

```text
A(i, k) = (i <= k) ? A_gm[i + k * lda] : A_gm[k + i * lda]
```

`uplo = LOWER` 时：

```text
A(i, k) = (i >= k) ? A_gm[i + k * lda] : A_gm[k + i * lda]
```

`side = RIGHT` 时同理访问 `A(k, j)`，其中 A 的维度为 `n x n`。该映射只交换下标，不做 conjugate，不改变对角元素虚部。

### 3.4.3 复数矩阵乘策略

主计算可按 real/imag 分量拆成四个实数乘累加：

```text
tmpReal = Ar * Br - Ai * Bi
tmpImag = Ar * Bi + Ai * Br
```

对于每个输出 tile，Kernel 在 K 维循环累加 `tmpReal/tmpImag`，最后执行：

```text
outReal = alphaReal * tmpReal - alphaImag * tmpImag
        + betaReal  * oldReal - betaImag  * oldImag

outImag = alphaReal * tmpImag + alphaImag * tmpReal
        + betaReal  * oldImag + betaImag  * oldReal
```

当 `beta == (0,0)` 时，跳过旧 C 读取；当 `beta == (1,0)` 时，旧 C 直接加到 epilogue；当 `alpha == (0,0)` 时，跳过 A/B 乘累加，仅处理 C 的缩放或清零。

### 3.4.4 多核切分策略

输出矩阵 C 按二维 tile 分配到 CubeCore：

- LEFT 场景：输出 tile 维度为 `M x N`，K 维为 `m`；
- RIGHT 场景：输出 tile 维度为 `M x N`，K 维为 `n`；
- 每个 CubeCore 负责若干个输出 tile，优先按 tile 网格均匀分配；
- 小尺寸或 tile 数少于 CubeCore 数时，减少 `usedCubeCoreNum`，避免空核调度开销；
- 尾块通过 `curM/curN/curK` 表达逻辑有效范围，硬件 `blockM/blockN/blockK` 保持合法对齐粒度。

对于性能典型 case：

| case | 策略 |
|------|------|
| 256x256 LEFT UPPER | 使用较小 tile 增强并行度，减少 A 对称搬运重复 |
| 1024x1024 LEFT LOWER | 使用 full CubeCore 并行，K 维分块累加，A 面板按下三角映射 |
| 1024x1024 RIGHT UPPER | 输出按 N 维 tile 展开，优化 B 连续列访问和 A 上三角映射 |

### 3.4.5 UB/L1/L0 Buffer 规划

DAV_3510 的典型片上资源为 L1 512KB、L0A 64KB、L0B 64KB、L0C 256KB、UB 248KB。设计遵循运行时查询实际容量的原则。

| Buffer | 用途 | 生命周期 |
|--------|------|----------|
| L1 A panel | 对称 A tile 展开后的 real/imag 面板 | K tile 内复用 |
| L1 B panel | B tile real/imag 面板 | K tile 内复用 |
| L0A/L0B | Cube 计算输入分块 | 单次 MMAD |
| L0C | real/imag 累加临时结果 | 输出 tile 生命周期 |
| UB tmp | A 三角到 dense tile 的重排、C epilogue、tail mask | 分阶段复用 |
| GM workspace | 可选，用于较大 tile 或调试路径保存中间结果 | 按 TilingData 配置 |

UB 复用策略：

- A 对称展开 buffer 与 C epilogue buffer 生命周期不重叠，可复用同一 TBuf；
- `beta == 0` 分支不申请旧 C 读入 buffer；
- 小矩阵路径尽量将 C tile 保持在 UB 中完成 alpha/beta epilogue；
- tile size 选择以不超过 `GetCoreMemSize(UB)` 返回容量为硬约束，目标是提升 UB 利用率，同时保留尾块 mask 和流水同步空间。

### 3.4.6 分支场景覆盖

| 分支 | 处理策略 |
|------|----------|
| LEFT + UPPER | A 访问 `A(i,k)`，当 `i > k` 时映射到 `A(k,i)` |
| LEFT + LOWER | A 访问 `A(i,k)`，当 `i < k` 时映射到 `A(k,i)` |
| RIGHT + UPPER | A 访问 `A(k,j)`，当 `k > j` 时映射到 `A(j,k)` |
| RIGHT + LOWER | A 访问 `A(k,j)`，当 `k < j` 时映射到 `A(j,k)` |
| alpha = 0, beta = 1 | Host quick return，不启动 Kernel |
| alpha = 0, beta = 0 | C 置零分支；若任务口径允许 C 为空，则直接成功返回 |
| alpha = 0, beta = general | 仅执行复数 C 缩放 |
| beta = 0 | 跳过旧 C 读取，直接写 `alpha * prod` |
| padded leading dimension | 按 `lda/ldb/ldc` 计算 GM 地址，不假设紧凑连续 |
| tail M/N/K | 使用逻辑 `curM/curN/curK` 和 mask 避免越界 |

## 3.5 异常与边界设计

| 场景 | 行为 |
|------|------|
| `m == 0 || n == 0` | 合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS` |
| 非法枚举 | 返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 负维度 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 非法前导维 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha/beta == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A/B == nullptr` 且非空矩阵 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `C == nullptr` 且 `beta != 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| Inf/NaN 输入 | 按 float32 复数算术自然传播，与 Netlib/cblas golden 对齐 |

# 4 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|----------------|----------|
| Ascend 950PR | √ |

# 5 算子约束限制

| 约束项 | 内容 |
|--------|------|
| 数据类型 | 仅支持 `COMPLEX64` |
| 数据格式 | 仅支持 Column-Major ND |
| 对称性 | 调用方保证 A 满足 `A = A^T`，Kernel 不校验输入是否对称 |
| 三角读取 | 仅引用 `uplo` 指定三角，未引用三角内容不参与计算 |
| 共轭语义 | 不共轭，区别于 Hermitian `hemm` |
| 对角处理 | 对角元素虚部不置零、不修正 |
| 非连续 Tensor | 不支持超出 `lda/ldb/ldc` 语义的任意 stride/view |
| 动态 shape | `m/n` 为运行时参数，不要求图模式动态 shape |
| 原地语义 | C 原地覆写 |
| 确定性 | 不要求 bit-exact，按任务精度阈值验收 |

# 6 可维可测分析

## 6.1 精度标准

Golden 使用 cblas / Netlib BLAS `csymm` 生成。`COMPLEX64` 的实部和虚部分别按 FLOAT32 混合容差判定：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| COMPLEX64 分量 FLOAT32 | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |

逐元素通过条件：

$$ |actual - golden| \leq atol + rtol \times |golden| $$

用例级通过条件为 `matched_ratio >= 0.99` 且 `max_abs_error <= max_abs_error_limit`。

## 6.2 性能标准

性能测试在 Ascend 950PR 上执行，先 warmup，再有效采样 50 次以上取平均。任务书给定性能门槛如下：

| case | m | n | side | uplo | 标杆耗时 Avg time |
|------|---|---|------|------|-------------------|
| 1 | 256 | 256 | LEFT | UPPER | 7.53 us |
| 2 | 1024 | 1024 | LEFT | LOWER | 89.27 us |
| 3 | 1024 | 1024 | RIGHT | UPPER | 86.34 us |

优化重点：

- 用 Cube 路径承担主矩阵乘累加，避免纯 Vector 标量循环；
- 利用 `beta == 0/1` 和 `alpha == 0` 分支减少 GM 读取和复数乘法；
- 对紧凑前导维性能 case 使用连续 GM 访问和 tile 级 A/B 面板复用；
- 对 padded 前导维保持正确寻址，性能用例优先覆盖 compact 模式；
- 小矩阵减少启动核数，避免调度开销吞噬计算收益。

## 6.3 自测设计

随任务提供的 `csymm_test.csv` 覆盖 1200 条用例，其中精度 1000 条、性能/内存 200 条。设计验收时需覆盖：

| 类别 | 覆盖内容 |
|------|----------|
| 基础功能 | LEFT/RIGHT 与 UPPER/LOWER 正交组合 |
| 尺寸扫描 | 0、1、小质数、2 的幂、2 的幂 ±1、非对齐值、大尺寸 |
| 标量特殊值 | `alpha/beta` 为 0、1、-1、纯虚数、大值 |
| 矩形矩阵 | 宽矩阵、窄矩阵 |
| 前导维 | `lda/ldb/ldc` 等于最小值和 padding 场景 |
| 输入分布 | 均匀、正态、全零、交替、极值、Inf、NaN |
| 负向用例 | 空指针、非法枚举、非法前导维、负维度 |
| 性能用例 | 任务书 3 条典型性能 case 及扩展性能扫描 |

## 6.4 兼容性分析

`aclblasCsymm` 是 ops-blas 中新增 API。接口声明放入公共头文件 `include/cann_ops_blas.h`，状态码和 `aclblasComplex` 复用 `include/cann_ops_blas_common.h`，不影响现有 `aclblasSsymm` 等接口。实现文件限定在 `blas/symm/arch35/`，后续如扩展其他产品线，应在同一 API 下增加对应 arch 实现，而不是新增私有接口。

# 7 开发交付规划

| 阶段 | 交付件 | 仓库 |
|------|--------|------|
| 设计评审 | `tasklist/08-aclblasCsymm-950/JaneConan/docs/design.md` | cann-ops-competitions |
| 算子实现 | `include/cann_ops_blas.h`、`blas/symm/arch35/` | ops-blas fork |
| 测试代码 | `test/symm/csymm/arch35/`、CSV 用例、README | ops-blas fork |
| 自测报告 | 精度、性能、内存占用和截图 | 验收系统附件 |

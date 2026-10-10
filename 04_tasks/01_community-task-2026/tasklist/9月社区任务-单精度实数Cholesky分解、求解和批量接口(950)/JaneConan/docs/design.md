# 【社区任务】aclsolverSpotrf / Spotrs / Spotri / SpotrfBatched / SpotrsBatched 算子设计文档

| 项 | 内容 |
| --- | --- |
| 文档版本 | v1.0 |
| 作者 | JaneConan |
| 对应任务 | 《Atlas 950 单精度实数 Cholesky 分解、求解和批量接口 任务书》 |
| 目标硬件 | Ascend 950PR（Atlas 950） |
| 工程模式 | ops-solver Host C API + AscendC / CATLASS Kernel 直调 |
| 说明 | 本文档为 5 个接口的统一设计文档；接口、功能、参数、约束与对标 CUDA cuSolver legacy API 一致，精度与性能按任务书 §3 验收 |

---

# 一、需求背景（required）

## 1.1 需求来源

通过社区任务完成开源仓（https://gitcode.com/cann/ops-solver ）算子贡献：在 Ascend 950PR 上新增 5 组单精度实数 Cholesky 族接口（共 7 个函数），语义与参数序列对齐 CUDA cuSolver legacy API，满足任务书规定的功能、精度、性能与确定性验收标准。五个接口作为**同一社区任务**一并交付，不允许拆分验收。

## 1.2 背景介绍

### 1.2.1 算子实现方案概述

- **工程模式**：ops-solver Host C API + AscendC / CATLASS Kernel 直调（非 aclnn 两段式、非 PyTorch 接口）。
- **公开头文件**：`include/cann_ops_solver.h`、`include/cann_ops_solver_common.h`（补充 `aclsolverFillMode_t` 枚举）。
- **接口清单与对标**：

| 交付接口 | 对标接口 | 数学语义 |
| --- | --- | --- |
| `aclsolverSpotrf` / `aclsolverSpotrf_bufferSize` | `cusolverDnSpotrf` / `cusolverDnSpotrf_bufferSize` | Cholesky 分解 `A = L·Lᵀ`（LOWER）/ `A = Uᵀ·U`（UPPER） |
| `aclsolverSpotrs` | `cusolverDnSpotrs` | 基于因子的线性求解 `A·X = B` |
| `aclsolverSpotri` / `aclsolverSpotri_bufferSize` | `cusolverDnSpotri` / `cusolverDnSpotri_bufferSize` | 基于因子的对称矩阵求逆 `A⁻¹` |
| `aclsolverSpotrfBatched` | `cusolverDnSpotrfBatched` | 批量 Cholesky 分解 |
| `aclsolverSpotrsBatched` | `cusolverDnSpotrsBatched` | 批量 Cholesky 求解（`nrhs=1`） |

### 1.2.2 算子现状分析

#### 1.2.2.1 支持的数据类型和数据格式

| 项目 | 说明 |
| --- | --- |
| 数据类型 | FLOAT32（单精度实数） |
| 数据排布 | 列主序（column-major）ND；`lda`/`ldb` 满足 `≥ max(1,n)`，不要求等于 n，支持合法 lda padding |
| 指针归属 | 矩阵、info、workspace 均为 Device 指针；Host 侧仅传标量维数、枚举与 handle |
| 维数类型 | `int`（32-bit，与 CUDA legacy API 一致） |
| 枚举 | `aclsolverFillMode_t`：LOWER=0 / UPPER=1（与 `cublasFillMode_t` 对齐） |
| 批量形态 | `Aarray`/`Barray` 为 Device 上的指针数组（非 `[batch, n, n]` 连续张量） |

#### 1.2.2.2 接口语义与 info 约定

1. **Spotrf**：`LOWER` 时 `A = L·Lᵀ`、`UPPER` 时 `A = Uᵀ·U`；仅处理指定三角、因子原地覆盖该三角；未使用另一半三角作为 internal workspace 可被覆盖（对齐 cuSolver）。`devInfo`：0 成功 / -i 参数非法 / k 第 k 阶顺序主子式不正定（分解止于第 k 行，前 k-1 列因子已算出）。
2. **Spotrs**：解 `A·X = B`（A 为已分解因子，只读），解原地覆盖 B；`devInfo` 仅报参数错误（0 / -i），正定性已在 potrf 阶段保证。
3. **Spotri**：输入三角因子、输出原地覆盖为对称逆矩阵的对应三角；`devInfo`：0 / -i / k（因子的第 k 阶顺序主子式为零 → 奇异）。
4. **SpotrfBatched**：逐矩阵执行与 Spotrf 相同的分解；`infoArray[i]` 逐矩阵独立写（k>0 表示第 i 个矩阵的第 k 阶顺序主子式不正定，其余矩阵照常分解）；参数错误写 `infoArray[0] = -i`。
5. **SpotrsBatched**：逐矩阵求解（仅 `nrhs=1`）；`info` 为标量、仅报参数错误（0 / -i）；正定性检查属 potrfBatched 职责。

通用约定：空问题（`n=0` / `nrhs=0` / `batchSize=0`）成功返回；相同输入、相同 stream 串行多次执行输出（含 info）bit-wise 一致；计算走调用方 stream，禁止无必要 Host 同步；核心计算在 AI Core 完成，不允许 CPU fallback。

#### 1.2.2.3 对标算法流程

五个接口共享**分块（blocked）线性代数骨架**，以 Spotrf 的分块 Cholesky 为核心：

- **Spotrf（分块右看 right-looking Cholesky）**：按列/面板分块循环推进，每个对角块（NB×NB）先分解为三角因子，随后对该面板执行三角求解（TRSM）求得非对角 panel，再对剩余尾部矩阵做对称秩-k 更新（SYRK，`A_next = A_next − panel·panelᵀ`）；重复至全部列处理完毕。
- **Spotrs**：基于因子做前代+回代（两次 TRSM），`nrhs>1` 时以 GEMM 形态批处理多右端项。
- **Spotri**：先对三角因子做三角形求逆（TRTRI），再用 `C = L⁻¹·L⁻ᵀ` 的对称秩-k 更新（LAUUM）得到完整对称逆的指定三角。
- **SpotrfBatched / SpotrsBatched**：逐矩阵独立计算，矩阵级并行映射到各 AI Core；小矩阵走 vector-only 轻量路径，大矩阵复用分块骨架。

---

# 二、需求分析（required）

## 2.1 外部组件依赖

- **CANN 9.0.0 及以上**（与 ops-solver 仓 README 已验证配套版本一致）；ACL / AscendC / CATLASS 工具链。
- **精度 golden**：NumPy / SciPy `float64` 的 `cholesky` / `cho_solve` / `inv`，或等价 LAPACK `dpotrf`/`dpotrs`/`dpotri`。
- **性能金标**：CUDA cuSolver（`bench_result.json` / `gpu_baseline.csv` 预采集参考数据）。
- **自验工具**：ops-solver 仓测试工程、`AscendOpTest`、任务包 `spotrf/`、`spotrs/`、`spotri/`、`spotrfBatched/`、`spotrsBatched/` 自测脚本。

## 2.2 内部适配模块

- **handle / stream 管理**：复用仓内既有 `aclsolverCreate / Destroy / SetStream / GetStream`，计算在调用方 stream 串行执行。
- **公共支撑**：参数校验与错误上报、workspace 计算（`_bufferSize`）、调度配置（tiling）生成、kernel 下发封装、Device 指针数组 staging（batched）。
- **计算原语**：CATLASS 提供的单精度 GEMM / SYRK / TRSM 原语（Cube 侧）+ AscendC Vector 侧三角/对角处理（sqrt、TRTRI、小规模 unblocked 分解）。

## 2.3 需求模块设计

### 2.3.1 Ascend C 算子原型（接口定义）

在 `cann_ops_solver_common.h` 中补充枚举：

```c
typedef enum {
    ACLSOLVER_FILL_MODE_LOWER = 0,  /* 下三角，对齐 CUBLAS_FILL_MODE_LOWER */
    ACLSOLVER_FILL_MODE_UPPER = 1   /* 上三角，对齐 CUBLAS_FILL_MODE_UPPER */
} aclsolverFillMode_t;
```

公开计算接口（统一返回 `aclsolverStatus_t`，语义对齐 cuSolver）：

```c
aclsolverStatus_t aclsolverSpotrf_bufferSize(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSpotrf(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    float *A, int lda, float *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverSpotrs(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
    const float *A, int lda, float *B, int ldb, int *devInfo);

aclsolverStatus_t aclsolverSpotri_bufferSize(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSpotri(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    float *A, int lda, float *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverSpotrfBatched(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    float *Aarray[], int lda, int *infoArray, int batchSize);

aclsolverStatus_t aclsolverSpotrsBatched(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
    float *Aarray[], int lda, float *Barray[], int ldb,
    int *info, int batchSize);
```

参数名、顺序、含义、Device/Host 归属与 cuSolver legacy API 对齐，不擅自增删或调换；维数统一 `int`（32-bit）。

### 2.3.2 参数与错误处理约定

- 参数错误按 LAPACK 约定以负下标上报（`-i`，handle 不计），按签名参数序校验、以首个非法参数为准；`devInfo`/`info` 自身为空指针时仅返回状态码。
- 空问题（`n=0` / `nrhs=0` / `batchSize=0`）成功返回。
- 列主序按 32 位索引寻址；空指针（`n>0` 时 A/Workspace、Batch 指针数组、输出指针）且规模非 0 时报错；`lda < max(1,n)` / `ldb < max(1,n)` 报错。
- `aclsolverSpotrsBatched` 仅支持 `nrhs = 1`（`n=0` 或 `batchSize=0` 时可放宽）；`nrhs ≠ 1` 且 `n>0` 必须报错。

### 2.3.3 算子相关约束与验收规模

- 统一返回 `aclsolverStatus_t`；若需短暂兼容现存 `aclError` 原型，在头文件与文档给出对照表，最终验收以 `aclsolverStatus_t` 公开接口为准。
- 验收规模（实现可声明更大范围）：`n ∈ [1, 4096]`；`nrhs ∈ [1, 128]`；`batchSize ∈ [1, 30000]`；`SpotrsBatched` 仅 `nrhs=1`。
- 不支持 broadcast；不要求图融合；dynamic shape 按本次调用生成调度配置；构造验证 case 内存占用控制在 4G 内。

---

# 三、需求详细设计（required）

## 3.1 调用方式

**Kernel 直调**路径：调用方创建 handle 并设置 stream → （potrf/potri）按 `_bufferSize` 准备 workspace → 调用计算接口（Host 完成参数校验、tiling 生成、kernel 下发）→ 结果与 info 写回 Device；全程走调用方 stream，无必要不 Host 同步。批量接口在 Host 侧把 Device 指针数组（`Aarray`/`Barray`）直接下发给 kernel（pointer array 位于 Device）。

## 3.2 需求总体设计

### 3.2.1 host 侧设计

#### 3.2.1.1 参数校验与错误上报

按 2.3.2 的约定在 kernel 下发前完成全部校验；校验顺序与负 info 下标严格对齐参数签名序（handle、uplo、n、nrhs、A、lda、…）。非法时立即返回 `aclsolverStatus_t` 错误码，仅在 `devInfo`/`info` 可写时写入 `-i`。

#### 3.2.1.2 调度配置（tiling）生成

Host 按本次调用的 `uplo`、规模（`n`/`nrhs`/`batchSize`）、形态（单矩阵 / 批量）生成 `TilingData`，核心字段：

- `uplo`、`n`、`nrhs`、`batchSize`；
- 分块大小 `NB`（面板/对角块边长，针对 950PR Cube 与 UB 容量选取，如 128/256）；
- 分核映射：大矩阵按面板/列块切分给多 AI Core，批量按矩阵序号切分；
- padding 信息：`lda`/`ldb` 与最小合法值之差（随机 `lda` padding 支持）；
- 非正定检测开关、对角线处理模式（sqrt / TRTRI 形状）。

tiling 经 Device 下发；kernel 按 `uplo` 选择 LOWER/UPPER 分支，按规模选择 blocked / unblocked（小矩阵）路径。

#### 3.2.1.3 数据分块与内存规划

- **单矩阵**：依据平台存储层级（GM / L1 / UB）与容量做面板/对角块分块与缓冲复用；原地语义与 workspace 需求在调度阶段统一处理。
- **Spotrf / Spotri 的 workspace**：由 `_bufferSize` 计算 `Lwork`（FLOAT32 元素个数），用于存放 panel 暂存、对角块分解中间量、info 暂存等；`Lwork=0` 时不应下发 workspace。
- **Spotrs**：因子只读、无需 workspace；多右端项（nrhs）在 UB 内分块搬运。
- **批量**：`Aarray`/`Barray` 为 Device 指针数组，kernel 内按 `block_idx` 取对应矩阵指针（指针数组位于 Device，逐核索引）。无必要不在 Host 做 pointer-array staging H2D 重建。

#### 3.2.1.4 分核与任务划分

- **大矩阵（单矩阵 Spotrf/Spotrs/Spotri）**：以多 Cube/Vector 核并行为基础，按面板（列块）循环推进，剩余尾部由尾核处理；小规模调用使用轻量 unblocked 路径规避调度开销。
- **批量（SpotrfBatched/SpotrsBatched）**：以矩阵为并行粒度映射到各核（参考仓内 `cmatinv_batched` 的矩阵级均分策略）：`matPerCore = ceil(batchSize / coreNum)`，余数前几个核多分一个；每核处理连续一段矩阵，核内再按 UB 容量对单矩阵做 panel 切分。

### 3.2.2 kernel 侧设计

#### 3.2.2.1 总体结构

五个接口共享"**分块算法 + 多核协作**"结构，结合 950PR 的 Cube（GEMM/SYRK/TRSM 主算力）与 Vector（对角块 sqrt、TRTRI、小规模 unblocked 分解、三角拷贝）能力实现。数值全程 FLOAT32；确定性按统一约定保证：固定计算顺序（固定块序与累加序）、阶段化同步（SetFlag/WaitFlag）、写者约束（每元素唯一写者）。

计算主循环（以 LOWER 的 Spotrf 为例）：

```
for k = 0, NB, 2*NB, ...:
    # 1) 对角块分解：A[k:k+NB, k:k+NB] -> L11 (unblocked: 逐列 sqrt + 行缩放 + SYRK)
    # 2) panel 三角求解：A[k+NB:, k:k+NB] = A[k+NB:, k:k+NB] · L11^{-T}   (TRSM, Cube/Vector)
    # 3) 尾部对称秩-k 更新：A[k+NB:, k+NB:] -= A[k+NB:, k:k+NB] · A[k+NB:, k:k+NB]^T  (SYRK, Cube)
```

UPPER 为转置对称形式（对角块 `U11·U11ᵀ`、panel `A[k:k+NB, k+NB:] = U11^{-1}·A[k:k+NB, k+NB:]`、尾部 `A[k+NB:, k+NB:] -= A[k:k+NB, k+NB:]ᵀ·A[k:k+NB, k+NB:]`）。

#### 3.2.2.2 各接口要点

- **Spotrf（Cholesky 分解）**
  - LOWER：`A = L·Lᵀ`；对角块 unblocked 分解（逐列 `L[j,j] = sqrt(A[j,j] − Σ L[j,1:j]²)`，`L[i,j] = (A[i,j] − Σ L[i,1:j]·L[j,1:j]) / L[j,j]`），panel 用 TRSM 求 `L21 = A21·L11^{-T}`，尾部 SYRK。
  - 非正定检测：分解对角元时对 `A[j,j] − Σ(…)² ≤ 0` 立即终止，写 `info = j+1`（第 j+1 阶顺序主子式不正定），前 j 列因子保留。
  - workspace 用于 panel 暂存与对角块中间量。

- **Spotrs（求解）**
  - `LOWER`：`L·Y = B`（前代 TRSM，下三角单位/非单位），再 `Lᵀ·X = Y`（回代 TRSM）。
  - `nrhs>1`：以 GEMM 形态批处理多右端（`B` 为 `n×nrhs`，逐 panel 处理右端块），提高 Cube 利用率。
  - 因子只读；`devInfo` 仅报参数错误。

- **Spotri（求逆）**
  - 对 `uplo` 指定侧三角因子做 TRTRI（三角形求逆，对角元素 `1/√` 类处理 + 面板回代）。
  - 对称秩-k 更新 `C = inv(L)·inv(L)ᵀ`（LAUUM，Cube SYRK），结果按 `uplo` 原地覆写为对称逆对应三角。
  - `info`：因子对角元为零（奇异）时写 `k`。

- **SpotrfBatched（批量分解）**
  - 逐矩阵独立执行与 Spotrf 相同算法；矩阵级并行到 AI Core。
  - 小矩阵（`n` 较小，如 `n ≤ 32`）走 vector-only 轻量路径（类 `cmatinv_batched`：核内整矩阵 UB 化、逐列 unblocked 分解、批量 gather/scatter），避免 GEMM 调度开销；大矩阵复用分块 + Cube。
  - `infoArray[i]` 按核/矩阵独立写；参数错误写 `infoArray[0] = -i`。

- **SpotrsBatched（批量求解）**
  - 逐矩阵 TRSM（`nrhs=1`，`B[i]` 为 `n×1` 列向量），矩阵级并行。
  - 仅 vector 前代+回代（单右端无需 GEMM 形态）；`info` 标量仅报参数错误。
  - `nrhs ≠ 1` 在 Host 校验阶段即报错。

#### 3.2.2.3 CATLASS / AscendC 分工

- **Cube（AIC）**：承担 SYRK / TRSM / LAUUM 中的 GEMM 主算力（面板更新、尾部秩-k 更新、批量大矩阵分块）。
- **Vector（AIV）**：对角块 unblocked 分解（sqrt、逐元素缩放与 rank-1 更新）、TRTRI、小矩阵整块处理、三角拷贝（DataCopyPad / Gather / Scatter）、info 写回。
- **协作**：Cube 与 Vector 通过 global memory 与阶段化同步（缺 Cube 资源时 Vector 经 GEMM 软件路径兜底，保证功能完整）。

#### 3.2.2.4 确定性

- 相同输入、相同 stream 串行多次执行，输出（含 `info`）bit-wise 一致。
- 手段：固定块序与累加顺序、固定 UB 划分、固定同步点；不引入基于运行态的动态路径分支（如按负载动态改 NB）；批量逐矩阵处理顺序固定。

### 3.2.3 与标杆（cuSolver）的差异

1. 并行与分块参数按 Ascend 950PR 平台特性定制（算法骨架与对标一致）；不依赖 CUDA。
2. 确定性保证更严格（bit-wise 可复现）。
3. 参数错误上报路径按工程实现约定（负下标 `-i`）处理，不影响成功路径的无多余同步特性。
4. 列主序、Device 指针、原地覆盖、合法 lda padding、批量指针数组形态均与 cuSolver 对齐。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（Atlas 950） | √ |

## 3.4 算子约束限制

1. 数据类型仅 FLOAT32；列主序；`lda`/`ldb ≥ max(1,n)`，支持合法 padding。
2. `uplo` 仅 LOWER/UPPER；Spotrs/Spotri/SpotrsBatched 的 `uplo` 须与分解时一致。
3. 原地语义：Spotrf/Spotri 覆盖指定三角（另一半可被 workspace 破坏）；Spotrs/SpotrsBatched 覆盖 B。
4. 规模约束：`n ∈ [1,4096]`、`nrhs ∈ [1,128]`、`batchSize ∈ [1,30000]`；SpotrsBatched 仅 `nrhs=1`。
5. 空问题成功返回；不支持 broadcast；不提供 CPU fallback。
6. 批量 `Aarray`/`Barray` 为 Device 指针数组。

---

# 四、特性交叉分析

- **handle / stream**：全部接口复用仓内 handle 管理，计算在调用方 stream 串行执行；非默认 stream 下结果正确（自验覆盖）。
- **确定性 × 性能**：确定性为统一要求（含批量逐矩阵）；优化遵循"改写不改序"原则，避免动态分支引入非确定。
- **原地语义 × 工作空间**：`_bufferSize`/`Lwork` 校验与原地覆盖语义在调度阶段统一处理；C 的另一半三角作为 workspace 可被覆盖，验收只比较 `uplo` 指定三角。
- **批量 × 指针数组**：寻址、校验与并行映射按 Device 指针数组形态统一处理；批量判定先核余槽 bit-wise 一致性、再对代表槽逐内容三层判定（参考任务包 `DELIVERY_NOTE`）。
- **列主序 × lda padding**：所有访存按 `a[i + j*lda]` 寻址，`lda > n` 时尾部不越界、不参与计算。
- 不涉及广播、图融合与 dynamic shape 特化（按调用生成 tiling）。

---

# 五、可维可测分析

## 5.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 逐元素混合容差判定（FLOAT32：rtol=2⁻¹⁰≈9.77e-4、atol=2⁻¹⁶≈1.53e-5、matched_ratio≥0.99、max_abs_error≤1e-2 或 32×ULP）；不通过时按 LAPACK 残差判据复核（potrf/spotrs 用 `max(5·ratio_cpu, 3·ratio_cpu_mean)`，potri 用 `max(5·ratio_cpu, 0.1)`）；`info` 语义正确（成功 0 / 非正定 k / 参数错 -i）；合法用例重复执行 bit-wise 一致 | 任务书 §3.2 |
| 性能标准 | 每个验收 case 满足 `T_NPU ≤ T_GPU数据 / 0.35`；NPU 耗时口径为平均单次 kernel 耗时（msprof `OpBasicInfo.csv` 的 `Task Duration(us)` 求平均），对照任务书 §3.3 性能对比表 | 任务书 §3.3 |

> **官方说明（精度验收口径）**：当前任务精度验证环节的逐元素对比在部分极端场景下无法有效衡量误差；最终精度验收以任务书 §3.2.2 的 LAPACK 残差判据为准。任务书将就此刷新，刷新内容会在本讨论区同步。

性能验收核心 case（任务书 §3.3 表，NPU 耗时为 950PR msprof 实测待填）：

| 编号 | 接口 | 规格 | GPU 性能 | NPU 目标 |
| --- | --- | --- | --- | --- |
| P-01 | aclsolverSpotrf | n=1024, uplo=LOWER, lda=n | 0.3927 ms | ≥ 0.35×GPU数据 |
| P-02 | aclsolverSpotrf | n=4096, uplo=UPPER, lda=n | 4.3341 ms | ≥ 0.35×GPU数据 |
| P-03 | aclsolverSpotrf | n=2048, uplo=LOWER, lda=n | 0.7899 ms | ≥ 0.35×GPU数据 |
| P-04 | aclsolverSpotrs | n=1024, nrhs=1, uplo=UPPER | 0.1324 ms | ≥ 0.35×GPU数据 |
| P-05 | aclsolverSpotrs | n=4096, nrhs=32, uplo=LOWER | 3.0109 ms | ≥ 0.35×GPU数据 |
| P-06 | aclsolverSpotrs | n=2048, nrhs=8, uplo=LOWER | 1.1836 ms | ≥ 0.35×GPU数据 |
| P-07 | aclsolverSpotri | n=1024, uplo=LOWER | 1.049 ms | ≥ 0.35×GPU数据 |
| P-08 | aclsolverSpotri | n=4096, uplo=UPPER | 12.3802 ms | ≥ 0.35×GPU数据 |
| P-09 | aclsolverSpotrfBatched | n=32, batchSize=102774, uplo=LOWER | 2.1221 ms | ≥ 0.35×GPU数据 |
| P-10 | aclsolverSpotrfBatched | n=128, batchSize=46256, uplo=LOWER | 10.6212 ms | ≥ 0.35×GPU数据 |
| P-11 | aclsolverSpotrsBatched | n=32, batchSize=99659, nrhs=1, uplo=LOWER | 1.4823 ms | ≥ 0.35×GPU数据 |
| P-12 | aclsolverSpotrsBatched | n=128, batchSize=45897, nrhs=1, uplo=LOWER | 4.7759 ms | ≥ 0.35×GPU数据 |

## 5.2 兼容性分析

新增接口，不涉及既有行为变更；与仓内工程模式（Host C API + Kernel 直调）一致；NPU 与 CUDA 参数序列一致（仅命名前缀 `aclsolver` vs `cusolverDn`、枚举类型 `aclsolverFillMode_t` vs `cublasFillMode_t`，枚举值对齐）；handle/stream 复用既有实现；建议与对照表一并合入 `include/`、`src/`、`test/`、`docs/zh/`、`api_list.md` 并按 README 同步更新。

## 5.3 可维护性与可测试性概述

- 代码按 host/kernel 分层组织，公共逻辑（参数校验、tiling、指针数组处理、info 上报）集中复用；kernel 以"分块骨架 + 主算力 Cube/Vector 分工"统一，便于理解与维护。
- 接口契约与约束在头文件与本文档声明；五个接口作为一组关联 PR 合入，不拆分。
- 自验覆盖（任务书 §3.5）：基础功能（potrf / potrf+potrs / potrf+potri / potrfBatched / potrsBatched）、uplo（LOWER/UPPER）、padding（lda/ldb 最小合法值及 +8/+32）、info（成功 0 / 非正定 k / 非法 -i）、批量边界（batchSize=1、n=1、nrhs=1 合法、potrsBatched nrhs=2 报错）、空问题（n=0/nrhs=0/batchSize=0 成功）、确定性（重复执行 bit-wise 一致）、非默认 stream。
- 精度用例分布（任务包 `cases.json` / `canonical_cases.json`）：A 以 50% 随机 `BᵀB + n·I`（均匀）与 50% 正态同构造生成 SPD，`n` 覆盖 2 的幂与 2 的幂−1；约 10% 用例对角减大数构造非正定用于 info 验证；B 均匀/正态各半，`nrhs` 覆盖 1/8/32/128 及 0；批量 `batchSize` 覆盖 1/8/128/1024/3000 及 0。
- 性能自测按 `PERF_COLLECTION_SUPPLEMENT.md`：每 case 采集 30 次取中位数，msprof `OpBasicInfo.csv` 逐 kernel `Task Duration(us)` 求平均；与任务书性能对比表逐条关联比对 `T_NPU ≤ T_GPU数据/0.35`。

---

# 附录 A：参考资料

- 《Atlas 950 单精度实数 Cholesky 分解、求解和批量接口 任务书》
- CUDA cuSolver legacy API 文档（potrf / potrs / potri / potrfBatched / potrsBatched）
- 生态算子开源精度标准（opbase）：`docs/zh/ops_precision_standard/experimental_standard.md`
- ops-solver 仓（https://gitcode.com/cann/ops-solver ）与 QUICKSTART、DELIVERY_NOTE、PERF_COLLECTION_SUPPLEMENT
- AscendOpTest（https://gitcode.com/HIT1920/AscendOpTest ）
- CATLASS（https://gitcode.com/cann/catlass ）
- 仓内参考实现 `cmatinv_batched`（批量矩阵级并行、vector-only 轻量路径、Kernel 直调样例）

# 附录 B：代码与文档交付目录（建议）

```
include/cann_ops_solver.h
include/cann_ops_solver_common.h        # 补充 aclsolverFillMode_t
src/spotrf/spotrf_host.cpp + spotrf_kernel.cpp
src/spotrs/spotrs_host.cpp + spotrs_kernel.cpp
src/spotri/spotri_host.cpp + spotri_kernel.cpp
src/spotrf_batched/spotrf_batched_host.cpp + spotrf_batched_kernel.cpp
src/spotrs_batched/spotrs_batched_host.cpp + spotrs_batched_kernel.cpp
test/spotrf/ test/spotrs/ test/spotri/ test/spotrf_batched/ test/spotrs_batched/
docs/zh/spotrf.md docs/zh/spotrs.md docs/zh/spotri.md docs/zh/spotrf_batched.md docs/zh/spotrs_batched.md
docs/api_list.md                         # 同步更新接口列表
README.md                                # 同步更新
```

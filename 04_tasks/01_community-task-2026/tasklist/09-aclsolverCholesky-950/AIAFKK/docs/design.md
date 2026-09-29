# 需求背景（required）

## 需求来源

9月社区任务——单精度复数Cholesky分解、求解和批量接口(950)，平台任务 ID `a3c754a14f4340d0b8d7e9e07957f19e`。验收通过后合入 [cann/ops-solver](https://gitcode.com/cann/ops-solver)。

## 背景介绍

### 算子功能

在昇腾 NPU（Ascend 950PR）上提供稠密 Hermitian 正定线性求解与配套批量能力，对标 cuSolver DN legacy API：

| NPU 交付接口 | 对标 CUDA 接口 |
|---|---|
| `aclsolverCpotrf` / `aclsolverCpotrf_bufferSize` | `cusolverDnCpotrf` / `cusolverDnCpotrf_bufferSize` |
| `aclsolverCpotrs` | `cusolverDnCpotrs` |
| `aclsolverCpotri` / `aclsolverCpotri_bufferSize` | `cusolverDnCpotri` / `cusolverDnCpotri_bufferSize` |
| `aclsolverCpotrfBatched` | `cusolverDnCpotrfBatched` |
| `aclsolverCpotrsBatched` | `cusolverDnCpotrsBatched` |

### 现状分析

ops-solver 仓现有 cgetrf/cgetri/sgetrf/sgetri/cmatinv_batched/cgetri_batched/cheevj 七个算子，**无复数 Cholesky 原型**，须全新实现。可复用资产：

- `src/utils/kernel/c64/`：复数（float 实/虚平面分离）kernel 工具库——`gemm.hpp`（CUBE `Mmad` 三乘法复矩阵乘，正/负虚平面组合表达共轭，含分段批量模式 `SetMatrixWithAImagNegSegmented`）、`trsm.hpp`（VEC 三角扫掠 + CUBE 尾随更新的分块 trsm 编舞）、`pad.hpp`（复数平面分离/合并）；
- `src/cgetrf/cgetrf_kernel.cpp`：单融合 kernel + `CrossCoreSetFlag/WaitFlag` 跨核同步的分块分解编排范式（本设计算法骨架参照）；
- `src/cgetri_batched/cgetri_batched_host.cpp`：批量接口 host 侧内部 `aclrtMalloc` 工作区（本仓既有先例，签名无 Workspace 参数的接口据此获得暂存）；
- handle 管理（`aclsolverCreate/Destroy/SetStream/GetStream`）与 `aclsolverStatus_t` 状态码已就绪（`cann_ops_solver_common.h`）。

# 需求分析（required）

## 需求描述

基于 AscendC 实现 `aclsolverCpotrf`、`aclsolverCpotrs`、`aclsolverCpotri`、`aclsolverCpotrfBatched`、`aclsolverCpotrsBatched`（外加 2 个 `_bufferSize`），列主序、Device 指针、真实 `info` 写入、合法 `lda` padding，满足：

- 精度：官方测试包 `verify_accuracy.py` 全量 case 通过。**判定以包内脚本为准**：layer1 逐元素 rtol = atol = 2⁻¹³（HT-14 单套容差；严于任务书 §3.2.1 表列 2⁻¹⁰/2⁻¹⁶，本设计按更严口径评估裕度）、required_matched_ratio = 0.99、max_abs ≤ max(1e-2, 32·ULP(g_low))，实/虚平面分别判定；layer1 不过时残差复核 ε = 2⁻²⁴（包口径；任务书文本 2⁻²³，该不一致在自测报告如实登记）；potrf/potrs 残差阈值 `max(5·ratio_cpu, 3·ratio_cpu_mean)`（批量逐矩阵判定，mean 取本 case 内 k 个代表矩阵的槽位加权平均），potri 单支 `max(5·ratio_cpu, 0.1)`；
- 性能：950PR 上全量用例（五算子 145/176/145/151/151 条）NPU kernel 耗时 ≤ GPU 参考耗时 / 0.35，msprof `OpBasicInfo.csv` 口径；
- 确定性：同输入同 stream 串行多次执行输出（含 info）bit-wise 一致。

## 需求拆解

1. 接口层：`cann_ops_solver_common.h` 补 `aclFloatComplex`、`aclsolverFillMode_t`；7 个公开函数按任务书签名实现，返回 `aclsolverStatus_t`；
2. 功能层：五个接口的 AI Core kernel（禁 CPU fallback），覆盖 n∈[1,4096]、nrhs∈[1,128]、batchSize∈[1,10⁶]、LOWER/UPPER、lda/ldb padding、空问题、非法参数 info=−i、非正定 info=k；
3. 质量层：官方包全量精度自验 + msprof 性能采集 + 自有功能矩阵（确定性/流/边界）；
4. 交付层：设计文档（本文件）、测试代码与 readme、自测报告、私仓代码地址。

# 详细设计（required）

## 算子分析

### 数学公式

`uplo = LOWER`（`UPPER` 为镜像情形）：

```
potrf:  A = L·Lᴴ,  L 下三角、对角元正实数；A、L 均 n×n
        L[j,j] = sqrt(A[j,j] − Σ_{k<j} L[j,k]·conj(L[j,k]))
        L[i,j] = (A[i,j] − Σ_{k<j} L[i,k]·conj(L[j,k])) / L[j,j],  i > j
potrs:  A·X = B  ⇒  X = L⁻ᴴ·(L⁻¹·B)（前代 + 回代）
potri:  A⁻¹ = (L·Lᴴ)⁻¹ = L⁻ᴴ·L⁻¹（三角求逆 trtri + 对称乘 lauum）
```

`uplo = UPPER`：`A = Uᴴ·U`，U 上三角、对角正实；potrs `X = U⁻¹·(U⁻ᴴ·B)`；potri `A⁻¹ = U⁻¹·U⁻ᴴ`。批量接口对 i = 0…batchSize−1 逐矩阵执行同式运算。

### 支持数据类型

COMPLEX64（kernel 内拆 float 实/虚平面计算；对角实数语义=实平面维护、虚平面对角恒 0）；info 为 INT32。

### 支持形状（验收下限，实际同上限）

| 接口 | n | nrhs | batchSize |
|---|---|---|---|
| Cpotrf / Cpotri | [1, 4096] | – | – |
| Cpotrs | [1, 4096] | [1, 128] | – |
| CpotrfBatched | [1, 4096] | – | [1, 10⁶] |
| CpotrsBatched | [1, 4096] | =1（n>0 时 ≠1 报错，含 nrhs=0） | [1, 10⁶] |

### 公开接口（逐字对齐任务书 §2.3，参数名/顺序/归属不变）

```
aclsolverStatus_t aclsolverCpotrf_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    aclFloatComplex *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverCpotrf(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    aclFloatComplex *A, int lda, aclFloatComplex *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverCpotrs(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
    const aclFloatComplex *A, int lda, aclFloatComplex *B, int ldb, int *devInfo);
aclsolverStatus_t aclsolverCpotri_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    aclFloatComplex *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverCpotri(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    aclFloatComplex *A, int lda, aclFloatComplex *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverCpotrfBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
    aclFloatComplex *Aarray[], int lda, int *infoArray, int batchSize);
aclsolverStatus_t aclsolverCpotrsBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
    aclFloatComplex *Aarray[], int lda, aclFloatComplex *Barray[], int ldb, int *info, int batchSize);
```

`cann_ops_solver_common.h` 补充（C 可调用；`aclsolverFillMode_t` 自 `cann_ops_solver.h` 迁入，原文件经 include 保持既有引用兼容）：

```
typedef struct { float real; float imag; } aclFloatComplex;
typedef enum { ACLSOLVER_FILL_MODE_LOWER = 0, ACLSOLVER_FILL_MODE_UPPER = 1 } aclsolverFillMode_t;
```

### 总体架构

**ops-solver Host C API + AscendC kernel 直调**：Host 做参数校验、tiling 生成、workspace 布局、kernel 下发（`kernel<<<numBlocks, ws, stream>>>`，numBlocks 取平台核数、0 防护回退 1、上限 20 对齐 cgetrf 设计）；Kernel 完成全部分解/求解计算。每次计算接口调用下发**一个融合 kernel**（cgetrf 范式），核内以 `CrossCoreSetFlag/WaitFlag` 编舞 VEC（面板/三角扫掠）与 CUBE（Mmad 尾随更新）协同。

**workspace 来源分三类**：
- `Cpotrf`/`Cpotri`：调用方 Workspace（`_bufferSize` 返回元素数，布局见下）；
- `Cpotrs`/`CpotrfBatched`/`CpotrsBatched`：签名无 Workspace 参数，host 侧 `aclrtMalloc`/`aclrtFree` 内部工作区（`cgetri_batched` 同先例；分配在调用方 stream 外完成、不引入 Host 同步；kernel 耗时不含分配时间，与性能口径一致）。

### 数据布局

- 外部：列主序 COMPLEX64，`lda/ldb ≥ max(1,n)`；仅 `uplo` 侧三角参与计算与验收，对侧半三角允许作 kernel 暂存（对齐 cuSolver）。
- kernel 内（分块路径）：计算三角搬至**平面分离布局**——real/imag/neg-imag 三平面，`strideN = align128(max(1,n))`、`M = align16(max(1,n))`（同 cgetrf `A_work`；复用 `SplitRealImag`/`MergeRealImag`/`TransPose` 与 `GenerateGatherTables`）；结束按三角 merge 回原地址。
- 单矩阵 `Cpotrf` Workspace 布局（float 元素计）：`3×M×strideN`（三平面）＋ gather1/2/3 表（uint32 视图）＋ 面板暂存 `align16(n)×NB×2` ＋ tiling/sync（int32/uint32 视图）。n=4096 时三平面 201MB，总 Lwork ≈ 2.52×10⁷ complex64 元素。
- `Cpotrs` 内部工作区：L 三平面 `3×M×strideN` ＋ B 实/虚平面 `2×M×align128(max(1,nrhs))` ＋ tiling/sync。
- 批量（n>32 分块路径）：每矩阵三平面同上，矩阵维连续堆叠；`Aarray`/`Barray` 指针数组先由 kernel 从 Device 拷至工作区（uint64 视图）供逐矩阵寻址。

### aclsolverCpotrf 设计（LOWER 为主；UPPER 见镜像规则）

分块右视 Cholesky，NB 自适应（n≤256: NB=16；n>256: NB=32，批量大 n 档 NB=64，按实测调优）：

```
for j = 0, n, NB:
    L11 = chol(A[j:j+NB, j:j+NB])        # 面板：VEC 核 UB 内无主元 Hermitian 分解
    A21 ← A21·L11⁻ᴴ                      # 右除：L11 显式求逆后转为批量窄 gemm（CUBE）
    A22 ← A22 − A21·A21ᴴ                 # HERK：CUBE Mmad，K 按 limK 累积
```

- 面板（新写 `CHolPanelSolver`）：单 VEC 核 UB 内对 NB×NB 对角块分解（逐列：`d = sqrt(Re a_jj − Σ|L|²)`，列体除以 d；实虚平面 Axpy/Dot 展开，扫掠模式取自 `SolveTrsm`）。`d ≤ 0`（含 NaN/Inf）即记 `info = j+1`，该矩阵停止分解、保留前 k−1 列因子。对角在实平面维护，虚平面对角恒 0。
- 右除：`A21·L11⁻ᴴ = A21·conj(L11⁻¹ᵀ)`——面板阶段顺带产出 `L11⁻ᴴ`（NB×NB，VEC 上三角形求逆+共轭转置），随后 `A21 ← A21 − (A21·L11⁻ᴴ − A21)` 以 CUBE 窄 gemm（M=n−j−NB、N=NB、K=NB）完成；`L11⁻ᴴ` 的共轭由 neg-imag 平面表达。窄 gemm 与后续 HERK 合并 K 段调度，面板延迟被 CUBE 流水隐藏（cgetrf limK 同法）。
- HERK 更新（CMatmulCustom，正/负虚平面组合）：
  - 实部：`A22re −= A21re·A21reᵀ + A21im·A21imᵀ`
  - 虚部：`A22im −= A21im·A21reᵀ − A21re·A21imᵀ`
  - B 操作数 `A21ᴴ` 由平面转置视读 + neg-imag 平面表达共轭。
- UPPER 镜像规则（右视、面板从左上起）：`U11 = chol(A11)`；`U12 ← U11⁻ᴴ·A12`（左除，`U11⁻ᴴ` 同样面板产出后作窄 gemm）；`A22 −= U12ᴴ·U12`（HERK 的 B 侧为 U12 本身共轭转置）。布局与同步协议同 LOWER。
- n≤NB 的矩阵退化为单面板步（无 gemm 阶段），同一 kernel 覆盖，无独立小 n 路径。

### aclsolverCpotrs 设计

`X = L⁻ᴴ·(L⁻¹·B)`，两段均为「VEC 解行块 + CUBE gemm 尾随更新」分块扫掠（复用 `solveTrsm` 编舞骨架；非单位对角：行块求解先做实对角缩放 1/d）：

- 前代 `Y = L⁻¹·B`：自上而下按 blockM 行块求解（VEC），尾随 `B[rest,:] −= L[rest,blk]·Y[blk,:]`（CUBE gemm，共轭由平面/N 转置表达）；
- 回代 `X = L⁻ᴴ·Y`：自下而上镜像；
- nrhs 按 256 列分块（`SolveTrsm::maxN` 同界）；nrhs=1 走窄向量快路径；
- UPPER：`X = U⁻¹·(U⁻ᴴ·B)`，L 换 Uᴴ 视读（平面转置+共轭），编舞相同。

### aclsolverCpotri 设计

`potri = trtri + lauum`（LAPACK 同构，LOWER）：

- trtri（下三角求逆）：右视分块——对角块 NB×NB 在 VEC UB 内直接求逆（含实对角缩放）；非对角块 `T21 = −L11⁻¹·A21·L22⁻¹` 分两步窄 gemm（CUBE）＋行块扫掠（VEC）完成；
- lauum：`A⁻¹ = L⁻ᴴ·L⁻¹` 自底向上分块累积——对角块 `T11ᴴ·T11` VEC 小 herk，非对角 `T21ᴴ·T11` gemm ＋ `T21ᴴ·T21` HERK（CUBE），结构与 potrf 的 HERK 件同源复用；
- UPPER 镜像；n≤NB 退化为单步。Workspace 由 `_bufferSize` 提供（布局同 potrf，另加 trtri 中间块一平面 `M×strideN`）。

### 批量接口设计

**Tier A（n ≤ 32）**：单 VEC kernel，矩阵间零同步。每核认领若干矩阵：从 `Aarray[i]`（列主序原址）整阵读入 UB（≤ 2×32×align32×4B = 16KB/矩阵），实虚分离（向量打散/交织）后直接 unblocked 分解，写回原址；`infoArray[i]` 逐槽写 0/k；非正定仅停该矩阵。批量 potrsBatched（nrhs=1）同档：UB 内前代+回代。

**Tier B/C（n > 32，统一锁步批量调度）**：全部矩阵共享同一分块步进表；按显存预算分 **wave**（每 wave G 个矩阵的平面驻留工作区），单次 kernel 调用内逐 wave 推进：

```
for wave in waves:                      # G 个矩阵的三平面已就位
    staging: G 个矩阵三角 → 平面分离（全体核参与搬运）
    for j = 0, n, NB:
        G 个面板：VEC 上按矩阵并行（面板彼此独立，批量并行度隐藏面板延迟）
        G 个右除：L11inv 批量产出后，segmented gemm（SetMatrixWithAImagNegSegmented）
        G 个 HERK：segmented gemm 尾随更新
    merge: G 个因子按三角写回原址；infoArray 逐槽写回
```

- wave 容量：`G = clamp(⌊32GB / 每矩阵平面字节数⌋, 1, batchSize)`（固定 32GB 预算常量，**不查询运行时显存**——G 是 (n, batchSize) 的确定函数，与确定性声明自洽）；极端组合（n=4096 大 batch）自动退化为小 G 多 wave，覆盖验收范围内全部 n×batch 组合，不设 batch 附加上限。
- 该调度下 FLOP 主体（segmented HERK/TRSM-gemm）连续占用 CUBE，面板 VEC 延迟被 G 路并行吸收——对照门禁：n=512/b=5467 预算 28.1µs/矩阵（≈12.8 TFLOPS 等效），n=4096/b=60 预算 8.3ms/矩阵（≈23 TFLOPS 等效），均在 950PR CUBE FP32 吞吐可达范围（实现期以 P-09/P-10 与全量 case 实测调优 NB/G）。
- potrsBatched（n > 32）：同锁步调度，前代/回代行块扫掠按矩阵并行（VEC）＋segmented gemm 尾随（CUBE）；B 单列直接平面化驻留。
- info：potrfBatched 的 `infoArray` 逐矩阵 0/k（kernel 写）；参数错误 host 侧经微 kernel（tiling 携值）写 `infoArray[0] = −i`，不做 host→device 拷贝；potrsBatched `info` 标量仅 0/−i。

### Host 侧设计（参数校验与错误码，逐条落点）

以 potrf 为例（其余接口同构，逐条对齐任务书 §2.4 各表；参数序号不含 handle）：

| 校验 | 返回 | devInfo（非空时） |
|---|---|---|
| handle == nullptr | ACLSOLVER_STATUS_HANDLE_IS_NULLPTR | 不写 |
| uplo ∉ {0,1} | ACLSOLVER_STATUS_INVALID_ENUM | −1 |
| n < 0 | ACLSOLVER_STATUS_INVALID_VALUE | −2 |
| A == nullptr 且 n>0 | ACLSOLVER_STATUS_INVALID_VALUE | −3 |
| lda < max(1,n) | ACLSOLVER_STATUS_INVALID_VALUE | −4 |
| Workspace == nullptr 且 Lwork>0（仅计算接口；Workspace 为第 5 形参） | ACLSOLVER_STATUS_INVALID_VALUE | −5 |
| devInfo == nullptr | ACLSOLVER_STATUS_INVALID_VALUE | （不可写，仅返回码） |
| n == 0 | SUCCESS | 0 |
| bufferSize 的 Lwork == nullptr | ACLSOLVER_STATUS_INVALID_VALUE | – |

（序号统一为**不计 handle** 的形参序，对齐任务书 §2.1 算法说明 4 与官方包口径：potrf 序为 uplo=1/n=2/A=3/lda=4/Workspace=5/Lwork=6/devInfo=7；potrs 序为 uplo=1/n=2/nrhs=3/A=4/lda=5/B=6/ldb=7；potri 同 potrf。potrsBatched nrhs≠1 且 n>0 → INVALID_VALUE，info=−3；batchSize==0 或 n==0 空问题 → SUCCESS。）参数错误的 devInfo 负值统一经微 kernel 写入（stream 序、无 host 同步）；合法路径 info 由主 kernel 首步清零。

tiling：按本次 (n, nrhs, batchSize, uplo) 生成（dynamic shape）：M/strideN/NB/limK/核分片/批量 wave 表/G——写入 workspace tiling 区（单矩阵同 cgetrf 机制，批量分片结构参照 `CmatinvBatchedTilingData` 的 startOffset/calNum）。

### 确定性

tiling 由 (op, n, nrhs, batchSize, uplo) 唯一决定；kernel 无原子操作、无跨核归约顺序差异；矩阵计算路径仅依赖内容与形状。同输入 ⇒ bit-wise 一致，满足官方包 A0 一致性层（同内容槽位逐位相等）与 §3.5 确定性用例。批量 Tier A/B 的每矩阵独立性与锁步调度均不引入写冲突（infoArray 逐槽、平面逐矩阵隔离）。

### 精度策略

输入为对角占优/良态 HPD（官方 gen_data：diag ≈ n，非对角 |·|≤0.36；或 BᴴB+nI）。分块右视算法 + FP32 平面计算（CUBE FP32 累加）：舍入误差随块内求和长度 O(NB) 与步数 O(n/NB) 增长，预期逐元素相对偏差 ≪ 2⁻¹³ 判据（golden 为 c128 链路降型）；残差 `‖F·Fᴴ−A‖₁/(n‖A‖₁ε)` 预期 O(10¹)，落在 `max(5·ratio_cpu, 3·ratio_cpu_mean)` 内。对角实数语义：实平面单独维护。若个别大 n case 触发 layer1 不过，按 §3.2.2 残差复核并出误差分析报告。

### 性能策略与口径

- FLOP 主体在 HERK/TRSM gemm（CUBE）：n=4096 单矩阵 ≈1.8×10¹¹ 实 FLOP，预算 19.4ms ⇒ ≥9.3 TFLOPS 等效；批量锁步调度把面板延迟摊到 G 路并行，CUBE 占空比逼近 1；
- 搬运：单矩阵大 n 三次 HBM 往返（split/分解/merge）≈201MB×2；批量 wave 内 staging 与计算重叠（跨核流水）；
- Tier A：UB 常驻零同步，102774@n32 目标 <1ms（预算 8.5ms）；
- **口径**：msprof `msprof op --application=...` 采集 `OpBasicInfo.csv`，按 kernel 名取 `Task Duration` 对调用次数求平均（任务书 §3.3 口径）；多 kernel 情况同时报告单次调用各 kernel 平均耗时与合计，达标判定以合计（保守）；`bufferSize` 与 potrf 前置、workspace 分配、host 校验均不计入（§3.3.1）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950 (Ascend 950PR) | √ |

## 算子约束限制

- 仅 COMPLEX64；矩阵须 Hermitian 正定（对角虚部按 0 语义处理）；非正定由 info 报告而非异常返回；
- potrs/potri/potrsBatched 的输入必须是 potrf/potrfBatched 的因子（本接口不校验因子合法性，语义由调用方保证，对齐 cuSolver）；
- potrsBatched 仅 nrhs=1（n>0 时 nrhs≠1 返回 INVALID_VALUE）；
- 未使用半三角可被破坏（验收只比 uplo 侧）；
- 不支持 broadcast、不要求图融合；单 case 输入内存 ≤4G 为构造侧约束，实现侧上限由 HBM 决定（n=4096×batch 63 ≈8.4GB 已验证可容纳）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 官方包 verify_accuracy.py 全量 case 通过（layer1 逐元素 rtol=atol=2⁻¹³、ratio 0.99、max_abs 动态上限；残差复核 potrf/potrs 双支、potri 单支 max(5·ratio_cpu,0.1)；批量逐矩阵判定+case 内槽位加权 mean），info 契约用例全过 | 任务书 §3.2 + 官方包脚本实测口径 |
| 性能标准 | 全量 case NPU kernel 耗时 ≤ GPU 参考耗时/0.35（msprof OpBasicInfo.csv，950PR 实测；bufferSize/前置不计入） | 任务书 §3.3 |
| 确定性 | 同输入重复执行 bit-wise 一致（含 info） | 任务书 §3.2.1.5/§3.5 |

## 测试方案

1. 官方包自验（全量 768 条）：`gen_data.py --canonical … --select all` 造数 → 执行器 `test/*/executor`（读 npz 输入、列主序 flatten(order='F') 上传、调接口、下载、写 `dut_out/<case_id>.npz` 三键：`out32`（potrf/potri 存储侧半三角另侧置 0；potrs 整块解；批量 (batch,n,cols) 三维不压维）/`info`（单算子标量；批量 infoArray 为 (batch,) int32 数组）/`status`（'ok'；info 契约用例预期接口报错时执行器判定返回码符合后仍写 'ok'））→ `verify_accuracy.py --package . --dut-out dut_out`；性能侧执行器逐 case 多次迭代计时产 `my_perf.json` → `verify_perf.py`；
2. 自有功能矩阵（`test/*/`，对齐任务书 §3.5 八类）：基础功能（potrf；potrf+potrs；potrf+potri；potrfBatched；potrsBatched）；LOWER/UPPER 全覆盖；lda/ldb 最小值及 +8/+32 padding；info（成功 0/非正定 k/非法参数 −i）；批量边界（batchSize=1、n=1、nrhs=1、potrsBatched nrhs=2 报错、n=0/nrhs=0/batchSize=0 空问题成功）；确定性（同输入重复执行逐位比对）；非默认 stream 正确性；n 覆盖 2 的幂与 2 的幂−1，nrhs 覆盖 1/8/32/128/0，batch 覆盖 1/8/128/1024/3000/0；
3. 性能：msprof 收敛到单用例采集（避免混叠），OpBasicInfo.csv 汇总，逐 case 与 perf_baseline.json 比对（全量门禁 0.35×）；
4. 回归：`bash build.sh --ops=cpotrf,cpotrs,cpotri,cpotrf_batched,cpotrs_batched --run`。

## 兼容性分析

- 新增算子不改既有接口行为；`aclsolverFillMode_t` 自 `cann_ops_solver.h` 迁至 `cann_ops_solver_common.h`（原头文件包含后者，cheevj 等既有引用不受影响）；
- 现存算子返回 `aclError`，新接口返回 `aclsolverStatus_t`，两套状态码对照表随 `docs/zh/` 各算子文档交付（任务书 §2.2 允许的短暂兼容口径，验收以 `aclsolverStatus_t` 公开接口为准）。

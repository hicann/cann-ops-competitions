# aclblasCsyrk 算子开发设计文档

## 需求背景（required）

### 需求来源

昇腾社区 2026 年 8 月社区任务：基于 ops-blas 开源仓（https://gitcode.com/cann/ops-blas ），
使用 Ascend C 编程语言实现单精度复数（complex64）对称秩-k 更新算子 `aclblasCsyrk`，
适配 Atlas 950PR（arch35 / DAV_3510）。验收通过后合入昇腾算子开源仓。

### 背景介绍

#### aclblasCsyrk 算子功能

`aclblasCsyrk` 计算对称秩-k 更新：

```
C = alpha * op(A) * op(A)^T + beta * C
```

- `trans = ACLBLAS_OP_N` 时：op(A) = A（A 为 n×k），`C = alpha * A * A^T + beta * C`；
- `trans = ACLBLAS_OP_T` 时：op(A) = A^T（A 为 k×n），`C = alpha * A^T * A + beta * C`；
- `trans = ACLBLAS_OP_C` 时：按 ACLBLAS_OP_T 等价处理（**不共轭**）——本算子为对称（非厄米特）运算，共轭转置语义属 `aclblasCherk`；
- C 为 n×n **对称**复数矩阵（`C = C^T`，非厄米特，对角元素虚部不假定任何值）；
  仅 `uplo` 指定的上三角（ACLBLAS_UPPER）或下三角（ACLBLAS_LOWER）被引用并更新，另一三角不被访问、由对称性隐含；
- alpha、beta、A、C 均为单精度复数（complex64，实部/虚部各 float32），列主序存储。

对标 cuBLAS `cublasCsyrk`，语义参考 Netlib `csyrk`（https://www.netlib.org/blas/csyrk.f ）。

#### 现状分析

ops-blas 仓已有同族实数算子 `aclblasSsyrk`（arch35）与 `aclblasCherk`（arch22），
但**尚无 `aclblasCsyrk` 接口声明与实现**，`include/cann_ops_blas.h` 需新增声明。

复数对称 rank-k 与实数 ssyrk 的本质区别：

1. alpha/beta 为复数，A/C 为复数；
2. 复数乘法的实部/虚部耦合（`(a+bi)(c+di)` 产生交叉项）；
3. 无共轭，但运算量是实数的 4 倍（4 个实数 GEMM）。

### 算子功能分析

| 参数 | 参数含义 | 数据类型 | 支持类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | BLAS 句柄 | aclblasHandle_t | - | 非空，绑定 stream | - |
| uplo | 三角方向 | aclblasFillMode_t | UPPER/LOWER | 非空 | - |
| trans | 转置 | aclblasOperation_t | N/T/C | C 等价 T | - |
| n | C 阶数 | int | - | ≥0 | C: n×n |
| k | 内维 | int | - | ≥0 | op(A): n×k |
| alpha | 标量 | const aclblasComplex* | complex64 | 非空 | - |
| A | 输入 | const aclblasComplex* | complex64 | 非空（n,k>0） | n×k 或 k×n |
| lda | A 前导维 | int | - | ≥max(1,n)（N）/ ≥max(1,k)（T/C） | - |
| beta | 标量 | const aclblasComplex* | complex64 | 非空 | - |
| C | 输出 | aclblasComplex* | complex64 | 非空 | n×n |
| ldc | C 前导维 | int | - | ≥max(1,n) | - |

### 边界/异常行为

- `n = 0`：合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`；
- `(alpha=(0,0) 或 k=0) 且 beta=(1,0)`：C 不变，直接返回；
- `alpha=(0,0) 或 k=0` 且 beta≠(1,0)：仅对 uplo 三角执行 `C = beta*C`（beta=(0,0) 时置零）；
- 非法枚举/空指针/前导维不足/负维度：返回 `ACLBLAS_STATUS_INVALID_VALUE`。

## 需求分析（required）

### 需求描述

使用 Ascend C 语言在 Ascend 950PR 上实现 `aclblasCsyrk`，功能/精度/性能与 cuBLAS 对标：

1. 支持 uplo×trans 全枚举（UPPER/LOWER × N/T/C）与 complex64；
2. 精度满足生态算子开源精度标准（complex64 按 FLOAT32 判定，atol=2⁻¹⁶, rtol=2⁻¹⁰, matched_ratio≥0.99, max_abs_error≤1e-2）；
3. 性能验收：NPU kernel 耗时 ≤ 4× H100 GPU 基线（PERF_THRESHOLD=0.25，msprof kernel Task Duration 口径）；
4. 200 条 TC_PF 性能用例与 1000 条精度用例需全量通过。

### 需求拆解

1. host 侧：句柄式 BLAS 接口、参数校验、workspace 规划、三阶段 tiling 计算、kernel 调度；
2. Phase 0（AIV）：复数 A 解交错（Deinterleave）为实矩阵 Ar、Ai；
3. Phase 1（AIC）：4 个实数 GEMM（`Ar*Arᵀ, Ai*Aiᵀ, Ar*Aiᵀ, Ai*Arᵀ`）融合并行；
4. Phase 2（AIV）：三角区域 combine（Q0-Q1 / Q2+Q3 → 复数 C），仅写 uplo 三角；
5. 精度保障：大 k 对角直算（useDirectDiag）、特殊值 host 回退（FLT_MAX 溢出语义）。

## 详细设计（required）

### 算子分析

#### 数学公式

复数对称 rank-k 更新的 4M 实数分解：

```
C = alpha * op(A) * op(A)^T + beta * C,   op(A) = Ar + i*Ai
Q0 = Ar * op(Ar)^T      (实·实)
Q1 = Ai * op(Ai)^T      (虚·虚)
Q2 = Ar * op(Ai)^T      (实·虚)
Q3 = Ai * op(Ar)^T      (虚·实)
Cr = Q0 - Q1            (C 实部)
Ci = Q2 + Q3            (C 虚部)
C  = alpha*(Cr + i*Ci) + beta*C_old
```

4 个 GEMM 均为实数乘累加，复用仓内成熟的 `SyrkGemmKernelImpl`（`common/helper/syrk_gemm_arch35.h`）。
等效计算量 4·n²k，比朴素复数 GEMM（6·n²k）节省 1/3；与
`08-10-矩阵乘系列算子开发` 设计文档中的 4 GEMM + 三角跳过方案一致（优于 3 GEMM + 转置读方案）。

#### 支持数据类型

complex64（float32 实部/虚部）。

#### 支持形状

n∈[1,4096]，k∈[1,4096]，方阵/宽矩形/窄矩形（trans=N 时 k≫n、trans=T 时 n≫k），
含奇数、非 32B 对齐、边界值。

### 算子实现

#### 实现方案总览

三阶段 pipeline（与 `aclblasSsyrk` arch35 同目录、同工程模式）：

```
A (complex, col-major)
  │  Phase 0: csyrk_deinterleave_kernel (AIV, 56 核)
  ▼
Ar, Ai (实矩阵, arLdc×arCols)
  │  Phase 1: csyrk_cube_kernel (AIC, 28 核, 4-quad 融合并行)
  ▼
temp[4n×tempLdc] = [Q0|Q1|Q2|Q3]  (4 个实数 GEMM 结果)
  │  Phase 2: csyrk_combine_kernel (AIV, 56 核)
  ▼
C (complex, 仅 uplo 三角)
```

#### host 侧设计

入口 `aclblasCsyrk` → `LaunchCsyrkKernel`，流程：

1. **参数校验**（`ValidateCsyrkParams`）：uplo/trans 枚举、n/k/lda/ldc 约束、指针非空；
2. **标量准备**（`PrepareCsyrkParams`）：alpha/beta 为 Device 内存，D2H 读回 host；
   特殊值回退判定（`skipTemp`）；
3. **快速返回**：`(alpha=0 或 k=0) 且 beta=(1,0)` 直接 return；
4. **特殊值安全网**：`n*k ≤ 128²` 且含 FLT_MAX 级/Inf/NaN 输入时，host 用 Netlib 参考实现
   计算并写回（`CsyrkTrySpecialValueFallback`），避免 4-GEMM 分解的溢出语义与 golden 不一致；
5. **workspace 规划**（`EnsureDefaultWorkspace`）：
   - 布局 `[Ar | Ai | temp]`；
   - `arLdc = ceil_align(n, 8)`，`tempLdc = ceil_align(n, 8)`（32B 对齐）；
   - `arBytes = 2 * arLdc * arCols * 4`，`tempBytes = 4 * n * tempLdc * 4`；
6. **核数规划**：
   - Phase 0/2 用 AIV 核数（`GetAivCoreCount()`，950PR 为 56）；
   - Phase 1 用 AIC 核数（`GetUsedAicCoreNum`，950PR 为 28）；
7. **三阶段 tiling 计算**（见下）并依次 `<<<numBlocks, nullptr, stream>>>` 启动。

##### 分核策略

- **Phase 0（deinterleave）**：按 A 的列均分——`colStart = blockIdx * ceil(cols/核数)`，
  每核处理连续列段，列内按 512 行分块（DEINT_BLOCK），块内连续读 8B 块 + 向量 DeInterleave；
- **Phase 1（cube）**：28 个 AIC 核按 `blockIdx` 切成 4 个核子集（每子集 7 核），
  各子集并行计算一个 quad（Q0..Q3），消除 4 轮串行 GEMM 的轮间空闲；
  每个 quad 内部沿用 `SyrkGemmKernelImpl` 的 L1 双缓冲 + L0 mmad 循环；
- **Phase 2（combine）**：64×64 三角 tile 按 tile 索引 stride 分核（大 n），
  或按行带分核（小 n）；**对角 tile 的对角行独立按行 stride 分核**（详见 §对角行分核）。

##### tiling 计算

三个 TilingData 结构（`csyrk_tiling_data.h`）：

- `CsyrkDeinterleaveTilingData`：nRows/nCols/lda/outLdc/colsPerCore；
- `CsyrkCubeTilingData`：n/k/arLdc/tempLdc/usedCoreNum/singleCoreM/singleCoreN/
  tileM/tileN/tileKChunk/isTransN/triangleMode；
  - tileM/tileN 默认 128，K chunk 默认 256；
  - L1 预算约束：`l1Budget = 512KB * 9/10`，`maxKChunk` 由
    `l1Budget / (L1_BUF_NUM * 4 * (alignedM + alignedN))` 反推，K chunk 不超过该上限；
- `CsyrkCombineTilingData`：n/ldc/tempLdc/k/arLdc/rowsPerCore/isTransN/
  alphaReal/alphaImag/betaReal/betaImag/uploMode/isAlphaZero/isKZero/isBetaZero/
  isFastCfg/useDirectDiag；
  - `isFastCfg = (alpha==(1,0) && beta==0)`：64×64 单 tile 特化路径；
  - `useDirectDiag = (k > 512)`：大 k 时对角线从 Ar/Ai 直接重算（精度兜底）；
    小 k 时 quad 值已满足容差，跳过昂贵的 Duplicate+ReduceSum（小 shape 性能关键）。

##### 分核/负载均衡优化（关键）

fast 分支 tile-stride 循环原实现为：`for (tileIdx = blockIdx; tileIdx < nBlocks²; tileIdx += blockNum)`，
对角 tile 的扁平索引为 `mb*(nBlocks+1)`。该序列对 `blockNum`（56）取模只落在
`gcd(nBlocks+1, 56)` 个剩余类上：

- `n=3465 → nBlocks=55 → gcd(56,56)=56 → 对角全堆 1 个核`（最病态）；
- `n=1692 → nBlocks=27 → gcd(28,56)=28 → 对角堆 2 个核`；
- 对角行处理（`ProcessDiagRowFast` + `ComputeDirectDiag`，每行 k/1024 轮
  Duplicate+Mul+ReduceSum）代价高，对角 tile 堆核会导致 combine 阶段出现极端负载不均。

**修复**：主循环跳过对角 tile（`mb==jb` 时 `continue`），对角行改用
`for (row = blockIdx; row < n; row += blockNum)` 独立均匀分发到全部 56 核。
对角行间无数据依赖（各自读 Ar/Ai 行、写 C 对角元素），任意 n 都均衡。

#### kernel 侧设计

三个 kernel 均通过 `csyrk_*_kernel_do` 以 `<<<numBlocks, nullptr, stream>>>` 启动。

##### Phase 0：csyrk_deinterleave_kernel（AIV）

- 输入：complex A（列主序，每列 `[re0,im0,re1,im1,...]` 交错）；
- 输出：Ar、Ai 实矩阵（`arLdc × nCols`，行主序存储）；
- 每核处理连续列段；列内按 512 行分块；
- 块内**连续读 cnt 个 8B block**（整列段连续读，避免逐元素 4B 离散读），
  `DeInterleave`（VECTOR 管道）拆分实/虚，分别 Compact 写回 Ar/Ai；
- 核内每块 `PipeBarrier<PIPE_ALL>` 串行（MTE2/vec/MTE3 同步）。

##### Phase 1：csyrk_cube_kernel（AIC）

- 输入：Ar、Ai；输出：temp[4n×tempLdc] = [Q0|Q1|Q2|Q3]；
- 4-quad 融合：`quad = blockIdx / quadCores; coreInQuad = blockIdx % quadCores`，
  每 quad 子集 7 核并行；核数 <4 时回退全核串行 4 轮；
- 每个 quad 用 `SyrkGemmKernelImpl`（`common/helper/syrk_gemm_arch35.h`）：
  L1 双缓冲（MTE1/M 事件）、L0C mmad 循环、triangleMode 跳过非 uplo 侧整块 tile；
- quad 选择：`QuadUsesArLeft(q)= (q==0||q==2)`，`QuadUsesArRight(q)= (q==0||q==3)`；
- trans=N：`left=X(n×k col-major), right=Yᵀ(k×n ND)`；
  trans=T/C：`left=Xᵀ(n×k ND), right=Y(k×n col-major)`；
- temp 布局 `tempLayout = DNExtLayoutPtn(n, tempLdc)`，quad 间距 `n*tempLdc`。

##### Phase 2：csyrk_combine_kernel（AIV）

`CsyrkCombineAIV` 类，Init + Process 两阶段：

- **Init**：UB buffer 规划。fast 路径 64×64 单 tile 用
  `crBuf/ciBuf/q2Buf/q3Buf/outBuf`（各 64×64×4B）+ 对角直算的 `arBuf/aiBuf/dSumReBuf/dSumImBuf`
  （各 1024×4B）；慢路径 64×32 用 `cInBuf/reBuf/imBuf/t1Buf/t2Buf/outReBuf/outImBuf/outBuf`。
- **Process**：fast/slow 两分支；每分支又分 tile-stride（大 n）与 row-band（小 n）两分发；
  - `ProcessHalfFast(iBase,jBase,rows,cols)`：非对角 64×64 块，
    4 条 `DataCopyPad` 读 Q0..Q3 的 rows×cols 块（一次 8 对齐 bulk + 0..7 行 strips），
    `Sub(crUb, Q0, Q1)` + `Add(ciUb, Q2, Q3)`，`Interleave` 交错成复元素，1 次 `DataCopyPad` 写回 C；
  - `ProcessDiagRowFast(iAbs, jBase, len)`：对角行，读 4 块 len 个元素、
    Sub/Add、写回；`overwriteDiag && useDirectDiag` 时用 `ComputeDirectDiag` 重算对角元素；
  - `ComputeDirectDiag(iAbs)`：从 Ar/Ai 按 k/1024 轮 Duplicate+Mul+ReduceSum
    重算 `C[i,i] = Σ(ar²-ai²) + i·Σ(2·ar·ai)`，大 k 精度兜底；
    trans=N 读 Ar/Ai 行（arLdc 步长），trans=T/C 读列（连续）；
  - `ProcessHalf/ProcessRowStrip/ProcessDiagRow`：慢路径（alpha/beta 复数缩放），
    读 Q0..Q3 + 旧 C（beta≠0），复数 alpha/beta 缩放（4 Muls + Sub/Add），Interleave 写回。

#### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（arch35 / DAV_3510） | √ |

#### 算子约束限制

1. 仅支持 complex64（单精度复数）；
2. C 为对称（非厄米特）矩阵，对角元素虚部不置零（与 cherk 区别）；
3. OP_C 按 OP_T 等价处理（不共轭）；
4. 仅更新 uplo 指定三角，另一三角由对称性隐含、不被访问。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | complex64 按 FLOAT32 判定：atol=2⁻¹⁶, rtol=2⁻¹⁰, matched_ratio≥0.99, max_abs_error≤1e-2 或 32×ULP | 生态算子开源精度标准 |
| 性能标准 | NPU kernel 耗时 ≤ 4× H100 GPU 基线（PERF_THRESHOLD=0.25，msprof kernel Task Duration 口径） | 任务书 §3.3 |

### 自测设计

- **精度**：1000 条精度用例（TC_L0/TC_SQ/TC_AB/TC_WS/TC_TH/TC_LD/TC_FL/TC_CV/TC_ED/TC_EX 等类别），
  ops-blas C++ GTest（`build/test/syrk/csyrk/csyrk_test`）加载 `csyrk_test.csv`，
  由 `verify_accuracy.py` 解析逐条 PASS/FAIL；
- **性能**：200 条 TC_PF 性能用例，`verify_performance.py` 执行 GTest TC_PF + msprof 采集，
  按 kernel Task Duration 与 `gpu_baseline.csv` 比对；
- **内存**：单用例 host 侧 ≤512MB（workspace 布局按此预算设计）。

### 兼容性分析

新算子，不涉及兼容性分析。接口声明放入 `include/cann_ops_blas.h`，与其他产品线共用
同一 `aclblasCsyrk` API，无 950PR 私有平行接口。

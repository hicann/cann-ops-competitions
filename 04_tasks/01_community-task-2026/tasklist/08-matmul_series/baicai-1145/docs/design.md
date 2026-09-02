# 需求背景（required）

## 需求来源

昇腾社区任务 2026「矩阵乘系列算子开发」（cann-competitions `04_tasks/01_community-task-2026`）：在 ops-blas 仓 `experimental/` 目录新增 5 个 complex64 BLAS-3 算子。验收：1587 条官方用例精度满足生态算子开源精度标准、整体性能 ≥ 0.8× A100 cuBLAS（逐用例）。

## 背景介绍

### 现状分析

ops-blas 仓 `experimental/` 交付路径中 **complex64 BLAS-3 尚为空白**（主树已有 arch22 complex 先例，见下表第四行）。最接近的四个参照：

| 参照 | 现状 | 对本任务的可用性 |
|---|---|---|
| `blas/herk/arch35/cherk` | complex64 cherk，4M 分解（AIV deinterleave → 4×AIC GEMM → AIV combine），1094 行 | **arch35 = 950PR/950DT 专属**：Cube 层裸 tensor API（`te::Copy/MakeTensor/L0` 手动管理）A2 编译不过；DeInterleave/Interleave 指令仅 Atlas 350 支持（CANN 文档核实）——算法结构可移植，实现层全部需重写 |
| `blas/symm/arch22/ssymm` | float32 ssymm，A2 可用 | 展示 Matmul 高阶 API 正确用法（`MatmulImpl` + `GetMMConfig<CONFIG_MDL>`）；但 host 编排仅覆盖 float/row-major，complex 需自建 |
| `blas/gemm3m/arch35/` | sgemm3m 3M 分解 | README 明确 A2/A3 不支持；算法结构可移植 |
| `blas/hemm/arch22`（!207，2026-08-20 合入） | arch22 `aclblasChemm` 复数 Hermitian 乘法功能实现（910b3 29/29 PASS） | 同硬件复数 Cube 路径先例，佐证 A2 方案可行；面向主树 aclblas 接口功能对齐，未含逐用例 0.8×A100 性能对标与本任务的 experimental 交付形态，与本设计互补 |

结论：**A2 上的 complex64 Cube 计算路径需自建**——复数分解流水线（取 arch35 cherk 的算法结构）+ Matmul 高阶 API（取 arch22 ssymm 的 Cube 调用方式）。

### 任务量化

1587 条官方用例（任务书 test_script）：

| 算子 | 用例数 | 维度范围 | 枚举组合 |
|---|---|---|---|
| chemm | 318 | m,n ∈ [0,13377] | side{L,R} × uplo{U,L} |
| cher2k | 327 | n,k ∈ [0,13377] | uplo × trans{N,C} |
| cherk | 308 | n,k ∈ [0,16384] | uplo × trans{N,C} |
| csymm | 318 | m,n ∈ [0,13377] | side{L,R} × uplo |
| csyrk | 316 | n,k ∈ [0,16384] | uplo × trans{N,T} |

用例分级：TC_L0（基础）/ TC_SQ（方阵 1→16384 含奇数边界）/ TC_AB（α/β 特殊值：0、1、-1、复数、大值、纯虚）/ TC_FW/TH（fat/thin）/ TC_LD（ld padding）/ TC_FL（填充模式）/ TC_CV（中等尺寸全组合）/ TC_ED（零维、null 指针、Hermitian 对角线）/ TC_PF（性能 100 条/算子）。

性能门禁代表性基线（A100 cuBLAS）：

| 算子/用例 | A100 ms | A100 GFLOPS |
|---|---|---|
| chemm 2048² R/L | 4.337 | 15847 |
| cher2k 2048² L/C | 4.156 | 16536 |
| cherk 2048² L/C | 2.024 | 16973 |
| csymm 2048² R/L | 4.363 | 15752 |
| csyrk 2048² L/T | 2.009 | 34203 |

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 5 个 complex64 BLAS-3 算子（列主序）：chemm / cher2k / cherk / csymm / csyrk，在 Atlas A2 上满足精度（opbase experimental_standard）与性能（0.8×A100 逐用例）门禁，合入 `ops-blas/experimental/`。

## 需求拆解

1. 接口与语义对齐 netlib BLAS（数值语义权威：netlib `.f` 参考实现）
2. host 侧参数校验（枚举/维度/ld，对照任务书约束表逐条）
3. A2 Cube 计算路径：复数→实数分解（4M，预留 3M 升级）+ Matmul 高阶 API
4. 精度：complex128 CPU golden + FP32 混合容差（实/虚部分开）
5. 性能：逐用例 ≥ 0.8×A100，附 roofline 论证与上板 derisk 计划
6. 1587 条官方用例全量通过（对接 verify_accuracy.py / verify_performance.py）

# 详细设计（required）

## 算子分析

### 数学公式

| 算子 | 公式 | 枚举 | 标量 | 输出 |
|---|---|---|---|---|
| chemm | L: C=αAB+βC；R: C=αBA+βC（A Hermitian） | side, uplo | α,β 复数 | m×n 一般 |
| cher2k | N: C=αABᴴ+conjg(α)BAᴴ+βC；C: C=αAᴴB+conjg(α)BᴴA+βC | uplo, trans | α 复数, β 实数 | n×n Hermitian |
| cherk | N: C=αAAᴴ+βC；C: C=αAᴴA+βC | uplo, trans | α,β 实数 | n×n Hermitian |
| csymm | L: C=αAB+βC；R: C=αBA+βC（A 对称） | side, uplo | α,β 复数 | m×n 一般 |
| csyrk | N: C=αAAᵀ+βC；T: C=αAᵀA+βC | uplo, trans(=T) | α,β 复数 | n×n 对称 |

### 支持数据类型

输入/输出 complex64（`complex<float>`），标量按算子为复数或实数（见上表）。

### 支持形状

列主序，不支持 broadcast。API 侧 m/n/k ≥ 0（=0 快速成功）；任务书用例域 m,n,k > 0，上限 16384。

### 语义细则（golden 严格按 netlib `.f`）

- Hermitian（chemm/cherk/cher2k）：A 仅引用 uplo 三角，对角线虚部不被引用（视为实数）；C 对角线虚部必须为 0（输出清零）
- 对称（csymm/csyrk）：仅引用 uplo 三角，无共轭
- `beta=0` 时不读 C（C 可含 NaN）；`k=0` 或 `α=0` 退化为 β·C
- csyrk 的 trans 取 N/T（无 C，转置不共轭）

### 接口设计

#### Host API 原型（5 算子）

对齐 ops-blas 仓 `experimental/` 现有 aclblas 接口风格（handle + 枚举 + 设备侧标量指针 + 显式 ld）：

```cpp
// chemm / csymm：C = alpha*A*B + beta*C（A 为 Hermitian / 对称，side/uplo 控制）
aclblasStatus_t aclblasChemm(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasSideMode_t side,
    int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);
aclblasStatus_t aclblasCsymm(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasSideMode_t side,
    int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);

// cherk：C = alpha*A*A^H + beta*C，alpha/beta 为实数
aclblasStatus_t aclblasCherk(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const aclblasComplex* A, int lda,
    const float* beta, aclblasComplex* C, int ldc);

// cher2k：C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C，alpha 复数、beta 实数
aclblasStatus_t aclblasCher2k(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, const float* beta, aclblasComplex* C, int ldc);

// csyrk：C = alpha*A*A^T + beta*C，alpha/beta 复数
aclblasStatus_t aclblasCsyrk(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);
```

#### 参数说明

| 参数 | 方向 | 类型 | 说明 |
|---|---|---|---|
| uplo | 属性 | aclblasFillMode_t | ACLBLAS_UPPER / ACLBLAS_LOWER，A 仅引用对应三角 |
| side | 属性 | aclblasSideMode_t | LEFT / RIGHT（仅 chemm / csymm） |
| trans | 属性 | aclblasOperation_t | cherk / cher2k：N / C；csyrk：N / T |
| m / n / k | 属性 | int | ≥ 0；=0 快速成功；上限 16384（官方用例域） |
| alpha / beta | 输入 | 设备侧标量指针 | 类型见原型；`beta=0` 时不读 C |
| A / B / C | 输入 / 输出 | aclblasComplex* | 列主序；A 为 Hermitian / 对称矩阵，仅引用 uplo 三角 |
| handle | 输入 | aclblasHandle_t | aclblas 句柄（承载 stream） |

ld 约束（逐条校验，非法 → `ACLBLAS_STATUS_INVALID_VALUE`）：

| 算子 | lda | ldb | ldc |
|---|---|---|---|
| chemm / csymm（side=L） | ≥ max(1,m) | ≥ max(1,m) | ≥ max(1,m) |
| chemm / csymm（side=R） | ≥ max(1,n) | ≥ max(1,m) | ≥ max(1,m) |
| cherk / cher2k / csyrk（trans=N） | ≥ max(1,n) | （cher2k）≥ max(1,n) | ≥ max(1,n) |
| cherk / cher2k / csyrk（trans=T/C） | ≥ max(1,k) | （cher2k）≥ max(1,k) | ≥ max(1,n) |

#### 错误码

| 场景 | 返回值 |
|---|---|
| 成功 | ACLBLAS_STATUS_SUCCESS（0） |
| handle 为空 | ACLBLAS_STATUS_HANDLE_IS_NULLPTR（9） |
| uplo / side / trans 非法枚举、m/n/k < 0、ld 越界、维度 > 0 时矩阵指针为 NULL | ACLBLAS_STATUS_INVALID_VALUE（3） |
| 资源申请失败 | ACLBLAS_STATUS_ALLOC_FAILED（2） |

#### Tiling 结构（host / kernel 共享定义）

```cpp
/** Phase 0 deinterleave（complex → Ar/Ai）tiling。 */
struct DeinterleaveTilingData {
    uint32_t rows;        // 逻辑行数（trans=N 时为 n，trans=T/C 时为 k）
    uint32_t cols;        // 逻辑列数
    uint32_t lda;         // 输入 complex 矩阵前导维度（float 视图 stride = 2*lda）
    uint32_t rowsPerCore; // 每核处理行数（CeilDiv(rows, aivCoreNum)）
};

/** 算子类型：驱动 combine 的 ± 组合公式与标量通路。 */
enum CombineOpType : uint32_t { COMBINE_OP_CSYRK = 0, COMBINE_OP_CHERK = 1,
    COMBINE_OP_CSYMM = 2, COMBINE_OP_CHEMM = 3, COMBINE_OP_CHER2K = 4 };
/** 输出写回模式。 */
enum CombineOutputMode : uint32_t { COMBINE_OUT_TRIANGULAR = 0, COMBINE_OUT_DENSE = 1 };

/** Phase 2 combine tiling。 */
struct CombineTilingData {
    uint32_t m, n, ldc;     // 输出矩阵（syrk 族 m=n）
    uint32_t tempLdc;       // GEMM temp 前导维度（CeilAlign(m, 16)）
    uint32_t rowsPerCore;   // 每核处理行数
    float alphaReal, alphaImag, betaReal, betaImag; // 标量（cherk / cher2k 的 beta 虚部恒 0）
    uint32_t uploMode;      // ACLBLAS_UPPER(121) / ACLBLAS_LOWER(122)
    uint32_t opType;        // CombineOpType：驱动 ± 组合公式
    uint32_t outputMode;    // CombineOutputMode：三角写回（syrk 族）/ 全矩阵（symm/hemm）
};
```

#### Kernel 入口

```cpp
// AIV，Phase 0：complex GM → 紧凑 Ar/Ai
extern "C" __global__ __aicore__ void blas_deinterleave_kernel(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, const DeinterleaveTilingData tiling);

// AIV，Phase 2：t1..t4（± 组合 × alpha + beta*C）→ 输出；t5..t8 仅 cher2k 使用
extern "C" __global__ __aicore__ void blas_combine_kernel(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4,
    GM_ADDR t5, GM_ADDR t6, GM_ADDR t7, GM_ADDR t8,
    GM_ADDR c, const CombineTilingData tiling);
```

Phase 1（4×/8× 实 GEMM）走 Matmul 高阶 API：`MatmulImpl` + `GetMMConfig<CONFIG_MDL>`，tiling 编译期静态化，运行时仅设置 OrgShape / SingleShape 与多核平铺参数（mBlocks×nBlocks ≤ cubeCoreNum）。

#### Workspace 布局

```
wsSize = AlignUp512( 2 × rowsA×colsA×4B               // Ar/Ai 紧凑实/虚分离
                   + 4 × tempLdc×n×4B )               // t1..t4（cher2k 为 8 块 temp）
tempLdc = CeilAlign(m, 16)
```

- α=0 或 k=0 快速路径（skipTemp）→ wsSize=0，跳过 Phase 0/1，仅执行 β·C
- 矩阵与标量由调用方持有；workspace 由框架按需申请（512B 对齐）
- tiling key：isAlphaZero / isKZero / isBetaZero / uplo / trans 位编 key（见 host 侧设计节）

## 算子实现

### 实现方案选型：为什么不能照抄现有代码

如背景节所述：arch35 的算法结构可移植但实现层（裸 tensor API、DeInterleave 指令）A2 不可用；arch22 只有 Cube 调用方式可借。**本设计 = arch35 的"算法流水线" + arch22 的"Cube 调用方式"，在 A2 上组装仓内尚不存在的 complex64 Cube 路径。**

### 核心方案：4M 分解 + A2 Matmul 高阶 API

以 cherk trans=N（C=αAAᴴ）为例，其余算子同构：

```
A = Ar + i·Ai                    （Phase 0: AIV deinterleave，64×64 tile；
                                   DeInterleave 指令仅 Atlas 350 →
                                   改用字节粒度 strided DataCopyPad(ISASI) 抽实/虚部）
t1 = Ar·Arᵀ   t2 = Ai·Aiᵀ        （Phase 1: 4 次 AIC Matmul，实数 fp32）
t3 = Ai·Arᵀ   t4 = Ar·Aiᵀ
Cr = α·(t1+t2)                   （Phase 2: AIV combine：± 组合 × α，
Ci = α·(t3−t4)                      + β·C_old，仅写 uplo 三角，
C  = Cr + i·Ci                       对角线虚部清零，LOWER 对角块
                                     整列写+非 uplo 恢复）
```

**升级路线（3M）**：Karatsuba 形式 P1=Ar·Br, P2=Ai·Bi, P3=(Ar+Ai)(Br+Bi)，Cr=P1−P2, Ci=P3−P1−P2。GEMM 量 4→3（−25%）。numpy 量化验证（`verify_3m.py`）：两分解常规用例均满足 FP32 混合容差；3M 在极端抵消用例下相对误差放大。**决策：4M 主推（实现简单、与流水线一一对应），3M 为性能预留**（csyrk 等紧用例不足时启用，代价是 K 拼接 3k 的预处理 GM 写与 tile 上限验证）。

### host 侧设计

**校验**（对齐仓内 `ACLBLAS_STATUS_*` 错误码体系）：
- 枚举：uplo ∈ {121,122}、side ∈ {L,R}、trans ∈ {N/T/C 按算子}，非法 → INVALID_VALUE
- 维度 m/n/k ≥ 0（=0 快速成功）；指针：α/β 非空，维度>0 时矩阵指针非空
- ld 逐条校验：chemm/csymm side=L → lda≥max(1,m)、side=R → lda≥max(1,n)、ldb/ldc≥max(1,m)；cherk/cher2k/csyrk trans=N → lda/ldb≥max(1,n)、trans=T/C → lda/ldb≥max(1,k)、ldc≥max(1,n)

**参数准备**：device 读 α/β → 置 isAlphaZero/isKZero/isBetaZero；归一化 trans（cherk 的 C≡T）。

**快速路径**：α=0 或 k=0 时跳过 Phase 0/1：β==1 直接返回；β==0 写 0 并恢复非 uplo 三角；否则仅 `C=β·C`。

**workspace**：`2×Ar/Ai + 4×temp`（temp 行宽 CeilAlign(n,16)），512B 对齐；tiling：deinterleave/combine 各一份 AIV tiling（按行分核）+ GEMM 多核平铺（mBlocks×nBlocks ≤ cubeCoreNum，最大化利用率）。

**tilingkey 规划**：isAlphaZero/isKZero/isBetaZero/uplo/trans 编入 tiling key，kernel 侧分支跳过对应阶段。

**Column-Major helper**：ld 寻址 + 三角读取 index 映射（uplo 半边遍历），独立可测模块。

### kernel 侧设计

**Phase 0 deinterleave（AIV）**：GM complex64（float stride=2·lda）→ UB → 抽实/虚（DataCopyPad 字节粒度：blockLen=4B、srcStride=8B）→ GM 紧凑 Ar/Ai（行宽=rows）。多核按行分块（rowsPerCore=CeilDiv(rows,aivCoreNum)）。

**Phase 1 4×GEMM（AIC，Matmul 高阶 API，arch22 模板）**：

```cpp
using MatmulTypeFp32 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulShapeParams shape{M, N, K, baseM, baseN, baseK};
constexpr MatmulConfig config = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shape, func, bias);
constexpr MatmulApiStaticTiling tiling = GetMatmulApiTiling<...>(config);
using MatMul = MatmulImpl<MatmulTypeFp32 ×4, tiling>;
matmulObj.Init(nullptr, &pipe);
matmulObj.SetOrgShape(M,N,K);  matmulObj.SetSingleShape(M,N,K);
matmulObj.SetTensorA(gmA, isTransA);  matmulObj.SetTensorB(gmB, isTransB);
matmulObj.IterateAll(outGM, 0);       matmulObj.End();
```

- 4 次独立 GEMM，α=1/β=0 传入（标量处理全留 Phase 2，一处收敛）
- 转置经 `SetTensorA/B` 第二参表达；arch35 gemm fixpipe 输出 (a·b)ᵀ 的操作数交换问题在 Matmul API 下重新推导（上板首日验证项）

**Phase 2 combine（AIV）**：t1..t4 ± 组合 × α → β·C_old 累加 → uplo 三角写回（非对角块 2D copy；LOWER 对角块整列写+未写部分恢复）→ Hermitian 对角线虚部清零（t3−t4 浮点残差必须清）。

### 分核策略

Phase 0/2 按 C 行数均分 AIV 核（rowsPerCore + 前 remainder 核 +1）；Phase 1 mBlocks×nBlocks 枚举 ≤ cubeCoreNum 且最大化面积利用率（同 arch35 cherk）。小矩阵（m·n < 阈值）走 AIV 退化路径避免 Cube 启动开销。

### 逐算子差异（共同基座上的增量）

| 算子 | 增量点 |
|---|---|
| csyrk（首发） | 基座本身：4M + combine（无共轭；对角线不清虚部） |
| cherk | + 共轭（Aᴴ）→ t 矩阵转置组合；对角线虚部清零；α,β 实数通路 |
| csymm | side 分支；B 引入（deinterleave 双矩阵）；combine 8 项实 GEMM 归约（α 复数展开）；输出全矩阵 |
| chemm | csymm + Aᴴ |
| cher2k | 两个 rank-k 项 + conjg(α) 第二项 → 8 次实 GEMM；β 实数；对角线清零 |

开发顺序：csyrk → cherk → csymm → chemm → cher2k（由简到繁，基座直接复用）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas 800I/T A2 | √ |

## 算子约束限制

- 仅 complex64；不支持 broadcast；列主序
- A 对称/Hermitian 时仅引用 uplo 三角（另一三角不读不写）
- 维度上限按官方用例最大 16384

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | complex64 实/虚部分别按 FP32 混合容差：\|actual−golden\| ≤ 2⁻¹⁶ + 2⁻¹⁰·\|golden\|；matched_ratio ≥ 0.99；max_abs_error ≤ 1e-2 或 32·ULP；非 uplo 三角 EXACT；Hermitian 对角线 beta=0 时 \|imag\| ≤ 2⁻¹⁶ | opbase experimental_standard.md（任务书引用） |
| 性能 | 逐用例 ≥ 0.8× A100 cuBLAS（`matmul_series_gpu_perf.csv`，1587 条基线） | 任务书 |

## 测试体系

1. **本地基座**（已完成）：`gen_golden.py` 官方 CSV → complex128 参考（netlib 语义，vs scipy BLAS 交叉验证 320/320 PASS）；4M/3M 分解 numpy 验证 ALL PASS；本地判定器 `verify_local.py`（官方同款混合容差+三角 EXACT）对 817 条小中尺寸用例 807 PASS / 0 FAIL（5 条 EXTREME 用例的 ±FLT_MAX 溢出 pattern 依赖求和顺序，留上板对真实 golden 裁定）
2. **官方验收**：1587 条 CSV → 仓库 `build.sh --ops=<op>` GTest + `verify_accuracy.py`（精度）/ `verify_performance.py`（0.8×A100）
3. **opbase AscendOpTest**：精度标准复现路径（上板后）
4. 用例分级全覆盖：TC_L0/TC_SQ/TC_AB/TC_FW/TH/TC_LD/TC_FL/TC_CV/TC_ED/TC_PF

## 性能可行性论证

- 2048² chemm 需 ~45 TFLOPS 实数等效（4M）；3M ~34 TFLOPS
- A2 Cube FP32 峰值 ~64 TFLOPS → 需利用率 53%（4M）~70%（3M）
- **roofline 逐用例建模**（`perf_model.py`，30 条 TC_PF 代表用例全扫）：square/fat 类 GEMM 用例只需 23–41% Cube 利用率即达标（余量 ×1.6–3.7）；风险集中在带宽受限的 thin/fat 小 k 用例（combine 阶段占比 63–100%，保守 200GB/s 假设下 chemm/csymm 500×13377 余量 ×0.94–0.99）——缓解依赖 HBM 实测带宽（若 ~1TB/s 则转安全）与 combine/GEMM 流水重叠；3M 分解对带宽型风险用例无效（GEMM −25% 但前置读 +25%），优先级降后
- csyrk square 2048 是 GEMM 类最紧用例（需 41% 利用率）
- **上板 derisk 计划（第一实验）**：① 实测 HBM 带宽；② sgemm（Matmul API，ND fp32）square 1024/2048/4096 吞吐校准利用率假设；③ DataCopyPad strided 抽取实测带宽；④ 小尺寸 launch 固定开销

## 兼容性分析

新算子（`experimental/`），不涉及存量接口兼容。

---

（附：开发排期——本地基座已完成：设计文档、4M/3M numpy 验证、complex128 golden（873 用例 npz）、host 骨架（clang 语法+校验 smoke 全过）、kernel 起草（API 已逐个过 CANN 文档核对）。上板阶段：编译→sgemm derisk→精度对齐→性能调优→1587 全量→自测报告。）

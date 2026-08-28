# aclblasCherk 算子设计文档

## 需求背景（required）

### 需求来源

昇腾 CANN 社区任务——8月社区任务-aclblasCherk算子开发（A2A3），要求在 Atlas 800I A2/A3 系列产品（arch22 架构，Ascend 910B3）上使用 Ascend C 编程语言开发单精度复数（complex64）Hermitian 秩-k 更新算子 `aclblasCherk`，与 cuBLAS `cublasCherk` 功能对齐，精度对标 Netlib `cherk` 参考实现。

### 背景介绍

#### cherk 算子数学语义

cherk（Complex Hermitian Rank-K Update）是 BLAS Level 3 标准接口，计算 Hermitian 矩阵的秩-k 更新：

- `trans = ACLBLAS_OP_N` 时：`C = alpha * A * A^H + beta * C`，A 为 n×k 复矩阵
- `trans = ACLBLAS_OP_C` 时：`C = alpha * A^H * A + beta * C`，A 为 k×n 复矩阵

其中 A^H 为共轭转置，alpha/beta 为实数标量，C 为 n×n Hermitian 复矩阵（对角虚部强制为 0）。

#### arch22 硬件特性

Atlas 800I A2（910B3，arch22/DAV-220）具备以下核心特性：

- **40 个 AIV（Vector）向量核**，无 AIC（Cube）矩阵乘引擎
- 每核 192KB Unified Buffer（UB），支持 SIMD 向量运算
- 向量指令宽度 64 floats/cycle，支持 FMA（Axpy: a*x+y 融合指令）
- GM 带宽约 1TB/s（聚合），UB 访问延迟约 1~2 cycle

#### arch35 参考实现分析

上游仓已有 arch35（Ascend 950PR）cherk 实现，采用 **4M 分解**策略：

```
4M 分解 (trans=N): C = α·A·A^H + β·C
  A·A^H = (Ar·Ar^T + Ai·Ai^T) + i·(Ai·Ar^T - Ar·Ai^T)
  Phase 0: Deinterleave A → Ar, Ai (AIV-only)
  Phase 1: 4 real GEMM via Cube engine (AIC×4)
  Phase 2: Combine/Scale/Hermitian (AIV-only)
```

arch35 利用 **Cube 矩阵乘引擎**完成 4 个实数 GEMM，这是 arch22 所不具备的硬件能力。arch22 必须使用 **向量 SIMD** 实现等效的 GEMM 计算。

## 需求分析（required）

### 需求描述

在 arch22 平台上实现 `aclblasCherk` 接口，满足：

1. 功能与 cuBLAS `cublasCherk` 完全对齐
2. 精度满足生态算子开源精度标准（rtol=2^-10, atol=2^-16, matched_ratio≥0.99）
3. 性能达标（n=1024 k=1024 UPPER N ≤ 472us, n=2048 k=2048 UPPER N ≤ 1674us, n=1024 k=1024 LOWER C ≤ 311us）

### 需求拆解

1. **Host 侧**：参数校验、快速返回（n=0 / alpha=0+k=0+beta=1 / alpha=0+beta缩放）、tiling 计算、kernel 启动
2. **Kernel 侧**：4M 分解 + A 列缓存 + SIMD 向量计算 + DMA 批量传输 + K-blocking + ping-pong 双缓冲
3. **测试侧**：CSV 驱动 GTest，1200 条测试用例，精度 golden 由 cblas 生成

## 详细设计（required）

### 算子分析

#### 数学公式

```
trans=N: C[i][j] = alpha * sum_k(A[i][k] * conj(A[j][k])) + beta * C[i][j]   (i ≤ j for UPPER, i ≥ j for LOWER)
trans=C: C[i][j] = alpha * sum_k(conj(A[k][i]) * A[k][j]) + beta * C[i][j]   (i ≤ j for UPPER, i ≥ j for LOWER)
```

对角元素虚部强制置 0（Hermitian 性质）。

#### 4M 分解

将复数 cherk 分解为 4 个实数 syrk/GEMM 操作：

```
trans=N:
  A·A^H = (Ar + j·Ai)(Ar - j·Ai)^T
         = Ar·Ar^T + Ai·Ai^T + j·(Ai·Ar^T - Ar·Ai^T)
  Cr = alpha·(Ar·Ar^T + Ai·Ai^T) + beta·Cr_old
  Ci = alpha·(Ai·Ar^T - Ar·Ai^T) + beta·Ci_old

trans=C:
  A^H·A = (Ar^T - j·Ai^T)(Ar + j·Ai)
         = Ar^T·Ar + Ai^T·Ai + j·(Ar^T·Ai - Ai^T·Ar)
  Cr = alpha·(Ar^T·Ar + Ai^T·Ai) + beta·Cr_old
  Ci = alpha·(Ar^T·Ai - Ai^T·Ar) + beta·Ci_old
```

其中 Ar、Ai 为 A 的实部、虚部矩阵（n×k for trans=N, k×n for trans=C）。

#### 4M 分解优势分析

| 指标 | 直接复数计算 | 4M 分解 |
|------|------------|---------|
| 每元素运算量 | 6 real FMA + 2 add/sub = 8 ops | 4 real FMA = 4 ops |
| GatherMask/ki | 需要（deinterleave） | 不需要（数据已实数） |
| 标量 GM 读取/ki | 需要（xR, xI） | 不需要（从 UB 读取） |
| SIMD 指令数/tile | 8 (muls_v+add_v+sub_v) | 4 (Axpy FMA) |
| PIPE_BARRIER/tile | 4~6 | 2 |
| 理论 FLOPS | 3·n²·k | 2·n²·k (1.5x 更少) |

### 算子实现

#### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    aclblasCherk (Host)                       │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ 参数校验  │→│ 快速返回  │→│ Tiling   │→│ Kernel启动    │ │
│  │ (§2.4)  │  │ (§2.1.5) │  │ 计算     │  │ (4M+cache)   │ │
│  └─────────┘  └──────────┘  └──────────┘  └──────┬───────┘ │
│                                                    │         │
└────────────────────────────────────────────────────┼─────────┘
                                                     │
                    ┌────────────────────────────────▼─────┐
                    │        cherk_kernel (AIV×40)          │
                    │                                       │
                    │  ┌─────────────────────────────────┐ │
                    │  │  Phase 0: Deinterleave A → Ar,Ai │ │
                    │  │  (DMA load + GatherMask → GM)    │ │
                    │  └──────────────┬──────────────────┘ │
                    │                 │                     │
                    │  ┌──────────────▼──────────────────┐ │
                    │  │  Phase 1: 4M Rank-K Update        │ │
                    │  │  (A列UB缓存 + SIMD Axpy +        │ │
                    │  │   K-blocking + ping-pong)         │ │
                    │  └──────────────┬──────────────────┘ │
                    │                 │                     │
                    │  ┌──────────────▼──────────────────┐ │
                    │  │  Phase 2: Combine + Scale +      │ │
                    │  │  Hermitian (Cr=t1+t2, Ci=t3-t4, │ │
                    │  │  对角虚部置0)                     │ │
                    │  └─────────────────────────────────┘ │
                    └───────────────────────────────────────┘
```

#### Host 侧设计

##### 参数校验

```
handle == nullptr     → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
uplo 非法枚举          → ACLBLAS_STATUS_INVALID_ENUM
trans == ACLBLAS_OP_T → ACLBLAS_STATUS_INVALID_VALUE
trans 非法枚举         → ACLBLAS_STATUS_INVALID_ENUM
n < 0 || k < 0        → ACLBLAS_STATUS_INVALID_VALUE
lda < max(1,n)/max(1,k) → ACLBLAS_STATUS_INVALID_VALUE
ldc < max(1,n)        → ACLBLAS_STATUS_INVALID_VALUE
alpha/beta == nullptr → ACLBLAS_STATUS_INVALID_VALUE
A == nullptr && n>0 && k>0 → ACLBLAS_STATUS_INVALID_VALUE
C == nullptr && n>0   → ACLBLAS_STATUS_INVALID_VALUE
```

##### 快速返回

```
n == 0                                → SUCCESS (no-op)
(alpha == 0 || k == 0) && beta == 1   → SUCCESS (no-op)
alpha == 0 && beta != 1               → 仅 beta 缩放 C (host 直接返回)
```

##### Tiling 策略

- **核间分配**：按列均分，每核处理 n/coreNum 列
- **核内分块**：TILE_SIZE=64 元素/tile，UB 中缓存 A 列数据
- **K-blocking**：K_BLOCK=64，一次加载 64 个 k 值的 A 数据

##### Tiling 数据结构

```cpp
struct CherkTilingData {
    uint32_t uplo;    // 0=LOWER, 1=UPPER
    uint32_t trans;   // 0=NO_TRANS, 1=CONJ_TRANS
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldc;
    float alpha;
    float beta;
    uint32_t coreNum;
    uint32_t kBlock;   // K-blocking size
};
```

#### Kernel 侧设计

##### Phase 0: Deinterleave

```
输入: A (complex64, n×k 或 k×n, 列主序)
输出: Ar, Ai (float32, n×k 或 k×n, 列主序, 存于 GM workspace)

流程:
  for each tile (rows × cols, 64×64):
    1. gm_to_ub_align: 加载 A tile (complex interleaved) 到 UB
    2. GatherMask(pattern=1): 提取实部 → Ar tile
    3. GatherMask(pattern=2): 提取虚部 → Ai tile
    4. ub_to_gm_align: 存储 Ar, Ai 到 GM
```

##### Phase 1: 4M Rank-K Update

核心计算阶段，对 4 个实数 syrk/GEMM 操作分别累加到 Cr 和 Ci：

```
for each column j (assigned to this core):
    for each i-tile [ib, ib+TILE):
        zero accumulator accR[ib], accI[ib]

        for each k-block [ki_base, ki_base+K_BLOCK):
            ┌─ Ping buffer A: DMA load Ar/Ai 列段 to UB (ping) ──┐
            │                                                     │
            │  while DMA loading:                                 │
            │    compute previous k-block from pong buffer        │
            │    for each ki in block:                             │
            │      x = Ar[j][ki] from UB (cached)                 │
            │      s = alpha * x                                   │
            │      Axpy(accR[ib], Ar[ib..ib+TILE][ki], s)        │ ← FMA
            │      Axpy(accI[ib], Ai[ib..ib+TILE][ki], s)        │ ← FMA
            │    (independent, no barrier)                         │
            │    PIPE_BARRIER(V)                                   │
            │    Axpy(accR[ib], Ai[ib..ib+TILE][ki], s_Ai)       │ ← FMA
            │    Axpy(accI[ib], Ar[ib..ib+TILE][ki], -s_Ai)      │ ← FMA
            │    PIPE_BARRIER(V)                                   │
            └─────────────────────────────────────────────────────┘
            swap ping/pong buffers

    merge: load C from GM, C = beta*C + alpha*acc, zero diag imag, store C
```

**关键优化点**：

1. **A 列 UB 缓存**：每个 ki 只做 1 次 DMA 加载全列到 UB，所有 i-tile 从 UB 读取（消除标量 GM 读取）

2. **K-blocking + ping-pong**：一次加载 K_BLOCK 个 k 值，ping/pong 双缓冲重叠 DMA 与计算

3. **跨 tile barrier 消除**：对同一 ki，所有 i-tile 的 Axpy 操作互不依赖（不同 accR[ib]），无需中间 barrier

4. **Axpy FMA 指令**：`dst += scalar * src` 一条指令完成乘加，理论吞吐量 2 FLOPS/cycle

##### Phase 2: Combine + Scale + Hermitian

```
for each column j:
    for each i-tile:
        1. gm_to_ub_align: 加载 C[ib..ib+TILE][j] (complex interleaved) 到 UB
        2. Cr = beta * C_real + alpha * accR    (SIMD: muls_v + add_v)
        3. Ci = beta * C_imag + alpha * accI    (SIMD: muls_v + add_v)
        4. 对角虚部置 0: if (i == j) Ci = 0
        5. ub_to_gm_align: 存储 C 回 GM
```

#### UB 内存布局

```
192KB UB 分配 (n ≤ 2048):

┌──────────────────────────────┬──────────┐
│ Buffer          │ Size      │ Offset   │
├─────────────────┼──────────┼──────────┤
│ aCplxBuf (ping)  │ TILE×2×4 │ 0        │  512B
│ aRealPing         │ n×4      │ 512      │  8KB
│ aImagPing         │ n×4      │ 8.5K     │  8KB
│ aCplxBuf (pong)  │ TILE×2×4 │ 16.5K    │  512B
│ aRealPong         │ n×4      │ 17K      │  8KB
│ aImagPong         │ n×4      │ 25K      │  8KB
│ allAccR           │ n×4      │ 33K      │  8KB
│ allAccI           │ n×4      │ 41K      │  8KB
│ tmp1              │ TILE×4   │ 49K      │  256B
│ tmp2              │ TILE×4   │ 49.3K    │  256B
│ cInBuf            │ TILE×2×4 │ 49.5K    │  512B
│ cOutBuf           │ TILE×2×4 │ 50K      │  512B
│ Total             │          │          │ ~51KB
└──────────────────┴──────────┴──────────┘
```

#### 多核负载均衡

UPPER 三角形：列 j 有 j+1 个元素，后列工作量大。
LOWER 三角形：列 j 有 n-j 个元素，前列工作量大。

策略：**交错列分配**（round-robin），使每核的工作量趋于均匀。

```
Core 0: 列 0, 40, 80, ...
Core 1: 列 1, 41, 81, ...
...
Core 39: 列 39, 79, 119, ...
```

#### 计算流程图

```
┌─────────────────────── Host ────────────────────────┐
│                                                      │
│  aclblasCherk(handle, uplo, trans, n, k,            │
│              alpha, A, lda, beta, C, ldc)           │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ 参数校验   │───→│ 快速返回  │───→│ 读取     │       │
│  │ §2.4     │    │ §2.1.5   │    │ alpha/beta│      │
│  └──────────┘    └──────────┘    └────┬─────┘       │
│                                       │              │
│                              ┌────────▼─────┐        │
│                              │ Tiling 计算   │        │
│                              │ (coreNum,    │        │
│                              │  K_BLOCK)    │        │
│                              └────────┬─────┘        │
│                                       │              │
│                    ┌──────────────────▼──────────┐   │
│                    │ cherk_kernel_do(A, C, tiling)│   │
│                    └──────────────────┬──────────┘   │
│                                       │              │
│                    ┌──────────────────▼──────────┐   │
│                    │ aclrtSynchronizeStream()     │   │
│                    └─────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────── Kernel (AIV × 40) ───────────────────┐
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Phase 0: Deinterleave A → Ar, Ai                │ │
│  │                                                   │ │
│  │  ┌─────────┐   ┌──────────┐   ┌──────────┐      │ │
│  │  │ DMA load │──→│GatherMask│──→│ Store GM │      │ │
│  │  │ A tile   │   │→Ar, Ai  │   │ workspace│      │ │
│  │  │(complex) │   │(deintl) │   │          │      │ │
│  │  └─────────┘   └──────────┘   └──────────┘      │ │
│  └─────────────────────────────────────────────────┘ │
│                          │                            │
│                          ▼                            │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Phase 1: 4M Rank-K Update (核心计算)             │ │
│  │                                                   │ │
│  │  for j = colStart → colEnd:                      │ │
│  │    ┌────────────────────────────────────────┐    │ │
│  │    │ for ib = 0 → iCnt step TILE:           │    │ │
│  │    │   zero accR[ib], accI[ib]              │    │ │
│  │    │                                        │    │ │
│  │    │   for ki_block = 0 → k step K_BLOCK:  │    │ │
│  │    │     ┌──────────────────────────────┐  │    │ │
│  │    │     │ Ping: DMA load Ar/Ai 列段→UB │  │    │ │
│  │    │     │ Pong: compute prev block     │  │    │ │
│  │    │     │   (Axpy FMA, 跨tile无barrier)│  │    │ │
│  │    │     └──────────────────────────────┘  │    │ │
│  │    │     swap ping/pong                    │    │ │
│  │    └────────────────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────┘ │
│                          │                            │
│                          ▼                            │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Phase 2: Combine + Scale + Hermitian            │ │
│  │                                                   │ │
│  │  for ib = 0 → iCnt step TILE:                    │ │
│  │    ┌─────────┐   ┌──────────┐   ┌──────────┐    │ │
│  │    │ DMA load │──→│ SIMD     │──→│ DMA store │    │ │
│  │    │ C tile  │   │β·C+α·acc │   │ C→GM     │    │ │
│  │    │(from GM)│   │diag=0   │   │          │    │ │
│  │    └─────────┘   └──────────┘   └──────────┘    │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---------------|---------|
| Atlas 800I/T A2 | √ |
| Atlas 800I A3 | √ |
| Ascend 950PR (arch35) | 已有实现 |

### 算子约束限制

| 约束项 | 内容 |
|--------|------|
| trans | 仅支持 OP_N 和 OP_C，不支持 OP_T |
| alpha/beta | 实数标量（float），Device 内存指针 |
| A | complex64，Device 内存，列主序 |
| C | complex64，Device 内存，原地覆写，列主序 |
| uplo | 仅更新指定三角，另一三角不访问 |
| 对角虚部 | 强制置 0 |
| n=0 | 合法 quick return |
| 非连续 Tensor | 不支持 |

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|---------|
| 精度标准 | rtol=2^-10(9.77e-4), atol=2^-16(1.53e-5), matched_ratio≥0.99, max_abs_error≤max(1e-2, 32×ULP) | 生态算子开源精度标准 |
| 性能标准 | n=1024 k=1024 UPPER N ≤ 472us; n=2048 k=2048 UPPER N ≤ 1674us; n=1024 k=1024 LOWER C ≤ 311us | 任务书 §3.3 |
| 测试方法 | CSV 驱动 GTest，1200 条用例，golden 由 cblas 生成 | 任务书 §3.5 |

### 兼容性分析

接口声明复用 `include/cann_ops_blas.h` 中已有 `aclblasCherk` 声明，与 arch35（Ascend 950PR）共用同一 API。arch22 实现与 arch35 实现共存于 `blas/herk/` 目录，通过 CMake 架构目录（arch22/arch35）自动选择。

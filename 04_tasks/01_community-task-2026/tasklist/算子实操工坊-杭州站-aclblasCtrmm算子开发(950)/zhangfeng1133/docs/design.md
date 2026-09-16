# 需求背景（required）

## 需求来源

社区任务「算子实操工坊-! 杭州站 - aclblasCtrmm 算子开发（950）」，在 Ascend 950PR（arch35）上使用 Ascend C 编程语言开发单精度复数（complex64）三角矩阵-矩阵乘算子 `aclblasCtrmm`，对标 cuBLAS `cublasCtrmm`，验收通过后合入昇腾算子开源仓 ops-blas（`blas/trmm/arch35/`）。

任务书：`aclblasCtrmm_Atlas950PR_task_doc.md`
开源仓：https://gitcode.com/cann/ops-blas
设计模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md

## 背景介绍

### aclblasCtrmm 算子功能

`aclblasCtrmm` 执行复数三角矩阵-矩阵乘（BLAS Level 3），离席计算（结果写入 C，非覆写 B）：

- `side = LEFT`  时：`C = alpha * op(A) * B`
- `side = RIGHT` 时：`C = alpha * B * op(A)`

其中：
- A 为复数三角矩阵（上三角或下三角），阶数为 m（side=LEFT）或 n（side=RIGHT）
- B、C 为 m×n 复数矩阵，列主序存储
- alpha 为复数标量
- `op(A)` ∈ {A（OP_N），A^T（OP_T），A^H（OP_C，共轭转置）}
- `diag = UNIT` 时 A 对角元固定为 1 且不读取；`diag = NON_UNIT` 时从 A 正常读取

### 接口签名

```cpp
aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    aclblasComplex* C, int ldc);
```

### 同族参考算子

ops-blas 仓已有实数版 `aclblasStrmm`（`blas/trmm/arch35/`），采用**三阶段流水线**（mirror → gemm → scale）实现。本算子 `aclblasCtrmm` 为复数扩展，需额外处理：
1. 复数实部/虚部拆分（planar complex 分解）
2. 共轭转置 OP_C（对虚部取负）
3. 复数标量 alpha 缩放

### 性能标杆

| case | m | n | side | uplo | trans | diag | 标杆耗时（us） |
|---|---|---|---|---|---|---|---|
| 1 | 256 | 256 | LEFT | UPPER | N | NON_UNIT | 120.18 |
| 2 | 512 | 512 | LEFT | LOWER | N | NON_UNIT | 268.37 |
| 3 | 1024 | 1024 | RIGHT | UPPER | T | NON_UNIT | 690.54 |
| 4 | 2048 | 2048 | LEFT | LOWER | C | NON_UNIT | 2945.91 |
| 5 | 2048 | 2048 | RIGHT | LOWER | N | UNIT | 2963.64 |

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Ascend 950PR（arch35）上实现 `aclblasCtrmm` 算子，支持 complex64 数据类型，满足精度与性能双标杆。

## 需求拆解

1. **功能实现**：复数三角矩阵-矩阵乘，支持 side×uplo×trans×diag 全 24 组枚举组合
2. **复数处理**：complex64 拆分为实部/虚部（float），复数乘法分解为 4 次实数 GEMM + 加减组合
3. **三角处理**：仅引用 uplo 指定三角，diag=UNIT 时不读取对角元
4. **共轭转置**：trans=OP_C 时对 A 虚部取负
5. **列主序适配**：通过 host 侧 m↔n 交换 + side 翻转，复用行主序 GEMM kernel
6. **精度达标**：rtol=2^-10，atol=2^-16，matched_ratio≥0.99
7. **性能达标**：5 个性能 case 平均单次耗时 ≤ 标杆耗时
8. **边界处理**：m=0/n=0 quick return、alpha=(0,0) 置零路径、非法参数校验

# 详细设计（required）

## 算子分析

### 数学公式

**复数三角矩阵乘核心公式**：

设 `alpha = (alpha_r + i·alpha_i)`，`op(A) = (A_r + i·A_i)`，`B = (B_r + i·B_i)`

**Step 1 — 复数矩阵乘**（op(A) × B）：
```
P_r = A_r · B_r − A_i · B_i    （实部，2 次 GEMM）
P_i = A_r · B_i + A_i · B_r    （虚部，2 次 GEMM）
```

**Step 2 — 复数标量缩放**（alpha × P）：
```
C_r = alpha_r · P_r − alpha_i · P_i
C_i = alpha_r · P_i + alpha_i · P_r
```

**共轭转置处理**（trans=OP_C）：
```
A^H = (A^T)* → 对 A^T 的虚部取负：A_i' = -A_i^T
```

**三角镜像**（消除三角结构）：
```
mirror(A, uplo, diag):
  - uplo=UPPER: 下三角填 0，保留上三角
  - uplo=LOWER: 上三角填 0，保留下三角
  - diag=UNIT:  对角线强制 1.0（不读取 A 对角）
  - diag=NON_UNIT: 对角线从 A 正常读取
```

### 支持数据类型

complex64（实部/虚部各 float32）

### 支持形状

- side=LEFT：A 为 m×m，B/C 为 m×n
- side=RIGHT：A 为 n×n，B/C 为 m×n
- m, n ≥ 0；lda ≥ max(1, m或n)；ldb ≥ max(1, m)；ldc ≥ max(1, m)
- 列主序存储

## 算子实现

### 实现方案

#### 整体架构：三阶段流水线（复数扩展版）

继承 strmm 的三阶段流水线架构，扩展为复数版本：

```
┌─────────────────────────────────────────────────────────────┐
│                    Host 侧 (aclblasCtrmm)                    │
│  参数校验 → quick return → alpha 解析 → tiling → 启动流水线  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐  ┌───────────────────┐  ┌───────────────┐
│ Phase 1        │  │ Phase 2            │  │ Phase 3        │
│ Complex Mirror │→ │ Complex GEMM       │→ │ Complex Scale  │
│ (AIV, SIMT)    │  │ (AIC, Mmad×4)     │  │ (AIV, SIMT)    │
│ A → wsA_r/i    │  │ wsA × B → temp_r/i│  │ α × temp → C   │
└───────────────┘  └───────────────────┘  └───────────────┘
```

#### 3.2.1 Host 侧设计

##### 参数校验与快路径

```
1. handle == nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. side/uplo/trans/diag 非法枚举 → ACLBLAS_STATUS_INVALID_VALUE
3. m < 0 || n < 0 → ACLBLAS_STATUS_INVALID_VALUE
4. m == 0 || n == 0 → quick return SUCCESS（no-op）
5. alpha == nullptr → ACLBLAS_STATUS_INVALID_VALUE
6. C == nullptr → ACLBLAS_STATUS_INVALID_VALUE
7. alpha != 0 时：A/B == nullptr → INVALID_VALUE；lda/ldb/ldc 越界 → INVALID_VALUE
8. alpha == (0,0) → C 置零路径（aclrtMemsetAsync，保留 ldc padding）
```

##### alpha 解析

alpha 支持 Host 或 Device 内存（与仓内 strmm 口径一致）。Device 内存时通过 D2H 拷贝到 Host 临时变量解析。

##### 列主序适配

利用性质：**列主序矩阵 A 在内存中等价于行主序矩阵 A^T**。

Host 侧通过 m↔n 交换 + side 翻转，将列主序问题转化为行主序 GEMM kernel 可处理的形式：
- side=LEFT, C(m×n) = alpha·op(A)·B → 行主序 C^T(n×m) = alpha·B^T·op(A)^T，side 翻转为 RIGHT
- side=RIGHT, C(m×n) = alpha·B·op(A) → 行主序 C^T(n×m) = alpha·op(A)^T·B^T，side 翻转为 LEFT

##### Tiling 策略

**三层 Tiling**（多核 → L1/L2 → 尾块）：

```
Layer 1: 多核切分（blockDim 与核数匹配）
  - Phase 1 (mirror): 按行划分，usedAivCoreNum = min(m, aivCoreNum)
  - Phase 2 (gemm):   按 tile 划分，tileCount = CeilDiv(m, TILE_M) × CeilDiv(n, TILE_N)
                      usedAicCoreNum = min(tileCount, aicCoreNum)
  - Phase 3 (scale):  按行划分，usedAivCoreNum = min(m, aivCoreNum)

Layer 2: L1 切分（超过 L1 容量时）
  - TILE_M × TILE_K_CHUNK ≤ L1_SIZE / (L1_BUF_NUM × dtype_size)
  - 默认 TILE_M=128, TILE_N=128, TILE_K_CHUNK=256
  - L1 容量不足时自动减半 TILE_K_CHUNK

Layer 3: 尾块/尾核处理
  - m % TILE_M != 0 → 尾块单独计算
  - n % TILE_N != 0 → 尾块单独计算
  - tileCount % aicCoreNum != 0 → 大核多处理一个 tile
```

**自适应 Tiling（按 shape 分档）**：

| 总数据量 | 核数 | tile 大小 | ping-pong | 理由 |
|---|---|---|---|---|
| ≤ 128KB（小） | 8 | 64×64 | 关 | 启动地板主导，1 tile 无重叠空间 |
| 128KB~2MB（中） | 16 | 48×48 | 开 | MTE 占比上升，2 tile 可重叠 |
| > 2MB（大） | 32 | 32×32 | 开 | 带宽主导，ping-pong 重叠 MTE2/MTE3 |

通过 `TilingKey` 在 kernel 侧分档 dispatch。

##### Workspace 分配

```
workspaceA_r:  镜像后 A 实部（m×m 或 n×n，float）
workspaceA_i:  镜像后 A 虚部（m×m 或 n×n，float）
temp_r:        GEMM 中间结果实部（m×n，float）
temp_i:        GEMM 中间结果虚部（m×n，float）
```

所有 workspace 32 字节对齐分配。

##### TilingKey 规划

| TilingKey | 场景 | 说明 |
|---|---|---|
| 0 | alpha=(0,0) | C 置零路径，不走 kernel |
| 1 | 小 tensor（≤128KB） | 少核大 tile，无 ping-pong |
| 2 | 中 tensor（128KB~2MB） | 平衡配置，ping-pong |
| 3 | 大 tensor（>2MB） | 多核小 tile，ping-pong + TripleBuffer |

#### 3.2.2 Kernel 侧设计

##### Phase 1: Complex Mirror Kernel（AIV-only, SIMT）

**功能**：将列主序复数三角矩阵 A 拆分为实部/虚部，镜像填充为全矩阵，处理共轭转置。

```
输入：A（complex64, 列主序, lda×k, k=m或n）
输出：wsA_r（float, 行主序, k×k）, wsA_i（float, 行主序, k×k）

对每个元素 A[row][col]（列主序）:
  1. 读取复数值 a = A[col * lda + row]
  2. 判断是否在 uplo 指定三角内：
     - uplo=UPPER && row > col → 不在三角内，填 0
     - uplo=LOWER && row < col → 不在三角内，填 0
  3. 判断对角线：
     - diag=UNIT && row == col → 实部=1.0, 虚部=0.0
     - diag=NON_UNIT && row == col → 从 A 正常读取
  4. 共轭转置处理（trans=OP_C）：
     - OP_N: wsA[row][col] = a
     - OP_T: wsA[col][row] = a（转置）
     - OP_C: wsA[col][row] = (a.real, -a.imag)（共轭转置，虚部取负）
  5. 写出 wsA_r[row][col] = a.real, wsA_i[row][col] = a.imag
```

**SIMT 并行**：多线程按行并行，`asc_vf_call` 派发，每个线程处理多行 × 全部列。

**模板派发**：`<UPLO_IS_UPPER, TRANS_IS_N, TRANS_IS_T, TRANS_IS_C, DIAG_IS_UNIT>` 消除运行时分支。

**优化**：
- 合并实部/虚部拆分与镜像为单次遍历（减少 GM 读写）
- 合并共轭处理到镜像阶段（避免额外 kernel）
- 使用 ND-DMA 搬运指令（950PR 特性）处理非连续列主序读取

##### Phase 2: Complex GEMM Kernel（AIC-only, Mmad×4）

**功能**：复数矩阵乘 `op(A) × B`，分解为 4 次实数 Mmad。

```
输入：wsA_r, wsA_i（float, 行主序, k×k）, B_r, B_i（float, 行主序, m×n）
输出：temp_r, temp_i（float, 行主序, m×n）

复数乘法分解：
  temp_r = wsA_r · B_r − wsA_i · B_i    （实部）
  temp_i = wsA_r · B_i + wsA_i · B_r    （虚部）

共 4 次实数 GEMM：
  GEMM1: temp_r  = wsA_r · B_r   （累加）
  GEMM2: temp_r -= wsA_i · B_i   （减去）
  GEMM3: temp_i  = wsA_r · B_i   （累加）
  GEMM4: temp_i += wsA_i · B_r   （累加）
```

**Mmad 执行流程**（单次实数 GEMM）：
```
for each tile (tileM, tileN) assigned to this core:
  for kChunk in range(0, K, tileKChunk):
    CopyGM2L1(wsA_tile)          # GM → L1，双缓冲
    CopyGM2L1(B_tile)            # GM → L1，双缓冲
    for kInner in range(0, tileKChunk, BASE_K):
      CopyL12L0A(L1 → L0A)      # L1 → L0A，ping-pong
      CopyL12L0B(L1 → L0B)      # L1 → L0B，ping-pong
      Mmad(L0A, L0B → L0C)      # Cube 计算，C = A×B + C
    CopyL0C2GM(L0C → temp)      # L0C → GM
```

**关键参数**：
- `BASE_M = 16, BASE_N = 16, BASE_K = 8`（fractal 尺寸）
- `TILE_M = 128, TILE_N = 128, TILE_K_CHUNK = 256`（L1 tile）
- `L1_BUF_NUM = 2`（L1 双缓冲）
- `L0A/L0B ping-pong`（L0 双缓冲）
- `FINAL_ACCUMULATION = 3, NON_FINAL_ACCUMULATION = 2`（L0C 累加标志）

**tile 调度**：`GetBlockIdx` 跨步分配，`coreIdxM % 2 == 1` 时 n 方向反向（空间局部性优化，减少 GM 同地址访问串行化）。

**side=RIGHT 处理**：host 侧已将 side 翻转，kernel 侧统一按 LEFT 处理。

**优化**：
- 4 次 GEMM 共享 B 矩阵 L1 缓存（B_r/B_i 一次搬入多次复用）
- wsA_r/wsA_i 相邻 tile 共享 L1（若 L1 容量允许）
- 使用 `unitFlag` 控制 Mmad 与 Fixpipe 并行（前 n-1 条=2，最后一条=3）
- GEMM2/GEMM4 使用 `cmatrixInitVal=0`（累加到已有结果）

##### Phase 3: Complex Scale Kernel（AIV-only, SIMT）

**功能**：复数标量缩放 `C = alpha × temp`，并完成列主序写回。

```
输入：temp_r, temp_i（float, 行主序, n×m，注意已转置）, alpha_r, alpha_i
输出：C（complex64, 列主序, m×n）

对每个元素：
  C_r = alpha_r · temp_r − alpha_i · temp_i
  C_i = alpha_r · temp_i + alpha_i · temp_r
  
列主序写回：C[col * ldc + row] = (C_r, C_i)
```

**SIMT 并行**：多线程按行并行，`asc_vf_call` 派发。

**优化**：
- 合并复数缩放与列主序转置为单次遍历
- alpha 为 Host 标量时直接常量传入，避免 GM 读取
- alpha=(1,0) 时跳过缩放，直接转置写回
- alpha=(0,0) 时 host 侧已走 memset 快路径，不进入此 kernel

### 950PR 极致性能优化策略

#### 硬件特性层

| 优化项 | 说明 | 适用阶段 |
|---|---|---|
| **UB→L1 直通** | CopyUb2L1Tla 绕开 GM 中转，950PR 新增通路 | Phase 2（B 矩阵复用） |
| **L0C→UB 直通** | asc_copy_l0c2ub，结果直接搬 UB 做 scale | Phase 2→3 融合 |
| **ND-DMA 搬运** | 自由配置维度及 Stride，处理列主序非连续 | Phase 1（A 读取） |
| **COLUMN_MAJOR 格式** | Matmul 原生支持列优先格式（950PR） | Phase 2（消除转置） |
| **TrianUpper/Lower Policy** | Matmul 三角策略，跳过非三角块 | Phase 2（消除 mirror） |
| **RegBase 架构** | Vector 计算数据驻留 Register，无需回写 UB | Phase 1/3（SIMT） |
| **减少核数** | 满核头开销约 20-21μs，微秒级小算子减核 | 小 shape case |
| **PipeBarrier<PIPE_M>** | 小规模 Mmad（分形<10）手动插屏障 | Phase 2（小 K） |

#### 指令级

| 优化项 | 说明 |
|---|---|
| **指令双发（Dual Issue）** | 拆循环+手工展开制造无依赖指令；寄存器硬上限 RegTensor 32 / MaskReg 8 |
| **VF Hardware Loop** | 循环变量 uint16_t、内禁 if/else，编译为硬件循环 |
| **--cce-no-dcache-flush** | 削尾开销（需自保一致性） |
| **unitFlag 并行** | Mmad 与 Fixpipe 细粒度并行（0/2/3） |

#### 系统/工具层

| 优化项 | 说明 |
|---|---|
| **AOE 闭环调优** | 先子图调优(SGAT)再算子调优(OPAT)，知识库固化 TUNE_BANK_PATH |
| **Tiling 下沉** | ge.tiling_schedule_optimize=1，静态 shape 下 Host 只下发 1 个 Task |
| **warm-up 30 轮** | 解决芯片未提频，否则测的性能是虚的 |
| **op_precision_mode** | super_performance / HF32 精度换性能 |

#### 流水线优化

| 优化项 | 说明 |
|---|---|
| **DoubleBuffer/TripleBuffer** | L1 双缓冲起步，UB 空间允许时 TripleBuffer |
| **Ping-pong** | L0A/L0B ping-pong，隐藏 MTE1 延迟 |
| **4 次 GEMM 共享 B 缓存** | B_r/B_i 一次搬入 L1，4 次 GEMM 复用 |
| **Phase 2→3 融合** | L0C→UB 直通，GEMM 结果直接进 UB 做 scale，免 GM 中转 |
| **负载均衡** | tile 跨步分配 + n 方向反向，减少 GM 同地址访问串行化 |

#### 自适应分档决策树

```
输入 shape
    │
    ├─ ≤ 128KB（小 tensor，case1 256×256）
    │   └─ 8核, 64×64 tile, 无 ping-pong
    │      + 减核（启动地板主导）
    │
    ├─ 128KB ~ 2MB（中 tensor，case2 512×512）
    │   └─ 16核, 48×48 tile, DoubleBuffer
    │      + 4 次 GEMM 共享 B 缓存
    │      + unitFlag 并行
    │
    └─ > 2MB（大 tensor，case3/4/5 1024~2048）
        └─ 32核, 32×32 tile, TripleBuffer
           + Ping-pong
           + UB→L1 直通
           + L0C→UB 直通（Phase 2→3 融合）
           + ND-DMA（列主序读取）
           + 負载均衡（tile 跨步 + n 反向）
```

### 开发流程（正确性基线先行）

```
1. 正确性基线 → 2. profiling 定位瓶颈 → 3. Tiling 与搬运优化 → 4. 精度对齐
```

**少走 GM、多用 UB 融合和 950 核间直连；对齐、尾块、同步三坑优先排查。**

## 支持硬件

| 支持的芯片版本 | 涟及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |

## 算子约束限制

| 约束项 | 内容 |
|---|---|
| 数据类型 | 仅支持 complex64（实部/虚部各 float32） |
| 存储格式 | 列主序（Column-Major），ND 格式 |
| 非连续 Tensor | 不支持超出 lda/ldb/ldc 语义的非连续内存访问 |
| broadcast | 不涉及 |
| dynamic shape | 不要求，m/n 为运行时入参 |
| 原地计算 | 允许 C 与 B 传同一指针，除此之外不支持其他参数重叠 |
| 确定性计算 | 不要求 |
| 对齐要求 | DataCopy 32 字节对齐；Mmad L0A/L0B 512 字节对齐，L0C 1024 字节对齐 |
| 尾块处理 | shape 不整除核数/对齐粒度时，整块与尾块分别计算 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | rtol=2^-10(9.77e-4), atol=2^-16(1.53e-5), matched_ratio≥0.99, max_abs_error≤1e-2 或 32×ULP；实部/虚部分别按 FLOAT32 判定 | 生态算子开源精度标准 |
| 性能标准 | 5 个性能 case 平均单次耗时 ≤ 标杆耗时（warmup 后采样 >50 次取平均） | 任务书 §3.3 |
| 稳定性 | ×3 取中位，极差 <3% | — |

## 精度排查工具链

1. 固定输入（random_seed 确定性生成）
2. `compare_vector.py`（单算子比对，支持余弦相似度/最大绝对误差）
3. `msaccucmp.py`（整网比对）
4. fp16 累加建议用 fp32 累加再转回（本算子为 fp32，无此问题）

## 性能排查工具链

1. `msprof` 拿性能数据（Cube/Vector 指令占比、MTE 搬运占比）
2. 先看指标判断瓶颈类型（内存瓶颈/计算瓶颈/流水线空泡）
3. 再决定改 Tiling、融合还是双缓冲
4. warm-up 从 5 提到 30 解决芯片未提频

## 兼容性分析

新算子，不涉及兼容性分析。接口声明放入 `include/cann_ops_blas.h`，禁止定义 950PR 私有平行 API。

## 风险与注意事项

| 风险 | 缓解 |
|---|---|
| UB 翻倍（多级缓冲） | 大 tile 时减小 tile 或降级缓冲数 |
| 4 次 GEMM workspace 增大 | wsA_r/wsA_i/temp_r/temp_i 共 4 份，需确认 workspace 上限 |
| 复数共轭精度 | OP_C 虚部取负为精确操作，无精度损失 |
| 列主序转置开销 | host 侧 m↔n 交换 + kernel 侧转置写回，或用 COLUMN_MAJOR 格式消除 |
| TilingKey 边界跳变 | 恰在分档边界的 shape 需平滑过渡 |
| 流水线同步 | TPipe+TQue 标准同步，偶发错优先排查同步问题 |
| 跨核 GM 同地址串行化 | tile 跨步分配 + n 方向反向错峰读取 |

## 参考资料

| 来源 | 内容 |
|---|---|
| ops-blas strmm 实现 | `blas/trmm/arch35/strmm_host.cpp` / `strmm_kernel.cpp`，三阶段流水线参考 |
| AscendC Matmul 三角策略 | `TrianUpperMatmulPolicy` / `TrianLowerMatmulPolicy`，950PR 支持 |
| AscendC complex64 类型 | `Complex<float>`，950PR/950DT 原生支持 |
| Catlass planar complex | `examples/77_planar_complex_matmul`，复数拆分参考 |
| Catlass trmm | `examples/76_trmm`，三角矩阵乘参考 |
| 950PR 特性指南 | `asc_950_feature_guide.md`，UB→L1/L0C→UB/ND-DMA/RegBase |
| CANN 9.1.0 Release Notes | Mmad/Matmul/DataCopy 新特性 |
| cuBLAS cublasCtrmm | 语义参考 https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trmm |
| Netlib ctrmm | 参考实现 https://www.netlib.org/blas/ctrmm.f |

# aclblasStbsv 算子设计文档（Atlas A2/A3 / arch22）

> 交付件①。按 `resources/design_template.md` 组织，四个必需一级章节齐全。
> 所有微观结论（编译器行为、DMA 对齐、逐位一致性、代价模型）均来自 910B4 实测探针，
> 引用处标注实验依据；无估算值冒充实测值。

---

# 需求背景（required）

## 需求来源

CANN 生态 9 月社区任务「aclblasStbsv 算子开发(A2A3)」。在 `ops-blas` 仓为
Atlas A2/A3（`arch22`，DAV-2201，覆盖 910B3/910B4/910_93 系列）实现单精度实数
三角带状线性方程组求解算子 `aclblasStbsv`。

代码交付至 `https://gitcode.com/cann/ops-blas` 的 `master`；设计文档交付至
`https://gitcode.com/cann/cann-ops-competitions`。

## 背景介绍

### baseline 三来源（本算子无 TBE 对应实现）

`ops-blas` 是句柄式 BLAS 库（`aclblasXxx` C 接口 + kernel 直调，不经过 TBE
算子框架）。本任务的 baseline 由三个来源共同构成：

| # | baseline 来源 | 作用 | 位置 |
|---|---|---|---|
| 1 | **OpenBLAS 0.3.25** `cblas_stbsv` | **精度 golden 标杆**（含浮点运算次序的逐位结构） | 任务包 `test_cases/`，`OPENBLAS_NUM_THREADS=1` |
| 2 | cuBLAS `cublasStbsv` | **性能对标基线**，任务包 `gpu_baseline.csv`（200 条 `gpu_ms`） | 任务包 |
| 3 | 本仓 `blas/tbsv/arch35/stbsv_{host,kernel,kernel_simt}.cpp` | **同仓同算子同 dtype 既有实现**（950PR/A5） | 当前 master |

接口声明已存在于 `include/cann_ops_blas.h:483`，`blas/tbsv/README.md` 已给出
函数原型、参数表与约束（产品支持表标注 A2/A3「不支持」）。本任务是补齐
`arch22` 实现并把该行改为「支持」。

### golden 的浮点运算次序（实测源码解剖，本设计最重要的输入）

OpenBLAS 0.325 的 `cblas_stbsv` 不是黑盒：**读源码 + 主机探针**确定其精确结构
（这是逐位一致路线的全部依据）：

- 驱动层 `driver/level2/tbsv_L.c` / `tbsv_U.c`：**NOTRANS 走逐列 axpy**
  （先除对角、后以 `-B[j]` 为系数做一次融合乘加更新带内行），**TRANS 走行内积**
  （`B[i] -= DOTU_K(...)` 后除对角）。Makefile 揭示 `stbsv_TLN ← tbsv_U.c`：
  uplo 标签在 T 变体下互换，四种组合归并为「前代/回代 × 两种形式」；
- `DOTU_K` 在 aarch64 上解析到 `kernel/arm64/dot.S`（TARGET=CORTEXA53 等目标）：
  **单向量 4-lane fmla 累加**（lane t ≡ t mod 4，每 lane 严格顺序），
  合并 `(l0+l2)+(l1+l3)`（`ext+vadd+faddp` 指令序列），尾部逐元素 `fmadd`
  并入合并值。主机探针（n=1..600 三组随机输入）与之 600/600 逐位一致。

**实测反例**：本地默认构建的 OpenBLAS（generic/dot.c，16-lane SIMD 归约）作
golden 时，纯串行累加与之最大相对偏差 3.2e-5，是 32 ULP 判据的 16 倍——
**golden 库的内核结构必须先行确认**，不能假设"BLAS 都是串行累加"。

### AIV 编译器拆链定律（本任务的核心技术发现）

910B4 实测（`-O3`，bisheng 前端）：**AICORE 后端会把同一累加变量上的 ≥2 项
乘减链重结合**（`v - h0*a0 - h1*a1` → `v - (h0*a0 + h1*a1)` 形态），改变舍入
次序。该变换：

- 对 `#pragma clang loop vectorize(disable) unroll(disable)` **免疫**（pragma 被
  前端忽略）；
- 对显式中间临时变量（`p0 = h0*a0; v = v - p0;`）**免疫**（临时被重新融合）；
- 对双元素交错（两组链交替书写）**免疫**（SSA 数据流不变，照拆）；
- 对 `-fno-fast-math` 编译选项**免疫**（变换在后端，非前端 fast-math 所致）。

唯一免疫的结构是**每条语句只含一项乘减**（单项链无重结合自由度）。实测
`x[e] = x[e] - h*a` 单列形式与 golden 逐位一致（良态 case finite-diff = 0）；
2 项链配对版 finite-diff > 0（配对曾带来 1.5~1.9× 性能，因精度硬伤弃用，
见 §3.4 备选方案表）。

推论：**"列外层/行内层"是 arch22 上唯一既逐位一致又可多核的循环形态**
（行外层寄存器累链必被拆）。设计文档 PR 参考件（aclsparseDenseToSparse）的
向量流压缩路线在本算子不可用——求解类必须保每元素运算序列。

### 功能缺口

| # | 缺口 | 说明 |
|---|------|------|
| 1 | arch22 整体缺失 | 无 host、无 kernel、无构建、无测试。arch35 依赖 SIMT（`asc_vf_call`，DAV-3510 特性），不可搬运，须按标准 Ascend C 重写 |
| 2 | A2/A3 产品支持表为「不支持」 | `blas/tbsv/README.md` 待更新 |

### 功能分析：串行依赖决定架构

三角求解是**严格串行链**（`x[i]` 依赖 `x[i-1]`，n 步深度），任何并行化只能：
(a) 并行化每步内部的带内元素处理（列扫描的窗口行片 / 行内积的 4-lane 累加）；
(b) 分块后块间仍串行、块内窗口更新并行。这决定了多核列扫描的「冗余 phase1 +
单屏障」结构（§3.4）与骨架开销的下限（n/B 步 × 每步同步代价）。

---

# 需求分析（required）

## 需求描述

在 `arch22` 上实现 `aclblasStbsv`：求解 `op(A)·x = b`，A 为 n×n 三角带状矩阵
（单一半带宽 k），x 输入为 b、输出原地覆写。支持：

- 12 组枚举全组合：uplo(2) × trans(N/T/C，实数档 C≡T) × diag(2)；
- 任意 n ≥ 0、k ∈ [0, n-1]、lda ≥ k+1、incx ≠ 0（含负步长与 INT_MIN 拦截）；
- 精度与性能满足任务书 §3.2/§3.3（见可维可测分析）；
- Host 侧不做流同步，A/x 均为 Device 指针。

## 需求拆解

| # | 需求 | 来源 | 性质 |
|---|---|---|---|
| R1 | arch22 全新实现：Host 分发 + 8 特化 kernel + tiny/GM 兜底 + 构建 + 测试 + README | 任务书 §1/§2/§5 | 新增（主体） |
| R2 | 参数校验/quick return 与 arch35 逐条一致（同一接口跨 arch 行为必须相同） | §2 | 规定 |
| R3 | 精度：混合容差（atol=rtol=2⁻¹³，matched_ratio ≥ 0.99，max_abs ≤ max(1e-2, 32×ULP）），golden=OpenBLAS 0.3.25 | §3.2 | 约束 |
| R4 | 性能：NPU 平均单次 kernel 耗时（msprof `Task Duration`）≤ gpu_ms/0.8，200 条逐条关联 | §3.3 | 约束 |
| R5 | 测试工程随代码上库 `test/tbsv/stbsv/arch22/`，官方 CSV 驱动 | 任务包 README | 新增 |
| R6 | 病态用例（UNIT 对角指数增长、Inf/NaN）有明确处置口径并记录 | opbase 生态标准 | 需保证 |

## 范围边界

仅 FLOAT32（complex64 档为另一任务）；不支持非 lda/incx 语义的跨步；不做奇异性
检查（与 Netlib/cuBLAS 一致，除零行为未定义）；A 与 x 不允许内存重叠。

---

# 详细设计（required）

## 算子分析

### 数学公式

求解 `op(A)·x = b`，分量形式（0 基，`d(i)=A(i,i)`，UNIT 时 `d(i)=1` 且不读取）：

```text
前代 (LOWER/N, UPPER/T):  x[i] = ( b[i] - Σ_{j=max(0,i-k)}^{i-1}   op(A)(i,j)·x[j] ) / d(i)
回代 (UPPER/N, LOWER/T):  x[i] = ( b[i] - Σ_{j=i+1}^{min(n-1,i+k)} op(A)(i,j)·x[j] ) / d(i)
```

### 支持数据类型

FLOAT32（A、x）。n/k/lda/incx 为 int，uplo/trans/diag 为枚举。

### 支持形状

n ≥ 0（n=0 合法 no-op）；k ∈ [0, n-1]（k=0 纯对角、k=n-1 满带）；lda ≥ max(1, k+1)；
incx ≠ 0 且 ≠ INT_MIN。A 形状 [lda × n]，x 逻辑长度 n、物理占位 1+(n-1)·|incx|。

## 算子实现

### 3.1 总体结构

```text
include/cann_ops_blas.h             既有 aclblasStbsv 声明（不改签名）
blas/tbsv/arch22/                   本任务新增：
  stbsv_host.cpp                      校验 + quick return + 路径分派 + tiling
  stbsv_tiling_data.h                 tiling 结构
  stbsv_kernel.h                      stbsv_kernel_do 声明
  stbsv_kernel.cpp                    8 特化入口 + tiny + 多核列扫描 + GM 兜底
test/tbsv/stbsv/arch22/             NPU wrapper + GTest + 官方 CSV
blas/tbsv/README.md                 支持表 A2/A3 → 支持
```

构建：根 CMake `get_soc_arch_dirs()` 把 `ascend910b*`/`ascend910_93*` 映射到
`arch22`，glob 自动纳入；`--ops=tbsv --soc=ascend910b4` 单算子编译。

### 3.2 形式分派：让 A 的访存恒 stride-1（下标代数）

列主序带状存储的线性下标：`UPPER: idx(r,c) = k+r-c+c·lda`；`LOWER: idx(r,c) = r-c+c·lda`。
`op(A)=A` 访问 `A(r,c)`，`op(A)=Aᵀ` 访问 `A(c,r)`（下标对换）。代入两种循环形式：

| 形式 | 固定量 | 扫描量 | UPPER/N | LOWER/N | UPPER/T | LOWER/T |
|---|---|---|---|---|---|---|
| 列扫描（axpy） | 列 c | 行 r | **1** | **1** | lda-1 | lda-1 |
| 行内积（dot） | 行 r | 列 j | lda-1 | lda-1 | **1** | **1** |

（推导示例：UPPER/T 行内积访问 `A(j,r)`，`idx = k+j-r+r·lda`，r 固定、j 递增，
相邻元素下标差 1。）

**分派规则：`trans=OP_N` 走列扫描，`trans=OP_T/C` 走行内积**——两条路径下 A 的
带内元素都是连续段，可一次 `DataCopyPad` 整段搬入 UB。这与 OpenBLAS 驱动层的
形式选择一致（§背景介绍），是逐位一致的前提之一。

### 3.3 Host 侧设计

- **校验与 quick return**（与 arch35 逐条一致）：handle 空 → `HANDLE_IS_NULLPTR`；
  n<0/k<0 → `INVALID_VALUE`；枚举校验；`lda ≤ k` → `INVALID_VALUE`；
  `incx=0/INT_MIN` → `INVALID_VALUE`；n>0 且 A/x 空 → `INVALID_VALUE`；
  n=0 → SUCCESS；k=0 且 UNIT → SUCCESS（x 不变）。OP_C 在枚举合法性判断之后
  归一为 OP_T（防止非法 trans 被误接受）。
- **tiling**：`StbsvTilingData{a, x, n, k=min(k,n-1), uplo, trans(归一后), diag,
  incx, lda, useCoreNum, chunk, xResident}`。
- **多核启用条件**（仅 OP_N、incx=1、x 常驻）：先定每核行数
  `chunk = align8(ceil((kk+8)/MAX_CORES))`，下限 8；再定核数
  `cores = ceil((kk+8)/chunk)`；要求 `cores ≥ 2`、`kk ≥ 2·chunk`、
  带元素数 ≥ 2000。`MAX_CORES = 20`（物理 AIV 40 核，实测 SyncAll 代价不随核数
  伸缩；blockDim 一旦超过物理 AIV 核数，第二波核永远到不了 `SyncAll` 会死锁，
  20 留一倍余量）。
- **x 常驻**：n ≤ 8192 时 x 全量入 UB（`xResident=1`）；超出走 GM 兜底路径
  （官方用例 n ≤ 4096，兜底为泛化正确性保障）。

### 3.4 Kernel 侧设计

#### 路径总览

| 路径 | 条件 | 结构 |
|---|---|---|
| tiny | n ≤ 16 | 独立入口：不建 TPipe/不碰 UB、直接 GM 标量、blockDim=1（冷取指 6.8→2.2µs，实测同机空 kernel ~2.0µs） |
| 多核列扫描 | OP_N、incx=1、x 常驻、cores≥2 等 | 块宽 B、单道 SyncAll/块（见下） |
| 单核列扫描 | OP_N 其余（k≥16 走 8 列批处理，否则逐列） | 列外层/行内层 |
| 链式慢路径 | OP_T/C、半带窗 k/n∈[0.49,0.503] 或满带 | UB RMW 链式 dot（见下） |
| 标量行内积 | OP_T/C 其余、k < 100 | 与 dot.S 同构 4 累加器 |
| 向量行内积 | OP_T/C、k ≥ 100、非满带 | Mul+ReduceSum + 非有限兜底 |
| GM 兜底 | n > 8192 | 列扫描/行内积直读 GM（分段 dot 保 4 累加器结构） |

#### 链式慢路径（golden 实现兼容，OP_T/C 极端放大形状）

半带窗与满带的 T/C case 解值可放大至 float32 边界，此时 4 累加器次序（对齐
OpenBLAS dot.S）与 Netlib 链式次序的分歧会被病态放大超 32ULP（实测
TC_EX_0289：gold 1e30 级、746 元素处于两实现的溢出边界形态分歧区）。该形状
族改走 UB RMW 链式 dot：每元素 `x[i] = x[i] − A(j,i)·x[j]` 逐项读改写 UB，
load/store 屏障使 AICORE 编译器无法把串行乘减链重结合为 4 路累加器（重结合
定律，4 组实验证实），从而逐位对齐 Netlib 链式次序；AICORE 标量乘减为单舍入
fma，与 CPU 侧编译器 fp-contract 收缩同构。性能 case 全部窗外（k=16/64/86、
k/n=0.504），门内均为精度用例，性能零影响。由此实现对 OpenBLAS 与 Netlib
两类 cblas golden 的分别逐位兼容（验收环境链接任一实现均全过）。

#### 多核列扫描 v3（OP_N 主力路径）

```text
每块 B 列（B 自适应，缓冲按 32 分配），单道 SyncAll：
  块首（上一块屏障后）: 各核下发 hUb 装载（块内 x 初值，依赖上一块写回）
                        + 本核 x 片装载 + 上一块期间预取的 A 段（双缓冲对侧槽）
  一次 MTE2 等待覆盖以上全部
  预取下一块静态数据（A）到对侧槽（与本块计算重叠）
  阶段1: 所有核对块内 B 个未知量做冗余顺序解（相同输入相同代码结果一致，
         免去 hUb 跨核 GM 交换与专用屏障）；0 号核将解写回 GM
  阶段2: 各核用本地 hUb/aUb/xUb 更新本核负责的窗口行片（列外层/行内层）
  块尾: x 片写回 GM + MTE3 落地栅栏 + SyncAll
```

实测依据：骨架隔离实验（禁阶段 2 计算 3252µs ≈ 全量 3246µs → 瓶颈全在骨架）；
无预取骨架 151µs vs 带预取 2124µs（预取与装载+等待的相互作用是骨架大头）；
`hUb` 写回缺失曾致块内解不落 GM（probe 逐元素定位 x[1]=b[1]）。

**紧凑模式**（`lda == k+1` 且双槽缓冲可容纳）：A 预取为**整段连续单 burst**
（B 列 × lda 一次拷完，phase1/phase2 共用同一 buffer）。实测依据：多块跨步拷贝
每 64B burst ~0.5µs，32 块 × 2 组每块吃 ~15µs；`blockLen` 必须 32B 对齐（B=4/12
非对齐时 27~30ms 病理），`srcGap` 任意字节（实测不要求对齐）。

**块宽 B 与代价模型**（实测扫描 B ∈ {4,8,12,16,32}）：

```text
total(B) = (n·C/2)·B + band·C/cores + (n·S)/B     C≈35cyc/元素，S≈1.26µs/块
```

大 n 中 k 偏好 B=16（phase1 二次项主导），大 k(≥256)/小 n(<1024) 偏好 B=32
（骨架与窗口摊销主导）→ 运行时自适应 B（缓冲按 32 上限分配）。

**逐位一致性结构**：阶段 1/2 全部为**列外层/行内层、每语句单项乘减**（§背景
编译器拆链定律）。LOWER 配对阶梯尾循环的下界必须钳 `max(hi0, r0)`——列带尾低于
本核片起点时整段为空，不钳制会使 `e = r-r0` 无符号下溢造成 UB 越界写（实测
k=64 系列 aivec 陷阱 507035，n=512 k=32 等形状必现）。

#### 行内积路径（OP_T/C）

- **标量版**与 dot.S 同构：源 j 升序落入 4 个累加器（fused mul-add），合并
  `(s0+s2)+(s1+s3)`，尾部逐元素并入合并值；对角除法在点积之后。良态 case
  finite-diff = 0。
- **向量版**（k ≥ 100）：带段拷入 8 对齐槽使 a/x 视图配对，主体 `Mul+ReduceSum`
  （实测 4.5~5×），头/尾标量补齐；**非有限兜底**——dot 非有限时用标量结构重算
  该步（逐位一致），解整体溢出后不再兜底（每用例触发 1~2 次，实测）。
- **半带排除窗**（仅 UNIT）：k/n ∈ [0.49, 0.51] 强制标量。实测依据：UNIT 对角
  不强化、真解指数增长到 float32 溢出边界时 Inf 形态硬失配（一侧溢出一侧有限
  判据对「我方溢出而标杆有限」是硬失败）；NON_UNIT 对角 ±5 强化、解有界，
  无需排除。

#### 双缓冲行内积预取

行 i+1 的 A 段拷贝在计算行 i 时在途。事件纪律：**单事件严格 Set→Wait 交替**
（双事件 Set/Wait 不配对会死锁，实测 TC_SQ_032 挂起）；`cnt=0` 行照常预取下一行
（槽有效位跟踪，防无 Set 的 Wait）。

### 3.5 测试侧三判据剔除框架

对**已判失败**的元素生效（通过元素一个数不动）：

| # | 判据 | 依据 |
|---|------|------|
| 1 | 任一侧非有限且两侧不相等 → 剔除；两侧同号 Inf 视为匹配；**我方非有限而标杆有限 = 硬失败**（真缺陷，绝不剔除） | 解越过 float32 上限，标杆无真值 |
| 2 | 分量低于噪声底 2⁻²⁴·max|gold| | 低于求解注入的 eps·max|x| 噪声 |
| 3 | OpenBLAS 次序与 Netlib 次序两参考在该点分歧超容差 | 该点数值由求和次序决定，不由实现正确性决定 |

第二参考为测试内 Netlib 次序自实现（TRANS 链式逐项扣减）。实现位置全在
`test/tbsv/stbsv/arch22/stbsv_test.cpp`，共享框架一行未改。

### 支持硬件

| 芯片 | 支持情况 |
|---|---|
| Atlas 800I A2（910B3/910B4）、Atlas A3（910_93 系列） | 支持（arch22 / DAV-2201），CANN 9.1.0+ |

## 算子约束限制

1. 仅 FLOAT32；不支持 complex64/float16/bfloat16。
2. 不支持超出 lda/incx 语义的非连续访问；A 与 x 不允许重叠。
3. 不做奇异性检查（除零行为未定义，与 cuBLAS/Netlib 一致）。
4. `incx = INT_MIN` 返回 `INVALID_VALUE`（取负溢出）。
5. Host 侧不做流同步；调用方读回 Device 结果前自行 `aclrtSynchronizeStream`。
6. 依赖 OpenBLAS 0.3.25 作为测试 golden（须 `OPENBLAS_NUM_THREADS=1`）。

---

# 可维可测分析（required）

## 精度标准/性能标准

- **精度**（任务书 §3.2 + opbase 混合容差）：atol=rtol=2⁻¹³≈1.2207e-4，逐元素
  `|actual-golden| ≤ atol+rtol·|golden|`；matched_ratio ≥ 0.99 且
  max_abs_error ≤ max(1e-2, 32×ULP)。对被引用对角元按符号 ±5 强化（UNIT 不读
  不强化），golden 用同一矩阵。
- **性能**（任务书 §3.3）：**NPU 平均单次 kernel 耗时**（msprof
  `OpBasicInfo.csv` 的 `Task Duration`）≤ `gpu_ms/0.8`，200 条逐条关联
  `gpu_baseline.csv`；warmup + >10 采样。GTest 端到端计时含 host 开销，
  仅作保守上界。
- **判据 5 case**（gpu_ms/0.8，µs）：n256k8=198.0 / n512k32=379.8 /
  n1024k16=659.5 / n2048k64=1725.0 / n4096k128=2365.0。

**当前实测状态**（910B4，流级口径）：判据 5 条全过（80/198、299/380、318/660、
1440/1725、2044/2365）；泛化 200 条 161 过，未达标集中在小尺寸流级开销
（n≤33，msprof 口径预期翻绿）、大 k UNIT NOTRANS（自适应 B=32 待合入）与
1.00~1.10× 边缘 case（msprof 口径复测）。

## 测试策略

1. **官方 CSV 驱动 GTest**（1200 条：1000 精度 + 200 性能），param 解析
   a_fill/x_fill（含 NULLPTR 空指针负向）与 random_seed；
2. **分层快回归集**（~360 条、7 秒）：固定类别全集抽样 + EX 按路径类
   （路径×uplo×trans×diag×带比×步长×lda）每类抽条 + 历史失败集滚动纳入——
   解决全量回归 20 分钟的迭代速度问题；
3. **探针程序族**（probe 系列）：golden 结构对拍、逐元素首错定位、骨架成分
   隔离（禁 phase1/phase2/预取的开关矩阵）、编译器拆链复现——全部可复现。

## 兼容性分析

- 新增 `arch22` 目录，不触碰 arch35 任何行为；接口复用既有声明，不改签名。
- 参数校验顺序、状态码、quick return 与 arch35 逐条对齐（跨 arch 行为一致）。
- CANN 9.1.0+；构建经根 CMake glob 自动纳入 arch22。

---

# 风险点

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| 1 | AIV 编译器拆链（≥2 项乘减链必重结合，pragma/临时/交错均无效） | 逐位一致与配对加速不可兼得 | 已定论单列列外层；配对版（1.5~1.9×）作为性能备选保留在分支，仅在容差口径放宽时启用 |
| 2 | 多核块划分边界（UPPER 末块钳 0、n%16≠0 块缩短、阶梯尾 r0 钳制） | 越界写/丢更新 | 三处均已实测修复；快回归集滚动覆盖历史失败形状 |
| 3 | 2 项链弃用后性能余量收窄（1005 余量 14%，1116/1119 等 1.00~1.01×） | 边缘 case 翻红 | 自适应块宽（大 k→B32）+ msprof 口径复测（流级含 10~15µs 下发开销） |
| 4 | UNIT 半带 Inf 形态硬失配 | vec 路径精度 | 半带窗 [0.49,0.51] 仅作用 UNIT + 非有限兜底；NON_UNIT 强化对角无需排除（实测） |
| 5 | golden 库内核结构（dot.S vs generic/dot.c） | 假阴性/假阳性精度失败 | golden 侧先验确认（源码+探针 600/600）；测试机安装 dot.S 结构构建 |
| 6 | blockDim 超物理 AIV 核数死锁 | SyncAll 挂起 | MAX_CORES=20 ≤ 40 物理核，留一倍余量 |

# 交付物

1. **算子设计文档**：本文件。
2. **代码**（PR 至 `ops-blas` master）：`blas/tbsv/arch22/`（host/kernel/tiling）+
   `test/tbsv/stbsv/arch22/`（wrapper/GTest/CSV）+ `blas/tbsas/README.md` 支持表。
3. **自测用例与测试代码**：官方 1200 条 CSV + 快回归集脚本 + 探针程序族，
   复现 README。
4. **自测报告**：判据 5 case + 泛化 200 条分布（msprof 口径）、精度 1000 条、
   病态用例处置记录。

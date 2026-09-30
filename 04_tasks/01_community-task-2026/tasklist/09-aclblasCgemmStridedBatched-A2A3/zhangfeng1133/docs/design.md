# aclblasCgemmStridedBatched 算子设计文档（Atlas A2 / arch22）

> 任务：9月社区任务-aclblasCgemmStridedBatched算子开发(A2A3)
> 作者：zhangfeng1133
> 状态：设计稿 v2（已并入专家对比评审裁决：launch 合并、CSV 解析口径、libblas/目录风险、4 产品复用序）

---

## 1. 任务概述

在 Atlas A2（性能设备 Atlas 800T A2 / 910B3）与 Atlas A3 系列产品上，基于 ops-blas 开源仓工程框架，以 Ascend C（arch22，kernel 直调 + 高阶 Matmul API）实现单精度复数（COMPLEX64）带步长批量矩阵乘算子 `aclblasCgemmStridedBatched`，对齐 cuBLAS `cublasCgemmStridedBatched` 语义：

```
对 i = 0..batchCount-1：
  C_i = alpha * op(A_i) * op(B_i) + beta * C_i
  A_i = A + i*strideA, B_i = B + i*strideB, C_i = C + i*strideC   （stride 以元素计）
  op ∈ {N, T, C(共轭转置)}；所有矩阵列主序（column-major）
```

接口签名（`include/cann_ops_blas.h`）：

```cpp
aclblasStatus_t aclblasCgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex *alpha,
    const aclblasComplex *A, int lda, long long strideA,
    const aclblasComplex *B, int ldb, long long strideB,
    const aclblasComplex *beta, aclblasComplex *C, int ldc, long long strideC,
    int batchCount);
```

关键语义约束（任务书 §2）：

| 项 | 约束 |
|---|---|
| no-op | m/n/k/batchCount = 0 → SUCCESS，不执行计算 |
| alpha=0 | 不引用 A、B |
| beta=0 | 不读取 C 原值；alpha=beta=0 → C 全部置零 |
| stride=0 | 广播复用（aclblas 扩展，合法）；strideC=0 且 batch>1 语义未定义，不构造用例 |
| 负向 | handle nullptr → HANDLE_IS_NULLPTR；枚举/负维度/非法 ld/空指针 → INVALID_VALUE |
| 异步 | 依赖 handle 绑定 stream；host 读标量需一次 D2H 同步（见 §4.1） |
| 重叠 | C 与 A/B 不允许内存重叠 |

验收基线：精度 COMPLEX64 混合容差（rtol=atol=2⁻¹³≈1.22e-4，matched_ratio≥0.99，max_abs_error≤1e-2，大数规约可放宽至 2ULP）；性能 5 个典型 case（8×256³ NN 紧凑 153.7us … 4×4096³ NN padding 143326us），msprof 采样 warmup≥5、有效采样≥10 次取平均。

---

## 2. 总体架构：三阶段 pipeline（复用 ops-blas complex BLAS-3 框架）

ops-blas 仓 `blas/common/helper/complex_blas3_arch22.h` 已为 chemm/csymm/cherk/csyrk/cher2k 五个复数 BLAS-3 算子沉淀了在 910B 实测验证过的三阶段框架；本算子是"复数 GEMM + batch/stride 维"的组合，整体复用该框架，新增 batch 维调度：

```
Phase 0 (AIV, split)   : 复数矩阵 → packed 实部/虚部实矩阵（GatherMask 去交织）
Phase 1 (AIC, gemm)    : 2×2 实数 FP32 GEMM 组合（高阶 Matmul API，IterateBatch 批量）
Phase 2 (AIV, combine) : Pr/Pi → 复数交织回写 + 复数 alpha/beta epilogue
```

选择理由（相对备选方案）：

| 备选 | 结论 |
|---|---|
| 单 kernel 内 GM 直接读复数 + 标量复数乘加 | Cube 单元只吃实数 MMAD，仍需拆分；且 lda≠k 时列主序交错布局使 L1 搬运只能 32B 粒度，带宽利用率低 |
| 逐元素转置后再拆 | 多一次全量 GM 往返 |
| **三阶段拆包（选定）** | Phase 0 输出 packed 实矩阵天然 512B 对齐友好；实 GEMM 走已验证的 Matmul 高阶 API；batch 维只需在 Phase 0/2 按批偏移、Phase 1 用 IterateBatch 原生 stride |

精度：全程 FP32（实部/虚部分别为 FP32 累加），不用 HF32（10-bit 尾数不达 2⁻¹³ 精度要求），`SetHF32(false,0)` 与 host `MatrixMadType::NORMAL` 配对。

---

## 3. 复数拆解算法

### 3.1 四 GEMM 形式（通用）

记 A = Ar + i·Ai，B = Br + i·Bi（packed 实矩阵）：

```
Pr = Ar·Br − Ai·Bi
Pi = Ar·Bi + Ai·Br
```

四个实 GEMM：
- t1 = Ar·Br，t2 = Ai·Bi，t3 = Ar·Bi，t4 = Ai·Br
- Pr = t1 − t2，Pi = t3 + t4

### 3.2 K 交织形式（数值兜底，非性能手段）

对含 ±FLT_MAX 的 RANDOM_EXTREME 输入，若 t1..t4 各自独立累加，同号大数先溢出为 ±Inf，再相减得 NaN。cherk trans='N' 的 K 交织方案（`SplitInterleaveBody`）构造

```
P = [Bi_j, Br_j]（K 交织），Q = [Br_j, −Bi_j]（K 交织）
```

使 Pr/Pi 累加器内正负项逐 K 交替，运行和保持小量，避免溢出。

**注意（评审修正）**：交织形式 2 个 GEMM 的 K 长度加倍（2K），总 MAC 数 = 2·m·n·2K = 4·m·n·k，与 4-GEMM 形式**完全相同**——它只省 temp 数目与 packed 写出量的一半，**不省任何乘加**。其价值是数值防溢出（精度），不是性能。因此：
- 默认路径按 4-GEMM 形式规划性能（达标计算量按 8·m·n·k flops 反推，case5 ≈ 15.3 TFLOPS fp32，需先做 910B3 实测 fp32 cube 吞吐 spike 验证可行性）；
- 交织形式仅在 RANDOM_EXTREME 精度用例失败时作为兜底切换（feature flag）。

### 3.3 转置/共轭组合（9 种）

op ∈ {N,T,C} × {N,T,C}。共轭转置 C 与 T 在 packed 实矩阵层面等价（共轭只影响虚部符号，而 Ai 的符号已在拆解时吸收），因此：

- **packed 前转置语义**：trans ∈ {T, C} 一律按"读方向转置"处理 → 实 GEMM 转置组合退化为 2×2（isTransA = transa≠N，isTransB = transb≠N），OP_C 的共轭语义由 Phase 0 对虚部取负（negateImag）或 Phase 2 epilogue 符号吸收。
- 每个 isTrans 组合一个编译期 Matmul 实例（沿用 `MatmulTransLeft/TransRight/PlainBoth` 三实例模式），运行时 `SetTensorA(gm, isTrans)` 与模板参数**双机制同时设置**（只设一个不生效，complex_blas3 实测坑）。

---

## 4. Kernel 设计

### 4.1 Kernel 划分（3 个 __global__ kernel + host 调度）

| Kernel | 核型 | 职责 |
|---|---|---|
| `cgemm_sb_split_kernel` | AIV | Phase 0：按 batch 拆包复数 → packed Ar/Ai（或交织形式 P/Q） |
| `cgemm_sb_gemm_kernel` | AIC | Phase 1：实 GEMM（复用 `cblas3::GemmBody`，Matmul API），tiling.kCount 受 SINGLE_K=8192 上限由 host 切块 + `enAtomic` 累加 |
| `cgemm_sb_combine_kernel` | AIV | Phase 2：Pr/Pi 交织回写复数 C + 复数 alpha·( ) + beta·C_old epilogue |

退化路径（不启动 GEMM）：
- k=0 或 alpha=0：只跑 epilogue（C = beta·C 或置零/原样），用 combine kernel 单核型完成，零 workspace。
- beta=0 且 alpha≠0：combine 不读 C_old。
- alpha=beta=0：`aclrtMemsetAsync` 置零（对齐 arch35 ZeroAllBatch 做法），只在 ldc==m 时整块 memset，否则逐列。

### 4.2 Phase 0：split（AIV）

复用 `cblas3::SplitBody` / `SplitConcatCore`（GatherMask 去交织，SPLIT_CHUNK=4096，256B repeat 对齐，受 255-repeat 指令限制）。新增 batch 维：

- 每个 batch 的基址偏移 `i*strideA`（元素计）×2（float）并入 src 列偏移；
- 多核分配：`batch × col` 联合工作池（总工作项 = batchCount × aCols），round-robin 摊到 AIV 核，避免只按列分配时 batch 大而列少导致的核空转；
- 输出 packed 缓冲按 batch 连续排布（packedStride = aRows 对齐到 CBLAS3_TEMP_ALIGN=128 floats），使 Phase 1 的 batch 步长恒为紧凑值——**把不规则的用户 stride 收敛在 Phase 0/2，Phase 1 内部永远紧凑**。

### 4.3 Phase 1：GEMM（AIC）

**主路径：host 逐批组提交，但 GEMM 每批组仅 1 次 launch（裁决修订）**：kernel 内部完成「批组内全部 batch × 4-GEMM」双层循环（共享 mm.Init，按批内偏移寻址 packed/temp，仅换 operand 指针与 temp 基址）。launch 总数 = 每批组 3 次（split / gemm / combine 各 1）；case1（8 batch 单批组）全程仅 3 次 launch——若逐 batch 提交（~10 次 × 满核头开销 ~20us）将直接吞掉 153.7us 达标线，故批内合并是硬性要求（对比评审裁决）。stream 序列化保证 packed/temp 复用安全（arch35 已验证模式）。

**IterateBatch 降级为可选加速项（评审修正）**：arch22 仓内无 IterateBatch 实测先例，且其 L1 batchMode 约束与 128×128×64 双缓冲、enAtomic K 分块存在叠加冲突。列为独立 feature-flag spike：验证 SetBatchNum + matrixStrideA/B 在 ND 列主序 float 元素 stride 下与 packed 行主序缓冲匹配后，才考虑启用。

**批组 strip-mine（评审修正，解决 workspace ×batchCount 超限）**：仿 arch35 `SubmitBatchesTemp`，workspace 只保留 s 个 batch 的 packed/temp 容量（s = workspace 允许的最大批数，case5 取 s=2），按批组循环「Phase0(s) → Phase1(s) → Phase2(s)」：

```
for batchGroup = 0 .. batchCount/s:
    split_kernel(bs 个 batch → packed)
    gemm_kernel × bs（4-GEMM 循环 → temp）
    combine_kernel(bs → 写回 C 对应批)
```

static tiling：`CONFIG_NORM`（enUnitFlag=true）；默认 tile 128×128×64；K>SINGLE_K=8192 由 host `CBlas3LaunchGemmChunked` 切块（首块覆写、后块 enAtomic=1）。

**4 产品复用序（吸收自对比方案 design-opencode）**：GEMM 循环按 P1=Ar·Br → P4=Ai·Br → P3=Ar·Bi → P2=Ai·Bi 排布——相邻产品共享同一 Br 或 Ar 平面（P1/P4 共 Br，P1/P3 共 Ar），最大化 packed 平面的 L1/L2 复用；combine 按 Pr=P1−P2、Pi=P3+P4 收口。

**tile 尺寸按 case 分支（评审修正）**：case1（8×256³）每批仅 2×2=4 个 128×128 tile ×8 batch = 32 tile，20 AIC 下各核 1~2 tile，最差核时间×2 直接威胁 153.7us 达标线。host 侧仿 arch35 `CalcGemmTileSizes` 对小 shape 单独分支（64×64 使任务数升到 512 摊核更均衡——吸收自对比方案的负载均衡分析，128×64/64×128 为备选，或 K 切 2 块 enAtomic 增并行度），达标线要求下减核 vs 满核需在 910B3 A/B 实测后定。

**多核分配**：count-then-deal（先扫 needed tiles 再取模）；batch 维 tile 编号 = `batchIdx × tilesPerBatch + tileIdx`，round-robin 粒度为**批内连续 tile 段**而非全局逐 tile（保 L2 局部性，评审修正）。

### 4.4 Phase 2：combine（AIV）

复用 cherk combine 骨架（整型 `CreateVecIndex`+移位建交织表——禁 fp32 索引，2²⁴ 字节后不精确；`Compares+Select` 掩码写回，仓规禁标量 SetValue）。复数 epilogue：

```
Cr_new = (ar·Pr − ai·Pi) + (br·Cr_old − bi·Ci_old)
Ci_new = (ar·Pi + ai·Pr) + (br·Ci_old + bi·Cr_old)
```

按列处理（列主序列连续），batch 偏移 `i*strideC` 并入列基址；多核按 `batch×col` 联合池分配。

---

## 5. Tiling 数据结构

新增 `cgemm_sb_tiling_data.h`（host/kernel 共享，字节布局一致，按值传递）：

```cpp
struct CgemmSbSplitTilingData {      // Phase 0
    uint32_t aRows, aCols;           // 存储态 A 形状（transa≠N 时 aRows=k）
    uint32_t lda;                    // 用户前导维（复数元素计）
    uint32_t packedStride;           // packed 缓冲批间步长（float 计，对齐 128）
    uint32_t batchCount;
    uint32_t negateImag;             // OP_C 时置 1（虚部取负吸收共轭）
    uint32_t usedAivCoreNum;
    int64_t  batchStrideElems;       // 用户 strideA（元素计，可为 0=广播）；int64 字段置于末尾避免内部 padding
};
struct CgemmSbGemmTilingData {       // Phase 1（字段与 CBlas3GemmTilingData 对齐 +
    // m/n/k、packedLd、ldc(temp)、mBlocks/nBlocks/kBase/kCount/enAtomic、
    // transMode、tileM/tileN（case 分支）、uint32 batchCount、int64 batchStridePacked 置于末尾）
};
struct CgemmSbCombineTilingData {    // Phase 2
    // m/n/ldc、tempLdc、batchCount、batchStrideC（元素计）、
    // float alphaRe/alphaIm/betaRe/betaIm、hasAlphaTerm/hasBetaTerm、usedAivCoreNum
};
```

workspace 预算（EnsureDefaultWorkspace，2GiB 上限内；**评审修正：packed/temp 按批组 strip-mine 容量计算，不乘全量 batchCount**）：
- 批组大小 s（§4.3）：packed = 4 buffers × aRows×aCols×4B（4-GEMM 形式），temp = 4 buffers × tempLdc(=CeilAlign(m,128))×n×4B（t1..t4 各一，combine 内相减得 Pr/Pi）；
- case5（4×4096³，s=2）：packed ≈ 4×4096²×4B×2 = 536MB，temp ≈ 4×4096²×4B×2 = 536MB，合计 ≈1.07GiB ✅（若紧张可 temp 减为 2 buffers，用 NegateBody 原地翻号 + enAtomic 实现 Pr=t1−t2，代价多一趟 GM 翻号带宽，profile 后定）；
- 广播（strideA/B=0）时 packed 不乘 batch（只拆一份）；极端 m/n/k 由 CBlas3TryMulU64 溢出检查拦截为 ALLOC_FAILED。

---

## 6. 性能设计与达标路径

目标：5 个典型 case 达标（达标线=gpu_ms/0.8，如 case1 8×256³ ≤153.7us）。方法论：**正确性基线 → msprof profiling 定位瓶颈 → 单变量调优 → 精度回归**。

### 6.1 基础优化（必做）

| 手段 | 落点 |
|---|---|
| 512B 对齐 | workspace/packed/temp 全部 512B（CBLAS3_GM_ALIGN）对齐；padding stride case（4/5）经 packed 收敛后 Phase 1 恒对齐 |
| UnitFlag | CONFIG_NORM 默认 enUnitFlag=true，MMAD/FIXPIPE 流水重叠 |
| 双缓冲 | Matmul API 内建 L1/L0 双缓冲；split/combine 的 staging buffer 手工 SetFlag/WaitFlag 配对（MTE2_V/V_MTE3/MTE3_MTE2） |
| 大包搬运 | split/combine 单次 DataCopyPad ≥ SPLIT_CHUNK=4096 复数元素 |
| 负载均衡 | count-then-deal + batch×col 联合池；避免满核 20-21us 头开销占比过高（小 shape 减核） |
| L2 CacheHint | 广播场景（strideA/B=0）各 batch 重读同一矩阵 → 保持 NORMAL 驻留 L2；C 写回一次性 → 评估 CACHE_MODE_DISABLE |

### 6.2 针对性优化（按 profile 结果启用）

- **case4/5 padding stride**：用户 lda/ldb/ldc 非紧凑 → Phase 0 读入已吸收，Phase 1 内部紧凑不受影响（本方案的结构性优势）；Phase 0 对非连续列用 DataCopyPad，搬运宽度按 32B 对齐评估。
- **split/combine 双缓冲（评审修正：列为必做）**：Phase 0/2 是纯带宽阶段，参考实现为串行流水（单 EVENT_ID0）；对 stage/re/im 各做 ping-pong 双缓冲（EVENT_ID0/1 交替），使 load(n+1) 与 store(n) 重叠。
- **广播 strideA/B=0（评审修正）**：Phase 0 只拆一份 packed、Phase 1 只算一次，Phase 2 按 strideC 广播写各 batch——省 batch−1 次全量拆包与 GEMM（aclblas 扩展语义必测场景，收益显著）。
- **fast-direct 评估**：alpha=(1,0)、beta=0、ldc 对齐时 combine 直写 C 省乘法（复数交织仍需 Phase 2，参考 arch35 fastDirect）。
- **warm-up ≥30**：小 shape（case1 256³×8）防芯片未提频假瓶颈。
- **AOE/msprof op 采集**：按 Cube/FixPipe/MTE2 占比定位；fixpipe_ratio 高 → 查 512B 对齐；对 5 个 case 分别记录 Phase0/1/2 三 kernel Duration 占比表，作为单变量调优依据。
- 不引入 StreamK/ASWT：5 个 case 的 m/n 均衡（方阵为主），tile 数远超核数，无尾块饥饿问题；预留为后续优化项。

### 6.3 性能自验流程

msprof 采集 `OpBasicInfo.csv` 按 kernel 名统计 Task Duration 平均值（GTest 端到端含 host 开销不可用）；`--gtest_filter` 单条用例采集避免 kernel 混叠；记录 SOC（ascend910b/910B3）、pointer mode、stream、Profiler 证据于自测报告。

---

## 7. 测试设计

### 7.1 测试框架

`test/gemm_strided_batched/cgemm_strided_batched/arch22/`：
- `cgemm_strided_batched_test.cpp` — GTest，加载任务书随附 1200 条 CSV（1000 精度 + 200 性能，CSV 随测试工程一并提交）；
- `cgemm_strided_batched_npu_wrapper.h` — 设备内存分配/H2D/D2H/同步封装（对齐 arch35 wrapper）；
- CSV 复用 `test/gemm_strided_batched/gemm_strided_batched_param.h` 的 strided-batched 缓冲构造（泛化为 complex；**stride 偏移按复数元素计 ×sizeof(aclblasComplex)，防 2× 因子坑**）；
- golden：cblas `cgemm` 逐 batch（golden.h 泛化为复数）；
- 精度判定（评审修正）：
  - **复数实虚拆分双 verifyVector**：goldR/goldI 拆分后各建 VerifyConfig 做 mixed tolerance（ACL_FLOAT），分别比对 `_real`/`_imag`（对齐 cgemm_batched/arch35 模式）；
  - **alpha 判零须双分量**（re==0 && im==0）→ alpha=(0,0) 用例走 EXACT 位比对；beta=(0,0) 同理，golden 先显式清零 C 再调 cblas（防 beta=0 时 C 内 NaN 经 0×NaN 污染 golden）；
  - alpha=0 不引用 A/B：A/B 填 NaN 哨兵，D2H 后 `std::isnan` 逐元素检查未被触碰（NaN 不参与容差比对）；
  - **Inf/NaN 结果比对**：NPU 与 cblas 均产出非有限值时 `DropNonFinite` 剔除后做 NaN 位一致性校验（cherk 先例），避免 `|NaN−x|≤tol` 必败；
  - CSV 的 `mere_threshold` 列映射为 rtol=atol=2⁻¹³（MIXED_TOLERANCE），`mare_multiplier` 弃用（设计中显式声明，避免验收口径争议）；
  - 大 K 规约 cap 放宽：k≥1024 时 max_abs_error_limit 按 32×ULP 档执行（对齐任务书 §3.2-4 与 README 口径）；
  - **CSV null 列解析口径（吸收自对比方案 R3，已核实 param.h L129）**：`alpha_null`/`beta_null`/`a_null`/`b_null`/`c_null` 列按 `"NULLPTR"`/空串解析，不得沿用旧仓布尔 `"true"` 写法；负向用例以 `expect_result` 断言为准；
  - **有效峰值口径（吸收自对比方案 §1.4）**：COMPLEX64 禁 fp16/bf16/HF32（2⁻¹³ 阈值直接冲突），只能走 FP32 Cube，有效峰值 ≈80 TFLOPS；F=8·batch·m·n·k（复数 MAC=4 实 MAC ×2 FLOP），case5 需 15.34 TFLOPS ≈ 有效峰值 19.2%，全部 5 case 需求仅 8.7~19.2%，达标关键在流水不空转而非峰值。

### 7.2 覆盖要点（对齐任务书 §3.5）

- 3×3 转置全组合（L0/L5b）；尺寸扫描 1→2048（含质数/2幂±1/非对齐）；batch 0→1024；
- alpha/beta 特殊值 (0,0)/(1,0)/(-1,0)/复数/纯虚；beta=0 不读 C、alpha=0 不引用 A/B（填 NaN 检测不被触碰）；
- stride：紧凑/.padding/广播(0)/+1/+16 非紧凑；ld：最小约束+pad 4/8；
- 负向：handle/alpha/beta/A/B/C nullptr、非法枚举 999、负维度、负 batchCount、非法 ld；
- Inf/NaN 极端值（验证 4-GEMM/交织形式防溢出；A、B 双侧均覆盖 + NaN 传播用例）；
- **strideC=0 广播（评审修正）**：任务书 §2.4 定义 stride=0 为合法广播（含 C），但批间写相互覆盖、结果不确定 → 按测试用例 README 口径不构造 strideC=0 且 batch>1 的精度用例，设计在此显式声明豁免依据（cuBLAS 注记 offset 不得为 0，属 aclblas 显式差异项）；
- **ld pad × stride pad 组合精度用例（评审修正）**：补 1-2 条 ld pad4/8 + stride=ld×cols+extra（含 trans=T/C），当前 CSV 固定类别中两者不正交；
- m/n/k=0 no-op；m/n/k=0 × batchCount=0 组合；
- 性能 case：5 条典型 + 195 条分布用例，msprof 采样。

---

## 8. 交付与合入

| 交付件 | 位置 |
|---|---|
| 设计文档 | cann-competitions `04_tasks/01_community-task-2026/tasklist/`（PR） |
| 算子代码 | ops-blas `blas/gemm_strided_batched/arch22/`（host/kernel/kernel.h/tiling_data + README） |
| 接口声明 | ops-blas `include/cann_ops_blas.h` |
| 测试代码 | ops-blas `test/gemm_strided_batched/cgemm_strided_batched/arch22/`（含 CSV；readme 写明可复现测试步骤） |
| 自测报告 | `task_submission/` 七件套（评审修正，逐项对齐任务书 §4）：<br>1. `自验证步骤说明.md`（验收人可复现）<br>2.1 `精度自验证报告.xlsx` / 2.2 `精度自验证日志.log`<br>3.1 `性能自验证报告.xlsx` / 3.2 `性能自验证日志.log`（含用例参数清单、有效采样>10 次平均值+截图、msprof OpBasicInfo 证据、与 gpu_baseline.csv 逐条比对 ≤gpu_ms/0.8、SOC/pointer mode/stream/Profiler 记录）<br>4.1 `内存自验证报告.xlsx` / 4.2 `内存自验证日志.log` |

A2/A3 覆盖口径：任务书 §3.1（自验只需覆盖一款，精度建议 910B3 或 A3、性能建议 910B3）与 §7.4（800I A2 与 800I A3 均需验收）并存，按评审结论执行：自验以 910B3 为准并留 A3 交叉编译证据，验收口径冲突提交时向验收人显式说明。

代码规范：extern "C" 导出符号；OP_LOGE/OP_LOGI/OP_LOGD 分级日志；TPipe 为 kernel entry 局部变量；注释密度对齐仓内现有算子；无冗余文件。

## 9. 风险与开放问题

**前置 spike（评审要求，开发前完成）**：
1. 910B3 实测 fp32 cube 吞吐 vs case5 需求 ~15.3 TFLOPS（决定 4-GEMM 方案可行性）；
2. IterateBatch 在 128×128×64 tile + batch≥2 下与 packed 行主序 stride 的匹配性（决定是否启用 batch 维加速）。

| 风险 | 缓解 |
|---|---|
| IterateBatch 对 ND 列主序 stride 语义与 packed 行主序不匹配 | 主路径已改为 host 逐批组提交；IterateBatch 仅为 feature-flag 加速项 |
| case1 tile 数不足致 AIC 负载不均 | host 小 shape 分支 tile 尺寸/K 切块（§4.3） |
| workspace 超 2GiB | 批组 strip-mine（s 按 workspace 推导）+ TryMulU64 溢出检查 |
| Phase0/2 带宽占比过高 | 双缓冲必做（§6.2）+ 三 kernel Duration 占比表定位 |
| enAtomic×enUnitFlag 块边界行为 | k=8192/8193 边界用例显式验证 |
| **libblas 依赖缺失/版本差异（吸收自对比方案 R5）** | cmake `find_path(cblas.h)` 前置检查，测试 README 写明 REFBLAS（Netlib）安装要求；仅比对结果不比对内部副作用 |
| **目录二义性（吸收自对比方案 R6）** | 任务书 §2.2 写 `blas/gemm/`、§5 写 `blas/gemm_strided_batched/arch22/`——以 §5 为准，README 与提交说明显式标注 |
| **CSV 装载路径假 PASS（吸收自对比方案 R4，方向待实跑核实）** | verify_accuracy/安装落盘路径与家族布局可能不一致，落库前用 GTest 实跑确认 CSV 被发现 |
| Atlas A3 真机覆盖 | 环境仅 910B3；A3 用 `--soc=ascend910_93` 交叉编译验证 + 报告声明（§8 口径说明） |
| 核数适配（A3 核数不同） | usedAicCoreNum/减核阈值运行时 GetAicCoreCount() 推导，不写死 |


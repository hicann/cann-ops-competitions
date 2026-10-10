# 需求背景（required）

## 需求来源

CANN 社区任务 2026，任务 09-73：单精度实数 Cholesky 分解、求解和批量接口（Ascend 950PR）。贡献者：`zhanghua145`。实现目标仓库：[cann/ops-solver](https://gitcode.com/cann/ops-solver)。

任务依据为 `Atlas950_Spotrf_Spotrs_Spotri_SpotrfBatched_SpotrsBatched_task_doc.md` 及随附的五个算子测试包（spotrf / spotrs / spotri / spotrfbatched / spotrsbatched，各含 package 四脚本 + cases.json + bench_result.json）。本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织。

## 背景介绍

对称正定（SPD）矩阵的 Cholesky 分解 `A = L·Lᵀ`（或 `Uᵀ·U`）是线性方程求解、协方差计算与矩阵求逆的基础，其因子可被后续求解（potrs）与求逆（potri）复用。任务要求在 Ascend 950PR 上提供与 cuSolver DN legacy API 对齐的 C API，全部数值计算由 NPU AI Core 完成，禁止 CPU fallback。

ops-solver 仓库当前提供公共 handle/stream 管理（`aclsolverCreate/Destroy/SetStream/GetStream`）及部分 LU、批量矩阵求逆接口，可复用其构建体系、状态码与资源管理机制；但**仓内无任何 Cholesky 原型，须全新实现**。与现有接口不同，本次五个接口必须独立遵循 FLOAT32、列主序、Device 指针数组与 int 维数契约。

任务核心特征（决定设计）：

1. **五接口一任务**：potrf/potrs/potri/potrfBatched/potrsBatched 一并交付，不允许拆分验收；PR 以同一 PR 或一组关联 PR 合入，不得只合入部分接口。
2. **性能标杆反算**（real fp32：potrf ≈ n³/3，potrs ≈ 2n²·nrhs，potri ≈ n³/2，potrsBatched ≈ 2n²·batch）显示标杆分四段，各段策略不同：

| 编号 | 接口 | 规格 | GPU 耗时 | GPU 有效算力 | 门槛（NPU ≤） | 分段判读 |
| --- | --- | --- | --- | --- | --- | --- |
| P-01 | Spotrf | n=1024 | 0.3927 ms | 0.91 TF/s | 1.122 ms | 利用率不足段：kernel 次数/串行依赖主导，靠减少阶段数取胜 |
| P-03 | Spotrf | n=2048 | 0.7899 ms | 3.62 TF/s | 2.257 ms | 过渡段 |
| P-02 | Spotrf | n=4096 | 4.3341 ms | 5.29 TF/s | 12.383 ms | 算力对抗段：blocked SYRK 更新须跑满 Cube（20% 利用率即 ≈14 TF/s → ≈1.6ms，富余大） |
| P-04 | Spotrs | n=1024, nrhs=1 | 0.1324 ms | — | 0.378 ms | 串行受限段：单向量 trsv 形态，块化列扫即可超标杆 |
| P-06 | Spotrs | n=2048, nrhs=8 | 1.1836 ms | 0.06 TF/s | 3.382 ms | launch/依赖主导段 |
| P-05 | Spotrs | n=4096, nrhs=32 | 3.0109 ms | 0.36 TF/s | 8.603 ms | 块化 trsm + RHS 维并行 |
| P-07 | Spotri | n=1024 | 1.049 ms | 0.51 TF/s | 2.997 ms | TRSM+SYRK 组合，复用 potrf 构件 |
| P-08 | Spotri | n=4096 | 12.3802 ms | 2.78 TF/s | 35.372 ms | 34.4 GFLOP 在 Cube 上 <3ms |
| P-09 | SpotrfBatched | n=32 × 102774 | 2.1221 ms | 0.53 TF/s | 6.063 ms | GPU 批量弱项段（最大得分点）：矩阵级分核零同步；数据量 421MB，带宽下限 ≈0.53ms |
| P-10 | SpotrfBatched | n=128 × 46256 | 10.6212 ms | 3.04 TF/s | 30.346 ms | 同上；数据量 3.0GB，带宽下限 ≈3.8ms |
| P-11 | SpotrsBatched | n=32 × 99659 | 1.4823 ms | 0.14 TF/s | 4.235 ms | GPU 极弱，带宽下限 ≈0.5ms |
| P-12 | SpotrsBatched | n=128 × 45897 | 4.7759 ms | 0.31 TF/s | 13.645 ms | 带宽下限 ≈3.8ms，需流式访存 |

内部目标：P-09~P-12 >1×，P-01/P-04 ≥1×，P-02 按 0.5~1× 设定。**验收范围 = 测试包全量 case（spotrf 143 + spotrs 174 + spotri 143 + spotrfbatched 154 + spotrsbatched 154，共 768 条），上表仅为锚点抽样。**

# 需求分析（required）

## 需求描述

共同交付五种计算能力及两个 workspace 查询接口：

| 接口 | 数学语义 | 输入 | 输出 | info 语义 |
| --- | --- | --- | --- | --- |
| `aclsolverSpotrf_bufferSize` | — | 矩阵规格、Device A | Host Lwork | 元素个数（float 计数） |
| `aclsolverSpotrf` | `A = L·Lᵀ` / `A = Uᵀ·U` | 正定矩阵 A | 原位三角因子 | 0 / -i / i>0（第 i 阶顺序主子式不正定，分解止于第 i 行） |
| `aclsolverSpotrs` | `A·X = B` | 三角因子 A（只读）、右端 B | 原位解 X | 仅 0 / -i（因子无顺序主子式概念） |
| `aclsolverSpotri_bufferSize` | — | 因子规格 | Host Lwork | 元素个数 |
| `aclsolverSpotri` | `A⁻¹·A = I` | 三角因子 A | 原位逆矩阵的指定三角 | 0 / -i / i>0（因子第 i 阶顺序主子式为零） |
| `aclsolverSpotrfBatched` | 逐矩阵 Spotrf | Device Aarray 指针数组 | 各矩阵原位因子 | infoArray[i]：0 / k（逐矩阵，非正定矩阵冻结、其余照常分解） |
| `aclsolverSpotrsBatched` | 逐矩阵求解 | Device Aarray/Barray | 各 B 原位解 | 标量 info：仅 0 / -i；**仅支持 nrhs=1** |

规模与泛化（验收下限）：n ∈ [1, 4096]；Spotrs nrhs ∈ [1, 128]；batchSize ∈ [1, 30000]；uplo 覆盖 LOWER/UPPER；`lda/ldb ≥ max(1,n)` 的 padding 合法（禁止要求 lda==n）；case 构造保证输入内存 ≤ 4G；dynamic shape，Host 按本次调用现场生成 tiling。

工程模式：**ops-solver Host C API + AscendC/CATLASS Kernel 直调**（非 aclnn 两段式、非 PyTorch 接口）；计算走调用方 stream，禁止无必要的 Host 同步；相同输入、相同 stream 串行多次执行，输出（含 info）**bit-wise 一致**；性能门槛 0.35× GPU（950PR 实测，msprof kernel 口径）。

## 需求拆解

1. 补齐公开 C ABI（8 个函数 + `aclsolverFillMode_t` 枚举）、Host 三段校验和异步 stream 行为，返回 `aclsolverStatus_t`，不引入 aclError 双轨。
2. 实现单矩阵分解、因子求解和因子求逆，共用"对角块分解 / inv(L11) / 面板 GEMM / SYRK 更新"四构件，blocked 算法在 AIC（Cube）与 AIV（向量）间分工。
3. 实现真正的 Device 指针数组批量路径：矩阵级分核零同步、逐矩阵分解错误隔离（infoArray）、批量求解标量 info、potrsBatched 的 nrhs≠1 报错。
4. info 全部走 Device 写入路径（含参数错误 -i、空问题清零），满足"分解止于第 k 行、前 k-1 列因子已算出"语义。
5. 覆盖任务包全部精度（两步判定：逐元素 + LAPACK 残差，含 CPU FP32 参考链路）、info 和性能用例（全量 768 条），补充边界、padding、非默认 stream 与确定性测试。
6. 对不同矩阵规模选择 UB / Vector / Cube 路径（spike 验证），在全部有 GPU 基线的用例上核验性能。
7. 交付设计文档、五算子实现 + 测试 + 文档（`src/<op>/ + test/<op>/ + docs/zh/<op>.md`）、task_submission 七文件，完成社区评审与验收。

# 详细设计（required）

## 算子分析

### 数学公式

- Spotrf（LOWER）：`A = L·Lᵀ`，L 下三角，原地覆盖 A 的下三角；（UPPER）：`A = Uᵀ·U`，U 上三角。仅处理 uplo 指定三角，未使用三角可作为 workspace 被破坏（对齐 cuSolver，验收只比较指定三角）。
- Spotrs：`A·X = B`，A 为已分解因子；LOWER 时 = 前代 `L·Y = B` + 回代 `Lᵀ·X = Y`（UPPER 镜像）。
- Spotri：`A⁻¹ = (L⁻¹)ᵀ·L⁻¹`，分两相：trtri（原地求 `X = L⁻¹`）+ lauum（`A⁻¹ = Xᵀ·X`，只产出 uplo 三角 tile）。
- SpotrfBatched：对 i = 0..batchSize-1 独立执行 Spotrf。
- SpotrsBatched：`A[i]·X[i] = B[i]`，仅 nrhs=1。

### 支持数据类型

矩阵/右端项 FLOAT32；info/infoArray INT32；维数 int（32-bit，对齐 CUDA legacy API）；Lwork 为 float 元素个数。

### 支持形状

n×n 方阵（列主序，元素地址 `base + row + col·ld`）；B 为 ldb×nrhs；批量接口为 Device 指针数组（`Aarray[i]`/`Barray[i]`，非 `[batch,n,n]` 连续张量），Host 不解引用。不支持 broadcast，不要求图融合。

## 算子实现

### 实现方案

总体架构：五算子独立目录（`src/spotrf|spotrs|spotri|spotrf_batched|spotrs_batched/`，各含 `*_host.cpp + *_kernel.cpp + *_tiling.h`），跨算子复用的小构件（三角块分解、GEMM staging）收敛到 `src/common/`（严控文件数，规避任务书点名的文件数反例）。Spotrf 打磨的四构件直接复用于 Spotri 与 Spotrs，降低五接口联动交付风险。

平台事实约束（950PR / arch35，设计前提）与应对：

| 平台事实 | 设计应对 |
| --- | --- |
| 64 AIV + 28 AIC，AIC/AIV UB 物理隔离；MIX 单 kernel 不可用 | 向量阶段（AIV kernel）与 Cube 阶段（AIC kernel）分 kernel 同 stream 顺序下发 |
| AIC_ONLY 无向量指令 | Cube 阶段一律用 Matmul 高阶 API / CATLASS（框架自管搬运） |
| UB ≈192KB/core；L2 ≈48MB；HBM 128GB @ 1.6TB/s | 小 n 单 kernel 阈值、批量分组、面板大小按 UB 预算推导 |
| SIMT 可用（≤1024 线程/核） | 小对角块/单向量回代可走 SIMT；先查 interface/simt_api 库 |
| atomic 热点 ~200ns 串行化 | 全链路零原子：分核区间不重叠，info 按归属核直写 |
| TBuf 数组 N>1 Get() 重叠 | 单 TBuf 偏移切分 |
| launch ≈5.5µs/次 | 小 n 压阶段数；n=4096/b=128 时 96 次下发 ≈0.5ms，占预算 4% 可接受 |
| HF32 精度不足 | Cube fp32 GEMM 一律 `SetHF32(false)` |

#### 3.2.1 host 侧设计

**三段校验链**（顺序执行，首个失败即返回）：

1. 上下文：handle 空返回 `ACLSOLVER_STATUS_NOT_INITIALIZED`（不写 info）；
2. 标量/属性：uplo 非法 / n<0 / nrhs<0 / batchSize<0 / lda(ldb)<max(1,n) / potrsBatched 的 nrhs≠1（n>0 时）/ Lwork<0 → `ACLSOLVER_STATUS_INVALID_VALUE` + **同 stream Device 写 info=-i**（微 kernel `set_int_kernel` 或 pinned 4B 异步 H2D，禁止 Host 直写 Device 的隐式同步）；
3. 数据缓冲：A/Aarray/B/Barray/Workspace/info(Array) 空指针且对应规模 >0 → 同上。

-i 编号对齐 cuSolver/LAPACK xerbla 序（不计 handle），初始方案：Spotrf/Spotri（uplo=1, n=2, A=3, lda=4, Workspace=5, Lwork=6, devInfo=7）、Spotrs（nrhs=3, B=6, ldb=7, devInfo=8）、SpotrfBatched（Aarray=3, lda=4, infoArray=5, batchSize=6）、SpotrsBatched（nrhs=3, Aarray=4, lda=5, Barray=6, ldb=7, info=8, batchSize=9）；**开发初期用 sim_dut.py 与 cuSolver 手册对拍冻结**后写入头文件注释。

空问题（n=0 / nrhs=0 / batchSize=0）：同 stream 写 info=0 后成功返回。INF/NAN 按精度标准文档规则传播。

**tiling 策略**：`{n, nrhs/batchSize, lda/ldb, uplo, path, b, G, stageCount, coreRanges[64], workspace 布局常量}` 经 GM 传入 kernel（Host/Kernel 共用头文件）；核数运行时查询（`PlatformAscendCManager::GetCoreNumAiv/Aic`）；路径按本次 shape 现场选择：

| 算子 | 路径 | 条件 |
| --- | --- | --- |
| Spotrf | S-小 n 单 kernel | n ≤ 128（全矩阵 64KB 驻 UB，spike 校准 [128,192]） |
| Spotrf | L-大 n blocked | n > 128，b ∈ {64,128,256} spike 选定，SetTail 处理尾块 |
| Spotrs | RHS 列切分 | active = min(nrhs, 64)，每核 ⌈nrhs/active⌉ 列；nrhs=1 单核 |
| Spotri | trtri + lauum blocked | b 同 potrf |
| SpotrfBatched | BV 批量向量化 / BL 逐矩阵块化 / BC 逐矩阵 Cube | n ≤ 64 / 64 < n ≤ 512 / n > 512 |
| SpotrsBatched | 批量前代+回代列扫 | 全 n（仅 nrhs=1） |

**任务均分**：batched 按矩阵连续区间切分到 64 AIV（等分 + 余数前缀分配，确定性）；spotrs 按 RHS 列；大 n GEMM 阶段由 Matmul tiling 自动多核（AIC 28 核）。

**Lwork 契约与 Workspace 布局**（32B 对齐）：

```
[0,        b²)              inv(L11)（b×b 紧致，按 stage 覆写）
[b²,       b² + n·b)        面板紧致 staging（兼容任意 lda）
[b²+n·b,   2b² + n·b)       L11 紧致 staging
[2b²+n·b,  +M0·N0)          GEMM C tile staging（仅当无 beta=-1 通路时启用）
[尾部,     +16)              int32 标志区：stopFlag（非正定早退）等
```

`Lwork = n·b + 3·b² + M0·N0 + 16`（float 元素数；M0×N0 为 Cube 输出 tile 上界如 128×128，beta 通路确认后置 0）；n=0 返回 0。Spotrs 与两个 Batched 接口无 Workspace 参数（对齐 cuSolver）：全程 UB + GM 原地 / GM 常驻 + UB 列窗流式。

#### 3.2.2 kernel 侧设计

**aclsolverSpotrf（blocked 右视，AIC+AIV 多 kernel）**——L 路径每 stage k 三步（LOWER；UPPER 为镜像索引层，核心 kernel 复用）：

| 步骤 | kernel | 核型 | 职责 |
| --- | --- | --- | --- |
| ① 对角块 | spotrf_diag | AIV×1 | A[k:k+b,k:k+b] 入 UB → 串行小 Cholesky 得 L11 与 inv(L11) → 回写；**非正定检测** d≤0 时写 `info=全局行号` 与 stopFlag，本 stage 内停在第 k 行（前 k-1 列因子已算出） |
| ② 面板 | spotrf_panel | AIC（Matmul） | `L21 = A21·inv(L11)ᵀ`：GEMM(M=n−k−b, N=b, K=b)；lda≠n 时面板先紧致 staging |
| ③ 尾部更新 | spotrf_update | AIC（Matmul） | `A22 −= L21·L21ᵀ`：GEMM alpha=−1/beta=1（CATLASS epilogue），无 beta 通路走 C-tile staging + AIV 减法回写。LOWER 下写满 A22 矩形合法——多写的严格上三角即 A 的未使用三角（可破坏） |

S-小 n 路径：单 AIV kernel 单 launch，整矩阵入 UB 逐列右视更新（sqrt/除 + 尾部外积向量 FMA），非正定即写 info 停止。**早退守卫**：每 stage kernel 入口先读 stopFlag（GM），非零空转返回——Host 无同步、后续 stage 照常下发但不改数据，保证 info 语义与 bit-wise 确定性。n=0 仅下发 info=0 写入。

**aclsolverSpotrs（AIV 主导，RHS 列并行）**：B 各右端列代入过程相互独立 → 按列分核，零同步零原子；核内块扫描（b 列面板）：面板 L11 + 对应 B 行块入 UB → 小规模前代/回代 → 尾部 `B2 −= L21·Y1` 向量 FMA（L21 列窗流式过 UB，strided DataCopy 原生支持任意 lda/ldb）。全程 AIV：2n²·nrhs 量级（P-05 为 1.07 GFLOP）在 64 AIV 聚合 ≈7 TF/s 下 <0.2ms，无需 Cube，规避 ldb 步长 C 直写问题。info 仅 0/-i。

**aclsolverSpotri（blocked trtri + lauum）**：Phase A 自右下向左上分块求 `X = L⁻¹`（b×b 对角块求逆含**零主元检测** → info=k、stopFlag；非对角块 `X21 = −X22·L21·X11` 走 GEMM）；Phase B 对 uplo 三角按 tile (I,J), I≥J 产出 `C_IJ = X[:,I]ᵀ·X[:,J]`（K 自 I·b 起截断——X 下三角零结构），只算只写目标三角 tile，原地写 A。输出为对称逆的 uplo 三角（另一侧允许为垃圾数据）。

**aclsolverSpotrfBatched（矩阵级分核 + 核内批量向量化）**：batchSize 连续区间切分到 64 AIV，核间零交互，infoArray[i] 按归属核直写（无原子）；kernel 先从 GM 的 Aarray[i] 取基址再访存。核内三档：

1. BV（n ≤ 64，覆盖 P-09）：G 个矩阵打包为向量操作维度（G·n 元素/指令），GM 常驻 + 列窗流式过 UB；逐列步：Gather 收各矩阵主元 → sqrt/除广播（Brcb）→ 尾部批量外积更新；**非正定冻结**：向量比较产生 per-matrix 掩码，d≤0 时写 infoArray[i]=k 且该矩阵后续更新被掩码冻结，其余矩阵照常分解；
2. BL（64 < n ≤ 512）：每矩阵 blocked 小块（b=32/64）AIV 内完成；
3. BC（n > 512）：Host 按批索引循环复用单矩阵序列（kernel 按批号从 Aarray 取基址）；此段 batch 受 4G 内存约束很小（n=4096 时 ≤64），launch 次数 batch×96 可接受（n=4096×64 ≈34ms << 门槛 791ms）。

**aclsolverSpotrsBatched（AIV，仅 nrhs=1）**：同矩阵级分核；核内列扫 k=0..n-1 前代 `x[k]=b[k]/L[k,k]`（G 宽向量除）+ `b[k+1:] −= L[k+1:,k]·x[k]`（G×(n−k) 批量向量 FMA，列窗流式），再反向回代——本质 batched trsv，A 每列只读一次，带宽主导。nrhs≠1 且 n>0 → Host 直接 INVALID_VALUE + info=-3，不下发计算 kernel。

**确定性**：无原子、无跨核归约、固定阶段顺序与固定块内归约顺序、stopFlag 早退不改变已写数据 ⇒ 同输入同 stream 串行多次执行**含 info bit-wise 一致**。**数值控制**：Cube GEMM 关 HF32、L0C fp32 累加；sqrt/div 走 IEEE 向量指令；blocked 误差增长 O(n·ε·κ(A))，SPD 用例 κ 有界。

**Spike 验证计划**（开发前小规模实验，结论回填并归档）：

- S1 potrf 大 n（P-02/P-03/P-01）：候选 A. AIV 对角块 + AIC Matmul 面板/更新多 kernel（b 扫 {64,128,256}）；B. 纯 AIV blocked；C. CATLASS device 级组合。判定门：大档最优 ≥ 0.21× GPU（门槛 0.35× 的 60%）才全量开发；首轮 < 0.175×（门槛一半）触发架构熔断复盘。附带标定 950PR Cube fp32 实吞吐（2048³ GEMM 探针）、launch 开销、Matmul beta=-1 可用性。
- S2 批量（P-09~P-12）：候选 A. 矩阵级分核 + BV/BL 分档；B. 分核 + 逐矩阵纯串行；C. Cube IterateBatch 连续化（需先 gather 指针数组为连续布局，多一趟拷贝）。扫描分组 G、列窗宽度、NB_VEC/NB_AIV 分档边界。
- S3 potrs/potri 构件（P-04/P-05/P-06/P-07/P-08）：spotrs 纯 AIV 块扫描 vs AIV 面板 + Cube GEMM 尾部更新；potri AIC 为主 vs 纯 AIV（n≤2048）；b ∈ {32,64,128}。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950（Ascend 950PR，arch35/dav-3510） | √ |

CANN ≥ 9.0.0，须与 ops-solver 仓 README 已验证配套版本一致；对标性能金标采集环境为 CUDA cuSolver。

## 算子约束限制

- 仅 FLOAT32；n ∈ [1,4096]、nrhs ∈ [1,128]、batchSize ∈ [1,30000]（允许实现支持更大范围并在文档声明）；
- `aclsolverSpotrsBatched` 仅支持 nrhs=1，nrhs≠1 且 n>0 必须报错（任务书点名负向用例）；
- 不支持 broadcast，不要求图融合；Spotrs/Spotri/SpotrsBatched 的输入必须是已分解的 Cholesky 因子（uplo 须与分解时一致，对齐 cuSolver 不做跨调用校验）；
- 未使用三角可作 workspace 被破坏的许可**仅适用于 A 为输入/输出型的 potrf/potri**；potrs/potrsBatched 的 A 只读；
- 输入数据内存 ≤ 4G 约束 case 构造；核心计算禁止 CPU fallback；NPU 实现禁止依赖 CUDA。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 两步判定：① 逐元素 vs float64 golden（NumPy/SciPy cholesky/cho_solve/inv 或 dpotrf/dpotrs/dpotri）：rtol 2⁻¹⁰、atol 2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32·ULP；② 不过时按 LAPACK 残差复核：potrf/potrfBatched `‖F·Fᵀ−A‖₁/(n·‖A‖₁·ε)`、potrs/potrsBatched `max_j‖B_j−A·X_j‖₁/(‖A‖₁·‖X_j‖₁·ε)` 判 ratio ≤ max(5·ratio_cpu, 3·ratio_cpu_mean)；potri `‖I−A·C‖₁/(n·‖A‖₁·‖C‖₁·ε)` 判 ratio ≤ max(5·ratio_cpu, 0.1)；ε=2⁻²³，batched 逐矩阵判定、ratio_cpu_mean 仅 case 内统计；info 正确性属精度（非正定 10% 构造用例必须给出正确正值下标） | 任务书 §3.2 + [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能标准 | 全量 case：NPU 平均单次 kernel 耗时 ≤ bench_result.json 的 perf.avg_ms / 0.35（950PR 实测）；msprof 采集（`msprof op --application=.../test/<op>/<op>_test`，收敛到单条用例防混叠），读 OPPROF_*/OpBasicInfo.csv 按 kernel 名取 Task Duration 按调用求平均，多 kernel 算子按单次调用全部 kernel 时长合计口径；`msprof op` 不可用时回退完整模式解析 op_summary（跳过前 21 次取稳态） | 任务书 §3.3 |
| 确定性 | 合法输入重复执行输出含 info bit-wise 一致 | 任务书 §2.1/§3.2 |

**测试方案**：ops-solver 测试工程（test/<op>/，`build.sh --pkg --soc=ascend950 --ops=<op>` + `--run`）+ 任务包四脚本（gen_data / sim_dut / verify_accuracy / verify_perf；sim_dut 先行锁定接口语义含 -i 编号，避免实现与验收口径漂移）+ AscendOpTest 全量回归。功能必测矩阵：五链路（potrf；potrf+potrs；potrf+potri；potrfBatched；potrsBatched）、uplo 双侧、lda/ldb 最小值与 +8/+32 padding、info 三态（0 / i / -i 每参数至少 1 条）、批量边界（batchSize=1；n=1；nrhs=1 合法；potrsBatched nrhs=2 报错）、空问题（n=0/nrhs=0/batchSize=0 成功）、确定性（含非正定用例）、非默认 stream。输入构造：50% 均匀 [-5,5] + 50% 正态的 `A = BᵀB + nI`，另 10% 对角减大数构造非正定；n 覆盖 2 的幂与 2 的幂−1；nrhs 覆盖 1/8/32/128 及 0；batchSize 覆盖 1/8/128/1024/3000 及 0。**Day-1 清点 cases/index.json 全量条数（bench_result.json 计数合计 768）写入 PROGRESS.md，验收只认全量数字。**

**风险与兜底**：

| 风险 | 缓解/兜底 |
| --- | --- |
| 950PR Cube fp32 吞吐未标定 | S1 首周 2048³ 探针标定；纯 AIV blocked 兜底在预算内（P-02：22.9 GFLOP @ 7 TF/s ≈ 3.3ms < 12.4ms） |
| Matmul beta=-1 通路不可用 | Workspace 已预留 M0·N0 tile 项；AIV 减法回写路径已设计 |
| lda≠n 的 GEMM GM 直读步长 | 统一面板紧致 staging（锚点均 lda=n 无性能损失）；AIV strided DataCopy 原生支持 |
| 极小 n 单矩阵 case（n≤8，GPU 12~28µs，门槛 34~80µs，最紧点） | 单 kernel 单 launch（≈5.5µs）+ 微秒级计算；info 清零融合进计算 kernel |
| batched 非正定"冻结更新"掩码复杂 | spike 先验证掩码正确性（batch 内混合正定/非正定用例）；BL 路径天然逐矩阵 |
| 大 n×大批量 launch 次数多 | 该段门槛富余极大（791ms vs 预计 ≈100ms）；仍超则改单 kernel 内批循环 |
| -i 编号与 cuSolver 口径不一致 | sim_dut 对拍 + cuSolver 文档双重核对后冻结 |
| msprof op 兼容性 | 双口径回退方案（见性能标准行） |
| CANN 版本指令集差异 | 以 ops-solver README 配套版本为准，跨版本迁移前重验 |

## 兼容性分析

全新接口（ops-solver 无 Cholesky 原型），不涉及既有算子行为变更。返回值直接采用 `aclsolverStatus_t` 公开接口（不做 aclError 双轨兼容，任务书 2.2 最终验收以 aclsolverStatus_t 为准）；handle/stream 复用仓内既有管理接口，`aclsolverFillMode_t` 枚举值 0/1 对齐 `cublasFillMode_t`。NPU 接口与 CUDA 接口参数序列一致，仅命名前缀与枚举类型名不同。交付时同步更新 README 与 docs/api_list.md；五接口同一 PR 或一组关联 PR 合入 ops-solver（`src/spotrf/ + src/spotrs/ + src/spotri/ + src/spotrf_batched/ + src/spotrs_batched/ + test/… + docs/zh/…`），不得只合入部分接口。

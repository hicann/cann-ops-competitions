# aclsolverCpotrf / Cpotrs / Cpotri / CpotrfBatched / CpotrsBatched 设计文档（Atlas 950PR）

> 交付件①。按 `resources/design_template.md` 组织，四个必需一级章节齐全。
> 数据口径：任务包指标（用例数、GPU 基线、判据阈值、info 契约形态）均摘自任务书与五包
> `canonical_cases.json` / `bench_result.json` / `manifest.json` 的固化实测值；实现侧内容为
> 开发前设计分析与预算推演（标注「设计推演」），950PR 实测数据在自测阶段回填，不以估算
> 冒充实测。

---

# 需求背景（required）

## 需求来源

CANN 生态 9 月社区任务「单精度复数 Cholesky 分解、求解和批量接口(950)」。在
`ops-solver` 仓为 **Ascend 950PR（Atlas 950，`--soc=ascend950`）** 实现 5 个稠密
Hermitian 正定求解接口：`aclsolverCpotrf`、`aclsolverCpotrs`、`aclsolverCpotri`、
`aclsolverCpotrfBatched`、`aclsolverCpotrsBatched`。**五接口为同一社区任务整体交付，
不允许拆分验收。**

- 代码交付至 `https://gitcode.com/cann/ops-solver` 的 `master`（建议目录见 §3.1）；
- 设计文档交付至 `cann-ops-competitions` 仓
  `04_tasks/01_community-task-2026/tasklist/09-单精度复数Cholesky分解、求解和批量接口(950)/<用户名>/docs/`
  （按仓 README 与自动检视口径，PR 仅含 `docs/design.md` 一个文件；本文件为对应设计稿）。

## 背景介绍

### baseline 三来源（本算子族无任何同仓原型）

| # | baseline 来源 | 作用 | 位置 |
|---|---|---|---|
| 1 | NumPy/SciPy LAPACK **z 前缀链路**（complex128） | **精度 golden**：`zpotrf` / `zpotrs` / `zpotri`，golden32 由 complex128 降型 | 任务包 `verify_accuracy.py`（判定时现场重生成） |
| 2 | cuSolver DN legacy（`cusolverDnCpotrf/Cpotrs/Cpotri/CpotrfBatched/CpotrsBatched`） | **接口/功能/约束/性能标杆** | 任务包 `bench_result.json`（25 iters，avg_ms） |
| 3 | 本仓既有算子 `cgetrf` / `cgetri` / `cmatinv_batched` / `cgetri_batched` | **工程模式参照**：Host C API + kernel 直调、复数处理、批量组织 | `ops-solver` 当前 master |

ops-solver 仓当前**无复数 Cholesky 原型**（`api_list.md` 仅 cgetrf/cgetri/sgetrf/sgetri/
cmatinv_batched/cgetri_batched 六算子），须全新实现：`include/cann_ops_solver_common.h`
补 `aclFloatComplex` 与 `aclsolverFillMode_t`，`cann_ops_solver.h` 补 7 个函数原型
（含 2 个 `_bufferSize`）。

### 任务包形态（自验依据，5 包一致）

| 包 | 精度用例 | info 契约用例 | perf 关联条目 | 维度覆盖 |
|---|---|---|---|---|
| cpotrf | 145 | 3 | 145 | n∈[2,4096]，uplo L70/U75 |
| cpotrs | 176 | 3 | 176 | n∈[2,4096]，nrhs∈{1,8,16,32,64} |
| cpotri | 145 | 3 | 145 | 同 cpotrf |
| cpotrfBatched | 151 | 3 | 151 | n∈[2,4096]，batch∈[32,10⁶]（n=2 配 10⁶） |
| cpotrsBatched | 151 | 3 | 151 | 同上，nrhs=1 |

纯脚本包（零数组数据，`gen_data.py` 现场造数，判定侧同配方重生成）；逐 case 固化
`ratio_cpu`、顶层固化 `ratio_cpu_mean`；批量包为 **A0 抽样**（每 case 仅 k=min(5,batch)
个代表内容矩阵 + `sample_map` 槽位映射）：判定**先核余槽 bit-wise 一致性，后代表槽三层
判定**——这要求被测实现对"同内容不同槽位"的矩阵产出**逐位一致**（见 D8）。info 契约
用例只比 `info == k_expected`，与数值精度分别出结论。

### 功能分析：依赖结构决定架构（与 stbsv 任务的经验衔接）

1. **对角列链是串行轴**。Cholesky 第 k 列因子的产生依赖前 k−1 列更新完成（同族结构
   见 PR #1973 aclblasStbsv 设计的"列外层/行内层"结论）。但与 stbsv 的逐位一致判据
   不同：**本任务精度判据是对 complex128 golden 的容差判定**（rtol 2⁻¹⁰），内部求和
   次序有自由度；约束退化为**确定性**（同输入同 stream 串行重复执行 bit-wise 一致，
   任务书 §2.1 条 7）——固定循环次序、固定 K 分割即可满足，无需逐位对齐某个标杆的
   浮点运算次序。
2. **串行轴在块级被压缩为面板**。分块算法把列链包进 NB 宽面板（面板内列链串行、
   O(NB²·n)），O(n³) 主量落在尾随 HERK 更新（L21·L21ᴴ 形态）——这是**稠密
   matrix-matrix** 形态，与 stbsv 的带内向量更新本质不同，**可上 Cube**。面板/尾随
   两相结构 + 面板全核冗余（免跨核数据交换）+ 单道同步的面板步循环，是本设计的
   骨架（stbsv 多核列扫描 v3 的同款结论在稠密场景的推广）。
3. **五接口是算法级联**：potrf 是根；potrs = 两段三角求解（前代 + 回代）；potri =
   trtri（三角求逆）+ lauum（Hermitian 乘积）；两个 Batched 接口与单矩阵版共享分块
   组件，仅批维组织不同（指针数组设备侧解引用）。共享组件是控制代码规模的手段
   （仓规范反例 1：文件数不得明显超出正常数量）。

---

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上实现 5 接口，满足：

- 语义对齐 cuSolver DN legacy：列主序、原地覆盖、`lda/ldb ≥ max(1,n)` padding、
  Device 指针（批量接口为 Device 上的指针数组）、`devInfo/info/infoArray` 真实写入、
  `_bufferSize` + `Lwork` 工作空间协议、handle/stream 复用；
- 规模（验收下限）：potrf/potri n∈[1,4096]；potrs n∈[1,4096]、nrhs∈[1,128]；
  potrfBatched n∈[1,4096]、batchSize∈[1,10⁶]；potrsBatched 同前且**仅 nrhs=1**
  （nrhs≠1 且 n>0 必须报错）；
- 精度：三层判定（逐元素混合容差 → LAPACK ε 归一化残差复核 → info 契约独立结论），
  详见可维可测分析；
- 性能：NPU 平均单次 kernel 耗时（msprof `Task Duration`，按 kernel 名统计）≤
  GPU 参考耗时 / 0.35，全量 perf 关联条目逐条达标；
- 确定性：合法用例同 stream 串行重复执行，输出（含 info）bit-wise 一致；
- 核心计算在 NPU AI Core 完成，无 CPU fallback，无 CUDA 依赖。

## 需求拆解

| # | 需求 | 来源 | 性质 |
|---|---|---|---|
| R1 | `cann_ops_solver_common.h` 补 `aclFloatComplex`（与 `cuComplex` 布局一致）与 `aclsolverFillMode_t`（0/1 对齐 CUBLAS 枚举） | 任务书 §2.3 | 新增 |
| R2 | 7 个公开函数原型（参数名、顺序、Device/Host 归属不得增删调换，维数用 int）+ C 可调用头 | §2.3 | 新增 |
| R3 | Host：校验顺序、quick return、`aclsolverStatus_t` 状态码、`devInfo=-i` 异步写入 | §2.1/§2.4 | 规定 |
| R4 | Kernel：potrf（LOWER/UPPER）、potrs、potri、两 Batched，n≥1 全范围、lda padding | §2.1 | 新增（主体） |
| R5 | 精度三层判定 + info 契约（含 k 值语义：最小不正定顺序主子式阶数，前 k−1 列已算出） | §3.2 | 约束 |
| R6 | 性能 0.35×GPU 门限（msprof 口径，30 次采样中位数，全量 perf 条目） | §3.3 | 约束 |
| R7 | 确定性（bit-wise 可重复）+ 批量 A0 余槽 bit-wise 一致 | §2.1 条 7 / 包判定 | 约束 |
| R8 | 测试工程随代码上库 `test/cpotrf/` 等 5 目录，接任务包执行器口径（npz 三键 `out32/info/status`） | §3.5/包 README | 新增 |
| R9 | README、`docs/zh/*.md`、`docs/api_list.md` 同步更新 | §5 | 新增 |
| R10 | potrsBatched nrhs=1 拦截、空问题（n/nrhs/batchSize=0）成功返回、非默认 stream 正确 | §2.4 | 规定 |

## 范围边界

仅 COMPLEX64（交错存储 float 对）；不支持 broadcast、不做图融合；potrs/potri/
potrsBatched 的输入是**已分解因子**（非原始 A）；未使用三角按 cuSolver 口径可作
workspace 破坏（验收只比 uplo 侧三角）；不对 potrs 做奇异性检查（因子正定性由 potrf
阶段保证）；dynamic shape 由 Host 按本次调用的 n/nrhs/batchSize 出 tiling，用例数据
内存 ≤4GB。

---

# 详细设计（required）

## 算子分析

### 数学公式

**cpotrf（分块右视，LOWER：A = L·Lᴴ）**，对 j = 0, NB, 2NB, …，jb = min(NB, n−j)：

```text
1) 面板对角块：   A11 = L11·L11ᴴ          （zpotf2，UB 内非分块，列链串行）
2) 面板 TRSM：    L21 = A21 · L11⁻ᴴ       （A21 = A[j+jb:n, j:j+jb]）
3) 尾随 HERK：    A22 ← A22 − L21·L21ᴴ    （A22 = A[j+jb:n, j+jb:n]）
```

UPPER（A = UᴴU）镜像：`U11` 同式、`U12 = U11⁻ᴴ·A12`、`A22 ← A22 − U12ᴴ·U12`。

面板内非分块列链（0 基，第 k 列）——串行轴，次序固定：

```text
d = Re(A[k,k])；d ≤ 0 → info = k+1，本矩阵分解止于第 k 列（前 k−1 列已写出）
L[k,k] = √d（实数）；L[i,k] = A[i,k]/L[k,k]（i > k，实除）
A[i,j] ← A[i,j] − L[i,k]·conj(L[j,k])（j > k，i ≥ j，单项乘减）
```

NaN 语义逐字对齐 LAPACK zpotrf：判据为 `Re(diag) ≤ 0`，NaN 不触发 info，向下传播
（与 golden 链路同口径）。

**cpotrs（LOWER）**：先解 L·Y = B（分块前代），再解 Lᴴ·X = Y（分块回代）；
UPPER 对应为 Uᴴ·Y = B、U·X = Y。分块前代每步：

```text
Y1 = L11⁻¹·B1（小三角求解）；B2 ← B2 − L21·Y1（GEMM 更新）
```

**cpotri**：A⁻¹ = L⁻ᴴ·L⁻¹，两级联（参考 LAPACK ztrtri + zlauum 的分块结构）：
trtri（对角块求逆 + 级联 TRSM 得 Linv）→ lauum（分块 HERK/GEMM 累积
C = Linvᴴ·Linv 至存储侧三角）。奇异检测：Re(diag) = 0 → info = k。

**批量接口**：对 i = 0..batchSize−1 逐矩阵执行同上算法；`Aarray[i]`/`Barray[i]`
为 Device 指针数组，kernel 侧解引用。potrfBatched 逐矩阵独立写 `infoArray[i]`
（非正定即 k，参数错写 `infoArray[0] = -i`）；potrsBatched 的 `info` 为标量、仅报
参数错（正定性属 potrfBatched 职责，cuSOLVER 手册 Remark 2 同口径）。

### 支持数据类型

COMPLEX64（`aclFloatComplex`：连续两个 FLOAT32，real 在前），A/B/Workspace 均为
Device 指针；info 系 INT32；维数/步长 int（32-bit，对齐 CUDA legacy）。

### 支持形状

A：lda×n 列主序（只需 uplo 三角有效，lda > n 合法）；B：ldb×nrhs；批量：指针数组
长度 batchSize。n∈[1,4096]（实现不设上限，泛化由 UB/GM 分块保证），nrhs∈[1,128]，
batchSize∈[1,10⁶]。

## 算子实现

### 3.1 总体结构

```text
include/cann_ops_solver.h            +7 原型（5 计算接口 + 2 bufferSize）
include/cann_ops_solver_common.h     +aclFloatComplex / aclsolverFillMode_t
src/cholesky_common/                 共享组件（五算子复用，控文件数）：
  ccholesky_layout.h/.cpp              布局变换：交错↔实虚平面（split-plane）
  ccholesky_panel.h/.cpp               面板：zpotf2 列链 / 小三角求逆 / NB 级 TRSM 布局准备
  ccholesky_gemm.h/.cpp                复数 GEMM/HERK 封装（3M 与 4-GEMM 两种实例化）
  fill_info_kernel.cpp                 info 微 kernel（写 -i / 0）
src/cpotrf/  src/cpotrs/  src/cpotri/
src/cpotrf_batched/  src/cpotrs_batched/   各含 <op>_host.cpp / <op>_kernel.cpp
test/cpotrf/ ... test/cpotrs_batched/      执行器 + GTest + 包对接
docs/zh/cpotrf.md ... docs/api_list.md README 同步
```

构建：`bash build.sh --pkg --soc=ascend950 --ops=cpotrf`（逐算子/组合均可）；
测试二进制 `build/test/<op>/<op>_test`。

### 3.2 公共设计决策

- **D1 布局——实虚平面（split-plane）**：交错复数对 Cube 实数 GEMM 不友好。potrf/
  potri 有 workspace：进入 kernel 后一次性把存储侧三角拆为实平面 + 虚平面（全平面
  n×n float，共 8n² B，与原三角数据量同阶），全程平面形态计算，结束时平面→三角
  写回 A（对角虚部按 0 语义写 0）。拆合仅 2 次整阵遍历；若保持交错形态逐面板拆合，
  尾随块将被额外读写 ~n/(NB) 次，拆合开销随分块步数放大（设计推演，NB=256、
  n=4096 时约 8:1）。potrs 无 workspace 参数：L21 面板块按用拆入 UB（右视算法
  每块只读一次，无重复访存）；B 做**就地平面化重排**（平面与交错总量相等，分块
  UB 双缓冲搬运，容量不变），结束时重排回交错。
- **D2 复数 GEMM——3M 分解为默认**：以 A·Bᴴ 为例（P=(pr,pi), Q=(qr,qi) 平面）：

  ```text
  m1 = pr·qrᵀ；m2 = pi·qiᵀ；m3 = (pr+pi)·(qr+qi)ᵀ
  ΔC.re = m1 + m2；ΔC.im = m3 − m1 − m2
  ```

  相对 4-GEMM 节省 25% Cube 量，代价是两趟廉价逐元素加/减。TRSM 的
  `L21·L11⁻ᴴ` 取 `B=W=L11⁻¹` 代入同式（(a+bi)(c−di) = ac+bd + i(bc−ad)）。
  4-GEMM 实例化保留为回退（数值口径等价，见可维可测分析）。
- **D3 uplo 双原生路径**：LOWER/UPPER 以模板参数区分（面板读侧、TRSM 转置方向、
  HERK 共轭侧镜像），不做物理翻转（翻转破坏 stride-1 访存）。两路径共享全部组件。
- **D4 确定性**：tiling 只由 n/nrhs/batchSize 决定；循环次序、GEMM 的 K 分割固定；
  计算路径无原子浮点累加（跨核同步只用于控制流）；数值与矩阵所处槽位/地址无关
  （对齐只影响 DMA 效率不影响值）→ 同输入重复执行 bit-wise 一致，且**构造性满足
  A0 余槽一致性**（同内容矩阵在任意槽位产出逐位相同）。
- **D5 info 契约**：参数错——Host 判定后返回 `ACLSOLVER_STATUS_INVALID_VALUE`，
  同时在**调用方 stream 上异步下发 `fill_info` 微 kernel**（blockDim=1，写
  `devInfo=-i`；批量写 `infoArray[0]=-i`；devInfo 空指针时仅返回状态码），
  无 Host 同步、无 Device 回读。正值 k——面板列链检测 `Re(diag) ≤ 0` 即
  `info=k+1` 并截断该矩阵后续面板/尾随相位（已算出的前 k−1 列保留）；NaN 不触发。
  空问题（n/nrhs/batchSize=0）成功返回并写 info=0（batchSize=0 无槽位，不写）。
- **D6 校验顺序**（第一失败项写 -i，i 为除 handle 外的 1 基参数序）：
  potrf：uplo=1 → n=2 → A=3（n>0 时空指针）→ lda=4 → Workspace=5（Lwork>0 时空
  指针）→ Lwork=6（小于所需）→ devInfo=7。potrs：uplo=1, n=2, nrhs=3, A=4,
  lda=5, B=6, ldb=7, devInfo=8。potri 同 potrf。potrfBatched：uplo=1, n=2,
  Aarray=3, lda=4, infoArray=5, batchSize=6。potrsBatched：uplo=1, n=2, nrhs=3,
  Aarray=4, lda=5, Barray=6, ldb=7, info=8, batchSize=9（nrhs≠1 且 n>0 →
  INVALID_VALUE + info=-3）。`_bufferSize`：Lwork 指针空 → INVALID_VALUE。
- **D7 流语义**：全部 kernel 下发到 `aclsolverGetStream(handle)` 的调用方 stream；
  不做 `aclrtSynchronizeStream`；复用 `aclsolverCreate/Destroy/SetStream/GetStream`。
- **D8 批量指针数组**：`Aarray/Barray` 只在 kernel 侧解引用（Host 不读 Device
  内存）；批量接口无 workspace 参数，大 n 组暂存采用 **handle 缓存式内部分配**
  （Host 已知 n/batchSize 标量，按需 `aclrtMalloc`、destroy 释放；legacy
  cusolverDnCpotrfBatched 亦为内部工作区口径，自测报告披露该行为）。
- **状态码**：计算接口统一 `aclsolverStatus_t`，语义对齐 cusolverStatus_t
  （SUCCESS/NOT_INITIALIZED/ALLOC_FAILED/INVALID_VALUE/INTERNAL_ERROR…，数值表随
  代码 PR 在头文件注释冻结）；与现存 `aclError` 原型的对照表按任务书 §2.2 要求
  随文档给出。

### 3.3 Host 侧设计

- **路径分派（tiling 的核心）**，全由 Host 标量决定：

  | 接口 | 路径 | 条件 | 结构 |
  |---|---|---|---|
  | cpotrf/potri | tiny | n ≤ 64 | 单核，整阵 UB，非分块列链，blockDim=1 |
  | | 分块主路径 | n > 64 | split-plane WS + 面板（AIV，全核冗余免数据交换）+ TRSM/HERK（Cube，3M）+ 每面板步 2 道整栅格同步 |
  | cpotrs | 向量路径 | nrhs=1 或 n ≤ 64 | 单核/少核，列链前代回代 |
  | | 分块路径 | 其余 | B 就地平面化 + 面板小三角解（AIV）+ B2−=L21·Y1（Cube GEMM，3M） |
  | 批量×2 | 打包 lane | n ≤ 48 | 见 3.4.4 |
  | | 组并行分块 | n > 48 | 见 3.4.5 |

  NB 自适应 64~256：n 大取大 NB（压尾随访存），面板串行占比大取小 NB（压面板
  O(NB²·n)）；AIV/Cube 核数与 UB 容量经平台接口（`PlatformAscendCManager`）运行期
  获取，不写死。
- **bufferSize**（COMPLEX64 元素计，设计推演，余量随实测收紧）：
  `Lwork_potrf = n² + NB·(2n+NB) + 64`（两平面 + 面板双缓冲）；
  `Lwork_potri = 2n² + NB·(2n+NB) + 64`（Linv 平面与结果平面各一）。调用方
  Lwork 小于所需 → INVALID_VALUE（i=6）。
- **quick return**：handle 空 → NOT_INITIALIZED；n=0 / nrhs=0 / batchSize=0 →
  SUCCESS（写 info=0 后直接返回，n>0 时才要求指针非空）。

### 3.4 Kernel 侧设计

#### 3.4.1 cpotrf 分块主路径（n > 64）

```text
相位0  布局：存储侧三角 → 平面（分块 DMA，32B 对齐段 + 尾块 Pad）
循环 j = 0, NB, …（每面板步）：
  相位1  面板（AIV，全核冗余）：zpotf2(A11) 列链（次序固定，单项乘减）
         + L11⁻¹（NB×NB 三角求逆，UB 内）+ L21 平面布局就绪；
         0 号核将 L11/L21 写回 GM；d ≤ 0 → 写 info=k+1，本矩阵退出循环
  同步①  整栅格 SyncAll（面板 GM 落地对全体可见）
  相位2  尾随（Cube，3M）：A22 ← A22 − L21·L21ᴴ（K=jb，beta=1 原位累加）；
         AIV 并行做下一面板步的 L21 预取布局（双缓冲重叠）
  同步②  SyncAll
相位3  平面 → 三角写回 A（对角虚部写 0；未用三角按 cuSolver 口径可破坏）
```

面板冗余成本 O(NB²·n)·核数，取小 NB 时占比低（NB=128、n=4096 时面板相位总量
约为尾随的 1/16，冗余后 ~2.5/16，设计推演），换取：面板数据零跨核交换、无第三道
同步（stbsv 设计同款结论）。lda > n 时相位 0/3 的行步进取 lda（列内仍 stride-1）。

#### 3.4.2 cpotrs

B 就地平面化重排（分块 UB 双缓冲，n·nrhs 总量不变）后：

```text
前代（LOWER）：j = 0, NB, …：Y1 = L11⁻¹·B1（AIV 小三角解，含 3M 小 GEMM 级联）
               B2 ← B2 − L21·Y1（Cube，3M；右视，L 每块只读一次）
回代（LOWER）：j = n−NB, …：X1 = L22⁻ᴴ·B1；B2 ← B2 − L12ᴴ·X1（镜像）
重排回交错，原地写回 B；UPPER 对称镜像
```

nrhs=1 走向量路径（列链前代/回代，B 常驻 UB）；大 nrhs（≥8）GEMM 收益随 nrhs
放大。A 只读不改（重排只作用于 B）。

#### 3.4.3 cpotri

trtri：j 循环内对角块 D⁻¹（AIV，UB）+ 级联 `L21·L11⁻¹`（Cube，3M，负号吸收进
布局）；lauum：分块 HERK 累积 `C = Linvᴴ·Linv`（对角块 HERK + 交叉块 GEMM），
只产出存储侧三角。两平面（Linv、C）并列，无三角镜像来回搬运。

#### 3.4.4 批量接口之一：打包 lane 路径（n ≤ 48）

性能主力（P-09/P-11 均在此档）。布局按 `[pos][g]`：pos 为三角列主序一维位置，
g 为组内矩阵序——固定 (i,j) 跨矩阵连续，向量维即批维：

```text
解引用：每核按段读 Aarray[i]，整阵三角 DMA 入 UB 打包布局（组大小 G=f(n,UB)，8 对齐）
列链 k = 0..n-1（串行轴）：
  对角 gather（跨矩阵向量）→ 逐 lane 判 d ≤ 0（写 infoArray，本矩阵置停算掩码）
  → sqrt → 实除缩放列 → rank-1：for j>k, i≥j：vec FMA（单项乘减，无跨 lane 归约）
打包 → 原位写回各槽位三角；infoArray 逐矩阵
```

**无跨 lane 归约**是该路径的构造性要点：单矩阵数值只依赖自身输入 → 槽位/组位置
无关，直接满足 A0 余槽一致性。指令量（设计推演）：P-09（n=32，b=102774，G≈32，
40 AIV）约 0.3ms 量级，对 8.47ms 预算余量充足；P-11（n=32 potrsBatched）更轻。

#### 3.4.5 批量接口之二：组并行分块路径（n > 48）

大 n 批量是最紧性能门限（见可维可测分析吞吐表），须保持 Cube 占用率：

```text
静态分组：矩阵 i 归组 i mod G（G 由 n 定：n ≥ 2048 取 4，512 < n < 2048 取 8~16）
组间零同步；组内 = 3.4.1 的分块算法（面板 AIV 冗余 + 尾随 Cube），矩阵顺序推进；
组内同步用 GM 邮箱屏障（计数器 + 双缓冲轮转），不占用整栅格 SyncAll，
避免组间互相牵制；某矩阵 info 截断只跳过自身后续相位，不影响他组
组暂存（split-plane）：D8 的内部工作区，n²·G_active·8B 上限，handle 缓存
```

尾随 HERK 在组内以"就绪矩阵批量 GEMM"聚合下发（静态分组的固定 M 路轮转，
CatLASS batched GEMM，3M），摊薄小 K 时的 Cube 启动开销。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950（Ascend 950PR，ascend950） | √ |

CANN 9.0.0 及以上（与 ops-solver 仓 README 已验证配套版本一致）。

## 算子约束限制

1. 仅 COMPLEX64；A/B/Workspace/info 均为 Device 指针，批量入参为 Device 指针数组。
2. `aclsolverCpotrsBatched` 仅支持 nrhs=1；nrhs≠1 且 n>0 返回 INVALID_VALUE（info=-3）。
3. potrs/potri/potrsBatched 输入须为 potrf( 或 potrfBatched) 产物；uplo 须与分解一致
   （不一致属调用方错误，结果未定义，同 cuSolver 口径）。
4. 未使用三角可被破坏（对齐 cuSolver）；验收只比 uplo 侧三角；对角为实数语义
   （写 0 虚部）。
5. 不支持 broadcast、不做图融合、无 CPU fallback；Host 不读 Device 数据
   （含指针数组）、无必要不同步。
6. 大 n 批量接口存在 handle 缓存式内部工作区（D8），上限随 n/batchSize 计算，
   文档与自测报告披露。

---

# 可维可测分析（required）

## 精度标准/性能标准

**精度（任务书 §3.2，三层）**：

1. 逐元素混合容差（实/虚部分别判定）：rtol=2⁻¹⁰（9.77e-4）、atol=2⁻¹⁶（1.53e-5）、
   required_matched_ratio=0.99、max_abs_error_limit=1e-2 or 32×ULP；golden 为
   complex128 z 链路降型。
2. 不过关时 LAPACK 残差复核（ε=2⁻²³）：
   potrf/Batched `‖F·Fᴴ−A‖₁/(n·‖A‖₁·ε) ≤ max(5·ratio_cpu, 3·ratio_cpu_mean)`；
   potrs/Batched `max_j‖B_j−A·X_j‖₁/(‖A‖₁·‖X_j‖₁·ε)` 同阈值式；
   potri `‖I−A·C‖₁/(n·‖A‖₁·‖C‖₁·ε) ≤ max(5·ratio_cpu, 0.1)`。
   Batched 逐矩阵判定，ratio_cpu_mean 在 case 内统计。残差超阈可举证申诉。
3. info 契约独立结论（每算子 3 例）：非正定/奇异/非法参数的 `k_expected` 精确比对。

TRSM-as-GEMM（显式 L11⁻¹）与 LAPACK TRSM 的舍入路径不同，但两者同为向后稳定
算法，残差同阶（2⁻¹⁰ 容差下裕量充分，设计推演）；如个别 case 触发复核，按第 2
层判据与申诉通道处理。

**性能（任务书 §3.3）**：NPU 平均单次 kernel 耗时 ≤ GPU/0.35，msprof `OpBasicInfo.csv`
按 kernel 名统计 `Task Duration`（30 次采样中位数，`--warm-up` 预热，单 case 收敛
采集避免 kernel 混叠），与 `bench_result.json`/`perf_baseline.json` 逐条关联
（145/176/145/151/151 条）。

**关键预算（设计推演，flops 按 4n³/3 口径与 bench gflops 对齐）**：

| case | GPU avg_ms | 预算 /0.35 | 派生持续需求 | 应对路径 |
|---|---|---|---|---|
| cpotrf n=4096 U | 6.7918 | 19.4 ms | 4.7 TF（3M 后 3.5 TF 实 GEMM） | 分块主路径 |
| cpotrf n=1024 L（P-01） | 0.79 | 2.26 ms | 小型化 | 分块主路径（NB 收缩） |
| cpotrs n=4016 nrhs=16 U | 4.4521 | 12.7 ms | 0.16 TF + 三角单遍访存 ~10 GB/s | 分块 TRSM |
| cpotri n=4096 U | 17.8843 | 51.1 ms | 3.6 TF | trtri+lauum 双平面 |
| **cpotrfBatched n=3527 b=151** | **326.0** | **931 ms** | **9.5 TF（3M 后 7.1 TF）★最紧** | 组并行 + 批量 HERK 聚合 |
| cpotrfBatched n=32 b=102774（P-09） | 2.9648 | 8.47 ms | 0.53 TF（指令 issuing 约束 ~0.3ms 推演） | 打包 lane |
| cpotrsBatched n=4096 b=63 | 69.42 | 198 ms | 21 GB/s 访存 | 组并行（轻） |
| cpotrsBatched n=128 b=45897（P-12） | 6.0311 | 17.2 ms | 2.5 TF | 组并行 + 批量 GEMM |

单矩阵门限只需 4.7 TF 持续，而批量大 n 需 9.5 TF——因为 GPU 批量接口每矩阵速率
高于其单矩阵速率（batch 填充更满）。这是组并行设计（保持 Cube 占用）与 3M 默认
开启的直接依据。950PR 具体余量以实测回填；若不足，先调 NB 与组数，再启 4-GEMM/
访存合并优化。

## 测试策略

1. **包驱动自验（主口径）**：`gen_data.py --select all` 造数（cpotrf 145 + info 3，
   余类推）→ `test/<op>/<op>_test` 执行器从 `data/cases/*.npz` 读入、调 aclsolver
   接口、产 `dut_out/<case_id>.npz` 三键（`out32`/`info`/`status`；半三角另侧置 0、
   批量按批维堆叠）→ `verify_accuracy.py` 三层判定 → msprof 采 30 次中位数 →
   `verify_perf.py` 对照。复现步骤入测试 README（交付件 2 要求）。
2. **包外补充（任务书 §3.5 必测场景）**：lda/ldb padding（+8/+32）；n=1、
   batchSize=1、nrhs=1/128 边界与 nrhs=0；potrsBatched nrhs=2 报错；空问题；
   非默认 stream；同输入重复执行 bit-wise 比对（确定性）；NaN/Inf 入参
   （info 语义与 opbase 规则）；uplo 全覆盖；批量混合正定/非正定（其余矩阵照常
   分解）。
3. **分层快回归**：tiny/分块/批量 × uplo × 正定/非正定 抽样快集（分钟级），
   全量包用例作出厂回归；历史失败形状滚动纳入。
4. **msprof 归因**：`--kernel-name` 锁定主计算 kernel（`fill_info` 等微 kernel
   耗时可忽略且不混入目标统计）。

## 兼容性分析

- 新增类型/枚举/原型只增不改；既有六算子与 handle 接口零改动；头文件保持 C 可调用
  （`extern "C"`，无 C++ 类型泄漏——`aclFloatComplex` 为 POD）。
- `aclsolverStatus_t` 与现存 `aclError` 原型的对照表随 `docs` 给出；最终验收以
  `aclsolverStatus_t` 公开接口为准（任务书 §2.2）。
- CANN 9.0.0+；构建经 `build.sh --soc=ascend950` 纳入，不影响其他 soc 变体。

---

# 风险点

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| 1 | 复数 Cube 效率（3M 有效吞吐、K=NB 中小 GEMM 效率）不达推演 | 单矩阵/批量门限 | NB 与组数扫描调参；批量 HERK 聚合；4-GEMM 回退对照；950PR 实测后回填预算表 |
| 2 | 批量大 n 持续 9.5 TF 门限（最紧约束） | P 表外全量条目 | 组并行保持占用；NB↑ 压尾随访存；必要时单矩阵版与批量版共享优化 |
| 3 | potrs 无 workspace 的 B 就地平面化重排（分块双缓冲边界） | 正确性 | 专项单测（nrhs=1..128 × padding）；与 potri 双平面路径交叉验证 |
| 4 | 邮箱屏障活性（组内死锁、计数器轮转） | 批量大 n 挂死 | 双缓冲计数器 + 段上限保护；组内压力用例（batch=10⁶ 小 n 走打包路径隔离验证） |
| 5 | info 语义边界（NaN、混合批、batchSize=0 指针空） | info 契约 FAIL | 判据逐字对齐 LAPACK；包内 3 例 + 包外 NaN/混合批用例 |
| 6 | A0 余槽一致性被布局/对齐差异破坏 | 批量包判定 FAIL | D4 构造性无地址依赖；打包路径无跨 lane 归约；包判定回归 |
| 7 | 面板冗余在小 n 区间占比上升 | 中 n 性能 | NB 自适应下限 64；tiny 路径上界 64；实测划界 |
| 8 | potrsBatched nrhs≠1 拦截遗漏（n=0 放宽口径） | 契约/合规 | D6 参数序校验表 + 单测（nrhs=2/0/1 × n>0/=0） |

# 交付物

1. **算子设计文档**：本文件（PR 提交时按仓规范置于
   `tasklist/09-单精度复数Cholesky分解、求解和批量接口(950)/<用户名>/docs/design.md`，
   PR 仅含该文件）。
2. **代码**（PR 至 `ops-solver` master，五接口同一 PR 或一组关联 PR）：
   `include/` 两头文件增量 + `src/cholesky_common/` + 5 个 `src/<op>/` +
   5 个 `test/<op>/` + `docs/zh/*.md` + `docs/api_list.md` + README 支持表。
3. **自测用例与测试代码**：任务包执行器对接 + 包外补充场景 + 复现 README。
4. **自测报告**：精度（三层 + info 契约）、性能（msprof 中位数 × 全量关联条目）、
   确定性、环境版本（CANN/NPU 950PR/驱动/CUDA 采集侧版本）；
   私仓 `task_submission/` 目录按任务书 §4 结构归档。

# 【社区任务】9月社区任务-单精度复数Cholesky分解、求解和批量接口(950)设计文档

> 按仓上 `04_tasks/01_community-task-2026/resources/design_template.md` 组织，四个必需一级章节齐全。
> 工程事实（目录结构、CMake 映射、既有组件、头文件契约）均核对自 `cann/ops-solver` master（2026-10 浅克隆），
> 引用处给出真实文件路径；算法与性能参数为**拟定值**（本方案为开发设计，尚未板卡实测），
> 凡涉及实测结论处均明确标注，不以估算冒充实测。

---

# 需求背景（required）

## 需求来源

CANN 生态 9 月社区任务「单精度复数 Cholesky 分解、求解和批量接口(950)」，任务书
`Atlas950_Cpotrf_Cpotrs_Cpotri_CpotrfBatched_CpotrsBatched_task_doc.md`。在
`https://gitcode.com/cann/ops-solver` 为 **Ascend 950PR（Atlas 950，SOC `ascend950`，
`dav-3510`）** 实现单精度复数（COMPLEX64）Hermitian 正定矩阵的 Cholesky 分解、求解、
求逆与批量能力，共 **5 个计算接口 + 2 个 bufferSize 查询接口**，整体交付。

- 代码交付：PR 至 `cann/ops-solver` `master`（`src/`、`test/`、`include/`、`docs/`）。
- 设计文档交付：PR 至 `cann/cann-ops-competitions`
  `04_tasks/01_community-task-2026/tasklist/09-{任务编号}-{算子名称}/docs/design.md`
  （合入路径规范见 §交付物；设计 PR 仅含 `docs/design.md`）。

## 背景介绍

### ops-solver 现状（已核对 master 源码）

| # | 事实 | 位置（已核对） |
|---|------|----------------|
| 1 | 句柄/流管理已有：`aclsolverCreate/Destroy/SetStream/GetStream`，C ABI | `include/cann_ops_solver.h:45-79` |
| 2 | 状态码 `aclsolverStatus_t`（INVALID_VALUE=3、HANDLE_IS_NULLPTR=9 等 12 项）在**公共 C 头** | `include/cann_ops_solver_common.h:26-42` |
| 3 | 既有 7 个数值算子（cgetrf/sgetrf/cgetri/sgetri/cgetri_batched/cmatinv_batched/cheevj）为 **C++ 区域声明**：`std::complex<float>` + `int64_t` 维数 + 返回 `aclError` | `include/cann_ops_solver.h:111-207` |
| 4 | `aclsolverFillMode_t`（LOWER=0/UPPER=1）已存在，但在 C++ 类型区域 | `include/cann_ops_solver.h:91-93` |
| 5 | CMake 已支持 `ascend950*` → `NPU_ARCH=dav-3510` + 架构目录 `arch35`，`src/CMakeLists.txt` 自动按 SOC 收集 `*/arch35/*.cpp` | 根 `CMakeLists.txt`、`src/CMakeLists.txt` |
| 6 | 共享复数 kernel 组件已存在：`utils/kernel/c32/` 下 `gemm.hpp`（matmul API 封装）、`trsm.hpp`（`SolveTrsm`，**maxN=256**）、`getf2.hpp`、`pad.hpp` 等 | `src/utils/kernel/c32/` |
| 7 | kernel 侧跨核同步基建已有先例：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` + `SetSyncBaseAddr(sync)` | `src/cgetri_batched/cgetri_batched_kernel.cpp:38-40` |
| 8 | 测试工程约定：`test/{op}/{op}_test.cpp` + `CMakeLists.txt` + `data/gen_data.py` + `data/verify_result.py` + `README.md` | `test/cgetri_batched/` 等 |

### 功能缺口

| # | 缺口 | 说明 |
|---|------|------|
| 1 | Cholesky 原型整体缺失 | 仓内无 potrf/potrs/potri 及任何批量 Cholesky 实现（LU/特征值/求逆有，Cholesky 无） |
| 2 | C ABI 复数契约缺失 | 任务要求 `aclFloatComplex`（C struct，8 字节）+ `int` 维数 + `aclsolverStatus_t` 返回；与既有 `std::complex` + `int64_t` + `aclError` 契约并行存在，须新增公共类型且不破坏既有 ABI |
| 3 | `aclsolverFillMode_t` 位置不满足 C 调用 | 需迁移至 `cann_ops_solver_common.h`（取值 0/1 不变），C++ 区域去重 |
| 4 | Device 指针数组批量形态仓内无先例 | 既有 batched 接口为**连续三维布局**（`[batchSize,n,n]`，见头文件注释）；本任务 Aarray/Barray 为 Device 指针数组，各矩阵可分散分配 |

### 算法背景与依赖结构

Cholesky 分解 `A = L·Lᴴ`（UPPER：`A = Uᴴ·U`）与 LU 同为 O(n³) 逐主元推进，但无选主元：
**面板（对角块+面板三角求解）严格串行，尾矩阵更新高度并行**。该依赖结构决定实现形态：
跨面板靠 kernel 间 stream 顺序（多 kernel 编排）或核内栅栏衔接，面板内与尾矩阵更新按 tile 分核。
求解（potrs）单 RHS 时为**严格串行链**（x[i] 依赖 x[0..i-1]，n 步深度），与 aclblasStbsv
（PR #1973）同构——其"列外层/行内层、每语句单项乘减"的精度结论在本任务单 RHS 路径直接复用。

---

# 需求分析（required）

## 需求描述

七个接口的公开原型（参数名、顺序、类型不得增删调换，维数为 32 位 `int`，对齐 cuSOLVER legacy 形态）：

```c
/* cann_ops_solver_common.h 新增（C 兼容） */
typedef struct { float real; float imag; } aclFloatComplex;

/* cann_ops_solver.h extern "C" 区域新增 */
aclsolverStatus_t aclsolverCpotrf_bufferSize(aclsolverHandle_t handle,
    aclsolverFillMode_t uplo, int n, aclFloatComplex *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverCpotrf(aclsolverHandle_t handle,
    aclsolverFillMode_t uplo, int n, aclFloatComplex *A, int lda,
    aclFloatComplex *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverCpotrs(aclsolverHandle_t handle,
    aclsolverFillMode_t uplo, int n, int nrhs, const aclFloatComplex *A, int lda,
    aclFloatComplex *B, int ldb, int *devInfo);
aclsolverStatus_t aclsolverCpotri_bufferSize(aclsolverHandle_t handle,
    aclsolverFillMode_t uplo, int n, aclFloatComplex *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverCpotri(aclsolverHandle_t handle,
    aclsolverFillMode_t uplo, int n, aclFloatComplex *A, int lda,
    aclFloatComplex *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverCpotrfBatched(aclsolverHandle_t handle,
    aclsolverFillMode_t uplo, int n, aclFloatComplex *Aarray[], int lda,
    int *infoArray, int batchSize);
aclsolverStatus_t aclsolverCpotrsBatched(aclsolverHandle_t handle,
    aclsolverFillMode_t uplo, int n, int nrhs, aclFloatComplex *Aarray[], int lda,
    aclFloatComplex *Barray[], int ldb, int *info, int batchSize);
```

能力矩阵（任务书口径）：

| 接口 | 语义 | 规模 | info（Device，int32） |
|---|---|---|---|
| Cpotrf | A 的指定三角**原地**覆写为因子 F | n=0…4096，LOWER/UPPER | 0；-i 参数错；k 首个失败主元（1-based） |
| Cpotrs | F 只读，B（n×nrhs）**原地**覆写为 X | n=0…4096，nrhs=0…128 | 0 / -i（无数值失败） |
| Cpotri | F **原地**覆写为 A⁻¹ 的指定三角 | n=0…4096 | 0；-i；k 首个零对角（1-based） |
| CpotrfBatched | Aarray 各矩阵原地分解 | n=0…4096，batchSize=0…1000000 | 逐矩阵 infoArray[i]：0/k；参数错写 infoArray[0]=-i |
| CpotrsBatched | Aarray 只读，各 B 原地求解 | 同上，非空时 **nrhs=1** | 单标量：0 / -i |

存储契约：列主序，元素地址 `base + row + col*ld`；`lda/ldb ≥ max(1,n)`，可带 padding，
**padding 与非指定三角不读不写**（反三角可被 POTRF 用作工作区，测试不要求其保持原值）；
`Lwork` 单位为**复数元素数**（非字节）；矩阵/B/Workspace/info 均为 Device 内存，`Aarray`/
`Barray` 为 Device 上的指针数组，各元素指向独立 Device 矩阵（非连续三维张量）；Host 不做
流同步，调用方读结果前自行同步。

## 需求拆解

| # | 需求 | 来源 | 性质 |
|---|---|---|---|
| R1 | 公共类型与 7 接口 C ABI 落位（fill mode 迁移、`extern "C"`、C/C++ 双编译） | 任务书 §接口 | 新增（契约） |
| R2 | Host 校验/quick-return/-i 序号/异步 info 写入，与 cuSOLVER legacy 语义逐条一致 | 任务书 | 规定 |
| R3 | POTRF：分块右视算法（面板串行 + 尾矩阵并行），失败主元中止且部分结果保留 | 任务书 | 新增（主体） |
| R4 | POTRS：双三角求解；nrhs=1 串行链保精度形态，nrhs>1 分块 + GEMM 更新 | 任务书 | 新增 |
| R5 | POTRI：零对角检查 → 三角逆 T → C=TᴴT（或 TTᴴ）→ 仅写指定三角 | 任务书 | 新增 |
| R6 | Batched：Device 指针数组真语义、逐矩阵失败隔离、batchSize 至 10⁶ 不逐矩阵 launch | 任务书 | 新增 |
| R7 | 精度：complex128 golden，元素级混合容差 + LAPACK 残差双判据（含任务书/随包脚本双口径，见 §可维可测） | 任务书 §3.2 | 约束 |
| R8 | 性能：有 GPU 基线用例逐条 `T_NPU ≤ T_GPU/0.35`（msprof kernel 口径） | 任务书 §3.3 | 约束 |
| R9 | 确定性：同输入同流串行重复运行输出与 info 逐位一致 | 任务书 | 需保证 |
| R10 | 测试工程随代码上库（官方 cases 驱动 + 补充回归），文档/示例/支持表同步 | 仓规范 | 新增 |

## 范围边界

仅 COMPLEX64；仅列主序；无选主元（Cholesky 语义）；不支持 strided-batch、批量多 RHS、
行主序、COMPLEX128；不做输入正定性/有限性前置检查（数值失败经 Device info 报告）；
A/B/Workspace/info 合法性与不重叠为调用方前置条件（裸指针接口无法完整检测，仅校验
Host 可见标量与外层指针非空）。

---

# 详细设计（required）

## 算子分析

### 数学公式

记 `H` 为共轭转置，`A₀` 原始正定矩阵，`F` 因子，`C` 逆矩阵，0 基下标，`d(i)=Re(A(i,i))`。

**POTRF**（分块右视，块宽 b；LOWER 为例，UPPER 取对偶）：

```text
for j = 0, b, 2b, ...:
  A11 = L11·L11ᴴ            # 对角块分解（面板内逐主元，串行）
  L21 = A21·L11⁻ᴴ           # 面板三角求解（TRSM）
  A22 ← A22 − L21·L21ᴴ      # 尾矩阵 Hermitian 秩-b 更新（并行）
面板内逐主元（0 基）：
  d = Re(A[j,j]) − Σ_{k<j} |L[j,k]|²          # d ≤ 0 或非有限 → info = j+1，该矩阵中止
  L[j,j] = sqrt(d)（虚部写 +0.0）
  L[i,j] = (A[i,j] − Σ_{k<j} L[i,k]·conj(L[j,k])) / L[j,j]
```

**POTRS**（两次三角求解，RHS 列间独立可并行）：

| uplo | 第一步 | 第二步 |
|---|---|---|
| LOWER | `L·Y = B`（前代） | `Lᴴ·X = Y`（回代） |
| UPPER | `Uᴴ·Y = B`（前代） | `U·X = Y`（回代） |

**POTRI**：先查零对角（`L[j,j]==0 → info=j+1` 中止），再 `T = L⁻¹`（对单位阵分块三角
求解），再 `C = Tᴴ·T`（LOWER；UPPER 为 `C = T·Tᴴ`，`T = U⁻¹`），仅写指定三角，对角虚部
写 `+0.0`。

**复数运算**：`(a+ib)(c+id) = (ac−bd) + i(ad+bc)`，共轭为虚部取反；对角主元按 FP32 实数
比较与开方；禁止把 Hermitian 更新化简为实对称运算。

### 支持数据类型

`aclFloatComplex`（8 字节，real/imag 偏移 0/4，编译期 `static_assert(sizeof==8)` 对齐
`cuComplex` 布局）；n/nrhs/lda/ldb/Lwork/batchSize/info 公开 `int32`；**内部寻址/偏移一律
uint64/int64**（batchSize=10⁶ × n² 体量下防 32 位溢出）。NPU 计算全程 FP32 实数算术；
complex128 仅测试侧 golden。

### 支持形状

n=0（合法空问题，各接口按 §Host 侧规则 quick return）；n=1…4096；nrhs=0…128（批量非空
仅 1）；batchSize=0…1000000；lda/ldb ≥ max(1,n) 且可任意大（有效区 n 行）；Aarray/Barray
元素可不连续分配。实际总规模受 Device 内存约束（canonical 大 batch 用例输入可达 ~15 GiB，
见 §可维可测口径说明）。

## 算子实现

### 总体结构（新增文件，复用仓内构建自动收集）

```text
include/cann_ops_solver_common.h      +aclFloatComplex；+aclsolverFillMode_t（自 .h 迁移，值不变）
include/cann_ops_solver.h            extern "C" 区域 +7 声明；C++ 区域 fill mode 去重
src/utils/cholesky_host.h            共享校验/-i 写入/空问题处理
src/utils/kernel/c32/chol_common.hpp 复数四实乘/共轭/列主序索引/块遍历公共层
src/cpotrf/        cpotrf_host.cpp  cpotrf_kernel.cpp（面板/尾更新 kernel 族）
src/cpotrs/        cpotrs_host.cpp  cpotrs_kernel.cpp（列扫描/分块 TRSM kernel 族）
src/cpotri/        cpotri_host.cpp  cpotri_kernel.cpp（零对角/三角逆/Hermitian 乘 kernel 族）
src/cpotrf_batched/  cpotrf_batched_host.cpp  cpotrf_batched_kernel.cpp
src/cpotrs_batched/  cpotrs_batched_host.cpp  cpotrs_batched_kernel.cpp
test/cpotrf/ … test/cpotrs_batched/  {op}_test.cpp + data/{gen_data.py,verify_result.py} + README.md
docs/zh/cpotrf.md … docs/api_list.md  接口文档与支持表
```

kernel 入口沿用仓内惯例：`extern void {op}_kernel_do(GM_ADDR …, GM_ADDR tiling_gm,
uint32_t numBlocks, void *stream)`，tiling 结构经 GM 传递、kernel 侧手动解码（同
`cgetri_batched_kernel.cpp` 模式）；所有 kernel 在 `aclsolverGetStream` 返回的**调用方
stream** 上顺序下发，kernel 间依赖由 stream 顺序保证，Host 不回读 info。

### Host 侧设计

**1）参数校验与 -i 语义。** 按公开参数顺序（不计 handle）返回首个非法参数；Host 可判定
的非法返回 `ACLSOLVER_STATUS_INVALID_VALUE`，并在 handle/stream 有效且 info 可写时，用
`aclrtMemcpyAsync`（4 字节 H2D，无需自制 kernel）在调用方 stream 排队写入 `-i`。序号表：

| 接口 | 参数序号（除 handle 外，1-based） |
|---|---|
| potrf / potri | uplo=1, n=2, A=3, lda=4, Workspace=5, Lwork=6, devInfo=7 |
| bufferSize | uplo=1, n=2, A=3, lda=4, Lwork=5 |
| potrs | uplo=1, n=2, nrhs=3, A=4, lda=5, B=6, ldb=7, devInfo=8 |
| potrfBatched | uplo=1, n=2, Aarray=3, lda=4, infoArray=5, batchSize=6 |
| potrsBatched | uplo=1, n=2, nrhs=3, Aarray=4, lda=5, Barray=6, ldb=7, info=8, batchSize=9 |

校验内容：handle 非空（`HANDLE_IS_NULLPTR`）；uplo ∈ {0,1}（`INVALID_ENUM` 归并
INVALID_VALUE 口径与仓内既有实践一致）；n/nrhs/batchSize 非负；lda/ldb ≥ max(1,n)；
非空问题的数据指针/指针数组基址非空；potrf/potri 的 `Lwork ≥ 查询值`。乘积与偏移先升
int64 校验溢出，再检查 `Lwork ≤ INT_MAX`。`potrsBatched` 在 n>0 且 batchSize>0 时要求
nrhs=1（nrhs=0 交叉口径见冻结项 F4）。

**2）空问题。** n=0（及 potrs 的 nrhs=0、batched 的 batchSize=0）：保留枚举/负数/ld 合法
性检查后 quick return SUCCESS；单矩阵与 potrsBatched 标量 info 异步写 0；potrfBatched 在
batchSize=0 时无可写元素不下发写入，n=0 且 batchSize>0 时逐元素写 0（memsetAsync）。

**3）workspace 规划。** 查询与执行共享同一 planner、返回保守公开上界，kernel 使用不超过
查询值（`potrf`：面板打包/控制区；`potri`：T 平面 + C 平面）：

```text
P   = align16(n)                       # 对齐到 16 复数元素（128B）
potrf:  Lwork = P·P + 1024             # n=4096 → 134.2 MiB；n=0 → 0
potri:  Lwork = 2·P·P + 1024           # T 平面 + C 平面 + 控制区
```

控制区（1024 元素）存放：info 镜像/中止标志（4B）、面板进度字、调试字段。`potrs` 与两个
批量接口无公开 workspace：小规模片上驻留，大规模 GM 原地分块（不建立整批 GM 转换副本）。

**4）分核与 tiling。** 从 `platform_ascendc.h` 平台信息取 AIV 核数（不硬编码）。单矩阵
POTRF：面板 kernel `blockDim=1`（对角块）或按面板行分核（TRSM），尾更新 kernel 按尾矩阵
tile 静态循环分核（`tile = core, core+G, …`，每 tile 唯一写者）。批量：静态循环
`matrix = core, core+G, …`；`G = min(batchSize, 核数)`。内部 dispatch key（不注册 ACLNN
模板）：`key = op(3bit) | upper(1bit) | path(3bit) | tail(1bit)`，path ∈ {EMPTY,
RESIDENT, STREAM, BATCH_RESIDENT, BATCH_STREAM}；同 shape/环境选路确定，不运行时随机搜索。

**5）tiling 结构（GM 传递）。** `{n, lda, ldb, nrhs, batchSize, uplo, b(块宽), r(RHS 块宽),
path, waveBase, waveCount, workspaceOffset}`；b/r 初值 b=128/r=32，n<256 降 b=64，最终以
编译资源报告与板卡测量定档（M4 收敛，见风险 2）。

### Kernel 侧设计

**公共复数层**（`chol_common.hpp`）：AoS（GM ABI）↔ SoA（实/虚平面）视图、四实乘、共轭、
列主序三角索引；`__gm__` 直接标量访问仅用于 tiny 路径，主路径全部 UB/GM 分块搬运
（`DataCopyPad` 处理非整块行/列与 ld padding）。

**POTRF 多 kernel 编排（V1 主路径）**：

```text
stream 序：[memsetAsync info=0] → 对每个面板 j（步长 b）:
  K1 diag:  单核，A11(b×b) 入 UB 逐主元分解；d≤0 → 写 info=j+t+1 + 置 GM 中止标志
  K2 trsm:  面板行片分核，L21 = A21·L11⁻ᴴ（复用/扩展 utils c32 SolveTrsm，b ≤ 128 < maxN=256）
  K3 herk:  尾矩阵 tile 分核，A22 −= L21·L21ᴴ（c32 gemm/matmul API，FP32 累加）
K1/K2/K3 入口先读中止标志，已失败矩阵直接返回（部分因子保留，与 LAPACK 语义一致）
```

launch 数 = 3·ceil(n/b)：n=4096、b=128 时 96 次（拟定以 msprof 实测确认 launch 占比；
若超标走 M4 融合方案，见风险 2）。UPPER 与 LOWER 同一模板机（kernel 模板参数 `uplo`），
仅索引代数与共轭方向不同：UPPER 面板读上三角**列段**（列主序下连续），无需物化转置——
下标代数：LOWER 面板列 j 元素 `A[j+i, j]`（i≥0，段内 stride 1）；UPPER 面板行 j 的对偶
元素 `A[j, j+i]`（stride = lda，按列段装载后转置视图入 UB）。

**POTRS**：

| 路径 | 条件（拟定） | 结构 |
|---|---|---|
| 驻留列扫描 | nrhs=1，n ≤ 4096 | B 整列常驻 UB；逐主元 `x[i] = (b[i] − Σ L[i,k]·x[k])/L[i,i]`；**列外层/行内层、每语句单项乘减**（复用 PR #1973 对 AIV `-O3` 乘减链重结合的实测结论：≥2 项链必被重结合，pragma/临时变量均免疫），保证求和次序确定 |
| 分块前代/回代 | nrhs>1 或大 n | 对角块多 RHS TRSM（UB 内）+ 尾部 GEMM 更新（Cube）；第二遍共轭回代同构；UPPER 按 §数学公式 双三角次序执行 |
| UPPER 对偶 | — | 前代用共轭点积、回代用列更新（或反之），与 LOWER 共享模板 |

A 全程只读；B 仅写有效区（n×nrhs），ldb padding 不触碰。

**POTRI**（workspace 两平面）：
`[查零对角 kernel] → [分块三角逆：对角块 UB 求逆 + 非对角块 GEMM×2 → T 平面] →
[C = TᴴT Hermitian 乘 → C 平面，仅算/写指定三角] → [拷回 A 指定三角 + 对角虚部 +0.0]`。
零对角检查与三角逆可按块融合（逐对角块先查后逆），失败即置标志停止后续写回。

**Batched**：

- kernel 入口从 Device `uint64` 指针数组逐矩阵取址（**真指针数组语义**，禁以连续布局替代）；
  Host 只校验数组基址，不解引用、不搬回。
- **n ≤ 64**：整矩阵（≤32 KiB）驻留 UB，一矩阵一核静态循环；UB 内逐主元分解（同
  POTRF 面板算法的无分块版），逐矩阵独立 infoArray 槽位，失败只中止该矩阵。
- **65 ≤ n ≤ 128**：一矩阵一核，面板流式（GM RMW，面板入 UB），避免逐矩阵多 kernel。
- **n > 128**：受限波次调度——每矩阵由 `cores_per_matrix`（1…4，按 batch 规模与核数
  定档）核协同，核内用 `SetSyncBaseAddr` 同步基建（仓内既有先例，见背景表 #7）做
  面板栅栏；波次窗口矩阵数 ≤ 核数/cores_per_matrix，顺序覆盖全批。V1 保守取
  cores_per_matrix=1（单核分块，正确性优先），波内并行作为 M4 提升项（风险 3）。
- **potrsBatched**（nrhs=1）：一矩阵一核，驻留/流式两档复用 POTRS 单 RHS 路径；标量
  info 仅由 Host 侧排队写入（参数错/清零），各 batch 不竞争写。
- batchSize 至 10⁶：不逐矩阵 launch（上文单 kernel 循环覆盖），不分配 batch×n² 中间
  副本；大 batch（n=224 级）按波次处理，全部 kernel 时间计入该次 API（性能口径见 §可维可测）。

**访存与同步纪律**：单事件严格 Set→Wait 交替；`cnt=0` 面板照常预取、槽有效位跟踪；
`blockLen` 32B 对齐（borrow PR #1973 实测：非对齐 blockLen 有毫秒级病理、srcGap 任意
字节可接受——该结论出自 910B4，950PR 上复测定档）；各核写区间不重叠，info 独立 4B 写。

**确定性与精度结构**：每输出元素唯一写者；GEMM 归约 k 序固定；无跨核浮点 atomic；
同输入同流重复运行逐位一致（R9）。Cube FP32 路径显式关闭 HF32/隐式降精度模式
（matmul 初始化参数，拟定项 M2 验证）；若精度门禁不过，回退 AIV `Mul+ReduceSum`
实虚四乘展开路径（性能次优、精度可控）。对角输出虚部统一 `+0.0`。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Atlas 950（`ascend950*`，dav-3510，构建走 `arch35` 目录） | √ |
| Atlas A2/A3、910B 系列、310P | 本任务不声明支持（dav-2201/2002 另任务） |

配套：CANN ≥ 9.0.0（仓 README 已验版本为 9.0.0；开发自验环境拟用 9.1.0）。

## 算子约束限制

1. 仅 COMPLEX64；无选主元；仅列主序。
2. POTRF 输入须 Hermitian 正定（仅指定三角被读）；数值失败经 info 报告，非 Host 返回码。
3. POTRS/POTRI 输入须为同 uplo 的有效因子（不重复做正定性检查）。
4. 批量求解非空问题仅 nrhs=1；批量矩阵可分散分配但同一调用内输出不得互相重叠。
5. `Lwork` 单位为复数元素；workspace 生命周期须覆盖 stream 完成。
6. A/B/Workspace/info 须为调用方管理的合法 Device 内存；Host 仅校验可见标量与外层指针。
7. 公开维数 int32；内部 int64 寻址；输入总体受 Device 内存约束（不静默截断 batch）。

---

# 可维可测分析（required）

## 精度标准/性能标准

**精度（任务书 §3.2 + 随包脚本双口径，F1 冻结项）**：complex128 CPU golden（NumPy/SciPy，
固定版本与线程配置），实/虚部分别判：

- 元素级：`|actual − golden| ≤ atol + rtol·|golden|`；任务书 rtol=2⁻¹⁰、atol=2⁻¹⁶；
  随包 `verify_accuracy.py` 为 rtol=atol=2⁻¹³、ε=2⁻²⁴——两套同时跑、分别报告，验收口径
  由维护者冻结；
- matched_ratio ≥ 0.99 且 max_abs_error ≤ max(1e-2, 32×ULP)（ULP 对象/聚合方式为冻结项 F1）；
- LAPACK 残差（ε=2⁻²³，升 complex128 计算，1-范数；UPPER 用 `UᴴU` 不用 `FFᴴ`，POTRI
  用原矩阵 `A₀` 不用因子——两处任务书文字歧义按数学正确口径，随包脚本已体现）：

| 接口 | 残差 | 阈值 |
|---|---|---|
| potrf / potrfBatched | `‖recon−A₀‖₁/(n·‖A₀‖₁·ε)`，recon=LLᴴ 或 UᴴU | ≤ max(5×ratio_cpu, 3×ratio_cpu_mean) |
| potrs / potrsBatched | `max_j ‖A₀·X_j−B₀_j‖₁/(‖A₀‖₁·‖X_j‖₁·ε)` | 同上 |
| potri | `‖I−A₀·C‖₁/(n·‖A₀‖₁·‖C‖₁·ε)` | ≤ max(5×ratio_cpu, 0.1) |

仅比较指定三角（potrf/potri）或 B 有效区（potrs），批量逐矩阵独立判定。

**性能（任务书 §3.3）**：有 GPU 基线用例逐条 `T_NPU ≤ T_GPU/0.35`（不可用全局平均替代
逐条门禁）。msprof `Task Duration` 口径；一次 API 多 kernel 时先按调用汇总全部 kernel
再跨调用平均；POTRS/POTRI 前置 POTRF、数据生成/拷贝/bufferSize 查询不计入；每 case 预热
+ ≥25 次采样，同时报 mean 与 median（口径冻结项 F3）。任务书 12 个锚点 case（GPU ms →
NPU 上限 ms = GPU/0.35）：

| # | 接口/规格 | GPU ms | NPU 上限 ms |
|---|---|---:|---:|
| P-01 | potrf n=1024 LOWER | 0.79 | 2.257 |
| P-02 | potrf n=4096 UPPER | 6.7918 | 19.405 |
| P-03 | potrf n=2048 LOWER | 1.5434 | 4.410 |
| P-04 | potrs n=1024 nrhs=1 UPPER | 0.2235 | 0.639 |
| P-05 | potrs n=4096 nrhs=32 LOWER | 2.9488 | 8.425 |
| P-06 | potrs n=2048 nrhs=8 LOWER | 1.0867 | 3.105 |
| P-07 | potri n=1024 LOWER | 1.4534 | 4.153 |
| P-08 | potri n=4096 UPPER | 17.8843 | 51.098 |
| P-09 | potrfBatched n=32 batch=102774 LOWER | 2.9648 | 8.471 |
| P-10 | potrfBatched n=128 batch=46256 LOWER | 18.1601 | 51.886 |
| P-11 | potrsBatched n=32 batch=99659 nrhs=1 LOWER | 1.5953 | 4.558 |
| P-12 | potrsBatched n=128 batch=45897 nrhs=1 LOWER | 6.0311 | 17.232 |

## 测试策略

1. **官方 cases 驱动 GTest**：任务包 cases 共 768 条（cpotrf 143 / cpotrs 174 / cpotri
   143 / cpotrfBatched 154 / cpotrsBatched 154；batch 原始索引含 3 组重复配置，按
   canonical 去重关联，有效 GPU 性能基线 762 条、2 条 std case 无基线记缺失不填假值）；
   测试程序直调公开 API，数据生成/golden 在 CPU、DUT 在 NPU，`test/{op}/data/` 脚本
   随仓上库。
2. **补充回归矩阵**（特性交叉）：

| 类别 | 覆盖 | 判定 |
|---|---|---|
| 边界 | n=0/1/2ⁿ/2ⁿ±1/b±1/4096；nrhs=0/1/128；batch=0/1/1024/3000/10⁶（小 n） | 数值+info |
| 存储 | LOWER/UPPER×lda/ldb=n+8/n+32（padding 哨兵）；指针数组乱序分散分配 | padding/只读区不变 |
| info | 首/中/末失败主元、POTRI 零对角、混合成败 batch、逐参数 -i 序号 | info 与 status 一致 |
| 负向 | 非法 uplo/负维数/非法 ld/空指针/Lwork 不足/批量 nrhs>1 | 无越界下发 |
| 空问题 | n=0、nrhs=0、batchSize=0 交叉 | quick return + info |
| 确定性 | 同输入同流 ≥10 次 | 有效区与 info 逐位一致 |
| 并发/stream | 非默认流、双 handle 独立流、连续变 shape | 无状态串扰 |
| ABI | 纯 C 编译器包含头文件/链接、导出符号、`sizeof/offset` 断言 | C/C++ 双通 |

3. **探针程序族**：UPPER 共轭索引对拍、AoS↔SoA 转换对拍、面板中止标志传播、blockLen
   对齐病理复测（950PR 上重验 PR #1973 的 DMA 结论）、launch 开销占比隔离——全部可复现，
   随测试仓交付。

## 兼容性分析

- 新增接口全部在 `extern "C"` 区域；`aclsolverFillMode_t` 迁移至公共 C 头（取值 0/1
  不变，C++ 区域删除重复定义，既有 C++ 调用方源码兼容）；既有 7 算子签名/符号零改动。
- 新旧契约并行：旧接口 `std::complex`+`aclError` 不受影响；`aclFloatComplex` 与
  `std::complex<float>` 布局一致但类型独立，不隐式互转。
- 构建零侵入：源文件经 `src/CMakeLists.txt` glob 自动纳入（已核对收集逻辑）；如引入
  dav-3510 专属实现置于 `{op}/arch35/`，仅 `ascend950*` 编译。
- CANN ≥ 9.0.0；仓 README 支持表、`docs/api_list.md`、`docs/zh/{op}.md` 同步更新；
  回归既有 cgetrf/cgetri/cheevj 测试防公共组件改动外溢。

---

# 风险点

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| 1 | Cube FP32 累加精度（HF32/隐式降精度模式）相对 complex128 golden 超差 | 精度门禁失败 | 显式关闭降精度模式；AIV 四实乘+归约备选路径；残差判据按 CPU 因子同级标定 |
| 2 | POTRF 多 kernel launch 开销（n=4096 约 96 次 launch，拟定） | 大 n 性能余量收窄 | b=128 起步；msprof 隔离 launch 占比；M4 面板融合（diag+trsm 单 kernel）与持久 kernel 方案备选 |
| 3 | 大 n 批量单核分块性能不足（V1 保守 cores_per_matrix=1） | P-10 级 case 边缘 | M4 波内多核协同（复用 SetSyncBaseAddr 基建）；正确性回归全量覆盖后再启用 |
| 4 | 双精度容差口径（任务书 2⁻¹⁰/2⁻¹⁶ vs 随包 2⁻¹³/2⁻¹³）与 ULP 聚合方式 | 验收判读分歧 | 双轨报告；冻结项清单随首个代码 PR 提交维护者裁定 |
| 5 | 任务书"输入 ≤4G"与 canonical 大 batch（n=224 级 ~15 GiB）冲突 | 测试环境 OOM | 128 GiB HBM 950PR 按原规格执行；差异记录不静默截断 |
| 6 | 批量 nrhs=0 成功 vs 非空 batched 要求 nrhs=1 的交叉口径 | 空问题行为分歧 | 按"非空运算才限制"实现，冻结项 F4 提交裁定 |
| 7 | UPPER 共轭/索引代数错误（复数 H 语义漏共轭得"看似合理"错果） | 精度静默失败 | 非对称实虚测试数据 + UPPER/LOWER 对偶互验探针 |
| 8 | 10⁶ batch 与大 n 组合的 int32 乘积溢出 | 越界/错址 | 内部全 int64；乘法前置溢出检查（沿用仓内 issue #139 修复范式） |
| 9 | Device 内层空指针/非法地址无法异步上报 | 错误契约争议 | Host 仅校验外层；内层作 device 诊断路径记录，冻结项随 F4 |

# 冻结项（提交维护者裁定）

| # | 内容 |
|---|---|
| F1 | 元素级容差采用任务书（2⁻¹⁰/2⁻¹⁶）还是随包脚本（2⁻¹³/2⁻¹³）；32×ULP 的 ULP 对象与聚合粒度 |
| F2 | POTRI 残差公式中 A 的语义（本设计按原矩阵 A₀） |
| F3 | 性能统计 mean vs median（补充材料要求 30 次 median） |
| F4 | 空问题与批量限制的交叉优先级（batched nrhs=0、Device 内层非法地址上报方式） |

# 实施计划与交付物

| 阶段 | 内容 | 退出条件 |
|---|---|---|
| M0 契约冻结 | 固定 ops-solver/CANN commit；ABI/校验/空问题/计时口径与 F1-F4 裁定 | 口径记录一致 |
| M1 基础层 | 公共类型、cholesky_host/chol_common、7 接口骨架与校验、C ABI 测试 | 负向/空问题全过 |
| M2 单矩阵 | POTRF→POTRS→POTRI 全路径（LOWER+UPPER） | 三接口官方精度用例过 |
| M3 批量 | 指针数组、失败隔离、10⁶ 小 n、大 n 波次 | 批量用例+确定性过 |
| M4 性能 | b/r 扫描、launch 融合、批量波内并行、HF32 决策 | 12 锚点 + 762 基线逐条 ≤ GPU/0.35 |
| M5 交付 | 文档/示例/报告/双 PR | 评审问题关闭 |

**交付物**：① 本设计文档（社区仓 PR，落位
`tasklist/09-{任务编号}-{算子名称}/docs/design.md`，标题带"【社区任务】9月社区任务-"前缀，
PR 仅含该文件，评论 `/compile` 触发构建）；② 代码 PR（ops-solver master：include/src/test/docs，
五接口不拆分验收）；③ 自测用例与脚本（官方 cases + 补充回归 + 探针族 + 复现 README）；
④ 自测报告（精度/性能/内存三报告 + 日志；内存无门禁但如实记录分配量与越界/泄漏检查）。

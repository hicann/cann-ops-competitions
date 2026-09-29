# aclsolverCpotrf / Cpotrs / Cpotri / CpotrfBatched / CpotrsBatched 算子设计文档

> 社区任务：Atlas 950 单精度复数 Cholesky 分解、求解和批量接口
> 适配硬件：Ascend 950PR（Atlas 950 / arch35 / DAV_3510）
> CANN 版本：CANN 9.0.0 及以上（与 ops-solver 仓 README 已验证配套版本一致）
> 对标接口：cuSolver DN `cusolverDnCpotrf / Cpotrs / Cpotri / CpotrfBatched / CpotrsBatched`
> 工程框架：ops-solver 开源仓 Host C API + AscendC/CATLASS Kernel 直调
> 交付约束：五个接口作为同一任务一并交付，同一 PR（或一组关联 PR）合入，不得拆分

---

## 一、需求背景

### 1.1 需求来源

通过**社区任务**完成开源仓算子贡献的需求。在昇腾 NPU（Ascend 950PR / arch35）上基于
AscendC/CATLASS 开发稠密 Hermitian 正定线性求解功能及配套批量接口，共 5 个 API：

| 序号 | NPU 交付接口 | 对标 CUDA 接口 | 功能 |
|------|--------------|----------------|------|
| 1 | `aclsolverCpotrf` / `aclsolverCpotrf_bufferSize` | `cusolverDnCpotrf` / `_bufferSize` | Hermitian 正定矩阵 Cholesky 分解 A = L·Lᴴ（或 A = Uᴴ·U） |
| 2 | `aclsolverCpotrs` | `cusolverDnCpotrs` | 基于 Cholesky 因子求解 A·X = B，原地覆盖 B |
| 3 | `aclsolverCpotri` / `aclsolverCpotri_bufferSize` | `cusolverDnCpotri` / `_bufferSize` | 基于 Cholesky 因子求逆 A⁻¹·A = I |
| 4 | `aclsolverCpotrfBatched` | `cusolverDnCpotrfBatched` | 批量 Cholesky 分解，逐矩阵写 `infoArray[i]` |
| 5 | `aclsolverCpotrsBatched` | `cusolverDnCpotrsBatched` | 批量三角求解，仅支持 nrhs = 1 |

- 目标硬件：Ascend 950PR（arch35 / DAV_3510）
- 代码仓：`ops-solver`（https://gitcode.com/cann/ops-solver ）
  - 公开头文件：`include/cann_ops_solver.h`、`include/cann_ops_solver_common.h`
  - 算子实现目录：`src/cpotrf/`、`src/cpotrs/`、`src/cpotri/`、`src/cpotrf_batched/`、`src/cpotrs_batched/`
  - 测试目录：`test/cpotrf/`、`test/cpotrs/`、`test/cpotri/`、`test/cpotrf_batched/`、`test/cpotrs_batched/`
- 语义参考：cuSolver DN legacy API（https://docs.nvidia.com/cuda/cusolver/index.html ）
- ops-solver 仓当前无对应复数 Cholesky 原型，本任务为**全新实现**；仓内已有
  `cmatinv_batched`（批量复数矩阵求逆，AIV 批处理范式）与 handle 管理接口
  （`aclsolverCreate/Destroy/SetStream/GetStream`）可直接复用。

### 1.2 背景介绍

#### 1.2.1 算子实现说明

本任务在 ops-solver 仓中新增复数 Cholesky 全家桶（分解 / 求解 / 求逆 / 批量），属
稠密线性求解（Solver DN）类算子。Cholesky 分解是 Hermitian 正定线性方程组求解与
矩阵求逆的基础分解，与同族算子的关系：

- `Cpotrf` = 对称正定三角分解（本任务核心，计算量 O(n³/3) 复数乘加）
- `Cpotrs` = 两次三角求解（前代 + 回代，O(n²·nrhs)）
- `Cpotri` = 三角阵求逆（TRTRI）+ 三角乘积（LAUUM），O(2n³/3)
- 批量接口 = 单矩阵核心在海量小矩阵上的调度复用

#### 1.2.2 标杆算子现状分析

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

与 cuSolver DN legacy API 保持一致：

| 项 | 取值 |
|----|------|
| 矩阵类型 | COMPLEX64（`aclFloatComplex`：两个连续 FLOAT32 real/imag，布局对齐 `cuComplex`） |
| 矩阵布局 | n×n 列主序（Column-Major），物理布局 lda×n，`lda >= max(1, n)`，禁止要求 lda==n |
| 三角引用 | 仅 `uplo` 指定三角被读写；对齐 cuSolver：未使用的另一半三角可作 workspace 被破坏，对角元保持实数语义 |
| info | INT32 Device 张量：`0` 成功；`-i` 第 i 个参数非法（不计 handle）；`k > 0` 第 k 阶顺序主子式不正定 |
| 维数 | `int`（32-bit），与 CUDA legacy API 一致 |
| 指针归属 | 矩阵 / info / workspace / 指针数组均为 Device 指针；Host 仅传标量维数、枚举与 handle |
| 批量数组 | `Aarray/Barray` 为 **Device 上的指针数组**（`aclFloatComplex *Xarray[]`），不是连续张量 |

##### 1.2.2.2 标杆算子实现描述

**（1）Cpotrf**：`uplo=LOWER` 时 A = L·Lᴴ，`uplo=UPPER` 时 A = Uᴴ·U；结果原地覆盖
uplo 侧三角。展开（LOWER，0-based）：

```
对每列 j = 0..n-1:
    r = A[j][j].real - Σ_{k<j} |L[j][k]|²          # 对角元实数
    若 r 非有限 或 r <= 0：info = j+1，分解止于第 k 行（前 k-1 列因子已算出）
    L[j][j] = sqrt(r)                               # 虚部置 0
    对 i = j+1..n-1:
        L[i][j] = (A[i][j] - Σ_{k<j} L[i][k]·conj(L[j][k])) / L[j][j]
```

复数运算核心：复数乘 `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`；共轭转置取 `conj`。

**（2）Cpotrs**：LOWER 时先前代 `L·Y = B` 再回代 `Lᴴ·X = Y`；UPPER 对称（先
`Uᴴ·Y = B` 再 `U·X = Y`）。X 原地覆盖 B，`uplo` 必须与分解时一致。info 仅报参数
错误（0 / -i）——正定性已在 potrf 阶段保证。

**（3）Cpotri**：两步，输入为 potrf 输出的三角因子，输出原地覆盖为 Hermitian 逆的
uplo 侧三角：
1. TRTRI：三角因子求逆 L → L⁻¹（因子奇异时 info = k）；
2. LAUUM：`A⁻¹ = L^{-H}·L⁻¹`，仅生成 uplo 侧三角，对角强制实数。

**（4）CpotrfBatched**：对 i = 0..batchSize-1 逐矩阵执行与 Cpotrf 相同的分解，
`infoArray[i] = k > 0` 表示第 i 个矩阵的 k 阶顺序主子式不正定，非正定矩阵止步、
其余矩阵照常分解。

**（5）CpotrsBatched**：对每矩阵 `A[i]·X[i] = B[i]`，X 原地覆盖 B；**仅支持
nrhs = 1**（`nrhs ≠ 1` 且 n>0 必须报错）；`info` 为标量，仅报非法参数（-i），
正定性由 potrfBatched 的 infoArray 反映（cuSOLVER 手册 Remark 2 同口径）。

**边界语义（对齐 cuSolver）：**
- 空问题 `n=0 / nrhs=0 / batchSize=0`：成功返回，不执行计算（devInfo 可写时写 0）
- 参数校验失败返回 `ACLSOLVER_STATUS_INVALID_VALUE` 等，并在 info 可写时写 `-i`
- 计算走调用方 stream，无必要的 Host 同步禁止

##### 1.2.2.3 标杆算子实现流程图

以 Cpotrf 为例（其余接口见第三章本设计流程图）：

```
┌──────────────────────────────────────────────────┐
│            cusolverDnCpotrf (入口)                │
└────────────────────┬─────────────────────────────┘
                     ▼
        ┌──────────────────────────┐
        │ 参数校验                  │──→ INVALID_VALUE / NOT_INITIALIZED
        │ handle/uplo/n/lda/ptr    │     （devInfo 可写时写 -i）
        └─────────┬────────────────┘
                  ▼
        ┌──────────────────────────┐
        │ n == 0 ?                 │──yes──→ devInfo=0，SUCCESS
        └─────────┬────────────────┘
                  ▼ no
        ┌──────────────────────────┐
        │ workspace 语义检查        │──→ Lwork 不足 → INVALID_VALUE
        └─────────┬────────────────┘
                  ▼
        ┌──────────────────────────┐
        │ 列主序分块 Cholesky       │
        │ 对角块分解→panel 求解→    │
        │ trailing 更新，逐步推进   │
        └─────────┬────────────────┘
                  ▼
        ┌──────────────────────────┐
        │ 非正定 → info=k 提前终止  │
        │ 成功   → info=0          │
        └─────────┬────────────────┘
                  ▼
             返回 SUCCESS
```

---

## 二、需求分析

### 2.1 外部组件依赖

| 依赖组件 | 版本/说明 | 用途 |
|----------|-----------|------|
| ACL (AscendCL) | CANN 9.0.0 | 设备内存、stream、kernel launch、H2D 异步写 info |
| AscendC kernel 运行时 | CANN 9.0.0 | `__gm__`/`__ubuf__`、DataCopy/DataCopyPad、GatherMask/Gather、Brcb、Mul/Add/Div、ReduceSum、CrossCoreSetFlag/WaitFlag |
| CATLASS | ops-solver 配套 | 大规模复数/实域 GEMM 模板（trailing 更新、LAUUM 主计算），Cube 路径 |
| tiling/platform | CANN 9.0.0 | `platform_ascendc.h` 获取 AIV/Cube 核数、UB 容量 |
| NumPy / SciPy (LAPACK) | 测试机自装 | COMPLEX128 golden（zpotrf/zpotrs/zpotri 语义）与残差判据计算 |
| msprof | CANN 9.0.0 | 性能采集（kernel 级 Task Duration） |

### 2.2 内部适配模块

| 内部模块 | 说明 |
|----------|------|
| `include/cann_ops_solver_common.h` | 补充 `aclFloatComplex`、`aclsolverFillMode_t`（本任务新增） |
| `include/cann_ops_solver.h` | 5 个计算接口 + 2 个 bufferSize 接口声明（本任务新增） |
| aclsolver handle 系列 | `aclsolverCreate/Destroy/SetStream/GetStream`（仓内已有，直接复用） |
| `src/utils/assert.h` 等 | `SOLVER_ECHECK` 参数校验宏、日志 |
| `test/` 测试工程 | 测试二进制框架、`gen_data.py` / `verify_*.py` 数据与判定脚本 |

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

`cann_ops_solver_common.h` 新增类型（与 `cuComplex` / `cublasFillMode_t` 布局与取值对齐）：

```c
typedef struct {
    float real;
    float imag;
} aclFloatComplex;

typedef enum {
    ACLSOLVER_FILL_MODE_LOWER = 0,  /* 下三角，对齐 CUBLAS_FILL_MODE_LOWER */
    ACLSOLVER_FILL_MODE_UPPER = 1   /* 上三角，对齐 CUBLAS_FILL_MODE_UPPER */
} aclsolverFillMode_t;
```

`cann_ops_solver.h` 新增 7 个接口（参数名、顺序、含义、Device/Host 归属与对标
CUDA 接口逐参数一致，不得增删调换）：

```c
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

统一返回 `aclsolverStatus_t`，状态码语义对齐 cuSolver。与现存 `aclError` 原型的
对照表随头文件与文档交付（`ACL_SUCCESS→ACLSOLVER_STATUS_SUCCESS`、
非法参数→`ACLSOLVER_STATUS_INVALID_VALUE`、空句柄→
`ACLSOLVER_STATUS_NOT_INITIALIZED`、内部错误→`ACLSOLVER_STATUS_INTERNAL_ERROR`），
最终验收以 `aclsolverStatus_t` 公开接口为准。

**devInfo/info 写入契约（-i 参数编号不计 handle）：**

| 接口 | -i 参数编号（i=1 起） | 正定信息 |
|------|----------------------|----------|
| Cpotrf / CpotrfBatched | 1=uplo, 2=n, 3=A, 4=lda, 5=Workspace, 6=Lwork（Batched：1=uplo, 2=n, 3=Aarray, 4=lda, 5=batchSize） | `k>0`：第 k 阶顺序主子式不正定；Batched 逐矩阵写 `infoArray[i]=k` |
| Cpotrs | 1=uplo, 2=n, 3=nrhs, 4=A, 5=lda, 6=B, 7=ldb | 仅 0 / -i |
| Cpotri | 同 Cpotrf | `k>0`：因子第 k 阶顺序主子式为零 |
| CpotrsBatched | 1=uplo, 2=n, 3=nrhs, 4=Aarray, 5=lda, 6=Barray, 7=ldb, 8=batchSize | 标量，仅 0 / -i（nrhs≠1 报错） |

#### 2.3.2 Ascend C 算子相关约束

与标杆算子（cuSolver DN）相比，本设计的功能范围：

| 功能点 | cuSolver | 本设计 | 说明 |
|--------|----------|--------|------|
| 数据类型 | COMPLEX64 | COMPLEX64 | 一致 |
| 布局 | 列主序 + lda padding | 列主序 + lda padding | `lda >= max(1,n)`，不要求 lda==n |
| uplo | LOWER/UPPER | LOWER/UPPER | 一致；未存侧三角可破坏，只写 uplo 侧 |
| 对角元 | 实数语义 | 实数语义 | 分解/求逆对角虚部强制置 0 |
| `_bufferSize` | 有（potrf/potri） | 有 | Lwork 以复数元素为单位 |
| info 契约 | 0 / -i / k | 0 / -i / k | 真实写入 Device，k 为最小不正定阶数 |
| 批量指针数组 | Device 指针数组 | Device 指针数组 | `Aarray[i]` 各自独立 lda |
| potrsBatched nrhs | 仅 1 | 仅 1 | nrhs≠1 且 n>0 → INVALID_VALUE |
| 确定性 | 不承诺 | **bit-wise 一致** | 任务书要求：固定分核 + 固定累加顺序，无原子累加 |
| CPU fallback | 无 | 无 | 核心计算全部在 NPU AI Core 完成 |

规模与泛化（验收下限）：`Cpotrf/Cpotri` n∈[1,4096]；`Cpotrs` n∈[1,4096]、
nrhs∈[1,128]；批量接口 n∈[1,4096]、batchSize∈[1,1000000]，覆盖 LOWER/UPPER 与
lda padding。不支持 broadcast，不要求图融合；dynamic shape 由 Host 按本次调用的
n/nrhs/batchSize 现场生成 tiling。

---

## 三、需求详细设计

### 3.1 调用方式

采用 **ops-solver Host C API + AscendC/CATLASS Kernel 直调**（非 aclnn 两段式）：

```
用户应用 → aclsolverCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    → host：参数校验 → 空问题判定 → tiling 计算（按 n 现场生成）
    → 直调 kernel（<<<grid, nullptr, stream>>>，走 handle 绑定 stream）
    → 全程无 Host 同步；devInfo 由 kernel 写回
bufferSize 两段式：aclsolverCpotrf_bufferSize(host 纯计算，不碰设备)
    → 调用方分配 Workspace → 调用计算接口
```

源码目录（每接口独立目录，符合 ops-solver 规范）：

```
src/cpotrf/          cpotrf_host.cpp / cpotrf_tiling_data.h / cpotrf_kernel.cpp
src/cpotrs/          cpotrs_host.cpp / cpotrs_tiling_data.h / cpotrs_kernel.cpp
src/cpotri/          cpotri_host.cpp / cpotri_tiling_data.h / cpotri_kernel.cpp
src/cpotrf_batched/  cpotrf_batched_host.cpp / ..._tiling_data.h / ..._kernel.cpp
src/cpotrs_batched/  cpotrs_batched_host.cpp / ..._tiling_data.h / ..._kernel.cpp
```

### 3.2 需求总体设计

#### 3.2.0 总体架构与算法分解

五个接口复用四个核心子功能，避免重复实现（kernel 内以可复用函数/模板组织）：

```
┌────────────────────────────────────────────────────────────────────┐
│                        Host（每接口独立）                            │
│  参数校验 / 空问题 / tiling / 路径选择(AIV|CATLASS) / launch / info  │
└──────────┬──────────────────┬──────────────────┬───────────────────┘
           ▼                  ▼                  ▼
   ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐
   │ UPOTRF 子功能  │  │ UTRSM 子功能   │  │ UTRTRI/LAUUM 子功能│
   │ 无分块复数     │  │ 三角求解       │  │ 三角求逆/三角乘积   │
   │ Cholesky 列递推│  │ (前代/回代)    │  │                   │
   └───────┬───────┘  └───────┬───────┘  └─────────┬─────────┘
           │                  │                    │
   ════════╪══════════════════╪════════════════════╪════════ (复用边界)
           │                  │                    │
   Cpotrf/CpotrfBatched  Cpotrs/CpotrsBatched   Cpotri
   （+分块驱动：panel=UTRSM，trailing=复数 HERK 形态 GEMM）
```

**（1）复数实域展开**（AIV 向量指令无原生复数乘，统一拆实虚计算）：

```
复数乘： (aᵣ+aᵢi)(bᵣ+bᵢi) = (aᵣbᵣ - aᵢbᵢ) + (aᵣbᵢ + aᵢbᵣ)i
复数除： (aᵣ+aᵢi)/(cᵣ+cᵢi) = ((aᵣcᵣ+aᵢcᵢ) + (aᵢcᵣ-aᵣcᵢ)i) / (cᵣ²+cᵢ²)
GEMM：   C = A·B        → 4 次实 GEMM（或带缓冲 3 次）
HERK：   C = A·Aᴴ(三角) → 3 次实 SYRK 形态：
         Cᵣ = Aᵣ·Aᵣᵀ + Aᵢ·Aᵢᵀ，Cᵢ = Aᵢ·Aᵣᵀ - Aᵣ·Aᵢᵀ（对角虚部恒 0）
```

UB 内采用**实虚分离视图**（`GatherMask` 交错→实/虚两块连续 float，仓内
`cmatinv_batched` 已验证该范式），所有 Mul/Add/Div/ReduceSum 按 float 向量执行。

**（2）Cpotrf：分块右视（blocked right-looking）Cholesky**，块大小 nb：AIV 对角
块/panel 路径取 **64**（64×64 块的实/虚分离视图整块可驻留 AIV UB，单核递推
延迟最优）；CATLASS trailing 路径分块可取 128~256：

```
for k = 0, nb, 2nb, ... < n:                        # 列块推进
  ① 对角块 A11(k×k)      ：无分块 UPOTRF（AIV 列递推，正定性检测点）
  ② panel   A21(m×nb)    ：UTRSM  A21 ← A21·L11^{-H}（AIV，列连续访问）
  ③ trailing A22(m×m)    ：A22 -= L21·L21ᴴ（复数 HERK 形态）
     小/中 n：AIV 分块向量 GEMM；大 n：CATLASS GEMM 模板（Cube）
     ①②③ 之间经 stream 串行多 kernel 天然同步（详见 3.2.1.1）
```

- **正定性检测（info 契约实现）**：①中对角步 `r = a_jj - Σ|L[j][k]|²`，
  `r 非有限 或 r ≤ 0` 时置 `info = j+1` 并终止后续列块；前 k-1 列因子已算出、
  第 k 列起不写（对齐 cuSolver“分解止于第 k 行”）。批量接口逐矩阵独立检测，
  用向量比较（`r ≤ 0` mask）+ 归约求最小失败列号，失败矩阵以核内 per-matrix
  flag 屏蔽后续列更新。
- **对角实数语义**：`L[j][j] = (sqrt(r), 0)`，虚部显式写 0。
- **三角隔离**：所有 GM 载入按 uplo 侧掩码取数，永不读取未存储侧（避免被破坏区
  NaN 污染）；输出只写 uplo 侧。
- **UPPER 实现**：利用恒等式 **U = conj(Lᵀ)**（其中 L 为同一矩阵的下三角分解，
  chol_lower(A) 与 chol_upper(A) 满足 L = conj(U)ᵀ）——kernel 内只实现 LOWER
  列递推，UPPER 输入在**写回阶段**折算：`U(t,j) = conj(L(j,t))`，即读下三角平面
  `plane[t·ldL + j]` 并将虚部取负后交织写 GM 上三角。无需转置 kernel、无需
  FILL 双模板实例；正定性检测不受影响（Aᵀ 与 A 的顺序主子式相同）。

**（3）Cpotrs：两次分块三角求解**（LOWER：`Y = L⁻¹B` 前代 → `X = L^{-H}Y` 回代）。
列主序下前代按列推进：`x_j = (b_j - L[0:j,j]ᵀ·x[0:j]) / L[j][j]`，随后
`b[j+1:] -= L[j+1:,j]·x_j`；两次求解的访问均沿列连续，AIV 向量化。nrhs 按列分块，
多核按 (行块 × rhs 列块) 二维切分；块内子更新（nrhs×nb 小矩阵乘）大 nrhs 时走
CATLASS 小 GEMM。

**（4）Cpotri：TRTRI + LAUUM 分块递推**（LOWER）：

```
TRTRI（求 L⁻¹）：按对角块递推
    inv11 = L11⁻¹（对角块内 AIV 无分块三角求逆）
    inv21 = -L22⁻¹·(L21·inv11)   → TRSM + GEMM（大 n 走 CATLASS）
LAUUM（A⁻¹ = L^{-H}·L⁻¹，只生成下三角）：
    invA[i][j] = Σ_{k≥max(i,j)} conj(L⁻¹[k,i])·L⁻¹[k,j],  j ≤ i
    → 实域展开 3 次实 SYRK/GEMM（大 n 走 CATLASS），对角虚部置 0
```

**（5）批量接口**：复用单矩阵核心，按规模分层调度（见 3.2.1.1）；
`potrsBatched` 仅 nrhs=1，`info` 标量只由 host 参数校验写入。

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

950PR AIV 核数以 `PlatformAscendCManager` 运行时查询（仓内惯例上限 40）。

**（a）单矩阵 Cpotrf / Cpotrs / Cpotri（n∈[1,4096]）**：

| 规模 | 路径 | 分核方式 |
|------|------|----------|
| n ≤ 64 | AIV 单核 | 单核无分块，避免多核同步开销 |
| 64 < n < 1024 | AIV 多核 | 列块间串行推进；①对角块单核、②panel 与 ③trailing 按列/行块切分到 `useCoreNum` 核（trailing 行块切分各核独立写不相交子块，无需核间通信）；块间同步用 stream 串行多 kernel |
| n ≥ 1024 | AIV + CATLASS 混合 | ①②AIV，③trailing（及 Cpotri 的 TRTRI 面板/LAUUM、Cpotrs 大 nrhs 子更新）用 CATLASS GEMM 模板按 Cube 核网格切分；若现场 Cube 复数/FP32 模板形态受限，回退 AIV 分块 GEMM 路径（正确性一致，tilingKey 切换） |

- trailing 切分：`A22` 按行列二维网格切分为 `ceil(m/BM)×ceil(m/BN)` 子块，核
  `tid` 负责第 `tid % gridN` 列、第 `tid / gridN` 行的子块（BM=BN=128 档，实测调
  优）；累加沿 K 维按 `nb` 顺序固定展开，**无原子累加**，保证确定性。
- stream 串行多 kernel 替代核间 barrier：每列块步 ①②③ 各一次 launch
  （n=4096、nb=256 共 16 步 × 3 ≈ 48 次 launch，按仓内 launch 开销量级约
  0.5~1ms，含在性能预算内），换取无死锁风险与可维性；若实测 launch 占比超阈值，
  备选方案为单 kernel 内 `CrossCoreSetFlag/WaitFlag` 环形同步（设计保留接口，
  不引入首版）。

**（b）批量 CpotrfBatched / CpotrsBatched（batchSize∈[1,1000000]）**：

| 规模 | 分核方式 |
|------|----------|
| n ≤ 64（小矩阵海量 batch，如 P-09/P-11） | **多矩阵核内批处理**：每核装入 `batchNumPerRepeat`（64~128）个矩阵的实/虚分离视图，逐列步一次向量指令推进全部矩阵（`cmatinv_batched` 范式：`Gather` 批量收集对角标量 → `Brcb` 广播 → 全矩阵行/列向更新）；矩阵到核的映射为前大核均分 `matPerCore[i] = batchSize/cores (+1)` |
| 64 < n ≤ 256（如 P-10/P-12） | **一矩阵一核**：核循环取矩阵，核内无分块列递推；n=128 单核算力已足够（P-10 预算 51.9ms，均摊 1.1ms/矩阵） |
| n > 256（batch 较小，如 batch≤63@n=4096） | **一矩阵多核**：复用单矩阵分块路径，网格维一维留给 batch（`grid = batch × 核组`），核组内按单矩阵策略切分 |

- `Aarray/Barray` 为 Device 指针数组：kernel 从 GM 读 `Aarray[i]` 后按各自
  `lda` 寻址，天然支持不等价指针；`infoArray` 各核写不相交段，无写冲突。
- 确定性：批量分核映射固定（均分 + 前余数多 1），核内顺序固定 → 同输入
  bit-wise 一致。

**（c）devInfo/info 写入**：计算路径由 kernel 直接写 Device；参数错误路径由
host 校验后经 `aclrtMemcpyAsync`（H2D，同一 stream）写入 `-i`，不引入 Host 同步。
空问题（n=0/nrhs=0/batchSize=0）不 launch kernel，devInfo 可写时异步写 0。

##### 3.2.1.2 数据分块和内存优化策略

**UB 规划（每核，可用上限按仓内惯例 192KB，容量 248KB）**——以 Cpotrf 无分块
列递推为例：

| Buffer | 内容 | 大小 |
|--------|------|------|
| col_tile | 当前列块（实/虚分离视图） | 2 × tile_m × 4B |
| factor_tile | 已算因子列（供点积 Σ|L[j][k]|²） | 2 × tile_m × 4B |
| scalar_buf | 批量接口对角标量收集 + Brcb 广播区 | batchNumPerRepeat × 2 × 4B |
| red_buf | ReduceSum 工作区（256B 树归约对齐） | ≤ 1KB |

- 列方向按 `tile_m`（如 512，对齐 32B×8）分块载入，点积用 `Mul + ReduceSum`，
  列更新用 `Mul/Add/Div` 链；双 buffer（MTE2↔V）流水。
- 批量路径同 `cmatinv_batched`：`DataCopyPad` 整批搬入 → `GatherMask` 实虚分离
  → 逐列批处理 → `Gather` 交织写回，规避 `DataCopy` 32B 对齐限制。

**Workspace / Lwork 契约**（`_bufferSize` 以复数元素计数）：

```
Cpotrf/Cpotri _bufferSize：
    n ≤ 小规模阈值（AIV 路径自足）      → Lwork = 0
    大规模（CATLASS 路径）             → Lwork = panel 缓冲 nb×n
                                          + GEMM 工作区（固定常量，对齐 32B）
计算接口：Lwork ≥ bufferSize 返回值即可；Workspace 为空且 Lwork>0 → INVALID_VALUE(-i)
```

`Cpotrs` 及两批量接口无 workspace 参数（对齐 cuSolver），核间/步间中间量全部
原地 + 核内 UB，必要时使用库内固定工作区（不占调用方契约）。

**GM 访存**：列主序下列连续，panel/trailing 沿列载入天然 coalesce；trailing 子块
写回用 `DataCopyPad` 处理 `m % 对齐` 尾块；lda padding 行不参与任何读写。

##### 3.2.1.3 tilingKey 规划策略

每接口独立 kernel（独立 `src/<op>/` 目录），tilingKey 仅区分计算路径与编译期
分支实例，形态参数（uplo 等）随 tiling 数据下发：

| 接口 | tilingKey | 含义 |
|------|-----------|------|
| cpotrf | 0 / 1 / 2 | 0=AIV 单核（n≤64）；1=AIV 多核分块；2=CATLASS 大 n（ Cube） |
| cpotrs | 0 / 1 | 0=AIV（含中小 nrhs）；1=CATLASS（大 n × 大 nrhs） |
| cpotri | 0 / 1 | 0=AIV；1=CATLASS（TRTRI 面板 + LAUUM） |
| cpotrf_batched | 0 / 1 / 2 | 0=多矩阵批处理（n≤64）；1=一矩阵一核（64<n≤256）；2=一矩阵多核（n>256） |
| cpotrs_batched | 0 / 1 / 2 | 同上（nrhs=1 固定） |

uplo（LOWER/UPPER）以模板参数编译期实例化（2 实例 × 上述 key），消除内核逐元素
分支。tiling 数据结构示例（cpotrf）：

```cpp
struct CpotrfTilingData {
    int32_t  n;
    int32_t  lda;
    uint32_t uplo;                 // 0=LOWER 1=UPPER
    uint32_t nb;                   // 分块大小
    uint32_t numBlk;               // 列块数
    uint32_t useCoreNum;           // trailing/panel 使用核数
    uint32_t gridM, gridN;         // trailing 二维网格
    uint32_t rowStart[MAX_CORE];   // 各核行块起点
    uint32_t rowCnt[MAX_CORE];     // 各核行块数
    uint32_t path;                 // 0=AIV 1=CATLASS
};

struct CpotrfBatchedTilingData {
    int32_t  n;
    int32_t  lda;
    uint32_t uplo;
    uint32_t batchSize;
    uint32_t useCoreNum;
    uint32_t batchNumPerRepeat;    // 核内批处理矩阵数（小 n 档）
    uint32_t matOffset[MAX_CORE];  // 各核起始矩阵号
    uint32_t matCnt[MAX_CORE];     // 各核矩阵数
};
```

##### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

以 **cpotrf 多核分块路径**（tilingKey=1/2）为例，每列块步 k：

1. **对角块 UPOTRF（AIV 单核）**：`nb×nb` 载入 UB 实虚分离；列递推
   （对角 `Mul+ReduceSum` 得 `r` → 正定检测 `r ≤ 0/NaN → info=k+1 提前终止`
   → `sqrt` → 列内 `复数除` 更新）；结果写回 GM（对角虚部置 0）。
2. **panel UTRSM（AIV 多核）**：`A21` 按列块切分到各核，每列做右视前代
   （沿列连续点积 + 复数除），读 L11 已写回因子。
3. **trailing HERK 更新**：实域展开 `Cᵣ = Aᵣ·Aᵣᵀ + Aᵢ·Aᵢᵀ`、
   `Cᵢ = Aᵢ·Aᵣᵀ - Aᵣ·Aᵢᵀ`（三角形态 3 次实 GEMM）；
   - AIV 路径：每核按行列子块做 `载入 L21 tile → Mul/Add 累加 → DataCopyPad 写回`，
     K 维沿 nb 顺序固定展开；
   - CATLASS 路径：调用 matmul 模板（FP32 实域），固定 32 分块 → 确定性。

**批量小 n 路径**（cpotrf_batched tilingKey=0）：整批矩阵搬入 UB → 实虚分离 →
循环列 j：`Gather` 收集全体矩阵第 j 列对角更新量 → `Compare(r≤0)` + 归约定位失败
矩阵并记录 flag → `sqrt/Brcb` 广播 → 全矩阵列更新向量指令一次完成 → 交织写回。
失败矩阵以 flag 屏蔽后续列（其 `infoArray[i]` 已定，其余矩阵不受影响）。

**cpotrs**：前代/回代两个子过程，各按列推进；多核按 (行块 × rhs 列块) 切分，
每核独立更新自己负责的 B 子块；nrhs=1 退化为三角矩阵-向量求解（纯 AIV）。

**cpotri**：TRTRI 对角块递推 + 面板 `-L22⁻¹·(L21·inv11)`（TRSM+GEMM），LAUUM 三角
乘积（3 次实 SYRK 形态）；只写 uplo 侧、对角虚部置 0。

##### 3.2.2.2 Ascend C 实现流程图

```
        aclsolverCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
                                │
        ┌───────────────────────▼────────────────────────┐
        │ handle==nullptr ──► NOT_INITIALIZED            │
        │ uplo 枚举 / n<0 / lda<max(1,n) / 空指针         │──► INVALID_VALUE
        │ （devInfo 可写 → H2D 异步写 -i）                │    （写 -i 后返回）
        │ n==0 ──► devInfo=0，SUCCESS（不 launch）        │
        │ Lwork < bufferSize(n) ──► INVALID_VALUE        │
        └───────────────────────┬────────────────────────┘
                                ▼
                    host tiling：nb / 路径 / 分核
                                ▼
        ┌───────────────────────────────────────────────┐
        │ for k = 0; k < n; k += nb:  （每步 3 个 kernel，│
        │             stream 串行 = 天然栅栏）            │
        │  ① UPOTRF 对角块（1 核）                       │
        │     列 j∈[k,k+nb): r=Mul+ReduceSum-对角旧值     │
        │        r≤0 或 NaN → devInfo=j+1，break         │
        │        L[j][j]=(sqrt(r),0)；列内复数除更新      │
        │  ② UTRSM panel（多核按列切）                    │
        │  ③ trailing：A22 -= L21·L21ᴴ                   │
        │     tilingKey=1: AIV 子块 GEMM（无原子累加）    │
        │     tilingKey=2: CATLASS 3×实 GEMM（Cube）     │
        └───────────────────────┬───────────────────────┘
                                ▼
                 成功：kernel 写 devInfo=0 → SUCCESS
```

批量（cpotrf_batched，小 n 档）：

```
 kernel（多核，每核 matCnt 个矩阵 × batchNumPerRepeat 一组）
   ├─ DataCopyPad 整组搬入 → GatherMask 实虚分离
   ├─ for j = 0..n-1:
   │    Gather 对角列更新量（全矩阵批量）
   │    Compare(r≤0)/NaN → 定位失败矩阵 → 置 fail flag，infoArray=k
   │    sqrt + Brcb 广播 → 列更新（一次向量指令推进全组矩阵）
   ├─ 交织写回（fail 矩阵止步，其余照常）
   └─ 各核写自己的 infoArray 段
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| 差异点 | 标杆（cuSolver DN） | Ascend C 本设计 | 原因 |
|--------|---------------------|-----------------|------|
| 执行结构 | 单 API 内部黑盒调度（host loop + 多 kernel 流） | host 侧列块步循环，每步 ①②③ 独立 kernel，stream 串行同步 | 同步显式化，无核间 barrier 死锁风险，可维性好；launch 开销已纳入性能预算 |
| 计算单元 | GPU SM cuBLAS/cuSOLVER 内核 | AIV 向量（无分块/panel/中小规模）+ CATLASS Cube（trailing/LAUUM 大 n） | NPU 异构算力：O(n³) 主体必须用 Cube 达 0.35×GPU 门禁；向量核承担递推与正定检测 |
| 复数运算 | 原生复数路径 | 实/虚分离视图 + 实域展开（乘 4 项、除 2 式、HERK 3 次实 GEMM） | 向量指令无复数类型；分离视图与仓内 cmatinv_batched 范式一致 |
| 正定检测 | 黑盒 | ①对角步显式 `r≤0/NaN` 判断，批量路径向量比较 + 归约定位 | info 契约要求给出最小不正定阶数 k 并止步 |
| 确定性 | 不承诺 | 固定分核映射 + K 维固定顺序 + 无原子累加 | 任务书 bit-wise 一致要求 |
| workspace | 内部 opaque | Lwork 契约显式化（0 或 panel+GEMM 工作区） | 对齐 `_bufferSize` 接口语义，调用方按契约分配 |
| 对角处理 | 黑盒 | 对角元虚部显式置 0（分解/求逆） | 任务书对角实数语义验收 |

### 3.3 支持硬件

| 芯片版本 | 支持 |
|----------|:---:|
| Atlas A2/A3（ascend910b*） | ✗ |
| Ascend 950PR（Atlas 950 / arch35 / DAV_3510） | ✅ |

CANN 9.0.0 及以上，与 ops-solver 仓 README 已验证配套版本一致。

### 3.4 算子约束限制

- 数据类型仅 COMPLEX64（`aclFloatComplex`）；维数 int32；列主序 + lda padding
  （`lda >= max(1, n)`）；矩阵/info/workspace/指针数组均为 Device 指针。
- 仅读写 `uplo` 指定三角；另一侧视为可破坏 workspace，不读取；对角元实数语义。
- 批量接口 `Aarray/Barray` 为 Device 指针数组，逐矩阵独立 lda；`potrsBatched`
  仅 nrhs=1。
- 规模下限：单矩阵 n∈[1,4096]；Cpotrs nrhs∈[1,128]；批量 batchSize∈[1,1000000]；
  空问题成功返回；非法类型/shape/枚举/值域返回 `INVALID_VALUE` 并写 `-i`。
- 不支持 broadcast、图融合；确定性 bit-wise 一致（合法用例重复执行）。

---

## 四、特性交叉分析

| 特性 | 分析 |
|------|------|
| 多核并行 | 单矩阵：列块步内 panel/trailing 多核切分，步间 stream 栅栏；批量：小 n 多矩阵核内批处理、中 n 一矩阵一核、大 n 一矩阵多核，核间无共享写 |
| 确定性 | 固定核映射 + 固定累加顺序（K 维沿 nb 顺序）+ 无原子累加/无 reduce 不确定序，满足 bit-wise 一致 |
| workspace | Lwork 契约显式（大规模 = panel 缓冲 + GEMM 工作区，小规模 = 0）；计算全程无 Host 同步 |
| 精度 | 复数乘加按 FP32 实域展开，与 LAPACK 同精度量级；golden 用 COMPLEX128；批量 A0 抽样 + 余槽一致性判定适配 |
| 三角性/Hermitian | 全链路三角掩码访存，未存侧零读取；输出对角虚部置 0；UPPER 经 U = conj(Lᵀ) 在写回折算，无转置 kernel |
| info 契约 | -i（host 校验，H2D 异步写）/ k（kernel 正定检测，最小阶数）/ 0；批量逐矩阵独立、互不干扰 |
| 并发安全 | kernel 只读写调用方传入的 Device 缓冲，无全局状态，可多 stream 并发 |
| 复用性 | UPOTRF/UTRSM/UTRTRI/LAUUM 四个子功能被 5 接口复用；批量调度框架与 cmatinv_batched 范式同构，便于后续 Z 版本与 getrf/getri 系列扩展 |

---

## 五、可维可测分析

### 5.1 精度标准 / 性能标准

**精度标准（对齐任务书 §3.2 与生态算子开源精度标准）：**

| 数据类型 | FLOAT32（实部 / 虚部分别判定） |
|----------|-------------------------------|
| rtol | 2⁻¹⁰ (9.77e-4) |
| atol | 2⁻¹⁶ (1.53e-5) |
| required_matched_ratio | 0.99 |
| max_abs_error_limit | 1e-2 或 32 × ULP |

- 逐元素 `|actual - golden| ≤ atol + rtol × |golden|`；golden 用 COMPLEX128
  （NumPy/SciPy `cholesky`/`cho_solve`/`inv`，即 zpotrf/zpotrs/zpotri 语义）。
- 逐元素不通过时按 LAPACK 残差判据复核（任务书 §3.2.2）：
  - potrf：`ratio = ‖F·Fᴴ − A‖₁ / (n·‖A‖₁·ε)`，`ratio ≤ max(5·ratio_cpu, 3·ratio_cpu_mean)`；
  - potrs：`ratio = max_j ‖B_j − A·X_j‖₁ / (‖A‖₁·‖X_j‖₁·ε)`，同上阈值；
  - potri：`ratio = ‖I − A·C‖₁ / (n·‖A‖₁·‖C‖₁·ε)`，`ratio ≤ max(5·ratio_cpu, 0.1)`；
  - 批量逐矩阵判定，ratio_cpu_mean 取 case 内 batch 均值；ε = 2⁻²³。
- info 契约用例：非正定（故意构造）→ `info = k`（最小不正定顺序主子式阶数）；
  非法参数 → `-i`；成功 → 0；确定性：合法用例重复执行 bit-wise 一致。
- INF/NAN 输入按精度标准文档对应规则验收。

**性能标准（对齐任务书 §3.3，950PR 实测，msprof 采 kernel 平均单次耗时，
T_NPU ≤ T_GPU数据 / 0.35）：**

| 编号 | 接口 | 规格 | GPU 性能 | NPU 达标上限 |
|------|------|------|----------|--------------|
| P-01 | Cpotrf | n=1024, LOWER, lda=n | 0.79 ms | ≤ 2.257 ms |
| P-02 | Cpotrf | n=4096, UPPER, lda=n | 6.7918 ms | ≤ 19.405 ms |
| P-03 | Cpotrf | n=2048, LOWER, lda=n | 1.5434 ms | ≤ 4.410 ms |
| P-04 | Cpotrs | n=1024, nrhs=1, UPPER | 0.2235 ms | ≤ 0.639 ms |
| P-05 | Cpotrs | n=4096, nrhs=32, LOWER | 2.9488 ms | ≤ 8.425 ms |
| P-06 | Cpotrs | n=2048, nrhs=8, LOWER | 1.0867 ms | ≤ 3.105 ms |
| P-07 | Cpotri | n=1024, LOWER | 1.4534 ms | ≤ 4.153 ms |
| P-08 | Cpotri | n=4096, UPPER | 17.8843 ms | ≤ 51.098 ms |
| P-09 | CpotrfBatched | n=32, batch=102774, LOWER | 2.9648 ms | ≤ 8.471 ms |
| P-10 | CpotrfBatched | n=128, batch=46256, LOWER | 18.1601 ms | ≤ 51.886 ms |
| P-11 | CpotrsBatched | n=32, batch=99659, nrhs=1, LOWER | 1.5953 ms | ≤ 4.558 ms |
| P-12 | CpotrsBatched | n=128, batch=45897, nrhs=1, LOWER | 6.0311 ms | ≤ 17.232 ms |

**性能设计核算**（支撑双路径决策）：P-02（n=4096 potrf）主计算 ≈ (8/3)n³ ≈
1.8e11 FP32 flops，19.4ms 预算需 ~9.5 TFLOPS，必须由 Cube（CATLASS）承担 trailing
主更新；P-09/P-11（batch≈1e5, n=32）主算力需求 ~1 TFLOPS、访存 ~0.9GB/8.5ms，
AIV 多矩阵批处理即可达标；P-10（batch=46256, n=128）约 5 TFLOPS，一矩阵一核
AIV 逼近上限，实测不足时切换 n>128 档到一矩阵多核/CATLASS 路径（tilingKey 预留）。

**自测**：使用包内 `gen_data.py` + `verify_accuracy.py` / `verify_perf.py`
（143~174 case/接口，覆盖 n 2 的幂与 ±1、nrhs 1/8/16/128 与边界 0、batch
1/8/128/1024/3000/1e6 档、uplo 双侧、lda +8/+32 padding、info 契约、批量 A0
抽样）；配合 ops-solver 测试工程（`bash build.sh --soc=ascend950 --ops=<op>
--run`）执行，msprof 采集性能，输出自测报告。

### 5.2 兼容性分析

| 兼容性项 | 说明 |
|----------|------|
| 接口兼容 | 5 个接口声明入公共头 `include/cann_ops_solver.h`，类型入 `cann_ops_solver_common.h`，供其他产品线共用，禁止 950PR 私有平行 API |
| 返回值迁移 | 现存 `aclError` 原型 → `aclsolverStatus_t` 对照表随头文件交付，验收以新状态码为准 |
| 数据格式 | `aclFloatComplex` 布局与 `cuComplex` 一致（两个连续 FLOAT32），与仓内既有复数算子（cmatinv_batched 等）同构 |
| 多代际 | arch35 专属实现位于 `src/<op>/`，不影响其他架构路径 |
| 语义对齐 | 空问题 / lda padding / info 契约 / potrsBatched nrhs=1 / 未存侧可破坏等边界语义与 cuSolver 对齐 |
| 文档配套 | `docs/zh/cpotrf.md` 等五篇算子文档 + `docs/api_list.md` + README 产品支持表同步更新（Ascend 950PR：支持） |

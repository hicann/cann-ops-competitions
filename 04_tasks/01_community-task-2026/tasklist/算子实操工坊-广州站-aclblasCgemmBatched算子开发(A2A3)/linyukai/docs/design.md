# 需求背景（required）

## 需求来源

CANN 社区任务：算子实操工坊-广州站 aclblasCgemmBatched 算子开发(A2/A3)。

- 任务书：`aclblasCgemmBatched_Atlas800IA3_task_doc.md`
- 设计文档以 PR 形式提交到 `cann-ops-competitions` 仓 `04_tasks/01_community-task-2026/tasklist`，PR 标题要求：【CANN社区任务】aclblasCgemmBatched算子设计文档
- 代码合入仓库：https://gitcode.com/cann/ops-blas
  - 算子目录：`blas/gemm_batched/arch22/`
  - 测试目录：`test/gemm_batched/cgemm_batched/arch22/`
- 接口对标 cuBLAS `cublasCgemmBatched`，单批数值语义以 Netlib BLAS `cgemm`（https://www.netlib.org/blas/cgemm.f）为准
- `aclblasCgemmBatched` 的声明已在 `include/cann_ops_blas.h` 中（当前仓版本第 437–442 行，任务书标注为 433–438 行，行号随仓版本不同）。该声明与 Ascend 950PR/950DT 的同名接口共用，本任务只补 arch22 实现，不另定义产品私有接口。

## 背景介绍

### aclblasCgemmBatched算子实现

在 ops-blas 仓的 Ascend C 框架下，用 kernel 直调方式在 Atlas A2/A3（架构目录 `arch22`，ASCEND_V220）上实现单精度复数（complex64）批量矩阵乘 `aclblasCgemmBatched`。

batch 中每个矩阵独立计算：

```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i],   i ∈ [0, batchCount-1]
```

`alpha`、`beta` 是单精度复数标量；A/B/C 通过设备侧指针数组逐批给出起始地址，矩阵全部为列主序（Column-Major）；`op(A[i])` 为 m×k、`op(B[i])` 为 k×n、`C[i]` 为 m×n。batch 是 uniform 的：所有批次共用 m/n/k、lda/ldb/ldc、transa/transb，批次之间只有地址不同。

和 strided batched 的区别在于：批次地址不满足 `base + i*stride` 的等差关系，而是放在设备侧指针数组里，所以地址解引用只能在 device 侧做（kernel 内读 `__gm__ uint64_t*` 指针数组）。

### 同类算子现状分析

仓内相关算子的现状：

| 算子 / 目录 | 数据类型 | 仓内现状 | A2/A3 |
| --- | --- | --- | --- |
| `aclblasSgemmBatched`（`blas/gemm_batched/arch35/`） | FLOAT32 | host + kernel + tiling 三件套 | 不支持 |
| `aclblasCgemmBatched`（`blas/gemm_batched/arch35/`） | COMPLEX64 | 与 S 版本同文件，走 4M 分解 | 不支持 |
| `aclblasChemm`（`blas/hemm/arch22/`） | COMPLEX64 | arch22 复数矩阵乘实现（host/kernel/tiling，非批量） | 支持 |
| `aclblasCsymm`（`blas/symm/arch22/`） | COMPLEX64 | arch22 实现（非批量） | 支持 |
| `blas/gemm/` | FLOAT32 等 | 仅 `arch35/` | 不支持 |

由此：

1. `blas/gemm_batched/` 下只有 `arch35/`，`arch22/` 为空；`test/gemm_batched/cgemm_batched/` 同样只有 `arch35/`，需要补 arch22。
2. arch35 已验证「反交错 → 4 次实数 GEMM → 合并」的 4M 路线，算法可以沿用，但 kernel 代码不能直接搬。arch35（DAV_3510）和 arch22（ASCEND_V220）硬件参数不同（`blas/common/arch/hardware.h`）：arch35 L1 32KB、L0C 256KB、Cube block 16；arch22 L1 512KB、L0C 128KB、Cube block 32。tiling 和流水要按 arch22 重做。
3. `blas/hemm/arch22/` 提供了 arch22 复数矩阵乘的工程参照：host 做校验、tiling、workspace 规划；device 侧一组 `extern "C" __global__ __aicore__` kernel（预处理 / Cube 主体 / 后处理）由 launcher 在同一 stream 上顺序发；workspace 按 `Ar | Ai | Br | Bi | RR | II | RI | IR` 的 4M 布局切分。本设计的工程结构与之保持一致。

### aclblasCgemmBatched算子功能分析

数学语义按 cuBLAS `cublasCgemmBatched` 文档和 Netlib `cgemm.f` 对齐。

`op(X)` 由转置标志决定：

| 标志 | 含义 | 数学形式 |
| --- | --- | --- |
| `ACLBLAS_OP_N` | 不转置 | `op(X) = X` |
| `ACLBLAS_OP_T` | 转置 | `op(X) = X^T` |
| `ACLBLAS_OP_C` | 共轭转置 | `op(X) = X^H = conjg(X)^T` |

物理维度（列主序下矩阵行数即前导维下界）：

| 矩阵 | transa/transb = N | transa/transb = T 或 C | 前导维约束 |
| --- | --- | --- | --- |
| A[i] | lda × k | lda × m | N: lda ≥ max(1,m)；T/C: lda ≥ max(1,k) |
| B[i] | ldb × n | ldb × k | N: ldb ≥ max(1,k)；T/C: ldb ≥ max(1,n) |
| C[i] | ldc × n（逻辑 m×n） | 同左（C 不受 trans 影响） | ldc ≥ max(1,m) |

参数说明（对应任务书 §2.4）：

| 参数名 | 输入/输出 | 描述 | 数据类型 | 内存位置 | 异常行为 |
| --- | --- | --- | --- | --- | --- |
| handle | 输入 | 库上下文句柄，携带 stream | 句柄 | Host | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| transa | 输入 | A[i] 的操作类型 | 枚举 int | Host | 非 {N,T,C} → `ACLBLAS_STATUS_INVALID_VALUE` |
| transb | 输入 | B[i] 的操作类型 | 枚举 int | Host | 非 {N,T,C} → `ACLBLAS_STATUS_INVALID_VALUE` |
| m | 输入 | op(A[i]) 与 C[i] 的行数 | int | Host | m<0 → `INVALID_VALUE`；m=0 合法 no-op |
| n | 输入 | op(B[i]) 与 C[i] 的列数 | int | Host | n<0 → `INVALID_VALUE`；n=0 合法 no-op |
| k | 输入 | op(A[i]) 的列数、op(B[i]) 的行数 | int | Host | k<0 → `INVALID_VALUE`；k=0 跳过矩阵乘 |
| alpha | 输入 | 复数乘数指针 | COMPLEX64 | Host | nullptr → `INVALID_VALUE` |
| Aarray | 输入 | 设备侧指针数组，各元素指向 A[i]，只读，列主序 | COMPLEX64 | Device | batchCount>0 且 nullptr → `INVALID_VALUE` |
| lda | 输入 | A[i] 前导维（列主序） | int | Host | 不满足约束 → `INVALID_VALUE` |
| Barray | 输入 | 设备侧指针数组，各元素指向 B[i]，只读，列主序 | COMPLEX64 | Device | batchCount>0 且 nullptr → `INVALID_VALUE` |
| ldb | 输入 | B[i] 前导维 | int | Host | 不满足约束 → `INVALID_VALUE` |
| beta | 输入 | 复数乘数指针；beta=(0,0) 时 C[i] 无需有效初值 | COMPLEX64 | Host | nullptr → `INVALID_VALUE` |
| Carray | 输出 | 设备侧指针数组，各元素指向 C[i]，原地覆写，列主序；各 C[i] 不得重叠 | COMPLEX64 | Device | batchCount>0 且 nullptr → `INVALID_VALUE` |
| ldc | 输入 | C[i] 前导维 | int | Host | ldc < max(1,m) → `INVALID_VALUE` |
| batchCount | 输入 | 指针数组长度 | int | Host | batchCount<0 → `INVALID_VALUE`；batchCount=0 合法 no-op |

返回值为 `aclblasStatus_t`，语义以 `include/cann_ops_blas_common.h` 为准。

边界行为（cuBLAS 文档与 Netlib `cgemm` 的 quick return 语义）：

| 条件 | 行为 | golden 路径 |
| --- | --- | --- |
| `m = 0` 或 `n = 0` 或 `batchCount = 0` | no-op，直接返回 `SUCCESS`，不启动 kernel | 直接返回 |
| `k = 0` 或 `alpha = (0,0)` | 跳过矩阵乘，逐批 `C[i] = beta * C[i]` | `ApplyComplexAlphaZeroGolden`（位精确 EXACT） |
| 其中 `beta = (0,0)` | C[i] 置零，调用前无需初始化 C | — |
| 其中 `beta = (1,0)` | C[i] 不变，直接返回 `SUCCESS`，不启动 kernel | — |
| 各 `C[i]` 之间重叠 | 行为未定义（接口要求不得重叠） | — |

# 需求分析（required）

## 需求描述

在 ops-blas 框架下为 Atlas A2/A3（arch22）补齐 `aclblasCgemmBatched` 的 Ascend C 实现。host 侧负责参数校验、tiling、workspace 管理；device 侧按「反交错（顺带完成转置/共轭折算）→ 4 次实数矩阵乘（Cube 或 AIV 标量兜底）→ 合并」的 4M 分解计算。精度满足生态算子开源精度标准，性能不低于任务书 §3.3 的标杆耗时，零维/空指针/非法参数等负向场景返回与 arch35 一致的状态码。同步交付 CSV 驱动的测试代码、测试 README 和自测报告。

## 需求拆解

1. 接口复用 `include/cann_ops_blas.h` 中已有声明（与 950PR/950DT 共用），参数序列与 `cublasCgemmBatched` 一致，不新增、不改声明。
2. 实现放在 `blas/gemm_batched/arch22/`，与 `arch35/` 并列；`blas/gemm_batched/README.md` 产品支持表把 cgemm 的 Atlas A2/A3 改为「支持」，sgemm 维持原状（本次不实现）。
3. host 侧（`gemm_batched_host.cpp`）：参数校验（顺序、状态码与 arch35 对齐）→ quick return → early exit（`k=0` 或 `alpha=(0,0)`）→ tiling → workspace（一次性申请，容量不足返回错误码，不做分批回退）→ 指针数组准备 → launch。
4. device 侧为 `gemm_batched_kernel.cpp` + `gemm_batched_tiling_data.h` 两个文件，launcher 声明放在 host cpp 里（与 arch35 相同，不建 `_kernel.h`）。共 5 个 kernel：
   - 反交错 kernel（AIV，A、B 各发一次）：交错复数拆成实部、虚部两个连续 fp32 矩阵；通过 tiling 的 `isTranspose/isConjugate` 在拆分时完成转置（N 路径保列主序，T/C 路径边拆边转置）与共轭取负；
   - Cube kernel（AIC）：一个 kernel、一次 launch 内串行算完 T1~T4 四个实数乘积；任务空间为 `batchCount × mTiles × nTiles`，tile 固定 64×64、K 步进 32；批地址从设备指针数组解引用；
   - fallback kernel（AIV）：形状不满足 Cube 对齐门控（m 或 n 非 64 对齐，或 k<32 / 非 32 对齐）时启用，标量点积 + Kahan 补偿求和，索引视角和输出布局与 Cube 一致；
   - 合并 kernel（AIV）：读 T1~T4 和 C 旧值，做 `alpha*(Pr + j·Pi) + beta*C_old`，按交错格式写回 C[i]；
   - beta_scale kernel（AIV）：纯标量 `C[i] = beta * C[i]`，覆盖 `k=0` / `alpha=(0,0)` 路径，运算顺序与 golden 一致，保证位精确。
5. 列主序用角色互换处理：计算 `P^T = op(B)^T · op(A)^T`。反交错直接按交换视角出行主序矩阵（左 X=op(B)^T，(n,k)；右 Y=op(A)^T，(k,m)），Cube 的 ND（行主序）Fixpipe 输出在物理上就是列主序的 P，不需要额外的转置搬运。
6. 边界与负向：零维/零 batch 返回 `SUCCESS`；`k=0`、`alpha=(0,0)` 走 `C=beta*C`；handle 空、标量空、指针数组空、非法枚举、非法前导维、负维度、负 batchCount 按任务书状态码返回。
7. 精度：一般路径实部、虚部分别按 FLOAT32 判定；`alpha=(0,0)` 走位精确路径（EXACT）。
8. 测试：CSV 驱动的 GTest（参照 `test/gemm_batched/cgemm_batched/arch35/`），golden 逐批调 `cblas_cgemm`。CSV 共 1200 条用例，25 列（22 个必需列 + mere_threshold、mare_multiplier、random_seed），覆盖冒烟、方阵、矩形、大 batch（≤1024）、alpha/beta 组合、早退、负向、Inf/NaN、4 个性能标杆及转置组合。附测试 README 给出复现步骤。

# 详细设计（required）

## 算子分析

### 数学公式

记 `X = Xr + i·Xi`（Xr、Xi 为 float32 矩阵），`alpha = ar + i·ai`、`beta = br + i·bi`。

**(1) 4M 分解**

```
op(A)·op(B) = (op_r(Ar)·op_r(Br) − op_i(Ai)·op_i(Bi))
            + i·(op_r(Ar)·op_i(Bi) + op_i(Ai)·op_r(Br))
```

记 `P = op(A)·op(B)`，引入 4 个实数矩阵乘：

```
T1 = op_r(Ar) · op_r(Br)      T2 = op_i(Ai) · op_i(Bi)
T3 = op_r(Ar) · op_i(Bi)      T4 = op_i(Ai) · op_r(Br)

Pr = T1 − T2
Pi = T3 + T4
```

**(2) 转置与共轭可以折算到反交错阶段**

转置只重排元素位置，对实部、虚部分别成立；共轭只改虚部符号：

```
op(X) = X       → op_r(Xr) = Xr，      op_i(Xi) = Xi
op(X) = X^T     → op_r(Xr) = Xr^T，    op_i(Xi) = Xi^T
op(X) = X^H     → op_r(Xr) = Xr^T，    op_i(Xi) = −(Xi^T)
```

即 OP_C 的「虚部取负」在反交错时做掉，之后实数通路只有「转置 / 不转置」两态：

| 接口 transa / transb | 实数通路 | 反交错虚部处理 |
| --- | --- | --- |
| `ACLBLAS_OP_N` | N | 原样 |
| `ACLBLAS_OP_T` | T | 原样 |
| `ACLBLAS_OP_C` | T | 取负 |

实现上不按 NN/NT/TN/TT 实例化 4 份 Cube kernel：转置/共轭全部由反交错 kernel 的运行期标志处理，进 Cube 的矩阵恒为行主序、无转置，Cube 内没有转置分支。

**(3) alpha / beta 合并**

`C_new = alpha·P + beta·C_old`，逐元素：

```
Re(C_new) = ar·Pr − ai·Pi + br·Cr − bi·Ci
Im(C_new) = ar·Pi + ai·Pr + br·Ci + bi·Cr
```

每元素 8 次乘法、6 次加减，合并 kernel 中走向量指令。

**(4) 列主序与 Cube 行主序的对齐**

Cube 的 L0C 和 Fixpipe 按行主序（ND）组织。计算交换视角：

```
C' ≜ op(B)^T · op(A)^T = P^T
```

C' 是 n×m 行主序矩阵，它的行主序内存布局和 P 的 m×n 列主序布局逐元素相同。host 直接按交换视角构造 tiling，不做 arch35 那种运行期字段 swap：

```
左矩阵 X = op(B)^T：(n, k) 行主序，行步长 leftLd = k
右矩阵 Y = op(A)^T：(k, m) 行主序，行步长 rightLd = m
输出：n 行 × m 列，行步长 outLd = CeilAlign(m, 8)（fp32，Fixpipe c0 对齐）
```

A/B 的反交错直接产出 X/Y 所需布局：trans=N 时保列主序紧凑写出（物理上即交换视角的行主序矩阵），trans=T/C 时边拆边转置出行主序。Cube Fixpipe 落盘的 n×m 行主序结果就是用户视角 m×n 列主序的 P；合并 kernel 按行步长 `tempRowStride = outLd` 读 T 平面各行（即 P 的各列），按 `ldc` 交错写回 C。

角色互换只是 host 侧的 tiling 变换，用户仍按 `(transa, transb, m, n, k)` 原语义传参，A/B 的身份和 lda/ldb 校验都不变。

### 支持数据类型

| 数据 | 类型 | 说明 |
| --- | --- | --- |
| alpha、beta | COMPLEX64（`aclblasComplex`） | 实部/虚部各 float32 |
| Aarray、Barray、Carray | COMPLEX64 | 交错存储，实部在偶数位、虚部在奇数位 |
| 中间量 T1~T4 | FLOAT32 | 仅存在于 workspace |
| 累加精度 | FLOAT32 | L0C 累加为 fp32，`SetHF32Mode(DISABLE)` |

复数类型以仓内 `include/cann_ops_blas_common.h` 为准：

```cpp
typedef struct aclblasComplex {
    float real;
    float imag;
} aclblasComplex;
```

本任务只做 COMPLEX64。实数通路全程 fp32，Cube fractal 的 c0 = 8。tile 常量（`TILE_M/TILE_N = 64`、`TILE_K = 32`、c0 = 8、向量对齐 8）定义在 `gemm_batched_tiling_data.h`。

### 支持形状

| 维度 | 取值范围 | 说明 |
| --- | --- | --- |
| m | `[0, 2^31)` | 0 为 no-op；上界受 tiling 字段宽度和 workspace 容量约束 |
| n | `[0, 2^31)` | 同上 |
| k | `[0, 2^31)` | 0 走 `C = beta*C` 路径 |
| batchCount | `[0, 2^31)` | 0 为 no-op；上限受显存和 workspace 上限（2 GiB）约束 |
| lda / ldb / ldc | 满足前导维下界即可 | 允许大于下界（padding） |

其他约定：

- batch uniform：批次间只允许地址不同；
- 只支持 lda/ldb/ldc 表达的前导维语义，不支持任意步长；
- 无广播；
- 无 dynamic shape 要求，维度都是运行时入参；
- C[i] 原地覆写，各 C[i] 之间不得重叠。

## 算子实现

### 实现方案

整体沿用 `blas/hemm/arch22` 的做法：host 校验 + tiling + workspace，device 多 kernel 同 stream 顺序发；算法沿用 arch35 已验证的 4M 分解。两档硬件差异：

| 项 | arch35（DAV_3510） | arch22（ASCEND_V220） |
| --- | --- | --- |
| L1 | 32 KB | 512 KB |
| L0A / L0B | 64 KB / 64 KB | 64 KB / 64 KB |
| L0C | 256 KB | 128 KB |
| UB | 248 KB | 192 KB |
| Cube block | 16 | 32 |

Cube 侧直接沿用 `blas/hemm/arch22/chemm_kernel.cpp` 里真机验证过的低层 Mmad 流水（`Nd2Nz` + `LoadData3DParamsV2` + `Mmad` + `FixpipeParamsV220`，fp32 c0=8），输出 tile 取 64×64、K 步进 32。占用上：单 tile 累加器 L0C 64×64×4B = 16 KB；A2/B2 各 8 KB；L1 侧 A1/B1/A2/B2 四个缓冲各 8 KB，共 32 KB。选这个小 tile 主要是为了直接复用 chemm 的流水代码，尾块、事件、Fixpipe 参数都不用重新踩坑；更大的 tile 和 L1 复用列为后续调优项（见性能章节），等真机 profiling 后再决定是否值得改。

新增文件（与 arch35 目录组织一致）：

```
blas/gemm_batched/arch22/
├── gemm_batched_tiling_data.h   # 5 个 tiling 结构体 + tile/对齐常量
├── gemm_batched_kernel.cpp      # 5 个 kernel + 5 个 launcher
└── gemm_batched_host.cpp        # 校验、tiling、workspace、指针数组、launch 编排
```

5 个 kernel 入口：`cgemm_batched_deinterleave_kernel`、`cgemm_batched_cube_kernel`、`cgemm_batched_fallback_kernel`、`cgemm_batched_combine_kernel`、`cgemm_batched_beta_scale_kernel`，对应 5 个 `_do` launcher。

总体流程：

```
                       aclblasCgemmBatched(handle, transa, transb, m, n, k,
                                            alpha, Aarray, lda, Barray, ldb,
                                            beta, Carray, ldc, batchCount)
                                          │
                                          ▼
                        ┌────────── 参数校验 ──────────┐
                        │ handle / alpha / beta 非空   │  ── 失败 ─▶ 对应状态码
                        │ m,n,k,batchCount ≥ 0         │
                        │ transa/transb ∈ {N,T,C}      │
                        │ lda/ldb/ldc 前导维约束       │
                        │ batchCount>0 时三数组非空    │
                        └──────────────┬───────────────┘
                                       ▼
                       m=0 || n=0 || batchCount=0  ── 是 ─▶ return SUCCESS
                                       │ 否
                                       ▼
                    k=0 || alpha=(0,0)  ── 是 ─▶ beta=(1,0)? ── 是 ─▶ return SUCCESS
                                       │                      └ 否 ─▶ beta_scale kernel
                                       │ 否
                                       ▼
                          tiling 计算（直接按交换视角构造）
                                       ▼
                          workspace 规划（一次性全量，失败返回错误码）
                                       ▼
              ┌──────────────── kernel 流水（同一 stream）────────────────┐
              │ ① deinterleave (AIV) ×2：A → Ar/Ai(Y)，B → Br/Bi(X)       │
              │    N：保列主序紧凑写出；T/C：边拆边转置；C 额外虚部取负     │
              │ ② cube / fallback 二选一：                                 │
              │    对齐门控通过 → cube (AIC)：单 kernel 串行 T1→T2→T3→T4   │
              │    否则           → fallback (AIV 标量+Kahan)              │
              │ ③ combine (AIV)：alpha/beta 合并，交错写回 C[i]             │
              └───────────────────────────────────────────────────────────┘
                                       ▼
                                return SUCCESS
```

#### host侧设计

**（1）参数校验（`ValidateCgemmBatchedParams`）**

顺序与 arch35 一致，保证多负向条件同时命中时返回相同状态码：

| 顺序 | 校验项 | 失败返回 |
| --- | --- | --- |
| 1 | `handle != nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `alpha != nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `beta != nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `m >= 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `n >= 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `k >= 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `batchCount >= 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `transa ∈ {N,T,C}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 9 | `transb ∈ {N,T,C}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 10 | `lda >= max(1, physRowsA)`，`physRowsA = isTransA ? k : m` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 11 | `ldb >= max(1, physRowsB)`，`physRowsB = isTransB ? n : k` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 12 | `ldc >= max(1, m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 13 | `batchCount > 0` 时 `Aarray/Barray/Carray` 均非空 | `ACLBLAS_STATUS_INVALID_VALUE` |

顺序有语义意义：比如 handle 和 alpha 同时为空时要先返回 `HANDLE_IS_NULLPTR`，负向用例的期望值按这个顺序写。

**（2）quick return 与 early exit**

- `m == 0 || n == 0 || batchCount == 0`：返回 `SUCCESS`，不分配 workspace、不发 kernel；
- `k == 0` 或 `alpha == (0,0)`：
  - `beta == (1,0)`：C 不变，返回 `SUCCESS`；
  - 否则发 beta_scale kernel（无 workspace），逐批 `C[i] = beta * C[i]`；`beta == (0,0)` 时直接写 0，不读 C 旧值。

**（3）tiling 计算**

不沿用 arch35 的二维块枚举，按固定 tile 64×64×32 直接构造 5 组 tiling（结构体见 `gemm_batched_tiling_data.h`）。

反交错 tiling（A、B 各一份，8 字段）：

| 字段 | A（N） | A（T/C） | B（N） | B（T/C） |
| --- | --- | --- | --- | --- |
| `physRows`（源列高） | m | k | k | n |
| `physCols`（源列数） | k | m | n | k |
| `srcLd` | lda | lda | ldb | ldb |
| `dstStride`（输出步长，fp32） | m | m | k | k |
| `isTranspose` | 0 | 1 | 0 | 1 |
| `isConjugate` | 0 | T=0 / C=1 | 0 | T=0 / C=1 |

AIV 核数：`usedAivCoreNum = CalcAivCoreNum(aiv, batchCount·physRows·physCols)`，其中 `CalcAivCoreNum(coreNum, totalElems) = min(coreNum, max(1, ceil(totalElems / 16384)))`，按每核约 16384 个元素切。

Cube / fallback tiling（交换视角，cube.m = 原 n、cube.n = 原 m）：

```
outLd    = CeilAlign(m, 8)                    // 原 m；Fixpipe c0=8 对齐
cube.m   = n;  cube.n = m;  cube.k = k
leftLd   = k;  rightLd = m
mTiles   = n / 64;  nTiles = m / 64
totalTasks   = batchCount · mTiles · nTiles
cubeBlocks   = min(aicCoreNum, totalTasks)
```

Cube 对齐门控（任一不满足走 fallback）：

```
m % 64 == 0  &&  n % 64 == 0  &&  k >= 32  &&  k % 32 == 0
```

fallback 复用同一交换视角（`m=n0, n=m0, k, leftLd=k, rightLd=m, outLd`），核数 `CalcAivCoreNum(aiv, batchCount·m·n)`，输出布局和 Cube 相同，combine 两条路径共用。

combine / beta_scale 核数同样按 `batchCount·m·n` 经 `CalcAivCoreNum` 得到；combine 的 `totalCols = batchCount·n`（按列分核），`tempRowStride = outLd`，`hasBeta = (beta ≠ (0,0))`。

`totalTasks` 用 `int64_t` 中间量计算，超过 `UINT32_MAX` 返回 `ACLBLAS_STATUS_INVALID_VALUE`，防止乘法回绕。

**（4）workspace 规划**

workspace 由 handle 托管（`EnsureDefaultWorkspace` / `GetEffectiveWorkspace`）：默认 32 MiB，不够时倍增重申请，上限 2 GiB（`ACLBLAS_DEFAULT_WORKSPACE_SIZE` / `ACLBLAS_MAX_WORKSPACE_SIZE`）。用户自带 workspace 容量不足时返回 `ACLBLAS_STATUS_ALLOC_FAILED`，不发 kernel。

布局分两段：低地址放 8 组设备侧指针数组（每组 `batchCount×8B`，64B 对齐）；其后是 SoA 数据区，各批次连续排，单批步长按 64B 向上对齐：

| 区域 | 单批字节数（对齐前） | 说明 |
| --- | --- | --- |
| `Ar` / `Ai`（右矩阵 Y 分量，(k,m) 行主序） | `m·k·4` | 紧凑，行步长 m |
| `Br` / `Bi`（左矩阵 X 分量，(n,k) 行主序） | `n·k·4` | 紧凑，行步长 k |
| `T1`..`T4`（(n,m) 行主序，物理即 P 列主序） | `n·outLd·4` | 行步长 outLd |

```
alignedPtrBytes = align64(batchCount × 8)
ptrRegion       = 8 × alignedPtrBytes
aPerBatch       = align64(m·k·4);  aTotal = aPerBatch·batchCount
bPerBatch       = align64(n·k·4);  bTotal = bPerBatch·batchCount
tPerBatch       = align64(n·outLd·4); tTotal = tPerBatch·batchCount
dataRegion      = 2·aTotal + 2·bTotal + 4·tTotal
totalWorkspace  = ptrRegion + dataRegion
```

数据区顺序 `Ar | Ai | Br | Bi | T1 | T2 | T3 | T4`。反交错输出丢掉 lda padding（紧凑步长 m / k），中间矩阵尺寸和用户 lda/ldb 解耦。

任务书 §3.3 四个性能 case（m=n=k、64 对齐，outLd=m，无对齐余数，单批合计 32·s² 字节）：

| case | s=m/n/k | batchCount | 数据区 | 指针区 | 总 workspace |
| --- | --- | --- | --- | --- | --- |
| 1 | 256 | 32 | 64 MiB | 2 KiB | 64.0 MiB |
| 2 | 512 | 16 | 128 MiB | 1 KiB | 128.0 MiB |
| 3 | 1024 | 8 | 256 MiB | 512 B | 256.0 MiB |
| 4 | 2048 | 4 | 512 MiB | 512 B | 512.0 MiB |

都在 2 GiB 上限内。这里的 workspace 是 device 侧中间缓冲，与任务书 §3.4「内存要求：不涉及」不冲突——那条约束的是测试侧 host 数据规模（单用例 ≤ 512 MB）。

超大形状不做分批回退：`totalWorkspace` 超容量时直接透传 `EnsureDefaultWorkspace` 的错误码。4M 分解的 workspace 与 `batchCount·m·n` 同阶，是这个方案的固有成本；分批只抬高形状上限，对四个标杆 case（最大 512 MiB）没有影响，所以首版不做，记为后续项。

**（5）kernel 编排**

一般路径 4 次 launch（cube 与 fallback 二选一）：

```
① cgemm_batched_deinterleave_do(aivCoresA, stream, Aarray, pArr[0], pArr[1], tilingA)
② cgemm_batched_deinterleave_do(aivCoresB, stream, Barray, pArr[2], pArr[3], tilingB)
③ cgemm_batched_cube_do(aicBlocks, stream,
                        pArr[2], pArr[3], pArr[0], pArr[1],     // 左 X = B 分量，右 Y = A 分量
                        pArr[4], pArr[5], pArr[6], pArr[7],     // T1..T4
                        cubeTiling)
   （门控不通过时换 cgemm_batched_fallback_do，参数序列相同，tiling 类型不同）
④ cgemm_batched_combine_do(aivCoresC, stream,
                           pArr[4], pArr[5], pArr[6], pArr[7], Carray, combineTiling)
```

early-exit 路径只有 1 次 launch，不申请 workspace：

```
cgemm_batched_beta_scale_do(aivCores, stream, Carray, betaScaleTiling)
```

Cube kernel 内对每个输出 tile 顺序做 T1→T2→T3→T4（每个乘积一轮完整 K 循环），4 个乘积复用同一套 64×64×32 流水代码和 Fixpipe 地址计算，只换 GM 指针。这样省掉 3 次 launch 和 3 份 tiling 下发；kernel 全场景只有一个版本，没有模板 switch 和转置分支。

**（6）指针数组准备**

kernel 要的是设备侧指针数组。host 在 workspace 的 `ptrRegion` 用 `std::vector<uint64_t>` 构造 8 组地址表，再用 8 次 `aclrtMemcpy(..., ACL_MEMCPY_HOST_TO_DEVICE)` 上传（每次 `batchCount×8B`；数据量很小，合成一次的收益可忽略）：

```
Ar[i] = dataBase + arOff + i * aPerBatch       Ai[i] = dataBase + aiOff + i * aPerBatch
Br[i] = dataBase + brOff + i * bPerBatch       Bi[i] = dataBase + biOff + i * bPerBatch
Tk[i] = dataBase + tkOff + i * tPerBatch       (k = 1..4)
```

用户入参 `Aarray/Barray/Carray` 本身就是设备侧指针数组，转成 `uint8_t*` 直接传给 kernel（kernel 内按 `__gm__ uint64_t*` 解引用），不搬到 host，也不引入 host 同步。

**（7）异步语义**

所有 kernel 通过 `h->stream` 下发，host 不插 `aclrtSynchronizeStream`，同步由调用方在读结果前做。workspace 扩容重申请前 `EnsureDefaultWorkspace` 会调 `SynchronizeHandleStream`，避免释放正在使用的缓冲。

#### kernel侧设计

kernel 清单：

| kernel | 单元 | launch 次数 | 触发条件 | 职责 |
| --- | --- | --- | --- | --- |
| `cgemm_batched_deinterleave_kernel` | AIV | 2 | 一般路径 | AoS→SoA，保序/转置/共轭取负 |
| `cgemm_batched_cube_kernel` | AIC | 1 | 一般路径且门控通过 | 单 kernel 串行 T1..T4，64×64×32 Mmad |
| `cgemm_batched_fallback_kernel` | AIV | 1 | 一般路径但门控不通过 | 标量点积 + Kahan，输出同布局 T1..T4 |
| `cgemm_batched_combine_kernel` | AIV | 1 | 一般路径 | alpha/beta 合并，交错写回 |
| `cgemm_batched_beta_scale_kernel` | AIV | 1 | `k=0`/`alpha=(0,0)` 且 `beta≠(1,0)` | 纯标量 `C=beta*C`（含置零），位精确 |

**（1）反交错 kernel（AIV）**

tiling（8 字段）：

```cpp
struct CgemmBatchedDeinterleaveTilingData {
    int32_t physRows, physCols;   // 源矩阵（列主序 AoS）列高 / 列数
    int32_t srcLd;                // 源前导维（复数元素）
    int32_t dstStride;            // SoA 输出步长（fp32）：A 传 m，B 传 k
    int32_t isTranspose;          // OP_T / OP_C → 1
    int32_t isConjugate;          // OP_C → 1（虚部取负）
    int32_t batchCount;
    int32_t usedAivCoreNum;
};
```

分核以列（N 路径）或行（T 路径）为工作单元：`totalUnits = batchCount × physCols`（N）或 `batchCount × physRows`（T），用 `CgbComputeRange` 给每核分连续区间，空区间的核直接退出。

两条路径：

- N 路径（保序，向量化）：一列在源（AoS，跨步 8B）和目的（SoA，连续）两侧都连续。列起始下标两侧同时 8 复数对齐时走向量快路径：一次 `DataCopy` 连续读至多 128 个复数（`VEC_CHUNK`，`2×chunk` 个 fp32）到 UB，用两条 `GatherMask<float>(..., {1, repeatTimes, 8, 8}, 0)`（selector 1U/2U，参数同 chemm 反交错）分离实/虚；`isConjugate` 时虚部 `Muls(-1.0f)`；两条 `DataCopy` 写出。块间用 `MTE2_V` / `V_MTE3` 事件加 `PipeBarrier<PIPE_ALL>` 串联。列内 8 对齐余数和不对齐的列走 `GetValue/SetValue` 标量路径，保证不丢元素、不越界。
- T/C 路径（边拆边转置，标量）：转置后两侧无法同时连续，逐元素 `dst(row,col) = src(col,row)`，虚部按需取负。标杆性能 case 固定为 NN，不经过这条路径。

UB 占用 3 块裸 buffer：AoS 1024 B + 实部 512 B + 虚部 512 B = 2 KB。kernel 结尾对输出 GM 做 `DataCacheCleanAndInvalid<ENTIRE_DATA_CACHE>` + `PipeBarrier<PIPE_ALL>`，给后续 AIC 读保 cache 一致。

**（2）Cube kernel（AIC）**

任务映射（grid-stride）：

```
totalTasks = batchCount × mTiles × nTiles，mTiles = n/64，nTiles = m/64
for (taskId = GetBlockIdx(); taskId < totalTasks; taskId += GetBlockNum()) {
    batchIdx  = taskId / (mTiles * nTiles);
    tile 坐标 (mBase = 64·mBlockIdx, nBase = 64·nBlockIdx);
    依次 ProcessProduct(T1: Xr·Yr) → T2: Xi·Yi → T3: Xr·Yi → T4: Xi·Yr;
}
```

各批地址从设备指针数组解引用。host 门控已保证 m、n 为 64 倍数、k 为 32 倍数且 k≥32，kernel 内没有尾块和裁边分支。

单乘积流水与 `chemm_kernel.cpp` 的 `ChemmBasicMmad` 一致：

| 阶段 | API 与参数 |
| --- | --- |
| GM→L1（MTE2） | `Nd2Nz` 到 A1/B1（各 2048 fp32 = 8 KB）；左 nValue64/dValue32/srcDValue=leftLd/dstNzC0Stride64，右 nValue32/dValue64/srcDValue=rightLd/dstNzC0Stride32 |
| L1→L0（MTE1） | `LoadData3DParamsV2`：左 `enTranspose=false`、右 `enTranspose=true`；A2/B2 各 2048 fp32 |
| L0 累加（M） | `MmadParams{m=64,n=64,k=32, cmatrixInitVal=initialize, cmatrixSource=false, kDirectionAlign=false}`；首块 initialize=true，后续块累加；`SetHF32Mode(HF32Mode::DISABLE)` |
| L0C→GM（Fixpipe） | `FixpipeParamsV220{mSize=64,nSize=64,srcStride=64,dstStride=(int16)outLd,ndNum=1}`，输出地址 `mBase·outLd + nBase`；c1 4096 fp32 = 16 KB |

事件链（单缓冲，固定事件 ID，同 chemm）：进 tile 前 `SetFlag<FIX_M>(0)`；每个乘积内 `WaitFlag<FIX_M>` → K 循环里 `CopyToL1`（双 `MTE2_MTE1` ID）→ `LoadToL0`（`SetFlag<MTE1_M>`）→ `Compute`（`Wait<MTE1_M>` + Mmad + PipeBarrier）→ 循环尾 `SetFlag<M_FIX>` → `StoreToGm`（`Wait<M_FIX>` + Fixpipe + `SetFlag<FIX_M>`）。相邻 K 步靠硬件事件细粒度重叠。

入口 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)` + `InitSocState()`；结尾对输出 GM 做 ENTIRE clean + `PipeBarrier<PIPE_ALL>`。

**（2b）fallback kernel（AIV）**

门控不通过时启用。按输出元素（交换视角 row∈[0,n)、col∈[0,m)，跨 batch）在 AIV 核间均分；每核跨度向上取齐 8、末块 clamp 到总元素数、多余核拿空区间（`CgbComputeRangeAligned`，不丢边界元素）。每个输出元素算 4 个点积：

```
lIdx = row·leftLd + t;   rIdx = t·rightLd + col;       // X=(n,k)、Y=(k,m) 行主序
T1 += Xr·Yr;  T2 += Xi·Yi;  T3 += Xr·Yi;  T4 += Xi·Yr;  // 4 个独立 Kahan 累加器
outIdx = row·outLd + col;
```

Kahan 补偿（`y = prod − c; t = s + y; c = (t − s) − y; s = t`）用来压长 k 标量累加的误差，k=1、质数维度、小矩阵这些 case 的精度裕度主要靠它。开工前对输入平面 invalidate，写完对 T1..T4 clean（ENTIRE，批区间有守卫），和 AIC 路径的 cache 维护口径一致。

**（3）合并 kernel（AIV）**

tiling（12 字段）：

```cpp
struct CgemmBatchedCombineTilingData {
    int32_t m, n, ldc;
    int32_t tempRowStride;       // = outLd = CeilAlign(m,8)
    float alphaReal, alphaImag, betaReal, betaImag;
    int32_t hasBeta;             // beta != (0,0) 时为 1
    int32_t batchCount;
    int32_t usedAivCoreNum;
    int64_t totalCols;           // batchCount × n
};
```

按列分核（`totalCols = batchCount·n`；一列对应 T 平面交换视角的一行，行内 m 个 fp32 连续），`CgbComputeRange` 均分。列内按 256 fp32（`VEC_TILE`）切块走向量流水，尾块用 `DataCopyPad`（`DataCopyExtParams{1,bytes,0,0,0}` + `DataCopyPadExtParams<float>{false,0,0,0}`，arch22 先例 `blas/hpr/arch22/chpr_kernel.cpp`）。

每块流水（TQue 深度 2，EnQue/DeQue）：

```
T1..T4 各 DataCopyPad 入 UB（4 个常驻队列）
Pr = T1 − T2;  Pi = T3 + T4
Cr = αr·Pr − αi·Pi;  Ci = αr·Pi + αi·Pr
if hasBeta:
    C0 AoS 一块 DataCopyPad 入 UB（aosBytes = 2·VEC_TILE·4B + 对齐）
    GatherMask{1,repeatTimes,8,8} selector 1U/2U 拆 C0r/C0i   // 余数走标量
    Cr += βr·C0r − βi·C0i;  Ci += βr·C0i + βi·C0r
AoS 交织写回 C（TQue + DataCopyPad GM 写出，参照 arch35 gemm_kernel.cpp）
```

`hasBeta = 0` 时不分配也不读 C 旧值，与 cuBLAS beta=0 语义一致，也避免读未初始化内存。

UB 峰值（hasBeta=1，`tileBytes = 1024+32 = 1056 B`、`aosBytes = 2048+32 = 2080 B`）：常驻 8 个 TQue × 深度 2 × 1056 B = 16896 B；beta 路径 4 个队列（cOrig 2080 + coR/coI/scale 3×1056 = 5248 B）× 深度 2 = 10496 B；outQue 2 × 2080 B = 4160 B；合计 31552 B ≈ 30.8 KB，UB 上限 192 KB。向量指令后加 `PipeBarrier<PIPE_ALL>` 再走标量余数（同 chemm 向量 epilogue）。结尾 ENTIRE clean。

**（4）beta_scale kernel（AIV）**

用独立的 8 字段 tiling（`m,n,ldc,betaReal,betaImag,batchCount,usedAivCoreNum,totalCols`）。逐元素纯标量 `GetValue/SetValue`，公式顺序与 golden 的 `ApplyComplexAlphaZeroGolden` 保持一致：

```
real = betaR·cr − betaI·ci;   imag = betaR·ci + betaI·cr;
```

这样 `alpha=(0,0)` 路径与 golden 位精确。`beta=(0,0)` 时两式直接得 0，且 host 已保证不读 C 旧值。分核用 `CgbComputeRangeAligned`（8 对齐跨度、末块 clamp）；结尾 ENTIRE clean + `PipeBarrier<PIPE_ALL>`。`beta == (1,0)` 在 host 侧已直接返回，不进本 kernel。

**（5）同步与 cache 一致性**

- kernel 之间：同一 stream，靠 stream 顺序保依赖，host 不加显式同步，kernel 内也不做核间 `SyncAll`，避免 AIC/AIV 混核同步；
- AIC 内部：`MTE2_MTE1`（双事件 ID）/ `MTE1_M` / `M_FIX` / `FIX_M` 事件链，首块预置、末块回收；
- AIV 内部：`MTE2_V` / `V_MTE3` 事件 + 块/列间 `PipeBarrier<PIPE_ALL>`；
- 跨单元 cache：AIV 写→AIC 读（deinterleave→cube）和 AIC 写→AIV 读（cube/fallback→combine）的边界，各 kernel 对自己产出的 GM 区域做 `DataCacheCleanAndInvalid<ENTIRE_DATA_CACHE>`，fallback 读输入前额外 invalidate。这也是 T1~T4 必须落全局 workspace 的原因。

**（6）S 版本**

任务书只要 `aclblasCgemmBatched`，本次不交付 `aclblasSgemmBatched` 的 arch22 实现，README 中 S 版本保持「不支持」。Cube 的实数 Mmad 流水以后可以直接给 S 版本用。

#### 性能优化设计

**（1）算力账**

4M 分解用 4 次实数乘实现一次复数乘，实数等效 FLOPs 为 `8·m·n·k·batchCount`。按标杆耗时反推所需实数算力：

| case | m/n/k | batch | 复数乘加次数 | 实数等效 FLOPs | 标杆耗时 | 反推实数算力 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 32 | 1.074 G | 4.295 GF | 342.44 us | ≈ 12.5 TFLOPS |
| 2 | 512 | 16 | 4.295 G | 17.18 GF | 1462.32 us | ≈ 11.7 TFLOPS |
| 3 | 1024 | 8 | 17.18 G | 68.72 GF | 5018.64 us | ≈ 13.7 TFLOPS |
| 4 | 2048 | 4 | 68.72 G | 274.9 GF | 18032.75 us | ≈ 15.2 TFLOPS |

四个 case 反推的算力水位在 12~15 TFLOPS，属于 Cube 算力受限区间，意味着性能主要取决于 Mmad 流水能不能喂饱，算法层没有额外加速空间。这个数字只是纸面估算，实际利用率要上机用 msprof 看，首版对应的做法：

1. Cube 直接用 chemm 验证过的 64×64×32 事件级流水，输出 tile 网格跨步跨 AIC 核并行，4 个乘积合一次 launch；
2. 反交错 N 路径整列连续读 + GatherMask 分离，合并阶段按列分核 + 256 fp32 向量块，压低两段 AIV 开销；
3. GM 流量上，4 个 T 矩阵的写再读是 `8·m·n·batchCount` 个 fp32 的额外往返，case 4 约 512 MiB（537 MB）；
4. 如果实测 Cube 利用率上不去，再从下面的备选调优项里挑，功能和接口不受影响。

**（2）首版已做的优化**

| 序号 | 项 | 说明 |
| --- | --- | --- |
| O1 | 4 乘积单 kernel 单 launch | 一个 tile 内串行 T1..T4，共享 tiling、任务映射和 Fixpipe 地址计算，省 3 次 launch 与 tiling 下发 |
| O2 | 角色互换 + 反交错折算转置/共轭 | Cube 内无转置分支，也没有单独的转置 kernel 和对应 GM 流量 |
| O3 | 反交错合并搬运 | N 路径整列一次 `DataCopy` 读入再两条 `GatherMask` 分离（对齐门控，余数标量），比两次跨步读少一倍 MTE2 事务 |
| O4 | 合并按列分核 + 向量化 | 列在 T 平面行内连续，无跨步访存；256 fp32 块 TQue 深度 2，尾块 DataCopyPad，beta 路径按需装载 |
| O5 | quick return / early-exit | 零维直返；`beta=(1,0)` 直返；`alpha=(0,0)` 走无 workspace 的标量 kernel，且位精确 |
| O6 | kernel 边界 ENTIRE clean/invalidate | 用同 stream 顺序代替 AIC/AIV 核间同步，cache 做法在 chemm 上已有先例 |
| O7 | 指针表 host 拼装、小 memcpy 上传 | device 内解析批地址，无 host 往返、无同步 |

**（3）备选调优项（首版不做，待 profiling 后取舍）**

| 序号 | 项 | 收益 / 代价 |
| --- | --- | --- |
| T1 | K 分块跨输出 tile 复用 | 同一 (batch, mBlock) 跨 nBlock 复用左矩阵 K 分块，MTE2 流量最多降 nTiles 倍；要改任务遍历顺序和 L1 驻留 |
| T2 | 加大 tile + L1 双缓冲 | 64×64×32 → 128×128×64 一类的大 tile、ping-pong，单核重叠度更高；流水和尾块要重写 |
| T3 | L1 bank 对齐放置 | ping/pong 按 bank 隔离（参照 cann-samples memory_optimization/l1_bank_conflict），减少 bank 冲突 |
| T4 | T/C 反交错向量化 | 转置路径目前纯标量，可改 UB 内小块（8×8）转置或 DataCopy 3D；标杆 case 是 NN，不影响验收 |
| T5 | workspace 分批回退 | 容量不足时按 batchChunk 多轮（`max(1, floor((avail−ptrRegion)/perBatch))`），抬高形状上限；标杆 case 不受益 |
| T6 | 全融合（T 不落 GM） | AIV 经 L0C↔UB 通路直接合并，省 workspace 和 GM 流量，但要做 AIC/AIV 核间同步，复杂度高，作为长期方向 |

**（4）方案取舍**

| 方案 | 思路 | 不取的原因 |
| --- | --- | --- |
| A（本设计） | 4 个全局 T 矩阵，分步 kernel：deinterleave×2 → Cube 或 fallback → combine | 采用。与 arch35/chemm 同构，无核间同步，任意形状有标量兜底；代价是 workspace 和 T 矩阵的 GM 往返 |
| B | 不做反交错，Cube 直接跨步读交错复数 | MTE2 有效带宽减半，转置组合下跨步读不好表达，省的 workspace 不抵带宽损失 |
| C | 全融合，T 不落 GM，AIV 直接读 L0C 合并 | workspace 和流量最省，但涉及 L0C↔UB 通路与 AIC/AIV 同步，OP_C 还要额外准备取负副本，首版调试成本高（即 T6） |
| D | 3M 分解（Karatsuba）：`k3=(Ar+Ai)(Br+Bi)`，`Re=k1−k2`、`Im=k3−k1−k2` | 实数乘 4→3，但 `k3−k1−k2` 有灾难性抵消，`|Ar|≈|Ai|`、`|Br|≈|Bi|` 时误差放大，rtol=2^-10 下不稳，放弃 |

**（5）性能验证方法**

- 按任务书 §3.3：warmup 后有效采样 > 50 次取平均；
- GTest 输出的耗时含 host 准备 + kernel + golden + 比对，只能作为保守上界；kernel 本体耗时另用 `msprof` 采集；
- 性能 case 的 workspace 和 device 内存占用一并记进自测报告（任务书 §4 要求）。

## 支持硬件

| 芯片 | 本次 |
| --- | --- |
| Atlas 800I/T A2（910B 系列） | 新增支持 |
| Atlas 800I/T A3（ascend910_93） | 新增支持 |
| Ascend 950PR / 950DT | 已有 arch35，本次不动 |

- 架构目录：`arch22`
- 验收环境：CANN 9.1.0
- 性能测试设备：Atlas 800T A2（910B3）；A3（ascend910_93）同样需要跑功能验收

## 算子约束限制

1. batch uniform：所有批次共享 m/n/k、lda/ldb/ldc、transa/transb，批地址由指针数组给出，不要求等差。
2. 各 C[i] 不得重叠，重叠时行为未定义。
3. 仅支持前导维语义的连续存储，不支持任意步长。
4. 无广播。
5. 任务书不要求确定性计算，本设计也不做额外保证（Cube 累加顺序固定，但不承诺跨版本位一致）。
6. `beta = (0,0)` 时 C[i] 无需初始化，kernel 不读旧值。
7. alpha、beta 在 Host 内存，传 Device 指针不支持（与 950 侧口径一致）。
8. transa/transb 只接受 {N,T,C}，其余返回 `ACLBLAS_STATUS_INVALID_VALUE`。
9. m/n/k、lda/ldb/ldc 与 tiling 字段需容纳于 `uint32_t`，超范围返回 `ACLBLAS_STATUS_INVALID_VALUE`。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 内容 | 来源 |
| --- | --- | --- |
| 精度判据 | 逐批全矩阵比对，实部、虚部分开判：`\|actual − golden\| ≤ atol + rtol × \|golden\|`；`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2`（或 32×ULP） | 生态算子开源精度标准 experimental_standard.md |
| 阈值 | atol = 2⁻¹⁶ ≈ 1.5259e-5；rtol = 2⁻¹⁰ ≈ 9.7656e-4；required_matched_ratio = 0.99；max_abs_error 限 1e-2 或 32×ULP | 同上 |
| 位精确特例 | `alpha = (0,0)` 时 `C[i] = beta * C[i]` 按 EXACT 判 | 任务书 §3.2 |
| golden | Netlib 无 batched 接口，逐批调 `cblas_cgemm`（CblasColMajor），语义对齐 `cgemm.f` | 任务书 §3.2 |
| 性能 | 四个 case 平均单次耗时不高于 342.44 / 1462.32 / 5018.64 / 18032.75 us；warmup 后有效采样 > 50 次 | 任务书 §3.3 |
| 测试工程口径 | MIXED_TOLERANCE：`cgemm_batched_test.cpp` 的 `VerifyCgemmBatchedPrecision` 对实/虚部分别 `applyMixedTolerance`，逐元素 `tolLimit = atol + rtol × |golden|`、`ulpLimit = max(1e-2, 32 × ULP(|golden|))`，matchedRatio ≥ 0.99 且无元素超 ulpLimit 判 PASS；实/虚独立判，比对范围整个 `ldc × n` 缓冲，逐批判；任一侧 NaN/Inf 即 FAIL。CSV 的 mere_threshold / mare_multiplier 列不参与本族判定 | `test/frame/verify.h`、arch35 现有测试 |

精度上的几点考虑：

1. 4M 分解里实部 `T1−T2`、虚部 `T3+T4` 各有一次加减，误差量级与单精度累加同阶（约 `√k · ε`），k ≤ 2048、rtol = 2^-10 下有裕度，但裕度多大要以上机实测为准；
2. Cube L0C 累加是 fp32，和 golden 的 `cblas_cgemm` 累加精度同级，没有降精度环节；
3. `alpha = (0,0)` 绕开 4M 通路，走位精确 beta 缩放，满足 EXACT；
4. Inf/NaN 传播存在不确定：`T1 = T2 = Inf` 时 `T1−T2` 产生 NaN，而 golden 复数乘法在不同运算次序下可能得到 Inf 或 NaN。这类用例的期望以 cblas 实际结果为准，CSV 中单独成组（TC_EX），不和常规用例混判。

## 兼容性分析

1. 接口：不新增、不修改对外声明，只在 `arch22/` 下补实现，与 arch35 按 SOC 版本分派；签名逐参数对应 `cublasCgemmBatched`。
2. README：`blas/gemm_batched/README.md` 中 cgemm 的 Atlas A2/A3 改为「支持」；sgemm 保持「不支持」，与交付内容一致。
3. 与 950 实现的关系：arch35、arch22 独立编译、按 SOC 选择，互不影响，只共享头文件声明和 README。
4. 与同族算子的一致性：复数分解、workspace 布局、kernel 编排参照 `blas/hemm/arch22`、`blas/symm/arch22`。
5. ABI/API：纯新增实现，不改任何已有接口行为和内存布局。
6. 测试：`test/gemm_batched/cgemm_batched/arch22/` 复用上层 `cgemm_batched_param.h`、`cgemm_batched_golden.h` 和 `cmake/test.cmake` 的 arch22 构建/拷贝规则，新增 `cgemm_batched_test.cpp`、`cgemm_batched_npu_wrapper.h`、`cgemm_batched_test.csv`（1200 条，25 列）、测试 `README.md`，不动 arch35 测试资产。

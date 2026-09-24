# 需求背景（required）

## 需求来源

9 月社区任务：在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3，arch22）上基于 Ascend C 开发单精度复数批量三角求解算子 `aclblasCtrsmBatched`。

- 任务书：`aclblasCtrsmBatched_A2A3_task_doc.md`（本文唯一语义真值源；配套 `test_cases/README.md` 若与任务书冲突，以任务书为准）
- 设计文档提交仓：`cann-ops-competitions`（`https://gitcode.com/cann/cann-ops-competitions`，任务书中的 `cann-competitions` 为其别名）
- 提交路径：`04_tasks/01_community-task-2026/tasklist/9月社区任务-aclblasCtrsmBatched算子开发(A2A3)/Ethan-ki/docs/design.md`
- 代码合入仓：`https://gitcode.com/cann/ops-blas`
  - 实现：`blas/trsmbatched/arch22/`
  - 测试：`test/trsmbatched/ctrsmbatched/arch22/`
- 对标接口：cuBLAS `cublasCtrsmBatched`；单批数值 golden 为 Netlib `cblas_ctrsm` 逐 batch
- 工程模式：kernel 直调（句柄式 BLAS），不走 ACLNN / tilingKey 注册

## 背景介绍

### aclblasCtrsmBatched 算子实现

在 ops-blas 开源仓框架下，用 Ascend C kernel 直调在 Atlas A2/A3（架构目录 `arch22`，`ArchType::ASCEND_V220` / DAV_2201）实现 COMPLEX64 批量三角线性系统求解。

对每个 batch `i ∈ [0, batchCount-1]` 独立求解，解原地覆写 `B[i]`：

```
side=LEFT :  op(A[i]) * X[i] = alpha * B[i]
side=RIGHT:  X[i] * op(A[i]) = alpha * B[i]
输出：B[i] ← X[i]
```

`op(A)` 由 `trans` 决定：`N` → `A`，`T` → `Aᵀ`，`C` → `Aᴴ`。`uplo` 指定引用上/下三角，`diag=UNIT` 时对角视为 1 且不读存储。矩阵列主序；A、B 为设备侧指针数组；batch 为 uniform（共享 m/n/lda/ldb 与全部枚举）。

### 同类算子现状分析

当前 `Ethan-ki/ops-blas`（已与 `cann/ops-blas` master `2c9b9b7` 对齐）相关基线：

| 算子 / 目录 | 数据类型 | 仓内现状 | A2/A3 |
| --- | --- | --- | --- |
| `aclblasStrsmBatched`（`blas/trsmbatched/arch35/`） | FLOAT32 | host + kernel + tiling；D2H 指针数组后 Host 逐 batch 发射 | 不支持 |
| `aclblasCtrsmBatched` | COMPLEX64 | **无声明、无实现** | 不支持 |
| `blas/trsmbatched/` | — | 仅 `arch35/`，无 `arch22/` | — |
| `test/trsmbatched/` | — | 仅 `strsmbatched/arch35/` | — |
| `aclblasCsyrk` / `Chemm` / `Csymm` / `Cherk`（`blas/*/arch22/`） | COMPLEX64 | 共用 `complex_blas3_arch22.h`：Phase0 反交错 + Phase1 实数 Matmul + Phase2 合并 | 支持 |

结论：

1. 本任务必须新增 `aclblasCtrsmBatched` 公共声明（`include/cann_ops_blas.h`），并在 `blas/trsmbatched/arch22/` 落地实现；不得另开产品私有 API。
2. arch35 `StrsmBatched` 的 **panel SIMT + trailing GEMM/AXPY** 算法可作实数参考，但：(a) 无复数；(b) Host 逐 batch 发射无法撑起任务书 §3.3 的 `batchCount=128` 性能 case；(c) arch35 L1=32KB / L0C=256KB / Cube block=16，arch22 L1=512KB / L0C=128KB / Cube block=32 / UB=192KB，tiling 必须按 arch22 重做。
3. arch22 复数 BLAS3 已验证「复数拆成实/虚 → Cube 走 FP32 Matmul → 合并写回」。trailing 更新走 4M（或 3M 折叠）实数 GEMM；panel 求解在 AIV 上用实/虚展开的前代/回代。

### aclblasCtrsmBatched 算子功能分析

**功能**：对一批大小一致、相互独立的三角系统做 COMPLEX64 批量求解，解原地覆写 `B[i]`。

**输入**：handle、side、uplo、trans、diag、m、n、alpha（Host 复数标量）、A[]（Device 指针数组）、lda、B[]（Device 指针数组，输入右端）、ldb、batchCount。

**输出**：`B[i]` 原地覆写为 `X[i]`。各 `B[i]` 之间不得重叠。不返回视图。

**支持数据类型**：仅 COMPLEX64（`aclblasComplex`，实部/虚部各 float32）。

**支持广播**：不涉及。无 dynamic shape 要求（m/n/batchCount 为运行时 Host 标量）。

**异步**：依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 ops-blas 的 `arch22` 路径实现 `aclblasCtrsmBatched`，接口与 cuBLAS `cublasCtrsmBatched` 参数序列对齐，覆盖 LEFT/RIGHT × UPPER/LOWER × N/T/C × NON_UNIT/UNIT 共 24 种组合，精度达到生态 COMPLEX64 混合容差，性能不超过任务书 §3.3 达标耗时。

函数原型（任务书 §2.3，禁止增加 C/ldc 离席输出）：

```cpp
aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* const A[], int lda,
    aclblasComplex* const B[], int ldb,
    int batchCount);
```

## 需求拆解

1. 公共头文件 `include/cann_ops_blas.h` 增加与 `aclblasStrsmBatched` 并列的 `aclblasCtrsmBatched` 声明；`blas/trsmbatched/README.md` 产品支持表将 Atlas A2/A3 标为支持。
2. 实现放 `blas/trsmbatched/arch22/`，文件名前缀 `ctrsmbatched_`，由顶层 CMake 按 SOC_ARCH_DIRS=arch22 编入。
3. Host 参数校验与错误码严格按任务书 §2.4/§2.5（非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`，不是 `INVALID_ENUM`）。
4. Kernel 直调：handle 取 stream；A/B 为设备侧指针数组。
5. 覆盖 24 种枚举组合、lda/ldb padding、alpha=(0,0) 置零、m=0/n=0 no-op、UNIT 跳过对角。
6. 精度：cblas `ctrsm` 逐 batch 作 golden，COMPLEX64 混合容差。
7. 性能：Atlas 800T A2 (910B3) 上 5 条标杆 Avg time 不超过任务书表。
8. 测试：`test/trsmbatched/ctrsmbatched/arch22/`，CSV 驱动 GTest；以任务配套 `ctrsmbatched_test.csv` 为底，冲突项按任务书修正。

# 详细设计（required）

## 算子分析

### 数学公式

记 `k = (side==LEFT) ? m : n`。`A[i]` 为 `k×k` 三角阵（数组形状 `lda×k`），`B[i]` 为 `m×n`（数组形状 `ldb×n`），列主序。

```
LEFT :  op(A[i]) X[i] = alpha B[i]
RIGHT:  X[i] op(A[i]) = alpha B[i]
B[i] ← X[i]
```

`op(A)`：

| trans | op(A) |
| --- | --- |
| `ACLBLAS_OP_N` (111) | A |
| `ACLBLAS_OP_T` (112) | Aᵀ |
| `ACLBLAS_OP_C` (113) | Aᴴ = conj(A)ᵀ |

前代/回代（LEFT、`diag=NON_UNIT`、`trans=N`、`uplo=LOWER`，对其余组合作对称改写）：

```
X[0,:]   = alpha * B[0,:] / A[0,0]
X[r,:]   = (alpha * B[r,:] - Σ_{c<r} A[r,c] * X[c,:]) / A[r,r] , r = 1..m-1
```

`diag=UNIT`：跳过对角读写，除数视为 `1+0i`。

复数乘除在 arch22 上展开为实数：

```
(a+bi)(c+di) = (ac-bd) + (ad+bc)i
(p+qi)/(u+vi) = ((pu+qv) + (qu-pv)i) / (u²+v²)
```

RIGHT 规约为 LEFT：先把 `B` 视为 `n×m`（转置），再解

```
op(A)ᵀ Xᵀ = alpha Bᵀ
```

trans 映射：`N→T`，`T→N`，`C→N` 且对 A 取共轭（因为 `(Aᴴ)ᵀ = conj(A)`）。解完再转置写回 `B`。

`alpha = (0,0)`：不引用 `A[i]`，逐批将 `B[i]` 的逻辑 `m×n` 置零后返回成功。

确定性：不要求逐位一致（浮点累加顺序不保证）。

### 支持数据类型

仅 COMPLEX64。比对时实部、虚部分别按 FLOAT32 档混合容差判定。

### 支持形状

- `m ≥ 0`，`n ≥ 0`，`batchCount ≥ 1`
- LEFT：`lda ≥ max(1,m)`；RIGHT：`lda ≥ max(1,n)`；`ldb ≥ max(1,m)`
- 每批独立地址，无广播、无超出 lda/ldb 的非连续访问
- `m=0` 或 `n=0`：合法 no-op，不启动计算 kernel

## 算子实现

### 实现方案

工程文件（任务书 §5 + ops-blas 目录规范）：

```
blas/trsmbatched/arch22/
├── ctrsmbatched_host.cpp          # 校验、路径选择、workspace、launch
├── ctrsmbatched_kernel.cpp        # AIV panel/zero/scale/transpose + AIC trailing GEMM
├── ctrsmbatched_kernel.h          # launcher 声明
└── ctrsmbatched_tiling_data.h     # Host/Device 共享 POD

test/trsmbatched/ctrsmbatched/
├── CMakeLists.txt
├── ctrsmbatched_param.h
├── ctrsmbatched_golden.h          # 逐批 cblas_ctrsm
└── arch22/
    ├── ctrsmbatched_npu_wrapper.h
    ├── ctrsmbatched_test.cpp
    └── ctrsmbatched_test.csv
```

#### 3.2.1 host 侧设计

入口 `aclblasCtrsmBatched` 流程：

```
1. handle == nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. 校验枚举 / m,n / alpha / A,B / lda,ldb / batchCount
3. m==0 || n==0 → SUCCESS（不再访问 A/B）
4. 取 stream、AIV/AIC 核数
5. D2H 拷贝指针数组（batchCount 个地址），检查每个 A[i]/B[i] 非空
   （alpha=(0,0) 时不要求 A 及 A[i] 有效）
6. 按 (side, k=m or n, n_rhs, batchCount) 选择 Path0 / Path1 / Path2
7. 在 handle->stream 上发射 kernel，异步返回
```

**参数校验（任务书 §2.4，全部在计算前完成）：**

| 条件 | 返回码 |
| --- | --- |
| handle 为 nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| side/uplo/trans/diag 不在规定枚举 | `ACLBLAS_STATUS_INVALID_VALUE` |
| m<0 或 n<0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| batchCount<1 | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| B 为 nullptr，或任一 B[i] 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha≠0 且（A 为 nullptr 或任一 A[i] 为 nullptr） | `ACLBLAS_STATUS_INVALID_VALUE` |
| lda/ldb 不满足前导维约束 | `ACLBLAS_STATUS_INVALID_VALUE` |

与仓内部分 arch22 算子使用 `INVALID_ENUM` 不同：本算子非法枚举按任务书返回 `INVALID_VALUE`。

**路径选择：**

| 路径 | 条件 | 行为 |
| --- | --- | --- |
| Path0 Zero | `alpha.real==0 && alpha.imag==0` | AIV 按 batch 将 B[i] 的 m×n 置零，不读 A |
| Path1 SIMT | `k ≤ PANEL_SIMT`（初值 64）或 trailing GEMM 无法饱和 Cube | AIV 按 (batch, 右端列) 做完整三角求解 |
| Path2 Blocked | 其余（含 §3.3 五条性能 case） | 沿 k 方向切 panel：panel SIMT + trailing 4M Cube GEMM |

`PANEL_SIMT`、panel 宽 `bs∈{32,64,128}` 按 k 与 AIC 核数在 Host 选定，写入 tiling；不使用 ACLNN tilingKey。

**指针数组策略：**

任务书要求「A/B 为设备侧指针数组，其中某地址为 nullptr 返回 INVALID_VALUE」。Host 必须 D2H 指针值做空指针检查。发射时仍把 **Device 侧原指针数组** 传给 kernel，由 kernel 按 `batchIdx` 解引用，避免 arch35 那种 Host 循环 `batchCount` 次 launch（`batchCount=128` 时 launch 开销不可接受）。D2H 仅拷贝 `batchCount * 8B`，可忽略。

**分核策略：**

- 优先满核。
- Path1：工作项 `work = batchCount * n_rhs`（LEFT 时 n_rhs=n，RIGHT 转置后 n_rhs=m）。核 `t` 处理 `work` 的 round-robin 子集，每项对应「一个 batch 的一列右端」。
- Path2 panel：同上，但每列只解当前 panel 的 `bs` 行。
- Path2 GEMM：按 `Ceil(Mtile)×Ceil(Ntile)×batchCount` 发 AIC block，`blockIdx` 映射 `(batch, tileM, tileN)`，超额时 round-robin。
- 不能均分时余数分给前若干核（大核多 1 个工作项）。

**数据分块与 workspace：**

arch22：`HardwareInfo<ASCEND_V220>`，UB=192KB，L1=512KB，L0A/L0B=64KB，L0C=128KB，Cube fractal C0=32（FP32 按 8）。

- Panel 在 UB 内保留：A 的 `bs×bs` 三角（实/虚）+ 一列 X 的 `bs` 复数 + 向量计算临时。`bs=64` 时 A 平面 `64*64*8=32KB` 量级，可 double buffer。
- Trailing GEMM 复用 `complex_blas3_arch22.h` 的 Matmul 形状：`CBLAS3_TILE_M/N=128`，`CBLAS3_BASE_K=64`（L0 双缓冲上限）。
- Workspace（handle 默认 workspace，不足则 `EnsureDefaultWorkspace`）：
  - RIGHT：`B` 转置缓冲 `batchCount * n * m * sizeof(aclblasComplex)`（可按 batch 分组切块，单组不超过 workspace）。
  - Path2：每 batch 的 trailing 实数临时 `Ar,Ai,Xr,Xi,Trr,Tii,Tri,Tir` 按 panel 复用，不按全 k 常驻。
  - 不在 kernel 内同步分配；Host launch 前完成容量检查，失败返回 `ACLBLAS_STATUS_ALLOC_FAILED` / workspace 不足码。

**tiling 结构（无 tilingKey，用 pathId + 标志位）：**

```cpp
struct CtrsmbatchedTilingData {
    uint32_t m, n, k, lda, ldb, batchCount;
    uint32_t side;      // 141/142
    uint32_t uplo;      // 121/122
    uint32_t trans;     // 111/112/113
    uint32_t diag;      // 131/132
    uint32_t pathId;    // 0=zero, 1=simt, 2=blocked
    uint32_t panelStart, panelSize;
    float alphaReal, alphaImag;
    uint32_t conjugateA; // RIGHT+OP_C 规约后的共轭标志
};
struct CtrsmbatchedGemmTilingData {
    uint32_t gemmM, gemmN, gemmK, tileM, tileN, tileK;
    uint64_t aOffset, xOffset, dstOffset;
    uint32_t batchCount;
};
```

字段 Host/Device 字节布局一致。

#### 3.2.2 kernel 侧设计

Init：绑定 GM 指针数组、tiling、UB TPipe。Process：CopyIn → Compute → CopyOut。

**1. 复数能力：** arch22 Vector/Cube 均无 native complex64。全程拆成 `float` 实/虚。不调用不存在的 `Add/Mul` complex 重载。

**2. Path0 Zero kernel（AIV）：**
按 `(batch, col)` 把 `B[i][0:m, col]` 置 `0+0i`。`DataCopyPad` 处理 `ldb` padding，只写逻辑 `m` 行。

**3. Path1 / panel SIMT kernel（AIV）：**
- CopyIn：从 `Aarray[batch]` 按列主序只搬 `uplo` 指定三角；`diag=UNIT` 不读对角。
- 对当前右端列，按求解方向做前代或回代。`trans=T/C` 时按行（即原列的转置视图）取 A；`trans=C` 或 `conjugateA=1` 时虚部取负。
- 对角除法：`NON_UNIT` 用上面的复数除；`UNIT` 跳过。
- CopyOut：原地写回 `Barray[batch]` 对应列。
- 列间无依赖，多核按列并行；同一列的行递推在单核内串行，保证依赖。

**4. Path2 trailing GEMM（AIC + AIV）：**
panel 解出 `X_panel` 后，trailing 块：

```
B_trail -= A_trail * X_panel     // trans=N
B_trail -= A_trailᵀ/ᴴ * X_panel  // trans=T/C
```

实现：

1. AIV 反交错：`A_trail`、`X_panel` → 实/虚 packed（列主序 packed，复用 `complex_blas3_arch22.h` Phase0）。
2. AIC：4 次 FP32 Matmul（`Ar*Xr, Ai*Xi, Ar*Xi, Ai*Xr`），tile 128×128×64。
3. AIV 合并：`real = Trr-Tii`，`imag = Tri+Tir`（`trans=C` 时虚部符号按共轭改写），再按 `alpha` 已在 panel 阶段乘到 X 上、trailing 只做减法。
4. 写回 `B_trail`。

Cube 形状与 `complex_blas3_tiling_data.h` 的 `CBLAS3_TILE_*` 保持单一来源，避免 Host/Device 两套常量漂移。

**5. 转置 kernel（RIGHT 规约，AIV）：**
`B(m,n,ldb) → Bt(n,m,ldbt)` 与反向。只转逻辑矩形，保留 padding 不参与。

**6. 同步：** 同一 stream 上顺序发射 Zero/Transpose/Panel/Deinterleave/GEMM/Combine。Panel 与其后 GEMM 之间依赖 `X_panel`，必须等 panel kernel 完成；不同 panel 串行。禁止跨核 UB 别名。

**7. Inf/NaN：** 不做额外清洗。输入 Inf/NaN 按浮点运算传播，精度用例按任务书允许场景比对。

Host 调度伪代码：

```
if Path0:  zero_kernel_do(Barray, tiling, blocks, stream); return;
if RIGHT:  transpose_all_B; swap(m,n) 语义上进入 LEFT; 映射 trans
for panel in panels(k, bs):          // Path1 视为一个 panel 覆盖全部 k
    panel_kernel_do(Aarray, Barray, panelTiling, stream)
    if Path2 and has_trailing:
        deinterleave + gemm4 + combine_sub
if RIGHT:  transpose_back_B
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B / arch22） | √ |
| Atlas A3 训练/推理系列（arch22） | √ |
| Ascend 950PR/DT（arch35） | 本任务不实现（已有/另任务） |

自验证覆盖一种款型即可；精度建议 910B3 或 A3，性能建议 910B3。

## 算子约束限制

- 仅 COMPLEX64，列主序，uniform batch
- 不支持超出 lda/ldb 的非连续访问，不做 broadcast，不做奇异性检测
- `diag=NON_UNIT` 时对角须非零（测试侧 boost 保证）；本算子不检测奇异
- `alpha` 仅 Host 指针（较 cuBLAS 收紧）
- 各 `B[i]` 不得重叠；原地覆写，不返回视图
- `batchCount < 1` 非法（任务书 §2.4）。配套 CSV `TC_ED_184 batch0_noop` 期望 SUCCESS，与任务书冲突，合入测试时改为 `ACLBLAS_STATUS_INVALID_VALUE`
- 配套 CSV 中 `null_carray` / `null_c_elem` / `invalid_ldc` 对应不存在的 C/ldc 参数，按任务书删除或忽略，不实现离席输出 C

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden = 逐 batch `cblas_ctrsm`；比对 B[i] 的逻辑 m×n。COMPLEX64：rtol=atol=`2^-13`≈1.22e-4，`matched_ratio≥0.99`，`max_abs_error ≤ 1e-2 或 32*ULP`。逐元素 `|actual-golden| ≤ atol + rtol×|golden|`。含规约时可放宽 max_abs 至 2ULP | 任务书 §3.2；[混合容差](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | 设备 Atlas 800T A2 (910B3)。warmup 后有效采样 >10 次取 Avg time (us)，不超过下表 | 任务书 §3.3 |
| 内存标准 | 不涉及 | 任务书 §3.4 |

性能标杆：

| case | batchCount | m | n | side | uplo | trans | diag | 达标耗时 (us) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 256 | 256 | LEFT | LOWER | N | NON_UNIT | 895.1 |
| 2 | 128 | 384 | 512 | LEFT | UPPER | C | NON_UNIT | 5551 |
| 3 | 32 | 1024 | 1024 | RIGHT | LOWER | T | UNIT | 13884 |
| 4 | 8 | 2048 | 2048 | LEFT | UPPER | N | NON_UNIT | 21816 |
| 5 | 2 | 4096 | 4096 | RIGHT | UPPER | C | UNIT | 40969 |

测试工程：CSV + C++ GTest，`--ops=ctrsmbatched`，`--soc ascend910b3`。golden 在 Host 调 cblas。负向用例只查返回码，不比数值。

## 兼容性分析

新接口。与已有 `aclblasStrsmBatched` 共存：实数走 arch35 文件，复数走 arch22 新文件，不改实数行为。公共头仅追加声明，不修改既有符号。CANN 9.1.0。

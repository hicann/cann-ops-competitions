# aclblasCtpmv 算子设计文档（Ascend 950PR）

# 需求背景（required）

## 需求来源

本需求来自 CANN 社区任务「算子实操工坊-上海站-aclblasCtpmv算子开发(950)」（任务链接：https://www.hiascend.com/activities/task-center/details/87b85654a9444c119727ee5339f149df?menu=tasks ）。

任务要求在昇腾 NPU（Ascend 950PR / arch35）上使用 Ascend C 开发单精度复数三角压缩存储矩阵-向量乘算子 `aclblasCtpmv`，功能、参数语义对标 cuBLAS `cublasCtpmv` 与 Netlib `ctpmv`，代码最终合入昇腾算子开源仓 ops-blas（https://gitcode.com/cann/ops-blas ）的 `blas/tpmv/arch35/` 目录，测试代码合入 `test/tpmv/ctpmv/arch35/`。

## 背景介绍

### aclblasCtpmv 算子实现

BLAS Level-2 的 TPMV（Triangular Packed Matrix-Vector multiply）完成

```
x := op(A) * x
```

其中 A 为 n×n 三角复数矩阵，以 packed（压缩）格式按列优先紧凑存放，只占 n(n+1)/2 个元素，无前导维 lda；x 为长度 n 的复数向量，**原地（in-place）**输入输出。TPMV 广泛用于三角方程回代、协方差/Cholesky 因子相关计算以及科学计算中的三角变换。

本次任务的实现路径：

- 目标硬件：Ascend 950PR（arch35），CANN 9.1.0
- 编程语言：Ascend C（arch35 SIMT 编程模型，`simt_api/asc_simt.h`）
- 工程模式：**Ascend C kernel 直调**。Host 侧提供句柄式 BLAS 接口（`aclblasHandle_t` 携带 stream），通过 `<<<blocks, nullptr, stream>>>` 直接下发 NPU kernel，不经过 aclnn 图/原型注册链路
- 实现目录：`blas/tpmv/arch35/`（Host + Kernel + Tiling）
- 接口声明位置：`include/cann_ops_blas.h`（新增声明，产品线共用，禁止 950PR 私有平行接口）

### packed 存储布局与一项关键勘误

Netlib `ctpmv.f` 与 cuBLAS 的实际内存布局如下（以下均为 **0-based** 下标）：

| uplo | 元素 A(i,j) 在 AP 中的下标 | 第 j 列的列首地址 colstart(j) | 第 j 列长度 |
|---|---|---|---|
| UPPER（i ≤ j） | `i + j*(j+1)/2` | `j*(j+1)/2` | `j+1` |
| LOWER（i ≥ j） | `i - j + j*(2n-j+1)/2` ⇔ `i + j*(2n-j-1)/2` | `j*(2n-j+1)/2` | `n-j` |

**勘误说明**：任务书 §2.1 中 LOWER 的描述为 `AP[i + (2*n-j+1)*j/2]`。该式实际是 **LOWER 第 j 列的列首地址 colstart(j)**，而非元素 A(i,j) 的地址，二者相差 `-j`。以 n=3 验证：按任务书公式，A(2,2) 的下标为 `2 + (6-2+1)*2/2 = 7`，而 LOWER 的 AP 只有 n(n+1)/2 = 6 个元素（合法下标 0~5），**直接越界**；按正确公式 `2 + 2*(6-2-1)/2 = 5`，与 Netlib 源码中 `KK` 指针的推进（`KK = KK - (N-J+1)`，对角位于 `AP(KK-N+J)`）完全一致。

因此对标口径明确为：**以 Netlib `ctpmv` 参考实现（golden 由 cblas 生成）的实际内存布局为准**，即

```
UPPER: A(i,j) -> AP[i + j*(j+1)/2]        (i <= j)
LOWER: A(i,j) -> AP[i + j*(2n-j-1)/2]     (i >= j)
```

> 该处已在「风险与待确认项」登记，建议同步在任务讨论帖反馈，避免其他开发者按任务书公式实现导致 golden 比对失败。

### 对标接口现状分析

| 接口 | 现状 | 说明 |
|---|---|---|
| `cublasCtpmv`（cuBLAS） | 语义基线 | 参数序列 `handle, uplo, trans, diag, n, AP, x, incx`；支持 N/T/C、UPPER/LOWER、UNIT/NON_UNIT |
| `ctpmv`（Netlib） | 参考实现 + golden | 与 cuBLAS 语义一致；额外含 `X(J).NE.ZERO` 提前跳过优化（仅在 Inf/NaN 与精确 0 组合时有差异，见 §风险项） |
| `aclblasStpmv`（ops-blas 已有声明） | 同族实数接口 | `include/cann_ops_blas.h:164`，维数参数为 `int`，本次 `aclblasCtpmv` 逐参数对齐 |
| `aclblasSgbmv`（ops-blas arch35 已实现） | 工程范式参考 | 同为 BLAS-2 矩阵-向量乘、arch35 SIMT kernel 直调，Host/Kernel/Tiling 结构与本项目直接对齐 |
| `aclblasCtpmv` | **本次新增** | `include/cann_ops_blas.h` 当前无该声明，需新增 |

### aclblasCtpmv 算子功能分析

- 计算 `x = op(A) * x`，`op(A) ∈ {A, Aᵀ, Aᴴ}`，原地覆写
- A 以 packed 格式存储，只引用 uplo 指定的三角
- `diag = ACLBLAS_UNIT` 时主对角视为 1，**不读取 AP 中对应对角位置**
- `n = 0` 为合法 no-op；`incx` 支持正/负步长；`incx = 0` 非法
- 数据类型：COMPLEX64（`aclblasComplex { float real; float imag; }`，8 字节）

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Ascend 950PR（arch35）上实现 `aclblasCtpmv`，接口签名、参数语义、异常返回与 cuBLAS `cublasCtpmv` 及同族 `aclblasStpmv` 逐参数对齐：

```cpp
aclblasStatus_t aclblasCtpmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const aclblasComplex* AP,
    aclblasComplex* x,
    int incx);
```

接口声明新增于 `include/cann_ops_blas.h`，实现置于 `blas/tpmv/arch35/`，测试工程置于 `test/tpmv/ctpmv/arch35/`。

## 需求拆解

1. **功能**：12 组枚举组合（uplo×trans×diag = 2×3×2）全部正确；支持 incx 正/负步长；支持 n=0 quick return；支持 diag=UNIT 对角不读。
2. **接口**：新增 `aclblasCtpmv` 声明，与 `aclblasStpmv` 逐参数对齐，禁止私有平行 API；返回值遵循 `cann_ops_blas_common.h`。
3. **精度**：实部/虚部分别按 FLOAT32 判定，atol = 2⁻¹⁶、rtol = 2⁻¹⁰、matched_ratio ≥ 0.99、max_abs_err ≤ 1e-2。
4. **性能**：4 条典型 case 平均单次耗时不高于任务书 §3.3 标杆（512/1024/2048/4096）。
5. **工程**：Host + Kernel + Tiling 三件套，kernel 直调；README 产品支持表标注 Ascend 950PR 支持；测试工程含 CSV 用例与 readme。
6. **自验**：覆盖 L0 基础、尺寸扫描、步长、填充（含 Inf/NaN）、边界负向、性能用例，输出自测报告。

# 详细设计（required）

## 算子分析

### 数学公式

```
x := op(A) · x

        ⎧ A      trans = ACLBLAS_OP_N
op(A) = ⎨ Aᵀ     trans = ACLBLAS_OP_T
        ⎩ Aᴴ     trans = ACLBLAS_OP_C
```

逐元素展开：

| uplo | trans | 输出行 i 的计算式 |
|---|---|---|
| UPPER | N | `x_i = Σ_{j=i}^{n-1} A(i,j)·x_j` |
| LOWER | N | `x_i = Σ_{j=0}^{i} A(i,j)·x_j` |
| UPPER | T | `x_i = Σ_{j=0}^{i} A(j,i)·x_j` |
| UPPER | C | `x_i = Σ_{j=0}^{i} conj(A(j,i))·x_j` |
| LOWER | T | `x_i = Σ_{j=i}^{n-1} A(j,i)·x_j` |
| LOWER | C | `x_i = Σ_{j=i}^{n-1} conj(A(j,i))·x_j` |

`diag = UNIT` 时主对角项 `A(i,i)` 不参与读取，等价替换为 `+ 1·x_i`。

复数乘加（a = ar + i·ai，x = xr + i·xi）：

```
非共轭:  acc.r += ar*xr - ai*xi ;  acc.i += ar*xi + ai*xr
共轭:    acc.r += ar*xr + ai*xi ;  acc.i += ar*xi - ai*xr
```

每一步保持「先乘后加」的两步语义，不引入 FMA 融合改写，确保 Inf/NaN 的传播路径与 Netlib golden 一致。

### packed 索引统一形式

记第 j 列列首地址与列长：

```
UPPER:  colstart(j) = j*(j+1)/2        colLen(j) = j+1     A(i,j) -> colstart(j) + i        (i ≤ j)
LOWER:  colstart(j) = j*(2n-j+1)/2     colLen(j) = n-j     A(i,j) -> colstart(j) + (i - j)  (i ≥ j)
```

由「输出行 i ⇔ A 的第 i 列」可得：**trans = T/C 时输出行 i 的数据就是 A 第 i 列的一段连续区间**；由「列 j 贡献给若干输出行」可得：**trans = N 时按列扫描同样落在连续区间**。二者共同构成本算子访存连续性的基础（见 §kernel 侧设计）。

### 支持数据类型

| 张量/标量 | 类型 | 说明 |
|---|---|---|
| AP | `const aclblasComplex*`（COMPLEX64） | Device 内存，只读，长度 n(n+1)/2 |
| x | `aclblasComplex*`（COMPLEX64） | Device 内存，输入/输出，长度 1+(n-1)·\|incx\| |
| n / incx | `int` | Host 标量 |
| uplo / trans / diag | 枚举（int） | Host 标量 |

### 支持形状

- A：n×n 方阵三角部分，packed 压缩为长度 n(n+1)/2 的一维数组（**无 lda**）
- x：长度 `1 + (n-1)·|incx|` 的一维数组（incx < 0 时从尾部 `(n-1)·|incx|` 处反向遍历）
- 不涉及广播；n 为运行时入参（dynamic shape 不要求）

### 参数与异常行为矩阵

| 参数 | 合法值域 | 非法时返回 |
|---|---|---|
| handle | 非 nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | `{ACLBLAS_UPPER, ACLBLAS_LOWER}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| trans | `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| diag | `{ACLBLAS_NON_UNIT, ACLBLAS_UNIT}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| n | n ≥ 0（n = 0 为合法 no-op，直接返回 SUCCESS） | n < 0 → `ACLBLAS_STATUS_INVALID_VALUE` |
| AP | n > 0 时非 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| x | n > 0 时非 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | incx ≠ 0 | `ACLBLAS_STATUS_INVALID_VALUE` |

校验顺序：handle → 枚举 → n → 指针/incx → n == 0 quick return。异常分支均在 Host 侧返回，不下发 kernel。

## 算子实现

### 工程结构与文件清单

```
include/cann_ops_blas.h                       # 新增 aclblasCtpmv 声明（与 aclblasStpmv 对齐）
blas/tpmv/README.md                           # 算子说明，产品支持表标注 Ascend 950PR：支持
blas/tpmv/arch35/ctpmv_host.cpp               # Host：校验、分核、tiling、launch
blas/tpmv/arch35/ctpmv_kernel.cpp             # Kernel：SIMT 计算核 + 核函数下发
blas/tpmv/arch35/ctpmv_tiling_data.h          # Tiling 结构体（Host/Kernel 共享）
blas/tpmv/arch35/ctpmv_gather_kernel.cpp      # Kernel：x -> workspace 的 gather（含 incx 展开）
test/tpmv/ctpmv/ctpmv_param.h                 # CSV 参数解析
test/tpmv/ctpmv/ctpmv_golden.h                # cblas ctpmv golden
test/tpmv/ctpmv/arch35/ctpmv_test.cpp         # GTest 用例
test/tpmv/ctpmv/arch35/ctpmv_test.csv         # 驱动用例（1200 条）
test/tpmv/ctpmv/arch35/CMakeLists.txt
test/tpmv/ctpmv/README.md                     # 测试步骤说明（可复现）
```

Host/Kernel 结构与仓内已合入的 `blas/gbmv/arch35/sgbmv_{host,kernel}.cpp` 对齐：Host 声明 `void ctpmv_kernel_do(...)`，Kernel 中以 `<<<numBlocks, nullptr, stream>>>` 下发；kernel 入口声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`。

### 实现方案

#### 3.2.1 Host 侧设计

##### 0. 核心难点：原地（in-place）语义

`x = op(A)·x` 中 x 既是输入又是输出，**任意输出行 i 的计算都必须使用全部旧值 x_old**。因此不能像 `gbmv`（y 与 x 分离）那样边算边写。

分析依赖方向（输出行 i 依赖的 x 下标区间）：

| uplo | trans | 依赖区间 | 方向 |
|---|---|---|---|
| UPPER | N | `[i, n-1]` | 后缀（suffix） |
| LOWER | N | `[0, i]` | 前缀（prefix） |
| UPPER | T/C | `[0, i]` | 前缀 |
| LOWER | T/C | `[i, n-1]` | 后缀 |

严格串行时可利用顺序消除冲突：后缀依赖按 i 递增、前缀依赖按 i 递减均安全。但**多 block 并行时 block 间的读写区间必然重叠**（例如 UPPER/N 下 block b 需要读 `[r0_b, n-1]`，而其中 `j > r1_b` 的部分由后续 block 写入），任何 grid 级同步方案都不可靠。

**结论：采用 workspace 中转，把"读旧值"与"写新值"彻底分离。** 这也是 cuBLAS 内部实现 tpmv/trmv 的通行做法。

##### 1. 执行路径

Host 侧按 n 选择两条路径：

| 路径 | 触发条件 | kernel 下发次数 | workspace |
|---|---|---|---|
| **路径 S（小 n）** | `n ≤ N_SINGLE`（默认 256，可按实测调优） | 1 | 不需要 |
| **路径 M（大 n）** | `n > N_SINGLE` | 2 | `n × sizeof(aclblasComplex)` 字节 |

**路径 M** 分两个 kernel（同一 stream，天然保序）：

- **K1 `ctpmv_gather_kernel`**：把 `x` 按 incx 步长 gather 到 workspace，得到**连续**的旧值副本 `xold[0..n-1]`。
  `incx > 0: xold[i] = x[i*incx]`；`incx < 0: xold[i] = x[(n-1-i)*(-incx)]`。
- **K2 `ctpmv_kernel`**：读 `AP` 与 `xold`，计算后按 incx 步长 scatter 回 `x`。
  由于读的是 `xold`、写的是 `x`，两者地址空间不同，**彻底消除 in-place 竞争，无需 grid 同步**。

顺带收益：K2 这一计算密集 kernel 对 x 的访问变为**连续**，而非 stride=|incx| 的跨步访问。

**路径 S**（n 小、延迟敏感）：单 block，先把 `x` 按 incx 展开缓存进 UB，块内 `asc_syncthreads()` 后再统一写回 `x`，一次下发完成，省掉一次 launch 与 workspace 申请开销。UB 预算：`2 × n × 8 B ≤ 96 KB`（UB 总容量 248 KB），n = 256 时仅 4 KB，充裕。

##### 2. workspace 获取

复用 ops-blas 既有机制（`common/helper/aclblas_handle_internal.h`）：

```cpp
size_t need = static_cast<size_t>(n) * sizeof(aclblasComplex);   // n=4096 时仅 32 KB
if (EnsureDefaultWorkspace(h, need) != ACLBLAS_STATUS_SUCCESS) { return ACLBLAS_STATUS_ALLOC_FAILED; }
uint8_t* ws = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
```

handle 默认 workspace 为 32 MiB，n ≤ 4096 时需求 ≤ 32 KB，不会触发扩容（扩容会做一次 stream 同步）。用户通过 `aclblasSetWorkspace` 注入自有 workspace 时不重新分配，容量不足直接返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

##### 3. 分核策略与负载均衡（关键）

朴素地按行均分（每 block `n/B` 行）会造成严重倾斜：以 UPPER/N 为例，输出行 i 的工作量 `L(i) = n - i`，编号最小的 block 工作量最大，最大/最小可相差数十倍，整体耗时由最慢 block 决定。

**方案：按"有效 AP 元素数"做工作量均衡的行区间划分，并用闭式前缀和反解直接求出每个 block 的行区间 `[r0, r1)`，无需 Host 传递数组。**

输出行 i 需访问的 AP 元素个数：

| uplo \ trans | N | T / C |
|---|---|---|
| UPPER | `L(i) = n - i` | `L(i) = i + 1` |
| LOWER | `L(i) = i + 1` | `L(i) = n - i` |

前缀和与其反解（E(r) = Σ_{i<r} L(i)）：

```
L(i) = i+1  :  E(r) = r(r+1)/2          =>  r = ⌊(√(1+8E) − 1)/2⌋
L(i) = n−i  :  E(r) = r(2n+1−r)/2       =>  r = ⌊((2n+1) − √((2n+1)² − 8E))/2⌋
```

于是第 b 个 block（`b = GetBlockIdx()`）的行区间为

```
E_total = n(n+1)/2                        // diag=NON_UNIT；UNIT 时为 n(n-1)/2
E0 = E_total * b       / numBlocks
E1 = E_total * (b + 1) / numBlocks
r0 = SolveR(E0);  r1 = SolveR(E1);  clamp 到 [0, n] 且保证 r1 > r0（空块直接 return）
```

说明：`diag = UNIT` 时 L(i) 实际为 `L(i) − 1`，前缀和变为 `E(r) − r`，无简洁闭式。这里仍用 `E(r)` 作均衡代理——每行仅差 1 个元素，块间最大偏差 ≤ `n/numBlocks` 个元素，相对总量可忽略。

##### 4. launch 配置

```
aivCoreNum = GetAivCoreCount();
rMaxExpected = 单个 block 的最大行数（由上述均衡划分决定，≈ √(2·E_total/numBlocks)）
numBlocks = clamp( min(aivCoreNum, 满足 rMax ≤ R_MAX 的最小块数), 1, n )
numThreads = clamp( CeilAlign(rMaxExpected / ROWS_PER_THREAD_TARGET, SIMT_MIN_THREAD_NUM),
                    SIMT_MIN_THREAD_NUM, SIMT_MAX_THREAD_NUM )      // [128, 2048]
```

常量取值沿用 `common/helper/kernel_constant.h`：`SIMT_MIN_THREAD_NUM = 128`、`SIMT_MAX_THREAD_NUM = 2048`；`R_MAX = 1024`（UB 累加器上界，见下）。优先用满核；核数不能被整除时由工作量均衡划分自然吸收余数，无大小核区分。

##### 5. Tiling 数据

```cpp
struct CtpmvTilingData {
    uint32_t n;            // 矩阵阶数
    uint32_t numBlocks;    // 实际下发 block 数
    uint32_t numThreads;   // 每 block 线程数
    uint32_t uplo;         // 0 = UPPER, 1 = LOWER
    uint32_t trans;        // 0 = N, 1 = T, 2 = C
    uint32_t diag;         // 0 = NON_UNIT, 1 = UNIT
    int32_t  incx;         // x 步长，可为负
    int64_t  xBase;        // incx < 0 时的起始下标 (n-1)*(-incx)，否则 0
    uint64_t totalElems;   // E_total，用于工作量均衡划分
};
```

配套常量（同文件内，Host/Kernel 共享）：

```cpp
constexpr uint32_t CTPMV_ROWS_PER_THREAD = 4U;  // numThreads 选取时的每行目标线程数
constexpr uint32_t CTPMV_THREAD_ALIGN    = 128U; // SIMT 线程数对齐粒度
```

##### 6. tilingkey 规划

tilingkey 用于让 kernel 侧在编译期展开分支、消除内层判断。编码：

```
tilingKey = uplo(1 bit) | trans(2 bit)<<1 | diag(1 bit)<<3 | path(1 bit)<<4
          = 共 12 组枚举 × 2 条路径 = 24 个实例
```

Host 侧计算 tilingKey 并通过 `SetTilingKey` / 模板分发；Kernel 侧以 `template<bool UPPER, bool TRANS_N, bool CONJ, bool UNIT>` 实例化 12 个计算变体，路径 S/M 各一份入口。trans = T 与 C 合并为同一模板，仅 `CONJ` 不同（共轭只翻转两个乘加的符号），减少实例化数量。

#### 3.2.2 Kernel 侧设计

Kernel 分 Init / Process 两段：Process = CopyIn（按需将 AP 连续段搬入 UB）→ Compute（复数乘加）→ CopyOut（按 incx scatter 写回 x）。

##### 1. 两类访存模式（保证 AP 访问连续）

**模式 DOT（trans = T / C）**——每个线程负责若干输出行，逐行做点积：

```
输出行 i  ⇔  A 的第 i 列（连续段）
UPPER: 起点 colstart(i) = i(i+1)/2，   长度 i+1，对角位于段内偏移 i（末元素）
LOWER: 起点 colstart(i) = i(2n-i+1)/2，长度 n-i，对角位于段内偏移 0（首元素）

线程 t 负责 i = r0 + t, r0 + t + numThreads, ... （grid-stride）
累加顺序：先算对角项（UNIT 时为 +1·xold[i]，NON_UNIT 时为 xold[i]*AP[diag]），
          再按 Netlib 的遍历序累加非对角项（UPPER 由 i-1 递减至 0，LOWER 由 i+1 递增至 n-1），
          使浮点结合顺序与 golden 一致，把差异压到 ULP 级。
累加器：寄存器（每线程同时只推进一行，占用 2 个 float 寄存器）。
段长 > UB_STAGE_TH 时先用 DataCopy 把连续段搬入 UB 再归约，double buffer 掩盖 GM 延迟。
```

**模式 AXPY（trans = N）**——block 内按列扫描，每列贡献一段连续的 axpy：

```
块内行区间 [r0, r1)，线程 t 负责 { i ∈ [r0,r1) : (i-r0) % numThreads == t }，
累加器 accUb[i-r0]（UB，R_MAX = 1024 个 complex64 = 8 KB），每线程只写自己拥有的行，无冲突、无需原子操作。

UPPER/N：列 j 扫 [r0, n-1]，本块命中的行区间 [r0, min(j, r1-1)]
         AP 连续段 = [colstart(j) + r0, colstart(j) + i1]，对角（i == j）位于 colstart(j) + j
LOWER/N：列 j 扫 [0, r1-1]，本块命中的行区间 [max(r0, j), r1-1]
         AP 连续段 = [colstart(j) + i0, colstart(j) + r1-1]，对角位于 colstart(j) + j

每列只需 xold[j] 一个标量（广播），线程束内地址连续 → 合并访存友好。
```

两种模式下 AP 的每个元素**恰好被读取一次**，且每次读取都落在连续区间内，避免了对 packed 矩阵按行跨步访问导致的 8 字节碎片读（后者在 n=4096 时会产生数倍带宽放大）。

##### 2. 对角处理（diag = UNIT 不读对角）

- **DOT 模式**：`acc = UNIT ? xold[i] : xold[i] * AP[diagOffset]`；随后只在非对角区间内累加，**对角地址的 load 指令被编译期剔除**（`if constexpr`）。
- **AXPY 模式**：`accUb[i-r0]` 初值为 `UNIT ? xold[i] : 0`；列 j 的连续段若包含 `i == j`，则把该段**拆成 `[i0, j-1]` 与 `[j+1, i1]` 两段**分别处理，确保对角地址不被 load。
- 这样即使 AP 对角位置被填入 NaN/Inf（用例 `TC_ED_004 unit_diag_not_read`），结果也不受影响，与 golden 语义一致。

##### 3. 共轭处理

`trans = C` 时读取的 AP 元素取共轭后再参与乘加，由模板参数 `CONJ` 在编译期选择乘加式：

```
CONJ = false:  acc.r += ar*xr - ai*xi ;  acc.i += ar*xi + ai*xr
CONJ = true :  acc.r += ar*xr + ai*xi ;  acc.i += ar*xi - ai*xr
```

##### 4. incx 处理

统一为两个宏/内联函数，全部 kernel 复用：

```
X_IDX(i)  = (incx >= 0) ? (xBase + i * incx) : (xBase - i * (-incx))   // xBase = (incx<0) ? (n-1)*(-incx) : 0
          ⇔ (incx >= 0) ?  i * incx : (n-1-i) * (-incx)
```

K1 gather 用 `X_IDX` 读、连续写；K2 用 `X_IDX` 写、连续读 workspace。`|incx| ≠ 1` 时写回为跨步访问，但总量仅 n 个元素（n=4096 时 32 KB），影响可忽略。

##### 5. 精度与 Inf/NaN

- 全程 FP32，无降精度中间类型
- 乘加保持"先乘后加"两步，不启用 FMA 融合改写，Inf/NaN 传播路径与 Netlib golden 一致
- 不做 `X(J) == 0` 提前跳过（Netlib 参考实现有该优化，但会在 `x_j = 0 且 AP 含 Inf/NaN` 时产生发散）。本算子用例中，`VALUE_NORM_0` 填充的张量不含 Inf/NaN，`Inf/NaN` 填充的张量不产生精确 0，因此两种口径结果一致；选择不跳过以对齐 cuBLAS
- 累加顺序尽量与 golden 一致（见 DOT 模式说明），残余差异为浮点结合律导致的 ULP 级噪声

##### 7. 复数数据的 float 标量视图（工程实现关键点）

Ascend C SIMT 编译模型**禁止 struct 跨地址空间拷贝**：`aclblasComplex` 含非平凡构造函数，从 GM load 到局部变量、或 GM 间赋值，均会报 "no viable constructor" 编译错误。因此 kernel 侧不通过 `aclblasComplex` 结构体读写数据，而是沿用仓内 `cgeam` 等复数算子的惯用法：

- 将 `AP` / `x` / workspace 的复数指针**重解释为 `__gm__ float*`**，以实部/虚部交错存放的标量视图访问（元素 i 的实部位于 `2i`，虚部位于 `2i + 1`）；
- 复数乘加由标量四元组完成：`CtpmvMad(ar, ai, xr, xi)` 内部执行两次乘加（非共轭/共轭按 `CONJ` 模板参数选择符号），结果以 2 个 float 累加；
- Host 侧接口签名保持 `const aclblasComplex*` / `aclblasComplex*` 不变，仅在 launch 前做一次 `reinterpret_cast`，对 ABI 无影响。

该视图同时消除了结构体对齐/拷贝开销，使 AP 的连续段访存可按 float 粒度精确控制。

##### 8. 边界与退化

- `n = 0`：Host 直接返回 SUCCESS，不下发 kernel（已覆盖）
- 空块（`r0 >= r1`，n 远小于核数时可能出现）：kernel 直接 return
- `n = 1`：走路径 S，单线程完成 `x = (UNIT ? 1 : AP[0]) * x`
- 索引统一使用 `int64_t` 计算 `colstart(j)` 与 `i*incx`，避免 `j*(j+1)/2` 在大 n 下 32 位溢出

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Ascend 950DT（arch35） | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 不支持 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 不支持 |

### 算子约束限制

- 仅支持 COMPLEX64（`aclblasComplex`）；不支持双精度复数与实数 dtype
- 不支持 `incx = 0`
- 不支持非连续 Tensor（超出 incx 语义的内存访问不支持）
- 不支持广播、不支持 dynamic shape 编译期特化（n 为运行时入参）
- A 必须为方阵三角，packed 存储无 lda，不支持普通二维矩阵输入
- `diag = UNIT` 时实现不读取也不校验 AP 对角位的内容
- 不保证跨不同 block 划分的 bit-exact 结果（确定性计算不作要求）
- 依赖 `aclblasSetStream` 绑定 stream，读回 Device 结果前需由调用方同步 stream

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部/虚部分别按 FLOAT32 判定：atol = 2⁻¹⁶(1.53e-5)、rtol = 2⁻¹⁰(9.77e-4)；逐元素 `\|actual−golden\| ≤ atol + rtol·\|golden\|`；用例级 matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2（或 32×ULP） | 任务书 §3.2 + 生态算子开源精度标准 |
| 测试工程口径 | MERE = 2⁻¹³(1.22e-4)，MARE 离群倍率 10.0；实部/虚部分别统计 | ops-blas test frame 惯例 |
| 性能标准 | 4 条典型 case 平均单次耗时 ≤ 标杆：n=512/UPPER/N → 44.89 us；n=1024/LOWER/N → 88.46 us；n=2048/UPPER/T → 184.2 us；n=4096/LOWER/C → 1842.81 us | 任务书 §3.3 |
| 内存标准 | 不涉及固定上限；workspace 借用 handle 既有 workspace（默认 32 MiB），n=4096 时仅需 32 KB | 任务书 §3.4 |
| 采集口径 | 先 warmup，再有效采样 > 50 次取平均 | 任务书 §3.3 |

性能可行性估算（AP 必须整读一遍，按 AIV 聚合带宽 200 GB/s 保守估计，另加 n×8 B 的 gather/scatter 与一次额外 launch）：

| case | n | AP 数据量 | 估算耗时 | 标杆 | 余量 |
| --- | --- | --- | --- | --- | --- |
| 1 | 512 | 1.05 MB | ≈ 5 us + 2 次 launch | 44.89 us | 充裕 |
| 2 | 1024 | 4.20 MB | ≈ 21 us | 88.46 us | ≈ 4× |
| 3 | 2048 | 16.8 MB | ≈ 84 us | 184.2 us | ≈ 2.2× |
| 4 | 4096 | 67.1 MB | ≈ 336 us | 1842.81 us | ≈ 5.5× |

（n=512 走路径 S，仅 1 次 launch；其余走路径 M，额外 launch 开销约数 us。）

## 测试设计

测试工程基于 ops-blas `test/frame`（CSV 驱动 + GTest），新建 `test/tpmv/ctpmv/`，golden 由 cblas（Netlib `ctpmv` 复数实现）生成，实部/虚部分别比对。

### 用例集（1200 条，由 gen_csv.py 固定种子生成，可扩展）

| 类别 | 前缀 | 条数 | 覆盖点 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 24 | 12 组枚举全组合 × 小尺寸 (4, 8) |
| L1 尺寸 | TC_SQ | 23 | 尺寸池 1→2048（含 1、质数 3/5/7、2 的幂及 ±1、非对齐值）× 枚举轮转 |
| L2 步长 | TC_INC | 72 | incx ∈ {±1, ±2, ±3} × 12 枚举组合 |
| L5 填充 | TC_FL | 12 | 均匀随机 / 全零 / 交替 / 极端值 / Inf / NaN × AP、x |
| L5b 覆盖 | TC_CV | 96 | 中尺寸 10~400 × 12 枚举组合 |
| L6 边界 | TC_ED | 11 | n=0 quick return(SUCCESS)、diag=UNIT 对角不读、AP/x 空指针(INVALID_VALUE)、三枚举非法(INVALID_ENUM)、n<0(INVALID_VALUE)、incx=0(INVALID_VALUE) |
| EX 扩展 | TC_EX | 763 | 尺寸 × 枚举 × 步长 × 填充的确定性采样 |
| PF 性能 | TC_PF | 200 | 4 条任务书典型 case（逐参数一致）+ 小尺寸 + 对数规模扫描 + 枚举×尺寸网格 |

CSV 列格式与仓内 `stpmv_param.h` 的 ReadMap 按名解析逻辑对齐：
`case_name, description, uplo, trans, diag, n, incx, aPacked, x, random_seed, expect_result, mere_threshold, mare_multiplier`。

### 补充用例（任务提供集合之外自行补齐）

1. **packed 布局交叉验证**：固定手算小矩阵（n=1/2/3），UPPER 与 LOWER 各 6 种枚举组合，逐元素硬编码期望值——用于直接暴露任务书 LOWER 索引公式的偏差。
2. **对角位毒化**：`diag = UNIT` 时把 AP 全部对角位置填为 NaN/Inf，验证结果不受影响（证明对角未被读取）。
3. **大规模 n**：n = 4096/8192，验证 `colstart(j)` 的 64 位索引与工作量均衡划分在跨多 block 时正确。
4. **核数边界**：n 小于/等于/略大于 AIV 核数的用例，覆盖空块 early-return 分支。
5. **incx 与尾部对齐**：`incx = -1` 且 n 为奇数，验证 `xBase = (n-1)*(-incx)` 反向遍历正确。

### 自验与交付

- 精度：`python verify_accuracy.py --repo <ops-blas> --soc ascend950 --csv ./ctpmv_test.csv`（自动排除 TC_PF_ 前缀用例）
- 性能：`python verify_performance.py --repo <ops-blas> --soc ascend950`，与 `gpu_baseline.csv` 关联比对，warmup 后采样 > 50 次
- 交付：设计文档 PR 合入截图、测试用例、用例结果自测报告（含实部/虚部精度与性能截图）、测试步骤指导文档

## 兼容性分析

- **新算子**，`include/cann_ops_blas.h` 原有声明不受影响，仅新增一个函数声明，无 ABI 破坏
- 与同族 `aclblasStpmv` 逐参数对齐（仅 dtype 由 `float` 变为 `aclblasComplex`），后续 `Dtpmv`/`Ztpmv` 可直接套用同一模板与 packed 索引工具
- 未定义 950PR 私有平行 API，其他产品线可直接复用该声明
- packed 索引、incx 展开、工作量均衡划分三处工具函数下沉到 `blas/common/helper/`，供 `tpsv`、`tbmv`、`spmv` 等后续 packed 族算子复用
- 不修改现有 `blas/gbmv`、`blas/asum` 等已合入算子的任何行为

## 风险与待确认项

| # | 事项 | 影响 | 处置 |
| --- | --- | --- | --- |
| 1 | 任务书 §2.1 LOWER packed 公式 `AP[i + (2*n-j+1)*j/2]` 与 Netlib/cuBLAS 实际布局不一致（实为列首地址，缺 `-j`，n=3 即越界） | 高：按任务书实现将全量比对失败 | 已按 Netlib 源码（`KK = KK - (N-J+1)`、对角在 `AP(KK-N+J)`）确认正确式 `i + j*(2n-j-1)/2`；**实现以 Netlib 为准**，并在任务讨论帖反馈勘误 |
| 2 | golden 所用 cblas 实现若带 Netlib 的 `X(J).NE.ZERO` 跳过优化，在 `x_j = 0 且 AP = Inf/NaN` 组合下会与"不跳过"发散 | 低 | 现有用例中零填充与 Inf/NaN 填充不交叉；如实测发散，改为与 golden 一致的跳过策略 |
| 3 | `gpu_baseline.csv` 的 `gpu_ms` 列尚未回填，性能用例在回填前全部标记 NO_REF | 中 | 以任务书 §3.3 的 4 条标杆为准先行达标；回填后复测 |
| 4 | 仓库当前无 `blas/tpmv`、`test/tpmv` 目录，测试工程需从零搭建（gen_csv.py 注释中提到的 `test/tpmv/stpmv/stpmv_param.h` 在 master 上不存在） | 中 | 按 `test/axpy/caxpy`（复数）与 `test/frame` 惯例新建；CSV 列格式与 `stpmv_param.h` 的按名解析口径保持一致 |
| 5 | 路径 S 阈值 `N_SINGLE`、累加器上界 `R_MAX`、UB 分段搬入阈值 `UB_STAGE_TH` 为经验值 | 低 | 首轮取 256 / 1024 / 4 KB，在 950PR 上用 TC_PF 小尺寸与典型 case 实测后回调 |

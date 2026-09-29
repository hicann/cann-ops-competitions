# 需求背景（required）

## 需求来源

2026 年昇腾社区任务「aclblasCtpsv 算子开发（A2/A3）」。

在昇腾 NPU（Atlas A2/A3 系列产品）上使用 Ascend C 编程语言开发单精度复数
（complex64）三角压缩存储（Packed Storage）线性方程组求解算子 `aclblasCtpsv`，
与 cuBLAS `cublasCtpsv`、Netlib `ctpsv` 的核心功能与参数语义对齐，
完成设计、开发、测试全流程，验收通过后合入昇腾算子开源仓
[ops-blas](https://gitcode.com/cann/ops-blas)。

## 背景介绍

### aclblasCtpsv 在 ops-blas 中的现状

ops-blas 仓在 `blas/tpsv/arch22/` 下已有**实数**版本 `aclblasStpsv`
（`stpsv_host.cpp` / `stpsv_kernel.cpp` 等），但**没有任何复数 tpsv 实现**；
`include/cann_ops_blas.h` 中也**尚无** `aclblasCtpsv` 声明。

因此本任务不是「在已有实现上做优化」，而是**新增算子**：
需要同时新增头文件声明、host 侧实现、kernel 侧实现与完整测试工程。

与本算子同族、可作为参照的对象：

| 参照对象 | 位置 | 与本算子的关系 |
| --- | --- | --- |
| `aclblasStpsv` | `blas/tpsv/arch22/stpsv_*.cpp` | 同族实数 packed 三角求解，接口参数序列与工程骨架对齐的基准 |
| Netlib `ctpsv.f` | https://www.netlib.org/blas/ctpsv.f | 语义标杆（packed 索引、前代/回代方向、边界行为） |
| cuBLAS `cublasCtpsv` | CUDA cuBLAS | 接口标杆 |

### 现有实数版本 aclblasStpsv 的能力分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄 | scalar | - | 非空 | - |
| uplo | 三角存储模式 | attr | 枚举 | UPPER / LOWER | - |
| trans | 矩阵操作类型 | attr | 枚举 | OP_N / OP_T / OP_C | - |
| diag | 对角线类型 | attr | 枚举 | UNIT / NON_UNIT | - |
| n | 矩阵维数 | scalar | int | n ≥ 0 | - |
| AP | packed 三角矩阵 | tensor | FLOAT32 | 只读 | n(n+1)/2 |
| x | 输入/输出向量 | tensor | FLOAT32 | 原地覆写 | 1+(n-1)·\|incx\| |
| incx | 向量步长 | scalar | int | incx ≠ 0 | - |

计算公式：`op(A) · x = b`，解 x 原地覆写回 b 所在向量。

### 复数版本相对实数版本的增量

复数版不是实数版的简单换型，需要额外解决以下问题：

1. **复数乘加与复数除法**：每个元素由一次实数乘变为 4 次实数乘 + 2 次加减，
   求解每步还包含一次复数除法；实部与虚部在内存中交错排布
   （`aclblasComplex` = {float re; float im;}），无法直接套用实数版的连续向量乘。
2. **OP_C 共轭语义**：实数版 OP_T 与 OP_C 等价可折叠；复数版二者是**不同算子**，
   OP_C 需对被引用三角的元素取共轭，必须独立分支。
3. **精度风险显著加大**：复数乘的 `re1·re2 − im1·im2` 是减法，接近时发生抵消；
   复数除法与参考实现的运算次序差异为 ULP 级，且**误差随回代/前代步数（n）累积**。
4. **串行依赖链**：三角求解为递推（Recurrence）问题，x_i 依赖所有已解出的 x_j，
   递推链本身 O(n) 步不可乱序、不可沿链切分，性能设计必须在此约束下组织向量化。

### aclblasCtpsv 功能分析

**功能**：求解 `op(A) · x = b`（A 为 packed 存储的 n×n 三角复数矩阵），解 x 原地覆写。

**输入**：`handle`、`uplo`、`trans`、`diag`、`n`、`AP`、`x`（入口为 b）、`incx`

**输出**：`x`（原地，出口为解向量）

**支持数据类型**：COMPLEX64（实部/虚部各 float32）

**支持广播**：不涉及（AP/x 为独立操作数，无广播语义）

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言，以 **ops-blas handle 式 kernel 直调**方式实现
`aclblasCtpsv`，支持 COMPLEX64 数据类型，支持 UPPER/LOWER × OP_N/OP_T/OP_C ×
UNIT/NON_UNIT 全部 12 组属性组合与正负步长，精度满足生态算子开源精度标准；
性能按 kernel 级事件窗口径采集（A100 对标基线未随任务包提供，
开发期以 NO_REF 模式采集并如实披露口径，见「可维可测分析」）。

## 需求拆解

1. **接口对齐**：在 `include/cann_ops_blas.h` 新增 `aclblasCtpsv` 声明，
   与 `cublasCtpsv` 及同族 `aclblasStpsv` 逐参数对应、元素类型换 `aclblasComplex`；
   **禁止定义 A2/A3 产品私有平行接口**，须可与含 Ascend 950PR 在内的其他产品线共用。
2. **packed 三角语义**：仅引用 uplo 指定三角，无前导维 lda，
   AP 长度 n(n+1)/2；任务书 LOWER packed 索引公式存在笔误，按布局推导与
   Netlib/CBLAS 定义采信勘误版（见「数学公式」）。
3. **UNIT 对角不得读取 AP 对角位置**——这是**行为要求而非优化**：
   UNIT 用例的 AP 对角位置可能是未初始化内存或 Inf/NaN，读了就错。
4. **三种 trans 语义**：OP_N / OP_T（纯转置，不共轭）/ OP_C（共轭转置）；
   复数下 OP_T 与 OP_C 输出可分辨，**不得折叠**。
5. **步长语义**：支持正负 incx，负步长按 Netlib 语义从尾部反向遍历；incx = 0 非法。
6. **边界与负向**：n = 0 为合法 no-op；n < 0、incx = 0、非法枚举、n > 0 时空指针、
   n 超出索引安全域（> 65535），以及超出 kernel UB 能力域的档位，
   均须按校验链返回规定状态码（支持域双层语义见「算子约束限制」）。
7. **精度**：实部/虚部分别按 FLOAT32 分量判定
   （rtol 2⁻¹⁰、atol 2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_error ≤ max(1e-2, 32 ULP)）。
8. **性能**：缩短 NPU 侧 kernel 耗时；对标基线补齐前按 NO_REF 口径采集留痕，
   禁止将任务包内 H100 口径数值当作 A2/A3 验收门限。

---

# 详细设计（required）

## 算子分析

### 数学公式

求解三角线性系统：

```
op(A) · x = b,    A ∈ C^{n×n} 三角（packed 存储）,  b, x ∈ C^n
```

**op(A) 选择（trans 属性）**：

| trans | op(A) | 说明 |
| --- | --- | --- |
| OP_N | A | 原矩阵 |
| OP_T | A^T | 纯转置，**不共轭** |
| OP_C | A^H = conj(A^T) | 共轭转置，独立分支 |

**packed 存储索引（0 基，列主序，按列无间隙堆叠含对角）**：

- `uplo = UPPER`，元素 A(i,j)（i ≤ j）存于 `AP[i + j(j+1)/2]`
- `uplo = LOWER`，元素 A(i,j)（i ≥ j）存于 `AP[i + (2n-j-1)·j/2]`

> **任务书勘误**：任务书 LOWER 条目写作 `AP[i + ((2n-j+1)·j)/2]`，
> 该式把列基址项直接当作 `i+` 的加数，比正确位置整体偏大 j 个元素——
> n=3 时字面公式即越界（AP 仅 6 个元素），与 Netlib CTPSV
> 「LOWER 按列连续存储」的定义不符。正确列基址为
> Σ_{k<j}(n-k) = j(2n-j+1)/2，元素位置 = 基址 + (i-j) = `i + (2n-j-1)·j/2`。
> 本设计以 Netlib/CBLAS packed 布局为准（spec/设计/测试三方按此锁定），
> 并在提交前向任务方确认此勘误。UPPER 公式无争议。

**递推求解（i 单调推进的固定顺序遍历）**：

```
x_i = ( b_i − Σ_{j∈D_i} T[i,j]·x_j ) / T[i,i],    T = op(A)
```

| uplo | trans | 有效三角 | 求解方向 | T[i,j]（j ∈ D_i） |
| --- | --- | --- | --- | --- |
| UPPER | OP_N | 上三角 | 回代（i: n-1→0） | A[i,j]，j > i |
| UPPER | OP_T | 下三角 | 前代（i: 0→n-1） | A[j,i]，j < i |
| UPPER | OP_C | 下三角（共轭） | 前代 | conj(A[j,i])，j < i |
| LOWER | OP_N | 下三角 | 前代（i: 0→n-1） | A[i,j]，j < i |
| LOWER | OP_T | 上三角 | 回代（i: n-1→0） | A[j,i]，j > i |
| LOWER | OP_C | 上三角（共轭） | 回代 | conj(A[j,i])，j > i |

- **diag=UNIT**：T[i,i] ≡ 1+0i，对角元**不被访问**，免除复数除法；
  **diag=NON_UNIT**：T[i,i] 从 AP 读取（UPPER 对角 = 各列末元素，
  LOWER 对角 = 各列首元素），x_i 计算含复数除法。
- **不做奇异性检测**（与 cuBLAS/Netlib 一致）：NON_UNIT 时调用方保证对角非零；
  对角为零时解按 IEEE 浮点自然产生 Inf/NaN，不返回错误码。
- **值级不变量**（作为测试判据）：b = 0 且 op(A) 非奇异 ⇒ x ≡ 0；
  对输出 x 施以 op(A) 左乘，残差闭合等于 b；
  OP_C 的解 ≡ OP_T 在逐元素共轭后的 AP 上的解——AP 虚部非零时
  OP_C 与 OP_T 输出必不同（OP_C 不折叠的数学依据）。

**复数乘与复数除**：

```
(a.re + i·a.im)·(x.re + i·x.im) = (a.re·x.re − a.im·x.im) + i·(a.re·x.im + a.im·x.re)

x / d  =  x · conj(d) / |d|²        （d = 对角元，|d|² = dr² + di²）
```

**非 bit-exact**：浮点乘加/除法次序与参考实现存在 ULP 级差异，
误差随递推步数（n）累积，以容差判定（实/虚分量分别比对）。

### 支持数据类型

COMPLEX64（`aclblasComplex`，实部/虚部各 float32），
定义以 ops-blas 仓 `include/cann_ops_blas_common.h` 为准。

### 支持形状

- A：n×n 三角矩阵，packed 存储，AP 长度 n(n+1)/2
- x：n > 0 时，存储跨度至少为 1 + (n−1)·|incx| 个 complex64；n = 0 时不引用 x
- 不涉及广播；不涉及 dynamic shape（n 为运行时入参）
- n = 0 为合法 no-op

### 公共接口与参数契约

接口声明新增到公共头文件 `include/cann_ops_blas.h`，不定义 A2/A3 私有平行接口：

```cpp
aclblasStatus_t aclblasCtpsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const aclblasComplex* AP,
    aclblasComplex* x,
    int incx);
```

| 参数 | 方向/位置 | 约束与语义 |
| --- | --- | --- |
| `handle` | 输入，Host | 非空；提供算子执行 stream |
| `uplo` | 输入，Host | 仅 `ACLBLAS_UPPER` / `ACLBLAS_LOWER` |
| `trans` | 输入，Host | 仅 `ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` |
| `diag` | 输入，Host | 仅 `ACLBLAS_NON_UNIT` / `ACLBLAS_UNIT` |
| `n` | 输入，Host | `n ≥ 0` 且 `n ≤ 65535`；`n = 0` 时 no-op |
| `AP` | 输入，Device | n > 0 时非空，至少 n(n+1)/2 个 complex64；UNIT 时对角地址不得被读取 |
| `x` | 输入/输出，Device | n > 0 时非空；入口为 b，出口按 incx 逻辑顺序原地覆写为解 |
| `incx` | 输入，Host | `incx != 0`，支持正、负步长 |

逻辑元素 `x[i]` 的物理位置：`incx > 0` 时 `i·incx`，`incx < 0` 时
`(n−1−i)·|incx|`。未被该映射选中的步长间隙不属于输出，算子不得改写。

返回值：`ACLBLAS_STATUS_SUCCESS`(0) / `ACLBLAS_STATUS_INVALID_VALUE`(3) /
`ACLBLAS_STATUS_NOT_SUPPORTED`(7) / `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`(9)。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

host 侧位于 `blas/tpsv/arch22/ctpsv_host.cpp`，职责为
**参数校验 → tiling 决策 → kernel 下发**。工程范式为 ops-blas handle 式
kernel 直调（`<<<numBlocks, nullptr, stream>>>` 单 kernel 形态），
**非 aclnn 两段式**：无 op_host 注册、无两段式 workspace；
AP/x 以 Device 指针经 tiling 结构携带直接传入 kernel，
host 侧除 tiling 结构体一次 H2D 外**零数据搬运、无内部同步**。

##### 1. 参数校验（先于一切设备动作，顺序锁定）

| 顺序 | 条件 | 返回 |
| --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `n == 0` | `ACLBLAS_STATUS_SUCCESS`（no-op，不引用 AP/x，不发起 launch；先于指针校验，即 n=0 时 AP/x 允许 nullptr） |
| 4 | uplo / trans / diag 不在合法枚举 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `incx == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `n > 0` 且 AP 或 x 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `n > 65535` | `ACLBLAS_STATUS_INVALID_VALUE`（索引安全域上限，见「算子约束限制」） |
| 8 | 档位超出 kernel UB 能力域（host 侧守卫不等式判定） | `ACLBLAS_STATUS_NOT_SUPPORTED` |

n=0 的 no-op 语义对齐 Netlib `ctpsv` 的 `IF (N.EQ.0) RETURN`。

##### 2. 分核策略

本算子为 **Recurrence 范式**：递推链 x_i 依赖所有已解出的 x_j，
链本身不可乱序、不可沿链切分。因此默认形态为**串行递推链单 AIV 核闭合**：

- 每步递推的重活（trailing 更新段）压到宽向量指令，
  O(n²) 工作量收敛为 O(n) 条向量指令，单核算力余量充分；
- 核数经 `GetBlockNum()` / `GetBlockIdx()` 动态获取（禁写死），
  host 侧 `useCoreNum` 随 tiling 下发（默认 1）；
- trailing 更新段（列 axpy / 列点积）无依赖回边，理论上可多核列切分
  且不改变逐元素结果——保留为后备杠杆，默认不启用，
  是否启用由实测 profiling 数据决定（避免为不必要的跨核编排支付握手成本）。

##### 3. Tiling 决策与字段

tiling 结构为定长 POD，host 组装后经 GM 单次 H2D 传递
（per-stream 槽免同步复用 + 内容去重，同内容免重传）：

```cpp
struct TilingDataDevice {
    uint64_t apPtr;       // AP Device 地址（complex64 packed，只读）
    uint64_t xPtr;        // x Device 地址（complex64，原地覆写）
    uint32_t n;           // 阶数（n=0 不进 kernel）
    int32_t  incx;        // 步长（非零，负值合法）
    uint32_t useCoreNum;  // 实际启用核数（默认 1）
    uint32_t chunk;       // AP 列段分块宽度（复数元素数，host 决策）
    uint32_t negInc;      // |incx| 预计算值（0 表示 incx=1 快路径）
    uint32_t reserved;    // 对齐垫片
};
```

**chunk（AP 列段分块宽度）按 n 与步长自适应**，是 UB 切分的核心参数：

| 条件 | chunk 取值 | 依据 |
| --- | --- | --- |
| incx = 1 且 n ≤ 4096 | n（整列单块） | 真机 A/B 实测大 n 档更快（n=4096 约 -11%），消除逐块流水缝隙 |
| incx ≠ 1 | min(n, 2048) | 跨度装载路径下整列单块会突破 UB 预算，须分块 |
| n > 4096 | 1024 | x 平面转为分段驻留，chunk 收缩保证 UB 占用有界 |

##### 4. 数据分块和内存优化策略

DAV_2201 单 AIV UB 为 196608 B（192 KiB）。设计要点：

- **x/b 双平面全程 UB 常驻**：复数 AoS 交错数据解交织为实部/虚部两个
  float32 平面（bR/bI），n=4096 时仅 2×16 KiB = 32 KiB，远小于 UB 上限；
  递推全程在 UB 内读-改-写，x 的更新不产生任何 GM 往返，
  仅最终解按 incx 步长位置一次性写回。
- **AP 全量流式过核**：packed 布局下每一列均为 AP 内**连续稠密段**
  （UPPER 列 j 起点 j(j+1)/2、长 j+1；LOWER 列 j 起点 j(2n-j+1)/2、长 n-j），
  MTE2 整段装载无空洞；n=4096 时 AP ≈ 67.1 MB，不要求常驻，逐列段过核。
- **按模式分别核算 UB**：OP_N 形态（axpy）与 OP_T/OP_C 形态（dot）缓冲需求不同，
  分别定义总量常量并做编译期约束（`static_assert(总占用 ≤ UB_BYTES)`）；
  最重档（OP_T/OP_C dot 形态、n=4096 整列单块）约占 UB 93%，余量约 5 KiB。
  任何 tile/chunk/buffer 修改必须同步更新求和公式，杜绝依赖运行期越界现象发现。
- **host 侧 UB 能力域守卫**：host 以与 kernel 分配同式的联合不等式预判
  当前 (n, trans, incx) 档位的常驻 UB 需求，超限档位直接返回
  `NOT_SUPPORTED` 而不进 kernel——这是支持域双层语义的第二层（见约束限制）。

##### 5. 路径分派策略（kernel 直调，无 TilingKey）

ops-blas 直调范式**不使用 registry TilingKey 机制**，其对应物为
**编译期 kernel 模板实例 × 运行期 tiling 字段**：

- uplo/trans/diag 三属性由 host 侧 `kernel_do` 三层 switch 分发到
  **12 个编译期模板实例**（见 kernel 侧设计），运行期零分支；
- incx 连续/跨度路径、chunk 档位由 `TilingDataDevice` 字段在 kernel 内选择；
- 小 n 快路径（n ≤ 8）为独立的标量 kernel 入口，由 host 按 n 分发，
  与向量主路径符号隔离。

#### 3.2.2 kernel 侧设计

kernel 侧位于 `blas/tpsv/arch22/ctpsv_kernel.cpp`，
以 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 声明纯向量任务。

##### 1. 12 编译期模板（uplo 2 × trans 3 × diag 2）

| 模板 | uplo | trans | diag | 方向 | AP 访存形态 | 共轭承载 |
| --- | --- | --- | --- | --- | --- | --- |
| T1/T2 | UPPER | N | NON_UNIT/UNIT | 回代 | 右视 axpy | 无 |
| T3/T4 | UPPER | T | NON_UNIT/UNIT | 前代 | 左视 dot | 无（纯转置） |
| T5/T6 | UPPER | C | NON_UNIT/UNIT | 前代 | 左视 dot | dot 符号模式融合；对角除 conj(d) |
| T7/T8 | LOWER | N | NON_UNIT/UNIT | 前代 | 右视 axpy | 无 |
| T9/T10 | LOWER | T | NON_UNIT/UNIT | 回代 | 左视 dot | 无（纯转置） |
| T11/T12 | LOWER | C | NON_UNIT/UNIT | 回代 | 左视 dot | dot 符号模式融合；对角除 conj(d) |

**OP_T 与 OP_C 独立实例化、双侧（host/kernel）均不折叠**：
二者的访问元素集合相同，但共轭使输出可分辨（值级不变量互证）。
共轭在三处承载：trailing 更新系数、dot 符号模式（或虚部平面取反一步完成）、
对角除法的 invIm 符号位编译期翻转——均为编译期行为，运行期零开销。

**两种计算形态的选择是定案而非偏好**：

- **OP_N → 右视列 axpy**：解出 x_i 后，以 AP 第 i 列对 trailing 段的 b 做
  复数 axpy 更新（`bR −= aR·xr − aI·xi`、`bI −= aR·xi + aI·xr`，
  Counter mask 计数模式向量 Axpy，repeatTimes ≤ 255 分批）；
- **OP_T/OP_C → 左视列 dot**：x_i 由 AP 第 i 列已解段与已解 x 的
  复数点积得出。**OP_T 无法 axpy 化**——packed 布局下 A^T 的列等于 A 的行，
  跨列散取不连续，dot 是唯一能保持列连续访存的等价形态。

##### 2. Init / Process 结构

- **Init**：解析 tiling 结构（GM 指针 reinterpret）；x 装载
  （incx=1 整段 DataCopyPad / incx≠1 物理跨度装载 + Gather 偏移表
  完成逻辑 i ↔ 物理位置的映射，偏移表一次构建全程复用）；
  AoS→SoA 解交织（`vreducev2` step=1 取实部 / step=2 取虚部，
  备选回退为 Gather 偏移表解交织，已验证可用、吞吐略降）；
  计数模式 mask 的 set/reset 提升到求解循环外。
- **主循环**：沿模板编译期决定的方向逐元素递推——对角处理（UNIT 不读对角
  免除法 / NON_UNIT 对角标量提取 + 复数除法）→ x_i 写入 bR/bI 平面 →
  trailing 段宽向量计算（axpy 或 dot）。
- **写回**：求解完成后实/虚平面交织回 AoS，incx=1 整段 DataCopyPad
  UB→GM 原地覆写；incx≠1 按步长分片写回，负步长按物理偏移表落位。

##### 3. AP 列段流水（look-ahead 双缓冲）与装载优化

AP 列段采用**双缓冲预取**：当前列段的向量计算与下一列段的 MTE2 装载重叠，
`SetFlag/WaitFlag<HardEvent::MTE2_V / V_MTE2>` 事件同步，事件对的
Set/Wait 即时相邻配对；单缓冲档位在 MTE2 覆写前先排空向量单元对该缓冲的读。

在双缓冲流水之上做了两类装载合并优化（均经成对 A/B 实测与位级一致性验证）：

1. **对角/列头搭车装载**：对角元与列头对齐残段原本以独立小尺寸微拷贝
   提取（每步 1~2 次 MTE2 操作，累计占总搬运操作数近半）——改为并入
   列段首块的批量 DataCopyPad（对角/头段元素由列缓冲内标量直读），
   消除高频小搬运。装载后向量消费从对齐位置起算，算术语序与逐元素
   独立拷贝时保持一致（位级等价）。
2. **dot 尾清零向量化**：级联归约前的尾部零垫原为标量循环
   （每步约 56 次标量写）——改为单条向量 Duplicate 指令完成，
   写域/写值不变（位级等价），dot 形态大 n 档实测改善 12%~20%。

两处优化均以「未改动模板逐位不变」为约束推进，保证改动收益可归因、
无隐性回退。

##### 4. 复数除法与精度方案

- 对角复数除法采用朴素公式 `p·conj(d)/|d|²` 的标量展开
  （OP_C 模板 invIm 符号位编译期翻转）；备选 Smith 稳健算法，
  以 golden 逐模式回归结果决定是否切换（当前朴素公式全量通过）。
- 累加为固定顺序遍历（sequential），无跨核累加顺序问题；
  非 bit-exact，以容差判定。
- **多 chunk 边界的位级一致性**：分块装载路径下，同一输入重复调用的
  输出必须逐位一致（确定性要求虽为「不要求」，但位级一致是装载路径
  正确性的强判据）——设专项用例（13 个敏感形态 × 12 次重复）验证
  重复间逐位一致，全部通过。

##### 5. 小 n 快路径

n ≤ 8 时向量路径的固定开销（装载/解交织/掩码建立）占比过高，
走**独立的标量 kernel 入口**（分离符号 + host 按 n 分发），
绕过全部向量通路固定成本。小 n 档实测耗时约减半（如 n=8 档
从约 9.4 µs 降至约 5.5 µs）。

##### 6. 边界与对齐处理

- **ragged 列头**：列首 ≤7 个复数元素残段以标量处理，主段自 8 float
  对齐位置起由向量指令消费（向量指令 tensor 子偏移 8 float 对齐硬约束）；
- **packed 列非均匀基址**：LOWER/UPPER 列基址非线性步进，
  每列段 DataCopyPad 的 GM 起址/长度逐列计算（blockCount ≤ 4095、
  UB 端起址 32 B 对齐）；
- **平台行为纪律**：Counter mask 计数区内禁止发射 Level-2 Gather/vreducev2
  （互踩风险），Gather/解交织一律在 normal mask 区执行；
  `WholeReduceSum` 不用于 count>64 全和（采用 Mul + `RepeatReduceSum`
  级联，RRS 目的起址 8-float/32B 对齐）；GM 逐元素标量访问为黑名单，
  标量读写仅在 UB 侧 `LocalTensor::GetValue/SetValue`；
- **Inf/NaN**：不检测不拦截，按 IEEE 浮点沿依赖链传播（行为对齐 golden）；
- **n=1**：循环一次、无 trailing 段。

##### 7. 流程图

```
aclblasCtpsv(host)
  │
  ├─ 校验链 ①→⑧（handle / n / n=0 no-op / 枚举 / incx / 指针 / 索引域 / UB 能力域守卫）
  │      └─ 违例 → HANDLE_IS_NULLPTR / INVALID_VALUE / NOT_SUPPORTED
  │
  ├─ n ≤ 8 → 小 n 标量 kernel（独立入口）
  │
  └─ 向量主路径
       ├─ tiling 组装 + 单次 H2D（per-stream 槽去重）
       └─ kernel_do 分发 12 编译期模板，<<<numBlocks, nullptr, stream>>> 异步下发

kernel(device, AIV only, 单核闭合)
  ├─ Init：x 装载（连续/跨度+Gather）→ AoS→SoA 解交织 → bR/bI 双平面常驻
  ├─ 串行递推链（方向/形态由模板编译期决定）：
  │     对角处理（UNIT 免读 / NON_UNIT 标量复除）
  │     → x_i 写入 bR/bI
  │     → trailing 段宽向量计算（OP_N 列 axpy / OP_T·OP_C 列 dot）
  │     AP 列段双缓冲预取与计算重叠；对角/头段搭车列段批量装载
  └─ 写回：bR/bI 交织回 AoS → 按 incx（整段/分片/负步长偏移表）原地覆写 x
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas A3 系列 | √ |

- 实现目录 `blas/tpsv/arch22/`（A2/A3 共用架构目录，与已合入的
  `aclblasStpsv` 同目录新增文件，不修改既有文件）
- 测试设备：Atlas 800I A2（Ascend 910B3）；CANN 版本 9.1.0
- 接口声明放入 `include/cann_ops_blas.h`，与含 Ascend 950PR 在内的
  其他产品线共用，**未定义任何 A2/A3 私有平行接口**

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | n ≥ 0；incx ≠ 0；uplo/trans/diag 须在合法枚举内；n > 0 时 AP、x 不可为 nullptr |
| 三角引用 | 仅引用 uplo 指定三角；packed 无前导维 lda，AP 长度 n(n+1)/2 |
| 对角处理 | UNIT 时主对角假定为 1 且**不读取** AP 对角位置；NON_UNIT 时读取并做复数除法 |
| 求解有效性 | 不做奇异/近奇异检测；NON_UNIT 时调用方保证对角非零 |
| **支持域双层语义** | ① **索引安全域 n ∈ [1, 65535]**：n(n+1)/2 < 2³¹ 的 uint32 索引域上限，**n > 65535 返回 `ACLBLAS_STATUS_INVALID_VALUE`**；② **UB 能力域**：超出 kernel 常驻 UB 预算的档位（约 n > 10K，随 trans/incx 组合浮动，host 守卫不等式与 kernel 分配同式判定）**返回 `ACLBLAS_STATUS_NOT_SUPPORTED`**。边界为真机逐点扫描实测：incx=1 时 trans=N 档 n=10528 放行 / 10529 拒绝，trans=T/C 档 n=9632 放行 / 9633 拒绝；incx=2 时分别为 21056 / 19264。两层语义分离的理由：参数非法（调用方错误）与平台能力域外（合法输入但超出本实现资源域）属不同类别，返回码不应混淆 |
| 非连续 Tensor | 不支持超出 incx 步长语义的非连续内存访问 |
| broadcast | 不涉及 |
| dynamic shape | 不要求，n 为运行时入参 |
| 原地与视图 | x 原地覆写（入口 b、出口解），不返回视图 |
| 确定性计算 | 不要求。非 bit-exact，以容差判定；固定顺序遍历，多 chunk 装载路径重复调用逐位一致 |
| 空 Tensor / 0 维 | n = 0 为合法 no-op，返回成功且不引用 AP/x |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream，算子内部无同步 |

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib 复数 `ctpsv`）生成，对输出 x 全量逐元素验证（含步长位置），**实部/虚部分别**按 FLOAT32 分量判定：rtol = 2⁻¹⁰、atol = 2⁻¹⁶、required_matched_ratio = 0.99、max_abs_error_limit = max(1e-2, 32·ULP)；逐元素通过条件 `\|actual − golden\| ≤ atol + rtol·\|golden\|`，用例须同时满足 matched_ratio 与 max_abs_error 两项；测试工程双口径（mere_threshold = 2⁻¹³ / mare_multiplier = 10.0）按 CSV 列配置 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)、任务书 §3.2 |
| 性能标准 | 对标基线为 A100（判定式 `NPU ≤ A100_gpu_ms / 0.8`）。A100 逐 case 基线未随任务包提供（随包数据为 H100 口径，任务书 §3.3 标杆值同源，均不可作为 A2/A3 门限），开发期按 **NO_REF 模式**采集 NPU 耗时并完整披露口径；实测数据见下表 | 任务书 §3.3、任务包 gpu_baseline 说明 |
| 内存标准 | 任务书未规定额外内存上限。设计上无独立 workspace（仅 tiling 结构体一次 H2D）；最重档 UB 占用约 93%/192 KiB，由分路径编译期断言约束；测试工程单用例 host 侧 ≤ 512 MB 预算（packed n=4096 时 AP ≈ 67 MB） | 任务书 §3.4 |

**性能实测（kernel 级事件窗口径：warmup 20 + 有效采样 60 取平均（w20/s60），
独占卡、910B3 真机、CANN 9.1.0；任务书 §3.3 四典型 case）**：

| case | n | uplo | trans | diag | incx | NPU 实测 (µs) | 任务书标杆 (µs，H100/0.4 口径，仅量级参考) | NPU/标杆 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | LOWER | N | NON_UNIT | 1 | **250.5** | 971.85 | 0.258 |
| 2 | 1024 | UPPER | N | NON_UNIT | 1 | **524.0** | 2074.42 | 0.253 |
| 3 | 2048 | LOWER | T | NON_UNIT | 1 | **2020.8** | 4012.85 | 0.504 |
| 4 | 4096 | UPPER | C | NON_UNIT | 1 | **4852.0** | 12481.55 | 0.389 |

- 200 条性能用例全量采集：多轮独立复采有效档（n > 512）两两
  geomean 0.997~0.999（噪声带内），数据可复现；
- 小 n 档（n ≤ 32）耗时受 kernel launch 地板支配，为固有高方差档；
- 优化历程按成对 A/B 对照计：列段双缓冲流水 + chunk 自适应 +
  OP_C 符号融合阶段整体 geomean 0.843；装载合并 + 尾清零向量化 +
  小 n 快路径阶段整体 geomean 0.813；
- A100 锚点补齐后，按同协议（kernel 事件窗 w20/s60）重跑 200 条
  做正式判定 `NPU ≤ A100_gpu_ms / 0.8`。

### 精度验证方法

1. **测试框架**：ops-blas test 目录框架，CSV 描述用例 + C++ GTest 加载执行，
   golden 由 cblas 生成，落在 `test/tpsv/ctpsv/arch22/`。
2. **用例规模**：CSV **1200 条**（TC_L0 24 / TC_SQ 92 / TC_INC 24 / TC_FL 8 /
   TC_CV 96 / TC_ED 9 / TC_EX 747 / TC_PF 200），覆盖 uplo×trans×diag
   全部 12 组正交组合、n 覆盖 0/1/小质数/2 的幂及非对齐值（513/1025 等）
   至大规模、incx 覆盖 ±1/±2/±3、AP 与 x 双轨填充
   （均匀 [−5,5] + 正态混合、全零/交替/极端值/Inf/NaN）、负向用例
   （非法枚举/负 n/incx=0/nullptr）。
3. **packed 映射定向验证**：对 n=1、2、3 构造每个 AP 位置都唯一的 marker，
   覆盖 UPPER/LOWER × N/T/C，用手工展开矩阵作第二参考，
   避免 kernel 与 golden 同时采用同一错误下标公式而互相「验证通过」；
   LOWER/n=3 显式检查列序。
4. **「UNIT 不读对角」两层验证**：数值层——AP 对角位置填 NaN/Inf/独特有限
   marker，输出必须与 golden 一致（证明对角值未参与计算）；访问层——
   审查全部 AP 搬运区间，证明 UNIT 路径的 GM 访问区间排除对角地址。
5. **步长与原地验证**：|incx|>1 的间隙位置填 canary，验证只改写 n 个逻辑位置；
   覆盖 incx=±1/±2/±3、n=0、n=1 和尾部非对齐。
6. **多 chunk 位级一致性**：分块装载敏感形态（大 n 跨度装载、n>4096 收缩档、
   T/C 形态、负步长）× 12 次重复，重复间输出逐位一致（156/156 通过）。
7. **对角 boost 策略**：NON_UNIT 测试输入对被引用对角元实部加偏移
   boost = max(5, n)，golden 使用同一偏移后矩阵回代，两侧输入严格一致
   （保证测试良态，非算子行为）。
8. **极端填充 OOD 护栏**：极端值填充用例存在 golden 自身溢出的风险，
   预置 OOD 判定口径，触发时显式留痕申请裁决，**不静默过滤**。
9. **最终结果**：全量 CSV + 机制用例 **1250/1250 通过**（spec 双门
   matched_ratio / max_abs_error 同时判定，全 EXEC 零 FAIL）；
   PyTorch 侧独立对照 991/991；host 侧 UT 65/65（含支持域守卫镜像用例，
   以双翻转点锚定 UB 能力域边界）。

## 兼容性分析

**新增算子，不涉及存量行为兼容性问题。** 具体分析：

| 维度 | 分析 |
| --- | --- |
| 接口新增 | `include/cann_ops_blas.h` 中此前**无** `aclblasCtpsv` 声明，纯新增，不改动任何既有声明 |
| 跨产品线 | 声明与 `cublasCtpsv`、同族 `aclblasStpsv` 逐参数对齐，可与含 Ascend 950PR 在内的其他产品线共用；**未定义 A2/A3 私有平行 API** |
| 既有算子 | 实现落在 `blas/tpsv/arch22/` 新增文件，**不修改** `stpsv_*` 等既有文件 |
| 测试工程 | 新增 `test/tpsv/ctpsv/`，对既有测试为纯增量 |
| ABI | `aclblasComplex` 沿用 `cann_ops_blas_common.h` 既有定义，未引入新类型 |
| 状态码 | 全部复用 `cann_ops_blas_common.h` 既有枚举（SUCCESS / INVALID_VALUE / NOT_SUPPORTED / HANDLE_IS_NULLPTR），未新增状态码 |

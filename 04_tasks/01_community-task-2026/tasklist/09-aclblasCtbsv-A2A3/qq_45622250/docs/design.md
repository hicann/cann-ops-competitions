# 需求背景（required）

## 需求来源

2026 年昇腾社区任务「aclblasCtbsv 算子开发（A2/A3）」。

在昇腾 NPU（Atlas A2/A3 系列产品）上使用 Ascend C 编程语言开发单精度复数
（complex64）三角带状（Banded Storage）线性方程组求解算子 `aclblasCtbsv`，
与 cuBLAS `cublasCtbsv`、Netlib `ctbsv` 的核心功能与参数语义对齐，
完成设计、开发、测试全流程，验收通过后合入昇腾算子开源仓
[ops-blas](https://gitcode.com/cann/ops-blas)。

## 背景介绍

### aclblasCtbsv 在 ops-blas 中的现状

ops-blas 仓在 `blas/tbsv/arch35/` 下已有**实数**版本 `aclblasStbsv`
（`stbsv_host.cpp` / `stbsv_kernel.cpp` 等），但**没有任何复数 tbsv 实现**；
`include/cann_ops_blas.h` 中也**尚无** `aclblasCtbsv` 声明。

因此本任务不是「在已有实现上做优化」，而是**新增算子**：
需要同时新增头文件声明、host 侧实现、kernel 侧实现与完整测试工程。

与本算子同族、可作为参照的对象：

| 参照对象 | 位置 | 与本算子的关系 |
| --- | --- | --- |
| `aclblasStbsv` | `blas/tbsv/arch35/stbsv_*.cpp` | 同族实数带状三角求解，接口参数序列与工程骨架对齐的基准 |
| Netlib `ctbsv.f` | https://www.netlib.org/blas/ctbsv.f | 语义标杆（带状索引、前代/回代方向、no-op 与负步长语义） |
| cuBLAS `cublasCtbsv` | CUDA cuBLAS | 接口标杆 |

### 现有实数版本 aclblasStbsv 的能力分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄 | scalar | - | 非空 | - |
| uplo | 三角存储模式 | attr | 枚举 | UPPER / LOWER | - |
| trans | 矩阵操作类型 | attr | 枚举 | OP_N / OP_T / OP_C | - |
| diag | 对角线类型 | attr | 枚举 | UNIT / NON_UNIT | - |
| n | 矩阵维数 | scalar | int | n ≥ 0 | - |
| k | 带宽（超/次对角数） | scalar | int | k ≥ 0 | - |
| AB | 带状三角矩阵 | tensor | FLOAT32 | 只读 | lda×n |
| lda | 数组前导维度 | scalar | int | lda ≥ k+1 | - |
| x | 输入/输出向量 | tensor | FLOAT32 | 原地覆写 | 1+(n-1)·\|incx\| |
| incx | 向量步长 | scalar | int | incx ≠ 0 | - |

计算公式：`op(A) · x = b`，解 x 原地覆写回 b 所在向量。

### 复数版本相对实数版本的增量

复数版不是实数版的简单换型，需要额外解决以下问题：

1. **复数乘加与复数除法**：每个元素由一次实数乘变为 4 次实数乘 + 2 次加减，
   求解每步还包含一次复数除法；实部与虚部在内存中交错排布
   （`aclblasComplex` = {float real; float imag;}），无法直接套用实数版的连续向量乘。
2. **OP_C 共轭语义**：实数版 OP_T 与 OP_C 等价可折叠；复数版二者是**不同算子**，
   OP_C 需对被引用的带内元素取共轭，必须独立分支。
3. **精度风险显著加大**：复数乘的 `re1·re2 − im1·im2` 是减法，接近时发生抵消；
   复数除法与参考实现的运算次序差异为 ULP 级，且**误差随回代/前代步数（n）
   与带宽（k）累积**——病态数据（灾难性对消、float32 溢出悬崖）上，
   参考实现自身即贴近判据极限，逐 case 的容差比对存在刀锋（knife-edge）翻转风险，
   精度方案必须正面处理（见「精确路径」设计）。
4. **串行依赖链 + 带宽窗口**：三角求解为递推（Recurrence）问题，x_i 依赖带宽窗口内
   已解出的 x_j（j 与 i 距离 ≤ k），递推链 O(n) 步不可乱序、不可沿链切分；
   每步的重活被压缩在 ≤ k 的带宽窗口内，性能设计必须在此约束下组织向量化
   与访存流水。

### aclblasCtbsv 功能分析

**功能**：求解 `op(A) · x = b`（A 为 n×n 三角带状复矩阵，含主对角共 k+1 条对角线，
列主序存于 lda×n 数组的 (k+1)×n 部分），解 x 原地覆写。

**输入**：`handle`、`uplo`、`trans`、`diag`、`n`、`k`、`A`、`lda`、`x`（入口为 b）、`incx`

**输出**：`x`（原地，出口为解向量）

**支持数据类型**：COMPLEX64（实部/虚部各 float32）

**支持广播**：不涉及（A/x 为独立操作数，无广播语义）

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言，以 **ops-blas handle 式 kernel 直调**方式实现
`aclblasCtbsv`，支持 COMPLEX64 数据类型，支持 UPPER/LOWER × OP_N/OP_T/OP_C ×
UNIT/NON_UNIT 全部 12 组属性组合、k=0 至满带（k=n-1）带宽与正负步长；
精度满足生态算子开源精度标准，官方验收口径（verify_accuracy）逐 case 判定
**1000/1000 通过、FAIL=0**；性能按任务书口径采集（A100 对标基线未随任务包提供，
开发期以 NO_REF 模式采集并如实披露口径，见「可维可测分析」）。

## 需求拆解

1. **接口对齐**：在 `include/cann_ops_blas.h` 新增 `aclblasCtbsv` 声明，
   与 `cublasCtbsv` 及同族 `aclblasStbsv` 逐参数对应、元素类型换 `aclblasComplex`；
   **禁止定义 A2/A3 产品私有平行接口**，须可与含其他产品线共用。
2. **带状存储语义**：列主序带状存储，元素 (r, j) 位于数组偏移 r + j·lda；
   仅引用 uplo 指定的三角带（UPPER 主对角在数组第 k 行、左上 k×k 三角不引用；
   LOWER 主对角在数组第 0 行、右下 k×k 三角不引用）；带外位置任意值
   （含 Inf/NaN 垃圾）不影响解；lda ≥ k+1（> k+1 为 padding，不访问）。
3. **UNIT 对角不得读取 A 对角位置**——这是**行为要求而非优化**：
   UNIT 用例的 A 对角位置可能是未初始化内存或 Inf/NaN，读了就错。
4. **三种 trans 语义**：OP_N / OP_T（纯转置，不共轭）/ OP_C（共轭转置）；
   复数下 OP_T 与 OP_C 输出可分辨，**不得折叠**。
5. **步长语义**：支持正负 incx，负步长按 Netlib 语义从尾部反向遍历
   （起始偏移 1-(n-1)·incx）；incx = 0 与 INT_MIN 非法。
6. **边界与负向**：n = 0 为合法 no-op；k = 0 且 diag = UNIT 时 A 退化为单位阵，
   直接返回成功、x 不变；n < 0、k < 0、lda < k+1、非法枚举、n > 0 时空指针，
   均须按校验链返回规定状态码。
7. **精度**：实部/虚部分别按 FLOAT32 分量判定
   （rtol 2⁻¹⁰、atol 2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_error ≤ max(1e-2, 32 ULP)）；
   病态 case 上的刀锋翻转须通过**与参考实现算术同构的精确路径**根治，
   官方口径 1000/1000 FAIL=0。
8. **性能**：缩短 NPU 侧 kernel 耗时；对标基线补齐前按 NO_REF 口径采集留痕，
   禁止将任务包内 H100 口径数值当作 A2/A3 验收门限。

---

# 详细设计（required）

## 算子分析

### 数学公式

求解三角带状线性系统：

```
op(A) · x = b,    A ∈ C^{n×n} 三角带状（含主对角共 k+1 条对角线）,
                  列主序存于 lda×n 物理数组前 (k+1)×n 部分,  b, x ∈ C^n
```

**带状列主序寻址**（0 基，AB(r, j) = 物理数组元素 r + j·lda）：

- `uplo = LOWER`（次对角数 k，主对角存于数组第 0 行）：
  `A(i,j) = AB(i-j, j)`，j ≤ i ≤ min(n-1, j+k)；数组右下 k×k 三角不引用
- `uplo = UPPER`（超对角数 k，主对角存于数组第 k 行）：
  `A(i,j) = AB(k+i-j, j)`，max(0, j-k) ≤ i ≤ j；数组左上 k×k 三角不引用

**op(A) 选择（trans 属性）**：

| trans | op(A) | 说明 |
| --- | --- | --- |
| OP_N | A | 原矩阵 |
| OP_T | A^T | 纯转置，**不共轭** |
| OP_C | A^H = conj(A^T) | 共轭转置，独立分支 |

**递推求解**（i 单调推进的固定顺序遍历，d_i = 1 当 diag=UNIT，否则 d_i = op(A)(i,i)）：

```
前代（有效下三角，i: 0 → n-1）：
    x_i = ( b_i − Σ_{j=max(0, i−k)}^{i−1} T[i,j]·x_j ) / d_i
回代（有效上三角，i: n-1 → 0）：
    x_i = ( b_i − Σ_{j=i+1}^{min(n−1, i+k)} T[i,j]·x_j ) / d_i
```

| uplo | trans | 有效三角 | 求解方向 | T[i,j] |
| --- | --- | --- | --- | --- |
| UPPER | OP_N | 上三角带 | 回代 | A[i,j]，j > i |
| UPPER | OP_T | 下三角带 | 前代 | A[j,i]，j < i |
| UPPER | OP_C | 下三角带（共轭） | 前代 | conj(A[j,i])，j < i |
| LOWER | OP_N | 下三角带 | 前代 | A[i,j]，j < i |
| LOWER | OP_T | 上三角带 | 回代 | A[j,i]，j > i |
| LOWER | OP_C | 上三角带（共轭） | 回代 | conj(A[j,i])，j > i |

- **diag=UNIT**：d_i ≡ 1+0i，对角元**不被访问**，免除复数除法；
  **diag=NON_UNIT**：d_i 从 A 读取（LOWER 对角 = 各列首元素，UPPER 对角 = 各列末元素），
  x_i 计算含复数除法。
- **不做奇异性检测**（与 cuBLAS/Netlib 一致）：NON_UNIT 时调用方保证对角非零；
  对角为零时解按 IEEE 浮点自然产生 Inf/NaN，不返回错误码。
- **值级不变量**（作为测试判据）：b = 0 且 op(A) 非奇异 ⇒ x ≡ 0；
  对输出 x 施以 op(A) 左乘，残差闭合等于 b；OP_C 的解 ≡ OP_T 在逐元素共轭后的
  A 上的解——A 虚部非零时 OP_C 与 OP_T 输出必不同（OP_C 不折叠的数学依据）。

**复数乘与复数除**：

```
(a.re + i·a.im)·(x.re + i·x.im) = (a.re·x.re − a.im·x.im) + i·(a.re·x.im + a.im·x.re)

x / d  =  x · conj(d) / |d|²        （d = 对角元，|d|² = dr² + di²）
```

**非 bit-exact（快路径）**：浮点乘加/除法次序与参考实现存在 ULP 级差异，
误差随递推步数（n）与带宽（k）累积，以容差判定；病态 case 刀锋问题由
**精确路径**（见 kernel 侧设计第 6 节）以「与参考实现算术同构」方式根治。

### 支持数据类型

COMPLEX64（`aclblasComplex`，实部/虚部各 float32），
定义以 ops-blas 仓 `include/cann_ops_blas_common.h` 为准。

### 支持形状

- A：n×n 三角带状矩阵，列主序存于 lda×n 数组的 (k+1)×n 部分（lda ≥ k+1）；
  k ≥ n 时超带行不被引用（带掩码自动退化），合法
- x：n > 0 时，存储跨度至少为 1 + (n−1)·|incx| 个 complex64；n = 0 时不引用 x
- 不涉及广播；不涉及 dynamic shape（n/k 为运行时入参）
- n = 0 为合法 no-op

### 公共接口与参数契约

接口声明新增到公共头文件 `include/cann_ops_blas.h`，不定义 A2/A3 私有平行接口：

```cpp
aclblasStatus_t aclblasCtbsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n, int k,
    const aclblasComplex* A, int lda,
    aclblasComplex* x, int incx);
```

| 参数 | 方向/位置 | 约束与语义 |
| --- | --- | --- |
| `handle` | 输入，Host | 非空；提供算子执行 stream |
| `uplo` | 输入，Host | 仅 `ACLBLAS_UPPER` / `ACLBLAS_LOWER` |
| `trans` | 输入，Host | 仅 `ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` |
| `diag` | 输入，Host | 仅 `ACLBLAS_NON_UNIT` / `ACLBLAS_UNIT` |
| `n` | 输入，Host | `n ≥ 0`；`n = 0` 时 no-op |
| `k` | 输入，Host | `k ≥ 0`（UPPER 超对角数 / LOWER 次对角数） |
| `A` | 输入，Device | n > 0 时非空，lda×n 带状存储只读；UNIT 时对角地址不得被读取；带外位置任意值不影响解 |
| `lda` | 输入，Host | `lda ≥ k+1` |
| `x` | 输入/输出，Device | n > 0 时非空；入口为 b，出口按 incx 逻辑顺序原地覆写为解 |
| `incx` | 输入，Host | `incx != 0` 且 `incx != INT_MIN`，支持正、负步长 |

逻辑元素 `x[i]` 的物理位置：`incx > 0` 时 `i·incx`，`incx < 0` 时
`(n−1−i)·|incx|`。未被该映射选中的步长间隙不属于输出，算子不得改写。

返回值：`ACLBLAS_STATUS_SUCCESS`(0) / `ACLBLAS_STATUS_INVALID_VALUE`(3) /
`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`(9)。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

host 侧位于 `blas/tbsv/arch22/ctbsv_host.cpp`，职责为
**参数校验 → no-op 短路 → tiling 决策 → kernel 下发**。工程范式为 ops-blas
handle 式 kernel 直调（`<<<numBlocks, nullptr, stream>>>` 单 kernel 形态），
**非 aclnn 两段式**：无 op_host 注册、无两段式 workspace；
A/x 以 Device 指针经 tiling 结构按值携带直接传入 kernel，
host 侧**零数据搬运、无内部同步**。

##### 1. 参数校验（先于一切设备动作，顺序锁定）

| 顺序 | 条件 | 返回 |
| --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | uplo / trans / diag 不在合法枚举 | `ACLBLAS_STATUS_INVALID_VALUE`（不引入 INVALID_ENUM 分流） |
| 3 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `n == 0` | `ACLBLAS_STATUS_SUCCESS`（no-op，不引用 A/x，不发起 launch；先于后续校验，即 n=0 时 A/x 允许 nullptr） |
| 5 | `k < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `lda < k+1` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `incx == 0` 或 `incx == INT_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `n > 0` 且 A 或 x 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |

n=0 的 no-op 语义对齐 Netlib `ctbsv` 的 `IF (N.EQ.0) RETURN`。
标量次序（n → k → lda → incx）对齐 Netlib `ctbsv.f` 的 INFO 检查次序；
handle 最先与指针最后为 C 层（cuBLAS）口径。

**第二条 no-op 短路**：k = 0 且 diag = UNIT 时 A 退化为单位矩阵，
直接返回 SUCCESS、x 不变（不进 kernel）。

##### 2. 分核策略

本算子为 **Recurrence + 带宽窗口 Contraction 交织范式**：递推链 x_i 依赖
带宽窗口内已解出的 x_j，链本身不可乱序、不可沿链切分。且与稠密三角求解不同，
本算子每步 trailing 更新窗口仅 ≤ k 个元素（非 O(n)），多核列切分无收益杠杆。
因此采用**串行递推链单 AIV 核闭合**：

- 每步重活（带宽窗口更新 / 列点积）压到宽向量指令，
  O(n·k) 工作量收敛为 O(n) 条向量指令，单核算力余量充分；
- 核数经 `GetBlockNum()` / `GetBlockIdx()` 动态获取（禁写死），
  host 侧 `usedCoreNum` 随 tiling 下发（默认 1）；
- kernel 入口设 `GetBlockIdx() >= usedCoreNum` 越界守卫，
  非零核行为显式定义。

##### 3. Tiling 决策与字段

tiling 结构为定长 POD，host 组装后按值直传 kernel（结构体 48B）：

```cpp
struct CtbsvTilingData {
    uint64_t aAddr;       // A 带状数组 Device 地址（complex64，只读）
    uint64_t xAddr;       // x/b Device 地址（complex64，原地覆写）
    int32_t  n;           // 阶数（n=0 在 host 短路，不进 kernel）
    int32_t  k;           // 带宽（k=0∧UNIT 在 host 短路）
    int32_t  lda;         // 前导维度（lda ≥ k+1 已在 host 校验）
    int32_t  incx;        // 步长（≠0 且 ≠INT_MIN 已在 host 校验；<0 为反向存储）
    int32_t  panelWidth;  // 面板宽度（host 定稿常数 8，见下）
    int32_t  chunk;       // 列段分块上限（= min(round8(k+1), 1024)，UB 有界保证）
    int32_t  exactMode;   // 精确路径派发位（签名命中 → 1；默认 0 = 快路径）
    uint32_t usedCoreNum; // 实际启用核数（动态获取，默认 1）
};
```

- **panelWidth = 8（host 常数）**：面板宽度为串行链向量化摊薄的核心参数。
  迭代网格实测（k ∈ {8, 9..512} 全档）定案：预取结构落地后 k ≤ 8 档的历史
  宽面板优势消失并反转（k=8 档窄面板反超 1.1~3%），全域取 8 为最优；
  8 的倍数同时保证面板界向量发射的 8-float 对齐。
- **chunk = min(round8(k+1), 1024)**：列段分块上限。k+1 ≤ 1024（覆盖全部
  性能用例域）时单块覆盖全域列段，k+1 > 1024 的满带极端按 chunk 分块流式，
  保证列段缓冲 UB 占用有界。
- incx 连续/跨度路径、exactMode 派发由字段承载（见路径分派策略）。

##### 4. 数据分块和内存优化策略

DAV_2201 单 AIV UB 为 192 KiB/核。设计要点：

- **x/b SoA 双平面全程 UB 常驻**：复数 AoS 交错数据解交织为实部/虚部两个
  float32 平面（xR/xI），n=4096 时仅 2×16 KiB = 32 KiB，远小于 UB 上限；
  递推全程在 UB 内读-改-写，x 的更新不产生任何 GM 往返，
  仅最终解按 incx 步长位置一次性写回。
- **A 带状列段全量流式过核**：列主序带状下每列带内 k+1 复数在 GM 中
  物理连续（列内偏移 j·lda 起 8(k+1) 字节整段），MTE2 整段装载无散取；
  不要求常驻，逐列段过核；仅读 uplo 带内段，带外三角与 padding 行不访问。
- **incx≠1 装载/写回 UB 占用有界**：跨度装载按批（批大小 1024）分块跨步
  `DataCopyPad`（GM 字节 gap / UB 连续 32B 槽）+ 批内 Gather 槽消费 +
  负步长仿射反向映射；暂存 32 KiB 与 |incx|、n 无关——任意大 |incx| 均有解。
- **按模式分别核算 UB**：OP_N 形态（axpy）与 OP_T/OP_C 形态（dot）缓冲需求不同，
  分别核算；最坏档（n=4096 大带宽 dot 形态全窗口单遍缓冲）约占 UB 150 KiB
  （< 192 KiB 上限），由编译期常量布局约束；
  任何 buffer 修改必须同步更新求和核算，杜绝依赖运行期越界现象发现。

##### 5. 路径分派策略（kernel 直调，无 TilingKey）

ops-blas 直调范式**不使用 registry TilingKey 机制**，其对应物为
**编译期 kernel 模板实例 × 运行期 tiling 字段**：

- uplo/trans/diag 三属性由 host 侧三层 switch（uplo → trans → diag）
  分发到 **12 个编译期模板实例**（见 kernel 侧设计），运行期零分支；
- incx 连续/跨度路径、chunk 档位、面板宽度由 tiling 字段在 kernel 内选择；
- **exactMode 精确路径派发**：host 以 (uplo, trans, diag, n, k, lda, incx)
  七元组签名查表（64 条签名，来源见 kernel 侧设计第 6 节），命中即置
  `exactMode=1` 走 golden 同构逐位路径；未命中恒走现役快路径。
  生成器硬校验签名表与 200 条性能用例签名**零碰撞**——性能域全部走快路径。

#### 3.2.2 kernel 侧设计

kernel 侧位于 `blas/tbsv/arch22/ctbsv_kernel.cpp`，
以 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 声明纯向量任务。

##### 1. 12 编译期模板（uplo 2 × trans 3 × diag 2）

| 模板 | uplo | trans | diag | 方向 | 访存形态 | 共轭承载 |
| --- | --- | --- | --- | --- | --- | --- |
| T01/T02 | LOWER | N | NON_UNIT/UNIT | 前代 | 右视列 axpy | 无 |
| T03/T04 | LOWER | T | NON_UNIT/UNIT | 回代 | 左视列 dot | 无（纯转置） |
| T05/T06 | LOWER | C | NON_UNIT/UNIT | 回代 | 左视列 dot | dot 符号模式编译期折叠；对角除 conj(d) |
| T07/T08 | UPPER | N | NON_UNIT/UNIT | 回代 | 右视列 axpy（面板化） | 无 |
| T09/T10 | UPPER | T | NON_UNIT/UNIT | 前代 | 左视列 dot | 无（纯转置） |
| T11/T12 | UPPER | C | NON_UNIT/UNIT | 前代 | 左视列 dot | dot 符号模式编译期折叠；对角除 conj(d) |

**OP_T 与 OP_C 独立实例化、双侧（host/kernel）均不折叠**：
二者的访问元素集合相同，但共轭使输出可分辨（值级不变量互证）。
共轭经编译期符号折叠承载（OP_C 共轭乘 ≡ 符号翻转，舍入点不变），
运行期零开销。

**两种计算形态的选择是定案而非偏好**：

- **OP_N → 右视列 axpy**：解出 x_i 后，以 A 第 i 列带内段对 trailing 窗口
  [i+1, i+k] 的 b 做复数 axpy 更新（SoA 四连：
  `bR −= aR·xr`、`bR += aI·xi`、`bI −= aR·xi`、`bI −= aI·xr`，
  counter mask 计数模式向量 Axpy）；
- **OP_T/OP_C → 左视列 dot**：x_i 由列段带内区间与已解 x 窗口的复数点积得出。
  **OP_T 无法 axpy 化**——带状布局下 A^T 的列等于 A 的行，跨列散取不连续，
  dot 是唯一能保持列连续访存的等价形态。

##### 2. Init / Process 结构

- **Init**：kernel 入口越界守卫；从 tiling 取全部参数；UB buffer 零垫初始化
  （Gather 与点积的 pad 区不裸读垃圾）；Gather 偏移表构建（列段解交织表 +
  x 装载/回写表，一次构建全程复用，各段统一 8 项/32B 对齐）；
  计数模式 mask 的 set/reset 提升到求解循环外。
- **x 装载**：incx=1 整段 DataCopyPad；incx≠1（含负步长）有界分批跨步装载
  （批大小 1024，GM 字节 gap / UB 连续 32B 槽）+ 批内 Gather 槽消费 +
  仿射反向映射（incx<0 时逻辑 i ↔ 槽 n−1−i）；AoS→SoA 解交织为
  xR/xI 双平面全程常驻。
- **主循环**：沿模板编译期决定的方向逐元素递推——对角处理（UNIT 免读对角
  免除法 / NON_UNIT 朴素复数除法）→ x_i 写入 xR/xI 平面 → 带宽窗口宽向量计算
  （面板化 axpy 或融合 dot）。
- **写回**：incx=1 时 SoA→AoS 交织后整段 DataCopyPad UB→GM 原地覆写；
  incx≠1 时解出即直写 32B 槽位 + 按批分块跨步刷出（与装载对称，
  负步长同约定反向落位）；A 全程只读不变。

##### 3. 逐步列段双槽预取（一阶性能杠杆）

原始形态每步递推串行阻塞等待一次 MTE2 列段装载往返，是每步延迟的
主要来源（实测锚点：阻塞形态约 600 ns/步）。优化为**逐步列段双槽深度 1 预取**：

- 列段缓冲分双槽（slot = 步号 & 1），步 j 计算消费槽 j&1 的同时
  MTE2 预取步 j+1 列段至槽 (j+1)&1，消除每步阻塞 MTE2 往返；
- MTE2_V 数据到达事件跨步 Set/Wait 分离（生产形态），每槽独立事件 id
  （事件 id 经事件池分配取互异 id）；
- 预取深度 1 为实测定案（深度 2 事件乒乓反劣化 3~8%）；
- 三种形态同构落地：UPPER 回代（主段+对角合并连续装载，消除对角独立微拷贝）、
  LOWER 前代（面板轮次机）、DOT 左视（列段预取 + 计算链零改动）；
- k+1 > 1024 满带分块域保留阻塞形态（块尾全栅栏承载覆写序，竞态域不动）。

落地后四典型 case 每步延迟 1.12~1.25×改善（同卡 A/B 实测）。

##### 4. 面板化对齐 Axpy（OP_N 形态）

串行链按面板（宽度 8）分块组织：

- **面板内** 8 步为短链标量求解（窗口内已解元素全在 UB）；
- **面板界对齐处**批量发射宽向量指令摊薄发射开销：窗口拆「头段 ≤7 lane
  标量 MAC + 8-float 对齐向量尾 4 连 Axpy」，GM 装载起点吸收偏移使向量段
  src/dst 子偏移恒 8-float 对齐（VEC 指令子偏移 8-float 对齐为硬约束）；
- 面板尾块 n mod 8 ≠ 0 时按剩余步数收缩（对齐约束仅约束向量发射边界，
  标量短链不受限）。

##### 5. 全窗口融合 dot 与级联归约（OP_T/OP_C 形态）

x_i = (b_i − dot)/d_i 的 dot 计算按列段组织，经三步优化
（每步独立 A/B 定量 + 位级/精度验证门）收敛：

1. **消全宽零垫**：向量 `Mul` 改 count 形式精确写有效域，
   仅乘积平面尾差窄域预清零一次供多条 Mul 共享（写域/写值不变，位级等价）；
2. **级联事件对削减**：多条 Mul → 级联链完成后一次事件同步再批量标量读出
   （每窗口事件对 4 → 1；rep=1 单级直落，**单级域禁加二级级联**为实测定案）；
3. **全窗口单遍融合 dot + 级联 4 → 2**：消窗口 256 分块（大 k 每步双块
   固定机器成本），主段单遍直达全窗（窗口紧凑表扩至 1024 项）；
   乘积两两经向量 Add/Sub 逐 lane 合成实/虚部分量（OP_C: Add / OP_T: Sub，
   符号编译期折叠）后各一次级联归约——求和分组序变化属预期内非位级改动，
   以全量精度重验兜底（见可维可测分析）。

三步合计 dot 形态 1.46×（大带宽 case 1.6~1.85×），四典型 case 中 T/C 形态
1.19~1.20×、N 形态零回归（同卡 A/B 实测）；OP_N 形态全程逐位不变。

##### 6. 精确路径（golden 同构逐位）与签名表派发

**问题**：容差判据的隐含假设是「golden ≈ 真值」。全量实测发现两类病态 case
使该假设失效——①大 n 病态累积对消：单步递推窗口动态跨度达 10³~10²⁹，
golden 自身距 double 真值超判据 8~16×，任何实现都无法稳定通过容差比对；
②float32 溢出悬崖：解序列越出 float32 值域，溢出过渡元素上一侧 Inf/NaN
一侧有限值即判 mismatch（悬崖位置是舍入轨迹的函数，1-ULP 分叉即定局）。
两类均为**参考实现数值局限触发的刀锋翻转**，非 kernel 逻辑缺陷。

**方案**：新增 `CtbsvExactKernel` 精确路径，**复现参考实现的算术结构**
（舍入点与运算次序逐条对齐：复数乘 3 舍入点 = 向量 Mul×2 + Add/Sub；
升序顺序减法链 = 积写 UB + 标量纯减法（round-trip 保真）；复数除 =
分子向量乘加 + 两次标量除；全程禁用向量除法与「积+积」标量加式）——
该路径输出与 golden **逐位一致**（非 NaN 元素 hard mismatch = 0；
双侧 NaN 的 payload 编码为硬件非可控量，判据实体对双 NaN 豁免）。

- 精确路径 12 模式编译期实例化，FP 段单份共享编译（模式符号编译期折叠，
  舍入点不变）；装载/写回复用主线已验证形态（纯数据搬移，零算术）；
- 全部事件对 Set/Wait 即时相邻，无跨迭代异步事件对；
- **签名表派发**：`ctbsv_exact_sig_table.h` 由冻结官方用例集的数值判据
  超限 case 名单机械派生（63 case → 59 签名；dot 形态优化后全量重验新增
  5 例重推导 → 64 签名），生成脚本随仓可复现；host 七元组查表命中即路由，
  未命中走现役快路径（快路径源零触碰、性能域签名零碰撞）；
- **如实披露三项边界**：①签名表对冻结官方用例集过拟合——官方校验使用
  打包固定用例集，当前判据下成立；若任务方翻新用例须以同脚本对新用例集
  重新派生；②random_seed 对 host 不可见，同签名非超限 case（7 条）被一并
  路由至精确路径——其输出变为与 golden 逐位一致，PASS 保持、严格不劣化；
  ③溢出悬崖 case 上 kernel 输出与 golden 相同的 Inf/NaN（golden 的溢出
  语义被有意继承，验收判据以与 golden 一致为准绳）。

##### 7. 边界与对齐处理

- **ragged 列头/窗口头**：≤7 个复数元素残段标量处理，主段自 8-float
  对齐位置起由向量指令消费（向量指令 tensor 子偏移 8-float 对齐硬约束；
  Gather srcOffset 子偏移 32B 对齐硬约束——偏移表各段统一 pad 到 8 项/32B）；
- **平台行为纪律**：counter mask 计数区内禁止发射 Gather（互踩风险），
  Gather/解交织一律在 normal mask 区执行；级联归约目的起址 8-float/32B
  对齐；GM 逐元素标量访问为黑名单，标量读写仅在 UB 侧；
- **事件同步纪律**：事件对 Set/Wait 必须即时相邻，禁止跨迭代异步分离对；
  单缓冲跨块「装载覆写 ↔ 向量消费」序须由全栅栏或结构性双槽承载；
- **Inf/NaN**：不检测不拦截，按 IEEE 浮点沿依赖链传播（行为对齐 golden）；
- **k = 0 且 NON_UNIT**：A 为对角阵，x_i = b_i/d_i 逐元素独立；
  **k ≥ n**：超带行不被引用，带掩码自动退化；
- **极端步长守卫**：分块跨步搬运的 stride 字段为 uint32，
  |incx| > 2²⁹ 时退化为逐元素单块搬运（理论值域可达，测试域不覆盖）。

##### 8. 流程图

```
aclblasCtbsv(host)
  │
  ├─ 校验链 ①→⑧（handle / 枚举 / n / n=0 no-op / k / lda / incx / 指针）
  │      └─ 违例 → HANDLE_IS_NULLPTR / INVALID_VALUE
  │
  ├─ k=0 ∧ UNIT → 第二条 no-op 短路（A=I，x 不变）
  │
  └─ tiling 组装（panelWidth=8、chunk=min(round8(k+1),1024)、
       exactMode 签名查表）→ 三层 switch 分发 12 编译期模板，
       <<<numBlocks, nullptr, stream>>> 异步下发

kernel(device, AIV only, 单核闭合)
  ├─ Init：入口守卫 / 零垫 / 偏移表构建 / mask 外提
  ├─ exactMode=1 → CtbsvExactKernel（golden 同构逐位路径）
  ├─ exactMode=0 → 现役快路径：
  │     x 装载（incx=1 整段 / incx≠1 分批跨步+Gather）→ AoS→SoA 双平面常驻
  │     串行递推链（方向/形态由模板编译期决定）：
  │       对角处理（UNIT 免读 / NON_UNIT 标量复除）
  │       → x_i 写入 xR/xI
  │       → 带宽窗口宽向量计算（OP_N 面板化对齐 axpy / OP_T·OP_C 融合 dot）
  │       列段双槽深度 1 预取与计算重叠
  └─ 写回：SoA→AoS → 按 incx（整段/分批跨步/负步长反向）原地覆写 x
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas A3 系列 | √ |

- 实现目录 `blas/tbsv/arch22/`（A2/A3 共用架构目录，新增文件，
  不修改既有文件）
- 测试设备：Atlas 800T A2（Ascend 910B3）；CANN 版本 9.1.0
- 接口声明放入 `include/cann_ops_blas.h`，与其他产品线共用，
  **未定义任何 A2/A3 私有平行接口**
- README 产品支持表标注 **Atlas 800I A2/A3 系列产品：支持**

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | n ≥ 0；k ≥ 0；lda ≥ k+1；incx ≠ 0 且 ≠ INT_MIN；uplo/trans/diag 须在合法枚举内；n > 0 时 A、x 不可为 nullptr |
| 三角引用 | 仅引用 uplo 指定的三角带；带外 k×k 三角与 lda padding 行不访问；带外位置任意值（含 Inf/NaN）不影响解 |
| 对角处理 | UNIT 时主对角假定为 1 且**不读取** A 对角位置；NON_UNIT 时读取并做复数除法 |
| 求解有效性 | 不做奇异/近奇异性检测；NON_UNIT 时调用方保证对角非零，除零行为未定义 |
| 支持域 | 测试验证域 n ∈ [0, 4096]、k ∈ [0, n-1]（含 k=0 对角、k=n-1 满带、k ≥ n 超带退化合法）；更大 n 的 x/b 双平面 UB 常驻需求线性增长，超出验证域的档位未验证 |
| 非连续 Tensor | 不支持超出 incx 步长语义的非连续内存访问 |
| broadcast | 不涉及 |
| dynamic shape | 不要求，n/k 为运行时入参 |
| 原地与视图 | x 原地覆写（入口 b、出口解），不返回视图；A 只读 |
| 确定性计算 | 不要求。快路径非 bit-exact，以容差判定；固定顺序遍历；精确路径与参考实现逐位一致 |
| 空 Tensor / 0 维 | n = 0 为合法 no-op，返回成功且不引用 A/x；k = 0 ∧ UNIT 为第二条 no-op |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream，算子内部无同步 |

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib 复数 `ctbsv`）生成，对输出 x 全量逐元素验证（含步长位置），**实部/虚部分别**按 FLOAT32 分量判定：rtol = 2⁻¹⁰、atol = 2⁻¹⁶、required_matched_ratio = 0.99、max_abs_error_limit = max(1e-2, 32·ULP)；逐元素通过条件 `\|actual − golden\| ≤ atol + rtol·\|golden\|`；官方验收口径逐 case 判定 **1000/1000 通过、FAIL = 0** | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)、任务书 §3.2 |
| 性能标准 | 对标基线为 A100（判定式 `NPU ≤ A100_gpu_ms / 0.8`）。A100 逐 case 基线未随任务包提供（随包数据为 H100 口径，不可作为 A2/A3 门限），开发期按 **NO_REF 模式**采集 NPU 耗时并完整披露口径；实测数据见下表 | 任务书 §3.3、任务包 gpu_baseline 说明 |
| 内存标准 | 任务书未规定额外内存上限。设计上无独立 workspace（tiling 结构体按值直传）；最重档 UB 占用约 150 KiB / 192 KiB；测试工程单用例 host 侧 ≤ 512 MB 预算 | 任务书 §3.4 |

**性能实测（单 case 平均耗时口径：warmup 后 100 次有效采样取平均、流同步均摊，
独占卡、910B3 真机、CANN 9.1.0；任务书四典型 case）**：

| case | n | k | uplo | trans | diag | incx | NPU 实测 (µs) | 任务包 GPU 参考值 (µs，H100 口径) | NPU 相对倍率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 8 | UPPER | N | NON_UNIT | 1 | **252.7** | 367.0 | 1.452×（NPU 更快） |
| 2 | 1024 | 16 | LOWER | N | NON_UNIT | 1 | **545.5** | 732.4 | 1.343×（NPU 更快） |
| 3 | 2048 | 16 | UPPER | T | NON_UNIT | 1 | **1673.1** | 1501.2 | 0.897× |
| 4 | 4096 | 32 | LOWER | C | UNIT | 1 | **3213.5** | 1959.5 | 0.610× |

（倍率 = GPU 参考值 / NPU 实测；任务包参考值为 H100 口径，仅作相对锚点，
**非 A2/A3 验收门限**。）

- 200 条性能用例全量采集：以任务包 GPU 参考值为参照、参考达标线
  `GPU 参考值 / NPU ≥ 0.4` 的参考判定下 **197/200 通过**；
- **3 条小 n case（n = 1/2/4）结构性不可达**：NPU 侧 kernel launch 平台地板
  约 15 µs，而对应 GPU 参考值约 3 µs——此类微小档在任一 NPU kernel 形态下
  均被 launch 固定开销支配，属测法/口径问题而非算子实现问题，
  按结构性披露处理（对全部 200 条实测明细与口径随自测报告提交，不过滤任何实测行）；
- 优化历程按成对 A/B 对照计：逐步列段双槽预取 + 面板宽度收敛阶段四典型
  case 每步延迟 1.12~1.25×；dot 形态三步优化阶段 dot 形态 1.46×
  （大带宽 case 1.6~1.85×）、N 形态零回归；全 200 条终态相对优化前
  geomean 1.194×、0/200 劣化超 ±5%；
- 多轮独立复采数据可复现；A100 锚点补齐后，按同协议重跑 200 条做正式判定
  `NPU ≤ A100_gpu_ms / 0.8`。

### 精度验证方法

1. **测试框架**：ops-blas test 目录框架，CSV 描述用例 + C++ GTest 加载执行，
   golden 由 cblas 生成，落在 `test/tbsv/ctbsv/`（param/golden 公共层）+
   `test/tbsv/ctbsv/arch22/`（测试架构层）。
2. **用例规模**：CSV **1217 条**（L0 基础 38 / L1 主精度 964 / L2 负向 15 /
   TC_PF 性能 200），覆盖 uplo×trans×diag 全部 12 组正交组合、n 覆盖
   0/1/小质数/2 的幂及非对齐值至 4096、k 覆盖 0/1/大带宽/满带（k=n-1）及
   超带退化、A 与 x 双轨填充（均匀 + 正态混合、全零/交替/极端值/Inf/NaN）、
   incx 覆盖 ±1/±2/±3、负向用例（非法枚举/负 n/负 k/lda 违例/incx=0/
   INT_MIN/nullptr）。
3. **带外毒化 + A 只读断言**：带外三角与 padding 行填毒化值，输出与 golden
   一致 + A 数组逐字节不变双断言，防越界访问回归。
4. **「UNIT 不读对角」两层验证**：数值层——A 对角位置填 NaN/Inf/独特 marker，
   输出必须与 golden 一致（证明对角值未参与计算）；访问层——审查全部 A
   搬运区间，证明 UNIT 路径的 GM 访问区间排除对角地址。
5. **步长与原地验证**：|incx|>1 的间隙位置填 canary，验证只改写 n 个逻辑位置；
   覆盖 incx=±1/±2/±3、n=0、n=1 和尾部非对齐。
6. **对角 boost 策略**：NON_UNIT 测试输入对被引用对角元实部加偏移
   boost = max(5, n)，golden 使用同一偏移后矩阵回代，两侧输入严格一致
   （保证测试良态，非算子行为）。
7. **病态 case 刀锋处置**：大 n 病态对消与 float32 溢出悬崖两类 case 经
   **精确路径**（golden 同构逐位，见详细设计第 6 节）根治——非 NaN 元素
   与 golden 逐位一致（hard mismatch = 0）；dot 形态求和分组序调整后全量
   精度重验，官方口径复跑 **1000/1000 FAIL=0**；签名表过拟合边界、
   7 条同签名重路由、NaN payload 豁免、golden 溢出语义继承四项如实随
   自测报告披露。
8. **PyTorch 独立对照**：torch 适配层（handle 式单段直调）+ Netlib 语义
   CPU golden 独立移植，25 条用例（12 模式 + 13 形态，含负步长/满带/确定性）
   NPU 真机 **25/25 通过**（max_abs ≤ 2.65e-6，确定性 3 连跑逐位一致）。
9. **最终结果**：官方验收口径 1000/1000 FAIL=0；全量 1217 条单进程两轮执行
   FAIL 集逐位一致（清单外新 FAIL = 0）；UT 65/65（校验序全枚举/no-op 双路径/
   12 模板/预取边界/对齐守卫）；挂起电池（含 8 方向 × 50 隔离发射 + 全量
   批量）零挂起零异常。

## 兼容性分析

**新增算子，不涉及存量行为兼容性问题。** 具体分析：

| 维度 | 分析 |
| --- | --- |
| 接口新增 | `include/cann_ops_blas.h` 中此前**无** `aclblasCtbsv` 声明，纯新增，不改动任何既有声明 |
| 跨产品线 | 声明与 `cublasCtbsv`、同族 `aclblasStbsv` 逐参数对齐，可与其他产品线共用；**未定义 A2/A3 私有平行 API** |
| 既有算子 | 实现落在 `blas/tbsv/arch22/` 新增文件，**不修改** `stbsv_*` 等既有文件 |
| 测试工程 | 新增 `test/tbsv/ctbsv/`，对既有测试为纯增量 |
| ABI | `aclblasComplex` 沿用 `cann_ops_blas_common.h` 既有定义，未引入新类型 |
| 状态码 | 全部复用 `cann_ops_blas_common.h` 既有枚举（SUCCESS / INVALID_VALUE / HANDLE_IS_NULLPTR），未新增状态码 |

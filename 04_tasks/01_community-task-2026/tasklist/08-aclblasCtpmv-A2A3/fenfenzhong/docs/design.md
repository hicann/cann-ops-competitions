# 需求背景（required）

## 需求来源

2026 年 8 月昇腾社区任务「aclblasCtpmv 算子开发（A2/A3）」。

在昇腾 NPU（Atlas A2/A3 系列产品）上使用 Ascend C 编程语言开发单精度复数
（complex64）三角压缩存储矩阵-向量乘算子 `aclblasCtpmv`，与 cuBLAS `cublasCtpmv`
核心功能、参数语义完全对齐，完成设计、开发、测试全流程，验收通过后合入昇腾算子
开源仓 [ops-blas](https://gitcode.com/cann/ops-blas)。

## 背景介绍

### aclblasCtpmv 在 ops-blas 中的现状

ops-blas 仓当前在 `blas/tpmv/arch22/` 下已有**实数**版本 `aclblasStpmv`
（`stpmv_host.cpp` / `stpmv_kernel.cpp`），但**没有任何复数 tpmv 实现**；
`include/cann_ops_blas.h` 中也**尚无** `aclblasCtpmv` 声明。

因此本任务不是「在已有 TBE 版本上做优化」，而是**新增算子**：
需要同时新增头文件声明、host 侧实现、kernel 侧实现与完整测试工程。

与本算子同族、可作为参照的既有实现：

| 参照对象 | 位置 | 与本算子的关系 |
| --- | --- | --- |
| `aclblasStpmv` | `blas/tpmv/arch22/stpmv_*.cpp` | 同族实数 tpmv，接口参数序列对齐的基准 |
| `aclblasCtpsv` / `aclblasTpsv` | `blas/tpsv/` | 同为 packed 三角存储，可参照 packed 下标推导 |
| Netlib `ctpmv.f` | https://www.netlib.org/blas/ctpmv.f | 语义标杆 |
| cuBLAS `cublasCtpmv` | CUDA cuBLAS | 接口标杆 |

### 现有实数版本 aclblasStpmv 的能力分析

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

计算公式：`x := op(A) · x`

**复数版本相对实数版本的增量**，是本设计需要额外解决的部分：

1. **复数乘加**：每个元素由一次实数乘变为 **4 次实数乘 + 2 次加**，且实部与虚部
   在内存中交错排布（`aclblasComplex` = {float re; float im;}），无法直接套用
   实数版本的连续向量乘。
2. **OP_C 共轭语义**：实数版本 OP_T 与 OP_C 等价，复数版本必须区分——
   OP_C 需对 A 的元素取共轭。
3. **精度风险显著加大**：复数乘的 `re1*re2 − im1*im2` 是**减法**，
   当两项接近时发生**灾难性抵消**；实部与虚部**各自独立**抵消，
   一个可以是 1e8 而另一个塌到 1e4。
4. **溢出/下溢与特殊值中间态**：复数乘包含四个实数乘积及加减重组，
   中间项可能溢出、下溢或产生 `Inf−Inf`，即使数学结果仍可表示，也需要明确处理。

### aclblasCtpmv 功能分析

**功能**：`x := op(A) · x`（原地覆写）

**输入**：`handle`、`uplo`、`trans`、`diag`、`n`、`AP`、`x`、`incx`

**输出**：`x`（原地）

**支持数据类型**：COMPLEX64（实部/虚部各 float32）

**支持广播**：不涉及（AP/x 为独立张量，无广播语义）

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言，以 **kernel 直调**方式实现 `aclblasCtpmv`，
支持 COMPLEX64 数据类型，支持 UPPER/LOWER × OP_N/OP_T/OP_C × UNIT/NON_UNIT
全部 12 组属性组合与正负步长，精度满足生态算子开源精度标准，
性能不高于任务书给定的三个标杆耗时。

## 需求拆解

1. **接口对齐**：在 `include/cann_ops_blas.h` 新增 `aclblasCtpmv` 声明，
   与 `cublasCtpmv` 及同族 `aclblasStpmv` 逐参数对齐；
   **禁止定义 A2/A3 产品私有平行接口**，须可与含 Ascend 950PR 在内的其他产品线共用。
2. **packed 三角语义**：仅引用 uplo 指定三角，无前导维 lda，AP 长度 n(n+1)/2。
3. **UNIT 对角不得读取 AP 对角位置**——这是**行为要求而非优化**，
   UNIT 用例的 AP 对角位置可能是未初始化内存或 Inf/NaN，读了就错。
4. **三种 trans 语义**：OP_N / OP_T（不共轭）/ OP_C（共轭转置）。
5. **步长语义**：支持正负 incx，负步长按 Netlib 语义从尾部反向遍历；incx = 0 非法。
6. **边界与负向**：n = 0 为合法 no-op；n < 0、incx = 0、非法枚举、
   n > 0 时空指针，均须返回规定状态码。
7. **精度**：实部/虚部分别按 FLOAT32 标准判定
   （rtol 2⁻¹⁰、atol 2⁻¹⁶、matched_ratio ≥ 0.99、max_abs_err ≤ max(1e-2, 32 ULP)；
   此处取二者较大值作为上限，为更严格的保守解读）。
8. **性能**：三个标杆 case 的 Avg time 分别不高于 18.53 / 39.34 / 86.02 us。

---

# 详细设计（required）

## 算子分析

### 数学公式

`x := op(A) · x`，A 为 n×n 三角复数矩阵，以 packed 格式存于 AP。

**packed 下标映射（列优先紧凑打包，0-based）**：

- `uplo = UPPER`，元素 A(i,j)（i ≤ j）存于 `AP[i + j(j+1)/2]`
- `uplo = LOWER`，第 j 列起点为 `j·n − j(j−1)/2`，元素 A(i,j)（i ≥ j）存于
  `AP[j·n − j(j−1)/2 + (i−j)]`，等价于 `AP[i + j(2n−j−1)/2]`

> **任务书勘误**：任务书写成 `AP[i + j(2n−j+1)/2]`，该式比正确位置多偏移
> `j` 个元素，与 Netlib CTPMV 的“LOWER 按列连续存储”定义不符。例如 n=3 时，
> 第 1 列（0-based）的对角元素 A(1,1) 应位于 `AP[3]`，而不是 `AP[4]`。
> 本设计以 Netlib/CBLAS packed 布局为准，并在提交前向任务方确认此勘误。

**逐元素展开**（以 UPPER 为例，`aᵢⱼ = AP[...]`，UNIT 时 aᵢᵢ ≡ 1 且不读 AP）：

| trans | 公式 | 遍历方向 |
| --- | --- | --- |
| OP_N | `xᵢ := Σ_{j≥i} aᵢⱼ · xⱼ` | 列 j 向外扩散到输出 i（**scatter**） |
| OP_T | `xⱼ := Σ_{i≤j} aᵢⱼ · xᵢ` | 输出 j 收集列内元素（**gather**） |
| OP_C | `xⱼ := Σ_{i≤j} conj(aᵢⱼ) · xᵢ` | 同 OP_T，元素取共轭 |

**复数乘**（`aclblasComplex` 实虚交错）：

```
(a.re + i·a.im) · (x.re + i·x.im)
      = (a.re·x.re − a.im·x.im) + i·(a.re·x.im + a.im·x.re)
```

**原地语义的关键约束**：`x` 既是输入又是输出。除退化矩阵外，OP_N、OP_T 和 OP_C
的任一输出都可能依赖其他位置的旧 `x`；因此任何模式都不能在仍有计算读取用户 `x`
时提前覆写它。设计采用两种安全路径：

1. 仅在满足 resident-x 条件时，所有参与核先把完整逻辑向量 `x` 快照到各自 UB，
   所有核通过同一个 `SyncAll` 后，才允许原地写回互不重叠的输出区域；
2. 其他情况全部先把结果写入 workspace，compute kernel 完成后再由同一 stream 上的
   scatter kernel 写回 `x`。kernel launch 的 stream 顺序保证 scatter 不会早于 compute。

两种路径都保证计算阶段读取的是调用前的旧 `x`，而不是依赖 trans 类型假定不存在竞态。

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

| 参数 | 方向/位置 | 约束与语义 |
| --- | --- | --- |
| `handle` | 输入，Host | 非空；提供算子执行 stream |
| `uplo` | 输入，Host | 仅 `ACLBLAS_UPPER` / `ACLBLAS_LOWER` |
| `trans` | 输入，Host | 仅 `ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` |
| `diag` | 输入，Host | 仅 `ACLBLAS_NON_UNIT` / `ACLBLAS_UNIT` |
| `n` | 输入，Host | `n ≥ 0`；`n = 0` 时 no-op |
| `AP` | 输入，Device | n > 0 时非空，至少 n(n+1)/2 个 complex64；UNIT 时对角地址不得被读取 |
| `x` | 输入/输出，Device | n > 0 时非空；按 incx 表示的逻辑顺序原地覆写 |
| `incx` | 输入，Host | `incx != 0`，支持正、负步长 |

逻辑元素 `x[i]` 的物理位置为：`incx > 0` 时 `i·incx`，`incx < 0` 时
`(n−1−i)·|incx|`。未被该映射选中的步长间隙不属于输出，算子不得改写。

校验顺序固定为：handle → 三个枚举 → `n`/`incx` → n > 0 时的 AP/x 指针 →
n = 0 quick return。对应返回值依次为 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`、
`ACLBLAS_STATUS_INVALID_ENUM`、`ACLBLAS_STATUS_INVALID_VALUE` 和
`ACLBLAS_STATUS_SUCCESS`。workspace 分配失败返回 `ACLBLAS_STATUS_ALLOC_FAILED`；
Host 能同步检测到的 kernel launch 提交失败返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，异步执行错误
由 stream 同步接口向调用者报告。

所有长度和偏移均先转换为 64 位无符号数再计算，包括
`n(n+1)/2`、`1+(n−1)|incx|`、complex64 到字节数的换算。负步长绝对值按
`-(int64_t)incx` 计算，不能对 `int` 直接取负，因此 `INT_MIN` 不因实现溢出而被无条件拒绝；
仅当长度或字节偏移超出实现可寻址范围时返回 `ACLBLAS_STATUS_INVALID_VALUE`。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

host 侧位于 `blas/tpmv/arch22/ctpmv_host.cpp`，职责为
**参数校验 → tiling 决策 → workspace 准备 → kernel 直调**。

##### 1. 参数校验（先于一切设备动作）

按任务书 §2.5 逐项校验，顺序如下：

| 顺序 | 条件 | 返回 |
| --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | uplo / trans / diag 不在合法枚举 | `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | `n < 0` 或 `incx == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `n > 0` 且 AP 或 x 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `n == 0` | `ACLBLAS_STATUS_SUCCESS`（no-op，不引用 AP/x，不发起 launch） |

**枚举先于数值校验**是有意的：一个同时非法枚举 + 负 n 的调用应报枚举错，
与 cuBLAS 行为一致。

##### 2. 分核策略

本算子的分核不能只按输出元素数均分：不同输出包含的三角元素数不同，且
OP_N 列 sweep 还有固定的逐列启动成本。设计中区分两种粒度：

- **正确性对齐粒度**：4 个 complex64 = 32 B。不同核直接写 GM 时，输出边界必须按
  此粒度对齐，避免两个核写同一 32 B 块。
- **性能 lane 粒度**：32 个 complex64。OP_N 向量路径的输出窗口以整 lane 为佳，
  但它不是所有模式的强制正确性粒度。

Host 首先读取物理 AIV 数 `aivCoreNum`，计算基础输出核数：

```text
alignedSlices = ceil(n / 4)
baseCoreNum   = max(1, min(aivCoreNum, alignedSlices))

if trans == OP_N:
    laneSlices = floor(n / 32)
    if laneSlices >= 2 and baseCoreNum > 2 * laneSlices:
        baseCoreNum = laneSlices
    else if uplo == LOWER and laneSlices >= 2 and baseCoreNum > laneSlices:
        baseCoreNum = laneSlices
```

OP_N 还根据 32-complex lane 调整核数，避免大量短于一个 lane 的输出片；最终核数不得
超过 `aivCoreNum`。各模式的输出边界如下：

| 模式 | 输出切分依据 | 边界约束 |
| --- | --- | --- |
| OP_N/UPPER | 默认按输出数量均分 | 直接写回时按 4 complex 对齐；向量 sweep 优先 32 complex 窗口 |
| OP_N/LOWER | 按“逐列固定成本 + 有效 lane 数”模型切分 | 同上 |
| OP_T/OP_C | 按三角元素数，并计入 AP 分块搬运次数 | workspace 输出切片按 4 complex 对齐 |

对第 k 个核，基础边界由累计代价达到 `totalCost·k/baseCoreNum` 的最小输出下标确定，
再向下对齐到 4 complex，并保持边界单调。三角元素累计代价采用 64 位闭式计算：递增工作量
`prefix(k)=k(k+1)/2`，递减工作量 `prefix(k)=k·n−k(k−1)/2`；OP_T/OP_C 的代价还加入
`ceil(columnLength/AP_TILE_COMPLEX)` 次 AP 搬运的权重。对齐后没有输出的核可退出，但凡路径
包含 `SyncAll`，空核只能在完成对应屏障后退出。

若基础输出切分未用满 AIV，再考虑以下二维切分；不减少已有输出核数去换列组：

**(a) 规则列分组（`colGroups > 1`）**

仅用于 resident-x 的 OP_N 路径。`colGroups` 取不超过
`floor(aivCoreNum / baseCoreNum)` 且不超过可用列 lane 数的 2 的幂。不同列组处理同一输出
区间的不同列范围，各自产生 partial；全部 worker 到达 `SyncAll` 后，由约定的归约核合并。

**(b) 不规则三角划分（ragged，仅 LOWER/OP_N）**

初始 `rowLen = 32`；若 `ceil(n/rowLen)` 超过可发射任务数，则将 `rowLen` 逐次加倍，
直到每个行组至少可分配一个 worker。对行组 `[r0,r1)`，LOWER/OP_N 中列 j 的有效工作量为：

```text
cost(j; r0, r1) = max(0, r1 - max(r0, j))
groupCost       = Σ cost(j; r0, r1), 0 ≤ j < r1
```

先按 `groupCost` 的最大余数法分配 worker，再按上述 cost 的前缀和切分列区间。
每个 task 记录 `rowStart`、`rowLen`、`colStart`、`colEnd`、`partialOffset`、
`leaderBlock` 和 `workerCount`；不产生列范围无法到达对应行区间的空任务。

所有包含 `SyncAll` 的路径必须满足：发射 block 数 `≤ min(aivCoreNum, CTPMV_MAX_TASKS)`，且每个
已发射 block 在任意分支上到达相同次数的屏障；空输出核也不得在屏障前提前返回。

UPPER 不启用 ragged：其默认输出切分已限制关键核，新增 partial 归约的收益不足以覆盖成本。

##### 3. Tiling 决策与字段

Tiling 决策顺序固定如下，避免同一组标志组合出现多种解释：

1. 根据 `n` 与设备 AIV 数选择 `baseCoreNum`；
2. 判断 resident-x 资格：`incx == 1 && 128 ≤ n ≤ 4096 && trans == OP_N`；
3. 对 LOWER/OP_N 优先尝试 ragged；否则在有空闲核时计算 `colGroups`；
4. 由 resident-x、ragged 和 colGroups 唯一确定输出路径及 workspace 大小；
5. 将最终发射核数与全部 task 描述写入 `CtpmvTilingData`。

| 字段 | 含义 | 合法性约束 |
| --- | --- | --- |
| `n/uplo/trans/diag/incx` | 公共接口参数的设备侧副本 | 已通过 Host 校验 |
| `useCoreNum` | 实际发射 AIV block 数 | `1 ≤ useCoreNum ≤ aivCoreNum` |
| `fused` | resident-x 资格标志，不等价于“workspace 必为 0” | 仅 OP_N、incx=1、128≤n≤4096 可置 1 |
| `colGroups` | 同一输出区间的列分组数 | ≥1；大于 1 时需要 partial workspace 和核内合并 |
| `ragged` | 是否采用 LOWER/OP_N 不规则二维划分 | 仅 resident-x 的 LOWER/OP_N 可置 1 |
| `taskCount/tasks[]` | ragged 任务数与任务描述 | `taskCount == useCoreNum ≤ MAX_TASKS`；行列覆盖无重叠、无遗漏 |

边界策略：n < 128、`incx != 1`、n > 4096 或 OP_T/OP_C 均进入 staged 路径；
尾部不足 4 complex 时由单一核负责，并通过 `DataCopyPad`/标量尾处理保证不越界；
所有 GM 索引、partial 偏移和 workspace 字节数使用 64 位计算并在 launch 前检查溢出。

##### 4. 数据分块与 UB 内存策略

DAV_2201 的单 AIV UB 为 196608 B（192 KiB）。不同计算路径申请的缓冲不同，不能把
互斥路径的缓冲混在同一张表中累加。以下均以 float 为计量单位，1 float = 4 B，
1 complex64 = 2 float = 8 B。

**OP_N 向量路径（最坏情况：`colGroups > 1`）**：

| 缓冲 | 元素数 | 字节数 | 用途 |
| --- | --- | --- | --- |
| `outBuf` | `1024 × 2 × 2 = 4096 float` | 16 KiB | partial 合并时的两份输出 tile |
| `apBuf` | `4096 × 2 = 8192 float` | 32 KiB | AP slab |
| `xBuf` | `(4096+32) × 2 = 8256 float` | 32.25 KiB | 常驻逻辑 x 与尾部 padding |
| `axpyBuf` | `28224 float` | 110.25 KiB | 两路累加器、peak/index、slab 产品和广播区；各子区按生命周期复用 |
| `accBuf` | `64 × 3 + 8 = 200 float` | 0.78125 KiB | Re/Im/peak 归约及结果槽 |
| **合计** | `48968 float` | **191.28125 KiB** | 剩余 736 B |

`colGroups == 1` 时 `outBuf` 只有一份，合计 183.28125 KiB。

**OP_T/OP_C 向量路径**：

| 缓冲 | 元素数 | 字节数 | 用途 |
| --- | --- | --- | --- |
| `outBuf` | `1024 × 2 = 2048 float` | 8 KiB | 输出 tile |
| `apBuf/apBufB` | `2 × 4096 float` | 32 KiB | AP 双缓冲 |
| `xBuf/xSwapRes` | `2 × 8256 float` | 64.5 KiB | 常驻 x 及实虚交换副本 |
| `swapBuf` | `4096 float` | 16 KiB | 非对齐窗口交换区 |
| `prodBuf` | `8192 float` | 32 KiB | 两路交错乘积 |
| `twoProdBuf` | `4096 float` | 16 KiB | 幅值/临时计算区 |
| `accBuf` | `200 float` | 0.78125 KiB | Re/Im/peak 归约及结果槽 |
| **合计** | `43336 float` | **169.28125 KiB** | 剩余 22.71875 KiB |

scatter kernel 仅需一份 `1024 × 2 float = 8 KiB` 的搬运缓冲。非向量计算路径同样只申请
自身实际使用的缓冲，不预留上述向量临时区。

UB 超限不应依赖运行时现象发现。为 OP_N 和 OP_T/OP_C 分别定义总量常量并做编译期约束：

```cpp
static_assert(OPN_ARENA_FLOATS * sizeof(float) <= UB_BYTES,
              "OP_N UB arena exceeds DAV_2201 UB");
static_assert(TRANSPOSE_ARENA_FLOATS * sizeof(float) <= UB_BYTES,
              "OP_T/OP_C UB arena exceeds DAV_2201 UB");
```

OP_N 最坏路径占用 **191.28125 KiB / 192 KiB**，是容量绑定项。任何 tile、lane、
column-group 或 buffer 复用关系的修改都必须同步更新上述求和公式，并执行 UB 越界负向构建测试。

##### 5. tilingkey 规划策略

本算子**不使用 tilingkey 分支**，改为**两个计算 kernel 入口、一个 scatter 入口，
配合 tiling 结构体内的标志位**：

| 入口 | 覆盖 |
| --- | --- |
| `ctpmv_kernel_n` | OP_N（`CtpmvBody<false>`） |
| `ctpmv_kernel_t` | OP_T / OP_C（`CtpmvBody<true>`，共轭由 tiling 内 trans 区分） |
| `ctpmv_scatter_kernel` | 非 fused 路径下把 packed 结果按 incx 散回用户 x |

理由：OP_N 与 OP_T 的数据流方向**相反**（scatter vs gather），
UB 布局、分核方式、归约结构均不同，合成一个 kernel 会让两条路互相拖累寄存器分配。
分成两个 `__global__` 入口后，`ctpmv_kernel_n` 的代码生成完全不受 OP_T 改动影响——
这一点在开发中被实测证实过（见 §「性能设计要点」第 3 条）。

其余分支由 `CtpmvTilingData` 的字段在运行期选择：
`fused`、`colGroups`、`ragged`、`taskCount`、`tasks[]`。其中 `fused` 只表示已经满足
resident-x/单 launch 的前置条件；是否为零 workspace 还必须同时检查
`colGroups == 1 && ragged == 0`。

##### 6. 执行路径与 workspace 策略

| 路径 | 条件 | x 的读取来源 | 中间结果 | 同步与写回 |
| --- | --- | --- | --- | --- |
| direct fused | `fused=1 && colGroups=1 && ragged=0` | 每核完整 UB 快照 | 无 workspace | 所有核快照后 `SyncAll`，各核写互不重叠的 x 区间 |
| grouped single-launch | `fused=1 && colGroups>1` | 每核完整 UB 快照 | 每列组一张对齐 partial 平面 | worker 写 partial 后 `SyncAll`，归约核合并并写 x |
| ragged single-launch | `fused=1 && ragged=1` | 每核完整 UB 快照 | `Σ tasks[t].rowLen` 个 complex64 的紧凑 partial | task 写 partial 后 `SyncAll`，组长合并并写 x |
| staged scatter | `fused=0`，包括 OP_T/OP_C、步长非 1、小/大 shape | GM 或 UB 快照，但计算期间不写 x | `roundUp(n,4)` 个 complex64 | compute 完成后，同一 stream 上启动单 block scatter 写回 x |

ragged 的 partial 采用**紧凑**布局而非规则网格，是因为规则网格要写
`taskCount × n`，其中绝大部分是零。所有 workspace 大小先在 64 位中计算并检查
乘加溢出；分配失败不得 launch kernel。各路径所需字节数为：

```text
direct fused:           0
grouped single-launch:  colGroups × roundUp(n, 4) × sizeof(complex64)
ragged single-launch:   (Σ tasks[t].rowLen) × sizeof(complex64)
staged scatter:         roundUp(n, 4) × sizeof(complex64)
```

#### 3.2.2 kernel 侧设计

kernel 侧位于 `blas/tpmv/arch22/ctpmv_kernel.cpp`，
两个入口共用模板化的 `CtpmvBody<isTranspose>`。
均以 `KERNEL_TYPE_AIV_ONLY` 声明——本算子是纯 vector 负载，不使用 cube。

##### 1. 原地覆写的正确性：快照 + 屏障

如 §「数学公式」所述，三种 trans 都必须基于调用前的完整旧 `x` 计算。设计提供两种
互斥的正确性机制。

**direct fused 路径**仅用于 `CanFuse` 成立且不需要 partial 的 OP_N：

```
① 每个核把整个 x 快照进自己的 UB（xTile 常驻）
② SyncAll  —— 核间屏障，保证所有核都已读完 x 的旧值
③ 每个核计算并把自己那片互不相交的 x 写回 GM（原地）
```

`CanFuse` 给出 resident-x 资格：

```cpp
incx == 1 && n >= VECTOR_MIN_COMPLEX(128) && n <= X_RESIDENT_MAX(4096)
          && trans == ACLBLAS_OP_N
```

其中 `fused=1` 只说明每个核可以持有完整旧 `x`；若 `colGroups>1` 或 `ragged=1`，
仍需 partial workspace 和一次核内 `SyncAll`，随后由归约核写回。所有参与 block 必须经过
相同的屏障，空任务也不能提前返回。

**staged scatter 路径**用于 OP_T/OP_C、`incx != 1`、n < 128 或 n > 4096：compute
kernel 在整个执行期间只读取旧 `x`，结果写入连续 workspace；compute 返回后，在相同 stream
上启动单 block `ctpmv_scatter_kernel` 写回用户 `x`。这里不需要 compute kernel 内的 x 快照
屏障，因为在 compute 完成前没有任何路径修改 `x`。

当前只让 OP_N 进入 resident-x 路径是性能选择，不是因为 OP_T/OP_C 天生没有原地依赖。
OP_T/OP_C 同样存在“一个输出提前覆盖另一个输出仍需读取的旧 x”问题，故保持 staged 路径。

因此三个性能 case 走两条不同的路：

| case | trans | fused | 路径 |
| --- | --- | --- | --- |
| 1（n=512 UPPER/N） | OP_N | 是 | 单 launch，原地写 x，**无 workspace** |
| 2（n=1024 LOWER/N） | OP_N | 是 | resident-x + ragged，使用紧凑 partial workspace |
| 3（n=2048 UPPER/T） | OP_T | 否 | compute → workspace → `ctpmv_scatter_kernel` 散回 |

不满足 fused 条件时（OP_T/OP_C、大 n 或非 1 步长）走
「compute → workspace → `ctpmv_scatter_kernel` 散回」两段式，
scatter kernel 只由 block 0 执行：目标是用户的 x，一般 incx 下切片不按块对齐，
多核标量 GM 写可能落在同一 32 B 块。Host 即使按 `useCoreNum` 发射，其他 block 也必须
在入口立即返回，因此文档语义上称其为“单 block scatter”。

##### 2. Init / Process 三阶段

计算 kernel 沿用 CopyIn → Compute → CopyOut 的结构：

- **CopyIn**：使用 `DataCopyPad` 把 AP 分块从 GM 搬进 UB。OP_T/OP_C 使用
  `apTile` / `apTileB` 双缓冲，以 `SetFlag/WaitFlag<HardEvent::MTE2_V>` 让下一输出列的
  搬运与当前列计算重叠；OP_N 使用单个较大的 AP slab，通过分组搬运减少逐列同步开销。
- **Compute**：复数乘加，见下。
- **CopyOut**：direct fused 将互不重叠、32 B 对齐的输出块直接写 x；grouped/ragged
  先写 partial，再在屏障后合并；staged 路径写 workspace，最后由单 block scatter 写 x。

非 32 B 整块的输入尾部使用 `DataCopyPad`，输出尾部只写逻辑有效元素，不能覆盖用户在
`x` 步长间隙中的数据。所有 MTE2→Vector、Vector/Scalar→MTE3 依赖均由匹配的 event 或
barrier 建立，双缓冲的两个在途槽必须使用彼此独立的 event id。

##### 3. 复数乘加的向量化

`aclblasComplex` 实虚交错，直接做向量乘得不到复数乘。采用**交换副本法**：

```
product  = ap × xPad      -> 得到 (a.re·x.re, a.im·x.im) 交错
productB = ap × xSwap     -> 得到 (a.re·x.im, a.im·x.re) 交错
```

`xSwap` 是把 x 相邻两个 float 对调后的副本（`xSwapRes` 全程常驻，一次建表）。
随后按交错位置折叠即得实部 `re·re − im·im` 与虚部 `re·im + im·re`。
`BinaryRepeatParams` 的 block stride 置零实现**广播**，
避免为每列复制一份 x。

OP_C 的共轭只是对 `a.im` 取反，在同一路乘法里用符号掩码完成，
**不增加一次乘法**。

##### 4. 精度设计：这是本算子的主要难点

复数乘的实部是 `re·re − im·im`，两项接近时发生灾难性抵消，
且实部与虚部可能独立抵消。设计采用“快速路径 + 可验证守卫 + 参考语义回退”，
但不把经验阈值表述成对所有输入的解析误差证明。

**(a) 向量快速路径**

快速路径使用普通 FP32 分道乘加和树形归约。树形归约缩短依赖链，但会改变与顺序累加相比的
舍入次序；它不承诺 bit-exact。Kahan 分道累加曾作为候选方案评估，但不是本设计的最终
向量路径，不能将其作为当前精度保证。

**(b) 快速路径守卫**

每个输出计算 `max(|AP.re|,|AP.im|) × max(|x.re|,|x.im|)` 的保守上界，并检查向量结果
及该上界是否为有限值。若上界大于经验阈值 `VECTOR_TERM_LIMIT = 1024.0f`，或任一检查值
为 Inf/NaN，则该输出回退到标量精确乘法路径。该阈值的地位是基于目标精度标准和覆盖用例
确定的经验门限，必须由阈值上下边界、强抵消和随机压力测试持续验证；单凭“最大项≤1024”
不能形式化保证最终结果必在 32 ULP 内。

> 这个守卫在开发中造成过一次典型陷阱：任何让结果变坏的改动，
> 都会把测试**静默改道**到标量路径，于是测试照样通过、性能却崩塌。
> 因此每个性能用例都必须同时打印 `ctx_guard_path`，
> 确认它确实走的是 `vector (as intended)` 而不是被守卫救回来的标量路径。

**(c) 回退复数乘与特殊值语义**

回退路径以 Dekker 分裂实现误差补偿的复数乘，并用 `#pragma clang fp contract(off)` 控制
浮点收缩。有限大数在需要时通过指数分析判断乘积是否可能上溢/下溢，再成对缩放后计算并
恢复量级；不得仅凭“单个操作数达到 2^126”判断乘积风险，因为两个较小操作数的乘积也可能
溢出。对下溢必须明确采用 gradual underflow/平台 FP 模式，并由接近最小正规数和次正规数的
用例验证。

Inf/NaN 的验收语义以任务书指定的 cblas/Netlib golden 为准：实部、虚部不仅比较有限数值，
还要比较 `finite/+Inf/-Inf/NaN` 分类；除非任务方书面确认其他语义，不允许因“特殊值输入”
跳过类别不一致。若保留 C99 Annex G 风格的复数恢复逻辑，必须先证明其输出类别与验收 golden
一致，否则应改为与 golden 相同的运算顺序和特殊值规则。

##### 5. OP_T/OP_C 的分道归约

OP_T/OP_C 将两路乘积及幅值上界分别保存在 Re、Im、peak lane 中。每个 AP tile 先做
树形折叠，多 tile 输出再合并到常驻累加器；最终通过整 lane reduction 得到四个乘积半和与
一个 peak，将实部重组为“偶位和 − 奇位和”，虚部重组为“偶位和 + 奇位和”。若守卫拒绝，
只对当前输出执行标量回退。这样避免逐项从 UB 读回，同时保持每个输出独立判定。

##### 6. OP_N 的列 sweep 分组

OP_N 按 packed 连续列构造 slab。每个 slab 内先批量发起各列 AP 切片搬运，再以 Gather
构造 x 的实部/虚部广播，形成两路乘积并树形折叠到当前输出 lane。这样把逐列的同步和指令
启动成本摊到一组列上；它与 OP_T/OP_C 的双 AP buffer 预取是两种不同的流水结构。

```cpp
SyncVToMte2();
for (column in group) {
    CopyColumnSliceToSlab(...);
}
SyncMte2ToV();
BroadcastXAndMultiply(...);
FoldSlabIntoOutputLane(...);
```

slab 槽位必须覆盖每列切片的完整对齐后长度，未被当前列覆盖的位置先清零。UNIT 模式下，
任何包含对角位置的 AP 区间都拆成对角前、对角后两个子区间分别搬运，对角地址不出现在
`DataCopyPad` 的源范围内；随后在 UB 对应位置显式写入 `(1,0)`。OP_T/OP_C 同样只搬运
off-diagonal run，NON_UNIT 才单独读取 diagonal。group 大小由
`min(COPY_GROUP_MAX, floor(AXPY_TILE_FLOATS/outPadded))` 给出，尾组按剩余列数缩短。

##### 7. 溢出/下溢的安全重算

有限输入先用指数或等价的无溢出比较判断四个实数乘积是否可能越过 FP32 正常范围。
存在上溢风险时，将一个大操作数缩小 2⁻⁶⁴，完成补偿复数乘后再把结果放大 2⁶⁴；
存在下溢风险且数学结果仍应保留时，采用相反方向的缩放，并在计算后恢复量级。恢复量级本身
也可能上溢或下溢，此时结果应按 cblas golden 的相同运算顺序得到对应 IEEE 值，不能无条件
声称“最终有限”。极小有限数和次正规数另设边界用例，验证缩放不会把本应保留的非零结果
提前冲零。

Inf/NaN 输入进入显式特殊值分支，但其最终 IEEE 分类仍必须与 cblas golden 对齐，详见
§「精度验证方法」。

##### 8. 流程图

```
aclblasCtpmv(host)
  │
  ├─ 校验：handle / 枚举 / n / incx / 指针      ── 失败 → 状态码返回
  ├─ n == 0                                    ── → SUCCESS（no-op）
  │
  ├─ tiling：useCoreNum / fused / colGroups / ragged
  │
  ├─ fused && colGroups==1 && !ragged
  │     └─ ctpmv_kernel_n：快照 x → SyncAll → 原地写 x，无 workspace
  │
  ├─ ragged（LOWER/OP_N）
  │     └─ EnsureWorkspace(紧凑 partial)
  │        ctpmv_kernel_n：快照 x → 各任务算 partial → SyncAll → 组长归约并写 x
  │
  ├─ colGroups>1（OP_N）
  │     └─ EnsureWorkspace(每列组一个 partial 平面)
  │        ctpmv_kernel_n：快照 x → worker 写 partial → SyncAll → 归约核写 x
  │
  └─ 其他
        └─ EnsureWorkspace(roundUp(n,4))
           ctpmv_kernel_n|t：读取旧 x，结果只写 workspace
           同一 stream 上单 block ctpmv_scatter_kernel 按 incx 散回 x

kernel(device, AIV only)
  resident-x 路径：快照 x → 必要的 SyncAll → AP 分块/分组 → 复数乘加与守卫 → 写 x/partial
  staged 路径：读取旧 x → AP 分块 → 复数乘加与守卫 → 写 workspace → scatter 写 x
```

### 性能设计要点

三个标杆 case 的达成依赖以下几项，均由消融或配对 A/B 实测支撑：

1. **AP 搬运与计算重叠**：关掉预取（no-prefetch 对照）让 Case 3 劣化 **+9.313 us**，
   说明重叠确实在起作用。
2. **OP_T/OP_C 分道树形归约**：见 §3.2.2-5；把逐项标量读回改为 UB 内归约，
   显著降低 Vector→Scalar 交互开销。
3. **补偿乘的寄存器形态**：`SoftwareFma` 的 20 个临时量原为 `volatile`，
   每个强制一次存-取往返。去掉后 Case 3 **−9.4 us**。
   但**必须按调用点限定**：直接全局去掉会让**根本不执行该函数**的 Case 1
   劣化 +0.389 us（其 kernel text 反而缩小了 984 字节，排除布局效应）。
   最终做法是给 `SoftwareFma`/`Multiply` 加 `registerForm` 模板参数，
   只有消融指认的那一个调用点传 `true`；
   **判据是 `ctpmv_kernel_n` 必须逐字节回到 30212**（实测正是）。
4. **不做的事**：`noinline` 实测两边都更差；
   把 UPPER 对角元从 GM 搬进 UB 因**向量操作数 32 字节对齐**要求而不可行
   （`apTile[count*2]` 四行里只有一行对齐，实测 26 个 reduce 用例失败），
   已记录为架构性否决。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas A3 系列 | √ |

- 实现目录 `blas/tpmv/arch22/`（A2/A3 共用架构目录）
- 性能测试设备：Atlas 800I A2（910B3）；开发验证设备 Ascend910B4
- CANN 版本：9.1.0
- 接口声明放入 `include/cann_ops_blas.h`，与含 Ascend 950PR 在内的其他产品线共用，
  **未定义任何 A2/A3 私有平行接口**

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | n ≥ 0；incx ≠ 0；uplo/trans/diag 须在合法枚举内；n > 0 时 AP、x 不可为 nullptr |
| 三角引用 | 仅引用 uplo 指定三角；packed 无前导维 lda，AP 长度 n(n+1)/2 |
| 对角处理 | UNIT 时主对角假定为 1 且**不读取** AP 对角位置；NON_UNIT 时读取 |
| 非连续 Tensor | 不支持超出 incx 语义的非连续内存访问 |
| broadcast | 不涉及 |
| dynamic shape | 不要求，n 为运行时入参 |
| 原地与视图 | x 原地覆写，不返回视图 |
| 确定性计算 | **不要求**。本算子为浮点乘加，非 bit-exact；且分核数随 n 变化会改变求和顺序 |
| 空 Tensor / 0 维 | n = 0 为合法 no-op，返回成功且不引用 AP/x |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib `ctpmv`）单标杆生成，输出 x 全量验证（含步长位置），**实部/虚部分别**按 FLOAT32 判定：rtol = 2⁻¹⁰、atol = 2⁻¹⁶、required_matched_ratio = 0.99、max_abs_error_limit = max(1e-2, 32·ULP)。逐元素通过条件 `\|actual − golden\| ≤ atol + rtol·\|golden\|`；用例须同时满足 matched_ratio 与 max_abs_error 两项 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)、任务书 §3.2 |
| 性能标准 | Avg time（先 warmup 再有效采样 >50 次取平均）不高于：case1 n=512 UPPER/N/NON_UNIT/incx=1 → **18.53 us**；case2 n=1024 LOWER/N/NON_UNIT/incx=1 → **39.34 us**；case3 n=2048 UPPER/T/NON_UNIT/incx=1 → **86.02 us** | 任务书 §3.3 |
| 内存标准 | 任务书未规定额外内存上限。设计上仅 direct fused 路径 workspace 为 0；OP_N 最坏 UB 占用 191.28125/192 KiB，并由分路径编译期断言约束 | 任务书 §3.4 |

### 精度验证方法

1. **测试框架**：ops-blas test 目录框架，CSV 描述用例 + C++ GTest 加载执行，
   golden 由 cblas 生成，落在 `test/tpmv/ctpmv/arch22/`。
2. **用例规模**：CSV **1200 条**，覆盖 uplo×trans×diag 全部 12 组正交组合、
   n 覆盖 0/1/小质数/2 的幂及 ±1/非对齐值直至大规模、incx 覆盖 ±1/±2/±3、
   AP 与 x 按均匀 [−5,5] 50% + 正态（μ∈[−5,5]，σ∈[0.1,2]）50% 生成，
   另含全零/交替/极端值/Inf/NaN 特殊值用例，以及负向用例。
3. **packed 映射定向验证**：对 n=1、2、3 构造每个 AP 位置都唯一的 marker，分别覆盖
   UPPER/LOWER 和 N/T/C，使用手工展开矩阵作为第二参考，避免 kernel 与 golden 同时采用
   同一个错误下标公式而互相“验证通过”。LOWER/n=3 必须显式检查列序
   `[A00, A10, A20, A11, A21, A22]`。
4. **「UNIT 不读对角」采用两层验证**：

   - 数值层：将所有 AP 对角位置填为 NaN/Inf/不同有限 marker，保持非对角与 x 不变；
     多次输出必须相同且为 golden 结果。这能证明对角值没有参与计算。
   - 访问层：审查并单测所有 AP 搬运区间和标量读取分支，证明 UNIT 路径的 GM 访问区间排除
     对角地址；若具备访存 trace/memcheck，则增加对角访问断言。NaN poison 单独不能证明
     "物理上没有读取后丢弃"，因此数值测试与访问层审查需同时满足。

   验证实施计划：测试工程中新增 `unit_diag_poison` 系列用例，将 AP 对角位置分别填为
   NaN、+Inf、−Inf 及与非对角截然不同的有限 marker（如 1e30），覆盖 UPPER/LOWER ×
   OP_N/OP_T/OP_C 共 6 组；同时在 kernel 代码审查中逐一标注 UNIT 分支的 AP 搬运区间
   起止计算，确认对角地址不出现在 `DataCopyPad` 源范围内。
5. **原地与步长验证**：除逐逻辑元素比对外，将 `|incx|>1` 的间隙位置填入 canary，
   验证 scatter 只改写 n 个逻辑位置；覆盖 `incx=±1/±2/±3`、n=0、n=1 和尾部非对齐。
6. **整数边界验证**：Host 单测覆盖 n(n+1)/2、x span 和 workspace 字节数恰好可表示及
   首个溢出点；覆盖 `incx=INT_MIN`，验证实现按 64 位绝对值处理或按已确认的接口约束拒绝，
   不能发生有符号溢出。
7. **特殊值验证**：有限值按任务书 mixed-tolerance 判据比较；Inf/NaN 逐分量比较 IEEE
   分类和无关元素隔离性。测试不得仅记录 cblas 与 NPU 的类别差异后仍判整例通过。

### 可测性设计：为什么容差测试不够

本算子有一段代码（补偿复数乘）**存在的意义就是控制最后一位**。
而套件里每个既有检查都是**容差**（32 ULP / atol 2⁻¹⁶ / matched_ratio 0.99）——
**一个改变每个输出最后一位的改动会全部静默通过**。

因此额外设计 `ctpmv_bitexact`：转储**原始 IEEE 位模式**
（10 个 n × 双 uplo × {N,T,C} × 双 diag × 3 种填充，
其中两种跨过 `IsLargeFinite` 的 2¹²⁶ 阈值并植入 Inf/NaN，
使缩放分支与特殊值分支都被走到，共 360 次调用），两个构建用 `cmp` 直接比较。

这一工具的必要性在开发中被证实两次：

| 场景 | 容差测试 | 位级比较 |
| --- | --- | --- |
| 补偿乘改寄存器形态 | 全过 | **IDENTICAL** —— 这才是「数值未变」的证明 |
| 一个流水化改动 | **全过** | **DIFFERS**：1.25% 的输出字，最大相对差 4.5e-4（重排了部分和的加法顺序） |

后者合法且良性，但**只有位级比较看得见**，本该在采纳前知道。

另一处覆盖缺口也由此暴露并补上：把精确复数乘换成朴素乘后，
既有的 reduce 测试报 **0 失败**——它的输入不触发特殊值分支。
**在 `ctpmv_bitexact` 之前，套件里没有任何测试能区分精确乘与朴素乘。**

### 可维性设计：构建来源门禁

开发中出现过一次典型故障：一次回归报出 86 条失败，看上去完全像真实缺陷，
实际是源码用 `cp` 还原后**没有重建** `out/lib64`——
git 跟踪源码，**没有东西跟踪 `.so`**，`git status` 与 `git diff` 都是干净的。

因此提供 `verify_ctpmv_build_identity.sh`，在任何性能结论产生**之前**
校验「实际被加载的库确实由当前源码在本次构建产出」：
commit、工作树状态、源码 SHA256、强制 scratch 重建、`.so` 与 harness 的 SHA256、
`ldd` 证明 loader 真的解析到该库、以及一次混序 smoke（含 n=400 与三个 case）。
**任一项失败即非零退出且不打印任何计时。**

该门禁本身也经过**四次下毒验证**（含「路径前置一个不同的 `.so`」一项，
第一次未通过——门禁曾无条件把自己的路径前置，导致相关检查不可证伪，已修正）。
**一个下毒不通过的门禁比没有门禁更危险，它给出虚假保证。**

## 兼容性分析

**新增算子，不涉及存量行为兼容性问题。** 具体分析：

| 维度 | 分析 |
| --- | --- |
| 接口新增 | `include/cann_ops_blas.h` 中此前**无** `aclblasCtpmv` 声明，纯新增，不改动任何既有声明 |
| 跨产品线 | 声明与 `cublasCtpmv`、同族 `aclblasStpmv` 逐参数对齐，可与含 Ascend 950PR 在内的其他产品线共用；**未定义 A2/A3 私有平行 API** |
| 既有算子 | 实现落在 `blas/tpmv/arch22/` 新增文件，**不修改** `stpmv_*` 等既有文件 |
| 测试工程 | 新增 `test/tpmv/ctpmv/`，对既有测试为纯增量 |
| ABI | `aclblasComplex` 沿用 `cann_ops_blas_common.h` 既有定义，未引入新类型 |
| 状态码 | 全部复用 `cann_ops_blas_common.h` 既有枚举，未新增状态码 |

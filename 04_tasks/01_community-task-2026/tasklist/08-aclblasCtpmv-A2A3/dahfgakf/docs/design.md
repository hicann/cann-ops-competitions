# 需求背景（required）

## 需求来源

CANN 社区任务「8月社区任务-aclblasCtpmv算子开发（A2A3）」，任务书 `aclblasCtpmv_Atlas800IA3_task_doc.md`。

在 `cann/ops-blas` 仓库中新增 BLAS Level-2 算子 `aclblasCtpmv`，对标 cuBLAS 的 `cublasCtpmv`，
适配 Atlas A2 / A3 系列产品。

## 背景介绍

### aclblasCtpmv 算子功能分析

`CTPMV` 是 BLAS Level-2 的标准接口，计算三角压缩（packed）矩阵与向量的乘法：

```
x := op(A) * x
```

| 项 | 内容 |
|---|---|
| 输入 | `AP`（complex64 三角压缩矩阵，长度 `n*(n+1)/2`）、`x`（complex64 向量，长度 `n`，步长 `incx`） |
| 输出 | `x`（原地更新） |
| 数据类型 | `aclblasComplex`（complex64 = float32 实部 + float32 虚部） |
| 属性 | `uplo`（UPPER / LOWER）、`trans`（OP_N / OP_T / OP_C）、`diag`（NON_UNIT / UNIT）、`incx`（任意非零整数，含负值） |
| 广播 | 不涉及（BLAS Level-2 接口，无广播语义） |

### ops-blas 仓库现状分析

`blas/tpmv/` 目录此前只有实数版本 `aclblasStpmv`（`arch22` / `arch35` 两套实现）。
复数版本 `aclblasCtpmv` 为**新增算子**，无历史 TBE 版本，无存量实现可对比。

复数场景相对实数版本的新增难点有三个，是本设计的主要着力点：

1. **复数在 UB 上以 `(re, im)` 交错存放**，而向量单元按 lane 逐元素运算，复数乘加需要"相邻 lane 配对"，
   不能直接用实数版本的向量流程；
2. **packed 布局中"列连续、行跳变"**，`trans = OP_N` 与 `OP_T/OP_C` 的访存形态完全不同，需要两套并行策略；
3. **复数三角乘在大 n 下存在强抵消**：行内正负项大量相消时输出量级可比各项量级低若干数量级，
   普通 FP32 顺序累加无法满足逐元素精度上限。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `aclblasCtpmv` 算子，支持 complex64 数据类型，
支持 `uplo` × `trans` × `diag` 共 12 组枚举全组合与任意非零步长 `incx`（含负步长），
在 Atlas A2 / A3 系列产品上满足任务书 §3.2 的精度标准与 §3.3 的性能标准。

## 需求拆解

1. 实现 `x := op(A) * x`，`op` 覆盖 `OP_N` / `OP_T` / `OP_C`（共轭转置）；
2. 支持 `uplo = UPPER / LOWER`、`diag = NON_UNIT / UNIT`，`diag = UNIT` 时**不得读取** `AP` 的对角线元素
   （该位置允许是任意值，包括 NaN）；
3. 支持任意非零 `incx`，含负步长；跨步产生的空隙保持原值；
4. 边界与负向行为对齐 cuBLAS：`n == 0` 为合法空操作且不访问指针；`handle` 为空、枚举非法、`n < 0`、
   `incx == 0`、`AP`/`x` 为空指针分别返回对应错误码；
5. 精度满足生态算子开源精度标准（实部、虚部分别按 FLOAT32 判定）；
6. 性能不高于任务书 §3.3 给出的标杆耗时；
7. 接口声明放入 `include/cann_ops_blas.h` 供其他产品线共用；`blas/tpmv/README.md` 产品支持表标注
   Atlas A2/A3 系列产品：支持。

---

# 详细设计（required）

## 算子分析

### 数学公式

```
x := op(A) * x
```

其中 `A` 为 `n × n` complex64 三角矩阵，以 packed 格式存储于长度 `n*(n+1)/2` 的数组 `AP`：

- `op(A) = A`（`OP_N`）、`A^T`（`OP_T`）、`A^H` 共轭转置（`OP_C`）；
- `diag = UNIT` 时对角线按 1 处理；
- `x` 中逻辑元素 `j` 的物理位置为 `j*|incx|`（`incx > 0`）或 `(n-1-j)*|incx|`（`incx < 0`）。

**packed 存储布局**（0 基下标，单位：complex 元素）：

| uplo | `A(i,j)` 所在位置 | 第 `p` 列起始位置（列内连续） |
|---|---|---|
| UPPER（`i ≤ j`） | `i + j*(j+1)/2` | `p*(p+1)/2`，列内保存 `A(0..p, p)` |
| LOWER（`i ≥ j`） | `i + j*(2n-j-1)/2` | `p*(2n-p+1)/2`，列内保存 `A(p..n-1, p)` |

**关键性质：packed 布局中"列"是连续的、"行"是带跳变的。** 两条向量化路径正是围绕这一点设计：

- `OP_N`：`y_i = Σ_p A(i,p)·x_p`，按 packed 列流式累加（每列对 `y` 做一次 AXPY），完全连续访存；
- `OP_T/OP_C`：`y_i = Σ_r op(A)(i,r)·x_r`，对固定输出行 `i` 恰好是 packed 的"第 `i` 列"，同样连续，
  但每行需要一次"列内点积"，故采用行并行 + 分段累加 + 归约。

### 支持数据类型

complex64（`aclblasComplex`，float32 实部 + float32 虚部）。

### 支持形状

| 参数 | 取值范围 |
|---|---|
| `n` | `n ≥ 0`；`n == 0` 为合法空操作 |
| `AP` | 长度 `n*(n+1)/2` 的 complex64 数组 |
| `x` | 长度 `(n-1)*|incx| + 1` 的 complex64 物理缓冲 |
| `incx` | 任意非零整数，含负值 |

不涉及广播。

### 接口定义

```cpp
aclblasStatus_t aclblasCtpmv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
                             aclblasDiagType_t diag, int n, const aclblasComplex* AP, aclblasComplex* x, int incx);
```

参数序列与对标接口 `cublasCtpmv` 一一对应。返回值约定：

| 输入 | 返回 |
|---|---|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` / `trans` / `diag` 非法 | `ACLBLAS_STATUS_INVALID_ENUM` |
| `n < 0`、`incx == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `AP == nullptr` 或 `x == nullptr`（`n > 0` 时） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n == 0` | `ACLBLAS_STATUS_SUCCESS`（不访问任何指针） |

## 算子实现

### 硬件约束与关键设计前提

1. **UB 向量操作数必须 32 字节对齐**（float 偏移为 8 的倍数）。开发早期通过探针核验证：对 UB 张量做
   "元素级错位"（偏移 1 个 float）的向量操作会触发
   `The UB address accessed by the VEC instruction is not aligned` 并导致 vector core exception。
   因此所有 UB 上的错位视图一律改为**独立的 GM 加载**（`DataCopyPad` 的 GM 侧无对齐要求）；
   无法 8 对齐的少量元素用标量方式处理。
2. **核间标量 GM 写在后续 kernel 的 DMA 读中可见，但不保证对 Host 可见**。故主 kernel 用标量 `SetValue`
   写 partial、写回 kernel 用 `DataCopyPad` 读 partial，最终结果一律通过 DMA 落回 `x`。
3. CANN 9.1 arch22 上 `TQue` 队列深度统一为 2；更深的队列在本算子上出现过 vector core 超时（内核挂死），
   故流水线深度固定为 2。

### 实现方案

#### 3.2.1 host 侧设计

**（1）参数校验**

按上表逐项校验，每个错误返回路径都带 `OP_LOGE` 并附函数名。`n == 0` 在指针校验之前返回成功。

**（2）tiling 策略：按规模分派的七条路径**

小 `n` 的代价由 kernel launch 决定（A2 上一次 launch 约 3.7 us，A3 约 6 us），大 `n` 的代价由访存/向量吞吐
决定。因此分成「单次 launch 的单核路径」与「两次 launch 的多核路径」两类：

| 路径 | 触发条件 | 并行维度 | 说明 |
|---|---|---|---|
| 直接返回 | `n == 1 且 diag == UNIT` | — | 结果即输入，不下发 kernel |
| `ONE` | `n == 1` | — | 单核单次 launch |
| `TINY` | `n ≤ 12`（任意 `trans`/`incx`） | — | A 与 x 整体载入 UB 后标量求值 |
| `SMALL` | `OP_T/OP_C`，`incx == 1`，`13 ≤ n ≤ 32` | — | 每个输出行是一条向量行 |
| `LINES` | `OP_T/OP_C` 的 `33 ≤ n ≤ 256`；`OP_N` 的 `13 ≤ n ≤ 128` | 输出行 | 单核阈值 `OP_T/OP_C` 为 48、`OP_N` 为 64，超过改多核写镜像 |
| `AXPY` | `OP_N`，`incx == 1`，`128 < n ≤ 4096` | packed 列 | 列并行 + 写回 |
| `DOT` | `OP_T/OP_C`，`incx == 1`，`n > 256` | 输出行 | 行并行 + 写回 |
| `SCALAR` | `incx != 1`、`n > 4096` 或可用向量核 < 4 | 行 | 标量兜底，正确性优先 |

`AXPY` / `DOT` / `SCALAR` 由**两个 kernel** 组成：主 kernel 每核算出 partial 写入 workspace，
写回 kernel 汇总 partial、补单位对角线项、逐行精度校验、写回 `x`。

> 选择两 kernel 而非单 kernel + `SyncAll`：主 kernel 的 UB 占用（列缓冲 + 累加器 + x 窗口）与写回 kernel
> 的 UB 占用叠加后接近/超过 192 KB 上限，且跨核同步有挂死风险。

**（3）AXPY 的分核策略：block-cyclic 列切分**

```
UPPER : 核 c 负责 p = c, c+C, c+2C, ...                  (C = 核数)
LOWER : 先按列号 mod 4 分成 4 个残类；核 c 属残类 r = c/perClass、类内序号 k = c%perClass，
        负责 p = r+4k, r+4k+C, ...
```

设计理由（实测驱动）：该路径耗时由"每核串行执行的 packed 列传输次数"决定，而非总字节数。
按**累计工作量**切分会让处理前段短列的核拿到 `O(√(n²/C))` 列，成为长尾；block-cyclic 让每核列数相同且
列长长短混搭，实测 `n=512 UPPER` 从 43.6 us 降到 32.1 us。

LOWER 要求同核所有列**残类一致**：累加器目的偏移为 `2*(p - colBase)`，只有同残类才能保证是 8 的倍数，
满足 UB 32B 对齐约束。

**（4）AXPY 的核数选择**

```
C = min(AIV 核数向下取 4 的倍数, max(24, roundup4(ceil(n / 32))))
```

即 `n ≤ 768` 用 24 核、`n = 1024` 用 32 核、`n ≥ 1536` 用满（A3 为 48 核、A2 为 40 核）。
每个 launch 的 block 都要付出启动与 UB buffer 初始化成本，且写回要串行汇总全部核的 partial；
A3 实测 48 核跑 `n=512` 反而比 24 核慢约 4 us。`C` 恒为 4 的倍数以满足 LOWER 残类划分。

**（5）DOT 的行切分**

按 `cost(i) = len(i) + 512`（列内元素数 + 一次固定行开销）做前缀和等分。
不加固定项时，处理大量短行的核会成为长尾。

**（6）workspace 布局**

```
partialU : useCoreNum * n 个 complex   （AXPY：每核全长度 planar U；DOT/SCALAR：紧凑 partial）
partialV : useCoreNum * n 个 complex   （仅 AXPY）
partialW : useCoreNum * n 个 complex   （仅 AXPY：每行各项的平方幅值，供逐行精度判据）
xSnap    : n 个 complex                （仅 AXPY：x 快照，写回重算行时读它）
stageIn / stageOut : 标量路径的 x 快照与结果暂存（仅 incx != 1）
```

各段按 512 B 对齐。`DOT` 与多核 `LINES` 只用一块 `n` 个 complex 的结果镜像，kernel 结束后由 host
在同一条 stream 上做 device-to-device 拷贝写回 `x`。

单核路径（ONE / TINY / SMALL / 单核 LINES）**不使用 workspace**。

#### 3.2.2 kernel 侧设计

**（1）AXPY 路径（`OP_N`）：planar 累加器**

向量单元处理的是 `(re, im)` 交错的 float 数组，直接做复数乘加需要"相邻 lane 配对"，因此改为保存两个实平面：

```
部分和（每核）：
  U[j] = Σ_p Re(A_float[j, p]) · Re(x_p)     j 为行 r 的 float 下标（2r 或 2r+1）
  V[j] = Σ_p Re(A_float[j, p]) · Im(x_p)

复述结果（写回阶段，利用 packed 列内相邻元素互为实/虚部）：
  out[2r]   = U[2r]   - V[2r+1]
  out[2r+1] = U[2r+1] + V[2r]
```

由于同一列内 `Im(A(r,p)) = Re(A(r+1,p))`，`V[2r+1]` 正好给出 `Σ Im(A(r,p))·Im(x_p)`。

写回需要 `V` 的 `±1` float 错位视图；UB 错位不可用，故改用 **3 次独立 DMA**（`U[b..e)`、`V[b+1..e+1)`、
`V[b-1..e-1)`）加载到三个对齐缓冲，再用掩码向量指令配对：

```
out = U - meven·V1 + modd·Vm1        meven = (1,0,1,0,…)   modd = (0,1,0,1,…)
```

> **先掩码再相加**比等价的更短写法 `out = U + Vm1 - meven·(Vm1 + V1)` 重要：后者在偶数 lane 上
> 先加 `V[2r-1]` 再减掉它，等于把**实部**与上一路的部分和做了一次大数相消。实测该写法在强抵消数据上
> 实部误差比虚部大一个数量级并超出逐元素上限；掩码写法多一条指令，误差回到正常水平。

**主循环流水**：每列一次 `DataCopyPad`（列内连续，长度 `2*segLen` float）+ 2 次 `Axpy`；
采用"先发射第 `t+1` 列传输、再等待第 `t` 列"的方式，使深度 2 的队列始终有 1 个传输在飞行中。

**（2）DOT 路径（`OP_T/OP_C`）：x 窗口预加载 + 每行一次归约**

核内一次性把 `x` 的一段读入 UB，并构造三个带掩码的副本：

```
x0s = x · msign      msign = (1,-1,1,-1,…)     → Σ(aSeg·x0s) = Σ(ar·xr) − Σ(ai·xi) = Re
x0m = x · meven      meven = (1,0,1,0,…)
x1m = (x<<1float) · meven                      → 偶数 lane 是 x 的虚部
```

于是内层循环每个 chunk 只需 **1 次 packed 加载**与 1 次错位加载：

```
accRe += aSeg  · x0s        (MulAddDst, FMA)
accIm += aSeg1 · x0m
accIm += aSeg  · x1m
```

`OP_C` 只需改变掩码符号。chunk 大小 512 complex；每行开始前对累加器窗口整体 `Duplicate(0)`
（`MulAddDst` 是累加语义，必须清零）；**每行只做 2 次 `ReduceSum`**（Re、Im），而不是每 chunk 归约一次。

LOWER 第 `i` 行的 x 偏移只有在 `(i + segHead - start) % 4 == 0` 时才 8 对齐，因此每行取出
`head = (4 - d%4) % 4`（≤3 个元素）用标量累加，剩余部分从对齐位置走向量路径；UPPER 偏移恒为 0，无需前导段。

**（3）小 n 单次 launch 路径（TINY / SMALL / LINES）**

- **TINY**（`n ≤ 12`）：A 与 x 的物理区间各一次 DMA 载入 UB，标量求值，整段一次 DMA 写回。
  跨步空隙原样带回，因此任意 `incx` 都支持。
- **SMALL**（`OP_T/OP_C`，`13 ≤ n ≤ 32`）：每个输出行取 packed 第 `i` 列，用 `Gather` 收成一条
  64 float 的向量行；`n` 条行拼成矩阵，与两条 x 线做广播重复乘，每条乘积行一次 `WholeReduceSum`。
- **LINES**：把 SMALL 推广到多核与更长的行——行按 `LN_SEG = 64` lane 分段，段 `q` 的第 `k` 行落在
  "块 `q` 的第 `k` 条线"上，同一块内所有行配同一段 x，广播乘一次搞定。

  `OP_N` 能复用这套机制的关键是 packed 行的寻址：

  ```
  A(i, j) = colBase(j) + i        colBase(j) = j(j+1)/2 (UPPER) 或 j(2n-j-1)/2 (LOWER)
  ```

  即"列基址表 + 行内偏移 `2i`"，整行只需一张**与行无关**的偏移表，`Gather` 基址取 `(guard + 2i)*4`，
  一次构建、所有行共用（表用向量指令算，`n ≤ 128` 时二次式全部落在 fp32 精确整数范围内，约 10 条指令）。

  `OP_N` + UPPER 的行长与 x 窗口方向恰好等同于 `OP_T` + LOWER，因此 kernel 内用 `lower_`（决定 packed 布局）
  与 `rev_`（决定元素顺序与 x 窗口方向）两个标志区分，分段乘加、归约、精度校验全部复用。

**（4）写回 kernel**

| 路径 | 工作内容 |
|---|---|
| AXPY | 每 512 行一个 block：汇总所有核的 `U`、`V[±1]`、`W`（深度 2 预取，每组 2 核 8 次传输），按掩码配对重建复数，`diag=UNIT` 补 `x`，逐行精度判定（必要时 EFT 重算），DMA 写回。**只 launch `ceil(n/512)` 个 block**，不再 launch 空闲核 |
| DOT / SCALAR | 每核负责自己那段行区间：DMA 读 partial，`diag=UNIT` 补 `x`，DMA 写回；`incx != 1` 时标量跨步写 |

**（5）精度设计：抵消场景的三条措施**

普通数据下 FP32 累加远在容差内，真正的难点是**抵消**：当 `|结果|` 远小于各项量级时，乘积与部分和的舍入
会把误差推到 `1e-2` 的绝对下限之上。

1. **不引入人为相消**：写回的复数重组按 lane 掩码后再相加（见上文），不再"先加后减"同一个部分和。
2. **逐行判据**：对行结果 `(re, im)` 与行内各项平方和 `S`

   ```
   lim = max(1e-2, min(|re|, |im|) · 2^-20)
   需要精确重算  ⟺  4 · 2^-24 · √S > lim
   ```

   `S` 的来源：DOT/LINES 用行内分段和的平方和；AXPY 用列方向累加出的第三个平面
   `W[j] = Σ_p Re(A_float[j,p])² · |x_p|²`（列缓冲在两次 `Axpy` 之后**原地平方**，不额外占 UB），
   行 `r` 的平方幅值 `S_r = W[2r] + W[2r+1]`。

   写回阶段每个 block 先对 `W` 做一次 `ReduceMax`：若 `4·S_max` 仍达不到最小逐元素上限（`1e-2`），
   整块直接写出（**常规数据都走这一支，标量循环完全不执行**）；否则逐行判定。
3. **精确重算**：被标记的行用误差自由变换（Dekker 拆分 + TwoSum，仅用 float 运算）在标量单元重算，
   `hi` 即普通 float 顺序和，因此非有限情形保持 float 语义。AXPY 的重算读主 kernel 拍下的 x 快照
   （写回是原地覆盖 `x`）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3 / 910B4） | √ |
| Atlas A3（910_93） | √ |
| Ascend 950（arch35，SIMT 实现，按任务要求只开发不测试） | √ |

## 算子约束限制

1. `incx != 1` 走单核标量路径（正确性优先，性能非最优）。任务书性能用例均为 `incx = 1`。
2. `n > 4096` 走标量兜底路径（UB 预算：DOT 路径需要 3 个 `2n` float 的 x 窗口 + 队列缓冲）。
3. `OP_N` 的 LINES 路径要求每核载入整个 packed 矩阵，上限为 `n = 128`（A 占 66 KB UB）；更大规模走 AXPY。
4. 强抵消数据在 AXPY 上会触发逐行精确重算，单行重算是 `n` 次标量全局读，极端数据（整行都被标记）下
   耗时显著高于常规数据——这是精度优先的取舍，任务书性能用例均为常规随机数据，不受影响。
5. arch35（Ascend 950）实现采用 SIMT 模型并已完成开发，但**未在硬件上验证**（任务要求"只开发不测试"）。
6. 不涉及广播、不涉及常驻显存。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部、虚部分别按 FLOAT32 判定：`rtol = 2^-10`、`atol = 2^-16`、`required_matched_ratio = 0.99`、`max_abs_error_limit = 1e-2 或 32×ULP`。逐元素通过条件 `\|actual − golden\| ≤ atol + rtol×\|golden\|`；用例同时满足 `matched_ratio ≥ 0.99` 与 `max_abs_error ≤ limit` 即判定通过 | 任务书 §3.2；生态算子开源精度标准 `opbase/docs/zh/ops_precision_standard/experimental_standard.md` |
| 性能标准 | COMPLEX64 场景下的平均单次耗时（Avg time，us），须先 warmup 再有效采样 >50 次取平均，不高于标杆耗时 | 任务书 §3.3 |
| 内存标准 | 不涉及 | 任务书 §3.4 |

**golden 与证据链**：golden 由双精度 `cblas_ztpmv` 计算（非有限时回退单精度），并用独立的 Netlib `CTPMV`
移植版与 OpenBLAS `cblas_ctpmv` 在 648 组参数（9 种 `n` × 2 `uplo` × 3 `trans` × 2 `diag` × 6 `incx`）下
逐元素交叉验证，作为"标杆本身正确"的证据。
另有专项用例把 packed 对角线写成 NaN，验证 `diag = UNIT` 时确实不读对角线。

**实测结果**（Atlas A2 / Ascend 910B4-1，40 AIV，独占单卡，CANN 9.1.0）：

| 验收项 | 结果 |
|---|---|
| 编译 | 0 错误 |
| 任务书精度用例（1241 条 CSV + 4 条专项） | **1245 / 1245 通过** |
| 逐用例 `matched_ratio` 最小值 | **0.998721**（要求 ≥ 0.99） |
| 任务书性能用例（200 条，上限 `gpu_ms / 0.8`） | **200 / 200 通过**，最差比值 **0.9147**，中位 0.5245 |
| 文档基准用例 `n = 512 / 1024 / 2048` | **15.85 / 21.65 / 82.44 us**（标杆 28.95 / 61.46 / 134.41），余量 45.2% / 64.8% / 38.7% |
| 多轮稳定性（文档用例连续 5 轮） | 变异系数 ≤ 1.78% |

**精度设计的额外验证**：除任务书用例外，另构造两套压力网格（不进仓库、不改动仓内用例 CSV）：

| 网格 | 内容 | 整改前 | 当前 |
|---|---|---|---|
| `OP_N` 专项 816 条 | `n = 13..160`，上/下三角 × 单位/非单位 × 8 种数据模式（含 `ALTER`、`EXTREME`、`INF`、`NAN`） | 71 条失败 | **816 / 816** |
| 大 n 强抵消 144 条 | `n = 129..4096`，`ALTER` / `EXTREME` 组合 | 85 条失败 | **144 / 144** |

**可测性**：测试工程以 CSV 描述用例（`test/tpmv/ctpmv/<arch>/ctpmv_test.csv`），GTest 加载后调用
`aclblasCtpmv`，覆盖小 shape 基础用例、尺寸扫描、步长组合、填充模式、边界与负向用例、Inf/NaN 特殊值、
以及性能/规模用例；另有 `BenchmarkDocCases` / `BenchmarkOverheadSeries` 两个性能用例，
按"warmup 10 次 + 背靠背提交 60 次取平均"的方式测量，满足任务书 §3.3 的采样要求。

## 兼容性分析

新增算子，不涉及存量接口的兼容性问题。具体而言：

1. **接口兼容**：参数序列与 `cublasCtpmv` 一一对应（`handle` 及后续参数顺序完全一致），
   无需额外映射说明；错误码语义对齐 ops-blas 仓已有的 `aclblasStpmv`。
2. **文件变更**：`include/cann_ops_blas.h` 仅新增 `aclblasCtpmv` 声明，不修改任何已有声明；
   `blas/tpmv/README.md` 仅新增 `aclblasCtpmv` 小节；其余均为新增文件。
3. **测试用例**：仓内 `ctpmv_test.csv` 前 1200 条与任务书随附用例**逐字节一致**（未修改、未剔除），
   后 41 条为自行补充的覆盖用例（`n = 4097 / 5000` 等标量兜底路径）。
4. **产品支持**：`blas/tpmv/README.md` 产品支持表标注 Atlas A2/A3 系列产品：支持。

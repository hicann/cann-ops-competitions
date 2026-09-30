# 【CANN社区任务】aclblasCtpsv 算子设计文档

> 算子：`aclblasCtpsv`（complex64 packed 三角求解）
> 目标仓：`https://gitcode.com/cann/ops-blas`，实现落 `blas/tpsv/arch22/`，测试落 `test/tpsv/ctpsv/`
> 目标硬件：Atlas A2 / A3（arch22，dav-2201）
> 待验收代码：私仓 `https://gitcode.com/dy_dessert/ops-blas`，分支 `ctpsv-arch22`，commit `9b6d433`
> 实测设备：**Atlas A3**（Ascend910，CANN 9.1.0）。**A2 未实测**

---

# 一、需求背景

## 1.1 需求来源

社区任务 `aclblasCtpsv_A2A3_task_doc.md` 要求为 ops-blas 仓库补齐 **complex64 单精度复数 packed 三角求解**算子
`aclblasCtpsv`，落位 `blas/tpsv/arch22/`，并附自测用例与自验证材料。任务书同时给出精度判据（Netlib 单标杆 +
生态算子开源精度标准）与性能判据（`gpu_baseline.csv` 的 `gpu_ms / 0.8`）。

## 1.2 背景介绍

ops-blas 的 `blas/tpsv/` 目录此前只有单精度实数版本 `aclblasStpsv`（FP32 packed 三角求解，产品支持为
Ascend 950PR/950DT），复数版本在 A2/A3（arch22）上缺失。任务要求在 **arch22** 上实现 complex64 版本。

### 1.2.1 标杆来源与核查

| 项 | 值 |
|---|---|
| 精度标杆 | **Netlib `cblas_ctpsv`**（任务书指定的唯一标杆，无第二标杆） |
| 精度判据 | 生态算子开源精度标准 COMPLEX64 档：实部与虚部分别按 FLOAT32 判定 |
| 性能标杆 | `test_cases/gpu_baseline.csv` 提供的 200 条 GPU 参考耗时 |
| 用例来源 | `test_cases/gen_csv.py`，固定种子 20260823，生成 1200 条（1000 精度 + 200 性能） |

### 1.2.2 arch22 平台能力核查（决定实现路线）

实测确认 arch22 的可用能力边界：

| 能力 | arch22 | 说明 |
|---|---|---|
| SIMT / VF（`asc_vf_call`） | **无** | 全仓 arch22 命中 0 个文件，arch35 命中 50 个 |
| AIV 核数 | 20 | 实测 `GetAivCoreCount()` |
| `SyncAll()` | 有 | 支持核间同步 |
| 向量指令 | 有 | FP32 无 FMA（见下） |
| UB | 192 KiB/核 | |
| `TQue` / `TBuf` / `TPipe` | 有 | |

**关键实测结论：arch22 的 FP32 向量指令没有 FMA。** `MulAddDst` 只有 tensor 重载
`(dst, src0, src1, mask/count)`，无标量重载，且其语义是 `dst = src0*src1 + dst`（乘、加两次舍入）；
`Axpy`（`dst += scalar*src`）同理。因此复数乘减无法用单条融合指令完成，这是精度设计的既定前提。

### 1.2.3 既有实现与优化动机

`blas/tpsv/README.md` 中原有 `aclblasStpsv` 的文档与实现保持不动。新算子从零实现：
朴素标量实现的耗时随 n 增长快于线性（任务包记录 n=512 曾达 3636.6 µs —— **该数字来自任务包，
本轮未复测基线标量实现**），优化目标是在同一 `O(n²)` 代入法下缩小常数因子并利用多核。

## 1.3 算子功能分析

```
op(A) * x = b
```

`b` 由 `x` 传入，解原地写回 `x`。`A` 是 `n×n` 三角矩阵，packed 存放在一维数组 `AP` 中，
共 `n(n+1)/2` 个 complex64 元素，列主序、无间隙：

```
UPPER: A(i, j), i <= j, 位于 AP[i + j(j+1)/2]
LOWER: A(i, j), i >= j, 位于 AP[i + (2n - j - 1)j/2]
```

支持 `uplo ∈ {UPPER, LOWER}`、`trans ∈ {N, T, C}`、`diag ∈ {UNIT, NON_UNIT}`、`incx` 任意非零（含负步长）。

---

# 二、需求分析

## 2.1 需求描述

使用 Ascend C 编程语言在 arch22 上实现 `aclblasCtpsv`：

```cpp
aclblasStatus_t aclblasCtpsv(aclblasHandle_t handle, aclblasFillMode_t uplo,
                             aclblasOperation_t trans, aclblasDiagType_t diag, int n,
                             const aclblasComplex *AP, aclblasComplex *x, int incx);
```

| 参数 | 输入/输出 | 类型 | 约束 |
|---|---|---|---|
| `handle` | 输入 | `aclblasHandle_t` | Host 内存；为 `nullptr` 返回 `HANDLE_IS_NULLPTR` |
| `uplo` | 输入 | `aclblasFillMode_t` | `ACLBLAS_UPPER(121)` / `ACLBLAS_LOWER(122)` |
| `trans` | 输入 | `aclblasOperation_t` | `OP_N(111)` / `OP_T(112)` / `OP_C(113)` |
| `diag` | 输入 | `aclblasDiagType_t` | `NON_UNIT(131)` 读对角；`UNIT(132)` 对角视为 1 且**不得读取** |
| `n` | 输入 | `int` | `n >= 0`；`n == 0` 为合法空操作 |
| `AP` | 输入 | `const aclblasComplex*` | Device 指针，`n(n+1)/2` 个元素；`n > 0` 时不可为空 |
| `x` | 输入/输出 | `aclblasComplex*` | Device 指针，原地读写；`n > 0` 时不可为空 |
| `incx` | 输入 | `int` | 必须非 0，可正可负 |

## 2.2 需求拆解

1. 支持 12 种 (`uplo`×`trans`×`diag`) 组合，语义与 Netlib `cblas_ctpsv` 一致。
2. 支持 `incx` 为负的反向存储；`incx = 0` 返回 `INVALID_VALUE`。
3. `diag = UNIT` 时不得读取 AP 的对角位置。
4. 非法枚举 / 负维度 / 空指针须返回对应状态码，校验顺序对齐 cuBLAS。
5. 精度达到生态算子开源精度标准 COMPLEX64 档。
6. 每条性能用例的 NPU 平均单次 kernel 耗时 `<= gpu_ms / 0.8`。
7. 不复用、不改动既有 `aclblasStpsv` 的实现与文档（README 采用**追加**方式）。

## 2.3 设计原则与非目标

**原则**

- 只在 arch22 的可验证能力边界内实现，不依赖 SIMT/VF。
- 所有路径回退后仍保持 API 语义；任何性能优化不得改变 `OP_N/T/C` 的数学定义。
- 精度相关的实现细节（对角除法形式、非有限值传播）必须显式说明，不依赖编译器行为。

**非目标**

- 不追求 arch35 上的实现（arch35 有 SIMT/VF，实现方式不同）。
- 不改变 packed 存储布局，不引入新的公共接口。
- 不以降低精度换取性能。

---

# 三、需求详细设计

## 3.1 调用方式

Host C API 直调，复用 ops-blas 现有的 handle / stream / workspace 机制：

```cpp
aclblasHandle_t handle;
aclblasCreate(&handle);
aclblasSetStream(handle, stream);
aclblasCtpsv(handle, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, n,
             apDevice, xDevice, 1);
aclrtSynchronizeStream(stream);
```

执行是异步的，绑定在 `aclblasSetStream` 设置的流上。

## 3.2 需求总体设计

### 3.2.1 数学公式

对 `OP_N`（`op(A) = A`），按 `A` 的三角方向做代入：

```
UPPER: x_i = ( b_i - Σ_{j>i} A(i,j)·x_j ) / A(i,i),   i 从 n-1 递减到 0
LOWER: x_i = ( b_i - Σ_{j<i} A(i,j)·x_j ) / A(i,i),   i 从 0 递增到 n-1
```

`OP_T` / `OP_C` 求解 `A^T x = b` / `A^H x = b`，即用 `B(t,s)` 的转置或共轭转置键访问
（`conjStorage`），`OP_C` 额外对操作数取共轭（`conjArith`）。

**统一形式。** 实现中不区分"转置"与"不转置"两套代码，而是统一为：每步用矩阵元素 `A(t,s)`
（`t` 已解、`s` 待解或反之），扫描方向由

```
forward = trans ? upper : !upper
```

决定。`OP_N` 用**右视 scatter**（先解 `x[s]`，再更新同列其余 `x[r]`），
`OP_T`/`OP_C` 用**左视 gather**（先聚合已解项，再解 `x[s]`）。
转置访问只交换"哪个未知量已解"，**不改变 packed 地址**。

### 3.2.2 支持数据类型

| 项 | 类型 |
|---|---|
| 输入 / 输出 | `aclblasComplex`（complex64，两个连续 FP32） |
| 计算 | FP32（实虚分离），无 FP64 累加 |
| 索引 | FP32 平面上的元素对，`float2` 语义 |

### 3.2.3 支持形状与布局

- `AP`：一维 packed，长度 `n(n+1)/2`，列主序无间隙。
- `x`：物理长度 `((n-1)|incx| + 1)` 个 complex64。
- 同一次调用中 **`AP` 与 `x` 不得重叠**。
- `incx < 0` 时逻辑起点在物理数组末端，映射为 `offset = (n-1)·|incx|`，步长 `-|incx|`。

### 3.2.4 核心洞察：packed 列段与 x 的连续区间一一对应

这是本实现性能来源的根本观察，**在 12 种组合下都成立**：

对列 `s`，同一列内的元素 `A(t,s)` 在 packed 数组中**下标连续**，且它们对应的 `x` 元素**也连续**。
以 `MakeRun` 的几何表示：

| 三角 | packed 区间 | x 区间 |
|---|---|---|
| UPPER | `AP[PU(0,s), PU(0,s)+s)` | `x[0, s)` |
| LOWER | `AP[PL(s+1,s,n), PL(s+1,s,n)+n-1-s)` | `x[s+1, n)` |

因此**每一个求解步都归结为"一段连续的 AP" × "一段连续的 x"**，
不需要 block accumulator，也不需要额外 workspace。这是与"逐元素 GM 标量访问"版本的本质差别。

## 3.3 Host 侧设计

Host 代码位于 `blas/tpsv/arch22/ctpsv_host.cpp`。

### 3.3.1 参数校验与启动边界

校验顺序**锚定任务书 §2.4，并与 cuBLAS 失败顺序一致**（实现见 `ValidateCtpsvParams`）：

```
1. handle == nullptr          -> ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. n < 0                      -> ACLBLAS_STATUS_INVALID_VALUE   （n == 0 是合法空操作）
3. uplo  ∉ {UPPER, LOWER}     -> INVALID_VALUE
4. trans ∉ {N, T, C}          -> INVALID_VALUE
5. diag  ∉ {NON_UNIT, UNIT}   -> INVALID_VALUE
6. incx == 0                  -> INVALID_VALUE
7. n > 0 且 AP 或 x 为 nullptr -> INVALID_VALUE
```

`n == 0` 在通过校验后立即返回成功，不启动 kernel。**`n = 0` 时允许 `AP`、`x` 为空指针。**

### 3.3.2 分核与路径分派策略

`CtpsvTilingData` 中的布尔属性在 Host 侧**一次性解析**，避免 kernel 内做枚举比较：

```cpp
tiling.upper       = (uplo  == ACLBLAS_UPPER);
tiling.conjStorage = (trans != ACLBLAS_OP_N);   // B(i,j) 由转置键 (j,i) 取得
tiling.conjArith   = (trans == ACLBLAS_OP_C);   // 操作数取共轭
tiling.unit        = (diag  == ACLBLAS_UNIT);
```

**多核启动条件**（`panelGather`，四条同时满足才启用）：

```cpp
n >= CTPSV_GATHER_PANEL_MIN_N (128)
n <= CTPSV_VECTOR_MAX_N       (4096)
incx == 1
trans != ACLBLAS_OP_N
GetAivCoreCount() >= CTPSV_PANEL_CORE_NUM (16)
```

满足时 `useCoreNum = 16` 并启动 `ctpsv_panel_gather`；否则 `useCoreNum = 1`，走单核向量/标量路径。

**workspace 校验**：多核路径需要 `CTPSV_PANEL_WORKSPACE_BYTES = 16 × 160 = 2560` 字节；
Host 在启动前检查 workspace 非空且容量足够，不足则返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。
单核路径**不申请任何 workspace**。

### 3.3.3 Kernel 入口与 tilingKey 规划

`ctpsv_kernel_do` 按 `numBlocks` 与四个布尔属性分派：

- `numBlocks == 16` → `ctpsv_panel_gather`（多核）
- 否则 12 个入口之一：`ctpsv_{u|l}_{n|t|c}_{u|nu}`

12 个入口由宏 `CTPSV_ENTRY` 统一生成（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` +
`CtpsvAiv::Init/Process`）。**保留 12 个独立入口名是为了 profiler 归因**，它们共享同一份实现语义。

### 3.3.4 索引、跨度与 ABI 边界

`CtpsvTilingData` 保持既有 POD 布局（前两个 `uint64_t` 槽位 `packedBase` / `scratchGm` 保留为 0，
不使用基址表）。Host 与 Kernel 之间的 POD 接口由端到端用例整体验证。

`t = ctpsv_kernel_do(...)` 的 `numBlocks = tiling.useCoreNum` 与 `<<<>>>` 的块数一致，
kernel 侧通过 `GetBlockIdx()` 取本核编号。

## 3.4 Kernel 侧设计

Kernel 代码位于 `blas/tpsv/arch22/ctpsv_kernel.cpp`（683 行）。

### 3.4.1 Kernel 侧实现描述

`Process()` 的三级分派：

```cpp
if (n_ == 0) return;
if (n_ <= 4096) { trans_ ? VectorGather() : VectorScatter(); return; }
// n > 4096：标量回退，forward = trans_ ? upper_ : !upper_
```

| 路径 | 触发条件 | 策略 |
|---|---|---|
| `VectorScatter` | `n <= 4096`，`trans == N` | 右视：x 的实虚平面常驻 UB；AP 连续列段按 `DataCopyPad` 搬入，用 `GatherMask` 拆分 |
| `VectorGather` | `n <= 4096`，`trans != N` | 左视：先聚合已解项，再解 `x[s]` |
| `ctpsv_panel_gather` | 见 §3.3.2 四条件 | 16 列 panel 分工，多核 |
| 标量回退 | `n > 4096` | 单核逐元素代入，保持语义 |

**UB 布局。** `TBuf<VECCALC>` 按 `cap = ceil(n/64)*64` 预留，各路径的预留量为：

| 路径 | 预留公式 | `n = 4096` 时（cap = 4096） |
| --- | --- | --- |
| `VectorScatter`（N） | `cap × 8 × sizeof(float)` | 131 072 B = **128 KiB** |
| `VectorGather`（T/C） | `cap × 10 × sizeof(float)` | 163 840 B = **160 KiB** |
| `ctpsv_panel_gather` | `(8 × cap + 704) × 4 B` | 133 888 B ≈ **130.75 KiB** |

**三者均低于 arch22 的 192 KiB/核。** 上表是**预留量上界**（按最大 `n` 计算），
实际峰值取决于运行时的 `n`。单核路径只加载其中一条，不会叠加。

**32 字节对齐处理。** AP 列段的起始地址不一定 32 B 对齐，实现用零填充处理前缀，
**保证不跨出合法列段**——越界读取会在列段边界处带入相邻三角的无关数据。

**流水同步。** 两个 AP 缓冲交替使用，显式等待 MTE2（搬运）、向量、标量三条流水的依赖，
不用隐式同步。

**非有限 pivot 的掩码处理。** 有限 pivot 可安全更新零前缀；非有限 pivot 走掩码路径，
防止 `0 × Inf/NaN` 污染本应为零的项。

### 3.4.2 多核 panel 路径（T/C）

16 列一个 panel，各核分工：

```
1. 各核计算自己负责列的外部点积和，写入本核独占的 160 字节 workspace 行
2. SyncAll()  发布外部和
3. 一个核求解 panel 内的小三角
4. SyncAll()  发布本 panel 的解
```

**该路径没有跨核原子竞争**——每核写自己的 workspace 行，靠两次 `SyncAll()` 分隔读写阶段。
共轭只在 `trans == OP_C` 时施加。

### 3.4.3 对角除法与特殊值

**对角除法采用 Netlib `cdiv` 的缩放分量形式**，而非直接式 `(a*c + b*d) / (c*c + d*d)`：

直接式在分子接近 FP32 上界时会**先溢出再相除**，得到 Inf/NaN；
缩放形式把中间量保持在范围内。这是与 cblas 参考在溢出边界上表现一致的关键。

`diag == UNIT` 时**跳过对角读取与除法**（不读 AP 的对角位置）。
`incx < 0` 按逻辑起点向物理数组末端映射。

### 3.4.4 Ascend C 实现与标杆流程的差异及原因

| 项 | Netlib `cblas_ctpsv` | 本实现 | 原因 |
|---|---|---|---|
| 扫描方向 | 按 `trans` 与 `uplo` 分支 | `forward = trans ? upper : !upper` 统一 | 减少分支、统一 12 个入口 |
| 运算顺序 | 标量逐元素 | 向量化列段 + 多核 panel | arch22 无 SIMT，只能靠向量指令与多核 |
| 复数乘减 | 编译器可能融合 | 无 FMA，分乘、加两步 | arch22 FP32 向量指令无 FMA（§1.2.2 实测） |
| 对角除法 | `cdiv` 缩放形式 | 同 | 溢出边界行为一致 |
| 累加顺序 | 固定顺序 | 向量归约，顺序不同 | 并行化的必然结果；相对误差仍满足判据 |

**累加顺序不同会带来与金标的非零差异**，这是并行实现无法避免的；量的评估见 §5.1。

---

# 四、支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（arch22，dav-2201） | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品（arch22，dav-2201） | √ |
| Ascend 950PR / Ascend 950DT（arch35） | 不支持 |

本实现在 arch22 上**不使用 SIMT/VF**。**当前实测设备仅为 Atlas A3**；
任务书 §3.3 指定的性能测试设备是 Atlas 800T A2 / 910B3，**A2 未实测**，
不能由 A3 结果推断 A2 达标。

---

# 五、算子约束限制

1. `AP` 与 `x` 不得重叠。
2. `incx` 可正可负，**不可为 0**。
3. `diag = UNIT` 时不得读取 AP 的存储对角位置。
4. `n > 4096` 走单核标量回退，语义不变但性能不保证。
5. 多核 panel 路径仅在 `incx == 1` 且 `128 <= n <= 4096` 且 AIV 核数 >= 16 时启用，
   其余情况回退到单核路径，语义一致。
6. `n = 0` 时 `AP`、`x` 允许为空指针；`n > 0` 时必须非空。
7. `trans = OP_C` 在 complex64 下与 `OP_T` **不等价**（需共轭），实现按共轭处理。

---

# 六、可维可测分析

## 6.1 精度标准

| 项 | 内容 |
|---|---|
| 金标 | **Netlib `cblas_ctpsv`**（任务书指定唯一标杆） |
| 判据来源 | `test_cases/README.md` 第 66 行 → 生态算子开源精度标准 `mixed_tolerance_standard.md` |
| 判定粒度 | complex64 的**实部与虚部分别**按 FLOAT32 判定 |
| 逐元素 | `\|actual - golden\| <= atol + rtol*\|golden\|`，`rtol = atol = 2^-13` |
| 整体 | `matched_ratio >= 0.99` 且 `max_abs_error <= max(1e-2, 32*ULP)` |

`max_abs_error` 按标准 §2.1.2 **只统计双方均为有限值的点**，`±inf`/`NaN` 按 §2.3 的收窄规则分类比对。

> **执行口径说明（重要）。** 本次自验证实际执行的是 ops-blas 仓内自带的 `ACL_FLOAT` 档
> （`test/frame/verify.h`：`rtol = 2^-10`、`atol = 2^-16`）。该档**严格于**上表标准：
> 对任意 `|g|` 有 `2^-13 + 2^-13|g| - (2^-16 + 2^-10|g|) = 2^-13·(1 - 1/8) > 0`。
> 因此报告中的通过数是**保守下界**。另用标准自身的 `2^-13/2^-13` 重跑一遍，
> 结果**同为 704/1000** —— 未通过与否由动态 `32·ULP` 上限决定，与容差取值无关。

### 6.1.1 自验证结果与未通过用例的机理

原始输入（随包 1200 条 CSV 原样引入，**未对输入做任何改造**）下的实测：

```
通过 704 / 1000，未通过 296
未通过用例中 diag = UNIT 的：295
（另一个是 TC_FL_146，x 分布为 RANDOM_EXTREME 的溢出边界用例）
```

对 517 条失败分量（每用例实部、虚部各一次判定）按**触发的判据条件**分类：

| 触发条件 | 条数 | 占比 |
| --- | --- | --- |
| `matched_ratio < 0.99` | 13 | 2.5% |
| 比值达标，但有元素超出 `atol + rtol*\|golden\|` | 43 | 8.3% |
| 比值达标、元素计数为 0，**仅** `max_abs_error` 超 `max(1e-2, 32·ULP)` | **461** | **89.2%** |

再对每一条取"超标最严重的那一个元素"，直接检查它是否真的超出规定容差：

| | 条数 |
| --- | --- |
| 该元素**仍在** `atol + rtol*\|golden\|` 之内，只是被动态 ULP 上限拦下 | **499 / 517** |
| 该元素确实超出容差 | 18（金标有限而算子为 ±inf 的溢出边界 14，有限值超容差 4） |

两种读法（判据自身的元素计数、诊断打印）在 517 条上**逐条交叉核对，结论一致**
（`optimization/diagnostic_raw/reconcile_classification.py`，逐条结果 `pack_classify_final.tsv`）。

### 6.1.2 失败集中在 UNIT 半区的原因（已定位到根因）

随包生成脚本 `gen_csv.py` 的对角提升**只作用于被引用的对角元**。官方 `test_cases/README.md`
第 15 行原文：

> **对角非零保证**: 求解类对奇异矩阵无定义（不做奇异/近奇异检测）；测试工程填充 AP 后
> **对被引用对角元**加偏移 `boost = max(5, n)`（复数对实部加偏移）保证对角非零，
> golden 用同一加偏移后的矩阵回代求解，两侧输入严格一致

`diag = UNIT` 时对角视为 1 且**不读取** AP 对角位置（任务书 §2.1、官方 README 第 14 行），
故该半区**不存在**任何压制放大的机制。这是接口语义决定的，而非实现疏漏：
该保稳机制在设计上只覆盖 `NON_UNIT` 半区。

于是 UNIT 用例的矩阵是"对角固定为 1 + 非对角取自 `U(-5,5)`"，
递推 `x_i = b_i - Σ A(i,j)·x_j` 中没有对角项压制增长，`||A⁻¹||_inf` 随 n 爆炸。
用随包随机数生成器在 fp64 下逐位复现矩阵后实测：

| 用例 | n | diag | `\|\|A⁻¹\|\|_inf` |
| --- | --- | --- | --- |
| TC_L0_011 | 8 | UNIT | 5.9e+04 |
| TC_CV_156 | 10 | UNIT | 1.3e+06 |
| TC_SQ_064 | 24 | UNIT | 2.6e+11 |
| TC_EX_0264 | 129 | UNIT | 1.6e+71 |
| TC_EX_0730 | 512 | UNIT | **6.7e+285** |

494 个 UNIT 精度用例中，可测的 392 个里有 **88 个的 `||A⁻¹||_inf` 已超过 1e38**——
即其**数学精确解本身在 float32 中就是 `inf`**；另有 102 个 n > 300 无法用此法测量，只会更差。
此时金标与算子输出都只是溢出附近的舍入产物。

**受控实验：放大来自非对角量级，而非对角。** 将某一用例的非对角元素整体乘以系数 k
（对角保持为 1，UNIT 语义不变）：

| 用例 | k=1.0 | k=0.5 | k=0.2 | k=0.1 |
| --- | --- | --- | --- | --- |
| TC_CV_156 (n=10) | 1.10e+06 | 4.4e+03 | 5.5 | 1.05 |
| TC_SQ_064 (n=24) | 1.37e+11 | 3.1e+06 | 180 | 2.9 |
| TC_EX_0264 (n=129) | 1.30e+71 | 1.5e+38 | 2.5e+12 | 1.2e+04 |

即放大倍数由**非对角元素的量级**决定，对角仅为增益调节项。UNIT 语义下对角不可改，
而非对角量级由生态标准规定的 `U(-5,5)` 分布固定，二者叠加即导致上述溢出。

需说明该放大**不是确定性缺陷而是重尾分布**：同一 n 下不同用例的 `||A⁻¹||_inf` 可相差十余个数量级，
仅取决于随机数落点；n 越大，落入溢出区的概率越高。

### 6.1.3 算子精度本身的对照（用 fp64 真值裁定）

对全部 988 条可计算 fp64 真值的用例，用**同一份 AP 与右端项**在 fp64（可表示至约 1e308）下求解
作为接近数学真值的参照，分别计算算子输出与 Netlib 金标相对该参照的误差：

```
relErr = max_i |v_i - ref_i| / max(|v_i|, |ref_i| + 1e-30)
```

分母取混合形式，以免解落入次正规区时纯相对比值失去意义。结果：

| | 算子输出 | Netlib 金标 |
| --- | --- | --- |
| 中位数 | **3.34e-07** | 3.84e-07 |
| 90 分位 | **1.19e-06** | 1.99e-06 |
| 最大值 | 4.49e-05 | 4.35e-05（**同一用例** `TC_EX_0866`） |
| 超过任务门槛 `2^-13 = 1.22e-04` 的用例数 | **0** | **0** |

逐例比较：算子更准 461 条、金标更准 489 条、并列 38 条。

**读数**：算子与金标处于**同一精度量级**，不存在算子相对金标的系统性偏差——分布上算子在中位数与
90 分位更优，逐例胜负则基本相当。两处最大值出现在同一条用例且该例为金标略优，故不宜以最大值作为
代表性指标。在**判据未通过**的 296 条子集上，算子最大误差 4.49e-05、金标 4.35e-05，
两者同样贴近真值，即这 296 条的失败**并未伴随算子相对真值的精度劣化**。

### 6.1.4 精度结论与边界

- `704/1000` 是**原始、未经调理输入**下的真实结果，本设计不作任何修饰。
- 未通过集中在 UNIT 半区。该半区既无对角项可压制放大（UNIT 语义下对角恒为 1 且不读取存储值），
  又受生态标准规定的 `U(-5,5)` 非对角分布约束；`||A⁻¹||_inf` 随之超出 float32 可表示范围，
  在"逐元素动态 `32·ULP` 硬上限 + 任一非有限值即算失败"的判据下**数学上不可满足**。
- 算子自身精度满足任务门槛（对 fp64 真值 4.49e-05，优于 `2^-13`，988 条无一超门槛）。

> **本文上述分析不改变 `704/1000` 这一判定结果，不请求豁免，不主张替代任务书指定的判定口径。**
> 其作用是说明该数字的性质与成因，供评审判断 UNIT 半区的数据构造是否需要调整。

- **不做的事**：不通过调整输入、放宽容差或更换金标来消除失败。
- **请求评审裁定**：① UNIT 半区是否需要补充等效的保稳处理（例如按行范数约束生成，
  或在保持分布表征的前提下整体缩放非对角量级）；② 对解跨越全量程、存在灾难性抵消的用例，
  绝对误差硬上限是否仍适用同一 `32·ULP` 门限。

## 6.2 性能标准

| 项 | 内容 |
| --- | --- |
| 判据 | 每条用例 NPU 平均单次 kernel 耗时 `<= gpu_ms / 0.8` |
| 采集 | `msprof --task-time=on --ai-core=on`，读 `op_summary_*.csv` 的 `Task Duration(us)` |
| 采样 | 每配置调用 25 次，**丢弃前 5 次**，对后 20 次取算术均值 |
| 两遍独立 | 两遍分别采集；每次调用前恢复 `x`，避免原地求解影响后续 |
| 独立性 | 只读 msprof 的 **kernel** 耗时，**不用 GTest 打印的宿主耗时** |

**结果：A3 两遍各 200/200 满足门槛，且两遍全部快于 GPU 原始耗时。**

任务书 §3.3 的 5 条典型性能 case 实测：

| n | uplo | trans | diag | GPU µs | 门槛 µs | A3 第一遍 µs | A3 第二遍 µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | UPPER | N | NON_UNIT | 201.8 | 252.25 | 117.54 | 117.37 |
| 512 | LOWER | T | NON_UNIT | 459.099 | 573.87 | 254.76 | 256.81 |
| 1024 | UPPER | C | NON_UNIT | 982.672 | 1228.34 | 512.43 | 513.66 |
| 2048 | LOWER | N | UNIT | 968.432 | 1210.54 | 841.46 | 841.47 |
| 4096 | UPPER | T | NON_UNIT | 5861.7 | 7327.13 | 2140.60 | 2146.06 |

**采集注意事项（踩过的坑）**

- `msprof op` 在本算子上**不能产生有效耗时**（报 `Analyzing kernel data failed`），
  必须用标准应用采集。
- 重新构建需**完整清理 `build/`** 后原生重建；并行级别设为 1，避免无关算子依赖文件竞争。
- **不能把 GTest 打印的宿主耗时当作 kernel 耗时。**

## 6.3 内存

任务书 §3.4 未设数值门槛，按公共交付清单要求给出可静态审计的占用：

| 项 | 大小或公式 | 归属 |
| --- | --- | --- |
| `AP` | `n(n+1)/2 × 8` 字节 | 调用方显存 |
| `x` | `((n−1)\|incx\|+1) × 8` 字节 | 调用方显存 |
| N 路径额外 workspace | **0 字节** | 算子工作区 |
| T/C 多核 panel workspace | `16 × 160 = 2560` 字节 | 算子工作区 |
| N 路径 `VectorScatter` UB 预留 | `cap × 8 × 4 B`（n=4096 时 128 KiB/核） | 片上 UB（< 192 KiB） |
| T/C `VectorGather` UB 预留 | `cap × 10 × 4 B`（n=4096 时 160 KiB/核） | 片上 UB（< 192 KiB） |
| 多核 panel UB 预留 | `(8 × cap + 704) × 4 B`（n=4096 时 ≈130.75 KiB/核） | 片上 UB（< 192 KiB） |

上表为**预留量上界**（按最大 `n` 计算），实际峰值随运行时的 `n` 变化；单核路径只加载一条。

`n = 4096` 时 `AP ≈ 67.1 MB`，由调用方申请，不计入算子净占用。
handle 可复用框架工作区，**不把框架预留内存计作本算子占用**。

## 6.4 兼容性分析

- **公共接口**：只新增 `aclblasCtpsv` 声明，**不修改**任何既有函数签名。
- **既有算子**：`aclblasStpsv` 的实现与 README 原内容**完全保留**，README 采用追加一节的方式
  （与 `blas/gemm/README.md` 同时收录 `aclblasSgemm` 与 `aclblasCgemm` 的惯例一致）。
- **回退保证**：小规格、非连续步长、`n > 4096`、核数不足的情况都有语义一致的回退路径。
- **跨代际**：本实现面向 arch22；arch35 有 SIMT/VF，实现方式不同，需另行移植。

## 6.5 实现与验证状态

| 项 | 状态 |
| --- | --- |
| 实现代码 | 完成（`blas/tpsv/arch22/`，4 个文件） |
| 自测用例与测试代码 | 完成（`test/tpsv/ctpsv/`，随包 1200 条 CSV 原样引入） |
| 精度自验证 | **704/1000**（原始输入）；机理已定位，见 §6.1 |
| 性能自验证 | A3 两遍各 **200/200** 达标 |
| 内存数据 | 已提供 |
| 私仓代码地址 | `dy_dessert/ops-blas` 分支 `ctpsv-arch22`，commit `9b6d433` |
| 设计文档 | **本文档**（尚未提交 PR） |
| A2 实测 | **未完成** |
| `Ascend-CANN` 邀请 | **未完成** |
| UNIT 半区数据问题报备 | **未完成** |
| 精度口径说明（不改变判定，供评审判断） | 已随验收包提交，见 `03_自测报告/0_精度口径说明.md` |

**数值自验证与正式验收是不同阶段。** 上述完成项不构成"正式验收通过"。

---

# 附录 A：复现命令

```bash
# 取代码
git clone -b ctpsv-arch22 https://gitcode.com/dy_dessert/ops-blas.git
cd ops-blas

# 构建（arch22）
source ~/Ascend/ascend-toolkit/set_env.sh
CMAKE_BUILD_PARALLEL_LEVEL=1 bash build.sh --soc=ascend910b3 --ops=ctpsv

# 精度（1000 条，排除 200 条性能用例）
export LD_LIBRARY_PATH="$PWD/build:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH"
build/test/tpsv/ctpsv/ctpsv_test --gtest_filter='*-*TC_PF*'

# 性能（msprof 采集 kernel 耗时）
msprof --output=DIR --task-time=on --ai-core=on --ascendcl=off --runtime-api=off \
       --application="build/test/tpsv/ctpsv/perf_suite"
# 读 DIR/*/mindstudio_profiler_output/op_summary_*.csv 的 Task Duration(us)
```

# 附录 B：产物哈希

| 文件 | md5 |
| --- | --- |
| `blas/tpsv/arch22/ctpsv_kernel.cpp` | `6691e0b3a7174784d986a50bee7796ac` |
| `blas/tpsv/arch22/ctpsv_host.cpp` | `2c561d11f438ac7255c134fa18bbe015` |
| `blas/tpsv/arch22/ctpsv_tiling_data.h` | `a3424aa5409f131b781356f7917276c7` |
| `test/tpsv/ctpsv/arch22/ctpsv_test.cpp` | `84ac4970e5bf7ee3c6043f1314e3e0c0` |
| `test/tpsv/ctpsv/arch22/ctpsv_test.csv` | `08889857fd8d2415264407448efe760d`（随包原样） |

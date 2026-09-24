# aclblasSsbmv 算子设计文档（Atlas A2/A3）

| 文档版本 | 日期 | 作者/团队 | 说明 |
| --- | --- | --- | --- |
| V1.0 | 2026-09-21 | yumeyo | 按社区任务设计文档模板整理的 A2/A3 评审稿 |

> 目标代码目录为 `ops-blas/blas/sbmv/arch22/`，测试目录为 `ops-blas/test/sbmv/ssbmv/arch22/`。
> 公开接口 `aclblasSsbmv` 由所有产品线共用，本文只设计 Atlas A2/A3（DAV_2201 / arch22，`SOC_VERSION=ascend910b3`）路径；`blas/sbmv/arch35/`（Ascend 950PR / 950DT）已有实现不受本次改动影响。

# 需求背景（required）

## 需求来源

本需求来自 CANN 社区任务「算子实操工坊-北京站-aclblasSsbmv算子开发(A2A3)」。

任务要求在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3 系列产品，DAV_2201，CANN 9.1.0）上使用 Ascend C 开发**单精度实数（float32）对称带状矩阵-向量乘**算子 `aclblasSsbmv`，核心功能与参数语义对标 cuBLAS `cublasSsbmv` 与 Netlib `ssbmv`；算子实现最终合入昇腾算子开源仓 ops-blas（https://gitcode.com/cann/ops-blas ）的 `blas/sbmv/arch22/` 目录，测试代码合入 `test/sbmv/ssbmv/arch22/`。

## 背景介绍

### 算子定位与数学定义

SBVM 属于 BLAS Level-2，计算

```
y := alpha * A * x + beta * y
```

其中 `A` 为 `n×n` **对称带状实数矩阵**（`A = Aᵀ`），半带宽为 `k`（对角带宽 `0..k`）；`x`、`y` 为长度 `n` 的单精度实数向量；`alpha`、`beta` 为单精度实数标量，`alpha`、`beta` 以 **Host 指针**传入。

`A` 以列主序（Column-Major）存储且**只存一半带状部分**（由 `uplo` 决定存上带还是下带），存储缓冲尺寸为 `lda × n`，`lda >= k+1`。未存储的另一半带按对称关系 `A = Aᵀ` 在索引层隐含引用，**不复制数据**。

矩阵-向量乘是稠密/稀疏线性代数的基础内核，带状版本广泛用于三对角/带状系统的迭代求解、协方差矩阵相关的统计计算与隐式时间推进等场景。它的核心特征是：**访存量与计算量都随带宽 `k` 线性增长**，且由于 `k` 通常远小于 `n`，逐输出行的求和长度只有 `2k+1` 项（长归约问题在本算子上并不成立），因此其性能瓶颈不在算力与带宽，而在**数据重排（banded→连续向量）与指令发射效率**——这是本设计的主要着力点。

### 带状存储布局与语义分支

以下均为 0-based 下标，`A_store[i]` 表示 GM 中第 `i` 个 float：

| uplo | 存储关系 | 第 `c` 列的带内元素 |
| --- | --- | --- |
| UPPER（存上半带） | `A(i, c) = A_store[(i − c + k) + lda·c]`，`i ∈ [c−k, c]` | `a_c[r] = A(c−k+r, c)`，`r = 0..k` |
| LOWER（存下半带） | `A(i, c) = A_store[(i − c) + lda·c]`，`i ∈ [c, c+k]` | `a_c[r] = A(c+r, c)`，`r = 0..k` |

向量 `x` / `y` 的逻辑长度为 `n`，物理下标映射为

```
UPPER/LOWER 通用：x_phys(j) = (incx > 0) ? (j·incx) : ((n−1−j)·|incx|)
                  y_phys(i) = (incy > 0) ? (i·incy) : ((n−1−i)·|incy|)
```

**语义分支（与 Netlib `ssbmv` / cuBLAS 一致，经直连系统 `libblas` 的独立探针实测确认）**：

| 场景 | 语义 |
| --- | --- |
| `n == 0` | 合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，`y` 不被触碰 |
| `alpha == 0` | **跳过矩阵乘、不引用 A 与 x**（故此时 A/x 可为 `nullptr`）：`beta == 1` → `y` 完全不变；`beta == 0` → `y` 各元素精确置 `+0.0`；其余 `beta` → `y := beta·y`（读改写） |
| `beta == 0`（`alpha != 0`） | **不读取 `y`**（先向 `y` 写入 0 再累加 `alpha·A·x`），故 `y` 中原有的 NaN/Inf 不传播 |
| 其它 | 常规 `y := alpha·A·x + beta·y` |

### 对标接口与现状分析

| 项目 | 内容 |
| --- | --- |
| 对标基线接口 | cuBLAS `cublasSsbmv`（语义参考 Netlib `ssbmv.f`）；参数序列与 cuBLAS 一一对应，无需额外映射 |
| 公开接口声明 | `include/cann_ops_blas.h:156`（**声明已存在**，本次不新增、不修改签名，禁止定义产品私有平行接口） |
| 仓内平台 | ops-blas 采用**句柄式 BLAS 接口 + Ascend C kernel 直调**工程模式：`aclblasHandle_t` 携带 stream（`aclblasSetStream` 绑定），Host 侧 `blas/sbmv/arch22/ssbmv_host.cpp` 完成校验与 Tiling 后直接下发 kernel，不经过 aclnn 图/原型注册链路 |
| 仓内现状 | `blas/sbmv/` 下**只有 `arch35/`**；`blas/sbmv/arch22/` 与 `test/sbmv/ssbmv/arch22/` **均不存在、需新建** |
| 接口枚举名口径 | 仓内头文件 `include/cann_ops_blas_common.h:35-38` 的枚举为 `ACLBLAS_UPPER = 121` / `ACLBLAS_LOWER = 122`；任务书 §2.4 与配套 CSV 中出现的 `ACLBLAS_FILL_MODE_UPPER/LOWER` 只是**用例文本层的字面量**，需在测试侧做文本转换 |
| 算子文档 | `blas/sbmv/README.md` 已存在，产品支持表与接口说明需补齐 A2/A3 路径 |

### 本次实现路径

- **目标硬件**：Atlas A2/A3 系列产品（DAV_2201，`arch22`），CANN 9.1.0
- **编程语言**：Ascend C（经典 arch22 编程模型：`TPipe` / `TQue` / `TBuf` / `GlobalTensor` / `LocalTensor`，**AIV only**，`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`）
- **代码架构**：**SIMD**，实现载体为 **MemBase**（dav-2201 无 SIMT 硬件；本算子非 Matmul（GEMM/BMM）类且目标芯片非 ascend950，故 Cube/Blaze 路线不适用；本算子为纯单卡算子，故通算融合 MC2 路线不适用；支持 RegBase 的芯片不使用 MemBase，两者互斥，dav-2201 属 MemBase 载体）
- **实现目录**：`blas/sbmv/arch22/`（Host + Kernel + Tiling）
- **测试目录**：`test/sbmv/ssbmv/arch22/`（GTest + CSV 驱动）

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现单精度对称带状矩阵-向量乘算子 `aclblasSsbmv`，支持 FLOAT32 单一数据类型，接口原型为：

```cpp
aclblasStatus_t aclblasSsbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* x, int incx,
    const float* beta,
    float* y, int incy);
```

算子语义为 `y := alpha * A * x + beta * y`：`A` 为 `n×n` 对称带状矩阵（列主序，仅存 `uplo` 指定三角的带状部分，前导维 `lda >= k+1`）；`x`、`y` 为长度 `n` 的向量，`incx` / `incy` 支持正负步长；`y` 为原地输出。算子须复用调用方绑定的 stream（异步执行），调用方在读取 `y` 前须自行同步该 stream。

## 需求拆解

1. **功能**：实现 `y := alpha·A·x + beta·y`，支持 UPPER / LOWER 两种带状存储模式，未存储半带按 `A = Aᵀ` 隐含引用，**不复制数据**。
2. **数据类型**：仅 FLOAT32（输入 `A`/`x`/`y` 与标量 `alpha`/`beta` 均为单精度实数），不涉及 FP16/BF16/INT。
3. **形状**：`n >= 0`、`k >= 0`（**`k` 无上界，`k >= n` 合法**）、`lda >= k+1`、`incx != 0`、`incy != 0`、`A` 缓冲 `lda×n`、`x`/`y` 逻辑长度 `n`（物理长度 `1+(n-1)·|inc|`）。
4. **边界语义**：`n == 0` 合法 no-op（不触碰 `y`）；`alpha == 0` 跳过矩阵乘且不引用 A/x（`beta ∈ {0,1}` 等情形逐项对齐 Netlib 语义，详见 §7.3 边界语义）；`beta == 0` 不读 `y`；`k >= n` 返回 `SUCCESS`（合法退化，除 `k < 0` 与 `lda < k+1` 外无任何带宽上界校验）。
5. **参数校验与返回码**：`handle == nullptr` → `HANDLE_IS_NULLPTR`；`n < 0` / `k < 0` / `lda < k+1` / `incx == 0` / `incy == 0` / `alpha == nullptr` / `beta == nullptr` → `INVALID_VALUE`；非法 `uplo` → **`INVALID_ENUM`**；`n > 0` 且 `alpha` 非零时 `A`/`x` 为 `nullptr` → `INVALID_VALUE`；`n > 0` 时 `y` 为 `nullptr` → `INVALID_VALUE`。所有校验在 launch 之前同步完成。
6. **精度**：输出 `y` 按 FLOAT32 档判定（`rtol = 2⁻¹⁰`、`atol = 2⁻¹⁶`、`matchedRatio >= 0.99`、逐元素绝对硬门 `max(1e-2, 32×ULP)`），全向量验证；**特例**：`alpha == 0` 且指针非空时 `y = beta·y` 须**位精确（EXACT）**。
7. **性能**：4 个性能 case 的平均单次耗时（先 warmup 再有效采样 >50 次取平均）不高于 `13.57 / 24.67 / 14.41 / 47.74 µs`。
8. **工程**：kernel 直调模式，句柄式接口，接口声明放入 `include/cann_ops_blas.h` 共用；实现落 `blas/sbmv/arch22/`，测试落 `test/sbmv/ssbmv/arch22/`（含 CSV）；算子 README 产品支持表标注 Atlas A2/A3 系列产品：支持。
9. **自验**：用例以 CSV 描述、C++ GTest 加载并调用接口执行，精度 golden 由 cblas（Netlib BLAS 实数实现）生成；覆盖小 shape 基础用例、shape 扫描、填充模式（`lda` padding）、对齐偏移、边界与负向用例（零维、空指针、非法步长、负维度）、规格允许的 INF/NAN 场景、性能用例；配套用例未覆盖的场景须自行补充（正态分布输入、`k >= n` 退化、非二进制浮点标量、`handle == nullptr`、`alpha` 大值等）。

# 详细设计（required）

## 算子分析

### 数学公式

```
y := alpha * A * x + beta * y
```

其中 `A ∈ R^{n×n}` 为对称带状矩阵（`A = Aᵀ`，半带宽 `k`），`x, y ∈ R^n`，`alpha, beta ∈ R`。

带状存储（列主序，`A_store` 长度 `lda·n`，`lda >= k+1`）：

```
UPPER：A(i, c) = A_store[(i − c + k) + lda·c]     (c − k <= i <= c)
LOWER：A(i, c) = A_store[(i − c)     + lda·c]     (c <= i <= c + k)
未存储半带由 A = Aᵀ 隐含：UPPER 的 A(i,c) = A(c,i) 取 c 列的镜像带内元素，反之亦然
```

由于 `A` 对称，存储列 `c` 的带内元素 `a_c[r]`（`r = 0..k`）同时参与**两个**输出行的累加（对角元素 `r=k`（UPPER）/ `r=0`（LOWER）两项重合，只保留 T1 一项）。这一「一列两用」的分解是本设计把带状矩阵乘化为**纯逐元素向量乘加**的关键：

| uplo | 项 | 输出行 `i` | 权重取自 `x` 的下标 `j` | `r` 范围 |
| --- | --- | --- | --- | --- |
| UPPER | **T1**（行 ≤ 列，直取） | `i = c − k + r` | `j = c` | `r = 0..k` |
| UPPER | **T2**（行 > 列，经对称镜像） | `i = c` | `j = c − k + r` | `r = 0..k−1` |
| LOWER | **T1**（行 ≥ 列，直取） | `i = c + r` | `j = c` | `r = 0..k` |
| LOWER | **T2**（行 < 列，经对称镜像） | `i = c` | `j = c + r` | `r = 1..k` |

### 支持数据类型

| 数据类型 | 说明 |
| --- | --- |
| FLOAT32 | 本算子唯一支持类型（输入 `A`/`x`/`y` 与标量 `alpha`/`beta` 均为单精度实数）。不涉及 FP16 / BF16 / INT，**全程无类型转换、无 `Cast`**。 |

> 全程 fp32 是精度口径的硬约束：任何引入 fp16/bf16 中间态的“优化”都会改变舍入链、使结果与 golden（fp32 的 `cblas_ssbmv`）不等价，属明确禁止项（见「算子约束限制」）。

### 支持形状

| 参数 | 取值范围 / 形状 | 说明 |
| --- | --- | --- |
| `n` | `n >= 0`；**功能范围 `n <= 8192`**（`n > 8192` 不保证功能） | 方阵阶数，也是 `x`/`y` 的逻辑长度；`n == 0` 为合法 no-op |
| `k` | `k >= 0`，**无上界**（`k >= n` 合法，返回 `SUCCESS`） | 半带宽；除 `k < 0` 与 `lda < k+1` 外无带宽上界校验 |
| `A` | 存储缓冲 `lda × n`（元素数），仅存 `uplo` 指定三角的带状部分 | Device 内存，只读，列主序 |
| `lda` | `lda >= k+1`；支持紧凑（`lda = k+1`）与列间 padding（`lda > k+1`） | Host 标量 |
| `x` | 逻辑长度 `n`，物理长度 `1 + (n−1)·|incx|` | Device 内存，只读 |
| `y` | 逻辑长度 `n`，物理长度 `1 + (n−1)·|incy|` | Device 内存，输入旧值、原地覆写输出新值 |
| `incx` / `incy` | `!= 0`，支持正负（含 `|inc| > n` 的缓冲区空洞） | Host 标量；`== 0` 非法 |
| 广播 / rank 提升 | 不涉及（`A`/`x`/`y` 为独立操作数） | — |
| dynamic shape | 不要求（`n`/`k`/`lda`/步长均为运行时入参） | — |

> 本算子**不支持**超出 `incx` / `incy` / `lda` 语义之外的非连续内存访问（任务书 §2.5 明确不要求）。

### 参数与异常行为

校验顺序为**全序**（每个条件互斥、按序短路），任一输入组合只会命中**首个**成立的条件，从而保证「同一输入不落入不同错误码」：

| 序 | 条件 | 返回码 |
| --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `uplo` 不属 {`ACLBLAS_UPPER`(121), `ACLBLAS_LOWER`(122)} | **`ACLBLAS_STATUS_INVALID_ENUM`** |
| 4 | `k < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `lda < k + 1` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `incx == 0 \|\| incy == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `alpha == nullptr \|\| beta == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `n == 0` | `ACLBLAS_STATUS_SUCCESS`（quick return，**不触碰 `A`/`x`/`y`**） |
| 9 | `alpha != 0.0f && (A == nullptr \|\| x == nullptr)`（条件化判空） | `ACLBLAS_STATUS_INVALID_VALUE` |
| 10 | `y == nullptr`（与 `beta` 无关，`n > 0` 时恒必需） | `ACLBLAS_STATUS_INVALID_VALUE` |
| 11 | 全部通过 | 计算并返回 `ACLBLAS_STATUS_SUCCESS` |

**关键设计点（顺序敏感，不得调换）**：第 8 步（`n == 0` 短路）刻意放在第 1–7 步**之后**、第 9/10 步**之前**。理由有三：① `n == 0` 是**合法空操作**，若置于合法性校验之前，则 `n=0` 与非法 `uplo` / `alpha=nullptr` / `k<0` 共存时会被误判为 `SUCCESS`；② `n == 0` 时 `A`/`x`/`y` 允许为空指针（语义上不引用它们），故其判空必须放在 `n == 0` **之后**，否则会把合法的空操作误判为 `INVALID_VALUE`；③ 该顺序与仓内 golden `test/sbmv/ssbmv/ssbmv_golden.h:25-40` 及 Netlib quick return 语义一致。

> 与仓内既有实现的一处**有意偏差（须在评审时知悉，不得判为缺陷）**：第 3 步非法 `uplo` 返回 `INVALID_ENUM`，而仓内先例（`blas/sbmv/arch35/ssbmv_host.cpp`、`blas/symv/arch22/ssymv_host.cpp`）返回 `INVALID_VALUE`。本算子按**任务书 §2.4/§2.5 与配套评测集 `TC_ED_156`（`uplo=999`）**取 `INVALID_ENUM`；测试链的负向路径**只比较返回码与 CSV `expect_result` 列、不调用 golden**，故该差异不产生判定冲突。

### 数学公式的向量化分解与算术域约定

对每个固定的带内行下标 `r`，令 `ttRow_r[c] = a_c[r]`（长度 `C` 的**连续向量**，由片上 `Gather` 从带状 tile 逐行提取），则每个 `(项, r)` 对退化为一次长度为 `cb` 的 `Mul` + `Add`，**全程无归约、无原子、无滑动窗口、无跨 lane 通信**；每 chunk 的向量指令数为 `2×(2k+1) = 4k+2` 条，目的地址均落在核内累加器 `accY + (i − r0)`。

各 `(项, r)` 的有效存储列区间（`i ∈ [r0, r1)` 且 `i ∈ [0, n)`，下标越界由裁剪自动去掉）：

| uplo / 项 | 输出行 `i` | 有效列区间 `c`（整数域；实现须把 `a−b` 换成 `subSat(a,b)`） | `x` 取用下标 `j` |
| --- | --- | --- | --- |
| UPPER T1 | `c − k + r` | `[max(cLo, r0+k−r), min(cHi, r1+k−r))` | `c` |
| UPPER T2 | `c` | `[max(r0, k−r), r1)` | `c − k + r` |
| LOWER T1 | `c + r` | `[max(cLo, subSat(r0,r)), min(cHi, subSat(r1,r), subSat(n,r)))` | `c` |
| LOWER T2 | `c` | `[r0, min(r1, subSat(n,r)))` | `c + r` |

其中 `cLo = max(0, r0−k)`、`cHi = min(n, r1+k)`、`C = cHi − cLo`。

**算术域声明（编码必须遵守的硬约束）**：上表是**整数域（ℤ）**下的闭开区间表达。`r0−r`、`r1−r`、`n−r` 三项在 `r >= r0`（LOWER T1 必然出现）或 `r >= n`（`k >= n` 时必然出现）时在 ℤ 下为负；而 kernel 内 `r0/r1/cLo/cHi/c/r` 全部是 `uint32_t`，**无符号减法会下溢成巨大正数**，造成两类**不报错的静默错误**：① 区间被误判为空 ⇒ 贡献被静默丢弃（结果错）；② 区间上界被放宽 ⇒ lane 越出 `xWin` ⇒ 越界读。故实现**一律使用饱和减法**：

```
subSat(a, b) ≡ ((a > b) ? (a − b) : 0U)          // uint32 安全，不产生下溢
```

代入后**无需任何 `if` 特判**且语义精确：`r >= r1` 或 `r >= n` ⇒ 对应上界为 0 ⇒ 区间自动为空；`r >= r0` ⇒ 下界自动退化为 `cLo`。**UPPER 两式在 uint32 下天生安全**（循环恒有 `r <= k`，按 `(r0+k)−r`、`(r1+k)−r`、`k−r` 的结合顺序求值即无下溢，无需 `subSat`）。

`x` 窗口（UB 内只保留本核需要的一段，长度 `Lx`）：

| uplo | `xBase` | 用到的 `j` 范围 | `Lx` | T1 的 lane 偏移 | T2 的 lane 偏移 |
| --- | --- | --- | --- | --- | --- |
| UPPER | `max(0, cLo − k)` | `[xBase, cHi)` | `<= C + k` | `k`（当 `cLo >= k`） | `r` |
| LOWER | `cLo` | `[cLo, cHi)` | `C` | `0` | `r` |

lane 偏移统一实现为 `lane(j) = j − xBase`（`cLo < k` 时 `xBase` 被 0 截断，偏移随之自动调整），保证所有 lane 下标非负且不越出 `xWin`。**该保证以上表 LOWER 两式的 `subSat` 写法为前提。**

## 算子实现

### 工程结构与文件清单

```
ops-blas/
├── blas/sbmv/arch22/                     # 新增
│   ├── ssbmv_host.cpp                    # 句柄式接口 + 参数校验 + Tiling 计算 + kernel launch
│   ├── ssbmv_kernel.cpp                  # kernel 实现（__global__ __aicore__）
│   ├── ssbmv_kernel.h                    # kernel 声明与 tiling 结构引用
│   └── ssbmv_tiling_data.h               # SsbmvTilingData 结构 + 核数/UB 上限常量
├── blas/sbmv/README.md                   # 补齐 A2/A3 产品支持与接口说明（纯追加）
├── include/cann_ops_blas.h               # 声明已存在，不改
└── test/sbmv/ssbmv/
    ├── CMakeLists.txt                    # arch22 分支接入 GTest 链
    ├── ssbmv_golden.h / ssbmv_param.h    # 既有资产，仅做向后兼容的追加式扩展
    └── arch22/                           # 新增
        ├── ssbmv_test.cpp                # CSV 驱动 GTest（含精度模式硬断言）
        ├── ssbmv_test.csv                # 任务配套提供集（逐字保留）
        ├── ssbmv_supp.csv                # 自行补充集
        ├── ssbmv_whitebox.csv            # 白盒补全集
        └── tools/                        # 补充集生成脚本、性能采集脚本
```

### 实现方案

#### 3.2.1 host侧设计：

Host 侧职责为**参数校验 → Tiling 计算 → kernel 下发**，不含任何设备同步。

##### 1. 参数校验

按上表 11 步全序校验（伪码与返回码见「参数与异常行为」节），全部通过后计算 Tiling 并 launch。`alpha` / `beta` 为 **Host 指针**，host 侧直接解引用后以标量形式卷入 Tiling。

##### 2. Tiling 参数（卷入 `SsbmvTilingData`）

| 参数 | 类型 | 含义 | 来源 / 约束 |
| --- | --- | --- | --- |
| `n` / `k` / `lda` | uint32 | 矩阵阶数 / 半带宽 / A 前导维 | 入参 |
| `incx` / `incy` | int32 | x / y 步长（含负值语义） | 入参（`==0` 已在第 6 步拦截） |
| `useCoreNum` | uint32 | 实际启用核数 | `min(可用 AIV 核数, ceil(n / R_target))`，见「核数与 R 的选取规则」 |
| `rowsPerCore` | uint32 | 除末核外每核行数 `R` | `ceil(n / useCoreNum)` |
| `lastCoreRows` | uint32 | 末核行数（可为 0 表示该核不参与） | `n − rowsPerCore·(useCoreNum−1)`；**kernel 必须消费该字段**（其核号映射用它算末核 `r1`），否则它是不可达的冗余配置项 |
| `cbCols` | uint32 | 列块宽度 `CB` | 按 UB 预算反推，见「UB 预算与 `CB` 选取」 |
| `alpha` / `beta` | float | 标量（host 指针已解引用） | 入参 |
| `isUpper` | uint32 | `uplo` 的 kernel 侧表示（1=UPPER, 0=LOWER） | 入参 |
| `alphaIsZero` | uint32 | `alpha == 0` 标志（kernel 侧直接短路 A/x 路径） | 入参 |
| `xUnitStride` / `yUnitStride` | uint32 | `|incx| == 1` / `|incy| == 1` 标志 | 入参；为 0 时 kernel 走标量寻址 |
| `xSign` / `ySign` | int32 | `incx` / `incy` 的符号（+1 / −1） | 入参；决定 x/y 的物理下标映射 |

##### 3. 核数与 `R` 的选取规则

目标是在 **halo 冗余可控**（`(R+2k)/R <= 1 + 2/HALO` 即 `R >= HALO·k`）与 **单核工作量不过小**（`R >= SBMV_R_MIN`）之间取平衡。**设计取值（本算子自设的策略常量，非继承仓内兄弟算子）**：

| 常量 | 取值 | 含义 |
| --- | --- | --- |
| `SBMV_POLICY_CORE_NUM` | **32** | host 侧自行设定的核数上限（host 策略值，在 `ssbmv_host.cpp` 内夹紧） |
| `SBMV_R_MIN` | **64** | 单核最小行数 |
| `SBMV_HALO_FACTOR` | **171/64 = 2.671875** | halo 冗余因子目标（取二进制可精确表示的整数比，避免浮点常量引入不确定的 `ceil` 边界） |
| `SBMV_MAX_CORE_NUM` | **50** | kernel 侧越界保护上限（注释明确“只做越界保护，不参与切分决策”）；实际核数上界 24 `< 50`，覆盖有余量 |

规则（host 实现）：

1. `availableCoreNum = GetAivCoreCount()`，为 0 时取 1；
2. `R_target = max(SBMV_R_MIN, ceil(SBMV_HALO_FACTOR · k))`（整数式 `max(64U, CeilDiv<uint32_t>(171U·k, 64U))`；**`ceil` 语义为必需**，截断会在 case2 给出错误核数）；
3. `useCoreNum = min(availableCoreNum, SBMV_POLICY_CORE_NUM, ceil(n / R_target))`，且 `useCoreNum >= 1`；
4. `rowsPerCore = ceil(n / useCoreNum)`，`lastCoreRows = n − rowsPerCore·(useCoreNum−1)`；
5. `R == 0` 的核（含 `lastCoreRows == 0` 的冗余核）**直接 return**，不执行任何搬入/搬出/同步，避免空核越界访问。

**取舍说明**：本算子的瓶颈是固定开销与指令发射而非带宽，故优先用满并行度——`R` 越小 halo 冗余越高（读取量上升），但换回固定开销的摊薄。反向方案（把核数压回 8、`R = 512`）在形状 4 上实测内核耗时 70.549 µs、高于标杆 47.74 µs（缺口 1.48×），读放大虽降到 1.25 却换不回固定开销的摊薄，故**不采用**。

**`SBMV_R_MIN = 64` 是本轮之后的核数瓶颈（据实登记）**：`k = 16` 的 case1/case3 若要与 24 核对齐，需 `R_target ∈ {43, 44}`，而 `R_target = max(64, 43) = 64` 被 `SBMV_R_MIN` 钉死 ⇒ 二者只能到 16 核。本设计**不为四形状核数一致而改 `SBMV_R_MIN`**：降它会在提高核数的同时推高 halo 冗余与全算子读取量，收益未经实测、代价已可量化，如需变更须回到需求侧重新审定。

##### 4. UB 预算与 `CB` 选取

**UB 占用硬上限取 128 KB**（仓内 `blas/common/helper/kernel_constant.h` 的 `UB_SIZE = 248*1024` 属通用/arch35 口径，**不适用于 arch22**；dav-2201 单核 AIV UB 容量为 192 KiB，本设计取更保守的 128 KB 以保证必然可落地）。`CB` 由该预算反推：

```
UB_budget      = 128 * 1024                     // bytes，本方案硬上限
per_col_bytes  = 2 * PITCH * 4                  // bandTile 双缓冲（BUFFER_NUM = 2）的「每列」成本
fixed(CB)      = idx(CB) + routeFixed(CB) + const0
idx(CB)        = 5 * (CB + WIDEN) * 4           // offVec / seqVec / gathBuf / xStage / mulBuf 各 (CB+WIDEN) 个元素
routeFixed(CB) = useColGuard ? 2*LEAD_COL*PITCH*4                  // 路线①：bandTile 每 buffer 多留 LEAD_COL 列前导栅格
                             : OFF_Q_NUM*align(CB+WIDEN,8)*4       // 路线②：OFF_Q_NUM 条夹紧偏移向量（单 slot 32B 对齐）
const0         = (LEAD_X + Lx)*4 + 2*R*4 + Y_ELEM_CHUNK*Y_BLOCK_FLOATS*(4+4)   // xWin 前导 + accY + yBuf + (yExp + yPat)
UB(CB)         = per_col_bytes * CB + fixed(CB)
CB             = 不动点迭代求解 CB = floor((UB_budget − fixed(CB)) / per_col_bytes)
CB             = max(CB, 1)
CB             = 1u << floor(log2(CB))          // 向下取到 2 的幂（硬步骤）
CB             = min(CB, 4095, C)               // 4095 = DataCopyPad 的 blockCount 上限；C = 本核列数
CB             = (C <= 4095 && UB(C) <= UB_budget) ? C : CB   // 例外：整条带一次装得下 ⇒ 取满 CB = C
```

**取值次序的硬性要求**：① 「向下取 2 的幂」必须在 `min(CB, 4095, C)` **之前**——若放在截断之后，整条带可一次装完的形状会被 2 的幂规则砍半、产生参差不齐的 chunk，白白牺牲 `Gather` 的迭代摊销；② `routeFixed` 取**该点实际选中路线**的分配量（路线② 的单 slot 必须 32B 对齐，其实配量恒不小于判据右端，故取「按实际选路」才是安全侧）；③ 末行例外是**一般形状**下保证“整条带一次装完、不产生 ragged chunk”的唯一机制，不可删。

**"向下取 2 的幂"为何是硬步骤**：若不执行该行，`CB` 会贴在 128 KiB 边界上（case4 补全模型下不动点解 201 ⇒ `UB = 130,936 B`，**仅余 136 B**），任何附加 buffer、对齐抬升或口径微调都会直接越限。

##### 5. kernel launch 与 stream

- stream 来源：`aclblasHandle_t handle` 内含 stream 字段（`aclblasSetStream(handle, stream)` 完成绑定），**复用调用方 stream，不自建 stream、不做 host 侧同步**；
- `ssbmv_kernel<<<numBlocks, nullptr, useStream>>>(...)`，`numBlocks` **恒等于 `tiling.useCoreNum`**（避免 launch 与 tiling 不一致导致空核越界或核数不足）；
- `aclblasSsbmv` 返回时只保证 kernel **已入队**；调用方在读取 `y` 之前必须自行同步该 stream（`aclrtSynchronizeStream`，测试侧用 `aclrtSynchronizeDevice` 亦可）；
- 全部参数校验在 launch **之前**同步返回错误码；kernel 执行期的设备错误由调用方在同步点获得；
- 核内无跨核同步、无全局状态、无静态 workspace ⇒ 不同 stream 上的多次调用互不干扰。

#### 3.2.2 kernel侧设计：

Kernel 形式为 `__global__ __aicore__ void ssbmv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, const SsbmvTilingData& tiling)`，`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`。kernel 侧**不做核数/行数分配决策**（那是 host 职责），只做由核号到地址的纯算术推导。

##### 1. 多核切分：按 `y` 的行区间切分

| 候选切分维度 | 评估 | 结论 |
| --- | --- | --- |
| **按 `y` 的行区间切分（row partition）** | 各核写互不相交的 `y` 行；`A·x` 各行只依赖 `x` 的 `[max(0,i−k), min(n,i+k))` 与 A 的对应存储列。核间无写冲突、无读-写依赖、无需同步、无需原子；代价是 halo 列冗余（每核需读 `R+2k` 列，冗余因子 `(R+2k)/R`） | **采用** |
| 按列区间切分 + `SetAtomicAdd` 归约 | 冗余为 0，但需跨核原子累加，`beta·y` 需额外同步或由单核后处理；原子加对 32B 存储对齐有额外约束，且 `beta` 与累加存在竞态 | **不采用**（记录为备选） |
| 按 A 的存储行（带内位置 `r`）切分 | 每个 `r` 的贡献是一段长度为 `n` 的向量乘，切分后各核写同一 `y` 的不同 part ⇒ 仍需归约；且带内行数仅 `k+1`，case1/case3（`k=16`）只有 17 份，并行度不足 | **不采用** |

**核号映射（核内）**：

```
blk  = GetBlockIdx();
r0   = blk * tiling.rowsPerCore;
r1   = (blk + 1 == tiling.useCoreNum) ? (r0 + tiling.lastCoreRows) : (r0 + tiling.rowsPerCore);
if (r1 <= r0) return;                              // 冗余核（含 lastCoreRows == 0）：不搬运、不计算、不写回
R    = r1 - r0;
cLo  = (r0 > k) ? (r0 - k) : 0;                    // max(0, r0-k)
cHi  = (r1 + k < n) ? (r1 + k) : n;                // min(n, r1+k)
C    = cHi - cLo;
K1   = k + 1;
PITCH = ((K1 + 7U) / 8U) * 8U;                     // align(K1, 8)，保证 UB 内每列 32B 对齐
```

> **`lastCoreRows` 必须被消费**：末核终点由 `r0 + tiling.lastCoreRows` 给出，而**不是**写成 `r1 = n`。理由：与 host tiling 严格自洽（`useCoreNum`/`rowsPerCore`/`lastCoreRows` 三者互为定义），避免 kernel 侧出现第二套行划分口径；核内所有循环边界以本核 `r0/r1` 为准，**唯一需要全局 `n` 的地方是列上界裁剪 `cHi = min(n, r1+k)`**。

**halo 列范围（每核私有，决定读多少 A）**：由带状存储 + 对称语义，行 `i` 的 `y[i]` 只依赖存储列 `c ∈ [max(0, i−k), min(n, i+k+1))`，故核内需读的存储列区间为 `cLo = max(0, r0−k)`、`cHi = min(n, r1+k)`；首末核的裁剪由该式自动完成，无需特判分支。

**核间隔离性论证（无竞态）**：

1. **读**：各核只读 A 的 `[cLo, cHi)` 列与 x 的窗口段（UPPER 为 `[xBase, cHi)`、`xBase = max(0, cLo−k)`）；A 为只读输入，多核重复读同一列是安全的（只读冗余，无写）；
2. **写**：各核只写 `y` 的 `[r0, r1)` 行，各核区间并集恰为 `[0, n)` 且两两不相交 ⇒ 无写冲突；
3. **无跨核归约**：每个输出行由唯一核完整算出（含 `beta·y` 部分），不需要原子加、不需要 `SyncAll`、也不需要“先算 A·x 再统一缩放”的两阶段；
4. **结论**：本方案**不需要任何跨核同步原语**（kernel 全文无 `SyncAll` / `CrossCoreSetFlag` / `CrossCoreWaitFlag`），只有核内流水 `SetFlag` / `WaitFlag`。

##### 2. Buffer 规划（单核视角）

| Buffer | 用途 | 大小（float/uint32 个数） | 位置与缓冲份数 |
| --- | --- | --- | --- |
| `bandTile` | 带状列块：`bandTile[c·PITCH + r] = A_store[r + lda·(cLo+chunk+c)]` | `(guardCols + CB) · PITCH`（`guardCols = 7`（路线①）或 `0`（路线②）；逻辑内容 `CB · PITCH`） | UB，`TQue<VECIN>`，**BUFFER_NUM = 2**（chunk 间 ping-pong，使 chunk `i+1` 的 MTE2 搬运与 chunk `i` 的向量计算重叠） |
| `offVec` | `Gather` 字节偏移向量：`offVec[t] = t·PITCH·4`（对本核所有 `r`、所有 chunk 复用，**只建一次**） | `CB + WIDEN`（uint32） | UB，`TBuf<VECCALC>`，1 份 |
| `gathBuf` | `Gather` 目的缓冲，逐 `r` 复用：`gathBuf[c] = bandTile[c·PITCH + r] = ttRow_r` | `CB + WIDEN` | UB，`TBuf<VECCALC>`，1 份 |
| `mulBuf` | `Mul` 的临时结果（`Add` 直接读它写入 `accY`，避免原地 `Mul` 破坏 `gathBuf` 的复用） | `CB + WIDEN` | UB，`TBuf<VECCALC>`，1 份 |
| `seqVec` | x 侧 `Gather` 的字节偏移向量：`seqVec[t] = t·4` | `CB + WIDEN`（uint32） | UB，`TBuf<VECCALC>`，1 份 |
| `xStage` | x 侧 `Gather` 目的缓冲（把 `xWin` 中窗口对齐后的那段取成连续向量） | `CB + WIDEN` | UB，`TBuf<VECCALC>`，1 份 |
| `xWin` | x 窗口（已预乘 `alpha`），覆盖逻辑下标 `[xBase, xBase+Lx)` | `LEAD_X + Lx` | UB，`TQue<VECIN>`，**BUFFER_NUM = 1**（全 kernel 只搬入一次、跨 chunk 只读复用，无 ping-pong 收益） |
| `accY` | `A·x` 部分和累加器，基址对应全局行 `r0`（下标 `i` ↔ 全局行 `r0+i`） | `R` | UB，`TBuf<VECCALC>`，1 份；**须显式 `Duplicate(accY, 0.0f, R)` 清零** |
| `yBuf` | y 的搬入 / 缩放 / 搬出缓冲 | `R` | UB，`TQue<VECOUT>`，**BUFFER_NUM = 1** |
| `yExp` / `yPat` | `|incy| != 1` 或 `incy < 0` 时 y 搬出用的“逐元素展开”缓冲与其字节偏移向量（每元素独占一个 32B 块，配合 `DataCopyPad` 的掩码写出） | `Y_ELEM_CHUNK × Y_BLOCK_FLOATS = 64 × 8 = 512` 各一份（合计 **4 KB**） | UB，`TBuf<VECCALC>`，各 1 份 |
| `offQ` | 路线②的 8 条夹紧偏移向量（`offQ[q][t] = max(t−q,0)·PITCH·4`） | `OFF_Q_NUM × align(CB+WIDEN, 8) = 8 × align(CB+8, 8)`（uint32） | UB，`TBuf<VECCALC>`，1 份，**仅在路线②时分配** |

符号常量（实现既有取值）：`WIDEN = 8`（各索引/中间向量预留的额外 lane 数，供窗口下取整到 8 的倍数后仍按元素 `t` 直接索引）、`LEAD_X = 8`（`xWin` 的前导 lane 数）、`LEAD_COL = 7`（路线①下 `bandTile` 每 buffer 多留的列数）、`OFF_Q_NUM = 8`、`Y_ELEM_CHUNK = 64`、`Y_BLOCK_FLOATS = 8`。

**规模上界（逐 case 按上述规则反推，全部为复算值）**：

| case | `R` | `C` | `K1` | `PITCH` | 不动点解 → 取 2 的幂 | 例外③ | 最终 `CB` | UB 占用（路线① / 路线②） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case1 / case3 | 64 | 96 | 17 | 24 | 587 → 512 | 触发（`UB(96) = 26,944 B <= budget`） | **96**（= C） | 26,944 / 28,928 B |
| case2 | 86 | 150 | 33 | 40 | 362 → 256 | 触发（`UB(150) = 58,944 B <= budget`） | **150**（= C） | 58,944 / 61,824 B |
| **case4（峰值）** | 171 | 299 | 65 | 72 | 201 → 128 | 不触发（`UB(299) = 189,344 B > budget`） | **128** | **87,428 / 87,748 B** |

case4 峰值的逐项展开（路线①，按实现的实际分配量）：`bandTile ×2 = 2×(7+128)×72×4 = 77,760` + `offVec/seqVec/gathBuf/xStage/mulBuf` 各 `(128+8)×4 = 544`（合计 2,720）+ `xWin = (8+363)×4 = 1,484` + `accY 684` + `yBuf 684` + `yExp/yPat 4,096` = **87,428 B**（≈ 128 KB 自设上限的 66.7%，余 43,644 B；路线②下余 43,324 B）。

**UB 预算即本方案的硬门禁**：任何新增 buffer（或把某 buffer 改为 2 份）都必须先代入上式复核 **case4** 与该 case 的路线② 余量，不得只看“逻辑长度”。若把 `xWin`/`yBuf` 各按 2 份的保守口径计，case4 为 89,564 B（87.5 KiB），仍在 128 KB 以内。

##### 3. 数据流（单核）

```
GM(A) --DataCopyPad(GM->UB, blockCount=CB, blockLen=K1*4B, srcStride=(lda-K1)*4B [byte, gap], dstStride=0 [32B dataBlock, gap])--> bandTile
                                                              |
                         Gather(gathBuf, bandRaw, offVec, srcBaseAddr, wlen) -- 对 r = 0..k 复用 --> ttRow_r
                                                              |
           xWin --Muls(xWin, xWin, alpha)--> 预乘           Mul / Add --> accY
                                                              |
GM(y) --DataCopyPad--> yBuf --Muls(beta) / Duplicate(0)--> Add(yBuf, yBuf, accY+off, R) --DataCopyPad--> GM(y)
```

**数据流要点**：

- **搬入方向**：A 的访问方向是“按存储列、列内连续”——`blockLen = K1·4` 字节、`srcStride = (lda−K1)·4` 字节，源侧读取为“每列一段连续、列间跳过 `lda` 补位”，**不产生 HBM 放大**。
- **`dstStride` 取 0（关键，勿改）**：`srcStride` / `dstStride` 的官方语义是**相邻数据块之间的“间隔（gap）”**（前一块**结束地址**与后一块**起始地址**之差），不是“跨步”；GM 侧单位 **byte**、UB 侧单位 **32B dataBlock**。且 `blockLen` 非 32B 对齐时**每个块都在 UB 侧被补齐到 32B**，落点即 `align(K1,8)·4 = PITCH·4` 字节/列 ⇒ 列在 UB 中**天然按 `PITCH` 紧邻排布**，`dstStride` 必须取 **0**；若按“跨步”语义传入 `PITCH/8`，等于在每列之间**再插一个 `PITCH·4` 的 gap**，列落点变成 `2·PITCH·4`，而 kernel 仍按 `c·PITCH + r` 寻址 ⇒ **静默错读（不报错）**。列步进由 GM 侧 `srcStride` 承担。
- **搬出方向**：`y` 按核内行区间一次性搬出；`yUnitStride && ySign > 0` 时连续（`DataCopyPad`，`srcStride/dstStride = 0`）；`|incy| != 1` 或 `incy < 0` 时改为 **“V 管线展开 + MTE3 掩码搬出”**（每元素独占一个 32B 块，`blockLen = 4B`、`dstStride = (|incy|−1)·4B`）——**不用 S 管线的标量写 GM**（实测：S 管线的 GM 标量写以 128B cacheline 为粒度做读-改-写，行切分下相邻核共享 cacheline 时会互相回退）。
- **不搬入**：不读 A 的未存储半带（对称性在索引层解析，不复制数据）；`beta == 0` 时**不读 `y`**。
- **对齐**：UB 侧所有操作数起始地址 32B 对齐；GM 侧无对齐要求。

##### 4. 逐 chunk 主循环

```
Duplicate(accY, 0.0f, R);        // 累加器必须显式清零为 +0.0；alphaIsZero 时整条 A/x/accY 路径都不执行
for (ch = 0; ch < C; ch += CB) {
    cb = min(CB, C - ch);                        // 本 chunk 的实际列数
    c0 = cLo + ch;                               // 本 chunk 首列的全局列号

    // (1) 搬入本 chunk 的 A 存储列（列内 K1 个元素连续、列间跳过 lda-K1）
    DataCopyExtParams p{ (uint16_t)cb, (uint32_t)(K1*sizeof(float)),
                         (uint32_t)((lda - K1)*sizeof(float)),   // srcStride：GM 侧 byte 单位的 gap
                         0U, 0U };                               // dstStride：UB 侧 32B dataBlock 单位的 gap，取 0
    DataCopyPad(bandTile, aGm[(uint64_t)c0 * lda], p, padParams);
    MTE2_V 同步（或 TQue::Deque）

    // (2) 逐带内行 r 提取转置行 + 两条向量乘加
    for (r = 0; r <= k; ++r) {
        // dst 基址必须 32B 对齐 ⇒ 把 dst 窗口向下取整到 8 的倍数（pre = dstStart & 7），
        // 被拉进来的前缀 lane 由 Duplicate(mulBuf, 0.0f, pre) 清零后参与 Add（+0.0 不改变 accY）。
        // 源侧不能用“张量视图 + lane 偏移”，必须走“原始 buffer + srcBaseAddr(byte) + 偏移向量”：
        //   band 侧：srcBaseAddr = r*4 + (w0 - c0 + 前导量)*PITCH*4
        //   x   侧：Gather(xStage, xWinRaw, seqVec, xOff, wlen)，seqVec[t] = t*4

        // T1：dst = accY + (i − r0)，i = c−k+r (UPPER) 或 c+r (LOWER)
        lo1 = isUpper ? max(cLo, r0 + k - r) : max(cLo, subSat(r0, r));
        hi1 = isUpper ? min(cHi, r1 + k - r) : min(cHi, min(subSat(r1, r), subSat(n, r)));
        lo1 = max(lo1, c0);  hi1 = min(hi1, c0 + cb);            // 与 chunk 区间求交
        if (lo1 < hi1) AccumTerm(lo1, hi1, dstStart1, laneLo1, r);

        // T2：dst = accY + (c − r0)，x 取 j = c−k+r (UPPER) 或 c+r (LOWER)
        lo2 = isUpper ? max(r0, k - r) : r0;
        hi2 = isUpper ? r1 : min(r1, subSat(n, r));
        lo2 = max(lo2, c0);  hi2 = min(hi2, c0 + cb);
        t2Valid = isUpper ? (r < k) : (r > 0);                   // 对角元只由 T1 承载
        if (t2Valid && lo2 < hi2) AccumTerm(lo2, hi2, dstStart2, laneLo2, r);
    }
}
// (3) y 的 beta 处理与写回（见「边界语义实现」）
```

`AccumTerm` 内部为：窗口对齐（`pre/w0/wlen`）→ 带内 `Gather` → `Mul` → 前缀清零 → `Add` 累入 `accY`。**T1 与 T2 各自发一次带内 `Gather`**（两者窗口不同：行区间不同，且 dst 窗口下取整后的 `pre`/`w0` 也不同），故每 chunk 的带内 `Gather` 次数为 `(k+1) + k`（T2 排除对角 `r`）。本设计**不常驻完整的 `(k+1)×CB` 转置 tile**（`gathBuf` 单行复用，UB 占用由 ~33 KB 压到 ~0.5 KB），这是允许更大 `CB` 的前提。

##### 5. 实现约束（写进代码注释，避免下游误改）

| # | 约束 | 原因 |
| --- | --- | --- |
| 1 | `bandTile` 的所有访问偏移（chunk 起点 `ch·PITCH`、列内 `c·PITCH + r`）均须 32B 对齐 | `PITCH` 取 8 的倍数 ⇒ `PITCH·4` 是 32B 倍数；**改成非 8 倍数即破坏对齐** |
| 2 | `DataCopyPad` 的 `srcStride` 单位是 **byte**、`dstStride` 单位是 **32B dataBlock** | 两方向单位不同，写反会导致静默错读（最危险的一类缺陷） |
| 3 | `offVec` 全程只建一次、所有 `r` 与所有 chunk 复用 | 减少标量/重排开销；其内容不随 chunk 变化 |
| 4 | `gathBuf` 同一时刻只服务一个 `r`，`Mul` 必须紧随其 `Gather` 之后 | 避免额外的常驻转置 tile |
| 5 | `mulBuf` 为 `Mul` 的落点，`Add` 直接读它写入 `accY` | 避免原地 `Mul` 到 `gathBuf`（会破坏 T2 的复用） |
| 6 | 不手工写 `SetFlag`/`WaitFlag` 用于 `TQue` 管理的 buffer | 「一 buffer 一机制」：`bandTile`/`xWin`/`yBuf` 交给 `TQue` 自动插同步；`offVec`/`gathBuf`/`mulBuf`/`accY` 为 `TBuf`，只用显式成对的手工 flag，两者不交叉 |
| 7 | **禁止对 `r0−r`、`r1−r`、`n−r` 做 uint32 无符号减法**：LOWER 两式的列区间上/下界必须走 `subSat` | `r >= r0` 时 `r0−r` 下溢 ⇒ 区间被静默判空 ⇒ 丢贡献（结果错）；`k >= n` 时 `n−r` 下溢 ⇒ 上界被放宽 ⇒ `xWin` 越界读。**两类均不报错** |
| 8 | **A 访问上界不变式**：全部 GM 访问的偏移 `off = (cLo+chunk+c)·lda + rIdx <= (n−1)·lda + k <= lda·n − 1`，即 **上界 = `lda×n`** | `lda >= k+1` 时恒不越界；若下游改用带 stride 的搬运变体或改动列区间裁剪，**必须重新证明本式** |
| 9 | 一切向量操作数（`Gather` 的 `dst`/`src`/偏移向量、`Mul`/`Add`/`Duplicate` 的 `dst`）起始地址必须 32B 对齐 ⇒ dst 窗口一律**向下取整到 8 的倍数**，前缀 lane 由 `Duplicate(mulBuf, 0.0f, pre)` 清零；源侧前导量只由 `srcBaseAddr`（byte）或偏移向量承担，**不得同时使用张量视图** | 非 32B 对齐的 `Gather` 视图会触发设备异常；`+0.0` 前缀不改变 `accY`（其初值 `+0.0`、只经 `Add` 累加，恒不为 `−0.0`） |
| 10 | A 搬入的 `dstStride` 取 **0** | `blockLen` 非 32B 对齐时 UB 侧每块被补齐到 32B ⇒ 落点恒为 `PITCH·4`；传 `PITCH/8` 会额外插入 gap ⇒ 静默错读 |
| 11 | **T2 的 `r` 必须裁掉对角 `r`**：`t2Valid = isUpper ? (r < k) : (r > 0)` | 该 `r` 上 T1 与 T2 的有效行区间完全相同 ⇒ 对角元被累加两次（结果错、不报错）；UPPER 的对角元在 `r=k`、LOWER 在 `r=0`，**只由 T1 承载** |

##### 6. `lda` 填充（padding）的处理方式

| 情形 | `srcStride` | 处理方式 |
| --- | --- | --- |
| `lda == K1`（紧凑） | `0` | block 之间紧邻，`DataCopyPad` 基础形式即正确；UB 内每列占 `PITCH >= K1` 个 float，仅列尾有 32B 对齐补位 |
| `lda > K1`（列间有填充） | `(lda−K1)·4 > 0`（byte） | 由 block 间跳转跳过填充，**填充字节不进入 UB**（`dstStride = 0`，各 block 因 `blockLen` 非 32B 对齐而被补齐到 32B ⇒ 落点恒为 `PITCH·4`）。**不需要** ascend950 的 `PaddingMode::Compact` 参数（A2/A3 不支持且无必要） |
| 越界风险 | — | 由 `cHi = min(n, r1+k)` 保证 `c0 + cb <= cHi <= n`，最后一列结束于 `(n−1)·lda + K1 <= lda·n`（因 `K1 <= lda`），**恒不越界** |

> 结论：本方案**不存在**“多填充”场景，`lda` 填充全部由 `srcStride` 在源侧跳过；UB 内的唯一冗余是每列尾部不足 32B 的对齐补位（`PITCH − K1 <= 7` 个 float），且该补位从不参与向量计算。

##### 7. 边界语义实现

| 场景 | 语义 | 实现路径 | 精度 / 一致性要求 |
| --- | --- | --- | --- |
| `n == 0` | 合法空操作 → `SUCCESS` | **host 侧直接 return**，不启动 kernel、不建 Tiling、不触碰 `A`/`x`/`y` 设备指针（第 8 步） | 无数值行为 |
| `alpha == 0` | 跳过矩阵乘（**不引用 A 与 x**，A/x 可为 `nullptr`）：`beta == 1` → `y` **位不变**（quick return）；`beta == 0` → `y` 各元素精确置 `+0.0`；其余 `beta` → `y := beta·y`（读改写） | host 置 `tiling.alphaIsZero = 1` → kernel **跳过整条 A/x 路径**：不搬 A、不搬 x、不建 `accY`、**不执行任何 `Add(accY)`**；收尾只走本表的 y 分支（`beta == 1` 时连 y 的搬入与 `Muls` 一并跳过） | **逐位一致（bit-exact）**。依据：Netlib `ssbmv` 在 `ALPHA == 0` 时走**提前返回分支**（`BETA == 0` 时 `Y(I) = ZERO`，否则 `Y(I) = BETA*Y(I)`；另有 `ALPHA==0 且 BETA==1` 的更早快速返回），**全程不引用 A/x、不做任何累加**。**本算子唯一的 bit-exact 强约束**：若对 `y = −0.0` 的元素执行 `Add(±0.0)`，`−0.0 + 0.0 = +0.0` 会翻转符号位而破坏位精确，故那条 `Add` 必须彻底跳过 |
| `beta == 0`（`alpha != 0`） | `y` 无需是合法输入；实现**先向 y 写入 0 再累加、不读 y**，`y` 中的 NaN/Inf 不传播 | `Duplicate(yBuf, 0.0f, R)` **直接置零、不搬入 y**；随后**仍执行** `Add(yBuf, yBuf, accY+off, R)` | 与 golden 一致。`0.0f + accY` 的逐位安全性：`accY` 初值为 `+0.0`、此后**只经 `Add` 累加**，IEEE-754 round-to-nearest 下两个零相加亦得 `+0.0` ⇒ `accY` **恒不为 `−0.0`**，故 `0.0f + accY` 逐位等于 `accY`（若将来改动 `accY` 初值或引入取负/`Muls` 路径，本行须重证） |
| `beta == 1` | `y := y + alpha·A·x` | **跳过 `Muls`**，直接 `Add(yBuf, yBuf, accY+off, R)` | 避免 `1.0f` 乘法引入任何（即便恒等的）额外舍入（`x·1.0f ≡ x`，`−0.0` 亦保持符号） |
| 其它 `beta` | `y := beta·y + alpha·A·x` | 搬入 y → `Muls(yBuf, yBuf, beta, R)` → `Add(yBuf, yBuf, accY+off, R)` | 容差内 |
| `incx == +1` | — | x 窗口一次 `DataCopyPad` 搬入 | 与 golden 一致 |
| `incx == −1`、`\|incx\| != 1`、或 `incx < 0` | — | **搬入**走标量回退（`GetValue` → `SetValue` → `FenceSToV()`）：负步长在 GM 中是**逻辑逆序**，而 `DataCopyPad` 的 `srcStride` 只能表达正向的块间间隔，无法表达“逐元素反向” | 数值路径不变（仅搬运方式不同） |
| `\|incy\| != 1` 或 `incy < 0` | — | **搬出不得**用 `GlobalTensor::SetValue` 写 GM，改走“V 管线 `Gather` 展开 + MTE3 掩码搬出” | 数值路径不变 |

**`alpha` 的注入方式**：`alpha` 通过一次 `Muls(xWin, xWin, alpha, Lx)` **折叠进 x 窗口**（每个 chunk 只需 1 次），而不是在每个 `Mul` 前对 `ttRow_r` 乘 alpha——这样每 chunk 只多 1 条指令，且与 Netlib 的 `temp = alpha·x(j)` 顺序同构，利于容差匹配。

**数值路径声明**：本分解的累加顺序与 `cblas_ssbmv`（按输出行、按列递增累加）**不同**，故一般情形**不保证逐位一致**，依赖 MIXED_TOLERANCE 判定；**唯一要求逐位一致的路径是 `alpha == 0`**。

##### 8. 核内同步与 dav-2201 实测能力边界

核内只采用三对**白名单**方向的手工事件：`MTE2→V`（搬入→计算）、`S→V`（索引/模式向量标量写→计算）、`V→MTE3`（计算→搬出），均由 `GetTPipePtr()->FetchEventID(...)` 动态取号、成对使用；跨趟复用保护一律用 `PipeBarrier<PIPE_ALL>()`。`TQue` 管理的 buffer 与手工 flag **不交叉**。以下是实测固化的边界（均已按此绕行落地）：

| # | 实测事实（dav-2201） | 本方案的绕行 |
| --- | --- | --- |
| 1 | int32 向量 `Adds`/`Maxs` 触发设备异常，连带使 `CreateVecIndex` 在 `count > 8` 的内部路径不可用 | 索引向量一律用**标量 `SetValue` 循环 + `FenceSToV()`** 建一次；**禁止任何 int32 向量算术** |
| 2 | `Gather` 的偏移张量“视图”须 32B 对齐（把多个 slot 紧排后取子视图会触发异常） | 路线②的 slot 按 8 个元素（32B）对齐；**禁用“源张量视图 + 非 8 倍数 lane 偏移”**的窗口表达法（前导量只由 `srcBaseAddr` 承担） |
| 3 | `SetFlag/WaitFlag<HardEvent::MTE3_S>` 间歇性挂核（跨趟链路 + 末趟“悬空置位”） | 该事件对**整段不使用**；y 搬出的跨趟复用保护改用 `PipeBarrier<PIPE_ALL>()` |
| 4 | `SetFlag/WaitFlag<HardEvent::MTE3_V>` 跨迭代挂核 | 不使用 `MTE3_V`；V 读 MTE3 结果的方向改用 `PipeBarrier<PIPE_ALL>` |
| 5 | `PipeBarrier<PIPE_MTE3>` 不建立 MTE3→V 依赖 | 分趟入口一律用 `PipeBarrier<PIPE_ALL>()` |
| 6 | S→MTE3 读到旧 UB（分趟时第 2 趟起读到的仍是上一趟内容） | 展开改由 **V 管线 `Gather`** 完成；S 只写一次模式向量并配 `S_V`（属白名单） |
| 7 | S 管线的 GM 标量写是 **128B cacheline 读-改-写**：行切分下相邻核共享 cacheline ⇒ 后写核把另一核已写入的元素**回退成旧值** | **禁止 `GlobalTensor::SetValue` 写 `y`**；改走 V 展开 + MTE3 掩码搬出 |
| 8 | 可用事件组合白名单：`MTE2→V`、`S→V`、`V→MTE3`（其余组合或挂核、或不产生依赖） | 本方案只用这三个方向，且 `TQue` 与手工 flag 不交叉；**约束下游**：任何新增同步路径必须先对照本白名单 |

##### 9. 窗口前导量的两条等价路线

dst 窗口下取整会引入 `w0 = lo − pre < lo`（最多比 `c0` 小 `LEAD_COL = 7` 列），x 侧同理（最多前移 7 个 lane、另有 `LEAD_X = 8` 的前导 lane）。为容纳这段越出 chunk 起点的前导量，实现按 UB 占用在两条**等价**路线中二选一（**结果与路线无关**：前缀 lane 读到的是无意义内容，随后被 `Duplicate` 清零）：

1. **路线①「前导列栅格」**：`bandTile` 每 buffer 多留 `LEAD_COL = 7` 列，源基址恒非负；成本 `2·LEAD_COL·PITCH·4`，**随 `PITCH`（即 `k`）增长**；
2. **路线②「夹紧偏移向量」**：不预留列，改用预建的 `OFF_Q_NUM = 8` 条偏移向量把整窗前移 `q = −dcolw` 列；成本 `OFF_Q_NUM·align(CB+WIDEN, 8)·4`，**随 `CB` 增长**。

判据：`useColGuard = 2·LEAD_COL·PITCH·4 <= OFF_Q_NUM·(CB+WIDEN)·4`（UB 预算模型中 `routeFixed` 取**按实际选路**的分配量）。**两条路线互为对方的坏情形**（大 `k` 用例被预算压到很小的 `CB` 而 `PITCH` 上千 ⇒ 路线①代价高；小 `k` 用例 `CB` 上千而 `PITCH = 8` ⇒ 路线②代价高），故必须保留二选一。四个性能 case 均选**路线①**（判据 `1,344 <= 3,328` / `2,240 <= 5,056` / `1,344 <= 3,328` / `4,032 <= 4,352`）。

##### 10. Ascend C 接口验证

验证方式：按 MemBase 路线查阅官方 API 文档（`$ASC_DEVKIT_DIR/docs/zh/api/`），并**对同一 API 逐个核对所有变体**；目标环境 Ascend 910B3 / `arch22` / CANN 9.1.0。**未通过验证的 API 不进入本方案**。

| 关键 API | 用途 | 变体核对结论 | 验证状态 |
| --- | --- | --- | --- |
| `DataCopyPad`（GM→UB） | 带状列块搬入（非 32B 对齐尾块） | 基础版在 A2/A3 支持；`PaddingMode` 模板参数变体为 ascend950 新增 → **不使用**。约束：dst UB 32B 对齐、src GM 1B 对齐、`blockCount <= 4095`、`blockLen` 须为 `sizeof(T)` 整数倍；`srcStride` 单位 **byte**、`dstStride` 单位 **32B dataBlock**，二者语义均为块间 **gap** | **已验证，支持**（`dstStride = 0` 的依据见「数据流要点」） |
| `DataCopyPad`（UB→GM） | y 搬出 | 同 API 反向变体，基础版支持；src UB 32B 对齐；`srcStride` 单位 32B dataBlock、`dstStride` 单位 byte；写出时 dummy padding 被丢弃、不污染 GM | **已验证，支持** |
| `DataCopy`（GM↔UB 连续） | 仅在长度已 32B 对齐时可用 | 支持；`count·sizeof(T)` 非 32B 对齐时**向下取整**（尾元素不搬）⇒ 凡长度不保证对齐的搬运一律改用 `DataCopyPad`。本方案 x 窗口与 y 搬出均走 `DataCopyPad` | **已验证，支持（本方案不依赖其截断语义）** |
| `Mul` / `Add` | 四项向量乘加 | 基础 `count` 版与高阶版均支持 float on 910B；本方案只用 `count` 形式，避开 A2/A3 上 mask ≤ 64 的 repeat 语义 | **已验证，支持** |
| `Muls` / `Adds` | `alpha` 预乘、`beta` 缩放 | 基础版支持；`_flexible_scalar` 变体为 ascend950 新增 → **不使用**。**注意**：`Adds`/`Maxs` 的**向量版本在 int32 载体上不可用**（实测第 1 条）——本方案的 fp32 `Muls`/`Adds` 不受影响，但**任何“用 int32 向量运算生成/平移索引”的写法都被排除** | **已验证，支持（仅限 fp32 载体）** |
| `Duplicate` | `beta == 0` 时零 y、窗口前缀清零、`accY` 清初值 | `Duplicate(dst, scalarValue, count)` 支持 | **已验证，支持** |
| `Gather`（UB→UB） | **本方案核心**：带状列块的片上转置（`banded → 连续 ttRow_r`） | `count` 形式与高阶形式均支持 int32/uint32/float 等；`srcOffset` 相对 `srcBaseAddr` 的偏移单位为 **byte**、须按元素宽度对齐；`dst`/`src`/`srcOffset` 均须 32B 对齐且 `src != dst`；A2 每次迭代消耗 8 个 DataBlock（256 B）⇒ `count` 不能过小。本方案 `offVec[c] = c·PITCH·4` 恒为 32B 整数倍 | **已验证，支持**（对齐约束见实测第 2 条） |
| `CreateVecIndex` | 生成基础索引序列 | 文档标注支持 float/int32，但 CANN 实现在 `count > 8` 时**并非纯标量路径**（前 8 个元素后转 `Adds(int32)`）⇒ 与实测第 1 条冲突 | **结论：不可用 → 已绕开**（改为标量 `SetValue` 循环建一次） |
| `Transpose`（基础 16×16） | 备选转置手段 | 伪代码以 float16 为载体，对 fp32 的适用性无可靠保证 | **已验证但舍去**（改用 `Gather` 逐带内行提取），记录以防下游误用 |
| `SetFlag`/`WaitFlag`（`HardEvent`） | 核内流水同步 | 支持 on A2/A3，须成对且 `eventId` 一致；`TPipe` 管理下须用 `FetchEventID()` 取号。**实测可用组合仅 4 条白名单**（见实测表） | **已验证，支持（组合受限）** |
| `TPipe` / `TQue` / `TBuf` | UB 资源管理 | `TQue<VECIN\|VECOUT, BUFFER_NUM>` 支持双缓冲；`TBuf<VECCALC>` 用于计算中间量 | **已验证，支持** |
| `ReduceSum` | **本方案不需要** | 转置-tile 设计把问题化为纯逐元素乘加，**完全不使用归约** | **已验证，本方案不依赖** |
| `AtomicAdd`（scalar）/ `SetAtomicAdd<T>` | 曾考虑用于列切分归约 | scalar 形态仅 ascend950 支持、A2/A3 不支持；`SetAtomicAdd<float>` 在 A2/A3 可用但不自动清零 GM、须配 `DisableDmaAtomic()` | **已验证，本方案不采用**（改用行切分，完全不用原子） |
| `GlobalTensor::GetValue` / `SetValue`、`LocalTensor::SetValue` | `\|incx\| != 1` / `incx < 0`、`\|incy\| != 1` / `incy < 0` 的标量寻址回退 | 支持；标量写走 **S 流水**，写 UB 后若交给 V 读必须补 `S_V`（白名单）；**`GlobalTensor::SetValue` 的 GM 标量写不得用于 y 的搬出**（实测第 7 条） | **已验证，支持（用法受实测收窄）** |
| `GetBlockIdx` / `GetBlockNum` | 核号与核数 | 支持，仓内广泛使用 | **已验证，支持** |

**硬件与编译器能力边界小结**：① 全程 fp32、**无类型转换链**，故不存在舍入语义与 golden 不等价的风险；② 搬运接口的对齐与单位约束（`srcStride` byte / `dstStride` 32B dataBlock、UB 侧 32B 对齐）已在上表固化；③ 编译器对手工同步指令的收敛行为不确定 ⇒ 同一 buffer **只采用一种同步方式**；④ 目标架构特有限制（无 SIMT、scalar `AtomicAdd` 不可用、`_flexible_scalar` 系列不可用、`DataCopyPad` 的 `PaddingMode` 不可用）均已给出具体绕行，方案内无残留依赖；⑤ UB 容量与 bank 冲突：dav-2201 单核 AIV UB = **192 KiB**（仓内 `blas/common/arch/hardware.h:27`），本方案保守取 128 KB 上限，峰值占用 case4 为 85.4 KiB。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（DAV_2201 / `arch22`） | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品（DAV_2201 / `arch22`） | √ |
| Ascend 950PR / Ascend 950DT | 本文不涉及（本交付路径为 `arch22`；950 路径由 `blas/sbmv/arch35/` 既有实现覆盖） |

> 本文的全部设计与实测结论均基于 **Ascend 910B3（Atlas A2，DAV_2201 / `arch22`，CANN 9.1.0）**；本环境无 A3（910_93）与 Ascend 950 设备，故 arch22 结论在 A3 上的适用性未上板验证，arch35 路径不在本次改动范围内。

## 算子约束限制

- **功能范围：`n <= 8192`（对应的非退化带宽区间为 `k <= n−1`），`n > 8192` 不保证功能。** 该上限的声明对象是 `n`；`k` 仍**无上界**（`k >= n` 合法、返回 `ACLBLAS_STATUS_SUCCESS`），二者不冲突。该范围声明**不代表该范围内所有 `(n, k, uplo, incx, incy)` 组合均已覆盖**——`k > n−1` 的退化区在配套用例集中仍属覆盖薄弱区。该上限**不得被曲解为“必须报错 / 拒绝”**：本实现侧**不存在 host 侧可行性守卫**，超出声明范围的输入**不会**返回错误码，其行为不保证。
- **`x` 与 `y` 同址（`x == y`）：结果未定义 / 未保证。** 接口层**不对 `x == y` 做任何检测或拒绝**；虽然当前实现的实测表现未见错误结果，但该安全性是**时序涌现、不是架构保证**（kernel 全文无 `SyncAll` / 跨核 flag，只有核内流水同步），若未来阶段划分变化使 `y` 的写提前，该安全性可能消失。**使用建议：不要依赖 `x == y` 的就地用法**，需要就地结果时请改用独立的输出缓冲区。
- **大带宽（`k` 接近 `n`）时的性能退化**：在该区域 host 侧**自设**的 128 KB 预算门限失效，`CB` 会**静默**塌缩为 1、chunk 数上升 ⇒ **保守且慢**，但**结果正确、真机安全**。该退化只影响性能，不影响功能正确性。
- **仅支持 FLOAT32**：不涉及 FP16 / BF16 / INT；**禁止引入 fp16/bf16 中间态**（会改变舍入链、使结果与 fp32 golden 不等价）。
- **不支持超出 `incx` / `incy` / `lda` 语义的非连续内存访问**；无广播语义；不要求 dynamic shape；不要求确定性计算。
- **异步语义**：算子复用调用方通过 `aclblasSetStream` 绑定的 stream，**不自建 stream、host 侧不做设备同步**；调用方在读取 `y` 之前必须自行同步该 stream。
- **非法 `uplo` 返回码有意偏差**：本算子返回 `ACLBLAS_STATUS_INVALID_ENUM`，与仓内先例（返回 `INVALID_VALUE`）不同，依据为任务书 §2.4/§2.5 与配套用例 `TC_ED_156`；如后续要求与仓内统一，须由需求侧重新审定，不得由实现侧自行改。
- **平台限定**：全部结论仅对 910B3（`arch22`）有效；无 A3 / Ascend 950 设备可上板验证。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准（FLOAT32 数值判定） | 输出 `y` 全向量验证。逐元素容差门 `\|actual − golden\| <= atol + rtol × \|golden\|`，其中 `rtol = 2⁻¹⁰ = 9.765625e-4`、`atol = 2⁻¹⁶ = 1.52587890625e-5`；逐元素绝对硬门 `\|actual − golden\| <= max(1e-2, 32 × ULP(\|golden\|))`（`ULP(x) = 2^(floor(log2\|x\|) − 23)`，`x != 0`，value-relative）；用例通过条件为 **`matchedRatio >= 0.99` 且无任何元素越过最大误差门** | 任务书 §3.2（FLOAT32 档精度要求表），并引用生态算子开源精度标准 FLOAT32 档（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ） |
| 精度标准（`alpha == 0` 特例） | `alpha == 0` 且 `alpha` 指针非空时跳过矩阵乘，`y = beta * y` 的结果须**位精确（EXACT 校验）**，即逐元素浮点相等 | 任务书 §3.2「补充说明」第 1 条 |
| 精度标准（负向用例） | 全部负向用例的返回码与评测集 `expect_result` 列**精确相等**；负向路径不调用 golden、不做张量比对 | 任务书 §2.4 参数表（异常行为列） |
| 性能标准 | FLOAT32 输入场景下的**平均单次耗时**（先 warmup 再有效采样 >50 次取平均，单位 µs）不高于下表标杆：case1（`n=1024, k=16, UPPER, incx=incy=1`）13.57；case2（`n=2048, k=32, UPPER, 1/1`）24.67；case3（`n=1024, k=16, LOWER, 1/1`）14.41；case4（`n=4096, k=64, UPPER, 1/1`）47.74 | 任务书 §3.3 |
| 内存要求 | 不涉及 | 任务书 §3.4「不涉及」 |

> 本表只登记标准与口径本身；判定结论以验收环节为准。
> 精度判定的落地实现位于测试侧：`test/frame/verify.h` 的 `MixedToleranceStrategy` 与 `getMixedToleranceDefaults(ACL_FLOAT)`（逐项与本表阈值一致），`alpha == 0` 分支切换 `ExactStrategy` 并以 `ASSERT_EQ` 硬断言模式已生效（杜绝口径静默退化）。配套 CSV 中的 `mere_threshold` / `mare_multiplier` 两列在本算子判定链路上**不被消费**，不另立第二套口径。

## 性能实测数据（已实现指标）

采集口径：**`msprof` 归档 `op_summary_*.csv` 的 `Task Duration(us)` 列**，测得量为 **device 侧内核执行时间**（不含 host 造数、H2D/D2H 拷贝、golden 计算与比对）；先 warmup 5 次，再 3 轮 × 每轮 120 次内核下发，每用例 **360 个有效样本**（满足“有效采样 >50 次”）；判据基准取**均值**（任务书原文“取平均”），中位数仅作稳健性参考。

| case | 形状（uplo / n / k / lda） | 标杆 (µs) | 实测内核耗时 mean (µs) | median (µs) | 样本数 | 标杆/实测 |
| --- | --- | --- | --- | --- | --- | --- |
| case1 | UPPER / 1024 / 16 / 17 | 13.57 | **9.0651** | 8.8600 | 360 | 1.497 |
| case2 | UPPER / 2048 / 32 / 33 | 24.67 | **15.3291** | 15.1000 | 360 | 1.609 |
| case3 | LOWER / 1024 / 16 / 17 | 14.41 | **9.0573** | 8.8400 | 360 | 1.591 |
| case4 | UPPER / 4096 / 64 / 65 | 47.74 | **38.3270** | 38.1010 | 360 | 1.246 |

> 「标杆/实测」列为标杆值与实测均值的**算术比值**，供读者对照；本表只登记标准与实测值本身。
> 核数已实测确认：框架汇总的 `block_dim` 列为 **16 / 24 / 16 / 24**，与 Tiling 预期档位（`R_target = 64 / 86 / 64 / 171` ⇒ `useCoreNum = 16 / 24 / 16 / 24`）逐一一致。
> 可复现采集命令：`python3 test/sbmv/ssbmv/tools/ssbmv_perf_collect.py --repo . --out-dir <out> --cases doc --warmup 5 --repeats 60 --rounds 3 --device 0`。
> 构建与运行命令：`bash build.sh --soc=ascend910b3 --ops=sbmv`（不传 `--device`，`build.sh` 默认设备逻辑 ID `0`，即为本环境的正确取值；亦可显式 `--device=0`）。

**性能档位与瓶颈（设计期结论）**：本算子的四个标杆**不随计算量等比缩放**（case4 的 FLOPs 是 case1 的 15.64×，时延只有 3.52×），说明其中含与规模无关的固定开销。设计期预判的瓶颈维度为：**① 规模无关的固定开销（墙钟口径）** 与 **② 数据重排（`Gather`）与向量指令发射/依赖开销（内核口径）**；**显式排除**浮点算力（case4 仅占峰值约 1%）、HBM 带宽（约 2.4–4.9% 占用）与 MTE2 搬运能力。据此，优化集中在“减少指令条数、提高 lane 利用率、降低固定项”上，任何以提升算力利用率为目标的优化（降精度、增大向量长度）在本算子上不会带来收益。

## 测试设计

测试以**任务配套 CSV 驱动的 C++ GTest** 为唯一执行形态：`.csv` 逐条描述算例（`uplo` / `n` / `k` / `alpha` / `a_fill` / `lda` / `x_fill` / `incx` / `beta` / `y_fill` / `incy` / `expect_result` / 随机种子），`ssbmv_test.cpp` 逐行加载、造数、调用 `aclblasSsbmv`、同步后与 golden 比对；新增算例只需追加 CSV 行，不改测试代码。

| 项目 | 设计 |
| --- | --- |
| golden | `cblas_ssbmv`（Netlib BLAS 实数实现，**同精度 fp32**）；不引入更高精度实现，以与任务书「输出与 cblas golden 比对」的评测口径一致 |
| 用例集构成 | ① 任务配套提供集 `ssbmv_test.csv`（**逐字保留、不改动**）：精度类 1000 条 + 性能类 200 条；② 自行补充集 `ssbmv_supp.csv`：形态/取值扫描、正态分布输入、非二进制浮点标量、退化带宽等；③ 白盒补全集 `ssbmv_whitebox.csv`：针对本方案实现约束的定向例 |
| 覆盖分解（黑盒） | 维度 = `uplo` × 填充模式（`lda == k+1` 紧凑 / `lda > k+1` padding）× 步长（`+1` / `−1` / `\|inc\| > 1`）× 标量档位（`alpha ∈ {0, 0.5, ±1, 2}`、`beta ∈ {0, 0.25, 0.5, −0.5, 1}`）× 数值域（常规 / 次正规 / 极大极小 / INF·NAN） |
| 边界用例 | `n == 0`（不触碰 `y`）、`n == 1`、`k == 0`（对角阵）、`k >= n`（合法退化）、`lda == k+1` 与 `lda` 大幅 padding、`incx`/`incy` 使缓冲区出现空洞 |
| 负向用例 | 与 `expect_result` 列逐条比对返回码，覆盖 11 步校验的每一分支（`handle == nullptr`、`n < 0`、非法 `uplo`、`k < 0`、`lda < k+1`、`incx == 0`、`incy == 0`、`alpha`/`beta == nullptr`、`A`/`x`/`y` 的条件化判空）；负向路径**不调用 golden** |
| 判定口径 | 唯一一套：MIXED_TOLERANCE（见本文件「精度标准/性能标准」表）；`alpha == 0` 分支切 EXACT。判据实现为测试框架的容差策略，不在用例层重复实现 |
| 可复现口径 | 编译 `bash build.sh --soc=ascend910b3 --ops=sbmv`；运行 `build/test/sbmv/ssbmv/ssbmv_test --gtest_filter=...`；性能见「性能实测数据」节命令 |
| 覆盖缺口（据实登记） | `k > n−1` 的退化区在任务配套集中零覆盖，由补充集补齐；`x == y` 就地用法**不设计用例**（语义未定义，见「算子约束限制」） |

## 兼容性分析

- **新算子，不涉及既有算子行为变更**：`aclblasSsbmv` 为新增算子的 A2/A3 实现，`blas/sbmv/arch22/` 与 `test/sbmv/ssbmv/arch22/` 均为新建目录；`blas/sbmv/arch35/`（Ascend 950PR / 950DT）既有实现**不受本次改动影响**。
- **公开接口不变**：接口声明位于 `include/cann_ops_blas.h:156`（**声明已存在**），本次**不新增、不修改签名**，不定义产品私有的平行接口；`aclblasStatus_t` 的取值语义沿用 `include/cann_ops_blas_common.h`。调用方无需改动调用方式。
- **与仓内既有实现的一处有意差异**：非法 `uplo` 返回 `ACLBLAS_STATUS_INVALID_ENUM`（仓内同族先例返回 `INVALID_VALUE`）。该差异由任务书 §2.4/§2.5 与配套评测集 `TC_ED_156` 明确规定，评测链的负向路径只比较返回码与 `expect_result`、不调用 golden，故不产生判定冲突；**评审时不应判为缺陷**。
- **测试链路新增**：`test/sbmv/ssbmv/CMakeLists.txt` 需增加 `arch22` 分支（当前 A2/A3 构建走 `else` 分支时因 `test/sbmv/ssbmv/` 下无可发现的测试目标而报 `FATAL_ERROR`）；`test/sbmv/ssbmv/ssbmv_golden.h` / `ssbmv_param.h` 与 `test/frame/` 仅做**向后兼容的追加式扩展**（新增正态分布填充能力、CSV schema 的文本层兼容处理落在 `ssbmv` 侧参数层），不改动既有判定语义与既有用例的期望值。
- **算子文档**：`blas/sbmv/README.md` 的产品支持表与接口说明需补齐 Atlas A2/A3 路径（纯追加，不改动既有 arch35 相关表述）。

## 风险与待确认项

| # | 事项 | 影响 | 处置 |
| --- | --- | --- | --- |
| 1 | 非法 `uplo` 返回 `INVALID_ENUM`，与仓内同族先例（返回 `INVALID_VALUE`）不一致 | 若后续要求与仓内统一，本算子需改实现 | 已按任务书 §2.4/§2.5 与配套用例 `TC_ED_156` 取 `INVALID_ENUM`；如需统一，由需求侧重新审定 |
| 2 | 功能范围 `n <= 8192` 的来源为任务发起方答复、**非 CP1 已确认项** | 消费者可能误读为“超出即报错” | 已在「算子约束限制」按原文登记，并明确实现侧无 host 守卫、超出范围不返回错误码 |
| 3 | `k > n−1` 的退化区间在任务配套用例集中零覆盖 | 该区间行为缺少回归保护 | 由补充集补齐 `k >= n` 用例；本设计仍将其登记为覆盖薄弱区 |
| 4 | `SBMV_R_MIN = 64` 使 case1/case3 的核数被钉在 16（与 case2/case4 的 24 不一致） | 该两形状可能留下性能余量 | 本设计不变更该常量（下调会推高 halo 冗余与全算子读取量，收益未实测）；如需变更回需求侧审定 |
| 5 | A 搬入的 `dstStride` 单位（32B dataBlock）与“块间 gap”语义易被下游按“跨步”理解误改 | 列落点偏移 ⇒ **静默错读（不报错）** | 已写入实现约束第 2、10 条，并给出对照论证 |
| 6 | `x == y` 就地使用的安全性为**时序涌现、非架构保证** | 未来阶段划分变化可能使其失效 | 已在「算子约束限制」声明为未定义/未保证；接口不做检测或拒绝 |
| 7 | A3（910_93）与 Ascend 950 无设备可上板验证 | `arch22` 结论在产品间外推未经实测 | 本次交付限定 A2/A3（`arch22`）；`arch35` 路径不在改动范围 |
| 8 | `CB` 反推中的「向下取 2 的幂」若被删除，case4 的 `CB` 会贴到 128 KB 预算边界（仅余 136 B） | 任何附加 buffer、对齐抬升或口径微调即越限 | 已写入 UB 预算规则并标注为**硬步骤**，不得省略 |

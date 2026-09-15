# aclblasCsrot 算子设计文档

> 本文档依据 `aclblasCsrot_Atlas950PR_task_doc.md`（Ascend 950PR 任务书）编写，面向社区任务评审。
> 算子实现提交至 ops-blas 仓 `blas/rot/arch35/`，测试代码提交至 `test/rot/csrot/arch35/`。

---

# 需求背景（required）

## 需求来源

CANN 社区任务 2026 — 算子开发任务 `aclblasCsrot`（Atlas 950PR）。

- 任务书：`aclblasCsrot_Atlas950PR_task_doc.md`
- 上游仓库：https://gitcode.com/cann/ops-blas ，目标目录 `blas/rot/arch35/`
- 接口声明：`include/cann_ops_blas.h`

## 背景介绍

### aclblasCsrot 算子实现优化

ops-blas 仓的 `rot` 算子族提供 Givens 平面旋转。实数版本 `aclblasSrot` 已完成 arch35（Ascend 950PR）适配；复数版本 `aclblasCsrot` 此前只有 arch22（Atlas A2/A3）实现，本任务基于 Ascend C 为 arch35 重新实现。

### 算子实现现状分析

arch22 的实现不能直接复用到 arch35，原因有三：

1. **舍入语义不符**：arch22 kernel 用 `AscendC::Axpy`（FMA 语义）计算 `c*x + s*y`，而 netlib `csrot.f` 对每个乘积、每个加法分别舍入。二者在一般情况下只差半个操作数 ULP，但当两项相消时（`c == s` 且 `x == y`），参考结果是精确 0 而 FMA 形式留下非零残差，相对误差无界。
2. **入口校验缺失**：arch22 的 host 不校验 `handle`，也不区分 `n <= 0` 的快速返回，与对齐 cublas 的边界语义不一致。
3. **并行策略**：arch22 按单核 SIMD 设计，未利用 arch35 的多 AIV 核与 SIMT 能力。

本实现在 arch35 目录下新建 `csrot_{host,kernel,tiling_data,kernel.h}` 四个文件，沿用同族 `srot` 的双路径架构。

### 算子功能分析

对两个等长复数向量 `x`、`y` 做原地 Givens 旋转，旋转因子 `c`、`s` 均为**实数**（与 `crot` 的区别：crot 的 `s` 为复数）。旋转后：

```
x[i] = c * x[i] + s * y[i]
y[i] = c * y[i] - s * x[i]      (使用旋转前的 x[i])
```

典型用途是 QR 分解中消去下三角元素，属 BLAS Level-1 auxiliary 例程。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `aclblasCsrot` 算子：

- 数据类型：complex64（单精度复数向量）+ float 实数旋转因子
- 适配硬件：Ascend 950PR / 950DT（arch35），CANN 9.1.0
- 完整对齐 netlib `csrot.f` 的边界语义与舍入语义
- 性能不低于任务书 §3.3 的标杆耗时

## 需求拆解

1. **双路径实现**：`incx == 1 && incy == 1` 走连续 SIMD 路径；其余步长组合（含正非 1、负、零及其混合）走 stride SIMT 路径
2. **原地运算**：x、y 既是输入又是输出，且允许 x/y 指向同一元素（如 `incx == incy`），需保证读到全部原始分量后再写
3. **边界语义对齐 netlib**：`n <= 0` 为 no-op 且不校验其余参数；`inc == 0` 不拦截（同一元素串行旋转 n 次）；负步长从尾端起算；`c == 1 && s == 0` 不短路
4. **标量指针位置自动判定**：`c`/`s` 可位于 host 或 device，二者独立
5. **舍入语义对齐 netlib**：每个乘积、每个加法各舍入一次，禁止收缩为 FMA
6. 性能不低于任务书 §3.3 标杆

# 详细设计（required）

## 算子分析

### 数学公式

```
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - s * x[k]        (x[k] 为旋转前的值)

k = (i-1) * incx,  j = (i-1) * incy,  i = 1..n
```

`c`、`s` 为实数，实部与虚部按同一组 `(c, s)` **独立**旋转，不存在复数乘法交叉项：

```
x'[k].real = c * x[k].real + s * y[j].real
x'[k].imag = c * x[k].imag + s * y[j].imag
y'[j].real = c * y[j].real - s * x[k].real
y'[j].imag = c * y[j].imag - s * x[k].imag
```

这一性质是整个 kernel 设计的基础：**一个 Muls 指令即可同时旋转实部与虚部**。

### 支持数据类型

| 参数 | 类型 |
|------|------|
| x, y | complex64（`aclblasComplex`，8 字节：float real + float imag） |
| c, s | float32 |
| 计算精度 | float32 |

### 支持形状

逻辑上一维 `[n]`，物理跨度 `1 + (n-1) * |inc|`。n 任意整数（含 0、负值）；incx/incy 任意整数（含正、负、零）。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

**内存布局约定**：`aclblasComplex { float real; float imag; }` 是一个简单的 8 字节对，因此 **n 个 complex 元素的数组等价于 2n 个 float lane 的数组，实部在偶数 lane、虚部在奇数 lane**。kernel 全程以 float 视图操作，向量指令作用在 `2 * elementCount` 个 lane 上。设备侧无需任何 layout 转换。

**tiling 策略**：

`CsrotTilingData` 中所有计数均以 **complex 元素**为单位：

| 字段 | 含义 |
|------|------|
| `tilingKey` | 0 = 连续（SIMD），1 = stride（SIMT） |
| `totalN` | 元素总数 |
| `cosValue` / `sinValue` | c/s 位于 host 时的标量值；位于 device 时置 0 占位 |
| `cIsDevice` / `sIsDevice` | c/s 各自的位置标志 |
| `perCoreN` / `remainder` | 连续路径每核元素数 / 前 remainder 个核各多 1 个 |
| `tileSize` | 连续路径单次搬运的 UB 上限（complex 元素） |
| `incx` / `incy` | stride 路径步长 |
| `nthreads` | stride 路径每 block 的 SIMT 线程数 |

##### 1. 分核策略：

**连续路径**：`numBlocks = min(n, aivCoreNum)`，均分 `perCoreN = n / numBlocks`，余数 `remainder = n % numBlocks` 分配给前若干个核。刻意不做 block 对齐取整——不做对齐时 `n < numBlocks` 也不会饿死任何核，负载均衡可一直下探到 `n < numBlocks`。host 用纯 `DataCopyPad` 搬运，GM 侧只要求 1 字节对齐，因此非 block 对齐的起始偏移与元素个数都安全。

**stride 路径**：`numBlocks = min(ceilDiv(n, SIMT_MIN_THREAD_NUM), aivCoreNum)`，`nthreads = min(alignUp(ceilDiv(n, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)`。小 n 不会启动满额 2048 线程只为了退休绝大多数线程；大 n 仍能占满每 block 2048 线程。kernel 内用 grid-stride 循环跨全部 `block_num * blockDim.x` 个线程遍历 n 个元素。

##### 2. 数据分块和内存优化策略：

连续路径是 `TQue` 三级流水（CopyIn / Compute / CopyOut），使用 **5 个 UB tile**：

- `inQueueX` / `inQueueY`：VECIN（MTE2 输入）
- `outNewX` / `outNewY`：VECOUT（Vector 输出）
- `workBuf`：VECCALC（纯 Vector 暂存，放 `round(s*y)` / `round(s*x)`）

`tileSize = (UB_SIZE / (5 * 8)) / COMPLEX_PER_BLOCK * COMPLEX_PER_BLOCK`，保持 block 对齐（`COMPLEX_PER_BLOCK = 32 / 8 = 4`），使每个 tile 都起始于 32 字节边界、没有复数跨块边界。

**原地安全性**：旋转是原地操作，新 x 依赖原始 x 与 y，新 y 依赖原始 x 与 y。实现上把原始数据在整个 Compute 阶段保留在 VECIN 队列中，结果写入**独立的 VECOUT 队列**，四路分量全部读完才落盘——因此 `x` 与 `y` 指向同一元素（如 `incx == incy`）也是良定义的。这条约束是 arch22 实现缺失的。

**stride 路径**线程与元素的绑定：一个线程一次处理**一个** complex 元素、同时触碰 `2i` 与 `2i+1` 两个 float lane，因此一个元素的实部/虚部永远落在同一个线程的寄存器组里，不会被拆到两个线程。

零步长是特例：netlib 在 `incx == 0` 时 `IX` 恒为 1，n 次迭代反复复用上一轮刚写入的值（**串行累加**）。多线程 grid-stride 会并发读同一地址并竞争，与 netlib 语义不符。因此当 `zeroIncX || zeroIncY` 时，只有 `block0.thread0` 按严格 netlib 顺序串行执行 n 次迭代，其余线程与 block 立即返回，保证被复用的地址只被单一程序序触碰。

##### 3. tilingkey 规划策略：

| tilingKey | 条件 | kernel |
|---|---|---|
| 0 | `incx == 1 && incy == 1` | SIMD 连续路径（TPipe/DataCopy + Muls/Add） |
| 1 | 其余全部（含负、零、混合） | SIMT stride 路径 |

`c`/`s` 的指针位置**不进入 tilingKey**：kernel 侧统一读 `cIsDevice`/`sIsDevice` 标志——为 0 用 tiling 里的标量值，为 1 解引用 GM 指针。两条路径共用同一套标志，因此 4 种指针位置组合 × 2 条路径不需要 8 个 tilingKey。

**数据检测（负向）**：

- `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（优先于一切）
- `n <= 0` → `ACLBLAS_STATUS_SUCCESS`（no-op，不校验 x/y/c/s，对齐 `IF (N.LE.0) RETURN`）
- `n > 0` 时 `x`/`y`/`c`/`s` 任一为 `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`
- stride 路径下 `(n-1) * inc` 超出 int32 范围 → `ACLBLAS_STATUS_INVALID_VALUE`（在 int64 中计算并检查，避免 kernel 侧 int32 元素偏移溢出）
- `aclrtPointerGetAttributes` 查询失败 → `ACLBLAS_STATUS_INVALID_VALUE`

#### 3.2.2 kernel 侧设计：

与 `srot` 同构，分为 Init 与 Process 两个阶段，Process 含 CopyIn / Compute / CopyOut 三段。

**连续路径（tilingKey 0）** —— `CsrotAIV`：

```
CopyIn : DataCopyPad 搬入 x/y tile，字节数 dataCount * 8，lane 偏移 curOffset * 2
Compute: laneCount = dataCount * 2                       // 复数 → float lane
         Muls(newX, xLocal,  cosValue, laneCount)        // round(c*x)
         Muls(work, yLocal,  sinValue, laneCount)        // round(s*y)
         Add (newX, newX,   work,      laneCount)        // round(c*x) + round(s*y)
         Muls(newY, yLocal,  cosValue, laneCount)        // round(c*y)
         Muls(work, xLocal, -sinValue, laneCount)        // round(-s*x)
         Add (newY, newY,   work,      laneCount)        // round(c*y) + round(-s*x)
CopyOut: DataCopyPad 写回
```

`Muls`/`Add` 是单次舍入的向量指令，天然满足 netlib 的分步舍入语义；实部与虚部在同一 lane 数组上被同一条指令一起处理，因此不需要按分量拆分。

**stride 路径（tilingKey 1）** —— `csrot_simt_compute`：

按 netlib 的访问序定位元素：`inc > 0` 时 `jx/jy = i`，`inc < 0` 时 `jx/jy = n-1-i`（等价于尾锚点 `(-N+1)*INC` 向后退），`inc == 0` 时 `jx/jy = 0`。基址 `jx * absIncX * 2`（乘 2 是复数元素 → float lane）。

**舍入**：stride 路径的 `c * xr + s * yr` 在 -O3 下会被编译器收缩成 FMA，省掉一次乘积舍入。残差量级是**操作数的 ~1 ULP 而不是结果的 1 ULP**，两项相消时相对误差无界。因此每个乘积都过一层 `CsrotRoundedMul`（`__simt_callee__` 内联函数，用 `volatile` 局部量钉住乘积）强制单次舍入。SIMD 路径不需要这层包装，因为 `Muls` 本身就是单次舍入。

**原地安全性（stride）**：四个分量（`xr/xi/yr/yi`）先全部读入寄存器，再执行四条写入，因此 x/y 别名不会污染源操作数。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Ascend 950DT（arch35） | √ |
| Atlas A3 训练/推理系列、Atlas A2 训练/推理系列 | √（沿用 arch22 既有实现，本任务不改动） |

## 算子约束限制

- `n` 为 `int`，需满足 `n <= INT32_MAX`
- stride 路径要求 `(n-1) * incx` 与 `(n-1) * incy` 均落在 int32 范围内，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `c`、`s` 必须为 `float`（实数）；复数正弦的 `crot` 不在本算子范围内
- 原地运算，`x`/`y` 必须位于 Device 内存

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部/虚部分别按 FLOAT32 判定：atol = 2^-16，rtol = 2^-10，matched_ratio ≥ 0.99，max_abs_error ≤ max(1e-2, 32×ULP) | 任务书 §3.2；生态算子开源精度标准 |
| 辅判据 | MERE ≤ 2^-13，MARE ≤ 10 × 2^-13（按 CSV 列配置） | ops-blas 测试框架 |
| 性能标准 | n=1048576 / 2097152 / 4194304（incx=incy=1）平均单次耗时 ≤ 21.69 / 43.19 / 119.38 us | 任务书 §3.3 |
| 内存标准 | 不涉及（逐元素原地运算，无 workspace） | 任务书 §3.4 |

**舍入语义说明**：本算子的误差来源仅为每元素两次实数乘加的浮点舍入。为避免 golden 与算子对"分步舍入 vs FMA"的理解不一致，两侧都显式固定了分步舍入：算子连续路径用 `Muls+Muls+Add`（本身即单次舍入），stride 路径用 `volatile` 包装的乘法；golden 用同构的包装函数，使其语义不随 `-std=` 取值漂移（GCC 在 `-std=gnu++17` 下默认收缩、ISO `-std=c++17` 下不收缩，而测试工程未显式设置标准）。

## 兼容性分析

接口 `aclblasCsrot` 的参数序列与 `cublasCsrot` 一一对应（handle 及参数顺序），无需额外映射说明（任务书 §3.5.2）。

arch22 的旧实现保留在 `blas/rot/arch22/` 目录下，按 SOC 分派，不影响既有产品线。

# 需求背景（required）

## 需求来源

本需求来源于 CANN 2026 年 8 月社区任务“aclblasCgeru 算子开发（Ascend 950PR）”。本设计以任务中心提供的 `aclblasCgeru_Atlas950PR_task_doc.md` 为依据。任务要求基于 `cann/ops-blas` 仓库，使用 Ascend C kernel 直调方式实现与 cuBLAS `cublasCgeru` 参数语义一致的单精度复数无共轭秩-1 更新接口，并完成设计、开发、精度及性能自验。

目标软件和硬件环境为 CANN 9.1.0 与 Ascend 950PR。

## 背景介绍

### aclblasCgeru 算子概述

Cgeru 是 BLAS Level 2 的单精度复数通用秩-1 更新算子。输入为复数标量 `alpha`、长度为 `m` 的复数向量 `x`、长度为 `n` 的复数向量 `y`，以及 `m x n` 的复数矩阵 `A`，输出原地写回 `A`：

```text
A = alpha * x * y^T + A
```

矩阵采用列主序存储。Cgeru 的 `u` 表示 unconjugated，计算中直接使用 `y[j]`，不对其取共轭。这是它与 Cgerc 的唯一数学语义差异：Cgerc 使用 `y^H`。

### ops-blas 现状分析

`ops-blas` 当前已有以下可复用基础：

- `blas/ger/arch35/sger_*`：Ascend 950 上的实数 GER SIMT 实现，可参考动态分核、连续 `x` 的 UB 缓存路径和通用 GM 路径；Cgeru 在性能子域采用连续复数 vector 快路径，在通用子域保留矩阵逻辑元素 SIMT fallback。
- `blas/gerc/arch22/cgerc_*`：单精度复数共轭 GER 的历史实现，可用于核对复数乘法与 Cgerc/Cgeru 的语义边界，但其工程骨架和 arch22 特定实现不直接复制。
- `include/cann_ops_blas_common.h`：定义 `aclblasComplex`，由两个 FP32 分量 `real`、`imag` 交错组成。

在本分支实施该需求之前，公共头文件没有 `aclblasCgeru` 声明，Ascend 950 的 `ger` 家族中也没有 Cgeru 实现。本需求新增统一公共接口，不定义 950PR 私有平行接口。

### aclblasCgeru 功能分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 上下文句柄，携带 stream | scalar | aclblasHandle_t | 非空 | - |
| m | 矩阵 A 的行数 | scalar | int | m >= 0 | - |
| n | 矩阵 A 的列数 | scalar | int | n >= 0 | - |
| alpha | 复数缩放因子，Host 内存 | scalar pointer | COMPLEX64 | 非空 | 1 |
| x | 输入复数向量，Device 内存 | tensor | COMPLEX64 | incx != 0 | 逻辑长度 m |
| incx | x 的元素步长 | scalar | int | 支持正负步长，不支持 0 | - |
| y | 输入复数向量，Device 内存，不取共轭 | tensor | COMPLEX64 | incy != 0 | 逻辑长度 n |
| incy | y 的元素步长 | scalar | int | 支持正负步长，不支持 0 | - |
| A | 输入/输出复数矩阵，Device 内存 | tensor | COMPLEX64 | 列主序，原地更新 | lda x n，更新前 m x n |
| lda | A 的前导维度 | scalar | int | lda >= max(1, m) | - |

当步长非零时，`x` 和 `y` 的物理长度分别为 `1 + (m - 1) * abs(incx)` 与 `1 + (n - 1) * abs(incy)`。负步长按 BLAS 语义从物理存储尾部反向取逻辑元素。

# 需求分析（required）

## 需求描述

使用 Ascend C 为 Ascend 950PR 实现 `aclblasCgeru`。接口参数顺序与 cuBLAS `cublasCgeru` 对齐，支持 COMPLEX64、任意合法 `m/n/lda`、正负 `incx/incy`、零维和零 `alpha` 快速返回，并保证 kernel 在 handle 绑定的 stream 上异步执行。

公共接口为：

```cpp
aclblasStatus_t aclblasCgeru(aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy, aclblasComplex* A, int lda);
```

## 需求拆解

1. 新增公共 `aclblasCgeru` 声明、算子 README 和 API 索引。
1. 在 `blas/ger/arch35/` 新增 Host、Kernel、共享声明和 Tiling 数据文件，使用仓库的自动源文件发现机制参与 Ascend 950 构建。
1. 实现 `A[i + j * lda] += alpha * x[i] * y[j]`，其中 `y[j]` 不取共轭。
1. 支持正负 `incx/incy` 及带 padding 的 `lda`，只更新矩阵前 `m x n` 区域。
1. 参数非法时返回约定状态码；`m == 0`、`n == 0` 或 `alpha == (0, 0)` 时不启动 kernel，也不访问或写入 `x/y/A`。
1. 使用动态 AIV 核数；单位步长且 `m >= 8` 时使用连续复数 vector 路径，并通过最多 4,088 行的行分块和小 `m` 多列批处理覆盖非对齐及大 shape；其余 shape 使用 SIMT UB-xy、UB-x 或 GM fallback。
1. Ascend 950 构建保持仓库全局 Debug 配置，仅对 `blas/ger/arch35/cgeru_kernel.cpp` 追加源文件级 `-O3`，使该性能交付 Kernel 使用优化编译且不改变其他算子的构建策略。
1. 使用任务配套 CSV 和 GTest 覆盖功能、负向、特殊值与性能场景，golden 使用 Netlib Cgeru 无共轭语义。

# 详细设计（required）

## 算子分析

### 数学公式

对 `i = 0, ..., m - 1`、`j = 0, ..., n - 1`：

```text
A[i + j * lda] = A[i + j * lda] + alpha * x[x_index(i)] * y[y_index(j)]
```

步长索引为：

```text
x_index(i) = i * incx                               , incx > 0
             (m - 1 - i) * (-(int64_t)incx)         , incx < 0

y_index(j) = j * incy                               , incy > 0
             (n - 1 - j) * (-(int64_t)incy)         , incy < 0
```

设 `alpha = ar + ai*i`、`x = xr + xi*i`、`y = yr + yi*i`。Kernel 先计算 `temp = alpha * y`：

```text
temp.real = ar * yr - ai * yi
temp.imag = ar * yi + ai * yr
```

再计算 `delta = x * temp` 并更新 A：

```text
delta.real = xr * temp.real - xi * temp.imag
delta.imag = xr * temp.imag + xi * temp.real
A.real += delta.real
A.imag += delta.imag
```

计算 `temp` 时直接使用 `yi`，不执行取反操作，因此严格保持无共轭语义。

### 支持数据类型

输入和输出均为 `aclblasComplex`（COMPLEX64），实部和虚部为 FP32。标量 `alpha` 位于 Host 内存；`x`、`y`、`A` 位于 Device 内存。

### 支持形状

- `x`：逻辑长度 `m`，物理长度 `1 + (m - 1) * abs(incx)`。
- `y`：逻辑长度 `n`，物理长度 `1 + (n - 1) * abs(incy)`。
- `A`：列主序 `lda x n`，仅更新每列前 `m` 个元素，`lda - m` 的 padding 保持不变。
- `m/n` 是运行时参数，不要求固定 shape。

## 算子实现

### 实现方案

#### Vector 路径与 SIMT fallback 选型分析：

当 `incx == 1`、`incy == 1` 且 `m >= 8` 时，Host 选择连续复数 vector 路径，不再要求 `m` 为 8 的倍数，也不设置 4,095 行上限。该路径把交错 COMPLEX64 的 x 和 A 搬入 UB，使用 `DeInterleave` 分离实部/虚部，执行向量标量乘法与加减，再用 `Interleave` 恢复交错布局。单个行 tile 最多处理 4,088 个有效元素；非 8 对齐的 tile 只在 UB 中把 extent 向上补齐到 8 的倍数，GM 搬入和搬出仍使用精确的 `rowCount`，因此不会读写 A 的 padding 或下一列。`m > 4088` 时按多个行 tile 处理，每个 AIV Core 仍独占其分配的连续列区间。

对 `lda == m`、`m <= 256` 且单核至少分到 8 列的宽矩阵，vector 入口进一步启用多列批处理。x 只搬入一次；A 的多个紧凑列通过多 block `DataCopyPad` 搬入按 `ceilAlign(m, 8)` 分隔的本地切片，逐列复用同一 x 完成计算，再批量搬出。单批本地行槽总数不超过 4,080，最后不足整批的列使用实际 `colCount`。普通行分块循环和多列批处理循环都在相邻 tile/batch 之间显式设置并等待 `MTE3_MTE2` 事件，确保上一轮搬出完成后才开始下一轮搬入，避免复用队列和 UB 时出现跨轮流水竞态。

`m < 8` 或任一向量步长不是 1 时保留 SIMT fallback：连续且较小的 x/y 走 UB-xy，只有 x 可缓存时走 UB-x，其余走 GM。带 padding 的任意合法 `lda` 在 vector 普通行分块或 SIMT 路径中按列地址处理；正负非 1 步长由 SIMT 路径覆盖。vector 与 SIMT 使用同一 Tiling 结构、同一复数运算结合顺序和同一外部 API。

仓库顶层固定使用 Debug 构建。为了避免用 `-O0` 评价性能交付 Kernel，顶层 `CMakeLists.txt` 仅在 `SOC_VERSION` 为 Ascend 950 时，对 `cgeru_kernel.cpp` 追加源文件级 `-O3`；全局配置仍保持不变，其他源文件不受影响。本次最终验收数据已在代码固化后重新采集，并与最终 commit 一一绑定。

#### host侧设计：

Host 入口拆分为参数校验、Tiling 计算与 Kernel 启动三个职责。外部 API 依次执行：

1. 检查 `handle`；空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
1. 检查 `m/n` 非负、`alpha` 非空、`incx/incy` 非 0、`lda >= max(1, m)`；失败返回 `ACLBLAS_STATUS_INVALID_VALUE`。
1. 若 `m == 0`、`n == 0` 或 `alpha == (0, 0)`，直接返回成功，不检查或访问 `x/y/A`。
1. 对有效计算检查 `x/y/A` 非空。
1. 获取实际 AIV Core 数，计算 Tiling，使用 handle 中绑定的 stream 启动 kernel。Host 不主动同步 stream。

Host 使用 `OP_LOGE` 记录参数或资源错误，使用 `OP_LOGD` 输出 Tiling 字段，使用 `OP_LOGI` 输出 block 数、Core 数和选择的性能路径。

Tiling 数据包含：

| 字段 | 含义 |
| --- | --- |
| m, n | A 的有效行数与列数 |
| lda | A 的列主序前导维度 |
| numThreads | SIMT fallback 每个 block 的线程数；vector 路径不使用该字段组织向量指令 |
| totalElements | 需要更新的矩阵元素总数，即 `(uint64_t)m * n` |
| alphaReal, alphaImag | alpha 的实部和虚部 |
| incx, incy | x、y 的逻辑步长 |

##### 1. 分核策略：

Host 首先判断 vector 条件：`incx == 1 && incy == 1 && m >= 8`。vector 路径使用 `min(n, aivCoreNum)` 个 block，将 n 列尽量均匀地分成连续区间；每个 block 只更新自己的列，不需要原子操作。Kernel 内部把行维切成不超过 4,088 行的精确 tile；当 `lda == m`、`m <= 256` 且当前 block 的列数不少于 8 时，改为使用不超过 4,080 个本地行槽的多列 batch。上述 tile/batch 只改变单核的数据搬运粒度，不改变 Host 的列分核结果。

不满足 vector 条件时，将 `m x n` 有效区域按列主序压平为 `totalElements` 个逻辑元素，动态使用设备 AIV Core：

```text
totalElements   = (uint64_t)m * n
numBlocks       = min(ceil(totalElements / 128), aivCoreNum)
elementsPerBlock = ceil(totalElements / numBlocks)
numThreads      = min(ceilAlign(elementsPerBlock, 128), 2048)
```

线程按全局 grid-stride 处理逻辑元素。对每个 `logical`，使用 `row = logical % m`、`col = logical / m` 映射到物理地址 `row + col * lda`。连续线程优先访问同一列内的连续行，跨列时跳过 `lda - m` 的 padding。每个有效 A 元素只由一个线程更新，因此无需原子操作。

##### 2. 数据分块和内存优化策略：

实现包含一个 vector 入口和三个 SIMT 子路径；vector 入口内部再选择行分块或多列批处理：

- vector 行分块：条件为 `incx == 1`、`incy == 1`、`m >= 8` 且未启用多列 batch。每个 tile 的有效行数为 `min(remaining, 4088)`，本地 extent 为 `ceilAlign(rowCount, 8)`。x 和每列 A 都只按有效 `rowCount` 从 GM 搬入；输出在补齐后的 UB extent 上执行 `Interleave`，再用精确长度 `DataCopyPad` 丢弃本地补齐后缀。单 tile 显式 buffer payload 上界为 `48 * 4088 = 196224` B，约 191.63 KiB。
- vector 多列批处理：条件为 `lda == m`、`m <= 256` 且当前 block 至少处理 8 列。单列本地跨度为 `ceilAlign(m, 8)`，每批列数受 4,080 个本地行槽限制。x 只搬入一次，A 使用多 block `DataCopyPad` 批量搬入/搬出；尾批按实际剩余列数处理。
- UB-xy SIMT fallback：当未走 vector、`incx == 1`、`incy == 1` 且 `m/n <= 8192` 时，block 内线程协作缓存原始 x、`alpha*y` 和原始 y 非零标志，静态数组 payload 上限为 160 KiB。
- UB-x SIMT fallback：当 `incx == 1` 且 `m <= 8192` 时，block 内线程协作缓存 x，静态数组 payload 上限为 64 KiB。
- GM SIMT fallback：其他情况直接按照 `incx/incy` 从 GM 读取 x 和 y；正负非 1 步长均由此覆盖。矩阵逻辑元素通过全局 grid-stride 分发，覆盖任意合法 shape 和尾部元素。

vector 行分块和多列 batch 在相邻轮次之间使用 `HardEvent::MTE3_MTE2` 建立显式搬出到搬入依赖。`A` 以列内连续优先的次序读改写，padding 不参与逻辑迭代。接口的 workspace 大小为 0，不需要调用方提供外部 workspace。内存：不涉及。

##### 3. tilingkey规划策略：

本算子只有 COMPLEX64 一种 dtype，数学公式和输出布局不随属性变化，不需要 TilingKey。Host 和 kernel-do 根据 `incx/incy/m` 选择 vector kernel 入口或 SIMT kernel 入口；vector 入口根据 `lda/m/单核列数` 选择行分块或多列 batch，SIMT 入口再选择 UB-xy、UB-x 或 GM。所有路径共享同一 Tiling 结构和外部接口。

#### kernel侧设计：

两个入口都声明为 AIV-only。`cgeru_aiv_kernel` 创建局部 `TPipe` 和 `CgeruVector`，负责连续复数 vector 快路径；`cgeru_kernel` 不创建 TPipe，通过 `asc_vf_call` 启动 UB-xy、UB-x 或 GM SIMT fallback。`TPipe` 不是对象成员，tiling 也不保存逐核数组。

Vector 路径执行流程：

1. 按 block 的连续列区间初始化，并根据 `m/lda/单核列数` 选择行分块或多列 batch；所有本地实部/虚部切片均按 8 个 FP32 元素对齐。
1. 行分块模式按最多 4,088 行循环：搬入当前 x tile，等待 MTE2→V 事件后 `DeInterleave`；逐列读取 y、跳过复数零列、搬入当前 A tile、计算并用精确 `rowCount` 搬出。
1. 多列 batch 模式只搬入一次完整 x，通过多 block `DataCopyPad` 搬入若干紧凑 A 列，逐列在各自对齐切片上计算，再按实际 `colCount` 批量搬出。
1. 非 8 对齐行数只补齐 UB extent，GM 访问始终使用有效行数；`lda-m` padding 不会被写回。
1. 若还有下一个 row tile 或 column batch，使用 `MTE3_MTE2` 事件等待本轮搬出完成后再进入下一轮搬入。

SIMT fallback 执行流程：

1. UB-xy 先协作缓存连续 x、缩放 y 和 y 非零标志；UB-x 只缓存连续 x；GM 不缓存。
1. 每个线程以全局 grid-stride 遍历逻辑元素并映射 `(row,col)`；原始 y 为零时跳过该元素。
1. 按路径从 UB 或 GM 读取 x/temp，并原地更新 `A[row + col*lda]`。任意正负步长和非对齐 shape 都由这些路径覆盖。

负步长索引先提升为 int64_t 再求相反数和计算非负的物理尾部偏移，因此包括 `INT_MIN` 在内的任意非零 int 步长都不会因 32 位有符号取负而溢出，也不会在 Device 指针之前访问。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

- 仅支持 COMPLEX64。
- 矩阵 A 仅支持列主序，`lda >= max(1, m)`。
- `m/n` 必须非负；`incx/incy` 必须非 0。
- 只支持由 `incx/incy/lda` 表达的非连续访问，不支持额外 Tensor stride 或广播。
- A 原地覆写，不返回视图。
- `m == 0`、`n == 0` 或 `alpha == (0, 0)` 时为合法 no-op。
- API 异步执行，Host 读取结果前必须同步 handle 绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部、虚部分别按 FP32 判断：rtol=2^-10、atol=2^-16、required_matched_ratio=0.99、max_abs_error_limit=1e-2 或 32*ULP | CANN 生态算子开源实验精度标准与任务书 |
| 性能标准 | COMPLEX64 平均耗时不高于任务书标杆：512x512 为 8.61 us，1024x1024 为 12.82 us，2048x2048 为 44.53 us；warmup 后有效采样超过 50 次 | aclblasCgeru 社区任务书 |

精度 golden 使用 Netlib BLAS `cgeru`，按 `A = alpha * x * y^T + A` 生成，实部与虚部分别比对。有限数的逐元素容差为 `abs(output-golden) <= atol + rtol*abs(golden)`，每个分量的 matched ratio 必须不低于 0.99，且任何元素不得超过 `max(1e-2, 32*ULP(abs(golden)))`。matched ratio 只统计逻辑 `m*n` 有效区；`lda` padding 从比例中排除并按原始字节精确检查未改写。alpha 为零或 y 全零时，完整 A 存储（含有效区和 padding）直接逐字节比较。非法参数、空指针、no-op 与 Inf/NaN 场景分别按状态码或特殊值语义汇总，不混入普通有限数误差聚合。自验覆盖计划如下：

- 基础 shape：0、1、小质数、2 的幂、2 的幂加减 1、非对齐矩形及大 shape。
- 步长：`incx/incy` 的 `+/-1`、`+/-2`、`+/-3` 组合。
- 前导维度：`lda == max(1, m)` 与多种 padding；检查 padding 区域不被修改。
- 对齐偏移：x、y、A 均从各自 Device 分配基址偏移一个 COMPLEX64 元素后调用 API，并检查 A 前后 guard 与 padding。
- 标量：`(0,0)`、`(1,0)`、纯虚数、一般复数和大值。
- 数据：均匀/正态分布、全零、交替值、极端值以及任务书要求的 Inf/NaN 场景。
- 负向：空 handle、负维度、空 alpha、非法步长、非法 lda，以及有效计算中的空 x/y/A。
- 无共轭专项：选择虚部非零的 y，使 Cgeru 与 Cgerc golden 明显不同，防止误引入共轭。
- 性能：完整执行任务书三个性能 case，先 warmup，再采样超过 50 次并记录平均耗时。
- 补充直接测试：在官方 1,000/200 用例之外增加空 handle、Device 指针对齐偏移、固定种子正态分布和 vector 数据搬运边界四个直接 GTest。边界测试覆盖 `m=3999/4087/4088/4089/4095/4096/4097`、`n=1/3`、`lda=m/m+3`，以及 `129x2000`、`255x1024` 的多列 batch 与尾批；不修改官方 1,200 行 CSV，也不计入官方 1,000/200 用例数。

自验严格使用任务包提供的 `cgeru_test.csv`、`verify_accuracy.py`、`verify_performance.py` 和 `gpu_baseline.csv`。官方 CSV 共 1,200 条，其中 1,000 条精度用例覆盖基础 shape、shape 扫描、标量、非方阵、前导维、正负步长、填充、边界与负向、Inf/NaN 等场景；200 条 `TC_PF` 用于性能测试。

精度脚本自动排除 `TC_PF`，按任务包 README 的官方命令执行：

```bash
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./cgeru_test.csv
```

精度结论要求脚本汇总严格为 `PASS=1000, FAIL=0`，并另行保存完整 1,004 条非性能 GTest 退出码为 0 的证据。原因是任务包脚本没有传播 GTest 进程退出码，且其正则只统计带参数后缀的官方 CSV 用例，不统计四个直接测试；不能仅凭脚本自身退出码宣称通过。

性能按任务包 README 的官方命令执行：

```bash
python verify_performance.py --repo /path/to/ops-blas --soc ascend950 --timeout 3600
```

测试工程对每条官方 `TC_PF` 用例只分配和拷贝一次输入，先进行 5 次 warmup，再使用 CANN Timeline Event 测量 100 次有效调用，并输出含 `case/m/n/incx/incy/lda/warmup/samples/device_average_us/host_average_us` 的 `[CGERU_PERF]` 记录。`device_average_us` 用于任务书平均单次耗时判断；512×512、1024×1024、2048×2048 三组结果分别与 8.61 us、12.82 us、44.53 us 的上限比较。

任务包原始 `verify_performance.py` 不做修改。该脚本从 GTest 行中只解析整条测试的整数毫秒耗时，而测试工程另外输出 Timeline Event 的微秒级单次平均值。本次最终复测同时保存了官方脚本原始输出、结果 CSV 和 `[CGERU_PERF]` 采样记录，并在设计文档、汇总说明和原始证据 README 中明确各字段的判定口径；自验证表按官方模板仅保留必要结果。不得把 `NO_REF`、脚本退出码或未绑定最终 commit 的历史数据直接表述为性能通过。

## 最终复测状态

最终复测绑定代码提交 `d90e9b77958c1a870f18626ff49fc11a20ce3c69`，在 Ascend 950PR、CANN 9.1.0 环境完成。构建退出码为 0；任务包 `verify_accuracy.py` 汇总为 `PASS=1000, FAIL=0`，完整非性能 GTest 为 1,004/1,004 通过，vector 数据搬运边界补测连续运行 20/20 轮通过。三组报告用例的实部、虚部 `matchedRatio` 均为 1.0，最大绝对误差均不超过 `3.8147e-06`。

前一轮完整性能回归对 200 条官方 `TC_PF` 用例逐条执行 5 次 warmup 和 100 次有效采样，并以 `[CGERU_PERF]` 的 `device_average_us` 与任务包 `gpu_baseline.csv` 逐条比较，结果为 200/200 通过；全量最差 `gpu_reference_us / device_average_us` 比值为 `0.4992565419`，不低于 0.4。该轮原始日志、parsed CSV/JSON 保持原样，其中三组任务书验收尺寸的记录值为 3.572920 us、5.889220 us、14.162490 us。`gpu_baseline.csv` 是任务书提供的 GPU 参考性能口径，不是 TBE 实现或 TBE 对照。

自验证表及 Ascend 截图采用本次 Ascend 950PR WebIDE 终端现场重跑值：

| 尺寸 | 本次 WebIDE 现场重跑 `device_average_us` | 任务包 GPU 参考值/us（非 TBE） | 任务书上限/us | 结果 |
| --- | ---: | ---: | ---: | --- |
| 512×512 | 3.592890 | 3.443 | 8.61 | 通过 |
| 1024×1024 | 5.843440 | 5.128 | 12.82 | 通过 |
| 2048×2048 | 14.171460 | 17.811 | 44.53 | 通过 |

任务包原始 `verify_performance.py` 仍输出 `PASS=0 FAIL=192 NO_REF=8`。这是脚本把 GTest 行末的整数毫秒当作单次设备耗时、且没有读取 `[CGERU_PERF]` 微秒字段造成的判定口径缺陷；该原始输出已随证据保留，不能掩盖，也不能用来否定上述 200 条 `device_average_us` 按任务包 GPU 基线逐条比对的结果。内存：不涉及。

自验证报告由提交人使用官方模板手工填写并提交；其中“TBE截图”和“TBE算子性能(us)”均留空，本任务无 TBE 对照，任务包 GPU 基线不填入 TBE 列。Ascend 截图为 Ascend 950PR WebIDE 终端现场重新截取的原始 JPEG：512×512 为 `webide_screenshots/ascend_512.jpg`（2898×1545），1024×1024 为 `webide_screenshots/ascend_1024.jpg`（2942×1553），2048×2048 为 `webide_screenshots/ascend_2048.jpg`（2878×1546）；每张截图同屏清楚显示实部、虚部均 `PASSED`、对应 `PF` 性能记录和整体 `[  PASSED  ]` 汇总。原始证据目录保存构建、精度、性能、边界稳定性、环境和被测提交信息，均与上述最终提交绑定；不以本次截图值改写前一轮完整 200 条回归的原始记录。

## 兼容性分析

`aclblasCgeru` 是新增公共接口，不改变已有接口和 ABI 行为。参数顺序、列主序、步长以及无共轭语义与 cuBLAS `cublasCgeru` 对齐；类型使用仓库既有 `aclblasComplex`。实现位于 `blas/ger/arch35/`，仅在 Ascend 950 对应的 arch35 构建中被自动发现；顶层 CMake 的 `-O3` 属性也只匹配 Ascend 950 的该源文件，不影响 arch20/arch22 或其他算子。

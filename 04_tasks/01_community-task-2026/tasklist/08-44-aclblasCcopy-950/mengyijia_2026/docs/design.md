# aclblasCcopy 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 目标硬件 | Ascend 950PR（`ascend950`，`arch35`） |
| 目标软件 | CANN 9.1.0、ops-blas 句柄式 BLAS 接口 |
| 实现目录 | `ops-blas/blas/copy/arch35/` |
| 测试目录 | `ops-blas/test/copy/ccopy/` |

# 需求背景（required）

## 需求来源

本设计对应 2026 年 8 月社区任务第 44 项“aclblasCcopy 算子开发（950）”。任务要求参考
cuBLAS `cublasCcopy` 和 Netlib BLAS `ccopy` 的语义，在 Ascend 950PR 上使用 Ascend C 实现
单精度复数向量复制，并将实现与测试工程提交至 ops-blas 仓库。

算子复用 `include/cann_ops_blas.h` 中已有的 `aclblasCcopy` 声明和
`include/cann_ops_blas_common.h` 中的 `aclblasComplex` 类型，不新增 950PR 私有接口。

## 背景介绍

### 算子功能

`aclblasCcopy` 将源向量的 `n` 个逻辑复数元素复制到目标向量。单个元素定义为：

```cpp
typedef struct aclblasComplex {
    float real;
    float imag;
} aclblasComplex;
```

一个 `aclblasComplex` 由相邻的实部和虚部两个 FP32 组成，占 8 B。算子不执行浮点运算或
类型转换，按位复制这 8 B 表示。`incx`、`incy` 是以复数元素为单位的非零步长，可以为正数
或负数；目标向量中未被逻辑索引命中的位置必须保持原值。

### 开发范围

| 内容 | 路径 | 作用 |
| --- | --- | --- |
| Host 实现 | `blas/copy/arch35/ccopy_host.cpp` | 参数校验、Tiling、可选 offset 表准备和 Kernel 下发 |
| Kernel 实现 | `blas/copy/arch35/ccopy_kernel.cpp` | 连续及离散步长复制 |
| Tiling 定义 | `blas/copy/arch35/ccopy_tiling_data.h` | Host 与 Kernel 共享切分参数 |
| 测试工程 | `test/copy/ccopy/` | CSV 驱动 GTest、CBLAS 参考结果及验收脚本 |

# 需求分析（required）

## 需求描述

公共接口如下：

```cpp
aclblasStatus_t aclblasCcopy(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    aclblasComplex* y,
    int incy);
```

接口参数顺序与 `cublasCcopy` 一致。`x`、`y` 位于 Device 内存，handle 保存执行 stream。
Kernel 在该 stream 上异步下发；调用方在读取 `y` 或释放相关内存前负责同步。

`n` 是非负逻辑元素数；`x`、`y` 是 Device 指针；`incx`、`incy` 是以复数元素为单位的
非零步长。具体校验顺序和返回码见 Host 侧设计。`n == 0` 时接口直接返回成功，不访问或校验
handle、x、y 和步长。

## 需求拆解

1. 支持 COMPLEX64 的按位复制，保持实部、虚部顺序及 y 的非写区域；
2. 支持 `incx`、`incy` 的所有正、负非零组合；
3. 连续输入输出采用双缓冲搬运，其他步长采用 Compact 离散搬运；
4. 根据运行时 AIV 数量和 UB 容量完成多核与核内切分；
5. 使用 CSV、CBLAS 参考结果和 msprof 完成功能、正确性及性能验证。

# 详细设计（required）

## 算子分析

### 数学公式

接口以缓冲区低地址为基址。对逻辑索引 `i`，其中 `0 <= i < n`，物理索引为：

$$
P(i, inc) =
\begin{cases}
i \cdot inc, & inc > 0 \\
(n - 1 - i) \cdot |inc|, & inc < 0
\end{cases}
$$

复制语义为：

$$
y[P(i, incy)].real = x[P(i, incx)].real
$$

$$
y[P(i, incy)].imag = x[P(i, incx)].imag
$$

这里的等号表示实部和虚部均按 32 位对象表示逐位复制。`n > 0` 时，x、y 所需的物理跨度分别为
`1 + (n - 1) * abs(incx)` 和 `1 + (n - 1) * abs(incy)` 个复数元素。

### 支持数据类型

算子仅支持 COMPLEX64。每个逻辑元素的物理布局为 `{float real; float imag;}`，大小为 8 B；
不支持其他数据类型或类型转换。

### 支持形状

x、y 的逻辑形状均为一维 `[n]`，不支持广播、矩阵前导维或额外 Tensor 描述。

### 计算与带宽特征

本算子是 AIV-only 数据搬移算子，不使用 Cube。连续路径主要使用 MTE2（GM 到 UB）和
MTE3（UB 到 GM）；离散且输入输出步长符号不同时，还需要在 UB 中对实部、虚部分别逆序。
每个逻辑元素从 GM 读取 8 B、向 GM 写入 8 B，理论最小流量为 16 B/元素，因此大规模连续
用例主要受 HBM 带宽和 MTE 流水效率限制。

## 算子实现

### 实现方案

Host 负责参数校验、分核、tileSize 计算及 Kernel 下发；AIV Kernel 根据步长选择连续路径或
Compact 离散路径。两条路径共用一个 Kernel 入口和一份 Tiling 数据。

```text
aclblasCcopy
  -> 参数校验
  -> 查询 AIV 数量和 UB 容量
  -> 计算分核、tileSize 和可选 offset 表
  -> 在 handle stream 下发 ccopy_kernel
       |- 连续双缓冲路径
       `- Compact 通用步长路径
```

### Host 侧设计

#### 参数检查

| 顺序 | 条件 | 返回值 |
| ---: | --- | --- |
| 1 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 2 | `n == 0` | `ACLBLAS_STATUS_SUCCESS` |
| 3 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 4 | `x == nullptr` 或 `y == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `incx == 0` 或 `incy == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | 平台报告的 AIV 核数为 0 | `ACLBLAS_STATUS_EXECUTION_FAILED` |

Host 不检查 Device 指针的实际分配长度、地址归属和生命周期，这些由调用方保证。

#### Tiling 数据

```cpp
struct CcopyTilingData {
    uint32_t totalN;
    uint32_t perCoreN;
    uint32_t extraBlockCores;
    uint32_t tailElements;
    uint32_t tileSize;
    int32_t incx;
    int32_t incy;
};
```

`ELEMENTS_PER_BLOCK` 为 4 个复数，即 32 B，用于计算连续搬运的整块长度。
`CORE_ELEMENTS_PER_BLOCK` 为 8 个复数，即 64 B，用于分配各核的逻辑区间。这里保证的是
每核逻辑起始偏移按 8 个复数切分。对 `(+1,+1)` 连续路径，这等价于相对 x/y 基址按 64 B
粒度切分；只有传入基址本身满足 64 B 对齐时，各核的连续 GM 起始地址才满足 64 B 对齐。
该结论不用于描述负步长下的物理低地址起点。

#### 分核策略

Host 运行时查询平台 AIV 数量：

```text
numBlocks = min(n, aivCoreNum)
```

Ascend 950PR 自测环境返回 56 个 AIV，因此大规模用例启动 56 个 AIV block；代码不写死核数。
令 `A = 8`、`B = numBlocks`：

```text
rawPerCore      = n / B
perCoreN        = floor(rawPerCore / A) * A
leftover        = n - perCoreN * B
extraBlockCores = leftover / A
tailElements    = leftover % A
```

前 `extraBlockCores` 个核各增加 8 个元素，最后一个核承担剩余的 `0..7` 个元素。第 c 个核的
逻辑起点和任务量为：

```text
offset(c) = c * perCoreN + min(c, extraBlockCores) * 8
count(c)  = perCoreN
          + (c < extraBlockCores ? 8 : 0)
          + (c == B - 1 ? tailElements : 0)
```

该策略避免把不足 8 个元素的尾部拆到多个核。

#### UB 与 tileSize

`ubSize` 初始值为 248 KiB；平台管理器有效时，Host 调用 `GetCoreMemSize` 获取实际 UB 容量并
覆盖该值。公共接口在平台管理器无效、无法取得 AIV 数量时会先返回失败，因此正常 Tiling 路径
使用平台查询值；当前实现假定 `GetCoreMemSize` 查询成功。计算前预留 256 B，tileSize 向下取
8 个复数的整数倍。下表中的数值以本次 950PR 环境的 248 KiB UB 为例：

| Host 判定场景 | UB 估算或命令限制 | tileSize | 248 KiB 下的结果 |
| --- | --- | --- | ---: |
| `incx == 1 && incy == 1` | 两个 queue 槽位，共 16 B/元素 | `floor((UB-256)/16/8)*8` | 15,856 |
| 仅一侧等于 `+1` | queue 16 B、逆序临时数据 8 B、offset 4 B，共 28 B/元素 | `floor((UB-256)/28/8)*8` | 9,056 |
| 两侧均不等于 `+1` | Compact 单命令 `blockCount <= 4095` | `floor(4095/8)*8` | 4,088 |

后两种分类只用于计算安全的 tileSize；Kernel 中所有非 `(+1,+1)` 组合均进入 Compact 路径。
15,856、9,056 和 4,088 都是单次 tile 的容量，不是单核总处理量；单核数据超过 tileSize 时
会循环处理多个 tile。

#### 可选 offset workspace 与 Kernel 下发

当输入、输出步长符号不同时，需要在 tile 内逆序：

```text
needReorder = (incx < 0) XOR (incy < 0)
offset[i]   = (tileSize - 1 - i) * sizeof(float)
```

Host 优先在 Host 内存中生成 offset 表，再复制到 handle 的有效 Device workspace，所需空间为
`tileSize * 4 B`。以本次 248 KiB UB 为例，单边 `+1` 场景需要 36,224 B，两侧均非 `+1` 场景
需要 16,352 B；这些数值随平台 UB 和 tileSize 变化，不是实现的固定上限。workspace 不可用或
容量不足时，Kernel 在 UB 中生成同一张表。

连续路径不准备 offset 表。异号步长且 workspace 可用时，Host 先生成表并通过 `aclrtMemcpy`
复制到 Device workspace，随后在 `handle->stream` 上异步下发 Kernel。接口不显式调用
`aclrtSynchronizeStream` 或 `aclrtSynchronizeDevice`；Kernel 执行期错误由调用方后续同步操作暴露。

### Kernel 侧设计

#### 初始化与路径选择

Kernel 标记为 `KERNEL_TYPE_AIV_ONLY`。每个 block 根据 Tiling 计算 `myOffset`、`myCount`，并用
64 位中间值计算物理跨度和 GM 偏移；步长在取绝对值前提升为 `int64_t`，避免 `INT_MIN` 直接取负
产生 32 位溢出。

GM 在 Kernel 中映射为 FP32 视图，每个复数对应相邻两个 float。Kernel 初始化两个槽位的
`TQueBind<VECIN,VECOUT>`；非连续路径额外初始化 Gather 临时缓冲区。

#### 连续路径

`incx == +1 && incy == +1` 时进入连续路径：

- 长度为 32 B 整数倍的主体使用 `DataCopy`；
- 剩余 1 至 3 个复数使用 `DataCopyPad`，只向 y 写回有效字节；
- 完整 tile 使用两个 queue 槽位执行 Prime-Pump-Drain，使 tile i 的 MTE2 读取与 tile i-1 的
  MTE3 写回重叠；
- 完整 tile 写回后，尾 tile 使用 `ContinuousIteration` 处理。

```text
Prime:  读取 tile 0
Pump:   读取 tile i       || 写回 tile i-1
Drain:                         写回最后一个完整 tile
Tail:   读取尾 tile       ->  写回尾 tile
```

该路径没有 Vector Compute 阶段，队列用于管理 MTE2/MTE3 缓冲区所有权和同步。

#### 离散与负步长路径

其他步长组合使用 `DataCopyPad<..., PaddingMode::Compact>`。每个 tile 在 UB 中拆成 real 和 imag
两个 FP32 平面。每个 Compact block 搬运一个 4 B 分量，相邻 block 之间需要跳过的字节数为：

```text
(2 * abs(inc) - 1) * sizeof(float)
```

因此，相邻同分量起始地址之间的距离为 `2 * abs(inc) * sizeof(float)`。

单个 Compact 命令最多处理 4,088 个逻辑元素；较大的 tile 在一次 `DiscreteIteration` 中拆为
多个 Compact batch。写回命令分别设置 real、imag 的目标 stride，只覆盖 y 的有效逻辑位置。

对逻辑区间 `[elementOffset, elementOffset + dataCount)`，物理低地址起点为：

```text
inc > 0: elementOffset * abs(inc)
inc < 0: (totalN - elementOffset - dataCount) * abs(inc)
```

Compact 按物理地址递增方向访问。incx、incy 同号时，输入和输出物理顺序一致；两者异号时，
real 和 imag 分别逆序。offset 起点满足 8 元素对齐时，当前 tile 使用 `Gather`；否则整个 tile
使用 `GetValue/SetValue` 逐元素处理。MTE2、Vector/Scalar、MTE3 之间通过事件同步。

#### 按位复制

Kernel 不执行浮点算术、共轭或类型转换。连续路径按交错 FP32 搬运，离散路径分别搬运 real、imag
的 32 位对象表示。

## 支持硬件

| 支持的芯片版本 | SoC/架构 | 状态 |
| --- | --- | --- |
| Ascend 950PR | `ascend950` / `arch35` | 支持 |

当前设计不声明对其他 SoC、架构或 CANN 版本的新增支持。

## 算子约束限制

1. x、y 的实际容量、地址有效性和异步执行期生命周期由调用方保证，Host 不做检查；
2. 负步长调用传入物理缓冲区的低地址；
3. 部分重叠且映射不同的 x/y 区域不提供 `memmove` 顺序保证。

# 可维可测分析

## 精度标准/性能标准

| 验收项目 | 判定方式 | 标准来源 |
| --- | --- | --- |
| 正确性 | 通过测试环境链接的 CBLAS `cblas_ccopy` 生成参考结果；完整 y 存储区中 real、imag 分别按 4 B 比较，要求逐位相等（`max_abs_error = 0`），非写区域保持不变 | 任务书 §2.1、§3.2、§3.5 |
| 性能 | warmup 后采集大于 50 次 `ccopy_kernel/KERNEL_AIVEC` 设备任务并取平均；当前自测按 `NPU avg_us <= gpu_ms * 1000 / 0.4` 判定 | 配套 README、`gpu_baseline.csv` 和验证脚本 |
| 内存 | 任务书未规定硬性阈值；报告实际资源用量。连续路径不使用 workspace，异号路径按 tileSize 使用可选 offset workspace | 任务书 §3.4、§4 |

本算子不包含数值计算，因此正确性采用比通用 FLOAT32 `rtol/atol` 更严格的逐位比较。

## 功能与正确性验证

`arch35/ccopy_test.csv` 驱动 C++ GTest，CPU 侧调用测试环境链接的 `cblas_ccopy` 生成参考结果。
测试流程为：

1. 按 n 和步长计算 x、y 的完整物理跨度，并在前后设置保护区；
2. 调用 NPU 接口并将完整 y 分配区复制回 Host；
3. 对每个存储位置的 real、imag 分别比较 4 B；
4. 对错误用例检查返回码，NullHandle 使用独立测试；
5. `verify_accuracy.py` 检查 CSV schema、用例注册集合和 GTest XML，拒绝失败、错误或跳过用例。

CSV 覆盖小尺寸、尺寸边界、正负步长组合、随机值、全零、正负交替、极值、Inf、NaN、地址偏移、
空指针、零步长和负 n。`verify_accuracy.py` 在运行期间生成并核查临时 GTest XML；正式提交时应
保存与对应代码版本匹配的终端日志和自测报告，供验收复核。

## 性能验证

`verify_performance_msprof.py` 调用 CANN 官方 `msprof`，使用 `--task-time=l0` 采集设备任务，
在导出的 `task_time_*.csv` 中筛选：

```text
kernel_name = ccopy_kernel
kernel_type = KERNEL_AIVEC
```

`task_time(us)` 表示 Device Kernel 任务耗时，不包含 GTest 数据生成、H2D/D2H、CBLAS 参考结果和
结果比较；它也不等同于指令级 on-core 时间。默认每个 case 执行 5 次 warmup 和 55 次有效采样，
输出平均值、中位数和 P95。

配套 README、`ccopy_test.csv`、`gen_csv.py` 和 `gpu_baseline.csv` 使用
`n=1,048,576 / 2,097,152 / 4,194,304` 三条典型连续用例，并按 GPU 基线除以 0.4 得到 NPU
限值。任务书与配套材料的尺寸不一致，当前自测采用配套材料口径；最终验收口径由任务发布方确认，
性能结果以对应代码版本保存的 CSV 和 msprof 记录为准。

## 兼容性分析

- 复用已有 `aclblasCcopy` 声明、`aclblasComplex` 类型和错误码，API/ABI 不变；
- arch35 实现与其他架构目录隔离；
- 沿用 ops-blas 的 handle、stream、workspace、日志和构建机制；
- 测试工程沿用 copy 族的 CSV 和 GTest 模式。

## 参考资料

1. 随社区任务发放的《aclblasCcopy 算子开发任务书》；
2. [cuBLAS Level-1 `cublasCcopy` 接口说明](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-copy)；
3. [Netlib BLAS `ccopy.f`](https://www.netlib.org/blas/ccopy.f)；
4. [ops-blas 开源仓](https://gitcode.com/cann/ops-blas)中的 `include/cann_ops_blas.h`、
   `include/cann_ops_blas_common.h`、`blas/copy/arch35/` 和 `test/copy/ccopy/`；
5. [Ascend C 算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)；
6. [Ascend C 基础 API 文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)。

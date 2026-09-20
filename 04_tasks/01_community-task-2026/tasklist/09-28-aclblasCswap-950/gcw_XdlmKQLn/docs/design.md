# 需求背景（required）

## 需求来源

本需求来源于 CANN 社区 `aclblasCswap 950 算子开发任务`。目标是在
Ascend 950PR、CANN 9.1.0 环境中，基于 `cann/ops-blas` 仓库实现
`aclblasCswap` 的 `arch35` 版本，并交付算子实现、测试代码、设计文档和
自测报告。

接口使用 `include/cann_ops_blas.h` 中已有的公共声明：

```cpp
aclblasStatus_t aclblasCswap(
    aclblasHandle_t handle,
    int n,
    aclblasComplex* x,
    int incx,
    aclblasComplex* y,
    int incy);
```

## 背景介绍

`aclblasCswap` 用于原地交换两个 COMPLEX64 向量的逻辑元素。每个复数由
两个 float32 分量组成；该算子只搬运数据，不执行浮点计算，因此实部、
虚部和 NaN payload 等原始位模式都必须保持不变。

基线版本 `7a80f5cde08ca51edd081d0c1f10ef83268a5303` 已包含公共 API 声明和
`arch22` 实现，但没有 Ascend 950PR 对应的 `blas/swap/arch35/` 实现，也
没有该产品上的 CSV 驱动精度测试和微秒级性能测试。本设计在不改变公共
接口和其他产品实现的前提下补齐 `arch35` 路径。

# 需求分析（required）

## 需求描述

给定两个设备侧 COMPLEX64 向量 `x`、`y`，交换各自的 `n` 个逻辑元素。
`incx`、`incy` 表示相邻逻辑元素在物理存储中的步长，可为正数或负数。
负步长遵循 Netlib BLAS 语义，从对应物理向量的尾部元素开始访问。

接口必须满足以下行为：

- `n <= 0` 是合法 no-op，在其余参数校验前直接返回成功；
- `n > 0` 时，空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
- `n > 0` 时，空 `x`/`y` 或零步长返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 在 handle 绑定的 stream 上异步执行，接口内部不申请临时设备内存、
  不执行 stream 同步；
- 所有被交换元素必须 bit-exact，步长空洞和对齐前缀不得被修改。

## 需求拆解

1. 增加 Ascend 950PR 的 Host 侧参数校验、任务配置和异步 kernel 启动。
2. 增加连续场景的低开销 SIMT kernel，满足三个官方大 shape 性能门槛。
3. 增加任意正、负非零步长的通用 SIMT kernel，使用 64 位物理索引避免
   大跨度地址计算溢出。
4. 以 64 位无符号整数搬运一个 COMPLEX64 元素，保证实部、虚部及特殊值
   的位模式完全不变。
5. 接入 1200 条官方 CSV 用例，并补充 quick return、空 handle、特殊值、
   混合步长、步长空洞及对齐偏移测试。
6. 使用设备事件进行 20 次预热和 100 次有效采样；完整运行 200 条性能
   用例，并对三个任务书指定 case 做硬门槛判定。

# 详细设计（required）

## 算子分析

### 数学公式

令逻辑下标 `i` 的范围为 `0 <= i < n`。正步长的起点为 0；负步长的起点
为物理向量尾部：

```text
x_start = (incx < 0) ? (n - 1) * |incx| : 0
y_start = (incy < 0) ? (n - 1) * |incy| : 0
x_index(i) = x_start + i * incx
y_index(i) = y_start + i * incy

tmp                 = x[x_index(i)]
x[x_index(i)]       = y[y_index(i)]
y[y_index(i)]       = tmp
```

每个向量所需的最小物理元素数为：

```text
span(n, inc) = 1 + (n - 1) * |inc|, n > 0
```

### 支持数据类型

| 对象 | 数据类型 | 存储说明 |
| --- | --- | --- |
| x | COMPLEX64 | 实部、虚部分别为 float32，共 8 Byte |
| y | COMPLEX64 | 实部、虚部分别为 float32，共 8 Byte |
| n、incx、incy | int | Host 侧标量 |

Kernel 将一个 COMPLEX64 元素视为不透明的 `uint64_t` payload。实现中没有
浮点运算和类型转换。

### 支持形状

- 逻辑 shape：一维 `[n]`；`n` 在运行时传入。
- 物理长度：分别为 `1 + (n - 1) * |incx|` 和
  `1 + (n - 1) * |incy|`。
- 支持 `incx`、`incy` 的任意正、负非零组合。
- `n <= 0` 不访问 `x`、`y`，直接成功返回。

## 算子实现

### 实现方案

实现采用 Host 直调的双 kernel 方案：

- `incx == 1 && incy == 1`：进入连续访存 kernel，逻辑下标可直接作为
  物理下标，避免通用步长计算进入性能热路径；
- 其他合法步长：进入通用步长 kernel，先计算负步长起点，再以 64 位有
  符号索引执行 grid-stride 循环。

每个 SIMT 线程负责一个或多个逻辑元素。单元素依次执行两次 8 Byte 读取
和两次 8 Byte 写回，总全局内存流量为 32 Byte。一次交换的两个读取均在
任一写回之前完成，避免覆盖本线程尚未保存的数据。

#### host侧设计

1. 首先判断 `n <= 0`，满足时立即返回 `ACLBLAS_STATUS_SUCCESS`。
2. 按 handle、指针、步长顺序完成参数校验，并返回公共状态码。
3. 通过 `GetAivCoreCount()` 获取可用 AIV 核数；核数为 0 时返回执行失败。
4. 块数取 `ceil(n / SIMT_MIN_THREAD_NUM)` 与 AIV 核数的较小值，至少为 1；
   每块线程数按 `SIMT_MIN_THREAD_NUM` 向上对齐，并限制在
   `SIMT_MAX_THREAD_NUM` 内。
5. 根据步长选择连续或通用 kernel，使用 `handle->stream` 异步启动。

Host 路径不创建 workspace、不进行 Host/Device 中间拷贝，也不调用任何
同步接口。

#### kernel侧设计

连续 kernel 使用以下 grid-stride 结构：

```text
index = blockIdx.x * blockDim.x + threadIdx.x
stride = gridDim.x * blockDim.x
for (; index < n; index += stride) {
    x_value = x[index]
    y_value = y[index]
    x[index] = y_value
    y[index] = x_value
}
```

通用 kernel 在循环前计算 `x_start`、`y_start` 及对应的物理 grid stride，
循环内只做两次 64 位 load、两次 64 位 store 和索引递增。负步长先从
`(n - 1) * |inc|` 开始，再向物理下标 0 方向移动。地址计算使用 `int64_t`
以覆盖 `int` 入参允许的跨度。

未采用 UB/MTE 双缓冲方案：在 Ascend 950PR 上，直接 SIMT 候选的实测结果
已明显低于全部官方门槛，增加第二条数据路径只会提高代码复杂度和维护
成本，未带来验收收益。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

- `n <= 0` 为合法 no-op；此时允许 handle、`x`、`y` 为空以及步长为 0。
- `n > 0` 时 handle、`x`、`y` 必须有效，`incx`、`incy` 必须非零。
- 调用方必须按 `span(n, inc)` 提供可访问的设备内存。
- 算子在 handle 绑定的 stream 上异步执行；Host 读回结果前须同步该 stream。
- 本任务不要求两个输入向量存在部分重叠时的顺序语义。
- 仅支持接口声明指定的 COMPLEX64，不进行广播或额外 leading-dimension
  padding；向量非连续性由步长表达。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | x、y 的实部和虚部均须 bit-exact；等价于 0 ULP、0 绝对误差 | 任务书 §3.2 |
| 精度覆盖 | 1000 条官方精度用例全部通过，并补充 quick return、空 handle、特殊位模式、混合步长和对齐偏移 | 任务书 §3.5 |
| 性能采样 | 每个 case 预热 20 次、设备事件有效采样 100 次 | 任务书 §3.3 |
| 性能门槛 | n=1048576/2097152/4194304 时平均耗时分别不高于 23.11/63.82/166.11 us | 任务书 §3.3 |
| 内存标准 | 算子 workspace 为 0 Byte | 任务书 §3.4 |

测试使用完整物理分配做 `memcmp`，因此不仅检查参与交换的元素，也检查
步长空洞及对齐前缀未被修改。性能计时仅覆盖同一 stream 上的 kernel
序列，不计 Host 数据准备、内存申请、golden 计算和结果比对。

## 兼容性分析

- 公共 API 签名保持不变，不增加 Ascend 950PR 私有接口。
- 新实现仅位于 `blas/swap/arch35/`；原有 `arch22` 源码保持不变。
- CMake 在 `ascend950` 下启用新增 GTest；其他 SoC 继续使用原测试路径。
- 返回值沿用 `cann_ops_blas_common.h` 中已有状态码，stream 行为与仓库
  现有句柄式 BLAS 接口一致。

## 本次交付及历史设计

本设计对应 2026-09-19 的 Ascend 950PR 交付。此前同一账号的设计位于
`tasklist/aclblasCswap/gcw_XdlmKQLn/docs/design.md/design.md`，保留为历史记录。
本次按任务编号 `09-28-aclblasCswap-950` 提交实际采用的 SIMT 设计，替代旧文档中的 UB/Compact 方案描述。

- 实现提交：`f501225a0187e6a484533eea1d3dbb136a28b07e`。
- 对应代码分支：`gcw_XdlmKQLn/ops-blas` 的 `delivery/cswap950-20260919`。
- 交付包：`aclblasCswap_Ascend950PR_delivery_20260919.zip`。
- 精度自测：官方 1000/1000，GTest 1004/1004；完整 200 条性能结果随包提交。
- 三个指定性能 case 两轮结果（us）：11.386/11.596、20.816/20.538、39.100/40.767，分别低于 23.11、63.82、166.11。
- 上述为交付包内设备自测结果，不代表平台验收或本 PR 已通过。

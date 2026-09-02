# 1 需求背景（required）

## 1.1 需求来源

本任务来源于 8 月社区任务《aclblasCdotc 算子开发任务书》，目标是在 Ascend 950PR、CANN 9.1.0 环境中，基于 ops-blas 实现单精度复数向量共轭点积接口 `aclblasCdotc`。

## 1.2 背景介绍

### 1.2.1 aclblasCdotc算子实现优化

基于 ops-blas 中已有的 Cdotc 能力，使用 Ascend C 为 Ascend 950PR 新增 arch35 实现，并补齐任意非 0 正负步长支持。

历史实现路径为 `ops-blas/blas/dot/arch22/`；本次实现路径为 `ops-blas/blas/dot/arch35/`。公共接口声明位于 `include/cann_ops_blas.h`。

### 1.2.2 aclblasCdotc算子TBE实现现状分析

本任务的历史实现位于 ops-blas，而非独立 TBE 目录；本小节沿用模板的“现状分析”组织方式，对已有 Cdotc 能力进行说明。

| 参数 | 参数含义 | 数据类型 | 当前支持能力 | 约束 | 逻辑形状 |
| --- | --- | --- | --- | --- | --- |
| `handle` | 执行句柄 | `aclblasHandle_t` | 已支持 | 必须有效 | - |
| `n` | 参与计算的元素数 | `int` | 已支持 | `n >= 0` | 标量 |
| `x`、`y` | 输入复数向量 | COMPLEX64 | 已支持 | 历史实现仅支持单位步长 | `[n]` |
| `incx`、`incy` | 输入元素步长 | `int` | 需扩展 | 本次支持任意非 0 正负步长 | 标量 |
| `result` | 输出复数标量 | COMPLEX64 | 已支持 | 必须为有效 Device 地址 | `[1]` |

计算公式：`result = Σ conj(x[i]) * y[i]`。

### 1.2.3 aclblasCdotc算子功能分析

算子功能：计算两个 COMPLEX64 向量的共轭点积，`result = Σ conj(x[i]) * y[i]`。

输入：`handle`、`n`、`x`、`incx`、`y`、`incy`。

输出：`result`，一个 COMPLEX64 标量。

支持数据类型：COMPLEX64。

支持步长：任意非 0 的正、负步长；不支持 broadcast。

# 2 需求分析（required）

## 2.1 需求描述

公共接口如下：

```cpp
aclblasStatus_t aclblasCdotc(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    const aclblasComplex* y,
    int incy,
    aclblasComplex* result);
```

| 参数 | 说明 |
| --- | --- |
| `handle` | ops-blas 句柄，提供执行 stream 和 workspace |
| `n` | 参与计算的复数元素数量，要求 `n >= 0` |
| `x`、`y` | Device 侧 COMPLEX64 输入向量；计算时对 `x` 取共轭 |
| `incx`、`incy` | 以复数元素为单位的步长，`n>0` 时必须非 0 |
| `result` | Device 侧 COMPLEX64 输出标量，必须有效 |

输入、输出均位于 Device 内存。调用方读取 `result` 前需同步 handle 绑定的 stream。

## 2.2 需求拆解

1. 正确计算 `conj(x[i]) * y[i]`，并分别累加实部、虚部。
2. 支持连续、非连续以及正负混合步长。
3. `n=0` 时不读取 `x`、`y`，不校验步长；在 `result` 有效时通过同一 stream 异步置零后返回成功。
4. `handle` 为空、`n<0`、`result` 为空，以及 `n>0` 时的空输入或零步长，均返回参数错误。
5. 连续单位步长路径满足任务书性能指标；通用步长路径以功能和精度正确为目标。
6. 精度 golden 使用 CBLAS `cblas_cdotc_sub` 生成，实部、虚部分别比对。

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

设 `x = xr + xi*j`、`y = yr + yi*j`，则：

```text
conj(x) * y = (xr * yr + xi * yi) + (xr * yi - xi * yr) * j
```

Kernel 分别累加：

```text
real += xr * yr + xi * yi
imag += xr * yi - xi * yr
```

### 3.1.2 支持数据类型

| 对象 | 数据类型 | 物理布局 |
| --- | --- | --- |
| `x`、`y` | COMPLEX64 | `{real0, imag0, real1, imag1, ...}`，每个元素 8 字节 |
| `result` | COMPLEX64 | `{real, imag}` |
| workspace 与核内累加 | FLOAT32 | 实部、虚部分开保存 |

### 3.1.3 支持形状

`x`、`y` 的逻辑形状为一维 `[n]`，`result` 为一个复数标量；不涉及 batch、broadcast 和 leading dimension。

调用方传入的 `x`、`y` 指向各自物理存储的低地址。对逻辑下标 `i∈[0,n)`，物理复数下标为：

```text
xIndex = incx > 0 ? i * incx : (n - 1 - i) * abs(incx)
yIndex = incy > 0 ? i * incy : (n - 1 - i) * abs(incy)
```

该规则与 Netlib `cdotc` 的负步长起始位置一致。地址计算使用 `int64_t`，实际存储长度由调用方保证：`1 + (n - 1) * abs(inc)`。

## 3.2 算子实现

### 3.2 实现方案

Host 完成参数校验、分核、workspace 检查和 Kernel 发射。各 AIV 核计算本核的实部、虚部部分和并写入 workspace；全部核完成后，core 0 归约 workspace 并写回 `result`。

```text
aclblasCdotc
  ├─ 参数校验
  ├─ n=0：在绑定 stream 上异步置零并返回
  └─ 启动 AIV Kernel
       ├─ incx=1 且 incy=1：连续向量化路径
       ├─ 其他步长：通用寻址路径
       ├─ 各核写入实部、虚部部分和
       └─ core 0 跨核归约并写入 result
```

#### 3.2.1 Host 侧设计

##### 3.2.1.1 参数校验与快速返回

参数按以下顺序处理：

1. `handle` 为空，返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n<0` 或 `result` 为空，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `n==0` 时，不访问 `x`、`y`，不校验 `incx`、`incy`；调用 `aclrtMemsetAsync(result, sizeof(aclblasComplex), 0, sizeof(aclblasComplex), handle->stream)` 将结果置零后返回成功。
4. `n>0` 时，`incx==0`、`incy==0` 或 `x`、`y` 为空，返回 `ACLBLAS_STATUS_INVALID_VALUE`。

上述置零操作与后续 Kernel 使用同一 stream，不在 Host 侧同步。

##### 3.2.1.2 分核与 workspace

参与计算的核数为：

```text
useCoreNum = min(GetAivCoreCount(), n)
```

每个核根据核号自行计算连续的逻辑区间，前 `remainder` 个核各多处理一个元素，使各核负载差不超过一个元素。

Host 下发以下 tiling 数据：

```cpp
struct CdotcTilingData {
    int64_t n;
    int64_t incx;
    int64_t incy;
    uint32_t useCoreNum;
};
```

workspace 分为两段，分别存储每个核的实部和虚部部分和：

```text
workspace[0 .. useCoreNum)              ：实部部分和
workspace[useCoreNum .. 2 * useCoreNum) ：虚部部分和
```

所需空间为 `2 * useCoreNum * sizeof(float)`；复用 handle 已有 workspace，不在接口调用中申请 Device 内存。Kernel 采用 `KERNEL_TYPE_AIV_ONLY`，`blockDim` 等于 `useCoreNum`。

##### 3.2.1.3 路径选择

不设置 tilingKey。Kernel 根据 `incx`、`incy` 选择路径：两者均为 1 时走连续向量化路径，其他组合走通用步长路径。

#### 3.2.2 Kernel 侧设计

##### 3.2.2.1 连续向量化路径

连续路径按 tile 将交织存储的 `x`、`y` 搬入 UB，使用 `GatherMask` 分离实部、虚部，完成复数乘加后对实部、虚部分别 `ReduceSum`。

tile 大小固定为 4800 个复数。变长 UB 空间为：x 双缓冲 4 倍 tile、y 双缓冲 4 倍 tile、实虚分量缓冲 4 倍 tile，共 `12 * 4800 * sizeof(float)=230400` 字节；加上归约临时区和累加器后仍在 248 KiB UB 预算内。最后一个不完整 tile 只参与实际元素的计算。

##### 3.2.2.2 通用步长路径

通用路径按逻辑下标计算 `xIndex`、`yIndex`，每次读取一个 COMPLEX64 元素，完成共轭乘加并累加到本核结果。该路径覆盖正负单位步长、正负非单位步长，以及两输入不同符号或不同绝对值的步长。

##### 3.2.2.3 跨核归约

每个核将实部、虚部部分和写入各自 workspace 槽位。所有核通过 CrossCore 同步后，core 0 分别归约实部、虚部，组合为 `{real, imag}` 并写入 `result`。归约顺序由分核和 tile 划分决定，不保证逐位确定性，符合任务书要求。

代码按现有 ops-blas 目录组织：

```text
ops-blas/
├── blas/dot/arch35/              # Cdotc Host、Kernel 与 tiling
├── test/dot/cdotc/arch35/        # Cdotc CSV 与 GTest 用例
└── blas/dot/README.md            # 950PR 支持信息
```

## 3.3 支持硬件

| 硬件 | 支持情况 |
| --- | --- |
| Ascend 950PR | 支持，CANN 9.1.0 |

## 3.4 算子约束限制

1. 仅支持 COMPLEX64。
2. `n` 必须大于等于 0；`n>0` 时步长必须非 0。
3. `result` 必须为有效 Device 地址；`n>0` 时 `x`、`y` 也必须有效。
4. 输入实际存储长度与 `result` 是否和输入地址重叠由调用方保证。
5. 接口异步执行；浮点归约不保证逐位确定性。

# 4 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 使用 CBLAS `cblas_cdotc_sub` 生成 golden；COMPLEX64 的实部、虚部分别按 FLOAT32 标准比对：`rtol=2^-10`、`atol=2^-16`、matched ratio 不低于 0.99、max abs error limit 为 `1e-2` 或 `32 * ULP`。 | 任务书 3.2 节、生态算子精度验收标准 |
| 性能标准 | 在 Ascend 950PR、CANN 9.1.0 环境中 warmup 后，对同一 stream 有效采样超过 50 次并取平均。`n=1048576/2097152/4194304` 且 `incx=incy=1` 时，平均耗时分别不高于 `24.61/29.4575/74.3525 us`。 | 任务书 3.3 节 |

测试复用 ops-blas 的 CSV 与 GTest 框架，覆盖基础功能、shape、正负步长、对齐偏移、特殊值、异常参数、扩展精度、性能和多次调用内存检查。随机输入按任务书要求覆盖均匀分布与正态分布；精度、性能和内存结果以实际自测报告为准。

## 4.2 兼容性分析

- 保持公共 API 和 `aclblasComplex` ABI 不变；
- 不修改 arch22 Cdotc/Cdotu 实现；
- 使用既有 CMake、CSV、GTest 和 workspace 机制，不引入新的生产依赖；
- 合入时更新 `blas/dot/README.md` 的 950PR 支持信息与步长约束。

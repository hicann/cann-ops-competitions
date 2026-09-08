# aclblasScnrm2 算子设计文档

# 需求背景

`aclblasScnrm2` 计算单精度复数向量的二范数，目标平台为 Ascend 950PR。设
$x_i=x_{i,r}+j x_{i,i}$，输出为：

$$
\lVert x\rVert_2=
\sqrt{\sum_{i=0}^{n-1}\left(x_{i,r}^2+x_{i,i}^2\right)}.
$$

输入由交错存储的 FP32 实部和虚部组成，输出为一个非负 FP32 标量。直接平方求和会让
大数提前溢出、小数提前下溢，因此实现需要在归约过程中进行缩放。

# 接口与行为

## 接口

```cpp
aclblasStatus_t aclblasScnrm2(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    float* result);
```

`handle` 位于 Host，`x` 和 `result` 位于 Device。正常计算在 handle 绑定的 stream 上异步
执行，接口内不进行 Host 同步。

## 参数

| 参数 | 说明 |
| --- | --- |
| `handle` | ops-blas 上下文，不能为空 |
| `n` | 复数元素个数，`n <= 0` 时快速返回 |
| `x` | COMPLEX64 输入，`n > 0` 时不能为空 |
| `incx` | 相邻逻辑元素的步长，`n > 0` 时不能为 0 |
| `result` | FP32 输出标量，不能为空 |

检查顺序为 handle、result、n、incx、x、输入跨度。`n <= 0` 时不启动计算 Kernel，只把
result 异步写为 `+0.0f`。其余非法输入返回对应状态码。

跨度计算使用 64 位中间量。先扩展 `incx` 再取绝对值，避免 `INT_MIN` 取反溢出。调用方
需要提供至少 `1 + (n - 1) * abs(incx)` 个复数元素的物理空间。

## 边界语义

| 场景 | 结果或行为 |
| --- | --- |
| 全零输入 | `+0.0f` |
| 输入包含 `NaN` | `NaN` |
| 无 `NaN`，但包含 `Inf` | `+Inf` |
| 有限结果超出 FP32 | `+Inf` |
| `incx < 0` | 按 `abs(incx)` 访问同一组元素 |
| 空指针、零步长或跨度溢出 | 返回错误，不启动计算 Kernel |

范数与遍历方向无关，因此正负步长共用同一套物理位置计算。Kernel 单独记录 `NaN` 和
`Inf`；两者同时存在时，`NaN` 优先。

# 实现方案

## 数值方法

每个复数拆成两个 FP32 分量。对有限分量取绝对值 `ax`，按数值大小累加到三个区间：

```text
ax >= tbigEff:           big += (ax * 2^-76)^2
ax < 2^-63:            small += (ax * 2^75)^2
2^-63 <= ax < tbigEff: medium += ax^2
```

大数缩小后平方，小数放大后平方，中等数直接平方。大值边界随输入长度调整：

```text
M = 2 * n
L = ceil(log2(M))
k = min(52, floor((126 - L) / 2))
tbigEff = 2^k
```

每个核产生 `big`、`medium`、`small`、`hasNaN` 和 `hasInf` 五项状态。合并后按以下方式
恢复结果：

```text
NaN                              -> NaN
Inf                              -> +Inf
big > 0                          -> 2^76 * sqrt(big + medium * 2^-152)
medium > 0 and small > 0         -> hypot(sqrt(medium), sqrt(small) * 2^-75)
small > 0                        -> sqrt(small) * 2^-75
otherwise                        -> sqrt(medium)
```

## 执行路径

Host 完成检查和切分后，根据 `abs(incx)` 选择路径：

| 条件 | 路径 |
| --- | --- |
| `abs(incx) == 1` | 连续向量路径 |
| `abs(incx) > 1` | 跨步线程路径 |

连续路径把交错复数视为 `2*n` 个连续 FP32 分量，成块搬入片上空间并完成分类、缩放和局部
归约。跨步路径用 64 位位置读取实部和虚部，每个线程处理若干逻辑元素，再做线程块内归约。

两条路径都只向工作区写一条每核结果。所有参与核同步后，由 0 号核完成最后合并并写回
result。整个过程只下发一次计算 Kernel。

## Host 侧

Host 侧负责：

1. 参数检查和快速返回；
2. 计算绝对步长、物理跨度和大值边界；
3. 从平台信息获取可用 Vector Core 数；
4. 按输入长度确定参与核数和每核区间；
5. 检查工作区并下发 Kernel。

小输入减少参与核数，大输入使用全部可用 Vector Core。切分结果通过固定大小的数据结构
传入 Kernel，包含 `n`、`absInc`、参与核数、每核元素数、尾部数量、路径和工作区大小。

每核工作区记录占 32 字节，保存三个累加值、两个特殊值标志和对齐字段：

```text
workspaceBytes = align32(useCoreNum * 32)
```

## Kernel 侧

连续路径使用双缓冲衔接搬运与计算，每个输入分量只从全局内存读取一次，不生成逐元素
中间结果。尾块只归约有效位置，补齐数据不参与特殊值判断。

跨步路径由线程独立计算五项局部状态，再通过树形方式合并。物理位置统一按
`logical * absInc` 计算，不在 Device 侧使用负地址递减。

所有启动的核都写出本核状态并到达同步点。同步后仅 0 号核读取工作区，分别合并三个数值
区间和两个标志，然后恢复最终范数。

# 验证设计

参考结果由 CBLAS/Netlib `scnrm2` 生成。普通 FP32 输出同时满足：

```text
abs(actual - golden) <= 2^-16 + 2^-10 * abs(golden)
abs(actual - golden) <= max(1e-2, 32 * ULP(abs(golden)))
```

每条标量结果单独判断；`NaN` 和 `Inf` 还需检查分类及符号。验证范围包括：

- 零长度、单元素、非对齐长度和大尺寸；
- `incx` 为 `±1`、`±2`、`±3`；
- 普通随机值、全零、极大值、极小值和大小数混合；
- 单独或混合出现的 `NaN`、`+Inf`、`-Inf`；
- handle、x、result 空指针，零步长和跨度溢出；
- 尾块、连续调用、多 stream 调用和输入边界保护。

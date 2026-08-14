# 需求背景（required）

## 需求来源与交付目标

本需求来源于 CANN 社区 2026 年 8 月任务「asc_elu 算子开发」（任务书：`04_tasks/01_community-task-2026/docs/202608/asc_elu_task_doc.md`）。

目标是基于 Ascend C C API（RegBase 编程模型）实现纯 Vector-Core 的 ELU（Exponential Linear Unit）算子，并交付至 asc-devkit 开源仓：

`examples/02_simd_c_api/03_c_api/02_reg_vector_compute/`

交付形态与该目录下的 `abs`、`div`、`mul`、`muls`、`cast` 等同级样例保持一致：

- 单个 `.asc` 文件同时包含 Device 核函数和 Host 调用样例；
- 配套提供 `CMakeLists.txt`、`data_utils.h`、`scripts/gen_data.py` 和 `README.md`；
- 不注册 GE/aclnn 算子，不采用 `op_host` / `op_kernel` / `op_api` 三段式结构，不产出自定义算子包（OPP）。

## 算子语义

ELU 是包含指数项的分段激活函数，语义以 Ascend C 官方算子库 `Elu` 为准：

$$
ELU(x)=
\begin{cases}
scale \cdot x, & x>0 \\
\alpha \cdot scale \cdot \left(e^{x\cdot inputScale}-1\right), & x\le 0
\end{cases}
$$

| 参数            | 含义                  | 类型     | 数据类型            | 说明                     |
| ------------- | ------------------- | ------ | --------------- | ---------------------- |
| `src`         | 输入张量 $x$            | Input  | FLOAT / FLOAT16 | ND 格式                  |
| `alpha`       | 激活系数 $\alpha$       | Input  | FLOAT           | 运行期标量                  |
| `scale`       | 缩放系数                | Input  | FLOAT           | 运行期标量                  |
| `input_scale` | 输入缩放系数 $inputScale$ | Input  | FLOAT           | 运行期标量                  |
| `dst`         | 输出张量                | Output | FLOAT / FLOAT16 | shape 与 `src` 一致，ND 格式 |

$x=0$ 归入 $x\le0$ 分支，结果为 0，与正分支在该点连续。

---

## 功能范围

使用 Ascend C C API（RegBase）实现 ELU 算子，支持 `float` 和 `float16`，可处理任意元素数的连续 ND 张量，并全程运行在 Vector-Core 上。

| 项    | 规格                                                                      |
| ---- | ----------------------------------------------------------------------- |
| 输入   | `x`：连续 ND 张量，元素数 $N$ 为任意非负整数，值域 $[-100,100]$，dtype ∈ {float, float16} |
| 输出   | `y`：shape、dtype 与 `x` 一致，ND 格式                                          |
| 标量参数 | `alpha`、`scale`、`input_scale`，均由 Host 侧在运行期传入，两条 dtype 路径均为 FLOAT     |
| 地址约束 | `x` 与 `y` 不支持地址重叠，即不支持 in-place                                         |
| 目标硬件 | Ascend 950PR / Ascend 950DT（`dav-3510`）                                 |

BFLOAT16 不在任务书要求范围内，本样例不实现。

## 关键要求

1. 支持 FLOAT32 和 FLOAT16 两种数据类型；
2. `alpha`、`scale`、`input_scale` 不得写死为编译期常量；
3. 支持 `shape=1`、非 32 字节对齐长度和超过单次 UB 缓冲容量的长度；
4. 精度满足《生态算子开源精度标准》experimental 档；
5. 功能逻辑与官方算子库 `elu` 对齐；
6. 全程禁止调用 Cube-Core；
7. Host 侧不得逐元素串行计算，逐元素运算全部由 Device 侧批量完成；
8. Local Memory 分配和 GM↔UB 搬运不得越界；
9. 提供可复现的编译、数据生成、精度测试和性能采集流程。

## 当前实现边界

- 仅接受连续存储；带 stride 的非连续张量需由调用方先连续化；
- 仅支持 `float` 和 `float16`；
- 元素计数使用 `uint32_t`，超过 `UINT32_MAX` 的输入不进入 Kernel。

---

# 详细设计（required）

## 实现方案

保留原分段语义，按官方实现的顺序先算负分支候选，再做分支选择：

1. $z=x\cdot inputScale$；
2. $negative=expm1(z)\cdot\alpha$；
3. $selected=(x>0)\ ?\ x:negative$；
4. $y=selected\cdot scale$。

两处设计取舍：

**乘 `scale` 放在选择之后，且不在 Host 侧预乘 `alpha*scale`。** 预乘会让合法有限入参提前溢出：$x=1,\alpha=2\times10^{38},scale=2$ 的正确结果是 2，预乘先得到 `Inf`，未选中的负分支再产生 `Inf\times0=NaN`。选择在乘 `scale` 之前完成，未选中分支的中间值不进入结果。

**三个标量在两条 dtype 路径上都保持 FLOAT。** 若 FLOAT16 路径把标量降为 `half`，$\alpha=60000,scale=1,inputScale=2\times10^{-8}$ 中的 `inputScale` 会舍入为 0，正确结果 $-0.12$ 将错误地变成 0。

$inputScale$ 无非负约束。其为负时负半轴指数自变量为正，指数项可能上溢；实现按原公式计算并传播可表示结果与 IEEE NaN/Inf。

## expm1 的近零处理

$e^z-1$ 在 $z\to0$ 时发生抵消，按 $\lvert z\rvert$ 分两路计算后选择：

| 区间                        | 计算方式                       |
| ------------------------- | -------------------------- |
| $\lvert z\rvert<0.1$      | Horner 多项式：float 五阶、float16 四阶 |
| $\lvert z\rvert\ge0.1$    | `asc_exp(z)` 后加常量 $-1$     |

float 五阶多项式为：

$$
expm1(z)\approx z\left(1+z\left(\frac12+z\left(\frac16+z\left(\frac1{24}+\frac{z}{120}\right)\right)\right)\right)
$$

float16 输出的量化误差更大，按官方实现省略五阶项，取四阶。

阈值、阶数、系数和 Horner 顺序均与官方实现 `cann/ops-nn` 的 `activation/elu/op_kernel/arch35/elu_dag.h`（`arch35` 即 dav-3510）一致，差别仅在接口族：官方使用 `AscendC::MicroAPI`，本样例使用任务书指定的 C API。Golden 不复用该近似，按原始分段公式在 float64 下独立计算。

## 计算流程

```text
输入 x（GM）
  └─ Host：按元素数与 dtype 计算 blockCount / blockLength，多核下发
      └─ 每个 Vector-Core 处理一段连续区间
          └─ tile 循环（TILE_LENGTH = 2048）
              ├─ asc_copy_gm2ub_align：按当前 tile 的实际字节数搬入 UB
              ├─ MTE2 → V 同步
              ├─ repeat 循环
              │   ├─ asc_update_mask_b32：按剩余元素数消耗式更新长度掩码
              │   ├─ asc_loadalign（FLOAT16 为 asc_loadalign_unpack + asc_half2float）
              │   ├─ z = x * inputScale
              │   ├─ poly = Horner(z)，normal = exp(z) - 1
              │   ├─ expm1 = select(|z| < 0.1, poly, normal)
              │   ├─ negative = expm1 * alpha
              │   ├─ result = select(x > 0, x, negative)
              │   ├─ result = result * scale
              │   └─ asc_storealign（FLOAT16 为 asc_float2half_rn + asc_storealign_pack）
              ├─ V → MTE3 同步
              └─ asc_copy_ub2gm_align：按当前 tile 的实际字节数写回 GM
输出 y（GM）
```

## 指令序列与 API 选型

FLOAT32 与 FLOAT16 使用相同的计算结构，区别只在首尾的寄存器格式转换和多项式阶数。

| #   | 计算                                    | C API                            | 说明                       |
| --- | ------------------------------------- | -------------------------------- | ------------------------ |
| 1   | `z = x * inputScale`                  | `asc_mul_scalar`                 | 指数输入缩放                   |
| 2   | `poly = Horner(z)`                    | `asc_mul_scalar` / `asc_add_scalar` / `asc_mul` | 近零多项式，float 五阶、half 四阶 |
| 3   | `normal = exp(z) - 1`                 | `asc_exp` + `asc_add_scalar`     | 常规路径                     |
| 4   | `smallMask = abs(z) < 0.1`            | `asc_abs` + `asc_lt_scalar`      | 近零判定                     |
| 5   | `expm1 = select(smallMask, poly, normal)` | `asc_select`                 | 选择 expm1 算法              |
| 6   | `negative = expm1 * alpha`            | `asc_mul_scalar`                 | 负分支候选，在分支选择之前            |
| 7   | `positiveMask = x > 0`                | `asc_gt_scalar`                  | 分段判定                     |
| 8   | `result = select(positiveMask, x, negative)` | `asc_select`              | 隔离未选中分支                  |
| 9   | `result = result * scale`             | `asc_mul_scalar`                 | 输出缩放，在分支选择之后             |

所有向量指令都受当前 repeat 的长度掩码约束，尾 repeat 中只有有效 lane 参与计算与写回。

## FLOAT16 的寄存器格式转换

FLOAT16 路径的中间计算全部在 `vector_float` 上完成，首尾各增加一次格式转换：

1. `asc_loadalign_unpack` 按 `UNPACK_B16` 把连续 half 展开到 32-bit 槽的低半部分；
2. `asc_half2float` 转为 `vector_float`，随后复用 FLOAT32 计算路径和 FLOAT 标量；
3. `asc_float2half_rn` 按 RINT（就近舍入、中间值取偶）转回 half；
4. 把结果寄存器和 UB 地址重解释为 32-bit，用 `asc_storealign_pack` 触发 `PACK_B32`，将每个 32-bit 槽的低 16 位压回连续排布后写出。

第 4 步使用 32-bit 重载而非 half 重载：half 重载对应 `PACK_B16`，会把 16-bit 槽继续压缩成 8-bit，与 float→half 得到的寄存器布局不匹配。因此 FLOAT16 路径的逻辑 repeat 宽度按 `vector_float` 的 64 个元素计算，掩码取 b32 位宽。

## 多核分块与尾块处理

UB 局部数组长度必须在编译期确定，输入长度在运行期传入，因此采用「Host 分核 + Device 固定上界 tile 循环」两级切分。

Host 侧分核参数：

| 参数                       | 取值            | 作用                     |
| ------------------------ | ------------- | ---------------------- |
| `MAX_BLOCKS`             | 56            | Ascend 950 的 AIV 数，限定单波 |
| `MIN_ELEMENTS_PER_BLOCK` | 512           | 每核最小工作量，避免分核过碎          |
| `FLOAT_SMALL_INPUT`      | 256           | FLOAT32 低于该长度时单核       |
| `HALF_SMALL_INPUT`       | 1024          | FLOAT16 低于该长度时单核       |
| `blockLength`            | 向上对齐到 32 字节   | 保证每个核的起点 32B 对齐         |

`blockCount` 由对齐后的 `blockLength` 重新取整得到，最后一个核取实际剩余长度。两种 dtype 共享同一组分核常量。

Device 侧 tile 循环：

- `TILE_LENGTH` 取 2048，输入和输出各分配一块 UB 缓冲，FLOAT32 占 16 KiB、FLOAT16 占 8 KiB；
- 每个 tile 再按 64 个元素划分 repeat，最后一个 repeat 由 `asc_update_mask_b32` 限定有效元素；
- GM↔UB 搬运长度按当前 tile 的实际字节数计算，不按固定上限搬运；
- tile 之间用 `asc_sync()` 收敛，确保上一块写回完成后再复用 UB。

该方案可处理 `shape=1`、非向量整倍数、非 32 字节对齐、超过 2048 元素以及每核多个 tile 的输入；空输入由 Host 直接返回，不下发 Kernel。

## GM↔UB 搬运接口

在 CANN 9.1.0-beta.3、`dav-3510` 环境下，GM↔UB 搬运采用完整参数版本：

```cpp
asc_copy_gm2ub_align(
    __ubuf__ float* dst,
    __gm__ float* src,
    uint16_t n_burst,
    uint32_t len_burst,
    uint8_t left_padding_num,
    uint8_t right_padding_num,
    bool enable_constant_pad,
    uint8_t l2_cache_mode,
    uint64_t src_gap,
    uint32_t dst_gap);

asc_copy_ub2gm_align(
    __gm__ float* dst,
    __ubuf__ float* src,
    uint16_t n_burst,
    uint32_t len_burst,
    uint8_t l2_cache_mode,
    uint64_t dst_gap,
    uint32_t src_gap);
```

## 接口设计

### Kernel 接口

```cpp
// FLOAT32
__vector__ __global__ void elu_custom(
    __gm__ float* x,
    __gm__ float* y,
    float alpha,
    float scale,
    float input_scale,
    uint32_t total_length,
    uint32_t block_length);

// FLOAT16
__vector__ __global__ void elu_custom_half(
    __gm__ half* x,
    __gm__ half* y,
    float alpha,
    float scale,
    float input_scale,
    uint32_t total_length,
    uint32_t block_length);
```

两个 Kernel 只做 dtype 特化，对外表达同一个算子原型。`total_length` 与 `block_length` 是 Host 计算后传入的调度参数，属于样例实现细节，不属于任务书原型；三个标量在两条路径上均为 `float`。

### Host 接口

```cpp
template <typename T>
std::vector<T> kernel_elu(
    std::vector<T>& x,
    float alpha,
    float scale,
    float input_scale);
```

Host 侧执行流程为：

长度校验 → 空输入直接返回 → 计算 `blockCount` / `blockLength` → `aclInit` → `aclrtMalloc` → H2D 拷贝 → 启核 → `aclrtSynchronizeDevice` → D2H 拷贝 → 释放资源。

Host 侧不参与逐元素 ELU 计算，Golden 与结果比较属于测试代码。

## 硬件与兼容性

| 项       | 说明                                                       |
| ------- | -------------------------------------------------------- |
| 硬件      | Ascend 950PR / Ascend 950DT（`dav-3510`）                  |
| CANN 版本 | 真机验证环境为 9.1.0-beta.3                                     |
| 编译模式    | CMake 支持 `npu`（默认）与 `sim` 开关，架构固定为 `dav-3510`；`sim` 尚未验证 |
| 数据格式    | ND，不涉及 NCHW、NZ 等格式转换                                     |
| 上游影响    | 仅新增样例目录与父级索引行，不修改公共 CMake、SDK 头文件或既有样例                   |
| 核类型     | 仅使用 Vector-Core，不涉及 Cube-Core、Matmul 或 Conv API          |

---

# 可维可测分析

## 精度标准

Golden 使用 NumPy float64 按原始分段公式计算，再以目标 dtype 落盘。

逐元素判定公式为：

$$
|actual-golden|\le atol+rtol\cdot|golden|
$$

| 数据类型    | rtol               | atol               | required_matched_ratio | max_abs_error_limit |
| ------- | ------------------ | ------------------ | ---------------------- | ------------------- |
| FLOAT32 | $2^{-10}$（9.77e-4） | $2^{-16}$（1.53e-5） | 0.99                   | 1e-2                |
| FLOAT16 | $2^{-9}$（1.95e-3）  | $2^{-9}$（1.95e-3）  | 0.99                   | 1e-1                |

用例需同时满足 matched_ratio 与 max_abs_error 两项。非有限值单独判定，不进入容差比较：Golden 为 NaN 时只接受 NaN，为 $\pm\infty$ 时只接受同号无穷，为有限值时实际输出必须有限。该规则避免 $|\infty-\infty|=NaN$ 造成漏判。

## 官方语义对齐验证

任务书要求功能逻辑与官方 `elu` 对齐。以 `cann/ops-nn` 的 `activation/elu/op_kernel/arch35/elu_dag.h` 为基准逐条对照：近零阈值 `0.1f`、float 五阶与 half 四阶系数、Horner 顺序、乘 `alpha` 在分支选择之前、乘 `scale` 在分支选择之后、不预乘、标量在两条路径均为 float、half 链路 `UNPACK_B16 → 升 float → RINT → PACK_B32`、half 路径掩码取 b32 宽度、尾部掩码消耗式更新。

## 精度测试设计

| 维度    | 取值                                                                                     |
| ----- | -------------------------------------------------------------------------------------- |
| dtype | `float32`、`float16`                                                                    |
| shape | `0`、`1`、`31`、`32`、`33`、`255`、`1023`、`1024`、`2048`、`4096`、`5000`、`131072`、`1000001` |
| 数据范围  | $[-100,100]$，固定 seed 生成                                                                |
| 特殊点   | $0$、$-0.0$、$\pm10^{-4}$、$\pm0.1$、$\pm1$、$\pm20$、$\pm100$，植入序列前缀                    |
| 系数组合  | SELU 风格、三值互异、近零 `expm1`、快速饱和、负 `alpha`、负 `scale`、负 `inputScale`、零 `inputScale`   |

其中 `131072` 与 `1000001` 分别使每个核执行 2 个和 9 个 tile，用于覆盖 tile 循环的第二次及后续迭代、跨迭代 UB 复用和循环末尾同步。

用例统计口径为 54 个：

- 2 种 dtype × 13 个 shape × 默认系数，共 26 个；
- 2 种 dtype × 8 组公共系数，共 16 个；
- 2 种 dtype × 2 个有限值反例（FLOAT 标量窄化、Host 预乘溢出污染正分支），共 4 个；
- 2 种 dtype × 2 个非有限语义用例（NaN 与同号无穷的匹配规则），共 4 个；
- 2 种 dtype × 2 个非有限输入用例（输入 NaN、输入 $\pm\infty$ 的分支传播），共 4 个。

## 性能数据采集

性能数据采集范围如下：

- dtype：`float32`、`float16`；
- shape：`32 / 256 / 1024 / 2048 / 4096 / 16384 / 65536`，共 14 个用例，覆盖单核与多核两个区间；
- 系数：固定为 $(1,1,1)$；
- 工具：`msprof --application --aic-metrics=PipeUtilization`；
- 指标：`Task Duration(us)`、`aiv_time(us)`、`aiv_total_cycles`、`Task Type`、`Block Num` 和各流水占比。

纯 Vector-Core 的判定条件为 `Task Type` 为 `AI_VECTOR_CORE`，且 Cube/AIC 相关时间、周期和利用率均为 0。

## 可复现性

输入和 Golden 均由脚本使用固定 seed 生成并落盘。编译、数据生成、运行、结果校验和性能采集命令统一记录在交付目录的 `README.md` 和自测报告中。

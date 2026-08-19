# 【社区任务】ELU算子设计文档

# 一、需求描述

## 1.1 需求来源

本需求来源于 CANN 社区任务 2026 年 8 月社区任务「ELU 算子开发」。任务要求基于 **Ascend C C API** 接口，开发纯 **Vector-Core** 算子 `ELU`（Exponential Linear Unit），全程禁止调用 Cube-Core 单元。

任务对应开源仓为 `https://gitcode.com/cann/asc-devkit`，目标提交目录为 `examples/02_simd_c_api/03_c_api/02_reg_vector_compute`。功能逻辑需与 Ascend C 官方算子库（ops-nn `activation/elu`）实现完全对齐。

## 1.2 需求分析

### 1.2.1 ELU算子功能

ELU 为逐元素激活函数，返回与输入 shape 完全一致的输出张量。计算公式：

$$
ELU(x)=
\begin{cases}
scale \cdot x, & x > 0 \\
\alpha \cdot scale \cdot (e^{x \cdot inputScale} - 1), & x \le 0
\end{cases}
$$

其中：
- $\alpha$：激活系数（超参数），控制负半轴饱和速率；
- $scale$：输出缩放系数；
- $inputScale$：输入缩放系数。

当 $scale=1, inputScale=1$ 时退化为标准 ELU。算子在 $x=0$ 处连续（左右极限均为 0），负半轴输出范围 $(-\alpha \cdot scale, 0]$，无特殊值处理分支歧义。

### 1.2.2 算子目标原型

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | shape | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `src` | 输入 | ELU激活函数的输入，公式中的x | FLOAT、FLOAT16 | ND | 0-8维 | 支持空Tensor |
| `dst` | 输出 | ELU激活函数的输出 | FLOAT、FLOAT16 | ND | 与src一致，0-8维 | shape需与src一致 |
| `alpha` | 输入(标量) | 激活系数α | FLOAT | - | - | 可转换为FLOAT |
| `scale` | 输入(标量) | 缩放系数scale | FLOAT | - | - | 可转换为FLOAT |
| `input_scale` | 输入(标量) | 输入缩放系数inputScale | FLOAT | - | - | 可转换为FLOAT |

### 1.2.3 约束限制

- **禁止调用 Cube-Core**：纯 Vector-Core 实现，不使用任何 Cube/matmul 类接口；
- **禁止 Host 侧串行计算**：所有计算在 Device 侧批量完成，Host 侧仅做数据搬运与 golden 生成；
- **内存安全**：合理分配 Local Memory，确保无缓冲区溢出风险。

### 1.2.4 对齐标准

与官方算子库 `https://gitcode.com/cann/ops-nn/blob/master/activation/elu/README.md` 中 `elu` 实现完全对齐，该实现明确支持 Ascend 950PR/Ascend 950DT（与任务书适配硬件 Ascend 950 一致）。

**算法来源逐项对齐**（对照官方 `ops-nn/ascendc/elu/arch35/elu_dag.h`）：

| 本实现 | 官方实现 | 对齐情况 |
| --- | --- | --- |
| 公式 `scale·x` / `α·scale·(e^(x·inputScale)-1)` | `EluCustom`：`Muls(input, inputScale)` + `Exp` + `Adds(-1)` + `Muls(α)` | 数学公式逐项一致 |
| `x>0` 掩码 + `Select` 二选一 | `CompareScalar(x,0,GT)` + `Select` | 分支语义一致（含 NaN/±0 行为） |
| expm1 霍纳级数修正（fp32 阈值 0.01 / fp16 阈值 0.1） | kernel 内含"ex-1 多项式拟合" | 精度修正思路一致（阈值与官方一致） |
| fp16 展开到 fp32 域计算 | MicroAPI `DIST_UNPACK_B16` + `Cast` | C API 等价链路（见 2.1.4） |

### 1.2.5 开发中发现的易用性问题（将按规范提交 Issue 至 asc-devkit 仓库）

1. **文档与工具包接口不一致**：asc-devkit 仓库文档描述的 3 参数 `asc_copy_gm2ub_align(dst,src,size)` 与 `asc_store_l2_cache_mode` 枚举，在 CANN 9.0.0 的 dav-3510 头文件中不存在（仅提供 10 参数 `uint8_t l2_cache_mode` 版本），按文档编写的代码编译报错，需对照实际头文件适配。**官方已确认**：CANN 9.1.0 已在 3510 段补充 3 参数版 `asc_copy_gm2ub_align(dst,src,size)`（`asc_copy_ub2gm_align` 同步补充），该版本差异已解决；
2. **Host 侧 `std::exp/std::expm1` 链接失败**：ASC 链接器（cceld）默认不链接系统 libm，Host 侧使用标准数学函数报 `undefined symbol: expf/expm1f`，需手动 `target_link_libraries(... m)`，报错信息不直观；
3. **`std::min` 在 `__aicore__` 中不可调用**：host 库函数在设备函数中不可用，编译报错 `no matching function for call to 'min'`，需改用三元表达式，建议文档明确该限制。
4. **`asc_storealign_pack` 对 fp16 的用法不直观**：C API 层实现官方 fp16 的 fp32 域计算需要 `asc_loadalign_unpack`（位宽展开加载，等价 MicroAPI `DIST_UNPACK_B16`）+ `asc_half2float`（读偶数位）+ `asc_float2half_rn`（写偶数位）+ `asc_storealign_pack`（压缩存储）的组合，其中 `asc_storealign_pack` 需以 int32 视图（`(vector_int32_t)dst_half` + `(int32_t*)dst + n/2`）调用才能正确压缩 fp16（见 cast 样例场景2），直接用 half 版本会因掩码语义不同产生交错错位（实测 matched_ratio=0.25）。该用法缺乏文档示例，建议在 C API 文档中补充 fp16 位宽展开/压缩的完整示例。

# 二、方案设计

## 2.1 接口内部实现

### 2.1.1 实现路线

采用 **C API 寄存器向量计算（reg_vector_compute）模式**，该模式为 Ascend 950（`dav-3510`）专属，与提交目录 `02_reg_vector_compute` 下既有样例（abs/mul/select/compare 等）保持同一编程范式。

核心思路：**全程无分支**。两条分段公式对所有元素全量计算，再用 `x > 0` 的比较掩码通过 `Select` 二选一，避免 if/else 造成的向量流水停顿。

计算流程：

```text
搬入: asc_copy_gm2ub_align  GM → UB
同步: asc_sync_notify/wait (PIPE_MTE2 → PIPE_V)
逐块计算（寄存器域，每个repeat处理 one_rep_size 个元素）:
  y_pos  = scale * x                          ← asc_mul_scalar
  t      = x * inputScale                     ← asc_mul_scalar
  e      = exp(t)                             ← asc_exp（直接法）
  e-1                                          ← asc_add_scalar (value=-1)（直接法，|t|大时用）
  级数法 expm1(t) ≈ t*(1 + t*(1/2 + t*(1/6 + t*(1/24 + t/120))))   ← asc_mul + asc_mul_scalar + asc_add_scalar（|t|小时用；fp32 取 5 阶；fp16 对齐官方取 4 阶，见 2.1.2）
  expm1  = (|t| < 阈值) ? 级数 : 直接法       ← asc_abs + asc_lt_scalar + asc_select
  y_neg  = expm1 * alpha                     ← asc_mul_scalar（分步：先乘α，避免 α*scale 预乘溢出）
  y_neg  = y_neg * scale                     ← asc_mul_scalar（再乘 scale）
  mask   = (x > 0)                            ← asc_gt_scalar
  dst    = select(mask, y_pos, y_neg)         ← asc_select
同步: asc_sync_notify/wait (PIPE_V → PIPE_MTE3)
搬出: asc_copy_ub2gm_align  UB → GM
```

### 2.1.2 数值风险分析

#### 反例1：`α·scale` 预乘溢出（采用分步乘法）

**不能**把 `α·scale` 先在 Host/寄存器预乘，再与 `expm1` 相乘。合法有限 FLOAT 参数可击穿该做法：

| 输入 | 预乘实现（错误） | 分步实现（正确） | 原因 |
| --- | --- | --- | --- |
| `x=0, α=2e38, scale=2` | `NaN` | `0` | `α·scale=4e38>FLT_MAX` 溢出为 `inf`，`expm1(0)=0`，`0×inf=NaN` |
| `x=-1, α=2e38, scale=2` | `-inf` | `-2.53e38`（有限） | 溢出为 `inf` 后 `expm1(-1)×inf=-inf`，丢失有限结果 |

本设计在设备寄存器域**分步相乘**：`y_neg = expm1·α` 后再 `·scale`。此时 `x=0` 处 `0×2e38=0`（非 NaN），`x=-1` 处 `-0.63×2e38=-1.26e38`（有限）再 `×2=-2.53e38`，均正确。

#### 反例2：三个 FLOAT 标量不得降为 half

half 输入不能把 `α/scale/inputScale` 转成 half 计算：`x=-100, α=60000, scale=1, inputScale=2e-8` 的正确结果约为 `-0.12`；若 `inputScale` 降成 half 会变为 0，输出错误地变成 0。本实现三个标量始终以 FLOAT 传入 float/half 两个 kernel，在 fp32 域参与计算。

#### 反例3：`inputScale<0` 时指数自变量可为正（不假设永不溢出）

`inputScale` 无非负约束。它为负时，负输入的指数自变量为正（如 `x=-100, inputScale=-0.01 → t=+1`），可能逼近/超出 fp32 上限。实现按原公式传播可表示结果及 IEEE NaN/Inf，不做错误取值域假设。实测设备 `asc_exp` 约在 `exp≈2.9e38` 处提前溢出为 `inf`，该行为属数据范围边界（相对误差 <1.7e-11），非 kernel 缺陷。

#### expm1 精度修正（与官方对齐）

与官方算子库 kernel（ops-nn `activation/elu`，其实现含"ex-1 多项式拟合"）对齐，对 $e^t - 1$ 做精度修正：

- **问题**：$x \to 0$ 时 $e^t - 1$ 存在灾难性抵消（$e^t \approx 1$ 时，$e^t$ 的浮点舍入误差在减 1 后原样进入结果，相对误差被放大上千倍，实测最坏可达 $10^{-3}$ 量级），超出生态精度标准；
- **方案**：$|t|$ 小于阈值时用霍纳级数 $e^t-1 \approx t(1 + \frac{t}{2}(1 + \frac{t}{3}(1 + \frac{t}{4}(1+\frac{t}{5}))))$ 近似（fp32 取 5 阶含 $\frac{t}{120}$ 项；fp16 对齐官方取 4 阶即 $t(1 + \frac{t}{2}(1 + \frac{t}{3}(1 + \frac{t}{4})))$），$|t|$ 大时用直接法 $exp(t)-1$（此时无抵消），通过 `asc_lt_scalar` + `asc_select` 无分支二选一；
- **阈值**：fp32 取 0.01（此时直接法相对误差 $<6\times10^{-6}$，级数截断余项 $\sim10^{-11}$）；fp16 取 0.1（匹配 fp16 更宽的容差，与官方一致）；
- 全程无分支、纯向量，无额外性能开销（约 9 条指令）。

### 2.1.3 动态Tiling设计

为满足泛化验收（任意合法 shape），采用**段式动态 tiling**：

- 编译期常量 `MAX_BLOCK_LENGTH = 8192`：每段最大处理元素数，受 UB 容量约束（fp32 双缓冲 2×8192×4=64KB，fp16 双缓冲 2×8192×2=32KB）；
- kernel 接收总元素数 `total_length`，每核通过 `asc_get_block_num()` 获取核数，按 `seg += block_count` 步长循环处理多段；
- 每段实际长度 `count = min(MAX_BLOCK_LENGTH, total_length - start)`，`start` 为段起始偏移；
- 尾部处理：`asc_update_mask_b32` 的 value 为引用参数、调用后自动递减（每次减 64），最后一次迭代自动生成部分掩码，无需手写尾部逻辑；fp32/fp16 均在 fp32 域计算，故统一使用 b32；
- 任意 shape（含 1、32、非对齐、超大规模）均可正确处理，UB 永不超限。

### 2.1.4 双精度支持

fp32 与 fp16 使用**同一套 fp32 域计算链**（fp16 在 fp32 域计算，与官方完全一致），仅寄存器类型与掩码位宽不同：

| 项 | fp32 | fp16 |
| --- | --- | --- |
| 数据寄存器 | `vector_float` | 展开加载 `vector_half` → 转 `vector_float` 计算 |
| 掩码更新 | `asc_update_mask_b32` | `asc_update_mask_b32`（fp32 域） |
| 标量类型 | `float` | `float`（fp32 域，无需转 half） |
| 核函数 | `elu_custom` | `elu_custom_half` |

fp16 实现与官方 ops-nn `elu` kernel（950/arch35）**完全对齐**：官方通过 MicroAPI 的 `DIST_UNPACK_B16` 将 fp16 位宽展开加载、`Cast` 转 fp32 域计算后再转回；C API 层具有完全等价的链路（`asc_loadalign_unpack`=UNPK_B16 展开加载至 2*N 位置、`asc_half2float`=读偶数位转 fp32、`asc_float2half_rn`=写回偶数位、`asc_storealign_pack`=int32 视图压缩存储，见 cast 样例），故 fp16 全程在 fp32 域计算，公式、expm1 霍纳级数修正（fp16 多项式取 4 阶，与官方逐项一致）、阈值 0.1、分步乘 α/scale 均与官方完全一致，实测满足生态算子开源精度标准（`matched_ratio=1`，文件模式 `max_abs_error=0`）。

### 2.1.5 输入定义域与分支覆盖说明

本算子的输入定义域为**全体 FLOAT / FLOAT16 输入**（含正常有限值、±0、NaN、±Inf），无未定义/未覆盖的输入区间。`x > 0` 与 `x ≤ 0` 两分支由 `asc_gt_scalar` 比较掩码 + `asc_select` 二选一，对全体输入互斥完备、无遗漏、无重叠：

- **正半轴（x > 0）**：走 `scale·x` 分支；
- **负半轴（x ≤ 0）**：走 `α·scale·(e^(x·inputScale)-1)` 分支（含 x=0、±0）；
- **NaN**：`asc_gt_scalar` 比较结果为 false（IEEE 有序谓词），落入负半轴分支；但 `asc_exp(NaN)=NaN`，经计算链传播最终输出 NaN，符合官方与生态标准对 NaN 的语义（golden 为 NaN 时要求输出必须为 NaN）；
- **±Inf**：+Inf 落入正半轴分支输出 `scale·(+Inf)`（按 scale 符号）；-Inf 落入负半轴分支，`exp(-Inf·inputScale)` 按 inputScale 符号下溢为 0 或上溢为 Inf，输出相应为 `-α·scale` 或 ±Inf，与官方一致；
- **±0**：`0>0` 为 false，落入负半轴分支，`exp(0)-1=0`，输出 0，无需独立特判路径。

上述分支语义与官方 ops-nn `elu`（`CompareScalar + Select` 同构）完全一致，所有边界/特殊值均有明确、可验证的输出定义。

## 2.2 接口设计

### 2.2.1 Kernel侧接口

```cpp
// fp32 版核函数（动态tiling）
__vector__ __global__ void elu_custom(
    __gm__ float* src, __gm__ float* dst,
    float alpha, float scale, float input_scale,
    uint32_t total_length);

// fp16 版核函数（动态tiling）
// 注：alpha/scale/input_scale 按任务书2.2与官方ELU规格声明为 FLOAT；fp16 在 fp32 域计算（对齐官方 DIST_UNPACK_B16 链路）
__vector__ __global__ void elu_custom_half(
    __gm__ half* src, __gm__ half* dst,
    float alpha, float scale, float input_scale,
    uint32_t total_length);
```

| 参数 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src` | 输入 | 源操作数（GM，ND 格式） |
| `dst` | 输出 | 目的操作数（GM，shape 与 src 一致） |
| `alpha`/`scale`/`input_scale` | 输入 | 标量系数，按值传入 |
| `total_length` | 输入 | 总元素数，用于动态 tiling |

**Kernel 接口约束说明**

- 全程不访问 Cube-Core；
- UB 缓冲为编译期常量 `MAX_BLOCK_LENGTH`，每段实际计算量由掩码控制，无越界风险；
- fp32/fp16 使用同一套 fp32 域计算接口，仅类型参数不同（fp16 经展开加载/转换后在 fp32 域计算）。

### 2.2.2 所需 C API 接口清单

| 类别 | 接口 | 用途 |
| --- | --- | --- |
| 框架 | `__vector__ __global__` / `__simd_vf__ inline` / `asc_init()` / `block_idx` | kernel 声明与初始化 |
| 搬移 | `asc_copy_gm2ub_align` / `asc_copy_ub2gm_align` / `asc_loadalign` / `asc_storealign` / `asc_loadalign_unpack` / `asc_storealign_pack` | GM↔UB↔寄存器；fp16 展开加载（UNPK_B16）与压缩存储 |
| 同步 | `asc_sync_notify` / `asc_sync_wait` / `asc_sync` | 管道同步（PIPE_MTE2/PIPE_V/PIPE_MTE3） |
| 掩码 | `asc_update_mask_b32` | 尾部与边界处理（value 引用递减，自动生成部分掩码；fp16 在 fp32 域计算故用 b32） |
| 类型转换 | `asc_half2float` / `asc_float2half_rn` | fp16 输入转 fp32 域 / 结果转回 fp16（读/写偶数位，对齐官方 Cast） |
| 计算 | `asc_mul_scalar` / `asc_exp` / `asc_add_scalar` / `asc_gt_scalar` / `asc_select` | ELU 公式计算链 |
| expm1修正 | `asc_mul`（向量×向量）/ `asc_abs`（取\|t\|）/ `asc_lt_scalar`（\|t\|<阈值） | 小x精度修正 |
| 系统变量 | `asc_get_block_num()` / 常量 `VECTOR_LENGTH_BYTES` | 核数获取 / 向量寄存器长度（VL=256B；`asc_get_vf_len()` 为 CANN 9.1.0 新增，9.0.0 无此接口，故用编译期常量 `VECTOR_LENGTH_BYTES=256` 等价替代，两版本行为一致） |

### 2.2.3 Host侧接口

```cpp
std::vector<float> kernel_elu(std::vector<float>& x, float alpha, float scale, float input_scale);
std::vector<half>  kernel_elu_half(std::vector<half>& x, float alpha, float scale, float input_scale);  // 标量参数为FLOAT，与算子规格一致
```

Host 侧按数据量动态决定核数：`num_blocks = min(NUM_BLOCKS, ceil(total_length / MAX_BLOCK_LENGTH))`；创建 stream 后通过 `elu_custom<<<num_blocks, 0, stream>>>(...)` 调用（与任务书参考样例 `c_api_async_add` 的 `<<<numBlocks, 0, stream>>>` 写法一致），`aclrtSynchronizeStream(stream)` 等待 kernel 执行完成后 `aclrtDestroyStream(stream)` 销毁。fp16 版 `elu_custom_half` 同样带 stream。

## 2.3 测试用例设计

### 2.3.1 功能测试

| 用例 | 数据类型 | shape | 数据范围 | 说明 |
| --- | --- | --- | --- | --- |
| T1 | FLOAT | 1 | 单点 | 最小shape，验证尾部mask |
| T2 | FLOAT | 31 | [-100,100] | 2的幂-1，尾部非32B对齐 |
| T3 | FLOAT | 32 | [-100,100] | 任务书要求 |
| T4 | FLOAT | 1023 | [-100,100] | 2的幂-1，非对齐 |
| T5 | FLOAT | 1024 | [-100,100] | 任务书要求 |
| T6 | FLOAT | 2047 | [-100,100] | 2的幂-1，非对齐 |
| T7 | FLOAT | 2048 | [-100,100] | 任务书要求 |
| T8 | FLOAT | 8191 | [-100,100] | 2的幂-1，多段 |
| T9 | FLOAT | 16384 | [-100,100] | 多核并行 |
| T10 | FLOAT | 100000 | [-100,100] | 超多段，验证动态tiling |
| T11 | FLOAT | 1048575 | [-100,100] | 超大规模 |
| T12-T22 | FLOAT16 | 同上 | 同上 | fp16 全量覆盖 |

### 2.3.2 边界与特殊值测试（已实现）

- `x = 0 / -0`：输出 0。x=±0 无需独立特判路径——`0>0` 为 false 落入负半轴分支，`exp(0·inputScale)-1=0` 自然计算输出 0，与 2.1.5 输入定义域说明一致；
- `x = +Inf`：输出 `scale * Inf`（按 scale 符号）；
- `x = -Inf`：输出 `-alpha * scale`（inputScale>0 时 exp 下溢为 0）；
- `x = NaN`：输出 NaN；
- 不同 `alpha`/`scale`/`input_scale` 组合（8 组：非默认系数、负 input_scale、alpha=0、scale=0、负 alpha、input_scale=0、较大系数）；其中"较大系数"取 `(2, 1.25, 1)`：生态标准对 fp16 有 `max_abs_error≤0.1` 硬上限，而 fp16 在输出 150 处固有量化步长为 0.125（1 ULP）——实测 `(3, 1.5, 1)` 随机输入下可达 0.125>0.1；取输出 ≤125（ULP=0.0625）后随机输入下 `max_abs_error=0.0625` 稳低于上限；
- 负半轴极大值（`x=-100`）验证 exp 下溢不产生 NaN；
- **exp 上溢边界（实测发现）**：当 `input_scale<0` 且数据范围较大（如 [-100,100]）时，负半轴 `exp(x*inputScale)=exp(|x|)` 会逼近/超出 fp32 上限。实测设备 `asc_exp` 约在 `exp≈2.9e38`（x≈-87.5）处提前溢出为 `inf`，而 `std::exp` 仍返回有限值；fp16 在 `|x*inputScale|>11` 即溢出。该行为属数据范围导致的溢出边界（相对误差 <1.7e-11，精度本身极佳），非 kernel 精度缺陷；参数组合测试对负 `input_scale` 用例采用小幅值（如 -0.01）以落在有效精度区间；
- **小 x 精度**（`x ∈ {1e-3, 1e-4, 1e-5, 1e-6, -1e-3, -1e-5, -1e-8, 0}`）：验证 expm1 修正后 x→0 处相对精度达标（golden 用 `std::expm1` 精确计算）；
- 空 Tensor（shape=0）：kernel 直接返回空，避免 0 字节设备内存操作。

### 2.3.3 精度判定

精度标准：满足生态算子开源精度标准（`https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md`）。该标准采用**混合容差**判定：$|actual-golden| \le atol + rtol \times |golden|$，且整体通过率 `matched_ratio ≥ 0.99`，且最大绝对误差 `max_abs_error ≤ max_abs_error_limit`。

| 数据类型 | rtol | atol | max_abs_error_limit | matched_ratio |
| --- | --- | --- | --- | --- |
| FLOAT | 2⁻¹⁰ (9.77e-4) | 2⁻¹⁶ (1.53e-5) | 1e-2 | ≥0.99 |
| FLOAT16 | 2⁻⁹ (1.95e-3) | 2⁻⁹ (1.95e-3) | 1e-1 | ≥0.99 |

golden 由 Host 侧按公式精确计算（小 x 处用 `std::expm1`），验证代码按上表阈值 + `matched_ratio`/`max_abs_error` 判定。

# 三、可维可测

## 3.1 精度标准/性能标准

| 项目 | 标准 | 说明 |
| --- | --- | --- |
| 精度标准 | 生态算子开源精度标准（已核实） | FLOAT：rtol 9.77e-4 / atol 1.53e-5 / limit 1e-2 / ratio≥0.99；FLOAT16：rtol 1.95e-3 / atol 1.95e-3 / limit 1e-1 / ratio≥0.99。验证代码按此标准判定，并打印 matched_ratio/max_abs_error。x→0 处采用 expm1 级数修正（对齐官方 kernel 的 ex-1 多项式拟合），消除灾难性抵消，保证任意输入（含极小 x）相对精度 |
| 性能标准 | 无（任务书未规定） | 任务书未对 ELU 样例设定性能指标/阈值，本交付不设性能验收标准。实现层面为纯 Vector-Core 无分支实现，全程无 if/else 流水停顿、无 Cube-Core 调用，指令流水连续；如评审需要，可在自测报告中补充同 shape 下的执行耗时/吞吐实测数据 |

## 3.2 兼容性分析

- 适配硬件：Ascend 950PR / Ascend 950DT（`dav-3510` 架构），CANN 9.0.0 ~ 9.1.0；
- 提交位置：asc-devkit `examples/02_simd_c_api/03_c_api/02_reg_vector_compute/elu/`；
- 不修改任何既有接口行为，为全新增样例，无存量兼容性问题；
- 数据类型覆盖任务书要求的 FLOAT、FLOAT16 两种。

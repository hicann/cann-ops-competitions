------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "WISE小队"

team_members:

- "成员1：黄丽丽-天津大学"
- "成员2：肖子博-天津大学"
- "成员3：韩坤书-天津大学"

operator_name: "Add"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# 算子测试报告

------

## 一、算子理解

### 1.1 算子功能

本次测试对象为 CANN `Add` 算子族，覆盖接口包括 `aclnnAdd`、`aclnnAdds`、`aclnnAddV3` 以及对应 inplace 接口。核心语义为：

```text
out_i = self_i + alpha * other_i
```

其中 `alpha` 为标量缩放因子。`aclnnAdd` 是 Tensor-Tensor 形态，`aclnnAdds` 是 Tensor-Scalar 形态，`aclnnAddV3` 是 Scalar-Tensor 形态；原地接口将结果写回输入 Tensor。

### 1.2 输入输出规格

| 接口 | self | other | 输出 | 计算语义 |
| --- | --- | --- | --- | --- |
| `aclnnAdd` | Tensor | Tensor | `out` Tensor | `out = self + alpha * other` |
| `aclnnAdds` | Tensor | Scalar | `out` Tensor | `out = self + alpha * other` |
| `aclnnAddV3` | Scalar | Tensor | `out` Tensor | `out = self + alpha * other` |
| `aclnnInplaceAdd` | Tensor | Tensor | 写回 `selfRef` | `selfRef = selfRef + alpha * other` |
| `aclnnInplaceAdds` | Tensor | Scalar | 写回 `selfRef` | `selfRef = selfRef + alpha * other` |
| `aclnnInplaceAddV3` | Scalar | Tensor | 写回 `otherRef` | `otherRef = self + alpha * otherRef` |

主要约束包括：

- Tensor-Tensor 输入需要满足 broadcast 规则；
- 输出 shape 需要等于 broadcast 后 shape；
- 原地接口要求 broadcast 后 shape 与写回目标 shape 一致；
- Tensor rank 不能超过 8；
- `self`、`other`、`alpha`、`out` 不能为空；
- `alpha` 需要能转换到推导后的计算 dtype；
- 计算 dtype 需要能转换到输出 dtype。

### 1.3 dtype 与边界行为

测试覆盖 dtype 包括 `FLOAT`、`FLOAT16`、`BF16`、`DOUBLE`、`INT8`、`UINT8`、`INT16`、`INT32`、`INT64`、`BOOL`、`COMPLEX64`。

**实现中 `alpha` 会影响路径选择：`alpha == 1` 倾向进入 direct `l0op::Add` 路径，`alpha != 1` 可能进入 `Axpy` / `AxpyV2` 路径，特殊 dtype 或组合可能进入 `Mul + Add` 或 AICPU fallback。测试过程中发现，部分 `alpha == 1` 且叠加 broadcast、scalar broadcast 或 inplace alias 的成功执行型 case 曾卡在 TDT / `aclrtSynchronizeStream`，因此相关 case 调整为 `alpha = 2` 或 workspace-only 探针。**

------

## 二、测试策略与用例设计

### 2.1 测试方法

当前测试包含 106 个测试用例。在保证可运行性的前提下覆盖成功路径、失败路径、dtype 分支、broadcast 分支、inplace 分支、empty tensor 分支、精度边界和 AICPU fallback。

失败项如下：

| case | 结果 | 日志现象 | 归因 |
| --- | --- | --- | --- |
| `CASE_Add_Double_Alpha1_DirectAdd_AiCpuFallback` | FAIL | `aclnnAddGetWorkspaceSize failed. ERROR: 561103` | `DOUBLE` 被路由到 AICPU fallback，当前环境不支持对应 AICPU 执行依赖 |
| `CASE_Add_Double_FinalMulAdd_AiCpuFallback` | FAIL | `aclnnAddGetWorkspaceSize failed. ERROR: 561103` | 同上，保留该公开 DOUBLE 用例用于暴露后端环境问题 |
| `CASE_Adds_Float32_NonND_FormatWarning` | FAIL | `aclnnAddsGetWorkspaceSize non-ND failed. ERROR: 561103` | 非 ND format 路径在当前环境下 workspace 查询失败 |

### 2.2 用例分类

| 类别 | 代表 case | 目的 |
| --- | --- | --- |
| 基础 Tensor-Tensor | `CASE_Add_Float32_Alpha1_DirectAdd` | 覆盖 direct add 基础成功路径 |
| alpha 分支 | `Alpha0`、`Alpha2`、负数小数 alpha | 覆盖不同 dispatch 和缩放语义 |
| mixed dtype | FP16+FP32、BF16+FP32、INT16+INT32 | 覆盖 dtype promote 和 cast |
| 整数/BOOL | INT8、UINT8、INT16、INT32、INT64、BOOL | 覆盖离散类型精确匹配 |
| broadcast | 3D broadcast、invalid broadcast、invalid out shape | 覆盖 shape 推导与失败检查 |
| inplace | InplaceAdd、InplaceAdds、InplaceAddV3 | 覆盖写回目标和 shape 约束 |
| empty tensor | `*_EmptyTensor_*` | 覆盖 zero workspace 与 no-kernel phase2 |
| L0 探针 | `CASE_L0AddInplace_*` | 覆盖 `add.cpp` 中 `Add` / `AddInplace` 可触达分支 |
| 精度观察 | FP32/FP16/BF16 precision observation | 观察真实数学误差和 dtype 量化损失 |
| 参数失败 | null、rank>8、unsupported dtype、cast fail | 覆盖参数校验失败分支 |

### 2.3 精度测试设计与容差

普通验收使用 dtype 预期：

```text
lhs = decode_by_self_dtype(self)
rhs = decode_by_other_dtype(other)
alpha_value = decode_by_alpha_dtype(alpha)
ideal = lhs + alpha_value * rhs
dtype_expected = cast_to_output_dtype(ideal)
actual_as_double = decode_by_output_dtype(device_output)
```

精度观察额外记录：

```text
ideal_abs_error = abs(actual_as_double - ideal)
ideal_rel_error = ideal_abs_error / abs(ideal)
dtype_abs_error = abs(actual_as_double - dtype_expected)
```

FP16 / BF16 在 host 侧以 `uint16_t` bit pattern 保存，比较时必须先解码为对应浮点值，再转 double。

| 数据类型 | atol | rtol | 说明 |
| --- | ---: | ---: | --- |
| Float32 | `1e-6` | `1e-6` | 严于 CANN 常见 `1e-4` 标准 |
| Float16 | `1e-4` | `1e-4` | 参考标准；half ULP 边界观察 case 使用更宽容差避免误判 |
| BFloat16 | `1e-2` | `1e-2` | BF16 尾数位更少，需要放宽 |
| Int32 | `0` | `0` | 精确匹配 |

------

## 三、覆盖率分析

### 3.1 测量结果

覆盖率来自真实环境运行后生成的 HTML 覆盖报告。重点评分文件结果如下：

| 文件 | 行覆盖 | 函数覆盖 | 分支覆盖 |
| --- | ---: | ---: | ---: |
| `op_api/aclnn_add.cpp` | 248 / 303 = 81.8% | 25 / 26 = 96.2% | 508 / 1594 = 31.9% |
| `op_api/add.cpp` | 50 / 59 = 84.7% | 8 / 8 = 100.0% | 123 / 264 = 46.6% |
| `op_api/aclnn_add_v3.cpp` | 73 / 77 = 94.8% | 9 / 9 = 100.0% | 164 / 446 = 36.8% |
| `op_host/arch35/add_tiling_arch35.cpp` | 82 / 90 = 91.1% | 12 / 12 = 100.0% | 63 / 158 = 39.9% |

四个重点文件按行数加权的综合行覆盖率为：

```text
85.6%
```

### 3.2 覆盖与未覆盖分析

已覆盖 `aclnnAdd`、`aclnnAdds`、`aclnnAddV3` 的正常和失败路径，inplace wrapper，L0 `AddInplace` 可触达路径，alpha 分支，FP32/FP16/BF16/整数/BOOL/DOUBLE/COMPLEX64 探针，broadcast、empty tensor、null 参数、rank 超限、unsupported dtype 和 alpha cast fail。

未覆盖部分主要来自平台相关分支、`IsRegBase()` 相关运行环境分支、内部 tiling 错误分支，以及当前环境下无法继续执行的 DOUBLE AICPU fallback。**部分 `alpha == 1` direct-add broadcast / inplace 路径存在卡死风险，因此未作为大量成功执行 case。**

------

## 四、精度分析

### 4.1 精度测试

将精度测试拆成两层来判断：

1. **算子实现是否正确**：NPU 输出是否等于按照输出 dtype 量化后的 golden，即 `dtype_expected`。这一层决定测试是否 PASS。
2. **浮点数本身损失了多少信息**：真实数学结果 `ideal_double` 与最终输出 `actual_as_double` 之间的差距是多少。这一层用于分析 FP32、FP16、BF16 的精度边界。

几个字段含义如下：

```text
ideal_double      = 输入按实际 dtype 解码后，在 double 中计算出的理想数学结果
dtype_expected    = ideal_double 按输出 dtype 重新量化后的期望结果
actual_as_double  = NPU 输出按输出 dtype 解码后转成 double 的结果
ideal_abs_error   = |actual_as_double - ideal_double|
ideal_rel_error   = ideal_abs_error / |ideal_double|
dtype_loss        = |ideal_double - dtype_expected|
dtype_abs_error   = |actual_as_double - dtype_expected|
```

重点关注 `dtype_abs_error` 和 `dtype_loss` 的区别：

- `dtype_abs_error = 0` 表示 **NPU 结果与当前 dtype 语义下的预期完全一致**，算子实现没有额外引入误差；
- `dtype_loss != 0` 表示 **从 double 理想值落到目标 dtype 时发生了量化损失**，这是浮点格式本身的限制；
- 如果 `ideal_abs_error` 很大但 `dtype_abs_error = 0`，说明问题不是 Add 算子算错，而是这个数在目标 dtype 中本来就表示不了。

### 4.2 FP32：大数加小数时的小量吞没

对应 case：

```text
CASE_Add_Float32_PrecisionObservation_LargePlusSmall_Swallowed
```

这个用例构造的是 `1e10` 量级的大数与 `1e-5` 量级的小数相加/相减。我想验证的是：在 FP32 中，小数是否还能对大数产生可见影响。

运行结果为 PASS，四个元素的实测日志整理如下：

| index | ideal_double | dtype_expected | actual_as_double | ideal_abs_error | ideal_rel_error | dtype_loss | dtype_abs_error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `10000000000.00001` | `10000000000` | `10000000000` | `9.5367431640625e-06` | `9.5367431640624907e-16` | `9.5367431640625e-06` | `0` |
| 1 | `9999999999.9999905` | `10000000000` | `10000000000` | `9.5367431640625e-06` | `9.5367431640625085e-16` | `9.5367431640625e-06` | `0` |
| 2 | `-9999999999.9999905` | `-10000000000` | `-10000000000` | `9.5367431640625e-06` | `9.5367431640625085e-16` | `9.5367431640625e-06` | `0` |
| 3 | `-10000000000.00001` | `-10000000000` | `-10000000000` | `9.5367431640625e-06` | `9.5367431640624907e-16` | `9.5367431640625e-06` | `0` |

从结果看，`actual_as_double` 与 `dtype_expected` 完全一致，说明 Add 算子没有额外误差；但 `ideal_double` 中的 `±1e-5` 级别变化在 FP32 输出中被完全吞掉了。

**测试分析**：FP32 只有约 24 bit 有效二进制精度，换算成十进制大约 7 位有效数字。当数值已经达到 `1e10` 量级时，相邻可表示 FP32 数之间的间隔远大于 `1e-5`，因此这个小数对最终存储结果没有影响。这个用例的测试价值在于，它能直观看到 Add 在大数背景下并不会保留远小于当前 ULP 的增量。

### 4.3 FP32：接近相等数相加后的近零结果

对应 case：

```text
CASE_Add_Float32_PrecisionObservation_Cancellation_NearZeroRelGuard
```

这个用例构造了类似下面的输入：

```text
1.000000119 + (-1.0)
2.000000238 + (-2.0)
-3.000000238 + 3.0
-4.000000477 + 4.0
```

测试目标是观察正负抵消后，结果落到 `1e-7` 量级时是否还能被正确保留。实测结果如下：

| index | ideal_double | dtype_expected | actual_as_double | ideal_abs_error | ideal_rel_error | dtype_loss | dtype_abs_error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1.1920928955078125e-07` | `1.1920928955078125e-07` | `1.1920928955078125e-07` | `0` | `0` | `0` | `0` |
| 1 | `2.384185791015625e-07` | `2.384185791015625e-07` | `2.384185791015625e-07` | `0` | `0` | `0` | `0` |
| 2 | `-2.384185791015625e-07` | `-2.384185791015625e-07` | `-2.384185791015625e-07` | `0` | `0` | `0` | `0` |
| 3 | `-4.76837158203125e-07` | `-4.76837158203125e-07` | `-4.76837158203125e-07` | `0` | `0` | `0` | `0` |

这组结果没有出现误差，因为选择的差值本身正好对应 FP32 在该范围内可表示的 ULP 倍数，例如 `1.1920928955078125e-07` 就是典型的 FP32 epsilon 量级。因此这里没有产生额外舍入。

### 4.4 FP32：ULP 边界与 `2^24` 相邻整数不可表示

对应 case：

```text
CASE_Add_Float32_PrecisionObservation_SwallowAndUlpBoundary
```

这个用例把几类 FP32 典型边界放在一起观察：

- `1e10 + 1e-2`：大数吞小数；
- `1.0 ± 2^-24`：1.0 附近半 ULP 级别变化；
- `16777216 + 1`：`2^24` 附近相邻整数不可表示。

实测结果如下：

| index | ideal_double | dtype_expected | actual_as_double | ideal_abs_error | ideal_rel_error | dtype_loss | dtype_abs_error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `10000000000.01` | `10000000000` | `10000000000` | `0.010000228881835938` | `1.0000228881825937e-12` | `0.010000228881835938` | `0` |
| 1 | `1.0000000596046448` | `1` | `1` | `5.9604644775390625e-08` | `5.9604641222677158e-08` | `5.9604644775390625e-08` | `0` |
| 2 | `-1.0000000596046448` | `-1` | `-1` | `5.9604644775390625e-08` | `5.9604641222677158e-08` | `5.9604644775390625e-08` | `0` |
| 3 | `16777217` | `16777216` | `16777216` | `1` | `5.9604641222677158e-08` | `1` | `0` |

这里最有代表性的是 index 3。`16777216 = 2^24`，FP32 的有效位数决定了从这个位置开始，并不是所有整数都能被逐个表示，`16777217` 会被舍入回 `16777216`。日志里 `dtype_abs_error = 0`，说明 NPU 输出与 FP32 golden 一致；`ideal_abs_error = 1`，说明数学真值已经无法由 FP32 输出表达。

**这个 case 比单纯测 `1e10 + 1e-5` 更直观，因为它直接展示了 FP32 的“整数连续表示能力”边界：`2^24` 之前整数基本可以连续表示，到了 `2^24` 附近后，相邻整数开始出现空洞。**

### 4.5 FP16：半精度 ULP 边界和大数区间吞小数

对应 case：

```text
CASE_Add_Float16_PrecisionObservation_TieAndLargeSmall
```

这个用例直接用 FP16 bit pattern 构造输入，避免十进制字面量先被 host 编译器转换时引入额外不确定性。实测结果如下：

| index | ideal_double | dtype_expected | actual_as_double | ideal_abs_error | ideal_rel_error | dtype_loss | dtype_abs_error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1.00048828125` | `1.0009765625` | `1` | `0.00048828125` | `0.0004880429477794046` | `0.00048828125` | `0.0009765625` |
| 1 | `-1.00048828125` | `-1.0009765625` | `-1` | `0.00048828125` | `0.0004880429477794046` | `0.00048828125` | `0.0009765625` |
| 2 | `2048.5` | `2048` | `2048` | `0.5` | `0.00024408103490358799` | `0.5` | `0` |
| 3 | `0.33349609375` | `0.33349609375` | `0.33349609375` | `0` | `0` | `0` | `0` |

这里有一个需要特别说明的点：index 0/1 的 `dtype_abs_error` 不是 0，但 case 仍然 PASS，因为该用例使用了更宽的 FP16 容差观察边界行为。**这个现象说明当前执行路径在 half tie 附近的实际舍入结果为 `1` / `-1`，而 host 侧 `FloatToHalfBits` 生成的 `dtype_expected` 为 `1.0009765625` / `-1.0009765625`。由于差值正好是一个 half ULP（`0.0009765625`），它暴露的是半精度边界舍入策略差异，而不是大范围数值错误。**

**index 2 中，`2048.5` 最终输出为 `2048`。FP16 尾数位较少，在 `2048` 这个量级附近的表示间隔已经大于 `0.5`，所以 `0.5` 被吞掉。这个结果和 FP32 大数吞小数是同一类问题，只是 FP16 的有效位更少，问题出现得更早、更明显。**

### 4.6 BF16：尾数位更少导致更粗的量化

对应 case：

```text
CASE_Add_BF16_PrecisionObservation_TieAndLargeSmall
```

BF16 与 FP32 有相同的指数位宽，但尾数位明显更少。因此它的动态范围很大，但局部分辨率比 FP16/FP32 都粗。实测结果如下：

| index | ideal_double | dtype_expected | actual_as_double | ideal_abs_error | ideal_rel_error | dtype_loss | dtype_abs_error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1.00390625` | `1` | `1` | `0.00390625` | `0.0038910505836575876` | `0.00390625` | `0` |
| 1 | `-1.00390625` | `-1` | `-1` | `0.00390625` | `0.0038910505836575876` | `0.00390625` | `0` |
| 2 | `256.25` | `256` | `256` | `0.25` | `0.00097560975609756097` | `0.25` | `0` |
| 3 | `0.3359375` | `0.3359375` | `0.3359375` | `0` | `0` | `0` | `0` |

这组结果非常符合 BF16 的特点：

- 在 1.0 附近，`1.00390625` 被量化为 `1`；
- 在 256 附近，`256.25` 被量化为 `256`；
- 所有点的 `dtype_abs_error = 0`，说明 NPU 输出与 BF16 预期 一致。

和 FP16 相比，BF16 的优势是指数范围更接近 FP32，不容易因为范围不够而溢出；但代价是尾数只有 7 bit 左右，1.0 附近的分辨率约为 `1/128` 量级。

### 4.7 十进制小数、特殊值和 bit pattern golden

除了上面几组专门的 precision observation，还保留了几类精度相关 case，用来覆盖常见输入风险：

| case | 结果 | 观察结果 |
| --- | --- | --- |
| `CASE_Add_Float32_FractionalAlpha_InputQuantized_Probe` | PASS | 覆盖 `0.1f`、`0.2f`、`0.3f` 等十进制小数不能被二进制浮点精确表示的情况；golden 按 FP32 实际存储值计算，而不是按十进制直觉计算。 |
| `CASE_Add_Float32_SpecialValues_NaN_Inf` | PASS | 覆盖 `NaN` / `Inf` 传播。运行日志中该 case 通过，同时模拟器打印过 `vec_err_idata_inf_nan_t0`，说明硬件/模拟器确实检测到了输入中的 inf/nan，但算子结果符合预期传播规则。 |
| `CASE_Add_Float16_OutputFloat16_InputBitsReference` | PASS | 用 FP16 bit pattern 作为输入和输出 golden，避免 half 字面量转换差异造成误判。 |
| `CASE_Add_BF16_OutputBF16_InputBitsReference` | PASS | 用 BF16 bit pattern 作为输入和输出 golden，保证 BF16 验证基线稳定。 |
| `CASE_Add_Int32_Alpha1_TilingInt32` | PASS | 整数路径不使用浮点容差，要求精确匹配。 |
| `CASE_Add_Int32_Alpha2_AxpyV2OrAxpy` | PASS | 验证 integer alpha 缩放后仍按整数语义精确比较。 |

### 4.8 本轮精度测试结论

本轮精度测试的结论可以概括为三点：

1. **通过的精度 case 中，NPU 输出整体符合 dtype golden。** FP32、BF16 的 precision observation 中 `dtype_abs_error` 均为 0；FP16 tie 边界 case 存在 1 个 half ULP 级别的舍入差异，但仍在该观察用例设置的容差内。
2. **主要误差来自 dtype 表示能力，而不是 Add 算子额外算错。** 大数吞小数、`2^24 + 1` 回落到 `2^24`、BF16 的 `1.00390625 -> 1` 都是目标 dtype 的量化结果。

因此认为，当前 Add 测试中的精度部分已经覆盖了大数吞小数、正负抵消、ULP 边界、FP16/BF16 低精度量化、十进制小数量化、NaN/Inf 传播和整数精确匹配等核心场景。

------
## 五、反思与改进

### 5.1 测试盲区

- 当前覆盖依赖真实运行平台，无法通过单个测试文件覆盖所有 SoC 分支。
- `DOUBLE` 在当前环境路由到 AICPU 后 workspace 查询失败，无法继续验证执行结果。
- 非 ND format 的 `Adds` case 在当前环境 workspace 查询失败。
- 部分 `alpha == 1` direct-add 成功执行路径存在卡死风险，因此采用了 `alpha=2` 或 workspace-only 规避。

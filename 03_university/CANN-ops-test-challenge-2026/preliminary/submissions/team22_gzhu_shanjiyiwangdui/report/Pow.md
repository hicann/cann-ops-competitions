------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "闪击翼王队" 

team_members:

- "成员1：周亚超-广州大学"
- "成员2：张雨桐-广州大学"
- "成员3：许恒恒-广州大学" 

operator_name: "mul" 
operator_library: "cann-ops-math" 
report_date: "2026-04-25"

------

# Pow 算子测试报告

------

## 一、算子理解

Pow 算子执行逐元素幂运算，数学定义为 `out[i] = self[i] ^ exponent[i]`。与 Add、Mul 等对称二元算子不同，Pow 的两个输入具有明确的非对称语义：第一个输入是底数，第二个输入是指数，二者交换后一般不会得到相同结果。例如 `2^3 = 8`，而 `3^2 = 9`。因此 Pow 在 API 层被拆分为 TensorScalar、ScalarTensor、TensorTensor、Exp2 以及对应 inplace 形式，不同入口会触发不同的参数校验、类型提升、shape 推导与设备侧调度路径。

本题中的 Pow 算子位于 `math/pow/` 目录下，整体采用 `op_api -> op_host -> op_kernel` 三层结构。`op_api/aclnn_pow.cpp` 主要覆盖 TensorScalar、ScalarTensor 与 inplace TensorScalar 等接口；`op_api/aclnn_pow_tensor_tensor.cpp` 覆盖 TensorTensor 与 inplace TensorTensor；`op_api/aclnn_exp2.cpp` 覆盖 `2^x` 特殊 API；`op_api/pow.cpp` 负责底层设备路由、dtype 支持判断与 AiCore/AiCpu 路径选择；`op_host/arch35/` 下的 tiling 文件负责根据 shape、dtype、广播关系等信息生成执行策略。

在数学行为上，Pow 比 Mul 更容易出现边界结果。指数为 0 时，非零底数的结果应为 1；指数为 1 时结果应保持原值；指数为 0.5 时等价于平方根，仅对非负底数有实数结果；指数为 -1 时等价于倒数，底数为 0 时会产生 inf 或异常边界；负底数配合整数指数可以得到有限实数，但负底数配合非整数指数通常会得到 NaN。浮点类型还会遇到上溢、下溢、NaN、Inf 传播等问题，因此测试时不能只覆盖普通正数输入。

从 dtype 角度看，Pow 支持 BF16、FLOAT16、FLOAT32、UINT8、INT8、INT16、INT32 等类型，其中 TensorTensor tiling 文件中会根据 dtype 分配不同的 OP_KEY。不同 dtype 不只是精度不同，也可能走到不同的 tiling 分支，因此 dtype 维度会直接影响覆盖率。

------

## 二、测试策略与用例设计

本次在 `math/pow/examples/test_aclnn_pow.cpp` 中基于官方示例进行了扩展，当前提交共包含 4 个端到端测试用例，均使用 ACL 运行时创建 device tensor、调用 aclnn API 获取 workspace、执行算子、同步 stream、将结果拷回 host，并在 CPU 侧使用 `std::pow` 计算期望值进行验证。

第一部分是基础 TensorScalar 功能验证。`TestCase1_Basic2x2_PositiveFloatExp` 构造 shape 为 `[2, 2]` 的 FP32 张量，输入为 `{0, 1, 2, 3}`，指数为 `4.1f`，调用 `aclnnPowTensorScalar` 计算 `self ^ 4.1`。该用例覆盖了普通 TensorScalar API、二维连续 tensor、正小数指数以及底数中含 0 的场景。CPU Oracle 对每个元素执行 `std::pow(input[i], exponent)`，再与 NPU 输出逐元素比较。

第二部分是负指数验证。`TestCase2_1D_NegativeIntExp` 构造 shape 为 `[5]` 的 FP32 一维张量，输入为 `{1, 2, 4, 8, 16}`，指数为 `-2.0f`，调用 `aclnnPowTensorScalar`。该用例主要覆盖倒数类计算路径，即 `x^-2 = 1 / x^2`，可以验证负指数下的数值正确性。由于输入均为正数且远离 0，预期输出为有限小数，不涉及除零或 NaN。

第三部分是特殊指数 0。`TestCase3_3D_ZeroExp` 构造 shape 为 `[2, 2, 2]` 的 FP32 三维张量，输入为 `{1, 2, 3, 4, 5, 6, 7, 8}`，指数为 `0.0f`，调用 `aclnnPowTensorScalar`。该用例覆盖三维 shape 和 `x^0 = 1` 的特殊指数分支。Pow 实现中通常会对 0、1、2、3、0.5、-1 等指数做特殊优化或快速路径判断，因此该用例对 API 层分支覆盖有直接价值。

第四部分是 inplace TensorScalar 验证。`TestCase4_Inplace_Asymmetric_OneExp` 构造 shape 为 `[3, 2]` 的 FP32 张量，输入为 `{10, 20, 30, 40, 50, 60}`，指数为 `1.0f`，调用 `aclnnInplacePowTensorScalar`。该用例覆盖 inplace API、非方阵二维 shape，以及 `x^1 = x` 的恒等变换场景。测试执行后直接从 self 的 device 地址拷回结果，验证 inplace 写回是否正确。

**Oracle 选择**：当前代码中的 `VerifyResult` 使用模板函数对每个输出元素执行 `std::pow(input[i], exponent)`，并以 `fabs(output[i] - expected) <= 1e-4` 作为通过条件。该做法能满足 FP32 常规小规模输入的结果验证要求，但仍有两个改进点：一是期望值最好显式提升为 double，即 `double expected = std::pow((double)input[i], (double)exponent)`，避免参考计算自身过早降精度；二是验证函数目前只打印 mismatch，并不会将失败状态返回给主流程，因此严格意义上“结果不一致时返回非 0”的要求还未完全满足。

**当前用例覆盖矩阵**：

| 用例 | API 类型 | API 名称 | dtype | shape | 指数 | 主要覆盖点 |
| ---- | -------- | -------- | ----- | ----- | ---- | ---------- |
| TestCase1 | TensorScalar | `aclnnPowTensorScalar` | FP32 | `[2,2]` | `4.1` | 基础二维、正小数指数、含 0 底数 |
| TestCase2 | TensorScalar | `aclnnPowTensorScalar` | FP32 | `[5]` | `-2` | 一维 shape、负指数、倒数类计算 |
| TestCase3 | TensorScalar | `aclnnPowTensorScalar` | FP32 | `[2,2,2]` | `0` | 三维 shape、特殊指数 0 |
| TestCase4 | Inplace TensorScalar | `aclnnInplacePowTensorScalar` | FP32 | `[3,2]` | `1` | inplace 写回、特殊指数 1、非方阵 shape |

从设计意图看，这 4 个用例优先保证 TensorScalar 主路径与 inplace TensorScalar 路径能端到端运行，并且覆盖了正小数指数、负整数指数、0 次幂、1 次幂等基本数学分支。

------

## 三、覆盖率分析

本次评分文件为题目 C 指定的 5 个源文件。由于当前报告未读取到实际 `gcov -b` 输出，下表按照测试代码可触达路径进行预期覆盖分析；最终提交时应以运行 `bash build.sh --run_example pow eager cust --vendor_name=custom --simulator --soc=ascend950 --cov` 后产生的 `.gcda/.gcov` 数据为准。

**评分文件**

| 文件 | 代码行数 | 预期行覆盖率 | 预期分支覆盖率 | 说明 |
| ---- | -------- | ------------ | -------------- | ---- |
| `op_api/aclnn_pow.cpp` | 236 | 中等 | 偏低 | 当前用例覆盖 `aclnnPowTensorScalar` 与 `aclnnInplacePowTensorScalar`，能触达 TensorScalar 参数校验、workspace 获取、调度与部分特殊指数逻辑，但 ScalarTensor 路径未覆盖。 |
| `op_api/aclnn_pow_tensor_tensor.cpp` | 73 | 0% 或接近 0% | 0% 或接近 0% | 当前代码没有调用 `aclnnPowTensorTensor` 或 `aclnnInplacePowTensorTensor`，因此该文件基本不会被执行。 |
| `op_api/pow.cpp` | 30 | 中等 | 偏低 | TensorScalar 调度最终会进入底层路由逻辑，但当前仅覆盖 FP32 正常路径，nullptr、非法 dtype、AiCpu fallback 等异常/分支路径未覆盖。 |
| `op_host/arch35/pow_tensor_tensor_tiling_arch35.cpp` | 126 | 0% 或接近 0% | 0% 或接近 0% | 当前未调用 TensorTensor API，也未覆盖 TensorTensor dtype OP_KEY 分发，因此该 tiling 文件预计基本未触达。 |
| `op_host/arch35/pow_tiling_arch35.cpp` | 53 | 低到中等 | 偏低 | TensorScalar 正常 shape 可能触发部分通用 tiling 逻辑，但当前 shape 均为无广播连续 tensor，tiling 分支覆盖有限。 |

**综合覆盖率预估**：

当前用例主要集中在 `aclnn_pow.cpp` 和 `pow.cpp` 的 FP32 TensorScalar 正常路径上。由于 5 个评分文件中有两个 TensorTensor 相关文件预计完全未覆盖，且 ScalarTensor、Exp2、多 dtype、广播、异常路径均未覆盖，综合行覆盖率和分支覆盖率预计不会很高。若实际 gcov 结果显示 `pow_tensor_tensor_tiling_arch35.cpp` 与 `aclnn_pow_tensor_tensor.cpp` 为 0%，这是当前 API 调用范围导致的合理结果，而不是覆盖率工具异常。

未覆盖部分的归因主要有四点。第一，API 入口不完整：当前只调用了 2 个 API，题目要求中的 7 个 API 尚未完全覆盖。第二，dtype 维度单一：所有用例均为 `ACL_FLOAT`，无法触达 FLOAT16、BF16、INT32、INT8 等不同 OP_KEY 分支。第三，shape 维度偏保守：所有输入输出 shape 相同，没有覆盖广播、标量 base、TensorTensor 广播对齐等路径。第四，异常路径未覆盖：nullptr、不支持 dtype、shape 不兼容等分支没有被测试。

------

## 四、精度分析

精度分析章节按当前代码实际覆盖到的四类典型场景展开。所有场景的参考值均来自 CPU 侧 `std::pow`，比较容差为 `1e-4` 绝对误差。对当前输入规模和 FP32 结果而言，该容差可以覆盖常规舍入误差；但对于极大值、极小值、NaN、Inf、FP16/BF16 等场景，后续应改为 dtype 感知的 `atol + rtol * abs(expected)` 策略，并对 NaN/Inf 单独判断。

### 场景一：正小数指数与 0 底数

**测试输入**：`self = {0, 1, 2, 3}`，`exponent = 4.1`，shape 为 `[2,2]`。

**预期输出**：

```
0^4.1 = 0
1^4.1 = 1
2^4.1 ≈ 17.148376
3^4.1 ≈ 90.460395
```

**分析**：

该用例验证了普通正数底数和正小数指数的基础路径。`0^4.1` 在数学上为 0，属于有效边界；`2^4.1` 与 `3^4.1` 不是整数结果，需要经过对数/指数或专用 pow 实现路径，能够检验浮点近似计算的正确性。由于结果量级不大，FP32 误差通常可以被 `1e-4` 容差覆盖。

相关测试用例：

```cpp
TestCase1_Basic2x2_PositiveFloatExp(stream);
```

**风险**：若指数为非整数且底数为负数，`std::pow` 会返回 NaN；当前用例尚未覆盖该风险。后续应补充 `{-1, -2, -4}` 配合 `0.5`、`2.0`、`3.0` 的对比，以区分负数整数次幂与负数非整数次幂。

------

### 场景二：负整数指数

**测试输入**：`self = {1, 2, 4, 8, 16}`，`exponent = -2`，shape 为 `[5]`。

**预期输出**：

```
1^-2  = 1
2^-2  = 0.25
4^-2  = 0.0625
8^-2  = 0.015625
16^-2 = 0.00390625
```

**分析**：

负指数本质上会引入倒数计算，`x^-2 = 1 / (x^2)`。该用例输入均为 2 的幂，理论结果可以被二进制浮点精确表示，因此非常适合作为验证负指数路径的稳定样例。如果该场景出现明显误差，通常说明调度路径、指数处理或结果写回存在问题，而不是普通浮点舍入误差导致。

相关测试用例：

```cpp
TestCase2_1D_NegativeIntExp(stream);
```

**风险**：当前输入未包含 0。若底数为 0 且指数为负数，例如 `0^-1`，数学上对应除以 0，通常会得到 inf 或触发特殊处理。该边界对 Pow 十分重要，后续建议单独补充 `self = {0, 1, 2}`、`exponent = -1` 的测试，并使用 `std::isinf` 判断结果。

------

### 场景三：0 次幂

**测试输入**：`self = {1, 2, 3, 4, 5, 6, 7, 8}`，`exponent = 0`，shape 为 `[2,2,2]`。

**预期输出**：

```
{1, 1, 1, 1, 1, 1, 1, 1}
```

**分析**：

`x^0 = 1` 是 Pow 的关键特殊指数场景。该用例使用三维 shape，既验证了特殊指数分支，又验证了非二维 tensor 的基本 shape 处理。由于所有底数均为正且非零，预期行为非常明确。若实现内部存在指数为 0 的快速路径，该用例可以覆盖相应分支。

相关测试用例：

```cpp
TestCase3_3D_ZeroExp(stream);
```

**风险**：当前用例未覆盖 `0^0`。在 C/C++ `std::pow(0.0, 0.0)` 中通常返回 1，但不同数学语境下 `0^0` 有争议，算子实现也可能显式定义其行为。由于题目边界中特别提到 `0^0`，后续应补充含 0 底数且指数为 0 的用例，并以实际算子规范为准进行验证。

------

### 场景四：inplace 与 1 次幂

**测试输入**：`self = {10, 20, 30, 40, 50, 60}`，`exponent = 1`，shape 为 `[3,2]`。

**预期输出**：

```
{10, 20, 30, 40, 50, 60}
```

**分析**：

`x^1 = x` 是另一个常见特殊指数分支。该用例调用 `aclnnInplacePowTensorScalar`，输出不再写入单独的 out tensor，而是原地覆盖 selfRef。测试完成后从 self 的 device 地址拷回结果，验证 inplace 写回路径是否正确。该用例对于覆盖 inplace 参数校验、workspace 获取、执行器调度和原地内存写回具有价值。

相关测试用例：

```cpp
TestCase4_Inplace_Asymmetric_OneExp(stream);
```

**风险**：由于指数为 1，数学结果与输入完全一致，若算子执行被错误跳过，仍可能表面通过。因此 inplace 用例后续建议增加指数为 2 或 -1 的版本，使结果发生实际变化，从而更有效地验证原地计算是否真的执行。

------

## 五、反思与改进

**API 覆盖不完整**。当前测试只覆盖 `aclnnPowTensorScalar` 与 `aclnnInplacePowTensorScalar`，尚未覆盖 `aclnnPowScalarTensor`、`aclnnPowTensorTensor`、`aclnnInplacePowTensorTensor`、`aclnnExp2`、`aclnnInplaceExp2`。题目明确指出 Pow 的不同 API 位于不同源文件中，不调用某类 API，对应源文件不会执行。因此下一步首要改进是补齐 7 个 API 的最小正确性用例。建议至少增加：ScalarTensor 的 `2.0 ^ exponentTensor`，TensorTensor 的 `baseTensor ^ exponentTensor`，inplace TensorTensor 的原地平方，Exp2 的 `2^x`，inplace Exp2 的原地写回。

**dtype 维度不足**。当前所有测试均为 FP32，无法触发 `pow_tensor_tensor_tiling_arch35.cpp` 中不同 dtype 对应的 OP_KEY 分发。建议补充 FLOAT16、BF16、INT32、INT8 至少各 1 组 TensorTensor 用例。对 FP16/BF16 需要实现位模式与 float 之间的转换，不能直接把 `uint16_t` 位模式当作数值做 `std::pow`；对 INT8/INT32 需要明确整数 pow 的溢出、截断或类型提升语义。

**shape 与广播覆盖不足**。当前 4 个用例均为输入 shape 与输出 shape 完全一致，不涉及 broadcasting。Pow 的题目说明强调支持 TensorScalar、ScalarTensor、TensorTensor 以及 shape 不一致时的广播对齐，因此后续建议增加 `[2,3] ^ [3]`、`[2,1,3] ^ [1,4,1]`、标量 base 与 tensor exponent 等组合。广播用例不仅提升功能覆盖，也更可能触达 host tiling 分支。

**特殊指数覆盖仍不完整**。当前覆盖了 `0`、`1`、`-2` 和 `4.1`，但尚未覆盖题目中特别提到的 `0.5`、`2`、`3`、`-1`。这些指数可能在 `aclnn_pow.cpp` 中有 sqrt、square、cube、reciprocal 等专用优化路径。建议分别设计 `sqrt`、平方、立方、倒数用例，并确保输入能暴露计算差异，例如 `self = {0.25, 1, 4, 9}` 配合 `0.5`，`self = {2, 3, 4}` 配合 `3`。

**边界数值覆盖不足**。当前未覆盖 NaN、Inf、极大指数导致上溢、负数非整数次幂、`0^0`、`0^-1` 等高风险场景。Pow 的数学边界远比 Mul/Add 丰富，建议为这些场景实现专门验证逻辑：NaN 使用 `std::isnan`，Inf 使用 `std::isinf`，上溢不要用普通 atol 比较，负数非整数指数应与 CPU `std::pow` 的 NaN 行为保持一致或按算子规范判断。

**结果验证逻辑需要强化**。当前 `VerifyResult` 能打印 mismatch，但函数返回类型为 `void`，主流程无法感知数值失败，因此即使存在误差超限，程序仍可能返回 0。为了满足“每个测试用例输出 [PASS] 或 [FAIL]，程序结尾输出汇总，有失败用例返回非 0 值”的要求，建议将验证函数改为返回 `bool` 或失败计数，并在 `main` 中累计失败数。示例逻辑如下：

```cpp
bool VerifyResult(const std::vector<float>& input,
                  float exponent,
                  const std::vector<float>& output,
                  const std::string& testName) {
    bool ok = true;
    for (size_t i = 0; i < output.size(); ++i) {
        double expected = std::pow((double)input[i], (double)exponent);
        double actual = (double)output[i];
        double atol = 1e-5;
        double rtol = 1e-5;
        if (std::isnan(expected)) {
            ok &= std::isnan(actual);
        } else if (std::isinf(expected)) {
            ok &= std::isinf(actual) && (std::signbit(expected) == std::signbit(actual));
        } else {
            ok &= std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
        }
    }
    LOG_PRINT("[%s] %s\n", ok ? "PASS" : "FAIL", testName.c_str());
    return ok;
}
```

**优先级排序**。若继续提升覆盖率，建议按以下顺序推进：① 补齐 7 个 API 入口，优先覆盖 TensorTensor 与 Exp2 独立源文件；② 补 TensorTensor 多 dtype，触发不同 OP_KEY；③ 补 broadcasting shape，提升 host tiling 覆盖；④ 补特殊指数 `0.5/2/3/-1`，触发优化分支；⑤ 补 NaN/Inf/0^0/负底数非整数指数/异常输入，提高分支覆盖率与测试鲁棒性。按照该顺序扩展后，预计 `aclnn_pow_tensor_tensor.cpp` 与 `pow_tensor_tensor_tiling_arch35.cpp` 的覆盖率会从当前近似空白状态显著提升，综合覆盖率也会比当前仅 TensorScalar 主路径高很多。

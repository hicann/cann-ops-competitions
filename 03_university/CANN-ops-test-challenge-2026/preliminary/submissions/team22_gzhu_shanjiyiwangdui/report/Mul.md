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

# Mul 算子测试报告
------

## 一、算子理解

Mul 算子执行逐元素乘法，数学定义为 `out[i] = x1[i] × x2[i]`。当两个输入张量形状完全一致时，输出与输入形状一致；当输入形状不一致但满足 broadcasting 规则时，框架会按广播语义对齐维度后逐元素计算。由于每个输出元素仅依赖对应位置的两个输入元素，因此该算子本身不存在归约类算子中常见的计算顺序误差传播问题。

在 CANN ops-math 仓库中，Mul 算子采用 `op_api → op_host → op_kernel` 的三层结构：`op_api` 负责参数校验、类型提升和 API 入口分发；`op_host` 负责 shape 推导和 tiling 策略计算；`op_kernel` 负责在 NPU 侧按照 tiling 方案执行乘法。题目要求关注的评分文件包括 `op_api/aclnn_mul.cpp`、`op_api/mul.cpp` 和 `op_host/arch35/mul_tiling_arch35.cpp`。

本算子对外提供四类 API：`aclnnMul(self, other, out)`、`aclnnMuls(self, other, out)`、`aclnnInplaceMul(selfRef, other)`、`aclnnInplaceMuls(selfRef, other)`。其中 `Mul` 覆盖 Tensor × Tensor 路径，`Muls` 覆盖 Tensor × Scalar 路径，两个 Inplace API 覆盖原地写回路径。不同 API 在 `op_api` 层通常会进入不同的参数检查、输出构造和执行器调度分支，因此 API 入口覆盖是提高 `aclnn_mul.cpp` 覆盖率的关键。

精度方面，Mul 的单次乘法误差通常较小，但仍需关注如下风险：浮点极大值相乘可能上溢为 `inf`，极小值相乘可能下溢为 0 或进入次正规区间，`NaN` 和 `Inf` 会沿后续计算传播，整数乘法若结果超出类型表示范围可能出现低位截断或实现定义/平台相关行为。因此，测试用例除覆盖普通功能外，还应覆盖边界数值和异常输入。

------

## 二、测试策略与用例设计

本次提交的 `test_aclnn_mul.cpp` 在官方示例基础上扩展了 ACL 初始化、张量构造、标量构造、API 调用、异常路径触发和最终汇总输出。代码中定义了 `FLOAT32_ATOL=1e-5`、`FLOAT16_ATOL=1e-3`、`BF16_ATOL=1e-2` 等容差常量，并实现了 `IsEqual` 与 `VerifyResult` 两个辅助函数，支持浮点容差比较、`NaN` 对 `NaN` 的匹配、同号 `Inf` 的匹配以及整数精确匹配。

但是需要特别说明：当前主要执行函数 `TestAclnnMulSimple` 在调用 `aclnnMul` 并同步 stream 后，仅打印 `[PASS] (execution only)`，没有将 device 侧输出拷回 host，也没有调用 `VerifyResult` 对实际输出与 CPU 期望值进行比对。因此，当前代码具备结果验证函数雏形，但多数用例仍属于“执行成功即通过”的覆盖率导向用例，而不是完整的数值正确性验证用例。若作为正式提交，建议在 `TestAclnnMulSimple` 和 `Muls_API_Test` 中补充 `aclrtMemcpy(..., ACL_MEMCPY_DEVICE_TO_HOST)`，并用 CPU 参考结果调用 `VerifyResult`。

本次用例设计可以分为五类。

第一部分是基础 Tensor × Tensor 功能测试。`Basic_MUL_Test` 使用 FP32 类型，输入 shape 为 `[2, 2]`，数据分别为 `{1, 2, 3, 4}` 与 `{2, 3, 4, 5}`，通过 `aclnnMulGetWorkspaceSize` 获取执行器并调用 `aclnnMul`。该用例主要覆盖最常规的同 shape、同 dtype、非原地 Tensor × Tensor 执行路径。

第二部分是 Tensor × Scalar API 覆盖。`Muls_API_Test` 构造 FP32 的 `[2, 2]` 输入张量，scalar 值为 `2.5f`，调用 `aclnnMulsGetWorkspaceSize` 和 `aclnnMuls`。该用例覆盖 `Muls` 入口，有助于触发不同于 `aclnnMul` 的标量参数处理分支、scalar 创建/销毁路径以及 `op_api` 层 API 变体分发逻辑。

第三部分是异常输入与边界 shape 覆盖。`TestMulCppCoverage` 内部包含 5 个子用例：`Mul_Nullptr_Test` 直接向 `aclnnMulGetWorkspaceSize` 传入三个空 tensor 指针并期望返回失败；`Mul_Incompatible_Shape_Test` 构造 `[2,3]` 与 `[4,5]` 两个不可广播 shape 并期望 workspace 获取失败；`Mul_Empty_Tensor_Test` 构造 shape 为 `[0,2]` 的空 tensor，允许算子接受空 tensor 或被参数校验拒绝，两种结果均视为覆盖成功；`Mul_Scalar_Single_Element` 使用空 shape `{}` 表示单元素标量张量；`Mul_Large_Shape` 使用长度为 1000 的一维张量覆盖较大输入规模。

第四部分是 dtype/promotion 相关路径尝试。`TestMixedDtypes` 的注释目标是覆盖 `FLOAT16 * FLOAT -> FLOAT` 的混合 dtype 提升逻辑，但当前实现因环境限制仍使用 `ACL_FLOAT` 与 FP32 数据模拟混合 dtype。因此，该用例能够覆盖一条额外的 FP32 `aclnnMul` 调用路径，但不能真正覆盖 op_api 中混合 dtype promote 分支。若要完成题目对 dtype 组合覆盖的要求，需要补充 `ACL_FLOAT16`、`ACL_BF16`、`ACL_INT32`、`ACL_INT64` 等真实 dtype 用例，并在张量创建、host 参考值计算和结果打印中按 dtype 正确编码/解码。

第五部分是测试输出与进程返回值控制。主函数维护 `totalTests` 与 `passedTests`，每个用例输出 `[PASS]` 或 `[FAIL]`，结尾输出 `Total tests`、`Passed tests`、`Failed tests`，并在所有用例通过时返回 0，否则返回 -1。这符合题目对测试程序输出格式和失败返回非 0 的要求。

**当前静态统计的用例清单如下：**

| 序号 | 用例名称 | API/路径 | dtype | shape/输入特征 | 预期目的 | 当前验证方式 |
| ---- | -------- | -------- | ----- | --------------- | -------- | ------------ |
| 1 | `Basic_MUL_Test` | `aclnnMul` | FP32 | `[2,2] × [2,2]` | 基础同 shape 乘法 | execution only |
| 2 | `Muls_API_Test` | `aclnnMuls` | FP32 + scalar | `[2,2] × scalar(2.5)` | 覆盖 Tensor × Scalar API | execution only |
| 3 | `Mul_Nullptr_Test` | `aclnnMulGetWorkspaceSize` | 无 | `nullptr` | 参数空指针异常路径 | 返回失败即 PASS |
| 4 | `Mul_Incompatible_Shape_Test` | `aclnnMulGetWorkspaceSize` | FP32 | `[2,3] × [4,5]` | 不可广播 shape 异常路径 | 返回失败即 PASS |
| 5 | `Mul_Empty_Tensor_Test` | `aclnnMul` / 参数校验 | FP32 | `[0,2] × [0,2]` | 空 tensor 边界 | 接受执行或拒绝均 PASS |
| 6 | `Mul_Scalar_Single_Element` | `aclnnMul` | FP32 | `{}` × `{}` | 单元素/标量张量 | execution only |
| 7 | `Mul_Large_Shape` | `aclnnMul` | FP32 | `[1000] × [1000]` | 较大一维 tensor | execution only |
| 8 | `Mul_Mixed_Dtype_Simulation` | `aclnnMul` | FP32 | `[2,2] × [2,2]` | 模拟混合 dtype 路径 | execution only |

**Oracle 选择**：当前代码中已定义 `IsEqual` / `VerifyResult`，设计上可作为 CPU Oracle 的比较入口。正式版本建议对每个测试用例在 host 端独立计算 `expected[i] = self[i] * other[i]` 或 `expected[i] = self[i] * scalar`，再将 device 输出拷回 host 后比较。FP32 建议使用 `atol=1e-5, rtol=1e-5`；FP16 建议使用 `atol=1e-3, rtol=1e-3`；BF16 建议使用 `atol=1e-2, rtol=1e-2`；整数类型应精确匹配。

------

## 三、覆盖率分析

本报告未收到实际 `gcov -b` 输出，因此以下覆盖率分析为基于代码结构的静态预估和待回填模板。正式提交前应在评测环境中执行题目给出的完整流程：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example mul eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
find build -name "*.gcda" | grep mul
gcov -b <gcda文件路径>
```

**评分文件**

| 文件 | 代码行数 | 行覆盖率 | 分支覆盖率 | 当前用例预计覆盖情况 |
| ---- | -------- | -------- | ---------- | -------------------- |
| `op_api/aclnn_mul.cpp` | 313 | 待 gcov 回填 | 待 gcov 回填 | 已覆盖 `aclnnMul` 与 `aclnnMuls` 两类入口；未覆盖 `aclnnInplaceMul` 与 `aclnnInplaceMuls`；真实混合 dtype promote 分支覆盖不足 |
| `op_api/mul.cpp` | 47 | 待 gcov 回填 | 待 gcov 回填 | 通过 nullptr、不可广播 shape、空 tensor 等用例触发部分参数校验/失败分支；普通 FP32 路径触发设备路由成功分支 |
| `op_host/arch35/mul_tiling_arch35.cpp` | 192 | 待 gcov 回填 | 待 gcov 回填 | 当前主要为 FP32、同 shape 或简单一维 shape；可能触发基础 tiling 路径，但 broadcasting、复杂 shape、不同 dtype 组合不足 |

**综合覆盖率计算方式**：

- 行覆盖率按题目给出的有效行数加权：`(aclnn_mul.cpp 覆盖行 + mul.cpp 覆盖行 + mul_tiling_arch35.cpp 覆盖行) / (313 + 47 + 192)`
- 分支覆盖率按 `gcov -b` 输出的分支命中数加权：`三文件已覆盖分支数之和 / 三文件总分支数之和`

**当前覆盖优势**：

1. `aclnnMul` 正常路径有多组调用，包含 `[2,2]`、`{}`、`[1000]` 等 shape。
2. `aclnnMuls` 标量 API 已有一组基础覆盖。
3. `mul.cpp` 相关异常路径有明确设计，包括 nullptr 和不可广播 shape。
4. 空 tensor 用例能覆盖 size 为 0 时的张量构造和参数处理路径。
5. 主函数具备统一汇总和失败返回非 0 的控制逻辑。

**当前覆盖短板**：

1. `aclnnInplaceMul`、`aclnnInplaceMuls` 完全未覆盖，`op_api/aclnn_mul.cpp` 中原地 API 相关分支大概率缺失。
2. 混合 dtype 用例目前只是 FP32 模拟，不能真正覆盖 `FLOAT16-FLOAT-FLOAT`、`BF16-FLOAT-FLOAT` 等 promote 分支。
3. broadcasting 功能覆盖不足，当前不可广播 shape 用于异常测试，但缺少合法广播用例，例如 `[2,3] × [3]`、`[2,1,3] × [1,4,3]`、`scalar × tensor` 等。
4. 结果验证函数未接入主执行路径，覆盖率评估虽然可能通过，但结果验证得分存在扣分风险。
5. 数值边界用例不足，尚未覆盖零值、负数、NaN、Inf、极大/极小浮点值和整数溢出等场景。

------

## 四、精度分析

当前代码中定义了浮点容差常量与通用比较函数，因此具备开展精度验证的基础。`IsEqual` 对浮点类型使用如下规则：若两者均为 `NaN` 则判定相等；若两者均为同号 `Inf` 则判定相等；否则使用 `|actual - expected| <= atol + rtol × |expected|`。整数类型则使用 `a == b` 精确匹配。

### 场景一：FP32 普通乘法

**测试输入**：`Basic_MUL_Test` 中 `self={1.0, 2.0, 3.0, 4.0}`，`other={2.0, 3.0, 4.0, 5.0}`，shape 均为 `[2,2]`。

**理论期望输出**：

```text
[2.0, 6.0, 12.0, 20.0]
```

**分析**：这些输入均为 FP32 可精确表示的小整数，乘积也在 FP32 精确表示范围内。因此，若 device 输出拷回 host，理论上应与 CPU 结果完全一致；即使用 `1e-5` 的 FP32 容差也应稳定通过。当前代码只验证执行成功，尚未验证上述数值结果。

相关建议实现：

```cpp
std::vector<float> expected = {2.0f, 6.0f, 12.0f, 20.0f};
std::vector<float> actual(4);
aclrtMemcpy(actual.data(), actual.size() * sizeof(float), outDeviceAddr,
            actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
VerifyResult(actual, expected, "Basic_MUL_Test", FLOAT32_ATOL, FLOAT32_RTOL);
```

### 场景二：Tensor × Scalar 乘法

**测试输入**：`Muls_API_Test` 中 `self={1.0, 2.0, 3.0, 4.0}`，scalar 为 `2.5f`，shape 为 `[2,2]`。

**理论期望输出**：

```text
[2.5, 5.0, 7.5, 10.0]
```

**分析**：该用例覆盖 `aclnnMuls` 的标量分支，输入与结果均可被 FP32 精确表示。当前代码能够验证 API 是否成功执行，但没有读取 `out` 结果。建议补充 host 侧期望值并调用 `VerifyResult`，否则只能说明调度路径未报错，不能说明标量乘法结果正确。

### 场景三：空 tensor

**测试输入**：`Mul_Empty_Tensor_Test` 使用 shape `[0,2]`，host 数据为空。

**理论期望行为**：空 tensor 的输出元素个数为 0。算子实现可以选择接受空 tensor 并快速返回，也可以在参数校验阶段拒绝空 tensor；两种行为均应由算子规范决定。当前代码将“成功执行”和“被校验拒绝”都视为 PASS，主要目的是覆盖空 shape/空数据路径。

**分析**：从功能正确性角度，若算子规范允许空 tensor，则应进一步验证输出 shape 和返回码；若规范不允许空 tensor，则应验证返回码属于预期错误类型。当前代码对两种结果都放行，适合覆盖率探索，但不适合严格语义验证。

### 场景四：不可广播 shape

**测试输入**：`Mul_Incompatible_Shape_Test` 使用 `[2,3]` 与 `[4,5]`。从右向左对齐维度时，`3` 与 `5` 不相等且均不为 1，`2` 与 `4` 不相等且均不为 1，因此不满足 broadcasting 规则。

**理论期望行为**：`aclnnMulGetWorkspaceSize` 应返回非 `ACL_SUCCESS`，测试判定为 PASS。

**分析**：该用例是当前异常输入覆盖中最明确的一项，可以触发 shape 推导或参数校验失败分支。正式报告中应将其归类为“异常路径验证”，而不是数值精度验证。

### 场景五：nullptr 输入

**测试输入**：`Mul_Nullptr_Test` 将 `self`、`other`、`out` 均置为 `nullptr` 传入 `aclnnMulGetWorkspaceSize`。

**理论期望行为**：接口应拒绝空指针并返回非 `ACL_SUCCESS`。

**分析**：该用例用于覆盖 API 层参数合法性校验，重点不是计算结果，而是验证异常输入不会导致进程崩溃或错误执行。建议在正式版本中进一步区分返回码类型，避免把非预期错误也计为 PASS。

### 场景六：较大一维 tensor

**测试输入**：`Mul_Large_Shape` 中 `self` 长度为 1000，所有元素为 `1.0f`；`other` 长度为 1000，所有元素为 `2.0f`。

**理论期望输出**：长度为 1000 的向量，所有元素均为 `2.0f`。

**分析**：该用例可以触发比 `[2,2]` 更大的元素数量，有助于覆盖 tiling 或任务切分相关路径。不过长度 1000 仍属于较小规模，可能不足以触发更复杂的 tiling 策略。若希望提升 `mul_tiling_arch35.cpp` 覆盖率，建议增加更大规模和非整齐 shape，如 `[1024]`、`[4096]`、`[33, 65]`、`[128, 1, 64] × [1, 32, 64]` 等。

------

## 五、反思与改进

**结果验证尚未真正接入主流程**。当前代码中虽然实现了 `IsEqual` 与 `VerifyResult`，但主要测试函数 `TestAclnnMulSimple` 没有从 device 拷回输出，也没有计算 expected 后调用验证函数，而是执行成功即输出 `[PASS] (execution only)`。这会影响题目中“必须包含有效结果验证逻辑”的评分要求。下一步应优先将输出拷回和 CPU Oracle 接入所有正常计算用例。

**API 覆盖仍不完整**。当前覆盖了 `aclnnMul` 和 `aclnnMuls`，但 `aclnnInplaceMul`、`aclnnInplaceMuls` 仍未覆盖。由于原地 API 会涉及输入输出别名、selfRef 写回和不同的参数检查分支，建议至少补充两组用例：`selfRef *= other` 的 `[2,2]` FP32 用例，以及 `selfRef *= scalar` 的 `[2,2]` FP32 用例，并验证 selfRef device 数据被正确更新。

**broadcasting 正常路径不足**。当前只有不可广播 shape 的异常测试，没有合法 broadcasting 的正常测试。题目明确强调 shape 不一致时按广播规则对齐，因此应补充 `[2,3] × [3]`、`[2,1,3] × [1,4,3]`、`[1] × [1000]` 等用例。这些用例不仅能验证功能，也更可能触发 host tiling 中与 shape 适配相关的分支。

**dtype 覆盖不足**。当前实际执行的 dtype 基本都是 `ACL_FLOAT`。虽然代码中定义了 FP16/BF16 容差，也有“混合 dtype 测试”的注释，但实际实现仍是 FP32 模拟。正式提交应补充 `ACL_FLOAT16`、`ACL_BF16`、`ACL_INT32`、`ACL_INT64`、`ACL_BOOL` 等合法 dtype，以及题目提到的混合 dtype 组合。FP16/BF16 的 host 侧 Oracle 不能直接把底层位模式当整数相乘，应先正确解码为 float，再计算并按目标 dtype 舍入。

**数值边界覆盖不足**。当前正常计算数据主要是小整数和简单小数，没有覆盖 0、负数、`NaN`、`Inf`、极大值、极小值、整数溢出等边界。建议增加：`0 × x`、`negative × positive`、`inf × finite`、`nan × finite`、`1e20f × 1e20f`、`1e-20f × 1e-20f`、`INT32_MAX × 2` 等场景，并为 `NaN`/`Inf` 使用专门判定逻辑。

**异常路径需要更细粒度断言**。目前 `Mul_Empty_Tensor_Test` 将成功执行和被拒绝都视为 PASS，这有利于覆盖率探索，但不利于严谨测试。建议根据算子规范固定预期：若空 tensor 合法，应验证返回成功、输出 shape 和不崩溃；若空 tensor 非法，应验证返回失败且错误码符合预期。

**覆盖率提升优先级**：若继续完善，建议优先级为：① 将结果验证接入所有正常用例；② 补齐 `InplaceMul` / `InplaceMuls`；③ 补合法 broadcasting；④ 补真实 dtype 和混合 dtype；⑤ 补 NaN/Inf/极值/整数溢出；⑥ 根据 `gcov -b` 结果定向补未命中的异常分支和 tiling 分支。预计这些改动会比单纯增加同 shape FP32 用例更有效地提升综合覆盖率和结果验证得分。

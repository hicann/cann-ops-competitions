------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "闪击翼王队" 

team_members:

- "成员1：周亚超-广州大学"
- "成员2：张雨桐-广州大学"
- "成员3：许恒恒-广州大学" 

operator_name: "add" 
operator_library: "cann-ops-math" 
report_date: "2026-04-25"

------

# Add 算子测试报告

------

## 一、算子理解

Add 算子执行逐元素加法，数学定义为 `out[i] = x1[i] + alpha × x2[i]`。其中 `alpha` 是标量缩放因子，默认语义等价于 1。当 `alpha = 1` 时，算子退化为普通逐元素加法；当 `alpha = 0` 时，第二个输入在数学上被完全屏蔽；当 `alpha` 为负数或非整数浮点数时，计算路径会从简单 Add 扩展到带缩放的融合路径或 `Mul + Add` 组合路径。因此，相比 Mul 算子，Add 的核心测试维度不仅包括 dtype 和 shape，还必须覆盖不同 `alpha` 值对调度逻辑的影响。

Add 支持 broadcasting。两个输入 shape 不完全一致但可广播时，算子会按广播规则扩展较小 shape 后逐元素计算。例如 `[2, 2] + [1]` 会将第二个输入视为标量样式广播到整个输出。测试中需要同时覆盖同 shape、标量 shape、单元素广播、大 tensor 等情况，以触发不同 shape 推断与 tiling 分支。

在接口层，Add 对外暴露 6 类 API：`aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnAddV3` 和 `aclnnInplaceAddV3`。其中 `Add` 是 tensor 加 tensor，`Adds` 是 tensor 加 scalar，`Inplace` 系列会把结果写回输入 tensor，`AddV3` 的第一个输入是 scalar，语义为 `scalar + alpha × tensor`。V3 API 有独立头文件与独立实现文件，如果测试代码只调用标准版 API，则 `aclnn_add_v3.cpp` 中的类型提升、alpha 分支与调度逻辑不会被覆盖。

在精度方面，Add 的单次运算相对简单，但仍存在几类高风险行为。浮点类型会受到输入量化、加法舍入、`alpha × other` 乘法舍入以及极大极小值处理的影响；当一个加数远大于另一个加数时，小量可能在舍入中被吞掉。FP16 与 BF16 的有效尾数更短，对小数和大动态范围输入更敏感。整数类型不做溢出检测，`x1 + alpha × x2` 超出目标 dtype 表示范围时，实际结果取决于底层实现的低位截断或类型转换行为，容易产生看似正常但数值错误的输出。

------

## 二、测试策略与用例设计

本次在 `math/add/examples/test_aclnn_add.cpp` 中设计了 30 个测试用例，覆盖 API 入口、alpha 参数、dtype、shape、边界值、混合精度和部分异常/可选路径。

第一部分是基础功能验证。代码构造了 FP32 同 shape 输入，测试 `Basic FLOAT32 same shape`，使用 `alpha = 1.2f` 验证 `x1 + alpha × x2` 的基本语义；随后用 `[2, 2]` 与 `[1]` 输入验证广播行为，确认 `otherData` 单元素能广播到输出所有位置。INT32、INT8 和 FLOAT16 也分别设置了基础用例，用于触发不同 dtype 的算子定义、类型判断与 tiling 分支。

第二部分是 alpha 维度覆盖。测试集中覆盖了 `alpha = 1`、`alpha = 0`、负数 alpha、浮点 alpha 和整数 alpha。`FLOAT32 alpha=0` 验证第二输入被屏蔽后的结果应等于 self；`FLOAT32 negative alpha` 验证减法样式路径；`Alpha=1 optimization path` 和 `AddV3 alpha=1` 重点触发 alpha 为 1 时的直接 Add 优化路径；`Add FLOAT Axpy path`、`Add INT32 Axpy path`、`AddV3 FLOAT Axpy path` 等用例覆盖 alpha 不为 1 且 dtype 支持融合计算的路径；`Add BF16 Mul+Add path`、`AddV3 BF16 Mul+Add path`、`AddV3 INT8 Mul+Add path` 则用于覆盖不走 Axpy 时的先乘后加路径。

第三部分是 API 变体覆盖。标准 `aclnnAdd` 由 `TestAdd` 模板统一封装；`aclnnAdds` 由 `TestAdds` 覆盖 tensor + scalar；`aclnnInplaceAdd` 由 `TestInplaceAdd` 覆盖 tensor 原地更新；`aclnnInplaceAdds` 在测试 16 中单独调用；`aclnnAddV3` 由 `TestAddV3` 以及后续多个 V3 特化场景覆盖。整体上，6 个 API 中已经覆盖 5 个，尚未覆盖 `aclnnInplaceAddV3`，这是后续最优先补齐的入口。

第四部分是 shape 与规模覆盖。测试中包含 `[4, 2]`、`[2, 2]`、`[2]`、`[1]`、`[0]` 和 `[1000]` 等 shape。`Scalar inputs` 覆盖单元素 tensor；`Large tensor test` 覆盖 1000 元素的一维大 tensor；`Empty tensor test` 以容错方式尝试创建 `[0]` 空 tensor，如果环境不支持则输出 `[SKIP]`。这些用例可以提升 shape 推断、workspace 申请、空输入处理和 tiling 边界的覆盖概率。

第五部分是边界数值与混合精度。`Boundary values with zeros` 和 `Boundary values test` 覆盖 0、极小值 `1e-30f`、极大值 `1e30f` 及负极大值；混合精度部分包含 `FLOAT16 + FLOAT -> FLOAT`、`BF16 + FLOAT -> FLOAT`，并在 V3 场景中覆盖 FLOAT scalar + FLOAT16 tensor、FLOAT scalar + BF16 tensor 等组合。这些用例能够触发 promote 逻辑、不同输出 dtype 判断以及 FP16/BF16 的低精度路径。

**Oracle 选择**：测试代码在 CPU 端独立计算期望值，核心公式为：

```cpp
double expected = static_cast<double>(selfData[i]) +
                  static_cast<double>(alphaValue) * static_cast<double>(otherData[i]);
```

对于 FP32 场景，该方式能较好地避免单精度中间结果带来的额外误差。比较函数 `IsClose` 支持 `NaN`、同号 `Inf` 和 `atol + rtol × |expected|` 容差判断，默认 `rtol = 1e-5`、`atol = 1e-8`。对于 FP16 与 BF16 的混合精度手写用例，代码额外实现了位模式到 float 的简化转换，并分别放宽到 `1e-3` 或 `1e-2` 的容差。

**需要注意的 Oracle 风险**：模板版 `TestAdd<uint16_t>` 用于 `ACL_FLOAT16` 基础测试时，会把 FP16 位模式 `0x3C00`、`0x4000` 等直接 `static_cast<double>`，这会把位模式解释为整数 15360、16384，而不是 1.0、2.0。因此，`FLOAT16 basic test` 的 CPU 参考存在语义风险。混合精度和 V3 FLOAT16 场景中已经手写了 FP16 位模式解码逻辑，建议将这部分逻辑抽成统一的 `Fp16ToFloat`，并让所有 FP16 测试复用。

**输出规范**：每个测试用例通过 `PrintTestResult` 输出 `[PASS]` 或 `[FAIL]`，全量用例结束后输出 Total、Passed、Failed 汇总。如果通过数不等于总用例数，`RunAllTests` 返回 1；全部通过则返回 0，符合题目要求的“失败用例返回非 0 值”。

------

## 三、覆盖率分析

本次评分文件为题目 B 规定的四个源文件。当前报告依据测试代码设计进行覆盖面分析，实际覆盖率应在如下流程完成后填入：

```bash
bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example add eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
find build -name "*.gcda" | grep add
gcov -b <gcda文件路径>
```

**评分文件**

| 文件                                      | 有效行数 | 行覆盖率       | 分支覆盖率       | 说明 |
| ----------------------------------------- | -------- | -------------- | ---------------- | ---- |
| op_api/aclnn_add.cpp                      | 287      | 待实测填写     | 待实测填写       | API 层调度：参数校验、类型提升、alpha 处理、标准 API 变体分发 |
| op_api/aclnn_add_v3.cpp                   | 247      | 待实测填写     | 待实测填写       | V3 版本 API：ScalarTensor 调度、独立类型提升、alpha 分支 |
| op_api/add.cpp                            | 55       | 待实测填写     | 待实测填写       | 设备路由：AiCore / AiCpu 选择、dtype 支持判断 |
| op_host/arch35/add_tiling_arch35.cpp      | 90       | 待实测填写     | 待实测填写       | Tiling 策略：dtype 组合分发、平台信息获取 |

**综合覆盖率**：

- 行覆盖率按有效行数加权：`(aclnn_add.cpp 已覆盖行 + aclnn_add_v3.cpp 已覆盖行 + add.cpp 已覆盖行 + add_tiling_arch35.cpp 已覆盖行) / (287 + 247 + 55 + 90)`
- 分支覆盖率按 gcov 分支数加权：`四个文件已覆盖分支数之和 / 四个文件总分支数之和`

从用例设计角度看，`aclnn_add.cpp` 预计能覆盖标准 Add、Adds、InplaceAdd、InplaceAdds 的主路径，以及 alpha=1、alpha=0、alpha!=1、广播、混合 dtype 的部分分支。`aclnn_add_v3.cpp` 预计能覆盖 AddV3 的三类关键路径：alpha=1 直接 Add、FLOAT/INT32 等 Axpy 路径、BF16/INT8 等 Mul+Add 路径。`add.cpp` 能通过 FP32、INT32、INT8、FLOAT16、BF16 以及混合精度组合触发多种 dtype 支持判断和设备路由。`add_tiling_arch35.cpp` 的覆盖主要依赖 dtype 与 shape 组合：同 shape、小 tensor、大 tensor、广播和空 tensor 都已涉及，但是否实际进入特定 tiling 分支需要以 gcov 结果确认。

未覆盖部分的初步归因如下。第一，`aclnnInplaceAddV3` 尚未调用，因此 V3 原地版本相关分支不会被覆盖。第二，异常输入覆盖不足：nullptr、非法 dtype、shape 不可广播、输出 dtype 不匹配等错误路径没有系统性用例。第三，dtype 覆盖仍不完整：DOUBLE、INT64、UINT8、BOOL、COMPLEX32、COMPLEX64 等组合未覆盖，其中 DOUBLE 在 CPU 模拟器下可能路由到 AICPU，需要谨慎处理。第四，当前广播用例只覆盖 `[2,2] + [1]`，还缺少 `[1,4] + [4,4]`、`[4,1] + [1,4]`、高维广播等更容易触发 tiling 分支的 shape。

------

## 四、精度分析

精度分析章节按六类典型场景展开。由于当前未提供实测输出，本节以测试代码中构造的输入和 CPU Oracle 的理论结果为基础说明预期行为；提交前可将实际 NPU 输出和误差填入对应代码块。

### 场景一：FP32 基础加法与非整数 alpha

**测试输入**：`self = [0,1,2,3,4,5,6,7]`，`other = [1,1,1,2,2,2,3,3]`，shape 为 `[4,2]`，`alpha = 1.2f`。

**理论输出**：

```text
[1.2, 2.2, 3.2, 5.4, 6.4, 7.4, 9.6, 10.6]
```

**分析**：

该用例是最基本的 `tensor + alpha × tensor` 场景，同时 `alpha = 1.2f` 不能被二进制浮点精确表示，计算中会同时出现 alpha 量化误差和加法舍入误差。FP32 下该误差通常在 `1e-6` 量级内，默认 `rtol = 1e-5`、`atol = 1e-8` 足以覆盖正常舍入。该用例主要验证标准 `aclnnAdd` 的主路径是否正确。

相关测试用例：

```cpp
TestAdd(selfData, otherData, shape, shape, shape,
        1.2f, ACL_FLOAT, "Basic FLOAT32 same shape");
```

------

### 场景二：alpha = 0 的屏蔽路径

**测试输入**：`self = [5,6,7,8]`，`other = [1,2,3,4]`，`alpha = 0.0f`。

**理论输出**：

```text
[5, 6, 7, 8]
```

**分析**：

当 alpha 为 0 时，数学结果应完全等于 `self`，第二输入不应影响输出。该场景可以验证实现是否存在错误的融合路径、未正确处理 alpha 标量、或在 `alpha × other` 中引入不必要的 NaN/Inf 传播。需要注意的是，严格 IEEE 语义下 `0 × NaN` 仍可能得到 NaN；本测试的 other 为有限值，因此期望就是 self。

相关测试用例：

```cpp
TestAdd(selfData, otherData, shape, shape, shape,
        0.0f, ACL_FLOAT, "FLOAT32 alpha=0");
```

------

### 场景三：负 alpha 与减法语义

**测试输入**：`self = [10,20,30,40]`，`other = [1,2,3,4]`，`alpha = -1.5f`。

**理论输出**：

```text
[8.5, 17.0, 25.5, 34.0]
```

**分析**：

负 alpha 将 Add 转化为 `self - |alpha| × other`。该场景不仅覆盖负数标量解析，还能发现实现中是否错误地把 alpha 当作无符号或忽略符号位。对整数 dtype 而言，负 alpha 还会带来整数截断和溢出风险；当前负 alpha 只覆盖 FP32，后续可补 INT32 负 alpha 用例。

相关测试用例：

```cpp
TestAdd(selfData, otherData, shape, shape, shape,
        -1.5f, ACL_FLOAT, "FLOAT32 negative alpha");
```

------

### 场景四：广播与标量 shape

**测试输入**：`self = [1,2,3,4]`，shape 为 `[2,2]`；`other = [10]`，shape 为 `[1]`；`alpha = 1.0f`。

**理论输出**：

```text
[11, 12, 13, 14]
```

**分析**：

该用例验证最简单的单元素广播。测试代码中的 Oracle 通过 `otherData[i % otherData.size()]` 实现循环取值，因此能正确处理 `[1]` 广播到任意输出位置的情况。该写法对单元素广播有效，但对更复杂的多维广播（如 `[4,1]` 广播到 `[4,4]`）并不等价于通用广播规则，后续如果新增复杂广播用例，需要实现基于 shape/stride 的 CPU 参考。

相关测试用例：

```cpp
TestAdd(selfData, otherData, {2,2}, {1}, {2,2},
        1.0f, ACL_FLOAT, "FLOAT32 broadcast");
```

------

### 场景五：FP16 与 BF16 混合精度

**测试输入**：FLOAT16 位模式 `[0x3C00, 0x4000, 0x4200, 0x4400]` 约表示 `[1,2,3,4]`，FLOAT 输入 `[10,20,30,40]`，`alpha = 1.0f`；BF16 输入 `[0x3F80,0x4000,0x4040,0x4080]` 约表示 `[1,2,3,4]`，FLOAT 输入 `[5,6,7,8]`，`alpha = 2.0f`。

**理论输出**：

```text
FLOAT16 + FLOAT: [11, 22, 33, 44]
BF16 + FLOAT:    [11, 14, 17, 20]
```

**分析**：

混合精度场景的关键不是简单加法，而是输入解码、类型提升和输出 dtype。FP16 只有 10 位尾数，BF16 只有 7 位尾数，但 BF16 动态范围接近 FP32。测试代码在混合精度路径中将 FP16/BF16 位模式解码为 float 后再计算 expected，并分别采用 `1e-3`、`1e-2` 容差，整体策略合理。需要改进的是：FP16 解码逻辑目前是简化实现，未完整处理所有 NaN、Inf、次正规数和舍入模式；BF16 转换使用 reinterpret 方式，在常见平台可工作，但最好改成 `std::memcpy` 避免严格别名问题。

相关测试用例：

```cpp
// FLOAT16 + FLOAT -> FLOAT
PrintTestResult(passed, "Mixed precision FLOAT16+FLOAT");

// BF16 + FLOAT -> FLOAT
PrintTestResult(passed, "Mixed precision BF16+FLOAT");
```

------

### 场景六：整数类型与溢出风险

**测试输入**：INT32 基础测试中 `self = [1,2,3,4]`，`other = [5,6,7,8]`，`alpha = 2`；INT8 基础测试中 `self = [10,20,30,40]`，`other = [1,2,3,4]`，`alpha = 2`。

**理论输出**：

```text
INT32: [11, 14, 17, 20]
INT8:  [12, 24, 36, 48]
```

**分析**：

这两组输入均未触发溢出，适合作为整数路径的功能正确性验证。真正的风险在于 `self + alpha × other` 超出 dtype 表示范围时，算子通常不会报错。测试代码在部分 V3 INT32/INT8 用例中检查了 expected 是否超出目标范围，若超出则跳过该点验证。这避免了 Oracle 因 C++ 有符号整数溢出产生未定义行为，但也意味着“溢出行为本身”尚未被验证。若需要专门测试溢出，应使用无符号中间值或更宽整数显式模拟低位截断，例如 INT8 使用 `uint8_t` 回绕后再转回 `int8_t`。

相关测试用例：

```cpp
TestAdd(selfData, otherData, shape, shape, shape,
        2, ACL_INT32, "INT32 basic test");

TestAdd(selfData, otherData, shape, shape, shape,
        static_cast<int8_t>(2), ACL_INT8, "INT8 basic test");
```

------

## 五、反思与改进

**API 覆盖仍缺一个关键入口**。当前代码已经覆盖 `Add`、`Adds`、`InplaceAdd`、`InplaceAdds` 和 `AddV3`，但没有覆盖 `InplaceAddV3`。由于 V3 文件有独立实现，且原地写回通常会包含额外的输出检查、引用合法性校验和 workspace 路径，建议优先补充 `aclnnInplaceAddV3GetWorkspaceSize` 与 `aclnnInplaceAddV3` 的基础 FP32 用例，再扩展 alpha=1 和 alpha!=1 两种路径。

**复杂广播 Oracle 需要升级**。当前 `i % data.size()` 的参考实现可以覆盖同 shape、标量和单元素广播，但不能泛化到多维广播。例如 `[4,1] + [1,4] -> [4,4]` 中，两个输入的索引映射不能用简单取模表示。后续若想提升 host tiling 覆盖率，必须补充复杂广播；同时应实现一个根据输入 shape、输出 shape 和 stride 计算源索引的通用 CPU Oracle。

**FP16 基础测试的参考实现需要修正**。模板版 `TestAdd<uint16_t>` 会把 FP16 位模式直接当整数参与 expected 计算，这会导致基础 FP16 用例的 Oracle 不可信。建议新增统一函数 `Fp16ToFloat(uint16_t)`、`Bf16ToFloat(uint16_t)`，并为 FP16/BF16 的同类型和混合类型都走解码后的 float/double 参考。这样既能避免误判，也能让精度分析章节中的误差数据更有说服力。

**异常路径不足**。题目要求包含 nullptr、不支持 dtype 等异常输入。当前代码只有空 tensor 与部分“API 不支持则 SKIP”的容错分支，没有系统地验证 `nullptr self`、`nullptr other`、`nullptr alpha`、`nullptr out`、shape 不可广播、输出 dtype 不匹配等错误返回。异常用例不需要执行 kernel，只需调用 GetWorkspaceSize 并断言返回非成功即可，通常能显著提升 API 层参数校验分支覆盖率。

**skip 计为 pass 会影响结果解释**。当前代码中多个可选路径在不支持时输出 `[SKIP]`，同时增加 `g_total_tests` 和 `g_passed_tests`。这种做法有利于最终程序返回 0，但从测试报告角度会掩盖实际未覆盖路径。建议将统计拆成 Passed、Failed、Skipped 三类，程序返回值仍只由 Failed 决定；报告中单独说明 skipped 用例未贡献覆盖率。

**stream 参数没有贯穿到算子调用**。初始化阶段创建了 `aclrtStream stream`，但各个封装函数内部调用 `aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnAddV3` 时传入的是 `nullptr` stream。若评测环境允许默认 stream，这可能仍可运行；但为了与官方示例和异步执行语义一致，建议将 stream 作为参数传入所有测试封装函数，并在 kernel 调用后执行必要的同步。

**dtype 覆盖可以继续扩展**。当前重点覆盖 FP32、FP16、BF16、INT32、INT8 和部分混合精度，但题目列出的 UINT8、INT64、BOOL、COMPLEX32、COMPLEX64、DOUBLE 等尚未覆盖。考虑到 DOUBLE 在 CPU 模拟器下可能路由到 AICPU，建议优先补 UINT8、INT64、BOOL，再根据环境决定是否纳入 DOUBLE。复数类型可以选择简单输入，如 `(1+2i) + alpha × (3+4i)`，分别验证实部和虚部。

**优先级排序**。若继续优化，建议按如下顺序推进：① 补 `InplaceAddV3`，覆盖最后一个 API 入口；② 修正 FP16/BF16 Oracle；③ 增加复杂广播并实现通用广播参考；④ 增加 nullptr、非法 dtype、不可广播 shape 等异常用例；⑤ 扩展 UINT8、INT64、BOOL、COMPLEX 类型；⑥ 重新执行覆盖率流程，用 gcov 数据替换本报告中的“待实测填写”。预计这些改动会显著提升 `aclnn_add.cpp`、`aclnn_add_v3.cpp` 和 `add_tiling_arch35.cpp` 的行覆盖率与分支覆盖率。

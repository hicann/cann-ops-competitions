# 题目A Mul 算子测试报告

## 1 测试概述

### 1.1 测试背景

Mul 算子是 CANN ops-math 仓库中的基础数学算子，用于实现逐元素乘法计算，其基本形式为：

[
y = x_1 \times x_2
]

当两个输入 Tensor 的 shape 不一致时，Mul 算子需要按照广播规则进行对齐后再完成逐元素乘法运算。该算子处于基础数学算子层，在上层框架调用、底层 NPU 执行以及混合类型计算中均具有较高的使用频率，因此其功能正确性、异常处理能力与接口稳定性都具有较强的工程意义。现有材料表明，该算子采用三层架构：op_api 层负责参数校验、类型提升和路由调度，op_host 层负责 shape 推断和 tiling 计算，op_kernel 层负责最终在 NPU 侧执行逐元素乘法。

### 1.2 测试目标

本次测试围绕 Mul 算子的端到端调用流程展开，目标是对以下 4 个 API 变体进行功能和健壮性验证：

- `aclnnMul`
- `aclnnMuls`
- `aclnnInplaceMul`
- `aclnnInplaceMuls`

测试既要验证正常输入下的计算结果是否正确，也要验证异常输入下接口是否能够正确返回失败状态。同时，结合 gcov 覆盖率统计结果，对当前测试集的质量进行评估，以识别尚未充分覆盖的逻辑分支。现有测试报告明确给出了这 4 类 API 的测试目标、场景划分和覆盖率统计。

### 1.3 测试脚本总体功能

从源码实现看，本测试脚本并非单纯调用接口，而是形成了一套较完整的测试执行框架。其核心功能包括：

1. 在 CPU 侧构造期望值并与 NPU 实际结果进行对比；
2. 覆盖成功路径，包括不同 dtype、不同 shape、广播、混合类型、原地修改等典型场景；
3. 覆盖异常路径，包括非法广播、输出 shape 不匹配、空指针输入等错误场景；
4. 统一输出 `[PASS] / [FAIL]` 日志，并通过全局变量 `g_pass` 与 `g_fail` 汇总测试结果；
5. 在测试结束后打印总用例数、通过数和失败数。源码中的 `Record`、`Init`、`Done` 和 `main` 主流程体现了这一完整结构。

------

## 2 测试需求分析

### 2.1 功能正确性需求

Mul 算子最基本的需求是对两个输入 Tensor 按元素进行乘法计算，并在输出 Tensor 中给出正确结果。对于 `aclnnMul` 而言，需要验证其是否能够在相同 shape 和可广播 shape 下正确产生输出；对于 `aclnnMuls`，则需要验证 Tensor 与标量相乘结果是否正确；对于 `aclnnInplaceMul` 和 `aclnnInplaceMuls`，则不仅要验证结果正确，还要验证原地写回后的输入 Tensor 是否被正确更新。源码中分别通过 `RunMul`、`RunMuls`、`RunInplaceMul` 和 `RunInplaceMuls` 完成这四类需求验证。

### 2.2 广播与 shape 兼容性需求

Mul 属于典型的逐元素算子，广播逻辑是其核心兼容性需求之一。测试必须验证：

- 同 shape 输入下的常规计算；
- 行广播、跨维广播等合法广播场景；
- 空 Tensor 场景；
- 非法广播输入场景；
- 原地接口下输出 shape 是否必须与 self shape 一致。

源码中定义了 `BroadcastShape`、`BcOffset`、`Unravel` 等辅助函数，在 CPU 侧完整模拟广播索引计算逻辑，同时在测试用例中显式加入 `TestMul_Fp32_BroadcastRow_Success`、`TestMul_Fp32_BroadcastCross_Success`、`TestMul_Fp32_EmptyTensor_Success`、`TestMul_InvalidBroadcastShape_FailExpected` 和 `TestInplaceMul_InvalidBroadcast_FailExpected` 等典型场景，说明 shape 兼容性是本次测试的重要需求之一。

### 2.3 数据类型兼容性需求

根据现有报告和源码实现，Mul 算子需要覆盖多种数据类型场景，包括 FLOAT32、FLOAT16、BF16、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BOOL，以及混合精度组合。源码中的 `DTypeStr`、`ToDouble`、`F16ToF32`、`F32ToF16`、`BF16ToF32` 和 `F32ToBF16` 等函数，正是为了支撑这些类型在测试中的构造、转换与比较。与此同时，源码中还额外加入了 `TestMul_Complex64_StatusOnly_Success` 与 `TestMul_Complex32_FailExpected`，说明本次测试不仅关注常规支持类型，也对复杂类型兼容性边界进行了状态级验证。

### 2.4 异常处理需求

一个高质量的算子测试不仅要验证“能算对”，还要验证“错输入能拦住”。因此，本次测试需要覆盖以下异常处理需求：

- 输入 Tensor 为空指针；
- 输出 shape 非法；
- 输入 shape 不可广播；
- 原地接口的广播结果与 self shape 不一致；
- 不支持的类型组合。

源码中分别通过 `RunMulExpectFail`、`RunMulsExpectFail`、`RunInplaceMulExpectFail` 和 `RunInplaceMulsExpectFail` 四类函数来验证这些失败路径，说明测试需求中已经明确包含了错误输入场景。

### 2.5 测试质量评估需求

除了功能验证，本次测试还需要从覆盖率角度评估测试质量。现有测试报告给出了 `op_api/aclnn_mul.cpp`、`op_api/mul.cpp` 和 `op_host/arch35/mul_tiling_arch35.cpp` 三个评分文件的行覆盖率、分支覆盖率、Taken at least once 和 Calls executed 等统计信息，这说明本次测试的需求中还包含了对测试集充分性的定量评价。

------

## 3 测试环境与测试方法

### 3.1 测试环境

从源码看，测试依赖 ACL 运行时环境和 Mul 算子头文件：

- `acl/acl.h`
- `aclnnop/aclnn_mul.h`

程序在 `main` 中调用 `Init(deviceId, &stream)` 完成 ACL 初始化、设备设置和流创建；测试结束后通过 `Done(deviceId, stream)` 完成流销毁、设备复位和 ACL 反初始化。因此可以确认该测试是在 ACL Runtime + NPU Stream 模式下执行的。

### 3.2 测试方法

本次测试采用“CPU 参考结果 + NPU 实际输出 + 误差比较”的方式进行验证，其基本流程如下：

1. 根据输入 shape 与数据构造输入 Tensor；
2. 对 Mul 类接口，先在 CPU 侧通过 `BuildMulExpected` 或 `BuildMulExpectedMixed` 计算期望结果；
3. 调用相应的 `aclnn*GetWorkspaceSize` 接口获取工作空间；
4. 分配 workspace 后执行目标算子；
5. 将 NPU 输出拷回主机端；
6. 根据 dtype 选择容差后，使用 `CompareVec` 比较期望值与实际值；
7. 输出测试通过或失败信息。

其中，`Tolerance` 函数定义了不同数据类型的误差容限：FLOAT16/BF16 使用 `1e-3`，FLOAT32 使用 `1e-5` 和 `1e-4`，DOUBLE 使用 `1e-12`，整数与布尔类型则要求精确匹配。`CompareVec` 还特别处理了 NaN 与 Inf 的比较逻辑，保证特殊值场景下比较行为正确。

### 3.3 资源管理方法

源码通过 `TensorHolder`、`WsHolder` 和 `ScalarHolder` 三个 RAII 风格结构体自动释放 Tensor、workspace 和 Scalar 资源，避免测试过程中出现内存泄漏。这使得测试脚本在工程质量上优于简单的过程式测试代码，也保证了多用例串行执行时资源状态相对稳定。

------

## 4 测试设计与生成思路

### 4.1 总体设计思路

本次测试采用“按 API 分类、按场景展开、按边界补强”的设计思路。首先，以接口语义为主线，将测试对象拆分为普通 Tensor 乘法、Tensor 乘标量、原地 Tensor 乘法和原地标量乘法四大类；其次，在每一类接口下进一步从数据类型、shape 组合、广播关系、数值边界和异常输入等维度组织测试；最后，再结合 gcov 结果对尚未充分触达的逻辑进行质量分析。现有测试报告中的“数据类型覆盖策略”“shape 组合覆盖策略”“数值边界覆盖策略”和“API 变体覆盖策略”与源码主流程是一致的。

### 4.2 数据类型生成思路

数据类型覆盖遵循“主流类型优先、低精度类型补充、混合类型重点验证、边缘类型状态检查”的原则。源码中的具体体现包括：

- 浮点类型：FLOAT32、FLOAT16、BF16、DOUBLE；
- 整数类型：INT8、UINT8、INT16、INT32、INT64；
- 布尔类型：BOOL；
- 混合类型：FLOAT16×FLOAT32、FLOAT32×FLOAT16、BF16×FLOAT32、FLOAT32×BF16；
- 边缘兼容性：COMPLEX64 状态成功、COMPLEX32 预期失败。

这样的生成思路兼顾了实际使用频率和接口边界，尤其是对混合精度路径的验证，能够较好检验 Mul 算子的类型提升与输出兼容性逻辑。

### 4.3 shape 与广播场景生成思路

shape 维度的用例设计遵循“从简单到复杂、从合法到非法”的原则。源码中包含：

- 常规同 shape 场景，如 `[2,3] × [2,3]`；
- 行广播场景，如 `[4,3] × [3]`；
- 跨维广播场景，如 `[3,1] × [1,4]`；
- 高维连续场景，如 `[1,1,1,2,3]`；
- 空 Tensor 场景，如 `[0,4]`；
- 大规模 Tensor 场景，如 `[1024,1024]`；
- 非法广播场景，如 `[2,3] × [4,5]`；
- 输出 shape 不匹配场景。

这类设计能够较系统地验证 Mul 算子的 shape 推断和广播逻辑，覆盖从常规业务输入到边缘场景的主要风险点。

### 4.4 数值边界生成思路

数值边界主要关注特殊数值在计算链路中的传播行为和比较逻辑，包括：

- 正负数乘法；
- 零值乘法；
- NaN 传播；
- Inf 传播；
- 大规模输入下数值稳定性。

其中，源码显式加入了 `TestMul_NaN_Success` 和 `TestMul_Inf_Success`，并在 `CompareVec` 中对 NaN 和 Inf 做了专门处理，因此这部分不是泛泛而谈，而是有明确代码实现支撑。

### 4.5 异常路径生成思路

异常路径设计采用“构造非法输入 + 预期接口失败”的方式。具体包括：

- `nullSelf = true` 模拟空指针输入；
- 非法广播 shape；
- 输出 shape 不匹配；
- 不支持的 Complex32 类型测试；
- 原地广播非法场景。

源码中的异常测试函数不会继续做结果比较，而是通过判断 `GetWorkspaceSize` 或实际执行阶段是否返回失败状态来判断测试是否通过，这与异常路径的验证目标是匹配的。

------

## 5 测试用例设计与执行情况

### 5.1 用例总体统计

根据现有测试报告统计与源码主函数比对，本次测试共设计 **39 个** 用例，分别覆盖成功路径与异常路径。报告中给出的用例分布如下：

- `RunMul`：18 个
- `RunMulMixed`：4 个
- `RunMulStatusOnly`：1 个
- `RunMuls`：3 个
- `RunInplaceMul`：1 个
- `RunInplaceMuls`：3 个
- `RunMulExpectFail`：4 个
- `RunMulsExpectFail`：2 个
- `RunInplaceMulExpectFail`：2 个
- `RunInplaceMulsExpectFail`：1 个

上述统计与源码 `main` 中的实际调用数量是一致的。

### 5.2 成功路径用例分析

成功路径主要覆盖以下几类场景：

#### （1）常规浮点与整数场景

如 `TestMul_Fp32_Basic_Success`、`TestMul_Int32_Basic_Success`、`TestMul_Int64_Basic_Success`、`TestMul_Int16_Basic_Success`、`TestMul_Int8_Basic_Success`、`TestMul_Uint8_Basic_Success` 和 `TestMul_Double_Basic_Success`，用于验证算子在常规同 shape 场景下的正确性。

#### （2）广播与高维场景

如 `TestMul_Fp32_BroadcastRow_Success`、`TestMul_Fp32_BroadcastCross_Success`、`TestMul_Fp32_Dim5_ContiguousFallback_Success`，用于验证广播与高维 shape 推断逻辑。

#### （3）特殊输入规模场景

如 `TestMul_Fp32_EmptyTensor_Success` 与 `TestMul_Fp32_Large1024x1024_Success`，分别覆盖空 Tensor 与大规模 Tensor 场景。

#### （4）低精度与混合精度场景

如 `TestMul_Fp16_Basic_Success`、`TestMul_Bf16_Basic_Success`、`TestMul_Mix_Fp16xFp32_Success`、`TestMul_Mix_Fp32xFp16_Success`、`TestMul_Mix_Bf16xFp32_Success` 和 `TestMul_Mix_Fp32xBf16_Success`，重点验证低精度输入和混合类型组合的行为。

#### （5）边界数值场景

如 `TestMul_NaN_Success` 和 `TestMul_Inf_Success`，用于验证 NaN 和无穷大在乘法运算中的传播情况。

#### （6）标量与原地场景

如 `TestMuls_Fp32_Success`、`TestMuls_Fp16_Success`、`TestMuls_Bf16_Success`、`TestInplaceMul_Fp32_Broadcast_Success`、`TestInplaceMuls_Fp32_Success`、`TestInplaceMuls_Fp16_Success` 和 `TestInplaceMuls_Bf16_Success`，用于验证标量计算与原地更新逻辑。

### 5.3 异常路径用例分析

异常路径主要包括：

- `TestMul_InvalidBroadcastShape_FailExpected`
- `TestMul_InvalidOutShape_FailExpected`
- `TestMul_NullInput_FailExpected`
- `TestMul_Complex32_FailExpected`
- `TestMuls_InvalidOutShape_FailExpected`
- `TestMuls_NullInput_FailExpected`
- `TestInplaceMul_InvalidBroadcast_FailExpected`
- `TestInplaceMul_NullInput_FailExpected`
- `TestInplaceMuls_NullInput_FailExpected`

这些用例覆盖了 Mul 系列接口中较典型的失败路径，对于验证接口的参数合法性校验逻辑具有较强针对性。

### 5.4 执行结果说明

源码在测试结束时会打印：

```cpp
Total=%d Passed=%d Failed=%d
```

因此脚本具备完整的测试结果统计能力。需要说明的是，当前用户提供的是源码与已有报告文本，而不是一次真实运行后的日志输出，因此本文能够确认**测试程序设计了总数统计和通过/失败统计机制**，但无法凭空给出某次实际执行时的具体 Passed/Failed 数值。为保证表述严谨，本文不虚构运行日志，而是基于源码结构和已有报告对测试设计与质量进行分析。

------

## 6 覆盖率结果与质量分析

### 6.1 覆盖率统计结果

现有测试报告给出的 gcov 统计结果如下：

| 文件                                   | 行覆盖率      | 分支覆盖率     | 至少命中一次   | 调用覆盖率    |
| -------------------------------------- | ------------- | -------------- | -------------- | ------------- |
| `op_api/aclnn_mul.cpp`                 | 78.96% of 328 | 38.49% of 1920 | 21.30% of 1920 | 43.21% of 803 |
| `op_api/mul.cpp`                       | 71.15% of 52  | 48.15% of 135  | 33.33% of 135  | 52.94% of 85  |
| `op_host/arch35/mul_tiling_arch35.cpp` | 89.22% of 102 | 58.33% of 120  | 29.17% of 120  | 37.25% of 102 |

该数据说明本次测试对主流程代码已有较好的触达，尤其是 `mul_tiling_arch35.cpp` 的行覆盖率较高，表明主要的 tiling 路径已被较充分地运行到。

### 6.2 质量优点分析

#### （1）接口覆盖完整

4 个 API 变体均有独立测试函数和对应用例，说明测试覆盖面完整。

#### （2）场景设计系统

测试不仅覆盖常规场景，还覆盖广播、空 Tensor、大 Tensor、混合精度、原地更新和异常路径，说明测试设计具有系统性。

#### （3）结果验证方法规范

通过 CPU 参考值构造、类型转换、按 dtype 设置容差和特殊值单独比较的方式，保证了测试结论的可信度。

#### （4）工程实现较规范

源码使用 RAII 风格资源管理，测试函数职责划分清晰，日志输出和结果统计统一，具有较好的可维护性和可扩展性。

### 6.3 存在问题分析

#### （1）分支覆盖率仍偏低

虽然行覆盖率较高，但 `aclnn_mul.cpp` 分支覆盖率仅 38.49%，说明仍有相当数量的条件分支未被触发，尤其可能集中在复杂异常处理和稀有类型组合路径。

#### （2）设备路由验证深度不足

`mul.cpp` 的分支覆盖率为 48.15%，现有报告也指出特殊设备路由分支尚未完全覆盖，说明 AiCore/AiCpu 相关路径仍有补充空间。

#### （3）部分边缘类型和特殊策略验证不足

虽然已包含 Complex64 状态测试和 Complex32 失败测试，但对更多复杂类型和极端 dtype 组合的验证仍较有限；同时，tiling 的一些特殊 shape 组合路径尚未完全覆盖。

#### （4）缺少性能维度验证

当前测试重点在功能正确性和接口健壮性，没有进一步统计不同规模输入下的性能指标，因此尚不能支持对算子效率的综合评价。

------

## 7 复现方式

### 7.1 gcda 路径

现有报告中给出了本次覆盖率统计所使用的 gcda 路径：

1. `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/aclnn_mul.cpp.gcda`
2. `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/mul.cpp.gcda`
3. `build/math/mul/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/mul_tiling_arch35.cpp.gcda`

### 7.2 复现命令

```bash
cd /home/cann/workspace/ops-math

find build -name "*.gcda" | grep -E "aclnn_mul\.cpp|/mul\.cpp|mul_tiling_arch35\.cpp"

gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/aclnn_mul.cpp.gcda
gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/mul.cpp.gcda
gcov -b build/math/mul/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/mul_tiling_arch35.cpp.gcda
```

上述命令可用于复现覆盖率分析结果。

------

## 8 测试结论

本次 Mul 算子测试围绕 `aclnnMul`、`aclnnMuls`、`aclnnInplaceMul` 和 `aclnnInplaceMuls` 四类 API 展开，测试设计覆盖了常规计算、广播计算、低精度与混合精度计算、原地更新、空 Tensor、大规模 Tensor、NaN/Inf 特殊值以及多类异常输入场景。源码实现表明，测试脚本已具备完整的参考值构造、误差比较、异常判断、资源管理与结果汇总机制；现有报告则进一步表明，本次测试共设计 39 个用例，并对 Mul 算子相关核心实现文件形成了较好的行覆盖率。

总体来看，本次测试能够较好支撑对 Mul 算子功能正确性和基础健壮性的验证，测试质量处于较好水平。其优点在于接口覆盖完整、场景设计系统、结果校验规范、工程实现较严谨；不足在于分支覆盖率仍偏低，特殊设备路由、边缘 dtype 组合和部分 tiling 分支仍有补充空间。后续可围绕异常注入、AiCpu 路由、更多边界类型组合及性能基准测试进一步完善测试集，从而持续提升测试完备性与报告说服力。


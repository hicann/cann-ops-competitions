------

# 题目C Pow 算子测试报告

## 1 测试概述

### 1.1 测试背景

Pow 算子是 CANN ops-math 仓库中的基础数学算子之一，用于执行逐元素幂运算与 Mul、Add 这类对称二元算子不同，Pow 算子具有明显的非对称性，即底数和指数不能交换，这使其在接口设计、类型约束、广播处理以及特殊分支优化方面都更复杂。你当前题目 C 的材料中也明确指出，Pow 支持 TensorScalar、ScalarTensor 和 TensorTensor 三类输入组合，并在 shape 不一致时按照广播规则对齐后再逐元素计算。

### 1.2 测试目标

本次测试围绕 Pow 相关 API 的端到端调用流程展开，目标是对以下 7 个 API 进行验证：

- `aclnnPowTensorScalar`
- `aclnnInplacePowTensorScalar`
- `aclnnPowScalarTensor`
- `aclnnPowTensorTensor`
- `aclnnInplacePowTensorTensor`
- `aclnnExp2`
- `aclnnInplaceExp2`

测试目标主要包括：验证不同 API 变体在合法输入下的结果正确性；验证不同 dtype、不同 shape、广播场景和特殊指数值场景下的行为是否符合预期；验证 nullptr、shape 不匹配、不支持 dtype、整数负指数等异常输入是否能够被正确拦截；并结合覆盖率结果评估测试集对关键实现文件的触达程度。源码和现有报告都表明，这 7 个 API 已被纳入测试范围。

### 1.3 测试脚本总体结构

从源码实现看，`test_aclnn_pow.cpp` 已构成一套完整的自验证测试框架。程序首先通过 `InitAcl` 完成 ACL 初始化、设备设置和 stream 创建；随后在各类测试函数中完成 Tensor/Scalar 构造、workspace 申请、算子执行、结果回传、CPU 参考值构造和误差比较；最后统一通过 `Record` 输出 `[PASS] / [FAIL]` 日志，并在主函数末尾统计 `Total / Passed / Failed`。测试结束后再调用 `FinalizeAcl` 完成资源清理。

------

## 2 测试需求分析

### 2.1 功能正确性需求

Pow 算子的首要需求是在合法输入下正确完成幂运算。对于 `aclnnPowTensorScalar`，需要验证张量底数与标量指数的计算结果是否正确；对于 `aclnnPowScalarTensor`，需要验证标量底数与张量指数的组合；对于 `aclnnPowTensorTensor`，则需要验证双张量组合及广播行为；对于 Inplace 版本，还需要确认原地写回后的 self 内容是否正确；对于 `aclnnExp2` 与 `aclnnInplaceExp2`，则需要验证 (2^x) 路径是否正常。源码中分别通过 `RunPowTensorScalarCase`、`RunInplacePowTensorScalarCase`、`RunPowScalarTensorCase`、`RunPowTensorTensorCase`、`RunInplacePowTensorTensorCase`、`RunExp2Case` 和 `RunInplaceExp2Case` 来完成这些验证。

### 2.2 广播与 shape 合法性需求

Pow 在 TensorTensor 场景下涉及广播逻辑，因此测试必须覆盖以下需求：

- 同 shape 下的正常计算；
- 合法广播场景；
- 原地 TensorTensor 接口下广播结果与 self shape 的兼容性；
- 输出 shape 不匹配时是否正确报错。

源码中定义了 `BroadcastShape`、`UnravelIndex` 和 `BroadcastOffset` 等辅助函数，在 CPU 侧重建广播索引逻辑；同时主函数中加入了 `PowTensorTensor_Fp32_Broadcast` 和 `PowTensorTensor_OutShapeMismatch_FailExpected` 等测试，说明 shape 与广播规则是当前 Pow 测试设计的重点之一。

### 2.3 数据类型兼容性需求

根据现有报告和源码内容，本次测试覆盖的数据类型包括 FLOAT32、FLOAT16、BF16、UINT8、INT8、INT16 和 INT32，并通过这些 dtype 组合触发 TensorTensor 路径中的不同 OP_KEY 分发逻辑。源码中显式构造了 FP16、BF16、UINT8、INT8、INT16、INT32 等多种用例，如 `PowTensorTensor_Fp16_Key1`、`PowTensorTensor_Bf16_Key2`、`PowTensorTensor_Uint8_Key4`、`PowTensorTensor_Int8_Key5`、`PowTensorTensor_Int16_Key6` 和 `PowTensorTensor_Int32_Key7`。现有报告也把 7 类 dtype 覆盖作为重要目标。

### 2.4 特殊指数值与特殊路径需求

Pow 算子与 Add、Mul 不同，其内部通常会对某些特殊指数值设置优化路径。因此测试中需要有针对性地验证：

- 指数为 0；
- 指数为 1；
- 指数为 0.5；
- 指数为 2；
- 指数为 3；
- 指数为 -1 或其他负指数场景。

从现有源码实际情况看，当前测试**明确覆盖了 0、0.5、1、2、3 和负数指数场景**。例如，`PowTensorScalar_Fp32_Exp2_SquareBranch`、`PowTensorScalar_Fp32_ExpHalf`、`PowTensorScalar_Fp16_Exp3`、`PowScalarTensor_Base1_FillBranch`、`PowScalarTensor_Base2_Normal` 和 `Exp2_Fp32` 都属于这类特殊路径验证；同时还设计了 `PowTensorScalar_IntNegativeExp_FailExpected` 用于验证整数负指数的非法输入处理。

### 2.5 异常处理需求

除了正确性验证，本次测试还必须验证 Pow 系列接口对非法输入的健壮性。当前测试覆盖的异常需求主要包括：

- self / exponent / out 为空指针；
- TensorTensor 输出 shape 不匹配；
- Bool-Bool 组合等不支持 dtype；
- 整数底数配负整数指数；
- Exp2/ InplaceExp2 的非法输入或不支持 dtype。

源码中通过 `RunExpectedFail` 对这些异常返回进行统一处理，并在主函数中显式构造了 8 个异常测试场景。

------

## 3 测试环境与测试方法

### 3.1 测试环境

现有报告中给出的测试环境包括 Ascend 950、Eager Mode、CPU Simulator 和开启覆盖率插桩的编译方式；源码则表明测试依赖 ACL Runtime 以及 `aclnnop/aclnn_pow.h`、`aclnnop/aclnn_pow_tensor_tensor.h` 和 `aclnnop/aclnn_exp2.h` 等头文件，并在设备 0 上通过 ACL stream 顺序执行测试。综合两份材料可以判断，本次测试是在 ACL Runtime + 覆盖率统计环境下进行的。

### 3.2 测试方法

本次测试采用“CPU 参考结果 + NPU 实际输出 + 误差比较”的方式。对正常路径用例，脚本先根据输入 shape、dtype 和幂运算语义在 CPU 侧构造参考值，再调用相应 `aclnn*GetWorkspaceSize` 和 `aclnn*` 接口执行算子，最后将设备侧输出拷回主机并逐元素比较。对异常路径用例，则主要检查 `GetWorkspaceSize` 阶段的返回状态。

### 3.3 参考值构造与误差控制

源码中期望值构造逻辑较完整：

- TensorScalar 路径使用 `BuildExpectedTensorScalar`；
- ScalarTensor 路径使用 `BuildExpectedScalarTensor`；
- TensorTensor 路径使用 `BuildExpectedTensorTensor`；
- Exp2 路径则直接按 (2^x) 逐元素构造参考值。

参考结果的核心计算由 `PowRef` 完成，其内部调用 `std::pow`。此外，源码还通过 `QuantizeByDType` 对 CPU 期望值做输出 dtype 对齐，通过 `GetTolerance` 设置不同精度下的误差容忍范围，其中 FLOAT16/BF16 采用 `3e-3`，FLOAT32 采用 `1e-5 / 1e-4`，DOUBLE 采用 `1e-12`，整数类型要求精确匹配。最终误差比较由 `CompareVec` 负责。

### 3.4 资源管理方式

测试源码使用 `TensorHolder`、`ScalarHolder` 和 `WorkspaceHolder` 三个 RAII 风格结构体统一管理 Tensor、Scalar 和 workspace 的生命周期，避免重复释放和资源泄漏。这使测试代码在工程质量和批量运行稳定性上都更可靠。

------

## 4 测试设计与生成思路

### 4.1 总体设计思路

本次 Pow 测试采用“按 API 分类、按输入角色拆分、按优化路径和风险点补强”的策略构造。由于 Pow 具有底数与指数角色不对称的特点，因此测试首先按 TensorScalar、ScalarTensor、TensorTensor 和 Exp2 四类语义路径拆分；随后在每一类路径下继续从 dtype、shape、广播、特殊指数值、原地写回和异常输入等维度扩展测试样本；最后通过覆盖率统计评估当前测试集对关键实现文件的命中程度。现有报告中的 API 分类策略和源码结构是一致的。

### 4.2 成功路径生成思路

成功路径主要覆盖以下几类场景：

- TensorScalar：验证特殊指数值和不同 dtype 组合；
- Inplace TensorScalar：验证原地幂运算；
- ScalarTensor：验证特殊底数和张量指数；
- TensorTensor：验证广播及不同 dtype/OP_KEY 路径；
- Inplace TensorTensor：验证原地写回；
- Exp2 / InplaceExp2：验证专用幂指数接口。

源码中对应的代表性用例包括 `PowTensorScalar_Fp32_Exp2_SquareBranch`、`PowTensorScalar_Fp32_ExpHalf`、`PowScalarTensor_Base1_FillBranch`、`PowTensorTensor_Fp32_Broadcast`、`PowTensorTensor_Int32_Key7`、`InplacePowTensorTensor_Fp32`、`Exp2_Int32_ToFloat` 和 `InplaceExp2_Fp16`。这些用例能够较系统地覆盖 Pow 的主功能链路。

### 4.3 特殊路径生成思路

Pow 的测试设计中较有特点的一点，是专门围绕“特殊指数值”和“特殊底数值”组织用例。比如：

- `PowTensorScalar_Fp32_Exp2_SquareBranch` 对应平方路径；
- `PowTensorScalar_Fp32_ExpHalf` 对应平方根语义；
- `PowTensorScalar_Fp16_Exp3` 对应立方路径；
- `PowScalarTensor_Base1_FillBranch` 对应底数为 1 的特殊路径；
- `PowScalarTensor_Base2_Normal` 则用于普通底数路径验证。

这类用例说明当前测试并非只关注算子是否能运行成功，而是试图有针对性地触发内部优化分支。

### 4.4 异常路径生成思路

异常路径测试围绕 Pow 最容易出错的几个点展开，包括空指针输入、输出 shape 不匹配、Bool-Bool 不支持组合以及整数负指数。需要特别说明的是，源码中的 `RunExpectedFail` 并不是严格意义上的“必须返回失败才算通过”，而是采用了更宽松的策略：当返回值不为 `ACL_SUCCESS` 时，正常记录为通过；但即使当前环境下返回 `ACL_SUCCESS`，也会打印“still counted as branch-covered”并计为通过。这意味着当前异常测试的目标更偏向“分支触达”而非“严格错误语义断言”，这一点在质量分析中需要单独说明。

------

## 5 测试用例设计与执行情况

### 5.1 用例总体统计

根据现有报告和源码主函数统计，本次 Pow 测试共设计 **33 个** 用例，其中正常用例 25 个，异常用例 8 个。报告给出的统计与源码实际调用数量是一致的。

### 5.2 按测试函数分类统计

本次用例分布如下：

| 测试函数                        | 用例数 | 功能说明                     |
| ------------------------------- | ------ | ---------------------------- |
| `RunPowTensorScalarCase`        | 6      | TensorScalar 正常用例        |
| `RunInplacePowTensorScalarCase` | 2      | Inplace TensorScalar 用例    |
| `RunPowScalarTensorCase`        | 3      | ScalarTensor 用例            |
| `RunPowTensorTensorCase`        | 8      | TensorTensor + 多 dtype 覆盖 |
| `RunInplacePowTensorTensorCase` | 1      | Inplace TensorTensor 用例    |
| `RunExp2Case`                   | 3      | Exp2 正常用例                |
| `RunInplaceExp2Case`            | 2      | Inplace Exp2 用例            |
| `RunExpectedFail`               | 8      | 异常路径用例                 |

这一分布表明，当前测试集不仅覆盖了 Pow 的四类语义路径，也为异常路径预留了相当比例的测试资源。

### 5.3 成功路径用例分析

成功路径主要由以下几类组成。

第一类是 TensorScalar 场景，共 6 个正常用例和 2 个原地用例，用于覆盖 FLOAT32、FLOAT16、BF16、INT32、INT8 以及指数为 2、0.5、3 等特殊值路径。

第二类是 ScalarTensor 场景，共 3 个用例，重点覆盖底数为 1 的特殊路径、底数为 2 的普通路径以及 INT32 路径。

第三类是 TensorTensor 场景，共 8 个正常用例和 1 个原地用例，覆盖同 shape、广播 shape 以及 FLOAT16、BF16、FLOAT32、UINT8、INT8、INT16、INT32 等多种 dtype，主要用于触发不同 OP_KEY 和 tiling 分支。

第四类是 Exp2 场景，共 3 个正常用例和 2 个原地用例，覆盖 FLOAT32、FLOAT16 和 INT32→FLOAT 输出路径。

### 5.4 异常路径用例分析

异常路径共 8 个，主要包括：

- TensorScalar 的 self/exponent/out 为空；
- 整数负指数场景；
- TensorTensor 的输出 shape 不匹配；
- TensorTensor 的 Bool-Bool 不支持类型组合；
- InplaceExp2 对 INT32 的不支持情况；
- Exp2 的空指针输入。

这些场景均在源码主函数中显式构造，因此当前异常路径并非泛化描述，而是有真实测试代码支撑的。

### 5.5 执行结果说明

源码在末尾会打印 `Total / Passed / Failed` 并以 `g_fail == 0 ? 0 : 1` 作为程序返回值，因此测试程序具备自动判定和批量回归能力。不过你当前提供的是源码和一份报告草稿，而不是一次真实运行日志，因此这里只能确认测试程序具备统计能力，不能伪造某一次执行时的具体通过数和失败数。

------

## 6 覆盖率结果与质量分析

### 6.1 覆盖率统计结果

现有报告中给出的 5 个评分文件覆盖率如下：

| 文件                                                 | 行覆盖率      | 分支覆盖率     | 至少命中一次   | 调用覆盖率    |
| ---------------------------------------------------- | ------------- | -------------- | -------------- | ------------- |
| `op_api/aclnn_pow.cpp`                               | 72.51% of 251 | 36.43% of 1021 | 20.27% of 1021 | 35.56% of 450 |
| `op_api/aclnn_pow_tensor_tensor.cpp`                 | 87.50% of 80  | 34.60% of 474  | 18.35% of 474  | 40.81% of 223 |
| `op_api/pow.cpp`                                     | 56.67% of 30  | 21.28% of 94   | 11.70% of 94   | 34.92% of 63  |
| `op_host/arch35/pow_tensor_tensor_tiling_arch35.cpp` | 96.03% of 126 | 74.51% of 204  | 45.59% of 204  | 79.46% of 112 |
| `op_host/arch35/pow_tiling_arch35.cpp`               | 81.48% of 54  | 39.29% of 56   | 23.21% of 56   | 12.90% of 31  |

从这些数据看，当前 Pow 测试对 op_host 层，尤其是 TensorTensor tiling 相关逻辑的覆盖效果较好，而 op_api 层中的部分路由和条件分支仍有提升空间。

### 6.2 质量优点分析

第一，API 覆盖完整。7 个 Pow/Exp2 相关 API 均在源码中被显式调用，这一点是当前测试最明显的优势。

第二，语义路径拆分清晰。源码将 TensorScalar、ScalarTensor、TensorTensor 和 Exp2 分开实现测试函数，便于精确定位不同输入角色和不同实现文件的覆盖来源。

第三，特殊指数和特殊底数路径覆盖较有针对性。当前测试并没有停留在普通幂运算，而是有意识地构造了 0、0.5、1、2、3、负数等关键数值场景。

第四，TensorTensor 路径覆盖效果突出。`pow_tensor_tensor_tiling_arch35.cpp` 的行覆盖率达到 96.03%，说明当前测试在 dtype 组合、OP_KEY 分发和广播逻辑方面设计得比较到位。

第五，结果验证方法规范。源码采用 CPU 参考值、dtype 量化、容差比较和统一日志输出的方式，整体上具备较好的可重复性和可信度。

### 6.3 存在问题分析

第一，`op_api/pow.cpp` 覆盖率仍偏低。其行覆盖率仅为 56.67%，分支覆盖率仅为 21.28%，说明设备路由或部分条件判断分支尚未被充分触发。现有报告也将其列为后续重点提升对象。

第二，`aclnn_pow.cpp` 分支覆盖率仍然偏低。虽然行覆盖率达到 72.51%，但分支覆盖率只有 36.43%，说明特殊指数优化路径仍可能存在未命中的情况。

第三，异常路径验证的“严格性”不足。源码中的 `RunExpectedFail` 即使在当前环境下收到 `ACL_SUCCESS` 也会直接计为通过，理由是“still counted as branch-covered”。这意味着这些异常用例更偏向于覆盖率驱动，而不是严格的错误语义断言。若从正式测试严谨性看，这会削弱“异常处理确已被正确验证”的说服力。

第四，边界数值覆盖仍可继续增强。当前源码虽然覆盖了许多特殊指数值，但并没有显式加入 NaN、Inf、极大/极小值等更极端的数值输入，因此在边界稳健性方面仍有提升空间。现有报告也把这一点列为后续改进方向。

------

## 7 复现方式

### 7.1 编译与运行流程

现有报告中给出的复现流程如下：

```bash
bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov

./build_out/cann-ops-math-custom_linux-x86_64.run --quiet

bash build.sh --run_example pow eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

这组命令可用于完成覆盖率插桩编译、安装算子包并在 CPU 模拟器环境中运行测试。

### 7.2 覆盖率文件路径

本次用于统计覆盖率的 `.gcda` 文件包括：

1. `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/pow/op_api/aclnn_pow.cpp.gcda`
2. `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/pow/op_api/aclnn_pow_tensor_tensor.cpp.gcda`
3. `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/pow/op_api/pow.cpp.gcda`
4. `build/math/pow/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/pow_tensor_tensor_tiling_arch35.cpp.gcda`
5. `build/math/pow/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/pow_tiling_arch35.cpp.gcda`

------

## 8 测试结论

本次 Pow 算子测试围绕 TensorScalar、ScalarTensor、TensorTensor 和 Exp2 四类语义路径展开，共覆盖 7 个 API，设计 33 个测试用例，其中正常用例 25 个，异常用例 8 个。测试内容覆盖了多 dtype、广播 shape、原地更新、特殊指数值、特殊底数值以及主要异常输入场景，同时实现了较完整的 CPU 参考值构造、dtype 量化对齐、误差比较和统一日志记录机制。

从测试质量上看，本次 Pow 测试的优势在于 API 覆盖完整、TensorTensor 路径覆盖效果突出、特殊指数值场景设计较有针对性；不足在于 `pow.cpp` 和 `aclnn_pow.cpp` 的部分条件分支仍未充分触发，且异常路径当前更强调“分支覆盖”而非“严格错误断言”。总体而言，这是一份完成度较高、能够支撑 Pow 主功能路径和基础健壮性评估的测试方案，但若要进一步提升说服力，后续仍建议补充更严格的异常断言测试，以及 NaN、Inf 和更多极端指数场景。


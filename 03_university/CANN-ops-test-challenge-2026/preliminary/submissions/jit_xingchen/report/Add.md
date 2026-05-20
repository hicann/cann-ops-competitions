------

# 题目B Add 算子测试报告

## 1 测试概述

### 1.1 测试背景

Add 算子是基础数学算子之一，与普通逐元素加法相比，Add 算子除了要完成张量间加法，还需要处理缩放因子参与计算、不同 shape 下的广播对齐、原地写回以及不同 API 变体之间的调用路径差异，因此其测试不仅要覆盖结果正确性，还要覆盖参数合法性、广播规则和异常输入处理能力。你当前提交的题目 B 材料中，测试目标也是围绕这些方面展开的。

### 1.2 测试目标

本次测试围绕 Add 算子的端到端调用流程，对以下 6 个 API 变体进行验证：

- `aclnnAdd`
- `aclnnAdds`
- `aclnnInplaceAdd`
- `aclnnInplaceAdds`
- `aclnnAddV3`
- `aclnnInplaceAddV3`

测试目标包括：验证各 API 在不同 dtype、不同 shape、不同 alpha 参数和不同调用语义下的结果正确性；验证广播、原地更新和 V3 接口路径是否符合预期；验证 nullptr、非法广播、输出 shape 不匹配等异常输入是否能够被正确拦截；并结合覆盖率结果，对当前测试集质量进行评估。现有报告已明确指出本测试覆盖了全部 6 个 API 变体。

### 1.3 测试脚本总体结构

从源码实现看，该测试脚本并不是简单调用接口，而是构建了一套完整的自验证框架。测试程序先通过 `InitAcl` 完成 ACL 初始化、设备设置和 stream 创建，再在每类测试函数中完成 Tensor/Scalar 构造、workspace 申请、算子执行、结果回传、CPU 参考值计算与误差比较，最后统一通过 `Record` 输出 `[PASS] / [FAIL]` 日志，并在 `main` 末尾汇总总用例数、通过数和失败数。测试结束后再通过 `FinalizeAcl` 完成 ACL 资源释放。

------

## 2 测试需求分析

### 2.1 功能正确性需求

Add 算子的基本需求是在合法输入下正确完成 (x_1 + \alpha \times x_2) 计算，并将结果写入输出 Tensor 或原地写回输入 Tensor。对 `aclnnAdd` 而言，需要验证普通张量加法、广播加法、混合类型加法和空 Tensor 场景；对 `aclnnAdds` 而言，需要验证 Tensor 与标量组合；对 `aclnnInplaceAdd` 和 `aclnnInplaceAdds`，还需要额外验证原地更新后的 self 内容是否正确；对 `aclnnAddV3` 和 `aclnnInplaceAddV3`，则需要验证“scalar + alpha * tensor”这一 V3 语义路径。源码中分别通过 `RunAddCase`、`RunAddsCase`、`RunInplaceAddCase`、`RunInplaceAddsCase`、`RunAddV3Case` 和 `RunInplaceAddV3Case` 实现这些验证。

### 2.2 广播与 shape 合法性需求

Add 算子属于逐元素算子，shape 推断与广播兼容性是核心需求之一。测试必须验证：

- 同 shape 输入下的正常计算；
- 合法广播场景；
- 原地接口中广播结果是否与 self shape 一致；
- 非法广播时接口是否返回失败；
- 输出 shape 不匹配时是否能够被正确拦截。

源码中定义了 `BroadcastShape`、`UnravelIndex` 和 `BroadcastOffset` 等辅助函数，在 CPU 侧完整模拟广播索引过程；同时，主函数中加入了广播成功用例和非法广播失败用例，如 `Add_Fp32_Broadcast_AlphaNeg`、`Add_InvalidBroadcast_FailExpected`、`Add_InvalidOutShape_FailExpected` 和 `InplaceAdd_SelfShapeMismatch_FailExpected`。这说明 shape 与广播规则是本次测试设计的重点之一。

### 2.3 数据类型兼容性需求

Add 算子测试不仅要覆盖常规 FLOAT32，还要覆盖低精度类型、整数类型以及接口中的特殊分支。结合源码实际调用情况，本次测试真实覆盖的数据类型包括 FLOAT32、FLOAT16、BF16、INT32、INT64、INT8 和 BOOL，其中 BOOL 主要出现在 `Adds_Bool_SpecialBranch` 的状态检查用例中；此外，还覆盖了 FP16→FP32 和 BF16→FP32 的混合类型计算场景。现有报告中将这些类型纳入主要覆盖对象，源码也确实实现了对应测试。

### 2.4 alpha 参数覆盖需求

alpha 是 Add 算子区别于普通加法的重要参数，不同 alpha 取值可能触发不同计算路径或边界行为。因此测试需要至少覆盖：

- `alpha = 1` 的标准路径；
- `alpha < 0` 的负缩放路径；
- 浮点 alpha 场景；
- 整数 alpha 场景。

从源码实际情况看，当前测试已经覆盖了 `alpha = 1`、负数 alpha、浮点 alpha 和整数 alpha，例如 `Add_Fp32_Basic_Alpha1`、`Add_Fp32_Broadcast_AlphaNeg`、`Add_Bf16_Basic_AlphaFloat`、`Add_Int32_AlphaNeg` 和 `AddV3_Fp32_AlphaNeg` 等；但源码中**没有显式的 `alpha = 0` 用例**，因此该点不宜在终稿中写成“已覆盖”。

### 2.5 异常处理需求

一个完整的算子测试不能只验证正常输入，还必须验证错误输入能否被正确拒绝。本次测试需要覆盖的异常需求包括：

- self / other / alpha / out 为 nullptr；
- 非法广播；
- out shape 不匹配；
- 原地接口的 self shape 不符合要求；
- V3 接口参数异常。

源码中通过 `RunExpectedFail_GetWorkspace` 对 13 个预期失败用例进行统一封装，只检查 `GetWorkspaceSize` 阶段是否返回错误，不再继续执行实际算子，这种做法与异常路径验证目标是匹配的。

------

## 3 测试环境与测试方法

### 3.1 测试环境

源码显示测试依赖 ACL Runtime 和 Add 相关算子头文件，包括：

- `acl/acl.h`
- `aclnnop/aclnn_add.h`
- `aclnnop/aclnn_add_v3.h`

测试在设备 0 上执行，并通过 ACL stream 串行调度各测试用例。初始化和清理由 `InitAcl` 与 `FinalizeAcl` 统一管理。

### 3.2 测试方法

本次测试采用“CPU 参考结果 + NPU 实际输出 + 误差比较”的方法。对正常路径用例，脚本先根据输入 shape、dtype 和 alpha 在 CPU 侧构造参考结果，再调用对应 `aclnn*GetWorkspaceSize` 和 `aclnn*` 接口执行实际算子，最后将设备侧输出拷回主机端，与 CPU 参考值做逐元素比较。对异常路径用例，则只验证接口是否在 `GetWorkspaceSize` 阶段返回失败。

### 3.3 期望值构造与误差控制

源码中参考值构造逻辑较完整。张量加张量场景使用 `BuildExpectedTensorTensor`，张量加标量场景使用 `BuildExpectedTensorScalar`，V3 的 scalar + tensor 场景使用 `BuildExpectedScalarTensor`。为了让 CPU 期望值与输出 dtype 保持一致，脚本还通过 `QuantizeByDType` 对参考值进行类型量化；误差控制则由 `GetTolerance` 完成，其中 FLOAT16/BF16 使用 `2e-3`，FLOAT32 使用 `1e-5 / 1e-4`，DOUBLE 使用 `1e-12`，整数和布尔类型则要求精确匹配。最终比较由 `CompareVec` 完成。

### 3.4 资源管理方式

测试源码使用 `TensorHolder`、`ScalarHolder` 和 `WorkspaceHolder` 三个资源管理结构体自动释放设备内存、Tensor 对象和 Scalar 对象，避免出现资源泄漏。这种 RAII 风格实现使测试代码在工程上更稳定，也更便于批量扩展新用例。

------

## 4 测试设计与生成思路

### 4.1 总体设计思路

本次测试采用“按接口分类、按场景展开、按风险点补强”的生成思路。首先按 API 语义将测试对象拆分为标准 Add、Adds、InplaceAdd、InplaceAdds、AddV3 和 InplaceAddV3 六类；其次围绕 dtype、shape、broadcast、alpha、原地写回和异常输入等维度组织测试样本；最后通过覆盖率统计反向识别薄弱区域。现有报告中对“API 变体覆盖”“测试维度划分”和“覆盖率改进方向”的描述，与源码结构是一致的。

### 4.2 成功路径生成思路

成功路径用例重点围绕以下几类场景生成：

- 常规同 shape 加法；
- 广播加法；
- 混合类型加法；
- 张量加标量；
- 原地更新；
- V3 scalar + tensor 场景；
- 空 Tensor；
- 大规模 Tensor。

源码中对应的代表性用例包括 `Add_Fp32_Basic_Alpha1`、`Add_Fp32_Broadcast_AlphaNeg`、`Add_Mix_Fp16_Fp32_Alpha1`、`Add_Mix_Bf16_Fp32_Alpha1`、`Adds_Fp32_Scalar_Alpha1`、`InplaceAdd_Fp32_Broadcast`、`AddV3_Fp32_AlphaNeg`、`Add_EmptyTensor` 和 `Add_Fp32_Large_512x256`。这些用例基本覆盖了 Add 算子的主功能链路。

### 4.3 异常路径生成思路

异常路径采用“构造非法参数 + 只检查接口失败”的思路实现。源码没有在异常用例中继续执行实际算子，而是通过 `RunExpectedFail_GetWorkspace` 统一检查 `GetWorkspaceSize` 返回值是否为失败。这一策略适合参数校验型异常用例，因为能够更快定位问题发生在接口前置校验阶段。异常场景覆盖了空指针、非法广播、输出 shape 错误、原地 shape 不兼容以及 V3 接口参数异常等。

### 4.4 特殊分支生成思路

当前 Add 源码里有一类比较特殊的测试是 `Adds_Bool_SpecialBranch`。这一用例并不比较数值结果，而只校验运行状态是否成功，用于触发 bool 相关的特殊实现路径。这类“状态检查型”用例说明测试设计不仅关注数值正确性，也考虑了某些特殊类型分支的实际可执行性。

------

## 5 测试用例设计与执行情况

### 5.1 用例总体统计

根据现有报告和源码主函数统计，本次 Add 算子测试共设计 **40 个** 用例，覆盖成功路径与异常路径两大类。现有报告中给出的分类统计与源码调用数量一致。

### 5.2 按测试函数分类统计

本次测试用例分布如下：

| 测试函数                       | 用例数 | 功能说明             |
| ------------------------------ | ------ | -------------------- |
| `RunAddCase`                   | 11     | 标准 Add API 测试    |
| `RunAddsCase`                  | 5      | Tensor + scalar 测试 |
| `RunAddsStatusOnlyCase`        | 1      | 特殊分支状态测试     |
| `RunInplaceAddCase`            | 3      | 原地 Add 测试        |
| `RunInplaceAddsCase`           | 2      | 原地 Adds 测试       |
| `RunAddV3Case`                 | 3      | V3 版本 Add 测试     |
| `RunInplaceAddV3Case`          | 2      | V3 原地版本测试      |
| `RunExpectedFail_GetWorkspace` | 13     | 预期失败用例         |

这说明当前测试集在结构上较均衡，既保证了标准路径的主体覆盖，也为异常路径保留了较大比重。

### 5.3 成功路径用例分析

成功路径主要由以下几类组成。

第一类是标准 Add 测试，共 11 个，用于覆盖 FLOAT32、FLOAT16、BF16、INT32、INT64、INT8 以及混合精度场景，同时加入广播、空 Tensor 和大 Tensor 用例。典型样例包括 `Add_Fp32_Basic_Alpha1`、`Add_Fp32_Broadcast_AlphaNeg`、`Add_Int64_Alpha2`、`Add_Mix_Fp16_Fp32_Alpha1`、`Add_EmptyTensor` 和 `Add_Fp32_Large_512x256`。

第二类是 Adds 场景，共 5 个数值校验用例加 1 个状态检查用例，覆盖 Tensor + scalar 组合、负 alpha、整数标量和 BOOL 特殊分支。`Adds_Bool_SpecialBranch` 是其中较有代表性的特殊分支测试。

第三类是原地更新场景，包括 3 个 `RunInplaceAddCase` 和 2 个 `RunInplaceAddsCase`，覆盖广播原地更新、INT64 原地更新、FP16 原地更新和标量原地更新。这些用例能够有效验证 self 写回结果是否正确。

第四类是 V3 场景，包括 3 个 `RunAddV3Case` 和 2 个 `RunInplaceAddV3Case`，覆盖 FLOAT32 和 INT8 路径，并验证带 alpha 的 scalar + tensor 语义及其原地版本。V3 接口有独立实现，因此这部分测试对于提高 `aclnn_add_v3.cpp` 的覆盖率尤为重要。

### 5.4 异常路径用例分析

异常路径共 13 个，主要包括：

- Add 的 self/other/alpha/out 为空；
- Add 的非法广播；
- Add 的输出 shape 错误；
- Adds 的 self/other/alpha 为空；
- InplaceAdd 的 self shape 不匹配；
- InplaceAdds 的 self 为空；
- AddV3 的 self 为空；
- AddV3 的输出 shape 错误。

这些异常用例并非泛化描述，而是在源码中逐一显式构造，说明当前测试在参数校验层面具有较强针对性。

### 5.5 执行结果说明

源码会在测试结束时输出 `Total / Passed / Failed` 统计结果，因此测试程序具备完整的自动判分能力。需要说明的是，你当前提供的是源码和一版报告文本，而不是某一次真实运行的控制台日志，所以这里可以确认**测试框架具备汇总与判定机制**，但不能凭空给出某次实际运行的具体通过数和失败数。正式终稿里这样写会更严谨。

------

## 6 覆盖率结果与质量分析

### 6.1 覆盖率统计结果

现有报告给出的 4 个评分文件覆盖率如下：

| 文件                                   | 行覆盖率      | 分支覆盖率     | 至少命中一次   | 调用覆盖率    |
| -------------------------------------- | ------------- | -------------- | -------------- | ------------- |
| `op_api/aclnn_add.cpp`                 | 69.64% of 303 | 36.26% of 1594 | 20.70% of 1594 | 40.09% of 651 |
| `op_api/aclnn_add_v3.cpp`              | 85.71% of 77  | 38.57% of 446  | 21.52% of 446  | 44.62% of 195 |
| `op_api/add.cpp`                       | 44.07% of 59  | 18.94% of 264  | 11.74% of 264  | 25.35% of 142 |
| `op_host/arch35/add_tiling_arch35.cpp` | 74.44% of 90  | 50.63% of 158  | 31.01% of 158  | 23.40% of 94  |

从行覆盖率看，`aclnn_add_v3.cpp` 和 `add_tiling_arch35.cpp` 表现较好，说明当前测试集对 V3 主路径和 tiling 主流程已有较强触达；但从分支覆盖率看，整体仍有较大提升空间。

### 6.2 质量优点分析

第一，API 覆盖完整。6 个 API 变体都在源码中被显式调用，这一点是本次 Add 测试最明显的优势。

第二，测试结构清晰。源码将标准 Add、Adds、原地 Add、V3 和异常路径拆分成独立函数，逻辑边界清楚，便于扩展和维护。

第三，数据类型与场景覆盖较均衡。测试不只覆盖 FLOAT32，还覆盖 FP16、BF16、INT32、INT64、INT8、BOOL，以及广播、空 Tensor、大 Tensor 和混合精度等典型场景。

第四，异常测试较充分。13 个预期失败用例占总用例数的 32.5%，说明当前测试不仅关注“算对”，也重视“错输能拦住”。

第五，参考值构造较规范。源码中期望值计算考虑了广播、dtype 量化和误差容限，因此结果比较逻辑比较可信。

### 6.3 存在问题分析

第一，分支覆盖率偏低。`aclnn_add.cpp` 的分支覆盖率仅为 36.26%，`aclnn_add_v3.cpp` 为 38.57%，说明虽然主要代码行已被执行到，但不少条件分支的另一侧并未真正触发。

第二，设备路由相关覆盖不足。`op_api/add.cpp` 行覆盖率只有 44.07%，分支覆盖率仅 18.94%，这通常意味着设备路由、类型支持判断或某些特定执行路径还没有被充分触发。现有报告也将 `add.cpp` 列为主要待改进文件。

第三，边界值覆盖还不够扎实。虽然原始草稿把 NaN、Inf、alpha=0 列入了目标，但从源码实际调用看，当前**没有单独的 NaN/Inf 用例，也没有显式的 alpha=0 用例**。因此，这些场景更适合作为后续增强方向，而不应在终稿中写成“已覆盖”。

第四，状态检查型用例较少。BOOL 特殊分支目前只做了运行状态验证，没有做结果校验，这对于快速触发特殊实现路径是有效的，但从严格的数值正确性角度看，仍可进一步增强。

------

## 7 复现方式

### 7.1 覆盖率数据文件路径

现有报告给出的 gcda 路径如下：

1. `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcda`
2. `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcda`
3. `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcda`
4. `build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcda`

### 7.2 复现命令

```bash
gcov -b submission_题目B_Add_20260412_152232/build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcda

gcov -b submission_题目B_Add_20260412_152232/build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcda

gcov -b submission_题目B_Add_20260412_152232/build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcda

gcov -b submission_题目B_Add_20260412_152232/build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcda
```



------

## 8 测试结论

本次 Add 算子测试围绕 `aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnAddV3` 和 `aclnnInplaceAddV3` 六类 API 展开，共设计 40 个用例，其中包括 27 个成功路径用例和 13 个预期失败用例。测试在标准 Add、Tensor + scalar、原地更新、V3 路径、广播、混合精度、空 Tensor 和大 Tensor 等方面形成了较系统覆盖，并配套实现了参考值构造、量化对齐、误差比较和统一判定输出机制。

从测试质量上看，本次 Add 测试的优点在于接口覆盖完整、异常测试比例较高、V3 路径覆盖较好、测试代码结构清晰；不足在于 `add.cpp` 设备路由相关覆盖不足，若干条件分支尚未充分触发，同时 alpha=0、NaN/Inf 等边界场景尚未在源码中显式补充。总体而言，这是一份完成度较高、能够支撑功能正确性和基础健壮性评估的测试方案，但若要进一步提升说服力和覆盖深度，仍建议后续补充边界 alpha、特殊数值和更多设备路由相关用例。


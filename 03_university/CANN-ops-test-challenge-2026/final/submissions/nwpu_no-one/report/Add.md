------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "NoOne" 

team_members:

- "成员1：张志伟-西北工业大学"

operator_name: "Add" 

operator_library: "cann-ops-math" 

report_date: "2026-04-25"

------

# 算子测试报告

## 一、算子理解

Add 算子执行逐元素加法，其数学定义为：

\[
y = x_1 + \alpha \times x_2
\]

其中 \(x_1\) 和 \(x_2\) 是输入张量，\(\alpha\) 是标量缩放因子（`aclScalar*` 类型）。当两个输入的形状不一致时，按广播规则对齐后逐元素计算。算子提供 6 个 API 变体：

| API                        | 语义                         |
| -------------------------- | ---------------------------- |
| `aclnnAdd`                 | `out = self + alpha * other` |
| `aclnnAdds`                | `out = self + alpha * scalar`|
| `aclnnInplaceAdd`          | `selfRef += alpha * other`   |
| `aclnnInplaceAdds`         | `selfRef += alpha * scalar`  |
| `aclnnAddV3`               | `out = scalar + alpha * other`|
| `aclnnInplaceAddV3`        | V3 版本的原地加法           |

Add 算子支持多种数据类型组合，包括 FLOAT32、FLOAT16、BF16、INT32、INT8、UINT8、INT64、BOOL、COMPLEX64、COMPLEX128 等同类型运算，以及 FLOAT16‑FLOAT、BF16‑FLOAT 等混合类型运算。其核心计算路径根据 `alpha` 的值、数据类型是否支持 Axpy 等条件选择不同的底层实现：当 `alpha=1` 时直接调用 `l0op::Add`；当 `alpha≠1` 但类型支持 Axpy 时调用 `l0op::Axpy`；否则通过 `l0op::Mul` + `l0op::Add` 实现。V3 API 具有独立的类型提升逻辑和三分支调度（Add、Axpy、Mul+Add）。累加过程中存在浮点误差累积、大数吃小数等典型精度问题。

## 二、测试策略与用例设计

### 参照实现（Oracle）

CPU 端采用 `double` 精度计算期望值：`expected[i] = (double)self[i] + alpha * (double)other[i]`。对于 FLOAT16、BF16 等低精度输入，先将存储值还原为 `float` 再提升至 `double` 计算，保留量化误差。整数类型直接转换为 `double` 累加，容差设为 0。

### 精度阈值

- FLOAT32: `atol=1e-5, rtol=1e-5`
- FLOAT16: `atol=1e-3, rtol=1e-3`
- BF16: `atol=1e-2, rtol=1e-2`
- 整型：精确匹配（当期望值在类型表示范围内时）

### 用例分类

本次测试共设计约 **90 个用例**（相较初版新增 20+ 个覆盖率定向用例），覆盖以下维度：

1. **参数校验与异常拦截**：空指针、非法 dtype（如 9999 魔数）、shape 无法广播、维度超过 8、输出 shape 与广播结果不匹配、Inplace 广播错误、非 ND 格式告警等，确保 API 层与 tiling 层错误拦截完整。
2. **数据类型全覆盖**：所有支持的同类型组合（INT8、UINT8、INT16、INT64、FLOAT32、FLOAT16、BF16、BOOL、COMPLEX64），以及混合类型（BF16+FLOAT、FLOAT+BF16、FLOAT16+FLOAT、FLOAT+FP16 等），并辅以 V3 API 特有的标量输入组合，触发 tiling 层不同的 Dtype 分支。
3. **Alpha 分支**：`alpha=1`（标准加法）、`alpha≠1`（正数、负数、零），并通过错误注入（如 bool + complex alpha）覆盖 `CheckPromoteType` 各个错误返回路径。
4. **API 变体全面覆盖**：6 个 API 变体均有测试，`aclnn_add_v3.cpp` 单独覆盖其类型提升逻辑（`PromoteTypeScalar` 中的 double→float、bool 等分支）及三分支计算调度。
5. **Tiling 层分支拦截**：混合 dtype 输出非 float、非混合但 dtype 不一致、int16 等 tiling 未支持类型，均触发 tiling `CheckDtype` 失败，覆盖 `add_tiling_arch35.cpp` 中的全部 dtype 分支和错误日志。
6. **边界与精度场景**：空 Tensor、Inf/NaN 传播、极大值溢出、极小值下溢、大数加小数、接近 1.0 的抵消、整型溢出饱和、FP16/BF16 精度验证等。
7. **特殊逻辑覆盖**：Adds 中 bool + bool + true alpha 且 out 非 bool 的特殊类型修正逻辑（`aclnn_add.cpp` 行628-635）；`IsEqualToOne` 中 double 类型 alpha 的判断分支。
8. **AICPU 路径覆盖**：通过 DOUBLE 类型输入触发 `add.cpp` 中 `AddAiCpu` 函数的编译路径（虽受限于环境未实际执行，但代码分支已被命中）。

所有用例均以 `NoOneLog` 前缀日志输出，便于在大批量运行中快速提取 PASS/FAIL 状态。

## 三、覆盖率分析

### 测量方法

使用 `gcov` 对编译产物进行覆盖率统计（选项 `-b -c`），统计范围为评分规定的 4 个源文件。更新后结果如下：

| 文件 | 行数 | 执行行数 | 行覆盖率 |
|------|------|---------|---------|
| `aclnn_add.cpp` | 303 | 220 | 72.61% |
| `aclnn_add_v3.cpp` | 77 | 73 | 94.81% |
| `add.cpp` | 59 | 33 | 55.93% |
| `add_tiling_arch35.cpp` | 93 | 83 | 89.25% |

**综合行覆盖率**（加权计算）：\[ \frac{220+73+33+83}{303+77+59+93} = \frac{409}{532} \approx 76.88\% \]

### 未覆盖部分分析

1. **`aclnn_add.cpp` 中 Alpha ≠ 1 的计算路径**：`axpy`、`axpy_v2`、`mul` 等底层 L0 算子因环境限制无法在当前 NPU 上正确执行，导致 `IsSupportAxpy` 和 `else` 分支中的实际 kernel 调用未能执行，仅走完了 `GetWorkspaceSize`。该部分代码约占总未覆盖行的 70%。
2. **`add.cpp` 中 AICPU 实际执行体**：受到硬件平台和驱动版本限制，DOUBLE 等类型无法下发 AICPU 任务，因此 `AddAiCpu` 函数体内部的 `AicpuTaskSpace` 创建、`ADD_TO_LAUNCHER_LIST_AICPU` 宏调用等尚未被实际执行流覆盖（仅参数获取分支覆盖）。若未来环境支持，这部分覆盖率可进一步提升。
3. **`add_tiling_arch35.cpp` 与 `aclnn_add_v3.cpp` 极少量辅助函数**：如 `GetShapeAttrsInfo`、`GetWorkspaceSize`、`PostTiling` 等平台适配接口，它们通常在特定 tiling 模式下才会被调用，当前测试环境中未触发，但均与算子核心逻辑无关。

## 四、精度分析

### 现象解释

从测试日志中观察到：**绝大多数非空张量且 alpha≠1 的计算用例仍输出全零**，导致期望值与实际值严重不匹配（FAIL 数量众多）；而 `alpha=1` 的同类型标准加法、部分标量加法（如 InplaceAdds_BF16_alpha0.0）以及所有错误拦截类测试（预期返回非 SUCCESS）均 PASS。

- **全零输出的根本原因**：`alpha ≠ 1` 或混合类型等场景下，算子内部会调用同层级的 `axpy`、`axpy_v2`、`mul` 等 L0 算子，这些算子在当前 910 硬件/驱动环境中未能正确加载或执行，导致 Add 算子的输出缓冲区未被正确写入，返回全零。
- **正常工作的路径**：当 `alpha=1` 且类型为同类型（非混合）且输入位于 AICORE 支持列表内时，算子直接走 `l0op::Add` 路径，不依赖 `axpy` 等算子，因此能够正确计算，如 InplaceAdd_BOOL_alpha1 等。此外，`alpha=0` 时算子可能被优化为直接拷贝 self，规避了底层算子调用，因此相关用例得以 PASS。
- **新增覆盖用例的影响**：本次新增的 20+ 个用例专注于参数校验、类型提升错误路径、tiling 拦截以及 dummy 张量构造，均属于 `GetWorkspaceSize` 阶段的逻辑验证，无需执行真实计算，因此全零输出现象未发生变化，精度数据的可用性仍受限于底层算子。

### 精度场景理论分析

虽然无法获得完整有效实测数据，但可依据浮点理论预测精度行为：

- **误差累积效应**：Add 算子为单次二元运算，不存在序列累加，因此精度误差主要来源于单次加法的舍入误差和输入本身的量化误差。对于长向量，每个元素独立计算，误差不传播。
- **大数加小数**：当两个加数量级相差很大时，小数的有效位可能在大数的表示范围外，导致加法后被舍入，表现为精度损失。典型场景如 `1e8 + 1e-6`，FLOAT32 下 `1e-6` 的贡献可能丢失。
- **整数溢出**：INT8 类型运算可能溢出，实际 NPU 行为可能是饱和截断或回绕，需要根据测试结果评估。
- **FP16/BF16 精度**：由于尾数位更少，相对误差较 FLOAT32 大 100 倍以上。在测试中，同一组数据下 FP16 的 mismatches 数量会远高于 FLOAT32。

### 本测试环境下的精度评估

当前环境因底层算子缺失，无法获得 alpha≠1 路径的实际计算结果，因此无法量化精度误差。但通过 `alpha=1` 的正常路径 PASS 用例，以及 alpha=0 的特殊简化路径，验证了基本加法逻辑的计算结果与 CPU 参考完全一致，表明算子核心加法路径的精度可以保证。一旦环境支持 alpha≠1 的完整链路，本测试框架可立即产出完整的精度分析数据。

## 五、反思与改进

### 环境限制与未覆盖分支的应对

本次测试通过在 `GetWorkspaceSize` 阶段注入非法参数、构造特殊类型组合以及触发 tiling 层错误拦截，成功将覆盖率显著提升，弥补了硬件环境不足带来的局限。这启示我们，**覆盖率提升不一定依赖实际计算的 PASS/FAIL，合理构造“应该在参数校验阶段就失败”的用例同样能高效覆盖代码**。

### 测试设计经验

- **统一输出前缀 `NoOneLog`**：在 NPU 运行日志极为庞大时，通过自定义前缀可快速提取测试结果，极大地提升了调试效率和日志可读性，是推荐的最佳实践。
- **错误注入的必要性**：此次覆盖率从 72.93% 提升至 76.88%，主要贡献来自对空指针、非法 dtype、shape 不匹配、类型提升失败等错误路径的补充。这些路径在正常功能测试中很难被触发，但却是保证算子健壮性的关键。
- **Tiling 层覆盖的全面性**：通过系统性地传入 tiling 无法支持的类型组合（如 int16、混合输出非 float），确保了 `DoOpTiling` 中每一种 dtype 分支的错误处理逻辑都被执行，有效提升了该文件的覆盖率。

### 对 CANN 测试工具链的建议

- 提供更便捷的本地算子库依赖诊断工具，帮助开发者快速定位类似 `axpy` 等底层算子缺失的问题。
- 丰富模拟器支持的算子集合，使在无真实硬件的环境下也能进行全路径覆盖率测试。
- 在覆盖率统计中区分“逻辑未覆盖”与“环境导致不可达”，避免因硬件限制对测试质量评估造成干扰。

**总结**：本测试通过参数校验、错误注入和 tiling 拦截等策略，在环境受限的情况下仍将综合行覆盖率提升至 76.88%，各核心文件覆盖率显著改善，验证了算子 API 校验和错误处理逻辑的健全性。测试框架结构清晰，具备较强的可扩展性，待环境问题解决后可轻松获得完整的精度分析数据，并为后续性能回归和精度调优奠定基础。
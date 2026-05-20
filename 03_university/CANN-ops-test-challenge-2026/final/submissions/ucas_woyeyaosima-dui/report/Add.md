------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "我也要死吗队"

team_members:

- "成员1：贾皓文-中国科学院大学"

operator_name: "Add"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# Add 算子测试报告

> 测试环境：本次日志显示 `COMPUTE_UNIT=ascend910_93`，构建参数包含 `-DENABLE_UT_EXEC=TRUE`、`-DENABLE_COVERAGE=TRUE`、`-DCMAKE_BUILD_TYPE=Release`，CANN 安装路径为 `/usr/local/Ascend/cann-9.0.0-beta.2`，C/C++ 编译器均为 GNU 11.4.0。运行日志中 64 个测试用例的汇总结果为 `Pass: 36, Fail: 28`。因此，本报告将“功能正确性、精度行为、运行异常与覆盖率”分开讨论，不把失败用例解释为单纯的数值误差。

------

## 一、算子理解

Add 算子的核心语义是对两个输入张量做带系数的逐元素加法：

```text
out[i] = self[i] + alpha * other[i]
```

其中 `self` 与 `other` 可以是形状相同的张量，也可以满足 broadcasting 规则；`alpha` 是标量系数，默认语义上等价于 1。与 Mul 相比，Add 的数值风险不是乘法溢出，而主要集中在**大数吸收小数、抵消误差、特殊值传播、整数回绕、布尔类型转换、混合 dtype promote/cast、inplace 写回别名关系**等场景。

本次测试覆盖了以下接口族：

| 接口/路径 | 语义 | 本次关注点 |
| --- | --- | --- |
| `aclnnAdd` | 张量 + alpha × 张量 | broadcasting、dtype 推导、alpha 分支、AiCore/AiCpu/内部组合路径 |
| `aclnnAdds` | 张量 + alpha × 标量 | 标量 other、低精度标量保持/提升、空张量 |
| `aclnnInplaceAdd` / `aclnnInplaceAdds` | 原地更新 | 输出写回、广播限制、别名与数据覆盖风险 |
| `aclnnAddV3` / `aclnnInplaceAddV3` | 标量 self + alpha × 张量 | scalar-first 形式、Mul+Add 组合路径、V3 参数校验 |

从测试代码看，本次实际涉及的 dtype 包括 FP32、FP16、BF16、DOUBLE、INT32、INT64、INT8、UINT8、BOOL，以及 FP16/FP32、FP32/FP16、FP16/BF16、Float/Int32 等混合类型。浮点类型按容差比较，整数与布尔类型按精确值比较。

需要重点关注的数学与工程性质如下：

1. **broadcasting**：`self` 与 `other` 的维度可不同，低 rank 输入按右对齐规则扩展；输出形状必须是两者可广播后的形状。
2. **alpha 对路径有强影响**：`alpha=0` 时理论上输出应等于 `self`，可触发优化路径；`alpha=1` 时可直接走 Add；其他 `alpha` 可能走 Mul/Add 或 Axpy/AxpyV2 等组合路径。
3. **加法不总是“可交换”**：数学上 `a + b = b + a`，但在算子中存在 `self + alpha * other` 的非对称结构；当 `alpha != 1`、dtype 不同、输出 dtype 固定或 inplace 写回时，交换输入不一定等价。
4. **浮点抵消与吸收**：`1e10 + 1e-3` 在 FP32 输出中很可能无法体现 `1e-3`；`100000 + 0.1 * (-1000000)` 这类近似抵消场景会暴露输入量化与舍入误差。
5. **特殊值传播**：NaN、+Inf、-Inf 的输出不能用普通有限 atol 判断，应单独检查 NaN/Inf 的符号与类型。
6. **整数与布尔类型**：整数加法若超出表示范围会发生底层整数语义下的截断/回绕风险；BOOL 路径需要明确是逻辑 OR 式语义，还是先做数值加法再转 BOOL。

------

## 二、测试策略与用例设计

### 2.1 总体策略

本次测试文件 `test_aclnn_add.cpp` 不是只验证“普通 FP32 加法”，而是围绕 Add 接口族设计了 64 个用例。整体思路参考 MUL 样例的组织方式：先覆盖基础功能与形状，再覆盖 dtype 与边界，最后覆盖精度风险、异常参数和覆盖率。

测试流程统一采用以下模式：

1. 在 host 侧准备输入数据、shape、dtype 与 alpha。
2. 通过 `aclCreateTensor` / `aclCreateScalar` 创建 ACL 张量与标量。
3. 先调用 `*GetWorkspaceSize` 获取 workspace 与 executor，再调用对应 `aclnn*` 执行。
4. 用 `aclrtSynchronizeStreamWithTimeout` 同步 stream，默认超时时间为 10000 ms。
5. 将 device 输出拷回 host。
6. 在 CPU 侧计算 Oracle，并按 dtype 选择精确比较或容差比较。

### 2.2 Oracle 选择

| 类型 | Oracle 方式 | 比较方式 |
| --- | --- | --- |
| FP32 | 输入按 float 存储后提升为 double 计算 `self + alpha * other` | `abs(a-e) <= atol + rtol*abs(e)`，常用 `1e-5/1e-5` 或场景化阈值 |
| FP16 | 先将 `uint16_t` FP16 位模式解码为 float/double，再计算 Oracle | 输出 FP16 解码后比较，常用 `1e-3/1e-3` |
| BF16 | 先将 BF16 位模式解码为 float/double，再计算 Oracle | 输出 BF16 解码后比较，常用 `1e-2/1e-2` |
| DOUBLE | CPU double 直接计算 | `1e-12/1e-12` |
| INT/UINT/BOOL | 按 C++ 对应类型计算期望值 | 精确匹配；BOOL 以非零转 true 的方式构造期望 |
| NaN/Inf | CPU Oracle 保留特殊值 | 检查 NaN/Inf 类型和符号，不能用普通差值解释 |

与 MUL 样例一致，低精度类型不能把 `uint16_t` 存储位模式直接当整数计算；必须先 decode 到浮点值，否则 Oracle 会和 NPU 输入语义不一致。FP32 场景也不能直接拿十进制数学真值做唯一标准，因为输入在写入 float 时已经量化。

### 2.3 用例分布

| 类别 | 代表用例 | 数量 | 通过/失败 | 设计目的 |
| --- | --- | ---: | ---: | --- |
| `aclnnAdd` 张量-张量 | FP32 基础、broadcast、4D、alpha=0、NaN/Inf、整数、FP16/BF16、混合 dtype、非 ND、空张量 | 25 | 7 / 18 | 覆盖主接口、形状、dtype、特殊值和内部 dispatch |
| `aclnnAdds` 张量-标量 | FP32、alpha=0、large+tiny、INT64、DOUBLE、BOOL、FP16/BF16、空张量 | 11 | 6 / 5 | 覆盖 scalar other 形式和低精度保持/提升 |
| `aclnnInplaceAdd` / `aclnnInplaceAdds` | FP32 broadcast、alpha=0、抵消、BOOL、FP16、invalid broadcast | 7 | 6 / 1 | 验证原地写回、广播限制与超时风险 |
| `aclnnAddV3` / `aclnnInplaceAddV3` | scalar self + tensor、FP16、BF16、Int8 MulAdd、empty tensor | 10 | 6 / 4 | 覆盖 V3 scalar-first 语义与组合路径 |
| 负向参数校验 | nullptr、shape mismatch、rank too large、invalid dtype、bool alpha float、V3 unsupported dtype | 11 | 11 / 0 | 验证参数校验能返回非成功状态 |
| **合计** |  | **64** | **36 / 28** |  |

从结果看，负向参数校验全部通过，说明 API 层的基本防御性检查有效；失败主要集中在运行后输出为 0、混合 dtype、BF16/INT 路径、非 ND 格式、部分 V3 与超时场景。

### 2.4 失败用例归类

| 失败模式 | 代表用例 | 现象 | 初步判断 |
| --- | --- | --- | --- |
| 输出保持初始化零值 | `Add_Float32_Basic_Alpha1`、`Add_Int64_Alpha1`、`Add_Bf16_Alpha1`、`Add_Mix_Fp16_Fp32`、`AddV3_Float32_Alpha1` 等 | `actual=0`，但 expected 为非零 | 更像执行路径或输出写回问题，不是普通浮点舍入误差 |
| 同步超时 | `Add_Mix_Fp16_Fp32_Broadcast_Alpha1`、`InplaceAdd_Bool_BroadcastKeep` | `sync stream failed/timeout, runtime=507035, timeout_ms=10000` | 混合 dtype + broadcasting 或 bool inplace broadcast 触发长耗时/卡死风险 |
| 特殊值传播失败 | `Add_Float32_NaN_Inf` | expected 为 NaN/Inf，actual 为 0 | 特殊值路径未得到正确写回或校验前执行异常 |
| INT/UINT/BOOL 路径不一致 | `Add_UInt8_Alpha3`、`Add_Int8_Alpha1`、`AddV3_Int8_MulAddPath` | 结果为 0 或与期望差 1 | 需要进一步区分 cast/promote 规则与实际内核行为 |
| 精度告警但通过 | `Add_Float32_Precision_CancellationAlpha`、`InplaceAdd_Float32_Precision_Cancellation`、`AddV3_Float32_Precision_Cancellation` | pass，但 `precision_metrics` 报 warning | 容差允许，但抵消场景存在可解释的 FP32 精度损失 |

------

## 三、覆盖率分析

### 3.1 测量方法

本次构建开启了 `ENABLE_COVERAGE=TRUE`，运行 `test_aclnn_add` 后收集四个 gcov 文件：

- `aclnn_add.cpp.gcov`
- `aclnn_add_v3.cpp.gcov`
- `add.cpp.gcov`
- `add_tiling_arch35.cpp.gcov`

统计口径如下：

- 行覆盖率：按 gcov 中可执行代码行统计，`#####` 或 `=====` 记为未执行，数字计数记为已执行。
- 分支覆盖率：按 gcov 的 `branch` 行统计，`taken > 0` 记为覆盖，`taken 0` 或 `never executed` 记为未覆盖。
- 函数覆盖率：按 gcov 的 `function ... called N` 统计，`N > 0` 记为覆盖。
- 综合覆盖率：按行数/分支数/函数数加权，而不是简单算术平均。

### 3.2 覆盖率结果

| 文件 | 角色 | 行覆盖率 | 分支覆盖率 | 函数覆盖率 | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| `op_api/aclnn_add.cpp` | Add/Adds/InplaceAdd API 层 | 213/303 = **70.3%** | 376/1546 = **24.3%** | 22/26 = **84.6%** | 主接口、参数校验、dtype promote、Axpy/Mul/Add 组合路径 |
| `op_api/aclnn_add_v3.cpp` | AddV3 API 层 | 69/77 = **89.6%** | 112/426 = **26.3%** | 9/9 = **100.0%** | V3 scalar-first 接口与 inplace V3 |
| `op_api/add.cpp` | L0 Add 路由层 | 33/59 = **55.9%** | 39/264 = **14.8%** | 7/8 = **87.5%** | AiCore/AiCpu 选择、dtype 支持、AddInplace L0 路径 |
| `op_host/arch35/add_tiling_arch35.cpp` | arch35 tiling | 83/93 = **89.2%** | 72/192 = **37.5%** | 12/12 = **100.0%** | tiling 参数、dtype 检查、workspace、platform 信息 |
| **加权合计** |  | 398/532 = **74.8%** | 599/2428 = **24.7%** | 50/55 = **90.9%** |  |

### 3.3 覆盖率解读

本次行覆盖率较高，说明测试确实触达了 Add 主接口、V3 接口、tiling 与 L0 路由层。但分支覆盖率明显偏低，主要原因有三类：

1. **C++ 宏与异常分支膨胀**：gcov 把 `OP_CHECK`、日志宏、异常/throw、短路条件等展开后的分支都计入统计，导致分支总数远大于人工可读分支数。
2. **未覆盖复杂/少见 dtype**：`aclnn_add.cpp` 中复数相关、部分 scalar dtype cast、RegBase 特定路径、`GetScalarDefaultDtype`、`InnerTypeToComplexType`、`CombineCategoriesWithComplex`、`GetCastedFloat` 等函数未被调用。
3. **部分低层路径未触达**：`add.cpp` 中 `AddInplace` L0 函数未被调用，部分非连续内存支持判断因平台条件或 `IsRegBase()` 结果没有走到后续分支。

覆盖率的正面结论是：测试已经能覆盖大部分接口入口与 tiling 主流程；负面结论是：分支覆盖仍不足以证明 dtype 矩阵、异常路径、非连续 Tensor、复数/特殊 scalar 路径已经充分验证。

------

## 四、精度分析

### 4.1 总体结论

本次精度分析必须分成两类：

- **可解释的数值误差**：例如 FP32 抵消、large+tiny 吸收、FP16/BF16 量化误差。这类误差可以用 ULP、atol/rtol 和输入量化解释。
- **不可归因于数值精度的失败**：大量失败用例输出为 0 或同步超时，这不符合浮点舍入模型，更接近执行路径、内核选择、输出写回或平台支持问题。

因此，不能简单写成“精度不达标”。更准确的结论是：**部分通过用例的误差符合预期；失败用例中相当一部分属于功能/执行异常，而不是纯精度误差。**

### 4.2 场景一：基础 FP32 与 broadcasting

**用例**：

- `Add_Float32_Basic_Alpha1`
- `Add_Float32_Broadcast_AlphaNeg`
- `Add_Float32_Broadcast4D_AlphaQuarter`
- `Add_Float32_Large_AlphaZero`

**结果**：

| 用例 | 输入特点 | 结果 | 分析 |
| --- | --- | --- | --- |
| `Add_Float32_Basic_Alpha1` | `{2,4}` 同形状，`alpha=1` | 失败，`index=0 actual=0 expected=1` | 基础同形状路径输出未写回预期值，不能用舍入解释 |
| `Add_Float32_Broadcast_AlphaNeg` | `{2,3}` + `{3}`，`alpha=-0.5` | 通过 | 说明 broadcast 与负 alpha 至少在该路径有效 |
| `Add_Float32_Broadcast4D_AlphaQuarter` | `{2,1,3,1}` + `{1,4,1,5}` -> `{2,4,3,5}` | 通过 | 说明 4D broadcasting 和非整数 alpha 可通过 |
| `Add_Float32_Large_AlphaZero` | 4096 元素，`alpha=0` | 通过 | 说明 alpha=0 优化或普通路径能保持 self |

该组结果最有价值的发现是：同为 FP32，基础同形状失败，而 broadcast/alpha=0/4D 通过，说明问题不只由 dtype 决定，而可能与内部路径选择、输出 Tensor 绑定或特定优化路径有关。

### 4.3 场景二：large + tiny 吸收

**测试输入**：

```text
self  = [1e10, -1e10, 16777216, -16777216, 123456.75, -123456.75]
other = [1e-3, -1e-3, 1, -1, 1e-5, -1e-5]
alpha = 1
```

**预期数值行为**：

在 FP32 中，`1e10` 附近的 ULP 很大，`1e-3` 无法改变最终 FP32 输出；`16777216 = 2^24` 附近，`+1` 也处在 FP32 分辨率边界。因此该用例允许较大的 atol（测试中用于 `MakeFloatCompareResult` 的 strict/warn 阈值更宽），重点是确认输出是否仍接近大数本身。

**实测**：

```text
Add_Float32_Precision_LargePlusTiny: actual=0, expected≈1.000000000e+10, maxAbsErr=1.000000000e+10
Adds_Float32_Precision_LargePlusTiny: actual=0, expected≈1.000000000e+10, maxAbsErr=1.000000000e+10
```

**分析**：

这不是“FP32 吸收 tiny 导致的合理误差”。合理情况下输出应接近 `1e10`，而不是 0。该结果应归类为执行/写回异常。数值层面的风险仍然存在：即使修复执行问题，`1e10 + 1e-3` 在 FP32 中也无法体现 `1e-3` 的贡献。

### 4.4 场景三：抵消误差与 alpha 量化

**测试输入**：

```text
self  = [100000, -100000, 123456.789, -123456.789, 1.0000001, -1.0000001]
other = [-1000000, 1000000, -123456, 123456, -1, 1]
alpha = 0.1f
```

**实测**：

```text
Add_Float32_Precision_CancellationAlpha: PASS
maxAbsErr=1.490116119e-03, maxRelErr=1.000000000e+00, warning=YES

InplaceAdd_Float32_Precision_Cancellation: PASS
maxAbsErr=1.490116119e-03, maxRelErr=1.000000000e+00, warning=YES
```

**分析**：

该结果是可解释的。`0.1f` 在二进制中不能精确表示，`100000 + 0.1f * (-1000000)` 的数学真值接近 0，但按“已量化 float 输入 + float alpha”的 Oracle 计算会出现约 `1.49e-3` 的残差。由于期望值接近 0，相对误差会被放大到 1.0，因此该场景更适合用绝对误差判断。测试通过但保留 warning 是合理的。

### 4.5 场景四：NaN/Inf 特殊值传播

**测试输入**：

```text
self  = [NaN, +Inf, -Inf, 3]
other = [2, -1, 0, +Inf]
alpha = 1
```

**预期**：

- `NaN + 2` 应为 NaN。
- `+Inf + (-1)` 应为 +Inf。
- `-Inf + 0` 应为 -Inf。
- `3 + +Inf` 应为 +Inf。

**实测**：

```text
Add_Float32_NaN_Inf: FAIL, index=0 actual=0 expected=nan
```

**分析**：

输出为 0 不符合 IEEE 特殊值传播，也不像普通容差问题。该用例暴露的是特殊值路径未产生有效输出，后续应结合 dump 或最小复现检查是否进入了预期的 Add 内核、输出地址是否正确传递、以及内核是否被异常跳过。

### 4.6 场景五：FP16、BF16 与混合 dtype

**通过用例**：

- `Add_Fp16_AlphaHalf`
- `Adds_Fp16_PromoteFloat`
- `InplaceAdd_Fp16_Broadcast`
- `AddV3_Fp16_AlphaHalf`
- `AddV3_Bf16_AlphaNeg`

**失败用例**：

- `Add_Bf16_Alpha1`
- `Add_Mix_Fp16_Fp32`
- `Add_Mix_Fp32_Fp16`
- `Add_Mix_Fp16_Fp32_Broadcast_Alpha1`（超时）
- `Add_Fp16_Bf16_Alpha1_CastPath`
- `Adds_Fp16_KeepFp16`
- `Adds_Bf16_KeepBf16`

**分析**：

FP16 的部分路径可以通过，说明 FP16 位模式转换、容差和基础内核并非完全不可用。但 BF16 keep、FP16/BF16 cast path、FP16/FP32 混合 path 失败较集中，说明 promote/cast 或具体组合路径存在更高风险。尤其 `Add_Mix_Fp16_Fp32_Broadcast_Alpha1` 出现 stream 同步超时，说明混合 dtype 与 broadcasting 的组合不仅是精度问题，还可能影响执行稳定性。

FP16/BF16 的合理误差来源包括：

- FP16 尾数位较少，输出一般只能保证约 1e-3 量级。
- BF16 尾数只有 7 位，输出精度通常弱于 FP16，1e-2 量级容差更合理。
- mixed dtype 需明确 promote 后的计算 dtype 与 out dtype；若 out 为 FP32，期望误差应小于低精度输出。

### 4.7 场景六：整数、UINT 与 BOOL

**通过用例**：

- `Add_Int32_Alpha2`
- `Adds_Double_MulPath_AiCpu`
- `InplaceAdds_Bool`

**失败用例**：

- `Add_Int64_Alpha2`
- `Add_UInt8_Alpha3`
- `Add_Int32_Alpha1`
- `Add_Int64_Alpha1`
- `Add_Int8_Alpha1`
- `Add_UInt8_Alpha1`
- `Add_Bool_AlphaTrue`
- `Adds_Int64`
- `Adds_Bool_SpecialCast`
- `AddV3_Int8_MulAddPath`
- `InplaceAddV3_Int8_MulAddPath`

**分析**：

INT32 alpha=2 能通过，但 INT32 alpha=1 失败，说明不同 alpha 触发的内部路径可能不同。UINT8/INT8/BOOL 的失败需要特别关注 cast 与 promote 语义：例如 BOOL 是先数值加法再非零转 true，还是走逻辑 OR；INT8 V3 的 MulAddPath 出现 `actual=-2 expected=-1` 这类非零差异，也比“输出全零”更接近路径语义或整数转换差异。

本次没有系统加入 INT32/INT64 溢出边界用例，例如 `INT32_MAX + 1`、`INT8_MAX + 1`、`UINT8_MAX + 1`。后续应补充这些用例，以明确算子是否按底层整数回绕、饱和、报错或类型提升处理。

### 4.8 精度结论

| 场景 | 是否达标 | 结论 |
| --- | --- | --- |
| FP32 broadcast / alpha=0 / 4D | 达标 | 输出满足容差，说明部分 FP32 路径稳定 |
| FP32 basic same-shape | 不达标 | 输出为 0，疑似功能/写回问题 |
| large+tiny | 不达标 | 失败幅度远超合理 FP32 舍入，属于执行异常 |
| cancellation | 条件达标 | 通过但 warning 合理，体现抵消误差 |
| NaN/Inf | 不达标 | 特殊值未传播，输出为 0 |
| FP16 | 部分达标 | FP16 alpha half 与部分 inplace/V3 通过 |
| BF16 / mixed dtype | 多数不达标 | BF16 keep、mixed dtype 与 cast path 风险高 |
| INT/UINT/BOOL | 部分不达标 | 参数校验好，但实际计算路径结果不稳定 |

------

## 五、反思与改进

### 5.1 本次测试的有效发现

本次测试并没有只得到“通过/失败”的表面结论，而是暴露出四类有价值问题：

1. **负向参数校验较完整**：nullptr、shape mismatch、rank too large、invalid dtype、unsupported dtype 等 11 个负向用例全部通过，说明 API 前置检查可靠。
2. **部分正常路径稳定**：FP32 broadcasting、4D broadcasting、alpha=0、empty tensor、部分 inplace、FP16 和 V3 低精度路径能够通过。
3. **大量输出为 0 的失败值得重点排查**：这类失败与数值误差无关，疑似输出未写回、内部路径未执行到有效 kernel、或某些 dtype/alpha 路径的输出绑定异常。
4. **超时用例可作为稳定性缺陷线索**：混合 dtype + broadcasting 与 bool inplace broadcast 在 10000 ms 超时，说明不仅要测正确性，还要测运行时行为。

### 5.2 当前测试盲区

1. **整数溢出边界不足**：缺少 `INT32_MAX + 1`、`INT32_MIN - 1`、`INT8_MAX + 1`、`UINT8_MAX + 1` 等边界。
2. **非连续内存覆盖不足**：目前主要是 ND/NCHW 与普通 contiguous tensor，尚未系统覆盖 slice/view/strided 非连续输入。
3. **dtype 矩阵不够系统**：已覆盖若干 mixed dtype，但还缺完整矩阵，例如 float/int、bf16/fp32、double/float、bool/int 等组合。
4. **平台差异未横向验证**：日志中出现 `ascend910_93` 与部分 “not supported” 信息，当前结果可能与平台、固件或 OPP 配置相关，需要在目标评分环境复测。
5. **失败根因仍需 dump/最小复现**：对于输出为 0 的失败，需要进一步打开算子 dump、比对实际 launch 列表、确认输出 tensor 地址和 kernel 返回状态。

### 5.3 后续改进建议

若继续扩展测试，建议按优先级做以下工作：

1. **先修复/定位输出为 0 的主路径失败**：从 `Add_Float32_Basic_Alpha1` 建最小复现，比较 `alpha=1` 与 `alpha=-0.5`、同 shape 与 broadcast、ND 与 NCHW 的内部路径差异。
2. **为失败模式增加分类断言**：将“同步超时”“GetWorkspace 失败”“Run 失败”“输出未写回”“数值误差超限”分成不同错误码，而不是统一记录为 FAIL。
3. **补全整数边界与特殊值用例**：加入整数溢出、NaN payload、`+Inf + -Inf`、subnormal、signed zero 等场景。
4. **提高 branch coverage 的有效性**：针对未调用函数定向补用例，如 complex、RegBase、scalar dtype cast、non-contiguous、L0 AddInplace 等路径。
5. **记录实际环境与 OPP 配置**：报告中应保留 CANN 版本、compute unit、timeout、失败用例详情，避免在不同 Ascend 平台上复现结果不一致时无法追踪。

### 5.4 总体评价

本次 Add 测试覆盖面较宽，能有效触达 API 层、V3 层、L0 路由层和 tiling 层，行覆盖率达到 74.8%，函数覆盖率达到 90.9%。但由于运行结果中有 28 个失败和 2 个超时场景，本轮测试更适合作为“问题发现型测试报告”，而不是“全部达标型测试报告”。

最终结论是：**Add 算子的参数校验与部分 FP32/FP16/广播/inplace 路径表现良好，但基础同形状、BF16/mixed dtype、部分整数/BOOL、特殊值与部分 V3 路径存在明显风险，需要进一步定位执行/写回路径后再确认数值精度是否达标。**

------

## 附录：运行结果清单

| 用例 | 结果 | 摘要 |
| --- | --- | --- |
| Add_Float32_Basic_Alpha1 | FAIL | actual=0, expected=1 |
| Add_Float32_Broadcast_AlphaNeg | PASS | - |
| Add_Float32_Large_AlphaZero | PASS | - |
| Add_Float32_Broadcast4D_AlphaQuarter | PASS | - |
| Add_Float32_PrecisionStress | FAIL | actual=0, expected≈1e10 |
| Add_Float32_Precision_LargePlusTiny | FAIL | maxAbsErr≈1e10 |
| Add_Float32_Precision_CancellationAlpha | PASS | maxAbsErr≈1.49e-3，warning=YES |
| Add_Float32_NaN_Inf | FAIL | actual=0, expected=NaN |
| Add_Int32_Alpha2 | PASS | - |
| Add_Int64_Alpha2 | FAIL | actual=0, expected=5 |
| Add_UInt8_Alpha3 | FAIL | actual=0, expected=7 |
| Add_Int32_Alpha1 | FAIL | actual=0, expected=6 |
| Add_Int64_Alpha1 | FAIL | actual=0, expected=11 |
| Add_Int8_Alpha1 | FAIL | actual=0, expected=-1 |
| Add_UInt8_Alpha1 | FAIL | actual=0, expected=7 |
| Add_OutAliasOther_Probe | FAIL | actual=2, expected=3 |
| Add_Fp16_AlphaHalf | PASS | - |
| Add_Bf16_Alpha1 | FAIL | actual=0, expected=3 |
| Add_Mix_Fp16_Fp32 | FAIL | actual=0, expected=3.5 |
| Add_Mix_Fp32_Fp16 | FAIL | actual=0, expected=3 |
| Add_Mix_Fp16_Fp32_Broadcast_Alpha1 | FAIL | stream sync timeout |
| Add_Fp16_Bf16_Alpha1_CastPath | FAIL | actual=0, expected=3 |
| Add_Bool_AlphaTrue | FAIL | actual=0, expected=1 |
| Add_Float32_NonNdFormat | FAIL | actual=0, expected=3 |
| Add_EmptyTensor | PASS | - |
| Adds_Float32 | PASS | - |
| Adds_Float32_AlphaZero | PASS | - |
| Adds_Float32_Precision_LargePlusTiny | FAIL | maxAbsErr≈1e10 |
| Adds_Float32_NonNdFormat | PASS | - |
| Adds_Int64 | FAIL | actual=0, expected=7 |
| Adds_Double_MulPath_AiCpu | PASS | - |
| Adds_Bool_SpecialCast | FAIL | actual=0, expected=1 |
| Adds_EmptyTensor | PASS | - |
| Adds_Fp16_KeepFp16 | FAIL | actual=0, expected=1.5 |
| Adds_Fp16_PromoteFloat | PASS | - |
| Adds_Bf16_KeepBf16 | FAIL | actual=0, expected=1.5 |
| InplaceAdd_Float32_Broadcast | PASS | - |
| InplaceAdd_Float32_AlphaZero | PASS | - |
| InplaceAdd_Float32_Precision_Cancellation | PASS | maxAbsErr≈1.49e-3，warning=YES |
| InplaceAdds_Float32 | PASS | - |
| InplaceAdds_Bool | PASS | - |
| InplaceAdd_Bool_BroadcastKeep | FAIL | stream sync timeout |
| InplaceAdd_Fp16_Broadcast | PASS | - |
| AddV3_Float32_Alpha1 | FAIL | actual=0, expected=3.25 |
| AddV3_Float32_AlphaNeg | PASS | - |
| AddV3_Float32_Precision_Cancellation | PASS | maxAbsErr≈72.3，warning=YES |
| AddV3_Fp16_AlphaHalf | PASS | - |
| AddV3_Bf16_AlphaNeg | PASS | - |
| AddV3_SelfFloat_OtherInt32 | FAIL | actual=0, expected=3.25 |
| AddV3_EmptyTensor | PASS | - |
| AddV3_Int8_MulAddPath | FAIL | actual=0, expected=-1 |
| InplaceAddV3_Float32 | PASS | - |
| InplaceAddV3_Int8_MulAddPath | FAIL | actual=-2, expected=-1 |
| NEG_Add_Nullptr | PASS | 返回非成功状态 |
| NEG_Add_ShapeMismatch | PASS | 返回非成功状态 |
| NEG_Add_RankTooLarge | PASS | 返回非成功状态 |
| NEG_Add_InvalidDtype | PASS | 返回非成功状态 |
| NEG_InplaceAdd_InvalidBroadcast | PASS | 返回非成功状态 |
| NEG_Adds_NullAlpha | PASS | 返回非成功状态 |
| NEG_Adds_ShapeMismatch | PASS | 返回非成功状态 |
| NEG_Add_BoolAlphaFloat | PASS | 返回非成功状态 |
| NEG_AddV3_Nullptr | PASS | 返回非成功状态 |
| NEG_AddV3_ShapeMismatch | PASS | 返回非成功状态 |
| NEG_AddV3_UnsupportedDtype | PASS | 返回非成功状态 |

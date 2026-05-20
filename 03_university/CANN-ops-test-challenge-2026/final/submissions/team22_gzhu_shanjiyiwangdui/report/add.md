------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "闪击翼王队" 

team_members:

- "成员1：周亚超-广州大学"
- "成员2：张雨桐-广州大学"
- "成员3：许恒恒-广州大学" 

operator_name: "Add" 

operator_library: "cann-ops-math" 

report_date: "2026-04-25"

------

# Add 算子测试报告

------

## 一、算子理解

Add 算子执行逐元素加法，其数学定义为：

```text
y = x1 + alpha * x2
```

其中 `alpha` 是 `aclScalar*` 类型的标量缩放因子，默认值为 1。当两个输入 tensor 的 shape 不一致时，算子按照 broadcasting 规则对齐后逐元素计算。Add 算子的输出元素只依赖当前位置的两个输入值，因此不存在类似 Cumsum 那样的长链式误差累积；但是 Add 引入了 `alpha` 参数，实际计算包含一次乘法和一次加法，所以误差来源不仅包括加法舍入，也包括 `alpha * x2` 的乘法舍入以及类型提升、输出回写时的舍入。

Add 算子位于 `math/add/` 目录下，整体采用 `op_api -> op_host -> op_kernel` 的三层结构。其中本题评分重点文件包括 API 层的 `aclnn_add.cpp`、`aclnn_add_v3.cpp`、`add.cpp`，以及 Host 层的 `op_host/arch35/add_tiling_arch35.cpp`。API 层主要完成参数检查、shape/broadcast 推导、dtype promote、alpha 处理和底层算子路由；Host 层主要完成 tiling 策略选择，受 dtype、shape、broadcast、contiguous/non-contiguous 等因素影响。

Add 对外提供 6 类 API：`aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnAddV3`、`aclnnInplaceAddV3`。其中 V3 API 与普通 Add 的关键区别是 `self` 参数为 scalar 而不是 tensor，语义为 `out = scalar + alpha * other`。由于 V3 有独立实现文件 `aclnn_add_v3.cpp`，若不显式调用 V3 API，该文件覆盖率会非常低。

在 dtype 方面，Add 支持 FLOAT32、FLOAT16、BF16、INT32、INT64、INT8、UINT8、BOOL、COMPLEX32、COMPLEX64 等类型及部分混合类型组合。不同 dtype 会触发不同的 promote 逻辑、底层 dispatch 路径和 tiling 模板，因此 dtype 是本次测试覆盖的核心维度之一。

------

## 二、测试策略与用例设计

本次测试在官方 `math/add/examples/test_aclnn_add.cpp` 基础上扩展，目标不是单纯增加用例数量，而是根据 gcov 结果对未覆盖分支进行定点补洞。初始覆盖结果显示：`add_tiling_arch35.cpp` 行覆盖已达到较高水平，而 `aclnn_add.cpp`、`add.cpp` 的行覆盖和分支覆盖仍然偏低。因此后续测试设计重点放在 API 分支、dtype 分支、fallback 路径、错误路径和 V3 独立路径上。

本次冲分版测试用例主要覆盖以下维度。

### 1. API 入口覆盖

测试覆盖全部 6 类 API：

| API | 测试目的 |
| --- | --- |
| `aclnnAdd` | tensor + tensor 的基础路径、alpha 分支、broadcast 路径 |
| `aclnnAdds` | tensor + scalar 路径，覆盖 scalar 参数构造与 Adds 分支 |
| `aclnnInplaceAdd` | 原地 tensor + tensor，覆盖 inplace 检查与 selfRef 输出路径 |
| `aclnnInplaceAdds` | 原地 tensor + scalar，覆盖 inplace scalar 分支 |
| `aclnnAddV3` | scalar + tensor，覆盖 V3 独立文件与 scalar promote |
| `aclnnInplaceAddV3` | V3 原地路径，覆盖 V3 inplace 检查与调度 |

### 2. alpha 参数覆盖

`alpha` 是 Add 算子区别于普通加法的关键参数。本次设计了以下 alpha 场景：

- `alpha = 1`：普通加法路径，部分实现会直接走 Add。
- `alpha = 0`：理论输出退化为 `x1`，用于覆盖特殊值分支。
- `alpha = 2`：常规 Axpy/AxpyV2 路径。
- `alpha = -1`：减法语义 `x1 - x2`，覆盖负 alpha。
- `alpha = 0.3`：非整数缩放，部分 dtype 下触发 fallback 或 Mul + Add。

### 3. dtype 覆盖

测试覆盖 FLOAT32、FLOAT16、BF16、INT32、INT64、INT8、UINT8、BOOL、DOUBLE、COMPLEX 等类别。其中 FLOAT32/INT32 用于稳定功能验证；FLOAT16/BF16 用于触发半精度与 BF16 dtype 相关路径；INT8/UINT8/BOOL 用于覆盖小整数与 bool 特判；DOUBLE/INT16 等可能不完全支持的组合以 GetWorkspace-only 方式尝试触发 API/L0 dispatch 和 fallback 分支，避免由于运行环境或 kernel 支持差异导致整套测试失败。

### 4. shape 与 broadcast 覆盖

测试包含以下 shape 类型：

- 同 shape：如 `[4]`、`[2, 3]`，验证基础逐元素路径。
- broadcasting：如 `[1] + [N]`、`[1, C] + [B, C]`，触发 broadcast 推导和 tiling 分支。
- scalar-like tensor：用于 tensor 与标量语义互补覆盖。
- 较大 tensor：用于触发不同 tiling 策略。
- 空 tensor：如 shape 中存在 0，覆盖 early return 或空输入处理分支。

### 5. non-contiguous 与 strided output 覆盖

根据覆盖率结果，Host 层行覆盖已经较高，但分支覆盖仍有提升空间，且 `broadcast_tiling_noncontiguous` 相关路径此前未充分触发。因此测试中加入了非连续 tensor 或 strided output 场景，用于覆盖 contiguous 与 non-contiguous 的分支差异。

### 6. 异常路径覆盖

异常路径是分支覆盖的重要来源。本次增加了以下负例：

- 输入 tensor 为 nullptr。
- 输出 tensor 为 nullptr。
- alpha 为 nullptr。
- shape 不可 broadcast。
- dtype 不支持或组合不支持。

这类用例只检查 `GetWorkspaceSize` 返回失败是否符合预期，不执行 kernel，避免错误输入导致设备侧异常。

### 7. GetWorkspace-only 策略

对于 DOUBLE、INT16、部分 fallback 或 AiCpu 相关路径，测试采用 GetWorkspace-only 策略：只调用 `GetWorkspaceSize` 以触发 API 层参数检查、dtype 判断、promote、dispatch 和 fallback 分支；若该路径在当前环境不支持，则接受预期失败，不把它作为硬失败用例。这样既能提高覆盖率，又不会因为不同 NPU/CANN 版本对 fallback kernel 支持程度不同而影响整套测试稳定性。

### 8. Oracle 与验证策略

对于实际执行的功能用例，CPU 参考结果使用 double 精度计算：

```cpp
double expected = static_cast<double>(x1[i]) + alpha * static_cast<double>(x2[i]);
```

浮点类型采用容差比较：

```cpp
abs(actual - expected) <= abs_tol + rel_tol * abs(expected)
```

容差设置：

| dtype    | abs_tol | rel_tol |
| ---      | ---     | ---     |
| FLOAT32  | 1e-6    | 1e-6    |
| FLOAT16  | 1e-4    | 1e-4    |
| BF16     | 1e-2    | 1e-2    |
| 整数类型 | 0        | 0      |

整数类型使用精确匹配；BOOL 按逻辑值匹配；特殊浮点值 NaN/Inf 使用 `std::isnan`、`std::isinf` 进行专门判断。

------

## 三、覆盖率分析

本次覆盖率统计对象为 Add 题目规定的 4 个评分文件：

| 文件 | layer | 说明 |
| --- | --- | --- |
| `op_api/aclnn_add.cpp` | API | Add/Adds/InplaceAdd/InplaceAdds 的参数检查、dtype promote、alpha 路径与调度 |
| `op_api/aclnn_add_v3.cpp` | API | V3 scalar + tensor API 的独立实现 |
| `op_api/add.cpp` | API/L0 | 底层 Add 路由、dtype 支持判断、AiCore/AiCpu/fallback dispatch |
| `op_host/arch35/add_tiling_arch35.cpp` | Host | dtype 与 shape 相关 tiling 策略 |

根据前一轮 gcov 结果，覆盖情况如下：

| 文件 | 行覆盖率 | 分支覆盖率 | 说明 |
| --- | --- | --- | --- |
| `op_api/aclnn_add.cpp` | 55.78% of 303 | Branches executed 28.91%，Taken at least once 15.91% | 普通 Add API 层，主要短板是 dtype、alpha、异常和 fallback 分支 |
| `op_api/aclnn_add_v3.cpp` | 80.52% of 77 | Branches executed 27.70%，Taken at least once 15.26% | V3 行覆盖较高，但分支仍集中在少数路径 |
| `op_api/add.cpp` | 44.07% of 59 | Branches executed 18.18%，Taken at least once 10.98% | 当前最大短板，底层 dispatch 与 fallback 路径覆盖不足 |
| `op_host/arch35/add_tiling_arch35.cpp` | 86.02% of 93 | Branches executed 54.17%，Taken at least once 33.85% | Host 层行覆盖较好，剩余主要是 non-contiguous 与 broadcast 细分分支 |

从数据看，`add_tiling_arch35.cpp` 行覆盖已经达到 86.02%，说明 dtype 和基础 shape 维度已经基本触发 Host tiling 主体逻辑；当前瓶颈已经不是 Host 层行覆盖，而是 API 层和底层 dispatch 的分支覆盖。尤其是 `add.cpp` 只有 44.07% 行覆盖、10.98% Taken 分支，是后续提升的重点。

针对上述结果，本次增强版测试做了以下定向改进：

1. 增加空 tensor case，覆盖 early return 或空输入处理逻辑。
2. 增加 `alpha != 1`、`alpha = 0`、`alpha < 0` case，覆盖 alpha 相关分支。
3. 增加 AxpyV2 候选路径，覆盖融合算子调度。
4. 增加 BF16 scalar 与 BOOL Adds 特判，用于触发特殊 dtype 分支。
5. 增加 V3 int8 + float alpha fallback case，覆盖 V3 的 Mul + Add 或 fallback 路径。
6. 增加 double/int16 的 GetWorkspace-only case，尝试触发 AiCpu fallback 或不支持 dtype 分支。
7. 增加 nullptr、非法 shape、非法 dtype 等负例，提升参数检查分支覆盖。
8. 增加 non-contiguous/strided output 场景，进一步提升 Host tiling 分支覆盖。

分支覆盖率低于行覆盖率是合理现象。API 层存在大量多条件判断，例如 dtype 是否支持、alpha 是否为 1、是否需要类型提升、是否 broadcast、是否 inplace、是否空 tensor、是否 fallback 等。行覆盖只要求某行执行过一次，而分支覆盖要求同一条件的真假两侧都被触达，因此更难提升。本测试后续改进方向仍然是围绕这些条件的反向分支继续补洞。

------

## 四、精度分析

Add 算子的精度问题主要来自浮点加法舍入、`alpha * x2` 的乘法舍入、不同 dtype 的输入量化与输出量化，以及特殊值传播。相比 Cumsum，Add 没有沿 axis 方向的累积误差，但单次加法中仍存在多个典型风险场景。

### 场景一：大数 + 小数导致小数被吞没

**测试输入**：

```text
x1 = [1e10, 1e10]
x2 = [1e-5, 1e-5]
alpha = 1
```

数学期望：

```text
[10000000000.00001, 10000000000.00001]
```

实际在 FLOAT32 中，1e10 附近的 ULP 已远大于 1e-5，因此 `1e-5` 对结果没有可观测影响，输出通常仍为：

```text
[10000000000.0, 10000000000.0]
```

**分析**：

FLOAT32 只有约 7 位十进制有效数字。当数值量级达到 1e10 时，小数部分已经超出该 dtype 的有效表示能力。此时不是 Add 算子实现错误，而是浮点数有限有效位导致的小量信息丢失。这类问题在梯度更新中常见，例如大权重加上极小梯度时，极小更新可能完全不起作用。

**验证策略**：

该场景不应使用严格的数学真值判定算子失败，而应在报告中作为精度现象记录。测试可以打印 expected、actual、error，并说明该误差符合 FLOAT32 表示能力限制。

------

### 场景二：正负抵消导致灾难性消减

**测试输入**：

```text
x1 = [1.0000001, 2.0000001]
x2 = [-1.0, -2.0]
alpha = 1
```

理论结果为接近 1e-7 的小数：

```text
[0.0000001, 0.0000001]
```

**分析**：

当两个接近的浮点数相减时，高位有效数字相互抵消，结果只剩低位差异。由于输入本身已经经过 FLOAT32 量化，低位有效位非常有限，抵消后的结果相对误差会明显增大。这种现象称为 Catastrophic Cancellation。Add 在 `alpha = -1` 或 `x2` 为负值时会表现为减法，因此必须覆盖该类场景。

**风险**：

对 loss 差值、残差计算、归一化修正等场景，正负抵消可能导致结果相对误差明显扩大。若业务强依赖小差值，应考虑使用 FP32 以上精度，或重写公式避免两个大且相近的数直接相减。

------

### 场景三：alpha 引入的额外误差

**测试输入**：

```text
x1 = [1.0, 2.0, 3.0]
x2 = [0.1, 0.2, 0.3]
alpha = 3.7
```

数学形式为：

```text
out = x1 + 3.7 * x2
```

**分析**：

普通加法只包含一次加法舍入，而 Add 包含 `alpha * x2` 和之后的加法。若 `alpha` 和 `x2` 均无法被二进制浮点精确表示，则乘法阶段会引入一次舍入；加到 `x1` 时再次舍入。因此 `alpha != 1` 的误差通常高于 `alpha = 1`。当 `alpha` 绝对值较大时，`x2` 中已有的量化误差也会被放大。

**测试意义**：

该场景用于覆盖 Axpy/AxpyV2 或 Mul + Add 路径，同时也是 Add 精度分析中区别于普通 Add 的核心场景。

------

### 场景四：混合 dtype 与类型提升

**测试输入示例**：

```text
x1 dtype = FLOAT16
x2 dtype = FLOAT32
alpha dtype = FLOAT32
```

或：

```text
x1 dtype = INT8
x2 dtype = FLOAT32
alpha = 0.5
```

**分析**：

混合 dtype 运算需要先进行类型提升。提升后的计算 dtype、输出 dtype 以及 scalar dtype 会共同影响最终精度。例如 FLOAT16 输入在进入计算前已发生量化；即便后续提升到 FLOAT32，丢失的信息也无法恢复。INT8 与 FLOAT32 混合时，整数可以精确转为 FLOAT32，但若输出再转回整数，则会发生截断或舍入相关误差。

**测试意义**：

混合 dtype 不仅是精度风险点，也是 API 层 dtype promote 分支和底层 dispatch 分支的重要覆盖来源。

------

### 场景五：NaN 与 Inf 特殊值传播

**测试输入**：

```text
x1 = [NaN, Inf, -Inf, 1.0]
x2 = [1.0, 2.0, -3.0, Inf]
alpha = 1
```

**预期行为**：

- `NaN + finite = NaN`
- `Inf + finite = Inf`
- `-Inf + finite = -Inf`
- `finite + Inf = Inf`
- `Inf + (-Inf) = NaN`

**分析**：

Add 应遵循 IEEE 754 的特殊值传播规则。对这类场景不能使用普通的绝对误差判断，因为 NaN 与任何值比较都返回 false，Inf 与有限容差比较也没有意义。验证逻辑必须使用 `std::isnan` 和 `std::isinf`。

**风险**：

NaN 和 Inf 具有强传播性，一旦中间结果出现 NaN/Inf，后续计算很可能继续产生无效值。因此 Add 测试应覆盖特殊值，以便确认算子不会对特殊值做非预期静默处理。

------

### 场景六：整数溢出

**测试输入示例**：

```text
x1 = [2147483647, -2147483648]
x2 = [1, -1]
alpha = 1
```

或：

```text
x1 = [120]
x2 = [120]
dtype = INT8
alpha = 1
```

**分析**：

对于整数类型，Add 一般不做溢出检测。如果结果超出目标 dtype 的表示范围，实际行为通常表现为底层整数表示的截断或回绕。需要注意的是，C++ 中有符号整数溢出是 undefined behavior，因此 CPU Oracle 不应直接用可能溢出的有符号表达式作为参考。更稳妥的做法是用更宽类型计算数学结果，再根据目标 dtype 的实际规则进行转换，或只在安全范围内做精确匹配。

------

## 五、反思与改进

本次测试的主要收获是：覆盖率提升不能靠盲目增加 case，而要依赖 gcov 精准定位未覆盖分支。前一轮结果显示 `add_tiling_arch35.cpp` 行覆盖已经达到 86.02%，继续堆 dtype 和普通 shape 对综合提升有限；真正的瓶颈在 `aclnn_add.cpp` 和 `add.cpp`，特别是 dtype promote、alpha 分支、fallback 分支、参数错误分支和底层 dispatch 路径。

**API 覆盖方面**，Add 题相比 Mul 更复杂，因为它有普通 Add、Adds、Inplace、V3 等 6 类 API，而且 V3 独立实现。如果只测试 `aclnnAdd`，`aclnn_add_v3.cpp` 会几乎没有覆盖。因此本次优先补齐 6 类入口，并为 V3 单独设计 `alpha = 1`、Axpy、fallback 三类 case。

**dtype 覆盖方面**，单纯测试 FLOAT32 和 INT32 不足以覆盖 Add 的 dtype 分支。BF16、FLOAT16、BOOL、INT8、UINT8、DOUBLE、COMPLEX 等 dtype 会触发不同分支。其中部分 dtype 在当前环境下可能无法完整执行，因此采用 GetWorkspace-only 策略覆盖 API/L0 分支，这是一种兼顾覆盖率和稳定性的折中方案。

**Host 层方面**，tiling 行覆盖已经较高，后续提升主要来自 broadcast 的复杂组合和 non-contiguous tensor。普通 contiguous tensor 很难继续提升分支覆盖，因此需要构造 strided output、transpose/slice 后的非连续输入，以及较复杂的 broadcasting shape。

**精度方面**，Add 虽然是简单逐元素算子，但并非没有精度风险。大数加小数、正负抵消、alpha 放大误差、混合 dtype、NaN/Inf 传播和整数溢出都需要在报告中说明。尤其是 `alpha != 1` 时，算子实际包含乘法和加法两个阶段，误差来源比普通加法更多。

**后续改进方向**如下：

1. 继续根据 `.gcov` 文件定位 `#####` 未覆盖行，补充更精确的分支 case。
2. 为 COMPLEX32/COMPLEX64 增加完整执行与验证，而不仅是结构触发。
3. 增加更多不可 broadcast shape 的负例，提升参数检查分支覆盖。
4. 增加更多 non-contiguous 输入和输出组合，提高 Host 层分支覆盖。
5. 对每类 dtype 建立独立 Oracle，避免 FP16/BF16/整数溢出场景中 CPU 参考本身出错。

总体而言，本测试方案已经覆盖 Add 算子的主要 API、dtype、alpha、broadcast、fallback、异常和精度风险场景，具备较好的覆盖率提升能力和测试报告完整性。下一步若继续优化，应以分支覆盖为主要目标，而不是继续扩展普通功能

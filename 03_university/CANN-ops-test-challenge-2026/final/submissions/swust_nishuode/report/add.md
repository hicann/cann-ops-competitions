# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "你说的"

team_members:

- "何嘉俊-西南科技大学"
- "韦汉鑫-西南科技大学"
- "黄世凯-西南科技大学"

operator_name: "Add"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# 算子测试报告

## 一、算子理解

本次测试对象为 CANN `cann-ops-math` 仓库中的 Add 算子。Add 算子在 aclnn 接口中的核心语义不是单纯的 `x1 + x2`，而是带有 `alpha` 缩放参数的加法：

```text
out = self + alpha * other
```

其中：

- `self`：第一个输入，可以是 Tensor，也可以在 V3 接口中表现为 Scalar。
- `other`：第二个输入，通常为 Tensor 或 Scalar。
- `alpha`：缩放因子，用于控制 `other` 在加法前的缩放。
- `out`：输出 Tensor，shape 通常由输入 broadcast 后确定。

### 1.1 API 形态

本次测试重点覆盖 Add 算子的 6 类对外 API：

| API | 语义说明 |
|---|---|
| `aclnnAdd` | Tensor + Tensor，计算 `self + alpha * other` |
| `aclnnAdds` | Tensor + Scalar，计算 `self + alpha * scalar_other` |
| `aclnnInplaceAdd` | 原地 Tensor + Tensor，结果写回 selfRef |
| `aclnnInplaceAdds` | 原地 Tensor + Scalar，结果写回 selfRef |
| `aclnnAddV3` | Scalar + Tensor，计算 `scalar_self + alpha * other` |
| `aclnnInplaceAddV3` | Scalar + Tensor 原地更新 other Tensor |

其中，V3 接口与普通 Add 接口不同，`self` 输入为 Scalar，因此需要单独设计用例，否则 `aclnn_add_v3.cpp` 无法被有效覆盖。

### 1.2 dtype 与 shape 特性

Add 算子需要重点关注 dtype、shape 和 broadcast 的组合影响。当前测试覆盖了以下主要 dtype：

| dtype | 测试关注点 |
|---|---|
| FLOAT32 | 基础浮点路径、特殊值、精度分析 |
| FLOAT16 | 低精度浮点输入输出、舍入误差 |
| BF16 | bfloat16 路径、较低尾数精度 |
| INT32 / INT64 | 整数加法、整数 alpha |
| INT8 / UINT8 | 小整数类型、tiling 分支 |
| BOOL | bool 输入、bool scalar |
| 混合 dtype | FLOAT16 + FLOAT32、FLOAT32 + BF16 等类型提升路径 |

shape 方面，Add 支持同 shape 输入和 broadcasting。测试中重点关注：

- 同 shape：例如 `{2, 3}` + `{2, 3}`；
- 一维 broadcast：例如 `{2, 3}` + `{3}`；
- 三维 broadcast：例如 `{2, 1, 3}` + `{1, 4, 3}`；
- scalar + tensor：主要用于 `Adds` 和 V3 接口；
- 原地更新：输出写回 selfRef 或 other；
- 非法 shape：用于覆盖参数检查分支。

### 1.3 数学性质与风险点

Add 算子虽然数学形式简单，但在算子实现和端到端测试中存在以下值得关注的性质：

1. **alpha 参数改变计算路径**  
   `alpha=1` 是普通加法；`alpha=0` 会消除 `other` 贡献；负数 alpha 接近减法语义；小数 alpha 会引入乘法和浮点表示误差。

2. **broadcast 会影响 shape 推导和 tiling**  
   不同 broadcast 形态会触发不同 host tiling 路径，是提升覆盖率的重要输入维度。

3. **低精度 dtype 容易产生舍入误差**  
   FLOAT16 和 BF16 有效位数有限，需要使用更合理的容差，而不能直接套用 FLOAT32 的误差阈值。

4. **特殊值需要单独判断**  
   NaN、Inf 不能用普通绝对误差比较，需要使用 `std::isnan`、`std::isinf` 和符号判断。

5. **接近值相减会放大相对误差**  
   Add 在 `self + alpha * other` 场景下可能退化为接近值相减，出现 catastrophic cancellation。

------

## 二、测试策略与用例设计

### 2.1 总体策略

本次测试采用端到端 example 测试策略：构造输入 Tensor / Scalar，调用 aclnn Add 系列接口，拷回 NPU 结果，在 CPU 侧计算参考值，并逐元素比较 actual 与 expected。

整体测试流程如下：

```text
构造输入数据
→ 创建 aclTensor / aclScalar
→ 调用 GetWorkspaceSize
→ 分配 workspace
→ 调用 aclnn API
→ 同步 stream
→ 拷回 NPU 输出
→ CPU 计算 expected
→ actual 与 expected 逐元素比较
→ 输出 PASS / FAIL
```

测试代码不只打印结果，而是实现了完整的 CPU Oracle 和自动判定逻辑。

### 2.2 Oracle 设计

CPU 侧参考实现按 API 类型分别计算。

对于 `aclnnAdd` 和 `aclnnInplaceAdd`：

```cpp
expected[i] = self[broadcast_i] + alpha * other[broadcast_i];
```

对于 `aclnnAdds` 和 `aclnnInplaceAdds`：

```cpp
expected[i] = self[i] + alpha * scalar_other;
```

对于 `aclnnAddV3` 和 `aclnnInplaceAddV3`：

```cpp
expected[i] = scalar_self + alpha * other[i];
```

为了避免 FP16、BF16 和整数输入直接使用 double 字面量导致 Oracle 与实际输入不一致，测试中先按 ACL dtype 对输入进行 pack，再 unpack 回 double，保证 CPU 侧 expected 与设备侧输入基线一致。

### 2.3 精度阈值设置

不同 dtype 的误差阈值不同：

| 输出 dtype | 阈值策略 |
|---|---|
| FLOAT32 | `atol=1e-6, rtol=1e-6` |
| FLOAT16 | `atol=1e-3, rtol=1e-3` |
| BF16 | `atol=1e-2, rtol=1e-2` |
| INT32 / INT64 / INT8 / UINT8 | 精确匹配 |
| BOOL | true / false 结果一致 |
| NaN / Inf | 使用 `std::isnan` / `std::isinf` 判断 |

误差判断公式为：

```cpp
abs(actual - expected) <= atol + rtol * abs(expected)
```

### 2.4 用例分类

#### 2.4.1 API 覆盖用例

| API | 用例设计 |
|---|---|
| `aclnnAdd` | FLOAT32 基础加法、broadcast、整数、bool、FP16、BF16、混合 dtype、特殊值 |
| `aclnnAdds` | Tensor + Scalar，覆盖 float scalar 和 bool scalar |
| `aclnnInplaceAdd` | selfRef 原地 Tensor + Tensor，覆盖 broadcast |
| `aclnnInplaceAdds` | selfRef 原地 Tensor + Scalar，覆盖 INT32 |
| `aclnnAddV3` | Scalar + Tensor，覆盖 FLOAT32、负数 alpha、INT8 |
| `aclnnInplaceAddV3` | Scalar + Tensor，原地更新 other Tensor |

#### 2.4.2 alpha 覆盖用例

| alpha | 用例目的 |
|---|---|
| `1` | 普通加法路径 |
| `0` | 验证 `other` 被缩放为 0 |
| `-0.5` | 负数缩放，近似减法路径 |
| `1.25` | 非 1 浮点缩放 |
| `0.1 / 0.3` | 小数 alpha 精度风险 |
| `2` | 整数缩放路径 |
| bool `true` | BOOL scalar 参与计算 |

#### 2.4.3 dtype 覆盖用例

| dtype | 用例目的 |
|---|---|
| FLOAT32 | 主浮点路径、精度风险、特殊值 |
| FLOAT16 | 低精度输入输出 |
| BF16 | bfloat16 输入输出 |
| INT32 | 整数 alpha 路径 |
| INT64 | 长整数路径 |
| INT8 / UINT8 | 小整数 dtype 分支 |
| BOOL | bool 输入和 bool scalar |
| 混合 dtype | 类型提升与输出类型验证 |

#### 2.4.4 shape 覆盖用例

| shape 类型 | 示例 |
|---|---|
| 同 shape | `{2, 3}` + `{2, 3}` |
| 一维 Tensor | `{4}`、`{5}` |
| 二维 broadcast | `{2, 3}` + `{3}` |
| 三维 broadcast | `{2, 1, 3}` + `{1, 4, 3}` |
| 较大 Tensor | `{64, 64}` |
| V3 scalar + tensor | scalar self + `{4}` tensor |
| 原地 broadcast | selfRef `{2, 3}`，other `{3}` |
| 非法 broadcast | `{2, 3}` + `{2, 2}` |

#### 2.4.5 异常用例

异常用例主要覆盖参数检查分支。考虑不同 CANN 版本错误码可能存在差异，负例采用“返回非 `ACL_SUCCESS` 即通过”的策略。

| 异常场景 | 预期行为 |
|---|---|
| nullptr 输入 | 返回非 `ACL_SUCCESS` |
| 输入无法 broadcast | 返回非 `ACL_SUCCESS` |
| 输出 shape 与推导结果不一致 | 返回非 `ACL_SUCCESS` |
| INT32 输入使用 FLOAT alpha | 返回非 `ACL_SUCCESS` |
| Inplace selfRef shape 非法 | 返回非 `ACL_SUCCESS` |
| V3 输出 shape 非法 | 返回非 `ACL_SUCCESS` |
| V3 BOOL dtype 不支持场景 | 返回非 `ACL_SUCCESS` |

------

## 三、覆盖率分析

### 3.1 覆盖率测量方法

覆盖率通过 `--cov` 参数编译并在运行 example 后使用 gcov 统计。核心命令如下：

```bash
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-*.run
bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

覆盖率统计命令：

```bash
gcov -b -c $(find build -name "aclnn_add.cpp.gcda" | head -1)
gcov -b -c $(find build -name "aclnn_add_v3.cpp.gcda" | head -1)
gcov -b -c $(find build -name "add.cpp.gcda" | grep "op_api" | head -1)
gcov -b -c $(find build -name "add_tiling_arch35.cpp.gcda" | head -1)
```

### 3.2 评分文件覆盖率结果

| 文件 | 行覆盖率 | 分支覆盖率 | Calls 覆盖率 | 说明 |
|---|---:|---:|---:|---|
| `math/add/op_api/aclnn_add.cpp` | 64.69% | 37.19% | 38.73% | 普通 Add / Adds / Inplace 路径 |
| `math/add/op_api/aclnn_add_v3.cpp` | 85.71% | 45.54% | 45.25% | V3 API 路径 |
| `math/add/op_api/add.cpp` | 44.07% | 18.18% | 22.78% | L0 Add 路由与 dtype 支持判断 |
| `math/add/op_host/arch35/add_tiling_arch35.cpp` | 84.95% | 54.17% | 28.18% | Host tiling / shape / broadcast 路径 |

按文件行数和分支数加权估算：

| 指标 | 加权估算值 |
|---|---:|
| 综合行覆盖率 | 约 68.99% |
| 综合分支覆盖率 | 约 37.93% |

### 3.3 覆盖率文件清单

最终提交中建议保留以下关键 `.gcda/.gcno` 文件：

```text
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcno

build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcno

build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcno

build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcda
build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcno
```

### 3.4 覆盖率结果分析

#### 3.4.1 V3 路径覆盖较好

`aclnn_add_v3.cpp` 行覆盖率达到 85.71%，说明 `aclnnAddV3` 和 `aclnnInplaceAddV3` 已经被有效触达。测试中设计了 scalar + tensor、负数 alpha、int8 V3 等用例，覆盖了 V3 的主要逻辑路径。

#### 3.4.2 Host tiling 路径覆盖较好

`add_tiling_arch35.cpp` 行覆盖率达到 84.95%，分支覆盖率达到 54.17%。这说明 `math/add/CMakeLists.txt` 中对 `ascend910_93 -> arch35` 的映射修复是有效的，同时 dtype、shape、broadcast 用例能够触发 host tiling 逻辑。

#### 3.4.3 普通 Add 主路径覆盖中等偏上

`aclnn_add.cpp` 行覆盖率为 64.69%，覆盖了 `aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnInplaceAdds` 及部分参数检查路径。由于该文件包含大量 dtype、format、shape 和异常组合判断，分支覆盖仍有提升空间。

#### 3.4.4 `add.cpp` 是当前短板

`add.cpp` 行覆盖率为 44.07%，分支覆盖率为 18.18%。该文件更多涉及底层路由、AiCore / AiCpu 选择、op binary fallback、dtype 支持判断等逻辑。部分分支依赖底层运行环境或较难通过普通 example 构造，因此本轮覆盖相对较低。

### 3.5 未覆盖部分归因

当前未覆盖部分主要来自：

1. COMPLEX、STRING 等复杂 dtype 未纳入本轮测试；
2. 部分内部 fallback 和平台能力判断依赖运行环境；
3. 某些异常路径可能需要更底层的 mock 或特殊构造才能稳定触发；
4. `add.cpp` 中部分路由分支并非单纯通过普通输入 shape / dtype 就能触达。

------

## 四、精度分析

### 4.1 误差度量方式

本次测试使用绝对误差和相对误差联合判断：

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

其中：

| dtype | 阈值 |
|---|---|
| FLOAT32 | `atol=1e-6, rtol=1e-6` |
| FLOAT16 | `atol=1e-3, rtol=1e-3` |
| BF16 | `atol=1e-2, rtol=1e-2` |
| 整数类型 | 精确匹配 |
| BOOL | true / false 匹配 |

特殊值单独判断：

- NaN 使用 `std::isnan`；
- Inf 使用 `std::isinf` 并检查符号；
- 整数类型不使用容差，直接精确比较。

### 4.2 CPU Oracle 的选择依据

CPU Oracle 使用 double 作为中间计算精度，可以降低 CPU 参考值自身的舍入误差。对于 FLOAT16、BF16 和整数输入，测试代码先将 host 字面量按照目标 ACL dtype pack，再 unpack 回 double 参与 expected 计算，确保 CPU Oracle 的输入值与 NPU 真实输入保持一致。

这种方式可以避免以下误判：

- 用 double 字面量直接作为 FP16 / BF16 参考值；
- 忽略输入在进入 NPU 前已经发生的 dtype 量化；
- 整数输入因四舍五入、截断或溢出导致 Oracle 不一致。

### 4.3 大数 + 小数

测试场景：

```text
self  = [1e10, 1e10]
other = [1e-5, -1e-5]
alpha = 1
```

数学期望值接近：

```text
[10000000000.00001, 9999999999.99999]
```

但 FLOAT32 在 `1e10` 附近的相邻可表示数间隔远大于 `1e-5`，因此小数增量会被舍入吞没，实际结果通常仍表现为 `1e10` 附近的同一可表示值。

该现象不是 Add 算子实现错误，而是 IEEE754 浮点格式限制。FLOAT32 只有约 24 位有效二进制位，数值越大，相邻浮点数间距越大，当小数增量小于当前数量级的 ULP 时，加法无法保留该增量。

### 4.4 正负抵消

测试场景：

```text
self  = [1.0000001, 2.0000001]
other = [-1.0, -2.0]
alpha = 1
```

数学期望值接近：

```text
[0.0000001, 0.0000001]
```

该场景会触发接近值相减。两个接近的浮点数相减时，高位有效数字互相抵消，结果主要由低位尾数组成。输入值在 FLOAT32 中已经经过舍入，抵消后相对误差会被放大。

本场景中绝对误差通常较小，但相对误差可能较明显，因此采用绝对误差和相对误差联合判定。

### 4.5 alpha 小数误差

Add 的实际计算链路包含：

```text
self + alpha * other
```

当 `alpha` 是 `0.1`、`0.3` 这类十进制小数时，它们无法被二进制浮点精确表示，会带来：

1. alpha 表示误差；
2. `alpha * other` 乘法舍入误差；
3. 乘法结果参与加法时的二次舍入；
4. 在 FLOAT16 / BF16 中误差进一步放大。

因此本次测试专门覆盖小数 alpha，而不只覆盖 `alpha=1`。

### 4.6 FLOAT16 / BF16 精度表现

FLOAT16 和 BF16 都是低精度浮点格式，但误差特点不同：

- FLOAT16 尾数位多于 BF16，但指数范围较小；
- BF16 指数范围接近 FLOAT32，但尾数更短；
- BF16 在大范围数值中更不容易溢出，但有效数字较少；
- 混合 dtype 场景中还可能出现类型提升和输出 dtype 回写误差。

因此测试对 FLOAT16 采用 `1e-3` 级容差，对 BF16 采用 `1e-2` 级容差。

### 4.7 NaN / Inf 特殊值

特殊值测试主要验证 Add 对 NaN 和 Inf 的传播行为：

- 输入为 NaN 时，对应输出应保持 NaN 语义；
- 输入为正 Inf / 负 Inf 时，输出应保持 Inf 及符号传播规则；
- `alpha=0` 时也需要关注 `alpha * other` 对特殊值传播的影响。

测试中使用 `std::isnan` 和 `std::isinf` 判断，而不是普通容差判断。

------

## 五、反思与改进

### 5.1 当前测试优势

1. **API 覆盖较完整**  
   覆盖了 Add 的 6 类主要 aclnn API，包括普通 Add、scalar Add、原地 Add 和 V3 API。

2. **alpha 维度覆盖较充分**  
   覆盖了 `alpha=0`、`alpha=1`、负数 alpha、小数 alpha、整数 alpha 等场景，避免只测试普通加法。

3. **dtype 与 shape 覆盖较广**  
   覆盖 FLOAT32、FLOAT16、BF16、INT32、INT64、INT8、UINT8、BOOL 和混合 dtype，同时覆盖同 shape、broadcast、原地更新、V3 scalar + tensor 等 shape 形态。

4. **结果验证方式较可靠**  
   测试代码实现 CPU Oracle，并逐元素比较 actual 与 expected，避免只打印结果。

5. **host tiling 覆盖修复有效**  
   通过修改 `CMakeLists.txt`，使 `add_tiling_arch35.cpp` 能在 `ascend910_93` 下参与覆盖率统计。

### 5.2 测试盲区

1. **`add.cpp` 分支覆盖率偏低**  
   该文件涉及底层路由、fallback、op binary、AiCore / AiCpu 选择等逻辑，不是所有路径都能通过普通 example 输入稳定触发。

2. **复杂 dtype 未完全覆盖**  
   COMPLEX、STRING 等类型未纳入本轮测试。主要原因是构造和 Oracle 验证复杂度较高，且当前测试重点放在常见数值 dtype 上。

3. **精度日志仍可进一步量化**  
   当前精度分析覆盖了典型场景和成因，但若时间更充足，可以补充每个精度 case 的 actual、expected、绝对误差、相对误差表格。

4. **异常路径仍可细分**  
   目前负例采用非 `ACL_SUCCESS` 判断，稳定性较好，但未进一步区分每个错误码对应的内部路径。

### 5.3 后续改进方向

如果继续优化，可以从以下方向扩展：

1. 针对 `add.cpp` 补充更多底层路由和 fallback 场景；
2. 增加 complex dtype 的端到端测试，并实现 complex CPU Oracle；
3. 增加非连续 Tensor、特殊 stride、更多高维 broadcast；
4. 对每类异常输入分别记录具体错误码，形成更细粒度异常路径覆盖；
5. 对精度场景增加结构化日志，输出 actual、expected、abs error、rel error；
6. 将测试用例按照 API / dtype / shape / alpha 自动参数化，减少重复代码。

### 5.4 方法论总结

本次测试中最重要的经验是：算子测试不能只追求“能跑”，还需要明确每个 case 对应的覆盖目标和风险目标。对于 Add 这种看似简单的算子，`alpha`、broadcast、dtype 提升、低精度浮点和 V3 API 都可能引入新的测试路径。只有将 API 维度、dtype 维度、shape 维度和数值风险维度组合起来，才能形成有效的端到端测试方案。

------

# 附录：提交文件建议

最终提交包建议包含：

```text
<队名>/
├── test_aclnn_add.cpp
├── report.md
└── build/
    └── 仅保留评分所需 .gcda / .gcno 文件
```

关键覆盖率文件包括：

```text
aclnn_add.cpp.gcda / aclnn_add.cpp.gcno
aclnn_add_v3.cpp.gcda / aclnn_add_v3.cpp.gcno
add.cpp.gcda / add.cpp.gcno
add_tiling_arch35.cpp.gcda / add_tiling_arch35.cpp.gcno
```

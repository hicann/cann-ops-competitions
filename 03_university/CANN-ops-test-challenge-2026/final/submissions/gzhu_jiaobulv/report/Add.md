---

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "蕉不绿队"

team_members:

- "成员1：姚杰涛-广州大学"
- "成员2：张欢-广州大学"
- "成员3：陈贝宁-广州大学"

operator_name: "Add"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

---

# Add 算子最终测试结果报告

> 本报告基于当前版本 `test_aclnn_add.cpp` 的执行结果，以及 4 个评分文件对应的 `.gcov` 产物整理形成。与上一版报告相比，这一版结论以本轮真实覆盖率为准，重点说明当前测试已经打到的主路径、仍未进入的分支，以及导致综合覆盖率下降的核心原因。

---

## 一、结果概览

本次实际统计的 4 个评分文件及覆盖率结果如下：

| 文件  | 行数  | 行覆盖率 | 分支数 | 分支覆盖率 | 至少命中一次 | 调用覆盖率 |
| --- | --- | --- | --- | --- | --- | --- |
| `math/add/op_api/add.cpp` | 55  | 89.09% | 263 | 74.90% | 41.44% | 83.10% |
| `math/add/op_api/aclnn_add.cpp` | 288 | 71.53% | 1581 | 54.52% | 30.80% | 62.67% |
| `math/add/op_host/arch35/add_tiling_arch35.cpp` | 93  | 0.00% | 182 | 0.00% | 0.00% | 0.00% |
| `math/add/op_api/aclnn_add_v3.cpp` | 76  | 90.79% | 440 | 64.55% | 37.50% | 73.20% |

按文件行数和分支数加权后，本轮综合结果约为：

- 综合行覆盖率：`63.28%`
- 综合分支覆盖率：`54.46%`
- 综合“至少命中一次”分支比例：`30.86%`
- 综合调用覆盖率：`60.78%`

从结果可以直接看出：

1. `add.cpp`、`aclnn_add.cpp`、`aclnn_add_v3.cpp` 三个 `op_api` 文件的覆盖率已经明显提升，主流程、异常输入和部分边界路径已经进入。
2. 当前综合覆盖率被 `add_tiling_arch35.cpp` 的 `0.00%` 明显拉低。
3. 因此，这一轮测试的主问题已经不再是 `op_api` 主路径完全没打到，而是 `host tiling` 路径完全没有进入。

---

## 二、结合 `.gcov` 的逐文件分析

### 1. `math/add/op_api/add.cpp`

该文件当前表现最好，行覆盖率达到 `89.09%`，分支覆盖率达到 `74.90%`。结合 `.gcov` 可见，以下关键路径已经被覆盖：

- `Add` 的 broadcast 成功与失败路径均已进入。
- `AiCore` 与 `AiCpu` 两条调度路径均被打到。
- mixed dtype 分支已经命中，`float16 + float`、`float + float16`、`bf16 + float` 等输出转为 `float` 的逻辑被实际执行。
- `l0op::Add` 的非法 broadcast 失败路径已进入。
- `AddAiCore`、`AddAiCpu` 两个底层调度函数都已被执行。

当前主要剩余缺口有：

- `DAV_3102` 对应的 `ASCEND610LITE_AICORE_DTYPE_SUPPORT_LIST` 分支未命中。
- `IsAddSupportNonContiguous` 只有 `IsRegBase()==false` 的返回路径，未体现真正的非连续输入支持分支。
- `AddInplace` 虽然进入了失败分支校验，但 `.gcov` 显示成功调度到 `AddAiCore` 或 `AddAiCpu` 的路径并未命中，说明当前更多是在测它的失败保护，而不是成功执行。

这说明 `add.cpp` 这一层的核心调度逻辑已经覆盖得比较充分，但架构特定分支与 `L0 AddInplace` 成功路径仍有缺口。

### 2. `math/add/op_api/aclnn_add.cpp`

该文件当前行覆盖率 `71.53%`，分支覆盖率 `54.52%`，已经从“只打到少量主路径”提升到了“主路径 + 部分异常路径都进入”的状态。结合 `.gcov`，当前已经明确命中的内容包括：

- `CheckNotNull` 的多种空指针失败分支。
- `CheckPromoteType` 中 `alpha` 可转换 / 不可转换、输出类型可转换 / 不可转换等路径。
- `CheckInplace` 中 broadcast 推导、shape 检查成功与失败路径。
- `aclnnAddGetWorkspaceSize` 主流程。
- `aclnnAddsGetWorkspaceSize`、`aclnnInplaceAddGetWorkspaceSize`、`aclnnInplaceAddsGetWorkspaceSize` 等接口入口。
- `alpha==1` 直走 `Add`、`Axpy` 路径、`Mul+Add` fallback 路径都已经命中。
- `bool`、mixed dtype、空 tensor、非法 shape、非法 alpha 等测试都对该文件分支覆盖有直接贡献。

`.gcov` 同时也显示出当前还没有覆盖到的典型区域：

- `GetScalarDefaultDtype`
- `InnerTypeToComplexType`
- `CombineCategoriesWithComplex`
- `GetCastedFloat`

这些函数全部是 `called 0`，说明当前复杂数相关、部分特殊 scalar 推导路径没有真正进入。

另外，`.gcov` 还显示：

- `aclnnInplaceAdd` 运行接口本身未被执行。
- `aclnnInplaceAdds` 运行接口本身未被执行。

这说明当前测试更多打到了 `GetWorkspaceSize` 和参数校验入口，但对应的最终执行封装函数并没有留下运行覆盖。也就是说，`Inplace` 系列“前半段”覆盖得不错，“执行落地”这一小段还需要补。

### 3. `math/add/op_api/aclnn_add_v3.cpp`

该文件当前行覆盖率 `90.79%`，分支覆盖率 `64.55%`，是本轮覆盖最好的文件之一。结合 `.gcov` 可确认：

- `CheckNotNull`、`CheckShape`、`CheckParams` 的主要真/假分支已被覆盖。
- `PromoteTypeScalar` 的多条推导路径被实际命中。
- `alpha==1` 直走 `Add`、`Axpy` 路径、`Mul+Add` 路径都已执行。
- 空 tensor 快速返回路径已命中。
- `promote -> out` 不可转换、shape 不匹配、max dim exceeded、不支持 dtype 等失败路径已进入。
- `AddV3` 的标量 + tensor 主语义已经测透到了较深位置。

当前明显未覆盖的点主要有：

- `aclnnInplaceAddV3` 运行接口 `called 0`。
- `promoteType == DT_BOOL` 这类很窄的分支没有打到。
- `promoteType == DT_UNDEFINED` 的失败路径没有打到。

因此，`aclnn_add_v3.cpp` 当前的状态可以判断为：主功能、主要异常输入、主要类型约束都已经覆盖到位，剩余缺口主要集中在非常窄的特殊类型分支和 `InplaceAddV3` 的最终执行封装。

### 4. `math/add/op_host/arch35/add_tiling_arch35.cpp`

这是本轮最关键的问题文件。当前结果是：

- 行覆盖率：`0.00%`
- 分支覆盖率：`0.00%`
- 调用覆盖率：`0.00%`

从 `.gcov` 可见，该文件中所有关键函数均为 `called 0`，包括：

- `AddTiling::IsMixedDtype`
- `AddTiling::CheckDtype`
- `AddTiling::DoOpTiling`
- `TilingForAdd`
- `TilingPrepareForAdd`

这说明本轮执行过程中，`host tiling` 路径完全没有进入。

基于 `.gcov` 可以做出一个明确判断：当前测试集虽然已经把 `op_api` 层覆盖得比较充分，但没有驱动到 `arch35` 对应的 tiling 编译/解析/模板分发路径。因此上一版报告里关于 `add_tiling_arch35.cpp` 的高覆盖结论，在本轮结果下已经不再成立，最终报告必须以本轮 `0.00%` 为准。

---

## 三、本轮测试内容能说明什么

从 `test_aclnn_add.cpp` 当前用例内容看，这一轮测试已经覆盖了以下几类核心场景：

### 1. 常规正确性路径

- `float32 broadcast + alpha!=1`
- `int32 + alpha=2`
- `float16 + float32`
- `float32 + float16`
- `double + alpha!=1` 触发 `Mul+Add` 路径
- `alpha=0` 的高维 broadcast
- `bool` 特殊分支
- `int8`、`uint8`、`int64` 边界值
- `NaN / Inf` 特殊浮点值

这些用例保证了 Add 的主数值语义已经不是空白覆盖。

### 2. API 变体路径

- `Adds`
- `InplaceAdd`
- `InplaceAdds`
- `AddV3`
- `InplaceAddV3`

并且对其中相当一部分接口补了：

- `alpha==1`
- `负 alpha`
- mixed dtype
- out dtype cast
- 空 tensor
- 非 ND format

因此当前 `op_api` 三个文件的覆盖提升是有真实用例支撑的，不是靠少量“撞到主函数”得来的。

### 3. 失败路径与边界路径

当前测试还显式包含了大量失败场景，例如：

- nullptr 输入
- `alpha` 为空
- 非法 broadcast
- out shape 不匹配
- max dim exceeded
- `alpha` 不可转换
- `promote -> out` 不可转换
- `AddV3` 不支持 dtype
- `L0 Add` / `L0 AddInplace` 非法输入

这也是为什么当前 `aclnn_add.cpp` 和 `aclnn_add_v3.cpp` 的分支覆盖率能比上一版明显提高。

---

## 五、结论

Add 测试的真实状态可以概括为：

- `op_api` 层已经从“基础覆盖”进入到“主路径 + 异常路径都较充分”的阶段。
- `add.cpp`、`aclnn_add.cpp`、`aclnn_add_v3.cpp` 的结果说明当前测试方案对 eager / op_api 层是有效的。
- 最大短板已经不是普通 Add/Adds/AddV3 的数值正确性，而是 `host tiling` 路径完全未进入。
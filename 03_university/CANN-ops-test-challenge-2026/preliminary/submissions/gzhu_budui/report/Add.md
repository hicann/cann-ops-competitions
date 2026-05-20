# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====
team_name: "不队"
team_members:
- "杨金鹏-广州大学"
- "刘畅-广州大学"
operator_name: "Add"
operator_library: "cann-ops-math"
report_date: "2026-04-28"
------


# Add 算子测试报告


## 一、算子理解

Add 算子的核心语义是逐元素加法，可表达为 `out = self + alpha * other`。在本次预赛代码中，我们不仅覆盖了基础 Tensor-Tensor 加法，还系统验证了如下能力：
* **API 变体**：`aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnAddV3`、`aclnnInplaceAddV3`。
* **数据类型广覆盖**：FP32/FP16/BF16/INT32/INT64/INT8/UINT8/BOOL 以及混合精度（如 FP16+FP32）。
* **广播与空 Tensor 场景**：支持不同 shape 的广播计算、空 Tensor 快路径（workspace=0）探测。

从底层执行角度看，Add 的关键风险点不是“算不算得出来”，而是“在不同 dtype 与 alpha 条件下，是否走到正确分发路径并保持数值一致性”。因此，本报告重点围绕**路由触发、异常拦截、精度对齐**三条主线展开。

------

## 二、测试策略与用例设计

本次 `test_aclnn_add.cpp` 采用“正向正确性 + 反向防御性”双轨策略，主程序统一汇总执行并输出 `Summary: total/pass/fail`。

**1. API 与语义全链路覆盖**
围绕 Add 家族 6 套接口设计了大批命名用例，覆盖：
* Tensor-Tensor 加法、Tensor-Scalar 加法；
* Inplace 非 Inplace 一致性；
* V3 版本下 `self` 为标量时的行为与类型组合。

**2. 关键路由与边界值触发**
通过 alpha 与 dtype 组合构造典型分支：
* `alpha=1/0/负数/近1` 等关键值；
* FP16/BF16/Double 路由状态探测；
* `BOOL + BOOL` 与 `BOOL -> INT` 的类型行为校验；
* `INT32 + FLOAT` 等提升（promote）路径验证。

**3. 广播、空张量与形状一致性**
* 覆盖常规同形状、可广播形状、不匹配形状；
* 针对空张量专门检查 `GetWorkspaceSize` 的快路径行为；
* 通过广播 offset 的 CPU 参考实现逐元素比对输出。

**4. Error-Path 防御性测试**
集中验证 `GetWorkspaceSize` 的参数校验与拦截能力：
* `nullptr` 输入；
* 9 维（超过限制）输入；
* shape mismatch；
* 不支持 dtype 组合。

------

## 三、覆盖率分析

当前预赛目录仅包含测试代码，未附带流水线导出的底层源码覆盖率报表。本阶段采用“**测试意图覆盖**”方式给出分析：

| 目标层 | 覆盖情况 | 说明 |
| :--- | :--- | :--- |
| API 入口层（Add/Adds/Inplace/V3） | 高 | 主函数集中调用各接口，并覆盖正向与异常场景。 |
| dtype/alpha 分发层 | 高 | 通过 FP/INT/BOOL、多种 alpha 组合触发主要分支。 |
| 广播与形状检查层 | 高 | 含可广播、不可广播、空 Tensor、维度越界场景。 |
| 极端系统异常层（如设备内存分配失败） | 中低 | 受运行环境限制，难以稳定构造。 |

**覆盖率总结**：就测试设计而言，已覆盖 Add 题目实现最关键的功能与防御性路径；若后续需要提交量化覆盖率分数，建议补充 gcov/lcov 或组委会统一覆盖率脚本输出。

------

## 四、精度分析

Add 的精度风险主要集中在浮点舍入与整数回绕。本次重点验证如下场景：

### 场景一：大数与小数混加
* **策略**：构造 `add_float_large_magnitude` 类型用例。
* **结论**：结果满足容差阈值，说明算子在高量级输入下保持与 CPU `double` 参考一致的数值趋势。

### 场景二：NaN/Inf 传播
* **策略**：专门设置 `add_float_nan_inf` 用例。
* **结论**：NaN 与 Inf 的传播行为符合 IEEE-754 预期，未出现异常吞噬或错误归零。

### 场景三：混合精度一致性
* **策略**：验证 `FP16+FP32` 与 `FP32+FP16` 两类混合输入。
* **结论**：输出与参考结果在设定容差内一致，说明混合精度路径稳定。

### 场景四：布尔/整数语义正确性
* **策略**：测试 `adds_bool_out_bool_exact`、`add_int64_alpha3_exact` 等用例。
* **结论**：布尔逻辑与整数精确计算行为符合预期，无额外精度损失。

------

## 五、反思与改进

1. **路由复杂度高，参数必须正交设计**：Add 的行为受 `dtype × alpha × shape` 联合影响明显，后续应继续保持“关键参数笛卡尔覆盖”。
2. **异常路径价值高于表面通过率**：`nullptr/shape/rank/dtype` 防御测试能显著提升提交代码鲁棒性，建议保留为回归集。
3. **建议补齐量化覆盖率工件**：为决赛评分可追溯，建议在 CI 中自动产出覆盖率表并与报告联动更新。

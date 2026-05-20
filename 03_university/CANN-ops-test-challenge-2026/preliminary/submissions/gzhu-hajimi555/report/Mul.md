# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====
team_name: "哈基米555"
team_members:
- "庞浩民-广州大学"
- "兰晨阳-广州大学"
- "罗景辉-广州大学"
operator_name: "Mul"
operator_library: "cann-ops-math"
report_date: "2026-04-28"
------

# Mul 算子测试报告

## 一、算子理解
Mul 算子用于逐元素乘法运算，包含 Tensor-Tensor 与 Tensor-Scalar（含 Inplace 模式）场景。其数学定义可抽象为：
* `out = self * other`（Tensor-Tensor 乘法）
* `out = self * scalar`（Tensor-Scalar 乘法）

Mul 的工程风险主要集中在以下维度：
* 数据类型极多，包含 16 种数据类型的混合精度转换与复数类型推导。
* 涉及连续（Contiguous）与非连续（Non-Contiguous）内存排布的底层硬件（AiCore/AiCpu）分发路由。
* 广播机制在各种维度下的兼容性校验。

------
## 二、测试策略与用例设计
`test_aclnn_mul.cpp` 采用基础校验与大规模分支触发覆盖结合的测试策略。

**1. 四类 API 形态全面覆盖**
* 覆盖 `aclnnMul` 与 `aclnnInplaceMul` 的两端 Tensor 运算。
* 覆盖 `aclnnMuls` 与 `aclnnInplaceMuls` 的 Tensor-Scalar 运算。

**2. 极限值与边界条件设计**
* 包含 `NaN`（非数字）、`Inf`（无穷大）及其符号组合验证。
* 包含 `0.0f`、`-0.0f`、`-1.0f` 以及 `1e15` 大数边界。
* 验证了 `1024` 元素（Shape 为 `[32, 32]`）的大张量（Large Tensor）乘法。

**3. 异常路径与鲁棒性验证**
* 集中验证全量参数为 `nullptr` 与单个参数为 `nullptr` 的情况。
* 验证维度超过 8 维（测试传入 9 维）的越界异常。
* 验证输出张量 shape 与广播推理 shape 不一致的拦截异常。

**4. C++ 内部实现分支白盒覆盖**
* 利用 `RunBranchTriggerTest` 精准打击底层类型推导分支，例如 `InferTensorScalarDtype` 函数路径。
* 覆盖了 `CombineCategoriesWithComplex` 的复数提升分支。

------
## 三、覆盖率分析

| 目标层 | 覆盖情况 | 说明 |
| :--- | :--- | :--- |
| API 接口层 | 高 | Tensor-Tensor/Tensor-Scalar 均包含正向、异常及 Inplace 用例。 |
| 数据类型与转换 | 极高 | 覆盖 16 种 DType，涵盖 FLOAT/INT 与 BOOL、COMPLEX 的混合提升（PromoteType）测试。 |
| 内存与 Shape 语义 | 高 | 触达 1D-5D 维度、广播机制、空张量（Empty Tensor）及修改步长（Strides）的非连续张量测试。 |
| 底层算子调度（L0） | 高 | 包含条件编译的 Ascend910B、950、610LITE 平台的 L0 级算子分发支持校验。 |

**覆盖率总结**：Mul 算子测试针对性极强，特别是大量使用 `RunBranchTriggerTest` 构建类型与 Shape 组合，高密度覆盖了底层 `mul.cpp` 与 `aclnn_mul.cpp` 中的底层推断逻辑。

------
## 四、精度分析
Mul 的精度校验主要由自定义的 `VerifyFloat`、`VerifyInt32` 以及 `VerifyMulScalarFloat` 函数保障。

### 场景一：常规浮点乘法
* **策略**：采用绝对误差 `atol = 1e-5` 和相对误差 `rtol = 1e-5` 双重校验，判定条件为 `std::fabs(act - exp) > atol + rtol * std::fabs(exp)`。
* **结论**：算法跳过了对 `NaN` 的严格数值比对，且针对同号的无穷大（`Inf`）进行了合理豁免，保障了特殊浮点数学稳定性的验证。

### 场景二：整型数据高精度
* **策略**：使用 `int64_t` 承载 `x1` 与 `x2` 的乘积期望值，随后转为 `int32_t` 进行等值校验。
* **结论**：验证逻辑无精度妥协，杜绝了底层算子可能的隐式浮点转换截断误差。

### 场景三：特殊标量验证
* **策略**：针对 Tensor-Scalar 操作独立封装校验逻辑，使用双精度浮点数 `double` 承接中间计算过程以防溢出。
* **结论**：确保系数放大器在逐元素计算与内存回写阶段数值对齐。

------
## 五、反思与改进
1. **多数类型测试缺失数值校验**：`RunBranchTriggerTest` 函数目前仅验证了接口是否返回 `ACL_SUCCESS`，未提取实际输出进行比对。对于复杂的混合类型组合（如 `COMPLEX64 * FLOAT`），强烈建议增加显式期望值比对逻辑。
2. **精度验证工具链存在局限**：测试用例涵盖了 16 种数据类型，但验证体系仅实现了针对 `FLOAT32` 与 `INT32` 的数学校验。需扩充半精度（BF16/FP16）的专用宽容差检验模块。
3. **未涉及性能基准测试**：当前实现全为功能覆盖型，建议补充极大规模数据下的算子耗时检测。
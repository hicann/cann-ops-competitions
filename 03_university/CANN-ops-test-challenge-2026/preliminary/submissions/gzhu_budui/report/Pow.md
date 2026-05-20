# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====
team_name: "不队"
team_members:
- "杨金鹏-广州大学"
- "刘畅-广州大学"
operator_name: "Pow"
operator_library: "cann-ops-math"
report_date: "2026-04-28"
------


# Pow 算子测试报告


## 一、算子理解

Pow 算子用于幂运算，既包含 Tensor-Scalar / Scalar-Tensor，也包含 Tensor-Tensor（含广播）场景。其数学定义可抽象为：
* `out = pow(self, exponent)`（Tensor-Scalar）
* `out = pow(base, exponentTensor)`（Scalar-Tensor）
* `out = pow(selfTensor, exponentTensor)`（Tensor-Tensor）

Pow 的工程风险高于 Add/Mul，原因在于：
* 输入域更复杂（负底数、分数指数、零指数等）；
* 广播与类型组合更多；
* 数值稳定性与特殊值传播（NaN/Inf）更敏感。

------

## 二、测试策略与用例设计

`test_aclnn_pow.cpp` 采用模块化测试函数 + 统一 `runCase` 汇总，输出 `Total/PASS/FAIL`。

**1. 三类 API 形态全面覆盖**
* Tensor-Scalar：`RunPowTensorScalarCase` 与 Inplace 版本；
* Scalar-Tensor：`RunPowScalarTensorCase`；
* Tensor-Tensor：`RunPowTensorTensor*Case`（含 Inplace 与广播）。

**2. 关键指数/底数组合设计**
在 Tensor-Scalar 与 Scalar-Tensor 侧覆盖：
* `exp=0`、`exp=2`、`exp=0.5`、`exp=-1`；
* `base=1` 等具有数学简化特性的场景。

**3. dtype 覆盖与参考实现一致性**
Tensor-Tensor 侧覆盖：
* Float、Double、Float16；
* Int8/Uint8/Int16/Int32 等整数类型；
* 统一使用 CPU 侧 `std::pow` + 广播索引映射生成 expected，再逐元素校验。

**4. Error-Path 集中验证**
分别构建三套错误用例组：
* `RunPowTensorScalarErrorCases`；
* `RunPowScalarTensorErrorCases`；
* `RunPowTensorTensorErrorCases`。

覆盖异常类型包括 `nullptr`、shape mismatch、广播失败、BOOL 不支持组合等。

------

## 三、覆盖率分析

由于预赛目录未附底层源码覆盖率报表，本节采用测试设计覆盖分析：

| 目标层 | 覆盖情况 | 说明 |
| :--- | :--- | :--- |
| Pow 三类 API 层 | 高 | Tensor-Scalar/Scalar-Tensor/Tensor-Tensor 均有正向与异常用例。 |
| 指数与底数边界语义 | 高 | 0、负数、分数指数及特定 base 场景均有触达。 |
| 广播与类型分发层 | 高 | 浮点、整数、半精度及广播/非广播路径均覆盖。 |
| 稀有系统级异常 | 中低 | 依赖环境故障注入，当前未强制触发。 |

**覆盖率总结**：Pow 测试在“语义边界 + API 分支 + 异常拦截”上完整度较高，能够较好支撑预赛阶段正确性证明。

------

## 四、精度分析

Pow 的精度分析重点在于幂运算放大误差与低精度量化误差。

### 场景一：分数指数与负指数
* **策略**：`exp=0.5`、`exp=-1` 等典型敏感场景。
* **结论**：输出与 CPU `std::pow` 参考值在容差内一致，基础幂运算稳定。

### 场景二：广播下的逐元素幂
* **策略**：`PowTensorTensor_broadcast_float`，使用坐标映射计算期望值。
* **结论**：广播索引与结果对齐正确，未出现错位计算。

### 场景三：Float16 路径误差控制
* **策略**：Float16 输入输出统一转换并采用较宽容差比较。
* **结论**：考虑量化后误差，结果与期望一致，符合半精度特性。

### 场景四：Double 路径高精度校验
* **策略**：`PowTensorTensor_double_aicpu`，容差设置为 `1e-10` 级别。
* **结论**：高精度路径通过，可作为精度基准参考。

------

## 五、反思与改进

1. **Pow 的“数学定义域”测试仍可继续加深**：后续建议补充更多负底数+非整数指数、极大/极小指数场景。
2. **广播索引逻辑是 Pow 测试核心资产**：当前通用索引实现可复用于其他双输入算子。
3. **建议后续增加性能与稳定性维度**：在正确性已达标基础上，补充大规模数据与多轮次压测。

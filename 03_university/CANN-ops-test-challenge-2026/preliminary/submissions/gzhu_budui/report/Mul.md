# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====
team_name: "不队"
team_members:
- "杨金鹏-广州大学"
- "刘畅-广州大学"
operator_name: "Mul"
operator_library: "cann-ops-math"
report_date: "2026-04-28"
------


# Mul 算子测试报告


## 一、算子理解

Mul 算子执行逐元素乘法，基础数学语义为 `out = self * other`，并扩展出 `Muls`（张量乘标量）及 Inplace 版本。在预赛实现中，Mul 的测试重点不止于“乘法结果对不对”，还包括：
* **类型路由**：实数、整数、布尔、复数、FP16/BF16/FP32 混合精度路径；
* **格式与广播行为**：标准 ND、广播计算、格式告警探测；
* **边界与防御性**：空 Tensor 快路径、`nullptr`、维度上限、shape mismatch。

Mul 在工程上的挑战是“广覆盖 + 可复现对比”：既要打到多种执行路径，又要保证 CPU 参考计算与设备结果严格可比。

------

## 二、测试策略与用例设计

`test_aclnn_mul.cpp` 采用大规模回归列表驱动，主函数通过 `totalFailed += RunXXX(...)` 方式组织，覆盖面较广。

**1. 主功能与 API 族覆盖**
覆盖接口包括：
* `aclnnMul`（Tensor-Tensor）；
* `aclnnMuls`（Tensor-Scalar）；
* `aclnnInplaceMul`、`aclnnInplaceMuls`。

**2. dtype 与数值类型广覆盖**
重点覆盖以下数据形态：
* 浮点：FP32 / FP16 / BF16 / Double；
* 整型：INT8/UINT8/INT16/INT32/INT64；
* 布尔：`bool_basic`；
* 复数：`complex64_basic` 与 `complex128_route_probe` 路由探测；
* 混合精度：FP16/FP32、BF16/FP32 双向混合输入。

**3. 广播/格式/空 Tensor 行为验证**
* 广播：如 `float32_broadcast_2x3_1x3`；
* 格式告警：`mul_format_nchw_warning`、`muls_format_nchw_warning`；
* 空 Tensor 快路径：`mul_empty_tensor_fastpath`、`muls_empty_tensor_fastpath` 及 inplace 版本。

**4. 防御性错误用例（Guard）**
集中覆盖 `GetWorkspaceSize` 的异常路径：
* `nullptr` 输入（self/other/out/scalar）；
* 非法广播与 out-shape 不匹配；
* 超过维度限制（9 维）场景；
* inplace 下 shape expand 与非法输入组合。

------

## 三、覆盖率分析

当前仓库为预赛提交目录，未包含底层源码覆盖率统计文件；本节按测试设计维度进行覆盖分析：

| 目标层 | 覆盖情况 | 说明 |
| :--- | :--- | :--- |
| Mul/Muls/Inplace API 层 | 高 | 主流程中逐项调用并统一汇总失败数。 |
| dtype 路由层 | 高 | 实数/整数/布尔/复数/混合精度均有专门用例。 |
| 广播与格式校验层 | 高 | 包含成功广播、广播失败、格式告警与快路径测试。 |
| 系统级极端异常（如 malloc 失败） | 中低 | 依赖特定环境，不易稳定复现。 |

**覆盖率总结**：Mul 测试具备“高广度 + 高可回归性”特征，尤其在 dtype 与 guard path 上覆盖较深，满足预赛阶段对稳定性与完整性的要求。

------

## 四、精度分析

Mul 的精度问题主要体现在低精度量化误差、复数乘法误差累计及混合精度对齐上。

### 场景一：FP16/BF16 量化误差
* **策略**：分别通过 `RunMulFp16Test` 与 `RunMulBf16Test` 进行验证。
* **结论**：FP16 与 BF16 均在预设容差（FP16 更严格、BF16略宽）内通过，量化误差受控。

### 场景二：混合精度乘法一致性
* **策略**：`fp16_fp32_mix_*` 与 `bf16_fp32_mix_*` 双向测试。
* **结论**：输出与按量化后参考值计算的 CPU 结果对齐，混合路径稳定。

### 场景三：整数与布尔精确语义
* **策略**：INT8/16/32/64、UINT8 与 BOOL 独立验证。
* **结论**：整数路径按精确乘法语义执行；布尔路径保持逻辑与预期一致。

### 场景四：复数路径正确性
* **策略**：`complex64_basic` 与 `complex128_route_probe`。
* **结论**：复数实部/虚部结果与参考实现一致，路由探测用例可帮助识别后端支持边界。

------

## 五、反思与改进

1. **Mul 测试已形成“算子族”回归骨架**：同一框架可迁移到 Div/Sub 等算子，建议沉淀公共模板。
2. **低精度 Oracle 需显式量化对齐**：FP16/BF16 的参考值必须基于量化后数据计算，避免误判为算子错误。
3. **建议补充性能与长稳测试**：当前重点在功能正确性，后续可增加大 Tensor 压测与多轮稳定性测试。

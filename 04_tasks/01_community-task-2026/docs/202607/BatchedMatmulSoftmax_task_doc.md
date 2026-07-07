# 7月社区任务-BatchedMatmulSoftmax算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Ascend 950
- **开源仓地址**：[https://gitcode.com/cann/catlass](https://gitcode.com/cann/catlass)
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 任务概述

本算子实现 Batched Matmul 后接行级 Softmax 的融合算子（Batched Matmul + Softmax Fusion）。Batched 输入矩阵经 Matmul 计算后，其输出送入 Softmax 激活函数。该融合方式是 Attention 机制中的核心计算模式。

**算子公式**

对每个 batch $b$ 独立计算：

$$ \mathbf{S}_b = \text{row\_softmax}\big(\mathbf{A}_b \times \mathbf{B}_b\big) $$

其中行级 Softmax 定义为：

$$ \mathbf{S}_b[m, n] = \frac{\exp\big(\mathbf{M}_b[m, n] - \max_{n'}(\mathbf{M}_b[m, n'])\big)} {\displaystyle\sum_{n'=0}^{N-1} \exp\big(\mathbf{M}_b[m, n'] - \max_{n'}(\mathbf{M}_b[m, n'])\big)} $$

其中 $\mathbf{M}_b = \mathbf{A}_b \times \mathbf{B}_b$，$\mathbf{A}_b$ 形状 $(M, K)$，$\mathbf{B}_b$ 形状 $(K, N)$。Matmul 中间结果在算子内部被 Softmax 消费，不对外暴露。

## 核心开发要求及验收标准

### 功能实现要求

1. 实现融合的 Batched Matmul + 行级 Softmax，单次 kernel launch 完成全部计算。
2. 沿 batch 维度独立执行 Matmul + Softmax。
3. 使用 max 归一化确保数值稳定性，防止 exp 溢出。
4. 可以使用多种matmul方案实现功能，从而达成更好性能。

### 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|--------------|------|---------|----------|----------|-------------|-------------|
| A | 输入 | Batched Matmul 左矩阵，形状 (batch, M, K) | RowMajor 布局 | FP16 | ND | 3 | — |
| B | 输入 | Batched Matmul 右矩阵，形状 (batch, K, N) | ColumnMajor 布局 | FP16 | ND | 3 | — |
| S | 输出 | Softmax 输出矩阵，形状 (batch, M, N) | RowMajor 布局 | FP16 | ND | 3 | — |

### 算子约束限制

- 无特殊约束

### 测试标准

1. 基于[CATLASS-optest测试工程](https://gitcode.com/cann/catlass/blob/master/tests/optest/README.md)补充测试交付件，基于[任务测试集](./self_test_case/BatchedMatmulSoftmax/)测试精度通过。
2. 输出optest测试交付件，可使用[catlass-example-to-pytest](https://gitcode.com/cann/catlass/blob/master/.agents/skills/catlass-example-to-pytest/SKILL.md) skill基于样例代码自动生成。

### 性能要求

性能标杆为 torch.bmm + torch.softmax 的小算子拼接方案，算子整体性能需达成1.2倍小算子拼接性能，标杆性能数据已在任务测试集中提供（使用Ascend 950PR硬件）。性能测试时若涉及不同实现方案和TileShape等参数调整，需要备注说明。性能采集使用`msprof op`工具，可以参考[CATLASS样例性能调试](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/evaluation/performance_tools.md)。

### 精度要求

算子计算精度需满足 [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。

## 文档规范要求

1. 算子设计文档需根据[参考模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；
2. 自验证报告需要覆盖所有功能场景，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)，含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可清晰指导算子使用与测试；
3. README 文档内容完整、规范。

## 验收交付件

1, 自测用例、测试结果报告、测试步骤指导文档
2, 算子代码的私仓邀请链接、代码仓路径、分支、算子目录

## PR 申请合入

测试通过后，在 Catlass 代码仓提交 PR 申请，申请将开发完成的算子合入https://gitcode.com/cann/catlass，具体目录参考该PR：https://gitcode.com/cann/catlass/pull/678 。

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[CATLASS创新样例开发流程指南](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/10_innovative_example_development_guide.md)
2. 课程类：[Ascend C在线课程](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
3. 参考样例：[https://gitcode.com/cann/catlass/blob/master/examples/44_quant_matmul_full_loadA_tla](https://gitcode.com/cann/catlass/blob/master/examples/44_quant_matmul_full_loadA_tla)
4. 参考合入PR：[https://gitcode.com/cann/catlass/pull/678](https://gitcode.com/cann/catlass/pull/678)

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab WebIDE 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  

3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。

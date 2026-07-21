# 7月社区任务-aclnnBernoulli算子开发任务书

## 基础信息

- **技术标签**：算子开发、内存优化
- **适配硬件**：Atlas A2/A3 训练系列产品
- **开源仓地址**：https://gitcode.com/cann/ops-math/tree/master/math
- **CANN 版本**：CANN 8.5.0及以上
- **开发语言**：Ascend C

## 任务概述

`Tensor.bernoulli_` 对应的 `aclnnBernoulli` 算子当前内存一致性不达标，与 GPU 存在内存差距，需要将内存差距降至 5% 以下。

**现状与根因**：该算子采用小算子拼接实现，`DropoutDoMask`、`Fill`、`Mask` 算子引入了中间临时输出，导致内存膨胀，膨胀比例与 shape、dtype 相关：bf16/fp32 膨胀约 50%，fp16/int64 膨胀约 25%。GPU 侧采用单个 kernel 实现，aclnn 内部为拼接实现（genmask(DSA)+fill+dropoutdomask）。

**参考方案**：`fill + DropoutDoMask` 做 inplace 融合，bf16/fp32 膨胀 1 份 fp32，存在性能风险但可接受。


## 核心开发要求

### 功能实现要求

1. 将 `fill + DropoutDoMask` 融合为 inplace 实现，消除中间临时输出，降低内存膨胀。
2. 内存占用与 GPU 差距控制在 5% 以下（bf16/fp32 膨胀由 50% 降至 1 份 fp32 量级）。
3. 计算结果与 PyTorch `torch.bernoulli` / `Tensor.bernoulli_` 语义一致，不影响原有功能。


### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|-----------|------|----------|----------|----------|-------------|---------------|
| self（aclTensor*） | 输入 | 输入tensor。 | 支持空Tensor。数据类型需要与out一致。shape需要与out的一致。 | FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16 | ND | 0-8 | √ |
| prob（aclScalar*） | 输入 | 公式中的prob。 | 满足0≤prob≤1。 | FLOAT16、FLOAT、DOUBLE、BFLOAT16 | - | - | - |
| seed（int64_t） | 输入 | 设置随机数生成器的种子。 | - | INT64 | - | - | - |
| offset（int64_t） | 输入 | 设置随机数偏移量。 | 取值约束：offset % 4 == 0，例如可以取0、4、8 ...，不满足约束会调用失败 | INT64 | - | - | - |
| out（aclTensor*） | 输出 | 公式中的out。 | 支持空Tensor。数据类型需要与self一致。shape需要与self的一致。 | FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16 | ND | 0-8 | √ |
| workspaceSize（uint64_t*） | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - | - |
| executor（aclOpExecutor**） | 输出 | 返回op执行器，包含了算子计算流程。 | - | - | - | - | - |

### 算子约束限制

无

## 测试标准

请根据给出的自测用例和测试指导完成自测，并输出自测报告。

### 内存一致性要求（核心验收）

`aclnnBernoulli` 实现与 GPU 内存差距 **5% 以下**。

### 精度要求

和原算子对齐，计算精度需满足 [AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest) 工具默认阈值。

### 性能要求

不低于原算子性能。

## 验收交付件

1. 算子设计文档（含 inplace 融合方案与内存优化分析），需根据[参考模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；评审通过后合入[cann-competitions 仓库](https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist)，详细说明见[readme](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。
2. 可指导算子使用与测试的交付件：自测用例、测试代码脚本及脚本运行 README；测试结果报告，需包含所有自测用例结果，含内存对比数据、精度结果、整体测试通过截图。
3. 算子代码的私仓邀请链接、代码仓路径、分支、算子目录。README 文档内容完整、规范。

## PR 申请合入

测试通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入对应目录：https://gitcode.com/cann/ops-math/tree/master/experimental/random。

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
2. PyTorch 参考：[torch.bernoulli](https://pytorch.org/docs/stable/generated/torch.bernoulli.html)
3. 代码样例：https://gitcode.com/cann/ops-math/tree/master/random/stateless_bernoulli

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  

3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 内存一致性为本任务核心验收项，需提供与 GPU 的内存对比数据；
4. 随机采样算子精度比对需采用合理的统计/固定种子策略。
5. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。

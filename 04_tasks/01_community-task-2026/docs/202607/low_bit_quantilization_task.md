# 7月社区任务-低比特量化算法开发任务书

## 基础信息

- **技术标签**：量化算法开发
- **适配硬件**：Atlas A5 训练系列产品/推理系列产品
- **开源仓地址**：[https://gitcode.com/cann/amct](https://gitcode.com/cann/amct)
- **CANN 版本**：AMCT开源仓指定版本
- **开发语言**：Python

## 任务概述

参考现有技术资料优化LLM低比特量化算法，针对不同模型场景进行定制化调整，实现精度损失控制在可接受范围的前提下，显存占用降低50%以上。

## 核心开发要求及验收标准

### 功能实现要求
1. 完成对LLM网络的低比特量化算法，目标数据类型可选择hifloat8、mxfp4、int4。
2. 低比特量化算法可以是对AMCT代码仓已有算法的优化，优化后的精度相比原本的需要有提升。
3. 低比特量化算法可以是AMCT代码仓中未适配的算法，并实现低比特量化下的精度优化（进阶项）

### 测试标准

1. 自验证报告完整、可复现，所有测试用例执行通过。使用下面的模型+数据集完成量化网络的精度测试，测试脚本和执行步骤请参考：https://gitcode.com/cann/amct/tree/master/examples/algorithms/ofmr
2. 测试模型：Qwen3-8B/Qwen3.6-35B-A3B
3. 测试数据集：wikitext2


### 精度要求
1. hifloat8：量化后的网络精度相比原始BF16网络精度，PPL劣化不超过0.1；量化层（torch.nn.Linear）占比不低于80%
2. mxfp4、int4：量化后的网络精度相比原始BF16网络精度，PPL劣化不超过0.4；量化层（torch.nn.Linear）占比不低于70%


### 文档规范要求

1. 设计文档内容完整、方案描述清晰准确，且必须通过评审；
2. 自验证报告需要覆盖所有功能场景，含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可清晰指导算法的使用与测试；
3. README 文档内容完整、规范。

## 验收交付件

1, 自测用例、测试结果报告、测试步骤指导文档

2, 算子代码的私仓邀请链接、代码仓路径、分支、算子目录

3，评审通过的设计文档（按模板填写），合入 [amct 仓库](https://gitcode.com/cann/amct/tree/feature/community-tasks) 详细说明见 [readme](https://gitcode.com/cann/amct/blob/feature%2Fcommunity-tasks/README.md)。


## PR 申请合入

测试通过后，在昇腾AMCT开源仓提交 PR 申请，申请将开发完成的代码合入 https://gitcode.com/cann/amct/tree/feature/community-tasks。

## 参考资料

1. 参考文档：
- [LLM量化工具](https://gitcode.com/cann/amct/blob/master/docs/zh/AMCT_Pytorch_LLM.md)，新开发的量化算法可以按照此代码结构进行开发
- [量化工具数据介绍](https://gitcode.com/cann/amct/blob/master/docs/zh/compression_concepts.md)
- [量化工具算法介绍](https://gitcode.com/cann/amct/blob/master/docs/zh/algorithm_brief.md)
- [hifloat8数据格式](https://arxiv.org/abs/2409.16626)
- [mxfp4数据格式](https://arxiv.org/abs/2310.10537)
- [业界量化算法——BATQuant](https://arxiv.org/abs/2603.16590)
- [业界量化算法——MR-GPTQ](https://arxiv.org/abs/2509.23202)

2. 参考代码：
- [算法代码样例](https://gitcode.com/cann/amct/tree/master/amct_pytorch/experimental/flatquant)
- [hifloat8仿真算子](https://gitcode.com/cann/amct/tree/master/amct_ops/hifloat8_cast)


## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab WebIDE 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  

3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 python 编程规范及开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。

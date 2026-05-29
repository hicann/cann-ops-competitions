# RightShift算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Atlas A2 训练系列产品
- **开源仓地址**：[https://gitcode.com/cann/ops-math](https://gitcode.com/cann/ops-math)
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 任务概述

参考aclnnRightShift实现，在昇腾 NPU 上基于 Ascend C 编程语言实现与torch.bitwise_right_shift功能精度一致的算子。完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。
当前shiftBits为负数时行为与torch不一致，需要开发与torch一致的算子。

## 核心开发要求及验收标准

### 功能实现要求

1. 支持aclnnRightShift原所有数据类型与格式，并且精度与torch.bitwise_right_shift一致
2. 必须实现算子泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。

### 参数说明
**当前第二个输入是负数时，与Pytorch行为不一致，新开发算子需要与Pytorch精度一致**

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------------|------|----------|----------|----------|-------------|--------------|
| input | 输入 | 需要进行按位右移的张量，公式中的input。 | 支持空Tensor，shape需与shiftBits满足broadcast关系。 | INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64 | ND | 0-8 | √ |
| shiftBits | 输入 | 右移操作数的张量，公式中的shiftBits。 | - | INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64 | ND | 0-8 | √ |
| out | 输出 | 输出张量，公式中的output。 | shape需与input保持一致。 | INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64 | ND | 0-8 | √ |

### 算子约束限制

无

### 测试标准

测试用例覆盖**常规场景、边界场景**等所有功能场景；自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

1. 算子整体性能达成原aclnnRightShift性能的10X


### 精度要求

算子计算精度需满足 [AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest) 工具默认阈值

### 文档规范要求

1. 算子设计文档需根据[参考模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；
2. 自验证报告需要覆盖所有功能场景，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)（不需要模板中TBE样例结果），含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可清晰指导算子使用与测试；
3. README 文档内容完整、规范。

## 验收规则与流程

### 提交验收申请

联系昇腾小助手，提交以下**三类交付件**进行验收：

1. 昇腾开源算子仓 fork 的个人代码仓链接（需包含：算子工程代码、算子 README 文档、多组 aclnn 调用测试代码）；
2. 算子自验证报告；
3. 华为评审通过的算子设计文档（按模板填写），合入 [cann-competitions 仓库](https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist) 详细说明见 [readme](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。

### 验收结果反馈

验收以提交验收申请时的代码为准，72小时内反馈验收结果，如代码更新请重新提交验收申请，验收时间同步刷新。

### PR 申请合入

验收通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入 https://gitcode.com/cann/ops-math/tree/master/experimental/math

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
2. 课程类：[Ascend C在线课程](https://www.hiascend.com/developer/courses/detail/1691696509765107713)

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab notebook 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  
   ![环境截图](pics/zaixiankaifa2.png)

3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。
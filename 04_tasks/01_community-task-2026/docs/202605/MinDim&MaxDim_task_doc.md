# MinDim&MaxDim算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Atlas A2 训练系列产品/Atlas A3 系列产品
- **开源仓地址**：[https://gitcode.com/cann/ops-math](https://gitcode.com/cann/ops-math)
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 任务概述

1. 参考昇腾版本内置aclnnMinDim算子（原TBE实现对应argminwithvalue）的 TBE 实现，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的算子，完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。
2. 参考昇腾版本内置aclnnMaxDim算子的 TBE 实现（内部实现为argmaxwithvalue），在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的算子，完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。
## 核心开发要求及验收标准

### 功能实现要求

1. 与原 TBE 算子 (argminwithvalue) 核心功能完全对齐，即在指定维度上求张量的最小值及其索引位置。支持float32、float16、bfloat16、int16等4种数据类型。
2. 必须实现算子泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。

### 参数说明

| 参数名 | 输入／输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor
|---|---|---|---|---|---|---|---|
| self | 输入 | 待计算的目标张量。 | self与out的数据类型保持一致，支持非连续的Tensor。 | FLOAT16、FLOAT、INT16、BFLOAT16 | ND | 支持[1, 8]维 | √
| dim | 输入 | 指定的维度。 | 数据类型为INT64的标量，取值范围在[-self.dim(), self.dim())内。 | INT64 | - | 标量 | -
| keepdim | 输入 | 是否保留reduce轴的维度。 | 数据类型为BOOL的标量。如果为false，则输出维度为self维度减1；如果为true，则输出维度等于self维度且指定轴维度值为1。 | BOOL | - | 标量 | -
| out | 输出 | 沿指定维度求最小值的输出张量。 | 数据类型和self一致，支持非连续的Tensor。如果keepdim为false，则输出维度为self维度减1；如果keepdim为true，则输出维度等于self维度。 | FLOAT16、FLOAT、INT16、BFLOAT16 | ND | 与self维度相同（keepdim=true）或少1维（keepdim=false） | √
| indices | 输出 | 最小值的索引输出张量。 | 数据类型为INT32，支持非连续的Tensor，维度与out相同。 | INT32 | ND | 与out相同 | √


| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|---|---|---|---|---|---|---|---|
| self | 输入 | 输入Tensor。 | 在指定维度上求最大值的输入。 | FLOAT16、FLOAT、BFLOAT16、INT16 | ND | [1,8] | √ |
| dim | 属性 | 指定的reduce维度。 | 数据类型为INT64，取值范围在[-self.dim(), self.dim())。 | - | - | - | - |
| keepdim | 属性 | reduce轴的维度是否保留。 | 如为true则保留指定维度的维度值为1；如为false则不保留。 | - | - | - | - |
| out | 输出 | 最大值输出。 | 数据类型与self一致。 | FLOAT16、FLOAT、BFLOAT16、INT16 | ND | - | √ |
| indices | 输出 | 最大值索引输出。 | 返回最大值在指定维度上的索引位置。 | INT32 | ND | - | √ |

### 算子约束限制

无。

### 测试标准

需参考内置 TBE 算子自行设计全场景自验证用例，验收阶段将采用泛化数据进行功能、精度、性能全维度验证。自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

1. 暂仅要求**所有核参与计算场景**下，性能不低于原 TBE 算子的 95%。
2. 如小shape无法达标（10us以下场景相差3us），提供性能仿真图和分析结论证明Ascend C实现与TBE完全一致或优于TBE实现。

### 精度要求

算子计算精度需满足 [AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest) 工具默认阈值。

### 文档规范要求

1. 算子设计文档需根据[参考模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；
2. 自验证报告需要覆盖所有功能场景，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)，含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可清晰指导算子使用与测试；
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

验收通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入（https://gitcode.com/cann/ops-math/tree/master/experimental/math）。

## TBE 参考实现路径

本次开发需参考昇腾CANN内置 TBE 算子实现，具体文件路径如下：

1. kernel 实现：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/arg_min_with_value`
2. 算子原型：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_graph/inc/ops_proto_math.h`
3. 算子信息库：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info-legacy.json`

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[TBE算子开发文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/tbeaicpudevg/atlasopdev_10_0001.html)、[算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
2. 课程类：[Ascend C在线课程](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
3. 代码样例：[https://gitcode.com/cann/ops-math/tree/master/math](https://gitcode.com/cann/ops-math/tree/master/math)

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab notebook 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  

3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。
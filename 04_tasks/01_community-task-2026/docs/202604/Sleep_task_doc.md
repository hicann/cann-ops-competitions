# Sleep算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Atlas A2 训练系列产品/Atlas A3 系列产品
- **开源仓地址**：[https://gitcode.com/cann/ops-nn](https://gitcode.com/cann/ops-nn)
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 任务概述

torch.cuda._sleep主要用途是测试中使用，用于精确控制kernel执行时间；可用于测试CUDA stream的异步/并发行为、vent、barrier的正确同步语义，主流三方库VeOmni中用到。当前npu上缺失对应算子和API，为提升兼容性，需补齐该API。算子实现类似spin_kernel的功能（[pytorch/aten/src/ATen/cuda/Sleep.cu at main · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/Sleep.cu)）。

## 核心开发要求及验收标准

### 功能实现要求

1. 在 <https://gitcode.com/cann/ops-nn/tree/master/control> 目录下创建sleep目录，实现对标 torch.cuda._sleep的功能。算子向当前执行流提交一个设备侧延时片段，延时过程在 AI Core kernel 内部通过 busy-spin 实现。
2. 算子原型定义如下：

   | 参数名   | 入参类型 | 默认值 | 基础类型 | 语义说明                                     |
   |----------|----------|--------|----------|----------------------------------------------|
   | cycles   | 必选属性 | 无     | int64    | 设备侧 busy-spin 的目标 cycle 数，表示设备 cycle 延时长度 |

3. 外接口为两段式 aclnn：
   - `aclnnSleepGetWorkspaceSize(int64_t cycles, uint64_t *workspaceSize, aclOpExecutor executor)`
   - `aclnnSleep(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
4. 在examples目录中使用pybind的方式绑定torch接口进行测试

### 测试标准

测试用例覆盖**常规场景、边界场景**等所有功能场景；自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

> 不涉及

### 精度要求

不涉及

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

验收通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入 [https://gitcode.com/cann/ops-nn/tree/master/experimental/control](https://gitcode.com/cann/ops-nn/tree/master/experimental/control)。

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
2. 课程类：[Ascend C在线课程](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
3. 代码样例：[https://gitcode.com/cann/ops-math](https://gitcode.com/cann/ops-math)

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
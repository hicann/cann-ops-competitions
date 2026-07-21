# 7月社区任务-prims.bessel_i0 API开发任务书

## 基础信息

- **技术标签**：高阶API开发
- **适配硬件**：Ascend 950PR
- **开源仓地址**：https://gitcode.com/ascend/asc-devkit
- **CANN 版本**：本次社区任务指定CANN版本
- **开发语言**：Ascend C

## 任务概述

基于 PyTorch IR 的 `prims.bessel_i0` 接口，在昇腾 NPU 上使用 Ascend C 编程语言实现对应的高阶API。该API用于计算第一类零阶修正贝塞尔函数I_0(x)，公式如下，完成API设计、开发、测试全流程工作，测试通过后将代码提交至asc-devkit开源仓。
$$\mathrm{I}_0(x) = \sum_{k=0}^{\infty} \frac{1}{(k!)^2}\left(\frac{x}{2}\right)^{2k}$$

## 核心开发要求及验收标准

### 功能实现要求

1. 与 PyTorch IR `prims.bessel_i0` 接口核心功能完全对齐，支持原API对应的所有数据类型（float32）、数据格式。
2. 支持API泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。
3. 正确处理边界情况（大值输入的数值稳定性、NaN/Inf特殊值处理）。

### 测试标准

需自行设计全场景自验证用例，验收阶段将采用泛化数据进行功能、精度、性能全维度验证。自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

1. 作为vector类高阶API，实现Vector Bound（aiv_vec流水占比超90%）。
2. 特殊场景无法达标的（如小shape场景等），需提供性能仿真图和分析结论佐证。

### 精度要求

按照float数据类型双万分之一要求，即万分之一的数据误差不超过万分之一（1万个数据中，误差超过万分之一的不超过1个）。

### 文档规范要求

1. API设计文档可根据[模板](./docs/BatchNorm接口设计文档.docx)填写，内容完整、格式规范，且必须通过评审；评审通过后合入[cann-competitions 仓库](https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist)，详细说明见[readme](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。设计文档需在asc-devkit仓提交文档评审issue，2工作日内反馈评审结果，如方案有更新，评审时间同步刷新。
2. 自验证报告除基础场景外还应覆盖边界值场景，具体可参考[算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)，需包含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可指导复现测试结果。
3. README 文档内容完整、规范，可参考代码仓API的接口说明文档。

## 验收交付件

1. asc-devkit开源仓 fork 的个人代码仓链接（需包含：API代码、API README 文档、UT代码、多场景的功能测试代码）。交付件可按照如下目录提交：

    ```
    ├── docs                                # API接口说明文档
    ├── impl                                # API接口实现源代码
    ├── include                             # API接口声明源代码
    └── tests                               # API的测试代码：包括单元测试和功能测试用例
    ```
2. 根据[高阶API验证](./self_test_case/prims_bessel_i0/)示例要求，完成API的自验证，并给出验证结果报告。
3. 评审通过的API设计文档。

## PR 申请合入

测试通过后，在asc-devkit开源仓提交 PR 申请，申请将开发完成的API合入（ https://gitcode.com/cann/asc-devkit/tree/master/experimental/ ）。

## PyTorch IR参考信息

- PyTorch官方文档: https://docs.pytorch.org/docs/stable/generated/torch.i0.html

## 参考资料

1. 文档类：[Ascend C高阶API贡献指南](https://gitcode.com/cann/asc-devkit/blob/master/docs/asc_adv_api_contributing.md)，[Ascend C快速入门](https://gitcode.com/cann/asc-devkit/blob/master/docs/guide/%E5%85%A5%E9%97%A8%E6%95%99%E7%A8%8B/Ascend-C%E6%A6%82%E8%BF%B0%E4%B8%8E%E5%AD%A6%E4%B9%A0%E8%B7%AF%E5%BE%84.md)，[Ascend C接口列表](https://gitcode.com/cann/asc-devkit/tree/master/docs/api)。
2. 课程类：[Ascend C在线课程](https://www.hiascend.com/developer/courses/detail/1691696509765107713)。
3. 参考样例：asc-devkit仓库中的现有高阶API实现。

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。
   ![环境截图](pics/yunkaifa.png)
2. 使用 hidevlab WebIDE 算力。
   ![环境截图](pics/zaixiankaifa1.png)
3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循[社区编码规范](https://gitcode.com/cann/community/tree/master/contributor/coding-standards)。
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请。
3. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)。


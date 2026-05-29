# SyncBatchNormGatherStats 算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Atlas 800T A2、Atlas 300V Pro
- **开源仓地址**：[https://gitcode.com/cann/ops-nn](https://gitcode.com/cann/ops-nn)
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 任务概述

参考 (https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/aolapi/context/ops-nn/aclnnSyncBatchNormGatherStats.md） 说明，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的算子，完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。

## 核心开发要求及验收标准

### 功能实现要求

1. 与原 TBE 算子核心功能完全对齐，支持参数说明中的数据类型、数据格式。
2. 必须实现算子泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 183px">
  <col style="width: 120px">
  <col style="width: 265px">
  <col style="width: 197px">
  <col style="width: 114px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>totalSum</td>
      <td>输入</td>
      <td>表示各设备的通道特征和，对应公式中的totalSum。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>totalSquareSum</td>
      <td>输入</td>
      <td>表示各设备的通道特征平方，对应公式中的totalSquareSum。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sampleCount</td>
      <td>输入</td>
      <td>表示各设备的样本计数，对应公式中的sampleCount。</td>
      <td>FLOAT16、FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示计算过程中的均值，对应公式中的runningMean。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>输入</td>
      <td>表示计算过程中的方差，对应公式中的runningVar。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>momentum</td>
      <td>输入</td>
      <td>runningMean和runningVar的指数平滑参数。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>输入</td>
      <td>用于防止产生除0的偏移。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchMean</td>
      <td>输出</td>
      <td>表示全局批均值，对应公式中的batchMean。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batchInvstd</td>
      <td>输出</td>
      <td>表示标准差倒数，对应公式中的batchInvstd。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明
  - 收集所有device的均值和方差，更新全局的均值和方差。
  - SyncBatchNormGatherStats 默认实现确定性计算。
  

### 测试标准

测试用例覆盖**常规场景、边界场景**等所有功能场景；自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

1. 配套Atlas 800T A2，算子整体性能须与现有实现持平，相比TBE性能不劣化（性能不低于原 TBE 算子的 95%）。
2. 如小shape无法达标（100us以下场景超过TBE耗时30%以上），提供性能仿真图和分析结论，证明Ascend C实现与TBE完全一致或优于TBE实现。
3. 配套Atlas 300V Pro，验收算子功能和精度，以配套Atlas 800T A2上性能验证结论为准。

### 精度要求

Atlas 800T A2和Atlas 300V Pro上的实现，算子计算精度需满足 [AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest) 工具默认阈值。

### Atlas 300V Pro，算子接入模型验证
- 验证模型：yolo-world
- 验证数据集：https://docs.ultralytics.com/zh/datasets/detect/african-wildlife
- 模型精度：与Atlas 800T A2 对比，box_loss不超过0.1，cls_loss不超过0.1，dfl_loss不超过0.1

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

验收通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入 https://gitcode.com/cann/ops-nn/tree/master/experimental/norm。

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
2. 课程类：[Ascend C在线课程](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
3. 代码样例：[https://gitcode.com/cann/ops-transformer/tree/master/experimental](https://gitcode.com/cann/ops-transformer/tree/master/experimental)

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
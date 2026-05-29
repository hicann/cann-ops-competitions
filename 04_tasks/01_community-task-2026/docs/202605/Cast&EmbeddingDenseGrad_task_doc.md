# Atlas 300V Pro Cast、EmbeddingDenseGrad 算子开发任务书

# Cast 算子任务

参考昇腾版本内置Cast算子的 TBE，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的算子，并新增支持BF16数据类型输入。完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Atlas 300V Pro
- **开源仓地址**：[https://gitcode.com/cann/ops-math](https://gitcode.com/cann/ops-math)
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 核心开发要求及验收标准

### 功能实现要求

1. 与原 TBE 算子核心功能完全对齐，支持原算子对应的所有数据类型、数据格式，并新增支持BF16数据类型输入。
2. 必须实现算子泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 200px">
  <col style="width: 460px">
  <col style="width: 120px">
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
      <td>self</td>
      <td>输入</td>
      <td>待进行cast计算的入参。</td>
      <td>BOOL、FLOAT16、FLOAT、INT8、UINT8、INT16、INT32、INT64、BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>cast计算的出参。</td>
      <td>BOOL、FLOAT16、FLOAT、INT8、UINT8、INT16、INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- 配套Atlas 300V Pro 新增支持BF16数据类型输入。


## 约束说明

- 针对数据类型从浮点数转换为整型的场景：输入数据中存在nan，则将nan转换为0。
- 针对数据类型从INT32转换为INT8的场景：只能保证输入数据在(-2048, 1920)范围内精度无误差。
- 针对数据类型从FLOAT64/COMPLEX64/COMPLEX128转换为UINT8的场景：只能保证输入数据为非负数精度无误差。
- 针对数据类型从INT64转换为FLOAT32的场景：只能保证输入数据在(-2147483648, 2147483647)范围内精度无误差。

### 测试标准

测试用例覆盖**常规场景、边界场景**等所有功能场景；自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

1. 算子整体性能须与现有实现持平，具体如下：BF16转为其他数据类型性能接近FP32输入性能（相同shape对比，90%以上），其他格式数据转换性能与现有TBE实现不劣化（性能不低现有实现的 95%）；
2. 如小shape无法达标（10us以下场景相差3us以上），提供性能仿真图和分析结论证明Ascend C实现与TBE完全一致或优于TBE实现。

### 精度要求

算子计算精度需满足 [AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest) 工具默认阈值

### 算子接入模型验证
- 验证模型：InternVL
- 验证数据集：https://github.com/BryanPlummer/flickr30k_entities
- 模型精度：与Atlas 800T A2 对比，train loss不超过0.1


# EmbeddingDenseGrad 算子任务

参考 (https://gitcode.com/cann/ops-nn/blob/master/index/embedding_dense_grad_v2/README.md) 实现，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的算子，完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Atlas 300V Pro
- **开源仓地址**：[https://gitcode.com/cann/ops-nn](https://gitcode.com/cann/ops-nn)
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 核心开发要求及验收标准

### 功能实现要求

1. 与参考算子的核心功能完全对齐，实现[Embedding]的反向计算, 将相同索引`indices`对应grad的一行累加到out上。
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
      <td>grad</td>
      <td>输入</td>
      <td>表示数据的原始梯度。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sort_indices</td>
      <td>输入</td>
      <td>表示grad输入对应的索引值。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示梯度求和的结果输出。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numWeights</td>
      <td>属性</td>
      <td>表示输出tensor的首轴大小。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_idx</td>
      <td>可选属性</td>
      <td><ul><li>将输出tensor中第paddingIdx行填充成0，如果paddingIdx为负数则不进行处理。</li><li>默认值为-1。</li></ul></td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_grad_by_freq</td>
      <td>可选属性</td>
      <td><ul><li>根据单词出现的频率，是否对梯度进行缩放。</li><li>默认值为false。</li></ul></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>


## 约束说明
- 在参数shape超过以下限制时，输出无法保证高精度，若开启了确定性计算，也无法保证高性能
  - grad合轴成二维shape后，第一个维度超过INT32_MAX(2147483647)
  - numWeights超过INT32_MAX(2147483647)
- sort_indices合轴后维度超过INT32_INF(2139095040)时，无法保证高性能
- grad合轴成二维shape后第二个维度（D）需要32字节对齐，否则无法保证高性能
- scale_grad_by_freq为True时，对梯度进行缩放，无法保证高性能

### 测试标准

测试用例覆盖**常规场景、边界场景**等所有功能场景；自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求
1. 按照 输入shape\*2除以（204GB*0.5显存带宽），得出算子理论耗时，算子实际耗时不得超过理论耗时的1.1倍。
2. 如小shape无法达标（100us以下场景超过理论耗时30%以上），提供性能仿真图和分析结论证明Ascend C实现已接近极限。

### 精度要求

算子计算精度需满足 [AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest) 工具默认阈值

### 算子接入模型验证
- 验证模型：Clip
- 验证数据集：https://github.com/BryanPlummer/flickr30k_entities
- 模型精度：与Atlas 800T A2 对比，train loss不超过0.1

# 验收规则与流程

## 文档规范要求

1. 算子设计文档需根据[参考模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；
2. 自验证报告需要覆盖所有功能场景，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)（不需要模板中TBE样例结果），含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可清晰指导算子使用与测试；
3. README 文档内容完整、规范。

## 提交验收申请

联系昇腾小助手，提交以下**三类交付件**进行验收：

1. 昇腾开源算子仓 fork 的个人代码仓链接（需包含：算子工程代码、算子 README 文档、多组 aclnn 调用测试代码）；
2. 算子自验证报告；
3. 华为评审通过的算子设计文档（按模板填写），合入 [cann-competitions 仓库](https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist) 详细说明见 [readme](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。

## 验收结果反馈

验收以提交验收申请时的代码为准，72小时内反馈验收结果，如代码更新请重新提交验收申请，验收时间同步刷新。

## PR 申请合入

验收通过后，在昇腾算子开源仓提交 PR 申请，Cast算子合入 https://gitcode.com/cann/ops-transformer/tree/master/experimental/math， EmbeddingDenseGrad算子合入 https://gitcode.com/cann/ops-nn/tree/master/experimental/index。

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
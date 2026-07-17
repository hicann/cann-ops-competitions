# RoaringBitMap容器开发任务书

## 基础信息

- **技术标签**：算子开发、框架开发
- **适配硬件**：Atlas 950 系列产品
- **开源仓地址**：https://gitcode.com/cann/ops-collections
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C、C++

## 任务概述

参考 [cuCollections](https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/roaring_bitmap.cuh)中的roaring_bitmap容器实现，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的容器及算子，完成容器及算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。

### 功能实现要求

1、与核心功能完全对齐，支持roaring_bitmap容器的构造、析构、Contains等功能，key的数据类型要求支持uint32、uint64类型。

2、算子需实现泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。

### 参数说明

接口入参顺序要求与cuCollections保持一致，参数类型可以适当进行调整，但要求参数含义仍然不变。如cuCollections中的device迭代器入参可使用void*指针进行替代，但是原参数含义为输入数据，更换数据类型后参数含义不变。以static_set的contains为例：

ops-collections实现：

void Contains(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| outputValues | void* | 输出 | Device侧指向输出值数组的指针（bool类型） |
| keyNum | Extent | 输入 | 要查找的键数量 |
| stream | aclrtStream | 输入 | ACL流 |

cucollections实现：

  void contains(InputIt first,
                InputIt last,
                OutputIt output_begin,
                cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const;

| 参数 | 类型 | 输入/输出 | 说明 |
| ------ | --- | ---------------- | ----------- |
| first  | InputIt  | 输入          | 被查询键的起始迭代器 |
| last   | InputIt  | 输入          | 被查询键的结束迭代器 |
| output_begin    | OutputIt  | 输入        | 布尔序列的起始位置，用于表示每个键是否存在   |
| stream | size_type  | 输入 | 执行的流        |

Contains接口的设计中，由于昇腾当前还未实现对应的基础数据结构，如device迭代器，因此当前使用起始地址+元素个数的方式来替代起始+结束迭代器的入参设计。

### 测试标准

请根据给出的自测用例和测试指导完成自测，并输出自测报告。

### 性能要求

I32场景下算子所有用例的性能需大于等于0.8倍GPU（A100）。
I64场景下算子所有用例的性能需大于等于0.6倍GPU（A100）。
如不达标，须有合理解释。

### 精度要求

算子的计算精度需满足二进制一致，结果保持一致。

### 文档规范要求

1. 算子设计文档需根据[参考模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；
2. 自验证报告需要覆盖所有功能场景，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)（不需要模板中TBE样例结果）及[static_set容器自验证报告](https://docs.qq.com/sheet/DRGRNQmN3R1B4eFd1?tab=BB08J2)，含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可清晰指导算子使用与测试；
3. README 文档内容完整、规范。

## 验收交付件

1. 算子设计文档，需根据参考模板填写，内容完整、格式规范，且必须通过评审；评审通过后合入[cann-competitions 仓库](https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist)，详细说明见[readme](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。
2. 可以清晰指导算子使用与测试的交付件，包括：自测用例、测试代码脚本及脚本运行reademe；测试结果报告，需要包含所有自测用例结果，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)（不涉及TBE内置算子的不需要模板中TBE样例结果）及[static_set容器自验证报告](https://docs.qq.com/sheet/DRGRNQmN3R1B4eFd1?tab=BB08J2)，含测试用例执行日志/截图、整体测试通过截图、性能数据截图。
3. 算子代码的私仓邀请链接、代码仓路径、分支、算子目录。README 文档内容完整、规范。

## PR 申请合入

验收通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入 https://gitcode.com/cann/ops-collections。

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
2. 课程类：[Ascend C在线课程](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
3. 代码样例：[https://gitcode.com/cann/ops-transformer/tree/master/experimental](https://gitcode.com/cann/ops-transformer/tree/master/experimental)

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  

3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。
# 7月社区任务-BloomFilter容器开发任务书

## 基础信息

- **技术标签**：算子开发、框架开发
- **适配硬件**：Atlas 950 系列产品
- **开源仓地址**：https://gitcode.com/cann/ops-collections
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C、C++

## 任务概述

参考 [cuCollections](https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/bloom_filter.cuh)中的bloom_filter容器实现，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的容器及算子，完成容器及算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。

### 功能实现要求

1、与核心功能完全对齐，支持bloom_filter容器的构造、析构、Add、Contains、Merge、Intersect功能，key的数据类型要求支持uint32、uint64及float32类型。

2、算子需实现泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。

### 参数说明

接口入参顺序要求与cuCollections保持一致，参数类型可以适当进行调整，但要求参数含义仍然不变。如cuCollections中的device迭代器入参可使用void*指针进行替代，但是原参数含义为输入数据，更换数据类型后参数含义不变。以static_set的insert为例：

ops-collections实现：

SizeType Insert(void *keys, Extent keyNum, aclrtStream stream);

| 名称        | 类别  | dtype       | 介绍          |
| --------- | --- | ----------- | ----------- |
| keys    | 输入  | Void*       | 被插入的一批键的地址 |
| keyNum | 输入  | Extent      | 需要插入的键个数   |
| stream    | 输入  | aclrtStream | 执行的流        |
| 返回值       | 输出  | SizeType    | 插入成功的键个数   |

cucollections实现：

  size_type insert(InputIt first,
                               InputIt last,
                               cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

| 名称     | 类别  | dtype            | 介绍          |
| ------ | --- | ---------------- | ----------- |
| first  | 输入  | InputIt          | 被插入键的起始迭代器 |
| last   | 输入  | InputIt          | 被插入键的结束迭代器 |
| stream | 输入  | cuda::stream_ref | 执行的流        |
| 返回值    | 输出  | size_type        | 插入成功的键个数   |

Insert接口的设计中，由于昇腾当前还未实现对应的基础数据结构，如device迭代器，因此当前使用起始地址+元素个数的方式来替代起始+结束迭代器的入参设计。

### 测试标准

请根据给出的自测用例和测试指导完成自测，并输出自测报告。

### 性能要求

I32场景下算子所有用例的性能需大于等于0.8倍GPU（A100）。
I64场景下算子所有用例的性能需大于等于0.6倍GPU（A100）。
如不达标，须有合理解释。

### 精度要求

搬移类算子计算精度需满足二进制一致，如插入1-10十个整数，在bloom_filter中需完整找到1-10这十个整数，且取出来的10个整数可以和输入时的十个整数做到二进制一一对应。

## 验收交付件

1. 算子设计文档，需根据参考模板填写，内容完整、格式规范，且必须通过评审；评审通过后合入[cann-competitions 仓库](https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist)，详细说明见[readme](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。
2. 可以清晰指导算子使用与测试的交付件，包括：自测用例、测试代码脚本及脚本运行readme；测试结果报告，需要包含所有自测用例结果，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)（不涉及TBE内置算子的不需要模板中TBE样例结果）及[static_set容器自验证报告](https://docs.qq.com/sheet/DRGRNQmN3R1B4eFd1?tab=BB08J2)，含测试用例执行日志/截图、整体测试通过截图、性能数据截图。
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

2. 使用 hidevlab webIDE 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  
   
3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。

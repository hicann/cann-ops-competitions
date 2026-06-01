# DynamicMap容器开发任务书

## 基础信息

- **技术标签**：算子开发、框架开发
- **适配硬件**：Atlas 950 系列产品
- **开源仓地址**：https://gitcode.com/cann/ops-collections
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C、C++

## 任务概述

参考 [cuCollections](https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/dynamic_map.cuh)中的dynamic_map容器实现，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的容器及算子，完成容器及算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。

### 功能实现要求

1、 与核心功能完全对齐，支持dynamic_map容器的构造、析构、Insert、Erase、Find、Contains、reserve、insert_or_assign功能，key与value数据类型要求支持uint16、uint32、uint64及float32类型。

2、算子需实现泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。

### 参数说明

接口入参顺序要求与cuCollections保持一致，参数类型可以适当进行调整，但要求参数含义仍然不变。如cuCollections中的device迭代器入参可使用void*指针进行替代，但是原参数含义为输入数据，更换数据类型后参数含义不变。以static_map的insert为例：

ops-collections实现：

SizeType Insert(void *values, Extent valueNum, aclrtStream stream);

| 名称        | 类别  | dtype       | 介绍          |
| --------- | --- | ----------- | ----------- |
| values    | 输入  | Void*       | 被插入的一批元素的地址 |
| valuesNum | 输入  | Extent      | 需要插入的元素个数   |
| stream    | 输入  | aclrtStream | 执行的流        |
| 返回值       | 输出  | SizeType    | 插入成功的元素个数   |

cucollections实现：

  size_type insert(InputIt first,
                               InputIt last,
                               cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

| 名称     | 类别  | dtype            | 介绍          |
| ------ | --- | ---------------- | ----------- |
| first  | 输入  | InputIt          | 被插入元素的起始迭代器 |
| last   | 输入  | InputIt          | 被插入元素的结束迭代器 |
| stream | 输入  | cuda::stream_ref | 执行的流        |
| 返回值    | 输出  | size_type        | 插入成功的元素个数   |

Insert接口的设计中，由于昇腾当前还未实现对应的基础数据结构，如device迭代器，因此当前使用起始地址+元素个数的方式来替代起始+结束迭代器的入参设计。

### 测试标准

测试用例覆盖**常规场景、边界场景**等所有功能场景；自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

整体要求：
I32及I16场景下算子整体性能需与0.7倍GPU（A100）持平。
I64场景下算子整体性能需与0.5倍GPU（A100）持平。

**Insert:**

| Key | Value | Distribution | 插入数据量    | BatchSize | InitSize  | GPU A100性能 |
| --- | ----- | ------------ | -------- | --------- | --------- | ---------- |
| I32 | I32   | UNIQUE       | 80000000 | 800000    | 40000000  | 8.46ms     |
| I32 | I32   | UNIQUE       | 80000000 | 800000    | 160000000 | 18.84ms    |
| I64 | I64   | UNIQUE       | 80000000 | 800000    | 40000000  | 142.29ms   |
| I64 | I64   | UNIQUE       | 80000000 | 800000    | 160000000 | 20.22ms    |
| I16 | I16   | UNIQUE       | 80000000 | 800000    | 40000000  | 8.45ms     |
| I16 | I16   | UNIQUE       | 80000000 | 800000    | 160000000 | 8.78ms     |

**Erase:**

| Key | Value | Distribution | 插入数据量    | InitSize | MatchingRate | GPU A100性能 |
| --- | ----- | ------------ | -------- | -------- | ------------ | ---------- |
| I32 | I32   | UNIQUE       | 80000000 | 40000000 | 0.1          | 38.41ms    |
| I32 | I32   | UNIQUE       | 80000000 | 40000000 | 0.5          | 41.15ms    |
| I32 | I32   | UNIQUE       | 80000000 | 40000000 | 1            | 40.15ms    |
| I64 | I64   | UNIQUE       | 80000000 | 40000000 | 0.1          | 40.1ms     |
| I64 | I64   | UNIQUE       | 80000000 | 40000000 | 0.5          | 42.99ms    |
| I64 | I64   | UNIQUE       | 80000000 | 40000000 | 1            | 41.98ms    |
| I16 | I16   | UNIQUE       | 80000000 | 40000000 | 0.1          | 15.47ms    |
| I16 | I16   | UNIQUE       | 80000000 | 40000000 | 0.5          | 15.6ms     |
| I16 | I16   | UNIQUE       | 80000000 | 40000000 | 1            | 15.71ms    |

**Find:**

| Key | Value | Distribution | 插入数据量    | InitSize | MatchingRate | GPU A100性能 |
| --- | ----- | ------------ | -------- | -------- | ------------ | ---------- |
| I32 | I32   | UNIQUE       | 80000000 | 40000000 | 0.1          | 27.96ms    |
| I32 | I32   | UNIQUE       | 80000000 | 40000000 | 0.5          | 26.5ms     |
| I32 | I32   | UNIQUE       | 80000000 | 40000000 | 1            | 23.27ms    |
| I64 | I64   | UNIQUE       | 80000000 | 40000000 | 0.1          | 29.71ms    |
| I64 | I64   | UNIQUE       | 80000000 | 40000000 | 0.5          | 28.06ms    |
| I64 | I64   | UNIQUE       | 80000000 | 40000000 | 1            | 24.58ms    |
| I16 | I16   | UNIQUE       | 80000000 | 40000000 | 0.1          | 5.42ms     |
| I16 | I16   | UNIQUE       | 80000000 | 40000000 | 0.5          | 5.4ms      |
| I16 | I16   | UNIQUE       | 80000000 | 40000000 | 1            | 5.4ms      |

**Contains:**

| Key | Value | Distribution | 插入数据量    | InitSize | MatchingRate | GPU A100性能 |
| --- | ----- | ------------ | -------- | -------- | ------------ | ---------- |
| I32 | I32   | UNIQUE       | 80000000 | 40000000 | 0.1          | 21.42ms    |
| I32 | I32   | UNIQUE       | 80000000 | 40000000 | 0.5          | 19.41ms    |
| I32 | I32   | UNIQUE       | 80000000 | 40000000 | 1            | 16.44ms    |
| I64 | I64   | UNIQUE       | 80000000 | 40000000 | 0.1          | 26.82ms    |
| I64 | I64   | UNIQUE       | 80000000 | 40000000 | 0.5          | 24.46ms    |
| I64 | I64   | UNIQUE       | 80000000 | 40000000 | 1            | 20.84ms    |
| I16 | I16   | UNIQUE       | 80000000 | 40000000 | 0.1          | 4.53ms     |
| I16 | I16   | UNIQUE       | 80000000 | 40000000 | 0.5          | 4.52ms     |
| I16 | I16   | UNIQUE       | 80000000 | 40000000 | 1            | 4.52ms     |

**创建：**

| Key | Value | InitSize | GPU A100性能 |
| --- | ----- | -------- | ---------- |
| I32 | I32   | 40000000 | 8.22ms     |
| I64 | I64   | 40000000 | 16.34ms    |
| I16 | I16   | 40000000 | 4.14ms     |

**销毁：**

| Key | Value | InitSize | GPU A100性能 |
| --- | ----- | -------- | ---------- |
| I32 | I32   | 40000000 | 1.04ms     |
| I64 | I64   | 40000000 | 1.95ms     |
| I16 | I16   | 40000000 | 3.52us     |

**Reserve:**

| Key | Value | Distribution | NumInputs | BatchSize | InitSize | GPU A100性能 |
| --- | ----- | ------------ | --------- | --------- | -------- | ---------- |
| I32 | I32   | UNIQUE       | 80000000  | 800000    | 40000000 | 22.8525    |
| I32 | I32   | UNIQUE       | 80000000  | 800000    | 80000000 | 15.60875   |
| I64 | I64   | UNIQUE       | 80000000  | 800000    | 40000000 | 46.0875    |
| I64 | I64   | UNIQUE       | 80000000  | 800000    | 80000000 | 30.88375   |
| I16 | I16   | UNIQUE       | 80000000  | 800000    | 40000000 | 12.28      |
| I16 | I16   | UNIQUE       | 80000000  | 800000    | 80000000 | 8.215      |

**Insert_or_assign：**

| Key | Value | Distribution | NumInputs | BatchSize | InitSize  | GPU A100性能 |
| --- | ----- | ------------ | --------- | --------- | --------- | ---------- |
| I32 | I32   | UNIQUE       | 80000000  | 800000    | 40000000  | 97.365     |
| I32 | I32   | UNIQUE       | 80000000  | 800000    | 160000000 | 163.0263   |
| I64 | I64   | UNIQUE       | 80000000  | 800000    | 40000000  | 161.3638   |
| I64 | I64   | UNIQUE       | 80000000  | 800000    | 160000000 | 293.5925   |
| I16 | I16   | UNIQUE       | 80000000  | 800000    | 40000000  | 51.9825    |
| I16 | I16   | UNIQUE       | 80000000  | 800000    | 160000000 | 158.1788   |

### 精度要求

参考该链接编写测试用例，使得Ascend C算子结果和预期实现结果一致：
https://gitcode.com/cann/ops-collections/tree/master/tests/static_map

### 文档规范要求

1. 算子设计文档需根据[参考模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；
2. 自验证报告需要覆盖所有功能场景，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)（不需要模板中TBE样例结果）及[static_set容器自验证报告](https://docs.qq.com/sheet/DRGRNQmN3R1B4eFd1?tab=BB08J2)，含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可清晰指导算子使用与测试；
3. README 文档内容完整、规范。

## 验收规则与流程

### 提交验收申请

联系昇腾小助手，提交以下**三类交付件**进行验收：

1. 昇腾开源算子仓 fork 的个人代码仓链接（需包含：算子工程代码、算子 README 文档、多组 aclnn 调用测试代码）；
2. 算子自验证报告；
3. 华为评审通过的算子设计文档（按模板填写），合入 [cann-competitions 仓库](https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist) 详细说明见 [readme](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。

### 验收结果反馈

验收以提交验收申请时的代码为准，96小时内反馈验收结果，如代码更新请重新提交验收申请，验收时间同步刷新。

### PR 申请合入

验收通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入 https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention。

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
2. 课程类：[Ascend C在线课程](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
3. 代码样例：[https://gitcode.com/cann/ops-transformer/tree/master/experimental](https://gitcode.com/cann/ops-transformer/tree/master/experimental)

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。
   
   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab notebook 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。
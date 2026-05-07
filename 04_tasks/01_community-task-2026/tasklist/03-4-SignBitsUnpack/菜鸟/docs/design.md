# 需求背景（required）

## 需求来源背景介绍

### SignBitsUnpack算子实现优化

基于SignBitsUnpack算子历史TBE版本使用Ascend C编程语言进行优化。

SignBitsUnpack算子（TBE）实现路径和相关API路径

SignBitsUnpack算子实现路径为：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl

SignBitsUnpack算子实现中的API路径：/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/dsl

### SignBitsUnpack算子现状分析

通过对SignBitsUnpack算子TBE版本的功能分析，当前支持的能力如下：

①输出参数为y，输入参数x，支持float16，float32，bfloat16 三种格式的输入.

②SignBitsUnpack算子功能是判断数据的符号。

$$
out =  \ _{1.0 if  \ \ bit = 1}^{0.0 if\ \ bit =0}
$$

SignBitsUnpack算子TBE版本的整体流程图如下图所示：
![tbe-0304.jpg](https://raw.gitcode.com/user-images/assets/7649531/07f70ed3-121a-4210-a4f9-172e4c91d451/tbe-0304.jpg "tbe-0304.jpg")

# 需求分析

## 外部组件依赖

不涉及外部组件依赖。

## 内部适配模块

适配Aclnn接口调用。

## 需求模块设计

### 算子原型

1. 原型设计

* 任务书中不需要支持int64


| 名称  | 类别 | dtype            | format | shape | 介绍                                           |
| ----- | ---- | ---------------- | ------ | ----- | ---------------------------------------------- |
| self  | 输入 | uint8            | ND     | all   | 1D张量，包含打包的符号位数据，N为元素个数      |
| size  | 参数 | int64            | scalar | all   | reshape时输出张量的第一个维度                  |
| dtype | 参数 | int64            | scalar | all   | 量化输出Tensor的数据类型，0=float, 1=float16   |
| out   | 输出 | float16、float32 | ND     | all   | 输出,解包后的float数据，2维，总元素数 = N × 8 |

1. 相关约束

Atlas A2 训练系列产品/Atlas 800I A2推理产品float32、float16

# 需求详细设计

## 使能方式


| **上层框架**     | **涉及的框架勾选** |
| ---------------- | ------------------ |
| TF训练/推理      |                    |
| Pytorch训练/推理 |                    |
| ATC推理          | √                 |
| Aclnn直调        |                    |
| OPAT调优         |                    |
| SGAT子图切分     |                    |

## 需求总体设计

**3.2.1 host侧设计：**

**tiling策略：**

当不需要广播的情况下，算子计算过程不涉及数据的维度信息，故在host侧将数据视为一维向量，仅考虑数据个数，不考虑数据维度信息。虽然tbe代码中有将标量1广播到x.shape的操作，在AscendC中通过Duplicate接口可以复制标量1，该接口不需要shape传入。

任务均分：coreNum 根据输入长度和块大小动态调整，确保每个核心处理的数据块数均匀。

批量搬运：tailTileLength 和 formerTileLength 计算单次搬运的数据量，通过 tailTileNum 和 formerTileNum 确定小核/大核的搬运次数，将多次搬运合并为批量操作，减少冗余开销。尾块的处理逻辑确保不完整块也能被合并到计算流程中，避免数据碎片。

1. 分核策略

优先使用满核的原则。

如果核间能均分，可视作无大小核区分，大核小核数据块一致；

如果核间不能均分，需要将余出的数据块分配到前几个核上。

输入数据大小计算：通过GetInputShape和GetDataTypeLength函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。

UB内存大小和核心数量获取：通过平台信息获取UB内存大小和核心数量，并根据这些信息调整核心数量。

1. 数据分块和内存优化策略

充分使用UB空间的原则。

需要考虑不同硬件的UB大小不同、是否开启double buffer、kernel侧API实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。

UB内存大小获取：通过GetCoreMemSize函数获取UB内存的大小，用于后续的数据切分计算。

Tile块计算：根据UB内存大小和预定义的BLOCK_SIZE及BUFFER_NUM，计算出每个Tile块的数据数量。

数据切分：将输入数据按照计算出的Tile块大小进行切分，计算出每个core需要处理的数据块数量和最后一个block的剩余数据量。

设置切分参数：将计算出的切分参数（如每个core的数据量、Tile块大小等）设置到SignBitsUnpackTilingData对象中。

这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

1. tilingkey规划策略
   无

**数据检测**：

对不支持AscendC::Cast()bfloat16向float32转换的硬件进行直接在tiling策略时返回Get Operator Workspace failed. error code is 561002报错。

**3.2.2 kernel侧设计：**

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. 由于支持Ascend C开发的硬件中AscendC::Select支持float16、float32数据的输入.
2. tilingkey始终为0。
3. 会用到的API有：Select等等。
4. Ascend C的SignBitsUnpack算子流程见下图。
   ![ascendC-0304.jpg](https://raw.gitcode.com/user-images/assets/7649531/989c985c-a12b-43e3-99a1-2c505ccb3525/ascendC-0304.jpg "ascendC-0304.jpg")

## 支持硬件


| **支持的芯片版本**        | **涉及勾选** |
| ------------------------- | ------------ |
| 香橙派OrangePi AIpro      |              |
| Atlas 200I/500 A2推理产品 |              |
| Atlas 800I/T A2           | √           |

## 算子约束限制

不支持广播。

# 特性交叉分析可维可测分析

## 精度标准/性能标准


| **验收标准** | **描述(不涉及说明原因)** | **标准来源** |
| ------------ | ------------------------ | ------------ |
| 精度标准     | 不低于TBE版本            |              |
| 性能标准     | 不低于TBE版本            |              |

## 兼容性分析

新算子，不涉及兼容性分析

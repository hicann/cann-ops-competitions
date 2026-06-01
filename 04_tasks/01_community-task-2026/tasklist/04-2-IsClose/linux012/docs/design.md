# 需求背景（required）

## 需求来源背景介绍

### IsClose算子实现优化

基于IsClose算子历史TBE版本使用Ascend C编程语言进行优化。

IsClose算子（TBE）实现路径和相关API路径

kernel 实现：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/
算子原型：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/
算子信息库：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b

### IsClose算子现状分析

通过对IsClose算子TBE版本的功能分析，当前支持的能力如下：

①支持float16,float,int32,bfloat16.

②IsClose算子功能是返回一个带有布尔元素的新张量，判断给定的self和other是否彼此接近，如果值接近，则返回True，否则返回False。

- 计算公式：
  返回一个新张量out，其数据类型为BOOL，表示输入self的每个元素是否“close”输入other的对应元素。
  closeness定义为：

  $$
  \left | self_{i}-other_{i}\right | \le  atol + rtol\times \left | other_{i} \right |
  $$

  当self和other都是有限值时，以上公式成立。当self和other都是非有限时，且仅当它们是相等时，结果为True。当equal_nan为True，NaNs被认为是close的；当equal_nan为False，NaNs被认为是不close。

IsClose算子TBE版本的整体流程图如下图所示：
![tbe-isclose.jpg](https://raw.gitcode.com/user-images/assets/9516645/946cd7b0-59b9-42e4-bd1e-659543013c83/tbe-isclose.jpg 'tbe-isclose.jpg')
# 需求分析

## 外部组件依赖

不涉及外部组件依赖。

## 内部适配模块

适配Aclnn接口调用。

## 需求模块设计

### 算子原型

1. 原型设计

| 参数名     | 输入/输出/属性 | 描述                                          | 数据类型                        | 数据格式 |
| ---------- | -------------- | --------------------------------------------- | ------------------------------- | -------- |
| x1         | 输入           | 待进行is\_close计算的入参，公式中的self\_i。  | BFLOAT16、FLOAT16、FLOAT、INT32 | ND       |
| x2         | 输入           | 待进行is\_close计算的入参，公式中的other\_i。 | BFLOAT16、FLOAT16、FLOAT、INT32 | ND       |
| rtol       | 属性           | 待进行is\_close计算的属性，默认值1e-05。     | FLOAT                           | 1       |
| atol       | 属性           | 待进行is\_close计算的属性，默认值1e-08。     | FLOAT                           | 1       |
| equal\_nan | 属性           | 待进行is\_close计算的属性，默认值false。                   | BOOL                            | 1       |
| y          | 输出           | 待进行is\_close计算的出参。                   | BOOL                            | ND       |

3. 相关约束
Atlas A2 训练系列产品/A3系列产品

# 需求详细设计

## 使能方式


| **上层框架**     | **涉及的框架勾选** |
| ---------------- | ------------------ |
| TF训练/推理      |                    |
| Pytorch训练/推理 |                    |
| ATC推理          |                    |
| Aclnn直调        | √                 |
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

设置切分参数：将计算出的切分参数（如每个core的数据量、Tile块大小等）设置到IsCloseTilingData对象中。

这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

1. tilingkey规划策略
   无

**数据检测**：

对不支持AscendC::Cast()bfloat16向float32转换的硬件进行直接在tiling策略时返回Get Operator Workspace failed. error code is 561002报错。

**3.2.2 kernel侧设计：**

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. tilingkey始终为0。
2. 会用到的API有：Add、Mul、Muls等等。
3. Ascend C的IsClose算子流程见下图。
![AscendC-isclose.jpg](https://raw.gitcode.com/user-images/assets/9516645/9a809154-c370-46d1-ba98-ddc39046597f/AscendC-isclose.jpg 'AscendC-isclose.jpg')
## 支持硬件

AtlasA2训练系列产品及A3系列产品

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

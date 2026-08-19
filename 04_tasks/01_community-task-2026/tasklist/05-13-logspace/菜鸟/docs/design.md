# Logspace算子AscendC实现设计文档

## 一、 需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献的需求


## 二、 需求分析

额外支持int16、int32、uint8、int8，性能不劣于fp32

### 2.1 外部组件依赖

不涉及外部组件适配

### 2.2 内部适配模块

aclnn接口

### 支持的硬件

Atlas A2 训练系列产品/AtlasA3系列产品

### 2.3 需求模块设计

#### 2.3.1 AscendC算子原型



| 参数名              | 输入/输出/属性 | 描述                                           | 使用说明 | 数据类型                 | 数据格式 | 维度(shape) | 非连续Tensor |
| ------------------- | -------------- | ---------------------------------------------- | -------- | ------------------------ | -------- | ----------- | ------------ |
| start (aclScalar\*) | 输入           | 表示LogSpace的第一个输入，对数序列的起始指数。 | -        | FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、INT32 | ND       | -           | √           |
| end (aclScalar\*)   | 输入           | 表示LogSpace的第二个输入，对数序列的结束指数。 | -        | FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、INT32 | ND       | -           | √           |
| steps (int64\_t)    | 输入           | 序列中的元素数量。                             | -        | int64\_t                 | -        | -           | -            |
| base (double)       | 输入           | 对数空间的底数。                               | -        | double                   | -        | -           | -            |
| out (aclTensor\*)   | 输出           | 表示LogSpace的输出，输出的对数间隔序列张量。   | -        | FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、INT32 | ND       | 2-8         | √           |

- 相关约束:
  Atlas A2训练系列产品/Atlas 800I A2推理产品FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、INT32

## 三、 需求详细设计

### 3.1 使能方式


| 上层框架             | 涉及的框架勾选 |
| -------------------- | -------------- |
| **TF训练/推理**      |                |
| **PyTorch训练/推理** |                |
| **ATC推理**          |                |
| **Aclnn直调**        | ✔             |
| **OPAT调优**         |                |
| **SGAT子图切分**     |                |

### 3.2 需求总体设计

#### 3.2.1 host侧设计

tiling策略：

当不需要广播的情况下，算子计算过程不涉及数据的维度信息，故在host侧将数据视为一维向量，仅考虑数据个数，不考虑数据维度信息。虽然tbe代码中有将标量1广播到x.shape的操作，在AscendC中通过Duplicate接口可以复制标量1，该接口不需要shape传入。

任务均分：coreNum 根据输入长度和块大小动态调整，确保每个核心处理的数据块数均匀。

批量搬运：tailTileLength 和 formerTileLength 计算单次搬运的数据量，通过 tailTileNum 和 formerTileNum 确定小核/大核的搬运次数，将多次搬运合并为批量操作，减少冗余开销。尾块的处理逻辑确保不完整块也能被合并到计算流程中，避免数据碎片。

分核策略
优先使用满核的原则。

如果核间能均分，可视作无大小核区分，大核小核数据块一致；

如果核间不能均分，需要将余出的数据块分配到前几个核上。

输入数据大小计算：通过GetInputShape和GetDataTypeLength函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。

UB内存大小和核心数量获取：通过平台信息获取UB内存大小和核心数量，并根据这些信息调整核心数量。

数据分块和内存优化策略
充分使用UB空间的原则。

需要考虑不同硬件的UB大小不同、是否开启double buffer、kernel侧API实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。

UB内存大小获取：通过GetCoreMemSize函数获取UB内存的大小，用于后续的数据切分计算。

Tile块计算：根据UB内存大小和预定义的BLOCK_SIZE及BUFFER_NUM，计算出每个Tile块的数据数量。

数据切分：将输入数据按照计算出的Tile块大小进行切分，计算出每个core需要处理的数据块数量和最后一个block的剩余数据量。

设置切分参数：将计算出的切分参数（如每个core的数据量、Tile块大小等）设置到EqualTilingData对象中。

这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

tilingkey规划策略
无
数据检测：

对不支持AscendC::Cast()bfloat16向float32转换的硬件进行直接在tiling策略时返回Get Operator Workspace failed. error code is 561002报错。

3.2.2 kernel侧设计：

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

tilingkey始终为0。
会用到的API有：Cast等等。
Ascend C的Logspace算子流程见下图。
![e](https://openi.pcl.ac.cn/attachments/5ac336c5-8bf1-4784-bb18-f0d79e094515?type=0)

### 3.3 支持硬件


| 支持的芯片版本            | 是否支持 |
| ------------------------- | -------- |
| 香橙派OrangePi AIpro      | 不支持   |
| A3系列产品 | 支持   |
| A2训练系列产品           | 支持     |

### 3.4 算子约束限制

- 无需广播

## 四、 特性交叉分析

## 五、 可维可测分析

### 5.1 精度标准/性能标准

精度和TBE保持完全一致。

# 需求背景

## 背景信息
基于Atan2算子历史TBE版本使用Ascend C编程语言进行优化。
## TBE源码分析
通过对Atan2算子TBE版本的功能分析，当前支持的能力如下：
① 算子支持仅float、half、bfloat16格式的输入输出，并且对于16位的输入数据均提升精度到fp32进行计算，最后转回原来的数据进行输出。

- 接口功能：对输入张量self和other进行逐元素的反正切运算，注（self表示y坐标，other表示x坐标）。

- 计算公式：
  $$
  \text{atan2}(y, x) =
  \begin{cases}
  \arctan\left(\frac{y}{x}\right) & \text{if } x > 0 \\
  \arctan\left(\frac{y}{x}\right) + \pi & \text{if } x < 0 \text{ and } y \geq 0 \\
  \arctan\left(\frac{y}{x}\right) - \pi & \text{if } x < 0 \text{ and } y < 0 \\
  \frac{\pi}{2} & \text{if } x = 0 \text{ and } y > 0 \\
  -\frac{\pi}{2} & \text{if } x = 0 \text{ and } y < 0 \\
  0 & \text{if } x = 0 \text{ and } y = 0
  \end{cases}
  $$
> 由于不考虑广播场景，故向量shape需要一致。


Atan2算子TBE版本的整体流程图如下图所示：
![image.png](https://raw.gitcode.com/user-images/assets/7649531/689aa0cd-ca69-4551-8f6a-0c581d76cbc4/image.png 'image.png')

# 需求分析

## 外部组件依赖
不涉及

## 内部适配模块
适配aclnn接口调用

## 算子原型

| 名称 | 类别 | 数据类型 | format | shape |
|--|--|--|--|--|
| x1 | 输入 | float、float16、bfloat16 | ND | all |
| x2 | 输入 | float、float16、bfloat16 | ND | all |
| y  | 输出 | float、float16、bfloat16 | ND | all |

## 算子支持型号
Atlas A2 训练系列产品/Atlas 800I A2推理产品

# 详细设计

## 使能方式

| 上层框架 |涉及的框架勾选 | 
|--|--|
| TF训练/推理 |  | 
| Pytorch训练/推理  |  | 
|ATC推理  |  | 
| Aclnn直调 |  ✅| 
| OPAT调优 |  | 
| SGAT子图切分 |  | 

## host侧设计方案
算子计算过程不涉及数据的维度信息，故在host侧将数据视为一维向量，仅考虑数据个数，不考虑数据维度信息。
任务均分：coreNum 根据输入长度和块大小动态调整，确保每个核心处理的数据块数均匀。
批量搬运：tileBlockNum 和 tileDataNum 计算单次搬运的数据量，通过 finalSmallTileNum 和 finalBigTileNum 确定小核/大核的搬运次数，将多次搬运合并为批量操作，减少冗余开销。尾块的处理逻辑确保不完整块也能被合并到计算流程中，避免数据碎片。

### 1) 分核策略
优先使用满核的原则。
如果核间能均分，可视作无大小核区分，大核小核数据块一致；
如果核间不能均分，需要将余出的数据块分配到前几个核上。
输入数据大小计算：通过GetInputShape和GetDataTypeLength函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。
UB内存大小和核心数量获取：通过平台信息获取UB内存大小和核心数量，并根据这些信息调整核心数量。

### 2) 数据分块和内存优化策略
充分使用UB空间的原则。
需要考虑不同硬件的UB大小不同、是否开启double buffer、kernel侧API实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。
UB内存大小获取：通过GetCoreMemSize函数获取UB内存的大小，用于后续的数据切分计算。
Tile块计算：根据UB内存大小和预定义的BLOCK_SIZE及BUFFER_NUM和不同类型下的ubDataNum，计算出每个Tile块的数据数量。
数据切分：将输入数据按照计算出的Tile块大小进行切分，计算出每个core需要处理的数据块数量和最后一个block的剩余数据量。
设置切分参数：将计算出的切分参数（如每个core的数据量、Tile块大小等）设置到Atan2TilingData对象中。
这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

### 3) tilingkey规划策略
不进行tilingkey划分，在kernel侧利用输入数据的类型来走不同的分支。

### kernel侧设计方案
进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

此算子会涉及到许多Compare、Select运算，并且中间量很多，为了充分利用ub空间，在设计实现时，采用向量复用，当一个中间量结束使用时，其内存空间直接交给下一个中间量使用，并且Compare在针对常量进行比较时使用CompareScalar替换，Select有针对至少一个常量进行选择时，采用Tensor_Scalar模式，用以减少buf空间的占用。

并且由于atan2中会使用atan的计算逻辑，实现中选择直接使用仓库中已有的atan实现。

依照TBE实现，输入为16位时需要转换为fp32，然后进行运算，结果Cast转换为原类型，然后搬出。
Ascend C的Atan2算子流程见下图。
![image.png](https://raw.gitcode.com/user-images/assets/7649531/37e0810d-9945-4533-b2cd-393321bbf6df/image.png 'image.png')

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |

## 算子约束限制
- 暂不考虑广播，两个输入需要相同的shape
- 两个输入的dtype一致

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 来源 |
|--|--|--|
|精度标准 | 不低于tbe版本 | 历史tbe对标 |
| 性能标准 | 不低于tbe版本 | 历史tbe对标 |

## 兼容性分析
新算子，不涉及

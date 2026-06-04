# 需求背景（required）

## 需求来源

## 背景介绍

### Logit算子实现优化

基于Logit算子历史TBE版本使用Ascend C编程语言进行优化。

Logit算子（TBE）实现路径和相关API路径

Logit算子实现路径为：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl

Logit算子实现中的API路径：/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/dsl

## 硬件支持

3.1 使能方式
上层框架	涉及的框架勾选
TF训练/推理	
Pytorch训练/推理	
ATC推理	
Aclnn直调	✅
OPAT调优	
SGAT子图切分	

### Logit算子分析

### Logit算子功能分析

- 算子功能：

  该算子是概率到对数几率（log-odds）转换的一个数学运算，常用于概率值的反变换。它可以将一个介于0和1之间的概率值转换为一个实数值。

- 计算公式：

  $$
  y_i = \ln \left( \frac{z_i}{ 1 -z_i} \right)
  $$

  其中：

  $$
  z_i = 
  \begin{cases} 
  \text{eps}, & \text{if } x_i < \text{eps} \\
  x_i, & \text{if } \text{eps} \leq x_i \leq 1 - \text{eps} \\
  1 - \text{eps}, & \text{if } x_i > 1 - \text{eps}
  \end{cases}
  $$

输入：input

属性：eps

输出：out

现在已支持数据类型：float16、float32、bfloat16

支持广播：支持

# 需求分析（required）

## 需求描述

使用Ascend C编程语言实现Logit算子，在现有的基础上增加int16、int8和uint8类型支持。

<table style="undefined;table-layout: fixed; width: 878px"><colgroup>
  <col style="width: 91px">
  <col style="width: 149px">
  <col style="width: 266px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>input</td>
      <td>输入</td>
      <td>公式中的输入input。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT16、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td><ul><li>输入x的eps限制边界，防止除0错误。</li><li>建议值-1。</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT16、INT8、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 需求拆解

1. 支持float16、float32、bfloat16和int16、int8及uint8数据类型
2. 支持广播功能
3. 性能不低于TBE版本

# 详细设计（required）

## 算子分析

### 支持数据类型

float16、float32、bfloat16、int16、int8、uint8

### 支持形状

支持广播

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

tiling策略：

当不需要广播的情况下，算子计算过程不涉及数据的维度信息，故在host侧将数据视为一维向量，仅考虑数据个数，不考虑数据维度信息。

在广播的情况下，在host侧获取输入的相应shape大小以及各自的维度dim信息，首先，通过函数实现输入的shape维度补全和统一，遍历所有输入对应的维度大小得到其中最大的维度数，维度缺失的输入在首部维度补1实现维度扩充。再根据对应的广播规则，遍历各输入对应的shape，纵向比较求得最大的维度数以此得到最终的广播形状，根据最终广播形状通过指定接口获得total_length。需要将host侧获取的输入形状，最终广播形状以及total_length变量传到kernel侧。

任务均分：coreNum 根据输入长度和块大小动态调整，确保每个核心处理的数据块数均匀。

批量搬运：tileBlockNum 和 tileDataNum 计算单次搬运的数据量，通过 finalSmallTileNum 和 finalBigTileNum 确定小核/大核的搬运次数，将多次搬运合并为批量操作，减少冗余开销。尾块的处理逻辑确保不完整块也能被合并到计算流程中，避免数据碎片。

##### 1. 分核策略：

优先使用满核的原则。

如果核间能均分，可视作无大小核区分，大核小核数据块一致；

如果核间不能均分，需要将余出的数据块分配到前几个核上。

输入数据大小计算：通过GetInputShape和GetDataTypeLength函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。

UB内存大小和核心数量获取：通过平台信息获取UB内存大小和核心数量，并根据这些信息调整核心数量。

##### 2. 数据分块和内存优化策略：

充分使用UB空间的原则。

需要考虑不同硬件的UB大小不同、是否开启double buffer、kernel侧API实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。

UB内存大小获取：通过GetCoreMemSize函数获取UB内存的大小，用于后续的数据切分计算。

Tile块计算：根据UB内存大小和预定义的BLOCK_SIZE及BUFFER_NUM，计算出每个Tile块的数据数量。

数据切分：将输入数据按照计算出的Tile块大小进行切分，计算出每个core需要处理的数据块数量和最后一个block的剩余数据量。

设置切分参数：将计算出的切分参数（如每个core的数据量、Tile块大小等）设置到LogitTilingData对象中。

这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

##### 3. tilingkey规划策略：

需要tilingkey的情况：需要感知host侧信息对kernel侧走不同分支。在host侧获取value输入的大小，如果value为单值tilingkey为0(只对input_data、x1、x2三个输入进行广播)，否则tilingkey为1(所有输入均进行广播)。
数据检测：

#### 3.2.2 kernel侧设计：

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. int16类型可以直接转换成float32处理，int8和uint8数据先转成half再转成float32进行计算，计算完成后再转换回来。
2. 在kernel侧的copyin阶段完成广播数据的填充，首先获取host侧存储各个输入shape的一维数组以及按照广播规则获得的最终广播shape，通过特定的函数完成指定数组元素的地址映射。
3. 根据不同的tilingkey执行不同的核函数。
4. Ascend C的Logit算子流程见下图。
![logit.jpg](https://raw.gitcode.com/user-images/assets/7665709/b3a6b0d1-c364-4170-8fbe-cbac10e1d227/logit.jpg 'logit.jpg')
## 支持硬件

Atlas950
Atlas A2 训练系列产品
Atlas A3 系列产品

## 算子约束限制


# 可维可测分析

## 精度标准/性能标准


| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| -------- | -------------------- | -------- |
| 精度标准 | 不低于TBE版本        |          |
| 性能标准 | 不低于TBE版本        |          |

## 兼容性分析

新算子，不涉及兼容性分析

# 需求背景（required）

## 需求来源

## 背景介绍

### Sign算子实现优化

基于Sign算子历史TBE版本使用Ascend C编程语言进行优化。

源码路径：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/sign.py
算子信息库：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b

TBE 源码参考文件包含处理输入类型、广播逻辑以及量化计算的核心实现。

### Sign算子现状分析
通过对Sign算子TBE版本的功能分析，当前支持的能力如下：
① x支持 float16,float32,bfloat16,int32四种格式的输入。
② 不支持广播
③ 计算表达式：sign(x) = (x>0) ? 1 : (x<0) ? –1 : 0
④ 对于 float16/float32/bfloat16 输入，v200之后使用 vcmp+cast_to+vsub 方式计算；v200之前使用放大到±1附近的方式计算。对于 int32 输入，直接返回1/0/-1，不涉及浮点类型转换。
⑤ v200之后使用 vcmp、cast_to、vsub 接口实现计算；v200之前使用 vmins、vmaxs、vmuls 接口实现计算。

Sign算子TBE版本的整体流程图如下图所示：
![aaaa.png](https://raw.gitcode.com/user-images/assets/7649531/4b885916-524b-4fbf-b62e-fae738352d90/aaaa.png 'aaaa.png')


算子原型

| 名称 | 类别 | dtype     | format | shape  | 介绍                                           |
| ---- | ---- | --------- | ------ | ------ | ---------------------------------------------- |
| x    | 输入 |fp16/fp32/int32/bf16 | ND     | all    | 	输入张量                        |
| y    | 输出 |fp16/fp32/int32/bf16 | ND     | 同输入 | 	符号位结果                   |

算子支持型号
Atlas A2 训练系列产品/Atlas A3 系列产品

相关约束
Atlas A2 训练系列产品/Atlas 800I A2推理产品/Atlas A3 系列产品支持float16、float32、int32、bfloat16、int16(任务书要求新增int16实现)
x、y 的shape和dtype必须完全一致
张量最大维度数为8

# 需求分析（required）

## 需求描述

使用Ascend C编程语言实现Sign算子，支持float16/float/int32/bfloat16/int16数据类型

## 需求拆解

1. 支持float16/float/int32/bfloat16/int16数据类型
1. 性能不低于TBE版本

# 详细设计（required）

## 算子分析

### 数学公式

sign(x) = (x>0) ? 1 : (x<0) ? –1 : 0

### 支持数据类型

float16/float/int32/bfloat16/int16

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

由于输入形状相同且不支持广播，算子计算过程不涉及数据的维度信息，故在host侧将数据视为一维向量，仅考虑数据个数，不考虑数据维度信息。
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
设置切分参数：将计算出的切分参数（如每个core的数据量、Tile块大小等）设置到SignTilingData对象中。
这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

##### 3. tilingkey规划策略：

tilingkey固定为0，无需分支感知,在kernel侧做运行时分支。

#### 3.2.2 kernel侧设计：

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。
将bfloat16的数据转成float32进行计算，其余数据类型保持原类型计算。
无广播填充需求，直接在Compute阶段执行元素级运算。
根据tilingkey（固定）执行核函数。

Ascend C的Sign算子流程见下图。
#注:Ascend c在kernel侧无soc区分api,此处与tbe不相同。
![cussign.png](https://raw.gitcode.com/user-images/assets/7649531/5a6d5d32-4604-40e6-9f07-dda5294a2316/cussign.png 'cussign.png')


## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas A3| √ |

## 算子约束限制

不支持广播。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 不低于TBE版本 |  |
| 性能标准 | 不低于TBE版本 |  |

## 兼容性分析

新算子，不涉及兼容性分析
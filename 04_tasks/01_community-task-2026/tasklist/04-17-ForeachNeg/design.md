# 一、需求背景
## 1.1 需求来源
通过社区任务完成开源仓算子贡献的需求
## 1.2  背景介绍
### 1.2.1 ForeachNeg算子实现优化
基于aclnnForeachNeg算子使用Ascend C编程语言进行优化，使其支持DT_INT16/DT_INT8/DT_UINT8数据类型。
### 1.2.2 ForeachNeg算子现状分析
通过对ForeachNeg功能分析，当前支持的能力如下：

|  参数| 输入/输出/属性 | 数据类型 | 数据格式|
|--|--|--|--|
| x |输入  | float32、float16、int32、bfloat16 | ND|
|y| 输出|float32、float16、int32、bfloat16 | ND|

计算公式：

$$x = [x_0, x_1, \dots x_{n-1}]$$

$$y = [y_0, y_1, \dots y_{n-1}]$$

$$y_i = -x_i \ (i = 0, 1, \dots n-1)$$

### 1.2.3 ForeachNeg算子功能分析
ForeachNeg算子功能：计算输入张量列表中每个张量的相反数。

$$y_i = -x_i \ (i = 0, 1, \dots n-1)$$
输入：x

输出：y

支持数据类型：float32、float16、int32、bfloat16

### 1.2.4 算子原型

| 参数名                     | 输入/输出 | 描述                                                                 | 使用说明                                                                                                                               | 数据类型                          | 数据格式 | 维度(shape) | 非连续Tensor |
|----------------------------|-----------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|----------|-------------|--------------|
| x（aclTensorList*）        | 输入      | 表示进行相反数计算的输入张量列表，对应公式中的`x`。                  | - 支持空Tensor。<br>- 该参数中所有Tensor的数据类型保持一致。                                                                          | FLOAT32、FLOAT16、INT32、BFLOAT16 | ND       | 0-8         | √            |
| out（aclTensorList*）      | 输出      | 表示进行相反数计算的输出张量列表，对应公式中的`y`。                  | - 支持空Tensor。<br>- 该参数中所有Tensor的数据类型保持一致。<br>- 数据类型和数据格式与入参`x`的数据类型和数据格式一致，shape size大于等于入参`x`的shape size。 | FLOAT32、FLOAT16、INT32、BFLOAT16 | ND       | 0-8         | √            |
| workspaceSize（uint64_t*） | 输出      | 返回需要在Device侧申请的workspace大小。                              | -                                                                                                                                      | -                                 | -        | -           | -            |
| executor（aclOpExecutor**）| 输出      | 返回op执行器，包含了算子计算流程。                                   | -                                                                                                                                      | -                                 | -        | -           | -            |
# 二、需求分析
## 2.1 需求描述
使用Ascend C 编程语言实现ForeachNeg算子，使其支持DT_INT16/DT_INT8/DT_UINT8数据类型。
## 2.2 需求拆解
1.修改 https://gitcode.com/cann/ops-nn/tree/master/foreach/foreach_neg 目录下的文件，使其支持DT_INT16/DT_INT8/DT_UINT8数据

2.修改相应文档
# 三、需求详细设计
## 3.1 算子分析
### 3.1.1 数学公式
$$y_i = -x_i \ (i = 0, 1, \dots n-1)$$
### 3.2.2 支持数据类型
支持数据类型：float32、float16、int32、bfloat16

### 3.2.3 支持形状
列表中所有张量的shape必须完全一样，不支持广播。
## 3.2 算子实现
### 3.2.1 host侧设计
**tiling策略：**

ForeachNeg 算子不支持广播，所有输入张量形状必须完全一致。算子计算过程不涉及数据的维度信息，故在 host 侧将数据视为一维向量，仅考虑数据个数，不考虑数据维度信息。

在 host 侧获取输入张量列表中各张量的 shape 大小、数据长度，计算得到总数据长度total_length。需要将 host 侧获取的输入形状、总数据长度变量传到 kernel 侧。
任务均分：coreNum 根据输入长度和块大小动态调整，确保每个核心处理的数据块数均匀。

批量搬运：tileBlockNum 和 tileDataNum 计算单次搬运的数据量，通过 finalSmallTileNum 和 finalBigTileNum 确定小核 / 大核的搬运次数，将多次搬运合并为批量操作，减少冗余开销。尾块的处理逻辑确保不完整块也能被合并到计算流程中，避免数据碎片。

**1. 分核策略**

优先使用满核的原则。如果核间能均分，可视作无大小核区分，大核小核数据块一致；如果核间不能均分，需要将余出的数据块分配到前几个核上。

输入数据大小计算：通过 GetInputShape 和 GetDataTypeLength 函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。

UB 内存大小和核心数量获取：通过平台信息获取 UB 内存大小和核心数量，并根据这些信息调整核心数量。

**2. 数据分块和内存优化策略**

充分使用 UB 空间的原则。
需要考虑不同硬件的 UB 大小不同、是否开启 double buffer、kernel 侧 API 实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。

UB 内存大小获取：通过 GetCoreMemSize 函数获取 UB 内存的大小，用于后续的数据切分计算。

Tile 块计算：根据 UB 内存大小和预定义的 BLOCK_SIZE 及 BUFFER_NUM，计算出每个 Tile 块的数据数量。

数据切分：将输入数据按照计算出的 Tile 块大小进行切分，计算出每个 core 需要处理的数据块数量和最后一个 block 的剩余数据量。

设置切分参数：将计算出的切分参数（如每个 core 的数据量、Tile 块大小等）设置到 ForeachNegTilingData 对象中。

这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

**3. tilingkey 规划策略**

ForeachNeg 算子不支持广播，无需根据广播分支区分逻辑。

tilingkey 固定为 0，仅用于标识标准非广播批量取反计算分支。

**数据检测**

校验输入张量列表中所有张量的shape 必须完全一致、数据类型合法。

### 3.2.2 kernel 侧设计
进行 Init 和 Process 两个阶段，其中 Process 包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

由于支持 Ascend C 开发的硬件中逐元素取反指令支持 float16、float32 和 int32 数据的输入，可以直接将 bfloat16 的数据都转成 float32 进行计算，其余数据类型保持原类型计算。

ForeachNeg 不支持广播，因此 kernel 侧无需实现广播数据填充逻辑，直接进行连续 / 非连续数据的地址映射与读取。

根据固定 tilingkey 执行标准取反核函数。

CopyIn：直接从 GM 搬运输入张量到 UB，无需广播扩展。

Compute：对张量执行逐元素取反计算。

CopyOut：将计算结果写回 GM 输出张量。

ForeachNeg 算子流程遵循 Ascend C foreach 类单目逐元素算子标准流程。
![流程图.png](https://raw.gitcode.com/user-images/assets/9829721/f0fa6f5b-72ef-4ea7-858b-2c62b20ccf50/流程图.png '流程图.png')

## 3.3 支持硬件

| 支持的芯片版本 |涉及勾选  | 
|--|--|
| Atlas A2 训练系列产品/Atlas A3 系列产品 | √ | 
## 3.4 算子约束限制
不支持广播。

# 四、可维可测分析
## 4.1 精度标准/性能标准

无

## 4.2 兼容性分析
新算子不涉及兼容性。




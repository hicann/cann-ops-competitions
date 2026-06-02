# 需求背景（required）

## 需求来源

## 背景介绍

### Arange算子实现优化

基于Arange算子历史AscendC版本使用Ascend C编程语言进行优化。当前未找到该算子的TBE实现，因此本设计不做TBE实现反推，改为分析仓库内已有AscendC实现，并在现有实现基础上补齐任务书要求的数据类型与全核并行能力。

Arange算子（AscendC）实现路径和相关文件路径

Arange算子实现路径为：`experimental/math/arange`

Arange算子当前关键文件路径：

- 算子定义：`experimental/math/arange/op_host/arange_def.cpp`
- Shape推导：`experimental/math/arange/op_host/arange_infershape.cpp`
- Tiling实现：`experimental/math/arange/op_host/arange_tiling.cpp`
- Kernel入口：`experimental/math/arange/op_kernel/arange.cpp`
- Kernel实现：`experimental/math/arange/op_kernel/arange.h`
- Tiling数据结构：`experimental/math/arange/op_kernel/arange_tiling_data.h`

### Arange算子AscendC实现现状分析

通过对Arange算子现有AscendC版本的功能分析，当前支持的能力如下：
① start、end、step为Scalar输入，out为一维Tensor输出。
② 当前算子定义支持float32、float16、bfloat16、int32、int64五种类型，任务书要求支持float32、float16、bfloat16、int8、uint8、int16，需要调整算子注册和kernel模板实例化范围。
③ 当前kernel侧仅使用start和step参与计算，end不参与kernel计算；输出元素个数由out的shape size决定。
④ 计算表达式（等差序列公式）：

$$
out_i = start + i \times step,\quad i \in [0, N)
$$

其中N为输出Tensor元素个数，通常由aclnn侧根据start、end、step推导。

⑤ 对于float32输出，当前使用`KernelArange<float, float, float>`直接按float32计算并写回；对于其他类型，当前使用`KernelArange_Cast<TYPE_START, TYPE_STEP, TYPE_OUT>`，先将start、step转换为float32，再按float32生成序列，最后cast回输出类型。需要注意，int8/uint8不能直接cast到float32，需先转换为half，再由half转换为float32；回写int8/uint8时也需要先由float32转换为half，再由half转换为int8/uint8。
⑥ 当前tiling侧根据UB大小计算单次处理的unitNum，并设置`blockDim = 1`，因此现有版本为单核实现，尚未满足“所有核参与计算场景”的性能要求。
⑦ 当前尾块通过按32B对齐后的DataCopy写回处理，后续全核实现需重点处理尾块安全写回，避免最后一核越界写。

Arange算子历史AscendC版本的整体流程图如下图所示：
![tbee.png](https://raw.gitcode.com/user-images/assets/9516645/bebb1ed7-5e72-4cda-854f-95c8106c119e/tbee.png 'tbee.png')
算子原型

| 名称 | 类别 | dtype | format | shape | 介绍 |
|------|------|-------|--------|-------|------|
| start | 输入 | float32/float16/bfloat16/int8/uint8/int16 | ND | scalar | 获取值的范围的起始位置 |
| end | 输入 | float32/float16/bfloat16/int8/uint8/int16 | ND | scalar | 获取值的范围的结束位置 |
| step | 输入 | float32/float16/bfloat16/int8/uint8/int16 | ND | scalar | 获取值的步长 |
| out | 输出 | float32/float16/bfloat16/int8/uint8/int16 | ND | 1D | 输出等差序列Tensor |

相关约束
Atlas A2 训练系列产品/Atlas A3 系列产品支持float32、float16、bfloat16、int8、uint8、int16。
start、end、step、out的数据类型需要保持一致。
step不等于0。
step大于0时，start小于end；step小于0时，start大于end。
当前输入为Scalar，输出为一维ND Tensor，不支持非连续Tensor。

# 需求分析（required）

## 需求描述

使用Ascend C编程语言实现Arange算子，支持float32/float16/bfloat16/int8/uint8/int16数据类型，根据start、end、step生成一维等差序列。算子需满足泛化shape与正、负步长场景，并计划实现全核并行版本，使大shape场景下所有可用AI Core参与计算。

## 需求拆解

1. 支持float32/float16/bfloat16/int8/uint8/int16数据类型。
1. 支持正步长和负步长，满足合法输入范围约束。
1. 支持输出长度为1、小shape、32B对齐shape、非32B对齐尾块、大shape等泛化场景。
1. 基于现有AscendC单核实现扩展为全核实现。
1. 性能在所有核参与计算场景下不低于原对标要求；小shape低于10us且相差3us以内时，提供性能仿真图和分析结论。
1. 精度满足AscendOpTest工具默认阈值。

# 详细设计（required）

## 算子分析

### 数学公式

Arange算子生成一维等差序列，数学公式如下：

$$
out_i = start + i \times step,\quad i \in [0, N)
$$

递推形式为：

$$
out_{i+1} = out_i + step
$$

其中N为输出Tensor元素个数。对于正步长场景，输出区间为[start, end)；对于负步长场景，输出区间为(start, end]方向上的递减序列，实际仍由上层推导出的out shape控制输出长度。

在kernel内部，为便于向量化计算，每个tile先构造局部下标：

$$
idx = [0, 1, 2, ..., tileDataNum - 1]
$$

全核实现中，每个核处理一段连续全局输出区间，设当前tile的全局起始下标为globalOffset，则该tile输出为：

$$
out_{globalOffset + idx} = start + (globalOffset + idx) \times step
$$

这种计算方式保证各核之间无数据依赖，可直接按输出区间并行拆分。

### 支持数据类型

float32/float16/bfloat16/int8/uint8/int16

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

Host侧负责算子注册、shape推导、tiling计算、分核规划和tilingkey设置。当前历史实现已具备基础tiling能力，但仅设置`blockDim = 1`。本次设计预计参考Cross算子的tile均分思想和Sign算子的big/small core分核字段，将Arange改造为全核实现。

Shape推导策略保持现有实现：infer shape阶段将输出设置为一维动态shape，实际输出Tensor由aclnn侧根据start、end、step创建。Tiling阶段通过`context->GetOutputShape(0)->GetOriginShape().GetShapeSize()`获取输出元素总数totalNum。

Tiling数据建议从当前：

- dtypeSize
- totalNum
- unitNum
- unitLoops
- tailNum

扩展为全核实现所需字段：

- totalNum：输出元素总数。
- dtypeSize：输出数据类型字节数。
- tileDataNum：单次tile处理的元素数。
- smallCoreDataNum：普通核处理的数据量。
- bigCoreDataNum：多分配一个32B block的核处理的数据量。
- finalSmallTileNum：普通核tile循环次数。
- finalBigTileNum：大核tile循环次数。
- smallTailDataNum：普通核最后一个tile的数据量。
- bigTailDataNum：大核最后一个tile的数据量。
- tailBlockNum：大核数量，即前tailBlockNum个核每核多处理一个32B block。

##### 1. 分核策略：
采用按32B block均分策略，确保每个参与计算的核至少处理一个32B block，避免无效空核。

首先通过平台信息获取：
AIV核数coreNum
UB大小ubSize

然后根据输出shape和dtype计算：
totalNum：输出元素总数
dtypeSize：输出类型字节数
totalBytes：totalNum * dtypeSize
totalBytesAlign32：totalBytes按32B向上对齐后的字节数
totalBlockNum：totalBytesAlign32 / 32

实际使用核数：

```text
if tileDataNum >= totalNum:
    usedCoreNum = 1
else:
    usedCoreNum = min(coreNum, totalBlockNum)
```

按32B block均分：

```text
everyCoreBlockNum = totalBlockNum / usedCoreNum
tailBlockNum = totalBlockNum % usedCoreNum
smallCoreDataNum = everyCoreBlockNum * 32 / dtypeSize
bigCoreDataNum = (everyCoreBlockNum + 1) * 32 / dtypeSize
```

前tailBlockNum个核为big core，处理bigCoreDataNum个元素；其余核为small core，处理smallCoreDataNum个元素。kernel侧根据`GetBlockIdx()`判断当前核类型并计算全局起始偏移：

```text
if blockIdx < tailBlockNum:
    coreDataNum = bigCoreDataNum
    globalOffset = blockIdx * bigCoreDataNum
else:
    coreDataNum = smallCoreDataNum
    globalOffset = tailBlockNum * bigCoreDataNum + (blockIdx - tailBlockNum) * smallCoreDataNum
```

由于最后一个核可能覆盖到按32B对齐补齐后的无效区域，kernel写回时需要结合totalNum裁剪真实有效数据量，最后tile使用DataCopyPad或按有效长度安全写回。

##### 2. 数据分块和内存优化策略：
遵循“尽量使用UB，同时保证中间buffer可容纳”的原则。

Host侧根据UB大小、dtype字节数和kernel临时buffer数量计算tileDataNum。Arange计算不需要搬入大输入Tensor，只需要读取start、step两个Scalar，然后生成输出。因此UB主要用于：

float32路径：
outQueue：输出tile
calc_init：tile内基础序列
calc_step：unit偏移或step向量
calc_temp：局部下标/累加偏移

cast路径：
inQueue：读取start、step scalar
outQueue：输出tile
calc_init/calc_step/calc_temp/calc_out：float32中间计算buffer
tempFloat：scalar转换临时buffer
tempHalf：int8/uint8两步转换临时buffer

tileDataNum按32B对齐元素数计算，建议公式如下：

```text
alignElem = 32 / dtypeSize
usableUb = ubSize / ubBufferCount
tileDataNum = floor(usableUb / dtypeSize / alignElem) * alignElem
tileDataNum = max(tileDataNum, alignElem)
```

其中ubBufferCount根据dtype和路径设置。float32路径buffer数量较少，可取较大tile；float16/bfloat16/int16路径由于统一转float32计算，需要预留更多float32中间buffer。int8/uint8虽然输出字节较小，但中间计算仍按float32分配UB，并且受Ascend C Cast接口约束，需要额外预留half中间buffer用于`int8/uint8 -> half -> float32`和`float32 -> half -> int8/uint8`两步转换，应避免仅按输出dtype低估UB占用。

每个核内部再按tileDataNum循环：

```text
tileNum = ceil(coreDataNum / tileDataNum)
processDataNum = min(tileDataNum, coreDataNum - tileIdx * tileDataNum)
globalTileOffset = globalOffset + tileIdx * tileDataNum
```

计算时直接使用globalTileOffset生成当前tile对应的全局序列值：

```text
out = start + (globalTileOffset + localIdx) * step
```

这种方式不需要跨tile累加状态，各tile可以独立计算，也避免多核场景下单核历史实现的`calc_temp`偏移逻辑扩展复杂度。

##### 3. tilingkey规划策略：


```text
tilingkey = 1：float32专用路径
tilingkey = 0：float16/bfloat16/int8/uint8/int16 cast路径
```

float32路径不需要输入/输出cast，直接按float32生成输出；其他类型统一走cast路径。float16、bfloat16、int16的start和step可转换为float32后计算，计算完成后按目标类型cast写回。int8/uint8路径需要单独在cast路径中处理两步转换：读取Scalar后先转half，再转float32参与计算；输出写回前先将float32结果转half，再由half转int8/uint8。

如后续发现int8/uint8/int16的cast舍入或饱和行为与aclnnArange不一致，可进一步拆分tilingkey，为整数类型增加独立kernel路径。若int8/uint8两步转换导致UB占用或性能差异明显，也可为int8/uint8单独规划tilingkey和tileDataNum计算逻辑。

#### 3.2.2 kernel侧设计：

kernel侧进行Init和Process两个阶段，其中Process包括计算准备（Prepare）、计算（Compute）、搬出（CopyOut）三个阶段。由于Arange只有Scalar输入，不需要像普通二元算子一样搬入整块输入Tensor。

Init阶段：
读取tiling数据，确定当前核的数据范围。根据`GetBlockIdx()`、`tailBlockNum`、`bigCoreDataNum`、`smallCoreDataNum`计算当前核的globalOffset、coreDataNum、tileNum和tailDataNum，并初始化GM Tensor和UB队列。

Prepare阶段：
读取start、step两个Scalar。float32路径可直接通过GlobalTensor读取；cast路径先将Scalar拷贝到UB，再转换为float32。对于float16、bfloat16、int16，可按支持的Cast链路转换到float32；对于int8/uint8，需要先Cast到half临时Tensor，再Cast到float32。随后构造tile内局部下标向量localIdx。

Compute阶段：
对每个tile执行如下计算：

```text
globalIdx = globalOffset + tileIdx * tileDataNum + localIdx
outFloat = startFloat + globalIdx * stepFloat
```

对于float32输出，outFloat直接写入输出LocalTensor。
对于float16/bfloat16/int16输出，outFloat按目标类型cast到输出LocalTensor。
对于int8/uint8输出，outFloat先cast到half临时Tensor，再由half cast到int8/uint8输出LocalTensor。

CopyOut阶段：
将输出LocalTensor写回out GM。非尾块使用普通DataCopy；尾块和最后一个核可能存在非32B对齐或对齐补齐数据，需要使用DataCopyPad/按有效元素数写回，保证只写真实输出范围内的数据。

数据类型处理策略如下：

对于float32数据类型，直接使用float32计算和写回。

对于float16和bfloat16数据类型，start、step转换为float32计算，计算完成后转换回float16/bfloat16。

对于int16数据类型，start、step转换为float32计算，计算完成后转换回int16。

对于int8、uint8数据类型，由于Cast接口不支持int8/uint8直接转换到float32，需要采用两步转换：

```text
输入Scalar：int8/uint8 -> half -> float32
输出Tensor：float32 -> half -> int8/uint8
```

int8/uint8路径需额外申请half临时buffer，并在测试中重点验证两步转换带来的舍入行为、负步长、边界值和可能的溢出场景。

end输入不在kernel中参与计算，仅用于上层输出长度推导和合法性校验。kernel侧以totalNum作为最终写回边界。

Ascend C的Arange算子流程见下图。
![ascc.png](https://raw.gitcode.com/user-images/assets/9516645/d7d894e2-0a72-4f6b-9887-794fad915a36/ascc.png 'ascc.png')
## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |

## 算子约束限制

start、end、step为Scalar输入。
out为一维ND Tensor输出。
start、end、step、out的数据类型保持一致。
step不等于0。
step大于0时，start小于end；step小于0时，start大于end。
当前不支持非连续Tensor。
kernel侧不使用end参与计算，要求上层正确推导out shape。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 不低于TBE版本 |  |
| 性能标准 | 不低于TBE版本 |  |

## 兼容性分析

全核实现会改变历史单核tiling和kernel数据划分方式，但输出数学公式不变。需要重点保证：
多核拆分后全局下标连续且不重不漏。
最后一个核和最后一个tile不会越界写。
float32路径与cast路径输出结果保持与历史语义一致。

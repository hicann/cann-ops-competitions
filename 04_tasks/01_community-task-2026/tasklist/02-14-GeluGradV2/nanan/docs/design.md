一、需求背景

1.1 需求来源

通过社区任务完成开源仓算子贡献的需求

1.2 背景介绍

1.2.1 GeluGradV2算子实现优化

1.2.1.1 TBE算子源码路径：${ASCEND_INSTALL_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/GeluGradV2.py

1.2.1.2 算子原型路径：${ASCEND_INSTALL_PATH}/opp/built-in/op_proto/inc/nonlinear_fuc_ops.h

1.2.1.3
算子信息库路径：${ASCEND_INSTALL_PATH}/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json
 中的GeluGradV2

1.2.2 GeluGradV2算子现状分析

1.2.2.1 TBE算子支持的数据类型和数据格式

通过对GeluGradV2算子TBE版本的功能分析，当前支持的能力如下：

1）数据类型：float16,float,bfloat16

2）数据排布：FRACTAL_NZ,NC1HWC0,ND

1.2.2.2 TBE算子实现描述

**功能概述**

GeluGradV2算子用于计算高斯误差线性单元（GELU）激活函数的梯度。它接收上游梯度input_dy和前向算子的输入input_x，计算并输出损失函数对input_x的梯度output_z。

**数据类型支持**

输入input_dy与input_x支持float16、float32及bfloat16三种数据格式，且两者的数据类型必须保持一致。输出数据类型与输入数据类型相同。

**计算精度与处理流程**

为保障计算精度，核心计算过程在float32精度下进行。当输入为float16或bfloat16时，算子内部会将其提升转换（cast）为float32进行计算，待计算完成后，再将结果转换回输入的原始数据类型。

**计算模式与实现**

算子的计算行为由approximate属性控制：

精确模式（approximate=”none”）：默认模式。基于误差函数(erf)实现，其核心数学原理为 output = dy *
[cdf(x) + x * pdf(x)]，其中cdf(x) = 0.5 * (1 + erf(x / √2))，pdf(x) =
exp(-x²/2) / √(2π)。为在不同硬件上实现最优性能与精度平衡，此模式内部根据芯片型号自动选择不同的深度优化实现：

在Ascend910B上，调用gelu_grad_compute_none_v3函数，其核心是计算output = dy * 0.5 *
erfc(-x/√2) + dy * (x / √(2π)) *
exp(-x²/2)，其中erfc通过特定系数的多项式分式(num(x)/den(x)) * exp(-x²)进行高精度近似。

在Ascend310P上，调用gelu_grad_compute_none_v2函数，其数学表达式与V3一致，但erfc采用了另一组不同的多项式系数进行近似计算，以适配该硬件特性。

在其他硬件上，调用基础实现gelu_grad_compute_none，通过组合tbe.vexp、tbe.vadd等算子直接计算cdf(x)与pdf(x)。

近似模式（approximate=”tanh”）：基于双曲正切函数进行近似计算，调用gelu_grad_compute_tanh函数实现。

**形状与广播**

本算子为逐元素操作，输出张量output_z的形状与输入input_x的形状完全相同。支持动态形状（Dynamic Shape）与动态秩（Unknown Rank）的输入。

**类型安全**

通过严格的数据类型转换流程确保类型安全。所有输入首先被统一至float32精度进行计算，最终结果会精确转换回与输入input_x一致的数据类型（包括bfloat16的特定舍入处理），确保输入输出类型的一致性。

1.2.2.3 TBE算子实现流程图

1）主流程图

![gelu_grad_v2_tbe流程图-0319.jpg](https://raw.gitcode.com/user-images/assets/7665709/6cb50b9e-0bfa-46db-9ca2-cc027ff25850/gelu_grad_v2_tbe%E6%B5%81%E7%A8%8B%E5%9B%BE-0319.jpg)

2）gelu_grad_compute_none流程图

![gelu_grad_compute_none流程图.png](https://raw.gitcode.com/user-images/assets/7665709/d5e9d33b-6269-4932-a77c-5c18ab56697a/gelu_grad_compute_none%E6%B5%81%E7%A8%8B%E5%9B%BE.png)

3）gelu_grad_compute_none_v2(Ascend310P优化)

![gelu_grad_compute_none_v2.png](https://raw.gitcode.com/user-images/assets/7665709/366fb3d5-6fbe-4da1-9d75-5c9742fb2935/gelu_grad_compute_none_v2.png)

4）gelu_grad_compute_none_v3 (Ascend910B优化)

![gelu_grad_compute_none_v3 (Ascend910B优化).png](https://raw.gitcode.com/user-images/assets/7665709/86afe7fd-4c73-4509-8945-4702aff573fd/gelu_grad_compute_none_v3%E2%80%8B__Ascend910B%E4%BC%98%E5%8C%96_.png)

5）gelu_grad_compute_tanh

![gelu_grad_compute_tanh.png](https://raw.gitcode.com/user-images/assets/7665709/bff87870-523b-414b-b0e7-c83f7fce170f/gelu_grad_compute_tanh.png)

（1）gelu_grad_compute_tanh 公式

![图片.png](https://raw.gitcode.com/user-images/assets/7665709/a26e4dbe-7677-4772-9b25-43dc1c3f0717/%E5%9B%BE%E7%89%87.png)

（2）gelu_grad_compute_tanh_v2 公式

![图片.png](https://raw.gitcode.com/user-images/assets/7665709/0b83a83f-c83a-4002-a481-9a7ef510dfdd/%E5%9B%BE%E7%89%87.png)

（3）原始复杂实现公式

![图片.png](https://raw.gitcode.com/user-images/assets/7665709/1b470878-3f36-4a03-a42b-0c3f504c7c84/%E5%9B%BE%E7%89%87.png)

二、需求分析

2.1 外部组件依赖

不涉及外部组件依赖。

2.2 内部适配模块

适配 Aclnn 接口和图模式调用，基于 TBE (Tensor Boost Engine) 开发，支持动态形状（Dynamic Shape）及动态秩（Unknown Rank）场景。

2.3 需求模块设计

2.3.1 AscendC算子原型

与op_api接口参数保持一致

| 名称        | 类别         | dtype          | format                | shape  | 介绍                                                                                                          |
| ----------- | ------------ | -------------- | --------------------- | ------ | ------------------------------------------------------------------------------------------------------------- |
| input_dy    | 输入         | fp16/fp32/bf16 | FRACTAL_NZ,NC1HWC0,ND | all    | 输入                                                                                                          |
| input_x     | 输入         | fp16/fp32/bf16 | FRACTAL_NZ,NC1HWC0,ND | 同dy   | 输入                                                                                                          |
| output_z    | 输出         | fp16/fp32/bf16 | FRACTAL_NZ,NC1HWC0,ND | 同输入 | 输出                                                                                                          |
| approximate | 可选输入属性 | str            | scale                 | 1      | 计算使用的激活函数模式, 默认"none"，取值："none"或者"tanh"，其中"none"代表使用erf模式，"tanh"代表使用tanh模式 |

2.3.1 AscendC算子相关约束

1. 输入一致性：input_dy 与 input_x 的数据类型（dtype）和形状（shape）必须完全一致。
2. 广播机制：暂不支持 input_dy 与 input_x 之间的自动广播操作，两者的 Shape 必须严格一致。
3. 精度约束：内部计算优先转换为 float32 进行以保证精度，输出时会回转至输入类型（bfloat16 类型采用 Round 舍入，其他类型采用 Cast 截断）。

三、需求详细设计

3.1 使能方式

Aclnn直调。

3.2 需求总体设计

3.2.1host侧设计

可以将host侧将数据视为一维向量，不需要考虑到排布对实现的影响，只需要考虑数据个数，不考虑数据维度信息。

任务均分：coreNum 根据输入长度和块大小动态调整，确保每个核心处理的数据块数均匀。

批量搬运：GeluGradV2BlockNum 和 GeluGradV2DataNum 计算单次搬运的数据量，通过
finalSmallGeluGradV2Num 和 finalBigGeluGradV2Num
确定小核/大核的搬运次数，将多次搬运合并为批量操作，减少冗余开销。尾块的处理逻辑确保不完整块也能被合并到计算流程中，避免数据碎片。

3.2.1.1分核策略

优先使用满核的原则。

如果核间能均分，可视作无大小核区分，大核小核数据块一致；

如果核间不能均分，需要将余出的数据块分配到前几个核上。

输入数据大小计算：通过GetInputShape和GetDataTypeLength函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。

UB内存大小和核心数量获取：通过平台信息获取UB内存大小和核心数量，并根据这些信息调整核心数量。

3.2.1.2数据分块和内存优化策

充分使用UB空间的原则。

需要考虑不同硬件的UB大小不同、是否开启double buffer、kernel侧API实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。

UB内存大小获取：通过GetCoreMemSize函数获取UB内存的大小，用于后续的数据切分计算。

GeluGradV2块计算：根据UB内存大小和预定义的BLOCK_SIZE及BUFFER_NUM和不同类型下的ubDataNum，计算出每个GeluGradV2块的数据数量。

数据切分：将输入数据按照计算出的GeluGradV2块大小进行切分，计算出每个core需要处理的数据块数量和最后一个block的剩余数据量。

设置切分参数：将计算出的切分参数（如每个core的数据量、GeluGradV2块大小等）设置到SelectV2TilingData对象中。

这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

3.2.1.3tilingKey规划策略

根据不同tilingkey，启动不同kernel。

3.2.2kernel侧设计

3.2.2.1kernel侧实现描述

kernel侧分为Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1）将非fp32数据精度转换为fp32计算后，在转换为原始精度类型。

3.2.2.2AscendC实现流程图

![gelu_grad_v2_AscendC流程图.jpg](https://raw.gitcode.com/user-images/assets/7665709/ba144995-d059-4e37-97e7-d87e87d82397/gelu_grad_v2_AscendC%E6%B5%81%E7%A8%8B%E5%9B%BE.jpg)

3.2.2.3AscendC实现流程图与TBE流程图存在的差异点和原因

相对于TBE流程图，仅保留了适配硬件平台部分，没有TBE适配其它硬件平台的流程。

3.3 支持硬件

Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件

3.4 算子约束限制

仅支持“Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件”，其它平台可能会有精度问题。

四、特性交叉分析

无

五、可维可测分析

5.1精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| -------- | -------------------- | -------- |
| 精度标准 | 不低于TBE版本        |          |
| 性能标准 | 不低于TBE版本        |          |

5.2兼容性分析

新算子，不涉及兼容性分析

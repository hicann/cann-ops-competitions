# NllLoss 算子设计文档

# 需求背景

## 需求来源

根据 CANN 训练营 2026 暑期季西安交通大学专场 NllLoss 算子开发任务书，参考 CANN 内置 aclnnNLLLoss 与历史 TBE 实现，在 Atlas A2 训练系列产品和 Atlas A3 系列产品上使用 Ascend C 完成等价实现。

NllLoss（Negative Log Likelihood Loss，负对数似然损失）通常接收 LogSoftmax 的输出和目标类别索引。算子从每个样本的类别维中选取目标类别值，乘以类别权重并取负，再根据 reduction 选择逐样本输出、求和或加权平均，同时给出有效样本权重之和。

本文档参考以下资料：

- [NllLoss 社区任务书](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/NllLoss_task_doc.md)
- [NLLLoss 官方开源目录](https://gitcode.com/cann/ops-nn/tree/master/loss/nll_loss)
- [NLLLoss 算子原型](https://gitcode.com/cann/ops-nn/blob/master/loss/nll_loss/op_graph/nll_loss_proto.h)
- [aclnnNLLLoss 接口文档](https://gitcode.com/cann/ops-nn/blob/master/loss/nll_loss/docs/aclnnNLLLoss.md)
- [aclnnNLLLoss2d 接口文档](https://gitcode.com/cann/ops-nn/blob/master/loss/nll_loss/docs/aclnnNLLLoss2d.md)
- [参考 PR 587](https://gitcode.com/cann/cann-ops-competitions/pull/587)

任务书中的算子名称写作 NllLoss，官方内部算子注册名为 NLLLoss，ACLNN 接口名为 aclnnNLLLoss。本文在描述任务时使用 NllLoss，在描述代码实体时保留官方大小写。

TBE 参考路径：

- Kernel 实现：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/
- 算子原型：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/
- 算子信息库：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/

## 功能语义

### 算子原型

官方 NLLLoss 原型包含三个输入、两个输出和两个属性：

~~~cpp
REG_OP(NLLLoss)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(target, TensorType({DT_INT32, DT_INT64, DT_UINT8}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(total_weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
    .OP_END_FACTORY_REG(NLLLoss)
~~~

参数定义：

| 名称 | 类别 | 说明 |
| --- | --- | --- |
| x | 输入 | 对数概率或待计算的类别值，类别维为 C。 |
| target | 输入 | 目标类别索引。每个元素必须位于 [0, C) 或等于 ignore_index。 |
| weight | 可选输入 | 每个类别的缩放权重，长度为 C；缺省时各类别权重均为 1。 |
| reduction | 属性 | none、mean 或 sum，默认 mean。 |
| ignore_index | 属性 | 被忽略的 target 值，默认 -100。 |
| y | 输出 | reduction 为 none 时输出逐位置损失，否则输出单元素结果。 |
| total_weight | 输出 | 有效位置的类别权重之和，单元素输出。 |

### 数学定义

将 target 中的位置展平为 p，令 t_p = target[p]。有效标志、有效权重和单位置损失定义为：

~~~text
valid_p = (t_p != ignore_index)

effective_weight_p =
    0                    , valid_p == false
    weight[t_p]          , valid_p == true 且提供 weight
    1                    , valid_p == true 且未提供 weight

loss_p = -effective_weight_p * x_at(p, t_p)
total_weight = sum(effective_weight_p)
~~~

x_at 的地址规则：

~~~text
x 为 [C]：
    x_at(0, t) = x[t]

x 为 [N, C]：
    x_at(n, t) = x[n, t]

x 为 [N, C, H, W]：
    p = (n, h, w)
    x_at(p, t) = x[n, t, h, w]
~~~

归约规则：

~~~text
reduction == none:
    y[p] = loss_p

reduction == sum:
    y = sum(loss_p)

reduction == mean:
    y = sum(loss_p) / total_weight
~~~

mean 是按有效类别权重之和进行加权平均，不是简单除以 target 元素数。全部位置均被忽略或 total_weight 为 0 时，mean 输出 NaN；sum 输出 0，total_weight 输出 0。该行为与 PyTorch NLLLoss 和官方 ACLNN 空输入处理保持一致。

### Shape 组合

| 场景 | x | target | reduction=none 的 y | sum/mean 的 y |
| --- | --- | --- | --- | --- |
| 无 batch | [C] | 标量 | 标量或单元素 | 标量或单元素 |
| 二维 | [N, C] | [N] | [N] | 标量或单元素 |
| 二维空间 | [N, C, H, W] | [N, H, W] | [N, H, W] | 标量或单元素 |

约束如下：

- x 仅支持 1 维、2 维或 4 维。
- target 分别为 0 维、1 维或 3 维，并与 x 的 N、H、W 对应。
- weight 为可选 1 维 Tensor，提供时长度必须等于 C。
- y 与 total_weight 的 dtype 跟随 x。
- total_weight 为单元素输出；在 ACLNN 接口中 reduction 为 none 时其值不作为有效结果使用。
- 不支持广播。

### 数据类型

| 参数 | 支持类型 | Kernel 处理 |
| --- | --- | --- |
| x | float16、float32、bfloat16 | 选值后统一提升为 float32 计算。 |
| target | int32、int64、uint8 | Kernel 按模板类型读取，uint8 提升为 int32。 |
| weight | 与 x 相同 | 缓存或按 target 间接读取，再提升为 float32。 |
| y | 与 x 相同 | float32 结果按输出类型转换。 |
| total_weight | 与 x 相同 | float32 累加结果按输出类型转换。 |

float16 和 bfloat16 使用 float32 完成乘法与归约，减少长序列求和误差。float32 路径也使用 float32 累加，与现有接口的计算精度保持一致。

### 数据格式

内部算子支持 ND。ACLNN 侧负责将非连续 Tensor 连续化后再调用内部算子。NLLLoss2d 的 4 维输入按 NCHW 语义解释，Kernel 直接使用 N、C、H、W 计算偏移，不依赖 NC1HWC0 或 FRACTAL_NZ 等特殊格式。

### 特殊输入

- target 等于 ignore_index 时，不读取 x[target] 和 weight[target]，该位置的 loss 与有效权重均为 0。
- target 除 ignore_index 外必须位于 [0, C)，非法索引不进行静默 clamp。
- 被选中的 x 或 weight 为 NaN 时，结果按 IEEE 浮点规则传播。
- 被选中的 x 为正负 Inf 时按乘法规则传播；0 权重乘 Inf 的结果为 NaN，不做额外改写。
- weight 可为 0、负数或非整数，不额外限制其符号。
- 空 target 在 none 模式输出空 Tensor；sum 输出 0；mean 输出 NaN；total_weight 输出 0。
- x 的类别数 C 为 0 且 target 非空属于非法输入。

## 现状分析

NllLoss 的主要成本不是普通逐元素计算，而是 target 驱动的间接访存和可选跨核归约：

1. 每个 target 位置只读取 x 中一个类别值，直接搬运整行 x 会产生 C 倍以上无效流量。
2. target 分布不可预测，x 和 weight 访问地址不连续，难以形成普通连续向量搬运。
3. none 模式各位置互不依赖，适合按 target 元素并行。
4. sum 和 mean 模式需要同时归约 loss 与 effective_weight。
5. 4 维输入的类别维不是最后一维，地址为 n×C×H×W + t×H×W + h×W + w。
6. 小 shape 更受 Kernel 启动和同步开销影响，大 shape 更受随机访存和跨核归约影响。

TBE 计算可抽象为“索引读取、加权取负、归约写回”三个阶段：

~~~mermaid
flowchart LR
    A["x / target / weight"] --> B["读取 target 并判断 ignore_index"]
    B --> C["按 target 间接读取 x 与 weight"]
    C --> D["float32 计算 loss 与有效权重"]
    D --> E{"reduction"}
    E -- "none" --> F["逐位置写回 y"]
    E -- "sum" --> G["归约 loss"]
    E -- "mean" --> H["归约 loss 与 weight 后相除"]
    G --> I["写回 y / total_weight"]
    H --> I
~~~

# 需求分析

## 外部依赖

算子不依赖第三方组件，使用以下 CANN 基础能力：

- Ascend C Kernel 编程接口和 kernel_operator.h。
- op_host 的 InferShape、InferDataType、Tiling 和平台信息接口。
- ACLNN 两段式执行框架。
- DataCopyPad、Cast、Gather、Mul、ReduceSum、Select、SyncAll 等基础接口。
- ops-nn 仓库的公共构建、测试和算子注册能力。

## 模块划分

| 模块 | 职责 |
| --- | --- |
| op_graph | 声明 NLLLoss 输入、输出、属性和类型范围。 |
| op_host | Shape 与 dtype 推导、参数校验、分核、UB 切分、TilingKey 和 workspace 规划。 |
| op_kernel | target 搬入、间接取值、权重处理、loss 计算、局部归约和跨核归约。 |
| op_api | aclnnNLLLoss 与 aclnnNLLLoss2d 的参数检查、连续化、空 Tensor 处理和结果拷贝。 |
| tests | InferShape、Tiling、Kernel、ACLNN、精度、泛化和性能验证。 |
| docs | API 文档、设计文档、自验证报告和 README。 |

## 支持型号

Atlas A2 训练系列产品、Atlas 800I A2 推理产品和 Atlas A3 系列产品，对应 ascend910b 与 ascend910_93。

# 详细设计

## 使能方式

| 上层框架 | 涉及勾选 |
| --- | --- |
| TensorFlow 训练/推理 | |
| PyTorch 训练/推理 | √ |
| ATC 推理 | √ |
| ACLNN 直调 | √ |
| OPAT 调优 | |
| SGAT 子图切分 | |

## 总体架构

整体采用“按 target 位置分核、按 tile 搬运 target、按索引读取选中值、float32 局部计算、按需跨核归约”的方案。

~~~mermaid
flowchart TD
    A["ACLNN 参数检查与连续化"] --> B["InferShape / InferDataType"]
    B --> C["Host Tiling"]
    C --> D{"执行模式"}
    D -- "none" --> E["多核独立输出"]
    D -- "小规模 sum/mean" --> F["单核归约"]
    D -- "大规模 sum/mean" --> G["多核局部归约"]
    G --> H["workspace + SyncAll"]
    H --> I["core0 固定顺序汇总"]
    E --> J["结果写回"]
    F --> J
    I --> J
~~~

## ACLNN 设计

### 参数检查

aclnnNLLLossGetWorkspaceSize 与 aclnnNLLLoss2dGetWorkspaceSize 完成以下检查：

1. self、target、weight、out、totalWeightOut、workspaceSize 和 executor 指针合法。
2. self 支持 float16、float32、bfloat16。
3. target 支持 int32、int64、uint8。
4. weight 与 self dtype 一致，长度等于 C。
5. reduction 为 0、1、2，分别对应 none、mean、sum。
6. self 与 target 的 rank 和维度关系符合接口约束。
7. out 与 totalWeightOut 的元素个数和 dtype 可接受内部结果。

内部 NLLLoss 原型保留可选 weight。上层接口未提供类别权重时，可在 API 适配层构造等价的全 1 权重语义，或由内部算子通过 hasWeight 分支处理。

### 连续化

- self、target、weight 使用 Contiguous 转换为连续 Tensor。
- 非连续 out 和 totalWeightOut 通过 ViewCopy 写回。
- uint8 target 可在 API 侧提升为 int32，也可由 Kernel 模板直接读取。
- 为保持接口 dtype，内部 float32 结果在返回前转换为 out 和 totalWeightOut 的 dtype。

### 空 Tensor

空输入不启动 NllLoss AICore Kernel：

| reduction | out | totalWeightOut |
| --- | --- | --- |
| none | 空输出 | 不作为有效结果使用 |
| sum | 0 | 0 |
| mean | NaN | 0 |

### 两段式接口

~~~cpp
aclnnStatus aclnnNLLLossGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weight,
    int64_t reduction,
    int64_t ignoreIndex,
    aclTensor *out,
    aclTensor *totalWeightOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnNLLLoss(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
~~~

NLLLoss2d 使用同样的两段式结构，输入 shape 为 [N, C, H, W] 和 [N, H, W]。

## Host 设计

### Shape 推导

~~~text
total_weight.shape = scalar

x.rank == 1:
    y.shape = scalar

x.rank == 2:
    reduction == none ? y.shape = [N] : y.shape = scalar

x.rank == 4:
    reduction == none ? y.shape = [N, H, W] : y.shape = scalar
~~~

动态 shape 场景中，未知 rank 传递未知 rank；已知 rank 时严格按上述规则推导。

### dtype 推导

~~~text
y.dtype = x.dtype
total_weight.dtype = x.dtype
~~~

target dtype 不影响输出 dtype。

### 规模展开

Host 将不同 rank 映射为统一的逻辑位置数 M：

| x rank | M | C | 位置到 x 的映射 |
| --- | --- | --- | --- |
| 1 | 1 | x[0] | t |
| 2 | N | x[1] | p×C+t |
| 4 | N×H×W | x[1] | n×C×H×W+t×H×W+h×W+w |

4 维模式额外下发 H、W、H×W 和 C×H×W，Kernel 避免循环内重复乘法。

### 参数校验

| 校验项 | 规则 |
| --- | --- |
| x rank | 1、2 或 4。 |
| target rank | 分别为 0、1 或 3。 |
| target shape | 与 x 的 N、H、W 对应。 |
| C | C 大于 0，或 target 为空。 |
| weight | 缺省或 shape=[C]，dtype 与 x 一致。 |
| reduction | none、mean、sum。 |
| 平台 | AIV 核数和 UB 大小有效。 |

target 的数值范围属于运行时数据约束，Host 无法读取 Device Tensor 内容。Kernel 必须先判断 ignore_index，再计算访存地址，不允许对 ignore_index 形成越界地址。

### 调度模式

Host 根据 M、reduction 和每核工作量选择四种模式：

| 模式 | 条件 | 方案 |
| --- | --- | --- |
| scalar | x 为 1 维或 M 很小 | 单核直接读取并输出。 |
| none_parallel | reduction=none 且 M 足够大 | 多核按位置切分，直接写 y。 |
| reduce_single | sum/mean 且 M 较小 | 单核完成，避免 SyncAll。 |
| reduce_multi | sum/mean 且 M 足够大 | 多核局部归约，core0 汇总。 |

分核以 target 位置数 M 为主：

~~~text
minPositionsPerCore = 根据实测确定的最小工作量
usedCoreNum = min(aivCoreNum, ceilDiv(M, minPositionsPerCore))
usedCoreNum = max(usedCoreNum, 1)
positionsPerCore = ceilDiv(M, usedCoreNum)
usedCoreNum = ceilDiv(M, positionsPerCore)
~~~

性能调优阶段通过 CANN Judge 和 profiler 校准 minPositionsPerCore，不针对单个固定 case 写死。

### UB 切分

每个 tile 的 UB 缓冲包括：

| 缓冲 | 用途 |
| --- | --- |
| targetQueue | 连续搬入 target。 |
| selectedRawBuf | 每个位置的选中 x 值，按 32B 槽位暂存。 |
| selectedFloatBuf | 压紧并转换后的 float32 x 值。 |
| weightRawBuf | 按 target 读取的原始权重。 |
| weightFloatBuf | float32 有效权重。 |
| lossFloatBuf | float32 单位置损失。 |
| outputQueue | none 模式输出。 |
| reduceTmpBuf | 核内 ReduceSum 临时空间。 |
| weightCacheBuf | C 较小时缓存完整 weight。 |

weight 模式：

- C×sizeof(weight) 能放入预留 UB 时，每核只搬入一次 weight，后续在 UB 内 Gather。
- weight 过大时，不缓存整段 weight，按 target 从 GM 间接读取。
- 未提供 weight 时直接生成 1，不分配 weightRawBuf。

tilePositions 根据 UB 大小、target 字节数、输出字节数、32B 选值槽位和 float32 临时空间计算，并按 32B 与矢量 API 对齐要求向下取整。tilePositions 至少为 1。

### TilingKey

TilingKey 由以下维度组合：

| 字段 | 取值 |
| --- | --- |
| xMode | 0=FP16，1=FP32，2=BF16。 |
| targetMode | 0=INT32，1=INT64，2=UINT8。 |
| rankMode | 0=1D，1=2D，2=4D。 |
| reductionMode | 0=none，1=mean，2=sum。 |
| scheduleMode | 0=scalar，1=parallel，2=multi-reduce。 |

hasWeight 和 cacheWeight 作为 TilingData 字段在 Kernel 初始化阶段读取，避免为可选输入额外扩大编译组合。

### TilingData

NllLossTilingData 规划如下：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| totalPositions | uint64 | 逻辑位置数 M。 |
| classNum | uint64 | 类别数 C。 |
| dimN | uint64 | N。 |
| dimH | uint64 | H，非 4D 时为 1。 |
| dimW | uint64 | W，非 4D 时为 1。 |
| hwSize | uint64 | H×W。 |
| chwSize | uint64 | C×H×W。 |
| ignoreIndex | int64 | 忽略索引。 |
| reduction | uint32 | 0=none，1=mean，2=sum。 |
| rankMode | uint32 | 1D、2D、4D。 |
| hasWeight | uint32 | 是否提供 weight。 |
| cacheWeight | uint32 | 是否将完整 weight 缓存到 UB。 |
| usedCoreNum | uint32 | 实际核数。 |
| positionsPerCore | uint64 | 每核最大位置数。 |
| tilePositions | uint32 | 每个 tile 的位置数。 |
| partialStride | uint32 | workspace 中每核部分和的对齐步长。 |

### Workspace

none 和单核归约不需要用户归约 workspace。多核 sum/mean 为每个核保存两个 float32 部分和：

~~~text
lossPartialBytes   = alignUp(usedCoreNum * sizeof(float), 32)
weightPartialBytes = alignUp(usedCoreNum * sizeof(float), 32)
userWorkspaceBytes = lossPartialBytes + weightPartialBytes
~~~

系统 workspace 由框架按平台接口追加。Kernel 使用 GetUserWorkspace 获取用户区，loss 和 weight 区域互不重叠。

## Kernel 设计

### 初始化

每个核计算自己的位置区间：

~~~text
positionStart = blockIdx * positionsPerCore
positionEnd   = min(positionStart + positionsPerCore, totalPositions)
corePositions = positionEnd - positionStart
~~~

若 positionStart 大于等于 totalPositions，该核直接退出。GlobalTensor 分别绑定 x、target、weight、y、total_weight 和 workspace。

### target 搬入

target 在位置维连续，按 tile 使用 DataCopyPad 搬入 UB：

- int32 直接读取。
- int64 保留 64 位比较和地址计算，避免大 ignore_index 截断。
- uint8 提升到 int32 后参与比较和偏移计算。
- 尾块补零只用于对齐，不参与有效长度计算。

### 选中值读取

每个有效位置先读取 t，再判断 t 是否等于 ignore_index：

1. ignore 位置直接生成 xValue=0、effectiveWeight=0，不访问 x 或 weight。
2. 合法位置根据 rankMode 计算 xOffset。
3. 使用单元素 DataCopyPad 或标量 GM 读取将选中元素放入 32B 槽位。
4. 通过 UB Gather 将每个槽位的第一个有效元素压紧。
5. FP16、BF16 转换为 float32。

2 维偏移：

~~~text
xOffset = globalPosition * C + target
~~~

4 维偏移：

~~~text
n  = globalPosition / (H * W)
hw = globalPosition % (H * W)
xOffset = n * (C * H * W) + target * (H * W) + hw
~~~

Kernel 不把非法 target clamp 到边界。合法输入约束由接口保证；ignore_index 在偏移计算前被屏蔽。

### 权重读取

~~~text
if ignored:
    effectiveWeight = 0
else if hasWeight == false:
    effectiveWeight = 1
else if cacheWeight == true:
    effectiveWeight = weightCache[target]
else:
    effectiveWeight = GM weight[target]
~~~

缓存模式使用 UB Gather；GM 模式采用与 x 相同的单元素间接读取方式。结果统一转换为 float32。

### 单位置计算

~~~text
loss = -selectedX * effectiveWeight
~~~

通过逻辑掩码 Select 将 ignore 位置的 loss 与 effectiveWeight 置 0。比较使用逻辑真假，不依赖二进制逐位比较结果。

### none 路径

none 模式直接将 loss 转换为 x dtype 并写入 y 对应位置：

~~~text
CopyIn target
IndirectLoad x / weight
Compute loss
Cast loss
CopyOut y
~~~

各核处理互不重叠的连续 target 区间，无原子操作和跨核同步。4 维输出按 target 的线性顺序写回，物理上自然对应 [N, H, W]。

### sum 路径

每个 tile 对 lossFloatBuf 执行 ReduceSum，并累加到本核 float32 标量 localLoss。有效权重同样累加到 localWeight，用于 total_weight：

~~~text
localLoss   += reduce_sum(tileLoss)
localWeight += reduce_sum(tileEffectiveWeight)
~~~

单核模式直接写回：

~~~text
y = localLoss
total_weight = localWeight
~~~

多核模式将 localLoss 和 localWeight 写入本核独占 workspace 槽位，执行 SyncAll 后由 core0 按 coreIdx 从小到大汇总并写回。

### mean 路径

mean 与 sum 共享局部归约和跨核归约流程。core0 得到 totalLoss 与 totalWeight 后执行：

~~~text
if totalWeight == 0:
    y = NaN
else:
    y = totalLoss / totalWeight

total_weight = totalWeight
~~~

不得把 totalWeight 为 0 的 mean 结果改写为 0。

### 1D 路径

x 为 [C]、target 为标量时仅启动一个核：

~~~text
if target == ignore_index:
    loss = 0
    totalWeight = 0
else:
    w = hasWeight ? weight[target] : 1
    loss = -w * x[target]
    totalWeight = w

none / sum: y = loss
mean:       y = totalWeight == 0 ? NaN : loss / totalWeight
~~~

当 w 非 0 时，mean 数学上等于 -x[target]，实现仍使用统一除法语义，以正确覆盖 w=0。

### 跨核归约

~~~mermaid
flowchart TD
    A["各核计算 localLoss / localWeight"] --> B["写入独立 workspace 槽位"]
    B --> C["SyncAll"]
    C --> D{"blockIdx == 0"}
    D -- "否" --> E["结束"]
    D -- "是" --> F["固定顺序读取所有部分和"]
    F --> G["ReduceSum loss / weight"]
    G --> H{"reduction"}
    H -- "sum" --> I["y = totalLoss"]
    H -- "mean" --> J["y = totalLoss / totalWeight 或 NaN"]
    I --> K["写回 total_weight"]
    J --> K
~~~

workspace 归约避免多个核直接原子累加同一输出，固定读取顺序也便于确定性验证。若后续 profiling 证明原子路径在特定大 shape 上明显更快，可作为独立 TilingKey 增加，但不能改变数值语义。

### 流水设计

none 路径采用 target 搬入、间接取值、向量计算、输出搬出的流水。sum/mean 路径省略逐位置输出队列，改为 tile 内归约：

~~~mermaid
flowchart LR
    A["MTE2: target"] --> B["Scalar/MTE2: selected x / weight"]
    B --> C["Vector: Cast / Select / Mul"]
    C --> D{"reduction"}
    D -- "none" --> E["MTE3: y"]
    D -- "sum/mean" --> F["Vector: ReduceSum"]
    F --> G["workspace / final output"]
~~~

target 与 none 输出可使用双缓冲。随机 x/weight 读取是否能与 Vector 计算有效重叠，需要根据 profiler 的 MTE2、Scalar 和 Vector 指标决定，避免仅增加 UB 占用而无收益。

## 性能优化

### 按需访存

每个位置只读取一个 x 元素和一个可选 weight 元素，不搬运完整类别行。C 较大时可显著降低无效 GM 流量。

### 权重缓存

小 C 场景将 weight 常驻 UB，避免每个位置重复访问 GM。大 C 场景按 target 读取，保证泛化能力。

### 特殊路径

- C=1 时 x 的选中值连续，可切换为连续 DataCopy 和纯向量路径。
- 无 weight 时跳过 weight GM 访问和缓存。
- 1D 和极小 M 使用单核标量路径，降低启动与同步开销。
- none 模式不申请用户归约 workspace，不执行 SyncAll。
- 4D 模式直接计算 NCHW 偏移，避免为 NLLLoss2d 额外转置整个 x。

### 负载均衡

按 target 位置均分，每个位置的基础工作量近似一致。前 usedCoreNum-1 个核处理 positionsPerCore 个位置，末核处理尾部，避免空核。

### 归约精度

FP16、BF16 的 loss 与 weight 均在 float32 中归约，最后一次转换回输出 dtype。跨核部分和也使用 float32，避免低精度原子累加。

### 性能判定

性能优化必须基于相同 dtype、shape、reduction、ignore_index、weight、预热次数和测量次数与 TBE 对比。重点观察：

- AIV Core 利用率和实际参与核数。
- 随机读取导致的 MTE2 或 Scalar stall。
- weight 缓存命中收益。
- none 模式的流水重叠。
- sum/mean 的 SyncAll 和 core0 最终归约开销。
- 小于 10 us 场景的固定启动开销。

## Ascend C 流程

### ACLNN 流程

~~~mermaid
flowchart TD
    A["GetWorkspaceSize"] --> B["空指针 / dtype / shape / reduction 检查"]
    B --> C["Contiguous 输入"]
    C --> D{"空 Tensor"}
    D -- "是" --> E["Fill 空输入语义"]
    D -- "否" --> F["调用内部 NLLLoss"]
    F --> G["Host Tiling"]
    G --> H["返回 workspaceSize / executor"]
    E --> H
    H --> I["aclnnNLLLoss"]
    I --> J["在 stream 上执行"]
    J --> K["Cast / ViewCopy 输出"]
~~~

### Host 流程

~~~mermaid
flowchart TD
    A["读取 x / target / weight / attrs"] --> B["校验 rank、shape、dtype"]
    B --> C["计算 M、C、N、H、W"]
    C --> D["选择 rank / dtype / target TilingKey"]
    D --> E{"reduction 与规模"}
    E -- "none" --> F["none_parallel 或 scalar"]
    E -- "sum/mean 小规模" --> G["reduce_single"]
    E -- "sum/mean 大规模" --> H["reduce_multi"]
    F --> I["计算 usedCore / tilePositions"]
    G --> I
    H --> I
    I --> J["判断 weight cache"]
    J --> K["规划 workspace"]
    K --> L["写 TilingData / SetBlockDim / SetTilingKey"]
~~~

### Kernel 流程

~~~mermaid
flowchart TD
    A["Init"] --> B["计算本核位置区间"]
    B --> C{"缓存 weight"}
    C -- "是" --> D["weight 搬入 UB 并转 float"]
    C -- "否" --> E["按需读取或使用常量 1"]
    D --> F["按 tile 搬入 target"]
    E --> F
    F --> G["判断 ignore_index"]
    G --> H["计算 xOffset"]
    H --> I["间接读取 x / weight"]
    I --> J["float32 loss 与有效权重"]
    J --> K{"reduction"}
    K -- "none" --> L["Cast / CopyOut y"]
    K -- "sum/mean" --> M["核内 ReduceSum"]
    M --> N{"多核"}
    N -- "否" --> O["直接写回"]
    N -- "是" --> P["workspace + SyncAll + core0 汇总"]
~~~

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| 香橙派 OrangePi AIpro | |
| Atlas 200I/500 A2 推理产品 | |
| Atlas A2 训练系列产品 / Atlas A3 系列产品 | √ |

## 约束限制

- x 支持 1D [C]、2D [N,C] 和 4D [N,C,H,W]。
- target 分别为标量、[N] 和 [N,H,W]。
- x、weight、y、total_weight 支持 FP16、FP32、BF16。
- target 支持 INT32、INT64、UINT8。
- weight 缺省或为 [C]，dtype 与 x 一致。
- reduction 支持 none、mean、sum。
- target 元素必须位于 [0,C) 或等于 ignore_index。
- 内部格式为 ND，不支持广播。
- ACLNN 支持非连续 Tensor，内部 Kernel 按连续地址处理。
- mean 的 total_weight 为 0 时输出 NaN。

# 可维可测分析

## 功能测试

| 类别 | 测试点 |
| --- | --- |
| rank | 1D、2D、4D。 |
| dtype | x/weight 为 FP16、FP32、BF16；target 为 INT32、INT64、UINT8。 |
| reduction | none、sum、mean。 |
| weight | 无 weight、全 1、随机正数、包含 0、包含负数。 |
| ignore_index | 默认 -100、合法类别内的值、所有位置忽略、部分位置忽略、无位置忽略。 |
| shape | 标量、单样本、C=1、小 C、大 C、32B 对齐、非 32B 对齐、尾块、满核、大 shape。 |
| 4D | H=1、W=1、N=1、不同 N/C/H/W 组合。 |
| 特殊值 | +0、-0、NaN、+Inf、-Inf、0×Inf。 |
| 空输入 | none、sum、mean。 |
| 内存 | 连续和非连续输入输出。 |
| 分支 | weight cache、weight GM、单核、多核、C=1 快速路径。 |

UINT8 target 无法表示默认 -100，相关用例应使用 [0,C) 内的 ignore_index 验证忽略语义。

## Golden 实现

~~~python
import numpy as np

def nll_loss_golden(x, target, weight=None, reduction="mean", ignore_index=-100):
    x32 = np.asarray(x, dtype=np.float32)
    target64 = np.asarray(target, dtype=np.int64)

    if x32.ndim == 1:
        rows = x32.reshape(1, x32.shape[0])
        target_flat = target64.reshape(1)
        out_shape = ()
    elif x32.ndim == 2:
        rows = x32
        target_flat = target64.reshape(-1)
        out_shape = target64.shape
    elif x32.ndim == 4:
        rows = np.transpose(x32, (0, 2, 3, 1)).reshape(-1, x32.shape[1])
        target_flat = target64.reshape(-1)
        out_shape = target64.shape
    else:
        raise ValueError("x rank must be 1, 2 or 4")

    class_num = rows.shape[1]
    weight32 = np.ones(class_num, np.float32) if weight is None else np.asarray(weight, np.float32)
    valid = target_flat != ignore_index
    if np.any((target_flat[valid] < 0) | (target_flat[valid] >= class_num)):
        raise IndexError("target out of range")

    loss = np.zeros(target_flat.shape, np.float32)
    effective_weight = np.zeros(target_flat.shape, np.float32)
    idx = np.nonzero(valid)[0]
    cls = target_flat[valid]
    effective_weight[valid] = weight32[cls]
    loss[valid] = -rows[idx, cls] * effective_weight[valid]

    total_weight = np.sum(effective_weight, dtype=np.float32)
    if reduction == "none":
        return loss.reshape(out_shape), total_weight
    if reduction == "sum":
        return np.sum(loss, dtype=np.float32), total_weight
    if reduction == "mean":
        total_loss = np.sum(loss, dtype=np.float32)
        return np.float32(np.nan) if total_weight == 0 else total_loss / total_weight, total_weight
    raise ValueError("invalid reduction")
~~~

最终自验证以 PyTorch aten.nll_loss_forward 或官方 ACLNN/TBE 结果为主，NumPy Golden 用于定位 shape、索引和归约问题。

## 精度标准

| 验收项 | 标准 | 来源 |
| --- | --- | --- |
| 数值精度 | 满足 CANN Judge 对应题目的默认阈值。 | 任务书 |
| ignore 语义 | ignore 位置损失和权重均为 0。 | 官方算子语义 |
| mean 语义 | 除以 total_weight，不除以位置数。 | 官方算子语义 |
| 零分母 | total_weight=0 时 mean 为 NaN。 | PyTorch/ACLNN 语义 |
| 特殊值 | NaN、Inf 的位置和符号符合逻辑值比较与 IEEE 计算结果。 | 数值语义 |

比较采用逻辑值和数值误差阈值，不以输出二进制逐位一致作为唯一标准。NaN 按同位置 NaN 等价，Inf 检查位置和符号。

## 性能标准

| 验收项 | 标准 | 来源 |
| --- | --- | --- |
| 满核场景 | 所有核参与计算时，Ascend C 性能不低于 TBE 的 95%。 | 任务书 |
| 小 shape | 10 us 以下场景若相差不超过 3 us，可结合性能仿真图说明固定开销。 | 任务书 |
| 测试一致性 | Ascend C 与 TBE 使用相同输入、属性、预热、测量次数和设备状态。 | 自验证规范 |

## 兼容性分析

本算子为 TBE 到 Ascend C 的等价迁移。内部算子名称、输入输出、属性默认值、ACLNN 两段式接口和 PyTorch NLLLoss 语义保持不变。新实现仅替换 A2/A3 的 AICore 计算路径，不改变调用方接口。

## 风险分析

| 风险 | 影响 | 应对方案 |
| --- | --- | --- |
| target 随机导致 x 访存离散 | MTE2 利用率低 | 仅搬运选中元素、槽位压紧、循环展开，并用 profiler 校准 tile。 |
| C 很大导致 weight 无法常驻 UB | 重复 GM 访问 | Host 选择 cacheWeight 或按需读取两种模式。 |
| 4D 地址计算错误 | 选错类别或空间位置 | 下发 hwSize、chwSize，专项覆盖 N/C/H/W 非对称 shape。 |
| ignore_index 先后顺序错误 | 越界访问 | 在任何 x/weight 地址计算前判断 ignore。 |
| mean 除数错误 | 精度或语义不一致 | loss 和 effective_weight 同时归约，零分母输出 NaN。 |
| FP16/BF16 长归约误差 | CANN Judge 不通过 | float32 局部与跨核累加，末端再转换。 |
| 多核归约非确定性 | 重复运行波动 | 独立 workspace 槽位和 core0 固定顺序汇总。 |
| 小 shape 同步开销 | 性能低于 TBE | reduce_single 和 scalar 路径避免 SyncAll。 |
| 尾块越界 | 数据破坏 | DataCopyPad、有效长度和独立尾块处理。 |

## 关联 Issue

暂无。

## 文档更新

本文档。

## 类型标签

- [ ] Bug 修复
- [ ] 新特性
- [ ] 性能优化
- [x] 文档更新
- [x] 其他：社区任务算子设计文档

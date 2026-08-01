# 需求背景（required）

## 需求来源

本设计对应 2026 年 7 月社区任务 `aclnnRoll`。任务要求基于已有
`aclnnRoll` 扩展 `complex64` 输入/输出支持，使 `torch.fft.fftshift` 和
`torch.fft.ifftshift` 在 complex64 频谱输入场景下可调用 NPU Roll 路径。

任务书规定：实现语言为 Ascend C，CANN 版本为 8.5.0 及以上；适配目标为
Atlas A2、A3、A5 训练系列产品。complex64 输出需与 CPU 对标，使用
AscendOpTest 默认阈值；原有 dtype 功能与性能不得回归。

## 背景介绍

### aclnnRoll 算子 complex64 扩展

`Roll` 沿指定维度循环移动 Tensor 元素。`torch.fft.fftshift` 和
`torch.fft.ifftshift` 通过该语义调整 FFT 结果的频谱顺序，而 `torch.complex64`
是该场景的常用输入类型。已有 `aclnnRoll` 未覆盖 complex64 时，框架侧会因
dtype 不支持而报错。

本任务的目标不是增加复数算术，而是让 Roll 对 complex64 完成与其他 dtype 相同
的坐标置换：一个 complex64 元素的实部和虚部必须一起移动，输出 shape 和 dtype
与输入保持一致。

### aclnnRoll 现有实现现状分析

已有实现的核心文件位于 `experimental/math/roll/`，分层职责如下：

| 层次 | 关键文件 | 现有职责 | complex64 扩展点 |
| --- | --- | --- | --- |
| ACLNN 接口 | `op_api/aclnn_roll.cpp` | 参数校验、连续化、executor 组装 | dtype 支持列表、非连续 view 数据流保持不变 |
| 算子定义 | `op_host/roll_def.cpp` | 输入输出 dtype、ND format、AICore 配置 | 输入/输出注册 `DT_COMPLEX64` |
| Host tiling | `op_host/roll_tiling.cpp` | shape 规整、shift 归一化、分核、UB 切分 | complex64 元素宽度按 8 byte 计算 |
| Kernel 入口 | `op_kernel/roll.cpp` | 读取 tiling 数据并启动统一模板 kernel | complex64 降低为 8-byte payload 搬运类型 |
| Kernel 主体 | `op_kernel/roll.h` | 坐标回绕、连续段/块搬运、尾块处理 | 复用已有置换路径，不引入复数运算 |

现有对外参数语义和本任务新增 dtype 如下：

| 参数 | 输入/输出 | 数据类型 | 格式 | shape | 非连续 Tensor |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入 | BF16、FP16、FP32、INT8、UINT8、INT32、UINT32、COMPLEX64 | ND | 0--8 维 | 支持 |
| `shifts` | 输入属性 | int64 列表 | - | - | - |
| `dims` | 输入属性 | int64 列表 | - | - | - |
| `out` | 输出 | 与 `x` 相同 | ND | 与 `x` 相同 | 按输出 view 语义写回 |
| `workspaceSize` | 输出 | uint64 | - | - | - |
| `executor` | 输出 | aclOpExecutor* | - | - | - |

`dims` 为空时 `shifts` 长度必须为 1；`dims` 非空时，二者长度必须一致，且
每个维度取值在 `[-rank, rank)`。输入输出 dtype、shape 与 ND format 必须一致，
rank 最大为 8。

# 需求分析（required）

## 需求描述

在不改变 `aclnnRoll` 函数原型和既有合法输入语义的前提下，完成 complex64 的
注册、tiling 和 Ascend C kernel 支持。complex64 输入下结果与 CPU
`torch.roll` 对齐，并保持现有 float、bfloat16、整型路径正常工作。

## 需求拆解

1. **接口与注册闭环**：在 ACLNN dtype 校验、OpDef 输入输出和算子 binary
   选择所依赖的 metadata 中登记 complex64，拒绝未注册 dtype。
2. **位宽闭环**：Host tiling 使用 complex64 的 8-byte 元素宽度，分核、GM
   对齐和 UB 容量不能按单个 float32 分量计算。
3. **语义闭环**：保留 `dims=[]` 展平、负维度、重复维度、正负及超大 shift 的
   归一化语义；输出仍是与输入同 shape、同 dtype 的循环置换。
4. **layout 闭环**：非连续输入先整理为连续逻辑 view；不能直接写入的输出在
   L0 Roll 后按 view 语义回写。
5. **验证闭环**：complex64 与 CPU 对标并采用 AscendOpTest 默认阈值，同时对
   原有 dtype 和泛化属性组合回归。

# 详细设计（required）

## 算子分析

### 数学公式

设输入的逻辑 shape 为

$$
\mathbf{S}=(S_0,S_1,\ldots,S_{r-1}),\quad N=\prod_{k=0}^{r-1}S_k.
$$

对属性中第 `j` 个维度和位移 `(d_j, s_j)`，先归一化维度和位移：

$$
a_j=((d_j\bmod r)+r)\bmod r,
$$

$$
\Delta_j=((s_j\bmod S_{a_j})+S_{a_j})\bmod S_{a_j}.
$$

同一维度出现多次时，等效位移为该维度所有归一化位移之和再取模：

$$
\widehat{\Delta}_a=\left(\sum_{j:a_j=a}\Delta_j\right)\bmod S_a.
$$

当 `dims` 为空时，输入先展平为长度 `N` 的一维逻辑数组：

$$
y_{flat}[t]=x_{flat}[(t-\Delta)\bmod N].
$$

当 `dims` 非空、输出坐标为 `\mathbf{o}` 时，对活跃维度采用
`(o_k-\widehat{\Delta}_k) mod S_k` 取源坐标，其他维度保持不变。因此 Roll
只做坐标回绕和数据搬运，不改变元素值。

### 支持数据类型

| dtype | 元素宽度 | 计算/搬运语义 |
| --- | --- | --- |
| uint8、int8 | 1 byte | 复用原始搬运路径 |
| bfloat16、float16 | 2 byte | 复用原始搬运路径 |
| float32、int32、uint32 | 4 byte | 复用原始搬运路径 |
| complex64 | 8 byte | 作为完整 payload 搬运，不做实虚拆分、cast 或算术 |

complex64 的语义桥接为：

```text
framework complex64 [real32 | imag32]
              -> uint64_t payload
              -> GM / UB / GM 搬运
              -> framework complex64 [real32 | imag32]
```

该桥接只定义内部搬运单元，不修改输出的 framework dtype。由于不存在数值计算，
有限值、零、NaN 和 Inf 都随元素原始 payload 置换；精度验收仍统一使用任务书
规定的 CPU 对标和 AscendOpTest 默认阈值。

### 支持形状

支持 ND、rank 0--8。`dims=[]` 使用一维展平语义；`dims` 非空时支持多轴、
负维度和重复维度。Roll 不涉及广播，输入输出的元素总数、逻辑 shape 和 dtype
始终一致。

## 算子实现

### 实现方案

整体执行流程如下：

```text
aclnnRollGetWorkspaceSize
  -> 参数/format/dtype/shape 校验
  -> 非连续 x: Contiguous(x)
  -> L0 Roll: Host tiling + AIV kernel
  -> 非直接输出: ViewCopy(rollResult, out)
  -> 返回 workspaceSize 和 executor

aclnnRoll
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

该流程只在本算子自有的 Ascend C 路径中选择数据整理和 kernel 分支，不调用
CPU、PyTorch、reference、vendor whole-op 或其他后端作为运行时 fallback。

#### 3.2.1 Host 侧设计

**参数检查。** Host 先检查 `x/shifts/out` 非空、dtype 在支持列表、x/out dtype
相同、ND format、shape 相同、rank 不超过 8，以及 `shifts/dims` 的长度和范围。
空 Tensor 直接返回零 workspace；rank 0 只允许空 `dims` 和一个 `shift`。

**连续性处理。** 输入 view 不是致密布局时通过 `Contiguous` 生成本次 executor
拥有的连续输入；输出不能直接写入时，先生成连续 Roll 结果，再通过 `ViewCopy`
按目标 view 写回。kernel 不承担非连续地址推导，从而让输入、临时结果和输出的
所有权明确。

**shape 与 shift 规整。** tiling 读取逻辑 shape，初始化每维 `shape`、`stride`
和 `shift`。`dims=[]` 改写为 `shape=[totalNum]`、`stride=[1]`；非空 dims 将
负维度转为非负维度并累加重复维度的 shift。随后生成：

```text
totalNum、shapes[rank]、strides[rank]、shifts[rank]
activeDimCount、activeDim、outerSize、dimSize、innerSize、activeShift
```

**分核策略。** 从平台读取 AIV 核数，将输出元素区间均分给各核：

```text
rawPerCore      = ceil(totalNum / coreNum)
perCoreElements = align_up(rawPerCore, layout_alignment)
blockDim        = ceil(totalNum / perCoreElements)
lastCoreElements = totalNum - (blockDim - 1) * perCoreElements
```

对小数据优先单核，减少启动和跨核分段开销；大数据按不重叠输出区间分核。对最后
活跃维、完整行或完整块优先采用连续段对齐，降低零散 GM 访问。

**数据分块和 UB 优化。** 设元素宽度为 `typeSize`：

```text
elementsPerGM = max(GM_BLOCK_BYTES / typeSize, 1)
ubElements    = max(UB_BYTES / typeSize, 1)
```

每次 Copy 的元素数不超过 `ubElements`，并按 GM 对齐、行/块边界和末核尾段
切分。complex64 的 `typeSize=8`，因此其 UB 容量和对齐均按完整复数元素计算。

**tiling key 规划。** 本实现使用统一 Roll 模板 tiling key；不同搬运路径由
tilingData 中的活跃维度、`innerSize`、`dimSize`、位移和连续布局等运行时元数据
决定。它们均来自真实输入 metadata，不依赖 case id、公开 shape 或输入值分布。

#### 3.2.2 Kernel 侧设计

kernel 入口读取 `RollTilingData`，初始化 GM 输入/输出指针以及输入、输出 UB 队列。
每个 AIV block 根据 `GetBlockIdx()` 得到自己的 `startIndex` 和 `elementCount`；
当 block 超出 `usedCoreNum`、区间为空或超出总元素数时不执行搬运。

`Process` 根据 Host 提供的元数据选择以下自有路径：

| 运行时条件 | 搬运路径 | 设计目的 |
| --- | --- | --- |
| 所有归一化 shift 为 0 | Identity copy | 避免不必要的坐标计算 |
| 一维/展平 Roll | 两段连续源区间复制 | 正确处理环绕点 |
| 单一首维 Roll | source run 搬运 | 复用内层连续块 |
| 单一末维 Roll | 按行两段搬运 | 复用行内连续性 |
| 单一中间维 Roll | 以 `dimSize * innerSize` 为块搬运 | 保持内层连续区间 |
| 多活跃维 Roll | 以最后活跃维进行行/块搬运 | 减少逐元素寻址 |

所有路径将环绕边界拆分成合法 Copy segment，末段按剩余元素数处理；输入和输出
不会因分核产生交叠写入。complex64 仅将模板搬运类型替换为 `uint64_t`，完整复用
上述坐标、分段和尾块逻辑。

## 支持硬件

| 支持的芯片版本 | 任务目标 | 本设计说明 |
| --- | --- | --- |
| Atlas A2 训练系列产品 | √ | 按 Ascend C AIV/GM/UB 数据搬运模型设计 |
| Atlas A3 训练系列产品 | √ | 使用相同 ACLNN、tiling 和 kernel 数据流，需按目标包配置构建回归 |
| Atlas A5 训练系列产品 | √ | 使用相同 ACLNN、tiling 和 kernel 数据流，需按目标包配置构建回归 |

每个目标 SoC 的 OpDef、tiling、kernel 编译单元和 package 配置必须同步登记；
单一 SoC 的构建结果不能替代其他 SoC 的实际回归。

## 算子约束限制

1. 仅支持 ND，rank 为 0--8；输入输出 shape 和 dtype 必须一致。
2. `dims=[]` 时 `shifts` 长度必须为 1；`dims` 非空时二者长度相等，dims 取值
   在 `[-rank, rank)`。
3. 本任务新增 complex64，不扩展 complex128，也不增加复数算术语义。
4. 不支持的 dtype、format、shape 或属性组合在 ACLNN 入口明确返回参数错误，
   不存在静默 fallback。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| complex64 精度 | 与 CPU `torch.roll` 对标，采用 AscendOpTest 默认阈值；比较复数实部和虚部 | 任务书 |
| 原 dtype 回归 | BF16、FP16、FP32、INT8、UINT8、INT32、UINT32 结果和既有功能保持一致 | 任务书 |
| 泛化能力 | 覆盖合法 rank 0--8、空 dims、多轴、负/重复 dims、正负/超大 shifts、非连续输入 | 任务书 |
| 性能要求 | 任务书未设置独立量化门槛；设计上不增加 dtype 分量转换和 whole-op fallback | 任务书 |

自测矩阵至少包含以下类别：

| 类别 | dtype | 形状/属性 | 验证点 |
| --- | --- | --- | --- |
| 基础复数多轴 | complex64 | `(2, 3)`，`shifts=[1,-1]`，`dims=[0,1]` | 多轴坐标映射和实虚配对 |
| 展平 | complex64 | 多维 shape，`dims=[]` | 一维逻辑展平语义 |
| 属性归一化 | complex64 | 负 dims、重复 dims、超大正负 shifts | 正模和合并语义 |
| 边界 | complex64 | 0-D、空 Tensor、维度长度为 1 | 空路径、identity 与尾块 |
| 非连续 | complex64 与既有 dtype | transpose/slice view | Contiguous 与输出回写 |
| dtype 回归 | 所有已支持 dtype | 一维、末维、中间维、多维 | 既有路径不回归 |
| 非法参数 | 支持/不支持 dtype | rank>8、shape/dtype 不一致、dims 越界 | 明确失败、无 fallback |

每个用例记录 CPU/NPU 输出、dtype、shape、`shifts/dims`、连续性和默认阈值比较
结果。complex64 检查以单个复数元素为单位，禁止把实部和虚部拆成两个独立 Roll
结果再拼接。

## 兼容性分析

本次变更不修改 `aclnnRoll` 函数原型、属性格式、输出 shape 规则或既有 dtype 的
公开语义，只增加一个合法 dtype。已有调用保持原有 Host/tiling/kernel 路径；
complex64 调用共享同一参数校验、shape 规整、非连续处理和运行时元数据分发，
仅在内部搬运类型上采用完整 8-byte payload bridge。

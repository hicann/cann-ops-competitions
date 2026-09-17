# MaxPool2dWithMaskBackward 算子设计方案

# 需求背景（required）

## 需求来源

2026 年 9 月 CANN 社区任务 09-47：MaxPool2dWithMask Backward 算子开发。贡献者：Miomugi。

本文为设计及验收计划，描述拟实现的功能、技术方案和验证目标。

## 背景介绍

最大池化反向传播根据正向记录的最大值索引，将上游梯度累加到输入对应位置。窗口重叠时，同一输入位置可能接收多个梯度，因此需要兼顾累加精度、并行写入安全及确定性。拟使用 Ascend C 实现该算子，适配 A2 和 310P，并按任务要求进行 YOLOv11 / DOTAv1 模型验证。

# 需求分析（required）

## 需求描述

拟提供 `aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize` 和 `aclnnMaxPool2dWithMaskBackward` 两阶段调用能力，保持既有接口参数及 mask 语义。开发时将结合目标仓现有接口组织实现，避免重复注册公共接口。

| 参数 | 含义及计划支持范围 |
| --- | --- |
| gradOutput | 上游梯度，shape 与正向 out 一致，支持非连续视图 |
| self | 正向输入，用于确定输入形状、dtype 和格式 |
| indices | INT8 索引容器，按每通道独立的 INT32 argmax 解码，支持非连续输入 |
| kernelSize | 长度 1 或 2，元素为正整数 |
| stride | 长度 0、1 或 2；为空时采用 kernelSize |
| padding | 长度 1 或 2；按反向任务要求，各方向满足 0 ≤ padding ≤ kernelSize |
| dilation | 长度 1 或 2，值为 1 |
| ceilMode | 与正向使用的输出形状计算方式一致 |
| gradInput | 输入梯度，shape、dtype、格式与 self 一致，支持非连续输出 |

## 需求拆解

1. 校验属性、形状和 dtype，归一化参数并处理非连续张量。
2. 正确解码 mask，对同一输入位置的多个上游梯度进行累加，未命中位置输出零。
3. 支持 `aclrtCtxSetSysParamOpt` 确定性设置；拟优先采用确定性累加方案，同时满足默认模式语义。
4. 基于通用几何关系、UB 容量和核数设计可泛化的分核与分块方案。
5. 完成官方全量精度及逐点性能验证、独立泛化测试，以及 310P 对照 A2 的模型验证计划。
6. 准备可复现源码、测试程序、报告、真实运行截图和待验收仓库。

# 详细设计（required）

## 算子分析

### 数学公式

设 `q = oh×Wo+ow`，从 indices 解码得到通道内索引 `a[n,c,q]`，则：

`gradInput[n,c,i] = Σ gradOutput[n,c,q]`，求和范围为所有满足 `a[n,c,q] = i` 的输出位置 q。

没有匹配索引的位置结果为零。索引为输入通道内的 `ih×W+iw`，不包含 batch/channel 偏移。

在 dilation = 1 时，正向输出高度初值为 `Ho = round((H + 2Ph - Kh) / Sh) + 1`，宽度同理；round 由 ceilMode 选择 floor 或 ceil。ceilMode 为真且 `(Ho - 1)×Sh ≥ H+Ph` 时，Ho 减 1；宽度执行相同修正。gradOutput 必须符合推导形状。

NCHW 形式的 indices 形状为 `[N,C,Kh×Kw,(ceil(Ho×Wo/16)+1)×32]`。按照随任务 golden 的约定，整个 INT8 容器前 `4×N×C×Ho×Wo` 个连续字节为小端 INT32 索引流，尾部为 padding。不能按 mask 外观形状误解为逐窗口 bit mask，也不能按每通道容器跨度跳读有效索引。

计划以 FP32 累加，按输出位置 q 的行优先次序处理贡献值，在完成一个输入位置的累加后转换为输出 dtype，避免多次低精度舍入。该顺序将与随包 golden 对照验证。

### 支持数据类型

| 设备 | gradOutput / self / gradInput | indices |
| --- | --- | --- |
| Atlas 800T A2 | FLOAT32、FLOAT16、BFLOAT16 | INT8 |
| Atlas 300V Pro（310P） | FLOAT32、FLOAT16 | INT8 |

任务文字中的训练系列 FLOAT 转 FP16 描述与随包 golden 的 FP32 累加行为存在差异，拟以随包 golden 作为自测参考，并在评审时明确兼容性边界。

### 支持形状

以任务描述的四维 NCHW 为主要接口形状，并对 API/golden 支持的 CHW 及 ND 形式进行兼容验证。CHW 内部按 N=1 处理，输出保留原 rank。gradOutput、indices 和 gradInput 的非连续 stride、storage offset 拟在接口适配层通过连续化和视图拷贝处理。

形状和地址计算拟采用经过溢出检查的 64 位整数，校验 mask 容量、INT32 索引可表示范围和合法输出尺寸。窗口面积很小等容量边界将单独对照 API、shape 公式和 golden；遇到规格歧义将在评审中澄清，不修改官方用例来绕开问题。

## 算子实现

### 实现方案

#### host 侧设计

1. 归一化属性，校验 kernel/stride/padding/dilation 约束、dtype 一致性及所有输入输出形状。反向 padding 采用本任务允许的范围，不直接套用正向较窄的参数限制。
2. 对非连续输入执行连续化，规划输出视图回写，计算必要工作空间及整个 API 的设备执行流程。
3. 查询可用核数、UB 容量和数据搬运能力，规划梯度、索引、FP32 累加区和临时向量缓冲；由资源预算推导 tile 大小。
4. 优先将互不重叠的输入空间区间分配给不同核心；每个核心独占所负责 gradInput 区域，减少核间写冲突。小规模输入按工作量减少启用核数。
5. 根据窗口重叠程度、输入输出空间大小及 dtype 选择通用路径。分支不得依赖 case 编号或硬编码测试 shape；tiling 传递形状、属性、区间和缓冲大小等必要信息。

#### kernel 侧设计

1. 拟以“输入位置归属核心”的 gather-reduce 方案作为通用确定性路径。对输入位置 (ih,iw)，根据窗口起点关系计算可能覆盖它的输出范围，并裁剪至 `[0,Ho)`、`[0,Wo)`。
2. 对候选输出按行优先顺序读取 argmax 和 gradOutput，匹配 `ih×W+iw` 时累加到 FP32 缓冲；不匹配则忽略。完成后一次转换并写出，未命中位置写零。
3. 对能够放入 UB 的通道或 tile，评估顺序读取索引/梯度、局部累加及批量写出的实现，以减少重复 GM 读取；跨 tile 的梯度贡献须完整保留。
4. 对窗口不重叠的通用几何场景评估无需多次归约的快速路径；仅在能够证明写入互斥时采用直接写入策略，零初始化仍覆盖整个输出。
5. 默认模式可复用确定性路径；开启确定性时避免依赖核间浮点原子加的执行顺序。任何后续并行优化均需重新检验重复运行一致性和累加精度。
6. 所有地址计算以整数语义为准，搬运前检查边界并处理尾块；越界索引不得形成非法内存访问。合法输入由兼容正向生成，非法索引作为健壮性测试单独验证。

## 支持硬件

计划支持 Atlas 800T A2 和 Atlas 300V Pro（310P），使用 CANN 9.0.0 或 9.1.0。分别构建并记录设备、驱动、工具链和运行依赖，不假设不同芯片或 CANN 版本的二进制直接兼容。

## 算子约束限制

- dilation 仅支持 1；padding 按反向任务要求满足 0 ≤ padding ≤ kernelSize。
- 输入不支持 NaN、-Inf；不进行广播，gradOutput 和 indices 应与正向几何和 mask 协议一致。
- BFLOAT16 仅在 A2 上作为支持类型。
- 默认接口为非确定性实现要求，允许提供更强的确定性行为；开启确定性设置时必须保证重复结果一致。
- 未定义参数组合和规格冲突将形成评审问题；不得通过默默缩窄合法范围或修改官方 golden 规避验收。

# 可维可测分析

## 精度标准/性能标准

| 验收项目 | 计划判据 | 标准来源 |
| --- | --- | --- |
| gradInput 精度 | 与官方 golden 对比，匹配比例和最大误差须同时达标 | 任务书及生态算子开源精度标准 |
| 确定性 | 开启确定性后，同输入多次运行结果逐元素一致；默认模式同步记录 | 任务接口要求 |
| A2 性能 | 每个官方点满足 `T_candidate ≤ T_TBE / 0.95` | 任务要求性能不低于 TBE 的 95% |
| 310P 性能 | 每个支持点以同一最终实现对应 A2 耗时 5 倍为优化目标，逐点报告 | 任务书 310P 耗时基准 |
| 仿真分析 | 小 shape 在 100 μs 以下且超过 TBE 耗时 30%，或 310P 超过 5 倍基准 30% 时提供仿真图及分析；仍以所有点达到上述目标作为自验收目标 | 任务书性能条款 |
| 模型精度 | YOLOv11 / DOTAv1，310P 与 A2 的 mAP50 绝对误差不超过 0.01 | 任务模型验证要求 |

拟采用 `abs(actual-golden) ≤ atol + rtol×abs(golden)`，匹配比例不低于 99%，并同时满足最大误差硬上限：

| dtype | rtol | atol | 最大绝对误差上限（或 32 ULP） |
| --- | --- | --- | --- |
| FLOAT16 | 2^-9 | 2^-9 | 0.1 |
| BFLOAT16 | 2^-6 | 2^-6 | 1.0 |
| FLOAT32 | 2^-10 | 2^-16 | 0.01 |

### 算子测试与证据计划

1. 保留官方 JSON 和 golden 原件。A2 覆盖全部 140 个官方点，310P 覆盖支持 dtype 的 103 个点，37 个 BF16 点标明硬件不适用。每个支持点同时进行精度与性能验证，逐点记录基线、候选耗时、门槛及判定。
2. 补充重叠窗口、高重复索引、正负梯度抵消、非方形窗口、空 stride、ceilMode、padding 边界、非对齐尾块、大索引、CHW、非连续输入输出及确定性设置测试。固定随机种子，均匀分布与正态分布各占一半，遵循 NaN/-Inf 限制。
3. 独立验证 mask 解码和正反向联动，使用参考程序产生合法索引与梯度。构造多窗口集中命中同一输入位置的压力样例，检查累加顺序和低精度输出舍入。
4. 在相同环境采集真实 TBE 与候选实现 profile。拟预热 5 次、采集 64 次有效样本，以全部有效样本均值作为主判据；辅助去头尾统计不得替代主判据。统计一次 API 必需的设备计算、清零、转换和拷贝，排除外部输入准备及 CPU golden。
5. 固定最终源码和动态库版本后重新执行全量测试，并验证被计时调用输出。保留命令、环境、源码/库哈希、原始日志、逐次样本与解析脚本。
6. 按官方模板制作自测报告，包含参数、精度、性能、逐点结论和真实服务器运行截图；开发记录及实测结论将在后续报告中单独提供。

### 模型接入计划

计划在 YOLOv11 中为对应池化模块接入兼容的正向 mask 与本反向算子，并通过自动求导调用链验证 gradInput 及参数梯度，记录实际算子调用证据，防止模型评估只运行正向而遗漏反向验证。

两设备拟采用相同的模型版本、权重、DOTAv1 数据版本和划分、预处理、图像尺寸、阈值及评估配置，固定随机种子。分别保留预测结果、完整评估日志和 mAP50，计算两设备 mAP50 绝对差并与 0.01 门槛比较。训练/反向检查和模型评估的执行范围、样本数及环境将清楚列入复现说明，不以少量样例代替任务规定的数据评估。后续按任务要求向 modelzoo-GPL 对应目录贡献接入方案。

## 兼容性分析

接口、输出 dtype/shape、正向 mask 布局和确定性配置拟保持兼容。适配层承担非连续张量转换，内部路径切换不得改变梯度累加语义。任务文字、API 与 golden 的差异将单独记录和澄清。后续代码拟按 ops-nn 规范组织到 `experimental/pooling`，经任务验收后申请合入。

## 实施与交付计划

按“设计评审 → 接口及通用梯度实现 → 泛化精度和确定性验证 → 全量逐点性能优化 → 模型验证 → 自验收与社区提交”推进。拟交付设计文档、算子/API README、源码及构建脚本、官方及泛化测试程序、模型接入与复现说明、自测报告及原始证据、待验收仓库地址和分支。待验收仓将按任务要求为 Ascend-CANN 配置开发者访问权限。

## 参考资料

- [社区任务及目录规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
- [算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [aclnnMaxPool2dWithMaskBackward 接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnMaxPool2dWithMaskBackward.md)
- [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
- [模型贡献目录](https://gitcode.com/Ascend/modelzoo-GPL/tree/master/contrib/PyTorch/Research/cv/image_object_detection)
- 任务附件：MaxPool2dWithMaskBackward 任务书、官方 JSON 及 golden。

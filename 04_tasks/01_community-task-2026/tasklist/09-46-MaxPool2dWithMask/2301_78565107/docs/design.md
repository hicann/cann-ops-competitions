# MaxPool2dWithMask 算子设计方案

# 需求背景（required）

## 需求来源

2026 年 9 月 CANN 社区任务 09-46：MaxPool2dWithMask 算子开发。贡献者：KousakaReina（GitCode：2301_78565107）。

本文为设计及验收计划，描述拟实现的功能、技术方案和验证目标。

## 背景介绍

二维最大池化用于空间特征下采样。除池化值外，本任务还要求输出供反向传播使用的最大值索引容器。拟使用 Ascend C 实现与任务提供的接口、用例及 golden 一致的算子，适配 A2 和 310P，并优化不同窗口、步长和输入规模下的访存与计算效率。

# 需求分析（required）

## 需求描述

拟提供 `aclnnMaxPool2dWithMaskGetWorkspaceSize` 和 `aclnnMaxPool2dWithMask` 两阶段调用能力，保持既有参数语义和输出约定。开发时将结合目标算子仓现有接口组织代码，避免重复注册公共接口。

| 参数 | 含义及计划支持范围 |
| --- | --- |
| self | 输入特征张量；支持任务要求的数据类型、格式及非连续视图 |
| kernelSize | 长度 1 或 2，元素为正整数 |
| stride | 长度 0、1 或 2；为空时采用 kernelSize |
| padding | 长度 1 或 2；各方向满足 0 ≤ padding ≤ floor(kernelSize / 2) |
| dilation | 长度 1 或 2，值为 1 |
| ceilMode | 控制输出尺寸向下或向上取整 |
| out | 最大池化值，dtype 与 self 一致，支持非连续输出 |
| indices | INT8 容器，保存每通道独立的 INT32 argmax 字节流 |

## 需求拆解

1. 实现属性归一化、参数校验、输出形状推导、工作空间计算及算子调用。
2. 实现最大值选择、相等值取首个索引、padding 边界处理和完整 mask 写出。
3. 根据硬件资源和通用形状特征选择分核、分块策略；覆盖合法输入的泛化场景。
4. 按官方完整用例逐条验证精度、采集 TBE 基线并评估性能；补充独立泛化和异常输入测试。
5. 准备可复现测试代码、自验收报告、真实运行截图及待验收代码仓。

# 详细设计（required）

## 算子分析

### 数学公式

记输入空间尺寸为 H、W，窗口为 Kh、Kw，步长为 Sh、Sw，padding 为 Ph、Pw。对于输出位置 (n,c,oh,ow)，窗口中的有效输入坐标为：

`ih = oh × Sh - Ph + kh`，`iw = ow × Sw - Pw + kw`。

`out[n,c,oh,ow] = max(self[n,c,ih,iw])`，其中 0 ≤ kh < Kh、0 ≤ kw < Kw，且坐标位于输入范围内。越界位置按负无穷处理。窗口按行优先顺序扫描，仅在新值严格大于当前最大值时更新索引，从而保证相等最大值取首个位置。

在 dilation = 1 时，输出高度初值为 `Ho = round((H + 2Ph - Kh) / Sh) + 1`，宽度同理；round 由 ceilMode 选择 floor 或 ceil。ceilMode 为真且 `(Ho - 1) × Sh ≥ H + Ph` 时，Ho 减 1；宽度执行相同修正，避免末窗起点完全落在右侧 padding 中。

mask 的末维为 `M = (ceil(Ho × Wo / 16) + 1) × 32`。NCHW 输入对应 indices 形状 `[N,C,Kh×Kw,M]`。按照随任务 golden 的约定，整个 INT8 容器的前 `4×N×C×Ho×Wo` 个连续字节保存小端 INT32 索引 `ih×W+iw`，其余字节清零；索引不包含批次或通道偏移。

### 支持数据类型

| 设备 | self / out | indices |
| --- | --- | --- |
| Atlas 800T A2 | FLOAT32、FLOAT16、BFLOAT16 | INT8 |
| Atlas 300V Pro（310P） | FLOAT32、FLOAT16 | INT8 |

计划将 FP16/BF16 值无损提升到 FP32 进行比较，输出仍使用原 dtype；FP32 保留原生比较精度。任务文字中训练系列 FLOAT 转 FP16 的说明与随包 golden 的原生 FP32 行为存在差异，拟以随包 golden 作为自测参考，并在评审时明确该兼容性边界。

### 支持形状

以四维 NCHW 为主要接口形状，支持任务 API/golden 所覆盖的三维 CHW 形式；CHW 内部按 N=1 处理，返回时保留原 rank。格式覆盖任务要求的 NCHW、ND。连续化和输出视图拷贝拟在接口层完成，以支持非连续 stride 和 storage offset。

输出尺寸、元素数量、mask 容量及偏移计算拟使用经过溢出检查的 64 位整数。INT32 索引须能够表示通道内空间位置；容器容量须覆盖完整有效字节流。对于窗口面积很小等容量边界，将单独验证 API、shape 公式与 golden 的一致性，规格歧义将在评审中澄清，不通过修改官方用例规避。

## 算子实现

### 实现方案

#### host 侧设计

1. 归一化单值/双值属性，校验 dtype、rank、格式、取值范围和输出描述符；拒绝非法尺寸及不支持的 dilation。
2. 推导输出与 mask 形状，校验容量及地址计算范围；针对非连续输入输出组织连续化与视图回写。
3. 查询 A2/310P 的可用核数、UB 容量和指令能力，综合输入缓存、最大值缓存、索引缓存及流水临时空间计算 tile 大小。
4. 以通道和连续输出区间分配工作，按工作量限制启用核数，处理大小核与尾块，避免小输入过度分核。
5. tiling 信息计划包含 H/W、Ho/Wo、窗口/步长/padding、总通道数、tile 大小和核内区间。分支选择只依赖这些通用属性及硬件能力，不依赖官方 case 编号或固定 shape 白名单。

#### kernel 侧设计

1. 采用 CopyIn、Compute、CopyOut 流程。每个核心负责互不重叠的输出区间，维护 FP32 最大值和整数索引。
2. 对连续输出批量计算窗口坐标，使用有效位置掩码过滤越界访问，按固定窗口顺序执行 compare/select。
3. 对规则连续访存场景尝试向量化、输入行复用及流水重叠；对大窗口、边界和非对齐 tile 保留通用路径。涉及坐标向量化时校验数值精确范围，超出范围使用整数计算路径。
4. 分别写回池化值和 INT32 索引字节流，处理尾部对齐与 mask 剩余区域清零；保证核间写入区间无重叠，无越界搬运。
5. 以真实 profile 识别访存、索引计算和启动开销，优化 tile 与流水参数；每次优化后重新执行精度及全量性能验证。

## 支持硬件

计划支持 Atlas 800T A2 和 Atlas 300V Pro（310P），使用 CANN 9.0.0 或 9.1.0。分别构建、记录工具链和运行环境，不假设不同 CANN 版本或芯片的二进制可直接互换。

## 算子约束限制

- 默认确定性计算；重复调用的池化值和索引应保持一致。
- 输入不支持 NaN、-Inf；dilation 仅支持 1。
- padding、输出尺寸及推理系列 ceilMode 的适用范围遵循任务/API 约束，相关边界将列入参数测试。
- 不进行广播；out 和 indices 必须符合推导形状。BFLOAT16 不作为 310P 支持类型。
- 未在任务要求中定义的输入行为不作为本方案的新增承诺；规格冲突将记录并提交评审确认。

# 可维可测分析

## 精度标准/性能标准

| 验收项目 | 计划判据 | 标准来源 |
| --- | --- | --- |
| out 精度 | 按生态标准检验 matched ratio、最大绝对误差及 ULP；另记录逐元素一致性辅助定位 | 任务书及生态算子开源精度标准 |
| indices 精度 | INT8 容器逐字节一致，覆盖有效 INT32 索引与零填充区 | 随任务 golden |
| A2 性能 | 每个官方点满足 `T_candidate ≤ T_TBE / 0.95` | 任务要求性能不低于 TBE 的 95% |
| 310P 性能 | 每个支持点以同一最终实现的对应 A2 时间 5 倍为优化目标，逐点报告比值 | 任务书 310P 耗时基准 |
| 仿真分析 | 小 shape 在 100 μs 以下且超过 TBE 耗时 30%，或 310P 超过 5 倍基准 30% 时，按任务要求提供仿真图及分析；仍以所有点达到上述目标作为自验收目标 | 任务书性能条款 |

浮点判定拟采用 `abs(actual-golden) ≤ atol + rtol×abs(golden)`，匹配比例不低于 99%，并同时满足最大误差硬上限：

| dtype | rtol | atol | 最大绝对误差上限（或 32 ULP） |
| --- | --- | --- | --- |
| FLOAT16 | 2^-9 | 2^-9 | 0.1 |
| BFLOAT16 | 2^-6 | 2^-6 | 1.0 |
| FLOAT32 | 2^-10 | 2^-16 | 0.01 |

### 测试与证据计划

1. 保留官方 JSON 和 golden 原件，按原参数逐例执行。A2 覆盖全部 153 个官方点；310P 覆盖支持 dtype 的 103 个点，其余 50 个 BF16 点明确标为硬件不适用。所有支持点同时进行精度和性能验证，不以汇总平均或部分样例替代逐点判定。
2. 补充不同 batch/channel、非方形窗口、空 stride、ceil/floor、padding 边界、非对齐和尾块、非连续视图、重复最大值及大索引等泛化测试。采用固定随机种子，均匀分布与正态分布各占一半，服从 NaN/-Inf 限制；追加非法参数和重复运行确定性检查。
3. 同一设备环境中采集真实 TBE 与候选实现 profile。拟先预热 5 次，再采集 64 次有效样本，以全部有效样本均值作为主判据；若提供去头尾统计，仅作辅助。计入一次 API 必需的全部设备计算、清零、转换和拷贝，不混入外部输入准备或 CPU golden 时间。
4. 固定最终源码和动态库版本后重新完成全量精度及性能测试，并验证被计时调用的输出。记录命令、环境、源码/库哈希、原始日志、逐次耗时及解析脚本。
5. 按官方自测报告模板整理参数、结果、截图和逐点判定；截图通过真实服务器运行界面采集，保留可追溯原始证据。开发、验证和验收结论将在后续自测报告中单独给出。

## 兼容性分析

接口参数、输出形状和 mask 语义拟与任务要求保持一致。通过适配层处理非连续张量，内部优化不得改变相等值索引选择和确定性行为。与既有 TBE 的数值语义差异、硬件限制和版本依赖将通过对照测试明确。后续代码拟按 ops-nn 的规范组织到 `experimental/pooling`，在验收通过后申请合入。

## 实施与交付计划

按“设计评审 → 接口及通用功能开发 → 泛化精度验证 → 全量逐点性能优化 → 自验收材料整理 → 社区验收与代码提交”的顺序推进。拟交付设计文档、接口与算子 README、源码和构建脚本、官方及泛化测试代码、自测报告及原始证据、待验收仓库地址和分支。待验收仓将按任务要求为 Ascend-CANN 配置开发者访问权限。

## 参考资料

- [社区任务及目录规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
- [算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [aclnnMaxPool2dWithMask 接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnMaxPool2dWithMask.md)
- [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
- 任务附件：MaxPool2dWithMask 任务书、官方 JSON 及 `max_pool2d_with_mask_golden_func.py`。

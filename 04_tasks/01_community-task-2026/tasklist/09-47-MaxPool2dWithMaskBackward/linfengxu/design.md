# MaxPool2dWithMaskBackward 算子设计

版本：1.0。日期：2026-09-23。

## 需求背景

本设计对应 9 月社区任务 09-47，在 Atlas 800T A2 和 Atlas 300V Pro 上使用 Ascend C 实现 MaxPool2dWithMask 的反向传播，适配 CANN 9.0.0 或 9.1.0。交付覆盖 ACLNN 接口、功能泛化、精度、性能以及 YOLOv11/DOTAv1 模型验证。本次选择 CANN 9.1.0 作为实现与验证环境。

配套前向为同批次社区任务开发的 **MaxPool2dWithMask Ascend C 实现**，前后向共同使用任务附件规定的 INT32 argmax 前缀。A2 和 310P 使用相同索引协议。设备自带 SDK 的其他 mask 表示不属于本设计的输入协议。

本文规定目标实现与验证方法；第“当前验证范围”节单独说明已有结果，设计定稿不表示完整实现或验收已通过。

任务来源：[MaxPool2dWithMaskBackward 算子开发](https://www.hiascend.com/activities/task-center/details/41ecbfa3d3cd4306adaf8236460aed08?menu=tasks)。

## 需求分析

### 接口定义

沿用两段式 ACLNN 接口：

```cpp
aclnnStatus aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices,
    const aclIntArray* kernelSize, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnMaxPool2dWithMaskBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

| 参数 | 定义与约束 |
| --- | --- |
| gradOutput | 4D 逻辑 NCHW，shape 为 `[N,C,Ho,Wo]`，支持非连续存储。 |
| self | 配套前向的输入，shape 为 `[N,C,Hi,Wi]`，提供输入形状和 dtype，支持非连续存储。 |
| indices | INT8 容器，shape 为 `[N,C,Kh*Kw,(ceil(Ho*Wo/16)+1)*32]`，支持非连续存储。 |
| kernelSize | 长度为 1 或 2，数值为正；长度为 1 时两个空间维度使用相同值。 |
| stride | 长度为 0、1 或 2；非空时数值为正，空数组表示与 kernelSize 相同。 |
| padding | 长度为 1 或 2，各维满足 `0 <= padding <= kernelSize`；还须满足下述有效索引约束。 |
| dilation | 长度为 1 或 2，各维均为 1。 |
| ceilMode | 控制输出尺寸的向上或向下取整，并应用尾部窗口校正。 |
| gradInput | 与 self 相同的 shape、逻辑格式和 dtype，支持非连续存储。 |

self、gradOutput、gradInput 的 dtype 相同。A2 支持 FP32、FP16、BF16，310P 支持 FP32、FP16；输入不支持 NaN/-Inf。公开逻辑布局为 NCHW，存储描述使用原型规定的 ND；通过 aclTensor 的逻辑 shape、strides 和 storage offset 解释非连续存储。

对单个空间维度，令 `A=I+2P-(K-1)-1`，floor 模式取 `O=floor(A/S)+1`，ceil 模式取 `O=ceil(A/S)+1`。ceil 模式下若 `(O-1)*S >= I+P`，将 O 减一。gradOutput 的空间尺寸须与该计算一致。

### 索引协议和边界

indices 开头的 `4*N*C*Ho*Wo` 个逻辑连续字节保存小端 INT32 argmax。第 `(n,c,oh,ow)` 项位于前缀下标 `(n*C+c)*Ho*Wo+oh*Wo+ow`，值为通道内 `ih*Wi+iw`。各 NC 平面的有效前缀连续排列，不按整个容器平面的跨度读取。配套前向将前缀之后的字节清零；backward 只读取有效前缀。

非连续 indices 先按逻辑顺序转连续再解释字节。每个索引必须位于输入平面内，且属于对应池化窗口的有效输入位置。self 的数值不参与重新求最大值；若窗口有多个相同最大值，反向遵从配套前向已经记录的选中位置。

K=1 不采用 SDK 的另一种 mask 路径，仍使用上述前缀。规定的容器必须能容纳全部索引；例如 self=`[1,2,8,8]`、kernel/stride=1、padding=0 时，容器为 320 字节而前缀需要 512 字节，接口返回参数错误，不越界写入或隐式改编码。附件 Golden 限定 K≥2，因此容量足够的 K=1 使用独立参考验证，不修改原始 Golden。

padding 保留任务书规定的上限。对超过半核但仍有有效输入的窗口，按有效索引执行反向；不存在任何有效输入位置的空窗口无法提供合法 argmax，返回参数错误，不增加附件未定义的哨兵索引。前向组合测试覆盖配套前向已支持的参数范围；反向更宽的有效窗口范围独立验证。

### 数学语义与精度

对每个 `(n,c,oh,ow)`，将 gradOutput 累加至 indices 指定的输入位置，其他 gradInput 元素为零。重叠窗口必须累加，不能覆盖写。

令 `p=n*C+c`、`q=oh*Wo+ow`、`t=ih*Wi+iw`，则：

`gradInput[p,t] = sum_q(gradOutput[p,q] if indices[p,q] == t)`。

按附件 Golden 使用 FP32 累加，q 按展平顺序递增，最后一次转换为输出 dtype。默认路径同样采用固定顺序；开启或关闭 `aclrtCtxSetSysParamOpt` 确定性选项均保持上述语义。

## 详细设计

### Host 与接口适配

第一阶段校验必选参数、dtype 一致性、4D shape、属性数组长度和值、输出尺寸、空窗口及索引前缀容量。形状、偏移和字节数采用 64 位计算，检查乘法溢出及 INT32 索引可表示范围。Device 侧索引数据由配套前向提供，Golden/负向测试检查其有效窗口约束；kernel 以索引比较选择梯度，避免把未验证的索引直接用于越界散射地址。

非连续输入通过 `Contiguous` 规范化。kernel 写入紧凑的中间输出，再通过 `ViewCopy` 按目标 strides 和 offset 写回 gradInput，保持存储间隙不变。中间张量和 workspace 由 executor 管理。

自定义包提供同名 ACLNN 接口，测试程序显式链接自定义库并记录符号来源。配套前向和本反向同时安装时，记录两个算子的版本及实际加载库，确保组合测试调用两项 Ascend C 实现。内置基线与自定义实现分别执行，使用官方 profiler 核实实际 kernel。

### Tiling 与 kernel

确定性基线按 gradInput 的空间块分配核心，每个输出位置由唯一核心写入。核心枚举覆盖该块的输出窗口，通过 INT32 索引筛选梯度，并按 q 递增顺序在 FP32 中累加，最后转换并写回。未被任何窗口选中的输入位置直接写零，不依赖预先清零的输出缓冲区。

对固定输入行 ih，需要遍历的输出行范围为：

`max(0, ceil((ih+Ph-Kh+1)/Sh)) <= oh < min(Ho, floor((ih+Ph)/Sh)+1)`。

列范围采用相同推导；整数除法明确使用数学 floor/ceil，避免负数截断改变窗口范围。只读取可能覆盖本块的 gradOutput 和索引，减少重复遍历。

Tiling 根据 NC、输入/输出空间大小、窗口重叠程度、dtype、平台 UB 容量与可用核心数划分块长和并行度。UB 预算包含梯度、索引、FP32 累加区、类型转换及搬运临时区；A2 与 310P 分别选择受支持的指令和同步方式。

无重叠窗口可使用无冲突散射，但仍须覆盖未选中位置的清零。重叠窗口优先采用独占输出块的归约，利用索引批处理和行数据复用优化；任何后续并行归约变化都重新验证 FP32 累加精度与确定性，不用低精度原子加替代既定数值语义。

尾块只处理有效元素，DMA 对齐与缓存维护按平台实现，避免越界访问及输出间隙污染。完整 ACLNN 调用中的规范化、计算和回写均计入相应功能与性能分析。

## 可维可测分析

### 功能、精度与组合验证

通过原始附件和 AscendOpTest 标准入口执行 A2 原始 140 条、310P 适用的 103 条；310P 的 37 条 BF16 标为不适用。测试同时覆盖非连续输入输出、矩形窗口、重叠/不重叠、默认 stride、ceilMode、K=1 容量边界、有效 padding、确定性开关及 tile/core 分界。

组合验证将配套 Ascend C MaxPool2dWithMask 输出的 out/indices 直接交给本算子，保持 indices 字节不变，并与独立 PyTorch 池化反向及附件 Golden 对照。验证相同最大值、正负梯度、重叠累加及非连续输出。两平台都执行该链路，不能用 CPU 生成索引或 SDK forward 的结果替代。

精度同时记录附件逐 dtype 的 err_threshold 和[生态混合容差标准](https://gitcode.com/cann/opbase/blob/30d1140e31c778b9305faab733a075b383eb759b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)的结果。该标准采用 `abs(actual-golden) <= atol+rtol*abs(golden)`，满足比例至少 0.99；FP32 的 atol/rtol 为 2^-13，FP16 为 2^-10，BF16 为 2^-7。内部自验同时限制最大绝对误差：FP32 不超过 0.01，FP16/BF16 不超过 0.1，不使用 ULP 分支放宽此限制。任一组要求未满足均记录失败，不选择较宽阈值掩盖差异。标准来源已从原链接 experimental_standard.md 重命名，交付记录固定实际使用版本。

### 性能

使用官方 msopprof 及其实际支持的分析能力，在相同 SDK、输入、dtype 和计时范围下逐 case 对比；记录设备、频率、预热及采样方式，并核对实际被调用的基线 kernel。一次 ACLNN 调用包含的多个 kernel 全部纳入对应调用分析，不能只选计算主 kernel。

A2 要求每例 `基线耗时/候选耗时 >= 0.95`，平均加速不能抵消慢例。小于 100 us 且超过参考耗时 30% 的场景，按任务书提供官方仿真图和分析。310P 以对应 A2/910B3 时间的五倍为参考，超过该参考 30% 时提供官方仿真和分析。自研前向用于功能配套，不替代 backward 性能比较所要求的基线。

### YOLOv11 / DOTAv1 模型验证

采用 YOLO11n-OBB，固定 Ultralytics 源码 `9d25f80daa5b3af71c16521eba353cb6e7fb66e2` 和 yolo11-obb.yaml 的 n 规模，初始权重使用官方 yolo11n-obb.pt。使用 DOTAv1 官方训练/验证划分；执行前归档权重和数据清单哈希、预处理及全部训练配置。

通过 autograd Function 替换 SPPF.m。SPPF 的池化配置为 kernel=5、stride=1、padding=2，连续调用三次。forward 调用配套 Ascend C 前向并保存其 indices；backward 调用本自定义 ACLNN 接口，不重新计算 argmax，不静默回退。每次组合调用记录实际库和 kernel 来源。

模型对照分为 A2 内置 backward、A2 自定义 backward、310P 自定义 backward 三组；配套前向及其他训练条件保持一致。采用 FP32、batch=2、imgsz=640、1 epoch、SGD、lr0=0.001、momentum=0.9、weight_decay=0.0005、seed=52，关闭 AMP 和随机增强，固定样本顺序。先在相同真实训练批次上核对输入梯度、参数梯度和权重更新，再完成微调与完整验证集评估。

本设计采用受控算子接入：池化前后向在对应 NPU 执行，其他训练层及优化器采用相同 CPU 路径，以隔离待测算子的影响。更新后的两组自定义模型分别导出并在 A2、310P 上运行完整验证集推理，CPU 负责一致的预处理、NMS 和指标统计；mAP50 绝对差不超过 0.01。该安排是本设计提交评审的模型验证方案，不表示已经获得模型验收认可。

报告保留真实前后向调用次数、梯度对照、权重变化、导出和编译配置、完整验证集规模及 mAP50。仅推理或仅池化子图梯度一致均不作为完整模型接入通过的证据。

## 当前验证范围

截至 2026-09-23，最小 FP32 工程在 A2/310P 的 CANN 9.1.0 上均完成标准构建、安装、自定义 ACLNN 示例以及连续/非连续输入输出验证。三次池化的 autograd 子图均实际执行自定义 NPU backward，输入梯度与独立 PyTorch 对照的最大绝对误差为 0。

当前工程原型限制每个输入/输出总元素数不超过 4096、单维不超过 64，采用单核累加，尚不具备目标设计的完整功能和性能。子图验证的索引由框架打包产生；配套 Ascend C 前向与本反向的直接组合、全部 dtype、原始 140/103 条、性能及完整 YOLO/DOTAv1 验证仍需完成。这些实现进展不改变以上目标接口和验收指标。

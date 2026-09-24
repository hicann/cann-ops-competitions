# MaxPool2dWithMaskBackward 算子设计

# 需求背景（required）

## 需求来源

本算子来自 2026 年 9 月 CANN 社区任务“MaxPool2dWithMaskBackward 算子开发”。任务要求参考
`aclnnMaxPool2dWithMaskBackward` 的公开语义，在 Ascend C 中实现 `MaxPool2dWithMask` 的反向传播，目标硬件为
Atlas 800T A2 和 Atlas 300V Pro。

本文只描述实现设计。精度、性能、内存和 Profiler 结论必须来自目标设备上的正式测试，不以静态分析代替。

## 背景介绍

正向最大池化为每个输出位置保存输入平面内的最大值索引。反向算子读取上游梯度和索引，将每个梯度累加回
对应输入位置；未被引用的位置保持为零。池化窗口重叠时，多个输出可能指向同一个输入位置，因此实现必须保留
累加语义。

公开 ACLNN 接口为两段式调用：第一段根据输入和属性返回 workspace 大小及 executor，第二段在调用方传入的
stream 上执行。输入顺序为 `gradOutput, self, indices`，输出为独立的 `gradInput`。

# 需求分析（required）

## 需求描述

- 输入和输出均为逻辑 NCHW 四维张量。
- `gradOutput`、`self`、`gradInput` 使用同一 dtype，支持 FP16、FP32；BF16 仅在 A2 支持。
- `indices` 是 INT8 物理容器，其有效前缀按小端 INT32 解码。
- 支持 `kernelSize` 长度 1/2、`stride` 长度 0/1/2、`padding` 长度 1/2、`dilation` 长度 1/2且值为 1，以及
  `ceilMode` 的 true/false。
- 支持非连续输入及输出。正式公共L2入口使用 `Contiguous` 物化输入，使用设备 `ViewCopy` 保留输出stride、storage offset与未选中存储区；不能只依赖输出 `AutoContiguous`。
- 默认允许非确定性累加；运行时开启确定性后必须走无跨核写冲突的固定顺序路径。

## 需求拆解

1. 完整校验 shape、dtype、属性、mask 容器和 workspace 溢出。
2. 将 mask 有效前缀直接解释为每个 N/C 平面内的 INT32 扁平索引。
3. 统一使用 FP32 累加，最终只转换一次输出 dtype。
4. 默认路径为低 N/C 场景提供跨输出元素的多核并行；确定性路径以完整 N/C 平面分核。
5. 在 A2 和 310P 上分别确认实际构建 SoC，不使用 A3 型号代替。
6. 按官方用例和正式计时协议验证精度与性能，不在设计阶段预设通过结论。

# 详细设计（required）

## 算子分析

### 数学公式

令 `p(n,c,oh,ow)` 为 `indices` 解码得到的、位于 `[Hi, Wi]` 平面中的扁平索引，则：

```text
gradInput[n,c,:] = 0
gradInput[n,c,p(n,c,oh,ow)] += float32(gradOutput[n,c,oh,ow])
```

所有输出位置处理完成后，将 FP32 累加结果一次性转换为 `gradOutput` 的 dtype。

### Mask 物理布局

`indices` 的完整 shape 为：

```text
[N, C, kernelH * kernelW, ((Ho * Wo + 15) / 16 + 1) * 32]
```

容器开头 `N * C * Ho * Wo * 4` 字节构成连续的小端 INT32 数组；尾部是 padding，不参与计算。每个索引均
相对于自己的 `[Hi, Wi]` 平面，合法范围为 `[0, Hi * Wi)`。

### 输出形状

对每个空间维度，在 dilation 固定为 1 时：

```text
floor: out = floor((input + 2 * padding - kernel) / stride) + 1
ceil : out = ceil ((input + 2 * padding - kernel) / stride) + 1
```

ceil 模式还要应用末窗修正：若 `(out - 1) * stride >= input + padding`，则 `out -= 1`。`gradInput.shape`
严格等于 `self.shape`。

## 算子实现

### Host 侧设计

Host tiling 完成以下工作：

1. 校验四个张量均为四维，N/C 一致，且 `gradInput.shape == self.shape`。
2. 解析长度为 1 或 2 的属性；空 `stride` 使用 `kernelSize`。
3. 校验 kernel、stride、padding、dilation，并按上述公式重新计算 Ho/Wo。
4. 校验 `indices` 的完整物理 shape 和 dtype 组合。
5. 使用 64 位安全乘法计算平面大小、元素总数和 FP32 workspace；任何溢出均返回失败。
6. 根据实际UB大小及各路径实际队列/数组/掩码字节数计算tile。散射按16元素对齐；向量路径按64元素对齐以满足Compare完整repeat，最多8192元素。原子更新每个FP32值占独立32字节槽位。
7. 通用非确定性模式按扁平梯度范围分核，通用确定性模式按完整N/C平面分核；无写冲突的向量路径按输入位置或步长行带分核，两种确定性设置均可使用。
8. FP16、FP32、BF16 使用不同 tiling key。310P 的算子配置不注册 BF16，Host 再次拒绝 BF16。
9. 310P 初始基线固定单 AI Core，不与 A2 共用 AIV 数量或硬件 barrier。

N或C为0时，L2直接返回空executor，不追加内核任务。H/W和Ho/Wo必须为正。

### 正式公开API与视图适配

公开函数保持任务原型的参数顺序与可写输出声明：

```text
aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(
  const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices,
  const aclIntArray* kernelSize, const aclIntArray* stride,
  const aclIntArray* padding, const aclIntArray* dilation,
  bool ceilMode, aclTensor* gradInput,
  uint64_t* workspaceSize, aclOpExecutor** executor)
aclnnMaxPool2dWithMaskBackward(
  void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
```

L2先返回标准NULLPTR/INVALID参数错误，再创建标准executor。非空时依次加入输入Contiguous、
自定义AICORE算子和输出ViewCopy；所有临时张量由executor管理，全部设备任务使用调用方stream/workspace。
没有CPU计算fallback，不通过内部辅助入口替代正式API测试。C++驱动以严格函数指针static_assert检查接口，
用绝对路径dlopen并核对dladdr；本实现没有Torch注册schema，不编造Torch schema证据。

### Kernel 侧设计

#### 默认非确定性路径

1. 各核清零互不重叠的 FP32 workspace 范围。
2. 全核同步后，各核读取自己负责的连续 `gradOutput` 和 INT32 索引范围。
3. 梯度经 scalar 读入并转换为 FP32，存入独立 32 字节对齐的 UB 槽位；scalar 写入完成后显式同步到 MTE3。
4. 使用 FP32 原子加将梯度散射到 workspace。不同核心可能以不同顺序更新同一位置，因此该路径是非确定性的。
5. 第二次全核同步后，各核读取互不重叠的 workspace 范围，转换一次并写出 `gradInput`。

该路径设置需要全核同步的调度模式。同步位置分别覆盖“清零完成后”和“全部原子累加完成后”，避免读写阶段交叠。

#### 确定性路径

1. 每个 N/C 平面只交给一个核心；FP32 workspace 平面 stride 按 128 元素（512 字节）对齐，避免相邻平面的 scalar cache false sharing。
2. 核心按扁平输出索引递增顺序处理梯度，与官方 golden 的累加顺序一致。
3. scalar 更新后按对齐的 cache-line 地址刷回；完成整个平面后再转换为输出 dtype，搬出时不带 workspace padding。

确定性路径优先保证固定累加顺序。大空间平面的进一步分块优化必须在不改变顺序和无跨核冲突的条件下进行。

#### A2 3x3 stride-one 向量 gather

针对 kernel=3x3、stride=1、padding=1、Hi/Wi>=3、每平面元素数小于2^24且总输入不少于16384的场景，
按输入梯度范围分核，每个输出位置只由一个核心写入。按扁平输出梯度索引递增顺序，比较九个候选窗口的
INT32 argmax 与目标像素索引，向量选择对应梯度并做FP32累加，最后一次转换写出。小于2^24的索引可以
无损转换为FP32用于向量相等比较；超出该范围保持原路径。

DMA按平面边界及每个候选的有效区间切分，整块越界候选跳过，禁止跨平面读取。W>=3时九个扁平偏移互不重复；
行边界的绕行候选不可能通过合法正向3x3窗口的argmax比较。该路径不使用原子加、跨核同步或用户workspace，
默认和确定性模式均可选择；其余参数保留上述通用路径。该优化不改变公开API、mask布局或用例数据分布。

#### A2 不重叠与反向窗口向量路径

kernel各维为2或4、stride等于kernel且padding=0时，每个输入最多对应一个候选输出。按输入分块Gather
候选索引和梯度，比较argmax后写零或梯度；floor未覆盖末行/末列与ceil部分窗口均显式处理。
满足UB约束的2x4窗口采用window-major Brcb广播，最后一次Gather恢复NCHW，减少按输入逐元素Gather。
形状地址表在每次NPU调用内部构造并复用，不跨调用缓存，不由测试预计算。

对于有界的重叠窗口（strideW>=2，每输入候选数<=9等资源条件），每个tile只载入覆盖候选的连续输出跨度。
越界候选只在UB中填充不可能索引及零梯度，GM读取严格裁剪到本通道；候选按扁平输出顺序递增累加FP32。
地址域限定每平面<2^21，通过半单位偏置和明确CAST_FLOOR计算整数坐标，超出域保持通用路径。
ceil/floor、奇数宽度、非连续存储、确定性和原始数据分布均通过独立套件验证，不以case编号决定算法。

当前另有2x2广播及3x3/stride2/padding1紧凑步长单元广播候选，尚在开发验证。后者先在紧凑候选数组Gather，
再广播到步长单元，以参考顺序累加，最后重排有效输入。未完成该候选真机测试前，不将其计入已验证版本。

#### 310P 初始功能基线

编译时对低于 220 的目标架构选择独立 header，避免编译器解析 A2-only API。Host 固定单核心；每个平面先
清零复用的 FP32 workspace，再按输出扁平索引递增逐项累加，最后转换一次并 scalar 写入 GM。结束时使用
310P 支持的双模板参数 cache flush。此路径设计为 NPU AI Core 设备端执行，不含 CPU 计算分支；但是源码尚未编译，
真实 kernel 路由和性能也尚未验证，不能宣称通过。单核版本只是待验证的功能基线。

### Workspace

workspace 大小为：

```text
A2 默认: systemWorkspace + N * C * Hi * Wi * sizeof(float)
A2 无冲突向量gather/broadcast: systemWorkspace
A2 确定性: systemWorkspace + N * C * align_up(Hi * Wi, 128) * sizeof(float)
310P 非空: systemWorkspace + Hi * Wi * sizeof(float)
```

上述为自定义kernel的工作区需求。systemWorkspace来自实际平台`GetLibApiWorkSpaceSize`，经标准CompileInfo传入tiling。
Host对大小加法做溢出检查，kernel通过`GetUserWorkspace`获取用户区域。公开L2返回的总workspace还可能包含
非连续输入物化和输出ViewCopy临时缓冲，不能把kernel工作区公式当作公开API总大小。空N/C无需这些设备缓冲。

### Tiling key

| tiling key | 输入/输出 dtype | 目标硬件 |
|---|---|---|
| 0 | FP16 | A2、310P |
| 1 | FP32 | A2、310P |
| 2 | BF16 | A2 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas 800T A2（构建 SoC 需由实际设备确认） | √ |
| Atlas 300V Pro / Ascend 310P | √ |
| A3 或其他型号 | × |

## 算子约束限制

- dilation 仅支持 1。
- 不支持 NaN 和 `-Inf` 作为合法精度输入。
- 当前任务原型限定四维 NCHW。
- mask前缀必须包含足够的INT32索引，并来自合法正向池化：索引属于对应窗口内的输入位置。不是任意散射索引API。
- 任务材料没有授权输入输出原地别名。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | 逐 case 使用官方阈值，并同时保存 mixed tolerance、matched ratio 和最大绝对误差 | 官方 cases；生态算子开源精度标准 |
| A2 性能 | 相对现有 TBE 不低于 95% | 官方任务书 |
| 小 shape | 小于 100 us 且慢于 TBE 30% 以上时提供仿真图和分析 | 官方任务书 |
| 310P | 功能与精度必测；耗时以 A2(910B3) 的 5 倍为参考，超出 30% 时提供分析 | 官方任务书 |

所有性能统计必须复刻正式 runner 的 warmup、采样次数、同步点、计时范围以及平均值/中位数/P90 计算方式。
这些参数未确认前，不给出性能通过结论。

## 测试分层

1. preflight：CANN、编译器、NPU 型号、Git HEAD/tree、工作树和构建 SoC。
2. build：从确认的清洁 HEAD 构建目标 experimental 算子。
3. smoke：dtype/重叠索引通过正式公开API执行；另行验证ceilMode、空batch、非连续和确定性，并分别保存未运行或失败状态。
4. 重点/临界性能：连续至少三轮，以最差轮判定。
5. 全量精度：未经修改的 140 条官方用例单独执行并逐 case 记录阈值与结果。
6. 全量性能、内存、msprof：保存原始 CSV/JSON、日志与 Profiler 目录。
build 后、smoke 前采集 runtime identity：保存测试程序和最终共享库的绝对路径、SHA-256、`ldd`、`readelf` 和 `nm -D`。
静态二进制身份只证明待测文件，实际加载与 kernel 路由由 smoke 的 `dladdr`、设备日志以及最终 msprof 共同确认。

## 兼容性与接入风险

仓库现有 `aclnnMaxPool2dWithMaskBackward` 在部分架构上会经适配层调用既有
`MaxPool3DGradWithArgmax`/`MaxPoolGradWithArgMaxV1` 路径，而官方任务 cases 的直接入口是
`op_name=MaxPool2dWithMaskBackward`。因此真机 smoke 必须同时记录直接算子 harness 和公开 ACLNN 入口的实际
加载库与 kernel 名称，确认测试确实命中新实现，而不是旧库或 CPU fallback。若公开 ACLNN 路径没有命中新实现，
需要在不改变公开函数签名的前提下补齐适配层接入，并单独保存修改原因和回归证据。

## 开发验证进展（2026-09-19，非正式验收结论）

A2实际设备为910B3，CANN9.0.0，构建family为ascend910b。开发提交645b4ba4已完成清洁构建与二进制身份、
3dtype smoke、原7组/不重叠22组/反向窗口52组泛化（各重复3次）、18重点连续3轮，原始140条精度以及
补充正态分布/非连续视图140条精度均通过；分别保存生态标准和独立FP64复核。该HEAD全量开发性能140条、
公开API边界、内存观测和三种dtype的Profiler已完成；归档3350288字节，SHA256为
`3727386cf1eed1e2043990643fbb066dbb16bcc2cbe040e027090ad7bedc3263`，3512归档成员逐项哈希验证。
系统公开API同口径开发对比仍有42/140项低于95%诊断线，不将采样完成写成正式TBE指标达标。
后续stride-cell/2x2优化961c0854已完成独立清洁构建、7/22/56组泛化各三次和18重点连续三轮；
仅为重点开发版，不挪用645b4ba4的完整回归结论。新的四行窗口优化尚在测试。
此前4d6a5ec1等完整开发版已独立封存，不将旧HEAD结果搬到新HEAD。

gradOutput采用已记录的本地uniform假设；补充正态分布不替代原始uniform。开发计时为5warmup/20samples，
同时保存Execute+sync与完整公开API分配+sync的mean/median/P90；不能声称复刻原包未提供的正式runner。
系统公开API对照的实际Profiler路由为MaxPool3DGradWithArgmax，不冒称已证实TBE。
普通执行精度通过不等于内存安全通过：插桩检查在系统StridedSlice视图转换中仍报告错误，保留失败日志并继续诊断。
310P/model与正式TBE门槛仍未完成，不标记通过。个人代码fork已创建并已邀请Ascend-CANN开发者（待接受），
尚未推送最终代码或提交正式验收。

AI辅助说明：OpenAI Codex；本次修订使用用户选择的GPT-6-Astra。早期草稿记录为GPT-5系列，具体变体未留档，
不补写未经核实的历史型号。接口、源码、命令和测试结论以保存证据为准。

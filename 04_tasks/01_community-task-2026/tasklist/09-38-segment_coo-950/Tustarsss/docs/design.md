# segment_coo 算子设计文档

# 需求背景（required）

## 需求来源

本算子来源于 2026 年社区任务“segment_coo 算子开发”。任务要求参考 `torch_scatter 2.1.2` 的 `segment_coo`、`segment_sum_coo`、`segment_add_coo`、`segment_mean_coo`、`segment_min_coo` 和 `segment_max_coo`，在 Ascend 950 系列上以 Ascend C Kernel、Host 适配和 PyTorch 接口实现相同前向语义。

验收以 PyTorch 层公开接口为准，目标环境为 Ascend 950 系列、CANN 9.1.0 及以上、PyTorch 2.7 及配套 `torch_npu`。C++ aclnn 接口不是本任务必选项。

## 背景介绍

Segment COO 根据已经沿归约维非降序排列的 COO 索引，把连续属于同一编号的源元素归约到对应输出段。它适用于图邻接边聚合、有序分组统计和 PyG 稀疏消息传递。与通用 scatter 相比，Segment COO 可以利用索引有序性确定每段边界，使每个输出元素由唯一任务顺序计算，不需要全局原子竞争。

### 原接口能力与约束

设 `src` 为 $n$ 维，`index` 为 $m$ 维且 $m\le n$，归约维为 $d=m-1$。`index` 的前 $m-1$ 维可以广播到 `src`，最后一维沿 $d$ 维非降序。输出形状与 `src` 相同，仅第 $d$ 维替换为段数 $S$。

| 参数 | 类型 | 数据类型 | 约束 |
|---|---|---|---|
| `src` | 输入 Tensor | FP16、FP32、BF16、INT8、UINT8、INT32、INT64 | rank 1～8，NPU strided Tensor |
| `index` | 输入 Tensor | INT64 | `index.dim() <= src.dim()`；最后一维有序、非负且不越界 |
| `out` | 可选输入/输出 Tensor | 与 `src` 相同 | shape 除归约维外与 `src` 相同；支持可证明不重叠的非连续布局 |
| `dim_size` | 可选属性 | INT64 标量 | 非负；有 `out` 时按 CPU 语义忽略 |
| `reduce` | 属性 | STRING | `sum`、`add`、`mean`、`min`、`max` |
| `arg_out` | min/max 子接口输出 | INT64 | shape 与 value 输出相同 |

### 实现前现状与主要难点

任务开始时目标仓没有 Ascend NPU 的 Segment COO 实现。直接照搬 scatter 会丢失有序索引带来的性能优势；只按 shape 猜测“等长分段”又会在 index 内容变化时产生错误。实现还必须同时处理：

1. 广播 index、非连续 Tensor、已有 `out`、空桶和 min/max arg 语义；
2. INT8/UINT8/INT32/INT64 的固定位宽溢出和整数 mean 向零截断；
3. FP16/BF16 累加、次正规数和极值哨兵置零规则；
4. 原生 NPU 分配器与 DLPack 等外部存储的异步生命周期；
5. 小 shape 的 Host/launch 开销和大 shape 的 GM/UB 搬运开销；
6. 任务固定的 464 项性能矩阵全部满足 `reference_ms / npu_ms >= 0.45`。

# 需求分析（required）

## 需求描述

在 DAV_3510（Ascend 950PR）上实现与 CPU `torch_scatter 2.1.2` 对齐的 Segment COO 前向算子。公开接口必须支持七种数据类型、五种归约模式、广播、空张量、非连续输入/输出、可选 `out`、可选 `dim_size` 以及 min/max 的 arg 输出。非法 reduce、非法 shape、无序/负数/越界 index 必须在破坏用户 `out` 之前报错。

性能路径仍需逐次验证 index 内容，不缓存“某个 Tensor 曾经等长”的结论。优化不得改变 CPU 标杆、容差、固定性能参考值或 0.45 门槛。

## 需求拆解

1. 提供六个与任务书一致的 Python API；`segment_coo` 只返回 value，min/max 子接口返回 `(value, arg)`。
2. 支持 FP16、FP32、BF16、INT8、UINT8、INT32、INT64；index 固定 INT64。
3. 支持 `sum/add/mean/min/max`，拒绝 `mul` 和其他字符串。
4. 支持 rank 1～8、index 广播、空输入、空桶、推导或显式 `dim_size`。
5. 支持原生连续/非连续 Tensor、逻辑负视图、外部 DLPack 存储和 src/out 地址重叠。
6. 在设备侧完整检查 index 的非负、有序、上界和最大值；检查失败不得修改 out。
7. 通用路径覆盖全部功能；针对任务矩阵建立经过内容验证的等长快路径和 FP32 长段路径。
8. 整数与 arg 逐位一致；浮点以 CPU `torch_scatter` 为单一标杆并满足任务精度标准。
9. 正式 464 项性能全部达到 0.45，并保留多进程、逐窗口稳定性证据。
10. 当前任务只要求前向；启用梯度且输入需要梯度时明确拒绝，避免返回错误梯度。

# 详细设计（required）

## 算子分析

### 数学公式

对 batch $b$、段 $s$、特征 $k$，定义 $J_{b,s}=\{j\mid index_{b,j}=s\}$。各模式输出为：

$$
out_{b,s,k}^{sum}=\sum_{j\in J_{b,s}}src_{b,j,k},
$$

$$
out_{b,s,k}^{mean}=\begin{cases}
\operatorname{narrow}\left(\sum_{j\in J_{b,s}}src_{b,j,k}\right)/\operatorname{narrow}(|J_{b,s}|),&|J_{b,s}|>0,\\
0,&|J_{b,s}|=0,
\end{cases}
$$

$$
out_{b,s,k}^{min}=\min_{j\in J_{b,s}}src_{b,j,k},\qquad
out_{b,s,k}^{max}=\max_{j\in J_{b,s}}src_{b,j,k}.
$$

无 `out` 时空桶 value 为 0；有 `out` 时空桶保持原值。min/max 采用严格比较，因此普通相等值保留首个位置。若 CPU 有限极值哨兵在非空段中始终未被严格更新，则 value 按 CPU `masked_fill` 规则置 0。arg 为空桶时为 $N$；无严格 winner 时沿 CPU 规则继承前序 carry。

### 支持数据类型

| `src/out` dtype | sum/add | mean | min/max | 正式性能 |
|---|:---:|:---:|:---:|:---:|
| FP16、FP32 | √ | √ | √ | √ |
| INT32、INT64 | √ | √ | √ | √ |
| BF16、INT8、UINT8 | √ | √ | √ | 功能验收 |

FP16/BF16 在通用和等长归约中使用 FP32 累加，再在 CPU 语义要求的位置窄化。整数运算保持存储位宽溢出，不经过浮点中转；整数 mean 对负数按向零截断处理。

### 支持形状与逻辑展开

Host 将逻辑布局展开为：

```text
src    : [B, N, K]
index  : [B, N]（从原 index 广播）
output : [B, S, K]
```

其中 $B$ 是归约维之前各维乘积，$N=src.size(d)$，$K$ 是归约维之后各维乘积。所有乘积均检查 `INT64_MAX` 溢出。物理格式接受线性 ND/NCHW/NCDHW；私有格式输入规范化为线性连续布局。用户 `out` 可以是转置、步进或逻辑负布局，但 stride 必须能证明内部无重叠。

### 路径选择

| 路径 | 触发条件 | 作用 |
|---|---|---|
| 通用路径 | 所有合法输入的兜底 | 完整广播、out、arg、七 dtype 和任意有序分段 |
| 等长 RegBase 快路径 | 无 out/arg；显式 dim_size；单 batch、一维 index；N 可等分；段长为不超过512的2次幂；K∈{32,64,128,256}；FP16/FP32/INT32/INT64 | 融合 index 验证、DMA 和寄存器归约，降低小算子 Host/launch 开销 |
| FP32 长段路径 | 无 out/arg；FP32；K∈{32,64,128,256}；存在至少513行的段；输出密度满足限制 | 使用 UB 分块和特征维 SIMD 处理长段，保持原加法顺序 |

shape 只决定候选路径。等长快路径必须 DMA 读取并验证其负责范围内的全部 INT64 index；任何不匹配都会等待已启动工作完成后回退通用路径。

## 算子实现

### Host 侧设计

Host 入口 `SegmentCooForwardImpl` 依次执行：

1. 检查 device、layout、rank、dtype、index dtype、广播关系、reduce、out shape/stride 和 `dim_size`；
2. 在任何规范化或异步提交之前，对原始 src/index/out 调用 `NPUCachingAllocator.recordStream`；
3. 根据 shape 选择等长候选；候选内部完整验证 index，失败则回退；
4. 通用路径启动融合 index 检查，读取固定 2048×2 个 INT64 状态，得到错误位、最大索引和长段标记；
5. 推导段数并分配 output/arg/bounds/summary；
6. 查找每段边界，处理 src/out 地址重叠，启动通用或长段归约；
7. 对非连续 out 执行一次写回并维护版本计数。

框架队列路径通过 `torch_npu::OpCommand::RunOpApi` 保持与前序框架算子的提交顺序。等长快路径在输入规范化和输出分配后显式排空 Host 队列，再直接提交到当前 ACL stream，减少重复 Host 包装。

对 DLPack 等外部存储，`recordStream` 不能保护非缓存分配器 owner。接口会持有原始 Tensor，并在正常返回、早返回和异常路径中排空 Host 队列、同步当前 stream。该路径同步返回；原生存储仍保持异步。

### Tiling 与多核策略

通用路径的 `Tiling` 包含 batches、rows、features、segments、dtype、reduce、workspace 和已有 out 标志。Bounds 阶段按 `(batch, segment boundary)` 划分；归约阶段按 `(batch, segment, feature)` 划分，每个输出只有一个写者。block 数由实际工作量和平台可用 AIV 数动态限制，不写死某个 SKU 的核数。

通用边界查找对每个段号执行 lower_bound，复杂度为 $O(B\times S\times\log N)$。归约阶段只顺序读取 `[begin,end)`，不使用全局原子。

等长快路径按完整段切分到各 AIV。每核使用：

| UB 区域 | 大小 | 用途 |
|---|---:|---|
| input | 128 KiB | 一个或多个完整段，必要时按特征切片 |
| output | 64 KiB | 当前段结果 |
| index | 32 KiB | 当前核负责范围的 INT64 index |
| status | 32 B | 独立校验状态槽 |

特征 tail 使用有效通道生成 `MaskReg`，UB 容量和 DMA offset 使用对齐长度；padding 不参与归约也不写回。MTE2→V、V→MTE3 等真实生产者/消费者关系使用对应 `SetFlag/WaitFlag`，不以 `MaskReg` 代替同步。

### index 完整校验

通用检查 Kernel 将 index 元素分给 2048 个 lane，并融合计算：

- 是否存在负数或同一原始 index 行内降序；
- 在段数已知时是否越界；
- 最大 index，用于推导 `dim_size`；
- 候选 FP32 长段标记。

相邻比较不跨原始 index 行边界。广播前检查原始 index 即可覆盖所有逻辑副本。非法 index 在 output 归约和用户 out 写入之前抛错。

等长快路径的 `CheckUniform` 按候选段长检查低32位等于期望段号、高32位为0。每个 AIV 写入自己的映射状态槽；Host 只在所有核状态成功后接受候选结果，不缓存 index 值、Tensor 版本或历史结论。

### 通用 Kernel

通用 Kernel 分两遍：

1. `BoundsKernel` 输出 `[B,S+1]` 段边界；
2. Reduce Kernel 让一个任务顺序处理一个 `(batch, segment, feature)`。

sum/add 以已有 out 或 0 为种子；mean 先按 dtype 语义累加，再按窄化段长除法；min/max 使用有限类型界值或已有 out 为初始值。INT8/UINT8 mean 若非空段长度窄化到 0，会在修改 out 之前拒绝，以避免 CPU 未定义除零。

src 与 out 即使属于不同 `StorageImpl` 也可能共享设备地址。Host 根据真实地址区间检查重叠；重叠时用独立 Ascend C 字节拷贝 Kernel 保存 src，避免输出覆盖后续输入。

### min/max arg 传播

当需要 arg 且 `B×S<=256` 时，按 feature 串行传播 winner carry。更大输出按每256段划块：

1. 每个 `(feature, block)` 独立归约并输出块摘要；
2. 按 feature 传播块摘要；
3. 并行修补块内尚未解决的前缀。

空桶 arg 初值为 $N$，不改变 carry；carry 可以跨 batch 延续，与 CPU 固定语义一致。只返回 value 时不分配 arg 或 summary，也不启动传播 Kernel。

### 等长 RegBase 快路径

RegBase 实现位于 `segment_coo_uniform.h`。Kernel outer shell 管理 GM/UB 绑定、DMA、tile 循环和 MTE/V 同步；`CheckUniform`、`UniformReduce`、`Half32` 和 `Int64ExtremaPairs` 为 `__simd_vf__` 寄存器计算主体。VF 从 UB 以 `LoadAlign` 读入 `RegTensor`，通过 `UpdateMask` 生成有效通道，用 Add/Compare/Select 完成计算，再由 `StoreAlign` 写回 UB；API 签名已按 CANN 9.1 当前 SDK header 核对。

短段采用多段交错隐藏寄存器依赖：FP32/INT32 最多八段，常规路径四段；FP16 C32 将两段打包进 FP32 向量。INT64 min/max 只在原始特征数为32、完整 index 校验通过、无 out/arg 的非空等长段使用双段独立累加器，共享行地址计算；奇数尾段、宽通道和不足两段的 tile 保持原单段路径。

FP16 在寄存器中升为 FP32 顺序累加；INT32/INT64 保持整数位宽。浮点 mean 先按 CPU 规则窄化，再乘精确二进制倒数；整数 mean 对负数增加偏置后算术右移，实现向零截断。编译关闭 FTZ，保留次正规数。

### FP32 长段路径

融合 index 检查在同一原始行内比较相距512的 index；相等表示存在至少513行的段。显式输出过稀疏时不启用，派发要求 `segments<=4096` 或 `segments/8<=rows`，推导 `dim_size` 后再次检查。

通用 SIMT Kernel 处理短段，独立向量 Kernel 处理长段。长段输入按原行顺序搬入128 KiB UB，在特征维做 SIMD 归约；FP32 累加状态跨 UB 块完整保存，不使用改变加法顺序的部分和。mean 的有序和先写回 GM，再由后续 SIMT Kernel 除以完整段长，避免 RegBase 除法在目标机上清零次正规数商。out/arg、其他 dtype 和不支持的特征数继续走通用路径。

### 数值精度设计

| 场景 | 内部策略 | 输出策略 |
|---|---|---|
| FP16/BF16 sum/mean | FP32 顺序累加 | 按 CPU 语义窄化；mean 再除 |
| FP32 sum/mean | FP32 顺序累加 | 不重排加法；长段 mean 用 SIMT 除法 |
| INT8/UINT8/INT32/INT64 | 原位宽模运算 | mean 向零截断 |
| min/max | 严格比较 | 有限哨兵未更新时置0；arg保持CPU carry规则 |
| tail | 只对有效 lane 建 mask | 无效 lane 不参与计算、不写回 |

NaN 只要求类别一致，不约束 payload 或符号；Inf 必须一致。整数、有限 sum/min/max 和 arg 使用逐位比较；正式 mean 使用任务测试容差，并另有次正规数零容差专项。

## 支持硬件

| 支持的芯片版本 | 是否支持 | 验证状态 |
|---|:---:|---|
| Ascend 950PR（DAV_3510） | √ | 已在 CANN 9.1.0 真机完成构建、功能、精度、性能和稳定性验证 |
| Ascend 950DT | 设计目标同属 Ascend 950 系列 | 未在本工作区单独上板，不将 950PR 成绩冒充异机成绩 |
| Atlas A3/A2、其他架构 | × | 当前只编译 `arch35` / DAV_3510 路径 |

## 算子约束限制

1. `index` 必须沿最后一维非降序、非负且不超过输出段数；实现会逐次检查。
2. 不支持 `mul`、BOOL、COMPLEX、Float8。
3. 当前只支持前向，不支持 autograd/backward。
4. 用户 `out` 必须是 ND/NCHW/NCDHW 线性物理格式，且 stride 可证明内部不重叠。
5. BF16、INT8、UINT8 只参与功能验收，不进入任务固定性能矩阵。
6. 外部存储路径为保证 owner 生命周期会同步返回；原生分配器路径保持异步。
7. 等长和长段优化都有明确触发范围，其他输入始终回退通用路径。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
|---|---|---|
| 功能标准 | 六个 PyTorch API、七 dtype、五 reduce、广播、out、arg、非连续、空输入/空桶及错误路径与 CPU `torch_scatter 2.1.2` 一致 | 任务书及原题测试 |
| 整数精度 | INT8、UINT8、INT32、INT64 value 与 INT64 arg 逐位一致 | 任务书 |
| 浮点精度 | 使用 CPU `torch_scatter 2.1.2` 单标杆；正式用例按任务容差，补充 sum/min/max 及次正规数专项采用更严格比较 | 《生态算子开源精度标准》及任务书 |
| 性能标准 | 29 shape × 4 dtype × 4 reduce 共464项；每项20次预热、100次公开API整体计时、末尾同步；`reference_ms / npu_ms >= 0.45` | 任务书固定标杆 |
| 稳定性标准 | 完整轮次不拼接最好值；专项保留全部窗口和最慢值，失败不得以P99覆盖 | 项目复验口径 |

当前版本在 Ascend 950PR / CANN 9.1.0 上完成：本仓 1090 项检查全部通过（含9项 CPU 接口检查），独立 CSR 集成5项通过；三个独立进程的正式矩阵均为464/464精度与性能通过，最低性能比值分别为0.643709、0.644157、0.644501。16个新进程、8个代表场景共19200/19200长尾窗口通过，最差比值0.614453。有限测试不等于其他机器或无限运行保证。

## 测试分层与复现

| 层级 | 入口 | 目的 |
|---|---|---|
| 原题 | `test/segment_coo/original/test_segment_coo.py` | 任务TC、dtype/reduce、out、非连续、广播、负向接口 |
| 补充接口 | `test_segment_coo_npu.py`、`test_segment_coo_extra_npu.py` | 公开API、版本、别名、边界语义 |
| 优化专项 | uniform、lifecycle、external storage、generic、long、INT64 pairs 等测试 | 每条优化路径、回退、精度和生命周期 |
| 正式性能/精度 | `benchmark_segment_coo.py`、`scripts/remote/performance.sh` | 464项CPU正确性及公开接口性能 |
| 日常长尾 | `scripts/remote/stability.sh` | 完整464通过后复测最低比值与最小绝对余量项 |

目标环境复现：

```bash
bash scripts/remote/build.sh
bash scripts/remote/regression.sh
bash scripts/remote/performance.sh
# 完整矩阵通过后，按需执行精简长尾专项
bash scripts/remote/stability.sh
```

## 兼容性分析

这是 ops-gnn 新增算子，不替换已有算子。公开 Python API 对齐 `torch_scatter 2.1.2`，`sum` 与 `add` 等价；有 `out` 时采用真实 CPU 行为而不是任务附件中的简化描述。实现仅注册新的 `segment_coo_forward` 扩展入口，对其他 ops-gnn 算子无 ABI 修改。

当前构建路径面向 CANN 9.1.0、PyTorch 2.7.1、`torch_npu 2.7.1.post8` 和 DAV_3510。其他 CANN/PyTorch 组合、其他芯片、外部存储实现及未来 torch_scatter 版本需要重新构建和复验，不能直接沿用本报告成绩。

## 已知边界与取舍

- Host Release 编译选项和宽通道 INT64 双段候选没有形成稳定净收益，未保留。
- 当前 C32 INT64 min/max 双段路径在定向配对中获得约1.11～1.21倍中位收益；完整矩阵总体近似持平，诊断中仍有跨进程回退，未宣称所有配置加速。
- 正式矩阵和长尾专项均观察到非致命墙钟波动；现有最慢窗口仍高于门槛，但不声称完全没有系统尖峰。
- 仿真只用于功能与流水诊断，仿真耗时不作为真机性能成绩。

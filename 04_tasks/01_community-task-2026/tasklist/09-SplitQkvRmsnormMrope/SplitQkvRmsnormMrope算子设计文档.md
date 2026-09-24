# 需求背景（required）

## 需求来源

昇腾社区「9月社区任务-split_qkv_rmsnorm_mrope算子开发」。目标代码目录为 `ops-transformer/experimental/posembedding/split_qkv_rmsnorm_mrope`，采用 ACLNN 算子工程模式。

## 背景介绍

Qwen3-VL 的线性投影输出由逐 head 交错的 Q/Gate 和连续 K、V 组成。融合拆分、RMSNorm 和三轴 MRoPE，可省去中间张量落入全局内存以及多个 kernel 的发射开销。对照实现为 vllm-ascend 的 `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`。

# 需求分析（required）

## 需求描述

输入输出均采用 BF16、ND 格式；head_size 固定 256，rope_dim 固定 64，mrope_section 固定 [11,11,10]。Q/K 的平方求和、归一化、加权、可选偏置、旋转均在 FP32 中计算，只在最终写回时转换 BF16。V/Gate 按 BF16 原始数据搬运。

## 需求拆解

| 参数 | 形状/含义 | 约束 |
|---|---|---|
| qkv | [T, (1+has_gate)×QH×256+2×KH×256] | BF16，连续 |
| q_weight/k_weight | [256] | BF16 |
| q_bias/k_bias | [256] | 独立可选，BF16 |
| cos_sin | [3,T,64] | 每轴前32为cos、后32为sin |
| q_output/gate_output | [T,QH×256] | 无Gate时gate_output为[T,0] |
| k_output/v_output | [T,KH×256] | BF16 |
| num_q_heads/num_kv_heads | 默认16/4 | 正整数，当前实现上限65535 |
| eps | 默认1e-6 | 正有限值 |
| interleaved | 默认true | 控制轴选取，不改变旋转配对 |
| has_gate | 默认true | 控制输入Q/Gate布局 |

# 详细设计（required）

## 算子分析

### 数学公式

对每一个 Q 或 K head，`z[d] = x[d] × rsqrt(sum(x²)/256+eps) × weight[d] + bias[d]`，无 bias 时省去该项。

对 `i∈[0,31]`：

- `y[i] = z[i]×cos[a(i),t,i] - z[i+32]×sin[a(i),t,i]`
- `y[i+32] = z[i+32]×cos[a(i),t,i] + z[i]×sin[a(i),t,i]`
- `y[64:256] = z[64:256]`

non-interleaved：轴区间为 [0,11)、[11,22)、[22,32)。interleaved 按任务 golden.py 的条件计算 height/width 掩码，再以 temporal 填补剩余位置。特别注意：两种模式均配对前32维与后32维，不是相邻维度配对。

### 支持数据类型与形状

支持任务书七种规格，以及一般正 QH/KH 的尾 head 分组。Host 检查输入宽度、权重/偏置长度、cos_sin 的三轴与 token 维、输出形状和 dtype。空 token 输入不进行全局内存读写；该场景仍需补充运行时实测。

## 算子实现

### host侧设计

1. 从平台获取 AIV 核数和 UB 容量，UB 至少160 KiB。
2. 当 QH=16、KH≤4 时，一个任务合并处理同一 token 的 Q/K heads，在一次归一化与旋转批次中共享统计流程和cos/sin。按 `T` 个任务划分连续区间。
3. 其余形状以每组最多16个同类 head 处理，`jobs = T × (ceil(QH/16)+ceil(KH/16))`。使用 `min(jobs,AIV核数)` 个核，job 按比例划分连续区间，平衡 Q/K 工作量，并复用相邻任务同一 token 的旋转表。
4. tiling 结构传递 T、QH、KH、has_gate、interleaved、两个bias存在标记和eps。使用单模板 key=0；模式标志为运行时值。
5. 应用数据无额外 GM 中间张量；构建框架保留16 MiB系统workspace，不作为算法中间存储。

### kernel侧设计

- 每核预加载 Q/K 权重及存在的偏置到 FP32 UB。
- Q 以二维跨步 DMA 从 Q/Gate 交错布局提取；Gate 从相同布局另行原样搬运。K、V 连续搬运。
- UB按最多20行预留、每行256维。合并路径批次包含Q和K，通用路径每组最多16行。RMSNorm 先计算平方，再合并4个64维片段，WholeReduceSum得到每行统计量，采用 Sqrt 加 Div 得到倒数平方根，避免近似 Rsqrt 在 BF16 舍入边界放大误差；Brcb将统计量广播为重复运算需要的布局。
- 权重和偏置按64维片段广播，利用 repeat stride 一次处理多行。
- 加载该 token 的3轴cos/sin，通过固定64元素 gather索引形成所需32个cos及32个sin；行间共享该表。
- 旋转先将4路乘积写入临时UB，再同时计算前后两半，避免覆盖后续仍需读取的输入。
- TQue/TQueBind 管理输入、输出及直通数据，显式向量流水屏障保护相依计算。
- 输入、输出和直通数据采用双缓冲，cos/sin 加载后的屏障只约束向量流水，允许其他搬运流水继续执行；后续优化以真实设备采样和精度回归为依据。

### UB预算

输入/输出/直通各2×10240 B，cos输入384 B，权重4096 B，主值/平方/临时各20480 B，统计1024 B，cos/sin及gather结果2048 B，索引256 B，总计130688 B，以160 KiB门槛预留对齐余量。

## 支持硬件

计划支持 Atlas A2 和 A3，分别配置 `ascend910b`、`ascend910_93`。本次独立调试环境为 Liu_QKVM_0923、Ascend910_9382、CANN 9.1.0。任务页面写A3、任务书3.1写Atlas800T A2，正式验收平台需维护者确认。

## 算子约束限制

不支持任意 head_size、rope_dim 或 mrope_section，不做输入广播。可选bias独立处理。主机侧自动连续化由算子注册声明提供，实际非连续输入调用路径待补充验证。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 严格标准 |
|---|---|
| Q/K | 任务golden及Triton逐元素最大绝对误差<1e-3 |
| V/Gate | 原始BF16位模式完全相同 |
| 性能 | NPU kernel时延严格小于对应Triton baseline/2 |
| 自测数据 | 任务包未附case.json引用的bin文件，生成数据必须记录seed及分布，不声称使用了缺失文件 |
| 性能统计 | 区分host launch+sync时延和NPU kernel时延；使用相同设备、同步及预热口径进行对照 |

七个验收case与baseline以原始spec/case.json和spec/requirements.md为准。当前扩展自测覆盖QH=17尾组、KH=3、两种轴模式、有/无Gate及双边bias随机输入。计划补充单边bias、零值/常量、空token及形状拒绝路径。保留真实运行日志和失败记录，未通过前不申请验收。

## 兼容性分析

新增算子，不修改已有算子接口。新增文件集中在独立算子目录；远端源码、日志及依赖均在独立环境和 `/workspace/Liu_QKVM_0923` 工作目录内。

## 当前验证状态（2026-09-23）

CANN 9.1.0 配套 v9.1.0 源码上编译、安装与 ACLNN 调用已完成。七个规定形状以seed=21928生成BF16输入，和任务golden.py在NPU上的输出逐位一致；四组尾head/bias/Gate/轴模式扩展用例也与独立FP32参考逐位一致。v7设备采样中七种规定形状均低于任务书给定baseline的一半，最长序列8192的kernel中位数为282.837 us（门槛334.7495 us）。补充112组随机输入、单/双边bias和不同head数测试，与任务原始golden.py在NPU上的输出均逐位一致。独立CPU参考在8组中存在BF16舍入差异，保留诊断记录。与同设备Triton实测kernel对照时，七组均未达到2倍加速；任务书所列baseline与本机实测值不同，需维护者确认验收口径。正式验收还需设计评审通过、确认目标平台及原始数据。此设计提交不表示任务已经验收通过。

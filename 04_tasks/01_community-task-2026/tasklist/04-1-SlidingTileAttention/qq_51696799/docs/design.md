# 需求背景（required）

## 需求来源

FastVideo 视频生成模型使用 SlidingTileAttention 在三维视频 token 画布上执行局部注意力。社区任务要求参考 FastVideo 的语义，在 Atlas A2 上使用 Ascend C 完成算子设计、开发与测试，并支持 `float16`、`bfloat16` 和 BNSD 布局。

本文依据当前任务书、当前测试集合同样本以及 `ops-transformer` 中现有 Ascend C 实现编写。测试集中的 case 仅用于覆盖合同边界和验证实现，不参与生产路径分派。

## 背景介绍

全量注意力的 QK 和 PV 计算量随序列长度平方增长。SlidingTileAttention 将图像 token 按固定三维 tile 划分，每个 image query 只访问所属 head 对应的邻域 tile；存在 text token 时，image query 还访问全部 text KV，text query 则访问完整 image 与 text 序列。该数据流既保留视频局部相关性，也避免为每个 image query 构造完整序列注意力矩阵。

当前实现位于 `experimental/attention/sliding_tile_attention`，公开接口为两段式 ACLNN，自有 Ascend C MIX kernel 在一次外部 kernel launch 内完成 QK、online softmax、PV、重标定、归一化和输出写回。

# 需求分析（required）

## 需求描述

实现与 FastVideo SlidingTileAttention 核心语义一致的 Ascend C 算子。输入 `q`、`k`、`v` 为相同 shape、dtype 的 rank-4 BNSD Tensor，输出与输入 shape、dtype 一致。算子根据三维画布、固定 tile、逐 head window 和 text metadata 生成注意力可见域，并执行 scaled dot-product attention。

公开 ACLNN 原型如下：

```cpp
aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclIntArray *windowSize,
    int64_t textLength,
    bool hasText,
    int64_t seqShape,
    aclTensor *output,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnSlidingTileAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

参数合同如下：

| 参数 | 输入/输出 | dtype | shape/取值 | 说明 |
| --- | --- | --- | --- | --- |
| `q` | 输入 | `float16`、`bfloat16` | `[B,N,S,D]` | query，BNSD |
| `k` | 输入 | 与 `q` 相同 | 与 `q` 相同 | key |
| `v` | 输入 | 与 `q` 相同 | 与 `q` 相同 | value |
| `windowSize` | 输入 | `int64` | 3 或 `3*N` 个值 | 单个 `[t,h,w]` 广播到全部 head，或逐 head 三元组 |
| `textLength` | 属性 | `int64` | 与序列族一致 | text token 数 |
| `hasText` | 属性 | `bool` | `true/false` | 是否包含 text token |
| `seqShape` | 属性 | `int64` | `1/2/3` | 三种已实现画布枚举 |
| `output` | 输出 | 与 `q` 相同 | 与 `q` 相同 | attention 输出 |

L2 对非连续 `q/k/v` 执行设备侧 `Contiguous`，输出按 ACLNN executor 约定写回。shape 一致的空 Tensor 在 GetWorkspaceSize 阶段返回 `workspaceSize=0`，不进入 L0 和 device kernel。

## 需求拆解

1. 完成 FP16/BF16、BNSD、广播 window 和逐 head window 的接口闭环。
2. 根据 `seqShape` 还原三维画布和固定 tile `[6,8,8]`，正确生成边界 clamp 后的局部 KV 区间。
3. 使用 runtime metadata 完成分核、tiling key 和内核数据流选择，不按 case id、输入值或 timing 特征分派。
4. 在一个 MIX kernel 内完成 QK、fp32 online softmax、PV 和输出归一化，避免独立 Cast/Softmax/PV kernel 链。
5. 正确处理 query、D 列和 KV stack 尾块，闭合 AIC/AIV 数据发布、等待和 workspace 生命周期。
6. 对非法 rank、dtype、shape、sequence/window metadata 明确报错，不调用 framework、CPU 或 vendor whole-op fallback。

# 详细设计（required）

## 算子分析

### 数学公式

对 batch `b`、head `n` 和 query `i`，令 `V(i,n)` 为该 query 的合法 KV token 集：

```text
score(i,j) = dot(Q[b,n,i,:], K[b,n,j,:]) / sqrt(D)
p(i,j)     = exp(score(i,j) - m(i)) / sum(k in V(i,n), exp(score(i,k) - m(i)))
O[b,n,i,:] = sum(j in V(i,n), p(i,j) * V[b,n,j,:])
```

其中 `m(i)` 为该行在全部 KV segment 上的最大值。image query 的 `V(i,n)` 由局部三维 window 对应的 image segments 和可选 text segment 组成；text query 的 `V(i,n)` 为完整序列。本算子为非 causal attention，不附加上三角 mask。

图像序列族如下：

| `seqShape` | 画布 `[T,H,W]` | image token 数 | text 合同 |
| ---: | --- | ---: | --- |
| 1 | `[30,48,80]` | 115200 | 当前源码合同为 `textLength=128, hasText=true` |
| 2 | `[36,48,48]` | 82944 | `textLength=0, hasText=false` |
| 3 | `[18,48,80]` | 69120 | `textLength=0, hasText=false` |

### 支持数据类型

| 输入/输出 dtype | QK | softmax/累加 | PV 与输出 |
| --- | --- | --- | --- |
| `float16` | FP16 CATLASS | fp32 row max/sum | FP16 V，fp32 临时累加，FP16 写回 |
| `bfloat16` | BF16 CATLASS | fp32 row max/sum | V 在同一 STA kernel 内暂存为 half，fp32 临时累加，BF16 写回 |

BF16 的 V staging 是本 kernel 内部表示转换，不发射独立 Cast kernel，也不调用其他算子实现。

### 支持形状

- 非空计算要求 `B > 0`、`1 <= N <= 128`、`D > 0` 且 `D % 8 == 0`。
- `q/k/v/output` 的 `[B,N,S,D]` 必须一致。
- tile 固定为 `[6,8,8]`，每个 tile 含 384 个 query token。
- `windowSize` 为 3 个广播值或 `3*N` 个逐 head 值；当前源码要求每个值为正奇数。
- 当前 8 月测试集 `public_v6` 含 95 个合同样本，覆盖 FP16/BF16、`B=1..16` 的离散边界、`N=1..128` 的离散边界、`D=8..512` 的 8 对齐取值、三种画布、text 边界及广播/逐 head window。该 case 集是验证样本，不是生产 dispatch 表。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

调用链如下：

```text
aclnnSlidingTileAttentionGetWorkspaceSize
  -> 参数、shape、dtype、window 和 sequence 合同校验
  -> 空 Tensor 快返
  -> q/k/v Contiguous
  -> window 规范化为 3*N
  -> L0 SlidingTileAttention
  -> SlidingTileAttentionFai tiling
  -> solution-owned MIX_AIC_1_2 kernel
```

Host 从 runtime attrs 和 input desc 填充 `SlidingTileAttentionMeta`，并再次校验 BNSD 元素数、dtype、画布、text 和 window。所有 `B*N*S*D`、workspace element/byte 计算均使用带溢出检查的 64 位乘加；序列、维度或容量不合法时返回失败。

分核基本单位是 `(batch, head, query work)`：

- 常规路径按每 128 个 query row 形成一个 job；text 尾块使用实际 `qRows`。
- grouped/compact 路径将一个 384-row tile 作为 host job，kernel 内再处理 3 个 128-row chunk。
- `blockDim = min(AIC core 数, totalJobs)`，每个 core 以 stride=`blockDim` 领取后续 job，输出所有权互不重叠。

Host workspace 包含每 active core 的 QK/P/OTmp ring、online-softmax row max/sum/dm 状态、非 16 对齐或大 D 的 Q/K staging，以及 BF16 V staging。基础 ring 为 5 个 stage；D 输出列向 16 对齐，QK 的 D 累加以 128 为 slice 上限。系统 workspace 和用户 workspace 合并前均执行溢出检查。

Host tiling key 只选择两种编译变体：

| runtime 条件 | tiling key | 编译变体 | 失败行为 |
| --- | ---: | --- | --- |
| `dtype=BF16 && D=64` | 1 | BF16 D64 lean | metadata 不合法时 host 失败 |
| 其他合法 FP16/BF16、D | 0 | general | metadata 不合法时 host 失败 |

key 0/1 内部继续按同一份 runtime metadata 选择数据流：

| 条件 | 内部路径 | 目的 |
| --- | --- | --- |
| 已实现固定画布、`N=8,D=64` | fixed 384-row decode | 减少通用 task 解码开销 |
| FP16、无 text、`D=8` | grouped QK | 三个 query chunk 共享 tile 级调度 |
| FP16、无 text、`64<D<=128`、非全窗口 | grouped QK + dynamic PV | 复用 K/V stack 并保留逐 head window 正确性 |
| FP16、无 text、`64<D<=128`、全窗口 | compact full-window | 使用紧凑 fp32 score 布局 |
| 其余合法 metadata | dynamic general | 按 query chunk、KV segment 和 D slice 泛化执行 |

host 解析出的 sibling 条件构成互斥路由；selector 顺序不是所有权依据。逐 head window、shape 和 dtype 一旦与变体条件不一致，走同一算子拥有的 general 路径或直接报错，不调用外部实现。

#### 3.2.2 kernel侧设计

Kernel 类型为 `MIX_AIC_1_2`，一个 AIC 与配对 AIV 协同完成以下流程：

1. 将 query task 解码为 `batch/head/qStart/qRows`；固定快路径用已知 384-row tile 关系减少除法，通用路径按 runtime N、画布和 text chunk 解码。
2. image query 先由 token 位置得到 tile id 和 `(t,h,w)`，再以逐 head window 计算 clamp 后的 `[tStart,tEnd) x [hStart,hEnd) x [wStart,wEnd)`。
3. 同一 `(t,h)` 下的连续 width tile 合并成一个 KV segment；有 text 时追加 text segment。text query 直接生成一个完整序列 segment。
4. AIC 用 CATLASS 分块计算 QK。`D>128` 时多个 QK slice 以 fp32 累加到同一 score tile。
5. AIV 对每个 KV stack 执行 online softmax，维护跨 segment 的 row max、row sum 和重标定系数，并把概率以 half 形式供 PV 使用。
6. AIC 计算 P@V；AIV 根据前后 stack 的 max/sum 对历史输出执行 rescale，最后除以全局 row sum 并转换为输出 dtype。

每个 query chunk 最多 128 行，KV stack 基础块为 512 行。D 输出按列块处理；`D=64` 是一个列块，较大 D 由多个列块覆盖。尾部 `qRows`、D 列和 KV segment 均携带实际有效长度，`DataCopyPad`/有效 lane 只读写真实范围。

同步协议按生产者和消费者区分：

- `qkReady`：AIC 发布 QK score，AIV 消费；
- `softmaxReady`：AIV 发布概率，AIC 执行 PV；
- `pvReady`：AIC 发布 PV 临时结果，AIV rescale/写回；
- `qStageFree/kvStageFree`：大 D staging buffer 复用；
- dynamic task done flag：大 D job 的 AIV 写回完成后，AIC 才复用该 job 的 workspace。

event id 使用固定命名常量，进入路径时初始化、退出时成对等待。每个 active core 独占 ring workspace 和输出 query chunk；跨 core 不共享可变 softmax 状态。BF16 V staging 在 AIV 完成后使用一次全核同步，使 AIC 只读取已经发布的 half V。

### 优化策略

1. 以局部 tile window 替代全序列 image attention，按连续 width segment 搬运 KV。
2. 将 QK、online softmax、PV 和输出归一化融合为一次外部 kernel launch，避免算子间 GM 往返。
3. 使用 5-stage ring 让 AIC QK/PV 与 AIV softmax/rescale 通过 FFTS flag 流水协同。
4. 对固定 384-row、grouped QK、全窗口和 BF16 D64 使用 metadata 守卫的专用路径；未命中时保留完整 general 路径。
5. online softmax 只保存每行 max/sum/dm，不物化完整全序列概率矩阵。
6. D 维分块和 Q/K staging 支持 8 对齐的较大 head dimension，同时限制 UB/L1 和单核临时空间。

## 支持硬件

| 支持的芯片版本 | 编译 SoC | 涉及勾选 |
| --- | --- | --- |
| Atlas A2 | `ascend910b` | √ |
| Atlas A3 | `ascend910_93` | √ |

OpDef、binary config、CMake compute unit 和 package 路径必须保持相同 SoC 集合。

## 算子约束限制

1. 当前源码只支持 rank-4 BNSD、FP16/BF16，且 q/k/v/output shape 与 dtype 必须一致。
2. 非空输入要求 `1<=N<=128`、`D` 为正的 8 倍数；三维 tile 固定为 `[6,8,8]`。
3. 当前源码只接受三种已实现画布/sequence metadata，并要求 window 为 3 或 `3*N` 个正奇数。
4. shape 一致的空 Tensor 由 ACLNN L2 快返；其他不支持 metadata 明确失败。
5. 生产路径不读取 case id、测试文件、输入值分布或 timing 特征，不依赖 Python、CPU、PyTorch、FastVideo、Triton/CUDA、vendor whole-op、peer 或其他 backend fallback。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 严格执行下述任务书 3.2 的 dtype 阈值、匹配比例、最大误差、shape 和非有限值门禁 | 社区任务书 3.2 |
| 性能背景 | 任务书给出相对 A100 的性能条款并要求排除 GPU warmup；该条款仅作为任务背景，本轮文档更新不声明性能达标，也不把性能作为语义/shape/精度对齐的完成条件 | 社区任务书 |
| 泛化标准 | 当前完整 testCase 与隐藏合法输入均不得按 case identity 分派 | 社区任务书与当前测试集 |
| fallback 标准 | 每个合法输入均由本目录 Ascend C 实现执行；不支持输入明确失败 | 当前实现边界 |

### 任务书 3.2 精度判定

| dtype | `rtol` | `atol` | `required_matched_ratio` | 最大误差门禁 |
| --- | ---: | ---: | ---: | --- |
| FP16 | `2^-9 = 0.001953125` | `2^-9 = 0.001953125` | `>= 0.99` | `max_abs_error <= 0.1` 或 `max_ulp_error <= 32` |
| BF16 | `2^-6 = 0.015625` | `2^-6 = 0.015625` | `>= 0.99` | `max_abs_error <= 1.0` 或 `max_ulp_error <= 32` |

判定顺序如下：

1. candidate 与 reference 的 rank、各维 shape 和元素数必须完全一致；shape 不一致直接失败，不进入容差统计。
2. candidate 或 reference 中出现 NaN、正负 Inf 等非有限值时直接失败，不能通过 matched ratio 或最大误差规则豁免。
3. 对每个有限元素按 `abs(candidate-reference) <= atol + rtol * abs(reference)` 统计 matched ratio；FP16 和 BF16 都要求 `required_matched_ratio >= 0.99`。
4. 在匹配比例门禁之外，FP16 还必须满足 `max_abs_error <= 0.1` 或 `max_ulp_error <= 32`；BF16 必须满足 `max_abs_error <= 1.0` 或 `max_ulp_error <= 32`。
5. 上述 shape、nonfinite、matched-ratio 和 dtype 最大误差门禁共同组成精度合同；不得用平均误差、部分采样或性能结果替代。

当前验证边界如下：

- 当前 `ops-transformer` 源码树摘要 `b776888815e70cc63a95290aced1af8647f0c6d2acd0f67d3cea1f6416467c14` 与已保存的 A3 reviewer 日志一致；该日志记录了 22 个 case 和空 Tensor 快返检查。
- 该历史日志使用当时 runner 的判定输出，未保存本节任务书 3.2 所需的完整 `required_matched_ratio` 与 ULP 汇总；本轮未重跑，因此不将 22-case 结果升级表述为任务书 3.2 精度通过。
- 当前测试集已扩展为 `public_v6` 95 cases；现有 22-case 源码仓日志不是 95-case 完整通过证据。
- 当前源码证据目录的汇总校验清单仍有日志/CSV 条目未同步，因此不声明证据包完整性通过；设计评审后仍需对最终源码重新执行完整测试与校验。

本轮设计文档更新只确认语义、shape 与上述精度合同文本对齐，不包含新的设备测试或性能完成结论。

## 兼容性分析

算子通过新的 ACLNN 自定义实现提供 FastVideo SlidingTileAttention 语义，不替换或调用 vendor whole-op。A2/A3 共用同一接口与 tiling ABI，差异仅由各自编译 SoC 产物承载。非连续 Tensor 由 ACLNN executor 的设备侧连续化节点处理；空 Tensor 走无 kernel 快返；其他不满足合同的输入返回参数错误，因此不存在静默兼容或 fallback 路径。

# BatchedMatmulSoftmax 算子设计文档

任务来源：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/BatchedMatmulSoftmax_task_doc.md>  
目标开源仓：<https://gitcode.com/cann/catlass>  
适配硬件：Ascend 950PR / Ascend 950DT  
开发语言：Ascend C / C++ / CATLASS TLA

# 需求背景（required）

## 需求来源

本设计面向 2026 年 CANN 社区任务 `BatchedMatmulSoftmax`，目标是在 Ascend 950 上以单次 kernel launch 完成 Batched Matmul 与行级 Softmax 融合计算。

输入输出约定如下：

| 参数 | 含义 | 数据类型 | Layout | Shape |
| --- | --- | --- | --- | --- |
| A | batch 内左矩阵 | FP16 | RowMajor, ND | `(batch, M, K)` |
| B | batch 内右矩阵 | FP16 | ColumnMajor, ND | `(batch, K, N)` |
| S | Softmax 输出 | FP16 | RowMajor, ND | `(batch, M, N)` |

每个 batch 独立计算：

~~~text
Logits_b = A_b x B_b
S_b[m, n] = exp(Logits_b[m, n] - max(Logits_b[m, :])) /
            sum_j exp(Logits_b[m, j] - max(Logits_b[m, :]))
~~~

Matmul 中间结果只在融合算子内部消费，不作为外部 Tensor 暴露。

## 背景介绍

`Batched Matmul + row Softmax` 是 Attention 中 `QK^T -> Softmax` 的核心计算模式。使用 `torch.bmm + torch.softmax` 拼接时，Matmul 结果需要完整写回 GM，再由 Softmax 重新读取，同时产生额外的 kernel launch 和中间 Tensor 管理开销。

本设计采用 AIC/AIV 协同的融合结构：

- AIC 负责 A/B 数据搬运、MMAD、FP32 累加和 logits tile 生产。
- AIV 负责行最大值、指数、求和、归一化、类型转换和结果写回。
- 片上 family 通过 handoff stage 交接 logits；通用 workspace family 使用算子内部 FP32 GM 区域。两者都不暴露 Matmul 中间 Tensor，也不产生第二次 kernel launch。
- 统一 Host 入口根据 N/K、`batch x M` 工作量、tile 数、核团队宽度和片上资源选择预实例化 kernel family。大部分 shape 在 family 内复用同一维度驱动 planner；少量性能关键结构使用编译期精确 shape 专用化。路由不依赖 case ID 或环境变量。

## 复用基础

| 能力 | 参考方向 | 本设计复用方式 |
| --- | --- | --- |
| Ascend 950 构建框架 | Catlass 950 系列工程 | 使用 `CATLASS_ARCH=3510` 构建目标 |
| Batched Matmul | Ascend950 batched matmul | 复用 batch stride、layout 和 GEMM TLA 组装方式 |
| AIC 计算组件 | TileCopy、BlockMmad、Fixpipe | 作为 Matmul producer 的基础组件 |
| AIC/AIV 协同 | Ascend950 mix kernel | 复用 event、stage 和核间数据交接机制 |
| Row-wise Softmax | FlashAttention、Softmax epilogue | 复用稳定 max/sum/exp/normalize 计算思路 |
| 测试框架 | `tests/optest`、AscendOpTest | 构建精度、功能和性能验证入口 |

# 需求分析（required）

## 需求描述

使用 Ascend C / C++ / CATLASS TLA 实现 `BatchedMatmulSoftmax` 融合算子，满足以下目标：

1. 支持任务书约定的 FP16、shape 和 layout。
2. batch 之间独立执行 Matmul 与 Softmax。
3. 单次 kernel launch 完成全部计算。
4. 使用 max 归一化保证 Softmax 数值稳定性。
5. Matmul 使用 FP32 accumulator；跨 N tile/shard 的全局 Softmax 统计量和归一化 scale 使用 FP32，局部 tile 统计精度由已验证策略确定。
6. 根据工作集和片上资源选择合适的 Matmul 与 Softmax 组织方式。
7. 以任务测试集的 `torch.bmm + torch.softmax` 为基线，整体性能达到 1.2 倍以上。
8. 满足生态算子开源精度标准，并提供可复现的 optest 验证入口。

## 需求拆解

| 编号 | 子需求 | 设计响应 |
| --- | --- | --- |
| F1 | batch 独立 | task 映射携带 `batchIdx`，A/B/S 使用独立 batch stride |
| F2 | 单 kernel 融合 | mix kernel 内由 AIC 生产 logits、AIV 完成 Softmax |
| F3 | 固定数据契约 | Host 侧校验 dtype、rank、layout、shape 和连续存储 |
| F4 | FP16 输入输出 | MMAD 使用 FP32 accumulator；全局统计使用 FP32，局部向量计算按策略使用 FP16 或 FP32 |
| F5 | 数值稳定 | 对完整逻辑行执行 max-subtract-exp-sum-normalize |
| F6 | 多工作集适配 | 统一入口按维度和工作量选择通用 workspace、resident-B、segmented/cohort、persistent long-row 或 N-shard reduce-scatter family |
| F7 | 尾块处理 | 通过 `validRows`、`validCols` 和 actual K 处理非整 tile |
| F8 | 中间结果内部消费 | logits 仅保存在片上 stage、UB row-state 或算子内部 workspace，不作为外部 Tensor 暴露 |
| F9 | 可测可维护 | 统一接口、统一 Softmax 状态契约，并补充 optest 交付件 |

## 范围边界

本设计覆盖：

- A/B/S 均为 FP16、连续 ND 物理存储。
- Shape rank 固定为 3D：`(batch, M, K)`、`(batch, K, N)`、`(batch, M, N)`。
- Softmax 沿 N 维执行 row-wise normalization。
- M/N/K 非 tile 对齐时的尾块处理。
- Ascend 950PR / Ascend 950DT 上的功能、精度和性能验证。

本设计不包含：

- 非连续 Tensor。
- mask、dropout、causal、bias、scale 等 Attention 附加能力。
- Matmul 中间结果外部输出。
- BF16、FP32 输出等额外 dtype。
- 与 `P x V` 继续融合成完整 Attention。

# 详细设计（required）

## 算子分析

### 数学公式

对每个 batch 独立计算：

~~~text
Logits[b, m, n] = sum_k A[b, m, k] * B[b, k, n]
Max[b, m] = max_n Logits[b, m, n]
Sum[b, m] = sum_n exp(Logits[b, m, n] - Max[b, m])
S[b, m, n] = exp(Logits[b, m, n] - Max[b, m]) / Sum[b, m]
~~~

Matmul 在 FP32 accumulator 中完成 K 维累加。跨 N tile/shard 合并所需的 global max、global sum 和归一化 scale 使用 FP32。局部 tile 的 max/exp/sum 可以在 FP32 上完成，也可以在已通过精度门禁的专用路径中先将 logits 转为 FP16 后完成，再把局部统计提升为 FP32 参与全行合并；最终结果统一为 FP16。

### 支持数据类型

| 数据 | 类型 | 说明 |
| --- | --- | --- |
| A | FP16 | Matmul 左输入 |
| B | FP16 | Matmul 右输入 |
| S | FP16 | Softmax 输出 |
| Matmul accumulator | FP32 | K 维累加 |
| Softmax statistics | FP16 / FP32 | 局部 tile 统计由策略确定；跨 tile/shard 的全局统计和 scale 使用 FP32 |
| Softmax 临时值 | FP16 或 FP32 | 由精度与片上资源共同决定 |

### 支持形状

| 参数 | Shape | 基本约束 |
| --- | --- | --- |
| A | `(batch, M, K)` | `batch > 0, M > 0, K > 0` |
| B | `(batch, K, N)` | `batch > 0, K > 0, N > 0` |
| S | `(batch, M, N)` | 与 A/B 推导结果一致 |

Host 侧先按 N/K 工作集进入对应 kernel family，再由 family planner 根据 `batch x M`、tile 数、核团队宽度、对齐关系和资源预算生成任务图。多数 shape 共用维度驱动 planner；少量热点结构使用编译期固定的 batch/M/N/K 实例。合法但不满足专用路径约束的 shape 由通用 workspace family 通过 masked copy 和有效区间完成，不改变公开数学语义。

## 算子实现

### 总体架构

算子由 Host 参数准备与 family dispatch、family 内任务图、AIC Matmul producer、AIC/AIV handoff 和 AIV Softmax consumer 组成：

~~~text
Host contract check
          |
          v
N/K family dispatch -> workload/resource planner
                    |
                    v
       one selected mix kernel
       +------------+-------------+
       |                          |
       v                          v
 AIC Matmul producer        AIV Softmax consumer
 TileCopy -> MMAD           row max / exp / row sum
 -> Fixpipe                 -> normalize -> FP16 S
       |                          ^
       +-- handoff / workspace ---+
~~~

一次执行只启动一个融合 kernel。Host 侧先按维度进入预实例化 family，再由共享 planner 或编译期固定任务图确定核数、团队宽度、M/N 分片和流水深度；Device 侧只执行已选配置。精确专用化使用 batch/M/N/K 结构条件，不使用测试 case ID。

### 模块拆分

核心实现按职责拆分：

| 模块 | 职责 |
| --- | --- |
| Host entry | 参数校验、layout/stride 构造、family 选择、workspace 计算和单次 kernel launch |
| Family planner | 根据 N/K、`batch x M`、tile 数和核团队宽度生成任务图；必要时选择编译期精确实例 |
| Matmul producer | A/B TileCopy、K 分段、MMAD、accumulator 和 Fixpipe |
| Handoff / row-state | 管理 logits stage、FP16 exp row-state、event 和 AIC/AIV 可见性 |
| Team reduction | 对 split-N 或 N-shard 路径合并 local max/sum，并向各 destination 发布 scale |
| Output / mailbox | 写回最终 FP16 S；部分团队路径临时借用输出行发布 epoch、stats 和 scale，并在退出前恢复 |
| General workspace | 为未进入片上专用 family 的合法 shape 保存 FP32 logits，供同一 kernel 内 AIV 完成全行 Softmax |

实现按 kernel family 组织如下：

~~~text
include/catlass/gemm/kernel/
  batched_matmul_softmax_tla.hpp
  batched_matmul_softmax_direct_tla.hpp
  batched_matmul_softmax_n128_grouped_resident_tla.hpp
  batched_matmul_softmax_n512_pipeline_tla.hpp
  batched_matmul_softmax_n512_k512_segmented_tla.hpp
  batched_matmul_softmax_n512_k1024_packed_only_tla.hpp
  batched_matmul_softmax_n512_k1024_cohort_tla.hpp
  batched_matmul_softmax_long_row_persistent_tla.hpp
  batched_matmul_softmax_m128_k1024_nshard_tla.hpp
  batched_matmul_softmax_m128_local_npair_tla.hpp
  batched_matmul_softmax_m256_nshard_reduce_scatter_tla.hpp
~~~

通用 workspace family 复用标准 BlockMmad 与 Fixpipe BlockEpilogue。BMS 专用 Softmax、row-state 和 mailbox 生命周期保留在对应 kernel family 内，避免把不同团队拓扑强行塞入单一 epilogue 模板。Catlass example 负责调用和参数展示，`tests/optest` 负责正式精度与性能验证。

### Tiling 设计

Matmul 使用参数化的 `TileM x TileN x TileK` 配置：

- `TileK` 决定一次载入和 MMAD 的 K 分段宽度，全部 K segment 在同一 FP32 accumulator 中累加。
- `TileN` 同时影响 B 工作集、Fixpipe 输出粒度和 AIV 单次处理的行宽。
- `TileM` 影响 A 工作集、L0C accumulator 数量、AIC/AIV handoff 粒度和尾 wave 利用率。
- M/N/K 尾块分别通过 `validRows`、`validCols` 和 actual K 控制，不读取或写入无效区域。

`TileM` 不固定为 16。长行场景可使用较小的 M micro-tile 降低单个 row group 的状态量并改善尾块调度；当较小 TileM 导致 L1/L0 利用不足或搬运次数增加时，应选择更大的 TileM 配置。最终选择同时受片上容量、并行波数、搬运量和 AIC/AIV 负载影响。

### 执行策略

各 family 共享公开数据契约、FP32 MMAD accumulator 和稳定全行 Softmax 语义，但任务图、状态保存和同步范围不同：

| Kernel family | 适用特征 | 调度与状态设计 |
| --- | --- | --- |
| 通用 workspace | 任意合法 shape、非主 tile 对齐或未命中片上专用条件 | AIC 将 FP32 logits 写入算子内部 workspace，AIV 在同一 kernel 内按完整行归一化 |
| Direct / resident-B | N 较短或 B 工作集可驻留，任务能够直接覆盖完整逻辑行 | 以 M tile/group 为任务，B 在连续 M 任务间复用，logits 经片上 handoff 交给 AIV |
| N512 segmented / cohort | N512 且 K/M 工作集需要更宽物理 parent 或分段流水 | 使用 packed M16、M64/M128 或 M96 cohort 等有限实例，共享维度 planner 并按资源选择 |
| Persistent long-row | N1024/N2048，需要 split-N 或 paired/full-row | 按 row group 常驻处理，K segment、handoff ring、queue depth 和 wave buffer 由 family planner 决定 |
| N-shard reduce-scatter | N1024/N2048 且适合固定 W4/W8 团队 | lane 常驻 N256 shard，row owner 汇总跨 shard stats，再把 destination scale 分发回各 lane |

大部分 shape 先由 N/K 进入共享 family，再根据 `batch x M`、任务波数和资源预算生成 groups/core/team/task map。只有共享 planner 无法表达且性能收益经过验证的热点结构，才使用编译期精确 shape 实例；这些实例与共享 family 仍使用相同公开接口和数值门禁。

### 通用 workspace family

通用路径保证完整 shape 覆盖：

~~~text
A/B GM -> L1/L0 -> MMAD(FP32) -> Fixpipe -> internal FP32 workspace
internal workspace -> AIV full-row stable Softmax -> FP16 S
~~~

它仍由一次 mix-kernel launch 完成，workspace 由算子内部申请，不对外暴露。该 family 负责非整 tile、较小工作量以及不满足片上专用约束的合法 shape，并通过 `validRows`、`validCols` 和 actual K 处理尾块。

### Direct、resident-B 与 N512 segmented/cohort family

当 N 较短或 B panel 可在 L1 中保持时，AIC/AIV 使用片上 stage 直接消费 logits：

~~~text
load or retain B tile
for each assigned M tile/group:
  load or roll A tile
  K segments -> MMAD(FP32) -> Fixpipe
  handoff -> local/full-row Softmax -> S
~~~

N512 根据 K segment 数、M 工作量和物理 accumulator parent 选择 packed、segmented 或 cohort 实例。共享 planner 决定 group 数和核数，少量已验证结构可以固定 parent、cohort 或任务映射。B 驻留、A rolling、handoff 深度和 output bank 数均受 L1/L0/UB 静态预算约束。

### Persistent long-row 与 N-shard reduce-scatter family

Persistent long-row family 对 N1024/N2048 采用 split-N 或 paired/full-row 任务。AIC 按 K segment 连续生产 N tile，AIV 将 FP16 exp row-state 保留在 UB，待完整逻辑行的统计量到齐后完成归一化。split-N 使用有限范围 stats mailbox 或同步波次，paired/full-row 则在本任务内完成全行合并。

N-shard reduce-scatter family 进一步把任务组织为固定团队：

~~~text
W4: lane0..3 each owns one N256 shard of an N1024 row cluster
W8: lane0..7 each owns one N256 shard of an N2048 row cluster

each lane:
  compute its shard -> local max/sum -> publish epoch + stats
row owner:
  gather one row slice from all lanes -> global max/sum -> destination scales
each lane:
  gather its scales -> normalize UB row-state -> write FP16 S
~~~

M 方向以固定 M64/M128 accumulator parent 组成 M128/M256 cluster；团队数和 cluster 分配由 workload planner 或编译期任务图确定。N2048 可使用主 W8 团队加 W4 tail 团队处理尾 cluster。无论采用哪种拓扑，row max 和 row sum 必须覆盖完整逻辑 N，禁止对每个 shard 独立归一化。

### 多 N tile Softmax

第 `j` 个 N tile 先计算本地统计量：

~~~text
localMax_j[m] = max_n logits_j[m, n]
localExp_j[m, n] = exp(logits_j[m, n] - localMax_j[m])
localSum_j[m] = sum_n localExp_j[m, n]
~~~

随后使用稳定公式更新全局状态：

~~~text
newGlobalMax[m] = max(globalMax[m], localMax_j[m])
newGlobalSum[m] =
    globalSum[m] * exp(globalMax[m] - newGlobalMax[m]) +
    localSum_j[m] * exp(localMax_j[m] - newGlobalMax[m])
~~~

全部 N tile 完成后执行 final normalize：

~~~text
scale_j[m] = exp(localMax_j[m] - globalMax[m]) / globalSum[m]
S_j[m, n] = cast_fp16(persistedLocalExp_j[m, n] * scale_j[m])
~~~

`localExp_j` 和对应 local stats 必须活到 final normalize。AIC/AIV handoff ring 只承担短生命周期流水交接，stage 会被后续 tile 或 parent 复用，因此不能作为持久 row-state。

Persistent long-row 和 N-shard family 将 FP16 `localExp_j` 保存在 AIV UB row-state 中，将跨 tile/shard 合并所需的 stats 和 scale 保存在 FP32 UB planes。每个 N-shard lane 只保存自己负责的 N256 数据，row owner 汇总完整逻辑行后向各 destination lane 发布 scale。部分团队路径会临时借用尚未最终写回的 S 行作为 epoch、stats 和 scale mailbox；对应最终输出先在 UB deferred rows 中保留，确认所有参与 lane 完成读取后再覆盖 mailbox 行。

通用 workspace family 不复用 S 保存 localExp，而是保存完整 FP32 logits，随后在同一 kernel 内由 AIV 执行全行 Softmax。具体物理存储由已选 family 决定，不改变公开输出语义。

### AIC/AIV handoff 与同步

handoff stage 遵循固定生命周期：

~~~text
free -> AIC writing -> ready -> AIV reading -> free
~~~

同步约束如下：

1. AIC 取得 free token 后才能写入 stage。
2. Fixpipe 和相关搬运完成后，AIC 才发布 ready token。
3. AIV 完整读取当前 tile 后才归还 free token。
4. L0C accumulator 在全部 K segment 完成前不得 Fixpipe 或复用。
5. producer 和 consumer 对 `(batch, rowGroup, nTile, mTile)` 使用一致的逻辑顺序。
6. 尾 group 只减少有效行列数，不提前跳出要求所有参与核到达的同步轮次。
7. N-shard 团队用递增 launch epoch 发布 `statsReady` 和 `scaleDone`，读取方只有观察到本轮 epoch 后才能消费 mailbox。
8. 输出行被借作 mailbox 时，必须先保存其最终值；所有读取者完成后再执行安全回收和最终覆盖。
9. 同步范围优先限制在 team 或当前 wave；仅需要安全回收共享输出行的路径在末尾执行一次参与核 barrier。
10. kernel 返回前 drain 所有 handoff、mailbox 读取和异步输出 stage。

### Host 侧设计

Host 侧负责：

1. 校验 A/B/S 的 dtype、rank、layout、shape、连续性和空维。
2. 构造 A/B/S layout 与 batch stride。
3. 先按 N/K 选择 kernel family，再按 `batch x M`、tile 数、核团队宽度和资源约束生成任务图。
4. 对少量性能关键结构选择编译期精确 shape 实例；其余 shape 复用 family planner。
5. 根据所选 family 计算内部 workspace 大小并完成单次 kernel launch。
6. 将 tile、有效范围、任务映射、launch epoch 和可选 workspace 信息传入 Device。

参数结构示意：

~~~cpp
struct Arguments {
    uint32_t batchCount;
    GemmCoord problemShape;  // (M, N, K)
    GM_ADDR ptrA;
    LayoutA layoutA;
    GM_ADDR ptrB;
    LayoutB layoutB;
    GM_ADDR ptrS;
    LayoutS layoutS;
    GM_ADDR internalWorkspace;  // only used by workspace-backed family
};
~~~

layout 和 batch stride：

~~~text
layoutA = RowMajor(M, K), strideA = M * K
layoutB = ColumnMajor(K, N), strideB = K * N
layoutS = RowMajor(M, N), strideS = M * N
~~~

### Workspace 与片上存储

| 存储 | 生命周期 | 用途 |
| --- | --- | --- |
| L0C accumulator | 单个 Matmul tile | FP32 K 维累加 |
| Handoff ring | 单个流水 stage | AIC 向 AIV 传递 logits tile |
| UB row-state | 单个 row group / cluster | 保存各 N tile/shard 的 FP16 exp state |
| FP32 stats planes | 单个 row group / cluster | 保存跨 tile/shard 合并所需的 max、sum 和 destination scale |
| S mailbox rows | 当前团队任务完成前 | 临时发布 launch epoch、local stats 和 scale，最终安全恢复为输出 |
| FP32 GM workspace | 通用 workspace family 的一次调用 | 保存完整内部 logits，供同一 kernel 内的 AIV 全行归一化 |

Workspace 由所选 family 决定，而不是公开接口的一项固定要求。有的性能关键 shape 通过编译期精确结构路由；大部分 shape 则先进入统一 N/K family，再由维度与工作量 planner 选择 group、core、team、parent 和流水深度。使用片上 handoff、UB row-state 或团队 mailbox 的 family 返回 `workspace=0`；通用 workspace family 为任意合法 shape 分配 `batch x M x N x sizeof(float)` 的内部空间。两种方式都保持同一个 Torch 接口和单次 kernel launch。

### 资源约束

| 资源 | 设计约束 |
| --- | --- |
| L1 | A/B tile、常驻数据和双缓冲总量不得超过容量；复用范围由工作集决定 |
| L0A/L0B | bank 数与 TileCopy/MMAD 物理布局一致，已占用 bank 不得被预取覆盖 |
| L0C | 每个活动 M tile 独占 accumulator，深度受静态容量约束 |
| UB | handoff、row state、临时向量和输出 stage 总量必须满足容量要求 |
| GM workspace | 仅由通用 workspace family 使用；大小按完整 FP32 logits 计算，且不对外暴露 |

所有模板实例在编译期检查静态存储上限，Host 侧在 launch 前检查 shape、对齐和动态 workspace 条件。

### 数值稳定性设计

1. MMAD 使用 FP32 accumulator。
2. Softmax 始终执行 max 归一化后再计算 exp。
3. 通用 workspace 和高精度局部路径在 FP32 上计算 local stats；经精度验证的吞吐路径允许先把 logits 转为 FP16 后完成 local max/exp/sum。
4. 所有跨 tile/shard 的 global max、修正 denominator 和 destination scale 使用 FP32。
5. 多 N tile 使用稳定 max/sum merge 公式；不同 lane 不得独立归一化后直接拼接。
6. final normalize 的存储顺序和舍入方式由 family 固定，最终输出为 FP16。
7. 对极值、重复最大值、全零输入、大 K 累加、FP16 subnormal 和不同局部精度路径进行专项验证。

### 编译与调用

Ascend 950 构建目标使用：

~~~bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -DCATLASS_ARCH=3510 <batched_matmul_softmax_target>
~~~

算子通过 Catlass example 提供独立调用入口，并在 `tests/optest` 中注册对应测试实现。具体 example 编号和构建 target 以合入时 Catlass 主线目录为准。

## 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | --- |
| Ascend 950PR | 支持 |
| Ascend 950DT | 支持 |

## 算子约束限制

| 约束项 | 设计策略 |
| --- | --- |
| dtype | A/B/S 固定 FP16 |
| layout | A RowMajor，B ColumnMajor，S RowMajor |
| shape rank | 固定 3D batched matmul |
| 非连续 Tensor | 不支持 |
| 空维 | 不支持 |
| mask/scale/dropout/causal | 不支持 |
| 中间 Matmul 输出 | 不对外暴露 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足生态算子开源精度标准；输出无 `NaN/Inf`；每行 Softmax sum 接近 1 | 任务书与算子数学定义 |
| 性能标准 | 在任务测试集相同 shape 下达到 `torch.bmm + torch.softmax` 拼接方案的 1.2 倍以上 | 社区任务要求 |

CPU golden 使用高精度 Matmul 与稳定 Softmax，最终转换为 FP16：

~~~python
matmul = np.matmul(A.astype(np.float64), B.astype(np.float64))
shifted = matmul - np.max(matmul, axis=-1, keepdims=True)
softmax = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
golden = softmax.astype(np.float16)
~~~

精度报告记录：

1. 最大绝对误差、最大相对误差、MERE 和 MARE。
2. 失败元素数、失败 shape 和误差位置。
3. 每行输出和与 1 的偏差。
4. `NaN/Inf` 检查。
5. FP16 量化边界与高精度 reference 的差异。

性能验证要求：

1. 融合算子与基线使用相同 shape、输入、预热次数、重复次数和统计口径。
2. 以 NPU Event 做快速回归，以 `msprof op` 采集正式耗时和 AIC/AIV 指标。
3. 记录 `speedup = baseline_elapsed / fused_elapsed`。
4. 关注 AIC/AIV lifetime、MTE2/MTE3、Cube、Fixpipe、vector 和同步开销。
5. 以任务测试集整体结果判断性能，不用单一最优点代替整体门槛。

## 功能测试矩阵

| 类别 | 覆盖内容 | 验证目的 |
| --- | --- | --- |
| 通用 workspace | 小 batch、非整 tile 和未命中片上专用条件的合法 shape | 验证完整 FP32 logits workspace 与全行 Softmax |
| Batch 独立性 | 多 batch 且各 batch 输入不同 | 验证 batch stride 和任务映射 |
| Resident / segmented | 短 N、连续 M task、不同 K segment 数 | 验证 B 复用、cohort parent 和 AIC/AIV 流水 |
| Persistent long-row | N1024/N2048、split-N 与 paired/full-row | 验证 UB row-state、stats 合并和 final normalize |
| N-shard team | W4/W8、主团队加 tail 团队、不同 cluster 分配 | 验证 row-owner reduce-scatter、epoch 和 destination scale |
| Mailbox 生命周期 | 同一 shape 连续多次 launch、不同 launch epoch | 验证无 stale epoch、提前覆盖、死锁或偶发错误 |
| 尾块 | M/N/K 非 tile 对齐 | 验证 valid range 和 masked copy |
| 数值压力 | 全零、重复最大、极值、大 K 累加 | 验证局部 FP16/FP32 路径和全局 FP32 merge 的稳定性 |
| 路由边界 | 相邻工作量桶、共享 planner 与精确实例边界 | 验证路由唯一、资源合法且数学语义一致 |
| 接口拒绝 | 错误 dtype/layout/rank、空维、非连续输入 | 验证 Host 参数检查 |

基于任务自测试集完成全量精度验证，并补充不少于 200 组随机用例，覆盖不同 batch、M/N/K、尾块和数值边界。

## 可维护性分析

1. Host entry、family planner、Matmul producer、handoff/row-state、team reduction 和 output/mailbox 职责明确。
2. 各 family 共享公开数据契约、FP32 MMAD accumulator 和稳定全行 Softmax 语义；任务图和状态生命周期在 family 内封装。
3. tile、parent、queue depth、team width 和 task map 均为有限编译期实例；新增实例前完成资源、精度和性能验证。
4. 路由条件集中在统一 Host dispatch：多数 shape 使用维度/工作量 planner，少量 shape 使用精确结构条件；不保留环境变量策略覆盖或 case-ID 路由。
5. Host 暴露满足所有候选 family 的安全 task/core 上界，选定 kernel 再收缩到实际团队数，避免 Host/Device 任务覆盖不一致。
6. example、README、optest 和自验证报告使用同一输入输出约定与性能口径。

## 兼容性分析

该算子以独立 kernel、example 和 optest 入口组织，不改变 Catlass 其他 Matmul、Softmax 或 fusion 能力。

兼容性边界如下：

1. 构建入口沿用 Catlass 现有机制，目标架构限定为 Ascend 950PR / Ascend 950DT。
2. 不满足 dtype、layout、rank 和连续性要求时，Host 侧返回不支持。
3. workspace 大小由所选 family 决定且只用于内部中间状态，不改变公开输出语义。
4. 后续扩展 BF16、scale、mask 或完整 Attention 时，通过独立模板参数和配置增加，不影响 FP16 基础路径。

# 风险与应对

| 风险 | 等级 | 影响 | 应对 |
| --- | --- | --- | --- |
| TileM 过小导致片上空间利用不足 | 高 | L1/L0 浪费且搬运次数增加 | TileM 参数化，并结合资源占用、任务波数和搬运量选择配置 |
| 多 N tile row-state 被 handoff stage 覆盖 | 高 | final normalize 读取错误数据 | handoff ring 与 UB FP16 row-state 分离；通用 family 使用独立 FP32 logits workspace |
| AIC/AIV stage 生命周期失配 | 高 | 数据竞争、死锁或错误结果 | 固化 token 顺序、逻辑 tile 坐标和 kernel 退出前 drain 规则 |
| Epoch mailbox 被提前覆盖或读取到旧轮次 | 高 | 偶发精度错误或死锁 | 使用递增 launch epoch、`statsReady/scaleDone` 状态和安全 mailbox reclaim |
| N-shard 全局统计量合并错误 | 高 | 每个 shard 被错误地独立归一化 | row owner 使用 FP32 stable max/sum merge，并验证所有 destination scale |
| 精确路由与共享 planner 重叠 | 高 | 重复覆盖、错误核数或错误实例 | 路由按优先级集中管理，静态验证 shape 唯一命中和 Host task 上界 |
| AIV Softmax 成为关键路径 | 中 | 融合收益不足 | 通过 tile 并行、向量归约和流水重叠平衡 AIC/AIV 工作量 |
| B 常驻收益不足 | 中 | 预载与同步开销超过复用收益 | 根据 N/K、连续 M 任务数和 L1 预算选择 resident 或其他 family |
| FP16 局部 Softmax 精度不足 | 中 | 误差超出验收阈值 | 全局合并保持 FP32；局部 FP16 仅用于通过专项精度门禁的实例，否则选择 FP32 局部路径 |
| 通用 workspace GM 流量较高 | 中 | 合法 shape 正确但性能不足 | 优先让高频工作集进入片上 family，保留 workspace family 作为统一完整覆盖路径 |
| 尾块利用率和边界处理 | 中 | 性能波动或越界 | 使用 valid range、masked copy 和非对齐随机用例验证 |

# 参考资料

- 任务书：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/BatchedMatmulSoftmax_task_doc.md>
- 任务设计模板：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>
- Catlass 仓库：<https://gitcode.com/cann/catlass>
- Catlass 参考方向：Ascend950 batched matmul
- Catlass 参考方向：Ascend950 flash attention / row Softmax
- Catlass 参考方向：Ascend950 matmul epilogue / EVG

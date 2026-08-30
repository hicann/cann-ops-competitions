# SlidingTileAttention 算子设计文档

## 1. 需求背景（required）

### 1.1 需求来源

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 8 月社区任务 - SlidingTileAttention 算子开发（A2） |
| 任务书 | https://www.hiascend.com/activities/task-center/details/3c50a2daa47f426cad2e69b4a7059496 |
| 验收要求 | https://www.hiascend.com/developer/activities/details/11014a50a8794171a4a08688fd398774 |
| 对标接口 | FastVideo.sliding_tile_attention（GitHub: hao-ai-lab/FastVideo） |
| 目标硬件 | Ascend 910B3（A2 系列） |
| 算力环境 | 昇腾社区开放 A2 算力 / 自备服务器（开发容器 CANN 9.1.0） |

### 1.2 背景介绍

SlidingTileAttention（滑动瓦片注意力，STA）是 FastVideo 针对视频生成模型提出的高效注意力机制。视频帧序列被组织为 3D 网格（T×H×W），并进一步切分为固定大小的 tile（6×8×8 = 384 个 token）。每个 head 独立地在 **tile 粒度的 3D 滑动窗口**内做稠密注意力计算，同时 text token 对全部位置做全局注意力。相比全量注意力，STA 将计算复杂度从 O(S²) 降到 O(S·W)（W 为滑窗内的 tile 数量），显著降低视频生成模型（HunyuanVideo、Wan、Stepvideo 等）的计算量与显存占用。

本任务参考 FastVideo 中 `sliding_tile_attention` 的 CUDA 实现（ThunderKittens C++ kernel 及 Triton fallback），在昇腾 A2 上基于 **Ascend C** 编程语言、采用 **aclnn 算子工程模式**实现功能一致的算子，提交至昇腾算子开源仓（ops-transformer/experimental/attention）。

## 2. 需求分析（required）

### 2.1 需求描述

使用 Ascend C 实现 `SlidingTileAttention` 算子，功能与 `FastVideo.sliding_tile_attention` 完全对齐：

- 支持数据类型：`bfloat16`、`float16`；
- 支持数据排布：`BNSD`（q/k/v/output 均为 4 维，shape 完全一致）；
- 实现 tile 级 3D 滑动窗口注意力 + text 全局注意力；
- 支持 `has_text` 场景下的序列 padding（384 对齐）与完成后截断恢复；
- 支持确定性计算（相同输入多次执行结果一致）；
- 采用 aclnn 算子工程化模式开发；
- 满足生态算子开源精度标准与性能标准。

### 2.2 需求拆解

1. 完成 aclnn 算子框架（`GetWorkspaceSize` + 算子执行入口），host 侧完成参数校验、workspace 计算与 tiling 数据生成；
2. 完成 kernel 侧（Cube + Vector 单元）注意力核心计算：QKᵀ → softmax → PV；
3. 支持 bf16 / fp16 两种 dtype 的模板化实现；
4. 实现 tile 级 3D 滑窗掩码（含边界 shifted-window）与 text 全局可见；
5. 支持 `has_text` 时 padding 到 384 对齐、计算后截断；
6. 确定性计算：固定 KV 遍历顺序、单遍 online-softmax、避免原子操作与非确定归约；
7. 精度满足 fp16 rtol/atol = 2⁻⁹、bf16 rtol/atol = 2⁻⁶（matched_ratio ≥ 0.99）；
8. 性能不低于 0.8 × A100（FastVideo GPU 基线）。

## 3. 详细设计（required）

### 3.1 算子分析

#### 3.1.1 接口定义

```c
aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(
    const aclTensor          *q,
    const aclTensor          *k,
    const aclTensor          *v,
    aclTensor                *output,
    const aclIntArray *const *windowSize,   // 每 head 的 (t,h,w) 窗口，或长度为 1 时广播
    uint64_t                  windowSizeLen,
    int64_t                   textLength,
    bool                      hasText,
    const char               *seqShape,      // 形如 "30x48x80"
    uint64_t                 *workspaceSize,
    aclOpExecutor           **executor);

aclnnStatus aclnnSlidingTileAttention(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

#### 3.1.2 参数说明

| 参数 | 输入/输出/属性 | 描述 | dtype | 排布 | 维度 | 约束 |
| --- | --- | --- | --- | --- | --- | --- |
| q | 输入 | query tensor | fp16/bf16 | BNSD | [B,N,S,D] | q/k/v/output shape 完全一致；B,N,S,D>0；S ≥ img_seq_len + text_length；不支持 broadcast；q/k/v dtype 一致 |
| k | 输入 | key tensor | fp16/bf16 | BNSD | [B,N,S,D] | 同上 |
| v | 输入 | value tensor | fp16/bf16 | BNSD | [B,N,S,D] | 同上 |
| output | 输出 | attention 结果 | fp16/bf16 | BNSD | [B,N,S,D] | 独立输出，shape 与 q 一致 |
| window_size | 属性 | 每 head 的 (t,h,w) 奇数窗口（tile 单位）；长度 1 广播到所有 head，长度 N 逐 head 生效 | list_list_int | - | - | 奇数，> 0 |
| text_length | 属性 | text token 数；has_text=false 时按 0 | int64 | - | - | ≥ 0 |
| has_text | 属性 | 是否存在 text token | bool | - | - | true/false |
| seq_shape | 属性 | image token 三维尺寸（如 "30x48x80"） | string | - | - | T,H,W>0，且被 tile 整除 |

#### 3.1.3 序列结构（与 FastVideo 对齐）

输入序列 S 的布局（**image 在前、text 在后**，与 FastVideo 内核一致）：

```
[ image tokens (0 .. img_seq_len-1) | text tokens (img_seq_len .. img_seq_len+text_length-1) | padding ]
```

- `img_seq_len = T × H × W`，由 `seq_shape` 指定；image token 按 3D 网格 (T, H, W) 排列（先 W、再 H、再 T 的 row-major 顺序）。
- `has_text=false` 时 `S = img_seq_len`，`text_length` 按 0 处理，无 padding。
- `has_text=true` 时 `S = img_seq_len + text_length`，为保证与 tiling（tile 大小）对齐，将序列 padding 到 **384 的倍数**（重复末尾 token），计算完成后截断回原始长度 S。

#### 3.1.4 数学公式与算法

**Tile 划分**：将 image 3D 网格 (T, H, W) 以固定 tile 尺寸 `(6, 8, 8)` 切分，

```
num_tiles = (Tt, Th, Tw) = (T/6, H/8, W/8),   tile_size = 6×8×8 = 384
```

**注意力分数**（逐 head，scale = 1/√D）：

```
score(i, j) = (Q_i · K_j) × (1 / √D)
```

**滑窗掩码（image query）**：image query token i 的 tile 坐标为 `q = (qt, qh, qw)`，其窗口半径 `r_d = ⌊w_d / 2⌋`（w_d 为该 head 的 window_size 分量）。采用 **shifted-window**（Swin 语义）保证窗口内 tile 数恒为 w_d：

```
center_d = clamp(q_d, r_d, num_tiles_d - 1 - r_d)
key 可见 ⇔ key 属于 text 段（全局），
          或 tile(key_d) ∈ [center_d - r_d, center_d + r_d]（三轴同时满足）
```

**text query（全局注意力）**：text query token 对所有 key（全部 image + 全部 text）可见。

**输出**：

```
O_i = Σ_{j: Mask(i,j)=1} softmax(score(i,:))_j · V_j
```

#### 3.1.5 参考实现的双阶段结构（与 FastVideo 一致）

FastVideo 的 `sliding_tile_attention` 分两阶段计算：

1. **逐 head 的 image query 阶段**：对每个 head，image query 在其 3D 滑窗内（tile 粒度）+ text KV 上做注意力；
2. **text query 全局阶段**（仅 has_text=true）：text query 对所有 image + text KV 做全局注意力。

本算子沿用相同的两阶段语义，保证输出与 `FastVideo.sliding_tile_attention` 逐元素一致。

#### 3.1.6 支持数据类型

`bfloat16`、`float16`（内部以 float32 高精度累加）。

#### 3.1.7 支持形状

- 张量：`[B, N, S, D]`（BNSD），B、N、S、D 均 > 0；
- `seq_shape` 主要支持三类（与 FastVideo 对齐）：
  - `"30x48x80"`（HunyuanVideo，img_seq_len=115200）
  - `"36x48x48"`（Stepvideo，img_seq_len=82944）
  - `"18x48x80"`（Wan，img_seq_len=69120）
- tile 固定为 `(6,8,8)`，要求 T、H、W 分别能被 6、8、8 整除。

### 3.2 算子实现

#### 3.2.1 整体方案

采用 **FlashAttention 式分块 + 在线 softmax** 的 Cube/Vector 混合实现，在 host 侧完成 tiling 规划，在 kernel 侧以 Cube 单元做矩阵乘、Vector 单元做 softmax 归约。核心计算流程：

```
对每个 (batch, head, query-block)：
    m = -inf, l = 0, acc = 0
    遍历该 query 可见的 key-block（滑窗内 tile 块 + text 块，固定顺序）：
        S_block = Q_block @ K_blockᵀ × (log2(e) / √D)      # Cube 矩阵乘
        S_block[masked] = -inf                              # 滑窗掩码
        m_new = max(m, rowmax(S_block))
        P = exp2(S_block - m_new)                           # 矢量 exp2
        l = l × exp2(m - m_new) + rowsum(P)
        acc = acc × exp2(m - m_new) + P @ V_block           # Cube 矩阵乘
        m = m_new
    O_block = acc / l                                       # 归约归一
    O_block 由 fp32 转为 fp16/bf16 后写出
```

- **滑窗掩码实现**：掩码在 tile 粒度，直接由 query block 的 tile 坐标 + 该 head 的 window_size 推导出「可见 key block 的 tile 区间」，在 host 侧或 kernel 侧做索引区间算术，**无需物化完整 mask 张量**，仅需对每个 key-block 判断其 tile 坐标是否落入窗口。
- **确定性**：key-block 遍历顺序固定（按 tile 索引升序 + 末尾 text 块）；`m/l/acc` 均单遍在线更新，不使用原子累加；exp2 舍入与累加顺序固定，保证相同输入多次执行结果一致。

#### 3.2.2 host 侧设计（tiling 策略）

1. **参数解析与校验**：
   - 读入 q/k/v 的 shape、dtype、排布；校验 q/k/v/output 四者 shape、dtype 一致，BNSD 为 4 维；校验 S ≥ img_seq_len + text_length。
   - 解析 `seq_shape` → T、H、W，计算 `img_seq_len`、`num_tiles=(T/6, H/8, W/8)`。
   - 校验 window_size：长度 1（广播）或 N（逐 head），分量均为正奇数。
   - `has_text` 时计算 `target_S = ceil(S / 384) * 384` 与 `pad_size = target_S - S`。

2. **分核策略**：
   - 以 `(B × N × num_query_blocks)` 为最小调度单元，尽量满核、均匀分配；不能均分时将余量分配到前几个核。
   - 图像阶段：每个 head 的 image query 按 tile 切分（每 tile 384 token，可再按 128 行为子 block）。
   - text 阶段（has_text 时）：text query 以全局注意力方式独立调度。

3. **UB 数据切分**：
   - 依据平台 UB 大小（`GetCoreMemSize`）、是否 double buffer、Cube 计算的临时 buffer 需求，选取 query 行块 `Br` 与 key 行块 `Bc`；
   - 建议初值（后续按性能调优）：`Br = 128`、`Bc = 128`；head_dim D ∈ {32, 64, 128} 时单次 QKᵀ 产出 `128×128` fp32 = 64KB，K、V 各 `128×D`，整体可置于 UB 内并用 double buffer 隐藏搬运。
   - 每个 384-token 的 tile 对应 3 个 128 行子块。

4. **tilingkey 规划**：
   - 依据 `has_text`、`dtype`、D、单 head 窗口是否各不相同等，规划 kernel 侧分支配对的 tilingkey（如 `has_text` 是否生效、text 阶段是否需独立 kernel 等）。

5. **workspace**：本算子分块/在线 softmax 的中间量（m、l、acc）均存放于 UB，无需额外大 workspace，`workspaceSize` 可按 0（或对齐留少量）返回。

   将上述参数（B、N、S、D、img_seq_len、text_length、num_tiles、T/H/W、window_size（逐 head 或广播）、scale、pad_size 等）填充到 TilingData 结构并下传 kernel。

#### 3.2.3 kernel 侧设计

按 Ascend C 标准 `Init` / `Process` 结构组织，`Process` 内含 CopyIn / Compute / CopyOut。

- **Init**：读入 TilingData，计算本核负责的 (batch, head, query-block) 索引，推导 query block 的 tile 坐标与可见 key-block 区间。
- **CopyIn**：按分块将 Q_block、K_block、V_block 从 GM 搬入 UB（DMA），配合 double/multi-buffer 与 Compute 流水。
- **Compute**：
  1. `Q_block @ K_blockᵀ`（Cube，fp32 累加），乘以 `log2(e)/√D`（Vector）；
  2. 对窗口外的 key-block 置 `-inf`（在 tile 区间判断，非窗口块整块跳过/掩码）；
  3. 在线 softmax：`rowmax` → `exp2`（减 max 后）→ `rowsum`，更新 `m`、`l`、`acc`（Vector）；
  4. `P @ V_block`（Cube，fp32 累加）累加到 `acc`。
- **CopyOut**：`acc / l`（Vector 除法，fp32）后转 fp16/bf16，写出到 output 对应位置。
- **text 阶段**（has_text）：text query 走全局注意力分支（不施加滑窗掩码），遍历全部 image tile 块 + text 块。
- **padding 处理**：padding token 通过「重复末尾 token」在 host 前处理；kernel 内 text key 的掩码按 `text_length` 屏蔽 padding 副本，image 阶段计算完成后按原始 S 截断写出。

#### 3.2.4 数据精度设计

- QKᵀ、PV 均以 Cube fp32 累加，输入以 fp16/bf16 参与矩阵乘；
- softmax 中间态（m、l、acc、P）全程 fp32；
- 指数采用 `exp2` + `log2(e)` 缩放（与参考实现一致），保证及 exp 精度并贴合硬件矢量指令；
- 输出由 fp32 最终转回 fp16/bf16。

### 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（Ascend 910B3） | √ |

### 3.4 算子约束限制

1. 输入暂不支持 broadcast；q、k、v、output 必须均为 BNSD 布局且 shape 完全一致；
2. q、k、v 三者 dtype 必须一致（fp16 或 bf16）；
3. `seq_shape` 的 T/H/W 需分别被 6/8/8 整除；
4. `window_size` 分量必须为正奇数；长度为 1 或 N；
5. `has_text=false` 时 `text_length` 按 0 处理；
6. 算子为前向（推理）算子，不实现反向。

## 4. 可维可测分析

### 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 与 FastVideo.sliding_tile_attention 输出一致；fp16 rtol=atol=2⁻⁹(1.95e-3)、bf16 rtol=atol=2⁻⁶(1.56e-2)，matched_ratio ≥ 0.99，且 max_abs_error ≤ 1e-1(或 32×ULP, fp16)、≤ 1e-0(或 32×ULP, bf16) | 生态算子开源精度标准 |
| 性能标准 | fp16/bf16 下 910B3 性能 ≥ 0.8 × [FastVideo (GPU A100) 基线] | 任务性能验收 |

性能自测 case 基线：

| B | H | D | Img_seq | Text_len | Window_size | GPU A100 基线 |
|---|---|---|----|----------|-----------|---------------|
| 1 | 8 | 64 | 30·48·80 | 128 | [(1,1,1),(3,3,3),(3,5,5),(5,5,5),(1,3,3),(3,3,5),(5,5,7),(7,7,7)] | 9 Calls，Total 64.204 ms，Avg 7.134 ms |
| 1 | 8 | 64 | 36·48·48 | None | [(3,3,3)]×H | 8 Calls，Total 11.691 ms，Avg 1.461 ms |

### 4.2 兼容性分析

新算子，不涉及对既有算子的兼容性；遵循 ops-transformer 仓库算子规范提交。

## 5. 测试方案（简）

1. 精度自测：使用 ATK 工具（`node.yaml`/`perf.yaml`/`gen.py`/`exe.py`），输入 q/k/v 为 (0,1) 正态分布、window_size/text_length/seq_shape 按任务书规则覆盖，比对 aclnn 算子输出与 GPU 上 FastVideo.sliding_tile_attention 输出；
2. 性能自测：按上述性能 case 测量 910B3 耗时，与 0.8×A100 基线比较；
3. 交付自测报告（含用例参数、精度对比截图、性能截图）与可复现 README。
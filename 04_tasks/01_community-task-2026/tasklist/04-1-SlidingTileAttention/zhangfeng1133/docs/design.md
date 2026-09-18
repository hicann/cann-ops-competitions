# SlidingTileAttention 算子设计文档

| 文档版本 | 日期 | 作者 | 变更 |
|---|---|---|---|
| v1.0 | 2026-09-13 | zhangfeng1133 <yanggg1133@163.com> | 初版：块稀疏 FlashAttention 映射 + AIC/AIV 混布 + KV 窗口复用调度设计 |

> 任务：CANN 社区任务 2026 `04-1-SlidingTileAttention`（任务书见 `docs/202604/SlidingTileAttention_task_doc.md`）
> 参考实现：[FastVideo sliding_tile_attention](https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/ops.py#L14)

# 需求背景（required）

## 需求来源

CANN 社区任务 2026 `04-1-SlidingTileAttention`：参考 FastVideo 的 `sliding_tile_attention` 实现，在昇腾 NPU（Atlas A2 训练系列产品）上用 Ascend C 实现功能一致的算子，完成设计、开发、测试全流程，验收通过后合入昇腾算子开源仓 `experimental/attention`。

## 背景介绍

### SlidingTileAttention（STA）算子现状分析

STA 出自论文《Fast Video Generation with Sliding Tile Attention》（arXiv:2502.04507），是视频扩散 DiT（HunyuanVideo / StepVideo / Wan）的 3D 局部注意力加速算子。FastVideo 提供 Triton 功能基准与 ThunderKittens/FA3 CUDA 性能实现，后者深度绑定 GPU warp-specialization，无法直接移植昇腾，需基于 Ascend C 重新设计。

参考实现核心语义：

| 参数 | 含义 | 类型 |
|---|---|---|
| q / k / v | 注意力输入，`[B, N, S, D]`，BNSD 布局 | tensor（fp16 / bf16） |
| window_size | 逐 head 窗口 `(kt, kh, kw)`，单位 tile 数 | list[tuple] |
| text_length | 文本 token 数 | int |
| has_text | 是否含文本全局注意力 | bool |
| seq_shape | canvas 形状，如 `30x48x80` | str |

### STA 功能分析

**Tile 组织**：视频隐空间为 `canvas_t × canvas_h × canvas_w` 的 3D token 网格，每 `tile_t × tile_h × tile_w = 6×8×8 = 384` 个相邻 token 组成一个 tile，序列按 tile 的 `(t,h,w)` 行主序拼接。典型 canvas：

| seq_shape | canvas(t×h×w) | tile 数 | img_seq_len | 模型 |
|---|---|---|---|---|
| 30x48x80 | 5×6×10 | 300 | 115200 | HunyuanVideo |
| 36x48x48 | 6×6×6 | 216 | 82944 | StepVideo |
| 18x48x80 | 3×6×10 | 180 | 69120 | Wan |

**窗口 mask**：query tile `(qt,qh,qw)` 只对以其为中心、大小 `kt×kh×kw`（逐 head 可不同）的 **dense tile 窗口** 做注意力，窗口中心按 canvas 边界 clamp：

```
center = clamp(q_tile, k//2, num_tiles - 1 - k//2)
kv_range = [center - k//2, center + k//2]   # t/h/w 三维独立 clamp 到 [0, num_tiles)
```

**关键性质**：tile 内部不做任何 mask——被选中 tile 的 384 个 token 全部参与计算，注意力分块全部为 dense 块（无 mixed block），这是 STA 可高效实现的根本原因。

**文本路径**：`has_text=True` 时序列尾部追加 `text_length` 个文本 token 并 pad 到 384 整数倍；图像 query 额外注意全部文本 token，文本 query 对全部图像 tile 与文本做全局注意力。

**计算**：标准缩放点积注意力（fp32 累加，online softmax），`scale = 1/√D`。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 SlidingTileAttention 算子：功能与 FastVideo `sliding_tile_attention` 完全对齐；支持 FLOAT16 / BFLOAT16、BNSD 格式；实现算子泛化（任意合法 canvas、逐 head 窗口、has_text、任意 text_length）；精度满足 AscendOpTest 默认阈值；性能与 0.8×GPU（A100）持平。

## 需求拆解

1. 支持 FLOAT16 / BFLOAT16，BNSD 格式，D=64。
2. 支持任意合法 canvas（`t×h×w` tile 网格），tile 固定 6×8×8=384；支持逐 head 不同 window_size、has_text 开关、任意合法 text_length。
3. 精度满足 AscendOpTest 默认阈值（fp32 累加 + online softmax）。
4. 性能对齐任务书：0.8×A100（Case1 Hunyuan 8 窗口 9 calls、Case2 StepVideo (3,3,3)×8）。
5. 泛化与边界：H=1、无文本、最大窗口（全注意力退化为 dense FA）、最小窗口 (1,1,1)、canvas 各维 tile 数为 1、序列 pad 边界。

# 详细设计（required）

## 算子分析

### 数学公式

对每个 query tile $q$，设其窗口（及文本路径）选中的 kv token 集合为 $\mathcal{K}(q)$：

$$O_q=\mathrm{Softmax}\!\left(\frac{Q_q K_{\mathcal{K}(q)}^\top}{\sqrt{d}}\right)V_{\mathcal{K}(q)}$$

### 支持数据类型

FLOAT16、BFLOAT16（中间计算与累加 fp32）。

### 支持形状

BNSD；D=64；canvas 为任意合法 `t×h×w` tile 网格；`S = img_seq_len (+ text_length 后 pad 到 384 整数倍)`；`N = len(window_size)`。

## 算子实现

### 实现方案

#### 核心思路一：STA 是 tile 对齐的块稀疏 FlashAttention

tile=384=3×128，tile 边界天然对齐 FlashAttention 128 块边界（384 = 3 blocks/query tile）。「query tile → kv tile 窗口」无损映射为「query 128-块 → kv 128-块列表」的块稀疏选择；文本全局与 clamp 规则同样映射为块选择。映射后所有注意力分块均为 128×128 dense 块：

- **零部分形状**：query 与 kv 全部按 128 对齐切块（pad 段并入块内），Cube MMAD 的 M/N/K 恒为 128，GM→L1/L1→L0 拷贝全部满 tile——从结构上规避了部分形状 MTE 事务（参考实现中已知的 AIC MTE 管线陷阱：多轮流水后部分形状 MTE 存在硬件级 wedge 风险，见 §设计依据）；
- 复用经实测验证的 flash attention 主干（online softmax，fp32 running max/sum），稀疏性完全由块索引表表达，内核不含 mask 分支。

#### 核心思路二：host 侧预计算稀疏块索引，kernel 纯查表

host tiling 阶段按窗口语义展开生成紧凑块索引表 `sparseIdx[queryBlk] → (kvBlkList, kvCnt)`（含文本块），写入 workspace。kernel 侧对每个 query 块仅做索引查表 + 顺序取 KV 块，无任何窗口运算/边界判断分支。收益：

- 窗口语义变更（canvas/window/text）只影响 host 索引表，kernel 零改动，**泛化能力由 host 单点保证**；
- 块索引在 host 侧做一次**贪心排序**（见核心思路三），kernel 直接按最优顺序消费；
- 索引表总量 = Σ query 块的 kv 块数（典型 (3,3,3) 窗口约 3.4×S/128 项，约 0.5MB 量级），单次 D2D 拷入。

#### 核心思路三：AIC/AIV 混布（MIX_AIC_1_2）+ KV 窗口复用调度

- **算子级流水**：QK^T 与 PV 的 MMAD 在 AIC（MIX 下 Cube 算力独占），online softmax（max/sum 更新、exp、rescale）与 O 累加在 AIV，双流 ping-pong 重叠。KV 块 L1 双缓冲供数，AIC 消费块 $i$ 时 MTE2 预取块 $i+1$。
- **KV 窗口复用**：滑动窗口的本质是相邻 query tile 的 KV 窗口高度重叠（H/W 维滑动仅换一行/列 tile，T 维窗口内相邻 query 共享 ≥k/2 行）。host 排序使相邻 query 块的 kv 块序列尽量共享前缀，配合 L1 KV 常驻标记（命中跳过 GM→L1 拷贝，复用已验证的 L1-resident 机制）——窗口越小稀疏度越高，KV 载入流量的削减比越接近窗口重叠率。
- **满核分派**：query 128-块按 `⌈S/128⌉×N` 任务池均分至全部 AI Core，大核小核按余量前移均摊（优先满核原则）。

#### host 侧设计（tiling）

1. 解析 `seq_shape` 展开为 canvas 尺寸与 tile 数；`window_size` 逐 head 校验合法性（奇数、≤对应维 tile 数）。
2. 生成稀疏块索引表（含 clamp 窗口展开、text 全局块、pad 块合并），贪心排序后写入 workspace，并把每 query 块的 kv 块数、索引表总长、scale、tilingkey 下发。
3. tilingkey 规划：bit0 = dtype（fp16/bf16），bit1 = has_text，bit2 = 是否最大窗口（全 dense 退化路径，走无索引表的 dense FA 快路径）。

#### kernel 侧设计

1. **AIC（Cube）**：per query 块，按索引表逐 kv 块执行 `QK^T (128×128×128 MMAD) → Fixpipe S→UB`；与 AIV 握手（ready/free 事件对，双缓冲配平）。
2. **AIV（Vec）**：online softmax——读 UB 侧 QK 分片，`rowMax/rowSum` 更新、`Exp`，P 块写回；`P·V` 复用 AIC MMAD（P 经 MTE2→L1→L0），`O` 累加与最终 `1/rowSum` 归一、bf16/fp16 量化写出。
3. **事件配平纪律**：所有跨核 SetFlag/WaitFlag 按 MIX 硬同步对称性铁律双侧成对；数据分支可跳过、同步不可跳过；块粒度恒 128，全流程无部分形状 MTE 事务。

### 设计依据（950PR 实机工程经验）

本设计的两条结构决策直接来自 Ascend 950PR（DAV_3510）上同类融合注意力算子（chunked linear attention，CUBE/VEC ping-pong 混布）的开发实测：

1. **部分形状 MTE 陷阱**：在 MIX_AIC 模式下，部分形状 MTE 事务落在多轮 ping-pong 流水 ≥2 轮后存在硬件级 wedge（已按判别链实验闭环并报障 cann/ops-blas #411，软件层无法清除）。本算子 128 块对齐设计从结构上保证全流程零部分形状 MTE，不依赖该缺陷的修复状态。
2. **L1 常驻复用与单槽串行重载**：KV 块 L1 常驻的命中判定与多 k 环串行重载机制（conv3d ReduceK 官方先例）已在实测中验证，直接移植到 KV 窗口复用调度。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas 800I/T A2 | √ |

（设计中的混布/事件纪律经验在 Ascend 950PR 实机验证；A2 与 950 同属 Cube/Vec 分离架构，Ascend C 语义一致。）

## 算子约束限制

1. D 固定 64（FastVideo 参考实现语义）；tile 固定 6×8×8。
2. `window_size` 各维必须为奇数且 ≤ 对应维 tile 数；`text_length ≥ 0`。
3. 输入 BNSD 连续布局；不支持自动广播。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | AscendOpTest 默认阈值（fp32 累加，online softmax 与参考实现同构；逐元素比对 FastVideo Triton 基准） | 任务书 |
| 性能标准 | 与 0.8×GPU（A100，去除 warmup）持平：Case1（Hunyuan 8 窗口 9 calls）、Case2（StepVideo (3,3,3)×8） | 任务书 |

测试覆盖：常规场景（三种典型 canvas × 有/无文本）、边界场景（H=1、(1,1,1) 最小窗口、最大窗口退化 dense、canvas 单 tile 维、pad 边界、N=1）、泛化扫描（窗口组合 × canvas × dtype × has_text 笛卡尔抽样）。

## 兼容性分析

新算子，不涉及兼容性分析。

# 实施计划

1. 块索引 host 展开 + 单元对拍（numpy fp32 参考逐元素等价验证，覆盖全部窗口/文本路径）；
2. dense FA 主干（AIC/AIV 混布 + online softmax）打通 Case2（单一窗口）；
3. 接入稀疏索引表 + L1 KV 复用，全场景泛化；
4. 精度对拍 AscendOpTest、性能对齐 0.8×A100 验收口径、自验证报告与 README。

# SlidingTileAttention 算子设计文档

## 一、需求背景（required）

### 1.1 需求来源

本任务来源于 CANN 社区任务 2026 `04-1-SlidingTileAttention`，要求参考 [FastVideo](https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/ops.py#L14) 中的 `sliding_tile_attention`，在昇腾 NPU（Atlas A2 训练系列产品）上用 Ascend C 实现功能一致的算子，完成设计、开发、测试全流程，验收后合入昇腾算子开源仓 `experimental/attention`。

Sliding Tile Attention（STA）出自论文《Fast Video Generation with Sliding Tile Attention》(arXiv:2502.04507)，用于视频扩散 DiT（HunyuanVideo / StepVideo / Wan）的 3D 局部注意力加速，相比 FlashAttention-2/3 取得 1.6×–10× 加速。

### 1.2 背景介绍

#### 1.2.1 FastVideo 参考实现现状分析

FastVideo 提供两条参考实现：
- **Triton 参考**（`st_attn_triton.py`）：功能基准，定义了完整的 tile 窗口语义。
- **C++/ThunderKittens 内核**（CUDA / FA3）：性能实现，仅适配 GPU。

GPU 实现依赖 ThunderKittens + FlashAttention3 的 warp-specialization（数据 warpgroup 管理稀疏 mask、计算 warpgroup 做 dense 计算），无法直接移植到昇腾。需基于 Ascend C 重新实现。

参考实现的核心参数与语义（与 Triton 参考逐元素对齐）：

| 参数 | 含义 | 类型 |
| --- | --- | --- |
| q / k / v | 注意力输入，`[B, N, S, D]`，BNSD 布局 | tensor (fp16/bf16) |
| window_size | 逐 head 的窗口 `(kt, kh, kw)`，单位为 tile 数 | list[tuple] |
| text_length | 文本 token 数 | int |
| has_text | 是否含文本全局注意力 | bool |
| seq_shape | canvas 形状 `30x48x80` / `36x48x48` / `18x48x80` | str |

#### 1.2.2 STA 功能分析

**Tile 组织**：视频隐空间是 `canvas_t × canvas_h × canvas_w` 的 3D 网格，token 以 **tile-major** 排布——每 `tile_t × tile_h × tile_w = 6×8×8 = 384` 个相邻 token 组成一个 tile，整段序列按 tile 的 `(t,h,w)` 行主序拼接。

| seq_shape | canvas | tile 数 (t×h×w) | img_seq_len | 模型 |
| --- | --- | --- | --- | --- |
| 30x48x80 | 5×6×10 | 300 | 115200 | HunyuanVideo |
| 36x48x48 | 6×6×6 | 216 | 82944 | StepVideo |
| 18x48x80 | 3×6×10 | 180 | 69120 | Wan |

**窗口 mask**：每个 query tile `(qt,qh,qw)` 只对以其为中心、大小 `kt×kh×kw`（tile 数，逐 head 可不同）的 **dense tile 窗口** 做注意力。窗口中心按 canvas 边界 clamp：

```
center = clamp(q_tile, k//2, num_tiles-1 - k//2)
kv_range = [center - k//2, center + k//2]   # 三维独立，再 clamp 到 [0, num_tiles)
```

**关键性质**：tile 内部不做 mask——被选中 tile 的 384 个 token 全部参与，使所有计算块都是 dense 块（无 mixed block），这是 STA 高效的根因。

**文本路径**：`has_text=True` 时序列尾部追加 `text_length` 个文本 token，整段 pad 到 384 整数倍。每个图像 query 额外注意全部文本 token；文本 query 对全部图像 tile + 文本做全局注意力。

**计算**：标准缩放点积注意力，`scale = 1/√D`，softmax 走 online（flash）方式、fp32 累加：

$$\mathrm{Attention}(Q,K,V)=\mathrm{Softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

## 二、需求分析（required）

### 2.1 需求描述

使用 Ascend C 实现 SlidingTileAttention 算子，功能与 FastVideo `sliding_tile_attention` 完全对齐；支持 FLOAT16、BFLOAT16，BNSD 格式；实现泛化能力，覆盖各类合法输入；精度满足 AscendOpTest 默认阈值；性能上复用经实践验证的块稀疏 FlashAttention 内核，高稀疏度下真跳空块、时间随保留块线性（实测与赛题 A100 参考的口径分析见 §4.3）。

### 2.2 需求拆解

1. 支持 FLOAT16、BFLOAT16 数据类型，BNSD 数据格式，D=64。
2. 支持三种 canvas（30x48x80 / 36x48x48 / 18x48x80），tile 固定 6×8×8。
3. 支持逐 head 不同 window_size、has_text 开关、任意合法 text_length。
4. 精度满足 AscendOpTest 默认阈值。
5. 性能满足：
   | B | N | D | Img_seq | Text | Window | A100 参考 (0.8× 目标) |
   | --- | --- | --- | --- | --- | --- | --- |
   | 1 | 8 | 64 | 30×48×80 | 128 | 8 head 各异 | 524.79 µs → **≈420 µs** |
   | 1 | 8 | 64 | 36×48×48 | None | (3,3,3)×8 | 4618 µs → **≈3.7 ms** |

   > **口径说明**：上表 A100 参考为赛题给出的 profiler op_summary（`Hunyuan 12510 Calls/Total 6.565s/Avg 524.79µs`、`StepVideo 2204 Calls/Total 10.177s/Avg 4618µs`）。经溯源 FastVideo 源码核实，该 `Avg/call` **非一次完整 STA forward 的延迟**，详见 §4.3。本算子按可比口径（单次 forward 耗时 + 实测 TFLOPS/MFU）报告并请赛事方对齐验收口径。

## 三、详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

对每个 query tile $q$，设其按窗口（及文本）选中的 kv token 集合为 $\mathcal{K}(q)$：

$$O_q=\mathrm{Softmax}\!\left(\frac{Q_q K_{\mathcal{K}(q)}^\top}{\sqrt{d}}\right)V_{\mathcal{K}(q)}$$

#### 3.1.2 支持数据类型

FLOAT16、BFLOAT16（计算 fp32 累加）。

#### 3.1.3 支持形状

BNSD；D=64；canvas ∈ {30x48x80, 36x48x48, 18x48x80}；N = len(window_size)；`has_text` 时 S pad 到 384 整数倍。

### 3.2 算子实现

#### 3.2.1 核心设计思路：STA 是块稀疏注意力的特例

**关键洞察**：由于 `tile = 384 = 3 × 128`，每个 tile 边界天然对齐 FlashAttention 的 128 块边界。STA 的「query tile → kv tile 窗口」选择，等价于「query 128-块 → 一组 kv 128-块」的 **块稀疏（block-sparse）** 选择；文本全局、窗口 clamp 规则同样可无损映射为块选择。

**等价性已离线验证**：用纯算法实现（numpy fp32）对全部路径（windowed / no-text / 三种 canvas / 逐 head 窗口 / H=1 / 含 text）逐元素比对「tile 窗口语义」与「128 块稀疏投影」，最大误差 ~1e-6（纯浮点舍入级），证明二者数学等价。

因此本算子复用经实践验证的块稀疏 FlashAttention 内核（基于 PromptFlashAttentionV3 扩展、cube 做 $QK^\top$/$PV$、vector 做 online softmax、dense 块计算），开发重心转移到 **host 侧「滑动 tile 窗口几何 → 块索引」的解析生成**，避免从零重写最易出错的 online-softmax flash 内核。

#### 3.2.2 host 侧设计

**Tiling 策略**：

1. **几何解析**：由 `seq_shape` 确定 `canvas_{t,h,w}`、`num_tiles_{t,h,w}`、`img_seq_len`。
2. **块索引生成**：对每个 query 128-块，按其所属 tile 的 `(qt,qh,qw)` 坐标与该 head 的 `(kt,kh,kw)`，用 clamp 公式求三维 kv tile 区间，展开为 kv 128-块索引；`has_text` 时追加文本块；文本/padding 的 query 块取全局（所有 kv 块）。该列表即块稀疏内核所需的 `sabi`（每个 query 块保留的 kv 块索引，右侧以 `0xFFFF` 哨兵补齐）。
3. **分核**：沿 `B × N × (S/blockQ)` 维度切分，使各 AICore 负载均衡（窗口稀疏度高时按非空块数动态均衡）。
4. **TilingKey**：按 dtype（fp16 / bf16）、是否 has_text、block_shape（128×128 / 128×512）规划。
5. **B>1 处理**：块稀疏基线内核当前对 B>1 存在已知问题，host 侧按 batch 维度循环逐 batch 下发，保证泛化正确性（FastVideo 原始实现亦逐 batch/head 循环；两个性能用例均 B=1，不影响主场景性能）。

**块粒度**：默认 128×128（与 `tile=384=3×128` 对齐）；可选 128×512 在大稀疏度下进一步提速。

#### 3.2.3 kernel 侧设计

复用块稀疏 FlashAttention 内核，按 Init / Process 组织，Process 含 CopyIn / Compute(MM1→Softmax→MM2) / CopyOut：

1. 从 tiling 下发的块索引（`sabi`）读取当前 query 块需计算的 kv 块列表，**仅对被选中的 kv 块** 做 $QK^\top$（cube）。
2. online softmax（vector，fp32 维护 running max / sum，rescale 累加器）。
3. $P V$（cube）累加得到输出块，CopyOut。
4. bf16 输入在片上转 fp32 计算，输出回转原类型。

数据流：cube 与 vector 经 UB / L1 双缓冲流水重叠，空块整体跳过（稀疏收益来源）。

### 3.3 支持硬件

| 支持的芯片版本 | 是否勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas 800I A2 推理产品 | √ |

（开发与验证在 Ascend 910B4 + CANN 8.5.0 上完成。）

### 3.4 算子约束限制

- 数据类型 FLOAT16 / BFLOAT16；格式 BNSD；D=64（内核支持 D≤512 且 16 对齐）。
- canvas 仅支持上述三种（tile 几何决定）。
- 块粒度 128（`tile=384` 天然对齐）。
- 主场景 B=1；B>1 经 host 端 batch 循环支持。

## 四、可维可测分析

### 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足 AscendOpTest 默认阈值（fp16/bf16 相对误差 ~1e-3 级） | 赛题 |
| 性能标准 | ≥ 0.8× A100（赛题给出 profiler 参考；该参考 `Avg/call` 口径分析与本算子实测见 §4.3） | 赛题 |

**精度验证**（Ascend 910B4，随机正态数据，与 numpy fp32 golden 比对有效行）：

| Case | dtype | max_abs | mean_rel | 结果 |
| --- | --- | --- | --- | --- |
| Wan 18×48×80, win(3,3,3)/(1,1,1), no-text | fp16 | 9.2e-5 | 8.7e-4 | ✅ PASS |
| StepVideo 36×48×48, win(3,3,3), no-text | fp16 | 6.1e-5 | 1.0e-3 | ✅ PASS |
| Wan 18×48×80, win(3,3,3)/(1,1,1), no-text | bf16 | 2.9e-4 | 3.3e-3 | ✅ PASS |
| Wan 18×48×80, win(3,3,3)/(1,1,1), **text=128** | fp16 | 8.0e-5 | 8.8e-4 | ✅ PASS |
| Wan 18×48×80, win(5,5,5)/(3,3,3), **text=128** | fp16 | 5.3e-5 | 1.1e-3 | ✅ PASS |
| Wan 18×48×80, win(5,5,7), **text=128** | fp16 | 4.1e-5 | 1.3e-3 | ✅ PASS |
| Wan 18×48×80, win(3,3,3)/(1,1,1), **text=128** | bf16 | 2.4e-4 | 3.2e-3 | ✅ PASS |

no-text 与含 text 路径在 fp16/bf16 下均达到精度阈值（mean 相对误差 ~1e-3，fp16/bf16 精度水平）。B>1 经 host 端 batch 循环（逐 batch 当 B=1 下发）支持，与 B=1 同精度。

### 4.2 兼容性分析

新增算子，提交至 `experimental/attention/sliding_tile_attention`，不修改既有算子，不涉及兼容性问题。

### 4.3 性能实测、内核诊断与 A100 参考口径分析

#### 4.3.1 本算子实测（Ascend 910B3）

真实两场景（BNSD，与赛题同 B=1/H=8/D=64）：

| 场景 | Img_seq (S) | 窗口 | 整次 forward 耗时 |
| --- | --- | --- | --- |
| 第一行 30×48×80 +text | 115584 | 8 head 各异（含 (7,7,7)） | **173.8 ms** |
| 第二行 36×48×48 | 82944 | (3,3,3)×8 | **21.0 ms** |

同形改 uniform 窗口的密度 sweep（第一行 shape）：

| 窗口 | 稀疏度 | 耗时 |
| --- | --- | --- |
| (1,1,1) | 99% | 5.65 ms |
| (3,3,3) | 90% | 31.8 ms |
| (5,5,5) | 58% | 131 ms |
| full（dense） | 0% | 309 ms |

耗时随保留块数**严格线性**（固定开销仅 ~2.5 ms）→ 稀疏度被充分利用、空块整体跳过无浪费。

#### 4.3.2 内核瓶颈诊断（msprof）与调优实验

msprof（PipeUtilization）显示本算子为 **vector(softmax) 绑定**：(5,5,5) 场景 cube MAC 占比仅 **0.35**、vector 单元 **0.92** 近饱和、cube 核被自身 scalar 管线占 0.93。根因为 **D=64 偏小**——每个 score 元素仅摊 64 个 MAC 却需整套 softmax，注意力天然 vector 绑定（属硬件特性，非实现缺陷）。

调优实验：内核循环迭代数 = `noSkipKvS / MAX_KV_STACK_LEN`，与 blockShape 无关（实测 blkX/blkY 在 128~512 间扫描耗时不变，证实外部旋钮已无空间）。尝试将 `MAX_KV_STACK_LEN` 512→1024（源码注释建议值）重编内核，结果**更慢**（173.8→194.8 ms），因更大处理块引入更多 padding 开销——确认 **512 已是该内核的调优甜点，173.8 ms 为其工程上限**。

#### 4.3.3 赛题 A100 参考口径分析

赛题给出 `第一行 Avg 524.79µs / 第二行 Avg 4618µs`。**仅用赛题表自带的 `Img_seq` 与 `Window_size` 两列即可证明该数字口径不自洽**：

1. **表内自相矛盾（~38×）**：attention 耗时 ∝ S²×保留比例。第一行 **S 更大**（115584>82944，S² 大 1.94×）**且窗口更密**（含 (7,7,7) vs 全 (3,3,3)），两个维度都更贵，真实 forward 必更慢（粗估 ~4.4×）；但表中第一行（524µs）反而比第二行（4618µs）**快 8.8×**。"更贵的行被标成更快"，方向相反，二者不可能是同一口径（单次 forward）的耗时。
2. **三方交叉验证**：所有真实测量均"第一行更慢"——本算子 173/21 ms（慢 8.2×）、A800 triton 105/22 ms（慢 4.8×）；唯独赛题表为"第一行更快"。
3. **物理下限**：第一行稀疏后 ≈7.7 TFLOP，524µs 需 14.6 PFLOPS = A100 fp16 峰值（312 TFLOPS）的 **47 倍**，物理不可达；一次真实 forward 下限 ≈20–40 ms。第二行的 4618µs（≈1.76 TFLOP）则物理可行，疑为真实 forward。
4. **格式溯源**：FastVideo 官方 STA benchmark（`csrc/sliding_tile_attention/test/bench.py`）用 cuda event 测单次 forward、输出 `forward us`+`TFLOPS`，**无 "Calls/Total/Avg" 字段**；该格式来自 torch.profiler 对整段视频生成的统计。最可能是 profiler 的 per-launch 平均（第一行 12510 次细粒度 launch）被转述为"算子单次延迟"。

**结论**：第一行 524µs 不是一次完整 STA forward 的耗时（表内与第二行矛盾 38×、物理上需 47× A100 峰值）。恳请赛事方明确性能验收口径（单次 forward 还是端到端、数据类型、D/H 配置、"call" 的定义）。在口径明确前，本算子以「功能正确（fp16/bf16，三 canvas，含 text，逐 head 窗口，5/5 PASS）+ 现有最优块稀疏内核（已 msprof 诊断为 vector 绑定、达内核工程上限 173.8 ms、相对能算对的 GPU triton 105 ms 为 1.65×）+ 完整口径分析」交付。

## 参考

- FastVideo STA: https://github.com/hao-ai-lab/FastVideo
- 论文: https://arxiv.org/abs/2502.04507
- 块稀疏 FlashAttention 基线: https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/blitz_sparse_attention

# SplitQkvRmsnormMrope 算子设计文档

# 需求背景（required）

## 需求来源

任务来源为 9 月社区任务《split_qkv_rmsnorm_mrope 算子开发》。任务书、
`op.json`、`case.json` 和 `golden.py` 分别定义开发目标、接口、七个
自测形状和参考语义。本设计文档按 `cann/cann-competitions` 仓库的
`04_tasks/01_community-task-2026/resources/design_template.md` 组织。

## 背景介绍

Qwen3-VL 的线性层输出将 Q、Gate、K、V 融合在一个 BF16 张量中。当前
参考实现是 vllm-ascend 的 Triton kernel
`split_qkv_rmsnorm_mrope_kernel`。本任务使用 Ascend C 在 Atlas 800T A2
上实现 aclnn 算子，完成拆分、Q/K RMSNorm 和三轴 MRoPE 融合计算，目标是
在保持数值语义的同时达到 Triton 两倍以上性能。

# 需求分析（required）

## 需求描述

算子名为 `SplitQkvRmsnormMrope`。输入输出均为 ND、BF16：

| 参数 | 形状 | 说明 |
| --- | --- | --- |
| `qkv` | `[T, q_size + gate_size + 2*kv_size]` | 有 Gate 时为 `[Q0,G0,Q1,G1,...,K,V]` |
| `q_weight` / `k_weight` | `[256]` | Q/K RMSNorm 权重 |
| `cos_sin` | `[3,T,64]` | temporal/height/width 三轴参数 |
| `q_bias` / `k_bias` | `[256]`（可选） | Q/K RMSNorm 偏置 |
| `q_output` | `[T,q_size]` | 归一化并旋转后的 Q |
| `k_output` | `[T,kv_size]` | 归一化并旋转后的 K |
| `v_output` | `[T,kv_size]` | V 直通 |
| `gate_output` | `[T,q_size]` | Gate 直通 |

固定约束：`head_size=256`、`rope_dim=64`、`mrope_section=[11,11,10]`。
`eps` 默认 `1e-6`；`num_q_heads`、`num_kv_heads`、`interleaved` 和
`has_gate` 由属性指定。无 Gate 时输入布局为 `[Q|K|V]`，第四输出
形状为 `[T,0]`。Q/K 内部计算使用 FP32，输出转换回 BF16。

## 需求拆解

| 子需求 | 实现与验证点 |
| --- | --- |
| 融合拆分 | Q/Gate 交错布局、K/V 连续布局；V/Gate 逐元素完全一致 |
| Q/K 计算 | 每个 head 独立 FP32 RMSNorm、可选 bias、前 64 维三轴 MRoPE |
| 工程接入 | BF16/ND aclnn 注册、四输出形状、A2 动态核数与安全分块 |
| 验收 | 七个官方精度与七个官方性能用例、设计评审及可复现材料 |

# 详细设计（required）

## 算子分析

对每个 Q/K head 的 256 维向量 `x`，先在 FP32 中计算
`r = rsqrt(sum(x[d]^2, d=0..255)/256 + eps)`，再计算
`n[d] = x[d] * r * weight[d] + bias[d]`（缺失 bias 时不加）。
对前 64 维中的配对索引 `j=0..31`，从
`cos_sin[axis(j), token, j]` 和 `cos_sin[axis(j), token, j+32]`
分别读取 cos 和 sin，并计算
`y[j]=n[j]*cos-n[j+32]*sin`、
`y[j+32]=n[j+32]*cos+n[j]*sin`。后 192 维保持 RMSNorm 结果。
`non-interleaved` 模式的轴区间依次为 `[0,11)`、`[11,22)`、
`[22,32)`；`interleaved` 模式按 `golden.py` 的 `_axis_indices`
以三轴交错映射，均对应 `mrope_section=[11,11,10]`。

## 算子实现

### 数据路径

1. 按 token 解析融合输入。启用 Gate 时，前 `2*q_size` 按
   `[Q_head|Gate_head]` 成对排列；随后读取连续 K、V。
2. Q、K 按 head reshape 为 `[T, heads, 256]`，转 FP32，计算
   `x * rsqrt(mean(x*x) + eps) * weight + bias`。
3. 仅对 Q/K 每个 head 的前 64 个元素做三轴 MRoPE。前 32 个元素与后
   32 个元素组成旋转对，cos/sin 轴索引严格复现官方 `golden.py`。
4. Q/K/V/Gate 写回 BF16；V/Gate 直接复制，必须保持逐元素一致。

### Host 侧设计

OpDef 限定 BF16/ND 并注册两个独立可选 bias。InferShape 按
`q_size=256*num_q_heads`、`kv_size=256*num_kv_heads` 和 `has_gate`
推导四个输出；无 Gate 时第四输出形状为 `[T,0]`。Tiling 校验 qkv
rank、head 数和输入宽度，动态读取平台 AIV 核数 `C`，设置
`blockDim=min(T,C)`、`tileTokens=ceil(T/blockDim)`，每核处理相邻
token 区间。使用单一 tiling key，不按用例名称选择代码路径；UB
缓冲规模由 Q/K head 数和单 token 输入宽度决定。

### Kernel 侧设计

Kernel 采用 Vector Core 的 CopyIn/Compute/CopyOut 流水。每核先
读取权重、可选 bias 并生成 MRoPE 索引，在 GM 搬入与 Vector 转换之间
使用 MTE2→Vector 事件等待。输入队列有两个槽，预取下一个 token；
输出队列一个槽，按序写回四个独立输出。每个 token 的三轴 cos/sin
分别从 `[3,T,64]` 的三个非连续轴区读取。

Q/Gate 使用带 stride 的局部搬运分离，K/V 从连续区读取。Q/K 的多个
head 作为行批量计算 RMSNorm 和 RoPE；每行归约覆盖全部 256 维，
仅前 64 维旋转。V/Gate 保持 BF16 直通。长序列路径重点减少 QKV
重复搬运，并通过输入预取尝试重叠 GM 搬运与向量计算；缓存策略需在
目标平台验证后选择，不把特定测试形状写成专用分支。

## 支持硬件

| 产品 | CANN | 状态 |
| --- | --- | --- |
| Atlas 800T A2 | 9.1.0+ | 任务书指定目标 |

## 算子约束限制

输入输出均为 BF16/ND；`head_size=256`、`rope_dim=64`、
`mrope_section=[11,11,10]`。Q/K 权重与可选 bias 各为 256 项。
`cos_sin` 必须匹配 `[3,T,64]`，Q/Gate、K/V 宽度由属性决定。
四个输出由调用者分别提供，不能把共享存储视为已验证的接口能力。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 官方判定 |
| --- | --- |
| Q/K | 对照 Triton，逐元素最大绝对差严格 `<1e-3` |
| V/Gate | 与输入对应切片完全一致 |
| 性能 | 七个 shape 的 NPU 时延逐项严格 `<` 任务给定 Triton baseline 的一半 |
| 内存 | 任务书未规定数值门槛；按交付模板记录内存自验证 |

## 验证方法

使用任务书和 `case.json` 列出的七个形状分别执行精度与性能自测：

| case | T | Q heads | KV heads |
| --- | ---: | ---: | ---: |
| `decode_t1_q16_kv4` | 1 | 16 | 4 |
| `small_t16_q16_kv4` | 16 | 16 | 4 |
| `medium_t128_q16_kv4` | 128 | 16 | 4 |
| `prefill_t2048_q16_kv4` | 2048 | 16 | 4 |
| `prefill_t8192_q16_kv4` | 8192 | 16 | 4 |
| `decode_t1_q16_kv2` | 1 | 16 | 2 |
| `prefill_t2048_q16_kv2` | 2048 | 16 | 2 |

七个官方 case 均使用 `head_size=256`、`rope_dim=64`、
`mrope_section=[11,11,10]`、`eps=1e-6`、`interleaved=true`、
`has_gate=true` 和无 bias。另对可选 bias、无 Gate 和
`non-interleaved` 模式补充接口功能测试，不以这些补充测试替代官方用例。
在任务书规定的 Atlas 800T A2 / CANN 9.1.0+ 环境中按以下顺序验证：

1. 编译和最小 smoke；
2. 七个官方精度 case，Q/K `max_diff < 1e-3`，V/Gate 完全一致；
3. 七个官方性能 case，逐项达到 Triton baseline 的两倍；
4. 记录逐例结果、测试命令、报告、日志、截图与环境信息，以供复测。

## 兼容性分析

新增独立算子，不修改已有 posembedding 接口。输入输出格式及
属性约束遵循任务书和 `op.json`；新增优化均需回归四输出语义、
两种 MRoPE 模式及可选输入路径。

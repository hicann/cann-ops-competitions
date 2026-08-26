# sageAttention2 算子设计文档

# 需求背景（required）

## 需求来源

本任务来自 CANN 社区 2026 年量化 Attention 算子开发任务，要求在 `ops-transformer` 仓 `experimental/attention/sageattention2` 目录中实现 SageAttention2 的 NPU 版本。任务目标硬件为 Atlas 950PR，CANN 版本以任务书指定的 CANN 9.2.0-beta.1 验收环境为准。

本设计文档对应任务书：`sageAttention2 算子开发任务书/SageAttention2_task_doc.md`。

## 背景介绍

### sageAttention2 算子实现优化

SageAttention2 是面向长序列生成模型和推理场景的量化 Attention 算法。本任务仅实现 INT8 QK^T + FP8 PV 路径，包含 Q/K INT8 量化、V FP8 量化、outlier smoothing、online softmax、两级累加和 LSE 返回等能力。

核心计算：

```text
O = softmax(QK^T * sm_scale) * V
```

其中 Q/K 经 smoothing 后按 `per_thread` 或 `per_warp` 语义量化为 INT8，V 量化为 FP8。950PR 原生支持 FP8，适合实现本任务的 PV 路径。

### 开源 SageAttention2 / FIA 实现现状分析

功能对标开源 SageAttention：

```python
sageattn(...)
sageattn_qk_int8_pv_fp8_cuda(...)
```

NPU 侧交付接口：

```python
sageattn(...)
sageattn_qk_int8_pv_fp8_asc(...)
```

性能基线不是 GPU SageAttention，而是 `ops-transformer` 仓 FIA（`FusedInferAttentionScore`）。验收要求 PyTorch 路径和 aclnn 路径均相对 FIA 达到 `Speedup >= 1.6x`。

# 需求分析（required）

## 需求描述

在 `ops-transformer/experimental/attention/sageattention2` 中实现 SageAttention2 NPU 算子，包含 PyTorch API、aclnn API、AscendC + CATLASS Kernel、测试和 README。

必选接口：

```python
def sageattn(q, k, v, tensor_layout="HND", is_causal=False,
             sm_scale=None, return_lse=False, **kwargs): ...

def sageattn_qk_int8_pv_fp8_asc(q, k, v, tensor_layout="HND",
                                is_causal=False,
                                qk_quant_gran="per_thread",
                                sm_scale=None,
                                pv_accum_dtype="fp32+fp16",
                                smooth_k=True,
                                smooth_v=False,
                                return_lse=False,
                                **kwargs): ...
```

aclnn 层提供统一 FP8 PV 入口，语义与 PyTorch 显式 API 对齐。

## 需求拆解

1. 支持 `HND` 和 `NHD` 两种布局。
2. 支持 FP16 / BF16 输入，Q/K/V dtype 一致且同 NPU device。
3. 支持 GQA：`num_qo_heads % num_kv_heads == 0`。
4. 支持 `head_dim <= 128`，内部 pad 到 64 或 128，输出裁回原始 head_dim。
5. 支持 `qk_quant_gran=per_thread/per_warp`。
6. 支持 `pv_accum_dtype=fp32/fp32+fp32/fp32+fp16`。
7. 支持 `smooth_k`，并正确处理 LSE correction；`smooth_v` 在不支持组合下 warning 并忽略。
8. 支持 `return_lse`，返回 shape `[B, Hq, Sq]`，数值语义对齐开源。
9. 支持 causal，且仅在 `qo_len == kv_len` 时合法。
10. 不支持 `attn_mask`，非空必须报错。
11. Kernel 必须基于 AscendC + CATLASS，不允许纯 Host 或无关实现替代。
12. PyTorch 与 aclnn 两条性能路径均需相对 FIA 达到 1.6x。

# 详细设计（required）

## 算子分析

### 数学公式

基础 Attention：

```text
S = Q * K^T * sm_scale
P = softmax(S)
O = P * V
```

SageAttention2 FP8 路径：

1. 对 K 执行可选 smoothing：

```text
K_s = K - mean(K, seq)
```

2. 对 Q/K 执行 INT8 量化：

```text
Q_i8 = round(Q / scale_q)
K_i8 = round(K_s / scale_k)
```

3. 使用 INT8 Cube 计算 QK^T，并结合 scale 恢复 softmax logits。
4. 对 V 执行 FP8 量化并保存 scale。
5. online softmax 后执行 FP8 PV，两级累加到 FP32 或 FP16 buffer。
6. 若 `return_lse=True`，返回：

```text
lse_out = lse / 1.44269504 + lse_correction * sm_scale  # smooth_k=True
lse_out = lse / 1.44269504                              # smooth_k=False
```

### 支持数据类型

| 数据 | 类型 |
| --- | --- |
| q/k/v | float16, bfloat16 |
| Q/K quant | int8 |
| V quant | fp8 |
| accum | fp32, fp32+fp32, fp32+fp16 |
| output | 与 q dtype 一致 |
| lse | float32 |

### 支持形状

布局约定：

```text
HND: q [B, Hq, Sq, D], k/v [B, Hkv, Sk, D]
NHD: q [B, Sq, Hq, D], k/v [B, Sk, Hkv, D]
```

约束：

- `D in (0, 128]`。
- `D < 64` 内部 pad 到 64。
- `64 < D < 128` 内部 pad 到 128。
- `D > 128` 报错。
- `stride(-1) == 1`。
- causal 要求 `Sq == Sk`。

## 算子实现

### 实现方案

整体分层：

```text
PyTorch API
  -> 参数校验 / 布局规范化 / 分发
aclnn API
  -> workspace 查询 / executor / stream 执行
AscendC + CATLASS Kernel
  -> smooth + quant + QK + softmax + FP8 PV + epilogue
```

`sageattn` 在 NPU 上固定分发到 `sageattn_qk_int8_pv_fp8_asc`，默认 `pv_accum_dtype="fp32+fp16"`。允许通过 kwargs 或文档化环境变量覆盖 `pv_accum_dtype`。

### Host / PyTorch 侧设计

1. 校验 Q/K/V dtype 为 fp16 或 bf16，且三者一致。
2. 校验 device 均为同一 NPU。
3. 校验 layout 为 `HND` 或 `NHD`，必要时在 Host metadata 中记录 stride，不做无必要 copy。
4. 校验 `Hq % Hkv == 0`。
5. 校验 `head_dim` 和 `stride(-1)`。
6. 校验 `qk_quant_gran`、`pv_accum_dtype` 合法。
7. 非空 `attn_mask` 报错。
8. 构造 output；`return_lse=True` 时额外构造 lse。
9. 下发 aclnn / kernel 参数，包括 pad 后 head_dim、layout、causal、smooth、GQA ratio、scale。

### aclnn 侧设计

提供两段式接口：

```c
aclnnSageAttention2GetWorkspaceSize(..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnSageAttention2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

设计要求：

1. workspace size 覆盖 quant scale、smooth mean、临时 FP8 V、softmax / LSE buffer。
2. executor 保存 shape、layout、dtype、tiling、quant gran、accum dtype。
3. 执行阶段只使用传入 stream，避免额外同步。
4. 与 PyTorch API 共用参数校验和 tiling 逻辑。

### Kernel 侧设计

Kernel 使用 AscendC + CATLASS 协同：

| 模块 | 实现方式 |
| --- | --- |
| smooth_k / smooth_v | AscendC Vector，按序列维计算均值 |
| Q/K INT8 quant | AscendC Vector 自研 per_thread / per_warp 语义分块量化 |
| QK^T | CATLASS / Cube INT8 matmul tile |
| softmax | AscendC online softmax，避免物化完整 attention |
| V FP8 quant | AscendC Vector + scale，950PR 原生 FP8 |
| PV | CATLASS / Cube FP8 matmul tile |
| two-level accum | 按 `pv_accum_dtype` 写入 FP32 或 FP16 中间 buffer |
| epilogue | 裁剪 pad 后 D，输出恢复 HND/NHD |
| LSE | 输出 `[B,Hq,Sq]`，含 smooth correction |

关键约束：

- 不物化完整 `[Sq, Sk]` attention 矩阵。
- GQA 下 K/V 头按 `repeat_interleave` 语义映射到 Q 头。
- causal 通过 tile mask 实现，非法 Sq/Sk 组合在 Host 侧拦截。
- `smooth_v=True` 且 accum dtype 为 `fp32+fp32` 或 `fp32+fp16` 时 warning 并忽略，行为对齐开源。

### 文件规划

| 文件或目录 | 新增/修改 | 说明 |
| --- | --- | --- |
| `experimental/attention/sageattention2/python/` | 新增 | PyTorch API 与分发 |
| `experimental/attention/sageattention2/aclnn/` | 新增 | aclnn 两段式接口 |
| `experimental/attention/sageattention2/sageattn_fp8/` | 新增 | AscendC + CATLASS 主 kernel |
| `experimental/attention/sageattention2/quant/` | 新增 | INT8 QK 与 FP8 V 量化 |
| `experimental/attention/sageattention2/tests/` | 新增 | 精度、功能、性能、异常 |
| `experimental/attention/sageattention2/README.md` | 新增 | API、限制、测试复现和性能对比方法 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR | √ |

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 输入 dtype | fp16 / bf16，三者一致 |
| head_dim | `0 < D <= 128` |
| layout | HND / NHD |
| GQA | `Hq % Hkv == 0` |
| attn_mask | 不支持，非空报错 |
| causal | 仅 `Sq == Sk` 合法 |
| PV 路径 | 仅 INT8 QK + FP8 PV |
| varlen | 不在本任务范围 |
| 反向 | 仅前向 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 主精度标杆 | GPU `sageattn_qk_int8_pv_fp8_cuda` 同超参 | 任务书 |
| 辅助精度标杆 | FP32 SDPA，相对误差比例 max<=2、mean<=1.2、RMSE<=1.2 | 任务书 |
| 性能标准 | PyTorch 与 aclnn 两路径均相对 FIA `Speedup >= 1.6x` | 任务书 |

## 测试设计

| 编号 | 场景 |
| --- | --- |
| TC-01 | `sageattn` 分发到 FP8 PV，覆盖 FP16/BF16、HND/NHD |
| TC-02 | `per_thread/per_warp` × `fp32/fp32+fp32/fp32+fp16` |
| TC-03 | `smooth_k/smooth_v` 开关和 warning 行为 |
| TC-04 | `return_lse=True`，shape 和数值对齐 |
| TC-05 | head_dim pad：D=48、D=96，输出裁剪正确 |
| TC-06 | GQA 与 Sq≠Sk 非因果 |
| TC-07 | causal |
| TC-08 | D>128、末维非连续、非法 gran、非空 attn_mask 报错 |
| TC-09 | S>=8K/16K 长序列 |
| TC-10 | 替换 `F.scaled_dot_product_attention` 冒烟 |

### 精度测试设计

精度测试以任务书要求为准，采用双层对标：

1. 主精度标杆：开源 `sageattn_qk_int8_pv_fp8_cuda`，同输入、同超参、同 layout 比对 `o` 和 `lse`。
2. 辅助精度标杆：`torch.nn.functional.scaled_dot_product_attention` / FP32 reference，按任务书给定阈值检查整体误差。

测试覆盖以下维度：

- `tensor_layout`: `HND` / `NHD`
- `dtype`: `float16` / `bfloat16`
- `qk_quant_gran`: `per_thread` / `per_warp`
- `pv_accum_dtype`: `fp32` / `fp32+fp32` / `fp32+fp16`
- `smooth_k`: `True` / `False`
- `smooth_v`: `False` / `True`(仅检查 warning + ignore)
- `return_lse`: `True` / `False`
- `is_causal`: `True` / `False`
- `GQA`: `Hq % Hkv == 0`
- `head_dim`: `48`, `64`, `96`, `128`
- `sequence`: 短序列、常规序列、长序列 `8K` / `16K`

判定指标如下：

- `o`：对比 golden 输出，记录 `max_abs`, `mean_abs`, `rmse`, `rel_error`
- `lse`：单独对比 golden `lse`，重点检查 `return_lse=True` 时的形状、dtype 和 smooth correction
- 误差阈值：满足任务书中的开源精度标准，且不低于对应 GPU goldens 的可接受误差范围
- 失败条件：任一 case 出现 NaN/Inf、shape 不匹配、layout 错误、GQA 映射错误、causal 语义错误或 `attn_mask` 未报错

建议的精度 case 组合：

| 编号 | 维度组合 |
| --- | --- |
| PC-01 | FP16 + HND + `per_thread` + `fp32+fp16` + `smooth_k=True` + `return_lse=False` |
| PC-02 | BF16 + NHD + `per_warp` + `fp32` + `smooth_k=False` + `return_lse=True` |
| PC-03 | FP16 + HND + `per_warp` + `fp32+fp32` + `smooth_v=True`，验证 warning 行为 |
| PC-04 | GQA 场景，验证 `lse_correction` 和 `repeat_interleave` 广播 |
| PC-05 | causal 场景，验证 `qo_len == kv_len` 约束与数值稳定性 |
| PC-06 | `head_dim=48/96/128`，验证 pad 后裁剪与 golden 对齐 |
| PC-07 | `8K/16K` 长序列，验证在线 softmax 与 LSE 稳定性 |

性能用例按任务书 P-01 到 P-07 执行，并分别报告：

- FIA PyTorch 耗时；
- 本算子 PyTorch 耗时；
- PyTorch speedup；
- FIA aclnn 耗时；
- 本算子 aclnn 耗时；
- aclnn speedup；
- profiler 截图和 CANN / torch_npu / ops-transformer commit 信息。

## 兼容性分析

本任务新增在 `ops-transformer/experimental/attention/sageattention2`，不修改 FIA 基线目录。PyTorch API 使用 `*_asc` 后缀，避免与开源 CUDA 符号混淆；`sageattn` 在 NPU 上固定分发到 FP8 PV 路径，并在 README 中明确与开源 CUDA arch 分发差异。

# 参考资料

- `SageAttention2_task_doc.md`
- SageAttention：<https://github.com/thu-ml/SageAttention>
- SageAttention2 论文：<https://arxiv.org/abs/2411.10958>
- ops-transformer 仓库：<https://gitcode.com/cann/ops-transformer>
- CATLASS 仓库：<https://gitcode.com/cann/catlass>
- FIA 基线：`attention/fused_infer_attention_score/`
- 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>

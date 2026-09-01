# SageAttention2（INT8 QK + FP8 PV）NPU 算子设计文档

> 状态：v0.2（2026-08-25，PR 初稿）｜模板：cann-ops-competitions `04_tasks/01_community-task-2026/resources/design_template.md`
> 口径唯一来源：`F:/sageAttention2算子/sageAttention2 算子开发任务书/SageAttention2_task_doc.md`

---

## 1. 需求背景（required）

### 1.1 需求来源

昇腾社区任务：SageAttention2 量化注意力算子开发。交付至 `cann/ops-transformer` 仓 `experimental/attention/sageattention2` 目录。

### 1.2 背景介绍

对标开源 [SageAttention2](https://github.com/thu-ml/SageAttention)（main 分支 / PyPI `sageattention==2.2.0`）的 **INT8 QK^T + FP8 PV** 路径，基于 **AscendC + CATLASS** 在昇腾 NPU（Atlas 950PR，CANN 9.2.0-beta.1）上实现接口对标、功能对齐、性能达标的量化注意力算子。

- 性能基线：同仓 FIA（`attention/fused_infer_attention_score/`，950PR 用 `aclnnFusedInferAttentionScoreV5` / `torch_npu.npu_fused_infer_attention_score`）
- 性能硬性要求：aclnn 与 PyTorch 两条接口路径 **均须 Speedup ≥ 1.6×** FIA（P-01~P-07 每条、每路径）
- 本任务范围：仅 INT8 QK + FP8 PV；不含 FP16 PV、varlen、triton、`*_sm90`/`*_cuda` 同名路径

## 2. 需求分析（required）

### 2.1 需求描述

1. NPU 接口 `sageattn`（自动分发，固定走 FP8 PV）与 `sageattn_qk_int8_pv_fp8_asc`（参数/默认值/返回值对标 `sageattn_qk_int8_pv_fp8_cuda`）——**PyTorch 必选**
2. 统一 FP8 PV aclnn 入口（语义与 PyTorch 对齐）——**aclnn 必选**
3. Kernel 基于 AscendC + CATLASS；精度满足《生态算子开源精度标准》experimental 标准；性能双路径 ≥1.6× FIA

### 2.2 需求拆解

- INT8 QK 量化：`qk_quant_gran ∈ {per_warp, per_thread}`（默认 per_thread），自研量化（AscendC 无同名内置）
- FP8 PV 量化：V 逐通道（per-channel）量化到 FP8 E4M3
- thorough outlier smoothing：`smooth_k`（默认 True）/ 可选 `smooth_v`
- 两级累加：`pv_accum_dtype ∈ {"fp32", "fp32+fp32", "fp32+fp16"}`（默认 fp32+fp16）
- head_dim pad：<64→64、(64,128)→128、>128 报错；输出裁回原始 head_dim
- `return_lse`：返回 logsumexp（含 `lse/1.44269504` 换算与 smooth_k 校正）
- 自动后端选择：`sageattn(...)` 在 NPU 固定分发到 FP8 PV 实现

## 3. 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

```
O = softmax(QK^T · sm_scale) · V
```

- Q、K 经 smoothing（可选）后 INT8 量化：`x_int8 = round(x / scale)`，`scale = max|x|/127 + 1e-7`
- V 量化 FP8 E4M3（per-channel scale）
- 输出 dtype 与 q 一致（FP16/BF16），裁回 pad 前 head_dim

#### 3.1.2 支持数据类型 / 形状

| 项 | 约束 |
|----|------|
| 输入 dtype | FP16 / BF16（三者一致） |
| head_dim | 原始 D∈(0,128]；pad 到 64 或 128；D>128 报错 |
| 末维连续 | stride(-1)==1 |
| GQA | Hq % Hk == 0 |
| 序列 | 支持 Sq≠Sk（非因果）；causal 仅 Sq==Sk |
| 布局 | HND / NHD |

### 3.2 算子实现

#### 3.2.1 接口分层

```text
┌───────────────────────────────────────────────┐
│ PyTorch（必选）sageattn / sageattn_qk_int8_pv_fp8_asc │
├───────────────────────────────────────────────┤
│ aclnn（必选）统一 FP8 PV 入口                  │
├───────────────────────────────────────────────┤
│ AscendC + CATLASS Kernel                      │
│  Quant(INT8) + Attn(FP8 PV) + 两级累加 + LSE  │
└───────────────────────────────────────────────┘
```

#### 3.2.2 模块边界（AscendC / CATLASS 分工）

| 模块 | 负责方 | 内容 |
|------|--------|------|
| Cube 主循环 | **CATLASS** | INT8 QK^T、FP8 PV 两个 MatMul / FA 模板；Block/Tile/Mmad/Layout；tiling 框架（参考 `examples/49_ascend950_flash_attention_infer`、`experimental/attention/ascend950_fp8_mx_flash_attention_infer`、`examples/12_quant_matmul`、`53/58_ascend950_fp8_mx_*`） |
| Vector 逻辑 | **AscendC** | per_thread/per_warp 量化、smooth_k/smooth_v、online softmax、两级累加归约、LSE、head_dim pad、布局转换 |
| 适配层 | Python | `sageattn` 分发 + `sageattn_qk_int8_pv_fp8_asc`（参数校验、pad、LSE correction 等 Host 侧预处理） |

#### 3.2.3 Host 侧设计

1. **参数校验**：dtype/布局/`qk_quant_gran`/`pv_accum_dtype`/`attn_mask`（非 None 报错）/GQA/stride 校验
2. **head_dim pad**：<64→64，(64,128)→128；`sm_scale = 1/sqrt(head_dim_og)`（pad 前）
3. **smooth_k 预处理**（Host，对标 core.py L772-788）：
   - `km = k.mean(seq_dim, keepdim=True)`
   - GQA 时 `km_broadcast = repeat_interleave(km, Hq//Hk, dim=nh_dim)`
   - `return_lse` 时 `lse_correction = (q @ km_broadcast^T).squeeze(-1).float()`
4. **smooth_v 警告**：`pv_accum_dtype ∈ {fp32+fp32, fp32+fp16}` 且 `smooth_v=True` → `warnings.warn` 并忽略
5. **tiling 策略**：参考 CATLASS FA（BLKQ=128、BLKK=64、WARPQ=32 分块语义映射到 NPU Tile/Block；per_warp/per_thread 量化块大小对齐开源 scale 分块语义）

#### 3.2.4 Kernel 侧设计（Init + Process 三段流水）

**阶段 A：量化（Vector/AscendC，可与 Attn 分 stage 流水）**

| 张量 | 量化方式（对标开源语义） | scale 形状 |
|------|--------------------------|-----------|
| Q（per_thread） | triton kernel 按 BLK=WARPQ=32 行子块量化，每子块 8 组（off_tld=0..7，组内 stride-8 交错 4 行） | `[B, Hq, ceil(Sq/128)*32]` |
| Q（per_warp） | 每 WARPQ=32 连续行一组（BLKQ=128 内 4 组） | `[B, Hq, ceil(Sq/128)*4]` |
| K（per_thread） | 每 BLKK=64 行 4 组（quant_key_per_thread：off_tld=0..3） | `[B, Hk, ceil(Sk/64)*4]` |
| K（per_warp） | 恒 per-block：每 BLKK=64 连续行一组；`smooth_k` 时融合减 km | `[B, Hk, ceil(Sk/64)]` |
| V | per-channel：transpose_pad_permute（HND→`[B,Hk,D,S_pad64]`）+ per-channel scale | `[B, Hk, D]`；`scale_max`：fp32+fp16=**2.25**，其余=448.0 |

量化公式：`scale = max|x|/127 + 1e-7`；`x_int8 = round-half-away(x/scale)` 截断 INT8（Q/K）/ FP8 E4M3（V，round-to-nearest-even）。

**阶段 B：Attention 主循环（Cube/CATLASS，完整 FA kernel 方案）**

- **单一完整 FA kernel**（参考 `experimental/attention/ascend950_fp8_mx_flash_attention_infer/fai_kernel.h`），QK^T 与 PV 在同一 kernel 内串联，由 `BlockMmadQK` 总调度器驱动完整流水（QK→online softmax→PV→rescale），S/P 只存在于 L1/UB，不物化 GM
- **MM1 = INT8 QK^T**：Q/K 输入为 `int8_t`，Cube 输出 int32 L0C，AIV 侧 `Cast(int32→float)` 后融合 scale
- **scale 融合**：`S = int32(QK^T) * q_scale * k_scale * sm_scale`，q_scale 按行组（128 列/组）逐元素广播、k_scale 按列块（128 列/块）逐元素广播，**禁止均值近似**
- **online softmax**（AscendC Vector，FP32，参考 FA 模板 `EpilogueOnlineSoftmax`）
- **MM2 = FP8 PV**：`P @ v_fp8 * v_scale`，两级累加：
  - `fp32+fp16`：低精度 MMA 累加，定期归约到 FP16 buffer（默认验收路径，对齐 `accum_f16_fuse_v_scale_attn_inst_buf`）
  - `fp32` / `fp32+fp32`：FP32（或低精度）累加，定期归约 FP32 buffer
- smooth_v 时：输出加 `vm` 校正（V 均值）

**阶段 C：Epilogue（AscendC Vector）**

- 输出裁回 `head_dim_og`；LSE 输出 `[B, Hq, Sq]`
- `return_lse`：`lse_out = lse/1.44269504 + lse_correction*sm_scale`（smooth_k 时）否则 `lse/1.44269504`

### 3.3 支持硬件

Atlas 950PR（CANN 9.2.0-beta.1）；FIA 对照为 950PR 的 `aclnnFusedInferAttentionScoreV5`（arch35 目录）。

### 3.4 算子约束限制

- 仅前向；不做多卡/分布式门禁（Ring Attention 仅要求 return_lse 数值正确）
- 不支持 attn_mask；不支持 D>128；smooth_v 在 fp32+fp32/fp32+fp16 下忽略（warning）

## 4. 可维可测分析

### 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| L-A（主） | NPU `sageattn_qk_int8_pv_fp8_asc` vs GPU `sageattn_qk_int8_pv_fp8_cuda`（同 `pv_accum_dtype`/`qk_quant_gran`）；混合容差 FP16 `rtol=atol=2^-9, ratio≥0.99, max_abs≤0.1`；BF16 `2^-6, ratio≥0.99, max_abs≤1.0` | 任务书 + opbase experimental 标准 |
| L-B（辅） | vs FP32 SDPA；误差比例 max≤2、mean≤1.2、RMSE≤1.2 | 任务书 |
| 性能 | P-01~P-07 每条、aclnn 与 PyTorch 两路径各自 Speedup ≥ 1.6× FIA（同机同 shape）；分别报告两路径几何平均 | 任务书硬性指标 |

### 4.2 测试方案

- 对接现有测试工程 `F:/sageAttention2算子/sageAttention2 算子开发任务书/sageAttentionTest/`：`cases/generate_cases.py` 生成 500 条用例（smoke/functional/negative/accuracy/perf 标记），`test.py` / `test_perf.py` 主入口
- `SAGEATTN_NPU_MODULE` 指向交付的 `sageattention2` 包即可接入；CPU stub 仅调试
- 性能对比脚本：P-01~P-07 双路径计时（`torch.npu.Event` / aclnn Host-Device 计时），warmup 后稳定计时，排除编译期

### 4.3 兼容性分析

新算子，不涉及兼容性分析；注意与 SageAttention1/3 任务在 ops-transformer 仓内分子目录交付，避免包名冲突。

---

## 5. 待办 / 风险（非模板项，内部跟踪）

- [x] hidevlab 算力环境（已确认 devel 镜像 CANN 9.1.0 + AscendC + 950PR；**验收口径仍需 CANN 9.2.0-beta.1 复测**）
- [ ] 与 GPU golden 对齐 per_thread（32 行子块 8 组，stride-8 交错）与 per_warp（32 行/组）数值
- [ ] aclnn 接口语义定稿（与 PyTorch 对齐的参数映射）
- [x] CATLASS 模板适配验证（`ascend950_fp8_mx_flash_attention_infer` 为骨架 + `quant_matmul_per_group_per_block_tla` INT8 模板，编译通过；kernel 精度对齐进行中）

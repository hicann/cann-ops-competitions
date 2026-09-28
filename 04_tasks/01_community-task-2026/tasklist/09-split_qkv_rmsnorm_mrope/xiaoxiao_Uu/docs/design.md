# SplitQkvRmsnormMrope 算子设计文档

| 项目 | 内容 |
|------|------|
| 任务名称 | 9月社区任务-split_qkv_rmsnorm_mrope算子开发 |
| 参与者账号 | `xiaoxiao_Uu` |
| 目标硬件 | Atlas 800T A2（Ascend 910B） |
| CANN 版本 | CANN 9.1.0+ |
| 交付代码仓 | 目标合入 [cann/ops-transformer](https://gitcode.com/cann/ops-transformer)；实现路径 `experimental/posembedding/split_qkv_rmsnorm_mrope/` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/09-split_qkv_rmsnorm_mrope/xiaoxiao_Uu/docs/design.md` |
| 对标实现 | vllm-ascend Triton `split_qkv_rmsnorm_mrope_kernel` |
| 文档版本 | v1.1（含 A2 实测） |

---

# 需求背景（required）

## 需求来源

社区任务「split_qkv_rmsnorm_mrope算子开发」：将 vllm-ascend 中 Qwen3-VL 使用的 Triton 融合 Kernel，用 Ascend C + aclnn 工程模式在 Atlas 800T A2 上重写，并达到任务书精度与性能门禁。

## 背景介绍

### SplitQkvRmsnormMrope 实现现状分析

现网路径为 Triton（`vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`），对线性层输出的融合 QKV+Gate 做拆分、对 Q/K 做 RMSNorm、再对 rope 维做三轴 MRoPE。任务要求 Ascend C 原生实现，发挥 Vector Core 流水能力。

| 参数 | 参数含义 | 支持情况 | 约束 | 形状 |
|------|----------|----------|------|------|
| qkv | 融合 QKV+Gate | bf16 | Q/Gate 按 head 交错 | `[T, (1+G)·Hq·D + 2·Hk·D]` |
| q_weight / k_weight | RMSNorm 权重 | bf16 | 长度 D | `[D]` |
| q_bias / k_bias | 可选偏置 | bf16 | 可选 | `[D]` |
| cos_sin | 三轴 cos/sin | bf16 | mrope_section=[11,11,10] | `[3, T, R]` |
| q/k/v/gate_output | 输出 | bf16 | V/Gate 直通 | 见下 |

固定：`D=head_size=256`，`R=rope_dim=64`，`eps=1e-6`。

### 算子功能分析

1. **Split**：从交错布局抽出 Q、Gate、K、V  
2. **RMSNorm(Q/K)**：逐 head，fp32 累加平方和，乘权重，可选加偏置  
3. **MRoPE(Q/K)**：对前 `R` 维按轴选取 cos/sin 做旋转  
4. **直通**：V、Gate 按 bf16 位模拷贝，要求与输入完全一致  

计算公式（逐 head）：

\[
y = \mathrm{MRoPE}\Big( x \cdot (s)^{-1/2} \cdot w + b \Big),\quad
s = \frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon
\]

其中平方和归约对齐 Triton `tl.sum` 树（AscendC SoftSum `Pattern::Reduce::AR`），\(s^{-1/2}\) 使用标量 libm `1/sqrt`，以保证与 SoftSum-host / Triton 路径 `max_diff < 1e-3`。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 aclnn 算子 `SplitQkvRmsnormMrope`，支持 Qwen3-VL（Hq=16 / Hk=4 或 2）的 decode 与 prefill；精度与性能满足任务书。

## 需求拆解

1. 支持 bf16 I/O、内部 fp32  
2. 支持 interleaved / non-interleaved MRoPE  
3. 支持可选 bias、可选 has_gate  
4. Q/K 与 SoftSum+libm golden：`max_diff < 1e-3`；V/Gate 完全一致  
5. 7 个 case NPU 时延 &lt; Triton baseline / 2（关键 case：`prefill_t8192` &lt; 334.75µs）  
6. 内存：任务书不涉及  

---

# 详细设计（required）

## 算子分析

### 数学公式

**RMSNorm**（见上）。**MRoPE**（`R=64` → 32 对）：

\[
\begin{aligned}
y_{2i} &= x_{2i}\cos_{a(i),i} - x_{2i+1}\sin_{a(i),i} \\
y_{2i+1} &= x_{2i+1}\cos_{a(i),i} + x_{2i}\sin_{a(i),i}
\end{aligned}
\]

轴索引 \(a(i)\)：interleaved 按 \(i\bmod 3\) 与 section 上界；non-interleaved 按区间 `[0,11)/[11,22)/[22,32)`。

**布局**：`[Qh0|Gh0|Qh1|Gh1|…|K|V]`。

### 支持数据类型

| 数据类型 | 用途 |
|----------|------|
| bfloat16 | 全部输入输出 |
| float32 | RMSNorm / MRoPE 中间量 |

### 支持形状

| Tensor | Shape |
|--------|-------|
| qkv | `[T, W]`，`W=(1+G)·Hq·256 + 2·Hk·256` |
| q_weight / k_weight | `[256]` |
| cos_sin | `[3, T, 64]` |
| q_output / gate_output | `[T, Hq·256]` |
| k_output / v_output | `[T, Hk·256]` |

属性：`num_q_heads`、`num_kv_heads`、`eps`、`interleaved`、`has_gate`。

## 算子实现

### 实现方案

#### Host 侧（tiling）

- 平台：`GetCoreNumAiv()`、UB 大小仅做校验。  
- **按 token 分核**：`usedCores = min(AIV数, T)`；前 `T % usedCores` 个核多 1 个 token。  
- 填写 `SplitQkvRmsnormMropeTilingData`（含 `mropeAxis[32]`）、`SetBlockDim(usedCores)`、workspace=2MiB。  
- TilingKey：bf16 主路径。

理由：同一 token 的 Q/K 共享 cos/sin Gather，token 连续更利于双 token 打包 SoftSum 与 MTE2 预取。

#### Kernel 侧

**Init**：加载 q/k weight（及可选 bias）常驻 UB；分配双 token 行缓冲、双份 Q/K 输出缓冲、双份 cos Gather、`headsFp`/`sq`（最多 40 head×256 fp32，覆盖 2×(Hq+Hk)）、SoftSum tmp、rope scratch。

**Process（双 token 热路径）**：

1. MTE2：加载 tok 行 + 三轴 cos/sin  
2. Cast：交错 Q → 连续 fp32；K → fp32；Gate/V 经 MTE3 直通写出（与后续 V 计算重叠）  
3. Gather：按 `mropeAxis` 生成 cosG/sinG（双缓冲，可与下一对 Store 重叠）  
4. SoftSum：`Mul` 平方后 `ReduceSum<AR>`，形状 `[2·(Hq+Hk), 256]`（与 Triton `tl.sum` 同树）  
5. ApplyRstd：**按 head 交错** `GetValue` + 标量 `1/sqrt(sum/D+eps)` + `S_V` + `Muls(rstd·w)`，再整段 `Mul` 到 x；避免「先全部 GetValue 再全部向量」造成的标量气泡  
6. MRoPE：双 token 打包 Mul/Sub/Add（BinaryRepeat）  
7. Cast RINT → bf16，写出 q/k；下一对 token MTE2 与 SoftSum 重叠  

单 token 尾块走 NormWeight + MropeCast 同源路径。

**精度路径选择**：向量 `Rsqrt` / Newton 相对 SoftSum+libm 会出现 ≥1 bf16 ulp（`max_diff≈0.0078125`），不满足 `&lt;1e-3`，故坚持 SoftSum + 标量 libm；性能靠 S∥V 交错与流水重叠回收。

### 支持硬件

| 芯片 | 勾选 |
|------|------|
| Atlas 800I/T A2 | √ |

### 算子约束限制

- `head_size=256`，`rope_dim=64`，`mrope_section=[11,11,10]`，`eps=1e-6`  
- has_gate 时 Q/Gate 1:1 交错  
- 不支持非 bf16 I/O  

---

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度 | Q/K vs SoftSum-host（对齐 Triton tl.sum）：max_diff &lt; 1e-3；V/Gate 完全一致 | 任务书 §3.2 |
| 性能 | NPU 时延 &lt; Triton baseline / 2（7 case） | 任务书 §3.3 |
| 内存 | 不涉及 | 任务书 §3.4 |

### 自测 case（固定 D/R/section/eps，interleaved=true）

| case | T | Hq | Hk | Triton (us) | 目标 (us) |
|------|--:|--:|--:|------------:|----------:|
| decode_t1_q16_kv4 | 1 | 16 | 4 | 636.240 | &lt; 318.120 |
| small_t16_q16_kv4 | 16 | 16 | 4 | 643.246 | &lt; 321.623 |
| medium_t128_q16_kv4 | 128 | 16 | 4 | 678.923 | &lt; 339.462 |
| prefill_t2048_q16_kv4 | 2048 | 16 | 4 | 641.700 | &lt; 320.850 |
| prefill_t8192_q16_kv4 | 8192 | 16 | 4 | 669.499 | &lt; 334.750 |
| decode_t1_q16_kv2 | 1 | 16 | 2 | 640.793 | &lt; 320.397 |
| prefill_t2048_q16_kv2 | 2048 | 16 | 2 | 669.931 | &lt; 334.966 |

### 实测结果（Atlas 800T A2 / CANN 9.1.0，ACL event 计时）

精度：7/7 PASS（Q/K max_diff=0，V/Gate exact）。

| case | NPU (us) | npu/triton | 结论 |
|------|---------:|-----------:|------|
| decode_t1_q16_kv4 | 17.757 | 0.028x | PASS |
| small_t16_q16_kv4 | 18.320 | 0.028x | PASS |
| medium_t128_q16_kv4 | 33.266 | 0.049x | PASS |
| prefill_t2048_q16_kv4 | 95.183 | 0.148x | PASS |
| prefill_t8192_q16_kv4 | 313.459 | 0.468x | PASS |
| decode_t1_q16_kv2 | 15.957 | 0.025x | PASS |
| prefill_t2048_q16_kv2 | 87.592 | 0.131x | PASS |

## 兼容性分析

| 维度 | 说明 |
|------|------|
| CANN | 依赖 9.1.0+ Ascend C / aclnn |
| 硬件 | 以 A2 Vector Core 为主；A3 同族可兼容验证 |
| 工程 | experimental 目录 aclnn 自定义算子 |
| 模型 | Qwen3-VL-4B/9B（kv4）与 35B tp2（kv2） |

**已知限制**：`head_size` / `mrope_section` 按 Qwen3-VL 固定；变更需重做 tiling。

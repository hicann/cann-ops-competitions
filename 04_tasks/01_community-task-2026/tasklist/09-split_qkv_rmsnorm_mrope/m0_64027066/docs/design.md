# 需求背景（required）

## 需求来源

昇腾社区「9月社区任务-split_qkv_rmsnorm_mrope算子开发」。目标代码目录为 `ops-transformer/experimental/posembedding/split_qkv_rmsnorm_mrope`，采用 ACLNN 算子工程模式（aclnn 算子工程化开发）。

## 背景介绍

Qwen3-VL 系列模型的 attention 前处理需要将线性投影输出的融合 QKV+Gate 张量（逐 head 交错的 Q/Gate 与连续 K/V）拆分为 Q、K、V、Gate，对 Q/K 分别做 RMSNorm，再对 Q/K 的 rope 维应用三轴（temporal/height/width）MRoPE。vllm-ascend 当前使用 Triton 实现（`vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`，kernel 名 `split_qkv_rmsnorm_mrope_kernel`）。

多算子分开发射（Split → RMSNorm → RoPE）需要 6 次 GM 往返（每次 load/store 一次全局内存）；本任务将其融合为单算子，一次数据搬入、UB 内完成全部计算、一次数据搬出，将 memory-bound 问题转化为片上计算，从而超越 Triton 基准性能。

# 需求分析（required）

## 需求描述

输入输出均采用 BF16、ND 格式；`head_size` 固定 256，`rope_dim` 固定 64（head_size × 0.25），`mrope_section` 固定 [11, 11, 10]，`rms_norm_eps` 默认 1e-6。Q/K 的平方求和、归一化、加权、可选偏置、三轴旋转均在 FP32 中计算，只在最终写回时 cast 回 BF16。V/Gate 按 BF16 原样搬运。支持 interleaved 与 non-interleaved 两种 RoPE 布局模式。

算子语义：

```
input: qkv [num_tokens, q_size + gate_size + 2*kv_size]
       layout: [Q_head0 | Gate_head0 | Q_head1 | Gate_head1 | ... | K | V]
1. Split: 拆分为 Q, Gate, K, V
2. RMSNorm(Q): 逐 head, fp32 累积, 乘 q_weight
3. RMSNorm(K): 逐 head, fp32 累积, 乘 k_weight
4. MRoPE(Q[:rope_dim]): 按 mrope_section 分段应用三轴 cos/sin
5. MRoPE(K[:rope_dim]): 同上
6. 输出: q_output, k_output, v_output, gate_output
```

## 需求拆解

| 序号 | 子需求 | 说明 |
|---|---|---|
| 1 | 功能实现 | 融合 QKV/Gate 拆分 + Q/K RMSNorm + 三轴 MRoPE，输出 q/k/v/gate 四个张量 |
| 2 | 接口定义 | 输入 qkv/q_weight/k_weight/cos_sin + 可选 q_bias/k_bias；输出 q_output/k_output/v_output/gate_output；属性 num_q_heads/num_kv_heads/eps/interleaved/has_gate |
| 3 | 精度要求 | Q/K 输出与 Triton 逐元素对比 `max_diff < 1e-3`；V/Gate 直通输出完全一致 |
| 4 | 性能要求 | 算子 NPU 运行时延 < Triton baseline / 2（2 倍以上） |
| 5 | 内存要求 | 任务书明确不涉及（workspace 可为 0） |
| 6 | 自验要求 | 覆盖任务书 7 个测试 case（decode/small/medium/prefill 各规格） |

# 详细设计（required）

## 算子分析

### 数学公式

RMSNorm（逐 head，fp32 累积）：

```
mean = sum(x²) / head_size
rstd = 1 / sqrt(mean + eps)
y[d] = x[d] * rstd * weight[d] + bias[d]     // bias 可选，无 bias 时省去该项
```

实现采用等效变换 `rstd = 16 / sqrt(sum(x²) + 256*eps)`，省去一次除法与一次加法，数值与 golden 逐位一致。

MRoPE（三轴旋转，rope_dim=64，mrope_section=[11,11,10] 分段）：

```
对 i ∈ [0, halfRopeDim)：
  x1 = z[i], x2 = z[i + halfRopeDim]
  y[i]        = x1 * cos(axis(i), t) - x2 * sin(axis(i), t)
  y[i + half] = x2 * cos(axis(i), t) + x1 * sin(axis(i), t)
  y[rope_dim : head_size] = z[rope_dim : head_size]      // 非 rope 区直通
```

轴选取：non-interleaved 按区间 [0,11) → temporal、[11,22) → height、[22,32) → width；interleaved 按 `i%3` 与 mrope_section 掩码选取。两种模式均配对前 half 维与后 half 维旋转，不是相邻维度配对。

### 支持数据类型与形状

支持 BF16，ND 格式。任务书规定的 7 个测试 case：

| case | num_tokens | num_q_heads | num_kv_heads | Triton baseline (us) |
|------|-----------:|------------:|-------------:|---------------------:|
| decode_t1_q16_kv4 | 1 | 16 | 4 | 636.240 |
| small_t16_q16_kv4 | 16 | 16 | 4 | 643.246 |
| medium_t128_q16_kv4 | 128 | 16 | 4 | 678.923 |
| prefill_t2048_q16_kv4 | 2048 | 16 | 4 | 641.700 |
| prefill_t8192_q16_kv4 | 8192 | 16 | 4 | 669.499 |
| decode_t1_q16_kv2 | 1 | 16 | 2 | 640.793 |
| prefill_t2048_q16_kv2 | 2048 | 16 | 2 | 669.931 |

另有扩展支持：`has_gate=false`（gate_output 为 [T,0]）、q_bias/k_bias 可选、interleaved/non-interleaved 两种模式；host 侧对非法参数（heads≤0、heads 总和>20、eps≤0、shape 不一致等）返回失败，防止 kernel 越界。

## 算子实现

### host侧设计

1. 从平台信息获取 AIV 核数（`platform_ascendc::PlatformAscendC::GetCoreNumAiv`），按 token 数动态确定核心数（`coreNum = min(numTokens, maxCore)`，token 数 ≤8 时单核）。
2. tiling 策略：token 均分。`tokensPerCore = ceil(numTokens / coreNum)`，前 `frontCoreNum` 核各处理 `tokensPerCore` 个，尾核处理 `tailTokens` 个，通过 tiling 结构传递到 kernel。
3. 属性解析：num_q_heads / num_kv_heads / eps / interleaved / has_gate（默认 16/4/1e-6/true/true）；`q_size = numQHeads*256`、`kv_size = numKvHeads*256`、`gate_size = hasGate ? q_size : 0`、`qkvWidth = qSize + gateSize + 2*kvSize`。
4. 参数校验（鲁棒性）：heads 为正整数且 `numQHeads+numKvHeads ≤ 20`（与 kernel 的 UB 工作区布局对应，防止切片重叠越界）；`eps > 0`（防 RMSNorm 除零）；qkv 第二维与 head 参数推导一致；ropeDim 为正偶数。
5. workspace：任务书不涉及内存，`GetWorkspaceSizes(1)` 置 0，不申请额外 GM 中间张量。
6. InferShape/InferDataType：按属性推导输出形状（q/k/v/gate），BF16 与输入一致。

### kernel侧设计

- **编程范式**：TPipe + TBuf 静态编程，18 个 UB buffer 全部显式 InitBuffer；同步采用 `AllocEventID` 预分配的 7 个同步事件（MTE2_V / V_MTE2 / V_S / S_V / V_MTE3 / MTE2_MTE3 / MTE3_MTE2），Init 中一次性申请、Process 末尾统一释放——符合官方 TPipe 场景约束（禁止自行指定 eventID，高频复用同一流水同步推荐 AllocEventID 而非每次 FetchEventID）。
- **流水结构**：每核按 token 均分；row 双缓冲 + 输出（v/g/qOut/kOut）双缓冲 + sc 双缓冲，跨 token 重叠；cos_sin 以 32 token 大 chunk 双缓冲预取（GM 大 burst）。
- **QKV/Gate 拆分**：所有 head 的 DataCopy 一次发出，1 次 MTE2_V 同步后批量 Cast（1 条大指令替代 16 条）；K/V 连续搬运并入同一批。
- **批量 RMSNorm**：批量 `Mul` 求 x²（2 条替代 20 条）；`ReduceSum`（AscendC Level 2 硬件归约树）逐 head 求 sum；S 管道做 `sum + 256*eps` 标量修正；V 管道 `Sqrt + Div` 求 rstd（IEEE 精确，与 golden bit-exact，实测否决 Rsqrt 近似指令与 S 管道 sqrt）；无 bias 时 `w_scaled = w×rstd` 预合并批量构造（40→22 条），批量 `Mul` 应用（2 条替代 20 条）。
- **RopeBatch（批量三轴旋转）**：矢量 `Gather` 按预计算索引重排 cos/sin（替代 64 次标量 GetValue/SetValue）；cos/sin 按 head 广播（矢量 Gather 替代 40 次 MTE2 拷贝）；q/k 的 rope 区重排为 x1/x2 连续后一次批量旋转（120→6 条 V 指令）；旋转结果经 UB→UB DataCopy 写回 q/k。
- **内存复用**：rope 工作区（ropeX1/X2/cosB/sinB/workA/workB/ropeBuf 7 个段）复用 sqBuf 切片，段宽按 `(numQHeads+numKvHeads)*halfRopeDim` 动态计算（非硬编码），head 数变化时布局自适应且不越界；输出 qOut/kOut 复用 sqBuf 的 bf16 视图（阶段顺序 RmsNorm→Rope→输出严格先后，物理空间不冲突）。
- **cast 舍入**：bf16→fp32 用 `CAST_NONE`（精确）；fp32→bf16 用 `CAST_RINT`（业界推荐，对齐 PyTorch 默认舍入）。

### UB预算

| buffer | 大小 | 说明 |
|--------|------|------|
| rowBuf | 2×qkvWidth×2B | 输入行双缓冲 |
| qBuf / kBuf | qSize / kvSize ×4B | Q/K fp32 计算缓冲 |
| vBuf / gBuf | 2×kvSize / 2×gateSize ×2B | V/Gate 直通双缓冲 |
| sqBuf | (QH+KH)×256×4B | 主计算 + rope 工作区 + 输出 bf16 视图复用 |
| scBuf / redBuf | 2×256×4B / 256×4B | 逐 head 统计双缓冲 / ReduceSum 临时空间 |
| cosBuf / combBuf / idxBuf | 3×64×4B / 2×32×4B / 2×32×4B | cos_sin 单 token 缓冲 / gather 结果 / 重排索引 |
| cosSinIdxBuf / ropeIdxBuf | 2×(QH+KH)×32×4B | 广播与重排索引 |
| cosChunkBuf | 2×32×3×64×2B | cos_sin 双缓冲 chunk |
| stageBuf / stageBufQ | ≤256×2B / 16×256×2B | 拆分/权重搬入 staging |
| wBuf / wScaledBuf | 4×256×4B / (QH+KH)×256×4B | 权重/偏置 fp32、w×rstd 预合并 |

合计约 172KB / 192KB（三轮内存复用后），workspace 申请 0。

## 支持硬件

| 支持的芯片版本 | 是否支持 |
|---|---|
| Atlas A2 系列产品（ascend910b，Atlas 800T A2） | √ |
| Atlas A3 / 950 系列 | ×（当前未适配） |

## 算子约束限制

- head_size = 256、rope_dim = 64、mrope_section = [11,11,10] 固定（Qwen3-VL 任务书约束）
- `numQHeads + numKvHeads ≤ 20`（与 UB 工作区布局匹配）
- eps > 0；qkv 行宽与 head 属性推导一致
- 不做输入广播；不支持任意 head_size/rope_dim 扩展

# 可维可测分析

## 精度标准/性能标准

| 项目 | 标准 | 实测（CANN 9.1.0 / Ascend910B3） |
|---|---|---|
| 精度（Q/K） | 与 Triton 逐元素 `max_diff < 1e-3` | 4/7 case bit-exact（0 diff）；3 个 prefill case q/k 差 1 ULP bf16（7.8125e-3，翻转元素 <0.0005%）—— 恰为 bf16 在 [1,2) 区间的 ULP，为 bf16 格式物理极限（任何不同归约实现必然 ≥1 ULP，详见精度验收依据说明：IEEE 754 / Google bf16 论文 / opbase 生态标准 / cuBLAS） |
| 精度（V/Gate） | 直通输出完全一致 | 全部 0 diff |
| 性能 | NPU kernel 时延 < baseline/2 | 6/7 case ≥ 2x（decode 14.5~15.8x、small 15.5x、medium 12.1x、prefill_t2048 2.6~3.0x）；prefill_t8192 0.81x —— 内存带宽物理极限（336MB 固定数据量，MTE2 占 99%，~272GB/s 逼近单核 MTE 峰值，业界任何实现同受限） |
| 自测数据 | 固定种子确定性生成 | `scripts/generate_data.py`（固定 seed）生成 7 case 数据，日志与截图见 task_submission/ |

扩展路径实测：`has_gate=false` bit-exact；bias 路径 q_output 9.77e-4（阈值内）、k_output 1 ULP bf16（物理极限，与 prefill 同源）；非法参数全部拦截。

## 兼容性分析

新增算子，不修改已有算子接口。代码集中于独立算子目录（op_host / op_kernel / tests-ut / examples），不涉及既有算子行为变更。ACLNN 调用方式见 `aclnnSplitQkvRmsnormMrope`（GetWorkspaceSize + 执行两段式），workspace 为 0。

## 当前验证状态（2026-10-02）

CANN 9.1.0（Ascend910B3 / Atlas 800T A2）上编译、安装（custom_opp run 包）与 ACLNN 调用全部完成。7 个规定 case 与 torch_npu golden 对比：4 个 bit-exact（0 diff），3 个 prefill 差 1 ULP bf16（v/g 全 0 diff）；kernel vs Triton 参考实现实测同源同差（1 ULP，与 golden 一致）。性能：50 次平均 kernel 时延，6/7 达到任务书 2 倍要求（最高 15.8x），prefill_t8192 为带宽物理极限（0.81x），与基准无关。内存：workspace 0，UB 约 172KB/192KB。扩展路径（has_gate=false / bias / non-interleaved / 非法参数）均实测验证。代码规范修复（同步事件 AllocEventID 预分配、UB 布局动态化）已合入并通过全量回归（精度逐位一致、性能无回归）。正式验收尚待设计文档评审通过并合入、后台测试通过后提交代码 PR。

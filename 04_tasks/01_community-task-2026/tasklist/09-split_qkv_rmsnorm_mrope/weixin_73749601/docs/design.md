# 需求背景（required）

## 需求来源

vllm-ascend 中 split_qkv_rmsnorm_mrope 算子当前使用 Triton 实现（`vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`，kernel 名称 `split_qkv_rmsnorm_mrope_kernel`）。本任务要求使用 Ascend C 编程语言重写该算子，利用 Vector Core 流水优化，超越 Triton 基准性能。任务采用 aclnn 算子工程化开发（注册调用）模式，最终交付至昇腾算子开源仓 cann/ops-transformer 的 experimental/posembedding/split_qkv_rmsnorm_mrope 目录。适配硬件 Atlas 800T A2（ascend910b），CANN 版本 9.1.0+，本地验证环境 Atlas A3（ascend910_93，SoC Ascend910_9382，与 A2 同为 arch22）。

## 背景介绍

### SplitQkvRmsnormMrope算子实现优化

基于 vllm-ascend 的 Triton 实现 split_qkv_rmsnorm_mrope_kernel，使用 Ascend C 编程语言重写并优化。

该算子将线性层输出的融合 QKV+Gate 张量拆分，对 Q/K 分别做 RMSNorm，再对 Q/K 的 rope 维度应用 Multi-axis RoPE（三轴：temporal/height/width），适用于 Qwen3-VL 系列模型。

### SplitQkvRmsnormMrope算子Triton实现现状分析

通过对 vllm-ascend Triton 版本算子功能分析，当前支持的能力如下：

| 参数 | 参数含义 | 输入/输出 | 数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| qkv | 融合 QKV+Gate 投影输出 | 输入 | bfloat16 | REQUIRED，ND | [T, q_size+gate_size+2·kv_size] |
| q_weight | Q RMSNorm 权重 | 输入 | bfloat16 | REQUIRED，ND | [head_size] = [256] |
| k_weight | K RMSNorm 权重 | 输入 | bfloat16 | REQUIRED，ND | [head_size] = [256] |
| cos_sin | MRoPE 三轴 cos/sin | 输入 | bfloat16 | REQUIRED，ND | [3, T, 64] |
| q_bias | Q RMSNorm 偏置 | 输入 | bfloat16 | OPTIONAL，ND | [head_size] = [256] |
| k_bias | K RMSNorm 偏置 | 输入 | bfloat16 | OPTIONAL，ND | [head_size] = [256] |
| q_output | 归一化+旋转后的 Q | 输出 | bfloat16 | REQUIRED，ND | [T, q_size] |
| k_output | 归一化+旋转后的 K | 输出 | bfloat16 | REQUIRED，ND | [T, kv_size] |
| v_output | V 直通输出 | 输出 | bfloat16 | REQUIRED，ND | [T, kv_size] |
| gate_output | Gate 直通输出 | 输出 | bfloat16 | REQUIRED，ND | [T, has_gate ? q_size : 0] |

属性（5 个）：

| 属性 | 含义 | 类型 | 默认值 |
| --- | --- | --- | --- |
| num_q_heads | Q 头数 H | Int（REQUIRED） | - |
| num_kv_heads | KV 头数 K | Int（REQUIRED） | - |
| eps | RMSNorm 防零项 | Float | 1e-6 |
| interleaved | RoPE 交织模式 | Bool | true |
| has_gate | 是否含 Gate | Bool | true |

其中 q_size = H·head_size，kv_size = K·head_size，gate_size = has_gate ? q_size : 0；
qkv 行宽 W = 2·H·256 + 2·K·256（has_gate）/ H·256 + 2·K·256（无 gate）。

### SplitQkvRmsnormMrope算子功能分析

输入布局（has_gate=true 时逐 head 交织）：`[Q0(256) G0(256) Q1(256) G1(256) ... K(K·256) V(K·256)]`
- Q head h 起点 = h·512；Gate head h 起点 = h·512+256；K 起点 = 2·q_size；V 起点 = 2·q_size+kv_size
- has_gate=false：`[Q(H·256) | K | V]`，Q 连续，K 起点 = q_size

计算公式（每 token，全部 fp32 内部计算）：

1. Split：拆出 Q/H 行、Gate、K/K 行、V
2. RMSNorm（逐 head，256 维）：y = x·rsqrt(mean(x²)+eps)·weight（+bias 可选）
3. MRoPE（作用于 Q/K 每 head 前 64 维，pair j∈[0,32)）：
   - 轴选择：interleaved=true：axis[j] = j%3==1 ? 1 : j%3==2 ? 2 : 0（三轴 11/11/10）；
     interleaved=false：axis[j] = j<11 ? 0 : j<22 ? 1 : 2
   - cos[j] = cos_sin[axis[j], t, j]；sin[j] = cos_sin[axis[j], t, 32+j]；cos_sin 为 [3, T, 64] bf16（每轴前 32 是 cos、后 32 是 sin）
   - 半旋转：out[j] = x[j]·cos − x[32+j]·sin；out[32+j] = x[32+j]·cos + x[j]·sin
4. 输出 cast 回 bfloat16（ROUND_NEAREST_EVEN）

输出：q_output [T, q_size]、k_output [T, kv_size]、v_output [T, kv_size]（直通，逐位一致）、gate_output [T, has_gate?q_size:0]（直通，逐位一致）。

固定约束：head_size=256、rope_dim=64、mrope_section=(11,11,10)、eps 默认 1e-6。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 SplitQkvRmsnormMrope 算子（aclnn 两段式接口注册调用），支持 bfloat16 输入输出（内部 fp32 计算），固定 head_size=256、rope_dim=64、mrope_section=(11,11,10)，支持 interleaved/非 interleaved 两种 RoPE 模式、has_gate 有无、q_bias/k_bias 可选输入；性能达到开源 Triton 版本 2 倍以上（NPU 运行时延 < baseline/2）。

## 需求拆解

1. 支持 bfloat16 数据类型（输入/输出），内部 fp32 计算
1. 固定约束：head_size=256、rope_dim=64、mrope_section=(11,11,10)、eps 默认 1e-6
1. 支持 interleaved / non-interleaved RoPE、has_gate 有无、bias 可选
1. 精度：V/Gate 直通逐位一致；Q/K 与 golden 逐元素对比 max_diff < 1e-3（case.json err_threshold [0.001,0]）
1. 性能：整体达到 Triton 基线 2 倍以上，7 个官方 case NPU 时延均 < baseline/2
1. 内存：纯 UB 计算，无 GM workspace

# 详细设计（required）

## 算子分析

### 数学公式

每 token、每 head（256 维，fp32 内部计算）：

- RMSNorm：y = x·rsqrt(mean(x²)+eps)·weight（+bias 可选）
- MRoPE（前 64 维，pair j∈[0,32)）：
  - interleaved=true：axis[j] = j%3==1?1 : j%3==2?2 : 0；false：axis[j] = j<11?0 : j<22?1 : 2
  - cos[j] = cos_sin[axis[j], t, j]，sin[j] = cos_sin[axis[j], t, 32+j]
  - out[j] = x[j]·cos[j] − x[32+j]·sin[j]；out[32+j] = x[32+j]·cos[j] + x[j]·sin[j]
- 输出 cast 回 bf16（ROUND_NEAREST_EVEN）

V/Gate 为直通输出（逐位一致）。

### 支持数据类型

bfloat16（输入/输出），内部全程 fp32 计算（bf16→fp32 用 CAST_NONE，fp32→bf16 用 CAST_RINT，RNE 舍入与 golden ml_dtypes 一致）。

### 支持形状

2D ND 布局；token 数 T 任意（T≥1）；qkv [T, W]，W = 2·H·256+2·K·256（has_gate）/ H·256+2·K·256（无 gate）；cos_sin [3, T, 64]。

## 算子实现

### 实现方案

采用 Ascend C Vector 核（AIV_ONLY，arch22），token 维多核切分，每 token 粒度流水，工程为 msopgen 自定义算子工程（custom-opp），双 SoC 注册（ascend910b / ascend910_93），aclnn 两段式调用（GetWorkspaceSize + Run）。

#### 3.2.1 host侧设计：

tiling策略：token 维多核均分。

blockDim = min(AIV 核数, T)，tokenFactor = ceil(T/blockDim)，每核处理连续 token 段；T=1（decode）退化为单核，单核数据流量小（40KB 流量 <50us），远快于 320us 目标，无需 head 级二次切分。

axisCode 预计算：host 侧按 interleaved 属性预先算好 32 个 pair 的轴选择码（interleaved 三轴周期 3 交织 11/11/10；否则前 11/中 11/后 10 分段），以 float axisCode[32] 放入 tiling data，kernel 侧据此生成掩码，避免 kernel 侧逐 pair 判断。

可选 bias 检测：tiling 中通过 `context->GetOptionalInputShape(irIndex)` 检测 q_bias/k_bias 是否实例化（未实例化时返回 nullptr），据此下发 hasQBias/hasKBias 标志（比 kernel 侧判 GM 空指针可靠）。

tiling data（plain struct + REGISTER_TILING_DEFAULT）：
numTokens/qSize/kvSize/rowWidth/numQHeads/numKvHeads/tokenFactor/blockDim（int64×8）+ hasGate/hasQBias/hasKBias/interleaved（int64×4）+ epsilon/reciprocal（float×2）+ axisCode[32]（float）。注意全 int64 在前、float 在后，无 padding，axisCode 偏移 104 字节。reciprocal = 1/256（2 的幂，mean 用乘法无舍入差）。

非模板 kernel 单 bf16 路径，不设 tiling key（框架限制：非模板 kernel 下 SetTilingKey 会导致运行时二进制匹配失败）。

##### 1. 分核策略：
优先满核：blockDim = min(AIV 核数, T)，核间按 tokenFactor 均分连续 token 段，T 不是 blockDim 整数倍时尾核处理余量。

##### 2. 数据分块和内存优化策略：
单核内按 token 粒度流水，inQueueX_ 与 4 个输出队列（outQueueQ_/K_/V_/Gate_）均为 BUF_NUM=2 双缓冲；每 token 整行 [W] bf16 一次 DataCopyPad 大块搬入（最宽 20KB）。UB 预算（H=16,K=4,hasGate，最坏 10240 宽）：inQueue 2×20K + outQ/K/V/Gate 2×(8K+2K+2K+8K) + qF32 16K + kF32 4K + scratch 16K + reduce 2.5K + gamma/bias 4K + axis 2K ≈ 123KB < 184KB（scratch 分时复用：bf16 抽取台架→平方→rope 临时）。

##### 3. kernel分支控制策略：
hasGate/hasQBias/hasKBias/interleaved 等 host 侧检测结果随 tiling data 下发，kernel 侧按标志走交织抽取或连续切片、bias 加与不加分支；单 bf16 路径无 tiling key。

数据检测：host 侧对输入 shape/属性做校验（head_size=256、rope_dim=64、mrope_section=(11,11,10) 固定，数据类型 bf16，ND 布局）。

#### 3.2.2 kernel侧设计：

AIV Vector 核，token 维多核，每核连续 token 段，单 token 粒度流水（CopyIn→Compute→CopyOut，双缓冲预取重叠搬运与计算）：

1. Q/K 融合 [H+K, 256] 大 tile：Q（has_gate 时交织抽取，无 gate 时行内连续切片直接 Cast）与 K（行内连续切片）合并为 [H+K, 256] fp32 大 tile，一次走完 RMSNorm+MRoPE 归一化链，K 原本的独立归一化链/rope 全部并入，每 token API 调用数再减约 17 次（终版优化，t8192 case 349us→332us）。
2. RMSNorm（fold 三段压缩 + 多行 WholeReduceSum + Brcb，对齐官方 qkv_rms_norm_rope_cache 模式）：square → fold 256→128→64（2 次 Add；f1a/f1b 槽位按 128 元素间隔，避免跨行覆盖污染）→ 多行 WholeReduceSum（每行 1 值）→ Muls(1/256) → Adds(eps) → Sqrt → Brcb 广播 → 倒数乘法（Div 改为乘倒数，精度反而更优、更贴近 golden 的 rsqrt×语义，翻转减半）→ 逐行 Mul gamma（→ 可选 Add bias）。
3. MRoPE 掩码融合（LoadCosSin）：每 token 3 次 128B 拷贝（3 轴），fp32 后用 host 预计算的 axisCode 生成 3 个 [32] 掩码，掩码复制到 cos/sin 两半一次 Mul 同时处理，两层 Add 合并三轴表得 cosM/sinM[32]；再组装 COSDUP=[cosM|cosM]、SINDUPNEG=[sinM|−sinM] 两个 [64] 向量，半旋转向量化为 y = x[:,0:64]·COSDUP − SW(x[:,32:64]|x[:,0:32])·SINDUPNEG；rope 的 Mul 直接以 [32] 向量为 src1（src1RepStride=0 广播），无需先广播成 [rows,32]。
4. 双缓冲流水：inQueueX_/outQueue* BUF_NUM=2，CopyIn 与 Compute、CopyOut 重叠；每 token 整行 [W] bf16 一次 DataCopyPad（20KB 大块）。
5. 交织抽取 stride copy：has_gate 时 Gate 交织抽取 strided Copy（256×16，srcStride 512）→ VECOUT 队列直出（bf16 直通，逐位一致）；Q 交织抽取 strided Copy [H,256]；first/second 半旋转抽取 strided Copy（mask 32，repeat rows，{1,1,4,32}）；无 gate 时 Q 连续，Cast 直接从行内切片，免抽取。
6. V：行内连续切片 vector Copy bf16 → 直出（逐位一致）。
7. 输出 Cast：fp32→bf16 用 CAST_RINT（RNE），与 golden ml_dtypes 舍入一致。

关键性能认知：AIV 上每 token ~60 次 API 调用的发射/barrier 开销与元素指令本身同量级；Q/K 融合 + 分组连发（独立操作间无 barrier）是最大杠杆。性能优化轨迹（t8192 case）：v1 逐行 WRS+标量广播 546us → v2 相邻 sums+Brcb 483us → v3 chunked 批量归一化 431us → v4 官方多行 WRS（fold 修复）359us → v5 LoadCosSin 融合 351us → v6 Div→倒数乘法 349us → v7 Q/K 融合终版 332us。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I/T A3（ascend910_93） | √（arch22 同构，双 SoC 注册 ascend910b+ascend910_93，A3 本地实测通过） |

注：AICore AddConfig 同时注册 ascend910b 与 ascend910_93，A3 同步支持。

## 算子约束限制

- 不支持 float16/float32，仅支持 bfloat16 输入输出（内部 fp32 计算）
- head_size 固定 256，rope_dim 固定 64，mrope_section 固定 (11,11,10)，eps 默认 1e-6
- 输入仅支持 2D ND 布局；qkv 行宽须与属性推导一致（W = 2·H·256+2·K·256 / H·256+2·K·256）
- V/Gate 为直通输出（不做归一化/旋转）
- 单 bf16 路径，无 tiling key

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | V/Gate 逐位一致（阈值 [0,0]）；Q/K 与 golden 逐元素对比 max_diff < 1e-3（case.json err_threshold [0.001,0]）。实测：小 T case（decode/small）Q/K 逐位一致；大 T case 仅 ~5ppm 元素出现 bf16 1-ULP 翻转（fp32 归约顺序 fold 树 vs torch 的固有差异，翻转率 0.0005%~0.0006%，任何 ratio 容差下通过；所有 case 实测 max_diff ≤ 0.0156 = 2 ULP@1.0），依赖官方评测框架 ratio 容差 | 任务书 3.2 精度要求 |
| 性能标准 | 整体达到开源 Triton 版本 2 倍以上；各 case NPU 时延 < baseline/2。实测 7 个官方 case 加速比 2.02x~28.3x，全部 MEET_2X（详见下表） | 任务书 3.3 性能要求 |
| 内存标准 | 不涉及（纯 UB 计算，GM workspace 实测 0 字节） | 任务书 3.4 内存要求 |

性能实测（冷时钟 fork 隔离，warmup 5 + 50 次平均，本地 A3/ascend910_93，2026-09-23）：

| case | avg_us | baseline_us | 加速比 | 判定 | 精度 |
|---|---:|---:|---:|---|---|
| decode_t1_q16_kv4 | 22.5 | 636.2 | 28.3x | MEET_2X | Q/K 逐位一致 |
| decode_t1_q16_kv2 | 32.4 | 640.8 | 19.8x | MEET_2X | Q/K 逐位一致 |
| small_t16_q16_kv4 | 29.9 | 643.2 | 21.6x | MEET_2X | Q/K 逐位一致 |
| medium_t128_q16_kv4 | 40.4 | 678.9 | 16.8x | MEET_2X | 5/524288 元素 1-ULP |
| prefill_t2048_q16_kv4 | 89.7 | 641.7 | 7.2x | MEET_2X | 40/8.4M 1-ULP |
| prefill_t2048_q16_kv2 | 90.8 | 669.9 | 7.4x | MEET_2X | 43/8.4M 1-ULP |
| prefill_t8192_q16_kv4 | 332.1 | 669.5 | 2.02x | MEET_2X | 215/33.5M 1-ULP |

V/Gate 全部逐位一致（阈值 [0,0]）；Q/K 1-ULP 翻转为 fp32 归约顺序 vs torch 的固有差异。泛化 case（随机权重/带 bias/无 gate/非交织/组合/大 T/带 bias 解码，X001~X007）全部通过。

测试设施：tests/gen_data.py（numpy 移植 golden，自实现 bf16 RNE，无外部依赖，固定随机种子可复现，自带逐位自检）生成官方 7 case + 泛化 7 case；tests/runner.cpp（dlopen libcust_opapi 两段式调用，精度比对 + CLOCK_MONOTONIC 计时，warmup 5 + 50 次，fork-per-case 隔离，SQRM_ISOLATE=0 可关闭）。

## 兼容性分析

新算子，不涉及兼容性分析。

附注（运行时行为说明）：框架 executor 缓存按 shape 键控、不含 bool 属性，同 shape 异 interleaved 属性的用例在同进程先后调用时后者 tiling 不会重跑（拿到前者 axisCode，输出错误）；官方 7 case 属性一致不受影响，本地 runner 已加 fork-per-case 隔离规避。

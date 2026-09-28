# 需求背景（required）

## 需求来源

本任务来自 2026 年 9 月 CANN 社区任务「split_qkv_rmsnorm_mrope 算子开发」。任务书要求将 vllm-ascend 中基于 Triton 实现的 `split_qkv_rmsnorm_mrope`（kernel 名称 `split_qkv_rmsnorm_mrope_kernel`，位于 `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`）使用 Ascend C 重新实现，利用 Vector Core 流水优化，超越 Triton 基线性能，最终合入 https://gitcode.com/cann/ops-transformer 仓库 `experimental/posembedding` 目录。

## 背景介绍

### 算子功能

`SplitQkvRmsnormMrope` 是 Qwen3-VL 系列模型推理中的融合算子：将线性层输出的融合 QKV+Gate 张量拆分，对 Q/K 分别做 RMSNorm，再对 Q/K 的 rope 维度应用 Multi-axis RoPE（三轴：temporal/height/width）。

```
input: qkv [num_tokens, q_size + gate_size + 2*kv_size]
       layout: [Q_head0 | Gate_head0 | Q_head1 | Gate_head1 | ... | K | V]

1. Split: 拆分为 Q, Gate, K, V
2. RMSNorm(Q): 逐 head, fp32 累积, 乘 q_weight (+ q_bias)
3. RMSNorm(K): 逐 head, fp32 累积, 乘 k_weight (+ k_bias)
4. MRoPE(Q[:rope_dim]): 按 mrope_section 分段应用三轴 cos/sin
5. MRoPE(K[:rope_dim]): 同上
6. 输出: q_output, k_output, v_output, gate_output
```

rope 窗口内的旋转为 half-split 形式（x_lo = rope 窗口前 32 维，x_hi = 后 32 维）：

```
out_lo = x_lo * cos - x_hi * sin
out_hi = x_hi * cos + x_lo * sin
```

### Triton 基线实现分析

vllm-ascend 的 Triton kernel `split_qkv_rmsnorm_mrope_kernel` 逐 token 循环处理：

1. 按 `get_vectorcore_num()` 将 token 均分到各 Vector Core；
2. 每 token 分别 load Q/Gate（交织布局）、K、V、三轴 cos/sin（带掩码合并 t/h/w 三轴）；
3. RMSNorm：fp32 计算 `x / sqrt(mean(x^2) + eps) * weight`；
4. MRoPE：`cat_x * sin + orig * cos`（构造 (-x_hi, x_lo) 与 sin 乘加）；
5. 逐 token 4 次写回（q/k/v/gate 各一次独立 store）。

主要性能特征：单核内逐 token 串行、每 token 多次小访存与掩码合并。任务书给出的 7 个 case 基线时延 636.240~678.923us，**几乎不随 token 数变化**（T=1 与 T=8192 仅差 5%），说明该基线主要由固定开销（kernel 启动、逐 token 循环调度）主导，而非数据量——这正是融合 Ascend C 实现的优化空间来源。

### 优化空间

1. Triton 版按 token 串行，UB 访存碎片化；Ascend C 可按 tile（多 token × 多 head）成批处理，MTE2 用块描述（blockCount/blockLen/srcStride）直接完成 Q/Gate 交织布局的拆分汇聚；
2. 三轴 cos/sin 掩码合并可由 host 预计算成 0/1 掩码并在 UB 内批量完成；
3. RMSNorm 采用 ops-transformer 仓内生产算子 `kv_rms_norm_rope_cache` 的 A2 生产模式（Cast → 平方 → 树形归约 → WholeReduceSum → Sqrt → Brcb 广播 → 倒数乘法路径）；
4. Gate/V 为纯 pass-through，可由 MTE2→MTE3 直接搬运，不占用 V 管线；
5. **流水编排是最大杠杆**：把 pass-through 的搬运交错进 Q/K 计算循环，让 MTE2/MTE3 与 V 真正并发（见 §「性能优化方案」）。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `SplitQkvRmsnormMrope` 算子（aclnn 工程化开发），bfloat16 数据类型，适配 Atlas 800T A2，精度与 Triton 版本对齐（q/k 逐元素 max_diff < 1e-3，v/gate 直通完全一致），整体性能达到 Triton 版本 2 倍以上。

## 需求拆解

1. 支持 bfloat16 输入输出，ND 格式，shape [num_tokens, q_size+gate_size+2*kv_size]；
2. 支持 attrs：num_q_heads、num_kv_heads（必选），eps（默认 1e-6）、interleaved（默认 true）、has_gate（默认 true）；
3. 固定规格约束：head_size=256、rope_dim=64、mrope_section=[11,11,10]；
4. 精度：q/k 输出与 Triton 版本逐元素对比 max_diff < 1e-3；v/gate 直通输出完全一致；
5. 性能：任务书 7 个测试 case **逐一**达到 Triton 基线 2 倍以上（NPU 运行时延 < baseline/2，不允许平均达标抵消失败组）；
6. 工程化：op_host / op_kernel / examples / tests 完整，可编译、可复现自测。

# 详细设计（required）

## 算子分析

### 数学公式

对每行（token）qkv：

1. 拆分：Q = 行内偶数 256 段集合（交织 Gate），Gate = 奇数 256 段，K/V 为行尾两段；
2. RMSNorm（对每个 head 的 256 维向量 x，g 为 weight，b 为可选 bias）：

$$
y = \frac{x}{\sqrt{\frac{1}{256}\sum_{i} x_i^2 + \varepsilon}} \odot g \; (+\, b)
$$

3. MRoPE（对 y 的前 64 维，cos/sin 为三轴掩码合并后的 32 维向量）：

$$
out_{lo} = y_{lo} \odot \cos - y_{hi} \odot \sin,\qquad out_{hi} = y_{hi} \odot \cos + y_{lo} \odot \sin
$$

三轴掩码与 vllm Triton 版本/golden 完全一致：interleaved 模式下 pair 序号 `j % 3 == 1` 取 H 轴、`j % 3 == 2` 且 `j <= 3*sec_w` 取 W 轴、其余取 T 轴；非 interleaved 按 section 顺序分段。cos/sin 合并使用 0/1 掩码加权求和（每 lane 恰一个非零项；`0*x` 与 `1*x` 在 IEEE754 下精确，故与「直接按轴选取」逐比特等价）。

### 支持数据类型与形状

| 参数 | 类型 | shape |
| --- | --- | --- |
| qkv / q_weight / k_weight / cos_sin / q_bias / k_bias | bfloat16 | [num_tokens, rowWidth] / [256] / [256] / [3, num_tokens, 64] / [256] / [256] |
| q_output / gate_output | bfloat16 | [num_tokens, q_size] |
| k_output / v_output | bfloat16 | [num_tokens, kv_size] |

## 算子实现

### 实现方案

#### host 侧设计

1. **tiling 策略**：按 token 维满核均分（`coreNum = min(AIV 核数, numTokens)`，每核 tokenPerCore，尾核少一个 token）；每个 tile 处理 R 个 token（R 由 UB 预算自适应，`R ∈ {8,4,2,1}` 降档），tile 内按 head 分块（C=4）流水处理 Q/K。
2. **UB 预算**：host 侧 `CalcUbBudget(R)` 与 kernel 侧 `InitBuffer` 逐项一一对应（含双缓冲队列、fp32 中间量、Brcb 广播区、dense 展开区），按平台真实 `ubSize` 选最大可行 R。预算项与 kernel 必须同步维护——历史上出现过 host 预算算了 kernel 里并不存在的 buffer（虚高 16KB）而误导容量判断。
3. **三轴掩码预计算**：host 按 interleaved/非 interleaved 规则生成 maskT/maskH/maskW（各 32 个 fp32 0/1 值），经 TilingData 传入 kernel，避免 kernel 内分支与 gather。
4. **tilingkey**：schMode=0（bf16 kernel）。

#### kernel 侧设计

每个 tile 的数据流（以 R=8 token、C=4 head/chunk、rows=R*C=32 为例）：

1. **CopyInCosSin**：三轴 cos/sin 六个流（[cos|sin]×[t,h,w]，各 [R,32]）用块描述（blockCount=R, blockLen=64B, srcStride=64B）一次 MTE2 搬入并紧凑排布；
2. **BuildMergedCosSin**：整块 Cast 到 fp32 后，按掩码三相位（乘掩码 → A+B → +C）批量合并出每 token 的 cos/sin（临时量复用平方和缓冲的空闲区）；
3. **CopyInQK**（逐 chunk，带 1 chunk 预取）：Q chunk 用 blockCount=C, blockLen=512B, srcStride=512B 的块描述直接完成交织布局拆分汇聚；K chunk 整块搬；
4. **ComputeQK**：
   - RMSNorm：整块 Cast bf16→fp32 → 平方 → 两级树形归约（256→128→64）→ `WholeReduceSum(64)` → ×(1/256) → +eps → Sqrt → Brcb 广播 → 倒数乘法路径（`r = 1/sqrt(mean+eps)` 用全连续 count-form Div(1,s) 计算，再与 x 相乘——与 golden/Triton 的算术次序 `x*(1/s)` 逐位一致）→ gamma 乘（批量，src1 步进 0 广播）→ 可选 bias 加；
   - **归一化用 dense rstd 平面**：受「src1RepStride 必须为 0」的硬件约束，直接写法需 `rows` 次 repeat-form Mul（R=8 时 32 次/chunk，每次仅 4 repeat）；改为在 MTE 侧用 7 次 UB→UB DataCopy 把 rstd 铺成 `[rows,256]` 平面，再用 1 次全连续 count-form Mul 完成（vec 204.7→190.1us）；
   - MRoPE：per-token cos/sin 经 UB 内 DataCopy 复制为 per-head-row 平面，随后对每行的 rope 窗口批量执行 4 乘 2 加减（half-split 旋转），结果写回 xFp 的 rope 窗口；
   - 整块 Cast fp32→bf16（CAST_RINT 就近偶数舍入，与 numpy/torch 的 bf16 转换一致）；
5. **CopyOutQK**：每 token 一次 3 参 DataCopyPad（UB→GM）写回连续输出；
6. **Gate/V 直通**：Gate 逐 chunk 拆分搬入（MTE2），免 V 计算直接 MTE3 写回；V 整块搬入写出；两处 pass-through 均以 MTE2→MTE3 与 MTE3→MTE2 事件同步保证直通数据一致性；
7. **屏障纪律**：所有跨指令 RAW 依赖处显式 `PipeBarrier`；scalar 写（axis 掩码）后 `S_V` 事件同步；关闭 auto-sync 后必须保留显式同步（删光会静默读到未就绪数据）。

### 性能优化方案

按「先测量、再动手」推进，msprof 数据是每一步决策的依据。

| # | 优化项 | 依据（msprof） | 效果 |
| - | ------ | -------------- | ---- |
| ① | tiling 字段缓存进寄存器 | `td_` 在 GM 上，热循环每次解引用都是一次 GM 读，`aiv_scalar_time` 占 38% | -10us（400.3→391.0） |
| ② | tileTokens 4 → 8 | 每 tile 的队列 Alloc/EnQue/DeQue/Free 约 40~60 次 + cos/sin 构建 + 事件同步，都是 per-tile 固定开销；scalar 占 38% 主要来自这里 | t8192 457.9→365.8（-20.1%），scalar 141→100.5us（-29%） |
| ③ | dense rstd 平面 + 单次 count-form Mul | per-row Mul 在 vec 里是 55% 的大头，而 mte2 有富余 | vec 204.7→190.1us |
| ④ | **gate/V 搬运交错进 Q/K 循环** | `mte2 187 + vec 190 > aiv_time 347` ⇒ 两条管道几乎**零重叠**：原实现把 gate 循环与 V 搬运排在 Q/K 循环**之后**，那 101MB 搬运期间 vec 完全空闲 | **t8192 354.9→311.9us（最大单项收益）** |

优化 ④ 的改法（零 UB 代价，纯调度）：gate 的 MTE2 读置于 `ComputeQK` **之前**、MTE3 写置于 `ComputeQK` **之后**，V 的读提到 Q/K 循环外先行发出。这样 gate/V 的 101MB 搬运全部藏进 vec 的忙碌期；gate 队列仍为 depth 1（读写在同一次迭代内完成，store 无需等待）。

最终结果（910B2，AIV 核 48，全量 7 case）见 §「精度标准/性能标准」。

### 已验证但无效的方案（工程记录，避免重复踩坑）

| 方案 | 结果 | 结论 |
| ---- | ---- | ---- |
| 输出 staging（每 tile 合并成一次 ≥16KB 连续写） | 399.6 vs 400.3us | 无收益，且吃 64KB UB 反压 tileTokens → 已撤除 |
| BUFFER_NUM=3 + 两拍预读（三段流水） | 无收益，且两拍预读会 `AllocTensor` 死锁 | 已回退到 depth 2 |
| 三次 buffer / 输出延迟一拍 | 365.64 vs 365.83us | MTE3 仅占 14%，串行化它无代价，不值得增加复杂度 |
| 整行 256 元素一次 `WholeReduceSum` | 全 case q/k FAIL | **`WholeReduceSum` 的 count 上限是 64**，传 256 只归约部分元素 → 分母错；gate/v 直通不受影响（症状可据此判别） |
| `Rsqrt` 替代 Sqrt+Div | decode_t1 从 bit-exact 变成 max_diff=0.0156 | AscendC 的 `Rsqrt` 与 torch 的 `rsqrt` 不是同一实现，**不要用** |
| 把 eps 折进前置缩放 `(sum+256ε)·(1/256)` | 无变化 | 数学等价但舍入路径改变，未能翻转边界元素 |
| 归约折叠配对用「连续 64 块」`(s0+s1)+(s2+s3)` | t2048_kv4 的 q 残留 1/8388608 个元素、1 个 bf16 ULP | torch_npu 的 `ReduceMean` 是**跨步**折叠 `(s0+s2)+(s1+s3)`；把重复步长由 8 改为 32 blocks（一整行）后**7/7 全部 bit-exact** |

**关键教训**：`npu` 侧"挂死"要先在代码层确认无死锁可能（空队列 DeQue 自旋、队列深度 vs 预读深度），确认后再考虑 UB/硬件因素——本项目两次"疑似 UB 溢出挂死"最终都定位为漏写一次预读种子导致空队列死等。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2 | √ |
| Atlas 800I A2 | √ |
| Atlas 900 A3 (ascend910_93) | √（同源实现验证） |

> 说明：任务书 §3.1 指定「适配硬件：Atlas 800T A2」，未细分 910B2/910B3/910B4；本实现的 tiling 在 host 侧查询实际设备的 AIV 核数与 UB 容量，不依赖特定 SKU 常量。

## 算子约束限制

- head_size = 256（Qwen3-VL 固定），rope_dim = 64，mrope_section = [11,11,10]；
- 输入行宽必须等于 q_size + gate_size + 2*kv_size；
- 仅支持 ND 连续输入（AutoContiguous）；
- 自身业务 workspace 需求为 0，无 GM 中间结果（单 kernel 内完成全部计算）；aclnn 返回的 `16777216` 字节
  是 host 侧 tiling 登记的**固定系统预留** `WS_SYS_SIZE`（16MB，与 shape 无关，实测 7 个用例恒定），
  kernel 形参 `GM_ADDR workspace` 未被引用。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | q/k 与 Triton 版逐元素 max_diff < 1e-3；v/gate 直通 bit 级一致；任务 case.json 全部 7 用例通过（对拍任务 golden.py） | 任务书 3.2 |
| 性能标准 | 7 用例**逐一** NPU 运行时延 < Triton baseline/2 | 任务书 3.3 |
| 内存标准 | 不涉及 | 任务书 3.4 |

**实测结果（Atlas 800T A2 / 910B2，CANN 9.1.0，AIV 核 48 / AICore 24，warmup 10 / iters 100，aclrtEvent 计时）**：

| case | Triton baseline (us) | 达标线 baseline/2 | Ascend C 实测 (us) | 加速比 | 精度 |
| --- | ---: | ---: | ---: | ---: | --- |
| decode_t1_q16_kv4 | 636.240 | 318.120 | **7.025** | 90.6× | PASS（bit-exact） |
| small_t16_q16_kv4 | 643.246 | 321.623 | **8.748** | 73.5× | PASS（bit-exact） |
| medium_t128_q16_kv4 | 678.923 | 339.462 | **14.410** | 47.1× | PASS（bit-exact） |
| prefill_t2048_q16_kv4 | 641.700 | 320.850 | **62.269** | 10.3× | PASS（bit-exact） |
| prefill_t8192_q16_kv4 | 669.499 | 334.750 | **302.241** | **2.22×** | PASS（bit-exact） |
| decode_t1_q16_kv2 | 640.793 | 320.397 | **6.914** | 92.7× | PASS（bit-exact） |
| prefill_t2048_q16_kv2 | 669.931 | 334.966 | **58.200** | 11.5× | PASS（bit-exact） |

**性能：7/7 全部达标。** 端到端口径（host 墙钟，含 `GetWorkspaceSize` + 执行）仅多约 1us，同样全部达标。

**精度：7/7 全部 bit-exact** —— q/k/v/gate 四路输出与任务 `golden.py` 逐元素**完全一致**（`max_diff = 0`），无任何超标元素。

### 归约折叠配对修正（早前 1 ULP 残留的根治）

早前版本（commit `4125b4d9`）在 `prefill_t2048_q16_kv4` 的 q 上曾有 1/8388608 个元素差 1 个 bf16 ULP
（`0.00195312`）。当时的逐段 fp32 实证已确认算法无误——**variance、rstd 与 golden 完全一致（diff = 0）**，
rope 窗口的 fp32 输出与 golden 等价实现**逐位相同**——差异只在归约末位。进一步定位过程：

1. 用 `msprof` 包住 `mean(dim=-1)` / `rsqrt`，确认 torch 侧走 `ReduceMean` 与 `Rsqrt`；
2. 以**可分离输入**构造探针（极端指数铺开 / 大数吸收 / 四段量级 / 贴近真实的平方），穷举 12 种归约结构
   并与 `torch.mean` **逐位**比对：**跨步折叠 + 64 线二分树** 的候选在 10000 行探针上不一致数为 **0**，
   其余候选最少 201 个；
3. CANN 源码 `reduce_common_regbase.h` 的 `CalculateSquareReduceSumCommon<..., DICHOTOMY_ADD_COEFF>`
   独立佐证「二分（配对树）」结构。

⇒ 病因是**归约的折叠配对方式**：原设计按连续 64 块配对 `(s0+s1)+(s2+s3)`，而 torch_npu 的 `ReduceMean`
是跨步配对 `(s0+s2)+(s1+s3)`（128→64 折叠）。把该段的重复步长由 8 blocks 改为 32 blocks
（= 一整行 256 个 fp32）后，三条 `Add` 成为每行自身槽位内的逐元素就地运算（**不新增任何 UB 缓冲**）；
`WholeReduceSum` 本身即为配对树，故只需修正折叠配对。

**验证方式为同会话 A/B 对照**（同一台机、同一份输入、同一份 `golden.py`，仅 kernel 不同）：
对照组（`db4d768d`，kernel 同 `4125b4d9`）复现 `NG q FAIL(max_diff=0.00195312)`，
修复组四路输出全为 0；性能不变（±0.5% 量级）。

## 兼容性分析

新算子，不涉及兼容性分析。

## 自测方法

`tests/run_st.py`：以固定种子（`PYTHONHASHSEED=0`）按 `case.json` 生成全部用例输入 → 任务 `golden.py` 在 NPU 计算真值 → `examples/test_aclnn_split_qkv_rmsnorm_mrope.cpp` 加载相同输入运行 → 逐元素比对（q/k 报 max_diff，v/gate 报位级一致性）并输出每用例平均时延（aclrtEvent 计时，warmup 10 / iters 100）。复现步骤见 `tests/README.md`。

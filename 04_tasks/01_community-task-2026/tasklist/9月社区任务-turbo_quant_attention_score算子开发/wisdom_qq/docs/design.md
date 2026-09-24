# TurboQuantAttentionScore 算子设计文档

> 任务来源：昇腾社区 2026 年 9 月社区任务「turbo_quant_attention_score 算子开发」
> 目标仓库：https://gitcode.com/cann/ops-transformer （合入 experimental/attention）
> 关联算子：KvCacheTurboQuant（编码侧，同批社区任务，设计文档已评审提交）

---

# 1. 需求背景（required）

## 1.1 需求来源

vllm-ascend 推理框架在长序列场景下 KV cache 显存占用成为主要瓶颈。同批社区任务「kv_cache_turbo_quant 算子开发」已实现 TurboQuant 在线向量量化的**编码侧**（KV 向量 → bit-packed 主码 + QJL 残差符号 + 两个范数标量，head_dim=128、3-bit 配置下 68 bytes/head，理论显存压缩比约 3.76x）。

本任务实现配套的**解码/attention 侧**算子：在标准 MHA/GQA attention 计算时，**直接基于 TurboQuant 编码的 Paged Key cache 计算 attention score**——不完整反量化 Key 向量到原空间，而是在旋转空间中按需解码主量化分支（查表反量化后点积），并叠加 QJL 残差修正项。本算子不覆盖 MLA，也不负责 quantized Value cache 的解码与 value aggregation（输出 score 之后接 softmax + V matmul，均不在本算子范围）。

目标仓库已有直接对标的参考算子 `experimental/attention/turbo_quant_sparse_attn_sharedkv`（4bit TurboQuant 解压与 sparse attention 融合：vector 核将压缩 slot 搬入 UB、按固定码本批量解压后接 matmul 流水，paged layout + block_table 寻址，支持 A2/A3），本算子的工程结构与解压流水以其为主要参照；差异在于本算子为 3bit（非对齐位宽）+ 1bit QJL 双分支编码、且仅输出 score 不含 softmax/V 聚合。

## 1.2 背景介绍

### TurboQuant 编码格式回顾（编码侧算子产物，即本算子输入）

对每个 head_dim=128 的 Key 向量 x，编码侧输出四元组（详见 KvCacheTurboQuant 设计文档）：

| 组件 | 存储 | 说明 |
|---|---|---|
| idx（主量化码） | 48 bytes（mse_bits=3，bit-packed） | 旋转空间 y = H·u 逐维 Lloyd-Max 量化索引；每 8 维一组打包为 24bit 字（小端），16 组 × 3B |
| qjl（残差符号） | 16 bytes（1bit×128，bit-packed） | 残差 r = y − ŷ 经高斯投影 S·r 的符号位，每 8 维 1 字节 |
| x_norm | 2 bytes（bf16） | 原向量 L2 范数 |
| gamma | 2 bytes（bf16） | 原尺度残差范数 x_norm·‖r‖ |

### score 计算原理

query 侧在每个 decode step 由调用方预计算两个投影（**不在本算子范围**，直接作为输入给出）：

```
Hq        = H @ q          # query 旋转，[head_dim]
query_qjl = S @ Hq         # 旋转后 query 的 QJL 投影，[qjl_dim]
```

对每个 cached key：

```
score = score_mse + score_qjl
score_mse = x_norm · <Hq, centroid[idx]>                    # 主量化分支：查表反量化后点积
score_qjl = (sqrt(pi/2) / qjl_dim) · gamma · <query_qjl, sign(qjl)>   # QJL 残差修正分支
```

### 适用范围

- 仅支持标准 MHA/GQA（num_q_heads / num_kv_heads 为整数），不支持 MLA attention；
- Paged KV Cache 布局，经 block_table 间接寻址，block_size = 128（与 vllm-ascend 当前配置一致）；
- 论文中 3.5 bit/channel 混合精度模式不纳入范围。

# 2. 需求分析（required）

## 2.1 需求描述

使用 Ascend C 编程语言，基于 aclnn 算子工程化模式，在 Atlas 800T A2（CANN 9.1.0+）上实现 TurboQuantAttentionScore 算子：输入预旋转 query、QJL 投影 query 与 TurboQuant 编码的 Paged Key cache，输出 fp32 attention score（超出各请求 seq_len 的位置填 -inf）。

## 2.2 需求拆解

1. 支持 Paged KV Cache 布局：block_table [max_batch, max_blocks_per_seq]（int32）查物理块，block_size 固定 128，seq_lens [max_batch]（int32）控制有效长度；
2. 支持 MHA/GQA：num_q_heads / num_kv_heads 为整数（官方 case 为 32/8，group=4）；
3. 支持 mse_bits = 2/3/4 三种主量化位宽（attr，默认 3）：主码先按 mse_bits 解包，再查质心表映射为 fp32 centroid；
4. QJL 分支为实数 query_qjl 与 {-1,+1} 符号向量内积：可用符号选择 + 向量加减优化，**不能**等价为两个 sign bit-vector 的 popcount；
5. 输出 attn_scores fp32 [num_q_tokens, num_q_heads, max_kv_len]，positions ≥ seq_len 处为 -inf；
6. 精度：与 golden（torch_npu 参考，fp32 全程计算）对比 err_threshold [1e-3, 0]；端到端 softmax 后 attention weight 与全精度版本 KL 散度 < 0.01、PPL 增长 < 1%（与编码侧算子联合测试口径）；
7. 性能：decode 场景整体 attention score 计算（含反量化）延迟 ≤ 全精度 Paged Full-QK baseline 的 1.2 倍，显存占用降低 3x 以上。

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

对每个 (q_token t, q_head h, kv 位置 i)，记 kv_head k = h / group（group = num_q_heads / num_kv_heads）：

```
phys_block  = block_table[batch(t), i / 128]
offset      = i % 128
idx[0:128]  = unpack_mse(key_cache_idx[phys_block, offset, k], mse_bits)   # 见下
sgn[0:128]  = unpack_sign(key_cache_qjl[phys_block, offset, k])            # bit→±1
ŷ           = centroids[idx]                                               # fp32 查表
score_mse   = key_cache_norm[phys_block, offset, k]  · Σ_d Hq[t,h,d]·ŷ[d]
score_qjl   = (sqrt(pi/2)/128) · key_cache_gamma[phys_block, offset, k] · Σ_d query_qjl[t,h,d]·sgn[d]
score[t,h,i] = i < seq_lens[batch(t)] ? score_mse + score_qjl : -inf
```

**bit 解包格式**（与编码侧/golden 逐字节一致，无自由发挥空间）：

- **主码 idx**：48 bytes 视为 16 个 24bit 字 w（小端：w = b0 | b1<<8 | b2<<16），每个字拆 8 个 3bit 索引（`idx[lane] = (w >> 3*lane) & 7`），共 128 维；mse_bits=2/4 时为每字节 4/2 个索引的规则打包；
- **QJL 符号**：16 bytes，每字节 8 个 bit，`sgn = 2·bit − 1` ∈ {-1,+1}。

所有点积与加权累加全程 fp32（golden 为 fp32 计算，err_threshold 1e-3 下 fp32 累加充分）。

### 3.1.2 支持数据类型

| 参数 | 方向 | 数据类型 | shape |
|---|---|---|---|
| query_rotated | 输入 | bfloat16 | [num_q_tokens, num_q_heads, 128] |
| query_qjl | 输入 | bfloat16 | [num_q_tokens, num_q_heads, 128] |
| key_cache_idx | 输入 | uint8 | [num_blocks, 128, num_kv_heads, ceil(128·mse_bits/8)] |
| key_cache_qjl | 输入 | uint8 | [num_blocks, 128, num_kv_heads, 16] |
| key_cache_norm | 输入 | bfloat16 | [num_blocks, 128, num_kv_heads] |
| key_cache_gamma | 输入 | bfloat16 | [num_blocks, 128, num_kv_heads] |
| block_table | 输入 | int32 | [max_batch, max_blocks_per_seq] |
| seq_lens | 输入 | int32 | [max_batch] |
| attn_scores | 输出 | float32 | [num_q_tokens, num_q_heads, max_kv_len] |
| mse_bits | attr | int，默认 3，支持 2/3/4 | — |

注：任务书参数表中 norm/gamma 标注「float16 / bfloat16」，官方 op.json 定死为 bfloat16，首版按 op.json 口径仅支持 bf16（与编码侧算子输出 dtype 一致）。

### 3.1.3 支持形状

num_q_tokens ≥ 1（decode 单 token 至 prefill 数百 token），num_q_heads ∈ [4, 32]、num_kv_heads ∈ [4, **16**] 且整除（num_kv_heads 上限由 kernel 的 norm/gamma 平面 UB 常驻预算决定，见 3.2.1；官方 case 为 32/8），head_dim 与 qjl_dim 固定 128，block_size 固定 128。输出 max_kv_len = max_blocks_per_seq × 128（官方 case 中即等于各请求 seq_len）。

注：golden.py 内部按 `max_kv_len = seq_lens.max()`（device tensor 值）动态确定输出长度；而 aclnn infershape 在 host 侧执行、无法读取 device 上的 seq_lens 内容，故本算子输出第三维只能按静态信息 `max_blocks_per_seq × block_size` 推导。官方 5 个 case 中 seq_len 恰为 block 数 × 128，两种口径的输出 shape 完全一致，验收无影响；非整块场景由 kernel 内 -inf masking 兜底（见 3.2.2），实际有效长度仍以 seq_lens 为准。

## 3.2 算子实现

### 3.2.1 host 侧设计

**输出 shape 推导**：attn_scores = [num_q_tokens, num_q_heads, max_blocks_per_seq × 128]。

**tiling 策略**：

1. **并行维度选取**：每个输出元素 score[t,h,i] 相互独立（沿 kv 维无归约），为 embarrassingly parallel。按 (q_token t, kv_head k, kv block 段) 三级展平为 work item 均分到各 AI Core；同一 work item 内处理 group 个 q_head 与一段连续 kv block。各规模并行度：decode kv4k = 1×8×32 = 256 个 work item，kv128k = 8192 个，prefill 再乘 512 个 q_token——所有 case 均可满核且负载天然均衡（每 key 计算量固定）；
2. **UB 切分（实测口径，Atlas 800T A2，192KB UB）**——注意：**瓶颈不是原始打包数据（每 key 仅 68B），而是解包后的中间平面**。以 TILE_K = 16 个 key 为一批的实际缓冲（约 165KB）：
   - block_table / seq_lens 常驻：32KB（上限 8192 条 × 4B）/ 4KB；
   - query 常驻：2 路（Hq、S(Hq)）× group ≤ 8 × 128 维 × 4B = 8KB，另加**维度置换缓冲** 8KB（见 3.2.2）；
   - 打包数据：idx 128×64B = 8KB（48B 行按 32B 对齐占 64B）、qjl 128×32B = 4KB；
   - norm/gamma：bf16 原平面各 4KB + fp32 平面各 8KB（上限由 num_kv_heads ≤ 16 决定）= 24KB；
   - 解包中间平面（TILE_K=16：16×128 = 2048 元素）：idxInt / signInt / idxF / yhat / sign / prodA / prodB 共 7 个 8KB 平面 = 56KB，另 word/byte/scratch 平面各 1KB；
   - 输出 tile：group × 128 × 4B ≤ 4KB；另有 dup/mask 等小缓冲；
   - 若把 TILE_K 提到 32，中间平面翻倍（+64KB）会超出 UB，故维持 TILE_K=16；
3. **tilingkey 规划**：按 attr mse_bits（2/3/4）设 3 个 tilingkey，编译期固化解包位宽与质心个数，消除 kernel 内分支；
4. **输入校验**：维度/dtype/格式按 op.json 校验；num_q_heads % num_kv_heads == 0 且 num_kv_heads ≤ 16；末维 128；mse_bits ∈ {2,3,4}；batch > 1 时须 batch == num_q_tokens（decode 语义）；block_table 总条目 ≤ 8192；仅支持 ND 连续 tensor。

**aclnn 工程化接口**（按 ops-transformer experimental/attention 既有算子规范）：标准两段式 `aclnnTurboQuantAttentionScoreGetWorkspaceSize(...)` + `aclnnTurboQuantAttentionScore(...)`；无全局中间 tensor，workspace 为 0。

### 3.2.2 kernel 侧设计

Init / Process 两段式，Process 内按 work item 循环 CopyIn → Compute → CopyOut：

1. **CopyIn**：block_table / seq_lens 小段搬入（每核仅涉及本 batch 行）；每个 kv block 按主流 paged 惯例做地址翻译——标量侧查 `phys = block_table[b, logical]` 算 GM 偏移、再按 128-key 整块搬运（同构参考：vllm-ascend SparseAttentionScore 的 paged 地址翻译设计；GQA 下 group 个 q_head 共享同一次 KV 搬运）；idx/qjl/norm/gamma 用 DataCopyPad 搬入。DataCopyPad 对齐注意（48B/16B 行均非 32B 整数倍）：`blockLen` 以**字节**为单位；pad 补零字节数上限 255B；**补零字节会进入 UB 参与后续计算**——本算子点积为求和语义、且解包仅取有效 bit 位（24bit 字内移位、16B 内逐 bit），补零位天然被排除在有效 lane 之外，需在实现中保持该不变式；query 在 work item 起始一次性搬入并转 fp32 常驻；
2. **Compute（实测版本，按 TILE_K = 16 个 key 一批，fp32 全程）**：
   - **主码解包（向量化，实测关键路径）**：S 域先把每 key 的打包字节按 3 字节（mse_bits=3）拼成 16 个 24bit 小端字，写入「字平面」`[key][word]`（每 key 12 次 u32 读 + 16 次写，较逐维写法省约 5× 标量 UB 访问）；随后矢量按 lane 取位：`ShiftRight` → `ShiftLeft` → `ShiftRight`（**CANN 的 `And` 在 dav_c220 上仅支持 16bit 整型，故用双移位取位**），每 lane 结果按 16 个连续元素写回 `[key][l*16+w]` 形式的维度序平面；
   - **维度重排（消除 scatter 的关键设计）**：主码第 l lane、第 w 字对应原维度 `d = 8w+l`。把 ŷ 平面按 `d' = l*16+w` 排布、并在 `LoadQuery` 时把 query 一次性置换到同一顺序（QJL 同理按 `d'' = l*16+b`），则 lane 提取天然落在连续地址上，无需跨步 scatter；点积与顺序无关，故数值不受影响；
   - **质心查表**：整 tile 平面做 `Compare`（EQ）+ `Select` 链（与编码侧算子同一形态）；
   - **QJL 解包**：每 key 16 字节取位 → 0/1 → `Cast` 转 fp32 后 `Muls(2)+Adds(-1)` 得 ±1，同样按 `d''=l*16+b` 落位；内积即实数乘法（等价于符号加减，且满足任务书「不得等价为两个 sign bit-vector 的 popcount」的约束）；
   - **两路点积**：逐 key 行 `Mul` + **`TreeReduce8`**（Add 64/32/16/8 折叠到 8 个 32B 对齐的部分和）+ S 域收拢 8 个部分和；fp32 全程；
   - **加权合成**：score = norm·main + (sqrt(pi/2)/128)·gamma·corr，norm/gamma 由 bf16 转 fp32；
   - **masking**：无效位置填**真 IEEE -inf**（fp32 位模式 0xFF800000，用 `Duplicate<float>` 直接写常量），与 golden 的 `-torch.inf` 逐位一致——注意 fp32 `lowest()`（-3.4e38）≠ -inf，不可混用；输出 tile 先填 -inf 再写回 i < seq_len 的有效位置，尾块（seq_len 非 128 整数倍）同样处理；精度比对脚本对 -inf 位置按位模式特判；
3. **CopyOut**：fp32 score 按 [t, h, kv段] 写出（每 (t,h) 行 512B 整块 `DataCopyPad`）。
4. **同步纪律（TBuf 跨 item/tile 复用的必备防护）**：CopyIn 前补 `V/S→MTE2`、`LoadQuery` 前补 `V→MTE2`、CopyOut 后补 `MTE3→V`，并由显式 `SetFlag/WaitFlag` 覆盖 S↔V 边界（缺同步会出现「有效位置被 -inf 覆盖」的确定性错图）。

#### 3.2.2.1 与 CANN 版本相关的取舍（重要）

CANN 9.1.0 起官方调整了归约类接口：**`BlockReduceSum` 改名为 `ReduceDataBlock` 并调整 `mask`/`repeatTime` 参数顺序**（同名误用会静默错算）。本算子开发机为 CANN 9.0.0、任务书要求验收环境 9.1.0+，因此**点积归约只采用纯算术实现（`TreeReduce8`），不依赖任何版本敏感的归约 API**；位运算只用跨版本稳定的 `ShiftRight/ShiftLeft(uint32)`（A2 支持矩阵明确），规避迁移风险。

**DataCopyPad 单位口径**（官方 A2 参数表，host 侧易错点）：GM 侧 `srcStride` 单位为**字节**、UB 侧 `dstStride` 单位为 **32B 块**；UB 侧每行长度非 32B 对齐时硬件自动右侧补零至 32B 对齐 —— 故 48B 的 idx 行在 UB 中占 64B（本实现按该 pitch 索引），16B 的 qjl 行占 32B。

**计算实现选型**：vector 单元，不引入 cube。依据（含实测）：

- decode 为访存 bound：kv128k 全量编码数据约 71MB（vs 全精度 bf16 268MB）；
- 但 **baseline 是朴素 paged 实现**：按任务书数值反推其有效带宽仅约 8 GB/s（kv128k 读 268MB 用 35682µs），相差 HBM 地板两个数量级，因此 1.2× 的实际含义是**绝对时间线**（kv4k ≤ 1802µs、kv32k ≤ 11134µs、kv128k ≤ 42818µs），而非「随便写都过」——实测也印证：S 域逐元素解包版本会超线 1.5~1.8×；
- 实测（910B3，CANN 9.0.0，本实现）：decode kv4k **1380µs（0.92×）**、kv32k **9988µs（0.90×）**、kv128k **39333µs（0.92×）**，均在 1.2× 预算内；
- prefill（512 q token）任务书未设性能条款，本版本仅保证精度（kv4k 628.8ms、kv32k 5.02s）；若后续需要 prefill 性能，再评估 Cube 批量路径（scores = Hq @ Ŷ^T）。

### 3.2.3 质心表与系数（编译期常量）

```
mse_bits=2: ±0.04002048075, ±0.1335033178                                   (4 质心)
mse_bits=3: ±0.02166347019, ±0.06682205945, ±0.1187859178, ±0.19020693      (8 质心)
mse_bits=4: ±0.01134236995, ±0.03428063914, ±0.05802082643, ±0.08325428516,
            ±0.1109927073, ±0.1429702938, ±0.1828317791, ±0.2414890379      (16 质心)
QJL 系数：sqrt(pi/2) / 128 ≈ 0.0097915167（fp32 编译期常量）
```

质心常量按与 golden 相同 double 字面量经 round-to-nearest 转 fp32 的位模式硬编码（与编码侧算子同一份常量定义，已验证一致）。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas 800T A2 | √ |

## 3.4 算子约束限制

- head_dim 与 qjl_dim 首版固定 128（质心表维度相关）；block_size 固定 128；
- 仅支持标准 MHA/GQA，不支持 MLA；仅计算 score，不含 softmax 与 V cache 解码/聚合；
- query 旋转（Hq = H@q）与 QJL 投影（query_qjl = S@Hq）由调用方预计算，不属于本算子；
- norm/gamma 首版仅支持 bfloat16（依据 op.json，见 3.1.2）；
- attr mse_bits 仅支持 2/3/4，其余取值 host 侧直接返回参数错误；
- 仅支持 ND 格式连续 tensor；
- num_kv_heads ≤ 16（kernel norm/gamma 平面 UB 常驻预算所限，见 3.2.1）；batch > 1 时按 decode 语义要求 batch == num_q_tokens；block_table 总条目数（max_batch × max_blocks_per_seq）≤ 8192。

# 4. 可维可测分析

## 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | ① 与 golden 输出对比 err_threshold [1e-3, 0]（含 -inf 位置精确一致）；② softmax 后 attention weight KL 散度 < 0.01（联合口径）；③ PPL 增长 < 1%（与编码侧算子联合测试） | 任务书 3.2 节、case.json |
| 性能标准 | decode 场景 score 计算（含反量化）延迟 ≤ Paged Full-QK baseline × 1.2；显存占用降低 3x 以上（默认配置理论压缩比约 3.76x） | 任务书 3.3 节 |

## 4.2 本地预验证方案（先行对拍）

上真机前先用 numpy/torch 独立复现 kernel 语义并与 golden.py 对拍（复用编码侧任务已建的对拍脚本框架），锁定三个易错点：

1. 3bit 解包位级正确性（24bit 字小端拼接、lane 位移），对拍 5 个官方 case 全部输入组合；
2. QJL 符号解包（bit→±1）与两路加权系数（sqrt(pi/2)/128）；
3. -inf masking 边界（含尾块、seq_len 非整块），-inf 位置按位模式（0xFF800000）特判比对。

**已完成验证（截至 2026-09-24）**：

- **步骤一（离线，numpy 双路径）**：golden 语义复现（24bit 字解包）与 kernel 语义复现（独立的小端比特流解包 + pairwise 归约树）互验，覆盖 5 个官方 case + mse_bits=2/4 + 尾块变长 batch 共 8 个用例，全部 PASS；有效位置最大绝对偏差 ≤ 1.4e-6（fp32 累加顺序噪声），**-inf 位置位模式 0/156064 不一致**。
- **步骤二（真机 A2 910B3，CANN 9.0.0）**：自建 aclnn 测试程序（host golden 与本 kernel 同算法）在 8 个用例上全部 PASS，最大绝对偏差 ≤ 9.5e-7，-inf 位置位模式零不一致；decode 性能如 3.2.2.1 所列（0.90~0.92×）。

## 4.3 自测用例规划

| case | num_q_tokens | kv_len | 类型 | 说明 |
|---|---:|---:|---|---|
| gqa_decode_kv4k / kv32k / kv128k | 1 | 4096 / 32768 / 131072 | 精度 + 性能 | 官方 case |
| gqa_prefill_kv4k / kv32k | 512 | 4096 / 32768 | 精度 + 性能 | 官方 case |
| mse2 / mse4 位宽 case | 1 / 512 | 4096 | 精度 | 覆盖 3 个 tilingkey |
| seq_len 非 128 整数倍 / 变长 batch | 4 | 如 5000 | 精度 | masking 边界 |

交付件按 ops-transformer 仓 experimental/attention 目录规范组织（参照同目录既有 experimental 算子，含同批的 kv_cache_turbo_quant）：算子根目录 = `CMakeLists.txt` + `README.md`（产品支持情况 / 功能说明 / 参数说明 / 调用说明 / 贡献说明五个必备章节）+ `docs/aclnnTurboQuantAttentionScore.md`（接口文档）+ `op_host/`（`*_def.cpp` 原型、`*_infershape.cpp` 推导、`*_tiling.cpp/.h` 与 tiling data；aclnn 两段式接口由 `add_modules_sources(OPTYPE ... ACLNNTYPE aclnn)` 按 OpDef 自动生成，无需手写 op_api）+ `op_kernel/`（kernel 入口 + kernel 头 + tiling data）+ `examples/`（设备端精度/性能自测程序）。构建：`source <CANN>/set_env.sh && bash build.sh --experimental --ops=turbo_quant_attention_score`（A2 对应 arch22；**experimental 目录算子必须带 `--experimental`**，否则构建期即报 cust_proto 无源码）。

## 4.4 与任务方确认的开放问题

| # | 事项 | 影响 | 应对措施 |
|---|---|---|---|
| 1 | prefill_kv32k 的 baseline 微秒数在任务书中空缺 | 该 case 的 1.2× 上限无直接数值 | 按 prefill_kv4k baseline 随 kv_len 线性外推（×8）自证；性能余量大，不影响达标结论；正式口径向任务方索取 |
| 2 | case.json 引用的 block_table/seq_lens bin 数据文件未随任务下发 | 正式验收环境数据无法本地逐字节复现 | 自测阶段按 value_range 自行生成（golden 现算期望值，不依赖固定输入）；正式验收以任务方数据为准 |
| 3 | KL 散度 / PPL 为「配合算子 2 联合测试」指标，测试程序与口径未提供 | 单算子验收口径外指标不可独立复现 | 以编码侧已交付的联合链路复用；向任务方确认联合测试程序与判定口径 |
| 4 | err_threshold [1e-3, 0] 的判定公式未明确：本地对拍（golden 双路径独立复现）实测 score≈0 的位置 max_abs ≈ 1.4e-6（fp32 累加顺序噪声），但 max_rel 可达 10 量级——若被解读为「纯相对误差、零绝对容差」，任何实现（含 golden 自身更换累加顺序）在零值附近都无法满足 | 验收判定口径不可预期 | 预期框架为 `abs <= atol + rtol*|ref|` 组合判定；M4 真机首跑时优先用官方 case 确认实际判定公式，若为纯相对口径则与任务方确认容差解释 |

# 5. 兼容性分析

新算子，不涉及兼容性分析。

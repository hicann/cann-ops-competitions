# TurboQuantAttentionScore 算子设计文档

## 1 需求背景（required）

### 1.1 需求来源
华为昇腾 CANN 社区任务（广州站）——算子实操工坊。本任务要求在 Atlas 800T A2 上，使用 Ascend C 编程语言开发 TurboQuantAttentionScore 算子，并合入昇腾算子开源仓（ops-transformer）。

### 1.2 背景介绍

#### 1.2.1 TurboQuantAttentionScore 算子实现优化
基于 TurboQuant 量化方案，在标准 MHA/GQA attention 计算时直接基于量化后的 Key cache 计算 attention score，无需完整反量化 Key 向量到原空间。该算子在长上下文推理场景中可显著降低显存占用（3x 以上），同时保持计算精度。

#### 1.2.2 TurboQuantAttentionScore 算子参考实现现状分析
当前 `ops-transformer` 开源仓中尚无 TurboQuant 相关算子实现。本算子需参照仓内现有 attention 算子工程框架，实现 aclnn 算子工程化开发，完成从算子原型定义、Tiling 设计到 Kernel 实现的完整流程。

#### 1.2.3 TurboQuantAttentionScore 算子功能分析
**输入**：
- `query_rotated` (bfloat16): 预旋转后的 query (Hq)，shape `[num_q_tokens, num_q_heads, head_dim]`
- `query_qjl` (bfloat16): 旋转后 query 的 QJL 投影 S(Hq)，shape `[num_q_tokens, num_q_heads, qjl_dim]`
- `key_cache_idx` (uint8): 主量化编码（bit-packed），shape `[num_blocks, block_size, num_kv_heads, ceil(head_dim*mse_bits/8)]`
- `key_cache_qjl` (uint8): 残差 QJL 编码（bit-packed），shape `[num_blocks, block_size, num_kv_heads, ceil(qjl_dim/8)]`
- `key_cache_norm` (bfloat16): Key 向量 L2 范数，shape `[num_blocks, block_size, num_kv_heads]`
- `key_cache_gamma` (bfloat16): Key 原尺度残差范数，shape `[num_blocks, block_size, num_kv_heads]`
- `block_table` (int32): Paged block 映射，shape `[max_batch, max_blocks_per_seq]`
- `seq_lens` (int32): 各请求 KV 序列长度，shape `[max_batch]`

**输出**：
- `attn_scores` (float32): Attention 分数，shape `[num_q_tokens, num_q_heads, max_kv_len]`

**属性**：
- `mse_bits` (int, 默认 3): 主量化比特数，支持 2/3/4

**对标基线**：全精度 `Q @ K^T` 的 FlashAttention score 计算。

## 2 需求分析（required）

### 2.1 需求描述
使用 Ascend C 编程语言实现 TurboQuantAttentionScore 算子，支持 Paged KV Cache 布局、GQA 分组、bit-packed 主量化与 QJL 残差解码，在旋转空间中直接计算 attention score，满足精度与性能要求。

### 2.2 需求拆解
1. **功能实现**：
   - 按 `mse_bits` 解包主量化编码，通过查表映射到 centroid，计算 `score_mse = x_norm * <Hq, dequant(idx)>`。
   - 解包 QJL 符号编码，计算 `score_qjl = coeff * gamma * <S(Hq), qjl>`。
   - 合并 `score = score_mse + score_qjl`，无效位置填 `-inf`。
2. **Paged KV Cache 支持**：通过 `block_table` 将逻辑 block 映射到 physical block，支持变长序列。
3. **GQA 支持**：`num_q_heads / num_kv_heads` 为整数，query 按 group 展开。
4. **工程化开发**：实现 aclnn 算子工程，包含算子原型、Tiling、Kernel、Host 侧接口。
5. **性能达标**：Decode 场景延迟 ≤ 全精度 FlashAttention 的 1.2 倍，显存占用降低 3x 以上。

## 3 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式
对每个 query token `q`，计算与所有 cached key 的 attention score：

$$
\text{score}[i] = \text{score\_mse}[i] + \text{score\_qjl}[i]
$$

其中：

$$
\text{score\_mse}[i] = x\_norm[i] \cdot \langle Hq, \text{dequant}(idx[i]) \rangle
$$

$$
\text{score\_qjl}[i] = \frac{\sqrt{\pi/2}}{qjl\_dim} \cdot \gamma[i] \cdot \langle S(Hq), qjl[i] \rangle
$$

预计算（每个 decode step 一次）：

$$
Hq = H \cdot q
$$

$$
query\_qjl = S \cdot Hq
$$

- `H` 为旋转矩阵，`S` 为 QJL 投影矩阵。
- `dequant(idx)` 通过 centroid 查表实现，centroid 表由 `mse_bits` 决定。
- QJL 符号向量元素为 `{-1, +1}`，由 bit-packed 编码解包得到。
- 无效位置（position ≥ seq_len）输出 `-inf`。

#### 3.1.2 支持数据类型
| 参数 | 数据类型 |
|---|---|
| query_rotated | bfloat16 |
| query_qjl | bfloat16 |
| key_cache_idx | uint8 |
| key_cache_qjl | uint8 |
| key_cache_norm | bfloat16 |
| key_cache_gamma | bfloat16 |
| block_table | int32 |
| seq_lens | int32 |
| attn_scores | float32 |
| mse_bits | int |

#### 3.1.3 支持形状
- `query_rotated`: `[num_q_tokens, num_q_heads, head_dim]`，head_dim = 128
- `query_qjl`: `[num_q_tokens, num_q_heads, qjl_dim]`，qjl_dim = 128
- `key_cache_idx`: `[num_blocks, block_size, num_kv_heads, ceil(head_dim*mse_bits/8)]`，block_size = 128
- `key_cache_qjl`: `[num_blocks, block_size, num_kv_heads, ceil(qjl_dim/8)]`
- `key_cache_norm`: `[num_blocks, block_size, num_kv_heads]`
- `key_cache_gamma`: `[num_blocks, block_size, num_kv_heads]`
- `block_table`: `[max_batch, max_blocks_per_seq]`
- `seq_lens`: `[max_batch]`
- `attn_scores`: `[num_q_tokens, num_q_heads, max_kv_len]`

### 3.2 算子实现

#### 3.2.1 Host 侧设计

**接口定义**（aclnn 风格）：

```cpp
aclnnStatus_t aclnnTurboQuantAttentionScore(
    const aclTensor* query_rotated,
    const aclTensor* query_qjl,
    const aclTensor* key_cache_idx,
    const aclTensor* key_cache_qjl,
    const aclTensor* key_cache_norm,
    const aclTensor* key_cache_gamma,
    const aclTensor* block_table,
    const aclTensor* seq_lens,
    int64_t mse_bits,
    aclTensor* attn_scores,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

**参数校验**：
1. 校验所有输入输出指针非空。
2. 校验数据类型与 shape 符合约束。
3. 校验 `mse_bits ∈ {2,3,4}`。
4. 校验 `block_size = 128`，`head_dim = 128`。
5. 校验 `num_q_heads % num_kv_heads == 0`。

**Tiling 策略**：
- **Decode 场景**（`num_q_tokens = batch_size`，通常为 1）：按 `(batch, kv_head, block)` 三维切分，每个核处理若干 block 的 score 计算。
- **Prefill 场景**（`num_q_tokens > 1`）：按 `(q_token, q_head, kv_block)` 切分，利用 Cube 进行批量矩阵乘。
- Tiling 参数包括：`num_q_tokens`、`num_q_heads`、`num_kv_heads`、`head_dim`、`qjl_dim`、`block_size`、`max_kv_len`、`mse_bits` 等。
- Workspace 用于存放中间结果（如解包后的 centroid 索引、符号向量等）。

**Kernel 下发**：Host 侧完成 Tiling 后，通过 `<<<block_dim, nullptr, stream>>>` 下发 Kernel，并同步 stream 以保证结果可见。

#### 3.2.2 Kernel 侧设计

**整体流程**：采用多核并行，每个核处理一部分 `(batch, kv_head, kv_block)` 的计算。

1. **CopyIn 阶段**：
   - 读取 `query_rotated`、`query_qjl` 对应 token 和 head 的数据。
   - 根据 `block_table` 和 `seq_lens` 计算当前处理的物理 block 和偏移。
   - 搬入 `key_cache_idx`、`key_cache_qjl`、`key_cache_norm`、`key_cache_gamma` 对应 block 的数据。

2. **Compute 阶段**：
   - **解包主量化**：按 `mse_bits` 将 uint8 数据解包为索引，3-bit 时按 3 字节一组解出 8 个值，LSB 优先。
   - **查表反量化**：根据 `mse_bits` 选择 centroid 表，将索引映射为 bfloat16 值。
   - **计算 score_mse**：`score_mse = norm * dot(Hq, centroid)`。
   - **解包 QJL 符号**：每字节解出 8 个符号位，映射为 ±1。
   - **计算 score_qjl**：`score_qjl = coeff * gamma * dot(query_qjl, sign)`，其中 `coeff = sqrt(pi/2)/qjl_dim`。
   - **合并**：`score = score_mse + score_qjl`。
   - **掩码**：若 `position >= seq_len`，置 `-inf`。

3. **CopyOut 阶段**：
   - 将 `score` 写入 `attn_scores` 对应位置。

**核心算法伪代码**：

```cpp
// 每个核处理 (batch, kv_head, block) 子集
for each assigned (b, kv_h, blk) {
    physical_block = block_table[b][blk];
    seq_len = seq_lens[b];
    for (pos = 0; pos < block_size; ++pos) {
        kv_pos = blk * block_size + pos;
        if (kv_pos >= seq_len) {
            score = -INF;
        } else {
            // 解包主量化
            idx = unpack_primary(key_cache_idx[physical_block][pos][kv_h], mse_bits);
            centroid = lookup_centroid(idx, mse_bits);
            // 计算 MSE 分数
            dot_mse = dot(query_rotated[b][q_h][:], centroid);
            score_mse = key_cache_norm[physical_block][pos][kv_h] * dot_mse;
            // 解包 QJL
            sign = unpack_signs(key_cache_qjl[physical_block][pos][kv_h]);
            dot_qjl = dot(query_qjl[b][q_h][:], sign);
            score_qjl = coeff * key_cache_gamma[physical_block][pos][kv_h] * dot_qjl;
            score = score_mse + score_qjl;
        }
        attn_scores[b][q_h][kv_pos] = score;
    }
}
```

**特殊值处理**：
- 无效位置填 `-inf`。
- 支持 `mse_bits = 2/3/4`。
- 支持 batch > 1 和 batch = 1。

### 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2 | √ |

### 3.4 算子约束限制
1. 仅支持标准 MHA/GQA，不支持 MLA attention。
2. `block_size = 128`，`head_dim = 128`。
3. `num_q_heads / num_kv_heads` 必须为整数。
4. 主量化编码需先按 `mse_bits` 解包，再通过查表映射到 centroid。
5. QJL 分支为实数 `query_qjl` 与 `{-1,+1}` 符号向量的内积，不能直接等价为两个 sign bit-vector 的 popcount。
6. 输出 `attn_scores` 后续接 softmax + V matmul（不在本算子范围内）。

## 4 可维可测分析

### 4.1 精度标准/性能标准

#### 4.1.1 精度验收标准
- 与全精度 `Q @ K^T` 对比，score 相对误差在统计意义上可控。
- 端到端指标：经 softmax 后的 attention weight 分布与全精度版本的 KL 散度 < 0.01。
- 最终生成质量：PPL 增长 < 1%（配合算子 2 联合测试）。

#### 4.1.2 性能验收标准
**Decode 场景**：整体 attention score 计算（含反量化）的延迟不超过全精度 FlashAttention 的 **1.2 倍**，同时显存占用降低 **3x 以上**。

| case | num_q_tokens | batch_size | kv_len | num_q_heads | num_kv_heads | head_dim | Paged Full-QK baseline (us) | 说明 |
|---|---|---|---|---|---|---|---|---|
| gqa_decode_kv4k | 1 | 1 | 4096 | 32 | 8 | 128 | 1501.634 | 短上下文 decode |
| gqa_decode_kv32k | 1 | 1 | 32768 | 32 | 8 | 128 | 9278.507 | 中等上下文 decode |
| gqa_decode_kv128k | 1 | 1 | 131072 | 32 | 8 | 128 | 35681.690 | 长上下文 decode |
| gqa_prefill_kv4k | 512 | 1 | 4096 | 32 | 8 | 128 | 75984.253 | 单序列 prefill |
| gqa_prefill_kv32k | 512 | 1 | 32768 | 32 | 8 | 128 | - | 单序列 prefill |

### 4.2 自验用例覆盖
基于 `case.json` 的用例集（5 条），覆盖以下场景：

| 类别 | 用例数 | 说明 |
|---|---|---|
| Decode 短上下文 | 1 | gqa_decode_kv4k |
| Decode 中上下文 | 1 | gqa_decode_kv32k |
| Decode 长上下文 | 1 | gqa_decode_kv128k |
| Prefill 短上下文 | 1 | gqa_prefill_kv4k |
| Prefill 中上下文 | 1 | gqa_prefill_kv32k |

每个用例验证：
- 输出 shape 正确。
- 无效位置为 `-inf`。
- 有效位置 score 与 golden 对比满足精度要求。
- 性能满足基线要求。

### 4.3 兼容性分析
本算子为新增算子，不涉及存量代码兼容性改造。接口遵循 `ops-transformer` 仓规范，算子原型定义、Tiling、Kernel 实现均参照仓内现有 attention 算子。合入目录为 `experimental/attention`，代码结构遵循仓内贡献指南。
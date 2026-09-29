# TurboQuantAttentionScore 算子设计文档

# 需求背景

## 需求来源

本算子是社区任务 2026 年 9 月 TurboQuant Attention Score 开发任务，目标在 Atlas 800T A2 上实现基于 TurboQuant 编码 KV Cache 的高效 Attention Score 计算。

## 背景介绍

### TurboQuant Attention 原理

传统 FlashAttention 在计算 `Q @ K^T` 时需要完整加载 K 向量到 L1/UB。TurboQuant 对 KV Cache 进行 3-bit 量化编码（主分支 + QJL 残差），在计算时按需反量化，实现显存压缩比约 3.76x 的同时保持精度。

### 算子语义

```
score[i] = score_mse[i] + score_qjl[i]

score_mse[i] = x_norm[i] * <Hq, dequant(idx[i])>
score_qjl[i] = (sqrt(pi/2) / qjl_dim) * gamma[i] * <S(Hq), sign_vec(qjl[i])>

预计算:
  Hq = H @ q          # query 旋转
  query_qjl = S @ Hq  # QJL 投影
```

输入 `query_rotated` 和 `query_qjl` 已预计算，算子直接使用该结果。

---

# 需求分析

## 需求描述

使用 Ascend C 编程语言实现 `TurboQuantAttentionScore` 算子，支持：
- bf16 query (rotated + qjl)，uint8 bit-packed key cache (3bit idx + 1bit sign)，bfloat16 norm/gamma
- Paged attention 布局（block_table + seq_lens）
- GQA：32 q-heads / 8 kv-heads，head_dim=128，block_size=128
- mse_bits 支持 2/3/4 bit（默认 3）
- 输出 fp32 attn_scores

## 需求拆解

1. 支持 bf16 query 和 uint8 bit-packed key cache 输入
2. 支持 Paged KV Cache（block_table 查表 + seq_lens 有效性掩码）
3. 支持 GQA head 扩展（kv_head -> q_head 广播）
4. 支持 mse_bits=2/3/4 动态查表反量化
5. 精度：与 torch_npu golden 对比，MSE < 0.001
6. 性能：decode 场景不超过全精度 FlashAttention 的 1.2x，显存降低 3x+

---

# 详细设计

## 算子分析

### 数学公式

$$\text{score}[t, q, k] = \text{x\_norm}[b, k] \cdot \langle H_q[t, q], \text{dequant}(\text{idx}[b, k]) \rangle + \frac{\sqrt{\pi/2}}{D_{qjl}} \cdot \gamma[b, k] \cdot \langle \text{query\_qjl}[t, q], \text{sign}(\text{qjl}[b, k]) \rangle$$

其中：
- `dequant(idx)` = `centroid_table[idx]`，通过 unpack uint8→uint32 索引查表得到 bf16 向量
- `sign(qjl)` = `2 * bit - 1`，从 uint8 解出 8 个 ±1 符号
- `x_norm` 和 `gamma` 为 per-block per-kv_head 的标量
- 无效位置（k >= seq_len）填充 `-inf`

### 支持数据类型

| 参数 | 数据类型 | 形状 |
|------|----------|------|
| query_rotated | bfloat16 | [T, 32, 128] |
| query_qjl | bfloat16 | [T, 32, 128] |
| key_cache_idx | uint8 | [nb, 128, 8, ceil(128×mse_bits/8)] |
| key_cache_qjl | uint8 | [nb, 128, 8, 16] |
| key_cache_norm | bfloat16 | [nb, 128, 8] |
| key_cache_gamma | bfloat16 | [nb, 128, 8] |
| block_table | int32 | [1, max_blocks] |
| seq_lens | int32 | [1] |
| attn_scores | float32 | [T, 32, max_kv_len] |

### 支持形状

测试用例覆盖：
- **Decode**: T=1, kv_len=4096/32768/131072
- **Prefill**: T=512, kv_len=4096/32768

## 算子实现

### 实现方案

#### Host 侧设计

**Tiling 策略：**

```python
# 核心 tiling 参数
TilingData {
    core_num: uint32        # 使用核数（由平台 UB 大小决定）
    tile_T: uint32          # 每个核处理的 T 维度 chunk
    tile_S2: uint32         # KV 长度方向每块大小（block 数）
    double_buffer: uint32   # 双缓冲深度
    
    # 数据布局
    num_q_tokens: uint32    # T
    num_q_heads: uint32     # 32
    num_kv_heads: uint32    # 8
    head_dim: uint32        # 128
    block_size: uint32      # 128
    mse_bits: uint32        # 2/3/4
    max_kv_len: uint32      # 从 seq_lens 推导
    max_blocks: uint32      # ceil(max_kv_len / block_size)
}
```

**任务均分：**
- 以 T 轴为主要切分维度（T 方向可多核并行）
- KV 方向按 block 粒度切分（S2），每个核处理连续的 block 范围
- 核数根据平台 UB 大小动态计算，确保每个核的 UB 占用不超限

**Tiling Key 规划：**
- `tiling_key = mse_bits << 8 | is_prefill`（用于 kernel 分支选择）
- `mse_bits=2/3/4` → 不同 centroid table 大小（4/8/16 entries）
- `is_prefill` → T>1 时使用批量 Matmul 路径，T=1 使用 Vector 路径

#### Kernel 侧设计

**总体架构：AIV（Vector Core）+ AIC（Cube Core）混合流水**

```
┌─────────────────────────────────────────────────────────────┐
│                    单核处理流程                              │
├─────────────────────────────────────────────────────────────┤
│  AIV (Vector Core)                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Unpack   │  │ Centroid │  │ Sign     │  │ Multiply │   │
│  │ idx→int3 │  │ Lookup   │  │ Unpack   │  │ ×norm/γ  │   │
│  │ Cast to  │  │ →bf16    │  │ →fp32    │  │ + Reduce │   │
│  │ bf16     │  │          │  │          │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘   │
│                                                   │         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │         │
│  │ Load     │  │ Vec     │  │ Final    │◄──────┘         │
│  │ query    │  │ Mul     │  │ Add      │                 │
│  │ Hq       │  │ norm*dot│  │ into fp3 │                 │
│  │ qjl      │  │ γ*dot   │  │ output   │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
├─────────────────────────────────────────────────────────────┤
│  AIC (Cube Core) — Prefill 批量路径                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Matmul   │  │ Matmul   │  │ Scale+   │                 │
│  │ Hq@Y^T   │  │ qjl@S^T  │  │ Reduce   │                 │
│  │ (bf16→f)│  │ (bf16→f) │  │ ×norm/γ  │                 │
│  └──────────┘  └──────────┘  └────┬─────┘                 │
│                                    │                        │
│  ┌──────────┐  ┌──────────┐       │                        │
│  │ Block    │  │ Output   │◄──────┘                        │
│  │ Epilogue │  │ to GM    │                                │
│  │ (-inf)   │  │ fp32     │                                │
│  └──────────┘  └──────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

**核心优化策略（按优先级）：**

##### 1. 极致性能：双阶段流水

**Decode 路径（T=1，小批量）：**
- AIV 串行处理每个 KV block，减少 GM 读取次数
- 关键：在 UB 内完成 unpack → lookup → dot → scale 全流程
- 使用 `CopyUb2L1` 预取下一 block 数据（如 A2 支持）
- 利用 AIC/AIV 间 SetFlag/WaitFlag 交叉核同步

**Prefill 路径（T=512，大批量）：**
- 将同一 query batch 对所有 KV block 的内积 batch 化为 Matmul
- AIC 执行 `Hq (bf16) × Y_hat^T (bf16) → fp32` 主项
- AIC 执行 `query_qjl (bf16) × sign_matrix^T (fp32) → fp32` 残差项
- 两项在 fp32 UB 中合并后乘以 norm/gamma 标量
- 使用 `BlockEpilogue` 设置无效位置为 -inf

##### 2. 减少 GM 读取

| 策略 | 实现方式 |
|------|----------|
| 双缓冲 | 使用 2 个 double buffer 交替搬运 query 和 key cache |
| 数据复用 | query 在 UB 中缓存，多次参与不同 KV block 的计算 |
| 按块加载 | 只加载当前处理的 block 到 UB，而非整个 KV cache |
| L1 缓存 | key cache norm/gamma 等小数据放入 L1 重复使用 |

##### 3. 共享 UB/Buffer/算子融合

- `query_rotated` 和 `query_qjl` 合并为单一 L1 buffer（连续内存布局）
- norm/gamma 合并为一个 L1 buffer（减少独立 CopyIn 次数）
- unpack idx 和 unpack qjl 在同一次 CopyIn 中完成
- 主项和残差项的 dot product 结果在 UB 内直接相加，无需回写 GM

##### 4. 950 跨核 AIC/AIV 通信（如适用）

- Decode 路径：AIC 负责 Matmul，AIV 负责 unpack 和标量运算
- 使用 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 在 AIC 和 AIV 间同步
- flag ID 分配：SYNC_V0_C1=6, SYNC_C1_V1=7, SYNC_V1_C2=8, SYNC_C2_V2=9

##### 5. 选择性使用 KFC

- 小 tile 场景：使用 KFC（Kernel First Class）加速 ub↔l1 数据搬运
- 大 tile 场景：回退到标准 DataCopy + DoubleBuffer

---

### 核心算法实现细节

#### AIV 侧：Unpack + Centroid Lookup

```cpp
// uint8 bit-packed 3-bit idx 解包（A2 限制：位运算需 uint16）
// 输入: key_cache_idx [nb, 128, 8, 48] (uint8)
// 输出: centroids [nb, 128, 8, 128] (bf16)

for each block in nb:
    // 1. GM → UB: 搬运当前 block 的 idx 数据
    DataCopy(idx_ub, idx_gm[block_offset], byte_num);
    
    // 2. Unpack: uint8 → uint32 indices (需 cast 到 uint16 做位运算)
    //    48 bytes = 16 × 3 bytes → 每个 3-byte 组解出 1 个 12-bit index
    //    实际: 128 dims / 8 bits-per-byte * 3 bits = 48 bytes
    LocalTensor<uint16_t> idx_u16 = ReinterpretCast<uint16_t>(idx_ub);
    // 使用 VecShiftLeft/VecAnd 提取每个 3-bit field
    // 注意：A2 上行运算不支持 int8/uint8，必须 cast 到 uint16
    
    // 3. Centroid Table Lookup
    //    mse_bits=3 → 8 个 centroid (bf16)
    //    使用 Gather 指令：Gather(centroid_table, idx_u16) → bf16_vector
    LocalTensor<bf16_t> centroid_vec = Gather(centroid_table, idx_u16);
    
    // 4. 标量乘法: score_mse = x_norm[block] * dot(Hq, centroid_vec)
    float norm_f32 = Cast<float>(x_norm_gm[block]);
    float dot_val = ReduceSum(Hq_f32 * centroid_vec_f32);
    float score_mse = norm_f32 * dot_val;
```

#### AIV 侧：Sign Unpack + QJL 修正

```cpp
// uint8 bit-packed 1-bit sign 解包
// 输入: key_cache_qjl [nb, 128, 8, 16] (uint8) 
// 16 bytes = 128 bits → 每个 bit 对应一个 ±1 符号

LocalTensor<uint16_t> qjl_u16 = ReinterpretCast<uint16_t>(qjl_ub);
// 提取每个 bit: sign = 2 * bit - 1
LocalTensor<float> sign_f32 = VecDup(2.0f) * Cast<float>(qjl_u16 & 1) 
                              - VecDup(1.0f);  // 或 VecSel 方式

// score_qjl = coeff * gamma * dot(query_qjl, signs)
float coeff = sqrt(M_PI / 2.0f) / 128.0f;
float gamma_f32 = Cast<float>(gamma_gm[block]);
float dot_qjl = ReduceSum(query_qjl_f32 * sign_f32);
float score_qjl = coeff * gamma_f32 * dot_qjl;
```

#### AIC 侧：批量 Matmul（Prefill 路径）

```cpp
// 将 query batch 与反量化后的 key batch 做矩阵乘法
// A: Hq [T, 32, 128] → 重排为 [T*32, 128]
// B: Y_hat [nb*128*8, 128] → 重排为 [128, nb*128*8]
// C: scores [T*32, nb*128*8] → fp32

// Matmul tiling: M=128, N=128, K=128 (full dim)
MatmulParams params;
params.mode = GEMM_MODE_NORMAL;
params.transposeA = false;
params.transposeB = true;

// 双缓冲 Matmul
for each tile in nb_blocks:
    // 1. 预取下一 tile 的 key cache 到 L1
    DataCopy(key_l1[tile+1], key_gm[block_offset], ...);
    
    // 2. Cube Matmul: Hq × Y_hat^T
    TileMmad(res_ub, a_ub, b_ub, params);
    
    // 3. 乘上 norm 标量
    VecMul(res_ub, res_ub, norm_ub);
```

#### 无效位置处理（SetFlag/WaitFlag）

```cpp
// 在 Epilogue 阶段设置无效位置为 -inf
for each valid position:
    if position < seq_len:
        output[t, q, position] = score
    else:
        output[t, q, position] = -INF

// 使用 VecSel 实现条件赋值
LocalTensor<float> mask = Cast<float>(position < seq_len);
LocalTensor<float> inf_val = VecDup(-3.402823466e+38f);
VecSel(output_ub, score_ub, inf_val, mask);
```

---

### Paged Attention 处理

```cpp
// block_table[b, logical_block_idx] → physical_block_id
// 通过 block_table 查表获取物理 block 索引
int32_t phys_block = BlockTableGm[batch_idx][logical_idx];

// 计算物理偏移
uint64_t block_offset = phys_block * block_size * kv_head_num * ...;
```

---

### GQA Head 处理

```cpp
// num_q_heads=32, num_kv_heads=8, groups=4
// 每个 kv_head 对应 4 个 q_head
// 在 dot product 后 broadcast norm/gamma 到所有 q_groups

// norm broadcast: [nb, 128, 8] → [nb, 128, 32]
// 使用 VecDup + VecMul 完成广播乘法
for g in 0..groups-1:
    score[t, kv*h+g, k] = norm[b, k, h] * dot(Hq[t, kv*h+g], Y[k, h])
```

---

### 三层 Tiling 设计

| 层级 | 维度 | 说明 |
|------|------|------|
| 核间 | T, batch | 多个核处理不同的 T chunk 或 batch |
| 核内 M | T 维度 chunk | 单核内 T 方向的 tile 大小 |
| 核内 N/S2 | KV block 范围 | 单核内 KV 长度方向的 tile |

**核内 Tiling 参数：**
- `tile_T`: 根据 UB 大小和 query 数据量计算
- `tile_S2`: KV block 数量，确保 L1 能容纳当前 tile 的 key cache
- `double_buffer`: 2 级双缓冲

---

## 其他优化方法（层次化）

### 硬件特性层

| 方法 | 实现 | 适用场景 |
|------|------|----------|
| BT Buffer | 使用 1KB BT Buffer + `CopyL1ToBT` 预取 | KV cache 小 tile |
| FB Buffer | 如可用，大 tile 场景 | Prefill |
| Shuffle K | 矩阵转置优化 | Matmul K 维度非对齐 |
| L2 Cache Hint | `SetL2CacheHint` 标记频繁访问数据 | key_cache_norm/gamma |
| Atomic Add | `SetAtomicAdd` 实现 GM 原子累加 | 多核合并结果 |
| Kernel Type | 选择 `KERNEL_TYPE_MIX_AIC_1_2` | 混合 AIC/AIV |

### 指令级优化

| 方法 | 实现 | 注意事项 |
|------|------|----------|
| Dual Issue | 交错 MTE 和 Vector 指令 | 需要手动 schedule |
| VF 硬件循环 | `SetLoop` + 循环展开 | 减少指令开销 |
| 低延迟 Reduction | `ReduceRepeat` 替代 `WholeReduceSum` | A2 必须用 ReduceRepeat |
| `--cce-no-dcache-flush` | 编译选项禁用 cache flush | 减少 overhead |

### 系统/工具层

| 方法 | 工具/配置 |
|------|-----------|
| AOE 自动调优 | 对 Matmul 路径使用 AOE 搜索最优 tiling |
| 模型/Tiling 下沉 | 静态 tiling 参数写入算子配置 |
| Warm-up | 首帧多跑 30 次稳定 cache |
| `--opt-level=3` | Bissheng 最高优化级别 |
| `--dump-asm` | 导出汇编分析指令级瓶颈 |

### Catlass 层

| 方法 | API | A2 可用性 |
|------|-----|-----------|
| CopyUb2L1Tla | `CopyUb2L1Tla` | ⛔ A3 only |
| StreamK | `StreamK` | ⛔ A3 only |
| ASWT S-shaped window | `ASWT` | ⛔ A3 only |
| Preload/MTE2Preload | `MTE2Preload` | ✅ 可用 |
| BT Buffer | `CopyL1ToBT` | ✅ 可用 |
| BlockEpilogue | `BlockEpilogue` + `EpilogueAtlasA2PerTokenDequant` | ✅ 可用 |
| GemmL1/GemmTlaT | `DeviceGemm` / `BasicMatmul` | ✅ 可用 |

---

# 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---------------|---------|
| Atlas 800T A2 (Davinci C220, arch22) | √ |
| Atlas 800I/T A2 | √ |

---

# 算子约束限制

1. 仅支持标准 MHA/GQA，不支持 MLA attention
2. block_size 固定为 128
3. head_dim 固定为 128
4. mse_bits 支持 2/3/4，默认 3
5. num_q_heads % num_kv_heads == 0（GQA group 整除）
6. query_rotated 和 query_qjl 需为 bf16
7. key_cache_idx/qjl 为 uint8 bit-packed
8. 输出 attn_scores 为 fp32，后续接 softmax（不在本算子范围内）
9. 无效 KV 位置（>= seq_len）输出 -inf

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---------|------|---------|
| 精度标准 | 与 torch_npu golden 对比，MSE < 0.001 | 任务书 3.2 节 |
| 精度标准 | 经 softmax 后 attention weight KL 散度 < 0.01 | 任务书 3.2 节 |
| 性能标准 | decode 场景不超过全精度 FA 的 1.2x | 任务书 3.3 节 |
| 性能标准 | 显存降低 3x+ | 任务书 3.3 节 |

## 测试用例

| case_name | T | kv_len | q_heads | kv_heads | 说明 |
|-----------|---|--------|---------|----------|------|
| gqa_decode_kv4k | 1 | 4096 | 32 | 8 | 短上下文 decode |
| gqa_decode_kv32k | 1 | 32768 | 32 | 8 | 中等上下文 decode |
| gqa_decode_kv128k | 1 | 131072 | 32 | 8 | 长上下文 decode |
| gqa_prefill_kv4k | 512 | 4096 | 32 | 8 | 单序列 prefill |
| gqa_prefill_kv32k | 512 | 32768 | 32 | 8 | 较长序列 prefill |

## 兼容性分析

- 新算子，不涉及向后兼容性问题
- 与 ops-transformer 仓库其他 attention 算子（turbo_quant_sparse_flash_attention, quant_flash_attn）保持一致的编码风格
- 支持 CANN 9.1.0+，适配 A2 架构（arch22）

---

# 性能优化计划（Loop 迭代）

## 第一阶段：基础实现 + 精度验证
1. 实现 AIV 串行路径（decode T=1）
2. 实现 AIC Matmul 路径（prefill T=512）
3. 使用 `ascendebug simulator` 进行 CPU 仿真
4. 精度比对 golden.py

## 第二阶段：Profiling + 优化
1. 使用 msprof 采集性能数据
2. 分析 MTE 指令占比（目标 < 80%）
3. 分析 Vector 计算单元利用率
4. 使用 MindStudio Advisor 定位瓶颈
5. 优化双缓冲深度、tiling 参数

## 第三阶段：极致性能
1. 应用 BT Buffer + CopyL1ToBT
2. 应用 L2 Cache Hint
3. 应用 `--cce-no-dcache-flush`
4. 应用 AOE 自动调优
5. 评估是否需要 StreamK（如 A3 可用）

## 收敛条件
- 精度：所有测试 case MSE < 0.001
- 性能：decode 128k 场景 < 1.2x baseline（35681μs × 1.2 ≈ 42818μs）
- 如不达标，重新进入多专家文档搜索 → 修订 plan → 新分支执行循环

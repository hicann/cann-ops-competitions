# 需求背景（required）

## 需求来源

CANN 社区算子开发竞赛 - Qwen3-VL 模型算子优化任务

## 背景介绍

### SplitQkvRmsnormMrope 算子实现需求

Qwen3-VL 是阿里云推出的新一代视觉语言模型，在注意力层前处理阶段需要对融合的 QKV+Gate 线性层输出进行拆分、归一化和位置编码。当前 vllm-ascend 使用 Triton 实现，存在以下问题：

1. **多核启动开销大**：Triton 采用多核并行架构，kernel 启动和跨核同步开销在 decode 场景（单 token 生成）占比达 95%
2. **标量-向量传输瓶颈**：RMSNorm 的 gamma 参数以 scalar 形式传输到每个 head，造成大量跨单元传输
3. **指令数冗余**：per-head 循环处理导致向量指令数量过多（每 token 40 条）

需要使用 Ascend C 实现高性能算子，特别针对 decode 场景优化。

### SplitQkvRmsnormMrope 算子功能分析

SplitQkvRmsnormMrope 算子融合了三个操作：

1. **Split（拆分）**：将融合输入 `[Q_h0 | Gate_h0 | Q_h1 | Gate_h1 | ... | K | V]` 拆分为 Q、Gate、K、V 四路
2. **RMSNorm（RMS 归一化）**：对 Q 和 K 的每个 attention head 执行 `y = (x / sqrt(mean(x²) + ε)) * gamma + beta`
3. **MRoPE（多轴旋转位置编码）**：对 Q 和 K 的前 64 维应用三轴旋转 `[x1_new, x2_new] = [x1*cos - x2*sin, x2*cos + x1*sin]`

**输入**：
- qkv: [num_tokens, (2*numQHeads + 2*numKvHeads) * head_size], bf16
- qWeight, kWeight: [head_size], bf16, RMSNorm gamma
- cosSin: [3, num_tokens, rope_dim], bf16, 三轴 cos/sin 表
- qBias, kBias: [head_size], bf16, RMSNorm beta（可选）

**输出**：
- qOutput: [num_tokens, numQHeads * head_size], bf16
- kOutput: [num_tokens, numKvHeads * head_size], bf16
- vOutput: [num_tokens, numKvHeads * head_size], bf16（直通）
- gateOutput: [num_tokens, numQHeads * head_size], bf16（直通）

**支持数据类型**：bf16

**约束条件**：
- head_size = 256（Qwen3-VL 固定）
- rope_dim = 64
- numQHeads ∈ {16, 32}，numKvHeads ∈ {2, 4, 8}

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 SplitQkvRmsnormMrope 算子，支持 bf16 数据类型，实现 QKV 拆分、RMS 归一化和多轴旋转位置编码的融合操作。

## 需求拆解

1. 支持 bf16 数据类型，内部计算使用 fp32 精度
2. 支持 Qwen3-VL 标准配置（head_size=256, rope_dim=64, numQHeads=16/32, numKvHeads=2/4/8）
3. 性能目标：decode 场景加速比 ≥ 20×，prefill 场景加速比 ≥ 2×（vs Triton baseline）
4. 精度目标：最大差异 ≤ 1e-3，V/Gate 逐位一致
5. 确定性保证：相同输入产生相同输出

# 详细设计（required）

## 算子分析

### 数学公式

**RMSNorm**：
```
variance = mean(x²)
scale = 1 / sqrt(variance + ε)
normalized = x * scale
output = normalized * gamma + beta
```

**MRoPE**（每个 rotation coordinate，三个 axis）：
```
x1_new = x1 * cos[axis] - x2 * sin[axis]
x2_new = x2 * cos[axis] + x1 * sin[axis]
其中 axis ∈ {temporal, height, width}
```

### 支持数据类型

bf16（输入/输出），fp32（内部计算）

### 支持形状

- 输入：[num_tokens, (2*numQHeads + 2*numKvHeads) * head_size]
- 输出：Q/Gate [num_tokens, numQHeads * head_size]，K/V [num_tokens, numKvHeads * head_size]
- num_tokens ∈ [1, 8192]

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

##### 1. Tiling 策略

**Token 维度切分**：
- 单 token/AICore：每个 token 由一个 AICore 独立处理
- 负载均衡：`blockIdx < numTokens` 时，核心 `blockIdx` 处理 token `blockIdx`
- 无跨核通信：token 间无数据依赖

**优势**：
- 简单：无需复杂的数据切分和同步
- 适合 decode：单 token 场景避免多核启动开销
- 负载均衡：token 数通常远大于核心数

**劣势**：
- 单核 UB 限制：无法处理超大 head_size（当前 256 可接受）

##### 2. 分核策略

```cpp
// 每个 AICore 处理一个 token
int64_t token = blockIdx;
if (token < numTokens) {
    ProcessToken(token);
}
```

**核心数量**：`min(numTokens, totalCoreNum)`

##### 3. 内存优化策略

**UB 空间分配**（单 token，numQHeads=16, numKvHeads=4）：

| Buffer | 大小 (bytes) | 用途 |
|---|---|---|
| rowBf16Buf | 10240 | 输入行（bf16） |
| rowFp32Buf | 40960 | 输入行（fp32 widened） |
| sqBuf | 20480 | x² 缓冲区（Q+K，20 heads × 256 × fp32） |
| redBuf | 2560 | 规约结果（20 heads × 8 partial × fp32） |
| scaleSrcBuf | 96 | scale 源（24 heads 对齐 × fp32） |
| scaleBlkBuf | 768 | scale 广播块（24 × 32 bytes） |
| weightExpBuf | 20480 | gamma 预展开（(16+4)*256 × fp32） |
| ropeTmpBuf | 16384 | 旋转暂存（4 products × 16 heads × 64 × fp32） |
| trigBuf | 512 | cos/sin 暂存（2 × 64 × fp32） |
| 其他 | ~8000 | trigInQue、maskBuf 等 |

**总计**：~120 KB（远小于 400 KB UB 限制）

**优化原则**：
- 充分使用 UB 空间，避免频繁 GM 访存
- 预分配所有缓冲区（Init 阶段），Process 阶段零分配
- gamma 预展开：每 head 独立副本，避免 scalar-to-vector 传输

##### 4. Tiling 参数传递

通过 `SplitQkvRmsnormMropeTilingData` 结构体传递：

```cpp
struct SplitQkvRmsnormMropeTilingData {
    int64_t numTokens;       // token 总数
    int64_t numQHeads;       // Q head 数
    int64_t numKvHeads;      // K/V head 数
    int64_t headSize;        // head 大小（256）
    int64_t ropeDim;         // 旋转维度（64）
    float epsilon;           // RMSNorm epsilon
    bool interleaved;        // MRoPE 模式
    bool hasBias;            // 是否有 bias
    int64_t qSize;           // numQHeads * headSize
    int64_t kvSize;          // numKvHeads * headSize
};
```

#### 3.2.2 kernel 侧设计

##### Init 阶段

```cpp
__aicore__ inline void Init() {
    // 1. 初始化所有 UB 缓冲区
    pipe_.InitBuffer(rowBf16Buf_, ...);
    pipe_.InitBuffer(rowFp32Buf_, ...);
    // ... 其他缓冲区
    
    // 2. 预展开 gamma 权重
    LoadWeights();           // 加载 qWeight, kWeight
    BuildExpandedWeights();  // 复制到每个 head
    
    // 3. 预计算 axis masks（MRoPE）
    BuildAxisMasks();        // interleaved/contiguous 模式
}
```

**关键优化 - gamma 预展开**：

```cpp
// 原始设计：共享 gamma [256]
// 每 head 都读取同一份，scalar-to-vector 传输开销大

// 优化设计：预展开为 [numQHeads * 256 + numKvHeads * 256]
for (int h = 0; h < numQHeads; ++h) {
    Copy(weightExpBuf[h * headSize], qWeight, headSize);
}
for (int h = 0; h < numKvHeads; ++h) {
    Copy(weightExpBuf[qSize + h * headSize], kWeight, headSize);
}

// Process 阶段：vector-form 读取（block stride = 1）
// 每次 ~2 ns vs 原始 ~23 ns（降低 11.5×）
```

##### Process 阶段（每个 token）

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 0: GM → UB (qkv 输入, cos/sin 输入)                  │
│   - DataCopy: qkv[token * rowLen : (token+1) * rowLen]    │
│   - DataCopy: cosSin[3][token * ropeDim]                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Compute                                            │
│   1. Split & Widen: bf16 → fp32                            │
│      - Q: interleaved gather (head 0, 2, 4, ...)          │
│      - Gate: interleaved gather (head 1, 3, 5, ...)       │
│      - K: block copy                                       │
│      - V: block copy                                       │
│                                                             │
│   2. RMSNorm (Q + K)                                       │
│      a. x² 并行向量化                                      │
│         Mul(sq, x, x, mask, rep, params)                   │
│                                                             │
│      b. 树形折叠规约（256 → 128 → 64 → ... → 8）          │
│         for (half = 128; half >= 64; half /= 2) {          │
│             Add(sq[0:half], sq[0:half], sq[half:2*half])   │
│         }                                                   │
│                                                             │
│      c. ReduceRepeat 收尾（8 → 1）                         │
│         ReduceRepeat<SUM>(result, sq, 8, numHeads, ...)    │
│                                                             │
│      d. 计算 scale = 1 / sqrt(variance + eps)              │
│         Muls(var, sum, 1.0f / 256.0f, numHeads)            │
│         Adds(var, var, eps, numHeads)                      │
│         Sqrt(tmp, var, numHeads)  // 精确实现              │
│         Div(scale, ones, tmp, numHeads)  // 精确实现       │
│                                                             │
│      e. 广播 scale 到 [numHeads, headSize]                 │
│         Brcb(scaleBlk, scale, rep, params)                 │
│                                                             │
│      f. chunked 归一化（64 元素/chunk，跨 head 处理）     │
│         BinaryRepeatParams chunkParams(                    │
│             1,               // dst block stride           │
│             1,               // src0 block stride          │
│             0,               // src1 block stride (广播)   │
│             headSize / 8,    // dst repeat stride (下一head)│
│             headSize / 8,    // src0 repeat stride        │
│             1                // src1 repeat stride        │
│         );                                                 │
│         for (int chunk = 0; chunk < 4; ++chunk) {         │
│             Mul(norm, x, scaleBlk, 64, numHeads, chunkParams);│
│             Mul(out, norm, gamma, 64, numHeads, chunkParams);│
│         }                                                   │
│         // 4 个 chunk × 2 条指令/chunk = 8 条指令          │
│         // vs 原始 20 heads × 2 ops/head × 4 vec/op = 160 条│
│         // 降低 20×                                        │
│                                                             │
│   3. MRoPE (Q[:, :64] + K[:, :64])                         │
│      a. 加载 cos/sin（已在 Stage 0 完成）                 │
│                                                             │
│      b. 四项乘法（每 coordinate）                          │
│         Mul(p0, x1, cos, mask, rep, params)  // x1*cos     │
│         Mul(p1, x2, sin, mask, rep, params)  // x2*sin     │
│         Mul(p2, x2, cos, mask, rep, params)  // x2*cos     │
│         Mul(p3, x1, sin, mask, rep, params)  // x1*sin     │
│                                                             │
│      c. axis 选择并累加（使用 Axpy）                       │
│         // 对每个 axis（temporal/height/width）：         │
│         Axpy(x1New, p0, p1, -1.0f, axisMask[axis], ...)    │
│         Axpy(x2New, p2, p3,  1.0f, axisMask[axis], ...)    │
│         // 三个 axis 的结果叠加，未匹配的 axis mask=0     │
│                                                             │
│   4. Narrow: fp32 → bf16                                   │
│      Cast(outputBf16, outputFp32, totalLen)                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: UB → GM (四路输出)                                │
│   - DataCopy: qOutput[token * qSize]                       │
│   - DataCopy: kOutput[token * kvSize]                      │
│   - DataCopy: vOutput[token * kvSize]  (直通，无修改)     │
│   - DataCopy: gateOutput[token * qSize] (直通，无修改)    │
└─────────────────────────────────────────────────────────────┘
```

##### 三级流水并行

```cpp
// 伪代码
for (int token = 0; token < numTokens; ++token) {
    pipe_barrier(PIPE_ALL);  // 同步上一个 token
    
    // Stage 0: GM → UB
    if (token < numTokens) {
        CopyIn(token);
    }
    
    // Stage 1: Compute
    if (token - 1 >= 0) {
        Compute(token - 1);
    }
    
    // Stage 2: UB → GM
    if (token - 2 >= 0) {
        CopyOut(token - 2);
    }
}
```

**流水效果**：隐藏 GM 访存延迟（~100 ns/块）

##### 关键优化技术

**1. 向量化规约（树形折叠 + ReduceRepeat）**

问题：原始设计中，最后 8 个 partial sum 用 scalar GetValue 读取，每次 ~30 ns，总计 160 * 30 ns = 4.8 μs。

优化：
```cpp
// 树形折叠到 8 个 partial（256 → 128 → 64 → ... → 8）
// 然后用 ReduceRepeat 一次性收尾
ReduceRepeat<SUM>(
    result,           // [numHeads]，输出
    sq,               // [numHeads * 8]，输入（8 个 partial/head）
    8,                // 每个 head 规约 8 个元素
    numHeads,         // 20 个 head
    1,                // src block stride = 1
    1,                // dst block stride = 1
    headSize / 8      // src repeat stride = 32 blocks（下一 head）
);
// 单条向量指令，~17 ns，加速 282×
```

**2. Chunked 归一化（跨 head 向量化）**

问题：原始设计中，每 head 独立处理，循环 20 次，每次 4 条向量指令，总计 160 条。

优化：
```cpp
// 以 64 元素为 chunk，跨 head 处理
// repeat stride = headSize：每个 repeat 跳到下一 head 的对应位置
BinaryRepeatParams chunkParams(
    1,                    // dst block stride = 1（连续写）
    1,                    // src0 block stride = 1（x 连续读）
    0,                    // src1 block stride = 0（scale 广播）
    headSize / 8,         // dst repeat stride = 32 blocks（下一 head）
    headSize / 8,         // src0 repeat stride = 32 blocks
    1                     // src1 repeat stride = 1 block（下一 head 的 scale）
);

for (int chunk = 0; chunk < 4; ++chunk) {
    Mul(output[chunk * 64], x[chunk * 64], scaleBlk, 64, numHeads, chunkParams);
}
// 4 个 chunk × 1 条指令/chunk = 4 条指令（降低 40×）
```

**3. 精确倒平方根（Sqrt + Div vs Rsqrt）**

问题：`Rsqrt` 是近似指令，误差 ~1e-5，实测导致 13% 元素超过 1e-3 阈值。

优化：
```cpp
// 使用精确实现
Sqrt(tmp, var, numHeads);           // 误差 < 0.5 ulp
Div(scale, ones, tmp, numHeads);    // 误差 < 0.5 ulp
// 总误差 < 1 ulp（vs Rsqrt 的 ~1e-5）

// 性能代价：2 条指令 vs 1 条，增加 ~17 ns/token
// 但精度从 13% 超差降到 0.0006% 超差
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2 (Ascend 910B) | √ |

## 算子约束限制

1. **数据类型**：仅支持 bf16（输入/输出），内部计算 fp32
2. **head_size**：固定 256
3. **rope_dim**：固定 64
4. **numQHeads**：推荐 16 或 32
5. **numKvHeads**：推荐 2、4 或 8
6. **num_tokens**：≤ 8192（受单核 UB 限制）
7. **不支持动态 shape**：所有维度在编译时确定
8. **不支持广播**：输入形状必须严格匹配 Qwen3-VL 规范

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 最大差异 < 1e-3，V/Gate 逐位一致 | 任务书 §3.2 |
| 性能标准 | 加速比 ≥ 2.0× (vs Triton baseline) | 任务书 §3.3 |
| 确定性标准 | 5 次重复启动逐位一致 | 任务书 §3.2 |

**实测结果**：

| 指标 | 要求 | 实际 | 状态 |
|---|---|---|---|
| 精度（不一致率） | - | 0.000613% | ✓ |
| 精度（max_diff） | < 1e-3 | 0.015625 (1 ulp) | ⚠️ 见注 |
| V/Gate 精度 | - | 逐位一致 | ✓ |
| 性能（最小加速比） | ≥ 2.0× | 2.23× | ✓ PASS |
| 性能（decode 场景） | ≥ 2.0× | 24.5× | ✓ PASS |
| 确定性 | - | 5 次启动一致 | ✓ PASS |

**注**：245 个元素（占 0.000481%）的差异超过 1e-3，但都 ≤ 1 bf16 ulp。原因是 bf16 在 |x| ≥ 0.125 处的步长本身 > 1e-3（在 |x|=0.25 处为 1.95e-3，在 |x|=1.0 处为 7.81e-3）。两个都正确的 bf16 实现在舍入边界上选择不同方向，差异必然超过 1e-3。实际精度已达 bf16 数值表示极限。

## 兼容性分析

新算子，不涉及兼容性分析。

---

**设计文档版本**：v1.0  
**编写日期**：2026-09-25  
**作者**：Irene-  
**联系方式**：https://gitcode/Irene-/split_qkv_rmsnorm_mrope

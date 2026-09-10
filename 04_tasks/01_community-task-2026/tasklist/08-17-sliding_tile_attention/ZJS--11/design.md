# SlidingTileAttention 算子设计文档

# 需求背景（required）

## 需求来源

社区任务：参考 FastVideo 的 sliding_tile_attention 实现
（https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/ops.py#L14），
在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的算子，完成算子设计、开发、测试全流程，
验收通过后提交至昇腾算子开源仓（ops-transformer/experimental/attention）。

## 背景介绍

### SlidingTileAttention 算子实现分析

SlidingTileAttention（滑窗 3D 注意力）是视频生成 DiT 模型（HunyuanVideo / StepVideo / Wan 等）
中用于降低长序列注意力计算量的稀疏注意力算子。视频 token 按 3D 网格 (T, H, W) 排列，
每个 attention head 拥有独立的 3D 滑窗尺寸 (kt, kh, kw)，每个 query 只与以自身为中心的
3D 窗口范围内的 key/value 计算注意力；序列尾部拼接的 text token 对所有 query 全局可见。

基线实现：FastVideo 仓 fastvideo-kernel 中的 `sliding_tile_attention`
（CUDA ThunderKittens kernel `sta_fwd`，triton fallback `st_attn_triton.py`，两者语义一致）。

### FastVideo 参考实现功能分析

通过对参考实现（commit 4ddcdf5）的功能分析，当前支持的能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| q | 输入 query tensor | tensor | float16, bfloat16 | 4 维 BNSD 连续 | (B,N,S,D) |
| k | 输入 key tensor | tensor | float16, bfloat16 | 与 q 一致 | (B,N,S,D) |
| v | 输入 value tensor | tensor | float16, bfloat16 | 与 q 一致 | (B,N,S,D) |
| window_size | 每个 head 的 (t,h,w) 窗口 | list[list[int]] | int | 正奇数；长度 1 广播或 N 逐 head | - |
| text_length | text token 数 | scalar | int | ≥0；has_text=false 按 0 | - |
| has_text | 是否存在 text token | scalar | bool | - | - |
| seq_shape | image 三维尺寸 | string | - | "TxHxW"，需被 tile (6,8,8) 整除 | 如 30x48x80 |
| output | 输出 tensor | tensor | float16, bfloat16 | 与 q 一致 | (B,N,S,D) |

参考实现的关键语义（以代码实际行为为准，已交叉核对 CUDA 与 triton 两个实现）：

1. **序列布局**：image 段在前（img_len = T*H*W 个 token），text 段在后
   （下标 ∈ [img_len, img_len + text_length)）。
2. **tile 粒度窗口**：image 网格按 6×8×8 划分 tile（384 token/tile），同一 tile 内所有 query
   共享同一窗口：对 query 所在 tile 坐标 (tt,th,tw)，逐维度将窗口中心 clamp 到合法范围
   （center = clamp(t_idx, k//2, num_tiles-1-k//2)），可见 kv tile 为
   [center-k//2, center+k//2] 闭区间的所有 tile。
3. **text 全局可见**：image query 额外可见全部 text token；text 段 query（下标 ≥ img_len）
   对全部 image tile + 全部 text token 做全局注意力。
4. **padding**：has_text=true 时参考实现内部把 S padding 到 384 的倍数再截断，
   对输出 [0, S) 无任何影响（实现 artifact，本算子不需要）。
5. scale = 1/√D，softmax 数值稳定实现（在线 max/sum 归一），fp32 累加。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 SlidingTileAttention 算子，aclnn 工程化（registry-invoke）方式开发：

1. 功能与 FastVideo.sliding_tile_attention 完全对齐（语义以上述代码分析为准）；
2. 支持 float16、bfloat16 数据类型，数据格式 BNSD；
3. 支持确定性计算（相同输入多次执行结果一致）；
4. 精度满足生态算子开源精度标准，输出与基线接口一致；
5. 性能：fp16/bf16 场景下基于 910B3 与 0.8×【FastVideo 接口 + GPU A100】持平
   （以 ATK 同用例 NPU/GPU 实测对比为准）。

## 需求拆解

1. 支持 float16、bfloat16 两种 dtype；
2. 支持 per-head 窗口（window_size 长度 1 广播 / 长度 N 逐 head）；
3. 支持 has_text 两种模式、三种典型 seq_shape（30x48x80 / 36x48x48 / 18x48x80）
   及任意被 tile (6,8,8) 整除的 canvas；
4. 支持 D ∈ {16, 32, 64, 128}，B/N/S 任意合法值；
5. aclnn 两段式接口，参数校验完备；
6. 精度：fp16 rtol=atol=2^-9、bf16 rtol=atol=2^-6，matched_ratio ≥ 0.99。

# 详细设计（required）

## 算子分析

### 数学公式

对于每个 head n（窗口 (kt,kh,kw)），Q,K,V ∈ R^{B×N×S×D}，scale = 1/√D：

1. score_{i,j} = (Q_i · K_j) × scale；
2. 掩码 Mask：image 段 query i（所在 tile 坐标 (tt,th,tw)）可见 key j 当且仅当
   - j 属于 text 段（j ∈ [img_len, img_len+text_length)），或
   - j 属于 image 段且 j 所在 tile (jt,jh,jw) 满足逐维度
     |clamp(td, kd//2, nd-1-kd//2) - jd| ≤ kd//2（d ∈ {t,h,w}，nd 为该维 tile 数）；
   text 段 query（i ≥ img_len）可见全部 key j ∈ [0, img_len+text_length)；
3. O_i = Σ_{j: Mask=1} softmax_j(score_{i,j}) · V_j。

### 支持数据类型

float16、bfloat16（q/k/v/output 必须一致）。

### 支持形状

- q/k/v/output 均为 4 维 [B,N,S,D]，shape 完全一致，BNSD 连续，不支持 broadcast；
- B,N,S,D > 0；S ≥ img_len + (has_text ? text_length : 0)，img_len = T*H*W；
- seq_shape 的 T/H/W 须分别被 6/8/8 整除；
- D ∈ {16, 32, 64, 128}；N ≤ 1024（tiling data 定长数组约束）。

## 算子实现

### 实现方案

整体采用 flash attention 式单趟在线 softmax 结构：每个 query block 只遍历其可见的
kv tile（滑窗），避免构造 S×S 注意力矩阵，计算量与窗口大小成正比。

#### host 侧设计

**tiling 策略**：

1. **参数解析与校验**：解析 q shape（B,N,S,D）与属性 window_size/text_length/has_text/seq_shape；
   完成 §算子约束限制 的全部校验；window_size 展开为 per-head (kt,kh,kw) 存入 tiling data
   （长度 3 广播到所有 head，长度 3N 逐 head）。
2. **工作单元划分**：BLOCK_Q = 128（一个单元 = 128 个连续 query）：
   - image 单元数 = B×N×numTiles×3（每个 6×8×8 tile 恰好 3 个 query block，384=3×128）；
   - text 单元数 = B×N×ceil((S-img_len)/128)（[img_len, S) 段 query，全局注意力）；
   - 所有单元拉通编号，按核数静态均分（连续分段，余数分配到前几个核）。
3. **分核策略（mix kernel 适配）**：本 kernel 使用 matmul 高阶 API，910B 上为
   MatmulClient 混合 kernel（AIC:AIV = 1:2），SetBlockDim 语义为 AIC block 数，
   kernel 内 GetBlockIdx() 为扁平 AIV 编号；工作单元按 aivNum 个 AIV 槽位均分，
   blockDim 折算为 CeilDiv(usedCores, aivPerAic)，避免超订挂死。
4. **cube tiling**：通过 MatmulApiTiling 分别计算 QK^T（M=128, N=128, K=D，B 逻辑转置，
   C 输出 fp32 至 UB）与 PV（M=128, N=D, K=128，A 在 UB，C 输出 fp32 至 UB）的 tiling，
   随 tiling data 下发。
5. **tilingkey 规划**：ASCENDC_TPL_SEL_PARAM(dtype, headDim)，kernel 按 (T, HEAD_DIM)
   模板实例化，编译期确定 matmul 形状与 UB 预算。

**workspace**：仅声明 matmul 高阶 API 所需的系统 workspace（32MB），算子本身无额外 GM workspace。

#### kernel 侧设计

每个工作单元（128 行 query block）的处理流程：

1. **单元解码**：unit id → (b, n, tile_idx, sub_block)；image 单元由 tile 坐标和该 head 窗口
   计算 kv tile 范围 [(t0,t1),(h0,h1),(w0,w1)]（中心 clamp 语义）；text 单元取全 canvas 范围。
2. **初始化**：acc[128,D] fp32 清零，行最大值 m[128] 置 -FLT_MAX，行分母 l[128] 清零。
3. **kv 循环**（固定 t→h→w→sub 顺序，保证确定性）：
   - QK^T：matmul Q[128,D] × K^T[128,D] → scores[128,128] fp32（fixpipe 直写 UB）；
   - 在线 softmax（全向量化，无标量回读）：
     Muls 乘 scale →（text 尾块列 mask 置 -FLT_MAX）→ 二元 Max 折叠 + WholeReduceMax
     得行最大值 → Max/Sub/Exp 更新 alpha → Brcb 行广播 + 零步长 Sub + Exp 得
     p=exp(score-mNew) → Add 折叠 + WholeReduceSum 得行和累入 l → Cast 回输入 dtype；
   - PV：matmul P[128,128] × V[128,D] → fp32，acc 先按行广播乘 alpha 再累加。
   - text kv 追加在 image kv 之后（尾部 block 按 text_length 做列 mask）。
4. **收尾**：out = acc × Reciprocal(l)（Brcb 行广播 Mul），Cast 回输入 dtype，
   经 32 行 staging buffer 分块 DataCopyPad 写回 GM（V→MTE3 同步，D=128 时控制 UB 占用）。

**UB 预算（BLOCK=128, D=64, fp16，共约 178KB / 192KB）**：
scores fp32 64KB + p fp16 32KB + acc fp32 32KB + stat ~6.5KB + out staging 4KB + matmul 内部开销。

**确定性**：kv 遍历顺序固定、单趟累加、无核间通信写输出，多次执行结果一致
（fp16 实测逐位一致）。

**关键同步纪律**：fixpipe（matmul C→UB）后接向量计算需正确的事件同步；
Cast（V 管道）后接 DataCopyPad（MTE3 管道）必须 SetFlag/WaitFlag(V_MTE3)；
bf16 的 fp32→bf16 Cast 使用 CAST_RINT 舍入。

### aclnn 接口

```c
aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, aclTensor *output,
    const aclIntArray *const *windowSize, uint64_t windowSizeLen,
    int64_t textLength, bool hasText, const char *seqShape,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnSlidingTileAttention(void *workspace, uint64_t workspaceSize,
                                      aclOpExecutor *executor, aclrtStream stream);
```

L2 层完成参数校验（非空/dtype/format/shape 一致性）、windowSize list_list_int 扁平化
（兼容长度 1 广播、长度 N 逐 head、单扁平数组 3/3N 三种传法）、输入 contiguous 化、
L0 调用与 ViewCopy 到输出。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列（ascend910b，arch22，实测 910B3） | √ |

## 算子约束限制

1. q/k/v/output 必须 4 维 [B,N,S,D]、shape 一致、dtype 一致（fp16/bf16），不支持 broadcast；
2. B,N,S,D > 0；S ≥ img_len + text_length（has_text=false 时 text_length 按 0 计）；
3. seq_shape 格式 "TxHxW"，且 T%6==0、H%8==0、W%8==0；
4. window_size 每个元素 > 0（语义上为正奇数窗口，偶数按 k//2 向下取整兼容）；
   长度须为 1（广播）或 N（逐 head）；N ≤ 1024；
5. D ∈ {16, 32, 64, 128}；
6. 不支持 private format；输入非连续时由 aclnn 层 contiguous 化。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 与 FastVideo.sliding_tile_attention 输出一致；fp16 rtol=atol=2^-9、bf16 rtol=atol=2^-6，matched_ratio≥0.99，max_abs_err≤0.1(fp16)/1.0(bf16) | 任务书 §3.2 + 生态算子开源精度标准 |
| 性能标准 | fp16/bf16 下 910B3 性能与 0.8×【FastVideo 接口 + GPU A100】持平（ATK 同用例实测对比） | 任务书 §3.3 |
| 确定性 | 相同输入多次执行结果一致 | 任务书 §2.1 |

测试方案：tests/acceptance/ 下 12 例 golden 交叉验证（GPU 侧 FastVideo 官方实现生成 golden，
NPU 侧 aclnn 示例运行比对），覆盖三种 seq_shape、fp16/bf16、D∈{16,32,64}、
逐 head/广播窗口、有/无 text、S 尾部多余 token 等边界；另有 16 例小规模回归
（含 D=128、多核均分尾块、text 奇数长度对齐等）。

## 兼容性分析

新算子，不涉及兼容性分析。

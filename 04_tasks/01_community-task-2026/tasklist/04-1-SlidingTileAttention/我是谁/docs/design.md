# SlidingTileAttention 算子设计文档

# 需求背景

## 需求来源

社区任务：参考 FastVideo 中 `sliding_tile_attention` 的 GPU/Triton 实现，在
Atlas A2 训练系列产品上使用 Ascend C 实现功能一致的 SlidingTileAttention
自定义算子。

## 背景介绍

### SlidingTileAttention 算子实现优化

SlidingTileAttention 是面向长视频序列的局部窗口注意力算子。视频生成模型中
的视频 token 序列通常很长，若直接做全量 self-attention，计算量与显存访问量
都会随序列长度平方增长。FastVideo 的 SlidingTileAttention 将视频 token 按
固定 3D tile 划分，每个视频 query 只关注自身附近的局部视频 KV tile；当输入
包含文本 token 时，视频 query 还需要关注所有文本 KV，文本 query 则关注整个
有效序列。

原始 GPU 版本来自 FastVideo，相关链接如下：

- Python API：<https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/ops.py>
- Triton kernel：<https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/st_attn_triton.py>

本设计需要在昇腾 NPU 上实现等价语义，支持 `float16`、`bfloat16` 和 BNSD
输入输出格式，并满足社区任务要求的精度、泛化和性能验收。

### GPU 实现现状分析

FastVideo 源接口为：

```python
sliding_tile_attention(q, k, v, window_size, text_length, has_text, seq_shape)
```

输入、属性和输出如下：

| 参数 | 参数含义 | 类型 | 支持数据类型 | 约束 | 形状/取值 |
| --- | --- | --- | --- | --- | --- |
| q | Query 输入 | tensor | float16, bfloat16 | 与 k/v 的 B/N/S/D 一致 | `(B, N, S, D)` |
| k | Key 输入 | tensor | float16, bfloat16 | 与 q/v 的 B/N/S/D 一致 | `(B, N, S, D)` |
| v | Value 输入 | tensor | float16, bfloat16 | 与 q/k 的 B/N/S/D 一致 | `(B, N, S, D)` |
| window_size | 每个 head 的 3D 窗口 | list[tuple[int, int, int]] | - | 长度等于 head 数 | `[(Kt, Kh, Kw), ...]` |
| text_length | 文本 token 长度 | int | - | `has_text=true` 时有效 | `0 <= text_length <= 256` |
| has_text | 是否包含文本 token | bool | - | 控制文本分支 | `true/false` |
| seq_shape | 视频帧网格形状 | string | - | 当前任务涉及 3 类配置 | `"30x48x80"`、`"36x48x48"`、`"18x48x80"` |
| output | 输出 | tensor | float16, bfloat16 | dtype 与输入一致 | `(B, N, S, D)` |

源实现主要流程如下：

1. 根据 `seq_shape` 解析视频画布尺寸，固定 tile 尺寸为 `(6, 8, 8)`，单个
   tile 包含 384 个 token。
2. 当 `has_text=true` 时，输入序列由 `img_seq_len` 个视频 token 和
   `text_length` 个文本 token 组成。源实现会按 384 对齐序列长度，最终输出
   裁剪回原始 `S`。
3. 视频 query 分支逐 head 计算。每个 head 使用自身窗口 `(Kt, Kh, Kw)`，
   对 query tile 计算 clamp 后的固定大小邻域，并遍历该邻域内的视频 KV tile。
4. 当 `has_text=true` 时，视频 query 在局部视频 KV 之后继续关注所有文本 KV。
5. 文本 query 分支仅在 `has_text=true` 时执行。文本 query 不使用局部窗口，
   直接关注所有视频 KV 和所有有效文本 KV。
6. 源 kernel 使用 online softmax 累加方式，不保存完整 score 矩阵。

### 算子功能分析

SlidingTileAttention 是“视频局部滑动窗口 + 文本全局可见”的 FlashAttention
变体。

- 视频 token：将视频序列按 `(tile_t, tile_h, tile_w)=(6,8,8)` 划分为 3D
  tile。每个视频 query tile 只和当前 head 窗口内的邻近视频 KV tile 计算
  attention。
- 文本 token：当 `has_text=true` 时，视频 query 关注全部文本 KV；文本 query
  关注全部视频 KV 和文本 KV。
- 多 head 窗口：`window_size` 按 head 配置，不同 head 可使用不同
  `(Kt, Kh, Kw)`。

输入：`q`、`k`、`v`

属性：`window_size`、`text_length`、`has_text`、`seq_shape`

输出：`output`

支持数据类型：`float16`、`bfloat16`

支持数据格式：BNSD，即 `(B, N, S, D)`

# 需求分析

## 需求描述

使用 Ascend C 实现 SlidingTileAttention 算子，使其功能与 FastVideo 源实现
对齐。算子需支持 `float16` 和 `bfloat16`，支持 BNSD 数据格式，内部 score、
softmax 和 `P @ V` 累加使用 `float32`，并满足任务书中的精度和性能要求。

## 需求拆解

1. 输入 `q/k/v` 为同设备、连续 BNSD Tensor，shape 和 dtype 必须一致。
2. 输出 `output` 与 `q` 的 shape、layout、dtype 一致。
3. 支持 `float16` 和 `bfloat16` 输入，计算中使用 `float32` 保存 score、
   softmax 中间值、归一化因子和累加器。
4. 支持 HunyuanVideo `30x48x80`、StepVideo `36x48x48`、Wan `18x48x80`
   三类视频画布语义。
5. 支持每个 head 独立配置 `window_size`，窗口维度按 tile 单位解释。
6. 支持 `has_text=true` 时的视频 + 文本混合 attention，并正确处理 text tail
   和按 384 对齐产生的 padding 语义。
7. 支持常规场景、边界 tile clamp、不同窗口组合、`float16/bfloat16`、有文本
   和无文本路径的测试覆盖。
8. 正式验收实现不得调用 FastVideo、Triton/CUDA、FlexAttention、PyTorch
   组合算子、CPU 参考实现或其他后端作为得分路径。

## 接口定义

正式开源算子工程建议采用 aclnn 接口，对外提供如下接口。Python 源实现中的
`window_size` 和 `seq_shape` 在 aclnn 接口中转为稳定属性：

```cpp
aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(
    const aclTensor* q,
    const aclTensor* k,
    const aclTensor* v,
    const aclIntArray* windowSize,
    int64_t textLength,
    bool hasText,
    int64_t seqShape,
    const aclTensor* output,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnSlidingTileAttention(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

参数说明如下：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| q/k/v | `aclTensor*` | BNSD 输入 Tensor，shape 为 `(B, N, S, D)` |
| windowSize | `aclIntArray*` | 展平后的窗口数组，长度为 `3 * N`，第 `i` 个 head 的窗口为 `(windowSize[3*i], windowSize[3*i+1], windowSize[3*i+2])` |
| textLength | `int64_t` | 文本 token 数，`hasText=true` 时参与计算 |
| hasText | `bool` | 是否包含文本 token |
| seqShape | `int64_t` | 视频画布枚举：`1` 表示 `30x48x80`，`2` 表示 `36x48x48`，`3` 表示 `18x48x80` |
| output | `aclTensor*` | 输出 Tensor，shape 和 dtype 与 `q` 一致 |

当前开发实现曾使用 direct-launch torch custom op 作为调试封装，Python 入口
只做 metadata 规范化，并调用本地 Ascend C kernel。该封装用于确认语义、调度
和边界条件；正式提交到 ops-transformer 时应按上表整理为 aclnn 工程接口。

# 详细设计

## 算子分析

### 数学公式

对于视频区域中的第 X 个 query tile，其输出为：

$$
Output[X] =
\operatorname{Softmax}\left(
\frac{Q_{img}[X] \cdot [K_{img}[Neighbors], K_{text}[All]]^T}{\sqrt{D}}
\right)
\cdot [V_{img}[Neighbors], V_{text}[All]]
$$

对于文本区域中的第 Y 个 query block，其输出为：

$$
Output[Y] =
\operatorname{Softmax}\left(
\frac{Q_{text}[Y] \cdot [K_{img}[All], K_{text}[All]]^T}{\sqrt{D}}
\right)
\cdot [V_{img}[All], V_{text}[All]]
$$

当 `has_text=false` 时，文本 KV 分支和文本 query 分支均不参与计算。

### 支持数据类型

| 输入 dtype | 中间计算 dtype | 输出 dtype |
| --- | --- | --- |
| float16 | float32 | float16 |
| bfloat16 | float32 | bfloat16 |

### 支持形状

| seqShape | 场景 | canvas_t | canvas_h | canvas_w | img_seq_len | 文本支持 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | HunyuanVideo | 30 | 48 | 80 | 115200 | 支持，`textLength <= 256` |
| 2 | StepVideo | 36 | 48 | 48 | 82944 | 无文本 |
| 3 | Wan | 18 | 48 | 80 | 69120 | 无文本 |

固定 tile 尺寸为 `(tile_t, tile_h, tile_w)=(6,8,8)`，单 tile 包含 384 个 token。

### Shape 与序列布局

1. BNSD 中 `S` 维的前 `img_seq_len` 个 token 为视频 token，按 tile 顺序连续
   存放。
2. `hasText=true` 时仅支持 `seqShape=1`。此时 `img_seq_len=115200`，合法输入
   满足 `0 < textLength <= 256` 且 `S=img_seq_len+textLength`。
3. `hasText=false` 时不包含文本 token，合法输入满足 `textLength=0` 且
   `S=img_seq_len`。
4. 为对齐源实现，`hasText=true` 时内部可按 `alignedS=ceil(S/384)*384` 组织
   text query block。`alignedS-S` 为 padding token 数，不属于输出有效范围。
5. 本仓库的短序列回归用例只用于快速验证功能和边界行为；正式性能验收以任务书
   中的长序列 shape 为准，不能以缩小后的短序列替代正式输入。

## 算子实现

### Host 侧设计

Host 侧负责参数校验、场景解析、tiling 数据构造和 kernel 发射。

1. 参数校验：校验 `q/k/v/output` 的 dtype、shape、format 一致；校验
   `windowSize` 长度与 head 数匹配；校验 `seqShape`、`hasText`、`textLength`
   组合合法。
2. 形状推导：输出 shape 与输入 `q` 保持一致。`hasText=true` 且 `S` 未按 384
   对齐时，kernel 内部按对齐后 block 组织任务，但只读写有效 `[0,S)` 范围。
3. 场景映射：根据 `seqShape` 解析 `canvasT/canvasH/canvasW`，计算
   `imgSeqLen`、`numTilesT/H/W`、`numTiles` 和 `tileNumel`。
4. 窗口整理：将 `windowSize` 保存为长度 `3 * headNum` 的数组，kernel 根据
   当前 head 读取 `(Kt, Kh, Kw)`。
5. 分核策略：以 `(B, N, queryBlock)` 为基本任务粒度。视频 query 分支覆盖所有
   image tile；文本 query 分支覆盖 text tail。
6. tiling key 规划：至少区分 `hasText=false` 与 `hasText=true` 两类主路径，并可
   进一步按 dtype、head_dim、blockQ/blockKv 配置扩展。

Host 侧详细校验如下：

| 校验项 | 合法条件 |
| --- | --- |
| dtype | `q/k/v/output` dtype 一致，且为 `float16` 或 `bfloat16` |
| format | 仅支持 BNSD，rank 必须为 4 |
| shape | `q/k/v/output` 的 `B/N/S/D` 一致，`B>=1`，`N>=1`，`D>0` |
| head_dim | 基础性能路径支持 `D=64`；若扩展其他 `D`，需补充 block 配置和精度验证 |
| windowSize | 长度为 `3*N`；每个窗口维度为正整数，建议提交版本约束为正奇数 |
| textLength | `hasText=true` 时 `0 < textLength <= 256`；`hasText=false` 时 `textLength=0` |
| seqShape | `hasText=true` 支持 `seqShape=1`；`hasText=false` 支持 `seqShape=2/3` |
| S | `hasText=true` 时 `S=115200+textLength`；`hasText=false` 时 `S=imgSeqLen` |

本地验证实现中，Python 入口支持单个 window triple 广播到 8 个 head；C++
wrapper 接收展平后的 8 个 window triple，并要求 `N=8`、`D=64`。若正式提交
版本继续保留该能力边界，需要在 README 和自验证报告中明确说明。

### Tiling 数据

Tiling 数据至少包含以下字段：

| 字段 | 说明 |
| --- | --- |
| `batch`, `headNum`, `seqLen`, `alignedSeqLen`, `headDim` | 基础 shape 信息 |
| `imgSeqLen`, `textLength`, `hasText`, `seqShape` | 场景和文本信息 |
| `canvasT/H/W`, `tileT/H/W`, `numTilesT/H/W`, `numTiles` | 3D tile 映射信息 |
| `blockQ`, `blockKv`, `blockDim` | kernel 内部 block 配置 |
| `windowSize[3 * headNum]` | 每个 head 的窗口配置 |
| `videoTaskNum`, `textTaskNum`, `taskStart`, `taskEnd` | 分核任务范围 |

### Kernel 侧设计

Kernel 侧完成实际 attention 计算。

1. dtype 分发：根据输入 dtype 选择 `half` 或 `bfloat16_t` 模板实例，内部
   score、softmax、denominator、accumulator 使用 `float`。
2. 视频 query 分支：根据当前 query tile 的 3D 坐标和当前 head 的
   `(Kt, Kh, Kw)` 计算局部 KV tile 范围。边界 tile 使用 clamp 逻辑移动窗口中心，
   避免越界并保持有效窗口大小。
3. 文本 KV 分支：`hasText=true` 时，视频 query 在遍历局部视频 KV 后继续遍历
   `[imgSeqLen, imgSeqLen+textLength)` 范围内的文本 KV。
4. 文本 query 分支：`hasText=true` 时，文本 query 遍历全部有效视频 KV 和文本 KV，
   完成全局 attention。
5. FlashAttention 累加：每处理一个 KV block，执行 `QK^T`、scale、mask、
   row max 更新、softmax denominator 更新和 `P @ V` 累加，避免保存完整 score
   矩阵。
6. 输出写回：计算 `acc / l` 后转换为输入 dtype 写回 output；padding query 或
   padding KV 不参与有效输出。

### 当前开发实现的调度细节

当前开发实现中，Python 入口只做 metadata 规范化，不执行 attention 计算；C++
插件负责 host 校验并发射本地 Ascend C kernel，实际计算在 Ascend C kernel 中
完成。

验证实现采用 query chunk 调度：

- `S <= 320` 时按 metadata 派生的 query chunk 分配任务。image chunk 位于同一
  image tile 内，共享 KV window；text chunk 关注整个有效序列。
- 常见 `qRows=16/8/4` 使用固定展开路径，降低动态行循环开销。
- `S > 320` 时按 `kLargeLaunchChunks=1024` 分段发射 chunk，使用 streaming
  row-softmax：第一遍遍历 KV 求 row max，第二遍计算 softmax denominator 和
  `P @ V`。

该验证实现仍属于 GM scalar-heavy 路径，主要用于证明语义和形成后续优化基线。
正式高性能版本建议继续向 UB 向量化 reduction 或 CUBE 辅助 QK/PV 方向演进。

### Padding 处理策略

1. kernel 不访问原始输入 `S` 以外地址。对齐到 384 只用于任务组织，不要求实际
   读取 padding token。
2. 文本 KV 只访问 `[imgSeqLen, imgSeqLen+textLength)` 的有效范围，padding KV
   不参与 softmax。
3. 当 `textLength=0` 时，不产生文本 KV 和文本 query 有效任务。

### 边界窗口处理

窗口中心按如下逻辑 clamp：

```text
radius = window / 2
maxCenter = extent - 1 - radius
center = min(coord, maxCenter)
center = max(center, radius)
```

随后枚举 `[center-radius, center+radius]` 范围内的 tile。正式提交版本建议将
`windowSize` 约束为正奇数，避免偶数窗口带来的 floor-radius 解释歧义。若保留
偶数窗口，必须在 README 和测试说明中显式描述其语义。

## 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |

## 算子约束限制

1. 当前仅支持 BNSD 格式。
2. 当前仅支持 `float16` 和 `bfloat16`。
3. 当前仅支持 `seqShape=1/2/3`，分别对应 `30x48x80`、`36x48x48`、
   `18x48x80`。
4. `hasText=true` 仅支持 `30x48x80`，`textLength <= 256`。
5. `hasText=false` 时，`36x48x48` 要求 `S=82944`，`18x48x80` 要求
   `S=69120`。
6. tile 尺寸固定为 `(6,8,8)`。
7. `windowSize` 长度必须为 `3 * headNum`；正式提交版本建议每个窗口维度为正
   奇数。
8. 基础性能验收场景中 `B=1`、`N=8`、`D=64`。如实现声明支持其他 `B/N/D`，
   需补充对应自验证记录。
9. 正式验收路径不得调用 FastVideo、Triton/CUDA、FlexAttention、PyTorch
   attention、CPU 参考实现或其他后端作为整算子 fallback。

# 可维护可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足 AscendOpTest 默认精度阈值 | AscendOpTest |
| 性能标准 | 两个正式性能场景平均耗时不高于 0.8 倍 A100 基准，即约 `655.97us` 和 `5771us` | 社区任务书 |

## 测试用例设计

测试覆盖应至少包含以下场景：

| 类型 | 覆盖点 |
| --- | --- |
| dtype | `float16`、`bfloat16` |
| 文本路径 | `hasText=true`，`textLength=128/256` |
| 无文本路径 | `36x48x48`、`18x48x80` |
| 窗口配置 | 单窗口广播、每 head 独立 window mix |
| 边界行为 | 首尾 tile、空间/时间边界 clamp |
| 性能目标 | `30x48x80+textLength=128` 和 `36x48x48` 两个正式长序列场景 |
| 泛化 | 非性能小 shape 功能测试、不同 batch、不同 value range |

## 自验证报告补充要求

自验证报告需要与本文接口和约束保持一致，至少补齐以下材料：

1. aclnn 工程构建、安装、调用样例和完整测试执行日志。
2. `float16`、`bfloat16` 两类 dtype 的功能测试结果。
3. `hasText=true`、`hasText=false`、单窗口广播、per-head window mix、首尾
   tile clamp、`textLength=128/256` 等覆盖项。
4. 两个正式长序列性能场景的耗时截图和 profiling 记录。
5. 当前实现若保留 `N=8/D=64`、正整数窗口或 direct-launch 调试封装，需要在
   README、自验证报告和提交说明中写明能力边界；正式验收路径不得使用调试封装、
   Python 参考实现或其他后端 fallback。

## 兼容性分析

SlidingTileAttention 是新增 experimental 算子，不涉及历史版本兼容问题。接口
和工程结构应与 ops-transformer experimental attention 类算子的 aclnn 调用方式
保持一致，便于 README、样例和自验证报告复现。

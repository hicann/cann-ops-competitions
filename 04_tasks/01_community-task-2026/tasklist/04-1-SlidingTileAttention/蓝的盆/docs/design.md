# SlidingTileAttention 算子设计文档

# 需求背景

## 需求来源

社区任务：基于 FastVideo 中 `sliding_tile_attention` 的 GPU/Triton 实现，使用 Ascend C 在 Atlas A2 训练系列产品上实现功能一致的自定义算子。

## 背景介绍

### SlidingTileAttention 算子实现优化

SlidingTileAttention 是面向长视频序列的局部窗口注意力算子。原始 GPU 版本来自 FastVideo，其 Python API 和 Triton kernel 链接如下：

- Python API：<https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/ops.py>
- Triton kernel：<https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/st_attn_triton.py>

本任务需要在昇腾 NPU 上基于 Ascend C 实现等价功能，并满足精度、泛化和性能验收要求。

### GPU 实现现状分析

通过分析 FastVideo 的 Python API 和 Triton 实现，SlidingTileAttention 当前支持的输入、属性和输出如下：

| 参数 | 参数含义 | 类型 | 支持数据类型 | 约束 | 形状/取值 |
| --- | --- | --- | --- | --- | --- |
| q | Query 输入 | tensor | float16, bfloat16 | 与 k、v 的 B/N/S/D 一致 | (B, N, S, D) |
| k | Key 输入 | tensor | float16, bfloat16 | 与 q、v 的 B/N/S/D 一致 | (B, N, S, D) |
| v | Value 输入 | tensor | float16, bfloat16 | 与 q、k 的 B/N/S/D 一致 | (B, N, S, D) |
| window_size | 每个 head 的 3D 窗口 | list[tuple[int, int, int]] | - | 长度等于 head_num | [(Kt, Kh, Kw), ...] |
| text_length | 文本 token 长度 | int | - | `has_text=true` 时有效 | 非负整数，验收场景中不超过 256 |
| has_text | 是否包含文本 token | bool | - | 控制文本全局可见分支 | true/false |
| seq_shape | 视频帧网格形状 | string | - | 当前支持 3 种配置 | "30x48x80"、"36x48x48"、"18x48x80" |
| output | 输出 | tensor | float16, bfloat16 | dtype 与输入一致 | (B, N, S, D) |

源实现流程图如下：
![video.png](https://raw.gitcode.com/user-images/assets/9516645/5993518d-fb2e-47e4-afec-3e9411603553/video.png 'video.png')

1. 当 `has_text=true` 时，输入序列由 `img_seq_len` 个视频 token 和 `text_length` 个文本 token 组成。源实现先将 `S` 维按 384 对齐，padding 内容来自输入尾部 token，最终输出裁剪回原始 `S`。
2. 视频 query 分支逐 head 计算。每个 head 使用各自的 `(Kt, Kh, Kw)`，视频 query tile 仅关注 clamp 后固定大小窗口内的视频 KV tile；当 `has_text=true` 时额外关注全部有效文本 KV。
3. 文本 query 分支仅在 `has_text=true` 时执行。文本 query 关注全部视频 KV 和全部有效文本 KV，窗口参数在该分支不参与计算。
4. 源实现采用 online softmax 累加方式，避免保存完整 score 矩阵；每个 KV block 更新当前行最大值、归一化因子和累加输出。


### 算子功能分析

SlidingTileAttention 是“视频局部滑动窗口 + 文本全局可见”的 FlashAttention 变体。

- 视频 token：将视频序列按固定 `tile_t * tile_h * tile_w = 6 * 8 * 8 = 384` 划分为 3D tile。每个视频 query tile 只与当前 head 对应窗口内的邻近视频 KV tile 做 attention。
- 文本 token：当 `has_text=true` 时，视频 query 除关注局部视频 KV 外，还需要关注全部文本 KV；文本 query 不使用局部窗口，需要关注全部视频 KV 和全部文本 KV。
- 多 head 窗口：`window_size` 的长度必须等于 head 数 `head_num`，不同 head 可使用不同的 `(Kt, Kh, Kw)`。

输入：`q`、`k`、`v`

属性：`window_size`、`text_length`、`has_text`、`seq_shape`

输出：`output`

支持数据类型：`float16`、`bfloat16`

支持数据格式：BNSD，即 `(B, N, S, D)`

# 需求分析

## 需求描述

使用 Ascend C 实现 SlidingTileAttention 算子，功能与 FastVideo 源实现对齐，支持 `float16` 和 `bfloat16`，支持 BNSD 数据格式，并满足验收要求中的泛化、精度和性能指标。

## 需求拆解

1. 支持 BNSD 输入输出布局，输入 `q/k/v` 的 shape 和 dtype 保持一致。
2. 支持 `float16`、`bfloat16` 输入，内部 attention 累加、softmax 最大值和归一化因子使用 `float32`。
3. 支持 3 种视频网格配置：HunyuanVideo `30x48x80`、StepVideo `36x48x48`、Wan `18x48x80`。
4. 支持每个 head 独立配置 `window_size`。
5. 支持 `has_text=true` 时的视频 + 文本混合 attention，并处理按 384 对齐产生的 padding。
6. 支持常规场景和边界场景测试，精度满足 AscendOpTest 默认阈值。
7. 性能不低于 GPU A100 基准性能的 0.8 倍。

## 接口定义

算子采用 aclnn 调用方式，对外提供如下接口。Python 源实现中的 `window_size: list[tuple[int, int, int]]` 和 `seq_shape: str` 在 aclnn 接口中转为可稳定传递的属性。

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

参数说明：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| q/k/v | aclTensor | BNSD 输入 tensor，shape 为 `(B, N, S, D)` |
| windowSize | aclIntArray | 展平后的窗口数组，长度为 `3 * N`，第 `i` 个 head 的窗口为 `(windowSize[3*i], windowSize[3*i+1], windowSize[3*i+2])` |
| textLength | int64_t | 文本 token 数，仅 `hasText=true` 时参与计算 |
| hasText | bool | 是否包含文本 token |
| seqShape | int64_t | 视频网格枚举：`1` 表示 `30x48x80`，`2` 表示 `36x48x48`，`3` 表示 `18x48x80` |
| output | aclTensor | 输出 tensor，shape 和 dtype 与 `q` 一致 |

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

当 `has_text=false` 时，公式中的文本 KV 分支和文本 query 分支均不参与计算。

### 支持数据类型

| 输入 dtype | 中间计算 dtype | 输出 dtype |
| --- | --- | --- |
| float16 | float32 | float16 |
| bfloat16 | float32 | bfloat16 |

### 支持形状

| seq_shape | 场景 | canvas_t | canvas_h | canvas_w | img_seq_len | 是否支持文本 |
| --- | --- | --- | --- | --- | --- | --- |
| 30x48x80 | HunyuanVideo | 30 | 48 | 80 | 115200 | 支持 |
| 36x48x48 | StepVideo | 36 | 48 | 48 | 82944 | 不支持 |
| 18x48x80 | Wan | 18 | 48 | 80 | 69120 | 不支持 |

固定 tile 尺寸为 `(tile_t, tile_h, tile_w) = (6, 8, 8)`，单个 tile 包含 384 个 token。

### Shape 与序列布局

1. BNSD 中 `S` 维的前 `img_seq_len` 个 token 为视频 token，视频 token 按 tile 顺序连续存放。
2. `hasText=true` 仅支持 `seqShape=1`，即 `30x48x80`。此时 `img_seq_len=115200`，合法输入满足 `0 <= textLength <= 256` 且 `S = img_seq_len + textLength`。当 `textLength=0` 时，仅执行视频 query 分支中的视频 KV attention，不执行有效文本 KV 访问。
3. `hasText=false` 时不包含文本 token，合法输入满足 `S = img_seq_len`。当前对齐源实现，仅支持 `seqShape=2` 和 `seqShape=3` 的无文本场景；如需扩展 `30x48x80` 无文本场景，需要在 golden 和验收用例中单独确认。
4. 为对齐源实现，`hasText=true` 时内部按照 `aligned_s = ceil(S / 384) * 384` 组织文本 query block。`aligned_s - S` 为 padding token 数，padding token 不属于输出有效范围。

## 算子实现

### Host 侧设计

1. 参数校验：校验 `q/k/v` 的 dtype、shape、format 一致；校验 `window_size` 长度等于 head 数 `head_num`；校验 `seq_shape`、`has_text`、`text_length` 的组合合法。
2. 形状推导：输出 shape 与输入 q 保持一致，即 `(B, N, S, D)`；`has_text=true` 且 S 未按 384 对齐时，kernel 内部按对齐后长度组织任务，最终仅写回原始 S 范围内的输出。
3. 场景映射：根据 `seq_shape` 解析 `canvas_t/canvas_h/canvas_w`，计算 `img_seq_len`、`num_tiles_t`、`num_tiles_h`、`num_tiles_w` 和 `num_tiles`。
4. tiling 策略：以 `(B, N, query_block)` 为基本任务粒度。视频 query 分支中，`query_block` 覆盖所有视频 tile 内的 Q block；文本 query 分支中，`query_block` 覆盖文本区域的 Q block。
5. 分核策略：优先保证 AI Core 满核。host 侧根据总任务块数、head 数、batch 数和 query block 数进行任务均分，并记录每个核的起止任务范围。
6. tiling key 规划：按 `has_text`、输入 dtype、是否存在文本 query 分支、head_dim/block 配置生成 tiling key。基础设计中至少区分 `has_text=false` 与 `has_text=true` 两类主路径。

Host 侧详细校验如下：

| 校验项 | 合法条件 |
| --- | --- |
| dtype | `q/k/v/output` dtype 一致，且为 `float16` 或 `bfloat16` |
| format | 仅支持 BNSD，要求 4 维输入 |
| shape | `q/k/v/output` 的 `B/N/S/D` 一致，`B >= 1`，`N >= 1`，`D > 0` |
| head_dim | 基础性能路径优先支持 `D=64`；其他 `D` 作为泛化路径时需满足 kernel block 约束并通过精度验证 |
| windowSize | 长度为 `3*N`，每个维度为正奇数，且不得超过对应 tile 网格维度 |
| textLength | `hasText=true` 时满足 `0 <= textLength <= 256`；`hasText=false` 时要求 `textLength=0` 或忽略该属性 |
| seqShape | `hasText=true` 仅支持 `seqShape=1`；`hasText=false` 支持 `seqShape=2/3` |
| S | `hasText=true` 时 `S=115200+textLength`；`hasText=false` 时 `S=img_seq_len` |

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

1. dtype 分发：根据输入 dtype 选择 `half` 或 `bfloat16_t` 模板实例，score、softmax、累加器使用 `float`。
2. 视频 query 分支：根据当前 query tile 的 3D 坐标和当前 head 的 `(Kt, Kh, Kw)`，计算局部 KV tile 范围；边界 tile 使用 clamp 逻辑保证窗口大小稳定并避免越界。
3. 文本 KV 分支：`has_text=true` 时，视频 query 在完成局部视频 KV attention 后，继续遍历文本 KV block，并使用 mask 处理 `text_length` 和 padding。
4. 文本 query 分支：`has_text=true` 时，文本 query 遍历全部视频 KV tile 和全部文本 KV block，完成全局 attention。
5. FlashAttention 累加：每处理一个 KV block，执行 `QK^T`、scale、mask、online softmax 更新和 `PV` 累加，避免保存完整 score 矩阵。
6. 输出写回：计算 `acc / l` 后转换为输入 dtype 写回 output；padding 区域不写回或写回后在 host 侧输出 shape 中裁剪。

Padding 处理策略：

1. kernel 不访问原始输入 `S` 以外的地址。对于 `aligned_s > S` 的 padding query，仅作为文本 query block 的无效行参与任务排布，读 Q 时通过 `q_idx < S` mask 置零，写 output 时通过同一 mask 跳过。
2. 文本 KV 只访问 `[img_seq_len, img_seq_len + textLength)` 的有效范围，padding KV 不参与 softmax。该策略与源实现的补齐后裁剪在有效输出范围内保持等价，且避免 kernel 越界访问。
3. 当 `textLength=0` 时，文本 KV block 的 mask 全 false；视频 query 分支仅累加视频 KV，文本 query 分支不产生有效输出。


FlashAttention 流程图如下：
![asc.png](https://raw.gitcode.com/user-images/assets/9516645/466ae81b-01d3-4df9-a7fb-2f98d4c3c5ad/asc.png 'asc.png')


## 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |

## 算子约束限制

1. 当前仅支持 BNSD 格式。
2. 当前仅支持 `float16` 和 `bfloat16`。
3. 当前仅支持 `seq_shape` 为 `30x48x80`、`36x48x48`、`18x48x80`。
4. `has_text=true` 仅支持 `30x48x80` 场景，`text_length <= 256`，视频 token 长度固定为 115200。
5. `has_text=false` 时，`36x48x48` 场景要求 `S=82944`，`18x48x80` 场景要求 `S=69120`。
6. tile 尺寸固定为 `(6, 8, 8)`，即单 tile 384 token。
7. `window_size` 长度必须等于 `3 * head_num`，且每个窗口为奇数尺寸 `(Kt, Kh, Kw)`。
8. 基础性能验收场景中 `B=1`、`N=8`、`D=64`；泛化实现需覆盖 `B>=1`、`N>=1`，并对非 `D=64` 场景明确 fallback block 配置和测试结论。
9. `has_text=true` 时 `S=115200+text_length`，且 kernel 内部 padding 仅用于任务对齐，不访问输入 `S` 以外地址。
10. 当前对齐源实现的场景限制：`has_text=false` 不支持 `seq_shape=30x48x80`；如验收要求增加该场景，需要同步更新 golden、约束和测试用例。

# 可维护可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足 AscendOpTest 默认精度阈值 | AscendOpTest |
| 性能标准 | 2 个验收性能场景平均算子用时分别不高于 655.97 us、5771 us | 0.8 倍 GPU A100 性能 |



## 兼容性分析

本算子为新增 experimental 算子，不涉及历史版本兼容问题。接口设计需与 ops-transformer experimental 算子目录下现有 aclnn 调用方式保持一致，便于后续 README、样例和自验证报告复现。

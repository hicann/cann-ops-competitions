# SlidingTileAttention 算子设计文档

## 一、需求背景（required）

### 1.1 需求来源

本任务来自 2026 年 8 月社区任务 `SlidingTileAttention`，要求参考 FastVideo 中的 `sliding_tile_attention` 实现，在 Atlas A2 训练系列产品上使用 Ascend C 完成功能一致的自定义算子开发、测试与交付。

FastVideo 原始接口如下：

```python
sliding_tile_attention(q, k, v, window_size, text_length, has_text=True, seq_shape="30x48x80")
```

验收目标是对齐其前向语义，支持任务书规定的 `float16`、`bfloat16` 和 BNSD 数据格式。

### 1.2 背景介绍

#### 1.2.1 SlidingTileAttention 算子实现优化

SlidingTileAttention 是面向长视频序列的局部窗口注意力算子。视频 token 按 3D 网格组织，query 只关注自身附近的局部视频 KV tile；当输入包含文本 token 时，视频 query 还需要关注全部文本 KV。

与全量 self-attention 相比，该算子可以把注意力计算限制在局部窗口内，降低长序列场景下的计算量和访存量。

#### 1.2.2 FastVideo 实现现状分析

FastVideo 的 Python 层负责参数组织和序列布局，底层 kernel 完成滑窗 attention 计算。任务书要求在昇腾 NPU 上复现相同的输入输出语义，验收口径以 Python 层 `sliding_tile_attention()` 为准。

输入、属性和输出如下：

| 参数 | 含义 | 类型 | 支持数据类型 | 约束 | 形状/取值 |
| --- | --- | --- | --- | --- | --- |
| q | Query 输入 | tensor | float16, bfloat16 | 与 k/v 的 B/N/S/D 一致 | `(B, N, S, D)` |
| k | Key 输入 | tensor | float16, bfloat16 | 与 q/v 一致 | `(B, N, S, D)` |
| v | Value 输入 | tensor | float16, bfloat16 | 与 q/k 一致 | `(B, N, S, D)` |
| window_size | 每个 head 的 3D 窗口 | list | int | 与 head 数匹配 | `[(Kt, Kh, Kw), ...]` |
| text_length | 文本 token 长度 | int | int | `has_text=true` 时参与计算 | 非负整数 |
| has_text | 是否包含文本 token | bool | bool | 控制文本分支 | `true/false` |
| seq_shape | 视频网格形状 | str | string | 任务书给定 3 类配置 | `"30x48x80"`、`"36x48x48"`、`"18x48x80"` |
| output | 输出 | tensor | float16, bfloat16 | dtype 与输入一致 | `(B, N, S, D)` |

源实现的关键流程如下：

1. 解析 `seq_shape` 得到视频 canvas 的三维尺寸。
2. 将视频 token 按固定 tile 组织，每个 tile 包含 `6 x 8 x 8 = 384` 个 token。
3. `has_text=true` 时，视频 token 之后拼接文本 token，并在内部按 384 对齐。
4. 视频 query 按 head 独立处理，每个 head 使用自己的 3D 窗口。
5. attention 采用 online softmax 累加方式，不保存完整 score 矩阵。

### 1.3 算子功能分析

SlidingTileAttention 可以看作“视频局部滑窗 + 文本全局可见”的 FlashAttention 变体。

- 视频 token：只与当前 head 对应窗口内的邻近视频 KV tile 进行 attention。
- 文本 token：当 `has_text=true` 时，视频 query 还需要关注全部文本 KV。
- 多 head：`window_size` 按 head 生效，不同 head 可以使用不同窗口。

## 二、需求分析（required）

### 2.1 需求描述

使用 Ascend C 实现 SlidingTileAttention 算子，功能与 FastVideo 源实现对齐，支持 `float16` 和 `bfloat16`，支持 BNSD 数据格式，并满足任务书中的精度、性能和自验要求。

### 2.2 需求拆解

1. 支持 `q/k/v` 为 `float16`、`bfloat16`，且三者 shape 和 dtype 必须一致。
2. 支持 `window_size`、`text_length`、`has_text`、`seq_shape` 四个属性。
3. 支持 `seq_shape=30x48x80`、`36x48x48`、`18x48x80`。
4. 支持 `has_text=true/false` 两条路径。
5. 支持内部按 384 对齐的序列组织方式。
6. 内部累加、softmax 最大值和归一化因子使用 `float32`。
7. 输出 shape 与输入 `q` 完全一致。
8. 精度满足任务书要求，性能满足 A100 的 0.8 倍基准。

### 2.3 接口定义

按任务书提供的 aclnn 形式实现：

```cpp
aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(
    const aclTensor          *q,
    const aclTensor          *k,
    const aclTensor          *v,
    aclTensor                *output,
    const aclIntArray *const *windowSize,
    uint64_t                  windowSizeLen,
    int64_t                   textLength,
    bool                      hasText,
    const char               *seqShape,
    uint64_t                 *workspaceSize,
    aclOpExecutor           **executor);

aclnnStatus aclnnSlidingTileAttention(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## 三、详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

对每个 query 行，注意力输出为：

$$
O = \mathrm{Softmax}\left(\frac{QK^T}{\sqrt{D}}\right)V
$$

其中 softmax 只在窗口掩码允许的 KV 范围内计算。`has_text=true` 时，视频 query 还要把全部文本 KV 纳入可见集合。

#### 3.1.2 支持数据类型

| 输入 dtype | 中间计算 dtype | 输出 dtype |
| --- | --- | --- |
| float16 | float32 | float16 |
| bfloat16 | float32 | bfloat16 |

#### 3.1.3 支持形状

| seq_shape | canvas_t | canvas_h | canvas_w | img_seq_len |
| --- | ---: | ---: | ---: | ---: |
| `30x48x80` | 30 | 48 | 80 | 115200 |
| `36x48x48` | 36 | 48 | 48 | 82944 |
| `18x48x80` | 18 | 48 | 80 | 69120 |

固定 tile 尺寸为 `(6, 8, 8)`，每个 tile 包含 384 个 token。

### 3.2 算子实现

#### 3.2.1 Host 侧设计

Host 侧负责参数校验、场景解析、窗口归一化、tiling 构造和 kernel 发射。

1. 参数校验：
   - `q/k/v` 必须为 4 维 BNSD Tensor。
   - `q/k/v` 的 shape、dtype、device 必须一致。
   - `q/k/v` 仅支持 `float16` 和 `bfloat16`。
   - `window_size` 的维度和内容必须合法。
   - `text_length`、`has_text`、`seq_shape` 的组合必须符合任务书。
2. 形状推导：
   - 输出 shape 与输入 `q` 保持一致。
   - 根据 `seq_shape` 解析 canvas 的三维尺寸和 `img_seq_len`。
3. 窗口整理：
   - 将 `window_size` 统一整理为每个 head 的 `(Kt, Kh, Kw)`。
   - 若验收输入给出单个窗口三元组，则在 host 侧广播到所有 head。
4. 序列组织：
   - `has_text=true` 时，内部按 384 对齐处理序列尾部。
   - padding 仅用于任务组织，不参与有效输出写回。
5. 分核策略：
   - 以 `(B, N, query_block)` 为基本任务粒度。
   - 优先保证 AI Core 负载均衡。
6. tiling key 规划：
   - 至少区分 `has_text=false` 与 `has_text=true` 两类主路径。
   - 可按 dtype、head_dim 和 block 配置继续细分。

Host 侧校验表如下：

| 校验项 | 合法条件 |
| --- | --- |
| dtype | `q/k/v` 统一，且为 `float16` 或 `bfloat16` |
| format | 仅支持 BNSD，rank 为 4 |
| shape | `B/N/S/D` 一致，`B >= 1`，`N >= 1`，`D > 0` |
| window_size | 长度与 head 数匹配，每个窗口为正整数三元组 |
| text_length | 非负 |
| has_text | bool 类型 |
| seq_shape | 仅支持任务书给定三类字符串 |

#### 3.2.2 Kernel 侧设计

Kernel 侧采用在线 softmax 的块式 attention 结构，核心流程如下：

1. 读取当前 query 所属 tile 的三维坐标。
2. 根据当前 head 的 `(Kt, Kh, Kw)` 计算局部 KV tile 范围。
3. 遍历窗口内的视频 KV block。
4. `has_text=true` 时，追加遍历文本 KV block。
5. 每处理一个 KV block，执行 `QK^T`、scale、mask、online softmax 更新和 `PV` 累加。
6. 将结果转换回输入 dtype 写回 output。

Kernel 中使用 `float32` 维护 softmax 的 running max、sum 和累加器，保证数值稳定性。

#### 3.2.3 数据流

| 数据 | 说明 |
| --- | --- |
| Q/K/V | BNSD 输入，按 head 和 tile 组织 |
| window_size | 每个 head 的局部窗口配置 |
| output | 与输入 shape 一致的结果 Tensor |

### 3.3 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |

### 3.4 算子约束限制

1. 仅支持 `float16` 和 `bfloat16`。
2. 仅支持 BNSD 格式。
3. 仅支持任务书指定的 3 类 `seq_shape`。
4. `window_size` 必须合法，且与 head 数一致。
5. `has_text=true` 时需按任务书处理文本 token 和内部对齐。
6. 仅实现前向。

## 四、可维可测分析

### 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出与 FastVideo 基线一致，满足任务书精度要求 | 任务书 |
| 性能标准 | 2 个性能场景达到 A100 基准的 0.8 倍 | 任务书 |

按任务书给出的 A100 参考，性能阈值可折算为：

| 场景 | A100 平均耗时 | 0.8 倍阈值 |
| --- | --- | --- |
| `30x48x80` + text | 524.79 us | 655.99 us |
| `36x48x48` | 4618 us | 5772.5 us |

### 4.2 测试设计

精度测试建议覆盖以下场景：

| 编号 | 场景 | 说明 |
| --- | --- | --- |
| TC-01 | `seq_shape=30x48x80`，`has_text=true` | 覆盖文本路径和 384 对齐 |
| TC-02 | `seq_shape=36x48x48`，`has_text=false` | 覆盖无文本路径 |
| TC-03 | `seq_shape=18x48x80`，`has_text=false` | 覆盖另一类无文本路径 |
| TC-04 | `float16` / `bfloat16` | 覆盖两种 dtype |
| TC-05 | 不同 `window_size` | 覆盖小窗、中窗和大窗 |
| TC-06 | 边界与非法输入 | 覆盖 shape、dtype、属性校验 |

性能测试覆盖任务书给定的两组基准场景：

| B | N | D | Img_seq | Text_len | Window_size |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 64 | `30x48x80` | 128 | 8 head 各异 |
| 1 | 8 | 64 | `36x48x48` | None | `(3,3,3) * 8` |

### 4.3 兼容性分析

该算子为新增接口，不影响现有算子行为。输入不满足 dtype、shape、device 或属性约束时，host 侧直接报错，避免产生错误结果。

## 参考

- FastVideo `sliding_tile_attention`：<https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/ops.py>
- FastVideo Triton kernel：<https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/st_attn_triton.py>

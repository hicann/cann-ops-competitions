# SlidingTileAttention 算子设计文档

# 需求背景（required）

## 需求来源

本需求来源于 2026 年 8 月 CANN 社区任务 `04-1-SlidingTileAttention`。任务要求参考 FastVideo 固定提交 `4ddcdf541f32b63b5c684016c903658e2e2b6f67` 中的 `sliding_tile_attention`，在 Atlas A2 系列产品上使用 Ascend C 实现功能一致的 aclnn 算子。

参考实现：

- Python 入口：<https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/ops.py#L14>
- Triton 实现：<https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/st_attn_triton.py>

## 背景介绍

SlidingTileAttention 面向视频生成 DiT 中的超长视频 token 序列，利用视频 token 在时间和空间上的局部性，将图像 query 的可见范围限定为三维 tile 窗口，同时保留 text token 的全局注意力。与构造完整 `S×S` mask 的实现相比，内核只读取窗口内的 KV tile，以降低计算量和访存量。

FastVideo 实际代码具有以下关键语义：

1. Q/K/V 为 BNSD 布局，序列中 image token 在前，text token 在后。
2. 视频 token 按固定 `6×8×8=384` token 的 tile-major 形式组织。
3. `window_size=(wt,wh,ww)` 的单位是 tile；同一 tile 内的 384 个 token 使用相同的 KV tile 集合。
4. 边界处会将窗口中心向 canvas 内部移动，尽可能保持完整窗口，而不是以 query tile 为中心直接截断。
5. image query 关注局部 image KV 和所有有效 text KV；text query 关注所有有效 image/text KV。
6. 使用 FP32 online softmax 维护行最大值、指数和与输出累加器。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 SlidingTileAttention 算子，对外提供 aclnn 两段式接口，支持 FLOAT16/BFLOAT16、BNSD 布局、单窗口广播和逐 head 窗口，支持 image 局部注意力与 text 全局注意力，并满足确定性、精度和性能验收要求。

## 需求拆解

1. 输入输出格式为 `[B,N,S,D]`，q/k/v/output 的 shape 和 dtype 一致。
2. 支持 FLOAT16 和 BFLOAT16，QK、softmax 状态和 PV 累加使用 FP32。
3. `window_size` 长度为 1 时广播至全部 head，长度为 N 时逐 head 生效。
4. `has_text=false` 时将有效 `text_length` 视为 0。
5. `has_text=true` 时实现 image-to-text 全局可见和 text-to-all 全局注意力。
6. 相同输入、相同属性和相同执行环境的多次输出一致。
7. 针对验收主场景 `B=1,N=8,D=64` 设计高性能 tiling，并保留非主场景的泛化路径。

# 详细设计（required）

## 算子分析

### 数学公式

对 batch `b`、head `n` 和 query token `i`，记其有效 KV token 集合为 `K(n,i)`：

$$
s_{i,j}=\frac{Q_iK_j^T}{\sqrt D},\qquad j\in\mathcal K(n,i)
$$

$$
O_i=\frac{\sum_{j\in\mathcal K(n,i)}e^{s_{i,j}-m_i}V_j}
{\sum_{j\in\mathcal K(n,i)}e^{s_{i,j}-m_i}},\qquad
m_i=\max_{j\in\mathcal K(n,i)}s_{i,j}
$$

对 image query，`K(n,i)` 为当前 head 的三维窗口中的 image tile token 与全部有效 text token。对 text query，`K(n,i)` 为全部有效 image/text token。

### 窗口边界语义

设 tile grid 一个维度的大小为 `G`，query tile 坐标为 `q`，窗口大小为正奇数 `w`：

```text
half   = w / 2
center = clamp(q, half, G - 1 - half)
start  = max(0, center - half)
end    = min(G, center + half + 1)
```

三个维度分别计算后取笛卡尔积，即得到当前 query tile 的 image KV tile 集合。

### 支持数据类型

| 输入类型 | Q/K/V 片上类型 | score/softmax/累加器 | 输出类型 |
| --- | --- | --- | --- |
| FLOAT16 | half | float | half |
| BFLOAT16 | bfloat16_t | float | bfloat16_t |

### 支持形状

- q/k/v/output：4 维 BNSD。
- `B,N,S,D > 0`。
- `seq_shape` 解析为正数 `(T,H,W)`。
- FastVideo-exact 路径要求 `T,H,W` 分别可被 `6,8,8` 整除，主场景为 `30x48x80`、`36x48x48`和 `18x48x80`。
- `S >= T×H×W + effective_text_length`。
- 验收主路径为 `D=64`；其他 D 由通用 tiling 覆盖，具体上限由片上空间和 Matmul 模板确定。

## 算子实现

### Host 侧设计

Host 侧完成参数校验、shape 解析、窗口广播、tiling 选择、workspace 计算和任务分配。

#### 参数校验

| 校验项 | 合法条件 |
| --- | --- |
| rank/format | q/k/v/output 均为 4 维 BNSD |
| shape | q/k/v/output 的 B/N/S/D 完全一致且每一维为正 |
| dtype | 均为 FLOAT16 或均为 BFLOAT16 |
| seq_shape | 符合 `TxHxW` 格式，三个数为正 |
| sequence | `S >= T×H×W + effective_text_length` |
| text | `has_text=true` 时 `text_length>0`；否则按 0 处理 |
| window | 数量为 1 或 N，每项包含 3 个正奇数 |

#### Tiling 数据

TilingData 至少包含：

```text
B, N, S, D
canvasT, canvasH, canvasW
tileGridT, tileGridH, tileGridW
imgSeqLen, textLength, validSeqLen
blockQ, blockKv, blockDim
hasText, dtypeKey, shapeMode
windowT[N], windowH[N], windowW[N]
imageTaskCount, textTaskCount
```

若 N 超过 tiling 结构的内联上限，窗口数组放入 workspace；主场景 `N=8` 直接放入 tiling data。

#### 分核策略

以 `(batch, head, queryTile, queryBlockInTile)` 为 image 分支的基本任务，以 `(batch, head, textQueryBlock)` 为 text 分支的基本任务。当不同 head 的窗口差异较大时，Host 侧按每个任务的 KV block 数进行加权分核，减少大窗口 head 造成的长尾。

#### TilingKey 规划

TilingKey 至少区分：

- FLOAT16 / BFLOAT16；
- `has_text=false/true`；
- `D=64` 主优化路径/通用 D 路径；
- 小窗口/中窗口/接近全局窗口。

### Kernel 侧设计

#### 整体流程

1. 读取当前任务的 q block；
2. 计算 query tile 坐标及三维 KV tile 范围；
3. 使用 Cube Matmul 计算 `QK^T`；
4. Vector 侧执行 scale、tail mask 和 FP32 online softmax；
5. 使用 Cube Matmul 计算 `P×V`，更新 FP32 output accumulator；
6. 继续处理下一个 image KV tile，以及可选 text KV block；
7. 执行最终归一化并按输入 dtype 写回。

#### Online softmax

对每个 KV block 的 score `S_c`，以 FP32 维护 `m,l,acc`：

```text
m_new   = max(m, row_max(S_c))
alpha   = exp(m - m_new)
p       = exp(S_c - m_new)
l_new   = alpha * l + row_sum(p)
acc_new = alpha * acc + p @ V_c
```

完成全部 KV block 后输出 `acc/l`。实现优先复用 `SoftmaxFlashV2`、`AdjustSoftMaxRes` 及现有 attention 算子中的 Matmul 配置，避免建立完整 score 矩阵。

#### 数据流和缓冲

- Q block 在处理其全部 KV 窗口期间保留在片上。
- K/V tile 以 block 形式双缓冲 CopyIn。
- Cube 的 QK/PV 与 Vector softmax 尽可能通过 event 同步流水重叠。
- tile token 数 384 是 128 的整数倍，主路径使用 `blockQ/blockKv=128`，text tail 用 mask 处理。

#### Padding 处理

FastVideo Python 层将有 text 的序列补到 384 的整数倍后再截断输出。Ascend C 内核不必物化 padding tensor，仅按对齐后的 query/KV block 调度，通过 `qIdx<validSeqLen` 和 `kvIdx<validSeqLen` mask 防止越界。padding KV 不参与有效 query 的 softmax，有效输出与参考实现截断后保持一致。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Ascend 910B | √ |

## 算子约束限制

1. 仅支持 BNSD 格式的 FLOAT16/BFLOAT16。
2. q/k/v/output 不支持 broadcast，shape 和 dtype 必须一致。
3. FastVideo-exact 路径的 tile 大小固定为 `(6,8,8)`。
4. `window_size` 必须为正奇数三元组，数量为 1 或 N。
5. 非 `D=64` 路径以功能和精度为主，性能优化以验收的 `D=64` 为主。

## 工程落地设计

已调研 ops-transformer `experimental/attention` 中已合入的 `blitz_sparse_attention`、`turbo_quant_sparse_flash_attention`、`minimax_sparse_attention_split_kv` 及 `quant_block_sparse_attn`。SlidingTileAttention 按现有仓库约定采用以下真实目录结构：

```text
experimental/attention/sliding_tile_attention/
├── CMakeLists.txt
├── README.md
├── docs/
│   └── aclnnSlidingTileAttention.md
├── examples/
│   └── test_aclnn_sliding_tile_attention.cpp
├── op_host/
│   ├── CMakeLists.txt
│   ├── sliding_tile_attention_def.cpp
│   ├── sliding_tile_attention_infershape.cpp
│   ├── sliding_tile_attention_tiling.cpp
│   ├── sliding_tile_attention_tiling.h
│   └── op_api/
│       ├── aclnn_sliding_tile_attention.cpp
│       ├── aclnn_sliding_tile_attention.h
│       ├── sliding_tile_attention.cpp
│       └── sliding_tile_attention.h
├── op_kernel/
│   ├── sliding_tile_attention.cpp
│   ├── sliding_tile_attention_common.h
│   ├── sliding_tile_attention_base.h
│   ├── sliding_tile_attention_service_cube.h
│   ├── sliding_tile_attention_service_vector.h
│   └── sliding_tile_attention_template_tiling_key.h
└── tests/
    ├── CMakeLists.txt
    ├── ut/
    │   ├── CMakeLists.txt
    │   ├── op_host/
    │   └── op_api/
    └── atk/
        ├── README.md
        ├── function_sliding_tile_attention.py
        └── sliding_tile_attention.csv
```

`experimental/attention/CMakeLists.txt` 使用 `file(GLOB CURRENT_DIRS ...)` 扫描算子目录，仅需 SlidingTileAttention 目录自身提供 `CMakeLists.txt`。根目录使用 `add_op_to_compiled_list()` 注册算子，`op_host/CMakeLists.txt` 使用 `add_modules_sources()` 与 `target_sources()` 纳入 def/infer-shape/tiling/aclnn 源文件。

构建命令对齐已合入的 `blitz_sparse_attention` README：

```bash
bash build.sh --make_clean --experimental -j16 --pkg \
  --soc=ascend910b \
  --ops=sliding_tile_attention
```

example 复现命令规划为：

```bash
bash build.sh --experimental \
  --run_example sliding_tile_attention eager cust \
  --soc=ascend910b \
  --vendor_name=custom
```

ops-transformer 当前 CI 对 experimental 目录包含 x86_64/aarch64、Ubuntu 20.04/24.04 的 Ascend 910B 构建任务，并执行 UT 和静态检查。代码 PR 将在本地/910B 自验后评论 `/compile` 触发仓库 CI。

## 开发与交付准备矩阵

| 交付项 | 已确定结构 | 验收前输出 |
| --- | --- | --- |
| Host | def、infer-shape、tiling、op_api | Host UT 与参数校验结果 |
| Kernel | Cube/Vector service + online softmax | FP16/BF16 Kernel 及 TilingKey |
| API | aclnn GetWorkspaceSize + Execute | C++ example 及 API 文档 |
| 精度 | FastVideo golden + ATK | matched ratio、max abs error、ULP、原始报告 |
| 确定性 | 固定输入重复执行 | 至少 10 次输出一致性报告 |
| 性能 | A100 FastVideo 与 910B3 aclnn 统一 forward 口径 | avg/P50/P99、profiler 截图、CSV/JSON |
| 文档 | README + aclnn API + ATK README | 可独立复现的编译/安装/执行步骤 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| FLOAT16 精度 | `rtol=atol=2^-9`，matched ratio ≥ 0.99，max abs error ≤ `1e-1 or 32 ULP` | 社区任务书 |
| BFLOAT16 精度 | `rtol=atol=2^-6`，matched ratio ≥ 0.99，max abs error ≤ `1e0 or 32 ULP` | 社区任务书 |
| 确定性 | 相同输入多次运行的输出一致 | 社区任务书 |
| 性能 | 910B3 上达到 A100 FastVideo 基线的 0.8 倍 | 社区任务书 |

### 自测计划

1. 小 shape 逻辑用例：单 tile、边角/边/中心 tile、单 head、多 head 不同窗口。
2. 功能用例：`has_text=true/false`、窗口广播/逐 head、text tail、FP16/BF16。
3. 泛化用例：任务书规定范围内的 `window_size`、`text_length`和 `seq_shape`。
4. 异常用例：rank/dtype/shape 不匹配、非法字符串、非法窗口、S 不足。
5. 性能用例：完整执行任务书的两个用例，报告 warm-up、执行次数、统计方法、平均值、P50 和 P99。

## 兼容性分析

本算子为新增 experimental 算子，不修改现有算子接口或语义，不存在历史版本兼容问题。提交目录为 `experimental/attention/sliding_tile_attention`。

## 待评审确认项

FastVideo 实际源码与任务书的文字描述存在差异，为保证验收口径一致，设计评审时需确认：

1. text token 的序列位置是 FastVideo 源码的 image-first/text-last，还是任务书公式中的 text-first。
2. `window_size` 的单位是 FastVideo 的 tile 还是任务书表面描述的 token 坐标。
3. 泛化 `seq_shape` 是否必须满足 `(6,8,8)` 整除，以及非整除时的 tile 和 tail 定义。
4. 性能基线的 `Avg time` 是完整 STA forward 还是单个内核 launch 的 profiler 统计。

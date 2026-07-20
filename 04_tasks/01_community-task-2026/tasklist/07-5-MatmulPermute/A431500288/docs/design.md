# 需求背景（required）

## 需求来源

7 月社区任务 `MatmulPermute` 算子开发任务书，要求在 CATLASS 仓中实现 Ascend 950 上的
`Matmul + Tensor4DPermute0213` 融合算子，并补充 optest 测试交付件、README、
自验证报告和设计文档。

## 背景介绍

### MatmulPermute算子实现优化

任务中的基线方案为 `torch.mm + reshape + permute + contiguous` 小算子拼接。该方案会先将
矩阵乘结果 `C(M,N)` 写回 GM，再由后续 permute kernel 重新读取并重排输出，存在额外
kernel launch 和 GM 读写开销。

本方案基于 CATLASS matmul 路径实现融合算子，在 matmul epilogue 阶段直接按照
`Tensor4DPermute0213` 的目标布局写回输出，减少中间矩阵落盘和后处理 kernel 开销。

### MatmulPermute算子实现现状分析

当前任务面向 FP16 RowMajor 输入，固定验收 `S1=8`、`S2=4`。算子输入输出如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| A | 左矩阵 | tensor | float16 | RowMajor | `(M,K)` |
| B | 右矩阵 | tensor | float16 | RowMajor | `(K,N)` |
| D | 输出矩阵 | tensor | float16 | `M % S1 == 0`, `N % S2 == 0` | `(M/S1,S2,S1,N/S2)` |

### MatmulPermute算子功能分析

MatmulPermute 先计算标准矩阵乘：

```text
C(M,N) = A(M,K) x B(K,N)
```

随后将 `C` 视为：

```text
reshape(M / S1, S1, S2, N / S2)
```

再执行：

```text
permute(0, 2, 1, 3)
```

最终输出形状为：

```text
(M / S1, S2, S1, N / S2)
```

# 需求分析（required）

## 需求描述

使用 CATLASS/Ascend C 实现 Ascend 950 上的 FP16 RowMajor
`Matmul + Tensor4DPermute0213` 融合算子。算子需在单次 kernel launch 内完成矩阵乘和
输出重排，输出与 PyTorch 参考实现保持一致，并在 Ascend 950PR 真机上达到
`torch.mm + permute` baseline 的 1.1 倍性能目标。

## 需求拆解

1. 支持输入 `A(M,K)`、`B(K,N)`，数据类型为 FP16，布局为 RowMajor。
1. 支持输出 `D(M/S1,S2,S1,N/S2)`，数据类型为 FP16。
1. `S1`、`S2` 为编译期常数；任务验收固定 `S1=8`、`S2=4`。
1. 约束 `M % S1 == 0`、`N % S2 == 0`。
1. 单次 kernel launch 完成 Matmul 与 Permute 写回。
1. 基于 CATLASS optest 补充 JIT kernel、Python wrapper 和 pytest。
1. 正式性能使用 Ascend 950PR `msprof op` 采集，并与任务测试集 baseline 对比。

# 详细设计（required）

## 算子分析

### 数学公式

标准矩阵乘：

```text
C[m, n] = sum(A[m, k] * B[k, n]), k in [0, K)
```

融合写回时，逻辑位置 `C[m,n]` 映射到 4D 输出：

```text
d0 = m / S1
d1 = n / (N / S2)
d2 = m % S1
d3 = n % (N / S2)
D[d0, d1, d2, d3] = C[m, n]
```

展平 GM 写回偏移为：

```text
dst = (((d0 * S2 + d1) * S1 + d2) * (N / S2) + d3)
```

### 支持数据类型

| 参数 | 数据类型 | 格式 | 说明 |
| --- | --- | --- | --- |
| A | FP16 | RowMajor ND | 左矩阵，shape `(M,K)` |
| B | FP16 | RowMajor ND | 右矩阵，shape `(K,N)` |
| D | FP16 | ND | 4D permute 后输出，shape `(M/S1,S2,S1,N/S2)` |

### 支持形状

| 维度 | 约束 |
| --- | --- |
| M | `M > 0` 且 `M % S1 == 0` |
| N | `N > 0` 且 `N % S2 == 0` |
| K | `K > 0` |
| S1 | 编译期常数，验收为 `8` |
| S2 | 编译期常数，验收为 `4` |

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

host 侧主要完成参数检查、输入输出 buffer 申请、CATLASS kernel 参数组装和 kernel launch。
实现会检查 `M % S1 == 0`、`N % S2 == 0`，并通过 `CATLASS_JIT_S1`、
`CATLASS_JIT_S2` 宏注入编译期 `S1/S2`，当前默认值为 `8/4`。

host 侧执行流程如下：

1. 解析输入矩阵形状，得到 `M/N/K`。
1. 检查输入 dtype、RowMajor 布局、`M % S1 == 0`、`N % S2 == 0` 等约束。
1. 根据 `M/N/K`、`N/S2`、尾块情况和计算规模选择样例中预定义的 kernel 配置。
1. 由 host 侧分发到对应 `KernelSelector`，不同 selector 绑定不同 tile、调度和 epilogue 写回路径。
1. 组装 CATLASS matmul 参数并发起 kernel launch。

当前实现中使用 `MatmulPermuteTilingData` 作为样例内部的配置描述结构。它不是通用算子
框架中的 tiling 路由机制，而是 host 侧选择 `KernelSelector` 时使用的元数据。主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `kernelFamily` | kernel 类别，用于区分基础融合、向量化写回、流水增强、尾块增强等路径 |
| `tilePreset` | L1/L0 的 `M/N/K` tile 组合 |
| `l1M/l1N/l1K` | L1 tile 的 M/N/K 维度 |
| `l0M/l0N/l0K` | L0 tile 的 M/N/K 维度 |
| `vectorPackMode` | epilogue 向量化/pack 写回模式 |
| `windowLen` | ASWT 调度窗口长度 |
| `splitKFactor` | split-K 场景下的 K 维切分因子 |
| `mTailSplit/nTailSplit` | 尾块增强场景下的 M/N 尾块切分方式 |
| `forceFullAic` | 是否强制使用满 AIC 调度 |

CATLASS 样例本身已经是面向固定问题形态的样例代码，不依赖通用算子框架中的 tiling
机制做运行时路由。本方案在样例 host 文件中根据 shape 特征选择不同 `KernelSelector`，
再实例化对应的编译期 specialization。这样既能避免把官方 CSV 中的逐 shape 配置写死为
逐项分支，也能让同一类 shape 复用同一个 kernel 配置。

kernel 选择策略：

本算子以矩阵乘 `M/N/K` 为主维度进行 tiling，输出重排所需的 `S1/S2` 固定为编译期常数。
host/JIT 侧首先计算 `nOuter = N / S2`，并检查 `M % S1 == 0`、`N % S2 == 0`。
合法 shape 进入 host 侧 kernel 选择流程。选择逻辑主要考虑：

1. `M/N/K` 规模：决定 L1/L0 tile 形状和 K 维切分粒度。
1. `N/S2` 大小：决定输出重排后连续列段长度。
1. M/N 尾块：决定是否需要尾块增强调度，避免尾块 task 过少或写回碎片化。
1. tile 网格数量：决定是否限制启动核数，避免小 shape 过度调度。
1. 输出连续性：决定 epilogue 走直接连续写回、UB pack 后写回或通用分段写回。
1. 小 M/N、大 K 场景：当 `M/N` 较小导致 MN tile 数不足、但 `K` 较大时，当前实现提供
   split-K / 多核切 K 路径，提高 AIC 并行度；该路径需要额外处理 K 维部分和归约写回。

当前实现中的 kernel family 规划：

| kernel 类别 | 作用 |
| --- | --- |
| `Basic` | 标准 CATLASS matmul + 自定义 epilogue 写回 |
| `VectorPack` | epilogue 中启用连续段或 UB pack 写回，降低 GM 小段 scatter 开销 |
| `Window` / `CustomWindow` | 使用 ASWT window 调度，增强 MMAD 与搬运流水重叠 |
| `FixedTailSplit` | 对尾块进行固定拆分，缓解尾块 task 过少的问题 |
| `TailPartialMSplit` | 对 M 方向尾部 task 做局部拆分，并可强制满 AIC 调度 |
| `MutexVectorPack` | 使用 mutex pingpong 调度配合向量化写回 |
| `CustomVectorPack` | 使用自定义 dispatch policy 的向量化写回 |
| `SplitK` | 面向小 M/N、大 K 导致 MN 并行度不足的场景，沿 K 维切分提升多核利用率 |

host 分发规划：

host 分发逻辑用于选择 kernel 类型和对应参数。当前主要表达以下信息：

1. kernel 类型：选择 `Basic`、`VectorPack`、`Window`、`FixedTailSplit`、`TailPartialMSplit`、
   `MutexVectorPack`、`CustomVectorPack` 或 `SplitK`。
1. matmul tile 参数：选择 L1/L0 的 `M/N/K` 分块，决定单个 AIC task 的计算范围。
1. 调度参数：根据 tile 网格数量和硬件 AIC 数量确定实际使用 core 数，避免小 shape 过度起核。
1. epilogue 参数：根据 `N/S2`、N 尾块和目标写回连续性选择写回路径。
1. 尾块参数：当 `M` 或 `N` 不能被当前 tile 整除时，kernel 侧按实际 tile 有效范围处理。
1. K 维切分参数：小 M/N、大 K 场景下可走 `SplitK`，增加 K 维分片数和归约策略。

不同 host 分发分支绑定不同 `KernelSelector`。当前设计不在文档中枚举逐 shape 配置，而是按
shape 特征选择 kernel 类别与参数：小 shape 侧重降低调度开销；大 K shape 侧重提高
MMAD 利用率；低 K 大 M shape 侧重减少无效搬运；小 N 或尾块 shape 侧重优化 epilogue 写回。
同一个 selector 可覆盖多个 shape，新增 shape 时优先复用已有 selector，只有当计算规模、
尾块特征、K 维并行度或写回连续性发生明显变化时才新增 selector。

对于小 M/N、大 K 场景，当前实现已包含 `SplitK` kernel family。该路径先由 AIC 将不同
K 分片的部分和写入 workspace，再由 AIV 对 split-K workspace 做归约，同时按
`Tensor4DPermute0213` 目标布局写回输出。该方案可在 MN tile 数不足时增加 AIC 并行度，
代价是需要额外 workspace 和一次归约搬运，因此仅适合 K 维足够大、MN 并行度不足的场景。

性能优化策略：

1. 融合 matmul 与 permute，减少中间结果 GM 读写和额外 kernel launch。
1. 在 epilogue 阶段尽量按连续片段写回，降低重排带来的小粒度 scatter 开销。
1. 通过 host 侧 kernel 选择为不同 shape 使用合适 kernel 形态，使不同规模的 shape
   使用更匹配的计算和搬运策略。

#### 3.2.2 kernel侧设计：

kernel 基于 CATLASS `KernelMatmulMixFixpipeOpti` 路径实现，整体流程如下：

1. AIC 侧执行矩阵乘，结果经 fixpipe 写入 UB。
1. AIV 侧在 epilogue 中读取 UB tile。
1. epilogue 根据 `Tensor4DPermute0213` 的地址映射直接写回 GM，避免先输出中间
   `C(M,N)` 再启动单独 permute kernel。

kernel 侧按 host 选择的 `KernelSelector` 确定 tile shape，处理一个或多个 `(M,N)` tile。每个 tile 内先完成
K 维循环累加，再由 fixpipe 将结果送入 UB。尾块场景下，kernel 根据当前 tile 的实际
`blockShapeM/blockShapeN` 裁剪有效计算和写回范围，避免越界访问。

Matmul 部分复用 CATLASS 的 `BlockMmadTla`、`MmadPingpong`、`MmadPingpongMutex`
和 `BlockSchedulerAswt` 能力。融合部分集中在自定义 epilogue 中完成，核心计算不改变
矩阵乘语义，只改变输出写回地址。

permute 后的目标地址不是简单的 RowMajor 连续写回。实现中将输出列按 `N / S2`
拆分为若干连续段，优先使用连续搬运写回 GM；对不满足对齐要求的小段，先在 UB 中整理为
对齐的临时片段，再批量写回。

epilogue 写回路径分为以下几类：

1. 通用分段写回：按 `dst = (((m/S1*S2+n/(N/S2))*S1+m%S1)*(N/S2)+n%(N/S2))`
   计算目标地址，逐连续段搬运。
1. 连续段写回：当 tile 的列段与 `N/S2` 边界对齐时，将多个输出行合并为连续
   `DataCopyPad`，减少小段搬运次数。
1. 小 N pack 写回：当 `N/S2 <= 8` 且 tile 覆盖完整 N 维时，按 `S1 x N` 行组先在
   UB scratch 中整理为目标布局，再一次连续写回 GM。
1. ConfusionTranspose pack 写回：当 tile 覆盖完整 N 维且启用对应 `vectorPackMode` 时，
   先把 `S1 x N` 行组 copy 到 compact scratch，再使用 `ConfusionTranspose102` 或同等
   UB 内重排能力生成 packed 数据，最后连续写回 GM。
1. UB residue pack 写回：当源 UB 地址或搬运长度不满足 32B 对齐时，先把 residue 段整理到
   对齐 scratch，再使用 `DataCopyPad` 搬出。
1. split-K reduce+permute 写回：split-K 路径中 AIV 先从 workspace 读取多个 K 分片结果，
   在 UB 中逐片累加，cast 到输出 dtype 后按 0213 地址映射写回。

该重排不是单独启动的 transpose/permute kernel；它发生在 matmul epilogue 的输出阶段。
L0C 到 UB 的搬运由 fixpipe 完成，最终 0213 映射在 UB 到 GM 写回路径中完成。

epilogue 的 UB 与流水规划如下：

1. UB 分配：fixpipe 首先将当前 `(blockShapeM, blockShapeN)` tile 的 matmul 结果写入
   UB 主区域。scratch 起始地址位于主结果之后，按
   `fullBlockShapeM * srcStrideM` 计算；当 `SPLIT_M` 开启时，每个 sub block 使用独立
   scratch 区，避免两个 AIV sub block 写同一段 UB。
1. 单次处理长度：通用路径以 `N/S2` 为自然连续段边界，单次处理长度为当前 tile 与
   `N/S2` 边界的交集；连续段路径在 `baseN` 与 `N/S2` 对齐时可把多个 row 的同一列段
   合并为一次 `DataCopyPad`；小 N pack 与 ConfusionTranspose pack 路径的单次整理长度
   为一个 `S1 x N` 行组。
1. 对齐处理：当源 UB 地址与搬运长度满足 32B 对齐时直接搬运；否则先逐元素写入 scratch，
   scratch 按 block 对齐后再搬出到 GM。
1. 流水同步：MMAD 与 fixpipe 由 CATLASS 主流程调度，epilogue 在读取 fixpipe 写入的 UB
   tile 后执行 pack 与 GM 搬出；当 scratch 或 packed buffer 被写入后，使用
   `PipeBarrier<PIPE_ALL>` 保证 pack 数据对后续 `DataCopy`/`DataCopyPad` 可见，避免读写乱序。
1. 尾块处理：`blockShapeM/blockShapeN` 由 scheduler 给出实际有效范围，epilogue 仅处理
   有效元素，最后一块 M/N 不整除当前 tile 时不会访问越界地址。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 | √ |
| Ascend 950PR | √ |

## 算子约束限制

1. 当前仅支持 FP16 RowMajor 输入。
1. 当前验收路径固定 `S1=8`、`S2=4`；模板支持编译期覆盖。
1. `M` 必须能被 `S1` 整除。
1. `N` 必须能被 `S2` 整除。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出与 `torch.mm + reshape + permute` 参考结果满足生态算子开源精度标准 | 任务书、生态算子开源精度标准 |
| 性能标准 | Ascend 950PR 上 fused kernel 整体性能达到 `torch.mm + permute` baseline 的 1.1 倍 | 任务书 |

## 兼容性分析

该任务为 CATLASS 新增创新样例和 optest 交付件，不修改已有公开算子语义。新增注册项仅用于
`ascend950_matmul_permute` JIT/optest 路径；已有 CATLASS 示例和 optest 接口不受影响。

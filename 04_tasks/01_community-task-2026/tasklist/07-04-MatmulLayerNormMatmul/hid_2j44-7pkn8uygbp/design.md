
# MatmulLayerNormMatmul算子开发文档


# 1. 需求背景

## 1.1 需求来源

7月社区任务-MatmulLayerNormMatmul算子开发任务书

## 1.2 背景介绍

### 1.2.1 MatmulLayerNormMatmul算子实现

基于Ascend C编程语言和CATLASS模板库，开发融合LayerNorm的双Matmul算子。

MatmulLayerNormMatmul算子实现路径和相关API路径：
- 算子开源仓地址：<https://gitcode.com/cann/catlass>
- 参考样例路径：`catlass/examples/44_quant_matmul_full_loadA_tla`
- 测试工程路径：`catlass/tests/optest`

### 1.2.2 MatmulLayerNormMatmul算子实现现状分析

通过对任务书的功能分析，当前需要实现的算子支持的能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| A0 | 第一个Matmul左矩阵 | tensor | float16 | RowMajor布局 | (M₀, K₀) |
| B0 | 第一个Matmul右矩阵 | tensor | float16 | ColumnMajor布局 | (K₀, N₀) |
| B1 | 第二个Matmul右矩阵 | tensor | float16 | ColumnMajor布局 | (N₀, M₁) |
| gamma | LayerNorm缩放参数 | tensor | float32 | 可学习参数 | (N₀,) |
| beta | LayerNorm偏置参数 | tensor | float32 | 可学习参数 | (N₀,) |
| C1 | 第二个Matmul输出矩阵 | tensor | float16 | 唯一对外输出 | (M₀, M₁) |

计算公式：
1. C0 = A0 × B0
2. C0_norm = LayerNorm(C0, gamma, beta)
3. C1 = C0_norm × B1

### 1.2.3 MatmulLayerNormMatmul算子功能分析

MatmulLayerNormMatmul算子功能：单次Kernel Launch完成 Matmul → LayerNorm → Matmul 三段计算，中间结果无需中间显存读写。

输入：A0、B0、B1、gamma、beta

输出：C1

支持数据类型：float16（矩阵）、float32（LayerNorm参数）

支持广播：不支持

# 2. 需求分析

## 2.1 需求描述

使用Ascend C编程语言实现MatmulLayerNormMatmul算子，单次Kernel完成三段计算，Mean和Variance等中间结果由算子内部Workspace管理，不对外暴露。

## 2.2 需求拆解

1. 实现 Matmul → LayerNorm → Matmul 三段计算的单Kernel融合
2. 内部Workspace管理中间结果，消除中间显存读写
3. 支持多种Matmul方案以达成最优性能
4. 性能不低于 torch.mm + F.layer_norm + torch.mm 拼接方案的1.1倍
5. 精度满足生态算子开源精度标准

# 3. 详细设计

## 3.1 算子分析

### 3.1.1 数学公式

阶段一（第一个Matmul）：C0 = A0 × B0，输出形状 (M₀, N₀)
阶段二（LayerNorm，对C0的每一行独立计算）：
- μ[m] = (1/N₀) * Σ C0[m, n]
- σ²[m] = (1/N₀) * Σ (C0[m, n] - μ[m])²
- C0_norm[m, n] = (C0[m, n] - μ[m]) / √(σ²[m] + ε) * γ[n] + β[n]
阶段三（第二个Matmul）：C1 = C0_norm × B1，输出形状 (M₀, M₁)
其中 ε = 10⁻⁶

### 3.1.2 支持数据类型

float16（矩阵计算）、float32（LayerNorm参数）

### 3.1.3 支持形状

不支持广播，严格按照任务书定义的矩阵维度进行计算

## 3.2 算子实现

### 3.2.1 实现方案

#### 3.2.1.1 host侧设计：

tiling策略：

本算子为矩阵乘法与归一化的融合算子，Host侧主要负责Tiling参数的计算与下发。由于存在两次Matmul且共享中间矩阵C0，Tiling策略需要同时兼顾两次Matmul的切分以及LayerNorm的行级归一化边界。

任务均分：根据M₀、N₀、K₀、M₁的维度大小，结合Ascend 950的AI Core数量和UB内存大小，计算最优的TileShape。确保每个Core分配的Tile块大小均匀，避免长尾效应。

批量搬运：针对A0、B0、B1三个输入矩阵，设计Double Buffer流水线。在Host侧计算好每次搬运的数据偏移量和大小，Kernel侧通过异步DMA搬运指令实现数据加载与计算的并行。

##### 1. 分核策略：
优先使用满核的原则。

根据M₀的行数进行切分，如果核间能均分，大核小核数据块一致；
如果核间不能均分，需要将余出的数据块（尾部Tile）分配到前几个核上，确保所有Core都有计算任务。

当前分核现状
- Matmul 阶段：M/N 二维分块调度，充分利用多核算力；
- LN 阶段：仅 M0 按行分核，每行完整落在单核内，逻辑简单、无通信开销、稳定性高。
各 M0 规格核利用率分析
针对测试集 M0∈{128,512,1024,2048}，结合  核 AIC+AIV 硬件资源：
- M0=128（小场景）：单迭代行数少，、核均分后每核仅处理约 3 行，行粒度已足够细。且 LN 耗时占比 <15%，不是性能瓶颈，无需 N0 方向补充分核。
- M0=512/1024/2048（中大场景）：M 方向 Tile 数量充足，核利用率饱满，性能瓶颈集中在 MTE2 带宽，不属于分核策略问题。

输入数据大小计算：通过A0、B0、B1的Shape和数据类型长度，计算出各矩阵的总字节数。

UB内存大小和核心数量获取：通过平台信息获取Ascend 950的UB内存大小和AI Core数量，根据矩阵维度动态调整Tile的行列大小。

##### 2. 数据分块和内存优化策略：

充分使用 UB 空间，并深度结合 Ascend 950PR/DT 硬件特性的原则。

由于中间矩阵 C0 不回写 Global Memory，且需充分利用 Ascend 950PR/DT 引入的 L0C -> UB / UB -> L1 专用数据通路，C0 的 Tile 必须完全驻留在 UB 中，且空间分配需兼顾 Cube 核与 Vector 核间的数据高效分发。

UB 内存大小获取：通过 GetCoreMemSize 函数获取当前硬件的 UB 内存大小，用于后续的数据切分计算。
Tile 块大小计算与空间分配：

根据 UB 内存大小、Double Buffer 需求，以及 LayerNorm 规约所需的临时变量空间，计算出每个 Tile 块在 M 和 N 方向的数据数量。

针对 L0C -> UB 通路：在规划 Tile 大小时，需确保 C0 在 L0C 中的计算分块大小能够平滑、高效地通过 Fixpipe 专有通道直接分发至 Vector 核的 UB 中，避免经由 GM 中转带来的冗余操作。

针对 UB -> L1 通路：在进行第二次 Matmul 计算时，规划好 UB 中 C0_norm 数据向 L1 Buffer 的搬运策略，确保数据能直接从 UB 搬运至 L1，提升融合算子的整体搬运效率。



##### 3. tilingkey规划策略：

需要tilingkey的情况：根据M₀、N₀、K₀、M₁的对齐情况以及是否开启Double Buffer，在Host侧生成不同的tilingkey，Kernel侧根据tilingkey选择最优的Matmul计算模板和LayerNorm规约策略。

#### 3.2.1.2 kernel侧设计：

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

**算子计算流程图：**
![image.png](https://raw.gitcode.com/user-images/assets/10320568/adef24ca-ecaf-42ab-bad9-930648ec5593/image.png 'image.png')


**关键实现细节：**
1. 在Kernel侧的CopyIn阶段，利用Double Buffer机制异步加载A0、B0、B1的Tile块到UB。
2. Compute阶段采用分段流水线设计：第一阶段循环调用Matmul模板完成C0 = A0 × B0计算，并将结果通过L0C->UB通路暂存于UB中，直至凑齐当前Core分配的完整行（N0维度）所有C0 Tile；第二阶段对整行C0利用Vector指令并行计算Mean和Variance，完成归一化及γ、β缩放偏置后，通过UB->L1通路将C0_norm搬运至L1，与B1进行第二次Matmul计算得到C1。
当前测试集最大N0=8192，单行 LN 计算量可完全容纳在UB 内，无需 N0 方向分块、无需核间协同。

因此每行完整处理：

  每行流程（单 pass）：
  1. DataCopy: GM→UB 载入一整行 C0[row, 0:N0] (fp16)
  2. Cast fp16→fp32
  3. ReduceSum → mean = sum / N0
  4. Adds(x - mean) → Mul(平方) → ReduceSum → var = sq_sum / N0
  5. rstd = 1/sqrt(var + eps)
  6. Muls(×rstd) → Mul(×gamma) → Add(+beta)
  7. Cast fp32→fp16 (CAST_RINT)
  8. DataCopy: UB→GM 写回 C0norm[row, 0:N0]
  
若后续业务 N0 超过该阈值，将触发 N0 分块逻辑，采用标准 Two-Pass 归一化方案：
Pass1：按 Chunk 逐段载入、计算 partial_sum / partial_sq_sum，累积全局统计量；
汇总得到全局 mean、var、rstd；
Pass2：再次逐 Chunk 载入数据，完成归一化、Gamma/Beta 缩放偏移；
  workspace 需求：仅 mean/var 标量（已在寄存器/scalar），gamma/beta 需按 chunk 分段加
  载（或常驻部分 UB）。

  LN 与第二次 Matmul 的衔接策略当前实现：全部 C0_norm 计算完毕后，再整体启动 mm1。


3. 在CopyOut阶段，将C1的结果异步写回Global Memory。
4. Mean和Variance等中间变量仅在寄存器或UB临时区存活，不分配Global Memory空间。


## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 | √ |

## 3.4 算子约束限制

无

# 4. 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 算子计算精度需满足生态算子开源精度标准 | 生态算子开源精度标准 |
| 性能标准 | 算子整体性能需达成 torch.mm + F.layer_norm + torch.mm 小算子拼接方案的1.1倍 | 7月社区任务书性能要求 |

## 4.2 兼容性分析

新算子，不涉及兼容性分析。需基于CATLASS-optest测试工程补充测试交付件，并使用catlass-example-to-pytest自动生成Pytest测试脚本。
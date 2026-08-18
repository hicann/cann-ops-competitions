# 需求背景（required）

## 需求来源

本设计对应 CANN 社区 2026 年 7 月任务“MatmulGatherScatter 算子开发”。

## 背景介绍

MatmulGatherScatter 面向推荐系统、稀疏特征行更新等场景。该类场景中，参与矩阵乘的左矩阵行通常只占完整矩阵 `A(M,K)` 的一部分，行号由 `indices(J)` 指定；矩阵乘结果需要再写回输出矩阵 `D(M,N)` 的相同行位置。

算子语义如下：

```text
D[indices[i], j] = sum(A[indices[i], k] * B[k, j]), k in [0, K)
D[other rows, :] = 0
```

等价向量化表达为：

```text
D[indices, :] = A[indices, :] x B
```

其中 `D` 的非索引行保持为 0。任务要求实现融合的 Gather A（按行）+ Matmul + Scatter D（按行），减少小算子拼接中的中间数据搬运和 launch 开销。

## 输入输出说明

| 参数 | 参数含义 | 数据类型 | 格式 | 形状 | 约束 |
| --- | --- | --- | --- | --- | --- |
| A | 左矩阵 | FP16 | RowMajor | `(M,K)` | 按 `indices` gather 行 |
| B | 右矩阵 | FP16 | RowMajor | `(K,N)` | 连续矩阵 |
| indices | 行索引 | INT32 | ND | `(J)` | 调用方保证不重复且在 `[0,M)` 范围内 |
| D | 输出矩阵 | FP16 | RowMajor | `(M,N)` | 仅 `indices` 指定行被写入，其余行为 0 |

# 需求分析（required）

## 需求描述

基于 CATLASS/Ascend C 实现 `Gather A + Matmul + Scatter D` 融合算子。主计算采用 Cube/MMAD 完成矩阵乘，避免使用 AIV 标量循环实现 matmul。算子输入为 FP16 RowMajor 矩阵和 INT32 行索引，输出为 FP16 RowMajor 矩阵。

## 需求拆解

1. 支持 `A/B/D` FP16 RowMajor 和 `indices` INT32。
2. 支持任务测试集覆盖的 `M/K/N/J` 组合。
3. `indices` 由调用方保证合法、不重复，kernel 内不做越界和重复性校验。
4. Gather A 和 Scatter D 使用同一份 `indices`。
5. 输出矩阵 `D` 需要满足非索引行置 0、索引行写入 matmul 结果的语义。
6. 矩阵乘主体使用 CATLASS MMAD 计算路径。
7. 根据 shape 和索引模式选择不同 tiling 或 kernel 路径，以兼顾小 `J`、大 `K/N`、连续索引和普通索引场景。

# 详细设计（required）

## 算子分析

### 数学公式

```text
D[indices[i], j] = sum(A[indices[i], k] * B[k, j])
```

其中：

```text
i in [0, J)
j in [0, N)
k in [0, K)
```

`D` 中不在 `indices` 的行保持为 0。

### 支持数据类型

| 输入/输出 | 数据类型 |
| --- | --- |
| A | FP16 |
| B | FP16 |
| indices | INT32 |
| D | FP16 |

### 支持形状

支持任务测试集中覆盖的 shape 组合。融合 matmul 的逻辑 shape 为：

```text
(J, K) x (K, N) -> (J, N)
```

最终输出 shape 为：

```text
(M, N)
```

## 算子实现

### 整体方案

整体采用 CATLASS 中 GEMM 的分层组织方式，在矩阵乘主干中融合 A 行 gather，并在输出阶段融合 D 行 scatter。

```text
A Tile Copy with row gather
        |
        v
Cube/MMAD GEMM
        |
        v
Epilogue with row scatter
```

核心思路：

1. 将 `indices` 作用在 A operand 的 tile copy 阶段，MMAD 看到的是逻辑连续的 `(J,K)` 左矩阵。
2. 使用 CATLASS GEMM kernel 和 block 层的 MMAD 组件完成 `(J,K) x (K,N)` 主计算。
3. 在 epilogue 阶段根据同一份 `indices` 将 `(J,N)` 结果按行写回 `D(M,N)`。
4. 对连续索引、等差索引和普通索引分别设计路由，减少不必要的 scatter 或地址计算开销。

### Host 侧设计

Host 侧负责参数解析、tiling 选择和执行路径选择，不在 host 侧校验 `indices` 的合法性。

主要工作：

1. 读取输入 shape，得到 `M`、`K`、`N`、`J`。
2. 设置 `A`、`B`、`indices`、`D` 以及必要 workspace 地址。
3. 根据 `M/K/N/J` 选择合适的 tile shape、block 数和 work item 数。
4. 根据索引模式选择 direct-GM 或通用 scatter 路径。
5. 将 tiling 参数、shape 参数、路由标识写入 tiling data。

索引模式识别用于选择更合适的执行路径：

| 索引模式 | 判断条件 | 设计用途 |
| --- | --- | --- |
| prefix 连续索引 | `indices[i] == i` | 可按普通 GEMM 前缀行处理，减少 gather/scatter 地址映射 |
| 等差索引 | `indices[i] == firstRow + i * rowStride` | 可用带 stride 的 layout 表达 A/D 行访问 |
| 普通索引 | 不满足上述条件 | 走通用 gather/scatter 路径 |

其中 `firstRow` 为 `indices[0]`，`rowStride` 为相邻索引差值。等差索引判断需要遍历 `indices`，当任意位置不满足等式时退回普通索引路径。

### Kernel 侧设计

Kernel 侧采用 AIC 主计算配合必要 AIV 后处理的组织方式。AIC 负责 tile copy 和 MMAD，AIV 负责普通路径下的 scatter 或输出置零辅助工作。对于可由 layout 表达的连续或等差索引路径，优先让 AIC 直接写回 GM。

#### Gather A Tile Copy

普通 GEMM 读取 A 的方式为：

```text
A[m, k]
```

MatmulGatherScatter 的逻辑读取方式为：

```text
A[indices[m], k]
```

因此在 A operand 的 tile copy 中增加行索引映射：

```text
for localM in tileM:
    srcRow = indices[rowBase + localM]
    copy A[srcRow, kBase:kBase+tileK] -> A tile
```

其中 `rowBase` 是当前输出 tile 在逻辑 `J` 维上的起始行，`kBase` 是当前 K 分块的起始列。该设计避免在 GM 中生成额外的 `A_gathered(J,K)` 临时矩阵。

当 `K` 大于单次 L1/L0 tile 可覆盖范围时，GEMM 主循环按 `K` 维分块迭代。每次迭代从 `kBase` 开始搬运当前 `tileK` 范围内的 A/B 数据，MMAD 结果在 accumulator 中累加，直到覆盖完整 `K` 维。

#### MMAD 主计算

主计算复用 CATLASS GEMM 的 MMAD 路径，逻辑 problem shape 为：

```text
GemmCoord{J, N, K}
```

核心模板形态如下：

```text
BlockMmadTla<
    DispatchPolicy,
    L1TileShape,
    L0TileShape,
    half,
    half,
    half,
    void,
    TileCopy
>
```

其中 `TileCopy` 根据路由选择不同实现：

| TileCopy | 作用 |
| --- | --- |
| Gather/Scatter TileCopy | 通用索引下，A 按 `indices` gather，C tile 交给 epilogue scatter |
| Direct-GM TileCopy | 连续或等差索引下，通过 layout stride 表达 A/D 行访问，C tile 直接写回 GM |

#### Scatter D Epilogue

通用路径下，MMAD 输出先形成逻辑 `(J,N)` tile，再由 epilogue 按 `indices` 写回 `D(M,N)`：

```text
for localM in tileM:
    dstRow = indices[rowBase + localM]
    copy C[localM, :] -> D[dstRow, nBase:nBase+tileN]
```

其中 `nBase` 是当前输出 tile 在 `N` 维上的起始列。该路径支持任意合法、不重复的行索引。

#### Direct-GM 路径

对于 prefix 连续索引或等差索引，A 和 D 的行访问可通过 layout stride 表达：

```text
A row offset = (firstRow + logicalM * rowStride) * K
D row offset = (firstRow + logicalM * rowStride) * N
```

此时 epilogue 不需要额外 scatter 循环，MMAD 输出可直接写入目标行。该路径用于降低规则索引场景中的 AIV 访存和同步开销。

### CATLASS 组件改动设计

根据 CATLASS GEMM 的分层结构，本算子拟新增或修改以下组件：

| 组件层级 | 设计内容 | 说明 |
| --- | --- | --- |
| example | 新增 MatmulGatherScatter example | 提供算子入口、shape 参数解析、数据初始化、结果校验和性能计时 |
| gemm/kernel | 新增 MatmulGatherScatter kernel 封装 | 负责接收 tiling data，组织 AIC/AIV 执行流程，并根据路由选择具体 block/epilogue 组合 |
| gemm/block | 扩展 GEMM block tile copy | 在 A operand copy 中加入 `indices` 行映射，同时复用 MMAD 主计算 |
| epilogue/block | 新增 scatter epilogue block | 负责将逻辑 `(J,N)` 输出 tile 按 `indices` 写回 `D(M,N)` |
| epilogue/tile | 新增或扩展 tile 级写回逻辑 | 支持普通 scatter、direct-GM 和尾块处理 |
| host tiling | 新增 shape/indices 路由参数 | 根据 `M/K/N/J`、索引模式和硬件资源选择 tile shape、block 数和执行路径 |
| optest wrapper | 新增 Python/C++ 调用封装 | 用于任务自测、精度验证和性能验证 |

### Shape 与 Tiling 策略

不同测试 shape 的瓶颈不同，设计上按 shape 特征选择 tiling 或 kernel 路径，而不是所有场景固定一套 tile。

任务测试集覆盖的主要范围如下：

| 维度 | 取值范围 |
| --- | --- |
| `M/J` | `(400,128)`、`(500,256)`、`(1000,512)`、`(2000,1024)`、`(4000,2048)`、`(8000,4096)` |
| `K` | `128`、`256`、`512`、`768`、`2048` |
| `N` | `128`、`256`、`512`、`1024`、`2048`、`4096` |

据此将 shape 分为以下几类：

| 场景 | 适配范围 | 设计策略 |
| --- | --- | --- |
| 小 `J` | `J <= 256`，对应 `M <= 500` | 减小 M tile，并优先沿 N 维展开任务，提高 AIC 并行度 |
| 普通 `J` | `512 <= J <= 1024`，对应 `1000 <= M <= 2000` | 使用通用 MMAD + scatter 路径，平衡单 tile 计算量和任务数 |
| 大 `J` | `J >= 2048`，对应 `M >= 4000` | 使用较大的 M tile，减少 tile 调度和 scatter 元数据开销 |
| 小/普通 `N` | `N <= 1024` | 通过较小 N tile 增加 `nTiles`，避免任务数过少 |
| 大 `N` | `N >= 2048` | 使用较大 N tile，提升单 tile 计算强度并减少 B 重复搬运 |
| 小/普通 `K` | `K <= 768` | K 维循环次数较少，优先保证 M/N 维任务并行 |
| 大 `K` | `K = 2048` | 按 K 维分块循环累加，tileK 选择需满足 L1/L0 容量约束 |
| prefix 连续索引 | `indices[i] == i` | 退化为前缀行 GEMM 或 direct-GM 写回 |
| 等差索引 | `indices[i] == firstRow + i * rowStride` | 使用 stride layout，减少显式 scatter |
| 普通索引 | 无法由简单 stride 表达 | 使用通用 gather/scatter 路径保证功能覆盖 |

候选 tile shape 会围绕 `M/N/K` 三个维度配置，例如：

| Tile 类型 | 设计目标 |
| --- | --- |
| 小 M tile | 提升小 `J` 场景并行度 |
| 通用 M tile | 覆盖常规 shape |
| 大 M/N tile | 提升大 `J`、大 `N` 场景 Cube 利用率 |
| Direct-GM tile | 服务 prefix 或等差索引路径 |

具体 tile 参数在实现阶段结合任务测试集 shape、L1/L0 容量、block 数和 workspace 约束确定。

### 任务分核方案

分核以输出逻辑矩阵 `(J,N)` 的 tile 网格为基本单位。给定 `tileM`、`tileN`、`tileK`：

```text
mTiles = ceilDiv(J, tileM)
nTiles = ceilDiv(N, tileN)
kTiles = ceilDiv(K, tileK)
taskCount = mTiles * nTiles
```

每个 task 负责一个 `(tileM,tileN)` 输出 tile，并在内部完成 `kTiles` 次 K 维循环累加。task 到 AIC 的映射采用静态 round-robin：

```text
for taskId = blockIdx; taskId < taskCount; taskId += blockNum:
    mTileIdx = taskId / nTiles
    nTileIdx = taskId % nTiles
    rowBase = mTileIdx * tileM
    nBase = nTileIdx * tileN
    for kTileIdx in [0, kTiles):
        kBase = kTileIdx * tileK
        Gather A tile and load B tile
        MMAD accumulate
    Epilogue scatter or direct-GM store
```

`blockNum` 设计原则如下：

| 场景 | 分核原则 |
| --- | --- |
| `taskCount >= aicCoreNum` | `blockNum` 取可用 AIC core 数，task 按 round-robin 均分 |
| `taskCount < aicCoreNum` 且 `N` 较大 | 优先减小 `tileN`，增加 `nTiles` |
| `taskCount < aicCoreNum` 且 `J` 较大 | 优先减小 `tileM`，增加 `mTiles` |
| `taskCount` 仍不足且 `K` 较大 | 保留 K 维循环累加；如引入 split-K，需要额外 workspace/reduction 设计 |
| 普通索引 scatter 路径 | AIC 产生输出 tile，AIV 按相同 task 粒度完成 scatter，AIC/AIV 通过 flag 同步 |
| direct-GM 路径 | AIC task 直接写回目标行，不再分配 AIV scatter task |

该分核方式保证不同 shape 下的执行入口一致，差异主要体现在 host 侧选择的 `tileM/tileN/tileK/blockNum` 和 epilogue 路由。

### 输出置零设计

任务语义要求 `D` 的非索引行为 0。设计上将置零语义与主计算路径解耦：

1. `Gather A + Matmul + Scatter D` kernel 只写入 `indices` 指定行。
2. `D` 的初始置零可由调用侧或独立初始化流程完成。
3. 若某些路径需要在 kernel 内补充置零，仅作为满足输出语义的辅助步骤，不改变主计算仍为 MMAD 的设计。

这样可以使主 kernel 聚焦于 gather、matmul 和 scatter，也便于在验收时分别统计初始化和融合计算的耗时。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

1. `A/B/D` 为 FP16 RowMajor。
2. `indices` 为 INT32。
3. `indices` 由调用方保证不重复、合法。
4. kernel 内不做 `indices` 越界检查和重复性校验。
5. 输出矩阵 `D` 在写入索引行前需要已满足非索引行置 0 的语义。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 与更高精度参考实现比较，满足生态算子开源精度标准 | 任务书 |
| 性能标准 | 相对官方小算子拼接 baseline 达到任务要求的加速比 | 任务书 |

参考实现语义：

```python
expected = torch.zeros((M, N), dtype=torch.float16, device="npu")
expected[indices, :] = torch.matmul(A[indices, :].float(), B.float()).half()
```

其中 `indices` 与任务输入保持 INT32，以上表达仅用于说明索引语义。

检查内容：

1. `D[indices, :]` 与参考 matmul 结果一致。
2. `D` 非索引行保持为 0。
3. 输出 shape 为 `(M,N)`。
4. 输出 dtype 为 FP16。

## 测试计划

| 验证项 | 验证内容 |
| --- | --- |
| 基础功能 | 验证输出 shape、dtype 和 selected rows 的 matmul 结果 |
| zero 语义 | 验证未被 `indices` 选中的行保持为 0 |
| 索引模式 | 验证 prefix 连续索引、等差索引和普通合法索引 |
| K 维分块 | 验证大 `K` shape 下 K 循环累加正确 |
| 边界 shape | 覆盖任务测试集中给定的功能和性能 shape |
| 性能验证 | 与官方小算子拼接 baseline 对比 |

## 验证计划

1. 先做功能验证，覆盖不同 shape 和不同索引模式。
2. 按生态算子开源精度标准统计 matched ratio 和最大绝对误差。
3. 再使用官方 baseline 数据进行性能对比。
4. 使用 profiling 工具分析 MMAD、gather/scatter 访存、同步等待和初始化开销。
5. 对不同路由分别验证功能一致性，保证专用路径和通用 fallback 对外语义一致。

## 兼容性分析

该算子为社区任务新增算子，不涉及已有接口兼容性变更。接口语义与任务书定义保持一致，内部根据 shape 和索引模式选择不同实现路径，对外输出语义保持统一。

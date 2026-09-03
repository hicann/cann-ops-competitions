# 需求背景（required）

## 需求来源

参考 cuSPARSE `cusparseSpMV` 接口语义，在 Atlas A2/A3（DAV_2201，`arch22`）上基于 C++ Host 与 Ascend C Kernel 实现 `aclsparseSpMV`，并在 ops-sparse 仓完成公开 C++ API、Python/ATen 入口、C++ UT 与端到端测试交付。

算子计算公式：

```text
Y = alpha * op(A) * X + beta * Y
```

其中 `A` 为 CSR 稀疏矩阵，`X` 与 `Y` 为稠密向量，`op(A)` 支持非转置、转置和 complex64 场景下的共轭转置语义。

## 背景介绍

### SpMV 算子功能

SpMV 是稀疏矩阵与稠密向量乘加运算，是稀疏线性代数、图计算、稀疏 MoE/LLM 推理和训练中的基础算子。CSR 格式通过三个数组描述稀疏矩阵：

```text
csrRowOffsets: 行偏移数组，长度 rows + 1
csrColInd:     列索引数组，长度 nnz
csrValues:     非零值数组，长度 nnz
```

非转置计算：

```text
for row in [0, M):
    acc = 0
    for p in [rowOffsets[row], rowOffsets[row + 1]):
        acc += values[p] * X[colInd[p]]
    Y[row] = alpha * acc + beta * Y[row]
```

转置计算：

```text
for row in [0, M):
    for p in [rowOffsets[row], rowOffsets[row + 1]):
        col = colInd[p]
        Y[col] += alpha * values[p] * X[row]
```

直接执行转置路径会产生多行写同一 `Y[col]` 的冲突。为保证确定性和减少 atomic 依赖，本设计在 preprocess 阶段将 CSR 转换为 CSC，使 `A^T` 可以等价为按行访问的 CSR 结构，再复用非转置 kernel 路径。

### 工程现状

本任务面向 ops-sparse 仓 `master` 分支交付，要求：

1. 标准 `aclsparseSpMV` 三阶段接口，不与 `aclsparseSpMVOp` 混用。
2. 复用 `aclsparseHandle_t`、SpMat、DnVec 描述符和调用方 stream。
3. A2/A3 `arch22` 实现与 A5 公共 Host 逻辑解耦。
4. 提供 PyTorch 2.7+、torch_npu 26.0.0+ 的 ATen Dispatcher NPU 注册。
5. 公开入口 `torch.mv(mat, vec)` 映射 `aten::mv`，仅覆盖 NPU CSR 二维稀疏矩阵乘一维稠密向量，不允许 CPU fallback。

# 需求分析（required）

## 需求描述

实现 CSR SpMV 的完整 C++/Ascend C/Python 调用链，覆盖 workspace 查询、可选 preprocess 和 execute 三阶段：

```c
aclsparseStatus_t aclsparseSpMVGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    size_t *bufferSize);

aclsparseStatus_t aclsparseSpMVPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    void *externalBuffer);

aclsparseStatus_t aclsparseSpMV(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    const void *beta,
    aclsparseDnVecDescr_t vecY,
    aclDataType computeType,
    aclsparseSpMVAlg_t alg,
    void *externalBuffer);
```

## 功能拆解

1. 支持 CSR 输入、I32 索引、index base 0/1。
2. 支持 `ACLSPARSE_OPERATION_NON_TRANSPOSE`、`ACLSPARSE_OPERATION_TRANSPOSE` 和 complex64 场景下的共轭转置。
3. 支持动态 `M/K/nnz`、`nnz=0`、空行、单行/单列、长尾行、尾块不足、重复坐标。
4. 支持 `alpha/beta` 缩放，`beta=0` 时不依赖 `Y` 原输入值。
5. 支持声明的 dtype 与 computeType 组合。
6. 支持 `GetBufferSize -> Preprocess -> SpMV` 生命周期，workspace 可复用。
7. 支持 `torch.mv` / `aten::mv` 到 NPU aclsparse 调用链的端到端执行。
8. 参数非法、dtype 不支持、layout 不匹配、设备不匹配等场景返回明确错误或抛出 PyTorch 异常。

## 类型组合

| A/X 输入 dtype | Y 输出 dtype | computeType | 计算策略 |
| --- | --- | --- | --- |
| int8 | int32 | int32 | int32 精确累加 |
| int8 | float32 | float32 | 输入转 float32 后累加 |
| float16 | float32 | float32 | FP32 累加，输出 FP32 |
| bfloat16 | float32 | float32 | FP32 累加，输出 FP32 |
| float16 | float16 | float32 | FP32 累加，写回前转 FP16 |
| bfloat16 | bfloat16 | float32 | FP32 累加，写回前转 BF16 |
| float32 | float32 | float32 | FP32 累加 |
| complex64 | complex64 | complex64 | complex64 复数乘加 |
| float32 | complex64 | complex64 | A/X 转 complex64 参与复数累加 |

未列出的 dtype、索引类型、稀疏格式和 computeType 组合返回 `ACLSPARSE_STATUS_NOT_SUPPORTED` 或仓库约定的等价错误码。

## 参数语义

| 参数 | 语义 | 约束 | 异常行为 |
| --- | --- | --- | --- |
| `handle` | 上下文和 stream | 非空、生命周期有效 | 空或失效返回错误 |
| `opA` | 矩阵操作类型 | 非转置、转置、共轭转置 | 非法枚举返回错误 |
| `alpha/beta` | 缩放标量 | dtype 与 `computeType` 匹配 | 空指针或 dtype 不匹配返回错误 |
| `matA` | CSR 稀疏矩阵描述符 | 二维 CSR，I32 索引，base 0/1 | 空、格式错误、shape 错误返回错误 |
| `vecX` | 输入稠密向量 | 连续一维 NPU Tensor | 长度、dtype、device 不匹配返回错误 |
| `vecY` | 输入输出稠密向量 | 连续一维 NPU Tensor | 长度、dtype、device 不匹配返回错误 |
| `computeType` | 累加与标量解释类型 | 与类型组合表一致 | 不支持返回错误 |
| `alg` | SpMV 算法枚举 | 使用仓库声明算法 | 非法或未支持返回错误 |
| `bufferSize` | workspace 字节数输出 | Host 指针，非空 | 空指针返回错误 |
| `externalBuffer` | workspace | Device 指针，容量不小于查询值 | 需要 workspace 时为空返回错误 |

维度要求：

| `opA` | A 形状 | X 长度 | Y 长度 |
| --- | --- | --- | --- |
| NonTranspose | `[M, K]` | `K` | `M` |
| Transpose / ConjugateTranspose | `[M, K]` | `M` | `K` |

# 详细设计（required）

## 总体架构

```text
Python torch.mv
  -> ATen aten::mv NPU registration
  -> CSR layout / dtype / device / shape validation
  -> create or reuse SpMat and DnVec descriptors
  -> aclsparseSpMVGetBufferSize
  -> allocate externalBuffer
  -> aclsparseSpMVPreprocess
  -> aclsparseSpMV
  -> return dense vector

C++ aclsparse API
  -> Host validation
  -> tiling and workspace layout
  -> optional CSR transpose preprocessing
  -> arch22 Ascend C kernel launch

Kernel
  -> row/block task dispatch
  -> vector path for irregular or sparse rows
  -> dense-block path for regular high-density windows
  -> alpha/beta fusion and Y writeback
```

## Host 侧设计

### 参数检查

Host 侧校验顺序：

1. 检查 `handle`、`matA`、`vecX`、`vecY`、`alpha`、`beta`、`bufferSize` 指针。
2. 检查 `opA`、`alg`、`computeType` 是否在支持集合内。
3. 检查 `matA` 为 CSR 格式，行数、列数、`nnz` 非负，row offsets 和 col indices 为 int32。
4. 检查 `idxBase` 为 0 或 1。
5. 检查 `vecX`、`vecY` 为一维连续向量，长度匹配 `opA` 语义。
6. 检查 A/X/Y dtype 和 `computeType` 是否属于支持组合。
7. 检查所有 Tensor 位于 NPU device，ATen 路径不得 CPU fallback。
8. 检查 workspace 计算过程无整数溢出。

`nnz=0` 时仍执行输出语义：

```text
Y = beta * Y
```

当 `beta=0` 时直接写零，不读取 Y 输入值。

### Workspace 规划

`GetBufferSize` 按矩阵结构、dtype、opA 和算法精确计算 workspace：

```text
workspace =
    aligned(sizeof(SpmvMeta))
  + aligned(windowNum * sizeof(WindowInfo))
  + aligned(rowBlockScheduleBytes)
  + aligned(vectorTaskBytes)
  + aligned(blockTaskBytes)
  + aligned(transposeCscBytes)
  + aligned(reorderValueBytes)
  + aligned(tempReduceBytes)
  + aligned(syncBytes)
```

设计原则：

1. 所有区域按 64B 对齐。
2. `transposeCscBytes` 仅在 `opA` 为转置/共轭转置时申请。
3. `reorderValueBytes` 仅在 block 路径启用时申请。
4. 不申请完整稠密矩阵副本。
5. workspace 与 `matA` pattern、`opA`、dtype 组合和 alg 绑定，pattern 变化后重新 preprocess。

### Preprocess

Preprocess 完成与稀疏结构相关的工作：

1. 校验 row offsets 单调性、首尾 base、`rowOffsets[M] - base == nnz`。
2. 对 col indices 做范围校验：base 0 时属于 `[0, K)`，base 1 时属于 `[1, K]`。
3. 对每行统计 `rowNnz`，识别空行、长尾行和高密度区间。
4. 按固定 `ROW_WINDOW=16` 行划分 window，尾 window 按实际行数处理。
5. 根据 window 内 `nnz` 分布生成 Vector 路径和 Block 路径任务表。
6. 转置/共轭转置场景将 CSR 转为 CSC，并把 CSC 作为 `op(A)` 的 CSR 等价结构供 kernel 使用。
7. complex64 共轭转置保留原 values，在 kernel 读取时取共轭，避免复制一份共轭 values。

转置转换关系：

```text
A_cscColPtr -> op(A)_csrRowOffsets
A_cscRowInd -> op(A)_csrColInd
A_cscVal    -> op(A)_csrValues
```

### 路径选择

每个 window 统计：

```text
windowNnz
maxRowNnz
avgRowNnz
paddingRatio = (actualRows * maxRowNnz - windowNnz) / max(1, actualRows * maxRowNnz)
```

路径选择：

| 条件 | 路径 |
| --- | --- |
| `windowNnz == 0` | 只执行 beta 缩放或写零 |
| 行 nnz 分布不均、长尾明显、极低稀疏 | Vector path |
| 行 nnz 分布较均匀、padding 开销可控 | Block path |
| dtype/shape 不适合 block 计算 | Vector path |

首版以确定性和全 dtype 覆盖为优先，Vector path 作为完整兜底；Block path 用于 P-01/P-02/P-03 中每行约 64 个非零的均匀场景。

### 多核调度

Host 侧基于任务代价做贪心分配：

```text
cost(window) = windowNnz + blockPenalty + transposePenalty
```

将 window 分配给当前累计 cost 最小的 core，生成每核 task offset 和 task list。这样能缓解长尾行导致的核间负载不均。

## Kernel 侧设计

### Vector Path

Vector path 直接遍历 CSR 行：

```text
for task in assignedTasks:
    for row in task.rows:
        acc = zero(computeType)
        for p in [rowOffsets[row] - base, rowOffsets[row + 1] - base):
            col = colInd[p] - base
            a = cast(values[p], computeType)
            x = cast(X[col], computeType)
            if conjugate:
                a = conj(a)
            acc += a * x
        Y[row] = alpha * acc + beta * Y[row]
```

优化点：

1. `beta=0` 分支不读取 Y。
2. `alpha=1`、`beta=1` 分支减少复数或浮点乘法。
3. 空行直接执行 `Y=rowBeta`，避免进入内层循环。
4. 行内累加顺序固定，保证确定性。

### Block Path

Block path 面向每行非零数量接近、适合规则化的 window。Preprocess 将 window 内 values 和 col indices 重排为固定宽度 block：

```text
blockWidth = align_up(maxRowNnz, BLOCK_ALIGN)
```

Kernel 对一个 window 的多行并行处理：

1. 搬入 `values` block 和 `colInd` block。
2. 根据 col indices gather `X` 到 UB。
3. 对每行做乘加归约。
4. 执行 alpha/beta 融合。
5. 写回对应 Y 元素。

padding 的 values 按 0 处理，col index 可填 0 但不会影响结果。Block path 仍保持固定归约顺序，保证同输入下结果确定。

### 转置和共轭转置

转置路径在 preprocess 后转换为对 `A^T` 的非转置 CSR 计算：

```text
outputRows = K
inputXLen  = M
```

对于 complex64 共轭转置：

```text
acc += conj(values[p]) * X[col]
```

对于实数 dtype，共轭转置与转置等价。

### dtype 计算策略

| computeType | 累加类型 | 写回 |
| --- | --- | --- |
| int32 | int32 | int32 |
| float32 | float32 | 按 Y dtype 写回 |
| complex64 | complex64 | complex64 |

FP16/BF16 输入统一转 FP32 累加。FP16/BF16 输出在写回前按硬件转换规则处理。complex64 使用实部/虚部分量计算，乘法公式为：

```text
(a + bi) * (c + di) = (ac - bd) + (ad + bc)i
```

### TilingKey 规划

| 字段 | 说明 |
| --- | --- |
| `opMode` | N / T / H |
| `dtypeMode` | int8-int32、int8-f32、fp16-f32、bf16-f32、fp32-f32、complex64、fp32-complex64 |
| `pathMode` | Vector only / Block only / Mixed |
| `scaleMode` | beta=0、alpha=1 beta=1、general |
| `baseMode` | index base 0 / base 1 |

Host 侧组合生成 TilingKey，kernel 侧减少运行时分支。

## Python/ATen 设计

### 入口约束

公开入口：

```python
torch.mv(mat, vec)
```

仅支持：

1. `mat` 为二维 CSR sparse tensor。
2. `vec` 为一维 dense tensor。
3. `mat` 和 `vec` 均位于 NPU。
4. dtype 属于支持组合。
5. 输出为一维 dense tensor。

ATen 路径固定按：

```text
alpha = 1
beta = 0
```

调用底层 `aclsparseSpMV`。其他 alpha/beta 组合由 C++ aclsparse API 提供。

### Dispatcher 注册

在 torch_npu/ops-sparse 适配层注册 `aten::mv` 的 CSR NPU kernel：

```text
aten::mv(Tensor self, Tensor vec) -> Tensor
```

处理流程：

1. 判断 `self.layout == SparseCsr`，否则走已有 dense mv 分支或报错。
2. 校验 `self.dim()==2`、`vec.dim()==1`、`self.size(1)==vec.size(0)`。
3. 校验 device、dtype、stride 和 contiguity。
4. 构造输出 dense vector。
5. 创建 SpMat/DnVec 描述符。
6. 调用三阶段 aclsparse 接口。
7. 释放临时描述符，返回输出。

异常行为与 PyTorch 对齐，不支持的 CSR 属性显式报错，不做 CPU fallback。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列 910B3 | √ |
| Atlas A2 训练系列 910B4 | √ |
| Atlas A3 任务环境型号 | √ |

## 算子约束限制

1. 稀疏矩阵仅支持 CSR。
2. 索引仅支持 int32，index base 支持 0/1。
3. 不支持 BSR、COO、CSC 作为外部输入格式；CSC 仅作为内部转置 preprocess 结果。
4. Python/ATen 路径仅支持 `torch.mv(CSR, dense_vector)`，不支持稀疏矩阵乘矩阵。
5. ATen 路径固定 `alpha=1`、`beta=0`。
6. Device pointer mode 仅在 `aclsparseSetPointerMode` 对应分支实现后声明支持。
7. 重复坐标按 CSR 存储顺序累加，行内累加顺序固定。
8. 不使用 CPU fallback 完成核心计算。
9. Preprocess 元数据与矩阵 pattern、opA 和 dtype 组合绑定，结构变化后必须重新 preprocess。

# 可维可测分析

## 精度标准

精度 golden：

| 类型组合 | Golden |
| --- | --- |
| int8 -> int32 | CPU int64 中间校验后 int32 exact match |
| int8/fp16/bf16 -> fp32 | CPU FP32 golden |
| fp32 -> fp32 | CPU FP64 golden |
| complex64 -> complex64 | CPU complex128 golden |
| fp32 -> complex64 | CPU complex128 golden |

浮点与复数分量按：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

验收阈值：

| dtype | rtol | atol | matched ratio |
| --- | --- | --- | --- |
| float16 | `2^-9` | `2^-9` | `>= 0.99` |
| bfloat16 | `2^-6` | `2^-6` | `>= 0.99` |
| float32 | `2^-10` | `2^-16` | `>= 0.99` |
| complex64 | 实部/虚部分别按 float32 | 实部/虚部分别按 float32 | `>= 0.99` |

int8 到 int32 路径要求逐元素完全一致。

## 性能标准

性能倍率定义：

```text
GPU baseline median_us / NPU median_us
```

目标为达到 GPU 标杆 0.25 倍以上性能。采集规则：

1. GPU 和 NPU 均使用设备侧 Event 统计同一调用范围。
2. warmup 至少 10 次，采样至少 30 次。
3. 报告 median 和 p90。
4. 不计首次编译、数据生成、Host/Device 数据搬运。
5. 复用描述符、workspace 和 preprocess 结果。

核心性能场景：

| 编号 | 场景 | shape / nnz | GPU median_us 范围 | 目标 |
| --- | --- | --- | --- | --- |
| P-01 | Llama 3.1 70B | `8192 x 28672`, `nnz=524288` | `198.144-1090.752` | `>= 0.25x` |
| P-02 | Qwen3-235B-A22B | `4096 x 1536`, `nnz=262144` | `199.120-964.496` | `>= 0.25x` |
| P-03 | DeepSeek-V3 | `7168 x 2048`, `nnz=458752` | `199.312-1054.288` | `>= 0.25x` |

## 内存标准

满足以下条件之一：

1. 输入输出总量超过 500 MB 时，同 Torch API 调用范围内，NPU 额外峰值内存不超过 GPU 使用内存总量的 50%。
2. 无等价 GPU 接口或无法对齐时，方案固有 workspace 不超过目标硬件 L2 Cache 容量。

内存统计字段包括：

```text
input_baseline_*_bytes
peak_*_bytes
extra_peak_*_bytes
```

实现上不保留完整稠密矩阵副本；转置场景只保存 CSC 稀疏结构与必要临时缓冲。

## 自测用例设计

自测覆盖：

| 类别 | 覆盖内容 |
| --- | --- |
| 功能 | N/T/H、base 0/1、alpha/beta、输出覆盖 |
| shape | 方阵、长矩阵、宽矩阵、单行、单列、空矩阵、`nnz=0/1` |
| 稀疏 pattern | uniform、random、diagonal、banded、block_like、many_empty_rows、one_long_row、power_law、skewed、highly_imbalanced |
| dtype | 全部声明 dtype/computeType 组合 |
| 边界 | 空行、长尾行、重复坐标、列索引边界、尾 window |
| 负向 | 非 CSR layout、非 NPU device、dtype 不支持、shape 不匹配、索引越界、rowOffsets 非单调、非法 alg |
| Python | `torch.mv`、`aten::mv` 注册、异常、无 CPU fallback |
| 性能 | P-01/P-02/P-03 和 extra cases |
| 内存 | GPU/NPU peak 和 extra peak 对比 |

测试材料：

```text
test_cases/aclsparseSpMV_testCase/accuracy_cases.json
test_cases/aclsparseSpMV_testCase/performance_cases.json
test_cases/aclsparseSpMV_testCase/gpu_performance_result_benchmark.md
test_cases/aclsparseSpMV_testCase/nodes_accuracy.yaml
test_cases/common/
```

## 可维护性分析

1. Host 侧公共校验、描述符解析、workspace 规划与 arch22 kernel 调度解耦。
2. Vector path 作为全量功能兜底，Block path 只针对适合规则化的 window 优化。
3. 转置在 preprocess 阶段统一转换，kernel 侧复用非转置 CSR 执行框架。
4. dtype 转换与 alpha/beta 后处理集中封装，减少不同路径结果语义漂移。
5. Python/ATen 层只负责 PyTorch 语义适配，底层数值计算统一走 aclsparse C++ API。

## 兼容性分析

本任务新增标准 `aclsparseSpMV` A2/A3 实现，不改变已有描述符 ABI 和公开 API 形态。A2/A3 代码位于 `sparse/spmv/arch22/`，测试位于 `test/spmv/arch22/`，公共 Host 逻辑与 A5 路径通过产品分支和 TilingKey 区分，避免影响其他产品线。

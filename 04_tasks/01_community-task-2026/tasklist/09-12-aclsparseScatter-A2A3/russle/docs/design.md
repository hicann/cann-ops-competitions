# aclsparseScatter A2/A3 算子设计文档

# 需求背景（required）

## 需求来源

本设计面向 9 月社区任务 `aclsparseScatter` 算子开发（A2/A3），目标仓库为 `ops-sparse`，设计文档提交至 `cann-ops-competitions`。

算子参考 cuSPARSE `cusparseScatter` 语义，实现稀疏向量 values 按 indices 写入稠密向量的能力：

```text
Y[X.indices[i] - idxBase] = X.values[i], i in [0, nnz)
```

其中 `vecX` 为只读稀疏向量，`vecY` 为原地写入的稠密向量。核心 scatter 计算在 NPU 完成，不使用 CPU fallback。

## 背景介绍

Scatter 是稀疏向量到稠密向量的基础写入算子，常用于图计算、稀疏特征回填、词表索引写入和稀疏格式转换前后的数据整理。该算子计算逻辑简单，但对接口完整性、dtype 覆盖、index base、重复索引语义、原地输出和调用方 stream 语义要求较高。

任务要求面向 Atlas A2/A3（DAV_2201，`arch22`）实现 `aclsparseScatter`，并补齐 Python/ATen NPU Dispatcher，使 `Tensor.index_copy_(0, index, source)` 能进入 NPU 原生实现路径。

# 需求分析（required）

## 需求描述

实现 `aclsparseScatter` C++ Host 接口、Ascend C Kernel、C++ 测试、Python/ATen 适配与端到端测试。算子接收稀疏向量描述符 `vecX` 和稠密向量描述符 `vecY`，根据 `vecX.indices` 将 `vecX.values` 写入 `vecY` 对应位置。

## 需求拆解

1. 支持 `int8`、`float16`、`bfloat16`、`float32`、`complex64` 五种 values dtype。
2. 支持 I32 indices，支持 index base 0 和 index base 1。
3. 支持 `nnz=0/1`、动态 `size/nnz`、乱序索引、尾块和重复索引。
4. 无重复索引时输出 bit-wise exact；重复索引遵循非确定性 last-write-wins，不实现累加语义。
5. `vecX.indices` 和 `vecX.values` 只读，`vecY` 原地输出，未写入位置保持不变。
6. Host 侧完成参数校验、dtype/base 分发、tiling 生成和调用方 stream 下发。
7. Kernel 侧完成多核并行 scatter，不申请线性 workspace。
8. Python/ATen 层接入 `Tensor.index_copy_(0, index, source)`，禁止 CPU fallback。
9. 提供正确性、性能、内存、Profiler 和 NPU Dispatch 自验证方案。

# 详细设计（required）

## 算子分析

### 数学公式

```text
for i in [0, nnz):
    y_index = indices[i] - idxBase
    Y[y_index] = values[i]
```

`idxBase` 来自稀疏向量描述符，取值为 0 或 1。性能用例保证 indices 无重复；精度用例覆盖重复 indices，此时输出只要求来自对应输入 values 之一，不承诺确定写入顺序。

### 支持数据类型

| 数据 | 类型 |
| --- | --- |
| `vecX.indices` | `int32` |
| `vecX.values` | `int8`、`float16`、`bfloat16`、`float32`、`complex64` |
| `vecY.values` | 与 `vecX.values` 一致 |

`complex64` 按两个连续 `float32` 的 8 字节布局整体搬运，不拆分实部和虚部，也不进行数值计算。

### 支持形状

`vecX.indices` 和 `vecX.values` 形状为 `[nnz]`，`vecY` 形状为 `[size]`。要求换算 base 后的有效 indices 位于 `[0, size)`。

## Host 侧设计

### 参数校验

Host 侧按以下顺序进行校验：

1. `handle`、`vecX`、`vecY` 不为空。
2. `vecX.indices`、`vecX.values`、`vecY.values` 指针有效。
3. `vecX` index type 为 I32，index base 为 0 或 1。
4. `vecX.values` 与 `vecY.values` dtype 一致，且属于声明支持范围。
5. `vecX.nnz >= 0`、`vecY.size >= 0`；`nnz=0` quick return。
6. 输入输出 device 与 handle 绑定上下文一致。
7. 不支持的 layout、stride、alias 组合显式返回错误。

Host 不在常规路径上扫描 Device indices，因此索引越界由调用方前置条件保证，或按异步执行错误协议暴露。

### Tiling 设计

Tiling 数据包含：

| 字段 | 含义 |
| --- | --- |
| `nnz` | 稀疏向量非零元素个数 |
| `size` | 稠密向量长度 |
| `idxBase` | index base，0 或 1 |
| `valueType` | values dtype 分发标签 |
| `elemSize` | 单个 values 元素字节数 |
| `blockDim` | AIV 核数 |
| `tileElems` | 单核单次处理元素数 |

分核策略采用按 `nnz` 均匀切分：

```text
for global_i in core_id; global_i < nnz; global_i += blockDim
```

小规模 `nnz` 自动降低有效核数，避免过多空核；大规模按硬件 AIV 数量满核并行。

### dtype 分发

Host 根据 dtype 下发不同 Kernel 模板：

| dtype | Kernel 元素类型 |
| --- | --- |
| `int8` | `int8_t` |
| `float16` | `half` |
| `bfloat16` | `bfloat16_t` |
| `float32` | `float` |
| `complex64` | `uint64_t` 或等价 8 字节原始类型 |

`complex64` 仅做原始字节复制，不参与复数运算。

## Kernel 侧设计

### 基础执行路径

Kernel 使用 AIV 并行执行，每个 AIV 处理若干 `i`：

1. 从 GM 读取 `indices[i]`。
2. 计算目标位置 `dst = indices[i] - idxBase`。
3. 从 GM 读取 `values[i]`。
4. 将 value 写入 `vecY[dst]`。

由于 Scatter 是非连续写，主要瓶颈在随机 GM 写入。基础实现优先保证无 CPU fallback、低 launch 开销和 dtype 完整覆盖。

### 连续索引优化路径

对于局部连续 indices：

```text
indices[i + k] == indices[i] + k
```

Kernel 可将连续 run 合并为一次向量化搬运，减少逐元素写入开销。该路径不改变重复索引语义；遇到非连续或重复索引时回退逐元素写入。

### 小 dtype 处理

`int8`、`float16`、`bfloat16` 按元素类型写入。若使用 DataCopyPad 搬运，需要按 32B 对齐 UB 临时缓冲，尾部 padding 不写回有效输出之外。`complex64` 使用 8 字节原始搬运，保证 bit-wise 保真。

### 异步语义

Host 使用 `handle` 绑定 stream 下发 Kernel。接口返回不代表计算完成，调用方需保证描述符和 Device 内存在 stream 完成前有效。算子内部不做无必要 Host 侧同步。

## Python/ATen 适配

公开入口：

```text
Tensor.index_copy_(0, index, source)
```

适配层实现要求：

1. 注册 `aten::index_copy_` 的 NPU Dispatcher。
2. 校验 `dim == 0`，`index` 为一维 I32/I64 Tensor；任务核心路径转换或限制为 I32。
3. 校验 `self` 和 `source` dtype 一致且属于支持范围。
4. 校验 `source.shape[0] == index.numel()`，其余维度与 `self` 对齐。
5. 将一维场景映射为 SpVec/DnVec 描述符并调用 `aclsparseScatter`。
6. 对不支持的维度、layout、stride、device 和 alias 组合显式报错。
7. Profiler 中确认进入 NPU Kernel，不走 CPU fallback。

## 支持硬件

| 支持芯片 | 架构目录 |
| --- | --- |
| Atlas A2 训练系列 910B3/910B4 | `arch22` |
| Atlas A3 任务环境型号 | `arch22` |

## 算子约束限制

1. 本任务仅声明 I32 indices。
2. `vecX.values` 与 `vecY` dtype 必须一致。
3. `nnz=0` 成功返回且不修改 `vecY`。
4. 性能 case 不含重复索引；重复索引用于正确性语义验证。
5. 不提供累加 scatter，也不承诺重复索引的确定写入顺序。
6. 不申请 preprocess，不申请与 `nnz` 或 `size` 成线性关系的额外 workspace。

# 可维可测分析

## 精度测试方案

精度以 CPU Golden 为基准：

| dtype | Golden 精度 |
| --- | --- |
| `int8` | 原始 bit 模式 |
| `float16` | float32 Golden |
| `bfloat16` | float32 Golden |
| `float32` | float64 Golden |
| `complex64` | complex128 Golden |

测试覆盖：

1. 五种 dtype。
2. base 0 和 base 1。
3. `nnz=0/1`、普通规模、尾块规模和动态规模。
4. 顺序 indices、乱序 indices、重复 indices。
5. 正负混合、零、小值、离群值、允许的 INF/NAN。
6. `vecX` 只读性、未写入 `vecY` 元素保持不变。
7. 空 handle、空描述符、空指针、dtype 不匹配、非法 index type/base 等异常。

## 性能测试方案

性能按任务书 `performance_cases.json` 中 P-01/P-02/P-03 执行，固定随机种子，性能 case 不使用重复 indices。

测试要求：

1. warmup 不少于 10 次，repeat 为 30 次。
2. 统计 NPU 同调用范围总耗时的 median 和 p90。
3. 与任务书 GPU baseline 的设备 Event `median_us` 对比。
4. 每个有效 dtype/base case 均达到 0.25x 以上。
5. 排除首次编译、数据生成、Host-to-Device 搬运和无关初始化。
6. 分别记录 aclsparse Kernel 耗时和 Python/ATen 端到端耗时。

## 内存测试方案

Scatter 不需要 workspace。内存验证按任务测试包执行：

1. 输入内存为 `indices` 与 `values`。
2. 输出原地复用 `vecY`。
3. 不申请与 `nnz` 或 `size` 成线性关系的额外 Device/Host 临时内存。
4. 使用 NPU memory 脚本与 Profiler 记录峰值内存和固有 workspace。

## 无 CPU fallback 验证

验证方式：

1. Python/ATen 端到端调用 `Tensor.index_copy_(0, index, source)`。
2. 开启 NPU Profiler，确认出现 `aclsparseScatter` 或对应 Ascend C Kernel。
3. 检查日志中无 CPU fallback、Host 逐元素 scatter 或 Device-to-Host 索引回读。
4. 使用调用方 stream 场景验证异步顺序和原地输出复用。

## 兼容性分析

本算子为 ops-sparse 中的 aclsparse Scatter 能力补齐。A2/A3 实现位于 `arch22`，需与 950/A5 的 `arch35` 实现和公共 Host 描述符逻辑共存。公共头文件与 API 命名保持向前兼容。

## 风险与后续计划

1. A2/A3 `arch22` 不直接依赖 950/A5 的 SIMT 实现，避免不同 CANN 编译环境下的 SIMT 头文件兼容问题。
2. 优先完成 `float32`、base0 的既有路径稳定性，再补齐 base1 和其余 dtype。
3. `complex64` 采用原始 8 字节搬运，重点验证 bit-wise 保真和对齐。
4. 完成 C++ UT 后，再接入 Python/ATen Dispatcher，并补 Profiler 无 CPU fallback 证据。
5. 最终在 CANN 9.1.0 及任务指定 PyTorch/torch_npu 环境下完成 A2/A3 全量复验。

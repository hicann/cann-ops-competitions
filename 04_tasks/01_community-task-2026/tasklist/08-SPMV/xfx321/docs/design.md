# SpMV 算子设计文档

| 版本 | 日期 | 修改人 | 修改内容 |
|---|---|---|---|
| v1.0 | 2026-08-26 | Codex | 按 8 月社区任务 SPMV 任务书重写设计文档 |

# 1. 需求背景（required）

## 1.1 需求来源

任务书要求参考 NVIDIA cuSPARSE `cusparseSpMV`，在 Ascend 950PR 上基于 Ascend C 实现功能一致的 SpMV 算子，完成设计、开发、测试和验收交付。

SpMV 计算公式为：

```text
Y = alpha * op(A) * X + beta * Y
```

其中 `A` 为 CSR 稀疏矩阵，`X` 为稠密向量，`Y` 为输入输出稠密向量，`op(A)` 支持非转置和转置两种模式。

## 1.2 背景介绍

SpMV 是稀疏线性代数中的基础算子，核心成本不在乘加本身，而在 CSR 带来的不规则访存、行长度不均衡和转置场景的输出聚合。

本任务要求与 cuSPARSE 接口风格对齐，分三阶段完成：

1. `GetBufferSize` 获取 workspace 大小；
2. `Preprocess` 可选预处理；
3. `SpMV` 执行计算。

本设计采用“host 侧负责参数校验、形状分析和可选预处理；kernel 侧负责按行并行计算和转置累加”的实现思路，优先保证功能完整和可泛化，再做性能优化。

# 2. 需求分析（required）

## 2.1 需求描述

实现与 cuSPARSE SpMV 核心能力对齐的 Ascend C 算子，支持：

- CSR 稀疏格式；
- `trans=false` 和 `trans=true`；
- 任务书规定的数据类型组合；
- `alpha` / `beta` 标量缩放；
- 泛化矩阵规模与 50%~99.9% 稀疏度；
- 非连续 Tensor 输入；
- 三阶段接口调用流程。

## 2.2 数据类型组合

| 输入 A、X 类型 | computeType | 输出 Y 类型 |
|---|---|---|
| float32 | float32 | float32 |
| int8 | int32 | int32 |
| int8 / float16 / bfloat16 | float32 | float32 |
| float16 | float32 | float16 |
| bfloat16 | float32 | bfloat16 |

## 2.3 输入输出与约束

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 形状 |
|---|---|---|---|---|
| csrRowPtr | 输入 | 行偏移数组 | int32 | [M + 1] |
| csrColInd | 输入 | 列索引数组 | int32 | [NNZ] |
| csrVal | 输入 | 非零元素值 | float16 / bfloat16 / float32 / int8 | [NNZ] |
| x_vec | 输入 | 稠密向量 X | float16 / bfloat16 / float32 / int8 | [K] 或 [M] |
| y_vec | 输入输出 | 稠密向量 Y | float16 / bfloat16 / float32 / int32 | [M] 或 [K] |
| trans | 属性 | 是否转置 | bool | - |
| alpha | 属性 | 前置缩放系数 | float32 | - |
| beta | 属性 | 后置缩放系数 | float32 | - |
| compute_type | 属性 | 计算精度 | int32 / float32 | - |

约束：

1. `csrRowPtr` 单调递增，且 `csrRowPtr[0] = 0`；
2. `csrColInd` 在有效列范围内；
3. `trans=false` 时，A 维度为 `M x K`，`x_vec` 长度为 `K`，`y_vec` 长度为 `M`；
4. `trans=true` 时，A 维度视为 `K x M`，`x_vec` 长度为 `M`，`y_vec` 长度为 `K`；
5. 所有输入支持非连续 Tensor；
6. `alpha`、`beta` 按 computeType 对应精度参与计算。

# 3. 详细设计（required）

## 3.1 总体方案

算子实现拆分为三层：

1. 接口层：对外提供 `aclsparseSpMVGetBufferSize`、`aclsparseSpMVPreprocess`、`aclsparseSpMV`；
2. Host 层：完成参数校验、矩阵维度分析、workspace 规划、任务切分和 preprocess；
3. Kernel 层：按行并行执行 SpMV，并完成转置/非转置两种路径的计算。

整体执行流程如下：

```text
API -> 参数校验 -> workspace size / preprocess -> host tiling -> kernel launch -> 结果写回
```

## 3.2 算法设计

### 3.2.1 非转置路径

非转置路径直接按 CSR 行遍历：

```text
for row in [0, M):
    acc = 0
    for p in [csrRowPtr[row], csrRowPtr[row + 1]):
        acc += csrVal[p] * x_vec[csrColInd[p]]
    y_vec[row] = alpha * acc + beta * y_vec[row]
```

该路径适合按“行”并行调度，每个线程或线程组处理若干行，避免跨行同步。

### 3.2.2 转置路径

转置路径有两种实现策略：

1. 直接按原 CSR 反向聚合到输出向量；
2. 在 preprocess 阶段构建转置后的辅助结构，再复用非转置 kernel。

本设计选择第二种方式。原因是：

- 可将转置场景转化为按行遍历，保持 kernel 逻辑一致；
- 降低转置写回时的原子冲突；
- 便于复用非转置路径的行切分和向量化逻辑。

Preprocess 阶段生成：

- 转置后的行偏移；
- 转置后的列索引；
- 可选的临时 value 缓冲；
- 供 kernel 使用的任务划分信息。

## 3.3 Host 侧设计

### 3.3.1 参数校验

Host 侧完成以下检查：

- 描述符是否为空；
- `csrRowPtr` / `csrColInd` / `csrVal` / `x_vec` / `y_vec` dtype 是否符合组合表；
- shape 是否匹配 `trans` 模式；
- `alpha`、`beta` 是否可读取；
- `computeType` 是否落在支持集合中；
- `csrRowPtr` 是否单调递增，`NNZ` 是否一致。

### 3.3.2 workspace 规划

workspace 主要包含：

1. preprocess 的转置辅助结构；
2. 行切分和核任务映射表；
3. 可选的临时归约缓冲；
4. kernel 启动所需的 tiling 数据。

`GetBufferSize` 只负责返回最大可能 workspace，避免多次调用造成重复分配。

### 3.3.3 任务切分

SpMV 的负载与每行 nnz 相关，因此切分以“行块”为单位，而不是简单按总行数均分。

切分原则：

1. 优先按 nnz 均衡；
2. 保持行区间连续，减少地址跳变；
3. 单个 core 处理的 nnz 近似一致；
4. 对长行进行单独标记，避免拖慢普通行批次。

Host 侧生成如下 tiling 信息：

- `core_num`：实际使用核数；
- `row_start[row_group]` / `row_end[row_group]`；
- `nnz_start` / `nnz_end`；
- `row_group_type`：短行 / 中等行 / 长行；
- `transpose_flag`；
- `compute_type`。

### 3.3.4 预处理设计

`Preprocess` 的职责是将昂贵但可复用的操作前置，包括：

- 转置辅助结构构建；
- 行 nnz 统计；
- 行块分桶；
- workspace 初始化。

对于重复调用同一矩阵的场景，preprocess 可显著减少执行阶段开销。

## 3.4 Kernel 侧设计

### 3.4.1 并行模型

Kernel 采用按行并行的方式：

- 一个 core 处理一个或多个连续行块；
- 每个行块内按 nnz 逐项累加；
- `alpha * sum + beta * y_old` 在行结束后一次完成。

这样可以保证：

- 非转置路径无跨行依赖；
- 转置路径复用同样的行遍历逻辑；
- kernel 控制流清晰，便于维护。

### 3.4.2 计算精度处理

为满足任务书中的组合表，计算采用如下规则：

- `float32` 输入：直接以 `float32` 累加；
- `float16` / `bfloat16` 输入：先提升到 `float32` 计算，再按输出类型回写；
- `int8` 输入：乘法和累加统一提升到 `int32`；
- `alpha` / `beta` 使用 computeType 对应精度做缩放。

### 3.4.3 数据搬运与流水

对于每个行块：

1. 读取 `csrRowPtr` 定位行区间；
2. 分批搬运 `csrColInd` 与 `csrVal`；
3. 按列索引 gather `x_vec`；
4. 在本地累加；
5. 与原始 `y_vec` 融合后写回。

对长行采用分段累加，避免单次循环过长导致核利用率下降。

### 3.4.4 转置路径执行

转置路径在 preprocess 后会得到转置辅助结构，因此 kernel 仍按“行内 dot product”执行。

这样做的好处是：

- 避免大量原子加；
- 统一 kernel 代码路径；
- 转置/非转置仅在输入描述符上有所区别。

## 3.5 性能优化方案

1. 按 nnz 均衡切分：优先保证各 core 工作量接近；
2. 连续行区间分配：降低随机访存和调度开销；
3. 预处理复用：转置辅助结构一次构建，多次执行复用；
4. 计算精度上移：低精度输入统一用 float32 / int32 累加，减少中间误差；
5. 分段处理长行：防止少数超长行拖慢整体吞吐；
6. alpha/beta 融合：在行末统一完成缩放与回写，减少额外 pass。

# 4. 可维可测分析

## 4.1 精度标准 / 性能标准

| 验收标准 | 描述 |
|---|---|
| 精度标准 | 满足《生态算子开源精度标准》；任务书要求的数据类型组合均可正确计算 |
| 性能标准 | 整体性能达到 A100 标杆的 0.5 倍以上 |

## 4.2 测试场景

测试覆盖以下场景：

1. 非转置 / 转置；
2. float32 / float16 / bfloat16 / int8；
3. `alpha` / `beta` 典型组合；
4. 不同矩阵规模和稀疏度；
5. 边界场景，如空行、单元素行、极稀疏矩阵；
6. 非连续 Tensor 输入。

## 4.3 性能基准

采用任务书给出的四组基准作为性能验收参考：

| 编号 | 矩阵规模 | 稀疏度 | 数据类型 | GPU A100 性能 |
|---|---|---|---|---|
| 1 | 128x128 | 95% | float32 | 43.9 us |
| 2 | 1024x1024 | 99% | float32 | 46.3 us |
| 3 | 2048x4096 | 97.5% | float32 | 45.4 us |
| 4 | 160220x68750 | 99.9% | float32 | 193.2 us |

## 4.4 风险点

1. 转置路径的辅助结构可能带来额外 workspace 开销；
2. 长行分布极端时，核间负载均衡仍可能不理想；
3. int8 / int32 路径需要严格控制溢出和类型转换；
4. 非连续 Tensor 需要 host 侧正确读取 stride 和偏移信息。

# 5. 结论

本设计围绕 cuSPARSE 风格接口、CSR 计算特点和 Ascend 950PR 的执行模式，给出了可落地的三阶段 SPMV 方案：

- host 侧负责校验、切分和预处理；
- kernel 侧按行并行完成乘加；
- 转置路径通过 preprocess 复用统一执行逻辑；
- 通过 nnz 均衡切分和分段长行处理保证泛化性能。

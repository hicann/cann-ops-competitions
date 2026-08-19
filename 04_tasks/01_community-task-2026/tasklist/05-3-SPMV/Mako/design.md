# 需求背景（required）

## 需求来源

社区任务：基于 Ascend C 编程语言实现 SpMV（Sparse Matrix-Vector Multiplication，稀疏矩阵向量乘）算子，对标 NVIDIA cuSPARSE 的 `cusparseSpMV` 接口。

## 背景介绍

### SpMV 算子概述

SpMV 是稀疏线性代数中的基础算子，用于计算稀疏矩阵与稠密向量乘法，广泛应用于图计算、科学计算、迭代求解器、推荐系统和稀疏神经网络等场景。稀疏矩阵中大部分元素为 0，计算时只遍历非零元，因此算子的核心性能取决于稀疏格式解析、非零元访存、输入向量 gather、行内归约和输出写回效率。

SpMV 算子的标准数学表达式如下：

```text
Y = alpha * op(A) * X + beta * Y
```

其中 A 为稀疏矩阵，X 为输入稠密向量，Y 为输入输出稠密向量，alpha 和 beta 为标量系数，op(A) 表示是否对 A 做转置。

SpMV 算子对标实现路径为 cuSPARSE：`cusparseSpMV`。

### SpMV cuSPARSE 实现现状分析

通过对 SpMV 算子 cuSPARSE 版本的功能分析，其接口能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| matA | 稀疏矩阵 A | sparse matrix descriptor | CSR、COO 等格式 | 本设计以 CSR 为主格式 | M x K |
| x | 输入稠密向量 X | dense vector descriptor | float32、float16、bfloat16、int8 | 与 op(A) 的列数匹配 | op=N 时 K，op=T 时 M |
| y | 输入输出稠密向量 Y | dense vector descriptor | float32、float16、bfloat16、int32 | 与 op(A) 的行数匹配 | op=N 时 M，op=T 时 K |
| opA | A 的转置模式 | enum | NON_TRANSPOSE、TRANSPOSE | 无 | - |
| alpha | 前置缩放系数 | scalar | 与 computeType 一致 | 指针指向的数据类型需与 computeType 匹配 | - |
| beta | 后置缩放系数 | scalar | 与 computeType 一致 | 指针指向的数据类型需与 computeType 匹配 | - |
| computeType | 中间计算类型 | enum | float32、int32 | 与输入输出类型组合匹配 | - |
| externalBuffer | 外部 workspace | void* | device buffer | 用于预处理或临时缓存 | - |

计算公式：

- 非转置：$Y_i = alpha \times \sum_j A_{i,j} X_j + beta \times Y_i$
- 转置：$Y_j = alpha \times \sum_i A_{i,j} X_i + beta \times Y_j$

### SpMV 算子功能分析

SpMV 算子功能：基于 CSR 格式稀疏矩阵执行稀疏矩阵向量乘，支持非转置和转置计算，结果写回 Y。

输入：CSR 稀疏矩阵 A（row_offsets、col_indices、values）、稠密向量 X、输入输出向量 Y、opA、alpha、beta、computeType。

输出：Y 被覆盖为计算结果。

支持数据类型：float32、float16、bfloat16、int8；其中 float16/bfloat16/int8 路径可使用 float32 中间累加，int8 也支持 int32 中间累加。

支持模式组合：opA(N/T) × dtype(fp32/fp16/bf16/int8) × computeType(fp32/int32) × alpha/beta 常用组合。

支持 CSR 泛化规模：覆盖小矩阵、中等矩阵、大矩阵、不同稀疏率、不同 row nnz 分布、空行和长行场景。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 SpMV 算子，支持 CSR 稀疏矩阵格式，支持非转置和转置计算，支持任务要求的数据类型组合，支持 alpha、beta 标量融合，提供与 cuSPARSE SpMV 对齐的 buffer size、preprocess、compute 三阶段接口，性能达到任务要求。

## 需求拆解

1. 支持 CSR 格式输入，正确解析 row_offsets、col_indices、values。
1. 支持非转置 `A * X` 和转置 `A^T * X` 两类计算模式。
1. 支持 float32、float16、bfloat16、int8 输入类型，支持 float32 和 int32 中间计算类型。
1. 支持 alpha、beta 标量融合，覆盖 alpha=1、alpha 通用、beta=0、beta=1、beta 通用等常见分支。
1. 支持泛化 shape，覆盖方阵、宽矩阵、高矩阵、小规模和大规模矩阵。
1. 支持不同稀疏率和 row nnz 分布，包括均匀分布、空行较多、长行、长尾分布等场景。
1. 支持 preprocess 阶段生成必要的 tiling、分块、重排或辅助信息，并允许 compute 阶段复用。
1. 支持可维护的测试设计，从功能、精度、边界、性能和稳定性维度验证算子。

# 详细设计（required）

## 算子分析

### 数学公式

**非转置模式（opA = NON_TRANSPOSE）**：

```text
for row in [0, M):
    sum = 0
    for j in [csrRowOffsets[row], csrRowOffsets[row + 1]):
        col = csrColInd[j]
        sum += csrValues[j] * X[col]
    Y[row] = alpha * sum + beta * Y[row]
```

其中 A 为 M×K 的 CSR 稀疏矩阵，X 为长度 K 的稠密向量，Y 为长度 M 的输入输出向量。

**转置模式（opA = TRANSPOSE）**：

```text
for col in [0, K):
    Y[col] = beta * Y[col]
for row in [0, M):
    xVal = X[row]
    for j in [csrRowOffsets[row], csrRowOffsets[row + 1]):
        col = csrColInd[j]
        Y[col] += alpha * csrValues[j] * xVal
```

直接按 CSR 处理转置模式时会对 Y[col] 产生写冲突，需要使用 atomic scatter、分块归约或预转置重排方案。本设计采用按场景选择的策略：单次调用或小规模矩阵使用直接 scatter atomic；重复调用或大规模矩阵在 preprocess 阶段生成转置等价结构，再复用非转置计算路径。

### 支持数据类型

float32、float16、bfloat16、int8。

alpha 和 beta 为标量指针，其指向的数据类型需与 computeType 保持一致：computeType 为 float32 时使用 float32 标量，computeType 为 int32 时使用 int32 标量。

| 输入 A/X 类型 | computeType | 输出 Y 类型 | 计算策略 |
| --- | --- | --- | --- |
| float32 | float32 | float32 | FP32 乘加和归约 |
| float16 | float32 | float16/float32 | 输入转换为 FP32，FP32 累加，输出前转换 |
| bfloat16 | float32 | bfloat16/float32 | 输入转换为 FP32，FP32 累加，输出前转换 |
| int8 | int32 | int32 | int8 乘法，int32 累加 |
| int8 | float32 | float32 | int8 转 FP32 后计算 |

### 支持形状

A：M×K CSR 稀疏矩阵；NNZ 为非零元数量。

非转置时 X 长度为 K，Y 长度为 M；转置时 X 长度为 M，Y 长度为 K。`csrRowOffsets` 长度为 M+1，`csrColInd` 和 `csrValues` 长度为 NNZ。索引类型使用 int32。

## 算子实现

### 接口设计

接口采用与 cuSPARSE SpMV 对齐的三阶段形式：先通过 `aclsparseSpMVGetBufferSize` 查询外部 workspace 大小，再通过 `aclsparseSpMVPreprocess` 对矩阵结构和算法路径做预处理，最后通过 `aclsparseSpMV` 执行计算。`alpha` 和 `beta` 以 `const void *` 传入，其指向的数据类型必须与 `computeType` 保持一致。

```c
/**
 * @brief 获取稀疏矩阵向量乘法（SpMV）所需的缓冲区大小
 */
aclsparseStatus_t aclsparseSpMVGetBufferSize(aclsparseHandle_t           handle,
                                             aclsparseOperation_t       opA,
                                             const void                *alpha,
                                             aclsparseConstSpMatDescr_t matA,
                                             aclsparseConstDnVecDescr_t vecX,
                                             const void                *beta,
                                             aclsparseDnVecDescr_t      vecY,
                                             aclDataType                computeType,
                                             aclsparseSpMVAlg_t         alg,
                                             size_t                    *bufferSize);

/**
 * @brief 对稀疏矩阵进行预处理，加速后续 SpMV 计算
 */
aclsparseStatus_t aclsparseSpMVPreprocess(aclsparseHandle_t           handle,
                                          aclsparseOperation_t       opA,
                                          const void                *alpha,
                                          aclsparseConstSpMatDescr_t matA,
                                          aclsparseConstDnVecDescr_t vecX,
                                          const void                *beta,
                                          aclsparseDnVecDescr_t      vecY,
                                          aclDataType                computeType,
                                          aclsparseSpMVAlg_t         alg,
                                          void                      *externalBuffer);

/**
 * @brief 稀疏矩阵向量乘法（SpMV）计算入口
 */
aclsparseStatus_t aclsparseSpMV(aclsparseHandle_t           handle,
                                aclsparseOperation_t       opA,
                                const void                *alpha,
                                aclsparseConstSpMatDescr_t matA,
                                aclsparseConstDnVecDescr_t vecX,
                                const void                *beta,
                                aclsparseDnVecDescr_t      vecY,
                                aclDataType                computeType,
                                aclsparseSpMVAlg_t         alg,
                                void                      *externalBuffer);
```

### 实现方案

#### host侧设计：

tiling策略：

采用“行维度分核 + 核内 tile + 行分布感知路径选择”的整体策略。Host 侧根据矩阵规模、NNZ、row nnz 分布、opA、dtype、computeType、alpha/beta 特征和平台 core/UB 信息生成 tiling 数据。Kernel 侧根据 tilingKey 和 tiling data 选择对应计算路径。

非转置 CSR 的基础计算路径采用 SIMT 行级归约：一个 warp 处理一行，warp 内 lane 以 stride=warpSize 遍历该行非零元，通过 shuffle 归约得到行结果。该路径控制逻辑简单、workspace 需求低，适合均匀分布和中短行场景。

针对行 nnz 分布差异较大的矩阵，设计补充长短行分流：短行可合并多行减少 lane 空闲；中等行使用一 warp per row；长行使用多 warp 或多 tile 分段归约，避免单 warp 串行遍历过长行。转置路径根据矩阵复用情况选择 scatter atomic 或 preprocess 生成转置结构。

##### 1. 分核策略：

优先使用满核的原则。Host 侧通过平台接口获取可用 core 数和 UB 大小，禁止写死核数。

非转置模式下按输出行 M 切分任务，每个 core 处理连续行区间。若行数不能被核数整除，将余数分配给前几个 core 或通过 tiling 记录尾块范围。小矩阵场景下实际使用核数不超过有效行数，避免大量空核。

转置模式下如果采用 scatter atomic，按 CSR 原始行切分任务；如果采用预转置结构，则按转置后的输出行切分任务，复用非转置分核策略。

长行分流场景下，Host 侧统计 row nnz，将超长行拆分为多个 work item，多 core 或多 warp 协同完成 partial sum，再做二级归约写回。

##### 2. 数据分块和内存优化策略：

充分利用 UB 空间并减少不规则 GM 访问。基础 SIMT 路径中，row_offsets、col_indices、values 和 X 可以直接从 GM 访问，UB 保存 tile 输出和必要的临时归约数据；优化路径中可按 tile 将 row_offsets 片段、values 片段、col_indices 片段和 X gather 结果搬入 UB，降低重复访问开销。

UB 规划需要考虑以下内容：

1. 本 tile 的输出缓冲 Y_tile。
1. row_offsets 或 row metadata 缓冲。
1. values / col_indices 分段缓存。
1. X gather 缓冲或局部热数据缓存。
1. 长行 partial sum 临时空间。
1. 数据类型转换临时空间。

分块大小由 UB 容量、单次 SIMT 调用最大 warp 数、DataCopy 对齐要求和 dtype 大小共同决定。FP32 输出按 32B 对齐，行数可按 8 个元素补齐；其他 dtype 根据元素大小换算对齐粒度。

preprocess 阶段可根据矩阵结构生成 row block 信息、长行列表、转置辅助结构、分块边界和 workspace 偏移，compute 阶段直接读取这些信息，避免重复分析 CSR 结构。

##### 3. tilingkey规划策略：

需要 tilingKey 的情况：opA、dtype、computeType、idxBase、alpha/beta 特化和算法路径会影响 kernel 分支。

设计采用组合编码方式生成 tilingKey：

```text
tilingKey = op_code | dtype_code | compute_code | beta_code | alg_code
```

其中 op_code 区分非转置和转置；dtype_code 区分 fp32/fp16/bf16/int8；compute_code 区分 fp32/int32；beta_code 区分 beta=0、beta=1、beta 通用；alg_code 区分基础行级 SIMT、短行合并、长行分段、转置 scatter、预转置复用等路径。

Host 侧设置 tilingKey 和 tiling data，Kernel 侧通过模板或编译期分支选择具体实现，减少运行时分支开销。

#### kernel侧设计：

进行 Init 和 Process 两个阶段，其中 Process 包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. Init阶段：读取 TilingData，解析 TILING_KEY 得到 opA、dtype、computeType、算法路径、alpha/beta 分支等信息；绑定 CSR、X、Y、workspace 的 GM 地址；计算本 core 负责的 work item 范围；初始化 UB 临时空间。

2. Process阶段按 work item 遍历。非转置基础路径以 row tile 为单位，每个 tile 启动若干 warp，每个 warp 负责一行：

   CopyIn阶段：读取 row_offsets 或 row metadata，必要时搬入 values、col_indices 或 X gather 片段到 UB；基础路径可直接从 GM 访问 CSR 和 X。

   Compute阶段：每个 lane 以 warpSize 为步长遍历该行非零元，执行乘加累加。float16/bfloat16/int8 输入先转换到 computeType 再参与累加。warp 内通过 shuffle 指令完成归约。长行路径先生成 partial sum，再由二级归约合并。

   CopyOut阶段：根据 alpha/beta 分支生成最终结果。beta=0 时跳过旧 Y 读取；beta=1 时执行加法特化；beta 通用时读取旧 Y 并执行乘加。输出前按 Y dtype 做类型转换并写回 GM。

3. 转置路径：scatter atomic 路径按 CSR 行遍历非零元，将 `alpha * A[row,col] * X[row]` 原子累加到 Y[col]；预转置路径在 preprocess 阶段构造 CSC 或等价 row-major 转置结构，compute 阶段按输出列作为行进行归约，复用非转置路径，适合矩阵重复调用。

4. 边界情况：空行输出为 beta * Y；全零矩阵只执行 beta 分支；单行超长矩阵走长行分段归约；小矩阵限制实际启动核数；NNZ 为 0 时跳过非零元遍历；idxBase 为 1 时在读取 row_offsets 和 col_indices 后统一转换为 0-base。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

仅支持 CSR 格式作为主输入格式；row_offsets 和 col_indices 使用 int32 索引；输入 CSR 需满足 row_offsets 单调不减、首尾 offset 与 idxBase/NNZ 一致、col_indices 不越界；alpha 和 beta 标量的数据类型需与 computeType 保持一致；不在支持表内的数据类型组合返回不支持；转置 scatter 路径性能受列冲突影响，重复调用场景建议使用预转置 preprocess 路径。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 以更高精度的 CPU 计算结果作为参考真值（golden），对 NPU 实际输出（actual）进行误差评估 |《生态算子开源精度标准》 |
| 性能标准 | 达到任务要求的 GPU A100 对标比例 | 任务要求 |

## 测试设计

功能测试设计覆盖算子参数空间和典型稀疏结构。按 opA 设计非转置和转置用例；按 dtype 设计 fp32、fp16、bf16、int8 组合用例；按 alpha/beta 设计 beta=0、beta=1、beta 通用、alpha 通用用例；按 CSR 结构设计空行、全零矩阵、单行长行、行 nnz 均匀、长尾分布和列索引乱序用例；按 shape 设计小矩阵、方阵、宽矩阵、高矩阵和大规模矩阵用例。每类用例均使用 CPU 或 cuSPARSE 结果作为 golden，对输出向量逐元素比较。

精度测试设计按数据类型分别设置阈值。float32 使用严格绝对误差和相对误差阈值；float16/bfloat16 根据输出 dtype 和中间累加精度放宽阈值；int8/int32 路径验证整数累加一致性；int8/float32 路径按浮点误差标准验证。测试同时覆盖 beta 分支，确保读取旧 Y、跳过旧 Y 和通用缩放均正确。

性能测试设计从规模、稀疏率、行分布和复用场景四个维度展开。规模维度覆盖小中大矩阵及典型真实业务 shape；稀疏率维度覆盖中等稀疏到极高稀疏；行分布维度覆盖均匀、空行多、长行和长尾；复用维度分别统计首次 preprocess + compute 和多次 compute 复用 preprocess 的性能。性能指标包括平均耗时、有效带宽、非零元吞吐、preprocess 开销占比和不同算法路径收益。

稳定性测试设计覆盖非法参数和边界条件。包括空指针、非法 dtype 组合、row_offsets 非单调、col_indices 越界、NNZ 不一致、shape 不匹配、workspace 不足、idxBase 不匹配等输入，验证接口返回明确错误码且不产生越界访问。

## 兼容性分析

新算子，不涉及兼容性分析。接口语义与 cuSPARSE SpMV 的 buffer size、preprocess、compute 三阶段流程对齐，后续可在不改变外部调用方式的前提下扩展数据类型、转置 preprocess 和更多算法路径。

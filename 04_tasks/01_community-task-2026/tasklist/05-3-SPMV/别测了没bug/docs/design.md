# SpMV 算子设计文档

# 需求背景

## 需求来源

社区任务要求参考 NVIDIA cuSPARSE `cusparseSpMV`，在 Ascend 950PR 上基于 Ascend C 实现功能一致的 SpMV 算子，并完成设计、开发、测试和验收交付。算子计算公式为：

```text
Y = α · op(A) · X + β · Y
```

其中 `A` 为 CSR 稀疏矩阵，`X` 为稠密输入向量，`Y` 为输入输出稠密向量，`op(A)` 支持非转置和转置。

本设计面向 Ascend 950PR 的硬件资源和 CSR 稀疏矩阵计算特点，采用 Vector/Cube 双路径混合执行方案，核心思路为：

1. 以固定 16 行为一个 window，每个 window 作为一个独立 task 进行划分；
2. preprocess 阶段遍历所有 window，根据 window 内 nnz 分布特征将其归入 mix 路径（Vector+Cube 混合异构）或 onlyVector 路径；
3. 设计纯 Vector 路径，直接由 Vector 核完成乘加；
4. 设计 `SpmvCsrInfo` / `SpmvCsrSubMatInfo` 元数据结构，用于描述 Vector/Cube 任务划分和多核负载均衡信息；
5. Cube 路径与 Vector 路径均使用 DoubleBuffer 掩盖数据搬运和计算开销；

## 背景介绍

CSR 格式使用三个数组描述稀疏矩阵：

```text
csrRowPtr : 每行非零元素起止位置，长度 M + 1
csrColInd : 非零元素列索引，长度 NNZ
csrVal    : 非零元素值，长度 NNZ
```

非转置计算为：

```text
for row in [0, M):
    acc = 0
    for p in [csrRowPtr[row], csrRowPtr[row + 1]):
        acc += csrVal[p] · X[csrColInd[p]]
    Y[row] = α · acc + β · Y[row]
```

从计算形式看，SpMV 的单次乘加并不复杂，主要难点来自 CSR 稀疏结构的不规则性和片上资源利用效率：

1. **行长度不均衡**：不同 row 的非零元数量差异较大，按行静态分配时容易出现部分核任务很少、部分核拖尾的问题；
2. **Cube padding 代价**：若直接将不规则 CSR row 整理为规则 Cube tile，短行或稀疏度很高的 window 会产生大量 padding，导致无效计算；
3. **片上缓存容量约束**：Cube 路径需要同时容纳 values、colIdx、gather 后的 X 片段和中间结果，需要根据 L1、L0A、L0B、L0C 容量保证单个 window 的数据不超过片上限制；
4. **转置路径转换开销**：`A^T · X` 场景可在 preprocess 阶段将 CSR 转换为 CSC，使 `A^T` 等价为按行连续访问的 CSR 结构；该方式能复用非转置 kernel，但需要额外考虑转换 workspace、转换耗时和预处理结果复用。

# 需求分析

## 需求描述

实现与 cuSPARSE SpMV 核心功能对齐的 Ascend C 算子，接口流程分为 workspace size 获取、preprocess 和 execute 三阶段，支持 CSR、转置/非转置、α/β、泛化 shape、50%~99.9% 稀疏度和任务书规定 dtype 组合。

## 需求拆解

1. 支持 CSR 输入，正确解析 `csrRowPtr`、`csrColInd`、`csrVal`。
2. 支持 `Y = α · A · X + β · Y` 和 `Y = α · A^T · X + β · Y`。
3. 支持如下数据类型组合：


| 输入 A、X 类型                 | computeType | 输出 Y 类型  |
| ------------------------- | ----------- | -------- |
| float32                   | float32     | float32  |
| int8                      | int32       | int32    |
| int8 / float16 / bfloat16 | float32     | float32  |
| float16                   | float32     | float16  |
| bfloat16                  | float32     | bfloat16 |


1. Cube 路径和 Vector 路径均使用 DoubleBuffer，使下一块数据搬运与当前块计算流水化。

# 详细设计

## 算子分析

### 数学公式

```text
Y = α · op(A) · X + β · Y
```

转置场景下，preprocess 阶段将 CSR 转换为 CSC。由于 CSC 的列指针、行索引和值数组可等价表示 `A^T` 的 CSR 行指针、列索引和值数组，因此 kernel 侧可以将转置后的矩阵视作普通 CSR 进行计算，复用非转置的 Vector/Cube 双路径流程，避免直接在 kernel 中进行按列 scatter 累加。

### 支持形状

设 CSR 矩阵形状为 `M x K`，非零元数量为 `NNZ`。


| op            | X 形状  | Y 形状  |
| ------------- | ----- | ----- |
| NON_TRANSPOSE | `[K]` | `[M]` |
| TRANSPOSE     | `[M]` | `[K]` |


约束：`csrRowPtr` 长度为 `M + 1`，`csrRowPtr[M] - idxBase == NNZ`，`csrColInd` 在合法列范围内，索引类型按 int32 设计。

### 数据类型策略


| 类型路径                                            | 计算策略                  |
| ----------------------------------------------- | --------------------- |
| float32 -> float32 -> float32                   | 直接以 float32 累加        |
| float16/bfloat16 -> float32 -> float32          | copy in 后转 float32 累加 |
| float16/bfloat16 -> float32 -> float16/bfloat16 | float32 累加，输出前转换      |
| int8 -> int32 -> int32                          | int8 乘法，int32 累加      |
| int8 -> float32 -> float32                      | int8 转 float32 后累加    |


## 接口设计

SpMV 算子对外提供三阶段调用接口，与 cuSPARSE 设计对齐：

#### 1. Handle 管理接口

```cpp
/**
 * @brief 创建稀疏矩阵处理器
 * @param handle [OUT] 返回创建的稀疏矩阵处理器句柄
 * @return AclSparseStatus 执行状态
 */
AclSparseStatus aclSparseCreate(AclSparseHandler *handle);

/**
 * @brief 销毁稀疏矩阵处理器
 * @param handle [IN] 稀疏矩阵处理器句柄
 * @return AclSparseStatus 执行状态
 */
AclSparseStatus aclSparseDestroy(AclSparseHandler handle);
```

#### 2. 稠密向量描述符接口

```cpp
/**
 * @brief 创建稠密向量描述符
 * @param dnVecDescr [OUT] 稠密向量描述符
 * @param size [IN] 向量元素个数
 * @param values [IN] 向量数据指针（Device 内存）
 * @param valueType [IN] 数据类型
 * @return AclSparseStatus 执行状态
 */
AclSparseStatus aclSparseCreateDnVec(AclSparseDnVecDesc *dnVecDescr, 
    int64_t size, void *values, aclDataType valueType);

/**
 * @brief 销毁稠密向量描述符
 * @param dnVecDescr [IN] 稠密向量描述符
 * @return AclSparseStatus 执行状态
 */
AclSparseStatus aclSparseDestroyDnVec(AclSparseDnVecDesc dnVecDescr);
```

#### 3. 稀疏矩阵描述符接口

```cpp
/**
 * @brief 创建 CSR 格式稀疏矩阵描述符
 * @param spMatDescr [OUT] 稀疏矩阵描述符
 * @param rows [IN] 矩阵行数
 * @param cols [IN] 矩阵列数
 * @param nnz [IN] 非零元素个数
 * @param csrRowOffsets [IN] 行偏移数组指针（Device 内存），长度 rows+1
 * @param csrColInd [IN] 列索引数组指针（Device 内存），长度 nnz
 * @param csrValues [IN] 非零元素值数组指针（Device 内存），长度 nnz
 * @param csrRowOffsetsType [IN] 行偏移数组索引类型
 * @param csrColIndType [IN] 列索引数组索引类型
 * @param idxBase [IN] 索引基值（0 或 1）
 * @param valueType [IN] 非零元素数据类型
 * @return AclSparseStatus 执行状态
 */
AclSparseStatus aclSparseCreateCsr(AclSparseSpMatDesc *spMatDescr, 
    int64_t rows, int64_t cols, int64_t nnz,
    void *csrRowOffsets, void *csrColInd, void *csrValues, 
    AclSparseIndexType csrRowOffsetsType, AclSparseIndexType csrColIndType, 
    AclSparseIndexBase idxBase, aclDataType valueType);

/**
 * @brief 创建 CSC 格式稀疏矩阵描述符
 * @param spMatDescr [OUT] 稀疏矩阵描述符
 * @param rows [IN] 矩阵行数
 * @param cols [IN] 矩阵列数
 * @param nnz [IN] 非零元素个数
 * @param cscColOffsets [IN] 列偏移数组指针（Device 内存），长度 cols+1
 * @param cscRowInd [IN] 行索引数组指针（Device 内存），长度 nnz
 * @param cscValues [IN] 非零元素值数组指针（Device 内存），长度 nnz
 * @param cscColOffsetsType [IN] 列偏移数组索引类型
 * @param cscRowIndType [IN] 行索引数组索引类型
 * @param idxBase [IN] 索引基值（0 或 1）
 * @param valueType [IN] 非零元素数据类型
 * @return AclSparseStatus 执行状态
 */
AclSparseStatus aclSparseCreateCsc(AclSparseSpMatDesc *spMatDescr, 
    int64_t rows, int64_t cols, int64_t nnz,
    void *cscColOffsets, void *cscRowInd, void *cscValues, 
    AclSparseIndexType cscColOffsetsType, AclSparseIndexType cscRowIndType, 
    AclSparseIndexBase idxBase, aclDataType valueType);

/**
 * @brief 销毁稀疏矩阵描述符
 * @param spMatDescr [IN] 稀疏矩阵描述符
 * @return AclSparseStatus 执行状态
 */
AclSparseStatus aclSparseDestroySpMat(AclSparseSpMatDesc spMatDescr);
```

#### 4. SpMV 核心接口（三阶段）

```cpp
/**
 * @brief 阶段1：获取 SpMV 所需 workspace 大小
 * @param handle [IN] 稀疏矩阵处理器句柄
 * @param op [IN] 矩阵操作类型（转置/非转置）
 * @param alpha [IN] 标量 α（Host 或 Device 内存）
 * @param mat [IN] 稀疏矩阵描述符
 * @param x [IN] 输入稠密向量描述符
 * @param beta [IN] 标量 β（Host 或 Device 内存）
 * @param y [IN] 输出稠密向量描述符
 * @param computeType [IN] 计算精度类型
 * @param alg [IN] SpMV 算法类型
 * @param size [OUT] 返回所需 workspace 字节数
 * @return AclSparseStatus 执行状态
 */
AclSparseStatus aclSparseSpmvGetBufferSize(AclSparseHandler handle, 
    AclSparseOp op, const void *alpha,
    AclSparseSpMatDesc mat, AclSparseDnVecDesc x, 
    const void *beta, AclSparseDnVecDesc y, 
    aclDataType computeType, AclSparseSpmvAlg alg, size_t *size);

/**
 * @brief 阶段2：SpMV 预处理
 * @param handle [IN] 稀疏矩阵处理器句柄
 * @param op [IN] 矩阵操作类型（转置/非转置）
 * @param alpha [IN] 标量 α
 * @param mat [IN] 稀疏矩阵描述符
 * @param x [IN] 输入稠密向量描述符
 * @param beta [IN] 标量 β
 * @param y [IN] 输出稠密向量描述符
 * @param computeType [IN] 计算精度类型
 * @param alg [IN] SpMV 算法类型
 * @param buffer [IN] workspace 缓冲区指针（Device 内存）
 * @return AclSparseStatus 执行状态
 *
 * @note 预处理阶段完成以下工作：
 *       - 若 op=TRANSPOSE，将 CSR 转换为 CSC
 *       - 按 window（16行×所有列）划分子矩阵
 *       - 统计每个 window 的 nnz 分布和稠密度
 *       - 将 window 划分到 mix 路径或 onlyVector 路径
 *       - 生成 cubeInfoIdx 和 vectorInfoIdx 多核调度表
 *       - 重排 Cube 路径的 values/colIdx 数据
 */
AclSparseStatus aclSparseSpmvPreprocess(AclSparseHandler handle, 
    AclSparseOp op, const void *alpha,
    AclSparseSpMatDesc mat, AclSparseDnVecDesc x, 
    const void *beta, AclSparseDnVecDesc y, 
    aclDataType computeType, AclSparseSpmvAlg alg, void *buffer);

/**
 * @brief 阶段3：执行 SpMV 计算
 * @param handle [IN] 稀疏矩阵处理器句柄
 * @param op [IN] 矩阵操作类型（转置/非转置）
 * @param alpha [IN] 标量 α
 * @param mat [IN] 稀疏矩阵描述符
 * @param x [IN] 输入稠密向量描述符
 * @param beta [IN] 标量 β
 * @param y [IN/OUT] 输出稠密向量描述符
 * @param computeType [IN] 计算精度类型
 * @param alg [IN] SpMV 算法类型
 * @param buffer [IN] workspace 缓冲区指针（Device 内存）
 * @return AclSparseStatus 执行状态
 *
 * @note 若未执行 preprocess，本接口会自动调用 preprocess
 *       计算公式：Y = α · op(A) · X + β · Y
 */
AclSparseStatus aclSparseSpmv(AclSparseHandler handle, 
    AclSparseOp op, const void *alpha, 
    AclSparseSpMatDesc mat, AclSparseDnVecDesc x, 
    const void *beta, AclSparseDnVecDesc y, 
    aclDataType computeType, AclSparseSpmvAlg alg, void *buffer);
```

## 算子实现

### 总体方案

```text
aclSparseSpmvGetBufferSize
  -> 参数检查与 workspace 估算
aclSparseSpmvPreprocess
  -> 若为转置场景，将 CSR 转换为 CSC，并将 CSC 作为 A^T 的 CSR 等价结构
  -> 按固定 16 行将矩阵划分为 window，每个 window 作为一个 task
  -> 统计每个 window 的 nnz 分布特征
  -> 根据 nnz 分布将 window 归入 mix 路径或 onlyVector 路径
  -> mix 路径 window 生成重排 values/colIdx
  -> 生成 cubeInfoIdx 和 vectorInfoIdx 多核调度表
aclSparseSpmv
  -> 读取 preprocess 元数据
  -> Vector 核根据 vectorInfoIdx 执行 onlyVector window
  -> AIV/AIC 协同根据 cubeInfoIdx 执行 mix window
  -> α/β 后处理并写回 Y
```

### host 侧设计

#### 1. 参数检查

检查 `handle`、`mat`、`x`、`y`、`α`、`β`、`buffer`、`size` 非空；检查 CSR/CSC 格式、op、shape、idxBase、index dtype、value dtype、computeType 合法；workspace size 必须满足 preprocess 元数据、重排数据和 swap buffer 需求。

#### 2. 转置预处理策略

当 `op = TRANSPOSE` 时，host 侧在 preprocess 阶段将输入 CSR 转换为 CSC。对于原矩阵 `A`，其 CSC 表示可等价看作 `A^T` 的 CSR 表示：

```text
A 的 cscColPtr  -> A^T 的 csrRowPtr
A 的 cscRowInd  -> A^T 的 csrColInd
A 的 cscVal     -> A^T 的 csrVal
```

完成转换后，kernel 不需要单独实现转置 scatter 路径，而是直接将转换后的 CSC 元数据作为 CSR 输入，复用非转置场景下的 Vector/Cube 双路径计算。这样可以避免多个输入 row 同时写同一个 `Y[col]` 所带来的 atomic 冲突。CSR 到 CSC 的转换结果保存在 workspace 中，并与当前矩阵结构绑定；如果矩阵结构变化，需要重新执行 preprocess。

#### 3. workspace 规划

```text
workspace =
    GM_SYNC_SIZE
  + sizeof(SpmvCsrInfo)
  + windowNum * sizeof(SpmvCsrSubMatInfo)
  + mixInfoIdxWorkspace
  + onlyVInfoIdxWorkspace
  + cubeInfoIdxWorkspace
  + vectorInfoIdxWorkspace
  + cubeReorderWorkspace
  + swapWorkspace
  + alignPadding
```

所有区域按 64B 对齐。`mixInfoIdx`、`onlyVInfoIdx`、`cubeInfoIdx`、`vectorInfoIdx` 只保存 window 索引，不直接保存矩阵数据；kernel 根据索引定位对应的 `SpmvCsrSubMatInfo`，再通过其中的 `blockPtr`、`colIdx`、`values` 访问实际数据。`swapWorkspace` 按 DoubleBuffer 和最大 window 数据量规划，供 AIV 与 AIC 交换 gather 后的 X 和 Cube 结果。

#### 4. 数据结构设计

子矩阵信息结构：

```cpp
typedef struct {
     uint64_t blockNum;
     uint64_t startCol;
     uint64_t colNum;
     uint64_t nnz;
     uint64_t maxBlockSize;
     uint64_t rowBlockLen;
     uint64_t colIdxLen;
     uint64_t valueLen;
     uint64_t workspaceLen;
     uint64_t pading[4];
     __gm__ uint32_t *blockPtr;
     __gm__ uint32_t *colIdx;
     __gm__ float *values;
} SpmvCsrSubMatInfo;
```

全局调度信息结构：

```cpp
typedef struct {
    uint64_t maxBlockSize;
    uint64_t mixWindowNum;
    uint64_t onlyVWindowNum;
    __gm__ uint8_t *swap;
    __gm__ uint32_t *ptrs;
    __gm__ uint32_t *idxs;
    __gm__ void *values;
    __gm__ uint32_t *mixInfoIdx;
    __gm__ uint32_t *onlyVInfoIdx;
    __gm__ uint32_t *cubeInfoIdx;
    __gm__ uint32_t *vectorInfoIdx;
    SpmvCsrSubMatInfoExt infos[0];
} SpmvCsrInfo;
```

四个调度指针含义：


| 指针              | 说明                                                        |
| --------------- | --------------------------------------------------------- |
| `mixInfoIdx`    | 属于 mix 路径（Vector+Cube 混合）的 window 索引列表，长度为 `mixWindowNum` |
| `onlyVInfoIdx`  | 属于 onlyVector 路径的 window 索引列表，长度为 `onlyVWindowNum`        |
| `cubeInfoIdx`   | 每个 Cube 核负责计算的 window 索引表，kernel 根据核编号读取本核负责的 window 列表   |
| `vectorInfoIdx` | 每个 Vector 核负责计算的 window 索引表，kernel 根据核编号读取本核负责的 window 列表 |


#### 5. window 划分与路径选择策略

**window 定义**：以固定 16 行（`MAX_SUB_ROW_SIZE = 16`）为一个 window，矩阵共划分为 `windowNum = (rows + 15) / 16` 个 window。

**路径选择规则**：

对每个 window 统计其内 16 行的 nnz 分布特征，根据以下规则决定归入 mix 路径或 onlyVector 路径：

```text
if window_nnz == 0:
    skip (不分配任何路径)
else if window 内行 nnz 分布相对均匀且适合 padding:
    归入 mix 路径 -> mixInfoIdx
else:
    归入 onlyVector 路径 -> onlyVInfoIdx
```

具体判断策略：统计 window 内 16 行的最大 nnz（`maxNnz`）和平均 nnz（`avgNnz`），计算 padding 比例 `padding_ratio = (16 * maxNnz - total_nnz) / (16 * maxNnz)`。若 `padding_ratio < PADDING_THRESHOLD`（建议 0.5~0.7），则归入 mix 路径，否则归入 onlyVector 路径。

### kernel 侧设计

#### 1. Cube 路径

Cube 路径在 preprocess 阶段已经完成 window 内数据 padding、重排和调度表生成。kernel 侧 Cube 核根据 `cubeInfoIdx` 中的 window 索引定位 `SpmvCsrSubMatInfo`，再通过其中的 `blockPtr`、`colIdx`、`values` 访问已经重排好的数据并执行计算。

该路径处理 nnz 分布相对均匀、padding 比例较低的 window。nnz 分布稀疏或行间差异过大的 window 在 preprocess 阶段已经归入 onlyVector 路径。

AIV 和 AIC 协同执行：

```text
AIV:
  CopyIn colIdx
  Gather X[colIdx] 到 swapAivToAic
  SetFlag AIV_TO_AIC
AIC:
  WaitFlag AIV_TO_AIC
  CopyIn values 和 gathered X 到 L1/L0
  Mmad 计算当前 window 结果
  CopyOut partial 到 swapAicToAiv
  SetFlag AIC_TO_AIV
AIV post:
  WaitFlag AIC_TO_AIV
  提取每行结果
  执行 α/β
  写回 Y
```

DoubleBuffer 使用两个 buffer 交替处理：

```text
for window in cubeWindows:
    buf = window % 2
    预取当前 window 数据到 buffer[buf]
    计算上一 window 的 buffer[1 - buf]
    写回已完成结果
```

#### 2. Vector 路径

Vector 核根据 `vectorInfoIdx` 读取本核负责的 window 索引，再通过 `infos[windowIdx]` 定位 `SpmvCsrSubMatInfo`，直接遍历 window 内 16 行的 CSR 数据完成乘加：

```text
for windowIdx in vectorInfoIdx[coreId]:
    subInfo = infos[windowIdx]
    rowStart = windowIdx * MAX_SUB_ROW_SIZE
    rowEnd   = min(rowStart + MAX_SUB_ROW_SIZE, rows)
    for row in [rowStart, rowEnd):
        acc = 0
        for p in [csrRowPtr[row], csrRowPtr[row + 1]):
            acc += csrVal[p] · X[csrColInd[p]]
        Y[row] = α · acc + β · Y[row]
```

优化点：`β=0`、`β=1`、`α=1` 使用专用分支减少计算；转置场景通过 preprocess 生成 CSC，将 CSC 等价作为 `A^T` 的 CSR 输入，复用同一套 Vector 计算路径，无需 kernel 侧额外处理。

### 多核负载均衡设计

#### 1. Cube-Cube 均衡

将 `mixInfoIdx` 中的 window 按轮询或负载估算方式分配到各 Cube 核，写入 `cubeInfoIdx`。可采用简单轮询策略，将 window 顺序分配到各核；也可根据每个 window 的 nnz 总量估算计算代价，采用贪心策略将 window 分配到当前负载最轻的核。

#### 2. Vector-Vector 均衡

将 `onlyVInfoIdx` 中的 window 采用与 Cube-Cube 相同的策略分配到各 Vector 核，写入 `vectorInfoIdx`。可采用轮询或基于 nnz 总量的贪心分配。

### tiling key 规划

通过 tiling key 区分 `opType`、`dtypePath`、`computeType`、`scaleMode` 和 `pathMode`。


| 字段          | 说明                                            |
| ----------- | --------------------------------------------- |
| `opType`    | 非转置 / 转置                                      |
| `dtypePath` | f32、fp16->f32、bf16->f32、int8->int32、int8->f32 |
| `scaleMode` | β=0、α=1/β=1、一般 α/β                            |
| `pathMode`  | Vector only、Vector + Cube mixed               |


## 支持硬件


| 支持的芯片版本      | 涉及勾选 |
| ------------ | ---- |
| Ascend 950PR | √    |


## 算子约束限制

1. CSR 行偏移和列索引按 int32 设计；
2. task 划分以固定 16 行为一个 window，最后一个 window 的实际行数可能不足 16 行，kernel 需处理尾行边界；
3. Cube 路径 window 内数据需满足 L0A、L0B、L1、L0C 容量约束，不满足时该 window 归入 onlyVector 路径；
4. 转置场景需要在 preprocess 阶段将 CSR 转换为 CSC，转换后的 CSC 元数据与原矩阵结构绑定；
5. workspace 元数据与当前矩阵结构绑定，矩阵结构变化后需重新 preprocess；
6. 外部 API 保持三阶段调用方式不变。

# 可维可测分析

## 精度标准/性能标准

### 精度标准

采用生态算子开源精度标准，以更高精度的 CPU 计算结果作为参考真值（golden），对 NPU 实际输出（actual）进行误差评估。

**误差指标定义**：

平均相对误差（Mean Relative Error，MERE）：采样点中相对误差平均值。

$$\text{MERE} = \text{avg}\left( \frac{\text{abs}(actual - golden)}{\text{abs}(golden) + 1\text{e-}7} \right)$$

最大相对误差（Max Relative Error，MARE）：采样点中相对误差最大值。

$$\text{MARE} = \max\left(\frac{\text{abs}(\text{actual} - \text{golden})}{\text{abs}(\text{golden}) + 10^{-7}}\right)$$


**通过标准**：MERE < Threshold 且 MARE < 10 × Threshold

**各数据类型阈值**：

| 数据类型 | 阈值 (Threshold) |
| --- | --- |
| FLOAT32 | \(2^{-13}\) ≈ 1.22e-4 |
| FLOAT16 | \(2^{-10}\) ≈ 9.77e-4 |
| BFLOAT16 | \(2^{-7}\) ≈ 7.81e-3 |

**int8 → int32 路径**：累加结果为整数，直接比较 NPU 输出与 CPU golden 是否完全一致（bit-wise exact）。

### 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能标准 | 覆盖 CSR、转置/非转置、α/β、全 dtype、边界 shape | 任务书 |
| 精度标准 | 满足上述生态算子开源精度标准 | 任务书 |
| 性能标准 | 目标达到 0.5 倍 A100 参考性能 | 任务书 |


## 测试设计

功能测试覆盖：小/中/大矩阵，稀疏度 50%、90%、95%、97.5%、99%、99.9%，均匀行、短行占优、长行占优、极端不均匀行，转置/非转置，α/β 特殊值和一般值，全 dtype 组合，NNZ=0、空行、单行、单列、尾行不足 16 行、最后一个 window 边界等场景。

精度测试使用 CPU golden 对比，float 路径比较绝对误差和相对误差，int8->int32 路径比较整数累加结果。

性能测试重点统计 mix window 与 onlyVector window 占比、mix window 的 padding ratio、各核 window 分配数量、DoubleBuffer 开启前后流水效果，以及 CSR 转 CSC 预处理开销。

## 兼容性分析

本方案采用 `aclSparseSpmvGetBufferSize`、`aclSparseSpmvPreprocess`、`aclSparseSpmv` 三阶段对外调用方式。执行模式包含 Vector only 和 Vector/Cube mixed 两类，通过 workspace version、tiling key 和 pathMode 区分。元数据结构仅服务于内部 preprocess 和 kernel 解析逻辑，用户侧只需要按照三阶段接口完成 workspace 查询、预处理和算子执行。
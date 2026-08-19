# SpSM 算子设计文档 V5

# 需求来源

社区任务要求参考 NVIDIA cuSPARSE `cusparseSpSM`，在 Ascend 950PR 上基于 Ascend C 实现功能一致的 SpSM（Sparse triangular Solve with Multiple right-hand sides）算子，并完成设计、开发、测试和验收交付。算子求解公式为：

$$
op(A) \cdot C = \alpha \cdot B \quad\Rightarrow\quad C = \alpha \cdot op(A)^{-1} \cdot B
$$

其中 `A` 为 CSR 稀疏三角方阵，`B` 为稠密多右端输入矩阵，`C` 为稠密解矩阵，`op(A)` 支持非转置和转置，`alpha` 为全局缩放标量。cuSPARSE SpSM 核心流程为 **bufferSize 预备 → analysis → solve 三步骤**，另含专有描述符 `spSMDescr` 管理 analysis 缓存与 `updateMatrix` 可选数值更新路径。

## 背景介绍

### CSR 稀疏存储格式

CSR 格式使用三个数组描述稀疏矩阵，索引类型为 int32（`ACL_SPARSE_INDEX_32I`）：

- `csrRowOffsets`：每行非零元素起止位置，长度 M + 1
- `csrColInd`：非零元素列索引，长度 NNZ
- `csrValues`：非零元素值，长度 NNZ

### 三角求解计算模式

**下三角前代（Forward Substitution）** 求解 `L · X = B`：

```
for i in [0, M):
    acc = 0
    for p in [csrRowOffsets[i], csrRowOffsets[i + 1]):
        j = csrColInd[p]
        if j < i:
            acc += csrValues[p] * C[j, :]
        if j == i:
            diag = csrValues[p]
    C[i, :] = (alpha * B[i, :] - acc) / diag       // diag_type = NON_UNIT
    C[i, :] = alpha * B[i, :] - acc                 // diag_type = UNIT
```

**上三角回代（Backward Substitution）** 求解 `U · X = B`：

```
for i in [M-1, 0]:
    acc = 0
    for p in [csrRowOffsets[i], csrRowOffsets[i + 1]):
        j = csrColInd[p]
        if j > i:
            acc += csrValues[p] * C[j, :]
        if j == i:
            diag = csrValues[p]
    C[i, :] = (alpha * B[i, :] - acc) / diag       // diag_type = NON_UNIT
    C[i, :] = alpha * B[i, :] - acc                 // diag_type = UNIT
```

**转置路径（opA = T）**：行列索引交换访问方向。CSR 存储的 A 在转置后等价于按列访问，依赖方向反转（下三角转置等价于上三角、上三角转置等价于下三角）。

### SpSM 核心难点

1. **行间依赖约束**：下三角第 i 行依赖所有 j < i 且 A[i,j] ≠ 0 的已求解行；上三角同理。不能简单按行号范围均分到多核并行
2. **依赖图不规则性**：稀疏三角矩阵的依赖图结构取决于非零元分布，不同矩阵的并行度（maxLevel）差异很大
3. **UB 容量与多 RHS**：多 RHS 场景下 UB 需同时容纳 B tile 和 C 依赖行缓存，大 nrhs 需要分块处理
4. **analysis 缓存复用**：同一稀疏结构、不同数值的多次求解应复用 analysis 结果，通过 `updateMatrix` 仅更新数值时跳过 re-analysis
5. **转置路径的正确性**：A 转置下行列索引交换，依赖方向反转，须在 Kernel 中根据 opA 参数正确处理
6. **bit-wise 确定性**：solve 阶段相同输入须产生完全一致的输出（cuSPARSE 语义要求）

### ops-sparse 仓库上下文

ops-sparse 是 CANN 算子库中提供高性能稀疏矩阵计算的算子库（版本 1.0.0，要求 CANN 9.0.0+）。当前已实现算子：SpMV（arch22）、SpMM（arch35）、Snnz（arch35, Legacy API）。SpSM 为仓内全新算子（`src/spsm/` 当前不存在），须从零交付。

---

# 需求分析

## 需求描述

实现与 cuSPARSE §6.6.12 SpSM 核心功能对齐的 Ascend C 算子。接口流程为：

1. **CreateDescr / DestroyDescr** — SpSM 专有描述符生命周期管理
2. **bufferSize** — workspace 大小预备（matB/matC 的 values 可为 NULL）
3. **analysis** — 符号分析（依赖图构建 → Level Scheduling → 分核策略），结果缓存于 spsmDescr
4. **solve** — 数值求解（基于 analysis 缓存，level 间 barrier 同步，支持 in-place）
5. **updateMatrix** — 数值更新，保持结构不变时跳过 re-analysis

**约束**：

- 稀疏格式仅支持 CSR，索引类型 int32（`ACL_SPARSE_INDEX_32I`）
- 索引基址仅支持 0-based（`ACL_SPARSE_INDEX_BASE_ZERO`）
- 数据类型首期仅支持 fp32 同精度（A/B/C/computeType/alpha 均为 `ACL_FLOAT`）
- `opA` 支持 N 和 T；`opB` 仅支持 N（与 cuSPARSE SpSM 一致）
- `fill_mode` 支持 LOWER 和 UPPER；`diag_type` 支持 NON_UNIT 和 UNIT
- 稠密矩阵仅支持列主序（`ACL_SPARSE_ORDER_COL`）
- matA 须为方阵（M × M），M ≥ 1 且 M ≤ INT32_MAX
- bufferSize / analysis 阶段：matB/matC 的 values 可为 NULL，描述符不可为 NULL
- solve 阶段：matB/matC 须提供有效 values；支持 in-place（matB 与 matC 共用同一 values 指针）
- updateMatrix 首期仅支持 `ACL_SPARSE_SPSM_UPDATE_GENERAL`（全量数值更新）

## 需求拆解

| 编号 | 需求项 | 说明 |
|------|--------|------|
| REQ-01 | CSR 格式解析 | 正确解析 csrRowOffsets、csrColInd、csrValues |
| REQ-02 | fill_mode / diag_type | 下三角/上三角 与 显式对角/单位对角，4 种组合 |
| REQ-03 | opA 转置 | opA = N / T，转置时行列索引交换 |
| REQ-04 | 标量 alpha | 全局缩放标量，float32 |
| REQ-05 | 多 RHS | nrhs ≥ 1，列主序稠密矩阵 |
| REQ-06 | in-place | matB 与 matC 可共用同一 values 指针 |
| REQ-07 | NULL values 语义 | bufferSize/analysis 阶段 matB/matC values 可为 NULL |
| REQ-08 | analysis 缓存 | 稀疏结构不变时跳过重复依赖分析 |
| REQ-09 | updateMatrix | 仅更新数值时复用 analysis 缓存 |
| REQ-10 | bit-wise 确定性 | solve 阶段同输入须完全一致输出 |
| REQ-11 | FP32 同精度 | A/B/C/computeType/alpha 均为 float32 |

---

# ops-sparse 仓库限制与约束

以下约束提取自 ops-sparse 仓库现有代码规范（`agent/skills/`），在 SpSM 开发中必须遵守。

## 接口命名规范

ops-sparse 严格对齐 cuSPARSE，使用 Generic API 体系：

| 维度 | 规范 | SpSM 应用 |
|------|------|-----------|
| 库前缀 | `aclsparse`，小写，无空格 | `aclsparseSpSM` |
| 描述符管理 | `aclsparseCreate/Destroy{Type}` | `aclsparseSpSMCreateDescr` / `aclsparseSpSMDestroyDescr` |
| 矩阵运算 | `aclsparseSp{MM/SM/GEMM}` | `aclsparseSpSMAnalysis` / `aclsparseSpSMSolve` |
| 目录名 | snake_case | `spsm`（对齐任务书） |
| 文件名 | snake_case | `spsm_host.cpp` / `spsm_kernel.cpp` |

禁止项：下划线分隔命名（如 `aclsparse_sp_sm`）、全大写（如 `aclsparseSPSM`）。

## 参数顺序规范

```c
// 接口参数统一顺序（对齐 cuSPARSE Generic API）
aclsparseSpSM*(...):
    1. aclsparseHandle_t handle                // Handle 永远第一
    2. aclsparseOperation_t opA                // A 的转置操作
    3. aclsparseOperation_t opB                // B 的转置操作
    4. const void *alpha                       // 标量（const void*）
    5. aclsparseConstSpMatDescr_t matA         // 稀疏矩阵 read-only
    6. aclsparseConstDnMatDescr_t matB         // 稠密输入 read-only
    7. aclsparseDnMatDescr_t matC              // 稠密输出
    8. aclDataType computeType                 // 计算精度
    9. aclsparseSpSMAlg_t alg                  // 算法选择
    10. aclsparseSpSMDescr_t spsmDescr         // SpSM 专有描述符
    11. size_t *bufferSize / void *buffer      // workspace（bufferSize/analysis）
```

## 描述符管理规范

- Create 时 `*descr` 必须为 nullptr（防止内存泄漏）
- Destroy 必须安全处理 nullptr 输入（直接返回 SUCCESS）
- `const_cast` 集中在 `ToMatInner` / `ToVecInner` 等统一转换函数中，禁止在业务代码中直接 cast
- 所有 `new` 必须使用 `std::nothrow`
- Create/Destroy 必须配对调用

## 状态码使用规范

| 场景 | 状态码 |
|------|--------|
| 正常完成 | `ACL_SPARSE_STATUS_SUCCESS` |
| handle 为 nullptr | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| 描述符/指针为 nullptr | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 维度为负数/不合法 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 格式不支持 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| 矩阵类型不支持 | `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED` |
| 数据类型组合不支持 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| 算法不支持 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| 芯片不支持 | `ACL_SPARSE_STATUS_ARCH_MISMATCH` |
| UB 容量不足 | `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` |
| Kernel 执行失败 | `ACL_SPARSE_STATUS_EXECUTION_FAILED` |
| 内部错误 | `ACL_SPARSE_STATUS_INTERNAL_ERROR` |
| 内存分配失败 | `ACL_SPARSE_STATUS_ALLOC_FAILED` |

## Handle 管理

```c
aclsparseStatus_t aclsparseCreate(aclsparseHandle_t *handle);
aclsparseStatus_t aclsparseDestroy(aclsparseHandle_t handle);
aclsparseStatus_t aclsparseSetStream(aclsparseHandle_t handle, aclrtStream stream);
aclsparseStatus_t aclsparseGetStream(aclsparseHandle_t handle, aclrtStream *stream);
```

- Create 调用前 `*handle` 必须为 nullptr
- Stream 由用户托管，Destroy 不释放 stream

## 新增枚举类型

需在 `include/cann_ops_sparse.h` 中声明以下新枚举：

```c
// SpSM 算法选择枚举
typedef enum aclsparseSpSMAlg_t {
    ACL_SPARSE_SPSM_ALG_DEFAULT = 0
} aclsparseSpSMAlg_t;

// SpSM 更新策略枚举
typedef enum aclsparseSpSMUpdate_t {
    ACL_SPARSE_SPSM_UPDATE_GENERAL = 0    // 全量数值更新（首期唯一支持）
} aclsparseSpSMUpdate_t;

// SpSM 三角填充模式枚举
typedef enum aclsparseFillMode_t {
    ACL_SPARSE_FILL_MODE_LOWER = 0,       // 下三角
    ACL_SPARSE_FILL_MODE_UPPER = 1        // 上三角
} aclsparseFillMode_t;

// SpSM 对角类型枚举
typedef enum aclsparseDiagType_t {
    ACL_SPARSE_DIAG_TYPE_NON_UNIT = 0,    // 显式对角元
    ACL_SPARSE_DIAG_TYPE_UNIT = 1         // 单位对角元
} aclsparseDiagType_t;
```

---

# 详细设计

## 算子分析

### 数学公式

```
op(A) · C = alpha · B
```

### 支持形状

| 矩阵 | 形状 | 排布 | 约束 |
| ---- | ---- | ---- | ---- |
| A（稀疏三角） | [M, M] | CSR | M ≥ 1，NNZ ≥ M（每行至少含对角元，unit_diag 除外） |
| B（稠密输入） | [M, nrhs] | 列主序 | nrhs ≥ 1 |
| C（稠密输出） | [M, nrhs] | 列主序 | 与 B 同形状 |

约束：`csrRowOffsets` 长度为 `M + 1`，`csrRowOffsets[M] == NNZ`，`csrColInd` 在合法范围内，索引类型 int32。

### 单次求解公式（列主序）

**下三角前代**（fill_mode = LOWER, opA = N）：

```
C[i, k] = (alpha * B[i, k] - sum_{j < i, A[i,j] != 0} A[i,j] * C[j, k]) / A[i,i]    // NON_UNIT
C[i, k] = alpha * B[i, k] - sum_{j < i, A[i,j] != 0} A[i,j] * C[j, k]                 // UNIT
```

**上三角回代**（fill_mode = UPPER, opA = N）：

```
C[i, k] = (alpha * B[i, k] - sum_{j > i, A[i,j] != 0} A[i,j] * C[j, k]) / A[i,i]    // NON_UNIT
C[i, k] = alpha * B[i, k] - sum_{j > i, A[i,j] != 0} A[i,j] * C[j, k]                 // UNIT
```

### 数据类型策略

首期仅支持 fp32 同精度（A / B / C / computeType / alpha 均为 `ACL_FLOAT`），计算策略为直接 float32 累加。后续可扩展 fp64、fp16、complex 等类型路径。

## 接口设计

### SpSM 专有描述符

```c
// SpSM 专有描述符句柄（opaque pointer）
typedef struct aclsparseSpSMDescr *aclsparseSpSMDescr_t;
```

`aclsparseSpSMDescr_t` 是 SpSM 的核心设计点，其职责：

- 缓存 analysis 阶段的符号分析结果（依赖 DAG、level 分组、reorder 表、group_edges 表）
- 记录当前 active buffer 指针，实现 analysis 到 solve 的缓存复用
- 记录 `analysisDone` 标记，solve 阶段根据该标记判断是否需要自动触发 analysis
- 存储 fill_mode、diag_type、opA 等三角矩阵属性（用于 TilingKey 决策）

三角属性（fill_mode / diag_type）和依赖图分析结果均独立存储在 `spSMDescr` 中，`matA` 保持为只读的通用稀疏矩阵描述符，解除三角专属属性与通用稀疏矩阵描述符的耦合。

### 六接口体系

#### 2.1 描述符生命周期

```c
/**
 * @brief 创建 SpSM 专有描述符
 * @param spsmDescr [OUT] 返回创建的 SpSM 描述符句柄。调用前 *spsmDescr 必须为 nullptr
 * @return ACL_SPARSE_STATUS_SUCCESS: 成功
 *         ACL_SPARSE_STATUS_INVALID_VALUE: spsmDescr 为 nullptr
 *         ACL_SPARSE_STATUS_ALLOC_FAILED: 内存分配失败
 */
aclsparseStatus_t aclsparseSpSMCreateDescr(aclsparseSpSMDescr_t *spsmDescr);

/**
 * @brief 销毁 SpSM 专有描述符
 * @param spsmDescr [IN] SpSM 描述符句柄。传入 nullptr 时直接返回 SUCCESS
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSpSMDestroyDescr(aclsparseSpSMDescr_t spsmDescr);
```

#### 2.2 三段式核心接口

```c
/**
 * @brief 阶段1：计算 analysis / solve 所需 workspace 字节数
 *
 * @note matB / matC 的 values 可为 NULL（仅描述符本身不可为 NULL）
 */
aclsparseStatus_t aclsparseSpSMGetBufferSize(
    aclsparseHandle_t            handle,
    aclsparseOperation_t         opA,
    aclsparseOperation_t         opB,
    const void                  *alpha,
    aclsparseConstSpMatDescr_t   matA,
    aclsparseConstDnMatDescr_t   matB,
    aclsparseDnMatDescr_t        matC,
    aclDataType                  computeType,
    aclsparseSpSMAlg_t           alg,
    aclsparseSpSMDescr_t         spsmDescr,
    size_t                      *bufferSize);

/**
 * @brief 阶段2：执行符号分析
 *
 * @note  完成依赖图构建、Level Scheduling、Greedy Bin-Packing，
 *        生成分核策略并写入 workspace。分析结果缓存在 spsmDescr 中
 * @note  matB / matC 的 values 可为 NULL（仅描述符本身不可为 NULL）
 * @note  analysis 结果与 fill_mode / diag_type / opA 绑定
 */
aclsparseStatus_t aclsparseSpSMAnalysis(
    aclsparseHandle_t            handle,
    aclsparseOperation_t         opA,
    aclsparseOperation_t         opB,
    const void                  *alpha,
    aclsparseConstSpMatDescr_t   matA,
    aclsparseConstDnMatDescr_t   matB,
    aclsparseDnMatDescr_t        matC,
    aclDataType                  computeType,
    aclsparseSpSMAlg_t           alg,
    aclsparseSpSMDescr_t         spsmDescr,
    void                        *externalBuffer);

/**
 * @brief 阶段3：执行数值求解（bit-wise 确定性）
 *
 * @note  基于 spsmDescr 中缓存的 analysis 结果，运行 level-scheduled 三角回代 Kernel
 * @note  若未显式执行 analysis，本接口自动触发 analysis
 * @note  支持 in-place：matB 与 matC 可使用同一 values 指针
 * @note  matB / matC values 必须为有效指针
 * @note  异步执行，通过 handle 关联的 stream 调度
 */
aclsparseStatus_t aclsparseSpSMSolve(
    aclsparseHandle_t            handle,
    aclsparseOperation_t         opA,
    aclsparseOperation_t         opB,
    const void                  *alpha,
    aclsparseConstSpMatDescr_t   matA,
    aclsparseConstDnMatDescr_t   matB,
    aclsparseDnMatDescr_t        matC,
    aclDataType                  computeType,
    aclsparseSpSMAlg_t           alg,
    aclsparseSpSMDescr_t         spsmDescr);
```

#### 2.3 updateMatrix 接口

```c
/**
 * @brief 更新稀疏矩阵数值，保持结构不变时可跳过 re-analysis
 * @param handle      [IN]  稀疏算子库上下文句柄
 * @param spsmDescr   [IN]  SpSM 描述符（须已完成 analysis）
 * @param newValues   [IN]  新的非零元数值数组指针（Device 内存）
 * @param updatePart  [IN]  更新策略（首期仅 ACL_SPARSE_SPSM_UPDATE_GENERAL）
 * @return aclsparseStatus_t 执行状态
 *
 * @note 仅替换 matA 的 csrValues 指针，稀疏结构（csrRowOffsets / csrColInd）不变
 */
aclsparseStatus_t aclsparseSpSMUpdateMatrix(
    aclsparseHandle_t            handle,
    aclsparseSpSMDescr_t         spsmDescr,
    void                        *newValues,
    aclsparseSpSMUpdate_t        updatePart);
```

### 接口约束矩阵

| 约束条件 | GetBufferSize | Analysis | Solve | UpdateMatrix |
| -------- | :-----------: | :------: | :---: | :----------: |
| handle 不可为 nullptr | 是 | 是 | 是 | 是 |
| matA 描述符不可为 nullptr | 是 | 是 | 是 | -- |
| matB 描述符不可为 nullptr | 是 | 是 | 是 | -- |
| matC 描述符不可为 nullptr | 是 | 是 | 是 | -- |
| spsmDescr 不可为 nullptr | 是 | 是 | 是 | 是 |
| matB values 不可为 nullptr | -- | -- | 是 | -- |
| matC values 不可为 nullptr | -- | -- | 是 | -- |
| spsmDescr→analysisDone 为 true | -- | -- | 自动触发 | 强制 |
| matA 结构不可变 | 是 | 是 | 是 | 是（仅数值变） |
| bit-wise 确定性 | -- | -- | 是 | -- |
| opB 为 N | 是 | 是 | 是 | -- |

### 调用流程

```
1. aclsparseCreate(&handle)
2. aclsparseSetStream(handle, stream)
3. aclsparseCreateCsr(&matA, ...)              // CSR 三角矩阵
4. aclsparseCreateDnMat(&matB, ..., dB)         // 稠密右端项 B
5. aclsparseCreateDnMat(&matC, ..., dC)         // 稠密解 C
6. aclsparseSpSMCreateDescr(&spsmDescr)

   // ---- 阶段1：查询 workspace ----
7. aclsparseSpSMGetBufferSize(handle, ..., spsmDescr, &bufferSize)

   // ---- 阶段2：符号分析 ----
8. aclrtMalloc(&dBuffer, bufferSize, ...)
9. aclsparseSpSMAnalysis(handle, ..., spsmDescr, dBuffer)

   // ---- 阶段3：数值求解（可多次调用，复用 analysis 缓存）----
10. aclsparseSpSMSolve(handle, ..., spsmDescr)

   // ---- 可选：更新数值后 re-solve ----
11. aclsparseSpSMUpdateMatrix(handle, spsmDescr, dNewValues, UPDATE_GENERAL)
12. aclsparseSpSMSolve(handle, ..., spsmDescr)   // 跳过 re-analysis

   // ---- 清理 ----
13. aclsparseSpSMDestroyDescr(spsmDescr)
14. aclsparseDestroySpMat(matA)
15. aclsparseDestroyDnMat(matB)  /  aclsparseDestroyDnMat(matC)
16. aclsparseDestroy(handle)
17. aclrtFree(dBuffer)
```

## 算子实现

### 总体方案

```
aclsparseSpSMCreateDescr
  -> new (std::nothrow) SpsmDescrInternal，初始化 analysisDone = false

aclsparseSpSMGetBufferSize
  -> 参数校验：handle / matA / matB / matC / spsmDescr / bufferSize 非空
  -> 校验 opA / opB、CSR 格式、shape、idxBase、index dtype、value dtype、computeType、alg
  -> 最坏情况 workspace = header + TilingData + reorder[M] + group_edges[blockDim x (M+1) + 1]
  -> 输出 bufferSize

aclsparseSpSMAnalysis
  -> 完整参数校验（同 bufferSize）
  -> D2H 拷贝 csrRowOffsets / csrColInd
  -> 构建依赖 DAG（下三角：边 u->v 当 A[v,u]!=0 且 u<v；上三角：边 u->v 当 A[v,u]!=0 且 u>v）
  -> 拓扑分层（Level Scheduling）
  -> 每层内 Greedy Bin-Packing 分配行到各 Core
  -> 生成 reorder 表和 group_edges 表
  -> H2D 写入 workspace
  -> 填充 SpsmTilingData 并写入 device workspace
  -> 标记 spsmDescr->analysisDone = true, spsmDescr->activeBuffer = externalBuffer
  -> 缓存 fill_mode / diag_type / opA 到 spsmDescr

aclsparseSpSMSolve
  -> 检查 spsmDescr->analysisDone；若未完成则自动调用 analysis
  -> 刷新 SpsmTilingData 中动态参数（alpha 可能变化）
  -> Launch Kernel: SPMD 多 Core 并行
  -> Kernel 内按 level 顺序执行，level 间 barrier 同步
  -> level 内各 Core 处理分配的行，核内 SIMT 线程并行处理 RHS 列

aclsparseSpSMUpdateMatrix
  -> 校验 spsmDescr->analysisDone 为 true
  -> 替换 matA->values 指针为新数值
  -> 保持 spsmDescr 中 analysis 缓存有效（不清除）
```

### Host 侧设计

#### 参数校验

Host 侧拆分两个静态函数：`ValidateSpsmParams`（参数校验）+ `SpsmDoAnalysis` / `LaunchSpsmKernel`（执行/启动），强制使用 dlog 日志（`OP_LOGE` / `OP_LOGD` / `OP_LOGI`），禁止 `printf` / `std::cout`。

每个参数独立校验，禁止批量校验。校验项清单：

| 校验项 | 失败状态码 |
|--------|-----------|
| handle 非空 | `HANDLE_IS_NULLPTR` |
| matA / matB / matC 描述符非空 | `INVALID_VALUE` |
| matB / matC 的 values 非空（solve 阶段） | `INVALID_VALUE` |
| spsmDescr 非空 | `INVALID_VALUE` |
| 稀疏格式为 CSR | `MATRIX_TYPE_NOT_SUPPORTED` |
| 索引基值为 0-based | `NOT_SUPPORTED` |
| 索引类型为 int32 | `NOT_SUPPORTED` |
| dtype 为 ACL_FLOAT | `NOT_SUPPORTED` |
| computeType 为 ACL_FLOAT | `NOT_SUPPORTED` |
| alg 为 ALG_DEFAULT | `NOT_SUPPORTED` |
| opA 为 N 或 T | `NOT_SUPPORTED` |
| opB 为 N | `NOT_SUPPORTED` |
| A 为方阵 | `INVALID_VALUE` |
| 维度一致性 | `INVALID_VALUE` |
| M ≥ 1, M ≤ INT32_MAX | `INVALID_VALUE` |
| fill_mode 为 LOWER 或 UPPER | `NOT_SUPPORTED` |
| diag_type 为 NON_UNIT 或 UNIT | `NOT_SUPPORTED` |
| order 为 COL | `NOT_SUPPORTED` |

#### 依赖分析与 Level Scheduling

analysis 阶段的核心任务是对稀疏三角矩阵构建有向依赖图 G = (V, E)：

- 顶点集 V = {0, 1, …, M-1} 代表矩阵的 M 行
- 下三角：边 u→v 存在当 A[v, u] ≠ 0 且 u < v（行 v 依赖行 u）
- 上三角：边 u→v 存在当 A[v, u] ≠ 0 且 u > v

对 DAG 执行拓扑排序分层：

```
for v = 0 .. M-1:
    indegree[v] = count({u | u->v ∈ E})
    if indegree[v] == 0:
        queue.push(v), level[v] = 0

maxLevel = 0
while queue not empty:
    u = queue.pop()
    for each v where u->v ∈ E:
        indegree[v]--
        level[v] = max(level[v], level[u] + 1)
        if indegree[v] == 0:
            queue.push(v)
            maxLevel = max(maxLevel, level[v])
```

同 level 内的行之间无相互依赖，可完全并行求解。Level 间按序串行执行。

转置路径（opA = T）的处理：行列索引交换访问方向，依赖方向反转。在依赖图构建阶段完成：

| fill_mode | opA | 边 u→v 的条件 | 等价语义 |
|-----------|-----|-------------|---------|
| LOWER | N | A[v,u] ≠ 0 且 u < v | 下三角前代 |
| LOWER | T | A[v,u] ≠ 0 且 u > v | 下三角转置 = 上三角回代 |
| UPPER | N | A[v,u] ≠ 0 且 u > v | 上三角回代 |
| UPPER | T | A[v,u] ≠ 0 且 u < v | 上三角转置 = 下三角前代 |

#### workspace 布局

```
Device ExternalBuffer 内存布局
════════════════════════════════════════════════════════════════
  偏移量         │ 大小                   │ 内容
─────────────────┼────────────────────────┼─────────────────────
  0              │ 64 B                   │ header（保留/填充）
  tilingOff      │ sizeof(SpsmTilingData) │ SpsmTilingData 结构
  reorderOff     │ M × 4 B                │ int32 reorder[M]
                 │                        │ 逻辑行 -> 原始行号
  groupOff       │ N_entries × 4 B        │ int32 group_edges[]
                 │                        │ N_entries = BlockDim
                 │                        │   × (maxLevel+1) + 1
─────────────────┼────────────────────────┼─────────────────────
  endOff         │ (总字节数)              │ 64B 对齐
════════════════════════════════════════════════════════════════
```

对齐常量和大小计算：

```cpp
#define SPSM_WS_HEADER_BYTES  64
#define SPSM_WS_ALIGN         64

static inline int64_t spsm_workspace_size(int64_t m, int64_t blockDim) {
    int64_t tilingOff  = SPSM_WS_HEADER_BYTES;
    int64_t reorderOff = spsm_align_up(
        tilingOff + (int64_t)sizeof(SpsmTilingData), SPSM_WS_ALIGN);
    int64_t groupOff   = spsm_align_up(
        reorderOff + (int64_t)sizeof(int32_t) * m, SPSM_WS_ALIGN);
    int64_t endOff     = spsm_align_up(
        groupOff + (int64_t)sizeof(int32_t) * (blockDim * (m + 1) + 1),
        SPSM_WS_ALIGN);
    return endOff;
}
```

bufferSize 计算使用 `maxLevel = M` 作为最坏情况估计。

#### SpsmDescrInternal 内部结构

```cpp
struct SpsmDescrInternal {
    bool analysisDone = false;         // 是否已完成 analysis
    void *activeBuffer = nullptr;      // 当前 active 的 workspace buffer
    int32_t fillMode = 0;             // LOWER=0, UPPER=1
    int32_t diagType = 0;             // NON_UNIT=0, UNIT=1
    int32_t opA = 0;                  // N=0, T=1
    int32_t tilingKey = 0;            // bit0:fill, bit1:diag, bit2:opA
    int64_t m = 0;                    // 矩阵维度
    int64_t nnz = 0;
    int32_t maxLevel = 0;             // 最大层级数
};
```

#### SpsmTilingData 结构

```cpp
typedef struct SpsmTilingData {
    int32_t m;                  // 方阵行数/列数
    int32_t nrhs;               // 右端项列数（多 RHS）
    int32_t nnz;                // CSR 非零元素总数
    int32_t maxLevel;           // level scheduling 最大层级数
    int32_t ldc;                // 列主序 leading dimension (>= m)
    int32_t reorder_offset;     // reorder 表在 workspace 中的字节偏移
    int32_t group_offset;       // group_edges 在 workspace 中的字节偏移
    int32_t tiling_key;         // bit0:fill, bit1:diag, bit2:opA
    float   alpha_host;         // alpha 标量值（solve 时可刷新）
} SpsmTilingData;
```

**遵守 R4 规则**：TilingData 仅含标量字段和偏移量指针，禁止使用数组（如 `startOffset[MAX_CORE]`）。核间分配信息通过 workspace 中的 group_edges 表间接访问。

#### Greedy Bin-Packing 分核策略

对每个 level 内各行按 nnz 进行贪心分桶，分配到 blockDim 个核：

```
算法：GreedyBinPackByLevel(levelGroups, nnz[], blockDim)

cursor = 0
group_edges[0] = 0

对于每个 level L = 0 .. maxLevel:
    1. 将 levelGroups[L] 内行按 nnz 降序排序 -> sorted_rows
    2. 初始化 core_load[0..blockDim-1] = 0, buckets 为空
    3. 遍历 sorted_rows 中每行 r:
       a. 找到当前 core_load 最小的核 c = argmin(core_load)
       b. 将行 r 放入 buckets[c]
       c. core_load[c] += nnz[r]
    4. 按 core 顺序整理 reorder:
       for c = 0 .. blockDim-1:
           for r in buckets[c]:
               reorder[cursor++] = r
           group_edges[L * blockDim + c + 1] = cursor
```

reorder 表与 group_edges 表语义：

| 数组 | 说明 |
| ---- | ---- |
| `reorder[r]` | 逻辑行号 r 映射到原始行号，Kernel 通过此映射定位 CSR 行数据 |
| `group_edges[L*blockDim + c]` | Core c 在 level L 的起始逻辑行号（inclusive） |
| `group_edges[L*blockDim + c + 1]` | Core c 在 level L 的结束逻辑行号（exclusive） |

分核示意图（4 核示例，M=9）：

```
Level 0 (2行):  [行0 nnz=5, 行3 nnz=2]
  贪心分配 ->  Core0:行0(5)  Core1:行3(2)  Core2:空  Core3:空

Level 1 (4行):  [行1 nnz=8, 行2 nnz=3, 行4 nnz=6, 行6 nnz=2]
  贪心分配 ->  Core0:行2(3)  Core1:行1(8)  Core2:行4(6)  Core3:行6(2)

Level 2 (3行):  [行5 nnz=7, 行7 nnz=4, 行8 nnz=1]
  贪心分配 ->  Core0:行5(7)  Core1:行7(4)  Core2:行8(1)  Core3:空

Kernel 执行顺序：
  Level 0 -> barrier -> Level 1 -> barrier -> Level 2 -> 完成
```

#### Active Buffer 与 analysis 缓存

analysis 完成后 `spsmDescr->activeBuffer = externalBuffer`。后续 solve：

- 若传入相同 buffer 且 `analysisDone` 为 true：仅刷新 TilingData 中动态字段（alpha_host），跳过重复依赖分析
- 若传入不同 buffer 或 `analysisDone` 为 false：自动触发 analysis
- `updateMatrix` 仅替换 `matA->values` 指针，保持 `analysisDone = true` 和 activeBuffer 有效

#### Host 代码规范

**include 最小集合**：

```cpp
#include <cstdint>
#include <new>                                  // std::nothrow
#include "log/log.h"                            // 强制 dlog，禁止 printf
#include "cann_ops_sparse.h"
#include "spsm.h"                               // 类型映射、TilingData、转换函数
#include "spsm_kernel.h"                         // kernel_launch 声明
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
```

禁止引入：`acl/acl.h`、`cann_ops_sparse_common.h`、`tiling/platform/platform_ascendc.h`（由公共头文件间接引入）。

**Host 函数结构**：

```cpp
namespace {
    static aclsparseStatus_t ValidateSpsmParams(/*...*/) {
        // 逐参数独立校验
    }
    static aclsparseStatus_t LaunchSpsmKernel(/*...*/) {
        // 描述符解析 -> TilingData 刷新 -> kernel_launch
    }
}

extern "C" {
    aclsparseStatus_t aclsparseSpSMAnalysis(/*...*/) {
        aclsparseStatus_t st = ValidateSpsmParams(/*...*/);
        if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
        return SpsmDoAnalysis(/*...*/);
    }
    aclsparseStatus_t aclsparseSpSMSolve(/*...*/) {
        aclsparseStatus_t st = ValidateSpsmParams(/*...*/);
        if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
        if (!descrInternal->analysisDone) {
            st = SpsmDoAnalysis(/*...*/);
            if (st != ACL_SPARSE_STATUS_SUCCESS) return st;
        }
        return LaunchSpsmKernel(/*...*/);
    }
}
```

**Kernel Launch 规范**：

```cpp
// spsm_kernel.h — 独立声明头文件，禁止在 host.cpp 中 extern 前向声明
extern "C" void spsm_kernel_launch(
    const void *csrRowOffsets, const void *csrColInd,
    const void *csrValues,     const void *matB,
    void       *matC,          void       *workspaceGM,
    void       *tilingGM,      int32_t     dataType,
    int32_t     tilingKey,     uint32_t    blockDim,
    void       *stream);
```

关键约束：

- `blockDim` 运行时动态获取（`PlatformAscendCManager::GetCoreNumAiv()`），禁止硬编码
- Kernel 启动是异步的，host 侧调用后立即返回
- TilingData 通过 device workspace 传递，禁止 `aclrtMalloc` + `aclrtMemcpy(H2D)` 传递
- 禁止在 host 侧调用 `aclrtSynchronizeStream`

### Kernel 侧设计

#### 总体架构

Kernel 侧采用 Ascend C SIMT 编程模型，基于 `__simt_vf__` 核函数 + `__aicore__` SPMD 外层实现。整体执行流程：

```
Init 阶段（每个 Core 执行一次）:
  1. 从 tilingGM 读取 SpsmTilingData（GM -> 寄存器）
  2. 解析 TilingKey -> 编译期常量（IsUpper / IsUnit / TransA）
  3. 获取 coreId = GetBlockIdx(), coreNum = GetBlockNum()
  4. 定位 reorder 和 group_edges 指针

Process 阶段（外层 Level 循环，Level 间 barrier 同步）:
  for level = 0 .. maxLevel:
    1. 获取本 core 在 level 内的行分配范围:
       start = group_edges[level * coreNum + coreId]
       end   = group_edges[level * coreNum + coreId + 1]
    2. 对分配的每行 r (r in [start, end)):
       执行稀疏内积 + 对角缩放，SIMT 线程并行处理 RHS 列
    3. barrier() — 核间同步，确保所有 Core 完成当前 level
       ↑ 即使空 bin 也必须执行 asc_vf_call + barrier（dav-3510 架构约束）
```

#### SIMT 核函数

```cpp
template<int TilingKey>
__simt_vf__ __aicore__ __launch_bounds__(kMaxSimtThreadsPerBlock)
void SpsmCsrSimtCompute(
    __gm__ int32_t *csrRowOff,      // CSR 行偏移 [m+1]
    __gm__ int32_t *csrColInd,      // CSR 列索引 [nnz]
    __gm__ float   *csrValues,      // CSR 非零值 [nnz]
    __gm__ float   *matB,           // 稠密 B [ldc x nrhs] col-major
    __gm__ float   *matC,           // 稠密 C [ldc x nrhs] col-major
    __gm__ int32_t *reorder,        // 逻辑行 -> 原始行映射
    int32_t nrhs, int32_t ldc,
    int32_t rowStart, int32_t rowEnd, float alpha)
{
    // 编译期常量提取（TilingKey 实现零开销分支消除）
    constexpr bool kIsUpper = (TilingKey & 1) != 0;   // bit0: fill_mode
    constexpr bool kIsUnit  = (TilingKey & 2) != 0;   // bit1: diag_type
    constexpr bool kTransA  = (TilingKey & 4) != 0;   // bit2: opA

    uint32_t threadNum = blockDim.x;
    uint64_t threadIdxX = static_cast<uint64_t>(threadIdx.x);

    for (int32_t r = rowStart; r < rowEnd; ++r) {
        int32_t origRow = reorder[r];
        int32_t s = csrRowOff[origRow];
        int32_t e = csrRowOff[origRow + 1];

        // 定位对角元素（unit_diag 时跳过，使用 diagVal = 1.0）
        float diagVal = 1.0f;
        if constexpr (!kIsUnit) {
            for (int32_t p = s; p < e; ++p) {
                if (csrColInd[p] == origRow) {
                    diagVal = csrValues[p];
                    break;
                }
            }
        }

        // SIMT 线程并行处理 RHS 列（grid-stride loop）
        for (int64_t k = threadIdxX; k < static_cast<uint64_t>(nrhs);
             k += static_cast<uint64_t>(threadNum)) {

            float acc = 0.0f;

            // CSR 稀疏内积：累加依赖行
            for (int32_t p = s; p < e; ++p) {
                int32_t j = csrColInd[p];
                bool depends = false;

                if constexpr (!kTransA) {
                    if constexpr (kIsUpper) {
                        depends = (j > origRow);
                    } else {
                        depends = (j < origRow);
                    }
                } else {
                    if constexpr (kIsUpper) {
                        depends = (origRow > j);
                    } else {
                        depends = (origRow < j);
                    }
                }

                if (depends) {
                    float cVal = matC[static_cast<uint64_t>(j) *
                                      static_cast<uint64_t>(ldc) + k];
                    acc += csrValues[p] * cVal;
                }
            }

            // 对角缩放与 alpha
            float bVal = matB[static_cast<uint64_t>(origRow) *
                              static_cast<uint64_t>(ldc) + k];
            if constexpr (kIsUnit) {
                matC[static_cast<uint64_t>(origRow) *
                     static_cast<uint64_t>(ldc) + k] = alpha * bVal - acc;
            } else {
                matC[static_cast<uint64_t>(origRow) *
                     static_cast<uint64_t>(ldc) + k] =
                     (alpha * bVal - acc) / diagVal;
            }
        }
    }
}
```

#### KernelSpsmSimt 封装类

```cpp
template<int TilingKey>
class KernelSpsmSimt {
public:
    __aicore__ inline KernelSpsmSimt() {}

    __aicore__ inline void Init(
        GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
        GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        tilingData_ = LoadSpsmTilingData(tilingGM);   // GM -> 寄存器
        wsBase_     = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
        rowOff_     = reinterpret_cast<__gm__ int32_t *>(csrRowOffsets);
        colInd_     = reinterpret_cast<__gm__ int32_t *>(csrColInd);
        values_     = reinterpret_cast<__gm__ float *>(csrValues);
        matB_       = reinterpret_cast<__gm__ float *>(matB);
        matC_       = reinterpret_cast<__gm__ float *>(matC);
        reorder_    = reinterpret_cast<__gm__ int32_t *>(
            wsBase_ + static_cast<uint64_t>(tilingData_.reorder_offset));
        groupEdges_ = reinterpret_cast<__gm__ int32_t *>(
            wsBase_ + static_cast<uint64_t>(tilingData_.group_offset));
    }

    __aicore__ inline void Process()
    {
        const int32_t m        = tilingData_.m;
        const int32_t nrhs     = tilingData_.nrhs;
        const int32_t maxLevel = tilingData_.maxLevel;
        const int32_t ldc      = tilingData_.ldc;
        const float   alpha    = tilingData_.alpha_host;

        const int32_t coreNum = static_cast<int32_t>(GetBlockNum());
        const int32_t coreId  = static_cast<int32_t>(GetBlockIdx());

        for (int32_t level = 0; level <= maxLevel; ++level) {
            int32_t idx = level * coreNum + coreId;
            int32_t rowStart = groupEdges_[idx];
            int32_t rowEnd   = groupEdges_[idx + 1];

            int32_t numRows = (rowEnd > rowStart) ? (rowEnd - rowStart) : 0;
            uint32_t simtThreadNum = 1u;
            if (numRows > 0 && nrhs > 0 && coreNum > 0) {
                uint64_t totalWork =
                    static_cast<uint64_t>(numRows) * static_cast<uint64_t>(nrhs);
                simtThreadNum = static_cast<uint32_t>(
                    totalWork / static_cast<uint64_t>(coreNum));
                if (simtThreadNum > kMaxSimtThreadsPerBlock) {
                    simtThreadNum = kMaxSimtThreadsPerBlock;
                }
                if (simtThreadNum < 1u) simtThreadNum = 1u;
            }

            // 空 bin 也必须执行 asc_vf_call（dav-3510 架构约束）
            asc_vf_call<SpsmCsrSimtCompute<TilingKey>>(
                dim3{simtThreadNum},
                rowOff_, colInd_, values_, matB_, matC_,
                reorder_, nrhs, ldc,
                rowStart, rowEnd, alpha);

            // 核间 barrier 同步
            __barrier();
        }
    }

private:
    __gm__ uint8_t  *wsBase_{nullptr};
    __gm__ int32_t  *rowOff_{nullptr};
    __gm__ int32_t  *colInd_{nullptr};
    __gm__ float    *values_{nullptr};
    __gm__ float    *matB_{nullptr};
    __gm__ float    *matC_{nullptr};
    __gm__ int32_t  *reorder_{nullptr};
    __gm__ int32_t  *groupEdges_{nullptr};
    SpsmTilingData   tilingData_{};
};
```

#### fp32 Kernel 入口与 TilingKey 派发

```cpp
extern "C" __global__ __aicore__ void spsm_custom_fp32(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd, GM_ADDR csrValues,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const SpsmTilingData tiling = LoadSpsmTilingData(tilingGM);

    switch (tiling.tiling_key) {
        case 0: { KernelSpsmSimt<0> op; op.Init(csrRowOffsets, csrColInd,
            csrValues, matB, matC, workspaceGM, tilingGM); op.Process(); } break;
        case 1: { KernelSpsmSimt<1> op; op.Init(csrRowOffsets, csrColInd,
            csrValues, matB, matC, workspaceGM, tilingGM); op.Process(); } break;
        case 2: { KernelSpsmSimt<2> op; op.Init(csrRowOffsets, csrColInd,
            csrValues, matB, matC, workspaceGM, tilingGM); op.Process(); } break;
        case 3: { KernelSpsmSimt<3> op; op.Init(csrRowOffsets, csrColInd,
            csrValues, matB, matC, workspaceGM, tilingGM); op.Process(); } break;
        case 4: { KernelSpsmSimt<4> op; op.Init(csrRowOffsets, csrColInd,
            csrValues, matB, matC, workspaceGM, tilingGM); op.Process(); } break;
        case 5: { KernelSpsmSimt<5> op; op.Init(csrRowOffsets, csrColInd,
            csrValues, matB, matC, workspaceGM, tilingGM); op.Process(); } break;
        case 6: { KernelSpsmSimt<6> op; op.Init(csrRowOffsets, csrColInd,
            csrValues, matB, matC, workspaceGM, tilingGM); op.Process(); } break;
        case 7: { KernelSpsmSimt<7> op; op.Init(csrRowOffsets, csrColInd,
            csrValues, matB, matC, workspaceGM, tilingGM); op.Process(); } break;
        default: break;
    }
}

// Host 侧 Launch 派发
extern "C" void spsm_kernel_launch(
    const void *csrRowOffsets, const void *csrColInd,
    const void *csrValues,     const void *matB,
    void       *matC,          void       *workspaceGM,
    void       *tilingGM,      int32_t     dataType,
    int32_t     tilingKey,     uint32_t    blockDim,
    void       *stream)
{
    if (dataType == SPSM_DTYPE_FP32) {
        spsm_custom_fp32<<<blockDim, nullptr, stream>>>(
            (GM_ADDR)csrRowOffsets, (GM_ADDR)csrColInd, (GM_ADDR)csrValues,
            (GM_ADDR)matB, (GM_ADDR)matC,
            (GM_ADDR)workspaceGM, (GM_ADDR)tilingGM);
    }
}
```

### TilingKey 规划

通过 TilingKey 编码不同算法分支的编译期常量，实现 Kernel 内的零开销分支消除：

| 位域 | 宽度 | 含义 | 编码 |
| :--: | :--: | ---- | ---- |
| bit[0] | 1 | fill_mode | 0 = LOWER（下三角），1 = UPPER（上三角） |
| bit[1] | 1 | diag_type | 0 = NON_UNIT（显式对角），1 = UNIT（单位对角） |
| bit[2] | 1 | opA | 0 = N（非转置），1 = T（转置） |
| bit[7:3] | 5 | reserved | 保留，固定为 0 |

8 种模板实例：

| TilingKey | 组合 |
|-----------|------|
| 0 | LOWER + NON_UNIT + N |
| 1 | UPPER + NON_UNIT + N |
| 2 | LOWER + UNIT + N |
| 3 | UPPER + UNIT + N |
| 4 | LOWER + NON_UNIT + T |
| 5 | UPPER + NON_UNIT + T |
| 6 | LOWER + UNIT + T |
| 7 | UPPER + UNIT + T |

首期仅支持 fp32，数据类型不作为 TilingKey 维度。后续扩展 fp64 / fp16 / complex 时可增加 bit[4:3] 编码。

### 并行策略

三层并行设计：

| 并行层级 | 并行方式 | 职责 |
| -------- | -------- | ---- |
| Level 内并行 | 同 level 内不同行分配到不同 Core | 无依赖的行同时计算 |
| Core 内线程并行 | 同一行内 SIMT 线程分担不同 RHS 列 | 每线程处理 ceil(nrhs / 线程数) 列 |
| Level 间串行 | level 间 barrier 同步 | 保证行间依赖正确性 |

### dav-3510 架构特别约束

空 bin（某 Core 在某 level 无分配行）仍须执行 `asc_vf_call`（simtThreadNum=1），不可提前 return。原因：dav-3510 的 split AIC/AIV core 架构要求每个 outer core 必须均匀发出 `asc_vf_call`，否则 cube/vector handshake deadlock，导致 `aclrtSynchronizeStream` 永久挂起。空 bin 传入 rowStart == rowEnd，SpsmCsrSimtCompute 内部立即返回（numRows ≤ 0），不做实际计算但参与 dispatch。

### 代码工程目录

```
ops-sparse/src/spsm/
├── arch35/
│   ├── spsm_host.cpp              # Host侧：参数校验 -> 依赖分析 -> Level Scheduling -> Launch
│   ├── spsm_kernel.cpp            # Device侧：SIMT 层次求解 kernel
│   ├── spsm_kernel.h              # kernel_launch 签名声明（独立头文件）
│   ├── spsm_csr_mat.h             # CSR三角矩阵预处理（依赖图分析 + 层次调度）
│   ├── spsm_csr_mat.cpp           # 依赖图构建 + Level Scheduling + Bin-Packing
│   └── spsm.h                     # 公共头文件：类型映射 + SpsmTilingData + 内联转换函数

test/spsm/
├── CMakeLists.txt                  # ops_sparse_add_test(spsm ${OPS_SPARSE})
├── README.md                       # 算子说明文档（含 aclsparse C++ 调用示例）
└── arch35/
    └── spsm_test.cpp               # 测试主文件
```

---

## 支持硬件

| 支持的芯片版本 | 状态 |
| -------------- | ---- |
| Ascend 950PR / 950DT | 支持 |

SpSM 源码位于 `src/spsm/arch35/`，编译时通过 `--soc=ascend950` 参数指定目标平台。非 `ascend950*` 平台的编译构建将跳过本算子。

---

## 算子约束限制

1. 稀疏格式仅支持 CSR（`ACL_SPARSE_FORMAT_CSR`）
2. 索引类型仅支持 int32（`ACL_SPARSE_INDEX_32I`），索引基址仅支持 0-based（`ACL_SPARSE_INDEX_BASE_ZERO`）
3. 数据类型仅支持 fp32 同精度（A / B / C / computeType / alpha 均为 `ACL_FLOAT`）
4. `opA` 支持 N 和 T；`opB` 仅支持 N（与 cuSPARSE SpSM 一致）
5. 稠密矩阵仅支持列主序（`ACL_SPARSE_ORDER_COL`）
6. A 须为方阵（M × M），M ≥ 1，且 M ≤ INT32_MAX
7. `updateMatrix` 仅支持 `ACL_SPARSE_SPSM_UPDATE_GENERAL`（全量数值更新）
8. analysis 与 solve 之间不得修改 matA 结构、externalBuffer、spsmDescr（与 cuSPARSE 语义一致）
9. matA 行列索引可以不排序（Kernel 不依赖 CSR 列索引有序，遍历整行查找对角元和非对角元）
10. 空 bin 必须执行 `asc_vf_call`（dav-3510 架构约束），使用 simtThreadNum=1 兜底

---

# 可维可测分析

## 精度标准

### 评估方法

以 CPU 端双精度（fp64）三角求解作为参考真值（golden），对 NPU fp32 输出（actual）进行误差评估。

### 误差指标

**平均相对误差（MERE）**：

```
MERE = (1 / (M * nrhs)) * sum_i sum_j |actual[i,j] - golden[i,j]| / (|golden[i,j]| + 1e-7)
```

**最大相对误差（MARE）**：

```
MARE = max_i max_j |actual[i,j] - golden[i,j]| / (|golden[i,j]| + 1e-7)
```

### 通过标准

| 标准来源 | 要求 |
| -------- | ---- |
| 生态算子开源精度标准 | fp32: MERE 小于 2^(-13) ≈ 1.22e-4，MARE 小于 10 × 2^(-13) |
| ATK 双标杆 L2 | 相对误差 ≤ 2 × 10^(-6)，均方误差比 ≤ 1.2，最大误差比 ≤ 1.2 |

ATK 双标杆测试需要 A100 环境，开发者自行准备。

### 精度风险与缓解

| 风险 | 说明 | 缓解措施 |
| ---- | ---- | -------- |
| fp32 累加舍入 | 大规模三角求解中累加项数可达 O(10^4) | 后续版本可选 Kahan 补偿求和 |
| 除法误差 | 对角元量级差异大，除以小对角元放大误差 | 检测对角元量级，对近奇异矩阵标记警告 |
| 计算顺序影响 | Level scheduling 改变行求解顺序 | 数学等价，solve 保证 bit-wise 确定性 |

## 性能标准

| 验收维度 | 描述 | 来源 |
| -------- | ---- | ---- |
| 功能标准 | 覆盖 6 接口全链路 | 任务书 |
| 精度标准 | 满足生态算子精度标准 + ATK 双标杆 L2 | 任务书 |
| 性能标准 | solve 阶段 NPU 性能 ≥ A100 cuSPARSE 的 1.0 倍 | 任务书 |

### 固定参考用例

| 编号 | 场景 | M | NNZ | nrhs | GPU 参考 (A100, μs) | 达标 |
|------|------|---|-----|------|---------------------:|------|
| P-01 | 下三角多 RHS 基础 solve | 256 | 2725 | 8 | 2203 | ≥ 1.0x |
| P-02 | updateMatrix 后 re-solve | 128 | 1322 | 4 | 2291 | ≥ 1.0x |
| P-03 | analysis 阶段 NULL values | 128 | 1327 | 4 | 2014 | ≥ 1.0x |

### 泛化覆盖范围（另抽 200 组）

| 维度 | 覆盖范围 | 抽样策略 |
| ---- | -------- | -------- |
| 矩阵规模 M | 10^2 ~ 10^5 | 对数均匀分布 |
| nnz | 与 M 成比例，平均度 5~50 | 均匀采样 |
| nrhs | 1, 8, 32, 128 | 全覆盖 |
| 三角属性 | 下三角/上三角 × unit_diag × trans | 8 种组合 |
| 稀疏结构 | 随机三角 CSR | 固定种子复现 |
| updateMatrix 路径 | 至少 5 组 | 数值变、结构不变 |
| in-place | 至少 5 组 | matB/matC 共用 device 指针 |
| NULL values | 至少 5 组 | bufferSize/analysis 阶段 values=NULL |
| 边界条件 | M=1, nrhs=1 | 极小规模 |
| 近奇异对角 | 5 组 | 仅功能验收 |

## 测试设计

### 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01 | 下三角 unit_diag=False，fp32 基础求解 | cuSPARSE CSR sample |
| TC-02 | 上三角 + transpose | 自行构造 |
| TC-03 | nrhs=1 与 nrhs>1 对比验证 | 自行构造 |
| TC-04 | updateMatrix 后 re-solve（数值变、结构不变） | cuSPARSE |
| TC-05 | matB/matC in-place（同一 values 指针） | cuSPARSE |
| TC-06 | bufferSize/analysis 阶段 matB/matC values 为 NULL | cuSPARSE |
| TC-07 | alpha 特殊值（0, 1, 负值） | 自行构造 |
| TC-08 | M=1 边界条件 | 自行构造 |
| TC-09 | analysis 缓存复用（多次 solve 同一结构不同数值） | cuSPARSE |

### 测试目录结构

```
test/spsm/
├── CMakeLists.txt                    # ops_sparse_add_test(spsm ${OPS_SPARSE})
├── README.md                         # 算子说明 + aclsparse 调用示例
└── arch35/
    └── spsm_test.cpp                 # 测试主文件
```

### 测试程序流程

```
1. Init: aclInit -> aclrtSetDevice -> aclrtCreateStream
2. 生成测试数据：随机三角 CSR + 稠密 B + CPU fp64 golden 参考解
3. Device 内存分配 + H2D 拷贝
4. aclsparseCreate -> SetStream -> CreateCsr -> CreateDnMat -> CreateDescr
5. GetBufferSize -> 分配 Buffer -> Analysis -> Solve（计时）
6. aclrtSynchronizeStream -> D2H 拷贝结果
7. 精度验证：MERE/MARE vs fp64 golden, 输出 PASS/FAIL
8. 清理：DestroyDescr -> DestroySpMat -> DestroyDnMat -> Destroy -> aclrtFree
```

---

## 兼容性分析

| 维度 | 分析结论 |
| ---- | -------- |
| 接口兼容性 | 6 个 aclsparseSpSM 接口均为新增；新增枚举（Descr_t / Alg_t / Update_t / FillMode_t / DiagType_t）均无冲突 |
| ABI 兼容性 | 首期交付 |
| 数据格式兼容性 | 复用已有 aclsparseSpMatDescr_t / aclsparseDnMatDescr_t 描述符；三角属性通过 spsmDescr 独立存储 |
| 平台兼容性 | 限定 Ascend 950PR/950DT，arch35 目录隔离 |
| CANN 版本 | 依赖 CANN 9.0.0+ |
| 多线程安全 | spsmDescr 单线程持有；handle 内 stream 绑定确保单流语义 |
| 前向兼容 | alg 参数预留算法扩展；computeType 预留 fp64/complex 扩展；TilingKey reserved bits 预留；updatePart 预留分块更新路径 |

### 代码复用清单

| 复用模块 | 来源 | 复用方式 |
| -------- | ---- | -------- |
| aclsparseSpMatDescr / aclsparseDnMatDescr 内部结构 | `src/common/aclsparse_descr_internal.h` | 直接复用 |
| AclsparseValidateSupportedCsrIndexTypes | `src/common/aclsparse_descr_internal.h` | 直接调用（索引校验） |
| PlatformAscendCManager::GetCoreNumAiv | CANN PlatformAscendC | 直接调用（核数获取，遵守 R2） |
| Workspace 64B 对齐策略 | ops-sparse 仓库约定 | 独立定义 SPSM_WS_ALIGN |
| asc_vf_call + 空 bin 兜底 | ops-sparse 仓库现有实现 | 设计模式参考，独立实现 |
| GM_ADDR / CHECK_RET 宏 | 仓库公共约定 | 算子头文件中独立定义 |
| log/log.h dlog 日志系统 | CANN 公共库 | 强制使用，禁止 printf |

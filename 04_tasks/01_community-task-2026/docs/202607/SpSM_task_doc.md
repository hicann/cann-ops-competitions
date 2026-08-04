# 7月社区任务-SpSM算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Ascend 950PR
- **开源仓地址**：https://gitcode.com/cann/ops-sparse
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C + C++（aclsparse 接口）
- **对标参考**：[cuSPARSE cusparseSpSM §6.6.12](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespsm)


## 任务概述

参考 cuSPARSE **SpSM**（Sparse triangular Solve with Multiple right-hand sides），在昇腾 NPU 上实现稀疏三角求解：

$$
op(A) \cdot X = \alpha \cdot op(B) \quad\Rightarrow\quad X = \alpha \cdot op(A)^{-1} op(B)
$$

或等价形式 $C = \alpha \cdot op(A)^{-1} op(B)$（cuSPARSE 以稠密 `matC` 存解）。

**各参数含义说明：**

1. $A$：稀疏三角系数矩阵，仅支持上三角/下三角格式，存储格式为 **CSR**；
2. $op(\cdot)$：矩阵操作符，可选不转置 `op(A)=A` 或转置 `op(A)=A^T`；
3. $X$：方程组求解输出，稠密矩阵，与等价式中 $C$ 完全等同；
4. $B$：多右端稠密输入矩阵，多列代表多组并行方程组；
5. $\alpha$：全局浮点缩放标量，用于整体缩放计算结果；
6. $A^{-1}$：稀疏三角矩阵的逆，算子内部通过三角回代间接求解，不生成显式逆矩阵。

**验收口径**：

- **C++**：`aclsparseSpSM*` 与 cuSPARSE 调用流程对齐（`bufferSize` 预备 → `analysis` → `solve`；`updateMatrix` 为可选数值更新路径）。

### 典型场景

图预条件、三角分解回代（ILU/IC 求解）、稀疏三角系统多 RHS 求解。

---

## 核心开发要求及验收标准

### 1. C++ 层接口（**必选，对齐 cuSPARSE**）

cuSPARSE SpSM 核心流程为 **`bufferSize` 预备 → `analysis` → `solve` 三步骤**（另含描述符 `createDescr`/`destroyDescr` 与可选 `updateMatrix`），须完整实现：

```C
aclsparseStatus_t aclsparseSpSMCreateDescr(aclsparseSpSMDescr_t *spsmDescr);
aclsparseStatus_t aclsparseSpSMDestroyDescr(aclsparseSpSMDescr_t spsmDescr);

aclsparseStatus_t aclsparseSpSMGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr,
    size_t *bufferSize);

aclsparseStatus_t aclsparseSpSMAnalysis(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr,
    void *externalBuffer);

aclsparseStatus_t aclsparseSpSMSolve(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr);

aclsparseStatus_t aclsparseSpSMUpdateMatrix(
    aclsparseHandle_t handle,
    aclsparseSpSMDescr_t spsmDescr,
    void *newValues,
    aclsparseSpSMUpdate_t updatePart);
```


**格式**：仅支持 **CSR**（与 cuSPARSE 对齐）。

**调用语义**（与 cuSPARSE §6.6.12 一致）：

- **`bufferSize` / `analysis`**：`matB`、`matC` 的 **values 指针可为 NULL**（仅用于 workspace 估算与符号分析）；**描述符本身不可为 NULL**；
- **`solve`**：`matB`、`matC` 须指向有效 values；
- **in-place**：`matB` 与 `matC` 可使用**同一 values 设备指针**（其余稠密描述符字段如 `order` 可独立设置）。

**工程目录**：

```
ops-sparse/src/spsm/
```

### 2. 数据类型支持

**对标范围**：对齐 [cuSPARSE cusparseSpSM §6.6.12](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespsm) 官方 dtype 表。SpSM 为**同精度**：`matA` values、`matB`、`matC`、`computeType` **四者 dtype 相同**（cuSPARSE 文档无 Mixed-precision 小节）。

**列含义**：

- **A**：稀疏三角系数矩阵 `matA` 的 values；
- **B**：稠密右端项 `matB`（多 RHS）；
- **C**：稠密解 `matC`；
- **computeType**：实际计算精度。

**稀疏格式与索引**：

| 项目 | 本任务 |
|------|-----------|
| 稀疏格式 | **CSR** |
| 索引 | **int32**（`INDEX_32I`） |
| fill / diag | LOWER/UPPER、UNIT/NON_UNIT（映射 cuSPARSE fill/diag 属性） |

SpSM 为**全新算子**（ops-sparse 仓当前无 `src/spsm/`），`aclsparseSpSM*` + Kernel 均须从零交付。

##### 2.1 dtype 组合

| A / B / C / computeType | cuSPARSE 枚举 | 状态 |
|---------------------------|---------------|------|
| fp32 | `CUDA_R_32F` | **待实现** |

`alpha` 标量 dtype 与 `computeType` 一致（为 **float32**）。

**验收**：须覆盖 ACL `aclsparseSpSM*`（bufferSize / analysis / solve）+ `updateMatrix` 全链路。

### 3. 算子约束

1. **analysis 与 solve 间** 不得修改 matA 结构、`externalBuffer`、`spsmDescr`（与 cuSPARSE 一致）；
2. **updateMatrix** 仅更新数值、结构不变时可跳过 re-analysis（对标 cuSPARSE）；
3. **solve 阶段 bit-wise 确定性**（cuSPARSE 保证）；
4. 多 RHS：`matB`/`matC` 为 `[m, nrhs]` 稠密矩阵；
5. **in-place**：`matB` 与 `matC` 可共用同一 values 指针（cuSPARSE 支持）；
6. **`bufferSize` / `analysis`**：`matB`/`matC` 的 values 可为 NULL；**`solve`** 须提供有效 values；描述符不可为 NULL。

### 4. 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01 | 下三角 unit_diag=False | cuSPARSE CSR sample |
| TC-02 | 上三角 + transpose | 自行构造 |
| TC-03 | nrhs=1 与 nrhs>1 | 自行构造 |
| TC-04 | updateMatrix 后 re-solve | cuSPARSE |
| TC-05 | matB/matC in-place（同一 values 指针） | cuSPARSE |
| TC-06 | bufferSize/analysis 阶段 matB/matC values 为 NULL | cuSPARSE |

### 5. 性能要求

对标 **cuSPARSE SpSM（A100）**，NPU 性能须达到 GPU 参考实现的 **1.0 倍**及以上（即 ≥1.0×）。

**固定参考用例**（必测 3 组；在 **NVIDIA A100** 上实测 cuSPARSE 耗时填入「GPU 参考耗时」，取多次运行 **avg**，analysis 一次性开销可单独报告，作为 NPU 验收基准）：

| 编号 | 场景 | m | nnz | nrhs | dtype | GPU 参考耗时 (μs) | 达标要求 |
|------|------|---|-----|------|-------|------------------:|----------|
| 01 | 下三角多 RHS 基础 solve | 256 | 2725 | 8 | fp32 | 2203 | NPU ≥ 1.0× |
| 02 | updateMatrix 后 re-solve | 128 | 1322 | 4 | fp32 | 2291 | NPU ≥ 1.0× |
| 03 | analysis 阶段 NULL values | 128 | 1327 | 4 | fp32 | 2014 | NPU ≥ 1.0× |

**泛化覆盖范围**（**另抽 200 组**用例；由下列维度组合抽样或网格扫描生成，须附完整用例列表；**solve 阶段**性能达标 ≥1.0×）：

| 维度 | 覆盖范围 |
|------|----------|
| 矩阵规模 | `m`：10²～10⁵；`nnz` 与 `m` 成比例（平均度约 5～50） |
| 多 RHS | `nrhs`：1、8、32、128 |
| 三角属性 | 下三角 / 上三角；`unit_diag` True / False；`trans` True / False 等组合须纳入 200 组抽样 |
| 稀疏结构 | 随机三角 CSR |
| 更新路径 | `updateMatrix` 后 re-solve 至少 5 组（数值变、结构不变） |
| in-place / NULL values | matB/matC 共用 values 至少 5 组；bufferSize/analysis 阶段 NULL values 至少 5 组 |
| 边界 | `m=1`；`nrhs=1`；近奇异对角（条件数偏大样本 5 组，仅功能验收） |

### 6. 精度要求

满足《[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)》；使用 [ATK](https://gitcode.com/Ascend/ATK) 进行**单标杆**精度验收，以 CPU 高精度结果为 golden，对 NPU（aclsparse）输出进行误差评估。测试用例及测试工程见[测试目录](./self_test_case/sp_sm/)的 `atk_test/aclSparseSpsm/result/aclSparseSpsm/json/all_aclSparseSpsm.json`（至少完成 200 个 case 测试）。


### 7. 接口分层

```
aclsparseSpSM*（必选）
    → Ascend C Kernel（必选）
```

### 8. 验收交付件

1. 算子设计文档，需根据[参考模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；评审通过后合入[cann-competitions 仓库](https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist)，详细说明见[readme](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。
2. 可指导算子使用与测试的交付件：自测用例、测试代码脚本及脚本运行 README；测试结果报告，需包含所有自测用例结果，含内存对比数据、精度结果、整体测试通过截图。
3. 算子代码的私仓邀请链接、代码仓路径、分支、算子目录。README 文档内容完整、规范。

### 9. PR 合入路径

`https://gitcode.com/cann/ops-sparse/tree/master/src/spsm`

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。
   ![环境截图](pics/yunkaifa.png)
2. 使用 hidevlab WebIDE 算力。
   ![环境截图](pics/zaixiankaifa1.png)
3. 如需额外环境资源，请联系昇腾小助手。


## 特别注意事项

1. SpSM **analysis 缓存**是性能关键，须在文档中说明复用条件；
2. 新增接口需对齐 cuSPARSE 风格；
3. 三角性、对角类型须在描述符层显式设置（映射 cuSPARSE fill/diag 属性）；
4. 设计文档按[社区模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)；
5. README 含 **aclsparse C++** 调用示例；
6. 自验证报告含 aclsparse 测试、AscendOpTest/ATK 日志。

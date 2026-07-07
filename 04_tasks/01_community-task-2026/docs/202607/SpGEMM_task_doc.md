# 7月社区任务-SpGEMM算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Ascend 950PR
- **开源仓地址**：https://gitcode.com/cann/ops-sparse
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C + C++（aclsparse 接口）
- **对标参考**：[cuSPARSE cusparseSpGEMM §6.6.14](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespgemm)（Generic API）

## 任务概述

参考 cuSPARSE **SpGEMM**（稀疏矩阵 × 稀疏矩阵），在昇腾 NPU 上实现 **Ascend C Kernel + aclsparse C++ 接口**，完成设计、开发、测试全流程，验收后合入 **ops-sparse**。

**验收口径**：

- **C++ 层**：`aclsparseSpGEMM*` 系列接口与 cuSPARSE 调用流程、语义对齐（命名与目录结构参考现有 `aclsparseSpMM` / `src/spmm`）。

### 功能定义

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
$$

- A ∈ R(m×k)、B ∈ R(k×n)、C ∈ R(m×n) 均为**稀疏 CSR**；
- α、β 为标量；

SpGEMM 输出 C 的稀疏结构由乘法确定，非零元值由乘积累加得到。

---

## 核心开发要求及验收标准

### 1. C++ 层接口（**必选，对齐 cuSPARSE / ops-sparse 命名**）

接口命名、三阶段（或多阶段）流程须参考 `include/cann_ops_sparse.h` 中 **`aclsparseSpMM`** 及 cuSPARSE **`cusparseSpGEMM`**，新增 **`aclsparseSpGEMM*`** 至 `cann_ops_sparse.h`：

```C
/* 描述符 */
aclsparseStatus_t aclsparseSpGEMMCreateDescr(aclsparseSpGEMMDescr_t *descr);
aclsparseStatus_t aclsparseSpGEMMDestroyDescr(aclsparseSpGEMMDescr_t descr);

/* 阶段 1：工作估算 */
aclsparseStatus_t aclsparseSpGEMMWorkEstimation(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize1, void *externalBuffer1);

aclsparseStatus_t aclsparseSpGEMMGetNumProducts(
    aclsparseSpGEMMDescr_t spgemmDescr, int64_t *numProds);

/* 阶段 2：内存估算（ALG2/ALG3） */
aclsparseStatus_t aclsparseSpGEMMEstimateMemory(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    float chunkFraction,
    size_t *bufferSize3, void *externalBuffer3,
    size_t *bufferSize2);

/* 阶段 3：计算 */
aclsparseStatus_t aclsparseSpGEMMCompute(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize2, void *externalBuffer2);

/* 阶段 4：拷贝结果到 matC（若 compute 与 copy 分离） */
aclsparseStatus_t aclsparseSpGEMMCopy(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr);
```


### 2. 数据类型支持

**对标范围**：对齐 [cuSPARSE cusparseSpGEMM §6.6.14](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespgemm) 官方 dtype 表。SpGEMM **仅同精度**：稀疏 `matA`/`matB` values、稀疏 `matC` values、`computeType` **四者 dtype 相同**（文档**无 Mixed-precision 小节**）。

**列含义**：**A/B/C** 分别为三个稀疏矩阵的 values dtype，与 `computeType` 一致。

**格式与索引**：

| 项目 | 要求 |
|------|------|
| 稀疏格式 | **CSR**（`aclsparseCreateCsr`） |
| 索引 | **int32**（`ACL_SPARSE_INDEX_32I`） |
| opA / opB | **仅 `NON_TRANSPOSE`**（与 cuSPARSE §6.6.14 一致；接口保留形参，传入 `TRANSPOSE` 须返回错误） |

SpGEMM 为**全新算子**（ops-sparse 仓当前无 `src/spgemm/`），`aclsparseSpGEMM*` + Kernel 均须从零交付。

##### 2.1 同精度（Uniform-precision）

| ID | A / B / C / computeType | cuSPARSE 枚举 | ACL / Kernel |
|----|---------------------------|---------------|--------------|
| U1 | fp32 | `CUDA_R_32F` | **待实现** |
| U2 | fp16 | `CUDA_R_16F` | **待实现** |
| U3 | bf16 | `CUDA_R_16BF` | **待实现** |

> cuSPARSE 文档将 `CUDA_R_16F`/`CUDA_R_16BF` 同精度行标为 **[DEPRECATED]**，本任务仍须对齐 cuSPARSE 官方 fp16/bf16 路径。



### 3. 算子约束

1. **输出 C 的 CSR 列索引须 sorted**（与 cuSPARSE 一致）；
2. **`opA`、`opB` 须均为 `NON_TRANSPOSE`**（cuSPARSE SpGEMM 不支持转置）；
3. **输入 A/B 索引须 sorted**（与 cuSPARSE 一致）；
4. **确定性**：对标 cuSPARSE bit-wise 确定性（float32）。

### 4. 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01 | 小矩阵 SpGEMM | cuSPARSE sample / 自行构造 |
| TC-02 | 空矩阵 / 零 nnz 边界 | 自行构造 |
| TC-03 | alpha/beta ≠ 1 | cuSPARSE 样例 |
| TC-04 | 高稀疏度大图 | 性能用例 |
| TC-05 | aclsparse 多阶段调用完整性 | `test/spmm/spmm_test.cpp` 模式 |

### 5. 性能要求

对标 **cuSPARSE SpGEMM（A100）**，NPU 性能须达到 GPU 参考实现的 **1.0 倍**及以上（即 ≥1.0×）。

**固定参考用例**（必测 3 组；在 **NVIDIA A100** 上实测 cuSPARSE 耗时填入「GPU 参考耗时」，取多次运行 **avg**，作为 NPU 验收基准）：

| 编号 | 场景 | 规模 (m×k×n) | nnz(A) | nnz(B) | dtype | GPU 参考耗时 (μs) | 达标要求 |
|------|------|--------------|--------|--------|-------|------------------:|----------|
| 01 | 标准 fp32 SpGEMM | 128×128×128 | 819 | 819 | fp32 | 3654 | NPU ≥ 1.0× |
| 02 | 零 nnz 边界 | 128×128×128 | 0 | 0 | fp32 | 13 | NPU ≥ 1.0× |
| 03 | α/β 非 1 | 128×64×128 | 163 | 163 | fp32 | 2404 | NPU ≥ 1.0× |

**泛化覆盖范围**（**另抽 200 组**用例；由下列维度组合抽样或网格扫描生成，须附完整用例列表；性能达标 ≥1.0×）：

| 维度 | 覆盖范围 |
|------|----------|
| 矩阵规模 | `(m,k,n)`：128～10⁴ 及以上；输出 `nnz(C)` 从 10² 到 10⁶+ |
| 输入稀疏度 | `nnz(A)`、`nnz(B)` 独立变化；稀疏率约 0.01%～10% |
| dtype | fp32（U1，必测）；fp16 / bf16 同精度（U2 / U3）实现后须纳入 200 组抽样 |
| α / β | `α=β=1` 外，含 `α≠1` 或 `β≠1` 至少 10 组 |
| 算法阶段 | `WorkEstimation` → `Compute`（→ `Copy`）全链路|
| 边界 | 空矩阵 / 零 nnz；高稀疏度大图（`nnz`≥10⁵）；输出 nnz 膨胀显著样本 5 组 |

### 6. 精度要求

1. 满足《[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)》
2. [ATK](https://gitcode.com/AscendTest/ATK) 双标杆（`cv_fused_double_benchmark`），NPU/同精度 CPU 最大相对误差比例 ≤ **2**，平均相对误差比例 ≤ **1.2**，均方根误差比例 ≤ **1.2**。
测试用例见[测试目录](./self_test_case/SpGEMM/)。

**说明**：ATK双标杆测试需要A100环境，请开发者自行准备。

### 7. 接口分层

```
aclsparseSpGEMM*（必选，C++ 验收基准）
    → Ascend C Kernel（必选）
```

### 8. 验收交付件

1, 自测用例、测试结果报告、测试步骤指导文档

2, 算子代码的私仓邀请链接、代码仓路径、分支、算子目录

### 9. 文档规范要求

1. 算子设计文档需根据[参考模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；
2. 自验证报告需要覆盖所有功能场景，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)，含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可清晰指导算子使用与测试；自验证报告含 aclsparse 测试、AscendOpTest/ATK 日志。
3. README 文档内容完整、规范，含 **aclsparse C++** 调用示例。

### 10. PR 合入路径

`https://gitcode.com/cann/ops-sparse/tree/master/src/spgemm`

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。
   ![环境截图](pics/yunkaifa.png)
2. 使用 hidevlab WebIDE 算力。
   ![环境截图](pics/zaixiankaifa1.png)
3. 如需额外环境资源，请联系昇腾小助手。


## 特别注意事项

1. **命名必须与 ops-sparse 现有 SpMM 一致**（`aclsparse` 前缀、驼峰 SpGEMM、目录 `src/spgemm`）；
2. SpGEMM 多阶段 workspace 须文档化，参考 cuSPARSE ALG1/2/3 内存策略；
3. 开发前阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)。

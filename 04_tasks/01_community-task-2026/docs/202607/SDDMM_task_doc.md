# 7月社区任务-SDDMM算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Ascend 950PR
- **开源仓地址**：https://gitcode.com/cann/ops-sparse
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C + C++（aclsparse 接口）
- **对标参考**：[cuSPARSE cusparseSDDMM §6.6.13](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsesddmm)

## 任务概述

参考 cuSPARSE **SDDMM**（Sampled Dense-Dense Matrix Multiplication），在昇腾 NPU 上实现 Ascend C Kernel + **aclsparse** C++ 接口。

**验收口径**：

- **C++**：`aclsparseSDDMM*` 与 cuSPARSE 对齐（命名参考 `aclsparseSpMM`）。

### 功能定义

$$
C = \alpha \cdot (op(A) \cdot op(B)) \circ spy(C_{pattern}) + \beta \cdot C
$$

- A ∈ R(m×k)、B ∈ R(k×n) 为**稠密**矩阵；
- C ∈ R(m×n) 为**稀疏 CSR**，仅更新 **已有非零位置** 的值；
- ⊙ 为 Hadamard 积，spy(C) 为 C 的结构掩码；
- 须支持 cuSPARSE 通用 α/β。

**GNN 典型用途**：在固定边结构（邻接）上计算注意力分数 (Q * K^T) ⊙ A。

---

## 核心开发要求及验收标准

### 1. C++ 层接口（**必选，对齐 cuSPARSE**）

参考 `aclsparseSpMMGetBufferSize` / `Preprocess` / `SpMM` 三阶段，新增：

```C
aclsparseStatus_t aclsparseSDDMMGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstDnMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSDDMMAlg_t alg,
    size_t *bufferSize);

aclsparseStatus_t aclsparseSDDMMPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstDnMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSDDMMAlg_t alg,
    void *externalBuffer);

aclsparseStatus_t aclsparseSDDMM(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstDnMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSDDMMAlg_t alg,
    void *externalBuffer);
```

**稠密矩阵描述符**：复用 `aclsparseCreateDnMat` / `aclsparseCreateConstDnMat`（同 SpMM）。

**工程目录**：

```
ops-sparse/src/sddmm/
```

### 2. 数据类型支持

**对标范围**：对齐 [cuSPARSE cusparseSDDMM §6.6.13](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsesddmm) 官方 dtype 表。

**列含义**：

- **A/B**：稠密 `matA`、`matB` 使用**相同** dtype（文档单列 `A/B`）；
- **C**：稀疏输出 `matC` 的 **values** dtype（结构由已有 nnz 决定）；
- **computeType**：实际计算/累加精度（`α`/`β` 亦为此类型）。

**文档 NOTE**：`CUDA_R_16F`、`CUDA_R_16BF` 一律视为混精度；混精行 **`computeType` 均为 `CUDA_R_32F`**。

**稀疏格式与索引**：

| 项目 | 本任务  |
|------|-----------|
| 稀疏 C | **CSR**；索引 int32 |
| 稠密 A/B | row/col major 须满足 cuSPARSE 性能推荐组合（见 §4） |
| 索引 | int32 |

SDDMM 为**全新算子**（ops-sparse 仓当前无 `src/sddmm/`），`aclsparseSDDMM*` + Kernel 均须从零交付。

##### 2.1 同精度（Uniform-precision）

稠密 A/B、稀疏 C values、`computeType` **四者相同**：

| ID | A/B / C / computeType | cuSPARSE 枚举 | 状态 |
|----|------------------------|---------------|------|
| U1 | fp32 | `CUDA_R_32F` | **待实现** |

##### 2.2 混精度（Mixed-precision）— CSR

与线上一致（`computeType` 均为 `CUDA_R_32F`）：

| ID | A/B（稠密 matA/matB） | C（稀疏 values） | computeType | 状态 |
|----|----------------------|------------------|-------------|------|
| M1 | `CUDA_R_16F` | `CUDA_R_32F` | `CUDA_R_32F` | **待实现** |
| M2 | `CUDA_R_16F` | `CUDA_R_16F` | `CUDA_R_32F` | **待实现** |

> M1：fp16 稠密输入、**fp32 稀疏输出 values**；M2：fp16 稠密输入、**fp16 稀疏输出 values**（fp32 累加）。


### 3. 算子约束

1. **仅更新 matC 已有 nnz 位置**；index 可 unsorted（cuSPARSE 允许）；
2. **opA/opB** 与 **DnMat order** 组合须满足 cuSPARSE 性能说明（row-major + NON_TRANSPOSE 等）；
3. **确定性**：对标 cuSPARSE bit-wise（float32）。


### 4. 功能验收用例

| 编号 | 场景 | 参考 |
|------|------|------|
| TC-01 | 基础 SDDMM，β=0 | cuSPARSE sample |
| TC-02 | 含 β·C 累加 | cuSPARSE |
| TC-03 | 1D 向量外积语义 | cuSPARSE / DGL 语义 |
| TC-04 | GNN 边特征 (E,1)·(1,E) | 自行构造 |
| TC-05 | 非连续 / 非方阵 | 自行构造 |
| TC-06 | 空 nnz | 边界 |


### 5. 性能要求

对标 **cuSPARSE SDDMM（A100）**，NPU 性能须达到 GPU 参考实现的 **1.0 倍**及以上（即 ≥1.0×）。

**固定参考用例**（必测 3 组；在 **NVIDIA A100** 上实测 cuSPARSE 耗时填入「GPU 参考耗时」，取多次运行 **avg**，作为 NPU 验收基准）：

| 编号 | 场景 | m×n | nnz | k | dtype | GPU 参考耗时 (μs) | 达标要求 |
|------|------|-----|-----|---|-------|------------------:|----------|
| 01 | fp32 基础 SDDMM，β=0 | 256×256 | 327 | 64 | fp32 | 1912 | NPU ≥ 1.0× |
| 02 | 含 β·C 累加 | 128×128 | 163 | 32 | fp32 | 1683 | NPU ≥ 1.0× |
| 03 | fp16 混精 M1 | 256×256 | 327 | 64 | A/B fp16，C fp32 | 1768 | NPU ≥ 1.0× |

**泛化覆盖范围**（**另抽 200 组**用例；由下列维度组合抽样或网格扫描生成，须附完整用例列表；性能达标 ≥1.0×）：

| 维度 | 覆盖范围 |
|------|----------|
| 矩阵规模 | `m`、`n`：10³～10⁵；`k`：16、64、128、256、512 |
| 稀疏度 | `nnz`：10⁴～10⁷；含「宽短矩阵」（`m≫n` 或 `n≫m`） |
| dtype | fp32（U1，必测）；fp16 混精 M1 / M2 实现后须纳入 200 组抽样 |
| α / β | 默认 `β=0`（必测）；含 `β≠0` 累加至少 10 组 |
| 布局 | row-major 推荐组合（§3）与非连续 stride 各 10 组 |
| 边界 | `nnz=0`；1D 外积语义；GNN 边特征 `(E,1)·(1,E)` |

### 6. 精度要求

满足《[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)》，[AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest) + 必要时 [ATK](https://gitcode.com/AscendTest/ATK) 双标杆L2（2 / 1.2 / 1.2）。

**真值**：以 **cuSPARSE CPU 标杆** 或双精度参考为准。


### 7. 接口分层

```
aclsparseSDDMM*（必选）
    → Ascend C Kernel（必选）
```

### 8. 验收交付件

1, 自测用例、测试结果报告、测试步骤指导文档

2, 算子代码的私仓邀请链接、代码仓路径、分支、算子目录

### 9. 合入路径

`https://gitcode.com/cann/ops-sparse/tree/master/src/sddmm`


## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。
   ![环境截图](pics/yunkaifa.png)
2. 使用 hidevlab notebook 算力。
   ![环境截图](pics/zaixiankaifa1.png)
3. 如需额外环境资源，请联系昇腾小助手。


## 特别注意事项

1. 命名遵循 `aclsparseSDDMM*`，与 ops-sparse 现有 SpMM 风格一致；
2. 设计文档按[社区模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)；
3. README 含 **aclsparse C++** 调用示例；
4. 自验证报告含 aclsparse 测试、AscendOpTest/ATK 日志。

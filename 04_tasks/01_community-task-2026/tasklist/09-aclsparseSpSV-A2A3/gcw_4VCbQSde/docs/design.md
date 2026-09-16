# 需求背景

## 需求来源

CANN 2026年9月社区任务：aclsparseSpSV 算子开发（A2/A3）。

## 背景介绍

### aclsparseSpSV 算子概述

aclsparseSpSV（Sparse Triangular Solve）用于求解稀疏三角线性方程组：

$$\text{op}(A) \cdot Y = \alpha \times X$$

其中 A 为稀疏三角方阵，X 为已知右端稠密向量，Y 为待求解向量，alpha 为缩放系数。op(A) 可为 A 本身（N）、A 的转置（T）或 A 的共轭转置（H）。

### 标杆接口参考

参考 cuSPARSE `cusparseSpSV` 系列接口语义，在昇腾 Atlas A2/A3（arch22）上实现。接口遵循 `BufferSize → Analysis → Solve → UpdateMatrix` 生命周期模型。

### 现状分析

当前 ops-sparse 仓库中尚无 arch22 平台的 SpSV Host/Kernel 路径，需从零建立。

# 需求分析

## 需求描述

在 Atlas A2/A3（arch22）上使用 Ascend C 实现 aclsparseSpSV 算子，交付公开 C++ 接口、Host 实现、Ascend C Kernel、UT/ST 及性能脚本。主计算运行于 NPU，禁止 CPU fallback。

## 需求拆解

1. 实现 createDescr / destroyDescr / bufferSize / analysis / solve / updateMatrix 六个公开接口
2. 支持 CSR、CSC、COO、SLICED_ELL 四种格式
3. 支持 FP32 与 complex64
4. 支持 LOWER/UPPER、UNIT/NON_UNIT、N/T/H、Host/Device pointer mode、base 0/1
5. 支持 GENERAL/DIAGONAL UpdateMatrix、X/Y 原地求解、未排序索引确定性规范化

## 接口定义

公开接口与 `include/cann_ops_sparse.h` 现有原型逐字一致：

```C
aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr);
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr);

aclsparseStatus_t aclsparseSpSV_bufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr, size_t *bufferSize);

aclsparseStatus_t aclsparseSpSV_analysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr, void *externalBuffer);

aclsparseStatus_t aclsparseSpSV_solve(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr);

aclsparseStatus_t aclsparseSpSV_updateMatrix(
    aclsparseHandle_t handle, aclsparseSpSVDescr_t spsvDescr,
    void *newValues, aclsparseSpSVUpdate_t updatePart);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 | dtype | 异常行为 |
|---|---|---|---|---|
| handle | 输入 | 调用流和 pointer mode | — | NULL 返回错误 |
| opA / alg | 属性 | 操作类型 / 算法 | — | 不支持枚举返回错误 |
| alpha | 输入 | 缩放系数 | FP32/complex64 | 指针或类型非法返回错误 |
| matA | 输入 | 稀疏三角方阵 | FP32/complex64, I32 索引 | 格式/属性/维度非法返回错误 |
| vecX / vecY | 输入 / 输出 | 右端向量 / 解向量 | FP32/complex64 | Solve 时 NULL/dtype 非法返回错误 |
| computeType | 属性 | 计算类型 | FP32/complex64 | 与 A/X/Y 不一致返回错误 |
| spsvDescr | 输入输出 | 跨阶段状态 | — | 生命周期不匹配返回错误 |
| bufferSize | 输出 | workspace 字节数 | size_t* | 空指针返回错误 |
| externalBuffer | 输入 | Device workspace | — | 空指针/未对齐返回错误 |
| newValues / updatePart | 输入 / 属性 | 更新 values / 更新模式 | FP32/complex64 | 不匹配返回错误 |

# 详细设计

## 算子分析

### 数学公式

$$\text{op}(A) \cdot Y = \alpha \times X$$

- LOWER 采用前代法（forward substitution），UPPER 采用回代法（backward substitution）
- UNIT 对角视为 1；NON_UNIT 缺失或零对角按 IEEE 754 传播 INF/NAN
- H 路径对 A 非零值执行共轭后再做转置求解

### 支持数据类型

| 数据类型 | values | 索引 | Golden |
|---|---|---|---|
| ACL_FLOAT | float32 | int32 | float64 |
| ACL_COMPLEX64 | complex64 | int32 | complex128 |

### 支持格式与属性组合

格式：CSR / CSC / COO / SLICED_ELL

属性：LOWER/UPPER × UNIT/NON_UNIT × N/T/H × HOST/DEVICE pointer × base 0/1 × FP32/complex64 × GENERAL/DIAGONAL update

## 算子实现

### 整体架构

分三层：公开 API 层 → Host 层（参数校验、生命周期、Tiling、格式转换调度）→ Ascend C Kernel 层（格式规范化、三角求解、UpdateMatrix）。公共 Host 逻辑与 arch22 差异层解耦，便于与其他架构共同维护。

### 生命周期

```
createDescr → bufferSize → analysis → solve → destroyDescr
                                │         ▲
                                └► updateMatrix ─┘
```

- **bufferSize**：根据格式/维度/dtype 计算 workspace 大小，vecX/vecY 可为 NULL
- **analysis**：Host 校验 + 缓存状态 + NPU Kernel 执行格式预处理（排序、去重、格式转换、提取对角）+ 计算 Tiling
- **solve**：校验 vecX/vecY 有效后，通过 stream 异步下发三角求解 Kernel
- **updateMatrix**：GENERAL 替换全部 values 并刷新状态；DIAGONAL 仅替换对角

### Host 侧设计

- 参数校验在 Host 侧同步完成，覆盖 NULL 检查、方阵校验、dtype 一致性、维度匹配、枚举合法性、workspace 有效性等，返回确定错误码
- 对未排序索引执行确定性规范化，原始 matA 只读，结果写入 workspace
- 所有非 CSR 格式在 Analysis 阶段统一转为 CSR 中间表示（由 NPU Kernel 执行）
- Tiling 采用行级分段策略，利用三角矩阵的依赖结构在行间进行并行调度

### Kernel 侧设计

- Analysis 阶段 Kernel：索引排序规范化、格式转换、对角提取
- Solve 阶段 Kernel：按行顺序（前代/回代）执行稀疏三角求解，支持 FP32 和 complex64 两个特化版本
- UpdateMatrix Kernel：执行 values 替换
- 原地求解（X/Y 同指针）由求解遍历顺序天然保证正确性
- complex64 H 路径在求解前对 values 执行共轭操作

### 代码组织

```
ops-sparse/
├── include/cann_ops_sparse.h         # 公开接口
├── sparse/spsv/                      # 公共 Host 逻辑
│   └── arch22/                       # arch22 Ascend C Kernel
├── test/spsv/arch22/                 # UT / ST / 性能脚本
└── test_cases/aclsparseSpSV_testCase/# 专项测试
```

## 支持硬件

| 芯片型号 | 支持 |
|---|---|
| Atlas A2 训练系列 910B3 / 910B4 | ✓ |
| Atlas A3 | ✓ |

## 算子约束限制

1. A 必须为方阵；fill mode 之外的三角项按语义忽略
2. computeType 必须与 A/X/Y dtype 一致；Device 索引 I32
3. 未排序/重复坐标由确定性预处理规范化；NON_UNIT 零对角传播 INF/NAN
4. externalBuffer 在 Analysis 至异步 Solve 完成前保持有效
5. 算法仅支持 ACL_SPARSE_SPSV_ALG_DEFAULT

# 可维可测分析

## 精度标准

- FP32 使用 float64 Golden，complex64 使用 complex128 Golden
- rtol=2^-10, atol=2^-16, A=1e-2；逐元素 |actual-golden| ≤ atol + rtol × |golden|
- 整体匹配率 ≥ 0.99；重复执行 bitwise deterministic

## 性能标准

| 编号 | m / nnz | dtype | 标杆 median_us (μs) | 目标 |
|---|---|---|---|---|
| P-01 | 65,536 / 524,288 | FP32, complex64 | 514,144–514,203 | ≥ 0.25× 标杆 |
| P-02 | 131,072 / 1,572,864 | FP32, complex64 | 1,228,206–1,230,583 | ≥ 0.25× 标杆 |
| P-03 | 262,144 / 3,932,160 | FP32, complex64 | 3,078,027–3,078,256 | ≥ 0.25× 标杆 |

## 内存标准

输入输出 > 500 MB 时额外内存不超过 GPU 的 50%；workspace 不超过 L2 Cache 容量。

## 兼容性分析

公共 Host 与 arch22 差异层解耦，接口与现有头文件一致，使用 CANN 9.1.0 及后续版本。

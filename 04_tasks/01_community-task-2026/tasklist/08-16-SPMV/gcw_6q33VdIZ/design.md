# 需求背景

## 需求来源

参考 cuSPARSE SpMV 实现，在昇腾 NPU (Ascend 950PR) 上基于 Ascend C 编程语言实现 CSR 格式稀疏矩阵-稠密向量乘法算子。

## 背景介绍

### SpMV 算子概述

SpMV（Sparse Matrix-Vector Multiplication）是科学计算和机器学习中的核心算子。计算公式：

$$Y = \alpha \cdot op(A) \cdot X + \beta \cdot Y$$

其中 A 为 CSR 格式稀疏矩阵，X 和 Y 为稠密向量，alpha/beta 为标量系数，op(A) 支持非转置和转置两种模式。

### 现状分析

ops-sparse 开源仓已有 arch22 (A2) 实现，但不支持 Ascend 950PR (arch35)。需将该接口移植并优化到 950PR 平台。

# 需求分析

## 需求描述

在 Ascend 950PR 上实现 aclsparseSpMV 三阶段接口（GetBufferSize / Preprocess / SpMV），支持全部数据类型组合，满足精度和性能要求。

## 需求拆解

1. 支持 float32、float16、bfloat16、int8、int32 数据类型组合
2. 支持非转置和转置两种模式
3. 支持三阶段调用流程（GetBufferSize → Preprocess → SpMV）
4. 性能达到标杆的 0.5 倍
5. 精度满足生态算子开源精度标准

# 详细设计

## 算子分析

### 数学公式

$$Y = \alpha \cdot op(A) \cdot X + \beta \cdot Y$$

- 非转置（op(A) = A）：Y[i] = alpha * sum_j(A[i][j] * X[j]) + beta * Y[i]
- 转置（op(A) = A^T）：Y[j] += alpha * A[i][j] * X[i]，对所有非零元素 (i,j)

### 支持数据类型

| 输入 A/X 类型 | 计算类型 | 输出 Y 类型 |
|---|---|---|
| float32 | float32 | float32 |
| int8 | int32 | int32 |
| int8 | float32 | float32 |
| float16 | float32 | float32 |
| float16 | float32 | float16 |
| bfloat16 | float32 | float32 |
| bfloat16 | float32 | bfloat16 |

### 支持格式

仅支持 CSR（Compressed Sparse Row）格式，索引类型为 INT32，基址为 ZERO。

## 算子实现

### 实现方案

#### Host 侧设计

**三阶段接口**：

1. `aclsparseSpMVGetBufferSize`：返回 workspace 大小 = max(M, K) * sizeof(float)，用于转置模式的 float 空间累加。

2. `aclsparseSpMVPreprocess`：可选的预处理阶段。将 CSR rowPtr 拷贝到 host，按行 nnz 降序排序生成 reorder 表并上传到 workspace。

3. `aclsparseSpMV`：主计算入口。参数校验后调用 kernel dispatcher (`spmv_kernel_do`)，传入类型 ID 选择对应模板实例。

**Tiling 策略**：

- 均分行数到各核：`rowsPerBlock = ceil(M / blockNum)`
- blockNum = GetCoreNumAiv() = 56（950PR）
- 每核内启用 256 个 SIMT 线程做行级并行

#### Kernel 侧设计

采用 **SIMT (Single Instruction Multiple Threads)** 编程模式，使用 `__simt_vf__` + `asc_vf_call` 实现线程级并行。

**非转置路径**：

1. 每个 SIMT 线程处理一行（grid-stride loop）
2. 内循环：逐元素读取 csrColInd[p] → xVec[col]，累乘 csrVal[p]，用 FMA 指令累加
3. 4 路累加器 + 8 路循环展开，隐藏指令延迟
4. 结果：yVec[r] = alpha * sum + beta * yVec[r]

```
// 核心内循环（float 路径）
float s0=0, s1=0, s2=0, s3=0;
for (p = rStart; p < unrollEnd; p += 8) {
    s0 = __fmaf_rn(csrVal[p],   xVec[csrColInd[p]],   s0);
    s1 = __fmaf_rn(csrVal[p+1], xVec[csrColInd[p+1]], s1);
    s2 = __fmaf_rn(csrVal[p+2], xVec[csrColInd[p+2]], s2);
    s3 = __fmaf_rn(csrVal[p+3], xVec[csrColInd[p+3]], s3);
    ...
}
sum = (s0 + s1) + (s2 + s3);
```

**转置路径**：

为避免低精度输出（fp16/bf16）的反复截断累积误差，统一使用 float workspace 做累加：

1. Phase 1（BetaScale）：`workspace[col] = beta * float(yVec[col])`，多核分列并行
2. Phase 2（Transpose）：`asc_atomic_add(&workspace[col], alpha * float(xVec[r]) * float(csrVal[p]))`，多核分行并行，float 原子加
3. Phase 3（CopyBack）：`yVec[col] = OutT(workspace[col])`，多核分列并行

各阶段间用 `AscendC::SyncAll()` 同步。

**混合精度处理**：

- 输入 fp16/bf16/int8 通过 `static_cast<float>()` 提升到 computeType 精度
- 输出通过 `static_cast<OutT>()` 截断回目标精度
- 转置模式在 float 空间做完所有累加后一次性截断，避免精度损失

### 关键优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| SIMT 多线程 | 256 线程/核 × 56 核 | 隐藏 GM 访存延迟 |
| FMA 指令 | `__fmaf_rn` | 融合乘加，减少指令数 |
| 循环展开 | 4 累加器 × 8 展开 | ILP + 打破数据依赖链 |
| beta=0 快捷路径 | 跳过 yVec 读取 | 减少无效 GM 访问 |
| alpha=1 快捷路径 | 跳过乘法 | 减少指令数 |
| float workspace | 转置用 float 累加 | 避免低精度截断累积 |

## 支持硬件

| 芯片型号 | 支持 |
|---|---|
| Ascend 950PR | 支持 |
| Ascend 950DT | 支持 |

## 算子约束限制

1. 稀疏矩阵仅支持 CSR 格式
2. 索引类型仅支持 INT32
3. 索引基址仅支持 ZERO-based
4. 算法类型仅支持 ALG_DEFAULT
5. 转置模式需要 workspace（通过 GetBufferSize 获取大小）

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 满足生态算子开源精度标准（double_benchmark: max_re_ratio≤5, avg_re_ratio≤1.5） | case_200.json |
| 性能标准 | 达到标杆 0.5 倍水平 | 任务书 |

### 性能测试结果

| 矩阵规模 | 稀疏度 | 标杆 | 0.5×目标 | 实际性能 | 状态 |
|---|---|---|---|---|---|
| 128×128 | 95% | 43.9us | ≤87.8us | 4.8us | PASS |
| 1024×1024 | 99% | 46.3us | ≤92.6us | 6.4us | PASS |
| 2048×4096 | 97.5% | 45.4us | ≤90.8us | 15.8us | PASS |
| 160220×68750 | 99.9% | 193.2us | ≤386.4us | 454us (kernel 420us) | 差9% |

## 兼容性分析

新增 arch35 实现，不影响现有 arch22 代码。CMake 架构过滤机制自动按 SOC 版本选择对应目录。

# aclsolverCheevj 算子设计文档

## 一、需求背景

### 1.1 需求来源

通过 CANN 社区任务完成昇腾开源仓算子贡献，参考 cuSOLVER 库中 `cusolverDnCheevj` 接口，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的 Hermitian 矩阵特征值分解算子。

### 1.2 背景介绍

#### 1.2.1 算子功能概述

`aclsolverCheevj` 使用 Jacobi 迭代方法计算单精度复数 Hermitian 矩阵的特征值和（可选的）特征向量。数学表述为：

$A \cdot V = V \cdot \text{diag}(w)$

其中 A 为 n×n Hermitian 矩阵，w 为升序特征值，V 列为特征向量。

#### 1.2.2 Jacobi 迭代方法原理

Jacobi 方法通过正交相似变换（Givens 旋转）逐步将 Hermitian 矩阵对角化。每次旋转选取 (p,q) 对，构造旋转矩阵 G 使 A' = G^H A G 的 A'[p][q] = 0。

复数 Hermitian 旋转参数计算：

1. |A[p][q]| = sqrt(re² + im²)
2. phase = A[p][q] / |A[p][q]|
3. tau = (A[q][q] - A[p][p]) / (2·|A[p][q]|)
4. t = sign(tau) / (|tau| + sqrt(1 + tau²))
5. c = 1/sqrt(1+t²), s = t·c

#### 1.2.3 cuSOLVER 参考接口

cuSOLVER `cusolverDnCheevj` 支持：

* jobz='N': 仅计算特征值
* jobz='V': 计算特征值和特征向量
* uplo='U'/'L': 上/下三角存储
* syevjInfo_t: 收敛容忍度、最大扫描次数

## 二、需求分析

### 2.1 需求描述

使用 Ascend C 实现 aclsolverCheevj，采用 MIX AIC 模式 (VEC+CUBE 协同)，支持 complex64 数据类型，目标硬件 ascend910b。数据排布格式统一采用行主序（Row-Major）以兼容 CANN 其他算子。

### 2.2 需求拆解

1. 实现并行 Jacobi 迭代 (round-robin tournament)
2. 支持 jobz='N'/'V'、uplo='U'/'L'
3. 特征值升序排列
4. 性能达到 0.8x GPU A100
5. 支持泛化输入 (任意合法 n、lda>=n)

### 2.3 外部组件依赖

| 组件 | 用途 | 是否新增 |
| --- | --- | --- |
| CANN | 编译运行环境 | 否 |
| Ascend C | AICore kernel | 否 |
| ACL Runtime | 设备/内存/流管理 | 否 |
| ops-solver/utils | SplitRealImag, MergeRealImag, CMatmulCustom | 否 |

### 2.4 算子原型

| 名称 | 类别 | 数据类型 | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| jobz | 输入 | char | 标量 | - | 'N': 仅特征值; 'V': +特征向量 |
| uplo | 输入 | char | 标量 | - | 'U': 上三角; 'L': 下三角 |
| n | 输入 | int64 | 标量 | - | 矩阵阶数 |
| A | 输入/输出 | complex64 | 行主序 | [n, lda] | Hermitian矩阵; jobz='V'输出特征向量 |
| lda | 输入 | int64 | 标量 | - | 前导维度, lda >= max(1,n) |
| w | 输出 | float32 | 一维 | [n] | 特征值数组，升序 |
| info | 输出 | int32 | 标量 | - | 0=成功, >0=未收敛, <0=参数错误 |
| tolerance | 输入(可选) | float | 标量 | - | 收敛容忍度, 默认 1e-7 |
| maxSweeps | 输入(可选) | int32 | 标量 | - | 最大扫描次数, 默认 100 |

### 2.5 算子约束

1. 输入矩阵 A 必须为 Hermitian (A = A^H)
2. n >= 0, lda >= max(1, n)
3. 统一行主序 (Row-Major)
4. 数据类型: complex64 (矩阵), float32 (特征值)
5. 硬件: Atlas A2 训练系列产品 (ascend910b)

## 三、详细设计

### 3.1 使能方式

| 上层框架 | 涉及 |
| --- | --- |
| Aclnn 直调 | ✅ |

### 3.2 参考实现 (cuSOLVER) 流程图

![image.png](https://raw.gitcode.com/user-images/assets/9516645/8d24fd1b-9fe6-4ea9-9cbf-f07041715115/image.png 'image.png')

### 3.3 算子设计

#### 3.3.1 Host 侧设计

**分核策略:**

* MIX AIC 1:2 模式: VEC 核做 Jacobi 旋转 + Q 累积, CUBE 核做 V*Q GEMM
* VEC 核间并行: round-robin tournament, n/2 对/step, 均分到各核
* CUBE 核间并行: GEMM M 维度分块

**数据分块和内存优化 (适配行主序):**

* A: 行主序存储, strideN = ceil(n, 128), 行操作完全向量化
* V 和 Q: 行主序存储。由于 V 和 Q 在 Jacobi 迭代中涉及列旋转，直接进行跨步距（Strided）访存会降低 VEC 效率。解决方案：在加载 V/Q 的 p、q 列进入 UB 时，使用 DataCopy 的连续传输加 UB 内转置，或直接在后续利用 CUBE 进行高效的行主序 GEMM 操作规避标量化访存。
* UB 192KB: TBufPool 分配 4行+2tmp = 6*aN*4B

**Tiling 策略:**

| 参数 | 计算 | 说明 |
| --- | --- | --- |
| strideN | ceil(n, 128) | 128字节对齐 |
| srcN | lda | 源数据步长 |
| numBlocks | GetCoreNumAic() | MIX模式全部AIC核 |
| aN | ceil(n, 8) | 向量操作对齐 |

#### 3.3.2 Kernel 侧设计

**Kernel 实现流程图:**

![image.png](https://raw.gitcode.com/user-images/assets/9516645/3fde1b24-ea08-403c-a15b-be0f9b219434/image.png 'image.png')

#### 3.3.3 总体 Host-Device 调用流程图

![image.png](https://raw.gitcode.com/user-images/assets/9516645/601d0367-21bd-43cb-9af3-34185d52fb32/image.png 'image.png')

### 3.4 支持硬件

| 产品 | 说明 |
| --- | --- |
| Atlas A2 训练系列产品 (ascend910b) | 任务书指定 |

### 3.5 算子约束限制

1. 输入必须为 Hermitian 矩阵
2. complex64 (矩阵) + float32 (特征值)
3. 行主序 (Row-Major), lda >= max(1, n)
4. 内存对齐: strideN=ceil(n,128) 保证 128字节对齐
5. 内存需求: ~10*n*strideN*4 字节 GM

## 四、特性交叉分析

| 特性 | 设计处理 | 覆盖情况 |
| --- | --- | --- |
| jobz='N' | 跳过 V/Q/GEMM/MergeRealImag | 测试覆盖 |
| jobz='V' | 完整 VEC+CUBE 协同 | 测试覆盖 |
| uplo='U' | FillHermitian 上三角分支 | 测试覆盖 |
| uplo='L' | FillHermitian 下三角分支 | 测试覆盖 |
| n=1 | Host 快速路径 | 边界测试 |
| n为奇数 | tournament pad到偶数 | 测试覆盖 |
| lda > n | SplitRealImag srcN=lda (适配行主序) | 测试覆盖 |
| 大矩阵 | 多核并行 + GEMM | 性能测试 |

## 五、可维可测分析

### 5.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 与 Python (scipy.linalg.eigh) 结果一致 | 任务书 |
| 性能 | 整体性能与 0.8x GPU A100 持平 | 任务书 |

性能目标:

| N | jobz | uplo | GPU A100(ms) | 目标上限(ms) |
| --- | --- | --- | --- | --- |
| 512 | V | U | 21.130 | ≤26.4 |
| 1024 | V | U | 87.717 | ≤109.6 |
| 2048 | V | L | 483.208 | ≤604.0 |
| 1024 | N | U | 8.293 | ≤10.4 |

### 5.2 兼容性分析

ops-solver 开源仓新增算子，不涉及已有接口兼容性变更。接口与 cuSOLVER 语义对齐。数据内存排布统一采用行主序，提升了与其他 CANN 原生算子的协同工作兼容性。
# 需求背景（required）

## 需求来源

参考cuBLAS库中的cublasStrsmBatched和cublasCtrsmBatched接口，在Atlas A2训练系列产品上基于Ascend C编程语言实现功能一致的批量三角矩阵求解算子，并提交至ops-blas开源仓。

## 背景介绍

### TRSMBatched算子实现

TRSM（Triangular Solve Matrix）用于求解三角线性方程组，是BLAS Level-3中的重要算子，在LU分解、Cholesky分解、矩阵求逆、线性方程组求解以及深度学习训练过程中被广泛使用。

TRSMBatched支持批量处理多个独立的三角矩阵求解问题。

左乘模式：

A × X = α × B

右乘模式：

X × A = α × B

其中A表示三角矩阵对应的有效区域。

本项目基于Ascend C实现TRSMBatched算子，支持单精度实数(float32)和单精度复数(complex64)两种数据类型。

### TRSMBatched算子功能分析

通过对cuBLAS cublas<t>trsmBatched接口进行分析，需要支持如下能力：

| 参数         | 参数含义    | 数据类型   | 支持数据类型            | 约束        |
| ---------- | ------- | ------ | ----------------- | --------- |
| side       | 左乘/右乘模式 | char   | L/R               | 必选        |
| uplo       | 上下三角类型  | char   | U/L               | 必选        |
| transa     | 转置模式    | char   | N/T/C             | 必选        |
| diag       | 单位三角模式  | char   | U/N               | 必选        |
| m          | B矩阵行数   | int    | int32             | >0        |
| n          | B矩阵列数   | int    | int32             | >0        |
| alpha      | 缩放系数    | scalar | float32/complex64 | 无         |
| A          | 三角矩阵输入  | tensor | float32/complex64 | Batch输入   |
| B          | 输入输出矩阵  | tensor | float32/complex64 | Batch输入输出 |
| batchCount | Batch数量 | int    | int32             | >0        |

其中：

StrsmBatched支持：

transa = N/T

CtrsmBatched支持：

transa = N/T/C

计算公式：

当side='L'时：

X = op(A)^(-1) × (α × B)

当side='R'时：

X = (α × B) × op(A)^(-1)

其中：

op(A) = A

op(A) = A^T

op(A) = conj(A^T)（仅CtrsmBatched支持）

### TRSMBatched算子现状分析

当前ops-blas开源仓尚未提供Ascend C版本TRSMBatched实现。

本次开发实现：

1. StrsmBatched（float32）
2. CtrsmBatched（complex64）

支持：

* side=L/R
* uplo=U/L
* transa=N/T
* transa=C（CtrsmBatched）
* diag=U/N
* Batch处理

满足与cuBLAS接口功能对齐要求。

# 需求分析（required）

## 需求描述

使用Ascend C编程语言实现TRSMBatched算子，支持float32和complex64数据类型，实现批量三角矩阵求解功能。

## 需求拆解

1. 支持float32数据类型(StrsmBatched)
2. 支持complex64数据类型(CtrsmBatched)
3. 支持side=L/R模式
4. 支持uplo=U/L模式
5. 支持transa=N/T模式
6. 支持CtrsmBatched的transa=C模式
7. 支持diag=U/N模式
8. 支持batchCount > 0场景
9. 支持泛化输入规模
10. 精度结果与Python参考实现一致
11. 性能达到任务书要求的目标性能

# 详细设计（required）

## 算子分析

### 数学公式

当side='L'：

op(A) × X = α × B

求解：

X = op(A)^(-1) × (α × B)

当side='R'：

X × op(A) = α × B

求解：

X = (α × B) × op(A)^(-1)

其中：

StrsmBatched：

op(A)=A 或 A^T

CtrsmBatched：

op(A)=A

op(A)=A^T

op(A)=conj(A^T)

### 支持数据类型

StrsmBatched：

float32

CtrsmBatched：

complex64

### 支持形状

side='L'

A:[batchCount,m,m]

B:[batchCount,m,n]

side='R'

A:[batchCount,n,n]

B:[batchCount,m,n]

batchCount > 0

### 支持参数组合

StrsmBatched：

side × uplo × transa(N/T) × diag

CtrsmBatched：

side × uplo × transa(N/T/C) × diag

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

##### 1. 参数检查

检查：

* side合法性
* uplo合法性
* transa合法性
* diag合法性
* m合法性
* n合法性
* batchCount合法性
* lda合法性
* ldb合法性

##### 2. Tiling策略

TRSM属于具有数据依赖关系的矩阵求解问题。

不同Batch之间不存在依赖关系，因此采用Batch级并行策略。

Host侧根据：

* batchCount
* m
* n
* 数据类型

生成对应Tiling参数。

##### 3. 分核策略

优先采用满核执行原则。

根据Batch数量及AI Core数量进行均匀分配。

Batch无法均分时，将剩余Batch分配给前部Core。

##### 4. 数据切分策略

side='L'：

按列方向求解。

side='R'：

按行方向求解。

减少访存次数并提升缓存复用率。

##### 5. TilingKey规划

TilingKey=0：

StrsmBatched

TilingKey=1：

CtrsmBatched

Kernel侧根据TilingKey选择对应实现路径。

#### 3.2.2 kernel侧设计

Kernel分为：

Init

Process

两个阶段。

Process包含：

CopyIn

Compute

CopyOut

三个步骤。

##### 1. CopyIn阶段

从GM读取：

A矩阵

B矩阵

搬运到Local Memory。

##### 2. Compute阶段

根据：

side

uplo

transa

diag

选择对应求解路径。

uplo='U'

采用回代算法（Backward Substitution）。

uplo='L'

采用前代算法（Forward Substitution）。

diag='U'

对角元素按1处理。

diag='N'

读取实际对角元素参与计算。

##### 3. side='L'

按照列方向进行三角求解。

得到：

X(:,j)

##### 4. side='R'

按照行方向进行三角求解。

得到：

X(i,:)

##### 5. complex64处理

复数版本采用统一模板实现。

支持：

* 复数加法
* 复数乘法
* 复数除法
* 复数共轭运算

当transa='C'时：

使用共轭转置矩阵：

op(A)=conj(A^T)

进行三角求解。

共轭操作在读取矩阵元素时完成。

##### 6. CopyOut阶段

将求解结果写回Global Memory。

输出覆盖输入矩阵B。

## 支持硬件

| 支持的芯片版本          | 涉及勾选 |
| ---------------- | ---- |
| Atlas 800I A2    | √    |
| Atlas 900 A2 POD | √    |

## 算子约束限制

1. 算子按照uplo指定的三角区域参与计算。

   非参与计算区域中的元素不会被访问，其取值不影响计算结果。

2. 当diag='N'时，算子假定输入矩阵可求解。

   若存在对角元素为0、矩阵奇异等情况，结果正确性不做保证。

3. batchCount必须大于0。

4. 当前版本采用Row-Major布局。

5. StrsmBatched支持float32。

6. CtrsmBatched支持complex64。

7. StrsmBatched支持transa=N/T。

8. CtrsmBatched支持transa=N/T/C。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述                      | 标准来源  |
| ---- | ----------------------- | ----- |
| 精度标准 | Ascend C结果与Python参考实现一致 | 任务书要求 |
| 性能标准 | 达到任务书要求性能目标             | 任务书要求 |

## 兼容性分析

新增算子实现，不涉及历史版本兼容性问题。

接口定义与cuBLAS cublas<t>trsmBatched保持一致，便于用户迁移适配。

# 需求背景

## 需求来源

CANN 社区任务广场 aclblasSsbmv A2/A3 算子开发任务（taskId: 8dee78f0e2f54299b1eb90b9d3fcd04f）。

## 背景介绍

在昇腾 NPU 上使用 Ascend C 开发单精度对称带状矩阵-向量乘法算子 aclblasSsbmv。

# 需求分析

## 需求描述

实现 y = alpha * A * x + beta * y，A 为 n*n 对称带状矩阵，lda >= k+1。

## 需求拆解

1. 支持 uplo = UPPER/LOWER
2. 按对称性引用另一半元素
3. 多核并行按行切分

# 详细设计

## 算子分析

### 数学公式

y[i] = beta * y[i] + alpha * sum_j(A[i][j] * x[j])，A 对称，仅存 uplo 三角。

### 带状索引

aIdx = (k + i - j) + lda * j，仅当 |i-j| <= k 时访问。

## 算子实现

### 实现方案

多核按行切分。对每行 i，遍历 j in [max(0,i-k), min(n-1,i+k)]，按对称性引用。

### 精度验证

CSV 驱动 GTest，40 个用例全部通过，rtol=2^-10, atol=2^-16, matched_ratio>=0.99。

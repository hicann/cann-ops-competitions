# 需求背景

## 需求来源

CANN 社区任务广场 aclblasSspr2 A2/A3 算子开发任务（taskId: 11705182f9ec455d907d60a540bb10d5）。

## 背景介绍

在昇腾 NPU 上使用 Ascend C 开发单精度对称矩阵秩-2更新算子 aclblasSspr2。

# 需求分析

## 需求描述

实现 A = alpha*x*y^T + alpha*y*x^T + A，A 为对称矩阵，packed 存储。

## 需求拆解

1. 支持 uplo = UPPER/LOWER
2. packed 存储格式
3. 多核并行按列切分

# 详细设计

## 算子分析

### 数学公式

A[i][j] += alpha * (x[i]*y[j] + x[j]*y[i])

### packed 索引

UPPER: idx = col*(col+1)/2 + row (row <= col)
LOWER: idx = col*(2*n-col+1)/2 + (row - col) (row >= col)

## 算子实现

### 实现方案

多核按列切分。用 GetValue 读 x/y，DataCopyPad 写回 A 避免多核竞争。

### 精度验证

CSV 驱动 GTest，43 个用例全部通过，rtol=2^-10, atol=2^-16, matched_ratio>=0.99。

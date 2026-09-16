# 需求背景

## 需求来源

CANN 社区任务广场 aclblasSgbmv A2/A3 算子开发任务（taskId: 57fcc5dad709456caae4b2b6d6003371）。

## 背景介绍

在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3）上使用 Ascend C 编程语言开发单精度带状矩阵-向量乘法算子 aclblasSgbmv。

# 需求分析

## 需求描述

实现 y = alpha * op(A) * x + beta * y，其中 A 为 m*n 带状矩阵，列主序带状存储，lda >= kl+ku+1。

## 需求拆解

1. 支持 trans = N/T/C 三种操作
2. 支持 incx/incy 正负步长
3. 支持 alpha=0 / beta=0/1 快速路径
4. 多核并行按行切分

# 详细设计

## 算子分析

### 数学公式

y[i] = beta * y[i] + alpha * sum_j(A[row][col] * x[j])

### 带状索引

aIdx = (ku + row - col) + lda * col

## 算子实现

### 实现方案

#### host 侧设计

参数校验后计算 tiling（useCoreNum = min(n, 8)），kernel 直调。

#### kernel 侧设计

多核按行切分（vecIdx = GetBlockIdx()）。incx=1/incy=1 用 DataCopy 批量搬运，否则用 GetValue 逐元素。用 TQue + Mul/ReduceSum 做 dot product。

### 精度验证

CSV 驱动 GTest，51 个用例全部通过，rtol=2^-10, atol=2^-16, matched_ratio>=0.99。

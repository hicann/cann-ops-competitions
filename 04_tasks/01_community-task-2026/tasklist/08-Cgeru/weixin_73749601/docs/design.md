# 需求背景（required）

## 需求来源

昇腾社区任务广场——8月社区任务-aclblasCgeru算子开发（950）

## 背景介绍

aclblasCgeru 是单精度复数无共轭秩-1 更新算子，基于 Ascend C 在 Ascend 950PR 上实现，合入 ops-blas 开源仓。

### 数学公式

A = alpha * x * y^T + A

其中 x 为 m 维复数列向量，y 为 n 维复数行向量，A 为 m×n 列主序复数矩阵。无共轭（与 Cgerc 区分）。

# 详细设计（required）

## 实现方案

### Kernel 侧

SIMT 范式，2 条计算路径：
- **GM**：列并行（grid-stride over columns），每列先算 ay = alpha * y[col]，再逐行 A[i,col] += x[i] * ay
- **UB**：incx==1 且 m ≤8192 时缓存 x 到 `__ubuf__`

复数 FMA：`A[i,j] += (xR*ayR - xI*ayI) + i*(xR*ayI + xI*ayR)`

alpha=(0,0) 时提前退出（no-op，不写 A）。

### Host 侧

参数验证 → alpha=(0,0) 短路 → tiling → kernel launch。需在 include/cann_ops_blas.h 新增 aclblasCgeru 声明。

## 支持硬件

| 芯片 | 支持 |
|---|---|
| Ascend 950PR | √ |

代码仓：https://gitcode.com/weixin_73749601/aclblas-cgeru

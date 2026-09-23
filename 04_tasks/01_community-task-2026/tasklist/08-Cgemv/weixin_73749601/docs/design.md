# 需求背景（required）

## 需求来源

昇腾社区任务广场——8月社区任务-aclblasCgemv算子开发（950）

## 背景介绍

aclblasCgemv 是单精度复数矩阵-向量乘算子，基于 Ascend C 在 Ascend 950PR 上实现，合入 ops-blas 开源仓。

### 数学公式

y = alpha * op(A) * x + beta * y

其中 op(A) 根据 trans 取 A（N）、A^T（T）或 A^H（C，共轭转置）。A 为 m×n 列主序复数矩阵。

# 详细设计（required）

## 实现方案

### Kernel 侧

SIMT 范式（`__simt_vf__` + `asc_vf_call`），6 条计算路径：
- **N-GM / N-UB**：行并行，每线程计算一个输出元素（行点积）
- **T-GM / T-UB**：列并行，每线程计算一个输出元素（列点积）
- **C-GM / C-UB**：同 T 但对 A 取共轭（虚部取反）

复数 FMA：`accR += aR*xR - aI*xI; accI += aR*xI + aI*xR`

UB 路径：incx==1 且 x 向量 ≤8192 复数元素（64KB）时缓存到 `__ubuf__`。

### Host 侧

参数验证（trans 枚举/m/n/lda/inc/alpha/beta/指针）→ tiling 计算（核数=min(⌈outDim/128⌉, aivCores)）→ kernel launch。

## 支持硬件

| 芯片 | 支持 |
|---|---|
| Ascend 950PR | √ |

代码仓：https://gitcode.com/weixin_73749601/aclblas-cgemv

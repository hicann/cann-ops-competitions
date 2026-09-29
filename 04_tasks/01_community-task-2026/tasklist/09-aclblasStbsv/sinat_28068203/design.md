# 需求背景（required）

## 需求来源

昇腾社区任务“9月社区任务-aclblasStbsv算子开发(A2/A3)”。任务要求在 Atlas A2/A3 系列产品上以 Ascend C 实现单精度实数三角带状求解接口 `aclblasStbsv`，并对齐 cuBLAS `cublasStbsv` 的参数和计算语义。

任务详情：[9月社区任务-aclblasStbsv算子开发(A2/A3)](https://www.hiascend.com/activities/task-center/details/29c59d7ef9fa4aa8ad26b82969c5e2b6?menu=tasks)

## 背景介绍

`aclblasStbsv` 属于 BLAS Level 2 三角带状求解算子。矩阵只存储指定半带宽范围内的系数，避免按稠密矩阵保存；右端向量和求解结果共用 `x` 缓冲区。实现需保留 handle 绑定流的异步调用方式，并支持 A2、A3 的 arch22 内核路径。

### 功能现状与目标

目标接口的参数顺序和运算语义与 `cublasStbsv` 对齐。输入矩阵 `A` 使用列主序带状存储，`x` 初始保存右端向量 `b`，算子原地求解并写回 `x`。实数输入下，`ACLBLAS_OP_C` 与 `ACLBLAS_OP_T` 等价。

| 参数 | 含义 | 类型 | 约束 |
| --- | --- | --- | --- |
| `handle` | BLAS 上下文及 stream | `aclblasHandle_t` | 不可为空 |
| `uplo` | 选择上三角带或下三角带 | `aclblasFillMode_t` | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| `trans` | 选择 `A`、`Aᵀ` 或 `Aᴴ` | `aclblasOperation_t` | `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 或 `ACLBLAS_OP_C` |
| `diag` | 指定对角线是否为单位对角 | `aclblasDiagType_t` | `ACLBLAS_NON_UNIT` 或 `ACLBLAS_UNIT` |
| `n` | 方阵阶数及向量逻辑长度 | `int` | `n ≥ 0`；`n=0` 为 no-op |
| `k` | 半带宽，主对角线上方或下方的带数 | `int` | `k ≥ 0`；`k=0` 为对角矩阵 |
| `A` | 带状矩阵数据 | `const float*` | 设备内存；`n>0` 时不可为空 |
| `lda` | 带状矩阵的列间距 | `int` | `lda ≥ max(1, k+1)` |
| `x` | 输入右端向量并接收解 | `float*` | 设备内存；`n>0` 时不可为空 |
| `incx` | 向量逻辑元素步长 | `int` | 非零，可为正或负 |

接口声明：

```cpp
aclblasStatus_t aclblasStbsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    int k,
    const float* A,
    int lda,
    float* x,
    int incx);
```

# 需求分析（required）

## 需求描述

使用 Ascend C kernel 在 Atlas A2/A3（arch22）上求解三角带状线性方程组：

\[
\operatorname{op}(A)x=b
\]

其中 `b` 初值存放在 `x` 中，结果覆盖写回 `x`。支持单精度实数、上下三角带、无转置/转置/共轭转置、单位/非单位对角以及带符号的向量步长。接口参数错误应返回与 ops-blas 公共状态码定义一致的错误状态；有效调用通过 handle 的 stream 异步提交。

## 需求拆解

1. 在公共头文件声明 `aclblasStbsv`，并在 arch22 目录提供 Host 入口与 A2/A3 kernel 实现。
2. 按列主序三角带格式读取矩阵，正确处理 `uplo`、`trans`、`diag`、`lda` 和正负 `incx`。
3. 对空维度、对角带宽、非法枚举、非法维度、空指针和非法步长提供任务书规定的行为。
4. 使内核沿三角依赖方向完成前代或回代，不读取带外矩阵元素；`UNIT` 对角模式不读取对角元素。
5. 按任务书的精度、性能和 A2/A3 适配要求准备可复现的自验证用例与步骤说明。

# 详细设计（required）

## 算子分析

### 数学公式

`op(A)` 为 `A`、`Aᵀ` 或 `Aᴴ`。对实数 `float32`，`Aᴴ=Aᵀ`。算子按有效三角方向逐个求解逻辑分量 `i`：

\[
s_i=b_i-\sum_{j\in D_i}\operatorname{op}(A)_{ij}x_j,
\qquad
x_i=\begin{cases}
s_i,&\text{diag=UNIT},\\
s_i/\operatorname{op}(A)_{ii},&\text{diag=NON\_UNIT}.
\end{cases}
\]

`D_i` 只包含三角带范围内且已完成求解的分量。有效下三角按 `i=0…n-1` 前向求解；有效上三角按 `i=n-1…0` 反向求解。

### 数据排布

`A` 按列主序束带格式存储，每列的有效带宽为 `k+1`，可由 `lda` 提供额外 padding。采用从 0 开始的下标：

- 上三角带中，`A(i,j)`（`max(0,j-k) ≤ i ≤ j`）位于 `A[k+i-j+j*lda]`。
- 下三角带中，`A(i,j)`（`j ≤ i ≤ min(n-1,j+k)`）位于 `A[i-j+j*lda]`。
- 转置访问通过交换逻辑矩阵的行、列索引映射到原矩阵带存储位置。
- `x` 的逻辑长度为 `n`，物理长度为 `1+(n-1)*|incx|`。当 `incx>0` 时逻辑元素 `i` 位于 `x[i*incx]`；当 `incx<0` 时逻辑起点位于物理缓冲区高端，逻辑元素 `i` 位于 `x[(n-1-i)*|incx|]`。

### 支持数据类型与形状

- 数据类型：`float32`。
- `A`：设备侧带状矩阵缓冲区，布局为 `lda × n`。
- `x`：设备侧带步长的一维向量，原地读写。
- 不涉及广播、批处理或额外输出张量。

## 算子实现

### 实现方案

#### Host 侧设计

Host 入口先检查 handle、枚举、`n`、`k`、`lda`、`incx` 及设备指针。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；其余非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。`incx=0` 和无法安全取绝对值的 `INT_MIN` 均为非法值。`A`、`x` 仅在 `n>0` 时要求非空。

所有参数校验通过后，`n=0` 直接返回成功，不提交 kernel。`k=0` 且 `diag=UNIT` 时解等于输入向量，也可直接返回成功。其余合法调用把矩阵/向量地址、维度、步长和枚举信息传给 arch22 kernel，并通过 handle 中的 stream 提交；不在 Host 侧同步 stream。

#### Kernel 侧设计

三角求解存在逐分量的数据依赖，求解方向由 `uplo` 与 `trans` 共同决定：

| `uplo` | `trans` | 求解方向 |
| --- | --- | --- |
| LOWER | N | 前向 |
| LOWER | T/C | 反向 |
| UPPER | N | 反向 |
| UPPER | T/C | 前向 |

无转置路径按参考 BLAS 的列更新顺序执行。每个主元按有效三角方向访问；若当前主元向量值为零，则跳过本轮。非单位对角先用对角元素缩放主元，再将其贡献更新到带内、尚未求解的向量分量。上三角从后向前处理，下三角从前向后处理。该顺序避免构造稠密矩阵，并使累加顺序与参考实现保持一致。

转置与共轭转置路径按有效三角方向逐行求解。每行从原向量值开始，按参考 BLAS 顺序减去带内已求解分量与对应矩阵元素的乘积，之后在非单位对角模式下除以对角元素并写回。实数类型下共轭转置使用与转置相同的路径。单位对角模式不读取对角元素。

向量访问统一映射到正、负 `incx` 对应的逻辑下标。矩阵地址计算只覆盖有效带内元素。内核不需要临时全局缓冲区；串行依赖顺序在一个 kernel 执行单元内保持。

## 算子约束限制

1. 支持 `n≥0`、`k≥0`、`lda≥max(1,k+1)` 和 `incx≠0`；`incx` 支持正、负步长。
2. `uplo`、`trans`、`diag` 必须为接口枚举定义的合法值。
3. `n>0` 时 `A`、`x` 不可为空；`A` 与 `x` 不允许内存重叠。
4. `n=0` 为合法 no-op；非单位奇异三角矩阵的结果未定义，算子不做奇异性检查。
5. 除 `lda` 与 `incx` 定义的访问步长外，不支持其他非连续视图；不涉及广播。
6. 读回设备结果前由调用方同步 handle 所绑定的 stream。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列（含 Atlas 800I A2） | √ |
| Atlas A3 系列（含 Atlas 800I A3） | √ |

## 可维可测分析

### 精度标准与测试范围

精度标杆为 CBLAS/Netlib `stbsv`。`float32` 使用混合容差 `rtol=2^-13`、`atol=2^-13`；逐元素比较条件为 `|actual-golden| ≤ atol+rtol×|golden|`，整体需满足 `matched_ratio≥0.99` 且最大绝对误差不超过任务书限值 `1e-2` 或 `32×ULP`。

自验证用例覆盖 `uplo × trans × diag` 组合、零维和小维度、尺寸与带宽扫描、`lda` padding、正负 `incx`、单位对角不读取、特殊浮点输入以及非法参数。任务配套 CSV 为精度和性能用例提供统一输入描述；测试步骤通过仓库测试工程加载 CSV 并调用接口。

### 性能标准

性能指标为 Atlas 800T A2（910B3）上 `float32` kernel 平均单次耗时，按任务书设定 warmup 后采样。以下为各指定性能用例的耗时上限：

| 用例 | `n` | `k` | `uplo` | `trans` | `diag` | 平均耗时上限（μs） |
| --- | ---: | ---: | --- | --- | --- | ---: |
| 1 | 256 | 8 | UPPER | N | NON_UNIT | 198.0 |
| 2 | 512 | 32 | LOWER | N | NON_UNIT | 379.8 |
| 3 | 1024 | 16 | UPPER | T | UNIT | 659.5 |
| 4 | 2048 | 64 | LOWER | T | NON_UNIT | 1725 |
| 5 | 4096 | 128 | UPPER | N | UNIT | 2365 |

### 内存标准

任务书未设置单独的内存占用指标。实现不构造稠密矩阵，不申请额外工作区，使用输入带矩阵和原地向量作为唯一数据缓冲区。

## 兼容性分析

该接口属于 ops-blas 新增 BLAS 接口，保持与 cuBLAS `cublasStbsv` 相同的参数顺序和求解语义。公共声明放入 `include/cann_ops_blas.h`，状态码复用 `include/cann_ops_blas_common.h`，A2/A3 kernel 与测试文件按仓库的 arch22 目录布局组织。产品支持信息需明确列出 Atlas A2/A3 系列。

## 参考资料

1. [任务详情](https://www.hiascend.com/activities/task-center/details/29c59d7ef9fa4aa8ad26b82969c5e2b6?menu=tasks)
2. [ops-blas 仓库](https://gitcode.com/cann/ops-blas)
3. [算子设计文档模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
4. [FLOAT32 混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
5. [Netlib BLAS `stbsv` 参考实现](https://www.netlib.org/lapack/lapack-3.10.1/BLAS/SRC/stbsv.f)

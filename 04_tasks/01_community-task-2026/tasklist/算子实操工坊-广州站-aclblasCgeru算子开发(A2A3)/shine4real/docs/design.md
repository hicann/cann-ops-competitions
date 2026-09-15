# aclblasCgeru 算子设计文档（Atlas A2/A3）

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| V1.0 | 2026-09-14 | 初始版本，对应 ops-blas 分支 feat/aclblas-cgeru @ 60b3580 |

# 需求背景（required）

## 需求来源

本需求来源于 CANN 社区任务「算子实操工坊-广州站-aclblasCgeru算子开发(A2A3)」。任务书要求在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3）上使用 Ascend C kernel 直调实现单精度复数无共轭秩 1 更新 `aclblasCgeru`，验收通过后合入 [ops-blas](https://gitcode.com/cann/ops-blas)。

- 代码仓：https://gitcode.com/shine4real/ops-blas_5068
- 分支：`feat/aclblas-cgeru`
- CANN：9.1.0
- 实现目录：`blas/ger/arch22/`（A2/A3 共用 arch22）

## 背景介绍

### aclblasCgeru 算子功能

列主序复数矩阵原地更新：

```
A = alpha * x * y^T + A
```

逐元素：`A[i][j] = alpha * x[i] * y[j] + A[i][j]`（含步长时按 incx/incy 索引）。**y 不取共轭**，与同族 `aclblasCgerc`（`y^H`）的唯一语义差异。

| 项 | 内容 |
|----|------|
| 输入 | alpha（Host）、x（Device，长度 m）、y（Device，长度 n） |
| 输出 | A（Device，lda×n，原地更新前 m×n） |
| 类型 | COMPLEX64（实部/虚部各 float32） |
| 广播 | 不涉及 |

### 现状分析

| 现状 | 差距 |
|------|------|
| `include/cann_ops_blas.h` 已有 `aclblasCgerc` | 任务开始时无 `aclblasCgeru` 公共声明 |
| `blas/ger/arch22` 已有 `aclblasSger` | 无复数无共轭 GER |
| `blas/gerc/arch22` 已有 Cgerc | 共轭语义，不可复用 golden |

本实现新增公共 `aclblasCgeru`，不定义产品私有平行接口，与 950PR 同名接口共用同一声明。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（arch22）上实现与 cuBLAS `cublasCgeru` / Netlib `cgeru` 语义一致的句柄式 BLAS 接口：COMPLEX64、列主序、正负步长、lda padding、m=0/n=0/alpha=(0,0) no-op、handle 绑定 stream 异步执行。

## 算子原型

```cpp
aclblasStatus_t aclblasCgeru(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
    aclblasComplex* A, int lda);
```

## 参数说明

与任务书 §2.4 一致。非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`；handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。

## 需求拆解

1. 公共头文件新增声明，README / api_list 标注 A2/A3 支持。
2. Host：参数校验、列切分 tiling、异步 launch。
3. Kernel：fast（incx==1）向量路径 + general 步长路径。
4. 精度：COMPLEX64 按 FLOAT32 分量，rtol=2^-10，atol=2^-16，匹配率 ≥ 0.99。
5. 性能：任务书测机为 910B3；标杆 512/1024/2048/4096 为 5.7/10.1/55.71/219.35 us。

# 详细设计（required）

## 算子分析

### 数学公式

```
coeff = alpha * y[j]          # 无 conj
A[:, j] += coeff * x
```

展开：

```
cr = αR*yR - αI*yI
ci = αR*yI + αI*yR
A.r += cr*x.r - ci*x.i
A.i += cr*x.i + ci*x.r
```

alpha 实部或虚部为 0 时少做一次 0×Inf 分量乘，以对齐 Inf 用例。

### 支持数据类型与形状

COMPLEX64。m、n 运行时入参；A 列主序，仅更新前 m×n。

## 算子实现

### Host 侧

文件：`blas/ger/arch22/cgeru_host.cpp`。

校验顺序：handle → m/n → incx/incy（含 INT_MIN）→ alpha → lda → 计算时 x/y/A 非空。  
m=0 或 n=0 或 alpha=(0,0) 直接成功返回，不 launch。

分核：按列均分，`useCore = min(n, GetAivCoreCount())`。  
`incx==1` 且 `m<512` 时 cap 32 核；`m==512` 时 cap 30 核；1024 及以上用满 AIV。  
切分结果写入 `CgeruTilingData`（perCoreCols / remainderCols / chunkMax），无按核数组。

UB block / 向量寄存器宽度通过 `Ops::Base::GetUbBlockSize`、`GetVRegSize` 获取，避免硬编码 32B。

### Kernel 侧

文件：`blas/ger/arch22/cgeru_kernel.cpp`。入口：

- `cgeru_fast_kernel`：`incx==1 && m>=32`，不构造 TPipe
- `cgeru_kernel`：其余形状，栈上 TPipe

**Fast 交错域：** 将 x 转为交错 `xKeep` 与 pair-swap 后带符号的 `xSwap`，列循环 `cr*xKeep + ci*xSwap`。

- RMW（读 A → Axpy → 非原子写回）：`m∈[512,1024]` 且 `lda==m` 且 `nFloat%64==0`；三级流水，512 融 8 列、1024 融 4 列。
- AtomicAdd：其余 fast 形状（含 2048/4096、lda≠m）。
- 对齐 DMA 用 `DataCopy`，非 32B 对齐用 `DataCopyPad`。
- y 的 MTE2 完成后再用 yUb；先计算当前块再预取下一块 A。

**General：** DataCopyPad 按 complex 突发搬 x；负步长按 Netlib 从尾部索引；小 chunk 走标量复数乘加。

UB 按 192KB 划分为 6×32KB 槽（A2/A3 实测 UB 均为 192KB）。

### 目录

```
include/cann_ops_blas.h
blas/ger/arch22/cgeru_{host,kernel,kernel.h,tiling_data}.h/.cpp
test/ger/cgeru/   # GTest + CSV
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I A3 | √ |
| Ascend 950PR/DT | 不在本任务范围 |

# 算子约束限制

- m≥0，n≥0；incx≠0，incy≠0；lda≥max(1,m)；alpha 非空
- 不支持超出 incx/incy/lda 语义的额外非连续访问
- 无广播；A 原地覆写，不返回视图
- 异步执行，读回 Device 结果前须同步 stream

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | COMPLEX64 按 FLOAT32：rtol=2^-10，atol=2^-16，匹配率≥0.99；y 不共轭 | 任务书 §3.2 |
| 性能 | 测机 Atlas 800I A2（910B3）；warmup 后 >50 次平均；512/1024/2048/4096 不高于 5.7/10.1/55.71/219.35 us | 任务书 §3.3 |
| 内存 | 不涉及 | 任务书 §3.4 |

自测：仓内 GTest + CSV（`test/ger/cgeru/arch22/cgeru_test.csv`）。A2（910B2）全量 203 通过；A3（910_93）抽测 7 条通过。性能在 910B2/910_93 上用 msprof Task Duration 对照标杆；正式性能口径仍为 910B3。

## 兼容性分析

新增公共接口，不修改已有 `aclblasCgerc` / `aclblasSger` 语义。与 950PR 同名接口共用 `aclblasCgeru` 声明。

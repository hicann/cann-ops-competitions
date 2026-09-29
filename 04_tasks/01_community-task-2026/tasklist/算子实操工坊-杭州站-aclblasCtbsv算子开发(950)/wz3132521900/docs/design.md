# 需求背景（required）

## 需求来源

通过算子实操工坊（杭州站）社区任务完成开源仓 `ops-blas` 的 `aclblasCtbsv` 算子贡献。任务书：aclblasCtbsv 950 算子开发，目标仓 https://gitcode.com/cann/ops-blas 。实现目录 `blas/stbsv/arch35/`，测试目录 `test/stbsv/ctbsv/arch35/`。

## 背景介绍

### aclblasCtbsv 算子实现

ops-blas 仓已有实数三角带状求解 `aclblasStbsv`（`blas/stbsv/arch35/`）。公开头文件 `include/cann_ops_blas.h` 原先无 `aclblasCtbsv` 声明。本任务在同一目录新增单精度复数接口，与 cuBLAS `cublasCtbsv` 参数一一对应，语义对齐 Netlib `ctbsv`。

参考路径：

- 仓内实数基线：`blas/stbsv/arch35/`
- 公开头文件：`include/cann_ops_blas.h`
- 精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
- cuBLAS：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-tbsv
- Netlib：https://www.netlib.org/blas/ctbsv.f

本算子不是 TBE / ACLNN 迁移，而是 ops-blas 句柄式 BLAS + Ascend C kernel 直调（arch35 / Ascend 950PR）。

### aclblasCtbsv 标杆现状分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库句柄 | scalar | aclblasHandle_t | 非空 | - |
| uplo | 引用上/下三角带 | attr | UPPER / LOWER | 非法返回 INVALID_VALUE | - |
| trans | op(A) | attr | N / T / C | 非法返回 INVALID_VALUE | - |
| diag | 单位/非单位对角 | attr | UNIT / NON_UNIT | UNIT 不读对角 | - |
| n | 矩阵阶 | scalar | int | >=0；0 为 no-op | - |
| k | 带宽 | scalar | int | >=0 | - |
| A | 三角带状矩阵 | COMPLEX64 | 列主序 ND | 只读，仅引用指定带 | lda×n，lda>=k+1 |
| x | 右端 / 解 | COMPLEX64 | 步长 incx | 原地覆写 | 1+(n-1)*\|incx\| |
| incx | x 步长 | scalar | int | 非 0 且非 INT_MIN | - |

计算公式：`op(A) * x = b`。入口时 x 存放 b，出口时解覆写 x。

- `op(A) = A`（OP_N）
- `op(A) = A^T`（OP_T，不共轭）
- `op(A) = A^H`（OP_C，共轭转置）

列主序带状：LOWER 对角在第 0 行；UPPER 对角在第 k 行。不做奇异性检查。

### 算子功能分析

输入：handle、uplo、trans、diag、n、k、A、lda、x、incx  
输出：原地 x  
支持数据类型：COMPLEX64（实部/虚部 float32）  
不支持广播。

# 需求分析（required）

## 外部组件依赖

无 TBE / ACLNN / PyTorch 依赖。运行时依赖 CANN 9.1.0（acl runtime、Ascend C）以及 ops-blas 句柄。精度 golden 使用 cblas `cblas_ctbsv`。

## 内部适配模块

- Host：`blas/stbsv/arch35/ctbsv_host.cpp`，公开符号 `aclblasCtbsv`
- 标量 kernel：`ctbsv_kernel.cpp`
- SIMT kernel：`ctbsv_kernel_simt.cpp`
- Tiling：`ctbsv_tiling_data.h`
- 测试：`test/stbsv/ctbsv/`，golden 为 `cblas_ctbsv`

## 需求描述

使用 Ascend C 在 Ascend 950PR 上实现 `aclblasCtbsv`，接口放入 `include/cann_ops_blas.h`，禁止 950PR 私有平行 API。实部、虚部分别按 FLOAT32 混合容差判定。任务书四条性能 case 的平均单次耗时不高于标杆。

## 需求拆解

1. 声明并实现 `aclblasCtbsv`，参数语义对齐 cuBLAS `cublasCtbsv`
2. 覆盖 N/T/C、UPPER/LOWER、UNIT/NON_UNIT、n=0、k=0 且 UNIT、负步长、非法参数
3. OP_N 用 axpy 回代，OP_T/OP_C 用点积回代，累加顺序对齐 Netlib
4. 四条性能 case 耗时不高于 917.5、1831.11、3753.09、4898.68 微秒

# 详细设计（required）

## 算子分析

### 数学公式

复数乘：`(ar, ai) * (br, bi) = (ar*br - ai*bi, ar*bi + ai*br)`。除法乘以共轭再除以模平方。

OP_N 对齐 Netlib axpy：先处理对角，当前 x 实部虚部都为 0 则跳过该列，否则更新带内其他未知数。

OP_T / OP_C 对齐 Netlib 点积：先累加带内已知项，再除对角。LOWER 转置从远端到近端，UPPER 转置从小下标升到近端。OP_C 对读到的 A（含对角）虚部取反。

前向条件：`(LOWER 且 OP_N)` 或 `(UPPER 且转置)`。其余为后向。

`n == 0` 校验通过后返回成功。`k == 0` 且 `diag == UNIT` 返回成功，x 不变。`incx < 0` 时物理偏移为 `(n - 1 - idx) * (-incx)`。

### 支持数据类型

COMPLEX64。GM 上按交错 float 存放，kernel 用 `float*` 读写。

### 支持形状

n、k 为运行时入参。A 至少 `lda * n` 个复数，`lda >= k + 1`。本任务 CSV 最大 n 为 4096。

## 算子实现

### 实现方案

一次启动 1 个 AIV block。沿 n 的回代保持串行，使累加顺序接近标量 cblas。

#### Host 侧设计

校验顺序：handle、n、k、枚举与 lda/incx/指针、`n == 0`、`k == 0 && UNIT`，然后 launch。

非法 uplo、trans、diag、`n < 0`、`k < 0`、`lda <= k`、`incx == 0`、`incx == INT_MIN` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。`n > 0` 时 A 或 x 为空指针返回该状态。handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。Host 不对 stream 做同步。

线程数：

- OP_T 与 OP_C：`numThreads = 0`，走标量 kernel。点积只有一条累加链。
- OP_N 且 `n < 128`：标量。
- OP_N 且 `n >= 128`：SIMT。带宽加 1 不超过 32 时用 32 线程，超过 32 用 64，超过 64 用 128，并且不超过 n。带内 axpy 写不同的 x[j]，线程数不改变舍入顺序。

标量路径把全部 x 放进 UB。SIMT 在 `n <= 2048` 时把 x 放进 UB，更大时走 GM。

#### Kernel 侧设计

标量 12 个模板，覆盖 uplo、三种 trans、两种 diag。SIMT 同样 12 个 `asc_vf_call`，只服务 OP_N。OP_N 的对角除法和全零跳过由 thread 0 完成，带内更新按线程步长进行，每行两次 syncthreads。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

README 产品支持表只标注 Ascend 950PR：支持。

## 算子约束限制

不检查对角是否为零。不支持 lda/incx 语义以外的非连续访问。不要求 bit-exact。不要求确定性。Host 异步，调用方读回前同步 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部、虚部分开按 FLOAT32：rtol=2^-10，atol=2^-16，matched_ratio>=0.99，且 \|diff\|<=max(1e-2, 32*ULP) | 生态算子开源精度标准 |
| 性能标准 | 四条 case 平均耗时不高于 917.5 / 1831.11 / 3753.09 / 4898.68 微秒 | 任务书 §3.3 |

自测用 msprof：预热 10 次、采样 60 次、去掉 10 个最大 Task Duration 后取平均。任务书另要求 warmup 后采样多于 50 次的 aclrtEvent 平均，两套数字分开记录。

## 兼容性分析

新接口，声明放入已有头文件，供其他产品线共用。不另做 950 私有平行 API。

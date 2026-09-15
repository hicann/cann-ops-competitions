# aclblasCgbmv 算子设计文档（Ascend 950PR）

# 需求背景（required）

## 需求来源

本需求来自 CANN 社区任务「aclblasCgbmv 950 算子开发」（任务书：`aclblasCgbmv_Atlas950PR_task_doc.md`）。

任务要求在昇腾 NPU（Ascend 950PR / arch35）上使用 Ascend C 开发单精度复数带状矩阵-向量乘算子 `aclblasCgbmv`，功能、参数语义对标 cuBLAS `cublasCgbmv` 与 Netlib `cgbmv`。代码合入 ops-blas（https://gitcode.com/cann/ops-blas ）目录 `blas/gbmv/arch35/`，测试合入 `test/gbmv/cgbmv/arch35/`。

## 背景介绍

### aclblasCgbmv 算子实现

BLAS Level-2 的 GBMV（General Banded Matrix-Vector multiply）完成

```
y := alpha * op(A) * x + beta * y
```

其中 A 为 m×n 带状复数矩阵，含 kl 条下次对角线、ku 条上次对角线，按**列主序带状**存储；x、y 为复数向量。GBMV 用于稀疏带宽结构下的矩阵-向量乘，避免稠密 GEMV 对带外零元的访存与计算。

本次实现路径：

- 目标硬件：Ascend 950PR（arch35），CANN 9.1.0
- 编程语言：Ascend C；AIV-only
- 工程模式：**Ascend C kernel 直调**。Host 提供句柄式 BLAS 接口（`aclblasHandle_t` 携带 stream），通过 `<<<numBlocks, nullptr, stream>>>` 下发 kernel，不经过 aclnn 图/原型注册
- 实现目录：`blas/gbmv/arch35/`（与实数 `aclblasSgbmv` 同族）
- 接口声明：`include/cann_ops_blas.h`（新增公共声明，禁止 950PR 私有平行接口）

### 列主序带状存储

元素 A(i,j)（1-based）存于 `A(ku+1+i-j, j)`。0-based 地址：

```
A[ku + i - j + j * lda]
```

约束：

- 主对角线在第 ku+1 行（1-based）
- 带外左上 ku×ku、右下 kl×kl 三角**禁止读取**
- `lda >= kl + ku + 1`

### 对标接口现状

| 接口 | 现状 | 说明 |
|------|------|------|
| `cublasCgbmv` | 语义基线 | 参数序列与本接口一一对应 |
| `cgbmv`（Netlib / cblas） | 参考实现 + golden | 精度比对标杆 |
| `aclblasSgbmv`（ops-blas arch35） | 同族实数已实现 | SIMT host/kernel 工程范式 |
| `aclblasCgbmv` | **本次新增** | 原 `cann_ops_blas.h` 无该声明 |

### 算子功能分析

- `op(A)`：`ACLBLAS_OP_N` → A，`ACLBLAS_OP_T` → Aᵀ，`ACLBLAS_OP_C` → Aᴴ
- 数据类型：COMPLEX64（`aclblasComplex`，实部/虚部各 float32，交错存储）
- `m = 0` 或 `n = 0`：合法 no-op
- `alpha = (0,0)` 且 `beta = (1,0)`：quick return，不引用 A/x/y
- `alpha = (0,0)`：仅执行 `y = beta * y`
- `beta = (0,0)`：不读 y 旧值
- `incx` / `incy` 支持负步长

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上以 Ascend C kernel 直调实现 `aclblasCgbmv`，接口、异常码、带状语义与 cuBLAS / Netlib 对齐，精度满足生态开源 FLOAT32 MIXED_TOLERANCE，性能不高于任务书 §3.3 标杆。

## 需求拆解

1. **功能**：N/T/C 全覆盖；正负步长；lda padding；quick return / scale-only；不读带外三角。
2. **接口**：在 `include/cann_ops_blas.h` 新增声明，与 `aclblasSgbmv` 参数序列、`aclblasCgemv` 复数惯例对齐。
3. **精度**：实部/虚部分别按 FLOAT32，atol = 2⁻¹⁶，rtol = 2⁻¹⁰，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32×ULP。
4. **性能**：5 条官方 case warmup 后采样 >50 次取平均，平均耗时 ≤ 标杆。
5. **工程**：Host + Kernel + Tiling；README 标注 Ascend 950PR 支持；CSV GTest + benchmark。

## 外部组件依赖

不引入新的第三方库。复用 ops-blas 已有能力：

| 依赖 | 来源 | 用途 |
|------|------|------|
| handle | `common/helper/aclblas_handle_internal.h` | stream |
| host 工具 | `common/helper/host_utils.h` | `CHECK_RET` / `CeilDiv` / `GetAivCoreCount` |
| SIMT 常量 | `common/helper/kernel_constant.h` | `SIMT_MIN_THREAD_NUM` / `SIMT_MAX_THREAD_NUM` |
| UB 块大小 | `Ops::Base::GetUbBlockSize()` | kernel 侧对齐，避免硬编码 32 |
| 平台 UB | `PlatformAscendCManager::GetCoreMemSize` | host 查询 UB 预算 |
| golden | 系统 cblas `cblas_cgbmv` | 测试比对 |

## 内部适配模块

| 模块 | 文件 | 职责 |
|------|------|------|
| Host | `blas/gbmv/arch35/cgbmv_host.cpp` | 校验、标量加载、tiling、快路径选择、kernel 直调 |
| Kernel 入口 | `blas/gbmv/arch35/cgbmv_kernel.cpp` | scale / N 快路径 / SIMT UB / SIMT GM 分发 |
| N 快路径 | `blas/gbmv/arch35/cgbmv_n_col_atomic_aiv.h` | residue-8 列归属 + 2D DMA + Vector AXPY |
| Tiling | `blas/gbmv/arch35/cgbmv_tiling_data.h` | host/kernel 共享 POD |
| 精度测试 | `test/gbmv/cgbmv/arch35/cgbmv_test.cpp` + CSV | 排除 `TC_PF_` 后 1001 条 |
| 性能测试 | `test/gbmv/cgbmv/arch35/cgbmv_benchmark.cpp` | warmup 20 + 采样 80，`aclrtEvent` |
| 文档 | `blas/gbmv/README.md`、`test/gbmv/cgbmv/README.md` | 接口与复现步骤 |

`blas/CMakeLists.txt` 按 `SOC_ARCH_DIRS` 收集 `arch35/*.cpp`，无需改顶层构建。

## 接口原型

```cpp
aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

# 详细设计（required）

## 算子分析

### 数学公式

```
y := alpha * op(A) * x + beta * y

        ⎧ A      trans = ACLBLAS_OP_N
op(A) = ⎨ Aᵀ     trans = ACLBLAS_OP_T
        ⎩ Aᴴ     trans = ACLBLAS_OP_C
```

N 路径第 i 行（0-based）只累加列 `j ∈ [max(0, i-kl), min(n, i+ku+1))`：

```
acc_i = Σ_j A(i,j) * x_j
y_i   = alpha * acc_i + beta * y_i
```

T/C 路径第 j 列累加行 `i ∈ [max(0, j-ku), min(m, j+kl+1))`；C 对 A 的虚部取共轭。

### 支持数据类型

COMPLEX64，实部/虚部 FLOAT32。

### 支持形状

运行时 `m/n/kl/ku/lda/incx/incy`；不要求 dynamic shape 框架，不支持除 lda/inc 以外的非连续视图。

## 算子实现

总体采用 **双路径**：

| 路径 | 条件 | 实现 |
|------|------|------|
| N 向量快路径 | `trans=N`，`incx=incy=1`，`beta=(0,0)`，`m≥32`，`w=kl+ku+1≤2048`，AIV ≥ 8 | `CgbmvNColAtomicAiv` |
| SIMT UB | `incx==1` 且 x 瓦片可放入 UB | `CgbmvUb<TRANS>` |
| SIMT GM | 负步长、非单位 incx、过小 shape、x 超 UB | `CgbmvGm<TRANS>` |
| Scale | `alpha=(0,0)` | `CgbmvScale` |

### Host 侧设计

1. 校验 handle、枚举、维度、带宽、lda、步长、alpha/beta 非空；失败前 `OP_LOGE`。
2. `m==0` 或 `n==0` 直接成功。
3. `LoadComplexScalar`：Host 直接读，Device 则 D2H + stream 同步。
4. `alpha=0 && beta=1` quick return。
5. `alpha≠0` 时检查 A/x 非空；始终检查 y 非空。
6. 查询 AIV 核数与 UB 预算（`GetCoreMemSize`，上限 `CGBMV_UB_X_FLOATS_MAX=16384` floats）。
7. 默认 SIMT tiling：按输出维 `outDim`（N 为 m，T/C 为 n）均分到 AIV，`numThreads` 对齐到 `[SIMT_MIN_THREAD_NUM, SIMT_MAX_THREAD_NUM]`。
8. 满足快路径条件时改写 tiling：`useSimd=1`，`numBlocks` 向下对齐到 8 的倍数（`n<8` 仍启动 8 核，避免漏 residue 类），`tileCols=64`；launch 前 `aclrtMemsetAsync(y)`。
9. `cgbmv_kernel_do` 直调。`ubBlockSize` 由 kernel 侧 `Ops::Base::GetUbBlockSize()` 取得，Host 不硬编码 32。

`CgbmvTilingData` 主要字段：`m/n/kl/ku/lda/trans/scaleOnly`、`alpha/beta`、`incx/incy`、`numBlocks/numThreads/rowsPerBlock`、`ubXFloats/ubBlockSize`、`useSimd/tileCols`。

### Kernel 侧设计

入口 `cgbmv_kernel`：`KERNEL_TYPE_AIV_ONLY`。

```
if scaleOnly:                         CgbmvScale
else if useSimd && N && incx==incy==1: CgbmvNColAtomicAiv
else:                                 CgbmvDispatch<trans>
```

#### N 快路径：`CgbmvNColAtomicAiv`

**分核（residue-8）**

- `blockIdx = GetBlockIdx()`
- `residue = blockIdx & 7`，`chunk = blockIdx >> 3`
- `jMod = (ku + residue) & 7`
- 本核处理列 `j = jMod + 8t`，t 按 chunk 均分

同一 residue 的列在带状存储中 packed 步长为 `8*lda` 个复数，可走 2D `DataCopyPad`。模数必须为 8：plateau 上 `rowLo` 随 t 步进 8，本地累加下标 `dest = rowLo - yBase` 保持 8 对齐，才能用向量 `Add`。

**plateau / ramp**

- plateau：`ku ≤ j` 且 `j+kl+1 ≤ m`，带宽完整，长度为 `w`
- ramp：带宽被矩阵边界截断，走单列 DMA + `AxpyStore`（Atomic 写回）

**计算**

1. Init：按 UB 约 40000 float 收缩 `tileCols≤64`；为 aR/aI 平面 `Duplicate` 后再 `DeInterleave`（官方 `w≡1 (mod 8)` 时 `Align8(2w)/2` 与 `Align8(w)` 差 4，必须填零）。
2. `Duplicate(accR/accI)=0`，`SetAtomicAdd`。
3. 连续 plateau 且 `tileCols≥2`：整核一次 2D DMA A（`blockCount=nCols`，`blockLen=2w` floats，`srcGap=8*lda-w` 复数）+ 2D DMA x（stride 8 个复数）；标量 `zr/zi = alpha * x[j]`；逐列 DeInterleave + Vector 复乘累加到 `acc`。
4. 否则单列 `DataCopyPad` + DeInterleave + `AxpyAcc` / `AxpyStore`。
5. `FlushY`：`Interleave` 后一次 `DataCopyPad` Atomic 写 `y[yBase : yBase+ySpan)`。
6. `SetAtomicNone`。

Host 已将 y memset 为 0，AtomicAdd 的目的地初值为 0。8 个 residue 核在几乎所有 `y[i]` 上冲突，这是该映射下正确归约所必需的。非对齐尾块一律 `DataCopyPad`，不用 `DataCopy`。

#### SIMT：`CgbmvUb` / `CgbmvGm`

- 按 `outIdx` 网格跨步；每线程寄存器累加一带内点积。
- UB 路径：block 协作把 `x[xBase:xEnd)` 搬进 `__ubuf__`（容量 `CGBMV_UB_X_FLOATS_MAX`），`asc_syncthreads()` 后标量读 x、标量读 A。
- GM 路径：x 也走 GM。`incx≠1`、`outDim < GetUbBlockSize()`、或 x 瓦片超过 UB 时走 GM。
- `CgbmvWriteY`：先算 `alpha*acc`；仅当 beta 非零才读旧 y。
- 负步长：`offset = (dim-1-idx)*(-inc)`。

T/C 官方 case 列内 A 连续，SIMT UB 已低于标杆，不再强制向量化。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|----------------|----------|
| Ascend 950PR / 950DT（arch35） | √ |
| Atlas A2 | 不支持 |
| Atlas A3 | 不支持 |

## 算子约束限制

- `handle` 非空，否则 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- `trans ∈ {N,T,C}`，否则 `ACLBLAS_STATUS_INVALID_ENUM`
- `m≥0`，`n≥0`；`0≤kl≤m-1`（m>0），`0≤ku≤n-1`（n>0）
- `lda ≥ kl+ku+1`；`incx≠0`，`incy≠0`
- alpha、beta 非空；`alpha≠0` 时 A、x 非空
- 不读带外三角；x/y 不与 A 重叠原地混用
- 异步执行，读回前须同步 stream

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度标准 | 实部/虚部分别 FLOAT32 MIXED_TOLERANCE：atol=2⁻¹⁶，rtol=2⁻¹⁰，matched_ratio≥0.99 | 任务书 §3.2 / 生态开源精度标准 |
| 性能标准 | warmup 后采样 >50 次平均 ≤ 下表标杆 | 任务书 §3.3 |
| 内存 | 任务书 §3.4 不涉及 | 任务书 |

| case | trans | m | n | kl | ku | 标杆（us） |
|------|-------|---|---|----|----|------------|
| 1 | N | 1024 | 1024 | 32 | 32 | 25.58 |
| 2 | N | 2048 | 2048 | 64 | 64 | 34.25 |
| 3 | T | 2048 | 2048 | 64 | 64 | 35.85 |
| 4 | C | 2048 | 2048 | 64 | 64 | 35.88 |
| 5 | N | 4096 | 4096 | 128 | 128 | 52.79 |

实测（`cgbmv_benchmark`，warmup 20 + 采样 80）：N 走 `CgbmvNColAtomicAiv`，T/C 走 `CgbmvUb`，五条均低于标杆。

## 测试设计

- 精度：`cgbmv_test --gtest_filter='-*TC_PF_*'`，cblas golden，约 1001 条。
- 门禁：`TC_L0*`、`TC_SQ_028`、`TC_AB_082`、`TC_SQ_051`。
- 性能：必须 `cmake --install` 后加载 `out/lib64/libops_blas.so`，再跑 `cgbmv_benchmark`。
- 负向：空指针、非法枚举、负维、零步长、kl/ku 越界。

## 兼容性分析

新接口，不破坏已有 `aclblasSgbmv`。声明放在公共头文件，供产品线共用。

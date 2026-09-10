# aclblasCscal 算子设计文档

# 需求背景（required）

## 需求来源

本任务来源于 CANN 社区任务 2026 `aclblasCscal` 算子开发任务。目标是在 **Ascend 950PR**、**CANN 9.1.0** 环境中，基于 `ops-blas` 工程，以 Ascend C/CATLASS 的 Kernel 直调方式实现单精度复数向量缩放接口 `aclblasCscal`。

算子实现位于：

```text
ops-blas/blas/scal/arch35/
```

测试实现位于：

```text
ops-blas/test/scal/cscal/arch35/
```

公共接口声明复用 `ops-blas/include/cann_ops_blas.h` 中的既有 `aclblasCscal` 声明；本任务不新增 Ascend 950PR 私有 API。

## 背景介绍

### 算子功能

`aclblasCscal` 是 BLAS Level-1 原地缩放算子。设复数标量 `alpha = ar + ai*i`、向量元素 `x = xr + xi*i`，则：

```text
x[j] = alpha * x[j]
     = (ar*xr - ai*xi) + (ar*xi + ai*xr)*i
```

逻辑元素遵循 Netlib/Fortran 的步长规则：

```text
j = 1 + (i - 1) * incx,  i = 1..n
```

在 C/C++ 的零基存储中，访问位置为 `x[i * incx]`，其中 `i = 0..n-1`。`aclblasComplex` 由两个 FP32 分量（real、imag）组成，数据在 Global Memory 中以交织的 `[re, im, re, im, ...]` 形式存储。

### 参考语义与边界行为

功能语义对齐 cuBLAS `cublasCscal`，快速返回行为对齐 Netlib `cscal`：

| 条件 | 返回值 | x 行为 |
| --- | --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 不访问 x |
| `n <= 0` | `ACLBLAS_STATUS_SUCCESS` | 合法 no-op，不修改 x |
| `incx <= 0` | `ACLBLAS_STATUS_SUCCESS` | 合法 no-op，不修改 x；不支持负步长反向遍历 |
| 执行场景且 `alpha == nullptr` 或 `x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 不执行 |
| `alpha == (1,0)` | `ACLBLAS_STATUS_SUCCESS` | 等价 no-op |
| `alpha == (0,0)` | `ACLBLAS_STATUS_SUCCESS` | **全部逻辑元素写为精确的 `(+0,+0)`，不是 no-op** |

其中 handle 空指针校验优先于快速返回；`n <= 0` 或 `incx <= 0` 的 no-op 则先于 `alpha/x` 数据指针校验，以保证与 Netlib 语义一致。

# 需求分析（required）

## 需求描述

实现如下 handle 式 BLAS 接口：

```cpp
aclblasStatus_t aclblasCscal(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* alpha,
    aclblasComplex* x,
    int incx);
```

- `alpha` 位于 Host 内存；
- `x` 位于 Device 内存，按 `incx` 原地访问；
- 支持 `COMPLEX64`；
- 通过 `aclblasSetStream` 绑定的 stream 异步启动 NPU Kernel；调用方在读回结果前同步 stream；
- 对正步长 `incx` 支持通用逻辑索引访问；物理元素长度应至少为 `1 + (n - 1) * incx`。

## 需求拆解

1. 实现 COMPLEX64 复数乘法与原地写回；
2. 在 Host 侧完成返回码、no-op 与空指针语义；
3. 对连续与正步长非连续向量均正确访问，且不写入 stride gap；
4. 避免编译器 FMA contraction 改变 Netlib golden 的舍入/Inf/NaN 分类；
5. 对 `alpha=(0,0)` 在 Inf/NaN 输入下仍写出精确正零；
6. 采用多 AIV Core + SIMT 并行，满足 950PR 上三个大规模连续性能用例；
7. 建立 cblas golden 的 CSV 驱动测试与性能采样。

# 详细设计（required）

## 算子分析

### 数据类型、布局与计算公式

| 项目 | 设计 |
| --- | --- |
| 输入/输出 dtype | `COMPLEX64`（两个 FP32 分量） |
| x 布局 | 一维、交织 `[re, im]`，逻辑元素按 `incx` 取址 |
| 计算 | `(ar*xr-ai*xi, ar*xi+ai*xr)` |
| 输出方式 | x 原地更新 |
| 广播/动态 shape | 不涉及；`n` 为运行时标量 |
| 额外 Device workspace | 0 B |

## 算子实现

### 总体流程

```text
aclblasCscal(handle, n, alpha, x, incx)
    │
    ├─ handle == nullptr ? HANDLE_IS_NULLPTR
    ├─ n <= 0 || incx <= 0 ? SUCCESS（no-op）
    ├─ alpha/x == nullptr ? INVALID_VALUE
    ├─ alpha == (1,0) ? SUCCESS（no-op）
    ├─ 获取 AIV Core 数，构造 CscalTilingData
    └─ 在 handle->stream 启动 cscal_simt_kernel
          └─ 每个 Core 调用 CscalSimtCompute 处理自身逻辑元素区间
```

### Host 侧设计

Host 实现文件为 `blas/scal/arch35/cscal_host.cpp`。

#### 参数校验与快速返回

Host 侧严格按以下优先级执行：

```text
handle 校验 → n/incx no-op → alpha 校验 → x 校验 → alpha=(1,0) quick return → Kernel 启动
```

这样既保证 `handle=nullptr` 的返回码优先级，又允许合法 no-op 在传入空 `alpha/x` 时直接成功返回、不访问 Device 数据。

#### 分核与 tiling

`CscalTilingData` 向 Kernel 传递：

```text
alphaReal / alphaImag
incx
useCoreNum
startOffset[core]
calCount[core]
nthreads
```

分核采用逻辑元素均分：

```text
baseCount = n / useCoreNum
remain    = n % useCoreNum
```

前 `remain` 个 Core 各多处理一个元素。Core 数由可用 AIV Core 数和 `n` 共同限制，tiling 数据最多容纳 64 个 Core。每个 Core 的线程数由平均工作量计算，并对齐到 `SIMT_MIN_THREAD_NUM` 后上限裁剪到 `SIMT_MAX_THREAD_NUM`。

Kernel 在 handle 绑定的 stream 上异步执行；Host 不进行强制同步，也不分配额外 workspace。

### Kernel 侧设计

Kernel 实现文件为 `blas/scal/arch35/cscal_kernel.cpp`，采用 AIV-only Kernel Shell + SIMT VF 计算函数：

```text
cscal_simt_kernel
  └─ asc_vf_call<CscalSimtCompute>(...)
```

选择 SIMT 方案的原因是 COMPLEX64 交织布局需要相邻实/虚部标量访问；相比依赖 Gather、DeInterleave/Interleave 的向量路径，SIMT 可直接进行标量 load/store，避免额外 lane-shuffle 开销。一个 SIMT thread 在循环中处理一个逻辑复数元素：

```text
idx = (startOffset + i) * incx
re  = x[2 * idx]
im  = x[2 * idx + 1]
```

#### 精确置零路径

`alphaReal == 0.0f && alphaImag == 0.0f` 时，Kernel 进入专用路径：

```text
x[2 * idx]     = +0.0f
x[2 * idx + 1] = +0.0f
```

该路径不读取输入、也不执行乘法，因此对于 `(Inf, NaN)` 等非有限输入不会发生 `0*Inf` 或 `0*NaN` 产生 NaN 的问题；同时只写入 `i*incx` 对应的逻辑元素，不触及 stride gap。

#### 普通复数乘法与舍入控制

Netlib cblas golden 按独立乘积的 FP32 舍入结果计算：

```text
RN(ar*xr) - RN(ai*xi)
RN(ar*xi) + RN(ai*xr)
```

若将表达式直接写成 `ar*xr-ai*xi`，Device 编译器可能收缩为 FMA，在大 alpha、灾难性消没或溢出场景改变 Inf/NaN 分类。实现将四个乘积先通过 `+0.0f` 物化：

```cpp
float pr = alphaReal * re + 0.0f;
float pi = alphaImag * im + 0.0f;
float qr = alphaReal * im + 0.0f;
float qi = alphaImag * re + 0.0f;

outReal = pr - pi;
outImag = qr + qi;
```

从而使最终加减操作使用已经单次舍入的乘积，保持与 CPU cblas golden 一致的数值行为。

## 支持硬件

| 芯片版本 | 是否支持 | 说明 |
| --- | --- | --- |
| Ascend 950PR | 支持 | 本任务目标硬件，已完成实机构建、精度和性能验证 |

软件版本：CANN 9.1.0。

## 算子约束限制

1. 仅支持 COMPLEX64；
2. `x` 为 Device 内存中的原地向量，`alpha` 为 Host 内存中的复数标量；
3. `incx > 0` 时按正步长处理；`incx <= 0` 为合法 no-op，不支持反向遍历；
4. 调用方负责保证 x 的物理存储长度满足步长访问需求；
5. 本算子不涉及 broadcast、额外 leading dimension、随机数或额外 Device workspace；
6. 读回 Device 结果前，调用方需要同步绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 验证方式 |
| --- | --- | --- |
| Golden | cblas（Netlib `cscal`） | 实部/虚部分别比对 |
| rtol | `2^-10 ≈ 9.7656e-4` | 任务书 COMPLEX64 标准 |
| atol | `2^-16 ≈ 1.5259e-5` | 任务书 COMPLEX64 标准 |
| matched ratio | `>= 0.99` | GTest verifier 输出 |
| max abs error | `<= 1e-2` 或 `32*ULP` | GTest verifier 输出 |
| 性能采样 | warmup 后有效采样 >50 次 | 20 次 warmup + 200 次 event 计时 |

## 功能与精度测试设计

CSV 用例 `cscal_test.csv` 共 1200 条：1000 条精度用例与 200 条性能/内存用例。精度用例覆盖：

- 小 shape、尺寸扫描、2 的幂与 ±1、质数和非对齐尺寸；
- `incx=1/2/3/5/7`；
- `alpha=(0,0)`、`(1,0)`、纯虚、负值、大值；
- 随机、全零、交替、极端、Inf/NaN 填充；
- `n=0`、负 n、`incx=0`、负 incx 的 no-op；
- 空 alpha、空 x 的 `INVALID_VALUE` 负向场景。

除 CSV 用例外，GTest 还验证：

1. `NullHandle`；
2. `NullHandleTakesPrecedenceOverQuickReturn`；
3. `QuickReturnSkipsDataPointerValidation`；
4. `ZeroAlphaWritesPositiveZerosForNonFiniteStridedInput`：以 `alpha=(0,0)`、Inf/NaN 输入、`n=3, incx=2` 验证逻辑元素被写为正零，物理 gap 保持不变。

在 Ascend 950PR / CANN 9.1.0 上，执行：

```bash
./cscal_test --gtest_filter=-*TC_PF*
```

结果：

```text
1004 / 1004 PASS
```

其中复数结果按实部、虚部分别与 CPU golden 比对；本轮日志中的分量比较均为 `matchedRatio=1.0`、`maxAbsErr=0`。

## 性能测试与结果

性能测试使用 ACL event 包围同一 stream 中的 Kernel 调用，先执行 20 次 warmup，再对 200 次调用取平均。任务书的三个典型连续访存用例实测如下：

| n | incx | 平均耗时（us） | 任务书门槛（us） | 结果 |
| ---: | ---: | ---: | ---: | --- |
| 1,048,576 | 1 | **11.324** | 13.73 | PASS |
| 2,097,152 | 1 | **18.067** | 21.22 | PASS |
| 4,194,304 | 1 | **31.583** | 43.31 | PASS |

全部性能点满足门槛。

## 内存与可维护性

算子仅原地读写 x，额外 Device workspace 为 0 B。最大性能 case 的 x 输入为：

```text
4,194,304 * sizeof(COMPLEX64) = 32 MiB
```

该内存属于调用方的输入/输出，不属于算子额外内存。实现复用公共接口，arch35 文件与其它产品架构隔离；专项 `alpha=(0,0)` × Inf/NaN 测试用于防止后续优化破坏精确置零语义。

## 参考资料

1. `aclblasCscal_Atlas950PR_task_doc.md`（任务书）；
2. ops-blas：<https://gitcode.com/cann/ops-blas>；
3. Netlib BLAS cscal：<https://www.netlib.org/blas/cscal.f>；
4. cuBLAS cublasCscal：<https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-scal>；
5. 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>。

# aclblasCgbmv 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 9月社区任务-aclblasCgbmv算子开发(A2A3) |
| 任务 ID | `0d3c27ed59a340eea2a44a56fec8a10f` |
| 目标平台 | Atlas A2/A3，`arch22`；性能基准设备 Atlas 800T A2（910B3） |
| 软件版本 | CANN 9.1.0 |
| 目标代码仓 | `cann/ops-blas` |
| 设计阶段 | 本文为待评审方案，不代表代码、精度或性能已经验收 |

# 需求背景（required）

## 需求来源

本任务来自 [昇腾任务中心](https://www.hiascend.com/activities/task-center/details/0d3c27ed59a340eea2a44a56fec8a10f?menu=guide)。要求基于 `ops-blas` 工程，以 Ascend C kernel 直调方式实现与 `cublasCgbmv` 参数和语义对齐的单精度复数一般带状矩阵向量乘，并提交设计文档、代码、测试和自验材料。

## 背景介绍

### 算子现状

`ops-blas` 已有 `blas/gbmv/README.md` 和 arch35 的 `sgbmv` 工程，arch22 的 `gemv` 中也有复数计算实现可参考。目前 `blas/gbmv/arch22/` 没有本任务的 `cgbmv` 实现，公共头 `include/cann_ops_blas.h` 也尚未声明 `aclblasCgbmv`。本任务需新增接口声明和 arch22 实现，不能直接复用实数 `sgbmv` 的计算路径：复数乘法、共轭转置及特殊值传播均不同。

### 算子功能分析

设 `A` 是固定为 `m × n` 的列主序一般带状矩阵，`kl`/`ku` 分别为下/上带宽。输出原地写回 `y`：

```text
y = alpha * op(A) * x + beta * y
op(A) = A      (ACLBLAS_OP_N)
op(A) = A^T    (ACLBLAS_OP_T)
op(A) = A^H    (ACLBLAS_OP_C)
```

`N` 时 `x`/`y` 的逻辑长度为 `n`/`m`，`T` 和 `C` 时为 `m`/`n`。`A` 始终按原矩阵维度存储，转置不改变 `m`、`n` 或 `lda` 的含义。

目标公共接口为：

```cpp
aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, int kl, int ku,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

`aclblasComplex` 采用仓库公共类型，由两个 `float` 成员 `real`、`imag` 组成。`alpha`/`beta` 为 Host 指针；`A`/`x`/`y` 为 Device 指针。计算在 `handle` 绑定的 stream 上异步提交，调用方在读取结果前同步 stream。

# 需求分析（required）

## 需求描述

实现 COMPLEX64（单精度复数）`aclblasCgbmv`，覆盖 `N/T/C`、矩形矩阵、非对称及退化带宽、`lda` padding、正负 `incx/incy`、`alpha/beta` 特例、零维 no-op 和非法参数校验。结果写回 `y`；不支持 `y` 与 `A`、`x` 重叠，不要求额外的非连续 Tensor 视图或广播。

## 需求拆解

1. 在 `include/cann_ops_blas.h` 声明接口，在 `blas/gbmv/arch22/` 实现 Host 校验、tiling、kernel launch 与 README 产品支持说明。
2. 正确处理带状压缩索引、`N/T/C` 三种运算及负步长，并保证每个输出元素只由一个执行单元写入。
3. 以无工作区、单次 kernel launch 为主要方案；针对连续向量和任务书的性能用例优化数据复用与访存。
4. 在 `test/gbmv/cgbmv/arch22/` 增加 C++ GTest、CSV 驱动和 Netlib CBLAS golden，对随任务提供的用例做精度、边界和性能自验。
5. 使用 Atlas 910B3 和 CANN 9.1.0 实测，保存构建、精度、性能和内存证据；A3 的最终适配与验收按任务书要求执行。

# 详细设计（required）

## 算子分析

### 数学公式与带状布局

使用 0-based 索引。只有 `max(0, j-ku) <= i <= min(m-1, j+kl)` 的 `A(i,j)` 有效，物理偏移为：

```text
band_row = ku + i - j
A_offset = j * lda + band_row
lda >= kl + ku + 1
```

`A_offset` 的乘加使用至少 64 位整数，避免大维度下 Host/Kernel 地址计算溢出；只读取有效带内位置，不读取带状存储的左上、右下填充区。

`N` 模式按输出行 `i` 计算：

```text
j_begin = max(0, i - kl)
j_end   = min(n - 1, i + ku)
sum     = Σ[j_begin..j_end] A(i,j) * x[j]
y[i]    = alpha * sum + beta * y_old[i]
```

`T/C` 模式按输出列 `j` 计算：

```text
i_begin = max(0, j - ku)
i_end   = min(m - 1, j + kl)
sum_T   = Σ[i_begin..i_end] A(i,j)       * x[i]
sum_C   = Σ[i_begin..i_end] conj(A(i,j)) * x[i]
y[j]    = alpha * sum_{T/C} + beta * y_old[j]
```

对于 `a = ar + i*ai`、`x = xr + i*xi`，乘积为 `(ar*xr - ai*xi) + i*(ar*xi + ai*xr)`；`C` 模式仅对 `A` 取共轭，不对 `x` 或 `alpha` 取共轭。实部、虚部以 FP32 累加。规约顺序允许与 Netlib 不同，最终按精度标准判定。

### 正负步长

传入的 Device 指针指向向量存储区起点。逻辑索引 `k`（`0 <= k < len`）的物理偏移为：

```text
inc > 0: k * inc
inc < 0: (len - 1 - k) * abs64(inc)
```

`abs64(inc)` 表示先转成 64 位整数再取绝对值，避免 `inc == INT_MIN` 时对 32 位整数取负溢出。分别以 `lenX` 和 `lenY` 计算 `x`、`y` 的起始偏移，不把 `incx` 的符号误用于 `y`。步长相关乘法使用 64 位地址计算。

### 数据类型与形状

| 对象 | 类型与位置 | 逻辑长度或形状 | 约束 |
| --- | --- | --- | --- |
| `A` | Device `aclblasComplex` | `m × n`，物理存储 `lda × n` | 只访问带内元素 |
| `x` | Device `aclblasComplex` | `n`（N）或 `m`（T/C） | `incx != 0` |
| `y` | Device `aclblasComplex` | `m`（N）或 `n`（T/C） | 原地输出，`incy != 0` |
| `alpha`, `beta` | Host `aclblasComplex` | 标量 | 非空指针 |

`m/n/kl/ku/lda` 是运行时参数，不要求编译期固定 shape。`m == 0` 或 `n == 0` 为合法 no-op，不启动 kernel。

## 算子实现

### 实现方案

#### Host 侧设计

Host 层拟实现 `cgbmv_host.cpp` 和与 kernel 共用的轻量 tiling 数据定义，沿用仓库的 `aclblasHandle_t`、`GetAivCoreCount()`、stream launcher 与错误码约定。

1. 校验 `handle`，空值返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；校验 `trans` 仅为 `N/T/C`，维度及带宽非负、步长非零、`alpha/beta` 非空，非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. `m == 0 || n == 0` 时直接成功返回，不访问 Device 数据。任务书同时写有 `kl <= m-1`、`ku <= n-1`，与零维 no-op 在数学上冲突；设计按 no-op 优先处理带宽上界，负值仍拒绝，具体参数校验优先级在测试中与验收方确认。
3. 对非零维度继续检查 `kl < m`、`ku < n` 和 `lda >= kl + ku + 1`。用 64 位中间值计算带宽和；非零输出要求 `y != nullptr`。仅在 `alpha != 0` 时要求 `A/x != nullptr`。
4. `alpha == 0` 时采用只处理 `y = beta*y` 的 kernel 分支，不读取 `A/x`；`alpha == 0 && beta == 1` 可直接返回。`beta == 0` 时绝不读取原 `y`，避免未初始化值或 NaN 污染结果。
5. 计算 `outLen = (trans == N ? m : n)`，按可用 AI Core 数量、输出长度及 UB 容量分配 block；小 shape 减少 block 数，大 shape 让不同 block 处理互不重叠的输出区间。将矩阵维度、带宽、前导维、操作类型、标量的实虚部、步长及分核信息按值传给 kernel。
6. 使用 `handle` 当前 stream 启动 kernel 后返回，不在公共接口内同步 stream，不在每次调用中分配或释放大块 Device 工作区。

建议 tiling 信息包括 `m/n/kl/ku/lda`、`trans`、`incx/incy`、`alpha/beta` 的实虚部、`outPerBlock`、单次 UB 分块长度和可选的 `x` 缓存策略。字段类型以仓库现有 Host/Kernel ABI 为准；不新增公共 API 参数。`T` 与 `C` 共享地址区间计算，但保留独立的共轭计算分支。

#### Kernel 侧设计

首版采用 arch22 的 Ascend C 向量计算路径，按输出维度分核：每个 block 独占一段输出，在 UB 中完成复数乘加和规约，最后合并 `alpha`、`beta` 并写回该段 `y`。输出拥有者唯一，无跨核原子加、全局中间结果或第二次 launch。通过本地计算 `band_row`、`xOffset`、`yOffset`，仅加载有效地址；尾块用有效长度控制，不参与规约的 UB 元素初始化为零。

优化分支在精度基线通过后启用：

| 路径 | 使用条件 | 数据复用与优化重点 |
| --- | --- | --- |
| 通用步长路径 | 负步长、padding、窄/小 shape 或 UB 容量不足 | 按逻辑索引搬运有效元素，缩小小问题上的搬运与同步开销 |
| 连续向量路径 | `incx == 1` 且可容纳相关 `x` 片段 | 每个 block 将所需 `x` 范围缓存到 UB，复用到多个输出；容量不足时分块或回退通用路径 |
| 宽带 T/C 路径 | 带宽大且单输出规约成为瓶颈 | 评估按列分组及分段规约；只有实测更快且精度通过才采用 |

`N` 模式把相邻输出行组织为向量分块，对当前列的有效行段做带内搬运与复数乘加；`T/C` 模式按输出列分组，每列的有效带内元素连续，分段搬运后在 UB 内规约。UB 缓存只装载本 block 实际需要的 `x` 区间，尾块按有效范围处理。复数数据按交错的 `(real, imag)` 对读取/写回，避免把对齐搬运扩展到带外或用户未分配的区域。`C` 模式在复乘之前取 `ai` 的相反数。

### 支持硬件

| 产品 | 目标目录 | 验证安排 |
| --- | --- | --- |
| Atlas A2（含 Atlas 800T/800I A2） | `blas/gbmv/arch22/` | 910B3 为性能指定设备和首要自验平台 |
| Atlas A3（含 Atlas 800I A3） | `blas/gbmv/arch22/` | 与 A2 共用方案，完成任务书要求的适配/验收 |

### 算子约束限制

| 条件 | 行为 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 非法 `trans`、负维度/带宽、非零维度下带宽越界、`lda` 不足、零步长、空 `alpha/beta` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 非零维度下 `y == nullptr`，或参与计算时 `A/x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `m == 0 || n == 0` | 合法 no-op，不访问 `A/x/y` |
| `alpha == 0` | 不访问 `A/x`，仅计算 `beta*y` |
| `beta == 0` | 不读取原 `y`，直接写入 `alpha*op(A)*x` |
| `y` 与 `A/x` 内存重叠 | 不支持，由调用者保证 |

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 设计目标及验证方式 | 来源 |
| --- | --- | --- |
| 精度 golden | Netlib CBLAS `cgbmv`，比较全部有效输出元素；检查 N/T/C、实虚部和特殊值 | 任务书 §3.2、§3.5 |
| COMPLEX64 有限值 | 混合容差 `atol=rtol=2^-13`，`matched_ratio >= 0.99`，`max_abs_error <= max(1e-2, 32*ULP)`；记录逐元素误差和最大误差点 | 任务书 §3.2、[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| Inf/NaN | 按测试框架和精度标准的特殊值规则分别核对，有限值最大误差不计入非有限点 | 任务书 §3.5、生态算子精度标准 |
| 性能 | 910B3 使用 `msprof op` 统计单次 kernel 耗时，先 warmup，随后超过 10 次有效采样取平均，保存原始 profiling 和汇总 | 任务书 §3.3、§7 |
| 额外内存 | 以单次 launch、无全局临时工作区为目标，记录实际内存占用并提交报告；UB 用量受 tiling 约束 | 任务书 §4 |

随附测试指导要求复数的实部、虚部分别按 FLOAT32 混合容差判定。测试实现将分别记录两个分量的匹配率、最大绝对误差，并记录复数模长误差供核对；任务书以 COMPLEX64 列出阈值，最终统计粒度和 ULP 口径需与官方验收框架一致，不以自定义宽松判定替代。

任务书明确列出的性能门槛如下，均为 COMPLEX64、`incx=incy=1`；表中数字是**验收目标，不是实测结果**：

| m × n | kl/ku | trans | alpha/beta | Avg time 上限（us） |
| --- | --- | --- | --- | ---: |
| 256 × 256 | 16/16 | N | 1/0 | 10.665 |
| 512 × 512 | 32/32 | N | 1/1 | 17.328 |
| 1024 × 1024 | 64/64 | T | 1/0 | 25.875 |
| 2048 × 2048 | 128/128 | N | 1/0 | 30.640 |
| 4096 × 4096 | 64/256 | C | 1/0 | 52.487 |

随任务提供的 `gpu_baseline.csv` 另含 200 条性能用例。自验将记录全部 `TC_PF` 的 NPU kernel 时间，并按测试指导与 `gpu_ms / 0.8` 对照；不会将 GTest 墙钟时间代替 kernel 时间。

## 测试设计

测试代码落在 `test/gbmv/cgbmv/arch22/`，沿用 `ops-blas` 的 CSV/GTest 结构，独立构建 Netlib CBLAS golden。随任务用例为 1000 条精度和 200 条性能，至少覆盖：

| 类别 | 覆盖内容 |
| --- | --- |
| 基础与尺寸 | 小 shape、矩形矩阵、奇数、2 的幂及其邻值、大尺寸 |
| 转置与带宽 | N/T/C、`kl=0`、`ku=0`、单边/全带、非对称带宽、`lda` padding |
| 向量与标量 | `incx/incy=±1/±2/±3`、`alpha/beta=0/1/-1` 和一般复数 |
| 异常与特殊值 | 零维、空指针、非法枚举/维度/步长/前导维、Inf/NaN、`beta=0` 且原 `y` 为 NaN |
| 性能 | 任务书 5 条目标 case 和其余 `TC_PF` 扫描用例，单条用例 profiling |

测试框架应在 Device 结果读取前同步 stream，检查每条用例的返回码及输出。现有 `verify_accuracy.py` 只负责调用尚待实现的 GTest，并不能单独证明精度通过；需要核对运行的 case 数、进程退出码和超时结果。性能脚本当前未随任务提供，需在代码阶段补齐采样、CSV 对齐和日志归档。

## 兼容性分析

- 接口参数顺序、`trans`、列主序带状布局和正负步长与 `cublasCgbmv` 对齐；不添加私有变体或改变现有 BLAS 接口。
- 公共声明放入 `include/cann_ops_blas.h`；A2/A3 实现放入 `blas/gbmv/arch22/`，测试放入 `test/gbmv/cgbmv/arch22/`。复用现有句柄、stream 和构建规则，不影响 arch35 的 `sgbmv`。
- 算子 README 产品支持表标注 Atlas A2/A3 系列（含 Atlas 800I A2/A3）。

## 设计评审与后续交付

本设计文档先提交到 `cann/cann-ops-competitions` 对应任务目录评审。评审通过后开发代码，在 910B3/A3 环境完成真实精度和性能自验，按任务书模板提交日志、截图、报告与个人仓库地址；收到代码合入通知后再向 `cann/ops-blas` 提交实现和测试 PR。本文不填写未经运行的通过率、耗时或性能提升。

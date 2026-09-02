# aclblasCtpmv 算子设计文档

# 需求背景（required）

## 需求来源

本需求来自 CANN 2026 年 8 月社区任务第 41 号“aclblasCtpmv 算子开发（A2/A3）”。目标是在 Atlas A2/A3 系列产品上，基于 `cann/ops-blas` 工程和 Ascend C Kernel 直调模式，实现单精度复数三角压缩存储矩阵-向量乘接口 `aclblasCtpmv`。

接口参数和核心语义与 cuBLAS `cublasCtpmv` 对齐，packed 存储、转置、单位对角、负步长和 quick return 语义参考 Netlib `ctpmv`。开发完成后，公共接口声明放入 `include/cann_ops_blas.h`，A2/A3 实现放入 `blas/tpmv/arch22/`，测试放入 `test/tpmv/ctpmv/arch22/`。

## 背景介绍

### 算子功能

`aclblasCtpmv` 属于 BLAS Level-2，计算：

```text
x := op(A) * x
```

其中 `A` 是 `n × n` 的单精度复数三角矩阵，按 packed 列优先格式存储于长度为 `n(n+1)/2` 的数组 `AP` 中；`x` 是逻辑长度为 `n` 的单精度复数向量，结果原地写回 `x`。

- `trans = ACLBLAS_OP_N`：`op(A) = A`；
- `trans = ACLBLAS_OP_T`：`op(A) = A^T`，不共轭；
- `trans = ACLBLAS_OP_C`：`op(A) = A^H`，计算前对矩阵元素取共轭；
- `uplo` 指定引用上三角或下三角；
- `diag = ACLBLAS_UNIT` 时主对角按 `(1, 0)` 处理，且不得读取 `AP` 中的对角槽；
- `incx` 支持正、负非零步长。

接口原型如下：

```cpp
aclblasStatus_t aclblasCtpmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const aclblasComplex* AP,
    aclblasComplex* x,
    int incx);
```

### 现有基础与主要难点

`ops-blas` 已有同族实数接口 `aclblasStpmv`，可作为公共句柄、stream、状态码、工程目录和 Kernel 直调方式的参考。本任务在此基础上补充 complex64 能力，主要关注：

1. 复数乘法以及 `OP_T`、`OP_C` 的差异；
2. packed 三角存储的地址计算和连续搬运组织；
3. `x` 原地覆写时保持全部输入元素在计算期间可用；
4. `diag = UNIT` 时从访问路径上跳过主对角；
5. 正负 `incx`、边界尺寸和未参与位置的保护；
6. 在保持完整通用语义的前提下满足任务书三组性能门槛。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 `aclblasCtpmv`，支持 UPPER/LOWER、N/T/C、UNIT/NON_UNIT、正负步长、参数异常处理和 `n = 0` quick return；使用任务书提供的测试框架完成精度、功能、异常和性能验证，并在 Atlas 800I A2（Ascend 910B3）、CANN 9.1.0 环境达到规定的性能标准。

## 需求拆解

1. 新增共享的 `aclblasCtpmv` API，不定义 A2/A3 私有平行接口。
2. 支持 COMPLEX64 数据类型和三种 `trans` 模式。
3. 支持上、下三角 packed 列优先存储，只引用 `uplo` 指定区域。
4. 支持 UNIT/NON_UNIT；UNIT 模式不得读取 `AP` 对角槽。
5. 支持 `incx = ±1/±2/±3` 等正负非零步长，拒绝零步长。
6. 保证原地输出正确，步长空洞位置不被修改。
7. `n = 0` 时直接成功返回，不访问 `AP`、`x`，不启动 Kernel。
8. 参数错误返回任务书规定的状态码，接口保持异步执行。
9. 提供覆盖全部合法参数的通用路径，并对连续 packed 访问和正式性能场景规划优化路径。
10. 按官方 CSV/GTest、Netlib/cblas golden 和任务书精度标准完成自验。

# 详细设计（required）

## 算子分析

### 数学公式

对输出元素 `y(i)`：

```text
trans = N: y(i) = sum_j A(i, j) * x(j)
trans = T: y(i) = sum_j A(j, i) * x(j)
trans = C: y(i) = sum_j conj(A(j, i)) * x(j)
```

求和范围只包含 `uplo` 指定的三角。复数乘法展开为：

```text
(a + b*i) * (c + d*i) = (a*c - b*d) + (a*d + b*c)*i
```

packed 列优先地址采用 Netlib/cuBLAS 的连续无空洞布局，0-based 公式为：

```text
UPPER（i <= j）：AP[i + j*(j+1)/2]
LOWER（i >= j）：AP[i + j*(2*n-j-1)/2]
```

任务书 LOWER 公式的文字表达与上述连续布局存在歧义，设计阶段将以 Netlib/cblas 行为为准，并用 `n = 1/2/3` 手工索引用例冻结规则；如评审口径不同，将按评审意见同步修订文档与测试。

负步长按 Netlib 语义确定逻辑首元素。0-based 下：

```text
incx > 0：x(i) 位于 i * incx
incx < 0：x(i) 位于 (n - 1 - i) * abs(incx)
```

地址和 workspace 大小计算使用足够宽的无符号整数，并在 Host 侧检查溢出。

### 支持数据类型

| 操作数 | 数据类型 | 说明 |
| --- | --- | --- |
| `AP` | `aclblasComplex` | Device 侧 complex64 packed 三角矩阵，只读 |
| `x` | `aclblasComplex` | Device 侧 complex64 向量，原地输入/输出 |
| `n`、`incx` | `int` | Host 侧维数和步长 |
| `uplo`、`trans`、`diag` | 枚举 | Host 侧属性 |

### 支持形状

| 对象 | 逻辑形状 | 物理约束 |
| --- | --- | --- |
| `AP` | `[n*(n+1)/2]` | packed 列优先，仅包含一个三角 |
| `x` | `[n]` | 至少 `1 + (n-1)*abs(incx)` 个 complex64 元素 |

`n` 为运行时参数；本算子不涉及广播、矩形矩阵或 `lda`。

## 算子实现

### 实现方案

整体采用“Host 参数与路径调度 + 输入快照 + Ascend C AIV 计算”的方案。设计阶段先完成覆盖全部合法参数的通用路径，再基于 910B3 Profiler 数据优化连续 packed 搬运、分核、UB 使用和向量流水。三组正式性能场景可进入优化路径，其余组合始终由通用路径保证正确性。

#### Host 侧设计

1. 按接口约定依次校验 `handle`、枚举、`n`、`incx` 和指针：
   - 空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   - 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；
   - `n < 0` 或 `incx == 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
   - `n == 0` 在指针检查前 quick return；
   - `n > 0` 时空 `AP` 或空 `x` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. 使用 handle 管理的 workspace 保存逻辑连续的输入快照和必要的中间结果，避免原地写回破坏尚未使用的旧 `x`。
3. TilingData 记录 `n`、`incx`、枚举、有效核数、单核任务范围和尾块信息。
4. 路径选择只依据参数组合、有效工作量、对齐和硬件资源；功能正确性不依赖性能路径。
5. 数据搬运和 Kernel 启动全部加入 handle 绑定的 stream，Host 不隐式同步。

#### Kernel 侧设计

通用 Kernel 计划覆盖全部合法组合：

1. 按 signed stride 把输入 `x` 映射为不可变逻辑快照；
2. 根据 `uplo`、`trans` 和 `diag` 计算每个输出元素对应的 packed 区间；
3. UNIT 模式在生成搬运区间和取数分支时直接跳过对角地址；
4. 在 UB 中完成 complex64 拆分、复数乘加和结果合并；
5. 结果按 `incx` 写回原向量，仅修改逻辑元素位置；
6. 尾块使用 mask/padding 处理，保证不越界。

性能优化方向按两类访问组织：

- `OP_N`：优先按 packed 列的连续区间搬运，再将列贡献累加到输出片段；
- `OP_T/OP_C`：优先按输出对应的 packed 列或列片段做连续点积归约，`OP_C` 在计算阶段处理共轭；
- 单位步长使用连续大块搬运，非单位步长保留 gather/scatter 兼容路径；
- 依据三角工作量进行多核划分，减少核间不均衡和跨核写冲突；
- 根据 UB 容量和实际流水重叠收益确定 tile、缓冲数量和事件同步，不预先承诺固定常量。

### Tiling 通用性

Tiling 由运行时参数、packed 有效元素数、输出任务数、AIV 核数和 UB 容量共同决定。正式性能 shape 可以触发相应的 size-class 优化配置，但不会缩窄 API 的合法范围。禁止让非性能 shape 因未命中优化路径而返回错误或产生不同语义。

## 支持硬件

| 芯片版本 | 支持情况 | 验证计划 |
| --- | --- | --- |
| Atlas 800I/T A2（Ascend 910B3） | 支持 | CANN 9.1.0；完成正式功能、精度和性能验证 |
| Atlas A3（arch22） | 支持 | 使用同一公共 API 和 arch22 源码完成构建、功能与边界验证 |

## 算子约束限制

1. 仅支持 COMPLEX64，不支持混合精度。
2. `A` 仅支持三角 packed 列优先存储，不存在 `lda` 参数。
3. `n >= 0`，`incx != 0`；负步长按 Netlib 语义处理。
4. `diag = UNIT` 时不得读取 `AP` 对角槽。
5. `x` 为原地输入/输出，调用方负责输入输出 Device 内存和 stream 生命周期合法。
6. `n == 0` 为合法 no-op，不访问 Device 数据。

# 可维可测分析

## 精度标准/性能标准

### 精度标准

golden 使用 Netlib/cblas `ctpmv` 生成。对 `x` 的逻辑元素逐个比较，实部、虚部分别按 FLOAT32 标准判定：

| 指标 | 标准 |
| --- | --- |
| `rtol` | `2^-10`（约 `9.77e-4`） |
| `atol` | `2^-16`（约 `1.53e-5`） |
| required matched ratio | `>= 0.99` |
| max abs error | `<= 1e-2` 或 `<= 32 ULP` |

逐元素条件为：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

每条用例需同时满足 matched ratio 和 max abs error 要求。UNIT 对角使用 NaN 哨兵验证“不读取”，步长空洞使用 canary 验证“不修改”，Inf/NaN 按官方测试规则比对。

### 性能标准

正式性能设备为 Atlas 800I A2（Ascend 910B3），CANN 9.1.0，COMPLEX64 输入：

| case | n | uplo | trans | diag | incx | 门槛（Avg time，us） |
| --- | ---: | --- | --- | --- | ---: | ---: |
| 1 | 512 | UPPER | N | NON_UNIT | 1 | `<= 18.53` |
| 2 | 1024 | LOWER | N | NON_UNIT | 1 | `<= 39.34` |
| 3 | 2048 | UPPER | T | NON_UNIT | 1 | `<= 86.02` |

正式计时使用 Release 构建（`-O3 -DNDEBUG`），先 warmup，再进行超过 50 次有效采样并取平均值。以设备侧 Kernel/Task 时间为主要依据，保留 CANN、驱动、设备、代码提交、构建选项和原始 Profiler 数据；Debug（`-O0 -g`）只用于定位问题，不作为性能结论。

### 测试覆盖

| 类别 | 覆盖内容 |
| --- | --- |
| 枚举组合 | UPPER/LOWER × N/T/C × UNIT/NON_UNIT 全覆盖 |
| 尺寸 | `n=0/1`、小质数、2 的幂及 ±1、非对齐值和大规模 |
| 步长 | `incx=±1/±2/±3`，零步长负向用例和空洞 canary |
| 对角与三角 | UNIT 对角 NaN 投毒、上下三角 packed 手工索引 |
| 数据 | 均匀/正态分布、全零、交替值、极端值、Inf/NaN |
| 接口 | 空 handle、非法枚举、负 `n`、空指针、quick return |
| 原地行为 | 输入快照、重复 launch、结果确定性和边界保护 |
| 性能 | 三条正式性能门槛及任务书提供的参考性能用例 |

官方测试 CSV、golden 和 GTest 框架保持原样；新增辅助测试只用于补充手工索引、sentinel、随机回归和 Profiler 结果整理。

## 兼容性分析

1. API 兼容：只新增 `aclblasCtpmv` 公共符号，不修改已有结构体、状态码和其它接口。
2. 产品兼容：A2/A3 共用 `arch22` 实现，不新增产品私有接口，不影响其它架构目录。
3. 行为兼容：参数顺序与 cuBLAS `cublasCtpmv` 对齐；packed、转置、UNIT 对角、负步长和 quick return 语义与 Netlib/cblas 对齐。
4. 工程兼容：实现和测试分别归档到 `blas/tpmv/arch22/` 与 `test/tpmv/ctpmv/arch22/`，复用 ops-blas 现有构建与测试框架。
5. 性能兼容：通用路径覆盖全部合法输入；优化路径只影响执行效率，不改变接口语义和精度判定。

## 本 PR 范围

本 PR 只提交任务初期的算子设计文档，不包含算子代码、自测结果、性能达标结论或验收材料。设计评审通过后，再按社区流程开展开发、自测和 IT 验收；收到验收通过通知后，另行向 `cann/ops-blas` 提交代码 PR。

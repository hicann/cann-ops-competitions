
# aclblasCtpmv 算子设计文档

# 需求背景（required）

## 需求来源

本任务来自 CANN 2026 年 8 月社区任务，要求在 Atlas A2/A3 系列产品上使用 Ascend C 实现单精度复数三角压缩矩阵-向量乘算子 aclblasCtpmv，并完成公共 API、Host、Kernel、精度、异常和性能验证。

接口语义与 cuBLAS cublasCtpmv 保持一致，计算基线采用 Netlib BLAS 的 ctpmv 实现。代码使用 ops-blas 的句柄式 BLAS 接口和 Ascend C Kernel 直调框架。

## 背景介绍

TPMV 是 BLAS Level-2 三角矩阵-向量运算。n 阶三角矩阵只保存上三角或下三角的 n(n+1)/2 个复数，并按照列优先规则连续压缩到一维 AP 中。与 TBMV 不同，TPMV 没有带宽参数 k，也没有前导维参数 lda。

本任务在仓内已有实数版 aclblasStpmv 的同族目录中补充 complex64 能力，需要同时处理复数乘加、N/T/C 三种操作、UNIT 对角不读、正负 incx、原地输出以及 packed 下标计算。公共接口放在 include/cann_ops_blas.h 中，与其他产品线共用，不定义 Atlas A2/A3 私有平行 API。

### 接口现状分析

接口原型与任务书保持一致：

~~~cpp
aclblasStatus_t aclblasCtpmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const aclblasComplex* AP,
    aclblasComplex* x,
    int incx);
~~~

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状或内存布局 |
| --- | --- | --- | --- | --- | --- |
| handle | BLAS 计算上下文 | aclblasHandle_t | 有效句柄 | 不为空 | Host 句柄，携带 stream |
| uplo | 三角存储模式 | aclblasFillMode_t | ACLBLAS_UPPER / ACLBLAS_LOWER | 只能取合法枚举 | Host 标量 |
| trans | 矩阵操作类型 | aclblasOperation_t | ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C | 只能取合法枚举 | Host 标量 |
| diag | 对角线类型 | aclblasDiagType_t | ACLBLAS_NON_UNIT / ACLBLAS_UNIT | 只能取合法枚举 | Host 标量 |
| n | 矩阵阶数和逻辑向量长度 | int | int32 | n >= 0 | 标量 |
| AP | packed 三角矩阵 | const aclblasComplex* | COMPLEX64 | n > 0 时不为空，只读 | [n(n+1)/2]，列优先 packed |
| x | 输入/输出向量 | aclblasComplex* | COMPLEX64 | n > 0 时不为空，原地更新 | 物理长度为 1 + (n-1) * abs(incx) |
| incx | x 的逻辑元素步长 | int | int32 | 非零，支持正负值 | Host 标量 |

### 算子功能分析

1. n < 0 返回 ACLBLAS_STATUS_INVALID_VALUE。
2. n = 0 是合法 no-op，直接返回 ACLBLAS_STATUS_SUCCESS，不访问 AP 和 x，也不发起 Kernel。
3. n > 0 时，仅读取 uplo 指定的 packed 三角区域。
4. trans = ACLBLAS_OP_N 计算 A*x；trans = ACLBLAS_OP_T 计算 A^T*x；trans = ACLBLAS_OP_C 计算 A^H*x。
5. diag = ACLBLAS_UNIT 时主对角元素视为 1+0i，不能读取 AP 中对应的对角槽；diag = ACLBLAS_NON_UNIT 时读取 AP 对角元素。
6. 输出覆盖 x 的 n 个逻辑元素，incx 的物理空洞保持不变。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Atlas A2/A3 的 arch22 平台实现 aclblasCtpmv，支持 complex64 packed 三角矩阵和 complex64 向量的原地矩阵-向量乘，完成如下功能：

- UPPER/LOWER 两种 packed 列优先存储；
- OP_N、OP_T、OP_C 三种操作；
- UNIT/NON_UNIT 两种对角线语义；
- 正负非零 incx；
- n = 0 quick return 和非法参数状态码；
- 与 Netlib/cblas ctpmv 的复数实部、虚部精度口径一致；
- 满足任务书三组绝对性能门槛。

## 需求拆解

1. 在 include/cann_ops_blas.h 中新增共享 aclblasCtpmv 声明，并与 aclblasStpmv 和 cublasCtpmv 逐参数对齐。
2. 在 Host 侧完成句柄、枚举、n、指针、incx 和资源长度校验。
3. 使用 64 位中间量计算 packed AP 长度、x 物理跨度和 workspace 大小，避免整数回绕。
4. 正确解析 UPPER/LOWER packed 下标，并覆盖 N/T/C、UNIT/NON_UNIT 的 12 组组合。
5. 在 x 被覆盖之前完成输入向量快照，保证原地更新不会改变后续读数据。
6. 以 AIV 多核和 UB 分块完成复数乘加，尾块和非 32B 对齐数据使用独立处理路径。
7. 对正负 incx 执行逻辑向量的 gather 和结果 scatter，保护物理空洞。
8. 使用 Netlib/cblas golden、CSV、C++ GTest 和设备计时完成精度、异常、内存安全和性能闭环。

# 详细设计（required）

## 算子分析

### 数学公式

设 A 为 n 阶三角复数矩阵，x 为逻辑向量，则：

~~~text
x <- op(A) * x

op(A) =
    A      trans = ACLBLAS_OP_N
    A^T    trans = ACLBLAS_OP_T
    A^H    trans = ACLBLAS_OP_C
~~~

对输出逻辑元素 i：

~~~text
y[i] = sum over valid j of op(A)[i,j] * x[j]
x[i] = y[i]
~~~

复数乘加在 FLOAT32 分量上展开：

~~~text
real += ar * xr - ai * xi
imag += ar * xi + ai * xr
~~~

trans = ACLBLAS_OP_C 时先对矩阵系数虚部取反，再执行上述公式。累加顺序按固定的逻辑列顺序执行，避免不必要的数值顺序变化。

x 的物理位置以分配首地址为基准。第 k 个逻辑元素的复数位置为：

~~~text
p(k) = k * incx                         incx > 0
p(k) = (n - 1 - k) * abs(incx)          incx < 0
~~~

其中 k 的范围为 0..n-1。每个复数占用两个 FLOAT32。

### AP packed 存储规则

UPPER 模式保存每列的行 0..j：

~~~text
APIndexUpper(i,j) = j * (j + 1) / 2 + i
                   0 <= i <= j < n
~~~

LOWER 模式的第 j 列从对角线开始保存行 j..n-1。先计算列起点，再加列内偏移：

~~~text
columnStartLower(j) = j * (2 * n - j + 1) / 2
APIndexLower(i,j) = columnStartLower(j) + (i - j)
                   = i + j * (2 * n - j - 1) / 2
                   j <= i < n
~~~

所有下标和长度计算使用 UINT64 中间量，并在形成 AP 地址前先完成三角区域合法性判断。未存储的另一三角不会生成 AP 访问。

trans 到原矩阵坐标的映射如下：

| trans | op(A) 元素对应的原矩阵坐标 | 是否共轭 |
| --- | --- | --- |
| ACLBLAS_OP_N | A(i,j) | 否 |
| ACLBLAS_OP_T | A(j,i) | 否 |
| ACLBLAS_OP_C | A(j,i) | 是 |

### 支持数据类型

| 数据 | 类型 |
| --- | --- |
| AP | COMPLEX64 / aclblasComplex |
| x 输入 | COMPLEX64 / aclblasComplex |
| x 输出 | COMPLEX64 / aclblasComplex |
| Device 计算分量 | FLOAT32 real/imag |

### 支持形状

~~~text
AP: packed vector, n * (n + 1) / 2 complex elements
x:  physical vector, 1 + (n - 1) * abs(incx) complex elements
x logical view: n complex elements
~~~

n 是运行时参数。API 的 n 语义范围为 0 <= n <= INT_MAX；最终测试重点覆盖 0 <= n <= 4096，不将测试覆盖上限误写为 API 最大值。

## 算子实现

### 实现方案

实现采用 ops-blas 现有 BLAS 直调框架，整体分为 Host 参数处理、输入向量连续化、AIV Kernel 分块计算和结果写回四个阶段。所有异步操作使用 handle 绑定的同一 stream。

~~~text
aclblasCtpmv
    -> 参数检查和 checked arithmetic
    -> n == 0 quick return
    -> gather x 到连续逻辑快照
    -> arch22 AIV 主 Kernel
    -> copyback 或 scatter 结果到 x
~~~

#### 3.2.1 Host 侧设计

1. 先检查 handle。handle 为空返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR。
2. 检查 n。n < 0 返回 ACLBLAS_STATUS_INVALID_VALUE；n = 0 直接返回成功，不引用 AP、x 或 workspace。
3. 对 n > 0 检查 uplo、trans、diag 枚举。非法枚举返回 ACLBLAS_STATUS_INVALID_ENUM。
4. 检查 AP、x 和 incx。AP 或 x 为空，或 incx = 0，返回 ACLBLAS_STATUS_INVALID_VALUE。
5. 将 n 和 incx 提升到 64 位后计算：
   - packedElems = n * (n + 1) / 2；
   - absIncx = abs(int64(incx))；
   - physicalSpan = 1 + (n - 1) * absIncx；
   - AP、x、snapshot、y 和 workspace 的字节数。
6. 任一乘加、对齐或 size_t 转换不可表示时，不发起 Kernel，返回 ACLBLAS_STATUS_INVALID_VALUE。
7. 使用 handle 既有 workspace 管理策略取得持久 workspace，不在每次调用中执行无法安全异步释放的 aclrtMalloc/aclrtFree。
8. 为 xSnapshot 和 yContiguous 规划互不重叠、按 32B 对齐的区域。xSnapshot 保存 n 个按逻辑顺序排列的 complex64，yContiguous 保存 n 个输出元素。
9. 先在 handle stream 上发起 gather。incx = 1 使用连续搬运；其他步长按照 p(k) 将物理 x 收集到连续 xSnapshot。
10. 在同一 stream 上发起主 Kernel。主 Kernel 完成后，incx = 1 使用连续 copyback，其他步长发起单核 scatter，将 yContiguous 写回 p(k) 对应位置。
11. API 返回 SUCCESS 仅表示任务已按 stream 入队。调用者在同步前必须保持 handle、AP、x 和 workspace 有效。

workspace 的规划大小为：

~~~text
workspaceBytes >= align32(8 * n) + align32(8 * n)
~~~

若实现为连续 incx=1 路径保留直接输入别名，仍须保证主 Kernel 的输出区域与正在读取的 x 不重叠；默认设计使用 xSnapshot 统一隔离原地读写。

#### 3.2.2 Kernel 侧设计

Kernel 由 Init 和 Process 两部分组成。Process 按 CopyIn、Compute、CopyOut 组织，并以输出逻辑行作为并行所有权单位。

1. 初始化 AP、xSnapshot、yContiguous 的 GM 视图和本核 UB 队列。
2. 按 uplo/trans 确定每个输出 row 的有效列区间。
3. 清零当前输出 tile 的 real/imag 累加器；diag = UNIT 时先注入对应 xSnapshot 元素作为单位对角贡献。
4. 将 AP 的连续段、非连续短段和对应 xSnapshot 列块搬入 UB，完成 AoS 到 real/imag 的分离。
5. 对每个有效矩阵元素执行复数乘加。trans = C 时仅改变矩阵系数虚部符号。
6. 当前 row tile 计算完成后，把 real/imag 交错为 complex64 并写入 yContiguous 的唯一对应位置。
7. 主 Kernel 不直接写 x，不使用跨 AIV atomic 作为正确性基础；每个输出逻辑元素只有一个 owner。
8. yContiguous 完成后，由 copyback/scatter 阶段按 incx 写回 x，物理空洞不被修改。

对于每一行，非转置和转置操作的有效区间为：

| 场景 | 原矩阵坐标 | 有效 j 范围 |
| --- | --- | --- |
| N + UPPER | A(i,j) | i..n-1 |
| N + LOWER | A(i,j) | 0..i |
| T/C + UPPER | A(j,i) | 0..i |
| T/C + LOWER | A(j,i) | i..n-1 |

判断 UNIT 对角必须早于 AP 地址计算和 AP 搬运。对角位置使用 1+0i，不能因为 AP 中填有 NaN 或 Inf 而改变输出。

#### 3.2.3 数据搬运和复数计算

AP 采用 float raw view 搬入 UB，避免把 aclblasComplex 当作未经验证的专用 DataCopy 数据类型。搬运按以下顺序处理：

1. 对齐且连续的主体段使用连续 DataCopy；
2. 可合并的相邻 AP 段合并后批量搬运；
3. 首尾不足 32B 的段使用 DataCopyPad 或等效的显式尾块处理；
4. 只有无法合并的短段使用受控的索引/标量访问，不能让其替代主体搬运路径。

每个复数在 UB 中形成 ar、ai、xr、xi 四个 FLOAT32 分量。普通输入使用向量算术完成乘减和乘加，尾块通过有效元素数 mask 控制。所有无效 lane 在计算前清零，不读越界的 AP、x 或 UB 区域。

为满足精度要求，计算不使用近似倒数、低精度中间结果或改变复数乘法语义的替换。对 Inf、NaN、带符号零和极端有限值保留 IEEE FLOAT32 传播语义；必要的非有限分类路径与普通有限值向量路径分离，并以 Netlib/cblas golden 作为唯一判定依据。

#### 3.2.4 Tiling 和多核策略

环境目标为 DAV_2201/arch22，最多使用 40 个 AIV。根据 n 动态选择核数：

~~~text
useCoreNum = min(n, 40)
n < 32 时 useCoreNum = min(n, 8)
~~~

输出 row 的有效工作量按操作类型估算：

| 场景 | 第 i 行的工作量 |
| --- | --- |
| N + UPPER、T/C + LOWER | n - i |
| N + LOWER、T/C + UPPER | i + 1 |

以三角工作量前缀和划分 row 区间，使各 AIV 的有效复数乘加数量接近总工作量除以 useCoreNum。小 n 减少空核，大 n 避免只按 row 数平均造成三角端长尾。

初始 Tiling 参数采用 rowTile = 1024、columnTile = 128，并根据 UB 可用空间和边界动态缩小。TilingData 至少包含：

| 字段 | 用途 |
| --- | --- |
| n、packedElems、physicalSpan | 逻辑长度、AP 长度、x 物理跨度 |
| incx、uplo、trans、diag | 访问和计算属性 |
| useCoreNum、rowTile、columnTile | 多核和 UB 分块 |
| rowBegin/rowEnd 或等价分区参数 | 本核输出范围 |
| xSnapshotOffset、yOffset、workspaceBytes | workspace 生命周期和边界 |
| tailRows、tailColumns | 尾 tile 有效长度 |

#### 3.2.5 Buffer 规划和流水

| Buffer | 用途 | 规划 |
| --- | --- | --- |
| xSnapshot | 连续的逻辑输入向量 | 8*n 字节，主 Kernel 只读 |
| yContiguous | 连续输出向量 | 8*n 字节，主 Kernel 按 row 唯一写 |
| AP tile | packed 矩阵局部数据 | float AoS staging，按 row/column tile 分块 |
| A real/imag | 解包后的矩阵系数 | UB 内分量视图 |
| x real/imag | 当前输入列块 | UB 内分量视图 |
| acc real/imag | 当前输出 row tile 累加器 | UB 内双缓冲或阶段缓冲 |
| temporary/mask | 复数运算中间值和尾块 mask | 根据 tile 大小保留安全余量 |

数据流为：

~~~text
CopyIn x -> xSnapshot
CopyIn AP tile and x column tile
    -> Vector complex multiply-add
    -> Store yContiguous
CopyOut yContiguous -> x or strided scatter -> x
~~~

A tile 搬入与当前 tile 计算使用双缓冲，MTE2 和 Vector 阶段通过队列依赖和必要的 PipeBarrier 建立顺序。所有 Buffer 的起址、跨度和尾段长度必须在 Tiling 阶段核算，禁止越过 UB 或 workspace 边界。

#### 3.2.6 异步和错误处理

- handle 绑定的 stream 是唯一的执行顺序来源，gather、主 Kernel、copyback/scatter 按同一 stream 顺序入队。
- Kernel 不依赖跨 stream 的隐式同步，也不要求 API 内执行全设备同步。
- Host 侧 launch、workspace 或异步复制失败时映射为仓库约定的 BLAS 状态码。
- 用户在 stream 同步前不得释放或复用 AP、x、handle workspace。
- AP 只读，x 的物理空洞不写，输出只写合法的 n 个逻辑元素。

## 支持硬件

| 支持的芯片版本 | 架构 | 涉及勾选 |
| --- | --- | --- |
| Atlas 800I A2 / Ascend 910B3 | DAV_2201 / arch22 | √ |
| Atlas 800I A3 系列 | DAV_2201 / arch22 | √ |

CANN 版本按任务书使用 9.1.0。SIMD 是本目标架构的实现范式，Kernel 采用 MemBase 风格的 GM/UB 数据搬运和 AIV 向量计算；不使用仅适用于 ascend950 的 RegBase/Cube/Blaze 路线。

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 数据类型 | AP、x 均为 COMPLEX64，实部和虚部为 FLOAT32 |
| 存储格式 | 仅支持 UPPER/LOWER packed 列优先，不含 lda |
| 操作类型 | 支持 OP_N、OP_T、OP_C |
| 对角线 | 支持 UNIT/NON_UNIT；UNIT 对角不读 AP |
| n | API 语义为 n >= 0；n = 0 为合法 no-op |
| incx | incx != 0，支持正负步长；地址计算使用 64 位中间量 |
| 原地语义 | x 原地覆写，物理空洞保持原值 |
| 批处理和广播 | 不支持，不涉及 AP/x 广播 |
| 非连续 Tensor | 不要求超出 incx 语义的额外 view 或任意布局 |
| dynamic shape | n 为运行时标量，不引入额外 dynamic-shape Tensor 约束 |
| alpha、矩形矩阵和 lda | tpmv 接口没有 alpha、m、矩形维度或 lda；任务书中相关用例不适用于本算子 |
| 确定性 | 不额外承诺跨调用的确定性；单个输出 row 由唯一 owner 按固定顺序累加 |
| 资源错误 | packed 长度、物理跨度或 workspace 算术不可表示时返回 INVALID_VALUE，不发起 Kernel |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | x 的 real 和 imag 作为两个独立 FLOAT32 分量，分别与 Netlib/cblas ctpmv golden 比对 | 任务书 3.2、生态算子精度标准 |
| 性能标准 | COMPLEX64 三组硬性能 case 的平均 Device 耗时不高于任务书标杆 | 任务书 3.3 |
| 状态码标准 | 合法和非法参数返回值与任务书及 include/cann_ops_blas_common.h 一致 | 任务书 2.4、仓库公共头文件 |

精度逐元素条件为：

~~~text
abs(actual - golden) <= atol + rtol * abs(golden)
~~~

实部、虚部各自采用：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64 的 FLOAT32 real/imag 分量 | 2^-10 = 9.765625e-4 | 2^-16 = 1.52587890625e-5 | 0.99 | 1e-2 或 32 * ULP |

Inf、NaN 和带符号零按测试工程的权威比较规则处理；UNIT 对角 golden 同样不读取 AP 对角槽。任何一个分量未达到匹配率、最大误差或非有限值匹配要求，该用例判定失败。

性能测试使用 Atlas 800I A2（910B3）设备，先执行 100 次 warmup，再执行 100 次有效采样取平均值。三组绝对硬门槛为：

| case | n | uplo | trans | diag | incx | 标杆耗时（Avg time, us） |
| --- | ---: | --- | --- | --- | ---: | ---: |
| 1 | 512 | UPPER | N | NON_UNIT | 1 | 18.53 |
| 2 | 1024 | LOWER | N | NON_UNIT | 1 | 39.34 |
| 3 | 2048 | UPPER | T | NON_UNIT | 1 | 86.02 |

测试脚本提供的 gpu_ms / npu_ms 比值或 HAP 只作补充回归指标，不能替代上述绝对 us 门槛。存在可匹配基线时同时报告 gpu_ms、npu_ms 和 HAP；基线缺失或为 NO_REF 时只标记无补充参考，不豁免硬门槛。

## 测试设计

### 精度测试

golden 使用 cblas/Netlib ctpmv，使用与被测接口相同的 uplo、trans、diag、n、AP、x 和 incx。成功用例在 NPU 输出同步后，对 x 的 n 个逻辑元素逐一读取，并分别比较 real、imag。

覆盖内容如下：

| 编号 | 场景 | 说明 |
| --- | --- | --- |
| TC-01 | 基础全组合 | UPPER/LOWER x OP_N/OP_T/OP_C x UNIT/NON_UNIT，共 12 组 |
| TC-02 | 尺寸边界 | n = 0、1、2、3、小质数、2 的幂及相邻非对齐值，扩展到 2048 |
| TC-03 | 正负步长 | incx = +1、-1、+2、-2、+3、-3，校验逻辑元素和物理空洞 |
| TC-04 | 数据分布 | 均匀、正态、全零、交替、极端有限值 |
| TC-05 | 非有限值 | AP/x 中的 Inf、NaN、带符号零，校验与 golden 的传播结果 |
| TC-06 | UNIT 对角投毒 | AP 对角填 NaN + NaN*i，12 组属性和步长均不得传播该值 |
| TC-07 | 内存保护 | AP/x 首尾 canary、x 物理空洞、尾 tile 边界和 workspace 边界 |

### 异常和边界测试

| 编号 | 场景 | 预期结果 |
| --- | --- | --- |
| TC-08 | n = 0，AP/x 为空 | ACLBLAS_STATUS_SUCCESS，且不访问 AP/x |
| TC-09 | n < 0 | ACLBLAS_STATUS_INVALID_VALUE |
| TC-10 | n > 0 且 AP 为空 | ACLBLAS_STATUS_INVALID_VALUE |
| TC-11 | n > 0 且 x 为空 | ACLBLAS_STATUS_INVALID_VALUE |
| TC-12 | uplo、trans 或 diag 为非法枚举 | ACLBLAS_STATUS_INVALID_ENUM |
| TC-13 | incx = 0 | ACLBLAS_STATUS_INVALID_VALUE |
| TC-14 | handle 为空 | ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| TC-15 | checked arithmetic 或 workspace 不可满足 | ACLBLAS_STATUS_INVALID_VALUE 或仓库约定的分配失败状态 |

### 性能测试

性能测试使用 ops-blas 的 C++ GTest/CSV 框架和设备 event 计时。Host 数据准备、H2D、D2H、golden 计算、精度比对和 warmup 不计入硬门计时区间；每组保留原始样本、平均值、最小值、最大值、离散度和 kernel 阶段拆分。

除总耗时外，性能报告至少记录：

- AP 搬入、x 搬入、结果搬出和计算的耗时或占比；
- AIV Vector、MTE 和 Scalar 的设备侧占比；
- 各 AIV 的任务量和最大/最小核耗时；
- workspace 使用量、tile 参数和实际 launch 核数；
- 三组 hard case 的 actual_us <= limit_us 判定。

### 自测要求

1. 测试工程固定放在 test/tpmv/ctpmv/arch22/，CSV 列与仓内同族测试参数解析方式一致。
2. 运行精度测试前确认 CBLAS/Netlib golden 可用，不能以自建宽松容差替代权威精度口径。
3. 运行完整的 12 组属性组合、尺寸、步长、特殊值、异常和 UNIT 对角投毒用例。
4. 性能用例先 warmup，再完成不少于 50 次有效采样；本方案的标准采集口径为 100 次 warmup 加 100 次有效采样。
5. 测试报告同时给出用例参数、状态码、real/imag 分量指标、性能数据和设备环境。
6. 生产代码和测试代码分别保留可复现的构建命令、CSV 版本、设备型号、CANN 版本和原始日志路径。

## 兼容性分析

该算子为新增共享 API，不改变已有 aclblasStpmv 和其他 BLAS 算子的接口行为。兼容性风险和处理方式如下：

| 特性 | 风险 | 处理方式 |
| --- | --- | --- |
| 与 Stpmv 同族目录 | 文件名、构建目标冲突 | 使用 ctpmv 专用 Host、Kernel、Tiling 和测试文件名 |
| UPPER/LOWER packed 下标 | 公式偏移导致错误或越界 | 使用 n=1/2/3 手工下标表和 CBLAS golden 双重校验 |
| N/T/C x UNIT/NON_UNIT | 分支遗漏或误读对角 | 12 组正交组合测试，UNIT 对角使用 NaN 投毒 |
| 原地输出和多核 | 读写竞争 | xSnapshot 与 yContiguous 分离，主 Kernel 不直接覆写 x |
| 正负 incx | 逻辑/物理地址反向或越界 | 64 位位置计算，步长和物理空洞 canary 测试 |
| packed AP 和大 n | n(n+1)/2 或字节数溢出 | Host 侧 checked arithmetic，失败时不发起 Kernel |
| A2/A3 资源差异 | 固定 tile 或核数不适配 | 使用 arch22 能力边界和动态 useCoreNum、tile 参数 |
| 异步生命周期 | stream 未完成时释放输入 | 文档和测试明确同步点，workspace 由 handle 管理 |

## 交付文件和工程位置

| 模块 | 路径 | 内容 |
| --- | --- | --- |
| 公共 API | include/cann_ops_blas.h | aclblasCtpmv 声明 |
| Host | blas/tpmv/arch22/ctpmv_host.cpp | 参数校验、资源计算、workspace、launch |
| Kernel | blas/tpmv/arch22/ctpmv_kernel.cpp | packed 解码、复数计算、分块和写回 |
| Tiling | blas/tpmv/arch22/ctpmv_tiling_data.h | Host/Kernel 参数协议 |
| 测试 | test/tpmv/ctpmv/arch22/ | CSV、golden、GTest、性能采集和 README |
| 设计文档 | docs/design.md | 本设计文档 |

# 参考资料

1. 社区任务书：dev/task/aclblasCtpmv_Atlas800IA3_task_doc.md
2. 评测用例和测试指导：dev/task/test_cases/README.md
3. Netlib BLAS ctpmv：https://www.netlib.org/blas/ctpmv.f
4. cuBLAS cublasCtpmv 文档：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-tpmv
5. ops-blas 开源仓：https://gitcode.com/cann/ops-blas
6. 生态算子精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
7. Ascend C 算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
8. Ascend C API 文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html

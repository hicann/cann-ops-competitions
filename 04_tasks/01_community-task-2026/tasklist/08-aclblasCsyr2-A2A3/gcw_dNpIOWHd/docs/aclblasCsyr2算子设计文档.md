# 需求背景（required）

## 需求来源

本需求来自 CANN 2026 年 8 月社区算子任务（第 43 号，8月社区任务-aclblasCsyr2算子开发（A2/A3）），目标是在昇腾 ops-blas 开源仓为 aclblasCsyr2 补齐 Atlas A2/A3 的 arch22 实现，使公共 BLAS 接口在 A2/A3 上与 cuBLAS cublasCsyr2、Netlib ssyr2 的参数语义和异常行为一致。验收基线为 Atlas 800T A2（Ascend 910B3）、CANN 9.1.0。

## 背景介绍

### aclblasCsyr2算子实现优化

aclblasCsyr2 实现单精度复数对称秩-2 更新：

    A = alpha * x * y^T + alpha * y * x^T + A

其中 alpha 为单精度复数标量，x、y 为 n 元素复数向量，A 为 n×n 对称复数矩阵。关键语义：

- A 为**对称（symmetric）复数矩阵，满足 `A = A^T`（普通转置，不共轭）**：仅 `uplo` 指定的上三角或下三角被引用与更新，未引用部分不被读取、由对称性隐含。
- 与 Hermitian 族（her2 等）的区别：本算子**不含共轭运算**，对角元素虚部无特殊假定，按普通对称矩阵处理。
- 复数类型 `aclblasComplex` 以 ops-blas 仓 `include/cann_ops_blas_common.h` 定义为准（实部/虚部各 float32，即 complex64）。

本任务与仓内已有实数同族接口 `aclblasSsyr2`（`blas/syr2/arch22/` 已实现）同族归档，复用其 Host 直调、tiling、kernel launch 和错误处理框架，新增 complex64 复数运算与三角访问。由于两个复数 rank-1 外积之和本质是 O(n²) 的三角内存受限更新，首版采用 AIV 三角分块直接累加；n 最大 2048（性能 case），性能不达标再考虑拆分为实/虚 GEMM 走 Cube。

### aclblasCsyr2算子现状分析

| 来源 | 位置/版本 | 可复用内容 | 不能直接照搬的部分 |
|---|---|---|---|
| ops-blas 实数 syr2 | `blas/syr2/arch22/ssyr2_host.cpp`、`ssyr2_kernel.cpp`，`blas/syr2/arch35/` | Host 直调结构、参数校验、负步长遍历、tiling/分核、kernel launch、README 与测试框架 | 实数为 `float`，本算子为 `aclblasComplex` 复数运算与三角访问，需按 complex64 定制 |
| include 头文件 | `include/cann_ops_blas.h`（当前仅有第 339 行 `aclblasSsyr2`） | 声明风格、状态码与 handle 定义 | 当前**无** `aclblasCsyr2` 声明，本任务需新增（与 950PR 版共用同一声明，不另起平行 API） |
| Netlib BLAS | `ssyr2.f` | 权威参数语义、三角引用、负步长反向起点、quick return | 仅实数；复数 golden 需按 ssyr2 语义自实现（cblas/netlib 无复数 syr2） |
| NVIDIA cuBLAS | `cublas<t>syr2` | API 行为和 uplo/步长对照 | CUDA kernel 不开源，不能作为代码依赖 |

### aclblasCsyr2算子功能分析

算子功能：计算复数对称秩-2 更新 `A = alpha*x*y^T + alpha*y*x^T + A`，仅 uplo 指定三角被引用并原地覆写（对称不共轭）。

输入：handle、uplo、n、alpha、x、incx、y、incy、A、lda。

输出：A 原地更新（uplo 指定三角）。

支持数据类型：COMPLEX64（`aclblasComplex`，实/虚各 float32）。

支持形状：x、y 为逻辑长度 n 的向量（按 incx/incy 步长取元素）；A 为 n×n 列主序矩阵（lda 前导维）。

支持广播：不涉及广播。

# 需求分析（required）

## 需求描述

使用 Ascend C 为 aclblasCsyr2 实现 Atlas A2/A3 对应的 arch22 Host 与 Kernel，在 `include/cann_ops_blas.h` 新增 `aclblasCsyr2` 声明，实现复数对称秩-2 更新（仅更新 uplo 三角、对称不共轭），支持负步长与 no-op quick return；在 910B3 / CANN 9.1.0 上满足精度与三条性能门槛。

## 需求拆解

1. 在 `include/cann_ops_blas.h` 新增 `aclblasCsyr2` 声明（参数序列与 cuBLAS cublasCsyr2 一一对应）。
2. 支持 uplo=ACLBLAS_UPPER/LOWER，仅引用/更新指定三角。
3. 支持 COMPLEX64，实/虚各 float32。
4. 支持负步长 incx/incy（按 Netlib 反向起点遍历）。
5. n=0 或 alpha=(0,0) 为合法 quick return，不更新 A。
6. 参数校验：非法枚举返回 INVALID_ENUM，非法参数返回 INVALID_VALUE，空 handle 返回 HANDLE_IS_NULLPTR。
7. 与 cuBLAS cublasCsyr2 参数语义、异常行为对齐。
8. 精度：实/虚按 FLOAT32 判定，rtol=2^-10、atol=2^-16、matched_ratio>=0.99、max_abs_error<=1e-2 或 32 ULP。
9. 三条性能 case（n=512/1024/2048）平均单次耗时达标。
10. 提供 CSV/GTest 自测与按 Netlib ssyr2 语义自实现的复数 golden。

# 详细设计（required）

## 算子分析

### 数学公式

    A = alpha * x * y^T + alpha * y * x^T + A

对 uplo 三角内每个元素 (i,j)，逐元素计算：

    A(i,j) = A(i,j) + alpha * ( x(i)*y(j) + y(i)*x(j) )

其中所有量为复数（complex64）。`x(i)*y(j)` 与 `y(i)*x(j)` 为复数乘，`alpha * (...)` 为复数标量乘，最后为复数累加。对称性 `A = A^T`（不共轭）由实现保证：只写 uplo 三角，另一三角由对称性隐含，不读取、不传播共轭。

### 支持数据类型

仅支持 COMPLEX64（`aclblasComplex`）。alpha、x、y、A 均为 `aclblasComplex`，实部/虚部各 float32。接口类型固定，不存在 dtype 枚举和混合精度组合。

### 支持形状

- x、y：逻辑长度 n 的向量，内存按 `1 + (n-1)*|incx|` / `1 + (n-1)*|incy|` 描述，支持负步长。
- A：n×n 列主序矩阵，`lda >= max(1, n)`。
- 不涉及 broadcast、dynamic shape、ND 维度 ≥ 3 的高维运算。

## 算子实现

### 实现方案

复用仓内 `aclblasSsyr2`（arch22）的 Host 直调 + tiling + kernel launch 框架，新增 complex64 复数运算与对称三角访问。总体流程：Host 参数校验 → no-op 判定 → 分核 tiling → 在 handle 绑定 stream 上发射单 kernel。

#### 3.2.1 host侧设计

##### 1. 参数校验

1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `uplo` 不在 {ACLBLAS_UPPER(121), ACLBLAS_LOWER(122)} 返回 `ACLBLAS_STATUS_INVALID_ENUM`。
3. `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `alpha == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. `incx == 0` 或 `incy == 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. `lda < max(1, n)` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
7. `n > 0` 且 `alpha != (0,0)` 时，`x/y/A` 任一为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。

##### 2. no-op quick return

`n == 0` 或 `alpha == (0,0)` 时直接返回 `ACLBLAS_STATUS_SUCCESS`，不解引用 x/y/A，不发射 kernel，不更新 A。

##### 3. Tiling与分核

- 将 uplo 指定三角按行块（或连续三角块）分给 AIV 核，每核处理一段不重叠的三角区域；边界用 `min` 截断，避免多个核写同一输出。
- 负步长按 Netlib 参考实现换算反向起点：`kx = 1 - (n-1)*incx`、`ky = 1 - (n-1)*incy`（列主序下标），内核按步长正向累加访问。
- 本算子无 workspace、无动态 tiling data（可保留空 POD 或仅含 uplo/n/步长），在 handle 绑定 stream 上发射单 kernel。

##### 4. 头文件声明

在 `include/cann_ops_blas.h` 新增：

```cpp
aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* A, int lda);
```

#### 3.2.2 kernel侧设计

1. 每个 AIV 核处理分配给它的三角块；块内按列主序 `offset = i + j * lda` 访问 A。
2. 对三角内每个 (i,j)：先读 x[i]、y[j]、x[j]、y[i]，计算复数 `temp = x[i]*y[j] + y[i]*x[j]`，再 `A[i,j] += alpha * temp`。
3. 仅写 uplo 指定三角，另一三角不读不写；对角元素虚部无特殊处理（对称不共轭）。
4. 负步长场景从反向起点按步长取元素，结果与步长无关。
5. 若性能不达标，可将 rank-2 更新拆为实/虚四个 GEMM（`alpha*(x*y^T + y*x^T)` 按实虚展开）走 Cube 主体 + AIV 三角回填；首版以 AIV 三角分块保证正确性。

## 支持硬件

| 支持的芯片版本 | 支持情况 | 验证说明 |
|---|---|---|
| Atlas A2（910B/910B3） | 支持 | 性能验收设备为 Atlas 800T A2（910B3），CANN 9.1.0 |
| Atlas A3 | 支持 | 沿用 arch22 产品注册，单独编译和功能验证 |

## 算子约束限制

1. 仅支持 COMPLEX64，不支持其他 dtype 和混合精度。
2. 仅引用/更新 uplo 指定三角；对称性质（A=A^T，不共轭）由实现保证，对角元素虚部不做特殊处理。
3. `n >= 0`、`incx != 0`、`incy != 0`、`lda >= max(1, n)`；越界返回 INVALID_VALUE。
4. 非连续访问仅由 incx/incy/lda 描述，不要求超出步长/前导维语义的访问。
5. 不涉及 broadcast、dynamic shape、ND 维度 ≥ 3 运算。
6. A 原地覆写，不返回视图；不要求确定性计算。
7. `n=0` 或 `alpha=(0,0)` 为合法 quick return。
8. 依赖 `aclblasSetStream` 绑定 stream，Host 侧不做流同步，读回结果前调用方需自行同步。

风险与待确认项：

| 风险/冲突 | 处理方案 |
|---|---|
| cblas/netlib 无复数 syr2 | golden 由测试工程按 Netlib ssyr2 语义自实现复数参考版本，单标杆比对 |
| 对称与 Hermitian 语义混淆 | 明确 A=A^T（不共轭）；对角虚部无特殊处理；用共轭元素、纯实、纯虚用例验证 |
| 负步长遍历边界 | 按 Netlib 反向起点 `kx=1-(n-1)*incx` 换算，覆盖 incx/incy = -1/-2/-3 用例 |
| n=2048 三角更新性能 | 三角分块 + 行块分核；不达标再拆实/虚 GEMM 走 Cube |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | golden 按 Netlib ssyr2 语义自实现复数版本；输出 A 按 uplo 三角验证，实部/虚部分别按 FLOAT32：rtol=2^-10、atol=2^-16、matched_ratio>=0.99、max_abs_error<=1e-2 或 32 ULP | CANN 生态算子精度标准、任务书 |
| 性能标准 | 910B3 平均单次耗时：n=512 UPPER ≤4.82 us、n=1024 LOWER ≤6.22 us、n=2048 UPPER ≤12.19 us；warmup 后有效采样 >50 次 | 任务书 |
| 内存标准 | 无 workspace，无动态 tiling 缓冲 | 实现设计 |

功能矩阵：

| 类别 | 必测内容 |
|---|---|
| 基本功能 | UPPER/LOWER 全覆盖；n 为 0、1、小质数、2 的幂及 ±1、非对齐值、大规模 |
| 标量 | alpha 均匀/正态分布、特殊值 (0,0)/(1,0)/纯实/纯虚/大值；alpha=(0,0) no-op |
| 向量 | x/y 均匀/正态分布，实/虚独立采样，含 Inf/NaN 特殊值 |
| 步长 | incx/incy 覆盖 ±1/±2/±3，incx=0/incy=0 负向用例（期望 INVALID_VALUE） |
| 矩阵 | A 按对称约束填充（仅 uplo 三角有效，不共轭） |
| 前导维 | lda 取 max(1,n) 紧凑场景及多个 padding 场景；lda<max(1,n) 负向用例 |
| 负向 | 空 handle、非法枚举、负维度、空指针、非法 ld、零步长 |

性能计时只包围异步执行对应的 stream 区间，先 warmup 再采样 >50 次取平均；记录平均值、P50/P90 与各阶段时间。若未达标，按三角分块、复数乘加、内存带宽分段定位。

## 兼容性分析

1. API 兼容：在 `include/cann_ops_blas.h` 新增 `aclblasCsyr2` 声明，参数序列与 cuBLAS cublasCsyr2 一一对应，与其他产品线共用，不另起平行 API。
2. 产品兼容：新增 arch22 文件，arch35 与实数 aclblasSsyr2 不受影响。
3. ABI 兼容：不新增导出符号之外的公共结构体改动，不修改既有状态码和 handle 定义。
4. 构建兼容：按 ops-blas 既有 syr2 构建方式接入 `blas/syr2/arch22/` 与 `test/syr2/csyr2/arch22/`。
5. 行为兼容：与 cuBLAS cublasCsyr2、Netlib ssyr2 的参数语义、quick return、负步长和三角引用对齐。

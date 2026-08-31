# 需求背景

## 需求来源

本需求来源于“8月社区任务-aclblasCgeam算子开发”。目标是在 Ascend 950PR、CANN 9.1.0 环境中，基于 ops-blas 开源仓和 Ascend C Kernel 直调框架，实现单精度复数矩阵通用加法接口 `aclblasCgeam`，并完成设计、开发、精度验证和性能验证。

对标接口为 cuBLAS 扩展接口 `cublasCgeam`。Netlib BLAS 无对应 GEAM 接口，因此功能语义以任务书、ops-blas 公共接口及 cuBLAS GEAM 为准。

## 背景介绍

`aclblasCgeam` 完成两个单精度复数矩阵的缩放、可选转置和相加：

```text
C = alpha * op(A) + beta * op(B)
```

`op(A)`、`op(B)` 和 C 的逻辑形状均为 `m × n`。矩阵采用 Column-Major 存储，元素 `X[i,j]` 的物理地址为 `X[i + j * ldX]`。算子支持 `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 和 `ACLBLAS_OP_C`，其中 `ACLBLAS_OP_C` 对转置后的复数元素执行共轭。

ops-blas 公共头文件 `include/cann_ops_blas.h` 已有 `aclblasCgeam` 接口声明。本任务补齐 Ascend 950PR 的 Host 参数校验、Tiling、AIV Kernel 和测试代码，实现放在 `blas/geam/arch35/`，通过 handle 绑定的 stream 异步执行，不新增 Ascend 950PR 私有接口。

# 需求分析

## 需求描述

新增 Ascend 950PR 上的 `aclblasCgeam` 实现，接口保持为：

```cpp
aclblasStatus_t aclblasCgeam(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* beta, const aclblasComplex* B, int ldb,
    aclblasComplex* C, int ldc);
```

接口正确处理 N/T/C、Column-Major、leading dimension、复数缩放、共轭转置、标量短路和受限原地计算；`m==0` 或 `n==0` 时合法返回成功。算子在 Ascend 950PR 上满足任务书规定的精度和性能要求。

## 需求拆解

1. 保持公共接口、参数顺序、返回状态及 handle/stream 语义不变。
2. 支持 transa、transb 的 9 种组合，并正确处理 Column-Major 和 `lda/ldb/ldc`。
3. alpha 或 beta 为零时不读取对应矩阵；两者均为零时将 C 的逻辑区域清零。
4. `m==0` 或 `n==0` 时不读取 A、B、C，也不下发 Kernel。
5. 支持满足任务书约束的 `C==A` 和 `C==B`，拒绝不合法原地参数及其他部分重叠。
6. Host 完成参数校验、计算模式选择和多核 Tiling，Kernel 融合搬运、共轭、复数缩放、相加和写回。
7. 测试覆盖功能、异常、padding、Inf/NaN、零维、原地、精度和性能场景。

# 详细设计

## 算子分析

### 数学公式

对任意 `0 <= i < m`、`0 <= j < n`：

```text
C(i,j) = alpha * Read(A, transa, i, j)
       + beta  * Read(B, transb, i, j)
```

```text
Read(X, N, i, j) = X[i + j * ldX]
Read(X, T, i, j) = X[j + i * ldX]
Read(X, C, i, j) = conj(X[j + i * ldX])
```

设共轭处理后的 A、B 分别为 `ar+ai*i`、`br+bi*i`，alpha 为 `xr+xi*i`，beta 为 `yr+yi*i`，则：

```text
C.real = (xr*ar - xi*ai) + (yr*br - yi*bi)
C.imag = (xr*ai + xi*ar) + (yr*bi + yi*br)
```

当 alpha 或 beta 的实部、虚部均为 0 时，必须跳过对应输入，不能通过计算 `0 * Inf/NaN` 实现短路。

### 支持数据类型

| 输入/输出 | 数据类型 | 说明 |
| --- | --- | --- |
| A | `aclblasComplex` / COMPLEX64 | Device 输入，实部和虚部均为 FP32 |
| B | `aclblasComplex` / COMPLEX64 | Device 输入，实部和虚部均为 FP32 |
| C | `aclblasComplex` / COMPLEX64 | Device 输出，实部和虚部均为 FP32 |
| alpha | `aclblasComplex` / COMPLEX64 | Host 标量 |
| beta | `aclblasComplex` / COMPLEX64 | Host 标量 |

`aclblasComplex` 按 `[real, imag]` 交错存储，每个复数元素占 8 Byte。Kernel 使用 FP32 完成计算，不进行降精度转换。

### 支持形状

| 矩阵 | 操作 | 有效形状 | 物理存储形状 |
| --- | --- | --- | --- |
| A | N | `m × n` | `lda × n` |
| A | T/C | `n × m` | `lda × m` |
| B | N | `m × n` | `ldb × n` |
| B | T/C | `n × m` | `ldb × m` |
| C | - | `m × n` | `ldc × n` |

`m`、`n` 为运行时参数。每列只处理有效元素，输入 padding 不参与计算，C 的 padding 不被修改。

## 算子实现

### 实现方案

实现主要涉及以下文件：

| 文件 | 作用 |
| --- | --- |
| `blas/geam/arch35/cgeam_host.cpp` | 参数校验、Tiling 计算、Kernel 下发 |
| `blas/geam/arch35/cgeam_kernel.cpp` | 复数 GEAM AIV Kernel |
| `blas/geam/arch35/cgeam_kernel.h` | Kernel launcher 声明 |
| `blas/geam/arch35/cgeam_tiling_data.h` | Host/Kernel 共用 Tiling 数据 |
| `test/geam/geam_golden.h` | CPU 参考实现 |
| `test/geam/cgeam/arch35/` | Ascend 950PR GTest 与 CSV 用例 |

#### 3.2.1 host侧设计：

Host 侧按以下顺序处理参数：

1. handle 为空时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. transa、transb 不属于 N/T/C 时返回 `ACLBLAS_STATUS_INVALID_ENUM`。
3. m 或 n 为负数时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. alpha 或 beta 为空时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. 校验 leading dimension：

```text
ldaMin = (transa == N) ? max(1,m) : max(1,n)
ldbMin = (transb == N) ? max(1,m) : max(1,n)
ldcMin = max(1,m)
```

6. `m==0 || n==0` 时直接返回成功，此时不要求 A、B、C 非空。
7. 正尺寸时要求 C 非空；alpha 非零时要求 A 非空，beta 非零时要求 B 非空。
8. `C==A` 时要求 `transa==N && lda==ldc`；`C==B` 时要求 `transb==N && ldb==ldc`。不满足时返回 `ACLBLAS_STATUS_INVALID_VALUE`。

根据标量选择四种计算模式：

| 模式 | 条件 | Kernel 行为 |
| --- | --- | --- |
| GENERAL | alpha、beta 均非零 | 读取 A、B 并计算两项 |
| A_ONLY | alpha 非零、beta 为零 | 只读取和计算 A |
| B_ONLY | alpha 为零、beta 非零 | 只读取和计算 B |
| ZERO_FILL | alpha、beta 均为零 | 不读取 A、B，生成复数正零 |

Tiling 优先沿输出列分核；当输出列数少于 AIV 核数时，再沿输出行 Tile 分核。Host 在完成零维快速返回后才计算以下参数，因此 `n>0`、`colBlocks>=1`：

```text
totalMTiles = ceil_div(m, tileM)
colBlocks   = min(n, aivCoreNum)
mBlocks     = min(totalMTiles, aivCoreNum / colBlocks)
blockDim    = colBlocks * mBlocks
```

每个 LocalTensor 保存 `tileM` 个 FP32 分量。GENERAL 模式按 12 个等效 FP32 Tile Buffer 估算 UB：A 的实部/虚部输入队列双缓冲占 4 份，C 的实部/虚部输出队列双缓冲占 4 份，B 的实部/虚部占 2 份，复数乘法临时空间占 2 份。`tileM` 按下式确定：

```text
rawTileM = floor((ubSize - ubReserve) / (12 * sizeof(float)))
tileM    = round_down(min(rawTileM, 4095), 8)
```

`tileM` 是 FP32 元素数；取 8 的倍数保证每个 LocalTensor 的有效字节数按 32 Byte 对齐，4095 是 arch35 扩展搬运接口的 `blockCount` 上限。短路模式释放不需要的输入或临时 Buffer，但采用相同的 Tile 上限。

主要 Tiling 字段包括：

| 字段 | 含义 |
| --- | --- |
| `m/n/lda/ldb/ldc` | 矩阵形状与 leading dimension，单位为复数元素 |
| `alphaR/alphaI/betaR/betaI` | Host 标量分量 |
| `alphaIsZero/betaIsZero` | 输入短路标志 |
| `opA/opB` | N/T/C 模式 |
| `tileM` | 单 Tile 的输出行数，也是单个分量 Buffer 的 FP32 元素数 |
| `colBlocks/perCoreN/remainder` | 列方向分核参数 |
| `mBlocks/perCoreMTile/mTileRemainder` | 行 Tile 分核参数 |

Host 将标量值和 Tiling 数据随 Kernel 参数下发，在 `handle->stream` 上启动一个 AIV Kernel。各 block 处理互不重叠的 C 区域，不需要核间归并或额外 workspace；接口不主动同步 stream。

#### 3.2.2 kernel侧设计：

Kernel 采用 `Init` 和 `Process` 两阶段，`Process` 内按 `CopyIn -> Compute -> CopyOut` 执行。`Init` 完成 GlobalTensor 绑定、Tiling 字段缓存、当前 block 行列范围计算及 TQue/TBuf 初始化。

Kernel 将 GM 中的交错复数视为 FP32 数组。以下 `baseComplex` 均以复数元素为单位，访问 GM 时转换为 `baseFloat = 2 * baseComplex`。对于当前输出列 `col`、行 Tile 起点 `rowStart`，实部和虚部分别使用一次扩展搬运；每个 block 的长度为 4 Byte，`blockCount=curM`：

| 访问对象 | `baseComplex` | 相邻 block 的间隔 |
| --- | --- | --- |
| N 模式输入 | `col * ld + rowStart` | `4 Byte` |
| T/C 模式输入 | `col + rowStart * ld` | `(2 * ld - 1) * 4 Byte` |
| C 输出 | `col * ldc + rowStart` | 目标间隔 `4 Byte` |

N 模式下，相邻复数元素的同一分量之间隔一个 FP32 分量；T/C 模式下，相邻输出行对应的输入相隔 `ld` 个复数元素。上述步长直接生成连续的 UB 实部、虚部数组，写回时再分别写入 C 的交错位置，不生成完整转置矩阵。所有复数地址乘加及乘 2 转换均使用 `uint64_t`，避免 32 位溢出。

`ACLBLAS_OP_C` 在搬入后对相应输入的虚部乘 `-1.0f`。随后使用 `Muls`、`Add`、`Sub` 完成复数缩放和相加。GENERAL 计算两项，A_ONLY 和 B_ONLY 只执行对应项，ZERO_FILL 使用 `Duplicate(0.0f)` 生成实部和虚部正零。

A 的实部、虚部输入 TQue 和 C 的实部、虚部输出 TQue 使用双缓冲，B 分量和复数乘法临时量使用 TBuf。TQue 的 `EnQue/DeQue` 及必要的事件同步保证 MTE2、Vector、MTE3 之间的数据依赖；尾 Tile 使用实际 `curM` 搬运和计算，不访问越界元素。

合法原地场景中，当前 Tile 的有效输入全部搬入 UB 后才写 C。N 模式下每个输出只依赖同一逻辑位置，且各 block 写入范围互不重叠，因此 `C==A`、`C==B` 可正确执行；转置原地存在跨 Tile 读写覆盖风险，由 Host 拒绝。

## 支持硬件

| 支持的芯片版本 | 目录 | 说明 |
| --- | --- | --- |
| Ascend 950PR | arch35 | 支持 |

## 算子约束限制

1. 仅支持 `aclblasComplex` / COMPLEX64。
2. 仅支持 Column-Major，不支持 Row-Major。
3. transa、transb 仅支持 N/T/C。
4. 不支持广播；非连续访问仅通过 `lda/ldb/ldc` 表达。
5. alpha、beta 必须位于 Host 内存。
6. `m==0` 或 `n==0` 为合法 no-op，负维度非法。
7. 支持满足约束的 `C==A`、`C==B`，不支持其他部分重叠。
8. 接口异步执行，读取 C 前需要同步 handle 绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 实部、虚部分别按 FP32 校验：`rtol=2^-10`、`atol=2^-16`、`matched_ratio>=0.99`，且 `max_abs_error_limit=1e-2` 或 `32*ULP` | 任务书、生态算子开源精度标准 |
| 性能标准 | Ascend 950PR 上先 warmup，再有效采样大于 50 次取平均；平均耗时不超过任务书标杆 | 任务书 |

有限值逐元素通过条件为 `|actual-golden| <= atol + rtol*|golden|`。用例同时满足 `matched_ratio >= 0.99` 和 `max_abs_error <= max_abs_error_limit` 时，精度验证通过。golden 分量为 NaN 时要求对应输出分量也是 NaN；golden 分量为正、负 Inf 时要求输出为同符号 Inf，Inf/NaN 不参与普通有限值误差统计。

CPU golden 使用 `test/geam/geam_golden.h` 中的 `std::complex<float>` 逐元素实现。精度和功能测试包括：

- transa×transb 的 9 种组合，以及 T 与 C 结果不同的复数输入；
- `m/n` 为 0、1、小质数、2 的幂及相邻值、非对齐值、非方阵和大尺寸；
- 最小合法 leading dimension 和多组 padding，并以哨兵值检查 C padding 未被修改；
- alpha、beta 的 `(0,0)`、`(1,0)`、纯虚数和大值，以及 A、B 的 Inf/NaN；
- 合法和非法原地、空指针、非法枚举、负维度及非法 leading dimension；
- A、B、C、alpha、beta 的均匀分布 `[-5,5]` 和正态分布各占 50%，正态分布参数为 `μ∈[-5,5]`、`σ∈[0.1,2]`，复数实部、虚部独立采样。

任务书性能用例如下，alpha、beta 均为 `(1,0)`，leading dimension 取最小合法值：

| case | m | n | transa | transb | 最大平均耗时 |
| --- | ---: | ---: | --- | --- | ---: |
| 1 | 1024 | 1024 | N | N | 1.99 us |
| 2 | 2048 | 2048 | N | N | 10.18 us |
| 3 | 2048 | 2048 | T | C | 10.45 us |

每个性能用例先执行 10 次预热，再执行 5 轮测试，每轮使用 ACL Event 在 handle 绑定的 stream 上记录 100 次连续调用的总耗时，除以 100 得到该轮平均单次耗时，共计 500 次有效调用。计时结束后同步 stream 并校验输出，报告 5 轮总平均值以及轮平均值的最小值、最大值；总平均值不超过上表限制时通过。

## 兼容性分析

本需求复用 `include/cann_ops_blas.h` 中已有接口，只新增 `blas/geam/arch35/` 的 Cgeam 后端，不修改函数签名、公共类型、状态码、handle 或 stream 语义，因此不产生 API/ABI 兼容性影响。Cgeam 调用既有 GEAM 公共参数校验能力，但不改变 Sgeam 的接口和执行路径；提交前同时回归 Sgeam 测试，确认共享代码行为未发生变化。

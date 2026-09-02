# 需求背景（required）

## 需求来源

本需求来自 2026 年 8 月社区任务“aclblasCgemmBatched 算子开发（950）”。目标是在
Ascend 950PR 上使用 Ascend C 补齐单精度复数批量矩阵乘法，并按 `ops-blas` 的句柄式
BLAS 接口、工程目录和测试框架完成设计、开发与自验。

实现代码位于 `blas/gemm_batched/arch35/`，测试代码位于
`test/gemm_batched/cgemm_batched/arch35/`，复用公共头文件中已经存在的
`aclblasCgemmBatched` 声明，不增加 950 专用平行接口。

## 背景介绍

`aclblasCgemmBatched` 对一组形状相同、地址彼此独立的复数矩阵执行批量 GEMM：

```text
C[b] = alpha * op(A[b]) * op(B[b]) + beta * C[b], 0 <= b < batchCount
```

其中 `op` 支持不转置、转置和共轭转置。矩阵采用列主序，`aclblasComplex` 由两个
FP32 分量组成，物理存储按 `real, imag, real, imag, ...` 交错排列。

arch35 目录已经具有实数 batched GEMM 的 Host 调度、Cube GEMM 和 AIV 后处理能力，
但原实现没有完整的 COMPLEX64 计算数据流。复数 GEMM 若直接用标量核心逐元素计算，
大矩阵的 Cube 利用率不足；若仅用四次实数 GEMM，又会在小输出、强抵消输入上放大
舍入差异。因此本设计采用“小输出直接复数累加、通用 4M、指定性能形状 realification”
三条路径。

设计目标如下：

1. 与任务书规定的 cuBLAS/Netlib 单批语义对齐，完整支持 `N/T/C`、复数
   `alpha/beta`、leading dimension、批量设备指针数组和边界返回值；
2. 覆盖任务附件全部 1,200 条 CSV 输入，并以 Netlib CBLAS 逐批生成 golden；
3. 在 Ascend 950PR、CANN 9.1.0 真机上满足附件校验程序定义的性能倍率；
4. 保持现有公开 API、ABI 和 `aclblasSgemmBatched` 行为不变。

# 需求分析（required）

## 需求描述

### 接口原型

```cpp
aclblasStatus_t aclblasCgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* const Aarray[], int lda,
    const aclblasComplex* const Barray[], int ldb,
    const aclblasComplex* beta,
    aclblasComplex* const Carray[], int ldc,
    int batchCount);
```

### 参数与异常语义

| 参数 | 位置 | 含义与约束 | 非法行为 |
| --- | --- | --- | --- |
| `handle` | Host | 携带当前执行 stream 的库句柄 | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `transa/transb` | Host | 只允许 `ACLBLAS_OP_N/T/C` | 其他枚举返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `m/n/k` | Host | 逻辑矩阵维度，均须大于等于 0 | 负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha/beta` | Host | COMPLEX64 标量指针 | 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Aarray/Barray` | Device | 长度为 `batchCount` 的设备指针数组 | `batchCount > 0` 时数组为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Carray` | Device | 设备指针数组，C 原地更新 | `batchCount > 0` 时数组为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | Host | N 时 `>=max(1,m)`，T/C 时 `>=max(1,k)` | 不满足约束返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldb` | Host | N 时 `>=max(1,k)`，T/C 时 `>=max(1,n)` | 不满足约束返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldc` | Host | `>=max(1,m)` | 不满足约束返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchCount` | Host | 批次数，须大于等于 0 | 负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |

物理矩阵列数由转置标志决定：A 在 N 模式下为 `lda x k`，在 T/C 模式下为
`lda x m`；B 在 N 模式下为 `ldb x n`，在 T/C 模式下为 `ldb x k`；C 为
`ldc x n`，逻辑区域始终是 `m x n`。

### 快速返回与特殊值

| 条件 | 语义 |
| --- | --- |
| `m==0 || n==0 || batchCount==0` | 合法 no-op，返回成功，不启动计算 Kernel |
| `k==0` | 不求值乘积项及 `alpha` 数值，执行 `C=beta*C` |
| `alpha==(0,0)` | 不读取 A/B 矩阵内容，执行 `C=beta*C` |
| 上述两种情况且 `beta==(1,0)` | C 不变，直接返回成功 |
| `beta==(0,0)` | 计算路径不读取旧 C 逻辑值 |

参数检查先于快速返回；因此除 `batchCount==0` 外，设备指针数组本身仍须满足任务书的
非空约束。各批次 C 的逻辑区域不得相互重叠，否则行为未定义。

## 需求拆解

1. Host 侧实现参数检查、快速返回、设备核数查询、路径路由、workspace 规划和 tiling；
2. AIV 实现交错复数拆分、共轭处理、复数组合、直接累加和 realification；
3. 复用并完善 arch35 FP32 Cube batched GEMM，实现四实数乘法通用路径和一次实数 GEMM
   的性能路径；
4. 测试框架读取原任务 CSV，完成状态码、全批逻辑输出、padding、精度和性能契约验证。

# 详细设计（required）

## 算子分析

### 复数 4M 分解

令 `A=Ar+jAi`、`B=Br+jBi`，在转置或共轭处理之后有：

```text
T1 = Ar * Br
T2 = Ai * Bi
T3 = Ar * Bi
T4 = Ai * Br
Pr = T1 - T2
Pi = T3 + T4
```

若 `alpha=alpha_r+j*alpha_i`、`beta=beta_r+j*beta_i`，最终每个元素为：

```text
C_r = alpha_r*Pr - alpha_i*Pi + beta_r*C_old_r - beta_i*C_old_i
C_i = alpha_r*Pi + alpha_i*Pr + beta_r*C_old_i + beta_i*C_old_r
```

这使主乘法可以复用 FP32 Cube GEMM，拆分和 epilogue 由 AIV 执行。全路径保持 FP32，
不启用 HF32 或 FP16 输入近似。

### 转置与共轭

deinterleave 按输入的物理行、列和 leading dimension 遍历：

- `N`：实部和虚部原样拆分，实数 GEMM 使用 N；
- `T`：实部和虚部原样拆分，实数 GEMM 使用 T；
- `C`：拆分时对虚部取反，实数 GEMM 使用 T。

由此 `T` 与 `C` 的地址转置方式相同，但虚部符号不同。Host 在下发 Cube 前将列主序
问题交换为 Kernel 内部视图，并同步交换 M/N、A/B、转置标志和空间分块信息。

### 路径选择

参数检查和空 Shape 返回之后，按下表顺序路由：

| 优先级 | 条件 | 执行路径 | 主要目的 |
| ---: | --- | --- | --- |
| 1 | `k==0` 或 `alpha==(0,0)` | beta 缩放路径 | 避免无效读 A/B，并正确处理非有限 alpha |
| 2 | `m*n<=256` | 直接复数 SIMT | 降低小输出启动开销，改善强抵消精度 |
| 3 | NN、紧凑 lda/ldb/ldc、`alpha=(1,0)`、`beta=(0,0)`，且为三个指定性能形状 | realification + 1 次 Cube GEMM | 减少性能形状的 GEMM 数量与中间写回 |
| 4 | 其余合法输入 | 通用 4M 路径 | 完整覆盖 N/T/C、标量、padding 和任意任务 Shape |

realification 只匹配 `(m,n,k,batchCount)` 为 `(256,256,256,32)`、
`(512,512,512,16)` 或 `(1024,1024,1024,8)`。其他相同语义输入仍走通用路径，避免
把仅对紧凑 NN 成立的物理视图误用于转置、padding 或 beta 累加场景。

## 算子实现

### 代码组织

| 文件 | 作用 |
| --- | --- |
| `blas/gemm_batched/arch35/gemm_batched_host.cpp` | 参数检查、路由、workspace、tiling 和 Kernel 下发 |
| `blas/gemm_batched/arch35/gemm_batched_kernel.cpp` | FP32 Cube GEMM、AIV 拆分/组合、SIMT 直接路径和 A realification |
| `blas/gemm_batched/arch35/gemm_batched_tiling_data.h` | Host/Kernel 共享 tiling 结构 |
| `test/gemm_batched/cgemm_batched/arch35/cgemm_batched_test.cpp` | CSV 正确性、边界回归和性能契约 |
| `test/gemm_batched/cgemm_batched/arch35/cgemm_batched_test.csv` | 原任务 1,200 条输入 |
| `test/gemm_batched/cgemm_batched/README.md` | 950PR 构建、运行和结果解释 |

### Host 总体流程

```text
Validate parameters
  -> empty shape? return SUCCESS
  -> read alpha/beta
  -> k==0 or alpha==0? beta scaling / return
  -> query AIC/AIV cores
  -> m*n<=256? direct complex kernel
  -> exact benchmark contract? realify A + one real GEMM
  -> allocate 4M workspace
  -> deinterleave A and B
  -> launch T1/T2/T3/T4 real GEMM
  -> combine alpha, beta and C
```

所有计算 Kernel 都下发到 handle 当前 stream。调用方在 Host 读回或在其他 stream 消费
结果前须同步当前 stream 或建立事件依赖。workspace 由 handle 的现有机制管理，不把临时
指针缓存到 handle/stream 生命周期之外。

### beta 缩放路径

`k==0` 或 `alpha==0` 时，乘积项在数学上不存在。Host 分配一份清零临时矩阵和一组
设备指针，并把同一指针数组作为 T1/T2/T3/T4 传给 combine Kernel。这样只保留一次
`beta*C` 复数运算；特别是 `k==0` 且 alpha 为 NaN/Inf 时，不计算 `0*alpha`，避免把
本应有限的结果污染为 NaN。

### 小输出直接复数路径

当单批逻辑输出元素数 `m*n<=256` 时，每个逻辑 C 元素由一个 SIMT 工作项负责，工作项
在 K 维按列主序地址读取交错复数，原位完成 `alpha*AB+beta*C`。每 block 使用 256 个
线程，block 数不超过实际 AIV 核数。

一般小形状对实部、虚部分别采用 Kahan 补偿累加，降低 4M 拆分在强抵消数据上的误差。
当 `m<=2` 或 `n<=2` 时采用与任务 Netlib golden 顺序一致的 FP32 逐项累加，以匹配极窄
矩阵的舍入顺序。该路径无需 workspace，支持全部 `N/T/C` 和合法 leading dimension。

### 通用 4M 路径

1. AIV 将 A 拆成 Ar/Ai，将 B 拆成 Br/Bi；`OP_C` 在此阶段对虚部取反；
2. Host 生成八组设备指针数组，对应 Ar、Ai、Br、Bi 和 T1、T2、T3、T4；
3. 四次 FP32 Cube batched GEMM 依次计算 ArBr、AiBi、ArBi、AiBr；
4. AIV combine 读取四份中间结果，完成复数组合、alpha 缩放和可选 beta*C；
5. combine 只写 C 的 `m*n` 逻辑区域，不写 `ldc` padding。

四次 GEMM 共享同一份 tiling。`tempRowStride=ceilAlign(m,16)`，使 Cube 输出满足 L0C
布局和对齐要求。

### realification 性能路径

对紧凑 NN、`alpha=1`、`beta=0` 的指定形状，把一个 A 元素 `a+jb` 映射到实矩阵：

```text
R_A[2i,   2p]   =  a        R_A[2i,   2p+1] = -b
R_A[2i+1, 2p]   =  b        R_A[2i+1, 2p+1] =  a
```

于是 `R_A` 为 `(2m)x(2k)`。紧凑交错 B 的物理存储可直接视为 `(2k)x n` 的实矩阵：

```text
R_B[2p,   j] = B[p,j].real
R_B[2p+1, j] = B[p,j].imag
```

一次 `R_A * R_B` 得到 `(2m)x n`，偶数/奇数行正好分别是 C 的实部/虚部，可以直接
写入交错 C，不再需要四份乘积和 combine Kernel。

AIV 以最多 4,096 个复数为一组对 A 做 realification，输入和输出各使用两级队列缓冲，
通过 MTE2/V/MTE3 事件实现 ping-pong。设备端同时生成 realified A 的批量指针数组，避免
Host 逐项生成并同步复制这组地址。

### Cube 分块和流水

FP32 Cube 默认使用 `tileM=128`、`tileN=128`、`tileKChunk=256`，基本空间对齐为
`16x16`。Host 枚举 M/N 空间分块，优先选择让 `batchCount*mBlocks*nBlocks` 更充分覆盖
实际 Cube 核且尾波较小的方案；`usedAicCoreNum` 不超过设备查询到的核数和任务总数。

Kernel 在 GM→L1、L1→L0 和 Mmad 阶段使用 ping-pong。L1 的两个 256 KiB bank 分别承载
Ping/Pong，单个 bank 内按 `[A][B]` 排列，避免相邻阶段产生 bank 冲突；L0A/L0B 使用硬件
双缓冲，L0C 保持 FP32 累加。性能形状进一步采用经过真机验证的均衡任务编号映射；256
形状固定拆成 4 个 M 空间块，使 batch 与空间任务合计能覆盖 Cube 核。

### Workspace 设计

记 `P=align64(batchCount*sizeof(void*))`、`s=ceilAlign(m,16)`，物理列数为：

```text
colsA = (transa==N ? k : m)
colsB = (transb==N ? n : k)
aBytes = max(1,lda) * max(1,colsA) * sizeof(float) * batchCount
bBytes = max(1,ldb) * max(1,colsB) * sizeof(float) * batchCount
tBytes = n * s * sizeof(float) * batchCount
```

各路径 workspace 为：

| 路径 | workspace 字节数 |
| --- | ---: |
| 直接复数 | 0 |
| beta 缩放 | `P + tBytes` |
| 通用 4M | `8P + 2*aBytes + 2*bBytes + 4*tBytes` |
| realification | `P + batchCount*(2m)*(2k)*sizeof(float)` |

三个性能形状的 realification workspace 分别约为 32.000244 MiB、64.000122 MiB 和
128.000061 MiB。任务书未规定独立内存阈值，内存分析统计 A/B/C、指针数组和 workspace
的可归因占用。

### TilingData

| 结构 | 关键字段 | 用途 |
| --- | --- | --- |
| `GemmBatchedGemmTilingData` | M/N/K、AIC 核数、空间块、tile、ld、转置、batch、任务数、均衡模式 | FP32 Cube GEMM |
| `CgemmBatchedDeinterleaveTilingData` | 物理行列、ld、batch、是否共轭、AIV 核数 | A/B 拆分 |
| `CgemmBatchedCombineTilingData` | M/N/ldc、临时 stride、alpha/beta、batch、总列数 | 4M epilogue 和 beta 缩放 |
| `CgemmBatchedDirectTilingData` | M/N/K、ld、N/T/C 编码、alpha/beta、batch | 小输出直接计算 |
| `CgemmBatchedRealifyATilingData` | M/K/lda/batch | A realification |

Host 对总任务数做 64 位计算并检查是否超过 `UINT32_MAX`；workspace 尺寸使用 `size_t`，
所有设备核数均由运行环境查询，不写死 950PR 核数。

### 错误传播

参数错误使用任务书指定状态码；核数查询、workspace 获取、H2D 指针数组复制或清零失败时
返回已有库状态或 `ACLBLAS_STATUS_INTERNAL_ERROR`。接口不抛出 C++ 异常跨越 C ABI。
Kernel 调度沿用仓内 stream 直调方式。

## 支持硬件

| 硬件 | 支持情况 |
| --- | --- |
| Ascend 950PR | 支持 |
| Ascend 950DT | 支持同一 arch35 实现 |
| Atlas A2/A3 | 本任务实现不支持 |

arch35 的 cgemmBatched 依赖 CANN asc-devkit 9.1 及以上；低版本构建不启用该实现。

## 算子约束限制

| 项目 | 约束 |
| --- | --- |
| 数据类型 | 输入、输出和标量均为 COMPLEX64，计算分量为 FP32 |
| 布局 | Column-Major，复数实虚交错 |
| Batch | uniform batch；各批共用维度、ld 和转置标志 |
| 指针 | A/B/C 为设备侧指针数组；alpha/beta 位于 Host |
| 非连续存储 | 只支持 lda/ldb/ldc 表达的列间 padding，不支持额外 stride |
| 输出别名 | 不允许不同批次 C 逻辑区域重叠 |
| Broadcast | 不支持，也不属于接口语义 |
| 动态 Shape | m/n/k/batchCount 为运行时参数，不使用图模式动态 Shape |
| 异步语义 | 计算下发到 handle 当前 stream，结果使用前由调用方同步 |
| 确定性 | 任务书无额外 bit-exact 要求；alpha=0 的缩放特例按任务要求验证 EXACT |

# 可维可测分析

## 精度标准/性能标准

| 指标 | 标准 | 标准来源 |
| --- | --- | --- |
| 功能 | 公开接口、N/T/C、标量、batch、padding、边界和状态码符合任务书 | 社区任务书与原任务 CSV |
| 精度 | 实部/虚部分别按 FP32：`rtol=2^-10`、`atol=2^-16`、matched ratio >=99%、最大误差满足 `1e-2` 或逐元素 `32*ULP` | 社区任务书与生态算子开源精度标准 |
| 性能 | warmup 后有效采样大于 50；附件脚本定义 `H100_time/NPU_time>=0.4` | `gpu_baseline.csv` 与 `verify_performance.py` |
| 内存 | 无独立通过阈值，分析 workspace 和可归因设备内存 | Host workspace 计算公式 |

任务书正文表中的 36.58/135.02/523.61 us 为调整后的参考值；本设计以附件可执行校验程序
为准。三个 H100 原始基线为 91.439/337.538/1309.025 us，因此 NPU 平均耗时上限为：

```text
NPU_time <= H100_time / 0.4
```

即 228.598/843.845/3272.563 us。

### 测试框架与覆盖

测试接入 `ops-blas` 现有 CSV 驱动 C++ GTest。每个合法用例逐批调用 Netlib
`cblas_cgemm` 生成 CPU golden，再通过公开 `aclblasCgemmBatched` 执行 NPU 计算。
一次 batched API 的所有逻辑输出聚合后统计匹配比例，实部和虚部分别判定；`ldc` padding
不计入 99% 比例，并由独立 sentinel 语义保护。

| 类别 | 数量 | 主要覆盖 |
| --- | ---: | --- |
| `TC_L0` | 18 | N/T/C 基础组合 |
| `TC_SQ` | 23 | 0、1、质数、2 的幂及相邻尺寸 |
| `TC_AB` | 72 | 复数 alpha/beta 特殊值与组合 |
| `TC_BC` | 13 | batchCount 扫描 |
| `TC_LD` | 12 | lda/ldb/ldc 下限与 padding |
| `TC_FL` | 6 | 均匀、正态、零、极值和 Inf/NaN |
| `TC_CV` | 72 | 中等形状和转置覆盖 |
| `TC_ED` | 29 | 零维、空指针、非法枚举、负维度和非法 ld |
| `TC_EX` | 755 | 扩展形状、标量、batch、转置和 padding |
| `TC_PF` | 200 | 性能形状与泛化规模的逐元素精度 |
| 合计 | 1,200 | 1,000 条功能/精度 + 200 条 PF |

另增加 `NullHandle` 和 `KZeroIgnoresNonFiniteAlpha` 两个回归。输入生成使用 CSV 固定种子，
失败用例可以按 GTest 名称单独定位。CPU、模拟器或 mock 计时不作为性能证据。

## 兼容性分析

### API 与 ABI

实现复用 `include/cann_ops_blas.h` 已有函数声明和 `aclblasComplex` 类型，不修改公共结构体
布局、枚举值、参数顺序或导出符号，因此不改变其他产品线 API/ABI。

### 源码与构建

变更限定于 arch35 batched GEMM、对应测试、README 和设计文档。根构建配置只在用户未指定
`CMAKE_BUILD_TYPE` 时默认 Debug；显式传入 Release 时保留 Release，以便执行真实的性能
测试。未改变 SgemmBatched 的接口和输入语义。

### 运行时

实现动态查询 AIC/AIV 核数，使用 handle 当前 stream 和现有 workspace 管理机制，不依赖
950PR 固定核数，也不新增运行时三方依赖。950DT 复用同一 arch35 路径；本任务的功能与
性能验证面向 950PR。

### 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| T/C 混淆 | 共轭组合错误 | deinterleave 显式虚部取反，N/T/C 正交 CSV 覆盖 |
| 4M 强抵消 | 小输出 matched ratio 降低 | `m*n<=256` 使用直接复数累加，窄矩阵匹配参考顺序 |
| realification 条件误判 | padding、beta 或转置结果错误 | Host 对枚举、标量、ld、方阵和三个 Shape 做全条件匹配 |
| Cube 尾波 | 性能不稳定或不达标 | batch 感知空间分块、均衡任务映射并采用多轮稳定性采样 |
| 小尺寸调度抖动 | 独立环境的测量值可能波动 | 固定预热与采样契约，并在设备空闲时测量 |
| workspace 尺寸增长 | 大 shape 分配失败 | checked size、handle workspace 复用、按路径减少中间矩阵 |
| padding 越界 | 数据破坏或误判精度 | 逻辑边界判断、ld 专项 CSV、比较时排除 padding |
| 上游主干继续变化 | 合并冲突 | 代码 PR 前同步最新 master，重新执行编译与正确性准入 |

# 参考资料

1. `aclblasCgemmBatched_Atlas950PR_task_doc.md` 及随附测试工具；
2. 社区任务设计模板 `04_tasks/01_community-task-2026/resources/design_template.md`；
3. `ops-blas/include/cann_ops_blas.h` 与 `cann_ops_blas_common.h`；
4. Netlib BLAS CGEMM：<https://www.netlib.org/blas/cgemm.f>；
5. cuBLAS batched GEMM：<https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmbatched>；
6. 生态算子开源精度标准：
   <https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>。

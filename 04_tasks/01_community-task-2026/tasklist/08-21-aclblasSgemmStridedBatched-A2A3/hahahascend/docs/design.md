# aclblasSgemmStridedBatched 算子设计文档

任务：8 月社区任务第 21 项，`aclblasSgemmStridedBatched` 算子开发（A2/A3）

目标开源仓：<https://gitcode.com/cann/ops-blas>

适配硬件：Atlas A2/A3 系列产品（arch22 / DAV_2201）
开发语言：C++ / Ascend C

# 需求背景（required）

## 需求来源

本设计来源于 CANN 2026 年 8 月社区任务 `aclblasSgemmStridedBatched` 算子开发（A2/A3）。目标是在
`ops-blas` 开源仓中补充 Atlas A2/A3 产品的 FP32 跨步批量矩阵乘能力，并复用仓内已有的公共接口：

```cpp
aclblasStatus_t aclblasSgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    int64_t strideA, const float* B, int ldb, int64_t strideB,
    const float* beta, float* C, int ldc, int64_t strideC, int batchCount);
```

接口以 handle 绑定的 stream 异步提交任务，`alpha`、`beta` 为 Host 指针，A、B、C 为 Device 指针。
实现代码位于 `blas/gemm_strided_batched/arch22/`，不新增产品私有平行接口。

## 背景介绍

Strided Batched GEMM 是 BLAS Level-3 批处理接口。与传入指针数组的 Batched GEMM 不同，该接口只传入
batch 0 的首地址，并通过固定 stride 定位后续矩阵，因此适用于 batch 内矩阵规格一致、存储间隔固定的场景。
它可减少 Host 侧指针数组的构造和拷贝，在小矩阵批处理、分块线性代数和深度学习计算中较常见。

任务语义与 `cublasSgemmStridedBatched` 的核心参数和列主序约定对齐，并包含一项 `ops-blas` 扩展：
`strideA=0` 或 `strideB=0` 表示所有 batch 广播复用同一输入矩阵。

### 现状分析

`ops-blas` 已声明公共 `aclblasSgemmStridedBatched` API，并有其他产品线实现，但 Atlas A2/A3 对应的 arch22
目录缺少实现。本设计在不改变 API 和其他产品代码的前提下补充 arch22 Host、Kernel、构建注册和测试用例。

本算子的关键工程问题如下：

1. 用户矩阵采用列主序，而 Ascend C Matmul 内部按行主序组织，需要无额外全矩阵转置地完成布局映射。
2. FP32 GEMM 必须关闭 HF32，满足 FLOAT32 精度门槛。
3. `alpha`、`beta`、padding leading dimension、转置组合和零维语义要求通用后处理路径。
4. 大矩阵中间结果不能假设 workspace 可容纳完整 C，需要有界复用 workspace。
5. 性能 case 是 NN 方阵批处理，需要降低 Host launch 开销并提高 Cube 搬运与计算流水效率。

# 需求分析（required）

## 需求描述

对每个 batch 独立执行：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
A_i = A + i * strideA
B_i = B + i * strideB
C_i = C + i * strideC
i = 0, 1, ..., batchCount - 1
```

其中 stride 单位为 FP32 元素个数而非字节，矩阵均为列主序。`op(X)` 支持 N、T、C；由于数据类型为
实数 FP32，C 与 T 等价。

## 需求拆解

| 编号 | 子需求 | 设计响应 |
| --- | --- | --- |
| F1 | 公共 API 对齐 | 沿用 `include/cann_ops_blas.h` 已有签名和状态码 |
| F2 | FP32 与 N/T/C | A/B/C 均为 FP32，N/T/C 组合全部支持 |
| F3 | Column-Major | 在 Kernel 内交换 A/B，将问题映射为计算 `C^T` |
| F4 | Strided Batch | Host/Kernel 根据 stride 计算每个 batch 的 GM 地址 |
| F5 | A/B 广播 | `strideA=0`、`strideB=0` 直接复用 batch 0 地址 |
| F6 | alpha/beta | 直接写 C 快路径和 workspace + AIV combine 通用路径 |
| F7 | BLAS 边界 | 支持零维 no-op、`k=0`、`alpha=0` 和 beta-only |
| F8 | Leading dimension | 校验并正确处理 lda/ldb/ldc padding |
| F9 | 大 shape | 按 C 列 strip-mine，中间缓冲区在 strip 和 batch 间复用 |
| F10 | 性能 shape | NN 紧凑方阵采用手写 Cube 路径，其他 shape 使用通用 MatmulImpl |
| F11 | strict FP32 | 两条 AIC 路径均显式关闭 HF32 |
| F12 | 单卡执行 | 一个 API 调用只使用 handle 所在设备，不进行跨卡或跨节点拆分 |

## 范围边界

本设计支持：

- FP32 输入、输出与累加；
- N/T/C 的全部正交组合；
- 运行时 m/n/k/batchCount；
- lda/ldb/ldc padding；
- 非负 stride，包含 A/B 的 stride 0 整矩阵广播；
- C 原地更新以及 handle stream 异步执行。

本设计不支持或不定义：

- 非 FP32 数据类型；
- batch 内逐元素广播；
- 负 stride；
- C 的 batch 子矩阵重叠。特别是 `strideC=0 && batchCount>1` 时语义未定义；
- 跨设备、跨卡或跨节点协同计算；
- 调用方声明的 shape、leading dimension 和 stride 导致的越界显存访问。

# 详细设计（required）

## 算子分析

### 数学公式

对 `i` 号 batch：

```text
D_i(r, c) = sum(t=0..k-1) op(A_i)(r, t) * op(B_i)(t, c)
C_i(r, c) = alpha * D_i(r, c) + beta * C_i(r, c)
```

逻辑 shape 为：

```text
op(A_i): m x k
op(B_i): k x n
C_i:     m x n
```

列主序地址为 `matrix[col * ld + row]`。当 `trans=N` 时读取原矩阵；当 `trans=T/C` 时交换逻辑行列。

### 支持数据类型

| 数据 | 类型 | 存储位置 | 说明 |
| --- | --- | --- | --- |
| alpha、beta | FP32 | Host | 全 batch 共用 |
| A、B | FP32 | GM | 只读输入 |
| C | FP32 | GM | 输入/输出，原地更新 |
| Cube accumulator | FP32 | L0C | HF32 关闭 |
| workspace temp | FP32 | GM | 仅通用 alpha/beta 路径使用 |

### 支持形状

| 参数 | 物理 shape | 约束 |
| --- | --- | --- |
| A，transA=N | `lda x k` | `lda >= max(1,m)` |
| A，transA=T/C | `lda x m` | `lda >= max(1,k)` |
| B，transB=N | `ldb x n` | `ldb >= max(1,k)` |
| B，transB=T/C | `ldb x k` | `ldb >= max(1,n)` |
| C | `ldc x n` | `ldc >= max(1,m)` |

`m,n,k,batchCount >= 0`。维度由运行时参数给出，不依赖静态 shape；非 tile 对齐的尾块由 Matmul layout
或带 padding 的数据搬运处理。

## 算子实现

### 总体架构

```text
Public API
   |
   +-- 参数校验 / quick return
   |
   +-- k=0 or alpha=0 ----------> beta-only AIV / memset / no-op
   |
   +-- alpha=1, beta=0,
   |   ldc aligned --------------> AIC direct-write C
   |                                 |-- NN square hot path: manual Cube
   |                                 `-- general path: MatmulImpl
   |
   `-- other alpha/beta ---------> strip-wise MatmulImpl -> workspace temp
                                                    |
                                                    `-> AIV: alpha*temp+beta*C
```

所有 Kernel 按 handle 绑定的同一 stream 异步提交。通用 workspace 在同一 stream 上按“GEMM 写入、AIV
读取、下一个 strip 覆盖”的顺序复用，无需 Host 同步，也不会把中间结果暴露给调用方。

### 列主序映射

对于列主序矩阵，内存可等价解释为其转置矩阵的行主序存储，因此：

```text
C^T = op(B)^T * op(A)^T
```

实现将原 B 作为 Matmul 左输入、原 A 作为右输入，内部有效维度为：

```text
mEff = n
nEff = m
kEff = k
ldLeft  = ldb
ldRight = lda
transLeft  = (transB != N)
transRight = (transA != N)
```

该映射避免在 GM 中显式转置 A、B、C。对实数 FP32，`ACLBLAS_OP_C` 与 `ACLBLAS_OP_T` 进入同一布局分支。

### Host 侧设计

#### 参数校验和边界处理

Host 侧按以下顺序检查：

1. handle 非空；
2. transA/transB 属于 N/T/C；
3. m/n/k/batchCount 非负；
4. lda/ldb/ldc 满足 BLAS leading dimension 下界；
5. strideA/strideB/strideC 非负；
6. alpha/beta Host 指针非空；
7. 非空计算中按 `k`、`alpha`、`beta` 语义检查 A/B/C 指针。

边界分支如下：

| 条件 | 行为 |
| --- | --- |
| `m=0 || n=0 || batchCount=0` | 合法 no-op，返回成功，不启动 Kernel |
| `k=0 || alpha=0`，`beta=1` | C 不变，不启动 Kernel |
| `k=0 || alpha=0`，`beta=0` | 只清零 C 的逻辑 `m x n` 区域，不改 padding |
| `k=0 || alpha=0`，其他 beta | AIV 按列执行 `C=beta*C` |

#### Tiling 与分核

通用 MatmulImpl 的静态 shape 参数采用 `128 x 512 x 4096`，基础 Cube 粒度为 `128 x 128 x 64`。
Host 侧根据 `mEff/nEff` 和可用 AIC 数将核心数分解到 M/N 两个方向，每核负责一个大区域；区域内部以
M/N tile 和 K chunk 迭代。K chunk 初始取 256，并根据 A/B 双缓冲所需 L1 空间收缩，保证不超过
arch22 的 L1 预算。M/N 方向使用蛇形遍历，降低相邻任务间的数据重载。

AIV combine 和 beta-scale 按原 C 的列分核。列内的 m 个元素在列主序下连续，每次处理最多 2048 个
FP32 元素；尾段使用 padding copy，写回时只覆盖有效元素。

#### 执行路径选择

1. **beta-only 路径**：跳过 A/B 和 Cube 计算，仅处理 C。
2. **直接写 C 路径**：当 `alpha=1 && beta=0 && ldc%8=0` 时，Fixpipe 直接写入用户 C，避免 workspace
   和 AIV 后处理。
3. **NN 方阵热点路径**：在直接写路径基础上，若 m=n=k、N/N、尺寸为 128 的倍数且范围为
   `[128,2048]`，lda/ldb/ldc 紧凑且 A/B/C stride 均为单矩阵元素数，则使用手写 Cube 双缓冲路径。
4. **通用路径**：其他转置、矩形、padding 或标量组合均使用 strict-FP32 MatmulImpl。需要标量后处理时
   先写 workspace，再由 AIV combine。

直接写路径中，如果 C 的各 batch 不重叠，batch 任务融合到一次 Kernel launch，Kernel 通过 task index
解析 batch 和 tile；`strideA/strideB=0` 自然映射到相同输入基址。通用后处理路径因复用同一 workspace，
按 batch 和 strip 在同一 stream 上顺序提交。

#### Workspace 设计

通用路径需要保存未乘 alpha 的 GEMM 临时结果。为避免 `m*n*sizeof(float)` 随 shape 无界增长，Host 按
C 的列方向分条：

```text
tempRowStride = max(align_up(m, 8), align_up(n, 8))
bytesPerColumn = tempRowStride * sizeof(float)
nStrip = min(n, effectiveWorkspaceBytes / bytesPerColumn)
```

每个 strip 执行：

```text
GEMM(B_strip, A) -> temp
combine(temp, C_strip) -> C_strip
```

workspace 来自 handle 的有效 workspace，默认 4 MiB 也可工作；若连一列临时结果都放不下，返回执行失败，
调用方可通过 handle 注入更大的 workspace。workspace 在不同 strip 和 batch 之间复用，因此峰值用量有界。

### Kernel 侧设计

#### 通用 MatmulImpl Kernel

Kernel 根据 transLeft/transRight 选择 NDExt/DNExt layout，将 GM 中的 B、A 解释为交换后的左右操作数。
每个 AIC 以 grid-stride 方式遍历 `(batch,mTile,nTile)` 任务，通过 `strideLeft/strideRight/strideOut`
定位 batch。在 Kernel 初始化阶段调用：

```cpp
matmul.SetHF32(false, 0);
```

保证 FP32 输入和 FP32 accumulator 不降为 HF32。MatmulImpl 完成 GM 搬运、MMAD 累加和 Fixpipe 输出，并
依据有效 M/N/K 处理边界 tile。

#### NN 方阵手写 Cube Kernel

热点路径采用 `128 x 128 x 64` tile：

- A、B 的 L1 与 L0A/L0B 使用双缓冲；
- K 维以 64 为步长执行 `Mmad`，首个 K tile 初始化 L0C，后续 tile 累加；
- M tile 采用蛇形 N 遍历；
- Fixpipe 将 FP32 L0C 结果直接写回 C；
- 通过 MTE1/MTE2/M/FIX 硬事件保证缓冲区读写依赖；
- 初始化阶段调用 `SetHF32Mode(HF32Mode::DISABLE)`，保持 strict FP32。

该路径只针对满足完整 tile 和紧凑内存约束的 shape，不影响其他合法输入的通用语义。

#### AIV Combine Kernel

AIV 按列读取 temp 和原 C，在 UB 中执行：

```text
out = alpha * temp
if beta != 0:
    out = out + beta * C
```

随后原地写回 C。`beta=0` 时不读取旧 C。搬入、计算、搬出使用队列双缓冲，尾块只写有效 FP32 元素。

#### AIV Beta-scale Kernel

退化场景仅读取 C，执行 `Muls(C,beta)` 后原地写回。`beta=0` 优先由异步 memset 清零逻辑区，
`beta=1` 直接 no-op，从而避免不必要的 AIV launch。

## 支持硬件

| 支持的芯片版本 | 架构目录 | 涉及勾选 |
| --- | --- | --- |
| Atlas A2 系列产品（含 Atlas 800I/T A2） | arch22 / DAV_2201 | √ |
| Atlas A3 系列产品（含 Atlas 800I A3） | arch22 / DAV_2201 | √ |

构建与验收环境按任务书使用 CANN 9.1.0。

## 算子约束限制

1. 仅支持 FP32。
2. A、B、C 均按列主序解释，stride 的单位是元素个数。
3. stride 必须非负；仅 A/B 的 stride 0 定义为广播。
4. `strideC=0 && batchCount>1` 或其他造成 C batch 重叠的布局语义未定义。
5. 调用方负责保证基址、shape、leading dimension 和 stride 对应的显存范围有效。
6. C 为原地输出，不产生新的 Tensor 或视图。
7. 算子在一个设备上执行，不通过多卡拆 batch 提升单次调用性能。
8. Host API 异步返回，调用方读回结果前须同步 handle 所绑定的 stream。

# 可维可测分析

## 可维护性分析

- arch22 代码与 arch35 产品实现分目录隔离，公共 API 不变。
- Host 参数校验、tiling/路由、Kernel 实现和测试用例分层组织。
- 专用 NN 方阵路径有严格且可读的路由条件，所有不满足条件的合法输入回退到通用路径。
- tiling data 只传递运行所需的维度、布局、分核、tile、batch 和 stride 信息，不依赖测试 case ID、环境变量
  或输入数据内容。
- 日志输出选择的路径和 tiling，便于定位参数、workspace 和性能问题。

## 可测试性分析

测试使用 `test/gemm_strided_batched/` 的 CSV + GTest 框架，golden 逐 batch 调用 CBLAS SGEMM 生成。
测试维度包括：

| 类别 | 覆盖内容 |
| --- | --- |
| 基础功能 | 小尺寸、非方阵、质数和非 tile 对齐 shape |
| 转置 | transA/transB 的 N/T/C 正交组合 |
| 布局 | 紧凑与 padding lda/ldb/ldc、紧凑与非紧凑 stride |
| 广播 | strideA=0、strideB=0 及二者同时为 0 |
| 标量 | alpha/beta 的 0、1、-1 和一般值 |
| 边界 | m/n/batchCount=0，k=0，alpha=0，batchCount 大至 1024 |
| 异常 | 空 handle/标量/数据指针、非法枚举、负维度、非法 leading dimension/stride |
| 特殊值 | 0、正负交替、Inf、NaN |
| 性能 | 1024、2048、4096 NN 方阵批处理 |

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FP32：rtol=`2^-10`，atol=`2^-16`，matched ratio >= 0.99，max abs error <= `1e-2` 或 `32*ULP` | 生态算子开源精度标准、任务书 |
| 性能标准 | warmup 后有效采样大于 50 次取平均；平均耗时不高于任务书给出的三个门槛 | 社区任务书 |

任务书列出的性能门槛为：

| m=n=k | batchCount | 门槛（us） |
| --- | --- | --- |
| 1024 | 16 | 1858.46 |
| 2048 | 8 | 3697.51 |
| 4096 | 4 | 5781.46 |

## 兼容性分析

- 公共函数签名、枚举、状态码和 handle/stream 使用方式保持不变。
- 新增能力仅注册到 Atlas A2/A3 的 arch22 构建路径，不改变 arch35 实现。
- N/T/C、column-major、leading dimension 和边界语义与任务书及 Netlib SGEMM 保持一致。
- `strideA/strideB=0` 是任务书明确要求的 `ops-blas` 扩展；其他行为不扩展 cuBLAS 语义。
- 新增实现依赖 CANN 9.1 的 Ascend C 编译与 Matmul 能力；验收和发布时按任务书版本构建。

## 参考资料

1. [CANN 社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)
2. [ops-blas 开源仓](https://gitcode.com/cann/ops-blas)
3. [cuBLAS cublasSgemmStridedBatched](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched)
4. [Netlib SGEMM](https://www.netlib.org/blas/sgemm.f)
5. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)

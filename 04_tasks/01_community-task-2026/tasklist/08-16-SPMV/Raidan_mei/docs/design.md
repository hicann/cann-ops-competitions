# SpMV 算子设计文档（Ascend 950PR）

# 需求背景（required）

## 需求来源

本需求来源于 CANN 2026 年 8 月社区任务“SpMV 算子开发”，任务标识为
`6151bcefa6b24bd1ba2f018e1be072f3`。任务要求在 `cann/ops-sparse` 中补齐
Ascend 950PR（DAV_3510、arch35）上的 CSR SpMV 能力，并提供算子源码、接口说明、
设计文档、功能与精度测试以及性能测试结果。

## 背景介绍

### SpMV 算子实现目标

SpMV（Sparse Matrix-Vector Multiplication）用于计算 CSR 稀疏矩阵与稠密向量的乘积，
是图计算、推荐系统、稀疏神经网络和科学计算中的基础算子。本任务沿用
`aclsparseSpMV` 三阶段接口，在不改变公开接口语义的前提下新增 arch35 Host 与 Kernel
实现，使算子能够在 Ascend 950PR 上运行。

实现和接口文件位于：

- Host 与公开接口：`sparse/spmv/arch35/spmv_host.cpp`、`sparse/spmv/arch35/spmv.h`
- Tiling 数据：`sparse/spmv/arch35/spmv_tiling_data.h`
- Kernel：`sparse/spmv/arch35/kernels/`
- 测试：`test/spmv/arch35/spmv_test.cpp`

### SpMV 算子实现现状分析

现有 `ops-sparse` 已提供 SpMV 接口框架及 arch22 兼容实现。本任务新增的 arch35
实现保留三阶段 API：

| 接口 | 当前实现 |
| --- | --- |
| `aclsparseSpMVGetBufferSize` | 校验基础参数；转置且半精度输出时返回 FLOAT32 workspace 大小 |
| `aclsparseSpMVPreprocess` | 调用 BufferSize 校验，当前为幂等 no-op |
| `aclsparseSpMV` | 校验描述符与类型，准备 Tiling，按类型和转置模式分发 arch35 Kernel |

CSR 仅存储行偏移、列索引和非零值，访存量与非零元数量 `nnz` 相关。其性能主要受
CSR 行长度分布、间接读取 `x` 的局部性、空行比例、核间负载均衡和 Kernel 启动开销影响。
小矩阵更容易受启动开销限制，大规模且行分布不均匀的矩阵则容易出现长尾核。

### SpMV 算子功能分析

算子计算公式为：

```text
y = alpha * op(A) * x + beta * y
```

其中 `A` 为形状 `[M, N]` 的 CSR 稀疏矩阵，`x`、`y` 为一维稠密向量，
`op(A)` 支持非转置和转置。非转置输出长度为 `M`，转置输出长度为 `N`。

当前 arch35 实现支持：

1. CSR 非转置与转置计算；
2. `alpha`、`beta` 缩放及 `y` 原位累加；
3. FLOAT32、FLOAT16、BFLOAT16、INT8 及提交包内已实例化的混合精度组合；
4. 空行、单非零元行、超长 CSR 行、极高稀疏度和大规模矩阵；
5. 默认算法 `ACL_SPARSE_SPMV_ALG_DEFAULT`。

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上实现与 `aclsparseSpMV` 接口约定一致的 CSR SpMV。实现应正确处理
非转置/转置、不同数据类型、`alpha`/`beta`、边界矩阵和随机稀疏矩阵；功能与精度结果
应通过 CPU CSR 参考实现校验，性能应达到任务书规定的 0.5 倍标杆水平。

## 需求拆解

1. 实现 `GetBufferSize`、`Preprocess`、执行接口三阶段调用流程；
2. 校验 handle、stream、描述符、CSR 格式、索引类型与基址、维度、算法和数据类型；
3. 支持非转置 `A * x` 和转置 `A^T * x`；
4. 支持 FLOAT32、FLOAT16、BFLOAT16、INT8 及混合精度计算；
5. 根据 AIV Core 数和矩阵行数进行多核切分，对大规模非转置 FLOAT 任务增加并行 block；
6. 对 FLOAT32 非转置提供 SIMT 快路径，其余组合使用通用 Ascend C 模板 Kernel；
7. 覆盖功能、精度、异常参数、边界和任务书性能用例；
8. 保持公开接口与既有 arch22 路径兼容，不引入额外 workspace。

# 详细设计（required）

## 算子分析

### 数学公式

对非转置模式，输出第 `i` 个元素为：

```text
y[i] = alpha * sum(A[j] * x[col_idx[j]], j=row_ptr[i]..row_ptr[i+1)-1)
       + beta * y[i]
```

对转置模式，每个非零元 `(i, col_idx[j])` 对输出产生贡献：

```text
y[col_idx[j]] += alpha * A[j] * x[i]
```

累加贡献前先执行 `y = beta * y`。

### 输入输出与接口参数

| 参数 | 输入/输出 | 类型 | 说明 |
| --- | --- | --- | --- |
| `handle` | 输入 | `aclsparseHandle_t` | ops-sparse 上下文，必须已设置非空 stream |
| `opA` | 输入 | `aclsparseOperation_t` | 支持 NON_TRANSPOSE、TRANSPOSE |
| `alpha` | 输入 | `const void *` | 计算缩放系数，类型与 `computeType` 一致；不可为空 |
| `matA` | 输入 | 稀疏矩阵描述符 | CSR 矩阵，包含 rowPtr、colInd、values |
| `vecX` | 输入 | 稠密向量描述符 | 输入向量，值类型与矩阵 values 一致 |
| `beta` | 输入 | `const void *` | 原输出缩放系数，类型与 `computeType` 一致；不可为空 |
| `vecY` | 输入/输出 | 稠密向量描述符 | 原位读写输出向量 |
| `computeType` | 输入 | `aclDataType` | 支持 `ACL_FLOAT`、`ACL_INT32` |
| `alg` | 输入 | `aclsparseSpMVAlg_t` | 仅支持 DEFAULT |
| `externalBuffer` | 输入 | `void *` | 非转置可为空；转置半精度输出时为 FLOAT32 workspace |

### 支持数据类型

提交包生成了以下 Kernel 实例：

| A values / x | computeType | y | Kernel 计算类型 |
| --- | --- | --- | --- |
| FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
| FLOAT16 | FLOAT32 | FLOAT16 | FLOAT32，写回时转 FLOAT16 |
| FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |
| BFLOAT16 | FLOAT32 | BFLOAT16 | FLOAT32，写回时转 BFLOAT16 |
| BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |
| INT8 | FLOAT32 | FLOAT32 | 输入转 FLOAT32 后计算 |
| INT8 | INT32 | INT32 | INT32 标量乘加 |

### 支持形状

矩阵 `A` 为二维 CSR，逻辑形状为 `[M, N]`，`x`、`y` 为一维向量：

| 模式 | x 最小长度 | y 最小长度 | 输出有效长度 |
| --- | ---: | ---: | ---: |
| 非转置 | `N` | `M` | `M` |
| 转置 | `M` | `N` | `N` |

行偏移和列索引均使用 INT32，因此矩阵规模、`nnz` 和所有索引必须处于 INT32 可表示范围。

## 算子实现

### 实现方案

#### Host 侧设计

##### 1. 三阶段接口

`aclsparseSpMVGetBufferSize` 检查必需描述符、操作类型和算法类型，并将 workspace
大小设置为 0。`aclsparseSpMVPreprocess` 复用该校验逻辑，不生成持久化数据。
`aclsparseSpMV` 完成完整校验、Tiling 和 Kernel 启动。执行热路径不把 rowPtr 回读到 Host，
也不执行 stream 同步，避免小、中矩阵延迟被 D2H 和同步开销主导。

##### 2. 参数与数据检查

执行接口按以下顺序检查：

1. `handle`、`matA`、`vecX`、`vecY` 非空；
2. `alg` 为 DEFAULT，`opA` 为 NON_TRANSPOSE 或 TRANSPOSE；
3. `computeType` 为 FLOAT 或 INT32；
4. 矩阵格式为 CSR，rowPtr/colInd 均为 32 位索引，索引基址为 zero；
5. handle 中 stream 非空，平台 AIV Core 数大于 0；
6. CSR 三个 Device 指针以及 x/y Device 指针非空；
7. x/y 长度满足转置模式对应的最小长度；
8. 矩阵 values 与 x 的值类型相同；INT32 compute 要求 values 和 y 均为 INT32。

CSR 的 rowPtr 单调性、末项与 `nnz` 一致性以及列索引范围应在描述符创建或预处理阶段保证。
Kernel 对非单调 rowPtr 做了防下溢保护，但这不替代调用方提供合法 CSR 数据的责任。

##### 3. 分核策略

默认 `blockDim` 取平台 `GetCoreNumAiv()`。非转置、FLOAT 计算且 `M > 32768` 时，
block 数提升为 AIV Core 数的 8 倍，并限制不超过 `M`，以减少大规模超稀疏矩阵的长尾。

通用 Kernel 将连续行均分到 block：前 `M % blockDim` 个 block 多处理一行。每个 block
只绑定自身的 rowPtr 段以及对应 values/colInd 段。FLOAT32 非转置 SIMT 路径则按 block
划分连续行区间，每个 block 启动 256 个 SIMT 线程，线程以 128 为步长处理区间内的行；每
两个相邻 lane 协作完成一行的稀疏点积。

##### 4. Tiling 数据与 TilingKey

Host 下发矩阵行列数以及稠密向量、CSR 数组元素 stride：

```cpp
struct SpmvTilingData {
    uint32_t totalRowsNum;
    uint32_t totalColNum;
    uint64_t xStride;
    uint64_t yStride;
    uint64_t rowPtrStride;
    uint64_t colIndStride;
    uint64_t valuesStride;
};
```

当前实现不使用数值型 TilingKey。Host 分发器根据 `computeType`、values 类型、y 类型和
`trans` 选择已实例化 Kernel；FLOAT32/FLOAT32/FLOAT32 非转置单独选择 SIMT 快路径。

##### 5. workspace 与同步

非转置路径不申请外部 workspace；转置且 FLOAT16/BFLOAT16 输出时使用
`GetBufferSize` 返回的 FLOAT32 workspace，`externalBuffer` 必须非空。Kernel 启动在 handle 的 stream 上
异步执行；除测试代码在取回结果时同步外，执行接口本身不主动同步 stream。

#### Kernel 侧设计

##### 1. FLOAT32 非转置 SIMT 快路径

每个 SIMT 线程负责若干 CSR 行，按 rowPtr 遍历该行非零元。内层循环四路展开，减少循环
控制开销；空行、单非零元行、`alpha == 1` 和 `beta == 0` 使用快速分支。每行只由一个
线程写回，不存在输出写冲突。

##### 2. 通用非转置 Ascend C 路径

通用 `SpmvKernel<CompT, ValT, OutT>` 采用 `Init` 和 `Process` 两阶段。每一行依次执行：

1. `CopyIn`：搬入 colInd 和 values，按需将 ValT 转为 CompT，并读取原 y；
2. `Compute`：按列索引间接读取 x，计算逐元素乘积并执行 ReduceSum；
3. 计算 `alpha * sum + beta * y`；
4. `CopyOut`：按需转换为 OutT 后写回 y。

CSR 的 block-local GM 视图按 `logical_index * stride` 计算基址和访问偏移，rowPtr、colInd、
values 在 block 切分后仍以原始 CSR 数组为基准，保证 stride>1 的非连续存储不会重复偏移。

每个 block 在 Device 侧扫描自身 rowPtr 段得到最大行长 `tileLength`，并据此初始化 UB
队列。`BUFFER_NUM` 当前为 1。全空行时使用最小 `tileLength=8`，避免零长度 UB 缓冲区；
单个空行直接计算 `beta * y`。FP32 非转置路径将 tile 限制为 4096 个元素，超长行切换到
GM 标量分块遍历，避免单行长度导致 UB 溢出。

##### 3. 转置路径

转置 `SpmvKernelTrans` 分为两个阶段：

1. 各 block 均分互不重叠的 y 列区间，先执行 `y = beta * y`；
2. 各 block 处理自己的 CSR 行，为每个非零元计算 `alpha * A[i,j] * x[i]`，再使用
   AtomicAdd 累加到 `y[j]`。

两个阶段之间使用 `SyncAll`，防止 beta 缩放与原子累加交叉。原子加解决了多个输入行向
同一输出列写入的冲突。转置 Host 当前固定使用一个 AIV block，CSR 行按固定顺序遍历，
因此本实现具备确定性执行顺序；INT32 路径要求精确一致。

##### 4. 混合精度策略

FLOAT16、BFLOAT16 和 INT8 values 在 FLOAT compute 路径中转换为 FLOAT32，再完成乘法、
归约及 alpha/beta 计算；y 为低精度类型时，在写回前按 `CAST_ROUND` 转换。这样避免低精度
长行归约造成过大的累计误差，同时保留低精度输入/输出能力。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Ascend 950PR（DAV_3510 / arch35） | √ | 已完成构建、功能、精度和性能实测 |
| Ascend 950DT（DAV_3510 / arch35） | √ | 与 950PR 共用 arch35 实现；提交包未单独实测 |
| Atlas A2 / Atlas A3 | 不涉及本次新增 | 沿用仓库既有 arch22 能力，本次不修改 |

## 算子约束限制

1. 仅支持 CSR，不支持 COO、CSC 等其他稀疏格式；
2. rowPtr 和 colInd 仅支持 `ACL_SPARSE_INDEX_32I`，索引基址仅支持 zero；
3. 仅支持 NON_TRANSPOSE 和 TRANSPOSE，不支持共轭转置；
4. 仅支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`；
5. values 与 x 必须同类型，支持组合以“支持数据类型”表为准；
6. x、y 必须满足操作模式对应的长度要求；
7. 输入必须是结构合法的 CSR，列索引必须位于 `[0, N)`；
8. 当前无额外 workspace，Preprocess 不缓存行统计或转置结构；
9. 转置路径固定单 block 顺序累加，保证确定性执行。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能标准 | 非转置/转置、alpha/beta、边界与随机矩阵结果正确 | 任务书、CPU CSR 参考实现 |
| FLOAT32 精度 | `rtol=2^-10`、`atol=2^-16`，匹配率不低于 99%，最大绝对误差不超过 1e-2 | 提交包测试代码 |
| FLOAT16 精度 | `rtol=2^-9`、`atol=2^-9`，匹配率不低于 99%，最大绝对误差不超过 1e-1 | 提交包测试代码 |
| BFLOAT16 精度 | `rtol=2^-6`、`atol=2^-6`，匹配率不低于 99%，最大绝对误差不超过 1.0 | 提交包测试代码 |
| INT32 精度 | 逐元素精确匹配，匹配率 100% | 提交包测试代码 |
| 性能标准 | 达到 0.5 倍标杆水平，即实测耗时不高于标杆耗时的 2 倍 | 社区任务书 |

### 功能与精度验证

验证环境为 Ascend 950PR（DAV_3510）和 CANN 9.0.0-beta.2，构建命令为：

```bash
source /home/developer/Ascend/cann-9.0.0-beta.2/bin/setenv.bash
bash build.sh --ops=spmv --soc=ascend950 --run
```

测试使用 CPU CSR 实现生成 golden，覆盖 FLOAT32、FLOAT16、BFLOAT16、INT32 与混合精度、
非转置/转置、alpha/beta、原位累加、空行、单行、长宽矩阵、随机矩阵以及 50%～99.9%
稀疏度。共执行任务书官方 200 个 case，结果为：

```text
OFFICIAL CASE SUMMARY: total=200 passed=200 failed=0
```

### 性能验证

设置 `SPMV_PERF_ONLY=1`，每个用例执行 20 次 warmup 和 100 次计时：

| M×N | 稀疏度 | 类型 | 标杆耗时 | 实测平均耗时 | 实测/标杆 | 结论 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 128×128 | 95% | FLOAT32 | 43.9 us | 3.80084 us | 0.087 | 通过 |
| 1024×1024 | 99% | FLOAT32 | 46.3 us | 4.94892 us | 0.107 | 通过 |
| 2048×4096 | 97.5% | FLOAT32 | 45.4 us | 13.0115 us | 0.287 | 通过 |
| 160220×68750 | 99.9% | FLOAT32 | 193.2 us | 340.509 us | 1.762 | 通过 |

四组用例均满足“实测耗时不高于标杆 2 倍”的验收口径。

## 兼容性分析

本任务新增稠密向量 stride 兼容 API，同时继续使用 `aclsparseSpMVGetBufferSize`、
`aclsparseSpMVPreprocess`、`aclsparseSpMV` 及既有描述符。非转置 workspace 大小保持为 0，
转置半精度输出按 `Y.numel * sizeof(float)` 返回 workspace，调用方需按返回值传入
`externalBuffer`。新增代码位于 arch35 路径，类型实例和 Kernel 分发
封装在 arch35 实现内，不改变既有 arch22 行为。

后续如增加新的索引类型、稀疏格式、算法枚举或转置预处理结构，应同步扩展接口校验、
Tiling 数据、Kernel 实例、README 和测试矩阵，避免 Host 支持范围与 Kernel 实例不一致。

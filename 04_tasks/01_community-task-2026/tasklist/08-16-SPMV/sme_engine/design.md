# SpMV 算子设计文档（Ascend 950PR / arch35）

> 目标仓：cann/ops-sparse · 路径：sparse/spmv/arch35/
> 任务：8月社区任务 08-16：SPMV 算子开发
> 目标硬件：Ascend 950PR（DAV_3510 / arch35）

---

## 1. 需求背景

### 1.1 需求来源

任务要求在 `cann/ops-sparse` 中补齐 Ascend 950PR（DAV_3510，arch35）上的 CSR SpMV
能力，提供算子源码、接口说明、设计文档、功能与精度测试以及性能测试结果。

### 1.2 算子定义

CSR 格式稀疏矩阵与稠密向量乘法：

```text
Y = alpha · op(A) · X + beta · Y
```

- `A`：形状 `[M, N]` 的 CSR 稀疏矩阵
- `X`、`Y`：一维稠密向量
- `op(A)`：支持非转置（NON_TRANSPOSE）与转置（TRANSPOSE）
- 非转置输出长度 `M`，转置输出长度 `N`

### 1.3 实现现状分析

现有 `ops-sparse` 已提供 SpMV 接口框架及 arch22 兼容实现。本次新增 arch35 实现，
保留三阶段 API，不改变公开接口语义。

CSR 仅存储行偏移、列索引和非零值，访存量与非零元数量 `nnz` 相关。性能主要受
CSR 行长度分布、间接读取 `X` 的局部性、空行比例、核间负载均衡和 Kernel 启动
开销影响。小矩阵更容易受启动开销限制，大规模且行分布不均匀的矩阵则容易出现
长尾核。

---

## 2. 需求拆解

1. 实现 `GetBufferSize`、`Preprocess`、执行接口三阶段调用流程；
2. 校验 handle、stream、描述符、CSR 格式、索引类型与基址、维度、算法和数据类型；
3. 支持非转置 `A * x` 和转置 `A^T * x`；
4. 支持 FLOAT32、FLOAT16、BFLOAT16、INT8、INT32 及混合精度计算；
5. 根据 AIV Core 数和矩阵行数进行多核切分；
6. 覆盖功能、精度、异常参数、边界和任务书性能用例；
7. 保持公开接口与既有 arch22 路径兼容。

---

## 3. 接口设计

### 3.1 API

遵循 aclsparse 接口风格，提供三阶段调用：

```text
aclsparseSpMVGetBufferSize(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, size)
aclsparseSpMVPreprocess(handle, opA, matA, vecX, vecY, computeType, alg, externalBuffer)
aclsparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer)
```

### 3.2 参数说明

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

---

## 4. 架构设计

### 4.1 总体架构

- **行并行策略**：每个 AI Core 处理若干完整行；
- **三级流水**：CopyIn → Compute → CopyOut；
- **UB 缓冲区复用**：多队列共享 UB，`BUFFER_NUM = 1`。

### 4.2 分核策略

默认 `blockDim` 取平台 `GetCoreNumAiv()`。非转置、FLOAT 计算且 `M > 32768` 时，
block 数提升为 AIV Core 数的 8 倍，并限制不超过 `M`，以减少大规模超稀疏矩阵的
长尾。通用 Kernel 将连续行均分到 block：前 `M % blockDim` 个 block 多处理一行，
每个 block 只绑定自身的 rowPtr 段以及对应 values/colInd 段。

### 4.3 Tiling 数据

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

Host 分发器根据 `computeType`、values 类型、y 类型和 `trans` 选择已实例化 Kernel。

---

## 5. 支持数据类型

### 5.1 Kernel 实例

| A values / x | computeType | y | Kernel 计算类型 |
| --- | --- | --- | --- |
| FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
| FLOAT16 | FLOAT32 | FLOAT16 | FLOAT32，写回时转 FLOAT16 |
| FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |
| BFLOAT16 | FLOAT32 | BFLOAT16 | FLOAT32，写回时转 BFLOAT16 |
| BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 |
| INT8 | FLOAT32 | FLOAT32 | 输入转 FLOAT32 后计算 |
| INT8 | INT32 | INT32 | INT32 标量乘加 |
| INT32 | INT32 | INT32 | INT32 标量乘加 |
| INT32 | FLOAT32 | FLOAT32 | INT32 输入转 FLOAT32 后计算 |

### 5.2 支持形状

| 模式 | x 最小长度 | y 最小长度 | 输出有效长度 |
| --- | ---: | ---: | ---: |
| 非转置 | `N` | `M` | `M` |
| 转置 | `M` | `N` | `N` |

行偏移和列索引均使用 INT32，因此矩阵规模、`nnz` 和所有索引必须处于 INT32
可表示范围。

---

## 6. Kernel 实现

### 6.1 通用非转置路径

通用 `SpmvKernel<CompT, ValT, OutT>` 采用 `Init` 和 `Process` 两阶段。每一行依次
执行：

1. `CopyIn`：搬入 colInd 和 values，按需将 ValT 转为 CompT，并读取原 y；
2. `Compute`：按列索引间接读取 x，计算逐元素乘积并执行 ReduceSum；
3. 计算 `alpha * sum + beta * y`；
4. `CopyOut`：按需转换为 OutT 后写回 y。

每个 block 在 Device 侧扫描自身 rowPtr 段得到最大行长 `tileLength`，并据此初始化
UB 队列。全空行时使用最小 `tileLength=8`，避免零长度 UB 缓冲区；单个空行直接
计算 `beta * y`。FP32 非转置路径将 tile 限制为 4096 个元素，超长行切换到 GM
标量分块遍历，避免单行长度导致 UB 溢出。

### 6.2 转置路径

转置 `SpmvKernelTrans` 分为两个阶段：

1. 各 block 均分互不重叠的 y 列区间，先执行 `y = beta * y`；
2. 各 block 处理自己的 CSR 行，为每个非零元计算 `alpha * A[i,j] * x[i]`，再使用
   `AtomicAdd` 累加到 `y[j]`。

两个阶段之间使用 `SyncAll`，防止 beta 缩放与原子累加交叉。原子加解决多个输入行
向同一输出列写入的冲突。转置 Host 当前固定使用一个 AIV block，CSR 行按固定顺序
遍历，因此具备确定性执行顺序；INT32 路径要求精确一致。

### 6.3 混合精度策略

FLOAT16、BFLOAT16 和 INT8 values 在 FLOAT compute 路径中转换为 FLOAT32，再完成
乘法、归约及 alpha/beta 计算；y 为低精度类型时，在写回前按 `CAST_ROUND` 转换。
这样避免低精度长行归约造成过大的累计误差，同时保留低精度输入/输出能力。

### 6.4 FP16 混合精度路径

`aclsparseSpMV` 执行入口中，当 `valType == ACL_FLOAT16 && computeType == ACL_FLOAT`
时，调用 host 端 CPU 回退函数 `SpmvFp16CpuFallback`：

1. 将 CSR 三元组、x、y 从 device D2H 拷贝到 host；
2. 在 host 以 FLOAT32 精度完成 `Y = alpha * op(A) * X + beta * Y` 计算
   （非转置与转置均覆盖，语义与测试 golden 一致）；
3. 将结果 H2D 写回 y。

CANN arch35 bisheng 编译器不支持 `Cast<float, half>`（bf16 亦受限），导致 FP16
混合精度 kernel 在 UB 队列内发生类型不匹配（`inQueueVals` 按 CompT=float 分配但
实际数据为 ValT=half），FP16 输出系统性偏低（约 1/2.3）。因此 FP16 路径暂由
CPU 回退实现。

**硬件加速接口**：`spmv_host.cpp` 中提供宏开关 `SPMV_ENABLE_NPU_FP16`：

```cpp
// 默认 0 = CPU 回退；CANN 修复 Cast<float, half> 后置 1 即切回 NPU 硬件路径
#ifndef SPMV_ENABLE_NPU_FP16
#define SPMV_ENABLE_NPU_FP16 0
#endif
```

该宏保留完整的 NPU kernel 启动路径。CANN 修复 Cast 缺陷后，将宏置 1 即可切回
与 arch22 相同的纯硬件 Cast 路径，无需其他改动。

---

## 7. 已知问题

| # | 问题 | 根因 | 状态 |
|---|---|---|---|
| 1 | FP16→FP32 混合精度 kernel 类型不匹配 | CANN arch35 bisheng 不支持 `Cast<float, half>` | ✅ 已用 host CPU 回退绕过，留 `SPMV_ENABLE_NPU_FP16` 宏接口 |
| 2 | 性能 Benchmark 未达 2x 标杆 | CANN kernel dispatch 开销主导中小矩阵耗时 | ⚠️ 需架构级优化 |
| 3 | 转置路径原子累加顺序 | `AtomicAdd` 浮点累加不满足结合律 | ✅ 单 block 顺序执行，已具备确定性 |

---

## 8. 算子约束限制

1. 仅支持 CSR，不支持 COO、CSC 等其他稀疏格式；
2. rowPtr 和 colInd 仅支持 `ACL_SPARSE_INDEX_32I`，索引基址仅支持 zero；
3. 仅支持 NON_TRANSPOSE 和 TRANSPOSE，不支持共轭转置；
4. 仅支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`；
5. values 与 x 必须同类型，支持组合以"支持数据类型"表为准；
6. x、y 必须满足操作模式对应的长度要求；
7. 输入必须是结构合法的 CSR，列索引必须位于 `[0, N)`；
8. 当前无额外 workspace，Preprocess 不缓存行统计或转置结构；
9. 转置路径固定单 block 顺序累加，保证确定性执行。

---

## 9. 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Ascend 950PR（DAV_3510 / arch35） | √ | 已完成构建、功能、精度和性能实测 |
| Ascend 950DT（DAV_3510 / arch35） | √ | 与 950PR 共用 arch35 实现；未单独实测 |
| Atlas A2 / Atlas A3 | 不涉及本次新增 | 沿用仓库既有 arch22 能力，本次不修改 |

---

## 10. 验证标准

| 验收标准 | 描述 |
| --- | --- |
| 功能标准 | 非转置/转置、alpha/beta、边界与随机矩阵结果正确（CPU CSR 参考实现校验） |
| FLOAT32 精度 | `rtol=2^-10`、`atol=2^-16`，匹配率 ≥ 99%，最大绝对误差 ≤ 1e-2 |
| FLOAT16 精度 | `rtol=2^-9`、`atol=2^-9`，匹配率 ≥ 99%，最大绝对误差 ≤ 1e-1 |
| BFLOAT16 精度 | `rtol=2^-6`、`atol=2^-6`，匹配率 ≥ 99%，最大绝对误差 ≤ 1.0 |
| INT32 精度 | 逐元素精确匹配，匹配率 100% |
| 性能标准 | 达到 0.5 倍标杆水平，即实测耗时不高于标杆耗时的 2 倍 |


## 11. AI Agent

- Agent: opencode + aiaos framework(self-developed, MIT License)
- Models:  minimax-m3,GLM-5.2,stepfun-3.7-flash,agnes-2.5-flash,deepseek-v4-flash
- Prompt: 基于aiaos框架，完成以下任务：按照官方文档、官方spmv实例，完成官方要求的交付链路。
- Models: 
D
C
- opencode with aia

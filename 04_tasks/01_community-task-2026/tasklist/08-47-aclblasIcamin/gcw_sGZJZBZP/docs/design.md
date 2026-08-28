# aclblasIcamin 算子设计文档

# 需求背景（required）

## 需求来源

本需求来源于 2026 年 8 月 CANN 社区任务“`aclblasIcamin` 算子开发（Ascend 950PR）”。目标是在
[`ops-blas`](https://gitcode.com/cann/ops-blas) 仓库中新增单精度复数向量最小模元素索引接口，使用
Ascend C kernel 直调方式适配 Ascend 950PR，并与 cuBLAS `cublasIcamin` 的核心功能和参数序列对齐。

代码实现位于：

- `blas/iamin/arch35/icamin_host.cpp`
- `blas/iamin/arch35/icamin_kernel.cpp`
- `test/iamin/icamin/`

待验收个人代码分支：

- 仓库：`https://gitcode.com/gcw_sGZJZBZP/ops-blas`
- 分支：`aclblas-icamin-950`

## 背景介绍

`Icamin` 用于查找单精度复数向量中 BLAS 模值最小元素的逻辑索引。BLAS 复数模定义为实部和虚部
绝对值之和，而不是欧几里得模：

$$
m_i = |\operatorname{Re}(x_{k_i})| + |\operatorname{Im}(x_{k_i})|,
\quad k_i = i \times incx, \quad i \in [0,n)
$$

返回值采用 Fortran/BLAS 惯例的 1-based 索引：

$$
result = 1 + \operatorname*{argmin}_{i \in [0,n)} m_i
$$

若多个元素模值相同，必须返回最小逻辑索引。输入 `aclblasComplex` 由两个 FP32 分量构成，在 GM 中
按照 `real0, imag0, real1, imag1, ...` 交错存储。

ops-blas 已有实数接口 `aclblasIsamin` 的 Ascend 950 arch35 实现，可复用其句柄、stream、workspace
及分块归约框架；但复数版本需要同时读取两个 FP32 分量，并保证模值相同、NaN、Inf 等情况下索引
语义与任务 CPU golden 完全一致。

# 需求分析（required）

## 需求描述

新增以下公共接口：

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```

接口通过 `handle` 绑定的 stream 异步执行，输入、输出均为 Device 内存。功能和约束如下：

| 参数 | 含义 | 类型/位置 | 约束与异常行为 |
| --- | --- | --- | --- |
| `handle` | BLAS 句柄及 stream | Host 标量 | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n` | 逻辑复数元素个数 | Host `int` | `n < 0` 返回 `INVALID_VALUE`；`n == 0` quick return |
| `x` | 复数向量 | Device COMPLEX64 | 正常计算路径不可为空 |
| `incx` | 复数元素步长 | Host `int` | `incx < 1` quick return，不反向遍历 |
| `result` | 1-based 最小索引 | Device INT32 | 不可为空；quick return 写 0 |

## 需求拆解

1. 新增公共 API 声明及 Ascend 950PR Host 实现。
2. 支持 `incx=1` 连续输入和 `incx>1` 非连续输入。
3. 每个复数元素计算 `abs(real)+abs(imag)`，返回全局最小值对应的 1-based 索引。
4. tie 场景严格返回最小逻辑索引。
5. NaN 按 CPU golden 语义处理：首元素为 NaN 时建立 `FLT_MAX` 基准，其余 NaN 跳过。
6. `n==0` 或 `incx<1` 不启动 kernel，异步写 `result=0`。
7. 输出 INT32 索引与 CPU golden bit-exact 一致。
8. 在 Ascend 950PR 上满足任务书的三组性能上限。

# 详细设计（required）

## 算子分析

### 数学公式

对每个逻辑元素 `i`：

```text
physicalIndex = i * incx
magnitude     = abs(x[physicalIndex].real) + abs(x[physicalIndex].imag)
result        = first_argmin(magnitude) + 1
```

比较规则为 `(magnitude, logicalIndex)` 的字典序最小值：先比较模值，模值相同再比较逻辑索引。

### 支持数据类型

| 输入/输出 | 数据类型 |
| --- | --- |
| `x` | COMPLEX64（real/imag 均为 FP32） |
| `result` | INT32 |

### 支持形状

- 逻辑输入为一维向量 `[n]`。
- `incx>=1` 时物理复数元素长度为 `1+(n-1)*incx`。
- 不涉及 broadcast、leading dimension 或原地计算。

## 算子实现

### Host 侧设计

#### 参数校验与 quick return

Host 按以下顺序校验：

1. `handle == nullptr`：返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `result == nullptr`：返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `n < 0`：返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 正常计算路径中 `x == nullptr`：返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. `n == 0 || incx < 1`：调用 `aclrtMemsetAsync` 在绑定 stream 上写 `result=0`，不启动 kernel。

#### 分核策略

```cpp
numBlocks = min(n, GetAivCoreCount());
perCoreN  = n / numBlocks;
lastCoreN = perCoreN + n % numBlocks;
```

前 `numBlocks-1` 个 block 分别处理 `perCoreN` 个逻辑元素，最后一个 block 处理余数。Ascend 950PR
性能规模使用全部可用 AIV 核，以获得足够的 GM 并行读取带宽。

#### 线程数策略

每个 block 的线程数根据单核数据量动态计算，并按 `SIMT_MIN_THREAD_NUM` 对齐：

```cpp
nthreads = min(
    AlignUp(CeilDiv(perCoreN, SIMT_MIN_THREAD_NUM), SIMT_MIN_THREAD_NUM),
    SIMT_MAX_THREAD_NUM);
```

三组性能用例分别使用约 128、512、2048 个线程。实测减少线程数会降低 GM 读取并行度，因此最终保留
动态满吞吐策略。

#### Workspace 规划

第一阶段每个 block 输出两个 32-bit 数据：

```text
workspace[2*block]     = blockMinValue (float)
workspace[2*block + 1] = blockMinIndex (uint32 bit pattern)
```

Host 按 64 个 float 对齐检查 workspace。64 个 AIV block 时有效及对齐空间均为：

```text
64 blocks * 2 * 4 bytes = 512 bytes
```

workspace 来自 handle 内部有效 workspace，不进行额外动态分配。

### Kernel 侧设计

Kernel 分为两个阶段：

```text
icamin_simt_kernel (多 block)
    每线程局部 argmin
    → block 内 tree reduce
    → workspace[block]

icamin_reduce_kernel (单 block)
    读取所有 block partial
    → 最终 argmin
    → result = index + 1
```

#### 第一阶段：线程局部计算

每个线程以 `blockDim.x` 为步长遍历本 block 数据：

```cpp
for (i = threadIdx.x; i < count; i += blockDim.x) {
    logicalIndex = start + i;
    offset = logicalIndex * incx * 2;
    magnitude = abs(x[offset]) + abs(x[offset + 1]);
    updateLocalArgmin(magnitude, logicalIndex);
}
```

每个线程只保留局部最小模值和对应的全局逻辑索引，不产生完整中间模值数组，GM 数据仅读取一次。

#### 连续地址专用化

性能用例均为 `incx=1`。kernel 使用模板生成连续和通用两个版本：

```cpp
template <bool UNIT_STRIDE>
IcaminCompute(...)
```

- 连续路径：`offset = logicalIndex * 2`。
- 通用路径：`offset = logicalIndex * stride * 2`。

连续路径消除循环内的通用 stride 乘法及参数依赖，A/B 实测提升约 1.1%～2.3%。

#### Block 内归约

线程局部结果写入 UB 数组后进行二叉树归约。比较规则为：

```cpp
rhsValue < lhsValue ||
(rhsValue == lhsValue && rhsIndex < lhsIndex)
```

因此归约顺序不会改变 tie 场景的最终索引，结果具有确定性。

#### 第二阶段归约

单 block kernel 将最多 64 组 partial result 从 GM 搬入 UB，串行比较得到最终 0-based 逻辑索引，
转换为 1-based INT32 后写入 `result`。该阶段仅处理固定的小规模数据。

#### NaN 和 Inf 处理

- 正常 NaN 元素不参与最小值更新。
- 当逻辑首元素为 NaN 时，将候选设置为 `(FLT_MAX, 0)`，与 CPU golden 的首元素基准一致；后续等于
  `FLT_MAX` 的元素不会替换更小索引。
- `abs(real)+abs(imag)` 允许溢出为 `Inf`，按照 IEEE FP32 大小关系参与比较。
- 全 NaN 场景返回最小逻辑索引 1。

## 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | --- |
| Ascend 950PR | √ |
| 其他产品 | 本任务不要求 |

## 算子约束限制

1. `n` 必须大于等于 0。
2. `incx<1` 按 BLAS iamin 族 quick return 处理，不支持负步长反向遍历。
3. 正常计算路径中 `x`、`result` 必须是有效 Device 指针。
4. 返回值为 1-based 索引；quick return 返回 0。
5. 不支持 broadcast，不涉及原地计算。
6. 结果读取前调用方必须同步 handle 绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | NPU INT32 索引与 CPU golden 完全一致，`actual == golden` | 社区任务书及生态算子精度标准 |
| 性能标准 | warmup 后有效采样大于 50 次，平均耗时不超过任务标杆 | 社区任务书 |

### 精度测试

测试使用 ops-blas GTest/CSV 框架，任务 CSV 共 1200 条，覆盖：

- 小尺寸、2 的幂及非对齐尺寸；
- `incx=1/2/3`；
- tie、全零、正负交替；
- NaN、Inf、极端值；
- `n<0`、`n==0`、`incx<=0`；
- handle、x、result 空指针；
- 大尺寸和性能规模。

另增加“首元素 NaN、后续模值为 `FLT_MAX`”定向回归用例。Ascend 950PR + CANN 9.1.0
环境中任务 CSV 及接口用例全部通过。

### 性能测试

测试采用 10 次 warmup、ACL Event 统计 100 次有效采样。连续地址专用化后的多轮中位数如下：

| n | incx | 实测平均耗时中位数 (us) | 任务上限 (us) | 结果 |
| ---: | ---: | ---: | ---: | --- |
| 1048576 | 1 | 35.290 | 58.17 | PASS |
| 4194304 | 1 | 49.635 | 114.95 | PASS |
| 16777216 | 1 | 119.889 | 389.12 | PASS |

最新定向语义修正后的单轮复测为 35.119、49.409、117.852 us，未引入性能回退。

## 内存分析

- 输入占用：`(1+(n-1)*incx) * sizeof(aclblasComplex)`。
- 输出占用：4 bytes。
- 最大额外 workspace：当前 64 blocks 时 512 bytes。
- 每个 SIMT block 的 UB partial 数组上限约为 `2048 * (4+4) = 16 KiB`，另有少量运行时开销。
- 算子不进行 Host/Device 动态内存分配，不存在随 `n` 增长的额外中间结果。

## 兼容性分析

本接口为新增算子，不改变 `aclblasIsamin` 及其他既有接口。声明加入公共
`include/cann_ops_blas.h`；实现仅在 arch35 目录参与 Ascend 950 构建。`incx=1` 专用路径与
`incx>1` 通用路径共享相同的比较规则和输出格式，不影响接口语义。

## 风险分析

| 风险 | 控制措施 |
| --- | --- |
| 多 block 归约导致 tie 索引不确定 | 比较键同时包含模值和全局逻辑索引 |
| NaN 破坏普通浮点比较 | 显式检测 NaN，并对首元素采用任务 golden 基准 |
| 复数地址计算错误 | 所有偏移以复数步长计算后乘 2，连续路径单独模板化 |
| workspace 越界 | Host 按 block 数计算并检查有效 workspace 大小 |
| 性能优化影响精度 | 每次调整后执行完整 CSV 回归和独立性能采样 |

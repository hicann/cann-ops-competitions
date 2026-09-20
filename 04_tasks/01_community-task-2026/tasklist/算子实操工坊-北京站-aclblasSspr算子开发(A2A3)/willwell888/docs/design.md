# Sspr算子实现设计文档（Atlas A2 / arch22）

> 依据 cann-ops-competitions 04_tasks/01_community-task-2026/resources/design_template.md 结构撰写。

# 需求背景（required）

## 需求来源

社区任务 2026（cann-ops-competitions 04_tasks/01_community-task-2026）：使用 Ascend C 为 ops-blas 仓补齐 spr 算子在 Atlas A2 训练系列（910B3, dav-2201 / arch22）的实现。对标接口 cublasSspr（参数序列一一对应）。

## 背景介绍

### Sspr算子实现

Sspr（Symmetric Packed Rank-1 Update）：对称 packed 矩阵的秩-1更新。

- ops-blas 已有 arch35（Ascend 950 系列）实现，本任务新增 arch22（Atlas A2）路径。
- arch22 参考实现路径：CANN 包内内置算子 syr/sger/tpmv（arch22）与 ops-blas arch35 sspr。

### Sspr算子现状分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库句柄（携带 stream） | aclblasHandle_t | - | 非空 | - |
| uplo | 更新三角 | aclblasFillMode_t | - | UPPER(121)/LOWER(122) | - |
| n | 矩阵阶数 | int | - | n >= 0 | - |
| alpha | 标量乘数 | const float* | float32 | 非空 | 标量 |
| x | 输入向量 | const float* | float32 | 非空（n>0 且 alpha≠0） | 1+(n-1)*|incx| |
| incx | x 步长 | int | - | != 0 | - |
| ap | packed 输入/输出矩阵 | float* | float32 | 非空（n>0 且 alpha≠0） | n*(n+1)/2 |

计算公式：`AP := alpha * x * x^T + AP`（仅 uplo 指定三角，column-major packed）。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 aclblasSspr，float32，支持 UPPER/LOWER、任意非零步长（含负步长），性能不低于任务书标杆（4 条性能 case）。

## 需求拆解

1. 支持 float32 数据类型，UPPER/LOWER 两种填充模式。
2. 支持 incx = ±1/±2/±3 等任意非零步长（负步长 = 逻辑反向）。
3. 覆盖 n = 0/1/质数/2 的幂±1/非对齐/大规模（2048+）等 shape。
4. 性能达标：n=512 UPPER ≤11.31us、n=1024 LOWER ≤27.07us、n=2048 UPPER ≤79.93us、n=4096 LOWER ≤475.91us（有效采样>50 均值）。

# 详细设计（required）

## 算子分析

### 数学公式

`AP[k] += alpha * x[i] * x[j]`，k 为 (i,j)（i≥j 或 j≥i，由 uplo 决定）在 column-major packed 数组中的索引。

### 支持数据类型

float32

### 支持形状

n ∈ [0, INT_MAX]；x 物理长度 1+(n-1)*|incx|；AP 长度 n*(n+1)/2。

## 算子实现

### 实现方案

#### host侧设计

tiling 策略：

- 参数校验对齐 cublas 语义：handle/alpha/x/ap 空指针、n<0、incx=0 返回 INVALID_VALUE；非法枚举返回 INVALID_ENUM；n=0 或 alpha=0 直接成功返回（无 kernel launch）。
- 分核：`useCoreNum = min(n, GetAivCoreCount())`，每核分得连续列块 `colsPerBlock = ceil(n/useCoreNum)`。列块对应 packed 数组的连续区间，天然免核间同步（每个 AP 元素只属于一个列，即一个核）。
- tiling data（sspr_tiling_data.h）：apAddr、xAddr、n、uplo、incx、alpha、useCoreNum，按值传递给 kernel。

#### kernel侧设计

Init（TPipe 初始化）+ Process，Process 按列批次组织，每批次仅付一次 MTE2_V / V_MTE3 / MTE3_MTE2 事件同步三连，将同步开销摊薄到批次而非每列（这是达到小 n 性能标杆的关键）。

**UPPER（列 c 需要因子 x[c]，行跨度 x[0..c]）——共享前缀：**

1. CopyIn：一条扁平 DMA 搬入共享前缀 x[0..colB) 到 UB 偏移 0（对齐）。
2. Compute：每列一条 `Muls(ubOut[outOff], ubX[0], alpha*x[c], len)`，输出槽 8 浮点（32B）对齐。
3. CopyOut：每列一条 MTE3 原子加 DataCopyPad 直写 AP（无 AP 回读，无 UB 内 read-modify-write）。

**LOWER（列 c 需要因子 x[c]，行跨度 x[c..n)）——stride-8 分组：**

连续列的后缀起点逐 1 递增，共享跨度会使 Muls 源操作数非 32B 对齐（向量算子硬约束，实测违反即 kernel 异常）。将批次按 `(c-col)%8 == r` 分成 8 组：组内各列源偏移均为 8 的倍数，一条跨度 DMA `x[col+r..n)` 即可服务整组，组内逐列对齐 Muls + 原子写。每组一次同步三连，同步次数仍比逐列模式少 8 倍以上。

**关键实现约束（实测踩坑结论）：**

1. 向量算子（Muls 等）的 src 与 dst LocalTensor 均必须 32B 对齐；GM 侧非对齐由 DataCopyPad 处理。
2. 标量单元（SetValue/GetValue）与向量/MTE 队列之间必须 `pipe_barrier(PIPE_V / PIPE_ALL)`（PIPE_BARRIER 宏在本工具链不可用，用小写 pipe_barrier）。
3. 原子加回写使用 `SetAtomicAdd<float>()` + DataCopyPad，结束后 `set_atomic_none()`；禁止从 VECIN 队列张量直接回写 GM。
4. aicore 端禁用 double 与浮点-整数转换；device 端不使用 lambda，辅助函数用 `__aicore__ inline`。

**strided（incx≠1）：**

x 按逻辑序一次性 Scatter 进 UB 缓存（xCache，≤12288 浮点），列因子与行跨度均从缓存取；超长 x 退化为逐点 GM SetValue 收集。之后复用 incx=1 的列处理纪律。

**UB 预算：** xBuf 66KB + outBuf 112KB（+ strided xCache ≤48KB）≤ 192KB。

### 性能优化历程（要点）

- v0.x 逐列 TQue 双缓冲：小 n 每列 ~2us 同步开销主导，n=512 达 49us（超标）。
- 引入 MTE3 原子加消除 AP 读回、修正向量算子对齐约束后（v12）：n=512 → 28.5us。
- 列批次摊薄同步（v16）：UPPER 4.2us、LOWER 45.5us（LOWER 逐列跨度 DMA 偏重）。
- LOWER stride-8 分组（最终版）：n=1024 LOWER → 15.7us，4/4 达标。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（910B3） | √ |

（Ascend 950PR/DT 走既有 arch35 实现；Atlas A3 走对应 arch 路径。）

## 算子约束限制

- 仅 float32；n ≥ 0；incx ≠ 0；uplo 仅 UPPER/LOWER。
- n=0 或 alpha=0 时不做任何更新。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib BLAS）生成；逐元素 `max(1e-2, 32*ULP)` + matchedRatio≥0.99；1001 条 GTest 全通过 | 任务书 §3.5 / 测试框架 |
| 性能标准 | 4 条 case 有效采样 120 次（>50）均值全部低于标杆：4.16/15.68/12.42/101.16us vs 11.31/27.07/79.93/475.91us | 任务书 §3.3 |

## 兼容性分析

接口已存在于 `include/cann_ops_blas.h`（aclblasSspr 声明，供其他产品线共用），本次仅新增 arch22 实现目录与测试，不改接口，不涉及兼容性问题。

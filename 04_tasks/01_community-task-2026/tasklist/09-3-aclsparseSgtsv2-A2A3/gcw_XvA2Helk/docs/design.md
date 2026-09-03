# 【社区任务】aclsparseSgtsv2 算子（A2/A3）设计文档

目标仓：https://gitcode.com/cann/ops-sparse
目标目录：`sparse/gtsv2/arch22/`（新建）
基线版本：`e0015bb`

---

# 需求背景（required）

## 需求来源

CANN 社区任务 2026 · 9 月批次，`9月社区任务-aclsparseSgtsv2算子开发(A2/A3)`。

## 背景介绍

### 任务形态：接口和算法都是现成的，缺的只有 arch22 这一支

`aclsparseSgtsv2` 与 `aclsparseSgtsv2_bufferSizeExt` 的 C++ 接口已在
`include/cann_ops_sparse.h:1336-1386` 定义并冻结，两阶段调用流程为

```
aclsparseSgtsv2_bufferSizeExt  →  aclsparseSgtsv2
```

仓内 `sparse/gtsv2/` 下只有 `arch35/`（Ascend 950 专属，677 行），`sparse/gtsv2/README.md`
的产品支持表把 A2/A3 两行明确标成「不支持」。本任务就是把这两行改成「支持」。

对我们是利好的地方在于，下面这些东西不用自己造：

- 接口声明已冻结，**不需要动 `include/`**；
- `test/gtsv2/gtsv2_golden.h` 与 `gtsv2_param.h` 是 arch 无关的公共文件，CPU golden 现成；
- `test/gtsv2/arch35/gtsv2_test.{cpp,csv}` 里 56 条用例（含白盒构造的主元路径）与 arch 无关，
  可以整份沿用到 arch22；
- 算法本身，`arch35/gtsv2_kernel.cpp` 的注释已经把带主元 Thomas 的两类 fill-in 和循环不变式
  逐条写清楚了，连和 LAPACK `dgtsv` 的对应关系都标了。

所以实质工作量落在**架构适配**，不在算法。

### arch35 那份实现搬不过来

`arch35/gtsv2_kernel.cpp` 用的是 Ascend 950 专有的 SIMT 模型：

```cpp
#include "simt_api/asc_simt.h"
__simt_callee__ / __simt_vf__ / asc_vf_call<Gtsv2SimtCompute>
constexpr uint32_t kGtsv2PivotThreadsPerBlock = 32U;   // 每 thread 一个 RHS 列，直接读写 GM
```

全仓扫了一遍，`sparse/*/arch22/` 下没有任何 `simt_api` / `__simt_callee__` / `asc_vf_call`
的使用——`scatter`、`sddmm`、`spmv` 的 arch22 一律是经典 Ascend C 向量模型
（`__aicore__` + `GetBlockIdx()` + `TPipe` + `TBuf/TQue` + `LocalTensor` + `DataCopyPad`）。
`gtsv2` 和 `gtsv2_nopivot` 是全仓仅有的两个 SIMT 算子，且都只有 arch35。

结论是 arch22 得按 SIMD 重新组织并行方式和内存布局，只有数值语义照抄 arch35。

### 算法

带部分选主元的 Thomas，对齐 LAPACK `dgtsv`。前向消元第 `i` 步比较 `|d'[i-1]|` 与 `|dl[i]|`：

不交换时 `mult = dl[i]/d'[i-1]`，走标准消元；需要交换时 `mult = d'[i-1]/dl[i]`，
并产生两类 fill-in——`du'[i] = -mult·du[i]` 落在第一上对角线，`du2'[i-1] = 原 du[i]`
落在第二上对角线（行交换把 old `du[i]` 推到了 row `i-1`）。

回代相应变成三项：`x[i] = (b'[i] - du'[i]·x[i+1] - du2'[i]·x[i+2]) / d'[i]`。

不对奇异矩阵做保护，零主元按 IEEE-754 自然产生 Inf/NaN——这是 arch35 README 里写明的
既有语义，任务书 2.1 第 4 条也这么要求，照做即可，不要自作主张加保护分支。

---

# 需求分析（required）

## 需求描述

在 arch22 上实现 `aclsparseSgtsv2` / `aclsparseSgtsv2_bufferSizeExt`，ABI、参数校验、
数值语义与 arch35 完全一致，核心计算在 NPU 完成，不得 CPU fallback。

## 需求拆解

| 编号 | 项 |
| --- | --- |
| T-1 | arch22 Kernel：SIMD 版前向消元 + 回代 |
| T-2 | arch22 Host：校验、分核、workspace 查询、下发 |
| T-3 | 大 `m` 分块路径，保证 `m` 无上限 |
| T-4 | C++ UT/ST：新建 `test/gtsv2/arch22/`，沿用公共 golden |
| T-5 | 精度：200 条官方泛化用例 |
| T-6 | 性能：203 条官方性能用例逐 case 对标 cuSPARSE |
| T-7 | 内存：workspace 与峰值内存采集 |
| T-8 | A5 交叉回归 |
| T-9 | README 产品支持表 + 架构差异说明 |

T-1/T-2 是主干，T-3 单独列出来是因为它有独立的正确性风险（见 3.2.2）。

---

# 详细设计（required）

## 算子分析

### 数学公式

`A · X = B`，其中 `A = tridiag(dl, d, du) ∈ R^{m×m}`，`B ∈ R^{ldb×n}` 列主序，
解 `X` 原地覆盖 `B`。调用方保证 `dl[0] = 0`、`du[m-1] = 0`。

### 支持数据类型

只有 `float32`。函数名前缀 `S` 即单精度实数，公开接口固定 `float*`，
`D`/`C`/`Z` 前缀不在本任务范围。

### 支持形状

`m ≥ 3`，无上限；`n ≥ 0`，`n = 0` 时提前返回 SUCCESS 且 `bufferSize = 0`；
`ldb ≥ max(1, m)`，允许 `ldb > m` 的 padding。

## 算子实现

### 实现方案

整个设计建立在一条 arch35 没有利用的性质上：

> 选主元只看 `dl`/`d`/`du`，跟 `B` 无关；而三对角矩阵 A 对所有 RHS 列是同一个。

`arch35/gtsv2_kernel.cpp` 里每个 SIMT thread 处理一个 RHS 列，各自把整套分解重算一遍，
workspace 因此是 `3·m` floats **每 RHS 列**。SIMD 下没有必要这么做，由此引出两个
arch22 独有的做法：

1. **同一批列共享一次分解**——分解只算一次，`cnt` 列复用同一组 `mult` 与交换标志；
2. **分解结果原地覆盖输入**——前向消元时 `d'`/`du'`/`du2'` 直接覆盖 `d`/`du`/`dl`。

第 2 点成立的依据是这条循环不变式：第 `i` 步读下标 `i`，只写下标 `i-1`。读恒比写超前
一个位置，任一槽位在被覆盖前都已经读过；而回代只需要 `d'`/`du'`/`du2'`，原始
`dl`/`d`/`du` 到那时已经没用。

两点合起来的效果是：**`m` 能整列驻留 UB 时，一个字节的 GM workspace 都不需要**。

#### 3.2.1 host 侧设计

**分核**沿 RHS 列维度切，每核一段连续列，`blockNum = min(n, aivCoreNum)`，
并回收上取整产生的空核。

**分块**分两种模式，共用同一套循环：整列能驻留 UB 时为常驻模式；否则沿 `m` 分块，
每块带一个 halo 前置元素，使块内局部下标继续满足上面那条「读 `i`、写 `i-1`」的不变式，
不必为分块另写一套索引。常驻模式本质是「只有一块、且这块覆盖整列」的退化情形，
不是独立的代码路径。

分块模式下块间只用寄存器传递跨块状态，没有核间同步；分解结果落 GM workspace 供回代
反向读回，每核独占 `3·m` floats（与 `n` 无关——分解对所有列是同一个）。

**平台相关量全部运行时查询**：AIV 核数取 `GetCoreNumAiv()`，核内 UB 容量取
`GetUbSize()`（两者仓内 `sparse/common/aclsparse_host_utils.h` 已提供）。
任务书要求在 910B3、910B4 和 A3 上分别验收，三款同为 DAV-2201、走同一份 arch22 代码，
把这两个量写死就要按型号改代码。查询结果做进程级缓存，避免每次求解都触发平台查询。

**参数上界**：tiling 的若干 int32 中间运算在 `m` 极大时会有符号溢出，
校验层设 `m` 的上界并在两个 Validate 函数中各拦一道，避免 kernel 拿到回绕后的参数。

**workspace 查询**：`n = 0` 返回 0；常驻模式返回一个 128 字节的最小对齐块；
分块模式返回 `blockNum × 3 × m × 4`（128 对齐）。常驻模式其实一个字节都用不上，
返回 128 是为了不破坏接口契约——参数校验在 `n > 0` 时要求 `pBuffer` 非空且 128 对齐
（对齐 arch35 的行为），返回 0 会让现有调用方传 `nullptr` 然后被自己的校验拦下。

`bufferSizeExt` 与 launch 共用同一个 tiling 计算入口。分块模式下 workspace 大小依赖
`blockNum`，两处各算一遍就有算得不一致、分配不足的风险。

#### 3.2.2 kernel 侧设计

```
Process
├── Eliminate   前向：逐块消元，末行单独落定
└── Substitute  后向：从末行起头，逐块回代
```

分三层组织：调度层只管块的推进顺序，算法层只做数学、不碰搬运，传输层只管 DMA 与同步。
选主元的判据抽成一个纯函数，前向和后向各调用一次——两处必须用同一个判据，
写成两份迟早会漂移。

末行 `i = m-1` 没有下一行可消去，直接落定，是两种模式唯一的分叉点：常驻模式下它就在
工作区里，分块模式下它不属于任何一块，需要单独搬运。

具体的分块尺寸、UB 内部布局与偏移、跨引擎同步事件的选取与插入位置、以及标量访问方式，
属于实现细节，由实现根据运行时查询到的硬件资源确定，不构成公开接口约束，本文不展开。

#### 3.2.3 关于并行度的一个取舍

按列分核意味着并行度等于 `n`。`n = 1` 时只起 1 个核，而 `m` 方向的 Thomas 是严格串行的，
这类场景吃不满机器；cuSPARSE 走的是 CR/PCR，深度只有 `O(log m)`，单个系统内部就能并行。
两者的性能曲线因此形态不同：我们的耗时随 `m` 线性增长，标杆基本与 `m` 无关。

反过来在大 `n` 上按列分核是有利的，因为 Thomas 的工作量是最优的（每列约 `2m` 次运算），
PCR 要做 `m·log m` 次。`n` 够大时既能占满核、又不做冗余计算，而任务书的三个正式性能
场景（P-01/02/03）都是 `n = 64/128`。

理论上可以上分块 Thomas + reduced system（SPIKE 那一套）让多核协作解单列，但带部分
选主元时 reduced system 的构造要复杂不少，正确性风险明显，且它只对 `n` 小的场景有意义。
本版不做，记在这里作为后续可选优化。

PCR 快路径不适用：它需要访问 `i ± delta` 直到 `m/2`，无法沿 `m` 分块，UB 容量不足以支撑。

### 目录结构

```
sparse/gtsv2/arch22/          新增：host、kernel、tiling 定义与 README
test/gtsv2/arch22/            新增：C++ UT/ST，沿用 ../gtsv2_golden.h 与 ../gtsv2_param.h
test/gtsv2/README.md          新增：环境、编译与复现步骤
sparse/gtsv2/README.md        修改：产品支持表，补架构差异说明
docs/zh/api_list.md           修改：产品支持表与 workspace 描述
```

CMake 不用改。`sparse/CMakeLists.txt` 会 GLOB 全部 `.cpp` 再按路径里的 `/archXX/` 过滤，
`SOC_VERSION=ascend910b3` 自动只编 arch22。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列（910B3 / 910B4） | √ |
| Atlas A3 系列 | √ |
| Atlas 800I/T A2 | √ |
| Ascend 950PR / 950DT | arch35 已支持，本任务不改动 |

三款 A2/A3 型号同为 DAV-2201 架构，CMake 把 `ascend910b*` 与 `ascend910_93*` 都映射到
`arch22`，共用同一份实现。型号间的差异（AIV 核数、核内 UB 容量）全部由 tiling 在运行时
查询，不需要按型号改代码或加分支。

## 算子约束限制

`m ≥ 3`；`n ≥ 0`；`ldb ≥ max(1, m)`；`n > 0` 时 `dl`/`d`/`du`/`B`/`pBuffer` 均不可为
`nullptr` 且 `pBuffer` 需 128 字节对齐；`dl[0]` 与 `du[m-1]` 由调用方置 0；仅 `float32`；
奇异矩阵不做保护，零主元按 IEEE-754 产生 Inf/NaN。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | CPU Golden 用 float64 求解，`\|actual - golden\| ≤ atol + rtol·\|golden\|` 且整体匹配率不低于 0.99，同时绝对误差不超过 `max(A, 32·ULP(golden))`；`rtol = 2^-10`、`atol = 2^-16`、`A = 1e-2` | 任务书 3.2 / [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能标准 | 逐 case 性能倍率 = 标杆 GPU 设备 Event median ÷ NPU median ≥ 0.25；预热不少于 10 次、正式采样不少于 30 次，报 median 与 P90，采样期间复用 workspace 并每轮恢复 B | 任务书 3.3 |
| 内存标准 | 输入输出量级远低于 500 MB，按「方案固有 workspace 不超过目标硬件 L2 Cache」验收 | 任务书 3.4 第 2 条 |

## 自测方案

**C++ UT/ST**：`test/gtsv2/arch22/`，沿用 arch35 的用例集并按任务书 3.5 的清单补齐。
覆盖单/多 RHS、不同 `m`/`n`/`ldb`、`m = 3` 最小值、`n = 0/1`、`ldb > m` 的 padding、
对角占优、随机触发主元、白盒构造的全交换/首步交换/末步交换/极端主元比、fill-in 累积链、
奇异矩阵的 Inf/NaN 传播、空指针与非法参数、未对齐 workspace、整数溢出边界、
`dl`/`d`/`du` 的输入只读性（逐位比较）、B 的原地输出语义，以及
`bufferSizeExt → 执行 → 释放` 的完整流程。

另有一组针对常驻/分块切换点附近取值的边界用例：官方性能用例的大 `m` 场景全部走常驻
路径，分块路径需要专门构造才能覆盖到。

**精度**：跑官方 `accuracy_cases.json` 的 200 条泛化用例，按上述判据判定。

**性能**：跑官方 `performance_cases.json` 的 203 条用例，逐 case 对比
`baseline_results/` 里的标杆 median。计时口径要留意：GPU 基线是 `cusparseSgtsv2` 的
C++ 调用，NPU 侧也要用 `aclrtEvent` 包住同样范围的 C++ 调用，才是 3.3 说的
「按对应来源的相同调用范围」。经 PyTorch 包装层的墙钟时间额外含适配开销，
不能直接与 C++ 基线比较。

**内存**：`collect_sparse_ops_npu_memory.py` 采 `input_baseline_*` / `peak_*` /
`extra_peak_*`，并核对 `bufferSizeExt` 查询值。

**Profiler**：`msprof --aic-metrics=PipeUtilization`，确认 Task Type 全部落在
AI_VECTOR_CORE（无 CPU fallback）、Block Num 等于预期核数。

**stream 异步语义**：单独一组用例覆盖非阻塞下发、同步后正确性、同 stream 连续下发的
顺序语义、`aclsparseSetStream` 切换 stream，以及与 `aclrtMemcpyAsync` 交织。

方案在提交本设计文档前已完成一轮端到端预研验证，上述精度、性能、内存三项标准在
910B3 上均达标。正式数据以自测报告中三型号（910B3 / 910B4 / A3）的实测结果为准。

## 兼容性分析

### 与 arch35（A5）的交叉回归

改动范围是两个新增目录加两处文档修改，不涉及任何 arch35 源文件、公共 Host、
`include/` 头文件或 CMake。

编译级回归用 `SOC_VERSION=ascend950` 构建验证，确认 CMake 的 arch 过滤把 arch22 完全
排除在外——编入的 gtsv2 目标文件只有 arch35 的那几个，零个 arch22 目标文件，
两个架构在编译期就不共享任何编译单元。950 硬件上的运行级回归待补。

### 接口兼容性

函数名、参数数量与顺序、参数类型、返回类型、导出符号与 arch35 完全一致，
`include/cann_ops_sparse.h` 不做修改。参数校验行为逐条对齐（`n < 0`、`m < 3`、`ldb` 下界、
空指针、`pBuffer` 对齐），`n = 0` 提前返回 SUCCESS 且 `bufferSize = 0`，
`bufferSizeExt` 允许 `B == NULL`。

### 数值兼容性

arch22 与 arch35 共用 `test/gtsv2/gtsv2_golden.h`。前向消元与回代逐行对齐 arch35 的
`Gtsv2ForwardSweep` / `Gtsv2BackwardSubst`，包括两类 fill-in 的处理和零主元的
Inf/NaN 传播语义。

---

# 风险点

| 风险 | 影响域 | 应对 |
| --- | --- | --- |
| 任务书 3.1 要求 910B3 / 910B4 / A3 三款型号分别验收，其中 910B4 在当前可申请的算力环境中未提供 | 该型号缺少实测数据 | tiling 依赖的两个平台量（AIV 核数、核内 UB 容量）均运行时查询，三款同为 DAV-2201 且共用同一份 arch22 代码，不存在按型号分支。已在 910B3 与 A3（`Ascend910_9382`）上分别完成构建、功能、精度、性能与内存的全量验证，同一份源码零改动通过，两款的精度误差、性能倍率与 workspace 查询值均一致；910B4 待环境到位后补充 |
| A5（Ascend 950）无可用硬件 | 3.5 要求的 A5 回归证据只能做到编译级 | 以 `SOC_VERSION=ascend950` 构建验证 CMake 的 arch 过滤，确认编入的 gtsv2 目标文件中零个来自 arch22，两个架构在编译期不共享任何编译单元；运行级回归待硬件到位后补充 |
| 按 RHS 列分核，并行度等于 `n` | `n` 很小时（尤其 `n = 1`）吃不满机器，性能低于以 CR/PCR 为基础的 GPU 标杆 | 属算法选型的固有特性，换取的是最优工作量（每列约 `2m` 次运算，PCR 为 `m·log m`）；已确认最差场景仍高于 0.25 倍验收线，且三个正式性能场景（P-01/02/03）均为多右端项，落在按列分核占优的区间 |
| FP32 误差沿消元方向累积 | 大 `m` 时尾部元素相对误差偏大 | 精度判据采用「整体匹配率 ≥ 0.99」与「逐元素不超过 `max(A, 32·ULP)`」双重标准，与任务书 3.2 一致；CPU golden 用 float64，避免参考值本身引入误差 |
| 奇异矩阵不做保护 | 零主元产生 Inf/NaN 并向后传播 | 对齐 arch35 与 cuSPARSE 的既有语义，任务书 2.1 第 4 条明确要求不加保护分支；测试中单列一组奇异用例，校验 Inf/NaN 的出现位置与 golden 一致 |
| 大 `m` 走分块路径，而官方性能用例的大 `m` 场景全部落在常驻路径 | 分块路径的问题不会被官方用例暴露 | 在常驻/分块切换点附近逐个取值单独构造一组边界用例，强制覆盖分块路径 |
| `m` 极大时切分计算的 int32 中间量溢出 | 溢出后会得到回绕的负值并据此启动 kernel | 在两个参数校验函数中各设一道 `m` 上界拦截，并配套溢出边界用例；实际先到的约束是 workspace 需要 `12·m` 字节，远早于该上界即会分配失败 |

---

# 交付物

| 类别 | 内容 |
| --- | --- |
| 算子实现 | `sparse/gtsv2/arch22/`：`gtsv2_host.cpp`（参数校验、切分决策、workspace 查询与下发）、`gtsv2_kernel.cpp`（前向消元与回代）、`gtsv2_kernel.h`、`gtsv2_tiling_data.h` |
| 测试代码 | `test/gtsv2/arch22/`：`gtsv2_test.cpp`（参数化用例 + 异常用例两个套件）、`gtsv2_test.csv`（分级用例表）、`gtsv2_npu_wrapper.h`；公共 CPU golden 与用例生成沿用 `test/gtsv2/` 下与架构无关的现有文件 |
| 文档 | `test/gtsv2/README.md`（环境、编译、复现步骤与接口说明）、`sparse/gtsv2/README.md`（产品支持表与架构实现说明）、`docs/zh/api_list.md`（产品支持与 workspace 描述） |
| 自测报告 | 环境信息（SOC 型号、CANN / 驱动 / 固件版本、代码提交号）、功能与精度结果、性能数据（median 与 P90）、峰值内存与 workspace 查询值、Profiler 证据、失败项说明 |
| 目标仓库 | https://gitcode.com/cann/ops-sparse ，接口声明沿用 `include/cann_ops_sparse.h`，不做修改 |

接口、参数校验与数值语义与 arch35 保持一致，本任务不改动 arch35 的任何源文件、公共 Host 层、`include/` 头文件与 CMake 配置。

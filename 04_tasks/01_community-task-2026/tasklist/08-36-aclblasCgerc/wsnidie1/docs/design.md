# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务 2026 年 8 月社区任务 —— aclblasCgerc 算子开发（950），任务列表编号 36。

## 背景介绍

### aclblasCgerc 算子实现优化

本算子为 ops-blas 仓 gerc 算子新增 arch35（Ascend 950PR）实现，基于 Ascend C 编程语言。gerc 此前仅有 arch22 实现（`blas/gerc/arch22/`），本设计面向 950PR 的 AIV 向量核特性全新实现连续访存场景的共轭秩-1 更新。

> **版本说明**：本设计文档为 **v2（寄存器融合架构）版**，整体重写自早期 strip 列分组多段 DMA 版本。早期 strip 版本性能未达标（0/4）并曾以"物理不可达"叙事提交，已被官方口径与真机复测证伪；本版改用寄存器分块融合 + 六列分组双缓冲路径，性能从零复测 4/4 达标（双口径），精度 1201/1201 全通过。

### 实现路径与 API 路径

- 算子实现路径：`blas/gerc/arch35/cgerc_host.cpp`、`blas/gerc/arch35/cgerc_kernel.cpp`、`blas/gerc/arch35/cgerc_tiling_data.h`
- API 路径：`include/cann_ops_blas.h` 中已有 `aclblasCgerc` 声明（禁止定义 950PR 私有平行接口）
- 测试路径：`test/gerc/cgerc/arch35/`（CSV 驱动 GTest 框架）

### aclblasCgerc 算子实现现状分析（Ascend C / Kernel 直调，非 TBE）

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 库上下文句柄 | aclblasHandle_t | - | 指向已创建的有效句柄；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | - |
| m | 矩阵 A 的行数 | int | - | m ≥ 0；m=0 为合法 no-op | - |
| n | 矩阵 A 的列数 | int | - | n ≥ 0；n=0 为合法 no-op | - |
| alpha | 复数标量乘数（只读） | const aclblasComplex* | COMPLEX64 | 不可为 nullptr；=(0,0) 为合法 no-op | 标量 |
| x | m 元复数向量（只读） | const aclblasComplex* | COMPLEX64 | m>0 且 n>0 时不可为 nullptr | 1+(m-1)·\|incx\| |
| incx | x 元素步长 | int | - | ≠ 0 且 ≠ INT_MIN（有意收紧，见§算子约束限制） | - |
| y | n 元复数向量（只读，计算取共轭） | const aclblasComplex* | COMPLEX64 | m>0 且 n>0 时不可为 nullptr | 1+(n-1)·\|incy\| |
| incy | y 元素步长 | int | - | ≠ 0 且 ≠ INT_MIN（有意收紧） | - |
| A | m×n 复数矩阵（原地更新） | aclblasComplex* | COMPLEX64 | m>0 且 n>0 时不可为 nullptr | lda×n（列主序） |
| lda | A 的主维度 | int | - | ≥ max(1, m) | - |

计算公式（Fortran 1-based 索引）：`A(I,J) ← A(I,J) + ALPHA · X(I) · CONJG(Y(J))`，I = 1..m，J = 1..n，A 列主序存储，原地更新。

### aclblasCgerc 算子功能分析

复数共轭秩-1 更新：以向量 x 与共轭后的向量 y 的外积乘 alpha 叠加进矩阵 A。与 geru 的唯一差异是对 y 取共轭。

- 输入：复数标量 alpha、复数向量 x/y（只读）、标量 m/n/incx/incy/lda
- 输出：复数矩阵 A（原地覆写，lda>m 时的 padding 区不修改）
- 支持数据类型：COMPLEX64（`aclblasComplex = {float real; float imag;}`，8 字节/元素）
- 支持广播：不涉及（x/y/A 为独立操作数）
- incx/incy 语义：任意非零整数（含负步长），负步长按 Netlib 从向量末端反向遍历，即第 col 个逻辑元素位于物理下标 `(n-1-col)*(-incy)`

# 需求分析（required）

## 需求描述

在昇腾 Ascend 950PR NPU 上使用 Ascend C 开发单精度复数共轭秩-1 更新算子 `aclblasCgerc`，实现与 cuBLAS `cublasCgerc` / Netlib BLAS `cgerc` 接口完全对齐的功能，精度按生态算子开源标准（`task_doc §3.2`：rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio≥0.99），性能达到官方标杆线（4 case 全部 PASS）。

## 需求拆解

1. 支持 COMPLEX64；语义严格对齐 Netlib cgerc：共轭只作用于 y，负步长从向量末端反向遍历
2. 支持 m/n ≥ 0 任意规格（含 m=0/n=0/alpha=(0,0) 的合法 no-op）、任意非零步长（INT_MIN 除外）、lda ≥ max(1,m) 的 padding 布局
3. 接口签名与 `include/cann_ops_blas.h` 已有声明一致，禁止定义 950PR 私有平行接口
4. 参数校验严格先于 quick return：handle → m/n ≥ 0 → alpha/incx/incy/lda → 指针（仅 m>0 且 n>0）→ no-op
5. 性能达标：4 case（512²/1024²/2048²/4096²）计时全部低于标杆线；官方口径 = msprof API 级（≈批量口径，见§性能标准）
6. 测试工程基于 ops-blas 仓 CSV 驱动 GTest 框架（1200 条用例），golden 由 cblas_cgerc 生成
7. 精度与稳定性不作为性能代价：保留 Netlib 括号顺序、显式物化 FP32 乘积、修复输出缓冲复用缺陷（见§详细设计）

# 详细设计（required）

## 算子分析

### 计算与主要开销

Cgerc 执行复数秩一更新 `A += alpha * x * conj(y)^T`。矩阵更新量为 O(mn)，而 x/y 输入量仅为 O(m+n)。因此性能关键在于：**减少矩阵 A 的重复读写**，并把 x、y 和 alpha 的复用尽量留在寄存器/片上缓冲中，避免中间数据反复进出 UB 与 GM。

### 数学公式

**Fortran 1-based 索引**：

```
A(I,J) ← A(I,J) + ALPHA · X(I) · CONJG(Y(J))
  I = 1..m, J = 1..n
  ALPHA = (aR, aI)   复数标量
  X(I)  = (xR, xI)   复数向量，物理下标 = (I-1)*incx（incx<0 时从末端反向：(m-I)*(-incx)）
  Y(J)  = (yR, yI)   复数向量，物理下标 = (J-1)*incy（incy<0 时同理反向）
```

复数展开（记 w = alpha · x，即 wRe = aR·xR − aI·xI，wIm = aR·xI + aI·xR；conj(y) = (yR, −yI)）：

```
A.re(I,J) += wRe(I) · yR(J) + wIm(I) · yI(J)
A.im(I,J) += wIm(I) · yR(J) − wRe(I) · yI(J)
```

其中 **strip/legacy 回退路径**将 alpha 折叠进 x 侧的 w（w = alpha·x，每行块构造一次，按列摊销）以复用 Axpy；**寄存器融合优化路径按 Netlib cgerc.f 原始顺序**先算 temp = alpha·conj(y)（每列一次）再 A += x·temp。二者数学等价、FP 舍入路径不同但均在容差内（见§精度标准）。

### 支持数据类型

| 数据类型 | 说明 |
| --- | --- |
| COMPLEX64 | 单精度复数，实部/虚部各 float32，内存布局 `{float real; float imag;}`，等价于交织 float 对，8 字节/元素 |

全程 float32，无 cast、无精度升降；仅涉及交织↔平面/寄存器布局转换。

### 支持形状

- A 为 m×n 列主序矩阵（物理存储 lda×n，lda ≥ max(1,m)）
- x/y 为一维向量，物理长度 1+(len−1)·|inc|
- 不涉及广播
- 原地更新：A 既是输入也是输出，lda padding 区不修改

## 算子实现

### 实现方案概述（寄存器融合架构）

**核心思路**：将计算链重构为"寄存器驻留 + 一次向量融合读写"，把 A 的热路径 GM 往返压到最低。四项关键设计（对应 `Cgerc_优化思路_20260910.md` §2）：

1. **寄存器分块融合**：将 x 以 64 复数为一个行块，四个 64 复数行块驻留寄存器，共用同一份 y 广播及 `alpha * conj(y)` 折叠结果；对每个 A 行块，用**一次向量融合**完成 A 的读、更新、写，消除中间 UB 搬运。
2. **列分组与双缓冲预取**：默认**六列分组**，配双缓冲预取——当当前列组在向量管道上计算时，下一列组的 A 数据已在 MTE2 上预取入 UB，重叠 DMA 与计算。UB 容量计算同时计入 x、y、A，不只按单一矩阵块估算。
3. **按输入约束分流**：对齐、单位步长（incx=incy=1、lda=m、地址对齐）走优化入口；非单位步长和不满足对齐条件的输入保留已验证的**通用回退路径**。零 y 列（`y[J]==0`）在优化入口与标量路径**显式保留** A 原始位模式（回退路径的覆盖范围见§通用回退路径），不参与计算。
4. **限定编译优化范围**：Cgerc 专用 `-O2 -ffp-contract=off`，通过 `CGERC_OPTIMIZED` CMake option 限定于 `cgerc_host.cpp` + `cgerc_kernel.cpp`，不修改其他算子编译选项。

#### 3.2.1 host 侧设计

##### 1. 参数校验与 no-op

校验链严格先于 quick return：

1. `handle != nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `m >= 0`、`n >= 0` → `ACLBLAS_STATUS_INVALID_VALUE`
3. `ValidateCgercParams`：alpha 非空 → incx/incy ∉ {0, INT_MIN} → lda ≥ max(1,m) → x/y/A 非空（仅 m>0 且 n>0）
4. Quick return：`m==0 || n==0 || alpha==(0,0)` → 返回 SUCCESS，**不读 x/y**

`INT_MIN` 收紧理由：`-INT_MIN` 是 C/C++ 未定义行为（signed overflow），kernel 用 `-incx` 做反向索引依赖此约束。已在 `blas/gerc/README.md` 登记。

##### 2. 数据分块和内存优化策略（UB 容量与分核）

- **UB 容量查询**：`GetCachedUbSize()` 通过 `PlatformAscendCManager::GetCoreMemSize(CoreMemType::UB, ...)` 查询实际 UB 大小，fallback 为仓内具名常量 `UB_SIZE = 248*1024 = 253952`（`blas/common/helper/kernel_constant.h:16`）；最终 `ubUsable = min(ubSize, UB_SIZE)`，扣除安全余量后作为可分配预算。UB 预算同时计入 x 行块、y 缓存与 A 双缓冲列组三部分。
- **AIV 侧核数确定**：`numBlocks = min(n, aivCoreNum)`，`aivCoreNum = GetCachedAivCoreCount()`（进程内缓存）。零值防护：查询失败返回 0 时 fallback 为 1 并打印 OP_LOGE，避免除零。
- **Kernel 侧列分配**：`colsPerCore = ceil(n / blockNum)`；`start = blockIdx × colsPerCore`；`end = min(start + colsPerCore, n)`；`start >= n` 的核闲置（start=end=0）。核间列段 disjoint、无 atomic 竞争。

##### 3. tilingKey 规划策略

host 侧按输入约束决策优化入口 vs 回退路径，kernel 服从 `tiling.tilingKey`：

| tilingKey | 路径 | 判定条件 |
| --- | --- | --- |
| 3 | 寄存器融合优化入口 | `incx==1 && incy==1 && m%64==0 && lda>=m && lda%4==0 && perCore<=4096` + 地址对齐 + UB 装得下六列分组双缓冲 |
| 2 | TINY 简化路径 | `m*n<=65536`（简化 Init 序列、关闭 yCache/strip） |
| 1 | strip 列分组路径（旧优化路径，保留） | 对齐 + 单位步长但不满足 key=3 门控 |
| 0 | 通用回退路径（legacy） | 非单位步长 / 不对齐 / 其余所有形状（正确性优先） |

`Tiling` 结构体零初始化（POD 按值传入 kernel，未赋值字段否则会携带不确定值到达设备端）。Tiling 决策链在非优化路径通过 `OP_LOGD` 打印 key/m/n/lda/rowTile/分组宽度等；**key=3 优化入口为静默 dispatch（不打日志）**，其可观测性由自研 perf_bench 与 sanitizer 矩阵覆盖。

#### 3.2.2 kernel 侧设计

##### 优化入口（寄存器融合 + 六列分组双缓冲）

```
Init：绑定 xGM_/yGM_/aGM_、计算列区间 [colStart_, colEnd_)、
      初始化 x 行块寄存器缓冲 + y 广播缓冲 + A 双缓冲列组、
      一次性加载 y（incy==1 走 DMA + 标量尾补齐；否则标量收集）

Process（每行块 r0，四个 64 复数行块驻留寄存器）：
  1. 折叠 alpha*conj(y) 与 x 行块 → w（每行块一次，按列摊销）
  2. for each 六列分组 [g0, g0+gc)：
       a. 双缓冲预取：MTE2 预取下一列组 A → UB buffer[next]
       b. 一次向量融合：读 A(buffer[cur]) → 用 w 与 y 广播更新 → 写回 A
       c. MTE3 排空后翻转 cur/next
```

- **六列分组**：每组一次处理 6 列，配双缓冲使 DMA 与向量计算重叠。
- **一次向量融合读写**：A 的读-更新-写在向量管道上一次完成，不落中间 UB 缓冲，显著减少搬运。
- **双缓冲预取**：cur/next 两块 A 列组缓冲交替，MTE2 预取与向量计算流水重叠。

##### 通用回退路径

非单位步长 / 不对齐输入走已验证的逐列/逐块回退路径，正确性优先，不复用优化入口的对齐假设。零 y 列（`y[J]==0`）直接保留 A 原始位模式（不写、不读改写），避免 `0 × Inf/NaN` 破坏 A 中既有的特殊值语义。**覆盖范围**：reg 路径（zeroY mask + Select 保留原位）与小矩阵标量路径（显式 skip）为显式保留；strip/legacy 路径依赖 Axpy(0) 恒等（有限值安全），Inf/NaN+zero-y 组合由标量路径覆盖（1201/1201 全过）。

### 精度与稳定性设计（不作为性能代价）

对应 `Cgerc_优化思路_20260910.md` §3，三项硬约束：

1. **保留 Netlib 运算括号顺序**：`-ffp-contract=off` 禁止隐式 FMA 融合，且不做任意重关联，避免改变极值（Inf/NaN）语义。求值严格按 `A += (alpha·x) · conj(y)` 的括号化次序。
2. **小矩阵标量路径显式物化 FP32 乘积**：使溢出后的 Inf/NaN 行为与 Golden（cblas_cgerc）一致。
3. **修复旧 VECIN 输出缓冲 MTE3 前复用缺陷**：旧实现中 VECIN 队列同时作为输出缓冲时，槽位在 MTE3 尚未读完时即被 MTE2 复用，导致块内随机错误。本版补 **MTE3→MTE2 同步**后再释放槽位。该修复针对已复现的块内随机错误，属正确性修复，**不是放宽精度容差**。

### GM 侧数据传输约束

GM 侧块数据传输统一走 `DataCopyPad`（GM 侧 Scatter 在 arch35 上两次独立实验证实静默失败：写入数据丢失、无错误码，故禁用）。UB 内部的交织↔平面/寄存器布局转换为可靠原语，全程在 UB 内、不涉及 GM、不在 A 数据热路径。`DataCopyPad` 按 32B 补齐写入，回退路径的 <32B 尾块依赖此行为。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950T A2 | |
| Atlas 950T A3 | |
| Ascend 950PR | √ |

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | m ≥ 0、n ≥ 0；incx/incy ≠ 0 且 ≠ INT_MIN（`-INT_MIN` 是未定义行为，kernel 依赖 `-incx` 做反向索引）；lda ≥ max(1,m)；alpha 不可为 nullptr；m>0 且 n>0 时 x/y/A 不可为 nullptr |
| 非连续 Tensor 支持 | 向量通过 incx/incy 支持任意非零步长（含负步长，Netlib 反向语义）；非单位步长/不对齐走通用回退路径，不支持超出 inc/lda 语义的非连续内存访问 |
| broadcast 规则 | 不涉及 |
| dynamic shape 要求 | 不要求，m/n 为运行时入参 |
| 原地与视图语义 | A 原地覆写，不返回视图；lda padding 区不修改；零 y 列保留 A 原始位模式 |
| 确定性计算要求 | 不要求（`task_doc §2.5` 明文），但本实现核间列段 disjoint、无 atomic 竞争，结果实际确定 |
| 空 Tensor 与 0 维处理 | m=0 或 n=0 或 alpha=(0,0) 为合法 no-op，返回 SUCCESS 且不读 x/y |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |
| 浮点语义 | 保留 Netlib 括号顺序，禁隐式 FMA（`-ffp-contract=off`）与任意重关联，保证 Inf/NaN 传播次序与 golden 一致 |
| <32B GM 写 | `DataCopy`（非 Pad）不支持 <32B GM 写；`DataCopyPad` 按 32B 补齐，回退路径依赖此行为，已由 m=1/2/3 及 `m%4≠0` 系列用例真机验证通过 |

# 可维可测分析

## 精度标准

### 验收口径（`task_doc §3.2`，以此为验收依据）

| 参数 | 值 |
| --- | --- |
| 逐元素判据 | \|actual − golden\| ≤ atol + rtol × \|golden\| |
| rtol | 2⁻¹⁰ = 9.765625e-4 |
| atol | 2⁻¹⁶ = 1.52587890625e-05 |
| required_matched_ratio | 0.99 |
| max_abs_error_limit | 1e-2 或 32×ULP（逐元素取 max） |
| golden | cblas_cgerc（Netlib BLAS 复数实现，列主序，`OPENBLAS_NUM_THREADS=1` 单线程） |

### 实测精度（从零复测，DevEnv_189086，kernel 零修改）

| 测试集 | 结果 | 说明 |
| --- | --- | --- |
| 全量（1201 条 = 1200 CSV-driven + 1 NullHandle） | **1201/1201 PASS** | 耗时 183,462 ms；严格门限：matchedRatio≥0.99、atol=2⁻¹⁶、rtol=2⁻¹⁰、maxErr≤max(0.01, 32×ULP)、NaN/Inf 分类一致、padding 逐位保持 |
| 偶发失败用例压力测试 | **60/60 PASS** | TC_PF_1120 重复 30 次 = 30/30；TC_PF_1147 重复 30 次 = 30/30；零偶发 |

golden = cblas_cgerc（Netlib，列主序，单线程）。修复旧 VECIN 输出缓冲复用缺陷后，历史块内随机错误已闭合。

### Sanitizer 检查（四工具 24 job 全覆盖，覆盖闭合，不宣称零警告）

针对早期单 job 600s 超时导致的采集缺口（memcheck 仅 2/6、racecheck/synccheck 从未运行、工具版本未采集），本轮改为**单工具 × 单用例拆分为 18 job**（3 工具 × 6 真实用例），每 job timeout 550s + `nohup/setsid` 后台脱离 SSH 会话，**18/18 全部完成 rc=0、无超时**（最长 racecheck TC_PF_1120 = 512s）。6 个 filter 先经 `--gtest_list_tests` / grep csv 验证均命中真实用例（`TC_PF_1001`、`TC_PF_1010`、`TC_PF_1120`、`TC_PF_1147`、`TC_FL_104`、`TC_EX_0123`），修复了早期 TC_PF_2/500/800 前缀未命中问题。随后以同规格补测第四工具 initcheck（单工具 × 单用例 6 job，日志 19..24），**6/6 rc=0、无超时、0 次重试**，至此四工具 × 6 用例 = **24/24 job 全部完成**。

- **工具**：`mssanitizer`，路径 `/usr/local/Ascend/cann-9.1.0/tools/mssanitizer/bin/`（不在 PATH，runner 内 export）
- **工具版本（已采集）**：`mssanitizer 26.1.0-a9b9e9ea5c701cc8939017999fdd02c2e15905c2`（24 job 全部一致）
- **调用语法**：`mssanitizer --tool=<memcheck|racecheck|synccheck|initcheck> --log-level=warn <wrapper.sh>`（`-t=` 报 `param 'tool' contains invalid characters`，必须用 `--tool=`；wrapper 包裹被测二进制以规避 mssanitizer 吞掉 `--gtest_filter`）
- **被测二进制**：官方仓 `gitcode.com/cann/ops-blas` @0ea22c8 + changes.patch 构建的 `cgerc_test`，kernel/host 源码零修改

| 工具 | 用例 | rc | 计数结果 | dispatch kernel |
| --- | --- | --- | --- | --- |
| memcheck | 6/6 全部 | 0 | **内存错误 0**（各例全部 kernel launch `No error detected.`；PERF 例 55 次 / 功能例 1 次），test_passed=1/failed=0 | TC_PF_1001→cgerc_register_kernel；余 5 例→cgerc_kernel |
| racecheck | 6/6 全部 | 0 | **竞争警告 0**（各例 `No error detected.`），test_passed=1/failed=0 | 同上 |
| synccheck | TC_PF_1001 | 0 | **0 警告**（真实 clean，走 register 路径，55 launch 全 `No error detected.`） | cgerc_register_kernel |
| synccheck | TC_PF_1010 | 0 | Redundant wait_flag = **3520** | cgerc_kernel |
| synccheck | TC_PF_1120 | 0 | Redundant wait_flag = **779240** | cgerc_kernel |
| synccheck | TC_PF_1147 | 0 | Redundant wait_flag = **466620** | cgerc_kernel |
| synccheck | TC_FL_104 | 0 | Redundant wait_flag = **64** | cgerc_kernel |
| synccheck | TC_EX_0123 | 0 | Redundant wait_flag = **3886** | cgerc_kernel |

**分工具结论**：

- **memcheck — 6/6 clean（0 内存错误）**：可支持"未发现内存错误"结论。
- **racecheck — 6/6 clean（0 竞争）**：可支持"未发现数据竞争"结论。
- **synccheck — 1/6 clean、5/6 有真实冗余同步警告（不宣称零警告，实测非 0）**：仅 `TC_PF_1001` 真实 0 警告（dispatch 到 `cgerc_register_kernel` register 路径）；其余 5 例 dispatch 到主 tiled kernel `cgerc_kernel`，均报 `Redundant wait_flag instructions detected`，方向 `PIPE_MTE2 → PIPE_S in cgerc_kernel`，源码定位 `cgerc_kernel.cpp:552 / 581 / 643 / 741`，条数如上表如实列出。该警告属**性能类冗余同步**（多余的 wait_flag 指令），**非内存错误、非数据竞争**，所有 synccheck job 功能仍 PASSED（rc=0）；按红线未改 kernel/host 源码，仅如实记录条数与源码行号。
- **initcheck — 6/6 全 clean（未初始化告警实测 0）**：uninitialized 全类 / read 细分 / `====== WARNING` / `====== ERROR` 四类 grep 计数全 0，PERF 用例 55 次 launch、功能用例 1 次全部 `Sanitizer finished ... No error detected.`，gtest 全 PASSED；日志拉回本地后二次 grep 复核一致。

**四工具 × 6 用例总矩阵（24 job，全 rc=0）**：

| 用例 | memcheck 内存错误 | racecheck 竞争警告 | synccheck 冗余同步警告 | initcheck 未初始化告警 | 汇总 |
| --- | --- | --- | --- | --- | --- |
| TC_PF_1001 | 0 | 0 | 0 | 0 | 四工具全 clean |
| TC_PF_1010 | 0 | 0 | 3520 | 0 | synccheck 有冗余同步警告 |
| TC_PF_1120 | 0 | 0 | 779240 | 0 | synccheck 有冗余同步警告 |
| TC_PF_1147 | 0 | 0 | 466620 | 0 | synccheck 有冗余同步警告 |
| TC_FL_104 | 0 | 0 | 64 | 0 | synccheck 有冗余同步警告 |
| TC_EX_0123 | 0 | 0 | 3886 | 0 | synccheck 有冗余同步警告 |

**覆盖闭合披露（原两项残留缺口均已闭合，无未闭合项）**：

- **initcheck 已补测（6/6 全 clean）**：未初始化内存告警实测 0（四类 grep 计数全 0 且本地复核一致），四工具覆盖闭合。
- **4 个大日志已完整归档本地（双重 sha256 校验）**：synccheck 的 4 份原始日志（TC_PF_1120 ≈ 353MB、TC_PF_1147 ≈ 211MB、TC_PF_1010 ≈ 1.6MB、TC_EX_0123 ≈ 1.75MB）已经远端 `gzip -c`（压缩后 3.9 / 2.8 / 0.03 / 0.026MB）+ PTY base64 分片传输（逐片字节 + sha256 校验，无重传，总耗时 14.6s）完整拉回本地 `full_logs/`，合并 gunzip 后与远端做 gz 层 + 解压原文层**双重 sha256 对比全部一致**；本地对完整日志 grep 复核 warning 计数与上表逐例一致；远端源文件仅只读未改动。
- **唯一保留的如实表述**：synccheck 5/6 用例存在 `Redundant wait_flag` 冗余同步警告（**性能类、非正确性**，条数 3520 / 779240 / 466620 / 64 / 3886 如实列出）——**不宣称零警告**；其余 memcheck / racecheck / initcheck 三工具实测 0 告警。

> 对应 F 表：**F164-F170（DevEnv_189086）**。

## 性能标准

### 官方验收口径

官方验收口径 = **msprof API 级耗时**（设备侧/kernel 级），经 E48 三计时器偏差实验，msprof API 级 ≈ **批量口径（batch）**，偏差 **2.3%**，二者可互换；**host launch 底座不计入**验收耗时（验收判定对象为 device-side span）。950PR 有 L2 缓存，H100 L2 公平性论点不获豁免。

> ⚠️ 早期 strip 版本的"物理不可达 / 0/4 达标 / verify_performance.py 口径数学缺陷"等论证**已全部作废**：官方明确"不可达论证 + 如实报告"不能作为性能通过依据，性能必须真达标。本版寄存器融合实现在官方口径下 4/4 达标（下表）。

### 达标线与判定式

```
npu_us ≤ 标杆线（gpu_ms / 0.4）即 PASS
```

| Case | m×n | 标杆线 (μs) | A 矩阵字节 |
| --- | --- | --- | --- |
| 1 (TC_PF_1001) | 512×512 | **8.675** | 2.0 MiB |
| 2 (TC_PF_1002) | 1024×1024 | **12.850** | 8.0 MiB |
| 3 (TC_PF_1003) | 2048×2048 | **44.633** | 32.0 MiB |
| 4 (TC_PF_1004) | 4096×4096 | **241.085** | 128 MiB |

### 实测性能（从零复测，DevEnv_189086 / Ascend 950PR / CANN 9.1.0，kernel 零修改）

测试参数：complex64、incx=incy=1、lda=m、alpha=(1,0)；20 warmup；每 rep 100 采样；**3-rep p50 取 median**（因 4096² spread >8% 加跑第 4 rep）；温度 61-64°C，功耗 211-213W，无热节流。

**Isolated 口径**（单发 ACL event：record→call→record→sync→elapsed）：

| Case | 3-rep Median (μs) | 标杆 (μs) | 余量 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 512×512 | **7.024** | 8.675 | 19% | **PASS** |
| 1024×1024 | **11.526** | 12.850 | 10% | **PASS** |
| 2048×2048 | **27.039** | 44.633 | 39% | **PASS** |
| 4096×4096 | **99.891** | 241.085 | 59% | **PASS** |

**Batch 口径**（host chrono，N=100 连续提交 + 末尾 sync /N，≈ msprof API 级官方口径）：

| Case | 3-rep Median (μs) | 标杆 (μs) | 余量 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 512×512 | **4.512** | 8.675 | 48% | **PASS** |
| 1024×1024 | **8.798** | 12.850 | 32% | **PASS** |
| 2048×2048 | **24.460** | 44.633 | 45% | **PASS** |
| 4096×4096 | **96.708** | 241.085 | 60% | **PASS** |

**口径实现位置**：batch 口径由 PR 内测试 wrapper（cgerc_npu_wrapper.h，连续 N 提交+末尾单次 sync）实现，用于 CI 冒烟；isolated 口径（aclrtRecordEvent→call→record→sync→elapsed）由独立 perf_bench 工具实现、非 PR 交付件，两口径偏差经 E48 实验为 2.3% 可互换。

**结论：4/4 达标（双口径全部 PASS，余量 10-60%）。** batch 系统性低于 isolated，差额即单发 event record/sync 的固定开销（512² 差 2.5μs、4096² 差 3.2μs）。

### 稳定性与限定披露

| Case | Caliber | Spread% | Status | 备注 |
| --- | --- | --- | --- | --- |
| 512×512 | isolated / batch | 1.97% / 2.80% | **STABLE** | |
| 1024×1024 | isolated / batch | 0.90% / 1.09% | **STABLE** | |
| 2048×2048 | isolated / batch | 0.47% / 0.41% | **STABLE** | |
| 4096×4096 | isolated / batch | 11.90% / 12.27% | **UNSTABLE（判定仍 PASS）** | 跨 rep 双峰 |

- **4096² 跨 rep 双峰（限定披露）**：rep1/3 ≈99.9μs、rep2/4 ≈111.7μs，spread 11.9%。归因 4096² complex64 A 矩阵 = 128MB 远超 L2 64MB，65s rep 间隔后 cache 冷/热态交替。**最差值 112.271μs 仍仅为标杆 241.085μs 的 47%**，PASS 判定不受影响。同一进程内连续 3-rep（原始 bench）为 batch 105.83-105.97μs（spread 0.13% STABLE）、isolated 109.65-110.89μs（spread 1.13% STABLE）。
- **1024² outlier 噪声（限定披露）**：原始 bench 交叉验证中 1024² isolated rep2 均值 = 12.89μs，微超标杆 12.85μs 仅 **+0.3%**，属测量噪声（原始 bench isolated 为 100 sample 算术平均值，含 outlier）；自研 perf_bench 的 p50 口径下，1024² isolated 全部 4 rep 均 ≤11.672μs，远低于标杆。

### 原始 bench 交叉验证

| Case | batch_event (3-rep) | isolated_event (3-rep, 均值) | 标杆 | All PASS? |
| --- | --- | --- | --- | --- |
| 512×512 | 4.54/4.51/4.49 μs | 8.32/8.47/8.18 μs | 8.675 | Yes |
| 1024×1024 | 8.77/8.79/8.83 μs | 12.77/12.60/12.89 μs | 12.850 | 11/12（rep2 均值 12.89 微超 +0.3%，噪声；p50 口径全 ≤11.672） |
| 2048×2048 | 24.32/24.34/24.31 μs | 28.21/28.59/28.38 μs | 44.633 | Yes |
| 4096×4096 | 105.88/105.83/105.97 μs | 110.16/109.65/110.89 μs | 241.085 | Yes |

交叉验证与自研 perf_bench 同向，性能结论一致。

### 硬件发现（950PR 实机调试经验）

| 发现 | 说明 |
| --- | --- |
| GM 侧 Scatter 静默失败 | 两次独立实验证实；A 的 GM 写入统一走 `DataCopyPad` |
| UB 内布局转换可靠 | 交织↔平面/寄存器转换全程在 UB 内，已验证正确 |
| AIV 核数 | 由 `GetCachedAivCoreCount()` 查询，无硬编码；实测返回 56。交付路径仅用 AIV，分核规范 |
| `DataCopyPad` 支持 <32B | 按 32B 补齐写入，回退路径依赖此行为 |
| MTE3→MTE2 同步承载性 | 输出缓冲槽位须在 MTE3 读完后再被 MTE2 复用，否则块内随机错误 |
| `-ffp-contract=off` | 禁隐式 FMA，保留 Netlib 括号顺序，保证 Inf/NaN 语义与 golden 一致 |
| 寄存器驻留 + 双缓冲 | x 行块驻留寄存器 + 六列分组双缓冲预取，重叠 DMA 与向量计算，是 4/4 达标的结构性来源 |

## 从零复测记录

**目的**：验证本 PR 补丁在独立环境从零搭建后仍达标（精度 + 性能），kernel/host 源码零修改。

**环境**：DevEnv_189086 / Ascend 950PR / CANN 9.1.0 / openEuler 24.03 LTS-SP3；基线 commit `0ea22c81b4b721f723730586c48a21c6d759da6a`（官方仓 `gitcode.com/cann/ops-blas`）；补丁 `changes.patch`（317,130 bytes）。日期 2026-09-11。

| 步骤 | 操作 | 结果 |
| --- | --- | --- |
| 系统依赖 | `dnf install -y lapack-devel blas-devel gcc-gfortran` | rc=0（lapack 3.12.0 + blas 已装） |
| CBLAS wrapper | gcc `-shared -fPIC` 自编 **Fortran→C cblas_cgerc wrapper**（远端无外网，wget netlib 失败） | rc=0，`cblas_cgerc` 符号确认 |
| Clone repo | `git clone https://gitcode.com/cann/ops-blas.git` | rc=0（官方仓成功） |
| Checkout | `git checkout 0ea22c81...` | rc=0，HEAD = `0ea22c8` |
| Apply patch | `git apply --check && git apply changes.patch` | check rc=0、apply rc=0、无冲突 |
| CMake configure | `-DSOC_VERSION=ascend950 -DCGERC_OPTIMIZED=ON -DBUILD_TEST=ON -DTEST_NAMES=cgerc` | rc=0 |
| CMake build | `--target cgerc_test -j8` | rc=0，产物 `cgerc_test` + `libops_blas.so` |
| Bench build | g++ `-std=c++17 -O2 -ffp-contract=off` | rc=0 |

**关键确认**：`-ffp-contract=off` 通过 `CGERC_OPTIMIZED` CMake option 限定于 `cgerc_host.cpp` + `cgerc_kernel.cpp`，不影响其他算子。全链路 rc=0，kernel/host 源码零修改。

**复测结论**：

| 维度 | 判定 | 详情 |
| --- | --- | --- |
| 搭建 | **成功** | clone → 依赖安装 → patch → build 全链路 rc=0 |
| 精度 | **达标** | 1201/1201 + 偶发用例 60/60 stress |
| 性能 | **PASS（4/4）** | 双口径全部低于标杆，余量 10-60%（见§性能标准） |
| Sanitizer | **四工具 24 job 全覆盖（覆盖闭合）** | memcheck/racecheck/initcheck 6/6 clean（0 内存错误、0 竞争、0 未初始化）；synccheck 1/6 clean、5/6 有性能类冗余 wait_flag（非正确性，功能全 PASSED，条数如实）；4 份大日志已全量归档（双重 sha256）；工具版本已采集 mssanitizer 26.1.0（见§Sanitizer 检查 / F164-F170） |

**最终判定：从零搭建仍达标（精度 + 性能两维度）。** Sanitizer 维度四工具覆盖闭合（24/24 job rc=0：memcheck/racecheck/initcheck 实测 0 告警、synccheck 冗余警告如实计数不宣称零警告、4 份大日志已全量归档双重 sha256 校验）。

**复测产物指针**（只读数据源）：`ascend950/measurements/run_zero_189086/`
- `gate_report.md`：从零复测门禁报告（搭建/精度/性能/sanitizer/结论全文）
- `perf_results.csv`：性能汇总 CSV（32 行，4 case × 2 口径 × 4 rep）
- `setup_from_zero.sh`：可复跑从零搭建脚本
- 18 份分阶段日志（00_connectivity ~ 07c_rep4_data）
- F 表追加 `950PR硬件手册_F表.md` F156-F163（含 F162b，共 9 行）

## 验收证据与交付物（2026-09-11）

**验收重交包**：`aclblasCgerc_验收重交_20260911.zip`（7.0 MB / 47 文件 / manifest.sha256 46 行全量校验通过）：

| 包内项 | 内容 | 对应 9/7 打回意见 |
|---|---|---|
| 自测报告.md | 按任务书自测报告模板 8 章（从零搭建/精度/性能/Sanitizer/遗留披露/覆盖说明/总结） | “参考任务书自测报告模板提供自测报告” |
| 精度数据/ | 1201 逐用例结果 csv + 偶发 60/60 记录 | “完整的精度数据” |
| 性能数据/ | perf_results.csv 32 行（4 case × 双口径 × 4 rep）+ 性能对比表 | “性能对比数据” |
| 截图/ | accuracy_tail / perf_table / sanitizer_matrix / build_rc 四张 PNG（由远端日志渲染，首行标注，原始日志见 logs/ 与 logs/full_logs/） | “及截图” |
| logs/ + logs/full_logs/ | 18 份分阶段日志 + 4 份完整 synccheck 大日志（双重 sha256）+ sanitizer 24 job 日志 | 可复核性 |
| design_v2 + links + README | 本设计文档、PR/分支链接、目录与证据对应表 | — |

**最终判定数值**：精度 1201/1201 + 偶发 60/60；性能 4/4 双口径 PASS（isolated 7.024/11.526/27.039/99.891 μs、batch 4.512/8.798/24.460/96.708 μs，标杆 8.675/12.85/44.633/241.085，余量 10-60%，3-rep p50）；sanitizer 四工具 24/24 job rc=0（mem/race/init 实测 0 告警，synccheck 5/6 冗余 wait_flag 性能类如实计数）；从零搭建全链路 rc=0（kernel/host 零修改）。

**披露**：不宣称零警告（synccheck 冗余非零，条数见上）；4096² 跨 rep 冷热双峰 spread 11.9%（最差 112.271 μs = 标杆 47%，同进程连续 3-rep 0.13%/1.13% STABLE）。

## 兼容性分析

新算子（gerc 族新增 arch35 分支），不涉及跨版本兼容性分析。接口声明复用 `include/cann_ops_blas.h` 中已有 `aclblasCgerc` 声明，可与其他产品线共用。

**测试方案说明**：基于 ops-blas 仓 CSV 驱动 GTest 框架（1200 条用例 + 1 条独立 NullHandle 测试 = 1201），golden 使用 cblas_cgerc（Netlib BLAS 复数实现，列主序，`OPENBLAS_NUM_THREADS=1` 单线程）。测试类别覆盖：L0 基础、SQ 尺寸、AB alpha 特殊值、RC 矩形、LD lda padding、INC 步长、FL 填充含 Inf/NaN、ED 边界负向、EX 扩展、PF 性能。

**范围说明**：本 PR 交付寄存器融合优化入口（对齐 + 单位步长）与通用回退路径（非单位步长/不对齐），二者均仅用 AIV，分核规范。

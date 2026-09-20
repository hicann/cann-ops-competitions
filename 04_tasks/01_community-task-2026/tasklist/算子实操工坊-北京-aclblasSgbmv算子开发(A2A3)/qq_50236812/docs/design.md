# aclblasSgbmv 算子设计文档

| 项目 | 内容 |
|---|---|
| **任务名称** | 算子实操工坊-北京-aclblasSgbmv算子开发(A2/A3) |
| **taskId** | 57fcc5dad709456caae4b2b6d6003371 |
| **任务链接** | https://www.hiascend.com/activities/task-center/details/57fcc5dad709456caae4b2b6d6003371?menu=trends |
| **任务分类** | CANN 算子开发 |
| **开发者账号** | qq_50236812 (BuerHj) |
| **适配硬件** | Atlas A2 / Atlas A3 系列产品（架构目录 arch22，测试芯片 Ascend 910B3） |
| **CANN 版本** | CANN 9.1.0 |
| **涉及仓库** | `cann/ops-blas`（`blas/gbmv/arch22/`、`include/cann_ops_blas.h`、`test/gbmv/sgbmv/arch22/`） |
| **待验收代码仓** | https://gitcode.com/qq_50236812/ops-blas |
| **待验收分支** | `feat-aclblasSgbmv-arch22` |

---

# 需求背景（required）

## 需求来源

昇腾任务中心“算子实操工坊-北京-aclblasSgbmv算子开发(A2/A3)”。面向 Atlas A2/A3 系列产品（Ascend 910B，arch22），使用 Ascend C 编程语言开发单精度实数（FLOAT32）通用带状矩阵与向量乘法算子 `aclblasSgbmv`。通过 ops-blas 开源仓工程框架，提供基于句柄（Handle）的 Kernel 直调接口，完成算子设计、高性能双路径实现以及严苛的精度和性能自测全流程。

## 背景介绍

### 1. 算子应用与数学背景
通用带状矩阵-向量乘法（General Banded Matrix-Vector Multiply，`gbmv`）是 BLAS-2（Basic Linear Algebra Subprograms Level 2）的核心运算之一。带状矩阵广泛存在于偏微分方程数值解（有限元法、有限差分法）、样条插值、动力学系统仿真以及大型稀疏物理系统建模中。高效的带状矩阵乘法能在保证稀疏内存紧凑存储的同时，极大加速线性系统的迭代求解过程。

### 2. 对标规范
本算子完全对标 NVIDIA cuBLAS 库中的 `cublasSgbmv` 以及 Netlib BLAS `sgbmv` 参考实现，其功能与参数语义、边界特判、快速返回等逻辑与工业标准严格一致。

### 3. 现有工程现状
在 `cann/ops-blas` 开源仓中，已有针对 arch35（GPU 风格 SIMT 架构）的 `blas/gbmv/arch35` 实现。本任务面向 Atlas A2/A3（arch22，华为自研 DaVinci AI Core/AIV 架构），需全新开发基于 Ascend C 的高效 Vector 实现，两者在架构目录上保持完全隔离解耦，公共接口统一由 `include/cann_ops_blas.h` 声明。

---

# 需求分析（required）

## 需求描述

### 1. 核心数学公式
`aclblasSgbmv` 计算单精度实数带状矩阵与向量的乘加更新：
$$y = \alpha \cdot op(A) \cdot x + \beta \cdot y$$

其中：
- $A$ 为 $m \times n$ 的带状矩阵，以列主序（Column-Major）紧凑带状格式存储，前导维度为 $lda$；
- $kl$ 为矩阵的下对角线数（sub-diagonals），$ku$ 为矩阵的上对角线数（super-diagonals）；
- $x, y$ 为单精度实数（FLOAT32）向量，支持步长参数 $incx, incy$（可为正或负）；
- $\alpha, \beta$ 为单精度实数（FLOAT32）标量指针（Host 内存）；
- $op(A)$ 由枚举参数 $trans$ 控制：
  - $trans = \text{ACLBLAS\_OP\_N}$ 时：$op(A) = A$；
  - $trans = \text{ACLBLAS\_OP\_T}$ 时：$op(A) = A^T$；
  - $trans = \text{ACLBLAS\_OP\_C}$ 时：$op(A) = A^T$（实数域下共轭转置与普通转置等价）。

### 2. 向量长度与映射语义
- **非转置模式（$trans = \text{OP\_N}$）**：
  - $x$ 的逻辑长度为 $n$，$y$ 的逻辑长度为 $m$；
- **转置模式（$trans \in \{\text{OP\_T}, \text{OP\_C}\}$）**：
  - $x$ 的逻辑长度为 $m$，$y$ 的逻辑长度为 $n$；
- **步长寻址规则**：
  - 若步长 $> 0$，第 $k$ 个逻辑元素（0-based）物理偏移为 $k \cdot stride$；
  - 若步长 $< 0$，依据 Netlib BLAS 标准规范，第 $k$ 个逻辑元素物理偏移为 $(len - 1 - k) \cdot |stride|$。

## 需求拆解

1. **接口规范性**：严格契合 `include/cann_ops_blas.h` 声明，符合 C 链接规范；
2. **入参全防御**：对指针空值、负维度、非零步长、合法带宽高宽比、枚举值域进行完整校验；
3. **特例与快速返回**：支持 $m=0$ 或 $n=0$ 的合法 no-op；$\alpha=0$ 时的纯缩放或恒等更新；$\beta=0$ 时覆盖输出；
4. **双路径协同架构**：针对单位步长主流场景构建 8 输出向量化 Gather 快速路径；针对极端步长及超大带宽构建列主序流式累加展开回退路径；
5. **数值鲁棒性与带外隔离**：严格隔离带状矩阵存储中的带外填充（即使填充含 NaN/Inf 也绝不污染计算结果），$\beta=0$ 下彻底规避脏输出数据读取；
6. **多核负载均衡**：调用 `GetAivCoreCount()` 动态探测 AIV 物理核心，以 32B/128B 对齐切分输出区间，杜绝多核 Cacheline 伪共享（False Sharing）；
7. **自测全面性**：覆盖 1201 条 GTest 用例（1200 条 CSV + 空句柄）与 2496 组严格逐元素扩展回归，精度 100% 通过，4 组性能标杆超越幅度达到 44.8% ~ 62.0%。

---

# 详细设计（required）

## 算子分析

### 数学公式与物理存储映射

在列主序带状存储（Column-Major Band Storage）中，逻辑矩阵元素 $A(i, j)$（$0 \le i < m, 0 \le j < n$）仅在有效带状区域内占用存储空间，即满足：
$$\max(0, j - ku) \le i \le \min(m - 1, j + kl)$$

其在物理一维数组中的索引偏移计算公式为：
$$\text{offset}(i, j) = j \cdot lda + (ku + i - j)$$

物理存储矩阵尺寸为 $lda \times n$，其前导维满足约束：
$$lda \ge kl + ku + 1$$

对于每一列 $j$，其有效行元素在内存中物理地址严格连续排列，带状有效元素分布示意如下：

```
                       Column j
            +-------------------------------+
            |               :               |
[ku + i - j]    | A(i, j)                       |  <-- 物理连续存储段 (带宽 = kl + ku + 1)
[ku + i+1 - j]  | A(i+1, j)                     |
            |               :               |
            +-------------------------------+
```

### 支持数据类型与形状

- **数据类型**：FLOAT32（输入输出矩阵与向量、标量 $\alpha, \beta$ 均为单精度实数）；
- **内存分布**：
  - 矩阵 $A$、向量 $x$、向量 $y$ 位于 Device 内存（支持全局内存连续或带步长映射）；
  - 标量 $\alpha$、标量 $\beta$ 为 Host 内存指针；
  - 句柄 `handle` 位于 Host 内存，包含下发计算任务的 `aclrtStream`。

---

## 算子实现

### 实现方案

```
aclblasSgbmv (Host 入口)
   │
   ├─ 1. 入参完整性校验 (Handle, Trans, m, n, kl, ku, lda, incx, incy, 指针校验)
   │
   ├─ 2. 快速返回分支判断 (m=0/n=0 -> Success; alpha=0 -> 缩放/返回)
   │
   ├─ 3. 40核无伪共享负载均衡划分 (非转置32B对齐，转置128B对齐，确定各核处理行区间)
   │
   └─ 4. Stream 异步启动 Kernel <<<gridDim, nullptr, stream>>>
         │
         └─ sgbmv_kernel (Device / AIV)
               │
               ├─ 分支 A：向量化 Gather 快速路径 (incx=1, incy=1, 33<=带宽<=129, x无0值)
               │     ├─ 向量倍增生成 A/x 偏移表
               │     ├─ 向量 Compares 快速检测零值 x
               │     ├─ 8行批量 Gather/Gatherb 载入与广播向量乘加
               │     ├─ 向量 Compares + Select 矩阵边缘 NaN 批量置零
               │     ├─ 三段式向量规约与合并
               │     └─ 向量流水线完成 alpha/beta 缩放与 DataCopyPad 写回
               │
               └─ 分支 B：标量流式展开回退路径 (大带宽、小带宽、输入含0、非单位步长)
                     ├─ 列主序连续切片按列遍历累加
                     ├─ 8 路循环展开消除 RAW 指令延迟
                     └─ 零值 x 快速列跳过 (Zero Skip)
```

### 3.2.1 Host 侧设计

#### 1. 参数校验与防御性编程
Host 侧遵循工业级 BLAS 标准实施严密校验：
- `handle == nullptr`：返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
- `trans` 不属于 `ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C`：返回 `ACLBLAS_STATUS_INVALID_ENUM`；
- 维度边界：$m < 0$ 或 $n < 0$ 或 $kl < 0$ 或 $ku < 0$：返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 带宽越界：$m > 0$ 时 $kl \ge m$ 或 $n > 0$ 时 $ku \ge n$：返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 前导维：$lda < kl + ku + 1$ 或 $lda < 1$：返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 步长：$incx == 0$ 或 $incy == 0$：返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 指针为空判断：$\alpha == \text{nullptr}$ 或 $\beta == \text{nullptr}$ 返回 `INVALID_VALUE`；在 $m > 0$ 且 $n > 0$ 时，若 $y == \text{nullptr}$ 或（$\alpha \ne 0$ 时 $A == \text{nullptr}$ 或 $x == \text{nullptr}$）均返回 `INVALID_VALUE`。

#### 2. 特例与快速返回（Quick Return）
- **空矩阵 No-Op**：若 $m = 0$ 或 $n = 0$，算子无需执行任何计算，直接返回 `ACLBLAS_STATUS_SUCCESS`；
- **$\alpha = 0$ 纯缩放**：
  - 若 $\beta = 1.0f$，输出向量 $y$ 保持不变，直接返回 `ACLBLAS_STATUS_SUCCESS`；
  - 若 $\beta == 0.0f$ 或其他值，仅执行 $y = \beta \cdot y$，无需读取矩阵 $A$ 和向量 $x$。

#### 3. 分核策略：40 核动态负载均衡与无伪共享对齐切分
Atlas 800I A2 具备 40 个 AIV 核。为了彻底根除多核并发写回时的 L2 Cacheline 伪共享（False Sharing），切分算法依据模式实施精细对齐：
- **非转置连续模式**：以 8 个 float（32 字节）为分核单位，保证各核 DMA 写回起点对齐。在 40 核、4096 行场景下，最重核心由原 128 行降为 104 行，核心间负载均衡度大幅提升。
- **转置与非连续模式**：以 32 元素（128 字节）整对齐切分，前 $useNumBlocks - 1$ 个核心的处理区间 $[startRow, endRow)$ 均严格对齐 128B，尾部多余元素归属最后一个核心。

```cpp
const bool fastN = tiling.alpha != 0.0f && IsFastpathN(tiling);
CalcCoreRowRange(outDim, numBlocks, blkIdx, startRow, endRow, fastN ? SIMD_WIDTH : CORE_ALIGN_SIZE);
```

#### 4. Tiling 数据结构
```cpp
struct SgbmvTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t kl;
    uint32_t ku;
    uint32_t lda;
    int64_t incx;
    int64_t incy;
    uint32_t trans;
    float alpha;
    float beta;
    uint32_t numBlocks;
    uint32_t outDim;
};
```

---

### 3.2.2 Kernel 侧设计

#### 1. 双路径协同计算拓扑

算子根据带宽高低与输入特征自适应分流：
1. **向量化 Gather 快速路径**：
   - 触发条件：单位步长（$incx=1, incy=1$）、带宽 $33 \le kl+ku+1 \le 129$、满足 UB 容量约束且输入 $x$ 无零值；
   - 一次协同计算 8 个连续输出元素（对应一个 32B 连续段），全面发挥 DaVinci Vector 算力。
2. **标量流式累加展开回退路径**：
   - 触发条件：大带宽（$>129$）、极小带宽（$<33$）、输入 $x$ 片段含零值或非单位步长；
   - 采用列主序连续切片按列遍历累加 + 8 路循环展开，无跨步开销与 RAW 依赖。

#### 2. 向量化 Gather 核心技术突破

- **向量倍增生成偏移表**：
  标量仅写入 8 个初始索引，随后使用向量加法指令（`Adds`）进行倍增扩展与缩放（$8 \to 16 \to 32 \dots$），消除单核启动时的冗长标量循环。行间距向上对齐到 8 个 float，每份临时表上限由 6144 字节压缩至 4352 字节。
- **向量 Compares 快速检测零值 $x$**：
  在矩阵 $A$ 的 DMA 搬运与准备期间，利用向量比较指令 `Compares(..., EQ, 0.0f)` 产生位掩码，标量按 32 位字汇总，并显式屏蔽尾字多余位。若检测到 0 值则平滑降级至标量分支，规避无效访存。
- **带外填充 NaN 隔离与批量边缘掩码**：
  带状矩阵边界外的内存槽位可能包含未初始化数据或 NaN。通过逐行索引比较与 `Select` 向量指令，将越界位置的无效乘积置零，从而使得整组 8 行输出能够完全统一复用高效的分段规约，彻底杜绝 NaN 污染。
- **三段式向量规约与合并**：
  带宽最多拆分为三个 64 元素片段进行规约，再合并分段和，流水线完全填满。
- **向量流水线写回与 $\beta=0$ 防护**：
  计算结果的 $\alpha$ 与 $\beta$ 缩放合并完全在向量流水线上执行。当 $\beta=0$ 时，禁止从全局内存读取旧 $y$（规避旧 $y$ 中的脏数据或 NaN 干扰），直接完成写回；非对齐尾部采用 `DataCopyPad` 安全写入。

#### 3. 标量流式连续累加核心算法（回退路径）

外层遍历矩阵列 $j$，内层针对 $A[:, j]$ 的物理连续段进行流式累加：
$$y[r_{start} \dots r_{end}] \mathrel{+}= x_j \cdot A[k_{start} \dots k_{end}, j]$$

```
[传统行遍历 - 跨步跳跃访存]                 [列主序流式连续累加 - 连续单位步长]
Row i: A[i, j0] -> A[i, j1] -> A[i, j2]        Col j: A(rStart, j)
         |          |          |                         |
       Stride     Stride     Stride                     Contiguous Slice
       (lda-1)    (lda-1)    (lda-1)                    (Unit-Stride / Stride 1)
         v          v          v                         v
       +128       +128       +128               y[rStart ... rEnd] += A[:, j] * xj
 (长依赖链 / 标量溢出)                           (零数据依赖 / 指令级流水满载)
```

内层采用 8 路展开消除气泡：
```cpp
uint32_t k = 0;
for (; k + 7 < len; k += 8) {
    yCol[k + 0] += aCol[k + 0] * xj;
    yCol[k + 1] += aCol[k + 1] * xj;
    yCol[k + 2] += aCol[k + 2] * xj;
    yCol[k + 3] += aCol[k + 3] * xj;
    yCol[k + 4] += aCol[k + 4] * xj;
    yCol[k + 5] += aCol[k + 5] * xj;
    yCol[k + 6] += aCol[k + 6] * xj;
    yCol[k + 7] += aCol[k + 7] * xj;
}
for (; k < len; ++k) {
    yCol[k] += aCol[k] * xj;
}
```

#### 4. 192 KiB UB 内存布局规划与分时复用

所有偏移均经过静态断言检查（`static_assert`），总占用严格控制在 192 KiB 以内：

| 区域名称 | UB 字节起点 | 预留大小 | 用途说明与复用规则 |
|---|---:|---:|---|
| `UB_OFFSET_Y` | 0 | 1024 B | 当前 Tile 输出累加缓冲区 |
| `UB_OFFSET_Y_ORIG` | 1024 | 768 B | 原 $y$ 输入缓冲区 / 分时复用 Gatherb 块偏移 |
| `UB_OFFSET_EDGE_MASK` | 1792 | 256 B | 边缘有效判定位掩码 |
| `UB_OFFSET_X` | 2048 | 2048 B | 向量 $x$ 输入暂存区 |
| `UB_OFFSET_SUMS` | 4096 | 4096 B | 分段和 / 分时复用零值检测位掩码 |
| `UB_OFFSET_A_INDICES` | 8192 | 4352 B | 矩阵 $A$ 批量 Gather 偏移表 |
| `UB_OFFSET_X_INDICES` | 12544 | 4352 B | 向量 $x$ 批量 Gather 偏移表 |
| `UB_OFFSET_A_VALUES` | 16896 | 4352 B | Gather 载入的矩阵元素 / 向量乘积 |
| `UB_OFFSET_X_VALUES` | 21248 | 4352 B | Gather 载入的 $x$ 元素 / 重排向量 |
| `UB_OFFSET_A` | 32768 | 剩余空间 | 矩阵 $A$ 原始列切片主搬运区 |

#### 5. 硬件流水线同步依赖矩阵

| 阶段转换 | 同步原语 | 作用解释 |
|---|---|---|
| MTE2 搬运 $x$ $\to$ 向量零检测 | `SET_FLAG(MTE2, V)` / `WAIT_FLAG(MTE2, V)` | 确保 $x$ 搬运入 UB 后再执行向量比较 |
| 向量零检测 $\to$ 标量条件分支 | `SET_FLAG(V, S)` / `WAIT_FLAG(V, S)` | 保证掩码位字读取在比较完成后执行 |
| 索引准备与 A DMA 重叠 | `SET_FLAG(MTE2, S)` / `WAIT_FLAG(MTE2, S)` | 标量分支等待矩阵 A 数据就绪 |
| 原 $y$ 缓冲区复用 | `SET_FLAG(V, MTE2)` / `WAIT_FLAG(V, MTE2)` | 确保 Gatherb 地址表使用完毕后方可载入原 $y$ |
| 向量结果写回 | `SET_FLAG(V, MTE3)` / `WAIT_FLAG(V, MTE3)` | 确保计算向量缩放合并完成后再触发 DMA 写回 |

---

## 支持硬件

| 产品 / 芯片型号 | 支持情况 | 已验证环境 |
|---|:---:|---|
| Atlas 800I A2（Ascend 910B3） | 支持 | CANN 9.1.0，40 个 AIV 核实机验证通过 |
| Atlas 800T A2（Ascend 910B4） | 支持 | 架构兼容（arch22） |
| Atlas 800I/T A3 系列 | 支持 | 统一 DaVinci arch22 矢量指令集支持 |

---

## 算子约束限制

1. **矩阵存储**：仅支持单精度实数（FLOAT32）列主序带状格式，$lda \ge kl + ku + 1$ 且 $lda \ge 1$；
2. **步长要求**：$incx \ne 0$ 且 $incy \ne 0$，原生支持负步长及跨步寻址；
3. **内存边界**：调用方须确保 Device 显存跨度合法；
4. **异步执行**：算子在 Handle 指定 stream 上异步下发，主机读回前须调用 `aclrtSynchronizeStream` 同步。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准项 | 标准要求 | 标准来源 | 达成情况 |
|---|---|---|---|
| **精度标准** | FLOAT32：rtol $\le 2^{-10}$ ($9.77\times 10^{-4}$)，atol $\le 2^{-16}$ ($1.53\times 10^{-5}$)，通过率 $\ge 99\%$，最大绝对误差 $\le 10^{-2}$ 或 $32\times\text{ULP}$ | 昇腾生态算子开源精度标准 | **达成率 100.0%**（1201/1201 GTest 全部通过；2496/2496 严格回归全部通过，最大绝对误差 $5.80\times 10^{-6}$） |
| **性能标准** | 4 组基准用例平均单次耗时（Avg time，$\mu\text{s}$）不高于任务书要求标杆（Warmup 10 次 + 采样 100 次） | 任务书 §3.3 性能要求表 | **4/4 全部大幅超越**，性能优于标杆 44.8% ~ 62.0% |

### 1. 性能实测数据与标杆对比（Ascend 910B3 实测，10 Warmup + 100 迭代采样）

#### 官方基准场景（$\alpha=1.0, \beta=0.0$）

| 用例编号 | 矩阵规模与形态 | 操作模式 | 任务书标杆阈值 | 最新实测耗时 (中位数) | 多轮测试范围 ($\mu\text{s}$) | 达标余量（优于标杆） | 判定结论 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Case 1** | $1024 \times 1024, kl=16, ku=16$ | $trans = \text{OP\_N}$ | $\le 12.530\ \mu\text{s}$ | **$4.767\ \mu\text{s}$** | 4.576 ~ 4.834 | **+62.0%** | **PASS** |
| **Case 2** | $2048 \times 2048, kl=32, ku=32$ | $trans = \text{OP\_N}$ | $\le 15.540\ \mu\text{s}$ | **$6.992\ \mu\text{s}$** | 6.735 ~ 7.137 | **+55.0%** | **PASS** |
| **Case 3** | $1024 \times 1024, kl=16, ku=16$ | $trans = \text{OP\_T}$ | $\le 10.580\ \mu\text{s}$ | **$4.332\ \mu\text{s}$** | 3.985 ~ 4.359 | **+59.1%** | **PASS** |
| **Case 4** | $4096 \times 4096, kl=64, ku=64$ | $trans = \text{OP\_N}$ | $\le 21.980\ \mu\text{s}$ | **$12.129\ \mu\text{s}$** | 11.723 ~ 12.165 | **+44.8%** | **PASS** |

#### 全功能累加场景（$\alpha=1.0, \beta=1.0$）

| 用例编号 | 矩阵规模与形态 | 操作模式 | 任务书标杆阈值 | 实测耗时 (中位数) | 多轮测试范围 ($\mu\text{s}$) | 达标余量 | 判定结论 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Case 1** | $1024 \times 1024, kl=16, ku=16$ | $trans = \text{OP\_N}$ | $\le 12.530\ \mu\text{s}$ | **$4.941\ \mu\text{s}$** | 4.629 ~ 4.987 | **+60.6%** | **PASS** |
| **Case 2** | $2048 \times 2048, kl=32, ku=32$ | $trans = \text{OP\_N}$ | $\le 15.540\ \mu\text{s}$ | **$7.403\ \mu\text{s}$** | 7.112 ~ 7.422 | **+52.4%** | **PASS** |
| **Case 3** | $1024 \times 1024, kl=16, ku=16$ | $trans = \text{OP\_T}$ | $\le 10.580\ \mu\text{s}$ | **$4.485\ \mu\text{s}$** | 4.124 ~ 4.506 | **+57.6%** | **PASS** |
| **Case 4** | $4096 \times 4096, kl=64, ku=64$ | $trans = \text{OP\_N}$ | $\le 21.980\ \mu\text{s}$ | **$12.401\ \mu\text{s}$** | 11.881 ~ 12.487 | **+43.6%** | **PASS** |

### 2. 精度全量测试结果

测试工程结合 GoogleTest CSV 驱动与全要素双精度回归：
1. **原生 GTest 用例**：**1201 / 1201 全部 PASS**（包含 1200 条 CSV 用例及空句柄边界拦截）；
2. **扩展严格回归**：**2496 / 2496 全部 PASS**（共测试 10,082,016 个输出元素，每用例重复 3 次验证稳定一致性，最大绝对误差仅 **$5.80\times 10^{-6}$**）；
3. **极端特征覆盖**：
   - 包含带外内存填充 NaN/Inf 的严苛隔离验证；
   - 包含 $\beta=0$ 场景下旧 $y$ 为脏内存/NaN 的免读覆盖验证；
   - 包含正负步长（$\pm 1, \pm 2, \pm 3$）、单侧带状（$kl=0$ 或 $ku=0$）、非对称带、非对齐大尺寸（如 $m=2917, 8193$）的全面验证。

---

## 兼容性分析

1. **接口兼容性**：完全对齐 NVIDIA cuBLAS `cublasSgbmv` 与 Netlib CBLAS，入参与出参语义严格契合公共声明 `include/cann_ops_blas.h`；
2. **架构隔离性**：实现代码完整封装于 `blas/gbmv/arch22/`，与 GPU 风格架构 `arch35` 严格隔离解耦；
3. **标杆兼容性**：测试 Golden 基于权威 Netlib CBLAS，消除自研参考实现的歧义。

---

## 代码与复现入口

- **待验收代码仓**：https://gitcode.com/qq_50236812/ops-blas
- **待验收分支**：`feat-aclblasSgbmv-arch22`
- **算子核心源码**：`blas/gbmv/arch22/`
- **单测工程目录**：`test/gbmv/sgbmv/arch22/`
- **复现构建与执行命令**：
  ```bash
  # 1. 激活 CANN Toolkit 环境变量
  source /usr/local/Ascend/ascend-toolkit/set_env.sh

  # 2. 编译算子与单测工程 (指定昇腾 910B3 芯片)
  bash build.sh --soc=ascend910b3 --ops=sgbmv

  # 3. 设置动态库路径并运行全量单测
  export LD_LIBRARY_PATH=$(pwd)/build/lib:$(pwd)/out/lib64:$LD_LIBRARY_PATH
  ./build/test/gbmv/sgbmv/sgbmv_test

  # 4. 运行性能基准测试
  ./bench_sgbmv
  ```

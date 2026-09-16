# aclblasSgbmv 算子设计文档

| 项目 | 内容 |
|---|---|
| **任务名称** | 算子实操工坊-北京-aclblasSgbmv算子开发(A2/A3) |
| **taskId** | 57fcc5dad709456caae4b2b6d6003371 |
| **任务链接** | https://www.hiascend.com/activities/task-center/details/57fcc5dad709456caae4b2b6d6003371?menu=trends |
| **任务分类** | CANN 算子开发 |
| **开发者账号** | qq_50236812 |
| **适配硬件** | Atlas A2 / Atlas A3 系列产品（架构目录 arch22） |
| **CANN 版本** | CANN 9.1.0 |
| **涉及仓库** | `cann/ops-blas`（`blas/gbmv/arch22/`、`include/cann_ops_blas.h`、`test/gbmv/sgbmv/arch22/`） |
| **待验收代码仓** | https://gitcode.com/qq_50236812/ops-blas |
| **待验收分支** | `feat-aclblasSgbmv-arch22` |

---

# 需求背景（required）

## 需求来源

昇腾任务中心“算子实操工坊-北京-aclblasSgbmv算子开发(A2/A3)”。面向 Atlas A2/A3 系列产品（Ascend 910B，arch22），使用 Ascend C 编程语言开发单精度实数（FLOAT32）通用带状矩阵与向量乘法算子 `aclblasSgbmv`。通过 ops-blas 开源仓工程框架，提供基于句柄（Handle）的 Kernel 直调接口，完成算子设计、高性能实现以及严苛的精度和性能自测全流程。

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

1. **接口规范性**：严格契合 `include/cann_ops_blas.h` 声明，禁止定义私有平行接口；
2. **入参全防御**：对指针空值、负维度、非零步长、合法带宽高宽比、枚举值域进行完整校验；
3. **特例与快速返回**：支持 $m=0$ 或 $n=0$ 的合法 no-op；$\alpha=0$ 时的纯缩放或恒等更新；$\beta=0$ 时覆盖输出；
4. **多核负载均衡**：充分利用 Atlas A2 的 40 个 AIV 核，以 128 字节对齐切分输出行，根除多核写伪共享（False Sharing）；
5. **内层流式连续访存**：突破带状矩阵传统行遍历跨步性能瓶颈，采用列主序带状流式累加 + 8 路循环展开，消除 RAW 指令依赖；
6. **自验可复现**：覆盖 1000 条 CSV 精度用例与 4 组指定性能标杆，精度达成 100%，性能领先任务书要求 17%~60%。

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

对于每一列 $j$，其有效行元素在内存中物理地址严格连续排列，这一几何特征构成本算子性能优化的核心基石。

```
                       Column j
            +-------------------------------+
            |               :               |
[ku + i - j]    | A(i, j)                       |  <-- 物理连续存储段
[ku + i+1 - j]  | A(i+1, j)                     |
            |               :               |
            +-------------------------------+
```

### 支持数据类型与形状

- **数据类型**：FLOAT32（输入输出矩阵与向量、标量 $\alpha, \beta$ 均为单精度实数）；
- **内存分布**：
  - 矩阵 $A$、向量 $x$、向量 $y$ 位于 Device 内存；
  - 标量 $\alpha$、标量 $\beta$ 为 Host 内存指针；
  - 句柄 `handle` 位于 Host 内存，包含下发计算任务的 `aclrtStream`。

---

## 算子实现

### 实现方案

```
aclblasSgbmv (Host 入口)
   │
   ├─ 1. 入参完整性校验 (Handle, Trans, m, n, kl, ku, lda, incx, incy, 指针)
   │
   ├─ 2. 快速返回分支判断 (m=0/n=0 -> Success; alpha=0 -> Host/Kernel 缩放)
   │
   ├─ 3. 40核无伪共享负载均衡划分 (128B Cacheline 对齐，确定各核处理行区间)
   │
   └─ 4. Stream 异步启动 Kernel <<<gridDim, nullptr, stream>>>
         │
         └─ sgbmv_kernel (Device / AIV)
               │
               ├─ 连续模式 (incx=1, incy=1) -> 列主序流式连续累加 + 8路展开 + Zero Skip
               └─ 非连续模式 (incx!=1 或 incy!=1) -> 步长映射通用计算
```

### 3.2.1 Host 侧设计

#### 1. 参数校验与防御性编程
Host 侧遵循工业级 BLAS 标准实施严密校验：
- `handle == nullptr`：返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
- `trans` 不属于 `ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C`：返回 `ACLBLAS_STATUS_INVALID_ENUM`；
- 维度边界：$m < 0$ 或 $n < 0$ 或 $kl < 0$ 或 $ku < 0$：返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 带宽越界：$m > 0$ 时 $kl \ge m$ 或 $n > 0$ 时 $ku \ge n$：返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 前导维：$lda < kl + ku + 1$：返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 步长：$incx == 0$ 或 $incy == 0$：返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- 指针为空判断：$\alpha == \text{nullptr}$ 或 $\beta == \text{nullptr}$ 返回 `INVALID_VALUE`；在 $m > 0$ 且 $n > 0$ 时，若 $y == \text{nullptr}$ 或（$\alpha \ne 0$ 时 $A == \text{nullptr}$ 或 $x == \text{nullptr}$）均返回 `INVALID_VALUE`。

#### 2. 特例与快速返回（Quick Return）
- **空矩阵 No-Op**：若 $m = 0$ 或 $n = 0$，算子无需执行任何计算，直接返回 `ACLBLAS_STATUS_SUCCESS`；
- **$\alpha = 0$ 纯缩放**：
  - 若 $\beta = 1.0f$，输出向量 $y$ 保持不变，直接返回 `ACLBLAS_STATUS_SUCCESS`；
  - 若 $\beta == 0.0f$ 或其他值，仅执行 $y = \beta \cdot y$，无需读取与解引用矩阵 $A$ 和向量 $x$。

#### 3. 分核策略：40 核负载均衡与 128B 无伪共享对齐切分
Atlas 800I A2 具备 40 个 AIV 核，全局内存与 L2 Cache 间按 128 字节（32 个 FLOAT32 元素）为单位调度。若多个核心并发写入 $y$ 且核心分界点不是 32 的整数倍，相邻核心将写入同一 Cacheline，引发总线锁冲突与 Cache 乒乓颠簸（False Sharing）。

切分算法将输出维度 $outDim$ 切分为 32 元素的整对齐块，每个核分发整数个对齐块：
```cpp
uint32_t numChunks = outDim / 32U;
uint32_t useNumBlocks = (incy == 1 && numChunks >= 2U) ? std::min(numChunks, aivCoreNum) : 1U;
uint32_t chunksPerCore = numChunks / useNumBlocks;
uint32_t remChunks = numChunks % useNumBlocks;

uint32_t startChunk = (blkIdx < remChunks)
    ? blkIdx * (chunksPerCore + 1U)
    : remChunks * (chunksPerCore + 1U) + (blkIdx - remChunks) * chunksPerCore;
uint32_t myChunks = (blkIdx < remChunks) ? (chunksPerCore + 1U) : chunksPerCore;

uint32_t startRow = startChunk * 32U;
uint32_t endRow = (blkIdx == useNumBlocks - 1U) ? outDim : (startRow + myChunks * 32U);
```
**数学证明**：前 $useNumBlocks - 1$ 个核心的处理区间 $[startRow, endRow)$ 均满足 $startRow \equiv 0 \pmod{32}$ 且 $endRow \equiv 0 \pmod{32}$；最后一个核心的起始点同样对齐 32，尾部多余行包含在其内部。从而彻底消除多核写冲突。

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

#### 1. 计算流程（Init -> CopyIn -> Compute -> CopyOut）
Kernel 执行分为三个紧密衔接的阶段：
1. **Init**：获取 block_idx，计算本核心专属的行区间 $[startRow, endRow)$；
2. **Process**：
   - 读取 $\beta$ 对应原 $y$ 值（若 $\beta \ne 0$ 则初始化为 $\beta \cdot y$，若 $\beta = 0$ 则直接置 0）；
   - 按列遍历有效 $x_j$，将矩阵 $A$ 该列对应行切片连续流式加载并乘加累加；
3. **CopyOut**：计算完成后的结果写回全局内存 $y$。

#### 2. 列主序带状流式连续累加核心算法（突破性能瓶颈）

传统带状矩阵乘法常按输出行 $i$ 循环，由于列主序存储下同一行相邻列的物理地址跨度为 $lda - 1$，会引发严重跨步访存与标量累加的 RAW 数据依赖延迟。

本算子实施**计算拓扑反转**：外层遍历列 $j$，内层针对 $A[:, j]$ 的物理连续段进行流式向量化累加：
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

#### 3. 8 路循环展开与寄存器分配
在内层流式累加中，采用 8 路展开彻底消除流水线气泡：
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
- **零依赖展开**：8 个并发更新对应不同行，相互之间无 RAW 依赖；
- **严格数值保真**：按 $j$ 递增顺序完成累加，与 Netlib cblas 结合律顺序完全一致；
- **Zero Skip 优化**：检测到 $x_j = 0.0f$ 时直接跳过整列的加载与累加。

---

## 支持硬件

| 产品 / 芯片型号 | 支持情况 | 已验证环境 |
|---|:---:|---|
| Atlas 800I A2（910B3） | 支持 | CANN 9.1.0，40 个 AIV 核实机验证通过 |
| Atlas 800T A2（910B4） | 支持 | 架构兼容（arch22） |
| Atlas 800I/T A3 系列 | 支持 | 统一 DaVinci arch22 矢量指令集支持 |

---

## 算子约束限制

1. **矩阵存储**：仅支持单精度实数（FLOAT32）列主序带状格式，$lda \ge kl + ku + 1$；
2. **步长要求**：$incx \ne 0$ 且 $incy \ne 0$，支持负步长；
3. **内存边界**：调用方必须保证 $A$、$x$、$y$ 具备足够的 Device 显存物理跨度；
4. **异步同步**：算子在句柄绑定 stream 上异步执行，Host 读回输出前须显式同步 stream。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准项 | 标准要求 | 标准来源 | 达成情况 |
|---|---|---|---|
| **精度标准** | FLOAT32：rtol $\le 2^{-10}$ ($9.77\times 10^{-4}$)，atol $\le 2^{-16}$ ($1.53\times 10^{-5}$)，通过率 $\ge 99\%$，最大绝对误差 $\le 10^{-2}$ 或 $32\times\text{ULP}$ | 昇腾生态算子开源精度标准 | **达成率 100.0%**（1000/1000 用例全部通过，零失败） |
| **性能标准** | 4 组基准用例平均单次耗时（Avg time，$\mu\text{s}$）不高于任务书要求标杆（Warmup 10 次 + 采样 100 次） | 任务书 §3.3 性能要求表 | **4/4 全部达标**，性能优于标杆 17.5% ~ 59.8% |

### 1. 性能实测数据与标杆对比（Atlas 800I A2，10 Warmup + 100 有效采样）

| 用例编号 | 矩阵规模与形态 | 操作模式 | 任务书标杆 ($\mu\text{s}$) | 实测平均耗时 ($\mu\text{s}$) | 性能优势（达标余量） | 判定结论 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Case 1** | $m=1024, n=1024, kl=16, ku=16$ | $trans = \text{OP\_N}$ | $\le 12.53$ | **5.03** | **+59.8%** | **PASS** |
| **Case 2** | $m=2048, n=2048, kl=32, ku=32$ | $trans = \text{OP\_N}$ | $\le 15.54$ | **6.94** | **+55.3%** | **PASS** |
| **Case 3** | $m=1024, n=1024, kl=16, ku=16$ | $trans = \text{OP\_T}$ | $\le 10.58$ | **8.73** | **+17.5%** | **PASS** |
| **Case 4** | $m=4096, n=4096, kl=64, ku=64$ | $trans = \text{OP\_N}$ | $\le 21.98$ | **10.30** | **+53.1%** | **PASS** |

### 2. 精度全量测试结果

测试工程基于 GoogleTest + CSV 驱动，比对 Golden 由 Netlib CBLAS（`cblas_sgbmv`）单标杆生成：
- **总测试用例数**：1000 项；
- **测试通过数**：**1000 项通过，0 项失败**；
- **特殊值覆盖**：包含极值、交替正负、高维非对齐、零维 No-Op 等边界用例，绝对误差均为 0.0000e+00。

---

## 兼容性分析

1. **接口兼容性**：函数原型及参数顺序与 cuBLAS `cublasSgbmv` 及 ops-blas `cann_ops_blas.h` 完全一致，支持与其他平台共用声明；
2. **架构兼容性**：代码独立存放于 `blas/gbmv/arch22/`，与现存 `arch35` 架构并存互不干扰；
3. **Golden 兼容性**：精度比对基准完全采用工业标准 Netlib CBLAS，确保数学逻辑完全无歧义。

---

## 代码与复现入口

- **待验收代码仓**：https://gitcode.com/qq_50236812/ops-blas
- **待验收分支**：`feat-aclblasSgbmv-arch22`
- **算子代码路径**：`blas/gbmv/arch22/`
- **测试代码路径**：`test/gbmv/sgbmv/arch22/`
- **复现编译与测试命令**：
  ```bash
  # 1. 编译算子与测试工程（Atlas 800I A2 / ascend910b3）
  bash build.sh --soc=ascend910b3 --ops=gbmv

  # 2. 运行精度测试与性能测试
  cd build/test/gbmv/sgbmv/arch22
  ./test_aclblas_sgbmv
  ```

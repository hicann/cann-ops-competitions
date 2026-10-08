# aclblasCgbmv 算子设计文档（Ascend C / arch22）

> 任务名称：9 月社区任务 — `aclblasCgbmv` 算子开发（A2/A3）  
> 任务书：[aclblasCgbmv_A2A3_task_doc.md](../../../../../../Tasks/aclblasCgbmv_A2A3/requirements/aclblasCgbmv_A2A3_task_doc.md)  
> 交付模板：[cann-ops-competitions design_template.md](../../../../resources/design_template.md)  
> 目标仓库：[cann/ops-blas](https://gitcode.com/cann/ops-blas)，落盘路径 `blas/gbmv/arch22/` 与 `test/gbmv/cgbmv/arch22/`  
> 提交路径：`04_tasks/01_community-task-2026/tasklist/09-aclblasCgbmv-A2A3/aihos/docs/design.md`  

---

# 需求背景（required）

## 需求来源

昇腾开源社区 2026 年 9 月任务：在昇腾 NPU（Atlas A2/A3 系列产品，性能测试对标设备 Atlas 800T A2 (910B3)，CANN 版本 9.1.0）上，基于 Ascend C 编程语言开发单精度复数（COMPLEX64）一般带状矩阵向量乘算子 `aclblasCgbmv`，执行：
$$y = \alpha \cdot \operatorname{op}(A) \cdot x + \beta \cdot y$$
算子完成设计、开发、测试全流程工作，验收通过后合入昇腾算子开源仓 [cann/ops-blas](https://gitcode.com/cann/ops-blas)。

## 背景介绍

### 算子现状分析

在 `ops-blas` 仓中：
1. `blas/gbmv/` 目录下**当前仅有 arch35 实现**（`sgbmv_kernel.cpp` / `sgbmv_host.cpp` / `sgbmv_tiling_data.h`），面向 Ascend 950PR/950DT，且仅支持实数单精度 FLOAT32。
2. arch35 版本采用 SIMT 编程模型（依赖 `simt_api/asc_simt.h`、`__simt_vf__`、`threadIdx` 等），该模型在 Atlas A2/A3（架构目录 `arch22`）上不可用。经全仓检索确认，`blas/*/arch22/` 下无任何算子使用 SIMT API，arch22 统一采用经典的 Ascend C SIMD 向量核模型（AIV 上的 `TPipe` / `TQue` / `LocalTensor` / 向量指令）。
3. 因此本任务是**在 arch22 架构下全新开发首个复数一般带状矩阵向量乘算子**，补齐 Atlas A2/A3 产品线支持能力。需采用 Ascend C 向量计算范式重新设计，不可直接搬用 arch35 的 SIMT 内核，但复用仓内标准的 Host 入参校验、Tiling 传递与 GTest 测试框架结构。

### 现有工程基础与代码复用边界

| 现有参考代码 | 本任务参考复用内容 | 处理边界与独立设计 |
|---|---|---|
| `gbmv/arch35/sgbmv_host.cpp` | handle 校验、零维处理、基础参数校验、按值 tiling 和 stream 传递模式 | 新增任务要求的带宽上界检查（$kl \le m-1, ku \le n-1$）；不沿用 SIMT 分核参数 |
| `ger/arch22/sger_host.cpp`、`sger_kernel.cpp` | arch22 架构下的按值传递 POD Tiling 启动与异步直接发射范式 | 复用工程模式与启动包装，不照搬其参数限制 |
| `gemv/arch22`、`gemv_batched/arch22` | complex64 实虚解交错、向量四路乘加、规约处理 | 重新设计带状紧凑存储索引和 UB 分块；不照搬其固定核数、每次 GM 工作区分配及内部全局同步 |
| `test/gbmv/sgbmv` | 家族测试目录结构、CBLAS 调用模式、GTest 组织规范 | 建立 Cgbmv 专用 21 列参数解析与复数带状填充器 |
| `test/frame/verify.h` | MixedToleranceStrategy、ULP 计算、特殊值比较实现 | 在 Cgbmv 测试局部显式配置 VerifyConfig，不修改全仓共享默认值 |

### 对标接口

对标 NVIDIA cuBLAS `cublasCgbmv`（底层语义参考 Netlib BLAS `cgbmv` 复数单精度实现）：
```cpp
aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, int kl, int ku,
    const aclblasComplex *alpha,
    const aclblasComplex *A, int lda,
    const aclblasComplex *x, int incx,
    const aclblasComplex *beta,
    aclblasComplex *y, int incy);
```
参数序列、属性定义与 cuBLAS 一一对应，无需额外映射。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现单精度复数一般带状矩阵-向量乘法算子 `aclblasCgbmv`，数据类型为 COMPLEX64（实部 FP32 + 虚部 FP32，单元素 8 字节）。支持三种变换模式（`ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C`）、任意合法带状存储（`lda ≥ kl+ku+1`）、正负步长（`incx ≠ 0, incy ≠ 0`），以及 `m=0` 或 `n=0` 的合法零维 no-op 处理。

## 需求拆解

| 序号 | 需求维度 | 具体要求与设计落地 |
|---|---|---|
| **R01** | 数学语义与模式覆盖 | 支持 $y = \alpha \cdot \operatorname{op}(A) \cdot x + \beta \cdot y$。覆盖 `N`（不转置）、`T`（转置）、`C`（共轭转置）。C 模式仅对矩阵 $A$ 取复共轭。 |
| **R02** | 紧凑带状索引映射 | 严格实现 $lda \times n$ 列主序带状打包存储的坐标转换；处理 $x$、$y$ 正负步长（负步长末端反向寻址）。 |
| **R03** | 边界与短路径优化 | 零维（$m=0$ 或 $n=0$）合法快速返回；$\alpha=0, \beta=1$ 纯无损直返；$\beta=0$ 时不依赖原 $y$（支持原 $y$ 为 NaN 未初始化）；$\alpha=0, \beta \neq 1$ 走极速 ScaleY 向量缩放。 |
| **R04** | 精度标准达标 | 实部与虚部独立按 FLOAT32 生态混合容差标准判定：$rtol = atol = 2^{-13}$，匹配率 $\ge 0.99$，逐点最大绝对误差 $\le \max(0.01, 32 \times \text{ULP})$。 |
| **R05** | 性能指标达标 | 任务书 §3.3 规定的 5 组典型性能用例（10.665 μs ~ 52.487 μs）全部达标；全量 200 条性能用例平均单次耗时均满足 $\le gpu\_ms / 0.8$。 |
| **R06** | 零额外内存开销 | 算子实现保持 **新增 GM Workspace = 0 字节**，纯在 AIV UB 内部完成累加规约与临时运算，不依赖外部句柄 workspace。 |
| **R07** | 接口与工程合规 | 接口声明合入 `include/cann_ops_blas.h`；Kernel 直调代码放入 `blas/gbmv/arch22/`；测试代码与固定 CSV 放入 `test/gbmv/cgbmv/arch22/`。 |

## 已确认的执行口径

为确保方案严谨且与工程基线完全统一，本方案严格固化以下经评审确认的口径：

| 项目 | 本方案执行口径 | 依据与说明 |
|---|---|---|
| **零维处理** | `m=0` 或 `n=0` 为 no-op，返回 `ACLBLAS_STATUS_SUCCESS`，不计算、不改写 y、不发射 kernel | cuBLAS 规范与任务书 §2.4 一致；优先于带宽上界与指针判空执行，彻底消除数学冲突 |
| **非零维度带宽** | 严格实现 `0 ≤ kl ≤ m-1`、`0 ≤ ku ≤ n-1`；越界返回 `ACLBLAS_STATUS_INVALID_VALUE` | 严格满足任务书 §2.4 要求；现有 CSV 无超界用例，不影响现有用例通过率 |
| **输入分布与标量** | 以现有 CSV 为准，不补正态分布、纯虚标量、beta=-1 等测试 | `gen_csv.py` 坦承底层工程暂未支持高斯生成器；CI 评测以真实 CSV 用例为准 |
| **测试集合** | 严格执行现有 `cgbmv_test.csv` 的 1200 行（1000 精度 + 200 性能） | 覆盖 L0~L6、EX、PF 全部分类，不人为扩缩集合 |
| **基线数据保护** | 使用已生成的 CSV，**坚决不重新运行 `gen_csv.py`** | 保护 `gpu_baseline.csv` 中已有的 200 条 GPU 实测数据不被空串抹除 |
| **精度判据** | 实部/虚部分别按 FLOAT32 混合容差判定：$rtol=atol=2^{-13}$，$\ge 0.99$，逐点 $\le \max(0.01, 32 \times \text{ULP})$ | 严格对齐生态标准与 [verify.h:L308](https://gitcode.com/cann/ops-blas/blob/master/test/frame/verify.h) 的底层实现 |
| **性能采样** | 5 次预热后，恰好取 10 次有效采样平均（`Task Duration(us)`） | 排除前 5 次 warmup，每次调用前恢复 $y_0$（确保 $\beta=1$ 数据一致性） |
| **硬件款型** | 面向 Atlas A2 / A3；正式性能对标硬件为 Atlas 800T A2 (910B3) | 自验证覆盖真实款型，不以封装方式推导指令性能完全一致 |

---

# 详细设计（required）

## 算子分析

### 数学公式

$$y = \alpha \cdot \operatorname{op}(A) \cdot x + \beta \cdot y$$

设复数展开：$A = A_r + i A_i$，$x = x_r + i x_i$，$\alpha = \alpha_r + i \alpha_i$，$\beta = \beta_r + i \beta_i$。
单点复数内积乘积项：
$$P_r = A_r \cdot x_r - A_i \cdot x_i, \quad P_i = A_r \cdot x_i + A_i \cdot x_r$$
累加和向量 $s = \sum P$。写回更新：
$$y_r \leftarrow \alpha_r s_r - \alpha_i s_i + \beta_r y_r - \beta_i y_i$$
$$y_i \leftarrow \alpha_r s_i + \alpha_i s_r + \beta_r y_i + \beta_i y_r$$
若为 `ACLBLAS_OP_C`（共轭转置），则在计算前令 $A_i \leftarrow -A_i$。

### 带状存储语义（0-based 映射）

一般带状矩阵 $A$ 采用列主序压缩打包存储于尺寸为 $lda \times n$ 的数组中：
* 逻辑矩阵元素 $A(i, j)$（$0 \le i < m, 0 \le j < n$）在带内的充要条件为：$-ku \le i - j \le kl$；
* 其在列主序物理存储数组中的一维元素偏移为：
  $$\text{bandRow}(i, j) = ku + i - j$$
  $$\text{Index}(i, j) = j \times lda + (ku + i - j) \quad (\text{单位：complex64 元素})$$
  *(即主对角线 $i = j$ 恒位于带状矩阵每列的第 $ku$ 行，0-based)*；
* 带外三角区域（左上角 $ku \times ku$ 与右下角 $kl \times kl$）属于无定义 padding，算子坚决不引用。

| trans 模式 | 输出逻辑元素 | 内部遍历与规约区间 | A 读取元素 |
|---|---|---|---|
| **N** | $y[i], 0 \le i < m$ | $j = \max(0, i-kl) \dots \min(n-1, i+ku)$ | $A(i, j)$ |
| **T** | $y[j], 0 \le j < n$ | $i = \max(0, j-ku) \dots \min(m-1, j+kl)$ | $A(i, j)$ |
| **C** | $y[j], 0 \le j < n$ | $i = \max(0, j-ku) \dots \min(m-1, j+kl)$ | $\operatorname{conj}(A(i, j))$ |

### 向量步长与物理定位

向量 $x$ 与 $y$ 的逻辑长度为 $L_x, L_y$：
* 当 `trans == N` 时：$L_x = n, L_y = m$；
* 当 `trans == T/C` 时：$L_x = m, L_y = n$。

步长 $inc$ 允许为负，物理起始偏移与元素地址公式（统一采用 64 位有符号整型计算，防溢出）：
$$\text{start}(L, inc) = \begin{cases} 0, & inc > 0 \\ (L - 1) \times (-inc), & inc < 0 \end{cases}$$
$$\text{offset}(k, L, inc) = \text{start}(L, inc) + k \times inc \quad (0 \le k < L)$$
$$\text{span}(L, inc) = \begin{cases} 1 + (L - 1) \times |inc|, & L > 0 \\ 0, & L = 0 \end{cases}$$

API 与 CBLAS 均接收物理缓冲区基址。负步长起点由 Kernel 内部根据公式定位，不在 API 入口前移指针，防止双重偏移。

---

## 算子实现

### 实现方案架构

```
+-----------------------------------------------------------------------------------+
| Host 侧 (cgbmv_host.cpp): 参数合法性校验 -> 快速路径识别 -> 多核 Tiling -> 异步发射  |
+-----------------------------------------------------------------------------------+
                                         | 按值传递 POD TilingData
                                         v
+-----------------------------------------------------------------------------------+
| Device Kernel 侧 (cgbmv_kernel.cpp & cgbmv_kernel_impl.h): AIV 向量核加速计算      |
|                                                                                   |
|  [路径 1: Host no-op]      m=0 或 n=0，或 alpha=0 且 beta=1 (不发射 kernel)         |
|  [路径 2: ScaleY]          alpha=0, beta!=1 (不访问 A/x，对 y 执行向量纯缩放/清零) |
|  [路径 3: Whole-Band Cache] A 存储 <= 64KB (极速整块吞入 UB，针对 Case 1 极限冲刺) |
|  [路径 4: BandN]           trans=N, alpha!=0 (输出行块归属单核，按列分块连续累加)  |
|  [路径 5: BandTC]          trans=T/C, alpha!=0 (输出列归属单核，列内连续向量点积)  |
|  [路径 6: 通用步长写回]    incy!=1 或负步长 (单核安全处理，精确只写有效复数)        |
+-----------------------------------------------------------------------------------+
```

#### 3.2.1 Host 侧设计

##### 1. 参数校验与快速返回（Quick Return）顺序
严格遵照口径和 cuBLAS 语义，校验次序设计如下：
1. **句柄校验**：`handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. **基础维度校验**：`m < 0 || n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
3. **零维 No-op 快速直返**：`if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;`（此时不发射 kernel，不引用后续参数与指针）；
4. **变换枚举校验**：`trans` 不在 `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
5. **带宽上下界校验**：`kl < 0 || kl > m - 1 || ku < 0 || ku > n - 1` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. **存储步长校验**：`lda < kl + ku + 1 || incx == 0 || incy == 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
7. **标量指针校验**：`alpha == nullptr || beta == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
8. **数据指针校验（依参与计算状态判定）**：
   * 若 $\alpha = 0, \beta = 1$：退化为 $y = y$，直接返回 `ACLBLAS_STATUS_SUCCESS`；
   * 若 $\alpha \neq 0$：$A$ 和 $x$ 必须非空（`A == nullptr || x == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`）；
   * $y$ 必须非空（`y == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`，即使 $\beta=0$ 仍需写回，故 $y$ 不能为空）。

##### 2. Tiling 与多核均分策略
* **核数获取**：通过 `GetAivCoreCount()` 获取当前芯片可用的 AIV 核心数 $N_{\text{max}}$（910B3 通常为 20~24 核）；
* **小尺寸动态限核**：为避免小矩阵调度过多空核带来发射开销，设计自适应分核规则：
  $$\text{neededCores} = \left\lceil \frac{dim_{\text{out}}}{16} \right\rceil, \quad \text{useCoreNum} = \max(1, \min(N_{\text{max}}, \text{neededCores}))$$
  （其中 $dim_{\text{out}}$ 为输出维度：N 模式为 $m$，T/C 模式为 $n$）；
* **任务均分**：各核分配的处理元素数 $rowsPerBlock = \lceil dim_{\text{out}} / useCoreNum \rceil$，核 $k$ 负责输出区间 $[k \times rowsPerBlock, \min(dim_{\text{out}}, (k+1) \times rowsPerBlock))$，各核写回区间天然互斥，无需原子操作与跨核同步。

##### 3. TilingData 规划
Tiling 结构体声明为纯 POD 类型（`CgbmvTilingData`），按值拷贝传入 Kernel：
```cpp
struct CgbmvTilingData {
    uint32_t m, n, kl, ku, lda;
    uint32_t outDim, inDim;
    int64_t incx, incy;
    int64_t xStart, yStart;
    float alphaReal, alphaImag;
    float betaReal, betaImag;
    uint32_t trans;          // 0: N, 1: T, 2: C
    uint32_t pathType;       // 0: ScaleY, 1: WholeBandCache, 2: Standard
    uint32_t useCoreNum;
    uint32_t rowsPerBlock;
};
```

##### 4. 异步与资源生命周期
* 新增 GM Workspace 严格为 **0 字节**；
* Host 侧在 `handle->stream` 上以非阻塞方式异步 launch kernel，不在入口执行 stream 同步，保证高性能异步队列流水。

---

#### 3.2.2 Kernel 侧设计

##### 1. 复数实虚拆分与向量化计算（使用硬件级原语）
数据在 GM 中以交错格式存储：$[r_0, i_0, r_1, i_1, \dots]$。在 UB 中采用 Ascend C 专用指令进行高效拆分与乘加：
* **解交错**：采用 `GatherMask` 向量指令按 Pattern 1 和 2 快速将连续数据解离到独立实部、虚部 Tensor：
  ```cpp
  GatherMask(matRealLocal, inALocal, 1, false, 0, {1, repeatNum, 8, 8}, rsvdCnt); // 提取实部
  GatherMask(matImagLocal, inALocal, 2, false, 0, {1, repeatNum, 8, 8}, rsvdCnt); // 提取虚部
  ```
* **四路实数向量乘加**：
  设 $A = A_r + i A_i$，$x = x_r + i x_i$，若为 `OP_C` 则 $A_i \leftarrow -A_i$：
  $$\text{Prod}_r = A_r \cdot x_r - A_i \cdot x_i, \quad \text{Prod}_i = A_r \cdot x_i + A_i \cdot x_r$$
  利用向量 `Mul`、`Mul`、`Add`、`Sub` 指令打满 AIV 向量流水。
* **复数重组写回**：利用本地构建的交错索引通过 `Gather` 指令将实虚部分打包回交错复数写回 GM。

##### 2. N 模式（非转置）：按输出行块分配、逐列累加
* **算法组织**：每个核负责输出行区间 $[r_0, r_1)$，在 UB 中开辟长度为 $r_1 - r_0$ 的实/虚累加器。
* **外层列循环**：遍历与该行区间相交的有效列 $j \in [\max(0, r_0 - kl), \min(n, r_1 + ku))$。
* **列内内存连续性与 32B 向下对齐**：
  列 $j$ 与行区间 $[r_0, r_1)$ 的相交有效行范围为 $[lo, hi) = [\max(r_0, j-ku), \min(r_1, j+kl+1))$。该区间在带状矩阵 $A$ 中是**连续存储**的，物理偏移为 $offset = j \cdot lda + ku + lo - j$。
  * **对齐处理**：将源地址向下取整到 32 字节边界（4 个复数）：$aligned\_offset = \lfloor offset / 4 \rfloor \times 4$；
  * **前缀跳过**：`DataCopy` 搬运进 UB 后，向量计算偏移跳过 $offset - aligned\_offset$ 个前缀无效元素，确保 100% 内存安全对齐。

##### 3. T / C 模式（转置与共轭转置）：按列分配、连续点积规约
* **算法组织**：每个核负责输出若干个完整列 $j$。由于 $A$ 是列主序存储，第 $j$ 列的有效行范围 $[i_0, i_1) = [\max(0, j-ku), \min(m, j+kl+1))$ 在 GM 中是**完全物理连续的一整段内存**！
* **连续点积**：计算等价于连续复数向量内积。按 $K_{\text{tile}} = 512$ 分块，连续搬入 $A(:, j)$ 与 $x$，实虚部分别执行高效向量累加，块间使用 `ReduceSum` 汇聚为标量部分和。
* **零核间竞争**：各核负责不同列的输出 $y[j]$，彼此绝对独立，并行度极高。

##### 4. 极致性能突破：小矩阵全带缓存路径（Whole-Band Cache Path）
为攻克 Case 1（$m=256, n=256, kl=16, ku=16$，目标 **10.665 μs**）的极限耗时：
* 当 $lda \times n \times 8 \le 64 \text{ KB}$ 时，各核在进入循环前直接一次性将所需的带状数据大块读入 UB；
* 在 UB 内做全向量片上计算，彻底消除 N 模式下 256 次微型小数据包 DMA 发射的流水打断与排队开销，确保 10 μs 级严苛达标。

##### 5. 步长与写回安全隔离
* **纯连续高性能路径**（$incx = 1, incy = 1$）：性能用例 100% 命中，采用 32 字节对齐（4 个复数组）连续搬运并发写回；
* **非连续通用路径**（$|incx| > 1$ 或 $|incy| > 1$ 或负步长）：首版由单核分块串行处理，精确寻址只写回有效复数，严格保持间隙数据不变，彻底消除多核写冲突。

##### 6. UB 显式空间预算与 Double Buffer 双缓冲设计
AIV 单核可用 UB 为 192 KiB，设计显式单核内存预算控制在 96~128 KiB 内，预留足额对齐和计算 scratch 空间：

| 缓冲区名称 | 分配规划与计算依据 | 估算大小 |
|---|---|---:|
| **$A$ / $x$ 交错输入双缓冲** | 针对 $K_{\text{tile}} = 512$ 复数，两路输入各自乒乓双缓冲：$2 \times 2 \times 8 \times 512 \text{ B}$ | 32 KiB |
| **$A$ / $x$ 实虚拆分工作区** | 实部与虚部分离各 4 路：$4 \times 4 \times 512 \text{ B}$ | 16 KiB |
| **复数乘积与临时向量** | 四路乘加中间临时结果向量：$4 \times 4 \times 512 \text{ B}$ | 16 KiB |
| **规约 scratch / 索引 / 输出暂存** | 预留给 Ascend C ReduceSum 与输出组装的局部缓冲区 | 32 KiB |
| **单核 UB 显式预算合计** | 留出约 96 KiB 安全余量，彻底规避 UB 溢出风险 | **96 KiB** |

配合 Ascend C `TQue` / `SetFlag` / `WaitFlag` 实现 MTE2 搬入与 Vector 计算的高效乒乓流水覆盖。

##### 7. 特殊值与边界防污染保护
* **$\beta = 0$ 保护**：直接跳过原 $y$ 的读取，新值直接写回，支持原 $y$ 未初始化为 NaN 的场景；
* **$\alpha = 0$ 保护**：不访问 $A$ 与 $x$，直接调用 ScaleY 逻辑；
* **空规约区间防污染**：对于带内无有效元素的行或列，代码逻辑判定有效长度为 0 时直接令内积项为数学 0，**严禁执行 $\alpha \times 0$ 的浮点运算**，避免 $\alpha = \text{Inf}$ 时产生 $\text{NaN}$。

---

## 支持硬件

| 支持的芯片版本 | 是否涉及 | 验证说明 |
|---|:---:|---|
| Atlas 800T A2（910B3 训练卡） | **√** | 性能测试对标设备与主验证平台 |
| Atlas 800I A2（推理卡） | **√** | 共享 arch22 架构指令集 |
| Atlas A3 系列产品（910_93） | **√** | 共享 arch22 架构能力，自动适配核心数 |

---

## 算子约束限制

1. **维度与参数范围**：$m \ge 0, n \ge 0$；$0 \le kl \le m-1, 0 \le ku \le n-1$；$lda \ge kl+ku+1$；$incx \neq 0, incy \neq 0$。
2. **指针重叠**：输出向量 $y$ 与输入矩阵 $A$、输入向量 $x$ 不允许存在任何内存地址重叠（No In-place aliasing）。
3. **前导维与步长限制**：超出标准 BLAS 定义的非常规非连续内存不在本算子支持范围内。

---

# 可维可测分析（required）

## 精度标准 / 性能标准 / 内存标准

### 1. 精度标准
遵循最新生态算子开源精度标准（COMPLEX64 档，实部与虚部分别按单精度 FLOAT32 独立判定）：
* **逐元素混合容差门限**：
  $$|actual - golden| \le atol + rtol \times |golden| \quad (rtol = atol = 2^{-13} \approx 1.2207 \times 10^{-4})$$
  要求满足条件的元素比例：`matched_ratio ≥ 0.99`；
* **全张量逐点硬上限（Hard Limit）**：
  每一个有限分量必须同时满足：
  $$|actual - golden| \le \max(10^{-2}, 32 \times \text{ULP}(|golden|))$$
* **测试工程局部配置策略**：
  在 `cgbmv_test.cpp` 内部局部配置 `VerifyConfig`，直接传入 $2^{-13}$ 与 $10^{-2}$，**零修改保持公共 `test/frame/verify.h` 完好无损**，彻底阻断对全仓其他 50+ 算子的意外干扰：
  ```cpp
  VerifyConfig cfg{};
  cfg.mode = PrecisionMode::MIXED_TOLERANCE;
  cfg.mixedRtol = 0.0001220703125;             // 2^-13
  cfg.mixedAtol = 0.0001220703125;             // 2^-13
  cfg.mixedRequiredMatchedRatio = 0.99;
  cfg.mixedMaxAbsErrorLimit = 1e-2;             // 内部自动配合 32.0 * getUlpsAt 取 max
  cfg.mixedMantissaBits = 23;
  cfg.mixedEmin = -126;
  ```

---

### 2. 性能标准

#### (1) 理论算力与访存建模
有效非零元素数由各列有效区间长度求和：
$$nnz = \sum_{j=0}^{n-1} \max\Big(0, \min(m-1, j+kl) - \max(0, j-ku) + 1\Big)$$
一般复数乘加约需 $8 \times nnz$ 个实数浮点操作，矩阵 $A$ 的有效读取量约为 $8 \times nnz$ 字节。

#### (2) 5 组关键性能用例与数据量
任务书 §3.3 规定的 5 组关键性能指标必须绝对达标：

| Case | trans | m=n | kl/ku | alpha/beta | 任务书目标耗时 | 有效 A 元素数 | 有效 A 数据字节 |
|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|
| **TC_PF_1001** | N | 256 | 16/16 | 1/0 | **10.665 μs** | 8,176 | 65,408 B |
| **TC_PF_1002** | N | 512 | 32/32 | 1/1 | **17.328 μs** | 32,224 | 257,792 B |
| **TC_PF_1003** | T | 1024 | 64/64 | 1/0 | **25.875 μs** | 127,936 | 1,023,488 B |
| **TC_PF_1004** | N | 2048 | 128/128 | 1/0 | **30.640 μs** | 509,824 | 4,078,592 B |
| **TC_PF_1005** | C | 4096 | 64/256 | 1/0 | **52.487 μs** | 1,279,840 | 10,238,720 B |

#### (3) 全量性能用例达标条件与采样规范
* 全量 200 条性能用例平均单次耗时必须满足：
  $$t_{\text{NPU, avg}}(\mu s) \le \frac{1000 \times gpu\_ms}{0.8}$$
* **采样要求**：先进行 5 次 warmup，随后使用 `msprof` 采集 10 次有效调用的 `Task Duration(us)` 取平均；每次调用前恢复 $y_0$ 初始值以保证 `TC_PF_1002` ($\beta=1$) 的测试一致性。

#### (4) 性能优化推进顺序
1. 确认单次调用发射单个计算 Kernel，无冗余中间同步与 GM 搬运，$\beta=0$ 无原 $y$ 依赖；
2. N 模式针对短列优化 panel 搬运与 $x$ 缓存，调整输出块与活动核数；针对 $\le 64\text{KB}$ 矩阵启用 Whole-Band Cache；
3. T/C 模式优化列内向量连续规约，相邻列共用 $x$ 缓存，打满 Double Buffer；
4. 小尺寸采用自适应限核逻辑，消灭空核启动延迟。

---

### 3. 内存标准与记录口径

1. **任务书依据说明**：
   * 依据任务书 §3.4 规定，本算子**内存要求：不涉及**（官方未针对本算子设定如“额外内存 ≤ GPU 50%”等特定的硬件内存超额阈值）。
2. **算子实现指标**：
   * 本算子采用纯 AIV 片上 UB 累加与内积规约架构，**新增 Device GM Workspace 恒为 0 字节**，不依赖 handle 的外部临时显存，不产生与矩阵规模线性相关的中间临时副本。
3. **交付验收与测试统计口径**：
   * 按照任务书 §4 交付件要求（提交 `4.1 内存自验证报告` 与 `4.2 内存自验证日志`），在执行测试用例时真实记录并输出：
     * **必要张量物理存储量**：
       $$A_{\text{bytes}} = 8 \times lda \times n$$
       $$x_{\text{bytes}} = 8 \times \operatorname{span}(L_x, incx)$$
       $$y_{\text{bytes}} = 8 \times \operatorname{span}(L_y, incy)$$
     * **设备峰值内存与释放**：通过 Profiling/内存监测工具记录测试执行期间的实际 HBM 显存占用与释放情况，严格验证无内存泄漏、无越界非法写入。
     * **存储层级区分**：报告明确区分输入输出必要存储、句柄基础资源与测试框架为 CPU golden 和 $y_0$ 准备的测试副本。生成脚本中的 512 MB 预算属于 Host 端测试数据构造预算，不是算子自身的内存限制。

---

## 固定 CSV 测试工程设计

### 1. 文件冻结与 12 类用例覆盖台账

测试集合严格冻结为官方提供的 1200 条用例，覆盖分布如下：

| 类别代号 | 类别名称 | 用例条数 | 覆盖特征与说明 |
|:---:|:---:|---:|:---|
| **L0** | 基础用例 | 6 | trans 全枚举 × 小尺寸 (4, 8)，kl=ku=1 |
| **SQ** | 方阵尺寸 | 69 | 23 种尺寸扫描（1 到 2048）× 3 种 trans 枚举 |
| **AB** | 标量组合 | 24 | 8 组典型复数 alpha/beta 特殊值 × 3 种 trans |
| **RC** | 矩形网格 | 36 | 宽矩阵（$n \gg m$）与高矩阵（$n \ll m$）× 3 种 trans |
| **LD** | 前导维 padding | 9 | $lda = kl+ku+1+4$ 的非紧凑步长场景 |
| **BW** | 带宽极值 | 13 | $kl=ku=0$（对角矩阵）、上界、仅上带、仅下带等 |
| **INC** | 向量步长 | 36 | $incx \times incy \in \{\pm 1, \pm 2, \pm 3\}$ 全组合（含负步长） |
| **FL** | 特殊值填充 | 12 | 极端值、交替值、Inf、NaN 等数据鲁棒性测试 |
| **CV** | 中等尺寸 | 24 | 尺寸 10~400 区间扫描 × 3 种 trans |
| **ED** | 边界与负向 | 19 | 零维、空指针、非法枚举、非法 lda、负维度、零步长、负带宽 |
| **EX** | 规模扩展 | 752 | 尺寸池 × trans × 带宽 × 步长确定性采样组合 |
| **PF** | 性能测试 | 200 | 5 条任务书典型 case + 小尺寸 + 对数扫描 + 带宽扫描（纯连续访存） |
| **合计** | - | **1200** | **1000 条精度/接口测试 + 200 条纯连续性能测试** |


### 2. 参数解析与负向注入机制

测试工程解析 CSV 时，对特殊字符串进行精确映射，确保负向用例安全触发：

| CSV 编码字段 | 测试工程解析行为 |
|---|---|
| `case_name=TC_ED_235, description=null_handle` | 显式传入 `handle = nullptr` 调用接口，验证 `HANDLE_IS_NULLPTR` |
| `alpha_real="null"` / `beta_real="null"` | 设置标志位，显式传入 `alpha = nullptr` 或 `beta = nullptr` |
| `a_fill/x_fill/y_fill="NULLPTR"` | 显式传入对应的 `A = nullptr`、`x = nullptr` 或 `y = nullptr` |
| `trans="999"` | 保持非法整型枚举值传入接口，验证 `INVALID_VALUE` |
| 负维度、负带宽、非法 lda、零步长 | 原样传入接口验证状态码；测试层分配安全微型占位 buffer，防申请越界 |

### 3. 数据生成与 Golden 对齐
* 随机种子惯例：$A$、$x$、$y$ 的实部种子分别为 `random_seed`、`random_seed+1`、`random_seed+2`，虚部种子在实部基础上加 `1000`；
* Golden 严格链接 Netlib BLAS `cblas_cgbmv`，实测数据与 Golden 采用同一份 Host 输入，杜绝不同平台伪随机数差异。

---

## 实施步骤与风险防御矩阵

### 1. 实施阶段计划

| 阶段 | 核心任务 | 交付与验收证据 |
|:---:|---|---|
| **第 1 阶段** | 接口声明、Host 入参校验、按值传递 TilingData、21 列 CSV 解析器与 CBLAS 接入 | 1200 条用例完整列出，所有负向与零维用例正确返回状态码 |
| **第 2 阶段** | Kernel N/T/C 路径、复数四路乘加、32B 向下对齐、正负步长处理 | 1000 条精度用例 100% 通过（实虚部 mixed 容差与硬上限全绿） |
| **第 3 阶段** | 连续高性能路径优化、Whole-Band Cache 路径、双缓冲流水打满 | 200 条性能用例逐条对比达到 $\le gpu\_ms / 0.8$，Case 1 达标 |
| **第 4 阶段** | 交付件整理：设计文档、自验证报告（.xlsx）、日志（.log）及 README | task_submission 目录交付件齐备，PR 提交合入 ops-blas |

### 2. 重点技术风险与处理原则

| 潜在风险项 | 诱发原因 | 本方案采取的防御处理原则 |
|---|---|---|
| **带状索引越界** | 矩形矩阵（$m \neq n$）端点截断不对称 | 严格推导半开区间公式，显式处理空行、空列的规约边界 |
| **32 字节非对齐异常** | 带状偏移 $ku + lo - j$ 不是 4 的倍数 | 源地址强制向下对齐到 4 的倍数，UB 计算时跳过前缀无效元素 |
| **跨核并发写覆盖** | 非单位步长或负步长跨核写入同一个 cache line | 连续步长按 32B（4 复数）分组；非单位步长统一走单核安全路径 |
| **Inf / NaN 污染** | 补齐读取时混入带外垃圾数据或 $0 \times \text{Inf}$ | 乘法前无效 lane 强制清零；空规约区间严禁执行 $\alpha \times 0$ 乘法 |
| **Case 1 性能超时** | 小矩阵微型 DMA 发射开销过大 | 增加 Whole-Band Cache 路径，整个带状数据一次性入片上 UB 计算 |
| **公共框架污染** | 修改公共 `verify.h` 破坏全仓其他算子 | 在 `cgbmv_test.cpp` 局部显式配置 `VerifyConfig`，公共文件零侵入 |
| **性能基线被抹除** | 误运行 `gen_csv.py` 覆盖 gpu_baseline | 固化现有 CSV，禁止重新运行生成脚本 |
| **测试跑测不完整** | 过滤器写错或超时导致漏测 | 校验器检查退出码、用例清单及非 PF 唯一通过数恰好为 1000 |

---

## 兼容性分析

1. **接口规范兼容性**：函数签名与符号导出与 cuBLAS `cublasCgbmv` 及 ops-blas 公开声明完全一致，行为与异常返回值符合业界标准。
2. **工程集成兼容性**：遵循 `ops-blas` CMake 自动架构收集规范（`arch22`），头文件与源文件组织规范纯净，零修改公共框架代码，保证 PR 能够极速无冲突合并。

# 需求背景（required）

## 需求来源

社区任务：基于 Ascend C 编程语言实现 trsm_batched（批量三角矩阵求解）算子，对标 cuBLAS 的 cublasStrsmBatched / cublasCtrsmBatched 接口。

## 背景介绍

### trsm_batched 算子概述

TRSM（TRiangular Solving Matrix）是 BLAS Level 3 标准算子，用于求解三角线性方程组。trsm_batched 是其批量版本，支持同时处理多个相同尺寸的三角求解问题，广泛应用于科学计算、线性代数库、深度学习推理等领域。

trsm_batched 算子（cuBLAS 版本）实现路径和相关 API 路径

trsm_batched 算子对标实现路径为 cuBLAS：cublasStrsmBatched / cublasCtrsmBatched

### trsm_batched cuBLAS 实现现状分析

通过对 trsm_batched 算子 cuBLAS 版本的功能分析，当前支持的能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| a | 三角矩阵 A（批量） | tensor array | float32, complex64 | 上三角或下三角 | batchCount × K × K |
| b | 右端项矩阵 B（批量） | tensor array | float32, complex64 | 无 | batchCount × m × n |
| side | 左乘/右乘 | enum | 'L', 'R' | 无 | - |
| uplo | 上三角/下三角 | enum | 'U', 'L' | 无 | - |
| transa | 是否转置/共轭转置 A | enum | 'N', 'T', 'C' | 'C' 仅 complex64 有效 | - |
| diag | 单位/非单位对角 | enum | 'N', 'U' | 无 | - |
| alpha | 标量系数 | scalar | float32, complex64 | 无 | - |

计算公式：
- 左乘：$op(A) \cdot X = \alpha \cdot B$，结果 X 覆盖写回 B（in-place）
- 右乘：$X \cdot op(A) = \alpha \cdot B$，结果 X 覆盖写回 B（in-place）

### trsm_batched 算子功能分析

trsm_batched 算子功能：求解批量三角线性方程组，结果 in-place 覆盖写回 B

输入：a（三角矩阵批量）、b（右端项矩阵批量）、side、uplo、transa、diag、alpha

输出：b 被覆盖为求解结果 X

支持数据类型：float32、complex64

支持模式组合：float32 为 side(L/R) × uplo(U/L) × transa(N/T) × diag(N/U) = 16 种；complex64 为 side(L/R) × uplo(U/L) × transa(N/T/C) × diag(N/U) = 24 种（其中 transa='C' 为共轭转置，仅 complex64 有效）

支持批量处理：batchCount 个独立的三角求解问题

# 需求分析（required）

## 需求描述

使用Ascend C编程语言实现trsm_batched算子，支持float32、complex64数据类型，支持side/uplo/transa/diag共16种模式组合，支持批量处理，结果in-place覆盖写回B，性能不低于0.8x GPU A100。

## 需求拆解

1. 支持float32、complex64数据类型
1. 支持side（L/R）、uplo（U/L）、transa（N/T/C）、diag（N/U）模式组合，其中 transa='C'（共轭转置）仅 complex64 有效
1. 支持批量处理（batchCount > 0），每个batch为独立的三角求解问题
1. 支持泛化前导维度（lda != m, ldb != n）
1. 支持in-place写回（结果X覆盖B）
1. 性能不低于0.8x GPU A100

# 详细设计（required）

## 算子分析

### 数学公式

**左乘模式（side='L'）**：

$op(A) \cdot X = \alpha \cdot B$

其中 A 为 K×K 三角矩阵，B 为 m×n 右端项矩阵，X 为 m×n 未知矩阵（结果覆盖写回 B）。$op(A) = A$（transa='N'）、$op(A) = A^T$（transa='T'）或 $op(A) = A^H$（transa='C'，共轭转置，仅 complex64）。

**右乘模式（side='R'）**：

$X \cdot op(A) = \alpha \cdot B$

其中 A 为 n×n 三角矩阵，B 为 m×n 右端项矩阵。

**回代算法（Back Substitution，以 side=L, uplo=U, transa=N 为例）**：

对 B 的每一列 j，逐行从 m-1 到 0：
1. $B[i,j] = B[i,j] / A[i,i]$（diag='N' 时执行）
2. 对 k = 0, ..., i-1：$B[k,j] = B[k,j] - B[i,j] \times A[k,i]$（秩-1 更新）

**前代算法（Forward Substitution，以 side=L, uplo=L, transa=N 为例）**：

对 B 的每一列 j，逐行从 0 到 m-1：
1. $B[i,j] = B[i,j] / A[i,i]$（diag='U' 时跳过）
2. 对 k = i+1, ..., m-1：$B[k,j] = B[k,j] - B[i,j] \times A[k,i]$（秩-1 更新）

### 支持数据类型

float32、complex64（complex\<float\>）

### 支持形状

A：K×K 三角矩阵（行主序，前导维度 lda）；B：m×n 矩阵（行主序，前导维度 ldb）

左乘时 K=m；右乘时 K=n。支持泛化前导维度 lda >= K, ldb >= n。

## 算子实现

### 实现方案

#### host侧设计：

tiling策略：

采用分块算法（Blocking）将连续的 nb 行打包成一个面板（Panel），面板内使用 Vector 单元执行 TRSV（逐行串行消元），面板间使用 Cube 单元执行 GEMM（矩阵乘更新）。将 B 的列方向切分为 n_tile 大小的 tile，按需搬运避免完整 B 加载到 UB。

任务均分（两级切分）：Level 1 按 batchCount 均分到各 AI Core；Level 2 单个 batch 内按 B 的可并行维度切分（左乘切分列 n，右乘切分行 m）。

数据分块和内存优化策略：

分块大小 nb 由 UB 容量约束决定。UB 需容纳 A_diag(nb×nb)、A_offdiag(K_rem×nb)、B_tile(nb×n_tile) 及临时空间。GEMM 所需的 A2/B2/CO1 Buffer 位于独立 L1 Buffer 空间，不消耗 UB 容量。nb 必须是 16 的倍数（Mmad 分形对齐要求）。

##### 1. 分核策略：

优先使用满核的原则。使用 GetBlockDim() 动态获取核数，禁止写死。

如果核间能均分，可视作无大小核区分，大核小核数据块一致；如果核间不能均分，将余出的 batch 分配到前几个核上。

batchCount 小于核数时，实际使用核数 = batchCount，剩余核空闲。

##### 2. 数据分块和内存优化策略：

充分使用UB空间的原则。需要考虑不同数据类型（float32 vs complex64）的 UB 占用差异，complex64 实部/虚部分离导致所有 UB 需求翻倍。

UB内存大小获取：DAV_2201 (Ascend 910B4) 总 UB = 192KB，系统保留 8KB，向量可用 184KB。

Tile块计算：nb 由 UB 约束方程决定：sizeof(D_T) × nb × nb + sizeof(D_T) × K × nb + sizeof(D_T) × nb × n_tile + 临时空间 <= 184 × 1024。nb 取满足约束的最大值，并向下对齐到 16 的倍数。

数据切分：B 的列方向按 n_tile 分批（n_tile 由 UB 剩余空间决定），每批处理 B 的 n_tile 列，避免完整 B 加载到 UB。

设置切分参数：将计算的 tilingKey、batch 切分参数、nb、nTileCols 等设置到 TrsmBatchedTilingData 结构体中。

##### 3. tilingkey规划策略：

需要tilingkey的情况：需要感知host侧信息（side/uplo/transa/diag）对kernel侧走不同分支。

采用 5bit 编码，将 side/uplo/transa/diag 组合编码为 0-31 的 TilingKey：tilingKey = (side_code << 4) | (uplo_code << 3) | (transa_code << 1) | diag_code。其中 side_code: L=0, R=1；uplo_code: U=0, L=1；transa_code: N=0, T=1, C=2（2bit 编码，支持 3 个值）；diag_code: N=0, U=1。

Host 侧通过 context->SetTilingKey(tilingKey) 设置后，框架自动路由到对应 Kernel 实例。Kernel 侧 TilingKey 为编译期模板常量，通过 if constexpr 零开销分支。

#### kernel侧设计：

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. Init阶段：读取 TilingData，解析 TILING_KEY 得到 side/uplo/transa/diag 编译期常量，计算当前 Core 负责的 batch 范围，从 Workspace 读取 A/B 指针数组获取每个 batch Tensor 的 GM 地址。

2. Process阶段执行 Panel Loop（外层按 nb 步长遍历面板，内层按 nTileCols 步长遍历 B 列分批）：

   CopyIn阶段：通过 DataCopy 从 GM 搬入 A 对角线块、A 非对角线块、B tile 到 UB。泛化 lda/ldb 时使用非连续 DataCopy（srcStride = lda × sizeof(T)）。

   Compute阶段分为两个部分：

   TRSV（Vector 单元）：在 UB 中对角线块逐行求解。回代方向从 panel_size-1 到 0（前代方向相反）。每行先执行对角线除法 B[i] /= A[i,i]（diag='N' 时使用 Duplicate+Div），再执行秩-1 更新 B[k] -= B[i]*A[k,i]（Mul+Sub）。行间存在数据依赖必须串行，行内元素可并行。

   GEMM（Cube 单元）：面板间更新使用 Mmad 矩阵乘。通过 Load2D 从 GM 直接加载到 A2（A 非对角块，自动 ND→ZZ 格式转换）和 B2（B 已求解行，自动 ND→ZN 格式转换）。Mmad 计算结果在 CO1，通过 Fixpipe 搬出到 GM（自动 NZ→ND），更新 B 上面未求解的行。DataCopy 不支持搬运到 A2/B2，必须用 Load2D。

   CopyOut阶段：通过 DataCopy 将 TRSV 求解后的对角线块从 UB 写回 GM（in-place）。GEMM 更新结果通过 Fixpipe 直接写回 GM，两者写回不同行区间无重叠。

3. 复数处理（CtrsmBatched）：DAV_2201 的 Vector API（Add/Sub/Mul/Div）和 Mmad 均不支持 complex64。采用实部/虚部分离策略，将 complex\<float\> 拆分为两个 float 数组（real[], imag[]），分别使用 float API 计算。复数乘法展开为 real = a_r×b_r - a_i×b_i, imag = a_r×b_i + a_i×b_r；复数除法展开为 denom = a_r²+a_i², real = (b_r×a_r+b_i×a_i)/denom, imag = (b_i×a_r-b_r×a_i)/denom。复数 GEMM 展开为 4 次 float Mmad（A_r×B_r, A_i×B_i, A_r×B_i, A_i×B_r），串行复用同一组 A2/B2/CO1，L1 空间不翻倍。

共轭转置处理（transa='C'）：对 A 执行共轭转置等价于对 A 转置后取虚部符号取反（$A^H = \overline{A^T}$，即 real 不变、imag 取反）。在 TRSV 阶段，读取 A[k,i] 元素时对虚部取反（Neg）；在 GEMM 阶段，加载 A 非对角块时对虚部取反（通过 Vector Neg 或在 Mmad 展开中调整符号），等效于 4 次 float Mmad 中将涉及 A_i 的项符号翻转：A_r×B_r 保持、-A_i×B_i（原 +A_i×B_i）、A_r×B_i 保持、-A_i×B_r（原 +A_i×B_r）。

4. 边界情况：nb >= K 时矩阵整体可放入 UB，跳过 GEMM 阶段仅执行 TRSV。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |

## 算子约束限制

不支持列主序存储；不支持 non-in-place 模式（结果必须覆盖B）；lda >= K, ldb >= n（左乘）或 ldb >= m（右乘）；transa='C'（共轭转置）仅 complex64 有效，float32 时不支持。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 不低于cuBLAS版本 | 社区标准 |
| 性能标准 | 不低于0.8x GPU A100 | 任务要求 |

## 兼容性分析

新算子，不涉及兼容性分析

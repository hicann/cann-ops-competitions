# 需求背景（required）

## 需求来源

本需求来源于 CANN 社区任务 2026 的 Ascend 950PR `aclblasCgemmBatched` 算子开发任务。任务要求基于 `cann/ops-blas` 工程，在 `blas/gemm_batched/arch35/` 使用 Ascend C Kernel 直调方式实现单精度复数批量矩阵乘，并沿用 `include/cann_ops_blas.h` 中已有的公共接口：

```cpp
aclblasStatus_t aclblasCgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m,
    int n,
    int k,
    const aclblasComplex* alpha,
    const aclblasComplex* const Aarray[],
    int lda,
    const aclblasComplex* const Barray[],
    int ldb,
    const aclblasComplex* beta,
    aclblasComplex* const Carray[],
    int ldc,
    int batchCount);
```

适配环境为 Ascend 950PR、CANN 9.1.0、asc-devkit 9.1 及以上。算子语义对齐 `cublasCgemmBatched`，单批边界行为参考 Netlib `cgemm`。

本文档按照社区任务设计文档结构编写。实现及验收文件为：

- `blas/gemm_batched/arch35/gemm_batched_host.cpp`：参数校验、四 plane 等价工作区规划、分核和 Kernel 直调；
- `blas/gemm_batched/arch35/gemm_batched_kernel.cpp`：A expanded-AoS Pack、B 双平面 Pack、standard 2-wide MIX 和无乘积缩放；
- `blas/gemm_batched/arch35/gemm_batched_tiling_data.h`：Host/Device 共享 tiling 数据；
- `test/gemm_batched/cgemm_batched/arch35/`：GTest、cblas golden、ACL Event benchmark 和进度输出；
- `tools/`：安装、Release/O3 构建校验、精度、性能和静态审计入口。

## 背景介绍

### aclblasCgemmBatched 算子功能

对 uniform batch 中每个矩阵执行：

$$
C_i=\alpha\,op(A_i)op(B_i)+\beta C_i,
\qquad 0\le i<batchCount.
$$

其中 $op(A_i)$ 为 $m\times k$，$op(B_i)$ 为 $k\times n$，$C_i$ 为 $m\times n$。所有矩阵按列主序存储，`aclblasComplex` 在 GM 中按 `[real, imag]` 交错排列。所有 batch 共享维度、操作类型和 leading dimension，但 A/B/C 的实际地址逐项从设备侧指针数组读取。

`ACLBLAS_OP_N`、`ACLBLAS_OP_T` 和 `ACLBLAS_OP_C` 分别表示原矩阵、转置和共轭转置。复数场景中 T 与 C 不等价；C 操作必须对虚部取反。

### 任务要求与当前实现能力

任务要求同时满足完整接口语义、1000 条 CSV 精度用例、独立 NullHandle 负向用例和 200 条性能用例。性能用例全部使用 `alpha=(1,0)`、`beta=(0,0)`，但功能用例覆盖一般复标量、N/T/C 九组合、padding、batchCount 1～1024、Inf/NaN 和 quick return。

当前所有正常矩阵乘统一执行 standard 2-wide，固定为两次异步 launch：

1. 一个 AIV Kernel 同时把所有 batch 的 A 打包为 expanded AoS `op(A)^T [Kp,2Mp]`，并把 B 打包为 Br/Bi 两平面；
2. 一个 `__mix__(1,2)` Kernel 用两次宽 FP32 MMAD 产生标准复乘的四个实点积，在片上完成重构、alpha/beta 和最终 C 写回。

不存在按 alpha、shape、输入数据或精度结果切换数学算法。单位标量只在同一 epilogue 内省去旧 C 读取和冗余标量运算。D/E 实数结果不落 GM，不存在独立的结果合成 Kernel，也不在 Host 逐 batch 调用其他 GEMM。

`k=0` 或 `alpha=(0,0)` 不进入矩阵乘路径，只启动一个 AIV 缩放/清零 Kernel；`beta=(1,0)` 时直接返回。`m=0`、`n=0` 或 `batchCount=0` 为合法 no-op。

本地开发机没有 CANN 9.1.0 和 Ascend 950PR，本文只能记录源码设计、静态约束和服务器复现方法，不能把静态审计写成设备精度或性能验收结论。

# 需求分析

## 需求描述

实现必须同时解决四类问题：

1. **语义完整性**：正确处理设备侧指针数组、列主序、N/T/C、一般复数 alpha/beta、leading dimension 和所有 quick return；
2. **数值正确性**：四个标准实点积、跨 AIC/AIV 事件、batch 地址和尾块写回必须无遗漏、无竞争；
3. **批量并行性**：不能在 Host 或单核内按 batch 串行执行完整 GEMM，batch 必须进入 Pack 和 Cube 的任务轴；
4. **性能**：减少固定 launch、中间 GM 往返和重复 operand 搬运，并通过 AIC/AIV 配对、K64 双缓冲和宽 MMAD提高硬件利用率。



## 需求拆解

| 子问题 | 设计约束 | 当前实现 |
| --- | --- | --- |
| API | 复用公共接口，不增加 arch35 私有 API | 公共 `aclblasCgemmBatched` |
| 参数 | 非法值在 launch 前返回 | Host 集中校验 |
| Batch 指针 | A/B/C 地址位于 Device 指针数组 | Pack 读取 A/B 指针；MIX 读取 C 指针 |
| 布局 | Column-Major complex AoS | Aexp + Br/Bi；尾部合并回 AoS |
| N/T/C | 九组合全覆盖，C 需要共轭 | Pack 吸收地址变换与虚部符号 |
| 复数乘 | 标准四实点积，避免高消去误差 | 两次 expanded-M 宽 MMAD |
| Launch | 小矩阵不能被多阶段固定开销淹没 | 正常路径固定 2 次 |
| 中间数据 | 避免 D/E 写回再读 | L0C→UB 后重构并写 C |
| 并行 | batch、M、N 同时参与任务 | `(batch,N64,M128)` 展平并 grid-stride |
| 流水 | 避免 Copy/Compute 完全串行 | 双 L1、双 L0A、Aexp 驻留 L0B |
| 精度修正 | 不使用串行或多 Kernel 实现换精度 | standard 2-wide 本身即并行主算法 |
| 可复现 | Release/O3、链接库和进度可观察 | 脚本强校验并输出阶段、case、心跳和 ETA |

# 详细设计

## 算子分析

### 数学公式

令共轭符号已在 Pack 阶段吸收：

$$
A=A_r+iA_i,\qquad B=B_r+iB_i.
$$

Cube 利用列主序恒等式：

$$
C^T=op(B)^T op(A)^T.
$$

Pack 把 $op(A)^T$ 的每个复数元素扩展为相邻 real/imag 浮点槽，形成：

$$
A_{exp}=[A_r(:,0),A_i(:,0),A_r(:,1),A_i(:,1),\ldots],
$$

物理视图为 $P_K\times2P_M$。B 生成 $B_r$、$B_i$ 两个平面。两次宽 MMAD 为：

$$
D=B_r A_{exp},\qquad E=B_i A_{exp}.
$$

对每个 complex 输出，D/E 的偶、奇槽包含：

$$
D_{even}=B_rA_r,\quad D_{odd}=B_rA_i,
$$

$$
E_{even}=B_iA_r,\quad E_{odd}=B_iA_i.
$$

标准复乘重构为：

$$
R=D_{even}-E_{odd},
\qquad
I=D_{odd}+E_{even}.
$$

令 $\alpha=\alpha_r+i\alpha_i$、$\beta=\beta_r+i\beta_i$、旧 C 为 $C_r+iC_i$：

$$
\begin{aligned}
C'_r &= \alpha_rR-\alpha_iI+\beta_rC_r-\beta_iC_i,\\
C'_i &= \alpha_iR+\alpha_rI+\beta_rC_i+\beta_iC_r.
\end{aligned}
$$

四个实点积被成对承载在两次 256-float 宽 MMAD 中；Aexp 在两次 MMAD 间保留于 L0B。该方案没有重复执行、Host 同步、额外 Kernel 或 GM 结果面。

### 计算量分析

每个 batch 的 Cube 主体为四个标准实矩阵乘累加：

$$
N_{realMAC}=4mnk.
$$

按复数 GEMM 常用口径，名义计算量约为：

$$
F_{complex}\approx8\,mnk\times batchCount.
$$

每个 N64×M128 complex tile 不是提交四个窄 MMAD，而是提交两个 N64×expanded-M256 的宽 MMAD。两次 MMAD 复用同一 Aexp L0B，D/E 在片上按偶奇槽拆分。Pack 对每个 A/B 元素只读一次；Epilogue 对每个输出元素重构一次并融合 alpha/beta。

### 支持数据类型

| 对象 | GM/Host 类型 | 片上计算类型 | 说明 |
| --- | --- | --- | --- |
| A/B/C | `aclblasComplex` | FP32 | 实部/虚部交错，列主序 |
| alpha/beta | Host `aclblasComplex` | FP32 标量 | 按值写入 tiling |
| Packed A | FP32 | FP32 | expanded `[Ar,Ai]` 浮点槽 |
| Packed B | FP32 | FP32 | Br、Bi 两平面 |
| L0C D/E | FP32 | FP32 | 全 K 归约；不写 GM |
| 指针数组 | Device `uint64_t[]` | Device 地址 | 每批独立取址 |

所有 Cube 计算和累加固定为 FP32，不启用 HF32/TF32。

### 支持形状

运行时支持所有满足接口约束且工作区大小、任务数和对齐计算不溢出的非负 m/n/k/batchCount。实现显式处理：

- m、n、k 的非对齐尾块；
- 1、小质数、2 的幂及其相邻尺寸；
- batchCount 1～1024 的官方覆盖；
- lda/ldb/ldc 大于最小值的列主序 padding；
- N/T/C 九种组合；
- `m=0`、`n=0`、`k=0`、`batchCount=0` 和 `alpha=0`。

## 算子实现

### 实现方案

正常路径的数据流如下：

```text
Device Aarray/Barray
        │
        ▼
Launch 1: batch-parallel Pack (AIV)
        │
        ├── A: batch-major Aexp = op(A)^T [Kp,2Mp]
        └── B: batch-major [Br | Bi]
        │
        ▼  同一 stream，未插入 Host 同步
Launch 2: standard 2-wide MIX (1 AIC : 2 AIV)
        │
        ├── GM→L1→L0→MMAD D/E→L0C→Fixpipe→UB
        ├── 偶奇槽标准复数重构
        ├── alpha/beta；beta=0 不读旧 C
        └── AoS + ldc 写回 Device Carray
```

Workspace 只保存 Aexp 和 Br/Bi。正常路径没有设备选择数组、workspace 指针表、GM 结果平面或额外写回阶段；所有 batch 地址通过批次 stride 计算。

#### 3.2.1 Host 侧设计

##### 1. 参数校验与 quick return

Host 在 Kernel 启动前依次检查：

1. `handle` 非空，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. alpha、beta 非空；
3. m/n/k/batchCount 非负；
4. transa/transb 属于 N/T/C；
5. `lda >= max(1, transa==N ? m : k)`；
6. `ldb >= max(1, transb==N ? k : n)`；
7. `ldc >= max(1,m)`；
8. batchCount>0 时 Aarray/Barray/Carray 非空。

校验成功后：

- `m=0 || n=0 || batchCount=0`：返回成功，不启动 Kernel；
- `k=0 || alpha=(0,0)`：若 beta=(1,0) 返回成功，否则启动一次缩放 Kernel；
- 其余所有标量和 shape：进入 standard 2-wide 两次 launch。

所有 Kernel 都排入 handle 绑定的 stream。API 内不执行 D2H，也不调用 `aclrtSynchronizeStream`；由调用方在读取输出前同步。

##### 2. 对齐与四 plane 等价工作区

设：

$$
P_M=align_{16}(m),\qquad
P_N=align_{16}(n),\qquad
P_K=align_{8}(k).
$$

Aexp 对每个 complex M 列保存两个 FP32 列，所以 $P_M$ 对齐 16 时宽视图已经是 32 FP32 列的整数倍，满足 Cube/Fixpipe 尾宽。并且 $16\times8\times4=512$ Byte，使最小 A 基础平面和 batch 步长仍天然满足工作区对齐。继续按 32 complex 对齐不会增加合法性，只会增加窄 M 的 Pack、GM 和 padded MMAD 工作。

单个 batch 的基础 A/B 平面字节数为：

$$
S_A=4P_MP_K,\qquad S_B=4P_NP_K.
$$

Aexp 恰好占 $2S_A$，Br/Bi 恰好占 $2S_B$。每个 packed batch 与 B 区起点按 512 Byte 对齐：

$$
S_{ABatch}=align_{512}(2S_A),\qquad
S_{BBatch}=align_{512}(2S_B),
$$

$$
W=align_{512}(batchCount\cdot S_{ABatch})+
  batchCount\cdot S_{BBatch}.
$$

布局为：

```text
workspace
  A batch 0 [Aexp: Kp × 2Mp][alignment]
  A batch 1 [Aexp: Kp × 2Mp][alignment]
  ...
  512-byte aligned B base
  B batch 0 [Br][Bi][alignment]
  B batch 1 [Br][Bi][alignment]
  ...
```

所有乘法、加法和对齐都做显式溢出检查。工作区申请失败直接返回相应错误，不在失败后改走另一套实现。

##### 3. N/T/C 地址与布局

Aexp 始终表示 $op(A)^T$：

| transa | 源物理矩阵 | Aexp 一行的来源 | 虚部 |
| --- | --- | --- | --- |
| N | m×k，lda≥m | A 的第 k 列，M 个 complex 连续 | 原值 |
| T | k×m，lda≥k | 跨物理列收集 A(k,m) | 原值 |
| C | k×m，lda≥k | 跨物理列收集 A(k,m) | 取反 |

B Pack 的 source/destination leading dimension 分开记录：

| transb | 源物理 rows×cols | packed 物理视图 | 虚部 |
| --- | --- | --- | --- |
| N | k×n | $P_K\times P_N$ 的 Br/Bi | 原值 |
| T | n×k | $P_N\times P_K$ 的 Br/Bi | 原值 |
| C | n×k | $P_N\times P_K$ 的 Br/Bi | 取反 |

MIX 只需根据 B 的操作类型在 tile 主循环外选择 ND/DN layout；Aexp 已经统一成 $P_K\times2P_M$，核心循环中不再判断 transa。

##### 4. Pack 分核

A、B 共用一个 Kernel launch。block `[0,aCoreNum)` 处理 A，后续 block 处理 B。

Aexp 工作域为：

$$
T_A=batchCount\cdot P_K,
$$

即每个任务负责一个 batch 的一个 expanded K 行。B 工作域为：

$$
T_B=batchCount\cdot dstCols_B.
$$

Host 按 $2S_A$ 与 $2S_B$ 的总字节量比例分配 AIV 核，同时保证 A、B 至少各一个核、每侧核数不超过任务数。每侧再用商余数分段，核间任务差不超过 1。

每个核获得连续的 `(batch,K-row)` 或 `(batch,physical-column)` 区间，跨 batch 边界时重新绑定一次源指针和 packed batch 基址。batch 是真实并行维，不存在 Host 的 batch 循环。

##### 5. MIX 任务与分核

内部结果按 $C^T$ 组织，complex tile 为 N64×M128；Cube 看到的 float tile 为 64×256，K chunk 为 64。

任务数：

$$
T=batchCount
\left\lceil\frac{n}{64}\right\rceil
\left\lceil\frac{m}{128}\right\rceil.
$$

可用和实际 MIX 逻辑核数：

$$
C_{mix}=\min(C_{AIC},\lfloor C_{AIV}/2\rfloor),
$$

$$
C_{used}=\min(C_{mix},T).
$$

一个逻辑 MIX 核由一个 AIC 和两个 AIV 组成。AIC 的 `blockIdx` 为逻辑核号；AIV 使用 `blockIdx/2` 得到相同逻辑核号。AIC 与两个 AIV 都按：

```text
task = logicalCore + q * usedCoreNum
```

遍历完全相同的 `(batch,N-tile,M-tile)` 序列。每个输出 tile 只有一个逻辑 owner，两个 AIV 分别处理 M tile 的 64 个 complex 列，因此不存在跨 batch、跨 tile 或跨 AIV 的 C 写冲突。

当前不沿 K 维生成跨核 partial。每个 tile 的 K64 chunk 在 owner 的 D/E L0C 中顺序累加，以避免 partial workspace、原子归并、额外 C 扫描和额外 launch。K chunk 的数据搬运与 MMAD 已做双缓冲重叠；对深 K 且输出任务很少的 shape，K 轴并行度仍是需要服务器 profiler 验证的性能边界，文档不把它描述成已实现的跨核并行。

##### 6. 为何不直接套用 IterateBatch

Ascend C Matmul 提供 `IterateBatch` 批量迭代能力，适合 Matmul 对象能够按统一批布局推进的场景。本接口要求 Aarray/Barray/Carray 是设备侧指针数组，每个 batch 地址可任意且彼此不连续；地址之间没有可由 base+stride 推导的关系。

当前实现因此显式读取每批指针，并把 batch 与二维输出 tile 一起展平。这样可以：

- 保持 cuBLAS pointer-array 语义；
- 让不同 batch 与不同 tile 同时占用 AIC/AIV；
- 不在单个 Matmul 对象中串行迭代 batch；
- 不建立额外连续指针表或搬移用户矩阵；
- 对小矩阵的大 batch 保留足够任务数。

如果未来平台 API 明确支持任意 Device pointer-array 且能保持相同的跨 batch 并行调度，可以重新评估；当前设计不假设该能力。

##### 7. Tiling 数据

Pack operand tiling 的关键字段：

| 字段 | 含义 |
| --- | --- |
| rows/cols/srcLd | 逻辑 M/K 或 B 的物理有效范围 |
| dstLd/dstCols | packed 目标的对齐范围 |
| conjugate | 是否对虚部取反 |
| transposed | Aexp 是否使用二维 8-Byte gather；B 的 layout 信息 |
| layout | expanded AoS 或双平面 |
| planeBytes/batchBytes | 平面与 packed batch 地址跨度 |

MIX tiling 的关键字段：

| 字段 | 含义 |
| --- | --- |
| m/n/k | m/n 为逻辑输出，k 为对齐后的归约长度 |
| paddedM/N/K | packed 物理尺寸 |
| ldc/batchCount | 输出布局与批次数 |
| usedCoreNum | 实际 MIX 逻辑核数 |
| hasBeta | 是否需要读取旧 C |
| leftTransposed | Br/Bi 的 ND/DN 视图 |
| aBatchBytes | Aexp batch 地址跨度 |
| bPlaneBytes/bBatchBytes | Br→Bi 与 B batch 地址跨度 |
| alphaReal/Imag、betaReal/Imag | 融合 epilogue 标量 |

这些结构为按值传入 Kernel 的 POD。所有 packed 地址均由基址和 stride 算术推导，不需要 workspace 指针数组初始化。

##### 8. 异步语义与错误处理

- Pack 和 MIX 在同一 stream 依次入队，stream 顺序保证 packed 数据对 MIX 可见；
- API 不执行同步，因此 launch 后的设备错误由测试 wrapper 在 `device_sync` 阶段报告；
- 参数、对齐、字节数和任务数溢出在 Host 拒绝；
- 核数为零、MIX 缺少两个 AIV 或工作区失败返回错误；
- 任何错误都不会触发另一套算法重新计算。

#### 3.2.2 Kernel 侧设计

##### 1. Expanded A Pack

Aexp 的每行是 `op(A)^T` 的一个 K 行：

```text
Aexp[k,:] = [Ar(k,0), Ai(k,0), Ar(k,1), Ai(k,1), ...]
```

实现细节：

1. Aexp 以 `(batch,padded-K-row)` 分核，保证 padded K 行也是显式任务，不会保留旧 workspace 内容；
2. transa=N 时，一个 K 行来自 A 的连续 complex 列，按最多 1024 complex 的 tile 批量 DMA；
3. transa=T/C 时，对每个 M 位置执行二维 8-Byte block gather，再用 `GatherMask` 压紧 real/imag；
4. C 操作在 UB 中对 imag 取负；
5. Aexp 的 padded M 后缀与 padded K 行全部写零；
6. AoS 输入与 expanded AoS 输出保持 real/imag 相邻，不生成额外求和输入；
7. transposed gather 的 UB 地址高水位约 56 KiB，使用固定偏移和编译期容量检查。

##### 2. B 双平面 Pack

B 侧执行：

```text
[r0,i0,r1,i1,...]
       ├── Br = [r0,r1,...]
       └── Bi = [i0,i1,...]       # transb=C 时取反
```

主要优化：

1. UB tile 为 8192 complex；物理 rows 不超过 tile 时一次处理多个列；
2. 二维 `DataCopyPad` 把源列放入按 32 complex 对齐的 UB 槽；
3. 槽宽等于有效 rows 时不预清零，有对齐尾部时才清零；
4. `GatherMask` 单次最多拆 4096 complex，8192 tile 分两次，避免 repeat 字段回绕；
5. 只写 Br/Bi 两平面；不计算、不写未使用平面；
6. 行尾 padding 和完整 padding 列在同一 Kernel 中清零；
7. 两张平面以二维 MTE3 写回，不逐元素访问 GM。

##### 3. Standard 2-wide Cube

每个输出任务的 tile 为 N64×M128 complex，K 以 64 为 chunk。片上区域：

- 每个 L1 slot：Br 16 KiB、Bi 16 KiB、Aexp 64 KiB，共 96 KiB；
- 两个 L1 slot 分别位于两个 256 KiB bank，实际有效 payload 共 192 KiB；
- L0A 两个 16 KiB 数据槽，地址间隔 32 KiB；
- L0B 64 KiB，保存当前 Aexp；
- L0C D/E 各 64 KiB，共 128 KiB。

每个 K64 chunk 的流程：

1. MTE2 把 Br、Bi、Aexp 装入当前 L1 slot；
2. 当前 slot 被消费时，MTE2 提前装载下一 chunk 到另一 slot；
3. Br 从 L1 进入 L0A0，Aexp 从 L1 进入 L0B；
4. 第一次 MMAD 计算 D；
5. Bi 从 L1 进入 L0A1，Aexp 保持在 L0B；
6. 第二次 MMAD 计算 E；
7. 首个 chunk 设置 L0C 初始化，后续 chunk在同一 D/E L0C 中累加；
8. 当前 slot 与 L0A 槽释放给后续 chunk。

因此每个 chunk 只有一次 Aexp 的 L1→L0B 搬运。两个实结果跨全部 K chunk 独立归约，结束后才 Fixpipe 到 UB，不产生 GM partial。

##### 4. AIC/AIV 事件协议

| 事件 | 生产者 → 消费者 | 保护对象 |
| --- | --- | --- |
| `MTE1_MTE2(slot)` | MTE1 → MTE2 | L1 slot 已释放 |
| `MTE2_MTE1(slot)` | MTE2 → MTE1 | Br/Bi/Aexp 已就绪 |
| `M_MTE1(buf)` | Matrix → MTE1 | L0A 槽已释放 |
| `MTE1_M(buf)` | MTE1 → Matrix | 当前 L0 数据已就绪 |
| `M_FIX` | Matrix → Fixpipe | D/E 全 K 归约完成 |
| `FIX_M` | Fixpipe → Matrix | D/E L0C 可复用 |
| `D_READY/E_READY` | AIC Fixpipe → 两个 AIV | 对应 UB 半 tile 可消费 |
| `TILE_DONE` | 两个 AIV 分别确认 → AIC | 有效侧在 MTE3 提交后确认；空侧消费 D/E 后立即确认 |

AIC 与 AIV 使用同一个 grid-stride task 次序，ready/done 不会跨 batch 或跨 tile 串线。两个 AIV 即使分到完全 padding 的半 tile，也必须消费 D/E ready 并回传 done；空 half 消费两个 ready 后不访问 UB，立即安全确认，从而同时避免无效向量工作与跨 tile UB 竞争。Kernel 开始时初始化空闲事件，结束时逐项 drain。

```text
MTE2 ──MTE2_MTE1──> MTE1 ──MTE1_M──> Matrix
  ^                     ^                  │
  └──MTE1_MTE2──────────┘                  ├──M_FIX──> Fixpipe
                                           │             │
                                           │        D/E_READY
                                           │             ▼
                                           └──TILE_DONE── AIV/MTE3
```

##### 5. AIV Epilogue 与尾块

Fixpipe 把 D/E 的 expanded-M tile 各分给两个 AIV。每个 AIV 最多处理 64×64=4096 个 complex 输出：

1. `SplitComplex` 把 D 的偶/奇槽拆成 BrAr/BrAi；
2. `SplitComplex` 把 E 的偶/奇槽拆成 BiAr/BiAi；
3. `real=D_even-E_odd`，`imag=D_odd+E_even`；
4. 单位 alpha、零 beta 时直接 interleave，不执行一般标量向量式、不读旧 C；
5. beta 非零时只读取当前 AIV 独占的有效 C 矩形，再应用复数 alpha/beta；
6. SIMD-VF 把 real/imag 交织为 AoS，MTE3 按真实 ldc 写回有效矩形；
7. 两个 AIV 分别发出 `TILE_DONE` 确认；有效侧在 MTE3 提交后发送，空 half 在消费 D/E 且不访问 UB 后立即发送。

两个 AIV 只在 MTE3 时裁剪逻辑尾部，但都消费完整物理半 tile 的 D/E 事件。这样同时保证无事件遗留、无 padding 写入和无跨 AIV C 重叠。

若任一分量为 NaN，interleave 保持参考实现的 complex-NaN partner 语义。Tensor Vector 与 SIMD-VF、SIMD-VF 与 MTE3 之间均有显式依赖桥，不使用逐行标量处理。

##### 6. 无乘积缩放 Kernel

`k=0` 或 `alpha=0` 时，每个 AIV 核按元素总量的商余数获得连续范围，并在 batch/column 边界切段：

- beta=0：不读 C，生成全零 AoS；
- beta=(1,0)：Host no-op；
- 其他 beta：按 8192 complex tile 读取、拆分、复数缩放、交织并原地写回。

该路径没有 packed workspace，也不会读取 A/B。

##### 7. 片上资源规划

| Kernel | UB 高水位 | L1 有效 payload | L0A | L0B | L0C | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Expanded A Pack | 56 KiB | 0 | 0 | 0 | 0 | 1024 complex gather/expand |
| B 双平面 Pack | ≤192 KiB | 0 | 0 | 0 | 0 | 8192 complex 批量拆分 |
| 2-wide MIX | 192 KiB | 192 KiB | 双16 KiB | 64 KiB | 128 KiB | N64×M128 complex，D/E |
| Scale | 192 KiB | 0 | 0 | 0 | 0 | 8192 complex 缩放 |

MIX 的两个 L1 slot 使用 256 KiB bank 间距，因此最高寻址位置低于 352 KiB，仍在 512 KiB L1 内。L0A 第二槽以 32 KiB 为基址，最高寻址为 48 KiB。UB、L1、L0A、L0B、L0C 均使用固定偏移和 `static_assert`；MIX 的 UB 与 L0C 恰好达到 192/128 KiB 上界，不能隐式增加临时张量。

##### 8. 数据移动分析

忽略缓存命中并按逻辑字节计数：

$$
R_{source}=8\,batchCount\,(mk+kn),
$$

$$
W_{pack}=8\,batchCount\,(P_MP_K+P_NP_K),
$$

$$
R_{mix}=8\,batchCount\,P_K
\left[
P_M\left\lceil\frac{n}{64}\right\rceil+
P_N\left\lceil\frac{m}{128}\right\rceil
\right],
$$

$$
W_C=8\,batchCount\,mn.
$$

beta 非零时再增加：

$$
R_C=8\,batchCount\,mn.
$$

D 和 E 各含两个实点积；若把它们写入 GM 再读回，会额外产生：

$$
32\,batchCount\,mn\ \text{Byte}.
$$

当前实现把这部分完全留在 L0C/UB。Aexp 在同一 K chunk 的两个 MMAD 间驻留 L0B，减少一次 L1→L0B；Pack 只写两个输入容量，不产生未使用平面。

上述是静态逻辑流量，不包含 L2 命中、事务对齐和流水重叠，不能直接换算为设备耗时。必须用新二进制的 ACL Event 和 profiler 判断验收结果。

##### 9. 性能优化汇总

当前源码落实的性能措施：

1. 正常路径固定两个 launch；
2. A/B 在一个 AIV launch 中并行 Pack；
3. batch 进入 Pack 和 MIX 的任务轴；
4. Aexp、Br、Bi 均为紧凑 FP32，不复制 caller padding；
5. expanded-A 使用最小安全的 complex-M 16 对齐，减少窄 M 的工作区、数据移动和 padded MMAD；
6. N/T/C 在 Pack 吸收，MIX 主循环统一；
7. 四个实点积合并为两次 expanded-M 宽 MMAD；
8. K64 双 L1 预取，MTE2 与当前 chunk 的 MTE1/Matrix 重叠；
9. Aexp 在 L0B 驻留，D/E 共享一次 L1→L0B；
10. D/E 独立 L0C 全 K 归约，不产生 partial GM；
11. 两个 AIV 并行处理 M tile 两半；
12. 单位标量跳过旧 C 读取和一般 alpha/beta Vector 运算；
13. D/E 重构、复标量和 AoS 写回融合；
14. aligned Pack 避免不必要的整槽清零；
15. Pack 8192 complex、Aexp 1024 complex，摊薄 DMA/Vector 设置开销；
16. 固定地址片上张量和编译期容量检查，主循环不做动态分配。

##### 10. 精度与并行不变量

1. 所有正常 shape 使用同一 D/E 标准四实点积公式；
2. 每个输出 tile 由唯一逻辑 MIX 核持有，K chunk 只在该 owner 的 L0C 中累加；
3. AIC 和两个 AIV 的 task 映射、batch 绑定和事件次序完全一致；
4. D/E 各自完成完整 K 归约后才发 ready；AIC 收到两个 AIV 的 done 后才进入下一 task；
5. Pack 显式清零所有参与 Cube 的 padding；C 操作只改变虚部符号，不改变地址范围；
6. alpha/beta 在最终 tile epilogue 中一次完成，beta=0 不读取未初始化 C；
7. 路径不读取输入数值、case 名、golden、误差统计或先前执行结果；


## 支持硬件

| 支持的芯片版本 | 支持情况 |
| --- | --- |
| Ascend 950PR（arch35 / dav-3510） | 支持 |

软件约束：CANN 9.1.0、asc-devkit 9.1 及以上；Cube 路径依赖 arch35 `tensor_api`。

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 输入输出类型 | 仅 COMPLEX64，实部/虚部均为 FP32 |
| 存储布局 | A/B/C 列主序 complex AoS；支持 lda/ldb/ldc |
| Batch | uniform batch；地址来自 Device 指针数组 |
| 转置 | transa/transb 支持 N/T/C，C 为共轭转置 |
| 广播 | 不涉及 |
| 原地语义 | C 原地覆写；不同 C[i] 不得重叠 |
| 异步语义 | 在 handle stream 异步排队；API 内不主动同步 |
| 精度模式 | FP32 MMAD/FP32 累加，不使用低精度 Cube 模式 |
| 工作区 | handle 管理；Aexp 与 Br/Bi 两容量 |
| 空 Tensor | m=0、n=0 或 batchCount=0 no-op；k=0 或 alpha=0 执行 C=beta*C |
| 主 tile | N64×M128 complex，K64 |
| K 并行 | tile 内 K64 流水归约；当前不做跨核 K partial |
| 错误处理 | 不在错误或精度失败后自动重算 |

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 |
| --- | --- |
| COMPLEX64 精度 | 实部/虚部分别按 FLOAT32：rtol=$2^{-10}$、atol=$2^{-16}$ |
| 用例通过 | matched ratio ≥0.99，且 max abs error ≤1e-2 或 32×ULP |
| alpha=0 | `C=beta*C` 的相应用例按任务书要求 EXACT |
| 性能采样 | Ascend 950PR；先 warmup，再有效采样至少 51 次 |
| 性能判定 | NPU avg time 不高于每条 CSV 的 required time |

任务书给出的三条典型目标：

| m | n | k | batch | transA/B | alpha | beta | 目标 Avg time |
| ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| 256 | 256 | 256 | 32 | N/N | (1,0) | (0,0) | 228.60 us |
| 512 | 512 | 512 | 16 | N/N | (1,0) | (0,0) | 843.85 us |
| 1024 | 1024 | 1024 | 8 | N/N | (1,0) | (0,0) | 3272.56 us |

### 测试覆盖

官方 CSV 共 1200 行：1000 条非 TC_PF 功能/精度用例和 200 条 TC_PF 性能用例。GTest 另含一条硬编码 NullHandle，因此完整精度入口预期 1001 个测试。

| 类别 | 数量 | 主要覆盖 |
| --- | ---: | --- |
| TC_L0 | 18 | 九种转置组合和小尺寸 |
| TC_SQ | 23 | 方阵尺寸扫描 |
| TC_AB | 72 | alpha/beta 特殊值 |
| TC_BC | 13 | batchCount 扫描 |
| TC_LD | 12 | lda/ldb/ldc padding |
| TC_FL | 6 | 零、极值、Inf、NaN 等填充 |
| TC_CV | 72 | 中等尺寸九种转置组合 |
| TC_ED | 29 | quick return、空指针、非法枚举/维度 |
| TC_EX | 755 | 非对齐 shape、标量、batch 和 leading dimension 组合 |
| TC_PF | 200 | 性能与同二进制数值检查 |

200 条 TC_PF 的静态形状特征为：194 条方阵、6 条矩形，batchCount 1～1024；转置组合 NN/NT/NC/TN/TT/TC/CN/CT/CC 分别为 44/24/18/17/25/20/14/19/19。全部使用单位 alpha、零 beta，因此同时覆盖主 Cube 和单位标量 epilogue 特化。

### 测试方法

1. `validate_cases.py` 在 C++ CSV loader 之前检查 1200/200 行数、必需字段和数值格式，避免无上下文的 `stoi/stof` 异常；
2. `audit_cgemm_batched.py` 检查源码结构、四 plane 等价工作区、任务公式和禁止的残留符号；
3. 精度由 ops-blas GTest 逐 batch 调用 cblas `cgemm` 生成 golden，实部/虚部分别验证；
4. 性能由独立 benchmark 预分配数据，使用 ACL Event 测设备时间；warmup=20、repeat=51；
5. 性能脚本逐行核对 TC_PF 与 GPU baseline 的 transA/transB/m/n/k/batchCount，防止错位比较；
6. Release 校验显式配置 `CMAKE_BUILD_TYPE=Release`，要求 ASC `-O3` 且拒绝 `-O0`；若 ops_blas 目标没有 CXX 编译单元，则 CXX 优化显示 `not_applicable`；
7. 构建校验强制重编译本算子所有 `.cpp.o`，确认最终共享库包含当前实现版本；运行前按 revision 选择该库并前置其目录，GTest 与重新链接后的 benchmark 均用 `ldd` 校验实际动态加载路径；
8. 精度实时转发 GTest，每个 case 输出完成进度并定时心跳；性能输出 case start/end、编号、耗时、总耗时、ETA 和心跳。

### 可复现命令

在 CANN 9.1.0、Ascend 950PR 服务器执行：

```bash
cd /workspace/aclblasCgemmBatched
bash tools/run_smoke.sh /workspace/ascend-work/ops-blas 0
```


完整精度：

```bash
cd /workspace/aclblasCgemmBatched
CGEMM_PROGRESS_INTERVAL_SECONDS=10 \
CGEMM_ACCURACY_TIMEOUT_SECONDS=7200 \
bash tools/run_accuracy.sh /workspace/ascend-work/ops-blas 0
```

完整性能：

```bash
cd /workspace/aclblasCgemmBatched
CGEMM_PROGRESS_INTERVAL_SECONDS=10 \
CGEMM_PERFORMANCE_CASE_TIMEOUT_SECONDS=3600 \
bash tools/run_performance.sh /workspace/ascend-work/ops-blas 0
```

一键执行安装、精度和性能：

```bash
cd /workspace/aclblasCgemmBatched
bash run.sh /workspace/ascend-work/ops-blas 0
```



## 兼容性分析

1. **API 兼容性**：复用 `include/cann_ops_blas.h` 现有声明，不增加 950PR 私有接口；
2. **语义兼容性**：参数顺序、列主序、N/T/C、alpha/beta、Device 指针数组和 quick return 与任务书一致；
3. **工程兼容性**：实现位于 `blas/gemm_batched/arch35/`，通过 asc-devkit 版本条件参与构建；
4. **stream 兼容性**：所有 Kernel 使用 handle stream，保持 ops-blas 异步调用模型；
5. **数值兼容性**：所有 shape 使用标准四实点积和 FP32 Cube，以官方 FLOAT32 门禁验证；
6. **工作区兼容性**：使用 handle 工作区，启动前完成精确字节数和溢出检查；
7. **维护兼容性**：所有正常调用共享唯一 Pack/MIX 数据流、任务映射和事件协议；
8. **验证兼容性**：脚本自动加载 CANN 环境，Release/O3 校验允许纯 ASC 目标没有 CXX_FLAGS。

参考资料：

1. `aclblasCgemmBatched_Atlas950PR_task_doc.md`；
3. [cann/ops-blas](https://gitcode.com/cann/ops-blas/tree/master/blas)；
4. [Ascend C Batch Matmul 基础实现](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0041.html)；
5. [Ascend C IterateBatch](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0642.html)；
6. [Ascend C Matmul Tiling](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0703.html)；
7. [Ascend C Double Buffer 最佳实践](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0033.html)；
8. [Ascend C 512 Byte 对齐最佳实践](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_00013.html)；
9. [Netlib CGEMM](https://www.netlib.org/blas/cgemm.f)；
10. [cuBLAS GEMM Batched API](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmbatched)；
11. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。

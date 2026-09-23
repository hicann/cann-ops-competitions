# 9月社区任务-aclblasCtrsmBatched算子开发(A2/A3)设计文档

# 需求背景（required）

## 需求来源

本设计对应《aclblasCtrsmBatched_A2A3 任务书》，目标算子为 `aclblasCtrsmBatched`，代码归属 `cann/ops-blas` 的 `blas/trsmbatched/arch22/`。接口语义对齐 cuBLAS `cublasCtrsmBatched`，在 Atlas A2/A3 上求解多组相互独立的单精度复数三角线性系统。

## 背景介绍

`ops-blas` 的批量三角求解目录已有面向 Ascend 950 的实数实现。A2/A3 的单精度复数接口需要覆盖左右求解、上下三角、转置及共轭转置、单位对角和一般对角；每个批次使用独立的设备矩阵地址，解原地写回 `B[i]`。

# 需求分析（required）

## 需求描述

接口原型如下，矩阵采用列主序，`A` 和 `B` 是位于 Device 的指针数组，数组元素分别指向各批次的 Device 矩阵：

```cpp
aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* const A[], int lda,
    aclblasComplex* const B[], int ldb, int batchCount);
```

对任意批次 `i`，`side=LEFT` 求解 `op(A[i]) X[i] = alpha B[i]`；`side=RIGHT` 求解 `X[i] op(A[i]) = alpha B[i]`。`op(A)` 分别为 `A`、`Aᵀ`、`Aᴴ`。`A[i]` 的阶数为 `side=LEFT ? m : n`；`B[i]` 为 `m×n`，其有效元素由求得的 `X[i]` 原地覆盖。

## 需求拆解

1. 支持 `COMPLEX64`、列主序和 `lda/ldb` 前导维填充；批次间地址、计算与写回相互独立。
2. 覆盖 `side∈{LEFT,RIGHT}`、`uplo∈{UPPER,LOWER}`、`trans∈{N,T,C}`、`diag∈{NON_UNIT,UNIT}` 的组合。`UNIT` 模式不读取存储的对角元素。
3. `m=0`、`n=0` 或 `batchCount=0` 不执行 Device 计算；`alpha=(0,0)` 时将有效 `B[i]` 元素置零且不读取 `A[i]`。非单位对角模式要求所引用的对角元素非零，不承担奇异性检测。
4. 校验句柄、枚举、维度、批次数、标量指针、前导维、设备指针数组及其有效元素；无效输入返回规定的状态码。
5. 通过句柄绑定的 stream 异步下发计算；调用方在读取结果前同步 stream。

### 输入输出与边界语义

| 参数组 | 数据位置与组织 | 设计约束 |
| --- | --- | --- |
| `handle`、枚举、`m/n/lda/ldb/batchCount` | Host 标量 | 所有批次共享；先完成合法性校验，再决定是否下发计算。 |
| `alpha` | Host 指向的 `aclblasComplex` | 读取实部、虚部一次，并作为当前调用的公共标量；空指针非法。 |
| `A` | Device 指针数组，元素为 `A[i]` 的 Device 地址 | `A[i]` 是阶数 `k=(side=LEFT?m:n)` 的列主序三角矩阵，只读；零标量路径不引用它。 |
| `B` | Device 指针数组，元素为 `B[i]` 的 Device 地址 | `B[i]` 是 `m×n` 列主序矩阵，输入为右端项，输出为原地解；批次的有效写入区域不得重叠。 |

枚举、维度与前导维即使在空计算场景也须合法。`m=0`、`n=0`、`batchCount=0` 的有效参数调用返回成功，不读取矩阵数据、不发射求解 kernel。`batchCount<0` 为非法参数。非空调用中，零标量仍要求有效的 `B` 地址，以便写入零矩阵；`A` 不参与计算。除被引用的三角元素外，`A` 的另一半及 `lda/ldb` 填充区域不影响输出。

### 返回状态与检查边界

| 条件 | 返回状态 | Device 行为 |
| --- | --- | --- |
| `handle=nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 不下发计算。 |
| 非法枚举、负维度、`batchCount<0`、不足的前导维、`alpha=nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 不读取矩阵元素，不下发计算。 |
| 非空调用所需的 `B` 地址无效，或非零标量调用所需的 `A` 地址无效 | `ACLBLAS_STATUS_INVALID_VALUE` | 不进入三角求解。 |
| 合法的 `m=0`、`n=0` 或 `batchCount=0` | `ACLBLAS_STATUS_SUCCESS` | 不读取矩阵，不产生输出写入。 |
| 合法的非空调用 | `ACLBLAS_STATUS_SUCCESS` | 在句柄 stream 上异步计算；结果在该 stream 完成后可读。 |

校验次序只用于确定确定性的参数错误返回，不能依赖无效 Device 地址的内容。对角奇异、NaN/Inf 等数值内容不归入 Host 参数校验；非单位对角的非零性由调用方保证。

# 详细设计（required）

## 算子分析

### 数学公式

记 `T=op(A[i])`、`Y=alpha B[i]`。左求解满足 `TX=Y`，右求解满足 `XT=Y`。复数乘法按 `(aᵣbᵣ−aᵢbᵢ, aᵣbᵢ+aᵢbᵣ)` 计算。`trans=C` 时读取交换行列后的元素，并对虚部取反。对角求解采用按实部或虚部较大者缩放的复数除法，降低中间平方溢出的风险。

`trans=N` 保持三角方向，`trans=T/C` 交换有效的上下三角。左求解对下三角从首行向末行前代、对上三角反向回代；右求解对上三角从首列向末列前代、对下三角反向回代。

令 `Y=alpha B`，`T=op(A)`，下标均从零开始。左求解对第 `j` 列按依赖顺序计算

```text
X[r,j] = (Y[r,j] - Σ(p 已求解且 p≠r) T[r,p]·X[p,j]) / d[r]
```

右求解对第 `r` 行按依赖顺序计算

```text
X[r,c] = (Y[r,c] - Σ(p 已求解且 p≠c) X[r,p]·T[p,c]) / d[c]
```

其中 `diag=UNIT` 时 `d[q]=1`，否则 `d[q]=T[q,q]`。`T[r,c]` 在 `OP_N` 下取 `A[r+c·lda]`，在 `OP_T` 下取 `A[c+r·lda]`，在 `OP_C` 下取 `conj(A[c+r·lda])`。有效三角方向及求解次序如下：

| `uplo` | `trans` | `T` 的方向 | 左求解行次序 | 右求解列次序 |
| --- | --- | --- | --- | --- |
| UPPER | N | 上三角 | 递减 | 递增 |
| LOWER | N | 下三角 | 递增 | 递减 |
| UPPER | T / C | 下三角 | 递增 | 递减 |
| LOWER | T / C | 上三角 | 递减 | 递增 |

复数乘法采用四次 FP32 实数乘法和两次加减法。非单位对角的除法以较大模分量为分支进行缩放，避免直接计算 `dᵣ²+dᵢ²` 时的中间溢出或下溢。`OP_T` 不改变虚部符号，`OP_C` 只对读取的 `A` 元素取共轭；这两个路径不共用数值变换标志。

`alpha` 只作用一次：求解前形成当前右端项 `Y=alpha B`，后续 panel 更新不再次乘以 `alpha`。在原地写回中，一条右端向量的已求解元素供后续元素使用，未求解元素在轮到它之前保持为当前更新后的右端项。这个不变量同时适用于 LEFT 的列向量和 RIGHT 的行向量，可避免复数标量重复缩放或把尚未更新的数据作为解引用。

### 支持数据类型

`A[i]`、`B[i]` 和 `alpha` 均为 `COMPLEX64`，每个复数由两个 FP32 分量组成。累加、复数乘除和输出均使用 FP32 分量，不降低输入精度。

### 支持形状

`m,n≥0`、`batchCount≥0`；`lda≥max(1, side=LEFT ? m : n)`、`ldb≥max(1,m)`。`A[i]` 为相应阶数的方阵，`B[i]` 为 `m×n` 矩阵；仅 `uplo` 指定的三角部分参与计算。批次形状相同，指针地址独立。

有效元素的线性偏移分别为 `A[i][row+col·lda]`、`B[i][row+col·ldb]`，偏移单位为一个 `aclblasComplex`。因此 `lda/ldb` 可以大于逻辑行数，但不能小于逻辑行数的最小约束。每个批次只写 `B[i]` 的 `m×n` 有效区域；不写列尾填充。矩阵阶数取决于 `side`，不能在 RIGHT 场景将 `A[i]` 误按 `m×m` 访问。

## 算子实现

### 实现方案

#### Host 侧设计

Host 先校验句柄、枚举、维度、`batchCount`、`alpha` 和前导维，句柄为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，其余非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。空矩阵或零批次在参数校验后直接返回成功。非空计算校验 `B` 指针数组及每项；`alpha≠0` 时再校验 `A` 指针数组及每项。设备指针数组的元素通过 Device-to-Host 地址表读取并校验，矩阵数据保留在 Device，不回传矩阵内容。

Host 根据 `m`、`n`、`batchCount`、有效三角方向和设备核数选择小阶直接求解或分块求解，按同一 stream 下发。Tiling 数据包含批次任务范围、矩阵维度、前导维、有效三角方向、转置/共轭标志、`alpha`、panel 尺寸和尾块长度。跨核仅分配互不重叠的批次及右端行/列；每个 kernel 的写回区间唯一。

参数处理次序为：检查句柄，检查枚举与整数范围，检查 `alpha` 和前导维，判定空计算，随后按计算路径检查指针数组及其元素。这样，空计算不会因为未使用的矩阵地址触发无关的 Device 访问。指针数组从 Device 读取的是每批矩阵地址，矩阵元素始终留在 Device；地址表尺寸与 `batchCount` 成正比，且须在 Host 分配前进行整数溢出与容量检查。设备运行失败通过仓库现有状态码语义返回，不把失败误报为精度通过。

Tiling 采用两级任务编号：先映射批次 `i`，再映射该批次的独立右端向量。LEFT 的独立向量数为 `n`（每列一组），RIGHT 为 `m`（每行一组）。单向量内沿三角依赖方向串行推进，不跨核拆分同一条依赖链。分块路径的并行任务改为 `(batch, 输出 tile)`；只对同一 panel 已完成的尾部更新并行，下一 panel 必须等待本 panel 的依赖结果可见。所有下发使用 `handle` 的 stream，维持调用者观察到的异步顺序。

Tiling 信息由 Host 统一确定，Kernel 不再推断输入矩阵形状：

| 字段 | 含义与约束 |
| --- | --- |
| `m, n, lda, ldb, batchCount` | 计算逻辑维度、列主序跨度与指针数组长度；计算索引前转为足够宽的无符号偏移，避免乘法溢出。 |
| `side, effectiveUplo, trans, conjugate, unitDiag` | 决定矩阵阶数、三角依赖方向、地址交换、共轭读取和对角读取策略。 |
| `alphaReal, alphaImag` | 当前调用的 Host 标量值；零标量走独立路径。 |
| `panelSize, rhsTile, tailSize` | 分块边界及尾块有效长度，均由实际片上容量约束。 |
| `batchBegin, batchEnd, taskCount` | 每次下发负责的批次范围和独立任务数，末任务须检查范围。 |

小阶路径的任务编号 `t` 可按 `batch=t/rhsCount`、`rhs=t%rhsCount` 还原，其中 LEFT 的 `rhsCount=n`，RIGHT 的 `rhsCount=m`。分块路径以 panel 序号确定全局依赖顺序，再把该 panel 的剩余更新区域切成互不重叠的输出 tile。`batchCount×rhsCount`、矩阵字节数和地址表字节数均在 Host 侧使用宽整数检查，避免尺寸合法但总量计算溢出。

#### Kernel 侧设计

小阶路径将 `(batch, 右端列)`（LEFT）或 `(batch, 右端行)`（RIGHT）映射为独立任务。每个任务依照三角方向顺序完成前代或回代，并只写本任务对应的 `B[i]` 有效元素。矩阵读取地址由 `lda` 和转置标志计算，`B` 地址由 `ldb` 计算；非法三角和 `UNIT` 对角位置不触发数据读取。`alpha=(0,0)` 使用独立的零填充路径，仅访问 `B`。

大阶路径采用 panel 求解加尾部更新。panel 大小以 64 个复数元素为初始候选，并由实际 UB/L1 容量及右端 tile 宽度约束；不足一个 panel 的尾部使用有效长度，不读取或写入越界区域。LEFT 按行 panel 推进，RIGHT 按列 panel 推进。对角 panel 在 Vector 核完成顺序三角求解；已解 panel 对剩余矩阵的更新由 Cube 核执行四个 FP32 实数矩阵乘积并组合为复数乘积。各 panel 的求解与尾部更新在同一 stream 上顺序下发，后续 panel 只读取已完成的更新结果。

对于非对角块，更新形式为 `B_tail ← B_tail − A_offdiag·X_panel`（LEFT）或 `B_tail ← B_tail − X_panel·A_offdiag`（RIGHT）。设复数块为 `P=Pᵣ+iPᵢ`、`Q=Qᵣ+iQᵢ`，则 `PQ=(PᵣQᵣ−PᵢQᵢ)+i(PᵣQᵢ+PᵢQᵣ)`；四个实数乘积在合成后写回同一输出 tile。更新只使用 `uplo` 对应的非对角块，`trans` 决定块坐标交换，`OP_C` 同时对虚部取负。对角 panel 的顺序求解与非对角更新之间保留显式依赖，避免读取尚未更新的右端项。

分块流程的抽象顺序如下，panel 遍历方向与前述三角方向表一致：

```text
for each panel p in dependency order:
    solve diagonal block T[p,p] against current B[p]
    for each unresolved output tile q in parallel:
        if side == LEFT:
            B[q] = B[q] - contribution(T[q,p], X[p])
        else:
            B[q] = B[q] - contribution(X[p], T[p,q])
```

`solve` 阶段仅在当前 panel 内串行处理必需的三角依赖；`update` 阶段可按批次和输出 tile 并行。两阶段之间及相邻 panel 之间存在写后读依赖，同一批次的更新不得越过对应的对角求解。不同批次无数据依赖，可在各阶段并行。

小阶路径不需要全矩阵转置缓冲，直接使用索引变换读取 `A`；大阶路径对转置/共轭采用 tile 级重排或索引变换，临时存储规模受 tile 尺寸限制。对 `m`、`n` 或 panel 长度不足对齐粒度的末块，搬运前计算有效行列数，计算后仅写回有效元素。多批地址表在任务映射时按 `i<batchCount` 限界，尾批不访问数组外地址。

输入和计算 tile 在 UB/L1 中复用，不构造完整尺寸的临时 `A` 或 `B` 副本。`UNIT` 对角由常量 1 替代；共轭仅改变所读取虚部分量的符号。搬入与写回以有效 tile 尺寸、前导维和对齐要求计算，非对齐边缘使用受保护的尾块路径。运行时查询可用核数与片上存储容量，保证 tile 和并行任务数不超过设备资源。

片上存储预算按输入 `A` tile、输入/输出 `B` tile、复数实虚部计算 tile、四路乘积的必要暂存及同步开销合计，不超过目标设备实际可用容量。若候选 panel 与右端 tile 组合超出预算，优先缩小右端 tile，再缩小 panel；任何尺寸均保留至少一个有效复数元素。Device 侧中间空间按 tile 分配，不随完整 `m×n` 结果规模复制一份 `B`。Host 地址表用于参数检查，不改变 Device 矩阵的驻留与原地输出语义。

片上容量判定采用 `A_tile_bytes+B_tile_bytes+compute_bytes+temporary_bytes+reserved_bytes≤available_bytes`。这里每个 COMPLEX64 元素按两个 FP32 分量计入，复数拆分和重排如需额外缓冲也计入 `temporary_bytes`。容量不足时缩小 tile，而不是扩大 Device 工作区复制完整矩阵。Device 全局访存必须同时满足 `batch<batchCount`、`row<m`、`col<n` 及相应矩阵阶数边界；尾块不得用对齐后长度覆盖 `lda/ldb` 的填充元素。

### 性能优化方案

小阶矩阵利用批次和独立右端向量扩大并行度，避免为每个标量步骤单独发射 kernel。大阶矩阵在 panel 间保持必要的顺序依赖，在同一 panel 的尾部更新中沿批次和二维输出 tile 并行；panel 数据在片上缓存并复用。四路实数乘积共享已搬入的复数分量，写回前合成复数结果。`alpha=0` 与空矩阵路径跳过三角求解和乘法。

路径选择同时考虑三角阶数 `k`、独立右端数和 `batchCount`。小 `k`、大批次时优先减少下发次数和每任务固定开销；大 `k`、宽右端时优先把 `O(k³)` 量级的尾部更新转为块乘法。实际 panel 与 tile 参数以片上容量和设备并行度为上界，而不对 A2/A3 固定假设相同核数。性能比较采用任务书五个场景的 NPU kernel 平均耗时；参数校验、Host 地址表检查与结果同步分别核算，避免与 kernel 耗时混为一项。

## 支持硬件

| 支持的芯片版本 | 架构与目录 | 支持情况 |
| --- | --- | --- |
| Atlas A2 系列（含 Atlas 800I A2、Atlas 800T A2） | DAV_2201，`arch22` | 支持 |
| Atlas A3 系列（含 Atlas 800I A3） | DAV_2201，`arch22` | 支持 |

软件环境为 CANN 9.1.0。

## 算子约束限制

- `batchCount<0`、负维度、无效枚举或不足的前导维均为无效参数；`m=0`、`n=0` 或 `batchCount=0` 为合法空计算。
- `alpha` 为 Host 侧复数标量指针；`A`、`B` 为 Device 侧指针数组。非空计算时，每个 `B[i]` 有效；`alpha≠0` 时每个 `A[i]` 有效。
- `B[i]` 原地保存结果，批次之间的 `B[i]` 不重叠；不支持超出列主序及 `lda/ldb` 语义的任意视图。
- `diag=NON_UNIT` 时引用的对角元非零；`diag=UNIT` 时不读取对角存储值。算子不检测数值奇异性，也不保证不同并行切分下逐位相同。
- `A[i]` 与 `B[i]` 的有效矩阵区域不得相互覆盖；不同批次的写入区域不得重叠。调用方负责保持输入和输出 Device 内存直到绑定 stream 上的计算结束。
- `alpha=0` 的结果以复数零写入每个有效 `B[i]` 元素，不以 `0×B` 代替零填充，因此不依赖 `B` 原有值；填充区域保持原值。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 设计判定口径 | 标准来源 |
| --- | --- | --- |
| 精度 | 与 Netlib BLAS `ctrsm` 的各批次输出逐元素对比；复数实部、虚部分别采用 `atol=rtol=2^-13`，匹配比例不低于 0.99，最大绝对误差不高于 `max(1e-2,32×ULP)` | 本任务书 §3.2 与官方测试附件 |
| 性能 | Atlas 800T A2 (910B3) 上按有效采样大于 10 次的平均 NPU kernel 时延判定；五个典型场景上限依次为 895.1、5551、13884、21816、40969 μs | 本任务书 §3.3 |
| 场景 | 左右求解、三角方向、N/T/C、UNIT/NON_UNIT、矩形 B、前导维填充、空矩阵、零标量、边界及非法参数 | 本任务书 §2–§3 与官方用例 |

精度对比以 `B[i]` 的有效 `m×n` 元素为范围，实部与虚部分别计算混合容差。测试输入采用同一份三角矩阵及右端矩阵生成参考结果；`UNIT` 模式的参考求解同样忽略 `A` 对角存储值。性能统计先完成预热，再对有效采样求平均；单个场景的矩阵规模、批次数及枚举组合与任务书性能表一致。

| 验证类别 | 重点覆盖 | 设计判定 |
| --- | --- | --- |
| 基础与组合 | LEFT/RIGHT × UPPER/LOWER × N/T/C × UNIT/NON_UNIT | 逐批结果与参考三角求解一致，转置与共轭转置结果可区分。 |
| 形状与布局 | `m/n=1`、奇数、非方阵、非对齐维度、`lda/ldb` 填充 | 仅访问逻辑矩阵，输出填充和相邻批次不被改写。 |
| 标量与特殊值 | `alpha=0/1/-1`、一般复数、`UNIT` 对角哨兵值 | 零标量不读 `A`；单位对角不读存储值；实部和虚部按各自容差比较。 |
| 边界与异常 | 空矩阵、零批次、空指针、非法枚举、负维度、非法前导维 | 返回预期状态，合法 no-op 不下发计算，非法输入不发生越界访问。 |
| 规模与性能 | 批次数扫描、宽/高矩阵及五个典型性能场景 | panel 依赖正确，kernel 平均耗时满足对应场景上限。 |

## 兼容性分析

新增 `aclblasCtrsmBatched` 句柄式接口及 `arch22` 实现，与既有 `aclblasStrsmBatched` 的实数接口和 Ascend 950 路径独立。状态码、stream 绑定和矩阵列主序遵循 `ops-blas` 公共接口语义。

公共接口声明位于 `include/cann_ops_blas.h`，算子实现位于 `blas/trsmbatched/arch22/`，测试工程位于 `test/trsmbatched/ctrsmbatched/arch22/`。新增符号与实数接口采用不同名称，不改变原有函数签名；架构选择限定在 A2/A3 的 `arch22` 路径。测试侧以官方 CSV 字段构造同一组输入，按 `expect_result` 核对返回状态，并对成功场景逐批核对原地 `B[i]` 输出。

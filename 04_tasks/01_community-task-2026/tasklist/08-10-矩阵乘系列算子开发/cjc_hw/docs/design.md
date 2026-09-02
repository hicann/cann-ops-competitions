# 矩阵乘系列算子设计文档

# 需求背景（required）

## 需求来源

本需求来源于 CANN 2026 社区任务“矩阵乘系列算子开发”，要求在昇腾 NPU 上基于 Ascend C 实现 5 个 BLAS Level-3 单精度复数矩阵算子：

1. `chemm`：Hermitian 矩阵乘法；
2. `cher2k`：Hermitian 秩-2K 更新；
3. `cherk`：Hermitian 秩-K 更新；
4. `csymm`：对称矩阵乘法；
5. `csyrk`：对称秩-K 更新。

任务要求算子与标准 BLAS/cuBLAS 核心语义对齐，支持任务书规定的 `side`、`uplo`、`trans` 等模式，完成设计、开发、自测和性能优化。

## 背景介绍

### 矩阵乘系列算子功能概述

| 算子       | 类型                      | 核心计算              | 输出范围          |
| -------- | ----------------------- | ----------------- | ------------- |
| `chemm`  | Hermitian Matrix-Matrix | `A=A^H`，A 左/右乘 B  | 完整 C          |
| `csymm`  | Symmetric Matrix-Matrix | `A=A^T`，A 左/右乘 B  | 完整 C          |
| `cherk`  | Hermitian Rank-K        | `A·A^H` / `A^H·A` | 仅 `uplo` 指定三角 |
| `cher2k` | Hermitian Rank-2K       | `A·B^H + B·A^H`   | 仅 `uplo` 指定三角 |
| `csyrk`  | Symmetric Rank-K        | `A·A^T` / `A^T·A` | 仅 `uplo` 指定三角 |

共同特点：

- 数学数据类型均为 `complex64`，即实部、虚部各为 FP32；
- 数据布局为 Column-Major；
- 通过 `lda/ldb/ldc` 支持非紧凑 leading dimension；
- 不支持 broadcast；
- Hermitian/Symmetric 输入只引用 `uplo` 指定半边；
- HERK/HER2K/SYRK 只更新 `uplo` 指定输出三角；
- 性能目标为 NPU 性能不低于 0.8×A100 基准性能。

### 现有工程能力与设计基础

1. Atlas A2 的架构标识为 `arch22`；源码、测试和构建应使用目标 `ops-blas` 仓库及与 CANN 匹配的版本。
2. Ascend C 的 Cube/Matmul 能力可承担 FP32 主矩阵乘；Vector 侧适合完成复数拆分、结构矩阵恢复、标量运算和输出重组。
3. `ops-blas` 当前主干已有 `aclblasCgemm` 的“复数 GEMM 分解为多个实数 GEMM”工程实现，可作为 **Host 校验、Column-Major、leading dimension、workspace、尾块和复数 4M 分解的架构参考**。当前该实现面向其他架构，**不能直接等同于 A2/arch22 可复用 Kernel**。

***

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas A2 训练系列产品上实现 `chemm`、`cher2k`、`cherk`、`csymm`、`csyrk`，功能与标准 BLAS/cuBLAS 对齐，满足任务包的泛化、异常参数、精度和性能验收要求。

## 需求拆解

### 1. 公共功能要求

1. 数学数据类型为 `complex64`；公共 C API 的具体复数 ABI 类型以目标分支 `cann_ops_blas.h` 为准，不在 Kernel 内依赖 `std::complex<float>` 布局假设。
2. Column-Major 地址语义：`offset(row,col,ld)=row+col*ld`。
3. 正确支持 `lda/ldb/ldc > logical_rows`，不得把 padding 当作有效元素。
4. 不支持 broadcast。
5. 支持奇数尺寸、非 16/32B 对齐尾块、小矩阵、大矩阵以及瘦高/矮宽矩阵。
6. 对非法枚举、负维度、非法 leading dimension、必要指针空值返回目标仓库规定错误码。
7. 0 维合法 quick return。
8. 支持 `alpha=0`、`beta=0/1`、Rank-K `k=0` 等 BLAS 快速路径。
9. C 在 API 表中可能标记为“输出”，但数学语义上当 `beta!=0` 时必须读取旧 C，因此实现层面按 **in/out buffer** 处理。
10. 性能满足任务书 NPU ≥ 0.8×A100 的要求。

### 2. 各算子功能要求

#### chemm

- `side='L'`：`C = alpha*A*B + beta*C`，A 为 `m×m` Hermitian；
- `side='R'`：`C = alpha*B*A + beta*C`，A 为 `n×n` Hermitian；
- `uplo='U'/'L'`；
- `alpha/beta` 为复数；
- 输出完整 `m×n` C。

#### csymm

- `side='L'`：`C = alpha*A*B + beta*C`，A 为 `m×m` complex symmetric；
- `side='R'`：`C = alpha*B*A + beta*C`，A 为 `n×n` complex symmetric；
- `uplo='U'/'L'`；
- `alpha/beta` 为复数；
- 输出完整 `m×n` C。

#### cherk

- `trans='N'`：`C = alpha*A*A^H + beta*C`；
- `trans='C'`：`C = alpha*A^H*A + beta*C`；
- `alpha/beta` 均为实数 FP32；
- 只更新 `uplo` 指定三角；
- 输出对角线虚部为 0。

#### cher2k

- `trans='N'`：`C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C`；
- `trans='C'`：`C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C`；
- `alpha` 为复数，`beta` 为实数 FP32；
- 只更新 `uplo` 指定三角；
- 输出对角线虚部为 0。

#### csyrk

- `trans='N'`：`C = alpha*A*A^T + beta*C`；
- `trans='T'`：`C = alpha*A^T*A + beta*C`；
- `alpha/beta` 为复数；
- 只更新 `uplo` 指定三角；
- 对角线可为复数，不做 Hermitian 清零。

### 3. 设计方案选择

| 方案 | 方法                              | 优点                          | 缺点                 | 结论          |
| -- | ------------------------------- | --------------------------- | ------------------ | ----------- |
| A  | 纯 Vector 复数乘加                   | 开发简单                        | Cube 利用率低，大矩阵性能风险高 | 不采用         |
| B  | **4M 复数 GEMM + 结构化 tile 前后处理**  | 数学稳定、可复用 FP32 Cube、5 算子主干统一 | 有 AIV/Workspace 开销 | **验收基线**    |
| C  | Mixed AIC/AIV tile 流水 + 专用结构化优化 | 降低中间落 GM，性能上限高              | 同步、编译与调试复杂         | **性能主优化方向** |
| D  | 3M 复数 GEMM                      | Cube GEMM 次数由 4 降 3         | 舍入误差与 AIV 加法增多     | P2 实验优化     |

最终策略：

```text
先用 B 建立正确性与精度基线
→ 在 A2 上评估 C 的 Mixed AIC:AIV 流水
→ 仅在性能仍不足且完整精度回归通过时评估 D(3M)
```

**3M 不作为首版验收必选路径，也不提前纳入首版 TilingKey 组合。**

***

# 详细设计（required）

## 算子分析

### 数学公式

设：

```text
X = Xr + iXi
Y = Yr + iYi
```

#### 4M 复数 GEMM（首版基线）

```text
P0 = Xr * Yr
P1 = Xi * Yi
P2 = Xr * Yi
P3 = Xi * Yr

Dr = P0 - P1
Di = P2 + P3
```

4M 需要 4 次 FP32 实数矩阵乘，数学表达直接，作为首版精度和泛化基线。

#### 3M 复数 GEMM（后续实验）

```text
P0 = Xr * Yr
P1 = Xi * Yi
P2 = (Xr + Xi) * (Yr + Yi)

Dr = P0 - P1
Di = P2 - P0 - P1
```

3M 理论上减少一次实数 GEMM，但增加输入加法和误差传播，只能在 A2 实机验证其“总耗时收益 > AIV 额外开销”且完整精度套件通过后启用。

### Hermitian / Symmetric 语义

#### Symmetric

```text
A(i,j) = stored(i,j)      若 (i,j) 在 uplo 指定三角
A(i,j) = stored(j,i)      否则
```

complex symmetric 是 `A=A^T`，镜像时**不共轭**。

#### Hermitian

```text
A(i,j) = stored(i,j)          若在 uplo 指定三角且 i!=j
A(i,j) = conj(stored(j,i))    若在另一半且 i!=j
A(i,i) = real(stored(i,i)) + 0i
```

即实部关于对角线对称，虚部关于对角线反对称，真实对角元素虚部按 0 解释。

### 支持数据类型

| 对象         | chemm     | cher2k    | cherk     | csymm     | csyrk     |
| ---------- | --------- | --------- | --------- | --------- | --------- |
| A/B/C 数学类型 | complex64 | complex64 | complex64 | complex64 | complex64 |
| alpha      | complex64 | complex64 | float32   | complex64 | complex64 |
| beta       | complex64 | float32   | float32   | complex64 | complex64 |
| Cube 主计算   | FP32      | FP32      | FP32      | FP32      | FP32      |

全路径使用 FP32 主计算和 FP32 累加；不引入 HF32 路径。

### 支持形状与 leading dimension

#### chemm/csymm

- `m>=0, n>=0`；
- B、C 逻辑 shape `[m,n]`；
- `side=L`：A `[m,m]`，`lda>=max(1,m)`；
- `side=R`：A `[n,n]`，`lda>=max(1,n)`；
- `ldb>=max(1,m)`；
- `ldc>=max(1,m)`。

#### cherk/csyrk

- `n>=0, k>=0`；
- C `[n,n]`；
- `trans=N`：A `[n,k]`，`lda>=max(1,n)`；
- `cherk trans=C` / `csyrk trans=T`：A `[k,n]`，`lda>=max(1,k)`；
- `ldc>=max(1,n)`。

#### cher2k

- `n>=0, k>=0`；
- C `[n,n]`；
- `trans=N`：A/B `[n,k]`，`lda/ldb>=max(1,n)`；
- `trans=C`：A/B `[k,n]`，`lda/ldb>=max(1,k)`；
- `ldc>=max(1,n)`。

### Quick Return 与 Scale-only 语义

参数结构合法性检查必须先完成，再做计算快速路径选择。首版按以下语义实现：

1. `m==0 || n==0`（HEMM/SYMM）直接成功返回；
2. `n==0`（HERK/HER2K/SYRK）直接成功返回；
3. Rank-K/2K 中 `k==0`，主矩阵乘退化为 `beta*C`；
4. `alpha==0`，不执行 A/B 主矩阵乘；
5. `beta==0`，Epilogue 不读取旧 C；
6. `beta==1` 且主项为空时按 BLAS quick-return 直接 no-op，不额外改写 C；HERK/HER2K 只有在 Scale-only 路径实际发生写回（如 `beta!=1`）时才将被更新对角元素 imag 置 0；
7. A/B 是否允许在 `alpha==0` 等路径下为空，不在设计中自行扩大接口行为，按目标分支公共 API 和任务测试契约执行。

***

## 算子实现

### 实现方案

### 3.2.0 总体执行架构

首版提供两个执行模式：

#### Mode A：Mixed Tile Pipeline（性能优先，推荐重点验证）

Atlas A2 支持 `KERNEL_TYPE_MIX_AIC_1_1` / `KERNEL_TYPE_MIX_AIC_1_2`。优先评估 AIC 与 AIV 协作的 tile 流水：

```text
Host Validate/Tiling
      ↓
AIV：AoS→planar + mirror/conj/transpose + panel pack
      ↓  bounded tile workspace / 同步
AIC：4×FP32 Matmul（4M）
      ↓
AIV：复数组合 + alpha/beta + uplo mask + interleave
      ↓
C
```

建议从 `MIX_AIC_1_2` 原型开始评估，因为一个输出 tile 同时存在较重的复数前处理和后处理；实际 1:1 还是 1:2 由 Profiling 决定，不在设计阶段固定。

**约束：** Mixed Kernel 内的实现必须与 Host 声明的 Kernel Type 配比一致，并显式解决 AIC/AIV 之间的 workspace 数据依赖和同步。

#### Mode B：Staged Fallback（正确性/工程回退）

如果目标分支 Matmul 高阶 API、Mixed 编译或同步限制导致 Mode A 不稳定，则使用：

```text
Kernel-1 AIV_ONLY：按 wave 预处理/pack
Kernel-2 AIC_ONLY：FP32 Matmul
Kernel-3 AIV_ONLY：Epilogue + CopyOut
```

该模式 Kernel Launch 和 GM round-trip 更多，但职责清晰，适合作为 bring-up 与问题定位回退路径。Workspace 仍按 wave/tile 有界分配，不允许默认物化整个大矩阵的多份副本。

#### FP32 计算约束

- 首版及后续优化均保持 FP32 主计算和 FP32 累加；
- 不调用 `SetHF32`，也不将 HF32 作为 TilingKey 或性能优化候选；
- 性能优化仅在 FP32 精度契约内调整数据搬运、调度、流水和 tile。

***

## 3.2.1 Host 侧设计

### 1. 参数校验

建议遵循“先结构参数、后快速路径、再执行资源规划”的顺序：

1. handle / context 有效；
2. `side/uplo/trans` 枚举合法；
3. `m/n/k >= 0`；
4. `lda/ldb/ldc` 满足当前模式 BLAS 约束；
5. `alpha/beta` 标量指针满足目标公共 API 契约；
6. 零输出维 quick return；
7. 对非零问题检查 C 及实际需要的 A/B 指针；
8. 用 64 位整数计算元素数、byte offset、workspace 大小，检查乘法/加法溢出；
9. 决定 quick path / general path 后再计算 blockDim 和 workspace。

错误码、参数检查先后顺序以及指针是否可在特殊标量路径放宽，必须与目标分支 `ops-blas` 既有 BLAS API 约定一致。

### 2. 统一问题描述符

将 5 个算子归一化为：

```text
D = op(X) * op(Y)
D logical shape = [M,N]
K = contraction dimension
```

并在 `ProblemDesc` 中记录：

```text
M, N, K
src lda/ldb/ldc
side, uplo, trans
structuredMode = NONE / SYMMETRIC / HERMITIAN
outputMode = FULL / TRIANGULAR
alphaType / betaType
kernelMode = MIXED / STAGED
quickPath
```

映射关系：

- HEMM/SYMM `side=L`：`X=A, Y=B, M=m,N=n,K=m`；
- HEMM/SYMM `side=R`：`X=B, Y=A, M=m,N=n,K=n`；
- HERK/SYRK：`M=N=n,K=k`；
- HER2K：`M=N=n,K=k`，每个目标 C tile 需要构造两个 T block。

### 3. Column-Major 与 Leading Dimension

GM 逻辑地址统一使用：

```text
complex_offset(row,col,ld) = row + col*ld
byte_offset = complex_offset * sizeof(complex64)
```

对 AoS complex64，不在 Host 侧假定“整个矩阵可直接按 float2 紧凑重排”。所有 panel pack 必须同时携带：

```text
logicalRows / logicalCols
sourceLd
tileStartRow / tileStartCol
validRows / validCols
```

Cube 内部使用的 ND/NZ/NDExt 等布局由 pack/Matmul 层适配，不能通过忽略 `ld` padding 来简化。

### 4. 结构化 Tile Loader

为了避免每个元素都走复杂 `if(uplo)`，Host/Kernel 将结构化输入 tile 分类为：

```text
DIRECT      : tile 完全位于已存储三角，可顺序读取
MIRRORED    : tile 完全位于另一半，从转置 tile 读取
DIAGONAL    : tile 穿过真实对角线，需要局部 mask/mirror
```

对 `MIRRORED`：

- SYMMETRIC：转置，不共轭；
- HERMITIAN：转置并对虚部取负。

对 `DIAGONAL`：

- 按元素坐标判断来源；
- HERMITIAN 的 `i==j` 强制 imag=0；
- 只在少数对角 tile 上承担分支开销。

### 5. 两级 Tiling

#### 5.1 输出 Tile

首版候选可从 FP32 Cube 常见尺寸族中选择，例如：

```text
(tileM,tileN,tileK)
= (64/128, 64/128, 32/64/...)
```

这里只定义候选空间，不把 `128×128×64` 作为固定结论。Host 根据：

- M/N/K；
- AIC/AIV 可用核数；
- L1/L0/UB 容量；
- leading dimension 与尾块比例；
- FULL/TRIANGULAR 输出；
- Mixed/Staged 模式；
- A2 Profiling 数据；

选择最终档位。

#### 5.2 K 维分块

每个输出 tile 在 K 维循环累加，目标是：

```text
当前 K-panel Mmad
与下一 K-panel GM→L1 / pack
尽可能重叠
```

### 6. 三角输出 Tile 判定

HERK/HER2K/SYRK 不允许简单假设“`tileRow<=tileCol` 就一定安全”，因为尾块、非方形 tile 或不同 tileM/tileN 会让块索引与真实元素坐标不完全等价。

用实际元素区间分类。设输出 tile 行范围 `[r0,r1]`、列范围 `[c0,c1]`（均为有效闭区间）：

#### uplo=U

```text
FULL_TARGET : r1 <= c0
OUTSIDE     : r0 >  c1
DIAGONAL    : 其他情况
```

#### uplo=L

```text
FULL_TARGET : r0 >= c1
OUTSIDE     : r1 <  c0
DIAGONAL    : 其他情况
```

- `OUTSIDE` 不进入调度；
- `FULL_TARGET` 整 tile 计算并写回；
- `DIAGONAL` 计算后逐元素 mask；
- Rank-K 首版优先使用 `tileM==tileN`，降低三角调度复杂度，但正确性逻辑不依赖这一假设。

### 7. 多核分配与写所有权

#### FULL 输出

```text
activeTiles = ceil(M/tileM) * ceil(N/tileN)
```

#### TRIANGULAR 输出

只生成 `FULL_TARGET + DIAGONAL` 的 active tile list。

分配规则：

```text
base  = activeTiles / blockDim
extra = activeTiles % blockDim
count(core) = base + (core < extra)
```

**关键约束：一个输出 C tile 在同一阶段只允许一个 owner。**

- 不允许两个 AIC/AIV team 同时写同一 C tile；
- HER2K 的 `(i,j)` 与 `(j,i)` T block 可由同一 tile group 协作计算，但最终只由 `(i,j)` 目标 tile owner 写回；
- 这样避免 GM atomic、写写冲突和额外全核同步。

### 8. BlockDim 与 Mixed 配比

Mode A 若采用 `MIX_AIC_1_2`：

- `blockDim` 按 AIC team 数量和 active tile 数决定；
- 一个 block 对应 1 个 AIC + 2 个 AIV 的协作单元；
- 小矩阵主动减核，避免大量空闲 team；
- `blockDim` 的上限和目标硬件实际 AIC 数由 Platform API 动态获取。

Mode B 的 AIV\_ONLY 和 AIC\_ONLY 阶段分别按各自有效 work units 定核，不共用固定 blockDim。

### 9. Workspace 规划

原设计中“完整 planar + 3M/4M 全矩阵临时矩阵”只能作为极小 shape 的调试思路，**不能作为大尺寸默认 Baseline**。性能用例尺寸可能达到万级，多份 `M×N` FP32 workspace 会显著放大显存占用和带宽。

首版原则：**workspace 必须有界化**。

#### Mode A Mixed

每个 team 维护有限数量 ping-pong tile/panel buffer，例如：

```text
packed X real/imag panel
packed Y real/imag panel
P0/P1/P2/P3 当前输出 tile scratch（可按生命周期复用）
Epilogue tile
```

Workspace 规模与“并行 active team × tile 大小”相关，而不是与整个矩阵四份副本线性增长。

#### Mode B Staged

使用 wave 调度：

```text
一次只预处理 W 个 output tile / panel
→ AIC 计算这一 wave
→ AIV Epilogue
→ 复用同一 workspace 处理下一 wave
```

通过 `workspaceBudgetBytes` 限制最大 wave，避免完整矩阵中间结果常驻 GM。

### 10. 首版 TilingData

建议公共字段控制在真正运行时需要的数据：

```text
M, N, K
lda, ldb, ldc
side, uplo, trans
structuredMode
outputMode
kernelMode
quickPathType

tileM, tileN, tileK
mTiles, nTiles, activeTileCount
blockDim / waveTileCount

workspace offsets / workspace bytes
packedPanelLd / scratchLd
alpha / beta 的安全传递字段或索引
```

首版不加入：

```text
complexAlgo=3M
```

`3M` 仅在真实收益得到验证后再扩展，避免初版组合爆炸。

### 11. 首版 TilingKey

TilingKey 只编码会明显改变编译期资源和 Kernel 结构的分支：

```text
K0: FULL + GENERAL
K1: TRIANGULAR + GENERAL
K2: FULL + SCALE_ONLY
K3: TRIANGULAR + SCALE_ONLY
```

如 Mixed/Staged 需要编译成不同 Kernel，可将执行模式再作为高位维度；`side/uplo/trans` 等轻量语义优先放 TilingData，而不是穷举所有组合。

### 12. 快速路径

- `alpha==0 && beta==0`：目标区域清零；
- `alpha==0 && beta!=0`：Scale-only；
- Rank-K/2K `k==0`：Scale-only；
- `beta==0`：不加载旧 C；
- `beta==1`：省略 beta 乘法；
- HERK/HER2K Scale-only：若该路径实际写回 C（例如 `beta!=1`），目标对角元素 imag=0；若满足标准 no-op 条件则不触碰 C；
- zero dimension：Host 返回，不启动 Kernel。

***

## 3.2.2 Kernel 侧设计

### 1. Kernel 执行职责

逻辑阶段仍为：

```text
Init
→ PreProcess
→ Compute
→ Epilogue/CopyOut
```

但物理执行方式由 `kernelMode` 决定：

- Mixed：AIV/AIC 在同一 Kernel team 内协作；
- Staged：不同阶段由不同 Kernel 完成。

不能把“逻辑四阶段”误解为所有 Ascend C API 在任意 Kernel Type 中都可调用。

### 2. Init

- 读取 TilingData；
- 绑定 A/B/C/workspace；
- 建立当前 block/team 的 active tile 范围；
- 初始化 Vector Queue/LocalTensor 或 Cube Matmul 对象；
- 计算当前 tile 的 `validM/validN/validK`；
- 建立 tile owner 和 workspace slot 映射。

### 3. PreProcess

#### 3.1 AoS complex64 → planar FP32

从：

```text
[r0,i0,r1,i1,...]
```

得到：

```text
real[]
imag[]
```

要求：

- 主体使用向量化搬运/解交错；
- 尾部用合法长度搬运并在 Local/Workspace 补 0；
- 不能越过 `logical rows/cols` 读取 GM；
- 不写源矩阵 padding。

#### 3.2 结构矩阵恢复

根据 `DIRECT/MIRRORED/DIAGONAL`：

- DIRECT：直接读存储侧；
- MIRRORED-SYMM：读 `(j,i)`；
- MIRRORED-HERM：读 `(j,i)` 并 imag 取负；
- DIAGONAL-HERM：真实 `i==j` imag 强制 0。

#### 3.3 Transpose / Conjugate Transpose

尽量在 panel pack 时融合：

```text
T: row/col 交换
C/H: row/col 交换 + imag 取负
```

避免单独做一次全矩阵转置或共轭 GM round-trip。

### 4. Compute：4M FP32 Cube 主路径

对每个输出 tile，在 K 维分块累加：

```text
P0 += Xr * Yr
P1 += Xi * Yi
P2 += Xr * Yi
P3 += Xi * Yr
```

完成后交给 AIV：

```text
Dr = P0 - P1
Di = P2 + P3
```

实现时允许按生命周期复用 P scratch，而不是强制同时保存 4 份完整 tile，只要不破坏数据依赖。

### 5. Epilogue

设 `D=Dr+iDi`。

#### complex alpha

```text
alpha = ar + i*ai
Rr = ar*Dr - ai*Di
Ri = ar*Di + ai*Dr
```

#### complex beta

```text
beta = br + i*bi
Br = br*Cr - bi*Ci
Bi = br*Ci + bi*Cr
Cnew = (Rr+Br) + i*(Ri+Bi)
```

- `beta==0` 不读取 C；
- `beta==1+0i` 直接加旧 C；
- HERK 的 alpha/beta、HER2K 的 beta 使用实数特化；
- 最终按 AoS complex64 interleave 写回。

### 6. 三角 CopyOut

- FULL\_TARGET tile：写全部有效元素；
- DIAGONAL tile：按真实 `globalRow/globalCol` 做元素 mask；
- OUTSIDE tile：不产生任务；
- 不写 `ldc` padding；
- 不写非 `uplo` 区域。

### 7. Hermitian 对角线

对 CHERK/CHER2K：

```text
if globalRow == globalCol:
    output.imag = 0
```

这是**元素级规则**，不能把整个“对角 tile”误认为纯实数矩阵。

***

# 5 个算子的专用路径

## 1. chemm

### side=L

```text
A(Hermitian m×m) * B(m×n)
```

- 结构 Loader 恢复 A 当前 panel；
- B 正常 AoS→planar；
- 4M Cube；
- complex alpha/beta Epilogue；
- 写完整 C。

### side=R

```text
B(m×n) * A(Hermitian n×n)
```

逻辑左右操作数交换。优化重点是让结构化 A panel 在相邻输出 tile 间复用，减少重复 mirror/conj。

## 2. csymm

与 CHEMM 共用主干，唯一区别是结构 Loader 的镜像规则：

```text
SYMMETRIC mirror = transpose only
HERMITIAN mirror = transpose + conjugate
```

因此 Host/Tiling/4M/Epilogue 可共享，避免两套重复实现。

## 3. cherk

```text
trans=N: C = alpha*A*A^H + beta*C
trans=C: C = alpha*A^H*A + beta*C
```

- 只创建目标三角 active tiles；
- A 与其共轭转置 panel 可由同一输入 pack 逻辑生成；
- alpha/beta 为实数；
- 非目标三角完全不写；
- 真正的对角元素 imag=0。

理论输出天然 Hermitian，因此不需要计算另一半 C。

## 4. csyrk

```text
trans=N: C = alpha*A*A^T + beta*C
trans=T: C = alpha*A^T*A + beta*C
```

调度与 HERK 共用 TRIANGULAR 框架，但：

- 使用 transpose，不共轭；
- alpha/beta 为复数；
- 对角线可以有非零虚部；
- 不执行 Hermitian diagonal-zero。

## 5. cher2k

标准式：

```text
trans=N:
C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C

trans=C:
C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C
```

定义：

```text
trans=N: T = alpha*A*B^H
trans=C: T = alpha*A^H*B
```

则：

```text
C = T + T^H + beta*C
```

### 5.1 非对角目标 block

对目标 C block `C[I,J]`：

```text
C[I,J] = T[I,J] + conj(transpose(T[J,I])) + beta*C[I,J]
```

因此：

1. 计算 `T[I,J]`；
2. 计算配对块 `T[J,I]`；
3. AIV 对 `T[J,I]` 做转置+共轭；
4. 相加并只写目标 `uplo` 的 `C[I,J]`；
5. `T[J,I]` 是临时数据，不直接写非目标 C。

两个 T block 尽量由同一 tile group 处理，以复用 A/B K-panel。

### 5.2 对角 block

原设计中“对角 tile 直接 `2*Re(T)`”不成立。正确规则为：

```text
C[I,I] = T[I,I] + T[I,I]^H + beta*C[I,I]
```

其中一个 block 内：

```text
out(r,c) = T(r,c) + conj(T(c,r))
```

只有真实标量对角元素 `r==c` 才简化为：

```text
out(r,r) = 2*real(T(r,r))
```

因此对角 block 仍需保留 block 内转置+共轭组合，只在真实对角元素将 imag 强制为 0，再按 `uplo` mask 写回。

### 5.3 Workspace

HER2K 不物化完整 `T(n×n)`：

- 当前 `(I,J)` 和配对 `(J,I)` 使用有限 tile scratch；
- 组合后立即写目标 C；
- scratch slot 在后续 tile 复用。

***

# 尾块、非对齐与安全性

1. GM→Local/Workspace 只搬有效元素；
2. Cube 对齐区的无效元素在本地补 0；
3. CopyOut 只写 `validM×validN`；
4. Rank-K/2K 对 DIAGONAL tile 额外执行三角 mask；
5. 所有地址计算使用 64 位中间量；
6. `ld` padding 不读作数学数据、不写回；
7. 对结构化输入，不访问任务语义中“未存储且不应直接引用”的另一半地址，而应映射到其存储镜像位置；
8. 不允许多个 owner 同时写同一输出 tile。

***

# 性能优化策略

按优先级实施，而不是一次性堆叠全部技巧。

## P0：先建立高效正确基线

1. Cube FP32 4M 主计算；
2. 三角输出 tile 裁剪；
3. 结构矩阵 tile 级 mirror/conj，不展开全矩阵；
4. `beta=0/1`、`alpha=0` 快速路径；
5. leading dimension 和尾块走批量搬运，不退化为全量逐元素 Scalar；
6. bounded workspace；
7. 小矩阵动态减核。

## P1：A2 软硬件协同

1. Mixed AIC:AIV 1:1 / 1:2 实机对比；
2. AIV pack 与 AIC K-panel 消费流水；
3. panel ping-pong；
4. L1 复用：
   - HEMM/SYMM 根据 `side` 选择复用结构化 A；
   - Rank-K 重用同一个 A panel 生成普通/转置/共轭视图；
   - HER2K 配对 block 尽量复用 A/B panel；
5. Epilogue 融合 `Dr/Di + alpha + beta + mask + interleave`；
6. 对 DIRECT/MIRRORED tile 走无 per-element branch 的专用 pack。

## P2：仅在 P0/P1 后仍不足时

1. 3M；
2. 针对任务热点 shape 固化 tile heuristic；
3. 对少数性能瓶颈算子引入更深的专用 Cube Kernel。

### Profiling 关注指标

- Cube busy / pipe utilization；
- AIV 与 AIC 时间占比；
- MTE/GM/L1 stall；
- Mixed team 等待时间；
- workspace GM 流量；
- K-panel reuse；
- 小 shape launch 开销；
- HERK/HER2K/SYRK 是否确实只计算目标三角；
- tail/ld 路径是否大量落入 Scalar copy。

***

# 精度策略

1. 全路径采用 FP32 主计算和 FP32 累加，不启用或评估 HF32；
2. 4M 为验收基线；
3. 3M 必须独立跑全量回归后才允许进入启发式；
4. complex 实部、虚部分别校验；
5. HERK/HER2K 非 `uplo` 区域保持原值；
6. HERK/HER2K 真正对角元素 imag=0；
7. SYRK 对角线不做虚部清零；
8. `beta=0` 时不读取旧 C；
9. 极值/纯虚数/特殊 alpha-beta/奇数尺寸/大 K 等场景必须覆盖，防止 4M 中间项误差或溢出模式遗漏。

***

# 支持硬件

| 支持的芯片版本                            | 涉及勾选 |
| ---------------------------------- | ---- |
| Atlas A2 训练系列产品（`arch22`） | √    |

测试环境按任务书要求使用 CANN 9.0.0 及以上，并确保 CANN 与 `ops-blas` 源码版本匹配。

***

# 算子约束限制

1. 矩阵数学类型仅支持 complex64；
2. Column-Major；
3. 不支持 broadcast；
4. 维度参数允许 0，负数非法；
5. HEMM/SYMM：`side=L/R`、`uplo=U/L`；
6. HERK/HER2K：`trans=N/C`、`uplo=U/L`；
7. SYRK：`trans=N/T`、`uplo=U/L`；
8. HERK alpha/beta 为实数；HER2K beta 为实数；其他标量按任务书为复数；
9. Hermitian 输入另一半不直接引用，真实对角 imag 按 0；
10. HERK/HER2K/SYRK 只修改 `uplo` 指定输出三角；
11. leading dimension 必须满足 BLAS 约束；
12. 仅支持 BLAS Column-Major + leading dimension，不支持任意 Tensor stride；
13. 仅使用 FP32 主计算和 FP32 累加，不提供 HF32 路径；
14. 首版不以 3M 作为必须路径。

***

# 算子性能可测分析

## 精度标准/性能标准

| 验收项          | 标准                     | 来源           |
| ------------ | ---------------------- | ------------ |
| 功能           | 与任务要求/BLAS 核心语义一致      | 任务书 + Netlib |
| 泛化           | 通过任务包 CSV 全部合法/异常/边界场景 | 任务包          |
| 精度           | 满足生态算子开源精度标准           | 任务书 + opbase |
| 三角语义         | 非 `uplo` 输出区保持原值       | 任务包          |
| Hermitian 对角 | CHERK/CHER2K 对角 imag=0 | BLAS + 任务包   |
| 性能           | NPU ≥ 0.8×A100         | 任务书          |

## 1. 自测规模

任务包各算子 README 定义：

| 算子     |     精度用例 | TC\_PF 性能用例 |       合计 |
| ------ | -------: | ----------: | -------: |
| chemm  |      218 |         100 |      318 |
| cher2k |      227 |         100 |      327 |
| cherk  |      208 |         100 |      308 |
| csymm  |      218 |         100 |      318 |
| csyrk  |      216 |         100 |      316 |
| **总计** | **1087** |     **500** | **1587** |

`verify_accuracy.py` 会排除 `TC_PF_*`，因此上述“精度 + 性能”划分成立。

覆盖重点：

- side/uplo/trans 全组合；
- 1、3、5、7、65 等边界/奇数尺寸；
- 大尺寸、瘦高、矮宽；
- alpha/beta 0、1、-1、复数、纯虚数、大值；
- zero/random/alternate/extreme 填充；
- leading dimension padding；
- 0 维；
- 空指针/非法枚举/非法 ld；
- Hermitian 对角与非 `uplo` 保持。

任务包当前 `cherk_test.csv` 未单独暴露 `nullBeta` 分类时，仍应按公共接口规范校验 beta，并建议在最终自测中补充该异常测试。

## 2. 精度验收

任务包 README 快照给出的 FP32 容差：

```text
atol = 2^-16
rtol = 2^-10
```

但最终 PR 必须服从目标分支当前生态精度框架，不能把 README 快照值写死到 Kernel。

建议验收：

- HEMM/SYMM：完整 C 的 real/imag；
- HERK/HER2K/SYRK：目标 `uplo` 容差比较；
- 非目标三角：保持原值/按任务脚本 EXACT；
- HERK/HER2K：对角 imag；
- fast path：`alpha=0`、`beta=0/1`、`k=0` 单独回归；
- Mixed 与 Staged 两模式必须产生一致结果；
- 3M 如启用，必须与 4M FP32 基线做全量差异回归。

## 3. 性能验收

任务标准：

```text
NPU_perf >= 0.8 * A100_perf
```

相同工作量下等价于：

```text
NPU_time <= A100_time / 0.8
         <= 1.25 * A100_time
```

任务书 20 个重点基准点：

| 算子     | Shape / 模式     | A100 ms | NPU 目标 ≤ ms |
| ------ | -------------- | ------: | ----------: |
| chemm  | 1024×1024, L/U |   0.609 |       0.761 |
| chemm  | 2048×2048, L/U |   4.951 |       6.189 |
| chemm  | 1024×1024, R/L |   0.490 |       0.613 |
| chemm  | 2048×2048, R/L |   4.337 |       5.421 |
| cher2k | 1024×1024, U/N |   0.654 |       0.818 |
| cher2k | 2048×2048, U/N |   4.055 |       5.069 |
| cher2k | 1024×1024, L/C |   0.548 |       0.685 |
| cher2k | 2048×2048, L/C |   4.156 |       5.195 |
| cherk  | 1024×1024, U/N |   0.314 |       0.393 |
| cherk  | 2048×2048, U/N |   1.929 |       2.411 |
| cherk  | 1024×1024, L/C |   0.250 |       0.313 |
| cherk  | 2048×2048, L/C |   2.024 |       2.530 |
| csymm  | 1024×1024, L/U |   0.615 |       0.769 |
| csymm  | 2048×2048, L/U |   4.902 |       6.128 |
| csymm  | 1024×1024, R/L |   0.494 |       0.618 |
| csymm  | 2048×2048, R/L |   4.363 |       5.454 |
| csyrk  | 1024×1024, U/N |   0.307 |       0.384 |
| csyrk  | 2048×2048, U/N |   1.945 |       2.431 |
| csyrk  | 1024×1024, L/T |   0.249 |       0.311 |
| csyrk  | 2048×2048, L/T |   2.009 |       2.511 |

### 性能数据使用规则

1. 每算子 `gpu_baseline.csv`：当前 `verify_performance.py` 直接用于 PASS/FAIL 的 4 个任务书基准点；
2. 全局 `matmul_series_gpu_perf.csv`：包含 1587 条 GPU 采样记录，可用于扩展趋势对比、热点 shape 选择和回归分析；
3. 500 个 `TC_PF` 用例均需要采集 NPU 性能，但“有 GPU 数据”与“当前验收脚本会作为硬门槛比较”必须区分；
4. `msprof` Kernel 时间用于微观优化；GTest per-case 时间包含 Host/框架开销，适合作为端到端回归而非唯一 Kernel 指标。

## 4. 性能优化判定顺序

若未达 0.8×A100，不应直接上 3M。按顺序定位：

```text
是否误算非 uplo tile
→ Cube busy 是否不足
→ AIV pack/epilogue 是否拖住 AIC
→ GM/L1/MTE stall
→ workspace 是否大量落 GM
→ panel reuse 是否不足
→ 小 shape 是否 launch 主导
→ tail/ld 是否退化
→ 最后再评估 3M/专用 kernel
```

***

# 兼容性分析

1. 本系列为新增 BLAS 算子，不改变存量算子数学行为；
2. API 的具体类型名、状态码、const 约束、标量指针形式以目标分支公共头文件为准；
3. Column-Major、leading dimension、quick-return 与标准 BLAS/任务验收保持一致；
4. arch22 实现与其他架构隔离；
5. 公共代码仅下沉本系列稳定复用模块；
6. 最终 PR 前必须重新核对目标分支 `cann_ops_blas.h`、CONTRIBUTING、CI、精度标准，避免设计时的 master 状态与提交时发生漂移。

***

# 设计风险与回退策略

| 风险                         | 影响            | 处理                                                                    |
| -------------------------- | ------------- | --------------------------------------------------------------------- |
| HER2K block 数学实现错误         | 精度失败          | 使用 `T[I,J] + conj(T[J,I]^T)`；对角 block 也做块内 H 组合                       |
| 三角 tile 仅按 tile index 判定   | 尾块/非方 tile 误写 | 用真实元素区间分类 FULL/OUTSIDE/DIAGONAL                                       |
| 非 `uplo` 被覆盖               | EXACT 失败      | active tile 裁剪 + diagonal mask + 单 owner 写回                           |
| 全矩阵 planar/4M workspace 过大 | 内存/带宽爆炸       | bounded tile/wave workspace                                           |
| Mixed AIC/AIV 同步错误         | 死锁/脏读         | 明确 kernel type、workspace slot、producer-consumer 同步；保留 Staged fallback |
| AIV 吞吐不足                   | Cube 等待       | pack/transpose/conj 融合，DIRECT/MIRRORED 专用路径，1:1/1:2 profiling         |
| 3M 误差偏大                    | 精度失败          | 首版不用；通过完整回归后才进入启发式                                                    |
| leading dimension/尾块越界     | 泛化失败          | 64 位地址、valid shape、local zero-pad、masked copyout                      |
| 小矩阵多核开销大                   | 性能差           | 动态减核/Scale-only/小矩阵特化                                                 |
| 仓库接口/目录变化                  | PR 冲突         | 开发前锁定 tag/commit，按 SIG 最终目录和 ABI 实现                                   |

***

# 实施与验收顺序

建议按以下里程碑推进：

### M0：接口与测试基线

- 锁定 CANN/ops-blas commit；
- 接入 5 个 public API；
- 跑通参数校验/zero-dim/scale-only；
- 导入任务 CSV。

### M1：4M Staged 正确性

- 先完成 AoS→planar、4M、Epilogue；
- HEMM/SYMM；
- HERK/SYRK；
- 最后 HER2K；
- 通过全部非性能精度/异常用例。

### M2：三角与 workspace 优化

- active triangular tile；
- DIRECT/MIRRORED/DIAGONAL；
- bounded wave workspace；
- 消除不必要 GM round-trip。

### M3：Mixed AIC/AIV 性能路径

- 1:1、1:2 原型；
- profiling；
- 逐算子达到 20 个重点 A100 门槛；
- 扩展到 500 PF 回归。

### M4：P2 优化（仅必要时）

- 3M；
- 热点 shape 特化。

***

# 参考资料

## 任务资料

1. `matmul_series_task_doc.md`：矩阵乘系列算子开发任务书。
2. `design_template.md`：社区任务设计模板。
3. `test_script/{chemm,cher2k,cherk,csymm,csyrk}`：README、测试 CSV、`verify_accuracy.py`、`verify_performance.py`、`gpu_baseline.csv`。
4. `test_script/matmul_series_gpu_perf.csv`：系列算子扩展 GPU 性能采样。

## 官方/上游资料

1. Netlib BLAS/LAPACK：CHEMM、CHER2K、CHERK、CSYMM、CSYRK。
2. NVIDIA cuBLAS：HEMM/HER2K/HERK/SYMM/SYRK 接口语义。
3. Ascend C CANN 9.0 Kernel Type 文档：A2 支持 AIV\_ONLY、AIC\_ONLY、MIX\_AIC\_1\_1、MIX\_AIC\_1\_2 等模式。
4. Ascend C Matmul 文档：FP32 Matmul 配置、Tiling 与 Kernel Type 约束。
5. `cann/ops-blas` QUICKSTART：A2/A3 对应 `arch22`。
6. `cann/ops-blas` CONTRIBUTING：新算子需先经 SIG 评审，并按 SIG 分配目录提交；PR 前完成编译、测试和 Markdown 检查。
7. `cann/ops-blas` 当前 GEMM/Cgemm 实现：复数 4M、Validate/Launch、workspace、Column-Major/leading-dimension 和尾块工程参考。
8. 生态算子开源精度标准：目标提交版本 `cann/opbase` 对应精度规范。

***

# 设计结论

本系列首版不追求“5 个算子各写一套大 Kernel”，而采用统一的结构：

```text
统一 BLAS Host 校验/归一化
→ structured tile loader
→ bounded workspace
→ FP32 4M Cube 主计算
→ fused complex Epilogue
→ FULL / TRIANGULAR 输出调度
```

其中：

- HEMM/SYMM 的核心差异收敛到 `HERMITIAN/SYMMETRIC` Loader；
- HERK/SYRK 共享三角输出调度；
- HER2K 使用正确的配对 block `T[I,J] + conj(T[J,I]^T)`；
- A2 性能优化优先从 Mixed AIC/AIV、三角裁剪、panel reuse 和 bounded workspace 获取；
- 3M 只作为数据证明有效后的后续优化，而不是首版设计复杂度来源。

该设计优先保证 **数学正确、接口可验收、内存有上界、性能优化可渐进**，并为 Atlas A2 上达到 0.8×A100 留出明确的软硬件协同优化路径。

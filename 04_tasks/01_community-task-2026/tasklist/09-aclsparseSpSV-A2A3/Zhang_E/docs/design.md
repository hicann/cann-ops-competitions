# aclsparseSpSV 算子设计文档（Atlas A2/A3）

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 9 月社区任务—aclsparseSpSV 算子开发（A2/A3） |
| 目标硬件 | Atlas A2/A3（DAV-2201，`arch22`） |
| 软件版本 | CANN 9.1.0 及后续配套版本 |
| 实现仓库 | `ops-sparse` |
| 提交者 | `Zhang_E` |

# 需求背景（required）

## 需求来源

本设计对应《aclsparseSpSV 算子开发（A2/A3）任务书》。任务要求参考稀疏三角向量求解接口语义，在 Atlas A2/A3 上实现公开 C++ Legacy API、Host 参数校验与生命周期管理、Ascend C Kernel、C++ UT/ST、性能脚本和文档，主计算全程运行于 NPU 调用流，不使用 CPU fallback。

本次实现涉及的主要文件如下：

| 内容 | 路径 |
| --- | --- |
| 公开接口 | `include/cann_ops_sparse.h` |
| SpSV 描述符 | `sparse/common/aclsparse_spsv_descr.h` |
| Host 实现 | `sparse/spsv/arch22/spsv_host.cpp` |
| Kernel 实现 | `sparse/spsv/arch22/spsv_kernel.cpp` |
| Tiling 数据 | `sparse/spsv/arch22/spsv_tiling_data.h` |
| 接口文档 | `sparse/spsv/README.md` |
| 设计与复现说明 | `sparse/spsv/arch22/README.md` |
| C++ 功能与精度测试 | `test/spsv/arch22/spsv_test.cpp` |
| 性能测试 | `test/spsv/arch22/spsv_perf.cpp`、`test/spsv/arch22/run_perf.sh` |

## 背景介绍

### SpSV 算子功能

SpSV（Sparse Triangular Solve with Vector）求解稀疏三角线性方程：

$$
op(A)Y=\alpha X,
$$

其中 $A$ 是 $m\times m$ 稀疏三角方阵，$X$、$Y$ 是长度为 $m$ 的稠密向量，$\alpha$ 为标量。`op(A)` 支持原矩阵 N、转置 T 和共轭转置 H；`diagType=UNIT` 时逻辑对角按 1 处理，`diagType=NON_UNIT` 时使用显式对角值。

接口按以下生命周期调用：

```text
CreateDescr
  -> BufferSize
  -> Analysis
  -> Solve（可重复）
  -> UpdateMatrix（可选）
  -> Solve（复用 Analysis 结果）
  -> DestroyDescr
```

`BufferSize` 查询 Device workspace，`Analysis` 将输入格式确定性规范化为 0-based CSR 并建立对角位置，`Solve` 完成前代或回代，`UpdateMatrix` 在稀疏结构不变时更新全部 values 或逻辑对角 values。

### 工程现状与本次增量

`ops-sparse` 已有公共描述符、handle、SpMat/DnVec 和 SpSV 接口框架。本任务新增并完善 `arch22` 路径：

1. 支持 CSR、CSC、COO、SLICED_ELL 四种输入格式和 I32 Device 索引。
2. 支持 FP32、complex64，complex64 的 H 路径执行真实共轭转置。
3. 支持 LOWER/UPPER、UNIT/NON_UNIT、base 0/1、Host/Device pointer mode。
4. 补齐 BufferSize、Analysis、Solve、GENERAL/DIAGONAL UpdateMatrix 生命周期。
5. 将格式转换、转置、稳定排序、值更新和求解放在 NPU Kernel 上执行。
6. 提供未排序和重复坐标的确定性规则，以及 X/Y 完全同址原地求解。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（DAV-2201）上实现：

```text
op(A) * Y = alpha * X
```

其中 A/X/Y/alpha/computeType 同型，支持 `ACL_FLOAT` 和 `ACL_COMPLEX64`；矩阵为 CSR、CSC、COO 或 SLICED_ELL 格式，Device 索引为 I32，index base 支持 0/1。接口复用调用方 `aclsparseHandle_t` 中的 stream，并保证 Solve 与 UpdateMatrix 异步下发。

公开接口保持 `include/cann_ops_sparse.h` 中的原型不变：

```cpp
aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr);
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr);
aclsparseStatus_t aclsparseSpSV_bufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr,
    size_t *bufferSize);
aclsparseStatus_t aclsparseSpSV_analysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr,
    void *externalBuffer);
aclsparseStatus_t aclsparseSpSV_solve(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr);
aclsparseStatus_t aclsparseSpSV_updateMatrix(
    aclsparseHandle_t handle, aclsparseSpSVDescr_t spsvDescr,
    void *newValues, aclsparseSpSVUpdate_t updatePart);
```

## 需求拆解

| 编号 | 子项 | 要求 |
| --- | --- | --- |
| R1 | 接口 | 六个公开接口名称、参数顺序和枚举保持不变 |
| R2 | dtype | FP32、complex64，A/X/Y/alpha/computeType 一致 |
| R3 | 格式与索引 | CSR/CSC/COO/SLICED_ELL，I32，base 0/1 |
| R4 | 三角语义 | LOWER/UPPER、UNIT/NON_UNIT、N/T/H |
| R5 | 生命周期 | Analysis 绑定结构、属性、上下文和 workspace，Solve/Update 不得越过 Analysis |
| R6 | 数值更新 | GENERAL 更新物理 values；DIAGONAL 接收 `m` 个连续逻辑对角值 |
| R7 | 运行语义 | Host/Device alpha、调用方 stream、X/Y 完全同址 |
| R8 | 确定性 | 未排序和重复坐标采用固定规范化与累加顺序 |
| R9 | 异常处理 | Host 参数检查和 Device 稀疏结构检查均返回确定错误码 |
| R10 | 验收 | CPU Golden、C++ UT/ST、Profiler、性能和内存验证 |

## 输入输出规格

| 对象 | 位置 | 类型/规格 | 主要约束 |
| --- | --- | --- | --- |
| A | Device | FP32/complex64，`[m,m]` | 四种稀疏格式，I32 索引，base 0/1 |
| X | Device | FP32/complex64，`[m]` | Solve 时描述符和值指针有效，只读 |
| Y | Device | FP32/complex64，`[m]` | 输出，可与 X values 完全同址 |
| alpha | Host/Device | 一个 FP32/complex64 标量 | 位置由 handle pointer mode 决定 |
| externalBuffer | Device | 至少为 BufferSize 返回值 | 首地址 512 B 对齐，异步任务完成前保持有效 |
| newValues | Device | GENERAL 为源格式 values；DIAGONAL 为 `m` 个值 | dtype 与 A 一致，结构不变 |

BufferSize 和 Analysis 允许 vecX/vecY 描述符为 NULL；Solve 要求二者非 NULL 且 values 是有效 Device 指针。

# 详细设计（required）

## 算子分析

### 数学公式与求解方向

令 $B=op(A)$。对下三角系统，第 $i$ 行前代为：

$$
y_i=\frac{\alpha x_i-\sum_{j<i}b_{ij}y_j}{d_i},
\qquad
d_i=\begin{cases}
1,&\text{UNIT},\\
b_{ii},&\text{NON\_UNIT}.
\end{cases}
$$

上三角系统按行号逆序回代，并把求和范围改为 $j>i$。转置或共轭转置会翻转有效三角方向：

| fillMode | `opA=N` | `opA=T/H` |
| --- | --- | --- |
| LOWER | 前代 | 回代 |
| UPPER | 回代 | 前代 |

complex64 的基本运算使用相邻两个 FP32 表示实部和虚部。H 路径在规范化或更新 values 时执行虚部取反，从而得到 $A^H=\overline{A}^{T}$；FP32 下 H 与 T 等价。

### 支持范围

| 项目 | 支持范围 |
| --- | --- |
| dtype | `ACL_FLOAT`、`ACL_COMPLEX64` |
| 索引 | row pointer/row index 与 column index 均为 `ACL_SPARSE_INDEX_32I` |
| index base | `ACL_SPARSE_INDEX_BASE_ZERO`、`ACL_SPARSE_INDEX_BASE_ONE` |
| 稀疏格式 | CSR、CSC、COO、SLICED_ELL |
| 属性 | LOWER/UPPER、UNIT/NON_UNIT、N/T/H |
| 算法 | `ACL_SPARSE_SPSV_ALG_DEFAULT` |
| shape | A 为 `[m,m]`，X/Y 为 `[m]`；`m`、`nnz` 不超过 `INT32_MAX` |

## 算子实现

### 总体方案

```text
Host
  参数、dtype、shape、format、pointer、生命周期校验
  -> 计算 512B/64B 对齐的 workspace 布局
  -> 缓存 Analysis 绑定状态与 Tiling 数据
  -> 在 handle stream 上启动 arch22 Kernel

Analysis Kernel（单 AIV）
  枚举 CSR/CSC/COO/SLICED_ELL
  -> N/T/H 坐标变换和 base 归一化
  -> 两遍计数/稳定散射，构造 0-based CSR
  -> 每行按 (column, source-position) 稳定排序
  -> 合并重复对角并记录 diag position

Solve Kernel（单 AIV）
  读取 Host/Device alpha
  -> 按有效三角方向前代/回代
  -> 写回 Y（支持 X/Y 同址）

Update Kernel（单 AIV）
  GENERAL：按 source permutation 更新全部值并重新规范化对角
  DIAGONAL：按行更新已有逻辑对角
```

### Host 侧设计

#### 参数检查

Host 侧按以下类别校验：

1. `handle`、`alpha`、`matA`、`spsvDescr` 和阶段所需参数非空。
2. pointer mode 仅允许 Host/Device；Device alpha 通过 `aclrtPointerGetAttributes` 校验。
3. `opA` 仅允许 N/T/H，`alg` 仅允许 DEFAULT。
4. computeType 仅允许 FP32/complex64，且与 A/X/Y dtype 一致。
5. A 必须为方阵，`m/nnz <= INT32_MAX`，索引必须为 I32，base 为 0/1。
6. format、fillMode、diagType、SLICED_ELL 的 slice 数和宽度合法。
7. 矩阵索引、values、X/Y、workspace、newValues 为有效 Device/Managed 指针并满足自然对齐。
8. X/Y 长度必须为 `m`；Solve 时描述符和值指针必须有效。
9. workspace 逐段计算时检查乘法、加法和最终对齐的 `size_t` 溢出。

Host 无法安全读取 Device 索引内容，因此行指针单调性、首尾边界、列索引范围和 SLICED_ELL 物理布局由 Analysis Kernel 检查。Kernel 将状态写入 workspace 头部，Host 只回读 4 字节状态并同步一次调用 stream，把非法结构转换为 `ACL_SPARSE_STATUS_INVALID_VALUE`。Solve 与 UpdateMatrix 保持异步下发。

#### 生命周期状态

Analysis 在 opaque SpSV 描述符中缓存以下弱引用或快照：

- handle、stream、pointer mode；
- matA/可选 vecX/vecY 描述符和值指针；
- `m`、`nnz`、format、fillMode、diagType、opA、base、dtype；
- CSR/SLICED_ELL 结构指针和 SLICED_ELL 参数；
- workspace 指针、大小和各内部区域偏移；
- Host alpha 的按位值，或 Device alpha 指针。

Solve 会重新校验缓存状态。handle/stream/pointer mode、矩阵结构或属性、opA、dtype、workspace 绑定发生变化时返回错误并要求重新 Analysis。Analysis 时 vecX/vecY 为 NULL 时，Solve 可首次绑定合法向量；若 Analysis 已绑定向量，则描述符和值指针必须保持一致。

#### Workspace 规划

当 `m=0` 时 BufferSize 返回 0，不启动 Device Kernel。`m>0` 时 workspace 布局为：

```text
0
| 512 B header（Analysis 状态与实际 nnz）
| canonical CSR rowPtr: (m + 1) * int32
| canonical CSR colInd: nnz * int32
| canonical CSR values: nnz * valueStride * float
| source permutation: nnz * int32
| diagonal position: m * int32
| 末尾补齐到 512 B
```

内部区域按 64 B 对齐，workspace 首地址和总大小按 512 B 对齐。`valueStride` 在 FP32 时为 1，在 complex64 时为 2。近似空间为：

```text
FP32:     512 + 8*m + 12*nnz + alignment
complex64:512 + 8*m + 16*nnz + alignment
```

Analysis 复用 `diagonal position` 区域作为第二遍稳定散射 cursor，避免再分配一组 nnz 规模的临时坐标和值缓冲。

#### Tiling 数据

Host 向 Kernel 传递：`m/nnz`、五个 workspace 偏移、Host alpha 值或 Device alpha 地址、format、fillMode、diagType、opA、idxBase、valueStride、SLICED_ELL slice 数和宽度。arch22 当前采用单 AIV 确定性路径，三类 Kernel 均以一个 block 启动，不依赖跨核原子写入顺序。

### Kernel 侧设计

#### Analysis：格式统一和确定性规范化

Analysis 对源格式执行两遍枚举：

1. 第一遍按输入物理顺序枚举坐标，应用 base 归一化；T/H 交换行列；统计每个有效目标行的元素数。
2. 对行计数做前缀和，得到 canonical CSR rowPtr，并复用 diag 数组作为行 scatter cursor。
3. 第二遍按同一物理顺序稳定散射 colInd、values 和 source-position permutation。
4. complex64 H 在写入 canonical values 时对虚部取反。
5. 每行用稳定插入排序按 `(column, source-position)` 排序。目标负载行较短，该方案不额外申请 nnz 级排序缓冲，并保证固定遍历顺序。
6. 重复对角项按稳定顺序求和并写入首个对角位置，其余重复对角 values 置零；重复非对角项保留，Solve 按固定顺序逐项累加。

各格式枚举规则：

| 格式 | 枚举方式 |
| --- | --- |
| CSR | rowPtr 给出物理行范围，colInd 给出列 |
| CSC | colPtr 给出物理列范围，rowInd 给出行，再统一映射到逻辑坐标 |
| COO | 按 rowInd/colInd 的物理位置顺序枚举 |
| SLICED_ELL | 校验每个 slice 的物理槽位，负列坐标视为 padding 并跳过 |

Analysis 记录首个非法状态。压缩格式检查 row/col pointer 连续、单调、范围与 nnz 一致；所有格式检查坐标属于 `[0,m)`。状态异常时 Solve 不访问非法结构，并写出 NaN 防止返回未定义结果。

#### Solve：前代/回代

Solve 先从 Tiling 读取 Host alpha，或从 Device alpha 指针读取标量。随后根据 fillMode 与 opA 选择前代/回代：

```text
sum = alpha * X[row]
for p in canonical row:
    if col is a solved dependency:
        sum -= value[p] * Y[col]
if NON_UNIT:
    sum /= diagonal
Y[row] = sum
```

每一行在写 Y 前先完整读取本行 X；已求解行通过 Y 读取，因此 X/Y 指向同一块 Device values 时仍满足原地求解依赖。UNIT 忽略输入中的显式对角；NON_UNIT 缺失或零对角按 IEEE-754 除零语义传播 Inf/NaN，不做 CPU 侧特殊回退。

#### UpdateMatrix

- GENERAL：通过 Analysis 生成的 source-position permutation，从新 values 的原物理布局刷新 canonical values；H 路径同时共轭；刷新后再次合并重复对角。
- DIAGONAL：输入为 `m` 个连续逻辑对角值，逐行更新 canonical CSR 中已存在的首个对角位置，并将其余重复对角置零。原结构缺失的对角不新增坐标，仍保留缺失对角语义。
- `m=0` 或 `nnz=0` 时不启动更新 Kernel，仅更新描述符状态。

### 确定性设计

确定性来自以下固定规则：

1. 格式枚举使用固定物理顺序。
2. 坐标变换和 base 归一化无原子竞争。
3. 每行按 `(column, source-position)` 稳定排序。
4. 重复对角按稳定顺序累加，重复非对角按稳定顺序参与求解。
5. 前代/回代行顺序和行内累加顺序固定。
6. 单 AIV 路径不依赖运行时跨核调度到达顺序。

因此同一输入、属性和生命周期状态重复执行时输出 bit-wise 一致。

### 性能与内存优化

当前实现以完整语义和确定性为优先，结合目标场景做以下优化：

1. 统一 canonical CSR，使四种格式及 N/T/H 复用同一 Solve 主路径，降低运行时分支。
2. Analysis 采用两遍计数/散射，并复用 diag 区域作为 cursor，避免完整 COO 临时副本。
3. source permutation 允许 UpdateMatrix 只刷新 values，无需重新构建结构。
4. Solve 直接顺序访问 canonical rowPtr/colInd/values，不申请整向量副本。
5. X/Y 原地执行避免额外输出或 alias 临时缓冲。
6. GENERAL Update 与 Solve 均在同一 stream 顺序执行，性能计时不包含 Host/Device 数据生成。

## 支持硬件

| 支持的芯片版本 | 架构目录 | 设计支持 | 当前验证状态 |
| --- | --- | --- | --- |
| Atlas A2 训练系列 910B3 | `sparse/spsv/arch22/` | √ | 编译通过，待真实设备功能/精度/性能日志 |
| Atlas A2 训练系列 910B4 | `sparse/spsv/arch22/` | √ | 编译通过，待真实设备功能/精度/性能日志 |
| Atlas A3 任务环境 9382 | `sparse/spsv/arch22/` | √ | 编译及实机验证通过 |

## 算子约束限制

1. A 必须为二维方阵；fillMode 外元素按三角语义忽略。
2. A/X/Y/alpha/computeType 必须同型，仅支持 FP32、complex64。
3. arch22 仅支持 I32 Device 索引；`m` 和 `nnz` 均不得超过 `INT32_MAX`。
4. 算法仅支持 `ACL_SPARSE_SPSV_ALG_DEFAULT`。
5. externalBuffer 首地址必须 512 B 对齐，并在相关异步任务完成前保持有效且不被外部修改。
6. Analysis 缓存的矩阵 pattern、属性、handle、stream、pointer mode 和已绑定向量不得在 Solve 前改变。
7. UpdateMatrix 只更新 values，不改变 pattern、shape、format 或三角属性；这些内容变化后必须重新 Analysis。
8. A 和非原地 X 保持只读；Y 可与 X 完全同址，不支持部分重叠。
9. Analysis 为返回 Device 结构校验结果会执行一次 4 字节 D2H 回读并同步调用 stream；Solve 和 UpdateMatrix 异步返回。
10. NON_UNIT 缺失或零对角按 IEEE-754 传播 Inf/NaN。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 |
| --- | --- |
| FP32 精度 | CPU Golden 使用 float64；`rtol=2^-10`、`atol=2^-16`，匹配率不低于 0.99，单点绝对误差不超过 `max(1e-2, 32*ULP(golden))` |
| complex64 精度 | CPU Golden 使用 complex128，实部和虚部分别应用 FP32 标准 |
| 确定性 | 同一输入和状态重复执行，输出 bit-wise 一致 |
| 性能 | `GPU Device Event median / NPU 同范围 Kernel median >= 0.25` |
| 采样 | 预热 10 次，正式采样 30 次，报告 median/P90/mean、Analysis、Update、Solve 和 workspace |
| 内存 | 输入输出超过 500 MB 时按 GPU/NPU 额外峰值规则比较，或固有 workspace 不超过目标硬件 L2 Cache |

逐元素判定公式：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

### 当前性能验证结果

当前在 A3 `Ascend910_9382`、CANN 9.1.0 上对任务 P-01/P-02/P-03 执行 Device alpha + GENERAL Update，NPU 总耗时为同一 ACL Event 范围内的 Update + Solve：

| Case | base | Workspace (B) | Analysis (us) | Update median (us) | Solve median (us) | Total median (us) | GPU/NPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P-01：65,536 / 524,288 | 0 | 6,816,256 | 997,005.312 | 189,724.125 | 239,784.531 | 429,404.250 | 1.197 |
| P-01：65,536 / 524,288 | 1 | 6,816,256 | 997,410.125 | 189,738.797 | 239,793.875 | 429,534.719 | 1.197 |
| P-02：131,072 / 1,572,864 | 0 | 26,214,912 | 2,486,452.000 | 773,535.125 | 861,223.375 | 1,634,674.750 | 0.753 |
| P-02：131,072 / 1,572,864 | 1 | 26,214,912 | 2,486,839.000 | 773,857.000 | 861,263.375 | 1,635,070.875 | 0.751 |
| P-03：262,144 / 3,932,160 | 0 | 65,012,224 | 6,122,823.000 | 1,881,706.875 | 2,224,031.500 | 4,106,214.000 | 0.750 |
| P-03：262,144 / 3,932,160 | 1 | 65,012,224 | 6,128,806.500 | 1,880,714.000 | 2,223,758.750 | 4,104,531.000 | 0.750 |

六组实测倍率均超过 0.25 门槛。最大 P-03 workspace 为 65,012,224 B；当前平台配置 `l2_size=201,326,592` B，workspace 占 32.3%，满足任务书允许的 L2 容量分支。

## 自测用例设计

| 类别 | 覆盖内容 |
| --- | --- |
| 功能 | 四格式、base 0/1、LOWER/UPPER、UNIT/NON_UNIT、N/T/H |
| dtype | FP32、complex64；H 用例包含非零虚部、正负混合与抵消 |
| 生命周期 | NULL vec 的 BufferSize/Analysis、多次 Solve、GENERAL/DIAGONAL Update、错序与状态变更 |
| 边界 | `m/nnz=0/1`、空行、未排序、重复项、缺失/零对角、非法索引 |
| 运行语义 | Host/Device alpha、非默认 stream、X/Y 同地址、A/X 只读 |
| 异常 | NULL、Host pointer、dtype/shape/format/alg 不支持、I64、workspace 设备与对齐错误 |
| 性能/内存 | P-01/P-02/P-03，base 0/1，记录 Analysis/Update/Solve 和 workspace |
| Profiler | Analysis/Solve Kernel、AI_VECTOR_CORE task type 和调用 stream |

当前 A3 9382 实机执行 13 个 `SpSVArch22Test`，13/13 PASS；测试把 Device 输出复制回 Host，与 float64/complex128 CPU Golden 对比，并检查输入只读、原地执行和重复执行 bit-wise 一致。已通过的用例包括：

```text
FourFormatsBaseAndTranspose
ComplexConjugateTranspose
GeneralAndDiagonalUpdates
LifecycleRejectsChangedOperation
RejectsInvalidApiArgumentsAndHostPointers
InvalidDeviceSparseStructureIsReportedByAnalysis
LifecycleRejectsChangedBindingsAndInvalidUpdates
EmptyCooUnitDiagonal
RejectsI64AndMisalignedWorkspace
ZeroAndMissingDiagonalFollowIeee754
SingleElementNonUnit
ZeroDimensionLifecycle
UnsortedDuplicatesAreDeterministic
```

## 可维护性分析

1. arch22 Host/Kernel 收敛在 `sparse/spsv/arch22/`，公共 opaque 描述符仅扩展内部状态，不改变用户可见 ABI。
2. 参数检查、workspace 布局、状态缓存、Tiling 构建分别封装，BufferSize 与 Analysis 共用同一布局函数。
3. 四种格式、N/T/H 和两种 dtype 统一到 canonical CSR，Solve 和 Update 主路径复用。
4. 状态码只在 Analysis 阶段回读，异常来源明确；Solve/Update 不引入隐式 Host 计算。
5. 测试按功能、异常、生命周期、边界、精度和性能拆分，便于后续 arch22 优化回归。

## 兼容性分析

- 不修改六个 SpSV 公开接口的名称、参数顺序或枚举语义。
- 新增能力通过现有 `aclDataType`、SpMat/DnVec 和 SpSV 描述符分派。
- arch22 实现与 arch35/A5 Kernel 路径隔离，公共描述符扩展为库内部字段。
- 不删除现有 arch35 FP32/I64 能力；arch22 的 I32 限制在 Host 校验中显式返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。
- 构建已覆盖 `ascend910b3`、`ascend910b4`、当前 A3 `ascend910_9382`，并对 `ascend950` 的 `ops_sparse` 库目标执行兼容性编译。

## 阶段性进展

目前已完成 aclsparseSpSV A2/A3 的公开接口补齐、arch22 Host/Ascend C Kernel、四格式确定性规范化、FP32/complex64、N/T/H、Host/Device pointer mode、原地求解及 GENERAL/DIAGONAL 更新，并完成 C++ 功能/异常/精度测试、性能脚本、接口文档和复现说明。在 A3 9382 实机上 13/13 用例通过，P-01/P-02/P-03 的 6 组 GPU/NPU 性能倍率均高于 0.25，最大 workspace 低于平台 L2 容量；910B3/910B4 已完成编译，后续仍需在真实 A2 设备补齐功能、精度、性能和 Profiler 日志后再完成最终代码验收。

## 参考资料

1. `9月社区任务-aclsparseSpSV算子开发 (A2A3)/aclsparseSpSV_A2A3_task_doc.md`
2. `ops-sparse/include/cann_ops_sparse.h`
3. `ops-sparse/sparse/spsv/arch22/`
4. `ops-sparse/sparse/spsv/README.md`
5. `ops-sparse/test/spsv/arch22/`

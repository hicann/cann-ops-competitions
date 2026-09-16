# aclsparseSpSV 算子设计文档（Ascend 950）

# 需求背景（required）

## 需求来源

本设计对应“9月社区任务-aclsparseSpSV 算子开发（950）”。任务要求基于
`ops-sparse` 当前 `sparse/spsv/arch35/` 实现，在保持公开 Legacy API 不变的
前提下，补齐 Ascend 950 上 FP32、complex64 的功能、生命周期、测试、性能与
文档能力。最终代码提交至 `ops-sparse`，目标硬件为 Ascend 950PR
（DAV_3510/arch35），验收环境为 CANN 9.1.0 及后续配套版本。

## 背景介绍

### 算子功能

`aclsparseSpSV` 求解稀疏三角线性方程：

```text
op(A) · Y = alpha · X
```

其中 `A` 是 `m × m` 稀疏三角矩阵，`X`、`Y` 是长度为 `m` 的稠密向量，
`alpha` 是标量：

```text
NON_TRANSPOSE       : op(A) = A
TRANSPOSE           : op(A) = A^T
CONJUGATE_TRANSPOSE : op(A) = A^H = conj(A^T)
```

FP32 下 `A^H` 与 `A^T` 数值等价；complex64 下必须执行共轭转置，不能将 H
退化为 T。调用流程为：

```text
createDescr -> bufferSize -> analysis -> [updateMatrix] -> solve -> destroyDescr
```

`analysis` 负责格式规范化、依赖分析和层级调度表构造；`solve` 沿依赖层逐层
回代。Host 仅完成校验、状态管理和 Kernel 下发，矩阵规范化、依赖分析、更新和
求解均在 handle 绑定的 NPU stream 上异步执行。

### 当前实现与本次增量

设计基线为 `ops-sparse master` 提交 `c343a33ad88b3489993f09411260d537c0f380f7`。
现有实现已经具备：

- CSR、CSC、COO、SLICED_ELL 四种格式与 base 0/1；
- LOWER/UPPER、UNIT/NON_UNIT、N/T/H 枚举入口；
- I32/I64 索引组合、Host/Device pointer mode、原位 X/Y；
- format conversion、transpose、level scheduling、GENERAL/DIAGONAL 更新；
- 单核 `asc_syncthreads` 与多核 `SyncAll` 的层间同步。

本次主要缺口如下：

| 缺口 | 当前状态 | 设计处理 |
| --- | --- | --- |
| complex64 | Host 仅接受 `ACL_FLOAT`，Kernel values/alpha 固定为 `float` | 增加 POD 复数类型与 ValueT 模板分派 |
| H 语义 | H 与 T 共用“需要转置”判断，未共轭 values | 将“转置”和“共轭”拆为独立标志，Analysis 期处理共轭 |
| workspace | values 段固定按 4 B 计算；复杂数双副本最坏路径逼近 L2 | 按 dtype 定尺，并将最终规范化路径收敛为至多一份 canonical CSR |
| 生命周期绑定 | 未完整缓存 dtype、vec 描述符及全部阶段参数 | 扩展描述符快照与 Solve 一致性校验 |
| 测试 | 既有仓内用例以 FP32 为主 | 增加 complex64、H、生命周期、异常、性能与内存用例 |

SpSV 行长不规则、存在间接 GM 访存和跨行依赖，适合 DAV_3510 SIMT。现有纯
SIMT 分层调度已经与问题结构吻合，因此本次不改为 RegBase/SIMD 主路径，也不
引入 TPipe/TQue 式 UB 搬运流水；复数化通过标量类型参数化完成。

# 需求分析（required）

## 需求描述

在 Ascend 950 上实现功能完整、结果确定、可异步调用的 `aclsparseSpSV`：

1. 支持 `ACL_FLOAT`、`ACL_COMPLEX64`，A/X/Y/alpha/computeType 类型一致。
2. 支持 CSR、CSC、COO、SLICED_ELL，Device 索引满足任务要求的 I32/base 0/1，
   同时保留 master 已有 I64 兼容能力。
3. 支持 LOWER/UPPER、UNIT/NON_UNIT、DEFAULT alg、N/T/H。
4. 支持 Host/Device alpha、X/Y 原位、GENERAL/DIAGONAL update。
5. BufferSize/Analysis 允许 vecX/vecY 描述符为 NULL；Solve 必须提供非 NULL、
   dtype/长度正确且 values 非空的 Device 向量描述符。
6. 从 Analysis 到 Solve 绑定矩阵属性、op、dtype、pointer mode、workspace 和
   已提供的向量描述符；合法 update 后复用分析状态。
7. 接受未排序坐标并产生确定结果，不使用 CPU fallback。
8. 满足任务书精度、性能、内存和可复现要求。

## 需求拆解

### 公开接口

接口签名沿用 `include/cann_ops_sparse.h`，不增加新公开 API：

```cpp
aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr);
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr);
aclsparseStatus_t aclsparseSpSV_bufferSize(aclsparseHandle_t handle,
    aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr,
    size_t *bufferSize);
aclsparseStatus_t aclsparseSpSV_analysis(aclsparseHandle_t handle,
    aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr,
    void *externalBuffer);
aclsparseStatus_t aclsparseSpSV_solve(aclsparseHandle_t handle,
    aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr);
aclsparseStatus_t aclsparseSpSV_updateMatrix(aclsparseHandle_t handle,
    aclsparseSpSVDescr_t spsvDescr, void *newValues,
    aclsparseSpSVUpdate_t updatePart);
```

### 参数与状态约束

| 对象 | 支持范围 | 校验/绑定规则 |
| --- | --- | --- |
| matA | 方阵；四种稀疏格式；values 为 FP32/complex64 | format、m、nnz、fill、diag、base、索引类型、value dtype 在 Analysis 固化 |
| vecX/vecY | 连续 Device 向量，长度 m | BufferSize/Analysis 可为 NULL；Analysis 非 NULL 时缓存描述符和值指针，Solve 要求等价且有效 |
| alpha | Host 或 Device 标量 | 非空；pointer mode 在 Analysis 固化；FP32 4 B，complex64 8 B |
| opA | N/T/H | Analysis 与 Solve 一致；complex64 H 设置 transpose+conjugate |
| computeType | ACL_FLOAT/ACL_COMPLEX64 | 与 matA、vecX、vecY 一致 |
| alg | ACL_SPARSE_SPSV_ALG_DEFAULT | 其它枚举返回不支持 |
| externalBuffer | Device workspace | 非空（nnz=0 除外）、满足 512 B 对齐；描述符缓存其地址 |
| updatePart | GENERAL/DIAGONAL | GENERAL 输入 nnz 个值；DIAGONAL 输入 m 个值；不改变 pattern |

由于公开 `analysis` 只有 workspace 指针、没有实际分配字节数，Host 能校验非空和
对齐，不能可靠探测外部缓冲真实容量；调用方必须按同一参数调用 `bufferSize` 并
分配不少于返回值的 Device 内存。

当 Analysis 的 vecX/vecY 为 NULL 时，Solve 必然传入新描述符，因此一致性规则
采用“Analysis 非 NULL 才绑定描述符身份；NULL 时在 Solve 首次校验 dtype、长度、
layout 和 values”的条件绑定方式，避免与“Analysis 允许 NULL”的接口要求冲突。

### 数值与特殊场景

| 场景 | 语义 |
| --- | --- |
| UNIT | 对角视为 1，不读取存储对角值 |
| NON_UNIT 缺失/零对角 | 按 IEEE-754 除零传播 INF/NAN，不在 Host 修补数值 |
| fill 之外的项 | 不参与三角求解 |
| nnz=0, UNIT | `Y = alpha * X` |
| nnz=0, NON_UNIT | 通过 Device Kernel 传播奇异矩阵 INF/NAN |
| 未排序输入 | Analysis 按固定扫描次序写入 canonical CSR；同输入重复运行 bitwise deterministic |
| 重复坐标 | 不做数值归并，保持稳定输入次序；有效三角项按固定次序参与累加 |
| X 与 Y 同址 | 每行读取本行 X 后再写 Y；依赖行只读取已完成层的 Y |

# 详细设计（required）

## 算子分析

### 求解方向与依赖层

将四种输入格式映射为 `op(A)` 的 0-based canonical CSR。LOWER 采用前向回代，
UPPER 采用后向回代；转置会翻转有效 fill。对每行 `r`，仅将求解方向上已完成的
列作为依赖：

```text
level[r] = 1 + max(level[c]), c 为 r 行中参与求解的非对角依赖列
```

无依赖行位于 level 0。Analysis 生成 `levelPtr`、`levelRow`、`diagPtr`、
`validCount`；Solve 按 level 串行，层内每行独立并行。

### complex64 数据表示

Device values、X、Y 与 alpha 使用两个连续 FP32 分量：

```cpp
struct SpsvComplex64 {
    float real;
    float imag;
};
static_assert(sizeof(SpsvComplex64) == 8);
```

设备辅助函数提供逐分量 `Load/Store/Add/Sub/Mul/Div/Conj`，使用 SDK 支持的 SIMT
callee 限定符；不依赖未核实的 Device `std::complex` ABI。复乘与复除为：

```text
(a+bi)(c+di) = (ac-bd) + (ad+bc)i
(a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2)
```

输入分布位于任务给定范围，使用 FP32 标量运算。分母为零时保留 IEEE-754 的
INF/NAN 传播；不使用 epsilon 替换零对角。

## 算子实现

### Host 侧设计

#### 校验和状态机

统一校验顺序：handle -> opA -> matA/spsvDescr -> computeType/alg -> 矩阵属性 ->
阶段指针。`aclsparseSpSVDescr` 增加以下快照：

```text
cachedValueType / cachedValueSize
cachedPointerMode
cachedMatDescr / cachedVecXDescr / cachedVecYDescr
cachedVecXValues / cachedVecYValues
workspaceBuffer / workspaceSize
analysisLaunched / updateMatrixCalled
```

Solve 校验当前调用与快照一致；vec 描述符采用前述条件绑定规则。UpdateMatrix
只更新 values 及关联 workspace，不能绕过 format/op/dtype/pattern 校验。

#### op 与格式规范化

Host 将公开 op 拆成两个正交标志：

```text
needsTranspose = opA in {T, H}
needsConjugate = valueType == ACL_COMPLEX64 && opA == H
```

CSC 按 CSR 视图解释时，相当于先做一次转置，因此翻转 `needsTranspose`；
`needsConjugate` 保持不变。典型组合：

| 输入格式/op | canonical 结构操作 | values 操作 |
| --- | --- | --- |
| CSR + N | 可直接引用或 base 归一化 | 原值 |
| CSR + T | 构造转置 CSR | 原值 |
| CSR + H complex64 | 构造转置 CSR | 写入时共轭 |
| CSC + N | 构造/引用等价转置视图 | 原值 |
| CSC + T | 构造/引用原矩阵视图 | 原值 |
| CSC + H complex64 | 结构无需再次转置 | 强制 values 拷贝并共轭 |

#### Workspace 设计

首地址按 512 B 对齐，内部段按 64 B 对齐。布局为：

```text
[runtime tiling/header]
[diagPtr: m * permSize]
[levelPtr: (m+1) * 4]
[levelRow: m * 4]
[validCount: m * 4]
[canonicalRowPtr: (m+1) * rowPtrSize]  条件分配
[canonicalColInd: nnz * colIndSize]    条件分配
[canonicalValues: nnz * valueSize]     条件分配
[canonicalPerm: nnz * permSize]        条件分配
```

`bufferSize` 与 Analysis 的 offset 构造调用同一纯函数，并对每次乘法、加法和
AlignUp 执行 `size_t` 溢出检查。`valueSize` 为 FP32 4 B、complex64 8 B。

为覆盖不同 950 SKU 的 L2 容量，最终有效矩阵至多材料化一份 canonical CSR：

- CSR/CSC 的 base 归一化、转置和 H 共轭直接写最终 canonical；
- COO/SELL 计数阶段直接以最终行坐标（转置时为输入列）计数，再稳定散射到最终
  canonical，避免“先转 CSR、再复制转置 CSR”的双副本；
- 无需变换的 CSR base0 N 路径直接引用用户输入，不分配 canonical values；
- `canonicalPerm` 记录最终非零项到原 values 的稳定映射，供 UpdateMatrix 复用。

按任务最大 P-03（m=262144、nnz=3932160、I32、complex64）估算，单 canonical
副本约 65 MiB（含 4 MiB 层级元数据），低于 112 MiB 与 128 MiB 两类典型
Ascend950PR L2；实际验收使用平台信息记录目标 SKU L2，不在代码中硬编码容量。

#### Tiling 数据

在现有 `SpsvTilingData` 上最小增量：

```text
alphaReal / alphaImag
valueType / valueSize
opFlags: bit0=transpose, bit1=conjugate
canonical*Offset / permType
m / nnz / format / fillMode / diagType / index types / base
numSlices / sliceWidth / nthreads / numBlocks
```

Host alpha 直接写 `alphaReal/alphaImag`；Device alpha 保存 Device 地址，Kernel
按 dtype 读取一个或两个 FP32 分量。Tiling 按值传给 Kernel；运行时层数仍由
Analysis 写入 workspace header，Solve 从 workspace 读取。

### Kernel 侧设计

#### Analysis

Analysis 分为以下 Device 阶段：

1. 初始化 row count、level、diagPtr、validCount；
2. 依据 format/opFlags 将输入稳定转换为最终 canonical CSR；
3. complex64 H 在最终 values 写入点执行 `imag = -imag`；
4. 扫描 canonical CSR，过滤 fill 外元素、定位对角并计算依赖 level；
5. 统计每层行数并生成 `levelPtr/levelRow`；
6. 将 `numLevels` 写入 workspace header。

格式转换和转置均在 NPU Kernel 完成。计数、prefix 与散射使用固定阶段和固定输入
遍历顺序；不使用对 values 的不确定顺序原子累加。

#### Solve

`ValueT` 为 `float` 或 `SpsvComplex64`，索引模板保持 master 的组合。核心逻辑：

```cpp
ValueT rhs = alpha * Load(vecX + row);
for (p = rowBegin; p < rowEnd; ++p) {
    col = colInd[p];
    if (IsDependency(row, col, direction)) {
        rhs -= Load(values + p) * Load(vecY + col);
    }
}
if (diagType == NON_UNIT) {
    rhs = rhs / Load(values + diagPtr[row]);
}
Store(vecY + row, rhs);
```

单核路径每层使用 `asc_syncthreads`，多核路径每层使用 `SyncAll`，并保持
`SetScheduleMode(1)` 要求。层内行互不依赖，一线程一行；超长行协作仅作为
Profiler 证明瓶颈后启用的泛化优化，不按测试编号或固定 m/nnz 分支。

SIMT 线程数使用编译期常量，`LAUNCH_BOUND` 与 `VF_CALL Dim3` 一致；Host 只在
已编译的线程规格间选择，不把运行时任意值直接作为 VF 线程数。DCache/Local
Memory 按 DAV_3510 平台要求配置，不硬编码核数。

#### UpdateMatrix

- GENERAL：通过 `canonicalPerm[p]` 从 newValues 读取对应原始项，写回全部
  canonical values；H 路径写入时共轭。
- DIAGONAL：newValues 视为长度 m 的逻辑对角数组，通过 diagPtr 定位写回；UNIT
  模式保持对角不参与数值读取，但缓存状态仍一致更新。
- 直接引用输入 values 的路径更新描述符当前 values 指针；材料化路径只更新
  workspace values，不重新计算 pattern/level。

#### 空输入与原位

`m=0` 不启动无意义 Kernel；`nnz=0` 使用类型化 `scale_copy/scale_inf` Kernel。
原位时每个线程先将本行 `X[row]` 读入寄存器，再读取依赖行 Y，最后写本行 Y；
层间同步保证后继层可见。

### 支持硬件

| 芯片 | 架构目录 | 支持 |
| --- | --- | --- |
| Ascend 950PR | DAV_3510 / `arch35` | 是 |

实际核数、Local Memory 和 L2 由平台查询或验收环境记录，不按某一 SKU 硬编码。

### 算子约束限制

1. A 为二维方阵；仅 fill 指定的三角部分参与计算。
2. A/X/Y/alpha/computeType 必须同为 FP32 或同为 complex64。
3. 任务要求 Device I32 索引；保留 master 已有的合法 I64 兼容组合。
4. `externalBuffer` 由调用方按 bufferSize 分配，并保持到异步 Solve 完成。
5. vecX/vecY 为长度 m 的连续 Device 向量；允许 values 地址相同。
6. 不合并重复坐标；按稳定输入次序参与计算。
7. 不使用 Host 同步求解或 CPU fallback。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 验证方式 |
| --- | --- | --- |
| FP32 精度 | CPU float64 golden；rtol=2^-10，atol=2^-16，matched ratio>=0.99，且 max abs error<=max(1e-2,32 ULP) | 逐元素混合容差与硬上限 |
| complex64 精度 | CPU complex128 golden；实部、虚部分别执行 FP32 标准 | 分量化统计，不只使用单次 allclose |
| 确定性 | 同一输入重复求解 bitwise 一致 | 固定输入连续执行并逐字节比较 |
| 性能 | GPU Event / NPU 同范围耗时 >=0.3 | warmup 10、sample 30，报告 median/P90/Analysis/Solve |
| 内存 | 任务书两条件满足其一；本任务优先验证 workspace<=目标 L2 | 记录 bufferSize、平台 L2 与峰值内存 |
| 异步 | Host 不插入无要求同步，Kernel 位于 handle stream | Profiler 与多 stream 用例 |

任务书 P 场景换算后的 Solve median 上限：

| 场景 | m / nnz | GPU median_us 范围 | NPU 严格上限 |
| --- | --- | --- | --- |
| P-01 | 65,536 / 524,288 | 40,413.695～40,458.531 | <=134,712 us |
| P-02 | 131,072 / 1,572,864 | 93,850.750～94,739.203 | <=312,836 us |
| P-03 | 262,144 / 3,932,160 | 220,676.516～222,887.562 | <=735,588 us |

最终性能结论必须使用 GPU/NPU 相同调用范围；Analysis、Update 与 Solve 分项记录，
不能用“GPU update+solve”对比“NPU solve only”。任务包内不同文档或脚本若存在
口径差异，保留原始记录并以任务方书面确认后的口径判定。

## 测试设计

| 类别 | 覆盖内容 |
| --- | --- |
| 功能 | format × base × fill × diag × N/T/H × FP32/complex64 |
| 生命周期 | create、NULL vec 的 bufferSize/analysis、solve-before-analysis、solve、两种 update、destroy |
| pointer mode | Host/Device alpha，空指针、错位 Device alpha |
| 原位 | X/Y 分离与同 values 地址，链式依赖和宽层依赖 |
| 边界 | m/nnz=0/1、空行、长行、缺失/零对角、非法索引、溢出尺寸、workspace 对齐 |
| 稳定性 | 未排序、重复坐标、不同随机种子、重复运行 bitwise 比较 |
| 精度 | 普通值、抵消、近零、INF/NAN、UNIT、GENERAL/DIAGONAL update 后结果 |
| 性能 | 206 条性能集及 P-01/P-02/P-03，FP32/complex64、base 0/1 |
| 兼容回归 | master 既有 SpSV FP32/I32/I64 用例；触及 shared header 时执行关联算子构建回归 |

测试采用两条链路：

1. `test/spsv/arch35/` C++ GTest/CSV，直接调用公开 C API；
2. 任务包 Python benchmark/accuracy/memory 脚本，通过明确注册的 NPU hook 调用同一
   C API，不以 Python/CPU 实现替代 NPU Kernel。

仓内新增 complex128 CPU 回代 golden。普通值逐元素执行混合容差；INF/NAN 按
分类和符号语义比较。性能和内存结果关联同一冻结 Commit、同一 CANN/驱动/SOC。

## 兼容性分析

- 不修改 `aclsparseSpSV_*` 公开签名、枚举值和 ABI。
- FP32 路径保持原有算法和索引实例化；dtype 分支仅在 Kernel 入口发生一次。
- `SpsvTilingData` 为 Host/Kernel 内部按值结构，新增字段由同一版本共同编译。
- `aclsparseSpSVDescr` 是 SpSV 专属不透明状态，新增缓存字段不暴露给调用方。
- 保留 I64 支持，新增 complex64 后回归 SLICED_ELL/I64 等最大模板实例路径。
- 设计与实现目标为 CANN 9.1.0+；较早 SDK 仅用于早期编译诊断，不作为验收证据。

计划改动范围：

```text
sparse/common/aclsparse_spsv_descr.h
sparse/spsv/arch35/spsv_complex.h              [新增]
sparse/spsv/arch35/spsv_host.cpp
sparse/spsv/arch35/spsv_kernel.cpp
sparse/spsv/arch35/spsv_tiling_data.h
test/spsv/spsv_complex_golden.h                 [新增]
test/spsv/arch35/spsv_test.cpp
test/spsv/arch35/spsv_test.csv
test/spsv/CMakeLists.txt
sparse/spsv/README.md
```

开发完成后交付设计文档、源码、C++ UT/ST、任务包 NPU 适配、测试 README、自测
报告、Profiler 证据，以及个人 fork 的固定分支/Commit。所有“通过”结论必须来自
最终冻结 Commit 在 CANN 9.1.0+ Ascend 950 环境的实际执行结果。

# 需求背景（required）

## 需求来源

8月社区任务-aclsparseSpMM 算子开发(A2/A3)。

## 背景介绍

### 现状

PyTorch 的 `torch.sparse.addmm(input, mat1, mat2, *, beta=1, alpha=1)` 分派到
`aten::_sparse_addmm`，计算稀疏矩阵与稠密矩阵的乘加。该算子在昇腾 NPU 后端为空白。

`ops-sparse` 仓已有 `aclsparseSpMM*` 三阶段 C++ 接口及对应 Ascend C Kernel，以及 Handle、稀疏矩阵
描述符、稠密矩阵描述符与资源销毁接口。

### 差距

| 层次 | 现状 | 本任务需要补齐 |
| --- | --- | --- |
| Python / ATen | NPU 上无 `aten::_sparse_addmm` 实现 | 注册 NPU 实现，完成参数校验、元数据转换与输出构造 |
| aclsparse C++ | 三阶段接口已有实现 | 补齐 A2/A3 上四种数据类型在各布局与算法下的 workspace、预处理与执行能力 |
| Ascend C Kernel | 实数路径已有实现 | 补齐 `complex64` 的布局、精度与泛化能力 |

`complex64` 是本任务的必选能力，需打通 Python、ATen、aclsparse 与 Kernel 全链路。

# 需求分析（required）

## 需求描述

在昇腾 NPU 上实现 `aten::_sparse_addmm`，使 `torch.sparse.addmm` 在 NPU 上的接口、返回值、dtype、
shape、device 与异常行为与 PyTorch 2.7 及以上版本保持一致；复用并扩展 `ops-sparse` 已有的
`aclsparseSpMM*` C++ 接口及 Ascend C Kernel，补齐 Atlas A2 系列产品与 Atlas A3 系列产品上
`float16`、`bfloat16`、`float32` 与 `complex64` 在各声明布局、操作与算法组合下的能力。核心计算在 NPU 上
完成，不使用 CPU fallback。

## 需求拆解

1. 注册 `aten::_sparse_addmm` 的 NPU 实现，覆盖 COO 与 CSR 稀疏 `mat1`。
2. 复用已有 `aclsparseSpMM*` 三阶段接口与描述符接口，保持签名与调用流程兼容。
3. 补齐四种数据类型在 Row-major / Column-major、`opB` 三种取值、五种算法与两种索引基下的能力。
4. `input` 按 PyTorch 语义广播；`beta = 0` 时不读取也不传播 `input` 中的 NaN/Inf。
5. 支持规格内的 `M`、`K`、`N` 与 `nnz` 动态变化，覆盖零 nnz、空行、零长度维度与长尾行分布。
6. 精度按《生态算子开源精度标准》的混合容差单标杆方法验收。
7. C++ 接口使用调用方 stream，避免无必要的 Host 同步；满足确定性计算要求。
8. 与同期分发的 A5 任务共存：公共 Host 逻辑与硬件差异解耦，两套代码在同一主干并行。

# 详细设计（required）

## 算子分析

### 数学公式

ATen 层：

$$
\mathrm{out} = \beta \cdot \mathrm{input} + \alpha \cdot (\mathrm{mat1} \times \mathrm{mat2})
$$

`input` 为参与加法的稠密矩阵，`mat1` 为 `[M,K]` 稀疏矩阵，`mat2` 为 `[K,N]` 稠密矩阵，`out` 为
`[M,N]` 稠密矩阵。

C++ 层按 cuSPARSE SpMM 语义原地更新 C：

$$
C = \alpha \cdot \mathrm{op}(A) \times \mathrm{op}(B) + \beta \cdot C
$$

按行展开：C 的第 $i$ 行是 A 第 $i$ 行的每个非零元 $a_{ik}$ 与 op(B) 第 $k$ 行整行的乘积之和，再叠加
$\beta$ 倍的 C 旧值。

$$
C_{i,:} = \alpha \sum_{k \in \mathrm{nz}(A_{i,:})} a_{ik} \cdot \mathrm{op}(B)_{k,:} + \beta \cdot C_{i,:}
$$

因此每行的计算是一串「读 B 的一行、乘一个标量、累加」，行与行之间互不依赖。

### 支持数据类型

| 数据类型 | `aclDataType` | Golden 计算类型 |
| --- | --- | --- |
| float16 | `ACL_FLOAT16` | float32 |
| bfloat16 | `ACL_BF16` | float32 |
| float32 | `ACL_FLOAT` | float64 |
| complex64 | `ACL_COMPLEX64` | complex128 |

`input`、`mat1` 的 values 与 `mat2` 必须同 dtype，不做 Tensor 间 dtype 提升，输出沿用该 dtype。
`computeType` 取该 valueType，fp16/bf16 时也接受 `ACL_FLOAT`。A 的索引类型为 `ACL_SPARSE_INDEX_32I`。

A、B、C 与算法的组合仅覆盖 cuSPARSE 13.3 Update 1 的 SpMM 官方支持矩阵中允许的组合。

### 支持形状与边界

| 项 | 范围 |
| --- | --- |
| 维度 | `mat1`、`mat2` 均按二维计算，`mat1.size(1) == mat2.size(0)` |
| 广播 | 仅 `input` 按 PyTorch 语义广播为 `[M,N]`，如 `[M,N]`、`[N]`、`[1,N]`、`[M,1]`；`mat1`、`mat2` 不广播 |
| 空输入 | `nnz=0`、空行、零长度维度均为合法输入 |
| leading dimension | Row-major 要求 `ld >= cols`，Column-major 要求 `ld >= rows` |
| 索引基 | `ACL_SPARSE_INDEX_BASE_ZERO` 与 `ACL_SPARSE_INDEX_BASE_ONE` |
| 规模上限 | A 的 rows、cols、nnz 与 B/C 的维度、leading dimension 不大于 `INT32_MAX - 1` |

## 算子原型

### C++ 三阶段接口

接口命名、签名与调用流程沿用 `include/cann_ops_sparse.h` 中已有的 `aclsparseSpMM*` 声明。

```C
/* Workspace 查询 */
aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    size_t *bufferSize);

/* 预处理（可选） */
aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *externalBuffer);

/* 计算执行 */
aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *externalBuffer);
```

各阶段职责与本任务补齐范围：

| 阶段 | 接口 | 本任务补齐范围 |
| --- | --- | --- |
| Handle 管理 | `aclsparseCreate`、`aclsparseDestroy`、`aclsparseSetStream` | A2/A3 上的功能验证 |
| 稀疏矩阵描述符 | `aclsparseCreateCsr`、`aclsparseCreateConstCsr`、`aclsparseDestroySpMat` | 四种类型、索引与异常处理能力 |
| 稠密矩阵描述符 | `aclsparseCreateDnMat`、`aclsparseCreateConstDnMat`、`aclsparseDestroyDnMat` | 四种类型、布局与异常处理能力 |
| Workspace 查询 | `aclsparseSpMMGetBufferSize` | A2/A3 上四种类型各布局与算法对应的 workspace 计算 |
| 预处理 | `aclsparseSpMMPreprocess` | A2/A3 上四种类型各布局与算法能力 |
| 计算执行 | `aclsparseSpMM` | A2/A3 上四种类型的布局、精度与泛化能力 |
| 资源释放 | `aclsparseDestroySpMat`、`aclsparseDestroyDnMat`、`aclsparseDestroy` | 完整调用链无资源泄漏 |

关键参数语义：

| 参数 | 说明 |
| --- | --- |
| `opA` | 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE` |
| `opB` | 支持 NON_TRANSPOSE、TRANSPOSE、CONJUGATE_TRANSPOSE；共轭对实数类型等同于转置 |
| `alpha` / `beta` | 指针内存位置由 `aclsparseSetPointerMode` 控制；实数按 float 读取，`complex64` 按 `aclsparseComplex` 读 8 字节 |
| `alg` | DEFAULT、CSR_ALG1、CSR_ALG2、CSR_ALG3、CSR_FP32_HIGH_PRECISION_ALG |
| `externalBuffer` | 按 `GetBufferSize` 返回值分配的 Device workspace |
| `matC` | 输入/输出，`beta != 0` 时其旧值参与计算 |

`Preprocess` 可选：它校验 CSR 结构并把校验结果与当前 workspace 关联，成功后同一 matA 与同一 workspace
可供后续多次 `aclsparseSpMM` 复用；调用 `aclsparseCsrSetPointers` 后须重新预处理。未预处理或传入不同
workspace 时，`aclsparseSpMM` 在下发 kernel 前执行同样的结构校验。

### ATen 接口

```text
aten::_sparse_addmm(
    Tensor self,
    Tensor mat1,
    Tensor mat2,
    *,
    Scalar beta=1,
    Scalar alpha=1
) -> Tensor
```

Python 公开接口 `torch.sparse.addmm(input, mat1, mat2, *, beta=1, alpha=1)`。

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | shape | 非连续 Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| input / self | 输入 | 参与加法的稠密矩阵 | 须能按 PyTorch 广播规则扩展为 `[M,N]`，否则报错 | float16、bfloat16、float32、complex64 | Dense | `[M,N]` 或可广播形状 | 支持；无法直接描述时生成连续副本 |
| mat1 | 输入 | 稀疏矩阵 A | — | values 支持四种类型；索引为 int32 | CSR | `[M,K]` | 不适用 |
| mat2 | 输入 | 稠密矩阵 B | Row-major 要求 `ld >= cols`，Column-major 要求 `ld >= rows` | 四种类型 | Row-major 或 Column-major | `[K,N]` | 支持 |
| beta | 属性 | `input` 缩放系数 | 默认 1；`beta=0` 时忽略 `input` 数值，其 NaN/Inf 不传播到输出 | ATen Scalar | — | — | — |
| alpha | 属性 | `mat1×mat2` 缩放系数 | 默认 1；实数 dtype 不接受虚部非零的复数 | ATen Scalar | — | — | — |
| output | 输出 | 计算结果 | dtype 与三个输入的共同 dtype 相同，device 相同 | 四种类型，不做 dtype 提升 | Dense | `[M,N]` | — |

## 算子实现

### 总体架构

```
GetBufferSize ──> 参数与结构约束校验，按布局与算法算出 workspace 大小
Preprocess    ──> CSR 结构校验（同步 Device-to-Host 索引复制），与 workspace 关联
SpMM          ──> [可选] 重排 kernel: 生成 op(B) 的行主序副本
                  主 kernel:  C = alpha * A * op(B) + beta * C
                  [可选] 重排 kernel: C 副本写回列主序
```

`SpMM` 下发 kernel 后立即返回，调用方自行同步。

### 分核与 Tiling

`blockIdx` 拆成行组与列块两个维度。

**行方向**按工作量均衡切分。每个核在 CSR 的 `rowOffsets`（本身即行前缀和）上二分查出自己的行区间，
工作量同时计入非零元个数与每行的固定开销（累加器清零、组间标志等待、一次结果写回）；只按非零元均衡会让
行数多的核落后。划分产生连续行区间，因此不需要行重排表，workspace 中不存放任何划分结果。

**列方向**用于在行数少于核数时把空闲的核利用起来，不用于均衡长尾：同一行组内的每个核都要走完该组的全部
非零元，只是各自搬运 B 的不同列段，因此列块数越多，MTE2 的指令条数按同样倍数增长而每条搬运的数据量等比
缩小，总搬运量不变。行数足够时列块数取 1。

`ldb`、`ldc` 与索引基都作为寻址参数进入 Tiling，不产生额外分支。

Tiling 随 kernel 按值下发，不经过 workspace。它的内容是 Host 侧整数运算结果加上本次调用的 alpha/beta。
走 workspace 就要在每次 `aclsparseSpMM` 之前插一次同步的小包 Host-to-Device 拷贝，该拷贝时延与问题规模
无关，在小规模用例上会成为主要开销；按值下发同时省掉每个核 Init 阶段对 Tiling 各字段的全局内存标量读。

### Workspace 设计

workspace 只用于布局重排产生的中间副本：`op(B)` 需要行主序副本时按 `[K,N]` 计入，C 为列主序且
`beta != 0` 时按 `[M,N]` 计入。两者都不需要时 workspace 为常数大小。分核不产生需要落盘的结果，
`GetBufferSize` 的返回值因此只随布局与算法组合变化。

### Host 侧设计

**参数校验**：句柄与描述符非空；matA 为 CSR；索引类型均为 `ACL_SPARSE_INDEX_32I` 且相同；`opA` 为
NON_TRANSPOSE；A、B、C 的 valueType 相同且在四种类型内；维度满足 `A.cols == op(B).rows`、
`A.rows == C.rows`、`op(B).cols == C.cols`；leading dimension 满足对应布局的下界；算法与布局、`opB`
的组合在声明范围内（`CSR_ALG3` 不支持 `opB = CONJUGATE_TRANSPOSE`）。未声明组合返回明确错误。

**结构校验**：CSR 行偏移须以索引基值起始、单调非降并以 `nnz + base` 结束；列索引须位于
`[base, cols + base)`。该校验在 `Preprocess` 中通过一次同步的 Device-to-Host 索引复制完成，使结构错误在
kernel 启动前就能返回确定错误码。

**空操作**：输出行数或列数为 0 时，完成参数与结构校验后按空操作返回。

### Kernel 侧设计

**预取流水**

B 的行按组搬进 UB：一组是连续的若干个非零元，组内的搬运指令连续发出、中间不插同步标志，组末一次标志
等待覆盖整组，让这些访存彼此重叠。两个半区交替，发下一组与算这一组错开一拍。

预取沿本核的非零元流连续推进，不在行边界重置。消费仍按行进行：半区里剩下的份数不够当前行用完就换半区，
半区被读完才归还，因此一个半区可以被相邻几行接着用。若改成每行重建流水，行内非零元不足半组时该行就只有
一组，代码会「发出这一组、紧接着等它」，B 的一次访存延迟在每行上完整暴露；这一项开销与 N 无关，长尾行
分布下尤其显著。

`beta` 项要读的 C 旧值在开行时发出、收行时才等，整行的乘加就是它的延迟掩盖；它占独立的 UB 块，不占 B 的
半区，否则会盖掉已预取的分片。行结果也先落到独立的输出块再写回，累加器因此与写回解耦，下一行的清零不必
等搬出把它读完。

**元素类型**

fp32、fp16、bf16 与 fp32 高精度算法由同一个 kernel 模板实例化，元素类型只决定三件事：B 分片在 UB 上的
元素宽度、把一份 B 乘上标量累加进 fp32 累加器的那条指令、结果写回前是否需要 Cast。划分、预取与同步代码
完全共用，运行期没有类型分支。

累加一律在 fp32 上进行，与 `computeType` 无关。bf16 没有可用的 `Axpy<float, bfloat16_t>` 组合，因此每
半组数据搬进 UB 后整体 Cast 一次，而不是每个非零元 Cast 一次。`alpha` 提到行末统一乘，内层每个非零元的
标量就是 A 的原值，既省掉一次乘法，也不会把 `alpha` 先舍入到存储类型。

`complex64` 走单独的 kernel：它在交织域上计算，把虚轴算子提到行末，与实数路径的取舍不同。

`ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG` 按接口说明做 Kahan 补偿求和，抑制长行上的舍入与抵消
误差，代价是每个非零元由 1 条 Axpy 变成 5 条向量指令。DEFAULT、CSR_ALG1 与 CSR_ALG2 使用同一数值实现；
CSR_ALG3 提供确定性结果，但不支持 `opB = CONJUGATE_TRANSPOSE`。五种算法均不改变 CSR 数据。

**稠密矩阵排布**

主 kernel 只按行主序寻址：`op(B)[k][j]` 在 `k*ldb + j`，`C[i][j]` 在 `i*ldc + j`。描述符的 `order` 与
`opB` 组合出四种情形，其中行主序不转置、列主序转置两种天然满足这个形状；另外两种等价于「元素 `(r,c)` 在
`r + c*ld`」，先由重排 kernel 生成行主序副本，主 kernel 因此只有一条数据通路。

重排 kernel 按 64×64 分块：块从列主序读进 UB 时每列本来就连续，一条 `DataCopyPad` 搬完整块；块内转置用
一次 Gather 完成；转置后每行在 UB 上连续，再一条 `DataCopyPad` 写回。读写两侧的突发长度都是 64 个元素，
不会退化成 32 字节以下的碎片搬运。C 为列主序且 `beta != 0` 时先把 C 读成行主序副本，算完再写回列主序。

### 数值策略

- 累加一律在 fp32 上进行（`complex64` 为两路 fp32），与存储宽度无关；fp16/bf16 在写回前 Cast。
- `beta = 0` 时不读取 C 的旧值，`input` 中的 NaN/Inf 不进入输出；此时仍校验 `input` 的 shape、dtype 与
  device。
- 相同输入重复执行时，分核方式与行内累加顺序由 CSR 结构与 Tiling 唯一确定，不依赖运行期调度，结果按位
  一致。
- 共轭转置对实数类型等同于转置；`complex64` 下按共轭语义处理 B 的元素。

### PyTorch / ATen 设计

适配层把 aclsparse 接口注册为 `PrivateUse1` 后端实现，`import` 即完成注册，不导出额外 Python API。

| ATen 算子 | dispatch key |
| --- | --- |
| `aten::_sparse_addmm` | `SparsePrivateUse1`、`SparseCsrPrivateUse1` |
| `aten::addmm`、`aten::addmm.out` | `SparsePrivateUse1` |

注册范围依据目标 torch_npu 版本的 dispatch 表确定：`aten::_sparse_addmm` 是
`CompositeExplicitAutograd` 且无 backend 注册，`aten::addmm.out` 在稀疏 NPU key 上为空白，
`aten::addmm` 已有 `SparseCsrPrivateUse1` 注册。因此在 `SparsePrivateUse1` 上注册三个入口，在
`SparseCsrPrivateUse1` 上补 `_sparse_addmm`。

- **输入归一**：`mat1` 为 COO 时转换为 CSR；稠密侧必须是二维，非紧凑输入先生成连续副本。
- **标量转换**：`alpha`/`beta` 按 dtype 转成 Host 标量字节，`complex64` 用 `aclsparseComplex` 的
  `{real, imag}` 表示。实数 dtype 拒绝虚部非零的复数标量。
- **输出构造**：`aclsparseSpMM` 原地更新 C，因此 `out` 在调用前先 `copy_(self)`（`out` 与 `self` 为同一
  张量时跳过），使 `beta * input` 项以 C 的旧值形式进入计算。`input` 的广播在这一步按 PyTorch 语义完成。
- **调用序列**：GetBufferSize → Preprocess → SpMM，计算 kernel 提交到当前 NPU stream；`Preprocess` 在
  Host 侧同步校验 CSR 结构。
- **共享入口路由**：`aten::_sparse_addmm` 同时是稀疏×稀疏乘法在当前 PyTorch 版本下的入口，因此该入口
  按第二个乘数的 layout 分派：稠密乘数走本算子，稀疏乘数转交同一扩展中的 SpGEMM 实现。稠密乘数路径的
  参数处理与计算流程不受该分派影响。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品 | √ |
| Atlas A3 系列产品 | √ |

两系列均映射到 arch22，共用同一份 Kernel 实现。本任务与同期分发的 A5 任务共同修改 SpMM 的 Host 侧公共
代码，公共逻辑与硬件差异按目录与分支解耦，保证两个硬件范围的代码在同一主干共存。

## 算子约束限制

| 项 | 约束 |
| --- | --- |
| 稀疏格式 | 仅支持 CSR，不新增其他稀疏格式 |
| 布局 | B、C 支持 `ACL_SPARSE_ORDER_ROW`（`ld >= cols`）与 `ACL_SPARSE_ORDER_COL`（`ld >= rows`） |
| 操作类型 | `opA` 仅 NON_TRANSPOSE；`opB` 支持 N/T/H；组合限于 cuSPARSE 13.3 Update 1 官方支持矩阵，未声明组合返回明确错误 |
| 算法 | CSR_ALG2 优先按行主路径对齐，CSR_ALG1 优先按列主路径对齐，CSR_ALG3 仅支持 CSR 且 `opA` 为 NON_TRANSPOSE、不支持 `opB` 为 CONJUGATE_TRANSPOSE |
| 索引 | 仅 `ACL_SPARSE_INDEX_32I` 且两者相同；`idxBase` 支持 0 和 1；不支持 `ACL_SPARSE_INDEX_64I`；非法索引返回确定错误 |
| 维度 | `mat1.size(1) == mat2.size(0)`；仅 `input` 广播 |
| dtype | 三个输入同 dtype，不做 Tensor 间提升；输出沿用该 dtype |
| device | 三个输入位于同一 NPU 设备，跨设备输入报错 |
| alias 与 in-place | 按 PyTorch 语义处理，不发生未声明的输入覆盖 |
| 非连续 Tensor | 按参数表列出的范围支持，未支持场景返回明确错误 |
| 融合 | 作为独立稀疏算子实现，不要求图融合 |
| 异步执行 | `aclsparseSpMM` 下发后立即返回，调用方自行同步；除 `Preprocess` 的结构校验外不做 Host 同步 |
| 确定性 | 满足确定性计算要求 |
| 规模限制 | 最大 shape、nnz、leading dimension、workspace 与索引溢出边界在接口文档中明确 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差单标杆 | 《生态算子开源精度标准》、任务书精度要求 |
| 性能标准 | 每个 case×dtype 场景倍率 > 0.25×A100，全部场景算术平均 ≥ 0.35×A100 | 任务书性能要求 |

**精度判据**：以 CPU Golden 为单标杆，fp16/bf16 用 float32 计算 Golden，fp32 用 float64，`complex64` 用
complex128。逐元素按 `|actual - golden| ≤ atol + rtol × |golden|` 判定，整体匹配率不低于 0.99，且每个
元素的绝对误差不超过 `max(A, 32 × ULP(golden))`。

| dtype | rtol | atol | A |
| --- | --- | --- | --- |
| float16 | $2^{-9}$ | $2^{-9}$ | 1e-1 |
| bfloat16 | $2^{-6}$ | $2^{-6}$ | 1e0 |
| float32 | $2^{-10}$ | $2^{-16}$ | 1e-2 |
| complex64 | 实部与虚部分别按 float32 参数 | | |

确定性算法重复执行按 bit-wise 规则验收。INF/NAN 按精度标准文档的对应规则验收。

**性能口径**：标杆为 NVIDIA A100 cuSPARSE SpMM 的 NCU Kernel 总耗时；NPU 侧采集相同调用范围内的 Kernel
总耗时并计算倍率。每个 case 至少预热 10 次、正式采样 30 次，报告中位数与 90% 分位，每轮计时前执行设备
同步。描述符、workspace 与 preprocess 结果在采样期间复用，preprocess 的一次性耗时另行报告。计时不含首次
编译、数据生成、Host 到 Device 搬运与无关初始化。Python 端到端耗时与 C++ 执行阶段耗时作为补充数据单独
报告。

性能输入使用已按列索引排序、已合并重复坐标的 CSR。测试代码固定输入的生成或加载方式、最终 nnz、values、
稠密矩阵、`alpha`、`beta` 与 compute type，并在自测报告中完整记录该口径，以便与 A100 基准的输入口径逐项
核对。

按任务书要求，功能与精度提交 Ascend A2 与 Ascend A3 上的测试结果，性能提交 Ascend A3 上的测试结果；
无法全量执行的组合在验收前说明并取得确认。

## 测试覆盖

测试同时覆盖 Python/ATen 端到端、aclsparse C++ 接口与 Ascend C Kernel。

| 类别 | 覆盖设计 |
| --- | --- |
| 基础功能 | CSR 与稠密矩阵乘加，覆盖方阵、长矩阵、宽矩阵及 `alpha`/`beta` 默认值 |
| dtype 与标量 | 四种 dtype；`alpha`/`beta` 覆盖 0、1、普通实数，`complex64` 补充非零实部与虚部 |
| 稀疏边界 | `nnz=0`、`nnz=1`、空行、零长度维度及一种非均匀行分布 |
| 布局与操作 | B/C 分别验证 Row-major 与 Column-major，覆盖最小 leading dimension 与一个 padding 用例；CSR_ALG2 验证行主路径，CSR_ALG1 验证列主路径，CSR_ALG3 按其 `opA`/`opB` 限制验证 |
| 索引基 | `idxBase` 取 0 与 1 各覆盖一组 |
| 接口流程 | GetBufferSize、可选 Preprocess、SpMM 完整主流程，以及一个 workspace 不足用例 |
| C++ 接口契约 | 每条错误路径的返回码逐项核对；输入 A、B 在完整调用链前后逐字节不变；输出缓冲区与 workspace 末尾设哨兵，验证无越界写 |
| 异常 | 维度或 dtype 不匹配、非法索引、非法布局/操作/算法组合、跨设备输入 |
| 数值场景 | 普通值、小值、正负混合、零值、离群值、规格允许的 INF/NAN；`beta=0` 时 `input` 的 NaN/Inf 不传播 |
| 确定性 | 同一输入重复执行，结果按位比对 |
| 资源 | 连续创建、执行、销毁描述符，检查内存与资源泄漏 |
| ATen 与泛化 | COO 与 CSR 的 `mat1`、`input` 的各种可广播形状、`out` 变体与别名；对 shape、nnz、稀疏度做代表性组合抽样 |
| Dispatch 与 Profiler | 核对 ATen 分派命中 NPU 注册而非 CPU fallback，并由 Profiler 采集确认核心计算 Kernel 在 NPU 上执行 |

Kernel 侧的两条数据通路通过布局组合区分：行主序不转置与列主序转置直接进入主 kernel，另外两种组合经重排
kernel，重排用例覆盖非 64 整除的边界块。长尾行分布用于覆盖跨行预取流水，`CSR_FP32_HIGH_PRECISION_ALG`
用长行抵消场景与 DEFAULT 路径对比。

**测试环境**：Ascend A2 / Ascend A3；CANN 版本以算子开源仓指定版本为准，实际使用版本记录在自测报告中；
PyTorch 2.7 及以上；torch_npu 26.0.0 及之后。

## 兼容性分析

复用 `ops-sparse` 已有的 `aclsparseSpMM*` 接口声明与调用流程，不修改公开接口签名，对已有调用方保持源代码
与行为兼容。新增能力是既有布局、算法与类型分支的扩展，不改变已支持组合的执行路径。若设计评审需要调整
原型，调整结果保持源代码兼容或给出明确的兼容方案，并同步写入公开头文件与接口文档。

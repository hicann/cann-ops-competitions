# aclsparseSpMM 算子设计（Atlas A2/A3）

# 需求背景（required）

## 需求来源

本文档对应社区任务“aclsparseSpMM 算子开发（A2/A3）”。任务要求参考 PyTorch
`torch.sparse.addmm` 与 `aten::_sparse_addmm` 的接口行为，复用并扩展
`ops-sparse` 仓库已有的 aclsparse SpMM C++ 接口和 Ascend C Kernel，在 Atlas
A2、Atlas A3 系列产品上完成 Python/ATen 适配、数据类型和布局补齐、精度与性能
优化以及配套测试。

软件版本要求如下：

| 组件 | 版本要求 |
| --- | --- |
| PyTorch | 2.7 及以上 |
| torch_npu | 26.0.0 及之上 |
| CANN | `ops-sparse` 仓库指定版本，实际验证版本在自测报告中记录 |
| C++ 接口语义参考 | CUDA Toolkit 13.3 Update 1 中的 cuSPARSE SpMM |

核心计算必须在 NPU 上完成，不允许以 CPU fallback 代替 NPU Kernel。

## 背景介绍

SpMM（Sparse Matrix-Dense Matrix Multiplication）用于计算稀疏矩阵与稠密矩阵
的乘加：

$$
C = \alpha \cdot \operatorname{op}(A)\operatorname{op}(B) + \beta \cdot C
$$

其中，$A$ 为 CSR 稀疏矩阵，$B$ 和 $C$ 为稠密矩阵，$\alpha$ 和 $\beta$ 为
标量。PyTorch `torch.sparse.addmm` 对应的计算为：

$$
\mathrm{out} =
\beta \cdot \mathrm{input} +
\alpha \cdot (\mathrm{mat1} \times \mathrm{mat2})
$$

PyTorch 适配层负责将 `input`、CSR `mat1` 和稠密 `mat2` 转换为 aclsparse
描述符和输出缓冲区，底层 aclsparse 接口负责 workspace 查询、CSR 预处理和
NPU Kernel 执行。

## 现有能力与本任务目标

`ops-sparse` 已提供 Generic API 风格的 `aclsparseSpMM*` 三阶段接口。当前
arch22 实现覆盖 Ascend910B 和 Ascend910_93 系列，已打通 CSR、四种必选
dtype、稠密矩阵行主/列主布局以及 PyTorch NPU 注册。

本任务在复用现有接口的基础上完成以下目标：

1. 对齐 PyTorch 2.7 及以上版本的 `torch.sparse.addmm` 行为。
2. 打通 `float16`、`bfloat16`、`float32` 和 `complex64` 全链路。
3. 支持 CSR 泛化输入、稠密矩阵布局、标量语义、边界场景和错误处理。
4. 复用 CSR 预处理结果与 workspace，满足 A3 性能目标。
5. 在 A2、A3 上完成规定的功能和精度验证，并提供无 CPU fallback 证据。
6. 保持 A2/A3 与 A5 对应架构实现相互独立，可在同一主干共存。

# 需求分析（required）

## 需求描述

### Python 公开接口

```python
torch.sparse.addmm(input, mat1, mat2, *, beta=1, alpha=1) -> Tensor
```

### ATen Schema

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

NPU 实现注册到 `SparseCsrPrivateUse1` Dispatch Key。用户加载
`ops_sparse_torch` 后继续调用 PyTorch 公共接口，不修改或替换
`torch.sparse.addmm`。

### 输入输出定义

| 参数 | 说明 | shape | dtype | 布局与约束 |
| --- | --- | --- | --- | --- |
| `input/self` | 参与加法的稠密输入 | 可广播为 `[M,N]` | FP16、BF16、FP32、Complex64 | 支持 `[M,N]`、`[N]`、`[1,N]`、`[M,1]` 等合法广播形状；无法直接使用的 stride 由适配层处理 |
| `mat1` | CSR 稀疏矩阵 $A$ | `[M,K]` | values 与其他 Tensor 相同 | `crow_indices`、`col_indices` 均为 int32 |
| `mat2` | 稠密矩阵 $B$ | `[K,N]` | 与 `input`、`mat1.values` 相同 | 支持可直接描述的行主或列主布局；其他合法 stride 生成连续副本 |
| `beta` | `input` 缩放系数 | 标量 | 转换为共同计算类型 | `beta=0` 时不读取或传播 `input` 中的 NaN/Inf |
| `alpha` | 稀疏矩阵乘积缩放系数 | 标量 | 转换为共同计算类型 | 实数 dtype 不接受虚部非零的复数 |
| `output` | 稠密输出 | `[M,N]` | 与三个 Tensor 输入相同 | 位于输入所在的同一 NPU 设备 |

Tensor 输入之间不执行 dtype 提升。三个 Tensor 必须位于同一 NPU 设备。

## C++ Generic API

沿用公开头文件 `include/cann_ops_sparse.h` 中的接口，不新增重复接口。

```cpp
aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    size_t *bufferSize);

aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *externalBuffer);

aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    const void *beta,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    void *externalBuffer);
```

## 支持范围

### 硬件范围

| 架构目录 | 产品系列 | 支持情况 |
| --- | --- | --- |
| `sparse/spmm/arch22` | Ascend910B 系列 | 支持 |
| `sparse/spmm/arch22` | Ascend910_93 系列 | 支持 |
| `sparse/spmm/arch35` | A5 对应产品 | 由独立架构实现负责，不属于本文设计范围 |

构建系统根据目标 SoC 选择对应架构目录，公共接口声明保持一致，Host 和 Kernel
实现按硬件架构隔离。

### dtype 与 computeType

| A values | B | C | computeType | 计算说明 |
| --- | --- | --- | --- | --- |
| `ACL_FLOAT16` | `ACL_FLOAT16` | `ACL_FLOAT16` | `ACL_FLOAT` | FP32 累加，结果转换为 FP16 |
| `ACL_BF16` | `ACL_BF16` | `ACL_BF16` | `ACL_FLOAT` | FP32 累加，结果转换为 BF16 |
| `ACL_FLOAT` | `ACL_FLOAT` | `ACL_FLOAT` | `ACL_FLOAT` | FP32 乘加 |
| `ACL_COMPLEX64` | `ACL_COMPLEX64` | `ACL_COMPLEX64` | `ACL_COMPLEX64` | Complex64 复数乘加 |

A、B、C 必须具有相同 dtype。不支持表外的混合存储类型组合。

### operation、order 与算法

operation 缩写含义如下：

| 枚举 | 缩写 | 含义 |
| --- | --- | --- |
| `ACL_SPARSE_OP_NON_TRANSPOSE` | N | 使用原矩阵 |
| `ACL_SPARSE_OP_TRANSPOSE` | T | 普通转置 |
| `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE` | CT | 共轭转置 |

当前 arch22 声明的 C++ 支持矩阵如下：

| 算法 | opA | opB | B order | C order | 说明 |
| --- | --- | --- | --- | --- | --- |
| `ACL_SPARSE_SPMM_ALG_DEFAULT` | N | N、T、CT | Row、Col | Row、Col | 由实现选择内部计算策略 |
| `ACL_SPARSE_SPMM_CSR_ALG1` | N | N、T、CT | Row、Col | Row、Col | 列主场景优先使用 |
| `ACL_SPARSE_SPMM_CSR_ALG2` | N | N、T、CT | Row、Col | Row、Col | 行主场景优先使用 |
| `ACL_SPARSE_SPMM_CSR_ALG3` | N | N、T | Row、Col | Row、Col | `opA=N, opB=N/T` |

### 稠密矩阵布局

B、C 支持以下布局：

| order | leading dimension 约束 | 元素 `(row,col)` 的逻辑位置 |
| --- | --- | --- |
| `ACL_SPARSE_ORDER_ROW` | `ld >= cols` | `row * ld + col` |
| `ACL_SPARSE_ORDER_COL` | `ld >= rows` | `col * ld + row` |

允许 leading dimension padding。对于 `opB=T/CT`，B 描述符记录的是转置前的
物理 shape，维度校验使用 `op(B)` 的有效 shape。

### 维度关系

由于当前 `opA=N`，设 A 的物理 shape 为 `[M,K]`：

- `opB=N` 时，B 的物理 shape 为 `[K,N]`；
- `opB=T/CT` 时，B 的物理 shape 为 `[N,K]`；
- C 的 shape 始终为 `[M,N]`。

所有矩阵均按二维矩阵处理。PyTorch 层只允许 `input` 广播，`mat1` 和 `mat2`
不进行批次广播或矩阵维度广播。

### alpha/beta 指针模式

C++ 接口通过 handle 的 pointer mode 解释 `alpha` 和 `beta`：

| pointer mode | alpha/beta 位置 | 标量类型 |
| --- | --- | --- |
| `ACL_SPARSE_POINTER_MODE_HOST` | Host 内存 | 实数路径为 FP32，Complex64 路径为复数 FP32 |
| `ACL_SPARSE_POINTER_MODE_DEVICE` | Device 内存 | 类型同上 |

PyTorch 适配层使用 Host scalar。Complex64 接受实数或复数 alpha/beta；实数
Tensor dtype 接受整数或浮点标量，并拒绝虚部非零的复数标量。

### 空输入与特殊值

- `nnz=0`：乘法项为零，输出等于 `beta * C`。
- 空行：对应输出行仅保留 `beta * C`。
- `beta=0`：不读取 C 的原始数值，C 中的 NaN/Inf 不传播到输出。
- 零长度维度：输出 shape 仍按矩阵维度推导；不得访问空数据缓冲区。
- Complex64 的实部和虚部分别按复数乘加语义计算。

## 需求拆解

| 模块 | 主要职责 |
| --- | --- |
| PyTorch/ATen 适配 | NPU 注册、输入校验、广播、布局识别、输出构造、标量转换 |
| 描述符与 Handle | 复用现有 Generic API 描述符、pointer mode 和 stream |
| Workspace 查询 | 根据 shape 和预处理元数据计算所需 Device workspace |
| Preprocess | 构建可复用的 CSR 行调度与计算元数据 |
| Host 执行 | 校验参数、准备运行时参数、在调用方 stream 下发 Kernel |
| Ascend C Kernel | 执行四种 dtype、布局与 operation 对应的 SpMM 计算 |
| 测试与性能 | C++、Python 端到端、精度、性能、Profiler 和资源生命周期验证 |

# 详细设计（required）

## 总体架构

```text
torch.sparse.addmm
        │
        ▼
aten::_sparse_addmm / SparseCsrPrivateUse1
        │
        ▼
PyTorch Adapter
  ├─ 参数、dtype、device、shape 校验
  ├─ input 广播与 beta 语义处理
  ├─ mat2 布局识别或连续化
  └─ 描述符、workspace、preprocess 上下文复用
        │
        ▼
aclsparseSpMM Generic API
  ├─ GetBufferSize
  ├─ Preprocess
  └─ SpMM
        │
        ▼
arch22 Host + Ascend C Kernel
        │
        ▼
Dense NPU Tensor output
```

公共 API 与描述符位于仓库公共层；A2/A3 使用 `arch22` 实现，其他硬件架构使用
各自目录，避免将硬件专属计算逻辑混入公共接口。

## PyTorch/ATen 适配设计

### Dispatch 与调用

扩展注册 `aten::_sparse_addmm` 的 `SparseCsrPrivateUse1` 实现。适配层获取当前
NPU stream，将尚未提交的 PyTorch Host Task Queue 工作提交到该 stream，然后
调用 aclsparse C++ 接口。核心 SpMM 计算由 Ascend C Kernel 完成。

### 参数校验

适配层在创建描述符前完成以下校验：

1. `mat1` 必须为二维 CSR Tensor，`mat2` 必须为二维 Dense Tensor。
2. `input`、`mat1` 和 `mat2` 位于同一 NPU 设备。
3. 三个 Tensor dtype 相同且属于四种必选 dtype。
4. CSR 行偏移和列索引均为连续 int32 Tensor。
5. `mat1.size(1) == mat2.size(0)`。
6. `input` 能够广播到 `[M,N]`。
7. alpha/beta 能转换到对应计算类型，实数 dtype 不接受非零虚部。

不满足约束时直接返回明确异常，不进行 CPU fallback。

### input 与输出处理

`input` 先按 PyTorch 广播规则扩展到 `[M,N]`。当 `beta=0` 时创建不依赖
`input` 数值的新输出缓冲区，避免 NaN/Inf 传播；其他情况将广播后的 input
作为 C 初值。

输出为行主 Dense NPU Tensor，shape 为 `[M,N]`，dtype 和 device 与输入一致。
适配过程不覆盖 `input`、CSR 索引、CSR values 或 `mat2`。

### mat2 布局处理

能够用合法 leading dimension 表达的二维行主或列主 Tensor 直接创建稠密矩阵
描述符。其他合法非连续 Tensor 生成行主连续副本，再参与计算。布局转换属于
适配准备阶段，不改变数学语义。

### 上下文复用

适配层按设备、stream、CSR 结构、shape、dtype 和稠密布局维护可复用上下文。
在 CSR 结构和相关调用属性不变时复用：

- aclsparse handle；
- CSR 描述符；
- workspace；
- CSR 行重排和分桶结果；
- Preprocess 完成状态。

CSR 结构、设备、stream、shape、dtype 或布局发生变化时创建新的上下文。CSR
values 可以在结构不变时更新，不要求重新生成行调度信息。

## aclsparse C++ 接口设计

### 三阶段流程

1. `aclsparseSpMMGetBufferSize` 校验描述符、operation、dtype、order、算法和
   维度，返回所需 workspace 字节数。
2. `aclsparseSpMMPreprocess` 将 CSR 行调度信息和 Kernel 所需元数据写入
   workspace。该阶段可以只执行一次。
3. `aclsparseSpMM` 使用调用方传入的描述符和 workspace，在 handle 绑定的
   stream 上执行计算。

未显式调用 Preprocess 时，执行阶段可以在首次使用当前 workspace 时补建必要
元数据。性能测试和高频调用应显式 Preprocess 并复用结果。

### Workspace 生命周期

- workspace 由调用方根据 `GetBufferSize` 返回值分配，位于 Device 内存。
- 若要复用 Preprocess 结果，SpMM 应使用同一个有效 workspace；更换 workspace
  时，执行阶段需要重新生成对应的预处理数据。
- workspace 不得与 A、B、C 数据缓冲区重叠。
- 在相关 stream 上的 SpMM 完成前，调用方不得释放或复用该 workspace。
- CSR 结构或影响预处理结果的调用属性变化后，应重新执行 Preprocess。
- workspace 内部布局属于实现细节，不作为公开 ABI。

### 描述符生命周期

- handle、matA、matB、matC 均由调用方创建和销毁。
- matA、matB 在计算中只读，matC 为输入输出描述符。
- 描述符保存的数据指针在 Preprocess 和 SpMM 使用期间必须有效。
- SpMM 异步任务完成前，不得销毁描述符所引用的数据、workspace 或 stream。
- Create/Destroy 必须配对，重复创建、执行和销毁不得泄漏资源。

### 错误处理

| 场景 | 预期状态 |
| --- | --- |
| handle 为 nullptr | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| alpha、beta、size 或必需 buffer 为 nullptr | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 描述符为 nullptr | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 非 CSR 格式 | `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED` 或仓库统一的不支持状态 |
| 索引类型、dtype、operation、order 或算法不支持 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| 矩阵维度不匹配或规模溢出 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| Runtime 复制或 Kernel 下发失败 | `ACL_SPARSE_STATUS_EXECUTION_FAILED` |

三个 SpMM 阶段应复用一致的校验逻辑。当前代码中部分 nullptr 场景的状态码仍需
与公共规范统一，最终以公开接口文档和 C++ UT 为准。

## Preprocess 设计

Preprocess 根据 CSR 行偏移统计每行工作量，将稀疏行映射到可复用的执行顺序，
并生成 Kernel 需要的调度元数据。其目标是缓解空行、短行、长尾行和非均匀行
分布造成的核间负载不均。

预处理结果仅依赖 CSR 结构及与调度相关的调用属性，不依赖 CSR values 的具体
数值。结构不变时允许更新 values、B、C、alpha 和 beta，并复用已有行调度结果。
一次性 Preprocess 耗时与正式 SpMM 执行耗时分别统计。

具体的分桶边界、行分组阈值和 workspace 内部字段属于实现细节，不在公开设计
中固化。

## Kernel 实现设计

Kernel 按 CSR 行读取 A 的非零元素，对 B 的相应行或列执行乘加，并将结果按 C
的 order 和 leading dimension 写回。Host 根据 shape、dtype、布局和稀疏行特征
准备运行时参数，Kernel 在多个 Vector Core 之间分配输出行工作。

FP32 路径直接完成浮点乘加；FP16 和 BF16 路径转换到 FP32 进行累加，再转换回
目标 dtype；Complex64 路径完成复数乘加、复数 alpha/beta 以及 `opB=CT` 所需
的共轭处理。

实现根据数据布局和问题规模采用分块、行缓存、连续数据批量搬运及多行批处理
等策略，减少离散访存和标量调度。无法使用快速路径的合法组合进入通用布局
路径，保证功能完整性。

具体 tile 大小、UB 分配、DMA/Vector 指令顺序、双缓冲方式和快速路径阈值由
实现根据硬件资源确定，不属于公开接口约束，也不在本文中展开。

## 算法枚举处理

DEFAULT、ALG1、ALG2 和 ALG3 共用参数校验与计算框架。ALG1 优先用于列主布局，
ALG2 优先用于行主布局；ALG3 遵循 `opA=N` 且 `opB!=CT` 的限制。内部优化路径
可以根据实际 shape、dtype 和稀疏分布选择，不要求算法枚举与某一个固定 Kernel
函数一一对应。所有已声明组合必须满足正确性要求；声明为确定性的算法还必须
满足 bit-wise 确定性要求，其他算法按照混合容差标准验收。

## 规模与约束

| 项目 | 当前限制 |
| --- | --- |
| M、K、N | 每一维不超过 `UINT32_MAX` |
| nnz | 不超过 `UINT32_MAX` |
| leading dimension | 满足对应 order 的最小值，且地址计算不得溢出 |
| workspace | 以 `GetBufferSize` 返回值为准，大小随 M 和执行核数增长 |
| CSR 索引 | int32，base-zero 或 base-one |
| Batch SpMM | 不支持 |
| 稀疏格式转换 | 不在 SpMM 内执行 |
| 图融合 | 不要求 |

输入 CSR 应满足行偏移单调、首尾偏移与 index base/nnz 一致、列索引位于合法
范围。由于索引数组位于 Device，非法 CSR 内容的检测范围和错误返回时机需要在
接口测试中明确；不得因非法输入造成未声明的越界写。

## 兼容性与共存设计

- 不改变已有 `aclsparseSpMM*` 函数签名和枚举值。
- Python 层通过标准 ATen Dispatch 注册，不 monkeypatch PyTorch 公共 API。
- arch22 与 arch35 保持独立 Host/Kernel 实现，共用公开描述符和接口声明。
- 后续合入 A5 相关代码时，公共校验和描述符逻辑优先复用，硬件专属调度留在
  对应架构目录。

# 可维可测分析

## 可维护性分析

1. Python 适配、Generic API、预处理和 Kernel 分层，接口职责清晰。
2. dtype、布局和算法校验集中在 Host 公共路径，避免三个执行阶段行为不一致。
3. 硬件差异按架构目录隔离，不在公共接口中引入 SoC 专属参数。
4. workspace 内部格式和 Kernel 优化阈值不构成公开 ABI，可独立迭代。
5. 测试同时覆盖 C++ 接口和 Python 端到端行为，便于定位适配层与 Kernel 问题。

## 功能测试设计

| 类别 | 覆盖内容 |
| --- | --- |
| 基础功能 | 方阵、长矩阵、宽矩阵；alpha/beta 默认值 |
| dtype | FP16、BF16、FP32、Complex64 |
| 标量 | alpha/beta 为 0、1、普通实数；Complex64 增加非零实部和虚部 |
| 稀疏边界 | `nnz=0/1`、空行、非均匀行、长尾行、零长度维度 |
| 索引 | base-zero、base-one、int64 拒绝、非法类型和边界 |
| 布局 | B/C Row-major、Column-major、最小 ld、padding ld |
| operation | 已声明的 opB N/T/CT；ALG3 非法 CT 组合 |
| 算法 | DEFAULT、ALG1、ALG2、ALG3 的已声明组合 |
| 接口流程 | GetBufferSize、可选 Preprocess、SpMM、workspace 复用 |
| PyTorch | input 广播、非连续 mat2、输出属性、异常行为、无 CPU fallback |
| 特殊值 | `beta=0` 时 input 中 NaN/Inf 不传播 |
| 资源 | 输入只读、输出边界、workspace 边界、重复创建/销毁无泄漏 |
| 确定性 | 同输入重复执行，对要求确定的算法进行 bit-wise 比较 |

## 精度标准

CPU Golden 计算精度如下：

| NPU dtype | CPU Golden dtype | rtol | atol | 绝对误差硬上限参数 A |
| --- | --- | --- | --- | --- |
| FP16 | FP32 | $2^{-9}$ | $2^{-9}$ | $10^{-1}$ |
| BF16 | FP32 | $2^{-6}$ | $2^{-6}$ | $10^{0}$ |
| FP32 | FP64 | $2^{-10}$ | $2^{-16}$ | $10^{-2}$ |
| Complex64 | Complex128 | 实部、虚部分别使用 FP32 标准 | 同 FP32 | 同 FP32 |

普通实数 dtype 的逐元素判定条件为：

$$
|\mathrm{actual}-\mathrm{golden}|
\leq \mathrm{atol}+\mathrm{rtol}\times|\mathrm{golden}|
$$

整体匹配率不低于 0.99，并且每个元素的绝对误差不得超过
`max(A, 32 × ULP(golden))`。Complex64 对实部和虚部分别判定。数据覆盖普通值、
小值、正负混合、零值、离群值及规格允许的 NaN/Inf。

## 性能测试设计

### 采样方法

1. 使用任务书给定的固定 CSR 数据生成或加载方式，列索引已排序且重复坐标已合并。
2. 首次编译、数据生成、H2D 和无关初始化不计入性能时间。
3. 描述符、workspace 和 Preprocess 结果在正式采样期间复用。
4. 每个场景预热至少 10 次，正式采样至少 30 次。
5. 每轮在计时要求的位置进行设备同步，报告 Kernel 总耗时中位数和 P90。
6. Preprocess 一次性耗时、C++ 执行耗时和 Python 端到端耗时分别补充报告。
7. 提供 NPU Dispatch 和 Profiler 证据，确认核心计算未回退到 CPU。

### A100 基线与目标

| 编号 | M×K×N | nnz | dtype | A100 Kernel 总耗时（μs） | 满足倍率 >0.25 的 NPU 耗时上限（μs） |
| --- | --- | --- | --- | --- | --- |
| P-01 | 2,708×2,708×1,433 | 10,556 | FP32 | 87.040 | FP32：<348.160 |
| P-02 | 169,343×169,343×128 | 1,166,243 | FP16 / BF16 / FP32 | 321.536 / 413.920 / 204.576 | FP16：<1,286.144；BF16：<1,655.680；FP32：<818.304 |
| P-03 | 2,449,029×2,449,029×256 | 61,859,140 | FP16 / BF16 / FP32 / Complex64 | 25,446.496 / 33,704.096 / 16,571.232 / 38,806.400 | FP16：<101,785.984；BF16：<134,816.384；FP32：<66,284.928；Complex64：<155,225.600 |

性能倍率定义为：

$$
\mathrm{倍率} =
\frac{\mathrm{A100\ Kernel\ 总耗时}}
     {\mathrm{NPU\ Kernel\ 总耗时}}
$$

A3 的每个 `case × dtype` 场景倍率必须大于 0.25，所有量化场景的算术平均倍率
不低于 0.35。

## 验证环境与交付

功能和精度需要分别提供 A2、A3 结果，性能仅要求 A3。自测报告记录：

- SoC 型号；
- CANN、PyTorch、torch_npu 版本；
- 构建方式和代码提交；
- 用例参数、精度结果和失败项；
- 性能 median/P90、倍率和 Preprocess 耗时；
- NPU Dispatch/Profiler 截图或原始证据；
- 资源、边界与重复执行结果。

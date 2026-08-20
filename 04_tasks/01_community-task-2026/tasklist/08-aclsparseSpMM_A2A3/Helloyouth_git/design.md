# aclsparseSpMM A2/A3 算子设计文档

# 需求背景（required）

## 需求来源

本任务来自 CANN 社区 2026 年稀疏算子开发任务，要求在 `ops-sparse` 仓中补齐 `aclsparseSpMM` 在 Atlas A2 系列产品和 Atlas A3 系列产品上的 Python/ATen、C++ 接口和 Ascend C Kernel 能力。任务对齐 PyTorch 2.7 及以上版本的 `torch.sparse.addmm` 与 `aten::_sparse_addmm` 行为，并复用 `ops-sparse` 既有 `aclsparseSpMM*` 接口体系。

本设计文档对应任务书：`aclsparseSpMM 算子开发(A2A3)任务书/aclsparseSpMM_A2A3_task_doc.md`。

## 背景介绍

### aclsparseSpMM 算子实现优化

SpMM（Sparse Matrix Dense Matrix Multiplication）用于计算 CSR 稀疏矩阵与稠密矩阵的乘加：

```text
out = beta * input + alpha * (mat1 * mat2)
```

其中 `mat1` 为 CSR 稀疏矩阵，`mat2` 为稠密矩阵，`input` 为可广播到输出形状的稠密矩阵。该算子是 GNN、推荐系统、科学计算和稀疏线性代数中的基础算子。任务要求核心计算必须在 NPU 上完成，不能以 CPU fallback 替代。

### PyTorch / cuSPARSE 实现现状分析

PyTorch 公开入口为：

```python
torch.sparse.addmm(input, mat1, mat2, *, beta=1, alpha=1) -> Tensor
```

内部 ATen schema 为：

```text
aten::_sparse_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
```

C++ 层接口对标 cuSPARSE `cusparseSpMM` 的 workspace 查询、可选 preprocess 和 compute 三阶段语义，并统一采用 `aclsparseSpMM*` 命名。任务要求复用 `include/cann_ops_sparse.h` 中已有 handle、CSR 描述符、DnMat 描述符、workspace 与资源释放接口，同时补齐 A2/A3 上 `float16`、`bfloat16`、`float32`、`complex64` 的全链路能力。

# 需求分析（required）

## 需求描述

在 `ops-sparse` 中完成 `torch.sparse.addmm` NPU 适配，使 Python 入口、ATen NPU dispatch、aclsparse C++ 接口和 Ascend C Kernel 的行为满足任务书要求。

Python 侧行为：

```python
torch.sparse.addmm(input, mat1, mat2, *, beta=1, alpha=1)
```

C++ 侧主接口：

```c
aclsparseStatus_t aclsparseSpMMGetBufferSize(...);
aclsparseStatus_t aclsparseSpMMPreprocess(...);
aclsparseStatus_t aclsparseSpMM(...);
```

## 需求拆解

1. 支持 `aten::_sparse_addmm` NPU 注册，端到端命中 NPU 路径。
2. 支持 CSR 稀疏输入 `mat1`，`csrRowOffsets` 和 `csrColInd` 均为 `ACL_SPARSE_INDEX_32I`。
3. 支持 `input` 按 PyTorch 规则广播到 `[M, N]`。
4. 支持 `mat2` row-major / column-major 描述，正确处理 leading dimension。
5. 支持 `float16`、`bfloat16`、`float32`、`complex64`。
6. 支持 alpha / beta 标量转换；`beta=0` 时不传播 `input` 的 NaN/Inf 数值。
7. 支持 `CSR_ALG1`、`CSR_ALG2`、`CSR_ALG3` 中任务书声明的合法组合。
8. 支持动态 shape、动态 nnz、空行、零 nnz、非均匀行分布。
9. 非法 dtype、device、shape、index、layout、op 或 workspace 必须返回明确错误。
10. 性能在 A3 上按任务书要求：每个 case×dtype 均大于 A100 的 0.25 倍，整体平均不低于 0.35 倍。

# 详细设计（required）

## 算子分析

### 数学公式

令 CSR 稀疏矩阵 A 的形状为 `[M, K]`，稠密矩阵 B 的形状为 `[K, N]`，输入矩阵 D 可广播到 `[M, N]`，输出 C 为 `[M, N]`：

```text
C[i, j] = beta * D_broadcast[i, j] + alpha * sum(A[i, k] * B[k, j])
```

其中 A 的非零结构由 `rowOffsets`、`colIndices`、`values` 描述。对于 `complex64`，乘加和 alpha/beta 缩放均按复数语义执行。

### 支持数据类型

| 数据 | 支持类型 | 说明 |
| --- | --- | --- |
| input/self | float16, bfloat16, float32, complex64 | 稠密输入，可广播 |
| mat1 values | float16, bfloat16, float32, complex64 | CSR values |
| mat1 indices | int32 | rowOffsets / colIndices |
| mat2 | float16, bfloat16, float32, complex64 | 稠密矩阵 |
| alpha/beta | Scalar | 转换到共同计算类型 |
| output | float16, bfloat16, float32, complex64 | 与输入共同 dtype 一致 |

不执行 Tensor 间 dtype 提升；`input`、`mat1.values`、`mat2` dtype 不一致时直接报错。

### 支持形状

```text
input: 可广播到 [M, N]
mat1:  CSR [M, K], nnz >= 0
mat2:  Dense [K, N]
out:   Dense [M, N]
```

边界行为：

- `nnz == 0`：输出为 `beta * input_broadcast`，`beta=0` 时输出为全零。
- 存在空行：空行结果仅来自 beta 分支。
- `M`、`K`、`N` 为 0 的合法场景按照 PyTorch 2.7 语义返回空输出或报错。
- CSR 行偏移必须单调非降，末元素等于 `nnz`。

## 算子实现

### 实现方案

整体分为 Python/ATen 适配层、aclsparse Host 层、Ascend C Kernel 层和测试文档层：

| 模块 | 设计职责 |
| --- | --- |
| Python / ATen | NPU dispatch、参数校验、稀疏元数据转换、输出 Tensor 创建 |
| aclsparse C++ Host | 描述符解析、workspace 计算、preprocess、tiling、kernel launch |
| Ascend C Kernel | CSR 分桶/行块计算、稀疏值与 dense tile 乘加、alpha/beta epilogue |
| 测试 | Python 端到端、C++ 接口、精度、性能、异常和 profiler 证据 |

### Host 侧设计

Host 侧按以下顺序执行：

1. 校验 handle、stream、描述符、alpha/beta 指针、workspace 指针。
2. 校验 A 为 CSR，索引类型均为 `ACL_SPARSE_INDEX_32I`，`idxBase` 为 0 或 1。
3. 校验 B/C 为 dense matrix，order 和 leading dimension 满足接口约束。
4. 校验 dtype 与 computeType 合法组合，不支持组合返回 `ACLSPARSE_STATUS_NOT_SUPPORTED`。
5. 根据 shape、nnz、dtype、order、alg、硬件类型生成 tiling。
6. `GetBufferSize` 返回 preprocess / compute 所需 workspace。
7. `Preprocess` 对 CSR 行进行统计、分桶或必要重排，生成可复用元数据。
8. `SpMM` 复用 preprocess 结果或直接执行 compute kernel。
9. 将 `beta * input` 与 `alpha * A * B` 在 epilogue 合成输出，`beta=0` 时不读取 input 数值。

### Kernel 侧设计

Kernel 采用 CSR 行分块与 dense N 维分块并行。核心流程：

1. 每个 block 处理一组 CSR 行和一段 N 维列 tile。
2. 根据 `rowOffsets[i]` 到 `rowOffsets[i+1]` 遍历非零元素。
3. 读取 `A_values[p]` 与 `B[colIndices[p], j:j+tileN]`。
4. 对 fp16/bf16 可在 fp32 累加后按输出 dtype 回写；float32 使用 fp32；complex64 拆成实部/虚部乘加或使用统一 complex helper。
5. epilogue 执行 alpha / beta 缩放与广播 input 读取。
6. 对空行和 nnz=0 快速处理，避免越界访问。

A2/A3 分支通过 tiling 和 kernel capability 区分，不复制公共 Host 逻辑。A2/A3 与 950 任务共享的代码必须集中在公共 SpMM host / descriptor / validation 模块，硬件差异仅在 tiling 策略和 kernel specialization 中体现。

### 文件规划

| 文件或目录 | 新增/修改 | 说明 |
| --- | --- | --- |
| `include/cann_ops_sparse.h` | 修改 | 复用并补齐 SpMM 接口声明和枚举说明 |
| `src/sparse/spmm/` | 修改/新增 | Host 参数校验、workspace、preprocess、launch |
| `src/kernel/spmm/` | 修改/新增 | A2/A3 Ascend C Kernel |
| `torch_npu` 适配目录 | 修改/新增 | 注册 `aten::_sparse_addmm` NPU dispatch |
| `tests/spmm/` | 新增 | 精度、性能、C++接口、异常、profiler 测试 |
| `docs/spmm/README.md` | 新增 | 接口语义、限制、复现步骤 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品 | √ |
| Atlas A3 系列产品 | √ |

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 稀疏格式 | C++ 层仅支持 CSR |
| 索引类型 | rowOffsets / colIndices 仅支持 int32 |
| dtype | float16、bfloat16、float32、complex64 |
| opA/opB | 仅支持任务书声明的 cuSPARSE SpMM 合法组合 |
| layout | B/C 支持 row-major 和 column-major，leading dimension 必须合法 |
| 广播 | 仅 input 按 PyTorch 规则广播 |
| CPU fallback | 不允许以 CPU fallback 替代核心计算 |
| 异步 | 使用调用方 stream，避免不必要 Host 同步 |
| 反向 | 本任务仅要求前向 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 按生态算子开源精度标准，CPU Golden 单标杆混合容差 | 任务书 |
| 性能标准 | A3 每个 case×dtype > A100 0.25 倍，全部场景平均 >= A100 0.35 倍 | 任务书 |
| NPU 路径 | 提供 dispatch 与 profiler 证据，证明未回退 CPU | 任务书 |

精度 Golden：

- fp16 / bf16：CPU fp32 计算。
- fp32：CPU fp64 计算。
- complex64：CPU complex128 计算，实部和虚部分别按 fp32 容差判定。

## 测试设计

| 类别 | 覆盖场景 |
| --- | --- |
| 基础功能 | 方阵、长矩阵、宽矩阵、alpha/beta 默认值 |
| dtype | float16、bfloat16、float32、complex64 |
| 标量 | alpha/beta 为 0、1、普通实数；complex64 覆盖复数标量 |
| 稀疏边界 | nnz=0/1、空行、长尾行、非均匀行分布 |
| layout | Row-major / Column-major，最小 leading dimension 和 padding |
| 接口流程 | GetBufferSize、Preprocess、SpMM 完整流程 |
| 异常 | dtype、shape、device、索引、layout、workspace 不合法 |
| ATen | `torch.sparse.addmm` 端到端命中 NPU dispatch |
| 性能 | P-01 到 P-03 的任务书性能 shape 和 dtype |

性能计时要求：预热不少于 10 次，正式采样不少于 30 次，报告 median 和 p90；正式计时不包含首次编译、数据生成、H2D 搬运和无关初始化。

## 兼容性分析

本设计复用 `ops-sparse` 既有 aclsparse handle、CSR/DnMat 描述符和 SpMM 接口，不新增同名重复接口。A2/A3 与 950PR 的公共逻辑保持一致，硬件差异通过 tiling 和 kernel specialization 隔离，降低后续双任务合入冲突。

# 参考资料

- `aclsparseSpMM_A2A3_task_doc.md`
- PyTorch `torch.sparse.addmm`
- PyTorch `aten::_sparse_addmm`
- NVIDIA cuSPARSE SpMM
- ops-sparse 仓库：<https://gitcode.com/cann/ops-sparse>
- 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>

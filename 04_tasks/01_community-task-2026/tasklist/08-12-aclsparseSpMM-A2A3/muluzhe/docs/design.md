# 【社区任务】aclsparseSpMM（A2/A3）算子设计文档

# 一、需求背景（required）

## 1.1 需求来源

aclsparseSpMM 算子开发任务，来源为 ops-sparse 仓内《aclsparseSpMM 算子开发(A2A3)任务书》。任务要求参考 PyTorch `torch.sparse.addmm` 与 `aten::_sparse_addmm` 的接口和行为，在昇腾 NPU 上完成 Python/ATen 适配，并复用和扩展 ops-sparse 已有 aclsparse SpMM C++ 接口及 Ascend C Kernel。

## 1.2 背景介绍

### aclsparseSpMM 算子实现路径

aclsparse SpMM C++ 接口声明路径：`include/cann_ops_sparse.h`

aclsparse SpMM Host/Kernel 实现路径：`sparse/spmm/arch22`（Atlas A2，dav-2201）、`sparse/spmm/arch35`（Ascend 950，dav-3510，仅作 950/A5 方向参考）

Python/ATen 适配路径：ops-sparse 仓内新增（master 当前无 torch.sparse.addmm NPU 注册）

### aclsparseSpMM 算子实现现状分析

通过对 ops-sparse master 分支的源码分析，当前支持的能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| input/self | 参与加法的稠密矩阵 | tensor | float16, bfloat16, float32, complex64 | shape 须能广播到 [M,N] | [M,N] 或可广播形状 |
| mat1 | 稀疏矩阵 A | tensor | float16, bfloat16, float32, complex64 | CSR 格式，crow_indices/col_indices 为 int32 | [M,K] |
| mat2 | 稠密矩阵 B | tensor | float16, bfloat16, float32, complex64 | Row-major 或 Column-major | [K,N] |
| beta | input 缩放系数 | ATen Scalar | 实数或复数 | 默认 1 | - |
| alpha | mat1×mat2 缩放系数 | ATen Scalar | 实数或复数 | 默认 1 | - |
| output | 输出 tensor | tensor | float16, bfloat16, float32, complex64 | dtype 与输入一致 | [M,N] |

计算公式：output = beta * input + alpha * (mat1 @ mat2)

### aclsparseSpMM 算子功能分析

aclsparseSpMM 算子功能：output = beta * input + alpha * (mat1 @ mat2)

输入：input、mat1（CSR 稀疏矩阵）、mat2、alpha、beta

输出：output（稠密矩阵）

支持数据类型：float16、bfloat16、float32、complex64

支持广播：仅 input 按 torch.sparse.addmm 语义广播

# 二、需求分析（required）

## 2.1 需求描述

使用 aclsparse C++ 接口 + Ascend C Kernel 实现 `torch.sparse.addmm` 的 NPU 路径。FP16/BF16/Complex64 是本任务目标能力，当前 master 实际支持范围以源码和基线测试为准。支持 CSR 稀疏格式与 Row/Column-major 稠密布局，适配 Atlas A2 与 Atlas A3 双硬件。

## 2.2 需求拆解

1. 实现 `aten::_sparse_addmm` 的 NPU Dispatch，使 `torch.sparse.addmm` 在 NPU 上的接口、返回值、dtype、shape、device、异常行为与 PyTorch 2.7+ 一致
2. 复用 aclsparse SpMM 三个接口（GetBufferSize / Preprocess / SpMM），补齐 bfloat16、complex64、Row/Column-major、idxBase=1 等能力
3. 支持 float16、bfloat16、float32、complex64 四种 dtype 全链路打通
4. 适配 Atlas A2 系列产品和 Atlas A3 系列产品双硬件
5. 功能和精度在 A2/A3 验证，性能在 A3 验证，具体版本记录于自测报告

# 三、需求详细设计（required）

## 3.1 算子分析

### 数学公式

output = beta * input + alpha * (mat1 @ mat2)

即：out_ij = beta * input_ij + alpha * sum_p(values[p] * mat2[colInd[p], j])

### 支持数据类型

float16、bfloat16、float32、complex64

### 支持形状

mat1：[M, K]（CSR 稀疏矩阵）

mat2：[K, N]（稠密矩阵）

input：可广播到 [M, N]

output：[M, N]

## 3.2 算子实现

### 3.2.1 host侧设计

tiling策略：

Host 侧遍历 csrRowOffsets 计算每行 nnz，生成行 nnz 直方图。按行 nnz 量对行做重排，生成 reorder[M] 表。将 M 行分配到 blockDim 个核，生成 binEdge[blockDim+1] 表，长尾行优先分配以均衡负载。填充 SpmmTilingData（M/K/N/nnz/alpha/beta/opB/order_pair/high_precision 等），写入 workspace。对 idxBase=1 的 CSR 在读取时做偏移。

任务均分：coreNum 根据 M、nnz 和 UB 大小动态调整，确保每个核心处理的数据块数均匀。

批量搬运：tileBlockNum 和 tileDataNum 计算单次搬运的数据量，通过 finalSmallTileNum 和 finalBigTileNum 确定小核/大核的搬运次数，将多次搬运合并为批量操作，减少冗余开销。尾块的处理逻辑确保不完整块也能被合并到计算流程中，避免数据碎片。

##### 1. 分核策略：
优先使用满核的原则。

如果核间能均分，可视作无大小核区分，大核小核数据块一致；

如果核间不能均分，需要将余出的数据块分配到前几个核上。

输入数据大小计算：通过 GetInputShape 和 GetDataTypeLength 函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。

UB内存大小和核心数量获取：通过平台信息获取 UB 内存大小和核心数量，并根据这些信息调整核心数量。

##### 2. 数据分块和内存优化策略：
充分使用UB空间的原则。

需要考虑不同硬件的UB大小不同、是否开启double buffer、kernel侧API实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。

UB内存大小获取：通过 GetCoreMemSize 函数获取 UB 内存的大小，用于后续的数据切分计算。

Tile块计算：根据UB内存大小和预定义的BLOCK_SIZE及BUFFER_NUM，计算出每个Tile块的数据数量。

数据切分：将输入数据按照计算出的Tile块大小进行切分，计算出每个core需要处理的数据块数量和最后一个block的剩余数据量。

设置切分参数：将计算出的切分参数（如每个core的数据量、Tile块大小等）设置到SpmmTilingData对象中。

这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

##### 3. tilingkey规划策略：

需要tilingkey的情况：需要感知host侧信息对kernel侧走不同分支。在host侧获取orderB/orderC/opB/algorithm/dtype 组合，据此设置 tilingkey 以选择对应 Kernel 分支。
数据检测：

### 3.2.2 kernel侧设计

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. FP16/BF16 在 Kernel 侧转 FP32 累加，结果 cast 回原 dtype；FP32 直接累加；complex64 实部/虚部独立乘加。
2. Kernel 侧通过 order_pair/ldb/ldc 字段按 Row-major 或 Column-major 寻址，支持 opB=TRANSPOSE。
3. 根据 tilingkey 执行不同的核函数分支（dtype × 布局 × 算法）。
4. aclsparse SpMM 算子流程：CSR 行重排 → 分桶 → N 维切块 → 逐行 CSR×B 累加 → beta*input 广播 → CopyOut。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 训练系列产品 | √ |

## 3.4 算子约束限制

- 稀疏格式：仅支持 CSR
- 索引：crow_indices 和 col_indices 均为 int32（ACL_SPARSE_INDEX_32I），不支持 64I
- idxBase：支持 0 和 1
- 数据类型：float16、bfloat16、float32、complex64
- 稠密矩阵布局：Row-major（ld >= cols）或 Column-major（ld >= rows）
- opA：仅支持 NON_TRANSPOSE
- opB：支持 NON_TRANSPOSE 和 TRANSPOSE；CONJUGATE_TRANSPOSE 按 cuSPARSE 允许组合支持；CSR_ALG3 不支持 CONJUGATE_TRANSPOSE；未声明组合返回明确错误
- alpha/beta：标量在调用 C++ 接口前转换为共同计算 dtype；实数 dtype 不接受虚部非零复数；complex64 允许实数或复数标量
- beta=0：忽略 input 中的数值，input 中的 NaN/Inf 不传播
- 维度约束：mat1.size(1) == mat2.size(0)
- 广播约束：仅 input 按 torch.sparse.addmm 语义广播
- device 约束：input、mat1、mat2 须位于同一 NPU 设备

# 四、可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差单标杆方法；整体匹配率 ≥ 0.99；各 dtype 参数：FP16 rtol=2^-9 atol=2^-9 A=1e-1，BF16 rtol=2^-6 atol=2^-6 A=1e0，FP32 rtol=2^-10 atol=2^-16 A=1e-2，complex64 实部/虚部分别按 FP32 参数 | 生态算子开源精度标准 |
| 性能标准 | 每个 case×dtype 性能倍率 > 0.25 × A100，全部场景均值 ≥ 0.35 × A100 | 任务书性能要求 |

## 4.2 兼容性分析

复用既有aclsparseSpMM接口并保持签名兼容；新增ATen NPU路径不修改既有接口语义；A2/A3与A5公共Host代码保持可共存。

# 需求背景（required）

## 需求来源

CANN 社区任务 2026：aclsparseSpMM 算子开发（A2/A3）。

## 背景介绍

### aclsparseSpMM 算子实现优化

参考 PyTorch `torch.sparse.addmm` 与 `aten::_sparse_addmm`，在 Atlas A2 / A3 上复用 `ops-sparse` 已有 `aclsparseSpMM*` 接口及 Ascend C Kernel，补齐 dtype、布局、转置、算法与 Python/ATen 适配。

C++ 接口路径：`ops-sparse/include/cann_ops_sparse.h`、`ops-sparse/sparse/spmm/arch22/`

Python / ATen 路径：`ops-sparse/csrc/torch/`、`ops-sparse/python/aclsparse_npu/`

### aclsparseSpMM 算子现状分析

基线 arch22 实现能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| matA | CSR 稀疏矩阵 | 稀疏描述符 | float32 | 仅 CSR，idxBase=0，opA 仅 N | [M, K] |
| matB | 稠密矩阵 | 稠密描述符 | float32 | 仅紧凑行主，opB 仅 N | [K, N] |
| matC | 稠密输入/输出 | 稠密描述符 | float32 | 仅紧凑行主 | [M, N] |
| alpha / beta | 缩放系数 | 标量指针 | float32 | Host / Device pointer mode | - |

计算公式：C = alpha * op(A) * op(B) + beta * C

### aclsparseSpMM 算子功能分析

算子功能：out = beta * input + alpha * (op(A) * op(B))

输入：input（稠密）、mat1（CSR）、mat2（稠密）、alpha、beta

输出：out（稠密）

支持数据类型：float16、bfloat16、float32、complex64

支持广播：仅 input 按 PyTorch 规则广播到 [M, N]

# 需求分析（required）

## 需求描述

使用 Ascend C 复用并扩展 arch22 SpMM，打通 Python / ATen / aclsparse / Kernel 全链路，支持 float16、bfloat16、float32、complex64，覆盖任务书声明的 CSR 布局、转置、算法组合及空输入。

## 需求拆解

1. 复用 `aclsparseSpMMGetBufferSize`、`aclsparseSpMMPreprocess`、`aclsparseSpMM`，不新增同名接口
2. 支持 float16、bfloat16、float32、complex64
3. 支持 CSR、idxBase 0/1、B/C 行主与列主及 padded ld
4. 支持声明范围内的 opA / opB 与 DEFAULT、CSR_ALG1、CSR_ALG2、CSR_ALG3
5. 实现 `aten::_sparse_addmm`，使 `torch.sparse.addmm` 在 NPU 上可调用
6. 覆盖空 nnz、空行、零长度维度
7. 核心计算在 NPU 完成，使用调用方 stream
8. 精度满足混合容差单标杆；A3 性能满足任务书倍率

# 详细设计（required）

## 算子分析

### 数学公式

out = beta * input + alpha * (op(A) * op(B))

C++ 接口等价形式：

C = alpha * op(A) * op(B) + beta * C

### 支持数据类型

float16、bfloat16、float32、complex64。

实数 computeType 为 float32（fp16/bf16 在核内按 float32 累加）；complex64 的 computeType 为 complex64。

CSR 索引仅支持 int32，且 rowOffsets 与 colInd 类型必须相同。

### 支持形状

mat1、mat2 为二维矩阵，op 生效后满足 mat1 列数等于 mat2 行数。

input 可广播到 [M, N]，例如 [M, N]、[N]、[1, N]、[M, 1]。

M、K、N、nnz 允许为 0。

B、C 支持行主（ld >= cols）与列主（ld >= rows）。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

调用流程为三步：GetBufferSize 查询 workspace，Preprocess 完成行重排与分桶并写入 tiling，SpMM 在调用方 stream 上异步 Launch Kernel。

校验 CSR 格式、int32 索引、dtype 组合、维度、ld、以及 ALG3 对 opA / opB 的限制。未声明组合返回明确错误码。

workspace 包含 tiling、行重排表与分桶边界。opA 为转置或共轭转置时，在 workspace 中将 CSR 转为 CSC，再走同一 Kernel。

分核：blockDim 取设备 AICore 数，上限 24。DEFAULT / ALG1 按行 nnz 贪心装箱；ALG2 按行序装箱。ALG1 走列主 gather，ALG2 / DEFAULT 走行主拷贝。ALG3 与 DEFAULT 相同，但限制 opA 为非转置，且不支持 opB 共轭转置。

ATen 层负责参数校验、广播、非连续 Tensor 处理、标量转换，并注册 `aten::_sparse_addmm` 到 PrivateUse1 与 SparseCsrPrivateUse1。workspace 使用 NPU byte tensor，不在适配层做无必要 Host 同步。

##### 1. 分核策略：

优先使用满核。行按 nnz 或行序分到各核；核间不能均分时，将余出行分配到前若干核。空行核若无行可处理则直接返回。

##### 2. 数据分块和内存优化策略：

按 UB 容量切 N 方向 tile。ROW 且 opB 非转置时用 DataCopyPad 搬 B 行；COL 或转置时按元素 gather。

当一行列号连续、N 较小且 B 为紧凑行主时，一次搬入该行对应的 B panel，减少逐 nnz 小拷贝。

##### 3. tilingkey规划策略：

通过 tiling 传递 M、K、N、nnz、idxBase、order、ld、opB、prefCol、alpha/beta。dtype 与 C64 走不同 Kernel 入口，不额外使用 tilingkey 区分硬件。

#### 3.2.2 kernel侧设计：

Init 绑定 GM / UB 并按 tiling 初始化队列；Process 按核内行区间与 N tile 循环。

1. 空行只计算 beta * C（beta=0 时写 0），不占用 B 双缓冲。
2. 非空行对每个 nnz 做 acc += alpha * A_val * B[col, nTile]。
3. fp16/bf16 先转到 float32 再向量乘加；bf16 在 dav-2201 上无向量 Cast，转换走标量。
4. complex64 按实部/虚部分量完成复数乘加；opB 为共轭转置时对 B 取共轭。
5. 行主输出块拷贝写回；列主输出按元素写回。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 | √ |
| Atlas A3 | √ |

A2 / A3 共用 arch22（dav-2201）实现，编译 SOC 分别为 ascend910b3 与 ascend910_93。

## 算子约束限制

- 仅支持 CSR，不支持 COO 及对外 CSC
- 索引仅支持 int32，不支持 int64
- 三输入必须同 dtype、同 NPU 设备，不执行 dtype 提升
- 仅 input 广播，mat1 / mat2 不广播
- 仅支持任务书 / cuSPARSE 13.3 声明允许的 alg、op、layout、dtype 组合
- beta=0 时仍校验 input 的 shape、dtype、device，但不读取 input 中的 NaN/Inf
- 不要求图融合
- 核心计算不允许 CPU fallback

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差单标杆，匹配率不低于 0.99，并满足绝对误差硬上限 | 任务书；生态算子开源精度标准 |
| 性能标准 | 仅 A3：每个场景倍率大于 0.25 倍 A100，全部场景算术平均不低于 0.35 倍 | 任务书 |

## 兼容性分析

复用已有 `aclsparseSpMM*` 公开签名，保持源代码兼容。CSR_ALG2 / CSR_ALG3 在枚举末尾追加。arch22 与 arch35 分目录，Host 公共描述符与硬件差异解耦，便于与 A5 任务共存。

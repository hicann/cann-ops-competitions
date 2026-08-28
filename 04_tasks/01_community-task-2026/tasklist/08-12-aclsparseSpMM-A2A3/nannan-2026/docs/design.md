# 需求背景（required）

## 需求来源

本设计面向 CANN 社区任务“08-12-aclsparseSpMM 算子开发（A2/A3）”。任务要求在 Ascend A2、Ascend A3 上补齐 `ops-sparse` 仓库已有 `aclsparseSpMM` 能力，并打通 Python/PyTorch、ATen NPU、aclsparse C++ 接口、Host 与 Ascend C Kernel 全链路。

代码基于 `ops-sparse` 现有目录演进，不重复新增同名接口，主要涉及：

- `include/cann_ops_sparse.h`
- `sparse/spmm/arch22/`
- `test/spmm/arch22/`
- `python_adapter/`

## 背景介绍

SpMM（Sparse Matrix-Matrix Multiplication）用于计算 CSR 稀疏矩阵与稠密矩阵的乘积：

```text
C = alpha * op(A) * op(B) + beta * C
```

其中：

- `A` 为 CSR 格式稀疏矩阵；
- `B`、`C` 为稠密矩阵；
- `op(A)`、`op(B)` 可为不转置、转置或共轭转置；
- `alpha`、`beta` 为标量；
- Python 侧对应 `torch.sparse.addmm(input, mat1, mat2, beta, alpha)`。

SpMM 广泛用于图计算、推荐系统、稀疏神经网络和科学计算。任务重点不是新增另一套孤立实现，而是在已有 aclsparse SpMM 三阶段接口上补齐 A2/A3 数据类型、布局、转置、边界、Python/ATen 接入及测试能力。

## 现状与问题

原 arch22 实现的能力较窄，主要面向 CSR、FP32、非转置和紧凑行主序场景，无法满足任务书要求。需要解决的问题包括：

1. 数据类型需覆盖 `float16`、`bfloat16`、`float32`、`complex64`。
2. CSR 索引基址需支持 base-0 和 base-1。
3. 稠密矩阵需支持行主序、列主序以及合法 leading dimension。
4. `opA`、`opB` 需支持转置；复数场景需正确处理共轭转置。
5. 需要覆盖空稀疏矩阵、空行、中间空行、长尾行和合法边界输入。
6. Python/ATen 路径必须真正调度到 NPU，不允许回退 CPU。
7. 需要提供可复现的 C++ UT、端到端测试与 ATK 精度验证。

# 需求分析（required）

## 需求描述

在 A2/A3 上实现并验证完整的 aclsparse SpMM 链路，保持现有公开接口：

```cpp
aclsparseSpMMGetBufferSize(...);
aclsparseSpMMPreprocess(...);
aclsparseSpMM(...);
```

同时通过独立 ATen 扩展注册 `SparseCsrPrivateUse1`，将 NPU 上的 `aten::_sparse_addmm` 转发到 `aclsparseSpMM`。

## 功能拆解

| 能力 | 设计要求 |
| --- | --- |
| 稀疏格式 | CSR |
| CSR 索引 | int32 行偏移与列索引 |
| 索引基址 | base-0、base-1 |
| 数据类型 | float16、bfloat16、float32、complex64 |
| 计算类型 | 实数输入使用 FP32 累加；complex64 使用复数计算 |
| 稠密布局 | B/C 行主序、列主序，支持合法 `ld` |
| `opA` | NON_TRANSPOSE、TRANSPOSE、CONJUGATE_TRANSPOSE |
| `opB` | NON_TRANSPOSE、TRANSPOSE、CONJUGATE_TRANSPOSE |
| 算法 | DEFAULT、CSR_ALG1、CSR_ALG2、CSR_ALG3（受约束场景） |
| 标量 | 支持实数及 complex64 的 `alpha`、`beta` |
| 边界 | `nnz=0`、空行、中间空行、不同 shape/稀疏度、padding |
| Python | `torch.sparse.addmm` NPU 调度，不回退 CPU |

## 非功能需求

1. 对不支持的格式、索引类型、数据类型、布局、维度和算法组合返回明确错误码。
2. 输入描述符与输入数据只读，输出只写入合法矩阵区域，不覆盖 padding。
3. Kernel 异步执行，遵守调用方 stream；测试计时前后显式同步。
4. 代码沿用仓库既有描述符、Handle、CMake 和安装机制。
5. 精度按生态算子开源精度标准的混合容差单标杆方法验收。

# 详细设计（required）

## 算子分析

### 数学定义

设原始 CSR 矩阵 `A` 形状为 `m x k`，稠密矩阵 `B` 与操作类型共同决定输出列数 `n`，则：

```text
C(i,j) = alpha * sum_t(op(A)(i,t) * op(B)(t,j)) + beta * C(i,j)
```

对 complex64：

- `TRANSPOSE` 仅交换行列；
- `CONJUGATE_TRANSPOSE` 在交换行列的同时对元素取共轭；
- `alpha`、`beta` 的实部与虚部均参与计算。

### 形状推导

Host 侧根据 `opA` 和 `opB` 推导逻辑矩阵尺寸：

- `opA == NON_TRANSPOSE`：输出行数为 `A.rows`，归约维为 `A.cols`；
- 否则：输出行数为 `A.cols`，归约维为 `A.rows`；
- `opB == NON_TRANSPOSE`：`B.rows` 必须等于归约维，输出列数为 `B.cols`；
- 否则：`B.cols` 必须等于归约维，输出列数为 `B.rows`；
- `C` 的逻辑形状必须等于推导出的输出形状。

## 总体架构

```text
torch.sparse.addmm
        |
        v
ATen SparseCsrPrivateUse1 adapter
        |
        v
aclsparseSpMMGetBufferSize
        |
        v
aclsparseSpMMPreprocess
        |
        v
aclsparseSpMM
        |
        v
arch22 Host validation / tiling / launch
        |
        v
Ascend C Kernel (FP16/BF16/FP32/Complex64)
```

核心库不强制依赖 PyTorch。ATen 适配器以独立动态库构建，仅在用户显式加载或测试进程通过 `sitecustomize.py` 启用时注册。

## Host 侧设计

### 参数校验

三个公开入口共用统一校验逻辑，校验内容包括：

1. 描述符和输出参数指针非空。
2. `A` 必须为 CSR 格式。
3. CSR 行偏移与列索引必须为支持的 int32 类型。
4. `indexBase` 仅接受 ZERO 或 ONE。
5. A/B/C 元素数据类型必须一致。
6. 实数类型使用 `ACL_FLOAT` 作为 computeType；complex64 使用 `ACL_COMPLEX64`。
7. `opA`、`opB` 必须属于支持枚举。
8. 根据转置状态校验 A/B/C 逻辑维度。
9. 行主序要求 `ld >= cols`，列主序要求 `ld >= rows`。
10. 算法必须为 DEFAULT、CSR_ALG1、CSR_ALG2 或 CSR_ALG3。
11. CSR_ALG3 限定 `opA` 为 NON_TRANSPOSE，且 `opB` 不能为 CONJUGATE_TRANSPOSE。

非法组合在 Kernel 启动前返回 `INVALID_VALUE`、`NOT_SUPPORTED` 或矩阵格式相关错误，防止静默错算和越界访问。

### Workspace 布局

workspace 按 64 字节边界组织，包含：

```text
| header | tiling data | row reorder table | bin edge table |
```

大小由 `SpmmArch22WorkspaceSize(M, blockDim)` 计算：

- header：固定 64 字节；
- tiling data：`SpmmArch22TilingData` 向上对齐；
- reorder table：按输出/源行数存放 int32 行号；
- bin edge table：按 block 数存放边界。

`GetBufferSize` 返回总字节数；`Preprocess` 构建并写入 tiling 和预处理数据；`SpMM` 复用同一 buffer 启动计算。

### Tiling 数据

Host 向 Kernel 传递的主要字段包括：

- `M/K/N/nnz`
- `blockDim`、`nChunks`
- `dataType`
- `indexBase`
- `orderB/orderC`
- `opA/opB`
- `sourceRows`
- `ldb/ldc`
- `alpha/beta` 实部和虚部
- workspace 中 reorder/bin 数据偏移

### 分核与 N 维分块

arch22 最大使用 24 个计算核。输出行或源行按核切分；N 维根据 UB 容量切成多个 chunk，避免大 N 场景超过 UB。

实数 Kernel 预留输入、计算和输出 tile 空间，并基于约 192 KiB UB、预留开销和 8 元素对齐计算 `nChunks`。尾块单独按有效长度处理。

## Kernel 侧设计

### 实数路径

FP16、BF16、FP32 共用模板化执行框架：

1. 从 CSR rowOffsets 获取当前行 `[rowStart, rowEnd)`。
2. 对 base-1 索引统一减去 `indexBase`，转换为零基逻辑下标。
3. 按 N chunk 加载或访问 B。
4. 使用 FP32 进行乘加累积。
5. 计算 `alpha * accumulator + beta * oldC`。
6. 根据目标数据类型写回 FP16、BF16 或 FP32。

当 `beta == 0` 时不读取原 C，避免输入中的 NaN/Inf 对结果产生不符合接口语义的污染。

### complex64 路径

complex64 使用显式实部/虚部运算：

```text
(ar + i*ai) * (br + i*bi)
= (ar*br - ai*bi) + i*(ar*bi + ai*br)
```

累加后再与复数 `alpha`、`beta` 组合。对于 `CONJUGATE_TRANSPOSE`，读取 A 或 B 时将虚部取反。

### 转置 A

CSR 天然按原始行组织。`opA == NON_TRANSPOSE` 时按输出行直接遍历 CSR 行；`opA` 为 TRANSPOSE 或 CONJUGATE_TRANSPOSE 时，按源行遍历非零元素并将原列索引映射为输出行。复数共轭转置额外对 A 值取共轭。

### B/C 布局与 leading dimension

Kernel 根据 `orderB/orderC` 和 `ldb/ldc` 计算物理地址：

- 行主序：`offset = row * ld + col`
- 列主序：`offset = col * ld + row`

`opB` 决定逻辑 `(row, col)` 到物理 B 坐标的映射；共轭转置同时翻转虚部。C 仅写逻辑矩阵区域，padding 保持不变。

### 边界处理

- `nnz == 0`：乘积项为零，仅按 `beta` 处理 C。
- 空行：累加值为零，不访问 values/indices。
- 中间空行：每行独立读取 rowOffsets，不假设行内一定存在非零元素。
- N 尾块：仅处理有效列数。
- base-1：rowOffsets 和 colIndices 在使用时统一扣除基址。

## 算法设计

| 算法 | 设计 |
| --- | --- |
| DEFAULT | 默认确定性 CSR 路径 |
| CSR_ALG1 | 与默认 SIMT/AIV 路径一致 |
| CSR_ALG2 | 行导向确定性路径 |
| CSR_ALG3 | 使用预处理数据的 CSR 路径；限制 opA/opB 组合 |

算法选择由 Host 校验并写入相应预处理/执行流程。不支持的组合明确返回错误，而不是隐式降级。

## ATen NPU 适配设计

`python_adapter/sparse_addmm_npu.cpp` 注册 `SparseCsrPrivateUse1` Kernel，将 `aten::_sparse_addmm` 转发到 aclsparse：

1. 校验输入为单设备 CSR NPU Tensor。
2. 支持 int32 CSR 索引及 FP16/BF16/FP32/Complex64 values。
3. 将广播的 dense input 展开为输出形状。
4. 将非连续 dense input/materialized view 转换为连续 NPU Tensor。
5. 创建 aclsparse Handle、CSR 描述符和稠密矩阵描述符。
6. 依次执行 GetBufferSize、Preprocess、SpMM。
7. 返回 NPU Tensor，不执行 CPU 搬运或 CPU 计算回退。

扩展以独立 `.so` 构建，避免核心 `ops-sparse` 引入强制 PyTorch 依赖。测试工作进程可通过以下环境变量自动加载：

```bash
export PYTHONPATH=$PWD/python_adapter:${PYTHONPATH}
export OPS_SPARSE_TORCH_LIBRARY=$PWD/python_adapter/build/ops_sparse_torch_npu.so
```

未设置变量时 `sitecustomize.py` 不产生副作用；设置后若动态库不存在或注册失败，则显式报错，避免静默回退。

## 支持硬件

| 支持的芯片版本 | 支持情况 |
| --- | --- |
| Atlas A2（arch22 / DAV-2201） | 支持 |
| Atlas A3（arch22 / DAV-2201） | 支持 |

## 约束限制

1. 稀疏矩阵格式仅支持 CSR。
2. CSR 行偏移与列索引当前支持 int32。
3. A/B/C 元素类型必须一致。
4. 实数类型使用 FP32 computeType；complex64 使用 complex64 computeType。
5. CSR_ALG3 不支持转置 A，也不支持 B 的共轭转置。
6. 输入必须位于同一 NPU 设备。
7. Kernel 为异步执行，调用方如需读取结果或计时必须同步 stream。

# 可维可测分析

## 精度标准

精度按《生态算子开源精度标准》中的混合容差单标杆方法执行。测试覆盖：

- float16、bfloat16、float32、complex64；
- base-0、base-1；
- 行主序、列主序、带 padding 的 leading dimension；
- opA/opB 转置及复数共轭转置；
- alpha/beta 为 0、1、负数和复数；
- 不同 M/K/N、nnz、稀疏度、空行、长尾行；
- NaN/Inf 规则及 `beta == 0` 不读取输入 C 的语义；
- 非连续 PyTorch Tensor；
- 输入只读和输出 padding 边界。

A3 使用 ATK 执行 200 个精度用例，结果为 200/200 通过，准确率 100%。A2 完整报告随最终自测报告补充。

## 性能标准与测量方案

性能测试仅在 A3 提交，采用任务书规定的同范围调用与基线：

1. 预热至少 10 次；
2. 正式采样至少 30 次；
3. 每轮调用后同步设备；
4. 报告 Kernel 总耗时中位数和 P90；
5. 描述符、workspace 和 preprocess 结果在正式采样期间复用；
6. 不计首次编译、数据生成、Host 到 Device 搬运及无关初始化；
7. Python 端到端耗时和 C++ 执行耗时作为补充数据；
8. 使用 Profiler 证明核心计算调度至 NPU，未回退 CPU。

性能数值和 Profiler 截图在完成 A3 性能采集后写入自测报告，不在设计阶段虚构结论。

## 可维护性

1. 三个公开入口复用统一参数校验。
2. 数据类型通过模板实例化，减少重复控制逻辑。
3. Host 与 Kernel 通过明确 tiling 结构通信。
4. ATen 适配器与核心库解耦。
5. 不支持场景返回明确状态码。
6. C++ smoke、选择性 case、ATen 端到端和 ATK 全量测试形成分层回归体系。

## 兼容性分析

本设计复用现有公开 `aclsparseSpMM*` 接口并扩展原 arch22 能力，不新增同名接口。已有受支持场景继续走默认路径；新增数据类型、布局、转置和算法组合通过描述符与枚举表达。对非法或尚不支持的组合采用 fail-fast，不改变合法调用的返回语义。

# 测试设计

## C++ 测试

`test/spmm/arch22/spmm_test.cpp` 提供：

- 200 个基础 shape/nnz/alpha/beta case；
- 环境变量 `SPMM_CASE_INDEX` 选择单例运行；
- FP16/BF16 dtype smoke；
- layout/base/algorithm smoke；
- complex64 和转置 smoke；
- `nnz=0`、中间空行等边界 smoke。

## Python/ATen 测试

`python_adapter/test_sparse_addmm_npu.py` 覆盖 FP16、BF16、FP32、Complex64、广播、`beta == 0`、非连续输入，并检查 `SparseCsrPrivateUse1` 已注册。

## ATK 测试

ATK 测试工程提供：

- 200 个精度用例 JSON；
- 混合容差比较器；
- CPU Golden 与 NPU Under Test；
- A3 最终结果：`Total Task: 200, success 200, failed 0`，`acc_pass_result:Pass`。

# 风险与应对

| 风险 | 应对措施 |
| --- | --- |
| 大 N 导致 UB 超限 | 按 UB 容量计算 N chunk，并处理尾块 |
| base-1 造成越界 | Host 校验枚举，Kernel 统一减基址 |
| 转置维度错配 | Host 根据 opA/opB 推导并校验逻辑形状 |
| 列主序或 padding 写坏 | 统一布局寻址函数，测试 padding 不被覆盖 |
| complex64 共轭语义错误 | A/B 分别按 op 类型处理虚部符号，专项用例验证 |
| beta=0 仍读取 C | beta 为零时跳过旧 C 读取 |
| Python 静默回退 CPU | 显式注册与 dispatch table 检查，加载失败直接报错 |
| CPU Golden 对特殊非连续复数视图行为异常 | Golden 使用语义等价的连续输入，NPU 侧保留非连续输入覆盖，并记录诊断证据 |

# 版本与交付信息

| 项目 | 信息 |
| --- | --- |
| 个人代码仓 | `https://gitcode.com/nannan-2026/ops-sparse` |
| 开发分支 | `feature/aclsparse-spmm-a2a3` |
| 验证提交 | `f385698d7c0b25890f7c3ef6acf24d5b3b933b86` |
| CANN | 9.0.0 |
| A3 精度结果 | ATK 200/200 Pass，100% |
| 测试报告 | 随验收交付件提交 |

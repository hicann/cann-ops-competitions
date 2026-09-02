# 需求背景（required）

## 需求来源

本文对应 2026 年 9 月社区任务《aclsparseSDDMM 算子开发任务书（A5）》，目标是在 Ascend 950PR（DAV_3510，`arch35`）上完善 `aclsparseSDDMM`，并将 C++ 接口、Host、Ascend C Kernel、Python/ATen 适配和测试统一交付到 `ops-sparse` 仓库。

设计依据如下：

1. 任务书：`aclsparseSDDMM_A5_task_doc.md`。
2. 任务测试包：`test_cases/aclsparseSDDMM_testCase/`、`test_cases/common/` 和 `test_cases/baseline_results/`。
3. 社区任务设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`。
4. 目标代码仓：`https://gitcode.com/cann/ops-sparse`，目标分支为 `master`。
5. C++ 接口语义：任务书给出的 `aclsparseSDDMM*`、BSR 描述符及 strided-batch 接口，语义参考 cuSPARSE SDDMM。
6. Python/ATen 语义：PyTorch 2.7 及以上版本的 `torch.sparse.sampled_addmm` 和 `aten::sparse_sampled_addmm`。
7. 软件环境：CANN 9.1.0 及后续配套版本、PyTorch 2.7+、torch_npu 26.0.0+。
8. 精度标准：《生态算子开源精度标准》实验标准及任务书 3.2 节规定的单标杆混合容差。

本文只描述拟实现的技术方案和验证方法。功能、精度、性能、内存及 Profiler 数据必须在代码完成后从 Ascend 950PR 实测取得；设计阶段不把目标值写成已达成结果。

## 背景介绍

### SDDMM 功能

SDDMM（Sampled Dense-Dense Matrix Multiplication）只在稀疏矩阵 `C` 已有 pattern 的位置计算两个稠密矩阵的乘积：

```text
C_out = (alpha * op(X) * op(Y) + beta * C_in) .* spy(C)
```

其中 `spy(C)` 只表示 `C` 的稀疏结构。执行前后 CSR/BSR 的 offsets、indices、shape、base、块大小和块内布局保持不变，只原地更新 `C.values`。

对于 CSR 的非零元素 `p`，设其逻辑坐标为 `(i,j)`：

```text
C.values[p] = alpha * sum(op(X)[i,k] * op(Y)[k,j], k=0..K-1)
              + beta * C.values[p]
```

对于 BSR 的非零块 `(br,bc)` 及块内元素 `(ri,cj)`，逻辑坐标为：

```text
i = br * blockSize + ri
j = bc * blockSize + cj
```

每个非零块内的全部 `blockSize * blockSize` 个 values 都按同一公式更新；块内物理地址由 BSR `order=ROW/COL` 决定。

### ops-sparse 现状分析

设计阶段基于本地 `ops-sparse upstream/master` 提交 `f8a1151f12508bedeecff84e01fc9174dfab1d51` 分析现有实现。已有代码位于：

```text
sparse/sddmm/
├── README.md
└── arch35/
    ├── sddmm.h
    ├── sddmm_host.cpp
    ├── sddmm_kernel.cpp
    └── sddmm_kernel.h

test/sddmm/
├── CMakeLists.txt
└── sddmm/arch35/
```

可复用的基线能力包括三阶段接口、DnMat/CSR 描述符、调用方 stream、CSR 行分箱、workspace tiling 以及 FP16/FP32 的 Ascend C 向量计算框架。基线与本任务的差距如下：

| 层级 | 基线现状 | 本任务必须补齐 |
| --- | --- | --- |
| 稀疏格式 | 仅 CSR | CSR 和方块 BSR |
| dtype | CSR FP16/FP32，且组合较窄 | CSR FP16/FP32/complex64；BSR FP16/BF16/FP32/complex64及混合组合 |
| 计算精度 | FP16/FP32 同类型为主 | FP16/BF16 输入 FP32 累加；FP16/BF16 values 或 FP32 values；complex64 复数计算 |
| 索引 | I32、base 0 | I32、base 0/1 |
| 描述符 | 无本任务 BSR 与 batch 接口 | mutable/const BSR、DnMat Get/Set strided batch、BSR strided batch |
| 操作和布局 | N/T、DnMat ROW/COL 的部分组合 | N/T 全组合、ld/stride 校验、BSR 块内 ROW/COL |
| batch | 单 batch | 四类 X/Y/C strided-batch 广播组合，`batchCount <= 65535` |
| pattern cache | workspace 指针为 key，signature 只含 rows/nnz/k | 状态绑定 `matC`，完整识别 offsets/indices pattern，独立描述符/workspace 可并发 |
| Python/ATen | 无 `sampled_addmm` NPU 全链路 | NPU Dispatcher、参数校验、输出构造、异常/alias/异步语义，不得 CPU fallback |
| 测试 | 基础 C++ CSV | 任务包 200 条 ATK 精度、242 条性能/泛化清单及补充 C++/Python/异常/Profiler 测试 |

现有 SDDMM Host 在 Preprocess 中把 CSR row offsets 搬到 Host 做重排，且执行路径依赖进程全局 side table。新设计不沿用该状态模型；公共描述符、预处理计划和架构 Kernel 分层实现，避免与 A2/A3 的后续实现耦合。

### 任务测试包分析

任务包提供以下不可删减的验证输入：

- `accuracy_cases.json`：200 条 ATK 泛化用例，包含 CSR 88 条、BSR 112 条，覆盖 FP16/BF16/FP32/complex64、base 0/1、batch 1/2、ROW/COL、均匀/长尾/空行等 pattern。
- `performance_cases.json`：242 条性能与泛化配置，包含 CSR 118 条、BSR 124 条；P-01/P-02/P-03 以及 200 条 extra 规模，base 0/1、块大小 2/4/8/16/32/64 和多种分布。
- `generate_cases.py`：固定 P 场景、dtype、base、block、seed 及扩展用例生成规则。
- `function_sparse_ops.py`、`accuracy_sparse_ops.py`：ATK NPU 执行及 CPU Golden 比对入口。
- `benchmark_sparse_ops_npu.py`、内存采集与比较脚本：NPU 性能、峰值内存及结果汇总入口。
- `gpu_performance_result_benchmark.md`：任务包给出的逐 case GPU 基线及跳过原因。

附件仍需按任务书修正和补充，不能把“原脚本运行结束”等同于验收完成：

- 当前 `operator_adapter.reference()` 对所有实数选择 FP64；实现测试时必须改为 FP16/BF16 使用 FP32、FP32 使用 FP64、complex64 使用 complex128。
- 当前 ATK comparator 需要显式实现或配置任务书的逐元素混合容差、0.99 匹配率和 `max(A, 32*ULP)` 硬上限，complex64 实虚部分开统计。
- 200 条 JSON 主要编码 format/dtype/shape/base/block/direction/batch/pattern，不能替代 `opX/opY`、DnMat ROW/COL、padding ld、alpha/beta、computeType、algorithm、pointer mode 和错误码专项测试。
- NPU hook `torch.ops.ops_sparse_test.sddmm_npu` 必须连接真实 aclsparse/arch35 实现；验收模式禁止开启 reference fallback。

任务书 3.3 节摘要耗时区间与随包 `gpu_performance_result_benchmark.md` 的部分逐 case 数值不一致。实现和自测不自行选择更宽松值：性能倍率统一按 `GPU 对应接口 median_us / NPU 同调用范围 median_us` 计算；正式验收前以任务方确认的 canonical 逐 case 基线为准，并在报告中同时记录任务书版本、测试包校验值及确认结论。

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上完成 `aclsparseSDDMM` 的完整工程化实现：

1. 复用 DnMat/SpMat/Handle，提供 BufferSize、Preprocess、Execute 三阶段接口。
2. 支持 CSR/BSR、I32 index、base 0/1、ROW/COL 布局、N/T、strided batch 和动态 shape/nnz。
3. 打通任务书声明的全部 dtype/computeType 组合，包含必选 complex64 和 BSR BF16 混合精度。
4. 提供 `torch.sparse.sampled_addmm` 到 `aten::sparse_sampled_addmm` 的 NPU 适配，核心计算不得回退 CPU。
5. 满足任务书精度、P-01/P-02/P-03 性能、workspace/峰值内存、并发、生命周期和可复现测试要求。
6. A5 实现放入 `sparse/sddmm/arch35/`，公共接口、描述符和 Host 逻辑与 A2/A3 架构代码解耦。

## 需求拆解

| 编号 | 子需求 | 设计落点 |
| --- | --- | --- |
| R1 | 三阶段 C++ API | 公共校验、workspace 规划、device preprocess、异步 execute |
| R2 | CSR/BSR 描述符 | 扩展公共声明和内部 SpMat 元数据，补齐 create/const create/destroy/strided batch |
| R3 | dtype 全矩阵 | Host dispatch 与 Kernel 模板：FP16/BF16/FP32/complex64，FP32 或 complex64 累加 |
| R4 | op/layout/base | 统一逻辑维度推导，ROW/COL + ld 地址器，base 0/1 归一化 |
| R5 | batch | X/Y/C 三方 batchCount/stride 校验及四种广播组合 |
| R6 | pattern plan | `matC` 专属计划、完整 signature、失效规则、并发边界 |
| R7 | Python/ATen | SparseCsr NPU dispatch、函数式输出、当前 stream、无 CPU fallback |
| R8 | 泛化 | 动态 tiling、短行/长尾/空行/zero-nnz、块大小 2～128、通用 fallback |
| R9 | 精度 | CPU 单标杆、混合容差、硬上限、complex 实虚部分判定、INF/NAN |
| R10 | 性能 | P-01～P-03 固定输入，预热 10/采样 30，Kernel 总耗时及 0.3x 门槛 |
| R11 | 内存 | 精确 BufferSize、无完整稠密中间矩阵、workspace/L2 或 50% 规则 |
| R12 | 工程与回归 | `arch35` 目录、公开头文件、C++/ATen/Python UT、A2/A3 交叉回归 |

# 详细设计（required）

## 算子分析

### 数学公式与操作语义

逻辑维度统一按 `op` 后矩阵推导：

```text
op(X): [M,K]
op(Y): [K,N]
C:     [M,N]
```

`opX/opY` 只接受 `ACL_SPARSE_OP_NON_TRANSPOSE` 和 `ACL_SPARSE_OP_TRANSPOSE`。本任务不声明 CONJUGATE_TRANSPOSE；complex64 的 TRANSPOSE 只交换下标，不执行共轭。

存储 shape 与逻辑 shape 的对应关系：

| op | X 描述符存储 shape | Y 描述符存储 shape |
| --- | --- | --- |
| N | `[M,K]` | `[K,N]` |
| T | `[K,M]` | `[N,K]` |

X/Y 的 ROW/COL order 和 leading dimension 独立。Row-major 要求 `ld >= stored_cols`；Column-major 要求 `ld >= stored_rows`。所有地址乘加先使用有符号/无符号 64 位安全运算，溢出或超过 Kernel 支持范围时返回错误，不发生截断。

### 支持数据类型

| 稀疏格式 | X/Y valueType | C valueType | computeType | 内部累加 | 输出 |
| --- | --- | --- | --- | --- | --- |
| CSR/BSR | FP32 | FP32 | FP32 | FP32 | FP32 |
| CSR/BSR | complex64 | complex64 | complex64 | complex64 | complex64 |
| CSR/BSR | FP16 | FP32 | FP32 | FP32 | FP32 |
| CSR/BSR | FP16 | FP16 | FP32 | FP32 | FP16 |
| BSR | BF16 | FP32 | FP32 | FP32 | FP32 |
| BSR | BF16 | BF16 | FP32 | FP32 | BF16 |

说明：

- FP16/BF16 输入先转换为 FP32，K 维归约、alpha/beta 乘加均使用 FP32；写回低精度 values 时按 CANN 对应类型的 round-to-nearest 规则转换。
- complex64 外部 ABI 为连续两个 FP32（real、imag）。点积使用普通复数乘法，不做共轭；`alpha/beta` 为 complex64 标量。
- 不在表中的组合返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`，不隐式转换或降级。
- `beta=0` 时不读取旧 `C.values`，因此旧 values 中的 NaN/Inf 不应传播；`alpha=0` 仍保持结构并正确执行 `beta*C`。

### 支持 shape、格式和属性

| 项目 | 支持范围 |
| --- | --- |
| M/N/K | 动态非负整数；地址、元素数和 workspace 计算不得溢出，Kernel launch 前校验可表示范围 |
| CSR | `rowOffsets[M+1]`、`colInd[nnz]`、`values[nnz]`，I32，base 0/1 |
| BSR | 方块，block size 2/4/8/16/32/64/128，I32，base 0/1，块内 ROW/COL |
| BSR values | `blockNnz * blockSize * blockSize`，按 direction 编码 |
| 稀疏分布 | 均匀、长尾、one-long-row、many-empty-rows、banded、random、near-dense、zero-nnz |
| DnMat | ROW/COL、合法 padding ld、N/T、strided batch |
| batch | 1～65535；stride 非负且覆盖单 batch 存储范围 |
| 算法 | `ACL_SPARSE_SDDMM_ALG_DEFAULT`；非法枚举返回 NOT_SUPPORTED |

CSR/BSR offsets 必须单调，首尾与 base、nnz/blockNnz 一致，indices 扣除 base 后在合法列范围内。C 的结构内存只读；X/Y 只读；C values 原地更新。

### batch 广播语义

描述符 batchCount 为 1 时对所有输出 batch 广播；大于 1 时必须与输出 C 的 batchCount 相同。支持：

| 模式 | X batch | Y batch | C batch | 语义 |
| --- | ---: | ---: | ---: | --- |
| B0 | 1 | 1 | B | `C_i = X * Y` |
| B1 | B | 1 | B | `C_i = X_i * Y` |
| B2 | 1 | B | B | `C_i = X * Y_i` |
| B3 | B | B | B | `C_i = X_i * Y_i` |

CSR 没有新增专用 strided-batch setter时，Python 必测路径为单 batch；任务 C++ 范围内 BSR 使用 `aclsparseBsrSetStridedBatch`，DnMat 使用 `aclsparseDnMatSetStridedBatch`。batch stride 以元素为单位，Host 转字节地址时检查乘法溢出。

## 算子实现

### 总体架构

```text
torch.sparse.sampled_addmm
        |
        v
aten::sparse_sampled_addmm (SparseCsr NPU dispatch)
        |
        +-- dtype/device/shape/layout/stride/alpha/beta 校验
        +-- 构造函数式输出 CSR（复制 pattern 与初始 values）
        +-- 当前 NPU stream + RAII 描述符/workspace
        v
aclsparseSDDMMBufferSize
        |
aclsparseSDDMMPreprocess  -- plan 绑定 matC pattern
        |
aclsparseSDDMM            -- arch35 Host dispatch
        |
        +-- CSR generic / CSR K=128 fast path
        +-- BSR generic / BSR block-tiled fast path
        v
Ascend 950PR AIV Kernel -> 只更新 sparse C.values
```

Host 只处理描述符元数据、pointer-mode 标量、workspace/tiling 和 launch；不在 CPU 计算矩阵结果。稀疏结构分析在 device preprocess kernel 上完成并排入调用方 stream，避免现有 D2H row-offset 重排造成的 Host 同步。

### 代码组织

计划按上游仓库现有规范修改/新增：

```text
include/cann_ops_sparse.h

sparse/common/aclsparse_descr_internal.h
sparse/common/aclsparse_descr.cpp
sparse/sddmm/README.md
sparse/sddmm/arch35/
├── sddmm.h
├── sddmm_host.cpp
├── sddmm_kernel.cpp
├── sddmm_kernel.h
└── sddmm_tiling_data.h

pytorch/sparse_sampled_addmm_npu.cpp
python/ops_sparse_npu/__init__.py

test/sddmm/CMakeLists.txt
test/sddmm/arch35/
├── sddmm_test.cpp
├── sddmm_test.csv
└── README.md
test/python/test_sparse_sampled_addmm.py

test_cases/aclsparseSDDMM_testCase/               # 任务专项精度/性能/内存脚本
test_cases/common/                                # 公共生成、执行和比对逻辑
test_cases/baseline_results/                      # 原始标杆结果和 manifest
```

A5 Kernel 只放在 `sparse/sddmm/arch35/`。dtype/shape/descriptor 的公共校验放在可复用公共层；架构能力分派后再进入 arch35 tiling/launch。若合入前 A2/A3 PR 已改变公共结构，本分支基于最新 master rebase，保留不同架构实现并完成交叉回归，不复制一套同名 API。

### 公共接口和描述符设计

公开函数签名严格采用任务书 2.3 节，不改参数顺序和已有枚举值：

- `aclsparseSDDMMBufferSize`
- `aclsparseSDDMMPreprocess`
- `aclsparseSDDMM`
- `aclsparseCreateBsr` / `aclsparseCreateConstBsr`
- `aclsparseDnMatGetStridedBatch` / `aclsparseDnMatSetStridedBatch`
- `aclsparseBsrSetStridedBatch`

接口原型如下：

```cpp
aclsparseStatus_t aclsparseSDDMMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, size_t *size);

aclsparseStatus_t aclsparseSDDMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer);

aclsparseStatus_t aclsparseSDDMM(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY, const void *beta,
    aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer);

aclsparseStatus_t aclsparseCreateBsr(
    aclsparseSpMatDescr_t *spMatDescr, int64_t blockRows,
    int64_t blockCols, int64_t blockNnz, int64_t rowBlockSize,
    int64_t colBlockSize, void *bsrRowOffsets, void *bsrColInd,
    void *bsrValues, aclsparseIndexType_t bsrRowOffsetsType,
    aclsparseIndexType_t bsrColIndType, aclsparseIndexBase_t idxBase,
    aclDataType valueType, aclsparseOrder_t order);

aclsparseStatus_t aclsparseCreateConstBsr(
    aclsparseConstSpMatDescr_t *spMatDescr, int64_t blockRows,
    int64_t blockCols, int64_t blockNnz, int64_t rowBlockSize,
    int64_t colBlockSize, const void *bsrRowOffsets,
    const void *bsrColInd, const void *bsrValues,
    aclsparseIndexType_t bsrRowOffsetsType,
    aclsparseIndexType_t bsrColIndType, aclsparseIndexBase_t idxBase,
    aclDataType valueType, aclsparseOrder_t order);

aclsparseStatus_t aclsparseDnMatGetStridedBatch(
    aclsparseConstDnMatDescr_t dnMatDescr, int *batchCount,
    int64_t *batchStride);

aclsparseStatus_t aclsparseDnMatSetStridedBatch(
    aclsparseDnMatDescr_t dnMatDescr, int batchCount,
    int64_t batchStride);

aclsparseStatus_t aclsparseBsrSetStridedBatch(
    aclsparseSpMatDescr_t spMatDescr, int batchCount,
    int64_t offsetsBatchStride, int64_t columnsBatchStride,
    int64_t valuesBatchStride);
```

三阶段公共参数契约如下：

| 参数 | 契约 |
| --- | --- |
| `handle` | 有效 aclsparse handle，提供调用方 stream 和 pointer mode |
| `opX/opY` | 仅 N/T；决定存储 shape 到逻辑 `[M,K]`、`[K,N]` 的映射 |
| `alpha/beta` | dtype 等于 computeType；Host/Device 位置由 pointer mode 决定 |
| `matX/matY` | const DnMat，数值与元数据只读，支持 ROW/COL、ld、strided batch |
| `matC` | mutable CSR/BSR SpMat；结构只读，values 原地更新，承载 SDDMM plan 状态 |
| `computeType` | 只接受能力矩阵中的 FP32/complex64 |
| `alg` | 只接受公开声明的 DEFAULT；其他值 NOT_SUPPORTED |
| `size` | BufferSize 输出，非空 Host `size_t*`，返回精确对齐字节数 |
| `buffer` | 调用方 Device workspace；Preprocess/Execute 期间有效并满足查询值和对齐 |

各阶段状态语义：BufferSize 只查询且不修改数据/plan；Preprocess 可更新 `matC` 内部 plan 和 workspace，但不更新 C values；Execute 只在 plan 合法或 correctness fallback 下更新 C values。Preprocess/Execute 都提交到 handle stream；除 Preprocess 为返回非法 Device pattern 的确定错误码而回读极小状态外，不做无关 Host 同步。

SpMat 内部新增 BSR 元数据：`blockRows/blockCols/blockNnz`、`rowBlockSize/colBlockSize`、`order`、三个 batch stride 和 batchCount。DnMat 内部新增 batchCount/batchStride。默认创建后 batchCount=1，stride 为 0；Set 接口校验 batchCount、最小 stride、溢出和描述符类型。

SpMat 再增加 SDDMM 专属 plan，禁止复用通用 `activeBuffer`：

```text
sddmmPlanValid
sddmmActiveBuffer
sddmmPatternVersion
sddmmPatternDigest
sddmmPatternSnapshotOffsets / Indices
sddmmPlanPtrs / sddmmPlanIdxs
sddmmPlanRows / Cols / Nnz
sddmmPlanFormat / Base / IndexTypes
sddmmPlanBlockGeometry / BlockOrder
sddmmPlanBatchAndStrides
sddmmPlanAlgorithm
```

指针、shape、nnz、base、format、block、batch 或算法改变立即使 plan 无效。Preprocess 将 offsets/indices 的完整设备快照、摘要、计划版本和 workspace header 一起绑定到 `matC`；摘要只用于快速筛选，最终身份以数组长度和逐元素相等为准，不依赖可能碰撞的 hash。Execute 在同一 stream 上由 validation kernel 比较当前 pattern 与快照；不匹配时不得使用旧重排数据，转入直接读取当前结构的正确性 fallback 并标记计划无效，调用者再次 Preprocess 后恢复优化计划。若超大 pattern 的完整快照会使 workspace 超过目标 L2 约束，则不建立依赖 pattern 内容的优化计划，Execute 始终使用结构无关路径。这样同指针原地改写 pattern 也不会误用旧计划，同时不引入 Host 同步。

descriptor setter、destroy 和重新 create 时清理 plan 元数据。descriptor 不拥有用户 X/Y/C 数据和 workspace；只保存弱引用及元数据。不同 `matC` + 独立 workspace 没有共享可变全局状态，可安全并发。同一 `matC` 的 Preprocess/Execute 并发由调用者串行，和任务书约束一致。

### 三阶段流程

#### BufferSize

处理顺序：

1. 校验 handle、alpha/beta、X/Y/C、size、枚举和描述符签名。
2. 推导 op 后 M/N/K，校验 dtype 矩阵、format、index、base、ld、block、batch 和指针。
3. 用 checked-add/checked-mul 计算 workspace，任何溢出返回 `INVALID_VALUE`。
4. 返回 64/32 字节对齐后的精确字节数；zero-nnz 也返回执行所需的最小 header 大小或 0，并保证后续接口契约一致。

workspace 不保存完整 `[M,N]` 稠密结果。布局为：

```text
[plan header + tiling]
[pattern digest / validation status]
[offsets + indices device snapshot（仅优化计划需要）]
[per-core work ranges]
[row/block work descriptors]
[可选长尾 split descriptors]
[device preprocess scan temporary area]
```

工作描述符数量受 `M`、`blockRows`、`nnz` 和核数约束。BufferSize 与 Preprocess 使用同一 `WorkspaceLayout` 计算函数，避免查询值与实际写入偏差。

实现对外明确采用以下数值边界：`M/N/K/nnz/blockRows/blockCols/blockNnz <= INT32_MAX`，`batchCount <= 65535`；BSR scalar-values 数量、batch 地址、dtype 字节数和 workspace 中间值使用 checked `uint64_t/size_t` 计算，必须不超过 `SIZE_MAX`、device 可寻址范围及可分配内存。workspace offset 使用 64 位字段，不因内部 32 位临时量静默截断。任何一项超限均在 launch 前返回 `INVALID_VALUE` 或 `INSUFFICIENT_RESOURCES`。

#### Preprocess

Preprocess 的结构分析和计划构造在调用方 stream 上由 Device Kernel 完成：

1. 保存 offsets/indices 完整设备快照并生成摘要，写 workspace header；不建立优化计划时记录 direct-plan 标志。
2. CSR 根据 row offsets 构造行工作量；BSR 根据 block offsets 和块面积构造工作量。
3. 通过 device scan/partition 生成各核近似等工作量范围；超过阈值的单长行/长块行切分为独立 work item。
4. 生成 generic/fast-path tiling、K tile、output tile、block specialization 和 batch 地址参数。
5. 回读固定大小的 validation status；非法 offsets/indices 返回明确错误，合法时在 `matC` 中登记 plan 元数据和 workspace 关联，不更改 C 结构或 values。

Preprocess 不读取 X/Y 数值，不把完整 pattern 搬到 Host，也不使用进程全局 pattern map。只为兑现结构错误码进行必要的状态同步；重复执行可复用描述符、workspace 和计划。

#### Execute

Execute 重新校验会随调用变化的标量、X/Y values 指针、stream 和描述符元数据；确认 plan 后下发 pattern validation 与 SDDMM kernel。Kernel launch 异步返回，不在 Host 侧同步。

- 若调用者跳过显式 Preprocess，Execute 在同一 workspace 中构造保守计划后执行，语义正确；重复调用建议显式 Preprocess。
- 若 buffer 为空而 BufferSize 非零，返回 `INVALID_VALUE`。
- ABI 不含实际 buffer size，因此只校验空指针、对齐和查询契约，不声称能可靠检测欠配。
- zero-nnz、M=0 或 N=0 不启动主计算 Kernel，保持结构和空 values 正确。

### Host 校验和错误码

| 条件 | 返回 |
| --- | --- |
| handle/描述符未初始化 | `ACL_SPARSE_STATUS_NOT_INITIALIZED` |
| 必需指针为空、负维度、非法 ld/stride、shape 不匹配、溢出 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 非 CSR/BSR、非 I32、非法 dtype/computeType、非法 op/alg/order/block | `ACL_SPARSE_STATUS_NOT_SUPPORTED` 或矩阵类型错误 |
| 非 A5 且无对应架构实现 | `ACL_SPARSE_STATUS_ARCH_MISMATCH` |
| workspace/内部资源不足 | `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` / `ALLOC_FAILED` |
| runtime/kernel launch 失败 | `ACL_SPARSE_STATUS_EXECUTION_FAILED` |

结构数据位于 Device，完整单调性/范围校验由 preprocess validation kernel 完成。需要把设备校验结果转换为同步 Host 返回码的调试/严格模式可执行必要同步；正式异步主路径不为合法输入增加无关同步。C++ UT 同时覆盖 Host 可判定错误和 Device 结构错误。

### tiling 与分核策略

DEFAULT 算法不是按测试 ID 硬编码，而是按公开元数据选择：

| 路径 | 选择条件 | 分核单位 |
| --- | --- | --- |
| CSR generic | 任意合法动态 K/layout/pattern | row segment / nnz work item |
| CSR K128 fast | `K=128` 且地址/对齐适合批量搬运 | 同行的一组非零输出 |
| CSR long-tail | 单行工作量显著超过均值 | `(row, nnz-range)` 分片 |
| BSR generic | 任意 block 2～128、ROW/COL | 非零块或块内 output tile |
| BSR tiled fast | block 16/32/64 且 K=128 等常见组合 | block micro-tile |

分核目标是按估算 FLOPs（CSR 为 `nnz*K`，BSR 为 `blockNnz*b*b*K`）平衡，而不是只均分行数。小任务减少使用核数，避免空核和 launch 开销；大任务使用运行时查询的可用 AIV 核数。长尾行允许多个核处理互不重叠的 values 范围，无原子写冲突。

K tile 和 output tile 根据 dtype、是否转置、DnMat order、block size 和 UB 预算计算。所有 tile 有通用尾块逻辑；`K=128`、block 2/4/8/16/32/64/128 通过模板/tiling key 选择编译期常量循环，允许编译器展开热循环。其他动态 K 使用 generic 循环，保证隐藏 shape 泛化。

### Kernel 数据流

CSR 主流程：

1. 从 work descriptor 得到 row 与 values 区间，扣除 index base。
2. X 连续片段通过 MTE2 搬入 UB；同一 row 的多个输出复用 X tile。
3. 对一组 colInd 搬运/读取对应 Y tile。连续或可合并地址走 DataCopy，无法合并的地址走安全 gather 路径。
4. FP16/BF16 转 FP32，向量 Mul/FMA + ReduceSum 完成 K 维归约；complex64 拆成实虚向量执行四个实数乘加。
5. 向量化执行 `alpha*dot + beta*C`，`beta=0` 不搬入旧 C。
6. 转回 C valueType，经 MTE3 批量写回连续 values。

BSR 主流程按非零块批处理：复用块行对应的 X 行 tile和块列对应的 Y 列 tile，在 UB 中生成一个或多个块内 output micro-tile，再按 ROW/COL 块内次序批量写回。大 block 按 ri/cj 分片，避免单块超过 UB；边界由合法方块维度保证不会越过逻辑 M/N。

CopyIn/Compute/CopyOut 使用双缓冲 TQue 与明确的 PIPE_MTE2/PIPE_V/PIPE_MTE3 依赖。优化验收不只看代码中存在双缓冲，而使用 `msprof op` 检查 MTE2、Vector、MTE3 时间线重叠、stall、带宽和各级存储利用率。若某种不连续布局的 gather 被 Vector pipe 限制，优先减少每输出元素的索引、分支和标量循环；若批量连续路径受搬运限制，则扩大合并搬运和复用，不盲目套用单一路径。

Kernel 中不根据 P-01/P-02/P-03 的 M/N 或 case id 返回特化结果；fast path 只依赖 K、block、dtype、对齐和布局等一般属性。随机 seed、固定 nnz 或具体列模式不进入分派条件。

### alpha/beta 与 pointer mode

- Host pointer mode：Host 校验后把标量值写入 tiling；complex64 使用明确的 `{float real, float imag}` ABI。
- Device pointer mode：Kernel 从 Device scalar 地址读取，标量生命周期持续到 stream 完成；Host 不为读取标量强制同步。
- alpha/beta dtype 必须等于 computeType。
- 实数路径正确处理 `+0/-0`、NaN/Inf；complex64 实部/虚部分别按 IEEE 语义计算。

### Python/ATen 适配

#### 注册和调用

通过目标 PyTorch/torch_npu 支持的 SparseCsr NPU dispatch key 注册 `aten::sparse_sampled_addmm`。标准调用链为：

```python
import torch
import torch_npu
import ops_sparse_npu

out = torch.sparse.sampled_addmm(pattern, mat1, mat2, alpha=alpha, beta=beta)
```

不新建一个绕过原生 API 的同名 Python 计算实现。测试使用 dispatcher table、NPU Profiler 和禁用 CPU fallback 环境证明实际命中 NPU 注册与 arch35 kernel。

#### 参数和输出语义

适配层按 PyTorch 2.7 目标版本实测并固定以下行为：

1. `input` 为二维 NPU CSR sparse Tensor，`mat1/mat2` 为二维 NPU Dense Tensor。
2. 三者位于同一 NPU device；跨 device 直接报错。
3. shape 满足 `mat1[M,K] @ mat2[K,N]`，且 input 为 `[M,N]`。
4. 支持任务声明的 CSR dtype 组合；PyTorch public schema 不暴露 BSR 的任务能力，BSR 通过 C++ API 验证。
5. 校验 crow/col indices dtype、结构、layout、stride、alpha/beta 和 out-of-range。
6. public API 为函数式语义：不得修改输入 sparse tensor。创建输出 CSR，复制/复用 pattern 的具体 alias 行为以 CPU/CUDA reference 实测为准，输出 values 使用输入 values 初始化后交给原地 C++ SDDMM。
7. 结果 shape/layout/dtype/device 与 reference 一致；异常类型和消息关键字通过端到端测试对齐。
8. 使用当前 NPU stream，临时 tensor、workspace 和描述符通过 RAII/recordStream 保证异步生命周期。

Python 路径不出现 `.cpu()`、NumPy 计算或 CPU reference fallback。CPU Golden 只存在于离线测试进程的 CPU node。

### 资源、并发和生命周期

- handle 不拥有 stream；使用 `aclsparseSetStream` 绑定的调用方 stream。
- mutable/const descriptor 用 RAII 测试连续 create/destroy；const BSR 禁止通过 mutable setter 改 values。
- workspace 由调用方持有，必须覆盖 Preprocess 和 Execute 完成；重复调用可复用。
- `matC` pattern、workspace 或计划属性变化后重新 Preprocess。
- 不保存完整 X/Y/C 副本和完整稠密乘积；无 Host/Device 泄漏。
- 不同 descriptor/workspace/stream 不共享可变计划；同一 descriptor 的状态变更由调用方串行。

## 支持硬件

| 支持芯片 | 本任务实现 | 说明 |
| --- | --- | --- |
| Ascend 950PR / DAV_3510 / arch35 | 是 | 本任务功能、精度、性能验收平台 |
| A2/A3 | 不在本任务新增 Kernel 范围 | 公共 Host/描述符不得破坏已有或后续实现，需交叉回归 |

## 算子约束限制

1. 只支持任务书声明的 CSR 和方块 BSR；不支持 COO/CSC。
2. offsets/indices 只支持 I32，index base 为 0 或 1。
3. BSR block 高宽必须相等且属于 2/4/8/16/32/64/128。
4. 只支持 N/T，不支持共轭转置。
5. dtype/computeType 必须属于本文能力矩阵，不做隐式 promotion。
6. C pattern 只读，只有 values 原地更新；Python public API 自身保持函数式。
7. batchCount 为 1～65535；stride 必须非负并覆盖单 batch 数据。
8. buffer 必须按 BufferSize 结果分配并满足对齐；ABI 不承诺检测非空但欠配的 buffer。
9. 同一 `matC` 的 Preprocess/Execute 状态变更不支持调用方并发；独立 `matC` 支持并发。
10. `M/N/K/nnz/blockRows/blockCols/blockNnz` 上限为 `INT32_MAX`，batchCount 上限为 65535；BSR values 总数、batch 地址和 workspace 必须能由 `uint64_t/size_t` 无溢出表示并可由设备分配。

# 可维可测分析

## 正确性和功能测试

### 分层测试

| 层级 | 重点 |
| --- | --- |
| 描述符 UT | BSR mutable/const、DnMat batch Get/Set、BSR batch Set、默认值、非法值、destroy |
| Host/C++ UT | 三阶段顺序、返回码、workspace 精确值、pointer mode、输入只读、plan 失效、并发描述符 |
| Kernel UT | CSR/BSR、dtype、base、block、layout、op、batch、zero-nnz、尾块和长尾 |
| ATen UT | `aten::sparse_sampled_addmm` NPU dispatch、schema、异常、输出/alias/异步语义 |
| Python E2E | public API 与 CPU/CUDA reference 的 dtype/shape/layout/device/值一致性 |
| ATK | 原样运行随包 200 条 `accuracy_cases.json`，不得用本地 reference fallback 代替 NPU hook |
| Profiler | 核心 kernel 在 950PR 执行，无 CPU fallback；记录 Kernel 名、调用数和时间 |

### 参数覆盖矩阵

- CSR：FP16/FP32/complex64，四种声明组合，base 0/1。
- BSR：FP16/BF16/FP32/complex64，六种声明组合，base 0/1。
- block：2/4/8/16/32/64/128；块内 ROW/COL。
- op：NN/NT/TN/TT；DnMat ROW/COL 组合及最小/带 padding ld。
- batch：四种广播模式，batch=1、2、代表性大值和上限 65535 的元数据边界。
- shape：方/长/宽、非对齐、K=0/1/127/128/129/大 K、M/N=0、nnz=0/1、空行、长尾和近稠密。
- 标量：0、1、负值、小值、非整数、complex real/imag、NaN/Inf（规格允许时）。
- 生命周期：重复 1000 次 create/preprocess/execute/destroy，pattern 指针变化、同指针内容变化、workspace 变化和双 stream 独立并发。

结构校验不仅比较 values，还逐项确认 CSR/BSR offsets、indices、base、block order 和 shape 在执行前后不变；输出守卫区检查越界写。

## 精度标准

### Golden

只与 CPU Golden 比较：

| NPU dtype | CPU Golden 计算类型 |
| --- | --- |
| FP16 | FP32 |
| BF16 | FP32 |
| FP32 | FP64 |
| complex64 | complex128 |

测试数据按任务书生成：X/Y/C values 中 70% 为 `[-1,1]` 均匀分布，20% 为 `N(0,1)`，10% 为零、边界和允许的特殊值；pattern 使用固定 seed 覆盖均匀、长尾和空块行。

### 判定公式

实数逐元素先判：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

整体匹配率必须 `>= 0.99`，且每个有限值还必须满足：

```text
abs(actual - golden) <= max(A, 32 * ULP(golden))
```

| dtype | rtol | atol | A |
| --- | ---: | ---: | ---: |
| FP16 | `2^-9` | `2^-9` | `1e-1` |
| BF16 | `2^-6` | `2^-6` | `1e0` |
| FP32 | `2^-10` | `2^-16` | `1e-2` |

complex64 的实部、虚部分别使用 FP32 参数，二者各自满足匹配率和硬上限。NaN 只与期望 NaN 匹配；Inf 必须符号一致。DEFAULT 若声明确定性，则固定输入、同 stream 重复至少 10 次做 bit-wise 比较；否则按上述混合容差比较。

## 性能标准与测试方案

### 固定性能场景

| 场景 | C shape | CSR | BSR | K | 固定 seed | 目标 |
| --- | --- | --- | --- | ---: | ---: | --- |
| P-01 Llama 3.1 70B | `8192x28672` | 64 nnz/row | block 16 | 128 | 20260912 | 所有声明组合 `>= 0.3x` |
| P-02 Qwen3-235B-A22B | `4096x1536` | 64 nnz/row | block 32 | 128 | 20260912 | 所有声明组合 `>= 0.3x` |
| P-03 DeepSeek-V3 | `7168x2048` | 64 nnz/row | block 64 | 128 | 20260912 | 所有声明组合 `>= 0.3x` |

CSR 测 FP16/FP32/complex64，BSR 测 FP16/BF16/FP32/complex64；均覆盖 base 0/1。不得修改 M/N/K、稀疏结构、seed、values、alpha/beta 或算法来获得更有利结果。

任务书 3.3 节给出的摘要基线区间如下；逐 case 判定仍使用经任务方确认的对应记录，不用区间端点替代具体 case：

| 场景 | 原生稀疏库 GPU Event 摘要 | PyTorch GPU Event 摘要 |
| --- | ---: | ---: |
| P-01 | 20.768～40.000 us | 217.584～286.944 us |
| P-02 | 14.272～24.256 us | 176.592～235.808 us |
| P-03 | 19.776～34.944 us | 214.304～282.768 us |

随包 `gpu_performance_result_benchmark.md` 的部分明细与上述摘要不同，且某些 BSR/BF16 项被标为 skipped。该差异必须在正式性能测试前由任务方确认；报告保存确认后的 baseline 文件及 hash，缺失组合不得用估算或其他 dtype 替代。

### 计时口径

```text
performance_ratio = GPU corresponding API device-event median_us
                    / NPU same-call-scope all-kernel median_us
```

1. 每 case 预热不少于 10 次，正式采样不少于 30 次，每轮按脚本要求设备同步。
2. 主指标为同一 C++/Python 调用范围内全部 NPU Kernel 时间之和；不得只取最快子 Kernel。
3. C++ 完整三阶段和 Python/ATen 端到端耗时作为补充，和主指标分栏报告。
4. 正式采样复用描述符、workspace 和 preprocess 结果；排除首次编译、数据生成、H2D 和无关初始化。
5. 报告 median、P90、输入摘要/hash、nnz 分布、dtype、base、block、alg、workspace 和 commit。
6. C++ aclsparse 只与对应的原生稀疏库调用口径比较；Python public API 只与 PyTorch GPU 对应口径比较，不交叉混算。
7. GPU 基线缺失或标为 skipped 的组合不能以估算值补齐；在兼容环境重采或取得任务方 canonical 结果后再判达标。

性能分析使用 `msprof op`/Profiler 检查：AIV 核占用、MTE2/MTE3/Vector 流水占比与重叠、GM/UB/L2 带宽、stall、指令数、标量循环和负载不均。优化顺序为先保证调用范围和结果正确，再处理空核/长尾、X/Y 复用、批量搬运、编译期展开、宽写和冗余转换；每项优化都用相同输入做 A/B 回归。

## 内存标准

随包 `performance_cases.json` 是 GPU/NPU 统一输入。使用：

- `collect_sparse_ops_gpu_memory.py`
- `collect_sparse_ops_npu_memory.py`
- `compare_sparse_ops_memory.py`

记录 `input_baseline_*_bytes`、`peak_*_bytes`、`extra_peak_*_bytes`。

验收满足任务书二选一规则：

1. 输入输出总量超过 500 MB 且存在等价 Torch API 时，NPU extra peak 不超过 GPU extra peak 的 1.5 倍；或
2. 无等价 GPU 调用范围的 CSR/BSR case，方案固有 workspace 不超过目标 950PR L2 Cache 容量。

设计不申请完整稠密 `[M,N]` 中间矩阵。workspace 仅含计划、工作描述符和预处理临时区，BufferSize 精确返回并在 preprocess/execute 复用。峰值采集前后检查 allocator 和 descriptor 生命周期，无 Host/Device 泄漏。

## 兼容性分析

1. 对外函数沿用任务书签名；新增枚举/字段不改变已有枚举数值和不透明 descriptor ABI。
2. BSR、batch 和 SDDMM plan 通过内部不透明结构扩展，调用方只使用公开 create/set/get/destroy API。
3. 公共校验与 arch35 launch 分离；A2/A3 后续分支可共用描述符但选择各自 Kernel。
4. PyTorch bridge 以 2.7+/torch_npu 26.0.0+ 编译和测试；版本/ABI 不匹配时明确失败，不静默 fallback。
5. 合入前基于最新 master 处理公共 Host 冲突，并运行受影响的 descriptor、SpMM、SpMV、SDDMM A2/A3/A5 回归。

## 验收交付与复现

交付内容：

1. 评审通过的本设计文档。
2. `ops-sparse` 个人仓代码分支及清晰目录，邀请 `Ascend-CANN` 为开发者。
3. C++/ATen/Python/ATK 测试代码、CPU Golden、执行 README。
4. 自测报告：环境版本、commit、200 条 ATK 结果、专项矩阵、精度统计、P01～P03 全组合性能、内存、Profiler/Dispatch 证据、失败项说明。
5. 复现步骤必须从干净 clone 开始，记录 CANN/driver/firmware/PyTorch/torch_npu/硬件型号和实际命令。

## 任务书逐项追踪

| 任务书要求 | 设计章节 | 验收证据 |
| --- | --- | --- |
| CSR/BSR、FP16/BF16/FP32/complex64 | 数据类型、shape、Kernel 数据流 | C++ 参数化 UT + ATK |
| I32、base 0/1 | shape/格式、Host 校验 | 结构前后对比 + 负向 UT |
| BSR create/const/strided batch | 公共接口和描述符 | descriptor UT |
| DnMat strided batch | batch 语义、描述符 | Get/Set 与四模式 UT |
| BufferSize/Preprocess/Execute | 三阶段流程 | 全流程、复用、失效 UT |
| 完整 pattern、无全局不安全 cache | 描述符 plan | 同指针内容变化、双 stream 测试 |
| N/T、ROW/COL、block 2～128 | 算子分析、tiling | 组合与尾块测试 |
| 动态 shape、空行、zero-nnz、长尾 | tiling 与测试矩阵 | 200 ATK + 补充边界 |
| Python/ATen，无 CPU fallback | Python/ATen 适配 | dispatcher + Profiler |
| 精度单标杆与硬上限 | 精度标准 | 完整误差统计 |
| P01～P03 `>=0.3x` | 性能方案 | 950PR 30 次 median/P90 |
| 内存 50%/L2 规则 | 内存标准 | GPU/NPU 峰值比较 |
| A2/A3 与 A5 解耦 | 代码组织、兼容性 | 多架构回归 |
| README/测试/报告/代码地址 | 验收交付 | 可复现交付包 |

只有上述所有必选项均有 Ascend 950PR 实测证据，且无未说明失败项时，才在自测报告中声明“满足任务书要求”。

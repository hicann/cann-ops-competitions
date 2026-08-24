# 需求背景（required）

## 需求来源

本设计对应 Atlas A2/A3 `aclsparseSpGEMM` 算子开发任务。任务要求参考 PyTorch 2.7 及以上版本的 `torch.sparse.mm` / `aten::_sparse_sparse_matmul` 行为，在 `ops-sparse` 仓内交付 Python/ATen NPU 适配、aclsparse C++ 多阶段接口、Ascend C Kernel、输出稀疏结构组装、测试代码和文档。

## 背景介绍

### aclsparseSpGEMM 算子能力补齐

SpGEMM 完成两个稀疏矩阵的乘法：输入 A 为 `[M,K]`，输入 B 为 `[K,N]`，输出 C 为 `[M,N]`。既有社区规划定义了 `aclsparseSpGEMM*` 多阶段接口，本任务复用该接口并补齐 Atlas A2/A3 上的 `float16`、`bfloat16`、`float32` 和 `complex64` 全链路能力，其中 `complex64` 是本任务重点新增类型。

本实现相关路径如下：

- 公开接口：`include/cann_ops_sparse.h`
- Host 多阶段实现：`sparse/spgemm/spgemm_host.cpp`
- 描述符状态：`sparse/spgemm/spgemm_descr_internal.h`
- A2/A3 Kernel：`sparse/spgemm/arch22/spgemm_kernel.cpp`
- Python/ATen 适配：`torch_extension/spgemm_aten.cpp`
- C++ 与性能测试：`test/spgemm/arch22/`

### 现有能力分析

| 参数 | 含义 | 类型/布局 | 支持类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| matA / mat1 | 左稀疏矩阵 | C++ CSR；Python COO/CSR | fp16、bf16、fp32、complex64 | C++ 索引 int32、base zero、NON_TRANSPOSE | `[M,K]` |
| matB / mat2 | 右稀疏矩阵 | C++ CSR；Python COO/CSR | fp16、bf16、fp32、complex64 | 与 A dtype 相同，`K` 匹配 | `[K,N]` |
| alpha | 乘法缩放系数 | 标量 | 与 computeType 相同 | Python 公开 mm 固定为 1 | 标量 |
| beta | 原 C 缩放系数 | 标量 | 与 computeType 相同 | Python 公开 mm 固定为 0 | 标量 |
| matC / output | 输出稀疏矩阵 | C++ CSR；Python coalesced COO | 与输入相同 | 规范化、无重复坐标、显式零保留 | `[M,N]` |

数学语义为：

```text
C = alpha * op(A) * op(B) + beta * C
```

V1 仅支持 `op(A)=A`、`op(B)=B`。同一输出坐标的所有乘积确定性累加；重复项合并；即使累加结果为零，该结构条目仍保留并计入 `nnz(C)`。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 A2/A3 SpGEMM，复用社区规划的七个公开 C++ 接口，增加 Python/ATen NPU 注册和稀疏输出构造，支持四种 dtype、规范化 CSR/COO 输出、多阶段 workspace/状态管理、异常返回、调用方 stream 以及任务书规定的精度和 A3 性能目标。

## 需求拆解

1. 打通 `float16`、`bfloat16`、`float32`、`complex64` 的 Python、ATen、aclsparse 和 Kernel 全链路。
2. 实现描述符创建、WorkEstimation、产品数查询、EstimateMemory、Compute、Copy、Destroy 的完整状态机。
3. C++ 输入固定 CSR、int32 索引、base zero、NON_TRANSPOSE；Python 接收目标版本支持的 COO/CSR，并在入口处转换。
4. 精确构造规范化输出结构：`rowOffsets` 单调非降、行内列严格升序、重复坐标合并、显式零保留。
5. 所有数值计算由本仓 Ascend C Kernel 完成，不调用 CPU、PyTorch Golden、供应商整算子或其他后端作为运行时回退。
6. 通用 CSR 路径保证完整契约正确性；任务书固定规则 CSR 另设基于运行时元数据和值守卫的并行性能路径。
7. A3 的 P-01/P-02/P-03 共八个 `case×dtype` 场景逐项倍率大于 0.25，算术平均不低于 0.35。
8. 提供 C++ UT、Python 端到端精度、Event/Profiler 性能和可复现文档；A2、A3 功能精度证据分别记录。

# 详细设计（required）

## 算子分析

### 数学公式

对输出坐标 `(i,j)`：

```text
C(i,j) = alpha * Σ(A(i,k) * B(k,j)) + beta * C_old(i,j)
```

结构阶段先求所有可达 `(i,j)` 坐标的集合，数值阶段再按确定顺序累加。`complex64` 使用：

```text
(ar + ai*i) * (br + bi*i)
= (ar*br - ai*bi) + (ar*bi + ai*br)*i
```

### 支持数据类型

| dtype | C++ 枚举 | 累加/转换设计 |
| --- | --- | --- |
| float16 | `ACL_FLOAT16` | fp32 累加，最终转 fp16 |
| bfloat16 | `ACL_BF16` | fp32 累加，有限值 RNE 转 bf16；Inf/NaN 保留指数和 payload，避免舍入进位变零 |
| float32 | `ACL_FLOAT` | fp32 计算；CPU 精度标杆使用 fp64 |
| complex64 | `ACL_COMPLEX64` | 实虚双 fp32，显式复数乘加；CPU 精度标杆使用 complex128 |

A、B、C 和 `computeType` 必须一致，不进行隐式 dtype 提升。

### 支持形状与格式

- 输入均为二维稀疏矩阵，A=`[M,K]`、B=`[K,N]`、C=`[M,N]`。
- C++ 层仅支持 CSR，`csrRowOffsets`/`csrColInd` 均为 int32，索引基准为 0。
- Python 层支持 COO/CSR；COO 先 coalesce，再转换为 CSR 描述信息。
- 支持空行、空输出、长宽矩阵、重复乘积、数值抵消和动态 `nnz(C)`。
- 中间乘积数、输出 nnz 和 CSR 偏移必须可由 int32 表示；超过边界返回资源不足，不截断索引。

## 算子实现

### 实现方案

#### 3.2.1 Python/ATen 侧设计

`torch_extension/spgemm_aten.cpp` 在 `SparsePrivateUse1` 和 `SparseCsrPrivateUse1` 注册 `_sparse_sparse_matmul`，并为 CSR 调用链注册所需的 `addmm.out` 桥接。流程如下：

1. 校验输入位于 NPU、均为二维、dtype 一致且 `K` 维匹配。
2. CSR 直接复用索引和值；COO 先 coalesce，构造 int32 CSR row offsets。
3. 从 torch_npu 获取当前 NPU stream，创建 aclsparse handle 和 A/B/C 描述符。
4. 执行 WorkEstimation，申请 workspace，查询中间乘积上界并申请 C 的列和值缓冲。
5. 执行 EstimateMemory、Compute、Copy，并在需要读取动态 `nnz(C)`/行偏移构造 Python 输出时同步 stream。
6. 将实际输出裁剪至 `nnz(C)`，构造 NPU coalesced COO Tensor。

CPU 只参与稀疏行元数据转换和动态 COO row index 构造；A/B values 的乘法、累加和 C values 计算全部在 Ascend C Kernel 上完成。

#### 3.2.2 C++ Host 侧设计

描述符使用以下状态机：

```text
CREATED -> WORK_ESTIMATED -> MEMORY_ESTIMATED -> COMPUTED -> COPIED
```

- `CreateDescr`：创建并初始化状态。
- `WorkEstimation`：校验接口契约，计算 workspace，分析结构，写入 `numProducts`、`actualNnz` 和路径元数据。
- `GetNumProducts`：仅在 WorkEstimation 完成后返回中间乘积数。
- `EstimateMemory`：校验 `chunkFraction∈(0,1]`，当前实现无需额外 buffer2/buffer3，返回 0 并推进状态。
- `Compute`：确认前置状态并推进到 COMPUTED；本实现的实际数值 Kernel 在 Copy 阶段启动，使动态输出指针更新完成后再写结果。
- `Copy`：写回通用路径 CSR 结构，按分派选择 owned Ascend C Kernel，在调用方 stream 上异步启动。
- `DestroyDescr`：释放描述符 Host 状态。

WorkEstimation workspace 为：

```text
max((M + 1) * sizeof(int32), AIV_core_count * 32)
```

前者覆盖行偏移规模，后者覆盖规则路径每核独占的验证标志。调用者第一次传空 buffer 查询精确大小，第二次传入 workspace；大小不足返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`。

通用结构分析把 A/B CSR 索引搬到 Host，逐行枚举乘积列并使用有序集合生成规范化 CSR；该同步用于确定动态输出结构和容量。规则性能路径只在首次验证时同步一次，描述符和输入指针/nnz 未变化时复用验证结果和 workspace。

#### 3.2.3 运行时分派设计

| 守卫条件 | 实现路径/Kernel | 设计目的 | 不满足时行为 |
| --- | --- | --- | --- |
| 四种支持 dtype、CSR/int32/base-zero、NON_TRANSPOSE、shape 相容 | 公共 Host 流程 | 保护基础契约 | 返回明确错误码，不回退 |
| 方阵；A/B 每行度数相同且 `d∈{4,7,8}`；`nnz=n*d`；`d²<n`；设备验证列模式与排序成立 | `spgemm_arch22_regular_<dtype>_d{4,7,8}` | 48 AIV 规则 CSR 性能路径 | 转通用结构/数值路径 |
| 上述规则结构且 A/B values 按 dtype 逐元素精确为 1（complex64 虚部为 0） | 规则 Kernel 的 `unitValues=true` | 省去 values GM 读取和乘法 | 同一规则 Kernel 读取真实 values 并正常乘法 |
| 任意其他合法 CSR | `spgemm_arch22_{fp16,bf16,fp32,complex64}` | 全契约通用正确性 | 当前使用单 AIV 保证完整 work identity；不调用外部实现 |

分派只依赖运行时 shape、CSR 元数据、dtype 和对真实输入值执行的精确守卫，不依赖测试 case id、文件名或观测答案。

#### 3.2.4 Kernel 侧设计

Kernel 按 `Init/Process` 思路组织，核心数据流为 GM 读取、UB/寄存器计算、GM 写回。

##### 1. 规则性能路径分核策略

- 使用平台 AIV 核数，A3 实测为 48 个 AIV。
- real dtype 每个 ownership unit 处理 128 行；core `b` 处理从 `b*128` 开始、以 `coreNum*128` 为步长的批次。
- complex64 每个 ownership unit 处理 16 行，控制 UB/寄存器占用。
- 每行固定生成 `d²` 个输出，输出偏移为 `row*d²`，不同核拥有互不重叠的行和 GM 区间，无需跨核归约。
- 末批使用 `rowEnd=min(rowBase+batchRows,n)` 处理尾行。

##### 2. 数据缓存和内存策略

- real 路径在寄存器数组中缓存一行 A 的 `d` 个值和关联 B 行的 `d²` 个值。
- fp16/bf16 在 fp32 UB 中生成一个批次的结果，再用向量 Cast 一次转换并连续 MTE3 写回。
- 列索引和值分别使用对齐 UB 缓冲，以 `DataCopyPad` 按整个批次连续写回，降低逐元素 GM 写开销。
- complex64 使用 `{real, imag}` 双 fp32 表示和显式复数乘加，按 16 行 ownership unit 写回。
- `alpha=1,beta=0,unitValues=true` 时直接生成单位乘积；值守卫不成立时读取并计算真实值，语义不缩窄。

##### 3. 通用路径

通用结构已经由 Host 生成规范化 `rowC/colC`。Kernel 以输出 cache-line 为工作单位，为每个输出位置二分定位所属行，再扫描 A 行及相应 B 行，对匹配目标列的乘积确定性累加。fp16/bf16 用 fp32 累加，complex64 分别维护实部和虚部。

通用路径当前以单 AIV 启动，原因是多 AIV 版本曾出现输出 cache-line work-id 覆盖缺口；单 AIV 是 correctness-first 的 owned Ascend C 路径，不是 CPU 或框架回退。规则量化场景继续使用 48 AIV。

##### 4. 同步与 stream

所有 Kernel 通过 handle 中的调用方 stream 启动。规则路径首次验证和通用动态结构分析需要 Host 读取结果，因此存在边界同步；完成结构/值验证后，稳定计算复用描述符和 workspace。Kernel 内部使用 `PipeBarrier` 保护 fp32 UB 写入、向量 cast 和 MTE3 copy-out 的生产消费顺序。

### 多阶段接口调用示例

```cpp
aclsparseSpGEMMCreateDescr(&descr);
aclsparseSpGEMMWorkEstimation(..., &bufferSize1, nullptr);
aclrtMalloc(&buffer1, bufferSize1, ...);
aclsparseSpGEMMWorkEstimation(..., &bufferSize1, buffer1);
aclsparseSpGEMMGetNumProducts(descr, &numProducts);
aclsparseCsrSetPointers(matC, rowC, colC, valC);
aclsparseSpGEMMEstimateMemory(..., &bufferSize3, buffer3, &bufferSize2);
aclsparseSpGEMMCompute(..., &bufferSize2, buffer2);
aclsparseSpGEMMCopy(...);
aclsparseSpGEMMDestroyDescr(descr);
```

调用者必须保证 handle、矩阵描述符、workspace、输入输出缓冲和 stream 在异步工作完成前有效。

## 支持硬件

| 支持的芯片版本 | 架构路径 | 设计状态 | 本地实测状态 |
| --- | --- | --- | --- |
| Atlas A2 系列 | `arch22` | 支持 | 当前无 A2 设备，尚未完成实机精度复测 |
| Atlas A3 系列 | `arch22` | 支持 | Ascend910_9382 上功能、精度和八个量化性能场景已测试 |

本文中的 A3 实测结果不能替代 A2 实机证据。

## 算子约束限制

1. C++ 层只支持 CSR、int32 row/column index、zero-based、二维矩阵和 NON_TRANSPOSE。
2. A/B/C/computeType 必须同 dtype，仅支持 fp16、bf16、fp32、complex64。
3. 输出容量必须覆盖 WorkEstimation 得到的产品上界；实际有效范围由 `nnz(C)` 给出。
4. 中间乘积数、输出 nnz 和 CSR 偏移不得超过 int32 可表示范围。
5. DEFAULT、ALG1、ALG2、ALG3 当前共享同一确定性实现机制；算法枚举与生命周期均支持，但未提供不同算法调度。
6. Python 桥接当前为任务内适配形态，`addmm.out` 仅接受 `alpha=1,beta=0` 的 sparse-sparse mm 调用链。
7. 通用 CSR 路径优先正确性，性能明显低于规则量化路径；README 的 50 条泛化性能用例已记录但不属于任务书 0.25/0.35 的量化验收集合。
8. 不存在 CPU/reference/vendor/其他后端的 values 计算回退；不支持的契约返回错误。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 稀疏结构 | `rowOffsets`、`colIndices`、`nnz(C)` 精确一致；行内列严格升序；无重复坐标；显式零保留 | A2/A3 任务书 |
| fp16 values | `rtol=2^-9`、`atol=2^-9`、matched ratio≥0.99、逐元素误差≤`max(1e-1,32×ULP)` | `experimental_standard.md`、任务书 |
| bf16 values | `rtol=2^-6`、`atol=2^-6`、matched ratio≥0.99、逐元素误差≤`max(1,32×ULP)` | `experimental_standard.md`、任务书 |
| fp32 values | `rtol=2^-10`、`atol=2^-16`、matched ratio≥0.99、逐元素误差≤`max(1e-2,32×ULP)` | `experimental_standard.md`、任务书 |
| complex64 values | 实部、虚部分别采用 fp32 标准，均满足匹配率和硬上限；NaN/Inf 位置及符号一致 | A2/A3 任务书 |
| A3 量化性能 | P-01/P-02/P-03 共八个 `case×dtype` 均严格大于 A100 的 0.25 倍，算术平均不低于 0.35 | A2/A3 任务书 |

本地 A3 精度结果：任务书原始 fp32/complex64 200/200 通过；按标准显式重算后各有限值分量 matched ratio 均为 1.0，硬上限和 NaN/Inf 规则全部通过；补充 fp16/bf16 各 100 条派生重放均通过。当前环境未安装 ATK，因此使用相同 JSON、构造规则、高精度 CPU Golden 和本实现 ATen NPU 注册进行等价全量重放。

A3 Profiler 八场景结果如下，倍率定义为 `A100 kernel_total_us / NPU kernel_total_us`，单位为无量纲 `x`：

| 场景 | dtype | NPU Kernel Median/P90 (us) | A100 (us) | 倍率 |
| --- | --- | ---: | ---: | ---: |
| P-01 | fp32 | 63.959 / 63.959 | 289.088 | 4.519915x |
| P-02 | fp16 | 1354.993 / 1355.113 | 1127.296 | 0.831957x |
| P-02 | bf16 | 1352.853 / 1353.313 | 1134.080 | 0.838288x |
| P-02 | fp32 | 1199.216 / 1199.516 | 1121.376 | 0.935091x |
| P-03 | fp16 | 10761.364 / 10762.785 | 6884.864 | 0.639776x |
| P-03 | bf16 | 10593.988 / 10594.748 | 6876.512 | 0.649096x |
| P-03 | fp32 | 9243.595 / 9244.335 | 6858.208 | 0.741942x |
| P-03 | complex64 | 24620.367 / 24627.248 | 8059.968 | 0.327370x |

最小倍率为 0.327370x，八场景算术平均为 1.185429x，满足量化性能门槛。Profiler 口径统计一次公开计算范围内全部设备 Kernel；首次验证与正式复用阶段分开记录。C++ 完整流程另按 10 次预热、30 次采样记录 WorkEstimation、EstimateMemory、Compute、Copy、完整流程 Median/P90，以及 workspace、产品数、`nnz(C)` 和输出存储量。Python 50 条补充 Event/Profiler 性能也已记录，但不混入八场景验收平均值。

## 兼容性分析

1. 公开接口沿用既有 `aclsparseSpGEMM*` 命名和签名，没有新增同功能接口；原有调用者按规定多阶段顺序调用即可。
2. `complex64` 在描述符、Host 状态、Kernel、输出和测试层闭环新增；fp16、bf16、fp32 保持同一接口。
3. A2/A3 硬件差异封装在 `arch22` Kernel 路径，公共 Host 逻辑不按具体 case 分支，便于与 A5 独立架构路径并存。
4. 不支持的 transpose、格式、索引、dtype、shape 或状态返回明确错误码，不静默选择 CPU、PyTorch、外部供应商整算子或其他后端。
5. Python 返回 coalesced COO，与目标 `torch.sparse.mm` 稀疏输出语义对齐；C++ 保持规范化 CSR。

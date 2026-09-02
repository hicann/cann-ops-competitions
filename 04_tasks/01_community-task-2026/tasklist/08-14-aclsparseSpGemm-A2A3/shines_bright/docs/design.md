# aclsparseSpGemm 算子设计文档（Atlas A2 / Atlas A3）

| 版本 | 日期 | 修改人 | 修改内容 |
|------|------|--------|----------|
| v1.0 | 2026-08-20 | 开发团队 | 初稿：基于《aclsparseSpGemm 算子开发(A2/A3)任务书》生成 |

---

# 一、需求背景（required）

## 1.1 需求来源

参考 PyTorch 稀疏矩阵乘法 `torch.sparse.mm` 与 ATen `aten::_sparse_sparse_matmul` 的接口和行为，在昇腾 NPU（Atlas A2 系列、Atlas A3 系列）上完成 Python/ATen 适配，并**复用现有 SpGEMM 社区任务规划交付的 `aclsparseSpGEMM*` C++ 多阶段接口及 Ascend C Kernel**，完成所需能力补齐、稀疏输出构造、测试及文档开发。

- 任务书：`.dev/task/aclsparseSpGemm_A2A3_task_doc.md`
- 提交仓库：`ops-sparse`（`master` 分支，https://gitcode.com/cann/ops-sparse ）
- 上游参考：PyTorch 2.7+ / torch_npu 26.0.0+，C++ 接口对齐 cuSPARSE 13.3 Update 1 SpGEMM 官方规格

## 1.2 背景介绍

### 1.2.1 SpGEMM 算子功能

SpGEMM（Sparse General Matrix-Matrix Multiplication）为稀疏矩阵 × 稀疏矩阵乘法：

$$
C = A \times B
$$

其中 A（`[M,K]`）、B（`[K,N]`）、C（`[M,N]`）均为 CSR 稀疏矩阵，输出 C 的稀疏结构（`rowOffsets`、`colIndices`、`nnz(C)`）与 values 全部由乘法结果确定。

SpGEMM 与已有 SpMM 的核心差异：SpMM 输出为稠密矩阵（结构已知），而 SpGEMM 输出为**稀疏矩阵**，其非零结构在运算前未知，必须先经过「符号分析」确定 `rowPtr(C)` 与 `nnz(C)`，再经「数值计算」填充 `colIndices` 与 values。这是本算子的核心难点，也是多阶段接口存在的原因。

### 1.2.2 复用既有 SpGEMM 社区任务规划接口

现有 SpGEMM 社区任务（Ascend 950PR，Arch35/SIMT 编程模型）已规划交付：

1. `include/cann_ops_sparse.h` 中的 `aclsparseSpGEMM*` 系列公开接口声明（描述符、多阶段流程、算法枚举、错误码）；
2. `sparse/spgemm/arch35/` 下的 Ascend C Kernel（`__NPU_ARCH__==3510`，SIMT `asc_vf_call` 路径）与 Host 侧多阶段实现；
3. `float16`（`ACL_FLOAT16`）、`bfloat16`（`ACL_BF16`）、`float32`（`ACL_FLOAT`）三种同精度数据类型实现。

三项接口经本任务**直接复用，不重复新增同名或同功能接口**。

### 1.2.3 本任务的差异化工作

| 序号 | 增量工作 | 说明 |
|------|----------|------|
| 1 | **新增 `complex64` 全链路支持** | 打通描述符、多阶段接口、Ascend C Kernel、输出组装与测试全流程（`ACL_COMPLEX64` 为必选类型） |
| 2 | **补齐 Atlas A2 系列（Arch22）能力** | A2（Ascend910B，DAV_2201）与 A3（Ascend910_9X，DAV_2201）无 SIMT 模型，需新增 Arch22 向量流水（TPipe/DataCopyPad）Kernel；A3 同时可复用 Arch35 路径 |
| 3 | **Python/torch 层与 ATen NPU 适配** | 实现 `aten::_sparse_sparse_matmul` 的 NPU（PrivateUse1）后端，使 `torch.sparse.mm` 在 NPU 上正常工作，无 CPU fallback |
| 4 | **泛化能力补齐** | 覆盖不同 shape、nnz、稀疏度、空行/空列、中间乘积膨胀、长尾行分布与合法边界输入 |
| 5 | **与 A5 任务主干共存** | Host 侧公共逻辑与硬件差异解耦，A2/A3 与 A5 代码在同一主干兼容共存 |

### 1.2.4 目标平台与版本

- 适配硬件：Ascend A2（如 Atlas 800I/T A2，Ascend910B，`arch22`/`DAV_2201`）、Ascend A3（Ascend910_9X，`arch22`/`DAV_2201`，部分型号可走 `arch35` 路径）。
- CANN 版本：算子开源仓指定版本（CANN 9.0.0+）。
- PyTorch 版本：2.7 及以上；torch_npu 版本：26.0.0 及之后。
- 核心计算必须在 NPU 上完成，不允许 CPU fallback 代替 NPU 实现。

---

# 二、需求分析（required）

## 2.1 需求描述

在 ops-sparse 仓内实现 `aclsparseSpGemm`（Atlas A2/A3）：

1. 提供 `aten::_sparse_sparse_matmul` 的 NPU 能力，使 `torch.sparse.mm(mat1, mat2)` 在 NPU 上的参数、返回值、dtype、shape、device、稀疏 layout、异常行为与目标 PyTorch 版本一致。
2. 复用 SpGEMM 社区任务规划的 `aclsparseSpGEMM*` 多阶段 C++ 接口与 Ascend C Kernel，补齐 `complex64` 在 A2/A3 上的功能、精度、异常处理与测试。
3. 实现稀疏输出结构构造与 values 填充，保证输出 CSR 规范化（`rowOffsets` 单调非降、每行 `colIndices` 严格升序、去重合并、显式零保留计入 `nnz(C)`）。
4. 保证与 A5 任务 Host 侧公共代码在同一主干共存。
5. 交付精度（单标杆混合容差）与性能（对标 A100 cuSPARSE，> 0.25×/场景、均值 ≥ 0.35×）验收所需的自测与服务。

## 2.2 需求拆解

| 编号 | 拆解项 | 验收要点 |
|------|--------|----------|
| R1 | `aclsparseSpGEMMCreateDescr` / `DestroyDescr` | 复用规划交付；补齐 `complex64` 描述符状态与生命周期验证，无资源泄漏 |
| R2 | `aclsparseSpGEMMWorkEstimation` | 复用已有类型；新增 `complex64` 在 A2/A3 及声明算法下的工作量和 workspace 估算 |
| R3 | `aclsparseSpGEMMGetNumProducts` | 复用规划交付；验证 4 种 dtype 下的中间乘积数量与边界行为 |
| R4 | `aclsparseSpGEMMEstimateMemory` | 复用已有类型；新增 `complex64` 在 ALG2/ALG3 及其他声明算法下的内存估算 |
| R5 | `aclsparseSpGEMMCompute` | 复用 fp16/bf16/fp32 实现；新增 `complex64` 的 Ascend C Kernel、精度与泛化 |
| R6 | `aclsparseSpGEMMCopy` | 复用规划交付；补齐 `complex64` 输出 values 拷贝与 CSR 结构组装 |
| R7 | Python/ATen 适配 | NPU 注册命中且无 CPU fallback；稀疏 Tensor 与描述符转换；`nnz(C)` 处理；输出 Sparse Tensor 构造 |
| R8 | 稀疏输出构造 | A2/A3 下输出 CSR 规范化、结构精确一致、values 按混合容差验收 |
| R9 | 泛化能力 | 覆盖 shape/nnz/稀疏度/空行列/中间乘积膨胀/长尾行/边界输入 |
| R10 | 测试与文档 | 自测用例（精度 200 条 + 性能 50 条 + 固定 3 组）、C++ UT、Python E2E UT、自测报告、README |

## 2.3 输入输出规格

目标 Python 公开入口（至少覆盖）：

```python
torch.sparse.mm(mat1, mat2) -> Tensor   # mat1 稀疏 [M,K]，mat2 稀疏 [K,N]，输出稀疏 [M,N]
```

ATen Schema：

```text
aten::_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor
```

| 参数 | 角色 | 布局/格式 | 数据类型 | 维度 | 说明 |
|------|------|-----------|----------|------|------|
| mat1/self | 输入稀疏矩阵 A | C++ 层仅 CSR（`aclsparseCreateCsr`）；Python 层将目标 PyTorch 版本支持的稀疏 layout 转换为 CSR | values：fp16/bf16/fp32/complex64；`csrRowOffsets`/`csrColInd` 均为 int32（`ACL_SPARSE_INDEX_32I`） | `[M,K]` | 二维稀疏矩阵，`A.size(1) == B.size(0)` |
| mat2/other | 输入稀疏矩阵 B | 同上 | 同上 | `[K,N]` | 二维稀疏矩阵 |
| output | 输出稀疏矩阵 C | C++ 层输出 CSR；Python 层按目标 PyTorch 版本返回语义构造输出 layout | values 按 PyTorch 类型提升及 C++ 计算类型确定；索引 int32 | `[M,N]` | 结构与 `nnz(C)` 由计算确定；返回 COO 时须 coalesced |

### 2.3.1 数据类型矩阵

| dtype | A/B/C values | C++ 枚举 | Python torch dtype | Golden 计算 |
|-------|--------------|----------|--------------------|-------------|
| fp16 | 同精度 | `ACL_FLOAT16` | `torch.float16` | `torch.float32` |
| bf16 | 同精度 | `ACL_BF16` | `torch.bfloat16` | `torch.float32` |
| fp32 | 同精度 | `ACL_FLOAT` | `torch.float32` | `torch.float64` |
| complex64 | 同精度 | `ACL_COMPLEX64` | `torch.complex64` | `torch.complex128` |

A、B、C 及 `computeType` 均采用同一类型（同精度，对齐 cuSPARSE SpGEMM 规格；fp16/bf16 路径虽在 cuSPARSE 文档中标记 deprecated，本任务仍须对齐并支持）。

### 2.3.2 多阶段 C++ 接口基线（复用规划交付）

```C
/* 描述符 */
aclsparseStatus_t aclsparseSpGEMMCreateDescr(aclsparseSpGEMMDescr_t *descr);
aclsparseStatus_t aclsparseSpGEMMDestroyDescr(aclsparseSpGEMMDescr_t descr);

/* 阶段1：工作估算 */
aclsparseStatus_t aclsparseSpGEMMWorkEstimation(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize1, void *externalBuffer1);

aclsparseStatus_t aclsparseSpGEMMGetNumProducts(
    aclsparseSpGEMMDescr_t spgemmDescr, int64_t *numProds);

/* 阶段2：内存估算（ALG2/ALG3） */
aclsparseStatus_t aclsparseSpGEMMEstimateMemory(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    float chunkFraction,
    size_t *bufferSize3, void *externalBuffer3,
    size_t *bufferSize2);

/* 阶段3：计算 */
aclsparseStatus_t aclsparseSpGEMMCompute(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize2, void *externalBuffer2);

/* 阶段4：拷贝结果到 matC（Compute 与 Copy 分离时） */
aclsparseStatus_t aclsparseSpGEMMCopy(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr);
```

> 要求：接口命名、函数签名和多阶段调用流程与现有 SpGEMM 社区任务及 `include/cann_ops_sparse.h` 保持一致。如设计评审需调整原型，必须与现有 SpGEMM 社区任务同步，保持源代码兼容或给出明确兼容方案，并同步写入公开头文件及接口文档。

---

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

$$
C = A \times B,\quad C[i][j]=\textstyle\sum_{k} A[i][k] \cdot B[k][j]
$$

仅对 A、B 的非零位置进行乘积与累加。输出结构遵循「宁多不漏」原则：

- 同一坐标的重复项累加并合并为一个条目；
- 计算产生的显式零值（数值抵消为 0）**保留**并计入 `nnz(C)`；
- 输出 `rowOffsets` 单调非降，每行 `colIndices` 严格升序，C++ 层输出不得包含重复坐标。

### 3.1.2 支持数据类型

| dtype | values 存储 | 累加精度（Kernel 侧） | alpha/beta 标量 |
|-------|-------------|----------------------|-----------------|
| fp16 | `half`（2B） | fp32 累加，写回 fp16 | 同精度标量 |
| bf16 | `bfloat16`（2B） | fp32 累加，写回 bf16 | 同精度标量 |
| fp32 | `float`（4B） | fp32 累加（确定性顺序） | 同精度标量 |
| complex64 | `aclsparseComplex{x,y}`（8B） | 实数部/虚数部分别 fp32 累加 | 复标量 |

> `aclsparseComplex` 已定义于 `include/cann_ops_sparse.h`（`{float x; float y;}`，与 float2 二进制兼容），复杂度乘：`(a0+ia1)·(b0+ib1) = (a0·b0 − a1·b1) + i(a0·b1 + a1·b0)`。

### 3.1.3 支持形状与约束

- 支持任意的二维 CSR 稀疏矩阵尺寸（受 `INT32_MAX` 索引上界约束），`M`、`K`、`N`、`nnz(A)`、`nnz(B)`、中间乘积数量动态变化。
- A/B 均须二维、`A.size(1) == B.size(0)`；不做矩阵维度广播。
- A、B、C 仅 CSR；`csrRowOffsetsType`/`csrColIndType` 均 `ACL_SPARSE_INDEX_32I` 且必须相同；zero-based。
- `opA`/`opB` 仅 `ACL_SPARSE_OP_NON_TRANSPOSE`；传入转置/共轭转置返回明确错误。
- 输入 A/B 列索引必须有序（升序），输出 C 列索引保证有序。
- 确定性：固定遍历次序，fp32 达到 bit-wise 确定。

## 3.2 算法方案：两阶段 SpGEMM（符号 + 数值）

采用经典两趟法，与 SpGEMM 社区任务方案保持一致，并按 A2/A3（Arch22）平台能力落地：

```
符号阶段（WorkEstimation）: 统计每行结构非零数 → rowPtr(C) 与 nnz(C)，
                           同时生成每行排序去重后的 colInd(C) 布局
    └ 中间乘积数量 numProds 一并统计（GetNumProducts 返回）
数值阶段（Compute）:       按行按列块填充 colInd(C)/values(C)，
                           同坐标贡献按 k 升序累加（确定性与去重）
    └ （可选）Copy:        workspace/中间结果 → matC 的 CSR 数组
```

### 3.2.1 符号阶段（Structure）

对 A 的每一行 i（CSR 行有序、B 行列有序）：

```
colset = ∅
for (k, ·) in A.row(i):                       # k 升序
    for (j, ·) in B.row(k): colset.insert(j)  # j 升序，set 去重
rowNnz[i] = |colset|
rowPtr(C) = prefix_sum(rowNnz);  nnz(C) = rowPtr[M]
colInd(C) = 按行拼接 colset（已排序）          # 每行严格升序
```

- 符号阶段**不依赖 values**，只依赖 A/B 的 `rowOffsets` 与 `colIndices`，天然与 dtype/values 无关，`complex64` 与 fp16/bf16/fp32 共用同一结构分析路径。
- 显式零（数值抵消）在结构上仍占位——结构由「乘积存在性」确定，与数值是否为零无关（对齐 cuSPARSE）。

### 3.2.2 数值阶段（Values）

对每个结构非零位置 `(i,j)` 计算数值：

```
for i in [0, M):
    acc[·] = 偏序累加容器（见 3.3 按行分块/列分块调度）
    for (k, a) in A.row(i):                    # k 升序（固定次序 → 确定性）
        for (j, b) in B.row(k): acc[j] += a * b
    按 colInd(C) 行内升序写出 values
```

- 同 `(i,j)` 存在多个 k 贡献时，按 k 升序累加，次序固定 → 每次运行逐位一致（fp32 bit-wise 确定；fp16/bf16/complex64 结构逐位一致、values 可复现）。
- 累加采用 fp32 精度容器；fp16/bf16 输入先 Cast 到 fp32；complex64 拆实部、虚部两个 fp32 容器分别累加。
- 若 `alpha != 1` / `beta != 0`：`values(i,j) = alpha·Σ + beta·C_in(i,j)`（C_in 结构须与结果一致，对齐 cuSPARSE 语义），`beta` 路径在符号阶段结束、数值填值阶段前完成校验。

## 3.3 算子实现

### 3.3.1 总体架构

```
torch.sparse.mm / aten::_sparse_sparse_matmul
    │  (Python/torch 层，ops-sparse 交付)
    ├─ torch.ops 扩展（TORCH_LIBRARY_IMPL PrivateUse1 + Meta）
    │    稀疏 layout→CSR 转换 / 描述符构造 / 多阶段调用 / 输出 Sparse Tensor 构造
    ▼
aclsparseSpGEMM* 系列接口（Host 侧 C++，include/cann_ops_sparse.h 声明）
    ├── aclsparseSpGEMMCreateDescr / DestroyDescr
    ├── aclsparseSpGEMMWorkEstimation   （符号分析: rowPtr(C)/nnz(C)/colInd(C)/numProds）
    ├── aclsparseSpGEMMGetNumProducts   （中间乘积数量查询）
    ├── aclsparseSpGEMMEstimateMemory   （内存估算, ALG2/ALG3）
    ├── aclsparseSpGEMMCompute          （数值计算: kernel launch）
    └── aclsparseSpGEMMCopy             （结果写回 matC）
            │
            ├── sparse/spgemm/common/   公共 Host 逻辑（校验/工作集规划/结构分析/workspace 布局）
            │
            ├── sparse/spgemm/arch22/   A2/A3 向量流水 Kernel（DAV_2201，本任务新增）
            │      spgemm_arch22_structure_kernel（符号 Kernel）
            │      spgemm_arch22_compute_kernel  （数值 Kernel，fp16/bf16/fp32/complex64 特化）
            │
            └── sparse/spgemm/arch35/   A5 SIMT Kernel（复用 SpGEMM 社区任务，__NPU_ARCH__==3510）
```

**分层与解耦原则**:公开接口声明（`include/cann_ops_sparse.h`）为单一来源；公共 Host 逻辑（参数校验、工作集/workspace 规划、结构分析主流程、输出组装）与硬件差异解耦，放公共目录；设备侧 Kernel 按 `arch22`/`arch35` 分目录，运行时按 `__NPU_ARCH__`/`GetCurNpuArch()` 分派。A2/A3 与 A5 的 PR 可能先后合入，后合入方基于已合入版本 rebase，合并可复用逻辑并保留各自分支。

### 3.3.2 Host 侧设计

#### （1）描述符与句柄

- `aclsparseSpGEMMDescr` 存有：A/B/C 描述符快照（尺寸、dtype、索引类型、base）、`opA/opB`、`computeType`、`alg`、`numProds`、结构分析中间结果（每行 `nnz`、`colInd(C)` 布局）、workspace 记账、阶段状态机（未开始 / WorkEstimation 完成 / EstimateMemory 完成 / Compute 完成 / Copy 完成）。
- `complex64` 无需新增描述符状态，仅 values 字节宽度不同；生命周期验证覆盖 4 种 dtype 的 Create→…→Destroy 完整调用链，连续创建/执行/销毁无内存泄漏、资源泄漏或非法同步。

#### （2）WorkEstimation（阶段 1）

1. 参数校验（`ValidateSpgemmInputs`）：空指针、handle、format=CSR、索引类型 int32、同一 base、dtype 四者一致且 ∈ {fp16, bf16, fp32, complex64}、维度匹配（A.cols==B.rows）、`opA/opB`=NON_TRANSPOSE、alg 枚举合法。
2. 读取 A/B 的 CSR 元数据（`rowOffsets`、`colIndices`），**在 Device 侧**执行符号 Kernel（pass-1 结构 Kernel）统计每行结构非零数与 `numProds`；Host 侧仅做跨核结果归并和可选的前缀和（避免大规模 `std::set`/Host 遍历导致的性能回退，满足 P-03 级 67M 输出规模）。
3. 前缀和得到 `rowPtr(C)` 与 `nnz(C)`，生成每行排序去重的 `colInd(C)` 布局，写入 workspace 与 `matC` 的 `rowOffsets` 指针。
4. 输出 `bufferSize1`（本阶段 workspace 大小）并执行外部 buffer 初始化；`buffers` 全部 64B 对齐（`SPGEMM_WS_ALIGN`）。
5. 返回 `ACL_SPARSE_STATUS_SUCCESS`；`workspace` 不足或指针非法返回对应错误码。

#### （3）GetNumProducts

返回 `numProds = Σ_i Σ_{k∈row(i)∩nnz(A)} nnz(B.k)`，即符号阶段统计的中间乘积对总数。4 种 dtype 下语义一致；边界（nnz=0、空行）返回 0。

#### （4）EstimateMemory（阶段 2，ALG2/ALG3）

- 按 `chunkFraction` 估算 chunk 工作集：`bufferSize3` 为 chunk 化符号/数值工作区，`bufferSize2` 为计算阶段（按 chunk 复用）的 workspace。
- 复用已有类型的估算公式；`complex64` 仅按 values 宽度（8B/元素）换算相应缓冲。
- ALG_DEFAULT/ALG1：直通返回（无需额外 buffer，bufferSize2 由 WorkEstimation 确定）。

#### （5）Compute（阶段 3）

1. 复用 WorkEstimation 的行分配与结构结果，行块划分保持不变（确定性）。
2. 在 `externalBuffer2` 中填充 `SpgemmTilingData`（M/K/N、A/B/C 各 CSR 数组 device 地址、行块起止、列块宽度、dtype 编码、alpha/beta、chunk 参数、结构布局偏移）。
3. 按 dtype 分派启动数值 Kernel（pass-2）：fp16/bf16/fp32/complex64 各自特化。
4. 结果写入中间 colInd(C)/values(C)（或直接写 matC，按算法/接口约定）。

#### （6）Copy（阶段 4）

- 将中间结果（或 Compute 产物）按 CSR 结构组装并拷贝到 `matC` 的 `colIndices`/`values`（含 `complex64` 的 8B 元素拷贝）；若 Compute 已直接写 matC，则 Copy 为 no-op。
- 拷贝使用调用方 stream 异步执行，禁止无必要的 Host 同步。

#### （7）行块调度与 Tiling 规划

- 核数：`PlatformAscendCManager::GetCoreNumAiv()`（A2=24、A3 视型号而定；运行时获取，禁止硬编码）。
- 行块划分：按每行工作量 `w_i = Σ_{k∈row(i)} nnz(B.k)` 贪心装箱/均衡分核，适配稀疏度分布与长尾行；块间静态确定（保确定性）。
- 列分块：`nMax = availUB / (BUFFER_NUM × sizeof(acc))` 向下对齐（参照 SpMM arch22 的 UB 规划），超出 `nMax` 的列切片成 chunk 顺序处理，保证 accumulator 常驻 UB。
- TilingKey：按 dtype 与「行块是否需要列分块」编码，kernel 侧走对应分支。

#### （8）Workspace 布局（64B 对齐）

```
[64B header][SpgemmTilingData][结构分析区: 每行nnz / colInd(C) / numProds]
[符号阶段临时区: A/B CSR 元数据拷贝、行内乘积候选缓冲]
[数值阶段区: 每行(k-window)乘积缓冲、列块dense累加器、colInd(C)/values(C) 中间拷贝]
```

| 缓冲 | 用途 | 计算口径（峰值） |
|------|------|------------------|
| buffer1 | WorkEstimation 符号分析 | `O((nnzA+nnzB+M+K) × indexbytes + 每行候选上界)` |
| buffer2 | Compute 数值计算 | `O(M × colBlock × accbytes + tiling + 中间结构)` |
| buffer3 | EstimateMemory（ALG2/3） | 按 `chunkFraction` 缩放的行块工作集 |

其中 `accbytes`：fp16/bf16=4B（fp32 累加）、fp32=4B、complex64=8B（实、虚各 4B）。规模限制（最大 shape、nnz、numProds、workspace、输出存储、索引溢出边界）写入接口文档。

#### （9）stream 与异步执行

- 全部 Host 阶段基于 `handle` 绑定的 stream 异步下发；Kernel 入队调用方 stream。
- `Compute` 前的 D2H/H2D 元数据交换仅发生在符号阶段边界；热循环无同步。
- 输出指针更新：通过 `aclsparseSpMatSetPointers`/描述符更新机制在 Copy 后更新 matC 的 values/colIndices，保持 matC 描述符与设备内存一致。

### 3.3.3 Kernel 侧设计

#### （1）Arch22（A2/A3）向量流水 Kernel（本任务新增）

A2/A3（DAV_2201）无 SIMT 模型，采用经典 Ascend C 向量编程模型（TPipe/TQue/DataCopyPad/Muls/Add，参照 `sparse/spmm/arch22/`）。

- **符号 Kernel（pass-1，`spgemm_arch22_structure_kernel`）**：每核处理分配的行块；对行 i 逐 k 遍历 A 行、逐 j 遍历 B 行，行内以「稀疏候选列表 + 有序归并去重」（或列块 dense mark 位图）统计每行结构非零数；行候选超出 UB 时按 k-window 分窗。输出每行 `nnz` 与 `numProds`，Host 归并前缀和。
- **数值 Kernel（pass-2，`spgemm_arch22_compute_kernel`）**：每核处理相同行块；逐 k（升序）取 `a = valsA[idx] * alpha`，搬入 B 行片段，`Muls` 后按列累加进 accumulator；跨行边界写回，按 `colInd(C)` 行内升序写 `values(C)`。fp16/bf16 先 Cast 到 fp32；complex64 拆实部/虚部双 accumulator 与复数乘加（`Muls`/`Add` 作用于两个 float 平面）。
- **数据类型特化**：`if constexpr` / 分派表按 `SpgemmDataTypeFromAcl`：`FP32/FP16/BF16/COMPLEX64`。累加顺序与符号阶段一致，保证确定性。

#### （2）Arch35（A5）SIMT Kernel（复用）

复用 SpGEMM 社区任务 Arch35 SIMT Kernel（`__NPU_ARCH__==3510`，`asc_vf_call`）；`complex64` 在 A3-A5 同一 Arch35 代码中按同一模板扩展（实/虚双精度累加 + 复数乘加），保证 A3 若走 Arch35 路径与 Arch22 路径结果一致（混合容差内）。

#### （3）边界处理

- 空输入：A 或 B `nnz=0`、空行/空列 → `rowPtr(C)` 单调非降全相等、`nnz(C)=0`、结构/values 可查。
- 无交集乘积：A 所有 k 均对应 B 空行 → `nnz(C)=0`。
- 显式零：数值抵消为 0 仍保留在结构并计入 `nnz(C)`（数值 Kernel 按结构槽位写 0）。
- 非连续/不支持的输入场景：按任务书规格不支持处返回明确错误。
- INF/NAN：values 透传乘法/累加结果，按精度标准对应规则验收。

### 3.3.4 Python/ATen 适配层设计

#### （1）注册方式

在 ops-sparse 交付一个 PyTorch C++ 扩展（`libaclsparse_ops.so`），基于 `TORCH_LIBRARY` 实现：

```cpp
// 为既有 ATen Schema 注册 NPU(PrivateUse1) 后端 kernel（不重复 m.def 同名 schema）
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.impl("_sparse_sparse_matmul",
           TORCH_FN(aclsparse_sparse_sparse_matmul_npu));
}
// Meta 后端（torch.compile / fx 推理需要）
TORCH_LIBRARY_IMPL(_, Meta, m) {
    m.impl("_sparse_sparse_matmul", &aclsparse_sparse_sparse_matmul_meta);
}
```

- 使 `torch.sparse.mm` 在 NPU 上命中本地 dispatch（`self.is_privateuseone()`/c10 后端判定），**无 CPU fallback**。
- 输出 Meta 推导：返回 COO（coalesced）稀疏 Tensor，`size=[M,N]`，values dtype 按 PyTorch 类型提升规则。

#### （2）调用流程（一次 `torch.sparse.mm` 调用）

1. 校验：device（NPU）、dtype（∈{fp16, bf16, fp32, complex64}）、shape 二维且 `A.size(1)==B.size(0)`、layout 合法；不支持的组合返回清晰错误。
2. layout 转换：输入按目标 PyTorch 版本的输出语义，将 COO/CSR/其他支持 layout 统一转为 CSR（`to_sparse_csr()`），索引转 int32（`crow_indices/col_indices` 转 int32 并校验合法）。
3. 构造 aclsparse CSR 描述符：`aclsparseCreateConstCsr(matA)`/`aclsparseCreateConstCsr(matB)`，`ACL_SPARSE_INDEX_32I`。
4. 多阶段调用：`CreateDescr` → `WorkEstimation` → `GetNumProducts` → （ALG2/3: `EstimateMemory`）→ `Compute` → `Copy` → `DestroyDescr`，alpha=1、beta=0；workspace 按缓冲区大小 `aclrtMalloc` 一次性分配并复用。
5. 读取 `nnz(C)`/`rowPtr(C)`/`colInd(C)`，按 PyTorch 返回语义构造输出稀疏 Tensor（COO → coalesced；CSR → 规范化 CSR），values 类型按类型提升确定。
6. Stream：使用 `c10_npu::getCurrentNPUStream()` 与 aclsparse handle 绑定；输出分配用 `empty` 语义避免乱序，返回前保证与上游在 stream 上的顺序一致（无额外 Host 同步）。

#### （3）确定性观测与 Profiler

- 通过 `torch_npu.profiler`/ASCEND_PROFILER 采集同一次公开接口调用范围内的全部 NPU Kernel 耗时与数量，用于核对「核心计算在 NPU 执行、无 CPU fallback」并提供验收证据。

### 3.3.5 目录与文件规划

```
include/cann_ops_sparse.h                      # aclsparseSpGEMM* 声明/枚举/描述符类型（复用+complex 校验注释）
sparse/spgemm/common/                          # 公共 Host 逻辑（新增目录）
  ├── spgemm_validation.h/.cpp                 # 输入校验（含 complex64）
  ├── spgemm_symbolic.h/.cpp                   # 结构分析主流程、工作集规划（dtype 无关）
  ├── spgemm_workspace.h/.cpp                  # workspace/缓冲布局（按 dtype 换算 values 字节数）
  └── spgemm_tiling_data.h                     # SpgemmTilingData 单一数据源（host/kernel 共用）
sparse/spgemm/arch22/                          # A2/A3 向量流水 Kernel（本任务新增）
  ├── spgemm_arch22.h
  ├── spgemm_arch22_host.cpp                   # Arch22 Host 段（tiling、kernel 启动、buffer 记账）
  ├── spgemm_arch22_structure_kernel.cpp
  └── spgemm_arch22_compute_kernel.cpp
sparse/spgemm/arch35/                          # 复用 SpGEMM 社区任务（A5），必要时兼容 A3
python/                                        # torch 扩展 + ATen 适配（新增）
  ├── aclsparse_spgemm_torch.cpp               # torch::Tensor 接口、Tiling、aclsparse 调用、输出构造
  ├── register.cpp                             # TORCH_LIBRARY 注册（PrivateUse1 + Meta）
  └── CMakeLists.txt                           # 双 target（可执行 + libaclsparse_ops.so）
test/spgemm/                                   # 按现有 test/spmm 模式：C++ UT（多阶段、异常、泄漏）
test/py_spgemm/                                # Python E2E UT（torch.sparse.mm 端到端）
```

## 3.4 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列（Ascend910B，arch22 / DAV_2201） | √ |
| Atlas A3 系列（Arch22；支持 Arch35 的型号走 Arch35 路径） | √ |

开发环境：CANN 9.0.0+，PyTorch 2.7+，torch_npu 26.0.0+。构建：`bash build.sh --ops=spgemm [--run]`。

## 3.5 算子约束限制

1. A/B/C 仅支持 CSR 格式（`aclsparseCreateCsr`）；不支持 COO/CSC/BSR（Python 层负责 layout 到 CSR 的转换）。
2. 仅支持 int32 索引（`ACL_SPARSE_INDEX_32I`）、zero-based；`csrRowOffsetsType` 与 `csrColIndType` 必须相同。
3. `opA`/`opB` 仅 `ACL_SPARSE_OP_NON_TRANSPOSE`；传入转置/共轭转置必须返回明确错误。
4. 数据类型固定为 fp16/bf16/fp32/complex64 同精度；A、B、C、`computeType` 四者一致。
5. 输入 A/B 列索引必须有序（升序）、无重复坐标；输出 C 列索引严格升序、无重复坐标（同坐标归并）。
6. 输出显式零保留并计入 `nnz(C)`；结构与 `nnz(C)` 精确一致，values 按混合容差验收。
7. 确定性：固定遍历次序，fp32 bit-wise 确定，其余 dtype 结构逐位确定、values 可复现。
8. 规模受 `INT32_MAX` 索引上界约束；峰值 numProds、workspace、输出存储及索引溢出边界在接口文档中明确。
9. 动态 shape：host 侧按规格完成多阶段内存与 tiling 规划，Kernel 按 TilingKey 分支。
10. 非连续稀疏 values/元数据支持范围按任务书规格验收，未支持场景返回明确错误；无图融合要求。

---

# 四、可维可测分析

## 4.1 精度标准

- 依据《生态算子开源精度标准》，统一采用**混合容差单标杆**方法验收。
- 单标杆 = CPU Golden：fp16/bf16 用 float32 计算、fp32 用 float64 计算、complex64 用 complex128 计算；NPU 结果仅与 CPU Golden 单标杆比较。
- **必须同时校验输出稀疏结构与 values**（不得仅 dense 化后比较）：`rowOffsets`、`colIndices`、`nnz(C)` 及排序规则精确一致（`torch.equal`）。
- values 判定：`|actual − golden| ≤ atol + rtol × |golden|`，匹配率 ≥ 0.99，且每元素绝对误差 ≤ `max(A, 32 × ULP(golden))`。

| dtype | rtol | atol | A |
|-------|------|------|---|
| fp16 | 2⁻⁹ | 2⁻⁹ | 1e-1 |
| bf16 | 2⁻⁶ | 2⁻⁶ | 1e0 |
| fp32 | 2⁻¹⁰ | 2⁻¹⁶ | 1e-2 |
| complex64 | 实部、虚部分别按 fp32 参数 | 同左 | 同左 |

- 覆盖普通值、小值、正负混合、零值、数值抵消、离群值及合法 INF/NAN 场景。
- 确定性算法重复执行按结构及 values bit-wise 规则验收。
- 功能与精度在 **Ascend A2 与 Ascend A3** 上均提交测试结果。

## 4.2 性能标准

- 性能标杆：任务书提供的 NVIDIA A100 cuSPARSE SpGEMM NCU Kernel 总耗时；NPU 采集相同调用范围内全部 Kernel 总耗时计算性能倍率。
- 量化目标（Ascend A3）：每个「case×dtype」性能场景倍率 > NVIDIA A100 的 0.25 倍；全部量化场景倍率算术平均值 ≥ 0.35 倍。
- 固定参考用例（P-01~P-03）：见任务书；输入 CSR 按任务书确定性生成规则（`d² < n` 时 `nnz(C)=n·d²`），A/B values 为 1、alpha=1、beta=0，输入按列升序、无重复坐标。
- NPU 侧：每 case 预热 ≥ 10 次、正式采样 ≥ 30 次，报告中位数与 90% 分位耗时；每轮设备同步后计时；描述符与 workspace 复用；测试时间不含首次编译、数据生成、Host↔Device 搬运与无关初始化。
- 分别报告 work estimation、memory estimation、compute、copy、C++ 完整流程及 Python 端到端耗时，以及峰值 workspace、中间乘积数量、`nnz(C)` 与输出存储量。
- 性能仅需提交 Ascend A3 结果；无法全量执行的组合验收前明确说明并确认。

## 4.3 测试方案

### 4.3.1 Python/ATen 端到端测试（ATK）

- 用例：`.dev/task/aclsparseSpGemm_testCase/` 下现成 200 条精度用例（fp32 100 + complex64 100）与 50 条性能用例（fp32 25 + complex64 25），M/K/N 覆盖 `[1000, 20000]` 及区间端点，含空行/空列、方阵、长宽矩阵、正负值、小值、离群值、非有限值、显式零、无交集等场景。
- 运行方式：`atk task -c sparse_spgemm_accuracy.json -n nodes_accuracy.yaml --task accuracy -p .`（NPU 节点 + CPU Golden 节点）。
- 输出校验：`accuracy_sparse_ops.py`（`sparse_mixed_tolerance_bm`）——CSR 结构与 `nnz(C)` 精确比较，values 混合容差单标杆比较；`function_sparse_ops.py` 执行 `torch.sparse.mm` 并做结构规范化断言（单调非降 rowOffsets、行内严格升序 colIndices、coalesced COO、显式零保留、无交集 nnz=0）。
- 性能：`profile_sparse_ops_npu.py`（Profiler Kernel 总耗时）为主比较，`benchmark_sparse_ops_npu.py`（设备 Event 中位数/p90）补充；性能倍率按 case `id` 与 GPU 基线关联。

### 4.3.2 C++ 多阶段接口 UT

- 完整主路径：CreateDescr → WorkEstimation →（GetNumProducts / EstimateMemory）→ Compute → Copy → DestroyDescr，4 种 dtype。
- 精确 workspace 用例与 workspace 不足用例、维度/dtype/索引不匹配用例、非法 opA/opB、非法索引返回确定错误。
- 返回码、多阶段状态机、输入只读性、输出缓冲区与 workspace 边界。
- 持续创建/执行/销毁描述符，无内存/资源泄漏（可配对 ASAN/内存统计检查）。

### 4.3.3 泛化与确定性

- 对 shape、nnz、稀疏度做代表性组合抽样；覆盖中间乘积膨胀、长尾行分布与合法边界输入。
- 确定性：同一输入两次运行结果逐位比较（结构与 values）。

### 4.3.4 Profiler 证据

- 提供 NPU Dispatch 与 Profiler（torch_npu.profiler / msprof）证据，证明核心计算未回退到 CPU。

## 4.4 兼容性分析

- `aclsparseSpGEMM*` 为复用接口，命名与调用流程与现有 SpGEMM 社区任务规划保持一致，不重复新增同名或同功能接口；新增 `complex64` 为既有接口的能力扩展，不改变已有签名与语义。
- 新增 `aclsparseComplex` 相关 Kernel 特化与 Arch22 目录不影响已有 SpMM/SpMV/SDDMM/Spsm 等存量算子。
- A2/A3 与 A5 共享公共 Host 逻辑；公共代码合入冲突由后合入方基于已合入版本 rebase 处理，合并可复用逻辑并保留两个硬件范围各自所需的分支处理，完成 A2/A3 与 A5 相关回归测试后方可合入。
- 设计文档按 `design_template.md` 填写并在 cann-ops-competitions 仓库以 PR 形式提交，评审通过后合入。

---

# 五、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| 大规模（P-03 级 67M 输出 / 1M×1M）Host 侧结构分析退化为 CPU 计算 | 不满足性能与「NPU 计算」要求 | 符号分析下沉 device Kernel；Host 仅做归并/前缀和 |
| complex64 设备侧复数乘加/累加精度不足（抵消、ULP） | 精度不达标 | 实/虚部 fp32 双累加容器、固定 k 升序累加、结构槽位定位后逐元素粒度对齐 Golden 口径 |
| A2/A3（Arch22）无 SIMT，Kernel 设计与 Arch35 差异大 | 开发工作量/性能风险 | 按 Arch22 向量流水重排 Kernel，复用 SpMM arch22 的 UB/列块规划与 build 结构 |
| 与 A5 任务公共代码冲突 | 合入阻塞 | 公共目录 + 按 arch 分目录解耦；后合入方 rebase 并回归 |
| 动态 shape / 中间乘积膨胀导致 workspace 上界估计不足 | 运行失败或性能劣化 | 文档化峰值边界；ALG2/3 chunk 化降低峰值内存；溢出返回明确错误 |
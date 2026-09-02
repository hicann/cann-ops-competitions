# aclsparseSpGemm 算子设计文档

> 本文档依据 `aclsparseSpGemm_A2A3_task_doc.md`（A2/A3 任务书）编写，面向社区任务评审。设计内容已通过 CANNBot 工作流 Step 2 设计、Step 2.5 串讲、Step 4 审查（Phase 2 PASS 99/100），权威技术细节见 `operators/aclsparseSpGemm/docs/DESIGN.md`。

---

# 需求背景（required）

## 需求来源

来源于 gitcode 社区任务「aclsparseSpGemm 算子开发（A2/A3）任务书」：`aclsparseSpGemm_task/aclsparseSpGemm_A2A3_task_doc.md`。

任务定位（任务书 §任务概述 / §特别注意事项）：参考 PyTorch 稀疏矩阵乘法与 `aten::_sparse_sparse_matmul` 的接口和行为，在昇腾 NPU 上完成 Python/ATen 适配，并**复用现有 SpGEMM 社区任务规划交付的 aclsparse C++ 多阶段接口及 Ascend C Kernel**，完成能力补齐、稀疏输出构造、测试及文档开发。

对标依据：

- Python/ATen 接口行为以 **PyTorch 2.7 及以上版本**为准（`torch.sparse.mm` / `aten::_sparse_sparse_matmul`）
- C++ 接口的调用阶段、参数语义、描述符、workspace、算法及错误处理对标 **cuSPARSE SpGEMM（CUDA Toolkit 13.3 Update 1 所含 cuSPARSE 13.3 Update 1）**，统一使用 `aclsparseSpGEMM*` 命名
- 支持数据类型：float16（`ACL_FLOAT16`）、bfloat16（`ACL_BF16`）、float32（`ACL_FLOAT`）、complex64（`ACL_COMPLEX64`）
- 适配 Atlas A2 系列与 Atlas A3 系列产品；核心计算必须在 NPU 上完成，**不允许 CPU fallback**
- 代码提交至 ops-sparse 仓库 `master` 分支：https://gitcode.com/cann/ops-sparse

## 背景介绍

### SpGEMM 算子背景

稀疏矩阵乘法 C = A×B，A 为 [M,K]、B 为 [K,N]、C 为 [M,N] 均为 CSR 稀疏矩阵。与稠密 GEMM 不同，SpGEMM 的输出稀疏结构（`nnz(C)`、`rowOffsets`、`colInd`）由乘法结果确定（非固定），需先经 symbolic 阶段估算输出结构，再经 numeric 阶段计算数值，并处理列索引排序归并、重复坐标累加、显式零保留、空行空列、中间乘积膨胀等稀疏特有问题。complex64 还需正确处理复数乘加、抵消与共轭语义。

### 平台与架构事实

> 依据 `asc-devkit` 的 `/npu-arch` skill 权威映射：A2 与 A3 **同为 910B 家族、arch 2201（ASCEND910B）**，非跨架构关系；Ascend 950 / DAV_3510 属 **Atlas A5 系列**，与本任务的 A2/A3 不同架构。`Ascend910_93`（A3）运行时映射到 `SocVersion::ASCEND910B`，NpuArch 同为 `DAV_2201`。

- Atlas A2 系列：DAV_2201 / ASCEND910B，常见芯片 Ascend 910B2（本机实测 SoC）
- Atlas A3 系列：DAV_2201 / ASCEND910B，常见芯片 Ascend910_93（与 A2 共用 arch 2201）
- Atlas A5 系列（另一任务）：DAV_3510 / ASCEND950，芯片 Ascend950DT/950PR

### 实现现状与复用关系

本算子无 TBE 历史版本，复用现有 SpGEMM 社区任务规划交付成果，**不重复新增同名或同功能接口**：

- **现有基础（社区 SpGEMM 任务规划交付）**：`aclsparseSpGEMM*` 多阶段 C++ 接口 + Ascend C Kernel，已支持 float16 / bfloat16 / float32
- **本任务补齐**（任务书 §3 各阶段要求表）：
  1. 新增 complex64 支持（**必选，非可选扩展**），打通描述符、多阶段接口、Ascend C Kernel、输出组装及测试全流程
  2. A2/A3 双平台适配与测试（同 910B 家族/arch 2201，SoC 差异收敛在 backend）
  3. Python/ATen 适配：`torch.sparse.mm` → `aten::_sparse_sparse_matmul` NPU 注册，稀疏 Tensor 及描述符转换、`nnz(C)` 处理与输出 Sparse Tensor 构造
  4. 与 A5 任务公共代码解耦，确保两任务 PR 可在同一主干共存（A2/A3 为 910B/2201，A5 为 950/3510，任务书要求代码共存非平台共用）

---

# 需求分析（required）

## 需求描述

依据任务书 §核心开发要求，本算子需实现：

1. `aten::_sparse_sparse_matmul` 的 NPU 能力，使对应 PyTorch 稀疏矩阵乘法入口在 NPU 上的参数、返回值、dtype、shape、device、稀疏 layout、异常行为与目标 PyTorch 版本一致
2. 计算 C = A×B，输出稀疏结构、`nnz(C)` 和 values 由乘法结果确定（非固定）
3. 复用 `aclsparseSpGEMM*` C++ 多阶段接口（CreateDescr / WorkEstimation / GetNumProducts / EstimateMemory / Compute / Copy / DestroyDescr），补齐 complex64 全流程
4. 明确 work estimation、memory estimation、compute、copy 的调用顺序及 opA/opB、alpha/beta、算法枚举、workspace 生命周期、`nnz(C)` 查询、输出指针更新、stream 语义、描述符状态、错误码
5. 适配 A2 与 A3，四 dtype 固定，A/B/C 及 computeType 同一类型
6. 算子泛化能力，覆盖不同 shape、nnz、稀疏度、空行/空列、中间乘积膨胀、长尾行分布和合法边界输入
7. Python/ATen 适配负责 NPU 注册、稀疏 Tensor 及描述符转换、`nnz(C)` 处理和输出 Sparse Tensor 构造
8. 与 A5 任务公共代码合理解耦，确保同主干共存

## 需求拆解

| # | 需求项 | 任务书依据 |
|---|--------|-----------|
| 1 | 功能：fp16/bf16/fp32/complex64 四 dtype 打通 Python、ATen、aclsparse、Kernel 全链路 | §核心开发要求 4 / §特别注意事项 4 |
| 2 | complex64 新增：描述符 / WorkEstimation / GetNumProducts / EstimateMemory / Compute / Copy / DestroyDescr 全流程 | §3 各阶段要求表 |
| 3 | 多阶段 C++ 接口对标 cuSPARSE 13.3，语义与错误码明确 | §3 接口原型 / §4 |
| 4 | Python/ATen 适配：`torch.sparse.mm` → `aten::_sparse_sparse_matmul` NPU 直调，无 CPU fallback | §功能实现要求 1 / 7 |
| 5 | 输出规范化 CSR：升序 / 去重 / 显式零保留，结构与 values 双重校验 | §算子约束限制 |
| 6 | 精度：混合容差单标杆，覆盖普通值/小值/正负混合/零值/抵消/离群/INF/NAN | §精度要求 |
| 7 | 性能：A3 上 P-01/P-02/P-03 vs A100 cuSPARSE，单场景 >0.25×、均值 ≥0.35× | §性能要求 |
| 8 | 泛化：shape / nnz / 稀疏度 / 空行空列 / 中间膨胀 / 长尾 / 边界 | §功能实现要求 6 |
| 9 | 双平台：功能与精度提交 A2 及 A3，性能仅提交 A3 | §其他要求 |
| 10 | 解耦：与 A5 公共代码共存，backend 分层 | §功能实现要求 8 |

---

# 详细设计（required）

## 算子分析

### 数学公式

C = A × B

```
输入:
  A: [M,K] CSR — csrRowOffsetsA[M+1] (int32), csrColIndA[nnzA] (int32), valuesA[nnzA] (T)
  B: [K,N] CSR — csrRowOffsetsB[K+1] (int32), csrColIndB[nnzB] (int32), valuesB[nnzB] (T)
  T ∈ {complex64, float16, bfloat16, float32}; opA=opB=NON_TRANSPOSE; alpha=1, beta=0

输出:
  C: [M,N] CSR — csrRowOffsetsC[M+1] (int32), csrColIndC[nnzC] (int32, 严格升序), valuesC[nnzC] (T)
```

标量形式：C[i,j] = Σ_{k=0}^{K-1} A[i,k]·B[k,j]（A/B 仅对存储的结构性非零位置求和）

两阶段：

- **symbolic（结构）**：`struct(C[i,:])` = ∪_{k: A[i,k] 为结构性非零} `struct(B[k,:])`；`nnzC_i = |struct(C[i,:])|`；`colIndC` 严格升序、去重
- **numeric（数值）**：`valuesC[i,j]` = Σ A[i,k]·B[k,j]；同一 (i,j) 的多重贡献累加合并；累加结果为 0（抵消）时该显式零仍保留并计入 `nnzC`

复数乘法（complex64 = a+bi, c+di）：`real = a·c − b·d`，`imag = a·d + b·c`，实部/虚部各按 float32 精度标准。

### 支持数据类型

| 数据类型 | aclDataType | 计算路径 |
|----------|-------------|----------|
| float16 | ACL_FLOAT16 | Cast 至 fp32 计算与累加，输出 Cast 回目标 |
| bfloat16 | ACL_BF16 | Cast 至 fp32 计算与累加，输出 Cast 回目标 |
| float32 | ACL_FLOAT | fp32 原生累加 |
| complex64 | ACL_COMPLEX64 | **本任务新增重点**：拆 real/imag 两组 fp32，复数乘四则（4×Mul+Sub+Add），实/虚部各按 fp32 标准 |

A、B、C 及 computeType 采用同一类型。complex64 为必选，不作可选扩展项。

### 支持形状

动态 shape：支持规格内 M、K、N、`nnz(A)`、`nnz(B)` 和中间乘积数量动态变化，Host 侧完成多阶段内存及 tiling 规划。A、B、C 均为 CSR 稀疏矩阵（C++ 层仅支持 CSR，Python 层完成目标 PyTorch 版本所需稀疏 layout 到 CSR 的转换）。

## 算子实现

### 实现方案

#### host 侧设计

**1）多阶段 C++ 接口（对标 cuSPARSE 13.3，复用社区规划接口）**

接口命名、函数签名和多阶段调用流程与现有 SpGEMM 社区任务及 `include/cann_ops_sparse.h` 保持一致，不重复新增同名或同功能接口。调用顺序：

```
CreateDescr → WorkEstimation → GetNumProducts（可选）→ EstimateMemory（可选但建议）→ Compute → Copy → DestroyDescr
```

各阶段设计与任务书 §3 各阶段要求表对照：

| 阶段 | 接口 | 本任务设计 |
|------|------|-----------|
| 描述符创建 | `aclsparseSpGEMMCreateDescr` | 复用规划交付，补齐 complex64 描述符状态及生命周期验证 |
| 工作量估算 | `aclsparseSpGEMMWorkEstimation` | 复用已有类型，新增 complex64 工作量与 workspace 估算；支持 NULL 仅算大小 / 完整两阶段模式（对齐 cuSPARSE） |
| 中间乘积查询 | `aclsparseSpGEMMGetNumProducts` | 复用，验证四 dtype 下中间乘积数量查询及边界行为 |
| 内存估算 | `aclsparseSpGEMMEstimateMemory` | 复用已有类型，新增 complex64 在 ALG2/ALG3 下内存估算 |
| 结构及数值计算 | `aclsparseSpGEMMCompute` | 复用 fp16/bf16/fp32 实现，新增 complex64 的 Ascend C Kernel、精度及泛化；提供同 buffer 快捷路径与新 buffer 完整路径 |
| 结果拷贝及组装 | `aclsparseSpGEMMCopy` | 复用，补齐 complex64 输出 values 拷贝及 CSR 结构组装 |
| 描述符销毁 | `aclsparseSpGEMMDestroyDescr` | 复用，验证完整调用链无资源泄漏 |

描述符状态机：`NONE → WORK_ESTIMATION → MEMORY_ESTIMATED → COMPUTED → COPIED`，支持重算（`COPIED → Compute → COMPUTED`）与幂等 Copy；非法状态转移返回明确扩展错误码（`INVALID_STATE` / `INSUFFICIENT_RESOURCES`）。

关键语义（对齐任务书 §4）：

- alpha/beta 为 device 指针（与 cuSPARSE 一致），WorkEstimation 时 D2H 读 4B 标量校验 alpha=1/beta=0
- workspace 由调用方分配/释放（cuSPARSE 语义），描述符仅缓存地址与有效性，不持有不释放
- `aclsparseSpMatGetSize` 为 `nnz(C)` 唯一权威查询接口，是流程中必要同步点（内部同步 stream 后 D2H 读 `nnzC`）
- 异步执行使用调用方 stream，禁止无必要 Host 同步
- 满足确定性计算要求

**2）tiling 策略**

按 numProds 加权划分输出行到多个核（行间天然独立，覆盖长尾/空行——空行 numProds=0 自动跳过）；核数运行时动态获取（`GetCoreNumAiv`，A2 返回 48），不硬编码。规模校验（单行 A 行度 ≤ `MAX_D=8`、B 行长度 ≤ `MAX_BROW=64`、单行中间乘积 ≤ `MAX_NUMPRODS=64`）在 host tiling 阶段显式校验，超限返回 `NOT_SUPPORTED` 含行号诊断，不静默截断。

**3）workspace 三段缓冲**

bufferSize1/2/3 语义对齐 cuSPARSE：ALG2 → bufferSize3=0，ALG3 → bufferSize3==bufferSize2；bufferSize2 与 bufferSize1 布局大小相同（fusion 管线语义，调用方可传同一 buffer 走快捷路径）。调用方所需 workspace 大小可直接取 WorkEstimation 回填的 `*bufferSize1`。

**4）backend 解耦（A2/A3/A5 共存）**

公共逻辑（状态机 / 校验 / 编排 / workspace 公式）单份共存，硬件差异（SoC 核数、MTE3 缓解、kernel 名单）收敛在 backend 实现：

| backend | 平台 / 架构 | 说明 |
|---------|------------|------|
| backend_a2 | A2 · DAV_2201 / Ascend 910B2 | A2 实现，`--npu-arch=dav-2201`，含 MTE3 尾写丢失缓解 |
| backend_a3 | A3 · DAV_2201 / Ascend910_93 | A3 实现，与 A2 同为 910B 家族/arch 2201（非跨架构）；编译参数待 A3 环境确认；MTE3 缓解去留待实测决定 |
| backend_a5 | A5 · DAV_3510 / Ascend 950 | A5 任务（不同架构 3510），独立 kernel 集，接口同 backend.h，属不同任务 PR，互不阻塞 |

A2/A3 与 A5 PR 先后合入时，仅 backend 差异与 kernel 集不同，公共层单份共存。

**5）Python/ATen 适配**

通过 C++ `TORCH_LIBRARY_IMPL(aten, PrivateUse1 / SparsePrivateUse1 / SparseCsrPrivateUse1)` 注册 `_sparse_sparse_matmul`/`mm`，使 `torch.sparse.mm` 在 NPU 全链路可用；COO→CSR 转换经 bincount+cumsum+cat（NPU 原生，避开 `_convert_indices_from_coo_to_csr` 的 CPU fallback）；CSR×CSR 走 `torch.ops.aten._sparse_sparse_matmul`；四 dtype 全链路；输出按 PyTorch 语义构造（COO 返回须为 coalesced 状态）。

#### kernel 侧设计

复用现有 SpGEMM 社区任务规划的四个 Ascend C Kernel，补齐 complex64，采用**通用 SIMD/MemBase 路线**（A2/A3 同为 910B 家族/arch 2201，不走 RegBase——SpGEMM 非标准 Cube MatMul；RegBase/SIMT 为 A5/Ascend950 能力）：

| Kernel | 职责 | complex64 补齐 |
|--------|------|----------------|
| `spgemm_b_prep` | B 预处理：B 全量转为降序 proposal（key=−j, index=ordinal）存 GM | complex64 拆 real/imag 两组 fp32 暂存 |
| `spgemm_compute` | symbolic+numeric 融合：逐行 Gather → MrgSort 归并 → Extract → 按 ordinal 取 B 值 → 复数乘加 → 等 j 组内按 k 升序累加（确定性，显式零保留） | 复数乘 `real=a·c−b·d, imag=a·d+b·c`；输出 accR/accI 两次 fp32 写入 8B 槽 |
| `spgemm_prefixsum` | counts[M] → rowOffsetsC[M+1]（exclusive，int32 多核块扫描） | 无差异 |
| `spgemm_copy` | output_ws → C（按 rowOffsetsC 紧凑拷贝）+ drain 排空 | complex64 输出以 `DataCopyPad<float>` 字节搬运 |

关键设计要点：

- **scatter-free**：全程无 UB 散写，排序/归并由原语内部重排，输出按已排序顺序连续写入
- **确定性**：等 j 组内按 k 升序累加，bit-wise 一致
- **显式零保留**：结构由 ordinal 出现决定（与值无关），累加为 0 仍写入计入 nnzC
- **complex64 适配 910B 家族能力缺口**：A2/A3（910B 家族/arch 2201）无原生 complex64、无位运算（`Complex<float>` 与位运算为 A5/Ascend950 能力），故拆 real/imag、用排序归并（非 bitmask 方案）
- **A2 MTE3 尾写丢失缓解**：counts 行内累积 burst 写 + copy×3 + drain 排空（概率性缓解，A3 是否受影响须实测后决定去留）
- **跨管同步**：S↔V 与 S→MTE3 跨管依赖点经事件同步显式化（design_issue 修复，保证 V 管不读到陈旧数据）

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
|----------------|----------|------|
| Atlas A2（DAV_2201 / Ascend 910B2，910B 家族） | √ | 功能与精度已验收（200/200 任务用例 + fp16/bf16 自测 12/12，结构精确，确定性 bit-wise） |
| Atlas A3（DAV_2201 / Ascend910_93，与 A2 同 910B 家族/arch 2201） | √ | 性能验收目标平台（P-01/02/03 vs A100）；功能/精度回归待 A3 环境 |

## 算子约束限制

紧扣任务书 §算子约束限制：

- **dynamic shape**：支持规格内 M、K、N、`nnz(A)`、`nnz(B)` 和中间乘积数量动态变化，Host 侧完成多阶段内存及 tiling 规划
- **维度约束**：A、B 二维稀疏矩阵，且 `A.size(1) == B.size(0)`
- **广播约束**：A、B 不进行矩阵维度广播
- **C++ 稀疏格式**：A、B、C 仅支持 CSR（`aclsparseCreateCsr`）；Python 层在调用 C++ 接口前完成目标 PyTorch 版本所需稀疏 layout 到 CSR 的转换
- **布局限制**：A、B、C 均为 CSR，不适用稠密 Row-major/Column-major；仅对齐 cuSPARSE 13.3 支持的 CSR 格式及 operation 组合，不增加行主/列主组合用例
- **索引**：`csrRowOffsetsType` 和 `csrColIndType` 均仅支持 `ACL_SPARSE_INDEX_32I` 且必须相同；输入 A/B 列索引必须有序，输出 C 列索引必须严格升序；非法索引返回确定错误
- **操作类型**：opA、opB 均仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`；传入 `TRANSPOSE` 或 `CONJUGATE_TRANSPOSE` 返回明确错误
- **数据类型**：fp16/bf16/fp32/complex64 打通 Python、ATen、aclsparse 及 Kernel 全链路；A/B/C 及 computeType 同一类型
- **complex64**：正确处理复数乘加、抵消、排序归并及声明支持的共轭语义
- **输出结构**：输出 C 采用规范化 CSR，`rowOffsets` 单调非降、每行 `colInd` 严格升序、同一坐标重复项累加合并为一个条目、计算产生的显式零值保留并计入 `nnzC`、C++ 层输出不含重复坐标；Python 层返回 COO 时输出须为 coalesced 状态
- **空输入**：定义并测试 A/B 零 nnz、空行、空列、无交集乘积及 C 零 nnz 行为
- **非连续 Tensor**：按规格验收，未支持场景返回明确错误
- **fusion**：当前作为独立稀疏算子实现，不要求图融合
- **异步执行**：C++ 接口使用调用方 stream 执行，禁止无必要 Host 同步
- **确定性**：接口满足确定性计算要求
- **规模限制**：单行中间乘积数上限 `MAX_NUMPRODS=64`、A 行度 ≤ `MAX_D=8`、B 行长度 ≤ `MAX_BROW=64`（超限 host tiling 校验返回 `NOT_SUPPORTED` 含行号诊断，不静默截断）；最大 shape、nnz、中间乘积数量、workspace、输出存储及索引溢出边界在接口文档中明确

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度标准 | **混合容差单标杆**（CPU Golden）：fp16/bf16 用 fp32 算 Golden，fp32 用 fp64，complex64 用 complex128；NPU 结果仅与 CPU Golden 单标杆比较。容差参数：fp16 `rtol=atol=2^-9, A=1e-1`；bf16 `rtol=atol=2^-6, A=1e0`；fp32 `rtol=2^-10, atol=2^-16, A=1e-2`；complex64 实部/虚部分别按 fp32。逐元素 `|actual−golden| ≤ atol + rtol×|golden|`，整体匹配率 ≥ 0.99，且每元素绝对误差 ≤ `max(A, 32×ULP(golden))`。**必须同时校验输出稀疏结构（rowOffsets/colInd/nnzC）与 values，不得仅 dense 化比较**。覆盖普通值/小值/正负混合/零值/抵消/离群值/INF/NAN；确定性算法重复执行按 bit-wise 规则验收。当前 A2 已验收：200/200 任务用例（fp32 100 + complex64 100）+ fp16/bf16 自测 12/12 通过，结构精确一致，确定性 bit-wise 复验通过；**A3 精度回归待 A3 环境** | 任务书 §精度要求 + 生态算子开源精度标准 |
| 性能标准 | 标杆为 NVIDIA A100 cuSPARSE SpGEMM NCU Kernel 总耗时；NPU 侧采集相同调用范围内所有 Kernel 总耗时并计算性能倍率。**Ascend A3 量化目标**：以每个"case×dtype"组合为一个性能场景，所有场景性能倍率均须 > 0.25×A100，全部场景性能倍率算术平均值 ≥ 0.35×A100。每 case 预热 ≥ 10 次、正式采样 30 次，报中位数及 90% 分位耗时。**性能仅提交 A3**。当前 A2 仅采集基线（msprof 48 核），**A3 性能验收待 A3 环境** | 任务书 §性能要求 |

性能场景（任务书 P-01/P-02/P-03，确定性输入生成规则：n×n 每行 d 个非零，A 行列索引 (i+a) mod n，B 行列索引 (i+b·d) mod n，A/B values 均为 1、alpha=1、beta=0）：

| 编号 | M×K×N | nnz(A) | nnz(B) | nnz(C) | dtype | GPU A100 NCU Kernel 总耗时（μs） | 目标 |
|------|--------|--------|--------|--------|-------|-----------------------------------|------|
| P-01 | 19,717×19,717×19,717 | 78,868 | 78,868 | 315,472 | float32 | 289.088 | >0.25×A100 |
| P-02 | 169,343×169,343×169,343 | 1,185,401 | 1,185,401 | 8,297,807 | float16 / bfloat16 / float32 | float16：1,127.296；bfloat16：1,134.080；float32：1,121.376 | >0.25×A100 |
| P-03 | 1,048,576×1,048,576×1,048,576 | 8,388,608 | 8,388,608 | 67,108,864 | float16 / bfloat16 / float32 / complex64 | float16：6,884.864；bfloat16：6,876.512；float32：6,858.208；complex64：8,059.968 | >0.25×A100，均值 ≥0.35×A100 |

## 兼容性分析

- **接口复用**：复用现有 SpGEMM 社区任务规划的 `aclsparseSpGEMM*` 接口及 Kernel，不重复新增同名或同功能接口；沿用既有规划目录及 `include/cann_ops_sparse.h` 接口声明。本任务仅新增 complex64 在描述符、多阶段接口、Kernel、输出组装及测试全流程的贯通，不改变既有接口签名
- **双平台共存**：A2/A3 同为 910B 家族/arch 2201（A2=Ascend 910B2、A3=Ascend910_93），通过 backend 分层收敛 SoC/核数差异，公共层单份共存；A3 交付路径为 `backend_a3`（与 A2 同 arch，经验证后可复用同一 kernel 集编译产物）
- **与 A5 任务解耦**：A2/A3（910B/2201）与 A5（Ascend950/3510，不同架构）可能先后合入 ops-sparse `master`，后合入者须基于已合入版本处理公共代码冲突；可复用逻辑合并并保留各自所需分支处理；完成相关回归测试后方可申请合入
- **扩展错误码**：cuSPARSE 用 `INVALID_VALUE` 覆盖的状态机非法 / 资源不足，本实现定义 `INVALID_STATE` / `INSUFFICIENT_RESOURCES` 扩展码，文档明示与社区基线合入时同步

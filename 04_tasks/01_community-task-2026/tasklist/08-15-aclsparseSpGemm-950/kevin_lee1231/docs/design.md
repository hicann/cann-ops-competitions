# aclsparseSpGemm 算子开发(950)设计文档

| 版本 | 日期 | 修改人 | 修改内容 |
|------|------|--------|----------|
| v0.1 | 2026-08-18 | kevin_lee1231 | 初稿 |
| v0.2 | 2026-08-20 | kevin_lee1231 | 按 8 月第 15 号任务修正提交目录；同步 950PR 最终实现、自测结论与工程经验；补充交付边界 |

---

# 一、需求背景(required)

## 1.1 需求来源

8 月社区任务第 15 号(A5)——aclsparseSpGemm 算子开发(950)。参考 PyTorch 稀疏矩阵乘法与
`aten::_sparse_sparse_matmul` 的接口和行为,在昇腾 NPU(950PR)上完成 Python/ATen 适配,
并复用现有 SpGEMM 社区任务(7 月)规划交付的 aclsparse C++ 多阶段接口及 Ascend C Kernel,
补齐所需能力(含 complex64 全链路)、稀疏输出构造、测试及文档开发。

- Python/ATen 接口行为以 PyTorch 2.7 及以上版本为准;
- C++ 接口的调用阶段、参数语义、描述符、workspace、算法及错误处理对标 CUDA Toolkit 13.3
  Update 1 所含 cuSPARSE 13.3 Update 1 的 SpGEMM(Generic API §6.6.14);
- 统一使用 `aclsparseSpGEMM*` 命名,合入 `ops-sparse` 仓库 master 分支;
- 适配 PyTorch ≥ 2.7、torch_npu ≥ 26.0.0;核心计算必须在 NPU 上完成,不允许 CPU fallback。
- 本设计文档提交目录固定为
  `04_tasks/01_community-task-2026/tasklist/08-15-aclsparseSpGemm-950/kevin_lee1231/docs/design.md`。

## 1.2 背景介绍

SpGEMM(稀疏矩阵 × 稀疏矩阵)是图计算、代数多重网格、推荐系统的核心原语。PyTorch 的
稀疏矩阵乘法公开入口 `torch.sparse.mm(mat1, mat2)` 在 CUDA 上由
`aten::_sparse_sparse_matmul` 支持,但在昇腾 NPU 上尚无实现。本任务补齐该能力,链路分层:

```
torch.sparse.mm (Python 公开入口)
    └── aten::_sparse_sparse_matmul (ATen NPU 适配:注册、layout 转换、输出构造)
            └── aclsparseSpGEMM* (C++ Host 多阶段接口)
                    └── Ascend C Kernel (NPU 设备侧计算, arch35)
```

C++ 层语义(与 cuSPARSE SpGEMM Generic API 对齐):

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
$$

- A(m×k)、B(k×n)、C(m×n) 均为稀疏 CSR;α、β 为标量;
- 输出 C 的稀疏结构由乘法确定,非零元值由乘积累加得到;
- β 语义对齐 cuSPARSE:C 与结果 C' 具有相同稀疏结构,SpGEMM 不因 β·C 自动产生新的非零位置;
  matC 带内容且 β≠0 时,在既有结构上叠加 β·C_in。

## 1.3 现状分析

| 项 | 现状(截至 2026-08-20) | 结论 |
|---|---|---|
| ops-sparse 仓库 | 已有 `aclsparseSpMM*`、`aclsparseSpMV*` 等接口及 `sparse/spmm/arch{22,35}` 实现;`include/cann_ops_sparse.h` 已定义错误码枚举、`aclsparseOperation_t`、`aclsparseIndexType_t`、复数类型 `aclsparseComplex` | SpGEMM 沿用其命名、目录组织、错误码与测试框架 |
| SpGEMM 接口 | 7 月社区任务已规划 `aclsparseSpGEMM*` 多阶段接口(原型见 §3.2.1),覆盖 float16/bfloat16/float32;截至本文更新时仍未合入 ops-sparse master | A5 开发分支复用同一组接口并补齐 complex64,不得在后续合入时保留重复接口或不兼容原型 |
| 7 月规划架构 | 已合入 cann-ops-competitions 的设计文档(07-9-SpGEMM 目录)采用"符号 + 数值"两阶段:WorkEstimation 完成符号分析(输出结构),Compute 由 arch35 Kernel 完成数值计算,初版统一 ALG1 路径,ALG2/ALG3 预留;fp16/bf16 内部 fp32 累加 | A5 设计与规划保持一致 |
| A5 增量 | 相对 7 月规划,任务新增 Python/ATen NPU 适配、**complex64 全链路**(描述符、多阶段接口、Kernel、输出组装、测试)及 950PR 四 dtype 验证 | 均为 A5 必选交付；当前 v2 开发线已完成实现与任务书指标自测,最终结论仍以社区评审和验收为准 |
| 并行任务 | A2/A3 任务同时分发,均可能修改 aclsparse SpGEMM 的 Host 侧公共代码 | 公共逻辑与硬件差异解耦,保证同主干共存(§4.4) |
| 当前开发分支 | `ops-sparse` 的 `dev-aclsparseSpGemm-950pr-v2` 已具备 C++ 多阶段接口、arch35 数值 Kernel、四 dtype UT、Python 扩展及精度/性能脚本；官方精度 200 例、fp16/bf16 补充 100 例、P-01～P-03 和官方性能 50 例均已在 950PR 完成自测 | 源码与测试能力已闭环；测试证据、环境、代码哈希、ATK 兼容说明和原始 trace 在自测报告中统一冻结,设计文档不替代正式验收 |

---

# 二、需求分析(required)

## 2.1 需求描述

1. 实现 `aten::_sparse_sparse_matmul` 的 NPU 能力,使 `torch.sparse.mm` 在 NPU 上的参数、
   返回值、dtype、shape、device、稀疏 layout、异常行为与目标 PyTorch 版本保持一致;
2. 实现 C = A × B(§1.2),输出稀疏结构、`nnz(C)` 和 values 由乘法结果确定;
3. 复用 7 月规划交付的 `aclsparseSpGEMM*` 接口及 Ascend C Kernel,补齐 950PR 上 float16、
   bfloat16、float32、complex64 的功能、异常处理和测试;**complex64 为必选能力**,
   打通描述符、多阶段接口、Kernel、输出组装及测试全流程;
4. 精度满足《生态算子开源精度标准》混合容差单标杆验收(§4.1);性能满足 A100 cuSPARSE
   对标目标(§4.2);提供 NPU Dispatch 及 Profiler 证据,证明核心计算未回退 CPU。

## 2.2 需求拆解

| 编号 | 需求项 | 说明 |
|---|---|---|
| R1 | Python/ATen NPU 适配 | `torch.sparse.mm` 在 NPU 上注册命中;输入稀疏 layout → CSR 转换;输出 Sparse Tensor 构造(coalesced 语义) |
| R2 | aclsparse C++ 多阶段接口 | 复用 7 月规划接口;明确 WorkEstimation / EstimateMemory / Compute / Copy 调用顺序、workspace 生命周期、描述符状态与错误码 |
| R3 | Ascend C Kernel | arch35 实现,支持 float16 / bfloat16 / float32 / complex64;动态 shape、空行/空列、中间乘积膨胀、长尾行分布 |
| R4 | complex64 专项 | 复数乘加、抵消、排序归并、共轭语义;描述符/各阶段/Kernel/输出组装/测试全链路打通 |
| R5 | 精度达标 | 输出结构与 values 同时校验;结构精确一致,values 混合容差单标杆(CPU Golden 高精度) |
| R6 | 性能达标 | float16/bfloat16/float32 ≥ 1.0× A100;complex64 ≥ 0.8× A100;Kernel 总耗时口径 |
| R7 | 测试与文档 | 任务书自测用例 + C++ UT + 端到端 UT;README 说明环境、编译、测试步骤,保证可复现 |

## 2.3 输入输出规格

| 名称 | 角色 | 格式 | dtype | 索引 | 约束 |
|------|------|------|-------|------|------|
| mat1/self (A) | 输入稀疏矩阵 | CSR(C++ 层);Python 层目标 PyTorch 版本支持的 layout 转 CSR | float16/bfloat16/float32/complex64 | int32(`ACL_SPARSE_INDEX_32I`) | [M,K],列索引有序 |
| mat2/other (B) | 输入稀疏矩阵 | 同上 | 同上 | int32 | [K,N],列索引有序,`A.size(1)==B.size(0)` |
| output (C) | 输出稀疏矩阵 | CSR(C++ 层);Python 层按目标 PyTorch 返回语义构造 | 同输入(A/B/C/computeType 同类型) | int32 | [M,N],结构由计算确定,规范化 CSR |
| alpha/beta | 标量 | Host/Device 指针(由 handle pointer mode 控制) | 与 computeType 同类型 | - | Python 层固定 alpha=1、beta=0 语义 |

---

# 三、详细设计(required)

## 3.1 算子分析

### 3.1.1 数学公式

单元素计算:

$$
C_{ij} = \sum_{k \in \mathrm{Row}_A(i) \cap \mathrm{Col}_B(j)} A_{ik} \cdot B_{kj}
$$

仅对 A、B 中非零位置累加;同一坐标 (i,j) 的重复项累加并合并为一个条目;
计算产生的显式零值保留并计入 `nnz(C)`(结构非零遵循"宁多不漏"原则,与 cuSPARSE 一致)。

### 3.1.2 支持数据类型

| ID | A/B/C/computeType | acl 枚举 | 说明 |
|----|-------------------|----------|------|
| U1 | float32 | `ACL_FLOAT` | 主路径,确定性要求 |
| U2 | float16 | `ACL_FLOAT16` | cuSPARSE 标记 deprecated,本任务仍须对齐;内部 fp32 累加、输出截断 |
| U3 | bfloat16 | `ACL_BF16` | 同上 |
| U4 | complex64 | `ACL_COMPLEX64` | A5 新增必选;值类型使用头文件已有的 `aclsparseComplex`(layout {real, imag}) |

约束:SpGEMM 仅同精度,A、B、C 及 `computeType` 四者 dtype 相同;fp16/bf16 的
computeType 接口参数与 A/B/C 相同,内部累加精度为实现细节(按 7 月规划 fp32 累加,
以满足 §4.1 混合容差标准)。行偏移/列索引均为 int32(`ACL_SPARSE_INDEX_32I`)且三者一致。

### 3.1.3 支持形状与算子约束

- dynamic shape:M、K、N、nnz(A)、nnz(B) 及中间乘积数量在规格内动态变化,Host 侧完成
  多阶段内存及 tiling 规划;
- 支持矩形矩阵(m/k/n 为规格内非负整数,包括零维度),不做维度广播;
- opA/opB 仅 `ACL_SPARSE_OP_NON_TRANSPOSE`,传入 TRANSPOSE / CONJUGATE_TRANSPOSE 返回
  明确错误(NOT_SUPPORTED);
- 仅 CSR 格式、int32 索引、zero-based;输入 A/B 列索引须有序(sorted),非法索引返回确定错误;
- 输出 C 为规范化 CSR:`rowOffsets` 单调非降,每行 `colIndices` 严格升序,无重复坐标;
- 空输入(A/B 零 nnz、空行、空列、无交集乘积、C 零 nnz)必须定义并测试;
- 最大 shape、nnz、中间乘积数量、workspace、输出存储及索引溢出边界在接口文档中明确
  (int32 索引,规模受 INT32_MAX 上界约束)。

### 3.1.4 异步与确定性

- C++ 数值 Kernel 使用 handle 绑定的调用方 stream 异步执行。当前 V1 的符号阶段需要把
  A/B 的 CSR 结构搬到 Host 生成规范化 C 结构,因此 `WorkEstimation` 会同步该 handle 绑定
  的 stream,但不执行 device-wide synchronize。Python 适配层为保证 raw ACL 指针和描述符
  生命周期安全,每次调用结束销毁 handle 时还会同步同一 stream。因此当前实现不能写成
  “全流程异步”；这些同步属于结构分析和资源回收边界,不是把数值乘法回退到 CPU。自测
  报告需把它们与 NPU Kernel 耗时分开披露;
- 确定性:结构(CSR rowOffsets/colIndices/nnz)对同一输入始终精确一致;fp32 数值通过
  固定"按行序 × 行内列序"的遍历顺序实现 bit-wise 确定性,确定性算法重复执行按任务书
  bit-wise 规则验收,其他算法按混合容差标准验收(§4.1)。

## 3.2 算子实现

### 3.2.1 接口基线(与 7 月规划及 `include/cann_ops_sparse.h` 一致,不得重复新增)

```C
/* 描述符 */
aclsparseStatus_t aclsparseSpGEMMCreateDescr(aclsparseSpGEMMDescr_t *descr);
aclsparseStatus_t aclsparseSpGEMMDestroyDescr(aclsparseSpGEMMDescr_t descr);

/* 阶段1:工作估算 */
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

/* 阶段2:内存估算(ALG2/ALG3) */
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

/* 阶段3:计算 */
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

/* 阶段4:结果收尾(V1 中 Compute 直接写 matC,Copy 执行校验) */
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

如设计评审需调整原型,调整结果必须与现有 SpGEMM 社区任务同步,保持源代码兼容或给出
明确兼容方案,并同步写入公开头文件及接口文档。

### 3.2.2 多阶段调用流程与状态机

```
CreateDescr
   │
   ▼
WorkEstimation(handle, ..., alg, descr, &bufferSize1, externalBuffer1)
   │  —— Host 符号阶段:确定 C 结构、nnz(C) 与中间乘积数;V1 的 bufferSize1=0
   ▼
GetNumProducts(descr, &numProds)   —— 查询中间乘积数量
   │
   ├── alg ∈ {DEFAULT, ALG1}:直接 Compute
   │
   └── alg ∈ {ALG2, ALG3}:
         ▼
       EstimateMemory(handle, ..., chunkFraction, &bufferSize3, externalBuffer3, &bufferSize2)
         │  —— 校验并冻结参数,给出 Compute 所需精确 bufferSize2;V1 的 bufferSize3=0
         ▼
       Compute(handle, ..., &bufferSize2, externalBuffer2) —— 直接写 matC
         ▼
       Copy(handle, ..., descr)   —— V1 为状态、参数和输出结构校验 no-op
   │
   ▼
DestroyDescr
```

| 阶段 | 职责 | 描述符状态要求 | 产出 |
|---|---|---|---|
| CreateDescr | 分配描述符,初始化默认值 | 创建后未初始化 | descr |
| WorkEstimation | Host 符号分析:确定 C 的 rowPtr/nnz/结构非零集及中间乘积数 | 已创建 | `bufferSize1=0`、内部结构规划数据 |
| GetNumProducts | 返回中间乘积数量 | WorkEstimation 之后 | numProds |
| EstimateMemory | 校验 `chunkFraction` 及冻结参数,计算精确 bufferSize2 | WorkEstimation 之后(ALG2/ALG3) | `bufferSize2`、`bufferSize3=0` |
| Compute | 启动数值 Kernel,直接写入 matC 的 CSR 数组 | 前序阶段完成后 | matC(rowOffsets/colIndices/values) |
| Copy | 校验阶段状态、冻结参数及输出结构；V1 不再搬运数据 | Compute 之后 | 成功状态 |
| DestroyDescr | 释放描述符,无资源泄漏 | 生命周期内任意时刻 | — |

- 阶段错序调用返回确定错误码;
- 同一描述符从 WorkEstimation 到 Copy 必须保持 `alg`、A/B/C 描述符、computeType 与 stream
  上下文一致；ALG2/ALG3 未完成 EstimateMemory 时不得直接 Compute,不得把 DEFAULT 的
  WorkEstimation 与 ALG2 的后续阶段混用;
- workspace 生命周期:ALG2/ALG3 由调用方按接口返回大小申请,并保持有效至 Compute 已在
  handle 绑定 stream 上完成；DEFAULT/ALG1 使用 handle workspace,不要求外部 buffer2;
- `nnz(C)` 在 WorkEstimation 后通过 matC 描述符查询；调用方据此申请输出并在 Compute
  前更新 matC 指针,Copy 不再替换输出指针;
- 输入 A/B 只读,接口不得修改输入描述符指向的数据;
- 错误码使用仓内 `aclsparseStatus_t` 枚举:`ACL_SPARSE_STATUS_SUCCESS`、
  `NOT_INITIALIZED`(阶段错序/描述符未初始化)、`ALLOC_FAILED`、`INVALID_VALUE`
  (维度/dtype/索引不匹配)、`ARCH_MISMATCH`、`EXECUTION_FAILED`、`INTERNAL_ERROR`、
  `MATRIX_TYPE_NOT_SUPPORTED`、`NOT_SUPPORTED`(op 非 NON_TRANSPOSE、dtype 组合不支持)、
  `INSUFFICIENT_RESOURCES`(workspace 不足/索引位宽不满足规模)、`HANDLE_IS_NULLPTR`。

### 3.2.3 总体架构(复用 7 月规划:符号 + 数值两阶段)

```
aclsparseSpGEMM* 系列接口(Host 侧 C++)
    ├── aclsparseSpGEMMCreateDescr / DestroyDescr
    ├── aclsparseSpGEMMWorkEstimation  (符号阶段: 计算 rowPtrC / nnzC / 结构非零集)
    ├── aclsparseSpGEMMGetNumProducts  (中间产品数)
    ├── aclsparseSpGEMMEstimateMemory  (内存估算, ALG2/3)
    ├── aclsparseSpGEMMCompute         (数值阶段: kernel launch)
    └── aclsparseSpGEMMCopy            (V1:状态、参数与输出结构校验)
            │
            └── Ascend C Kernel (arch35, dav-3510)
                ├── 通用数值路径(多流归并 + 长尾二分兜底)
                ├── 单来源结构快路径(输出槽 → A 流/B 偏移映射)
                └── complex64 扩展(复数乘加, 实/虚部分别 fp32 累加)
```

**符号阶段(确定 C 结构)**:
```
for i in [0, m):
    colset = ∅
    for (k, _) in A.row(i):
        for (j, _) in B.row(k): colset.insert(j)   # 去重排序
    rowNnz[i] = |colset|
rowPtrC = prefix_sum(rowNnz);  nnz(C) = rowPtrC[m]
```
初版符号阶段在 Host 侧执行(仿 SpMM 预处理,便于与 CPU golden 对拍;性能不足时下沉
device kernel),输出结构与"宁多不漏"原则保证显式零保留(§3.1.1)。

**数值阶段(填值)**:
```
for i in [0, m):
    accum = {}                        # 列 j -> 累加值(fp32 / complex64 实虚分离)
    for (k, a) in A.row(i):
        for (j, b) in B.row(k): accum[j] += a * b   # 固定顺序 -> fp32 bit-wise 确定
    按列号升序输出 C.colIdx = j, C.values = alpha * accum[j] (+ beta * C_in[j])
```

**buffer 策略**(V1 实现与最终接口约束):

| buffer | 使用接口 | DEFAULT/ALG1 | ALG2/ALG3 |
|--------|----------|:---:|:---:|
| buffer1 | WorkEstimation | 0 | 0 |
| buffer2 | Compute | 外部大小为 0;tiling/映射表使用 handle workspace | 调用方 Device workspace,精确容纳 tiling 与可选单来源映射表 |
| buffer3 | EstimateMemory | 不适用 | V1 直接输出路径固定返回 0 |

A5 复用规划接口并公开当前算法关系:DEFAULT 与 ALG1 共享同一数值实现；ALG3 与 ALG2
共享直接输出 Kernel。ALG2/ALG3 要求调用 EstimateMemory,`chunkFraction` 必须为有限且位于
(0,1] 的数并在多阶段调用中保持一致；V1 尚不使用 buffer3 做分块,因此明确返回 0,不制造
“已申请但未消费”的虚假 workspace。后续可在不改变公开函数签名和状态机的前提下扩展
不同内存策略。

### 3.2.4 Host 侧设计

- **入参校验**(仿 `ValidateSpmmInputs` 模式):空指针、format=CSR、索引 int32、base=zero、
  dtype 四者一致、维度匹配(A.cols==B.rows)、opA/opB=NON_TRANSPOSE、alg 枚举、阶段状态；
  除负索引外必须检查 `colIndA < K`、`colIndB < N`、rowOffsets 末项/nnz/单调性及全部
  int32 溢出边界,在 Host 访问 rowOffsets 前先完成边界校验;
- **blockDim**:`PlatformAscendCManager::GetCoreNumAiv()`(950PR=64,fallback 24);
- **符号结构与分核**:Host 对每行生成去重且升序的 `rowPtrC/colIndC`,统计 numProds；
  外层 block 按连续行区间切分,每个 AIC/AIV 对只处理自己的 `[rowStart,rowEnd)`,避免
  多个 block 重复计算整张 C;
- **单来源结构识别**:当输出坐标只有一个乘积来源、A 行流数不超过 8、每行输出不超过
  64 且每条 B 流满足短连续访问条件时,生成反向 `streamMap`(输出槽 → A 行内流号/B 行内
  偏移)。只要任一行不满足条件,整次调用回到通用归并路径,保证正确性优先;
- **tiling/workspace**:Host 侧打包维度、dtype、alpha/beta、快路径标志及映射表偏移；
  workspace 采用 64B 对齐的 `[header][SpgemmTilingData][可选 streamMap]`。指针与结构
  H2D 应绑定调用方 stream,并检查 `bufferSize2` 的输入容量后再写入;
- **标量与输出语义**:alpha/beta 同时支持 Host/Device pointer mode,在 WorkEstimation
  冻结并由后续阶段校验一致性；beta≠0 时输入 matC 必须已提供与乘积结果完全相同的规范化
  CSR pattern。Compute 直接写 rowOffsets/colIndices/values,Copy 只做状态和结构校验;
- **complex64 扩展点**:描述符状态与生命周期验证覆盖 complex64;WorkEstimation/
  GetNumProducts 覆盖 complex64 的中间乘积估算与边界行为;EstimateMemory 覆盖 complex64
  (元素宽度 8B)在 ALG2/ALG3 的内存估算;Compute/Copy 覆盖 complex64 的 Kernel 调用、
  直接输出与收尾校验;alpha/beta 为复数标量时按复数语义参与计算。

### 3.2.5 Kernel 侧设计(arch35)

- **编程模型**:dav-3510(arch35),Ascend C SIMT 编程模型,通过 `<<<blockDim, nullptr,
  stream>>>` 启动 AIC 核,核内以 SIMT 线程并行处理行内非零元;所有 AIC 核统一执行入口,
  空行区间立即返回,避免 cube→vector 死锁;
- **通用正确性路径**:A 行长度不超过 16 时维护多个 B 行顺序游标,按 C 的升序列做流归并；
  更长的长尾行逐输出坐标在 B 行内二分查找,避免固定数组越界并覆盖泛化输入;
- **单来源性能路径**:按输出槽顺序读取反向映射,预载 A 值、B 行首与短 B 值流,把随机
  `colIndB` 查找变成寄存器索引。fp16/bf16 先批量计算 8 个结果,再把两个 16-bit 结果
  合成一个 32-bit 对齐事务写回,解决 2B store 吞吐瓶颈;
- **数据类型特化**:C++ 模板 + `if constexpr` 区分四种 dtype;fp16/bf16 以 uint16_t 存储、
  展开为 float 累加后截断回目标类型(与 7 月规划一致);fp32 原生累加、固定遍历顺序保证
  bit-wise 确定;complex64 按 `(a_r + i·a_i)(b_r + i·b_i) = (a_r b_r − a_i b_i) + i·(a_r b_i + a_i b_r)`
  展开,实部/虚部各自 fp32 累加,正确支持复数乘加、抵消与排序归并；fp16/bf16 转换路径
  显式保留 NaN/Inf 类别及 Inf 符号,避免普通有限值转换分支破坏非有限值;
- **边界处理**:空矩阵/零 nnz 由 WorkEstimation 处理,Compute 直接返回 SUCCESS;
  alpha=0 输出全零(结构保留);beta=0 跳过 C 初始值读取;行内无 C 非零时循环自动跳过。
- **编译器规避经验**:950PR bisheng 在“二重间接地址 + half/bfloat16 转换”和每线程大数组
  组合下出现过 507035 指令地址错误。最终实现把 B 行首提前缓存、限制寄存器预载规模并把
  转换与 4B 对写拆成两个循环；这些约束应以注释和回归用例固化,不能保留 `TEMP` 实验提交。
  `-O2` 只应作用于 SpGEMM 的 Ascend C 源文件,不得无回归证据地改变整个 ops-sparse 库。

### 3.2.6 Python/ATen 侧设计(A5 新增)

- **NPU 注册与 dispatch**:在 `SparsePrivateUse1` 为
  `aten::_sparse_sparse_matmul` 注册 NPU 实现；CSR 公开入口所需的
  `SparseCsrPrivateUse1::_sparse_addmm` 只在“两个稀疏输入且 beta=0、alpha=1”时桥接到
  SpGEMM,其余 addmm 语义继续调用 PyTorch native 实现。未覆盖全局 `zeros`/`zero_` 等
  dispatch,`torch.sparse.mm` 和 ATen 内部入口均由回归用例验证命中;
- **稀疏 Tensor ↔ 描述符转换**:支持 COO/CSR。COO 在 NPU 上按(row,col)排序并合并重复
  坐标,再通过 aclsparse COO→CSR 转换生成规范结构；CSR int32 走原生快路径,CSR int64 在
  逐项校验范围后收窄为 int32；非连续 values 显式 materialize 为连续 Tensor。整个转换
  和数值调用链不以 CPU 计算替代 NPU 实现;
- **nnz 与输出构造**:`nnz(C)` 由 WorkEstimation 的符号结果确定并写回 matC 描述符,
  从 matC 描述符取 rowOffsets/colIndices/values。输出 layout 跟随 mat1:COO 输入返回
  coalesced COO,CSR 输入返回 CSR 且保留 mat1 的 int32/int64 索引 dtype；Python 层做最终
  invariant 校验;
- **Python 公开入口映射**:`torch.sparse.mm(mat1, mat2)` → `aten::_sparse_sparse_matmul`,
  与 PyTorch 公开接口的映射保持一致。
- **资源安全**:描述符、handle、workspace 及 raw ACL 异步引用的 Tensor 使用 RAII/keep-alive
  管理,任何 `TORCH_CHECK` 异常路径均释放已创建资源；销毁 handle 的 stream 同步边界按
  §3.1.4 和性能报告公开记录。

### 3.2.7 目录规划(ops-sparse)

沿用仓库规范(`sparse/<算子>/<arch>/`,SOC 前缀匹配 `ascend950` → arch35,
`ascend910b/ascend910_93` → arch22):

```
sparse/spgemm/
└── arch35/                          # 950PR(dav-3510),A5 交付
    ├── spgemm.h                     # 算子内部头
    ├── spgemm_host.cpp              # Host 侧:参数校验、tiling、任务下发
    └── spgemm_kernel.cpp            # Device 侧:Ascend C 核函数(四 dtype)
include/cann_ops_sparse.h            # 追加 aclsparseSpGEMM* 声明 + 算法/描述符类型
CMakeLists.txt                       # 接入 Host 源文件和 arch35 Kernel
test/spgemm/
├── CMakeLists.txt
├── README.md                        # 环境、编译、测试步骤
└── arch35/
    ├── spgemm_test.cpp              # C++ 多阶段 UT(含 complex64 专项)
    └── spgemm_stage_bench.cpp       # 分阶段耗时/workspace/资源诊断(补充证据)
python/
├── torch_sparse_npu/{__init__.py,_C.cpp} # ATen NPU 注册与公开入口适配
├── setup.py                         # Python 扩展构建配置
├── accuracy_runner.py               # 结构 + values 单标杆校验
├── gen_extra_accuracy_cases.py      # fp16/bf16 补充精度用例
├── perf_runner_cxx.py               # 复用描述符/workspace 的 P-01～P-03 正式 Profiler
├── perf_runner_p.py                 # Python 公开入口性能补充证据
├── e2e_bench.py                     # Python 端到端/峰值内存/确定性补充数据
└── smoke_test_mm.py                 # 四 dtype CSR/COO 冒烟用例
```

> 说明:7 月任务书及规划文档写作 `src/spgemm`,与仓库实际顶层目录 `sparse/` 不一致;
> 本设计按仓库实际结构(`sparse/spgemm/`)组织。A2/A3 并行任务后续在
> `sparse/spgemm/arch22/` 共存,公共 Host 逻辑按 §4.4 处理合入冲突。

构建与测试命令(QUICKSTART 格式):

```bash
# 编译 SpGEMM 算子库
bash build.sh --ops=spgemm --soc=ascend950

# 编译并运行 SpGEMM 测试
bash build.sh --ops=spgemm --soc=ascend950 --run
```

### 3.2.8 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR(dav-3510 / arch35) | √ |
| Atlas 800I/T A2(dav-2201 / arch22) | ×(A2 并行任务覆盖,公共代码共存) |
| Atlas A3 | ×(同上) |

当前 950PR 开发环境实测为 CANN 9.1.0-beta.3、PyTorch
`2.14.0a0+gitbe42873`、GitCode 最新源码构建的 torch_npu
`2.14.0+git462f4e1` 及 bisheng 编译器。该源码版本满足任务要求；正式自测报告记录
torch_npu/CANN 的实际源码提交、构建标识和运行环境,避免只凭包版本字符串推断兼容性。

### 3.2.9 算子约束限制

1. 仅 CSR 格式,不支持 COO/CSC/BSR(C++ 层);Python 层负责输入 layout → CSR 转换;
2. 仅 int32 索引(`ACL_SPARSE_INDEX_32I`),A、B、C 三者一致,zero-based;
3. 仅 `ACL_SPARSE_OP_NON_TRANSPOSE`,TRANSPOSE / CONJUGATE_TRANSPOSE 返回明确错误;
4. 仅同精度组合:A/B/C/computeType 四者一致(float16/bfloat16/float32/complex64);
5. 输入 A/B 列索引须 sorted,非法索引返回确定错误;输出 C 列索引保证 sorted;
6. 输出 C 规范化 CSR:rowOffsets 单调非降、colIndices 严格升序、无重复坐标、
   显式零保留;Python 层 COO 输出 coalesced;
7. A、B 为二维稀疏矩阵,`A.size(1) == B.size(0)`,不做维度广播;
8. dynamic shape:M/K/N/nnz/中间乘积在规格内动态变化;
9. C++ 描述符接收连续 values 缓冲；Python 层对非连续 values 做显式连续化后再调用,
   已覆盖相应回归用例;
10. 独立稀疏算子,不要求图融合；数值 Kernel 在调用方 stream 上执行,WorkEstimation 和
    Python handle 回收的 stream 同步边界见 §3.1.4;
11. 规模限制:int32 索引,最大 shape/nnz/中间乘积/workspace/输出存储及溢出边界在
    接口文档中明确。

---

# 四、可维可测分析

## 4.1 精度标准

- 标准来源:《生态算子开源精度标准》(cann/opbase `experimental_standard.md`),
  统一采用**混合容差单标杆**方法验收;
- 单标杆使用 CPU Golden:float16/bfloat16 以 float32 计算 Golden,float32 以 float64 计算
  Golden,complex64 以 complex128 计算 Golden;NPU 结果仅与 CPU Golden 比较;
- **结构与 values 同时校验**:rowOffsets、colIndices、`nnz(C)` 及排序规则精确一致,
  不得仅 dense 化后比较;
- values 逐元素判定:`|actual − golden| ≤ atol + rtol × |golden|`;整体匹配率
  `matched_ratio ≥ 0.99`,且每个元素绝对误差 `≤ max(A, 32 × ULP(golden))`;

| dtype | rtol | atol | max_abs_error_limit(A) |
|---|---|---|---|
| float16 | 2^-9 | 2^-9 | 1e-1 |
| bfloat16 | 2^-6 | 2^-6 | 1e0 |
| float32 | 2^-10 | 2^-16 | 1e-2 |
| complex64 | 同 float32(实部/虚部分别比对) | 同 float32 | 同 float32 |

- 覆盖普通值、小值、正负混合、零值、数值抵消、离群值及规格允许的 INF/NAN 场景;
  INF/NAN 按精度标准文档规则验收(NaN/Inf 位置与符号逐点一致,有限点走混合容差);
- 确定性算法重复执行按结构及 values bit-wise 规则验收;其他算法按混合容差标准验收。
- **结构 Golden 与数值 Golden 分离**:结构只由 A/B 的 CSR 坐标做符号乘法得到,不能直接
  使用 SciPy/框架数值乘法的 `C.indices/C.nnz` 作为唯一结构标杆,因为它们可能删除输入或
  计算产生的显式零。values 再按该冻结结构做高精度累加;
- **complex64 分量独立**:实部和虚部分别计算 finite/NaN/±Inf 位置、符号、matched ratio
  与硬上限,禁止先 cast 到 float64 而丢弃虚部；32×ULP 使用目标 dtype 的相邻可表示数
  (`nextafter`)计算,不能用 `eps × max(1,|golden|)` 近似代替全部 ULP 场景。

## 4.2 性能标准

- 标杆:NVIDIA A100 cuSPARSE SpGEMM NCU Kernel 总耗时(任务书给定,无需自行采集);
- NPU 侧采集与 GPU 相同调用范围内所有 Kernel 总耗时并计算性能倍率;Python 端到端及
  C++ 完整多阶段耗时作为补充数据单独报告;
- 分别报告 work estimation / memory estimation / compute / copy / C++ 完整流程 / Python
  端到端耗时,并报告峰值 workspace、中间乘积数量、`nnz(C)` 和输出存储量;
- 每 case 预热 ≥10 次、正式采样 30 次,报告中位数及 90% 分位耗时;每轮测试执行设备
  同步后计时;描述符与 workspace 在正式采样期间复用;测试时间不含首次编译、数据生成、
  H2D 搬运和无关初始化开销;
- 输入 CSR 按确定性生成规则(列索引 `(i+a) mod n`、`(i+b·d) mod n`,values 全 1,
  alpha=1、beta=0),保证 `nnz(C)` 固定可重复验收;输入按列索引升序、无重复坐标;

| 编号 | M×K×N | nnz(A) | nnz(B) | nnz(C) | dtype | GPU A100 Kernel 总耗时(μs) | 目标 |
|---|---|---|---|---|---|---|---|
| P-01 | 19,717³ | 78,868 | 78,868 | 315,472 | float32 | 289.088 | ≥ 1.0× |
| P-02 | 169,343³ | 1,185,401 | 1,185,401 | 8,297,807 | fp16/bf16/fp32 | 1,127.296 / 1,134.080 / 1,121.376 | ≥ 1.0× |
| P-03 | 1,048,576³ | 8,388,608 | 8,388,608 | 67,108,864 | fp16/bf16/fp32/complex64 | 6,884.864 / 6,876.512 / 6,858.208 / 8,059.968 | fp 系列 ≥ 1.0×;complex64 ≥ 0.8× |

- 必须附 `torch_npu.profiler` 或 `msprof` 证据,确认核心计算未回退 CPU。

**当前 950PR Kernel 正式自测(2026-08-20)**:使用可复用描述符、矩阵描述符、输出缓冲和
workspace 的 C++ 路径,每项先执行 1 次不计时 sizing,再预热 10 次并正式采样 30 次。
msprof 原始 task-time 数据中的设备核函数为 `spgemm_custom_*`。以下结果证明任务书
P-01～P-03 的 Kernel 主指标；自测报告另行冻结代码哈希、环境和原始 trace。

| case | dtype | NPU median(μs) | NPU p90(μs) | GPU/NPU | 任务目标 | 当前结论 |
|---|---|---:|---:|---:|---:|---|
| P-01 | float32 | 16.302 | 16.579 | 17.7333× | ≥1.0× | 达到 |
| P-02 | float16 | 354.261 | 356.201 | 3.1821× | ≥1.0× | 达到 |
| P-02 | bfloat16 | 306.396 | 307.834 | 3.7014× | ≥1.0× | 达到 |
| P-02 | float32 | 379.199 | 380.893 | 2.9572× | ≥1.0× | 达到 |
| P-03 | float16 | 2829.101 | 2834.294 | 2.4336× | ≥1.0× | 达到 |
| P-03 | bfloat16 | 2527.668 | 2533.471 | 2.7205× | ≥1.0× | 达到 |
| P-03 | float32 | 3162.608 | 3166.673 | 2.1685× | ≥1.0× | 达到 |
| P-03 | complex64 | 4989.121 | 4994.432 | 1.6155× | ≥0.8× | 达到 |

分阶段诊断同时表明,Host 符号 WorkEstimation 在 P-03 上的中位数为
2.046～2.059 s,C++ 完整流程中位数为 2.083～2.096 s,远高于数值 Kernel 的毫秒级耗时。
任务的量化目标以 Kernel 总耗时为准,因此上表达标；但 Python 端到端与 C++ 完整流程必须
单独报告。符号阶段下沉 NPU、结构计划缓存和 Python handle/描述符复用是后续工程优化方向,
不能用 Kernel 达标掩盖端到端瓶颈。

## 4.3 测试方案

### 4.3.1 自测用例(任务书 aclsparseSpGemm_testCase)

| 类别 | 覆盖内容 | 工具 |
|---|---|---|
| 基础功能 | CSR 方阵/长矩阵/宽矩阵 × float16/bfloat16/float32/complex64 | 严格 runner + 端到端 UT(官方 fp32/complex64 另经 ATK) |
| 稀疏边界与输出 | nnz=0/1、空行/空列、无交集乘积、多项归并、显式零保留;校验 nnz(C)/rowOffsets/colIndices/values | 严格 runner + 专项用例 |
| 布局限制 | 仅 CSR × NON_TRANSPOSE 组合(对齐 cuSPARSE 13.3 Update 1 SpGEMM 官方规格) | C++ UT |
| 多阶段流程 | CreateDescr→WorkEstimation→GetNumProducts→EstimateMemory/Compute/Copy→DestroyDescr 完整主路径 | C++ UT |
| Workspace 与异常 | 精确 workspace、workspace 不足、维度/dtype/索引不匹配 | C++ UT |
| 精度 | 官方 200 例(FP32 100 + complex64 100),混合容差单标杆,结构精确一致；补充 fp16/bf16 100 例 | `accuracy_runner.py` + ATK 完整任务 |
| 性能 | 官方 50 例(FP32 25 + complex64 25)的 Event/Profiler；P-01～P-03 共 8 项正式 Kernel 对标 | 官方 `profile_sparse_ops_npu.py` / `benchmark_sparse_ops_npu.py` + `perf_runner_cxx.py` |
| NPU 注册与泛化 | dispatch 命中、无 CPU fallback;shape/nnz/稀疏度代表组合抽样 | 端到端 UT + Profiler 证据 |

> 任务书提供的精度/性能用例文件仅覆盖 FP32 + complex64；本任务已按同一生成规则和
> fp16/bf16 容差补充 100 条精度用例,并在 P-02/P-03 覆盖 fp16/bf16 性能组合。

### 4.3.2 补充自测要求

- C++ 接口测试需验证返回码、多阶段状态、输入只读性、输出缓冲区边界和 workspace 边界;
- 连续创建、执行和销毁描述符,不得出现内存/资源泄漏或非法同步;
- 自测报告记录实际使用的 CANN 版本、用例参数、输出结构及精度结果、性能数据、峰值内存、
  截图、Profiler 证据和失败项说明;
- A2/A3 与 A5 环境分别执行精度测试;后合入方完成 A2/A3 与 A5 相关回归测试后方可申请合入。

### 4.3.3 当前开发自测结论与交付边界

截至 2026-08-20,本设计对应的 v2 开发线已在 Ascend 950PR 完成以下自测:

1. 官方精度 JSON 经独立严格 runner 执行 200/200 通过,ATK 完整任务执行 200/200 通过；
   fp16/bfloat16 补充精度用例 100/100 通过。比较器已经采用显式零安全的符号 Golden,
   complex64 实部/虚部分开处理 NaN/Inf、matched ratio 与 32×ULP 硬上限;
2. C++ 多阶段 UT 全部通过,覆盖 DEFAULT/ALG1/ALG2/ALG3、Host/Device pointer mode、
   阶段错序、精确 workspace/少 1 字节、beta≠0、显式零、输入只读、输出 canary、
   维度/dtype/索引边界和负例；Python 四 dtype CSR/COO smoke test 全部通过,并覆盖
   重复/乱序 COO、int64 CSR、非连续 values、非默认 stream、异常及 scoped dispatch;
3. P-01～P-03 的 8 个量化性能项全部达到任务目标；官方性能 JSON 的 50 个 case 完成
   Event 10 次预热/30 次采样(50/50 成功)和 Profiler 采集(50/50 成功)。Profiler 中的
   `spgemm_custom_*` 设备 Kernel 与 dispatch 回归共同证明数值主计算未回退 CPU;
4. 连续创建、执行和销毁的 3 轮独立生命周期检查均通过,未观察到持续增长的 Host RSS,
   Device 内存窗口增量中位数满足测试阈值。

上述结论是开发者自测,不等同于社区最终验收。最终交付仍需在自测报告中冻结
`ops-sparse` v2 分支提交/文件哈希、CANN/PyTorch/torch_npu 实际版本、用例参数、峰值
workspace/内存、截图和原始 trace,并单独说明 ATK 版本兼容处理及失败项历史。ATK 适配只
作用于隔离的测试配置和运行环境,不修改官方原始用例,也不进入算子实现与本设计方案。

## 4.4 兼容性分析

- **与 A2/A3 共存**:公共 Host 逻辑(参数校验、tiling 计算、workspace 规划、描述符管理、
  接口声明)与硬件差异(arch22/arch35 Kernel、特定硬件约束)解耦;后合入方基于已合入
  版本更新并处理 Host 公共代码冲突,合并可复用逻辑并保留两个硬件范围各自所需的分支处理;
- **与仓内既有算子**:`aclsparseSpGEMM*` 为新增符号,不影响 `aclsparseSpMM*`/`aclsparseSpMV*`
  及现有测试;SpGEMM 复用其命名、目录组织与错误码体系。Ascend C `-O2` 等编译选项按源
  文件收敛,不得全局改变其他算子;
- **Python 语义**:`torch.sparse.mm` 在 NPU 上的参数、返回值、dtype、shape、device、
  layout、异常行为与目标 PyTorch 版本(≥2.7)一致,不影响 CUDA/CPU 路径或其他 CSR NPU
  算子；注册范围限定为 SparsePrivateUse1 的目标算子和 CSR 入口所需的 scoped addmm
  bridge,通用 addmm 委托 native 实现,未覆盖全局 `zero_`/`zeros` 语义;
- **cuSPARSE 对齐**:调用阶段、参数语义、描述符、workspace、算法及错误处理对标 cuSPARSE
  13.3 Update 1 SpGEMM;fp16/bf16 同精度路径虽在 cuSPARSE 文档标记 deprecated,仍须对齐支持;
- **资源安全**:连续创建、执行、销毁描述符的生命周期检查通过；数值阶段使用调用方
  stream；Host 符号阶段和 Python handle 回收的同步边界按 §3.1.4 公开记录并逐步收敛,
  不宣称当前全流程无同步。

---

# 五、参考资料

1. A5 任务书 `aclsparseSpGemm 算子开发(950)任务书` 及自测用例 `aclsparseSpGemm_testCase/`;
2. 7 月 SpGEMM 社区任务书:`cann-ops-competitions/04_tasks/01_community-task-2026/docs/202607/SpGEMM_task_doc.md`;
3. 7 月 SpGEMM 设计文档样例:`tasklist/07-9-SpGEMM/{CCuis, gcw_0Yt5TJT2}/docs/`;
4. PyTorch `torch.sparse.mm` 文档、`native_functions.yaml`、`SparseCUDATensorMath.cu` 参考实现;
5. cuSPARSE SpGEMM Generic API(§6.6.14,CUDA Toolkit 13.3 Update 1);
6. 《生态算子开源精度标准》`cann/opbase`(已通读,混合容差单标杆);
7. ops-sparse 仓库:`include/cann_ops_sparse.h`、`sparse/spmm/arch35/`、`test/spmm/`、
   `docs/zh/install/dir_structure.md`、`docs/QUICKSTART.md`、`CONTRIBUTING.md`;
8. 社区任务流程及注意事项(gitcode 讨论区 #39);
9. Ascend C 算子开发文档及 API 文档。

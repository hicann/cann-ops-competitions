# aclsparseSpGEMM 算子设计文档（Atlas A2 / A3）

---

# 一、需求背景（required）

## 1.1 需求来源

2026 年 8 月社区任务——**aclsparseSpGemm 算子开发（A2/A3）**。

参考 PyTorch 稀疏矩阵乘法与 `aten::_sparse_sparse_matmul` 的接口和行为，在昇腾 NPU 上完成
Python/ATen 适配，复用 SpGEMM 社区任务规划交付的 aclsparse C++ 多阶段接口及 Ascend C
Kernel，完成 **Atlas A2 系列产品与 Atlas A3 系列产品** 上的能力补齐、稀疏输出构造、测试与
文档开发。

- Python/ATen 接口行为以 **PyTorch 2.7+** 为准；
- C++ 接口的调用阶段、参数语义、描述符、workspace、算法及错误处理对标
  **CUDA Toolkit 13.3 Update 1 所含 cuSPARSE 13.3 Update 1** 的 SpGEMM；
- 统一使用 `aclsparseSpGEMM*` 命名，沿用既有规划的 SpGEMM 目录及
  `include/cann_ops_sparse.h` 中的接口声明，不重复新增同名或同功能接口；
- 数据类型：`float16`(`ACL_FLOAT16`)、`bfloat16`(`ACL_BF16`)、`float32`(`ACL_FLOAT`)、
  `complex64`(`ACL_COMPLEX64`)；
- 核心计算必须在 NPU 上完成，**不允许 CPU fallback**。

## 1.2 交付目标

| 交付物 | 交付位置 |
|--------|----------|
| **本设计文档** | `cann-ops-competitions` 仓对应社区任务目录，以 PR 形式提交并通过评审后合入 |
| Python/torch 层接口、ATen NPU 适配、输出构造、端到端 UT、aclsparse SpGEMM 公开接口与 Host 实现、Ascend C Kernel 的能力补齐代码、C++ UT | `ops-sparse` 仓 `master` 分支 |
| 自测用例及测试代码、自测报告 | 社区任务 IT 系统提交验收 |

A2/A3 与 A5 任务的 PR 可能先后合入，Host 侧公共代码须可同主干共存（见 §3.5）。

## 1.3 SpGEMM 计算特征

SpGEMM（Sparse General Matrix-Matrix Multiplication）计算 `C = A × B`，其中
A(`M×K`)、B(`K×N`)、C(`M×N`) 三者均为稀疏 CSR 矩阵。与 SpMM（稀疏 × 稠密 → 稠密）的
根本差异在于输出形态：

| 对比项 | SpMM | SpGEMM |
|--------|------|--------|
| 输出形态 | 稠密矩阵，形状已知 | **稀疏矩阵，nnz(C) 计算前未知** |
| 输出内存 | 调用前即可分配 | **必须先符号分析确定结构再分配** |
| 调用形态 | 单阶段（+preprocess） | **多阶段（估算 → 计算 → 拷贝）** |
| 负载均衡 | 按行 nnz 近似均衡 | 按行**中间乘积数**，长尾显著 |

`nnz(C)` 运行前未知，是 SpGEMM 必须采用多阶段接口的根本原因，也是本设计中 workspace
规划、跨核偏移、输出组装等设计的主线。

---

# 二、需求分析（required）

## 2.1 需求描述

在 Atlas A2/A3 上打通 **Python → ATen → aclsparse C++ 多阶段接口 → Ascend C Kernel**
全链路的稀疏矩阵乘法能力，支持 fp16/bf16/fp32/complex64 四种同精度类型，输出规范化 CSR
稀疏矩阵，精度满足《生态算子开源精度标准》混合容差单标杆，性能满足相对 A100 cuSPARSE 的
倍率约束。

## 2.2 需求拆解

| 编号 | 需求项 | 来源（任务书章节） | 验收方式 |
|------|--------|-------------------|----------|
| R1 | `torch.sparse.mm` / `aten::_sparse_sparse_matmul` NPU 能力，参数/返回值/dtype/shape/device/layout/异常与 PyTorch 2.7+ 一致 | 功能实现要求 1 | 端到端 UT + ATK 用例 |
| R2 | `C = A × B`，输出结构与 nnz(C) 由乘法确定 | 功能实现要求 2 | 结构精确比对 |
| R3 | 复用 `aclsparseSpGEMM*` 七个接口，不重复新增同名接口 | 功能实现要求 3 | 代码评审 + C++ UT |
| R4 | 补齐 `complex64`，打通描述符/多阶段/Kernel/输出组装/测试全流程 | 功能实现要求 3、特别注意 2 | complex64 专项用例 |
| R5 | 明确调用顺序、workspace 生命周期、nnz(C) 查询、指针更新、stream 语义、描述符状态、错误码 | 功能实现要求 4 | 接口文档 + C++ UT |
| R6 | 四种 dtype，A/B/C/computeType 同类型 | 功能实现要求 5 | dtype 矩阵测试 |
| R7 | 泛化能力：shape/nnz/稀疏度/空行空列/乘积膨胀/长尾行/合法边界 | 功能实现要求 6 | ATK 泛化用例 |
| R8 | Python/ATen 负责注册、描述符转换、nnz(C) 处理、输出 Sparse Tensor 构造 | 功能实现要求 7 | Dispatch/Profiler 证据 |
| R9 | A2/A3 与 A5 公共 Host 代码解耦，可同主干共存 | 功能实现要求 8、PR 合入 2 | 目录结构 + 回归 |
| R10 | 规范化 CSR 输出：rowOffsets 单调非降、行内 colIndices 严格升序、重复坐标合并、**显式零保留** | 算子约束限制 | 结构精确比对 |
| R11 | 异步执行，禁止无必要 Host 同步；确定性计算 | 算子约束限制 | Profiler + 重复执行比对 |
| R12 | 精度：混合容差单标杆，匹配率 ≥ 0.99，绝对误差 ≤ `max(A, 32×ULP)` | 精度要求 1–8 | ATK |
| R13 | 性能：每个 case×dtype 倍率 > 0.25×A100，全场景均值 ≥ 0.35×A100 | 性能要求 8 | Profiler Kernel 总耗时 |
| R14 | 无 CPU fallback 证据 | 其他要求 | Dispatch + Profiler |

## 2.3 输入输出规格

| 名称 | 角色 | 格式 | dtype | 索引 | 约束 |
|------|------|------|-------|------|------|
| `mat1`/`self`/`matA` | 输入稀疏矩阵 A | C++ 层仅 CSR | fp16/bf16/fp32/complex64 | int32 | `[M,K]`，列索引升序，zero-based |
| `mat2`/`other`/`matB` | 输入稀疏矩阵 B | C++ 层仅 CSR | 同 A | int32 | `[K,N]`，列索引升序，zero-based |
| `output`/`matC` | 输出稀疏矩阵 C | C++ 层 CSR | 同 A | int32 | `[M,N]`，结构由计算确定，规范化 |
| `alpha`/`beta` | 标量 | host/device 指针 | 同 computeType | - | Python 路径固定 `alpha=1, beta=0` |

Python 层负责将目标 PyTorch 版本支持的输入稀疏 layout 转换为 CSR，并按 PyTorch 返回语义
构造输出 layout（返回 COO 时须为 coalesced）。

---

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

C++ 接口保留 cuSPARSE 的完整形态：

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,\quad
(A \cdot B)[i][j] = \sum_{k} A[i][k] \times B[k][j]
$$

Python/ATen 路径为其特例：`opA = opB = NON_TRANSPOSE`，`alpha = 1`，`beta = 0`，即 `C = A × B`。

**结构语义（对齐 cuSPARSE）**：

- C 的结构非零集合 = $\{(i,j) \mid \exists k,\ A[i][k] \neq 0 \wedge B[k][j] \neq 0\}$，
  **由稀疏模式决定，与数值无关**；
- 同一坐标的多个乘积项**累加合并为一个条目**；
- 数值抵消产生的零（如 `1×1 + 1×(-1) = 0`）**保留为显式零并计入 nnz(C)**；
- 输出不含重复坐标，行内列索引严格升序。

### 3.1.2 支持数据类型

| ID | A/B/C values / computeType | acl 枚举 | 枚举值 | 累加精度 |
|----|---------------------------|----------|-------:|----------|
| U1 | float32 | `ACL_FLOAT` | 0 | fp32 |
| U2 | float16 | `ACL_FLOAT16` | 1 | **fp32** |
| U3 | bfloat16 | `ACL_BF16` | 27 | **fp32** |
| U4 | complex64 | `ACL_COMPLEX64` | 16 | **fp32 × 2（实/虚）** |

fp16/bf16 一律在 fp32 中累加，末尾转回目标类型，避免长行累加的精度塌陷。cuSPARSE 虽将
fp16/bf16 同精度路径标记 deprecated，本任务仍须支持。索引类型：`csrRowOffsets` 与
`csrColInd` 均为 `ACL_SPARSE_INDEX_32I` 且必须相同。

### 3.1.3 complex64 设计要点

A2/A3 无原生复数向量指令，采用**实虚分离 + 交错存储**方案：

- 内存布局：与 PyTorch/cuSPARSE 一致，`complex64` 按 `(re, im)` 交错存放，`nnz` 个元素占
  `8 × nnz` 字节；
- 复乘：$(a_r + a_i i)(b_r + b_i i) = (a_r b_r - a_i b_i) + (a_r b_i + a_i b_r)i$，
  4 次乘 + 2 次加，实虚两路各自在 fp32 累加；
- 归并：合并同列条目时实虚分量分别累加，**排序键仅为列索引**，与数值无关，故结构与
  fp32 路径完全一致；
- 抵消：实部或虚部单独为 0、或两者同时为 0，均保留为显式零条目；
- 共轭语义：`opA`/`opB` 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`；传入
  `ACL_SPARSE_OP_CONJUGATE_TRANSPOSE` 与 `ACL_SPARSE_OP_TRANSPOSE` 一律返回
  `ACL_SPARSE_STATUS_NOT_SUPPORTED`，不做隐式共轭。

**关键复用点**：符号阶段（结构计算）**与 dtype 完全无关**，四种类型共用同一份符号 Kernel；
仅数值阶段按 dtype 模板特化。这使 complex64 的增量集中在数值 Kernel 与描述符/校验的
dtype 分支，显著降低新增成本。

### 3.1.4 支持形状与规模边界

| 项目 | 范围 | 说明 |
|------|------|------|
| M / K / N | ≥ 0，受 int32 索引约束 | 自测用例主覆盖 `[1000, 20000]`，任务书性能表最大到 1,048,576 |
| 维度约束 | `A.size(1) == B.size(0)` | 否则返回 `ACL_SPARSE_STATUS_INVALID_VALUE` |
| nnz(A) / nnz(B) | 0 ~ INT32_MAX | 支持 nnz=0/1 |
| 中间乘积数 numProds | int64 计数 | 避免 int32 溢出 |
| nnz(C) | ≤ INT32_MAX | 超出返回 `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` |
| 广播 | 不支持 | A/B 不做矩阵维度广播 |

## 3.2 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│ Python 层    torch.sparse.mm(mat1, mat2)                        │
├─────────────────────────────────────────────────────────────────┤
│ ATen 适配    aten::_sparse_sparse_matmul / _sparse_csr_mm       │
│              (torch_npu 注册)                                    │
│   职责: layout→CSR 转换 | 描述符构造 | 多阶段编排 |              │
│         nnz(C) 查询与输出内存分配 | 输出 Sparse Tensor 构造      │
├─────────────────────────────────────────────────────────────────┤
│ aclsparse C++ 多阶段接口 (include/cann_ops_sparse.h)            │
│   CreateDescr → WorkEstimation → [EstimateMemory] →             │
│   Compute → (SpMatGetSize/CsrSetPointers) → Copy → DestroyDescr │
├─────────────────────────────────────────────────────────────────┤
│ Host 公共层  sparse/spgemm/common/   ← A2/A3 与 A5 共享         │
│   参数校验 | 描述符生命周期 | dtype 分派 | workspace 布局 |      │
│   算法枚举 | 错误码映射                                          │
├──────────────────────────┬──────────────────────────────────────┤
│ sparse/spgemm/arch22/    │ sparse/spgemm/arch35/                │
│ A2/A3: Ascend C 向量模型 │ A5: SIMT 模型（A5 任务交付）         │
│ TPipe/TQue/DataCopyPad   │ asc_vf_call / __simt_vf__            │
└──────────────────────────┴──────────────────────────────────────┘
```

## 3.3 算法方案：符号 + 数值两阶段

采用**符号（symbolic）+ 数值（numeric）**两趟法，并针对 A2/A3 的 UB 容量与本任务实际
负载特征做分级模板设计。

### 3.3.1 负载特征分析（驱动模板设计）

依据自测用例 `sparse_spgemm_performance.json`（50 条，规模 `[1000, 20000]`）与任务书性能表
P-01~P-03（规模最大 1,048,576³）统计：

| 特征 | testCase（50 条） | 性能表 P-01~P-03 | 设计含义 |
|------|-------------------|------------------|----------|
| `degree_a`（A 每行 nnz） | 1 ~ 8 | 4 / 7 / 8 | 行**短** |
| `degree_b`（B 每行 nnz） | 1 ~ 8 | 4 / 7 / 8 | 行**短** |
| 每行中间乘积数 `P_i` | 典型 ≤ 64 | 16 / 49 / 64 | 每行工作集**小** |
| N | 1000 ~ 20000 | 19717 ~ 1,048,576 | N 可能**很大** |
| nnz(C) | 6×10³ ~ 1.5×10⁵ | 3.15×10⁵ ~ 6.7×10⁷ | 大规模场景输出量大 |

**关键推论**：若采用"宽度为 N 的稠密 SPA 累加器"，在 N=20000 且 complex64 时需
`20000×8B(值) + 20000×4B(标记) = 240KB > 192KB UB`，**放不下**；即便 fp32 放得下，为每行
仅 ≤64 个乘积去维护 20000 宽的累加器也是巨大浪费。

因此本设计以 **k 路有序归并（k-way merge）** 为主路径：利用 B 的行内列索引本身有序的性质，
行 i 的乘积项天然是 `nnz(A[i])` 个**有序段**的并集，做多路归并即可**直接得到有序且去重的
输出**，复杂度 `O(P_i · log da)`，**内存占用与 N 无关**。这同时天然满足"行内列索引严格
升序"和"重复坐标合并"两条硬约束。

### 3.3.2 分级模板划分

| 模板 | 触发条件 | 算法 | UB 占用 | 适用场景 |
|------|----------|------|---------|----------|
| **T1 有序归并** | `da ≤ 32` 且 `P_i ≤ 1024` | k 路归并（胜者树） | `O(P_i)`，与 N 无关 | **主路径**，覆盖全部自测用例 |
| **T2 哈希累加** | `P_i ≤ P_hash`（UB 可容纳 2×P_i 哈希槽） | 开放寻址哈希 + 末尾排序 | `O(P_i)` | 中等长行、da 很大 |
| **T3 列分块稠密 SPA** | 其余（超长行 / 高膨胀行） | 按列分块的稠密累加器，多趟扫描 | `O(chunk)` 固定 | 长尾行、极端膨胀 |

三个模板的**累加顺序均严格定义且可复现**（T1 按段序、T2/T3 按 A 行序 × B 行内列序），
因此同一输入重复执行 bit-wise 一致，满足确定性要求（R11）。

行到模板的映射在 `WorkEstimation` 阶段依据 `P_i` 完成分箱（binning），随 tiling 下发。

### 3.3.3 符号阶段（确定 C 的结构）

```
for i in [0, M):                       # 按行并行
    colset = k-way-merge-unique({ B.cols(k) | k ∈ A.cols(i) })
    rowNnz[i] = |colset|
rowOffsets = exclusive_prefix_sum(rowNnz)   # device 扫描
nnz(C) = rowOffsets[M]
```

- **与 dtype 无关**，四种类型共用；
- 只读 A/B 的 `rowOffsets`/`colIndices`，**不读 values**，访存量小；
- 空行/空列/无交集乘积自然产生 `rowNnz[i] = 0`；
- 前缀和用 device 扫描 Kernel 完成，**不做 D2H 同步**（R11）。

### 3.3.4 数值阶段（填值）

```
for i in [0, M):
    base = rowOffsets[i]
    (cols, vals) = k-way-merge-accumulate(
        { (B.cols(k), A.val(i,k) * B.vals(k)) | k ∈ A.cols(i) })
    # 段序固定 → 累加顺序固定 → bit-wise 确定
    C.colIndices[base ...] = cols       # 已升序，无需再排序
    C.values[base ...]     = alpha * vals  (+ beta * C_in)
```

- 与符号阶段**共用同一归并骨架**，保证两趟结构完全一致；
- 不做任何"数值为零则丢弃"的判断 → **显式零自动保留**（R10）；
- fp16/bf16 在 fp32 累加后统一 Cast 回目标类型；complex64 实虚双路累加。

### 3.3.5 多核并行与长尾负载均衡

- 输出行之间**完全无依赖**（C 的第 i 行只依赖 A 的第 i 行与整个 B），是天然的并行维度；
- 分核**不按行数均分**，而按 `P_i`（每行中间乘积数）做**贪心装箱**：行按 `P_i` 降序依次
  放入当前负载最小的核，直接应对任务书要求的"长尾行分布"（R7）；
- 装箱结果以 `binEdge[coreIdx]` 数组下发；
- 跨核输出偏移由符号阶段的全局前缀和给出，数值阶段各核**直接写入最终位置**，
  无需跨核 barrier。

## 3.4 多阶段接口设计（Host 侧）

### 3.4.1 调用时序（对齐 cuSPARSE，明确 R5 全部语义）

```
aclsparseSpGEMMCreateDescr(&spgemmDescr)
  │
  ├─ aclsparseSpGEMMWorkEstimation(..., &bufferSize1, nullptr)   # 查询
  │  aclrtMalloc(&buffer1, bufferSize1)
  ├─ aclsparseSpGEMMWorkEstimation(..., &bufferSize1, buffer1)   # 执行：符号规划+分箱
  │
  ├─ aclsparseSpGEMMGetNumProducts(spgemmDescr, &numProds)       # 可选查询
  │
  ├─ [ALG2/ALG3] aclsparseSpGEMMEstimateMemory(..., chunkFraction,
  │                  &bufferSize3, nullptr, &bufferSize2)        # 查询
  │              aclrtMalloc(&buffer3, bufferSize3)
  │              aclsparseSpGEMMEstimateMemory(..., buffer3, &bufferSize2)
  │
  ├─ aclsparseSpGEMMCompute(..., &bufferSize2, nullptr)          # 查询
  │  aclrtMalloc(&buffer2, bufferSize2)
  ├─ aclsparseSpGEMMCompute(..., &bufferSize2, buffer2)          # 执行：符号+数值
  │
  ├─ aclsparseSpMatGetSize(matC, &m, &n, &nnzC)                  # 取 nnz(C)
  │  aclrtMalloc(dC_cols, nnzC*4); aclrtMalloc(dC_vals, nnzC*sizeof(T))
  │  aclsparseCsrSetPointers(matC, dC_rowOffsets, dC_cols, dC_vals)
  │
  ├─ aclsparseSpGEMMCopy(...)                                    # 结果搬运到 matC
  │
  └─ aclsparseSpGEMMDestroyDescr(spgemmDescr)
```

### 3.4.2 各阶段职责与本任务补齐点

| 阶段 | 接口 | 职责 | 本任务补齐（A2/A3 + complex64） |
|------|------|------|-------------------------------|
| 描述符创建 | `aclsparseSpGEMMCreateDescr` | 分配内部描述符，初始化状态机 | 补齐 complex64 描述符状态与生命周期验证 |
| 工作量估算 | `aclsparseSpGEMMWorkEstimation` | 校验入参；计算 `numProds`、每行 `P_i`、贪心装箱；产出 `bufferSize1` | 新增 complex64 在 A2/A3 及各算法下的工作量与 workspace 估算 |
| 乘积数查询 | `aclsparseSpGEMMGetNumProducts` | 返回 `numProds` | 验证四种 dtype 下的查询与边界行为 |
| 内存估算 | `aclsparseSpGEMMEstimateMemory` | ALG2/ALG3 分块内存规划，`chunkFraction` 控制分块比例 | 新增 complex64 在 ALG2/ALG3 下的内存估算 |
| 结构+数值 | `aclsparseSpGEMMCompute` | 符号 Kernel → 前缀扫描 → 数值 Kernel；结果暂存 buffer2 | 新增 complex64 Ascend C Kernel、精度与泛化能力 |
| 结果拷贝 | `aclsparseSpGEMMCopy` | 将 rowOffsets/colIndices/values 搬运至 matC 用户内存 | 补齐 complex64 values 拷贝与 CSR 结构组装 |
| 描述符销毁 | `aclsparseSpGEMMDestroyDescr` | 释放内部资源 | 验证完整调用链无资源泄漏 |

### 3.4.3 workspace 生命周期与边界（R5）

| buffer | 申请者 | 生命周期 | DEFAULT/ALG1 | ALG2/ALG3 |
|--------|--------|----------|:---:|:---:|
| `externalBuffer1` | 调用方 | WorkEstimation 执行 → Copy 结束 | 必需 | 必需 |
| `externalBuffer2` | 调用方 | Compute 执行 → Copy 结束 | 必需 | 必需 |
| `externalBuffer3` | 调用方 | EstimateMemory 期间 | 不需要 | 必需 |

- 三个 buffer 在 `Copy` 完成前**不得释放或改写**，否则返回
  `ACL_SPARSE_STATUS_INVALID_VALUE`；描述符内记录 buffer 指针与大小用于校验；
- `externalBuffer` 传 `nullptr` 表示**查询模式**，仅回填大小不执行计算；
- **workspace 不足**（调用方传入的 size 小于所需）返回
  `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`。

**buffer1 布局**（64B 对齐）：

```
[SpgemmTilingData][rowProducts(int64×M)][rowNnz(int32×M)][binEdge(int32×(blockDim+1))]
[rowTemplate(uint8×M)][scan 中间结果]
```

**buffer2 布局**：

```
[C.rowOffsets(int32×(M+1))][C.colIndices(int32×nnzC)][C.values(sizeof(T)×nnzC)]
```

### 3.4.4 描述符状态机与错误码

内部状态：`CREATED → ESTIMATED → MEM_ESTIMATED → COMPUTED → COPIED`。
阶段乱序调用（如未 WorkEstimation 直接 Compute）返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。

| 错误场景 | 返回码 |
|----------|--------|
| handle / descr / matA / matB / matC 为空 | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| 非 CSR 格式 | `ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED` |
| `opA`/`opB` 为 TRANSPOSE / CONJUGATE_TRANSPOSE | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| 索引非 int32 或 rowOffsets/colInd 类型不一致 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| 非 zero-based | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| dtype 四者不一致 / 不在支持集 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| `A.cols != B.rows` 或 C 维度不匹配 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 算法枚举非法 | `ACL_SPARSE_STATUS_NOT_SUPPORTED` |
| 阶段调用顺序错误 / buffer 校验失败 | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| workspace 不足、nnz(C) 溢出 int32 | `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES` |
| Kernel 执行失败 | `ACL_SPARSE_STATUS_EXECUTION_FAILED` |

### 3.4.5 stream 与异步语义（R11）

- 所有 device 操作均下发到 `aclsparseGetStream(handle)` 返回的调用方 stream；
- 阶段之间**不插入 `aclrtSynchronizeStream`**；
- 唯一必要的 D2H 同步发生在 `aclsparseSpMatGetSize` 读取 `nnz(C)`——这是 SpGEMM 输出
  规模运行时才确定的**协议性必然**，与 cuSPARSE 行为一致，且发生在两次 Kernel 批次之间，
  不构成"无必要的 Host 同步"。

## 3.5 A2/A3 与 A5 共存方案（R9）

任务书明确 A2/A3 与 A5 任务的 PR 可能先后合入，须保证 Host 侧公共代码可共存。设计如下：

```
include/cann_ops_sparse.h              # 沿用既有 aclsparseSpGEMM* 接口声明
sparse/spgemm/
├── common/                            # ★ 硬件无关公共层（两任务共享）
│   ├── spgemm_common.h                #   描述符定义、状态机、算法枚举
│   ├── spgemm_validate.cpp            #   入参校验（dtype/format/索引/维度/op/alg）
│   ├── spgemm_descr.cpp               #   描述符生命周期
│   └── spgemm_dispatch.h              #   按 SOC + dtype 分派到 arch 实现
├── arch22/                            # ★ 本任务交付（A2/A3）
│   ├── spgemm.h                       #   TilingData 定义
│   ├── spgemm_host.cpp                #   arch22 多阶段实现
│   ├── spgemm_symbolic_kernel.cpp     #   符号 Kernel（dtype 无关）
│   └── spgemm_numeric_kernel.cpp      #   数值 Kernel（4 dtype 模板特化）
└── arch35/                            # ☆ A5 任务交付（SIMT）
test/spgemm/{CMakeLists.txt, arch22/spgemm_test.cpp, arch35/...}
```

**冲突面控制**：

1. 公共层只承载**硬件无关**逻辑，A5 的 SIMT 细节不进入 `common/`；
2. `arch22/` 与 `arch35/` 目录**互不重叠**，CMake 依 `SOC_VERSION` 自动选择，与仓内
   `spmm` 的双 arch 并存格局一致；
3. 唯一共享文件 `include/cann_ops_sparse.h` 沿用既有声明，不重复新增同名或同功能接口；
4. **后合入方**须基于已合入版本更新代码，合并 `common/` 可复用逻辑，保留两个硬件范围
   各自所需的分支处理，并完成 A2/A3 与 A5 双向回归后再申请合入。

## 3.6 Python / ATen 适配设计（R1、R8）

### 3.6.1 注册点

| PyTorch 入口 | ATen 算子 | layout | NPU 注册 |
|--------------|-----------|--------|----------|
| `torch.sparse.mm(mat1, mat2)`（双稀疏） | `aten::_sparse_sparse_matmul` | COO | `SparseNPU` |
| 同上（CSR 输入） | `aten::_sparse_csr_mm` / `_sparse_sparse_matmul` | CSR | `SparseCsrNPU` |

自测脚本 `function_sparse_ops.py` 以 `torch.sparse_csr_tensor` 构造输入并调用
`torch.sparse.mm`，而校验函数 `_canonical_csr` 同时容忍 CSR 与 **coalesced COO** 输出。
故**两条 layout 路径均须注册**，且 COO 输出必须置 coalesced 标志。

### 3.6.2 适配层职责与流程

```
1. 入参校验：2D、dtype 一致、device 一致、A.size(1)==B.size(0)
              → 不满足则抛出与 PyTorch 一致的异常
2. layout 归一：COO → CSR（coalesce + convert）；CSR 直接使用
                索引 int64 → int32（含溢出检查）
3. 构造描述符：aclsparseCreateConstCsr(matA/matB)、
                aclsparseCreateCsr(matC, nnz=0, ptrs=null)
4. 多阶段编排：CreateDescr → WorkEstimation(查询/执行) → Compute(查询/执行)
5. nnz(C) 获取：aclsparseSpMatGetSize → 按 nnz(C) 分配输出 Tensor 存储
6. 指针回填：aclsparseCsrSetPointers → aclsparseSpGEMMCopy
7. 输出构造：按目标 layout 构造 Sparse Tensor
              CSR: torch.sparse_csr_tensor(crow, col, values, size)
              COO: 转换后标记 coalesced
8. 资源释放：DestroyDescr / DestroySpMat / 释放 workspace
```

### 3.6.3 类型提升与异常对齐

- dtype 提升遵循 PyTorch 语义；本任务 A/B 同类型，输出同类型；
- 空输入（nnz=0）、空行、空列均返回**合法的空结构 CSR**（`rowOffsets` 全零，
  `nnz(C)=0`），不抛异常；
- 不支持的 dtype/layout 组合，抛出与 PyTorch CPU/CUDA 一致的错误类型与信息。

### 3.6.4 无 CPU fallback 保证（R14）

- 注册 key 使用 `SparseNPU` / `SparseCsrNPU`，不注册 `CompositeImplicitAutograd` 兜底；
- 验收证据：
  1. `torch_npu.profiler` 采集的 `op_statistic.csv` 中出现 SpGEMM 相关 Kernel；
  2. Dispatch 日志（`TORCH_SHOW_DISPATCH_TRACE=1`）确认命中 NPU 注册；
  3. Profiler timeline 中 CPU 侧无对应计算算子。

## 3.7 Kernel 侧设计（arch22 / A2 / A3）

### 3.7.1 编程模型

- 架构：Atlas A2/A3（arch22），**Ascend C 向量编程模型**（`TPipe` / `TQue` /
  `DataCopyPad`），与仓内 `spmm/arch22` 一致；
- 与 A5(arch35) 的 SIMT `asc_vf_call` 模型**不共享 Kernel 代码**；
- 入口：`spgemm_arch22_symbolic_kernel` 与
  `spgemm_arch22_numeric_kernel_{fp32,fp16,bf16,c64}`，由 Host 按 dtype 分派。

### 3.7.2 TilingData

```cpp
struct SpgemmArch22TilingData {
    uint32_t M, K, N;              // 矩阵维度
    uint32_t nnzA, nnzB;           // 输入非零数
    uint32_t blockDim;             // 使用核数
    int64_t  numProds;             // 中间乘积总数
    float    alphaRe, alphaIm;     // 标量（complex64 用双分量）
    float    betaRe,  betaIm;
    uint32_t dtypeId;              // 0=fp32 1=fp16 2=bf16 3=complex64
    uint32_t mergeCapacity;        // T1 单行归并缓冲容量
    uint32_t hashCapacity;         // T2 哈希槽数
    uint32_t chunkWidth;           // T3 列分块宽度
    int64_t  binEdgeOffset;        // 贪心装箱边界数组偏移
    int64_t  rowNnzOffset;         // 每行 nnz 数组偏移
    int64_t  rowTemplateOffset;    // 每行模板选择数组偏移
    int64_t  cRowOffsetsOffset;    // buffer2 内 C 结构偏移
    int64_t  cColIndicesOffset;
    int64_t  cValuesOffset;
};
```

### 3.7.3 UB 预算（A2/A3：192KB，可用约 184KB）

以 T1 主路径、单行归并容量 `mergeCapacity = 1024` 计：

| 缓冲 | 大小（fp32） | 大小（complex64） | 说明 |
|------|-------------|------------------|------|
| A 行段索引/值 | 32×(4+8) = 384B | 32×(4+8) = 384B | ≤32 段的游标与标量 |
| B 行数据暂存 | 1024×(4+4) = 8KB | 1024×(4+8) = 12KB | 列索引 + 值 |
| 归并输出 col | 1024×4 = 4KB | 4KB | int32 |
| 归并输出 val | 1024×4 = 4KB | 1024×8 = 8KB | fp32 / complex64 |
| 胜者树 | 32×4 = 128B | 128B | k 路归并 |
| 双缓冲系数 | ×2 | ×2 | 搬运与计算重叠 |
| **合计** | **≈ 33KB** | **≈ 49KB** | **远小于 184KB ✅** |

对比"宽度为 N 的稠密 SPA"方案：后者在 N=20000 + complex64 时需 240KB，**超出 UB**。
T1 方案的 UB 占用与 N 完全无关，这是选择归并路径的核心理由。富余 UB 可用于提升
`mergeCapacity` 或加深 double buffer。

T3 列分块模板的 `chunkWidth` 由 Host 按 dtype 反推：
`chunkWidth = floor(可用UB / (sizeof(T) + 4))`，保证任意 N 都能分块处理。

### 3.7.4 数据流

```
符号阶段:
  GM(A.rowOffsets, A.colIndices, B.rowOffsets, B.colIndices)
    ─DataCopyPad→ UB ─k路归并去重→ rowNnz[i] ─DataCopyPad→ GM(buffer1)
  ─ device 前缀扫描 Kernel → GM(C.rowOffsets), nnz(C)

数值阶段:
  GM(A.*, B.*) ─DataCopyPad→ UB
    ─[fp16/bf16: Cast→fp32]→ k路归并累加(fp32) ─[Cast回目标类型]→
  UB ─DataCopyPad→ GM(C.colIndices[base], C.values[base])
```

所有 GM 写回均使用 `DataCopyPad` **按行批量**进行，避免逐元素标量写。

### 3.7.5 边界处理

| 场景 | 处理 |
|------|------|
| M=0 或 N=0 或 K=0 | `rowOffsets` 全零，`nnz(C)=0`，不启动数值 Kernel |
| nnz(A)=0 或 nnz(B)=0 | 同上 |
| A 行为空 | `rowNnz[i]=0`，归并直接跳过 |
| B 中被引用行为空 | 该段长度 0，归并自动忽略 |
| 无交集乘积 | `rowNnz[i]=0` |
| 数值抵消为 0 | **保留显式零** |
| INF/NAN 输入 | 按 IEEE754 正常传播，结构不受影响 |

## 3.8 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas 800I A2 推理产品 | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
| A5（arch35） | 由 A5 任务交付，本任务保证共存 |

- **CANN 版本**：算子开源仓指定版本（自测报告中记录实测版本）
- **PyTorch**：2.7 及以上；**torch_npu**：26.0.0 及之后
- 功能与精度需提交 A2 及 A3 上的测试结果；性能仅需提交 A3 上的测试结果

## 3.9 算子约束限制

1. **dynamic shape**：支持规格内的 `M`、`K`、`N`、`nnz(A)`、`nnz(B)` 和中间乘积数量动态
   变化，Host 侧完成多阶段内存及 tiling 规划；
2. A、B 必须为二维稀疏矩阵，且 `A.size(1) == B.size(0)`；不支持广播；
2. C++ 层 A/B/C 均**仅支持 CSR**（`aclsparseCreateCsr`），不支持 COO/CSC/BSR；
3. 索引：`csrRowOffsetsType` 与 `csrColIndType` 均仅支持 `ACL_SPARSE_INDEX_32I` 且必须相同；
   仅 zero-based；输入 A/B 列索引必须有序，非法索引返回确定错误；
4. `opA`、`opB` 仅支持 `ACL_SPARSE_OP_NON_TRANSPOSE`；TRANSPOSE / CONJUGATE_TRANSPOSE
   返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`；
5. dtype：fp16/bf16/fp32/complex64，且 A/B/C/computeType **四者一致**；
6. 输出 C 为规范化 CSR：`rowOffsets` 单调非降、行内 `colIndices` 严格升序、无重复坐标、
   **显式零保留并计入 nnz(C)**；Python 层返回 COO 时须为 coalesced；
7. 布局：三矩阵均为 CSR，不适用稠密 Row-major/Column-major 参数；
8. 不要求图融合，作为独立稀疏算子实现；
9. 异步执行，使用调用方 stream；确定性计算（重复执行 bit-wise 一致）；
10. 规模上限：`nnz(C)` 与索引受 INT32_MAX 约束，超限返回
    `ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES`；
11. 非连续 Tensor：稀疏 values 及元数据的支持范围按任务书明确规格验收，未支持场景需
    返回明确错误。

---

# 四、可维可测分析

## 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度标准 | 混合容差**单标杆**：NPU 结果仅与 CPU Golden 比较；结构（rowOffsets/colIndices/nnz(C)）**精确一致**，values 按逐 dtype 容差 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能标准 | 每个 case×dtype 场景倍率 **> 0.25×A100**，全部量化场景倍率**算术平均 ≥ 0.35×A100** | 任务书性能要求 8 |

### 4.1.1 精度参数

Golden 计算精度：fp16/bf16 → fp32；fp32 → fp64；complex64 → complex128。

| dtype | rtol | atol | A（绝对误差硬上限基值） |
|-------|------|------|------------------------|
| float16 | 2⁻⁹ | 2⁻⁹ | 1e-1 |
| bfloat16 | 2⁻⁶ | 2⁻⁶ | 1e0 |
| float32 | 2⁻¹⁰ | 2⁻¹⁶ | 1e-2 |
| complex64 | 实/虚部各按 float32 参数 | 同左 | 同左 |

判定规则：逐元素 `|actual − golden| ≤ atol + rtol × |golden|`；整体匹配率 **≥ 0.99**；
且每个元素绝对误差**不得超过** `max(A, 32 × ULP(golden))`。complex64 实虚两部分**均须**
满足匹配率与绝对误差硬上限。

**确定性验收**：确定性算法重复执行时，结构及 values 按 **bit-wise 一致**规则验收；
非确定性算法（若后续扩展）按上述混合容差标准验收。本设计的 T1/T2/T3 模板累加顺序均
严格定义，属于确定性算法。

**设计对精度的支撑**：fp16/bf16/complex64 全部在 **fp32 累加**；由于每行中间乘积数典型
≤64，累加链极短，舍入误差天然受控。数值抵消场景因保留显式零且累加顺序固定，结果可预测。
INF/NAN 按 IEEE754 正常传播，精度验收按《生态算子开源精度标准》中 INF/NAN 对应规则执行。

### 4.1.2 结构校验（不可仅 dense 化比较）

任务书与 `accuracy_sparse_ops.py` 均要求**同时**校验结构与数值：
`output_0(crow)`、`output_1(col)`、`output_3(nnz)` 走 `torch.equal` **精确比较**，
`output_2(values)` 走混合容差。本设计以纯符号结构 + 有序归并直接保证结构正确性。

## 4.2 测试方案

### 4.2.1 官方自测用例（`aclsparseSpGemm_testCase/`）

| 文件 | 内容 | 用途 |
|------|------|------|
| `sparse_spgemm_accuracy.json` | **200 条**精度用例（fp32 100 + complex64 100） | ATK 精度验收 |
| `sparse_spgemm_performance.json` | **50 条**性能用例（fp32 25 + complex64 25） | 性能验收 |
| `sparse_spgemm_gpu_ncu_baseline.csv` | 50 条对应的 A100 NCU Kernel 总耗时基线 | 性能对比（直接使用，无需重采） |
| `nodes_accuracy.yaml` | NPU 测试节点 + CPU Golden 节点 | ATK 拓扑 |

精度用例已覆盖：`value_mode` ∈ {0 普通值, 1 小值, 2 离群值, 3 正负抵消, 5 INF/NAN}，
`empty_a_every`/`empty_b_every` 覆盖空行，`degree_a`/`degree_b` ∈ [1,8] 覆盖不同稀疏度，
shape 覆盖 `[1000, 20000]` 含端点及方阵/长矩阵/宽矩阵。

执行：

```bash
atk task -c sparse_spgemm_accuracy.json -n nodes_accuracy.yaml --task accuracy -p .
python profile_sparse_ops_npu.py --case-file sparse_spgemm_performance.json --device 0
```

### 4.2.2 需自行补充的用例

官方精度用例仅含 fp32 与 complex64，而任务书要求四种 dtype 全链路打通，故须自建：

| 编号 | 场景 | 覆盖需求 |
|------|------|----------|
| S-01 | **fp16 / bf16** 精度用例（复用官方 shape 集，替换 dtype） | R6、精度要求 4/5 |
| S-02 | nnz(A)=0 / nnz(B)=0 / nnz=1 边界 | R7、空输入 |
| S-03 | 全空行 / 全空列 / 无交集乘积（nnz(C)=0） | R7 |
| S-04 | **显式零保留**（`1×1 + 1×(−1)`，断言 nnz(C)=1 且值为 0） | R10 |
| S-05 | 中间乘积高膨胀 + 长尾行分布（少数行 P_i 极大） | R7、T3 模板 |
| S-06 | **多阶段完整主路径** Create→WorkEstimation→[EstimateMemory]→Compute→Copy→Destroy | R5 |
| S-07 | **精确 workspace** 用例 + **workspace 不足** 用例 | 测试标准 |
| S-08 | 维度 / dtype / 索引类型不匹配的异常用例 | R5 |
| S-09 | `opA`/`opB` 传 TRANSPOSE / CONJUGATE_TRANSPOSE 返回 NOT_SUPPORTED | 约束 4 |
| S-10 | 连续 Create/Execute/Destroy ×N，检查无内存与资源泄漏 | 其他要求 |
| S-11 | 重复执行 bit-wise 一致性（确定性） | R11 |
| S-12 | ATen Dispatch 命中 NPU、无 CPU fallback | R14 |
| S-13 | 任务书性能表 P-01(19717³,d=4) / P-02(169343³,d=7) / P-03(1048576³,d=8) 确定性生成 | 性能要求表 |

### 4.2.3 性能基线与目标分析

#### 任务书性能表（主验收口径）

任务书定义的性能验收数据采用**确定性生成规则**：对于规模为 $n \times n$、每行非零元素数为
$d$ 的矩阵，A 的第 $i$ 行列索引为 $(i+a) \bmod n$，B 的第 $i$ 行列索引为
$(i+b \times d) \bmod n$，其中 $a,b \in [0,d-1]$。当 $d^2 < n$ 时，每行输出恰有 $d^2$
个非零元素，`nnz(C) = n × d²`。values 均为 1，`alpha=1, beta=0`。

| 编号 | M×K×N | nnz(A) | nnz(B) | nnz(C) | dtype | A100 NCU Kernel 总耗时(μs) |
|------|-------|--------|--------|--------|-------|---------------------------|
| P-01 | 19,717³ | 78,868 | 78,868 | 315,472 | fp32 | 289.088 |
| P-02 | 169,343³ | 1,185,401 | 1,185,401 | 8,297,807 | fp16/bf16/fp32 | 1127.296 / 1134.080 / 1121.376 |
| P-03 | 1,048,576³ | 8,388,608 | 8,388,608 | 67,108,864 | fp16/bf16/fp32/c64 | 6884.864 / 6876.512 / 6858.208 / 8059.968 |

**换算 NPU 耗时预算**（倍率 = GPU耗时 / NPU耗时）：

- 单场景达标线（倍率 > 0.25）：NPU < **4 × GPU**；
- 全部量化场景算术平均达标线（均值 ≥ 0.35）：NPU 平均 ≈ **2.86 × GPU**。

#### testCase 50 条性能用例（补充验收口径）

依据 `sparse_spgemm_gpu_ncu_baseline.csv` 统计（`[1000, 20000]` 规模，仅 fp32/complex64）：

| 指标 | A100 Kernel 总耗时 |
|------|-------------------:|
| 最小 | 224.1 μs |
| 中位 | 253.9 μs |
| 最大 | 403.8 μs |

#### 采集口径

- **C++ 性能**（对齐任务书要求 4/5）：至少预热 10 次、正式采样 30 次，报告耗时**中位数**
  及 **90% 分位耗时**；每轮执行设备同步后计时；描述符与 workspace 在采样期间复用；
  测试时间不含首次编译、数据生成、H2D 搬运及无关初始化开销；
- **Python 端到端**（补充数据）：每 case 独立 `torch_npu.profiler` 会话，5 warmup +
  5 active step，读取 `op_statistic.csv` 的 `Total Time(us)` 求和后除以 5；
- 分别报告 WorkEstimation / EstimateMemory / Compute / Copy / C++ 完整流程 / Python
  端到端耗时，并报告峰值 workspace、numProds、nnz(C) 与输出存储量。

## 4.3 风险评估与应对

| 编号 | 风险 | 影响 | 应对措施 |
|------|------|------|----------|
| RK-1 | **标量访存密集**：SpGEMM 的间接寻址（`B[A.col[k]]`）难以向量化，A2/A3 上逐元素访问开销大 | **性能不达标（最高风险）** | ①按行 `DataCopyPad` 批量搬运替代逐元素读；②k 路归并将总操作数降至 `O(P·log da)`；③贪心装箱消除长尾；④双缓冲重叠搬运与计算；⑤预留 T2 哈希路径作为备选 |
| RK-2 | complex64 无原生向量支持 | 性能/精度 | 实虚分离双路 fp32 累加；结构与 fp32 共用符号 Kernel，增量可控 |
| RK-3 | 任务书性能表 P-01~P-03（最大 1,048,576³，nnz(C)=67M）规模远超 testCase（≤20000），大规模下内存搬运与多核调度开销不同 | 性能达标线判定 | 两套口径均准备用例；大规模场景优化 DataCopyPad 批量搬运策略；必要时调整贪心装箱粒度 |
| RK-4 | 大规模场景 nnz(C) 接近 INT32_MAX | 溢出 | int64 计数中间乘积，输出前检查并返回明确错误码 |
| RK-5 | A5 任务先合入导致公共层冲突 | 合入受阻 | `common/` 仅存硬件无关逻辑；后合入方基于已合入版本更新 + 双向回归 |
| RK-6 | Python/ATen 层与 `ops-sparse` 既有构建体系的集成 | 工期 | 作为独立子模块搭建，优先打通 fp32 单链路再扩 dtype |

## 4.4 兼容性分析

- `aclsparseSpGEMM*` 沿用既有规划接口，不改动 `aclsparseSpMM`/`SpMV`/`SpSM` 等既有接口，
  对存量用户无影响；
- 与 SpGEMM 社区任务规划保持**源码兼容**（签名、调用顺序、语义完全一致），本任务扩展
  硬件范围（A2/A3）与数据类型（complex64）；
- ATen 注册为新增 NPU 实现，不改变 PyTorch 公开接口语义；
- 后续可扩展方向：int64 索引、COO/CSC 格式、ALG2/ALG3 深度分块优化、fp64/complex128。

## 4.5 交付件清单

| 序号 | 交付件 | 内容 | 交付位置 |
|------|--------|------|----------|
| 1 | 算子设计文档 | 本文档 | `cann-ops-competitions` 对应社区任务目录，PR 提交并通过评审 |
| 2 | 自测用例及测试代码 | 官方 200 精度 + 50 性能用例执行代码、S-01~S-13 补充用例、complex64 专项、README（环境/编译/测试步骤） | 社区任务 IT 系统 |
| 3 | 自测报告 | 用例参数、结构与精度结果、性能数据（含峰值 workspace / numProds / nnz(C) / 输出存储量）、截图、Profiler 证据、CANN 版本、失败项说明 | 社区任务 IT 系统 |
| 4 | 待验收代码地址 | `ops-sparse` 个人代码仓链接、分支及代码目录，邀请账号 `Ascend-CANN` 作为开发者；含 README、接口说明及限制说明；统一交付 Python/torch 层接口、ATen NPU 适配、输出构造、SpGEMM 接口及 Kernel 能力补齐代码、C++ UT 和端到端 UT | `ops-sparse` `master` 分支 |

## 4.6 迭代规划

| 迭代 | 目标 | 关键产物 |
|------|------|----------|
| 迭代一 | 接口骨架 + fp32 单核正确性 | 描述符、多阶段流程、符号+数值 Kernel（T1）、CPU golden 对拍 |
| 迭代二 | 多核 + 负载均衡 + fp16/bf16 | 贪心装箱、device 前缀扫描、dtype 模板特化 |
| 迭代三 | **complex64 全链路** | 实虚双路数值 Kernel、描述符/校验/组装补齐、专项用例 |
| 迭代四 | Python/ATen 适配 | 双 layout 注册、输出构造、端到端 UT、无 fallback 证据 |
| 迭代五 | 泛化 + T2/T3 模板 + 性能调优 | 长尾/膨胀场景、200 精度用例全通过、50 性能用例达标 |
| 迭代六 | A2/A3 双硬件验证 + 报告 | A2/A3 精度结果、A3 性能结果、自测报告 |

---

## 附录 A：待确认事项（提请评审确认）

| 编号 | 事项 | 背景 | 建议 |
|------|------|------|------|
| Q1 | **性能验收口径**：任务书性能表 P-01~P-03（最大 1,048,576³，nnz(C)=67M，含全部四种 dtype）与 `aclsparseSpGemm_testCase/` 提供的 50 条 `[1000,20000]` 规模用例（仅 fp32/complex64），两者规模与 dtype 均不一致 | 影响达标判定与用例准备 | 建议任务书性能表 P-01~P-03 为主验收口径（任务书明确定义了倍率目标），50 条 testCase 作为精度+性能的补充验证 |
| Q2 | **精度用例 dtype 覆盖**：官方 200 条精度用例仅 fp32/complex64，但任务书要求四种 dtype 全链路 | fp16/bf16 无官方基线 | 确认 fp16/bf16 由开发者自建用例（S-01）验收 |
| Q3 | **SpGEMM 规划接口的落仓状态与时间**：需确认 `ops-sparse` `master` 上 `aclsparseSpGEMM*` 声明与 SpGEMM 目录的可用版本 | 决定复用基线 | 确认接口签名基线；若验收时规划代码尚未合入，明确本任务按任务书给定原型交付接口声明与实现 |
| Q4 | **A5 任务合入顺序** | 影响公共层冲突处理 | 建议双方在设计评审阶段对齐 `common/` 边界与接口签名 |
| Q5 | **`beta ≠ 0` 时的 matC 初值语义**：cuSPARSE 约定 C 与结果结构相同 | 影响接口完整性 | Python 路径固定 `beta=0`；C++ 路径按 cuSPARSE 语义实现并在文档明确 |

## 附录 B：关键设计决策速查

| 决策 | 选择 | 理由 |
|------|------|------|
| 结构确定方式 | **纯符号**（与数值无关） | 直接满足"显式零保留"约束 |
| 行内累加算法 | **k 路有序归并**（主路径） | 输出天然有序去重；UB 占用与 N 无关；契合 da/db ≤ 8 的实际负载 |
| 累加器备选 | 哈希（T2）/ 列分块稠密（T3） | 覆盖长尾行与高膨胀行，保证泛化 |
| 分核策略 | 按中间乘积数**贪心装箱** | 直接应对长尾行分布要求 |
| 跨核偏移 | 符号阶段全局前缀和 | 数值阶段各核直写最终位置，省去跨核 barrier |
| 累加精度 | 统一 fp32（complex64 双路） | 满足混合容差；累加链短，误差可控 |
| Host 同步 | 仅 `SpMatGetSize` 一次 D2H | SpGEMM 协议性必然，与 cuSPARSE 一致 |
| 目录结构 | `common/` + `arch22/` + `arch35/` | 满足 A2/A3 与 A5 同主干共存要求 |

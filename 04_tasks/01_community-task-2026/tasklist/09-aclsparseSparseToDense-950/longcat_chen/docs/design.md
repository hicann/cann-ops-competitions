# aclsparseSparseToDense 算子设计文档（Ascend 950 / A5）

| 项目 | 内容 |
|------|------|
| 任务名称 | 9月社区任务-aclsparseSparseToDense算子开发（950） |
| 参与者账号 | `longcat_chen`（CLA 邮箱：`longcat_eason@139.com`） |
| 目标硬件 | Ascend 950（arch35 / DAV_3510，A5） |
| CANN 版本 | CANN 9.1.0 |
| 交付代码仓 | [longcat_chen/ops-sparse](https://gitcode.com/longcat_chen/ops-sparse) 分支 `feature/aclsparse-sparsetodense-950` → 合入 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse) `master` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/09-aclsparseSparseToDense-950/longcat_chen/docs/design.md` |
| 对标接口 | cuSPARSE `cusparseSparseToDense_bufferSize` / `cusparseSparseToDense`；PyTorch `Tensor.to_dense` / `aten::_to_dense` |
| 文档版本 | v1.0 |

---

# 需求背景（required）

## 需求来源

本设计对应 2026 年 9 月社区任务《aclsparseSparseToDense 算子开发（950）》。任务要求在 Ascend 950 上使用 Ascend C 直调 + aclsparse C++ 接口，实现 **Sparse → Dense** 转换：将 CSR / CSC / COO 稀疏矩阵按坐标 scatter 写入 ROW/COL 稠密矩阵；并补齐测试、文档与（按任务书要求的）Python/ATen 适配，禁止 CPU fallback。

设计依据优先级：

1. 任务书规定的 950 能力、dtype、精度/性能/内存验收目标
2. `ops-sparse` 公开头文件 `include/cann_ops_sparse.h`
3. 仓内 `sparse/sparse2dense/arch35/` 实现与公共描述符语义
4. cuSPARSE Generic SparseToDense 两阶段调用约定
5. PyTorch `to_dense` / `_to_dense` schema（ATen 适配章节）

当任务书示例命名与公开头文件不一致时，以公开头文件为准：沿用已声明的 `aclsparseSparseToDense_bufferSize`，不新增同功能重复符号。

## 背景介绍

### 算子要做什么

数学语义（与 cuSPARSE 一致）：

$$
B_{ij}=\begin{cases}
A_{ij} & (i,j)\in\mathrm{nnz}(A)\\
0 & \text{otherwise}
\end{cases}
$$

- `matA`：只读稀疏矩阵（CSR / CSC / COO）
- `matB`：可写稠密矩阵（ROW 或 COL，允许 `ld` padding）
- **纯数据搬运**：`B[idx] = A.values[p]`，无算术、无类型转换 → **bit-exact**
- 两阶段 API：`bufferSize`（当前恒为 0）→ `SparseToDense` 执行
- 执行前对 `matB` 存储区 `aclrtMemset` 清零；`nnz=0` 或空矩阵仅清零，不 launch kernel

### 与 DenseToSparse / Scatter 的关系

| 算子 | 方向 | 本任务角色 |
|------|------|------------|
| SparseToDense | 稀疏 → 稠密 | 本设计 |
| DenseToSparse | 稠密 → 稀疏 | 对偶转换（三阶段 Analysis/Convert） |
| Scatter | 稀疏向量 → 稠密向量 | 一维同类 scatter；本算子为二维矩阵版 |

950 路径位于 `sparse/sparse2dense/arch35/`，与 A2/A3 的 `arch22`（若另有任务）分目录共存，按 SOC 分流编译。

### 现状与增量

仓内 arch35 已具备 Host 校验 + SIMT kernel + CSV ST。本任务交付聚焦：

| 缺口 / 交付项 | 方案 |
|---------------|------|
| 950 验收对齐 | 固化 CSR/CSC/COO × 五 dtype × ROW/COL × base0/1 精度路径 |
| 文档 | `sparse/sparse2dense/README.md` + 本设计文档 |
| 任务包（若下发） | ATK 精度 / 性能 P 场景 / 内存对比接入 `test/sparse2dense/task_cases/` |
| Python/ATen（若任务书要求） | `python/npu_sparse_sparsetodense/` 注册 `_to_dense`，无 CPU fallback |

当前 C++ 路径支持 value dtype：FP32 / FP16 / BF16 / INT32 / INT8；**不支持** FP64 / complex64（返回 `NOT_SUPPORTED`）。若任务书强制 complex64，则按 uint64_t 8B 位拷贝扩展 Host/Kernel 分派（与 Scatter 950 同策略）。

### PyTorch 映射

| 公开入口 | ATen | aclsparse |
|----------|------|-----------|
| `Tensor.to_dense(...)` | `aten::_to_dense` | `SparseToDense_bufferSize` + `SparseToDense` |
| 任务包 hook（规划） | `ops_sparse_test.sparse_to_dense_npu` | 同上 |

约束：二维稀疏（COO/CSR/CSC）；索引 I32、base 0/1；输出新建 Dense Tensor；**禁止 CPU fallback**。

---

# 需求分析（required）

## 需求描述

在 Ascend 950 上交付完整 SparseToDense：

1. **C++ API**：`aclsparseSparseToDense_bufferSize` / `aclsparseSparseToDense`
2. **Kernel**：arch35 SIMT scatter，CSR/CSC 按行/列切分，COO 按 nnz grid-stride
3. **测试**：CSV 精度 ST + 负向 L2；任务包 ATK/性能/内存（以下发 zip 为准）
4. **ATen/Python**（任务书要求时）：NPU 注册 `_to_dense` / `to_dense`
5. **文档**：README 接口说明 + 本设计

## 需求拆解

| 编号 | 子项 | 验收要点 |
|------|------|----------|
| R1 | 公开 API | 与 `cann_ops_sparse.h` 签名一致；仅 `ALG_DEFAULT` |
| R2 | 格式 | CSR / CSC / COO；索引仅 I32；base 0/1 |
| R3 | dtype | FP32/FP16/BF16/INT32/INT8；A/B 同型；bit-exact |
| R4 | 布局 | ROW/COL；`ld` 合法（ROW≥n，COL≥m） |
| R5 | workspace | `bufferSize≡0`；`buffer` 可空 |
| R6 | 边界 | 空矩阵 / nnz=0 / ld padding / 宽高矩阵 |
| R7 | 确定性 | 无重复坐标时 bit-wise 可复现；COO 重复坐标 last-write-wins（并行下不确定，对齐 cuSPARSE） |
| R8 | PTA | `_to_dense` NPU 路径无 CPU fallback（任务书要求时） |
| R9 | 精度/性能 | CSV ST 全绿；任务书 P 场景 ≥ 规定倍率（以下发为准） |

## 输入输出规格

| 参数 | 方向 | 类型 | 说明 |
|------|------|------|------|
| handle | 输入 | aclsparseHandle_t | 携带 stream |
| matA | 输入 | aclsparseConstSpMatDescr_t | 稀疏源，只读 |
| matB | 输入/输出 | aclsparseDnMatDescr_t | 稠密目标，执行时写入 |
| alg | 输入 | aclsparseSparseToDenseAlg_t | 仅 DEFAULT |
| bufferSize | 输出 | size_t* | 恒为 0 |
| buffer | 输入 | void* | 可为 nullptr |

| value dtype | index | base | order | 950 |
|-------------|-------|------|-------|-----|
| FP32/FP16/BF16/INT32/INT8 | I32 | 0/1 | ROW/COL | ✅ |

约束：`m,n ≤ INT32_MAX`；COO 切分时 `nnz ≤ INT32_MAX`；`nnz>0` 时 ptrs/idxs/values 非空；indices 合法范围由调用方保证（Host 不读 Device 内容做越界检查）。

---

# 详细设计（required）

## 算子分析

### 格式 → 坐标

| 格式 | matA 字段 | 坐标恢复 |
|------|-----------|----------|
| CSR | `ptrs`=rowOffsets(m+1), `idxs`=colInd(nnz) | 行 `r`，`p∈[off[r],off[r+1])`，`c=colInd[p]-base` |
| CSC | `ptrs`=colOffsets(n+1), `idxs`=rowInd(nnz) | 列 `c`，`p∈[off[c],off[c+1])`，`r=rowInd[p]-base` |
| COO | `ptrs`=cooRowInd(nnz), `idxs`=cooColInd(nnz) | `r=row[p]-base`，`c=col[p]-base` |

Dense 物理偏移：

$$
\mathrm{offset}(r,c)=\begin{cases}
r\cdot ld+c & \mathrm{ROW}\\
c\cdot ld+r & \mathrm{COL}
\end{cases}
$$

### 计算流程

```text
Validate(handle / format / dtype / ld / dims / alg)
  → bufferSize: 返回 0
  → Execute:
      ZeroDense(matB) via aclrtMemset
      if m==0 or n==0 or nnz==0: return SUCCESS
      Tiling(splitDim → useBlocks, perBlock)
      sparse2dense_kernel_do(ptrs, idxs, values, dnMat, tiling, stream)
```

切分维度：CSR→`m`，CSC→`n`，COO→`nnz`。  
`useBlocks = min(AIV核数, ceil(dim / MaxThreadsPerBlock))`，`perBlock = ceil(dim / useBlocks)`。

### 编程模型

arch35 **AIV SIMT**（`KERNEL_TYPE_AIV_ONLY` + `asc_vf_call`）：

- CSR/CSC：不同行/列互不写冲突，线程安全
- COO：重复 `(r,c)` 多写者 last-write-wins，写序不确定（对齐 cuSPARSE 语义）
- 每元素一次 GM 写，无归约、无原子

## 算子实现

### Host（`sparse2dense_host.cpp`）

1. **校验**：handle/描述符非空；format∈{CSR,CSC,COO}；ptr/idx 均为 I32；base∈{0,1}；valueType∈{FP32,FP16,BF16,INT32,INT8}；A/B dtype 一致；dims 一致；ld 合法；alg=DEFAULT
2. **bufferSize**：写 `*bufferSize=0`
3. **Execute**：计算 `dnMatBytes`（ROW:`m*ld`，COL:`n*ld`）× elemSize → memset → tiling → launch
4. **Quick path**：空维 / nnz=0 仅清零

### Kernel（`sparse2dense_kernel.cpp`）

三层结构：

1. `__simt_vf__`：`Sparse2DenseSimtComputeCsrCsc` / `Sparse2DenseSimtComputeCoo`
2. `__global__` Dispatcher：读 tiling → `asc_vf_call`
3. `sparse2dense_kernel_do`：`<<<useBlocks>>>` 异步 launch

dtype 模板：`float` / `half` / `bfloat16_t` / `int32_t` / `int8_t`。

关键常量（`sparse2dense_tiling_data.h`）：

| 常量 | 值 | 含义 |
|------|-----|------|
| `kSparse2DenseMaxThreadsPerBlock` | 128 | 单核 SIMT 线程上限 |
| `kSparse2DenseWarpSize` | 32 | 线程数向上对齐 |
| `SPARSE2DENSE_VAL_*` | 0..4 | dtype 编码 |
| `SPARSE2DENSE_FMT_*` | 0..2 | CSR/CSC/COO |

### ATen / Python（规划，`python/npu_sparse_sparsetodense/`）

```text
_to_dense(self, dtype=None, masked_grad=None)
  → 校验 2D / NPU / layout∈{coo,csr,csc} / 支持 dtype
  → 构造 ConstSpMat + DnMat
  → SparseToDense_bufferSize + SparseToDense
  → 返回新建 Dense Tensor（默认 ROW contiguous）
```

- 注册：`torch.library` PrivateUse1（`aten::_to_dense` + 任务 hook）
- Handle 线程缓存；索引统一 I32
- UT：pytest 覆盖 dtype × layout × base

## 支持硬件

| 芯片 | 勾选 |
|------|------|
| Ascend 950 / 950PR / 950DT | ✅ 本任务验收 |
| Atlas A2/A3 | —（另见 A2/A3 任务设计，不在本 MR 范围） |

## 算子约束限制

- 仅 CSR/CSC/COO；Blocked-ELL 等不在本算子范围
- 索引仅 I32；I64 → `NOT_SUPPORTED`
- FP64 / complex64（当前）→ `NOT_SUPPORTED`；complex64 若任务书强制则后续按 8B 拷贝扩展
- indices 越界：未定义行为（与 cuSPARSE 一致，Host 不做 Device 扫描）
- COO 重复坐标：结果不确定
- 无 Device workspace；不得额外分配与 nnz 线性相关的临时 GM
- `m`/`n`（及 COO `nnz`）超过 `INT32_MAX` → `NOT_SUPPORTED`

---

# 可维可测分析（required）

## 精度 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度 ST | CSV bit-exact（golden 同 dtype 搬运） | 仓内 GTest |
| 负向 L2 | 空指针 / 维度 / dtype / ld / alg / 不支持类型 | `sparse2dense_l2_cases.csv` |
| 精度 ATK | 任务包泛化 case（若下发） | 任务 zip |
| 性能 | GPU/NPU 倍率 ≥ 任务书阈值（若下发 P 场景） | 任务书 §性能 |
| 内存 | IO 或 workspace 规则（若下发） | 任务书 §内存 |
| PTA | pytest / Profiler 无 CPU fallback | 仓内 |

当前仓内基线（arch35 README）：**50** 条 GTest（32 正确性 + 16 负向 + 2 跳过），Ascend 950 真机通过。

## 测试设计

| 类别 | 内容 | 位置 |
|------|------|------|
| 精度 ST | CSV 参数化 Sparse2DenseCases | `test/sparse2dense/arch35/sparse2dense_test.csv` |
| 负向 | L2 CSV + TEST_F | `sparse2dense_l2_cases.csv` / `sparse2dense_test.cpp` |
| Golden | CSR/CSC/COO CPU 参考 | `sparse2dense_golden.h` |
| 任务包 | ATK / perf / memory（下发后接入） | `test/sparse2dense/task_cases/` |
| 950 上传 | 一键脚本 → `logs/sparsetodense-950` | 私仓 `scripts/ci/` |

L0/L1 覆盖：三格式 × 五 dtype × ROW/COL × base0/1；空/单元素/全密/ld padding；三格式一致性（同矩阵输出一致）。

## 兼容性分析

- 与公开 API / 描述符体系兼容；不新增重复符号
- arch35 与未来 arch22 分目录；`build.sh --soc=ascend950 --ops=sparse2dense` 仅编 950
- 与 DenseToSparse / Scatter / Gather 共用 handle 与 SpMat/DnMat，无接口冲突

## 风险与对策

| 风险 | 对策 |
|------|------|
| COO 重复坐标不确定 | 文档声明对齐 cuSPARSE；任务用例保证唯一坐标 |
| 大矩阵 memset + scatter 带宽 | 多核切分 + 对标 GPU median；必要时调 `MaxThreads`/`useBlocks` |
| PTA layout 多样 | 入口统一转 I32 描述符；仅支持任务书布局 |
| 与 A2/A3 设计并行 | 本账号目录 `longcat_chen/`；仅提交 950 路径设计 |

---

# 代码目录（ops-sparse）

```text
sparse/sparse2dense/
├── README.md
└── arch35/
    ├── sparse2dense.h
    ├── sparse2dense_host.cpp
    ├── sparse2dense_kernel.cpp
    ├── sparse2dense_kernel.h
    └── sparse2dense_tiling_data.h

test/sparse2dense/
├── sparse2dense_param.h
├── sparse2dense_golden.h
└── arch35/
    ├── sparse2dense_test.cpp
    ├── sparse2dense_test.csv
    ├── sparse2dense_l2_cases.csv
    ├── sparse2dense_npu_wrapper.h
    └── sparse2dense_npu_execution.h

python/npu_sparse_sparsetodense/   # 规划
├── ...
└── tests/
```

---

# 交付物

1. 本设计文档（本 PR）
2. 代码：`longcat_chen/ops-sparse` @ `feature/aclsparse-sparsetodense-950`
3. 自测报告（腾讯文档模板）+ 950 日志分支证据（验收阶段）
4. 邀请 Ascend-CANN 为开发者（代码仓）

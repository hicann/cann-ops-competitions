# aclsparseGather 算子设计文档（Ascend 950 / A5）

| 项目 | 内容 |
|------|------|
| 任务名称 | 9月社区任务-aclsparseGather算子开发（950） |
| 参与者账号 | `baibai_Uu` |
| 目标硬件 | Ascend 950（arch35 / DAV_3510，A5） |
| CANN 版本 | CANN 9.1.0 |
| 交付代码仓 | [baibai_Uu/ops-sparse](https://gitcode.com/baibai_Uu/ops-sparse) 分支 `feature/aclsparse-gather-950` → 合入 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse) `master` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/09-aclsparseGather-950/baibai_Uu/docs/design.md` |
| 对标接口 | cuSPARSE `cusparseGather`；PyTorch `torch.index_select(input, 0, index)` |
| 文档版本 | v1.0 |
| 950 验证 commit | `a277bbf`（日志分支 `logs/gather-950`，stamp `20260902_022724`） |

---

# 需求背景（required）

## 需求来源

本设计对应 2026 年 9 月社区任务《aclsparseGather 算子开发（950）》。任务要求在 Ascend 950 上使用 Ascend C 直调 + aclsparse C++ 接口，实现 **Gather** 语义：从稠密向量 `Y` 按稀疏向量 `X` 的 indices 读取元素，**原地**写回 `X.values`；并补齐 **Python/ATen** 适配（`torch.index_select`，`dim=0` 一维场景）。

## 背景介绍

### 算子要做什么

数学表达式（与 cuSPARSE Gather 一致）：

$$
X.\text{values}[i] = Y[X.\text{indices}[i] - \text{idxBase}],\quad i \in [0, \text{nnz})
$$

- `vecY`：稠密源向量（只读 values）
- `vecX`：稀疏向量（读 indices + idxBase，写 values）
- **不修改** `vecY` 与 `vecX.indices`
- 无 workspace、无 preprocess，单步异步 kernel

### PyTorch 映射

| 公开入口 | ATen | aclsparse |
|----------|------|-----------|
| `torch.index_select(input, 0, index)` | `aten::index_select` | `aclsparseGather(handle, vecY, vecX)` |

约束：`dim=0`、一维连续向量；indices 转为 SpVec 描述符（I32、base 0/1）；输出构造与 alias 语义对齐 PyTorch；**禁止 CPU fallback**。

### 与仓内算子关系

Gather 属于 ops-sparse **Generic API** 稀疏向量运算，与 SpMV/SpGEMM 等矩阵算子独立；950 路径位于 `sparse/gather/arch35/`，与 A2/A3 arch22 路径可共存。

---

# 需求分析（required）

## 需求描述

在 Ascend 950 上完成：

1. **C++ API**：`aclsparseGather(handle, vecY, vecX)`
2. **Kernel**：arch35 SIMT gather，支持 FP16/BF16/FP32/complex64 + I32 索引 + base 0/1
3. **测试**：1000 精度 + 424 性能 CSV GTest；任务书 P-01/P-02/P-03 共 24 场景
4. **ATen/Python**：NPU 注册 `index_select`（dim=0），端到端 UT（后续 PR 交付）
5. **文档**：`sparse/gather/README.md` 接口说明

## 需求拆解

| 编号 | 子项 | 验收要点 | 950 状态 |
|------|------|----------|----------|
| R1 | 公开 API | `include/cann_ops_sparse.h` 声明一致 | ✅ |
| R2 | 功能正确 | 1000 bit-wise ST PASS | ✅ |
| R3 | dtype | FP16/BF16/FP32/C64 + I32 base0/1 | ✅ |
| R4 | 性能 | P-01/P-02/P-03：GPU/NPU ≥ 0.3 | ✅ 24/24 |
| R5 | 泛化 perf | 400 条 + 回归门禁 | ✅ 424/424 |
| R6 | ATen | `index_select` NPU 路径 | 🚧 设计中/开发中 |
| R7 | 内存 | P 场景 GPU/NPU 对比脚本 | 🚧 待跑 |

## 输入输出规格

| 参数 | 方向 | 类型 | 说明 |
|------|------|------|------|
| handle | 输入 | aclsparseHandle_t | 携带 stream |
| vecY | 输入 | aclsparseConstDnVecDescr_t | 稠密源，shape `[size]` |
| vecX | 输入输出 | aclsparseSpVecDescr_t | indices/values `[nnz]`，原地写 values |

| value dtype | index | base | 950 |
|-------------|-------|------|-----|
| FP16/BF16/FP32/complex64 | I32 | 0/1 | ✅ |
| FP64 | I32/I64 | 0/1 | API 预留，950 ST 未覆盖 |

约束：`nnz=0` 成功返回且不启 kernel；`vecY.size ≥ vecX.size`；indices 合法范围由调用方保证（kernel 不做越界检查）。

---

# 详细设计（required）

## 算子分析

### 数学公式

见上文；实现为 **间接寻址 gather**：每个输出元素独立，无归约，访存模式由 indices 分布决定（顺序/乱序/重复均合法）。

### 支持数据类型与硬件

| 芯片 | 目录 | 本任务 |
|------|------|--------|
| Ascend 950 / 950PR / 950DT | `sparse/gather/arch35/` | ✅ 实现与验收 |
| Atlas A2/A3 | arch22（若已有） | 不扩展 |

## 算子实现

### 总体方案

```text
aclsparseGather
  → Host 参数校验（dtype/base/shape/null）
  → GatherTilingData（nnz、numBlocks、threadsPerBlock）
  → gather_kernel_do()  dtype × idxType × idxBase 分派
  → SIMT VF：grid-stride + 4-way ILP gather
  → 异步写回 vecX.values
```

### Host 侧设计（`gather_host.cpp`）

1. **校验**：handle/描述符非空；valueType 一致；supportedType；`vecY.nums ≥ vecX.size`
2. **Quick path**：`nnz == 0` 直接返回 SUCCESS
3. **Tiling**：
   - `threadsPerBlock`：nnz < 8192 → 256，否则 512（降低小 nnz launch 开销）
   - `numBlocks = min(AIV核数, ceil(nnz / threadsPerBlock))`
4. **Launch**：`gather_kernel_do(indices, yValues, xValues, tiling, stream)`

### Kernel 侧设计（`gather_kernel.cpp`）

- **路径**：AIV SIMT（`KERNEL_TYPE_AIV_ONLY`），非 Vector 大 tile（gather 为随机访存，SIMT 更适配）
- **核心循环**：grid-stride，每线程 unroll=4 次独立 `yValues[indices[i]-base]`
- **模板分派**：`(ValT, IdxT, idxBase)` 编译期实例化；base0/base1 分支 DCE
- **complex64**：按 8B 元素拷贝（与 FP32 路径相同索引逻辑）

关键常量（`gather_tiling_data.h`）：

| 常量 | 值 | 含义 |
|------|-----|------|
| kGatherMaxThreadsPerBlock | 512 | 大 nnz 线程数 |
| kGatherSmallThreadsPerBlock | 256 | 小 nnz 线程数 |
| kGatherSmallNnzThreshold | 8192 | 自适应阈值 |
| kGatherStrideUnroll | 4 | ILP 展开 |

### ATen / Python 适配（设计，合入 ops-sparse 同一 PR 系列）

```text
torch.index_select(self, 0, index)
  → 校验：self/index 同 device、1D、dim==0、支持 dtype
  → 构造 DnVec(vecY=self) + SpVec(vecX: indices=index, values=out 或 in-place alias)
  → aclsparseGather(handle, vecY, vecX)
  → 返回 tensor（与 PyTorch shape/dtype 一致）
```

- 注册：`torch_npu` Dispatcher  PRIVATEUSE1 后端
- 异常：不支持组合返回明确错误，不 silent CPU fallback
- UT：Python + C++ ATen case，Profiler 证明 NPU dispatch

## 支持硬件

| 芯片 | 勾选 |
|------|------|
| Ascend 950 / 950PR / 950DT | ✅ |
| Atlas A2/A3 | — |

## 算子约束限制

- 仅 **一维** gather；高维 `index_select` 其他 dim 不在本任务范围
- indices 越界：未定义行为（与 cuSPARSE 调用约定一致）
- 无 workspace；不得额外分配与 nnz 线性相关的临时 GM
- `vecY` 与 `vecX.values` 重叠：参数错误

---

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 | 950 实测 |
|----------|------|----------|----------|
| 精度 | CPU Golden bit-exact；C64 实虚部分别 exact | 任务书 §3.2 + experimental_standard | **1000/1000 PASS** |
| 性能 | 倍率 = GPU median / NPU median ≥ **0.3** | 任务书 §3.3 P-01~P-03 | **24/24 PASS**，最低 **0.445** |
| 内存 | >500MB 时 NPU 额外内存 ≤ GPU 50%；或 workspace ≤ L2 | 任务书 §3.4 | 待脚本验收 |

## P-01 / P-02 / P-03 950 实测摘要（`a277bbf`）

| 场景 | vec_size | nnz | NPU median（μs）典型 | GPU 标杆（μs） | 倍率范围 |
|------|----------|-----|------------------------|----------------|----------|
| P-01 Llama | 128256 | 8192 | FP16 ~77；C64 ~135 | 60~112 | 0.45~1.44 |
| P-02 Qwen | 151936 | 4096 | FP16 ~77；C64 ~138 | 70~82 | 0.54~1.06 |
| P-03 DeepSeek | 129280 | 7168 | FP16 ~76；C64 ~131 | 68~75 | 0.56~0.99 |

泛化 perf：**424/424 PASS**；original cohort 160 条有基线，**0 条相对+绝对退化**。

## 测试设计

| 类别 | 内容 | 位置 |
|------|------|------|
| 精度 ST | 1000 case，random/dup/reversed，base0/1 | `test/gather/arch35/gather_test.csv` |
| 性能 ST | 400 泛化 + 24 任务 P case | `gather_perf.csv` + `gather_task_p_cases.csv` |
| 任务包脚本 | ATK/内存/GPU 标杆 | 任务包 `test_cases/` |
| 950 上传 | `scripts/ci/run_gather_950_upload_logs.sh` → `logs/gather-950` | 私仓 |

## 兼容性分析

- 新增 950 arch35 能力；与现有 aclsparse 描述符体系兼容
- A2/A3 若后续合入 arch22 Gather，Host 公共逻辑复用，Kernel 分目录

---

# 代码目录（ops-sparse）

```text
sparse/gather/arch35/
├── gather_host.cpp          # aclsparseGather + tiling + launch
├── gather_kernel.cpp        # SIMT gather_kernel_do
├── gather_kernel.h
└── gather_tiling_data.h

test/gather/
├── gather_golden.h
├── gather_param.h
└── arch35/
    ├── gather_test.cpp
    ├── gather_test.csv
    ├── gather_perf.csv
    └── gather_npu_wrapper.h

sparse/gather/README.md
include/cann_ops_sparse.h    # 公开声明
```

---

# 开发过程摘要

1. 参考 cuSPARSE Gather 语义与任务书 A5 要求，在 ops-sparse 新增 arch35 路径
2. 精度：1000 CSV bit-wise；补齐 complex64
3. 性能：SIMT + grid-stride ILP + 小 nnz 自适应线程；泛化 400 case
4. 任务 P case：嵌入 GPU median 门禁（GTest 名须 alphanumeric，禁用 `-`）
5. 950 私仓 `baibai_Uu/ops-sparse` + git 日志分支闭环验证

---

# 参考

- 任务书：`aclsparseGather_A5_task_doc.md`
- cuSPARSE Gather：https://docs.nvidia.com/cuda/cusparse/index.html
- ops-sparse README：`sparse/gather/README.md`
- 社区模板：`resources/design_template.md`

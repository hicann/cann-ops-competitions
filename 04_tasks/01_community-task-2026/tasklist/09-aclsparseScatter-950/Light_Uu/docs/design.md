# aclsparseScatter 算子设计文档（Ascend 950 / A5）

| 项目 | 内容 |
|------|------|
| 任务名称 | 9月社区任务-aclsparseScatter算子开发（950） |
| 参与者账号 | `Light_Uu` |
| 目标硬件 | Ascend 950（arch35 / DAV_3510，A5） |
| CANN 版本 | CANN 9.1.0 |
| 交付代码仓 | [only_test/ops-sparse](https://gitcode.com/only_test/ops-sparse) 分支 `feature/aclsparse-scatter-950` → 合入 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse) `master` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/09-aclsparseScatter-950/Light_Uu/docs/design.md` |
| 对标接口 | cuSPARSE `cusparseScatter`；PyTorch `aten::index_copy_`（`dim=0` 一维） |
| 文档版本 | v1.1 |
| 950 验证 commit | `8885dd1`（日志分支 `logs/scatter-950`，stamp `20260902_062545`，`all_green=true`） |

---

# 需求背景（required）

## 需求来源

本设计对应 2026 年 9 月社区任务《aclsparseScatter 算子开发（950）》。任务要求在 Ascend 950 上使用 Ascend C 直调 + aclsparse C++ 接口，实现 **Scatter** 语义：将稀疏向量 `X` 的非零值按 indices 写入稠密向量 `Y`（**原地**更新 `Y.values`）；并补齐 **Python/ATen** 适配（`tensor.index_copy_(0, index, source)`，一维 `dim=0`）。

## 背景介绍

### 算子要做什么

数学表达式（与 cuSPARSE Scatter 一致）：

$$
Y[X.\text{indices}[i] - \text{idxBase}] = X.\text{values}[i],\quad i \in [0, \text{nnz})
$$

- `vecX`：稀疏源向量（只读 indices / values / idxBase）
- `vecY`：稠密目标向量（**原地**写 values）
- **不修改** `vecX`；`Y` 未被命中的位置保持原值
- 无 workspace、无 preprocess，单步异步 kernel
- 纯数据搬运，**bitwise exact**（无浮点运算）

### 关键语义（与 cuSPARSE 对齐）

| 项 | 约定 |
|----|------|
| idxBase | ONE 时内部 `indices[i]-1`；ZERO 时直接用作偏移 |
| 索引排序 | 允许未排序 |
| 索引重复 | last-write-wins，多线程下行为不确定 |
| 索引越界 | Host 不读 Device，不做运行时边界检查；调用方保证合法 |
| nnz=0 | 成功返回，不启 kernel |

### PyTorch 映射

| 公开入口 | ATen | aclsparse |
|----------|------|-----------|
| `tensor.index_copy_(0, index, source)` | `aten::index_copy_` | `aclsparseScatter(handle, vecX, vecY)` |
| 任务包 hook | `torch.ops.ops_sparse_test.scatter_npu` | 同上（零初始化 Y 后 scatter） |

约束：`dim=0`、一维连续向量；indices 为 I32、base 0/1；**禁止 CPU fallback**。

### 与仓内算子关系

Scatter 属于 ops-sparse **Generic API** 稀疏向量运算，与 Gather 互为对偶（写 vs 读）；950 路径位于 `sparse/scatter/arch35/`，与已有 A2/A3 `arch22` 路径共存、按 SOC 分流编译。

相对 Gather，本任务额外要求：

- value dtype 含 **INT8** 与 **complex64**（A5）
- 任务书性能/内存场景含 int8（P 场景 30 条，非 Gather 的 24 条）

---

# 需求分析（required）

## 需求描述

在 Ascend 950 上完成：

1. **C++ API**：`aclsparseScatter(handle, vecX, vecY)`
2. **Kernel**：arch35 SIMT scatter，支持 FP16/BF16/FP32/INT8/complex64 + I32/I64 索引 + base 0/1
3. **测试**：CSV 精度 ST + 任务书 ATK 200 精度 + P-01/P-02/P-03 共 30 场景性能/内存
4. **ATen/Python**：NPU 注册 `index_copy_`（dim=0）与 `scatter_npu` hook
5. **文档**：`sparse/scatter/README.md` 接口说明

## 需求拆解

| 编号 | 子项 | 验收要点 | 950 状态（`8885dd1`） |
|------|------|----------|----------------------|
| R1 | 公开 API | `include/cann_ops_sparse.h` 声明一致 | ✅ |
| R2 | 功能正确 | CSV ST **400/400**；ATK 200/200；ACC_FB 200/200 | ✅ |
| R3 | dtype | FP16/BF16/FP32/INT8/C64 + I32/I64 + base0/1 | ✅ |
| R4 | 性能 | P-01~P-03：GPU/NPU ≥ 0.3（30 条） | ✅ **30/30 PASS**（含 complex64） |
| R5 | 内存 | P + 泛化 + IO>500MB（244 条） | ✅ **244/244**；14×大 IO `workspace_l2` |
| R6 | ATen | `index_copy_` + `scatter_npu` | ✅ pytest **14/14** |
| R7 | 打包 | `.run` 安装 libops_sparse | ✅ |

## 输入输出规格

| 参数 | 方向 | 类型 | 说明 |
|------|------|------|------|
| handle | 输入 | aclsparseHandle_t | 携带 stream |
| vecX | 输入 | aclsparseConstSpVecDescr_t | 稀疏源，indices/values `[nnz]` |
| vecY | 输入输出 | aclsparseDnVecDescr_t | 稠密目标，shape `[nums]`，原地写 |

| value dtype | index | base | 950 |
|-------------|-------|------|-----|
| FP16/BF16/FP32/INT8/complex64 | I32/I64 | 0/1 | ✅ |

约束：`nnz=0` 成功且不启 kernel；`vecX.size ≤ vecY.nums`；`nnz ≤ size`；`nnz ≤ UINT32_MAX`；indices 合法范围由调用方保证。

---

# 详细设计（required）

## 算子分析

### 数学公式

见上文；实现为 **间接寻址 scatter**：每个非零元素独立写 `Y[idx]`，无归约。访存为随机写，模式由 indices 分布决定（顺序/乱序/重复均合法；重复时结果不确定）。

### 支持数据类型与硬件

| 芯片 | 目录 | 本任务 |
|------|------|--------|
| Ascend 950 / 950PR / 950DT | `sparse/scatter/arch35/` | ✅ 实现与验收 |
| Atlas A2/A3 | `sparse/scatter/arch22/` | 已有基础路径，本任务不扩展 |

## 算子实现

### 总体方案

```text
aclsparseScatter
  → Host 参数校验（dtype/base/shape/null/stream）
  → ScatterTilingData（nnz、numBlocks、threadsPerBlock、valType、idxType、idxBase）
  → scatter_kernel_do()  按 idxType × valType 分派
  → SIMT VF：grid-stride + 4-way ILP scatter 写
  → 异步写回 vecY.values
```

### Host 侧设计（`scatter_host.cpp`）

1. **校验**：handle/描述符非空；valueType ∈ {FP32,FP16,BF16,INT8,C64}；idxType ∈ {I32,I64}；idxBase ∈ {0,1}；`size ≤ nums`；`nnz ≤ size`；`nnz>0` 时指针非空；stream 已绑定
2. **Quick path**：`nnz == 0` 直接返回 SUCCESS
3. **Tiling**（对齐 Gather 调优经验）：
   - `threadsPerBlock`：nnz < 4096 → 128，否则 **512**
   - `numBlocks = min(AIV核数, ceil(nnz / threadsPerBlock))`
4. **Launch**：`scatter_kernel_do(indices, values, yVec, tiling, numBlocks, stream)`

### Kernel 侧设计（`scatter_kernel.cpp`）

- **路径**：AIV SIMT（`KERNEL_TYPE_AIV_ONLY`）；随机写场景 SIMT 优于大 tile Vector
- **核心循环**：grid-stride，每线程 **unroll=4** 次独立 `yVec[indices[i]-idxBase] = values[i]`
- **dtype 分派**：FP32 / half / bfloat16_t / int8_t；**complex64 按 uint64_t 8B 拷贝**
- **idxBase**：运行时标量减法（0/1），避免额外模板爆炸

关键常量（`scatter_tiling_data.h`）：

| 常量 | 值 | 含义 |
|------|-----|------|
| kScatterMaxThreadsPerBlock | 512 | 大 nnz 线程数 |
| kScatterSmallThreadsPerBlock | 128 | 小 nnz 线程数 |
| kScatterSmallNnzThreshold | 4096 | 自适应阈值 |
| kScatterStrideUnroll | 4 | ILP 展开 |

### ATen / Python 适配（`python/npu_sparse_scatter/`）

```text
scatter_npu(values, indices, size, base)
  → 零初始化 Y[size]（complex64：float32[size*2].view(complex64) 快路径）
  → ConstSpVec + DnVec + aclsparseScatter
  → 返回 dense Y

aten::index_copy_(self, 0, index, source)
  → 校验 1D / dim==0 / 同 device
  → 原地 scatter 写入 self
```

- 注册：`torch.library` PrivateUse1（`ops_sparse_test.scatter_npu` + `aten::index_copy_`）
- Handle 线程缓存，降低重复 create/destroy
- UT：`python/npu_sparse_scatter/tests/test_scatter_npu.py`

## 支持硬件

| 芯片 | 勾选 |
|------|------|
| Ascend 950 / 950PR / 950DT | ✅ |
| Atlas A2/A3 | —（本任务不验收） |

## 算子约束限制

- 仅 **一维** scatter；高维 `index_copy_` 其他 dim 不在本任务范围
- indices 越界：未定义行为（与 cuSPARSE 一致）
- 重复 indices：结果不确定
- 无 workspace；不得额外分配与 nnz 线性相关的临时 GM
- `nnz > UINT32_MAX`：返回 `NOT_SUPPORTED`

---

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 | 950 实测（`8885dd1` / `20260902_062545`） |
|----------|------|----------|------------------------------------------|
| 精度 ST | CSV bit-exact | 仓内 GTest | **400/400 PASS** |
| 精度 ATK | 200 泛化 case | 任务包 ATK | **200/200 PASS** |
| 精度 Fallback | 同 200 case，无 ATK schema | NPU fallback | **200/200 PASS** |
| 性能 | 倍率 = GPU median / NPU median ≥ **0.3** | 任务书 §3.3 P-01~P-03（30 条） | **30/30 PASS**（含 complex64，ratio≈0.56~0.61） |
| 内存 | NPU vs GPU / workspace≤L2 | 任务书 §3.4 | **244/244 PASS**（含 14×IO>500MB） |
| PTA | pytest | 仓内 | **14/14 PASS** |

## P-01 / P-02 / P-03 场景定义

| 场景 | vec_size | nnz | 说明 |
|------|----------|-----|------|
| P-01 | 128256 | 8192 | Llama 类词表规模 |
| P-02 | 151936 | 4096 | Qwen 类 |
| P-03 | 129280 | 7168 | DeepSeek 类 |

每场景 × {int8, fp16, bf16, fp32, complex64} × base{0,1} = **30** 条。

## 测试设计

| 类别 | 内容 | 位置 |
|------|------|------|
| 精度 ST | CSV 参数化 ScatterCases | `test/scatter/arch35/scatter_test.csv` |
| 异常/白盒 | TEST_F 空指针、dtype、oob 信息项等 | `scatter_test.cpp` |
| 任务包 | ATK 200 + 内存 + Python P 场景 | `test/scatter/task_cases/` |
| 950 上传 | `scripts/ci/run_scatter_950_upload_logs.sh` → `logs/scatter-950` | 私仓 |

## 兼容性分析

- 扩展 arch35 的 INT8/complex64，与既有 FP32/FP16/BF16 Host/Kernel 分派兼容
- 与 A2/A3 arch22 分目录；`build.sh --soc=ascend950 --ops=scatter` 仅编 arch35
- 与 Gather 共享描述符 / handle 体系，无接口冲突

---

# 代码目录（ops-sparse）

```text
sparse/scatter/arch35/
├── scatter_host.cpp         # aclsparseScatter + tiling + launch
├── scatter_kernel.cpp       # SIMT scatter_kernel_do
├── scatter_kernel.h
├── scatter_tiling_data.h
└── scatter.h

test/scatter/
├── scatter_golden.h
├── scatter_param.h
├── arch35/
│   ├── scatter_test.cpp
│   ├── scatter_test.csv
│   └── scatter_npu_wrapper.h
└── task_cases/              # 任务书验收脚本与基线

python/npu_sparse_scatter/
├── scatter_npu.py           # scatter_npu + index_copy_
├── acl_binding.py
└── tests/

sparse/scatter/README.md
include/cann_ops_sparse.h    # 公开声明
scripts/ci/run_scatter_950_upload_logs.sh
```

---

# 开发过程摘要

1. 在 ops-sparse 既有 Scatter arch35 路径上补齐 **INT8 + complex64** 分派与校验
2. 精度：CSV ST + 任务包 ATK 200（适配 ATK 26.7.8：`outputs` 字符串、`api_type=sparse_scatter_public`、`--single_process`）
3. 性能：SIMT + grid-stride ILP + 自适应线程；P 场景对标 GPU TSV；complex64 持续优化（清零快路径 / handle 缓存 / 512 线程）
4. PTA：`scatter_npu` + `aten::index_copy_` PrivateUse1
5. 950 私仓日志闭环：`logs/scatter-950` tar 自取，禁止贴长日志

---

# 参考

- 任务书：`aclsparseScatter_A5_task_doc.md`
- cuSPARSE Scatter：https://docs.nvidia.com/cuda/cusparse/index.html
- ops-sparse README：`sparse/scatter/README.md`
- 社区模板：`resources/design_template.md`
- 对偶算子设计：`09-aclsparseGather-950/baibai_Uu/docs/design.md`

# aclsparseGather 算子设计文档（A2/A3）

> 任务来源：昇腾 CANN 社区任务「9月社区任务-aclsparseGather算子开发(A2/A3)」
> 提交仓：https://gitcode.com/cann/ops-sparse （`master`）
> 设计文档仓：https://gitcode.com/cann/cann-ops-competitions
> 参考语义：cuSPARSE `cusparseGather`（cuSPARSE 13.3 Update 1）、PyTorch `torch.index_select` / `aten::index_select`
> 目标硬件：Atlas A2（910B3、910B4）、Atlas A3（DAV_2201，arch22）

---

## 1. 需求背景（required）

### 1.1 需求来源

- 昇腾 CANN 社区 2026 年 9 月算子开发任务，算子名 `aclsparseGather`，面向 A2/A3 平台。
- 参考 cuSPARSE `cusparseGather` 接口语义与 PyTorch `torch.index_select` / `aten::index_select`，在 NPU 上实现「按稀疏向量索引从稠密向量读取元素并原地写入稀疏 values」的聚集能力。
- 代码提交至 `ops-sparse` `master`；必须交付 Python/torch 层、ATen Dispatcher 的 NPU 注册及 `aclsparseGather` 调用链，适配 PyTorch 2.7 及以上、torch_npu 26.0.0 及之后版本；公开入口为 `torch.index_select(input, 0, index)`，不得 CPU fallback。

### 1.2 背景介绍

在大模型推理的 embedding / 词表查找等稀疏访问场景中，需要从稠密向量按稀疏索引读取元素并写入稀疏向量。NPU 当前缺乏与 cuSPARSE Gather 对齐的稀疏向量聚集算子。本任务使用 C++ Host 与 Ascend C Kernel 开发并完善 `aclsparseGather`，复用 aclsparse 的 Handle、DnVec/SpVec 描述符与资源管理接口，核心计算在 NPU 完成，并补齐 Python/ATen 适配，使 `torch.index_select(input, 0, index)` 在 NPU 上完成 Gather。

### 1.3 现有实现现状

`ops-sparse` 仓库现有实现仅作为复用基线，提供 DnVec/SpVec 描述符、Handle 与资源管理能力。本任务在 `sparse/gather/arch22/` 新增 A2/A3 的 Host、Kernel、tiling 与 CMake 配置，将公开接口与类型声明更新至 `include/cann_ops_sparse.h`，并补齐 Python/torch 适配层、ATen NPU 适配、C++ UT/ST、ATen UT、Python 端到端 UT 与 README。任务书声明的目标硬件、dtype 与测试范围均须交付。

---

## 2. 需求分析（required）

### 2.1 需求描述

实现如下聚集计算：

$$X.values[i] = Y\big[X.indices[i] - idxBase\big],\quad i\in[0,nnz)$$

- `vecY`：稠密源向量 `Y`，连续一维，只读，不修改。
- `vecX`：携带 `indices`（I32）与 `values` 的 SpVec 描述符；只更新 `values`，不修改 `indices`。
- `idxBase ∈ {0, 1}`；换算 base 后索引须位于 `[0, size)`。
- 复用现有 Handle、DnVec/SpVec 描述符与资源管理；覆盖不同 `size`/`nnz`/稀疏度、乱序与重复索引、尾块及 `nnz=0/1`；相同输入和索引的输出必须 bit-wise 一致；禁止 CPU fallback。

### 2.2 需求拆解

1. 公开 C 接口 `aclsparseGather(handle, vecY, vecX)`（原型见 §3.2），保持源代码兼容。
2. A2/A3 Host 参数校验 + stream 异步下发 + Ascend C Kernel（arch22）。
3. Python/torch 层 + ATen Dispatcher NPU 注册 + `aclsparseGather` 调用链；以 `torch.index_select(input, 0, index)` 为公开入口；校验 `dim=0`、参数、dtype、shape、stride/layout、NPU device；完成 dense tensor 与 SpVec 描述符/稀疏元数据转换；构造输出；alias/in-place 与 stream 异步语义与 PyTorch 对齐；任何不支持组合显式报错，不得退回 CPU。
4. 支持 dtype：`float16`、`bfloat16`、`float32`、`complex64`；indices 为 `int32`；`idxBase` 0/1。
5. C++ UT/ST、ATen UT、Python 端到端 UT，覆盖入口/Dispatcher 注册、参数、dtype、shape、layout、stride、device、异常、输出及无 CPU fallback；README 同步交付。
6. 与 A5 公共描述符和 Host 逻辑保持可共存（后续 PR 基于已合入版本处理公共 Host 冲突并交叉回归）。

---

## 3. 详细设计（required）

### 3.1 算子分析

#### 数学公式

$$X.values[i] = Y\big[X.indices[i] - idxBase\big],\quad i\in[0,nnz)$$

#### 支持数据类型

- `values`：`float16`、`bfloat16`、`float32`、`complex64`；`vecY.values` 与 `vecX.values` dtype 必须相同。
- `indices`：`int32`。
- `idxBase`：`0` 或 `1`（SpVec 描述符属性）。

#### 支持形状 / 数据排布

- `vecY`：连续一维稠密向量 `[size]`，`size` 动态变化，`size ≥ 0`；当 `nnz > 0` 时 `size > 0`。
- `vecX`：一维 SpVec 描述符，`indices` 与 `values` 均为 `[nnz]`，`nnz` 动态变化，`nnz ≥ 0`；`idxBase` 0/1。
- 输出：原地复用 `vecX.values`（`[nnz]`），不得额外分配与输入规模线性相关的临时 Device/Host 内存。
- 计算为「按索引间接读 + 顺序写」，无归约、无跨核数据依赖，天然可并行且确定。

### 3.2 接口定义

以 `include/cann_ops_sparse.h` 为准；变更须保持源代码兼容，并同步公开头文件、README 与 UT。

```C
aclsparseStatus_t aclsparseGather(
    aclsparseHandle_t          handle,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseSpVecDescr_t      vecX);
```

#### 参数说明

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| `handle` | 输入 | aclsparse 句柄及调用方 stream | Handle | 不涉及 | Handle | 标量 | 有效已创建句柄 | 空句柄返回参数错误 |
| `vecY` | 输入 | 稠密源向量 Y | DnVec 描述符 | `float16`、`bfloat16`、`float32`、`complex64`；须与 `vecX.values` 相同 | 连续一维向量 | `[size]`；`size` 动态变化 | `size>=0`；当 `nnz>0` 时 `size>0` | 空描述符、非法 dtype、device 不一致或不满足 shape 约束返回参数错误 |
| `vecX` | 输入输出（原地输出 values） | 稀疏索引及被写入的 values | SpVec 描述符 | indices 为 I32；values 与 `vecY` 相同 | 一维 SpVec，index base 0/1 | indices/values 均为 `[nnz]`；`nnz` 动态变化 | `nnz>=0`；换算 base 后 indices 位于 `[0,size)` | 空描述符、非法 index type/base、dtype 或 device 不一致返回参数错误；Device 索引越界按异步错误协议或调用方前置条件处理 |

描述符和数据指针在接口异步执行完成前由调用方保持有效。`nnz=0` 成功返回且不启动 Kernel；零长度 `vecY` 仅允许 `nnz=0`。`vecY` 和 `vecX.indices` 只读；未声明的 `vecY`/`vecX.values` 重叠返回参数错误。最大 `size`、`nnz` 及索引溢出边界须在接口文档中明确（受 GM 地址空间与 int32 索引范围约束，换算 base 后索引须 `< 2^31`）。

### 3.3 算子实现

#### 3.3.1 Host 侧设计（参数校验 + Tiling + stream 下发）

**参数校验**（返回 `ACSPARSE_STATUS_INVALID_VALUE` / `..._ARCH_MISMATCH` 等）：

1. `handle` 非空且有效；
2. `vecY`、`vecX` 非空描述符；
3. `vecY` 与 `vecX` 设备一致且为 NPU；
4. `vecX.indices` dtype == `int32`、`idxBase ∈ {0,1}`；
5. `vecY.values` dtype == `vecX.values` dtype ∈ {f16, bf16, f32, c64}；
6. `size ≥ 0`、`nnz ≥ 0`；当 `nnz > 0` 时 `size > 0`；零长度 `vecY` 仅允许 `nnz = 0`；
7. 换算 base 后所有 `indices ∈ [0, size)`；
8. 检测 `vecY` 与 `vecX.values` 内存区间重叠（声明外别名）→ 参数错误。

**Tiling 策略**：

本算子为「间接读 + 顺序写」，无归约、无跨核数据依赖。以 `nnz` 为总量，按 AI Core 数 `coreNum` 均分：

- `tileSize = ceil(nnz / coreNum)`；
- 前 `nnz % coreNum` 个核多分 1 个元素（尾块处理），确保核间负载均衡、尾块不遗漏；
- 每个核处理连续 index 区间 `[coreStart, coreEnd)`，核内再按 UB 容量切成若干 TILE（如每 TILE 256/512 元素，开启 double buffer）；
- 不涉及 workspace / preprocess，不新增无必要 Host 同步；沿用调用方 stream 异步下发。

**tilingkey 规划**：

- `tilingkey` 按 `values` dtype 字节宽度分组：`0` = f16/bf16/f32（单元素 ≤ 4B），`1` = complex64（8B），决定 UB 缓冲大小与 per-element 拷贝宽度；可同时携带 `idxBase` 标志。kernel 侧据此走不同分支。

**stream 下发**：

从 `handle` 取调用方 stream，构造 `GatherTilingData`（含 coreNum、tileSize、nnz、base、dtype 编码），通过 `KERNEL_LAUNCH` 异步下发；`nnz=0` 直接成功返回、不启动 Kernel。

#### 3.3.2 Kernel 侧设计（Ascend C，arch22）

流程：`Init`（解析 TilingData、GM 地址、coreIdx / 核数）→ `Process`：

1. **CopyIn**：本核分片 `[s, e)` 的 `indices` 连续块拷贝入 UB（`int32` 向量块搬）。
2. **Compute**：UB 内向量减 `idxBase` 得到源位置 `src = indices[k] - base`（`int32` 向量减标量，exact）。
3. **Gather**：对分片内每个位置 `k`，按 `src` 从 `Y` GM 读取 `values[k]`，写入 `X.values` UB；按 dtype 字节宽度做 `DataCopy`（complex64 按 8B 整体搬，保证实部/虚部均 exact）；采用 double buffer 重叠 `indices` 搬入与 `Y` 间接读，提升 HBM 带宽利用率。
4. **CopyOut**：本核 `X.values[s:e)` 连续写回 GM。

**精度 / 确定性**：纯数据搬运，无算术运算，天然 bit-wise exact；重复索引各自独立写入对应 `X.values[i]`，多次执行结果一致；不引入原子操作或中间舍入。

**边界**：设备上 src 越界按异步错误协议处理（硬件异常或调用方前置校验）；`nnz=0` 核不启动。

#### 3.3.3 Python / ATen 适配设计（对应任务书 §2.0）

**入口映射**：`torch.index_select(input, dim, index)` → `aten::index_select`；在 NPU 上注册 `PrivateUse1`（NPU）分支，默认 `idxBase = 0`（index_select 语义）。

**适配函数 `index_select_npu(input, dim, index) -> Tensor`**：

1. 校验 `dim == 0`；`input` 为 1D 稠密 NPU tensor（多维权变 / 其他 dim 显式 `raise`，不回退 CPU）；
2. 校验 `input.dtype ∈ {f16, bf16, f32, c64}`、`index.dtype == int32`、`input` 与 `index` 同在 NPU、`index ≥ 0` 且 `< input.size(0)`（base=0）；
3. 构造 `aclsparseDnVecDescr(vecY)`：`data_ptr = input`，`size = input.numel()`，`dtype`；
4. 构造 `aclsparseSpVecDescr(vecX)`：`indices = index`（int32），`values = output`（新分配 `[nnz]` 同 dtype），`nnz = index.numel()`，`base = 0`；
5. 取 `c10::npu::getCurrentNPUStream()` 设为 aclsparse handle stream；
6. 调用 `aclsparseGather(handle, vecY, vecX)`；返回 `output`（即 `vecX.values`），保持异步、不 `synchronize`；
7. alias/in-place：输出为新分配 tensor，不与 `input` 重叠；若发生别名按 PyTorch 语义处理，显式报错而非 CPU fallback。

**直接入口**：同时暴露 `torch.ops.ops_sparse.gather(values, indices, base)`（对应 base 0/1 全能力路径，供 C++/Python 测试与原生 aclsparse 调用）；`torch.index_select` 走 base=0 分支。

**可共存**：适配层与 A5 公共 Host / 描述符逻辑共享，仅 arch22 Kernel 与 tiling 走 `sparse/gather/arch22/`。

### 3.4 支持硬件

- Atlas A2 训练系列：910B3、910B4；
- Atlas A3 具体型号（DAV_2201，arch22）；
- 各型号均执行功能、精度与性能验收。A5 公共描述符 / Host 逻辑可共存，后续 PR 基于已合入版本处理公共 Host 冲突并完成交叉回归。

### 3.5 算子约束限制

- 仅支持 `dim=0`、1D 稠密 `input` 的 `torch.index_select` NPU 路径；其他 dim / 多维权变显式报错。
- `values` dtype ∈ {f16, bf16, f32, c64}；`indices` 仅 `int32`；`base ∈ {0,1}`；`vecY`/`vecX` values dtype 须一致。
- 设备一致（均为 NPU）；禁止 CPU fallback。
- `nnz=0` 成功返回不启动 Kernel；零长度 `vecY` 仅允许 `nnz=0`。
- 不修改 `vecY`、`vecX.indices`；`vecY` 与 `vecX.values` 重叠（声明外）返回参数错误。
- 不涉及 workspace / preprocess；不额外分配与输入规模线性相关的临时内存。
- 最大 `size`/`nnz` 与索引溢出边界在接口文档明确（受 GM 地址空间与 int32 索引范围约束）。

---

## 4. 可维可测分析

### 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 以 CPU Golden 为精度基准，逐元素 bit-wise exact match；`float16`/`bfloat16` 用 `float32` Golden，`float32` 用 `float64` Golden，`complex64` 用 `complex128` Golden；`complex64` 实部、虚部均需 exact match。校验 `vecY`、indices 与非目标数据未被修改，并覆盖普通值、小值、正负混合、零、离群值及规格允许的 INF/NAN | experimental_standard.md |
| 性能标准 | 性能倍率 = GPU 标杆设备 Event 耗时 / NPU 同调用范围总耗时 ≥ 0.25×；P-01/P-02/P-03 每个有效 case 均达标 | 任务书 §3.3 |
| 内存标准 | 不涉及 workspace；输出原地复用 `vecX.values`；I/O 总量 > 500 MB 时 NPU 额外内存 ≤ GPU 使用内存总量 50%，否则方案固有 workspace 绝对值 ≤ 目标硬件 L2 Cache 容量 | 任务书 §3.4 |

性能基线（PyTorch GPU 设备 Event `median_us`）：P-01 60.128–112.064μs；P-02 69.552–81.888μs；P-03 67.776–76.032μs（各 8 条）。NPU 实测耗时与倍率在自测报告中记录，目标 ≥ 0.25× 标杆。

### 4.2 兼容性分析

新算子，与 A5 公共 Host / 描述符逻辑可共存；接口仅新增，保持源代码兼容。

### 4.3 测试用例设计（对齐任务书 §3.5）

| 类别 | 必测场景 | 来源 |
|---|---|---|
| 基础功能 | 不同 `size`/`nnz`、乱序和重复索引、base 0/1、`nnz=0/1`、尾块与动态规模 | §3.5 |
| 接口与异常 | 空 handle/描述符/指针、dtype 或 device 不一致、非法索引类型 / base、输出边界与输入只读性、vecY 与 values 重叠 | §3.5 |
| 一致性 | 四种 dtype 的 exact match、重复执行 bit-wise 一致、complex64 专项、INF/NAN | §3.5 |
| NPU 与资源 | NPU Dispatch / Profiler 无 CPU fallback；连续创建、执行和销毁描述符无泄漏或非法同步 | §3.5 |

数据生成规则（对齐任务书 §3.5）：DnVec `values` 70% 用 `[-1,1]` 均匀分布、20% 用 `N(0,1)`、10% 覆盖零值 / 边界值 / 规格允许特殊值；SpVec `indices` 在合法范围均匀生成并专项覆盖重复、首尾和越界异常；属性覆盖 `int32`、四种 `values` dtype、`base` 0/1、`nnz=0/1` 与动态长度组合。复用任务包 `test_cases/`（算子专项 `aclsparseGather_testCase/`、公共运行依赖 `common/`、标杆原始结果 `baseline_results/`）及 ATK 精度框架（`nodes_accuracy.yaml`、`accuracy_cases.json` 200 条泛化用例）。

### 4.4 自测方案

- **C++ UT/ST**：参数校验、四 dtype、nnz=0/1、base 0/1、乱序 / 重复 / 尾块、INF/NAN、complex64 exact、重叠检测、连续创建/执行/销毁无泄漏。
- **ATen UT**：入口 / Dispatcher 注册、dim/dtype/shape/stride/layout/device 校验、异常、输出构造、无 CPU fallback。
- **Python 端到端 UT**：`torch.index_select` NPU vs CPU Golden（exact）；`gather_npu` 直调（base 0/1）。
- **性能**：aclsparse Kernel 与 Python/ATen 端到端耗时；P-01/P-02/P-03 预热 ≥ 10 次、采样 30 次取中位数与 p90；固定随机种子、复用描述符；对照 GPU 标杆计算倍率。
- **内存**：使用 `performance_cases.json` 的 P-01/P-02/P-03 及泛化性能用例，运行 `collect_sparse_ops_{gpu,npu}_memory.py` + `compare_sparse_ops_memory.py` 对比 `input_baseline_*_bytes`、`peak_*_bytes`、`extra_peak_*_bytes`。
- **交付**：设计文档 PR 合入截图、测试用例、用例结果自测报告、测试步骤指导文档（README）；性能报告同时提供 aclsparse Kernel 与 Python/ATen 端到端耗时，附 Profiler 证据。

### 4.5 交付与目录

- **代码（ops-sparse `master`）**：
  - 公开接口与类型声明：`include/cann_ops_sparse.h`；
  - A2/A3 Host、Kernel、tiling 及构建配置：`sparse/gather/arch22/` 及相关 CMake；
  - C++ 测试、NPU wrapper、参数与 Golden：`test/gather/arch22/` 及 `test/gather/` 公共目录；
  - Python/torch 层接口、ATen NPU 适配、输出构造、能力补齐代码、C++ UT 与端到端 UT 随仓库规范目录交付；
  - 任务验收脚本、精度用例、性能 / 内存用例与标杆结果随任务包 `test_cases/` 交付（算子专项 `aclsparseGather_testCase/`、公共运行依赖 `common/`、标杆原始结果 `baseline_results/`）。
- **设计文档（cann-ops-competitions）**：PR 标题【CANN社区任务】aclsparseGather算子设计文档，路径 `04_tasks/01_community-task-2026/` 下；审核通过后合入仓库。

---

> 说明：本文档接口原型、参数约束、精度 / 性能 / 内存验收标准均严格遵循任务书 `aclsparseGather_A2A3_task_doc.md`；目录布局遵循任务书 §5。代码实现阶段将依据本设计在 `ops-sparse` 完成 Host/Kernel/适配与测试交付。

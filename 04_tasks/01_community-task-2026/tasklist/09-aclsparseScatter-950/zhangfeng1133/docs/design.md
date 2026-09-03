# aclsparseScatter 算子开发设计文档（A5 / Ascend 950）

> 算子：`aclsparseScatter`（稀疏向量散布写入，对标 cuSPARSE `cusparseScatter` 13.3 U1）
> 仓库：ops-sparse `sparse/scatter/arch35/`；配套 Python/ATen 适配 `python/`
> 硬件：Ascend 950（DAV_3510，`arch35`）；CANN 9.2.0-beta.1

# 需求背景（required）

## 需求来源

社区任务"2026·aclsparseScatter 算子开发（A5）"任务书：
- 参考 cuSPARSE Scatter 接口语义，面向 Ascend 950（A5，arch35）完成 `aclsparseScatter` 的
  迁移、能力补齐与工程化交付，代码合入 ops-sparse `master`；
- Python/ATen 适配为必选交付：`Tensor.index_copy_(0, index, source)` → `aten::index_copy_`，
  核心路径禁止 CPU fallback；
- 性能目标：P-01/02/03 全部声明 dtype ≥0.3× PyTorch GPU 标杆（设备 Event median）。

## 背景介绍

### aclsparseScatter 算子实现现状分析

ops-sparse 上游 `master` 已含 CANN 8.5.1 时期的 `sparse/scatter/arch35/` 实现（FP32/FP16/BF16 ×
I32/I64）。迁移验证结论（CANN 9.2.0-beta.1 + Ascend 950PR 实测）：

- 全量编译通过，无 breaking API 变更；
- 既有 121 条 gtest 全部通过，SIMT 路径（`__simt_vf__` + `asc_vf_call`）与
  `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 在 950PR 上可用。

因此本任务工作量集中在：**能力补齐（INT8 / complex64）、未声明 alias 校验、Python/ATen 适配、
测试与交付件扩展**，而非接口重写。

### aclsparseScatter 算子功能分析

散布写入：`Y[X.indices[i] - idxBase] = X.values[i], i ∈ [0, nnz)`。

| 参数 | 参数含义 | 输入/输出 | 数据类型（支持） | 约束 |
| --- | --- | --- | --- | --- |
| handle | aclsparse 句柄及调用方 stream | 输入 | Handle | 有效已创建句柄，空则参数错误 |
| vecX | 稀疏向量描述符（values + indices） | 输入（只读） | values：int8/float16/bfloat16/float32/complex64；indices：I32（保留 I64 超集） | base 0/1；`nnz>=0`；换算 base 后 indices ∈ [0,size) |
| vecY | 稠密向量描述符 | 输入输出（原地） | 与 vecX.values 相同 | `size>=0`；`nnz>0` 时 `size>0` |

语义约束：vecX 只读；未写入位置保持原值；`nnz=0` 成功返回且不改 vecY；重复索引为非确定性
last-write-wins（不得实现为累加）；无重复索引输出 bit-wise 一致；未声明的输入输出 alias
（vecX.values/indices 与 vecY.values 指针相同）返回参数错误；接口零 workspace、无新增 Host 同步。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Ascend 950（arch35）上完成 `aclsparseScatter`：
支持 int8 / float16 / bfloat16 / float32 / complex64 五种 values dtype、I32 索引与 index base 0/1；
提供 aclsparse C++ 公开接口、Host 校验与 Kernel 实现；提供 Python/ATen 适配
（`aten::index_copy_` 的 PrivateUse1 覆盖 + 任务包契约入口 `torch.ops.ops_sparse_test.scatter_npu`）；
精度 CPU Golden 逐位 exact；性能 ≥0.3× GPU 标杆；内存满足任务书 3.4（零 workspace）。

## 需求拆解

1. 迁移验证：CANN 8.5.1 基线在 CANN 9.2/950PR 编译与 gtest 全量回归；
2. 能力补齐：INT8/complex64 进入 tiling 编码、Kernel 分发、Host 校验、测试 CSV/golden/wrapper；
3. 合规补齐：未声明输入输出 alias 的 host 侧校验与文档；
4. Python/ATen 适配：torch 扩展双入口（契约入口 + ATen 覆盖），单核心路径直调 aclsparseScatter；
5. 测试：C++ gtest（CSV 驱动 + 异常 + alias + 白盒）、任务包精度 200 例、ATen/Python UT、
   性能 30 例（5 dtype × base0/1 × P-01/02/03）、内存对比 30 例。

# 详细设计（required）

## 算子分析

### 数学公式

```
Y[X.indices[i] - idxBase] = X.values[i],  i ∈ [0, nnz)
```

纯地址散布写入，无数值运算：任意 dtype 均为**位拷贝**，这是 INT8/complex64 与精度口径
（逐位 exact）的设计基点。

### 支持数据类型

| 项 | 支持 |
| --- | --- |
| values | int8、float16、bfloat16、float32、complex64 |
| indices | int32（int64 作为上游超集保留） |
| index base | 0 / 1 |

### 支持形状

一维稀疏向量（indices/values 均 `[nnz]`）写入一维稠密向量（`[size]`）；`nnz`、`size` 动态；
覆盖 `nnz=0/1`、尾块（nnz 非 256 倍数）、`size>nnz`、`size=nnz`（满写）。

## 算子实现

### 实现方案

#### Host 侧设计

沿用上游 host 框架（`ValidateScatterParams → LaunchScatterKernel`），与 A2/A3 公共逻辑共存：

1. **参数校验**（任一失败即返回，不触碰 NPU）：
   - 空句柄 → `HANDLE_IS_NULLPTR`；空描述符/nnz>0 时空指针 → `INVALID_VALUE`；
   - `idxType ∉ {I32, I64}` → `NOT_SUPPORTED`；`idxBase ∉ {0,1}` → `INVALID_VALUE`；
   - `nnz > size` → `INVALID_VALUE`；`size > vecY.nums` → `INVALID_VALUE`；
   - `valueType ∉ 五种声明` 或 X/Y 不一致 → `NOT_SUPPORTED`；`nnz > UINT32_MAX` → `NOT_SUPPORTED`；
   - **未声明 alias**（v2 新增）：`vecX.values == vecY.values` 或 `vecX.indices == vecY.values` →
     `INVALID_VALUE`（指针同一性判定；更细粒度区间重叠属调用方前置条件，与越界校验同口径）。
2. **Tiling/分核**：纯 GM 读-写、无 UB 依赖，tiling 仅传
   `ScatterTilingData{nnz, idxBase, idxType, valType}`（值传递，无 TilingKey，dispatcher 按 valType 分发）。
   AIV 核数 `coreNum = GetAivCoreCount()`；`useNumBlocks = min(coreNum, ceil(nnz/256))`；
   Kernel 内 grid-stride 循环覆盖任意 nnz（含尾块），优先满核、核间近似均分。
3. **下发**：`scatter_kernel_do<<<numBlocks, nullptr, stream>>>` 沿调用方 stream 异步执行；
   零 workspace、零临时 Device/Host 缓冲，无新增 Host 同步。

#### Kernel 侧设计（Ascend C / SIMT）

`__simt_vf__ ScatterSimtCompute<ValT, IdxT>`：每线程网格步进处理一个 nnz 元素，
`gmY[gmIndices[i] - idxBase] = gmValues[i]`，块内 256 线程：

| valType | ValT 特化 | 说明 |
| --- | --- | --- |
| SCATTER_VAL_FP32/FP16/BF16 | float / half / bfloat16 | 上游既有 |
| SCATTER_VAL_INT8（=3，v1 新增） | int8_t | 1 字节位拷贝，±0 符号位/位模式天然保持 |
| SCATTER_VAL_C64（=4，v1 新增） | uint64_t | complex64 以 8B 位拷贝承载（CCE 禁止 GM→GM struct 隐式拷贝，拷贝构造要求 Local Memory，故取 uint64 标量赋值等价 64-bit 位拷贝；实/虚部不拆分，INF/NAN payload 保持） |

编码追加置于末尾（0/1/2 不变），与已发布版本二进制兼容。禁止任何数值转换（scatter 红线）。

#### Python/ATen 适配设计（任务书 2.0 必选）

torch 扩展（ops-sparse `python/`，本仓 `npu_wrapper/` 为同源运行副本），两个入口一条核心路径：

1. **契约入口**：`torch.ops.ops_sparse_test.scatter_npu(values, indices, size, base) -> Tensor`。
   `at::empty` 分配输出 + `aclrtMemsetAsync` 清零（驱动级 memset，dtype 无关；规避 aclnn 对
   complex64 的 fill 慢路径：1MB≈107µs，曾致倍率 0.22×）→ 描述符 → `aclsparseScatter`。
2. **ATen 必选入口**：`TORCH_LIBRARY_IMPL(aten, PrivateUse1)` 覆盖 `aten::index_copy_`：
   - 校验后直调核心：`self.dim()==1 && dim==0`、self contiguous、index/source NPU 张量且同设备、
     `numel` 相等、dtype ∈ 五种声明、source contiguous、`index.dim()==1`；
     不满足抛明确错误，**禁止 CPU fallback**；index int32→I32 / int64→I64，base 恒 0；
   - 原位写入 `self` 存储并返回 `self`（in-place/alias 语义对齐 PyTorch）；`nnz=0` 直接返回；
   - 重复索引交由 Kernel 并发 last-write-wins（非确定性），符合 PyTorch `index_copy_` 口径；
   - 加载顺序：须在 `import torch_npu` 之后导入本扩展（后注册者生效）。
3. **Handle/Stream**：进程级单例 handle；每次调用 `aclsparseSetStream(getCurrentNPUStream(dev))`；
   `NPUGuard` 设备守卫。950PR AIVEC 概率性调度挂起（平台缺陷）对策：进程首启
   `aclrtSynchronizeStreamWithTimeout(120s)×10` 有界自愈，其后纯异步。

### 测试设计

- C++ gtest：CSV 参数化（126 行：5 dtype × base0/1 × sorted/unsorted × normal/special/extreme ×
  nnz=0/1/尾块）+ 异常/alias/白盒；golden 逐字节 memcmp，special/extreme 位模式表覆盖
  ±0 符号位、INF/NAN payload、subnormal；
- 任务包精度 200 例：无重复 → 逐位 exact；有重复 → 输出 ∈ 写入值集合 + 只读校验；
- ATen/Python UT：语义/异常/alias/重复索引/无 fallback 三层证据（msprof 采集 `scatter_kernel`、
  PrivateUse1 dispatch 键、行为证据）；
- 性能：NPU Event，预热 10 采样 30，固定种子、无重复索引，median/p90；对照组 PyTorch GPU
  设备 Event（任务包基线 TSV）；
- 内存：任务包 collect/compare 字段口径（input_baseline/peak/extra + workspace）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950（DAV_3510，arch35） | √ |

## 算子约束限制

- 一维向量；`nnz ≤ UINT32_MAX`；`nnz>0` 时 `size>0` 且 `size ≤ vecY.nums`；
- 索引合法域为调用方前置条件（host 不读 device 内存做越界校验，越界按异步错误协议，与 cuSPARSE 一致）；
- 重复索引输出非确定（last-write-wins）；未声明输入输出 alias 拒绝；
- ATen 入口仅覆盖 1-D `index_copy_`（dim=0）；不支持组合抛明确错误，不 fallback。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 实测（950PR，CANN 9.2.0-beta.1） |
| --- | --- | --- |
| 精度标准 | CPU Golden 逐位 exact；INT8 原始 bit；FP16/BF16←FP32、FP32←FP64、C64←complex128 | 任务包 200/200 + 性能配置补验 30/30 + gtest 147/147（含 special/extreme memcmp） |
| 性能标准 | 所有声明 dtype ≥0.3× GPU 标杆（设备 Event median） | 30/30 PASS，倍率 0.95×–1.07×（任务书口径 GPU/NPU）；ATen e2e 26–27µs |
| 内存标准 | 任务书 3.4：io>500MB 时额外内存 ≤ GPU 50%，或零 workspace ≤ L2 | 30/30 PASS：io ≤1.26MB（规则1不触发）；零 workspace 命中规则2；NPU peak ≤ GPU peak |

## 兼容性分析

- C++ 接口签名不变（`include/cann_ops_sparse.h`），tiling 编码追加式扩展，源代码/二进制兼容；
- 与 A2/A3 公共 Host 校验结构共存，无接口冲突；后续 PR 基于已合入版本处理公共逻辑并交叉回归；
- ATen 覆盖仅作用 PrivateUse1 后端且依赖加载顺序（torch_npu 先导入），不影响 CPU/其他后端。

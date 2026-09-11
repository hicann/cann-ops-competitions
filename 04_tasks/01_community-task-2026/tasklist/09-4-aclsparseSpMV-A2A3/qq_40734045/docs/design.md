# aclsparseSpMV 算子设计文档（A2/A3）

## 文档信息

| 项目 | 内容 |
|---|---|
| 任务名称 | 9月社区任务-aclsparseSpMV算子开发(A2/A3) |
| 目标仓库 | `cann/ops-sparse` |
| 目标目录 | `sparse/spmv/arch22/`、`test/spmv/arch22/`、`python/ops_sparse_torch/` |
| 公共接口声明 | `include/cann_ops_sparse.h` |
| 设计文档提交位置 | `04_tasks/01_community-task-2026/tasklist/09-4-aclsparseSpMV-A2A3/qq_40734045/docs/design.md` |
| 设计基线 | 官方任务包 SHA-256 `6eb39eb5f3f131333c5cbb8296e8da7a026677374c2f81b6100190cb5d7bddb` |
| 代码基线 | `ops-sparse` 分支 `ops-sparse`；平台条件实现提交 `cc65ca22657f352c89bee488ae0f7ccdca17f42a`；A2/A3 平台分支实测提交 `bdd54af71fb183dca9b8c2dd8eabe148f8d742da`；最新证据归档提交 `0f522f540c34dcfc821ba7057991e1c69d77b5c1` |
| 文档状态 | 设计文档 PR !1478 已提交，等待评审反馈 |

# 需求背景（required）

## 需求来源

面向 Atlas A2/A3（DAV-2201，`arch22`）实现 CSR 稀疏矩阵乘稠密向量：

```text
Y = alpha * op(A) * X + beta * Y
```

算子提供 `aclsparseSpMVGetBufferSize`、`aclsparseSpMVPreprocess`、`aclsparseSpMV` 三阶段 C++ API，并通过 `aten::mv` 的 `SparseCsrPrivateUse1` NPU 注册公开 `torch.mv(CSR二维, Dense一维)` 入口。核心计算和必要稀疏变换均在 NPU 完成，CPU 只承担参数校验、任务编排和测试 Golden。

成功条件包括：接口与 cuSPARSE 语义对齐、覆盖任务声明 dtype 与 `N/T/H`、精度达标、性能倍率不低于 0.25、workspace 满足内存标准、torch_npu 端到端无 CPU fallback、资源生命周期可管理，并提供可复现测试证据。

## 背景介绍

### aclsparseSpMV算子实现优化

CSR 由 `rowOffsets`、`colIndices`、`values` 表示。非转置路径按输出行独占，可避免写冲突；转置和共轭转置若按原始行直接累加，会形成多核对同一输出的竞争，因此先构建 CSC 视图，再按输出列独占归约。

主要优化方向：

1. **行批量化**：对行长小于或等于 UB tile 的连续行批量搬运、批量乘加和批量写回。
2. **exact-64 树归减**：每行 64 个非零元素时，用三次带 repeat stride 的向量 Add 将 64 元素归约到 8 元素，再合并，减少逐行 `ReduceSum` 开销。
3. **非 8 对齐尾块兼容**：`ReduceSum` 起点向下对齐到 8 元素边界，并扣除前一行尾巴，避免 UB 非对齐访问。
4. **负载自适应**：按 `nnz + rowCostWeight × rows` 划分 AIV 任务，兼顾空行、长行和极不均衡 pattern。
5. **complex 拆分与融合**：complex64 输入先拆成实部/虚部 FP32 数组；`xLen <= 8192` 的 N 路径使用 fused Gauss 三乘法，`xLen > 8192` 时复用四个 real partial kernel，避免实/虚双 x cache 超出 UB。
6. **Host plan cache**：torch hook 与公开 `torch.mv` 复用 Handle、CSR/DnVec 描述符、workspace 与 Preprocess 结果。策略按平台编译期区分：A2 910B3/910B4 使用同一优化路径，key 不含输出指针，int64→int32 转换仅 miss 时执行，输出缓冲区变化时重绑定 `vecY`；该路径已在 910B3 实机验证，910B4 仅完成构建与条件开关验证。A3 910_93 使用稳定路径，key 包含输出指针，公开入口每次调用先将 int64 索引转换为 int32。该拆分避免 A2 优化影响 A3 正确性。

### aclsparseSpMV算子实现现状分析

官方 `master` 原有 `aclsparseSpMV` 仅是头文件声明，`docs/zh/api_list.md` 标记 GetBufferSize/Preprocess 暂未支持。本任务补齐：

| 能力 | 交付位置 |
|---|---|
| 公共 API 声明 | `include/cann_ops_sparse.h` |
| Host 校验、workspace、调度 | `sparse/spmv/arch22/spmv_host.cpp` |
| real N/T kernel | `sparse/spmv/arch22/kernels/spmv_kernel*.cpp` |
| complex 拆分、融合与 epilogue | `spmv_split_kernel.*`、`spmv_complex_fused_kernel.*`、`spmv_epilogue_kernel.*` |
| CSC 预处理 | `spmv_csc_preprocess_kernel.*` |
| C++ UT/ST | `test/spmv/arch22/spmv_test.cpp` |
| Python/ATen 入口与 hook | `python/ops_sparse_torch/csrc/spmv_torch.cpp` |
| 官方 accuracy/performance/memory 复验脚本 | `test/spmv/python/` |

标准 `aclsparseSpMV` 与已有 A5 `aclsparseSpMVOp` 保持分离：本任务只修改 `sparse/spmv/arch22/` 和标准 API 链路，不混用 `sparse/spmv_op/arch35/` 的 legacy op 接口。

### aclsparseSpMV算子功能分析

- A 为二维 CSR，X/Y 为一维连续稠密向量。
- `N`：X 长度为 K，Y 长度为 M。
- `T`：X 长度为 M，Y 长度为 K。
- `H`：与 T 的维度一致；对 complex64 的 A values 取共轭，实数 A 等价于 T。
- `alpha/beta` 按 `computeType` 解释，complex64 支持复数标量。
- `beta=0` 不读取旧 Y，避免未初始化值或 NaN 污染输出。
- 输出 Y 原地更新；A、rowOffsets、colIndices、X 只读。
- 不支持广播，不支持 CPU fallback。

# 需求分析（required）

## 需求描述

实现动态 M/K/nnz、I32 索引、base 0/1、`N/T/H`、任务声明 dtype 组合、三阶段 workspace 与异常语义；交付 torch.mv/ATen NPU 适配和任务 hook；官方 200 条 accuracy、308 条 performance、1000 条泛化和内存指标全部可复现。

## 需求拆解

| ID | 要求 | 设计落点 | 证据 |
|---|---|---|---|
| R01 | 三阶段 API | `spmv_host.cpp` | C++ 220/220 |
| R02 | CSR/I32/base 0/1 | Host 校验与 UT | C++ 220/220 |
| R03 | `N/T/H` | real/CSC/complex 路径 | C++ 与官方 accuracy |
| R04 | 9 类 dtype 组合 | 模板 kernel 与 complex 拆分 | accuracy 200/200 |
| R05 | alpha/beta | Host 标量与 epilogue | C++ 与官方 accuracy |
| R06 | 确定性与输入只读 | 固定归约顺序、只读 GM、输入 SHA-256 校验 | A3/A2 910B3 均 9/9 通过 |
| R07 | torch.mv/aten::mv NPU | `SparseCsrPrivateUse1` 注册 | torch_npu 端到端通过 |
| R08 | 精度 | CPU FP64/complex128 Golden | accuracy 200/200 |
| R09 | 性能倍率 ≥0.25 | 批量归约、CSC、plan cache | 296/296 有基线 case 达标 |
| R10 | 内存 | workspace ≤ L2 | 最大 workspace 10,301,504 B |
| R11 | A2 B3/B4、A3 硬件 | arch22 代码与平台条件编译 | A3 与 A2 910B3 实测通过；A2 910B4 仅构建，实机待测 |
| R12 | C++/ATen/Python/Profiler | 测试脚本 | 报告目录与 profiler trace |
| R13 | 资源生命周期 | RAII、plan cache、workspace 复用 | C++ 回归与内存报告 |

官方用例规模：

| 集合 | 数量 | 结果 |
|---|---:|---|
| accuracy_cases.json | 200 | 200/200 通过 |
| performance_cases.json | 308 | 296 条有 GPU 基线全部达标，12 条无基线仅记录 NPU 实测 |
| 1000 条泛化 | 1000 | 1000/1000 通过 |

# 详细设计（required）

## 算子分析

### 数学公式

设 `s_i = rowOffsets[i] - base`，`e_i = rowOffsets[i+1] - base`，`j_p = colIndices[p] - base`。

非转置：

```text
Y[i] = alpha * Σ(p=s_i..e_i-1) values[p] * X[j_p] + beta * Y_old[i]
```

转置：

```text
Y[j] = alpha * Σ(i,p: j_p=j) values[p] * X[i] + beta * Y_old[j]
```

共轭转置在转置公式基础上对 `values[p]` 取复共轭。

### 支持数据类型

| A/X 类型 | computeType | Y 类型 | 说明 |
|---|---|---|---|
| INT8 | INT32 | INT32 | 整数精确乘加 |
| INT8 | FLOAT | FLOAT | 转 FP32 计算 |
| FP16 | FLOAT | FLOAT | FP32 累加 |
| FP16 | FLOAT | FP16 | FP32 累加后一次舍入 |
| BF16 | FLOAT | FLOAT | FP32 累加 |
| BF16 | FLOAT | BF16 | FP32 累加后一次舍入 |
| FP32 | FLOAT | FP32 | FP32 累加 |
| FP32 | COMPLEX64 | COMPLEX64 | 实数 dot、复数标量/epilogue |
| COMPLEX64 | COMPLEX64 | COMPLEX64 | 实虚拆分，N/T/H 全支持 |

INT8→INT32 官方 accuracy 要求 exact match；实测 24/24 exact。

### 支持形状

- A：`[M, K]`，CSR，I32 索引，base 0/1。
- X：N 为 `[K]`，T/H 为 `[M]`。
- Y：N 为 `[M]`，T/H 为 `[K]`。
- 支持 M/K/nnz 为 0、空行、长尾行、重复坐标按输入顺序累加。
- `M=0` 或 `K=0` 时要求 nnz 为 0。
- `nnz + base` 和每个列索引均不得超出 I32 范围。

## 算子实现

### 实现方案

#### host侧设计

1. **参数校验**：Handle、指针、pointer mode、CSR 格式、索引类型、base、dtype 组合、算法、向量长度和矩阵 shape。非法输入返回参数错误或 NOT_SUPPORTED。
2. **workspace 查询**：按 op、dtype、shape、nnz 计算 CSC、complex 拆分、partial 和对齐空间；所有加法、乘法和 align 使用溢出检查。
3. **预处理**：在调用方 stream 上构建 CSC 视图或 complex 实虚拆分；结果绑定 `matA->activeBuffer`，pattern/workspace 变化后重建。
4. **执行**：复用预处理结果；若未预处理或传入新 workspace，则在同一 stream 自动重建，保证功能路径可用。
5. **异步语义**：kernel 全部提交到 Handle 绑定的调用方 stream，不做隐式同步。

workspace 布局：

| 路径 | 内容 |
|---|---|
| real N | 通常 0 |
| real T/CSC | `cscColPtr`、`cscRowIdx`、可选重排 values |
| complex64 | `valueReal`、`valueImag`、`xReal`、`xImag`、4 个 partial |
| FP32→complex | 1 个 real partial + complex epilogue |

所有分段按 64-byte 对齐。官方 308 条性能/内存 case 中最大 workspace 为 `10,301,504` bytes。

##### 1. 分核策略

- 获取平台 AIV 数，blockDim 不超过可用 AIV。
- N 路径按输出行或 `nnz + rowCostWeight × rows` 均衡。
- 空输出不 launch；nnz=0 且输出非空时执行置零/缩放路径。
- T/H 使用 CSC 后按输出列独占，避免 GM atomic。
- rowCostWeight 根据平均行长自适应：平均行长较大时使用较小行权值，极不均衡场景仍保证空行和长行不丢失。

##### 2. 数据分块和内存优化策略

- real kernel tile 为 1024 个非零元素。
- `rowOffsets`、`colIndices`、`values` 按 tile 批量搬入 UB。
- X 按列索引 Gather 到 UB；不同 ValT/CompT 组合在局部完成类型转换。
- 长行按 tile 分段，固定次序累加。
- 行长小于 tile 的连续行合批，减少 kernel launch 内部循环和写回次数。
- complex64 先拆分为 FP32 实/虚数组；`xLen <= 8192` 使用 fused kernel，`xLen > 8192` 使用四个 real partial kernel。

##### 3. tilingkey规划策略

公共 `alg` 仅接受 `ACL_SPARSE_SPMV_ALG_DEFAULT`，非法值返回 NOT_SUPPORTED。内部通过模板实例和 Host 分支区分：

- compute/value/x/output dtype；
- N 与 T/H；
- real 与 complex；
- CSC 预处理与直接 CSR；
- 短行批量、exact-64、长行分段；
- complex fused 与四 partial 回退。

#### kernel侧设计

每个 kernel 采用 `Init → Process` 结构，Process 内部按数据依赖执行 CopyIn、Compute、CopyOut：

1. `SpmvKernel`：real N 路径，模板覆盖任务 dtype。
2. `SpmvKernelTrans` / CSC kernel：real T/H 路径。
3. `SpmvSplitComplexKernel`：将 complex values 与 X 拆成实/虚 FP32。
4. `SpmvComplexFusedKernel`：`xLen <= 8192` 的 complex N fused 路径，Gauss 三乘法。
5. real partial kernel 复用：`xLen > 8192` 的 complex N 路径分别计算四个实数 partial。
6. `SpmvCscPreprocessKernel`：构建 `colPtr`、`rowIdx` 与按列重排 values。
7. `SpmvComplexEpilogue`：组合实/虚 partial、alpha/beta 与旧 Y，输出 complex64。

精度策略：

- INT8→INT32 使用整数乘加，exact match。
- FP16/BF16 输入转 FP32 计算；输出为 FP16/BF16 时最后一次舍入。
- FP32 短行使用 Neumaier + Kahan 补偿求和，降低顺序累加误差。
- complex64 实虚均按 FP32 处理，Golden 使用 complex128。

## 支持硬件

| 芯片 | 状态 |
|---|---|
| Atlas A2 训练系列 910B3 | 平台条件分支（`bdd54af`，`REBIND_Y=ON`）干净重构建后，功能、精度、性能、内存、泛化、确定性、资源生命周期、Profiler 与 10 进程稳定性全部复验通过 |
| Atlas A2 训练系列 910B4 | Release 构建与平台条件开关验证通过；无 910B4 实机，功能/精度/性能未实测，不以 910B3 结果替代 |
| Atlas A3 训练系列 `ascend910_93` | 平台条件分支后的功能、精度、性能、内存、泛化、确定性、资源生命周期与 Profiler 全量复验通过 |
| A5 `aclsparseSpMVOp` | 保持既有 `arch35` 独立实现，未混用；`spmv_op_test` Release 构建通过，实机回归待 A5 环境 |

A3 环境为 `IT22HMDA_2_S`，8 个芯片均为 `Ascend910_93` / DAV-2201。A2 910B3 与 910B4 同属 A2 训练系列和 arch22 目标，平台条件均为
`SPMV_TORCH_PLAN_CACHE_REBIND_Y=ON`；910B3 实测结果可作为同族参考与风险判断依据，但不能替代 910B4 按任务书要求的实机功能、精度和性能验证。

## 算子约束限制

- 仅支持 CSR，不支持 CSC/COO/BSR。
- 仅支持 I32 索引和 base 0/1。
- 仅支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`。
- 仅支持 Host pointer mode；Device pointer mode 显式返回 NOT_SUPPORTED。
- 公开 `torch.mv` 仅支持 CSR 二维矩阵 × 一维稠密向量，输入必须在同一 NPU device；固定 alpha=1、beta=0、N。
- 公开 `torch.mv` 支持 FP32/FP16/BF16/complex64；INT8/INT32 等任务专用组合通过 `ops_sparse_test.spmv_npu` hook 覆盖。
- 不支持广播、CPU fallback、X/Y alias。
- CSR 行内重复列索引按输入顺序累加；建议输入满足常规 CSR 唯一有序约束以保证确定性和与库语义一致。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 结果 |
|---|---|
| 官方 accuracy 200 条 | 200/200 通过 |
| 最大绝对误差 | `3.8147e-6` |
| 最大相对误差 | `1.0495e-5` |
| INT8→INT32 | 24/24 exact |
| 官方 performance 有基线 case | 296/296 `GPU/NPU >= 0.25` |
| 最低 ratio | A3 `0.7356`；A2 910B3 平台分支 `0.5937` |
| ratio 中位数 | A3 `6.8713`；A2 910B3 平台分支 `2.7398` |
| ratio 最大值 | A3 `34.5433`；A2 910B3 平台分支 `15.5808` |
| 无 GPU 基线 | 12 条 INT8→INT32，记录 NPU 实测，不伪造 ratio |
| 最大 workspace | `10,301,504` bytes |
| L2 门槛 | 小于目标平台配置的最小 L2 `100,663,296` bytes |

### 代表性能

| Case | GPU median | NPU median | Ratio |
|---|---:|---:|---:|
| P-01 complex64 N base0 | 198.144 us | 269.370 us | 0.7356 |
| P-01 complex64 N base1 | 204.192 us | 260.300 us | 0.7844 |

A2 910B3 平台分支复验中，最差 case 仍为
`spmv-P-01-complex64-to-complex64-compute-complex64-base0-N`，NPU median `333.73 us`、
p90 `340.14 us`，ratio `0.5937`；`aten::mv` 端到端 median `139.38 us`、p90 `157.47 us`。

Profiler trace 显示每次调用包含 4 次 `spmv_kernel_float_float_float` 和 1 次 `spmv_epilogue_complex`，证明 large-cols complex 路径真实在 NPU kernel 中执行。

## 兼容性分析

- C++ ABI 与 `include/cann_ops_sparse.h` 保持一致。
- torch 适配层基于 PyTorch 2.7.1+cpu / torch_npu 2.7.1 实测。
- 使用调用方 NPU stream，不创建私有 stream。
- plan cache 策略按 SOC 编译期区分：A2 使用“输出指针不进 key + vecY 重绑定”优化，A3 使用包含输出指针的稳定 key；容量 8，超出整体清空重建。该设计同时保留 A2 性能收益与 A3 正确性。
- `SpmvPlan` 析构时销毁 Handle、SpMat、DnVec，并释放 workspace tensor。
- CANN 9.1.0 下 `spmv_split_kernel`、`spmv_epilogue_kernel`、`spmv_kernel_f32_i32_f32` 默认优化出现不稳定结果，CMake 固定这三个文件为 `-O0`；Host 和其余 kernel 保持 Release 优化。

# 验收证据与未闭合项

## 实测环境

| 项目 | Atlas A3 | Atlas A2 910B3 |
|---|---|---|
| SOC | `Ascend910_9382` / `ascend910_93` / DAV-2201 / `arch22` | `Ascend910B3` / DAV-2201 / `arch22` |
| 产品名 | `IT22HMDA_2_S` | 容器内不可查询，如实记录 |
| CANN | 9.1.0 | 9.1.0（V100R001C11SPC001B243） |
| Driver | 25.5.2 | 25.5.0（V100R001C23SPC005B219） |
| Firmware | 7.8.0.7.220 | 容器内不可查询，如实记录 |
| PyTorch | 2.7.1+cpu | 2.7.1+cpu |
| torch_npu | 2.7.1 | 2.7.1.post4 |
| 构建类型 | Release Host；三个稳定性敏感 kernel `-O0` | 同左，`build-a2-platform` 干净重构建，`SPMV_TORCH_PLAN_CACHE_REBIND_Y=ON` |
| 代码版本 | 平台条件实现 `cc65ca2`；证据提交 `bdd54af` | `bdd54af` |

## 测试结果

以下结果均来自 `ops-sparse` 分支 `ops-sparse` 的实机复验。A3 在平台条件实现
`cc65ca22657f352c89bee488ae0f7ccdca17f42a` 上完成全量复验；A2 910B3 在
`bdd54af71fb183dca9b8c2dd8eabe148f8d742da` 上完成平台分支干净重构建、全量复验和
10 进程稳定性加测；`0f522f540c34dcfc821ba7057991e1c69d77b5c1` 仅归档证据并同步
交付文档，未修改算子代码。

| 验证项 | Atlas A3 | Atlas A2 910B3 |
|---|---|---|
| C++ UT/ST | 220/220 | 220/220 |
| torch_npu + plan cache 语义 UT | 10/10 次独立进程通过 | 10/10 次独立进程通过 |
| 官方 accuracy | 200/200，最大绝对误差 `3.8147e-6`，最大相对误差 `1.0495e-5` | 200/200，误差与 A3 一致 |
| 官方 performance 有基线 case | 296/296，最低 ratio `0.7356`，中位数 `6.8713`，最大值 `34.5433` | 296/296，最低 ratio `0.5937`，中位数 `2.7398`，最大值 `15.5808` |
| 无 GPU 基线 case | 12 条 INT8→INT32 全部执行并如实记录 | 12 条 INT8→INT32 全部执行并如实记录 |
| 1000 条泛化 | 1000/1000，唯一投影 tuple 182 个 | 1000/1000，唯一投影 tuple 182 个 |
| 确定性与输入只读 | 9/9 | 9/9 |
| 资源生命周期 | 180 次调用无泄漏 | 180 次调用无泄漏 |
| Profiler hook 模式 | 30 次调用，180 个匹配事件 | 30 次调用，180 个匹配事件 |
| Profiler torch.mv 模式 | 30 次调用，180 个匹配事件 | 30 次调用，150 个匹配事件，`aten::mv` 端到端 median `139.38 us`、p90 `157.47 us` |
| 内存条件 | 最大 workspace `10,301,504 B`，小于目标平台最小 L2 `100,663,296 B` | 最大 workspace `10,301,504 B`，小于目标平台最小 L2 `100,663,296 B` |

证据索引：

- A3 平台分支全量复验：`docs/community_tasks/aclsparseSpMV_A2A3/reports/20260910_a3_platform_branch_full/`
- A2 910B3 平台分支复验：`docs/community_tasks/aclsparseSpMV_A2A3/reports/20260910_a2_platform_branch_recheck/`
- A2/A3 平台条件分支构建与实测：`docs/community_tasks/aclsparseSpMV_A2A3/reports/20260910_platform_branch_recheck/`
- 验收逐项审计：`docs/community_tasks/aclsparseSpMV_A2A3/aclsparseSpMV_A2A3_20260908/design_doc/acceptance_audit_20260909.md`

## 未闭合项

按任务书原文，当前不能将 A2 910B4 宣称为实机验证通过：

1. **A2 910B4 无实机环境**：`SOC_VERSION=ascend910b4` Release 构建通过，CMake 输出
   `SPMV_TORCH_PLAN_CACHE_REBIND_Y=ON`，与 A2 910B3 的平台策略一致。910B3 实机全链
   结果可作为同族 A2/arch22 参考证据，但不是 910B4 实测，不能替代任务书要求的
   910B4 功能、精度和性能验证。
设计文档 PR !1478 已提交：<https://gitcode.com/cann/cann-ops-competitions/merge_requests/1478>。当前仅等待评审反馈，不属于算子验收未闭合项。

A3 与 A2 910B3 的实机功能、精度、性能、内存、泛化、确定性、输入只读、资源生命周期
和 Profiler 证据均已完成；910B4 接入实机后应复跑与 910B3 相同的完整测试链路，并单独
记录 SOC、CANN、驱动、固件和代码提交版本。

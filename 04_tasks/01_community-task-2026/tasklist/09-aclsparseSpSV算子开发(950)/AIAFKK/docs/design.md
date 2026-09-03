# aclsparseSpSV 算子开发（Ascend 950PR / A5）设计文档

- 团队：AIAFKK
- 任务：《9月社区任务-aclsparseSpSV算子开发(950)》（任务书 `aclsparseSpSV_A5_task_doc.md`）
- 目标仓：`https://gitcode.com/cann/ops-sparse`，目标分支 `master`
- 参考基线：`sparse/spsv/arch35/`（DAV_3510）现有 FP32 实现

# 需求背景（required）

## 需求来源

本文对应 2026 年 9 月社区任务《aclsparseSpSV 算子开发任务书（A5）》，目标是在 Ascend 950PR（DAV_3510，`arch35`）上完善 `aclsparseSpSV`，在现有 FP32 能力之上补齐 `ACL_COMPLEX64` 类型支持，并同步交付 C++ UT/ST、性能脚本与文档。设计依据：

1. 任务书：`aclsparseSpSV_A5_task_doc.md`；
2. 任务测试包：`test_cases/aclsparseSpSV_testCase/`、`test_cases/common/`、`test_cases/baseline_results/`；
3. 设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`；
4. 目标代码仓：`https://gitcode.com/cann/ops-sparse`（master）；
5. 接口语义：任务书给出的 6 个 `aclsparseSpSV_*` 公开接口，语义对齐 cuSPARSE SpSV；
6. 精度标准：《生态算子开源精度标准》实验标准及任务书 3.2 节单标杆混合容差；
7. 软件环境：CANN 9.1.0 及后续配套版本，Ascend 950PR 实测。

本文只描述拟实现的技术方案与验证方法。功能、精度、性能、内存与 Profiler 数据在代码完成后于 Ascend 950PR 实测取得，设计阶段不把目标值写成已达成结果。

## 背景介绍

### SpSV 功能

SpSV（sparse triangular solve）求解稀疏三角线性系统：

```text
op(A) · y = alpha · x
```

其中 `A` 为 `[m,m]` 稀疏三角方阵（CSR/CSC/COO/SLICED_ELL 四种格式，I32 索引，base 0/1），`x`/`y` 为长度 `m` 稠密向量，`op` ∈ {N, T, H}（H 仅对 complex64 有共轭语义），`fill` ∈ {LOWER, UPPER}，`diag` ∈ {UNIT, NON_UNIT}。接口生命周期对齐 cuSPARSE：`createDescr → bufferSize → analysis(externalBuffer) → solve → updateMatrix → destroyDescr`，全程经 handle stream 异步执行，Y 可复用 X 的 Device values 指针原地求解，alpha 支持 Host/Device pointer mode。

### ops-sparse 现状分析（基线 = master `sparse/spsv/arch35/`）

基线代码 5 个文件（`spsv.h` 79 行 / `spsv_host.cpp` 936 行 / `spsv_kernel.cpp` 1819 行 / `spsv_kernel.h` 121 行 / `spsv_tiling_data.h` 74 行），已有能力与本任务差距如下：

| 层级 | 基线现状 | 本任务必须补齐 |
| --- | --- | --- |
| computeType/valueType | Host 校验 `computeType != ACL_FLOAT` 与 `valueType != ACL_FLOAT` 直接拒绝（`ValidateSpSVCommonParams`） | 放行 `ACL_COMPLEX64`；A/X/Y/computeType/alpha 一致性校验按 dtype 分支 |
| Kernel 值类型 | 所有 `__gm__ const float *values` 硬编码 float；`SpsvSolveRow` 的 `sum = alpha*x - Σ a[p]*y[col]` 为实数运算 | 值路径模板化 `VecT ∈ {float, complex64}`；复数乘/减/对角除/共轭 |
| opA 语义 | N/T 经 `SpsvTransposeCsr`（值直拷）+ LOWER/UPPER 翻译实现；H 无共轭 | complex64 下 T 保持值直拷；H 路径在转置/求解中对 values 取共轭 |
| alpha | `SpsvTilingData.alpha` 为 float；`ReadHostScalar` 只读 float；Device mode 传指针 | 按 dtype 读取（complex64 8B）；tiling 承载实部虚部或设备指针 |
| workspace 布局 | `ComputeWorkspaceOffsets` 中 csrValues/transValues 子分配按 `sizeof(float)` 计 | 按 `elemSize`（4B/8B）参数化，保持首块 512B、内部 64B 对齐策略不变 |
| 精度 | FP32 单标杆（float64 golden，rtol=2^-10/atol=2^-16/A=1e-2，匹配率≥0.99，误差上限 max(A, 32·ULP)） | complex64 复用同一标杆，实部、虚部分别判定 |
| 测试 | 无专项 UT/ST | C++ UT/ST：dtype/format/fill/diag/op/生命周期/原地/pointer mode/异常全组合，接入任务包 ATK 用例 |
| 文档 | 无接口文档 | README 更新 + 接口说明 + 限制说明 |

基线已具备且本设计直接复用的能力（不重复开发）：四格式→workspace CSR 归一（`SpsvBuildCsrFromCoo/SlicedEll/WorkspaceCsr`）、level-set 依赖分析与确定性调度（`SpsvComputeLevels`、`SpsvAnalysisSerial/Parallel/FinalPhase`）、对角定位与有效计数（`SpsvComputeDiagAndValidCount`）、深 level 单核回退启发式（`ComputeNumBlocks`）、cuSPARSE 式描述符属性缓存（`CacheMatrixAttributes`）、`nnz > INT32_MAX` 的 perm 类型自适应及 `SPSV_FORCE_PERMT_64` 调试开关。

### 任务测试包分析

任务包提供不可删减的验证输入：`accuracy_cases.json`（精度泛化用例，含 70% `[-1,1]` 均匀 + 20% `N(0,1)` + 10% 单位对角/零对角/特殊值的取值分布）、`performance_cases.json`（P-01/P-02/P-03 三档规模，锚点 Llama3.1-70B / Qwen3-235B / DeepSeek-V3 维度）、GPU 基线 `baseline_results/gpu_full_results.tsv`（标杆接口 Event 计时 median_us）、内存采集与对比脚本（`collect_sparse_ops_{gpu,npu}_memory.py`、`compare_sparse_ops_memory.py`）。设计阶段已通读上述清单，测试方案（§测试方案）直接建立在该测试包之上，不另行发明用例体系。

# 需求分析（required）

## 需求描述

在 `sparse/spsv/arch35` 现有 FP32 实现基础上，补齐 `ACL_COMPLEX64` 的端到端支持：公开接口签名不变（与 `include/cann_ops_sparse.h` 现有原型逐字一致），Host 校验/生命周期/workspace 按 dtype 参数化，Kernel 值路径模板化并实现复数运算（乘、减、对角除、共轭转置），精度满足任务书 3.2 节 complex128 单标杆，性能对 FP32 不回归、complex64 达到 0.3 倍 GPU 标杆，交付 C++ UT/ST 与性能/内存脚本证据。

## 需求拆解

1. Host：`ACL_FLOAT | ACL_COMPLEX64` 双 dtype 校验与一致性检查（A/X/Y/alpha/computeType）；
2. Kernel：`VecT` 值类型模板展开（`SpsvSolveRow`、COO/SELL 构建、转置拷贝、diag 求解）；
3. complex64 的 H（共轭转置）路径：`opA == CONJUGATE_TRANSPOSE` 时 values 参与处取共轭；
4. alpha Host/Device pointer mode 按 dtype 读取与传递；
5. workspace 布局 `elemSize` 参数化（4B/8B），偏移对齐策略不变；
6. NON_UNIT 复数对角除法与 UNIT 语义；零/缺失对角按接口定义传播 INF/NAN；
7. 精度：complex128 golden，实部虚部分别 rtol=2^-10 / atol=2^-16 / A=1e-2，匹配率≥0.99，逐元素误差 ≤ max(A, 32·ULP)，重复执行 bitwise deterministic；
8. 性能：预热 10 + 采样 30，报告 median/P90，FP32 不低于现状基线，两 dtype 均达 0.3×GPU 标杆；
9. C++ UT/ST 覆盖：四格式 × fill/diag × N/T/H × dtype × 生命周期 × 原地 × pointer mode × 异常矩阵；
10. 交付 README/接口文档/限制说明与 profiler 证据。

# 详细设计（required）

## 总体架构

维持基线的分层不变，值类型维度正交于索引类型维度：

```text
include/cann_ops_sparse.h        公开接口（不变）
sparse/spsv/
├── spsv.h / spsv_tiling_data.h  公共常量、对齐、tiling（dtype 字段扩展）
├── arch35/
│   ├── spsv_host.cpp            生命周期/校验/tiling/workspace（elemSize 参数化）
│   └── spsv_kernel.cpp/.h       SIMT kernel（VecT 模板化）
test/spsv/arch35/                C++ UT/ST（fp32 回归 + complex64 专项）
test_cases/aclsparseSpSV_testCase/  任务包（随 PR 提交，不改）
```

## dtype 与值类型抽象

Kernel 侧新增最小复数值类型（SIMT 环境不引入运行时依赖）：

```cpp
struct alignas(8) SpsvComplex { float re, im; };   // 内存布局与 aclsparse complex64 一致
```

值参与运算的函数族模板化为 `template <typename VecT>`：

- `SpsvSolveRow<RowPtrT, ColIndT, VecT, FORWARD, NON_UNIT, PermT>`：`sum = alpha*x` 与 `sum -= a[p]*y[col]` 的乘/减按 VecT 特化；NON_UNIT 对角除法实数走乘 `1/d`（保持基线策略），复数用一次共轭乘加一次实数除 `(a·conj(d)) / (d·conj(d))`，分母为零时按接口定义产生 INF/NAN；
- `SpsvBuildCsrFromCoo / SlicedEll / WorkspaceCsr / TransposeCsr` 等搬运类函数：增加 `CONJ` 布尔模板参，complex64 + H 时在写入侧取共轭，float + H 编译期拒绝（校验层已挡）；
- 编译期用 `if constexpr` 区分实/复路径，float 分支与基线代码生成的指令序列一致，保证 FP32 性能零回归。

模板展开维度控制：索引 3 组合（I32/I32、I64/I32、I64/I64）× perm 2 × VecT 2 × CONJ 2，显式实例化清单收敛在 `spsv_kernel.h` 的调度表，避免全组合膨胀。

## Host 设计

1. **校验**：`ValidateSpSVCommonParams` 的 dtype 段改为 `computeType ∈ {ACL_FLOAT, ACL_COMPLEX64}` 且 `== valueType == vecX->type == vecY->type`；alpha 非空且 pointer mode 与 handle 一致。其余校验（方阵、rows ≤ INT32_MAX、I32+nnz>INT32_MAX 拒绝、索引组合、格式、base、alg）保持不变。
2. **tiling**：`SpsvTilingData` 增加 `int32_t isComplex` 与复数 alpha 载荷（Host mode 直接内嵌 `float alphaRe, alphaIm`；Device mode 沿用 `alphaDevicePtr`，kernel 按 isComplex 解引用 4B/8B）。
3. **workspace**：`ComputeWorkspaceOffsets` 与 `ComputeWorkspaceSize` 增加 `size_t elemSize` 入参（float=4 / complex=8），仅 `csrValues`、`transValues` 两块按 elemSize 缩放；首块 512B（`kAlign`）与内部 64B（`kInternalAlign`）对齐、nnz==0 全 -1 哨兵等既有约定不变。`BufferSize` 返回值对同 pattern 的 fp32/complex64 分别稳定。
4. **生命周期**：`BufferSize`/`Analysis` 允许 vecX/vecY 描述符为 NULL 的既有行为保留；Analysis 缓存中增加 `cachedIsComplex`，Solve/UpdateMatrix 校验缓存一致性；`UpdateMatrix`（GENERAL/DIAGONAL）不改 pattern 与 dtype，值替换后按基线路径刷新关联状态。
5. **错误码**：全部沿用基线 `ACL_SPARSE_STATUS_*` 语义，dtype 不一致返回 `NOT_SUPPORTED`，参数/指针错误返回 `INVALID_VALUE`，与本设计新增分支一一对应写入接口文档。

## 精度与确定性

- complex64 全链路（含 H 共轭、对角除、alpha 缩放）以 complex128 golden 逐元素比较，实部虚部分别套用 `|actual-golden| ≤ atol + rtol·|golden|`（rtol=2^-10，atol=2^-16），匹配率 ≥0.99，且逐元素绝对误差 ≤ max(1e-2, 32·ULP(golden))；
- level-set 调度、行内固定遍历序（`validCount` 前缀 + 尾部清扫序）与基线一致，保证同输入重复执行 bitwise deterministic；
- 单元对角/非单位对角、零对角与缺失对角：UNIT 直接跳过除法；NON_UNIT 零/缺失对角按任务书约定传播 INF/NAN 并在测试报告记录。

## 性能方案

- 基线的深 level 单核回退与线程数分级（`ComputeNthreads`）对两 dtype 通用；
- complex64 有效带宽减半（元素 8B）：搬运类函数仅拷贝宽度变化；求解核心循环的复数乘（4 mul + 2 add）与除法（共轭乘 + 实除）为 SIMT 标量路径，不做向量化改写，先保证正确与确定性，性能依赖 0.3×GPU 标杆的宽松余量（P-01/P-02/P-03 目标线分别为 GPU median 的 30%）；
- FP32 路径编译产物与基线等价（`if constexpr` 分支不参与 float 实例化），预期性能不回归，实测对比后写入自测报告。

## 内存方案

workspace 值块 ×2（csr/trans）按 elemSize 缩放，索引/perm/level 块不变；满足任务书 3.4 的两条判据之一：与 GPU 同 case 对比 `extra_peak ≤ 50%`，或 workspace 绝对值 ≤ L2 容量。使用任务包 `performance_cases.json` 同 case 走 `collect/compare` 脚本出数。

## 与 A2/A3 的联合回归

Host 公共层（校验/描述符缓存/workspace 布局/tiling 字段）置于 `sparse/spsv/` 顶层共享，arch35 差异层仅 kernel 与平台常量；配套联合回归用例（同 case 集 × 双 dtype）跑通后作为两任务的共同交付证据（另见 A2/A3 版设计文档）。

# 测试方案

1. **精度（ATK + C++ UT）**：任务包 `accuracy_cases.json` 全量 + `run_accuracy_atk.sh` 复现路径；补充 C++ 用例覆盖四格式 × LOWER/UPPER × UNIT/NON_UNIT × N/T/H × fp32/complex64，重点：H 共轭语义（复矩阵 T≠H）、原地（Y 复用 X values）、base 0/1、未排序 COO/SELL、空行与 m/nnz=0/1、零/缺失对角的 INF/NAN、重复执行确定性。
2. **生命周期/异常**：NULL 描述符的 BufferSize/Analysis 合法性；Solve 阶段 NULL/不匹配拒绝；Analysis→Solve 参数变更触发重新 Analysis；externalBuffer 过早释放的错误路径；UpdateMatrix GENERAL/DIAGONAL 后复用 Solve；非法枚举/索引/dtype 全量错误码断言。
3. **性能**：`performance_cases.json` P-01~P-03 × 两 dtype，预热 10/采样 30，报告 median/P90 与 Analysis、Solve 分段；对 `gpu_full_results.tsv` 计算 0.3×达标线；CANN Profiler 证据（msprof 采集 kernel 耗时）。
4. **内存**：同 case 双端采集（GPU 基线沿用任务包数据，NPU 侧实跑），`compare_sparse_ops_memory.py` 出 `input/peak/extra_peak` 对比表。
5. **环境**：Ascend 950PR + CANN 9.1.0+，记录 CANN/驱动/SOC 版本；README 提供编译（CMake 目标）与逐步复现命令。

## 风险与对策

- **风险**：SIMT 复数标量路径性能余量不足（尤其 P-03 262k/3.9M nnz）。对策：0.3× 线对应 GPU cuSPARSE 的 30%，基线 FP32 已证明 level-set 调度在该规模可用；若 complex64 不达标，优先优化行内乘减循环的寄存器占用与 `#pragma unroll` 深度，其次评估对角倒数预计算进 workspace。
- **风险**：complex64 布局与 aclsparse 侧不一致。对策：`SpsvComplex` 用 `alignas(8)` 显式布局并在 UT 中 `static_assert(sizeof == 8)` + 与 `cann_ops_sparse.h` 常量交叉验证。

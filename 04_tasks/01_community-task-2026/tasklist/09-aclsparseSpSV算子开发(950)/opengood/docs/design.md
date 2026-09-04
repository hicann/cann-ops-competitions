# aclsparseSpSV 算子开发（Ascend 950 / A5）设计文档

- 团队：opengood
- 任务：《9月社区任务-aclsparseSpSV算子开发(950)》（任务书 `aclsparseSpSV_A5_task_doc.md`）
- 目标仓：`https://gitcode.com/cann/ops-sparse`，目标分支 `master`
- 参考基线：`sparse/spsv/arch35/`（DAV_3510）现有 FP32 实现（3029 行）

# 需求背景（required）

## 需求来源

1. 任务书：`aclsparseSpSV_A5_task_doc.md`（2026-09 版）；
2. 任务测试包：`test_cases/aclsparseSpSV_testCase/`、`test_cases/common/`、`test_cases/baseline_results/`（官方原样，随 PR 提交）；
3. 设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`；
4. 目标代码仓：`cann/ops-sparse` master（摸底基线 commit a5b24e8，其后 spsv 目录无变更）；
5. 接口语义：`include/cann_ops_sparse.h` 已声明的 6 个 `aclsparseSpSV_*` 公开接口，语义对齐 cuSPARSE SpSV；
6. 精度标准：《生态算子开源精度标准》实验标准 + 任务书 §3.2 单标杆混合容差；
7. 软件环境：CANN 9.1.0 及后续配套版本，Ascend 950PR/950DT 实测。

本文描述拟实现的技术方案与验证方法；功能、精度、性能、内存与 Profiler 数据在代码完成后于 Ascend 950PR 实测取得，设计阶段不把目标值写成已达成结果。

## 背景介绍

### SpSV 功能

求解稀疏三角线性方程组：

```text
op(A) · Y = alpha · X
```

A 为 `[m,m]` 稀疏三角方阵（CSR/CSC/COO/SLICED_ELL 四格式，接受未排序坐标，索引 I32/I64、base 0/1）；
X/Y 为长度 m 稠密向量，Y 可复用 X 的 Device values 指针原地求解；`op ∈ {N, T, H}`（H 仅对
complex64 有共轭语义，FP32 下 H≡T）；`fill ∈ {LOWER, UPPER}`；`diag ∈ {UNIT, NON_UNIT}`；
alpha 标量支持 Host/Device pointer mode。生命周期对齐 cuSPARSE：

```text
createDescr → bufferSize → analysis(externalBuffer) → solve(可重复)
                                      → updateMatrix → solve → destroyDescr
```

全程经 handle stream 异步执行；Analysis→Solve 期间描述符/参数/externalBuffer 一致性由调用方保证、
实现侧校验；未声明支持的场景（NON_UNIT 缺失/零对角）按接口定义传播 INF/NAN。

### ops-sparse 现状分析（基线 = master `sparse/spsv/arch35/`）

基线 5 文件（`spsv.h` 79 / `spsv_host.cpp` 936 / `spsv_kernel.cpp` 1819 / `spsv_kernel.h` 121 /
`spsv_tiling_data.h` 74 行），能力与差距：

| 层级 | 基线现状 | 本任务补齐 |
| --- | --- | --- |
| computeType/valueType | `ValidateSpSVCommonParams` 只认 `ACL_FLOAT`，其余拒绝 | 放行 `ACL_COMPLEX64`，A/X/Y/alpha/computeType 一致性按 dtype 校验 |
| Kernel 值类型 | values 硬编码 float；solve 行前推为实数乘减 | 值路径模板化 `VecT ∈ {float, SpsvComplex}`；复数乘/减/对角除/共轭 |
| opA | N/T 经 `SpsvTransposeCsr`（值直拷）；H 无共轭 | T 值直拷不变；H 在搬运写入侧取共轭（CONJ 模板参） |
| alpha | `SpsvTilingData.alpha` 为 float；Device mode 存 `alphaDevicePtr` | tiling 增 `isComplex` + 复数 alpha 载荷；kernel 按 dtype 解引用 |
| workspace | `ComputeWorkspaceOffsets` 值块按 `sizeof(float)` 计 | `elemSize` 参数化（4B/8B）；首块 512B（kAlign）、内部 64B（kInternalAlign）对齐不变 |
| 测试 | CSV 参数驱动 UT（fp32） | csv 扩 dtype 维度 + complex64 专项 + 边界/生命周期/异常 |
| 评测接入 | 仓内无 `ops_sparse_test` 任何基础设施（grep 零命中） | 新增 Python hook 胶水层（见详细设计 §hook） |

基线已具备且直接复用（不重复开发）：四格式→workspace CSR 归一（`SpsvBuildCsrFromCoo/SlicedEll/…`）、
level-set 依赖分析与确定性调度（`SpsvComputeLevels` + analysis 三段式 kernel 启动）、对角定位与有效计数、
深 level 单核回退启发式、cuSPARSE 式描述符属性缓存、nnz>INT32_MAX 的 perm 类型自适应。

### 任务测试包分析

- 精度：`accuracy_cases.json` 200 条，值分布 70% `[-1,1]` 均匀 + 20% `N(0,1)` + 10% 单位/零对角/特殊值，
  row_pattern 覆盖 diagonal/banded/uniform/highly_imbalanced/one_long_row/skewed/random/power_law；
- 性能：`performance_cases.json` 206 条（P-01/P-02/P-03 三档规模 × base0/1 + 200 extra），
  锚点 Llama3.1-70B / Qwen3-235B / DeepSeek-V3；
- 标杆：`baseline_results/gpu_full_results.tsv`（GPU Event median_us）；
- 内存：`collect_sparse_ops_{gpu,npu}_memory.py` + `compare_sparse_ops_memory.py`（精确判定规则）；
- 评测脚本对算子侧的硬契约是三 hook（下节），缺失即 `RuntimeError: register spsv_analysis_npu, …`。

# 需求分析（required）

## 需求描述

在 `sparse/spsv/arch35` FP32 基线上补齐 `ACL_COMPLEX64` 端到端支持：公开接口签名逐字不变；
Host 校验/生命周期/workspace 按 dtype 参数化；Kernel 值路径模板化并实现复数运算（乘、减、对角除、
共轭转置）；交付评测 hook 胶水层使官方精度/性能/内存脚本可运行；精度满足 complex128 单标杆
（实虚分别判定）、FP32 零回归；性能两 dtype 均达 0.3×GPU 标杆；交付 C++ UT/ST、文档与 4 项交付件。

## 需求拆解

1. Host：双 dtype 校验与一致性（A/X/Y/alpha/computeType）；tiling 复数 alpha 载荷；workspace elemSize；
2. Kernel：`VecT` 模板化（solve 行前推、COO/SELL 构建、转置搬运、diag 求解）+ CONJ 布尔模板参；
3. complex64 H 路径：`opA == CONJUGATE_TRANSPOSE` 时 values 参与处取共轭；
4. alpha Host/Device pointer mode 按 dtype 读取与传递；
5. NON_UNIT 复数对角除法与 UNIT 语义；零/缺失对角传播 INF/NAN；
6. hook 胶水层：`torch.ops.ops_sparse_test` 注册 `spsv_analysis_npu/spsv_update_npu/spsv_npu`；
7. 精度：complex128 golden，实虚分别 rtol=2⁻¹⁰/atol=2⁻¹⁶/A=1e-2，匹配率≥0.99，
   逐元素 ≤ max(A, 32·ULP)，重复执行 bitwise deterministic；
8. 性能：预热 10 + 采样 30，median/P90，FP32 不回归，两 dtype ≥0.3× 标杆；
9. C++ UT/ST：四格式 × fill/diag × N/T/H × dtype × 生命周期 × 原地 × pointer mode × 异常；
10. 文档：README 能力矩阵 + 接口说明 + 限制说明；设计文档/自测报告/复现说明。

# 详细设计（required）

## 算子分析

- 数学公式：`op(A)·Y = alpha·X`，level-set 前代/回代按层前推：`y[i] = (alpha·x[i] − Σ_{j<i} a[i,j]·y[j]) / a[i,i]`
- 支持数据类型：`ACL_FLOAT`、`ACL_COMPLEX64`（computeType=valueType=X/Y/alpha 一致）
- 支持形状：方阵 m×m，m≥0；CSR/CSC/COO/SLICED_ELL；I32/I64（含混排）；base 0/1

## 算子实现

### host 侧设计

1. **校验**：`ValidateSpSVCommonParams` dtype 段改为 `computeType ∈ {ACL_FLOAT, ACL_COMPLEX64}` 且
   `== valueType == vecX->type == vecY->type`；alpha 非空且 pointer mode 与 handle 一致；其余校验
   （方阵、rows≤INT32_MAX、I32+nnz>INT32_MAX 拒绝、索引组合、格式、base、alg）保持不变。
   dtype 不一致返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`，参数错误返回 `INVALID_VALUE`。
2. **tiling**：`SpsvTilingData` 新增 `float alphaImag`（与既有 `float alpha` 组成复数载荷）与
   `int32_t isComplex`（0=FP32, 1=complex64）。Host mode 下实/虚部分别经 `ReadHostScalar` 扩展读取；
   Device mode 沿用 `alphaDevicePtr`，kernel 按 `isComplex` 解引用 4B/8B。tiling 总尺寸复核不超
   kernel 传参上限（当前 ~160B，新增 8B 后仍在界内；若实测超限则把 `idxBase/permType` 等收进位压缩）。
3. **workspace**：`ComputeWorkspaceOffsets`/`ComputeWorkspaceSize` 增 `size_t elemSize` 入参，仅
   `csrValues`、`transValues` 两块按 elemSize 缩放；首块 512B 与内部 64B 对齐、nnz==0 哨兵等约定不变。
4. **生命周期**：`BufferSize/Analysis` 允许 vec 描述符为 NULL 的既有行为保留；Analysis 缓存新增
   `cachedIsComplex`，Solve/UpdateMatrix 校验一致性；`UpdateMatrix`（GENERAL/DIAGONAL）不改 pattern
   与 dtype，newValues 长度按 elemSize 校验（GENERAL: nnz；DIAGONAL: 对角元素数），值替换后走基线
   刷新路径，确定性不变。

### kernel 侧设计

1. **复数值类型**：`struct alignas(8) SpsvComplex { float re, im; };`——与 aclsparse complex64 内存布局
   一致，UT 中 `static_assert(sizeof==8)` 并与 `cann_ops_sparse.h` 常量交叉验证。
2. **值路径模板化**：值参与运算的函数族 `template <typename VecT>`：行前推
   `SpsvSolveRow<RowPtrT, ColIndT, VecT, FORWARD, NON_UNIT, PermT>`（`sum = alpha·x − Σ a[p]·y[col]`
   按 VecT 特化）、搬运/归一函数（COO/SELL→CSR、`SpsvTransposeCsr` 及并行版）增加 `CONJ` 布尔
   模板参（complex64+H 时写入侧取共轭；float+H 编译期排除，校验层已挡 FP32 H≡T）。
   `if constexpr` 区分实/复路径，float 实例化与基线指令序列一致 ⇒ **FP32 零回归**。
3. **复数算术**：cmul（4mul+2add）、csub；NON_UNIT 对角除 `(a·conj(d)) / (d·conj(d))`
   （分母 = re²+im² 一次实除）；分母为零/对角缺失按接口定义产生 INF/NAN；H 路径取共轭 cconj。
4. **模板膨胀控制**：索引 3 组合（I32/I32、I64/I32、I64/I64）× perm 2 × VecT 2 × CONJ 2，
   显式实例化清单收敛在 `spsv_kernel.h` 调度表。
5. **性能取向**：首轮复数路径为 SIMT 标量（正确性优先），搬运类仅拷贝宽度 4B→8B 变化；
   budget=标杆/0.3（P-01≤≈134.9ms / P-02≤≈315.8ms / P-03≤≈743.0ms）余量充足；
   若不达标按序：行内乘减 `#pragma unroll` 与寄存器占用 → 层合并/块重组（见相关工作）→
   对角倒数预计算进 workspace。

### 评测 hook 胶水层设计（新增，官方评测脚本硬依赖）

自 `operator_adapter.py` 提炼的契约（编码值以源码为准）：

```text
spsv_analysis_npu(row_offsets, col_indices, values, op, fill, diag, base, pointer_mode) -> plan
    op: 0=N/1=T/2=H; fill: 0=lower/1=upper; diag: 0=nonunit/1=unit;
    base: 0/1; pointer_mode: 0=host/1=device
spsv_update_npu(plan, values, update_kind) -> plan | None     # update_kind: 0=general/1=diagonal
spsv_npu(plan, x, alpha) -> y                                  # alpha: host=Python 标量(可复数)，
                                                                # device=0 维 compute-dtype tensor
```

评测流程语义（每性能采样）：analysis 仅一次 → `values` 原地先恢复初始值再写入声明修改 →
`update(plan, values, kind)`（返回 None 表示 plan 原地保留）→ `solve(plan, x, alpha)`；
精度对比只看 solve 输出（dense tensor）。`--allow-reference-fallback` 仅 m≤4096 冒烟，P 场景必须真 hook。

实现（独立扩展模块 `ops_sparse_test_hooks`，随交付件提供源码）：

1. `TORCH_LIBRARY(ops_sparse_test, m)` 注册三 schema（NPU 后端 dispatch key）；
2. `plan` 用 `torch::CustomClassHolder` 封装：`aclsparseSpSVDescr_t` + analysis 绑定的
   externalBuffer（`torch::empty` 分配、tensor 持有生命周期、计入 allocator 峰值满足内存口径）+
   matA/vecX/vecY 描述符 + 缓存属性（op/fill/diag/base/pointer_mode/dtype），跨三次调用持有；
3. analysis hook：构 CSR 描述符 → `bufferSize` → 分配 workspace → `analysis`；
   update hook：`aclsparseSpSV_updateMatrix`（GENERAL/DIAGONAL 映射）；
   solve hook：alpha host/device 分派 → `solve` → 返回 y；Y 原地允许共享 X storage；
4. Python 侧 `import` 即注册，无评测脚本侵入；构建参照 vllm-ascend csrc 的 NPU 扩展工程组织；
5. **hook 定位**：仅评测专用 ABI，不构成公开 Python API（公开交付接口仍是 `cann_ops_sparse.h`）；
6. **schema 细节（离线影子台实测，见下）**：
   - `spsv_npu` 的 alpha 需**双 overload**（`Scalar` 与 `Tensor` 各一）——host 模式传 Python
     标量（float/complex）、device 模式传 0 维 compute-dtype tensor，单一 `Scalar` schema 会拒收
     tensor（dispatcher 按实参类型自动选择 overload）；
   - determinism 判定必须用**位级比较**（byte view 相等，NaN==NaN）：官方 performance extra
     用例数值混沌（complex128 golden 量级已达 ~1e306，fp32 必然 inf/NaN），值相等会把确定性的
     NaN 误判为不确定；
   - 官方 performance 用例只计时，**精度结论只在小 m 的 ATK 用例上判定**（m∈{2,3,7,17,31,63}）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / 950DT（A5，SOC=ascend950，arch35/DAV_3510） | √ |

## 算子约束限制

- complex64 的 H（共轭转置）有真实共轭语义；FP32 下 H 等价 T（沿用基线）；
- 核心计算全部在 NPU kernel 执行，无 CPU fallback 路径；
- 未声明支持的场景（NON_UNIT 缺失/零对角）传播 INF/NAN 并在文档注明；
- workspace 生命周期：异步 Solve 完成前不得释放 externalBuffer。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | FP32→float64、complex64→complex128 单标杆（`torch.linalg.solve_triangular`）；实/虚分别 `\|a-g\| ≤ 2⁻¹⁶ + 2⁻¹⁰·\|g\|`，匹配率≥0.99，逐元素 ≤ max(1e-2, 32·ULP)；重复执行 bitwise deterministic | 任务书 §3.2 + 生态算子精度标准 |
| 性能 | 倍率 = GPU(H100 cuSPARSE) Event median / NPU 同范围 median ≥ 0.3（两 dtype）；预热 10、采样 30；报告 median/P90/Analysis/Solve/workspace + Profiler 证据 | 任务书 §3.3 |
| 内存 | 双路径精确规则（compare_memory.py）：I/O>500MB ⇒ (NPU峰值−GPU峰值)/GPU峰值 ≤5%；否则 workspace ≤ L2 Cache；满足其一即通过 | 测试包脚本（严于任务书 50% 表述，从严执行并在报告注明） |

## 测试方案

1. **精度**：`run_accuracy_atk.sh` 跑全量 200 条（真 hook）；C++ UT 用 `spsv_test.csv` 扩 dtype 维度
   （fp32/complex64 × attr 组合抽样），golden 对齐 complex128；重点：H 共轭（复矩阵 T≠H 专项）、
   原地（Y 复用 X）、base0/1、未排序 COO/SELL、空行与 m/nnz=0/1、零/缺失对角 INF/NAN、
   UpdateMatrix GENERAL/DIAGONAL 后复解、重复执行确定性。
2. **生命周期/异常**：NULL vec 描述符的 BufferSize/Analysis；Solve 阶段 NULL/不匹配拒绝；
   Analysis→Solve 参数变更触发重新 Analysis；externalBuffer 过早释放错误路径；非法枚举/索引/dtype
   全量错误码断言。
3. **性能**：`performance_cases.json` 全量；NPU case 与 `gpu_full_results.tsv` 按**唯一 id**
   （如 `spsv-P-01-base0`）关联，并逐项核对 dtype/m/nnz/base/op/fill/diag/pointer_mode/update/alpha
   及 TSV 的 `api`/`median_us`/`solve_mean_us`/`workspace_bytes` 列，不做 shape 模糊匹配；
   NPU 只与 CPU Golden 比精度，GPU 仅作性能标杆；msprof/torch_npu.profiler 采集 kernel 级耗时与
   无 fallback 证据。
4. **内存**：同 case 双端采集，`compare_sparse_ops_memory.py` 出 input/peak/extra_peak 或
   workspace vs L2 对比表（`--workspace-results`/`--l2-cache-bytes`）。
5. **环境**：记录 npu-smi/CANN/torch_npu/atc 版本；README 提供编译与逐步复现命令。

## 兼容性分析

1. 公开接口签名逐字不变；`include/cann_ops_sparse.h` 预计零改动（枚举齐备）；若复数场景确需新增
   类型别名，先评估对 arch22 及其他算子影响再提；
2. FP32 路径行为与性能零回归（`if constexpr` 隔离 + 全量既有 csv UT 保绿）；
3. 与 A2/A3（arch22）共享 Host 逻辑的差异层同步回归；
4. 不引入进程全局可变缓存、不增跨算子锁，并发能力不退化。

# 风险与对策

- **风险**：complex64 标量路径在 P-03（m=262,144/nnz=3,932,160）性能余量不足。对策：budget=标杆/0.3
  宽松；不达标按 unroll/寄存器 → 层合并 → 对角倒数预计算顺序优化，并保留 msprof 证据。
- **风险**：tiling 传参超限。对策：新增字段仅 8B 且已复核；超限时位压缩/下沉 workspace，不影响接口。
- **风险**：hook 与评测脚本契约偏差（update 返回 None 语义、device alpha 0 维 tensor）。对策：以
  `operator_adapter.py` 源码为准逐条对照，UT 先行 mock 契约测试。
- **风险**：内存口径歧义（任务书 50% vs 脚本 5%）。对策：按 compare_memory.py 精确规则执行并在
  自测报告注明依据。

# 相关工作（性能 contingency 参考）

并行 SpTRSV 经典方案（仅在不达标时取思路，本设计默认沿用基线 level-set）：线程级同步-free
（CapelliniSpTRSV）、块算法（recblock-sptrsv, ICPP'20）、自动调优（AG-SpTRSV）、同步-free CSC 系列。


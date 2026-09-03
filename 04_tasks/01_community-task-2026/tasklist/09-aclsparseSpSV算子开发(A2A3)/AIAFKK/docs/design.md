# aclsparseSpSV 算子开发（Atlas A2/A3，arch22/DAV_2201）设计文档

- 团队：AIAFKK
- 任务：《9月社区任务-aclsparseSpSV算子开发(A2A3)》（任务书 `aclsparseSpSV_A2A3_task_doc.md`）
- 目标仓：`https://gitcode.com/cann/ops-sparse`，目标分支 `master`
- 语义基线：`sparse/spsv/arch35/`（DAV_3510）现有 FP32 实现（接口/生命周期/精度语义同源）
- 移植参照：`sparse/spmm/arch22/`、`sparse/spmv/arch22/`（ops-sparse 已验证的 arch22 工程范式）

# 需求背景（required）

## 需求来源

本文对应 2026 年 9 月社区任务《aclsparseSpSV 算子开发任务书（A2/A3）》，目标是在 Atlas A2/A3（`arch22`/DAV_2201）上建立 `aclsparseSpSV` 的完整 Host/Kernel 路径，支持 `ACL_FLOAT` 与 `ACL_COMPLEX64`、CSR/CSC/COO/SLICED_ELL 四格式、N/T/H、LOWER/UPPER、UNIT/NON_UNIT 与 cuSPARSE 式全生命周期，交付到 `ops-sparse` 并与 950 版共享 Host 公共层、完成联合回归。设计依据：

1. 任务书：`aclsparseSpSV_A2A3_task_doc.md`；
2. 任务测试包：`test_cases/`（精度/性能/内存用例、GPU 基线、采集脚本）；
3. 设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`；
4. 语义参照：`sparse/spsv/arch35/`（接口逐字一致、行为语义同源）；
5. 工程参照：`sparse/spmm/arch22/`（host/kernel 文件组织）、`sparse/spmv/arch22/`；
6. 精度标准：任务书 §3.2 单标杆混合容差；
7. 软件环境：CANN 9.1.0 及后续配套版本；硬件覆盖 910B3、910B4 与任务环境提供的 A3 型号（任务书 §3.1），性能自验优先在 910B3 系完成。

## 背景介绍

### SpSV 功能与接口

`op(A)·y = alpha·x`，`A` 为 `[m,m]` 三角稀疏方阵，接口为 `aclsparseSpSV_createDescr/destroyDescr/bufferSize/analysis/solve/updateMatrix` 六函数，签名与 `include/cann_ops_sparse.h` 现有原型逐字一致，生命周期、NULL 描述符规则、原地求解、Host/Device pointer mode 语义与任务书逐条对应。全部主计算与格式相关计算由 NPU Kernel 在调用 stream 上异步执行，**无 CPU fallback 路径**。

### ops-sparse 现状分析

| 层级 | 现状 | 本任务 |
| --- | --- | --- |
| `sparse/spsv/` | 仅 `arch35/`（FP32，SIMT 编程模型） | 新建 `arch22/` 完整路径；Host 公共层上提共享 |
| arch22 工程范式 | `spmm/arch22`、`spmv/arch22` 已验证（Ascend C 向量范式） | SpSV 按同范式落地；`__simt_*` 设施不得用于 arch22 |
| dtype | arch35 仅 FP32 | arch22 支持 FP32 + complex64 |
| 测试 | 无 arch22 SpSV 测试 | C++ UT/ST（含生命周期/异常矩阵）+ 联合回归 |

基线（arch35）中与本任务同源的**语义资产**：CSC 经 LOWER/UPPER+op 翻译、level-set 依赖分析与确定性调度、对角定位/有效计数、未排序坐标确定性规范化、描述符属性缓存与 cuSPARSE 生命周期约定。实现层全部按 arch22 的 Ascend C 编程模型重写。

### 任务测试包分析

精度 200 条（ATK）与性能 206 条用例与采集脚本同任务包交付；性能目标为任务书 §3.3 的 **0.25×GPU 标杆线**；内存走任务书 §3.4 双判据：具备等价 GPU 接口且 IO>500MB 的 case，额外内存 ≤ GPU 使用内存总量的 50%；无等价 GPU 接口的 case，workspace ≤ 目标硬件 L2 容量。

# 需求分析（required）

## 需求描述

在 `ops-sparse` 中新建 `sparse/spsv/arch22/`，交付与 arch35 语义一致的 `aclsparseSpSV`：公开接口不变，Host 公共层（校验、描述符缓存）共享，arch22 差异层承载 Kernel、workspace 布局与平台调度；FP32 与 complex64 同版交付；与 950 版完成联合回归。

## 格式支持的分阶段说明

- **本阶段（代码 PR 随附）**：CSR 原生 + CSC（Host 归一化翻译），COO/SLICED_ELL 在 Host 校验层显式返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`（诚实边界，不静默错用）；
- **第二阶段**：COO/SELL → CSR 的结构归一化 kernel（计数排序+前缀和，行内确定性序；含 SLICED_ELL 的 slice 高度/宽度/填充坐标校验），在同一公开接口下放开，补充对应 UT；计划与性能自测/UT 补全同批交付（提交验收前完成）。

## 需求拆解

1. Host 公共层上提：dtype 校验/一致性、缓存字段（含 isComplex/cachedComputeType）、描述符状态机——两 arch 共用；
2. arch22 Kernel：level-set 分析、确定性求解的 Ascend C（非 SIMT）实现，索引类型模板化分发，值路径按 isComplex 运行时分支（complex64=相邻双 float lane）；
3. A2/A3 平台调度：核数/Block 划分按 DAV_2201 规格重定（不搬用 arch35 的 SIMT 线程模型）；
4. complex64：复数乘/减/对角除（共轭乘+实除）、H 共轭语义、complex128 精度标杆（实虚部分别判定）；
5. 生命周期/异常/原地/pointer mode：与任务书 §2.1 逐条对应，含 Host/Device 双模式 alpha（host 标量经 tiling 的 alpha/alphaIm 双字段，device 指针经 tiling.alphaDevicePtr 在 kernel 内装载）；
6. C++ UT/ST：生命周期矩阵 + arch22 专项；
7. 联合回归：同 case 集 × 双 dtype × 双 arch 全绿。

# 详细设计（required）

## 总体架构

```text
sparse/spsv/
├── common/                  共享描述符与缓存字段（aclsparse_spsv_descr）
├── spsv.h（各 arch 内）      平台常量 + workspace 布局函数 + 工具
├── arch35/                  既有实现（语义不动）
└── arch22/                  【本任务新增】
    ├── spsv_tiling_data.h   tiling 字段（公共语义 + arch22 平台字段 isComplex/isConj/levelGroup）
    ├── spsv.h               平台常量（kArch22MaxBlockDim=24、UB 192KB 等）+ 八分区 workspace 布局
    ├── spsv_host.cpp        六公开接口（校验/生命周期/缓存/launch 编排）
    └── spsv_kernel.cpp/.h   Ascend C kernel（analysis / solve / update_diag / 退化路径）
test/spsv/arch22/            C++ UT/ST + torch 绑定 + workspace 对账脚本
```

## 接口与 Host 设计（arch22 差异层）

- 生命周期状态机、NULL 描述符规则、错误码与 arch35 一致；`bufferSize/analysis` 接受 NULL vec 描述符，`solve` 强制非 NULL 且校验尺寸/dtype；analysis 至 solve 期间的描述符/参数/externalBuffer 一致性校验与任务书 §2.1.4 对应；
- **UpdateMatrix 分化设计**：GENERAL（全量 nnz 数组替换）为纯 Host 指针交换，零 kernel；DIAGONAL（m 长对角值序列，任务书 §2.4 双长度语义）由 `spsv22_update_diag` kernel 将对角值 scatter 到跟踪的 values 缓冲（T/H 路径经 transPerm 映射，共轭只在 solve 装载时施加，写入原始值避免双重共轭）；
- 差异三点：平台核数查询（`GetAivCoreCount`，上限 kArch22MaxBlockDim=24）、调度阈值（编译期常量，见下）、launch 形态（Ascend C 标准 `<<<blockDim, nullptr, stream>>>`）。

## tiling 与 workspace 设计

**tiling**：`SpsvTilingData`（~208B）由 Host 填充后经**同步** `aclrtMemcpy` 推送到 workspace 头部镜像（kernel 从 GM 逐字段读取）；`numLevels` 走 workspace 独立槽（不随镜像重推覆盖）。

**workspace 八分区布局**（与 dtype 无关——布局无 elemSize 项，BufferSize 对 FP32/complex64 同值）：

```
header(AlignUp(sizeof(SpsvTilingData))) + numLevels 槽
+ diagPtr(m×4；permType=I64 时 m×8) + levelPtr((m+1)×4) + levelRow(m×4) + validCount(m×4)
+ [仅 opA=T/H] transRowPtr((m+1)×4) + transColInd(nnz×4) + transPerm(nnz×4)
```

三条瘦身决策（对照 arch35 平移方案）：
- **D1**：base=1 不物化 0-based CSR 副本，kernel 读取时统一减 idxBase（T/H 的转置结构生成时已 0-based，solve 侧有效基址为 0，不二次平移）；
- **D2**：T/H 只物化转置**结构**（统一 I32），values 不搬运——solve 经 transPerm 间接读原始 values；
- **D3**：H = T + 装载时共轭（isConj 开关），无第三套结构。

预算（P-03：m=262,144/nnz=3,932,160，complex64×H）：arch35 布局平移为 65MB(base0)/126MB(base1)，**超目标硬件 L2（64MB 级）不满足任务书 §3.4-2**；本布局两 base 均为 **35.0MB**，且 206 条性能用例 workspace 峰值即 35.0MB（对账脚本与 C API BufferSize 交叉验证，576B@4×4 精确吻合）。注：L2 容量以官方规格最终引证为准（自测报告中记录实测途径），910B 系公开口径为 64MB 级、显著高于 35.0MB。

## arch22 Kernel 设计

1. **编程模型**：Ascend C 向量范式（GlobalTensor/DataCopy 设施），无 `__simt_*`；正确性优先的首版为标量通路（GM 逐元素读写），向量化（UB 批量装载/行合并）为实测后的迭代项；
2. **分析阶段 kernel**（Phase-1 单 launch 完成全部分析；多 block 三阶段拆分为二阶段项）：转置结构构建（两遍法：列计数+行序填充，**天然确定序**，优于基线原子填充的线程序依赖）、level-set 逐行依赖深度分层（Kahn 式，`SpsvComputeLevels` 语义）、直方图+前缀和、levelRow scatter、对角定位/validCount；
3. **求解阶段 kernel（S1）**：单 launch，按 level 逐层推进，层内行按 block 分片，**层间 `SyncAll()` 全局屏障**（910B 可用性已在 CANN 9.1.0 头文件与编译期验证）；备选 S2：Host 按 levelGroup 分组多次 launch，组间依赖由流序保证（tiling.levelGroup 字段已留）；深 level（估算平均层宽 < 256，编译期常量）单 block 回退；
4. **求解行内核**：`sum = alpha·x[row] − Σ a[p]·y[col]`（前向/后向由 fillMode×op 归一化方向决定，validCount 前缀 + 未排序尾部清扫）；NON_UNIT 对角除：FP32 直接除法、complex 共轭乘+实除；缺失对角（diagPtr=-1）除以 0.0 按 IEEE-754 传播 Inf/NaN；
5. **退化路径**：nnz=0 时 UNIT → `y=alpha·x`、NON_UNIT → Inf/NaN 广播（复数乘按元素对计算）；
6. **确定性**：转置两遍法行序填充 + level 内固定遍历序 + 行内前缀序，重复执行 bitwise deterministic。

## 精度、性能、内存

- 精度：fp32→float64、complex64→complex128 golden；rtol=2^-10/atol=2^-16/A=1e-2；匹配率≥0.99；逐元素 ≤ max(A, 32·ULP)；实虚部分别判定；
- 性能：任务包 P-01~P-03 × 两 dtype，**0.25×GPU 标杆线**（任务书 §3.3）；A2/A3 实测 median/P90 + Analysis/Solve 分段 + Profiler 证据；
- 内存：八分区布局峰值 35.0MB < L2；任务书 §3.4 双判据经任务包脚本 + workspace 对账脚本产出证据。

## 联合回归（与 950 版）

同 case 集 × {fp32, complex64} × {arch35, arch22} 四象限全绿；公共层修改双 arch 同步回归。回归清单作为两任务验收材料共用附件。

# 实现状态（截至本文修订）

- arch22 五件套 + torch 绑定已在 **Atlas A2（Ascend 910B4）/ CANN 9.1.0** 完成首轮真机验证：算子库编译通过；端到端数值验证（4×4 全生命周期含 GENERAL update）误差 ≤6e-08；测试包三 hook 链路数值正确（fp32 全链 1.0e-06 / complex64×T 2.7e-05 / complex64×H+base1+DIAGONAL 1.0e-05）；
- 待完成：性能首测（P-01 起）、UT/ST 断言体补全、内存/Profiler 数据、自测报告。

# 测试方案

1. **精度**：任务包 `accuracy_cases.json` 全量（`run_accuracy_atk.sh`）+ C++ UT 矩阵：CSR/CSC × LOWER/UPPER × UNIT/NON_UNIT × N/T/H × 两 dtype × base 0/1 × 未排序/重复坐标（确定性规范化后取首个/定义序，语义随实现固化并在接口文档标注）/空行/m,nnz∈{0,1} × 零/缺失对角 INF/NAN × 原地 × pointer mode × **过早释放 buffer** × 重复执行确定性；
2. **生命周期/异常**：NULL 描述符两阶段/参数变更后重 analysis/values 指针未声明 update 即拒绝/UpdateMatrix 双语义（GENERAL 全量、DIAGONAL m 长序列经 C API 直测）+ 调度边界（单核回退阈值两侧）；
3. **性能/内存**：任务包脚本 + `gpu_full_results.tsv` 基线，**0.25×** 达标表与内存对比表入自测报告；
4. **环境**：910B3/910B4/A3 型号矩阵 + CANN 9.1.0+，记录 CANN/驱动/SOC/Profiler 版本；README 给出编译与复现步骤。

## 风险与对策

- **风险**：向量范式下 level-set 层间串行依赖限制并行度。对策：S1 层内行分片并行 + 深 level 单核回退；0.25× 目标线（NPU≤4×GPU）留有余量；必要时 S2 分组 launch 与 UB 行合并迭代；
- **风险**：双 arch 共享层重构波及 arch35 已验收行为。对策：先纯上提（行为零变化+arch35 全量回归）再落 arch22；四象限联合回归作合并门；
- **风险**：与 950 版平台差异引入隐性分叉。对策：平台相关字段集中单一 tiling 文件，其余公共层唯一定义。

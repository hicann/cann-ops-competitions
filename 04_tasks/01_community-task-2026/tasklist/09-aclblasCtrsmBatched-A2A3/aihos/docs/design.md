# aclblasCtrsmBatched_A2A3 算子实现方案

日期：2026-09-27。版本：已确认微观优化整合版，保留同步取表方案。状态：实现前设计，尚未完成 NPU 编译、精度或性能验证。

# 需求背景（required）

## 需求来源

本设计对应 CANN 社区任务 2026 的 `aclblasCtrsmBatched_A2A3`，提交团队为 **aihos**，目标工程为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)。按[社区任务提交规范](../../../../README.md)和[算子设计文档模板](../../../../resources/design_template.md)组织，交付文件为本团队的 `docs/design.md`。

本文依据本任务原始资料、用户先前确认及本轮评审后确认的裁决，以及当前 ops-blas 工作树编写。仓库核对版本为 `fe54d86f00a4d449f55da1e8d144b898953d97c5`。本轮契约变更记录见第 2.2 节。

本文给出后续实现的接口契约、计算方案、工程落点和验收方式。本次交付仅修订设计文档；任务书、README、CSV、生成器和验证脚本均保持原样。后续遇到原始资料冲突时，按第 2 节的已确认行为执行，并在测试适配层及报告中记录差异，不能通过修改原始资料消除记录。

## 背景介绍

批量三角求解处理多个尺寸相同、矩阵地址独立的复数线性系统。LEFT 和 RIGHT 分别求解左侧及右侧三角系统，结果原地写回 B；支持转置、共轭转置及隐式单位对角。

主元之间存在依赖，设计利用 batch/RHS 并行，并比较 AIV 求解与 AIV/Cube 分块更新。工程复用 ops-blas 的公共复数类型、handle、stream、workspace 和 arch22 构建组织；性能方案与数值准入均需目标环境验证。

# 需求分析（required）

## 需求描述

在 CANN 9.1.0、Atlas A2/A3 上以 Ascend C 实现正式 `aclblasCtrsmBatched` 接口。支持 COMPLEX64、列主序及全部 24 种 side/uplo/trans/diag 组合，覆盖适用 CSV 的功能和性能要求；接口、原始冲突裁决和用例处置如下。

## 1. 目标和范围

在 CANN 9.1.0、Atlas A2/A3 的 `arch22` 工程中实现单精度复数批量三角求解。每个 batch 独立，采用列主序；A、B 均为设备侧指针数组，数组元素指向各批矩阵。标量 alpha 仅支持 Host 指针。

- LEFT：`op(A[b]) X[b] = alpha B_original[b]`。
- RIGHT：`X[b] op(A[b]) = alpha B_original[b]`。
- `op(A)` 支持 N、T、C；uplo 支持 UPPER、LOWER；diag 支持 NON_UNIT、UNIT。
- 解原地写回 B，无 Carray、ldc，也无新输出张量接口。
- 覆盖当前配套 CSV 中适用于本接口的全部用例；支持一般合法运行时尺寸和前导维，不能按 CSV 尺寸白名单实现。
- 提供精度、性能自验证流程。内存占用不设验收指标，不开展专项内存验收测试，不生成内存验收报告。
- 保留正常的容量计算、资源释放、越界防护和异步生命周期管理；不做内存验收并不意味着可以忽略这些实现正确性要求。

设计先形成完整可验证、零额外 GM scratch 的单次 launch AIV FP32 求解路径，再建立多 kernel 分块基线，并将设备内 panel 循环的 AIC/AIV 混合核作为大矩阵的主要性能候选，覆盖 UNIT 与 NON_UNIT。外部接口始终为 AoS、原地输出 B；内部比较有界结果中转后直接更新 B，以及跨 panel 保留 K×RHS 条带 SoA 工作区后写回 B 两种布局。纯 AIV 在片上转换当前 tile，供 AIC 消费的重排数据经 GM 暂存或对应系统 workspace 中转；不要求全量物化 T。本文中的 tile 候选值、分流阈值和性能收益均是待测设计，不是已测结论。

## 需求拆解

| 需求 | 实现与验证落点 |
|---|---|
| 原地 B、Host alpha、Device 指针表及状态码 | 第 2～4 节：确认行为、ABI、参数优先级及全表预检 |
| 24 种模式、非方阵、leading dimension、UNIT 引用规则 | 第 4～6 节：地址映射、依赖顺序、零路径和 AIV 基线 |
| 大尺寸更新、RIGHT 布局及异步可见性 | 第 6 节：分块基线、混合核、工作 RHS 和完整复数更新提交 |
| 尾块、资源容量、欠饱和与启动参数 | 第 7 节：任务映射、容量和精简 Tiling |
| FP32 复数算术与特殊值 | 第 8 节：稳健除法、有限 alpha 优化及分流水验证 |
| arch22 工程集成 | 第 9 节：公共声明、Host/Kernel 和测试落点 |
| 可追溯的功能和性能验收 | 第 10～13 节：golden、比较器、采样、完成条件及原始摘要 |

## 2. 原始冲突与已确认行为

下表是本方案的执行依据。用户确认优先于存在冲突的原始描述；原文件不回写。

| 编号 | 原始冲突或缺口 | 最终行为 | 落实位置 |
|---|---|---|---|
| D01 | 任务书原地 B；测试 README/生成器要求独立 C/ldc | 采用任务书和 cuBLAS 的原地 B 接口，不增加 C/ldc | 公共声明、Host、Kernel、golden |
| D02 | 三条 C/ldc 专属负向与原地接口不兼容 | `TC_ED_193`、`TC_ED_194`、`TC_ED_202` 标记 N/A，保留原行，不计 PASS | 测试适配层、报告；详见第 3 节 |
| D03 | `TC_ED_185` 描述为 `alpha0_zeroC` | 保留原 ID/description；适配层解释为 alpha=0 将 B 写零，报告显示别名 `alpha0_zeroB` | 测试适配、置零路径 |
| D04 | 任务书 batchCount≥1；CSV 的 0 期望 SUCCESS | handle 非空、m/n/batchCount 非负且枚举合法后，batchCount=0 为成功 no-op；handle 有效时，负维度或负 batchCount 返回 INVALID_VALUE | Host 快速返回 |
| D05 | 零维用例 lda/ldb=0 与最小值约束不一致；先前豁免了混合非法字段 | handle 优先；负数和非法枚举先拒绝；合法零维仅豁免 lda/ldb、容量及指针检查并成功 no-op | Host；零维豁免是本任务兼容规则 |
| D06 | alpha=0 不引用 A 与无条件非空要求不一致 | alpha=(0,0) 允许 A 数组及其元素为空，完全不读取 A；B 仍须有效并实际写零 | Host 顶层检查、调用者契约、Kernel |
| D07 | 任务书均匀/正态各 50%；生成器未支持正态 | 按当前 CSV 的固定 alpha 和填充标签测试，不补造正态分布要求 | 数据生成适配 |
| D08 | 对齐偏移覆盖未表达 | 按 CSV/生成器实际覆盖，保留已有紧凑和 padding 场景，不增加专项基地址偏移验收 | 测试范围 |
| D09 | 精度阈值、MERE/MARE、ULP 描述不统一；先前确认固定 1e-2，上一版重新确认 ULP 上限，本版保留 | 每个 batch 的实部/虚部分别比较；atol=rtol=2^-13，通过比例≥99%；每个有限元素另须满足 e≤max(1e-2,32×ULP_FP32(golden)) | 算子专用测试比较器；第 10.3 节 |
| D10 | Inf/NaN 可过滤与要求覆盖不一致 | 保留执行，按特殊值类别比较；不默认过滤，不以 SUCCESS 代替结果验证 | 第 10 节 |
| D11 | `blas/trsm/` 与 `blas/trsmbatched/arch22/` 两种目录 | 采用 `blas/trsmbatched/arch22/` | 工程落点 |
| D12 | 有效采样 10 次与 >10 次 | warmup 后有效采样恰好 10 次，取平均 | 性能验证 |
| D13 | 内存“不涉及”与内存报告要求 | 无内存验收指标，不做专项内存测试或交付内存报告 | 交付范围 |
| D14 | README 性能尺寸下限 16，实际含 8×8 | 按实际 CSV，包括 `TC_PF_1006/1007/1008` 的 8×8 | 用例加载 |
| D15 | 设备表内部空元素同步错误码与稳态无等待契约不能同时保证；曾选择豁免 | 保留已恢复的任务书功能：必要设备表同步 D2H 到算子内部局部 vector，全表判空后才下发；`TC_ED_190/192` 真实测试，不再 N/A。为支持同 stream 异步生成表，复制前等待绑定 stream；计算随后异步 | Host、取表生命周期、测试适配；第 3、4.3 节 |

### 2.1 与 cuBLAS 对齐的边界

原地 B、列主序、side/trans/uplo/diag 数学语义、alpha=0 不读取 A、UNIT 不读取对角，按标准接口语义设计。Host-only alpha、ops-blas 状态码和本文数值验收阈值是本任务的约束。

零维豁免 lda/ldb 和指针检查是用户选定的兼容行为。cuBLAS 公开参数约束仍列有 lda/ldb 最小值；Netlib CTRSM 先检查参数，再快速返回。因此不能把本方案的零维豁免称为严格 cuBLAS 行为。公开资料也未穷举零 batch 与其他非法参数混合时的返回优先级。

校验顺序固定为：handle 判空 → m/n/batchCount 非负 → side/uplo/trans/diag 枚举合法 → 零维或零 batch 返回 SUCCESS → 非空路径其余检查。合法 no-op 不读取 alpha、A/B 表或矩阵，不检查 lda/ldb，不申请资源、不下发 kernel。`m=0,n=-1`、`m=0,batchCount=-5`、`batchCount=0,side=999` 等组合返回 INVALID_VALUE；上述任一组合若 handle 为空，优先返回 HANDLE_IS_NULLPTR。原 CSV 未覆盖全部组合，后续测试工程补充这些优先级检查而不修改原 CSV。handle 优先检查遵循现仓 TRSM/TRSMBatched 的入口习惯，不声称它是整个主仓所有 API 的无例外规则。

cuBLAS 公开文档规定 A/B 为设备指针数组，但未承诺对数组内部空元素在 Host 返回前检查并报告 INVALID_VALUE。本任务按任务书的更强要求实现必要元素的同步判空，不推断 cuBLAS 私有实现“从不复制、同步或检查”。恢复 D15 检查不代表可以验证任意非空地址的合法范围，也不代表 Host 入口完全无阻塞。

### 2.2 评审确认及本轮设计调整

| 项目 | 历次选择及来源 | 本版最终确认的行为 |
|---|---|---|
| 校验优先级 | 初版零维先于 handle；上一版 handle 优先但零维仍豁免负数及非法枚举 | 再次收窄：handle、非负维度/批次、合法枚举都在 no-op 前；no-op 保留 lda/ldb 与指针豁免 |
| 内部空元素 | 初版 D2H 同步判空；上一版经用户确认取消检查并新增两条 N/A | 本轮经用户确认恢复同步判空，撤销两条验收豁免；不得按性能模式关闭检查 |
| 有限数绝对误差 | 先前用户明确确认固定 1e-2；上一版重新确认 ULP 上限 | 保留逐元素 max(1e-2,32×ULP)、99% 混合容差及特殊值规则 |
| 取表与 Host 缓冲 | 前版采用同 stream 异步 D2H，并计划新增公共 handle 的 pinned 缓存 | 改为局部 vector＋同步 aclrtMemcpy；默认复制前等待绑定 stream，保留同流生成表的覆盖；撤销两处公共 helper 的缓存修改计划 |
| UNIT 性能选路 | 前版将 UNIT 与非有限数据并列为初始保守路径 | 大 UNIT 与 NON_UNIT 同样进入分块/Cube 性能候选；UNIT 仅改变对角处理，数值及特殊值准入仍须验证 |
| Cube 更新布局 | 前版主要采用产品暂存后由 AIV 更新原 AoS B | 比较两结果平面中转与持久 RHS 条带 SoA＋GM AtomicAdd；计入布局转换、工作区、分组及完整同步成本 |
| 复数除法与性能采集 | 前版未明确共享倒数快速路径及 profiler 解析/重放准入 | 倒数仅在安全范围复用；性能按完整调用关联，不以名称黑名单过滤，不以单 kernel 默认采集代表完整求解 |
| alpha 缩放 | 前版无条件允许 alpha=(1,0) 跳过；未明确纯实/纯虚分量优化 | 有限 alpha 和有限被缩放值采用经整条求解验证的分量优化；identity 跳过也须满足有限路径或实际参考验证，非有限值遵循实际 Netlib 分支 |
| Cube 尾部与更新提交 | 前版未区分逻辑归约长度、物理补齐和提交前数值检查 | 区分总 K、2×actualPanelK 与物理块；明确补齐及有效写回，按完整复数更新确认数值准入；需要结果检查时先暂存再修改 W |
| 欠饱和调度 | 前版仅描述任务分配和 L2 局部性 | 分别按 AIV 核数和混核配对团队数评估，联合容量与完整调用耗时选择 rhsTile；保持完整 K 依赖链 |
| 参数与浮点实施验证 | 前版仅要求小 POD 和目标平台验证 | 算术团队映射、精简按值字段、完整参数包编译检查；按 Scalar/Vector/Cube/FIX/AtomicAdd 分别核验实际数值行为及支持的配置 |

以上变更已纳入本版的接口、用例计数、比较器和完成条件；不回写原任务书或 CSV。算法布局与调度调整见第 5～8 节，不改变原地 B 的公共 ABI。

## 3. 不适用用例记录和统计口径

本版仅三条 C/ldc 用例 N/A，原因是原地接口没有相应参数。上一版对 `TC_ED_190/null_a_elem`、`TC_ED_192/null_b_elem` 的豁免已撤销；两条恢复构造真实设备表空元素，调用生产 API 并检查实际 INVALID_VALUE 返回，不预先宣称已 PASS。

| 原始 case_name | 原始 description | 原 CSV 行号 | 原始 expect_result | 实际处置 | 原因 |
|---|---|---:|---|---|---|
| TC_ED_193 | null_carray | 194 | ACLBLAS_STATUS_INVALID_VALUE | N/A：不适用于原地接口 | 没有 Carray 参数；实际 A/B/lda/ldb 等合法 |
| TC_ED_194 | null_c_elem | 195 | ACLBLAS_STATUS_INVALID_VALUE | N/A：不适用于原地接口 | 没有 C[i] 参数 |
| TC_ED_202 | invalid_ldc | 203 | ACLBLAS_STATUS_INVALID_VALUE | N/A：不适用于原地接口 | 没有 ldc 参数；实际 ldb=16 合法 |

适配层使用精确 case ID 和预期 description 双重核对的名单，保留整行原始数据，在构造输入和调用生产 API 前对三个精确 ID 执行 N/A 处置。GTest 用 `GTEST_SKIP()` 表示不执行；汇总报告将这三条归为 `NOT_APPLICABLE`，reason 为 `INTERFACE_NOT_APPLICABLE`。不再使用设备表同步检查豁免名单。未知 skip、运行时崩溃、超时不能被这个名单吸收。

不能由测试框架看到 description 后直接制造 `INVALID_VALUE` 并计 PASS；也不将三个 C 用例偷偷映射为重复的 B 负向用例。API 不感知 case ID 或测试描述。

固定清单数量：

- 总计 1200 条 = 1000 条精度类 + 200 条性能类。
- 精度类 1000 = 997 条适用 + 3 条 N/A。
- 适用的 997 条包括 981 条期望 SUCCESS、15 条期望 INVALID_VALUE、1 条期望 HANDLE_IS_NULLPTR。成功类中含 4 条 no-op、103 条 alpha=0，以及 2 条显式 Inf/NaN 填充用例；997 是精度类适用记录数，不是 997 个数值矩阵比较用例。
- 性能类 200 条均适用，均期望 SUCCESS；不是 200 条已经通过的性能结果。
- 全量适用 1197 条。精度类的期望错误返回同样是有效测试，其返回码符合预期时计 PASS。

报告至少区分：原始总数、适用数、已执行数、PASS、FAIL、N/A、未执行数。验收完成时应覆盖全部适用用例，不用排除项虚增通过率。`TC_ED_185` 只是输出检查语义适配，不新增或删除 CSV 行。

# 详细设计（required）

## 算子分析

对 batch b，LEFT 求解 `op(A[b])*X[b]=alpha*B_original[b]`，RIGHT 求解 `X[b]*op(A[b])=alpha*B_original[b]`。A 为指定三角，UNIT 使用隐式对角 1；最终 B[b] 保存 X[b]。

| 对象 | 数据类型与形状 | 存储语义 |
|---|---|---|
| A[b] | COMPLEX64，K×K，K=LEFT?m:n | 列主序，只读实际引用三角 |
| B[b] | COMPLEX64，m×n | 列主序，原地输入输出 |
| alpha | 一个 COMPLEX64 标量 | Host 指针，入口复制值 |
| A/B 指针表 | batchCount 个矩阵地址 | Device 数组，生命周期覆盖异步使用 |

不涉及张量广播；batch 共用维度、leading dimension 和枚举参数。完整 ABI、合法 no-op 及前导维规则见第 4 节，六种数学映射见第 5 节。

## 算子实现

Host 完成参数优先级校验、必要设备表同步预检、容量和路径选择；Kernel 实现专用写零、完整 K 依赖链 AIV 基线及经准入的分块/Cube 候选。详细状态机、同步、Tiling、数值和工程计划完整保留如下。

## 4. 公共接口与 Host 行为

### 4.1 ABI 和数据契约

```cpp
aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* const A[], int lda,
    aclblasComplex* const B[], int ldb, int batchCount);
```

复用公共 `aclblasComplex { float real; float imag; }`，不把实验工程的 `std::complex<float>`、int64 参数签名直接作为正式 ABI。维度入参仍为 int，内部地址和容量计算使用经过溢出检查的 uint64_t/size_t。

令 `K=(side==LEFT ? m : n)`。A[b] 的有效位置为 `row + col*lda`，B[b] 为 `row + col*ldb`，偏移单位是 complex64 元素；转换字节偏移时再乘 8。非空路径要求 `lda>=max(1,K)`、`ldb>=max(1,m)`。

A 为只读。仅引用指定三角，UNIT 时不访问原对角值。各 B[b] 互不重叠，A/B 不应形成会破坏只读 A 的别名；本次不增加重叠检测验收。Host 检查必要顶层指针，并按第 4.3 节检查设备表中的必要元素是否为空；alpha=0 完全豁免 A 的有效性要求。调用者仍须保证非空地址指向足够容量的有效矩阵，并在异步使用期间保持原表和矩阵的生命周期及内容约束；判空不验证任意非空地址的合法性。

### 4.2 校验顺序

以下为行为伪代码，不是已经实现的库代码：

```text
if handle == nullptr:
    return HANDLE_IS_NULLPTR
if m < 0 or n < 0 or batchCount < 0:
    return INVALID_VALUE
if side/uplo/trans/diag 不在允许枚举集合:
    return INVALID_VALUE                   # 不使用 INVALID_ENUM
if m == 0 or n == 0 or batchCount == 0:
    return SUCCESS                         # 豁免 lda/ldb、容量及 alpha/A/B；不下发 kernel
if lda/ldb 不合法或容量/地址算术溢出:
    return INVALID_VALUE
if alpha == nullptr or B == nullptr:
    return INVALID_VALUE
alphaValue = 读取 Host alpha                # 复制两分量；不保留栈指针给异步 kernel
zeroAlpha = (alphaValue.real == 0 and alphaValue.imag == 0)
if not zeroAlpha and A == nullptr:
    return INVALID_VALUE
检查设备表字节数算术、申请局部 Host vector
等待 handle stream 的前序任务完成          # 默认保留同 stream 生成指针表的支持
同步 aclrtMemcpy 复制必要表到 Host          # 零 alpha 只复制 B；复制的不是矩阵数据
逐元素检查所有必要表                      # 同步复制成功返回后才读取 vector
if 任意必要表元素为 nullptr:
    return INVALID_VALUE                   # 尚未下发任何会写 B 的 kernel
完成选路、必要计算资源准备和按值 tiling
if zeroAlpha:
    在 handle stream 排队 zero-B kernel
else:
    计算资源计划并排队求解 kernels
return 下发状态
```

`+0/-0` 均视为零；alpha=(1,0) 仅在有限快速路径或实际参考行为已验证一致的分支中跳过缩放。纯实、纯虚 alpha 的分量优化及非有限传播按第 8.2 节执行。alpha=0 的路径不扫描、不复制、不检查 A 数组元素；非空输出 B 仍必须存在。alpha=0 不自动豁免非空路径的枚举、lda/ldb 校验。

### 4.3 设备指针表检查、arch35 参考与执行边界

#### 4.3.1 arch35 现有实现的真实顺序

`aclblasStrsmBatched` 在 Host 参数校验、no-op 处理及核数查询后，先调用 `CopyPtrArraysD2H`，随后调用 `RunTrsmbatchedBatchLoop`。其具体行为为：

1. `CopyPtrArraysD2H`（现仓第 623 行）计算 `batchCount*sizeof(float*)`，初始化两个 Host vector。needA=true 时同步复制 A 表；B 表始终复制；复制的是地址值，不是 A/B 矩阵内容。复制失败返回 INTERNAL_ERROR。
2. 该函数本身不遍历元素判空。`RunTrsmbatchedBatchLoop`（第 654 行）读取第 i 个 Host 地址，先检查 B[i]，再在 needA=true 时检查 A[i]，通过后立即调用该 batch 的 `LaunchTrsmbatchedKernel`；然后检查下一 batch。
3. 因此，若第 0 个 batch 有效、第 1 个为空，第 0 个已经可能提交计算，之后才返回 INVALID_VALUE。它实现了逐 batch 判空，但不是“全表检查完成后才下发”，也不保证空元素错误返回时前面的 B 未被修改。
4. 取表使用不带 stream 参数的 `aclrtMemcpy`。该 API 返回时自己的复制完成，但不会隐式同步 Device 或任意 stream；现有入口未显式等待 h->stream。因此不能据此声称已覆盖同 stream 先前 kernel 尚在生成指针表的场景。
5. Host vector 只供 Host 遍历，launch 接收的是矩阵地址值。vector 在函数返回后析构本身不会使矩阵地址悬空；问题只在误把 Host vector 的存储地址传给异步设备工作，或提前释放真正的矩阵/设备表。
6. 现有 `test/trsmbatched/strsmbatched/arch35/strsmbatched_npu_wrapper.h` 第 100、117 行先同步 H2D 写入设备 A/B 指针表，第 178 行才调用算子；第 185 行同步在调用之后，只用于结果完成。该调用方式保证入口读表前源表已就绪，README 示例也是同一顺序，不证明入口支持尚未完成的异步表生产者。

arch22 可以采用同一同步 aclrtMemcpy＋局部 vector 方法，不存在 arch35 可以而 arch22 不可以的硬件限制。该 Host 判空逻辑不依赖 SIMT/Cube 架构，设备表字节数按地址 ABI 计算，不随矩阵元素由 float 变为 complex64 而乘 8。需要重新处理复数 alpha、生产者依赖、完整预检及批量下发；不移植 arch35 的内核或逐 batch 调度。

#### 4.3.2 arch22 采用的全表预检

默认采用算子内部局部 vector 和同步 aclrtMemcpy。为保留本方案对同 stream 异步生成指针表的支持，复制前显式等待绑定 stream；不再申请 pinned Host 缓冲或扩展公共 handle：

```text
needA = !(alphaValue.real == 0 && alphaValue.imag == 0)
ptrBytes = checked(batchCount * sizeof(device_pointer))
申请局部 hostA/hostB vector                 # needA=false 时无需 hostA
SynchronizeStream(handle.stream)            # 先保证同 stream 的表生产者完成
if needA:
    Memcpy(Aarray -> hostA, DEVICE_TO_HOST)   # 同步；检查返回值
Memcpy(Barray -> hostB, DEVICE_TO_HOST)       # 同步；检查返回值
for b in [0, batchCount):
    if hostB[b] == nullptr: return INVALID_VALUE
    if needA and hostA[b] == nullptr: return INVALID_VALUE
准备全部必要计算资源
统一下发 zero-B 或 batch×RHS 网格的求解工作
```

这次前置等待保证同 stream 的表生产者在 D2H 前完成；跨 stream 的依赖由调用者显式建立，等待 handle.stream 不会替代未建立的跨 stream 依赖。同步 aclrtMemcpy 只保证自己的复制完成，不隐式等待任意 stream。先提交表生产 kernel、随后立即在 Host 取表，可能读到旧值或未完成的表；即使求解 kernel 排在生产者后，也不能补救已经发生的 Host 误判。这个边界对 arch35 同样成立。

必须区分地址表和矩阵内容：

| 入口状态 | 同步 D2H 取表的条件 |
|---|---|
| 表已通过同步 H2D 初始化，随后保持不变 | 可直接同步复制；arch35 测试采用此方式 |
| 表的异步生产任务已经由调用者等待完成 | 可直接同步复制 |
| 表稳定，仅 A/B 矩阵内容仍在同 stream 生成 | 取表本身无需为矩阵内容等待；后续求解由 stream 顺序消费矩阵 |
| 表本身仍由异步任务生成或修改 | 必须先保证生产者完成，再由 Host 读取 |

仅采用“指针表在 API 调用前已就绪”的前置契约时，可与 arch35 一样省略额外入口 stream 等待；这不是架构优化。本版默认不收窄为该契约，也不按 CSV、case ID 或性能模式切换检查方式，因此保留前置等待和第 10.1 节同流生成表测试。对于稳定表场景，这会额外等待前序矩阵计算，影响连续调用流水，应如实测量。未来若收窄就绪契约或改变依赖接口，需同步调整设计、测试和报告，不能静默删除这项支持。

alpha=(±0,±0) 时不复制、不读取、不检查 A，必要 B 元素仍必须非空，随后真正写零。UNIT 只豁免存储对角的读取，不豁免整个 A 矩阵地址。CSV 最大 batch=1024，在 64 位地址 ABI 下每张表为 8 KiB，A+B 合计 16 KiB；字节量小不意味着同步等待必定只有几微秒。

空元素判定在任何 B 写入前完成，因此 `TC_ED_190/192` 真实返回 INVALID_VALUE，不能由测试框架代造状态。不按 case ID、debug 开关或“性能模式”绕过生产检查。检查只保证必要地址非空，不保证任意非空野指针的有效分配范围。

#### 4.3.3 快照、生命周期和异步范围

arch22 批量 kernel 继续读取原 Device 指针表，Host 快照仅用于预检。默认不额外 H2D 一份设备快照；调用者须保持原表内容及矩阵有效至相关异步计算完成，不能在预检与实际取址之间改表。若未来改为设备快照，需要另行核算复制、scratch 和生命周期，不能直接使用临时 Host vector 地址。

恢复检查后，非空 API 在入口取表阶段可能阻塞，求解工作随后异步排队；并不保证整个 API 无 Host 等待。取表同步还可能等待前序计算，因此端到端和连续调用吞吐须实测。msprof 所选计算 kernel 的 Task Duration 不直接加上 Host 取表等待，但该成本必须在 CPU 提交、设备跨度和同步端到端报告中保留。

局部 vector 由 RAII 管理；同步复制成功返回后才读取，函数返回时正常析构。后续 kernel 读取原 Device 表，不持有 vector 的 Host 存储地址。vector 分配、前置同步或任一复制失败时，按库约定返回错误，不读取失败的快照、不启动任何计算；错误返回时 runtime 不得仍使用被释放的 Host 缓冲，异常状态按目标运行时错误清理约定处理。默认方案没有异步 D2H 已排队的 pinned 缓冲复用问题；若以后评估局部 pinned＋MemcpyAsync 备选，必须恢复已入队任务完成前不能释放缓冲的清理规则。不得吞错为 SUCCESS。

“空元素失败前没有 B 写入”仅覆盖完整预检发现的参数错误；后续多次 launch 或异步执行失败不提供全 batch 事务回滚。调用者读结果、释放或修改表/矩阵前仍须保证求解完成；测试同时检查 API 和后续同步状态。

### 4.4 资源生命周期

复用 handle 的 workspace 机制。当前库默认 32 MiB、最大 2 GiB；`EnsureDefaultWorkspace` 扩容可能同步。用户设置的 workspace 不足时不得擅自替换：先选择更小分组，或在任何 B 写入前选择第 6.2 节零额外 GM scratch 的 AIV 路径，不仅因优化暂存不足就返回 ALLOC_FAILED。避免为可选优化先扩容并释放旧缓冲，再发现可走零 scratch 路径；确需的 Host staging 或框架/runtime 资源申请失败仍传播相应错误。

零额外 GM scratch 是本算子求解路径的目标，不等于进程不占 GM。现仓 `aclblasCreate` 申请默认 workspace，`aclblasSetWorkspace` 拒绝空指针或大小为 0；本任务不改变这些公共辅助接口。不能承诺用户通过现有接口设置 0 workspace，或任何资源耗尽场景都成功。按值 tiling、尾部精确存取及有无必需系统 workspace 在目标工具链验证；若零 scratch 准入未完成，不标记该保底已实现。

Host 指针表与计算 GM workspace 分开管理。Host 表默认为 ctrsmbatched_host.cpp 内部的局部 vector，不新增公共 handle 字段，不修改 aclblas_handle_internal.h 或 aclblas_auxiliary.cpp 的缓存/销毁逻辑。任务书规定工程落点和禁止明显冗余，没有“只能修改三个路径”的绝对禁令；撤销公共缓存计划的理由是缺少实测必要性，并减少共享生命周期和维护范围。记录每调用分配、初始化及释放成本，不预设 8 KiB 分配只需 50 ns；若以后证明缓存必要，再单独评估通用资源机制。

稳态调用仍有第 4.3 节前置等待及同步取表，计算下发后不额外等待结果。首次资源准备、workspace 扩容和 handle 生命周期操作也可能阻塞。GM 工作区按库机制复用，Host vector 每调用局部管理；通过预热区分冷启动与稳态，但不能把每次仍发生的 Host 分配成本移出 CPU/端到端时间。性能统计按第 11 节记录真实同步边界。

在任何结果写入前检查计划尺寸和申请必要资源。禁止返回后立即释放仍被 kernel 使用的 GM scratch，禁止把临时 Host 栈对象通过异步 H2D 留给设备。小 POD tiling 优先采用仓库同类按值传递形式，精简字段并按第 7.3 节检查完整 kernel 参数包；具体编译接口在 CANN 9.1/arch22 验证。较大配置改用 GM tiling 时，使用明确生命周期的存储和有序传输，不使零额外 GM AIV 基线依赖可选配置。

同一 handle 及其 workspace 的并发使用须串行化；不在本任务中扩展库的线程安全契约。所有计算阶段使用绑定 stream。

## 5. 统一数学映射

将两种 side 统一为 `T Y = alpha D`，但不改变外部存储和输出位置。

| side | K：求解阶数 | R：独立 RHS 数 | T | D | Y |
|---|---:|---:|---|---|---|
| LEFT | m | n | op(A) | B_original | X |
| RIGHT | n | m | op(A) 的普通转置 | B_original 的普通转置 | X 的普通转置 |

RIGHT 的变换是普通转置，不是共轭转置；alpha 不取共轭。六种映射如下：

| side/trans | T(i,j) 从 A 读取 | T 的三角方向 |
|---|---|---|
| LEFT/N | A(i,j) | 原 uplo |
| LEFT/T | A(j,i) | 翻转 uplo |
| LEFT/C | conj(A(j,i)) | 翻转 uplo |
| RIGHT/N | A(j,i) | 翻转 uplo |
| RIGHT/T | A(i,j) | 原 uplo |
| RIGHT/C | conj(A(i,j)) | 原 uplo |

因此 `transposeA=(trans!=N) XOR (side==RIGHT)`，`conjugateA=(trans==C)`。按 transposeA 决定地址交换，按 conjugateA 对虚部取负。三角翻转与 transposeA 一致。

统一坐标 Y(i,q) 写回 B 的元素地址：LEFT 为 `i+q*ldb`，RIGHT 为 `q+i*ldb`。尾块写回只覆盖实际 `m*n`，不修改 lda/ldb padding。

这里的转置是逻辑视图，不要求物理转置完整 B。尤其 RIGHT 的 `Y(i,q)=B[q+i*ldb]` 已是行步长为 ldb、q 方向连续的 K×R 视图，可直接沿外部 B 的列块求解和更新。外部 A/B 的 GM 布局保持 AoS；直接路径在 UB 中转换当前 tile，持久 SoA 候选则按逻辑 K×RHS 条带初始化工作平面，最终按上述地址映射写回 B。Cube 路径按第 6.3 节提交供 AIC 消费的合法输入。A 的 N/T/C 同样通过地址映射、局部 tile 转置及共轭处理，不默认物化全量 T。

令 `S=op(A)`，在外部 B 上直接分块的等价关系如下。此处 B_P/B_rem 指已施加一次 alpha 并累计前序更新的工作 RHS，不是每个 panel 都重新缩放的原始输入。

| side | panel 求解 | 剩余区域更新 | panel 顺序 |
|---|---|---|---|
| LEFT | `S_PP * X_P = B_P` | `B_rem -= S_rem,P * X_P` | S 下三角从上到下；上三角从下到上 |
| RIGHT | `X_P * S_PP = B_P` | `B_rem -= X_P * S_P,rem` | S 上三角从左到右；下三角从右到左 |

顺序依据 op(A) 后的三角方向，而非未经 trans 处理的 uplo。例如 `TC_PF_1005` 为 RIGHT/UPPER/C，S 是下三角，直接 RIGHT 实现应从右向左解列 panel；与统一视图 T 为上三角、倒序求解一致。

对下三角按 i 从 0 到 K-1 前代，上三角按 i 从 K-1 到 0 回代：

```text
v = alpha * D(i,q) - sum(T(i,j) * Y(j,q), j 属于已求解索引)
Y(i,q) = v                         if diag == UNIT
Y(i,q) = complex_div(v, T(i,i))     otherwise
```

公式用于有限数数学说明；非有限值保守路径还需保留第 8 节约定的运算和短路顺序，不能假定所有代数等价改写的 NaN/Inf 传播都相同。

## 6. Kernel 路径与并行设计

### 6.1 零 alpha 路径

按 `(batch, column, row tile)` 分配 B 的有效区域，只写 `(0,0)`，不读 B 原值，不读 A。不能通过 `B *= 0` 实现，否则 Inf/NaN 输入可能不能得到零。padding 保持原值。空形状和零 batch 在 Host 返回，不启动该 kernel。

### 6.2 通用 AIV FP32 求解

以 `(batch, RHS stripe)` 为独立任务，单次 AIV launch 内，同一 stripe 在一个 AIV 上完整处理 K 方向的依赖链；其他核可以读取同一 A，但写入不同的 B 元素。禁止把互相依赖的 K 行直接分给多个核独立计算。小矩阵、大 batch 优先使用此路径，以批次和 RHS 并行摊薄下发成本，不由 Host 逐行或逐 panel 下发。alpha 缩放融合进每个尚未覆盖的 RHS 元素的首次读取，精确应用一次。

小尺寸优先将需要的三角数据、RHS stripe 放入 UB。GM 采用交织 complex64，UB 中按实/虚两个 FP32 数组组织，便于向量乘、加、减、除。LEFT 跨 RHS 读取的列主数据通过小块转置/打包形成 UB 连续条带；RIGHT 的 RHS 方向在外部 B 中本就连续。拆分实虚部后仍必须用同一复数公式组合。

任务数为 `Q=batchCount * ceil(R/rhsTile)`，与运行时查询的 AIV 核数比较并分配；单个核可循环处理多个任务。Q 小于可用 AIV 核数时评估缩小 rhsTile，候选可包含 4、1，以增加独立 RHS 任务；每个任务仍独占完整 K 依赖链。最小 rhsTile=1 时独立任务上限为 batchCount×R，不能保证填满全部 AIV。具体选择联合重复读 A、尾部存取、UB 容量和完整调用耗时验证，见第 7.1 节。小 batch、大 RHS 通过 RHS 切分提供并行，大 batch、小矩阵通过批次并行摊薄调度成本。A 过大不能完整放 UB 时按实际引用区间分段加载，已求出的 Y 采用有界本地缓存或 GM 读回，不将整个 K×K 强制驻留 UB。

功能保底以零额外算法 GM scratch 为目标：从已有 Device 表取得地址，直接读取 A，并用原地 B 保存已求出的元素，UB 采用有界微块滚动。K 增长时增加分段读取，不申请随 K²/KR 增长的矩阵副本。使用纯 Vector 和按值小 POD tiling，不依赖 Matmul 系统 workspace、全局分类数组、备份或跨核 GM flag；选用保守路径本身无需预扫描。必要时收缩到单 RHS，仍须保持实际 Netlib 的运算及特殊值规则。

零额外 GM 不包括调用者的 A/B/设备表、已有 handle 资源和 runtime 内部开销，也不意味着入口无需第 4.3 节 Host 取表缓冲。LEFT 非连续条带、8 字节复数元素及尾部写回必须通过精确合法范围存取实现；不能用 padding 作 scratch，不能以扩大 DMA 写回范围覆盖其他条带。索引使用经溢出检查的 64 位算术，并在目标工具链验证最小 UB/尾部路径。该保底保障可支持输入范围内的功能，不保证零资源环境、运行时故障或所有性能门限。

单核独占条带不等于各流水线自动可见：若 Y 经 MTE3 写回后由 MTE2 重载，必须在依赖点等待对应的写完成/读可见事件；Scalar 或缓存访问也须遵守相应可见性协议。UB 缓冲在 DMA 完成前不得复用。同一 stream 的 kernel 顺序只解决 kernel 之间的依赖，不能替代 kernel 内部的数据搬运同步。

基线首先采用能明确控制求解次序的实现；在结果验证后，将相同系数作用于多个 RHS 的更新向量化。UNIT 直接使用隐式对角 1，不进行原对角读和除法；未引用三角不能参与算术。

### 6.3 大矩阵分块求解

性能优化采用“对角 panel 求解 + 剩余区域更新”。对统一的下三角系统，W 表示已施加一次 alpha 并累计更新的工作 RHS。低容量候选以原地 AoS B 的逻辑视图保存 W；主性能候选为每个活动团队保存完整 K 方向、有限 RHS 宽度的持久 SoA 工作条带。两者都按第 5 节的地址映射将最终结果覆写原 B，不默认物化全量 T 或全部 batch 的工作 B：

```text
W = 选定的 AoS B 视图或持久 SoA RHS 条带    # 地址映射见第 5 节
初始化 W = alpha * B 的实际引用条带         # 只应用一次；alpha=0 走专用路径
for panel p 从前向后:
    W_p = solve(T_pp, W_p)                 # 已求解的 X/Y 保存在选定工作布局
    W_tail -= T_tail,p * W_p               # AIV 或 Cube 更新，末 panel 无 tail
若 W 为 SoA 条带，将最终解按实虚交织写回原 AoS B  # 每个元素仅做必要的一次最终写回
```

上三角按 panel 从后向前处理，更新尚未求解的前部。每个 panel 求解完成后才启动相应更新，当前依赖域的更新完成后才进入下一个 panel。alpha 只应用一次，不能每个 panel 重复缩放。多 kernel 基线可用独立 scale/Pack kernel；设备内循环方案可在团队拥有的 RHS 条带中初始化，但须在更新消费前完成。是否融合以及阶段事件必须显式设计，不能对已累计更新的 W 再乘 alpha。UNIT 在 panel 求解中不读存储对角且跳过除法；跨 panel 的 trailing 矩形块不含对角，同样可以采用 Cube 更新，但仍需通过第 10 节的数值和特殊值验证。

每个更新 tile 独占一个 B 子区间，不让不同核同时累加同一个输出。纯 AIV 在 UB 完成局部转换后直接写回 B。A2/A3 上 AIC 不能直接读取 AIV 的私有 UB；接口层面的 UB→L1 搬运通过 Matmul workspace 实现 UB→GM→L1。因此，凡由 AIV 解交织、转置或共轭后供 AIC 消费的数据，必须预留有界 panel/tile GM 输入暂存，或核算实际使用的系统 workspace，这部分中转不能按“收益不足”直接省略。避免全矩阵物化，不等于取消局部 GM 中转。

**低容量候选：两个 FP32 实/虚结果平面中转，再更新原 AoS B。**通过经验证的四实乘片上组合或两次归约长度为 2×actualPanelK 的实数嵌入 GEMM，优先只向 GM 写两个实/虚结果平面，再由 AIV 读取，从当前 B 扣减乘积并交织写回。四个独立乘积平面保留为易于检查的基线，不预设其能够满足性能门限。数据流为：

```text
GM AoS → AIV MTE2 加载 → UB 求解/转换
       → AIV MTE3 写输入暂存 → panel_ready
       → AIC MTE2/L1/L0 执行实数 GEMM
       → AIC FIX 写两个实/虚结果平面 → product_ready
       → AIV MTE2 读取 → Vector 从工作 B 扣减乘积、交织
       → AIV MTE3 写回原 AoS B → B_ready → 下一 panel
```

两结果平面方案须证明真实使用的 Matmul/基础 API 支持该组合与输出布局，不能仅在容量模型中将四平面改成两平面。四实乘不是硬件强制的唯一复数实现；原 AoS 指针也不能未经合法变换便当作普通实数矩阵使用。不默认 AIC 已能直接正确写回原 AoS B。

**主性能候选：持久 K×RHS stripe SoA，两次实数嵌入 GEMM，在数值准入通过后直接 AtomicAdd 更新工作平面。**团队首次把实际引用的原 B 条带转成 W_real/W_imag，并完成一次 alpha 缩放；后续所有 panel 在同一条带上求解和更新，不在每个 panel 重新 Pack/Unpack 尾 B。以更新 `W -= T*Y` 为例，沿 GEMM 的归约轴交替排列 T_real、T_imag；为实部构造 `[-Y_real, +Y_imag]`，为虚部构造 `[-Y_imag, -Y_real]`。两次实 GEMM 的逻辑归约长度均为 `2*actualPanelK`，分别生成负的复数乘积实部和虚部，以 `IterateAll(W_real, 1)`、`IterateAll(W_imag, 1)` 累加到现有工作平面。actualPanelK 是当前实际 panel 长度，不是 TRSM 总阶数 K，也不是 Cube 内部对齐长度。变换、符号、转置与尺寸按真实 tile 构造，尾部及提交契约见第 6.3.3 节。

CANN 9.1 的 `IterateAll` 第二参数为 enAtomic，1 表示 **GM AtomicAdd**；它不是将 GM 旧 B 读入 L0C 的 bias 融合，也不会自动初始化目标。A2/A3 支持 float 的 L0C→GM 原子累加。目标必须先包含正确的 `alpha*B` 及前序更新；不能清零后丢失 RHS。GM 原子读改写仍有成本，不能记作无目标读取。初版保持目标条带独占、panel 顺序固定，不用并发 split-K 或多个团队对同址累加换取吞吐；FP32/HF32 配置、实数嵌入造成的规约顺序变化及非有限传播均按第 8、10 节验收。

```text
原 GM AoS B → AIV UB 转换/一次 alpha 缩放 → MTE3 初始化持久 SoA W
for panel:
    AIV 读取 W_panel → UB 求解 → MTE3 写回解及带符号嵌入输入
    → 完成完整复数更新的数值准入及路径选择 → panel_ready
    → AIC 两次逻辑归约长度为 2*actualPanelK 的实 GEMM
    → FIX/GM AtomicAdd 更新 W_real、W_imag → update_ready
    → AIV 等待两平面更新完成 → 下一 panel
全部求解完成 → AIV 将 SoA 最终解重新交织 → MTE3 覆写原 AoS B → output_ready
```

`experimental/aclblasCtrsmBatched2` 可参考该结构，但其更新目标是持久 SoA workspace，RIGHT 还包含首尾布局转换，不能据此宣称在完全不保留工作 RHS 时获得同等收益。持久条带只需 K×rhsTile，不要求全量 T；全部活动条带、嵌入输入和系统 workspace 的容量按第 7.2 节核算，资源生命周期持续至相关计算及最终原 B 写回完成。

下表仅计一次复数尾部元素更新的输出侧**逻辑 GM 访问量**，不含 A/X 输入、对齐、缓存和同步，不代表实测 HBM 流量或耗时。AtomicAdd 的目标读改写按读旧值和写新值各一次估算：

| 更新布局 | 每复数更新元素的逻辑访问量 |
|---|---:|
| 四乘积平面写/读，加原 B 读/写 | 32+16=48 B |
| 两结果平面写/读，加原 B 读/写 | 16+16=32 B |
| 持久 SoA，两次融合结果 AtomicAdd | 约 16 B 目标读改写，另加条带首尾转换 |
| 每 panel 临时转换尾 B、AtomicAdd、再转回 | 约 32+16=48 B，不能保证流量收益 |

以 Case 5、panel=128 为例，两批完整尾部更新约有 520,093,696 个复数输出元素；四乘积相对原 B 读写增加的 32 B 中转约为 15.5 GiB。该值是逻辑流量模型，不是 HBM 实测。持久 SoA 会增加存储与首尾转换，选择不能只比较中间产品字节：须统计完整 A/B/输入暂存、资源分组、规约精度、事件等待和最终写回。大 RHS 分块更新可能具备 GEMM 式复用，不能笼统认定所有 TRSM 都是访存受限，也不承诺 AtomicAdd 必然更快。

#### 6.3.1 多 kernel 正确性基线与下发成本

基线以同一 stream 中不同 kernel 的先后顺序构成阶段边界，不使用未经验证的跨核自旋屏障。batch 维纳入一次 launch 的任务网格，避免每 batch 重复下发整套 panel kernels。该路径保留作为可验证基线和兼容路径，不预设它能满足全部性能目标。

令 `P=ceil(K/panelSize)`。每 panel 一次 solve、每非末 panel 一次融合复数 update 时，每活动资源组的主路径为 `2P-1` 次 launch；另加 scale/Pack、数据分类、实际需要的暂存/合并/Unpack 等辅助 kernel。一次 update kernel 内可包含两次实数嵌入 GEMM AtomicAdd，并不要求两次 Host launch；低容量中转若单独下发 AIV 扣减，应另计实际次数。必要暂存是否与已有 launch 融合是调度选择，不改变物理中转需求。batch 已在任务网格中，不再乘 batchCount；若 workspace 分 G 组，则重复 G 套主序列。

| K | panel=32 | panel=64 | panel=128 | panel=256 |
|---:|---:|---:|---:|---:|
| 256 | 15 | 7 | 3 | 1 |
| 2048 | 127 | 63 | 31 | 15 |
| 4096 | 255 | 127 | 63 | 31 |

这些是主路径 launch 计数，不是实测耗时。CPU 下发可能与设备计算重叠，不能将 `次数×CPU 提交耗时` 与全部 kernel 时长简单相加。按第 11.1 节分测 CPU 提交、设备跨度和同步端到端时间，并通过 trace 检查阶段间空隙，不以未测的“数毫秒”估计代替目标设备结果。

四次实数乘法形成复数更新：

```text
P_real = T_real * Y_real - T_imag * Y_imag
P_imag = T_real * Y_imag + T_imag * Y_real
W_real -= P_real
W_imag -= P_imag
```

四实乘提供复数数学说明和可检查基线；两次实数嵌入是相同数学运算的另一候选，不先采用 3M 重排。Cube 路径以 FP32 输入/累加、关闭 HF32 为精度起点，不以 FP16/BF16 下转或放宽比较器换取性能。仓库 arch22 复数 BLAS3 有可参考实现，但不同规约顺序的最终数值和吞吐必须在目标硬件验证。

目标是实际选定的实 GEMM 及必要复数组合在同一个 update kernel 内完成，以符合上述 `2P-1` 模型；四实乘并不等于四次 Host launch。如果实现必须分别下发四个实 GEMM，则主路径至少为 `P+4(P-1)=5P-4` 次，还要另计合并/减法 kernel；两次嵌入 GEMM 若分别下发同样按实际次数计数。Pack/Unpack 不能隐藏在主路径次数或性能报告之外。

#### 6.3.2 设备内循环的 AIC/AIV 混合核

大矩阵主要性能候选在一次混合核 launch 内遍历 panel：持久 SoA 路径由 AIV 初始化并求解工作条带，AIC 两次实数嵌入 GEMM AtomicAdd 更新实/虚平面，末尾由 AIV 写回原 AoS B；低容量候选由 AIC 输出两结果平面，AIV 扣减原 B。初始调度以 `(batch, RHS stripe)` 为独立团队的所有权范围，团队拥有该条带完整 K 链及其可写 scratch；不同团队写入不重叠的 B/工作条带，避免依赖其他未调度团队的求解结果。团队映射、实际常驻数和尾任务分配须按 A2/A3 核配置验证，不能直接沿用实验工程固定核数。

同步协议至少包含：

1. 持久 SoA 的 `alpha*B` 初始化先完成 Vector→MTE3 依赖和写入，再允许任何 AtomicAdd；每个 panel 的解及嵌入输入也须完成 MTE3 写入后发布 panel_ready。AIC 等待后才下发依赖该数据的加载与累加，不能以 AtomicAdd 的原子性代替初始化顺序。
2. 低容量路径在 AIC 完成 Cube 及 FIX→GM 两结果输出后发布 product_ready；AIV 等待后才加载、扣减/交织并经 MTE3 写原 B，达到 B_ready 后进入下一 panel。持久 SoA 路径须等待实部、虚部两次 AtomicAdd 的 FIX/GM 更新均完成，发布 update_ready，AIV 等待后才读取依赖的下一 panel；此时无需额外 AIV 扣减。实际语义随路径定义，不能把某一次实 GEMM 完成当作完整复数更新或原 B 最终结果完成。
3. 同一数据缓冲在所有消费者完成最后一次读取前不得复用；初版保守等待本路径的 B_ready/update_ready，后续才研究 consumed/ack。持久 SoA 在全部 panel 求解及最终 Unpack/MTE3 原 B 写回完成后才允许回收或复用。事件编号、初始化和复用代次须避免不同 panel/batch 串扰，并核对 Matmul、SyncAll 等接口占用的 flag，禁止冲突。
4. 奇数 batch、空 RHS 条带、尾 panel、按数据状态跳过的团队，仍须完成配对协议要求的握手，不能单方提前 return 使另一侧永久等待。按第 6.3.3 节选择 Cube/AIV 或暂存路径时，两侧消费同一已发布的路径状态；AIC 不执行更新的分支也须完成约定确认，由 AIV 完成更新后才能进入下一 panel。
5. 不建立等待未调度 wave 的全 grid 屏障。若改变团队划分而引入跨团队依赖，须重新证明常驻性、事件覆盖及无死锁，不能把局部跨核 flag 当作全设备屏障。

| 逻辑完成条件 | 发布前必须完成 | 典型流水或本地依赖 |
|---|---|---|
| 初始化完成 | 持久 W 中的 alpha*B 已经 MTE3 写入 | Vector→MTE3→跨核消费依赖，先于任何 AtomicAdd |
| panel_ready | AIV panel 解及变换输入经 MTE3 写入 | Vector→MTE3 依赖；`CrossCoreSetFlag<2, PIPE_MTE3>` |
| product_ready | 低容量路径 AIC 运算及 FIX 两结果写回 | 对应 Cube→FIX 依赖；`CrossCoreSetFlag<2, PIPE_FIX>` |
| B_ready | 低容量路径 AIV 扣减、交织及原 B 写回 | 正确 MTE2→Vector→MTE3→后续读取依赖；跨核消费则补握手 |
| update_ready | 持久路径实/虚两次 FIX/GM AtomicAdd 都完成 | 两次运算及 FIX 依赖；发布后 AIV 才读取下一 panel |
| output_ready | 持久路径最终解经 Unpack/MTE3 写回原 B | 最终工作条带读取及 Vector→MTE3 依赖 |
| 暂存可复用 | 所有消费者已完成该槽最后一次读取 | 等本路径 B_ready/update_ready；持久 W 等 output_ready |

B_ready、update_ready 和 output_ready 是逻辑完成条件，不要求每处再分配一个跨核 flag；同一 AIV 消费其独占 B 可用正确核内依赖，其他核消费则须建立相应跨核协议。AIV UB→GM 发布绑定 MTE3，AIC L0C→GM 普通输出或 AtomicAdd 发布通常绑定 FIX，不统一套用 MTE3。若 Matmul 使用异步 IterateAll，必须按接口规则等待完成再发布事件，不以 C++ 调用返回直接代替 FIX/GM 完成。

CANN 9.1 文档说明 A2/A3 的 CrossCoreWaitFlag 模板 modeId/pipe 不生效，实际阻塞后续所有流水指令。`WaitFlag<2, PIPE_MTE2>` 可表达意图，但不据此宣称只阻塞 MTE2；必须在下发依赖读取前等待，不能靠 wait 撤销已经下发的预读。SetFlag 的发布流水及参与者配置仍须与实际数据通路一致。mode2 涉及同一 AI Core 的 AIC 与全部配对 AIV；MIX_AIC_1_2 两个 AIV 均须完成约定参与，空任务不能擅自省略。

DMA staging 的完成顺序和 Scalar DCache 一致性分别处理。经 DataCopy 的 DMA 流需要正确事件依赖；Scalar 读其他核修改的 GM，或将 Scalar 写入发布给其他核，另需相应 DCache clean/invalidate。不能用清理缓存替代流水同步，也不能每 panel 无条件清空全部 DCache；正确性不要求数据必须已落到 HBM，可通过一致的 GM/L2 路径消费。

先实现“当前 trailing update 完整结束，再进入下一 panel，最终写回原 B”的保守协议；lookahead 或双缓冲重叠仅在事件、数据所有权和结果验证通过后研究。GM AtomicAdd 保证目标更新的原子性，不保证 panel 顺序、事件可见性或不同累加顺序的数值相同。单次 launch 只降低 Host 下发次数，不自动消除设备同步和等待成本。

`experimental/aclblasCtrsmBatched2/` 已有 `MIX_AIC_1_2`、AIV/AIC 内部 panel 循环及空任务握手结构，可用于研究。本任务仍需重新验证 ABI、Device 指针表、alpha、布局、特殊值和目标工具链，不能将实验代码存在视为 A2/A3 上已正确或已达标。

#### 6.3.3 Cube 尾部与完整复数更新的提交契约

分别记录当前 update 的逻辑 M/N、`2*actualPanelK` 和所用 API 的物理块/补齐尺寸。ND 原始尺寸与内部 base 块的约束分开处理，按实际 Matmul 接口配置有效尾部，例如 SetTail；内部块的对齐要求不直接作为原始矩阵尺寸的分流条件。尚未验证的尾块使用 AIV；Cube 尾部路径仅在加载、归约、特殊值与写回检查通过后启用。

需要显式补齐 K 方向时，两个实数操作数对应的无效位置均置零，新增归约项为 0×0；M/N 无效位置不写入有效 W 或外部 B，不向条带之外发起 AtomicAdd。实际引用位置中的 0×Inf、乘积溢出及抵消顺序变化属于数值准入问题，不能仅由物理对齐解决。尾块还须保持 UNIT 对角不读、未引用三角不读和外部 padding 不写。

直接 AtomicAdd 候选在每次 trailing update 修改 W 前，按完整复数更新确认两个结果分量的准入。条件覆盖实际 A tile、已求 panel、目标 W 及必要的中间运算范围；只有经验证的准入能保证该路径满足第 10 节时才启用，不能仅凭原始输入有限判断安全。状态在 AIC/AIV 之间有序发布，分支及下一 panel 的完成依赖遵守第 6.3.2 节。

若需要先计算结果才能判断安全，使用两个结果平面的暂存路径，在保持目标 W 未更新时完成乘积及目标加减更新的两分量和必要范围检查；确认后再更新实/虚工作 RHS。未通过时，只能在更新前 W 和所需操作数仍可用、相应 AIV 次序已验证且握手完整的条件下改走局部 AIV 路径；后续计算仍须执行，不能将拒绝 Cube 等同于成功完成。暂存和检查全部计入容量及完整调用耗时。

不能先 AtomicAdd 实部再检查虚部并声称可以无损回退，也不能在 AtomicAdd 已污染 W 后自动恢复。对已提交状态的重算仍受第 6.4 节原始 RHS、备份和写回边界约束；事后类别检查不构成直接提交路径的安全保证。

### 6.4 分流、原地写入与重算边界

| 场景 | 初始路径 | 优化准入 |
|---|---|---|
| 有效 handle 下空形状、零 batch | Host no-op | 固定行为 |
| alpha=0 | AIV 只写 B 零 | 固定行为 |
| 小 K、窄 RHS、小总工作量 | 单次 launch 通用 AIV | UB 容量和调度成本验证后调整 tile |
| 大尺寸、输入有限的 UNIT / NON_UNIT 数据 | 多 kernel 分块基线；两种 diag 的主要性能候选均为设备内循环混合核 | 覆盖有限输入的中间增长、溢出及结果类别，对所有适用数据满足第 10 节规则，含转换及同步后的性能实测有收益 |
| 检测到非有限数据 | 经参考次序验证的保守 AIV 求解；其他路径独立验证 | 对应优化路径通过严格类别与数值验证后才启用，不整类过滤 |

上述条件仅使用实际参数、平台能力和实际数据，不能读取 case 名或用固定 seed 决定结果。K/R 的初始分流候选可从 K=64/128 的边界试验，不能把这些候选写成未验证的最优阈值。

UNIT 不作为长期禁用 Cube 的分流条件：对角 panel 不读取存储对角、不做复数除法，引用的非对角矩形区域仍采用分块更新候选。其数值类别准入单独验证，不能以 UNIT 数学上非奇异为由无条件开启优化，也不能因初始 AIV 基线而放弃大尺寸性能验收。优先验证 Case 3/5 的 golden 有限/Inf/NaN 分布、各路径的中间增长及最终类别，再确定可上线的混合核路径。

按一个普通复数乘加更新约 8 个实数 FLOPs、每个 RHS 的三角更新约 `K(K-1)/2` 次估算，主要更新量为 `F≈4*batchCount*R*K*(K-1)`，不含缩放、panel 辅助计算、转换和同步。这是算法工作量模型，不是目标平台实测指令数；特殊值、短路及实际 kernel 组织另行核查。

| CSV 性能用例 | batchCount / K / R | 主要更新量 | 达标时间 | 工作量模型对应的有效算力 |
|---|---|---:|---:|---:|
| Case 3 / TC_PF_1003，RIGHT / UNIT | 32 / 1024 / 1024 | 137.305 GFLOPs | 13.883875 ms | 9.8895 TFLOP/s |
| Case 5 / TC_PF_1005，RIGHT / UNIT | 2 / 4096 / 4096 | 549.622 GFLOPs | 40.969125 ms | 13.416 TFLOP/s |

若仅为风险估算暂取 AIV 14 TFLOP/s，Case 5 的理想算术时间约 39.26 ms，并非已经超过 50 ms；但与验收门限仅相差约 1.71 ms，实际利用率、访存及依赖等待使纯 AIV 路径风险很高。这不构成 AIV 峰值或混合核达标承诺。不能用产品 FP16 280 TFLOP/s 指标代替本方案 FP32、关闭 HF32 的有效 Cube 算力，实际性能以目标 A2/A3 配置及第 11 节测量为准。

若优化路径需要按数据有限性分流，设备预扫描必须在首次修改 B 前完成，生成每 batch 状态；后续 AIV/Cube 根据状态选择工作，不将状态 D2H 后同步决定 Host 分支。该状态缓冲只属于相应优化路径；零额外 GM 保底直接使用保守 AIV，不为选择此路径申请全局分类数组。扫描只访问实际引用的数据，UNIT 不读存储对角，不扫描未引用三角或 padding，alpha=0 直接走零路径。同 stream kernel 边界或经验证的设备事件保证状态可见；状态写入使用独占槽位及归约或经过验证的原子操作。混合核中跳过计算仍须满足第 6.3.2 节的握手协议。

默认不独立保存完整 B_original：直接路径在求解中修改 B，持久 SoA 路径先修改工作 W，再按完成条带或资源组最终覆写 B。两者均不承诺整批任意中途异常后透明重算。有限输入也可能在后续运算中溢出，不能仅凭输入有限就保证优化路径满足特殊值规则；未经相应验证的路径保持关闭，不能在原始 RHS 已丢失后重新从当前 B/W 开始整批求解，或把跳过后续更新报告成成功结果。

需要透明重算的独立可选路径，必须证明相应原始 RHS 在重算时仍完整可用，预留恢复及求解资源，定义不会破坏混合核握手的调度。直接路径须在首次覆盖前保存必要备份；持久 SoA 中尚未写回原 B 的条带可能仍可取得原始 RHS，但不能据此假定已写回条带或整批仍可重启，需逐范围核对写回边界和完成协议。W_Y 是会被修改的工作布局，不是原始 RHS 备份。实际新增备份/恢复、分类和重新计算的全部成本计入性能；测试框架为 golden 保存的 B_original 不属于生产算子可依赖的备份。

不能宣称保守 AIV 天然解决所有数值误差：它提供可检查的执行次序，最终仍以 golden 和特殊值分类验证结果。UNIT 的 AIV 实现保留为正确性基线与资源受限保底；大尺寸 UNIT 与 NON_UNIT 同样推进混合核性能路径，相关性能用例仍须正确执行并满足验收，不因保底分流而豁免。

## 7. Tiling、容量和尾块

Host tiling 的候选字段包括：m/n/K/R、lda/ldb、batchCount、枚举及映射标志、alpha 两分量、实际 AIV/AIC 核数、rhsTile、宏 panelSize、内部微块、更新 tile、团队映射规则参数、batch 分组信息、workspace 偏移。只有 Kernel 真实使用的字段才保留，可由已有参数计算的值不重复存储；按不同 kernel 路径裁剪字段，团队映射采用第 7.1 节的算术规则，参数布局见第 7.3 节。

宏 panelSize 候选为 32/64/128/256，优先评估 128/256 对 Host 下发和设备阶段次数的改善；内部微块候选为 16/32/64，rhsTile 和更新 tile 根据 UB/L1/L0 容量与任务数选择。扩大宏 panel 不等于把整个对角块放入 UB；较大 panel 还会增加 AIV 串行工作、改变 Cube 更新形状，性能不保证单调提升。核数和可用片上容量查询目标平台，不硬编码实验代码中的 20 核，也不假定 A2/A3 相同资源配置。

### 7.1 RHS 团队映射与 L2 局部性

优先评估 batch-major 逻辑任务：`teamId=batch*numSplits+splitIdx`，使同 batch 的 RHS 团队连续排布。MIX_AIC_1_2 下，AIC 团队索引与 teamId 对应；AIV 使用 `teamId=aivBlockIdx/2`、`lane=aivBlockIdx%2` 再映射两个配对成员。任务数大于常驻团队数时按明确循环分配逻辑任务，不将逻辑编号直接当作同时驻留承诺。实际 GetBlockIdx 范围和核配比在目标平台验证。

令 Q=batchCount×ceil(R/rhsTile)。纯 AIV 路径将 Q 与实际可用 AIV 核数比较；MIX_AIC_1_2 中 Q 表示配对团队数，可用团队数按 `min(AIC核数, AIV核数/2)` 及实际可调度资源确定。配对的两个 AIV 均需参与协议，但不能据此推断两个成员始终满负荷。

欠饱和时评估缩小 rhsTile，联合 Cube 的有效 N 宽度及尾部浪费、重复 A 读取、事件开销、并发 workspace 和 L2 局部性选择；每次调整按第 7.2 节重新核算容量并测量完整 API 调用。以 20 AIC/40 AIV、R=16 为示例，batch=1、rhsTile=1 时最多 16 个独立团队；batch=2 时为 32 个团队，混核路径需要分 wave。示例核数不进入实现常量，也不据团队数承诺全部硬件忙碌。

使用实际 CSV 场景验证欠饱和策略：`TC_PF_1169` 的 K=2636、R=57、batch=3，rhsTile=64 时 Q=3，rhsTile=4 时 Q=45，比较两个选择及相邻候选的完整调用结果；这不是预先指定最优 tile。R 极小、无法增加独立 RHS 时，可结合下述共享只读 A-panel 及并行 trailing 输出 tile、加载流水评估，仍按 panel 依赖顺序推进，不任意拆分互相依赖的求解行；引入跨团队依赖须重新证明常驻与握手。

相邻编号是提高同批 A-panel 时间局部性的候选，不保证执行顺序、同 wave、同 panel 或 L2 命中；团队即使同时运行，也可能因尾块、不同路径而进度偏移。按目标平台查询实际 L2 能力，核算活跃 A-panel、B、暂存及并发团队的工作集与复用间隔，不硬编码“近百兆”，不要求整张 A 驻留 L2。

4096 阶 A 的完整 complex64 存储是 128 MiB，实际引用三角约 64 MiB：NON_UNIT 为 64 MiB+16 KiB，UNIT 为 64 MiB−16 KiB（未计 cache-line 粒度）。16 个团队各读一次有效三角产生约 1 GiB 逻辑读取请求，不等于约 1 GiB 必定来自 HBM。比较 batch-major、其他任务顺序、RHS 分片数和活动 batch 数，结合可用 L2 指标、HBM 流量、MTE2 时长及完整调用时间选择策略。

必要时共享只读 A-panel 的转换暂存，减少重复解交织；须明确生产者、所有消费者和回收时点，不能为缓存复用引入等待未调度团队的全 grid 屏障。L2 局部性不作为正确性、常驻性或性能必然达标的前提。

### 7.2 片上与 GM 容量

完整复数对角块的双 FP32 平面占用为 `8*panelSize^2`：128 块需要 128 KiB，256 块需要 512 KiB；128×64 的 RHS tile 另需 64 KiB，尚未计入转换、临时向量和双缓冲。不能默认整块驻留 UB。采用宏 panel 配内部微块的分层方案，并对同时存活的全部缓冲核算容量。

通用 AIV 保底的算法 GM scratch 为 0。混合核按第 6.3 节的实际布局核算有界输入暂存、两结果或四乘积暂存、持久工作 RHS、需要的设备状态及 Matmul 系统 workspace；仅计真实选用的区域，但不能遗漏隐藏于 UB→L1 接口中的 GM 中转。容量按实际并发槽位、内部对齐和双缓冲倍数计算。共享只读 panel 的生产/消费/回收按第 7.1 节定义。不能把每核可写 scratch 重叠分配，也不能为尚未常驻的全部任务重复分配同样的常驻槽位。

持久 SoA 主性能候选保留每个活动团队的完整 K 链、有限 RHS 宽度；设内部对齐为 Ka、r_a，实际并发条带数为 s，其工作 RHS 容量约为 `8*Ka*r_a*s`，另加实数嵌入输入、必要 A-panel 转换及系统 workspace。条带初始化后跨 panel 保留，不能每次更新尾 B 都转换后释放；一个槽仅在该条带全部计算及原 B 最终写回完成后交给下一任务。该布局不要求完整 W_T，但若同时处理所有 RHS，条带总量仍接近完整 W_Y；不能用每团队较小的容量掩盖全体并发容量。

例如 K=4096、rhsTile=128 时，每条带未计 padding 的双 FP32 工作平面为 4 MiB；Case 5 两批、全部 64 个 RHS 条带同时保留时共 256 MiB。用户 workspace 不足时减少并发条带或分组复用，不能强行超过用户容量；组数、A 重读、初始化/写回及调度成本都计入第 11 节性能。按实际可用资源再选择两结果中转或零额外 GM AIV，不将“功能保底”视作性能免验。

全量打包仅为按实测收益选择的独立方案。以 `Ka`、`Ra` 表示内部对齐尺寸，`g` 表示单次活动 batch 数，其主要额外 GM 需求约为：

```text
W_T = 8 * g * Ka * Ka            # 可选：完整复数 T 的两个 FP32 平面
W_Y = 8 * g * Ka * Ra            # 可选：完整工作 RHS / 解；全部持久条带的容量总和
W_total = aligned(W_T) + aligned(W_Y)
          + panel/update scratch + device 状态 +必要的元数据
```

以 `m=n=4096, batch=2` 为例，单个 complex64 矩阵为 128 MiB，全部 B 为 256 MiB。完整 B 的 Pack 读取原 B 并写工作平面，共 512 MiB；Unpack 读取工作平面并写原 B，又为 512 MiB，两阶段合计最低逻辑 GM 流量 `4*256 MiB=1 GiB`，还未包含 A、计算访问及其他 staging。所有条带分别只转换首尾一次时总量同样如此，不因切分自动减少总转换量；每 panel 重新转换尾块则额外增加第 6.3 节所列的访问。仅保留全部 B 工作平面需 256 MiB，全量 W_T+W_Y 额外容量为 512 MiB。若另存 B_original，还须单独追加备份容量和复制流量。上述是逻辑访问及容量模型，不是实测 HBM 流量/耗时，也不是新增内存验收指标。

完整 GM 转换增加可避免的流量，但不能据此断言所有 TRSM 都是访存受限：小 RHS 常受访存或依赖链约束，大 RHS 的分块更新可能具备 GEMM 式复用。是否保留全量打包，须比较含转换、辅助 kernel 和同步后的完整调用收益。

scratch 必须按真正并发的 tile/条带数计算并落实所有权，不能让并行核共用同一可写临时区。必要时按 batch/RHS 分组复用 workspace，组间同 stream 顺序执行，持久 SoA 及所有异步使用的 GM 缓冲持续有效至本组求解和最终写回完成；超出用户容量或库上限时缩小分组/并发槽位，或在写 B 前选择零额外 GM AIV，不能以任务书性能 case 为由溢出容量。

UB 预算计入复数双平面、AoS↔SoA 转换缓冲、对角 panel、RHS tile、临时向量和双缓冲；每次修改 tile 同步重新核算。尾块使用真实长度或 mask，合法 GM 范围之外不发起越界加载；输出 padding 不写。

内部对齐补零只用于确实允许的布局转换，逻辑尺寸、两侧 K-padding 和有效输出范围按第 6.3.3 节验证。实际尾部存取及所用 Matmul API 的内部补齐需在目标环境检查，不能以未初始化数据或原 GM padding 代替零填充。非有限路径不能把不存在的尾部或未引用三角当成额外的 `0*Inf` 参与计算。UNIT 的存储对角必须在加载规则上排除，不能先读出再用 1 覆盖并声称“不读取”。

### 7.3 精简 Tiling 与启动参数检查

`CtrsmBatchedTilingData` 使用标准布局、可平凡复制的小 POD；通过静态断言、sizeof/alignof 及 Host/Device 共用声明检查布局。按值结构不包含逐核团队映射数组，规则映射在 Kernel 内算术计算。以真实 kernel 签名检查完整启动参数包，包含所有指针、按值参数及对齐；不能以单个结构体的尺寸代替总参数规模检查。将此检查落实到 CANN 9.1、目标 arch22 的实际编译和最小启动验证。

较大的数组型配置或 Matmul 配置按所用接口放入有明确容量和对齐的 GM tiling/系统 workspace；核函数只接收设备地址或区域偏移。该区域与可写 staging 不重叠，H2D 传输完成后才消费，存储持续有效至所有消费者完成；异步传输还须保持 Host 源缓冲有效。可选 GM 配置仅属于对应优化路径，不为纯 AIV 基线新增必需 GM scratch。

## 8. 复数算术与数值风险

有限复数乘法使用 `(ar*br-ai*bi, ar*bi+ai*br)`。复数除法不直接照搬 `ar*ar+ai*ai` 分母写法作为所有输入的通用实现，因为极值可能使分母溢出或下溢。

采用按分母最大分量缩放、必要时按二进制指数归一化的复数除法，显式处理零、Inf、NaN 分支；该 helper 的极值行为需对照实际 Netlib 复数运算验证。不得通过给对角额外加 epsilon、截断输出或改变输入矩阵来消除数值问题。NON_UNIT 奇异性检测不属于 API 功能，不新增奇异性错误返回。

### 8.1 向量实现和开销控制

同一对角元素 `a_ii=ar+i*ai` 作用于整个 RHS 向量。Smith 式的 `abs(ar)>=abs(ai)` 可作为每个对角元素/条带一次的统一 Scalar 分支，计算比例、缩放指数及安全除数后广播；Vector 仅执行选中的公式，不必每 lane 求两套结果再 Select。以已确认安全范围、分母非零的输入为例：

```text
若 abs(ar) >= abs(ai):
    t = ai/ar; d = ar + ai*t       # Scalar；极值时先作必要归一化
    out_r = (v_r + v_i*t)/d        # Vector，对整个 RHS 条带
    out_i = (v_i - v_r*t)/d
否则:
    t = ar/ai; d = ai + ar*t
    out_r = (v_r*t + v_i)/d
    out_i = (v_i*t - v_r)/d
```

上述公式仅展示共享分支及向量组织，不是对全部 FP32 输入都安全的完整 helper。分子中间和差也可能溢出，按范围选择普通 Vector 路径与带二进制缩放的稳健路径；分子条件随 RHS 元素变化时仍需逐元素 mask/范围检查。零、Inf、NaN 按第 10.3 节的参考口径验证。可借鉴 LAPACK CLADIV/SLADIV 的缩放思路，不能只缩放分母就声称所有中间运算安全。

普通安全范围内增加共享标量倒数的性能候选：一次 Scalar 计算 `inv_d=1/d`，对该主元的 RHS 条带使用 `Muls` 代替重复 Vector `/d`，例如 `out_r=(v_r+v_i*t)*inv_d`、`out_i=(v_i-v_r*t)*inv_d`。准入须同时满足分母及倒数范围、分子中间结果范围和目标指令的次正规数处理要求；具体范围由目标 FP32 指令、舍入方式及 FTZ 行为核验确定，不能仅检查 `d!=0`。不满足准入条件的元素采用经验证的二进制缩放或稳健除法路径，不能先产生 Inf/NaN 再靠 Select 掩盖。

共享倒数不是一般等价改写。例如按 IEEE FP32，实数分子与分母均为 `2^-130` 时直接商为 1，但先计算 `1/d` 会溢出为 Inf；乘回原分子已不能恢复有限商。极大分母产生的次正规倒数也可能受 FTZ 影响，必须在实际目标核上验证。即使处于普通范围，先舍入倒数再乘与直接除法也可能有不同舍入误差，长链回代须重新通过逐 batch 精度和特殊值验收，不把单个除法段的正确性当作整批结果保证。

不无条件预计算 `1/a_ii` 后做复乘：极小分母与同量级分子可能有有限商，而倒数先溢出；复乘中间大项相减也可能制造 NaN。不同时计算可能含非法除法的两个分支再用 Select 掩盖结果。可复用的安全系数仅在生命周期和数值范围明确时缓存；UNIT 完全跳过对角读取和除法。

除法规模约 O(KR)，三角更新约 O(K²R)。大 K 下优先测量更新和同步瓶颈；小 K、窄 RHS 单独统计 Scalar 预处理、共享倒数及乘法、Vector 除法和极值路径成本，再测量完整 panel / API 收益。不写入未经目标平台测量的单周期或“提升 5～10 倍”承诺，也不为减少指令删除数值保护。UNIT 不含对角除法，因此共享倒数优化不能解决 Case 3/5 的主要更新和同步瓶颈。

### 8.2 数值和特殊值约束

alpha=0 优先进入专用路径。其余情况下，对 alpha 两分量和被缩放值 v 两分量均有限、且经整条求解验收的快速路径，按以下分量规则实现缩放：

| alpha 条件 | 缩放规则 |
|---|---|
| alpha=(1,0) | 跳过缩放 |
| alpha_i=0，非零 alpha | `out_r=alpha_r*v_r`，`out_i=alpha_r*v_i` |
| alpha_r=0，非零 alpha | `out_r=-alpha_i*v_i`，`out_i=alpha_i*v_r`，使用原始两分量后再写回 |
| 其他有限 alpha | 常规复数乘法；按既定误差规则验证 |

分量优化减少零乘法和交叉运算，但不作为所有非有限值的默认语义。Netlib CTRSM 源码未规定纯实/纯虚 alpha 的专用缩放；LEFT+T/C 分支仍直接执行 TEMP=ALPHA*B，没有 alpha=1 的统一短路。alpha 或实际被缩放值非有限时，按实际链接的参考库及 side/trans 分支核验缩放、求解和短路次序；只有参考行为已验证一致的分支才允许 identity 跳过或分量优化。不能以避免 0×Inf 为理由自行更改 golden 类别。有限输入产生的后续非有限中间量也按第 6.4 节验证，不由初始有限性保证整条路径正确。

需要特别防止以下改写造成错误：

- alpha=0 使用专用写零，不能通过普通乘法路径处理。
- UNIT 不读取对角，不使用对角存储值检查 NaN/Inf，也不做 NON_UNIT 的 boost。
- 三角之外、padding、对齐补区不参与真实数学计算。
- 用复数模长、相对误差或更高精度 CPU 解替代任务指定的逐分量 Netlib 验收，都必须另行明确，不能暗中替换标杆。

对于非有限数据，保守路径按实际 Netlib 的 side/uplo/trans 分支审查求解、缩放和短路次序；数学统一映射不等同于特殊值传播逐位一致。若与目标 cuBLAS 的类别不一致，按第 10.3 节记录处理，不硬编码本次 CSV 的输出。

大尺寸 UNIT 三角矩阵虽数学上非奇异，但非对角随机值可能引起解快速增长、病态放大及 FP32 溢出。当前数据不得为通过测试而缩小、归一化或修改。对这类结果记录 golden 的有限/Inf/NaN 数量；不能事先将整类跳过，也不能保证单靠关闭 HF32 或采用 ULP 上限就能满足数值和特殊值验收。

### 8.3 按计算流水验证 FP32 行为

在目标 CANN、A2/A3 实际设备上分别核验下列路径，记录指令/API、编译选项、数据类型、可用配置及输出；不将某一条流水的结果外推为所有 FP32 运算均具有相同行为。

| 路径 | 验证内容 |
|---|---|
| Scalar | 复数 helper 使用的除法、倒数及指数缩放，正常数边界与次正规数 |
| Vector | 实际乘加、纯实/纯虚缩放、除法及实际启用的共享倒数路径 |
| Cube | 关闭 HF32 的 FP32 Mmad/Matmul、归约增长、抵消及有效尾部 |
| FIX | 实际使用的非量化 FP32 搬出，类别及有效输出范围 |
| GM AtomicAdd | 两平面初始化、FP32 累加的溢出/特殊值/次正规行为及发布完成依赖 |

输入覆盖正常数边界、次正规输入/输出、溢出、Inf/NaN；同时保留能区分直接除法和倒数乘法的同量级极小分子/分母场景。配置只采用目标产品、数据类型和编译器支持的公开接口，Cube HF32 控制按实际所用 API 初始化并验证。某一控制位的设置不代替上述逐路径检查，初始化记录与整条求解结果共同作为优化准入依据。

发现次正规数处理或特殊值传播不满足参考及第 10 节规则时，采用已验证的二进制缩放、稳健 helper 或保守计算路径，再验证完整求解；不通过改变输入、输出截断或放宽比较器掩盖。此处是数值正确性验证，不新增专项内存验收或次正规结果逐位一致要求。

## 9. 工程落点与参考代码使用边界

以下目录树以 ops-blas 根目录为基准，是后续新增/修改计划，不代表本次已创建源码：

```text
include/cann_ops_blas.h                         # 新增正式 CtrsmBatched 声明
blas/trsmbatched/README.md                      # 补充 complex64 / A2/A3 说明
blas/trsmbatched/arch22/
  ctrsmbatched_host.cpp                         # 校验、局部 vector 同步取表、资源、tiling、调度
  ctrsmbatched_tiling_data.h                    # Host/Device 共用 POD
  ctrsmbatched_kernel.h                         # do 启动函数声明
  ctrsmbatched_kernel.cpp                       # zero、单次 launch AIV、可选数据分类/tile staging
  ctrsmbatched_blocked_kernel.cpp               # 分块基线、设备内循环混合核，稳定后按需新增
test/trsmbatched/ctrsmbatched/
  CMakeLists.txt
  ctrsmbatched_param.h                          # CSV + compatibility disposition
  ctrsmbatched_golden.h                         # Netlib、no-op、zero reference
  ctrsmbatched_verify.h                         # 本算子明确的有限/特殊值比较
  README.md                                    # 新测试工程步骤与本次裁决说明
  arch22/
    ctrsmbatched_npu_wrapper.h                  # 真正分配设备矩阵与设备指针表
    ctrsmbatched_test.cpp
    ctrsmbatched_test.csv                       # 配套原 CSV 的逐字节副本
```

当前正式公共头只有 StrsmBatched，没有本任务的 CtrsmBatched。复用 `cann_ops_blas_common.h` 类型和状态码，不增加 C 输出版本、指针检查开关或其他公共接口。

指针判空缓冲、复制及错误清理封装在 ctrsmbatched_host.cpp 内部；不为该缓存修改公共 handle/helper。工程仍可按实际集成需要更新公共声明、同族 README 和必要构建配置，不把任务书目录描述扩大为无依据的文件修改禁令。

当前 `blas/CMakeLists.txt` 按架构收集源码；`test` 框架支持 `trsmbatched/ctrsmbatched` 这种家族子目录。测试按同族方式调用 `ops_blas_add_gtest_tests`。应实际验证新增目标被 `--ops=ctrsmbatched` 发现，不能仅凭放置文件就认定已接入。

构建配置将 `ascend910b*` 和 `ascend910_93*` 映射为 arch22。A2/A3 使用目标设备实际匹配的 SOC 参数；不能机械地对所有 A3 都使用 README 示例的 ascend910b3。平台支持标注和实际验证硬件记录分开，未跑过的硬件不填写已验证。

Kernel 初始化落实第 8.3 节已支持的配置及所需流水依赖，Cube 路径在相关计算前配置并验证 HF32 关闭。编译记录包含真实 kernel 签名、POD 布局和完整参数包检查结果；数值微测试记录各流水的实际行为，未通过准入的优化分支不启用。两个 SOC 的编译与设备验证分别记录，不能用 Host 结构大小或单条指令测试替代完整路径检查。

可参考但不可直接照抄的现有代码：

| 参考 | 可复用思路 | 必须重新处理的差异 |
|---|---|---|
| `blas/trsmbatched/arch35/strsmbatched_*` | handle 优先检查、同步 D2H 地址值取表/局部 vector 判空、Host/tiling/do 组织 | 实数、arch35/SIMT、batch=0 规则不同；默认先等待绑定 stream 再同步取表，全表预检后统一下发；第 4.3 节详析 |
| `experimental/aclblasCtrsmBatched2/` | AIV panel、复数拆分、Cube 更新、设备内循环与配对事件研究 | 实验 ABI、Host 指针表、行主约定、硬编码核数、alpha0 和同步策略不同；须重审常驻映射、尾任务握手和缓冲可见性 |
| `blas/hemm/arch22/chemm_kernel.cpp` | arch22 FP32 Cube、关闭 HF32 的用法 | HEMM 与三角求解的依赖、UNIT、特殊值行为不同 |
| `test/trsmbatched/strsmbatched/` | CSV/GTest/CBLAS 工程模式 | 不得沿用 batchCount=max(1,...) 或默认误差策略；内部空元素真实测试，不修复输入后计 PASS |

不移植 arch35 SIMT 内核作为 arch22 已支持功能。实验工程已有性能报告不能视为本任务 200 条 CSV、当前参数契约或新实现的性能证明。

## 支持硬件

| 目标产品 | 工程与验证要求 |
|---|---|
| Atlas A2 | arch22；性能验收设备为 Atlas 800T A2（910B3），尚待实际编译和运行验证 |
| Atlas A3 | arch22；使用设备实际匹配的 SOC，单独记录编译及运行结果，不用 A2 结果替代 |

## 算子约束限制

- 仅支持 COMPLEX64 和 Host alpha，输入输出列主序，B 原地覆写；不新增 C/ldc 接口。
- no-op、alpha=0、UNIT、leading dimension 和设备表空元素处理以第 2～4 节最终契约为准。
- 调用者维持非空地址的容量、表内容和异步生命周期；前置判空不验证任意地址合法性，也不提供后续执行失败的整批事务回滚。
- 仅启用通过目标编译、数值、同步和容量验证的优化路径；不因内存免验或性能目标省略必要的资源与正确性检查。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 最终口径 | 详细依据 |
|---|---|---|
| 功能与用例 | 1200 条原始记录，1197 条适用，三条 C/ldc 用例单列 N/A；错误返回来自真实 API | 第 2、3、10 节 |
| 精度 | 每 batch 的实/虚通道独立，99% 混合容差及逐元素 max(1e-2,32 ULP)；特殊值类别和 Inf 符号全部匹配 | 第 10.3 节 |
| 性能 | 200 条性能用例均验证正确；5 次预热、10 次有效完整调用采样；以基线换算门限验收 | 第 11 节 |
| 内存 | 不设专项内存验收；容量、越界防护和异步资源生命周期仍须正确 | 第 4.4、7.2、12 节 |

## 兼容性分析

原地输出和主要数学语义对齐 cuBLAS，任务特定的零维豁免、Host-only alpha、状态码及设备表同步判空边界见第 2.1 节。正式实现复用公共类型和资源机制；局部 Host 判空缓冲不扩展公共 handle，不改变其他算子的全局精度标准。原始资料不改写，兼容差异通过第 2 节确认记录及测试适配显式保留。

## 10. 测试适配、golden 和精度验收

### 10.1 保持原始资料不变

测试加载原始 CSV 的逐字节副本，或明确只读其路径；禁止默认运行 `requirements/test_cases/gen_csv.py`，因为它会重写 CSV 并将 GPU 基线生成为空时间占位。需要研究生成规则时，仅允许内存构建或输出到独立临时目录。

CSV 按表头读取 19 列，检查列数、case_name 唯一性及清单完整性，不静默丢行。解析保留所有原始字段，并产生单独的有效参数/处置记录：

- 空白 lda/ldb 按紧凑值解析；显式的 0、负数、非法值必须保留，不能当成缺省。
- batchCount 原样传给 API，不使用 `max(1,batchCount)`。
- `alpha_real=null` 构造实际空 alpha 指针；`a/b=NULLPTR` 构造实际空顶层数组。
- `handle_null` 构造真实空 handle 并调用生产 API；其优先级高于其他检查。`TC_ED_190/null_a_elem`、`TC_ED_192/null_b_elem` 构造真实设备指针表并保留指定空元素，调用生产 API 核对 INVALID_VALUE；wrapper 不把空元素重新分配成有效地址，测试不代造返回码。
- 非法枚举保留为不属于合法集合的值，不静默降级成默认枚举。
- 只有第 3 节三个 C/ldc 精确 ID 记 N/A；其余 API 返回值必须来自真实调用，内部空元素不再豁免。
- `alpha0_zeroC` 的兼容解释只改变实际输出检查目标，不修改 CSV 原文。

负值和空指针测试在构造资源前识别分支，避免先按负 batchCount 转为无符号大数分配。非法 lda/ldb 的测试缓冲可安全超额分配，但传给 API 的原始值不能修正。失败返回前不调用 CBLAS 生成无意义 golden。

原 CSV 不变，在新测试工程补充状态机和检查顺序的组合覆盖：零维/零 batch 加负数或非法枚举应 INVALID_VALUE；合法 no-op 的 alpha/A/B 可空、lda/ldb 可零；任一组合的空 handle 优先 HANDLE_IS_NULLPTR。另检查“前一个 batch 有效、后一个空元素”时全部 B 保持哨兵值，以及同 stream 前序设备任务生成指针表后立即调用 API 能正确取表。这些补充项目单列，不改写原 1200 条清单计数。

### 10.2 输入和 golden

数据规则按 CSV：复数实/虚部独立采样；采用现有填充器的标签含义，特殊填充保留。固定并记录每个 batch 的种子派生，A/B 分离种子空间；GPU 基线复现或参考对照必须使用同一实际输入，不能仅称“种子一样”而忽略不同随机库。

测试侧对 A 清理未引用三角，并在 NON_UNIT 时添加任务指定的 `boost=max(5,K)`。复数没有统一的正负号；本方案将“符号保持偏移”具体化为仅对实部施加 `real += (real>=0 ? boost : -boost)`，虚部保持采样值，用以确保对角远离零。此为测试填充细节的设计选择，应在未来工程的 `test/trsmbatched/ctrsmbatched/README.md` 固定下来，不修改 requirements 下的 README；golden/NPU 必须共享 boost 后完全相同的 A。算子本身不得做 boost。UNIT 不 boost、不强行修改存储对角来掩盖错误读取。

保存 B_original。常规成功用例逐 batch 调用 Netlib `cblas_ctrsm(CblasColMajor, ...)`，golden 从 B_original 的独立副本计算，不能拿已经被 NPU 覆写的 B 做 golden。记录实际链接的 BLAS 库路径和版本；发现名为 libblas 的库不等于已经证实 Netlib。

no-op 只核对真实 API 状态和无计算行为，不向 Netlib 传入该任务特例的 lda=0 来生成 golden。alpha=0 的 golden 直接构造有效 m×n 区域全零，不解引用空 A。

比较仅覆盖每个 B[b] 的有效 m×n 区域，不把 ldb padding 计入精度分母。已有 padding 用例保留哨兵并检查未被改写，属于这些用例的原地输出正确性检查，不扩展为专项内存验收。

### 10.3 有限数与非有限值的明确规则

对每个 batch、每个实/虚分量通道独立执行：

1. 先检查有限/Inf/NaN 分类；不先计算差值。
2. golden 有限的位置，actual 必须有限，否则整条失败；golden 为 Inf 时 actual 必须同号 Inf；golden 为 NaN 时 actual 必须也是 NaN，不比较 NaN 的符号/payload。
3. 特殊值类别和 Inf 符号全部匹配，不用 1% 宽限容纳类别错误。
4. 对 golden/actual 均有限的集合 F，令 `e=abs(actual-golden)`、`tol=2^-13 + 2^-13*abs(golden)`；采用足够宽的 Host 计算类型统计差值，避免比较器自身 FP32 溢出。
5. 若 F 非空，要求 `count(e<=tol)/|F| >= 0.99`；另对每个有限位置 j 要求 `e_j<=max(0.01,32*ULP_FP32(golden_j))`，上限违例数必须为 0。混合容差的整数通过数至少为 `ceil(0.99*|F|)`；实部和虚部都须通过，每个 batch 都须通过。
6. 若 F 为空，有限误差比例/最大误差记 N/A；该通道由特殊值匹配结果决定，不能虚报最大误差为 0。

ULP 按每个 FP32 golden 分量的绝对值所在二进制区间定义：零和次正规数使用 `2^-149`；其他有限数使用 `2^(floor(log2(abs(golden)))-23)`。实际实现用指数位或 frexp 提取区间，避免 log2 在二次幂边界的舍入；以 double 计算差值和上限。不要使用 `nextafter(max_float,+Inf)-max_float` 作为该定义，否则最大有限数处会得到 Inf 并错误取消约束。

`2^-149` 是比较器的 ULP 定义；有限次正规结果仍按上述混合容差和绝对上限判定，不额外要求逐位保留。若次正规处理改变中间分母、依赖链或特殊值类别，则仍须通过第 8.3 节路径验证和最终输出验收。

例如 golden=1000 时 32 ULP 为 0.001953125，绝对上限仍为 0.01；golden=10000 时 32 ULP 为 0.03125，上限为 0.03125。两者都还须满足每 batch/通道的 99% 混合容差条件。此处采用逐元素 max 规则，不再叠加含义不清的额外“2 ULP”放宽。

不聚合全部 batch 来平均通过，不用矩阵最大值的 ULP 放宽其他元素，不把绝对上限替代 99% 混合容差。最大绝对误差仍报告为诊断量，但超过 0.01 不再单独导致失败；最终取决于上述两个条件及特殊值匹配。零输出路径可直接检查有效区域为复数零。

当前 `test/frame/verify.h` 的逐元素 `max(fixedLimit,32*ULP)` 可参考，但默认 FLOAT32 混合容差和特殊值处理不符合本任务口径，CSV 的 MERE/MARE 两列也不代表上述完整规则。新增局部比较器或显式配置能表达全部规则的接口，不直接使用默认值，不为本任务修改其他算子的全局精度标准。

`TC_FL_154/155` 保留执行、检查 API 状态、stream 同步状态和全矩阵输出类别。沿现有复数填充器时，其 B 分别为 `(Inf,Inf)`、`(NaN,NaN)`；不能预设输出必然全 Inf 或全 NaN。其他用例若生成非有限输出，也适用同一规则，不能仅对两个固定 ID 启用比较器。

若 Netlib 与目标 cuBLAS 对同一实际输入给出不同类别：保留双方输出、环境版本和差异位置，标记参考口径待核实，暂不宣称 PASS，也不自动过滤。按前轮建议，特殊值可采用经记录确认的目标 cuBLAS 类别作为例外参考；不存在实测证据时仍使用 Netlib，不编造 cuBLAS 结果。任何整条豁免或输入修改都需单独说明，不能扩展第 3 节已确认的三条 N/A 名单。

### 10.4 运行器与报告

`requirements/test_cases/verify_accuracy.py` 保持不变，作为原始参考脚本。最终验收通过新测试工程直接输出 GTest XML/结构化结果；若需要新的汇总运行器，放在测试工程中，不能回写 requirements。

汇总器必须处理 GTest 的真实退出码、超时、崩溃、零条执行、PASS/FAIL/SKIP 与三个允许的 N/A。不能仅扫描到几个 PASS 就宣布全部通过。按当前清单，精度运行应发现 1000 条，其中仅 3 条预期 N/A；正常完成时 997 条得到实际判定。补充组合测试另列，过滤运行明确标注为子集，不出具全量通过结论。

设备选择必须传递到真正的 `aclrtSetDevice`，并记录实际设备：旧脚本设置 `ASCEND_DEVICE_ID`，当前测试基类则使用 `TEST_DEVICE_ID`，不能仅设置环境变量就宣称换卡已生效。

报告逐 case 记录原参数、原始期望、处置原因及原因类别、API/同步状态、每个 batch 实/虚有限比例、最大绝对误差、逐元素绝对上限违例数及位置、特殊值数量及不匹配数量。N/A 的 API/同步状态记未执行，不伪造 SUCCESS 或 INVALID_VALUE；全特殊值通道的有限误差指标记 N/A。不仅输出整体布尔结果。

### 10.5 已确认优化的补充实现检查

以下项目在新增测试工程中单列，不修改 requirements 或原 CSV，不改变原始 1200 条、1197 条适用及三条 N/A 的统计口径：

- alpha 缩放：K=1、UNIT，提供有效 A/B 表和矩阵地址，存储对角仍不读取。使用 B=(Inf,2)/(2,Inf)/(NaN,2)，alpha=-1/100/i/1，覆盖 LEFT/RIGHT、N/T/C；记录实际 Netlib 的逐分量类别，cuBLAS 差异按第 10.3 节处理。另用有限及极值输入验证分量优化的完整求解精度。
- Cube 尾部与提交：覆盖实际 panel 尾长、RHS 尾宽、两侧 K-padding、有效输出写回及初始化哨兵；验证有限输入产生非有限 panel 解时的准入，以及暂存检查、局部 AIV 分支和两侧握手。检查实/虚完整更新，不能只验证其中一个平面。
- 欠饱和调度：使用实际 batch×R，核对纯 AIV 与混核任务分配、分 wave、尾任务协议及缩小 rhsTile 后的容量；按第 7.1 节对实际窄矩阵性能用例测量完整调用，不仅比较任务数。
- Tiling 与数值流水：完成第 7.3 节真实签名编译及最小启动、第 8.3 节 Scalar/Vector/Cube/FIX/AtomicAdd 微测试，再回归正式用例；所有记录注明目标设备和工具链版本。

补充检查验证真实实现风险，单独报告其执行及结果。算子内部因准入产生的分类、暂存、检查和转换成本仍属于第 11 节完整调用，测试侧独立微测试不混入正式 CSV 性能样本。

## 11. 性能验收设计

性能设备按任务要求为 Atlas 800T A2（910B3）。A3 的适配结果另行记录；不能以一个 SOC 的编译成功替代另一设备的运行验证。

每条性能用例按配套指导预热 5 次，再有效采样 10 次。若 msprof 工具自行重放，须核对实际调用次数，不能在测试中再嵌套重复造成 10×10。每次预热和每个有效样本前均恢复同一份 B_original；不对上次的解再次执行 TRSM。输入恢复、golden 计算、结果下载在 kernel 求解耗时范围之外，并记录实际边界。

### 11.1 完整调用耗时与下发分析

一次 API 调用先等待绑定 stream，再同步 D2H 取回必要指针表，然后可能是 zero、单个 AIV 求解或单个设备内循环混合核，也可能包含 scale、数据分类、panel solve、update、按实现产生的暂存/合并及可选备份/恢复/重算等辅助工作。持久 SoA 条带及全量打包路径须包含初始化转换和最终写回。每次样本统计该次调用实际执行的全部相关逻辑 device kernel 耗时之和，再对 10 次取平均；不能只取最快的一个 kernel，或把全部 panel 的平均当成一次完整求解耗时。Host 前置等待和同步取表不直接加到所选计算 kernel 的 Task Duration；复制与计算根据实际记录归组，不把 D2H 当作计算 kernel。

混合核按一次逻辑 launch 归组，使用工具可确认的完整 kernel 时长；如果 profiler 将 AIC/AIV 子任务分别列出，应按关联 ID 和起止时间还原该逻辑 kernel 的覆盖跨度，不把内部重叠子任务时长重复相加当作其墙钟耗时。原始子任务明细保留作利用率分析。当前方案只用绑定 stream；若将来引入并行 stream，需重新定义重叠统计，不能继续把 kernel 时长求和当作设备墙钟时间。

同时记录四类量，注明测量边界：

| 指标 | 定义与用途 |
|---|---|
| CPU 提交时间 | Host API 入口至返回，包含参数检查、前置 stream 等待、局部 vector 管理、同步 D2H、tiling/资源准备及下发；不含调用者随后等待求解完成的时间 |
| 完整 kernel 时长总和 | 本次求解全部逻辑 kernel 的时长之和；按第 11.2 节比较门限 |
| 设备执行跨度 | 本次调用可关联的首项设备复制或计算工作开始至最后一项完成，包含设备侧间隙及实际辅助复制；同步 D2H 不假设属于 handle.stream，须借助 API/任务关联确认；入口等待的既有前序工作不记成本次新产生的设备工作 |
| 同步端到端时间 | API 入口至该次结果完成并同步返回，包含实际资源准备、下发及等待；与 CPU 提交可能重叠的设备计算不能重复累加 |

默认生产路径和正式性能样本都保留完整指针表预检，不设置检查旁路。报告独立列出前置等待、同步 D2H 及前序工作对 Host 延迟的影响；冷启动、局部 vector 管理和 workspace 扩容与预热后的稳态分别记录。单调用测量前明确 stream 中无无关待执行任务，避免把其他工作归入该调用；连续提交的吞吐分析另行记录队列边界，明确入口同步会影响流水重叠。辅助复制若不是 kernel，也须出现在可确认的设备跨度和端到端结果中；不能用“未计入 kernel 门限”声称同步无成本。

profiling 以单个 case 收敛采集，记录采样 ID/调用边界，将原始 OpBasicInfo 等明细与汇总对应。warmup、测试侧输入恢复、CBLAS 和日志处理不进入有效求解 kernel 样本；算子内部的备份、恢复和重算不能按“输入恢复”排除。记录实际 kernel 数量、宏 panel/微块、batch 分组、staging/重算是否启用，避免优化减少下发后沿用旧的样本归组规则。没有 trace 和测量结果前，不根据 launch 次数直接断言必然增加若干毫秒或必然未达标。

#### 11.1.1 采集模式、任务归组与重放准入

同步取表消除了本方案原先两次 MemcpyAsync 排入绑定 stream 的具体统计来源，不保证 profiler 文件只包含计算 kernel。区分算子级 msprof op 与系统级 msprof：前者 OpBasicInfo 描述算子名、类型、Task Duration、Block Dim 等；后者可以采集 ACL/Runtime 的同步和异步复制及 task/API 明细。不能断言每个 MemcpyAsync 必定在 OpBasicInfo 有一行，也不能假定该表一定存在 Task Type 或调用关联字段。正式解析前记录 CANN、profiler 版本、命令及实际表头。

运行器按以下规则归组，不采用 memcpy/dma 名称黑名单：

1. 为每个有效 API 调用记录独立的 case/sample ID 和采集范围，并核对实际提交的计算 kernel 数量。名称/任务类型/关联 ID 按安装版本的真实字段读取；字段不足时联合系统 timeline 或 runtime 明细，不虚构关联。
2. 使用实现实际生成的 kernel 明确清单，覆盖 zero、scale、分类、panel solve、update、staging、Pack/Unpack、combine 及内部备份/恢复/重算；前缀或名称匹配只用于选择已知实现，不以名称含 dma 就认定是复制。结合调用范围剔除测试侧输入恢复等工作。
3. 对复制/API 记录单列耗时；对混合核依照前文还原一次逻辑 launch，不能把重叠 AIC/AIV 时长重复相加。不能累加整个文件的所有行，也不能漏掉内部辅助计算。
4. 无法可靠确认任务范围、缺失 kernel、存在未知记录或调用次数不一致时，样本记测量无效并重新采集，不能通过静默丢行得出达标结论。

当前官方 msOpProf 说明中，未指定 kernel-name 时只采首个算子，launch-count 默认 1，replay-mode 默认 kernel；正式环境须核对 CANN 9.1 安装版本，不机械套用其他版本参数。多 panel 依赖链不可采用默认只采首 kernel 的结果；需显式覆盖全部内部 kernel，并核对实际计数。单 kernel 重放必须确认依赖输入快照及原地状态，不能把对已求解 B 的重复计算当作新样本。优先选择能保留完整依赖链的 application/range 采集或系统 timeline；range 数量限制、混合核/二级指针支持情况及各模式的缓存策略均按安装工具验证，不保证所有模式都适用。

每次完整调用仍恢复 B_original，核对工具重放与运行器采样的关系，避免重复 10×10；十个有效样本须对应十次完整求解。缺少可用的全链采集方式时，可用经验证的系统任务时间记录形成完整 kernel 统计，单核 profiler 留作瓶颈分析，不能用不完整的 OpBasicInfo 替代验收数据。

### 11.2 基线关联和阈值

按原 CSV 的 TC_PF 顺序关联 baseline：前五条对应 `ctrsmBatched-base-000` 至 `004`，后续对应 `ctrsmBatched-perf-005` 至 `199`。关联后逐项核对 m/n/batchCount/side/uplo/trans/diag，并保存 case_name→baseline_id 映射。存在重复参数组合，不能仅用这七个参数做覆盖式字典 join。

单位必须统一：200 条统一使用 `limit_us = gpu_ms * 1000 / 0.8`，以基线原始数值计算、不预先舍入。配套 README 说明任务书五条耗时由该公式得到，因此任务书表值作为显示舍入值保留，不再另设第二套门限。五条重点如下：

| CSV case | batchCount | m×n | side/uplo/trans/diag | 任务书显示值 us | baseline 换算 limit_us |
|---|---:|---|---|---:|---:|
| TC_PF_1001 | 64 | 256×256 | LEFT/LOWER/N/NON_UNIT | 895.1 | 895.0875 |
| TC_PF_1002 | 128 | 384×512 | LEFT/UPPER/C/NON_UNIT | 5551 | 5550.6375 |
| TC_PF_1003 | 32 | 1024×1024 | RIGHT/LOWER/T/UNIT | 13884 | 13883.875 |
| TC_PF_1004 | 8 | 2048×2048 | LEFT/UPPER/N/NON_UNIT | 21816 | 21816.375 |
| TC_PF_1005 | 2 | 4096×4096 | RIGHT/UPPER/C/UNIT | 40969 | 40969.125 |

全部 200 条性能 case 均保留；包括 8×8 调度开销场景和大 UNIT 场景。性能前检查输出符合既定数值/类别规则，不能用 no-op、错误计算或未提交输出的 kernel 成绩验收。性能未达标时分别分析下发、转换流量、panel 串行段、更新吞吐、可选重算和设备同步；本设计不预先承诺已经达到阈值。

## 12. 实施步骤和完成条件

| 阶段 | 产出 | 完成条件 |
|---|---|---|
| 1. 接口与测试适配 | 正式声明、完整状态机、局部 vector 同步取表及全表判空、零路径、原 CSV 加载、三条 N/A 和状态报告 | 原始资料 hash 不变；适用负向真实调用，含 ED190/192；组合优先级及前置等待正确；no-op/alpha0 行为可复现；不新增公共 handle 缓存 |
| 2. 完整功能基线 | 零额外 GM AIV、24 组合、padding/非方阵、稳定复数 helper、已验证的有限 alpha 分量缩放、精简 POD 及逐元素 ULP 比较器 | 997 条适用精度类的真实状态及输出全部 PASS，3 条 N/A 单列；补充组合/取表、alpha 类别及真实 kernel 参数检查通过；实际使用的数值流水完成验证；Inf/NaN 单列验证，无隐式过滤；零 scratch 准入验证完成 |
| 3. 性能优化 | 覆盖大 UNIT/NON_UNIT 的设备内循环混合核、多 kernel 基线、分层 panel、两结果平面及持久 RHS 条带 SoA/AtomicAdd 布局对比、完整工作 RHS 与最终 B 写回协议、L2 映射及欠饱和 tile 评估；重算仅在显式备份路径启用 | 精度回归、尾部/完整复数更新准入、同步协议和容量核算通过；所用 Cube/FIX/AtomicAdd 数值验证及优化补充检查通过；200 条性能用例均正确，完整 kernel 耗时按 10 样本统计并全部达到第 11 节门限；采集无漏项，记录取表等待、转换、分组、下发、设备跨度和端到端开销 |
| 4. 平台和交付 | A2/A3 支持说明、实际设备结果、运行步骤、日志与精度/性能报告 | 前述功能/性能阶段通过，约定平台验证完成；写清设备、CANN/BLAS版本、提交版本；不提交内存验收件 |

存在 FAIL、未达性能门限、非名单 SKIP 或未执行项目时，对应阶段仍为未完成。可以交付问题分析和进展报告，但解释失败原因不能代替验收通过。

必需的实现级检查围绕风险开展：handle/负数/枚举/no-op 组合优先级、六种数学映射及 RIGHT 原位地址、共享倒数的安全范围/极值/FTZ 与稳健除法、真实设备表内部空元素、同 stream 指针表生产依赖及同步复制错误清理、alpha=0 的 A 豁免、padding 保持、混合核尾任务握手及完整工作 RHS/最终 B 写回、AtomicAdd 初始化及事件依赖、零 scratch 路径和异步资源生命周期、有限/非有限及 ULP 比较器、重复性能样本的 B 恢复及采集完整性。第 10.5 节另覆盖有限 alpha 分量优化与参考类别、Cube 尾部及完整更新提交、欠饱和任务和分 wave、真实参数包编译及分流水数值行为。补充项目与原 CSV 分开报告，不引入专项内存验收。无需为文档文字或数据表重写一套镜像测试；新检查应能检出上述真实失误。

以下风险必须在开发报告中有结论：

- 必要设备表内部空元素保留同步错误码保证，三条 N/A 只涉及 C/ldc；局部 vector、前置等待和同步取表不依赖 arch22/arch35 差异，入口阻塞、求解异步的边界和成本须如实记录，不得按性能用例关闭检查。
- Cube 路径的结果中转或持久 SoA 布局分别核算；产品就绪、工作 RHS 就绪与最终 AoS B 写回不得混淆，AtomicAdd 不是 L0C 读取旧 B 的偏置融合，也不替代发布流水、wait、配对参与者和 Scalar 缓存验证。
- batch-major 只提供局部性候选，不保证同 wave 或 L2 命中；零额外算法 scratch 不取消 handle/runtime 资源，也不保证任意 OOM 环境成功。
- 多 kernel 下发与设备间隙须实测；128/256 宏 panel 配微块及常驻混合核是否有收益，取决于串行求解、片上容量、更新形状和事件成本。
- 纯实/纯虚及 identity 缩放仅在有限快速路径或参考已验证的分支启用，极值类别不能由省略零乘法自行定义；Cube 逻辑尾部和物理补齐分别验证，提交前数值准入覆盖完整复数更新，结果暂存检查不能被已提交后的回退代替。
- 欠饱和分别按纯 AIV 核数、混核配对团队数及实际 wave 评估；缩小 rhsTile 的占用率收益须与重复读 A、Cube 尾部、事件和容量成本一起实测。精简 POD 的完整 kernel 参数及实际使用的数值流水，均须完成目标编译和设备验证。
- 默认不独立保存原始 RHS；直接 B 或持久 W 工作状态不能充当备份。透明重算须证明对应范围仍有完整原始输入、写回边界及同步协议正确，并核算新增备份/恢复成本，不能仅凭尚有工作区就承诺整批重启。
- A2/A3 的实际编译能力、Cube 精度、UB/L1/L0 使用和 API 名称需目标环境验证。
- 大 UNIT 的数值增长、非有限传播及每 batch/通道 99% 混合容差与逐元素 ULP 上限可能限制优化路径；不能修改原输入、忽略类别差异或再次放宽阈值掩盖。
- CBLAS 链接和特殊值行为必须有可追溯版本，不能把另一 BLAS 实现当作 Netlib 而不记录。
- 所有 200 条性能目标仍需跑测，实验原型数据不能直接继承。
- profiler 采集范围、字段、重放输入状态与逻辑 kernel 完整归组需实测确认；改用同步复制不取消这些要求，不以名称黑名单或仅首个 kernel 的数据判定达标。

# 附录

## 13. 原始输入快照和引用

### 13.2 原始资料和源码依据

原始任务资料按下列外部路径和第 13.1 节摘要定位；源码链接固定到本方案核对的 ops-blas 版本。仓库源码或实验实现的引用仅用于说明设计依据，不代表本任务已完成实现或验收。

- 任务书（`aclblasCtrsmBatched_A2A3_task_doc.md`）、测试 README（`test_cases/README.md`）、原始 CSV（`test_cases/ctrsmbatched_test.csv`）、生成器（`test_cases/gen_csv.py`）。
- [公共 ABI](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/include/cann_ops_blas.h)、[类型与状态码](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/include/cann_ops_blas_common.h)、[handle/workspace](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/blas/common/helper/aclblas_handle_internal.h)。
- [handle 创建及 workspace 设置](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/blas/common/helper/aclblas_auxiliary.cpp)：仅参考现有默认资源与零 workspace 参数边界；本版不在此接入算子专属 Host 缓存。
- [arch35 指针表初始化与调用顺序](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/test/trsmbatched/strsmbatched/arch35/strsmbatched_npu_wrapper.h#L100)：先同步 H2D 写表再调用 API；用于解释已有同步 D2H 方案的就绪条件，不证明异步表生产者已被入口等待。
- [同族 Host](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/blas/trsmbatched/arch35/strsmbatched_host.cpp)、[实验复数 Host](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/experimental/aclblasCtrsmBatched2/op_host/ctrsm_batched_host.cpp)、[实验 AIV panel](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/experimental/aclblasCtrsmBatched2/op_kernel/ctrsm_batched_kernel_aiv_solver.h)、[arch22 CHEMM](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/blas/hemm/arch22/chemm_kernel.cpp)。
- [实验混合核入口](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/experimental/aclblasCtrsmBatched2/op_kernel/ctrsm_batched_kernel.cpp)、[实验 AIV 循环](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/experimental/aclblasCtrsmBatched2/op_kernel/ctrsm_batched_kernel_aiv.h)、[实验 AIC 更新与握手](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/experimental/aclblasCtrsmBatched2/op_kernel/ctrsm_batched_kernel_aic.h)：用于研究设备内循环，不作为目标平台正确性或性能证明。
- [同族测试](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/test/trsmbatched/strsmbatched/arch35/strsmbatched_test.cpp)、[复数填充](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/test/frame/fill.h)、[公共比较器](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/test/frame/verify.h)、[测试构建](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/cmake/test.cmake)。

### 13.3 外部语义参考

- [NVIDIA cuBLAS trsmBatched](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsmbatched)：接口、原地输出与参数语义。
- [Netlib CTRSM 源码](https://www.netlib.org/blas/ctrsm.f)：列主三角求解、参数检查、零 alpha 写零与分支运算次序；无纯实/纯虚 alpha 专用缩放，LEFT+T/C 的 alpha 乘法不能统一视为 identity 短路。
- [LAPACK CLADIV/SLADIV](https://www.netlib.org/lapack/explore-html/d5/db7/group__ladiv_ga3a6531b09431433d2e81b1db7a691dd2.html)：避免不必要中间溢出的复数除法参考；不替代本任务与实际 Netlib golden 的结果核对。
- [aclrtMemcpy 同步复制语义](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/appdevgapi/aclcppdevg_03_0105.html)：复制自身完成不等于隐式同步 Device/stream；用于说明 arch35 裸同步复制的生产者排序边界。
- [CANN Runtime 数据复制指南](https://gitcode.com/cann/runtime/blob/master/docs/02_dev_guide/02-01_%E6%95%B0%E6%8D%AE%E5%A4%8D%E5%88%B6.md)：同步复制等待本次复制、异步复制的 stream 排序及 Host 内存约束；默认采用同步方法，正式实现仍核对安装的 CANN 9.1.0 运行环境。
- [CANN 9.1 IterateAll](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/advanced_api/matrix_compute/Matmul-Kernel/IterateAll.md)：enAtomic=1 为 AtomicAdd；不将它解释为旧工作 B 已在 L0C 通过 bias 融合。
- [TCubeTiling](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0673.html)、[CANN 9.1 SetTail](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/advanced_api/matrix_compute/Matmul-Kernel/SetTail.md)：ND 原始尺寸、内部基本块约束与有效尾部配置；所用 API 的实际补齐、写回及数值行为仍需目标验证。
- [核函数直调与框架开发对照](https://asc.gitcode.com/guide/programming_guide/advanced_programming/aclnn_operator_development/appendix/kernel_direct_call_comparison.html)：直调按值结构与 GM tiling 的工程形式；本方案检查实际完整 kernel 参数包。
- [SetHF32Mode](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0258.html)：Cube Mmad 的 FP32/HF32 控制；其他数值行为按实际流水和数据类型验证，不由此设置外推。
- [华为产品目录](https://www.hiascend.com/s/ascendstatic/lst/files/pdf/Flying_Product_Catalogue.pdf)：公开 FP16/FP32 指标有别，不能以 FP16 280 TFLOPS 代替严格 FP32 Cube 吞吐；目标 910B3 配置仍须实测。
- [msOpProf 数据字段](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md)、[采集与重放说明](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)：OpBasicInfo 字段、kernel 选择、launch 数及重放模式；实际参数按 CANN 9.1 安装版本确认。
- [CANN 系统级 profiling](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0011.html)：ACL/Runtime 同步与异步复制的采集，用于补足调用关联和时间边界，不与算子级 OpBasicInfo 混同。
- [CANN 9.1 UB→L1 搬运](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_UBToL1_continuous.md)：A2/A3 的软件 GM 中转，不将 LocalTensor 原型误读为私有 UB 可跨核直接寻址。
- [CrossCoreSetFlag](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md)、[CrossCoreWaitFlag](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.md)：发布流水、配对参与者、A2/A3 wait 参数边界及 flag 冲突。
- [DataCacheCleanAndInvalid](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cache_control/DataCacheCleanAndInvalid.md)：DMA 完成依赖与 Scalar DCache 一致性分别处理。

这些引用不替代第 2 节已确认的本任务裁决，也不构成当前实现已在任何 GPU/NPU 上通过验证的声明。

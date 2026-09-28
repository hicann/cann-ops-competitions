| 项 | 内容 |
| --- | --- |
| 任务编号 | 本地任务书 `aclblasCtrsmBatched_A2A3`；社区 `docs/README.md` 任务表尚无同名 A2/A3 条目，目录名待评审确认 |
| 任务名称 | 9月社区任务-aclblasCtrsmBatched 算子开发（Atlas A2/A3） |
| TeamName | ATEAM |
| 设计文档路径 | `04_tasks/01_community-task-2026/tasklist/9月社区任务-aclblasCtrsmBatched算子开发(A2A3)/ATEAM/docs/design.md` |
| 目标代码仓 | https://gitcode.com/cann/ops-blas.git ，基线 `master` `2c9b9b7` |
| 实现目录 | 合入目标 `blas/trsmbatched/arch22/`；测试 `test/trsmbatched/ctrsmbatched/arch22/` |
| 目标硬件 / arch | Atlas A2 / Atlas A3 系列，arch22 |
| 对标接口 | cuBLAS `cublasCtrsmBatched`；精度 golden 为 Netlib `ctrsm` 逐 batch |
| 当前实验代码 | `ops-blas/experimental/aclblasCtrsmBatched2/`，尚未进入上述合入目录 |

# 需求背景（required）

## 需求来源

- 任务书：`ops-blas/task/aclblasCtrsmBatched_A2A3_task_doc.md`。
- 目标硬件：Atlas 800I A2 / Atlas 800I A3（Atlas A2/A3 系列）。§3.1 写自验证覆盖一种款型、精度与性能建议在 Atlas 800T A2（910B3）采集；§7 写 A2 与 A3 均需验收。两处口径不一致，待评审确认。CANN 9.1.0。
- 目标仓：https://gitcode.com/cann/ops-blas 。合入路径以任务书第 5 节为准：`blas/trsmbatched/arch22/`。第 2.2 节写的 `blas/trsm/` 与第 5 节、仓内已有 `blas/trsmbatched/` 不一致，本设计沿用第 5 节，待评审确认。
- 对标：cuBLAS `cublasCtrsmBatched`。公开 ABI 以任务书签名和 `include/cann_ops_blas.h`、`include/cann_ops_blas_common.h` 为准。任务书把 `alpha` 收紧为仅 Host 指针，并把 `A[]`、`B[]` 规定为设备侧指针数组。
- 依据优先级：任务书 > 公开头文件 > 仓内同族实现 > 本地实验代码。实验代码与任务书冲突时，以任务书为合入目标，实验差异列入待对齐，不改写成已满足。

## 背景介绍

`aclblasCtrsmBatched` 对 `batchCount` 个尺寸相同、地址独立的三角系统求解，解原地覆写右端矩阵：

- `side = ACLBLAS_SIDE_LEFT`：`op(A[i]) * X[i] = alpha * B[i]`，`A[i]` 为 `m×m`
- `side = ACLBLAS_SIDE_RIGHT`：`X[i] * op(A[i]) = alpha * B[i]`，`A[i]` 为 `n×n`

`op(A)`：`ACLBLAS_OP_N` 为 A，`ACLBLAS_OP_T` 为转置，`ACLBLAS_OP_C` 为共轭转置。`uplo` 选择上/下三角，`diag = ACLBLAS_UNIT` 时对角视为 1 且不读取。矩阵列主序，前导维 `lda`、`ldb`。数据类型为 `aclblasComplex`（COMPLEX64）。

| 项 | 当前基线（`ops-blas` `2c9b9b7`） | 本任务 |
| --- | --- | --- |
| 接口 | `cann_ops_blas.h` 仅有 `aclblasStrsmBatched`（FP32） | 新增 `aclblasCtrsmBatched`，不另造平行符号 |
| 硬件 | `blas/trsmbatched/arch35/`，README 标明 A2/A3 不支持 | arch22，A2/A3 |
| 功能 | 实数批量三角求解 | COMPLEX64，side/uplo/trans/diag 全组合 |
| 测试 | 无本接口的 arch22 CSV 工程 | `test/trsmbatched/ctrsmbatched/arch22/`，golden 为逐 batch `ctrsm` |

本地实验实现位于 `experimental/aclblasCtrsmBatched2/`。它已有 MIX AIC:AIV = 1:2 的 panel 求解与 trail GEMM，但声明使用 `experimental/aclblas_minimal.h` 和 `std::complex<float>` / `int64_t`，不是公开头文件中的 `aclblasComplex` / `int`。该目录不是合入形态。本任务不涉及 TBE、ACLNN、广播或 PyTorch。

# 需求分析（required）

## 需求描述

在 ops-blas 句柄式 BLAS 上实现 `aclblasCtrsmBatched`：调用方 `aclblasCreate` 后用 `aclblasSetStream` 绑定 stream，Host 校验并 tiling，向该 stream 异步入队 arch22 Kernel。支持 COMPLEX64、列主序、uniform batch，以及 LEFT/RIGHT、UPPER/LOWER、N/T/C、UNIT/NON_UNIT。`m = 0` 或 `n = 0` 为成功 no-op。`alpha = (0,0)` 时不读 A，把各 `B[i]` 原地置零。不支持超出 `lda`/`ldb` 的非连续访问，不做奇异性检测，不要求逐位确定。950 / arch35、FP32 `aclblasStrsmBatched`、FP64 不在本提交范围。

## 需求拆解

1. 在 `include/cann_ops_blas.h` 增加任务书签名，标量类型为 `int` 与 `const aclblasComplex*`，不保留实验代码的 `int64_t` / `std::complex<float>` 作为公共 ABI。
2. 覆盖任务书 dtype、shape、format：COMPLEX64、ND、列主序；`m,n ≥ 0`；`batchCount ≥ 1`；`lda`、`ldb` 满足前导维约束。
3. Host 校验与错误码按任务书 §2.4：`handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举、负维度、`batchCount < 1`、空 `alpha`/`A`/`B`、数组元素空指针、非法 `lda`/`ldb` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. Kernel 沿用实验实现的 panel 三角求解（AIV）加 trail 复数 GEMM（AIC），Host 用 `CtrsmBatchedTilingData` 传参，不用 TilingKey。
5. Quick return：`m = 0` 或 `n = 0` 直接成功；`alpha = (0,0)` 置零后成功且不启动求解 Kernel。
6. 精度按 opbase COMPLEX64 混合容差；性能按任务书 5 个 case 的达标耗时。本设计阶段不填写未实测数据。
7. 不改变已有 `aclblasStrsmBatched` 的 arch35 行为。
8. 测试接入 ops-blas CSV / GTest，用例来自 `ops-blas/task/test_cases/`，缺项按任务书 §3.5 补齐。

# 详细设计（required）

## 算子分析

对每个 batch `i` 独立求解，只引用 `uplo` 指定三角。`ACLBLAS_UNIT` 不读对角。`ACLBLAS_NON_UNIT` 使用存储对角，对角须非零，本算子不检测奇异。`ACLBLAS_OP_C` 在读取 A 时对元素取共轭后再按转置访问。列主序下 `A[i](r,c)` 位于 `A[i] + c * lda + r`，`B` 同理使用 `ldb`。

`alpha` 为 Host 标量。任务书规定 `A`、`B` 为设备侧指针数组，长度 `batchCount`，各 `B[i]` 不得重叠。实验 Host 把调用方数组按 Host 指针 `aclrtMemcpy` 到设备后再给 Kernel，与任务书不一致，合入时改为直接使用设备侧指针数组。

空与特殊值：

- `m = 0` 或 `n = 0`：成功，不写 B，不读 A。
- `batchCount < 1`：`ACLBLAS_STATUS_INVALID_VALUE`。实验代码把 `batchCount == 0` 当成成功，合入时改正。
- `alpha = (0,0)`：不读 A，按列主序把每个 `B[i]` 的 `ldb * n` 个复数置零。实验置零长度是 `m * ldb` 个复数，合入时改为 `ldb * n`。
- Inf/NaN：按输入传播，对齐 cuBLAS，不额外清洗。
- 重复或越界索引：不涉及；指针数组元素为空时返回 `ACLBLAS_STATUS_INVALID_VALUE`。实验代码未逐元素检查，合入时补上。

Workspace：每个有效 batch（及列拆分）需要规范化后的 A/B 工作区和 GEMM 临时区，大小由 `workspaceOffset`、`splitBGemmSize` 决定，另加 Matmul 系统 workspace。调用异步：Host 只入队，不在 API 内 `aclrtSynchronizeStream`。实验 `LaunchCtrsmKernel` 在返回前同步 stream，合入时去掉，由调用方在读回前同步 `aclblasSetStream` 绑定的 stream。实验分配失败返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。头文件已有 `ACLBLAS_STATUS_ALLOC_FAILED`，合入时分配失败沿用仓内同族用法，不新增状态码。

## 算子实现

层次只有三层：公开 `aclblasCtrsmBatched`、arch22 Host、arch22 MIX Kernel。无 Python、ATen、TBE。

### Host

入口目标文件：`blas/trsmbatched/arch22/` 下的 Host 源文件。校验顺序与任务书一致。通过后：

- `m == 0` 或 `n == 0`：返回成功。
- `alpha` 实部与虚部均为 0：置零各 `B[i]` 后返回成功。
- 否则填充 `CtrsmBatchedTilingData` 并启动 `ctrsm_batched_mix12_kernel`。

Tiling 字段与实验 `op_kernel/ctrsm_batched_tiling_data.h` 一致：`m,n,lda,ldb,batchCount,nb,side,uplo,transa,diag,alphaReal,alphaImag,workspaceOffset,useOrigA,aEffStride,dualAivMode,numSplits,splitNCols,splitNColsAligned,lastNCols,lastNColsAligned,splitBGemmSize`。

分派不用 TilingKey。Kernel 内按 `side` 与有效 `uplo` 选四个编译期模板：`CtrsmLowerLeft`、`CtrsmLowerRight`、`CtrsmUpperLeft`、`CtrsmUpperRight`。`transa` 为 T/C（LEFT）或 N（RIGHT）时先规范化 A，并在转置时交换上下三角，再进入上述四路径。

分块与分核沿用实验 Host，不另起一套：

- `kDim = side==LEFT ? m : n`，`nCols` 为另一维。
- `nb = kDim > 1024 ? 32 : 16`。
- `kDim ≥ 128` 且对齐后列数 `≥ 128` 时 `dualAivMode = 1`，同一矩阵由该 MIX 单元的两个 AIV 分列。
- 仅在双 AIV 且 `batchCount` 不超过平台 AI Core 数的一半时，按列拆成多个 block，最小列宽取 `max(nb, 64)`，拆分对齐 16。
- `useOrigA`：不需要转置、共轭、补齐时直接用原 A，`aEffStride = lda * 2`；否则工作区步长为对齐后 `kDim * 2`（按 float 计）。
- Cube tiling 用 `matmul_tiling::MultiCoreMatmulTiling`，A/B/C 为 GM、ND、FP32、无 bias，`SetDim(1)`。复数乘由 AIV 把 AoS 拆成实部/虚部 SoA 后交给 AIC。

实验代码把 AI Core 总数写成常量 20。这是实验机参数，不是产品规格。合入时改为平台查询得到的 core 数，分核公式保持不变。

`ldb` 约束以任务书为准：`ldb ≥ max(1, m)`。实验校验写成 `ldb ≥ max(1, n)`，合入时改正。`lda`：LEFT 时 `≥ max(1, m)`，RIGHT 时 `≥ max(1, n)`。

### Kernel

入口：`ctrsm_batched_mix12_kernel`，`KERNEL_TYPE_MIX_AIC_1_2`。AIC 与 AIV 各走 `Init` 后的 `Process12`。不是单条 CopyIn/Compute/CopyOut 流水；一个 panel 的阶段是：

1. AIV 读入当前 panel 的 A 与对应 B 行，做 AoS 到 SoA、按 `uplo`/`transa`/`diag` 规范化，并乘 Host 传入的 `alpha`。
2. AIV 在 panel 内做复数前代或回代，写回已解出的 X。
3. AIV `CrossCoreSetFlag` 通知 AIC。
4. AIC 用 Matmul 更新 trail，再 `CrossCoreSetFlag` 交回 AIV。
5. AIV 等待后写回该 panel 行，并准备下一块。

`diag = UNIT` 时求解不读对角存储。双 AIV 时两个 Vector 核按列分段做 panel 求解和回写。未单独实现的 Device 侧查重、奇异检测、枚举再校验不写入本设计。实验代码未打开与 panel 无关的第二套双缓冲，本文不把它写成可流水。

Kernel 实际依赖的运行时能力：Ascend C `TPipe`、GM 读写、UB、`CrossCoreSetFlag` / `CrossCoreWaitFlag`、Matmul `REGIST_MATMUL_OBJ`。Host 侧为 `aclrtMalloc`、`aclrtMemcpy`、`aclrtMemset`、Kernel 启动。合入后 API 本身不再调用 `aclrtSynchronizeStream`。

### 目录与构建

```text
ops-blas/
├── include/cann_ops_blas.h
├── blas/trsmbatched/
│   ├── README.md
│   ├── arch35/                  # 既有 FP32，不改语义
│   └── arch22/                  # 本任务
│       ├── ctrsmbatched_host.cpp
│       ├── ctrsmbatched_kernel.cpp
│       ├── ctrsmbatched_kernel_aic.h
│       ├── ctrsmbatched_kernel_aiv.h
│       ├── ctrsmbatched_kernel_aiv_solver.h
│       ├── ctrsmbatched_kernel_aiv_convert.h
│       ├── ctrsmbatched_kernel_aiv_canon_a.h
│       ├── ctrsmbatched_kernel_aiv_canon_b.h
│       └── ctrsmbatched_tiling_data.h
└── test/trsmbatched/ctrsmbatched/arch22/
```

源文件从实验目录迁入并改成仓内命名与公共头文件，不保留 `experimental/` 作为交付路径。CMake 只在 arch22 / A2/A3 SOC 编译上述源文件，与 arch35 的 `strsmbatched` 源文件隔离，避免符号和编译宏冲突。README 产品表：Atlas A2/A3 系列（含 Atlas 800I A2/A3）为支持；950 保持既有 `aclblasStrsmBatched` 表述，不把本接口写成 950 已支持。

## 支持硬件

- Atlas A2 系列：本任务实现目标
- Atlas A3 系列：本任务实现目标，与 A2 同 arch22 源
- Ascend 950PR / 950DT：本提交不含

自验证采集一种款型。性能表的达标耗时对应任务书给出的 Atlas 800T A2（910B3）口径。本文不记录开发机型号或设备号。

## 算子约束限制

- 仅 COMPLEX64。无 FP16、BF16、FP32、FP64，无 strided batched。
- 仅列主序。`lda`/`ldb` 可以大于紧凑前导维；不支持任务书明确排除的其他非连续访问。
- uniform batch：各 batch 共享 `m,n,lda,ldb` 和 side/uplo/trans/diag。
- `batchCount ≥ 1`。`m,n ≥ 0`。零行或零列是 no-op。
- `alpha` 必须是 Host 指针。`A[]`、`B[]` 必须是设备指针数组，元素非空，`B[i]` 互不重叠。
- 不检测零对角或奇异矩阵。
- 不保证浮点累加逐位一致。
- 调用方负责 stream 同步后再读 B。
- 不覆盖 950，不修改 `aclblasStrsmBatched`。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64：`rtol = atol = 2^-13`，`matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2` 或 32 ULP。逐元素 `\|actual-golden\| ≤ atol + rtol×\|golden\|`。含规约时 `max_abs_error_limit` 可放宽到 2 ULP，放宽需在自测报告中说明 | 任务书 §3.2，opbase 混合容差 |
| 性能标准 | Atlas 800T A2（910B3），COMPLEX64，msprof 平均单次耗时，有效采样 10 次（工具自带 5 次 warmup）。case1–5 上限：895.1 / 5551 / 13884 / 21816 / 40969 us | 任务书 §3.3 |
| 内存标准 | 不涉及 | 任务书 §3.4 |

实验目录中的 320-case 报告对比的是另一套 GPU 比值，不是上表门限。本设计不引用那些倍率，也不把它们写成已达标。arch22 合入形态的精度和性能待测。

## 测试设计

正向：LEFT/RIGHT、UPPER/LOWER、N/T/C、UNIT/NON_UNIT 正交；`alpha` 为 0、1、-1 及一般复数；紧凑与带 padding 的 `lda`/`ldb`。

形状：`m,n` 覆盖 0、1、小质数、2 的幂及 ±1、非对齐值到大规模；`batchCount` 从 1 扫到 1024。任务书性能 5 case 单独列出。

边界与负向：`m` 或 `n` 为 0；`m,n,batchCount` 为负；`batchCount = 0`；`handle`、`alpha`、`A`、`B` 及数组元素为空；非法枚举；非法前导维。期望状态码按 §2.4。

特殊值：B 含 Inf/NaN，行为对齐 cuBLAS。golden 用 Netlib `ctrsm` 逐 batch 生成，全矩阵比对。

现有材料：`ops-blas/task/test_cases/ctrsmbatched_test.csv` 共 1200 条（无独立 C/`ldc` 列，输出按原地覆写 B），以及 `verify_accuracy.py`、`gen_csv.py`。这些脚本尚未接到 `test/trsmbatched/ctrsmbatched/arch22/` 的 GTest。没有 arch22 官方工程的通过记录，不写全部通过。

测试指导与任务书有两处冲突，合入以任务书为准：`test_cases/README.md` 写成离席 `Carray`/`ldc`，CSV 本身没有这两列；`TC_ED_184`（`batchCount=0`）期望 `ACLBLAS_STATUS_SUCCESS`，任务书 §2.4 要求 `batchCount < 1` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。接入 GTest 时改这条期望，不把 README 的离席接口写进 ABI。

性能方法：先 warmup，再对 5 个性能 case 各取超过 10 次的有效平均。内存报告按任务书写“不涉及”。

## 兼容性分析

- 新增：`aclblasCtrsmBatched` 与 arch22 源、arch22 测试。公共头文件只增加声明。
- 增量架构：A2/A3 新目录，不替换 arch35 `aclblasStrsmBatched`。
- dtype/shape：仅新增 COMPLEX64 批量三角求解；不扩展 FP32 接口的 shape 规则。
- 并行贡献：与其他 blas 算子目录无源文件共享。若同时修改 `cann_ops_blas.h` 或 `blas/trsmbatched/README.md`，合入时只追加本接口段落。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 实验 ABI、`batchCount==0`、`ldb`、置零长度、Host 指针数组、API 内同步与任务书不一致 | 合入前按本文 Host 节改校验和启动路径，再用负向 CSV 核对状态码 |
| 写死的 core 数不能代表 A2/A3 各款型 | 分核改为读取平台 core 数；性能只在任务书指定款型上采 |
| panel/GEMM 的复数规约可能超过 1 ULP | 按任务书混合容差判定；若需放宽到 2 ULP，在自测报告单列，不提前写成已通过 |
| 大 `batchCount` 的设备指针数组与 workspace 占用高 | 按 `workspaceOffset * 有效块数` 分配；失败返回分配错误。不在本设计里增加新的分批算法 |
| 任务书 §2.2 与 §5 目录不一致 | 代码放 `blas/trsmbatched/arch22/`，与现有 batched 目录和 §5 一致，并在评审中确认 |
| §3.1 自验证一种款型，§7 要求 A2 与 A3 都验收 | 实现共用 arch22 源；采集范围待评审确认，本文不把未测款型写成已通过 |
| 随包 CSV 的 `batchCount=0` 期望与任务书相反 | 负向用例按 §2.4 改为 `ACLBLAS_STATUS_INVALID_VALUE` |

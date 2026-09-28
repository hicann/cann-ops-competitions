# aclblasStrmm（Atlas A2/A3）算子设计文档

> 设计评审更新，对应个人私有代码仓 `ops-blas` 的 `feature/aclblas-strmm-arch22-a2a3` 分支，代码提交 [`5490618`](https://gitcode.com/m0_56676577/ops-blas/commit/549061889d7d0d8a6a584db44381d027540f7f3e)。严格读取及小矩阵分块优化已实现，910B3 精度/功能 1207/1207、20 次采样性能 200/200 通过；版本指纹和验证范围见 [VALIDATION.md](https://gitcode.com/m0_56676577/ops-blas/blob/549061889d7d0d8a6a584db44381d027540f7f3e/test/trmm/strmm/arch22/VALIDATION.md)（需私仓访问权限）。本 PR 只更新设计文档，不公开算子源代码、测试代码或原始日志；A3 实机及正式验收材料仍待补齐，历史 CSV 不代替当前版本证据。

## 严格读取与小矩阵分块优化

小矩阵按有效区间读取，常规打包只搬运有效三角，单位对角在 UB 合成。`LEFT/LOWER/N` 的 16/32 行小矩阵改为每核 8 行 × 16 列，批量搬运有效矩形并裁剪对角边缘；通过广播乘法和跨列批量树形规约减少重复 DMA、标量发射及同步开销。新路径对非有限数和绝对值不小于 1e18 的输入回退 Netlib 标量顺序，避免极值乘积/累加溢出重排；保留零 B 跳过规则及其它路径的保护分支。16×16、32×32 在完整 20 次采样回归中分别为 3.8990/4.3191 µs；另用 `msprof op` 独立回放各采 20 次，均值为 4.6261/4.5501 µs，两种方式均低于 5.5037/5.6262 µs 上限。

## 需求背景（required）

### 需求来源

2026 年 9 月社区任务 `aclblasStrmm` A2/A3 算子开发。任务要求在 [ops-blas](https://gitcode.com/cann/ops-blas) 工程内实现单精度实数三角矩阵乘，接口语义对齐 cuBLAS `cublasStrmm`，在 Atlas 800T A2（910B3）和 Atlas 800I A3 上完成验收。现有 `arch35` 面向 Ascend 950，本设计新增 `blas/trmm/arch22/` 实现。

### 背景介绍

STRMM 计算一个三角矩阵与一般矩阵的乘积，并把结果写入独立的输出矩阵 `C`。`A` 为三角矩阵，`B`、`C` 为列主序的 `m × n` 矩阵。与普通 GEMM 相比，必须遵守 `uplo` 和 `diag` 的读取约束：不能把未使用的半个 `A` 或单位对角线上的原始值带入计算。

## 需求分析（required）

### 需求描述

对 `side=LEFT`，计算 `C = α · op(A) · B`，其中 `A` 为 `m × m`；对 `side=RIGHT`，计算 `C = α · B · op(A)`，其中 `A` 为 `n × n`。`op(A)` 仅支持不转置 `N` 和转置 `T`。数据类型为 FP32，矩阵为列主序，`lda/ldb/ldc` 表示各列起始地址之间的元素跨度。

接口沿用 `include/cann_ops_blas.h` 中已有的公共声明：

```cpp
aclblasStatus_t aclblasStrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const float *alpha, const float *A, int lda,
    const float *B, int ldb, float *C, int ldc);
```

### 需求拆解

1. 校验句柄、枚举、维度、前导维和指针；`m=0` 或 `n=0` 直接成功，`α=0` 不读取 `A/B`，仅清零 `C` 的有效元素。
2. 覆盖左右乘、上下三角、`N/T`、单位/非单位对角、带 padding 的前导维、边界与负向输入；支持 `B` 与 `C` 同址。
3. 以 Netlib BLAS 单精度 `strmm` 为 golden，按任务书 FP32 混合容差验证；在 910B3 上逐条比较性能用例与 GPU 基线。
4. 保留已有 `arch35` 产品路径，不改变公共函数签名。

## 详细设计（required）

### 算子分析

列主序下，`A[i,j]` 的地址为 `A + j·lda + i`，`B[i,j]` 和 `C[i,j]` 分别按 `ldb`、`ldc` 寻址。逻辑计算为：

```text
LEFT:  C[i,j] = α · Σ(k=0..m-1) op(A)[i,k] · B[k,j]
RIGHT: C[i,j] = α · Σ(k=0..n-1) B[i,k] · op(A)[k,j]
```

数学语义只允许 `uplo` 指定的三角部分参与求和；`diag=UNIT` 时对角因子取 1，不应依赖 `A` 的对角元素。`N/T` 的差异通过索引变换或 Cube 转置处理。支持 `B≡C`，但不支持其它 `C` 与 `A/B` 的任意重叠。加载范围按上述规则裁剪，不通过读取无效区域再掩码来替代读取约束。

### Host 侧设计

入口 `aclblasStrmm` 位于 `blas/trmm/arch22/strmm_host.cpp`。它先检查 `handle`，随后检查 `side/uplo/trans/diag` 和非负维度；非空矩阵时再检查前导维与 `alpha/C`。`m=0` 或 `n=0` 在访问矩阵指针之前返回成功。`alpha` 可从 Host 读取；若位于 Device，则在绑定的 stream 上拷回并同步该读取。`α=0` 时用异步 memset 清零 `C` 的 `m × n` 有效区域，`ldc>m` 的 padding 保持不变。

其余输入按以下条件分派，均使用 handle 绑定的 stream 发射 kernel：

| 条件 | 路径 | 目的 |
| --- | --- | --- |
| `α ≥ 2` 或 `α ≤ -2` | 参考顺序 kernel → 输出拷贝 | 按接近 Netlib 的累加次序处理大幅值缩放后的 FP32 误差 |
| 未命中上一分支且 `m≤32,n≤32` | 小矩阵直接 kernel | 避免打包、GEMM 和多次启动开销 |
| 其它情况 | 三角打包 → Cube GEMM → 按需缩放/拷贝 | 以矩阵乘单元处理常规及大规模矩阵 |

上述阈值是当前实现的路径选择条件，不是输入规格限制。Host 从运行平台查询 AIV/AIC 核数，根据列数或 tile 数限制启动核数；对工作区大小使用 64 位乘加和上限检查，再通过库的默认工作区接口分配。`StrmmPackTilingData`、`StrmmDirectTilingData`、`StrmmReferenceTilingData`、`StrmmScaleTilingData` 在 `strmm_tiling_data.h` 中传递矩阵尺寸、步长、枚举与 `alpha`。

### Kernel 侧设计

**小矩阵路径。** `strmm_direct_kernel` 按 `side/uplo/trans/diag` 只加载有效三角，UNIT 对角在 UB 合成。对于 `LEFT/LOWER/N`、m=16/32、2≤n≤32、紧凑输出且 B≠C，按 8 行 × 16 列分配 AIV block：每核只读取这 8 行所需的 A 片段及对应 B 列，输出写入互不重叠的 32 字节行块。UB 广播 B 后做 FP32 乘法，各输出列通过 repeat stride 一起树形规约；NaN/Inf 和可能因重排溢出的输入保留标量顺序。其它小矩阵保留按列路径；短列、padded C 或 B≡C 使用单 block，先完整缓存 B 再写 C，避免原地写出破坏输入。新路径不增加 kernel 启动或 GM 工作区。超过 32 位 DMA 字节 stride 范围的前导维保留单核通用路径，用 64 位地址读取；已以 ldb=1073741840、约 8 GiB B 分配验证不发生 stride 截断。

**常规路径。** `strmm_pack_kernel` 把三角矩阵整理到对齐的 packed 工作区：无效半区和尾部填 0，单位对角写 1。对 `T` 模式保留 `A` 的源方向，由 Cube GEMM 的转置输入模式处理。利用列主序 `m×n` 的字节布局等同于行主序 `n×m`，把输出转置视图交给 GEMM：`LEFT` 对应 `Bᵀ·op(A)ᵀ`，`RIGHT` 对应 `op(A)ᵀ·Bᵀ`。`strmm_gemm_kernel` 复用工程的 `cblas3` tile 与分段 K 调度，关闭 HF32 模式。通常先写入对齐的临时矩阵，再由 `strmm_scale_kernel` 乘 `alpha` 并按 `ldc` 写回；在 `alpha=1`、`B≠C`、紧凑前导维且 `m≤128,n≥64` 时，GEMM 可直接写 `C`，省去缩放 kernel。

**大幅值路径。** `strmm_reference_order_kernel` 依三角结构和转置模式逐项累加到临时矩阵，随后拷贝到 `C`。这是精度优先分支，不把它描述为性能最优路径。

常规路径的工作区近似为 `alignGM(4·packedLd·dim) + 4·tempLd·n` 字节，其中 `dim` 为 `side=LEFT ? m : n`，`packedLd` 是 `dim` 向打包 tile 对齐后的行跨度，`tempLd` 是 `m` 向临时矩阵对齐粒度取整后的行跨度。大幅值路径只需 `4·tempLd·n` 字节；小矩阵直接路径不申请矩阵工作区。工作区分配失败或大小越界返回相应错误，不执行 kernel。

### 精度与异常行为

- `handle=nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举（包括 `trans=C`）、负维度、非法前导维或非空矩阵所需指针为空，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- `α=0` 时不读取 `A/B`。`diag=UNIT` 时不读取 A 对角，未选三角也不读取；通过裁剪 DMA 或条件读取保证，不依赖这些位置预先为零。
- FP32 结果以 Netlib BLAS 为单一 golden，验收阈值为 `rtol=atol=2^-13`、`matched_ratio≥0.99`，并检查任务书要求的最大绝对误差限制。浮点规约顺序不保证逐位一致。
- 发射与清零遵守 handle 的 stream 异步语义；读取结果前由调用者同步 stream。Device `alpha` 的读取需要在 Host 分派前同步该标量。

## 支持硬件

| 产品 | 状态与证据 |
| --- | --- |
| Atlas 800T A2（910B3） | CANN 9.1.0 环境完成自测 |
| Atlas 800I A3 / A3 系列 | `arch22` 实现已接入；尚未完成 A3 实机验证 |
| Ascend 950PR / 950DT | 沿用仓库既有 `arch35` 路径，不属于本次实现改动 |

## 算子约束限制

仅支持 FP32、列主序矩阵、`N/T` 转置模式；`A` 为一侧指定的三角矩阵，`B/C` 为一般矩阵。支持 `lda/ldb/ldc` 指定的列间 padding，不要求更一般的非连续 Tensor 或广播。`m,n` 在运行时传入；不要求确定性逐位结果。输入 `A/B` 位于 Device，`C` 为 Device 输出，`alpha` 按当前实现可位于 Host 或 Device。

## 可维可测分析

### 精度标准与测试

`test/trmm/strmm/arch22/` 收录任务包的 1000 条普通精度用例、200 条性能用例、逐项 GPU 基线、Netlib 适配与 GTest 代码；另有空句柄、`α=0` padding、`B≡C`、Device `alpha` 等专项测试。覆盖形状扫描、左右乘、上下三角、转置、单位对角、非法参数及特殊值。测试编译和复现步骤见该目录的 `README.md`。

在 Atlas 910B3 物理 7 号卡、CANN 9.1.0 环境，1000 条普通用例、7 条专项及 200 条性能用例的 Netlib 对照全部通过，精度/功能合计 **1207/1207**。性能精度按 GTest 4 分片各 50 条执行，case 名无遗漏、重复、失败或跳过；源码和二进制哈希在结束后全部复核一致。专项覆盖尾列、padding、正负 alpha、原地写出、混合特殊值和有限大数溢出，另有超大前导维诊断通过。原始证据见 `VALIDATION.md`，不代替 A3 验收。

### 性能标准与测试

任务性能判据为每条用例的 NPU 平均单次 kernel 时间 `≤ gpu_ms / 0.8`。采集时对每次调用的 `strmm_*` kernel 的 `Task Duration(us)` 求和，再对有效调用求平均；采集脚本位于 `test/trmm/strmm/arch22/report_perf_all.py`，当前逐条数据见 `perf_results_910b3_accept20.csv`（旧 CSV 仅保留历史证据）。

| 典型形状 | 256×256 | 512×512 | 1024×1024 | 2048×2048 | 4096×4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 910B3 实测平均耗时（µs） | 16.33 | 28.64 | 71.65 | 308.79 | 2176.58 |
| 任务上限（µs） | 58.43 | 138.19 | 342.21 | 1182.02 | 6681.36 |

当前记录为预热 5 次后每条有效采样 **20 次**，200/200 条达到基线目标，满足任务书“有效采样 >10 次”的口径。16×16、32×32 另采用 `msprof op --aic-metrics=BasicInfo` 各采 20 次，均值也达标；流水诊断的单次计时不混入验收均值。性能采集与并行精度任务分开执行，没有选择性丢弃失败样本。A3 性能与精度、正式截图和内存报告仍待补齐。

### 兼容性分析

公开 API 声明已存在于 `include/cann_ops_blas.h`，本改动新增 `arch22` 实现并保留 `arch35`。README 中按产品区分支持状态和精度标准；A3 的产品支持结论应以实机验收为准。

### 待完成和风险

- 严格读取与两条小矩阵性能阻塞已修复；仍需在正式评审中审查 DMA 范围，哨兵测试不能单独证明物理未读取。取消极值保护的掩码方案曾在 `TC_FL_094/102` 失败，已撤回，不应再次随意删除保护分支。
- 任务书 §3.1 允许单款自验证，本次 910B3 结果满足该项；§7 要求 A2/A3 均完成验收，A3 实机结果尚缺，不能宣称双产品验收完成。
- `task_submission` 的正式 XLSX、截图、内存报告仍待整理，并需落实 `Ascend-CANN` 的私仓开发者访问权限。设计评审、CI、正式验收及上游代码合入仍是后续事项；本次更新不表示这些流程已完成。

## 代码与材料索引

- 任务依据：社区发放的 2026 年 9 月 `aclblasStrmm_A2A3` 任务书；本 PR 不修改官方任务要求。
- 私有代码评审：[ops-blas PR #1](https://gitcode.com/m0_56676577/ops-blas/merge_requests/1)，以下链接需该私仓的访问权限。
- [Host 分派与参数校验](https://gitcode.com/m0_56676577/ops-blas/blob/549061889d7d0d8a6a584db44381d027540f7f3e/blas/trmm/arch22/strmm_host.cpp)。
- [Kernel 实现](https://gitcode.com/m0_56676577/ops-blas/blob/549061889d7d0d8a6a584db44381d027540f7f3e/blas/trmm/arch22/strmm_kernel.cpp)及同目录 `strmm_tiling_data.h`。
- [复现步骤](https://gitcode.com/m0_56676577/ops-blas/blob/549061889d7d0d8a6a584db44381d027540f7f3e/test/trmm/strmm/arch22/README.md)、[验证说明](https://gitcode.com/m0_56676577/ops-blas/blob/549061889d7d0d8a6a584db44381d027540f7f3e/test/trmm/strmm/arch22/VALIDATION.md)、[当前逐条性能](https://gitcode.com/m0_56676577/ops-blas/blob/549061889d7d0d8a6a584db44381d027540f7f3e/test/trmm/strmm/arch22/perf_results_910b3_accept20.csv)。

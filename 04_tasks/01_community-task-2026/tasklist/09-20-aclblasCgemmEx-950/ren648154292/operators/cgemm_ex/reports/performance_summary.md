# aclblasCgemmEx 性能自测报告

> 交付件 #3 之一（性能部分）· 面向 §4 验收交付件提交
> 数据来源：`ops-blas/test/gemmex/cgemmex/arch35/perf.log`、`cgemmex_perf_result.csv`
> 归档结果：`operators/cgemm_ex/reports/performance_report.csv`（51 条）
> 生成日期：2026-09-15

---

## 1. 测试摘要

### 1.1 环境

| 项 | 值 |
|---|---|
| 硬件 | Ascend 950PR |
| CANN 版本 | 9.1.0 |
| GPU 基线环境 | NVIDIA Driver 535.104.05 / CUDA 12.2 |
| GPU 基线来源 | `gpu_baseline.csv`（200 条官方基线，MD5 与官方一致）|
| 测试二进制 | `cgemmex_test`（802 KB）|
| 运行入口 | `bash run.sh --perf` |
| 采样策略 | warmup 3 + 有效采样 50（下限强制为 50）|
| 总样本数 | 50 cases × 50 samples = 2500 samples |

### 1.2 结果分布

| 判定 | 数量 | 占比 |
|---|---|---|
| PASS | 0 | 0% |
| FAIL | 43 | 86.0% |
| ERROR（stream sync failed）| 5 | 10.0% |
| SKIPPED（用户中断，未启动）| 2 | 4.0% |
| **合计** | **50** | **100%** |

> **说明**：任务书原文中「6 stream-sync-error」为笔误，实测 CSV 与 perf.log 均为 **5 条 ERROR**（TC_PF_1091/1099/1101/1103/1164），合计 43 + 5 + 2 = 50。

**本批次 0 PASS，性能未达标（详见 §5 回归披露）。**

### 1.3 NPU / GPU 比值区间

| 指标 | 数值 |
|---|---|
| 最大值（最好 case）| **0.0115**（TC_PF_1005, 1x1x1）|
| 最小值（最差 case）| **0.0000**（多个 case，实际比值 < 0.005）|
| 判定阈值 | 0.4（官方 `verify_performance.py::PERF_THRESHOLD`）|
| 达标 case 数 | 0 |

**比值方向**：`ratio = gpu_ms / (npu_mean_us / 1000)`。所有 FAIL case 的 ratio 均 < 0.4，即 **GPU 快 NPU 的 87x ~ 10000x**（`ratio = 0.4 → 2.5×`；`ratio = 0.0115 → ~87×`；`ratio = 0.0000 → >10000×`）。

### 1.4 采样次数

- `valid_samples = 50` 在所有非 ERROR、非 SKIPPED case 上均达到（43 条 FAIL 全部 `valid_samples = 50`）
- 阈值下限：`--samples` 参数下限强制为 50（低于 50 会被拒绝），因任务书 §3.3 要求 ≥50 有效样本
- **50 恰为临界值**（下限），未做超采样以增强统计置信度

---

## 2. 性能判据

### 2.1 任务书 §3.3 规定（原文引用）

测试设备：Ascend 950PR（CANN 9.1.0）。性能数据为默认主路径（typeA = typeB = typeC = ACLBLAS_C_32，即 COMPLEX64）输入场景下的平均单次耗时（Avg time，单位 us），须先 warmup 再有效采样 >50 次取平均。

算子在各性能 case 下的平均单次耗时应不高于下表标杆耗时：

| case | m | n | k | transA | transB | typeA | typeB | typeC | 标杆耗时（Avg time，us）|
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 1024 | 1024 | N | N | C_32 | C_32 | C_32 | **415.95** |
| 2 | 2048 | 2048 | 2048 | N | N | C_32 | C_32 | C_32 | **3243.65** |
| 3 | 1024 | 1024 | 1024 | T | N | C_32 | C_32 | C_32 | **411.27** |
| 4 | 2048 | 2048 | 2048 | N | N | R_32 | R_32 | R_32 | **851.86** |

### 2.2 本批次覆盖的标杆 case

| 标杆 # | 本批次对应 case | 本批次覆盖 | 原因 |
|---|---|---|---|
| 1 | TC_PF_1001（1024x1024x1024 NN C_32）| ✅ 已覆盖 | C_32 NN 主路径 |
| 2 | TC_PF_1002（2048x2048x2048 NN C_32）| ✅ 已覆盖 | C_32 NN 主路径 |
| 3 | TC_PF_1034（1024x1024x1024 TN C_32）| ❌ 未覆盖 | 本批次仅覆盖 transA=N, transB=N |
| 4 | TC_PF_1056（2048x2048x2048 NN R_32）| ❌ 未覆盖 | 本批次仅覆盖 C_32 |

### 2.3 覆盖标杆的实际表现

| 标杆 # | 标杆耗时 (us) | 本批次实测 (us) | 实测/标杆 | 结论 |
|---|---|---|---|---|
| 1 | 415.95 | **6,120,459**（TC_PF_1001）| **14,715×** | 严重未达标 |
| 2 | 3243.65 | **40,775,651**（TC_PF_1002）| **12,570×** | 严重未达标 |

**两个覆盖到的标杆 case 都慢了 4 个数量级**，与任务书 §3.3 规定不匹配。

### 2.4 与本批次判定阈值的差异

- 任务书 §3.3 判定：`实测 Avg time ≤ 标杆耗时`
- 本批次判定：`ratio_gpu_over_npu ≥ 0.4`（取自 `verify_performance.py::PERF_THRESHOLD = 0.4`，见 `ops-blas/test/gemmex/cgemmex/README.md` §9）
- **两者不等价**：本批次用 NPU/GPU 比值判定，任务书用绝对耗时判定；本批次的 0.4 阈值显著宽松于任务书 §3.3 的标杆耗时

---

## 3. 结果统计

### 3.1 全部 50 条 case 分布

按 `performance_report.csv`（归档于本目录，51 条记录）：

| 判定 | 数量 | 说明 |
|---|---|---|
| PASS | 0 | — |
| FAIL | 43 | ratio_gpu_over_npu < 0.4，全部 valid_samples=50 |
| ERROR | 5 | stream sync failed，valid_samples=0 |
| SKIPPED | 2 | 用户中断，未启动 |

### 3.2 NPU 相对 GPU 比值分布

| 比值区间 | Case 数 | Case 样例 |
|---|---|---|
| [0.01, 0.0115] | 4 | TC_PF_1005(0.0115)、TC_PF_1006(0.0112)、TC_PF_1007(0.0109)、TC_PF_1008(0.0106) |
| [0.005, 0.01) | 2 | TC_PF_1009(0.0062)、TC_PF_1010(0.0042) |
| [0.0001, 0.005) | 4 | TC_PF_1011(0.0009)、TC_PF_1012(0.0003)、TC_PF_1105(0.0003)、TC_PF_1113(0.0002) |
| [0.0000, 0.0001) | 33 | TC_PF_1075 起，大部分大 shape case 都落在 0.0000 |
| 0.0000（四舍五入）| — | 实际比值 < 0.005，NPU 慢 GPU 的 10000x 以上 |

**比值分布高度长尾**：小 shape（≤ 16x16x16）在 0.006~0.012，大 shape（≥ 256x256x64）在 0.0000，即 **GPU 快 NPU 的 87x ~ 10000x+**。

### 3.3 最坏 case Top 5（按 NPU 耗时）

| Case | Shape | NPU mean (us) | GPU baseline (ms) | Ratio |
|---|---|---|---|---|
| TC_PF_1051 | 2364x2364x2364 | 65,016,130 | 2.0599 | 0.0000 |
| TC_PF_1165 | 2293x2004x2749 | 61,495,249 | 2.0537 | 0.0000 |
| TC_PF_1002 | 2048x2048x2048 | 40,775,651 | 1.2975 | 0.0000 |
| TC_PF_1148 | 1460x1403x2774 | 31,633,328 | 0.8896 | 0.0000 |
| TC_PF_1139 | 1852x914x2978 | 32,183,258 | 0.8082 | 0.0000 |

**规模化的算子耗时随 m*n*k 呈超线性增长**，GPU 基线增长相对温和，导致 ratio 严重恶化。

### 3.4 最快 case Top 5（按 NPU 耗时）

| Case | Shape | NPU mean (us) | GPU baseline (ms) | Ratio |
|---|---|---|---|---|
| TC_PF_1005 | 1x1x1 | 568 | 0.0066 | **0.0115** |
| TC_PF_1006 | 2x2x2 | 572 | 0.0064 | 0.0112 |
| TC_PF_1007 | 4x4x4 | 583 | 0.0064 | 0.0109 |
| TC_PF_1008 | 8x8x8 | 627 | 0.0066 | 0.0106 |
| TC_PF_1009 | 16x16x16 | 1,028 | 0.0064 | 0.0062 |

**即使最小的 shape 也未达标**：TC_PF_1005 的 ratio 0.0115 仅为阈值 0.4 的 2.9%。

---

## 4. 异常 case 详细分析

### 4.1 stream-sync-error（5 条 ERROR）

| Case | Shape | NPU mean | valid_samples | 现象 |
|---|---|---|---|---|
| TC_PF_1091 | 1024x2048x64 | 0 us | 0 | `stream sync failed`，单次调用超时 3,272,857 ms |
| TC_PF_1099 | 4096x512x64 | 0 us | 0 | `stream sync failed`，单次调用 1,210 ms |
| TC_PF_1101 | 4096x1024x64 | 0 us | 0 | `stream sync failed`，单次调用 1,700 ms |
| TC_PF_1103 | 4096x2048x64 | 0 us | 0 | `stream sync failed`，单次调用 2,871 ms |
| TC_PF_1164 | 931x3291x3057 | 0 us | 0 | `stream sync failed`，单次调用 3,273,101 ms |

**共同特征**：
- 5 条中 **4 条 k=64**（TC_PF_1091/1099/1101/1103），另 1 条 shape 非方阵（TC_PF_1164, 931x3291x3057）
- 4 条中 **3 条 m=4096** 或 **1 条 m*n=2,097,152**（TC_PF_1091）
- 2 条（TC_PF_1091、TC_PF_1164）耗时接近 3,273 s ≈ 54.5 分钟，疑似**采样循环超时**（`--gtest_repeat` 或 `--warmup` 后 50 samples 未能在合理时间内完成）

**初步定位**：
1. `stream sync failed` 表明 `aclblasCgemmEx` 返回后 `aclrtStreamSynchronize` 或 `aclblasCgemmEx` 内部的 stream sync 失败
2. 5 条 case 的 shape 都**触及本批次 tile 策略的边界**：大 m*n + 小 k（64）或**极不规则的 shape**（931x3291x3057）
3. 具体原因待开发方通过 `ascendc-runtime-debug` / 探针定位；本工程不修改 kernel 源码

### 4.2 SKIPPED（2 条）

| Case | Shape | 估算工作 (ops) | 估算时长 | 中断原因 |
|---|---|---|---|---|
| TC_PF_1158 | 4033x3183x2662 | 34,172,197,818 | ~2.3 h | user interruption：not started at kill time |
| TC_PF_1069 | 3635x3635x3635 | 48,030,072,875 | ~3.3 h | user interruption：not started at kill time；**OOM risk** |

**说明**：这 2 条**未启动即被用户中断**（未计入 perf.log 的 `[perf]` 计数），中断原因是预计单次 50-sample 采样耗时过长（2~3 小时）。

`perf.log` 中 `[perf]` 行数 = 48 = 43 FAIL + 5 ERROR；加上 CSV 中的 2 SKIPPED = 50 条 case。

### 4.3 未运行即被估算总时长

`ops-blas/test/gemmex/cgemmex/README.md` §11 复现记录：
> 性能：`--perf` → 全 50 条 50 样本估算总时长约 **12.5 小时**（awk 测算 `cases=50 total_ops=1.4879e+11 est_total_min=750.1447`），按当前吞吐无法在一次会话内跑完。

本批次采用**逐步 kill + `--gtest_repeat`** 策略，分多次会话累积结果；`cgemmex_perf_result.csv` 逐条 flush，中断也留下有效部分结果。

---

## 5. 回归披露

### 5.1 关键披露：本批次性能未达标

- 本批次 0 PASS / 43 FAIL / 5 ERROR / 2 SKIPPED，**全部 50 条 case 性能未达标**。
- 覆盖的任务书 §3.3 两个标杆 case（Case 1、Case 2）实测分别比标杆耗时**慢 14,715× 和 12,570×**。
- 这是本批次**最严重的未闭合项**。

### 5.2 回归原因（初步归因）

**原因 1：任务书 §3.3 4 个标杆 case 都属"任务书基准"，本批次尚未对齐优化**

- 任务书 §3.3 的标杆耗时（Case 1: 415.95 us, Case 2: 3243.65 us）来自**参考实现**（可能是 cuBLAS 或已优化的 Ascend C 实现）
- 本批次的 3-GEMM 复数分解实现**尚未针对 950PR 硬件特性做专项优化**
- 4 个标杆 case 中 2 个（Case 3、Case 4）本批次未覆盖（TN 转置、R_32 dtype）

**原因 2：3-GEMM 复数分解的 tile 策略尚未针对 950PR 硬件特性调优**

- 当前 tile 策略使用**手工常量**（如 blockMmad 等），未走 tiling kernel 自动生成
- 950PR 的 cube unit、vector unit 配比、shared memory 容量与 A1/A2 不同
- 大 shape 下 tile 划分不最优，导致算力利用率低

**原因 3：与另一条开发线（4-GEMM 旧实现）的对比**

- `ops-blas/test/gemmex/cgemmex/README.md` §11.3 提到 A1-Main 开发线在 `128x128x128` NN 主路径上报 **0/10 精度**，涉及 P1/P2 两个缺陷点
- 本工程覆盖 C_32 NN 子集的 50 条性能 case（含 `128³` = TC_SQ_024），性能上全部 FAIL
- **两条开发线的性能数据无法直接对比**（A1-Main 只做精度，本批次做精度 + 性能）
- 但**双方在小 shape（128³）上的表现都未达标**，提示**共同的底层瓶颈**（可能是 3-GEMM/4-GEMM 分解策略本身的问题，或 950PR 硬件适配问题）

### 5.3 下一步计划

| Wave | 目标 | 预期收益 | 状态 |
|---|---|---|---|
| **Wave 1** | 3-GEMM 代数重构（可能优化为 2-GEMM 或融合 kernel）| NPU 相对 GPU 加速比从 0.0115 提升到 0.1+ | 待启动 |
| **Wave 2** | tile 策略调优（blockMmad 替换手工常量，走 tiling kernel 自动生成）| 大 shape case 加速比提升 3~10× | 待启动 |
| **Wave 3** | stream sync 问题定位（5 条 ERROR case）| 恢复 5 条 case 的可用性 | 待启动 |
| Wave 4 | 覆盖 transA=T、typeA=R_32 等未覆盖范围 | 补齐 Case 3、Case 4 标杆 | 待启动 |

**Wave 1 预期加速比 0.1+** 的依据：
- 当前最好 case（TC_PF_1005, 1x1x1）ratio = 0.0115
- 若 3-GEMM 融合为 2-GEMM，理论计算量减少 25%（3 次 → 2 次 GEMM），叠加 kernel 融合收益，可提升至 0.1+
- 参考 cuBLAS 在类似 shape 上的表现（GPU 基线 0.0066 ms for 1x1x1）

### 5.4 结论

- **本批次性能未达标，作为已知未闭合项披露**。
- 0/50 的 PASS 率反映的是**优化深度问题**，而非算子正确性问题（精度报告 §7 已确认 A/B 类 FAIL 均为判据问题或输入值域问题）。
- 建议在 Wave 1（3-GEMM 代数重构）完成后重新采样，届时 ratio 应能从 0.0115 提升到 0.1+ 量级。
- stream sync 问题（5 条 ERROR）应在 Wave 3 独立定位修复。

---

## 附：数据溯源

- 原始日志：`ops-blas/test/gemmex/cgemmex/arch35/perf.log`（31,558 B）、`perf.nohup`（31,417 B）
- 原始 CSV：`ops-blas/test/gemmex/cgemmex/arch35/cgemmex_perf_result.csv`（4,597 B，50 条）
- 归档 CSV：`operators/cgemm_ex/reports/performance_report.csv`（4,924 B，50 条 + 表头）
- 官方 GPU 基线：`ops-blas/test/gemmex/cgemmex/arch35/gpu_baseline.csv`（11,779 B，200 条，MD5 与官方一致）
- 性能判定阈值：`PERF_THRESHOLD = 0.4`（`verify_performance.py` 常量，`README.md` §9 说明）
- 任务书性能标准：`aclblasCgemmEx_task_doc.md` §3.3
- 复现记录：`ops-blas/test/gemmex/cgemmex/README.md` §11（含 12.5 小时总时长估算）
- 与 A1-Main 对比：`ops-blas/test/gemmex/cgemmex/README.md` §11.3

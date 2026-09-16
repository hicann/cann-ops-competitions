# aclblasCgemmEx §4 交付件报告索引

> 交付件 #3 目录索引 · 归档位置：`operators/cgemm_ex/reports/`
> 生成日期：2026-09-15

---

## 1. 交付件清单

| # | 文件 | 用途 | 数据来源 |
|---|---|---|---|
| 1 | `accuracy_report.md` | 精度自测报告（52 PASS / 14 FAIL，含 FAIL 归因与未覆盖范围）| `ops-blas/test/gemmex/cgemmex/arch35/accuracy.log` |
| 2 | `memory_report.md` | 内存占用自测报告（4 代表 case，3 measured + 1 formula）| `ops-blas/test/gemmex/cgemmex/arch35/memory_samples.csv` |
| 3 | `performance_summary.md` | 性能自测报告（0 PASS / 43 FAIL / 5 ERROR / 2 SKIPPED，含性能回归披露）| `ops-blas/test/gemmex/cgemmex/arch35/perf.log`、`cgemmex_perf_result.csv` |
| — | `performance_report.csv` | 性能数据原始归档（51 行 = 表头 + 50 case）| `cgemmex_perf_result.csv` 逐条对齐 |
| — | `REPORT_INDEX.md` | 本索引文件 | — |

---

## 2. 数据血缘映射

### 2.1 精度链路

```
任务书 §3.2 (aclblasCgemmEx_task_doc.md)
    ↓ 判定阈值
官方 verify.h::MereMareStrategy (ops-blas/test/frame/verify.h, 只读参考)
    ↓ mere_threshold = 1.221e-04, mare_multiplier = 10.0
cgemmex_precision.h (独立实现, 不 include 官方)
    ↓
cgemmex_test 二进制 (bash run.sh --accuracy)
    ↓ 66 cases × (实部 + 虚部)
arch35/accuracy.log (32,632 B)
    ↓
operators/cgemm_ex/reports/accuracy_report.md
```

### 2.2 内存链路

```
任务书 §3.4 (不涉及) + §4 表 #3 (内存数据要求)
    ↓
npu-smi info (HBM-Usage 列, 0.67s 单次延迟)
    ↓
sample_memory.sh (--gtest_repeat 拉长窗口, kill -STOP/-CONT 保护)
    ↓ 4 代表 case
arch35/memory_samples.csv (510 B, 3 measured + 1 formula)
    ↓
operators/cgemm_ex/reports/memory_report.md
```

### 2.3 性能链路

```
任务书 §3.3 (4 标杆 case)
    ↓ 覆盖 2/4
官方 verify_performance.py::PERF_THRESHOLD = 0.4
    ↓ ratio_gpu_over_npu 判定
cgemmex_test --perf (50 cases × 50 samples)
    ↓
arch35/perf.log (31,558 B, 48 条 [perf] 行)
arch35/cgemmex_perf_result.csv (4,597 B, 48 条)
    ↓ 补齐 2 SKIPPED
operators/cgemm_ex/reports/performance_report.csv (51 行归档)
    ↓
operators/cgemm_ex/reports/performance_summary.md
```

### 2.4 GPU 基线链路

```
官方 gpu_baseline.csv (200 条, MD5 与官方一致)
    ↓ 匹配 (m, n, k, transA, transB, type)
ops-blas/test/gemmex/cgemmex/arch35/gpu_baseline.csv (11,779 B)
    ↓
performance_report.csv::gpu_baseline_ms 列
```

---

## 3. 测试工程交叉引用

| 项 | 路径 |
|---|---|
| 测试用例族（TC_SQ/TC_AB/TC_LD/TC_FL/TC_CV/TC_ED/TC_PF）| `ops-blas/test/gemmex/cgemmex/arch35/` |
| 测试二进制 | `ops-blas/test/gemmex/cgemmex/arch35/cgemmex_test` (802 KB) |
| 运行入口 | `ops-blas/test/gemmex/cgemmex/arch35/run.sh` |
| 项目 README（含 §7 精度判据、§11 归因）| `ops-blas/test/gemmex/cgemmex/README.md` |
| 官方测试框架 | `ops-blas/test/frame/verify.h`、`verify_performance.py` |
| 任务书 | `aclblasCgemmEx_task_doc.md` |
| 本批次覆盖范围收窄记录 | `project-cgemm-ex-task.md` (memory) |

---

## 4. 关键数据校验

### 4.1 精度数据

| 校验项 | 值 | 来源 |
|---|---|---|
| 总用例 | 66 | `accuracy.log` 尾部 `[  PASSED  ] 52 tests` |
| PASS | 52 | `accuracy.log` 尾部 |
| FAIL | 14 | `accuracy.log` 尾部 `[  FAILED  ] 14 tests` |
| 通过率 | 78.8% | 52/66 |
| 有效数值覆盖 | 60（46 PASS + 14 FAIL）| 剔除 6 状态码 PASS + 6 退化 PASS |
| 判定阈值 | 1.221e-04 / 1.221e-03 | 官方 CSV 逐行读取 |
| 退化 PASS | 12 | 见 `accuracy_report.md` §3.2 |
| FAIL 归因 | 12 A类 + 2 B类 + 0 C类 | 见 `accuracy_report.md` §5 |

### 4.2 内存数据

| 校验项 | 值 | 来源 |
|---|---|---|
| 采样 case 数 | 4 | `memory_samples.csv` |
| 实测 case 数 | 3（TC_SQ_017/020/030）| 同上 |
| 公式估算 case 数 | 1（TC_SQ_031）| 耗时约束，未实测 |
| NPU 常驻 runtime 地板 | ~270 MB | delta_peak 最小值 |
| workspace 上限公式对齐 | 理论最小 workspace | `ws_formula_mb` 列 |
| 公式 vs 实测偏差 | 2~4 个数量级 | 见 `memory_report.md` §4.1 |

### 4.3 性能数据

| 校验项 | 值 | 来源 |
|---|---|---|
| 总 case 数 | 50 | `performance_report.csv`（含 2 SKIPPED）|
| PASS | 0 | `performance_report.csv::verdict` |
| FAIL | 43 | 同上 |
| ERROR（stream sync failed）| **5**（TC_PF_1091/1099/1101/1103/1164）| 同上 |
| SKIPPED | 2（TC_PF_1158/1069）| 同上 |
| 有效样本数 | 50（临界下限）| `--samples` 参数下限 |
| 最好 ratio | 0.0115（TC_PF_1005, 1x1x1）| `performance_report.csv::ratio_gpu_over_npu` |
| 阈值 | 0.4（`PERF_THRESHOLD`）| `verify_performance.py` |
| 达标 case 数 | 0 | — |
| 任务书 §3.3 覆盖 | 2/4（Case 1、Case 2）| `performance_summary.md` §2.2 |
| 覆盖标杆倍差 | 14,715× / 12,570× | 见 `performance_summary.md` §2.3 |
| 全量跑批估算 | ~12.5 h | `README.md` §11 |

### 4.4 与任务书要求的对齐度

| 任务书 §4 要求 | 本批次状态 | 说明 |
|---|---|---|
| #1 算子精度数据 | ✅ 已提交 | 78.8% 通过率，14 FAIL 均归因 |
| #2 算子性能数据 | ⚠️ 已提交，未达标 | 0 PASS，性能回归已披露 |
| #3 算子内存占用数据 | ✅ 已提交 | 4 case，含 270 MB 地板观察 |
| #4 单元测试覆盖率 | ⚠️ 部分 | 仅覆盖 C_32 NN 主路径 + 状态码；未覆盖 TC_EX、TC_RC、TC_PF 负向指针等 |
| #5 静态检查 / 代码规范 | — | 未纳入本批次交付 |
| #6 三方依赖合规 | — | 未纳入本批次交付 |

---

## 5. 已知偏差披露

### 5.1 用户任务描述与实测数据的偏差

| 项 | 用户描述 | 实测 | 说明 |
|---|---|---|---|
| stream-sync-error 数 | 6 | **5** | 任务描述为笔误；实测 CSV 与 perf.log 均为 5 条（TC_PF_1091/1099/1101/1103/1164）|
| FAIL 总数 | 未明确 | **14**（非 15）| README §11.1 记 15（含 TC_ED_707）；本次运行 TC_ED_707 已 PASS，实际 FAIL = 14 |

### 5.2 与任务书要求的偏差

| 项 | 任务书要求 | 本批次实际 | 原因 |
|---|---|---|---|
| 精度判据 | rtol/atol/matched_ratio | mere/mare（官方 verify.h 实现）| 官方仓实际实现与任务书规定不一致；本工程取官方实现为准（README §12）|
| 内存要求 | §3.4 不涉及 | 已提交内存数据 | §4 表 #3 要求内存占用数据，故交付 |
| 性能标杆 | 4 case（Case 1-4）| 覆盖 2 case（Case 1、Case 2）| 本批次仅覆盖 C_32 + NN，未覆盖 TN 与 R_32 |
| 性能达标 | 未达标 | 未达标 | 见 `performance_summary.md` §5 |
| 采样次数 | ≥50 | 50（临界）| `--samples` 参数下限强制为 50 |
| 全量 case 覆盖 | 覆盖 TC_EX/TC_RC/负向指针 | 仅 C_32 NN 主路径 + 状态码 | 本批次范围收窄（memory: project-cgemm-ex-task.md）|

---

## 6. 归档完整性

| 项 | 状态 |
|---|---|
| accuracy_report.md 存在 | ✅ |
| memory_report.md 存在 | ✅ |
| performance_summary.md 存在 | ✅ |
| performance_report.csv 存在（未修改）| ✅ |
| REPORT_INDEX.md 存在（本文件）| ✅ |
| 数据血缘可追溯 | ✅ 见 §2 |
| 关键数据交叉引用一致 | ✅ 见 §4 |
| 已知偏差已披露 | ✅ 见 §5 |
| 未修改 ops-blas 内任何文件 | ✅（全程只读）|
| 未修改 performance_report.csv | ✅（全程只读）|

---

## 附：交付件目录

```
operators/cgemm_ex/reports/
├── REPORT_INDEX.md         (本文件，索引与血缘)
├── accuracy_report.md      (精度自测报告)
├── memory_report.md        (内存占用自测报告)
├── performance_summary.md  (性能自测报告 + 回归披露)
└── performance_report.csv  (性能数据归档，未修改)
```

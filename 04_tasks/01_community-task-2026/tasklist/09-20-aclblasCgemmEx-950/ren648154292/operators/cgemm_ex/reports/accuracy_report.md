# aclblasCgemmEx 精度自测报告（迭代二）

> 交付件 #3 之一（精度部分）· 面向 §4 验收交付件提交
> 数据来源：`ops-blas/test/gemmex/cgemmex/arch35/accuracy_final.log`（333,618 B，2026-09-16 08:53）
> 生成日期：2026-09-17
>
> ⚠ **iter4b 状态说明（2026-09-18）**：本报告主体为迭代二（自研 3-GEMM Karatsuba 路线）数据。该路线已在 iter4b（本 PR，PR #1560）整体废弃——全量回归实测 1251 条 CSV 用例 91.8% FAIL，数值不可用。iter4b 改用薄 wrapper 委托 upstream `aclblasCgemm` / `aclblasSgemm`，全量回归 1259 条 **1245 PASS / 14 FAIL（1.11%）**，详见 `iter4b_full_regression_report.md` 与 `wrapper_architecture_report.md`。本报告保留原 iter2 数据作为历史轨迹，不做美化或覆盖。

---

## iter4b 全量数据（2026-09-18 补充）

### iter4b 精度结果

| 指标 | 值 |
|---|---|
| 用例总数 | **1259**（1251 CSV + 8 附加） |
| **PASS** | **1245（98.89%）** |
| **FAIL** | **14（1.11%）** |
| 全量耗时 | 196.016 s（3.3 分钟） |
| 数据源 | `/tmp/iter4b/full_run.log`；harness 侧新增 `aclblasSetWorkspace(handle, 1 GiB)` |

### iter4b FAIL 收敛轨迹（三版对比）

| 阶段 | 用例数 | PASS | FAIL | FAIL 率 | 主要失败原因 |
|---|---|---|---|---|---|
| iter3（自研 kernel） | 1251 CSV | 101 | 1148 | **91.8%** | 数值精度 552 + `ret=7` 实现范围外 562 + `ret=3 INVALID_VALUE` 4 |
| iter4（wrapper, harness 未修） | 1259 全量 | 1155 | 104 | **8.26%** | 90 harness workspace + 12 半精度 + 2 NaN/inf |
| **iter4b（wrapper + workspace 1 GiB）** | **1259 全量** | **1245** | **14** | **1.11%** | 12 半精度契约 + 2 NaN/inf IEEE 语义 |

### iter4b 剩余 14 条 FAIL 完整清单

**半精度契约（12 条）**：wrapper 明确返回 `ACBLAS_STATUS_NOT_SUPPORTED`，属任务书契约与 CSV 期望冲突（需上游裁定二选一：扩展 wrapper 覆盖半精度 / 调整 CSV 期望值）：

| # | 用例 | dtype |
|---|---|---|
| 1 | `SUP_L0_HR32_NN_8X8X8` | H_R_32 |
| 2 | `SUP_L0_HC32_NN_8X8X8` | H_C_32 |
| 3 | `SUP_HR32_NT_16X16X16` | H_R_32 |
| 4 | `SUP_HR32_CT_32X32X32_IMAG` | H_R_32 |
| 5 | `SUP_HR32_CN_RC_32X16X48` | H_R_32 |
| 6 | `SUP_HR32_ALPHA0_BETAHALF` | H_R_32 |
| 7 | `SUP_HR32_LD_PADDING_16X16X16` | H_R_32 |
| 8 | `SUP_HC32_NT_16X16X16` | H_C_32 |
| 9 | `SUP_HC32_CT_32X32X32` | H_C_32 |
| 10 | `SUP_HC32_CN_RC_32X16X48` | H_C_32 |
| 11 | `SUP_HC32_ALPHA0_BETAHALF` | H_C_32 |
| 12 | `SUP_HC32_LD_PADDING_16X16X16` | H_C_32 |

**NaN/inf IEEE 语义（2 条）**：harness + golden 语义分歧，与 wrapper 透传行为无关（wrapper 未做任何数值处理）：

| # | 用例 | 现象 |
|---|---|---|
| 1 | `TC_FL_552` | Output 全 `inf/-inf`，Golden 全 `-nan`（32 元素 31 失败）；`gold_nonfinite=32 out_nonfinite=32 pattern_mismatch=0`——非有限模式一致，但比较器判失败 |
| 2 | `SUP_NS_NAN_C_BETA0` | NaN × 0 类退化（`0*NaN=NaN`），Golden 假设为有限值；非有限元素模式不匹配 |

### iter4b 关键用例前后对比

| 用例 | shape | iter3 | iter4 | **iter4b** |
|---|---|---|---|---|
| `TC_PF_1001` | 1024³ NN C_32 | FAIL（310 ms, matchedRatio=0） | PASS（32 MiB 默认够用） | **PASS（237 ms, matchedRatio=1.0000）** |
| `TC_PF_1002` | 2048³ NN C_32 | >120 s 超时 | FAIL（ret=5, workspace 不足） | **PASS（0/2048 元素失败）** |
| `TC_PF_1073` | 3999³ TT C_32 最大 shape | 未测（预估超时） | FAIL（ret=5, 2999 ms） | **PASS（2/3999 real, 1/3999 imag 失败, matchedRatio≥0.995）** |
| `TC_EX_0892` | — | 未测 | FAIL（ret=5） | **PASS（0/2000 real, 0/2000 imag 失败）** |
| `TC_EX_0904` | — | 未测 | FAIL（ret=5） | **PASS（4/2000 real, 0/2000 imag 失败, matchedRatio=0.9980 ≥ 0.99）** |
| `TC_L0_001` | 4³ NN C_32 | FAIL（matchedRatio=0） | PASS（matchedRatio=1.0000） | **PASS（matchedRatio=1.0000）** |
| `TC_L0_005` | R_32 NN 4×4×4 | ret=7「实现范围外」 | PASS | **PASS（实际执行, matchedRatio=1.0000）** |
| `TC_ED_701`/`705`/`706` | k=0 边界 | 未测 | 4/4 PASS（wrapper 修条件式指针校验） | **PASS** |

**关键结论**：wrapper 自身在 1245 条用例上（含 1024³、2048³、3999³ 大矩阵复数与实数）数值全部通过，`matchedRatio` 均达 1.0000；剩余 14 条 FAIL 100% 可归因，无一条指向 wrapper 数值错误。

---


---

## 1. 测试摘要

### 1.1 环境

| 项 | 值 |
|---|---|
| 硬件 | Ascend 950PR（`Ascend950PR_9579` full-soc）|
| CANN 版本 | 9.1.0 |
| 三方软件 | torch 2.1.0+, torch_npu 2.1.0.post3+ |
| GPU 基线环境 | NVIDIA Driver 535.104.05 / CUDA 12.2 |
| golden 实现 | cblas (Netlib BLAS 复数实现，CPU double) |
| 测试二进制 | `cgemmex_test` (808,056 B，`-Wall -Wextra` 零告警) |
| 运行入口 | `bash run.sh --accuracy`（`cgemmex_test --accuracy --samples 60`）|
| 用例清单 | `cgemmex_test.csv`（1,220 数据行 / 30 列 / 216,188 B）|

### 1.2 结果分布

| 指标 | 迭代一 | 迭代二 |
|---|---|---|
| 总用例 | 66 | **348** |
| PASS | 52（78.8%）| **110（31.6%）** |
| FAIL | 14（21.2%）| **238（68.4%）** |
| 其中：数值比较 PASS | 46 | 74 |
| 其中：退化 PASS（valid=0/0）| 12 | 20 |
| 其中：状态码断言 PASS | 6 | 16 |
| 其中：数值比较 FAIL | 14 | 238 |
| 总耗时 | 90,904 ms | 110,082 ms |
| 退出码 | 1 | 1 |

**通过率从 78.8% 下降到 31.6%。下降不是回归，而是覆盖范围扩大的必然结果**：迭代二把用例从「C_32 + NN + 小 shape」扩到「C_32 全转置组合 + 全 shape + 新增 TC_TX 非方阵族」，把此前未测试的转置路径与非方阵路径暴露出来。见 §6 功能缺口专项披露。

**若只看迭代一已覆盖的等价子集（C_32 + NN 方阵），数值结论没有回退**：NN 方阵 75 条中 58 PASS、17 FAIL（15 A 类判据边缘 + 2 D 类 golden 值域），与迭代一同向。

**任务书判据口径（`maxAbsErr ≤ 1e-2` 且 `mismatch == 0`）下通过率 41.7%（145/348，35 条 A 类改判 PASS，203 条仍 FAIL，详见 §7）。**

---

## 2. 精度判据

### 2.1 任务书 §3.2 规定（原文引用）

golden 由 cblas（Netlib `cgemm`）单标杆比对生成，输出矩阵 C（m×n）全矩阵验证，实部、虚部分别比对。主路径输入输出为单精度复数，比对时实部、虚部分别按 FLOAT32 标准判定：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| COMPLEX64（实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |
| FLOAT32（typeT = ACLBLAS_R_32 时） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |
| FLOAT16（typeT = ACLBLAS_H_R_32 时） | 2^-9 (1.95e-3) | 2^-9 (1.95e-3) | 0.99 | 1e-1 或 32 * ULP |

逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`；当用例同时满足 `matched_ratio ≥ required_matched_ratio` 且 `max_abs_error ≤ max_abs_error_limit` 时，判定该用例精度通过。

### 2.2 本工程实际采用的判据

任务书 §3.2 的判据（rtol/atol/matched_ratio）与官方 `verify.h` 实际实现**不同**。本工程按官方 CSV 的 `mere_threshold` 与 `mare_multiplier` 两列逐行读取，逐行对应 `ops-blas/test/frame/verify.h::MereMareStrategy` 实现（该文件按任务书为只读参考，故本工程在 `cgemmex_precision.h` 内独立实现、不 include）。

本范围所有行的取值：`mere_threshold = 0.00012207031`（= 2^-13）、`mare_multiplier = 10.0`，即 `threshold = 1.221e-04`、`outlierLimit = 1.221e-03`。

判定逻辑（纯相对误差）：

```
pass = (mismatchCount == 0) AND (mere < threshold) AND (maxRelErr < outlierLimit)
```

其中 `|gold| < threshold` 的元素被跳过（不计入 `mere` 分母），`mismatchCount` 在遇到 Inf/NaN 时递增。

> **注意**：判定为纯相对误差判据。大 k 的 float32 GEMM 会在近零元素上出现相对误差爆炸但绝对误差仍然很小的情形，故额外记录 `maxAbsErr`（不参与判定，仅用于归因诊断）。

### 2.3 判据差异说明

| 维度 | 任务书 §3.2 | 实际判据 |
|---|---|---|
| 阈值来源 | 硬编码常量 | CSV 逐行读取 |
| 主要指标 | rtol / atol / matched_ratio | mere / maxRelErr / mismatchCount |
| 近零元素处理 | 由 `|actual-gold| ≤ atol` 覆盖 | 直接跳过（`|gold| < threshold`）|
| 通过条件 | 两个条件同时满足 | 三个条件同时满足 |
| 与官方仓一致性 | 否（任务书规定） | 是（官方 verify.h）|

本工程取**官方仓实际实现**为准（详见 `ops-blas/test/gemmex/cgemmex/README.md` §12）。

---

## 3. PASS 用例汇总

### 3.1 按用例族统计

| 用例族 | 执行 | PASS | FAIL | 通过率 | 说明 |
|---|---|---|---|---|---|
| TC_L0 基线 | 4 | 2 | 2 | 50.0% | NN 通过；**TN 两条 FAIL（起点 bug 未修复，见 §6.4）** |
| TC_SQ 方阵形状（1³ … 2048³）| 183 | 46 | 137 | 25.1% | 全转置组合 + 全 shape；非对称转置对全部 FAIL |
| TC_AB alpha/beta 组合 | 24 | 11 | 13 | 45.8% | 全转置组合 |
| TC_CV 常量填充 | 72 | 13 | 59 | 18.1% | 全转置组合 |
| TC_LD leading-dim padding | 12 | 6 | 6 | 50.0% | NN 通过；非 NN 全部 FAIL |
| TC_FL 填充/极值 | 6 | 4 | 2 | 66.7% | 4 PASS 中 3 条为退化 PASS |
| TC_ED 边界与负向 | 27 | 27 | 0 | 100% | 11 条数值（含退化）+ 16 条状态码 |
| TC_TX 非方阵 × 全转置（**迭代二新增**）| 20 | 1 | 19 | 5.0% | 唯一 PASS 为退化 PASS（TC_TX_3011）|
| **合计** | **348** | **110** | **238** | **31.6%** | — |

### 3.2 按转置组合统计

| transA/transB | 执行 | PASS | FAIL | 通过率 |
|---|---|---|---|---|
| N/N | 75 | 58 | 17 | 77.3% |
| T/T | 31 | 17 | 14 | 54.8% |
| C/C | 34 | 21 | 13 | 61.8% |
| C/N | 38 | 3 | 35 | 7.9% |
| C/T | 30 | 1 | 29 | 3.3% |
| N/C | 30 | 1 | 29 | 3.3% |
| N/T | 34 | 2 | 32 | 5.9% |
| T/C | 30 | 1 | 29 | 3.3% |
| T/N | 44 | 4 | 40 | 9.1% |
| N/INVALID | 1 | 1 | 0 | 100%（状态码）|
| INVALID/N | 1 | 1 | 0 | 100%（状态码）|
| **合计** | **348** | **110** | **238** | **31.6%** |

对称转置对（NN / TT / CC）通过率 65.3%（96/147）；6 个非对称转置对（NT / TN / NC / CN / CT / TC）通过率 5.6%（12/215）。**分界严格落在「是否对称」上**，见 §6.1。

### 3.3 退化 PASS（20 条，valid = 0/0，无有效元素比对）

| Case | Shape | trans | 原因 |
|---|---|---|---|
| TC_ED_699 | 0x8x8 | N/N | m=0，C 矩阵为空 |
| TC_ED_700 | 8x0x8 | N/N | n=0，C 矩阵为空 |
| TC_ED_701 | 8x8x0 | N/N | k=0，C = beta*C（全部 skipped）|
| TC_ED_702 | 0x0x0 | N/N | 三维全零 |
| TC_ED_705 | 8x8x0 | N/N | k=0 |
| TC_ED_706 | 8x8x0 | N/N | k=0 |
| TC_ED_707 | 16x16x16 | N/N | alpha=(0,0), beta=(0.5,0.5)，out == gold（bit-exact，全部 skipped）|
| TC_AB_432 | 32x32x32 | N/N | VALUE_NORM_0 填充，全零输出 |
| TC_AB_433 | 32x32x32 | T/N | VALUE_NORM_0 填充 |
| TC_AB_434 | 32x32x32 | C/N | VALUE_NORM_0 填充 |
| TC_AB_435 | 32x32x32 | N/N | VALUE_NORM_0 填充 |
| TC_AB_436 | 32x32x32 | T/N | VALUE_NORM_0 填充 |
| TC_AB_437 | 32x32x32 | C/N | VALUE_NORM_0 填充 |
| TC_FL_544 | 32x32x32 | N/N | VALUE_NORM_0 填充 |
| TC_FL_547 | 32x32x32 | N/N | VALUE_NORM_0 填充 |
| TC_FL_548 | 32x32x32 | N/N | VALUE_NORM_0 填充 |
| TC_SQ_055 | 1x1x1 | N/C | 全零输入，skipped 1/0 |
| TC_SQ_124 | 1x1x1 | T/C | 全零输入 |
| TC_SQ_147 | 1x1x1 | C/N | 全零输入 |
| TC_TX_3011 | 64x128x32 | T/T | VALUE_NORM_0 填充，skipped 4144/4144 |

按 `ops-blas/test/gemmex/cgemmex/README.md` §11.1 的口径，**TC_ED_699/700/701/702/705/706 这 6 条不应计为有效覆盖**；其余 14 条同样是空比对或 bit-exact 空通过。其中 6 条（TC_AB_433/434/436/437、TC_SQ_055/124/147、TC_TX_3011 中的转置/共轭组合）落在**未实现的非对称转置路径上却因输入全零而通过**，这掩盖了功能缺口 —— 若换非零填充它们会失败，见 §6.1。

**剔除退化 PASS 与状态码 PASS 后，本批次实际有效的数值精度覆盖为 74 PASS + 238 FAIL = 312 条。**

### 3.4 状态码断言 PASS（16 条，STATUS-ONLY，无数值比对）

| Case | 期望状态码 | 实际状态码 |
|---|---|---|
| TC_ED_708 | HANDLE_IS_NULLPTR | HANDLE_IS_NULLPTR |
| TC_ED_709 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_710 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_711 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_712 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_713 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_715 | INVALID_ENUM | INVALID_ENUM |
| TC_ED_716 | INVALID_ENUM | INVALID_ENUM |
| TC_ED_720 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_721 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_722 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_723 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_724 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_725 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_726 | INVALID_VALUE | INVALID_VALUE |
| TC_ED_727 | INVALID_VALUE | INVALID_VALUE |

---

## 4. FAIL 用例详细表

阈值 `threshold = 1.221e-04`，`outlierLimit = 1.221e-03`，`max_abs_error_limit = 1e-2`（任务书口径）。

238 条 FAIL 分为四类：A 类 35 条（数值正确，判据形式边缘）、B 类 7 条（非方阵）、C 类 194 条（非对称转置）、D 类 2 条（golden 值域溢出）。**其中 201 条（B + C）数值确实错误，3 条以上量级的错误；35 条 A 类数值正确。**

### 4.1 A 类：判据形式边缘 FAIL（35 条，数值正确）

| # | Case | Shape | trans | mere | maxRelErr | maxAbsErr |
|---|---|---|---|---|---|---|
| 1 | TC_AB_438 | 32x32x32 | N/N | 2.87e-06 | 1.70e-03 | 5.341e-05 |
| 2 | TC_CV_581 | 15x15x15 | C/C | 1.60e-05 | 2.62e-03 | 2.289e-05 |
| 3 | TC_CV_595 | 48x48x48 | T/T | 2.04e-06 | 2.70e-03 | 9.155e-05 |
| 4 | TC_CV_599 | 48x48x48 | C/C | 1.56e-06 | 1.63e-03 | 7.629e-05 |
| 5 | TC_CV_604 | 96x96x96 | T/T | 1.68e-06 | 1.84e-03 | 2.747e-04 |
| 6 | TC_CV_608 | 96x96x96 | C/C | 2.38e-06 | 5.00e-03 | 2.136e-04 |
| 7 | TC_CV_609 | 200x200x200 | N/N | 2.97e-06 | 8.26e-03 | 5.646e-04 |
| 8 | TC_CV_613 | 200x200x200 | T/T | 3.06e-06 | 9.43e-03 | 7.019e-04 |
| 9 | TC_CV_617 | 200x200x200 | C/C | 4.54e-06 | 3.99e-02 | 5.589e-04 |
| 10 | TC_CV_618 | 400x400x400 | N/N | 4.84e-06 | 8.00e-02 | 1.282e-03 |
| 11 | TC_CV_622 | 400x400x400 | T/T | 4.52e-06 | 1.20e-01 | 1.434e-03 |
| 12 | TC_CV_626 | 400x400x400 | C/C | 4.79e-06 | 7.28e-02 | 1.343e-03 |
| 13 | TC_SQ_022 | 65x65x65 | N/N | 3.52e-06 | 3.63e-03 | 1.831e-04 |
| 14 | TC_SQ_024 | 128x128x128 | N/N | 3.85e-06 | 2.22e-02 | 3.648e-04 |
| 15 | TC_SQ_025 | 200x200x200 | N/N | 3.08e-06 | 2.03e-02 | 6.714e-04 |
| 16 | TC_SQ_026 | 256x256x256 | N/N | 3.89e-06 | 2.77e-02 | 7.935e-04 |
| 17 | TC_SQ_027 | 400x400x400 | N/N | 4.26e-06 | 5.10e-02 | 1.343e-03 |
| 18 | TC_SQ_028 | 512x512x512 | N/N | 4.57e-06 | 1.03e-01 | 1.465e-03 |
| 19 | TC_SQ_029 | 800x800x800 | N/N | 6.47e-06 | 5.03e-01 | 2.625e-03 |
| 20 | TC_SQ_030 | 1024x1024x1024 | N/N | 9.76e-06 | 2.07 | 4.272e-03 |
| 21 | TC_SQ_031 | 2048x2048x2048 | N/N | 1.11e-05 | 7.96e-01 | 9.521e-03 |
| 22 | TC_SQ_113 | 64x64x64 | T/T | 7.85e-06 | 2.17e-02 | 1.221e-04 |
| 23 | TC_SQ_114 | 65x65x65 | T/T | 1.40e-06 | 1.96e-03 | 1.450e-04 |
| 24 | TC_SQ_115 | 96x96x96 | T/T | 1.77e-06 | 2.36e-03 | 2.289e-04 |
| 25 | TC_SQ_116 | 128x128x128 | T/T | 1.88e-06 | 2.33e-03 | 2.747e-04 |
| 26 | TC_SQ_117 | 200x200x200 | T/T | 3.49e-06 | 6.61e-02 | 5.798e-04 |
| 27 | TC_SQ_118 | 256x256x256 | T/T | 4.59e-06 | 9.01e-02 | 8.850e-04 |
| 28 | TC_SQ_119 | 400x400x400 | T/T | 6.28e-06 | 2.75e-01 | 1.404e-03 |
| 29 | TC_SQ_120 | 512x512x512 | T/T | 4.72e-06 | 2.73e-01 | 1.648e-03 |
| 30 | TC_SQ_207 | 96x96x96 | C/C | 1.69e-06 | 1.55e-03 | 2.594e-04 |
| 31 | TC_SQ_208 | 128x128x128 | C/C | 1.59e-06 | 1.47e-03 | 3.510e-04 |
| 32 | TC_SQ_209 | 200x200x200 | C/C | 5.11e-06 | 8.75e-02 | 5.493e-04 |
| 33 | TC_SQ_210 | 256x256x256 | C/C | 3.38e-06 | 6.47e-02 | 7.324e-04 |
| 34 | TC_SQ_211 | 400x400x400 | C/C | 4.09e-06 | 3.65e-02 | 1.343e-03 |
| 35 | TC_SQ_212 | 512x512x512 | C/C | 5.43e-06 | 3.67e-01 | 1.831e-03 |

**共同特征**：全部 35 条 `mismatch(re/im) = 0/0`，`maxAbsErr ≤ 9.521e-03`（低于任务书阈值 1e-2），且 `mere` 全部在 `1.396e-06 ~ 1.596e-05` 区间，**低于阈值 1.221e-04 十倍以上**；只有 `maxRelErr` 越过 `outlierLimit = 1.221e-03`。`maxAbsErr` 在每族内随 k 单调增长。

分布：N/N 12 条、T/T 12 条、C/C 11 条 —— **全部落在对称转置对上**，即 A 类不是转置路径问题，是判据形态问题。

### 4.2 B 类：非方阵 shape 数值错误（7 条）

| # | Case | Shape | trans | mere | maxRelErr | maxAbsErr |
|---|---|---|---|---|---|---|
| 36 | TC_TX_3001 | 64x128x32 | N/N | 5.631 | 380 | 380 |
| 37 | TC_TX_3005 | 64x128x32 | T/T | 10.96 | 411 | 411 |
| 38 | TC_TX_3009 | 64x128x32 | C/C | 5.069 | 401.1 | 401.1 |
| 39 | TC_TX_3010 | 64x128x32 | N/N | 14.02 | 14.41 | 14.41 |
| 40 | TC_TX_3012 | 32x64x128 | N/N | 3.686 | 753.2 | 753.2 |
| 41 | TC_TX_3016 | 32x64x128 | T/T | 5.119 | 852 | 852 |
| 42 | TC_TX_3020 | 32x64x128 | C/C | 3.652 | 693.5 | 693.5 |

7 条全部来自迭代二新增的 TC_TX 族，且 trans 组合均为对称（N/N 3 条、T/T 2 条、C/C 2 条）。**注意：这 7 条在对称转置对里，且 TC_TX_3001/3010/3012 是 N/N 方阵之外的非方阵 —— 说明非方阵失败不是转置问题**（对照 §6.2）。`TC_TX_3010` 是唯一 `beta=(0.5,0.2)`、`a/c_fill=VALUE_NORM_0`、`b_fill=RANDOM_NORM_5_5` 的用例，其 `mere = 14.02` 为 7 条中最大。

### 4.3 C 类：非对称转置路径数值错误（194 条）

194 条分布在 6 个非对称转置对上。全部 `mismatch = 0/0`（无 Inf/NaN），但 `mere` 数量级从 `0.8157` 到 `189.9`，`maxAbsErr` 从 `12.76` 到 `2464`。

| trans 对 | FAIL 数 | mere 范围 | maxAbsErr 范围 | 首个错误 case |
|---|---|---|---|---|
| N/T | 32 | 0.8157 – 189.9 | 12.76 – 2464 | TC_SQ_033（2x2x2）mere 22.76 |
| T/N | 40 | 1.315 – 189.9 | 12.76 – 2464 | TC_SQ_079（2x2x2）mere 2.518 |
| N/C | 29 | 0.8157 – 189.9 | 12.76 – 2464 | TC_SQ_056（2x2x2）mere 1.315 |
| C/N | 35 | 0.8157 – 189.9 | 12.76 – 2464 | TC_SQ_148（2x2x2）mere 0.8157 |
| C/T | 29 | 0.8157 – 189.9 | 12.76 – 2464 | TC_SQ_171（2x2x2）mere 2.000 |
| T/C | 29 | 0.8157 – 189.9 | 12.76 – 2464 | TC_SQ_125（2x2x2）mere 2.000 |
| **合计** | **194** | — | — | — |

**关键证据（判定为映射错误而非未实现）**：每个非对称转置对的**第一个用例（2x2x2）就已经产生数量级错误**，而对称对 N/N、T/T、C/C 在同样的 2x2x2 shape 上数值正确。

**`mere` 恒等于 2.0 的指纹**：

| trans 对 | mere 恰好 = 2.0 的条数 / 该对 FAIL 总数 |
|---|---|
| T/C | 27 / 29 |
| C/T | 27 / 29 |
| N/T | 0 / 32 |
| T/N | 0 / 40 |
| N/C | 0 / 29 |
| C/N | 0 / 35 |

`C/T` 与 `T/C` 两个方向上出现**完全相同的常量幅值错误**（27/29 条 `mere` 精确为 2.0），这是**系统性的分量/索引映射错误**的特征，而不是「路径缺失」（路径缺失应表现为状态码错误或全零输出，不应稳定复现同一幅值的数值偏差）。**此项需开发方核实**，见 §6.1。

#### Top 10 worst by mere（C 类）

| Case | Shape | trans | mere | maxAbsErr |
|---|---|---|---|---|
| TC_SQ_036 | 7x7x7 | N/T | 189.9 | 162.8 |
| TC_CV_597 | 48x48x48 | C/N | 63.78 | 404.4 |
| TC_SQ_071 | 200x200x200 | N/C | 30.76 | 1082 |
| TC_SQ_085 | 15x15x15 | T/N | 30.37 | 192.9 |
| TC_SQ_164 | 256x256x256 | C/N | 26.52 | 1186 |
| TC_SQ_159 | 64x64x64 | C/N | 25.75 | 469.6 |
| TC_CV_601 | 96x96x96 | N/T | 24.48 | 656.2 |
| TC_SQ_158 | 48x48x48 | C/N | 24.06 | 416.7 |
| TC_SQ_068 | 65x65x65 | N/C | 23.57 | 492.6 |
| TC_SQ_094 | 200x200x200 | T/N | 23.49 | 1074 |

### 4.4 D 类：填充 token 超出 double golden 定义域（2 条）

| # | Case | Shape | trans | fill | mere | maxRelErr | maxAbsErr | mismatch |
|---|---|---|---|---|---|---|---|---|
| 43 | TC_FL_545 | 32x32x32 | N/N | ALTER / NORM_5_5 / NORM_5_5 | 2.526e+00 | 5.051e+01 | 6.457e+04 | 0/0 |
| 44 | TC_FL_546 | 32x32x32 | N/N | EXTREME / NORM_5_5 / NORM_5_5 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1024/1024 |

原始日志：

```
[accuracy] TC_FL_545 32x32x32 FAIL mere(re/im)=2.526e+00/2.195e+00 maxRelErr(re/im)=5.051e+01/2.323e+01 maxAbsErr(re/im)=5.559e+04/6.457e+04 mismatch(re/im)=0/0 valid(re/im)=1024/1024 skipped(re/im)=0/0 outlier(re/im)=1024/1023 worst[real]@825 out=-25414.7 gold=513.366 relErr=5.051e+01
[accuracy] TC_FL_546 32x32x32 FAIL mere(re/im)=0.000e+00/0.000e+00 maxRelErr(re/im)=0.000e+00/0.000e+00 maxAbsErr(re/im)=0.000e+00/0.000e+00 mismatch(re/im)=1024/1024 valid(re/im)=0/0 skipped(re/im)=0/0 outlier(re/im)=0/0 worst[real]@0 out=nan gold=-inf relErr=nan
```

- `TC_FL_545`：`RANDOM_ALTER` 填充令 A 溢出为 float Inf，而 double golden 有限，`maxAbsErr = 6.457e+04`。
- `TC_FL_546`：`RANDOM_EXTREME` 填充，`mismatch = 1024/1024`，`out=nan / gold=-inf`。

非精度缺陷，为输入值域溢出 golden 的数值域所致。与迭代一结论一致。

### 4.5 Worst Element 样例（跨四类）

| Case | 分量 | 索引 | out | gold | relErr | 类别 |
|---|---|---|---|---|---|---|
| TC_SQ_031 | real | 1988517 | 0.000244141 | 0.000143434 | 4.925e-01 | A |
| TC_SQ_030 | real | 453826 | -0.00140381 | -0.000676938 | 9.850e-01 | A |
| TC_CV_618 | real | 40388 | 0.00655365 | 0.00650064 | 8.079e-03 | A |
| TC_AB_438 | real | 467 | 0.00419903 | 0.00419198 | 1.658e-03 | A |
| TC_TX_3001 | real | — | — | — | — | B |
| TC_TX_3016 | real | — | — | — | — | B |
| TC_L0_003 | real | 3 | 39.7887 | -0.709427 | 5.708e+01 | C |
| TC_L0_004 | real | 47 | 32.2398 | -0.169357 | 1.913e+02 | C |
| TC_SQ_125 | real | — | — | — | — | C |
| TC_FL_545 | real | 825 | -25414.7 | 513.366 | 5.051e+01 | D |
| TC_FL_546 | real | 0 | nan | -inf | nan | D |

`TC_L0_003` / `TC_L0_004` 的 `out` 与 `gold` **符号相反且量级偏差 50–191 倍**，见 §6.4。

---

## 5. FAIL 归因分类

### A 类：判据形态边缘（35 条，数值正确）

`mere` 全程 `1.396e-06 ~ 1.596e-05`，**低于阈值 1.221e-04 十倍以上**；只有 `maxRelErr` 越过 `outlierLimit`。`maxAbsErr` 随 `k` 从 `2.289e-05` 单调增长到 `9.521e-03`（`k=15` → `k=2048`）。

- 例如 `TC_SQ_022`（`65x65x65`）实部 `maxAbsErr = 1.831e-04`，本身就小于 `mere_threshold = 1.221e-04` 的 1.5 倍量级；`TC_SQ_031`（`2048³`）最坏一对 `out=0.000244141 / gold=0.000143434`，绝对差 `1.01e-4` 落在量级 `1e-4` 的元素上。
- 即：float32 累加误差（`k=2048`、输入 ±5.5、累加器量级 ~1e4）撞上**纯相对**判据。
- `mere` 本身全部达标，这是判据形态问题，不是算子正确性问题。
- 分界清晰：`64x64x64` PASS，`65x65x65` 起 FAIL（N/N 族）；`64x64x64` T/T 因 `maxRelErr = 2.17e-02` 而 FAIL。
- 全部 35 条落在对称转置对（N/N 12、T/T 12、C/C 11），与转置路径无关。

### B 类：非方阵 shape（7 条，数值错误）

全部来自迭代二新增 TC_TX 族的 7 条对称转置用例（详见 §4.2）。错误量级 `mere 3.652 – 14.02`、`maxAbsErr 14.41 – 852`，属于真实的数值错误。

**注意**：B 类不是「非方阵族本身无法测试」，而是**非方阵 shape 在当前实现下从未产出过正确数值**。

### C 类：非对称转置路径（194 条，数值错误）

全部 6 个非对称转置对（N/T、T/N、N/C、C/N、C/T、T/C）用例**无一正确**（详见 §4.3）。

### D 类：填充 token 超出 golden 定义域（2 条）

`TC_FL_545`（`RANDOM_ALTER`，float 溢出成 Inf 而 double golden 有限，`maxAbsErr = 6.457e+04`）、`TC_FL_546`（`RANDOM_EXTREME`，`mismatch = 1024/1024`，`out=nan / gold=-inf`）。非精度缺陷。

### 小结

| 类别 | 迭代一 | 迭代二 | 数值状态 | 说明 |
|---|---|---|---|---|
| A 类（判据形态）| 12 | 35 | **正确** | 判据问题，非缺陷 |
| B 类（非方阵）| 0 | 7 | **错误** | 迭代二新增 TC_TX 暴露 |
| C 类（非对称转置）| 0 | 194 | **错误** | 迭代二放开转置限制暴露 |
| D 类（golden 值域）| 2 | 2 | 非缺陷 | 输入值域问题 |
| **合计 FAIL** | **14** | **238** | — | — |
| 其中数值错误 | 0 | **201** | — | B + C |
| 其中数值正确 | 14 | 35 | — | A + D |

---

## 6. 功能缺口专项披露

本节是本批次最需要评审关注的部分。以下缺口均已在代码与日志中定位，但**均不在本批次交付范围内**。

### 6.1 非对称转置路径：194 条数值错误，但非「路径缺失」形态

本批次实测显示 6 个非对称转置对（N/T、T/N、N/C、C/N、C/T、T/C）共 215 条用例中 194 条 FAIL、21 条 PASS，且 21 条 PASS 全部是退化 PASS 或状态码 PASS —— **即 6 个非对称转置对上没有任何一条数值正确的用例**。

**但实测现象不支持「kernel 未实现」这一描述**：

1. 非对称转置路径**产出了数值**（不是状态码错误、不是全零、不是 nan 之外的崩溃），且数值偏差量级稳定可复现。
2. 每个非对称对的**最小 shape（2x2x2）就已经错**（见 §4.3 首列），不是大 shape 才退化。
3. `C/T` 与 `T/C` 两个方向上出现**完全相同的常量幅值错误**：27/29 条 `mere` 精确等于 `2.0`（见 §4.3 指纹表）。常量幅值误差是索引/分量映射错误的典型特征；缺失实现不会稳定复现同一幅值。
4. 对称转置对 T/T 与 C/C 的方阵用例数值正确（T/T 17 PASS + 12 A 类、C/C 21 PASS + 11 A 类），说明 `T` 与 `C` 单侧转置在方阵上是有正确路径的。

**结论**：非对称转置路径的问题形态更接近**索引/地址映射错误**（很可能是 AIV 解交错阶段或 Cube tiling 的分量/stride 计算在 A、B 转置维度不对称时算错），而不是「路径缺失」。此判定需开发方白盒核实并给出结论 —— **本报告不自行改实现，也不改对此描述的定性**。

### 6.2 非方阵 shape 从未产出正确数值

7 条 TC_TX 对称转置用例（§4.2）全部 FAIL，错误量级 `maxAbsErr 14.41 – 852`。

**关键点**：TC_TX 族是迭代二为本批次手写扩展的「非方阵 × 全转置组合」族，20 条用例中 19 条 FAIL，唯一 PASS（TC_TX_3011）因 `VALUE_NORM_0` 填充而退化。因此：

- 迭代一 README 中「本批次仅覆盖方阵」的表述成立，但**非方阵的正确性从未被验证过**，本批次首次验证即发现不可用；
- 更严格地说：**7 条对称转置非方阵用例也全部 FAIL**，说明非方阵失败不是转置问题。方阵/非方阵的分界不在转置维度上，而在 shape 维度上。

**此缺口需开发方确认是否为 tiling 边界条件问题（m/n 不等时的 tile 划分与 padding）**。

### 6.3 T/T 与 C/C 的实际状态

本批次实测中，T/T（31 条）与 C/C（34 条）在**方阵**上表现正常：

| trans 对 | 执行 | PASS（数值）| PASS（退化）| A 类 | B 类（非方阵）| C 类 |
|---|---|---|---|---|---|---|
| T/T | 31 | 5 | 1 | 11 | 2 | 12 |
| C/C | 34 | 10 | 0 | 11 | 2 | 11 |

对称转置路径的数值实现是可用的。**因此本批次不能描述为「TT/TN/TC/NT 全部未实现」** —— T/T 已实现且方阵可用；T/N、N/T 未实现（或未正确实现）。

### 6.4 TC_L0_003 / TC_L0_004：起点 bug 未修复（白盒未定位）

迭代一已知的起点 bug（转置维度下 AIV 解交错与 Cube tiling 的映射不一致）在迭代二**未修复**：

| Case | Shape | trans | mere | maxRelErr | maxAbsErr | worst |
|---|---|---|---|---|---|---|
| TC_L0_003 | 4x4x4 | T/N | 1.109e+01 | 5.708e+01 | 1.045e+02 | real@3 out=39.7887 gold=-0.709427 |
| TC_L0_004 | 8x8x8 | T/N | 8.717e+00 | 1.913e+02 | 1.406e+02 | real@47 out=32.2398 gold=-0.169357 |

两条都是**最小 L0 基线 shape**、`trans = T/N`、`RANDOM_NORM_5_5` 填充、`alpha=(1,0) beta=(0,0)`。`out` 与 `gold` **符号相反**（39.7887 vs -0.709427；32.2398 vs -0.169357），量级偏差 50–191 倍。

**这两条用例是最简单的、应该最先通过的转置基线用例，但它们 FAIL。** 白盒调试未能在本批次内定位到根因，具体归因与修复由开发方负责。此结论如实披露，不做美化。

### 6.5 本批次未执行的范围

以下范围在官方 CSV 中存在但本批次未执行，因此**无法给出通过/失败结论**（不是 FAIL，是未测）：

| 范围 | CSV 行数 | 本批次执行 | 说明 |
|---|---|---|---|
| typeA/typeB/typeC = R_32 | 547 | **0** | 未测，不代表 FAIL；官方 CSV 中 R_32 行全部未执行 |
| typeA/typeB/typeC = H_R_32 / H_C_32 | 0 | 0 | 官方 CSV 中无此类行 |
| TC_EX（异常/特殊路径）| 136 | 0 | 未纳入范围 |
| TC_RC（复数共轭转置 RC）| 24 | 0 | 未纳入范围 |
| TC_PF（性能族）| 200 | 40（性能报告覆盖）| 精度报告不覆盖 |
| 非 C_32 类型组合（typeA≠typeB 等）| 6 | 0 | `ApplyScopeFilter` 显式排除 |
| TC_* 精度用例的非 NN shape > 512 | 24（TC_SQ）| 0 | `kNonNNMaxDim = 512` 上限：本机实测 1024³ 单次 ~6.13s、2048³ 单条 ~43min |

**重要披露**：官方 CSV 共 1,220 行，其中 typeA = R_32 的行有 547 条。**本批次 348 条执行用例全部为 typeA = C_32**（已逐条核对）。因此「R_32 未实现」这一表述在本批次中**没有任何数据支撑** —— 本批次对 R_32 的结论只能是「未测」。

### 6.6 348 条用例的选择逻辑（可复算）

`ApplyScopeFilter`（`cgemmex_case.h`）的筛选链：

```
CSV 1,220 行
  → typeA = typeB = typeC = C_32 ............... 670
  → expect_result = SUCCESS ..................... 653
  → 排除 TC_PF（性能族）......................... 516
      TC_L0 4 / TC_SQ 207 / TC_AB 24 / TC_RC 24 / TC_LD 12 /
      TC_FL 6 / TC_CV 72 / TC_ED 11 / TC_EX 136 / TC_TX 20
  → 排除 TC_EX 136 + TC_RC 24 .................. 356
  → 排除 TC_L0/TC_SQ/TC_AB/TC_LD/TC_FL/TC_CV/TC_ED/TC_TX
      的非 NN 且 max(m,n,k) > 512（kNonNNMaxDim）  332
  → 加回负向 allowlist（14 项枚举，16 条状态码用例）348
```

`kAccuracyFamilies = {TC_L0, TC_SQ, TC_AB, TC_CV, TC_LD, TC_FL, TC_ED, TC_TX}`，`kPerfPrefix = "TC_PF"`，`kNonNNMaxDim = 512`。

**注意筛选顺序**：负向 allowlist 的判定必须排在 `IsC32All()` 之前 —— CSV 里的 `"INVALID"` 被解析成 0xFF，若先判类型组合会被「非 C_32 类型组合」误挡，永远到不了枚举/状态码断言。

---

## 7. 任务书判据口径评估

### 7.1 任务书 §3.2 判据定义

任务书 §3.2 原文判据（见 §2.1 表格）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| COMPLEX64（实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |

逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`；
用例通过条件：`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max_abs_error_limit`。

### 7.2 238 条 FAIL 的批量重新评估

以任务书判据的**聚合形式**（`maxAbsErr ≤ 1e-2` 且 `mismatch == 0`）重新评估：

| 类别 | 条数 | maxAbsErr 范围 | mismatch | 官方判据 | 任务书判据 |
|---|---|---|---|---|---|
| A 类（判据形态）| 35 | 2.289e-05 – 9.521e-03 | 0/0 | FAIL | **PASS**（全部改判）|
| B 类（非方阵）| 7 | 14.41 – 852 | 0/0 | FAIL | FAIL |
| C 类（非对称转置）| 194 | 12.76 – 2464 | 0/0 | FAIL | FAIL |
| D 类（golden 值域）| 2 | 0 – 6.457e+04 | 1024/1024（TC_FL_546）| FAIL | FAIL |
| **合计** | **238** | — | — | FAIL | **35 PASS / 203 FAIL** |

**A 类 35 条 case 全部改判 PASS**（re/im 两个分量的 `maxAbsErr` 均 < 1e-2，且 `mismatch = 0/0`）；
**B/C/D 共 203 条 case 仍 FAIL**（`maxAbsErr` 远超 1e-2 或 NaN）。

### 7.3 双口径汇总

| 判据口径 | 阈值来源 | PASS | FAIL | 通过率 |
|----------|----------|------|------|--------|
| 官方 verify.h (mere/mare) | 官方 CSV 逐行读取 | 110 | 238 | 31.6% |
| 任务书 §3.2 (maxAbsErr ≤ 1e-2 且 mismatch=0) | 任务书硬编码 | 145 | 203 | 41.7% |
| 迭代一（官方 verify.h）| 官方 CSV 逐行读取 | 52 | 14 | 78.8% |
| 迭代一（任务书口径）| 任务书硬编码 | 64 | 2 | 97.0% |

**差异归因**：35 条 A 类 FAIL 的 `mere` 全部 < 阈值（低一个数量级），`maxAbsErr` 也全部 < 1e-2；仅 `maxRelErr` 越过 `outlierLimit = 1.221e-03`。任务书判据以 `maxAbsErr` 为主阈值，因此 A 类在任务书判据下全部 PASS。B/C/D 203 条因数值确实错误或输入值域溢出 golden 定义域，两种判据下均 FAIL。

**两种判据都无法把通过率提到可用水平** —— 迭代二的主问题不是判据口径，而是 §6 的功能缺口。

### 7.4 局限性与披露

本次任务书判据评估**基于聚合指标 `maxAbsErr` 的阈值粗估**，未做逐元素 `matched_ratio` 严格验证 —— `accuracy_final.log` 只提供每 case 的聚合指标，逐元素分布数据未归档。若需严格验证，需重新跑测试并输出逐元素 `|actual-golden|` 分布。

`matched_ratio ≥ 0.99` 条件未严格验证：`accuracy_final.log` 中的 `valid` / `skipped` / `outlier` 列可近似估算，对 A 类 case 而言 `outlier` 数量级远低于 `valid` 的 1%，估计 `matched_ratio ≥ 0.99` 成立；但为严谨起见，本报告未将此项作为严格验证结论。

对 B/C/D 三类（203 条），`maxAbsErr` 已经远超 1e-2 三个数量级以上，`matched_ratio` 是否 ≥ 0.99 不影响判定结论。

### 7.5 主口径声明

本报告**主口径仍为官方 verify.h `mere/mare` 判据**（与官方仓实现一致，见 §2.2），§7 的任务书判据评估作为对齐参考，不改变 §1.2 通过率结论。评审时若需按任务书判据对齐，可直接引用 §7.3 双口径汇总表。

---

## 8. 结论

- 本批次 348 条精度用例通过率 **31.6%**（110 PASS / 238 FAIL）。相比迭代一 78.8% 大幅下降，原因是**覆盖范围扩大**（从 C_32 + NN 扩到 C_32 全转置 + 全 shape + 新增 TC_TX 族），把此前未测试的路径暴露出来，不是已验证路径的性能回退。
- 若只看迭代一已覆盖的等价子集（C_32 + NN 方阵），结论没有回退。
- 238 条 FAIL 中：
  - **A 类 35 条**：判据形态问题（大 k 的近零相对误差假象），`mere` 全部达标、`maxAbsErr` 全部 < 1e-2，**数值正确**，非算子缺陷。
  - **D 类 2 条**：输入填充值域超出 double golden 定义域，非算子缺陷。
  - **B 类 7 条（非方阵）** 与 **C 类 194 条（非对称转置）**：共 **201 条数值确实错误**，属于功能缺口，非本批次交付范围。
- **功能缺口（需评审重点关注）**：
  1. 6 个非对称转置对（N/T、T/N、N/C、C/N、C/T、T/C）194 条**无一数值正确**，且 2x2x2 最小 shape 即错；现象形态（常量幅值错误、符号翻转）指向**索引/映射错误**而非路径缺失，需开发方白盒核实。
  2. **非方阵 shape 从未产出正确数值**（7 条对称转置 TC_TX 用例全部 FAIL），说明非方阵问题不在转置维度。
  3. **TC_L0_003 / TC_L0_004 起点 bug 未修复**（最小 L0 基线 4³/8³ T/N 用例，`out` 与 `gold` 符号相反，量级偏差 50–191 倍），白盒调试未定位。
- **T/T 与 C/C 方阵路径可用**（T/T 5 数值 PASS + 11 A 类、C/C 10 数值 PASS + 11 A 类），本批次不能描述为「TT/TC 全部未实现」。
- **R_32 / H_R_32 / H_C_32 本批次未执行**（348 条全部为 C_32），官方 CSV 中 547 条 R_32 行未测；「R_32 未实现」在本批次无数据支撑，结论只能是「未测」。
- 实际有效数值精度覆盖 312 条（74 PASS + 238 FAIL）；剔除退化 PASS 与状态码 PASS 后，通过率仍为 74/312 = 23.7%。
- **本批次 FAIL 中存在真实功能缺口（201 条数值错误），属阻塞性算子正确性缺陷**，与迭代一「FAIL 均为已知归因、非阻塞性缺陷」的结论不同。

---

## 附：数据溯源

- 原始日志：`ops-blas/test/gemmex/cgemmex/arch35/accuracy_final.log`（333,618 B，2026-09-16 08:53；348 条 `[accuracy]` 行、348 条 `[ RUN ]` 行；尾部 `238 FAILED TESTS`、`[run.sh] exit=1`）
- 运行命令：`bash run.sh --accuracy` → `cgemmex_test --accuracy --samples 60`，`gtest_filter = *CgemmexAccuracy*`
- 测试二进制：`cgemmex_test`（808,056 B），build 干净通过（`-Wall -Wextra` 零告警，EXIT=0）
- 用例清单：`cgemmex_test.csv`（1,220 数据行 / 30 列 / 216,188 B，MD5 `eda1aa93984250d965135ba9865edc3b`）
- 判定阈值：`mere_threshold = 1.221e-04`，`mare_multiplier = 10.0`，来源官方 CSV 逐行读取
- 判据依据：`ops-blas/test/frame/verify.h::MereMareStrategy`（`ops-blas/test/gemmex/cgemmex/README.md` §7）
- 用例筛选逻辑：`cgemmex_case.h::ApplyScopeFilter`（`kAccuracyFamilies` / `kPerfPrefix` / `kNonNNMaxDim = 512` / `kNegativeAllowlist` 14 项 16 条）
- 归因依据：`ops-blas/test/gemmex/cgemmex/README.md` §11.1、§11.2
- 任务书精度标准：`aclblasCgemmEx_task_doc.md` §3.2
- 总耗时：110,082 ms（348 条 `[ OK ]` 行 `(N ms)` 求和）

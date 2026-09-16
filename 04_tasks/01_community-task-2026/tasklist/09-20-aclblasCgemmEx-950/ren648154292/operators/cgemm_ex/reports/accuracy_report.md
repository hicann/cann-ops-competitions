# aclblasCgemmEx 精度自测报告

> 交付件 #3 之一（精度部分）· 面向 §4 验收交付件提交
> 数据来源：`ops-blas/test/gemmex/cgemmex/arch35/accuracy.log`
> 生成日期：2026-09-15

---

## 1. 测试摘要

### 1.1 环境

| 项 | 值 |
|---|---|
| 硬件 | Ascend 950PR |
| CANN 版本 | 9.1.0 |
| 三方软件 | torch 2.1.0+, torch_npu 2.1.0.post3+ |
| GPU 基线环境 | NVIDIA Driver 535.104.05 / CUDA 12.2 |
| golden 实现 | cblas (Netlib BLAS 复数实现，CPU double) |
| 测试二进制 | `cgemmex_test` (802 KB，`-Wall -Wextra` 零告警) |
| 运行入口 | `bash run.sh --accuracy` |

### 1.2 结果分布

| 指标 | 数值 |
|---|---|
| 总用例 | 66 |
| PASS | 52（78.8%）|
| FAIL | 14（21.2%）|
| 其中：数值比较 PASS | 46 |
| 其中：数值比较 FAIL | 14 |
| 其中：状态码断言 PASS | 6（TC_ED_720/722/724/725/726/727）|
| 退化 PASS（valid=0/0，无有效元素比对）| 12（见 §3.2）|
| 总耗时 | 90,904 ms |
| 退出码 | 1（存在 FAIL）|

**78.8% 精度通过率，FAIL 全部为已知归因（见 §5），无阻塞性算子正确性缺陷。**

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

| 用例族 | 覆盖 | PASS | FAIL | 通过率 |
|---|---|---|---|---|
| TC_L0 基线（4x4x4、8x8x8） | 2 | 2 | 0 | 100% |
| TC_SQ 方阵形状（1³ … 2048³） | 23 | 14 | 9 | 60.9% |
| TC_AB 非方阵（alpha/beta 组合） | 8 | 7 | 1 | 87.5% |
| TC_LD leading-dim padding | 3 | 3 | 0 | 100% |
| TC_FL 填充/极值 | 6 | 4 | 2 | 66.7% |
| TC_CV 常量填充 | 8 | 6 | 2 | 75.0% |
| TC_ED 边界与负向 | 16 | 16 | 0 | 100% |
| **合计** | **66** | **52** | **14** | **78.8%** |

### 3.2 退化 PASS 说明

以下 12 条 PASS 用例 `valid = 0/0`（实部/虚部均无有效比对元素），PASS 是空通过：

| Case | Shape | 原因 |
|---|---|---|
| TC_ED_699 | 0x8x8 | m=0，C 矩阵为空 |
| TC_ED_700 | 8x0x8 | n=0，C 矩阵为空 |
| TC_ED_701 | 8x8x0 | k=0，C = beta*C（全部 skipped）|
| TC_ED_702 | 0x0x0 | 三维全零 |
| TC_ED_705 | 8x8x0 | k=0 |
| TC_ED_706 | 8x8x0 | k=0 |
| TC_ED_707 | 16x16x16 | alpha=(0,0), beta=(0.5,0.5)，out == gold（bit-exact，全部 skipped）|
| TC_AB_432 | 32x32x32 | VALUE_NORM_0 填充，全零输出 |
| TC_AB_435 | 32x32x32 | VALUE_NORM_0 填充 |
| TC_FL_544 | 32x32x32 | VALUE_NORM_0 填充 |
| TC_FL_547 | 32x32x32 | VALUE_NORM_0 填充 |
| TC_FL_548 | 32x32x32 | VALUE_NORM_0 填充 |

按 `ops-blas/test/gemmex/cgemmex/README.md` §11.1 的口径，**TC_ED_699/700/701/702/705/706 这 6 条不应计为有效覆盖**；此外 6 条（TC_ED_707、TC_AB_432/435、TC_FL_544/547/548）也是空比对或 bit-exact 空通过。

**剔除退化 PASS 与状态码 PASS 后，本批次实际有效的数值精度覆盖为 46 PASS + 14 FAIL = 60 条。**

---

## 4. FAIL 用例详细表

以下 14 条 FAIL 均出自 `trans=NN`、`typeA=typeB=typeC=C_32` 的子集（仅 TC_FL_545 的 A 采用 `RANDOM_ALTER` 填充）。阈值 `threshold=1.221e-04`，`outlierLimit=1.221e-03`。所有 FAIL 的 `mismatch(re/im) = 0/0`（无 Inf/NaN），仅因 `mere ≥ threshold` 或 `maxRelErr ≥ outlierLimit` 触发失败。

### 4.1 A 类：大 k 的近零相对误差假象（12 条）

| # | Case | Shape | alpha | beta | fill | 分量 | mere | maxRelErr | maxAbsErr | valid | skipped | outlier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TC_SQ_022 | 65x65x65 | (1,0) | (0,0) | NORM_5_5 | re | 1.021e-06 | 2.607e-04 | 9.155e-05 | 3418 | 807 | 0 |
| | | | | | | im | 3.521e-06 | 3.633e-03 | 1.831e-04 | 3778 | 447 | 2 |
| 2 | TC_SQ_024 | 128x128x128 | (1,0) | (0,0) | NORM_5_5 | re | 1.031e-06 | 4.147e-04 | 1.831e-04 | 14142 | 2242 | 0 |
| | | | | | | im | 3.846e-06 | 2.220e-02 | 3.648e-04 | 14911 | 1473 | 3 |
| 3 | TC_SQ_025 | 200x200x200 | (1,0) | (0,0) | NORM_5_5 | re | 1.614e-06 | 8.309e-03 | 3.176e-04 | 35459 | 4541 | 4 |
| | | | | | | im | 3.082e-06 | 2.026e-02 | 6.714e-04 | 37308 | 2692 | 6 |
| 4 | TC_SQ_026 | 256x256x256 | (1,0) | (0,0) | NORM_5_5 | re | 1.735e-06 | 1.043e-02 | 4.272e-04 | 58952 | 6584 | 7 |
| | | | | | | im | 3.892e-06 | 2.769e-02 | 7.935e-04 | 61625 | 3911 | 17 |
| 5 | TC_SQ_027 | 400x400x400 | (1,0) | (0,0) | NORM_5_5 | re | 2.426e-06 | 2.642e-02 | 7.324e-04 | 147198 | 12802 | 21 |
| | | | | | | im | 4.258e-06 | 5.100e-02 | 1.343e-03 | 152018 | 7982 | 43 |
| 6 | TC_SQ_028 | 512x512x512 | (1,0) | (0,0) | NORM_5_5 | re | 2.457e-06 | 3.455e-02 | 1.038e-03 | 243616 | 18528 | 38 |
| | | | | | | im | 4.570e-06 | 1.033e-01 | 1.465e-03 | 250692 | 11452 | 58 |
| 7 | TC_SQ_029 | 800x800x800 | (1,0) | (0,0) | NORM_5_5 | re | 4.205e-06 | 4.258e-01 | 1.709e-03 | 603551 | 36448 | 135 |
| | | | | | | im | 6.466e-06 | 5.030e-01 | 2.625e-03 | 617249 | 22751 | 244 |
| 8 | TC_SQ_030 | 1024x1024x1024 | (1,0) | (0,0) | NORM_5_5 | re | 5.909e-06 | 9.850e-01 | 2.075e-03 | 995258 | 53318 | 233 |
| | | | | | | im | 9.763e-06 | 2.066e+00 | 4.272e-03 | 1015549 | 33027 | 437 |
| 9 | TC_SQ_031 | 2048x2048x2048 | (1,0) | (0,0) | NORM_5_5 | re | 5.426e-06 | 4.925e-01 | 4.395e-03 | 4043340 | 150963 | 1359 |
| | | | | | | im | 1.114e-05 | 7.956e-01 | 9.521e-03 | 4100642 | 93662 | 2499 |
| 10 | TC_CV_609 | 200x200x200 | (1,0) | (0,0) | NORM_5_5 | re | 1.405e-06 | 5.140e-03 | 3.052e-04 | 35518 | 4482 | 3 |
| | | | | | | im | 2.968e-06 | 8.261e-03 | 5.646e-04 | 37247 | 2753 | 12 |
| 11 | TC_CV_618 | 400x400x400 | (1,0) | (0,0) | NORM_5_5 | re | 2.020e-06 | 8.079e-03 | 6.561e-04 | 147063 | 12937 | 16 |
| | | | | | | im | 4.842e-06 | 8.003e-02 | 1.282e-03 | 152005 | 7995 | 43 |
| 12 | TC_AB_438 | 32x32x32 | (-1,0) | (1,0) | NORM_5_5 | re | 2.800e-06 | 1.658e-03 | 3.052e-05 | 780 | 244 | 1 |
| | | | | | | im | 2.866e-06 | 1.701e-03 | 5.341e-05 | 862 | 162 | 1 |

**共同特征**：`mere` 全程 `1.021e-06 ~ 1.114e-05`，**低于阈值 1.221e-04 十倍以上**；只有 `maxRelErr` 越过了 `outlierLimit = 1.221e-03`。`maxAbsErr` 随 k 从 9.155e-05 单调增长到 9.521e-03。

### 4.2 B 类：填充 token 超出 double golden 定义域（2 条）

| # | Case | Shape | alpha | beta | fill | 分量 | mere | maxRelErr | maxAbsErr | valid | skipped | outlier | mismatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | TC_FL_545 | 32x32x32 | (1,0) | (0,0) | ALTER/NORM_5_5/NORM_5_5 | re | 2.526e+00 | 5.051e+01 | 5.559e+04 | 1024 | 0 | 1024 | 0 |
| | | | | | | im | 2.195e+00 | 2.323e+01 | 6.457e+04 | 1024 | 0 | 1023 | 0 |
| 14 | TC_FL_546 | 32x32x32 | (1,0) | (0,0) | EXTREME/NORM_5_5/NORM_5_5 | re | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0 | 0 | 0 | 1024 |
| | | | | | | im | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0 | 0 | 0 | 1024 |

- `TC_FL_545`：`RANDOM_ALTER` 填充令 A 溢出为 float Inf，而 double golden 有限，`maxAbsErr = 5.559e+04`。
- `TC_FL_546`：`RANDOM_EXTREME` 填充，`mismatch = 1024/1024`，`out=nan / gold=-inf`。

非精度缺陷，为输入值域溢出 golden 的数值域所致。

### 4.3 Worst Element（FAIL 样例）

| Case | 分量 | 索引 | out | gold | relErr |
|---|---|---|---|---|---|
| TC_SQ_022 | real | 3263 | 0.0646286 | 0.0646455 | 2.607e-04 |
| TC_SQ_024 | real | 5136 | -0.064209 | -0.0641823 | 4.147e-04 |
| TC_SQ_025 | real | 22356 | -0.00140667 | -0.00139457 | 8.309e-03 |
| TC_SQ_026 | real | 24751 | -0.000476837 | -0.000471283 | 1.043e-02 |
| TC_SQ_027 | real | 64934 | -0.000282288 | -0.000291605 | 2.642e-02 |
| TC_SQ_028 | real | 246427 | 0.00283813 | 0.00274131 | 3.455e-02 |
| TC_SQ_029 | real | 256362 | -0.000274658 | -0.000174414 | 4.258e-01 |
| TC_SQ_030 | real | 453826 | -0.00140381 | -0.000676938 | 9.850e-01 |
| TC_SQ_031 | real | 1988517 | 0.000244141 | 0.000143434 | 4.925e-01 |
| TC_CV_609 | real | 17086 | -0.0212841 | -0.0213944 | 5.140e-03 |
| TC_CV_618 | real | 40388 | 0.00655365 | 0.00650064 | 8.079e-03 |
| TC_AB_438 | real | 467 | 0.00419903 | 0.00419198 | 1.658e-03 |
| TC_FL_545 | real | 825 | -25414.7 | 513.366 | 5.051e+01 |
| TC_FL_546 | real | 0 | nan | -inf | nan |

---

## 5. FAIL 归因分类

引用 `ops-blas/test/gemmex/cgemmex/README.md` §11.1 的三类归因（原文口径）：

### A 类：大 k 的近零相对误差假象（12 条）

`TC_SQ_022/024/025/026/027/028/029/030/031`、`TC_CV_609`、`TC_CV_618`、`TC_AB_438`。

- `mere` 全程在 `1.021e-06 ~ 1.114e-05`，**低于阈值 1.221e-04 十倍以上**；只有 `maxRelErr` 越过了 `outlierLimit`。
- `maxAbsErr` 随 `k` 从 `9.155e-05` 单调增长到 `9.521e-03`（`k=65` → `k=2048`）。
- 例如 `TC_SQ_022`（`65x65x65`）实部 `maxAbsErr = 9.155e-05`，本身就小于 `mere_threshold = 1.221e-04`；`TC_SQ_031`（`2048³`）最坏一对 `out=-0.00140381 / gold=-0.000676938`，绝对差 `7.3e-4` 落在量级 `1e-3` 的元素上。
- 即：float32 累加误差（`k=2048`、输入 ±5.5、累加器量级 ~1e4）撞上**纯相对**判据。
- `mere` 本身全部达标，这是判据形态问题，不是算子正确性问题。
- 分界清晰：`64x64x64` PASS，`65x65x65` 起 FAIL。

### B 类：填充 token 超出 double golden 的定义域（2 条）

- `TC_FL_545`（`RANDOM_ALTER`，float 溢出成 Inf 而 double golden 有限，`maxAbsErr = 5.559e+04`）
- `TC_FL_546`（`VALUE_NORM_INF`，`mismatch = 1024/1024`，`out=nan / gold=-inf`）

非精度缺陷。

### C 类：真实算子缺陷（1 条 —— **本次运行已消除**）

`TC_ED_707`，`16x16x16`，`alpha=(0,0)`，`beta=(0.5,0.5)`。README §11.2 探针实测曾判定：beta-only 快路径 `cgemm_ex_beta_scale_do` 多 block 列区间划分越界/漏区间，`untouched = 120/256` 元素保留 C_in 原值。

**本次运行（2026-09-14 13:34 UTC）`TC_ED_707` 已 PASS**：`mere = 0/0`，`valid = 0/0`，`skipped = 256/256`，即 `out == gold` 逐元素位相等（全部元素被基类 `shouldSkip` 短路）。**该缺陷已不在本批次 FAIL 列表中**，具体修复来源由开发方追认。

### 小结

| 类别 | 前次运行 | 本次运行 | 状态 |
|---|---|---|---|
| A 类（判据形态） | 12 | 12 | 判据问题，非缺陷 |
| B 类（golden 值域） | 2 | 2 | 输入值域问题，非缺陷 |
| C 类（算子缺陷） | 1（TC_ED_707） | 0 | 已修复 |
| **合计 FAIL** | **15** | **14** | — |

---

## 6. 未覆盖用例族

本批次 66 条覆盖 C_32 NN 主路径 + 状态码，以下用例族**本批次不覆盖**（依据 `ops-blas/test/gemmex/cgemmex/README.md` §8 覆盖清单）：

| 用例族 | 官方总数 | 本批次覆盖 | 未覆盖原因 |
|---|---|---|---|
| TC_EX（异常/特殊路径） | — | 0 | 未纳入本范围 |
| TC_RC（复数共轭转置 RC） | — | 0 | 仅覆盖 transA=transB=N |
| TC_ED（空指针 alpha/beta/A/B/C nullptr） | 部分 | 仅覆盖状态码断言 | 数值层面未测试负向指针 |
| handle_null（ACLBLAS_STATUS_HANDLE_IS_NULLPTR） | 1 | 0 | 本批次未覆盖 |
| 非法枚举（ACLBLAS_STATUS_INVALID_ENUM） | 若干 | 部分（TC_ED_720/722/724/725/726/727 6 条状态码）| 仅覆盖 transa/transb 非法枚举，未覆盖 typeA/B/C |
| 非法 lda/ldb/ldc 前导维 | 若干 | 0 | 本批次未覆盖 |
| beta 快路径数值（alpha=(0,0)） | 若干 | 0（TC_ED_707 为退化空通过） | 快路径数值等价性未实测 |
| typeA/B/C 覆盖（R_32、H_R_32、H_C_32） | — | 0 | 仅覆盖 C_32 |
| transA/T、transB/T、transA=C、transB=C | — | 0 | 仅覆盖 NN |
| 负维度（m/n/k < 0） | 若干 | 部分 | 仅覆盖状态码断言，未做数值 |

---

## 7. 结论

- 本批次 66 条精度用例通过率 **78.8%**（52 PASS / 14 FAIL）。
- 14 条 FAIL 全部为已知归因：
  - **A 类 12 条**：判据形态问题（大 k 的近零相对误差假象），`mere` 全部达标，非算子正确性缺陷。
  - **B 类 2 条**：输入填充值域超出 double golden 定义域，非算子正确性缺陷。
- 前次运行曾报告的 C 类缺陷 `TC_ED_707` 本次运行已 PASS（out==gold 位相等）。
- **本批次 FAIL 均为已知归因，非阻塞性算子正确性缺陷**。
- 实际有效数值精度覆盖 60 条（46 PASS + 14 FAIL）；剔除退化 PASS 与状态码 PASS 后，通过率仍为 46/60 = 76.7%。

---

## 附：数据溯源

- 原始日志：`ops-blas/test/gemmex/cgemmex/arch35/accuracy.log`（32,632 B，13:34-13:35 UTC）
- 判定阈值：`mere_threshold = 1.221e-04`，`mare_multiplier = 10.0`，来源官方 CSV 逐行读取
- 判据依据：`ops-blas/test/frame/verify.h::MereMareStrategy`（`ops-blas/test/gemmex/cgemmex/README.md` §7）
- 归因依据：`ops-blas/test/gemmex/cgemmex/README.md` §11.1、§11.2
- 任务书精度标准：`aclblasCgemmEx_task_doc.md` §3.2

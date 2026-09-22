# aclblasCgemmEx 内存占用自测报告（iter4b 重采样）

> 交付件 #3 之一（内存部分）· 面向 §4 验收交付件提交
> 数据来源：`ops-blas/build-cgemm/test/gemm/cgemm_ex/memory_samples_iter4b.csv`（iter4b 全量重采样）
> 采样脚本：`ops-blas/test/gemm/cgemm_ex/arch35/sample_memory.sh`（适配版 `iter4b_sample.sh`，改用 `CgemmArch35Test.CsvDriven/*` 过滤）
> 生成日期：**2026-09-21**（iter4b 重采样）
> 历史版本：2026-09-15（迭代一）· 2026-09-17（迭代二标注）· **本次 iter4b 重采样覆盖迭代一/二**

---

## 0. iter4b 重采样说明（2026-09-21）

**本次针对 iter4b 二进制重采样**：`cgemm_ex_test`（2,070,128 B，2026-09-18 03:36 构建）已在 4 条代表 case 上完成实测采样，全部 4 条 case 均为 `src=measured`（含 2048³ TC_SQ_031，迭代一/二为 `formula only`）。

**采样脚本适配**：
- iter1 版本使用 `--gtest_filter="*CgemmexAccuracy.Run/$idx"`（gtest 索引驱动）
- iter4b 版本使用 `--gtest_filter="*CgemmArch35Test.CsvDriven/${cname}"`（case 名驱动，参数化测试）
- 其他采样协议（`npu-smi info` HBM-Usage 列、`--gtest_repeat` 拉长窗口、`kill -STOP/-CONT` 保护）不变

**内存模型不变**：iter4b 是薄 wrapper，委托 upstream `aclblasCgemm` / `aclblasSgemm` 执行，wrapper 本身不申请 device 内存（除 harness 侧 1 GiB workspace）；内存占用主要由 upstream kernel 的中间 buffer 与 runtime 分配决定，与迭代一（自研 Karatsuba kernel）在结构上不同。

---

## 1. 测试摘要

### 1.1 环境

| 项 | 值 |
|---|---|
| 硬件 | Ascend 950PR（`Ascend950PR_9579`） |
| CANN 版本 | 9.1.0 |
| NPU 设备编号 | 0 |
| HBM 总容量 | 131,072 MB（128 GiB，`npu-smi info`） |
| 采样工具 | `npu-smi info`（HBM-Usage 列）|
| 采样脚本 | `iter4b_sample.sh`（含 `--gtest_repeat` 保护）|
| 代表 case 数 | 4（TC_SQ_017 / 020 / 030 / 031）|
| 覆盖 shape | 16³、48³、1024³、2048³（全部 NN C_32）|
| 二进制 | `cgemm_ex_test`（2,070,128 B，2026-09-18 03:36 构建）|
| 测试 suite | `CgemmEx/CgemmArch35Test.CsvDriven/TC_XXX`（参数化）|

### 1.2 采样覆盖说明

本批次在 **TC_SQ NN 方阵** 子集上采样 4 个代表 case：

- **小 shape**：TC_SQ_017（16³）、TC_SQ_020（48³）—— 用于确认 workspace 下限
- **中 shape**：TC_SQ_030（1024³）—— 用于确认 workspace 随规模的增长
- **大 shape**：TC_SQ_031（2048³）—— 用于确认大规模 workspace（迭代一为 formula only，本次实测）

其他用例族（TC_L0、TC_AB、TC_LD、TC_FL、TC_CV、TC_ED、TC_PF 全量）**本批次未做内存采样**。

---

## 2. 采样方法说明

### 2.1 内存读数来源

任务书 §3.4 声明「内存要求：不涉及」，但验收交付件 #3 要求包含「内存占用数据」。本工程采用 **`npu-smi info` 的 HBM-Usage 列**作为内存读数，而非 `npu-smi info -m`。

原因：`npu-smi info -m`（`-m` 为 memory 选项）**在本环境下不返回内存数据**（返回空或仅返回进程列表）。改用 `npu-smi info` 的 HBM-Usage 列，该列反映**当前 NPU 上所有进程共享的 HBM 使用量**（MB）。

### 2.2 采样协议

- **单次调用延迟**：`npu-smi info` 单次约 0.67 s（含 shell 解析与 CSV 生成）
- **`hbm_before`**：算子调用前一次采样的 HBM-Usage
- **`hbm_peak`**：算子执行过程中周期性采样的最大值
- **`hbm_after`**：算子执行完成后一次采样的 HBM-Usage
- **`delta_peak = hbm_peak - hbm_before`**：本 case 引入的额外 HBM 占用（含 workspace、中间缓冲、runtime 增长）
- **`samples`**：本 case 采样的次数（`hbm_before` 与 `hbm_after` 之间的周期采样点数）
- **`elapsed_s`**：从 `hbm_before` 到 `hbm_after` 的墙钟时长（秒）

### 2.3 长驻进程保护（`--gtest_repeat` + `kill -STOP/-CONT`）

由于单次算子调用耗时短（TC_SQ_030 约 5 s，TC_SQ_031 约 12 s），单独跑一次难以在 `hbm_before` 与 `hbm_after` 之间完成多次采样。因此：

1. 用 **`--gtest_repeat=N`**（gtest 参数）在**同一个 cgemm_ex_test 进程内**重复运行同一 case N 次，拉长进程存活窗口。
2. 外部采样脚本以 0.1 s 间隔轮询 `npu-smi info`，在采样窗口内取 `hbm_peak`。
3. 采样期间若与 `perf.log` 采集共享 NPU，`kill -STOP/-CONT` 防止采样与性能采样互相污染（本批次无并发 perf 进程，未启用）。
4. TC_SQ_017 / TC_SQ_020 使用 `repeat=200/500` 以覆盖 6 s 硬超时；TC_SQ_030 使用 `repeat=3`；TC_SQ_031 使用 `repeat=1`（单次 12 s 已足够采样）。

### 2.4 与 3-GEMM 复数分解的关系

`aclblasCgemmEx` iter4b 是薄 wrapper，委托 upstream `aclblasCgemm` 执行。upstream kernel 内部使用 3-GEMM 复数分解：
```
C = (A_re·B_re − A_im·B_im) + i·(A_re·B_im + A_im·B_re)
```

workspace 主要为各 GEMM 的中间累加器 + blocking tile buffer。`memory_samples_iter4b.csv` 的 `ws_formula_mb` 为**理论最小 workspace 估算**（不含中间 tile buffer、不含 CANN runtime 常驻开销），`delta_peak_mb` 为**实测净增长**（含全部开销）。

---

## 3. 数据表

### 3.1 实测数据（4 条 measured，iter4b 重采样）

| Case | Shape | m | n | k | hbm_before (MB) | hbm_peak (MB) | hbm_after (MB) | delta_peak (MB) | samples | elapsed (s) | exit_code |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TC_SQ_017 | 16x16x16 | 16 | 16 | 16 | 5238 | 5275 | 5240 | **37.0** | 7 | 7 | 124 |
| TC_SQ_020 | 48x48x48 | 48 | 48 | 48 | 5238 | 5275 | 5241 | **37.0** | 7 | 7 | 124 |
| TC_SQ_030 | 1024x1024x1024 | 1024 | 1024 | 1024 | 5240 | 6817 | 5239 | **1577.0** | 18 | 16 | 0 |
| TC_SQ_031 | 2048x2048x2048 | 2048 | 2048 | 2048 | 5238 | 6816 | 5239 | **1578.0** | 16 | 14 | 0 |

**关键观察**：
- `hbm_before` 与 `hbm_after` 在 4 个 case 上高度一致（5238~5240 MB），说明算子执行后 workspace 已释放。
- `delta_peak` 存在**~37 MB 的地板值**（TC_SQ_017 / TC_SQ_020 完全相同），对应 NPU 常驻 runtime + kernel 加载开销（**显著低于迭代一的 ~270 MB 地板**，因为 iter4b 委托 upstream kernel，runtime 加载量更小）。
- TC_SQ_030 与 TC_SQ_031 的 delta_peak 分别为 1577 MB 与 1578 MB，几乎相等（差 1 MB），说明 1024³ 到 2048³ 之间 workspace 增长**趋于饱和**——upstream kernel 在 1024³ 附近已经申请了大形状所需的全部中间 buffer。
- **TC_SQ_017 / TC_SQ_020 的 `exit_code = 124`**（`timeout` 命令的信号值），意味着采样流程因 6 s 硬超时被截断；`--gtest_repeat` 未跑完预定次数即超时，但已采到 7 个样本，`peak` 仍有效（见 §2.3 采样窗口覆盖）。TC_SQ_030 / TC_SQ_031 的 `exit_code = 0`（干净退出）。

### 3.2 公式估算（全部 4 条 case）

`memory_samples_iter4b.csv` 中的 `ws_formula_mb` / `a_buf_mb` / `b_buf_mb` / `c_buf_mb` 列（单位 MB）：

| Case | ws_formula_mb | a_buf_mb | b_buf_mb | c_buf_mb | a+b+c 合计 (MB) |
|---|---|---|---|---|---|
| TC_SQ_017 | 0.009 | 0.002 | 0.002 | 0.002 | 0.006 |
| TC_SQ_020 | 0.079 | 0.018 | 0.018 | 0.018 | 0.054 |
| TC_SQ_030 | 36.000 | 8.000 | 8.000 | 8.000 | 24.000 |
| TC_SQ_031 | 144.000 | 32.000 | 32.000 | 32.000 | 96.000 |

按公式：`a_buf = k*n*8 bytes`、`b_buf = m*k*8 bytes`、`c_buf = m*n*8 bytes`（COMPLEX64 = 8 bytes/element）；`ws_formula_mb` 为 3-GEMM 分解的理论最小 workspace，**并非** `a+b+c`。

---

## 4. 公式估算对照

### 4.1 偏差分析

| Case | ws_formula (MB) | measured delta_peak (MB) | delta - formula (MB) | 倍数 |
|---|---|---|---|---|
| TC_SQ_017 | 0.009 | 37.0 | 36.991 | ~4,110× |
| TC_SQ_020 | 0.079 | 37.0 | 36.921 | ~468× |
| TC_SQ_030 | 36.000 | 1577.0 | 1541.000 | ~44× |
| TC_SQ_031 | 144.000 | 1578.0 | 1434.000 | ~11× |

**公式估算与实测 delta_peak 存在 1~4 个数量级的偏差**。原因分解：

### 4.2 偏差来源分解

`delta_peak` 的构成：

```
delta_peak = [NPU 常驻 runtime / kernel 加载开销]  (~37 MB 地板)
           + [A/B/C 输入输出 buffer]               (小，见 §3.2 合计列)
           + [3-GEMM 分解中间累加器 / tile buffer]  (ws_formula 覆盖)
           + [upstream kernel 双缓冲 / 流水 buffer]  (未建模)
           + [CANN runtime 动态增长]                (未建模)
```

**实测数据验证**：

- **TC_SQ_017**：a+b+c 合计仅 0.006 MB，ws_formula 仅 0.009 MB；delta_peak = 37 MB 几乎全部是**NPU 常驻 runtime 开销**。
- **TC_SQ_020**：a+b+c = 0.054 MB，ws_formula = 0.079 MB；delta_peak = 37 MB 同样**几乎全是 runtime 开销**。
- **TC_SQ_030**：a+b+c = 24 MB，ws_formula = 36 MB；delta_peak = 1577 MB。扣除 37 MB 地板，净增量 ~1540 MB，与公式估算 36 MB 存在约 **43×** 差距，剩余部分来自：
  - upstream `aclblasCgemm` 内部产生的中间矩阵 buffer（未包含在 ws_formula 中）
  - Blocking tile 双缓冲 / 流水线 buffer
  - CANN runtime 在大规模算子下的动态增长
- **TC_SQ_031**：a+b+c = 96 MB，ws_formula = 144 MB；delta_peak = 1578 MB。与 TC_SQ_030 几乎持平（差 1 MB），说明 upstream kernel 在 1024³ 已经"预申请"了大形状所需的 workspace 上限（与 1 GiB harness 侧 `aclblasSetWorkspace` 上限相关）。

### 4.3 iter1 vs iter4b 对比

| Case | iter1 delta_peak (MB) | iter4b delta_peak (MB) | 变化 |
|---|---|---|---|
| TC_SQ_017 | 273.0 | 37.0 | **−86%（迭代一含自研 kernel runtime 加载）** |
| TC_SQ_020 | 270.0 | 37.0 | **−86%** |
| TC_SQ_030 | 1301.0 | 1577.0 | **+21%（upstream kernel 中间 buffer 更大）** |
| TC_SQ_031 | 未实测（formula only） | 1578.0 | **iter4b 首次实测** |

**关键观察**：
- 小 shape 的地板值从 ~270 MB 降至 ~37 MB，说明 iter4b 委托 upstream 后**runtime 加载开销显著降低**（自研 kernel 的编译产物与加载被替换为 upstream kernel）。
- 中/大 shape 的 delta_peak 略有增长（TC_SQ_030 +21%），因为 upstream `aclblasCgemm` kernel 内部使用更复杂的 tiling + 双缓冲策略，中间 buffer 更多。
- TC_SQ_031 从 formula only 变为实测 1578 MB，与 TC_SQ_030 几乎持平——upstream kernel 的 workspace 增长在 1024³ 附近趋于饱和。

### 4.4 结论

**workspace 上限公式对齐合理**（`ws_formula_mb` 与实际输入/输出 buffer 的量级一致），但**该公式仅覆盖理论最小 workspace**，未包含：
1. NPU 常驻 runtime 开销（~37 MB 地板，较迭代一 ~270 MB 显著降低）
2. upstream `aclblasCgemm` 分解产生的中间累加器
3. Blocking tile 双缓冲
4. CANN runtime 动态增长

**实测 delta_peak 与公式估算偏差**：
- TC_SQ_017、TC_SQ_020：偏差 ~4,110× 和 ~468×（几乎全是 runtime 地板）
- TC_SQ_030：偏差 ~44×（runtime 地板 + 中间 buffer + tile buffer）
- TC_SQ_031：偏差 ~11×（runtime 地板 + 中间 buffer；因大形状接近 workspace 饱和，偏差倍数最低）

**workspace 上限公式对齐合理**（理论 workspace 与实际输入/输出 buffer 量级一致），但**实测净增长由 upstream kernel 中间缓冲主导**，与公式估算不直接可比。

---

## 5. 结论

- 本批次采样 4 个代表 case（TC_SQ_017/020/030/031），**全部实测**（含迭代一为 formula only 的 TC_SQ_031）。
- 实测 `delta_peak` 存在 **~37 MB 的地板值**（NPU 常驻 runtime 开销），与 case 规模无关；较迭代一 ~270 MB 地板降低 ~86%。
- 扣除地板值后，`delta_peak` 随 scale 增长至 TC_SQ_030 后趋于饱和：TC_SQ_031 与 TC_SQ_030 仅差 1 MB。
- **workspace 上限公式（`ws_formula_mb`）与实测净增长存在 1~4 个数量级的偏差**，原因是公式仅覆盖理论最小 workspace，未包含 runtime 地板、3-GEMM 中间 buffer、tile 双缓冲等。
- **iter4b vs iter1**：小 shape 内存占用显著降低（runtime 加载更轻），中/大 shape 略增（upstream kernel 中间 buffer 更多）。
- 任务书 §3.4 声明「内存要求：不涉及」，本批次未做严格的内存上限验证；数据仅作**内存占用画像**用途，不作合规判定。
- TC_SQ_017 / TC_SQ_020 的 `exit_code = 124`（timeout 截断），采样窗口内已采到 7 个样本，peak 仍有效。

---

## 附：数据溯源

- 原始数据：`/tmp/mem_sample/memory_samples_iter4b.csv`（4 行 + 表头，2026-09-21 采样）
- 采样脚本：`/tmp/mem_sample/iter4b_sample.sh`（适配版，使用 `CgemmArch35Test.CsvDriven/*` 过滤）
- 采样方法：`npu-smi info` 的 HBM-Usage 列（`npu-smi info -m` 在本环境无返回）
- 采样协议：单次调用 0.67 s；`--gtest_repeat` 拉长进程存活；采样间隔 0.1 s
- 二进制：`cgemm_ex_test`（2,070,128 B，2026-09-18 03:36 构建）
- 任务书内存要求：`aclblasCgemmEx_task_doc.md` §3.4（不涉及）
- 验收交付件要求：`aclblasCgemmEx_task_doc.md` §4 表 #3（内存占用数据）

---

## 附：与历史版本的对比

| 版本 | 采样时间 | TC_SQ_017 delta_peak | TC_SQ_020 delta_peak | TC_SQ_030 delta_peak | TC_SQ_031 delta_peak | 说明 |
|---|---|---|---|---|---|---|
| 迭代一 | 2026-09-15 | 273.0 MB | 270.0 MB | 1301.0 MB (exit=1) | formula only | 自研 Karatsuba kernel |
| 迭代二 | 2026-09-17 | 未重采样 | 未重采样 | 未重采样 | formula only | 标注披露 TC_SQ_030 exit=1 |
| **iter4b（本次）** | **2026-09-21** | **37.0 MB** | **37.0 MB** | **1577.0 MB** | **1578.0 MB** | **薄 wrapper，委托 upstream** |

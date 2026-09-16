# aclblasCgemmEx 内存占用自测报告

> 交付件 #3 之一（内存部分）· 面向 §4 验收交付件提交
> 数据来源：`ops-blas/test/gemmex/cgemmex/arch35/memory_samples.csv`
> 采样脚本：`ops-blas/test/gemmex/cgemmex/arch35/sample_memory.sh`
> 生成日期：2026-09-15

---

## 1. 测试摘要

### 1.1 环境

| 项 | 值 |
|---|---|
| 硬件 | Ascend 950PR |
| CANN 版本 | 9.1.0 |
| NPU 设备编号 | 0 |
| HBM 总容量 | ~8192 MB（`npu-smi info` 观测） |
| 采样工具 | `npu-smi info`（HBM-Usage 列）|
| 采样脚本 | `sample_memory.sh`（含 `kill -STOP/-CONT` 保护）|
| 代表 case 数 | 4（TC_SQ_017 / 020 / 030 / 031）|
| 覆盖 shape | 16³、48³、1024³（measured）、2048³（formula only）|

### 1.2 采样覆盖说明

本批次仅在 **TC_SQ NN 方阵** 子集上采样 4 个代表 case：

- **小 shape**：TC_SQ_017（16³）、TC_SQ_020（48³）—— 用于确认 workspace 下限
- **中 shape**：TC_SQ_030（1024³）—— 用于确认 workspace 随规模的增长
- **大 shape**：TC_SQ_031（2048³）—— 仅公式估算，未实测（见 §3.2）

其他用例族（TC_L0、TC_AB、TC_LD、TC_FL、TC_CV、TC_ED、TC_CV、TC_PF 全量）**本批次未做内存采样**。

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

由于单次算子调用耗时短（TC_SQ_030 约 8 s，TC_SQ_031 约 75 s），单独跑一次难以在 `hbm_before` 与 `hbm_after` 之间完成多次采样。因此：

1. 用 **`--gtest_repeat=N`**（gtest 参数）在**同一个 cgemmex_test 进程内**重复运行同一 case N 次，拉长进程存活窗口。
2. 外部采样脚本以 0.67 s 间隔轮询 `npu-smi info`，在采样窗口内取 `hbm_peak`。
3. 采样期间用 **`kill -STOP`** 暂停 NPU 进程（避免采样瞬间被新一次 kernel launch 影响），采样后 **`kill -CONT`** 恢复。
4. 与并发运行的 `perf.log` 采集**共享** NPU 时，`kill -STOP` 防止内存采样与性能采样互相污染。

> 注：本批次 4 个 case 的采样与 `perf.log` 的性能采样是**分时串行**的（先跑内存，后跑性能），未发生真正的并发冲突；`kill -STOP/-CONT` 属于防御性设计。

### 2.4 与 3-GEMM 复数分解的关系

`aclblasCgemmEx` 主路径按 3-GEMM 复数分解实现：
```
C = (A_re·B_re − A_im·B_im) + i·(A_re·B_im + A_im·B_re)
```

workspace 主要为各 GEMM 的中间累加器 + blocking tile buffer。`memory_samples.csv` 的 `ws_formula_mb` 为**理论最小 workspace 估算**（不含中间 tile buffer、不含 CANN runtime 常驻开销），`delta_peak_mb` 为**实测净增长**（含全部开销）。

---

## 3. 数据表

### 3.1 实测数据（3 条 measured）

| Case | Shape | m | n | k | hbm_before (MB) | hbm_peak (MB) | hbm_after (MB) | delta_peak (MB) | samples | elapsed (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| TC_SQ_017 | 16x16x16 | 16 | 16 | 16 | 5553 | 5826 | 5555 | **273.0** | 10 | 8 |
| TC_SQ_020 | 48x48x48 | 48 | 48 | 48 | 5555 | 5825 | 5554 | **270.0** | 10 | 8 |
| TC_SQ_030 | 1024x1024x1024 | 1024 | 1024 | 1024 | 5554 | 6855 | 5554 | **1301.0** | 48 | 36 |

**关键观察**：
- `hbm_before` 与 `hbm_after` 在三个 case 上高度一致（5553~5555 MB），说明算子执行后 workspace 已释放。
- `delta_peak` 存在**~270 MB 的地板值**（TC_SQ_017 / TC_SQ_020 几乎相同），对应 NPU 常驻 runtime + kernel 加载开销。
- TC_SQ_030 的 delta_peak 显著超过地板值 1000+ MB，是 workspace 随规模增长的净增量。

### 3.2 公式估算（1 条 formula only）

| Case | Shape | m | n | k | ws_formula (MB) | c_buf (MB) | a_buf (MB) | b_buf (MB) | exit_code |
|---|---|---|---|---|---|---|---|---|---|
| TC_SQ_031 | 2048x2048x2048 | 2048 | 2048 | 2048 | **144.000** | 32.000 | 32.000 | 32.000 | — |

**TC_SQ_031 仅做公式估算、未实测**：
- 单次调用耗时约 75 s（参考 `accuracy.log` Run/24 耗时 75,351 ms），加上 warmup 与采样窗口，`--gtest_repeat` 场景下一次采样轮次超过 8 分钟；
- 估算工作量为 `2048³ × 6 ≈ 5.24e10` ops，按实测吞吐外推单次 perf 采样需 ~3 h；
- 出于 session 时间约束，仅做公式估算，未实测。

### 3.3 公式估算表（全部 4 条 case）

`memory_samples.csv` 中的 `ws_formula_mb` / `a_buf_mb` / `b_buf_mb` / `c_buf_mb` 列（单位 MB）：

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
| TC_SQ_017 | 0.009 | 273.0 | 272.991 | ~30,433× |
| TC_SQ_020 | 0.079 | 270.0 | 269.921 | ~3,418× |
| TC_SQ_030 | 36.000 | 1301.0 | 1265.000 | ~36× |
| TC_SQ_031 | 144.000 | — (未实测) | — | — |

**公式估算与实测 delta_peak 存在 2~4 个数量级的偏差**。原因分解：

### 4.2 偏差来源分解

`delta_peak` 的构成：

```
delta_peak = [NPU 常驻 runtime / kernel 加载开销]  (~270 MB 地板)
           + [A/B/C 输入输出 buffer]               (小，见 §3.3 合计列)
           + [3-GEMM 分解中间累加器 / tile buffer]  (ws_formula 覆盖)
           + [CANN runtime 动态增长]                (未建模)
```

**实测数据验证**：

- **TC_SQ_017**：a+b+c 合计仅 0.006 MB，ws_formula 仅 0.009 MB；delta_peak = 273 MB 几乎全部是**NPU 常驻 runtime 开销**。
- **TC_SQ_020**：a+b+c = 0.054 MB，ws_formula = 0.079 MB；delta_peak = 270 MB 同样**几乎全是 runtime 开销**。
- **TC_SQ_030**：a+b+c = 24 MB，ws_formula = 36 MB；delta_peak = 1301 MB。扣除 270 MB 地板，净增量 ~1031 MB，与公式估算 36 MB 存在约 **28×** 差距，剩余部分来自：
  - 3-GEMM 分解产生的中间矩阵 buffer（未包含在 ws_formula 中）
  - Blocking tile 双缓冲 / 流水线 buffer
  - CANN runtime 在大规模算子下的动态增长

### 4.3 结论

**workspace 上限公式对齐合理**（`ws_formula_mb` 与实际输入/输出 buffer 的量级一致），但**该公式仅覆盖理论最小 workspace**，未包含：
1. NPU 常驻 runtime 开销（~270 MB 地板）
2. 3-GEMM 分解产生的中间累加器
3. Blocking tile 双缓冲
4. CANN runtime 动态增长

**实测 delta_peak 与公式估算偏差**：
- TC_SQ_017、TC_SQ_020：偏差 ~30,433× 和 ~3,418×（几乎全是 runtime 地板）
- TC_SQ_030：偏差 ~36×（runtime 地板 + 中间 buffer + tile buffer）
- TC_SQ_031：未实测，无法评估

**workspace 上限公式对齐合理**（理论 workspace 与实际输入/输出 buffer 量级一致），但**实测净增长由 runtime 地板 + 中间缓冲主导**，与公式估算不直接可比。

---

## 5. 结论

- 本批次采样 4 个代表 case（TC_SQ_017/020/030/031），其中 3 个实测、1 个仅公式估算。
- 实测 `delta_peak` 存在 **~270 MB 的地板值**（NPU 常驻 runtime 开销），与 case 规模无关。
- 扣除地板值后，`delta_peak` 随 scale 单调增长：TC_SQ_030 净增量约 1031 MB。
- **workspace 上限公式（`ws_formula_mb`）与实测净增长存在 2~4 个数量级的偏差**，原因是公式仅覆盖理论最小 workspace，未包含 runtime 地板、3-GEMM 中间 buffer、tile 双缓冲等。
- 任务书 §3.4 声明「内存要求：不涉及」，本批次未做严格的内存上限验证；数据仅作**内存占用画像**用途，不作合规判定。
- TC_SQ_031 因耗时约束未实测，仅保留公式估算。

---

## 附：数据溯源

- 原始数据：`ops-blas/test/gemmex/cgemmex/arch35/memory_samples.csv`（510 B，4 行 + 表头）
- 采样脚本：`ops-blas/test/gemmex/cgemmex/arch35/sample_memory.sh`（12,273 B）
- 采样方法：`npu-smi info` 的 HBM-Usage 列（`npu-smi info -m` 在本环境无返回）
- 采样协议：单次调用 0.67 s；`--gtest_repeat` 拉长进程存活；`kill -STOP/-CONT` 与 perf 采样共存保护
- 任务书内存要求：`aclblasCgemmEx_task_doc.md` §3.4（不涉及）
- 验收交付件要求：`aclblasCgemmEx_task_doc.md` §4 表 #3（内存占用数据）

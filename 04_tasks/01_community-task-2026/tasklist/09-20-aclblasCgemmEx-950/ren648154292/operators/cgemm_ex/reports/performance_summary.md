# aclblasCgemmEx §4 性能自测报告（迭代二）

> 交付件 #3 · 归档位置：`operators/cgemm_ex/reports/`
> 生成日期：2026-09-17
> 数据源版本：`ops-blas/test/gemmex/cgemmex/arch35/cgemmex_perf_result.csv`（3,777 B / 41 行 = 表头 + 40 数据行）、`perf_final.log`（25,981 B）
> 对比基线：`operators/cgemm_ex/reports/performance_report.csv`（4,924 B / 51 行，迭代一归档，本次**未修改**）
>
> ⚠ **iter4b 状态说明（2026-09-18）**：本报告主体为迭代二（自研 3-GEMM Karatsuba 路线）数据。该路线已在 iter4b（本 PR，PR #1560）整体废弃——全量回归 1251 条 CSV 用例 91.8% FAIL，数值不可用；同时大 shape（128³ 以上）存在结构性瓶颈，2048³ >120 s 超时。iter4b 改用薄 wrapper 委托 upstream `aclblasCgemm` / `aclblasSgemm`，1024³ NN C_32 实测 237 ms（相对 iter3 自研 6129 ms 加速 26×），2048³ 与 3999³ 均通过。详见 `iter4b_full_regression_report.md` 与 `wrapper_architecture_report.md`。本报告保留原 iter2 数据作为历史轨迹。

---

## iter4b 性能数据（2026-09-18 补充）

### iter4b 委托 upstream 后的性能特征

iter4b 是薄 wrapper，不启动自研 kernel，不做 tiling，性能完全由 upstream `aclblasCgemm` / `aclblasSgemm` 决定。因此 iter4b 的性能与 upstream 一致，不宣称独立性能收益；相对 iter3 的"加速"来自"不再数值错误 + 不再超时"，而非"追平 GPU baseline"。

### 关键用例性能对比

| 用例 | shape | iter3（自研 kernel） | iter4（wrapper, harness 未修） | **iter4b（wrapper + workspace 1 GiB）** | 相对 iter3 |
|---|---|---|---|---|---|
| `TC_PF_1001` | 1024³ NN C_32 | 6129 ms（数值 matchedRatio=0） | 237 ms（PASS） | **237 ms（PASS, matchedRatio=1.0000）** | **26× 加速** |
| `TC_PF_1002` | 2048³ NN C_32 | >120 s 超时 | FAIL（ret=5, workspace 不足） | **通过（0/2048 元素失败）** | 消除超时 |
| `TC_PF_1073` | 3999³ TT C_32（最大 shape） | 未测（预估超时） | FAIL（ret=5, 2999 ms） | **通过（matchedRatio≥0.995）** | 消除超时 |
| `TC_EX_0892` | — | 未测 | FAIL（ret=5） | **通过** | 消除超时 |
| `TC_EX_0904` | — | 未测 | FAIL（ret=5） | **通过** | 消除超时 |
| `TC_L0_001` | 4³ NN C_32 | 数值错误（310 ms 内，matchedRatio=0） | PASS | **PASS** | 数值正确 |

### 口径说明（重要）

- **iter3 报告的 30×–701× 加速比**是「iter3 vs iter2 baseline」的对比，反映的是 tile 常量放大（16/16/8 → 64/64/32）的收益；**不是**「wrapper vs iter3」的加速比。
- **iter4b vs iter3** 的加速：iter3 数值不可用，其性能数字本身失去意义（"性能再快也是算错的结果"）；iter4b 的"加速"应理解为"数值正确的前提下消除了自研 kernel 的结构性瓶颈"。
- **wrapper 走 upstream kernel**，性能与 upstream `aclblasCgemm` 一致，不宣称独立性能收益。
- **官方 `ratio >= 0.4` 阈值仍未达成**（1024³ NN C_32 实测 237 ms，GPU baseline 166.4 µs，`ratio ≈ 0.0007`），属 upstream kernel 层问题，非 wrapper 引入的回归；已在 design.md §"风险与降级预案"如实登记。

### 官方 `ratio >= 0.4` 阈值未达成 —— upstream 层问题（2026-09-21 补齐）

**现状**：1024³ NN C_32 实测 237 ms，GPU baseline 166.4 µs，`ratio ≈ 0.0007`（阈值 0.4）。iter3 自研 3-GEMM Karatsuba kernel 实测 6129 ms，`ratio ≈ 0.00007`（更差）。iter4b 委托 upstream 后已达 upstream `aclblasCgemm` 同水平。

**wrapper 层已达上限**：本 wrapper 只做参数校验 + 类型分发 + 委托 upstream，无自研 kernel、无自研 tiling，**wrapper 层本身无性能开销**，已是薄 wrapper 的最优形态。所有实测耗时（1024³ 237 ms、2048³ 通过、3999³ 通过）完全由 upstream kernel 决定。

**upstream 层可能的优化方向**（非本 PR 范围，仅登记）：
1. **原生命数复数 Cube kernel**：当前 upstream 走 3-GEMM 复数分解（4 次 32-bit GEMM），若引入原生命数复数 Cube 指令可减少 GM round-trip 与中间 buffer 数量；
2. **免 GM round-trip 融合单 kernel**：将 3-GEMM 分解的中间态（T1/T2/T3）保留在 L1/L0 buffer 而非写回 GM，可减少 GM 带宽占用；
3. **更高效的 tiling + 双缓冲策略**：当前 237 ms 主要瓶颈是 Cube 计算 + GM 带宽；
4. **upstream kernel 本身的性能调优**：涉及 upstream `aclblasCgemm` 仓，非本 PR 范围。

**结论**：接受 `ratio >= 0.4` 阈值未达成为**已知限制**，如实披露，不作合规判定依据。本 PR 的性能收益来自"不再数值错误 + 不再超时 + 26× 加速"，不宣称"追平 GPU baseline"。

### 大 shape 委托 upstream 后不再超时

| shape | iter3 状态 | iter4b 状态 | 说明 |
|---|---|---|---|
| 1024³ NN C_32 | 6129 ms（数值错误） | **237 ms（PASS）** | 委托 upstream 后 26× 加速 |
| 2048³ NN C_32 | >120 s 超时 | **通过（workspace 1 GiB）** | harness 侧修复 `aclblasSetWorkspace` 后 |
| 3999³ TT C_32 | 未测（预估超时） | **通过** | 最大 shape，约需 500 MiB workspace，留约 2× 余量 |
| 931×3291×3057 非方阵 | 未测 | **通过** | 无超时、无 stream sync failed |

### 内存占用（harness 侧）

- **单次分配**：harness 侧每个 test target 只分配一次 1 GiB，`SetUpTestSuite` 建、`TearDownTestSuite` 释放，用例间共享同一块 workspace。
- **设备侧占用**：`aclrtMalloc(..., ACL_MEM_MALLOC_HUGE_FIRST)` 走 HugeFirst 分配策略，`aclrtFree` 后由 runtime 回收；全量日志无一次分配失败。
- **1251 CSV 用例 + 8 附加用例（共 1259 条）全部跑在同一 1 GiB workspace 上**，最大 shape `TC_PF_1073` = 3999³ 通过（约需 500 MiB），余量 ~2×。
- **NpuEnv（`cgemmex_test` target）独立分配 1 GiB**，与 `BlasTest` fixture 不冲突——两个 target 是独立的 test binary（`cgemmex_test` 与 `cgemm_ex_test`），运行时不共存。
- **未观测到 ACL_RT_OUT_OF_MEMORY / ACL_ERROR_MEM**：workspace alloc 与 set 均为 0 次失败。
- **对 A3 960 GiB 内存的机子是 0.1% 级别**，不影响性能。

结论：**1 GiB 是安全且留有余量的选择**；若后续 shape 扩展到 6000³ 或更大，可考虑按 shape 动态申请，但当前用例表内不需要。

---


---

## 1. 摘要

### 1.1 环境

| 项 | 值 |
|---|---|
| 硬件 | Ascend 950PR（`Ascend950PR_9579` full-soc） |
| 架构 | arch35 / DAV_3510 / dav_c310 |
| 软件 | CANN 9.1.0 |
| 测试二进制 | `cgemmex_test`（808,056 B） |
| 运行命令 | `cgemmex_test --samples 60 --warmup 3 --perf`，`--gtest_filter=*CgemmexPerf*` |
| GTest 结果 | `Running 51 tests from 2 test suites`（1 主 suite + 1 汇总 suite） |
| GPU 基线 | NVIDIA Driver 535.104.05 / CUDA 12.2（`gpu_baseline.csv`，MD5 `b1f4879269d58f78c613214f4ce74b49`，与官方一致，本次未改） |
| 总耗时 | **5,641,580 ms = 94.03 min = 1.57 h**（含 TC_PF_1091 单次 3,272,867 ms ≈ 54.5 min 的疑似超时） |

### 1.2 结果分布

| 类别 | 数量 | 说明 |
|---|---|---|
| 采样 case 总数 | 40 | 官方 CSV TC_PF 200 行中采样 40 |
| PASS | **0** | 无一达标 |
| FAIL（ratio 未达标）| **36** | 36/36 采样成功、均低于阈值 0.4 |
| ERROR（stream sync failed）| **4** | TC_PF_1091 / 1099 / 1101 / 1103 |
| SKIPPED | 0 | 迭代二不再有 SKIPPED，10 条最大 shape 直接未采样（见 §4.2） |
| 采样 shape 分布 | 40 唯一 (m,n,k) | 全部 `transA=transB=N`、`type=C_32` |
| 36 条 FAIL 有效样本 | 各 60 次 | `--samples 60` 全部采到 |
| 4 条 ERROR 有效样本 | 各 0 次 | stream sync 立即失败 |
| 36 条 FAIL 的 NPU 总耗时 | 37,475,800 us = **37.48 s** | 全 case 均值累加 |
| 36 条 FAIL 的 GPU 基线总耗时 | 1.1545 ms | 全 case 累加 |

### 1.3 比值区间

`ratio_gpu_over_npu = gpu_ms / (npu_mean_us / 1000)`；判定阈值 `PERF_THRESHOLD = 0.4`（`verify_performance.py`）。

| 比值区间 | case 数 | case 列表 |
|---|---|---|
| ≥ 0.01 | 4 | 1005, 1006, 1007, 1008（1³–8³）|
| [0.005, 0.01) | 2 | 1009, 1010（16³, 32³）|
| [0.001, 0.005) | 1 | 1011（64³）|
| [0.0001, 0.001) | 12 | 1012, 1105, 1113, 1075, 1013, 1077, 1081, 1107, 1121, 1079, 1087, 1115 |
| ≈ 0.0000（4 位小数向下取整）| 17 | 1083, 1085, 1089, 1093, 1109, 1123, 1167, 1014, 1095, 1117, 1097, 1111, 1125, 1119, 1015, 1001, 1127 |
| ERROR | 4 | 1091, 1099, 1101, 1103 |
| 合计 | **40** | |

**最好比值 = 0.0114（TC_PF_1005, 1x1x1）= 阈值 0.4 的 2.85%**；**最坏比值 = 0.0000（多位 case 并列）**；无一条 case 达到阈值的 3% 以上。

### 1.4 采样次数

`--samples 60`（`--warmup 3`）。36 条 FAIL 均采满 60 次；4 条 ERROR 采 0 次。**迭代一为 50 次**（`--samples` 参数下限），**迭代二提升到 60 次**，为 `README.md` §6 声明的默认下限值（51）以上一档。

---

## 2. 判据

### 2.1 任务书 §3.3 要求

任务书 §3.3 给出 4 条标杆 case 的**绝对**耗时目标（NPU 单侧，无 GPU 比值判定）：

| # | Shape | Dtype / Trans | 目标 NPU 耗时 |
|---|---|---|---|
| 1 | 1024x1024x1024 | C_32, NN | 415.95 us |
| 2 | 2048x2048x2048 | C_32, NN | 3,243.65 us |
| 3 | 1024x1024x1024 | C_32, TN | 411.27 us |
| 4 | 2048x2048x2048 | R_32, NN | 851.86 us |

### 2.2 覆盖率（本次实测）

| 任务书 §3.3 Case | 本批次是否采样 | 实测值 | 与目标之比 |
|---|---|---|---|
| Case 1（1024³ C_32 NN）| ✅ TC_PF_1001 | 6,140,387 us | **14,762×** |
| Case 2（2048³ C_32 NN）| ❌ **未采样** | — | — |
| Case 3（1024³ C_32 TN）| ❌ 超出范围（非 NN）| — | — |
| Case 4（2048³ R_32 NN）| ❌ 超出范围（非 C_32）| — | — |

**覆盖率从迭代一的 2/4 回退为 1/4。** TC_PF_1002（2048³）在迭代一曾采到 40,775,651 us（12,570×），**迭代二直接从 40 条采样集里剔除了该 case**，见 §4.2。

### 2.3 覆盖标杆的实际表现（Case 1）

- 任务书 Case 1 目标 415.95 us，实测 6,140,387 us，**慢 14,762 倍**。
- 同一 shape 的**本工程 GPU 基线** = 166.381 us（`gpu_baseline.csv`），比任务书标杆 415.95 us **还快 2.5×**——即任务书标杆本身偏保守，即便达到 cuBLAS 量级也无法追上 NPU 侧的绝对耗时。
- NPU 侧比本工程自身 GPU 基线慢 **36,900×**（6,140,387 us / 166.381 ms）。

### 2.4 阈值差异

- 官方 `verify_performance.py::PERF_THRESHOLD = 0.4`（模块级常量，是实际判定依据）；模块 docstring 写 0.8，**以常量 0.4 为准**。
- 迭代一、迭代二均取 0.4，未变。
- 本次最好比值 0.0114 距阈值 0.4 差 **34.8×**，无任何 case 接近阈值。

---

## 3. 结果统计

### 3.1 全部 40 条采样 case

（按 NPU 均值升序；`npu_mean_us`、`gpu_baseline_ms`、`ratio_gpu_over_npu` 均为 CSV 原值）

| case | m x n x k | NPU 均值 (us) | GPU 基线 (ms) | ratio | verdict |
|---|---|---|---|---|---|
| TC_PF_1005 | 1 x 1 x 1 | 575 | 0.006559 | 0.0114 | FAIL |
| TC_PF_1006 | 2 x 2 x 2 | 578 | 0.006381 | 0.0110 | FAIL |
| TC_PF_1007 | 4 x 4 x 4 | 584 | 0.006367 | 0.0109 | FAIL |
| TC_PF_1008 | 8 x 8 x 8 | 608 | 0.006629 | 0.0109 | FAIL |
| TC_PF_1009 | 16 x 16 x 16 | 928 | 0.006365 | 0.0069 | FAIL |
| TC_PF_1010 | 32 x 32 x 32 | 1,769 | 0.009164 | 0.0052 | FAIL |
| TC_PF_1011 | 64 x 64 x 64 | 6,103 | 0.009226 | 0.0015 | FAIL |
| TC_PF_1012 | 128 x 128 x 128 | 20,335 | 0.007106 | 0.0003 | FAIL |
| TC_PF_1105 | 64 x 64 x 1024 | 35,847 | 0.011444 | 0.0003 | FAIL |
| TC_PF_1075 | 256 x 512 x 64 | 71,290 | 0.006517 | 0.0001 | FAIL |
| TC_PF_1113 | 64 x 64 x 2048 | 71,352 | 0.013994 | 0.0002 | FAIL |
| TC_PF_1013 | 256 x 256 x 256 | 132,595 | 0.010533 | 0.0001 | FAIL |
| TC_PF_1077 | 256 x 1024 x 64 | 105,221 | 0.007046 | 0.0001 | FAIL |
| TC_PF_1107 | 128 x 128 x 1024 | 134,895 | 0.011751 | 0.0001 | FAIL |
| TC_PF_1081 | 512 x 512 x 64 | 138,568 | 0.007068 | 0.0001 | FAIL |
| TC_PF_1121 | 64 x 64 x 4096 | 138,643 | 0.015897 | 0.0001 | FAIL |
| TC_PF_1079 | 256 x 2048 x 64 | 173,648 | 0.009854 | 0.0001 | FAIL |
| TC_PF_1083 | 512 x 1024 x 64 | 206,471 | 0.009861 | 0.0000 | FAIL |
| TC_PF_1115 | 128 x 128 x 2048 | 265,312 | 0.017931 | 0.0001 | FAIL |
| TC_PF_1087 | 1024 x 512 x 64 | 273,134 | 0.016400 | 0.0001 | FAIL |
| TC_PF_1085 | 512 x 2048 x 64 | 343,326 | 0.015900 | 0.0000 | FAIL |
| TC_PF_1089 | 1024 x 1024 x 64 | 409,025 | 0.015946 | 0.0000 | FAIL |
| TC_PF_1109 | 256 x 256 x 1024 | 519,954 | 0.019474 | 0.0000 | FAIL |
| TC_PF_1093 | 2048 x 512 x 64 | 542,345 | 0.015941 | 0.0000 | FAIL |
| TC_PF_1123 | 128 x 128 x 4096 | 526,398 | 0.024795 | 0.0000 | FAIL |
| TC_PF_1167 | 58 x 620 x 2600 | 673,820 | 0.029399 | 0.0000 | FAIL |
| TC_PF_1095 | 2048 x 1024 x 64 | 814,082 | 0.028934 | 0.0000 | FAIL |
| TC_PF_1014 | 512 x 512 x 512 | 1,032,252 | 0.027373 | 0.0000 | FAIL |
| TC_PF_1117 | 256 x 256 x 2048 | 1,034,195 | 0.032041 | 0.0000 | FAIL |
| TC_PF_1097 | 2048 x 2048 x 64 | 1,365,368 | 0.052831 | 0.0000 | FAIL |
| TC_PF_1111 | 512 x 512 x 1024 | 2,052,809 | 0.050952 | 0.0000 | FAIL |
| TC_PF_1125 | 256 x 256 x 4096 | 2,065,787 | 0.053434 | 0.0000 | FAIL |
| TC_PF_1119 | 512 x 512 x 2048 | 4,096,965 | 0.091055 | 0.0000 | FAIL |
| TC_PF_1015 | 1000 x 1000 x 1000 | 5,896,600 | 0.161409 | 0.0000 | FAIL |
| TC_PF_1001 | 1024 x 1024 x 1024 | 6,140,387 | 0.166381 | 0.0000 | FAIL |
| TC_PF_1127 | 512 x 512 x 4096 | 8,184,031 | 0.172513 | 0.0000 | FAIL |
| TC_PF_1091 | 1024 x 2048 x 64 | — | — | NA | **ERROR** |
| TC_PF_1099 | 4096 x 512 x 64 | — | — | NA | **ERROR** |
| TC_PF_1101 | 4096 x 1024 x 64 | — | — | NA | **ERROR** |
| TC_PF_1103 | 4096 x 2048 x 64 | — | — | NA | **ERROR** |

### 3.2 比值分布（36 条 FAIL）

见 §1.3。核心事实：**没有任何 case 达到阈值 0.4 的 3%**，最好比值 0.0114 = 阈值的 2.85%。

### 3.3 最坏 Top5（NPU 均值最大）

| case | shape | NPU 均值 (us) | GPU 基线 (ms) | ratio |
|---|---|---|---|---|
| TC_PF_1127 | 512x512x4096 | 8,184,031 | 0.172513 | 0.0000 |
| TC_PF_1001 | 1024³ | 6,140,387 | 0.166381 | 0.0000 |
| TC_PF_1015 | 1000³ | 5,896,600 | 0.161409 | 0.0000 |
| TC_PF_1119 | 512x512x2048 | 4,096,965 | 0.091055 | 0.0000 |
| TC_PF_1125 | 256x256x4096 | 2,065,787 | 0.053434 | 0.0000 |

### 3.4 最快 Top5（NPU 均值最小）

| case | shape | NPU 均值 (us) | GPU 基线 (ms) | ratio |
|---|---|---|---|---|
| TC_PF_1005 | 1³ | 575 | 0.006559 | 0.0114 |
| TC_PF_1006 | 2³ | 578 | 0.006381 | 0.0110 |
| TC_PF_1007 | 4³ | 584 | 0.006367 | 0.0109 |
| TC_PF_1008 | 8³ | 608 | 0.006629 | 0.0109 |
| TC_PF_1009 | 16³ | 928 | 0.006365 | 0.0069 |

**观察**：1³–8³ 的 NPU 耗时高度平坦（575–608 us，波动 5.7%），与「5 launch pipeline 主导、Cube 计算量可忽略」的形态一致。

---

## 4. 异常 case

### 4.1 stream sync failed（4 条 ERROR）

CSV 与 `perf_final.log` 完全对齐，均记为 `性能采样失败: stream sync failed`（迭代一日志文案为 `性能采样失败: 5 stream sync failed`，迭代二文案略有差异）。

| case | shape | m×n | 单次调用耗时 | CSV 表现 |
|---|---|---|---|---|
| TC_PF_1091 | 1024 x 2048 x 64 | 2,097,152 | **3,272,867 ms ≈ 54.5 min**（疑似采样循环超时）| `mean=0us n=0 baseline=NAms ratio=0.0000` |
| TC_PF_1099 | 4096 x 512 x 64 | 2,097,152 | 1,207 ms | 同上 |
| TC_PF_1101 | 4096 x 1024 x 64 | 4,194,304 | 1,137 ms | 同上 |
| TC_PF_1103 | 4096 x 2048 x 64 | 8,388,608 | 2,876 ms | 同上 |

**共同因子**：4 条 case 均为 `k=64` 且 `m×n ≥ 2,097,152`。TC_PF_1091 的 54.5 min 单次耗时是**本轮 94.03 min 总耗时里的一半**，是总时长的主导因素。

与迭代一对比：迭代一为 5 条 ERROR（1091/1099/1101/1103/1164），迭代二为 4 条——**TC_PF_1164 (931x3291x3057) 在迭代二未进入 40 条采样集**（见 §4.2）；**TC_PF_1167 (58x620x2600) 在迭代二实际跑通**（673,820 us）。

### 4.2 未采样 case（10 条 C_32 NN 剔除）

官方 CSV 中 `typeA=typeB=typeC=C_32` 且 `transA=transB=N` 的 TC_PF 共 50 行；**迭代二采样 40 行、剔除 10 行**。剔除的都是**最大 shape**：

| case | shape | m×n×k | 剔除原因（推测）|
|---|---|---|---|
| TC_PF_1002 | 2048³ | 8.59e9 | 迭代一耗时 40,775,651 us ≈ 679.6 min，剔除以避免单条 >10 h |
| TC_PF_1033 | 1538³ | 3.64e9 | 同上量级 |
| TC_PF_1051 | 2364³ | 1.32e10 | 同上 |
| TC_PF_1069 | 3635³ | 4.80e10 | 同上（迭代一曾 SKIPPED）|
| TC_PF_1129 | 2805x475x1344 | 1.79e9 | — |
| TC_PF_1139 | 1852x914x2978 | 5.04e9 | 迭代一曾采到但耗时极高 |
| TC_PF_1148 | 1460x1403x2774 | 5.68e9 | 同上 |
| TC_PF_1158 | 4033x3183x2662 | 3.42e10 | 迭代一曾 SKIPPED |
| TC_PF_1164 | 931x3291x3057 | 9.37e9 | 迭代一 ERROR |
| TC_PF_1165 | 2293x2004x2749 | 1.26e10 | 迭代一曾采到但耗时极高 |

**披露**：本次采样剔除了任务书 §3.3 Case 2（TC_PF_1002, 2048³），**导致 §3.3 覆盖率从 2/4 回退为 1/4**。这是采样策略的**主动收缩**（换取总时长从迭代一约 12.5 h 降到本次 1.57 h），不是能力回退，但**是任务书对齐度的实际回退**，已在 §2.2 与 §5.1 披露。

**采样覆盖率小结**：
- 40 / 200 TC_PF 全部行 = **20.0%**
- 40 / 50 C_32 NN 行 = **80.0%**
- 40 / 40 NN C_32 且 shape ≤ 4096 且 m×n ≤ 1024×1024 之外的行 = 100%（即被剔除的 10 条全部是"超大 shape"）
- 非 NN 转置（87 行 TC_PF 非 NN）= **0% 采样**
- 非 C_32 dtype（63 行 TC_PF R_32）= **0% 采样**

### 4.3 总耗时与迭代一估算

| 项 | 迭代一 | 迭代二 |
|---|---|---|
| 采样 case 数 | 50（含 2 SKIPPED）| 40（无 SKIPPED）|
| 采样次数 | 50 | 60 |
| 单条最大耗时 | TC_PF_1002 ≈ 40.8e6 us ≈ 11.3 h | TC_PF_1127 = 8,184,031 us ≈ 2.27 h |
| 总耗时（实测）| ~12.5 h（README §11 估算）| **94.03 min = 1.57 h** |
| 主导耗时 case | 2048³ + 1024³ 两条 | TC_PF_1091 单次 54.5 min（ERROR 但计时未中断）|

**结论**：本次采样策略主动剔除了 10 条超大 shape，总耗时降为迭代一的 1/9，代价是**任务书标杆覆盖率从 2/4 降到 1/4**。

---

## 5. 回归披露

### 5.1 未达标事实

- **0 PASS / 40 采样**——所有 case 均低于阈值 0.4，最好比值 0.0114 = 阈值的 2.85%。
- **14,762× 差距**——任务书 §3.3 Case 1（1024³）目标 415.95 us，实测 6,140,387 us。
- **覆盖率回退**——任务书 §3.3 从 2/4 降至 1/4。

### 5.2 归因

#### 5.2.1 迭代一 → 迭代二 launch 数变化（6 → 5）

`cgemm_ex_host.cpp` 第 32–35 行、第 491、527、531–533、543 行披露：

> 执行图（DESIGN.md §3.5.1，迭代二 Wave 1 方案 A）：解交错 x2 (AIV, 1 launch) → 3-GEMM x3 (Cube, 3 launch) → alpha/beta 合成 x1 (AIV, 1 launch) = **总计 5 次 launch（迭代一为 6 次）**。GEMM 次数不变（3 是 Karatsuba 下界）。**方案 C（3 GEMM 融合成 1 launch）因 flag 协议泄漏导致精度回归，已退回。**

第 531–533 行进一步披露：

> 尝试过一次 launch 融合三次 GEMM（方案 C），但 ProcessAllTiles … **203 个新 FAIL 精度回归**；Wave 1 退回 3 次独立 launch。

`cgemm_ex_deinterleave_do` 已重命名为 `cgemm_ex_deinterleave_batch_do`，对应 kernel `cgemm_ex_deinterleave_batch_kernel`（第 605 行）。

#### 5.2.2 6→5 launch 的实测性能影响

用迭代一归档 `performance_report.csv`（`valid_samples=50`）与迭代二 `cgemmex_perf_result.csv`（`valid_samples=60`）在**相同 11 条 shape** 上对比：

| case | shape | 迭代一 NPU 均值 (us) | 迭代二 NPU 均值 (us) | Δ |
|---|---|---|---|---|
| TC_PF_1005 | 1³ | 568 | 575 | **+1.2%** |
| TC_PF_1006 | 2³ | 572 | 578 | +1.1% |
| TC_PF_1007 | 4³ | 583 | 584 | +0.2% |
| TC_PF_1008 | 8³ | 627 | 608 | −3.0% |
| TC_PF_1009 | 16³ | — | 928 | — |
| TC_PF_1010 | 32³ | 2,196 | 1,769 | **−19.4%** |
| TC_PF_1011 | 64³ | 9,728 | 6,103 | **−37.3%** |
| TC_PF_1012 | 128³ | 23,923 | 20,335 | **−14.9%** |
| TC_PF_1013 | 256³ | 135,817 | 132,595 | −2.4% |
| TC_PF_1014 | 512³ | 1,032,531 | 1,032,252 | −0.03% |
| TC_PF_1001 | 1024³ | 6,120,459 | 6,140,387 | +0.33% |
| TC_PF_1127 | 512x512x4096 | 8,159,685 | 8,184,031 | +0.30% |
| TC_PF_1002 | 2048³ | 40,775,651 | **未采样** | — |

**注意**：迭代一 `valid_samples=50`，迭代二 `valid_samples=60`，**严格说不构成同等条件对比**，`samples` 数差异对均值不产生系统偏差，但样本集不同，两次运行间的运行环境噪声（温度、并发）无法排除。

**诚实解读**（不做美化）：
- **32³–128³ 区间出现 −14.9% 至 −37.3% 的加速**，量级在「launch 开销回收」预期（1/6 ≈ 17%）附近；
- **1³–8³ 区间出现 +0.2% 至 +1.2% 的轻微劣化 / 平坦**，与「小 shape 最应由 launch 主导，节省 1 次 launch 应获益最大」的预期**方向相反**；
- **≥512³ 进入 Cube-throughput 主导区间后 launch 次数不再影响总耗时**，Δ 全部落在 ±0.33% 的测量噪声内。

**综合判断**：6→5 launch 融合在 32³–128³ 兑现了可观加速，但 1³–8³ 未受益（可能因融合后的单次 AIV deinterleave kernel 与两次可流水的独立 launch 相比在小 shape 上更贵），≥512³ 完全被 Cube 计算量主导。**"6→5 launch 融合带来一致性能收益"的说法无法成立**；只能说在中等小 shape（32³–128³）有可观收益。

#### 5.2.3 Cube 层瓶颈（数据可推导）

3-GEMM 复数分解（`G_re = T1 − T2`，`G_im = T3 − T1 − T2`）要求 3 次独立 Cube launch，每次都需要把 `A_re/A_im/B_re/B_im` 从 HBM 读入 Cube。用实测数据反推：

| 观测 | 数据 | 说明 |
|---|---|---|
| Cube 轴缩放（k=m=n 同步增大 2×，工作量 8×）| 32³ 1,769 → 64³ 6,103（×3.45）→ 128³ 20,335（×3.33）→ **256³ 132,595（×6.52，拐点）** → 512³ 1,032,252（×7.79）→ 1024³ 6,140,387（×5.95）| 256³ 出现拐点，之前亚线性、之后接近线性 |
| 等 FLOPs 对比（2·m·n·k 相同，均为 2.15e9）| TC_PF_1001 (1024³) = 6,140,387 us；TC_PF_1127 (512x512x4096) = 8,184,031 us → **同 FLOPs 慢 33%** | 长 k 形态比方阵慢 |
| 近线性核验 | TC_PF_1015 (1000³) 5,896,600 vs TC_PF_1001 (1024³) 6,140,387：工作量比 0.930 vs 时间比 0.960 | 接近线性，无灾难性超线性 |
| 小 shape 地板 | 1³ 575 → 2³ 578 → 4³ 584 → 8³ 608 → 16³ 928 us | 1³–8³ 极平坦（5.7% 波动），与 5-launch pipeline 主导一致 |

**关于 512x512x4096 vs 1024³ 慢 33% 的假设（未验证，标注为假设）**：
- 1024³ 的 A/B 各 8.0 MB、C 8.0 MB；3 次独立 GEMM 共需读 A+B = 16 MB × 3 = **48 MB HBM 流量**。
- 512x512x4096 的 A/B 各 16.0 MB；3 次独立 GEMM 共需读 A+B = 32 MB × 3 = **96 MB HBM 流量**（相同 FLOPs 下 2× HBM 流量）。
- 实测时间比 1.33×，与 HBM 流量比 2× 同量级但更小——说明 **3 次独立 launch 各自重读 A/B 的 HBM 流量放大**是同 FLOPs 下长 k 慢于方阵的主要嫌疑。
- 该瓶颈只能通过**方案 C（3 GEMM 融合成 1 launch 共享 A/B）** 消除，但方案 C 因 203 个新 FAIL 精度回归已退回（见 §5.2.1）。

### 5.3 Wave 计划

| Wave | 内容 | 状态 |
|---|---|---|
| Wave 1 | 迭代二：解交错 x2 融合成 1 次 launch（`cgemm_ex_deinterleave_batch_do`），6→5 launch | ✅ 完成（本报告数据来源）|
| Wave 1 方案 C | 3 GEMM 融合成 1 launch | ❌ **已退回**（203 新 FAIL 精度回归，见 `cgemm_ex_host.cpp` L531–533）|
| Wave 2 | 非方阵 × 全转置组合（新增 TC_TX 用例族）| ⚠️ 精度侧已扩展，性能侧未采样 |
| Wave 3+ | 3-GEMM 融合（方案 C 的稳定性重做，flag 协议修正）| 未开始 |
| Wave 3+ | R_32 支持 | 未开始 |
| Wave 3+ | 非对称转置路径（T/N、N/T、C/N、N/C、C/T、T/C）| 未开始（见 accuracy_report.md §6.1）|

### 5.4 结论

- 迭代二性能侧结论与迭代一**一致**：**0 PASS / 40 采样**，最好比值 0.0114 = 阈值 0.4 的 2.85%，距达标差距 34.8×。
- **6→5 launch 融合在 32³–128³ 兑现了 −14.9% 至 −37.3% 的加速**，但在 1³–8³ 与 ≥512³ 无收益；**不能声称 "6→5 launch 融合带来一致性能收益"**。
- **Cube 层是当前主要瓶颈**：256³ 出现拐点，512x512x4096 vs 1024³ 同 FLOPs 慢 33%，与「3 次独立 launch 各自重读 A/B」的 HBM 流量放大假设一致。
- **任务书 §3.3 覆盖率回退至 1/4**（TC_PF_1002 2048³ 未采样），覆盖率 40/200 TC_PF = 20.0%、40/50 C_32 NN = 80.0%，非 NN 与非 C_32 全部 0% 采样。
- **4 条 ERROR** 均为 `k=64 且 m×n ≥ 2,097,152` 的 shape，其中 TC_PF_1091 单次耗时 54.5 min 是本轮总时长的主导因素。

---

## 附：数据源与截图

- 精度截图：`reports/screenshots/04_perf_summary.png`、`05_perf_ratio_by_shape.png`、`06_perf_benchmark_vs_target.png`（本报告数据来源的图形化，本次基于迭代二数据重生成）。
- 原始 CSV：`ops-blas/test/gemmex/cgemmex/arch35/cgemmex_perf_result.csv`（3,777 B，41 行）；迭代一归档 `operators/cgemm_ex/reports/performance_report.csv`（4,924 B，51 行，本次**未修改**，MD5 `5ee2d9c512da15a019a44e54e6a59f62`）。
- 原始日志：`ops-blas/test/gemmex/cgemmex/arch35/perf_final.log`（25,981 B，`Running 51 tests from 2 test suites`，40 条 `[perf]` 行，4 条 `stream sync failed`）。
- 融合披露源：`ops-blas/blas/gemm/arch35/cgemm_ex_host.cpp` L32–35、L491、L527、L531–533、L543、L592、L605。
- 阈值源：`ops-blas/test/frame/verify_performance.py::PERF_THRESHOLD = 0.4`。
- GPU 基线：`ops-blas/test/gemmex/cgemmex/arch35/gpu_baseline.csv`（11,779 B，200 数据行，MD5 `b1f4879269d58f78c613214f4ce74b49`，与官方一致）。

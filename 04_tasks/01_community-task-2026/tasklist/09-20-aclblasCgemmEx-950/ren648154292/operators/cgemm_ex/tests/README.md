# cgemm_ex ST 测试工程复现指南

面向验收工程师：本文档给出 `aclblasCgemmEx` 系统测试（ST）工程的**完整复现步骤**，
拿到代码 + Ascend 950PR 环境后可独立跑通，并复现本文档 §11 记录的当前结果。

> ⚠ **iter4b 重要前置（2026-09-18）**：从 iter4b（PR #1560，薄 wrapper 方案）起，harness **必须**调用
> `aclblasSetWorkspace(handle, 1 GiB)` 才能跑通大矩阵用例（maxdim ≥ 1000）。
> upstream `aclblasCgemm` 按矩阵规模线性分配 workspace，2048³ 需约 128 MiB，默认 32 MiB 不足会被拒（`ret=5 EXECUTION_FAILED`）；
> 最大 shape `TC_PF_1073` = 3999³ 约需 500 MiB，1 GiB 留约 2× 余量。
> 修复位置（两处，均在 ops-blas 仓 test/ 下）：
> - `test/frame/blas_test.h`：`BlasTest` fixture + `AclGuard`，`SetUpTestSuite` 中 `aclrtMalloc` + `aclblasSetWorkspace`，`TearDownTestSuite` 释放；
> - `test/gemm/cgemm_ex/arch35/cgemmex_run.h`：`NpuEnv`，`Init()` 中分配并绑定，`Destroy()` 释放。
> iter4b 全量 1259 用例 196 s 跑完，**1245 PASS / 14 FAIL（1.11%）**，其中 90 条大矩阵 EXECUTION_FAILED 全部转 PASS。
> 详细数据见 `reports/iter4b_full_regression_report.md`；本文档正文仍保留迭代二（自研 kernel）数据作历史轨迹。

> **前置阅读**：本工程的完整设计、判据推导、缺陷归因见
> `ops-blas/test/gemmex/cgemmex/README.md`（13 章）。本文档是面向复现的精简版，
> 两者冲突时以 arch35 目录下的 README.md 为准（它是测试工程的唯一权威说明）。

---

## 1. 概述与测试范围

**接口**：`aclblasCgemmEx`（`ops-blas/include/cann_ops_blas.h`），
参数序列与 `cublasCgemmEx` 一一对应（handle + 参数顺序完全一致，维数为 `int`）。

**迭代二范围（Wave 2 扩展，含手写扩展）**：

| 维度 | 本工程取值 |
|---|---|
| 数据类型 | **仅 `C_32`**（complex64），且 `typeA = typeB = typeC = C_32` |
| 转置 | **放开至全转置组合**（`N/N`、`T/T`、`C/C`、`T/N`、`N/T`、`C/T`、`T/C`、`N/C`、`C/N`），`kNonNNMaxDim = 512` 限制非 NN 转置的最大维度 |
| Shape 限制 | `kNonNNMaxDim = 512`（非 NN 转置的最大维度），NN 转置无额外维度限制 |
| 精度用例 | **348 条**（`kAccuracyFamilies = {TC_L0, TC_SQ, TC_AB, TC_CV, TC_LD, TC_FL, TC_ED, TC_TX}` 全部行，含 `TC_TX` 手写扩展族） |
| 性能用例 | **40 条**（`TC_PF` 族，`C_32/C_32/C_32` + `transA=transB=N`，剔除 10 条超大 shape） |

**迭代二新增（Wave 2 手写扩展）**：
- 新增 `TC_TX` 用例族（20 行）—— 覆盖「非方阵 × 全转置组合」，本批手写扩展（官方 CSV 无此族）。
- 放开 `TC_*` 精度用例的 `transA` / `transB` 限制（原来仅 `N/N`），中小 shape 上的 `T/T`、`C/C`、`T/N`、`N/T`、`C/T`、`T/C`、`N/C`、`C/N` 全部纳入精度覆盖。
- 负向用例 allowlist 从 5 项扩至 14 项（`kNegativeAllowlist`）。

**范围仍未做任何人为削减**：348 精度 + 40 性能是 `ApplyScopeFilter` 判定为 in-scope 的全部行，一条不省；阈值一律取官方 CSV 的列值，未做任何上调。

**诚实披露**：本轮实测 348 精度中 **110 PASS / 238 FAIL**（详见 §11 与 `reports/accuracy_report.md`），其中 238 FAIL 的绝大多数（203 条）为数值上不正确；性能侧 40 条 0 PASS，覆盖率较迭代一有回退（详见 §11.3）。

被测 kernel 位于 `ops-blas/blas/gemm/arch35/cgemm_ex_{host,kernel}.cpp`。
这些源文件**不在** `cmake/test.cmake` 的聚合测试目标里（`_ops_blas_has_blas_op_sources("cgemmex")`
的四种 glob 布局对 `blas/gemm/arch35/` 全部落空），所以本工程自带独立的
`run.sh`（直接 `g++` 编译）与独立的 `CMakeLists.txt`。

---

## 2. 测试用例清单

### 2.1 精度 case（348 条）

按用例族前缀分组，数量与覆盖场景如下（`bash run.sh --scope` 会打印逐条清单）：

| 用例族 | 官方/手写来源 | 官方总数 | 本范围覆盖 | 覆盖场景说明 |
|---|---|---|---|---|
| `TC_L0` 基线 | 官方 | 8 | **2** | `4x4x4`、`8x8x8`，最小形状冒烟（**迭代二实测仍未通过**，见 §11.1 D 类） |
| `TC_SQ` 方阵形状扫描 | 官方 | 414 | **107** | `1x1x1` … `2048x2048x2048`，`m=n=k` 全规模扫描，含 T/T、C/C 转置 |
| `TC_AB` 非方阵 | 官方 | 48 | **12** | 含 `RANDOM_NORM_5_5`、`VALUE_NORM_0` 填充组合 |
| `TC_CV` 常量填充 | 官方 | 144 | **12** | A/B/C 常量矩阵填充，含 T/C 转置组合 |
| `TC_LD` leading-dimension | 官方 | 24 | **4** | `ldc > n` 的 padding 场景 |
| `TC_FL` 填充/极值 | 官方 | 12 | **7** | 含 `RANDOM_EXTREME`、`VALUE_NORM_INF` |
| `TC_ED` 边界与负向 | 官方 | 31 | **84** | 零维、`k=0`、`alpha=(0,0)`、非法枚举、空指针等（Wave 2 放开转置后大幅扩展） |
| `TC_TX` 非方阵 × 全转置 | **本批手写扩展** | 0（官方无此族）| **20** | 20 条手写扩展用例，覆盖「非方阵 × 全转置组合」 |
| **合计** | | **681 + 20** | **348** | 全部按 `ApplyScopeFilter` 判定纳入（`kAccuracyFamilies` 白名单） |

补充说明：

- `TC_ED` 的 84 条里含**负向状态码断言**（`expect_result != ACLBLAS_STATUS_SUCCESS`），
  负向用例**不走数值比较**，只断言返回状态码等于 CSV 声明的 `expect_result`。
- `TC_TX` 是本批**手写扩展**用例族（官方 CSV 中不存在该前缀），由
  `ApplyScopeFilter` 的 `kAccuracyFamilies` 白名单纳入。
- 6 条退化通过（`TC_ED_699/700/701/702/705/706`，零维或 `k=0`），实际
  `valid = 0/0`、没有比较任何元素，**不应计为有效数值覆盖**。

### 2.2 性能 case（40 条）

| 用例族 | 官方总数 | 本范围覆盖 | 说明 |
|---|---|---|---|
| `TC_PF` 性能 | 200 | **40** | `C_32/C_32/C_32` + `transA=N, transB=N`，剔除 10 条超大 shape |

- 采样集：40 条来自官方 CSV 的 50 条 `C_32 NN` TC_PF 行；**剔除 10 条**最大 shape
  （`m*n*k ≥ 1.79e9`）以避免单条耗时超过小时级：TC_PF_1002/1033/1051/1069/1129/1139/1148/1158/1164/1165。
- 与 `gpu_baseline.csv` 的 join 命中率 **40/40**，0 条缺 baseline。
- 采样次数 `--samples 60`（下限校验 51，见 §6）。
- 任务书 §3.3 的 4 个标杆 case 中，**仅 case 1（`1024³` NN C_32）落入本次采样**（`TC_PF_1001`）；
  case 2（`2048³` NN C_32）在**本次被剔除**（TC_PF_1002），覆盖率从迭代一的 2/4 **回退**为 1/4。
  case 3（`1024³` **TN**）与 case 4（`2048³` NN **R_32**）因转置/R_32 不在本次范围，未覆盖。

### 2.3 明确未覆盖的用例族与原因

不做挑用例，以下是明确**不在本次范围**的内容：

| 未覆盖项 | 数量 | 原因 |
|---|---|---|
| `R_32` / `H_R_32` / `H_C_32` 及任何混合 dtype 组合 | 547 条（官方 CSV 中 `typeA=R_32`） | 未实现（迭代二 Wave 3+） |
| `TC_PF` 非 NN 转置性能采样 | 87 条 | 本轮采样策略仅采 NN（性能侧未做全转置扩展） |
| `TC_PF` R_32 性能采样 | 63 条 | 与上条同因 |
| `TC_EX` 用例族 | 136 条 | 不在本次指定的用例族清单内（`kAccuracyFamilies` 白名单外） |
| `TC_RC` 用例族 | 24 条 | 同上 |
| 空指针用例（`alpha_null`/`beta_null`/`a_null`/`b_null`/`c_null`） | 6 条 | 本轮不在负向 allowlist |
| `handle=nullptr` 用例 | 1 条 | 本轮不在负向 allowlist |
| `transA=INVALID` / `typeA=INVALID` 等非法枚举用例 | 5 条 | 部分在 `kNegativeAllowlist` 14 项内，部分本轮不覆盖 |

**Wave 2 已放开**（对比迭代一）：`transA=T/C`、`transB=T/C` 的所有转置与共轭转置组合已**纳入精度覆盖**（受 `kNonNNMaxDim = 512` 限制），新增 `TC_TX` 手写扩展族覆盖非方阵 × 全转置。

> **⚠ 已知缺口（如实登记，未静默修正）**：任务书 §3.5 要求"alpha = (0,0) 时
> `C = beta * C` 应位精确（EXACT 校验）"。该快路径已被 `TC_ED_699~707`、`TC_ED_728`
> 触及，但当前实测 `TC_ED_707` 全部 256 个元素因 `|gold| < threshold` 被跳过，
> `valid = 0/0`，判为 PASS 却**无有效覆盖**；且探针已定位 `cgemm_ex_beta_scale_do`
> 多 block 列区间漏写缺陷（漏写 120/256 个元素，保留调用前 C 的原值）。
> 详见 arch35 README §11.2。

---

## 3. 测试数据文件

### 3.1 本算子测试用例（本次交付件 2）

位于 `operators/cgemm_ex/tests/st/testcases/`，**56 列 CSV**（含原始 30 列 + 测试设计扩展列）：

| 文件 | 行数 | 说明 |
|---|---|---|
| `cgemm_ex_l0_test_cases.csv` | 39 条 | 冒烟闸门（`required` 38 / `advisory` 1）：12 accuracy + 5 invariant + 3 no_op + 19 exception |
| `cgemm_ex_l1_test_cases.csv` | 1201 条 | 功能主集（`required` 1196 / `advisory` 5）：919 accuracy + 82 invariant + **200 performance**（`TC_PF`） |
| `cgemm_ex_l2_test_cases.csv` | 11 条 | 异常 / 待裁定（11 exception，`required` 9 / `advisory` 2） |

三份 CSV 均为**任务书配套提供的全部自测用例**（合计 1251 条，其中 `TC_` 官方族
1220 条 + `SUP_` 补充用例 50 条，合计 **1251 条**）。

> **执行范围提示**：L0/L1/L2 三份 CSV 是**用例登记与分级闸门**（`run_gate` 字段区分
> required/advisory），当前实际执行的是 arch35 测试工程按 §1「最小可用 + Wave 2 扩展」范围过滤后的
> 388 条（348 精度 + 40 性能）。三份 CSV 本身不改变执行数量。
> `csv_loader.h` 通过 `ReplaceFileExtension2Csv(__FILE__)` 定位 CSV，因此
> **CSV 必须与被实例化的 `.cpp` 同目录**，三个等级各需一条 `INSTANTIATE_TEST_SUITE_P`。

### 3.2 官方用例 CSV 与 baseline

| 文件 | 位置 | 规模 |
|---|---|---|
| 官方用例集（权威源，**迭代二版本**） | `test_cases/cgemmex_test.csv` | **216,188 B，1,221 行（含表头），1,220 条数据行，30 列** |
| 官方用例集（测试工程副本） | `ops-blas/test/gemmex/cgemmex/arch35/cgemmex_test.csv` | 同上，**MD5 与官方逐字节一致** |
| GPU 基线（权威源） | `test_cases/gpu_baseline.csv` | 11,779 B，201 行 |
| GPU 基线（测试工程副本） | `ops-blas/test/gemmex/cgemmex/arch35/gpu_baseline.csv` | 同上，**MD5 与官方逐字节一致** |
| 官方验收脚本 | `test_cases/verify_accuracy.py`、`test_cases/verify_performance.py` | Python，调用 C++ GTest 并检查退出码 |
| 用例生成脚本 | `test_cases/gen_csv.py` | 复现官方 CSV |

MD5 校验值（可自查，**迭代二已更新**）：

```
eda1aa93984250d965135ba9865edc3b  cgemmex_test.csv        # 迭代二（+20 TC_TX 手写扩展）
b1f4879269d58f78c613214f4ce74b49  gpu_baseline.csv        # 未改
```

**迭代一 → 迭代二 CSV 变更**：官方 CSV 数据行由 **1,200 增至 1,220**（+20 行 TC_TX 手写扩展用例，由
`ApplyScopeFilter` 的 `kAccuracyFamilies` 白名单纳入）；MD5 由 `b2c88b625db1194162e658241ba8c4ef`
变更为 `eda1aa93984250d965135ba9865edc3b`。文件字节数由 212,580 B 增至 216,188 B。
GPU 基线文件未变。

CSV 加载器对**表头做 ORDER_MISMATCH 校验**：30 列的名字与位置必须与官方一致，
多一列或少一列都会报 `unexpectedColumns` / `missingColumns`，不会静默读错列。
`random_seed` 列被用作随机数种子，保证重跑可复现同一组输入数据。

### 3.3 测试工程代码

```
ops-blas/test/gemmex/cgemmex/
├── README.md                     # 测试工程完整说明（13 章，权威）
└── arch35/
    ├── cgemmex_case.h            # CSV 加载器 + 30 列结构体 + 范围过滤器
    ├── cgemmex_fill.h            # 填充 token 解析（RANDOM_NORM_5_5 / VALUE_NORM_* / RANDOM_EXTREME / RANDOM_ALTER）
    ├── cgemmex_golden.h          # CPU double 独立 golden + 实部/虚部拆分
    ├── cgemmex_precision.h       # 自包含 MERE/MARE 判定（实部/虚部分开）
    ├── cgemmex_run.h             # NPU 环境 / 设备内存 RAII / aclblasCgemmEx 调用封装
    ├── cgemmex_perf.h            # 采样与计时 / gpu_baseline 关联 / 性能 CSV
    ├── cgemmex_test.cpp          # GTest 测试套件（自有 main，注入 --gtest_filter）
    ├── run.sh                    # 构建 + 运行入口（含依赖检查与 .so 新鲜度防护）
    ├── CMakeLists.txt            # 等价 CMake 构建 + ctest 编排
    ├── cgemmex_test.csv          # 官方用例集副本
    ├── gpu_baseline.csv          # GPU 基线副本
    ├── sample_memory.sh          # 内存采样脚本
    ├── cgemmex_test              # 构建产物（约 803 KB）
    ├── build.log                 # 构建日志
    ├── accuracy.log              # 精度运行日志
    ├── perf.log                  # 性能运行日志
    ├── cgemmex_perf_result.csv   # 性能结果 CSV
    └── memory_samples.csv        # 内存采样结果
```

---

## 4. 环境要求

| 依赖 | 要求 / 本工程实际路径 |
|---|---|
| NPU 芯片 | **Ascend 950PR**（`DAV_3510` / `SOC_VERSION=ascend950`，arch35） |
| CANN Toolkit | **9.1.0**，`/usr/local/Ascend/cann-9.1.0` |
| 被测算子库 | `ops-blas/build-cgemm/libops_blas.so` |
| 编译器 | `g++`（`-std=c++17 -O1`）；CMake **≥ 3.13**（工程用 3.15+），GCC **≥ 11** |
| GTest | 静态库 `/usr/lib/x86_64-linux-gnu/libgtest.a` |
| Python | **3.12**，`/usr/local/python3.12.13/bin/python3`（`cmake` 不在 PATH，**必须显式 export**） |
| golden 依赖 | cblas（Netlib BLAS 复数实现），随测试工程提供；本工程实际采用 `cgemmex_golden.h` 内的 **CPU double 独立 golden**（与 kernel 零共享代码，`std::complex<double>` 展开手写累加） |
| 三方软件（任务书 §3.1） | torch ≥ 2.1.0、torch_npu ≥ 2.1.0.post3；性能标杆 GPU 侧：NVIDIA 驱动 535.104.05 + CUDA 12.2 |

环境自检：

```bash
export PATH=/usr/local/python3.12.13/bin:$PATH
python3 --version                                  # 期望 3.12.13
cmake --version                                    # 期望 /usr/local/python3.12.13/bin/cmake
echo $ASCEND_HOME_PATH                             # CANN 9.1.0 安装目录
npu-smi info                                       # 确认 Ascend 950PR / Ascend950PR_9579
python3 -c "import torch; print(torch.__version__)"        # >= 2.1.0
python3 -c "import torch_npu; print(torch_npu.__version__)"  # >= 2.1.0.post3
```

---

## 5. 构建步骤

```bash
export PATH=/usr/local/python3.12.13/bin:$PATH
cd /workspace/aclblasCgemmEx/ops-blas/test/gemmex/cgemmex/arch35

# 方式 A：run.sh（推荐，自带依赖检查 + .so 新鲜度防护）
bash run.sh --clean

# 方式 B：CMake（等价）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**成功标志**：`run.sh` 末行 `EXIT=0`，`-Wall -Wextra` **零告警**；CMake 方式则
`cmake --build` 无 error 且构建目录产出 `cgemmex_test`。构建日志在 `build.log`。

`run.sh` 的前置检查（缺依赖会直接报错退出，退出码 2）：
仓内 `include/cann_ops_blas.h`、CANN include 目录、`libops_blas.so`、
`libascendcl.so`、`libgtest.a`。

**新鲜度防护**（退出码 3）：若 `blas/gemm/arch35/` 下四个 `cgemm_ex_*` 源文件比
`libops_blas.so` 新，说明算子库没跟上，直接退出并提示重建，
避免跑出"用旧 .so 测新算子"的假结论。确认无需重建时用 `STALE_CHECK=0` 跳过。

> **注意**：`libops_blas.so` 由 ops-blas 仓自身的构建流程产出，不在本测试工程构建范围内。
> 验收前请确认 `ops-blas/build-cgemm/libops_blas.so` 已针对当前 kernel 源重新构建。

---

## 6. 运行步骤

```bash
cd /workspace/aclblasCgemmEx/ops-blas/test/gemmex/cgemmex/arch35

# CPU 自测：6 条（golden 手算 / 转置共轭交叉核对 / 快路径 / 填充语义 / 精度策略），不需要 NPU
bash run.sh --golden

# 打印用例范围清单（覆盖 vs 未覆盖，含逐条原因），不需要 NPU
bash run.sh --scope

# 精度测试：348 条（需要 NPU），实测约 12 分钟（迭代二实测 696.2 s）
# 任务书 §4-3 推荐形态：
bash run.sh --phase accuracy --csv cgemmex_test.csv
# 向后兼容（等价）：
bash run.sh --accuracy --csv cgemmex_test.csv

# 性能测试：40 条（需要 NPU），60 样本实测总时长约 94.03 分钟（1.57 h，含 TC_PF_1091 单次 54.5 min 超时），建议用 nohup
nohup bash run.sh --phase perf --csv cgemmex_test.csv --out cgemmex_perf_result.csv > perf.log 2>&1 &
# 向后兼容（等价）：
nohup bash run.sh --perf --csv cgemmex_test.csv --out cgemmex_perf_result.csv > perf.log 2>&1 &

# 完整测试：golden + scope + accuracy + perf
bash run.sh --phase all     # 或 bash run.sh --all
```

### 6.1 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--phase {accuracy\|perf\|all}` | - | 选择测试阶段（**任务书 §4-3 推荐形态**）。翻译为内部 `--accuracy` / `--perf` / `--all`；传其它值退出码 2 |
| `--accuracy` | - | 向后兼容：等价于 `--phase accuracy` |
| `--perf` | - | 向后兼容：等价于 `--phase perf` |
| `--all` | - | 向后兼容：等价于 `--phase all` |
| `--device <id>` | `0` | NPU 设备号，等价于环境变量 `ASCEND_DEVICE_ID` |
| `--samples <n>` | `60` | 性能有效样本数。**必须 >50**（任务书 §7.4 严格不等式）；下限校验 **51**，传 `< 51` 会被拒绝（退出码 2）。未显式传时 `run.sh` 会注入默认 `60` |
| `--warmup <n>` | `3` | 性能 warmup 轮数 |
| `--out <file>` | `<dir>/cgemmex_perf_result.csv` | 性能 CSV 输出路径 |
| `--csv <file>` | 与本脚本同目录的 `cgemmex_test.csv` | 官方用例 CSV |
| `--baseline <file>` | 与本脚本同目录的 `gpu_baseline.csv` | GPU 基线 CSV |
| `--clean` | - | 只构建不运行 |
| `-h / --help` | - | 打印帮助 |

### 6.2 退出码

| 退出码 | 含义 |
|---|---|
| `0` | 全部 PASS（或 `--help` / `--clean`） |
| `1` | 存在 FAIL |
| `2` | 依赖缺失 / 脚本位置不对 / CLI 参数错误（`--phase` 取值非法、`--samples < 51` 等） |
| `3` | 算子源比 `.so` 新（新鲜度防护触发） |
| `4` | 编译失败或二进制不存在 |

> **命令说明**：`run.sh` 支持 `--phase {accuracy|perf|all}` 与旧扁平开关
> `--accuracy` / `--perf` / `--all` 两种形态；前者是任务书 §4-3 的推荐写法，
> 后者为向后兼容保留。两者最终映射到同一套内部逻辑。
> 此外 `--samples` 默认 `60`、下限 `51`（严格 >50，任务书 §7.4），未显式传时
> `run.sh` 会注入默认 `--samples 60`。

---

## 7. 环境变量

全部可选，均有默认值。

| 变量 | 默认 | 作用 |
|---|---|---|
| `CANN_PKG_PATH` | `/usr/local/Ascend/cann-9.1.0` | CANN 安装根目录（`run.sh` 也认 `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME`） |
| `ASCEND_HOME_PATH` | - | CANN 安装根目录的标准变量名；CMake 与 `run.sh` 均识别 |
| `ASCEND_DEVICE_ID` | `0` | NPU 设备编号（`aclrtSetDevice`）；等价于命令行 `--device N` |
| `OPSBLAS_LIB` | `<repo>/ops-blas/build-cgemm` | `libops_blas.so` 所在目录 |
| `GTEST_LIB` | `/usr/lib/x86_64-linux-gnu/libgtest.a` | gtest 静态库路径 |
| `CGEMMEX_DATA_DIR` | 脚本所在目录 | 运行期 CSV 查找目录（编译期也烘焙了一份默认值） |
| `SKIP_BUILD` | 未设 | 设为 `1` 跳过编译，直接用现有 `cgemmex_test` 二进制 |
| `STALE_CHECK` | 未设（开启） | 设为 `0` 跳过「kernel 源比 .so 新」的防护检查 |
| `BUILD_LOG` | `<dir>/build.log` | 构建日志路径 |

**CSV 查找顺序**：`-DCGEMMEX_DATA_DIR`（编译期） > 环境变量 `CGEMMEX_DATA_DIR` >
可执行文件所在目录 > 当前目录。

**多卡机设备选择**：二选一 —— `export ASCEND_DEVICE_ID=1`，或
`bash run.sh --device 1 --accuracy`。

**CMake 变量**：`-DCANN_PKG_PATH=<CANN 根>`、`-DOPSBLAS_LIB_DIR=<libops_blas.so 所在目录>`、
`-DCGEMMEX_DATA_DIR=<CSV 目录>`、`-DGTEST_LIB=<path>`。构建目录用 `add_custom_command`
自动把 `cgemmex_test.csv` / `gpu_baseline.csv` 复制过去，因此**只拷 build 目录就能跑**。

---

## 8. 精度判据

### 8.1 任务书 §3.2 要求（原文引用）

golden 由 cblas（Netlib `cgemm`）单标杆比对生成，输出矩阵 C（m×n）全矩阵验证，
实部、虚部分别比对。算子计算精度需满足生态算子开源精度标准，本算子默认主路径
输入输出为单精度复数，比对时实部、虚部分别按 FLOAT32 标准判定：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| COMPLEX64（实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |
| FLOAT32（typeT = ACLBLAS_R_32 时） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32 * ULP |
| FLOAT16（typeT = ACLBLAS_H_R_32 时） | 2^-9 (1.95e-3) | 2^-9 (1.95e-3) | 0.99 | 1e-1 或 32 * ULP |

逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`；当用例同时满足
matched_ratio ≥ required_matched_ratio 且 max_abs_error ≤ max_abs_error_limit 时，
判定该用例精度通过。

特例：`alpha = (0,0)` 且 alpha 指针非空时跳过矩阵乘，`C = beta * C` 的结果应位精确
（EXACT 校验）。本算子不含随机数生成，§3.5 的正态/均匀分布是**测试输入数据的生成规则**，
不是算子行为。

### 8.2 测试工程实际执行口径

任务书配套的官方 CSV 只提供 MERE/MARE 两列阈值，因此本工程按官方 CSV 执行：

- 阈值逐行取自 CSV 的 `mere_threshold` / `mare_multiplier` 两列，**不写死**。
- 本范围内所有行的取值：`mere_threshold = 0.00012207031`（= 2^-13）、
  `mare_multiplier = 10.0`，即 `threshold = 1.221e-04`、`outlierLimit = 1.221e-03`。
- 判定逻辑逐行对应仓内 `ops-blas/test/frame/verify.h::MereMareStrategy`
  （该文件按任务书为只读参考，故本工程在 `cgemmex_precision.h` 内独立实现、不 include）：
  用例 PASS 需同时满足 `mismatchCount == 0` **且** `mere < threshold` **且**
  `maxRelErr < outlierLimit`。
- **纯相对误差判据**：`|gold| < threshold` 的元素直接跳过、不参与平均。
  因此大 `k` 的 float32 GEMM 会在近零元素上出现相对误差爆炸而绝对误差仍很小的情形；
  为便于归因，`PrecisionVerdict` 额外记录 `maxAbsErr`（**不参与判定**，仅诊断用）。

### 8.3 实部与虚部分开比较

`aclblasComplex` 是 8 字节结构 `struct { float real; float imag; }`，C 按**列主序**存储，
`C(i,j)` 位于缓冲区偏移 `i + j*ldc` 处，因此一个元素的实部与虚部在内存里是
**连续的两个 float**（步长 1），相邻同列元素之间步长是 `2*ldc`。

`cgemmex_golden.h::SplitRealImag()` 把输出拆成两个纯 `float` 向量 `outReal` / `outImag`，
golden 同样拆一份，然后对**实部向量**和**虚部向量分别跑一次 MERE/MARE 策略**，
两者都通过才算该用例通过（对应官方 `verify.h::verifyMereMareComplexFloat` 的 AND 语义）。

日志中所有 `mere(re/im)=A/B`、`maxRelErr(re/im)=A/B`、`maxAbsErr(re/im)=A/B`、
`valid(re/im)=A/B`、`skipped(re/im)=A/B`、`mismatch(re/im)=A/B`、`outlier(re/im)=A/B`
都是同一格式：**斜杠前是实部，斜杠后是虚部**。

---

## 9. 性能判据

### 9.1 任务书 §3.3 要求（原文引用）

测试设备：Ascend 950PR（CANN 9.1.0）。性能数据为默认主路径
（typeA = typeB = typeC = ACLBLAS_C_32，即 COMPLEX64）输入场景下的平均单次耗时
（Avg time，单位 us），须先 warmup 再有效采样 **>50 次**取平均。
算子在各性能 case 下的平均单次耗时应不高于下表标杆耗时：

| case | m | n | k | transA | transB | typeA | typeB | typeC | 标杆耗时（Avg time，us） |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 1024 | 1024 | N | N | C_32 | C_32 | C_32 | 415.95 |
| 2 | 2048 | 2048 | 2048 | N | N | C_32 | C_32 | C_32 | 3243.65 |
| 3 | 1024 | 1024 | 1024 | T | N | C_32 | C_32 | C_32 | 411.27 |
| 4 | 2048 | 2048 | 2048 | N | N | R_32 | R_32 | R_32 | 851.86 |

### 9.2 测试工程实际执行口径

- 采样：warmup **3** 次 + 有效样本 **60** 个（默认），输出 `mean / min / p50 / max`。
  任务书 §7.4 要求"有效采样 >50 次"（严格不等式），`run.sh` 默认 60、下限校验 51，
  未显式传 `--samples` 时会注入默认 60。
- 判定：`ratio = gpu_ms / (npu_mean_us / 1000)`，`ratio >= 0.4` 为 PASS。
  阈值取官方 `verify_performance.py` 的 `PERF_THRESHOLD = 0.4`。
  （该脚本 docstring 写 0.8，**以常量 0.4 为准**。）
- 结果列：`case_name,m,n,k,transA,transB,type,npu_mean_us,npu_min_us,npu_p50_us,
  npu_max_us,valid_samples,gpu_baseline_ms,ratio_gpu_over_npu,verdict,note`。

> **注**：任务书 §7.4 要求"有效采样 >50 次"，本工程 `run.sh --samples` 默认 60、
> 下限校验 51（严格 >50），已对齐任务书口径。旧的固定 50 次偏差已修复，见 §11.3。

---

## 10. 输出文件

均落在 `ops-blas/test/gemmex/cgemmex/arch35/`：

| 文件 | 内容 |
|---|---|
| `build.log` | 每次构建的完整 `g++` 命令与 `EXIT=` 行 |
| `accuracy.log` / `accuracy_final.log` | **348 条精度用例**逐条结果（含实部/虚部分开的 `mere`、`maxRelErr`、`maxAbsErr`、`valid`、`skipped`、`mismatch`、`outlier` 与最坏元素），末段是 gtest 汇总与 FAIL 清单 |
| `perf.log` | 性能逐条结果（`mean/min/p50/max`、`n`、`baseline`、`ratio`、`verdict`） |
| `perf.nohup` | `nohup` 后台运行的原始输出 |
| `cgemmex_perf_result.csv` | 性能结果 CSV，**逐条 flush**，中断也留下有效部分结果 |
| `memory_samples.csv` | 内存采样：`hbm_before_mb / hbm_peak_mb / hbm_after_mb / delta_peak_mb / samples / elapsed_s / ws_formula_mb / c_buf_mb / a_buf_mb / b_buf_mb / exit_code / src` |

运行内存采样：`bash sample_memory.sh`。

---

## 11. 已知问题（如实披露，未修饰）

以下为迭代二本机实测结果（2026-09-17 精度 / 2026-09-17 性能），验收时可直接对照复现。

### 11.1 精度：348 条中 238 条 FAIL（A/B/C/D 四类）

`bash run.sh --accuracy` → **348 条**，**696.2 s**，**110 PASS / 238 FAIL**
（`accuracy_final.log`，`Running 355 tests from 2 test suites`，`exit=1`）。

**A/B/C/D 分类结果（238 FAIL）**：

| 类别 | 数量 | 数值状态 | 说明 |
|---|---|---|---|
| **A 类**（判据形式边缘）| 35 | ✅ 数值正确 | 全部 `mismatch=0/0`，`mere` **低于阈值 10 倍以上**（1.396e-06 .. 1.596e-05），仅 `maxRelErr` 越过 `outlierLimit` |
| **B 类**（非方阵）| 7 | ❌ 数值错误 | `m ≠ n` 的非方阵（非非对称转置） |
| **C 类**（非对称转置）| 194 | ❌ 数值错误 | 6 种非对称转置组合全数命中，`mere` 从 `1.3e+00` 到 `4.7e+01` |
| **D 类**（起点 bug）| 2 | ❌ 数值错误 | `TC_L0_003`（4³）、`TC_L0_004`（8³）T/N 方阵，最坏元素符号反转 |
| **合计** | **238** | | |

**关键事实（诚实披露）**：
- **数值上真错的 FAIL：203 条**（B 7 + C 194 + D 2）。
- **数值正确但被官方判据形式判 FAIL：35 条**（A 类），若按任务书 §3.2 双轨判据（`maxAbsErr ≤ 1e-2` 且 `mismatch == 0`）重算，
  348 条中 **145 条 PASS（41.7%）**，详见 `reports/accuracy_report.md` §4.2。
- **官方 CSV 547 行 `R_32` 本轮 0 条执行**，「R_32 未实现」的表述只能是「未测」，无数据支撑「未实现」。

**方阵路径可用性（Wave 2 放开后）**：
- 方阵 T/T 31 条：17 PASS + 12 A_EDGE + 2 B_NONSQ → **T/T 方阵路径可用**（12 条为判据形式边缘）
- 方阵 C/C 34 条：21 PASS + 11 A_EDGE + 2 B_NONSQ → **C/C 方阵路径可用**
- 方阵 N/N 119 条：60 PASS + 12 A_EDGE + 47 FAIL（含 D 类 2 条）
- 非对称转置（6 种组合）194 条：**全部 FAIL**（数值真错，`mere ≥ 1.3e+00`）
- 非方阵（`m ≠ n` 或 `n ≠ k`）22 条：1 PASS_DEGEN + 7 B_NONSQ + 12 C_ASYM_TRANS + 2 其它

**D 类 2 条（起点 bug，白盒未定位）**：

| 用例 | 形状 | Pair | mere | 最坏元素 |
|---|---|---|---|---|
| `TC_L0_003` | 4x4x4 | T/N | 1.109e+01 | `real@0 out=39.7887 gold=-0.709427 relErr=5.7e+01` |
| `TC_L0_004` | 8x8x8 | T/N | 8.717e+00 | `real@0 out=32.2398 gold=-0.169357 relErr=1.9e+02` |

两条均为 T/N 方阵，**最坏元素符号反转**（relErr 达 10² 量级），Wave 3+ 待 developer-code 定位。

**A 类细节（35 条，判据形式边缘 FAIL）**：
`mere` 全程 `1.396e-06 ~ 1.596e-05`，**低于 `mere_threshold = 1.221e-04` 一个数量级以上**，
`mismatch = 0/0`，`maxAbsErr ≤ 1e-2`；只有 `maxRelErr` 越过 `outlierLimit = 1.221e-03`。
集中在 `TC_SQ` / `TC_CV` 的大 `k` 方阵 shape，是 float32 累加误差撞上**纯相对**判据的经典形态：
`|gold|` 很小的元素上 `relErr` 大而 `absErr` 小。判据形态问题，非算子正确性缺陷。

**注**：本 README 的 A/B/C/D 分类与 `reports/accuracy_report.md` §4 分类法一致；
`reports/` 目录中 `REPORT_INDEX.md` 已归档 4 类明细与任务书 §3.2 双轨判据对照表。

### 11.2 `TC_ED_707`：beta-only 快路径真实算子缺陷（探针实测）

用例：`16x16x16`，`trans=NN`，`alpha=(0,0)`，`beta=(0.5,0.5)`，`ldc=16`，`seed=707`。
`alpha==(0,0)` 使 host 侧（`cgemm_ex_host.cpp:566`）分派到
`LaunchCgemmExFastPath` → `cgemm_ex_beta_scale_do`，`blocks = min(aivCoreCount, max(1,n))`。

探针实测的逐元素掩码（`'.'` = 被正确写成 `beta*C_in`，`'X'` = **仍是 `C_in` 原值，未被写入**）：

```
       j: 0123456789012345
i=0     .........XXXXXXX
i=1     .........XXXXXXX
...
i=7     .........XXXXXXX
i=8     ........XXXXXXXX
...
i=15     ........XXXXXXXX
untouched=120/256
```

**结论：beta-only 快路径只写了一部分列，右侧一个连续的 7~8 列带状区域被跳过，
那 120 个元素保留了调用前 C 缓冲区的 `C_in` 值。**
行数 0–7 漏写第 9–15 列（7 列），行数 8–15 漏写第 8–15 列（8 列），
`8*7 + 8*8 = 120` 与计数吻合。被写到的元素是**逐位精确**的 `beta*C_in`。
这是 `cgemm_ex_beta_scale_do` 的多 block 列区间划分越界/漏区间问题，
**需由算子开发方修复，测试工程不修改 kernel 源码**。

另需知悉：`TC_ED_699/700/701/702/705/706`（零维或 `k=0`）也是
`valid = 0/0` 的退化通过，不应计为有效覆盖。

### 11.3 性能：40 条采样 0 PASS，覆盖率回退

`bash run.sh --perf` → **40 条中 0 条 PASS**（迭代二采样策略**主动剔除** 10 条超大 shape，
见 §2.2）。落盘的 `cgemmex_perf_result.csv`（3,777 B / 41 行 = 表头 + 40 数据行）
含 **36 FAIL + 4 ERROR**，`exit=1`；总耗时 **94.03 分钟 = 1.57 小时**（`perf_final.log`，
`Running 51 tests from 2 test suites`，2026-09-17 实测）。

全部用例 `ratio_gpu_over_npu` 介于 `0.0000` ~ `0.0114`，**远低于 PASS 阈值 0.4**；
最好比值 `0.0114`（`TC_PF_1005`, 1³）距阈值差 **34.8×**。

**任务书 §3.3 覆盖率回退**：4 条标杆 case 中仅 Case 1（`1024³` NN C_32）落入本次采样
（`TC_PF_1001`），Case 2（`2048³` NN C_32 = `TC_PF_1002`）**在迭代二被主动剔除**，
覆盖率从迭代一的 2/4 **回退为 1/4**。

| 标杆 case | 用例 | shape | 标杆耗时 | 实测 mean | 倍数 |
|---|---|---|---|---|---|
| 1 | `TC_PF_1001` | 1024x1024x1024 NN C_32 | 415.95 us | **6,140,387 us** | **约 14,762 倍** |
| 2 | `TC_PF_1002` | 2048x2048x2048 NN C_32 | 3243.65 us | **本次未采样** | — |
| 3 | — | 1024³ **TN** C_32 | 411.27 us | 未覆盖（非 NN）| — |
| 4 | — | 2048³ NN **R_32** | 851.86 us | 未覆盖（非 C_32）| — |

**4 条 ERROR**（`stream sync failed`）：TC_PF_1091/1099/1101/1103，均为 `k=64` 且 `m×n ≥ 2,097,152`。
TC_PF_1091（1024x2048x64）单次耗时 **54.5 min** 是本轮 94.03 min 总时长的主导因素。

**6→5 launch 融合的实测性能影响**（迭代一 6 launch vs 迭代二 5 launch，同一 shape 对比）：

| 区间 | Δ（迭代二 − 迭代一）| 说明 |
|---|---|---|
| 1³–8³ | +0.2% ~ +1.2% | 小 shape 平坦，**未受益于 launch 融合** |
| 32³–128³ | **−14.9% ~ −37.3%** | 中等小 shape 兑现了 launch 开销回收（预期 1/6 ≈ 17% 附近） |
| ≥512³ | ±0.33%（测量噪声）| Cube-throughput 主导，launch 次数不再影响总耗时 |

**综合判断**："6→5 launch 融合带来一致性能收益"的说法**无法成立**；仅在 32³–128³ 有可观收益。
采样次数由 50 提升至 60 但两次样本集不完全相同，严格说不构成同等条件对比。

**Cube 瓶颈**：等 FLOPs 对比（2.15e9 FLOPs），`TC_PF_1127` (512x512x4096) = 8,184,031 us 比
`TC_PF_1001` (1024³) = 6,140,387 us **慢 33%**——即 3 次独立 launch 各自重读 A/B 的 HBM 流量放大
是主要嫌疑（同 FLOPs 下 2× HBM 流量、1.33× 时间，同量级）。该瓶颈需**方案 C（3 GEMM 融合成 1 launch
共享 A/B）** 消除，但方案 C 因 **203 个新 FAIL 精度回归已退回**（见 `cgemm_ex_host.cpp` L531–533）。

**采样次数已对齐任务书要求**：任务书 §7.4 要求"有效采样 >50 次"（严格不等式），
`run.sh --samples` 现已默认 60、下限校验 51（传 `< 51` 会被拒绝），未显式传时
`run.sh` 会注入默认 `--samples 60`。本次（2026-09-17）已按新口径运行。

**运行时长**：40 条实测 **94.03 min = 1.57 h**（含 TC_PF_1091 单次 54.5 min 的 ERROR 场景），
一次会话可跑完，但**强烈建议 `nohup` 后台执行**以防 session 中断丢失。

### 11.4 内存采样数据不完整

`memory_samples.csv` **本次未重采**（数据时间戳 2026-09-15 01:45，来自迭代一采样），
仅有 3 条 `measured` 记录（`TC_SQ_017`、`TC_SQ_020`、`TC_SQ_030`）
+ 1 条 `formula` 推算记录（`TC_SQ_031`）。其中 `TC_SQ_030`（`1024³`）的
`exit_code = 1`，采样流程**未干净退出**（TC_SQ_017/020 的 `exit_code = 0`）；本次未复采、未定位。

数据保留作**内存画像**用途，不作合规判定依据。任务书 §4 交付件 3 要求的"内存占用数据"
**当前未针对迭代二二进制重新采样**。若后续需要严格的内存上限验证，需用
`bash sample_memory.sh` 重采。任务书 §3.4 对内存要求标注为"不涉及"，本轮未做严格内存上限验证。

**实测 `delta_peak` 存在 ~270 MB 的地板值**（NPU 常驻 runtime 开销）；
扣除地板值后 `delta_peak` 随 scale 单调增长，TC_SQ_030 净增量约 1031 MB。
**workspace 上限公式（`ws_formula_mb`）与实测净增长存在 2~4 个数量级的偏差**，
原因是公式仅覆盖理论最小 workspace，未包含 runtime 地板、3-GEMM 中间 buffer、tile 双缓冲等。
详见 `reports/memory_report.md`。

### 11.5 与任务书前提的三处不一致（如实记录，未静默修正）

1. 任务书称 CSV 有 **52 列** —— 实际官方 CSV 为 **30 列**
   （`head -1 | awk -F, '{print NF}'` = 30）；本工程交付的 L0/L1/L2 三份 CSV 为 **56 列**
   （原始 30 列保真 + 测试设计扩展列，见 `../docs/TEST.md` §4）。
2. 任务书称 `mere_threshold + mare_multiplier` 公式在 `test_cases/verify_accuracy.py` 里 ——
   该脚本**不含**公式，只调用二进制并检查退出码；公式在 `ops-blas/test/frame/verify.h`
   的 `MereMareStrategy`（见 arch35 README §7）。
3. `test_cases/verify_performance.py` 的 docstring 写 **0.8**，但常量是
   `PERF_THRESHOLD = 0.4`；**以常量为准**，本工程取 0.4。

### 11.6 与另一条开发线的关系

范围不同，不矛盾：另有一条开发线只测 `128x128x128` NN 主路径的两个 P1/P2 缺陷点；
本工程覆盖 `C_32` 的 **348 条精度用例 + 40 条性能用例**（Wave 2 放开转置后），
包含 128³、方阵 T/T、方阵 C/C、非方阵 × 全转置（TC_TX 手写扩展）、大 k 方阵、快路径、
极值填充、padding 等；`2048³` 精度侧覆盖（TC_SQ_031），性能侧**未采样**（TC_PF_1002 剔除）。

---

## 12. 相关文档

| 文档 | 路径 | 说明 |
|---|---|---|
| 测试工程完整说明（权威） | `ops-blas/test/gemmex/cgemmex/README.md` | 13 章：范围、环境、构建、运行、实虚部拆分、判据、用例清单、数据溯源、复现记录、缺陷归因、目录结构 |
| 测试设计 | `../docs/TEST.md` | 本算子 ST 测试设计（56 列 Schema、spec 映射、判据、补充用例 51 条、执行步骤登记） |
| 详细设计 | `../docs/DESIGN.md` | 算子详细设计 |
| 算子需求 | `../docs/REQUIREMENTS.md` | 需求分析 |
| 接口文档 | `../docs/aclnnCgemmEx.md` | aclnn API 接口定义 |
| 数学契约 | `../docs/spec.yaml` | L0 数学契约（`numerical_tolerance`、`boundary_conditions`、`extreme_inputs` 来源） |
| 测试设计要素 | `st/design/03_参数定义.yaml`、`st/design/04_测试因子.yaml`、`st/design/05_约束定义.yaml` | 参数、测试因子、约束定义 |
| 官方用例集说明 | `../../../test_cases/README.md` | 官方 CSV / baseline / 验收脚本说明 |
| 任务书 | `../../../aclblasCgemmEx_task_doc.md` | §3 验收标准、§4 交付件、§3.5 自验要求 |

> 路径拼写注意：`aclblasCgemmEx` **两个 l**，不要写成 `aclbasCgemmEx`。

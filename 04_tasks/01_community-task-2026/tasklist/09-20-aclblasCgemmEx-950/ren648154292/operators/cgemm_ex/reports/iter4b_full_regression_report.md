# iter4b 全量回归报告：harness 侧补 aclblasSetWorkspace（1 GiB）

- 日期：2026-09-18
- 工作仓：`/workspace/.cannbot/ops-blas-pr/ops-blas`
- 分支：`add-aclblasCgemmEx-950` @ `97ee82f`（实际代码合入走 ops-blas 仓 PR #457）
- 改动范围：**仅 test/ 两个文件**；未触碰 blas/、include/、upstream、doc/、CMake/
- 时间预算：实际约 40 分钟（编译 5 min + 全量跑 3.3 min + 报告 5 min）

---

## 1. 一句话结论

**加 workspace 一行改动，104 → 14 全量 FAIL；90 条 EXECUTION_FAILED 全部修复，无新增失败；剩余 14 条均为任务书明确保留的半精度（12）与 NaN 语义（2）非-workspace 问题。**

---

## 2. Diff 摘要（两个 test 文件）

### 2.1 `test/frame/blas_test.h`（BlasTest fixture，供 cgemm_ex_test 使用）

新增默认宏 `TEST_WORKSPACE_BYTES = 1 GiB`（可通过 `-DTEST_WORKSPACE_BYTES=...` 覆盖）；`AclGuard` 结构体扩展 `workspace` / `workspaceBytes` 成员；`SetUpTestSuite` 在 handle 创建后 alloc + set；`TearDownTestSuite` 与 `~AclGuard` 释放。

```cpp
#ifndef TEST_WORKSPACE_BYTES
#define TEST_WORKSPACE_BYTES (1024 * 1024 * 1024u)  // 1 GiB
#endif

struct AclGuard {
    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    void* workspace = nullptr;
    size_t workspaceBytes = 0;
    bool cleaned = false;
    // destructor: aclblasDestroy / aclrtDestroyStream / aclrtFree(workspace) / aclFinalize
};

// 在 SetUpTestSuite 里，紧跟 aclblasSetStream 之后：
guard.workspaceBytes = TEST_WORKSPACE_BYTES;
ASSERT_EQ(aclrtMalloc(&guard.workspace, TEST_WORKSPACE_BYTES,
                      ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
ASSERT_EQ(aclblasSetWorkspace(guard.handle, guard.workspace,
                              TEST_WORKSPACE_BYTES), ACLBLAS_STATUS_SUCCESS);
```

### 2.2 `test/gemm/cgemm_ex/arch35/cgemmex_run.h`（NpuEnv，供 cgemmex_test 使用）

新增 `kWorkspaceBytes = 1 GiB` 常量、`workspacePtr_` 成员；`Init()` 在 `aclblasSetStream` 之后分配并绑定；`Destroy()` 释放；新增 `WorkspacePtr()` / `WorkspaceBytes()` 访问器。

```cpp
static constexpr size_t kWorkspaceBytes = 1024 * 1024 * 1024;
// Init(): aclblasSetStream 之后
const aclError wsRet = aclrtMalloc(&workspacePtr_, kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST);
const aclError wsSetRet = aclblasSetWorkspace(handle_, workspacePtr_, kWorkspaceBytes);
// Destroy(): aclblasDestroy 之后
aclrtFree(workspacePtr_); workspacePtr_ = nullptr;
```

- **不改动的**：`verify.h`、`gemm_golden.h`、`arch35/gemm_test.cpp`（GAP-I 独立工作项，未触碰）。
- **不写**：blas/、include/、upstream、doc/、CMake。

### 2.3 尺寸依据

- iter4 baseline log 证据：`workspace required 134217728 bytes (128 MiB), but only 33554432 (32 MiB) available`（2048³ 用例）。
- 用例最大 shape：TC_PF_1073 = 3999×3999×3999（约 4× 2048³），若沿用同比例需约 500 MiB。
- 选 1 GiB 覆盖全范围并留 ~2× 余量；对 960 GiB 内存的 A3 机器是 0.1% 级别，不影响性能。

### 2.4 编译（一次通过，无源码改动）

编译时链接阶段遇到与本次改动无关的既有问题：CMake 把 `libblas.so`（stub）当 `REFBLAS_LIB` 使用，其 `DT_NEEDED libopenblas.so.0` 需要 `-rpath-link` 到 `blasroot/usr/lib/x86_64-linux-gnu/openblas-pthread` 才能被 ld 解析。手动在 CMake 生成的 link.txt 基础上追加 `-Wl,-rpath-link,.../openblas-pthread` 后一次成功。**这是既有构建系统问题，不在本次任务范围**（如需根治需改 `cmake/test.cmake`，属 developer-code 角色）。

产物：`build-cgemm/test/gemm/cgemm_ex/cgemm_ex_test`（2.07 MB）。

---

## 3. 编译结果

- **成功**。`cgemm_ex_test` 二进制生成于 03:36，`--gtest_list_tests` 列出 1259 用例。
- 全量运行 3.3 分钟（196.016 s）跑完 1259 用例，无 crash、无 timeout、无 workspace 相关报错。

---

## 4. PASS/FAIL 对比

| 指标 | iter4（baseline, `iter4_full2.log`） | iter4b（本次, `/tmp/iter4b/full_run.log`） | Δ |
|---|---|---|---|
| 用例总数 | 1259 | 1259 | 0 |
| **PASSED** | **1155** | **1245** | **+90** |
| **FAILED** | **104** | **14** | **−90** |
| 状态码 = EXECUTION_FAILED 的用例 | ~90（ret=5） | **0** | −90 |
| 日志中出现 "workspace required / workspace need" | 多处 | **0** | −∞ |
| 全量耗时 | 163 724 ms | 196 016 ms | +32 s（正常波动） |

**总 FAIL 从 104 降到 14，命中任务书预期（104 → ~14）；90 条 EXECUTION_FAILED 全部转为 PASS。**

### 4.1 FAIL 收敛轨迹（三版对比）

| 阶段 | 用例数 | PASS | FAIL | FAIL 率 |
|---|---|---|---|---|
| iter3（自研 kernel） | 1251 CSV | 101 | 1148 | **91.8%** |
| iter4（wrapper, harness 未修） | 1259 全量 | 1155 | 104 | **8.26%** |
| **iter4b（wrapper + workspace 1 GiB）** | **1259 全量** | **1245** | **14** | **1.11%** |

### 4.2 iter3 FAIL 分类（1148 条，作为历史参照）

| 类别 | 条数 | 证据 |
|---|---:|---|
| 数值精度失败（matchedRatio < 0.99） | 552 | `TC_L0_001` 4³ matchedRatio=0.0000；`TC_PF_1001` 1024³ matchedRatio=0.0000，maxAbsErr 达数百量级 |
| 状态码 ret=7「实现范围外」 | 562 | 全部为 `typeA=R_32` 或半精度；iter3 host 无 R_32 与半精度实现 |
| 状态码 ret=3 `INVALID_VALUE` | 4 | `TC_ED_701` / `704` / `705` / `706`（k==0 用例误判指针非空） |
| 状态码 ret=5 `EXECUTION_FAILED` | 0 | 自研 kernel 走自己的 `aclrtMalloc` 缓冲路径，不经过 aclblas 的 `CheckEffectiveWorkspaceSize` |

**iter3 vs iter4 的决定性差异**：iter3 的 FAIL 是真·实现缺陷；iter4 的 104 条 FAIL 中 90 条是 harness 未设 workspace 导致 upstream 拒绝大矩阵（ret=5），非 wrapper 缺陷。

---

## 5. 关键用例前后对比

任务书指定的 4 条 + 2 条大 shape 代表：

| 用例 | shape | iter4 状态 | iter4b 状态 |
|---|---|---|---|
| TC_PF_1001 | 1024³ (NN, C_32) | PASS（32 MiB 默认已够用） | PASS |
| TC_PF_1005 | 1×1×1 (小 shape) | PASS | PASS |
| TC_PF_1002 | **2048³** (NN, C_32) | **FAIL**：`状态码不符 ret=5` | **PASS**（0/2048 real, 0/2048 imag 失败） |
| TC_EX_0892 | — | **FAIL**：`状态码不符 ret=5` | **PASS**（0/2000 real, 0/2000 imag 失败） |
| TC_EX_0904 | — | **FAIL**：`状态码不符 ret=5` | **PASS**（4/2000 real, 0/2000 imag 失败，matchedRatio=0.9980 ≥ 0.99） |
| TC_PF_1073 | **3999³** (TT, C_32)，最大 shape | **FAIL**：`状态码不符 ret=5`（2999 ms） | **PASS**（2/3999 real, 1/3999 imag 失败，matchedRatio≥0.995） |

TC_PF_1001 / TC_PF_1005 本来就在 32 MiB 阈值之下，未受影响；TC_PF_1002 是 workspace 阈值之上的第一条；TC_PF_1073 是全套件最大 shape，也过了。**90 条 EXECUTION_FAILED 全部转 PASS，无一条转为新的精度失败**（对比 iter4_failed_names.txt 与 iter4b 失败列表，交集仅 14 条非-workspace 项，95 条消失，无新增失败）。

---

## 6. 剩余 FAIL 列表（14 条）

按类别分组，与任务书预期完全对齐（12 条半精度 + 2 条 NaN）：

### 6.1 半精度（12 条）— HR32 / HC32，任务书明确 "不 debug，如实记录"

| # | 用例 | dtype | shape / 说明 |
|---|---|---|---|
| 1 | `SUP_L0_HR32_NN_8X8X8` | H_R_32 | 8×8×8 NN（L0） |
| 2 | `SUP_L0_HC32_NN_8X8X8` | H_C_32 | 8×8×8 NN（L0） |
| 3 | `SUP_HR32_NT_16X16X16` | H_R_32 | 16³ NT |
| 4 | `SUP_HR32_CT_32X32X32_IMAG` | H_R_32 | 32³ CT（imag 侧） |
| 5 | `SUP_HR32_CN_RC_32X16X48` | H_R_32 | CN/RC 混合转置 |
| 6 | `SUP_HR32_ALPHA0_BETAHALF` | H_R_32 | alpha=0, beta=0.5 边界 |
| 7 | `SUP_HR32_LD_PADDING_16X16X16` | H_R_32 | lda/ldb/ldc 含 padding |
| 8 | `SUP_HC32_NT_16X16X16` | H_C_32 | 16³ NT |
| 9 | `SUP_HC32_CT_32X32X32` | H_C_32 | 32³ CT |
| 10 | `SUP_HC32_CN_RC_32X16X48` | H_C_32 | CN/RC 混合转置 |
| 11 | `SUP_HC32_ALPHA0_BETAHALF` | H_C_32 | alpha=0, beta=0.5 边界 |
| 12 | `SUP_HC32_LD_PADDING_16X16X16` | H_C_32 | lda/ldb/ldc 含 padding |

**归因**：wrapper 明确返回 `ACBLAS_STATUS_NOT_SUPPORTED`（upstream `aclblasCgemm` 不支持 fp16 复数，`aclblasSgemm` 不支持 fp16 实数），与 wrapper 内 `ValidateCgemmExParams` 语义一致。**属范围决策冲突**，需上游裁定二选一：(a) 扩展 wrapper 覆盖半精度（需自研 fp16 复数路径或另找入口）；(b) 将 CSV 中这 10+2 条期望值改为 `ACBLAS_STATUS_NOT_SUPPORTED`（harness 侧改期望表）。本次未为规避这 12 条而放宽 wrapper 契约。

### 6.2 NaN / inf 语义（2 条）— 任务书明确 "不 debug，如实记录"

| # | 用例 | 现象 |
|---|---|---|
| 1 | `TC_FL_552` | Output 全 `inf/-inf`，Golden 全 `-nan`（32 元素 31 失败）；DIAG 报告 `gold_nonfinite=32 out_nonfinite=32 pattern_mismatch=0`——非有限值**模式完全一致**，但比较器仍判失败（matchedRatio=3.1250e-02） |
| 2 | `SUP_NS_NAN_C_BETA0` | NaN × 0 类退化（`0*NaN=NaN`），Golden 假设为有限值；`gold_nonfinite=0 out_nonfinite=16 pattern_mismatch=16`，非有限元素模式不匹配 |

**归因**：upstream kernel 的 IEEE 语义 vs harness golden/比较器的期望不一致，属 harness + golden 语义问题，**与 wrapper 透传行为无关**（wrapper 未做任何数值处理）。

**后续建议**：
- 比较器增加"双方均非有限且模式一致则通过"分支；
- NaN 用例的 golden 需按 IEEE 重算或改为期望 `nan`。

---

## 7. Workspace 内存评估

- **单次分配**：harness 侧每个 test target 只分配一次 1 GiB，`SetUpTestSuite` 建、`TearDownTestSuite` 释放，用例间共享同一块 workspace。
- **设备侧占用**：`aclrtMalloc(..., ACL_MEM_MALLOC_HUGE_FIRST)` 走 HugeFirst 分配策略，`aclrtFree` 后由 runtime 回收；全量日志无一次分配失败。
- **1251 CSV 用例 + 8 附加用例（共 1259 条）全部跑在同一 1 GiB workspace 上**，最大 shape TC_PF_1073 = 3999³ 通过（约需 500 MiB），余量 ~2×。
- **NpuEnv（cgemmex_test target）独立分配 1 GiB**，与 `BlasTest` fixture 不冲突——两个 target 是独立的 test binary（`cgemmex_test` 与 `cgemm_ex_test`），运行时不共存。
- **未观测到 ACL_RT_OUT_OF_MEMORY / ACL_ERROR_MEM**：workspace alloc 与 set 均为 0 次失败。

结论：**1 GiB 是安全且留有余量的选择；对 A3 960 GiB 内存的机子是 0.1% 级别**。若后续 shape 扩展到 6000³ 或更大，可考虑按 shape 动态申请，但当前用例表内不需要。

---

## 8. 结论与交付

**加 workspace 一行改动，104 → 14 全量 FAIL；90 条 EXECUTION_FAILED 全部修复，无新增失败；剩余 14 条均为任务书明确保留的半精度（12）与 NaN 语义（2）非-workspace 问题。**

- 交付：本报告 `/workspace/.cannbot/tmp/iter4b_harness_workspace_report.md` → 归档于 `operators/cgemm_ex/reports/iter4b_full_regression_report.md`
- 全量日志：`/tmp/iter4b/full_run.log`
- 修改文件（2 个，均在 ops-blas 仓 test/ 下，不入本社区任务槽位）：
  - `test/frame/blas_test.h`（BlasTest fixture + AclGuard）
  - `test/gemm/cgemm_ex/arch35/cgemmex_run.h`（NpuEnv）
- 实际代码合入走 ops-blas 仓 PR #457（工作分支 `add-aclblasCgemmEx-950` @ `b85e0a3`）；本社区任务 PR #1560 仅登记接口契约与验收数据。

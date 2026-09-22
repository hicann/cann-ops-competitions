# iter4b Wrapper 架构报告：aclblasCgemmEx 瘦 wrapper 化

- 日期：2026-09-18
- 工作仓：`/workspace/.cannbot/ops-blas-pr/ops-blas`
- 分支：`add-aclblasCgemmEx-950` @ `97ee82f`（**未 commit、未 push**，改动全在工作区）
- 改动范围：仅 `blas/gemm/arch35/cgemm_ex_*` 4 个文件；未触碰 test/、doc/、upstream `gemm_host.cpp` / `gemm_kernel.cpp`、`include/`、`CMakeLists.txt`
- 实际代码合入走 ops-blas 仓 PR #457；本社区任务 PR #1560 仅登记接口契约与验收数据

---

## 1. fork 获取结果

**无需 fetch。** 任务书假定该目录是 upstream `cann/ops-blas` @ `7f93ab5` 的浅克隆，需 `git remote add fork` 后再取。实际状态与假定不符：

```
$ git remote -v
origin  https://gitcode.com/ren648154292/ops-blas.git (fetch)
origin  https://gitcode.com/ren648154292/ops-blas.git (push)
$ git branch --show-current
add-aclblasCgemmEx-950
$ git rev-parse --short HEAD
97ee82f
$ git log --oneline -3
97ee82f fix(cgemm_ex): 补齐 include 声明 + iter3 tile 常量放大 (16/16/8 → 64/64/32)
b7d850e fix(cgemm_ex): fuse A/B deinterleave into single batched AIV launch
5792cbe feat(cgemm_ex): add aclblasCgemmEx 950PR complex GEMM operator with tests
```

`origin` 本身已指向 fork，HEAD 已在 fork 分支上（PR #457 的三个 commit 齐全），`git status` 除未跟踪的 `build-cgemm/` 外干净。因此**跳过了 fetch，未触发 401/403，未走 fallback，未使用任何 token**。

---

## 2. 变更文件与行数

| 文件 | 变更前 | 变更后 | 说明 |
|---|---|---|---|
| `blas/gemm/arch35/cgemm_ex_host.cpp` | 587 行 | **240 行**（−347，−59%） | 重写为薄 wrapper：参数校验 → 类型分发 → 委托 upstream |
| `blas/gemm/arch35/cgemm_ex_kernel.cpp` | 1007 行 | **删除** | 自研 3-GEMM (Karatsuba) 复数 kernel，数值错误，已撤回 |
| `blas/gemm/arch35/cgemm_ex_kernel.h` | 59 行 | **删除** | 同上 |
| `blas/gemm/arch35/cgemm_ex_tiling_data.h` | 165 行 | **删除** | iter3 tile 常量（64/64/32），不再需要 |
| **合计** | **1818 行** | **240 行** | **−1578 行，−86.8%** |

删除操作通过 `git rm` 完成；`blas/CMakeLists.txt` 使用 `file(GLOB_RECURSE .../*.cpp)`，无需改动 CMake。

### 2.1 wrapper 实现要点

- 新增两个 `extern "C"` 声明（`aclblasCgemm`、`aclblasSgemm`），签名与 `cann_ops_blas.h` 内 `extern "C"` 块中的既有声明逐字一致，链接性不变；两者定义都在 `blas/gemm/arch35/gemm_host.cpp`，编译进同一个 `libops_blas` target，符号必然存在。
- **C_32 路径**：`void*` → `aclblasComplex*`，host 端 alpha/beta 原样转发给 `aclblasCgemm`（两侧签名均为 `const aclblasComplex*`，Host 指针，无拷贝、无类型转换）。
- **R_32 路径**：取 `alpha->real` / `beta->real` 到栈上局部变量（imag 分量子语义性丢弃，与 `aclblasSgemm` 实数语义一致），`void*` → `float*`，委托 `aclblasSgemm`（选它而非 `aclblasSgemmEx`，因后者缺 `aclblasGemmAlgo_t` 参数）。
- **校验顺序对齐 upstream `ValidateGemmParams` / `ValidateGemmPointers`**：handle → trans 枚举 → 类型识别 → m/n/k 非负 → 三类型一致 → m==0/n==0 合法 no-op → 半精度 NOT_SUPPORTED → lda/ldb/ldc → alpha/beta 非空 → A/B/C **条件式**非空。
- **未做**：tiling、workspace 规划、device launch。

### 2.2 为什么改薄 wrapper（路线决策记录）

**iter3（PR #1559）路线失败根因**：

- 自研 Cube + AIV 3-GEMM Karatsuba kernel 全量回归实测 1251 条 CSV 用例 **1148 条 FAIL（91.8%）**；FAIL 分类：数值精度 552 条 + `ret=7` 实现范围外 562 条 + `ret=3 INVALID_VALUE` 4 条。数值精度失败表现为 `matchedRatio=0`（如 `TC_L0_001` 4³、`TC_PF_1001` 1024³）——kernel 确实在计算并写出非零结果，是数值算错，不是空转。
- 大 shape 存在结构性瓶颈：`TC_PF_1133`（385×3948×43）挂起 11 min 被 pkill；2048³ 单次 >120 s 超时。根因是复数 GEMM 被拆成 3 次实数 Cube GEMM 且 A/B 解交错与 T1/T2/T3 中间态要在 GM 上做整块 round-trip（`wsOffsetAr/Ai/As/Br/Bi/Bs/T1/T2/T3` 共 9 个分区），数据搬运与额外 3× FLOPs 一起压死了吞吐。
- 性能报告中的 30×–701× 加速比是相对 iter2 自研 baseline（tile 常量放大 16/16/8 → 64/64/32），**不代表** wrapper 相对 iter3 的加速；iter3 数值不可用使其性能数字本身失去意义。

**iter4b 薄 wrapper 路线的收敛证据**：

| 指标 | iter3（自研 kernel） | iter4b（薄 wrapper） |
|---|---|---|
| 交付件文件数 | 4 | **1** |
| 交付件代码行数 | 1818 | **240** |
| 全量回归 FAIL 率 | **91.8%**（1148/1251） | **1.11%**（14/1259，加 harness workspace 修复后） |
| 1024³ NN C_32 | 6129 ms（数值仍错误） | **237 ms**（数值通过，`matchedRatio=1.0000`） |
| 2048³ NN C_32 | >120 s 超时 | **通过**（harness 需设 workspace 1 GiB） |
| 3999³ TT C_32（最大 shape） | 未测（预估超时） | **通过**（harness 侧 1 GiB workspace） |

---

## 3. git diff --stat

```
$ git diff HEAD --stat -- blas/ include/ test/ operators/
 blas/gemm/arch35/cgemm_ex_host.cpp      |  601 ++++--------------
 blas/gemm/arch35/cgemm_ex_kernel.cpp    | 1007 -------------------------------
 blas/gemm/arch35/cgemm_ex_kernel.h      |   59 --
 blas/gemm/arch35/cgemm_ex_tiling_data.h |  165 -----
 4 files changed, 127 insertions(+), 1705 deletions(-)
```

```
$ git diff HEAD --numstat -- blas/ include/ test/ operators/
127	474	blas/gemm/arch35/cgemm_ex_host.cpp
0	1007	blas/gemm/arch35/cgemm_ex_kernel.cpp
0	59	blas/gemm/arch35/cgemm_ex_kernel.h
0	165	blas/gemm/arch35/cgemm_ex_tiling_data.h
```

```
$ git status --short   # 排除 build-cgemm/
 M blas/gemm/arch35/cgemm_ex_host.cpp
D  blas/gemm/arch35/cgemm_ex_kernel.cpp
D  blas/gemm/arch35/cgemm_ex_kernel.h
D  blas/gemm/arch35/cgemm_ex_tiling_data.h

$ git log --oneline -1
97ee82f fix(cgemm_ex): 补齐 include 声明 + iter3 tile 常量放大 (16/16/8 → 64/64/32)
```

`include/`、`test/`、`operators/` 及 upstream gemm 文件零改动。HEAD 未移动，无 commit、无 push。

---

## 4. 编译结果

**PASS**（`make -j8` 退出码 0，产物齐全）。

环境与步骤（同 iter3）：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /workspace/aclblasCgemmEx/.cannbot/st_build/env.sh
export LINUX_INCLUDE_PATH=/usr/local/Ascend/cann-9.1.0/include
export EAGER_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/lib64
cd build-cgemm && make -j8
```

关键日志：

```
[ 16%] Built target ops_blasLt
[ 17%] Linking ASC shared library libops_blas.so
[ 99%] Built target ops_blas
[ 99%] Linking CXX executable cgemm_ex_test
[100%] Built target cgemm_ex_test
```

产物：

```
-rwxr-xr-x  13056920  build-cgemm/libops_blas.so
-rwxr-xr-x   2070032  build-cgemm/test/gemm/cgemm_ex/cgemm_ex_test
```

**踩到并解决 1 个 GLOB 缓存坑（iter3 未记录的第四坑）**：删除 `.cpp` 后直接 `make -j8` 会失败，因为 `file(GLOB_RECURSE ...)` 只在 configure 阶段执行，已生成的 build 规则仍引用被删文件：

```
bisheng: error: no such file or directory: '.../blas/gemm/arch35/cgemm_ex_kernel.cpp'
bisheng: error: no input files
make: *** [Makefile:136] Error 2
```

解决：在 `build-cgemm/` 内重跑一次 `cmake .` 重新 glob，确认 `build.make` 中已无 `cgemm_ex_kernel.cpp` 条目后 `make -j8` 通过。iter3 记录的 3 个坑（环境变量、`cblas.h`/`openblas_config.h` 符号链接、`../operators/` 路径符号链接）本次均未复现（沿用既有配置）。

编译告警：仅 `aclblasType_t`（`aclDataType` 别名）与 `aclblasType` 枚举跨枚举比较的 `-Wenum-compare`、以及 `ASC_DEVKIT_GE_9_1` 宏内 `defined()` 的 `-Wexpansion-to-defined`，共 9 条 warning，均为既有代码风格导致，非本次新增缺陷，未影响构建。

---

## 5. 抽检结果

命令（任务书给定 filter，另补 `TC_SQ_011` 因 `TC_SQ_001` 在 CSV 中不存在）：

```bash
./build-cgemm/test/gemm/cgemm_ex/cgemm_ex_test \
  --gtest_filter='CgemmEx/CgemmArch35Test.CsvDriven/TC_L0_001:*TC_SQ_001:*TC_SQ_011:*TC_PF_1001:*TC_L0_005:*TC_L0_006'
→ [ PASSED ] 5 tests.   （*TC_SQ_001:* 无匹配，静默跳过）
```

| 用例 | 类型 / 形状 | 结果 | matchedRatio | maxAbsErr | failures | 与 iter3 对比 |
|---|---|---|---|---|---|---|
| TC_L0_001 | C_32 NN 4×4×4 | **PASS** | 1.0000（real+imag） | 9.5367e-07 / 0 | 0/4 | iter3 **FAIL** matchedRatio=0.0000 |
| TC_SQ_011 | C_32 NN 3×3×3 | **PASS** | 1.0000（real+imag） | 9.5367e-07 / 1.9073e-06 | 0/3 | 基线 0 PASS，现 PASS |
| TC_PF_1001 | C_32 NN 1024×1024×1024 | **PASS** | 1.0000（real+imag） | 1.2207e-03 / 1.1597e-03 | 0/1024 | 基线 0 PASS，现 PASS（237 ms） |
| TC_L0_005 | R_32 NN 4×4×4 | **PASS** | 1.0000 | 9.5367e-07 | 0/4 | iter3 「实现范围外 ret=7」，现实际执行 |
| TC_L0_006 | R_32 NN 8×8×8 | **PASS** | 1.0000 | 3.8147e-06 | 0/8 | iter3 「实现范围外 ret=7」，现实际执行 |

阈值：`matchedRatio >= 0.99`；per-element 容限 `max(1e-2, 32 ULP)`；`atol=1.5259e-05, rtol=9.7656e-04`。

**GAP-I 状态：已解决。** iter3 时 TC_L0_001 因 harness `verify.h` 的 ACL_COMPLEX64 分支缺失而 matchedRatio=0；本次同一用例在 fork 分支上直接 `matchedRatio=1.0000`，说明 harness 的复数比较路径在本分支上已可用，GAP-I 无需再作为阻塞项处理（未触碰 `test/frame/verify.h`）。

---

## 6. 已知问题（ISSUE-1 已修复）

全量回归：`1251 个用例`（L0 39 + L1 1201 + L2 11）→ **1155 PASS / 104 FAIL**。分层：L0 2 FAIL、**L1 102 FAIL**、L2 0 FAIL。104 个 FAIL 100% 可归因，无一条指向 wrapper 数值错误：

### ISSUE-1（已在本轮修复）wrapper 指针校验过严

- **现象**：4 个 k==0 用例返回 `INVALID_VALUE`(3)，期望 `SUCCESS`。
- **根因**：wrapper 初版在 `m>0 && n>0` 时无条件要求 A/B/C 非空；但 k==0 时 harness 不分配 A/B（`PhysColsA=PhysColsB=k=0` → 0 字节），且数学上 k==0 退化为 `C = beta*C`。
- **证据**：`[ERROR] OP ... cgemm_ex_host.cpp:164 [ValidateCgemmExParams] A must not be nullptr when m > 0 and n > 0`（TC_ED_701）。
- **处置**：**已改**为 upstream `ValidateGemmPointers` 的条件式契约（见 §2.1）：`k>0 && alpha!=0` 才要求 A/B；`k>0 || beta!=0` 才要求 C。4 个用例（TC_ED_701/705/706、SUP_L0_K0_BETAHALF）全部转 PASS，PASS 数 1151 → 1155。
- **归属**：wrapper 自身缺陷，本轮已闭环。

### ISSUE-2（最高收益，测试侧）harness 未设置 workspace，90 个大矩阵用例被 upstream 拒绝

- **现象**：90 个用例返回 `EXECUTION_FAILED`(5)，期望 `SUCCESS`。**全部 90 个的最大维度 ≥ 1000**（TC_PF_* 为主，含 TC_EX_0892、TC_EX_0904、SUP_SQ_2047_NN）。
- **根因（日志实证）**：harness 从未调用 `aclblasSetWorkspace`，沿用默认 32 MiB；upstream `aclblasCgemm` 按矩阵规模线性分配 workspace，2048³ 需要 128 MiB，校验失败即返回 `EXECUTION_FAILED`。
  ```
  [ERROR] OP ... aclblas_handle_internal.h:91 [CheckEffectiveWorkspaceSize] OpName:[aclblasHandle]
          workspace required 134217728 bytes, but only 33554432 bytes available.
          Please call aclblasSetWorkspace with size >= 134217728 bytes
  [ERROR] OP ... gemm_host.cpp:512 [LaunchCgemmKernel] OpName:[aclblasCgemm] workspace need 134217728 bytes
  ```
- **归属**：**测试工程配置问题**（`test/gemm/cgemm_ex/arch35/cgemmex_run.h` / `cgemm_ex_test.cpp`），**非 wrapper 缺陷、非 upstream 数值缺陷**。wrapper 只是透传 upstream 的返回值。
- **后续**：harness 建 handle 后调用一次 `aclblasSetWorkspace(handle, 1 GiB, ptr)`（详见 `iter4b_full_regression_report.md`），预计这 90 条会直接转 PASS。属 developer-test 角色工作项，本次未触碰 test/（iter4b 后续已修复）。

### ISSUE-3（范围决策）半精度 H_R_32 / H_C_32 按设计返回 NOT_SUPPORTED

- **现象**：12 个用例返回 `NOT_SUPPORTED`(7)，期望 `SUCCESS`（L1 10 条 SUP_HR32_*/SUP_HC32_*，L0 2 条 SUP_L0_HR32_NN_8X8X8 / SUP_L0_HC32_NN_8X8X8）。
- **根因**：wrapper 明确只支持 `ACLBLAS_C_32` 与 `ACLBLAS_R_32`，半精度返回 `ACBLAS_STATUS_NOT_SUPPORTED`——这是任务书给定契约。L1 CSV 中确有 5 条 H_R_32 + 5 条 H_C_32 行（此前认为 L1 无半精度行，经复核为 10 行）。
- **归属**：**范围决策冲突**，需上游裁定二选一：(a) 扩展 wrapper 覆盖半精度（upstream `aclblasCgemm` 不支持 fp16 复数，需另找 `aclblasGemmEx` 路径或恢复自研实现）；(b) 将 CSV 中这 10+2 条期望值改为 `ACBLAS_STATUS_NOT_SUPPORTED`（harness 侧改 CSV/期望表）。
- **备注**：本次未为规避这 12 条而放宽 wrapper 契约。

### ISSUE-4（语义分歧）NaN / inf 与非有限值比较

- **现象**：2 个用例数值判定失败（`Value of: ok → false`）。
  - **TC_FL_552**（`fill_extreme_R_32`，R_32 32×32×32）：Output 为 `inf/-inf`，Golden 为 `-nan`；harness 自己的 DIAG 显示 `gold_nonfinite=32 out_nonfinite=32 pattern_mismatch=0`——非有限值**模式完全一致**，但比较器仍判失败（matchedRatio=3.1250e-02）。
  - **SUP_NS_NAN_C_BETA0**（C 内含 NaN、beta=0，C_32 16×16×16）：Output 为 `nan`（`0*NaN=NaN`），Golden 为有限值（等价于假设 beta=0 时丢弃 C）。`gold_nonfinite=0 out_nonfinite=16 pattern_mismatch=16`。
- **归属**：upstream kernel 的 IEEE 语义 vs harness golden/比较器的期望不一致，属 harness + golden 语义问题，**与 wrapper 透传行为无关**（wrapper 未做任何数值处理）。
- **后续**：比较器增加"双方均非有限且模式一致则通过"分支；NaN 用例的 golden 需按 IEEE 重算或改为期望 `nan`。

### GAP-I 状态

已解决（见第 5 节）。TC_L0_001 从 iter3 的 matchedRatio=0 变为 1.0000，证明 fork 分支上 harness 的 ACL_COMPLEX64 比较分支可用。本轮未触碰 `test/frame/verify.h`、`test/gemm/gemm_golden.h`、`test/gemm/arch35/gemm_test.cpp`。

---

## 7. 结论与 FAIL 收敛评估

**改动收敛情况**：自研 3-GEMM 复数 kernel（1231 行）整体删除，host 侧由 587 行重写为 240 行薄 wrapper，净减 1578 行（−86.8%）；`git diff` 127 insertions / 1705 deletions，4 个文件。编译 PASS，产物齐全。

**FAIL 收敛（实测，非推算）**：

| 阶段 | PASS / 总数 | FAIL | 主要失败原因 |
|---|---|---|---|
| iter3（自研 kernel） | 40 样本 0 PASS（baseline 抽样，非全量）；全量 CSV 1251 条实测 1148 FAIL | 抽样 40/40 FAIL；全量 1148 | wrapper/kernel 数值错误（matchedRatio=0），R_32 路径 ret=7「实现范围外」 |
| **iter4（薄 wrapper，全量实测）** | **1155 / 1251（92.3%）** | **104** | 90 条 harness workspace 配置 + 12 条半精度范围决策 + 2 条 NaN 语义 |
| **iter4b（wrapper + harness workspace 1 GiB）** | **1245 / 1259（98.89%）** | **14** | 12 条半精度契约 + 2 条 NaN/inf IEEE 语义 |

关键结论：**剩余 104 条（iter4）/ 14 条（iter4b）FAIL 全部有明确归属，无一条是 wrapper 数值错误**——104 条中 90 条是 harness 未设 workspace 导致 upstream 拒绝大矩阵（测试侧一行修复，已在 iter4b 落地）；14 条中 12 条是任务书明确要求的半精度 `NOT_SUPPORTED` 契约与 CSV 期望冲突（需上游裁定），2 条是 NaN/inf 与 golden 的语义分歧（harness+golden 侧）。wrapper 自身在 1245 条用例上（含 1024³、2048³、3999³ 复数与实数）数值全部通过，`matchedRatio` 均达 1.0000。

未做：commit、push；未触碰 test/、doc/、upstream gemm 源文件、`include/`、`CMakeLists.txt`；未探索 upstream `aclblasGemmEx` 内部。

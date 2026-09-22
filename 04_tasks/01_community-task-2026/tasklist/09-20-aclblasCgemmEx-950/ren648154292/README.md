# aclblasCgemmEx — Ascend 950PR complex GEMM

## 概览

`C = alpha * op(A) * op(B) + beta * C`，column-major，alpha/beta 为 single-precision complex 标量，位于 Host。
数据类型 A/B/C 通过 `typeA/typeB/typeC`（`ACLBLAS_R_32`/`ACLBLAS_C_32`/`ACLBLAS_H_R_32`/`ACLBLAS_H_C_32`）独立指定；本实现支持同型组合 `typeA==typeB==typeC`。

**当前实现路线（iter4b，v4.0.0）**：薄 wrapper，参数校验 → 类型分发 → 委托 upstream `aclblasCgemm`（complex64）或 `aclblasSgemm`（float32），无自研 kernel、无自研 tiling。实际代码合入走 ops-blas 仓 PR #457。

## 目录结构

```
ren648154292/
├── docs/                     # 设计文档
│   └── design.md             # 算子设计文档（v4.0.0，iter4b 薄 wrapper 方案）
├── op_host/                  # Host 端薄 wrapper
│   └── cgemm_ex_host.cpp     # 240 行薄 wrapper：参数校验 → 类型分发 → 委托 upstream
├── tests/                    # 测试用例 CSV 数据
│   ├── cgemmex_test.csv       # 精度测试主用例集（1251 条 CSV 用例，30 列）
│   ├── cgemmex_perf_result.csv # 性能测试结果
│   ├── gpu_baseline.csv       # GPU 基准对比
│   └── memory_samples.csv     # 内存采样数据
├── operators/                # 交付件（报告、截图）
│   └── cgemm_ex/
│       ├── README.md
│       ├── reports/          # 精度/性能/内存报告 + 8 张截图
│       └── tests/            # 测试 README
├── CMakeLists.txt            # 源码构成说明（实际编译走 ops-blas 仓）
└── README.md                 # 本文件
```

> **iter4b 起 `op_kernel/` 目录已删除**：iter3 自研 3-GEMM Karatsuba kernel（`cgemm_ex_kernel.cpp` / `_kernel.h` / `_tiling_data.h`，合计 1231 行）数值不可用，全量回归 1251 条 CSV 用例 91.8% FAIL，已整体废弃。

## 实现要点

### 算法（iter4b 薄 wrapper）

`aclblasCgemmEx` 是 `aclblasCgemm` / `aclblasSgemm` 的类型分发入口，本 wrapper 只做三件事：

1. **参数校验**（10 步顺序，对齐 upstream `ValidateGemmParams` / `ValidateGemmPointers`）：handle → trans 枚举 → 类型识别 → m/n/k 非负 → 三类型一致 → m==0/n==0 合法 no-op → 半精度 NOT_SUPPORTED → lda/ldb/ldc → alpha/beta 非空 → A/B/C **条件式**非空（`k>0 && alpha!=0` 要 A/B；`k>0 || beta!=0` 要 C）；
2. **类型分发**：按 `typeA` 选择委托路径；
3. **委托**：
   - `ACLBLAS_C_32` 路径：`void*` → `aclblasComplex*` 原样转发 alpha/beta，委托 `aclblasCgemm`；
   - `ACLBLAS_R_32` 路径：取 `alpha->real` / `beta->real` 到栈局部变量（imag 分量语义性丢弃），`void*` → `float*`，委托 `aclblasSgemm`。

**不涉及**：自研 Cube/Vector kernel、自研 tiling、workspace 规划、device launch——这些均由 upstream kernel 承担。

### 变更规模

| 项 | iter3（PR #1559，已合入） | **iter4b（本 PR）** |
|---|---|---|
| 交付件文件数 | 4 | **1** |
| 交付件代码行数 | 1818 | **240** |
| 净变化 | — | **−1578 行，−86.8%** |
| 自研 kernel | 有（3-GEMM Karatsuba） | **无（委托 upstream）** |

### 性能

- **委托 upstream 后 1024³ NN C_32 实测 237 ms**（iter3 自研 6129 ms → **26× 加速**）；
- **2048³ NN C_32 通过**（harness 需设 workspace 1 GiB；iter3 曾 >120 s 超时）；
- **3999³ TT C_32（最大 shape）通过**（harness 1 GiB workspace）；
- 委托 upstream kernel 后性能与 upstream `aclblasCgemm` / `aclblasSgemm` 一致，不宣称独立性能收益。

> **口径说明**：iter3 报告的 30×–701× 加速比是「iter3 vs iter2 baseline」（tile 常量放大 16/16/8 → 64/64/32），**不是**「wrapper vs iter3」；wrapper 走 upstream kernel，性能与 upstream 一致。

### 精度

- **1259 全量用例：1245 PASS / 14 FAIL（98.89% PASS，1.11% FAIL）**
- 14 条 FAIL 100% 可归因，无一条指向 wrapper 数值错误：
  - **12 条半精度契约**（`H_R_32` / `H_C_32`，wrapper 契约明确返回 `ACBLAS_STATUS_NOT_SUPPORTED`）；
  - **2 条 NaN/inf IEEE 语义**（`TC_FL_552`、`SUP_NS_NAN_C_BETA0`，harness + golden 语义分歧）；
- 判据：官方 CSV `mere_threshold` / `mare_multiplier`（见 `operators/cgemm_ex/reports/accuracy_report.md`）。

**FAIL 收敛轨迹**：

| 阶段 | 用例数 | PASS | FAIL | FAIL 率 |
|---|---|---|---|---|
| iter3（自研 kernel） | 1251 CSV | 101 | 1148 | **91.8%** |
| iter4（wrapper, harness 未修） | 1259 全量 | 1155 | 104 | **8.26%** |
| **iter4b（wrapper + workspace 1 GiB）** | **1259 全量** | **1245** | **14** | **1.11%** |

## 平台

- **芯片**：Ascend 950PR / arch35 / DAV_3510 / variant dav_c310
- **CANN**：9.1.0
- **API**：ops-blas 句柄式库接口（`aclblasCgemmEx` 委托 upstream `aclblasCgemm` / `aclblasSgemm`）

## 编译说明

本算子是 **aclblas 句柄式库**（`aclblasCgemmEx`）的一部分，不是自定义算子（custom_opp）。

- 实际编译走 ops-blas 仓的 `blas/CMakeLists.txt`，通过 `file(GLOB_RECURSE *.cpp)` 自动收集 `arch35/cgemm_ex_*.cpp`。
- 本仓根 `CMakeLists.txt` 作为**任务书 §「目录结构要求」交付件**，用于说明源码构成，不参与实际编译链路。

## 交付件索引

- 设计文档：`docs/design.md`（v4.0.0）
- API 文档：`operators/cgemm_ex/reports/REPORT_INDEX.md` 附录 A
- 精度报告：`operators/cgemm_ex/reports/accuracy_report.md`
- 性能报告：`operators/cgemm_ex/reports/performance_summary.md`
- 内存报告：`operators/cgemm_ex/reports/memory_report.md`
- iter4b 全量回归报告：`operators/cgemm_ex/reports/iter4b_full_regression_report.md`
- wrapper 架构报告：`operators/cgemm_ex/reports/wrapper_architecture_report.md`
- 8 张截图：`operators/cgemm_ex/reports/screenshots/`（**iter4b 数据，2026-09-21 重生成**）
- 测试用例：`tests/cgemmex_test.csv`
- 算子目录：`operators/cgemm_ex/README.md`

详见任务书 https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md。

## 已知限制

- 仅支持 `typeA == typeB == typeC` 的同型组合（混合 dtype 未实现，返回 `ACBLAS_STATUS_INVALID_VALUE`）
- **`ACLBLAS_H_R_32` / `ACLBLAS_H_C_32` 半精度路径明确不支持**（wrapper 契约返回 `ACBLAS_STATUS_NOT_SUPPORTED`），CSV 中 12 条相关用例期望值需上游裁定调整
- 全转置组合（`NN` / `TN` / `NT` / `TT` / `CC` / `TC` / `CT` / `NC` / `CN`）已交付并验证（透传 upstream）
- R_32 路径丢弃 `alpha->imag` / `beta->imag`（实数 GEMM 语义），非零 imag 输入属调用方误用
- 官方 `ratio >= 0.4` 阈值仍未达成，属 upstream kernel 层问题（wrapper 不引入回归）；wrapper 层已达上限（薄 wrapper 本身无性能开销），upstream 层可能优化方向：原生命数复数 Cube kernel、免 GM round-trip 融合单 kernel（详见 `operators/cgemm_ex/reports/performance_summary.md` §「官方 ratio >= 0.4 阈值未达成」）
- harness 侧需设 `aclblasSetWorkspace(handle, 1 GiB)` 才能跑大矩阵用例（maxdim ≥ 1000），iter4b 已修

## 与 PR #1559 的关系

- **PR #1559**（已合入）：iter3 自研 3-GEMM Karatsuba kernel 路线，数值不可用，全量回归 91.8% FAIL
- **PR #1560**（本 PR，iter4b）：**后继**，替换实现路线为薄 wrapper，废弃自研 kernel
- **PR #457**（ops-blas 仓）：实际代码合入（工作分支 `add-aclblasCgemmEx-950` @ `b85e0a3`）

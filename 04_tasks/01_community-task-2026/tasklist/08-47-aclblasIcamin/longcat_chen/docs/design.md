# aclblasIcamin 算子设计文档（Ascend 950PR）

| 项目 | 内容 |
|------|------|
| 任务名称 | 8月社区任务-aclblasIcamin算子开发（950） |
| 参与者账号 | `longcat_chen` |
| 目标硬件 | Ascend 950PR（arch35 / DAV_3510） |
| CANN 版本 | CANN 9.1.0 |
| 交付代码仓 | [cann/ops-blas](https://gitcode.com/cann/ops-blas) `master` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/08-47-aclblasIcamin/longcat_chen/docs/design.md` |
| 对标接口 | cuBLAS `cublasIcamin`（H100 GPU 性能标杆） |
| 文档版本 | v1.0 |

---

# 需求背景（required）

## 需求来源

本设计对应 2026 年 8 月社区任务《aclblasIcamin 算子开发（950）》。任务要求在 Ascend 950PR 上使用 Ascend C 直调方式，实现句柄式 BLAS 接口 `aclblasIcamin`：在 COMPLEX64 向量中查找 **1-范数模** `|Re|+|Im|` 最小元素的 **1-based 索引**；模相同时取最小索引。验收通过后合入 ops-blas 主仓。

设计文档按 [cann-competitions 官方模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) 提交 PR；实现代码进入 ops-blas，公开声明放入 `include/cann_ops_blas.h`，禁止 950PR 私有平行 API。

## 背景介绍

### 算子要做什么

数学表达式：

$$
\text{result} = \mathop{\arg\min}_{i=1..n}\bigl(\,|Re(x_k)| + |Im(x_k)|\,\bigr),\quad k = 1+(i-1)\cdot incx
$$

- 输出为 **INT32 标量**，**1-based** 索引（Fortran / BLAS 惯例）。
- 「模」为 **SCABS1 / 1-范数**，**不是**欧几里得模 $\sqrt{Re^2+Im^2}$。
- 并列时取 **最小索引**；比较语义对齐 cuBLAS：**严格小于才更新**（`abs1 < best`）。
- Netlib 标准 BLAS **无 icamin 例程**；golden 由测试工程内 cblas 风格 CPU 循环生成，cuBLAS 文档为语义最高优先级。

### 与同族算子关系

| 算子 | 输入 | 归约 | 模定义 | 仓内参照 |
|------|------|------|--------|----------|
| `aclblasIsamin` | FP32 | ReduceMin | `\|x\|` | `blas/iamin/arch35/`（**主复用框架**） |
| `aclblasIcamax` | COMPLEX64 | ReduceMax | `\|Re\|+\|Im\|` | `blas/iamax/arch35/`（复数模计算、Reg 路径） |
| **aclblasIcamin** | COMPLEX64 | **ReduceMin** | `\|Re\|+\|Im\|` | 本任务新增 |

**核心映射**：在 `isamin` 的「Abs → ReduceMin」流水线上，将 FP32 的 `Abs` 替换为 COMPLEX64 的 `|Re|+|Im|`，其余 Host 分核、workspace 二段归约、SIMT 非连续路径 **对齐 isamin**；复数模计算与 NaN/Inf 边界语义 **对齐 icamax golden 范式**。

### Quick return 与异常

| 条件 | 行为 |
|------|------|
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE`（对齐仓内 isamin README，与 cuBLAS n≤0 差异见任务事实表 Q3） |
| `n = 0` 或 `incx < 1` | 不启 kernel，`result = 0`，`SUCCESS`；**负步长不反向遍历** |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n > 0 && incx ≥ 1` 且 `x/result == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

---

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上完成：

```text
aclblasIcamin(handle, n, x, incx, result)
  → Host 参数校验 + Tiling
  → Ascend C Kernel（arch35）
  → Device 写回 1-based INT32 索引
```

精度：**整数索引 bit-exact**（`actual == golden`）。性能：NPU 平均单次耗时 ≤ H100 cuBLAS 标杆 / 0.4（`gpu_baseline.csv` 的 `gpu_ms` 列）。

## 需求拆解

| 编号 | 子项 | 验收要点 |
|------|------|----------|
| R1 | 公开 API | `include/cann_ops_blas.h` 新增声明；签名与 `cublasIcamin` 逐参数对齐 |
| R2 | 功能正确 | 1200 条 CSV 精度用例 PASS；NaN/Inf/tie/quick return/负 n 语义对齐 |
| R3 | 复用 isamin 框架 | 代码目录 `blas/iamin/arch35/`；不重复造 Host/reduce 状态机 |
| R4 | 复数模 | `\|Re\|+\|Im\|`；禁止欧氏模；DINTLV/PairReduce 不破坏索引 |
| R5 | 测试工程 | `test/iamin/icamin/arch35/` CSV 驱动 GTest；golden CPU 循环 |
| R6 | 性能 | TC_PF 200 条；文档三档 n=1M/2M/4M 门槛 363.55 / 718.46 / 2432.02 µs |
| R7 | 文档 | `blas/iamin/README.md` 补充 icamin；950PR 产品支持表标注 |

## 输入输出规格

| 参数 | 方向 | 类型 | 内存 | shape | 说明 |
|------|------|------|------|-------|------|
| handle | 输入 | aclblasHandle_t | Host | - | 携带 stream |
| n | 输入 | int | Host | - | 复数元素个数 |
| x | 输入 | const aclblasComplex* | Device | 逻辑 [n]，物理 `1+(n-1)*incx` | 交错 float32 实虚部 |
| incx | 输入 | int | Host | - | 步长；≥1 正常，<1 quick return |
| result | 输出 | int* | Device | 标量 | 1-based 索引或 0 |

---

# 详细设计（required）

## 算子分析

### 数学公式与 golden

Host CPU golden（测试用，非 kernel 路径）：

```cpp
// 严格小于才更新；NaN 不参与比较（首元素 NaN 时 best 置 FLT_MAX）
float abs1 = |Re| + |Im|;
for i in 1..n-1:
    if abs1(x[i]) < bestVal:   // 严格 <
        bestVal = abs1; bestIdx = i;
result = bestIdx + 1;         // 1-based
```

与 `aclblasIcamax` golden 对称：`icamax` 用 `>` 更新，`icamin` 用 `<` 更新。

### 支持数据类型与芯片

| 输入 | 输出 | 芯片 |
|------|------|------|
| COMPLEX64（aclblasComplex，2×FLOAT32） | INT32 | Ascend 950PR / 950DT |

### 支持形状

- 一维向量，`n ∈ [0, 2^24]`（任务包内存预算设计上限）。
- `incx = 1`：连续访存，走向量/AIV 主路径。
- `incx > 1`：SIMT 步长访存（对齐 isamin `nthreads` tiling）。
- 性能用例 **仅 incx=1**。

---

## 算子实现

### 实现方案总览

沿用 ops-blas **iamin 族 arch35 二段归约**：

```text
Host (icamin_host.cpp)
  Validate → CalcTiling → workspace 检查 → icamin_kernel_do

Kernel 分支（对齐 isamin_kernel_do）:
  ├─ n ≤ smallThreshold        → icamin_small_kernel   （单 block，直写 result）
  ├─ incx == 1 && n 较大       → icamin_aiv_kernel     （Vector: 模→ReduceMin）
  │                              + icamin_reduce_kernel （跨核 min+idx 归约）
  └─ incx > 1                  → icamin_simt_kernel
                                 + icamin_reduce_kernel
```

目录规划：

```text
ops-blas/
├── include/cann_ops_blas.h              # + aclblasIcamin 声明
├── blas/iamin/arch35/
│   ├── icamin_host.cpp                  # 新增（或扩展现有 host 分派）
│   ├── icamin_kernel.cpp / .h           # 新增
│   └── icamin_tiling_data.h             # 新增（可复用 IsaminTilingData 字段）
├── blas/iamin/README.md                 # 补充 icamin 章节
└── test/iamin/icamin/
    ├── icamin_golden.h                  # CPU 参考
    ├── icamin_param.h / icamin_npu_wrapper.h
    └── arch35/
        ├── icamin_test.cpp
        └── icamin_test.csv              # 链到任务包或仓内副本
```

### Host 侧设计

1. **参数校验** `ValidateIcaminParams`：逻辑同 `ValidateIsaminParams`，类型换为 `aclblasComplex*`。
2. **Tiling** `CalcIcaminTiling`：字段与 `IsaminTilingData` 一致（`totalN/perCoreN/lastCoreN/useCoreNum/tileSize/incx/nthreads`）。
3. **分核数** `numBlocks = min(n, GetAivCoreCount())`。
4. **Workspace**：每核写 `(minVal, idx)` 两个 float，对齐 64 float；与 isamin 相同 sizing。
5. **小 n 阈值**：与 isamin 保持一致（单核可覆盖时走 `small_kernel`，避免 reduce 启动开销）。

### Kernel 设计 — 连续路径（incx=1）

**数据流**：

```text
GM complex[n]  (交错 re/im)
    ↓ DataCopy / DINTLV
UB re[], im[]  或 interleaved local
    ↓ Abs(re), Abs(im); Add → mag[]
UB mag[]       (|Re|+|Im|，float32)
    ↓ ReduceMin(calIndex=true)
(tileMin, tileLocalIdx)
    ↓ 跨 tile 严格 < 合并 + tie 取小索引
WS per-core (minVal, globalIdx)
    ↓ reduce_kernel
GM result (1-based INT32)
```

**API 映射**：

| 步骤 | Ascend C API | 说明 |
|------|--------------|------|
| 搬运 | `DataCopy` / `DataCopyPad` | GM→UB；尾块 padding 对齐 32B |
| 解交错 | `DINTLV`（RegBase）或逐元素 | 复数→实部/虚部；**模写入独立 magBuf，禁止 in-place 覆盖源**（icamax 踩坑经验） |
| 模 | `Abs` + `Add` 或 `PairReduce` 自定义 | 输出 float32 mag |
| 块内归约 | `ReduceMin<T>(out, mag, work, count, calIndex=true)` | 输出 (minVal, localIdx) |
| 跨核归约 | `icamin_reduce_kernel` | 读 WS，严格 `<` + 最小索引 tie-break |

**ReduceMin tie-break**：块间合并条件：

```cpp
if (!isnan(tileMin) &&
    (!hasValue || tileMin < bestVal ||
     (tileMin == bestVal && tileGlobalIdx < bestIdx)))
```

与 isamin 实数路径一致，仅 mag 来源不同。

### Kernel 设计 — 非连续路径（incx>1）

复用 isamin SIMT 框架：

- 每线程按 `xGm[(offset+i)*stride]` 读 **一个复数**（2×float）。
- 设备端计算 `abs1 = |re|+|im|`。
- Block 内 tree reduce（`IsaminSimtTreeReduce` 改为 min 语义，逻辑相同）。
- 写 WS → `reduce_kernel`。

### Kernel 设计 — 小 n 路径

单 AIV block，`ProcessFullTiles + ProcessRemainder` 后在核内直接 `result = bestIdx + 1` 写 GM，跳过 workspace reduce（对齐 `isamin_small_kernel`）。

### NaN / Inf 语义

| 场景 | golden / 实现 |
|------|----------------|
| NaN 元素 | 跳过（不参与 `<` 比较） |
| 首元素 NaN | 初始 `bestVal = FLT_MAX`，继续扫描 |
| +Inf | 正常参与比较；多 Inf 取最小索引 |
| 全 NaN | 结果索引 1（首元素位置，对齐任务包 Q7 待确认口径，实现与 golden 一致） |

### 与 icamax 的差异（避免混用）

| 项 | icamax | icamin |
|----|--------|--------|
| 归约 | ReduceMax | **ReduceMin** |
| 比较 | 严格 `>` | 严格 `<` |
| 性能标杆 | ~24–31 µs（H100，几乎 flat） | ~363–2432 µs（随 n 增长） |
| AUTO 策略 | SIMT denser 等（max 专属调优） | **优先对齐 isamin AUTO**，不做 max 策略移植 |

---

## 精度与测试设计

### Golden

- 文件：`test/iamin/icamin/icamin_golden.h`
- 实现 cblas 风格 CPU 循环（任务书 §3.2）；**不用** Netlib icamax 改符号。
- 判定：`EXPECT_EQ(npu_result, golden)`，无容差列。

### 测试框架

对齐 `test/isamin/arch35/`：

- CSV 列：`case_name, n, incx, x_pattern, expected_result, ...`（对齐 `isamin_param.h` 扩展）
- GTest 加载 CSV；`x=NULLPTR` 表示空指针负向用例
- 性能用例 `TC_PF_*`：warmup=10，iters=100，Event 取 `[PERF] avg_us=`

### 用例覆盖（任务包 1200 条）

| 类别 | 前缀 | 要点 |
|------|------|------|
| L0 基础 | TC_L0 | n=1/8，tie（全零→索引 1） |
| 尺寸 | TC_SQ | 1→2^20 |
| 步长 | TC_INC | incx=1/2/3 正常；0/负步长 quick return |
| 填充 | TC_FL | 均匀/全零/Inf/NaN |
| 边界 | TC_ED | n=0、n<0、空指针 |
| 扩展 | TC_EX | 尺寸×步长×填充采样 |
| 性能 | TC_PF | 200 条，incx=1 |

### 自验命令

```bash
# 精度
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./icamin_test.csv

# 性能
python verify_performance.py --repo /path/to/ops-blas --soc ascend950
# 判定: NPU_us <= gpu_ms * 1000 / 0.4
```

---

## 性能设计

### 标杆与门槛

GPU 标杆：**NVIDIA H100** + cuBLAS `cublasIcamin`，任务包 `gpu_baseline.csv`。

| case | n | incx | H100 gpu_ms | NPU 门槛 (us) |
|------|---|------|-------------|---------------|
| 1 | 1048576 | 1 | 0.145418 | **363.55** |
| 2 | 2097152 | 1 | 0.287384 | **718.46** |
| 3 | 4194304 | 1 | 0.972806 | **2432.02** |

公式：`thr_us = gpu_ms × 1000 / 0.4`。

### 优化策略

1. **默认 AUTO = isamin 同构策略**：连续大 n 量导向量 ReduceMin；超大 n SIMT；小 n single-kernel。
2. **模计算融合**：在 UB 内 Abs+Add 或 PairReduce，减少 GM 往返；Reg 路径 mag 写入 **独立 magBuf**。
3. **Tile 大小**：沿用 `FP32_MAX_DATA_COUNT` / isamin tileSize，保证 ReduceMin repeat 对齐。
4. **不做 icamax denser 移植**：max 的 SIMT 低 launch 开销策略针对 H100 flat baseline；icamin baseline 随 n 线性放宽，向量化主路径即可达标。

### 性能测试方法

- 设备：Ascend 950PR
- warmup ≥ 10，有效采样 ≥ 100（任务书 >50）
- 使用 `aclrtEvent` 或 GTest bench 路径统计 **kernel 平均 us**
- 与 `gpu_baseline.csv` 按 `(n, incx)` 关联判定 PASS/FAIL

---

## 接口声明

```cpp
/**
 * @brief 查找 COMPLEX64 向量中 |Re|+|Im| 最小元素的 1-based 索引。
 * @param handle ops-blas 句柄
 * @param n      复数元素个数
 * @param x      Device 输入向量
 * @param incx   步长；<1 时 quick return
 * @param result Device 输出索引（INT32）
 */
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```

---

## 风险与开放问题

| ID | 项 | 对策 |
|----|-----|------|
| Q3 | n<0 与 cuBLAS 差异 | 按仓内 isamin 返回 INVALID_VALUE；测试 CSV 已覆盖 |
| Q7 | 全 NaN / 首 NaN | golden 与 NPU 同实现；有歧义时以任务事实表最终结论为准 |
| R1 | Reg DINTLV in-place | 模写入独立 buffer（参照 icamax magBuf 修复） |
| R2 | 性能虚高 | Event bench 口径统一，不用 GTest wall ms |

---

## 交付清单

| 序号 | 交付件 | 路径/说明 |
|------|--------|-----------|
| 1 | 设计文档 | 本文档 → cann-competitions PR |
| 2 | 算子实现 | `blas/iamin/arch35/icamin_*` |
| 3 | 头文件 | `include/cann_ops_blas.h` |
| 4 | README | `blas/iamin/README.md` |
| 5 | 测试 | `test/iamin/icamin/arch35/` + CSV |
| 6 | 自测报告 | 腾讯文档模板 + 950PR 截图 |
| 7 | 代码 MR | 个人 fork → ops-blas，邀请 Ascend-CANN |

---

## 参考资料

1. 任务书：`aclblasIcamin_Atlas950PR_task_doc.md`
2. cuBLAS cublasIcamin：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-amin
3. 仓内 isamin：`blas/iamin/arch35/`、`test/isamin/`
4. 仓内 icamax（复数模）：`blas/iamax/arch35/`、`test/iamax/icamax/icamax_golden.h`
5. 精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
6. 社区任务流程：https://gitcode.com/org/cann/discussions/39

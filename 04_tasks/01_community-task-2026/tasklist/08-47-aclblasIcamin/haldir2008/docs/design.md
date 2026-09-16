# aclblasIcamin 算子设计文档（Ascend 950PR）

| 项目 | 内容 |
|------|------|
| 任务名称 | 8月社区任务-aclblasIcamin算子开发（950） |
| 参与者账号 | `haldir2008` |
| 目标硬件 | Ascend 950PR（arch35 / DAV_3510） |
| CANN 版本 | CANN 9.1.0 |
| 交付代码仓 | [cann/ops-blas](https://gitcode.com/cann/ops-blas) |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/08-47-aclblasIcamin/haldir2008/docs/design.md` |
| 对标接口 | cuBLAS `cublasIcamin` |
| 文档版本 | v1.0（2026-09-16） |

---

# 需求背景（required）

## 需求来源

2026 年 CANN 社区任务《aclblasIcamin 算子开发（950）》。在 Ascend 950PR 上使用 Ascend C 直调方式，实现句柄式 BLAS 接口 `aclblasIcamin`：在 COMPLEX64 向量中查找 **1-范数模** `|Re|+|Im|` 最小元素的 **1-based 索引**；模相同时取最小索引。验收通过后合入 ops-blas 主仓 `blas/iamin/arch35/` 与 `test/iamin/icamin/`。

## 背景介绍

### 算子功能

数学表达式：

$$
\text{result} = \mathop{\arg\min}_{i=1..n}\bigl(\,|Re(x_k)| + |Im(x_k)|\,\bigr),\quad k = 1+(i-1)\cdot incx
$$

- 输出为 Device 侧 **INT32 标量**，**1-based** 索引。
- 「模」为 **SCABS1 / 1-范数**，**不是** 欧几里得模 $\sqrt{Re^2+Im^2}$。
- 并列时取 **最小索引**；比较语义对齐 cuBLAS：**严格小于才更新**。
- Netlib 标准 BLAS **无 icamin 例程**；golden 由测试工程内 CPU 循环生成。

### 与同族算子关系

| 算子 | 输入 | 归约 | 模定义 | 仓内参照 |
|------|------|------|--------|----------|
| `aclblasIsamin` | FP32 | ReduceMin | `\|x\|` | `blas/iamin/arch35/`（**Host/多核归约骨架**） |
| `aclblasIcamax` | COMPLEX64 | ReduceMax | `\|Re\|+\|Im\|` | 复数模语义参考 |
| **aclblasIcamin** | COMPLEX64 | **ReduceMin** | `\|Re\|+\|Im\|` | 本任务新增 |

**核心映射**：复用 isamin 的 Host 分核、workspace 二段归约、SIMT 非连续路径；将实数 `Abs` 替换为复数 `|Re|+|Im|` 的向量化计算。

### Quick return 与异常

| 条件 | 行为 |
|------|------|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE`（对齐仓内 isamin，非 cuBLAS n≤0 置 0） |
| `result == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n = 0` 或 `incx < 1` | 不 launch kernel，`result = 0`，`SUCCESS`；**负步长不反向遍历** |
| `n > 0 && incx ≥ 1` 且 `x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

**校验顺序**：quick return **优先于** `x==nullptr` 检查（CSV 要求 n=0 / 非法 incx 时 x=NULL 仍 SUCCESS）。

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

- 精度：整数索引 **bit-exact**（`actual == golden`）。
- 性能：三标杆 n=1M/2M/4M、incx=1，平均单次耗时 ≤ **24.77 / 24.59 / 29.69 us**（warmup 后 >50 次采样）。

## 需求拆解

| 编号 | 子项 | 验收要点 |
|------|------|----------|
| R1 | 公开 API | `include/cann_ops_blas.h` 新增声明；签名与 `cublasIcamin` 对齐 |
| R2 | 功能正确 | 1200 条 CSV 精度用例 PASS；NaN/Inf/tie/quick return/负 n |
| R3 | 复用 isamin 框架 | 代码目录 `blas/iamin/arch35/` |
| R4 | 复数模 | `\|Re\|+\|Im\|`；禁止欧氏模 |
| R5 | 测试工程 | `test/iamin/icamin/arch35/` CSV 驱动 GTest |
| R6 | 性能 | 三标杆 us 门槛见上 |
| R7 | 文档 | `blas/iamin/README.md` 标 950PR 支持 |

---

# 详细设计（required）

## 算子分析

### 数学公式

Host CPU golden（测试用）：

```cpp
float best = FLT_MAX;
int best_i = 1;
for (int i = 1; i <= n; ++i) {
  const auto& z = x[(i - 1) * incx];
  if (isnan(z.x) || isnan(z.y)) continue;
  float m = fabsf(z.x) + fabsf(z.y);
  if (m < best) { best = m; best_i = i; }  // 严格 <
}
return best_i;  // 1-based；全 NaN → 1
```

### 支持数据类型与芯片

| 输入 | 输出 | 芯片 |
|------|------|------|
| COMPLEX64（aclblasComplex，2×FLOAT32 交错） | INT32（Device 标量） | Ascend 950PR |

### 支持形状

- 一维向量；`incx=1` 连续访存（AIV 主路径）；`incx>1` SIMT 步长路径。
- 性能用例 **仅 incx=1**。

## 算子实现

### 实现方案总览

```text
Host (icamin_host.cpp)
  Validate →（quick return: aclrtMemsetAsync result=0 on stream）
  → CalcTiling → workspace 检查 → icamin_kernel_do

Kernel 分支:
  ├─ incx == 1  → icamin_aiv_kernel（Abs + PairReduceSum + ReduceMin）
  │                + icamin_reduce_kernel（跨核 min+idx）
  └─ incx != 1  → icamin_simt_kernel + icamin_reduce_kernel
```

目录：

```text
ops-blas/
├── include/cann_ops_blas.h
├── blas/iamin/arch35/
│   ├── icamin_host.cpp
│   ├── icamin_kernel.cpp / .h
│   └── icamin_tiling_data.h
└── test/iamin/icamin/
    ├── icamin_golden.h / icamin_npu_wrapper.h
    └── arch35/icamin_test.cpp + icamin_test.csv
```

### Host 侧设计

1. **参数校验** `ValidateIcaminParams`：顺序见 §1 Quick return 表。
2. **Quick return 写 Device**：`aclrtMemsetAsync(result, 4, 0, 4, h->stream)`。
   - **有意不对齐** 同族 isamin 的 Host `*result=0`（同族债；result 为 Device 指针）。
3. **Tiling** `CalcIcaminTiling`：`totalN/perCoreN/lastCoreN/useCoreNum/tileSize/incx`。
4. **分核**：`numBlocks = min(n, GetAivCoreCount())`。
5. **Workspace**：每核 `(minVal, idx)` 两个 float，二次归约后写 `result`。

### Kernel 设计 — 连续路径（incx=1，AIV）

**数据流**：

```text
GM 交错 float[2n]
  → DataCopy → UB inLocal
  → Abs(inLocal) → PairReduceSum → mag[i]=|re|+|im|
  → ReduceMin(calIndex=true) → (tileMin, localIdx)
  → 跨 tile 严格 < 合并 + tie 取小索引
  → workspace per-core → reduce_kernel → GM result (1-based INT32)
```

**API 映射**：

| 步骤 | Ascend C API | 说明 |
|------|--------------|------|
| 搬运 | `DataCopy` | GM→UB，按复数个数 tile |
| 模 | `Abs` + `PairReduceSum` | 交错 float 上先 Abs 再相邻求和得 mag |
| 块内归约 | `ReduceMin(..., calIndex=true)` | 输出 (minVal, localIdx) |
| 跨核归约 | `icamin_reduce_kernel` | 严格 `<` + 最小索引 tie-break |

**设计取舍**：曾评估 `DeInterleave+Abs+Add`，实测语义/性能不佳，已弃用（见任务目录 `性能冲刺策略.md`）。

### Kernel 设计 — 非连续路径（incx>1，SIMT）

- 每线程按 stride 读一个复数，计算 `abs1 = |re|+|im|`。
- Block 内 tree reduce（min 语义）。
- 写 workspace → `reduce_kernel`。

### NaN / Inf 语义

| 场景 | 实现 |
|------|------|
| NaN 元素 | 跳过（不参与 `<` 比较） |
| 全 NaN | 返回索引 1 |
| +Inf | 正常参与比较 |
| 空槽哨兵 | Quiet NaN bit pattern `0x7FC00000`（**不用 FLT_MAX**，因 FLT_MAX 是合法模） |
| AIV ReduceMin 被 NaN 污染 | 对该 tile 有限元标量回退扫描 |

**开放问题 Q7**：NaN 最终口径待社区确认；当前实现与 golden 一致。

### 与 isamin 的差异（评审可能问）

| 项 | isamin | icamin |
|----|--------|--------|
| 模 | `\|x\|` | `\|Re\|+\|Im\|` |
| AIV | Abs → ReduceMin | Abs → PairReduceSum → ReduceMin |
| quick return 写 result | Host `*result=0`（债） | **MemsetAsync + stream**（正本） |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 算子约束限制

- 负步长 **不反向遍历**，走 quick return。
- 性能标杆仅覆盖 `incx=1` 大 n 连续访存。
- `result` 必须为 Device 指针；调用方读回前需 sync stream。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | INT32 索引 bit-exact；`actual == golden` | 任务书 §3.2 |
| 性能标准 | 三标杆 ≤ 24.77 / 24.59 / 29.69 us | 任务书 §3.3 |

## 自测结果（2026-09-16，Ascend 950PR）

| 项 | 结果 |
|----|------|
| 全精度（排除 TC_PF） | **1005/1005 PASS** |
| 性能 case1 n=1M | **12.664 us** ≤ 24.77 |
| 性能 case2 n=2M | **16.760 us** ≤ 24.59 |
| 性能 case3 n=4M | **24.559 us** ≤ 29.69 |

复现命令：

```bash
source /workspace/tasks/icamin-950/codes/env.sh
bash /workspace/tasks/icamin-950/verify_icamin.sh --skip-build
```

## 兼容性分析

新算子，不涉及向后兼容；公开声明放入 `include/cann_ops_blas.h`，禁止 950PR 私有平行 API。

---

## 接口声明

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```

## 风险与开放问题

| ID | 项 | 对策 |
|----|-----|------|
| Q3 | n<0 与 cuBLAS 差异 | 跟 isamin → INVALID_VALUE |
| Q7 | NaN 最终口径 | 暂按「跳过 NaN；全 NaN→1」；文档标注待确认 |
| R1 | isamin Host 解引用债 | icamin 用 MemsetAsync，验收时主动说明 |

## 参考资料

1. 任务书：`aclblasIcamin_Atlas950PR_task_doc.md`
2. cuBLAS cublasIcamin
3. 仓内 isamin：`blas/iamin/arch35/`、`test/isamin/`
4. 社区任务流程：https://gitcode.com/org/cann/discussions/39

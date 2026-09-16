# aclblasScasum 算子设计文档（Ascend 950PR）

| 项目 | 内容 |
|------|------|
| 任务名称 | 8月社区任务-aclblasScasum算子开发（950） |
| 参与者账号 | `YGNwQHeVvBGnh7VoyHMac4Ts`（GitCode：`chanchp`） |
| PTA | `a3b7f2e2fd714e46b507ecbbda166c96.zip` |
| 目标硬件 | Ascend 950PR（arch35 / DAV_3510） |
| CANN 版本 | CANN 9.1.0 |
| 交付代码仓 | [chanchp/ops-blas](https://gitcode.com/chanchp/ops-blas)（私仓）分支 `feature/scasum-arch35-950` @ `96708915`；合入目标 [cann/ops-blas](https://gitcode.com/cann/ops-blas) `master` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/08-26-aclblasScasum-950/chanchp/docs/design.md` |
| 对标接口 | cuBLAS `cublasScasum`（H100 GPU 性能标杆）；语义对齐 Netlib `scasum.f` |
| 文档版本 | v1.0 |

---

# 需求背景（required）

## 需求来源

本设计对应 2026 年 8 月社区任务《aclblasScasum 算子开发（950）》。任务要求在 Ascend 950PR 上使用 Ascend C 直调方式，实现句柄式 BLAS 接口 `aclblasScasum`：对 COMPLEX64 向量各元素计算 `|Re| + |Im|` 并累加为 FLOAT32 标量。验收通过后合入 ops-blas 主仓。

设计文档按 [cann-competitions 官方模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) 提交 PR；实现代码进入 ops-blas，公开声明放入 `include/cann_ops_blas.h`，禁止 950PR 私有平行 API。

## 背景介绍

### 算子要做什么

数学表达式（1-based 索引兼容 Fortran / Netlib）：

$$
\text{result} = \sum_{i=1}^{n}\bigl(|Re(x_k)| + |Im(x_k)|\bigr),\quad k = 1+(i-1)\cdot incx
$$

- 输入：`aclblasComplex`（2×FLOAT32 交错存储）向量，逻辑长度 `n`，物理长度 `1+(n-1)*|incx|`。
- 输出：Device 上单个 **FLOAT32** 标量（非负；极端累加可溢出为 `+Inf`）。
- 「模」为 **SCABS1 / 1-范数** `|Re|+|Im|`，**不是**欧几里得模。

### 与同族算子关系

| 算子 | 输入 | 归约 | 模定义 | 仓内参照 |
|------|------|------|--------|----------|
| `aclblasSasum` | FP32 | ReduceSum | `\|x\|` | `blas/asum/arch35/`（**主复用框架**） |
| `aclblasScnrm2` | COMPLEX64 | 欧氏模平方和再开方 | $\sqrt{Re^2+Im^2}$ | 同族但语义不同 |
| **aclblasScasum** | COMPLEX64 | **ReduceSum** | `\|Re\|+\|Im\|` | 本任务新增 |

**核心映射**：连续访存（`incx == 1`）时，交错 COMPLEX64 的 `|Re|+|Im|` 求和 **等价于** 对 `2n` 个 float 做 `Abs + ReduceSum`，与 `aclblasSasum` 完全同构。非连续（`incx > 1`）需按复数步长同时累加实部与虚部绝对值，走 SIMT 路径。

### Quick return 与异常

| 条件 | 行为 |
|------|------|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` / 仓内等价未初始化码（对齐 `aclblasSasum`） |
| `n <= 0` 或 `incx <= 0` | 不启 kernel，`result = 0.0f`（H2D），`SUCCESS`；**含负步长，不反向遍历**（Netlib：`IF (N.LE.0 .OR. INCX.LE.0) RETURN`） |
| `n > 0 && incx > 0` 且 `x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `result == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

> 注意：与同批次部分复数范数算子（如 Scnrm2「incx=0 报错、负步长仍计算」）口径不同；本算子必须以 **cuBLAS / Netlib scasum / 仓内 Sasum README** 四方一致的 quick return 为准。

---

# 需求分析（required）

## 需求描述

在 Ascend 950PR 上完成：

```text
aclblasScasum(handle, n, x, incx, result)
  → Host 参数校验 + Tiling
  → Ascend C Kernel（arch35）
  → Device 写回 FLOAT32 标量
```

精度：FLOAT32 标量容差（`rtol=2^-10`，`atol=2^-16`，`max_abs_error` 限）。性能：NPU 平均单次耗时 ≤ H100 cuBLAS 标杆 / 0.4（`gpu_baseline.csv` 的 `gpu_ms` 列）。

## 需求拆解

| 编号 | 子项 | 验收要点 |
|------|------|----------|
| R1 | 公开 API | `include/cann_ops_blas.h` 新增声明；签名与 `cublasScasum` / `aclblasSasum` 同构 |
| R2 | 功能正确 | 1200 条 CSV（1000 精度 + 200 性能）PASS；quick return / 空指针 / Inf/NaN / 溢出语义对齐 |
| R3 | 复用 sasum 框架 | 代码目录 `blas/asum/arch35/`；连续路径复用 Abs+ReduceSum+AtomicAdd |
| R4 | 复数模 | `|Re|+|Im|`；禁止欧氏模；交错存储按 float 视图处理时不得漏虚部 |
| R5 | 测试工程 | `test/asum/scasum/arch35/` CSV 驱动 GTest；golden = Netlib/cblas `scasum` |
| R6 | 性能 | TC_PF 200 条；文档三档 n=1M/2M/4M 门槛 **29.4 / 30.53 / 35.52 µs** |
| R7 | 文档 | `blas/asum/README.md` 补充 scasum；950PR 产品支持表标注 |

## 输入输出规格

| 参数 | 方向 | 类型 | 内存 | shape | 说明 |
|------|------|------|------|-------|------|
| handle | 输入 | aclblasHandle_t | Host | - | 携带 stream |
| n | 输入 | int | Host | - | 复数元素个数 |
| x | 输入 | const aclblasComplex* | Device | 逻辑 [n]，物理 `1+(n-1)*\|incx\|` | 交错 float32 实虚部 |
| incx | 输入 | int | Host | - | 步长；`<=0` quick return |
| result | 输出 | float* | Device | 标量 | `|Re|+|Im|` 之和 |

---

# 详细设计（required）

## 算子分析

### 数学公式与 golden

Host CPU golden（测试用）：

```cpp
// Netlib scasum / cblas_scasum 语义
if (n <= 0 || incx <= 0) return 0.0f;
float sum = 0.0f;
for (int i = 0; i < n; ++i) {
    const aclblasComplex& z = x[i * incx];
    sum += fabsf(z.real) + fabsf(z.imag);
}
return sum;
```

连续路径等价变换（实现优化用，非接口语义变更）：

```text
incx == 1:
  view x as float[2n]
  scasum(n, x) ≡ sasum(2n, (float*)x)
```

### 支持数据类型与芯片

| 输入 | 输出 | 芯片 |
|------|------|------|
| COMPLEX64（aclblasComplex，2×FLOAT32） | FLOAT32 | Ascend 950PR / 950DT |

### 支持形状

- 一维向量，`n` 覆盖任务包精度至 `2^20`、性能至 `2^22`。
- `incx = 1`：连续访存，走 AIV 向量主路径（性能用例全部为此）。
- `incx > 1`：SIMT 步长访存（精度 TC_INC / TC_EX）。
- `incx <= 0`：quick return，不遍历。

---

## 算子实现

### 实现方案总览

沿用 ops-blas **asum 族 arch35**（`sasum_*`）结构，新增复数入口：

```text
Host (scasum_host.cpp)
  Validate → quick return → CalcTiling → scasum_kernel_do

Kernel 分支（对齐 sasum_kernel_do）:
  ├─ incx == 1  → scasum_aiv_kernel
  │                （float 视图长度 2n：Abs → ReduceSum → AtomicAdd 写 result）
  └─ incx > 1   → scasum_simt_kernel（按复数步长累加 |re|+|im|）
                   + scasum_reduce_kernel（跨核 partial sum → result）
```

目录规划：

```text
ops-blas/
├── include/cann_ops_blas.h                 # + aclblasScasum 声明
├── blas/asum/arch35/
│   ├── sasum_host.cpp / sasum_kernel.cpp   # 已有实数路径（保持）
│   ├── scasum_host.cpp                     # 新增
│   ├── scasum_kernel.cpp                   # 新增（可复用 SasumAIV 思路）
│   ├── scasum_tiling_data.h                # 新增（字段对齐 SasumTilingData）
│   └── （可选）共享 tiling / reduce 工具
├── blas/asum/README.md                     # 补充 scasum 章节
└── test/asum/scasum/
    ├── scasum_golden.h
    ├── scasum_param.h
    ├── CMakeLists.txt
    └── arch35/
        ├── scasum_npu_wrapper.h
        ├── scasum_test.cpp
        └── scasum_test.csv                 # 任务包 1200 条
```

### Host 侧设计

1. **参数校验** `ValidateScasumParams`：逻辑同 `ValidateSasumParams`，`x` 类型为 `const aclblasComplex*`。
2. **Quick return**：`n <= 0 || incx <= 0` → `aclrtMemcpy` 写 `0.0f` 到 Device `result`，返回 `SUCCESS`（**禁止** Host 侧解引用 Device 指针）。
3. **Tiling** `CalcScasumTilingData`：
   - `incx == 1`：按 **float 元素数 `2n`** 做多核均分（与 sasum 一致），`tiling.n = 2n`，`tiling.incx = 1`。
   - `incx > 1`：按 **复数元素数 `n`** 分核，`tiling.incx = incx`（复数步长），SIMT `nthreads` 计算同 sasum。
4. **分核数** `numBlocks = min(eleNum, GetAivCoreCount(), SCASUM_MAX_CORE_NUM)`。
5. **Workspace**：仅 `incx > 1` 需要，每核一个 float partial；上限检查 `GetEffectiveWorkspaceSize`。

### Kernel 设计 — 连续路径（incx=1）

**数据流**：

```text
GM complex[n]  ≡  float[2n] 交错
    ↓ DataCopy / DataCopyPad（按 float）
UB inLocal[]
    ↓ Abs
UB |x_f|[]
    ↓ ReduceSum
scalar tileSum
    ↓ SetAtomicAdd + DataCopyPad
GM result（多核原子累加）
```

**API 映射**：

| 步骤 | Ascend C API | 说明 |
|------|--------------|------|
| 搬运 | `DataCopy` / `DataCopyPad` | GM→UB；尾块 32B 对齐 padding |
| 绝对值 | `Abs` | 对实部、虚部一并取绝对值 |
| 块内归约 | `ReduceSum` | Level-1 / sharedTmpBuffer 对齐 sasum |
| 跨核累加 | `SetAtomicAdd` + `DataCopyPad` | 与 sasum 相同；Init 时先清零 result |

**UB 规划**：复用 `SasumAIV` 的 `UB_MAX_CHUNK_FLOATS = 27392`、`BUFFER_NUM=2` double buffer、`SAFETY_MARGIN=32KB` 约束方程，保证 arch35 UB 不溢出。

**可选实现策略（二选一，优先 A）**：

| 策略 | 做法 | 优劣 |
|------|------|------|
| **A. 独立 `ScasumAIV`（推荐）** | 代码结构复制 sasum，入口 `nFloat = 2*n` | 接口清晰，不耦合 sasum 符号 |
| B. Host 内转发 | `incx==1` 时直接调 `aclblasSasum(h, 2*n, (float*)x, 1, result)` | 最少代码，但类型别名依赖与审计可读性略差 |

本设计采用 **策略 A**，保证复数 API 独立可维护；数学上与 B 等价。

### Kernel 设计 — 非连续路径（incx>1）

复用 sasum SIMT 框架并改为复数读：

```cpp
// 每线程：
for (i = threadIdx; i < calNum; i += blockDim) {
    const float* p = &xGm[(startOffset + i) * stride * 2]; // stride=incx（复数）
    float re = p[0], im = p[1];
    partial += fabs(re) + fabs(im);
}
// block 内 tree reduce → workspace[blockIdx]
// 再 launch scasum_reduce_kernel 求和写入 result
```

要点：

- `stride` 为 **复数步长**；float 下标乘 2。
- `incx <= 0` 不会进入本路径（Host quick return）。
- Reduce kernel 与 sasum 同构（workspace float 数组求和）。

### NaN / Inf / 溢出语义

| 场景 | 行为 |
|------|------|
| 元素含 `NaN` | 浮点 `|NaN|` / 累加传播 `NaN`（与 cblas golden 一致比对） |
| 元素含 `±Inf` | `|Inf|=Inf`，和可为 `Inf` |
| 大 n × 极端值（`RANDOM_EXTREME`） | 允许溢出到 `+Inf`；与 golden 同为 Inf 则 PASS |
| 归约顺序 | **不要求** bit-exact；按 FLOAT32 容差判定 |

---

## 精度与测试设计

### Golden

- 文件：`test/asum/scasum/scasum_golden.h`
- 优先链接 cblas `cblas_scasum`；若环境无 cblas 复数例程，则内嵌与 Netlib 一致的 CPU 循环。
- 判定：标量 `|actual - golden| ≤ atol + rtol × |golden|`，并满足 `max_abs_error` 限。
- Inf 一致性：双方均为 Inf → PASS。

### 精度阈值（任务书 §3.2）

| 输出类型 | rtol | atol | matched_ratio | max_abs_error_limit |
|----------|------|------|---------------|---------------------|
| FLOAT32 标量 | 2^-10 (≈9.77e-4) | 2^-16 (≈1.53e-5) | 0.99（标量退化为单值） | 1e-2 或 32×ULP |

### 测试框架

对齐 `test/asum/sasum/arch35/`：

- CSV 列：`case_name, description, n, incx, x, result, expect_result, mere_threshold, mare_multiplier, random_seed`
- `x=NULLPTR` / `result=NULLPTR` 表达空指针负向用例
- 性能用例 `TC_PF_*`：warmup ≥ 10，iters ≥ 100（满足任务书 >50），Event 打印 `[PERF] avg_us=`

### 用例覆盖（任务包 1200 条）

| 类别 | 前缀 | 条数 | 要点 |
|------|------|------|------|
| L0 基础 | TC_L0 | 4 | n=1/8 × incx=1/2 |
| 尺寸 | TC_SQ | 38 | 1→2^20 |
| 步长 | TC_INC | 9 | incx=1/2/3 正常路径 |
| 填充 | TC_FL | 12 | 均匀/全零/交替/极端/Inf/NaN |
| 溢出 | TC_OV | 3 | 大 n × RANDOM_EXTREME |
| 边界 | TC_ED | 13 | n≤0、incx≤0 quick return；空指针 |
| 扩展 | TC_EX | 921 | 尺寸×步长×填充采样 |
| 性能 | TC_PF | 200 | 含任务书三档；一律 incx=1 |

### 自验命令

```bash
# 精度（排除 TC_PF）
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./scasum_test.csv

# 性能（TC_PF vs gpu_baseline）
python verify_performance.py --repo /path/to/ops-blas --soc ascend950
# 判定: NPU_us <= gpu_ms * 1000 / 0.4
```

---

## 性能设计

### 标杆与门槛

GPU 标杆：**NVIDIA H100** + cuBLAS `cublasScasum`，任务包 `gpu_baseline.csv`。

公式：`thr_us = gpu_ms × 1000 / 0.4`（倍率门禁 ≥ 0.4）。

| case | n | incx | H100 gpu_ms | NPU 门槛 (us) |
|------|---|------|-------------|---------------|
| 1 | 1048576 | 1 | 0.011760 | **29.40** |
| 2 | 2097152 | 1 | 0.012212 | **30.53** |
| 3 | 4194304 | 1 | 0.014209 | **35.52** |

与任务书 §3.3 三档一致。门槛相对偏紧（带宽型），必须走向量 Abs+ReduceSum+AtomicAdd，避免逐元素 Host 式循环。

### 优化策略

1. **连续大 n**：多核均分 `2n` float + AtomicAdd，对齐 sasum 已验证路径。
2. **Tile 大小**：沿用 `UB_MAX_CHUNK_FLOATS=27392`，保证 ReduceSum repeat 对齐与带宽利用率。
3. **Double Buffer**：`BUFFER_NUM=2`，CopyIn / Compute 重叠。
4. **Init 清零 result**：与 sasum 相同，保证 AtomicAdd 正确。
5. **不做欧氏模路径误用**：严禁 `Mul+Add+Sqrt`；仅 `Abs+Add`（或 float 视图整体 `Abs`）。

### 性能测试方法

- 设备：Ascend 950PR
- warmup ≥ 10，有效采样 ≥ 100（任务书 >50）
- `aclrtEvent` 统计 kernel 平均 us
- 与 `gpu_baseline.csv` 按 `(n, incx)` 关联判定 PASS/FAIL

---

## 接口声明

```cpp
/**
 * @brief 计算 COMPLEX64 向量各元素 |Re|+|Im| 之和（FLOAT32 标量）。
 * @param handle ops-blas 句柄
 * @param n      复数元素个数；n<=0 时 quick return
 * @param x      Device 输入向量
 * @param incx   步长；incx<=0 时 quick return
 * @param result Device 输出标量（FLOAT32）
 */
aclblasStatus_t aclblasScasum(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    float* result);
```

---

## 风险与开放问题

| ID | 项 | 对策 |
|----|-----|------|
| R1 | 连续路径 `2n` 与奇数对齐 | float 长度恒为偶数；尾块 DataCopyPad 按 float 对齐 |
| R2 | AtomicAdd 竞争导致与 golden 舍入差 | 容差判定；不要求 bit-exact |
| R3 | 性能门槛紧（~29–35 µs） | 严格复用 sasum AIV；禁止额外 GM 往返 |
| R4 | 与 Scnrm2 quick return 口径混淆 | 单测覆盖 incx=0/-1；文档醒目标注 |
| R5 | handle 空指针返回码 | 与仓内 sasum 保持一致，不以 cuBLAS 文案强行改仓内惯例 |

---

## 交付清单

| 序号 | 交付件 | 路径/说明 |
|------|--------|-----------|
| 1 | 设计文档 | 本文档 → cann-competitions PR |
| 2 | 算子实现 | `blas/asum/arch35/scasum_*` |
| 3 | 头文件 | `include/cann_ops_blas.h` |
| 4 | README | `blas/asum/README.md` |
| 5 | 测试 | `test/asum/scasum/arch35/` + CSV |
| 6 | 自测报告 | 腾讯文档模板 + 950PR 日志（`logs/scasum-950`） |
| 7 | 代码私仓 | https://gitcode.com/Eason_Cat/ops-blas `feature/scasum-arch35-950`，已邀请 Ascend-CANN |

---

## 参考资料

1. 任务书：`aclblasScasum_Atlas950PR_task_doc.md`（PTA：`a3b7f2e2fd714e46b507ecbbda166c96.zip`）
2. cuBLAS cublasScasum：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-asum
3. Netlib scasum：https://www.netlib.org/blas/scasum.f
4. 仓内 sasum：`blas/asum/arch35/`、`test/asum/sasum/`
5. 精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
6. 社区任务流程：https://gitcode.com/org/cann/discussions/39

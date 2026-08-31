# aclblasIcamax 算子设计文档（Ascend 950PR�?

| 项目 | 内容 |
|------|------|
| 任务名称 | 8月社区任�?aclblasIcamax算子开发（950�?|
| 参与者账�?| `-7m2FRYfKJXCaXD8P2gMKd1d` |
| 目标硬件 | Ascend 950PR（arch35 / DAV_3510�?|
| CANN 版本 | CANN 9.1.0 |
| 交付代码�?| [cann/ops-blas](https://gitcode.com/cann/ops-blas) `master` |
| 设计提交路径 | `04_tasks/01_community-task-2026/tasklist/08-46-aclblasIcamax/-7m2FRYfKJXCaXD8P2gMKd1d/docs/design.md` |
| 对标接口 | cuBLAS `cublasIcamax`（H100 GPU 性能标杆�?|
| 文档版本 | v1.0 |

---

# 需求背景（required�?

## 需求来�?

本设计对�?2026 �?8 月社区任务《aclblasIcamax 算子开发（950）》。任务要求在 Ascend 950PR 上使�?Ascend C 直调方式，实现句柄式 BLAS 接口 `aclblasIcamax`：在 COMPLEX64 向量中查�?**1-范数�?* `|Re|+|Im|` **最�?*元素�?**1-based 索引**；模相同时取最小索引。验收通过后合�?ops-blas 主仓�?

设计文档�?[cann-competitions 官方模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) 提交 PR；实现代码进�?ops-blas，公开声明放入 `include/cann_ops_blas.h`，禁�?950PR 私有平行 API�?

## 背景介绍

### 算子要做什�?

数学表达式：

$$
\text{result} = \mathop{\arg\max}_{i=1..n}\bigl(\,|Re(x_k)| + |Im(x_k)|\,\bigr),\quad k = 1+(i-1)\cdot incx
$$

- 输出�?**INT32 标量**�?*1-based** 索引（Fortran / BLAS 惯例）�?
- 「模」为 **SCABS1 / 1-范数**：`|Re|+|Im|`�?*不是**欧几里得模�?
- 并列时取 **最小索�?*；比较语义对�?Netlib/cuBLAS�?*严格大于才更�?*（`abs1 > best`）�?
- Golden �?**Netlib cblas `icamax`** 生成�?-based �?golden �?+1）�?

### 与同族算子关�?

| 算子 | 输入 | 归约 | 模定�?| 仓内参照 |
|------|------|------|--------|----------|
| `aclblasIsamax` | FP32 | ReduceMax | `\|x\|` | `blas/iamax/arch35/`�?*Host/归约框架**�?|
| `aclblasIcamin` | COMPLEX64 | ReduceMin | `\|Re\|+\|Im\|` | `blas/iamin/arch35/`（ReduceMin 镜像�?|
| **aclblasIcamax** | COMPLEX64 | **ReduceMax** | `\|Re\|+\|Im\|` | 本任务新�?|

**核心差异（相�?isamax�?*�?

1. 输入为交�?COMPLEX64，需先算 `|Re|+|Im|` �?ReduceMax�?
2. H100 性能标杆 **几乎不随 n 增长**（launch-bound），NPU 必须�?**�?overhead �?SIMT 融合路径**，不能简单照�?isamax 向量�?tile 方案�?

### Quick return 与异�?

| 条件 | 行为 |
|------|------|
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n = 0` �?`incx �?0` | 不启 kernel，`result = 0`，`SUCCESS`�?*负步长不反向遍历** |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n > 0 && incx �?1` �?`x/result == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

---

# 需求分析（required�?

## 需求描�?

```text
aclblasIcamax(handle, n, x, incx, result)
  �?Host 校验 + 策略选择 (AUTO / ICAMAX_STRAT)
  �?Ascend C Kernel（SIMT 主路�?/ Vector 显式路径�?
  �?Device 写回 1-based INT32 索引
```

精度�?*整数索引 bit-exact**。性能：`NPU_us �?gpu_ms × 1000 / 0.4`（H100 cuBLAS 标杆）�?

## 需求拆�?

| 编号 | 子项 | 验收要点 |
|------|------|----------|
| R1 | 公开 API | `include/cann_ops_blas.h` 新增 `aclblasIcamax`；对�?`cublasIcamax` |
| R2 | 功能正确 | CSV 精度用例 PASS；Netlib 严格 `>` / tie / NaN / Inf 对齐 |
| R3 | 复数�?| SCABS1；禁止欧氏模；Reg 路径 mag 写独�?buffer |
| R4 | 双路�?| 连续 incx=1 �?SIMT AUTO；Vector DB 仅显�?strat |
| R5 | 测试 | `test/iamax/icamax/arch35/` CSV + cblas golden |
| R6 | 性能 | TC_PF 200 条；文档三档门槛见下�?|
| R7 | Bench API | 可�?`aclblasIcamaxBench`（Event 性能自测�?|

## 输入输出规格

| 参数 | 方向 | 类型 | 内存 | shape | 说明 |
|------|------|------|------|-------|------|
| handle | 输入 | aclblasHandle_t | Host | - | 携带 stream |
| n | 输入 | int | Host | - | 复数元素个数 |
| x | 输入 | const aclblasComplex* | Device | 逻辑 [n] | 交错 float32 实虚�?|
| incx | 输入 | int | Host | - | �? 正常；≤0 quick return |
| result | 输出 | int* | Device | 标量 | 1-based �?0 |

---

# 详细设计（required�?

## 算子分析

### 数学公式�?golden

Host / 测试 golden（`icamax_golden.h`）：

```cpp
float abs1 = |Re| + |Im|;  // �?sqrt(re²+im²)
bestVal = abs1(x[0]); bestIdx = 0;
for i in 1..n-1:
    if abs1(x[i]) > bestVal:   // Netlib 严格 >
        bestVal = abs1; bestIdx = i;
result = bestIdx + 1;
```

cblas `icamax` 返回 0-based，测试侧 +1 后与 NPU 比对�?

### 支持数据类型与芯�?

| 输入 | 输出 | 芯片 |
|------|------|------|
| COMPLEX64 | INT32 | Ascend 950PR / 950DT |

---

## 算子实现

### 实现方案总览

采用 **SIMT 融合主路�?+ Vector Reg 显式路径** 双轨设计，目录：

```text
ops-blas/
├── include/cann_ops_blas.h
├── blas/iamax/arch35/
�?  ├── icamax_host.cpp          # 校验、AUTO 分档、Launch
�?  ├── icamax_kernel.cpp / .h   # SIMT / Vector / Hybrid / Reduce
�?  └── icamax_tiling_data.h     # IcamaxStratId、IcamaxTilingData
├── blas/iamax/README.md
└── test/iamax/icamax/
    ├── icamax_golden.h
    ├── icamax_npu_wrapper.h     # npu + bench 封装
    └── arch35/icamax_test.cpp + icamax_test.csv
```

**Launch 模式**�?

| launchMode | 含义 |
|------------|------|
| `FUSE_SYNCALL` | �?launch：SIMT/Vector 扫描 + block �?SyncAll 归约 + 直写 result |
| `DUAL` | simt_kernel + reduce_kernel 两次 launch（非 AUTO 默认�?|

### Host �?�?策略 AUTO 分档

环境变量 `ICAMAX_STRAT` 覆盖 AUTO（支持数�?id 与别名，�?`dense`、`vecdb`）�?

**AUTO（incx=1，未�?env）分�?*（`CalcIcamaxTiling`）：

| n 区间 | 内核策略 | 要点 |
|--------|----------|------|
| `< 8192` | `SIMT_UNROLL4` | 单核或少量核，避�?SyncAll �?|
| `8192 ~ 1.5M` | `SIMT_UNROLL2` + pack | ept48/max512 |
| `> 1.5M` | `SIMT_DENSE_U1` | ept32/max1024，grid-stride 扫描 |

`PickStratByShape` 入口 strat �?`SIMT_FUSE`（FUSE_SYNCALL）；具体 kernel 变体由上表在 tiling 阶段展开�?

**�?AUTO 显式路径**（A/B 与回归用，不进入 AUTO）：

| strat | 用�?|
|-------|------|
| `FULL_DB_8192 (7)` | Vector ping-pong + magBuf |
| `FULL_DB_VEC (11)` | Vector tile=12288 + magBuf Reg |
| `FULL_16320 (1)` | �?AIV 向量 baseline |
| `HYBRID (33)` | SIMD tile + SIMT rem（大 n 实测回归�?*�?AUTO**�?|

### Kernel 路径 A �?SIMT 融合（AUTO 默认�?

**数据�?*�?

```text
GM complex[n] (incx=1 连续)
    �?�?thread grid-stride �?complex
    �?设备�?SCABS1: |re|+|im|
    �?PreferCand / strict '>' 块内归约
    �?SyncAll + WS merge（FUSE）或 reduce_kernel（DUAL�?
GM result (1-based INT32)
```

**设计要点**�?

- **GM 直扫**：避�?Vector DataCopyPad + ReduceMax �?launch/pad 开销；对�?H100「flat baseline」特性�?
- **complex64 pack load**（`FLOAT2` 路径）：减少 GM 访存次数�?
- **tie-break**：相等保留较小索引；NaN 不参�?`>` 更新�?
- **非连�?incx>1**：强�?SIMT，按 `(offset+i)*stride` 读复数；`nthreads` �?perCoreN 自适应�?

### Kernel 路径 B �?Vector Reg（显�?`vecdb` / `db8192`�?

**数据�?*�?

```text
GM complex �?DataCopy/DINTLV �?UB re/im
    �?Abs + Add（或 PairReduce�?
UB magBuf[]          �?独立 buffer，禁�?in-place 写回 src
    �?ReduceMax(calIndex=true)
(tileMax, localIdx) �?�?tile 严格 '>' 合并
WS �?reduce �?result
```

**Reg magBuf 约束**（DAV_3510 UB ~248KB）：

- Double-buffer（TQue num=2）仅 **tile �?8192**；更�?tile �?serial inQue�?
- **禁止** DINTLV 后将 mag 写回交错 src（曾导致�?tile ACC 索引错误）�?

### Kernel 路径 C �?Hybrid（仅显式 strat=33�?

�?n �?Vector tile + SIMT remainder 合并；实测较 denser SIMT **~2× �?*�?*不纳�?AUTO**�?

### Workspace

- SIMT FUSE：per-core WS `(maxVal, idx)` 两个 float，reduce �?SyncAll merge�?
- Vector：与 isamax 类似 WS sizing；对�?64 float�?

### NaN / Inf 语义

| 场景 | 行为 |
|------|------|
| NaN | 严格 `>` 下不更新 winner；首元素 NaN 时结果常停留索引 1 |
| +Inf | 首个 Inf 胜出（Netlib�?|
| 全零 tie | 索引 1 |

---

## 精度与测试设�?

### Golden

- `test/iamax/icamax/icamax_golden.h`：手�?SCABS1 + 严格 `>`（与 Netlib 对齐）�?
- 可选对�?cblas `icamax` +1�?
- 判定：`EXPECT_EQ(npu, golden)`�?

### 用例规模（任务包�?

- 精度 + 性能合计 **1200** 条（可扩展）�?
- **TC_PF** 200 条性能用例�?*incx=1**；前 3 条与 `gpu_baseline.csv` 对齐�?

### 自验命令

```bash
python verify_accuracy.py --repo $OPS_BLAS --soc ascend950 --csv ./icamax_test.csv
python verify_performance.py --repo $OPS_BLAS --soc ascend950
```

性能须读 GTest/Event 日志 **`[PERF] avg_us=`**，不�?wall-clock ms�?

---

## 性能设计

### 标杆与门�?

GPU 标杆�?*NVIDIA H100** + cuBLAS `cublasIcamax`�?

任务�?`gpu_baseline.csv` 前三档（�?TC_PF_1001~1003 对应）：

| n | incx | H100 gpu_ms | NPU 门槛 (µs) |
|---|------|-------------|---------------|
| 1,048,576 | 1 | 0.009809 | **24.52** |
| 2,097,152 | 1 | 0.010324 | **25.81** |
| 4,194,304 | 1 | 0.012576 | **31.44** |

公式：`thr_us = gpu_ms × 1000 / 0.4`�?

> 说明：任务书 §3.3 表格 n �?CSV 略有出入（如 16M），**验收以任务包 CSV + verify_performance.py 关联 (n,incx) 为准**�?

### 性能特性分�?

H100 icamax baseline **�?n 几乎 flat**（~10 µs 量级），等价�?**kernel launch + 短扫�?* 主导；icamin 则随 n 线性上升。因�?NPU 优化重点�?**降低 SyncAll / 二次 launch / Vector pad 开销**，而非单纯放大 UB tile�?

### AUTO 策略选型依据

| 方案 | 1M / 2M / 4M 量级 (µs) | 结论 |
|------|------------------------|------|
| Vector FULL_16320 AUTO | ~38�?2 / ~100+ | 过慢，弃 AUTO |
| HYBRID (33) | ~58 / ~100 | 回归，禁 AUTO |
| **SIMT denser AUTO** | **~21 / ~29 / ~45** | **当前默认**�?M PASS�?M/4M 待优�?|

后续优化方向（不改变 AUTO 框架）：

1. 2M/4M �?msprof 热点（reduce / SyncAll / GM 带宽）�?
2. 显式 A/B：`ICAMAX_STRAT=vecdb` �?SIMT 对照精度；perf 不替�?AUTO  unless 全面 PASS�?
3. 禁止回退：大 n ILP/PIPE/soft-pipe/maxThr2048/CORE16 �?Vector AUTO�?

### 性能测试方法

- 设备：Ascend 950PR
- warmup=10，iters=100（任务书要求 >50�?
- `aclblasIcamaxBench` �?GTest PF 路径 + `aclrtEvent`
- 200 �?TC_PF �?CSV 逐条比对 PASS/FAIL

---

## 接口声明

```cpp
/**
 * @brief 查找 COMPLEX64 向量�?|Re|+|Im| 最大元素的 1-based 索引�?
 */
aclblasStatus_t aclblasIcamax(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);

/** 性能 bench：tiling 一�?+ Event 计时（测试专用） */
aclblasStatus_t aclblasIcamaxBench(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result,
    int warmup,
    int iters,
    float* avg_ms);
```

---

## 风险与已知问�?

| ID | �?| 对策 |
|----|-----|------|
| P1 | 2M/4M 未达 CSV 门槛 | 保持 SIMT denser AUTO；profiling 后定点优�?|
| P2 | Reg in-place mag | 已改 magBuf_ 独立 UB |
| P3 | HYBRID/Vector AUTO | 禁止进入 AUTO |
| P4 | 偶发 TC_EX 精度 flake | �?strat/同步相关时复测；�?200/200 PF ACC 为主 |
| P5 | icamax vs icamin CSV 不对�?| �?icamax 任务验收 icamax CSV；不混用 icamin 门槛 |

---

## 交付清单

| 序号 | 交付�?| 路径 |
|------|--------|------|
| 1 | 设计文档 | cann-competitions PR（本文档�?|
| 2 | 算子实现 | `blas/iamax/arch35/icamax_*` |
| 3 | 头文�?| `include/cann_ops_blas.h` |
| 4 | README | `blas/iamax/README.md` |
| 5 | 测试 | `test/iamax/icamax/arch35/` |
| 6 | 自测报告 | 腾讯文档模板 + 950PR 截图 |
| 7 | 代码 MR | fork �?ops-blas，邀�?Ascend-CANN |

---

## 参考资�?

1. 任务书：`aclblasIcamax_Atlas950PR_task_doc.md`
2. Netlib icamax：https://www.netlib.org/blas/icamax.f
3. cuBLAS cublasIcamax：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-amax
4. 仓内 isamax：`blas/iamax/arch35/`、`test/isamax/`
5. 精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
6. 社区任务流程：https://gitcode.com/org/cann/discussions/39

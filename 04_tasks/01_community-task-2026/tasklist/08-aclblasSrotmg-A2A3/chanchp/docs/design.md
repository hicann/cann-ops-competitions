# aclblasSrotmg A2/A3 算子设计文档

# 需求背景（required）

## 需求来源

本任务来自 CANN 社区 2026 年 8 月算子开发任务，要求在 `ops-blas` 开源仓中为 Atlas A2/A3 系列产品（含 Atlas 800I A2 / Atlas 800I A3）补齐单精度修正 Givens 旋转参数构造算子 `aclblasSrotmg` 的 arch22 实现、测试与文档。

对标接口：cuBLAS `cublasSrotmg`；算法语义参考 Netlib BLAS `srotmg`。接口声明复用 `include/cann_ops_blas.h` 已有声明，与 950PR（arch35）等产品线共用同一 API，不新增产品私有平行接口。

任务书：`8月社区任务-aclblasSrotmg算子开发（A2A3）/aclblasSrotmg_Atlas800IA3_task_doc.md`。

## 背景介绍

### aclblasSrotmg 算子功能

`aclblasSrotmg` 是 BLAS Level-1 纯标量算子：由输入标量 `d1`、`d2`、`x1`、`y1` 构造修正 Givens（modified Givens）变换矩阵 `H`，使

```text
H^T * diag(d1, d2) * H = diag(d1_new, d2_new)
H * [x1, y1]^T = [x1_new, 0]^T
```

`H` 为 2×2 矩阵，由输出数组 `param[5]` 按 BLAS 标准编码：`param[0] = flag` 决定 `H` 的预定义形式，flag 隐含的 `1.0 / -1.0 / 0.0` 不重复存入 param：

| param[0] (flag) | H 矩阵 | 存储的元素 |
| --- | --- | --- |
| -1.0 | [[h11, h12], [h21, h22]] | param[1..4] 全部存储 |
| 0.0 | [[1, h12], [h21, 1]] | param[2]=h21，param[3]=h12 |
| 1.0 | [[h11, 1], [-1, h22]] | param[1]=h11，param[4]=h22 |
| -2.0 | [[1, 0], [0, 1]] | 无（恒等变换，仅设 flag） |

输出语义：`d1`、`d2`、`x1` 原地覆写；`y1` 只读；`param` 为纯输出。

### 仓内现状分析

`ops-blas` 仓已有：

| 能力 | 现状 |
| --- | --- |
| API 声明 | `include/cann_ops_blas.h` 已声明 `aclblasSrotmg` |
| arch35（950PR） | `blas/rotmg/arch35/` 已实现 Host/Device 双路径 |
| 测试框架 | `test/rotmg/srotmg/` CSV 驱动 + cblas golden |
| arch22（A2/A3） | **缺失**，本任务补齐 |

仓内既有 Host/Device 双路径判别模式须保持一致：五个指针全部 Host → CPU 直算；全部 Device → 1 block Ascend C kernel；混合指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（arch22）上使用 Ascend C 实现 `aclblasSrotmg`，使功能、精度、性能满足任务书验收标准，并合入 `cann/ops-blas` 的 `blas/rotmg/` 与 `test/rotmg/srotmg/arch22/`。

接口：

```cpp
aclblasStatus_t aclblasSrotmg(
    aclblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);
```

## 需求拆解

1. 覆盖 Netlib `srotmg` 全部分支语义：`flag = -2/-1/0/1` 与 GAM 缩放保护路径。
2. Host 路径 CPU 直算；Device 路径 1 block Ascend C kernel，结果留在 Device。
3. 参数合法性与指针位置校验与 arch35 / README 口径一致。
4. 精度以 `cblas_srotmg` 为唯一 golden，逐标量比对 `d1/d2/x1/param[0..4]`。
5. 性能：warmup 后有效采样 >50 次取平均；单次平均耗时不高于任务书 §3.3 / `gpu_baseline` 口径。
6. 更新 `blas/rotmg/README.md` 产品支持表，标注 Atlas A2/A3 支持。
7. 补充 arch22 CSV 驱动精度用例与性能用例（含 TC_PF）。

# 详细设计（required）

## 算子分析

### 数学公式

输入标量 `d1, d2, x1, y1`，构造 `H` 与更新后的 `d1_new, d2_new, x1_new`：

```text
H^T * diag(d1, d2) * H = diag(d1_new, d2_new)
H * [x1, y1]^T = [x1_new, 0]^T
```

特殊分支（与 Netlib `srotmg` 一致）：

1. `d1 < 0`：`flag = -1`，H 元素全 0，`d1/d2/x1` 覆写为 0。
2. `d1 ≥ 0` 且 `d2·y1 = 0`：`flag = -2`（恒等），仅写 `param[0]`，其余输入不变。
3. `|d1·x1²| > |d2·y1²|`：`h21 = -y1/x1`，`h12 = (d2·y1)/(d1·x1)`，`su = 1 - h12·h21`；`su > 0` 时 `flag = 0` 并缩放 `d1/d2/x1`。
4. `|d1·x1²| ≤ |d2·y1²|`：若 `d2·y1² < 0` 则全零分支；否则 `flag = 1`，`h11/h22` 按标准公式计算并互换缩放 `d1/d2`。
5. 缩放保护：`GAM = 4096`，`GAMSQ = 1.67772e7`，`RGAMSQ = 5.96046e-8`；当 `d1` 或 `|d2|` 越出 `[RGAMSQ, GAMSQ]` 时循环缩放，并将 `flag` 翻转为 `-1`、补齐对应 H 元素为 `±1`。

实现上对齐 OpenBLAS / Netlib 细节：缩放循环中 `flag==-1` 时不错误重置 `h12`；`sp1==0` 时 `h22=0`；对非有限输入在缩放循环中加 `isfinite` 守卫，避免无限循环。

### 支持数据类型

| 参数 | 数据类型 | 说明 |
| --- | --- | --- |
| d1 / d2 / x1 / y1 / param | float32 | 纯标量算子，无其它 dtype |

### 支持形状

本算子无尺寸轴：五个参数均为标量指针（`param` 为长度 5 的标量数组）。不涉及 broadcast、动态 shape、非连续 Tensor。

## 算子实现

### 实现方案

整体沿用 ops-blas rotmg 工程模式，在 arch22 目录落地：

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| Host | `blas/rotmg/arch22/srotmg_host.cpp` | 句柄/空指针校验、Host/Device 判别、CPU 路径、kernel launch |
| Tiling | `blas/rotmg/arch22/srotmg_tiling_data.h` | `SrotmgTilingData { uint32_t repeat; }` |
| Kernel | `blas/rotmg/arch22/srotmg_kernel.cpp` | Device 路径 Ascend C 标量计算 |
| 测试 | `test/rotmg/srotmg/arch22/` | CSV 驱动精度 + 性能（GTest） |
| 文档 | `blas/rotmg/README.md` | 产品支持表更新为 A2/A3 支持 |

### Host 侧设计

执行顺序：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `d1/d2/x1/y1/param` 任一为空 → `ACLBLAS_STATUS_INVALID_VALUE`。
3. `aclrtPointerGetAttributes` 判定五个指针位置；混合 Host/Device → `ACLBLAS_STATUS_INVALID_VALUE`。
4. **全部 Host**：调用 `SrotmgCpuCompute`，同步返回，不 launch kernel。
5. **全部 Device**：组装 `SrotmgTilingData`（生产路径 `repeat = 1`），经 handle 绑定 stream 调用 `srotmg_kernel_do`，`numBlocks = 1`。

Tiling 说明：

- 纯标量算子，无需按 shape 分核/分块；单 block 即可。
- `repeat` 仅用于纯核微基准：生产路径固定为 1；性能测试可在单次 launch 内循环 `repeat` 次计算并每轮重置输入，摊销 launch 开销，对齐 QR 等迭代调用场景。

### Kernel 侧设计

1. `KERNEL_TYPE_AIV_ONLY`；仅 `GetBlockIdx() == 0` 执行。
2. 使用 `GlobalTensor<float>::GetValue/SetValue` 直接访问 GM 标量，避免 SIMT 线程开销与 DataCopy 复杂度。
3. 算法与 Host CPU 路径同源（Netlib `srotmg` + OpenBLAS 缩放细节对齐）。
4. `tiling.repeat` 循环内每轮从初始输入重载 `d1/d2/x1/y1`，保证批量连续调用语义正确。
5. 缩放循环对非有限值提前退出，保证极端输入可终止。

流程：

```text
Init GM buffers
→ for r in [0, repeat):
     reload inputs
     branch on d1 / d2*y1 / |sq1| vs |sq2|
     optional GAM scale loops
     write back d1/d2/x1/param
```

### 文件规划

| 文件 | 新增/修改 | 说明 |
| --- | --- | --- |
| `blas/rotmg/arch22/srotmg_host.cpp` | 新增 | arch22 Host 实现 |
| `blas/rotmg/arch22/srotmg_kernel.cpp` | 新增 | arch22 Kernel |
| `blas/rotmg/arch22/srotmg_tiling_data.h` | 新增 | tiling（含 repeat） |
| `blas/rotmg/README.md` | 修改 | A2/A3 产品支持表改为支持 |
| `test/rotmg/srotmg/arch22/srotmg_test.cpp` | 新增 | GTest CSV 驱动 |
| `test/rotmg/srotmg/arch22/srotmg_test.csv` | 新增 | 精度 + TC_PF 用例 |
| `test/rotmg/srotmg/arch22/srotmg_npu_wrapper.h` | 新增 | NPU 封装 |
| `test/rotmg/srotmg/srotmg_param.h` / `srotmg_golden.h` | 复用/小改 | 参数解析与 cblas golden |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（含 Atlas 800I A2 / 800T A2） | √ |
| Atlas A3 系列产品（含 Atlas 800I A3） | √ |

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 空指针 | handle 空 → `HANDLE_IS_NULLPTR`；其余空 → `INVALID_VALUE` |
| 内存一致性 | 五个指针须全部 Host 或全部 Device |
| dtype | 仅 float32 |
| 形状 | 纯标量，无维度/步长参数 |
| 原地语义 | d1/d2/x1 原地覆写；y1 只读；param 独立输出 |
| 异步 | Device 路径依赖 `aclblasSetStream`；读回前须同步 stream |
| 确定性 | 不要求 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32：`rtol=2^-10`，`atol=2^-16`，`required_matched_ratio=0.99`，`max_abs_error_limit=1e-2 或 32*ULP`；flag 须精确相等 | 任务书 §3.2 / 生态开源精度标准 |
| 性能标准 | warmup 后有效采样 >50 次；平均单次耗时不高于任务书 §3.3 标杆及随任务 `gpu_baseline` 口径 | 任务书 §3.3 |
| golden | `cblas_srotmg`（Netlib）单标杆 | 任务书 |

精度覆盖重点：

- flag=-2 快速返回、flag=-1 全零、flag=0/1 主路径
- GAM 缩放保护（大数/小数）
- 负 d1、负 d2、y1=0、边界 su
- 空指针负向用例
- Inf/NaN 特殊值（可终止、行为可判定）

性能覆盖重点：

- 单次调用延迟（一般 flag=1、flag=-2、缩放保护路径）
- 批量连续调用（iters=1000/10000）吞吐均值

## 兼容性分析

- API 与 arch35 共用 `aclblasSrotmg`，不新增平行接口。
- Host/Device 双路径判别与仓内 rotmg README 既有口径一致。
- arch22 新增目录不影响既有 arch35 产品线。

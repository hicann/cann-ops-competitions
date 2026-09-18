# aclblasCsrot 算子设计文档

> 本文档依据 `aclblasCsrot_Atlas950PR_task_doc.md`（Ascend 950PR 任务书）与官方 `design_template.md` 编写，面向社区任务评审。
> 算子实现提交至 ops-blas 仓 `blas/rot/arch35/`，测试代码提交至 `test/rot/csrot/arch35/`。
> 架构对齐同仓已合入的 arch35 `aclblasSrot`；复数语义对齐 Netlib `csrot.f` / cuBLAS `cublasCsrot`。

---

# 需求背景（required）

## 需求来源

CANN 社区任务 2026 — 算子开发任务 `aclblasCsrot`（Atlas 950PR / 西安邮电大学训练营）。

- 任务书：`aclblasCsrot_Atlas950PR_task_doc.md`
- 上游仓库：https://gitcode.com/cann/ops-blas ，目标目录 `blas/rot/arch35/`
- 接口声明：`include/cann_ops_blas.h`（已存在，禁止定义 950PR 私有平行接口）

## 背景介绍

### aclblasCsrot 算子实现优化

ops-blas 仓的 `rot` 算子族提供 Givens 平面旋转。实数版本 `aclblasSrot` 已完成 arch35（Ascend 950PR）适配；复数版本 `aclblasCsrot` 此前只有 arch22（Atlas A2/A3）实现，本任务基于 Ascend C 为 arch35 实现完整、可合入的版本。

### 算子实现现状分析

| 参数 | 参数含义 | 数据类型 | 当前支持 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库句柄 | scalar | 句柄式 API | 非空 | — |
| n | 元素个数 | int | 任意整数 | n≤0 为 no-op | — |
| x / y | 复数向量 | COMPLEX64 | Device，原地 | n>0 时非空 | 逻辑 [n] |
| incx / incy | 步长 | int | 正/负/零 | 不拦截；(n-1)*inc 须在 int32 内 | — |
| c / s | 旋转因子 | FLOAT32 指针 | Host 或 Device | 非空；单位旋转不短路 | 标量 |

计算公式（y 使用原始 x）：

```
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - s * x[k]
```

arch22 实现不能直接复用到 arch35，主要差距：

1. **舍入语义**：arch22 易走 FMA/`Axpy` 路径；Netlib 对每个乘积、每个加法分别舍入，相消场景相对误差可发散。
2. **入口校验**：需完整对齐 handle / n≤0 / 空指针 / int32 偏移溢出。
3. **并行与步长**：arch22 未覆盖 arch35 多 AIV + 连续/stride 双路径及负/零步长完备语义。
4. **c/s 指针位置**：任务要求 Host/Device/mixed，需 `aclrtPointerGetAttributes` 判定。

本实现在 `blas/rot/arch35/` 新增 `csrot_{host,kernel,tiling_data,kernel.h}`，沿用 `srot` 双路径架构，将 complex64 视为交错 float lane（每元素 2 个 float）。

### 算子功能分析

对两个等长复数向量 `x`、`y` 做原地 Givens 旋转；`c`、`s` 为**实数**标量（与 `crot` 区分：后者 `s` 为复数）。典型用途为 QR 等数值算法中的平面旋转，属 BLAS Level-1。

# 需求分析（required）

## 需求描述

使用 Ascend C 实现 `aclblasCsrot`：

- 数据类型：complex64 向量 + float 旋转因子
- 硬件：Ascend 950PR / 950DT（arch35），CANN 9.1.0
- 边界与舍入对齐 Netlib `csrot.f`
- 性能满足任务书 §3.3 标杆耗时

## 需求拆解

1. **双路径**：`incx==1 && incy==1` 走连续 SIMD；其余（含负、零、混合）走 stride SIMT
2. **原地安全**：先读齐原始分量再写；允许 x/y 别名
3. **边界对齐 Netlib**：`n≤0` no-op；零步长串行累加；负步长从尾端；`c==1&&s==0` 不短路
4. **c/s 位置自动判定**：host / device / mixed
5. **分步舍入**：禁止收缩为 FMA（连续路径用 `Muls`+`Add`；stride 路径强制乘积舍入）
6. 精度 / 性能达到 §3.2 / §3.3

# 详细设计（required）

## 算子分析

### 数学公式

```
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - s * x[k]        // x[k] 为旋转前

k = (i-1) * incx,  j = (i-1) * incy,  i = 1..n
```

实部 / 虚部独立旋转（无复数乘法交叉项）：

```
xr' = c*xr + s*yr;  xi' = c*xi + s*yi
yr' = c*yr - s*xr;  yi' = c*yi - s*xi
```

据此：**一条向量 `Muls` 可同时作用实部与虚部 lane**。

### 支持数据类型

| 参数 | 类型 |
|------|------|
| x, y | complex64（`aclblasComplex`，8 字节） |
| c, s | float32 |
| 计算精度 | float32 |

### 支持形状

逻辑一维 `[n]`，物理跨度 `1+(n-1)*|inc|`。n / inc 为任意整数（含 0、负值）。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

**内存布局**：`aclblasComplex { float real; float imag; }` 连续存放时，**n 个复数 ≡ 2n 个 float lane**（偶 lane 实部、奇 lane 虚部）。kernel 以 float 视图操作，无需额外 layout 转换。

**TilingData（计数以 complex 元素为单位）**：

| 字段 | 含义 |
|------|------|
| `tilingKey` | 0=连续 SIMD，1=stride SIMT |
| `totalN` | 元素总数 |
| `cosValue` / `sinValue` | c/s 在 host 时的标量；device 时占位 0 |
| `cIsDevice` / `sIsDevice` | 位置标志 |
| `perCoreN` / `remainder` | 连续路径分核 |
| `tileSize` | 连续路径 UB 单次搬运上限（complex 个数） |
| `incx` / `incy` | stride 路径步长 |
| `nthreads` | stride 路径每 block 线程数 |

##### 1. 分核策略

**连续路径**：`numBlocks = min(n, aivCoreNum)`，`perCoreN = n/numBlocks`，余数给前若干核。不做 block 对齐取整，保证小 n 负载均衡；`DataCopyPad` GM 侧仅需 1 字节对齐。

**stride 路径**：`numBlocks = min(ceilDiv(n, SIMT_MIN_THREAD_NUM), aivCoreNum)`，`nthreads` 按每核元素数向上对齐到最小线程粒度并封顶 `SIMT_MAX_THREAD_NUM`。kernel 内 grid-stride 遍历。

##### 2. 数据分块与 UB

连续路径 TQue 三级流水，**5 个 UB tile**：

- `inQueueX` / `inQueueY`：VECIN
- `outNewX` / `outNewY`：VECOUT
- `workBuf`：VECCALC（暂存 `round(s*y)` / `round(s*x)`）

`tileSize` 按 `UB_SIZE/(5*8)` 并向下对齐到 `COMPLEX_PER_BLOCK`（32/8=4），保证 32B 边界。

**原地安全**：Compute 阶段保留 VECIN 中原始 x/y，结果写入独立 VECOUT，读齐再写回。

**零步长**：Netlib 对同一地址串行累加。多线程并发会破坏语义；当 `incx==0 || incy==0` 时仅 `block0.thread0` 串行执行 n 次，其余线程返回。

##### 3. tilingKey

| tilingKey | 条件 | kernel |
|---|---|---|
| 0 | `incx==1 && incy==1` | SIMD 连续 |
| 1 | 其余 | SIMT stride |

c/s 位置**不进入** tilingKey，由 `cIsDevice`/`sIsDevice` 在 kernel 内分支。

**负向检测**：

- `handle == nullptr` → `HANDLE_IS_NULLPTR`
- `n <= 0` → `SUCCESS`（不校验其余）
- `n > 0` 且 x/y/c/s 空 → `INVALID_VALUE`
- `(n-1)*inc` 超 int32 → `INVALID_VALUE`
- `aclrtPointerGetAttributes` 失败 → `INVALID_VALUE`

#### 3.2.2 kernel 侧设计

Init + Process（CopyIn / Compute / CopyOut）。

**连续路径（key=0）**：

```
CopyIn : DataCopyPad 搬入 x/y（字节数 dataCount*8，lane 偏移 curOffset*2）
Compute: laneCount = dataCount * 2
         Muls(newX, x,  c,  laneCount)   // round(c*x)
         Muls(work, y,  s,  laneCount)   // round(s*y)
         Add (newX, newX, work, laneCount)
         Muls(newY, y,  c,  laneCount)   // round(c*y)
         Muls(work, x, -s,  laneCount)   // round(-s*x)
         Add (newY, newY, work, laneCount)
CopyOut: DataCopyPad 写回
```

**stride 路径（key=1）**：按 Netlib 索引；负步长从尾端；乘积经强制单次舍入（避免 -O3 FMA 收缩）；四分量先读寄存器再写。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Ascend 950DT（arch35） | √ |
| Atlas A2 / A3（arch22） | √（沿用既有实现，本任务不改动） |

## 算子约束限制

- stride 路径 `(n-1)*incx/incy` 须在 int32 内
- `c`、`s` 为 float 实数；复数 `s` 属 crot，不在本算子范围
- x、y 位于 Device 内存，原地更新
- 异步执行依赖 `aclblasSetStream`；读回前须同步

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32 分量：atol=2^-16，rtol=2^-10，matched_ratio≥0.99，max_abs_error≤max(1e-2,32×ULP) | 任务书 §3.2；生态精度标准 |
| 辅判据 | MERE≤2^-13，MARE≤10×2^-13 | ops-blas ST 框架 |
| 性能标准 | n=1M/2M/4M（inc=1）Avg ≤ 21.69 / 43.19 / 119.38 us | 任务书 §3.3 |
| 内存标准 | 不涉及（无 workspace） | 任务书 §3.4 |

自测：任务包 CSV（约 1200 条）+ `verify_accuracy.py` / `verify_performance.py`；golden 由 cblas/Netlib `csrot` 生成，实部/虚部分别比对。

## 兼容性分析

- 接口与 `cublasCsrot` 参数一一对应，无额外映射。
- arch22 实现保留，按 SOC 分派，不影响既有 A2/A3 产品线。
- 合入后更新 `blas/rot/README.md`：Ascend 950PR / 950DT 标注为**支持**，并修正接口原型与头文件一致。

## 文件规划

```
blas/rot/arch35/
  csrot_host.cpp
  csrot_kernel.cpp
  csrot_kernel.h
  csrot_tiling_data.h
test/rot/csrot/arch35/
  csrot_test.cpp
  csrot_npu_wrapper.h
  csrot_test.csv
  （param/golden 可与上层共享或参照 srot）
```

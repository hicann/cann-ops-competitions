# aclblasSspr 算子设计文档

| 项目               | 内容                                                         |
| ------------------ | ------------------------------------------------------------ |
| 任务名称           | 算子实操工坊-aclblasSspr 设计文档-A2/A3（taskId 目录 `08-aclblasSspr-A2A3`） |
| 适配硬件           | Atlas A2 / Atlas A3 系列产品（arch22，性能基准设备 Ascend910B3） |
| NpuArch / 编译参数 | DAV_2201 / `--npu-arch=dav-2201`（纯 Vector 算子，vec 变体） |
| CANN 版本          | CANN 9.1.0                                                   |
| 涉及仓库           | cann/ops-blas（`blas/spr/arch22/`、`include/cann_ops_blas.h`、`test/spr/arch22/`） |
| 对标接口           | cuBLAS `cublasSspr` / Netlib `sspr`                          |

> 本文档按官方 `resources/design_template.md` 的 required 章节组织，内容为本工程 **as-built**
> 实现（ops-blas 分支实测 1201 条 GTest 全绿、性能四标杆达标），非初版设计稿。

# 需求背景（required）

## 需求来源

昇腾社区任务广场《aclblasSspr A2/A3 算子开发》（算子实操工坊-北京站）。要求参考 cuBLAS
`cublasSspr` 的功能与参数语义，基于 ops-blas 开源仓工程框架，使用 Ascend C 在 Atlas A2/A3
上开发单精度实数对称矩阵秩-1 更新（打包存储）算子 `aclblasSspr`，完成设计、开发、测试全流程。

## 背景介绍

### aclblasSspr 算子简介

BLAS 对称矩阵秩-1 更新（Symmetric Packed Rank-1 Update，sspr）计算 `A := alpha · x · xᵀ + A`，
其中 A 为 n×n 对称实数矩阵，以 packed（压缩）格式存于长度 `n(n+1)/2` 的一维数组 AP 中，无前导
维 lda；x 为 n 元素单精度实数向量，alpha 为标量。`aclblasSspr` 为其 float32 版本，语义对齐
cuBLAS `cublasSspr` 与 Netlib `sspr`。

ops-blas 仓现状：arch35 已有 `aclblasSspr` 实现（SIMT 模型，DAV_3510/Ascend950）；arch22
（910B 系，SIMD/MemBase 模型）尚无实现。本任务补齐 arch22，与 arch35 独立编译、互不照抄
（两者编程模型不同：arch35 用 `__simt_vf__`/threadIdx 线程级并行，arch22 用 TPipe/TQue 向量级
UB 切分）。

### 接口定义

```cpp
aclblasStatus_t aclblasSspr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const float* alpha, const float* x, int incx, float* AP);
```

与 cuBLAS `cublasSspr` 逐参数对齐（无 lda、无 beta，AP 为打包存储一维数组）。声明已存在于仓侧
`include/cann_ops_blas.h`（签名 1:1），本任务不新增私有平行接口。x/AP 为 device 指针，alpha 为
host 指针（Host 侧解引用按值传入 kernel，零 H2D tiling）。

### 参数现状分析

| 参数   | 含义                     | 数据类型      | 支持类型              | 约束                       | 形状                          |
| ------ | ------------------------ | ------------- | --------------------- | -------------------------- | ----------------------------- |
| handle | 库句柄（绑定 stream）    | 指针          | —                     | 非空                       | —                             |
| uplo   | 引用/更新的三角          | 枚举          | UPPER(121)/LOWER(122) | 非法→INVALID_ENUM          | —                             |
| n      | 矩阵阶                   | 标量 int      | —                     | n≥0；n>16384→NOT_SUPPORTED | —                             |
| alpha  | 标量系数                 | host float*   | FLOAT32               | 非空指针；alpha=0 为 no-op | —                             |
| x      | 输入向量                 | device float* | FLOAT32               | n>0 且 alpha≠0 时非空      | 逻辑 n，物理 1+(n-1)·\|incx\| |
| incx   | x 步长                   | 标量 int      | —                     | ≠0，≠INT_MIN；支持负步长   | —                             |
| AP     | 打包对称矩阵（原地覆写） | device float* | FLOAT32               | n>0 且 alpha≠0 时非空      | n(n+1)/2                      |

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（910B3）上通过 handle 绑定 stream 直调 Ascend C kernel，实现 `aclblasSspr`：

1. 计算 `A := alpha · x · xᵀ + A`，原地覆写 AP；packed 列主序存储：
   - `UPPER`：第 j 列自对角 A(j,j) 向上至 A(0,j)，存于 `AP[j(j+1)/2 + i]`（i ≤ j）；
   - `LOWER`：第 j 列自对角 A(j,j) 向下至 A(n-1,j)，存于 `AP[j(2n-j+1)/2 + (i-j)]`（i ≥ j）。
2. `incx` 支持正/负步长（负步长按 Netlib 语义反向遍历）；`incx=0` 非法；`n=0`、`alpha=0` 为合法
   no-op（quick return SUCCESS，不引用 x、AP 不变）。
3. 精度满足 FLOAT32 生态标准：rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio≥0.99、max_abs_error≤max(1e-2, 32·ULP)。
4. 性能不低于任务书 §3.3 标杆（910B3，warmup 后 >50 次采样取均值）：
   n=512 UPPER 11.31us / 1024 LOWER 27.07us / 2048 UPPER 79.93us / 4096 LOWER 475.91us。

## 需求拆解

1. **接口与参数校验**：复用 `cann_ops_blas.h` 声明；按任务书 §2.4 的校验序返回状态码（见详细设计
   host 侧）。**关键：非法 uplo 返回 `INVALID_ENUM`**（官方竞赛 CSV `TC_ED_114` uplo=999 要求
   INVALID_ENUM；不采用 arch35 的 INVALID_VALUE）。
2. **packed 布局与三角引用**：仅引用/更新 uplo 指定三角；packed 列内连续（stride=1），适合批量搬运。
3. **kernel 设计**：外积更新向量化（`Muls`+`Add`）、多核**按 COST 前缀和二分开列**负载均衡、
   **块级 ≥16KB 连续 MTE2 搬运** + 块内逐列计算、incx=±1 快路径 / \|incx\|>1 通用 Gather 抽取路径。
4. **测试工程**：`test/spr/arch22/`，CSV 驱动 GTest（官方 1200 条），golden 由 cblas（Netlib
   `cblas_sspr`）生成；覆盖边界/负向/精度/步长/Inf-NaN 用例。

# 详细设计（required）

## 算子分析

### 数学公式

`A := alpha · x · xᵀ + A`。对 AP 中第 j 列、第 i 行元素：

- UPPER（i ≤ j）：`AP[j(j+1)/2 + i] += alpha · x[j] · x[i]`
- LOWER（i ≥ j）：`AP[j(2n-j+1)/2 + (i-j)] += alpha · x[j] · x[i]`
- 对角 i==j 按同式更新（`alpha·x[j]²`），无特判、无 1/2 因子。

**关键观察（决定实现方案）**：

- packed 布局中每列在 AP 内连续（stride=1）→ 适合一次大块 DataCopyPad 搬运；
- 外积更新无归约（每个 AP 元素只被一个 (i,j) 更新）→ 无需 ReduceSum、无原子操作；
- 列间 AP 区域互不重叠 → 多核并行处理不同列，无跨核同步；
- 每列内 `alpha·x[j]` 为标量常数 → 提到内循环外，用 `Muls`（标量乘向量）+ `Add` 处理整列段。

**按列向量化统一结构**：

```
LOWER 列 j:  AP[s(j) : s(j)+L(j)] += (alpha·x[j]) · x[j:n]     L(j)=n-j,  s(j)=j(2n-j+1)/2
UPPER 列 j:  AP[s(j) : s(j)+L(j)] += (alpha·x[j]) · x[0:j+1]    L(j)=j+1,  s(j)=j(j+1)/2
```

### 支持数据类型

输入 x/AP/alpha 固定 FLOAT32（需求指定，不扩展 half/bf16/int）；n/uplo/incx 为 Host 标量/枚举。

### 支持形状

支持域 `0 ≤ n ≤ 16384`（SSPR_MAX_SUPPORT_N，受 x 常驻 UB 预算所限）；`n>16384` Host 返回
`NOT_SUPPORTED`（§2.4.4 流式回退为未来兜底，as-built 未实现）。任务书测试域 n≤4096 与 4 标杆
性能 case 全落在支持域内。incx 支持任意非零整数（含负）。

## 算子实现

### 实现方案

#### host 侧设计（`blas/spr/arch22/sspr_host.cpp`）

**参数校验顺序与状态码**（对齐任务书 §2.4，逐字保留 as-built）：

```
1 handle==null            -> HANDLE_IS_NULLPTR
2 uplo∉{UPPER,LOWER}      -> INVALID_ENUM      ← 关键差异（arch35 为 INVALID_VALUE）
3 n<0                     -> INVALID_VALUE
4 alpha==null             -> INVALID_VALUE
5 incx==0 或 incx==INT_MIN-> INVALID_VALUE
6 n==0 || *alpha==0       -> SUCCESS（quick return，不读 x/AP，不 launch）
7 x==null || AP==null     -> INVALID_VALUE
8 n>SSPR_MAX_SUPPORT_N    -> NOT_SUPPORTED
9 动态 coreNum + chunkElem + sspr_kernel_do 启动 -> SUCCESS
```

**分核策略（动态满核 + 负载均衡）**：
`coreNum = min(GetAivCoreCount(), n, ⌈N / SSPR_MIN_ELEM_PER_CORE⌉)`（N=n(n+1)/2，
MIN_ELEM_PER_CORE 初值 2048，上板标定），至少 1；小 n 退化少核/单核以降低 launch/同步固定开销。

**UB 切分与 tiling（零 H2D）**：UB 容量经 Host-only `PlatformAscendCManager::GetCoreMemSize(UB)`
实测（禁止硬编码），`sspr_compute_chunk_elem` 据此反推**连续传输块**大小 chunkElem（块目标
≥16KB=4096 float，向下 32B 对齐），连同 n/uplo/incx/alpha 全部按值传入 kernel，无 TilingData
结构体 H2D、无 workspace。

**kernel 启动**：host.cpp 声明、kernel.cpp 定义 `sspr_kernel_do(x, ap, n, lower, incx, alpha,
chunkElem, numBlocks, stream)`，内部 `sspr_kernel<<<numBlocks, nullptr, stream>>>(...)` 异步提交
到 handle->stream（host/kernel 分属不同编译器 TU，`<<<>>>` 须留在 device TU）。

#### kernel 侧设计（`blas/spr/arch22/sspr_kernel.cpp`）

**block-based TPipe/TQue 单 kernel 发射**（DAV_2201 SIMD/MemBase，非 arch35 SIMT），类
`KernelSspr` 封装，`Init` + `Process` 两阶段：

1. **Init**：设置 GM buffer；按 **COST 前缀和整型二分**求本核列区间 `[j0, j1)`——
   `cost(j) = s(j) + K·j`（K=SSPR_COL_OVERHEAD_ELEMS=64，把每列固定的 tiny-store/Gather 开销
   计入，均衡 UPPER 大量 tiny 列），`t_c = totalCost·c/coreNum`，对单调 cost 二分 O(log n)，末核
   补到 n；分配 xUBQueue(VECIN 常驻 x) / apQueue(VECIN depth2) / outQueue(VECOUT depth2) /
   xScratch·apAdd(VECCALC 对齐 scratch) / offShift(Arange 生成的 {0,4,8,...} 字节偏移表)。
2. **BuildResidentX（序言一次）**：把 x 按逻辑序 0..n-1 载入常驻 xUB——incx==+1 一次连续
   DataCopyPad；\|incx\|≠1 先载物理跨度再一次 `Gather` 抽成逻辑序（负步长在物理基址换算 + 反向
   偏移表内消化）。此后热循环下标全为逻辑序，不受 incx 符号/步长影响。
3. **Process（逐块）**：把本核列区间按 chunkElem 切成**连续传输块**；每块：
   - **一次 MTE2** `DataCopyPad` 把整块 packed 区间载入常驻 apBuf（≥16KB，减少 UPPER tiny 列的
     逐列 DMA 次数）；
   - 块内**逐列** VEC：x 段与 AP 加数段经 `Gather` 重定位到 32B 对齐 scratch（mid-column 仅 4B
     对齐，不满足向量操作数 32B 起始约束；`r0==0` 时 x 段免 Gather）→ 取列标量
     `sc = alpha · xUB.GetValue(j)` → `Muls(rBuf, xSeg, sc, cl)` + `Add(rBuf, rBuf, apAdd, cl)`
     产出到 VECOUT 队列（队列自动完成 V→MTE3 同步，免手写跨流水事件）；
   - **逐列字节精确** `DataCopyPad` 写回 GM（严格 ⊆ 本列 → 核间零重叠，无跨核竞争/同步）。

**关键设计决策**：

- `Muls`+`Add` 两步（非初版 `Axpy` 单指令、非 `MulAddDst`）：让结果落标准 VECOUT 队列复用队列
  同步；Vector 非 memory-bound 瓶颈，2 指令不影响达标（round_006 4/4 实证）。
- 块级 ≥16KB MTE2（非 round_005 的逐列 GM 重载）：逐列 tiny DMA 延迟/建立开销主导，round_005 在
  LOWER 1024/4096 慢约 2×，round_006 回退块设计后达标。
- 无原子、无跨核屏障：packed 每个 AP 位置唯一属一列，多核处理不相交列区间。

## 支持硬件

| 支持的芯片版本                              | 是否勾选 |
| ------------------------------------------- | -------- |
| Atlas 800I/T A2（910B 系，arch22/DAV_2201） | √        |
| Atlas A3（910B3 性能基准设备）              | √        |

## 算子约束限制

- dtype 固定 FLOAT32，不支持 half/bf16/int。
- 支持域 `n ≤ 16384`；超出返回 `NOT_SUPPORTED`（x 常驻 UB 预算所限，流式回退未实现）。
- AP 原地覆写；packed 存储无 lda、无 beta。
- Inf/NaN 输入不做特殊分支，遵循 IEEE754 传播；乘法次序 `(alpha·xⱼ)·xᵢ` 与 Netlib
  `(alpha·xᵢ)·xⱼ` 不保证逐比特一致，按容差判定（任务书声明非 bit-exact）。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述                                                         | 标准来源                          |
| -------- | ------------------------------------------------------------ | --------------------------------- |
| 精度标准 | FLOAT32：rtol=2⁻¹⁰、atol=2⁻¹⁶、matched_ratio≥0.99、max_abs_error≤max(1e-2, 32·ULP)；golden = Netlib `cblas_sspr` 单标杆，仅 uplo 三角打包位置比对，非 bit-exact | 任务书 §3.2                       |
| 性能标准 | 910B3，warmup 后 >50 次采样均值，各 case ≤ 标杆：512U 11.31us / 1024L 27.07us / 2048U 79.93us / 4096L 475.91us | 任务书 §3.3（= gpu_baseline/0.8） |

**as-built 实测**（代码在 `cann/ops-blas` MR !474，分支 `feat/aclblasSspr`）：

- 功能：1201 条 GTest 全绿（1200 官方用例 + 1 冒烟），`Failed: 0`。

- 性能（msopprof `Task Duration`，1800MHz 额定频率，Block Dim=40 满核）：

  | case       | shape        | 样本 | p50(us) | p90(us) | 标杆(us) |
  | ---------- | ------------ | ---- | ------- | ------- | -------- |
  | TC_PF_1001 | n=512 UPPER  | 60   | 8.060   | 8.340   | 11.31    |
  | TC_PF_1002 | n=1024 LOWER | 60   | 20.920  | 21.700  | 27.07    |
  | TC_PF_1003 | n=2048 UPPER | 60   | 39.941  | 40.781  | 79.93    |
  | TC_PF_1004 | n=4096 LOWER | 30   | 104.582 | 105.342 | 475.91   |

  p50 与 p90 均低于标杆。采集需 `msprof op --replay-mode=application --aic-metrics=BasicInfo`：
  AP 为 in-place RMW，kernel 级回放无法逐实例还原设备状态，默认模式下仅首个实例可完整解析。
  n=4096 受单次采集时长限制采用 30 样本，未补足 50 样本，此处如实标注。

- 交叉核对：与直调工程（同一 kernel 逻辑）round_006 的 8.37 / 20.84 / 39.99 / 105.41 us 偏差
  ≤ 3.7%，两条测试通路结果一致。

## 兼容性分析

- 新增 arch22 实现，与既有 arch35（SIMT）独立编译，符号不冲突（各自 `sspr_kernel_do`）；不改
  arch35 行为。
- 接口签名复用 `include/cann_ops_blas.h` 既有声明，无 ABI 变更、无私有平行 API。
- 测试共享层 `test/spr/sspr_param.h` 采用**向后兼容超集补丁**：同时吃官方竞赛 CSV 方言
  （`x_fill/ap_fill`、`ACLBLAS_FILL_MODE_*`、alpha 空=`null`）与仓内 arch35 方言（`x/ap`、
  `ACLBLAS_UPPER`、`NULLPTR`），不改动共享 `csv_loader.h`/`fill.h`，对 arch35 零回归。

## 测试工程

- 目录：`test/spr/arch22/`（`sspr_test.cpp` + `sspr_npu_wrapper.h` + `sspr_test.csv`）+ 共享
  `test/spr/{sspr_param.h(补丁), sspr_golden.h(不改), CMakeLists.txt(不改)}`。
- golden：cblas Netlib `cblas_sspr`（`aclblasSspr_cpu`）；精度门禁 MIXED_TOLERANCE(ACL_FLOAT)。
- 用例：官方 1200 条（1000 精度 + 200 性能），uplo UPPER×606/LOWER×593/非法×1；expect_result
  SUCCESS×1194 / INVALID_VALUE×5 / INVALID_ENUM×1；含 Inf/NaN（TC_FL_087~090）与全部负向
  （TC_ED_107~116）。
- 复现：`bash build.sh --soc=ascend910b3 --ops=sspr`（910b3→arch22 自动发现）→ 跑 `sspr_test`
  GTest 全绿。
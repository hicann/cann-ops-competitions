# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务 2026 年 8 月社区任务 —— aclblasCcopy 算子开发（950），任务列表编号 44。

## 背景介绍

### aclblasCcopy算子实现优化

本算子为 ops-blas 仓新增算子，基于 Ascend C 编程语言实现：连续路径编写专用的高性能搬运 kernel（单满 UB 缓冲、int64_t 元素、串行逐 tile 读写，经多方案真机 A/B 实测择优）；非连续路径复用同族已有算子 scopy 的 AIV kernel，通过 host 侧参数转换将复数拷贝转化为等价的 float 拷贝。

算子实现路径：`blas/copy/arch35/ccopy_host.cpp` + `blas/copy/arch35/ccopy_kernel.cpp`

API 路径：`include/cann_ops_blas.h` 中已有 `aclblasCcopy` 声明

### aclblasCcopy算子参数说明

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 库上下文句柄 | aclblasHandle_t | - | 指向已创建的有效句柄 | - |
| n | 向量元素个数（复数） | int | - | n ≥ 0 | 逻辑一维 [n] |
| x | 源复数向量（只读） | const aclblasComplex* | COMPLEX64 | n>0 时不可为 nullptr | 物理长度 1+(n-1)*\|incx\| |
| incx | x 元素步长 | int | - | incx ≠ 0，可正可负 | - |
| y | 目标复数向量（原地覆盖） | aclblasComplex* | COMPLEX64 | n>0 时不可为 nullptr | 物理长度 1+(n-1)*\|incy\| |
| incy | y 元素步长 | int | - | incy ≠ 0，可正可负 | - |

计算公式：`y[j] = x[k]`，其中 `i = 1..n`，`k = 1+(i-1)*incx`，`j = 1+(i-1)*incy`（1-based 索引，兼容 Fortran）

### aclblasCcopy算子功能分析

aclblasCcopy算子功能：将复数向量 x 逐元素拷贝到复数向量 y，纯数据搬移。

输入：复数向量 x（只读），标量 n、incx、incy

输出：复数向量 y（原地覆盖）

支持数据类型：COMPLEX64（aclblasComplex = {float real; float imag}）

支持广播：不涉及（一维向量逐元素拷贝）

# 需求分析（required）

## 需求描述

在昇腾 Ascend 950PR NPU 上使用 Ascend C 开发单精度复数（COMPLEX64）向量拷贝算子 `aclblasCcopy`，实现与 cuBLAS `cublasCcopy` / Netlib BLAS `ccopy` 接口完全对齐的复数向量拷贝功能，要求 bit-exact 精度，性能对标任务书三档标杆。

## 需求拆解

1. 支持 COMPLEX64（aclblasComplex）数据类型，实部/虚部各 float32，内存布局为 `{float real; float imag;}`
2. 支持连续拷贝（incx=1, incy=1）和非连续拷贝（|incx|>1 或 |incy|>1），含负步长
3. 接口签名与 `include/cann_ops_blas.h` 已有声明一致，禁止定义 950PR 私有平行接口
4. 拷贝为纯数据搬移，bit-exact 精度：输出 y 与 golden 逐位相等，非写区域不被污染
5. 性能达标：n=1048576/4194304/16777216 三种典型 case 平均耗时不高于标杆
6. 测试工程基于 ops-blas 仓 CSV 驱动 GTest 框架，参照 `test/copy/scopy/` 模式扩展

# 详细设计（required）

## 算子分析

### 数学公式

y[j] = x[k]，其中 i = 1..n，k = 1+(i-1)*incx，j = 1+(i-1)*incy

### 支持数据类型

| 数据类型 | 说明 |
| --- | --- |
| COMPLEX64 | 单精度复数，实部/虚部各 float32，内存布局 `{float real; float imag;}`，等价于连续 2 个 float，整体 8 字节可按 1 个 int64_t 原子搬移 |

### 支持形状

逻辑一维向量，长度 n（复数元素个数）。物理长度由步长决定：1 + (n-1) * |inc|。不涉及广播。

## 算子实现

### 实现方案

**核心思路**：`aclblasComplex` 在内存上为 8 字节连续结构（实部在前、虚部在后），等价于 `float[2]`，也等价于 1 个 `int64_t`。按步长分两条路径：

**连续路径（incx=1, incy=1）——专用高性能 kernel（`ccopy_kernel.cpp`）**：

将 1 个复数元素视为 1 个 `int64_t`（8 字节），每个 32B MTE 数据块搬移 4 个复数元素。经 950PR 真机多方案 A/B 实测（单缓冲串行 / 双缓冲流水线 / 三缓冲预读 / 交错分配），**单满 UB 缓冲 + 串行逐 tile 读→写**为最优结构：

- Ascend 950PR 的 UB 读（MTE2）/写（MTE3）端口实际串行，多缓冲"重叠"只增加 SetFlag/WaitFlag 同步开销（实测单缓冲比双缓冲快 17~39%）
- 单缓冲使 tile 占满全部 UB（~248KB），tile 数与同步点减半
- 元素粒度均分到全部 56 个 AIV 核（负载均衡优于 tile 粒度分配）
- host 侧缓存 tiling 与核数，热路径零日志、零平台查询

**非连续路径（|incx|>1 或 |incy|>1）——复用 scopy kernel**：拆分为实部和虚部两次独立调用，每个复数元素占 2 个连续 float，因此 float 步长 = 2 × 复数步长。

```
实部: scopy(n, (float*)x + 0, 2*incx, (float*)y + 0, 2*incy)
虚部: scopy(n, (float*)x + 1, 2*incx, (float*)y + 1, 2*incy)
```

#### host侧设计

**参数校验**：

- handle == nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- n < 0 → ACLBLAS_STATUS_INVALID_VALUE
- n == 0 → ACLBLAS_STATUS_SUCCESS（no-op）
- x/y == nullptr → ACLBLAS_STATUS_INVALID_VALUE
- incx/incy == 0 → ACLBLAS_STATUS_INVALID_VALUE

**Tiling 策略**：

连续路径使用独立结构体 `CcopyTilingData`（`ccopy_tiling_data.h`），以复数元素为单位：

- `totalN`：复数元素总数 n
- `perCoreN`：每核基础元素数，4 元素（32B）对齐
- `extraBlockCores`：额外整块分配的核数
- `tailElements`：尾部非对齐元素数（0-3，分配给最后一个核，标量搬移）
- `tileSize`：(UB可用容量 / sizeof(int64_t)) 对齐到 4——单缓冲占满 UB

非连续路径复用 `ScopyTilingData`（float 级），半连续路径按 Gather 缓冲计容、纯非连续受 Compact 模式限制（≤4088）。

**分核策略**：

遵循满核原则，工作均匀分配到所有 AIV 向量核。numBlocks = min(n, aivCoreNum)。每个核自计算 offset：

```
myOffset_ = blockIdx_ * perCoreN + min(blockIdx_, extraBlockCores) * 4
myCount_ = perCoreN + (blockIdx_ < extraBlockCores ? 4 : 0)
         + (blockIdx_ == lastBlock ? tailElements : 0)
```

**数据分块和内存优化策略**：

- 连续路径：单满 UB 缓冲（~248KB/核），每 tile 串行 读→写，无缓冲切换与流水线同步开销
- 非连续路径：复用 scopy 的 DataCopyPad Compact 批量读写与 Gather 反转

**反向步长重排（非连续路径）**：

当 incx<0 XOR incy<0 时，需要在 UB 内对数据进行反转。通过 workspace 预生成偏移表（写一次、实部/虚部两趟复用），kernel 端通过 Gather 指令按偏移表收集实现反转。若 workspace 不可用，kernel 内标量循环计算偏移作为 fallback。

**执行流程**：

```
aclblasCcopy(handle, n, x, incx, y, incy)
  ├─ 参数校验（handle → n<0 → n==0 → x/y → incx/incy）
  ├─ 路径选择
  │   ├─ incx=1, incy=1 → 连续路径：ccopy_kernel_do（int64_t 单满UB缓冲）
  │   └─ 其他 → 非连续路径：floatIncx=2*incx, floatIncy=2*incy
  │              ws=PrepareCcopyOffsetWorkspace (写一次，两趟复用)
  │              Pass1: scopy_kernel_do(x+0, y+0, ...)  // 实部
  │              Pass2: scopy_kernel_do(x+1, y+1, ...)  // 虚部
```

#### kernel侧设计

**连续路径**：专用 `CcopyAIV` kernel（`ccopy_kernel.cpp`），KERNEL_TYPE_AIV_ONLY。

**Init 阶段**：
- 获取 blockIdx_，计算 myOffset_ / myCount_（元素粒度均分）
- SetGlobalBuffer：xGM_/yGM_ 按 int64_t 元素、span = totalN
- InitBuffer：单个 TBuf 占满 UB（tileSize × 8B）

**Process 阶段**：
- 对齐部分（4 元素对齐）按 tileSize 分块，每块串行执行：
  `DataCopy GM→UB（MTE2）→ WaitFlag(MTE2_V) → DataCopy UB→GM（MTE3）→ WaitFlag(V_MTE3)`
- 尾部 0-3 个复数元素：标量 GetValue/SetValue 搬移（最多 3 次，开销可忽略）
- 纯搬移无计算，实部/虚部逐位复制，天然 bit-exact

**非连续路径**：复用 `scopy_kernel.cpp` 的 `ScopyAIV`（Compact 批量读写 + Gather 反转），两趟分别搬移实部/虚部。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950T A2 | |
| Atlas 950T A3 | |
| Ascend 950PR | √ |

## 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | n ≥ 0；incx ≠ 0；incy ≠ 0；x、y 不可为 nullptr；非法参数返回 ACLBLAS_STATUS_INVALID_VALUE |
| 非连续 Tensor 支持 | 不要求（本批次不支持额外 leading dimension padding 场景；向量非连续由 incx/incy 表达） |
| broadcast 规则 | 不涉及，本算子为两个一维向量之间的逐元素拷贝 |
| dynamic shape 要求 | 不要求，n 为运行时入参 |
| 原地与视图语义 | y 被原地覆盖为 x 的拷贝，不返回视图 |
| 确定性计算要求 | 不要求（纯数据搬移，结果天然 bit-exact） |
| 空 Tensor 与 0 维处理 | n = 0 为合法 no-op，返回成功且不修改 y |
| 异步执行 | 依赖 aclblasSetStream 绑定 stream；读回 Device 结果前须同步 stream |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | bit-exact：输出 y 与 golden 逐位相等（实部/虚部分别），max_abs_error = 0，matched_ratio = 1.0 | 任务书 §3.2 + 生态精度标准（COMPLEX64 按 FLOAT32 分量：rtol=2^-10, atol=2^-16, matched_ratio≥0.99, max_abs_error≤1e-2 或 32*ULP） |
| 性能标准 | n=1048576 ≤ 13.01us；n=4194304 ≤ 23.71us；n=16777216 ≤ 67.69us；口径为设备侧平均单次 kernel 耗时（Avg time），warmup 后有效采样 >50 次取平均 | 任务书 §3.3（8-29 版） |

**性能测量口径说明**：

任务书 §3.3 要求"平均单次耗时（Avg time）"，其合理解读为 kernel 在设备侧的执行耗时，与 host 调度开销无关。故采用三种口径并以设备侧事件计时为主：

| 口径 | 方法 | 说明 |
| --- | --- | --- |
| 设备侧事件计时（主口径） | 100 次调用批量入队，aclrtEvent 首尾夹取，ElapsedTime/100 | 纯设备执行时间，排除 host 启动/同步开销 |
| chrono 墙钟（参考） | host 侧 100 次循环 + stream sync，chrono 计时 | 含 host 启动与同步开销，数值偏大（保守上界） |
| msprof 设备任务（交叉验证） | msprof 采集 KERNEL_AIVEC 任务时长取均值 | 官方 profiler，用于交叉验证 |

**实测性能（950PR 真机，56 AIV 核，CANN 9.1.0；warmup=10，5 轮 × 100 次采样取轮均值之 min）：**

| case | n (complex) | 标杆 (us) | 本实现墙钟 (us) | 本实现设备侧 (us) | scopy 设备侧 (us) | 判定（设备侧） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1048576 | 13.01 | 9.29 | **9.26** | 15.72 | **达标**（较基线 -41%） |
| 2 | 4194304 | 23.71 | 24.53 | 24.53 | 35.87 | 超标 3.5%（较基线 -32%） |
| 3 | 16777216 | 67.69 | 193.18 | 193.09 | 204.85 | 超标（带宽墙，见下论证） |

> 说明（9-1 真机复测，DevEnv_017543，warmup=10，5 轮 × 100 次采样取轮均之 min）：
> 设备侧事件口径与墙钟口径差异 <0.3%，证明 host 调度开销可忽略，实测即设备真实执行时间。
> 本实现设备侧较复用 scopy 基线快 5%~41%。

**性能结构选型依据（全部 950PR 上机 A/B 实证）**：

| 方案 | case1 | case3 | 结论 |
| --- | --- | --- | --- |
| 复用 scopy（float、双缓冲） | 15.95 | 204.24 | 基线 |
| 双缓冲手动事件流水线（int64_t） | 12.71 | 192.55 | 去队列+缓存见效 |
| 三缓冲深度 2 预读 | 13.68 | 201.46 | 反劣，弃 |
| **单满 UB 缓冲串行（采用）** | **9.51** | **188.58** | **最优** |

**case2/3 超标的物理上限论证**：

1. case3 数据量 128MB（读+写合计 256MB），设备侧实测 193.09us 对应双向 1.39 TB/s（纯设备执行时间，已排除 host 开销）；标杆 67.69us 隐含 ~4 TB/s 双向，为实测硬件上限的 ~2.8 倍。设备侧口径复测确认：case3 超标为数据量受 HBM 带宽物理约束所致。
2. 排除性实验（均上机验证）：56 AIV 核已全部利用；核间交错分配与连续分配结果一致（187.78 vs 187.77us，访存模式无关）；AIC 核（28 个）发射出现数据损坏不可用；aclrtMemcpy D2D 拷贝引擎 128MB = 236.33us（567.9 GB/s），本算子为其 2.5 倍。各结构/位宽/分配变体收敛于 187-195us，接近该硬件 HBM 带宽上限。
3. 综上，设备侧口径复测（9-1）确认 case2/3 仍超标（24.53/193.09us），属物理上限/标杆口径问题，恳请与任务方确认标杆测量口径与硬件环境。

**测试方案说明**：基于 ops-blas 仓 CSV 驱动 GTest 框架（1201 条用例 + 1 条独立 NullHandle 测试 = 1202 tests），参照 `test/copy/scopy/` 模式扩展。Golden 使用 cblas_ccopy（Netlib BLAS 复数实现）。测试类别覆盖：L0 基础(8)、L1 尺寸(38)、L2 步长(36)、L5 填充(12)、L5b 对齐(5)、L6 边界(7)、EX 扩展(894)、PF 性能(201)。真机全量结果：1202/1202 PASSED，0 失败。

## 兼容性分析

新算子，不涉及兼容性分析。接口声明复用 `include/cann_ops_blas.h` 中已有 `aclblasCcopy` 声明，可与其他产品线共用。

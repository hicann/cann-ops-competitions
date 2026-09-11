# aclblasCsscal 算子设计文档

> **算子名称**：aclblasCsscal
> **功能**：单精度复数向量 × 实数标量原地缩放（x = alpha * x）
> **目标硬件**：Ascend 950PR（arch35）
> **CANN 版本**：9.1.0
> **对标基线**：cuBLAS `cublasCsscal` / Netlib `csscal`
> **对应实现版本**：ops-blas 提交 8503a7c1368022b387067ae154dd7e2803c0faf9（csscal phase2 默认配置）。本文描述该版本实现；性能达标不等于社区验收已完成。

---

# 需求背景（required）

## 需求来源

CANN 训练营东南大学社区算子开发任务（任务目录：`09-27-aclblasCsscal-950`）：在昇腾 NPU（Ascend 950PR）上使用 Ascend C/CATLASS 开发 BLAS Level-1 算子 `aclblasCsscal`，完成算子设计、开发、测试全流程，验收通过后合入昇腾算子开源仓 `cann/ops-blas`。

设计依据：

- 任务书：`aclblasCsscal Atlas950PR task doc`（功能定义 §2、验收标准 §3、交付件 §4、PR 合入 §5）；
- 官方设计文档模板：`cann-ops-competitions/04_tasks/01_community-task-2026/resources/design_template.md`；
- 生态算子开源精度标准（opbase experimental standard）；
- ops-blas 仓现有 `sscal` arch35 实现及仓内测试工程。

## 背景介绍

### aclblasCsscal 算子说明

`aclblasCsscal` 是 BLAS Level-1 中的复数向量缩放算子，执行原地缩放：`x[j] = alpha * x[j]`（i = 1..n，j = 1+(i-1)*incx，1-based 索引兼容 Fortran），其中 **alpha 为 float 实数标量**，x 为单精度复数向量（complex64）。

实数标量对复数元素的语义为**实部、虚部分别乘以 alpha**，不涉及复数乘法。

与同族算子对比：

| 算子 | alpha 类型 | x 类型 | 公式 |
|---|---|---|---|
| `aclblasSscal` | float | float | x[i] = alpha * x[i] |
| **`aclblasCsscal`** | **float（实数）** | **complex64** | **x[i] = alpha * x[i]（实部/虚部分别乘 alpha）** |
| `aclblasCscal` | complex64 | complex64 | x[i] = alpha * x[i]（复数乘） |

`Csscal` 无复数交叉乘法，分量乘法次数为 `Cscal` 的一半；且 incx=1 时 complex64 在内存中连续排布为 `[re0, im0, re1, im1, ...]`，可等价视为长度为 `2*n` 的连续 float 流统一乘 alpha，为 AIV 向量化提供基础。

### 现有实现现状

- `include/cann_ops_blas.h` 已存在 `aclblasCsscal` 接口声明，arch35 目录下此前无对应实现；
- `aclblasComplex` 定义于 `include/cann_ops_blas_common.h`，实部/虚部各为 float32；
- 同族实数算子 `sscal` 已有 arch35 完整实现（AIV 连续路径 + SIMT 步长路径），可作为工程与 golden 封装（cblas 调用）的直接参照；
- 仓内 `test/scal/sscal/` 已有 CSV 驱动的 arch35 测试工程，`csscal` 测试工程参照其搭建。

### 算子功能分析

- **输入**：handle、n、alpha（float 实数标量指针，Host 内存）、x（complex64 向量，Device 内存）、incx（int，复数元素步长）；
- **输出**：x 原地更新；
- **支持数据类型**：complex64（实部/虚部各 FLOAT32）；
- **支持形状**：逻辑一维 `[n]`；incx>0 时物理长度为 `1+(n-1)*incx`；
- **不支持广播**：标量对一维向量逐元素操作，无广播语义；
- **原地更新**：不返回视图，仅原地改写 x。

---

# 需求分析（required）

## 需求描述

在 Ascend 950PR（arch35，CANN 9.1.0）上实现 `aclblasCsscal`，复用仓内公开接口：

```cpp
aclblasStatus_t aclblasCsscal(
    aclblasHandle_t handle, int n, const float* alpha,
    aclblasComplex* x, int incx);
```

功能、参数语义与 cuBLAS `cublasCsscal` 对齐，满足任务书精度标准（实部/虚部分别按 FLOAT32 判定）与性能标准（n=4M、incx=1 时 ≤ 43.02 us），代码合入 `cann/ops-blas`。

## 需求拆解

1. **功能正确性**
   - 实现 `x[j] = alpha * x[j]`，实部/虚部分别乘以 alpha；
   - 校验顺序固定，返回码优先级与 BLAS 语义一致（详见 3.2.1）；
   - 正确处理 no-op：n≤0 或 incx≤0 时合法返回 SUCCESS 且不修改 x；
   - 正确处理 alpha=0.0：**不是 no-op**，被索引元素实部/虚部须写成 bit-exact +0，步长间隙不变；
   - alpha=1.0 等价 no-op，可提前返回 SUCCESS。

2. **精度要求**
   - 实部、虚部分别与 cblas（Netlib BLAS）`csscal` golden 比对，逐元素条件 `|actual - golden| ≤ atol + rtol × |golden|`；
   - rtol=2^-10、atol=2^-16、matched_ratio≥0.99，且 max_abs_error ≤ max(1e-2, 32*ULP)；
   - alpha=0.0 用例额外做 +0 位级检查；
   - 正步长用例校验步长间隙（未被索引的 x 区间）保持不变。

3. **性能要求**
   - incx=1 连续访存为主路径：n=1M/2M/4M 分别 ≤ 13.57/21.05/43.02 us；
   - 配套 GPU baseline 的 200 条 `ratio = GPU_ms * 1000 / avg_us ≥ 0.4` 判定单独保留，不得用派生舍入值覆盖任务书门槛；
   - 有效采样超过 50 次取平均（实现中 warmup 后采样 100 次）。

4. **工程交付**
   - 实现位于 `blas/scal/arch35/`：`csscal_host.cpp`、`csscal_kernel.cpp`、`csscal_tiling_data.h`；
   - 接口声明复用 `include/cann_ops_blas.h` 已有声明，不重复导出；
   - 测试工程位于 `test/scal/csscal/arch35/`，配套 README 说明可复现的测试步骤；
   - 按任务书 §4/§5 提交设计文档 PR、代码 PR 与任务系统验收。

5. **验证完整性**
   - 如实区分"已验证结果、性能波动与待验收项"；任务书 §3.5 输入分布覆盖、950DT 支持声明等未闭环项不隐瞒（见"已知交付边界与待办"）。

---

# 详细设计（required）

## 算子分析

### 数学公式

```
x[j] = alpha * x[j]    (i = 1..n, j = 1+(i-1)*incx)
```

展开为分量形式：

```
x[j].real = alpha * x[j].real
x[j].imag = alpha * x[j].imag
```

其中 alpha 为 float 实数标量，x 为 complex64 向量。无复数交叉乘法；incx=1 时该公式等价于对长度为 `2*n` 的连续 float 流统一乘 alpha。

### 支持数据类型

| 参数 | 数据类型 | dtype | 说明 |
|---|---|---|---|
| alpha | float | FLOAT32 | 实数标量，按指针传入，指向 Host 可读内存 |
| x | aclblasComplex | COMPLEX64 | 复数向量，实部/虚部各 float32，指向 Device 内存 |
| n | int | INT32 | 向量元素个数（复数元素） |
| incx | int | INT32 | 复数元素步长，仅支持正步长 |

### 支持形状

逻辑一维 `[n]`；incx>0 时物理长度为 `1+(n-1)*incx`。incx 为复数元素间步长，不是 float 步长；非连续访问由 incx 表达，不涉及额外 leading dimension。

## 算子实现

### 实现方案

参照 `sscal` 采用 **AIV + SIMT 双路径**：

| 路径 | 触发条件 | 技术 | 定位 |
|---|---|---|---|
| AIV 路径 | `incx == 1` | complex64 视为 float 流，SIMD 向量乘 + UB 队列 | 连续访存主路径 |
| SIMT 路径 | `incx > 1` | 多线程逐元素处理 | 灵活支持任意正步长 |

**关键设计 1：complex64 展开为 float 流（仅 incx==1）**

complex64 内存排布 `[re0, im0, re1, im1, ...]` 在 incx==1 时等价于长度为 `2*n` 的连续 float 数组。对所有 float 统一乘 alpha 即等价于实部/虚部分别乘 alpha，因此 AIV 路径可直接复用向量乘逻辑，仅将 totalN 取为 `2*n`。

Host 侧先以 `uint64_t` 计算 `2*n`，缩窄到 `uint32_t` 前检查范围，避免乘法溢出。

**关键设计 2：8-float 对齐保证不劈开复数**

分核与 tile 均以 8 个 float（32B）为对齐单位。`perCoreN`、核内起始偏移、tile 大小均为 8 的倍数，因此核边界与 tile 边界落在完整复数（实部+虚部）之间，不会拆分单个复数元素。

**关键设计 3：alpha=0.0 不读输入，直接生成 +0**

IEEE 754 下 `0.0 * Inf = NaN`、`0.0 * NaN = NaN`，直接乘法无法得到 bit-exact 全零。alpha=0.0 时 AIV 路径不初始化输入队列、不搬入输入 GM，直接在 UB 内 `Duplicate(outLocal, 0.0f, count)` 生成正零并写回；SIMT 路径逐元素写 `0.0f`。

#### 3.2.1 host侧设计：

入参校验与快速返回顺序固定如下（保证返回码优先级与 BLAS 语义一致）：

```text
1. handle == nullptr            -> ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. n <= 0 || incx <= 0          -> ACLBLAS_STATUS_SUCCESS（合法 no-op，不修改 x）
3. alpha == nullptr             -> ACLBLAS_STATUS_INVALID_VALUE
4. x == nullptr                 -> ACLBLAS_STATUS_INVALID_VALUE
5. *alpha == 1.0f               -> ACLBLAS_STATUS_SUCCESS（等价 no-op，提前返回）
6. GetAivCoreCount() == 0       -> ACLBLAS_STATUS_EXECUTION_FAILED
```

注意：alpha=0.0（含 -0.0）**不是** no-op，进入计算路径置零；以上 no-op 均不启动 kernel。

tiling 策略：

##### 1. 分核策略：

- 获取 AIV 核数 C = `GetAivCoreCount()`，C==0 时返回 `EXECUTION_FAILED`；
- **AIV 路径（incx==1）**：F = 2*n，块数为 8 个 float 的块数 `workBlocks = ceil(F/8)`，实际核数 `B = min(workBlocks, C)`；
- **SIMT 路径（incx>1）**：按 complex 元素分配，核数 `U = min(ceil(n / SIMT_MIN_THREAD_NUM), C)`。

AIV 整数分核公式（核号 i = 0..B-1）：

```text
perCoreN        = floor(floor(F/B)/8)*8
leftover        = F - perCoreN*B
extraBlockCores = floor(leftover/8)
tailElements    = leftover % 8
offset(i)       = i*perCoreN + min(i, extraBlockCores)*8
count(i)        = perCoreN
                  + (i < extraBlockCores ? 8 : 0)
                  + (i == B-1 ? tailElements : 0)
```

这些区间拼接覆盖 `[0, F)`，所有核起始偏移均为 8 的倍数；`tailElements`（若存在）落在最后一核。空核在申请 UB 前直接返回。

SIMT 分核（无 per-core 数组，kernel 侧按公式计算）：

```text
baseCount = floor(n/U)
remainder = n % U
calNum(i)       = baseCount + (i < remainder ? 1 : 0)
startOffset(i)  = i*baseCount + min(i, remainder)
```

##### 2. 数据分块和内存优化策略：

AIV 路径按 float 元素分块，UB 预算固定由编译期配置决定：

- 默认构建配置（`build.sh` 不带实验开关）：`queue_slots=1`、`tile_buffer_count=2`、`optimize=ON`；
- `UB_SIZE = 248 * 1024 bytes`，`tileSize = floor(UB_SIZE / (tile_buffer_count * sizeof(float)) / 8) * 8 = 31744 floats`；
- 每 slot 实际容量按 `min(count, tileSize)` 向上对齐到 8 floats 申请；
- 非零 alpha 满 tile 时输入+输出共需 `253952 bytes` UB——这是源码预算，不是实测进程内存；
- 队列为 `TQue<QuePosition::VECIN/VECOUT, 1>`，默认每个方向一个物理 slot（无显式跨 tile prime/steady/drain 调度；本实现不以 DMA 自动重叠作为性能达标的必要条件）；
- alpha==0 时不初始化输入队列（`InitBuffer` 跳过 VECIN），减少 UB 占用且避免无谓读流量。

每 tile 处理时，对齐部分用 `DataCopy`，不足 32B 的尾部用 `DataCopyPad`（GM 尾部只写有效元素，padding 不写回）。

SIMT 路径不涉及 UB 分块，直接 GM 访问；线程数：

```text
nthreads = min(CeilAlign(ceil(n/U), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)
```

线程步长为 `blockDim.x`，kernel 启动时 `blockDim.x` 与 nthreads 必须一致。

##### 3. tilingkey 规划策略：

本算子为固定语义的逐元素缩放：kernel 侧仅需依据 `incx`（区分 AIV/SIMT）与 `alpha`（区分置零/向量乘）分支，两者均写入 tiling 数据；host 侧无需额外编码 tilingkey。

##### 4. tiling 数据结构：

```cpp
struct CsscalTilingData {
    uint32_t totalN;          // AIV: float 总数(2*n)；SIMT: complex 元素数 n
    uint32_t perCoreN;        // AIV: 每核基础 float 数，按 32B 对齐
    uint32_t extraBlockCores; // AIV: 前若干核各多处理一个 32B block
    uint32_t tailElements;    // AIV: 非对齐 float 尾数，交给最后一核
    uint32_t tileSize;        // AIV: 单次搬运的 float 数
    float alpha;              // 实数标量
    int64_t incx;             // 步长（complex 元素单位）
    uint32_t useCoreNum;      // SIMT: 实际使用核数
    uint32_t nthreads;        // SIMT: 每核线程数
};
```

kernel 启动：Host 侧经 `csscal_kernel_do(x, workSpace, tiling, numBlocks, stream)` 启动 AIV/SIMT kernel；`workSpace` 沿用仓内统一签名约定，本算子不使用额外 GM 工作空间，恒传 `nullptr`。kernel 为 AIV_ONLY，沿 handle 绑定的 stream 异步提交，公开算子内不新增同步。

#### 3.2.2 kernel侧设计：

##### AIV 路径（incx==1）

类 `CsscalAIV`，totalN = 2*n：

```text
Init():
  - 解析 tiling；blockIdx_ = GetBlockIdx()
  - 计算 myOffset_/myCount_（见 host 分核公式）
  - myCount_ == 0 时直接返回（空核不申请 UB）
  - xGM_.SetGlobalBuffer(x, totalN)
  - bufferFloats = ceil(min(myCount_, tileSize)/8)*8
  - alpha != 0: InitBuffer(inQueue_, 1, bufferFloats*sizeof(float))
  - InitBuffer(outQueue_, 1, bufferFloats*sizeof(float))

Process():
  - myCount_ == 0 时直接返回
  - tileLoop = myCount_ / tileSize；tileTail = myCount_ % tileSize
  - 前 tileLoop 个整 tile 依次 SingleIteration(offset, tileSize)
  - 若 tileTail > 0：SingleIteration(offset, tileTail)

SingleIteration(curOffset, dataCount):
  - alignedCount = floor(dataCount/8)*8；tailCount = dataCount % 8
  - out = outQueue_.AllocTensor()
  - alpha == 0:
      Duplicate(out, 0.0f, dataCount)          // 不读输入 GM
  - alpha != 0:
      in = inQueue_.AllocTensor()
      alignedCount>0: DataCopy(in, xGM_[curOffset], alignedCount)
      tailCount>0:    DataCopyPad(in[alignedCount], xGM_[curOffset+alignedCount],
                                  tailCount*4B, pad 0.0f)
      inQueue_.EnQue/DeQue；Muls(out, in, alpha, dataCount)
  - outQueue_.EnQue/DeQue
  - alignedCount>0: DataCopy(xGM_[curOffset], writeData, alignedCount)
  - tailCount>0:    DataCopyPad(xGM_[curOffset+alignedCount], writeData[alignedCount],
                                tailCount*4B)  // 只写有效元素
```

`GlobalTensor<float>` 的索引单位是 float 元素，不混用 bytes 或 complex 元素。tensor 复用依赖队列生命周期（EnQue/DeQue/FreeTensor 配对）。

##### SIMT 路径（incx>1）

kernel 按 `baseCount/remainder` 公式计算每核 `calNum/startOffset`，每个 AIV 核内启动 nthreads 个线程：

```cpp
for (uint32_t i = threadIdx.x; i < calNum; i += blockDim.x) {
    uint64_t complexIdx = (uint64_t(startOffset) + i) * uint64_t(incx);
    uint64_t floatIdx   = complexIdx * 2;      // re, im 连续
    if (alpha == 0.0f) {
        xGm[floatIdx]     = 0.0f;              // 实部 +0
        xGm[floatIdx + 1] = 0.0f;              // 虚部 +0
    } else {
        xGm[floatIdx]     = alpha * xGm[floatIdx];
        xGm[floatIdx + 1] = alpha * xGm[floatIdx + 1];
    }
}
```

索引计算全程使用 `uint64_t`，任意正 incx 下不会溢出 32 位。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Ascend 950PR（arch35） | √ |

> 本次真机自测仅确认 950PR。仓内 README 中同时存在的 950DT 支持声明尚未补证，待验证后保留或收窄，不据此推断 950DT 已通过。

## 算子约束限制

| 约束项 | 内容 |
|---|---|
| no-op 语义 | n≤0 或 incx≤0 → 合法 no-op，返回 SUCCESS，不修改 x、不启动 kernel |
| alpha=1.0 | 等价 no-op，Host 侧提前返回 SUCCESS |
| alpha=0.0 | **非 no-op**；被索引元素实部/虚部 bit-exact 写 +0，步长间隙不变 |
| alpha 特殊值 | alpha 可为 FLOAT32 全集（含 ±Inf/NaN）：非零 alpha 直接做浮点乘，行为与 cblas csscal 一致；仅 alpha==0.0 走置零路径 |
| 步长 | 仅支持正步长（incx≤0 按 no-op）；incx>0 时索引用 uint64_t 计算不溢出 |
| 调用方缓冲区 | x 须为实际可访问的 Device 缓冲；文档不宣称所有数学尺寸均可分配 |
| AIV 对齐依赖 | incx==1 路径按 32B 整块访问，要求 x 基址 32B 对齐（aclrtMalloc 分配天然满足）；SIMT 逐元素访问无该要求。此为实现依赖，不构成 API 级约束 |
| alpha 指针内存空间 | Host 内存指针（任务书 §2.4），调用方须保证可读 |
| 广播/dynamic shape | 不涉及；n 为运行时入参 |
| 原地更新 | x 原地更新，不返回视图 |
| 异步执行 | 依赖 handle 绑定 stream；读回前须同步；no-op 不触碰 stream |
| 编译期开关 | queue_slots/tile_buffer_count/optimize 为实验期编译开关，默认配置与 `--csscal-baseline` 对照配置均以实际构建参数为准，详见附录 |

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 实部/虚部分别按 FLOAT32 判定：rtol=2^-10、atol=2^-16、matched_ratio≥0.99、max_abs_error≤max(1e-2, 32*ULP) | 任务书 §3.2 + 生态算子开源精度标准 |
| 性能标准 | incx=1：n=1M ≤ 13.57 us、n=2M ≤ 21.05 us、n=4M ≤ 43.02 us | 任务书 §3.3 |
| 性能对照 | 配套 200 条按 `GPU_ms*1000/avg_us >= 0.4` 判定，使用原 GPU baseline | 任务书 §3.5 配套 |

**精度测试说明**：8503a7c 对应测试共 1014 项精度用例 = 1000 条非性能 CSV 用例 + 14 项 TEST_F，含负零检查、步长间隙 sentinel、1M+7 非对齐尾块（1048583 元素）以及独立 2M/4M 全量 cblas golden（2097152/4194304，alpha=2.5）。当前两个大尺寸用例实部/虚部失败数均为 0、最大绝对误差为 0。普通 golden 容差测试等同于逐元素比较，不等于对所有位模式做 memcmp。

**性能测试说明**：200 条唯一 `[PERF][ascend950]` 记录，每条 samples=100；计时口径为一次 H2D 后 warmup 10 次并同步，再计时 100 次公开 API 调用 + 一次 Device 同步，取总时间/100——不含输入准备与 H2D/D2H，含 Host 提交与同步，不能直接拆出纯 kernel 时间。GTest PASS 仅表示执行成功，性能判定另行按门槛与 GPU ratio 计算。

**内存与流量**：scal 为原地读写，每 complex64 元素读 8B、写 8B，算法流量为 `16*n bytes`（4M 时 67108864 bytes）。任务书 §3.4 要求的内存占用数据以归档的实测记录/工具输出为准；UB 预算 253952 bytes 是源码预算，不是进程实测内存。

## 兼容性分析

- 本实现为 arch35 新增实现，接口声明已存在于 `include/cann_ops_blas.h`，不引入破坏性变更；
- 参数序列与 cuBLAS `cublasCsscal` 一致（handle, n, alpha, x, incx），golden 语义与 Netlib `csscal` 一致；
- 与同族 `sscal` 共享 AIV/SIMT 双路径与 tiling 模式，维护成本低；
- kernel 为 AIV_ONLY，不改变既有 stream 语义，no-op 不触碰 stream。

## 已知交付边界与待办

以下内容为如实披露，不影响本版设计结论，但属验收前须闭环或留档事项：

1. **输入分布覆盖未闭环**：任务书 §3.5 要求测试输入 alpha/x 按均匀分布 50%、正态分布 50%（含特殊值）生成。当前 CSV 生成器中 `RANDOM_NORM` 的 NORM 为结构模式名，`RandomGenerator` 实际使用 `uniform_real_distribution`，不是高斯分布；算子本身不含随机数生成，差异在测试输入侧。需要补测试输入生成代码、输入清单及必要补测，不修改任务书。
2. **性能输入幅度**：当前性能用例使用约 ±[1e-9, 1e-6] 的小幅值输入，以避免 110 次连续乘法放大溢出；alpha 为固定用例值。因此不能证明 §3.5 所述 alpha/x 均匀与正态各 50% 的分布已在性能用例中完整覆盖。
3. **吞吐口径**：热态算法有效吞吐不等于 HBM 实测带宽；超过标称值不能仅归因于流水，更不能由 GPU 与 NPU 均超过标称值推断两侧测量口径相同。相关分析需附证据。
4. **950DT 声明**：支持声明只确认 950PR 本次真机结果；README 中 950DT 行待补证或收窄。
5. **证据归档**：任务书要求精度/性能截图与内存占用数据，统一从归档日志/测试输出展示真实结果，不制造终端截图。
6. **编译期审计**：当前构建全局仍为 Debug；csscal_host.cpp / csscal_kernel.cpp 经源文件级 `-O2` 追加后由 ASC/bisheng 编译，工具链 `compile_commands.json` 不含 ASC 源，优化审计以归档的 verbose 编译命令为准。`--csscal-baseline` 为同源码对照配置（queue_slots=0、tile_buffer_count=4、optimize=OFF），不是旧提交回滚；实验期 CMake FORCE 缓存残留问题在合入审查中处理，不修改本文档对应的已测代码。

---

# 附录：版本与交付引用

- **代码**：`2401_87688128/ops-blas`，分支 `codex/csscal-phase2-experiments` @ `8503a7c1368022b387067ae154dd7e2803c0faf9`；
- **材料**：`materials/phase2-default-8503a7c` @ `8665345f179d76716c73be0c6d79e7cb532f6fc0`；
- **测试说明**：`ops-blas/test/scal/csscal/README.md`；
- **报告与审查**：交付材料/最终自测报告、合入审查清单；
- **社区提交**：设计文档 PR、代码 MR、任务系统交付均须按任务书 §4/§5 流程执行；本文档不宣称上述环节已通过。

# 附录：参考资料

1. ops-blas 开源仓：https://gitcode.com/cann/ops-blas
2. 社区任务仓与设计文档模板：https://gitcode.com/cann/cann-ops-competitions
3. Netlib csscal：https://www.netlib.org/blas/csscal.f
4. cuBLAS cublasCsscal：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-scal
5. 生态算子精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md

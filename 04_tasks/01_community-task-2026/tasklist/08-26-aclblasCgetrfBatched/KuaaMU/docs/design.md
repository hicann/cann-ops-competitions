# 需求背景（required）

## 需求来源

本任务来源于 CANN 社区任务《8月社区任务-aclblasCgetrfBatched算子开发（950）》。任务要求在 ops-blas 开源仓中，基于 Ascend C 实现 Atlas 950PR 上的 `aclblasCgetrfBatched` 批量 LU 分解算子，接口语义对齐 cuBLAS `cublasCgetrfBatched`，算法语义参考 Netlib LAPACK `cgetrf`。验收交付包括设计文档、自测用例及测试代码、自测报告和待验收代码地址。

## 背景介绍

### aclblasCgetrfBatched 算子

`aclblasCgetrfBatched` 对指针数组中的一批 complex64 方阵分别执行带部分主元选取的 LU 分解。对第 `i` 个矩阵，数学关系为：

```text
P * A[i] = L * U
```

其中 `P` 是由 `PivotArray[i*n ... (i+1)*n-1]` 构造的行交换置换矩阵，`L` 是单位下三角矩阵，`U` 是上三角矩阵。`L` 和 `U` 原地写回 `A[i]`，`L` 的单位对角线不写回。该接口面向批量小矩阵、中矩阵和较大方阵分解场景，是 BLAS 生态中 `getrfBatched` 家族的基础算子。

ops-blas 已有 `aclblasSgetrfBatched` 单精度实数实现，其参数校验、quick return、1-based pivot、NO_PIVOT 和奇异 `info` 语义可以作为同族实现参考。complex64 需要额外处理实部/虚部交织存储、复数除法、复数 ICAMAX 主元选择和复数 rank-1/GEMM 更新。因此本任务不是简单把 `float` 替换为 `aclblasComplex`，还需要针对 AIV/SIMT 和可选 Cube 路径重新设计访存和分块。

### 当前实现进展

当前草稿实现已在 `blas/getrf_batched/arch35/` 提供 host、tiling 和 AIV/SIMT kernel 路径：host 侧完成公共参数校验和 quick return；kernel 侧按 batch 分派到 AIV，块内使用 256 个 SIMT 线程；`n <= 64` 且紧凑存储时先整矩阵进 UB 再执行面板分解；`n = 64`、`lda == n`、PIVOT 且 `batchSize >= 512` 的场景另有 FP16/Cube trailing update 试验路径，通过实部/虚部高低位三拷贝构造增强精度。测试目录已提供 CPU golden、CSV 驱动 GTest 和 1200 条功能/精度用例入口。

上述实现是阶段性进展，尚未在本设计文档中声明性能达标。后续优化以本设计中的分块、UB 流水、Cube 路线和性能实测为准；所有验收性能数据必须在 950PR 上按统一 warmup 和采样口径重新测量。

# 需求分析（required）

## 需求描述

在 `include/cann_ops_blas.h` 新增可共用的 `aclblasCgetrfBatched` 接口，在 `blas/getrf_batched/arch35/` 实现 Ascend C kernel，并在 `test/getrf_batched/cgetrf_batched/` 提供 GTest/CSV 精度测试工程。算子对每个 complex64 列主序方阵执行原地 LU 分解，输出 1-based 主元序列和奇异 `info`。禁止定义 950PR 私有平行 API。

## 需求拆解

1. 接口与语义：函数签名、参数顺序和返回状态对齐 cuBLAS `cublasCgetrfBatched`；支持列主序 complex64、`lda`、原地 L/U、1-based pivot、NO_PIVOT 和 `info` 语义。
2. 正确性：复数主元按 ICAMAX 选择 `|real| + |imag|` 最大者；奇异矩阵仍继续分解并保证 `P * A = L * U` 的存储语义；参数非法时返回指定状态码。
3. 边界：`n == 0` 或 `batchSize == 0` 时不启动 kernel 并直接返回成功；`PivotArray == nullptr` 时禁用主元；PIVOT 模式下 `infoArray` 不可为空。
4. 性能：先 warmup，再有效采样大于 50 次取平均；`n=64/batchSize=512 <= 294.43 us`，`n=256/batchSize=64 <= 3398.93 us`，`n=512/batchSize=32 <= 25619 us`。当前实现未声称已满足该目标。
5. 精度：`PivotArray` 和 `infoArray` 与 CPU golden 逐元素 bit-exact；非奇异 batch 的 L/U 实部和虚部按 FLOAT32 生态精度标准判定，奇异 batch 跳过 L/U 数值校验。
6. 工程交付：复用 ops-blas 句柄和 stream 机制，提供 950PR 支持说明、复现步骤、自测报告和回归用例；不修改与该任务无关的公共逻辑。

# 详细设计（required）

## 算子分析

### 支持数据类型

| 数据对象 | 类型 | 说明 |
| --- | --- | --- |
| `Aarray[i]` | `aclblasComplex` / complex64 | 实部和虚部均为 float32，按 8 字节交织存储 |
| `PivotArray` | `int32_t` | Device 侧 1-based 主元行号 |
| `infoArray` | `int32_t` | Device 侧奇异信息 |

矩阵按列主序存储。元素 `(row, col)` 的物理偏移为 `row + col * lda`，实部和虚部连续存放。允许 `lda >= max(1, n)`，因此列间可存在 padding；本任务不要求支持超出 `lda` 语义的非连续布局。

### 接口定义

```cpp
aclblasStatus_t aclblasCgetrfBatched(
    aclblasHandle_t handle,
    int n,
    aclblasComplex* const Aarray[],
    int lda,
    int* PivotArray,
    int* infoArray,
    int batchSize);
```

`Aarray` 是 Device 侧指针数组；每个元素指向一个 Device 侧 complex64 方阵。`PivotArray` 大小为 `n * batchSize`，按矩阵连续排布；`infoArray` 大小为 `batchSize`。三个输出都依赖 `handle->stream` 异步执行，调用方读回结果前必须同步 stream。

### 参数校验与 quick return

校验顺序如下：

| 顺序 | 条件 | 行为 |
| --- | --- | --- |
| 1 | `handle == nullptr` | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` 或 `batchSize < 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `lda < max(1, n)` | 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n=0` 时也要求 `lda >= 1` |
| 4 | `n == 0` 或 `batchSize == 0` | 合法 quick return，返回成功且不启动 kernel、不访问矩阵和输出 |
| 5 | 非 quick return 且 `Aarray == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `PivotArray != nullptr && infoArray == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

NO_PIVOT 模式下 `PivotArray == nullptr`。与同族 S 版实现保持一致，此时允许 `infoArray == nullptr`，kernel 不写奇异信息；若调用方传入非空 `infoArray`，则仍然输出每个矩阵的奇异 `k`。

### pivot、ICAMAX 和奇异 info 语义

PIVOT 模式下，对第 `k` 列在 `[k, n)` 范围内选择主元：

```text
score(row) = |A[row,k].real| + |A[row,k].imag|
选择 score 最大者；score 相等时选择最小 row
```

该口径对应 LAPACK 复数 `icamax`，不是先比实部再比虚部，也不是比较模长平方。`PivotArray[k]` 写入 `pivotRow + 1`。随后交换第 `pivotRow` 行和第 `k` 行的全部 `n` 列，并继续计算。

若最大 score 为精确零，则 `info[i] = k + 1`，表示 `U(k,k) == 0`。该值是合法计算结果，不是接口错误码；host 仍返回 `ACLBLAS_STATUS_SUCCESS`。分解继续处理后续列。为保证存储后的 `P * A = L * U`，零主元列不计算普通乘子，同时将该列 `k` 以下的 L 乘子槽位清零；后续列仍执行行交换和更新。`info` 只记录第一个遇到的零主元列。

NO_PIVOT 模式不选择主元、不交换行、不写 `PivotArray`。若当前对角元为精确零，同样写入第一个 `info=k+1` 并继续；否则执行复数乘子和 trailing update。

## 总体实现方案

### host 侧设计

host 侧流程分为校验、tiling、workspace 规划和 kernel 发射：

1. 从句柄获取 stream，先执行上文参数校验；quick return 分支不获取硬件核数、不分配 workspace。
2. 读取 950PR AIV 核数，根据 `batchSize` 和矩阵规模计算 `batchPerCore`、`usedCoreNum`、`batchTail`。设计目标是优先满核、避免空核，并让同一批多个矩阵复用同一个 tiling 数据。
3. 生成 `CgetrfBatchedTilingData`，包含 `n`、`lda`、分核信息、`threadNum`、`usePivot` 和面板范围。小矩阵整矩阵进 UB；中/大矩阵按列面板分块；超大矩阵可进一步启用 trailing tile。
4. 仅 Cube 增强路径申请 workspace。workspace 挂在 handle 默认 workspace 上，容量按最大面板的 half A/B 缓冲和 float C 缓冲估算，并做乘法溢出检查。分配失败返回对应错误状态，不把 host 异常透传成奇异 `info`。
5. 通过 kernel launcher 将 `Aarray`、`PivotArray`、`infoArray`、tiling 和 block 数发射到 `handle->stream`。host 不做 device 结果同步；异步语义由调用方控制。

当前草稿中的分核为每个 AIV block 处理一个矩阵，简单保证了矩阵内串行依赖。后续 tiling 会引入按核批量分派：`batchPerCore = ceil(batchSize / coreNum)`，尾核处理剩余矩阵。这能减少小矩阵大批量场景的调度轮次；矩阵内部的列依赖仍在一个 block 内串行完成。

### tiling 设计

tiling 按矩阵规模选择路径：

| 条件 | 路径 | 关键策略 |
| --- | --- | --- |
| `n <= 64` 且 `lda == n` | UB 整矩阵 | 将 `2*n*n` 个 float 复制到 UB，pivot、行交换、乘子和 rank-1 update 均在 UB 中完成，最后写回 GM |
| `64 < n` 且未启用 Cube | GM 面板/trailing tile | 按 16/32/64 列面板逐步分解，行交换和面板内更新保持依赖；trailing 更新按 `tileRows x tileCols` 分块并行 |
| `n = 64`、紧凑存储、PIVOT、大批量 | AIV 面板 + Cube trailing | AIV 完成面板主元和乘子，Cube 执行大块复数 rank-K 更新 |
| 非常大 `n` | 面板 + trailing tile | 降低单核 GM 随机访问量，把 update 拆成可重用 tile，优先提升 `n=256/512` 档位 |

tiling key 至少区分 `usePivot`、面板/整矩阵路径和是否启用 Cube。面板宽度和 tile 尺寸作为编译期模板常量或 tiling 数据传入，便于按 UB 容量、SIMT 线程数和 Cube 块大小调优，不把性能参数硬编码到语义逻辑中。

### kernel 侧 AIV/SIMT 设计

AIV block 使用 SIMT VF 模型，每 block 默认 256 线程。一个逻辑矩阵内的列分解必须串行，但每一步内部的行扫描、行交换、乘子计算和 update 可以并行。

#### 主元查找

线程先按 `row = col + threadIdx.x; row += blockDim.x` 扫描本列，计算 `abs(real) + abs(imag)`。线程局部保留最大 score；score 相同取更小行。随后使用 warp shuffle 做 32 lane 内归约，再由 warp leader 写共享 partial，最后归约出全局主元。整矩阵在 UB 中时使用 UB 版本，GM 路径使用 GM 版本；两者共享同一个比较规则。

#### 行交换和复数乘子

主元行确定后，全部线程覆盖列范围并行交换实部/虚部。对角元不为零时计算下三角乘子：

```text
dr = A[k,k].real, di = A[k,k].imag
if |dr| >= |di|:
    r = di / dr
    den = dr + di * r
    l.real = (x.real + x.imag * r) / den
    l.imag = (x.imag - x.real * r) / den
else:
    r = dr / di
    den = di + dr * r
    l.real = (x.real * r + x.imag) / den
    l.imag = (x.imag * r - x.real) / den
```

该公式与 CPU golden 一致，可避免复数除法中常规公式在大虚部/实部比例下的溢出。

#### rank-1 update 与 UB 复用

面板内 update 的数学形式为：

```text
A22 = A22 - L21 * U12
```

SIMT 基础路径按扁平元素并行，每元素读取乘子和主元行值后完成复数乘减。为了减少 GM 重复读取，优化方向是将面板和 trailing 子块复制到 UB：每次加载一个 `L21` 列向量和一组 `U12` 行向量，在 UB 中展开多个元素更新，并使用双缓冲隐藏 GM 到 UB 的搬运。紧凑 `n <= 64` 场景则整矩阵常驻 UB，只写回一次。

当前 GM 基础路径已经能表达串行依赖和复数公式，但中/大 `n` 的性能瓶颈主要来自重复 GM 访问。后续优化会优先把 trailing 更新改成 tile 化 UB 流水，而不是增加矩阵间并行破坏 LU 列依赖。

### 可选 Cube/FP16 分块路线

针对 `n=64/batchSize=512` 等大批量小方阵，当前草稿提供 Cube trailing update 试验路径。复数 rank-K 更新被转换成实数 GEMM：将乘子矩阵的实部/虚部按高低位拆分为三组 half 拷贝，将主元行矩阵构造为复数乘法所需的符号块，Cube 输出 float 实部和虚部 update，再由 AIV epilogue 从原矩阵中减去。三拷贝的目的是用 `high*high + high*low + low*high` 近似恢复 FP32 复数乘积，减少单纯 FP16 截断带来的误差。

该路线的约束如下：

1. 仅作为性能增强路径，语义必须与非 Cube 路径一致，尤其是 pivot 和奇异 `info`。
2. workspace 按每个 batch 的最大 `A/B/C` slot 计算，`halfBatchStride` 和 `floatBatchStride` 做 512B 对齐。
3. Cube 只处理无行交换依赖的 trailing 块；主元查找、行交换和面板乘子仍由 AIV/SIMT 完成。
4. 每个面板完成后依赖 stream 顺序执行 extract、Cube GEMM 和 epilogue；如引入多 stream，需要显式 event 同步，默认不启用。
5. 该路径开启前必须通过同一套精度用例，重点覆盖病态矩阵、普通非奇异矩阵和混合奇异 batch。

若实测显示 FP16/Cube 三拷贝精度或 workspace 成本不可接受，则回退到 AIV UB tile 路线；该取舍以性能和精度实测为准，不预设 FP16 路线一定优于 FP32 AIV 路线。

## 分核与内存设计

### 分核策略

批量矩阵相互独立，是粗粒度并行来源；矩阵内列分解存在串行依赖，只能在行、列和 tile 内并行。分核原则为：

1. `batchSize >= coreNum` 时优先一个 AIV block 处理一批连续矩阵，避免小矩阵批量被切成过多 kernel。
2. `batchSize < coreNum` 时使用 `batchSize` 个 block，每个矩阵独占一个 block；不把同一矩阵拆到多个核做 pivot 分解。
3. trailing update 阶段可以在矩阵内按 row tile 和 col tile 扩展 block，但必须等待对应面板完成。
4. 尾核和尾 tile 显式处理 `n % tileRows`、`n % tileCols`，不写 padding 列以外的用户数据。

### workspace 与内存

AIV/SIMT 基础路径不引入额外 device buffer，输出直接写回用户 `Aarray[i]`、`PivotArray` 和 `infoArray`。UB 消耗包括整矩阵或面板 tile、partial score/row 和共享 info。

Cube 路线的 workspace 分为两部分：

| 缓冲 | 内容 | 生命周期 |
| --- | --- | --- |
| half A/B workspace | 当前面板的乘子实部/虚部高低位拷贝，以及 U12 复数符号块 | 每个面板更新前写入，GEMM 结束后复用 |
| float C workspace | Cube 输出的 trailing 实部/虚部 update | 每个面板 epilogue 消费后复用 |

host 侧按所有面板的最大 slot 计算 `totalBytes = batchSize * (alignedHalfStride + floatStride)`，并检查 `batchSize` 乘法溢出。workspace 只挂接在 handle 默认 workspace 上；接口不返回 workspace 指针，不在 host 侧保存矩阵副本。

## 精度策略

1. 复数除法采用与 CPU golden 相同的 Smith 分支公式，减少中间量溢出。
2. 基础 AIV 路径全程使用 float32 存储和计算；不把输入降为 FP16。
3. FP16/Cube 增强路径使用实部/虚部高低位三拷贝，把主要误差控制在 FLOAT32 生态精度标准内；该路径只用于 trailing update，不用于主元判断。
4. pivot score 只用 `abs(real) + abs(imag)`，主元和整数输出不由 FP16 GEMM 决定。
5. 奇异列的乘子槽位清零，保证返回的已存储 L/U 继续满足 `P * A = L * U`。
6. 对病态矩阵使用部分主元降低数值误差，但不承诺 bit-exact L/U；整数 pivot/info 保持 bit-exact。
7. 特殊值用例按 CPU golden 行为比对；涉及 NaN 的排序和比较语义以 golden 可复现结果为准，不把 NaN 参与的 L/U 数值误差作为通用达标样本。

## 测试矩阵

### 功能与精度测试

当前测试工程通过 CSV 驱动，已包含 1200 条用例入口，覆盖以下类别：

| 用例组 | 覆盖内容 | 判定 |
| --- | --- | --- |
| 基础功能 | `n=1/4/8`、batch 1/2/4、PIVOT 和 NO_PIVOT | 返回值、info、pivot、L/U |
| shape 扫描 | 1、小质数、2 的幂及 ±1、非对齐值，最大到 4096；batch 至 1024 | 返回值、info、pivot、L/U |
| 紧凑与 padding | `lda=n` 和多个 padding 档位 | 列主序和 lda 偏移 |
| 矩阵类型 | 随机非奇异、对角占优、单位阵、Hilbert 病态、全零列奇异、相关行奇异、混合 batch | 按奇异/非奇异分别校验 |
| 边界 | `n=0`、`batchSize=0`、`n=-1`、`batchSize=-1`、非法 `lda` | quick return 或 `INVALID_VALUE` |
| 空指针 | `handle=nullptr`、`Aarray=nullptr`、PIVOT 且 `infoArray=nullptr` | 指定状态码 |
| 奇异行为 | 全零列、相关行、混合 batch；期望返回成功 | info/pivot 精确；奇异 batch 跳过 L/U |
| 特殊值 | 规格允许的 INF/NAN 场景 | 与 CPU golden 行为一致 |

精度判定规则：

1. `infoArray` 逐元素精确一致，包含奇异 batch 的首个 `k`。
2. `PivotArray` 为 1-based int32，逐元素 bit-exact。
3. 非奇异 batch 的 L/U 实部和虚部独立按 FLOAT32 生态标准比对：`rtol=2^-10`、`atol=2^-16`、`required_matched_ratio=0.99`、`max_abs_error_limit=1e-2` 或 `32 * ULP`。
4. 任务书补充口径为 MERE 小于 `2^-13`，MARE 离群倍率 10.0，实部和虚部独立统计。
5. CPU golden 在测试工程内实现，遵循 LAPACK `cgetrf` 和复数 ICAMAX 语义；测试 README 说明该 golden 来源。

### 性能测试

性能测试必须记录 950PR 设备、CANN 版本、CPU/GPU 负载、驱动版本、tiling 路径和样本数。每次先 warmup，再有效采样大于 50 次取 Avg time；计时范围只包含本接口在绑定 stream 上的异步执行区间，同步和结果搬运口径对两侧一致。复数 LU 理论 FLOPS 约为每个矩阵 `(8/3)n^3 + O(n^2)`，batch 总量乘以 `batchSize`。

| case | n | batchSize | pivot_mode | matrix_type | 验收上限 |
| --- | ---: | ---: | --- | --- | ---: |
| 1 | 64 | 512 | PIVOT | DIAGONALLY_DOMINANT | 294.43 us |
| 2 | 256 | 64 | PIVOT | DIAGONALLY_DOMINANT | 3398.93 us |
| 3 | 512 | 32 | PIVOT | DIAGONALLY_DOMINANT | 25619 us |

性能报告不得用少量样本、未同步 stream、不同 matrix type 或不同 FLOPS 口径替换上述标准。若某一路径未达标，报告中保留原始均值、样本数和瓶颈分析，不将未达标数据表述为通过。

## 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能标准 | 接口签名、quick return、参数校验、NO_PIVOT、1-based pivot、奇异 info、`P*A=L*U` 存储语义与 cuBLAS/LAPACK 口径一致 | 任务书 §2 |
| 精度标准 | info/pivot 精确；非奇异 L/U 实部/虚部按 FLOAT32 生态精度标准；MERE `< 2^-13`，MARE 倍率 10.0 | 任务书 §3.2 |
| 性能标准 | warmup 后有效采样 `>50` 次，三个指定 case 的 Avg time 不高于上表标杆 | 任务书 §3.3 |
| 可复现性 | golden 生成规则、设备信息、随机种子、命令、样本数和原始耗时可追溯 | 任务书 §3.5 |

## 兼容性分析

1. 新增 `aclblasCgetrfBatched` 到共用头文件 `include/cann_ops_blas.h`，不新增 950PR 私有接口；既有 `aclblasSgetrfBatched` 签名和行为不变。
2. 仅将 `aclblasSgetrfBatched` 的 `float` 替换为 `aclblasComplex`，参数顺序与 cuBLAS 对齐，便于生态迁移。
3. 适配目标为 Atlas 950PR，CANN 交付验收按任务书使用 9.1.0；当前开发与验证环境为 CANN 9.1.0-beta.3，最终性能与精度数据须在该版本链路上复核。AIV/SIMT 代码按 arch35 目录组织，不影响其他架构目录。
4. 所有输入输出指针均为 Device 内存；`Aarray[i]` 之间不得重叠。本接口不管理用户内存，不改变 ACL stream 生命周期。
5. 输出为原地覆写，没有视图、广播或 dynamic shape 语义；调用方读结果前同步 stream。
6. NO_PIVOT 下 `infoArray` 是否为空的行为与仓内 S 版实现和 README 保持一致；PIVOT 下必须提供 `infoArray`。

## 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| GM 随机访问过多 | `n=256/512` 档位耗时增加 | 引入列面板和 trailing tile，把 L/U 子块放入 UB，使用双缓冲和批量写回 |
| 批量调度轮次多 | 小矩阵大批量受启动和排队影响 | 按核聚合矩阵，减少 block 数；保留整矩阵 UB 路径 |
| Cube/FP16 精度不足 | L/U 分量误差超标准 | 保持主元判断 FP32；使用高低位三拷贝；按用例回退 AIV FP32 路径 |
| 零主元后续更新错误 | `P*A=L*U` 或 info 不正确 | 零主元列清零乘子，继续行交换和后续更新，增加奇异/混合 batch 回归 |
| 复数除法溢出 | 病态或特殊值用例失败 | 使用 Smith 分支除法；golden 和 kernel 使用同一公式；INF/NAN 用例单独比对 |
| workspace 分配失败 | Cube 路径不可用 | host 做容量和溢出检查；失败返回错误；可回退基础路径 |

## 交付计划

1. 语义与 golden：固化接口、参数校验、ICAMAX、NO_PIVOT、奇异 info 和 quick return 用例，保证 CPU golden 可复现。
2. AIV/SIMT 基础路径：完成全 shape 功能与精度回归，覆盖 1200 条 CSV 用例及负向、边界、特殊值用例。
3. UB tile 优化：实现面板/trailing UB 复用和双缓冲，记录每轮精度与性能测量。
4. Cube/FP16 路线：完成 workspace、三拷贝 GEMM、epilogue 和回退开关，只在精度回归通过后开启性能评估。
5. 性能收敛：按 warmup + `>50` 采样执行三个验收 case，分析 AIV/Cube 占比并调优面板宽度、tile 尺寸和分核策略。
6. 交付评审：更新算子 README、测试 README 和自测报告，提交设计文档、代码分支和验收材料；验收通过后在 ops-blas 提交合入 PR。

# aclblasCgetrfBatched 算子设计文档

> 适配产品：Ascend 950PR（arch35）
> 验收版本：CANN 9.1.0
> 数据类型：COMPLEX64（`aclblasComplex`）
> GitCode 账号：`zhouhongzhe`

# 需求背景（required）

## 需求来源

本需求来自“8 月社区任务——`aclblasCgetrfBatched` 算子开发（950）”。目标是在
`ops-blas` 仓中新增 Ascend C 直调实现，功能与 cuBLAS `cublasCgetrfBatched` 核心语义对齐，
数值语义参考 Netlib LAPACK `cgetrf`。

## 背景介绍

LU 分解是线性方程求解、矩阵求逆和行列式计算的基础。Batched 接口通过一次调用处理多个
独立小矩阵，减少 Host 启动开销。对每个列主序复数方阵执行部分主元 LU：

```text
P * Aarray[b] = L * U,  b = 0, ..., batchSize - 1
```

`L` 和 `U` 原地写回 `Aarray[b]`，`L` 的单位对角线不写回；`PivotArray == nullptr` 时执行无主元 LU。

### 现状分析

`ops-blas` 已有 `aclblasSgetrfBatched` 的公共声明、arch22/arch35 实现及测试框架，但无
complex64 公共 API 和 arch35 实现。C 版可复用句柄、stream、指针数组和 CSV GTest 工程结构；
不能复用实数主元度量和标量运算，必须新增 ICAMAX（`|re|+|im|`）、稳定复除法和复数
rank-1 update。新 API 在 `include/cann_ops_blas.h` 中与其他产品线共享，不定义 950PR 私有接口。

# 需求分析（required）

## 需求描述

使用 Ascend C Kernel 直调方式实现 complex64 批量 LU 分解，支持部分主元/无主元、泛化
`n`/`batchSize`、`lda` padding、奇异矩阵合法返回和零维 quick return，并满足任务书的精度与
三项硬性能门槛。

## 公共接口与参数契约

```cpp
aclblasStatus_t aclblasCgetrfBatched(
    aclblasHandle_t handle, int n, aclblasComplex* const Aarray[], int lda,
    int* PivotArray, int* infoArray, int batchSize);
```

| 参数 | I/O 与存储 | 数据类型/布局 | shape 与值域 | 异常或特殊语义 |
| --- | --- | --- | --- | --- |
| `handle` | Host 输入 | `aclblasHandle_t` | 已创建的有效句柄，携带 stream | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n` | Host 输入 | INT32 标量 | `n >= 0` | `<0` 无效；`0` 为 quick return |
| `Aarray` | Device 输入/输出 | 64 位 Device 指针数组；矩阵为 COMPLEX64、ND 列主序 | `batchSize` 个 `lda x n` 物理矩阵，逻辑区域 `n x n` | 非 quick return 时数组及每项必须是有效 Device 地址；矩阵不得重叠 |
| `lda` | Host 输入 | INT32 标量 | `lda >= max(1,n)` | 不满足时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `PivotArray` | Device 输出 | INT32，batch-major 线性布局 | `[batchSize,n]`，元素为 `[1,n]` | 可为空；空表示 NO_PIVOT |
| `infoArray` | Device 输出 | INT32 | `[batchSize]`，`{0} ∪ [1,n]` | PIVOT 模式不得为空；`>0` 是奇异计算结果，不是 API 错误 |
| `batchSize` | Host 输入 | INT32 标量 | `batchSize >= 0` | `<0` 无效；`0` 为 quick return |

对第 `b` 个矩阵，逻辑元素 `A(i,j)` 的复数偏移为 `i + j*lda`，内存中为相邻的
`{float real, float imag}`。主元偏移为 `b*n+k`，info 偏移为 `b`。置换约定为
`P=P1*P2*...*Pn`，每个 `PivotArray[b*n+k]` 是 1-based 行号。

## 返回值与校验顺序

| 条件 | 返回值 | Kernel |
| --- | --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 不启动 |
| `n < 0` / `batchSize < 0` / `lda < max(1,n)` | `ACLBLAS_STATUS_INVALID_VALUE` | 不启动 |
| `n == 0` 或 `batchSize == 0` | `ACLBLAS_STATUS_SUCCESS` | 不访问 Device 指针，不启动 |
| 非 quick return 且 `Aarray == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 不启动 |
| `PivotArray != nullptr && infoArray == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 不启动 |
| 参数合法（包含 `infoArray[b] > 0`） | `ACLBLAS_STATUS_SUCCESS` | 异步提交到 handle stream |

固定校验顺序为 handle → `n` → `batchSize` → `lda` → quick return → Device 顶层指针。
Device 指针数组中的每个地址由调用者保证有效，Host 不解引用该数组。异步执行错误在 stream 同步
处观察。

## 需求拆解与适用性裁决

1. 新增共享 C API、arch35 Host/Tiling/Kernel、公共 API 列表和算子 README。
2. 支持 PIVOT/NO_PIVOT、`lda` padding、奇异继续分解和异步 stream 语义。
3. 测试须包含 CPU golden、正向/边界/负向/特殊值/精度/性能用例与可复现 README。
4. 设计文档通过评审后，才将代码作为验收交付件。

| 任务书 §3.5.4 条目 | 适用性 | 处理方式 |
| --- | --- | --- |
| `alpha=(0,0)` no-op | 不适用 | 本接口无 alpha，不增加非标准参数 |
| 宽/窄非方阵 | 不适用 | `getrfBatched` 契约固定为 `n x n` 方阵 |
| stride/零步长 | 不适用 | 该 API 是指针数组而非 strided-batched；仅验证合法 `lda` padding |
| Inf/NaN | 适用 | 作为特殊值健壮性用例，验证合法返回、无越界/异常退出和输出分类 |

# 详细设计（required）

## 算子分析

对列 `k=0...n-1`，PIVOT 模式选择：

```text
p = argmax(i=k...n-1) (abs(real(A(i,k))) + abs(imag(A(i,k))))
```

并交换行 `k,p`；度量相等时选择最小行号。NO_PIVOT 模式固定 `p=k`。若当前主元为
精确复数零，仅在 info 尚为 0 时记录 `k+1`，跳过本列除法/update，但继续后续列及 pivot
写回。非零主元执行：

```text
A(i,k) = A(i,k) / A(k,k),                       i = k+1...n-1
A(i,j) = A(i,j) - A(i,k) * A(k,j),              i,j = k+1...n-1
```

结果上三角为 `U`，严格下三角为 `L` 的乘子，`L` 对角线隐含为 1。

## 总体流程

```text
Host 校验 → quick return?
                 └─否→ 获取 AIV 核数 → 构造值传递 Tiling → 单次异步 Kernel launch
                                                                  └→ block 按 batch 分工
                                                                      └→ 动态 SIMT 线程组处理一个矩阵
```

矩阵之间独立，按 batch 分核；一个矩阵始终由一个 block 完成，无跨核同步。列间存在 LU
依赖，串行推进；列内主元搜索、换行、乘子和 trailing update 由 SIMT 线程并行。

## Host 与 Tiling 设计

Host 只读标量和句柄，不拷贝矩阵或指针数组。当平台报告 AIV 核数为 0 时返回
`ACLBLAS_STATUS_INTERNAL_ERROR`。设 `B=batchSize`、`C=GetAivCoreCount()`：

```text
batchPerCore = ceil(B / C)
usedCoreNum  = ceil(B / batchPerCore)
batchTail    = B - (usedCoreNum - 1) * batchPerCore
```

block `t` 从 `t*batchPerCore` 开始，末 block 处理 `batchTail`，其他 block 处理 `batchPerCore`。

| Tiling 字段 | 类型 | 含义 |
| --- | --- | --- |
| `n` | `uint32_t` | 逻辑方阵边长 |
| `lda` | `uint32_t` | 列主序物理前导维 |
| `usedCoreNum` | `uint32_t` | 实际启动 block 数 |
| `batchPerCore` | `uint32_t` | 非末 block 处理的矩阵数 |
| `batchTail` | `uint32_t` | 末 block 处理的矩阵数 |
| `usePivot` | `uint32_t` | 1 为 PIVOT，0 为 NO_PIVOT |

PIVOT/NO_PIVOT 使用模板 Kernel 分支，避免主循环中重复判断。Tiling 按值传递，无 GM workspace。

## Kernel 设计

每个 block 按 `n` 启动 32/64/128/256 个 SIMT 线程，对其负责的每个 batch 执行：

1. 读取 `Aarray[b]` 的 64 位 Device 地址，由线程 0 将共享 info 置 0。
2. 对列 `k`，每线程以 `blockDim.x` 为步长扫描候选行，在 UB 中做树形 ICAMAX 归约。
3. 线程 0 写 1-based pivot 并记录首个零主元，所有线程协作交换两行的 `n` 个元素。
4. 若主元非零，行并行计算乘子。trailing update 按 64 列分片，先协作将
   `U(k,j:j+63)` 读入 UB，再由各线程更新不同 trailing rows，避免每个行线程重复读取同一
   pivot row，且不存在写冲突。
5. 归约、pivot/info 发布、换行、除法和 update 后都使用 block barrier，然后进入下一列。
6. 最后由线程 0 写 `infoArray[b]`（指针非空时）。

对有限值，ICAMAX 度量相等时固定选最小行号，以满足 pivot bit-exact 验收。Inf/NaN 的主元
次序不作为新增 API 契约：特殊值用例只验证合法返回、无越界/异常退出、输出分类与参考语义一致，
不将非有限 L/U 或其 pivot 顺序纳入普通混合容差与 bit-exact 统计。

### 并发正确性不变式

- 每列开始时，`0...k-1` 列已完成分解；进入下一列前，本列换行、乘子和 trailing update 已全部可见。
- 行交换覆盖全部 `n` 列，不只交换 trailing 区域，保证已写回的 `L` 乘子与置换语义一致。
- trailing update 中每个逻辑元素仅由一个线程写回；UB pivot-row tile 在 barrier 后只读。
- info 只允许从 0 转为首个 `k+1`，后续零主元不覆盖它。

### 复数运算

复数以两个 float 显式 load/store。设 `x=xr+i*xi`、`d=dr+i*di`，复除法采用 Smith 分支，
避免直接计算 `dr*dr+di*di` 的溢出/下溢：

```text
if abs(dr) >= abs(di):
    r = di/dr; den = dr + di*r
    q = (xr + xi*r)/den + i*(xi - xr*r)/den
else:
    r = dr/di; den = di + dr*r
    q = (xr*r + xi)/den + i*(xi*r - xr)/den
```

rank-1 update 将 `a -= x*y` 展开为：

```text
a.real -= x.real*y.real - x.imag*y.imag
a.imag -= x.real*y.imag + x.imag*y.real
```

## 资源与内存设计

每 block 的 UB 对象为 256 个 float score（1024 B）、256 个 INT32 行号（1024 B）、一个
INT32 info（4 B）和 64 个 complex64 pivot-row tile（512 B），合计约 2564 B（不含编译器对齐）。
即使保留双缓冲，该项也不超过 3076 B。不申请 GM workspace。输入/输出由调用者持有：

```text
Aarray slots = batchSize * 8 B
matrices     = batchSize * lda * n * 8 B
pivots       = batchSize * n * 4 B       (PIVOT only)
info         = batchSize * 4 B            (when provided)
```

| 用例 | 紧凑矩阵数据量 |
| --- | ---: |
| `n=64,batch=512` | 16 MiB |
| `n=256,batch=64` | 32 MiB |
| `n=512,batch=32` | 64 MiB |
| `n=1024,batch=16` | 128 MiB |

四项均低于任务书 4 GiB 参考预算。测试还要记录输入副本、golden 和 runtime 实际峰值，防止
测试框架本身超预算。

## 性能设计

一次 API 只启动一个 Kernel；batch 维使用可用 AIV；指针数组直接解引用，不做 Host-Device
中间拷贝。线程数从 `n` 确定，并保持 2 的幂以保证树形归约正确：

```text
threadNum = min(256, max(32, nextPowerOfTwo(n)))
updateTileColumns = 64
```

| 尺寸档 | SIMT 线程 | 主要优化目标 |
| --- | ---: | --- |
| `1 <= n <= 32` | 32 | 降低空闲线程和 barrier 开销 |
| `33 <= n <= 64` | 64 | 面向 `64x512` 启动敏感档，减少空闲 lane 和树形归约层数 |
| `65 <= n <= 128` | 128 | 平衡归约深度和行并行 |
| `n >= 129` | 256 | 面向 `256x64`/`512x32`，满足 trailing-row 并行度 |

64 列 pivot-row tile 只从 GM 协作读取一次，在同一片内被所有 trailing rows 复用。PIVOT/NO_PIVOT
使用模板分支，不在每列的内循环中重复判断。

复数 LU 理论计数约为每矩阵 `(8/3)*n^3 + O(n^2)` FLOP。按三项硬门槛反推的最低平均吞吐如下，
用于 profiler 复核：

| n | batchSize | pivot/matrix | 理论总计算量 | Avg time 硬上限 | 最低平均吞吐 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 64 | 512 | PIVOT / DIAGONALLY_DOMINANT | 0.358 GFLOP | 47.11 us | 7.60 TFLOP/s |
| 256 | 64 | PIVOT / DIAGONALLY_DOMINANT | 2.863 GFLOP | 543.83 us | 5.26 TFLOP/s |
| 512 | 32 | PIVOT / DIAGONALLY_DOMINANT | 11.453 GFLOP | 4099.04 us | 2.79 TFLOP/s |

`n=1024,batchSize=16` 仅是扩展性能/内存用例，不替代上述三项硬门槛。profiler 分别统计主元归约、
行交换、复除法和 rank-1 update；任一硬 case 超限都必须在代码中优化并重测，不得通过减少 batch、
改变矩阵类型、放宽精度或只报告最小耗时规避门槛。

## 支持硬件与约束

| 芯片 | 支持 |
| --- | --- |
| Ascend 950PR（arch35） | √ |

- 只支持 COMPLEX64 列主序方阵，不支持超出 `lda` 表达能力的非连续视图。
- 无 broadcast；不要求确定性计算；`n`/`batchSize` 为运行时标量，无额外 dynamic shape 推导。
- `Aarray[b]` 必须可写且相互不重叠，重叠时行为未定义。
- 普通精度契约面向有限 FLOAT32 输入；Inf/NaN 作特殊值健壮性检查。

# 可维可测分析（required）

## 测试架构与可诊断性

在 `test/getrf_batched/cgetrf_batched/` 中建立 CSV 驱动 GTest。CPU golden 在测试工程内独立实现
LAPACK cgetrf 语义，不依赖不存在的 CBLAS LU 接口。每条失败日志至少输出 case 名、`n/lda/batch`、
pivot 模式、矩阵类型、batch/element 下标、actual/golden、误差和阈值；性能日志以 `[PERF]`
前缀输出采样数、平均值、硬门槛和 PASS/FAIL，便于脚本解析与专家复核。

## 用例覆盖

CSV 使用固定随机种子，基线规模为 1200 条：1000 条功能/精度/负向用例和 200 条性能用例。

| 类别 | 数量 | 覆盖点 |
| --- | ---: | --- |
| L0 基础 | 8 | `n=1/4/8`、batch 1/2、PIVOT/NO_PIVOT |
| L1 尺寸 | 27 | 1、质数、2 的幂及 ±1、非对齐，最大到 2048 |
| L2 batch | 12 | 1 至 512 的小/中/大 batch 扫描；1024 由 PF 覆盖 |
| L3 `lda` | 3 | `lda=n+1/n+4/n+8`，padding 使用 sentinel 验证未改写 |
| L4 矩阵类型 | 7 | 随机非奇异、对角占优、单位阵、Hilbert、零列、相关行、混合奇异 batch |
| L5 pivot | 6 | 两种 pivot 模式与中尺寸 |
| L6 边界/负向 | 9+ | quick return、空 handle/Aarray/info、负 `n/batch`、非法 `lda` |
| EX 扩展 | 补足至 1000 | 尺寸×batch×矩阵类型×padding×pivot 确定性采样 |
| PF 性能 | 200 | 3 项硬门槛、`1024x16` 扩展及规模扫描 |

随机矩阵数据严格按任务书生成：均匀分布 `[-5,5]` 占 50%，正态分布
`mu in [-5,5], sigma in [0.1,2]` 占 50%，实部/虚部独立采样。另增 Inf/NaN 特殊值类；它们不计入
上述普通分布比例与有限值数值精度统计。

## 精度标准

1. 对有限值输入，`infoArray` 逐元素精确一致；PIVOT 模式的 1-based `PivotArray` 逐元素 bit-exact。
2. `infoArray[b] > 0` 的奇异 batch 只校验 info/pivot，按任务书跳过 L/U 数值比较。
3. 非奇异 batch 的实部、虚部分开统计。逐元素匹配条件为
   `abs(actual-golden) <= 2^-16 + 2^-10*abs(golden)`。
4. 每个分量同时满足 `matched_ratio >= 0.99` 且
   `max_abs_error <= max(1e-2, 32*ULP(golden))` 才通过；`ULP` 按对应 golden FLOAT32 分量计算。
5. 补充统计必须同时报告 `MERE < 2^-13`；MARE 离群阈值倍率为 10.0，不用其替换第 3、4 项硬判定。
6. 非奇异 batch 另验证 `P*A` 与拆出的 `L*U` 残差、输入矩阵之间无串扰、`lda` padding
   sentinel 未改写。奇异 batch 可记录残差作诊断信息，但不将它作为任务书 L/U 数值验收条件。

## 性能与内存测量标准

- 固定使用 Ascend 950PR 0 号卡；只在 0 号卡无法使用且其他卡确认空闲时切换，报告必须记录卡号。
- 对每个 case 使用 5 次 warmup 和 51 次有效采样（满足 `>50`）。每次测量前在计时区间外恢复输入；
  计时从 API 提交至该 stream 完成，不包含 CPU golden、正确性比对和数据生成。
- 报告平均/最小/P50/P95/最大耗时；三项硬门槛按平均值逐项判定，任一超限则性能验收失败。
- 内存报告包含理论 GM/UB 用量、runtime 峰值和 profiler 截图；不得仅报告理论矩阵字节数。

## 验收证据和门禁

| 门禁 | 必须提供的证据 | 通过条件 |
| --- | --- | --- |
| G0 设计评审 | 本文档、评审意见闭环记录 | 任务书条款无遗漏，设计与代码无矛盾 |
| G1 静态/编译 | 格式、pre-commit、Ascend 950/CANN 9.1.0 构建日志 | 全部成功，公共头可链接 |
| G2 功能/精度 | 全量 case 日志，实/虚部、pivot/info 结果及截图 | 负向状态正确，所有精度阈值通过 |
| G3 性能 | 51 次原始采样、统计、`[PERF]` 日志和 msprof 证据 | 三项 Avg time 均不超上限 |
| G4 内存/稳定性 | 内存峰值、重复运行和特殊值日志 | 无泄漏、越界、异常退出 |
| G5 交付完整性 | 仓库/分支/PR、README、自测报告、复现命令 | 验收人可在指定环境独立复现 |

G0 是代码验收的前置门禁。G1–G5 不得用“已编译”替代“已在空闲 Ascend 950PR 上完成实测”。

# 兼容性分析

- 新增 `aclblasCgetrfBatched` 与已有 `aclblasSgetrfBatched` 并存，不修改已有符号、参数或 ABI。
- 新实现仅编译到 arch35；README 产品支持表标记 Ascend 950PR，其他产品不因此获得未验证支持。
- 句柄/stream、状态码和公共复数类型均复用 `ops-blas` 现有定义，避免私有类型导致二进制不兼容。
- 回滚单元是新 API 声明、arch35 C 版文件、C 版测试和文档；不需要回退 S 版实现。

# 风险与应对

| 风险 | 影响 | 预防/定位方法 |
| --- | --- | --- |
| 列间串行依赖 | 中大 `n` 性能不达标 | batch 分核、列内并行；用 msprof 拆分归约/换行/update |
| 复除法溢出/下溢 | 精度失败 | Smith 分支；覆盖大小模混合及病态矩阵 |
| pivot tie/非有限值语义偏差 | pivot 或特殊值用例失败 | 有限值使用最小行号 tie-break；Inf/NaN 单独做健壮性和输出分类检查 |
| 奇异后提前停止 | info/pivot 或 `P*A=L*U` 错误 | 只跳过当前零主元列，继续后续列和 pivot |
| `lda` 索引错误 | padding 越界或串扰 | 统一 `i+j*lda`；padding sentinel 和多 batch guard |
| 繁忙设备干扰 | 性能数据无效 | 测试前记录 `npu-smi`；仅在空闲卡测试，保留原始采样 |

# 需求追踪与交付映射

## 工程文件影响

| 仓库相对路径 | 变更目的 |
| --- | --- |
| `include/cann_ops_blas.h` | 新增共享 `aclblasCgetrfBatched` 声明 |
| `blas/getrf_batched/arch35/cgetrf_batched_host.cpp` | 参数校验、quick return、分核与异步 launch |
| `blas/getrf_batched/arch35/cgetrf_batched_tiling_data.h` | 定义 Host/Kernel 值传递参数 |
| `blas/getrf_batched/arch35/cgetrf_batched_kernel.{h,cpp}` | complex64 LU、ICAMAX、PIVOT/NO_PIVOT 和性能优化 |
| `blas/getrf_batched/README.md` | 公共语义、参数、约束和 Ascend 950PR 支持表 |
| `docs/zh/api_list.md` | 新增公共 API 列表项 |
| `test/getrf_batched/cgetrf_batched/` | CPU golden、CSV 参数、arch35 GTest 和复现 README |

tasklist 设计文档 PR 仅包含本文档；上表代码与测试在设计评审通过后提交到 `ops-blas` PR，
两个仓库不混合提交。

| 任务书条款 | 设计落点 | 代码/文档落点 | 验收证据 |
| --- | --- | --- | --- |
| §2.1.1、§2.3 复数原地 LU/共享 API | 接口契约、算子分析 | `include/cann_ops_blas.h`、Host、Kernel | 公共头编译链接、L/U golden |
| §2.1.2 1-based pivot | 布局契约、Kernel 步骤 3 | pivot 写回 | 逐元素 bit-exact |
| §2.1.3 奇异合法结果 | 算子分析、奇异风险 | 首零 info 与继续循环 | 零列/相关行/混合 batch |
| §2.1.4 NO_PIVOT | 需求描述、Tiling | `usePivot=0` 模板路径 | NO_PIVOT 用例 |
| §2.1.5 ICAMAX | 算子分析、Kernel 步骤 2 | `abs(re)+abs(im)` 归约 | pivot bit-exact/tie case |
| §2.1.6 quick return | 校验顺序 | Host 直接返回 | 零维/零 batch，profiling 无 Kernel |
| §2.4、§2.5 参数/约束 | 参数表、校验表、约束 | Host validation | 负向状态码、padding/stream |
| §3.1 环境 | 文档顶部、硬件表 | arch35 目录、README | Ascend 950PR + CANN 9.1.0 日志 |
| §3.2 精度 | 精度标准 | CPU golden/GTest | 实/虚部及 pivot/info 原始结果 |
| §3.3 性能 | 性能设计与测量 | 性能 case/验证脚本 | 5 warmup + 51 samples，3 项硬门槛 |
| §3.5 自验 | 用例覆盖、适用性裁决 | CSV、GTest、测试 README | 1200 cases、Inf/NaN、可复现命令 |
| §4 交付件 | 验收门禁 | 设计 PR、测试、自测报告、代码 PR | 链接/分支/目录、截图、内存和原始日志 |

代码 PR 目标目录为 `blas/getrf_batched/arch35/` 和 `test/getrf_batched/cgetrf_batched/arch35/`；算子
README 必须标记 Ascend 950PR 支持。自测报告必须包含用例参数、实/虚部精度、pivot/info、性能、
内存占用的数据和截图，以及个人仓链接、分支和 Ascend-CANN 开发者权限说明。

# 修订记录

| 日期 | 版本 | 说明 |
| --- | --- | --- |
| 2026-08-26 | 1.0 | 首版：完成接口、算法、Tiling、精度、性能和基础追踪设计 |
| 2026-08-26 | 1.1 | 专家验收复审：补齐参数契约、校验顺序、数值/资源设计、任务适用性、测试证据和门禁 |
| 2026-08-27 | 1.2 | 验收否决项复审：量化性能预算，明确 SIMT/UB 复用策略、非有限值边界和精度适用范围 |

# aclblasCgbmv 算子设计文档（Atlas A2/A3）

> 状态：设计稿；尚未实现、编译或在 NPU 上测试。整理日期：2026-09-28。本文的算法和性能策略为拟实施方案，所有性能数字均为任务目标，不是实测结果。
>
> 依据：本任务随包 `aclblasCgbmv_A2A3_task_doc.md`（SHA-256 `1B09FA2907C3C33C0836438BA9BA0AF2382BB1C7F0A259ECBF114E402D7B967D`）；`cann/ops-blas` 源码基线 `fe54d86f00a4d449f55da1e8d144b898953d97c5`；[2026 社区任务设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)（读取时 blob `b2d9e2c47e2afe307a9f303735fb045011642b50`）。

# 需求背景（required）

## 需求来源

9 月社区任务要求在 Atlas A2/A3 的 `arch22` 目录新增单精度复数一般带状矩阵向量乘 `aclblasCgbmv`，对齐 [cuBLAS Cgbmv 接口](https://docs.nvidia.com/cuda/archive/12.0.0/cublas/index.html)，使用 Ascend C 内核直调和 ops-blas 句柄、stream 体系。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)；开发目标为用户指定的 [PzeYui/ops-blas](https://gitcode.com/PzeYui/ops-blas)。本文只准备设计，不含代码 PR 或验收结论。

## 背景介绍

在上述源码基线中，[`include/cann_ops_blas.h`](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/include/cann_ops_blas.h) 已声明 `aclblasSgbmv`，尚无 `aclblasCgbmv`。[`blas/gbmv/arch35/`](https://gitcode.com/cann/ops-blas/tree/fe54d86f00a4d449f55da1e8d144b898953d97c5/blas/gbmv/arch35/) 有实数 `Sgbmv` 的 Host、Tiling、Kernel 实现，提供带状索引及参数校验参考，但其 SIMT API 属于 arch35，不能直接作为 A2/A3 的内核。[`blas/gemv/arch22/`](https://gitcode.com/cann/ops-blas/tree/fe54d86f00a4d449f55da1e8d144b898953d97c5/blas/gemv/arch22/) 有复数 GEMV，可参考复数表示及 Ascend C 向量操作；该实现 Host 端有同步和临时分配，不能直接照搬到本接口要求的异步路径。[`test/gbmv/sgbmv/`](https://gitcode.com/cann/ops-blas/tree/fe54d86f00a4d449f55da1e8d144b898953d97c5/test/gbmv/sgbmv/) 提供 GBMV 的 CSV/GTest 结构；目标 `cgbmv/arch22` 测试尚需新建。

### 目标能力与输入输出

| 对象 | 类型和位置 | 形状及约束 | 作用 |
| --- | --- | --- | --- |
| `handle`, `trans`, `m`, `n`, `kl`, `ku`, `lda`, `incx`, `incy` | Host 属性 | `trans∈{N,T,C}`；运行时整数 | 选择分支、地址与分核 |
| `alpha`, `beta` | Host 上 `aclblasComplex*` | 复数标量，指针非空 | Kernel launch 前读入 tiling |
| `A` | Device，`complex64` | 列主序带状存储 `lda×n`，`lda≥kl+ku+1` | 只读矩阵 |
| `x` | Device，`complex64` | 逻辑长 `n`（N）或 `m`（T/C），`incx≠0` | 只读向量 |
| `y` | Device，`complex64` | 逻辑长 `m`（N）或 `n`（T/C），`incy≠0` | 原地输出；`beta≠0` 时先读旧值 |

`aclblasComplex` 在 [`cann_ops_blas_common.h`](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/include/cann_ops_blas_common.h) 中是相邻的两个 `float`（`real`, `imag`）。本任务没有广播，不要求任意 Tensor 视图或确定性逐位复现；`y` 不得与 `A`、`x` 重叠。

# 需求分析（required）

## 需求描述

接口签名按任务书 §2.3：

```cpp
aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, int kl, int ku,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

数学语义为 `y ← alpha·op(A)·x + beta·y`。`A` 的逻辑行列数始终为 `m×n`，N 时输出长 `m`，T/C 时输出长 `n`；C 对矩阵元素取共轭后转置，T 只转置。[Netlib CGBMV](https://www.netlib.org/lapack/explore-html/dd/df4/group__gbmv_ga2c8ce3cc7f009b8e8dffcb172dec30a9.html) 与任务书均要求零维快速返回。

## 需求拆解与验证对照

| 编号 | 要求及依据 | 设计落点 | 验证方式 |
| --- | --- | --- | --- |
| R1 | N/T/C、`complex64`、带状列主序；任务书 §2.1–2.4 | Host 分支、地址公式、复数乘加 | CPU `cblas_cgbmv` golden；N/T/C、方阵及矩形 |
| R2 | `lda` padding、正负 `incx/incy`；§2.1、§2.5 | 64 位地址计算与起始偏移 | CSV `TC_LD`、`TC_INC`；补充极端步长边界 |
| R3 | no-op、标量退化、非法参数/空指针；§2.4–2.5 | Host 校验顺序、Kernel 条件读 | `TC_ED` 与补充负向测试 |
| R4 | stream 异步与原地写回；§2.5 | 在 `handle->stream` 下发，单个输出仅一个核负责 | 异步调用、同步后读回；竞态检查 |
| R5 | COMPLEX64 混合容差；§3.2 | FP32 复数累加；与标杆逐点比较 | 1000 条随包精度用例及补齐项 |
| R6 | 910B3 上 5 条绝对耗时门限；§3.3 | 小尺寸低开销、N/T/C 分支、UB 分块 | `msprof op` 记录 `Task Duration(us)`，有效样本 >10 |
| R7 | A2/A3、CANN 9.1.0；§3.1、§7 | `arch22` 实现及文档支持表 | 至少一种款型真机自验；验收设备口径再核实 |
| R8 | 代码、测试、自测报告及设计文档；§4–5 | 接口、Host、Kernel、GTest、README | 交付前逐项核对；本稿不代表已完成 |

## 边界与待核实语义

`m=0` 或 `n=0` 的合法 no-op 与普通场景的 `kl≤m−1`、`ku≤n−1` 条件不能直接同时套用。拟先校验 `handle` 和非负维度，再快速返回；零维时其余参数如何校验，需要用任务指定的负向用例和评审意见固定。任务书把 `alpha/beta` 指针列为非空，也写明只在参与计算时要求 `A/x/y` 非空。拟在 `alpha=0` 时不读取 `A/x`，但 `y` 仍需写出 `beta·y`；这条需补 `alpha=0` 且 `A/x=nullptr` 测试明确。`beta=0` 表示不读旧 `y`，不是允许输出指针 `y=nullptr`。

`kl>m−1`、`ku>n−1` 按任务书应返回 `INVALID_VALUE`；cuBLAS 公共文档主要写非负和 `lda` 条件，这一更严格要求以本任务书为准。参数冲突时采用固定 Host 校验顺序，避免根据测试 case 名或测试输入特殊处理。

# 详细设计（required）

## 算子分析

### 数学公式与带状地址

以下均用 **0 起始**索引。合法的 `A(i,j)` 满足 `0≤i<m`、`0≤j<n`、`j−ku≤i≤j+kl`；其存储地址（复数元素单位）是

```text
band_index(i,j) = j*lda + (ku + i - j)
```

无转置时，每个输出 `i` 累加 `j∈[max(0,i−kl), min(n−1,i+ku)]` 上的 `A(i,j)·x(j)`。T/C 时，每个输出 `j` 累加 `i∈[max(0,j−ku), min(m−1,j+kl)]` 上的 `A(i,j)·x(i)`；C 分支使用 `conj(A(i,j))`。

对于逻辑长度 `L`、非零步长 `inc`，由底层分配首地址计算元素位置：`offset(k)=k·inc`（`inc>0`），`offset(k)=(L−1−k)·|inc|`（`inc<0`）。这与随包 `TC_INC` 以及 [`Sgbmv` 测试](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/test/gbmv/sgbmv/arch35/sgbmv_test.cpp) 的存储约定一致。所有乘法、偏移和 `|inc|` 先升为有符号 64 位，处理 `INT_MIN`，防止 Host/Kernel 32 位溢出。

复乘采用 FP32 分量：`(a_r + ia_i)(x_r + ix_i)=(a_r x_r−a_i x_i)+i(a_r x_i+a_i x_r)`；C 分支在乘法前令 `a_i→−a_i`。累加后再做复数 `alpha·sum` 与 `beta·y_old`。`beta=0` 分支完全不读旧 y，避免 NaN/未初始化值传播；`alpha=0` 分支跳过 A/x 读取。Inf/NaN 的传播需要对齐 cblas golden，不以数值替换或清零规避测试。

### 支持数据类型与形状

仅支持 COMPLEX64；`A` 是一维打包带存储，容量至少 `lda*n` 个复数。`x/y` 的物理容量分别至少 `1+(L−1)|inc|` 个复数（`L>0`）。支持紧凑与 padded `lda`、方阵和矩形、带宽 0 到任务书合法上限。零维按 no-op 处理。没有任意视图、batch 或广播。

## 算子实现

### 实现方案

拟在 [`include/cann_ops_blas.h`](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/include/cann_ops_blas.h) 添加公共接口声明，在 `blas/gbmv/arch22/` 新建 `cgbmv_host.cpp`、`cgbmv_kernel.cpp`、必要的 tiling 头，更新 `blas/gbmv/README.md` 的 A2/A3 支持表和调用示例。测试按仓库 [贡献指南](https://gitcode.com/cann/ops-blas/blob/fe54d86f00a4d449f55da1e8d144b898953d97c5/CONTRIBUTING.md) 放在 `test/gbmv/cgbmv/`，含 `CMakeLists.txt`、参数解析、cblas golden、NPU wrapper、`arch22` GTest/CSV。目标只改本算子必需文件。

#### Host 侧设计

1. 先判空 `handle`，返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；校验 `m,n≥0`。零维按上节约定返回成功，不下发 kernel。非零维依次检查 `trans`、`kl/ku` 的上下界、`lda≥kl+ku+1`、`incx/incy≠0`、`alpha/beta/y` 指针，以及确需参与计算的 `A/x`。带宽和地址容量用 64 位临时量计算，避免 `kl+ku+1`、`lda*n` 及步长绝对值溢出。失败返回任务书指定的 `INVALID_VALUE`；设备资源或 launch 异常按仓库现有状态码处理。
2. 将 `m,n,kl,ku,lda,trans,incx,incy`、`alpha/beta` 分量、输出逻辑长度、分核数、每核输出区间写入 tiling。`alpha/beta` 为 Host 指针，调用时读取值，不将 Host 地址传给 Device。按真实 AIV 核数与输出长度取 `usedCores=min(availableCores, ceil(outputLength/tileOutputs))`；小 shape 降核。tile 大小由可用 UB、所需 A/x/累加临时量和 32B 对齐约束推导，最终值在 910B3 实测后确定。
3. 在 `handle->stream` 下发内核。设计目标为一次下发、无全局 workspace、无 `aclrtSynchronizeStream`；结果由调用方同步后读取。若实现中引入动态临时资源，须重新设计其 stream 生命周期，不得在异步 kernel 尚未结束时释放。

#### Kernel 侧设计

按输出区间分核，每个逻辑 `y` 元素只由一个核写入，不用全局原子累加或跨核规约。N/T/C 由 Host tiling 选择分支；Kernel 以 `CopyIn → Compute → CopyOut` 处理。小尺寸共用单核路径减少下发和循环开销；合法任意步长始终有通用路径。

- **N 分支（按输出行块）**：一个核拥有连续行区间 `[r0,r1)`；UB 中保留该区间的复数累加器。遍历该行块所需列 `j∈[max(0,r0−kl), min(n,r1+ku))`，仅搬入这一列在行块内且位于带内的连续 A 段，并读取对应 `x(j)`。以复数向量乘加更新 UB 累加器；列结束后按 `alpha/beta` 合成并写回属于本核的 y。此组织使每列 A 读取连续，同时避免多核写同一个 y；带宽大时将输出行块再分 tile，控制 UB 用量。
- **T/C 分支（按输出列）**：每个输出列 `j` 的有效 A 带段在 GM 中连续。核一次取一段 A 和对应 x；`incx=1` 可连续搬入，其他步长做受界 gather。以 UB 中 FP32 实虚部分别累加并归约为该列的复数和；C 在乘法前对 A 虚部取负。带宽大于 UB 容量时沿有效行区间分段，在本核累加 partial，最终只写一次 y。列之间独立，可以并行执行，不需跨核同步。
- **退化分支**：`alpha=0` 仅对 y 做 `beta` 缩放；`beta=0` 不加载 y；两者皆零时写数值零。复杂数“零”的判定同时检查实部和虚部。保持 Inf/NaN 传播与参考实现一致，优化分支需由特殊值用例验证。

搬入/搬出使用 arch22 可用的 `DataCopyPad`、按元素读写或等价 Ascend C API 处理非 32B 对齐的首尾及步长；padding 只在 UB 内产生，不读取带外 A 或越过合法存储。相邻核在同一 GM 32B 块中的尾部写回需使用不会覆盖其他核元素的写法，并用内存检测/边界用例验证。具体 Ascend C API 选择以 CANN 9.1.0 编译能力为准。

#### 精度与性能策略

FP32 分量计算，不降到 FP16/BF16；改变规约顺序可能改变末位，按混合容差评估。先实现正确通用路径，再在 910B3 上比较 N 的行块宽度、T/C 的列并行度、UB 搬入次数和有效核数。主要成本为带内 A 读取（约 `O((kl+ku+1)·min(m,n))` 个复乘，矩形按实际有效带段求和）、x 复用和 launch 延迟。性能优化不依赖用例编号，不改变公共判定逻辑。

## 支持硬件

| 产品 | 设计目标 | 验证状态 |
| --- | --- | --- |
| Atlas A2（性能设备 Atlas 800T A2 / 910B3） | `arch22` 支持；CANN 9.1.0 | 未运行 |
| Atlas A3 系列 | 同一 `arch22` 接口与内核能力 | 未运行；实际款型待落实 |

任务书 §3.1 写“自验证只需覆盖一种款型”，§7 又写 Atlas 800I A2 与 800I A3 均需完成验收；设计按两者适配，实际自验/验收设备范围提交前需与任务方确认，性能门限以 **Atlas 800T A2 (910B3)** 为准。

## 算子约束限制

`y` 是原地输出且不得与 `A/x` 重叠；不支持任意非连续 Tensor 视图、广播、batch 或逐位确定性。标量指针在 Host 内存，矩阵和向量在 Device 内存。只允许绑定在 `aclblasSetStream` 的 stream 异步执行；错误参数必须在 Host 阶段返回，不能为了通过负向用例让 Device 越界。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 本任务门限与口径 | 来源 |
| --- | --- | --- |
| 精度 | COMPLEX64，`atol=rtol=2^-13`；通过比例 `≥0.99` 且有限值 `max_abs_error≤max(1e-2,32×ULP(g_low))`；Inf/NaN 按标准的特殊值规则。复数如何拆分/合并计分以最终仓库 Verifier 与任务方判定为准。 | 任务书 §3.2、[生态混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)（读取时 blob `0f8ed375ba70cd426d81620a88553dafd8f7792a`）、测试包 README |
| 性能 | 910B3、COMPLEX64、msprof 的 NPU kernel `Task Duration(us)`，5 次 warmup 后采集。§3.3 写有效采样 10 次，§7 写 **>10 次**；采用较严格的 >10 次，再取平均。 | 任务书 §3.3、§7，测试包 README |
| 内存 | §3.4 无数值阈值，但 §4 仍要求内存自验证报告与日志；记录峰值和 workspace。 | 任务书 §3.4、§4 |

五条绝对性能目标（均为“平均单次耗时不高于”，**尚未测量**）：

| case | trans | m×n | kl/ku | alpha/beta | 上限 us |
| --- | --- | --- | --- | --- | ---: |
| 1 | N | 256×256 | 16/16 | 1/0 | 10.665 |
| 2 | N | 512×512 | 32/32 | 1/1 | 17.328 |
| 3 | T | 1024×1024 | 64/64 | 1/0 | 25.875 |
| 4 | N | 2048×2048 | 128/128 | 1/0 | 30.640 |
| 5 | C | 4096×4096 | 64/256 | 1/0 | 52.487 |

随包 `gpu_baseline.csv` 共 200 行且 `gpu_ms` 均有值，前五行按 `gpu_ms×1000/0.8` 算得到约 10.665、17.3275、25.875、30.640、52.4875 us，与任务书显示值在末位有舍入差异；**五条门限以任务书原表为准**。其余性能行若按 README 的 `NPU_us≤gpu_ms×1000/0.8` 比对，应先核实 GPU 值的实测来源和验收规则，保留原 CSV，不擅自改标杆。

### 自测方案与测试包差异

随包 `cgbmv_test.csv` 实际为 **1200** 行：1000 条非 `TC_PF` 精度/负向用例，200 条 `TC_PF` 性能用例；其中 14 条期望错误状态。固定 CSV SHA-256 为 `12F93420496FD2CC378C9782084E34001845D33E2C880C6B8FD32C2CBC061630`。测试目录现无 CGBMV GTest 工程，因此先按现有 `Sgbmv` 结构接入 `cblas_cgbmv` golden、按复数实虚两分量初始化和比对，再运行随包脚本。

1. 小规模手工核对 N/T/C 的带状地址、矩形和 `lda` padding，覆盖上下角带外存储不被读取；用 NaN 哨兵填充 padding 和带外三角区帮助发现越界读。
2. 跑随包 1000 条精度与负向 case，逐项保存原始 GTest 输出、退出码、实际执行/通过/失败/跳过数量。`verify_accuracy.py` 的汇总解析没有独立校验“实际执行数=1000”，且超时路径可能返回 0 个 case 而退出 0；因此必须核对 GTest 原始 summary，不能仅看脚本的 `ALL PASS`。
3. 自补任务书要求而随包未覆盖的 `kl>m−1`、`ku>n−1`、多种对齐偏移、50% 正态输入以及 `alpha=0` 时 A/x 空指针；补测零维与无效参数同时出现的校验优先级。`gen_csv.py --dist mixed` 当前仍退化为均匀分布，不能把现有 1000 行声称为均匀/正态各半。
4. 在 910B3 分别用五条典型 case 采集 `msprof op` 原始 CSV；单 case 单独运行，5 次 warmup、>10 个有效样本，报告每条样本、均值、波动和 Kernel 名。其他 195 条按验收要求关联 GPU 基线。记录运行时 CANN、驱动、固件、设备型号和并发作业状态。
5. 运行内存采集并留原始日志；记录是否实际使用 workspace。修改算法后先回归相关 case，再复跑完整精度和性能，所有证据关联到同一固定源码提交。

## 兼容性分析

新增 `aclblasCgbmv`，现有 Sgbmv/Cgemv 接口不改；公共头只增一个声明。`arch22` 代码不复用 `arch35` 专用 SIMT 内核，避免构建架构不匹配。对 `blas/gbmv/README.md` 和测试注册做最小增量；不更改现有测试脚本的通过逻辑。

## 待确认项与阶段状态

1. 任务书给出的设计仓链接是 `cann/cann-competitions`，当前 2026 社区任务 README/模板位于 `cann/cann-ops-competitions`。截至 2026-09-28，官方任务列表尚无 Cgbmv A2/A3 的编号目录，已有的 Cgbmv 目录对应 950 任务。设计稿提交在描述性目录 `09-aclblasCgbmv-A2A3/PzeYui/docs/design.md`；若社区维护者指定正式编号目录，应按评审意见迁移。
2. 零维调用同时带非法 `trans/kl/ku/alpha/beta` 时的返回优先级、`alpha=0` 时 A/x 是否可空，需要在实现与验收前确定；本文先采用上述设计并保留补测。
3. 官方精度标准定义有限值最大误差与特殊值处理，但随包 README 写“复数按实部/虚部分别按 FLOAT32 判定”，任务书公式写复数模长。最终用仓库 Verifier 和任务方标准确认比较单位及 0.99 分母，不能自行修改判定器。
4. 尚未取得 A2/A3 真机版本和实测数据；本文为设计方案，不代表代码或性能已达标。

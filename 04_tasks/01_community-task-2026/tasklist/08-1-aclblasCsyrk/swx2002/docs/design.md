# aclblasCsyrk 算子设计（Ascend 950PR）

作者：swx2002。版本：2026-09-09，设计评审稿。

本设计面向 CANN 2026 年 8 月社区任务 aclblasCsyrk（950）。最终实现已通过 950PR 构建、1125/1125 项功能精度、306/306 项特殊值补充检查及200/200项设备Kernel性能门槛；同批API同步计时189/200通过。正式设计评审、计时口径确认、合入和激励验收尚未完成。

# 需求背景（required）

## 需求来源

- [社区任务](https://www.hiascend.com/activities/task-center/details/5024339a4df34e2babe0c8c297c2fd78?menu=guide)
- [任务书](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202608/aclblasCsyrk_Atlas950PR_task_doc.md)
- [官方代码仓库](https://gitcode.com/cann/ops-blas)
- [本任务讨论区](https://gitcode.com/org/cann/discussions/255)

## 背景介绍

ops-blas 提供 handle/stream 方式的 BLAS kernel 直调接口。本任务在 `blas/syrk/arch35/` 补充 complex64 对称秩-k 更新，并在公共头文件增加与任务书逐参数一致的声明。实现以官方 ops-blas 提交 `7f93ab5c73916c13f2b46f80c03a90badd65f9cf` 为基线，基于其 tensor_api 数据布局和硬件定义实现独立的 Csyrk 内核，不改动现有 GEMM 算法。

Csyrk 对复对称矩阵进行更新。转置不取共轭，输出对角线的虚部可以非零。C 使用列主序，只引用和更新 uplo 指定三角；padding 和另一三角必须保留。

# 需求分析（required）

## 需求描述

实现以下接口，维度和 leading dimension 均为 int：

```cpp
aclblasStatus_t aclblasCsyrk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);
```

| 参数 | 位置 / 类型 | 语义与约束 |
|---|---|---|
| handle | Host / aclblasHandle_t | 已创建的有效句柄，使用其绑定 stream；空句柄返回 HANDLE_IS_NULLPTR |
| uplo | Host / enum | UPPER 或 LOWER；非法值返回 INVALID_ENUM |
| trans | Host / enum | OP_N、OP_T、OP_C；OP_C 等价于 OP_T，不共轭；非法值返回 INVALID_ENUM |
| n | Host / int | C 的阶数，n>=0；n=0 为合法 no-op |
| k | Host / int | op(A) 的列数，k>=0；k=0 时只需按 beta 更新 C |
| alpha | Device / complex64 标量 | 不可为空，实部和虚部为 FP32 |
| A | Device / complex64 只读矩阵 | N：n×k；T/C：k×n；n>0 且 k>0 时不可为空 |
| lda | Host / int | N 时 >=max(1,n)，T/C 时 >=max(1,k)，单位为复数元素 |
| beta | Device / complex64 标量 | 不可为空；beta=0 时不读取旧 C |
| C | Device / complex64，原地输入输出 | n×n，只读写指定三角；当前方案 n>0 时要求可写且非空 |
| ldc | Host / int | >=max(1,n)，单位为复数元素 |

其他非法数值或必要指针为空返回 INVALID_VALUE。n=0 也先进行任务要求的参数校验，合法时不访问 Device 数据。k=0 允许 A 为空。

## 需求拆解

1. 新增公共 API、Host 参数校验、arch35 kernel 与 tiling。
2. 正确计算复数 alpha/beta、列主序、转置、非对齐 leading dimension、上下三角与 quick return。
3. 在 `test/syrk/csyrk/` 提供 CSV/GTest、CBLAS Golden、逐元素精度检查、完整性能基线映射和不可漏测的结果统计。
4. 在 Ascend 950PR / CANN 9.1 完成验收，提供源码版本、日志、截图、性能和内存报告。

# 详细设计（required）

## 算子分析

### 数学公式

记 B=op(A)，则 B 的逻辑形状为 n×k，计算 C=alpha·B·Bᵀ+beta·C。对于被选中的 (i,j)：

`C[i+j*ldc] = alpha * sum_p(B[i,p] * B[j,p]) + beta * C_old[i+j*ldc]`。

UPPER 处理 i<=j；LOWER 处理 i>=j。N 的 A 地址为 i+p*lda，T/C 为 p+i*lda。所有输入、标量均按复数计算。

### 支持数据类型和形状

输入输出与标量均为 complex64（两个 FP32 分量）。支持 n,k>=0 的合法列主序矩阵及 lda/ldc padding，不涉及广播、batch、额外 stride 或其他 dtype。大规格受实际可用内存及库 workspace 上限约束，不承诺无法分配的形状成功。

## 算子实现

### Host 侧设计

校验顺序：handle → 非负维度 → 必要指针 → 枚举 → leading dimension。所有错误都在读取 Device 标量前返回。n=0 在校验后直接返回 SUCCESS。

所有路径均在 Device 读取 alpha/beta，Host 不将标量拷回，也不显式同步 stream。准备、乘积与合成阶段按 handle 绑定的同一 stream 顺序下发。alpha=0 时准备和乘积阶段直接返回，不读取 A；合成阶段按 beta 缩放或清零，避免 NaN 输入污染零乘积。

现有工作区足够时，API 在设备完成计算前返回。独立运行时测试排队大矩阵计算和 Device 标量更新，在 API 返回后检查前序事件仍未完成，再校验两次运算结果。库公共工作区首次分配或扩容仍可能同步；这与稳定工作区下的异步下发分开说明。

分支规则：

| 条件 | 行为 |
|---|---|
| n=0 | 校验后成功，无 kernel |
| (alpha=0 或 k=0) 且 beta=1 | 成功，不访问 A/C 数据 |
| alpha=0 或 k=0，beta=0 | 不访问 A/旧 C，选定三角直接写零 |
| alpha=0 或 k=0，其他 beta | 不访问 A，仅缩放选定三角 |
| 一般乘积，beta=0 | 计算乘积，不读取旧 C |
| 一般乘积，beta!=0 | 计算乘积并合成选定三角的旧 C |

任务书 C 参数表只列出 beta!=0 时的空 C 错误，但一般 beta=0 仍需有效输出地址。当前统一采用 n>0 必须非空 C；对完全 no-op 与空 C 的组合，提交评审明确返回码，不能用悬空输出地址参与计算。

### 四个实数乘积与残差补偿

将 B 写作 Br+iBi：t1=Br·Brᵀ，t2=Bi·Biᵀ，t3=Br·Biᵀ，t4=Bi·Brᵀ。复数乘积实部为 t1-t2，虚部为 t3+t4。

一般紧凑矩阵采用三个内核，按同一 stream 顺序执行：

1. AIV 准备：将 A 解交织为紧凑实部/虚部；补偿路径额外生成低位残差、检查异常值并计算实部对角线。
2. AIC 乘积：每个三角块共用输入搬运，同时保留四个 FP32 L0C 累加器，生成四份实数乘积。
3. AIV 合成：应用复数 alpha/beta，仅写入指定三角；需要补偿的大规格对角线实部采用准备阶段结果，小规格采用原生 FP32 乘积结果。

n<=256 且 k<=512 时，Cube 使用原生 FP32 MMAD，对角线直接来自 t1-t2，省去准备阶段的重复归约。较大规格使用 950 HF32 的三项残差补偿。950 HF32 保留十位尾数，采用 nearest-even 转换。对每个 FP32 值 x，令 xh=HF32(x)，xl=HF32(x-xh)，以 `xh*yh + xl*yh + xh*yl` 近似乘积，FP32 累加。低位交叉项先累加，高位乘积后累加；不计算 xl*yl。它保留 FP32 输入/输出接口，但不是逐位等价的原生 FP32 运算，必须通过完整官方误差标准验证。

准备阶段在 SIMT 中按 nearest-even 位规则生成残差，MMAD 对输入执行相同 HF32 转换。Cube 进入时显式设置模式，结束时关闭 HF32，避免状态影响后续内核。[官方 HF32 说明](https://asc.gitcode.com/api/SIMD-API/c_api/cube_compute/asc_enable_hf32.html) 和 [950 格式图](https://asc.gitcode.com/assets/mmad_hf32_950.Dm8HV4Yt.png) 是硬件格式依据。

对于采用残差补偿的大规格对角线，先对每个复数计算 `real*real-imag*imag`，再补偿累加并归约，避免两个大平方和相减造成的精度损失。该过程与解交织处于同一内核，直接读取原始 A，因此没有依赖其他 AIV 核刚写入的数据。

### 分核、分块与异常路径

Cube 三角块循环分配到最多28个 AIC 核。n<=256、n<=1024、较大 n 的 BM 分别为32、64、128；BN=BM，BK=4096/BM。每份输入块16 KiB；原生路径使用4份 L1 输入、L0A/B 双缓冲，补偿路径使用8份 L1 输入、L0A/B 单缓冲，4个 L0C 累加器共16/64/256 KiB。事件控制 MTE2、MTE1、MMAD、Fixpipe 之间的读取和覆盖。

实际 PlatformAscendC 查询：L1=512 KiB，L0A/B各64 KiB，L0C=256 KiB，UB=248 KiB。Csyrk 独立使用目标容量限制，不修改上游公共硬件表。尾块传递真实 m/n/k 长度，对角块可计算完整临时矩阵，但写 C 时严格屏蔽另一三角。

补偿路径的准备阶段为每个 AIV 核写一个异常标记，识别非有限值、非零绝对值小于1e-16或绝对值大于1e16的输入。原生 FP32 小规格利用 t1/t2 对角线的非负平方和检测受影响的行和列，省去准备阶段的标志归约。合成阶段遇到异常输入或异常标量时放弃相应临时乘积，按 Netlib 参考次序直接计算所选三角。此分支保留浮点中间乘积的溢出语义，处理 NaN/Inf 与极值，不使用截断或替换输入。回退的复数乘法保留 FP32 中间值，并实现 C 复数乘法在两个临时分量均为 NaN 时的无穷大恢复规则；仅将实虚四次乘法机械组合并不足以匹配 Netlib 的特殊值行为。

紧凑且 n<=64、k<=256 时采用单内核复数点积。带 lda/ldc padding、k=0 或 n<=64且k>256时采用直接参考次序路径。直接点积发现输入超出数值安全范围、异常标量或非有限输出时，也按参考次序重新计算该元素。分派仅依据形状、步长和输入数值语义，不依赖用例编号、随机种子或预存答案。

合成阶段 n<=1024 使用 SIMT，按输出元素分配线程；大矩阵在 beta=0 时对非对角块采用64×64矢量搬运与计算，对角块使用 SIMT。beta!=0 时使用 SIMT 的 C 复数乘法处理旧 C 的特殊值；n>65535 且 beta!=0 采用参考次序路径以保留64位索引范围。仅在 n<=65535 时使用32位行列索引除法。所有路径只写指定三角。非对角块位于一个完整三角内，尾块仅搬运有效行列。

### Workspace 与内存

`tempLdc=align_up(n,16)`。四份实数乘积各占 `4*tempLdc*n` 字节；A 的实部、虚部及其残差共占 `16*n*k` 字节；对角线占 `4*n` 字节；核标志预留2048字节：

`W = align_up(16*tempLdc*n + 16*n*k + 4*n + 2048, 512)`。

所有乘法、加法先检查溢出，再调用 EnsureDefaultWorkspace。用户工作区不足时返回 ALLOC_FAILED；库拥有的工作区沿用公共扩容策略与2 GiB上限。实际申请仍受可用内存限制。当前紧凑乘积路径在 Host 未读取 alpha，因此 alpha=0 也可能先准备工作区；直接路径不需要乘积工作区。

n=k=2048 时请求128 MiB+10 KiB，4096时请求512 MiB+18 KiB；库实际保留容量可能更大。合成内核的大矩阵矢量路径使用96 KiB UB缓冲，其余存储由编译器分配。报告记录实际 workspace、Device 空闲内存的分阶段观测及 Host 进程累计峰值RSS，不将源码容量计算当成瞬时峰值测量。

### Kernel 搬运、计算与写回

- 四个乘积使用 `tempLdc`，有效输出只在合成阶段按列主序定位。
- 合成计算 t1-t2、t3+t4，应用复数 alpha 后加入 beta*C_old；零乘积路径只缩放或清零。
- beta!=0 时仅读取所选三角；beta=0 不读取旧 C，NaN 毒化验证覆盖此分支。
- 每个线程精确写入一个 complex64 或若干不重叠元素，不以向量整块写回覆盖另一三角。
- 任意 ldc 和非对齐边界由元素寻址处理；未引用三角与 padding 使用字节比较验证。

### 精度与性能风险

4M 会改变复数乘积的归约顺序，尤其在大 K 的实部相减中放大消减误差。讨论区出现 TC_SQ_035/058 的误差反馈，这两例纳入首轮 quick。官方 Golden 保持 CBLAS，不能通过更换为自己的 4M Golden 或放宽阈值掩盖差异。

失败时固定 seed、保存输入与逐元素误差位置，增加 FP64 标量累加诊断来区分 Golden 误差和 NPU 误差。必要时评估分块补偿或更高精度的受限规格路径；重新跑完整精度和全部性能，仍以获确认的官方标准为准。

最终实现 fused_hf32x3_v10 已通过1125/1125项精度、306/306项特殊值补充检查、200/200项内核性能，同批API同步计时189/200。最小Kernel余量约3.73%，具体用例和批次见自测报告。任何实现变更都需要与对应源码的回归结果关联，旧版结果不自动适用于新版。

## 支持硬件

| 芯片 / 软件 | 范围 |
|---|---|
| Ascend 950PR / arch35 | 本任务目标；已构建并完成全量功能精度验证 |
| CANN / asc-devkit 9.1 | tensor_api 构建前提，使用实际安装版本记录 |
| 其他架构 | 此次不新增实现；公共头文件声明供跨产品线一致使用 |

## 算子约束限制

A 只读、C 原地更新；不支持 A 与 C 重叠造成的自覆盖。alpha/beta 必须为 Device 指针，当前不提供 Host 标量模式。同一个 handle 的 workspace 生命周期及线程安全沿用 ops-blas 约束，并发调用需遵守库要求。

# 可维可测分析

## 精度标准 / 性能标准

| 标准 | 实施方式 | 来源 |
|---|---|---|
| Golden | cblas_csyrk；OP_C 映射到 CblasTrans；目标环境记录具体 Netlib/CBLAS 依赖及编译参数 | 任务书 3.2、3.5 |
| FP32 分量精度 | 实部/虚部分别统计 atol=2^-16、rtol=2^-10、matched_ratio>=0.99 | 任务书及官方精度标准 |
| 最大误差 | 每个有限值误差<=max(0.01,32ULP(golden))；子正常数 ULP=2^-149 | 任务书上界口径，提请评审确认 |
| 特殊值 | NaN/NaN 匹配，Inf 符号一致；不匹配直接失败 | FP32 特殊值语义 |
| 存储保护 | 未引用 C 三角与 padding 按字节保持；beta=0 不被旧 C 的 NaN 污染 | 任务接口语义 |
| 性能 | 每条 GPU 基线换算 gpu_ms×1000/0.4；10 次预热、60 次有效样本均值；官方200条均为beta=0 | 官方 200 条 GPU 基线和任务书 |

三个方阵 UPPER/N 限值与基线逐行取更严格者：512→126.27 µs，1024→382.66 µs，2048→1970.075 µs。前两项受任务书小数限值约束，第三项采用原始 GPU 数字精确换算。

## 用例与证据

保留官方 1200 条 CSV 原件及 200 条已填充 GPU 基线，不修改、重排或扩增官方 PF 编号。适配版本仅对 TC_ED_300/301 的非法枚举期望从 INVALID_VALUE 勘误为 INVALID_ENUM，并显式记录映射。

另加 120 条随机用例：60 均匀分布 [-5,5]，60 正态分布 mu∈[-5,5]、sigma∈[0.1,2]，实/虚独立，alpha/beta/A/C 使用分开的随机流，正态不裁剪。覆盖 2 uplo×3 trans×10 组形状，两种分布配对；lda 最小值+3，ldc=n+5。四条额外用例覆盖 alpha=0 或 k=0 且 beta=0、旧 C 含 NaN。

CSV 共1324条：1000官方精度/功能、124补充精度、200性能；另有NullHandle一例。最终 accuracy 必须1125/1125，performance必须200/200，任何跳过、重复、漏测或非零退出均失败。

GTest 输出逐例返回码、seed、实/虚误差统计、另一三角变化数及性能明细。Python 同时核对 XML、日志、CSV 和基线，不从文件名或静态源码推导 PASS。每次运行独立保存日志目录、源码提交/工作区状态、CSV 哈希、CANN/devkit 版本及 NPU 状态。

保留 API 同步计时：从调用前到绑定 stream 同步完成，输入分配及随机生成在计时外，包括Host下发和同步开销。同时提供 msprof 内核口径：每次调用全部内核耗时之和，预热 10 次后取 60 次平均，排除 Host 和内核间空隙。采集模式下 GTest 成功仅表示完整执行，独立脚本对全部 200 条限值给出性能 PASS/FAIL。日志记录 pipeline 版本并验证每次调用完整内核序列，遗漏或额外内核一律拒绝；保留同批 API 实测，不能将两个口径混为同一结论。官方测试说明允许 msprof 精确内核测量，最终口径提请评审确认。非零 beta 场景在计时外恢复 C。记录的输入存储量另标为计算值，最终验收提供对应版本的真实 Host/Device 内存和 workspace 数据及实/虚、性能截图。

## 兼容性分析

新增 API 声明，不修改现有 Ssyrk、Cherk、GEMM 接口。复用上游内部 kernel 时固定并记录依赖基线，修改后运行单算子构建并核对符号与测试发现结果。低于 asc-devkit 9.1 的 arch35 构建由现有版本列表过滤 Csyrk，不能将未构建或零测试当作通过。

## 评审待确认事项

1. 残差补偿计算满足任务精度的实现方案，以及官方允许的 msprof 全内核耗时口径。
2. INVALID_ENUM 勘误、完全 no-op 与空 C 组合的返回码，以及每元素 max(0.01,32ULP) 的精度解释。
3. 目标机器的 CBLAS 实现与编译选项；大 K 误差诊断不得替代官方参考结果。

设计确认、950PR 全量通过、证据材料齐备后，才能进入正式验收提交。

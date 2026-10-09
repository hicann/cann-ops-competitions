# 单精度实数 Cholesky 分解、求解和批量接口（950）算子设计文档

贡献账号：`bluerain117`。文档版本：2026-10-08（原设计 PR 的增量修订）。

本设计以 2026-10-05 的 56 文件候选源码快照 `ba599d64db1895372f3c24c9d409e59f8782885f2c4502325a117f01a9a7c952` 为基线，并在增量设计中说明后续 D9、wide、e321、A3 和 SpotrsBatched G2/grid56 方案。各候选使用独立源码身份及测试证据，尚未选定通过全量验证的最终组合版本。以下补充用于设计检视；设计评审与实现、自验证并行推进，不以全量测试结束作为提交或更新设计 PR 的前提。正式验收前仍需完成设计合入及同版本的完整自验证和交付材料。

# 需求背景（required）

## 需求来源

社区任务“9月社区任务-单精度实数Cholesky分解、求解和批量接口(950)”，任务 ID `e56f6d6b1eb042cb8d8dc794c8c53db6`。依据为[官方任务书附件](https://www.hiascend.com/p/resource/202609/e47c7f5c0721477cbd4f503951202f6f.zip)、[CANN 社区任务指南](https://www.hiascend.com/developer/activities/cann-community-task)及[本任务讨论帖](https://gitcode.com/cann/ops-solver/discussions/4)。

文档按[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织，已通过[原设计 PR #1985](https://gitcode.com/cann/cann-ops-competitions/pull/1985)提交，设计补充和检视答复持续在该 PR 更新。设计评审合入与实现、自验证可并行推进；正式平台验收时两者均需完成。产品实现目标仓为 [cann/ops-solver](https://gitcode.com/cann/ops-solver)，五个计算接口作为同一任务完整交付。

## 背景介绍

Cholesky 分解利用实对称正定矩阵的结构，得到可复用的三角因子。求解接口读取该因子执行两次三角代入；求逆接口由同一因子生成原矩阵的对称逆。批量接口处理多个独立设备矩阵，通过矩阵间并行提高小矩阵场景的吞吐。

接口、参数、列主序布局及状态语义对齐任务书指定的 [NVIDIA cuSolver DN](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-legacy-api)。cuSolver 用作接口与性能参照；核心计算采用 AscendC，在 NPU AI Core 执行。

| 接口 | 功能 | 输入与输出 |
| --- | --- | --- |
| `aclsolverSpotrf` | 单矩阵 Cholesky 分解 | SPD 矩阵输入；原地输出所选三角因子及标量 `devInfo` |
| `aclsolverSpotrs` | 单矩阵求解 | 已分解因子只读；`B` 原地覆盖为 `X`；标量 `devInfo` |
| `aclsolverSpotri` | 单矩阵求逆 | 已分解因子输入；原地输出所选三角的逆矩阵；标量 `devInfo` |
| `aclsolverSpotrfBatched` | 批量分解 | Device 指针数组；各槽原地输出因子；逐槽 `infoArray` |
| `aclsolverSpotrsBatched` | 批量求解 | Device 因子与右端指针数组；各槽右端原地覆盖为解；标量 `info` |
| `aclsolverSpotrf_bufferSize` / `aclsolverSpotri_bufferSize` | 工作区查询 | 返回以 float 元素数计的用户工作区需求 |

# 需求分析（required）

## 需求描述

公开接口保留任务书参数次序、int32 维数和 `aclsolverStatus_t` 返回类型；复用 `aclsolverCreate/Destroy/SetStream/GetStream`。所选三角为 `LOWER=0` 或 `UPPER=1`，矩阵为 FLOAT32 列主序，`lda/ldb >= max(1,n)`。矩阵、info、workspace 及批量指针数组位于 Device；bufferSize 输出参数 `Lwork` 为 Host 标量指针。

```cpp
aclsolverStatus_t aclsolverSpotrf_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverSpotrf(aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverSpotrs(aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, int nrhs, const float *A, int lda, float *B, int ldb, int *devInfo);
aclsolverStatus_t aclsolverSpotri_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverSpotri(aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverSpotrfBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *Aarray[], int lda, int *infoArray, int batchSize);
aclsolverStatus_t aclsolverSpotrsBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, int nrhs, float *Aarray[], int lda, float *Barray[], int ldb, int *info, int batchSize);
```

`info=0` 表示成功；非法参数返回 `ACLSOLVER_STATUS_INVALID_VALUE`，可写 info 时写入不计 handle 的负参数序号。分解失败报告最小非正定阶数 `k`；批量分解逐槽报告，其他槽继续。求逆因子奇异报告 `k`；两个求解接口只报告参数错误。空问题按任务书成功返回，批量求解的非空问题只支持 `nrhs=1`。

## 需求拆解

1. 实现五个计算接口及两个 bufferSize，支持 LOWER/UPPER 和合法 lda/ldb padding。
2. 覆盖任务书最低范围：n=1..4096；单矩阵求解 nrhs=1..128；批量接口 batch=1..30000。同时执行发布包中更大的批量规格，不能据最低范围删减用例。
3. 满足 FP64 golden 的逐元素精度或规定的残差复核，并独立检验 info、非默认 stream、重复执行 bit-wise 确定性和异常/空问题。
4. 全量性能逐例达到 GPU 平均耗时 / NPU kernel 平均耗时 >=0.35，保留固定 GPU 金标与完整 msprof 原始记录。
5. 完成公开头文件、算子及 API 文档、示例、测试、复现步骤和模板报告；验收方可访问同一版本源码及交付材料。

# 详细设计（required）

## 算子分析

### 数学公式

LOWER 分解为 `A=L L^T`；UPPER 为 `A=U^T U`。LOWER 求解先执行 `L Y=B`，再执行 `L^T X=Y`；UPPER 先执行 `U^T Y=B`，再执行 `U X=Y`。求逆等价于以单位矩阵为右端求解 `A C=I`。批量接口对设备指针指向的每个矩阵独立执行相同运算。

### 支持数据类型与形状

仅支持 FLOAT32，计算中沿用 float；不扩展到复数、半精度或广播。`A(i,j)` 的地址为 `A+i+j*lda`，右端地址为 `B+i+j*ldb`。不要求 `lda==n`，不将 Device 指针数组解释为连续三维张量。

候选实现按 n=128 分流：单矩阵分解及批量分解/求解使用不同的小、大矩阵路径。Spotrs、Spotri 和两个批量 Host 入口限制 n<=4096；Spotrf 入口当前未显式拒绝 n>4096，因此该候选不据此声明超出任务范围可用，超限处理纳入公开接口检视。

## 算子实现

### 3.2.1 host侧设计

采用 ops-solver Host C API + AscendC Kernel 直调。Host 负责参数检查、stream 获取、内部 workspace、tiling 和按依赖顺序下发。正常算术路径在设备完成；标量 info 使用设备写入 kernel，避免异步引用 Host 栈内存。

下表说明 10 月 5 日基线路由。新增候选的分工、开关和资源变化见后面的增量设计；未启用的实验不改变基线默认路由。

| 路径 | 基线 Host 路由 |
| --- | --- |
| Spotrf，n<=128 | 单 AIV block 的 UB 驻留小矩阵分解 |
| Spotrf，n>128 | NB=64，逐 panel 下发 diag、apply、trailing update；LOWER 直接处理，UPPER 转置到内部 workspace 后复用 LOWER，结束转回 |
| Spotrf trailing update | 仅 `lda==n` 且 Cube tiling 成功时走 Cube GEMM + AIV subtract；其他情形走 AIV SYRK |
| Spotrs 默认 v0 | 40 个 AIV block 按 rhs 分工，逐列做两次三角代入 |
| Spotrs v1 | `SPOTRS_V1=1` 且 n>=129 才选择；BR=64、rhs 面板宽<=128；三组 GEMM tiling 与 MIX kernel 配合 |
| Spotri v0 | 上传内部单位矩阵；设备扫描因子对角；读取 flags 并同步一次 stream；奇异时写 k，否则以 Spotrs v0 求解并 mask 回所选三角 |
| SpotrfBatched / SpotrsBatched | 40 个 AIV block 按 batch 槽分工；n<=128 小矩阵，n>128 大矩阵；求解合法调用先写标量 info=0 |

Spotrf 的 Cube tiling 从最大 trailing extent 一次生成后复用，各 panel 传入实际 m、nb，内核检查实际范围。AIC 数由 `GetCoreNumAic()` 获取，当前失败回退值为20；不能把一台设备的核数写为固定硬件事实。apply 默认40个 AIV block，`SPOTRF_APPLY_CORES` 可调整为1..40。本候选快照中 `SPOTRF_PANEL_V2` 默认1，显式设为0可选择原路径；这不是对最终交付默认值的预先承诺。正式自测记录所有运行开关，并确认 `SPOTRF_DEBUG_STEPS` 未截断分解。

候选 bufferSize 返回0，内部 workspace 由 handle 按需求增长并复用。公开的参数校验和该内部工作区约定需按任务书完成检视与边界测试，不能只凭常规成功调用判定接口合同已满足。

| 内部 GM 工作区 | 字节需求 |
| --- | --- |
| Spotrf 小矩阵 | 无额外 handle scratch |
| Spotrf 大矩阵 | `4096 + upper*(n*lda*4) + cube*((n-64)*64+(n-64)^2)*4`；Cube 不可用时实际走 AIV，但按预分配容量处理 |
| Spotrs v0 | 无额外 handle scratch |
| Spotrs v1 | `64 + 2*n*wMax*4 + maxMk*64*4`，`wMax=min(nrhs,128)`，`maxMk=(ceil(n/64)-1)*64` |
| Spotri v0 | `64+n*n*4`，flags 与单位矩阵/解缓冲 |
| 批量 AIV 路径 | 矩阵及指针数组由调用者分配；不为每槽复制一份 n*n 的 handle scratch |

handle 增长时保留退休 buffer，避免释放同 stream 中尚在使用的数据；达到退休列表容量时返回 `ALLOC_FAILED`。销毁 handle 释放当前及退休空间。同一 handle 的使用必须遵守调用方流和生命周期约定；跨流并发不作额外支持承诺。

Spotri v0 的一次 Host 同步用于读取奇异性标志并决定是否发射求解，其必要性和开销纳入评审。后续优化可以把单位矩阵生成及设备判定移入设备链路；候选方案不声称所有接口成功路径均无 Host 同步。

### 3.2.2 kernel侧设计

小矩阵将所需数据搬入 UB，按固定列序计算，避免矩阵内部无序归约。大矩阵按 panel 或 resident columns 分块。对齐及 tail 由 n、lda 和实际块尺寸控制，使用 DataCopy/DataCopyPad 搬运有效数据。用 `MTE2_V` 等事件确认搬入完成，并在下一次搬入覆盖缓冲前确认上一轮 vector/scalar/搬出已结束。

单矩阵分解的大矩阵步骤为：

```text
for k in 0..n step 64:
    factor diagonal panel A11; set info/abort if pivot is non-positive
    solve panel A21 with A11, using fixed-order vector operations
    if contiguous layout and Cube tiling available:
        materialize L21 for GEMM
        C = L21 * L21^T on Cube
        subtract C from the stored trailing triangle on AIV
    else:
        update stored trailing triangle on AIV
```

panel 的主要 AscendC 算术序列为按 pivot 求平方根/倒数、`Muls` 缩放列、`Axpy` 更新后续列；v2 使用逐列 Axpy 并合并可对齐的二维搬运；本基线的 `SPOTRF_PANEL_V2` 默认值为1，与 Host 侧说明一致。Cube 路径使用 AscendC Matmul API 和 `TCubeTiling`，没有用 CPU GEMM 替代核心计算。非正定 info、有效失败前缀及 abort 后 trailing 行为单独检验。

Spotrs v0 对每个 rhs 执行 forward/backward substitution：搬入列因子；按固定 j 顺序取对角并除法；`Axpy` 或 `Mul/ReduceSum` 计算剩余贡献；输出 X。n<=4096 的固定列缓冲支持合法 padding。候选源码中的 opt-in v1 存在循环边界、布局和跨核事件参与者等静态缺陷，暂停设备启用。后续分块求解方案采用同一 stream 上分别提交的 AIV 对角代入、因子条带打包、Cube GEMM 和 AIV 相减；所有方向统一 Ac 的行主序 M×K 与 Q 的行主序 n×w，分别按 uplo/pass 选择实际因子三角。该分阶段方案已形成独立候选，wide 求逆是其具体复用路径；实现边界见下文，不能据此声明本基线或最终组合已经全量通过。

Spotri v0 设备 prescan 检查因子对角中的首个零值，产生奇异性 flags；可逆时复用 v0 求解 `A C=I`，设备 mask kernel 写回所选三角。后续可评估专用三角求逆与乘积方案，但本候选不将该优化描述为已实现。

10月5日基线的批量接口在设备读取指针数组，按 `ceil(batch/40)` 槽区间分配到 AIV block，每槽使用独立矩阵地址。大矩阵分解为 AIV-only 左视 Cholesky，8列驻留在 UB，历史列流式搬入后 Axpy 更新当前8列。UPPER 用32x32成对块转置，两个原始块全部保存后再写任一目标，最后转回。该基线在每轮贡献读取前加入 scalar/vector/搬运依赖检查，防止下一轮 scalar 取系数越过前一轮列写回。

批量分解的 info 使用同 stream 的第二次独立 pass，逐矩阵检查保存对角并按分段 UB 搬出；避免不同 block 对同一32字节 info 区的非对齐写互相覆盖。大矩阵批量求解同为 AIV-only 列流式，两次三角代入，无矩阵间算术依赖。

### 候选增量设计（2026-10-07）

本节更新实际机制，不将不同候选的局部结果合并成最终验收结论。原有七个公开函数签名、列主序、所选三角、info 和调用方 stream 约定不因这些实验扩展而改变。

| 候选及源码身份 | 与基线的设计差异 | 路由及当前边界 |
| --- | --- | --- |
| D9，`d9b5cd3409b1e355cd474b7753b053890b70ed79931e2441314d5402d8751335` | 分解的 panel apply 使用 MicroAPI VF：以不超过64行的独立片段执行固定 j/c 顺序的缩放及 FMA 更新；复用原 tile、diag 和 inverse 缓冲，在 scalar 生产者与 vector 消费者之间同步，在循环间保留 vector store/load 依赖 | NB=64、diag/apply/trailing 的同 stream 分阶段结构不变；FMA 舍入与旧 Axpy/Mul/Sub 路径不预设逐位相同，精度和同版本重复确定性分别验证。该机制已包含在后续 e321 源码中；早期 D9 的44个有限用例和 P01/P02记录仅归属 D9 |
| wide，`83c25c74d8f7e704bcdffe6d9ec64cb474da6ef5b4fcc7a33709c1385c25cb28` | 以宽右端矩阵复用 BR=64 分块求解；Spotri 先设备扫描首个零对角并进行一次 Host 同步，再在 Device 生成单位矩阵 Q；依次执行 AIV 对角代入/因子打包、Cube GEMM、AIV 相减，最后只写回选定三角 | `SPOTRI_BLOCKED_WIDE=1` 显式启用，未设置时保留 v0。各实际 (m,w,k) 的 tiling 在发射前生成并检查覆盖，避免直接复用最大形状；阶段依赖由同 stream 的独立 kernel 顺序表达。83c 的有限48例与原始 P07/P08记录不等于最终779/768完整矩阵 |
| e321，`e321df04bdf5b1a6963b8a7fd637b0acfe7a891fb283eee5b2a36d530c3f9cb6` | 在83c基础上补齐 Spotrf 的 `Workspace==nullptr && Lwork>0` 校验，返回 `INVALID_VALUE`，仅当 devInfo 非空时写 `-5` | 检查位于 lda 检查后、负 Lwork 检查前；`Lwork<0`仍按 `-6`，不改变算术 kernel 或公开 ABI。e321 的49次有限原生调用记录不替代全量功能、精度和性能验证 |
| A3 narrow，`a3e74f0f4182559c2054a20c63ab867ed1376e8b162df393eb1e06f081f811bd` | 在e321基础上增加单右端 AIV 求解候选：BR=32 对角块、最多512行的因子条带，rhs 留在UB，按上下三角方向执行两次代入；VF中使用直接 Div，保留有效lane掩码及读写依赖 | 仅 `SPOTRS_NARROW1=1`、`SPOTRS_V1`未请求、`nrhs=1`、`0<n<=1024` 时启用，默认关闭。已实际编译，但首个 n33 LOWER 设备用例发生 VEC 访问UB越界；该实验仍需定位修复，未认定数值、边界保护或性能通过，默认路线保持不变 |

wide 求逆的内部GM工作区为 `512 + 2*align512(n*n*4) + align512(n*64*4)` 字节，依次容纳 flags、Q、GEMM结果及因子条带；公开 bufferSize 仍沿用内部handle工作区约定。所有因子读取结束后才发射选定三角写回，奇异分支在求解前返回；该候选仍有一次真实 Host 同步，不描述成完全无同步方案。

A3 narrow 的七块显式UB申请合计221312字节：rhs 16384字节；raw、col、panelOffset各65536字节；diagRaw、diag各4096字节；diagOffset 128字节。这是源码账本，不是运行时容量安全的证明，也不能据此将已发生的UB越界归因为总量溢出。后续诊断不改变其默认关闭状态，定位前不将其升级为交付路线。

批量分解/求解的基线语义保持不变；其余隔离的批量优化尚未确定为最终交付方案，本次不把候选性能或未验证资源布局写成产品承诺。最终采用的路由、开关和资源账本将在原PR中继续更新。设计评审可针对当前算法分工、公开契约及资源方案继续，不等待整个实现矩阵结束。

### SpotrsBatched G2/grid56 增量（2026-10-08）

本节对应候选源码 `93c628343af4232194ba3a68905503a90ab0a376f2d6177b78c45aa0c7273772`、实编库 SHA256 `41bde03dabbc301bfdd11627c96b17771715f059e37c959d516dc19140be2f46`。实现位置为 `src/spotrs_batched/spotrs_batched_host.cpp`、`spotrs_batched_g2_kernel.cpp`、`spotrs_batched_kernel.cpp` 和 `spotrs_batched_large_kernel.cpp`。公开参数、Device 指针数组、任意合法 lda/ldb、LOWER/UPPER、标量 info 及调用方 stream 约定不变；不依赖测试内容复用答案。

| 合法非空批量求解 | 当前候选路由与分片 |
| --- | --- |
| `1<=n<=32` | G2：两个独立系统各占一个32-lane区段，共64 lane；按 `groups=ceil(batch/2)` 分组，再按 `ceil(groups/GetBlockNum())` 分片，最后奇数槽只读取一个系统的 Device 指针 |
| `33<=n<=128` | 原 whole-matrix-in-UB small 路径，按 `ceil(batch/56)` 分槽；n128 仍使用原标量取 pivot/除法、scratch 行列构造及 Axpy 两次代入 |
| `129<=n<=4096` | 原列流式 large 路径，按 `ceil(batch/56)` 分槽，保持单系统两次代入算序 |

Host 的 `SPOTRS_BATCHED_NUM_BLOCKS`、small 的 `NUM_BLOCKS` 和 large 的 `LARGE_NUM_BLOCKS` 同为56；G2 使用实际启动的 `GetBlockNum()`。这是本候选的固定逻辑 grid，不能解释为已有56个当前可用物理核或保证加速；本次主机 SDK 查询虽返回 `GetCoreNumAiv()=56`，其当前分配、可用及物理芯片语义仍未确认。其他接口的基线40-block设置没有随之修改。

G2 为每系统分配32×32因子区和32元素 RHS 区，活动掩码同时限制实际 n 与有效系统数。因子通过二维 DataCopyPad 搬入固定32行 UB pitch，GM stride 根据实际 lda 计算；输出仅写 n 个 RHS 元素。Fast VF 缓存真实对角，对通过范围检查的对角使用原 `PRECISION_0ULP_FTZ_FALSE` Div 模式一次向量计算 `1/d`，随后每个 pivot 通过 RegGather 取倒数并用 Mul 求商，按固定前代/回代顺序用 MulAddDst 更新有效行。倒数乘法与逐次除法不预设逐位等价。

资格检查保留对角/倒数的正 `[2^-32,2^32]` 范围，以及初始 RHS、有效系数、更新值的零或绝对值范围；首个回代 pivot 的检查保留。有效系数放大（`abs(coefficient)>diagonal`）、严重相消（原值非零且 `abs(updated)<2^-12*abs(old)`）等异常累积为每对 sticky flag。检查失败的 lane 先记录异常，再以安全值完成 Fast VF；这些保守工程检查不是全部 FP32 输入精度等价的证明。每对结束归约全部64 lane 的异常标志，经 V→S 同步由 AIV scalar 判定；若任一系统不合格，在尚未写回 GM B 时重新读取原始 B，经搬入同步后对整对执行保留逐 pivot 原 Div 的 `SolvePair32` Slow VF，最后统一写回。每对开始清标志，标量读后的 S→V 及搬出后的 MTE3→V 依赖保留；奇数尾不读取不存在的第二系统。

G2 显式 UB 账本为因子8192字节、RHS256字节和标志32字节，共8480字节；n128 small 的66624字节及 large 的99328字节账本不变。当前实编 `CPU_BINDING.json`（SHA256 `3bb3db1ea7127ddbebe6e186339bd81816e1a7637656eadb57f776cb9099bb4e`）记录 Fast VF stack320字节/VReg23/PReg8，Slow VF stack576字节/VReg32/PReg8；这是该构建的编译器资源记录，不能据 stack 值推断 spill 或当成硬件容量证明。

同一 source93/实编库的有限回归与两个原始完整批量 case 已执行。下表均保留3次预热及30次正式调用的全部66条目标 kernel 记录，以每调用全部目标核合计后取30次算术平均；GPU金标与0.35门槛不变。

| 原始 case | 精度与完整性 | 当前性能结果 | 审计报告 SHA256 |
| --- | --- | --- | --- |
| P11 / `spotrsBatched-0008`，n32、batch99659、LOWER | FP64主判、原检查器数值检查、3次精度及profile末次完整B对冻结ca62位比较通过 | NPU均值4.0432917333 ms，GPU/NPU=0.3666072343，逐例门槛通过 | `5f98c01bd4a9999a72aec1f1e81c1bac4fec2386d15b28533f85c2dff086ba43` |
| P12 / `spotrsBatched-0024`，n128、batch45897、LOWER | 同上；3次完整A逐次D2H扫描/memcmp/SHA及B/info/guards通过，精度阶段不保存3份物理A副本 | NPU均值142.8255861333 ms，GPU/NPU=0.0334386865，仍未达到13.6454285714 ms上限 | `272461d565f6c752765adf924733531d43b7ecb847de0a818c7099e9c4fb9e79` |

上述结果仅归属该候选及其实验环境，不合并历史其他源码的通过项；原检查器数值通过也不等于官方验收裁定。P12性能、同版本五接口完整精度/功能/逐例性能矩阵、模板材料、设计合入及平台验收仍待完成。当前环境记录为950PR、CANN9.1.0B243；是否匹配最终验收及目标仓配套版本尚待确认，不能以本轮已运行替代正式版本匹配结论。

### Tiling 与 UB 规划

核间层面以 rhs、batch 槽或 trailing 行切分；核内层面以 panel、tile 或列缓冲切分。下表是10月5日基线源码显式分配的主要 UB 账本，单位为字节；后续新增候选按上节分别列账，编译器及 Matmul 额外资源以目标环境资源报告复核，不能把账本当成设备实测。

| Kernel 路径 | Buffer / 大小 | 显式总计 |
| --- | --- | ---: |
| Spotrf 小矩阵 | `ubuf` 固定160 KiB；矩阵视图使用 n<=128 的有效部分 | 163840 |
| Panel AIV | tile 64x512；diag与tmp各64x64；brcb 64x8；inv/row/offset各64，元素4B | 166656 |
| AIV SYRK | tile与tmp各128x128；row/col各128x64；brcb 128x8，元素4B | 200704 |
| Spotrs v0 / 大矩阵批量求解 | x/y/xwork/col/dot/redwork各4096 float；redout256 float | 99328 |
| Spotrs v1 AIV | r/z/tile/tileT各64x128 float；diag64x64；scale128 float；offset128 uint32 | 148480 |
| 批量小矩阵分解/求解 | `(nAligned^2+2*nAligned+16)*4`；`nAligned=ceil(n/8)*8` | n=128时66624 |
| 批量大矩阵分解 | resident8列+stream1列；sqrt64B；transpose/mirror各32x32 float | n=4096时155712 |
| 批量 info pass | 4096 int32 staging | 16384 |

所有分配按32字节对齐，平方根 src/dst 分离；大矩阵转置 staging 与临时流式列不复用同一段空间。边界块只搬运有效行/列，padding 保持对应 leading dimension。Cube 自身 L1/L0 与 Matmul UB 资源由 tiling 配置确定，正式资源报告与实际加载库一并留档。

## 支持硬件

| 芯片版本 | 适配情况 |
| --- | --- |
| Ascend 950PR（Atlas 950） | 本任务目标；独立构建已完成，完整设备验证正在推进 |

任务要求 CANN>=9.0.0 并与仓库配套版本一致。10月5日基线环境记录为 Ascend950PR、CANN9.1.0、GCC12.3.1；CPU 精度检查工具为 NumPy2.5.1/SciPy1.18，容器内存上限32 GiB。候选修订保留各自实际编译、驱动和设备环境记录，最终配置以对应版本的真实记录为准。构建、符号导出、实际加载库 SHA256 和完整测试结果在自测报告中记录，迁移前的结果不自动归属此环境。

## 算子约束限制

- 五接口必须完整交付，两个求解接口及求逆接口输入为已分解因子，不能传原始未分解 A。
- 批量求解非空问题仅 nrhs=1；nrhs=2 返回错误。空问题、非法参数顺序、可写 info 和 padding 都纳入专门测试。
- 未选三角可被用作 workspace，验收比较所选三角；有效失败前缀和 info 仍须满足合同。
- 默认路径、运行开关和支持范围随候选源码固定。opt-in 路径须有其自身回归与资源证据，不能继承默认路径结果。
- INF/NAN 按任务书引用的精度标准构造并记录。发布包与任务书存在的解释差异保留原始结果并提交复核，不通过删除用例或替换基线制造通过。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 设计及验证要求 | 来源 |
| --- | --- | --- |
| 逐元素精度 | FP64 CPU golden；rtol=2^-10、atol=2^-16；matched_ratio>=0.99 且 max_abs_error 满足任务书 `1e-2 or 32*ULP` 规则 | 官方任务书3.2.1；[生态精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 残差复核 | Spotrf/Spotrs按对应1范数公式，阈值 `max(5*ratio_cpu,3*ratio_cpu_mean)`；单矩阵按本接口case均值，批量按当前case槽位均值且逐槽达标；Spotri阈值 `max(5*ratio_cpu,0.1)` | 官方任务书3.2.2及发布包检查器 |
| 功能/状态 | LOWER/UPPER、padding、非SPD/奇异k、非法参数-i、成功0、空问题、真实设备指针数组和非默认stream | 官方任务书2.1、2.4、3.5 |
| 确定性 | 同输入、同stream串行重复执行，输出及info bit-wise一致 | 官方任务书2.1、3.2 |
| 性能 | 每个必测case `T_GPU_avg/T_NPU_kernel_avg>=0.35`；求解/求逆排除前置分解 | 官方任务书3.3 |
| 内存 | 任务书无内存性能阈值；仍提供要求的材料、资源账本及设备安全检查记录 | 官方任务书3.4、4 |

完整性能基线为 Spotrf143、Spotrs174、Spotri143、两批量各154原始行，共768行。批量各3个重复shape在 canonical 中去重，仍逐条关联原始GPU金标；单矩阵 canonical 各增加2个无GPU金标精度用例，在性能材料中如实标注。不能只跑任务书P-01..P-12，也不能把精度用例数当性能分母。

msprof 采集时按接口/case隔离，只归因目标接口的全部 kernel，保留调用次数、每次调用的kernel分组及原始单位，按一次接口调用合计目标 kernel 耗时再求平均；不直接相加按 kernel 名分组的均值。首次编译、造数、H2D/D2H 及前置分解不计为目标 kernel 性能。workspace 与设备指针数组在正式采样中复用。保存30次正式采样的原始值、算术平均和中位数，正式门禁使用任务书的平均值；补充采集文件的中位数口径分别记录供评审核对。

验证顺序为：冻结源码及五包身份 → 新环境干净构建和导出符号/实际加载库核对 → 五包完整精度/info → 原生功能及接口边界 → 非默认stream/确定性/内存与同步 → 全量逐例性能 → 模板报告和复现材料。任何修复建立新源码身份并复测受影响链路；失败、缺测、skip、超时及未知结果均保留，不宣称为通过。

## 兼容性分析

新增七个函数沿 ops-solver 现有公开 C API 和 handle/stream 扩展。实现涉及公开头文件、内部 handle workspace 生命周期及构建收集，因此完整库链接、导出符号和原有算子回归属于验证范围，不能仅以五接口单测说明旧接口兼容。

新增代码目录为 `src/spotrf`、`src/spotrs`、`src/spotri`、`src/spotrf_batched`、`src/spotrs_batched`，对应测试与中文 API 文档分别放入 `test/` 和 `docs/zh/`；同步维护两个公开头文件、README 和 `docs/api_list.md`。布局遵循[ops-solver贡献规范](https://gitcode.com/cann/ops-solver/blob/master/CONTRIBUTING.md)，不套用 aclnn/PyTorch 的双阶段工程模板。

设计合入后随最终实现提供私仓地址、分支、目录与固定源码身份，并为 Ascend-CANN 开通 Developer 访问。`task_submission` 包含设计合入截图、测试步骤、精度/性能/内存模板报告及原始日志，材料与受测版本一致。平台验收、后台测试及最终产品代码合入分别留存回执。

## AI 辅助使用声明

本设计文档使用 OpenAI Codex 辅助整理与源码核对；候选实现开发使用 Kimi Code 和 Codex 辅助。AI 输出不作为精度、性能或验收通过证据。贡献者 `bluerain117` 对所提交内容、测试证据和后续评审答复承担责任；实际提交的 AI 声明须与此使用范围一致，不声明未发生的人工审阅。

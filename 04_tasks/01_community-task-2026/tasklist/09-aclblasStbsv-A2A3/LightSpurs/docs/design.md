# aclblasStbsv 算子设计文档（Atlas A2/A3，arch22）

| 文档项 | 内容 |
|---|---|
| 文档版本 | V1.0，2026-09-29 UTC |
| 提交者 | LightSpurs |
| 社区任务 | [9月社区任务-aclblasStbsv算子开发(A2/A3)](https://www.hiascend.com/activities/task-center/details/29c59d7ef9fa4aa8ad26b82969c5e2b6?menu=guide) |
| 对应实现 | bounds5b 冻结候选；分支 feat/stbsv-arch22 |
| 工程基线 | ops-blas，fe54d86f00a4d449f55da1e8d144b898953d97c5 |
| 开发方式 | Ascend C，kernel 直调，句柄式 BLAS 接口 |
| 适配范围 | Atlas A2/A3 系列产品，arch22；本次实测 A2 / Ascend 910B4-1 |
| 软件环境 | CANN 9.1.0，真实 Netlib BLAS，C++ GTest + CSV |
| 文档状态 | 设计内容与冻结代码、已有测试证据核对完成；提交社区评审 |

本文按任务指定设计模板的“需求背景、需求分析、详细设计、可维可测分析”组织。本文提交到 cann-ops-competitions 的本任务目录；文中实现和测试路径均相对于独立的 ops-blas 工程，不表示这些代码或日志已包含在本设计 PR 中。原始自验证记录随验收交付件提供，正文中的实测汇总不等于社区验收结论。

# 需求背景（required）

## 需求来源

来源为《9月社区任务：aclblasStbsv 算子开发（A2/A3）》及配套测试材料。任务要求在 Atlas A2/A3 上以 Ascend C 或 CATLASS 实现单精度实数三角带求解，接口语义对齐 cuBLAS cublasStbsv，并完成设计、开发、精度验证、性能验证和交付。当前实现采用 Ascend C，不依赖 CATLASS。

任务代码目录为 blas/tbsv/arch22/，测试目录为 test/tbsv/stbsv/arch22/，公共接口声明位于 include/cann_ops_blas.h。设计文档、验证报告和提交材料按任务流程交付；社区 PR 评审与合入属于后续外部流程。

## 背景介绍

### 算子功能与工程现状

三角带状矩阵只保存主对角及其一侧的 k 条对角带，无须存储完整 n×n 矩阵。算子求解 op(A)x=b，调用前 b 存在 x 的逻辑元素中，调用后原地覆盖为解，A 保持只读。该能力用于需要保留带状结构的线性代数计算。

工程已经定义 aclblasStbsv 公共 API、BLAS handle/stream/workspace 管理以及 CSV 驱动测试框架。本任务新增 arch22 实现和对应测试，不改变公共函数签名，也不改写其他架构的实现。

### 主要实现难点

1. 每个未知量依赖前面已经求得的结果，整条前代/回代链不能直接按行均分到多个核。
2. 非转置时相邻带系数连续，转置时相邻系数跨列分布，访存步长为 lda−1 个 float，搬运效率差异明显。
3. 支持负 incx、较大 lda、k≥n、非对齐首地址以及 UNIT 对角不读，地址和边界处理不能只覆盖紧凑矩阵。
4. 病态三角系统会放大浮点累减误差；普通分离向量乘减与 Netlib 单次舍入乘减可能产生不同误差积累。
5. 可选转置整理必须遵守既有 handle 工作区与异步语义，不能在算子内部扩容工作区或通过 Host 同步换取正确性。

# 需求分析（required）

## 需求描述

实现 FP32、单矩阵、单右端向量的三角带状方程求解。支持 UPPER/LOWER、N/T/C、UNIT/NON_UNIT 的正交组合，支持正负步长和带存储 padding。T 与 C 在实数数据下等价。n=0 为校验通过后的合法 no-op；k=0 为对角系统。

本算子没有广播、批量接口或多右端矩阵接口，不承诺逐位确定性。n、k、lda、incx 为运行时输入，由 Host 组织参数直接启动设备 kernel，不使用图模式动态 shape 编译流程。

## 需求拆解

| 编号 | 需求 | 设计措施 | 验证方法 |
|---|---|---|---|
| R01 | FP32 三角带求解，x 原地输出 | 单 AIV 持有依赖链，逐主元更新带内剩余分量 | Netlib cblas_stbsv 全元素对比 |
| R02 | UPPER/LOWER、N/T/C、两种对角语义 | 编译期模板分派；C 走 T 路径；UNIT 不读取对角 | 正交组合、原对角填 NaN/Inf |
| R03 | 任意合法 lda、正负 incx | int64 元素偏移，逻辑向量与物理布局分离 | padding、负步长、地址偏移和保护区 |
| R04 | 空维、非法参数、空指针 | 先校验 handle 和参数，再处理 no-op | 12 条 INVALID_VALUE 用例及 NullHandle |
| R05 | 大 n、大带宽与跨块尾部 | x 常驻/分块两种布局，512 元素 tile | n=24576 边界、n=32768/32769、多 tile |
| R06 | 工作区受限时仍能运行 | 可选整理；容量/对齐不满足则直接求解 | 64 字节工作区、非128字节对齐工作区 |
| R07 | 句柄绑定流、异步执行 | 同一 stream 顺序启动整理与求解，不在 API 内同步 | 连续两次调用后统一同步 |
| R08 | 混合容差与特殊值 | 补偿向量更新、异常逐元素修复、NaN 传播处理 | rtol/atol、匹配率、绝对误差、特殊值单独校验 |
| R09 | 性能目标 | 缓存带矩阵、可选多核转置整理、常驻 x 与通用有界路径 | 全部200条性能用例，5次预热+20次有效采样 |
| R10 | 可复现交付 | 固定测试定义、原始记录、源码/二进制哈希、离线核查脚本 | CSV/XML/日志/XLSX 一致性及 SHA-256 |

## 接口定义与参数约定

~~~cpp
aclblasStatus_t aclblasStbsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n, int k, const float* A, int lda, float* x, int incx);
~~~

任务书描述的 FILL_MODE_UPPER/LOWER、DIAG_UNIT/NON_UNIT，在本仓公共头中的实际枚举名为 ACLBLAS_UPPER/LOWER、ACLBLAS_UNIT/NON_UNIT，测试和实现统一使用仓库定义。

| 参数 | 位置/方向 | 语义与实际检查 |
|---|---|---|
| handle | Host / 输入 | 已创建的有效句柄；nullptr 优先返回 HANDLE_IS_NULLPTR |
| uplo | Host / 输入 | ACLBLAS_UPPER=121、ACLBLAS_LOWER=122 |
| trans | Host / 输入 | ACLBLAS_OP_N、ACLBLAS_OP_T、ACLBLAS_OP_C |
| diag | Host / 输入 | ACLBLAS_NON_UNIT=131、ACLBLAS_UNIT=132 |
| n | Host / 输入 | n≥0；参数合法后 n=0 不启动 kernel |
| k | Host / 输入 | k≥0；允许 k≥n，有效非对角范围按 n 裁剪 |
| A | Device / 只读 | FP32 列主序带数组，分配覆盖 lda×n 个元素；n>0 时非空 |
| lda | Host / 输入 | lda>k；在 k≥0 时等价于 lda≥max(1,k+1)，同时避免 k+1 有符号溢出 |
| x | Device / 输入输出 | n>0 时覆盖 1+(n−1)×abs(incx) 个元素；非空，原位写解 |
| incx | Host / 输入 | 非0；实现按配套CSV和现有golden约定拒绝 INT_MIN |

handle 为 nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR；其余上述非法参数或非法枚举返回 ACLBLAS_STATUS_INVALID_VALUE。参数通过后，n=0 或 k=0 且 UNIT 返回 ACLBLAS_STATUS_SUCCESS；其余情况提交 kernel 后返回 SUCCESS。异步设备执行错误由调用方按 ACL 流同步/运行时错误机制检查，本接口不通过额外同步提前确认执行完成。

A、x 和活动 workspace 的有效范围及设备可访问性由调用方保证。A 与 x 不允许重叠；可选整理写入的 workspace 也不应与 A/x 重叠。当前 Host 只检查相关空指针、数值、容量与对齐，不探测设备分配边界或任意指针重叠。无单独奇异矩阵检测或 info 输出；零对角、Inf/NaN 按实际浮点求解传播，不映射为参数非法。

# 详细设计（required）

## 算子分析

### 数学公式

记 B=op(A)。B 为下三角时按 j=0…n−1 前代，为上三角时按 j=n−1…0 回代。下三角的数学形式为：

~~~text
x[j] = (b[j] − Σ B[j,i]·x[i]) / d[j],  i=max(0,j−k)…j−1
d[j] = 1（UNIT），或 B[j,j]（NON_UNIT）
~~~

上三角将求和区间改为 i=j+1…min(n−1,j+k)，并逆序求解。实现采用逐主元更新尚未求解的 RHS 的形式；对每个目标元素，主元顺序按依赖方向保持一致，不把整条依赖链改为无序并行规约。浮点结果以混合容差验收，不承诺与参考实现逐位一致。

| uplo | trans | op(A)三角方向 | 主元顺序 |
|---|---|---|---|
| UPPER | N | 上三角 | n−1 → 0 |
| LOWER | N | 下三角 | 0 → n−1 |
| UPPER | T/C | 下三角 | 0 → n−1 |
| LOWER | T/C | 上三角 | n−1 → 0 |

### 带状存储与向量寻址

使用从0开始的下标，所有以下元素地址均采用64位计算：

~~~text
UPPER: A(i,j) = A[j·lda + k + i − j], max(0,j−k) ≤ i ≤ j
LOWER: A(i,j) = A[j·lda + i − j],     j ≤ i ≤ min(n−1,j+k)

incx > 0: x_logical[i] = x[i·incx]
incx < 0: x_logical[i] = x[(n−1−i)·(−int64(incx))]
~~~

UPPER 主对角在带行 k，LOWER 在带行0。即使 k≥n，直接路径仍以原始 k、lda 定位输入，不擅自改变存储约定。T/C 的更新系数使用 A(j,i)，N 使用 A(i,j)。UNIT 时直接采用逻辑对角1，不读取原始对角值。缓存可能搬入分配范围内未参与运算的带存储 padding，但不把它们作为有效系数；不越过声明的 A 分配，也不修改 A。

### 支持数据类型

- 数据类型：A、x 与算术均为 FP32；偏移采用 int64/uint64；维度属性来自 int 接口。

### 支持形状

- 形状：A 为 lda×n 带数组，x 为带正负步长的逻辑长度 n 向量；支持 k≥n、lda padding。
- 当前已验证的正 n 最大为32769，正形状用例 k 最大2047；这是测试覆盖范围，不作为代码的人为形状上限。可分配内存与平台资源仍限制实际可运行规模。

### 计算与访存复杂度

- 令 m=min(k,n−1)，求解工作量 O(n·m+n)。可选整理的处理量和工作区随 n×roundUp(m+1,32) 增长。
- x 常驻时，其主要 GM 搬运为一次逻辑输入/输出；非连续步长以标量地址打包/散布。x 非常驻时，带内更新按最多512个元素分块，向量 GM 访问可能达到 O(n·m)。

## 算子实现

### 实现方案

~~~mermaid
flowchart TD
    A[aclblasStbsv] --> B{handle和参数合法?}
    B -- 否 --> C[返回对应状态码]
    B -- 是 --> D{n=0 或 k=0且UNIT?}
    D -- 是 --> E[返回SUCCESS 无kernel]
    D -- 否 --> F[组织StbsvTilingData]
    F --> G{转置整理条件满足?}
    G -- 否 --> H[单AIV直接求解]
    G -- 是 --> I[多个AIV整理转置带矩阵]
    I --> J[同一stream上的单AIV求解]
    H --> K[原位写回x]
    J --> K
~~~

该图描述设备任务的数据依赖。Host 在提交任务后返回，不等待图中的设备写回完成。

| 文件/函数 | 职责 |
|---|---|
| include/cann_ops_blas.h | 现有公共API声明 |
| stbsv_host.cpp / ValidateStbsvParams | 参数与枚举合法性检查 |
| stbsv_host.cpp / LaunchStbsvKernel | 填充Tiling，选择可选工作区与整理核数 |
| stbsv_tiling_data.h | Host/Device按值传递的参数结构 |
| stbsv_kernel.h / stbsv_kernel_do | Host启动接口与模板分派 |
| stbsv_kernel.cpp / StbsvKernel | 常驻/分块求解、系数缓存、数值更新与修复 |
| stbsv_kernel.cpp / StbsvPack | 多核转置带存储整理 |

以上同名源码文件位于 blas/tbsv/arch22/。

### 3.2.1 Host侧设计

#### 参数校验、早退出和异步约束

顺序固定为 handle → 枚举与整数范围 → n>0时的A/x指针 → no-op → Tiling与启动。因而 n=0 不绕过非法枚举、lda、incx 等校验。k=0且UNIT的早退出同样发生在 n>0 指针检查之后。

Host 不读取 A/x 内容，不申请或扩容 GM，不调用流同步。默认32MiB workspace 属于库的 handle 生命周期，算子仅复用当前已存在的活动 workspace；用户配置的工作区不足时不报额外分配失败，而是采用直接求解路径。不同流并发调用应由调用方保证 handle/workspace 不发生无序共享。

#### Tiling结构

本实现采用直接传参结构，不注册图算子TilingData，不使用独立整数 tilingKey。结构字段如下，默认 workspace、packedLda、packCores 为0：

| 字段 | 类型 | 作用 |
|---|---|---|
| a / x | uint64_t | A与x的Device地址 |
| n / k / lda | uint32_t | 参数校验后的维度和带存储步长 |
| incx | int32_t | 保留向量有符号步长 |
| uplo / trans / diag | uint32_t | 原始语义枚举 |
| workspace | uint64_t | 非0时启用转置整理的Device地址 |
| packedLda | uint32_t | 整理后每列的float元素数，按32个float对齐 |
| packCores | uint32_t | 整理任务的AIV核数 |

#### 转置整理选择与分核策略

只有同时满足下列条件时设置非0 workspace：

1. trans 为 T/C，n≥128 且 k≥32。
2. (uint64(lda−2)×4)≤UINT32_MAX，DMA间隔字段可表达。
3. 当前 workspace 非空且起始地址按128字节对齐。
4. packedLda=roundUp(min(k,n−1)+1,32)，并满足 packedLda≤workspace_size/4/n。用除法检查容量，避免在比较前引入乘法溢出。
5. GetAivCoreCount()返回正核数；packCores=min(核数,n)。

每个整理核处理 ceil(n/packCores) 个连续列，最后一段以 n 截断；末尾可能存在无任务核。packedLda是32个float的倍数，核间列区域不共享128字节写入行。求解 kernel 始终使用1个AIV核，不将“查询全部核”误用于有依赖的求解链。

任一条件不满足，直接启动原始带存储求解。代码中的176KiB UB预算是固定arch22设计值，并未在Host动态查询UB容量；动态查询仅用于整理核数。

#### 模板分派规则

直接求解使用 StbsvKernel<UPPER,TRANSPOSE,UNIT>，由两个三角方向、两个实际转置状态和两种对角状态构成8种组合；OP_C与OP_T共用TRANSPOSE=true。

整理路径先启动 StbsvPack<UPPER,UNIT>，再将 a 指向 workspace、k裁剪为min(k,n−1)、lda改为packedLda、三角方向翻转，以TRANSPOSE=false的求解模板处理已经转置的带矩阵。**Tiling中的原始trans字段仍保留**，使零RHS跳过规则继续按照原调用的N/T/C语义判断。

### 3.2.2 Kernel侧设计

#### Init / Process / CopyIn / Compute / CopyOut

Init绑定GM张量，根据 n 确定 x 常驻模式，分配UB缓冲、初始化索引和必要的FMA工作区，并选择带矩阵缓存。常驻时先加载完整逻辑x。Process按前代/回代顺序调用SolveColumn；更新区间按最多512元素分块。末尾将常驻结果写回x，清理/失效x的数据缓存并恢复向量mask状态。

~~~text
for j in dependency_order:
    pivot = logical_x[j]
    if k>0 and pivot is NaN:
        按依赖方向将尚未求解的分量置NaN，结束
    确保本主元需要的矩阵列已缓存（若启用缓存）
    if original_trans == N and pivot == 0:
        continue
    if NON_UNIT:
        pivot = pivot / effective_A[j,j]
        logical_x[j] = pivot
    for remaining band range in tiles of at most 512:
        获取x片段和系数
        按长度、数值条件选择标量/向量补偿更新
        logical_x[i] -= pivot * effective_A[i,j]
~~~

k>0时NaN主元经下一条带依赖继续传播，即使系数为0也有NaN×0，因此可按同一依赖方向填充剩余解；k=0不能使用此捷径。原始N模式在当前RHS为0时跳过整列，T/C不跳过，避免改变特殊值语义。该规则在转置整理后仍成立。

#### x搬运、非对齐前缀与尾块

- n≤24576：x按8个float对齐分配UB，逻辑分量常驻。incx=1用DataCopyPad精确搬入/搬出，任意其他步长（包括−1）用64位地址标量打包和散布。
- n>24576：x工作缓冲固定512个float；按块加载更新范围，再通过逻辑偏移写回。未更新的主元从GM标量访问。
- 常驻向量主体按8个float边界处理；前缀最多7个元素走标量路径。FAST路径可将不重叠的前缀计算与向量主体执行交错，仍在下一主元使用前完成所需同步。
- DataCopyPad按实际count搬运；向量mask限制有效元素。UB允许对齐补齐，但写回GM时不覆盖x首尾保护区或步长间隙。

#### 系数缓存与直接转置读取

UNIT缓存每列k个非对角带元素：UPPER从带行0开始，LOWER从带行1开始；NON_UNIT缓存k+1个带元素。UNIT不因批量搬运而读取原始对角。

默认缓存列步长为roundUp(cacheRows,8)。NON_UNIT且求解模板非转置、lda−cacheRows≤31时，允许紧凑连续缓存，cacheStride=lda。启用缓存还要求至少容纳一列、GM列间字节间隔可由32位DMA字段表示。每批缓存最多4095列；普通UNIT非转置路径限制每批最多64列，有界路径可使用更大的缓存批次。

直接转置模板需要缓存覆盖当前带窗口：cacheColumns≥min(k+1,n)，前移量为cacheColumns−窗口宽度+1。不能满足时停用该缓存。未缓存的N模式搬运连续系数；T/C模式按32字节UB块承接跨列单元素DMA，再Gather为连续系数。若DMA间隔超32位字段范围，则用64位标量寻址回退。

#### 转置整理Kernel

每个核先将自己负责的packed列清零，再按有效三角带范围读取原A。UNIT读取范围主动避开对角；COPY阶段通过跨列单元素DataCopyPad和Gather，生成连续packed列。

原UPPER转置后形成LOWER，原LOWER转置后形成UPPER。有效带宽为min(k,n−1)，packedLda对齐保留的尾部只作为安全padding；solver按新三角方向读取。整理与求解在同一stream中依次提交，依赖流内顺序，不需要Host同步或跨核求解屏障。

### 3.2.3 UB、GM与工作区预算

所有大小以字节计；roundUp按向上整数取整。S=512，F=544，M=64，P为x缓冲的float元素数（常驻roundUp(n,8)，否则512）。

| 缓冲区 | 分配大小 | 使用条件 |
|---|---:|---|
| xBuffer | 4P | 所有求解路径 |
| aBuffer | 4S=2048 | 当前更新片段系数 |
| fmaBuffer及双mask | 9×4F+2M=19712 | k≥40且n>16（UNIT）或n>32（NON_UNIT） |
| indexBuffer | 4S=2048 | 直接转置模板或启用矩阵缓存 |
| gatherBuffer | 32S=16384 | 直接转置且无矩阵缓存 |
| cacheBuffer | 4C | 缓存条件满足 |
| boundBuffer | 与有界模式cacheBuffer同量 | UNIT、求解模板非转置、常驻、40≤k≤128且n>32 |

缓存的基础元素预算为：

~~~text
UB_BUDGET_BYTES = 176 × 1024 = 180224
C_base = floor((180224 − 4P − (8S + 36F) − 2M) / 4)
普通缓存：C = C_base
有界缓存：C = floor(C_base/128) × 64，分别分配cacheBuffer与boundBuffer
~~~

预算保守预留了系数、索引、FMA和mask；未使用的缓冲不重新无限扩张。无缓存的转置路径另分配gatherBuffer，但不同时分配cacheBuffer。以P=24576为例，C_base=14528；有界模式C=7232，x+系数+索引+FMA+cache+bound总计179968字节，小于176KiB预算。

转置整理kernel使用 scatter 16384字节、output/zero/offset各2048字节，共22528字节（22KiB）UB。FMA_STRIDE=544相对512增加32个float，以错开工作数组的访问间距。

GM方面，原始A占4×lda×n字节，x逻辑物理跨度占4×(1+(n−1)×abs(incx))字节（n>0）。可选整理使用4×n×packedLda字节的既有workspace，默认32MiB由库创建handle时分配；算子新增GM分配次数为0。测试报告的A/x申请量还包括测试保护区，不能与算子输入净大小或峰值HBM混淆。

### 3.2.4 数值精度与优化路径

#### 标量与补偿向量更新

逐次更新目标为 old−pivot×coefficient。短片段使用允许编译器融合乘减的标量表达式，避免常规分离Mul/Sub放大误差。k≥40且片段长度至少16（UNIT）或32（NON_UNIT）时使用补偿向量路径；FMA缓冲初始化条件与可能触发该路径的n/k边界匹配。

VectorFma将pivot与系数按位拆分成高、低部分（掩码0xfffff000），计算普通乘积、乘积残差、减法残差，再补偿回写结果。相邻向量阶段使用PIPE_V屏障保证依赖。补偿用于改善混合容差表现，不宣称对所有FP32输入与硬件FMA逐位一致。

普通路径先保存旧值。RepairExceptional对乘积绝对值小于2^-60、结果非有限或达到可疑极值等元素生成无效mask，仅对这些元素重做标量表达式。非有限主元使用相应向量更新/NaN传播处理，最终验证NaN配对与Inf符号，不把特殊值差异忽略。

#### 有界快速路径及回退条件

该路径只在UNIT、求解模板非转置、x常驻、40≤k≤128、n>32且缓存可用时启用；**包括已经整理成非转置求解形式的原T/C调用**。它不依据测试名称、seed或固定输入内容分派。

初始化扫描RHS，缓存更新时扫描系数最大绝对值和最小非零绝对值。MagnitudeBound把绝对FP32位型视作非负整数数值再转换为float进行ReduceMax，使NaN/Inf位型不会被普通浮点max忽略。该整数尺度的舍入误差上界为64，代码加128形成保守包络，并以位阈值拒绝接近非有限范围的情况。最大幅度乘1.000001、最小非零系数乘0.999999作为裕量。

每列更新维护 valueBound=(valueBound+abs(pivot)×coefficientMax)×1.000001。只有初始RHS与当前缓存均有限，并且以下条件全部成立，才省略旧值复制与逐元素异常mask/修复：

~~~text
abs(pivot) × coefficientMin > 2^-59
valueBound < 2^110
abs(pivot) < 2^100
coefficientMax < 2^100
~~~

零系数从正最小值统计中排除；对应乘积为精确零。任何条件失败都使用同一补偿算术加完整修复路径。有界路径节省检查与复制，不改变验收容差、不改变解方程的定义。

### 3.2.5 同步、缓存一致性与状态恢复

| 依赖边界 | 主要同步措施 |
|---|---|
| 标量发布/释放缓冲 → DMA搬入 | S_MTE2，必要时处理x数据缓存 |
| DMA完成 → 标量或向量读取 | MTE2_S / MTE2_V |
| 标量写UB → 向量读取 | S_V |
| 相邻向量算术/掩码依赖 | PipeBarrier<PIPE_V> |
| 向量结果 → 标量主元或修复 | V_S |
| 标量/向量结果 → DMA写出 | S_MTE3 / V_MTE3 |
| DMA写出结束/整理缓冲复用 | MTE3_S / MTE3_MTE2 |
| 求解完成 | x数据缓存清理/失效；SetMaskNorm、ResetMask |

Fence使用同一EVENT_ID0做成对SetFlag/WaitFlag，并在每个局部依赖边界消费，不将它当作未完成事件队列。当前设计没有声明通用双缓冲或DMA全流水覆盖；重点是依赖正确、缓存复用以及不重叠前缀与向量主体的交错。

## 支持硬件

| 产品范围 | 工程支持 | 当前验证状态 |
|---|---|---|
| Atlas A2训练/推理系列（含800I/800T A2） | arch22，ascend910b*映射dav-2201 | A2/910B4-1实测完成，CANN 9.1.0 |
| Atlas A3训练/推理系列（含800I A3） | arch22，ascend910_93*映射dav-2201 | 代码适配范围；本版不声明A3实机验证完成 |

任务3.1写明自验证只需一种款型，910B3/A3属于精度建议、910B3属于性能建议。本次自验证覆盖A2；同时保留任务3.3列出910B3基准设备及7.4列出A2/A3验收的文字差异，供提交时说明，不将A2/910B4误写成另一平台。

## 算子约束限制

1. 仅FP32实数、单矩阵单向量；不提供广播、批量、复数、FP16/BF16或多RHS接口。
2. A按列主序带存储，x仅通过incx表达步长；不支持超出lda/incx语义的任意视图。
3. UNIT不读取对角；输入/输出及工作区的有效分配、生命周期与无非法重叠由调用方保证。
4. 无奇异性检测；浮点溢出、NaN/Inf不直接转换为参数错误。
5. 工作区容量不足、不对齐或DMA步长不可表达会影响性能分派，不改变数学求解功能。
6. API异步，不保证返回时x已经可由Host读取；调用方须在读回前同步。
7. incx=INT_MIN按原CSV/golden拒绝，与任务仅写incx≠0的文字差异在交付中明确披露。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 实施规则 | 来源与说明 |
|---|---|---|
| FP32逐元素误差 | abs(actual−golden)≤1/8192+(1/8192)×abs(golden) | 任务3.2混合容差 |
| 全局匹配率 | matched_ratio≥0.99 | 任务3.2 |
| 最大绝对误差 | ≤max(0.01,32×ULP(max abs(finite golden))) | 对任务“1e-2或32×ULP”的显式实现解释；未随性能调优放宽 |
| 特殊值 | NaN须配对；Inf须符号一致；特殊值不匹配数必须为0 | 测试VerifyResult独立拦截 |
| 性能采样 | 每条5次warmup、20次有效调用取平均 | 满足任务4/7有效采样>10次的要求 |
| 全量性能限值 | 平均NPU_us≤GPU_ms×1000/0.8；前五条再取与任务表上限的较小值 | 配套GPU基线和任务3.3，200条均检查 |
| 内存数值门槛 | 任务3.4“不涉及”；仍记录申请量与插桩状态 | 不把申请字节冒充HBM峰值 |

任务3.2另一段关于约1ULP/可酌情2ULP的文字与表格32ULP表述并不完全一致。本实现固定采用上述混合容差与最大误差公式，随报告提交，不把它描述成已获社区确认的唯一解释。

## 测试数据与用例覆盖

沿用原始1200条CSV（其中1000条非性能定义、200条性能定义），追加139条，另有1条NullHandle，共1340个GTest测试。CSV中1327条期望SUCCESS、12条期望INVALID_VALUE；有正n的成功输入1323条。性能用例的数值正确性也先行检查。

随机填充RANDOM_NORM_5_5按seed奇偶分派：偶数为[-5,5]均匀分布，奇数为真实正态分布，均值在[-5,5]、标准差在[0.1,2]抽取。适用正形状随机用例中，A为642均匀/642正态，RHS为643均匀/643正态。NON_UNIT对角按配套测试规则依符号加减5；NPU与golden使用同一修改后的A和RHS。UNIT对角设置NaN/Inf，检验实现确实忽略原对角。

| 覆盖组 | 场景与检查 |
|---|---|
| 基础/正交组合 | UPPER/LOWER × N/T/C × UNIT/NON_UNIT，n=1及小形状 |
| shape与布局 | 幂次及邻域、质数、lda padding、正负incx、带宽超过阶数 |
| 运行边界 | n=0、k=0、空指针、非法枚举、负维度、零步长、INT_MIN与整数溢出边界 |
| 大规模/跨tile | 带宽跨512、常驻上限24576、n=32768/32769 |
| 特殊数值 | NaN/Inf、零RHS、零非对角、孤立NaN、较大步长 |
| 工作区/异步 | 64字节workspace回退、非128字节对齐回退、整理后连续调用、同流两次调用后统一同步 |
| FMA与有界路径 | 48条UNIT短n=17/24/31/32、k=40；12条n=193、k=127/128/129 |
| 内存完整性 | A全分配按字节不变；x首尾保护区与步长间隙哨兵不变；组合/独立插桩分别判定 |

NullHandle和非法参数按接口要求直接检查返回码，不把CPU golden封装在这些负向场景的不同返回行为当作算子标准。合法非空数值用例调用真实Netlib cblas_stbsv，未以被测NPU输出生成golden。

## 性能测试设计与实测结果

每例执行25次算子调用且每次重新加载原始RHS；前5次预热丢弃，其余20次取算术平均。实际使用msprof设备task_time，将原始kernel记录按GTest顺序与专用kernel名称对应；转置整理场景将pack和solve的设备时间相加。统计不包括Host到Device输入准备时间，不把设备时间称为端到端API延迟，不只报solve部分。

本容器msprof op FFTS通道失败，原始失败日志保存在task_submission/msprof_failure.log。当前使用设备task_time是已披露的方法差异，不声称已有msprof op replay结果。每轮5000次算子调用，对应6675次kernel启动；日志和解析脚本逐项检查数量与映射。

最新A2复测：

| case | n | k | uplo/trans/diag | 平均us | 任务表上限us | 实际判定上限us |
|---|---:|---:|---|---:|---:|---:|
| TC_PF_1001 | 256 | 8 | UPPER/N/NON_UNIT | 77.89815 | 198.0 | 197.9725 |
| TC_PF_1002 | 512 | 32 | LOWER/N/NON_UNIT | 347.56395 | 379.8 | 379.8 |
| TC_PF_1003 | 1024 | 16 | UPPER/T/UNIT | 95.8958 | 659.5 | 659.5 |
| TC_PF_1004 | 2048 | 64 | LOWER/T/NON_UNIT | 753.6673 | 1725.0 | 1725.0 |
| TC_PF_1005 | 4096 | 128 | UPPER/N/UNIT | 56.448 | 2365.0 | 2365.0 |

| 同硬件、同冻结构建轮次 | 性能通过数 | PF1197平均us | PF1197上限us |
|---|---:|---:|---:|
| 首轮 | 200/200 | 104.617 | 108.415 |
| 复测1 | 200/200 | 104.687 | 108.415 |
| 复测2 | 200/200 | 104.673 | 108.415 |
| 最新A2复测 | 200/200 | 104.715 | 108.415 |

四轮来自同一A2/910B4容器，不是四种硬件。原始CSV、GPU基线、计算容差和冻结构建在复测中不变；全部200条均达标，不删失败项、不选择最快一次作为平均结果。

## 内存验证与已知风险

最新正确性执行1340/1340通过。内存插桩分别在53个组合场景和相同53个独立进程场景运行：独立检测53/53通过、74次干净kernel检查；组合53/53功能执行通过，但有456条工具告警，组合工具状态保留FAIL。不能依据工具进程退出码0将含告警的执行标为通过。

另有不调用Stbsv的独立写零复现：在合法32MiB分配内，单次启动0告警，同一分配连续启动521告警，中途重新分配0告警，输出首尾均为0。此对照支持检测器跨启动状态异常的解释，但尚未获厂商确认。原始诊断、最小程序及对照在task_submission/memory_tool_repro/，不将该解释写成已经排除全部内存问题的证明。

| 风险/差异 | 当前措施及状态 |
|---|---|
| A3未实测 | 说明arch22适配和实际A2款型；按任务3.1完成一种款型自验 |
| task_time与op replay不同 | 保留FFTS失败日志、原始计时记录和重建脚本，提交时说明 |
| 组合插桩告警 | 保留FAIL及完整诊断；独立检测、无Stbsv复现与控制组作为辅助证据 |
| INT_MIN与ULP文字差异 | 明示实现规则、配套用例依据和公式，未隐藏或自动放宽 |
| 性能余量受环境影响 | 四轮全量复测；PF1197最接近上限，保留全部有效样本 |
| 报告证据形式 | 采用日志、CSV、XML、XLSX缓存值及哈希；当前未提供任务交付字段中的截图，该差异需随验收说明 |

## 可维护性

代码保持4个实现文件，不引入与算子无关的设备分配或自定义执行框架。Host校验/分派、Tiling定义、求解与整理各自有清晰入口；任何调优必须保持同一合法性、UNIT不读对角、特殊值、异步和保护区约束。修改FMA门槛、tile、缓存或向量mask时，应优先重跑对应补充边界，再覆盖原始全量用例。

## 兼容性分析

公共函数签名、枚举值、handle布局、其他架构和外部库接口不变。原始测试参数保留，测试参数解析对k=INT_MAX避免默认k+1溢出；A2/A3走工程已有arch22选择逻辑。CANN 9.1.0以外的软件组合、其他架构的运行结果不在本版证据覆盖范围内。

## 可复现性

构建命令、Netlib路径及环境变量详见test/tbsv/stbsv/arch22/README.md和task_submission/1 自验证步骤说明.md。测试CSV路径由编译期源文件位置确定，移动源码后须重新编译。无需NPU的核对入口：

~~~bash
python3 task_submission/verify_evidence.py
python3 task_submission/verify_report_files.py
~~~

前者从原始task_time重建四轮结果，核查精度与内存覆盖、源码/二进制记录及报告哈希；后者逐项比较XLSX单元格缓存与当前A2 CSV。二者不代替真实硬件执行，也不修改结果。

## 交付、追溯与评审状态

| 材料 | 仓库路径/内容 |
|---|---|
| 设计正文 | 本PR的docs/design.md；ops-blas工程内保留DESIGN.md及交付副本，社区版本补充任务链接和提交范围说明 |
| 实现与接口 | blas/tbsv/arch22/；include/cann_ops_blas.h已有声明 |
| 测试代码与CSV | test/tbsv/stbsv/arch22/ |
| 最新精度/性能/内存原始记录 | task_submission/a2_evidence/下validation、profile、memory-group、memory-isolated |
| 三份报告与对应日志 | task_submission/2.1～4.2命名文件 |
| 需求证据索引 | task_submission/ACCEPTANCE.md |
| 冻结哈希与新复测证明 | task_submission/frozen_hashes.json、a2_evidence/frozen_verification.json |
| 完整大型原始归档 | a2_evidence/persistent_archive.json给出容器持久路径、大小与哈希；raw_manifest.json逐文件索引 |
| 设计与代码PR文案 | task_submission/submission/design_pr.md、code_pr.md |

冻结kernel SHA-256为08e4155318d7416c9e78a7db148b7ee1a78bf1085b862317a6632fdd51b5d64c，CSV SHA-256为7d65352e8d00c6ab558bcf31672ca8b8b4d71d17a9e457d9fc5eaae9fff59f35。设计文档修订不改变这两项或被测二进制。

本版已完成可供评审的设计正文与实测证据说明。本PR申请设计评审。代码验收仓提交、Ascend-CANN邀请、代码PR及社区最终验收仍属于后续流程，不以本设计提交替代社区验收。

## 参考依据

1. [社区任务详情](https://www.hiascend.com/activities/task-center/details/29c59d7ef9fa4aa8ad26b82969c5e2b6?menu=guide)及其配套《aclblasStbsv_A2A3任务书》：第2节接口与功能、第3节标准、第4节交付、第5/7节提交与注意事项。
2. [社区设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)的本地副本：SHA-256为a6ed8b29fbe1850499e1f5bb08db951cd6d75cac00dddda692c86b54b97114d5；本版保留其必填章节结构。
3. 本仓include/cann_ops_blas.h、include/cann_ops_blas_common.h、blas/common/helper/aclblas_handle_internal.h及CMakeLists.txt。
4. 冻结实现stbsv_host.cpp、stbsv_kernel.cpp、stbsv_tiling_data.h、stbsv_kernel.h。
5. 测试stbsv_test.cpp、stbsv_golden.h、stbsv_test.csv，以及最新A2原始日志、XML、CSV。

6. [社区任务流程与命名要求](https://www.hiascend.com/developer/activities/cann-community-task)；[2026社区任务目录规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。

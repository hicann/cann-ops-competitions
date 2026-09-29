# 需求背景（required）

## 需求来源

- 任务：2026 年 9 月 aclblasCtrsv 算子开发（Ascend 950PR）。竞赛仓任务编号为 09-33。
- 团队及提交账号：Gaomuer。
- 依据：任务方提供的《aclblasCtrsv 950 算子开发任务书》及 `test_cases/` 中的 CSV、GPU 基线和测试指导。
- 目标代码仓：[cann/ops-blas](https://gitcode.com/cann/ops-blas)，硬件 Ascend 950PR，CANN 9.1.0。
- 文档版本：2026-09-28，设计评审稿（补充真机实现方案）。性能标准表是任务目标；实测结果单独保存在自测报告中，不代表官方评审或验收状态。

## 背景介绍

### aclblasCtrsv 功能与工程现状

Ctrsv 求解单精度复数三角线性系统，输入右端向量 b 与输出解向量 x 共用 Device 存储，可用于复数线性系统的前代和回代。它是单右端 BLAS Level 2 接口，没有 side、alpha、beta 或 batch 参数，不能直接套用多右端 Ctrsm 的分核方案。

以本次核对的 ops-blas 提交 `fe54d86f00a4d449f55da1e8d144b898953d97c5` 为工程参考：公共头文件已有 `aclblasStrsv`，尚无 `aclblasCtrsv`；`blas/trsv/arch35/` 已有实数 Strsv 的 Host、Tiling 和 SIMT 内核。复用其句柄、stream、构建注册方式和工程组织，独立实现复数运算、共轭及边界语义。实数实现将 T/C 共用的做法不适用于复数。

本任务采用 Ascend C kernel 直调，不涉及 TBE 原型、ACLNN 两阶段接口、算子注册 JSON 或 950PR 私有平行 API。

# 需求分析（required）

## 需求描述

在 `include/cann_ops_blas.h` 新增以下公共声明，类型复用 `cann_ops_blas_common.h`，整数维参数保持 `int`：

```cpp
aclblasStatus_t aclblasCtrsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const aclblasComplex* A,
    int lda,
    aclblasComplex* x,
    int incx);
```

| 参数 | 位置与方向 | 类型/布局 | 约束与含义 |
| --- | --- | --- | --- |
| handle | Host 输入 | aclblasHandle_t | 有效句柄；使用绑定的 stream |
| uplo | Host 属性 | aclblasFillMode_t | UPPER(121) 或 LOWER(122)，指定原始 A 的可引用三角 |
| trans | Host 属性 | aclblasOperation_t | N(111)、T(112)、C(113)；C 额外取共轭 |
| diag | Host 属性 | aclblasDiagType_t | NON_UNIT(131) 或 UNIT(132)；UNIT 不读取 A 对角 |
| n | Host 输入 | int | n ≥ 0，n=0 为合法空操作 |
| A | Device 只读 | COMPLEX64，列主序 | 逻辑 n×n，物理 lda×n；实部、虚部各 float32 |
| lda | Host 输入 | int | lda ≥ max(1,n)，支持行尾 padding |
| x | Device 输入/输出 | COMPLEX64，带步长向量 | 逻辑长度 n；n>0 时物理跨度 1+(n-1)×abs(incx)；原地覆写 |
| incx | Host 输入 | int | 任意非零合法步长，支持正负；±1/±2/±3 为必测集合而非 API 上限 |

## 需求拆解

1. 支持 uplo×trans×diag 的全部 12 组枚举组合，正确处理实部/虚部和共轭。
2. 仅访问指定三角；UNIT 不加载对角；仅更新 x 的逻辑元素，不改写步长间隙，不写 A。
3. 支持运行时 n、合法 lda padding、负 incx、零维、非对齐尾部和非法参数返回。
4. 使用 Netlib `cblas_ctrsv`，逐例、分别对实部与虚部验证任务书精度标准。
5. 覆盖附带 1000 条功能与 200 条性能用例，补齐分布、毒化、步长保护及特殊值测试。
6. 五条典型用例满足任务书耗时上限，完整 200 条逐项与附带 GPU 基线对比；不得以少数典型用例替代全量性能验收。
7. 按目标仓目录、公共接口、README、测试注册及 CI 规范交付，并保留精度、性能、内存与复现证据。

# 详细设计（required）

## 算子分析

### 数学公式

令 M=op(A)，求解 Mx=b。若 M 为下三角，按 i=0,…,n-1 前代；若为上三角，按 i=n-1,…,0 回代：

```text
lower: x[i] = (b[i] - sum(M[i,j] * x[j], j < i)) / M[i,i]
upper: x[i] = (b[i] - sum(M[i,j] * x[j], j > i)) / M[i,i]
```

UNIT 直接省略除法且不加载原始对角。有效三角为 `effectiveUpper = (uplo == UPPER) XOR (trans != N)`；原始存储中的 uplo 不随转置改变。

| 原始 uplo | trans | 有效三角 | 求解方向 |
| --- | --- | --- | --- |
| UPPER | N | 上三角 | 逆序 |
| LOWER | N | 下三角 | 正序 |
| UPPER | T/C | 下三角 | 正序 |
| LOWER | T/C | 上三角 | 逆序 |

复数乘法 `(a+ib)(c+id)=(ac-bd)+i(ad+bc)`，共轭只翻转虚部符号。复数除法使用带缩放的稳定算法，避免直接计算 `c*c+d*d` 引起有限大数溢出；另对极小值、Inf/NaN、零分量逐项对照 Netlib。不得用钳位、更改输入或将特殊值替换为零掩盖误差。

### 支持数据类型与形状

仅 COMPLEX64；内部实部和虚部按 FP32 运算，不默认降为 FP16/BF16。n=0 时不申请向量存储、不访问 Device 指针。附带用例精度规模至 2048、性能规模至 4096；这是测试范围，不是接口硬上限。更大合法 n 通过有界 UB 分片及 GM 路径处理，仍受实际设备内存和地址可表示范围约束。

无广播，不支持超出 lda/incx 语义的任意 stride。n 是运行时标量，不需要额外动态图 shape 推导或返回视图。

### 地址计算

以复数元素为单位，使用 64 位中间量：

```text
A(row,col) offset = int64(col) * lda + row
start = incx < 0 ? (int64(n)-1) * (-int64(incx)) : 0
x(i) offset = start + int64(i) * incx
N: coefficient(i,j) = A(i,j)
T: coefficient(i,j) = A(j,i)
C: coefficient(i,j) = conj(A(j,i))
```

x 指向物理分配的起始地址，负步长由上述 start 处理，调用方不能再次移动指针。先扩展到 int64 再求负值，避免 `-INT_MIN` 溢出。Host 在计算字节数时检查乘加溢出；n=0 不计算负跨度。输入的分配容量由调用方保证。

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

参数检查顺序：

1. handle 为空，返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 检查三项枚举、n≥0、lda≥max(1,n)、incx≠0；失败返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. 合法 n=0 直接返回 SUCCESS，不要求 A/x 非空、不启动内核。
4. n>0 时检查 A/x 非空及地址跨度的可表示性；失败返回 INVALID_VALUE。
5. 构造 Tiling，使用 handle 的 stream 下发内核。资源与运行时错误遵循仓内状态码约定；异步执行错误在测试同步阶段单独检查，不能只检查 Host 返回成功。

任务书对 n=0 描述为“合法 no-op”，同时要求非法 lda/incx/枚举报错。本设计采用先验证标量再空返回的顺序，与 Netlib 参数检查顺序一致；不照搬当前 Strsv 在全部标量校验前空返回的顺序。补测 n=0 与非法标量组合，明确 Host 与测试期望一致。

Tiling 为 kernel 直调的 POD 数据，不新增图引擎 Tiling 注册。字段包括 n、lda、64 位带符号 incx、64 位 start、三项枚举。分块常量和线程数在实现中统一定义，入口根据公开参数和可用工作区分派，Host 不读取 A/x 内容。

##### 分核与并行策略

单右端存在顺序依赖，基线使用一个 AIV block，block 内 SIMT 线程并行处理每一步的独立更新或点积。不能将各行平均分给多个 core 独立求解，也不使用无保障的跨核自旋同步。线程数以 32/64/128 候选实测选择，不假设满核一定更快。

##### 分块与内存策略

小中规模把 x 的逻辑元素收集到 UB，结束后按 incx 散回；大于 UB 容量时用 GM 上的 x 配合有界分片缓冲。A 只按依赖区间读取，不物化整个 n×n 的 op(A)，不申请 O(n²) 转置 workspace。

缓存路径使用约 70 KiB 系数缓冲、48 KiB 的复数 x 与缩放因子缓冲、1 KiB 更新备份区；
kernel 动态 UB 预算为 208 KiB，并保留编译器使用空间。缓存求解和通用 GM 求解采用独立入口，
避免不同入口路径的局部数组/动态 UB 生命周期相互影响；已补测 4097 阶通用路径。

NON_UNIT、128≤n≤4096 且 handle 可用工作区不少于 3 MiB 时，采用 64 元对角块、128 行更新分块。
先并行预打包所有对角块，再按同一 stream 顺序执行“对角块求解→多核更新”。
工作区上界为 `(3*4096+64+ceil(n/64)*8256)*4` 字节，n=4096 时为 2162944 字节，属于 O(n)；
只预打包对角块，不物化整个 op(A)。核间数据通过 DMA 和 stream 依赖交换，不跨核自旋。
复用 handle 工作区，不改变其所有权，计算结束前调用方不得释放或跨 stream 竞争使用。
UNIT、小尺寸和工作区不足时走单核缓存路径，n>4096 走 GM 通用路径。

##### 路径规划

按 N 与 T/C、有效三角、UNIT、连续/带步长及 UB 可容纳性分派模板或运行时分支。共轭与对角分支尽量移到内层循环外。n=1 可用专门短路径；UNIT 下只保留 x，不读 A 对角。路径选择仅依据公开参数和设备资源，不针对 case 名、随机种子或 GPU 基线分派。

#### 3.2.2 Kernel 侧设计

1. Init：解析 Tiling，初始化必要缓冲；加载逻辑 x，线程同步后开始求解。
2. N 路径：按有效三角顺序求出枢轴 x[j]，广播给 block；线程分担同一列中剩余元素的 `x[i] -= A(i,j)*x[j]`。列主序下 A 读取连续，下一枢轴必须等待更新完成。枢轴为复数零时保留 Netlib N 路径的跳过语义，避免 `0*Inf` 造成额外 NaN。
3. T/C 路径：按参考 pivot 次序对剩余行更新，保持每个输出分量的减法顺序，不使用树形归约改变病态输入的数值行为。对角块预打包为有效三角的列布局，减轻 SIMD Gather 跨步读取；C 对所有被引用系数（包括非单位对角）取共轭，T 不取共轭。
4. 每步用 block 内同步保证枢轴发布、读取与更新的先后次序；尾部线程参与同步但屏蔽越界访问。严禁部分线程提前退出导致 barrier 死锁。
5. Store：只写 `start+i*incx`，不覆盖 x 间隙。GM 路径维护相同原地依赖与可见性，按 CANN 9.1.0 支持的同步方式实现并验证。

在 SIMT 函数内使用 `threadIdx/blockDim` 等接口；普通 `__global__ __aicore__` 入口使用 `AscendC::GetBlockIdx/GetBlockNum`，不混用作用域。沿用仓内 `asc_vf_call`、`asc_syncthreads` 和头文件组织，以目标版本编译验证接口合法性。

##### 三角与 UNIT 的严格访问保护

循环范围先排除未引用三角；UNIT 分支在 load 前排除对角，不能先加载整行/整块再用 mask 将对角改为 1。DMA 仅搬运完全位于允许区域的连续段；斜边、对角和尾部采用受保护的精确加载，不为对齐扩大到不可引用区域。A padding 不参与运算。

NaN/Inf 毒化验证“不依赖内容”，并配合源码/生成代码访问检查验证“不读取”；只做毒化测试不足以证明没有物理读取。

##### 数值与特殊值策略

FP32 顺序、复数除法和乘加融合可能改变极值行为。常规有限对角块用 SIMD 求解，Device 端根据实际数值判断不安全时转 SIMT 稳健路径；更新后出现非有限结果时，从更新前备份按相同 pivot 顺序重算，保留复数 Inf/NaN 恢复语义。复数乘法保留分步舍入，并关闭 FMA contraction。极大对角使用 2^24 的精确二进制缩放保存中间解，最终还原，避免后续运算中丢失次正规数；不改变输入矩阵、对角语义或测试阈值。已出现实虚部均 NaN 的 pivot 可按相同 IEEE 传播语义结束剩余更新。

运行时分类只使用公开参数和当前数据，不读取 case 名、随机种子、基线或测试期望。
分类的 warp 内归约用于安全标志，不改变浮点求解顺序。回退同样遵守三角和 UNIT 访问规则。

不对 UNIT 随机矩阵暗中缩放，不给 NON_UNIT 算法侧增加对角扰动。病态输入可能产生非有限解，测试保留输入、随机种子和 Netlib 输出分类；若配套用例确需修正，记录原始问题、理由和修正前后差异，经任务方确认，不能以过滤失败结果完成验收。

### 性能优化方案

先实现可解释的单 block 基线，重点减少每行 barrier、全向量重复搬运和 kernel launch 开销；利用 N 更新式和 T/C 点积式分别匹配列主序连续访问。单右端不默认使用 Cube，多右端 Ctrsm 的 Cube 分块优势不能直接推导到本任务。

在 msprof 中分别检查接口涉及的 kernel 数量、耗时、访存及同步开销；对线程数、缓存阈值和分片大小做参数扫描。分析运行与正式计时分开，避免 profiler 开销污染验收结果。必要时评估带显式依赖的分块求解/更新，但只有精度及完整接口性能均改善时才启用；不以单个内部 kernel 的耗时替代整个 API。

### 工程组织与构建安装

```text
include/cann_ops_blas.h                 # 公共 API 声明
blas/trsv/README.md                     # Ctrsv 参数、约束和产品支持
blas/trsv/arch35/
    ctrsv_host.cpp
    ctrsv_kernel.cpp
    ctrsv_tiling_data.h
test/trsv/ctrsv/
    CMakeLists.txt
    README.md
    ctrsv_param.h
    ctrsv_golden.h
    verify_accuracy.py
    verify_performance.py
    arch35/
        ctrsv_npu_wrapper.h
        ctrsv_test.cpp
        ctrsv_test.csv
        gpu_baseline.csv
docs/zh/api_list.md                     # 公共接口索引
```

遵循仓内源码自动收集，不在 `blas/trsv/arch35/` 另建一套工程；测试 CMake 使用 `ops_blas_add_gtest_tests()` 注册。生产实现位于 ops-blas，竞赛仓本 PR 仅提交设计文档。若合入前上游已新增公共声明，应复用而不是重复定义。

采用 `SOC_VERSION=ascend950`、Release、CANN 9.1.0 构建，显式链接 Netlib。安装核查包括构建所有安装目标、安装头文件、`aclblasCtrsv` 动态导出及 `ldd` 实际加载路径，随后从安装库运行测试。仅支持 950PR 的测试应由工程架构选择正确跳过 A2/A3，不把跳过记作 950PR 验证成功。新增编译优化仅作用于本算子必要源文件，遵循现有构建规则。

## 支持硬件

| 产品 | 本次支持范围 |
| --- | --- |
| Ascend 950PR，CANN 9.1.0 | 支持目标；需真机完成精度与性能验收 |
| Atlas A2/A3 及其他产品 | 本任务不实现、不宣称已验证 |

## 算子约束限制

- A 与 x 为对应设备上的有效存储，大小满足 lda 和 incx 描述；A 与 x 不得重叠，避免写 x 破坏只读系数。
- NON_UNIT 的对角非零由调用方保证，不增加奇异性或近奇异性扫描。
- 不包含 alpha/beta、batch、广播、额外 dtype 或任意非连续视图能力。
- 异步调用，读回结果前须同步绑定 stream；按仓内句柄约定使用，不承诺额外的跨 stream 同句柄并发保证。
- 不要求 bit-exact 或确定性；仍须满足任务书精度。不得把测试规模或测试内存预算解释成额外 API 限制。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 验收标准 | 来源 |
| --- | --- | --- |
| 精度 Golden | Netlib `cblas_ctrsv`，两侧使用相同输入副本 | 任务书 §3.1/3.2 |
| 有限值匹配 | 实部、虚部分别：abs(actual-golden) ≤ 2^-16 + 2^-10×abs(golden)，匹配率 ≥0.99 | 任务书 §3.2 |
| 最大误差 | 同时满足 1e-2 或 32×ULP 的最大绝对误差限制，单独报告最大误差及最差元素 | 任务书 §3.2 |
| 性能计时 | warmup 后有效采样 >50 次，报告平均单次设备耗时 us | 任务书 §3.3 |
| 性能集合 | 五条明确上限 + 全部 200 条 GPU 基线逐项核查 | 任务书及配套用例 |
| 内存 | 记录 Host/NPU 峰值、显式 workspace 及连续调用释放情况，无越界或泄漏 | 任务书 §4；§3.4 无额外内存数值门槛 |

32×ULP 以对应有限 Golden 分量计算，零/次正规数用 FP32 最小间隔，最大有限数用相邻有限数间距，避免 nextafter 向无穷导致无限容差。记录每个有限元素 `error/max(1e-2,32*ULP(golden))` 的最大值并要求不大于 1，同时报告 max_abs_error；此检查独立于 matched_ratio。Inf 必须同号匹配，NaN 需两侧均为 NaN，有限/非有限不一致直接失败，不将 NaN 比较排除后假报通过。MERE/MARE 可作为仓内附加统计，但不能替代任务书指标。

### 五条任务书性能上限

| n | uplo | trans | diag | incx | NPU 平均耗时上限 us |
| --- | --- | --- | --- | --- | --- |
| 512 | UPPER | N | NON_UNIT | 1 | 146.33 |
| 1024 | LOWER | N | NON_UNIT | 1 | 272.53 |
| 2048 | UPPER | T | NON_UNIT | 1 | 526.22 |
| 4096 | LOWER | C | NON_UNIT | 1 | 1177.84 |
| 4096 | UPPER | N | UNIT | 1 | 1030.61 |

附带 `gpu_baseline.csv` 实际已填写 200 条基线，`gpu_ms` 转换为 us 后计算 `GPU_us/NPU_us`。配套脚本阈值为 ≥0.4，开发验收目标采用严格 >0.4 并保留余量；前五条还需满足表中上限，取两者更严格者，不将已折算后的任务书上限再除以 0.4。相同参数的重复基线先检查一致性，关联键至少含 uplo/trans/diag/n/incx；不匹配或缺失不得计为通过。

### 测试输入与覆盖

保留原始 1200 条 CSV、生成脚本及基线的哈希，所有改动可追踪。使用固定种子，A/x 实部虚部分别独立生成。一般随机分布补齐均匀 [-5,5] 与正态（μ∈[-5,5]、σ∈[0.1,2]）各 50%，特殊值/负向用例另计；对同一参数和种子提供均匀、正态成对补充用例，明确新增数而不冒充原始用例。NON_UNIT 按任务要求加符号保持的 `max(5,n)` 对角偏移，特殊极值填充单独检查偏移结果；Host Golden 与 NPU 使用同一偏移后的 A。UNIT 不做额外数值缩放。

| 覆盖项 | 具体检查 |
| --- | --- |
| 枚举 | 12 组组合 × 代表尺寸，分别覆盖缓存和 GM 路径 |
| n | 0、1、小质数、2 的幂及 ±1、33/65/127/1025 等非对齐值、2048/4096；补充缓存分界和更大 n 的小负担验证 |
| lda/incx | 紧凑、n+4/n+8，±1/±2/±3；负步长起点、间隙哨兵及尾部保护 |
| 只读语义 | 未引用三角、UNIT 对角填 NaN/Inf/不同随机值，多次结果对照；A 前后保持不变 |
| 异常 | 空句柄、空 A/x、非法枚举、负 n、非法 lda、incx=0；n=0 合法与非法标量组合 |
| 浮点 | 0、正负零、极小/极大有限值、Inf/NaN；多个固定种子；N 的零枢轴与 T/C 的参考运算顺序 |
| 集成 | 自定义 stream、设备号选择、原地重复调用、安装库加载、错误路径资源清理 |
| 性能/内存 | 完整 200 条；逐条精度先验、计时、GPU 比值、分配及峰值记录 |

### 配套用例核对及修正计划

1. `TC_ED_095` 描述为 `lda0_n1`、期望 INVALID_VALUE，但原 CSV 的 lda 实际为 1，此输入按 API 应合法。生成器意图传入 lda=0；实现测试时保留原件，单独记录并修复工作副本为 lda=0，补充 n=1/lda=1 成功对照，不为适配错误 CSV 把合法输入判非法。修正差异提交任务方复核。
2. 配套 `verify_performance.py` 读取 GTest 总墙钟时间，包含准备、Golden 和比对，不能作为平均单次接口耗时；将解析独立的结构化性能输出，同时保留原 GTest 结果用于精度/完整性检查。
3. README 中 GPU 基线“待回填”与实际文件不符；以已提供的数据和单位为准，不重新虚构或放宽基线。
4. 生成器当前普通随机填充主要为均匀分布，不能因名称含 NORM 就当作正态；测试工程显式补充正态分布及比例统计。

### 计时与复现

每例先验证精度，warmup 10 次，有效采样 51 次。由于 x 原地覆写，每次调用前在同 stream 恢复原始 b；恢复操作在 start event 之前，end event 在完整 API 下发之后，保证计时覆盖所有内部 kernel，不计入 D2D 恢复、CPU Golden 或传输准备。记录平均值、分位值、采样数、设备型号、工具链、提交哈希及运行时库路径。测试程序读取明确设备号，所有 stream/event/Device 缓冲由同一设备上下文管理。

独立脚本要求 1000 条原始功能用例有逐项对应记录（含上述显式纠错）、新增用例单列、200 条 PF 唯一且齐全。缺失、重复冲突、异常退出、无基线、零样本、精度失败或性能不达标均失败退出，禁止把 NO_REF 当作验收成功。失败时保留具体参数、种子、实部/虚部最差误差与 Golden 分类，先定位后优化。

### 交付与可维护性

- 提交规范设计文档 PR，评审通过后开展正式实现；完成测试验收后再按 ops-blas 流程登记需求 Issue、提交代码 PR 并处理检视。
- 提供完整测试代码、CSV、基线、测试 README 和可复现命令；测试框架采用 ops-blas CSV + C++ GTest，不依赖 ATK 安装包。
- 按官方自测报告模板提供参数、实部/虚部精度明细及真实截图、200 条性能明细及截图、内存数据；报告与日志绑定同一提交，不用历史提交成绩替代当前版本验证。
- 验收信息填写个人代码仓、分支及 `blas/trsv/arch35/`、`test/trsv/ctrsv/arch35/`；提交前邀请 Ascend-CANN 为开发者。该操作属于交付检查，不在设计阶段声称已完成。
- 算子 README 产品支持表标注 950PR，保持公共 API 文档与实际调用一致；检查代码格式、静态检查、许可证及安装后真实调用。
- 证据在停机前同步本地并核对校验和。设计文档不附真实账号令牌、主机密钥或私密连接配置。

## 兼容性分析

新增公共符号，不改变既有 Strsv 的签名、枚举值和其他算子行为。950PR 实现只进入 arch35 构建；其他产品可复用公共声明，但本次不新增未验证的实现。保持 C ABI、句柄生命周期及 stream 用法与 ops-blas 一致。与实数 Strsv 的零维校验差异已在 Host 设计中明确，不顺带修改 Strsv。

## 参考资料

- [社区任务模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)及[社区任务目录规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。
- [ops-blas 贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)、公共头文件与 `blas/trsv/arch35/` 同族工程。
- [cuBLAS trsv 接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsv)与 [Netlib ctrsv](https://www.netlib.org/blas/ctrsv.f)：接口语义、负步长、引用范围和顺序行为参考。
- [生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)：本设计数值门槛按任务书 §3.2 明确给出的表格执行。
- [社区任务流程](https://gitcode.com/org/cann/discussions/39)。

# 需求背景（required）

## 需求来源

CANN 社区任务 2026，任务 74：单精度复数 Cholesky 分解、求解和批量接口（950）。贡献者/团队：`gcw_DvUFRDYr`。实现目标仓库：[cann/ops-solver](https://gitcode.com/cann/ops-solver)。

任务依据为 `Atlas950_Cpotrf_Cpotrs_Cpotri_CpotrfBatched_CpotrsBatched_task_doc.md` 及随附的五个算子测试包。本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织。

## 背景介绍

Hermitian 正定矩阵的 Cholesky 分解用于线性方程求解、协方差计算及矩阵求逆。分解结果可被后续求解和求逆复用。任务要求在 Ascend 950PR 上提供与任务书规定一致的 C API，全部数值计算由 NPU AI Core 完成。

现有 ops-solver 提供公共 handle、stream 管理及部分 LU/矩阵求逆接口，可复用其构建、状态码和资源管理机制。现有部分接口采用 C++ 复数和不同布局；本次接口必须独立遵循 COMPLEX64、列主序、Device 指针数组与 int32 参数契约，不通过已有接口改变调用语义。

# 需求分析（required）

## 需求描述

共同交付五种计算能力及两个 workspace 查询接口：

| 接口 | 输入 | 输出 | workspace/info |
| --- | --- | --- | --- |
| `aclsolverCpotrf_bufferSize` | 矩阵规格、Device A | Host Lwork | 复数元素数 |
| `aclsolverCpotrf` | 正定矩阵 A | 原位三角因子 | 调用者 workspace、Device 标量 info |
| `aclsolverCpotrs` | 三角因子 A、右端 B | 原位解 X | Device 标量 info；A 只读 |
| `aclsolverCpotri_bufferSize` | 因子规格、Device A | Host Lwork | 复数元素数 |
| `aclsolverCpotri` | 三角因子 A | 原位逆矩阵的指定三角 | 调用者 workspace、Device 标量 info |
| `aclsolverCpotrfBatched` | Device Aarray 指针数组 | 各矩阵原位因子 | 每矩阵独立 Device info |
| `aclsolverCpotrsBatched` | Device Aarray/Barray | 各 B 原位解 | Device 标量 info；nrhs=1 |

所有矩阵采用列主序，元素地址为 `base + row + col * ld`。`uplo=0` 表示 LOWER，`uplo=1` 表示 UPPER。`lda/ldb` 可以大于 n，矩阵外的 padding 不参与计算。调用者保持 Device 数据、指针数组和 workspace 存活，直到对应 stream 工作完成。

## 需求拆解

1. 补齐公开 C ABI、Host 参数校验和异步 stream 行为，保留既有接口 ABI。
2. 实现单矩阵分解、因子求解和因子求逆，共用三角访问、复数运算与分块调度。
3. 实现真正的 Device 指针数组批量路径、逐矩阵分解错误隔离和批量求解标量 info。
4. 覆盖任务包全部精度、info 和性能用例，补充边界、padding、非默认 stream 与确定性测试。
5. 对不同矩阵规模选择 UB、Vector 和 Cube 路径，在全部有 GPU 基线的用例上核验性能。
6. 交付设计、接口文档、示例、构建、测试工具、真实报告和提交材料；完成社区评审与验收。

# 详细设计（required）

## 算子分析

### 数学公式

以下统一采用 LOWER 因子 L。UPPER 输出 U=Lᴴ，但输入和输出仅访问调用者指定的三角。

分解：

`A = L Lᴴ = Uᴴ U`

`L[j,j] = sqrt(real(A[j,j]) - Σ(k<j) |L[j,k]|²)`

`L[i,j] = (A[i,j] - Σ(k<j) L[i,k] conj(L[j,k])) / L[j,j]`

对角根号内的数非正或非有限时停止该矩阵，写 `info=j+1`。成功的因子对角为实数，虚部置零。

求解：先解 `L Y=B`，再解 `Lᴴ X=Y`。UPPER 等价于先解 `Uᴴ Y=B`，再解 `U X=Y`。

求逆：由三角因子求解 `A X=I`，或等价地计算 `T=L⁻¹` 后形成 `X=Tᴴ T`；仅写指定三角。因子存在零对角时返回首个位置 `info=k+1`。

复数乘法采用四个 FP32 实数乘法：实部 `ar*br-ai*bi`，虚部 `ar*bi+ai*br`。共轭通过虚部取反实现，禁止将复数 Hermitian 更新简化为实数矩阵运算。

### 支持数据类型

公开类型：`aclFloatComplex { float real; float imag; }`，元素 8 字节；维度、leading dimension、batchSize、Lwork、info 为 int32。索引及字节偏移在内部使用 int64，避免大 batch 地址溢出。

NPU 存储与累加使用 FP32。CPU complex128 仅用于测试 golden；不参与 DUT 求解，不降低成 FP16/BF16 来替代精度要求。

### 支持形状

| 能力 | 范围 | 布局/限制 |
| --- | --- | --- |
| 单矩阵 | n=1…4096 | 列主序；lda≥max(1,n) |
| 单矩阵求解 | nrhs=1…128 | ldb≥max(1,n) |
| 批量 | batchSize=1…1,000,000 | 每矩阵地址来自 Device 指针数组 |
| 批量求解 | nrhs=1 | 不接受非空运算 nrhs>1 |
| 空操作 | n=0、nrhs=0、batchSize=0 | 按对应 API 契约校验其余参数，正确写 info |

矩阵地址不要求连续；不同 batch 元素可独立分配。实际总尺寸受 Device 可用内存约束。矩阵或右端在同一次批量调用中不得相互覆盖。

## 算子实现

### 实现方案

采用按规模选择的确定性分块算法。分解和三角求解的依赖顺序固定，尾矩阵按真实 n 处理。测试和性能记录明确区分各分支，阈值根据实机测量确定。

#### 3.2.1 host侧设计：

**接口与参数。** 七个新接口放入公共头文件的 `extern "C"` 区域；复数 struct 与 fill mode 放入 C 兼容公共类型头。Host 按原型中除 handle 外的参数顺序返回非法参数 `-i`，合法且排队成功返回 SUCCESS。无法使用 handle 或空 info 时只返回对应状态码，不能向非法地址写数据。Host 不读取矩阵内容或 Device 指针数组。

**异步执行。** 从 handle 获取调用者 stream，所有 info 初始化、布局转换及计算 kernel 都在该 stream 排队。计算结果异步返回，info 由调用者在 stream 完成后读取。内部 workspace 增长时等待该空间的上一次使用完成，已有足够空间时直接复用；不在每次计算后同步 stream。launch/runtime 错误由 Host 返回状态码，矩阵正定性和奇异性由 Device info 返回。

**workspace。** 令 P=align_up(n,32)，查询返回 `Lwork=4*P*P` 个复数元素，即 8 个 P×P FP32 plane，n=0 时为 0。调用者按 Lwork×sizeof(aclFloatComplex) 分配。plane 用于输入/因子实虚分量、共轭/转置、中间乘积与输出。查询与运行采用相同公式，并测试不足 workspace 的错误位置；n=4096 时公开 workspace 为 512 MiB。

**无公开 workspace 的接口。** n>128 的单矩阵求解使用 handle 管理的 `12*P*P*sizeof(float)` 字节临时空间。n>64 的批量分解使用不超过 65,536 个矩阵且不超过 8 GiB 的窗口 workspace，依次覆盖全部输入，全部 kernel 时长计入同一次 API 性能。64 列路径由整批规模选择，最后一个小窗口仍沿用四矩阵计算，使同一矩阵内容在不同窗口中得到逐位相同的结果。批量求解在 n≤128 时将完整因子保留在 UB，较大矩阵按原布局读取列面板，不建立整批因子的 GM 转换副本。临时空间按 context 和 stream 分开保留，以 event 记录最后使用；增长时等待上次使用完成，Destroy 等待并释放。

**分核与 tiling。** 从平台信息获取 Vector/Cube 核数。n≤32 的 batch 分解和求解将多个矩阵的对应元素映射为 Vector lane；n=32 的 batch 分解每组计算 32 个矩阵，按 8 个矩阵的小块搬运输入。33≤n≤64 的 batch 分解每组计算四个独立矩阵，每个矩阵占用 16 个 lane，并按 16 行分段更新。n>64 时采用分块算法：65≤n≤256 且 batchSize≥4×Vector 核数时使用 64 列 panel，其他情况使用 32 列 panel。Schur 更新 tile 为 128×128，FP32 Cube 按固定 k 顺序累加，计算范围为实际矩阵尺寸，两次乘法复用左操作数。批量较大时，每个 Cube 核连续处理同一矩阵的多个尾块。GM planar 转换和批量分解写回使用 64×64 tile，保持未指定三角和 padding。批量及完整 workspace 窗口均至少包含 4×Vector 核数个矩阵时，每组分解四个对角块，再在同一 kernel 内求解各矩阵的面板，每次处理最多 64 列。较大矩阵的 workspace 窗口较小时，32 列路径分别执行对角分解和并行面板求解。转置缓冲使用带 padding 的行步长。

**接入目录。** 计算 Host 入口分别位于 `src/cpotrf`、`src/cpotrs`、`src/cpotri`、`src/cpotrf_batched`、`src/cpotrs_batched`，共享 Host 校验在 `src/utils/cholesky_host.h`，共享 Device 组件放在 Cholesky 专用目录。由仓库 CMake 接入，避免改动其他算子的数学行为。

32 列和 64 列的四矩阵批量路径在尾面板不超过 16 列时，四个矩阵共用 SIMD lane 进行三角求解。在 64 列路径中，若剩余尾矩阵维度也不超过 16，则在同一 kernel 内完成 FP32 向量 Schur 更新和尾块分解，失败位置换算为原矩阵中的主子式序号。65≤n≤96 的四矩阵路径在同一 kernel 内完成根块、面板和尾块分解，面板的两组 16 列共用因子读取和对角元倒数，尾块复用根块的片上缓冲。对于 n≤512 且 n 不是 64 的整数倍的 LOWER 输入，按原列主序搬运列面板，用 SIMD 解交织和交织指令分离、组合实虚分量。每次处理 64 列，n>192 时处理 32 列，n>448 时处理 16 列，减少指定三角之外的数据搬运；片上行步长覆盖完整的向量读写。指定三角之外的数据和 leading dimension 的 padding 保持原值。

#### 3.2.2 kernel侧设计：

**小矩阵分解。** 单矩阵接口将完整因子保留在 UB，由一个 Vector 核完成固定顺序的 Cholesky 更新。批量接口按矩阵规模组织 SIMD：n≤32 时将多个矩阵的对应元素映射到不同 lane；33≤n≤64 时每组处理四个矩阵，各占 16 个 lane。CopyIn 按真实行列和 leading dimension 搬运，CopyOut 写回指定三角及各矩阵的 info。每个矩阵独立维护数值状态，失败不会影响同组其他矩阵。

**大矩阵分解。** 每个 panel 依次完成对角块分解、面板三角求解和 `A22 -= L21 L21ᴴ`。单矩阵采用 32 列 panel，批量接口按上述 tiling 条件选择 32 列或 64 列 panel。对角块和面板求解由 Vector 完成，在向量函数内按固定主元顺序计算。尾部更新将面板实部、虚部沿 K 维拼接，用两次 FP32 Cube GEMM 将负 Schur 乘积直接累加到尾矩阵的实部和虚部，关闭 HF32。各阶段通过同 stream 的 kernel 执行依赖衔接，Device info 控制失败矩阵的后续跳过。UPPER 在 CopyIn/CopyOut 时按共轭转置映射至统一 LOWER 内部表示。

**求解。** 小矩阵在 UB 完成两次三角求解，按 batch 或 RHS 分核。33≤n≤128 的批量求解缓存完整因子及其对角倒数，两种三角均保留输入因子的原布局。LOWER 前向使用列更新、后向使用共轭点积，UPPER 前向使用共轭点积、后向使用列更新。大矩阵单右端使用 128 列 Vector 分块：各核读取同一对角块并求解，再分别更新互不重叠的 RHS 片段；两个独立的 RHS 缓冲交替存放每一步结果。多个右端使用 32 列 panel 和 Cube 更新。n>128 的批量求解一核处理一个矩阵，直接读取原列主序因子的列面板，按同样的双三角顺序求解。全部运算使用 FP32；A 只读，B 原位写解，padding 保持原值。

**求逆。** 先保存输入因子并检查零对角。小矩阵按单位右端求解，输出指定三角；大矩阵采用分块三角求逆和 `TᴴT`，复用 TRSM/GEMM。输入保存、奇异检查和实际求逆都计入 POTRI 时间。仅计时前的 POTRF 准备被排除。

**batch。** Kernel 直接从 Device `uint64_t` 指针数组获取各地址，禁止以连续输入假设替代 Aarray/Barray。分解对不同矩阵分别判断正定性，坏矩阵不会阻止好矩阵。求解仅有一个 info，公共参数错误在 Host 校验；矩阵数值失败不伪造逐矩阵 solve info。

**搬运与同步。** 使用 Ascend C DMA、DataCopyPad 和 event/barrier 管理 UB 与 GM 的依赖，处理非整块行、列及 padding。各核写入范围不重叠；info 以独立 4 字节搬运写回。复数实虚 plane 在内部转换，外部保持 AoS ABI。所有测试包含带非零虚部的输入。

**短尾块求解。** 65≤n≤72 的批量求解分为 64 行主块和最多 8 行尾块。主块完成第一遍三角求解后，更新并求解尾块，再将尾块解回代到主块并完成第二遍三角求解。主块右端在每遍求解中保留在单组 Vector 寄存器，尾块按原因子布局读取耦合系数。

**小窗口负载分配。** 大矩阵 Schur 更新的 workspace 窗口至少覆盖两轮 Cube 核或恰好均分到各核时，按矩阵分配连续尾块。窗口只够一轮多、且不能均分时，将全部矩阵的尾块展开后分核，使各核承担接近的 tile 数。tile 内累加顺序不变。

**确定性。** 同一输入、设备及配置重复运行得到 bitwise 相同结果。每个输出 tile 唯一写者、累加 k 顺序固定，不使用跨核浮点 atomic 或依赖非确定性调度的归约。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / ascend950（dav-3510） | √ |
| Atlas A2/A3、其他芯片 | 本任务不声明支持 |

实机验证环境为 CANN 9.1.0、Python 3.11、NumPy 2.3.3、SciPy 1.16.2。任务书要求 CANN 9.0+，本次已验证的版本为 9.1.0。

## 算子约束限制

只支持 COMPLEX64，不支持 pivoting、稀疏布局或 row-major。POTRF 输入必须为 Hermitian 正定矩阵；POTRS/POTRI 输入为对应 uplo 的合法 Cholesky 因子。另一三角不作为有效输出。输入矩阵、B、指针数组、workspace 与 info 必须为调用者管理的合法 Device 内存。

batch 非空求解仅支持 nrhs=1。缓冲生命周期覆盖异步运算。公开接口中的 size 按 int32 约束，所有内部乘法先提升至 int64。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能精度 | complex128 golden；主文档 rtol=2⁻¹⁰、atol=2⁻¹⁶，matched ratio≥0.99，最大绝对误差≤1e-2 或 32 ULP；按原验证器解释组合逻辑 | 主任务书 |
| LAPACK 残差 | POTRF/POTRS：ratio≤max(5×CPU ratio,3×CPU mean ratio)；POTRI：ratio≤max(5×CPU ratio,0.1) | 主任务书 |
| 随包判定 | 同时执行包内 rtol=atol=2⁻¹³、EPS=2⁻²⁴ 的现有脚本，单独记录其结果 | 随包 verify_accuracy.py |
| 性能 | 全部有 GPU 基线的用例 T_NPU≤T_GPU/0.35；仅统计实际 DUT kernel 时间 | 主任务书、perf_baseline.json |
| 统计口径 | 记录每次样本、mean、median、离散度；主文档均值及补充说明 30 次 median 分别报告 | 主任务书、PERF_COLLECTION_SUPPLEMENT.md |
| 确定性 | 多次相同输入输出 bitwise 一致 | 主任务书 |
| 内存 | 独立记录 HBM 基线、调用前后和 2 ms 采样最大值，以及 Device 缓冲字节数；采样值不作为精确瞬时峰值 | 主任务书无独立门槛 |

### 用例覆盖及验证方法

| 算子 | canonical 精度用例 | info 用例 | 有效唯一 GPU 性能基线 |
| --- | ---: | ---: | ---: |
| Cpotrf | 145 | 3 | 143 |
| Cpotrs | 176 | 3 | 174 |
| Cpotri | 145 | 3 | 143 |
| CpotrfBatched | 151 | 1 | 151 |
| CpotrsBatched | 151 | 1 | 151 |
| 总计 | 768 | 11 | 762 |

原始 batch cases 存在三个重复配置，按 canonical ID/规格去重生成索引，保留原始映射。两个单算子 std case 缺少 GPU 基线，执行精度但性能记为无基线，不填假值。

测试程序直接调用公开 API，数据生成与 golden 在 CPU，DUT 在 NPU。batch 按原规格构造并计算全部 batchSize 个矩阵，输出按 sample_map 核验，并检查同内容槽位的一致性。

补充测试：n=0/1/4096、nrhs=0/1/128、batchSize=0/1、lda/ldb=n+8/n+32、两种 uplo、独立分配且地址顺序打乱的指针数组、混合非正定 batch、非默认 stream、参数位置 -i、不足 workspace、POTRI 零对角、重复运行确定性，以及 C 编译器头文件/链接验证。

msprof 采集 kernel 时长。POTRS/POTRI 的前置分解、数据生成、拷贝、编译和 bufferSize 查询不计入 DUT；接口自身的布局转换、检查及多 kernel 全部计入。报告覆盖所有有有效 GPU 基线的原规格用例。

### 任务包口径说明

1. 主任务书与随包脚本的逐元素容差不同，保留两套结果并向维护者确认正式裁决口径。
2. 主任务书“输入不超过 4G”与 canonical 大 batch 实际输入冲突：n=224 的用例可达到约 15.2 GiB。测试按 canonical 原规格构造全部矩阵，保留完整 batch，使用 128 GiB HBM 的 950PR 设备执行。
3. 主任务书性能均值与补充材料 median 不同，原始 30 次样本同时支持两种统计。
4. UPPER 分解的残差应为 UᴴU；POTRI 残差应使用原始正定矩阵，不能把因子当原矩阵。测试沿用随包实现中的正确公式，并说明任务书文字歧义。
5. 无公开 workspace 的接口由 handle 按 context 和 stream 分开缓存临时空间。容量增长及 handle 销毁时，等待已登记的完成 event 后再释放；更换 stream 时，后续调用使用对应 stream 的独立缓存。
6. 官方 ZIP 中两个 batch 索引原为 Git LFS 占位文件，现已从 LFS 下载真实对象，文件大小及 SHA-256 与占位记录一致。批量测试执行完整 batch，核对固定索引的规格及槽位映射，并直接读取其 `ratio_cpu_mean`；输入、golden 和逐内容 CPU 比值按原验证器现场生成。

### 交付与实测结果

七个接口已在 Ascend 950PR、CANN 9.1.0 上完成实机自验证。779 项原包精度及 info 检查、768 项任务书精度主判据、349 项补充回归和两个 C 示例/ABI 测试均通过。762 项性能用例各预热 5 次、采样 30 次，mean 和 median 均满足 GPU mean/0.35；另完成七项 HBM 采样及公开命令行复现检查。

交付包含 API 文档、C 调用示例、源码/CMake、测试与采集脚本、环境记录、自验证步骤、精度/性能/内存三个 xlsx 报告及各自日志。原始 profiler 数据、逐次 kernel 记录和 NPU 输出随证据包保存，可用于复算统计和核对结果。上述记录为工程自验证，平台验收与社区评审状态另行记录。

## 兼容性分析

新增函数使用 C linkage，公共复数类型可在 C 与 C++ 调用。现有使用 std::complex 的声明继续仅暴露给 C++；既有函数签名和符号保持不变。原有 fill mode 移至公共类型头并保持取值一致，不重复定义。

验证新增 API 的 C 编译、C++ 编译、导出符号，以及仓库现有构建和相关 handle/stream 测试。对不同 CANN 版本分别记录已验证范围。未完成编译或实机验证的版本不列为已支持。

# Ascend 950 单精度复数 Cholesky 算子设计与开发计划

版本 V1.1（设计评审稿）　编制日期 2026 年 9 月 28 日

提交者：gcw_l6gqPgyE。任务包未提供正式编号，当前公开任务列表亦未定位到对应条目，因此目录暂用 Cholesky-950；后续按社区确认的编号调整。

## 需求背景

需求来源为《Atlas 950 单精度复数 Cholesky 分解、求解和批量接口任务书》及配套五组测试包。目标是在 Ascend 950PR 上补齐稠密 Hermitian 正定矩阵的复数分解、求解、求逆和批量接口，保持与任务书指定的 cuSolver legacy API 语义对齐。设计提交至社区任务仓，完成后的实现代码进入 ops-solver。

本计划覆盖 Cpotrf、Cpotrs、Cpotri、CpotrfBatched、CpotrsBatched 五个计算接口及两个工作空间查询接口，供项目开发、测试和验收准备使用。建议采用“接口与测试基线先落实、共用计算组件先实现、五接口逐步贯通、全量精度和性能统一验收”的路线。

按一名熟悉 C++ 和 Ascend C 的主开发者全职投入、950PR 环境可持续使用估算，基础工作为 40 个工作日，另预留 5 至 10 个工作日处理精度、性能和评审问题，总计约 9 至 10 周。排期以启动 M0 为起点，等待算力和外部回复的时间另计；任务方尚未给出本计划可采用的截止日期。

**当前结论：资料足以制定实现方案，但正式测试基线尚需补齐和统一。** 已核对 64 个包内文件、五套用例和关键验收脚本，并检查上游代码快照。两个批量索引文件实际为 Git LFS 指针；精度容差、残差常数、性能统计方式和部分规模限制存在差异。第 3 节列出了证据、影响和处置。以下实现与测试内容均为开发安排，不代表已经完成 NPU 实现或通过验收。

## 需求分析

### 1 目标范围与交付边界

#### 1.1 必须完成的能力

| 接口 | 计算语义与结果位置 | 最低支持范围 |
|---|---|---|
| aclsolverCpotrf | 对 Hermitian 正定矩阵分解；LOWER 为 A=L·Lᴴ，UPPER 为 A=Uᴴ·U；指定三角原地写回 | n 为 1 至 4096，LOWER 和 UPPER，合法 lda padding |
| aclsolverCpotrs | 输入已有 Cholesky 因子；求解 A·X=B；X 原地覆盖 B | n 为 1 至 4096，nrhs 为 1 至 128，合法 lda 和 ldb |
| aclsolverCpotri | 输入已有 Cholesky 因子；将逆矩阵的指定三角原地写回 A | n 为 1 至 4096，LOWER 和 UPPER |
| aclsolverCpotrfBatched | 每个矩阵独立分解，逐矩阵写 infoArray | n 为 1 至 4096，batchSize 为 1 至 1000000，Device 指针数组 |
| aclsolverCpotrsBatched | 对已分解因子逐矩阵求解；覆盖 Barray 各矩阵 | 同上，nrhs 仅支持 1，info 为单个标量 |

另交付 aclsolverCpotrf_bufferSize 和 aclsolverCpotri_bufferSize。五个计算接口须作为同一任务完整交付，不能以其中一部分通过测试代替任务完成。[S1 §1、§2、§7]

统一使用 COMPLEX64，外部表示为两个连续 float 的 aclFloatComplex；维数使用 32 位 int。数据为列主序，地址按 base + row + col×ld 计算。lda、ldb 支持大于 n，不能沿用旧算子的 lda==n 限制。矩阵、计算用 workspace、info 和批量指针表在 Device；bufferSize 的 Lwork 是 Host 输出整数，单位为复数元素数而非字节。[S1 §2.3—§2.4]

#### 1.2 工程与执行要求

工程采用 ops-solver Host C API 加 AscendC/CATLASS Kernel 直调，NPU 实现不依赖 CUDA。核心计算在 950PR AI Core 上完成，CPU 仅承担参数与调度工作，以及测试侧参考计算。复用现有 handle 的创建、销毁和 stream 管理，计算在调用方 stream 上异步下发。

相同输入在同一 stream 串行重复运行，结果和 info 应逐位一致。每次测试必须恢复原始输入，避免将上一次原地输出作为下一次输入。LOWER/UPPER 均需真正实现复共轭；对角保持实数语义。

本期不扩展到双精度复数、广播、图融合或新增求逆批量接口。任务书第 3.4 节明确“内存不做要求”，因此不加入其他算子任务中的 L2 上限或 GPU 相对内存比例门槛；仍须检查越界、泄漏、溢出与实际容量。

### 2 资料清单与上游起点

#### 2.1 本次任务包核对结果

原始包为 a3c754a14f4340d0b8d7e9e07957f19e.zip，共 64 个非目录文件，内容包含任务书、接口列表、快速入门、交付说明、性能补充说明、五组 cases.json、bench_result.json，以及各组 package 内的生成器和校验器。包内没有本任务的完整 C++ 算子实现，也没有冻结的输入数组；数据按 seed 现场生成。

| 目录 | 原始性能记录 | 规范精度用例 | 额外 info 契约用例 | 可匹配 GPU 基线 |
|---|---:|---:|---:|---:|
| cpotrf | 143 | 145 | 3 | 143 |
| cpotrs | 174 | 176 | 3 | 174 |
| cpotri | 143 | 145 | 3 | 143 |
| cpotrfbatched | 154 | 151 | 1，生成器派生 | 151 |
| cpotrsbatched | 154 | 151 | 1，生成器派生 | 151 |
| 合计 | 768 | 768 | 11 | 762 |

两个“768”不是同一集合：三个单矩阵规范集各增加 2 条 std 精度用例，共 6 条没有 GPU 性能基线；两个批量规范集各去除 3 条重复规格。性能验收需保留原始记录到规范 case_id 的映射，不能只按总数或数组下标判断是否覆盖。计划精度范围为 768 条规范用例加 11 条 info 契约用例，共 779 条，另补接口契约与边界测试。[S2、S3、S4]

五份 manifest 中，canonical、生成器、模拟件、精度脚本和性能基线等已实际读取文件的指纹相符；五份 README 的 SHA256 与各自 manifest 记录不一致。批量 index 的真实内容未包含在 ZIP 中，无法验证其解压后指纹。本文保留影响实现和验收的核对结论；逐文件核对记录作为开发工作底稿留存，本次设计 PR 不附带原始任务包。[S8]

#### 2.2 上游代码现状

本次只读检查的 ops-solver 提交为 22bbbe69aa23da2bb8f8e73c66fe1339e3db227e。该快照没有五个目标接口及对应实现目录；有 handle、状态码、LU 求解相关代码和部分复数 GEMM/TRSM 工具。现有工具的布局、共轭、类型和 950 兼容性必须逐项验证，不能直接视为可用的 Cholesky 组件。[S6]

已经识别的接入事项如下。

- aclsolverFillMode_t 已存在于 cann_ops_solver.h。任务书要求其位于 common 头文件，实施时迁移或统一唯一声明，避免重复定义，并保持既有 Cheevj 调用兼容。
- 公开头文件含无条件 C++ 头和 std::complex，common 头含 cstddef。为满足新接口 C 可调用，需要条件化旧 C++ 声明、加入 C 可用类型和完整 extern "C" 边界，并分别编译 C 与 C++ 消费者。
- aclsolverStatus_t 已存在，应直接复用。旧计算接口返回 aclError，不应在本任务中全面改写其 ABI；新增接口按任务书返回 solver 状态码，内部转换 ACL 错误。
- CMake 已将 ascend950 映射到 dav-3510，并识别 arch35；正式构建仍需在 950PR 环境验证，不能把映射存在视为编译或运行成功。
- 上游 README 记录已验证 CANN 9.0.0 的两个构建时间戳 20260422000325096 和 20260325000325538。开工时按实际 950PR 驱动、Toolkit 与仓库版本冻结配套，不只记录“9.0.0 以上”。

### 3 开发前必须落实的事项

下表中的事实来自本次文件核对。拟采用的处理办法是开发建议，涉及验收口径的最终选择需由任务方确认；不得自行改写官方脚本并把结果标成官方通过。

#### 3.1 数据完整性和数值判据

| 编号 | 已确认的问题 | 影响与处理安排 |
|---|---|---|
| G0-01 | 两个 index.json.gz 均为 133 字节 Git LFS 指针；指向的对象分别应为 36060709 和 35897248 字节 | 开发前取得真实对象或任务方认可的重生成方案；核对压缩文件 OID 和解压后 manifest。不能对指针文本执行 gunzip 并期待获得索引 |
| G0-02 | 任务书 rtol=2^-10、atol=2^-16；五套脚本 rtol=atol=2^-13 | 两套规则并非简单的宽严关系。保留原脚本，在独立诊断报告中并列计算两套结果，取得统一的版本与阈值说明 |
| G0-03 | 任务书残差 ε=2^-23；脚本 EPS32=2^-24 | 同一残差数值会差一倍；固定阈值分支尤其受影响。ratio_cpu、mean 和 NPU 残差必须采用同一批准口径 |
| G0-04 | Cpotri 残差文字把 A 写成输入三角因子；脚本使用生成的原始 A32 并作 Hermitian 补全 | 逆矩阵验证应明确区分因子 F 与原矩阵 A0，禁止直接用 F·C 与 I 比较；记录原矩阵、实际输入因子与参考链路，确认公式解释 |
| G0-05 | 任务书 potrf 残差通式写 F·Fᴴ，但 UPPER 分解语义为 Uᴴ·U | 对 UPPER 使用 Uᴴ·U 还原；将公式解释写入测试说明，避免验收误判 |
| G0-06 | 五份 README 指纹均与 manifest 不符 | 保留原包和当前 SHA256，向任务方确认发布版本；差异不自动意味着数值脚本损坏，也不能宣称全包指纹通过 |

批量校验器可按 seed 重造 sample_map，但缺少 index 时会失去固化的 ratio_cpu_mean，并可能走单支兼容判定。因此，缺少 LFS 文件并不意味着所有开发测试都无法进行，却会影响完整复现和正式判据证据；应在 G0 关闭前禁止形成正式验收结论。[S3、S4]

#### 3.2 性能规模和接口边界

| 编号 | 已确认的问题 | 影响与处理安排 |
|---|---|---|
| G0-07 | 任务书要求平均单次 kernel 耗时；补充说明要求 30 次采样取中位数，并将中位数填入 avg_ms | 原始样本保留 mean、median、min、max；正式主表暂按任务书 mean 规划，补充参考表注明 median；最终口径确认后统一，不能混填 |
| G0-08 | 任务书称验证输入控制在 4G 内；两个批量规范集中各有 111 条仅输入矩阵就超过 4 GiB | 最大输入约 16.32 GB，且不含工作区、输出留存和 Host 副本。落实足够容量或取得任务方修订规格；不得为达标自行缩小 batch |
| G0-09 | 通用条款称 nrhs=0 为空问题；CpotrsBatched 专项要求 nrhs≠1 且非空时报错 | 冻结 quick return 与参数校验顺序。拟按专项规则：n>0 且 batchSize>0 时 nrhs=0/2 非法；n=0 或 batchSize=0 的行为单列确认 |
| G0-10 | 任务书 bufferSize 参数表把 A 空指针与计算接口一起描述；cuSolver 通则允许查询时未分配 A | 将 bufferSize 的 A=nullptr 与计算 API 的非空要求分开测试，最终以确认后的兼容契约为准 |
| G0-11 | 空 handle、无效枚举的现有仓库状态码与任务书通用 INVALID_VALUE 表述并不完全相同 | 制定 status/info 对照表；空 handle 无有效 stream 时不假定能写 Device info；不把 ACL 返回码直接转换为 solver 枚举 |
| G0-12 | 附件没有可直接引用的完整 GPU 机型和 CUDA/cuSolver 版本记录，gpu_baseline.csv 也未随包提供 | 以 bench_result.json 及 bench_key 做当前追溯；补取环境记录和正式基线版本，避免重新采集后无说明地替换参考值 |

两个最大输入例为 cpotrfBatched-0041，n=224、batch=40659，A 占 16320847872 字节；cpotrsBatched-0041，n=224、batch=40478，A+B 占 16320729600 字节。批量输入大小以 8×batch×(lda×n+ldb×nrhs) 计算，potrf 不含 B 项。以上为容量计算，不是运行峰值测量。[S8]

这些问题可与接口骨架、公共组件设计并行处理。G0 未全部关闭时可以进行探索性开发，但正式用例冻结、精度结论与性能验收不得跳过相关问题。

## 详细设计

### 4 技术架构与公开接口设计

#### 4.1 Host 与 Kernel 分工

Host 层负责校验维数和枚举、处理空问题、计算 workspace 和 tiling、取得 handle stream、转换状态码与下发 kernel。不能为读取 info 或逐批检查矩阵指针而无条件把 Device 数据拷回 Host。裸 C 指针本身不携带 dtype 和 shape，Host 不能宣称能识别所有调用者的错误内存布局；可检查项与调用者责任应在接口文档中明确。

Kernel 层负责复数数据搬运、共轭访问、分解面板、三角求解、尾块更新、三角求逆、结果写回和 Device info。批量指针数组由 kernel 读取，指针位宽按目标 ABI 处理；矩阵可以分散分配，不能用第一个地址加固定步长替代真实 Aarray/Barray。

公共层只提取确实共享的复数运算、列主序视图、三角访问、tile 搬运和确定性归约。初期不扩展为通用求解器框架。优先沿用仓库 Host/Kernel/Test 模式，保持实现文件数量与职责相称。

#### 4.2 类型和 ABI

aclFloatComplex 定义为 real、imag 两个 float；以编译期断言检查 sizeof 为 8、字段偏移分别为 0 和 4，并验证对齐要求。新增导出函数参数顺序完全按任务书，特别保留 CpotrsBatched 的指针表参数类型，不能为了内部便利换成连续三维张量。

在 common 头统一 FillMode 和复数类型；公开 C API 在 extern "C" 内声明。旧的 std::complex 接口保留 C++ 可见性和既有链接方式。验收增加 C 语言 include/link 测试、C++ 回归以及共享库符号检查。

七个公开函数原型按任务书保留如下，实际声明置于 C/C++ 兼容的公开头中：

```c
aclsolverStatus_t aclsolverCpotrf_bufferSize(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    aclFloatComplex *A,
    int lda,
    int *Lwork);

aclsolverStatus_t aclsolverCpotrf(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    aclFloatComplex *A,
    int lda,
    aclFloatComplex *Workspace,
    int Lwork,
    int *devInfo);

aclsolverStatus_t aclsolverCpotrs(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    int nrhs,
    const aclFloatComplex *A,
    int lda,
    aclFloatComplex *B,
    int ldb,
    int *devInfo);

aclsolverStatus_t aclsolverCpotri_bufferSize(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    aclFloatComplex *A,
    int lda,
    int *Lwork);

aclsolverStatus_t aclsolverCpotri(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    aclFloatComplex *A,
    int lda,
    aclFloatComplex *Workspace,
    int Lwork,
    int *devInfo);

aclsolverStatus_t aclsolverCpotrfBatched(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    aclFloatComplex *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

aclsolverStatus_t aclsolverCpotrsBatched(
    aclsolverHandle_t handle,
    aclsolverFillMode_t uplo,
    int n,
    int nrhs,
    aclFloatComplex *Aarray[],
    int lda,
    aclFloatComplex *Barray[],
    int ldb,
    int *info,
    int batchSize);
```

#### 4.3 错误与空问题契约

参数编号不计 handle。对每个 API 建立独立映射，不能复用一张不区分 nrhs 和 batchSize 的表。

| 接口 | 从 1 开始的参数顺序 |
|---|---|
| Cpotrf / Cpotri | uplo、n、A、lda、Workspace、Lwork、devInfo |
| Cpotrs | uplo、n、nrhs、A、lda、B、ldb、devInfo |
| CpotrfBatched | uplo、n、Aarray、lda、infoArray、batchSize |
| CpotrsBatched | uplo、n、nrhs、Aarray、lda、Barray、ldb、info、batchSize |

参数错误返回相应 Host status；在 info 可写且可取得有效 stream 时，以 stream 有序的方式写 -i。CpotrfBatched 的参数错误按任务书写 infoArray[0]。对空问题、空输出指针与多参数同时非法的优先级建立契约测试，经 G0 确认后固定。

数值失败与参数错误分开：potrf 写最小不正定顺序主子式下标，保留此前已完成因子；potrfBatched 各矩阵独立失败，其他矩阵继续。potri 写最早零对角的正下标；potrs 和 potrsBatched 的 info 仅报告参数问题，不追加正定性检测职责。

#### 4.4 工作空间与异步生命周期

bufferSize 使用受检的 64 位中间量计算字节数，再向上换算为复数元素数并检查 int 可表示范围。地址偏移、batch×lda×n 和分配字节均不得用 32 位乘法，已有用例超过 2 GiB，必须避免静默溢出。

potrf 和 potri 的暂存切片在返回的 Lwork 内统一分配并明确对齐。计算入口核验 Lwork 充足；Device info 写入、搬运和所有计算阶段均计入生命周期。用户 workspace 只有在所属 stream 完成后才能复用或释放。

无显式 workspace 参数的接口优先使用核内 tile、在位遍历和可证明安全的存储。若批量大矩阵路径需要内部暂存，须先确认仓库和运行时的异步内存管理方案，保证并发 stream 隔离、容量可控及释放时序；不能在核仍执行时释放，也不能通过每次全设备同步掩盖生命周期问题。

### 5 算法实现与优化顺序

#### 5.1 公共复数计算组件

先实现 COMPLEX64 搬运、乘加、共轭、实对角处理、固定顺序归约、三角视图及 padded ld 访问。初期保持 FP32 计算与累加路径；是否使用硬件矩阵单元及其计算模式，通过 950PR 工具链和数值实测决定，不预设某种低精度模式能够达标。

复杂更新可拆为实数乘法组合。对于 A·Bᴴ，实部为 Ar·Brᵀ+Ai·Biᵀ，虚部为 Ai·Brᵀ−Ar·Biᵀ。先采用清晰的四乘法表达，完成抵消误差和 UPPER 共轭验证后再评估其他变换。归约树和写回顺序固定，避免依赖调度顺序的浮点原子累加。

#### 5.2 Cpotrf 分解

小规模路径先采用固定次序的面板分解，在 AI Core 内完成对角检查、平方根、列更新。将其作为大矩阵分块算法和批量小矩阵算法的共用基础。

大规模采用分块右看式 Cholesky。LOWER 路径按列块依次执行：对角块 POTRF、面板右侧共轭三角求解、尾部 Hermitian 更新。UPPER 路径实现相应共轭对称流程，不能仅改索引而遗漏共轭。

分块宽度先以 32、64、128 等候选做实验，再依据本地存储、寄存器占用和 profiler 选取，候选数值不作为提前承诺。n 非整块时使用边界 mask；不读取未提供的另一半三角。阶段间通过 stream 顺序和明确的数据依赖衔接。面板发现不正定后，后续 kernel 读取 Device 状态并跳过该矩阵，避免 Host 每块读回同步。

#### 5.3 Cpotrs 求解

LOWER 执行 L·Y=B，再执行 Lᴴ·X=Y；UPPER 执行 Uᴴ·Y=B，再执行 U·X=Y。nrhs=1 使用向量三角求解路径，多 RHS 采用分块 TRSM 和更新；对 1、8、32、64、128 分别观察访存与并行效率。

正式接口保持因子只读、B 原地覆盖。测试同时覆盖“固定 CPU 参考因子输入”与“NPU potrf 输出接入 potrs”两条路径，前者定位求解自身问题，后者验证组合功能。性能统计不包含准备因子的 potrf 时间。

#### 5.4 Cpotri 求逆

LOWER 先求 T=L^-1，再计算 C=Tᴴ·T；UPPER 先求 T=U^-1，再计算 C=T·Tᴴ。只写用户指定的逆矩阵三角，避免对未存储侧作额外正确性承诺。先扫描对角并记录首个零对角，防止先发生除零再补写 info。

以分块三角求逆和三角乘积作为主路线。可用“对单位矩阵求解”作小规模独立交叉检查，不能未经计时就将其确定为全部规模的最终性能方案。workspace 明确保留仍会使用的原因子块，不能在原地写回时覆盖后续依赖。

#### 5.5 两个批量接口

小矩阵按一个计算单元处理一个或多个独立矩阵，依据 n 和 batchSize 分配工作量，降低逐矩阵启动开销。大矩阵按矩阵和 tile 两维调度，复用单矩阵计算组件；API 层一次接收原 batch，内部调度分块不能改变外部调用规格。

每个实际矩阵地址都从 Device 指针表取得。测试采用随机地址顺序、不同间隔和分别分配的内存，以排除对连续布局的偶然依赖。内容相同的不同 batch 槽位仍须实际处理，不得按测试 seed、case_id 或内容缓存答案。

potrfBatched 逐矩阵写 infoArray，单个失败不应阻塞整批。potrsBatched 仅接收 nrhs=1，写单个 info；不能把它误做为每矩阵一个状态。性能阶段使用完整 batch 单次调用，不把多个小批次耗时换算成官方原规格结果。

## 可维可测分析

### 6 测试工程与证据设计

#### 6.1 执行器接入

包内 Python 生成器负责输入与 Golden，拟新增 C++ NPU 执行器负责分配、列主序与 padding 物化、Device 指针表构建、调用、同步取回和保存结果。输出适配为 package 要求的 out32、info、status 三个字段，同时单独保存真实 aclsolverStatus_t、ACL 错误及日志。

status=ok 表示执行器完成了预期测试流程。非法参数探针的实际 API 返回值仍须被记录并核验，不可为了让校验器读文件而把错误状态改写为 API 成功。精度包中的半三角输出按要求整理未使用侧；该整理仅发生在结果导出层，不构成要求 kernel 将未使用三角清零。

单矩阵输出保留二维矩阵；批量输出保留 (batch,n,cols)，batch=1 也不压缩维度。potrfBatched 的 info 为 (batch,) int32 数组，potrsBatched 为标量。先用极小数据验证格式，再扩至全量。

#### 6.2 数值验收

采用 COMPLEX128 CPU Golden，实部和虚部分别检查。保留任务书与原脚本两个诊断视图，待 G0 冻结最终标准。报告列出 matched_ratio、最大绝对误差、ULP 锚点、残差、CPU 残差及实际阈值，不能只输出一个 PASS。

potrf 在 LOWER 比较 L，在 UPPER 比较 U，残差分别以 L·Lᴴ、Uᴴ·U 还原。potrs 使用实际 FP32 输入对应的原矩阵和 RHS 升精度计算残差，记录准备因子链路。potri 用原矩阵 A0 与补全后的逆矩阵 C 检查 I−A0·C；原矩阵与三角因子单独命名保存，避免 G0-04 中的歧义。

批量精度包按 k=min(5,batch) 个代表内容及 sample_map 展开：先检查同内容所有槽位的 bit-wise 一致性，再对代表内容做精度与残差判定。此机制减少 Golden 的重复计算，不减少 NPU 实际 batch 调用规模。info 契约用例单列计数，不进入精度均值。

#### 6.3 必须追加的功能用例

| 类别 | 追加场景和判定 |
|---|---|
| 尺寸边界 | n=0、1、2，2 的幂及其前后邻值，4095、4096；n<0 拒绝 |
| RHS | 单矩阵 nrhs=0、1、8、32、64、128；批量 nrhs=1 合法，0/2 等按冻结契约检查 |
| Padding | lda/ldb=n、n+8、n+32；padding 放哨兵，验证合法数据与不越界 |
| 批量 | batch=0、1、8、128、1024、3000、1000000；结合 n 控制容量；原性能大 case 单独保留 |
| 三角和共轭 | LOWER/UPPER；非存储侧放干扰值，保证结果只依赖指定三角；非零虚部检查共轭方向 |
| 错误与 workspace | 非法枚举、负维数、非法 ld、空指针、Lwork 不足和查询值一致性；逐项核验 -i |
| 数值失败 | 第 1 个、中间及末尾主子式失败；批量混合成功与失败；potri 首个零对角 |
| 因子与原地契约 | potrs 因子只读，B 覆盖；potri 和 potrf 原地；padding 和相邻矩阵保护 |
| Stream | 非默认 stream、两个 handle 不同 stream 的交错执行、工作区隔离、完成后再释放 |
| 确定性 | 恢复相同输入后至少 10 次串行复跑，逐位比较有效结果与 info；10 次为内部测试建议 |
| 特殊值 | 按确认后的标准检查 INF/NAN 和对角虚部规则，保留原脚本分类证据 |
| 工程兼容 | C/C++ 头文件与链接、导出符号、原有算子代表用例、仓库格式和静态检查 |

当前规范用例的 n 最小为 2，所有 lda/ldb 均无 padding；Cpotrs 最大 nrhs 为 64，两个批量集最小 batch 为 32，因此不能仅运行附件规范集便声称覆盖了全部功能范围。[S8]

#### 6.4 兼容性与可维护性

新增接口不改变既有导出符号和旧接口返回类型。公共枚举迁移、C 兼容头文件改动均需通过旧接口构建和代表用例回归；算子、分块组件、执行器与汇总器分层，错误日志关联 API、case_id、实现提交和 stream。任何 tiling 或精度优化变更均保留相邻尺寸、UPPER/LOWER、padding 与确定性回归，防止局部优化扩散为接口行为变化。

#### 6.5 通过条件

本地自测门槛是：规定的精度集和 info 项全部有真实 NPU 输出；无缺失、未知或执行错误条目；追加功能用例通过；重复性和 stream 测试通过；当前快照的必要旧接口回归通过。包内 formal 字段保持 PENDING_RULING，应另列“内部自测结论”和“任务方正式验收状态”。sim_dut.py 只可用于执行器格式联调，不能产生 NPU 验收证据。[S3、S4]

### 7 性能测量与目标

#### 7.1 统一计时办法

按任务书逐 case 计算 speedup=T_GPU/T_NPU，要求 speedup≥0.35；等价为 T_NPU≤T_GPU/0.35，或 T_NPU/T_GPU≤2.857142857。每条正式性能规格均须达标，平均加速比不能替代逐条判定。

使用 950PR 上的 msprof 原始输出进行 kernel 归因。一个 API 若包含多个 kernel，按同一次 API 调用汇总其全部必要计算阶段，再对多次调用求平均，不能只取最快的 kernel 名。输入恢复、H2D/D2H、数据生成、首次编译及 potrs/potri 的前置分解不混入目标计时。另报 API 总时间用于排查启动开销，但不混作 kernel 指标。

建议保留不少于 30 次有效样本，并同时报告 mean、median、min、max、P90；预热次数记录到元数据，最终依 G0-07 的批准协议执行。每轮重置 A/B，workspace 与 Device 指针数组在采样期间复用。批量输入的矩阵须占有各自可写存储，不能用别名指针复用同一矩阵来规避容量。

#### 7.2 任务书代表规格预算

| 编号 | 接口与规格 | GPU 平均 ms | NPU 上限 ms |
|---|---|---:|---:|
| P-01 | potrf n=1024 LOWER | 0.7900 | 2.2571 |
| P-02 | potrf n=4096 UPPER | 6.7918 | 19.4051 |
| P-03 | potrf n=2048 LOWER | 1.5434 | 4.4097 |
| P-04 | potrs n=1024 nrhs=1 UPPER | 0.2235 | 0.6386 |
| P-05 | potrs n=4096 nrhs=32 LOWER | 2.9488 | 8.4251 |
| P-06 | potrs n=2048 nrhs=8 LOWER | 1.0867 | 3.1049 |
| P-07 | potri n=1024 LOWER | 1.4534 | 4.1526 |
| P-08 | potri n=4096 UPPER | 17.8843 | 51.0980 |
| P-09 | potrfBatched n=32 batch=102774 LOWER | 2.9648 | 8.4709 |
| P-10 | potrfBatched n=128 batch=46256 LOWER | 18.1601 | 51.8860 |
| P-11 | potrsBatched n=32 batch=99659 LOWER | 1.5953 | 4.5580 |
| P-12 | potrsBatched n=128 batch=45897 LOWER | 6.0311 | 17.2317 |

表内上限为 GPU 原值除以 0.35 后四舍五入展示；程序判定使用未舍入值。这 12 条用于早期优化检查，全量性能仍需追溯 768 条原始记录及其 762 个已匹配规范 case。六条无 GPU 基线的 std 用例列为精度项和性能参考待补项，不编造耗时、不默认通过。[S1 §3.3、S3—S4]

verify_perf.py 的退出码 0 仅表示比值计算完成，脚本本身不设 0.35 门槛，而且可能存在未测基线。因此需新增独立汇总器，检查 case 集合完整性、正有限耗时、baseline matched、实际计时口径以及逐条阈值，并保存原始 profiler 到报告的映射。[S5]

#### 7.3 优化决策顺序

第一轮看正确性和算法复杂度，排除把全部大矩阵放进顺序标量算法。第二轮看小矩阵启动开销、批量负载分配和指针表访问。第三轮优化大矩阵面板、TRSM、尾部更新、搬运重叠与工作区复用。每次优化均在固定输入和版本上复测精度与确定性；任何通过降低精度取得的收益，必须同时满足已冻结的数值标准。

优先使用早期 P-02、P-08、P-10、P-12 试验发现大矩阵与批量瓶颈，并覆盖小 n、大 batch 端点。单个 case 提速后复查其相邻尺寸与 UPPER 路径，防止 tiling 分支只对已知规格有效。

## 实施计划与交付

### 8 开发排期与阶段完成标准

下表按一名主开发者顺序工作估算。测试与文档随阶段推进；“第几日”指有效工作日，不含等待算力、任务方回复及公共假期。若增加测试协作者，可并行准备执行器和报告，但不预设已有人员到位。

| 阶段 | 工作日 | 主要工作 | 阶段完成标准 |
|---|---|---|---|
| M0 基线与环境 | 1 至 3 | 处理 G0，冻结仓库/CANN/任务包，验证 950PR 最小 kernel 与 profiler | 有版本清单和最小设备日志；验收争议有明确处理状态；足够容量已落实 |
| M1 接口与测试骨架 | 4 至 7 | 七个 API 的契约设计、C ABI、handle/stream、类型与错误表、执行器格式 | C/C++ 消费者可编译链接；参数与空问题用例可执行；模拟件仅完成格式自查 |
| M2 复数组件与分解 | 8 至 15 | 三角访问、确定性归约、面板 POTRF、分块更新、LOWER/UPPER | Cpotrf 在 950PR 通过目标精度与 info；padding 和 n=4096 可运行；输出首轮性能曲线 |
| M3 求解 | 16 至 19 | 单/多 RHS、两次共轭 TRSM、端到端分解求解 | Cpotrs 覆盖 nrhs=1 至 128 关键值、padding、因子只读与非默认 stream |
| M4 求逆 | 20 至 24 | 三角求逆、三角乘积、零对角、workspace | Cpotri 精度、残差和奇异 info 通过；n=4096 有可追溯 profiler |
| M5 批量接口 | 25 至 31 | Device 指针表、小矩阵调度、大矩阵协作、逐矩阵 info | 两个批量接口贯通原规格；混合失败、非连续指针、百万 batch 小 n 通过 |
| M6 全量验收准备 | 32 至 37 | 779 项主集及追加用例、全部性能规格、必要优化和回归 | 无遗漏用例；正式性能规格逐条达标；确定性/stream/ABI 回归通过；未达标项继续修复后才进入最终交付 |
| M7 文档与提交包 | 38 至 40 | 设计、五份接口说明、自验步骤、报告、日志和版本锁定 | 在干净环境复现；交付清单齐全；五接口整体可评审 |
| 缓冲 | 41 至 45 或 50 | 性能难点、争议变更、环境问题及评审修改 | 用于已识别风险，不以压缩全量测试补偿延期 |

关键依赖为：M0/M1 → Cpotrf → Cpotrs/Cpotri；批量分解复用面板与更新，批量求解复用三角求解，最终五接口共同进入 M6。M2 期间安排一个小矩阵批量调度实验和大 n 更新实验，及早判断性能路线，避免直到 M5 才发现并行方案不可行。

若 M2 的大矩阵分解或批量实验显示性能路线不可达，应在进入完整实现前调整块算法和资源分配。若只是 LFS 或数值标准未回复，可继续不依赖该事项的编码和探索测试，但 M6 的正式验收准备必须等相关基线落实。

### 9 资源和工程产物安排

#### 9.1 环境资源

需要 Linux 950PR 开发环境、已确认配套的驱动和 CANN Toolkit、AscendC/CATLASS 组件及 msprof。正式验收必须在目标 NPU 环境执行，资料整理环境不能替代硬件验收环境。GPU 基线复采仅在任务方要求或需要诊断时安排，不以未经说明的新 GPU 数据替换已发基线。

CPU 参考环境按 manifest 记录准备 Python 3.12.11、NumPy 2.3.3、SciPy 1.16.2、scipy-openblas 0.3.30 或任务方认可版本，另记录线程数、平台和实际 BLAS。大批量用例需按完整输入、DUT 输出、临时副本和 Python 解压开销估算 Host/Device 峰值；建议先进行单条最大用例容量预演，再决定并发度和磁盘留存策略。该容量估算不另设任务书以外的内存达标门槛。

每个 case 计算完即保存必要结果和证据，避免把所有大数组同时驻留内存。原脚本若必须一次装入完整 out32，应为其预留实际容量，任何分段校验方案均另行验证等价性，不在不说明的情况下替换官方路径。

#### 9.2 建议文件组织

公开头文件使用 include/cann_ops_solver.h 和 include/cann_ops_solver_common.h。实现按 src/cpotrf、src/cpotrs、src/cpotri、src/cpotrf_batched、src/cpotrs_batched 划分；950 特有 kernel 在必要时置于对应 arch35 子目录，共用组件放在仓库现有 utils 体系中的明确子目录。

测试与文档对应五个接口建目录，增加 C API 契约、精度执行器和 profiler 汇总功能。包内目录名为 cpotrfbatched/cpotrsbatched，C++ 工程建议名带下划线，测试映射显式维护，避免大小写和目录别名造成漏测。README 与 docs/api_list.md 同步更新。

### 10 主要风险与处理

| 风险 | 触发信号 | 应对与验证 |
|---|---|---|
| 验收口径漂移 | 任务书、脚本、补充材料给出不同阈值 | 以 G0 问题单冻结版本；对差异双报，保留原包，不擅自替换裁决器 |
| 复杂数值误差 | 抵消、UPPER 或大 n 的残差异常 | 隔离共轭/三角组件，固定归约，检查实际因子和原矩阵，逐阶段对照高精度结果 |
| 性能不足 | 大矩阵尾部更新占主导，或小矩阵耗时接近启动成本 | 早期 profiler 分段归因，分别优化块算法与批量调度，保留相邻 shape 回归 |
| Device 指针表误用 | 连续布局通过、打乱地址后失败 | 独立分配、随机排列和不同间隔测试；检查指针位宽与 64 位偏移 |
| 内存和生命周期 | 大 case 分配失败，异步释放错误或多流污染 | G0 容量落实、逐 case 预算、stream 隔离、显式完成后释放，不以缩 batch 冒充验收 |
| info 与确定性失败 | 错误下标不稳定、单矩阵失败影响整批 | 固定最小下标选取、每矩阵状态、重复输入比特比较，禁止无序浮点聚合 |
| 上游兼容回归 | 新 C 类型/枚举导致旧接口重复定义或链接变化 | 单一类型声明，C/C++ 消费者测试，旧接口代表回归，保持原 ABI |
| 排期依赖 | 算力不足、基线补发延迟、G0 迟迟未关闭 | 更新依赖与关键路径，先做独立工作；发布新的预计日期，不能删测试来维持原日期 |

### 11 最终交付与验收清单

| 交付项 | 必须包含的内容 | 完成依据 |
|---|---|---|
| 设计文档 | API 契约、算法、tiling、workspace、并行、精度与风险 | 经项目评审，所有 G0 结论可追溯 |
| 实现代码 | 五计算接口、两 bufferSize、公共类型、构建接入 | 950PR 构建与运行日志，C ABI、旧接口回归 |
| 测试代码 | 输入生成接入、NPU 执行器、功能用例、精度与性能汇总 | 全量 case 清单、可复现命令及负例自检 |
| 自测报告 | 参数、CPU/NPU 数值、info、精度阈值、逐条性能、版本 | 关联原始输出、Profiler CSV、日志及截图 |
| 文档 | 五个算子 README/API 说明、例程和接口列表 | 与最终代码行为一致，列出已确认的支持范围 |
| 提交目录 | task_submission 下步骤、精度、性能日志与报告 | 文件齐全且可从干净环境复现 |
| 内存材料 | 按任务书模板保留 4.1 和 4.2 文件；说明无独立内存性能门槛 | 是否需要填写峰值由任务方确认；泄漏与越界检查仍记录 |
| 社区交付 | 设计文档评审、个人仓地址/分支、五接口整体 PR 或关联 PR | 提交后按远程实际状态核对，不将 CI 通过等同正式验收 |

任务书要求的提交文件包括 1 自验证步骤说明.md、2.1 精度自验证报告.xlsx、2.2 精度自验证日志.log、3.1 性能自验证报告.xlsx、3.2 性能自验证日志.log、4.1 内存自验证报告.xlsx 和 4.2 内存自验证日志.log。报告以冻结的 commit、任务包哈希和环境版本为索引，不能混用不同实现版本的精度与性能结果。

开发完成的判定是五接口功能、精度、确定性、stream、全部正式性能规格和复现材料均达到约定标准；最终验收通过还需任务方正式结论。本次 PR 处于开发前设计评审阶段；申请算力、实现代码、自测报告和最终验收按上述里程碑推进。

### 12 开工时的第一批任务

1. 根据第 3 节整理任务方问题单，优先补齐两个 LFS 对象并统一精度和性能协议；确认 4G 与原 batch 规格冲突的处理。
2. 在 950PR 固定仓库提交、CANN/驱动、CPU 参考依赖和 profiler 版本；跑最小 Device info 写入及非默认 stream 用例。
3. 固定七个公开接口、参数编号、空问题和状态码表；设计 C/C++ 兼容改动及单一 FillMode 定义。
4. 建立 case_id 到 canonical、bench_key、NPU 输出、精度报告、profiler 和性能阈值的一一对应清单。
5. 先完成 n=1、小 n 面板分解及其复数共轭检查，再进入分块 POTRF；同步做一次批量小矩阵调度实验和最大输入容量估算。

## 附录 资料索引与核对依据

S1：所交付任务包内的 Atlas950_Cpotrf_Cpotrs_Cpotri_CpotrfBatched_CpotrsBatched_task_doc.md。重点为第 2 节接口契约、第 3 节验收、第 4 至 5 节交付；具体差异见第 3 节。

S2：所交付任务包内的 DELIVERY_NOTE.md，以及五组 cases.json、bench_result.json。用于交付形态、原始性能集合和计时来源。

S3：三个单矩阵 package 的 canonical_cases.json、cases/index.json、manifest.json、gen_data.py、verify_accuracy.py、verify_perf.py、perf_baseline.json 与 README.md。以实际 JSON 和脚本为核对证据，README 内复制的固定数字不作为其他接口的用例数量。

S4：两个批量 package 的同类文件及 index.json.gz 指针。cpotrfbatched/package/verify_accuracy.py 第 64 至 66 行为 ε 与容差，第 1338 至 1346 行为固化均值读取及缺失兼容处理。Cpotri 残差输入见其 verify_accuracy.py 第 265 至 306 行。单矩阵容差见 cpotrf/package/verify_accuracy.py 第 47 至 52 行。

S5：所交付任务包内的 PERF_COLLECTION_SUPPLEMENT.md，以及 cpotrf/package/verify_perf.py 第 3 至 20 行和第 141 至 160 行。证明性能辅助件只计算参考比值，退出码不构成门槛通过。

S6：[ops-solver 公开仓库](https://gitcode.com/cann/ops-solver)，只读快照 22bbbe69aa23da2bb8f8e73c66fe1339e3db227e。已核对 README、CMakeLists、src/CMakeLists、两个公开头、handle 内部结构、文件树和贡献指南；公共计算工具仅确认存在，尚未进行目标硬件可用性验证。

S7：[NVIDIA cuSOLVER 官方文档](https://docs.nvidia.com/cuda/cusolver/index.html)，核对日期 2026 年 9 月 28 日。用于交叉核对 potri 与 potrsBatched 原型、批量求解单 RHS 和标量 info，以及 bufferSize 查询的空 Device 输入通则。外部文档仅作接口依据，不替代本任务冻结的验收规则。

S8：对所交付 64 个文件的只读统计、用例集合比对、输入容量计算和 SHA256 核对结果；关键结论已列于第 2、3、6 节。原 ZIP 的 SHA256 为 f5e9458cbc759854b326ebd4efed8e4b433bb1024aaf13186151c22f7c8710d1。核对脚本仅读取 JSON、源文件和指纹，未运行附件中的生成、模拟、验收或构建命令。

本稿已对照社区设计模板的需求背景、需求分析、详细设计、可维可测分析组织内容；模板及目录规范核对基于 cann-ops-competitions 提交 bf574b2d2fa37db6824b0b906b1c6493e1c9a9fb。任务书引用的 opbase 生态精度标准正文尚待补核，实施阶段须冻结其版本，本稿不声称已通过该标准验收。任务包 quickstart 的旧矩阵求逆示例仅作为工程接入参考，不能替代本任务的 Device 指针数组、info、padding 和性能契约。

S9：[社区设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/bf574b2d2fa37db6824b0b906b1c6493e1c9a9fb/04_tasks/01_community-task-2026/resources/design_template.md)与[社区任务目录规范](https://gitcode.com/cann/cann-ops-competitions/blob/bf574b2d2fa37db6824b0b906b1c6493e1c9a9fb/04_tasks/01_community-task-2026/README.md)。本次仅提交设计文档，未执行算子编译、单元测试、集成测试和 950PR NPU 验收。

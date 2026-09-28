# aclblasStrmm A2/A3 算子设计文档

# 需求背景（required）

## 需求来源

9月社区任务-aclblasStrmm算子开发(A2/A3)，任务页：
https://www.hiascend.com/activities/task-center/details/110d2b059b2c4347b59ccae220565ea3 。
任务书 SHA256：bf644903d8781d3506779c92f0d02faadcc9775860d2cae540439021107b9229。
官方任务目录：`04_tasks/01_community-task-2026/tasklist/08-aclblasStrmm-A2A3/`。
本设计按官方 design_template.md 的必需章节组织，采用本任务自己的指标和测试材料。

## 背景介绍

在 ops-blas arch22 上提供 FP32 三角矩阵乘公开接口 `aclblasStrmm`，对标 cuBLAS
`cublasStrmm`，使用 Netlib CBLAS STRMM 作为精度参考。公共声明位于
`include/cann_ops_blas.h`，复用已有 handle、stream 和 workspace 机制。
开发基线为 ops-blas `2c9b9b77bda979c25a6b89726614b082e73226fa`；实现包含从候选仓
`https://gitcode.com/bjisdbvjhbsdvk/ops-blas` 导入的 arch22 适配，再按任务修正参数校验、
alias、精度顺序、小矩阵性能和证据采集。保留原许可证，不将第三方代码声明为自行创作。

# 需求分析（required）

## 需求描述

LEFT：`C = alpha * op(A) * B`；RIGHT：`C = alpha * B * op(A)`。
A/B/C 均为列主序，A 为 uplo 指定三角，UNIT 对角视为 1。trans 仅 N/T，OP_C 按
任务收紧语义返回 INVALID_VALUE。m/n 零维为成功 no-op；alpha=0 不引用 A/B。

## 需求拆解

1. 实现 LEFT/RIGHT × UPPER/LOWER × N/T × UNIT/NON_UNIT 共16种语义组合。
2. 覆盖 lda/ldb/ldc padding、零维、空指针、非法枚举和负维度，支持 B=C 原地写出。
3. FP32 全量输出比较：rtol=atol=2^-13、匹配率>=0.99、逐元素硬界 max(0.01,32ULP)，
   Inf/NaN 位置/类型与 Inf 符号一致；alpha=0 精确零比较。
4. 全部200条性能case执行5次预热、11次有效采样，kernel总时延均值对GPU标杆，ratio>=0.8，
   同时满足任务书5条典型case门槛。
5. 要求A2/910B3和A3验证；旧候选已完成双硬件，新工作区候选A2已测、A3待补。
6. 原始1200条CSV完整保留，另运行同参数正态分布补充矩阵，日志与新增数量分开。

# 详细设计（required）

## 算子分析

### 数学公式、类型和形状

输出为 FP32 m×n 矩阵；LEFT 的 A 为 m×m，RIGHT 的 A 为 n×n。
列主序地址为 `base[col*ld+row]`；lda>=max(1,dimA)，ldb/ldc>=max(1,m)。
仅支持 FLOAT32，无广播，无超出 leading-dimension 语义的通用非连续视图。

## 算子实现

### Host侧设计

先检查handle、枚举、负维度，再做零维quick return；随后检查LD、alpha和C。
alpha=0仅清零active C，保留padding。需要计算时检查A/B非空。
B=C先将B复制到handle workspace，保证后续kernel读取原始B。

| 条件 | 计算路径 | 工作区 |
|---|---|---|
| alpha=0 | 按列active区域清零 | 无三角矩阵工作区 |
| m,n<=32 | 小矩阵AIV，每核完整输出列 | 片上UB暂存 |
| 其他alpha!=1或Device alpha | 有序AIV，按参考BLAS分支更新 | B=C时有B快照 |
| 其他Host alpha=1 | AIV三角预处理 + CATLASS Cube | 有效A副本 |

AIC/AIV核数来自平台查询。Cube路径通过C转置适配：交换m/n、左右乘数和side，
三角操作数随trans选择RowMajor或ColumnMajor源布局，输出RowMajor(n,m,ldc)
落到用户列主序C地址。

三角预处理将非活动区域置0、UNIT对角置1，内部A列步长对齐16个元素，阶数65..128
使用128元素步长以减少窄矩阵未对齐读取。工作区是内部布局，用户输入/输出不改变。
所有下发使用handle stream。2026-09-25修订把有效A副本与B=C快照放入handle工作区，
正常Cube路径不再每次同步并释放临时缓存；同一stream上的调用依序复用该工作区。
工作区大小为 `4*prepared_lda*dimA + (B==C ? 4*ldb*n : 0)`；有序AIV只需B快照。
用户工作区不足返回ALLOC_FAILED，库工作区增长沿用框架同步/扩容机制，切换stream和
销毁handle也保留生命周期同步。工作区就绪时异步返回，调用方读取结果前仍须同步。
首次运行和扩容不承诺Host固定返回时延。8组阻塞stream专项覆盖双调用复用、左右乘、
原地/离席和用户/库工作区，另验证不足空间错误；测试在旧同步库上作为负对照会失败。

### Kernel侧设计

小矩阵按列分配AIV，DataCopy/DataCopyPad搬入UB；16元素对齐的LEFT/N使用向量
乘法、三角掩码与分层规约，其余采用有序计算。極大/非有限结果回到有序路径。
输出经UB搬回，避免相邻列标量GM写回冲突。17行等尾部不能使用该向量分支。

有序AIV保留Netlib STRMM方向相关的更新顺序与零系数跳过，解决alpha大值、规约
及极端数值导致的误差。该路径仍在NPU执行。CATLASS的alpha epilogue在早期A3
曾挂起，最终Cube仅使用alpha=1；没有以提前缩放A放宽舍入差异。

Cube采用FP32 MMAD，禁用HF32。主要tile为128×128，K按64/128/256选取；小LEFT
阶数<=128使用K128配置。CATLASS版本固定为769cd40a8716b28650b6bebb08db4834eea4462f。
没有使用不满足L0容量和L1/L0形状约束的分块。

### 文件与依赖

| 路径 | 内容 |
|---|---|
| blas/trmm/arch22/strmm_host.cpp | 参数检查、workspace与stream下发 |
| blas/trmm/arch22/strmm_kernel.cpp | 小矩阵AIV、有序AIV、A预处理和CATLASS实例化 |
| blas/trmm/arch22/strmm_tiling_data.h | tiling描述与分块选择 |
| test/trmm/strmm/ | 原始CSV、strict golden、采集/审计/复现脚本 |
| task_submission/ | 本设计、自测xlsx、可直接打开的原始日志与结果表 |

完整仓构建依赖CANN9.1.0、CATLASS、ops-tensor固定版本和Netlib BLAS/LAPACK、GTest。
交付附件额外提供只含Strmm真实依赖的独立工程。为减少上传文件条目，将固定版本的
CATLASS依赖头汇编为可直接编译的文本头文件，仅改导出kernel的include指令；这不是
上游官方合并版。原始文件哈希、转换映射、许可证和全部通知保留，独立CMake与该布局
在A2另行构建并回归。双硬件原始测量仍对应仓库正常头文件布局，不冒称A3重测了附件。

## 支持硬件

| 产品 | 支持 | 实测 |
|---|---|---|
| Atlas A2系列，含Atlas 800I/T A2 | 是 | 工作区新候选Ascend910B3，CANN9.1.0完整回归通过 |
| Atlas A3系列，含Atlas 800I A3 | 目标支持 | 历史Ascend910_9382已测；工作区新候选A3待补测 |

## 算子约束限制

除B=C外，C与A/B不能重叠。测试及任务要求Host alpha；Device alpha为候选附加路径，
不以未单独覆盖的附加能力扩大本次验收结论。尺寸上限的已测范围为精度2048、PF4096，
并不是对任意极大int形状的内存可用性保证。arch35非本次设备验收范围。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 本任务采用口径 | 来源 |
|---|---|---|
| 精度 | rtol=atol=2^-13，ratio>=0.99，max(0.01,32ULP)，完整C比較 | §3.2 |
| 性能 | 全kernel任务时长之和，5预热+11有效，算术平均，GPU/NPU>=0.8 | §3.3、§4、§7、配套README |
| 内存 | 公开ACL分配请求量高水位、释放余额、设备前后点采样 | §4自测报告要求 |
| 参考 | Netlib BLAS STRMM 3.10.0，固定原始CSV和baseline | §3.1、§3.5 |

2026-09-25工作区新候选async_closeout01在A2完成1206/1206功能（1200CSV+6补充顶层）、
正态1200/1200、性能200/200、内存200/200；最低性能倍率0.93100790（TC_PF_1011）。
8组异步分支均在前置stream门控释放前返回，输出4745元素/组与padding292元素/组全量正确。
内存为每case新建到销毁独立handle的完整生命周期，包含输入输出和工作区；全部请求量归零到基线。
新候选A3尚未执行。历史A3 build43和A2 build01的1204/1200/200/200结果以及旧ZIP保持原身份，
其最低倍率0.80312722/0.91171365不能作为新候选A3已通过的依据。
日志含参数、种子、比较数量、误差、实际/期望值与状态码；原件、失败轮次和源码身份保留。
性能使用系统msprof设备任务时间通道，格式与任务示例msprof op不同，原始PROF数据及
差异说明同时交付。内存请求高水位最大256MiB，不声称为物理HBM连续峰值。

## 兼容性分析

复用公共API声明，增加arch22实现。共享golden/参数解析的改动用于active区域比較、
严格阈值及空LD推导，arch35硬件未在本任务验证。公开接口兼容性以任务N/T、状态码
和column-major语义为准。代码评审、CLA、CI和合入须以平台真实状态单独报告。

# 需求背景（required）

## 需求来源

本设计对应 **9月社区任务-MatmulLayerNormMatmul算子开发**，任务编号 `09-45`，目标为 Ascend 950，开发语言为 Ascend C，算子代码拟提交至 CATLASS。设计文档按社区模板提交到本目录，评审通过后开展代码开发；提交验收前完成设计文档合入。

需求依据为本任务下发的 `MatmulLayerNormMatmul_task_doc.md` 与 `MatmulLayerNormMatmul_测试集.csv`，不是七月同名任务。官方任务索引见 [社区任务列表](../../../../docs/README.md)，文档结构依据 [设计模板](../../../../resources/design_template.md)。

本文是实现前的设计方案。片上容量根据 CATLASS 源码核算，TileShape 为初始候选，未声称已完成硬件精度测试、性能达标或设计评审。源码参考基线为 CATLASS `7653472f11a17c3101d0964574c4001684dcd457`，后续实现时记录实际采用的版本。

## 背景介绍

### 算子功能与现状

算子依次计算 `A0 @ B0`、逐行 LayerNorm、归一化结果与 `B1` 的矩阵乘法。小算子拼接需要多次 kernel launch，并将两份中间矩阵写入和读出 GM。融合实现的目标是将中间矩阵保留在片上，由一个混合 Cube/Vector kernel 完成完整计算，只将最终结果写回 GM。

LayerNorm 依赖一行全部 `N0` 个元素。不能将普通 Matmul 的每个 N tile 独立归一化；也不能直接照搬 Flash Attention 的 online softmax 递推。本方案以行块为归属单位，先收齐整行，再归一化；当整行不能驻留时使用统计与重算两阶段方案。

### 已有组件的使用范围

- CATLASS Ascend 950 的 `CopyL0CToUBTla` 支持 `SPLIT_M`，可把 Cube 结果沿 M 分给配对的两个 AIV。
- `Epilogue::Tile::CopyUb2L1Tla` 提供 Ascend 950 的 UB RowMajor 到 L1 zN 搬运，供第二次 Matmul 消费。
- `49_ascend950_flash_attention_infer` 展示 Cube→Vector→Cube 的片上通路与同步；复用通路和分层方式，不复用 softmax 数学逻辑。
- 任务书所列 `44_quant_matmul_full_loadA_tla` 是量化样例，只参考左矩阵驻留思想；不能直接沿用其 INT8 字节数和 K 维限制作为本任务 FP16 约束。

# 需求分析（required）

## 需求描述

单次 kernel launch 完成 Matmul→LayerNorm→Matmul。唯一对外输出为 FP16 `C1`；`C0`、归一化矩阵、mean、variance 不作为接口输出。完整中间矩阵不得写回 GM；内部临时空间由算子管理。

| 参数 | 方向 | 逻辑形状 | 数据类型 | 物理布局 |
| --- | --- | --- | --- | --- |
| A0 | 输入 | `(M0, K0)` | FP16 | RowMajor，ND |
| B0 | 输入 | `(K0, N0)` | FP16 | ColumnMajor，ND |
| B1 | 输入 | `(N0, M1)` | FP16 | ColumnMajor，ND |
| gamma | 输入 | `(N0,)` | FP32 | 一维 ND |
| beta | 输入 | `(N0,)` | FP32 | 一维 ND |
| C1 | 输出 | `(M0, M1)` | FP16 | 本设计采用 RowMajor，ND |

`epsilon` 固定为 `1e-6`。不增加 bias、激活函数、广播或其他输出。物理 ColumnMajor 不改变 B0/B1 的数学形状；测试输入不能用错误的转置替代布局约定。

## 需求拆解

1. 给出完整 K0 归约、完整 N0 行统计和第二次 Matmul 的依赖与同步。
2. 采用 FP32 矩阵累加和 FP32 行统计，明确中间类型转换，避免 TileShape 改变计算语义。
3. 对已给定 116 个形状实现驻留路径；另设计分块重算路径，避免把 `N0<=8192` 当成接口约束。
4. 接入 optest，原始测试集全部通过；接入 ATK，至少 200 个泛化条目通过，覆盖尾块与数值边界。
5. 在 Ascend 950PR 上使用 `msprof op` 测量，满足任务书整体加速比要求，记录各方案和 TileShape。
6. 提交设计文档、README、代码与正式测试接入；验收材料包含自验证报告、运行步骤和日志证据。

# 详细设计（required）

## 算子分析

### 数学公式

对 `0<=m<M0`、`0<=n<N0`、`0<=j<M1`：

```text
C0[m,n] = sum_k A0[m,k] * B0[k,n]
mean[m] = sum_n C0[m,n] / N0
variance[m] = sum_n (C0[m,n] - mean[m])^2 / N0
Z[m,n] = (C0[m,n] - mean[m]) / sqrt(variance[m] + 1e-6)
         * gamma[n] + beta[n]
C1[m,j] = sum_n Z[m,n] * B1[n,j]
```

方差采用总体方差，分母为 `N0`，不是 `N0-1`。补齐列不计入 mean/variance 的计数；第二次 Matmul 的 K 维尾部则以零填充。

### 支持数据类型与精度路径

外部类型遵循上表。两个 Matmul 使用 FP16 输入、FP32 累加，最终 C1 转 FP16。本设计首先对齐任务书小算子组合的通常类型链：第一个 Matmul FP32 累加后转 FP16，得到片上的 C0；C0 转 FP32 计算 LayerNorm，gamma/beta 在 FP32 参与运算；Z 转 FP16 后进入第二个 Matmul。

中间 FP16 转换是显式设计选择，不是任务书已规定的中间精度。实现初期应对照正式 golden 核实这一选择，并同时使用高精度参考检查累计误差；不能只检查与自建低精度参考一致。若评审要求 C0 保持 FP32，使用较小行块或分块重算路径，并重新核算 UB；不将 FP16 中间值当成 FP32 结果。

对驻留路径，mean 用 FP32 分片归约后合并，variance 对中心化差值平方再次归约，避免直接用 `E[x^2]-E[x]^2` 产生严重消减。`rsqrt(variance+epsilon)` 及 affine 运算在 FP32 完成。初始实现不依赖额外近似；是否使用快速指令以官方精度测试为依据。

### 支持形状与布局

必须满足 `A0.shape[1]=B0.shape[0]`、`B0.shape[1]=B1.shape[0]=gamma.numel=beta.numel`，输出形状为 `(A0.shape[0], B1.shape[1])`。不要求逻辑维度为 tile 的倍数。

原始 CSV 用 GB18030 可正确解码（GBK 亦兼容），包含 116 个唯一形状，idx 为 1～116：

| 维度 | 给定测试集取值 |
| --- | --- |
| M0 | 128、512、1024、2048 |
| K0 | 768、2048、4096、8192 |
| N0 | 2048、3072、4096、8192 |
| M1 | 768、2048、4096 |

这不是四维完整笛卡尔积，也不是支持范围上限。最大 C0 有 16,777,216 个元素，因此不能整矩阵驻留片上。

紧致存储时的元素偏移为 `A0[m*K0+k]`、`B0[n*K0+k]`、`B1[j*N0+n]`、`C1[m*M1+j]`。接口同时检查 leading dimension、张量大小与存储边界；有行/列间 padding 时按合法 leading dimension 搬运。任务书未定义任意 strided view 的额外支持，本设计不在计时路径中隐式调用 `.contiguous()`，正式接口范围按任务布局与评审结果确认。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

host 负责类型/形状/布局校验、架构查询、资源预算、选择已编译的 tiling specialization、申请内部 workspace 并发起一个混合 kernel。CANN 版本依据 [CATLASS 基线 README 的版本兼容表](https://gitcode.com/cann/catlass/blob/7653472f11a17c3101d0964574c4001684dcd457/README.md)；该基线的 Ascend C 950 路径最低要求 CANN 9.0.0。具体驱动、固件、TorchNPU 与 CANN 配套版本在环境验证阶段固定。

##### 1. 分核策略：

以 `(rowBlock, outputColumnGroup)` 为独立任务。默认一个行块负责全部 M1 输出列，行块之间无共享写入、无需全局屏障。AIC 负责两个 Matmul，配对两个 AIV 各负责该行块的一半行；沿 M 分工使每个有效行的统计归属唯一。

若 `ceil(M0/BM)` 小于可用 AIC 数，可沿 M1 增加任务组：每组独立重算相同行块的 MM1/LN，只写自己不重叠的 C1 列区间。该方案用冗余计算换取核占用，不采用 GM 中间矩阵共享或跨核原子归约。仅在实测收益成立时启用，尤其关注 M0=128 的形状。

任务按实际可用核数进行轮转分配。尾行的两个 AIV 仍按物理 tile 协议参与同步，掩码禁止无效行读写；不能因一个 AIV 无有效行而跳过对端正在等待的通知。

##### 2. 数据分块和内存优化策略：

驻留路径保留一个行块的完整 C0，不保留整个矩阵。令 `Na=align_up(N0,16)`，初始候选如下：

| N0 场景 | BM | MM1 N tile | L1 K tile | L0 K tile | MM2 输出列 tile |
| --- | ---: | ---: | ---: | ---: | ---: |
| 不大于 2048 | 64 | 128 | 256 | 128 | 128 |
| 2049～4096 | 32 | 128 | 256 | 128 | 128 |
| 4097～8192 | 16 | 128 | 256 | 128 | 128 |

各候选必须通过容量条件后才能选中；表中范围不是硬编码容量证明。小尺寸、非对齐尺寸与大 N 允许降低 BM、缩小 tile 或进入重算路径。编译期检查布局对齐、缓冲数量，host 用实际平台容量与相同公式复核。

CATLASS `Arch::Ascend950` 定义每个 AIV 的 UB 为 248 KiB，AIC L1 为 512 KiB，L0A/L0B 各 64 KiB，L0C 为 256 KiB。以 `N0=8192, BM=16` 为例：

| 资源 | 缓冲及生命周期 | 预算 |
| --- | --- | ---: |
| 每 AIV UB | 8 行完整 C0，FP16，MM1 至 LN 完成 | 128 KiB |
| 每 AIV UB | 两个 MM1 FP32 收件 tile，`2*8*128*4` | 8 KiB |
| 每 AIV UB | FP32 分片、归约、gamma/beta 分片、统计量及 padding 的统一预算 | 不超过 48 KiB |
| 每 AIV UB | 上述合计 | 不超过 184 KiB |
| AIC L1：MM1 阶段 | A/B 双缓冲，`2*(16*256+256*128)*2` | 144 KiB |
| AIC L1：MM2 阶段 | FP16 Z 全行块 `16*8192*2` + B1 双缓冲 `2*256*128*2` | 384 KiB |
| AIC L0A | 两个 `16*128` FP16 tile | 8 KiB |
| AIC L0B | 两个 `128*128` FP16 tile | 64 KiB |
| AIC L0C | 两个 `16*128` FP32 输出 tile | 16 KiB |

最大 BM=64 的候选：MM1 L1 为 192 KiB，L0A 为 32 KiB，L0C 为 64 KiB；N0=2048 时每 AIV 完整 C0 为 128 KiB、收件区 32 KiB、临时区上限 48 KiB，合计 208 KiB。N0=3072/BM=32 的驻留量更小。所有数值按实际分配后的字节对齐再次检查，临时区若超预算必须减小 tiling，不能依靠编译器隐式分配剩余空间。

MM1 和 MM2 的 L1 按生命周期复用，所以资源条件取两个阶段的峰值，而不是同时放置两阶段的全部缓冲。MM1 的 L0C 只保存 `BM*128` 的列 tile；并不尝试保存 `BM*N0` 的 FP32 完整结果。gamma/beta 以列片搬入，不为每个 AIV 额外分配完整 N0 的两份 FP32 向量。

归一化后以分片 UB→L1 拷贝构建完整 Z。驻留路径通过容量与生命周期检查后，Z 在该任务的全部输出列块计算结束前保持驻留；MM2 循环加载 B1 的 K/列片。容量不满足则切换重算路径，不扩展到 GM 中间矩阵。L1 驻留 Z 不做整块双缓冲。基线调度不同阶段串行复用资源，后续只有在重新证明生命周期与容量后才增加跨行块重叠。

C0 的完整 FP16 行始终保存在对应 AIV 的 UB，LayerNorm 直接读该缓存，没有 L1→UB 的 C0 回读。L1 在 MM1 阶段放输入 A/B，在 MM2 阶段放归一化后的 Z 和 B1；`MM1L1Free` 只控制这两阶段的复用。

##### 3. tilingkey规划策略：

以 `ResidentRows`、`RecomputeRows` 区分驻留与重算模式，再由 BM、N/K tile、缓冲级数、输出列分组数形成编译期 specialization。`mode` 是设计标识，不预设未经注册的数值 ABI。

运行时参数包括 `M0/K0/N0/M1`、leading dimensions、有效尾块、任务数、输出列区间与内部缓冲描述。所有地址和字节规模使用足够宽的整数并检查溢出。精度路径、输入布局和 `epsilon` 不随形状静默改变。

#### 3.2.2 kernel侧设计：

##### 驻留路径的数据流

```text
A0/B0 (GM) -> L1 -> L0A/L0B -> MM1 (FP32 L0C)
    -> Fixpipe / SPLIT_M -> 两个 AIV 的 FP32 收件区
    -> FP16 C0 行缓存 (UB，覆盖完整 N0)
    -> FP32 mean、中心化 variance、affine (UB)
    -> FP16 Z 分片 (UB) -> CopyUb2L1Tla -> Z (L1)
    -> Z/B1 -> MM2 (FP32 L0C) -> FP16 C1 (GM)
```

1. AIC 对 N0 的每个列 tile 遍历全部 K0，得到完整 MM1 结果；每个输出累加器只在第一个 K tile 清零。
2. MM1 结果沿 M 拆给配对 AIV。AIV 等待拷贝完成，执行规定的 FP16 转换，写入自己的行缓存；收到全部 N0 列片后才计算统计量。
3. AIV 遍历有效列计算整行 mean，再遍历缓存计算中心化 variance，最后逐片计算 `Z`。统计时所有尾部 padding 都被排除；N0=1 得到零方差，Z 等于 beta。
4. 两个 AIV 分别将各自行的 Z 写入 L1 的互不重叠行区间；行尾/列尾补零后通知 AIC。归一化后的 padded 列必须写零，不能对 padded C0 应用 gamma/beta 后作为 MM2 输入。
5. AIC 等待两侧都准备完成，按输出列 tile 遍历 N0，完成 MM2。一个 C1 tile 的所有 N0 分片累加完成后才转换并写出；不在 GM 累积部分 C1。
6. 等 MM2 消费完 Z，释放该任务的 L1/UB 资源，再执行下一个任务。

##### 同步与缓冲所有权

使用 CATLASS 950 示例已有的跨核事件及同核 pipe 事件组织生产者/消费者。下表是逻辑事件，实际事件 ID 由实现统一分配，避免与 BlockMmad 内部事件重用：

| 事件 | 生产者→消费者 | 必须保证的条件 |
| --- | --- | --- |
| `C0Ready[slot]` | AIC Fixpipe→两个 AIV | 结果已写入 UB 后才能启动 vector 读取 |
| `C0Consumed[slot]` | 两个 AIV→AIC | 两侧均已复制/转换完毕后，才覆盖该收件 slot |
| `MM1L1Free` | AIC→两个 AIV | MM1 最后一次 L1 读取完成，才能复用 L1 放 Z |
| `ZReady[subBlock]` | AIV 搬运 pipe→AIC | 对应 UB→L1 拷贝已完成，而非仅完成 vector 运算 |
| `ZConsumed` | AIC→两个 AIV | 最后一个 MM2 读取完成后才复用 L1 Z 区 |

UB 上的 Z 临时片必须等搬运 pipe 读完才能覆盖。每个双缓冲 slot 在首次生产前初始化可写状态；复用遵循完整握手，退出前排空在途操作。事件方案只在同一个 AIC/AIV 配对内同步，不使用全设备自旋屏障。稀疏有效行和最后一个任务仍执行成对的事件收发。

##### 大 N 的分块重算路径

当完整 C0 或 Z 无法通过片上容量检查时，仍保持单次 launch，不把它们写入 GM：

1. 每个独立任务负责一个行块和一个输出列 tile。第一遍按 N tile 计算 MM1，逐行用 FP32 Welford/Chan 合并 `(count, mean, M2)`，只保留行统计量。
2. 对含 `n_b` 个有效元素的分片统计 `(mean_b,M2_b)`，与累计状态 `(n_a,mean_a,M2_a)` 合并：`delta=mean_b-mean_a`，`mean=mean_a+delta*n_b/(n_a+n_b)`，`M2=M2_a+M2_b+delta^2*n_a*n_b/(n_a+n_b)`；第一片单独初始化，最终 `variance=M2/N0`。
3. 第二遍按相同 K 归约顺序重算每个 C0 列片；两遍均执行与驻留路径一致的 FP16 C0 转换。使用第一遍的全行统计量归一化，经 UB→L1 交给 MM2，立即累加到当前 C1 输出 tile 的 L0C。两遍均在同一个融合 kernel 内执行。
4. 重算 MM1 的 L0C 与待完成 MM2 的 L0C 分配不同区间，禁止 MM1 初始化覆盖 MM2 累加器；两个阶段的 MM1 A/B 与 MM2 Z/B1 缓冲分时复用，按峰值预算。
5. 全部 N tile 完成后写出该 C1 tile；其他输出列 tile 作为独立任务执行。该模式重复 MM1/LN，代价较大，是容量泛化路径，不预先宣称性能收益。

两条路径均不允许以多 kernel 拼接或完整 GM C0 缓冲作为“融合”降级方式。Welford 合并顺序会与驻留路径产生浮点差异，必须独立覆盖精度回归。

##### workspace 与边界处理

mean、variance、Welford 状态和归约 scratch 由内部缓冲管理，放在本任务 AIV 的 UB/寄存器，不通过 API 返回。方案不需要用户可见的 GM 中间张量；若 CANN launch 接口需要框架 workspace，则单独申请并记录大小、用途，不能将 C0/Z 隐藏在该区。将“内部 workspace 管理”解释为内部临时空间的生命周期管理，提交评审时明确这一点。

- K0/N0/M1 尾块在片上按指令布局补齐；GM 仅搬运有效元素，禁止越界读。统计归约只计真实 N0。
- 空维度是任务书未明确的边界语义，以下作为评审方案：M0=0 或 M1=0 返回正确形状的空输出，不发起无工作量 kernel；K0=0 且 N0>0 时 C0 为零，Z 为 beta，在同一融合 kernel 内跳过 MM1。正式参考确认后纳入 wrapper 与测试。
- N0=0：逐行 mean/variance 公式无定义，不进入除法；空归一化与后续零长度矩阵乘法的最终接口行为需与正式参考框架对齐并单独测试，不能依靠除零生成结果。
- 非有限输入按参考语义传播和检查；禁止将 NaN 强行置零以通过精度测试。验证 FP16 中间溢出、近常数行与异常 gamma/beta。

##### 组件划分与接入

拟在 `experimental/matmul/matmul_layer_norm_matmul/` 添加 CMake、样例入口、README 和 optest 测试件；必要组件位于 `include/catlass` 的 Kernel/Block/Tile 分层。新增 LayerNorm 行块 epilogue；复用现有 tile copy 和 MMAD 基础组件，避免直接修改公共组件的默认行为。

`tests/optest` 接入覆盖 Python API/loader、C++ 注册、adapter、kernel runner 和 JIT 参数。接口建议为 `matmul_layer_norm_matmul(a0, b0, b1, gamma, beta) -> c1`，固定 epsilon，不返回中间张量。沿现有 prebuilt/JIT 混合核接入方式选择，并验证一次调用只 launch 一个非空融合 kernel。ATK 从 optest 接入扩展，正式代码评审时以仓库当时的文件布局为准。

## 支持硬件

| 支持的芯片版本 | 本次范围 |
| --- | --- |
| Ascend 950 | 功能实现目标；使用 3510 架构组件 |
| Ascend 950PR | 与任务标杆一致的性能验收平台 |

不将 950 的 UB→L1 通路默认推广到 A2/A3。950DT 的实测结果不得直接替代任务中 950PR 的性能结论。

## 算子约束限制

遵循任务书的输入类型、维度关系、布局与固定 epsilon，不额外限定测试集中的维度值。接口检查设备一致性、形状匹配、合法 leading dimension 与地址范围。输入和输出不重叠；内部 tiling 的对齐要求通过 padding 处理，不转化为外部形状限制。最大可分配规模由设备内存和接口索引范围决定，不能以“片上放不下”静默拒绝所有大 N 输入。

评审需确认两项原文未明确的语义：中间 C0 是否按小算子 FP16 输出舍入，以及非连续输入/空 N0 的正式接口范围。确认后同步 golden、wrapper 检查与本文，不能让测试框架和 kernel 各自解释。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 验收与记录方式 | 标准来源 |
| --- | --- | --- |
| 给定精度用例 | 116 例 optest 全部通过，保留 idx、shape、seed、误差指标和日志 | 本任务 CSV、任务书 |
| 泛化精度 | 至少 200 例 ATK 通过；本设计计划 256 个条目，另测特殊输入 | 任务书 |
| 元素容差 | FP16 输出采用 `atol=rtol=2^-9`；逐元素 `abs(actual-golden)<=atol+rtol*abs(golden)` | 生态算子开源精度标准 |
| 整体误差 | matched_ratio 至少 0.99，且满足最大绝对误差硬上限；FP16 表列 `1e-1 or 32*ULP` | 同上；严格沿官方判定实现，不自行用宽松 OR 替代 |
| 性能 | 标杆为 `torch.mm + F.layer_norm + torch.mm`，整体加速比严格大于 1.1 | 任务书 |
| 融合属性 | profiler 核实单次非空调用一个融合 kernel；源码与内存检查确认无 C0/Z GM 读写 | 任务书、设计数据流 |

精度标准读取日期为 2026-09-16，固定来源为 opbase commit `1cc94d9dda647ceb3e83094fa2c0f548d0dc4df2` 的 [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/1cc94d9dda647ceb3e83094fa2c0f548d0dc4df2/docs/zh/ops_precision_standard/experimental_standard.md)，上述数值来自第 2.1～2.3 节，尤其第 2.2 节 FLOAT16 阈值表。表中 `1e-1 or 32*ULP` 是标准原文，本文不另行定义两者选择算法。正式验收时保存实际采用的标准版本，`ULP` 硬上限、非有限值和空输出处理以框架实现及评审口径为准，不以单个 `torch.allclose` 代替整个标准；这些是验收目标，不是本算子的测试结果。

### 测试计划

1. 正式 golden：按照评审确认的中间类型链，使用 PyTorch/TorchNPU 拼接；独立高精度参考使用更高精度的矩阵累加和 LN 统计诊断误差。B0/B1 用正确 ColumnMajor 物理输入，参考计算仍使用其逻辑矩阵。
2. 原始 116 例逐一执行，不只覆盖每个维度的取值集合；结果与 CSV idx 对应。
3. ATK 至少生成 256 个可复现条目：96 个非对齐形状、64 个驻留/重算阈值附近形状、64 个小 M 和长 K/N 形状、32 个特殊数值条目。形状类别去重后不足的必须补齐，不能以重复同一用例凑数。
4. 对常规条目，均匀与正态分布各占一半；按官方标准覆盖取值范围与参数。特殊条目包括零输入、常数行、近常数行、gamma=0、beta 非零、N0=1、大动态范围、NaN/Inf、空维度与合法 leading dimension padding；独立覆盖每个输入角色。
5. 覆盖所有启用的 TileShape、双缓冲 slot 环绕、奇数尾行、K/N/M1 尾块、多核不整除、重复运行与同设备多 stream；sanitizer 检查越界和读写时序。非法类型/shape/layout 用例应明确报错，不计入精度通过数量。
6. 资源检查：编译期和 host 预算检查，核实各阶段不重叠的地址范围；通过小规模 C0→LN→C1 分阶段参考定位错误。调试中间输出仅在独立诊断版本使用，不保留在正式计时/验收实现中。

### 性能测量与优化顺序

使用 Ascend 950PR、固定 CANN/驱动/TorchNPU、代码 commit 与时钟/负载条件。先预热及完成 JIT 编译，再用 `msprof op` 采集；JIT、数据生成、输入布局准备、文件读写不混入 kernel 时延。标杆和融合路径使用相同输入、设备和采集口径；另记录端到端时延与内存申请开销。

CSV 给定标杆范围为 19.005～1188.58 us。报告保留每例标杆/融合时延、加速比、重复次数和波动，并同时提供 `mean(Tbaseline/Tfused)` 与 `mean(Tbaseline)/mean(Tfused)`。任务书的平均口径需要在验收前确认；不以一种统计量替换另一种来宣称达标。

优化顺序为：先验证融合正确性和同步，再按 M/N/K 分布选择 BM 和输出列分组；比较左矩阵驻留、B 缓冲预取、MM1/LN 重叠以及 copy 布局开销。每次改变需记录收益、资源峰值与精度结果。小 BM 的 Cube 利用率和 B0 重复读取可能抵消融合收益，尤其在小 M 场景；当前不预测或承诺已达到 1.1 倍。

### 主要风险与验证门槛

| 风险 | 验证/处置 |
| --- | --- |
| 中间 FP16 舍入与正式 golden 不一致 | 设计评审确认；如改 FP32，同步容量、TileShape、参考与测试 |
| L1 复用或 UB→L1 源缓冲过早覆盖 | 串行生命周期基线先通过，再启用重叠；验证所有 ready/consumed 事件 |
| 降低 BM 后核利用率或重复加载影响性能 | 以原始 116 例测量，按 shape 选择列分组与驻留方案 |
| 大 N 重算成本高 | 单独统计性能，不把该路径的正确性当成性能达标 |
| 非对齐搬运与统计 padding 错误 | 哨兵/边界用例、sanitizer、尾块数量核对 |
| 模板与实际 CANN API 版本不匹配 | 锁定环境和源码 commit；编译并执行基础 copy/MMAD 验证后实现融合 |

## 兼容性分析

新增 CATLASS 算子，不更改现有算子的接口或默认行为。源仓库中的 Ascend 950 特化组件按架构隔离，新增 epilogue 和调度策略需有对应回归测试。设计文档提交到社区任务仓库；算子代码后续提交到 CATLASS，不在本设计 PR 中提交实现、测试报告、二进制或临时脚本。

## 源码参考

以下链接固定到本次核对的 CATLASS commit，供评审复核能力与资源依据：

- [Ascend950 片上资源常量](https://gitcode.com/cann/catlass/blob/7653472f11a17c3101d0964574c4001684dcd457/include/catlass/arch/arch.hpp)
- [L0C→UB SPLIT_M](https://gitcode.com/cann/catlass/blob/7653472f11a17c3101d0964574c4001684dcd457/include/catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp)
- [UB RowMajor→L1 zN](https://gitcode.com/cann/catlass/blob/7653472f11a17c3101d0964574c4001684dcd457/include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp)
- [950 Flash Attention 混合核与事件](https://gitcode.com/cann/catlass/blob/7653472f11a17c3101d0964574c4001684dcd457/examples/49_ascend950_flash_attention_infer/fai_kernel.h)
- [950 Flash Attention 通路设计说明](https://gitcode.com/cann/catlass/blob/7653472f11a17c3101d0964574c4001684dcd457/examples/49_ascend950_flash_attention_infer/flash_attention_infer.md)
- [optest 接入](https://gitcode.com/cann/catlass/blob/7653472f11a17c3101d0964574c4001684dcd457/tests/optest/README.md)
- [社区设计评审与验收流程](https://www.hiascend.com/developer/activities/details/11014a50a8794171a4a08688fd398774#tab2)

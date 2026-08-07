# aclnnBernoulli 算子开发设计文档

# 需求背景（required）

## 需求来源

本任务来自 2026 年 7 月 CANN 社区任务，目标是在 Atlas A2/A3 训练系列产品上，使用 Ascend C 优化 `aclnnBernoulli`。任务要求保持现有 API 和随机采样语义不变，消除小算子拼接产生的中间临时张量，并将 NPU 与 GPU 单 Kernel 路径的 Device 峰值内存差距控制在 5% 以下。

- 任务书：`04_tasks/01_community-task-2026/docs/202607/aclnnBernoulli_task_doc.md`
- 现有源码：`cann/ops-math/random/stateless_bernoulli`
- 计划交付目录：`cann/ops-math/experimental/random/stateless_bernoulli`
- 适配硬件：Atlas A2/A3 训练系列产品
- CANN 版本：CANN 8.5.0 及以上
- 开发语言：Ascend C

## 背景介绍

### aclnnBernoulli 功能

`aclnnBernoulli` 根据标量概率 `prob` 生成与输入 Tensor 相同 shape、dtype 的伯努利随机输出。输入 Tensor 只提供 shape、dtype 和布局信息，其元素值不参与随机采样。对任意输出元素 `i`：

\[
out_i =
\begin{cases}
1, & u_i < prob \\
0, & u_i \ge prob
\end{cases}, \qquad u_i \in [0, 1)
\]

当 `prob=0` 时输出必须全为 0；当 `prob=1` 时输出必须全为 1。同一实现对相同 shape、dtype、`prob`、`seed` 和 `offset` 应产生可复现的结果。

### 现有实现及问题

任务书给出的现有 ACLNN 路径由随机掩码生成、`Fill` 和 `DropoutDoMask` 等小算子拼接完成。多个子算子需要在 GM 中保存掩码或填充值等中间结果，导致 Device 峰值内存增大：bf16/fp32 场景约有 50% 膨胀，fp16/int64 场景约有 25% 膨胀。GPU 对照路径使用单 Kernel，因而不存在同等规模的中间张量。

当前开源 `random/stateless_bernoulli` 目录已经包含 `StatelessBernoulli` 图算子和 Ascend 950 的 `arch35` AI Core 实现，但 A2/A3 尚缺少对应的 AI Core 闭环。L0 路径在非 RegBase 平台不能选择现有 AI Core Kernel，无法直接解决本任务的 A2/A3 内存问题。本次不新建 `arch32` 子目录：A2/A3 新增 Host、tiling data 和 Kernel 源文件直接放在现有 `op_host/` 和 `op_kernel/` 层级，由 SoC 配置与编译条件选择。

### 方案依据与已有原型结果

本设计采用单 Kernel 直接生成输出：Kernel 内部完成随机数生成、概率比较以及 0/1 编码，结果直接写入最终输出，不在 GM 中生成完整尺寸的随机数、掩码或填充 Tensor。输入数据不需要搬入 UB，Kernel 仅消费输出元素总数、概率阈值、`seed`、`offset` 和 dtype 等元数据。

仓库内已有 Ascend C 原型验证了该数据流的可行性：

| 项目 | 当前原型结果 | 结论边界 |
| --- | --- | --- |
| 公共正确性 | `public_v2` 60/60 通过，状态为 `CORRECTNESS_ONLY` | 覆盖 10 种 dtype、rank 0-8、空 Tensor、非连续输入、概率端点、尾块和百万元素压力 shape |
| 运行设备 | Ascend910_9382，CANN 9.0.0 | 属于 A2/A3 原型环境，不替代 CANN 8.5.0 源码仓构建验证 |
| 同源局部性能 | fp32 标量 `prob=0` 为 1.0615x；bool 单元素 `prob=1` 为 1.0833x | 仅两个端点微型场景，不代表完整公共集或任务性能达标 |
| 数据流 | 单个自有 Ascend C Kernel，输入读取 0 字节，workspace 为 0 | 证明可消除完整尺寸 GM 中间张量；尚未形成 GPU/NPU 峰值内存对比 |
| 核心验收 | 尚无 GPU 原始内存日志 | 不声明“内存差距小于 5%”已完成 |

上述原型结果用于支撑设计选择，不作为本源码 PR 的最终验收结果。源码交付后仍需在最终提交上完成 A2/A3 包构建、ACLNN 自测、全量精度/统计验证、性能对比以及 GPU/NPU 峰值内存测量。

# 需求分析（required）

## 需求描述

在不改变 `aclnnBernoulli` 和 `aclnnInplaceBernoulli` 对外接口的前提下，为 Atlas A2/A3 新增自有 Ascend C Kernel，但不创建 `arch32` 目录，并将原有“随机掩码 + Fill + DropoutDoMask”多算子数据流收敛为一次 Kernel 启动：

1. 在 Kernel 内生成与全局线性索引、`seed`、`offset` 关联的确定性随机字；
2. 将标量概率转换为统一阈值，在 Kernel 内完成比较；
3. 按输出 dtype 直接写入精确的 0/1 位模式；
4. 不创建完整尺寸的 GM 随机数、掩码或 Fill 中间 Tensor；
5. 保留现有 Ascend 950 `arch35` 路径，不使本次 A2/A3 交付改变其行为。

## 需求拆解

1. **接口兼容**：保持 ACLNN 两段式接口、参数顺序、错误码和执行器生命周期不变。
2. **语义正确**：输出 shape、dtype 与输入一致，输出值严格属于 `{0, 1}`，正确处理 `prob=0/1`、空 Tensor、合法 `seed` 和 `offset`。
3. **数据类型**：支持 FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BFLOAT16。
4. **shape 与布局**：支持 ND、rank 0-8、空 Tensor 和任务书声明的非连续 Tensor。
5. **内存优化**：连续输出主路径不申请 Device workspace，不产生与元素数成比例的 GM 临时 Tensor；非连续输出路径也不得无条件物化完整尺寸临时结果。
6. **性能要求**：在同一 NPU、同一 CANN、同一输入和同步边界下，性能不低于原算子；小 shape 需要单独控制启动下限。
7. **架构闭环**：`op_def`、`op_host/config/ascend910b`、`op_host/config/ascend910_93`、CMake 计算单元和二进制包路径必须声明一致的 A2/A3 SoC 集合。
8. **验证闭环**：功能、统计分布、确定性、尾块、内存、性能、源码构建和临时安装清理均需有可复现证据。

## 需求模块设计

### 对外 ACLNN 原型

```cpp
aclnnStatus aclnnBernoulliGetWorkspaceSize(
    const aclTensor* self,
    const aclScalar* prob,
    int64_t seed,
    int64_t offset,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnBernoulli(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

原地接口 `aclnnInplaceBernoulliGetWorkspaceSize` / `aclnnInplaceBernoulli` 复用同一 AI Core 语义实现，仅由 L2 层决定最终输出对象和视图关系。

### 分层职责

```text
ACLNN L2
  参数、shape、dtype、prob、offset、输出布局检查
  空 Tensor 快速返回
  连续/非连续输出路由
        |
        v
L0 StatelessBernoulli
  为 A2/A3 选择自有 AI Core 节点
  传递 shape、prob、seed、offset、dtype
        |
        v
op_host 根目录中的 A2/A3 tiling 实现
  计算 totalLength、blockDim、blockLength、tilingKey
  生成概率阈值和 dtype/布局元数据
        |
        v
op_kernel 根目录中的 A2/A3 Kernel 实现
  全局索引 -> 随机字 -> 阈值比较 -> dtype 0/1 编码 -> 最终 GM 输出
```

每一层只处理自身职责：L2 不实现设备计算，Host 不读取输入数据，Kernel 不依赖公共 case 或外部参考实现。A2/A3 与 950 的实现选择依据是 SoC 配置、计算单元和编译条件，不通过新增 `op_host/arch32` 或 `op_kernel/arch32` 目录表达。

# 详细设计（required）

## 算子分析

### 数学公式

设输出元素总数为 `N`，线性索引为 `i`。随机数发生器根据 `(seed, offset, i)` 生成 32 位随机字 `r_i`，概率 `prob` 映射为阈值 `T(prob)`：

\[
out_i = \operatorname{Encode}_{dtype}\left(r_i < T(prob)\right)
\]

其中 `Encode` 将布尔结果转换成目标 dtype 的精确 0/1 位模式。`seed` 与 `offset` 只改变随机序列，不改变 shape、dtype 和内存规划。`offset` 必须非负且满足 `offset % 4 == 0`。

原型使用 24 位概率阈值和模 `2^{32}` 的奇数步长计数置换。源码合入时允许替换为与仓内随机公共组件一致的等价发生器，但必须同时满足：同后端可复现、概率分布通过统计检验、所有 dtype 共享一致的索引所有权、无额外 GM 中间张量。

### 支持数据类型

| self / out dtype | 输出 0/1 编码 | 计划 Kernel 路径 |
| --- | --- | --- |
| FLOAT16 | `0x0000` / `0x3c00` | 大 Tensor 向量比较与 Select；小 Tensor 通用路径 |
| FLOAT | `0x00000000` / `0x3f800000` | 大 Tensor 向量比较与 Select；小 Tensor 通用路径 |
| BFLOAT16 | `0x0000` / `0x3f80` | 以 16 位原始位模式完成 Select |
| DOUBLE | IEEE 754 double 的 0.0 / 1.0 | 64 位通用写回路径 |
| UINT8 / INT8 / BOOL | 8 位 0 / 1 | 8 位通用路径；概率端点可使用批量填充 |
| INT16 | 16 位 0 / 1 | 16 位通用路径 |
| INT32 | 32 位 0 / 1 | 32 位通用路径 |
| INT64 | 64 位 0 / 1 | 大 Tensor Select 后扩宽；小 Tensor 通用路径 |

敏感 dtype 不通过框架或 CPU 兜底。若某个 Ascend C 原语在目标 SoC 上不支持相应 dtype，则使用算子自有的位模式或 cast-compute-cast 路径，并通过编译探针和边界用例验证。

### 支持形状和属性

- 数据格式：ND。
- rank：0-8。
- 元素数：支持 0；空 Tensor 不启动设备 Kernel，也不伪造 0 us 性能数据。
- `prob`：有限数，`0 <= prob <= 1`。
- `seed`：有符号 int64 全域。
- `offset`：非负有符号 int64，且能被 4 整除。
- 输入值：不参与计算；输入只提供 shape、dtype 和视图合同。
- 输出：与输入相同的逻辑 shape 和 dtype，元素严格为 0 或 1。

## 算子实现

### 实现方案总览

核心方案不是在原有拼接图后做临时张量复用，而是将 `Fill + DropoutDoMask` 的有效语义内联到同一个 AI Core Kernel：

```text
原路径：GenMask -> mask GM -> Fill -> fill GM -> DropoutDoMask -> out GM

新路径：index/seed/offset -> RNG -> compare(prob) -> encode(0/1) -> out GM
```

新路径只写最终输出，不读取 `self` 数据，不申请与 `N` 成比例的 Device workspace。比较产生的 packed mask 仅存在于单核 UB 中，生命周期限定在当前 tile，不能落入 GM。

### Host 侧设计

#### 参数与合同检查

L2 层保留一组清晰的合同检查：

1. `self`、`prob`、`out`、`workspaceSize`、`executor` 非空；
2. `self` 与 `out` 的逻辑 shape、dtype 一致；
3. rank 位于 `[0, 8]`；
4. dtype 位于任务书声明的 10 种类型；
5. `prob` 有限且位于 `[0, 1]`；
6. `offset >= 0 && offset % 4 == 0`；
7. 输出可写且 Device/格式满足当前路径要求。

重复的 dtype/rank/shape 判断在公共合同辅助函数中合并，避免 L2、L0 和 Host 各自维护不一致的白名单。L0 启动前只再次验证最终变换后的 dtype、SoC 和 Kernel 注册签名。

#### Tiling 参数

Host 根据运行时平台信息获取 AI Vector 核数，不硬编码设备核数。主要 tiling 参数如下：

| 参数 | 含义 |
| --- | --- |
| `totalLength` | 输出逻辑元素总数 |
| `blockDim` | 实际启用的 AI Vector 核数 |
| `blockLength` | 单核名义处理元素数 |
| `vectorTile` | 向量路径每 tile 256 个元素 |
| `probThreshold` | 由 `prob` 生成的统一整数阈值 |
| `seed` / `offset` | 随机序列身份 |
| `storageWidth` | 输出存储宽度 1/2/4/8 字节 |
| `tilingKey` | 概率端点、向量 dtype、通用 dtype、布局路径选择 |

分核遵循以下规则：

1. 每个活跃核原则上至少处理 256 个元素，避免小 shape 盲目满核；
2. 通用路径按 `64 / sizeof(StorageT)` 个元素对齐，使不同核不共享 64 字节 GM 写入线；
3. 向量路径按 256 个元素对齐；完整 tile 由各核独占，最后一个不足 256 元素的全局尾块由一个固定核处理；
4. 空 Tensor 由 L2 快速返回，不进入普通性能统计。

#### 布局处理

连续输出直接作为 Kernel 的最终 GM 地址。非连续输出不能默认创建完整尺寸的连续临时 Tensor，否则会抵消内存优化目标；设计优先将逻辑 shape/stride 传入 Host 和 Kernel，由 Kernel 将逻辑线性索引映射为实际输出地址。若源码阶段受现有 L0/tiling 接口限制必须使用 `ViewCopy`，该路径应单独记录临时内存并保持为未验收状态，直到满足 `<5%` 内存要求。

### Kernel 侧设计

#### 随机数和阈值比较

每个输出位置的随机字仅由 `(seed, offset, globalIndex)` 决定。各核拥有互不重叠的全局索引区间，因此无需核间同步，也不会因 blockDim 改变而重复写同一元素。

对 `0 < prob < 1`：

1. 生成当前 tile 的全局索引；
2. 混合 `seed` 高低 32 位和 `offset / 4` 得到序列盐值；
3. 对全局索引执行 32 位计数置换；
4. 将随机字与统一阈值比较，得到 UB 内 packed mask；
5. 使用 `Select` 或目标 dtype 位模式将 0/1 直接写入最终 GM。

对 `prob=0` 和 `prob=1`，跳过随机数计算，直接走精确端点填充。端点路径仍由本算子 Kernel 完成，不调用 `Fill` whole-op。

#### Dispatch 设计

| 运行时条件 | 选择路径 | 设计理由 | 未支持时行为 |
| --- | --- | --- | --- |
| `totalLength == 0` | L2 空 Tensor 快速返回 | 无设备工作，不伪造 Kernel 时延 | 合法返回 |
| `prob == 0 || prob == 1` | 按存储宽度的端点填充 | 省去 RNG 和比较，精确生成全 0/1 | 无框架 Fill 兜底 |
| FLOAT16/FLOAT/BFLOAT16/INT64 且 `totalLength >= 256` | 256 元素向量路径 | 使用 Arange、整数变换、float 比较和 Select | 原语不支持则构建失败或显式拒绝该注册，不静默回退 |
| 其余合法 interior 元数据 | 1/2/4/8 字节通用路径 | 覆盖小 shape、DOUBLE、BOOL 和整数 dtype | 完整合同内必须由自有 Kernel 覆盖 |
| 非连续输出 | stride-aware 直接写路径 | 避免完整尺寸连续临时 Tensor | 若未完成则验收失败，不路由 CPU/官方 whole-op |

Dispatch 只使用 dtype、shape/rank、元素数、布局、概率端点、存储宽度和目标 SoC 等合法运行时元数据，不使用公共 case id、公共 shape 白名单、计时特征或输入观察值。

#### UB 和 GM 所有权

原型中每个活跃 block 的 UB 估算如下：

| 向量路径 | 单 block UB 估算 |
| --- | ---: |
| FLOAT16 | 约 4.6 KiB |
| FLOAT | 约 6.1 KiB |
| BFLOAT16 | 约 4.6 KiB |
| INT64 | 约 8.1 KiB |

UB 只保存当前 256 元素 tile 的索引、随机字、比较 mask 和输出片段。GM 中只存在调用者持有的输入/输出及固定大小运行时对象；Kernel 不申请算子 workspace，不产生完整尺寸 mask、随机数或 Fill Tensor。

#### 尾块安全

Host 对完整 tile 和最终尾块做显式划分。完整 tile 使用向量读写长度；尾块只写 `[0, totalLength)` 内的有效元素。通用路径按存储宽度进行 64 字节所有权对齐，最后一个活跃核独占末尾物理写入线。任何对齐优化都不能用调用方的整除判断替代 Kernel 内的尾块边界保护。

## 支持硬件

| 支持的芯片版本 | 本次设计 | 说明 |
| --- | --- | --- |
| Atlas A2/A3（`ascend910b`、`ascend910_93`） | √ | 新增根目录 Host/Kernel 源文件、SoC 配置和打包闭环，不创建 `arch32` 子目录 |
| Ascend 950（`arch35`） | 保留 | 保留现有实现，本次文档不把其计入 A2/A3 验证结论 |

## 算子约束限制

1. `prob` 必须为有限数并满足 `0 <= prob <= 1`。
2. `offset` 必须非负且满足 `offset % 4 == 0`。
3. `self` 与 `out` 的逻辑 shape、dtype 必须一致。
4. 支持 ND、rank 0-8 和任务书声明的 10 种输出 dtype。
5. 相同实现、shape、dtype、`prob`、`seed`、`offset` 必须可复现；不要求与不同后端逐元素产生相同随机比特流。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 输出合同 | shape、dtype 与输入一致，所有值严格属于 `{0, 1}` | 任务书与 ACLNN 接口合同 |
| 概率端点 | `prob=0` 全 0，`prob=1` 全 1 | Bernoulli 数学定义 |
| 统计精度 | `0<prob<1` 时样本均值落入二项分布确定性 6-sigma 区间；并执行固定 seed/offset 的同后端复现性检查 | 随机算子统计验证原则 |
| AscendOpTest | 与原算子精度标准对齐，满足 AscendOpTest 默认阈值 | 任务书 |
| 内存 | 每个验收 case 的 `abs(NPU_peak-GPU_peak)/GPU_peak*100% < 5%` | 任务书核心验收要求 |
| 性能 | 同一 NPU/CANN/输入/同步边界下不低于原算子；以 active window 为主指标 | 任务书 |

## 自验证方案

### 功能与统计验证

至少覆盖以下维度：

- 10 种输出 dtype；
- rank 0-8、空 Tensor、标量、奇数尾块和百万元素压力 shape；
- 连续与非连续 Tensor；
- `prob=0`、`prob=1`、低概率、高概率和普通概率；
- 正负 seed、int64 边界 seed、多个合法 offset；
- 相同 `(shape, dtype, prob, seed, offset)` 重复运行一致；
- 输出值域、shape、dtype 和统计区间全部通过。

随机算子不能用不同 RNG 实现的逐元素 allclose 作为唯一判据。固定 seed/offset 的复现性检查与分布统计检查必须同时存在。

### 内存验证

内存验收在匹配的 NPU/GPU 环境中分别执行，同一 case 必须使用相同 shape、dtype、`prob` 和批次条件：

1. 每次测量前同步设备、清理缓存并重置峰值统计；
2. 预热 3 次，正式测量 5 次；
3. 记录输入/输出字节数、workspace、临时张量峰值、总峰值、设备和工具版本；
4. 逐 case 计算 `abs(NPU_peak-GPU_peak)/GPU_peak*100%`；
5. 每个验收 case 均需 `<5%`，不得只报告平均值；
6. 保存原始日志和汇总表，截图仅作为展示，不替代文本/CSV 证据。

预期连续输出主路径的算子 workspace 为 0，完整尺寸 GM 临时张量为 0。该预期必须由最终源码提交的实测峰值证明，不能仅依据源码结构宣称达标。

### 性能验证

性能对比在同一 NPU、同一 CANN、同一输入和同一同步边界下执行 3 次预热、5 次正式 profiler 测量：

- candidate active window / baseline active window 为主比较；
- kernel sum 仅用于诊断；
- 空 Tensor 只记录功能通过，不以 0 us 计入统计；
- 每个回归 case 必须保留，不用局部均值覆盖；
- 覆盖端点、小 shape、通用整数、向量 float/bf16/int64、尾块和压力 shape。

当前原型的两个端点微型场景已有局部正向结果，但完整性能验收仍待源码落地后重新执行。

### 源码与包验证

最终代码 PR 至少执行：

1. `ascend910b` 与 `ascend910_93` 的 package build；
2. op_host tiling/infershape UT；
3. op_kernel 正确性和尾块 UT；
4. ACLNN 两段式接口样例；
5. 自包含 reviewer rerun 脚本；
6. OAT、格式化、静态检查和 `git diff --check`；
7. 临时 OPP 安装前后备份/恢复检查；
8. `op_def`、SoC config、CMake 和二进制包的交叉签名检查。

## 兼容性分析

- 对外 ACLNN 接口、参数和输出语义不变。
- A2/A3 新增自有 AI Core 路径，Host/Kernel 新文件直接位于现有层级，不创建 `arch32` 子目录；现有 Ascend 950 `arch35` 路径保持不变。
- 不读取 `self` 的数值，符合 Bernoulli 输入只承载 shape/dtype 的语义。
- 连续输出主路径不需要 workspace；调用方仍按标准两段式接口处理 `workspaceSize`。
- 不引入运行时 whole-op fallback。任何未完成的 dtype、布局或 SoC 注册必须显式失败，不得静默切回 AICPU、CPU、框架或官方旧实现后再声称本方案通过。
- 实现边界：A2/A3 路径的输出必须由本设计自有 Ascend C Kernel 生成，不调用 PyTorch/torch-npu 等价算子、CPU/reference、官方 whole-op、AICPU、其他后端或外部包；该条属于实现方式与无 fallback 验收边界，不是算子输入、输出或属性约束。

## 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 24 位概率阈值和 float 比较存在量化边界 | 极端概率下分布偏差 | 覆盖极低/极高概率与大样本 6-sigma 检查；端点走精确路径 |
| 小 shape 的 Kernel 启动下限 | 性能可能低于原路径 | 限制 blockDim，设置单元素/端点专用路径，逐 case 保留回归结果 |
| BFLOAT16、DOUBLE、BOOL、INT64 原语能力差异 | 编译失败或错误编码 | 使用位模式/自有转换路径，针对目标 SoC 做最小编译探针和边界 UT |
| 非连续输出需要地址映射 | 可能引入完整尺寸临时 Tensor | 优先 stride-aware 直接写；若暂用 ViewCopy，则单独计量并在内存达标前保持未验收 |
| 多核尾块共享 GM 写入线 | 越界或覆盖 | 64 字节所有权对齐，单一尾块 owner，Kernel 内边界保护 |
| SoC 注册不完整 | 构建成功但包内缺 Kernel | 同时检查 op_def、config、CMake、binary.json 和 simplified key |
| 原型证据不是源码 PR 证据 | 误报完成 | 在最终提交上重新构建、运行、测量，并保留原始日志；原型数据只作为设计依据 |

## 参考资料

1. `04_tasks/01_community-task-2026/docs/202607/aclnnBernoulli_task_doc.md`
2. `04_tasks/01_community-task-2026/resources/design_template.md`
3. CANN Ascend C 算子开发文档
4. PyTorch `torch.bernoulli` / `Tensor.bernoulli_` 语义说明
5. `cann/ops-math/random/stateless_bernoulli` 现有实现

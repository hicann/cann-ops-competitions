# HyperLogLog 容器（950）设计

作者：gcw_rBObG5J6  
任务：[9月社区任务-hyperloglog容器开发(950)](https://www.hiascend.com/activities/task-center/details/f975dbdd02cb43eaacd0d7e3d3fa8821?menu=guide)  
讨论：[ops-collections #4](https://gitcode.com/cann/ops-collections/discussions/4)  
代码：[个人 ops-collections 仓库](https://gitcode.com/gcw_rBObG5J6/ops-collections)，分支 `codex/hyperloglog-950`

# 需求背景（required）

## 需求来源

2026年9月社区任务，在 Atlas 950 上实现 HyperLogLog（HLL）容器。参考 [cuCollections hyperloglog](https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/hyperloglog.cuh)，按 ops-collections 的纯头文件组织方式开发。

## 背景介绍

HLL 以固定大小的寄存器数组估算去重基数，适合数据规模大且允许统计误差的场景。Add 将 key 的哈希结果映射到寄存器，并记录尾部比特的秩；Merge 逐项取最大值；Estimate 使用寄存器调和统计量及小基数修正。状态空间取决于精度，不保存完整输入集合。

本任务是新容器开发，没有对应 TBE 历史实现。对标的是操作语义及任务书给出的精度、性能矩阵，不要求与 CUDA 的内部序列化格式兼容。

# 需求分析（required）

## 需求描述

| 参数/接口 | 类型和形状 | 约束 |
| --- | --- | --- |
| keys | Device ND 一维 [N]，I32/I64 | N 个连续有效元素；N=0 可为空指针 |
| sketchSizeKb | uint32 标量 | 8/16/32/64/128/256 |
| precision | uint32 标量 | 13..18，m=2^p 个 byte 寄存器 |
| standardDeviation | double 标量 | 有限正数；选择满足目标的最小合法精度，目标过严拒绝 |
| stream | aclrtStream | 当前设备上下文中的有效流 |
| other | 同模板类型容器 | 容量/精度一致 |
| estimate | uint64 标量 | 近似去重基数，空集为0 |

提供 Create、Destroy、Clear、Add、Merge、Estimate；提供容量、标准差及精度三种构造入口。公共接口大驼峰。Add 保持 `keys, Extent<size_t>(N), stream` 参数顺序，以指针和元素数替代 cuCollections 的输入迭代器对。

## 需求拆解

1. 实现固定无状态 SplitMix64 整数哈希、byte register 更新、合并和 HLL 估算。
2. 管理 ACL 分配/释放和同步流；明确移动所有权、非法参数及重复销毁行为。
3. 对小输入与大输入选择不同更新路径，避免跨核对热门寄存器的直接竞争。
4. 原样执行任务提供的 1280 个功能组合及 72 行性能用例；补充逐寄存器 CPU golden。
5. 提交设计、容器源码、README、API 使用说明、原始日志、源码哈希和逐行性能报告。

# 详细设计（required）

## 算子分析

### 数学公式

令 `h=SplitMix64(uint64_t(key))`，`j=h & (m-1)`，`w=h >> p`，则：

- `rank = clz64(w)-p+1`；当 w=0 时 rank=65-p。
- `M[j] = max(M[j], rank)`；p≥13，rank≤52，一个 byte 可表示。
- 合并：`Mdst[j] = max(Mdst[j], Msrc[j])`。
- `Z = Σ 2^(-M[j])`，`Eraw = αm*m*m/Z`，`αm=0.7213/(1+1.079/m)`。
- 令 V 为零寄存器数；当 `Eraw<=2.5m && V>0`，使用 `E=m*ln(m/V)`，否则使用 Eraw；最终四舍五入为 uint64，溢出饱和。
- 空状态 V=m，直接返回0。

该方案使用经典 HLL 及线性计数，没有引入经验偏差表。是否符合精度要求由任务矩阵和随机状态对照实测判定。

### 支持数据类型与形状

Key 限定 int32_t/int64_t；编译期 static_assert 拒绝其他类型。输入为连续一维数组，无广播。负数和整数边界按 C++ 无符号转换规则映射到模 2^64 的整数。哈希使用固定增量 0x9e3779b97f4a7c15 和 Stafford Mix13，无可变状态；参考 [SplitMix64 公开实现](https://prng.di.unimi.it/splitmix64.c)。选择该通用哈希策略兼顾整数混合成本与任务精度验证，不对具体输入值设置例外分支。设备实现以两个uint32表示64位值，通过显式进位与 __umulhi 组合乘积，计算完全相同的模2^64结果；补充测试逐值对照独立Host端64位算术，覆盖两种类型各65548个输入及进位边界。

## 算子实现

### 实现方案

#### host侧设计

入口为 `include/hyperloglog.h`，设备引用为 `include/hyperloglog_ref.h`，kernel 位于 `include/detail/hyperloglog/kernels.h`。

容器独占 byte sketch、估算 partial buffer 和可选大批量 workspace。禁止复制，允许 noexcept 移动构造/赋值，移动后原对象可析构但不能继续运算。Create 分配状态、清零并同步；异常时回收已分配资源。Destroy 幂等，析构不抛异常。容器必须在 ACL 上下文失效前析构。

构造容量映射为 p=log2(kb*1024)。标准差入口从 p=13 逐级寻找 `1.04/sqrt(2^p)<=sd`，超过18时拒绝。参数异常使用 invalid_argument/length_error；销毁或移出对象的运算使用 logic_error；ACL 错误携带操作名和返回码。

所有公共运算同步传入流。外部保证同一容器调用串行、输入位于对应设备；裸 void* 接口不携带实际分配长度，因此仅能检查空指针和整数溢出，不能证明任意非空指针所指缓冲区足够长。调用者承担有效 Device 指针/长度契约。

从平台取得 AIV 核数 C，小输入使用256线程，block 数为 min(C,ceil(N/256))。N≥2^20 时走私有 sketch 聚合；UB 预聚合与 Fold 使用1024线程，过滤使用2048线程，GM直接聚合使用256线程。kernel 在同一流顺序执行，阶段间不使用 kernel 内跨核自旋屏障。

#### kernel侧设计

**Add 小输入：**线程按全局 stride 遍历 key，计算寄存器索引和秩。每4个 byte 为一个对齐 uint32，以原子 CAS 读改写目标字节，其他三个字节保留 CAS 读取值。CAS 失败重读并重算，避免邻接寄存器覆盖。更新为 max，具幂等、交换和结合性质。

**Add 大输入：**以每核 uint32 工作区避免跨核竞争。256KB 或 N>4,000,000,000 时直接在每核 GM 聚合并 Fold，使用完整64位计数，避免过滤路径32位循环溢出。

8–128KB 的其他大批量先处理前 min(N,W) 个元素；I32/8KB 的 W=1,000,000，其余 W=2,000,000。8/16/32KB 在 UB atomic_max；64KB 分两块处理，复用128KB UB；128KB 在每核 GM 更新。Fold 对各核状态及已有 byte sketch 取最大值。若还有剩余输入，各核把合并结果转为只读 UB 字节阈值表，继续读取和哈希全部余下输入，跳过可证明不会改变状态的项，其余保守执行私有切片 atomic_max，再 Fold。

过滤不改变 HLL 状态：快照本身来自已处理输入及历史状态，其寄存器值 r 是最终值的下界。缓存 T(r)：r=0时为255，r≥8时为0，其余为 2^(8-r)-1；仅当哈希最高8位大于 T(r) 才跳过。这保证被跳过输入的 rank≤min(r,8)≤r。对 r>8 的保守近似可能多做更新，但不会丢掉任何可能提高最终值的输入。工作区保留预聚合结果，不重复清零。只读阈值表与写入工作区分离，初始化后使用核内屏障；以原生byte读取避免循环内的移位解包开销。

Fold 每个线程归并一个寄存器，各线程对工作区的访问连续；相邻4个线程用 asc_shfl_down 交换结果，由首线程打包为一个uint32写回。四线程组不跨warp边界，全部成员都参与shuffle；各组的完整word写入互不重叠，避免相邻byte覆盖。不创建 O(N) Device 输入副本。

**Clear / Merge：**每个线程处理完整 uint32 中的4个 byte；Clear 写零；Merge 解包逐 byte 取最大值再打包。各线程输出地址不重叠；当前使用单核以减少小状态启动成本。自合并跳过 kernel，但保留同步语义。

**Estimate：**单核256线程，每线程固定 stride 计算 float partial sum 和零计数。`2^-rank` 由 IEEE754 指数位构造（rank≤52）；各线程输出独立 partial。同步后复制2048字节至 Host，按固定顺序以 double 累加并计算修正公式。精度可能受单线程 float 累加舍入影响，因此保留实测误差；固定划分保证相同状态多次 Estimate 可重复。

### 内存预算

| 存储 | 大小 | 生命周期 |
| --- | --- | --- |
| Device 最终 sketch | m byte，8–256KB | Create 到 Destroy |
| Device/Host partials | 各256×8=2048 byte | Create/对象生命周期 |
| Device 私有 sketch | C×m×4 byte | 首次大批量 Add 分配，后续复用 |
| UB 私有 sketch / 只读快照 | 最大128KB/核，阶段间复用 | 聚合或过滤 kernel 生命周期 |
| 输入 | 调用方持有 N×sizeof(Key) | 容器不复制或持有其所有权 |

workspace 大小与 m 和核数相关，与 N 无关。Clear 保留 workspace 以便复用；Destroy 释放所有 Device 分配。

## 支持硬件

| 芯片/软件 | 适配情况 |
| --- | --- |
| Atlas 950PR | 当前开发与验证环境 |
| CANN 9.1.0 / Bisheng / dav-3510 | 当前编译配置 |
| CANN 9.0.0-beta.2及后续SIMT硬件 | 任务目标范围，须按具体安装版本编译及复测 |

## 算子约束限制

不支持精确计数、删除元素、动态调整精度、不同精度合并、跨 dtype 合并、自定义 hash/seed、并发修改同一对象或跨设备对象使用。Data 是只读 Device 指针；不得外部写入非法 rank。释放前须保证 ACL 上下文仍有效。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 验证方法 | 来源 |
| --- | --- | --- |
| 相对误差≤3×1.04/sqrt(m) | supplied Catch2 的精确基数与 Estimate 对比 | 任务书3.2 |
| 确定性、清零、分批、合并 | 原始矩阵与全寄存器独立 CPU rank golden | 任务书2.1/3.2 |
| 1280功能组合 | Create38/Destroy24/Clear180/Add726/Merge192/Estimate120 | 任务书3.5 |
| 72性能组合 | 原始程序全部运行，us转换ms，逐行比较 baseline/0.4 | 任务书3.3/3.5 |
| 无O(N)额外Device输入复制 | 分配路径检查与上述空间预算 | 任务书3.4 |

补充测试覆盖 I32/I64、6种容量、负数、极值、随机键、重复/逆序、逐字节状态一致性、自合并、合并联合集、移动赋值、Destroy 后异常、非法标准差和8388625个随机元素的大批量过滤路径。另构造哈希高32位为零、完整哈希为零及 rank 截断边界的 I64 原像，覆盖罕见分支；补充导致首轮失败的小基数回归用例。CPU golden 独立计算 SplitMix64 和标量位扫描 rank，逐字节核对全部寄存器；分批测试跨越小输入、大输入和预聚合阈值。

运行脚本按 dtype 与 SECTION 分区执行原始用例，保存命令、退出码、时间和源码 SHA-256；功能全部退出后才依次运行性能程序。性能框架以同步 API 墙钟时间报告 CPU/Device Time，不宣称纯kernel时延。初测发现部分大批量 Add 超过阈值，历史数据保留；最新冻结实现的正式72项均已达标，逐行结果见自测报告。不能以仅完成测试代替性能达标结论。

性能分析：原始直接聚合的100M输入 Add 初测约3.8–5.3ms，主要优化对象是哈希指令、随机原子访问和 C×m 归并。已比较不同线程数、packed UB、只读快照过滤、分块预聚合和归并并行度，保留状态正确且综合表现较好的路径。纯哈希对照用于区分计算与更新成本；调优时的并发环境数据仅作趋势分析，正式72项性能须在其他测试进程退出后单独运行。尚未达标的行必须保留并解释，不将编译成功、估算结果一致或试验耗时当成整体性能通过。

首轮 XXHash64 在64KB/I32、128个输入但64个不同整数的用例上出现63的估算，超过该小基数的误差阈值；CPU可复现寄存器碰撞。该轮日志保留为问题证据。版本06e5f3f已通过1280个原始组合及306个补充断言，正式性能63/72达标。其后在保持相同哈希值和完整状态语义的前提下，优化整数乘法、只读字节阈值和并行归并；候选Add连续三轮12/12达到性能门槛。冻结实现da6742102a4ac0b12977dd41cdb911ce598c8944现已通过1280/1280个原始功能组合、312/312个补充断言及72/72项正式性能门槛；多翻译单元编译链接通过。完整功能结束后六个性能程序串行运行，I32/64KB Add实测0.788540ms，低于0.8044625ms上限。62个实际源文件及依赖与冻结提交逐字节一致。源码SHA-256、编译命令、覆盖范围、退出码、逐项性能及历史未达标日志见[自测报告](https://gitcode.com/gcw_rBObG5J6/ops-collections/blob/codex%2Fhyperloglog-950/docs/hyperloglog_validation/README.md)。本结论仅为真机自测通过，仍需设计评审，不代表任务已验收。

## 兼容性分析

新增头文件与新增测试目录，不改变现有容器接口。CMake 仅增加 hyperloglog 源文件搜索路径。使用独立 HLL 哈希头文件及仓库现有 Extent、SIMT 宏，不增加运行期第三方依赖；Catch2 仅为测试依赖。

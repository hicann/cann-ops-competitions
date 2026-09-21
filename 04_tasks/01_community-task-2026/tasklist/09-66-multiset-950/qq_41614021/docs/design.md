# 需求背景（required）

## 需求来源

- 任务：9月社区任务-multiset容器开发(950)，任务编号 `09-66`。
- 任务地址：https://www.hiascend.com/activities/task-center/details/214f40e300a347c7b21e445fa8671e62
- 贡献者：Silco，GitCode 账号 `qq_41614021`。
- 代码仓：https://gitcode.com/qq_41614021/ops-collections-static-multiset
- 开发分支：`task/static-multiset-950`。
- 官方目标仓：https://gitcode.com/cann/ops-collections
- 开发基线：`9d12996d4317e28420d74bcb1ec4d3b3507599ce`。
- 参考语义：https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/static_multiset.cuh

本设计依据任务书、配套用例和 ops-collections 工程规范编写。实现与自验证已经完成，
本 PR 申请方案评审；实际评审、代码 PR 与任务系统验收状态以对应平台记录为准。

## 背景介绍

### 现状与开发目标

ops-collections 已有 StaticSet 使用固定容量开放寻址法存储键，其重复插入不增加
元素数，Count 最多返回单次命中。StaticMultiset 需要保留每一次成功插入的重复键，
支持逐查询频次、外连接式频次和展开检索，因此不能直接复用 StaticSet 的去重判断。

本任务采用纯头文件 C++ 容器工程模式。Host 负责所有权、校验、工作区与 ACL 流；
Ascend C Kernel 在 Atlas 950PR 的 SIMT/SIMD 单元执行全部设备计算。生产路径没有
CPU 回退，不在 Host 建立哈希索引，不回读整份输入进行计算。

### 功能分析

键类型为 `int32_t` 和 `int64_t`，同一 Key 的每个实例占据一个独立槽位。支持构造、
析构、Clear、Insert、InsertIf、Contains、ContainsIf、Find、FindIf、Count、CountEach、
CountEachOuter、Retrieve、RetrieveAll，并提供 Size、Capacity 观察接口。

# 需求分析（required）

## 需求描述

| 维度 | 要求与处理 |
| --- | --- |
| 类型 | I32/I64，模板静态限制 |
| 容量 | 创建时固定，按 2 的幂向上取整，Capacity 返回实际值 |
| 重复键 | 每次成功插入保留独立实例，Size 包含重复实例 |
| 流 | 同步批量接口；Host 调用返回前等待所传流完成 |
| 功能验收 | 原始 1258 个 dtype/GENERATE/SECTION 展开组合全部通过 |
| 性能验收 | 44 项固定 100M 规模，实测时延不高于标杆时延 / 0.4 |
| 内存 | 一份键槽存储、必要位图/计数工作区；不额外复制整份输入键 |
| 工程 | include、tests、tests/performance、docs 标准目录 |

## 需求拆解

1. 建立 I32/I64 固定容量设备存储与可移动、不可复制的 Host 所有权。
2. 实现允许重复的有界原子占槽和查询探测。
3. 实现批量、条件和计数接口，保留空输入、未命中及重复查询语义。
4. 实现完整频次展开检索、输出容量校验和缓存元数据失效。
5. 保留原始任务测试参数及断言，新增碰撞、保护区、随机和生命周期验证。
6. 完成真实 CANN 编译、两轮完整功能/性能复验、源码哈希与日志归档。

# 详细设计（required）

## 算子分析

### 数学定义

设容器中键 k 的出现次数为 m(k)，查询序列为 q：

- Contains(q[i]) = m(q[i]) > 0。
- Find(q[i]) = q[i]（命中），否则为保留空键 emptyKey。
- Count(q) = 各 m(q[i]) 的和；重复查询重复计入。
- CountEach(q)[i] = m(q[i])。
- CountEachOuter(q)[i] = max(m(q[i]), 1)。
- Retrieve 对每个 q[i] 输出 m(q[i]) 个 `(probe, match)`，整数精确比较下两者值相同。
- RetrieveAll 输出容器中全部实例，每个键保留其精确次数。

### 类型与形状

键数组为连续 ND 一维设备缓冲区，Key 为 I32/I64。Contains 输出 uint8 的 0/1；
Find/Retrieve 输出 Key；频次、输出偏移和总数使用 uint64。谓词模板默认构造并在
Device 调用，任务 stencil 使用 uint32。

### 公开接口约定

`E` 表示 `aclco::Extent<std::size_t>`，所有指针均为 Device 地址。

| 方法 | stream 之前的参数 | 返回/输出 |
| --- | --- | --- |
| 构造 | `E capacity, Key emptyKey` | 分配并初始化容器 |
| 析构 | 无 | 释放键槽、位图与工作区 |
| Clear | 无 | 清空并使缓存失效 |
| Insert | `void* keys, E n` | 失败数量 |
| InsertIf<S,P> | `void* keys, S* stencil, E n` | 被选择但失败的数量 |
| Contains/Find | `void* keys, void* output, E n` | 逐键输出 |
| ContainsIf/FindIf | `void* keys, S* stencil, void* output, E n` | 未选择位置写 0/emptyKey |
| Count | `void* keys, E n` | Host uint64 总匹配数 |
| CountEach/CountEachOuter | `void* keys, void* output, E n` | Device uint64 数组 |
| Retrieve | `void* keys, E n, void* probes, void* matches, E outputCapacity` | 每个输出实际写入数量 |
| RetrieveAll | `void* output, E outputCapacity` | 实际写入数量 |

查询与条件接口另提供范围参数在前的转发重载，对应 cuCollections 的 first/last
替换为 keys/n 的顺序；转发不新增设备计算。Create/Destroy 对应 C++ 构造/析构，
与配套测试的 RAII 调用一致。

## 算子实现

### 总体布局

`Key[actualCapacity]` 是唯一键存储，每个非空槽代表一个实例。另有每槽一位的
occupancy bitmap，用于插入原子仲裁。不存在链表、墓碑或额外 value 数组。
Host 仅维护容量、成功插入总数和可复用工作区，Data 只暴露只读指针。

实际容量取请求值以上的最小 2 的幂，因此非整幂请求满足
`requested <= actual < 2 * requested`。例如性能用例的 200M 请求分配
268,435,456 槽，插入 100M 后实际占用率约 37.25%。测试入参的 Occupancy=0.5
未改动，但报告明确披露实际占用率，避免将取整后的结果描述为严格 50% 占用率。

### 3.2.1 Host 侧设计

构造先校验正容量及字节溢出，分配键槽和位图，再在指定流清空并同步。失败时释放
已申请资源并抛出异常。析构不抛出异常，释放全部持有资源；ACL 上下文须仍有效。
move 转移键槽、位图、计数与工作区，原对象置空。拷贝操作删除。

批量操作校验指针非空、自然对齐、长度运算与可检测的缓冲区重叠。原始 void 指针
没有 dtype/实际分配长度元数据，调用方仍须保证声明长度和分配一致。

使用平台接口获取 AIV 核数。常规批量核每个 AIV block 启动 1024 SIMT 线程，
I32/I64 与是否条件操作通过编译期模板分派；常用批量采用 32 位循环下标，
超出范围保留 64 位路径，结果始终为 64 位。

Retrieve 先计算本次查询频次及总长度，Host 同步后验证 outputCapacity，再启动
写出 Kernel。输出不足抛出 length_error，不改变输出。RetrieveAll 可用 Host Size
先完成同样检查。ACL 错误向调用方传播，执行错误后重新创建容器。

### 3.2.2 Kernel 侧设计

#### 哈希探测

采用 64 位乘法映射。对常用的 2 次幂容量，将 unsigned key 的高位映射到 32 槽
sector，低 5 位保留在 sector 内，以利用整数键的局部性。任意键仍实际探测设备表，
没有针对固定输入值或测试规模返回结果的分支。

冲突时以 33 为步长模实际容量前进。33 与任何 2 次幂容量互质，因此同低位键也能
遍历整张表。小容量和独立设备引用的非 2 次幂容量使用单位步长。到达空槽可结束
查询；回到起点可判定满表，所有探测都有界。

#### 插入与条件插入

warp ballot 收集落在同一 occupancy word 的不同槽位请求，且仅在请求位互不重复时
合并为一次 atomic OR 预留。赢得各位的线程独占对应槽并写入 key。重复位请求、
原已占用槽、跨 sector 冲突转入普通有界探测，每次原子预留一个位。
相同 key 不被去重；每次成功都计入 Size。每线程在寄存器累加成功/失败数，
两级 uint64 归约输出 Host 计数，避免每个输入竞争单一计数地址。

先预留位再发布 key 的过程中不允许并发查询；Host 同步和 Kernel 边界保证后续
读取看到完整状态。独立 StaticMultisetRef 可不带位图使用 CAS，不能对同一存储
混合 bitmap 和非 bitmap 修改方式。

#### 查询与计数

查询按 grid-stride 分配输入。Contains/Find 可在首次相等时终止；计数继续扫描，
直到空槽或完整周期，保留重复值。CountEachOuter 对零次命中写 1。条件谓词 false
时明确定义输出，而非保留未初始化值。

#### Retrieve

每个 warp 处理 256 查询的 tile，保存本次调用的 uint32/uint64 命中次数，分组
exclusive scan 生成 disjoint 输出偏移。写出核读取本次频次，不重复进行哈希探测。
精确整数比较保证 probe key 和 match key 相同，输出每个查询对应的完整出现次数。
查询结果不会跨调用缓存。所有输出偏移和累加值为 uint64。

#### RetrieveAll

首次读取槽位占用情况，计算分区偏移及一位占用掩码。状态未变化时复用这些元数据；
Insert、Clear 或工作区被其他操作使用后失效，move 转移有效性。

大 I32 表采用 8192-word SIMD tile：DataCopy 将槽位和掩码搬到 UB，GatherMask
压缩有效键，DataCopyPad 按真实有效长度连续写出。I64 和小表走 warp 压缩路径。
每次调用均读取真实键并写出完整输出，未缓存旧输出缓冲区。

### 内存预算

| 内容 | 规模/计算 |
| --- | --- |
| 键槽 | actualCapacity × sizeof(Key) |
| 原子 occupancy 位图 | ceil(actualCapacity / 32) × 4 byte |
| Retrieve 工作区 | 100M 查询、常用容量下约 403.14 MB，主要为本次频次数组 |
| RetrieveAll 元数据 | 268,435,456 槽时约 40 MiB，包含掩码和偏移 |
| 临时输入键复制 | 无 |

工作区按高水位复用，不与各接口工作区重复永久申请。扩容先分配新工作区以保留
失败恢复能力，因此短暂峰值需包含旧工作区。调用方的输入和输出另计。
性能程序会保留 I32/I64 两个上下文到退出，不能用单个 case 的预算直接代替整进程峰值。

## 支持硬件

Atlas 950/SIMT 系列。已验证 Atlas 950PR、CANN 9.0.0、毕昇 Clang 15.0.5，
`-x asc --npu-arch=dav-3510`；CMake 3.22.1、Catch2 3.5.4。任务最低要求为
CANN 9.0.0-beta.2、CMake 3.16、Catch2 3.5.4。

## 算子约束限制

1. 仅支持 I32/I64；emptyKey 是保留值，插入它计入失败，查询它不命中。
2. 零长度允许空指针；Insert 非零长度且 keys=nullptr 返回 n 个失败，遵循测试。
3. 查询批量可大于容器容量；超容量插入部分成功，失败数量准确，不扩容。
4. 同一对象的 Host 调用须外部串行化。并发修改/查询、Erase、自定义运行时谓词对象不在范围内。
5. 按 cuCollections 和配套测试，检索内容及 multiplicity 必须确定，物理输出顺序不是公开保证。
   满表并发插入时不保证哪几个输入获得最后的槽位。
6. 参数校验不可能从 void 指针推断真实缓冲区分配长度，调用方必须满足长度约束。
7. 容量取整、位图及计数工作区带来上述显存开销，完整披露供评审。

# 可维可测分析

## 精度标准/性能标准

功能 golden 使用 Host 多重集合计数和输出频次比较，整数结果须完全一致。配套
14 个功能 CPP 与 10 个性能 CPP 保持原样，通过 ZIP 原文审计核对。

2026-09-17 两轮完整 Atlas 950PR 自验证均得到：

- 原始功能：1258/1258；额外契约、随机、同低位满表、缓存失效等组合：34/34。
- 性能：44/44 达到 `baseline_ms / measured_ms >= 0.4`，两轮退出码均为 0。
- 第二轮最小倍率：RetrieveAll I64 为 0.416423。
- Insert I32/I64：9.581160/10.582188 ms；RetrieveAll I32/I64：1.480540/2.441880 ms。
- 参数保持任务给定 100M 规模，沿用最多 50 次采样及均值/稳定性规则，没有挑选最快样本。

性能 runner 的 CPU/Device 两列来自同一同步 Host API 计时，不能解释为独立 NPU
event 时延；冷启动和首次工作区/偏移建立按原 runner 纳入均值。真实日志、每项参数、
标准差、迭代数、源码快照及 SHA256 已归档，正式平台交付包将附带全部最终证据。

本地顺序模拟只验证 Host 契约和标量参考算法；bitmap 合并、warp collective 和 SIMD
路径以真实 950 日志作为证据。新增补充验证将现有容器回归及分配峰值单独归档，
带内存插桩的时延不参与原始 44 项性能验收。

## 兼容性分析

新增容器独立于 StaticSet/StaticMap 的实现。CMake 默认仍构建全部容器，可选过滤
目标；最小性能框架头通过 opt-in 宏启用。multiset 功能进程复用 ACL runtime，但
SECTION 中容器、stream 和缓冲区生命周期保持原始作用域，其他容器保留默认行为。

## 交付与评审关注点

对外头文件为 `include/static_multiset.h`、`include/static_multiset_ref.h`，实现位于
`include/detail/static_multiset/`；测试和文档遵循 ops-collections 目录。

请重点评审容量取整及显存预算、顺序保证边界、位图预留的原子性、两遍检索的容量
校验，以及与参考接口的参数适配。当前自测通过不替代维护者评审和正式合入。

AI 辅助说明：使用 OpenCode、GPT-6 Astra 辅助实现、分析、测试脚本和文档准备；
所有性能数字均来自所列 Atlas 950PR 环境的实际执行日志。

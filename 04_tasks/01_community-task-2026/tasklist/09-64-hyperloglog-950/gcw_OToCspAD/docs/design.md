# HyperLogLog（950）容器设计文档

本文为社区任务 09-64 的设计评审材料，描述直接 GM CAS 基线。功能和数值精度已有实机验证；Add 性能未达标、全仓回归未完成，**不申请最终验收**。本文的实现及证据在委派时冻结，后续优化实验不自动成为本文结论。

| 版本绑定 | 值 |
| --- | --- |
| 任务 | 9月社区任务-hyperloglog容器开发(950)，任务清单序号 64 |
| ops-collections 上游基线 | `432c15a38192a5e93c7c15400d62cf43cd9521d7` |
| cuCollections 参考提交 | `532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4` |
| 被测实现候选清单 SHA-256 | `26a7d6ec1072bfa02ace31207e87558a38a09c96a9c91ae33f48d97834051511` |
| 被测归档 SHA-256 | `4c17191b49c3f9b8311b0454cfa846874c741d171ac3dc556d5f8c99cc0bade5` |
| 运行标识 | `address-space-fix` |
| 回归证据截止 | 远端原始时间 `2026-09-17T01:36:42.831390+00:00`，32/60 PASS，整体 RUNNING |

远端与本地时钟存在偏差；上述时间仅标识证据截面，不与本地时间相减计算耗时。配套 [evidence-summary.json](evidence-summary.json) 固定来源文件哈希；[performance.csv](performance.csv) 和 [accuracy.csv](accuracy.csv) 提供当前候选的 72 行性能数据和 180 点数值精度数据。本 PR 不含实现源码、连接材料或完整验收日志包。

# 需求背景（required）

## 1.1 需求来源

[任务页面](https://www.hiascend.com/activities/task-center/details/f975dbdd02cb43eaacd0d7e3d3fa8821)及[官方任务 ZIP](https://www.hiascend.com/p/resource/202609/66448a152b6f4bc1aaff5aad233cd467.zip)为需求依据。ZIP SHA-256 为 `994cda9427e7e9a3d52e4c0c432d2a585f37fbde796b72ae12d561ee146819fc`。目标实现仓为 `cann/ops-collections`；本设计按[社区任务目录规则](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)、[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)及[社区流程与检查清单](https://gitcode.com/org/cann/discussions/39)组织。

## 1.2 背景介绍与标杆现状

HyperLogLog 用固定数量的寄存器估算输入集合的去重基数，适用于数据统计、流式集合计数和分片结果合并。接口保留近似计数语义，不能通过识别测试输入或在 CPU 上精确去重来代替 HLL。

参考 [cuCollections HyperLogLog](https://github.com/NVIDIA/cuCollections/blob/532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4/include/cuco/hyperloglog.cuh) 的构造、清空、添加、合并和估算语义。参考库使用 CUDA 设备迭代器，默认 XXH64 哈希，通过寄存器最大值维护 Sketch，采用 HLL++ 偏差修正。其寄存器为 `int32_t`；本任务官方 `hll_test_common.h` 的 `RegisterCount` 明确要求 **1 byte/rank**，因此不能直接沿用参考库的四字节物理布局，也不能把同样 KB 数解释为相同的参考库寄存器数。

参考流程与本实现的语义对应关系：

```text
参考：device iterator [first,last) → hash → register max → HLL++ estimate
本实现：GM keys + count → XXH64 → packed byte max/CAS → 整数部分和 → host HLL++
Merge：两个兼容 Sketch → 对应寄存器逐项 max → 保留原容器配置
```

此处对齐的是输入序列、哈希策略、寄存器集合状态和估算方法。设备原子指令、存储布局、工作区、启动方式与 CUDA 实现不同。

# 需求分析（required）

## 2.1 外部依赖

| 依赖 | 要求与作用 |
| --- | --- |
| 硬件 | Atlas 950 系列及后续支持 SIMT 的产品；当前只实测 Ascend 950PR |
| 软件 | CANN >= 9.0.0-beta.2；当前实测 CANN 9.1.0 |
| 编译 | C++17、仓库支持的 ccec 或 Bisheng ASC；实测 Bisheng `dav-3510` |
| Host | 64 位 Linux；实测 Ubuntu 22.04 x86_64、CMake 3.22.1、g++ 11.4 |
| 测试 | Catch2 >= 3.5.4；官方原始功能/性能测试与 helper 共 13 个文件保持原字节 |
| 参考代码许可 | HLL++ finalizer 和校准表保留 cuCollections Apache-2.0 许可与来源声明 |

## 2.2 内部适配与工程位置

采用 ops-collections 纯头文件容器模式，复用 `Extent`、`xxhash_64`、SIMT 宏、设备原子 API、ACL Runtime 和平台 AIV 核数查询。新增接口及私有实现，不改变其他容器算法或公共哈希实现。

```text
include/hyperloglog.h                 Host owning container
include/hyperloglog_ref.h             Device non-owning reference
include/detail/hyperloglog/           config/bit operations/kernels/finalizer/tables
tests/hyperloglog/                    Official correctness + separate supplements
tests/performance/hyperloglog/        Official six performance programs
docs/                                API, usage and design documentation
```

本容器不走 ACLNN 两阶段 workspace 查询接口；没有 `.so` 算子注册、Host tiling 类或 `op_host/op_kernel` 独立工程。其固定工作区由 C++ 容器拥有，Kernel 在头文件中实例化。设计文档提交到 competitions 仓，未来实现提交应使用上述 ops-collections 目录。

## 2.3 需求拆解及接口

支持 `std::int32_t`、`std::int64_t`，输入为设备端连续一维 ND 数组 `[N]`。容量为 8/16/32/64/128/256 KiB，精度 `p=13..18`，寄存器数 `m=2^p`。不提供广播、浮点 Key、自定义分配器或任意自定义 Hash 类型。

```cpp
template<class Key> class aclco::HyperLogLog {
public:
    using ExtentType = aclco::Extent<std::size_t>;
    static HyperLogLog CreateWithSketchSizeKB(uint32_t kb, aclrtStream stream = nullptr);
    static HyperLogLog CreateWithStandardDeviation(double deviation, aclrtStream stream = nullptr);
    static HyperLogLog CreateWithPrecision(uint32_t precision, aclrtStream stream = nullptr);
    static HyperLogLog Create(uint32_t kb = 32, xxhash_64<Key> const& hash = {},
                              aclrtStream stream = nullptr);
    void Destroy();
    void Clear(aclrtStream stream = nullptr);
    void Add(void* keys, ExtentType count, aclrtStream stream = nullptr);
    void Merge(HyperLogLog const& other, aclrtStream stream = nullptr);
    uint64_t Estimate(aclrtStream stream = nullptr) const;
    uint32_t SketchSizeKB() const noexcept;
    uint32_t Precision() const noexcept;
    uint64_t HashSeed() const noexcept;
};
```

| 接口 | 入参、返回与状态语义 | 校验和边界 |
| --- | --- | --- |
| 三种构造工厂 | 返回拥有资源的对象；默认哈希种子为 0 | KB 必须属于上述六档；p 必须在 [13,18]；标准差必须有限且 >0，选择满足 `1.04/sqrt(2^p)<=deviation` 的最小合法 p，无法满足时报错 |
| Create | 参数顺序为容量、哈希策略、流；读取 `hash.Seed()` | 仅支持 `xxhash_64<Key>`，默认容量 32 KiB |
| Destroy / 析构 | 显式释放 / RAII 释放 | Destroy 对无效对象报错；析构不抛异常，不等于显式 Destroy 的错误报告能力 |
| Clear | 将全部寄存器设为 0；保留配置 | 对象和当前设备必须有效；之后可复用 |
| Add | `keys` 为设备地址，`count` 为元素个数，返回 void，更新 Sketch | N=0 允许 nullptr；N>0 校验非空与 Key 对齐；检查 `N*sizeof(Key)<=PTRDIFF_MAX` |
| Merge | 将 other 合并到当前对象，other 保持不变 | 两对象必须同类型、同 p、同种子、均属于当前设备；自合并无更新 |
| Estimate | 返回 uint64 基数估计；不改 Sketch | 空集为 0；非空为近似值，不要求每个输入精确计数 |

类型限制通过 `static_assert` 检查；容量/流/设备/状态问题分别由参数校验或 ACL 返回值报告。原始 `void*` 的完整分配边界和设备可访问性不能仅靠现有接口判定，调用者必须保证 `[keys, keys+N)` 有效；不能声称任意越界指针都在 Host 侧被完整验证。

任务书通用参数表将 `numInputs` 列在 Add/Estimate/Merge；正式用例及参考容器实际上仅让 Add 接收输入数量。当前以正式测试签名为准，Estimate/Merge 处理已有 Sketch，不引入无意义的 N 参数。三种工厂与显式 Destroy 补足官方用例的构造和生命周期需求。

# 详细设计（required）

## 3.1 调用方式及生命周期

调用方先初始化 ACL 并选择设备，再构造容器。对象是 move-only，禁用拷贝以防双重释放；移动转移三个缓冲区的所有权，源对象变为无效。对象必须在所属设备仍为当前设备且 ACL 尚未结束时析构或 Destroy。move assignment 会先按同样前提释放原资源。

构造先验证 p、记录设备，再申请 Sketch、Device 部分和及 pinned Host 部分和，随后 Clear 并同步。任一步失败会回收已申请资源再抛异常。显式 Destroy 对释放错误报告异常；析构/内部 Release 忽略 ACL 释放返回值并清空指针，不能将该行为描述成无条件的故障恢复保证。

```text
Create/config validation → allocate fixed storage → Clear → stream synchronize
Add/Merge/Clear → validate owner/current device → launch → stream synchronize
Estimate → launch integer partial sums → D2H 16 KiB → synchronize → finalizer
Destroy → validate owner → free device/pinned host buffers → invalid object
```

正常返回的公共计算 API 均为同步 API。Host 调用方必须串行访问同一对象；`const Estimate` 仍复用其工作区，不能据 const 推断线程安全。不同流上的顺序调用依赖前一个调用已经完成；默认 `nullptr` 流及跨流专项设备契约检查截至本截面为 **NOT_RUN**，不以显式流通过推断其专项验证已完成。

设备侧 `HyperLogLogRef<Key>` 是接收外部 words/p/seed 的非 owning view，仅提供单元素 Add。它不管理生命周期或执行跨流同步，调用者负责有效配置、存储存活和外部依赖。本公开 owning API 不暴露 Sketch 原始指针 getter。

## 3.2 数据布局、算法及 Kernel 设计

### 3.2.1 一字节寄存器与打包规则

Sketch 申请 `m` 字节，表示 `m` 个 uint8 逻辑 rank；从对齐 `uint32_t*` 访问，每字包含四个 rank。`j` 的字位置为 `j>>2`，字内偏移为 `8*(j&3)`，并不为每个 rank 分配 uint32。p>=13 保证 m 为 4 的倍数，不需尾部 padding。

| Sketch (KiB) | p | 逻辑寄存器 m | 32-bit 存储字数 |
| --- | --- | --- | --- |
| 8 | 13 | 8192 | 2048 |
| 16 | 14 | 16384 | 4096 |
| 32 | 15 | 32768 | 8192 |
| 64 | 16 | 65536 | 16384 |
| 128 | 17 | 131072 | 32768 |
| 256 | 18 | 262144 | 65536 |

四字节原子访问粒度只是更新协议，不改变一个寄存器占一字节的存储定义。这也引入相邻四个寄存器之间的原子竞争，需要在性能优化时保留字节语义。

### 3.2.2 哈希、rank 和 Add

对每个 Key 的 64 位哈希 `h=XXH64(key,seed)`，取高 p 位为桶索引，剩余位的前导零数加一为 rank：

```text
j = h >> (64-p)
suffix = (h << p) | (1ULL << (p-1))
r = clz64(suffix) + 1
M[j] = max(M[j], r)
```

哨兵保证 suffix 非零，`1<=r<=65-p`，即最大 rank 为 52..47，能放入一字节。当前实现用 32/16/8/4/2/1 位条件移位计算 rank；公式中的 clz 表示数学含义，不声称当前代码已经采用设备 clz 优化。空寄存器为 0。

Add 按 grid-stride 分配输入，各线程直接从 GM 读取 Key，再复制到线程局部 `Key const key` 后传给哈希引用参数，避免把 `__gm__` 元素直接绑定到普通地址空间的 `Key const&`。这是该候选的实机编译修复。

```text
word_addr = words + (j >> 2)
observed = atomic_cas(word_addr, 0, 0)      // atomic snapshot
loop:
    desired = replace_only_lane_with_max(observed, j & 3, r)
    if desired == observed: return
    previous = atomic_cas(word_addr, observed, desired)
    if previous == observed: return
    observed = previous
```

初读也通过原子 CAS 完成；失败后以最新完整字重算，保存其他字节的并发更新。每个 rank 单调不减；输入顺序、重复次数及线程调度不改变最终逐寄存器 max。不能把整个打包字用普通 atomic max 替换：整数大小关系并不等价于四个独立字节分别取 max。

### 3.2.3 Clear 与 Merge

Clear 的每个工作项拥有一个 uint32 字并直接写 0；Merge 的每个工作项读两个对应字，对四个字节分别取 max，再写目标字。不同线程不写同一字。Merge 本身为逐寄存器 max，具幂等、交换、结合性，但 Host 接口是“写左容器”的有状态操作。

Clear/Merge/Estimate 的普通读写不能与同一 Sketch 的外部 Add/ref 操作并发；公共同步 API 及调用者外部串行化共同保证阶段之间的顺序。未使用跨核 barrier。原子更新的能力不意味着容器全部接口可并发。

### 3.2.4 Estimate：确定顺序的整数部分和与 HLL++

设 `V` 为零寄存器数，`Z=sum(2^-M[j])`，原始估计为 `E=alpha(m)*m*m/Z`，本 p 范围使用 `alpha(m)=0.7213/(1+1.079/m)`。

设备固定输出 1024 个槽，每槽保存 uint64 缩放和与 uint64 零计数，slot s 处理 `j=s,s+1024,...<m`：

```text
scale = 65-p
Q[s] = sum(1ULL << (scale-M[j]))
V[s] = count(M[j]==0)
Device output = interleaved {Q[s], V[s]} for s in [0,1024)
```

合法 rank<=scale，移位无负数。每槽至多 `2^(p-10)` 个寄存器，最大整数和 `2^(p-10)*2^(65-p)=2^55`，uint64 不溢出；全部槽每次完整写入，无需预清工作区。

Host 回读固定 16 KiB，按槽序累加 `ldexp(double(Q[s]),p-65)` 和 V，再执行参考 HLL++ finalizer：有零桶时考虑 `m*log(m/V)` 的线性计数，小范围按参考阈值切换；需要时用六个邻近校准点估计偏差，最后 round，超出 size_t 范围饱和到上限。空 Sketch 的线性计数为 0。设备不进行浮点原子求和，固定槽序保证同环境下可重复；未承诺不同 CPU/libm 的结果逐 bit 相同。

### 3.2.5 分核、分块、LocalMemory 与 tiling key

每个 AIV block 启动 256 个 SIMT 线程，`threadIndex=blockIdx*threadNum+threadIdx`，`stride=blockNum*threadNum`。核数为 `min(availableAiv,max(1,ceil(work/256)))`；实测可用 56 个 AIV。Add 的 work=N，Clear/Merge 为 m/4，Estimate 为 1024（最多 4 个 block）。

当前方案不申请显式 UB/LocalMemory tile、双缓冲或输入中转数组，Key 直接读取 GM；线程私有标量与编译器管理寄存器不等于“设备资源零开销”。grid-stride 统一覆盖尾部，不按输入分布选择作弊路径。没有需要下发的 tiling key；Key 类型由模板实例化，p/seed/count 为运行参数。各 Kernel 使用独立入口，Estimate 阶段顺序由流保证。

### 3.2.6 内存开销与复杂度

| 资源/操作 | 容量或代价 |
| --- | --- |
| 持久 Device Sketch | `m` 字节（8..256 KiB） |
| 持久 Device 部分和 | `1024*2*8=16384` 字节 |
| pinned Host 部分和 | 16384 字节 |
| Add 输入 | 调用者拥有 `N*sizeof(Key)`，容器不复制或取得所有权 |
| Add | 输入遍历 O(N)，CAS 冲突重试使实际开销随竞争变化；无与 N 成比例的额外 Device 缓冲区 |
| Clear/Merge/Estimate | O(m)；Estimate 额外固定 16 KiB D2H |

每对象显式请求的 Device 字节合计为 `m+16 KiB`，不包括 Runtime/allocator 对齐、上下文或驱动开销；不能把申请值等同于整卡 HBM 占用。没有输入 D2H、Host 去重集合或长度为 N 的中间哈希数组。

## 3.3 支持硬件

| 硬件/软件组合 | 状态 |
| --- | --- |
| Ascend 950PR + CANN 9.1.0 + Bisheng dav-3510 | 编译、功能及精度已实测；性能见第 5 节 |
| 其他 Atlas 950 或后续 SIMT 产品、其他满足最低版本的 CANN | 目标兼容范围，NOT_RUN，需分别编译和实机验证 |
| A2/A3 或 CUDA GPU | 非本次交付目标 |

## 3.4 约束与参考差异

除前述单字节布局、构造签名、同步实现、参数 N 差异外，本版本不支持自定义 allocator、任意 Hash、不同 p 降采样合并、Sketch 序列化或外部导入。Merge 的同类型由 C++ 类型约束保证，p/seed 在 Host 校验。自定义 seed 不代表哈希抗攻击承诺。

容器不是安全边界；设备原始指针及 stream 必须来自有效 ACL 上下文。异常路径上的设备故障恢复、默认流专项、主机多线程并发均不能由已有单线程显式流测试推出。`HyperLogLogRef` 的调用者必须提供合法 p 和 rank 状态，非法外部存储会破坏估计移位前提。

# 特性交叉分析

| 交叉项 | 处理 |
| --- | --- |
| 现有容器/公共基础设施 | 不改其他容器算法、公共哈希与分配器；全仓构建通过，完整回归仍在执行 |
| 多流/多线程 | 公共接口正常返回前同步；同对象串行调用。不同流专项 NOT_RUN，不支持同对象 Host 并发 |
| 多设备 | 记录 owning device，公共调用校验当前设备；不跨设备合并，跨设备使用/析构由调用者避免 |
| 输入类型/格式 | 仅连续 I32/I64 ND；无广播/稀疏/复数语义 |
| 内存和大输入 | N 只影响遍历，不改变工作区；检查字节范围溢出，调用者负责真实分配边界 |
| 确定性/精度 | max 状态与顺序无关；固定整数部分和及 Host 顺序；统计误差仍按对应 m 的限值验收 |
| 许可/接口兼容 | HLL++ 适配保留 Apache-2.0 来源；新头文件独立命名空间及宏名；无既有 ABI 修改 |

# 可维可测分析

## 5.1 精度、性能与内存标准

按任务书，N>0 时 `abs(estimate-N)/N<=3*1.04/sqrt(m)`，N=0 单列检查 estimate=0。性能每行 `baseline_ms/measured_ms>=0.4`，等价于 `measured_ms<=baseline_ms/0.4`；不达标必须如实解释，不得由总体平均掩盖失败行。内存不得有与输入规模线性重复的额外 Device 拷贝。

## 5.2 已执行验证与未完成项

以下状态只属于页首冻结候选。完整官方用例源码未改；1280 是 dtype、GENERATE、SECTION 展开计数，不是 Catch2 顶层 case 数。

| 检查 | 截止时结果 | 证据边界 |
| --- | --- | --- |
| CANN 编译 | 14 个 HLL 程序 PASS | 绑定被测源码清单与二进制哈希 |
| Smoke | 4 项 PASS | 不代替完整功能验证 |
| 官方功能 | Create38、Destroy24、Clear180、Add726、Merge192、Estimate120，合计1280 PASS | 包含官方 large 场景；本地另从原始 XML 复核展开覆盖 |
| 数值精度 | I32/I64 × p13..18 × 15个基数，共180点 PASS | 包括0至1e8；见 accuracy.csv |
| 补充检查 | 已完成部分 PASS | 生命周期、负数/极值、分批及重复操作等；不覆盖所有后续专项 |
| 全仓回归 | 构建 PASS；截止时32/60 PASS，整体 RUNNING | 未完成28项，不宣称全量通过 |
| 性能 | 72行采集完成；Add12行数值FAIL，其他60行数值PASS | 全部行仍为 UNKNOWN_PROTOCOL，不等于性能验收PASS |
| APP内存审计 | 18个HLL进程、19份plog、3190条APP快照释放平衡 | 已读APP域无ERROR；不等于整卡HBM或驱动内存无泄漏证明 |
| 默认nullptr流、跨流专项、人工rank边界专项 | NOT_RUN | 当时仍等待串行设备资源，不引用未来结果 |
| 优化实验/单Add性能剖析 | 不属于本实现；未完成实验为 NOT_RUN | 不以编译成功当作设备结果 |

180 点中最大相对误差为 2.1499306%，最大“误差/该点限值”比例为 0.6410257；不同 p 的限值不同，不能拿最大相对误差与另一 p 的门槛比较。12 个 N=1e8 点最大相对误差为 0.33888%。

内存源码检查与 APP 记录互相支持：Add 大输入记录的峰值 400556032/801112064 字节匹配 I32/I64 单份输入按 2 MiB 对齐；性能框架同时保留两种类型输入，其 1201668096 字节峰值不能误判为容器重复复制输入。分配域各自历史峰值不能相加冒充同一时刻峰值。

## 5.3 性能现状、协议差异与优化计划

官方计时框架使用整数微秒的 Host chrono，并把同一同步耗时写入 CPU/Device 两列；“Device”列不是独立设备事件计时。最多50次迭代，累计报告DeviceTime达到0.5秒后才检查稳定性/停止条件，没有独立预热阶段。框架仅保留聚合统计，不提供每次样本；本次所有行至少两次迭代，但仍不足以确认与标杆一致。

任务表称 UNIFORM，原始 perf_add.cpp 实际生成顺序唯一整数1..N，每次计时 Add 开始时 Sketch 为空（Clear 在计时间隔外）。本文保留原测试，不用已成熟 Sketch 的重复 Add 代替空 Sketch 的正式性能。空 Sketch 不等同于硬件冷缓存。标杆完整计时协议截至截面仍为 **UNKNOWN_PROTOCOL**。

| dtype | Sketch (KiB) | 标杆(ms) | 允许最大(ms) | 实测Host同步(ms) | 数值结果 |
| --- | --- | --- | --- | --- | --- |
| I32 | 8 | 0.316691 | 0.7917275 | 336.973500 | FAIL |
| I32 | 16 | 0.317554 | 0.7938850 | 176.406667 | FAIL |
| I32 | 32 | 0.317865 | 0.7946625 | 99.365167 | FAIL |
| I32 | 64 | 0.321785 | 0.8044625 | 55.954667 | FAIL |
| I32 | 128 | 0.458247 | 1.1456175 | 40.381308 | FAIL |
| I32 | 256 | 2.031375 | 5.0784375 | 32.481125 | FAIL |
| I64 | 8 | 0.376610 | 0.9415250 | 337.164000 | FAIL |
| I64 | 16 | 0.377037 | 0.9425925 | 176.452000 | FAIL |
| I64 | 32 | 0.377799 | 0.9444975 | 99.485667 | FAIL |
| I64 | 64 | 0.381595 | 0.9539875 | 56.246778 | FAIL |
| I64 | 128 | 0.509386 | 1.2734650 | 40.472769 | FAIL |
| I64 | 256 | 2.138625 | 5.3465625 | 31.943625 | FAIL |

直接 GM CAS 需要每个输入至少一次原子快照，竞争时还需重试；一字打包使相邻寄存器共享原子更新粒度。这是源码层面明确的开销来源，尚无当前正式候选的设备剖析证明其占比，不将推测写成已定位的唯一瓶颈。

后续优化方向为先用独立 Kernel 测量哈希、rank、原子访存与同步开销，再评估核内私有 UB 聚合、按 epoch 过滤已知较小 rank、设备原子 max 等方案。它们必须保持全范围 rank（不能截断到4位）、一字节持久寄存器、重复/分批/合并等价及输入无额外线性副本；暂不属于本被测实现。任何采纳都要冻结新候选并重跑功能、精度、性能及受影响回归，不能继承旧候选 PASS。

## 5.4 测试组织与复现步骤

1. 核验任务 ZIP、候选归档和 MANIFEST；确认13个官方测试/helper字节不变，使用独立构建目录，保留源码/编译命令/二进制哈希。
2. 在目标950/CANN版本执行环境检查，再编译14个HLL程序；先 smoke，再六个官方功能程序、补充功能和180点数值精度程序。保存退出码、原始XML及覆盖展开记录。
3. 在无其他测量负载时运行六个官方性能程序，逐行导出72组输入、统计值及日志/二进制哈希，按各自baseline/0.4计算数值状态；记录实际协议和 UNKNOWN 项。
4. 完成全部原仓回归与专项接口/rank/内存检查，保留原始plog与同进程内计时。观察中尚未退出的进程只能记 RUNNING。
5. 对准备验收的同一最终候选进行需求/实现与安全/证据两项独立审查；社区设计 PR 评审、实现 PR 与最终验收是不同阶段。

补充用例需覆盖 N=0/1、非256倍数尾部、负数/极值、重复与多批输入、Clear重用、同种子合法合并与不兼容拒绝、move与Destroy后拒绝操作；后续私有缓存方案还需人工rank边界、字边界、最后字节及epoch尾部非冗余测试。Host 数学测试只辅助验证，不能替代设备原子/ACL行为或性能证据。

## 5.5 兼容性、风险和评审范围

当前设计已解决一字节物理存储与参考四字节布局差异，并通过现有实机功能/精度检查。仍需完成 Add 优化、统一/确认性能协议、全量回归及专项测试后才能进入最终验收。这里报告的本地审查不等于社区审核通过；PR 发布后由维护者作出设计审核结论。

本设计和配套证据摘要由 OpenAI Codex（GPT-6）基于冻结源码、官方模板及现有日志辅助编写，并由独立 AI 审查代理复核。AI 参与覆盖需求梳理、设计成文、证据检查和发布准备；未声称已完成人工代码审核，不生成或补造测试结果。后续社区审阅意见将按同一版本绑定修订。

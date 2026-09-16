# static_multiset 容器设计（Atlas 950）

## 1. 需求背景（required）

### 1.1 需求来源

本设计对应[9月社区任务-multiset容器开发(950)](https://www.hiascend.com/activities/task-center/details/214f40e300a347c7b21e445fa8671e62?menu=tasks)，以[官方任务书及测试附件](https://www.hiascend.com/p/resource/202609/d675112e170648019f09112863ab705f.zip)为验收依据。

参考接口为 [NVIDIA cuCollections static_multiset](https://github.com/NVIDIA/cuCollections/blob/532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4/include/cuco/static_multiset.cuh)。目标仓为 [cann/ops-collections](https://gitcode.com/cann/ops-collections)，工程接口参考版本为 `9d12996d4317e28420d74bcb1ec4d3b3507599ce`。

### 1.2 背景介绍

静态多值集合保存可重复的键，并提供批量插入、成员查询、查找、计数及检索。容量在构造时确定，重复键增加元素总数，不触发去重或自动扩容。

ops-collections 已有 StaticSet、StaticMap、DynamicMap 等容器。StaticSet 的插入会拒绝重复键，其单键 Count 等价于成员关系，不能直接用于本任务。拟复用仓库的 Extent、哈希、探测、设备存储、SIMT 调用及测试设施，新增独立的 StaticMultiset 实现。

## 2. 需求分析（required）

### 2.1 功能范围

提供 Create、Destroy、Clear、Insert、InsertIf、Contains、ContainsIf、Find、FindIf、Count、CountEach、CountEachOuter、Retrieve 和 RetrieveAll 共 14 项功能，并提供官方测试所需的 Capacity、Size 查询。Create/Destroy 按现有 C++ 容器惯例映射为构造/析构。

Key 支持 `int32_t`、`int64_t`。输入输出为 Device 侧连续一维数组，条件模板元素为 `uint32_t`，计数结果为 `uint64_t`。应支持空输入、重复键、重复查询、不同重复次数、不同命中率以及容量边界。

### 2.2 多值集合语义

记容器中键 k 的出现次数为 m(k)，查询序列为 q，长度为 n，元素总数为 S。

| 功能 | 输出或状态变化 |
| --- | --- |
| Create/Destroy | 构造分配固定存储并初始化空状态；析构释放所持有的设备资源。 |
| Clear | 所有 m(k) 变为 0，容量不变。 |
| Insert | 每个成功插入的键使对应 m(k) 增加 1，返回失败元素数量。 |
| InsertIf | 仅处理谓词成立的元素，被排除的元素不计入失败数。 |
| Contains | 第 i 项为 `m(q[i]) > 0`。 |
| ContainsIf | 谓词不成立时写 false，否则执行 Contains。 |
| Find | 命中时输出 q[i]，未命中时输出 emptyKey。 |
| FindIf | 谓词不成立时写 emptyKey，否则执行 Find。 |
| Count | 返回所有 `m(q[i])` 的总和，重复查询分别贡献计数。 |
| CountEach | 第 i 项输出 `m(q[i])`，未命中为 0。 |
| CountEachOuter | 第 i 项输出 `max(m(q[i]), 1)`。 |
| Retrieve | 对每个查询输出 m(q[i]) 对查询键和匹配键，返回有效输出对数 R。 |
| RetrieveAll | 输出 S 个已存储键，保留每个键的全部重复次数，返回 S。 |

例如容器为 `[2, 2, 3]`、查询为 `[2, 2, 4]`，CountEach 为 `[2, 2, 0]`，CountEachOuter 为 `[2, 2, 1]`，Count 为 4，Retrieve 输出四对 `(2, 2)`。

### 2.3 约束与错误处理

- 容量必须为正；构造容量为 0 时抛出 `std::invalid_argument`。实际 Capacity 可以因存储布局向上调整，但不小于请求值，Size 不得超过实际容量。
- emptyKey 为保留空键，不能作为有效输入插入。模板在编译期限制 Key 类型；内存大小和偏移计算检查整数溢出。
- n=0 时允许空输入、空输出及空 stencil，返回 0 或不执行设备计算。正常查询的输出按输入位置对应。
- Insert 的错误结果采用失败数报告。`Insert(nullptr, n)` 返回 n，容器保持不变；有效输入超出剩余容量时允许部分成功并返回失败数。该约定与附件 `insert_test.cpp` 和现有 StaticSet 测试一致。任务书未规定报错必须抛异常或全部回滚。
- InsertIf 在合法 stencil 下只统计被选择且未能插入的元素；非零长度但 keys/stencil 为空时，沿用现有插入接口以 n 报告无效调用，容器不变。
- 非零长度查询缺少必需指针时报告参数错误。Retrieve/RetrieveAll 在计算所需输出数量后检查显式输出容量；容量不足时报错，禁止越界写入。所需输出为 0 时允许空输出指针。
- 固定长度数组的有效长度由 n 定义，调用方须提供对应类型、长度和对齐的设备缓冲区；裸指针本身不携带数组实际长度。检索接口增加显式输出容量，以检查可变长输出。
- 同步接口返回前完成调用流上的相关工作并检查 ACL 返回状态。跨流读写同一容器须由调用方建立事件依赖；析构前须结束外部持有设备引用的工作。

### 2.4 确定性要求及评审确认项

任务书要求相同输入重复执行时容器状态和输出一致。设计保证整数计算不引入数值误差；逐项查询结果保持输入位置对应；容量足够时容器中每个键的出现次数唯一确定。

Retrieve 拟按查询顺序分配输出区间，使每个查询对应的全部重复匹配连续写出。RetrieveAll 拟采用固定槽位顺序的稳定压缩，保证未改变的表重复检索时输出顺序一致。

需要评审确认的范围如下：

1. Retrieve/RetrieveAll 是否按键及重复次数比较，还是要求清空后以相同输入重建时，原始输出数组顺序也完全一致？附件 RetrieveAll 排序后比较，参考库不承诺检索顺序；这些依据不能自动豁免任务书可能要求的更强确定性。
2. 超容量部分插入时，是否要求每次保留同一批键？附件只检查 Size 和失败数，没有指定被保留的键集合。

若要求跨重建输出逐元素一致，RetrieveAll 需采用与并行插入布局无关的规范顺序，例如键的升序；若要求超容量后的键集合一致，可按输入顺序选择最早的有效元素填充剩余容量。两种方案均需计入相应排序或选择成本和工作空间，不能用较弱测试结果替代该要求的验证。本设计请求明确这一比较范围，不将尚未确认的保证列为已经满足。

## 3. 详细设计（required）

### 3.1 容器原型与公开接口

以下为本任务公开接口的拟定原型。`Extent` 默认使用 `aclco::Extent<std::size_t>`；`SizeType` 使用 64 位无符号计数。函数名称沿用仓库大驼峰命名。构造、清空及下列批量函数均采用同步语义。

```cpp
template <class Key, class Extent = aclco::Extent<std::size_t>>
class StaticMultiset {
public:
    using SizeType = std::uint64_t;

    StaticMultiset(Extent capacity, Key emptyKey, aclrtStream stream);
    ~StaticMultiset();
    StaticMultiset(StaticMultiset const&) = delete;
    StaticMultiset& operator=(StaticMultiset const&) = delete;
    StaticMultiset(StaticMultiset&&);
    StaticMultiset& operator=(StaticMultiset&&);

    void Clear(aclrtStream stream);
    SizeType Capacity() const noexcept;
    SizeType Size(aclrtStream stream) const;

    SizeType Insert(void* keys, Extent n, aclrtStream stream);
    template <class StencilT, class Predicate>
    SizeType InsertIf(void* keys, Extent n, StencilT* stencil,
                      Predicate pred, aclrtStream stream);

    void Contains(void* keys, Extent n, void* output, aclrtStream stream) const;
    template <class StencilT, class Predicate>
    void ContainsIf(void* keys, Extent n, StencilT* stencil, Predicate pred,
                    void* output, aclrtStream stream) const;

    void Find(void* keys, Extent n, void* output, aclrtStream stream) const;
    template <class StencilT, class Predicate>
    void FindIf(void* keys, Extent n, StencilT* stencil, Predicate pred,
                void* output, aclrtStream stream) const;

    SizeType Count(void* keys, Extent n, aclrtStream stream) const;
    template <class ProbeKeyEqual, class ProbeHash>
    void CountEach(void* keys, Extent n, ProbeKeyEqual const& equal,
                   ProbeHash const& hash, void* output, aclrtStream stream) const;
    template <class ProbeKeyEqual, class ProbeHash>
    void CountEachOuter(void* keys, Extent n, ProbeKeyEqual const& equal,
                        ProbeHash const& hash, void* output, aclrtStream stream) const;

    SizeType Retrieve(void* keys, Extent n, void* probeOutput, void* matchOutput,
                      Extent outputCapacity, aclrtStream stream) const;
    SizeType RetrieveAll(void* output, Extent outputCapacity,
                         aclrtStream stream) const;
};
```

Contains 系列 output 为逐元素的一字节 0/1，Find 系列及检索 output 为 Key，CountEach 系列 output 为 uint64。Size 包含重复元素，Count 的返回值可能大于 Size，因此中间归约也使用 64 位计数并检查溢出。

以“输入指针、元素数量”对应参考库的“起始迭代器、结束迭代器”。CountEach/CountEachOuter 的显式等价谓词、探测哈希和输出顺序保留参考定义；哈希必须与建表策略兼容，传入的谓词和哈希状态必须参与设备计算。

为兼容官方附件和仓库既有调用方式，额外提供以下 Host 转发重载，不重复启动 Kernel 或同步：

```cpp
void Contains(void* keys, void* output, Extent n, aclrtStream stream) const;
void Find(void* keys, void* output, Extent n, aclrtStream stream) const;

template <class StencilT, class Predicate>
SizeType InsertIf(void* keys, StencilT* stencil, Extent n, aclrtStream stream);
template <class StencilT, class Predicate>
void ContainsIf(void* keys, StencilT* stencil, void* output,
                Extent n, aclrtStream stream) const;
template <class StencilT, class Predicate>
void FindIf(void* keys, StencilT* stencil, void* output,
            Extent n, aclrtStream stream) const;

void CountEach(void* keys, void* output, Extent n, aclrtStream stream) const;
void CountEachOuter(void* keys, void* output, Extent n, aclrtStream stream) const;
```

条件兼容重载传入 `Predicate{}`；计数兼容重载使用容器的键等价和哈希策略。Insert/InsertIf 按目标仓惯例返回失败数；参考版本的 `insert` 返回 void，`insert_if` 返回成功数，这里明确适配返回语义。Retrieve 返回输出对数，RetrieveAll 返回元素数，以指针加数量表示参考库返回的结束迭代器；显式输出容量用于安全校验。

### 3.2 工程结构

```text
ops-collections/
├── include/
│   ├── static_multiset.h
│   ├── static_multiset_ref.h
│   └── detail/static_multiset/
├── tests/
│   ├── static_multiset/
│   └── performance/static_multiset/
├── docs/static_multiset_API文档和使用示例.md
└── README.md
```

StaticMultiset 拥有设备存储，负责生命周期、参数检查、核函数启动及流同步。StaticMultisetRef 为不拥有内存的设备引用，提供针对单键的插入、查找和计数操作。实现与 Kernel 放入 detail/static_multiset，测试通过现有 CMake 入口集成。

### 3.3 Host 侧设计

构造时校验容量、键类型和分配大小，分配固定槽位存储及必要计数空间，按调用流初始化为 emptyKey。Clear 重新初始化槽位与 Size；析构只释放本对象拥有的资源，移动构造/赋值转移所有权。

批量调用依据元素数量及设备支持的 AIV 核数选择网格。输入直接由设备 Kernel 访问，不先复制到 Host 计算。Host 仅取回必要的标量数量或错误状态。Count、输出偏移和 Size 的运算保持 uint64，避免沿用旧单值容器的 uint32 总计数。

计数、归约和前缀和工作空间按实际操作申请或复用。Retrieve 先求匹配总数 R，再校验两个输出数组的显式容量；不足时在写输出之前返回错误。R=0 时不要求非空输出数组。

### 3.4 Kernel 侧设计

#### 3.4.1 重复插入与容量边界

采用开放寻址，每个成功插入的元素占一个槽位。SIMT 线程计算输入键的探测起点，遇到相同键继续探测，遇到空槽使用原子 CAS 竞争写入。CAS 失败后继续检查有效探测序列，禁止沿用 StaticSet 的重复键拒绝分支。

探测策略必须覆盖实际容量对应的槽位，且遍历次数有上界，满表时能够终止。合法键写入成功后计入成功数；汇总有效尝试数与成功数得到失败数，同时更新 Size。emptyKey 作为无效插入报告失败，不写入表。InsertIf 仅对谓词成立的有效元素执行插入。

I32/I64 原子操作与内存可见性依据目标版本的 SIMT 接口实现并验证。不同批次通过流顺序隔离；本任务同步批量接口不要求无依赖的跨流读写并发。

#### 3.4.2 查询与计数

Contains/Find 命中后可以提前结束。CountEach 必须累计该键完整有效探测区间中的全部匹配，不能在首次命中时结束。插入采用固定探测序列、按序寻找空位且不提供删除操作，因此插入批次结束后的查询可在到达第一个空槽时停止；满表查询则在遍历整个探测周期后结束。

CountEachOuter 将未命中的计数置为 1。Count 对逐查询计数进行分块归约，先生成块级部分和再做最终归约，减少对单一全局计数器的竞争。全部计数及输出偏移使用 uint64。

#### 3.4.3 检索与输出空间

Retrieve 采用“逐查询计数、排他前缀和、写出”的方式确定互不重叠的输出区间。对精确整数相等的查询，第 i 个区间写出 m(q[i]) 对 `(q[i], q[i])`，重复查询分别占据自己的区间。计数结果与输出区间必须来源于同一容器状态。

RetrieveAll 按固定顺序遍历槽位，对非空元素计数并做分块前缀和，再稳定压缩到输出数组。单次调用内不使用全局原子抢占输出位置。其跨重建顺序保证按 2.4 节评审结果确定；若需规范排序，该排序必须包含在端到端时延中。

### 3.5 内存与性能优化方向

容器基本存储量为 `Capacity × sizeof(Key)`，另有标量和必要的工作空间。普通查询直接访问输入和表；Count 优先采用块级归约，Retrieve 的工作空间用于计数和前缀和，不额外保存完整输入副本。禁止为绕过验收计算而缓存固定测试输入或预先计算输出。

优化重点为探测访问的访存效率、重复键导致的原子竞争、查询命中率导致的探测长度差异，以及检索的计数/压缩带宽。根据官方 msopprof 数据分析核执行及相关访存指标，结合端到端计时选择核数、线程数、桶布局和归约粒度。所列方向为待验证方案，不代表已有性能结果。

### 3.6 支持硬件与环境

| 项目 | 要求 |
| --- | --- |
| 硬件 | Atlas 950 系列及后续支持 SIMT 的昇腾产品。 |
| CANN | 9.0.0-beta.2 或更高版本。 |
| 编译器 | 仓库支持的 ccec 或毕昇 ASC 编译器，目标架构以安装版本支持值为准。 |
| C++ / 构建 | 沿用仓库 C++17，CMake 不低于 3.16。 |
| 功能测试 | Catch2 不低于 3.5.4。 |

## 4. 可维可测分析

### 4.1 精度标准与功能测试

状态变化、键、布尔值和计数须与参考语义精确一致，不使用浮点误差容限。Host 端使用 `std::unordered_multiset` 计算各键出现次数及期望结果；另按相同合法输入与固定参考版本核对语义。顺序验证与多值集合内容验证分别记录，不能把排序后相等当作原始输出顺序一致。

官方附件声明的功能用例展开数量如下，执行时以 Catch2 实际展开日志核对覆盖：

| 接口 | I32/I64 用例合计 |
| --- | ---: |
| Create | 14 |
| Destroy | 18 |
| Clear | 32 |
| Insert | 194 |
| InsertIf | 20 |
| Contains | 100 |
| ContainsIf | 56 |
| Find | 100 |
| FindIf | 56 |
| Count | 124 |
| CountEach | 122 |
| CountEachOuter | 122 |
| Retrieve | 98 |
| RetrieveAll | 202 |
| 合计 | 1258 |

保留全部官方断言。附件的公共辅助头同时依赖 StaticMultimap，接入时仅分离与 multiset 无关的依赖，不要求实现另一容器。额外验证全部相同键、重复查询、未命中、空输入、非法容量、保留空键、I64 边界、输出不足、移动与释放、满表探测终止，以及 uint64 累计溢出处理。

Retrieve 补充逐键输出频数的精确比较，避免只验证总数和键存在。重复执行测试分别覆盖同一表多次查询、Clear 后重建、不同输入顺序和容量不足；其中涉及顺序或保留集合的期望值按 2.4 节确认后的契约设置。

### 4.2 性能标准与测试方法

每个官方性能用例要求：

```text
T_950 <= T_reference / 0.4
speedup = T_reference / T_950 >= 0.4
```

任务书允许未达标时给出合理解释；解释是否被接受由验收方决定，不将其视为免测或通过。测试矩阵如下，全部 44 行标杆时延采用官方任务书 3.3 节给出的毫秒数值。

| 接口 | 每种 Key 的参数 | I32/I64 用例合计 |
| --- | --- | ---: |
| Create、Destroy | Capacity=1e8。 | 4 |
| Insert、RetrieveAll | NumInputs=1e8，Occupancy=0.5，Multiplicity=1。 | 4 |
| Contains、Find | UNIQUE，NumInputs=1e8，Occupancy=0.5，MatchingRate=0.1/0.5/1。 | 12 |
| Retrieve、Count、CountEach、CountEachOuter | UNIFORM，NumInputs=1e8，Occupancy=0.5，Multiplicity=1，MatchingRate=0.1/0.5/1。 | 24 |
| 合计 | 使用附件原始参数组合。 | 44 |

例如 I32 Insert 的标杆为 8.098100 ms，时延上限为 20.245250 ms；I32 RetrieveAll 的标杆为 0.660264 ms，上限为 1.650660 ms。其余用例逐行换算，不以总体平均加速比替代逐例门槛。

保留附件数据生成方式；其中标注 UNIFORM 且 Multiplicity=1 的性能文件实际生成连续唯一键，不擅自改为另一种随机分布。除 Create/Destroy 外，上述占用率对应请求容量 2e8，实际容量以容器返回值为准。

附件以主机墙钟包围同步接口调用，不计 Setup 中的数据生成与输入复制。Create/Destroy 按对应文件界定构造和释放范围；Insert 测量后 Clear 恢复状态。测试框架最多测量 50 次并按累计时间、波动条件提前结束，报告算术平均值；框架没有单独预热阶段，不自行替换为最小值或中位数。

附件同时向 CPU/Device 时间字段填入同一墙钟时延，报告中标明该事实，单位由微秒换算为毫秒。必须将设备完成等待纳入同步调用，不能以提交开销作为算子时延。官方 msopprof 的核耗时另行记录，不替代附件的端到端测量。

### 4.3 内存与兼容性验证

记录静态存储、输出及工作空间的峰值，确认没有与输入规模线性重复的额外输入拷贝。对构造、Clear、移动和析构执行循环检查，确认资源释放与异常路径行为；检索容量不足时检查输出保护区域，排除越界写入。

通过 `tests/CMakeLists.txt` 注册功能与性能目录，使用仓库构建脚本编译并定向运行 static_multiset 测试。记录代码版本、CANN/编译器/设备版本、输入参数、精确比较结果、端到端时延及完整日志。源码交付时补齐 README 的构建、运行和复现步骤。

这是新增容器，已有 StaticSet/StaticMap 的去重、查找和计数语义保持其原定义。必要的公共集成变更须运行对应回归测试；本设计不声明实现、自测或外部验收已经完成。

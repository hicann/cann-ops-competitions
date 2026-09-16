# Ascend 950 StaticMultimap 容器设计

# 需求背景（required）

## 需求来源

本设计对应 [9 月社区任务：mulimap 容器开发（950）](https://www.hiascend.com/activities/task-center/details/f14ba9cfcd154c5eab3a615f6b5ab469?menu=tasks)，以[官方任务书及测试附件](https://www.hiascend.com/p/resource/202609/758f7f2bb7fb4d428c512243afb9a939.zip)为验收依据，目标代码仓为 [cann/ops-collections](https://gitcode.com/cann/ops-collections)。

参考接口为 NVIDIA cuCollections 的 [static_multimap](https://github.com/NVIDIA/cuCollections/blob/532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4/include/cuco/static_multimap.cuh)。本文以该固定版本说明适配关系，不假定后续 dev 分支接口保持不变。

## 背景介绍

StaticMultimap 是固定容量的多值键值映射容器。同一个 Key 可关联多个 Value，完全相同的键值对也可以重复存储，适用于需要保留重复记录的关联、计数和批量检索。

ops-collections 已有 StaticMap、StaticSet、哈希函数、探测策略、存储及 ACL 流管理组件。StaticMap 的重复键插入会返回失败，单键 Count 等价于成员关系，不能直接作为多值映射使用。本设计沿用纯头文件容器工程模式，在独立 StaticMultimap 实现中增加多值插入、完整匹配计数和检索能力。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas 950 及后续支持 SIMT 的产品上实现容器的构造、析构、Clear、Insert、InsertIf、Contains、ContainsIf、Find、FindIf、Count、Retrieve 和 RetrieveAll。Key、Value 支持 I32、I64，任务验收组合为 I32/I32、I64/I64。

## 需求拆解

1. 容器容量固定，实际 Capacity 不小于请求容量；每个成功插入的键值对占一个逻辑元素，Size 包含重复次数。
2. 支持空输入、重复键、完全重复的键值对、不同 multiplicity、命中与未命中，以及容量边界。
3. 整数结果按精确语义比较；计数使用 uint64，检索不能遗漏或合并重复记录。
4. 按任务书的 32 个性能用例逐项满足 `T_NPU <= T_baseline / 0.4`；未达到指标时按任务书提交原因分析，不以整体平均性能代替逐项检查。
5. 除静态存储、输出及必要工作空间外，不重复复制整份输入到 Device 内存。

# 详细设计（required）

## 算子分析

### 数学定义与功能语义

令容器为键值对多重集 M，`m(k, v)` 为 `(k, v)` 的重复次数，`c(k) = sum_v m(k, v)`。

- `Size = sum_k c(k)`，每次成功插入都增加一个元素，包括完全重复的键值对。
- 对查询数组 Q，`Contains(Q[i]) = (c(Q[i]) > 0)`。
- `Count(Q) = sum_i c(Q[i])`，重复查询按其出现次数重复计数。
- Retrieve 对每个 Q[i] 输出全部匹配 `(Q[i], v)`，每个 Value 输出 m(Q[i], v) 次；未命中不输出。
- RetrieveAll 输出 M 的全部元素，保留每个键值对的重复次数。

例如，M 为 `{(1, 10), (1, 20), (1, 20), (2, 30)}`，查询 Q 为 `[1, 1, 3]`，Count 返回 6，Retrieve 输出两组 `(1, 10), (1, 20), (1, 20)`。

### 确定性规则

任务书要求重复执行时容器状态和输出一致，而参考库不规定 Find 的多值选择和检索顺序。为给出可测试的方案，本文拟采用以下明确规则，并在文末集中提出评审确认事项：

1. Find/FindIf 命中时选择匹配 Value 中的最小值；未命中或条件不成立时写 emptyValue。
2. Retrieve 按查询索引递增输出，每个查询的匹配 Value 按数值递增输出，保留重复次数。
3. RetrieveAll 按 `(Key, Value)` 字典序递增输出，保留重复次数。
4. 容量不足时，在合法且被谓词选中的输入中，按输入索引顺序接纳剩余容量能够容纳的前缀，其余返回失败。

同样初始逻辑内容和操作序列应得到同样的键值对多重集和可观察结果；内部物理槽位布局不作为容器状态对外暴露。采用这些规则需要将选择、前缀和与排序成本计入对应接口。若验收只要求无序内容等价，应通过设计评审明确后再调整排序要求，不能依据现有测试未检查顺序而自行取消任务书要求。

### 数据类型、形状和约束

| 参数 | 数据与位置 | 约束 |
| --- | --- | --- |
| capacity | Host 侧 Extent，单位为元素个数。 | 必须大于 0，容量和字节数计算不得溢出。 |
| pairs | Device 侧连续 `Pair<Key, Value>[n]`。 | Key/Value 类型匹配模板；插入保留重复元素。 |
| keys | Device 侧连续 `Key[n]`。 | 查询采用相同 Key 类型的精确相等匹配。 |
| stencil | Device 侧连续 `uint32_t[n]`。 | 与输入一一对应，由默认构造的设备谓词决定是否操作。 |
| output | Device 侧 bool 或 Value 数组。 | Contains/Find 输出分别至少容纳 n 个元素。 |
| probeOut、valueOut | Device 侧 Key、Value 数组。 | Retrieve 的两路输出都至少容纳 outputCapacity 个对应类型元素。 |
| keyOut、valueOut | Device 侧 Key、Value 数组。 | RetrieveAll 的两路输出都至少容纳 outputCapacity 个对应类型元素。 |
| emptyKey、emptyValue | 构造时指定的哨兵值。 | 按参考库有效域保留这两个值；官方测试使用对应类型的最小值。 |
| stream | 调用方的 ACL 流。 | 支持显式流；默认流沿用 ACL 语义，由调用方完成 ACL 初始化和设备选择。 |

非零长度输入与输出不得互相覆盖，调用期间不得释放。调用方保证裸指针真实类型和分配长度与声明一致；模板不能从 void* 推断真实 dtype 或缓冲区大小。模板约束类型和谓词可调用性，Host 检查可见的指针、数量及溢出，Device 检查需要读取数据才能判断的保留值。

零长度调用允许空数据指针。对于零个匹配的检索，输出容量为 0 时也允许空输出指针。不同流上对同一容器的修改与查询必须由调用方建立顺序；不承诺无外部同步的跨流并发修改。

## 容器接口原型

接口采用 ops-collections 命名及附件的调用形式。以下为公共接口概要，设备引用接口位于独立 `static_multimap_ref.h`。

```cpp
namespace aclco {
template<class Key, class Value>
class StaticMultimap {
public:
    using ExtentType = aclco::Extent<std::size_t>;
    using SizeType = std::size_t;

    StaticMultimap(ExtentType capacity, Key emptyKey, Value emptyValue,
                  aclrtStream stream = nullptr);
    ~StaticMultimap();
    StaticMultimap(StaticMultimap const&) = delete;
    StaticMultimap& operator=(StaticMultimap const&) = delete;
    StaticMultimap(StaticMultimap&&);
    StaticMultimap& operator=(StaticMultimap&&);

    SizeType Capacity() const noexcept;
    SizeType Size(aclrtStream stream);
    void Clear(aclrtStream stream);
    SizeType Insert(void* pairs, ExtentType n, aclrtStream stream);
    template<class StencilT, class Predicate>
    SizeType InsertIf(void* pairs, StencilT* stencil, ExtentType n,
                      aclrtStream stream);
    void Contains(void* keys, void* output, ExtentType n, aclrtStream stream);
    template<class StencilT, class Predicate>
    void ContainsIf(void* keys, StencilT* stencil, void* output,
                    ExtentType n, aclrtStream stream);
    void Find(void* keys, void* output, ExtentType n, aclrtStream stream);
    template<class StencilT, class Predicate>
    void FindIf(void* keys, StencilT* stencil, void* output,
                ExtentType n, aclrtStream stream);
    uint64_t Count(void* keys, ExtentType n, aclrtStream stream);
    SizeType Retrieve(void* keys, ExtentType n, void* probeOut, void* valueOut,
                      ExtentType outputCapacity, aclrtStream stream);
    SizeType RetrieveAll(void* keyOut, void* valueOut,
                         ExtentType outputCapacity, aclrtStream stream);
};
} // namespace aclco
```

| 接口 | 返回值和行为 |
| --- | --- |
| 构造、析构 | 采用 RAII 完成任务书的 Create、Destroy；禁止复制，移动转移资源所有权。 |
| Capacity、Size、Clear | Capacity 返回实际容量；Size 返回全部元素数量；Clear 清空但保留容量，允许重复调用。 |
| Insert | 返回插入失败数量，重复键和重复键值对不算失败；非零长度空输入返回 n，容器不变。 |
| InsertIf | 返回谓词成立且未插入的元素数，未选中的元素不计失败。非零长度空输入或空 stencil 抛参数异常；只对选中元素检查保留值。 |
| Contains、ContainsIf | 输出逐元素成员关系；条件不成立时写 false。 |
| Find、FindIf | 按上述确定性规则输出 Value；条件不成立或未命中时写 emptyValue。 |
| Count | 返回 uint64 标量总匹配数，不是逐查询数组。 |
| Retrieve、RetrieveAll | 返回输出数量；在写出前检查所需数量是否超过 outputCapacity，容量不足时报错，不截断匹配结果。 |

所有要求的批量接口为同步接口，返回前完成相关设备操作并检查 ACL 执行错误。Host 参数错误使用异常报告，容量不足的正常插入使用失败数报告。插入被选中元素含保留值时，先完成校验，再决定是否修改容器；校验失败时容器保持不变。非零长度查询的非法指针按参数错误处理。数量或总匹配数溢出时报告错误，不能截断为 32 位。

## 算子实现

### Host 侧设计

Host 侧负责容量计算、静态存储和工作空间分配、参数检查、ACL 流上的核函数组织及同步。构造初始化空槽和 Size，析构释放所拥有资源；移动后的对象不再拥有原资源。

容量按存储桶要求向上取整，Capacity 返回实际可用元素数。插入前校验输入，条件插入统计选中数量。若全部可容纳，直接进入批量插入；仅容量不足时执行选中标志的前缀和，确定接纳的输入索引前缀。已存在元素保持不变，成功插入数更新 Size。

对检索，先获得输出数量并检查容量，再安排前缀和、写出及所需排序。前缀和、归约和排序只使用必要工作空间，不在接口内部额外保留整份输入的 Device 副本。保留值检查、主机回读、同步和必要分配均属于对应同步接口的计时范围。

### Device 侧设计

采用开放寻址哈希表，每个槽保存一个键值对，相同 Key 的每个元素分别占槽。查询与插入共用哈希和探测序列，探测必须完整覆盖容量范围并有终止上限；没有 Erase 操作，Clear 统一恢复空槽，不引入删除墓碑。

插入遇到相等 Key 时继续寻找空槽，不能复用 StaticMap 的重复键失败判定。I32/I32 评估使用 64 位打包 CAS；I64/I64 使用键原子占槽和随后写入 Value，插入期间不与查询并发，后续查询在完整插入核结束并同步后运行。插入线程只需读取槽位 Key 判断占用，不依赖尚未发布的 Value；不假设存在 128 位 CAS，也不以永久轮询 Value 哨兵处理非法输入。原子 API、对齐和内存可见性按目标 CANN 编译器及设备验证。

Contains 遇到首个匹配即可结束。Count 和 Find 扫描完整有效探测范围，分别累加全部重复次数和选择最小 Value；在没有并发修改及删除的前提下，遇到探测序列上的首个空槽即可结束，满表时以遍历上限结束。

Count 在线程内累计，再做分层归约，使用 64 位中间值，避免每个命中都竞争单个全局计数器。条件查询直接写 false 或 emptyValue，不能依赖调用方预清零。

Retrieve 先逐查询计数、前缀和，再写入各查询的独立输出区间；区间内按 Value 排序，实现查询索引顺序和多值顺序的确定性。RetrieveAll 扫描有效槽并压缩到输出，然后按 Key、Value 排序。相同键值对排序后的先后不可区分，不影响逐元素结果。单元素区间省略排序；不得以仅测试 multiplicity=1 为由省略多值路径。

### 分核和性能优化

根据输入规模和实际 AIV 核数划分任务，使用 SIMT 线程处理输入或查询，调整探测桶大小与并行粒度，减少高占用率下的随机访存和重复探测。Clear、有效槽扫描、前缀和及归约优先使用连续访问；所有偏移计算保持足够位宽。

性能测试使用 1e8 个输入、占用率 0.5，即查询/插入场景请求容量为 2e8；Create/Destroy 的请求容量为 1e8。仅键值对槽位约占 1.6 GB（I32/I32）或 3.2 GB（I64/I64，十进制），另计输入、输出及工作空间。应报告内存峰值，并验证工作空间生命周期和重复调用的内存复用。

确定性排序可能成为 RetrieveAll 的主要成本，必须计入接口时延并单独分析；不能将排序移到测试计时之外。根据官方 msopprof 的核耗时、访存及实际可用指标定位瓶颈，结合有无重复键、不同占用率的对照验证优化效果。

### 工程组织

```text
ops-collections/
├── include/static_multimap.h
├── include/static_multimap_ref.h
├── include/detail/static_multimap/
├── tests/static_multimap/
├── tests/performance/static_multimap/
└── docs/static_multimap_API文档和使用示例.md
```

更新 tests/CMakeLists.txt 的功能与性能测试目录，以及仓库 README 的构建和运行入口。官方公共测试头同时引用 StaticMultiset，接入本任务时仅分离无关依赖，保持全部 StaticMultimap 测试断言。设备引用复用类型、哈希和存储基础组件，多值插入与计数实现保持独立，不改变现有 StaticMap/StaticSet 行为。

## 支持硬件与构建环境

| 项目 | 要求 |
| --- | --- |
| 硬件 | Atlas 950 及后续支持 SIMT 能力的昇腾产品。 |
| 软件 | CANN 9.0.0-beta.2 及以上。 |
| 编译器 | 仓库支持的 ccec 或毕昇 ASC 编译器，架构参数按安装版本确定。 |
| 工具 | CMake >= 3.16，Catch2 >= 3.5.4，按仓库 C++17 配置构建。 |

## 算子约束限制

本任务不提供动态扩容、Erase、异构键匹配或无同步的跨流并发修改。Key/Value 的哨兵值不能作为有效插入数据。内存分配与输出数量受设备可用容量限制，错误必须有界返回，不允许满表或哨兵值引发无限等待。

# 可维可测分析

## 精度标准与功能测试

以任务书及附件的 1086 个展开功能用例为基础，覆盖 12 个接口和两种验收类型。使用独立 Host 多重映射作为 oracle，逐项核对容器元素及重复次数、条件选择、未命中、零输入、容量边界和 Clear 后重用。

Find 除检查结果属于匹配集合外，另检查最小 Value 规则；Retrieve/RetrieveAll 除排序后的多重集等价外，按本文规定直接比较原始数组。对同一容器重复调用，并用相同输入重新构建后重复调用，验证结果和逻辑内容一致。满表部分插入应验证接纳的输入前缀，不能仅检查失败数量。

补充非零长度空指针、空 stencil、容量 0、输出容量不足、保留值、合法 I64 边界、完全重复的键值对、超出 32 位的累计计数和多批插入测试。为容量不足的输出检查保护区，确认没有越界写入；为异常输入检查容器未被部分修改。I64/I64 的键值关联应在高冲突输入下逐对验证，防止读取到未完成的 Value。

## 性能标准与测试方法

| 用例组 | 参数 | 用例数 |
| --- | --- | --- |
| Create、Destroy | 两种类型，Capacity=1e8。 | 4 |
| Insert、RetrieveAll | 两种类型，NumInputs=1e8，Occupancy=0.5，Multiplicity=1。 | 4 |
| Contains、Find、Retrieve、Count | 两种类型，NumInputs=1e8，Occupancy=0.5，Multiplicity=1，MatchingRate=0.1/0.5/1。 | 24 |

逐行使用任务书第 3.3 节标杆值，换算为 `T_baseline / 0.4` 的时延上限。例如 RetrieveAll 的 I32/I32 上限为 3.6727725 ms，I64/I64 上限为 8.5543825 ms；Insert 对应上限为 24.9621675 ms 和 34.98532 ms。全部 32 行均需报告，不能只选择达标子集。

沿用附件 Measure 与目标仓性能框架，记录同步接口的 CPU 墙钟时延及框架均值；将微秒换算为毫秒后比较。附件 CPU/Device 两个字段来自同一个墙钟读数，不能解释为独立设备核耗时。Setup 位于计时循环外，Insert 后的 Clear 位于单次计时外；Create 计构造、Destroy 计析构，遵循原始测试边界。函数内部的校验、同步和排序不得移出计时。

每条记录包含类型、规模、占用率、重复度、匹配率、迭代数、时延与对应标杆；记录设备、CANN、编译器和源码版本。Profiling 使用官方 msopprof，设备核分析与附件端到端计时分开。官方材料未给出完整标杆平台和采集环境，不能自行归因为某一 GPU 型号。

测试接入后按目标仓入口构建和运行：

```bash
bash scripts/build.sh -b
bash scripts/build.sh -r --test-name static_multimap
bash scripts/build.sh -p
```

性能测试逐个运行 `build/performance/static_multimap/` 中对应可执行文件，保留全部功能和性能日志。

## 兼容性分析与评审确认事项

本设计新增 StaticMultimap，保持现有容器接口不变。生命周期、失败数及查询形式沿用 StaticMap/StaticSet 和本任务附件；以下差异需在设计评审中确认，不以本文替代原始任务书：

1. 任务书通用表要求数量不超过容量且非法指针报错，而附件要求超容量部分插入及 `Insert(nullptr, n)` 返回 n。本文采用附件和已有静态容器行为；InsertIf 的非零长度非法指针仍按任务书抛异常。
2. Count 按附件、目标库及参考库返回标量总数，任务书通用表的 `[NumInputs]` 输出数组描述不适用于该接口。
3. 参考库 Retrieve 的 output_match 是完整键值对，附件为 Value 数组。本文采用后者；在同类型精确相等匹配中，probe key 可恢复对应存储 key，因此保持键值关联与重复次数。查询参数顺序和模板谓词也按附件适配，不表述为与参考库原型逐参数相同。
4. 本文为任务书的确定性要求提出最小值选择、固定输出顺序及超容量前缀接纳规则。请确认是否要求这种重新构建后的可观察结果一致性，或以无序多重集等价为验收标准；排序成本会影响性能方案。
5. 运行时检查仅覆盖接口可获得的信息，调用方保证实际缓冲类型和长度；emptyKey/emptyValue 均作为保留值。请确认不需要另增带 dtype 和长度信息的张量描述符接口。

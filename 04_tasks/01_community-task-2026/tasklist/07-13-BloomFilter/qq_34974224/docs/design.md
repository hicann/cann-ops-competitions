# BloomFilter 容器设计文档

# 需求背景（required）

## 需求来源

CANN 社区 2026 年 7 月任务 07-25-BloomFilter。参考 cuCollections 的
`cuco::bloom_filter`，在昇腾 NPU 上使用 Ascend C 实现功能一致的容器，
适配 Atlas 950 系列产品，最终交付到 `cann/ops-collections`。

## 背景介绍

BloomFilter 使用固定长度位数组表示键集合。查询返回 `false` 表示键一定未插入，
返回 `true` 表示键可能已插入；允许假阳性，不允许假阴性。它不保存原始键，
因此不提供枚举、还原或单键删除。

本设计采用 blocked Bloom Filter：位数组划分为等长 block，同一键的全部位模式
落在同一个 block 内，以减少随机访存范围。任务书默认配置为每 block
`8 × uint32_t = 256 bit`。

# 需求分析（required）

## 需求描述

新增头文件式容器 `aclco::BloomFilter`，支持构造、析构、`Add`、`Contains`、
`Clear`、`Merge`、`Intersect`，并提供对应异步接口。key 支持
`uint32_t`、`uint64_t` 和 `float`。

接口形态遵循 `ops-collections` 现有容器：设备地址使用 `void*`，数量使用
`Extent`，执行流使用 `aclrtStream`。哈希使用任务书指定的 `XXHash_64`。

## 需求拆解

- `BloomFilterPolicy`：描述 word、block、位模式和协作布局。
- `BloomFilterRef`：不持有存储，供自定义 SIMT kernel 逐 key 调用。
- `BloomFilter`：持有设备位数组和 Add 临时区，完成校验、生命周期和 kernel 下发。
- 设备实现：Add、Contains、Clear、Merge、Intersect。
- 验证：独立 CPU golden、功能/泛化测试、任务书性能矩阵和自验证报告。

# 详细设计（required）

## 算子分析

> 本任务交付 C++ 容器模板而非张量算子。本章中的“算子”指该容器及其设备 kernel。

### 数学定义

设过滤器有 `B` 个 block，每 block 有 `W` 个 word，每 word 有 `L` 位。
总位数为 `m = B × W × L`。对 key 计算 64 位哈希 `h`：

- 哈希高 32 位通过 multiply-shift 映射到 block；
- 哈希低 32 位结合固定 salt 生成各 word 的 bit mask；
- `Add` 对目标 block 置位；
- `Contains` 检查目标位是否全部存在；
- `Merge` 对两个兼容位数组逐 word 求或；
- `Intersect` 对两个兼容位数组逐 word 求与。

任务书配置为 `Word=uint32_t`、`W=8`、`PatternBits=8`，每个 key 在 block
的 8 个 word 中各置一位。

### 支持数据类型与形状

| 对象 | 类型/形状 |
| --- | --- |
| key | `uint32_t` / `uint64_t` / `float` |
| Word | 默认 `uint32_t`，模板保留 `uint64_t` 能力 |
| Add 输入 | `keys[keyNum]` |
| Contains 输入输出 | `keys[keyNum] -> uint8_t outputValues[keyNum]` |
| 位数组 | `words[numBlocks × WordsPerBlock]` |
| stream | `aclrtStream` |

`keyNum` 不要求按线程数、核数或协作组对齐。设备 kernel 参数使用 32 位计数，
超过 `UINT32_MAX` 的输入主动拒绝。

float key 按原始 bit pattern 哈希，因此 `+0.0f` 与 `-0.0f` 是不同键，
不同 payload 的 NaN 也是不同键。

### 公开接口

```cpp
template <class Key,
          class Extent = aclco::Extent<std::size_t>,
          class Word = std::uint32_t,
          std::uint32_t WordsPerBlock = 8,
          std::uint32_t PatternBits = 8,
          std::uint32_t AddHorizontalLayout = WordsPerBlock,
          std::uint32_t AddVerticalLayout = 1,
          std::uint32_t ContainsHorizontalLayout = WordsPerBlock,
          std::uint32_t ContainsVerticalLayout = 1,
          class Hash = detail::XXHash_64<Key>,
          class Allocator = DefaultAllocator<Word>>
class BloomFilter {
 public:
  using PolicyType = BloomFilterPolicy<Key, Word, WordsPerBlock, PatternBits,
                                       AddHorizontalLayout, AddVerticalLayout,
                                       ContainsHorizontalLayout, ContainsVerticalLayout,
                                       Hash>;
  using RefType = BloomFilterRef<Key, PolicyType>;

  BloomFilter(Extent numBlocks,
              Hash const& hash = {},
              Allocator const& allocator = {},
              aclrtStream stream = nullptr);
  ~BloomFilter();

  BloomFilter(BloomFilter const&) = delete;
  BloomFilter& operator=(BloomFilter const&) = delete;
  BloomFilter(BloomFilter&&) noexcept;
  BloomFilter& operator=(BloomFilter&&) noexcept;

  void Add(void* keys, Extent keyNum, aclrtStream stream);
  void AddAsync(void* keys, Extent keyNum, aclrtStream stream);

  void Contains(void* keys, void* outputValues, Extent keyNum,
                aclrtStream stream) const;
  void Contains(void* keys, Extent keyNum, void* outputValues,
                aclrtStream stream) const;
  void ContainsAsync(void* keys, void* outputValues, Extent keyNum,
                     aclrtStream stream) const;
  void ContainsAsync(void* keys, Extent keyNum, void* outputValues,
                     aclrtStream stream) const;

  void Clear(aclrtStream stream);
  void ClearAsync(aclrtStream stream);
  void Merge(BloomFilter const& other, aclrtStream stream);
  void MergeAsync(BloomFilter const& other, aclrtStream stream);
  void Intersect(BloomFilter const& other, aclrtStream stream);
  void IntersectAsync(BloomFilter const& other, aclrtStream stream);

  std::uint32_t NumBlocks() const noexcept;
  Extent BlockExtent() const noexcept;
  std::uint32_t NumWords() const noexcept;
  std::size_t SizeInBytes() const noexcept;
  Word* Data() noexcept;
  Word const* Data() const noexcept;
  RefType Ref() const noexcept;
};
```

`BloomFilterRef` 提供设备侧标量与协作版本的 `Add` / `Contains`。协作布局满足
`HorizontalLayout × VerticalLayout = WordsPerBlock`。

### 接口约束

| 项目 | 约束 |
| --- | --- |
| `numBlocks` | `0 < numBlocks ≤ UINT32_MAX / WordsPerBlock` |
| `keyNum` | `[0, UINT32_MAX]` |
| 输入地址 | `keyNum > 0` 时必须是 device 可访问的非空连续数组 |
| Merge/Intersect | 两侧模板类型、block 数和 hash seed 必须一致 |
| 异步生命周期 | 输入输出缓冲区和容器必须存活到 stream 上任务完成 |

容器不可拷贝、可以移动。移动、移动赋值和析构不隐式等待未知 stream，
调用方必须先完成该对象的异步操作。

## 算子实现

### host 侧

构造函数申请位数组并清零；同步接口在异步下发后等待 stream；异步接口只负责参数校验
和 kernel 下发。`Clear`、`Merge`、`Intersect` 按 word 连续处理。

归组式 Add 需要两份 8 字节中间记录数组，临时显存约为
`16 × keyNum + O(AIV核数 × 一级桶数)` 字节。80,000,000 个 key 约需 1.28 GB。
临时区按需申请、随对象复用和释放；扩容前先等待旧任务，避免释放仍被 kernel 使用的内存。

同一对象在不同 stream 上连续调用 `AddAsync` 时，实现会先等待上一条 Add stream，
避免临时区和普通 RMW 并发复用。同一对象不支持多个 host 线程并发调用成员函数。
其它跨 stream 操作仍由调用方用 stream/event 建立依赖。

### kernel 侧

设备代码采用 Atlas 950 的 SIMT 官方范式：模板化 `__global__` 入口，
`AscendC::Simt::VF_CALL` 调用 `__simt_vf__`，`LAUNCH_BOUND` 与 `Dim3`
使用同一线程数，按 AIV 核数和 grid-stride 覆盖输入。

Contains 默认由 8 个 lane 协作查询一个 block，并用 shuffle 做组内 AND 归约。
Clear、Merge、Intersect 按 word 平铺，连续读写且每个 word 只有一个写者。

### Add 设计思路

直接对随机 block 发 `atomic_or` 在 Atlas 950 上代价较高。默认 Add 先把 key
归组，再让唯一线程组负责一个 block，从而使用无原子的普通 RMW：

1. 各核统计 key 落入一级桶的数量；
2. 对计数做前缀和，得到桶的连续区间；
3. 散布 `(block, hashLo)` 中间记录；
4. 一个 AIV block 独占一个一级桶，桶内按 `(tile, block owner)` 再归组，
   每个 8-lane 组依次处理自己拥有的 block，8 个 lane 各写一个 word。

正确性来自写者排他：一级桶跨核不重叠；桶内 tile 串行；同一 block 始终映射到
同一个 owner 组；组内 lane 写不同 word。因此任意两个线程不会同时普通写同一个 word，
最终位数组与全原子 OR 逐位一致。

任务书默认配置使用 2048-block tile 和最多 512 个一级桶。32/256/2048 MB
三档分别形成 `r2=1/8/64`。不满足归组计划的合法规模自动回退到正确的全原子路径，
功能语义不变。按 56 个 AIV 核换算，当前归组优化窗口约为 3.5 MB 到 2048 MB；
该窗口之外只保证功能正确，不承诺 Add 性能。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950 系列 | √ |
| Atlas 800I/T A2 |  |
| Atlas A3 |  |
| Atlas 300I Duo / 推理系列 |  |

实现依赖 DAV_3510 SIMT 能力，早期架构不在本次范围。

## 算子约束限制

- BloomFilter 允许假阳性，但对正确完成的 Add 不允许假阴性。
- Intersect 是位数组求交，不等价于精确 key 集合求交。
- 不支持单 key 删除、枚举、还原或精确计数。
- 自定义 Hash 需要产生 64 位结果，并可由一个 seed 完整重建。
- 归组 Add 的临时显存与 key 数量成正比，调用方必须把它计入显存预算。
- 不满足归组计划的规模会回退到全原子 Add，正确性不变、性能可能明显下降。

# 可维可测分析

## 精度标准

CPU golden 独立实现哈希、block 选择和位模式，不调用被测 BloomFilter。功能验收采用：

- Add 后位数组与 CPU golden 逐 word 二进制一致；
- 已插入 key 的 Contains 全部为 true，零假阴性；
- Clear 后位数组全 0；
- Merge 等于 CPU 逐 word OR；
- Intersect 等于 CPU 逐 word AND。

覆盖 `uint32_t`、`uint64_t`、`float`，以及空输入、重复 key、非对齐尾部、
float 特殊位模式、移动语义、错误参数和头文件共存。

验收加固测试额外覆盖：

- 32/256/2048 MB 三个实际计划形状 `r2=1/8/64`；
- 三种 key 类型在归组路径上的完整位数组对拍；
- AddAsync 临时区扩容与跨 stream 顺序；
- 80,000,000 个 `uint32_t` key、2048 MB 过滤器的完整 CPU golden 对拍。

## 性能标准

只使用 Atlas 950 真机数据与任务书比较。每一项按任务书规则判断：

- I32：`实测时间 ≤ Total Time / 0.8`；
- I64：`实测时间 ≤ Total Time / 0.6`。

性能用例覆盖构造、析构、Add、Contains、Clear、Merge、Intersect，
过滤器规模为 32/256/2048 MB；Add 与 Contains 的输入数量为 80,000,000。
记录同一 commit、同卡同会话的原始日志、重复次数、均值和离散度。
simulator 和非 950 环境只用于定位与编译，不用于声明性能达标。

Add 需分别记录首次冷调用和稳定态调用。首次调用包含临时区分配；
任务书既有性能 harness 的正式判定口径保持不变，不通过虚构公开接口把分配移出计时区。

## 兼容性分析

BloomFilter 按 `ops-collections` 的头文件容器结构新增，不修改 StaticSet、StaticMap、
DynamicMap 的公开接口。公共哈希层只新增 `XXHash_64` 与别名，既有 `XXHash_32`
行为不变。

BloomFilter kernel 放在 `aclco::detail` 命名空间，并通过共存编译用例保证
`bloom_filter.h` 可与仓内其它容器头同时包含。代码采用模板化 kernel，
避免头文件跨翻译单元的 ODR 冲突。

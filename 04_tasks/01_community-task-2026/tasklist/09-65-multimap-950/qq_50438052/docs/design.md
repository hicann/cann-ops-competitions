# static_multimap 容器设计文档（Atlas 950 / Ascend C）

> 对应任务：9月社区任务-mulimap容器开发(950)（任务编号 65）
> 目标仓库：[cann/ops-collections](https://gitcode.com/cann/ops-collections)
> 对标实现：[NVIDIA cuCollections static_multimap](https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/static_multimap.cuh)

# 一、需求背景（required）

## 1.1 需求来源

通过社区任务向昇腾开源仓 ops-collections 贡献 `static_multimap` 容器：参考 cuCollections 的 `static_multimap`，在 Atlas 950 上使用 Ascend C（SIMT + SIMD 混合编程）实现功能一致的静态容量多值键值映射，完成设计、开发、测试并合入 ops-collections。

## 1.2 背景介绍

### 1.2.1 static_multimap 实现优化

该容器没有 TBE 版本，参考实现为 cuCollections（GPU/CUDA）。昇腾侧基于 ops-collections 已有的开放寻址基础设施（`StaticMap` / `StaticSet` 使用的 Extent、Pair、探测策略、哈希函数、SIMT 核函数调度方式）实现，不修改既有容器的公开接口。

参考实现路径（cuCollections `dev` 分支，commit `532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4`）：

| 类别 | 路径 | 说明 |
| --- | --- | --- |
| 容器接口 | `include/cuco/static_multimap.cuh` | 模板参数、构造函数与全部批量接口 |
| 容器实现 | `include/cuco/detail/static_multimap/static_multimap.inl` | 接口到 `open_addressing_impl` 的转发；`retrieve_all` 以 zip 迭代器同时输出 key 与 value |
| host 侧实现 | `include/cuco/detail/open_addressing/open_addressing_impl.cuh` | 核函数下发、计数器、`retrieve_all` / `size` 的 CUB 实现 |
| 设备侧实现 | `include/cuco/detail/open_addressing/open_addressing_ref_impl.cuh` | 探测、插入 CAS、`count` / `retrieve` 的逐 key 逻辑 |
| 核函数 | `include/cuco/detail/open_addressing/kernels.cuh` | `insert_if_n`、`contains_if_n`、`find_if_n`、`count`、`retrieve` |
| 比较器 | `include/cuco/detail/equal_wrapper.cuh` | `AllowsDuplicates=true` 时插入跳过 key 比较 |
| 存储 | `include/cuco/detail/storage/bucket_storage.inl` | 槽数组申请与初始化 |

昇腾侧复用的 ops-collections 基础（commit `9d12996d4317e28420d74bcb1ec4d3b3507599ce`）：

| 路径 | 复用内容 |
| --- | --- |
| `include/extent.h`、`include/detail/extent/extent.inl` | `Extent`、`MakeValidExtent`（素数桶数 × bucket 大小） |
| `include/pair.h` | `Pair<Key, T>` 槽类型 |
| `include/probing_scheme.h` | `LinearProbing` / `DoubleHashing` 探测策略类型 |
| `include/hash_functions.h` | `murmurhash3_32`、`xxhash_32` |
| `include/utility/equal_wrapper.h`、`include/utility/atomic_cas_wrap.h` | key 比较、SIMT 原子 CAS |
| `include/detail/storages/bucket_storage_ref.h` | 设备端槽数组引用 |
| `include/detail/open_addressing/kernels.h` | 小表清空用的 SIMT `Clear` 核函数 |
| `include/static_map.h` | SIMT 核函数下发方式（`<<<核数, 0, stream>>>` + `VF_CALL`）与同步/异步接口风格 |
| `tests/common/*`、`tests/performance/performance_test_framework.h` | 测试公共件与性能测试框架 |

### 1.2.2 static_multimap 现状分析

#### 1.2.2.1 参考实现支持的数据类型和数据格式

| 参数 | 含义 | cuCollections 支持 | 本任务要求 | 数据排布 |
| --- | --- | --- | --- | --- |
| key | 键 | 可按位比较的平凡类型 | I32、I64 | 一维（ND 展平） |
| value | 值 | 平凡类型 | I32、I64，验收场景与 key 同类型 | 一维 |
| 插入输入 | 键值对 | `cuco::pair<Key, T>` | `aclco::Pair<Key, T>` | 一维 |
| stencil | `*_if` 条件模板 | 任意可被谓词调用的类型 | uint32 | 一维，与输入等长 |
| Contains 输出 | 是否存在 | 可由 bool 赋值的类型 | bool | 一维，与 key 等长 |
| Find 输出 | key 关联的 value | value 类型 | 与 value 相同 | 一维，与 key 等长 |
| Count 返回 | 匹配总数 | `size_type`（`std::size_t`） | uint64 | 标量 |
| Retrieve 输出 | 查询 key / 匹配的 value | key 类型 / value 类型 | 与 key / value 相同 | 一维，长度 = 匹配数 |
| RetrieveAll 输出 | 全部 key / value | key 类型 / value 类型 | 与 key / value 相同 | 一维，长度 = 元素数 |

cuCollections 默认配置：`double_hashing<8, default_hash_function<Key>>`（协作组大小 8，哈希为 xxhash_32），`storage<2>`（bucket 大小 2）。

#### 1.2.2.2 参考实现描述

cuCollections `static_multimap` 是基于开放寻址的静态容量多值哈希映射，槽类型为 `pair<Key, T>`：

1. **构造 / 清空**：容量上取整为「素数 × bucket 大小」，申请槽数组后用 `thrust::for_each_n` 并行把每个槽写成 (空 key, 空 value)。
2. **插入（insert / insert_if）**：`insert_if_n` 核函数对每个键值对计算双重哈希探测序列，逐 bucket 查找可用槽并 CAS 占用；CAS 失败继续检查后续槽。`AllowsDuplicates=true` 时跳过 key 相等比较，同一 key 的多个键值对（包括完全相同的键值对）都占用新槽。宽槽先 CAS key 再写 value。`insert_if` 返回成功插入数，`insert` 不返回值。
3. **查询（contains / find 及 `_if` 版本）**：沿探测序列遇到第一个相等 key 即命中，遇到空槽即判定不存在；`find` 输出命中槽的 value 或空 value 哨兵。
4. **计数（count）**：`ref.count(key)` 沿探测序列逐 bucket 统计相等槽，直到出现空槽；线程块内求和后原子累加到全局计数器。
5. **检索（retrieve）**：单遍完成。线程块共享缓冲区暂存 (probe key, 匹配槽)，缓冲区满或结束时用原子计数器 `fetch_add` 预留输出区间再写出 key 与 value，输出顺序取决于线程调度。
6. **全量检索（retrieve_all）/ size**：`retrieve_all` 用 zip 迭代器组合 key、value 两个输出，再以 `cub::DeviceSelect::If`（谓词「槽非空」）压缩整张表；`size` 用 `cub::DeviceReduce::TransformReduce` 统计非空槽。

#### 1.2.2.3 参考实现流程图

**构造 / Clear**

```mermaid
flowchart TD
    C1["MakeValidExtent：容量上取整为 素数 × 2"] --> C2["cuda_allocator 申请 pair 槽数组"]
    C2 --> C3["thrust::for_each_n 并行写 (empty_key, empty_value)"]
```

**insert / insert_if**

```mermaid
flowchart TD
    I1["host：counter.reset，计算 grid_size"] --> I2["insert_if_n 核函数：每线程取键值对"]
    I2 --> I3{"pred(stencil) ?"}
    I3 -- 否 --> I9["跳过"]
    I3 -- 是 --> I4["double_hashing 生成探测序列"]
    I4 --> I5{"bucket 内有空槽?"}
    I5 -- 否 --> I6{"回到起点?"}
    I6 -- 否 --> I4
    I6 -- 是 --> I8["插入失败"]
    I5 -- 是 --> I7{"CAS 成功?（宽槽先 CAS key 再写 value）"}
    I7 -- 否 --> I5
    I7 -- 是 --> I10["成功，计数器 +1"]
    I10 --> I11["host 读回成功数（insert_if）"]
```

**contains / find / count**

```mermaid
flowchart TD
    Q1["核函数：每线程取 key，生成探测序列"] --> Q2{"逐槽比较 key"}
    Q2 -- 相等 --> Q3["contains/find：命中，find 输出槽的 value<br/>count：计数 +1 继续"]
    Q2 -- 空槽 --> Q4["结束：未命中（find 输出 empty_value）/ 返回计数"]
    Q2 -- 不等 --> Q5["下一个槽 / 下一个 bucket"] --> Q2
    Q4 --> Q6["count：线程块求和 → 原子累加 → host 读回"]
```

**retrieve**

```mermaid
flowchart TD
    R1["retrieve 核函数：线程块共享缓冲区"] --> R2["探测，匹配 (probe, slot) 写入缓冲区"]
    R2 --> R3["缓冲区满或结束：atomic fetch_add 预留输出区间"]
    R3 --> R4["写出 probe key 与 slot.value，顺序依赖调度"]
    R4 --> R5["host 读回计数器 = 匹配数"]
```

**retrieve_all / size**

```mermaid
flowchart TD
    A1["transform_iterator 遍历全部槽"] --> A2["retrieve_all：zip(keys_out, values_out)，cub::DeviceSelect::If(slot_is_filled)"]
    A1 --> A3["size：cub::DeviceReduce::TransformReduce(slot_is_filled)"]
    A2 --> A4["同步 stream，返回两个输出的末尾"]
    A3 --> A4
```

### 1.2.3 ops-collections 现状

ops-collections 已有的 `StaticMap` 语义为「一个 key 只对应一个 value」：插入遇到相等 key 返回 DUPLICATE（或 InsertOrAssign 覆盖），`Count` 只返回 0/1，且没有 `Retrieve`、`RetrieveAll`、`Size` 接口；其默认探测策略为线性探测。因此需要新增允许重复 key 的设备端探测逻辑和配套核函数，不在 `StaticMap` 上加开关，以免影响已合入容器的行为与性能。

# 二、需求分析（required）

## 2.1 外部组件依赖

| 组件 | 版本 | 用途 |
| --- | --- | --- |
| CANN Toolkit | 9.0.0-beta.2 及以上（开发验证环境为 9.1.0） | Ascend C 头文件、runtime、`platform` / `tiling_api` 库 |
| 编译器 | bisheng（`--npu-arch=dav-3510`）或 ccec（`--cce-aicore-arch=dav-c310`） | 编译 Ascend C 核函数 |
| CMake | 3.16 及以上 | 构建 |
| Catch2 | v3.5.4（仅测试） | 功能测试框架 |

不引入新的第三方组件；容器本身为纯头文件。

## 2.2 内部适配模块

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| `Extent` / `MakeValidExtent`、`Pair` | 复用 | 容量合法化、槽类型 |
| `LinearProbing` + `murmurhash3_32`（默认）、`DoubleHashing` + `xxhash_32` | 复用 | 探测策略与哈希 |
| `EqualTo` / `EqualWrapper` / `AtomicCasWrap` | 复用 | 比较与原子 CAS |
| `BucketStorageRef` | 复用 | 设备端槽数组引用 |
| `aclco::Clear`（`kernels.h`） | 复用 | 小表 / 非重复字哨兵的逐槽清空 |
| `detail/open_addressing/multi_value_*.h` | 新增 | 允许重复 key 的设备端探测、SIMT 核函数、host 侧实现（与 `StaticMultiset` 共用） |
| `static_multimap.h`、`static_multimap_ref.h`、`detail/static_multimap/*` | 新增 | 容器公开接口与设备端引用 |
| `tests/static_multimap/`、`tests/performance/static_multimap/`、`tests/common/multi_container_test_common.h` | 新增 | 任务提供的功能 / 性能用例 |
| `tests/CMakeLists.txt` | 修改 | 把新测试目录加入构建（唯一改动的既有文件） |

## 2.3 需求模块设计

### 2.3.1 容器原型

```cpp
namespace aclco {
template <class Key,
          class T,
          class Extent = Extent<size_t>,
          class KeyEqual = aclco::EqualTo<Key>,
          class ProbingScheme = aclco::LinearProbing<aclco::murmurhash3_32<Key>>,
          class Storage = Storage<2>>
class StaticMultimap;
}
```

接口与 cuCollections 的对应关系（device iterator 对以「Device 首地址 + 元素个数」替代，个数的位置遵循 ops-collections `StaticMap` 与任务测试用例的约定）：

| 接口 | ops-collections 原型 | cuCollections 原型 | 说明 |
| --- | --- | --- | --- |
| Create | `StaticMultimap(Extent capacity, Key emptyKey, T emptyValue, KeyEqual const& pred = {}, ProbingScheme const& probingScheme = {}, Storage storage = {}, aclrtStream stream = nullptr)`；`StaticMultimap(Extent capacity, Key emptyKey, T emptyValue, aclrtStream stream)` | `static_multimap(capacity, empty_key_sentinel, empty_value_sentinel, pred, probing_scheme, storage, alloc, stream)` | allocator 在昇腾侧不适用 |
| Destroy | `~StaticMultimap()` | `~static_multimap()` | RAII 释放 Device 内存 |
| Clear | `void Clear(aclrtStream stream)` / `ClearAsync` | `clear(stream)` / `clear_async` | |
| Insert | `SizeType Insert(void* values, Extent valueNum, aclrtStream stream)` / `InsertAsync` | `insert(first, last, stream)` | values 为 `Pair<Key, T>` 数组；返回插入失败数（ops-collections 约定） |
| InsertIf | `template <StencilT, Predicate> SizeType InsertIf(void* values, StencilT* stencil, Extent valueNum, aclrtStream stream)` / `InsertIfAsync` | `insert_if(first, last, stencil, pred, stream)` | 谓词为模板参数，在设备端构造；返回满足条件但失败的个数 |
| Contains | `void Contains(void* keys, void* output, Extent keyNum, aclrtStream stream)` / `ContainsAsync` | `contains(first, last, output_begin, stream)` | output 为 bool 数组 |
| ContainsIf | `template <StencilT, Predicate> void ContainsIf(void* keys, StencilT* stencil, void* output, Extent keyNum, aclrtStream stream)` / `ContainsIfAsync` | `contains_if(first, last, stencil, pred, output_begin, stream)` | 不满足条件的位置输出 false |
| Find | `void Find(void* keys, void* output, Extent keyNum, aclrtStream stream)` / `FindAsync` | `find(first, last, output_begin, stream)` | 命中输出探测序列上第一个匹配的 value，未命中输出空 value 哨兵 |
| FindIf | `template <StencilT, Predicate> void FindIf(void* keys, StencilT* stencil, void* output, Extent keyNum, aclrtStream stream)` / `FindIfAsync` | `find_if(first, last, stencil, pred, output_begin, stream)` | 不满足条件的位置输出空 value 哨兵 |
| Count | `SizeType Count(void* keys, Extent keyNum, aclrtStream stream)` | `count(first, last, stream)` | 重复查询分别计数 |
| Retrieve | `SizeType Retrieve(void* keys, Extent keyNum, void* probeOutput, void* matchOutput, Extent outputCapacity, aclrtStream stream)` | `retrieve(first, last, output_probe, output_match, stream)` | probeOutput 为 key，matchOutput 为 value；返回匹配数；增加输出容量参数用于越界检查 |
| RetrieveAll | `SizeType RetrieveAll(void* keysOutput, void* valuesOutput, Extent outputCapacity, aclrtStream stream) const` | `retrieve_all(keys_out, values_out, stream)` | 返回输出个数 |
| 其他 | `Size(stream)`、`Capacity()`、`Data()`、`EmptyKeySentinel()`、`EmptyValueSentinel()`、`KeyEq()`、`HashFunction()` | `size`、`capacity`、`data`、`empty_key_sentinel`、`empty_value_sentinel`、`key_eq`、`hash_function` | |

通用参数约束：

| 参数 | 输入/输出/属性 | 数据类型 | dtype | 值域 | 异常行为 |
| --- | --- | --- | --- | --- | --- |
| container | 输入/输出 | 句柄 | - | 非空 | 构造失败抛异常，不产生半初始化对象 |
| keys | 输入 | Device 指针 | I32、I64 | 不得等于空 key 哨兵 | keyNum > 0 且为空指针时抛 `std::invalid_argument` |
| values | 输入/输出 | Device 指针 | `Pair<Key, T>`（Insert）；T（Find/Retrieve/RetrieveAll） | T 为 I32、I64，任意值（可等于空 value 哨兵） | Insert/InsertIf 输入为空指针时返回 valueNum；输出为空指针时抛 `std::invalid_argument` |
| keyNum / valueNum | 属性 | uint64 | - | ≥ 0 | 0 时不下发任务、允许空指针 |
| capacity | 属性 | uint64 | - | [1, 2^31]（上取整后） | 0 或超限抛 `std::invalid_argument`；Device 内存不足抛 `std::bad_alloc` |
| output | 输出 | Device 指针 | bool / T | - | keyNum > 0 且为空指针时抛 `std::invalid_argument` |
| outputCapacity | 属性 | uint64 | - | ≥ 匹配数 | 不足时抛 `std::length_error`（不会越界写，输出内容未定义） |
| stencil | 输入 | Device 指针 | uint32 | 合法 uint32 | 为空指针时：查询类抛 `std::invalid_argument`，InsertIf 返回 valueNum |
| pred | 属性 | 仿函数类型 | - | 可在设备端默认构造并以 `COLLECTION_SIMT_DEVICE` 调用 | 类型不合法时编译失败 |
| stream | 输入 | aclrtStream | - | 有效 ACL 流 | 同步失败抛 `std::runtime_error`（含 ACL 错误码） |

### 2.3.2 与参考实现相比的约束

1. 不提供任务书未要求的 `erase`、`rehash`、`for_each`、`insert_and_find`、`count_outer`、`retrieve_outer`，以及带 `ProbeEqual` / `ProbeHash` 的重载。
2. key 与 value 仅支持 4 / 8 字节类型（编译期 `static_assert`），覆盖任务要求的 I32/I32、I64/I64。
3. 设备端按默认构造的比较器与探测策略执行（与现有 `StaticMap` 一致），自定义哈希种子只影响 host 侧 `HashFunction()` 的返回。
4. `Insert` / `InsertIf` 返回失败数（ops-collections 约定），而非 cuCollections 的 `void` / 成功数。
5. `Retrieve` / `RetrieveAll` 的输出顺序与 cuCollections 不同（两者都不保证特定顺序），但在相同输入和容器状态下确定、可复现；同一 key 有多个 value 时 `Find` 返回其中任意一个（探测序列上的第一个），与 cuCollections 语义一致。
6. 默认探测策略为线性探测（cuCollections 为双重哈希），双重哈希可通过模板参数选用；两者都满足多值容器的查询语义。
7. `RetrieveAll` 在 host 记录的元素个数已知时使用一块 Device 工作空间（约为输出规模的 6.4%），首次使用时申请并复用。

# 三、需求详细设计（required）

## 3.1 使能方式

纯头文件 C++ 模板库。使用者包含 `static_multimap.h`，在 host 侧调用容器接口；容器内部通过 Ascend C 核函数调用符 `kernel<<<aivCoreNum, 0, stream>>>(...)` 下发 SIMD/SIMT 任务。每个批量接口提供同步版本（返回前 `aclrtSynchronizeStream`）和 `Async` 版本（只下发）；需要回读结果的接口（`Insert`、`InsertIf`、`Count`、`Retrieve`、`RetrieveAll`、`Size`）只有同步版本（`InsertAsync` / `InsertIfAsync` 不统计失败数）。

## 3.2 需求总体设计

### 3.2.1 host 侧设计

#### 3.2.1.1 分核策略

启动全部 AIV 核（`PlatformAscendCManager::GetCoreNumAiv()`，记为 C）。SIMT 核函数在每个核上通过 `VF_CALL` 启动若干线程（均为 Warp 宽度 32 的整数倍），线程全局编号 `rank = blockIdx × 每核线程数 + threadIdx`，线程总数记为 T；同一 Warp 的 32 个线程对应连续的 32 个 rank。

| 核函数 | 每核线程数 | 说明 |
| --- | --- | --- |
| 查询、计数、检索、整表扫描 | 1024（T = C × 1024） | 以随机读为主，实测 1024 最快 |
| Insert（线性探测、8 字节槽即 I32/I32、无 stencil） | 1792（T = C × 1792） | 以随机写为主，每核线程越多越快（1536 起基本持平）；每核线程数增大后每个线程可用的寄存器减少，该核函数只保留最少的线程内状态，且不对局部变量取地址 |
| Insert（I64/I64 的索引布局，无 stencil） | 1792 | 每个键值对顺序写入稠密数组，再对 8 字节索引槽做一次 CAS，同样以随机写为主 |
| InsertIf（带 stencil）、双重哈希的槽布局 Insert；索引布局的带条件插入（先计数、再按序号写入） | 1024 | 通用核函数，线程内状态较多 |
| 搬移任务（Retrieve / RetrieveAll 收尾） | 搬移量 ≤ C × 32 × 1024 时为 32，否则为 1024 | 只有少数线程有任务时，空转线程的启动开销占主导 |
| Create / Clear（向量填充） | 每核 1 个 SIMD 任务 | 每核处理 `coreWords = ⌈W / C⌉`（向上对齐到 32）个字，W 为表的总字数 |

数据划分：所有 SIMT 核函数（包括整表扫描）都按**步长交错**划分，`rank` 号线程处理下标 `rank, rank + T, rank + 2T, …`（逐输入元素的操作为输入下标：Insert、Contains、Find、Count、Retrieve；整表扫描为槽下标：Size、RetrieveAll）。同一 Warp 的 32 个线程在每一步访问相邻的 32 个元素，输入读取、输出写出和槽读取都集中在少数缓存行内；每个线程的负载与数据分布无关。实测整表扫描用步长交错比“每线程一段连续槽”快一倍以上。

#### 3.2.1.2 数据分块和内存优化策略

1. **批次切分**：单次核函数调用最多处理 `2^30` 个元素，超过时 host 按批下发（线程内下标为 uint32，`i + T` 不溢出）。
2. **表存储**：按键值对宽度选择两种布局（模板参数在编译期决定），尾部都多申请 64 个槽容纳向量填充按 32 对齐产生的越界写，Device 内存以 `ACL_MEM_MALLOC_HUGE_FIRST` 申请。容量合法化沿用 `MakeValidExtent`：线性探测为 `⌈capacity / 2⌉ × 2`，双重哈希为「素数 × 2」。
   - **槽布局**（`sizeof(Pair<Key, T>) ≤ 8`，即 I32/I32）：`(capacity + 64) × sizeof(Pair<Key, T>)` 字节，键值对直接存放在槽中。空槽哨兵在 host 侧先整槽清零再写入 key、value，保证填充字节确定，整槽 CAS 与整槽比较的期望值即为该字节序列。
   - **索引布局**（更宽的键值对，即 I64/I64）：索引表 `(capacity + 64) × 8` 字节，每个槽为「key 的 32 位哈希值 << 32 | 键值对下标」，全 1 表示空槽；键值对数组 `capacity × sizeof(Pair<Key, T>)` 字节，按插入顺序追加，前 N 个有效（N 为 host 精确记录的元素个数）。16 字节槽的插入需要两次随机写（占用 key 的 CAS 与写 value），实测约为 8 字节槽的两倍；索引布局把它变成一次顺序写加一次 8 字节 CAS。
3. **向量填充的 UB 规划**（`FillWords` 核函数，单 buffer）：
   - `tileWords = ⌊ubSize / 4 / sizeof(Word) / 32⌋ × 32`（至少 32），`ubSize` 由 `GetCoreMemSize(UB)` 获取；
   - UB 占用 = `tileWords × sizeof(Word)` ≈ ubSize / 4；
   - 每核只 `Duplicate` 一次 tile，再按 tile 把同一块本地数据 `DataCopy` 到 GM 的各段，搬运次数 = `⌈coreWords / tileWords⌉`。
   - Word 的选择：槽字节是 4 字节字的重复（I32/I32 且两个哨兵相等）时用 uint32；是 8 字节字的重复（I32/I32 任意哨兵，或 I64/I64 且两个哨兵相等）时用 uint64；否则（I64/I64 且哨兵不同）或表字数少于 `C × 32` 时改用 SIMT 逐槽清空（`aclco::Clear`）。
4. **SIMT 线程不使用 UB 静态/动态内存**（`dyn_ub_size = 0`），GM 访存走 SIMT Data Cache。
5. **每线程计数数组**：构造时申请 Device `(C × 1792 + 1) × 8` 字节和同样大小的 host 数组。核函数把线程私有计数写到 `threadData[rank]`，0 号线程把实际线程总数写到末元素；host 先读回末元素校验线程数，再只读回实际线程数个计数。另有 2 个 uint64 的 Device 状态标志（插入失败、Warp 压实自检失败），设备端只会把它们置 1。`Count` 以查询核函数的线程总数 `C × 1024` 作为槽数，哨兵写在 `threadData[C × 1024]`，计数与哨兵一次读回；逐 key 输出的核函数另由前 64 个线程把输出分布写到 `threadData[0, 128)`（见 3.2.2.1「逐 key 输出预填」）。
6. **RetrieveAll 单遍方案的溢出区**：`W × S` 个 key 与同样个数的 value，W = T / 32 为 Warp 数，`S = ⌈(12 × √(N / W) + 256) / 512⌉ × 512`，N 为 host 记录的元素个数（Warp 的实际元素数约服从均值 N / W 的二项分布，线性探测的聚集会略微放大波动，取约 12 倍标准差）。N = 1e8、C = 56 时 W = 1792、S = 3584，约 640 万个元素，约为输出规模的 6.4%。首次使用时申请，之后按需扩容复用。
7. **搬移任务**：每个任务 4 个 uint64 `(src, dst, len, fromSpill)`，host 构建后一次拷贝到 Device，数组按需扩容复用。
8. **逐 key 输出预填**（Contains、Find）：批大小 ≥ 2^24 且已选定预填值时，先用 `FillWords` 把调用方输出缓冲的前 `⌊len / 32⌋ × 32` 个元素写成预填值（Contains 的 1 字节输出按 4 字节字填充，个数按 128 对齐；均向下取整，不会写出输出区间），核函数只写与预填值不同的结果。预填直接写在调用方的输出缓冲上，不申请额外 Device 内存；抽样统计每次只读回 128 个 uint64。
9. **内存总量**：槽布局为表 `capacity × sizeof(Pair<Key, T>)`，索引布局为 `capacity × (8 + sizeof(Pair<Key, T>))`（I64/I64 每槽 24 字节）；另有 64 槽填充、空哨兵槽、状态标志、每线程计数数组，以及（槽布局使用 RetrieveAll 单遍方案时的）溢出区与搬移任务。Retrieve / RetrieveAll 直接写入调用方的输出缓冲，不产生与输入规模线性相关的额外 Device 拷贝，满足任务书 3.4 内存要求。

#### 3.2.1.3 tilingKey 规划策略

不使用 tilingKey。所有分支在编译期或下发前确定，直接选择不同的核函数实例：

| 分支 | 决定方式 |
| --- | --- |
| key / value 宽度（8 字节槽 / 16 字节槽） | 模板参数；8 字节槽整槽 CAS，16 字节槽 CAS key 后写 value |
| 存储布局 | 模板参数：`sizeof(Pair<Key, T>) > 8` 时为索引布局，否则为槽布局；查询类核函数通过模板参数 `Indexed` 选择设备端引用 |
| 插入核函数 | 槽布局：线性探测且无 stencil 时用 `InsertLinear`，否则用 `InsertIf`；索引布局：无 stencil 时用 `IndexedInsert`，否则 `StencilCount` + `IndexedInsertIf` |
| 是否读取 stencil（`Insert` vs `InsertIf` 等） | 模板参数 `Predicate` 是否为内部占位谓词 `NoStencil`，编译期消除 stencil 访存 |
| 是否统计插入失败数（同步 / 异步） | 模板参数 `CountFailures` |
| Retrieve 方案 | 剩余输出容量 ≥ 本批输入数时走单遍（直接写 + 搬移），否则走两遍；单遍中某线程的匹配数超过其输入数时，按已得到的计数改走两遍 |
| RetrieveAll 方案 | 索引布局：`SplitEntries` 按插入顺序直接输出；槽布局：host 记录的元素个数 N 已知且 `W × 256 ≤ N ≤ 输出容量` 时走单遍，否则走两遍；某个 Warp 的溢出区放不下时改走两遍；Warp 压实自检失败后改用逐线程写出 |
| 搬移启动宽度 | 按搬移量选择每核 32 或 1024 个线程 |
| 清空路径（uint32 填充 / uint64 填充 / SIMT 逐槽） | host 按哨兵字节模式与表大小选择核函数 |
| 逐 key 输出预填（Contains、Find） | 批大小 ≥ 2^24 时由上一次同步调用的抽样统计决定预填值或不预填；Find 是否支持预填由 value 类型决定（模板参数） |

### 3.2.2 kernel 侧设计

#### 3.2.2.1 kernel 侧实现描述

设备端核心为 `MultiValueRefImpl`（`detail/open_addressing/multi_value_ref_impl.h`），`StaticMultimapRef` 对其做薄封装；核函数位于 `detail/open_addressing/multi_value_kernels.h`。各操作与参考实现的对应关系：

| 操作 | 昇腾实现 | 对应参考实现 |
| --- | --- | --- |
| 探测 | 默认线性探测：起点 `FastRange(murmurhash3_32(key), 桶数) × 2`，其中 `FastRange(h, n) = (h × n) >> 32`，逐 bucket 前进，表尾用一次条件减法回绕，热路径不含除法；也支持 `DoubleHashing`：步长 `(FastRange(h2, 桶数 − 1) + 1) × 2`，桶数为素数，保证遍历全表。bucket 大小 2 | `double_hashing` + `storage<2>` |
| Insert | 精简核函数 `InsertLinear`（线性探测、8 字节槽、无 stencil）：读整槽，等于空哨兵槽时整槽一次 `AtomicCas`（期望值即空哨兵槽）；CAS 失败或槽非空则前进一个槽，表尾回绕，访问完全部槽仍未占到空槽即失败。通用核函数 `InsertIf`：逐 bucket 只读 key 字段，是空槽则占用——8 字节槽整槽 `AtomicCas`，16 字节槽先对 key 做 `AtomicCas`，成功后该槽由本线程独占，再普通写 value。两者都不做 key 相等比较。索引布局 `IndexedInsert`：host 先确定本批可插入数 = min(输入数, 剩余空槽数)，第 i 个键值对写入 `entries[N + i]`，再对 8 字节索引槽做一次 `AtomicCas`（期望值为全 1），其余计为失败；带 stencil 时先由 `StencilCount` 统计各线程满足条件的个数，host 前缀和后 `IndexedInsertIf` 按序号写入 | `insert_if_n` + `AllowsDuplicates` |
| 插入失败统计 | 线程私有寄存器累加，线程结束时写 `threadData[rank]`；有失败的线程另把失败标志置 1。host 同步后只读回 8 字节的标志，置位时才读回各线程计数求和并清除标志 | 原子计数器 |
| Contains / Find | 遇到相等 key 返回 true / 该槽 value，遇到空槽返回 false / 空 value 哨兵；`*If` 版本先判 `pred(stencil[i])`；输出已预填时跳过结果等于预填值的写出（见下文「逐 key 输出预填」） | `contains_if_n` / `find_if_n` |
| Count | `RetrieveCount` 逐 key 统计匹配槽直到第一个空槽，线程内累加后写 `threadData[rank]`，线程总数哨兵紧跟在计数之后，host 一次读回后求和（不使用原子操作） | `count` 核函数 |
| Retrieve | 单遍（剩余输出容量 ≥ 输入数）：`RetrieveDirect` 中 rank 号线程把第 k 个匹配直接写到下标 `rank + k × T`（与输入相同的步长交错布局），线程匹配数写 `threadData[rank]`；host 求总数 M，若某线程匹配数超过其输入数则按已得计数走两遍，否则把下标 ≥ M 的匹配搬进下标 < M 的空位（任务按步长 T 组织）。两遍：`RetrieveCount` → host 前缀和 → `RetrieveFill` 每线程从起点顺序写出 (key, value) | `retrieve`（单遍 + 共享缓冲 + 原子预留） |
| 查询（索引布局） | `IndexedRefImpl` 与槽布局的探测顺序相同；非空索引槽先比较 32 位指纹（即 key 的哈希值），相同时再读 `entries[下标]` 比较 key，命中后 value 取自同一元素；Contains / Find / Count / Retrieve 的终止条件与槽布局相同 | 同上 |
| RetrieveAll | 索引布局：`SplitEntries` 把 `entries[0, N)` 拆成 key、value 两个输出（步长交错）。槽布局：Warp 压实扫描 `ScanFill`：同一 Warp 的 32 个线程每步读取相邻的 32 个槽，`WarpBallotSync(非空)` 返回本步非空线程的掩码 m，线程 i 的输出下标 = 本 Warp 已输出数 + `popcount(m & (2^i − 1))`，随后已输出数加 `popcount(m)`，同一 Warp 的输出是连续的一段。单遍（N 已知）：把 `[0, N)` 均分给各 Warp，超出本 Warp 区间的元素写入该 Warp 的溢出区，Warp 首线程把实际元素数写回；host 生成搬移任务，把越界元素和溢出元素搬进空洞。两遍：`CountFilled` 统计各线程非空槽 → host 前缀和 → `ScanFill` 以 Warp 首线程的起点写出 key 与 value | zip 迭代器 + `cub::DeviceSelect::If` |
| Size | `CountFilled` 后 host 求和，同时更新 host 记录的元素个数 | `cub::DeviceReduce::TransformReduce` |
| Create / Clear | `FillWords`：每核 `Duplicate` 一个 tile 后循环 `DataCopy` 到 GM；其余情况走 SIMT `Clear` | `thrust::for_each_n` |

**逐 key 输出预填**：950 上核函数在随机访存之间分散写出输出的代价明显高于一次向量化顺序填充（1e8 个 8 字节输出约 2 ms 对 0.7 ms），而逐 key 输出的结果通常集中在少数取值上（是否命中、未命中的空哨兵、计数 0 或 1）。因此批大小 ≥ 2^24 时，host 先用 `FillWords` 把输出的前 `⌊len / 32⌋ × 32` 个元素预填为选定值，核函数对预填区内结果等于该值的元素跳过写出，其余元素照常写出，输出与不预填时逐位相同。预填值由上一次同步调用决定：Contains 的候选为 false 与 true；Find 的候选为空 value 哨兵（value 为 4 / 8 字节整数时启用，由模板参数决定）；初始均不预填。线程号小于 64 的线程统计本线程结果中等于各候选值的个数（Contains 只统计未命中数，命中数取其余部分），写到 `threadData[2 × rank]`、`threadData[2 × rank + 1]`（步长交错划分下，这些线程负责的元素均匀分布在整个输入上）；同步接口在同步 stream 后读回这 128 个值，两个候选值中较多的一个占抽样元素的三分之一以上时作为下一次调用的预填值（一次顺序预填的代价约相当于省下三分之一的分散写出），否则下一次不预填。异步接口使用当前的选择，但不读回统计（此时核函数可能尚未执行完）。选择只影响性能，不影响结果。

**重复 key 正确性依据**：两次 `Clear` 之间槽只会由空变满（空槽即 key 字段等于空 key 哨兵的槽，value 允许等于空 value 哨兵；空槽的整槽内容恒为空哨兵槽，因此按整槽比较判空与按 key 判空等价）。插入总是占用探测序列上第一个「CAS 时仍为空」的槽，其前面的槽在当时已满且保持为满，所以同一 key 的全部键值对都位于其探测序列的第一个空槽之前；同一 bucket 内被占用的槽也构成前缀。因此查询遇到空槽即可停止，`Count` / `Retrieve` 不会漏计。精简插入核函数逐槽前进、通用核函数逐 bucket 前进，由于起点按 bucket 对齐、容量是 bucket 大小的整数倍，两者访问的槽序列相同。

**索引布局的正确性**：索引槽同样只会由空变满，同一 key 的全部键值对都登记在其探测序列的第一个空槽之前，查询遇到空槽即可停止；指纹只用于跳过不同 key，命中必须通过 key 比较确认，因此结果与槽布局完全一致。每个键值对的下标由 host 唯一分配（N + i 或 N + 序号），且每批只插入不超过剩余空槽数的键值对，所以设备端的每次登记都能成功（若出现意外失败，同步插入会抛出 `std::logic_error`）；entries 中不存在无效元素，元素个数 N 在 host 侧始终精确，`Size` 直接返回 N，`RetrieveAll` 直接输出 `entries[0, N)`。

**满表语义**：若某次插入失败，说明它检查过的所有槽在检查时均已满，而槽不会变空，故失败发生时整表已满；一批插入的失败数恰为 `max(0, 待插入数 − 剩余空槽数)`，满足容量边界用例。

**16 字节槽的可见性**：key CAS 与 value 写入之间槽对并发读者短暂可见为 (key, 空 value)。同步接口保证插入完成后才执行查询；与 cuCollections 相同，不支持对同一容器并发插入与查询。

**Warp 压实的正确性**：同一 Warp 的线程在同一步调用 `WarpBallotSync`，得到相同的掩码；非空线程的下标两两不同且连续，因此 Warp 的输出恰好填满 `[起点, 起点 + Warp 元素数)`。线程与掩码位的对应关系（第 `rank mod 32` 位）由核函数逐步自检（本线程非空当且仅当掩码对应位为 1），Warp 宽度（`GetWarpSize() == 32`）与线程总数在核函数入口校验；任一不符即置自检失败标志，host 改用逐线程写出（每线程从 `threadData[rank]` 起顺序写），此后该容器不再使用 Warp 压实。

**单遍 RetrieveAll 的收尾**：记 L = ⌊N / W⌋，Warp w 的区间为 `[w × L, w × L + L_w)`（最后一个 Warp 取余量），实际元素数为 c_w，总数 M = Σc_w。空洞为各区间中未写满且位于 M 之前的部分（M > N 时另加 `[N, M)`）；待搬元素为位于 M 及之后的区间内元素，以及各溢出区中的元素。二者个数相等，host 按顺序配对生成任务，每个任务的长度不超过「搬移量 / 线程数」（至少 256），使线程负载均衡。N 只影响方案选择与区间划分，结果只取决于扫描得到的实际计数：N 偏大时搬移量增大，N 偏小导致溢出区放不下时改走两遍。M 大于输出容量时抛 `std::length_error`；由于区间都位于 `[0, N) ⊆ [0, 输出容量)`，不会越界写。

**host 记录的元素个数**：构造与 Clear 后为 0；同步 `Insert` 累加成功数（`valueNum − 失败数`）；`InsertAsync`、带 stencil 的 `InsertIf` / `InsertIfAsync` 之后记为未知；`Size` 与 `RetrieveAll` 用扫描结果更新。通过 `Data()` 在设备端直接改表会使记录失准，但只影响方案选择，不影响结果。

**确定性**：输出位置只由输入、容器状态和线程总数决定（单遍与两遍方案的输出顺序可能不同，但相同输入与容器状态下每次的方案与输出都相同）。

#### 3.2.2.2 AscendC 实现流程图

**Create / Clear**

```mermaid
flowchart TD
    C1{"capacity == 0 或上取整后 > 2^31?"} -- 是 --> C0["抛 std::invalid_argument"]
    C1 -- 否 --> C2["aclrtMalloc：(capacity + 64) 个槽、空哨兵槽、状态标志、每线程计数数组"]
    C2 --> C3{"哨兵槽为重复字 且 字数 ≥ C×32?"}
    C3 -- 是 --> C4["FillWords（uint32 或 uint64 字）：每核 Duplicate 一个 tile，循环 DataCopy 到 GM"]
    C3 -- 否 --> C5["SIMT Clear：逐槽写空哨兵槽"]
    C4 --> C6["aclrtSynchronizeStream；host 记录元素个数 = 0"]
    C5 --> C6
```

**Insert / InsertIf**

```mermaid
flowchart TD
    I0{"valueNum == 0?"} -- 是 --> I00["返回 0"]
    I0 -- 否 --> I01{"values/stencil 为空?"}
    I01 -- 是 --> I02["返回 valueNum"]
    I01 -- 否 --> I1{"线性探测、8 字节槽且无 stencil?"}
    I1 -- 是 --> I1a["InsertLinear：每核 1792 线程，步长交错，按 ≤2^30 分批"]
    I1 -- 否 --> I1b["InsertIf：每核 1024 线程，步长交错，按 ≤2^30 分批"]
    I1b --> I2{"pred(stencil[i])?（无 stencil 时编译期省略）"}
    I2 -- 否 --> I9["跳过"]
    I2 -- 是 --> I3["计算探测起点，沿探测序列逐槽读取"]
    I1a --> I3
    I3 --> I4{"空槽?"}
    I4 -- 否 --> I5{"已访问全部槽?"}
    I5 -- 否 --> I3
    I5 -- 是 --> I6["本线程失败数 +1"]
    I4 -- 是 --> I7{"8 字节槽：整槽 AtomicCas<br/>16 字节槽：key AtomicCas 成功?"}
    I7 -- 否 --> I3
    I7 -- 是 --> I8["插入成功（16 字节槽随后写 value）"]
    I6 --> I10["线程结束：失败数写 threadData[rank]；有失败则置失败标志"]
    I8 --> I10
    I10 --> I11{"同步后读回失败标志：置位?"}
    I11 -- 否 --> I12["失败数 = 0"]
    I11 -- 是 --> I13["读回各线程失败数求和，清除标志"]
    I12 --> I14["记录元素个数 += 成功数（带 stencil 时记为未知）；返回失败数"]
    I13 --> I14
```

**Contains / Find / Count**

```mermaid
flowchart TD
    Q0["参数校验；按 ≤2^30 分批"] --> P1{"Contains / Find：批大小 ≥ 2^24 且已选定预填值?"}
    P1 -- 是 --> P2["FillWords：输出前 ⌊len/32⌋×32 个元素写成预填值"]
    P1 -- 否 --> Q1
    P2 --> Q1["下发核函数（每核 1024 线程）"]
    Q1 --> Q2["每线程按步长取 key，计算探测起点"]
    Q2 --> Q3{"槽 key"}
    Q3 -- 相等 --> Q4["Contains/Find：得到 true / 槽 value<br/>Count：计数 +1 继续"]
    Q3 -- 空槽 --> Q5["Contains/Find：得到 false / 空 value 哨兵<br/>Count：得到该 key 的计数"]
    Q3 -- 其他 --> Q2b["下一个槽 / 下一个 bucket"]
    Q2b --> Q3
    Q4 -- 计数类继续 --> Q2b
    Q4 -- Contains/Find --> Q6
    Q5 --> Q6{"结果等于预填值且下标在预填区内?"}
    Q6 -- 是 --> Q7["不写"]
    Q6 -- 否 --> Q8["写 output[i]（Count：线程内累加）"]
    Q7 --> Q9["线程结束：前 64 个线程写结果分布；Count 写 threadData[rank]"]
    Q8 --> Q9
    Q9 --> Q10["同步接口：同步后读回分布，较多的候选值占比超过 1/3 则作为下一次的预填值；Count 一次读回计数与哨兵求和"]
```

**Retrieve**

```mermaid
flowchart TD
    R0{"剩余输出容量 ≥ 本批输入数?"} -- 是 --> R1["RetrieveDirect：rank 号线程把第 k 个匹配写到 rank + k·T，匹配数写 threadData[rank]"]
    R0 -- 否 --> R5["RetrieveCount：各线程匹配数写 threadData[rank]"]
    R1 --> R2["同步；host 校验线程数，读回各线程匹配数，求总数 M"]
    R2 --> R3{"M > 输出容量?"}
    R3 -- 是 --> RX["抛 std::length_error"]
    R3 -- 否 --> R4{"某线程匹配数 > 其输入数?"}
    R4 -- 是 --> R8["按已得计数求前缀和"]
    R4 -- 否 --> R6["host 生成搬移任务：下标 ≥ M 的匹配 → 下标 < M 的空位（步长 T）"]
    R6 --> R7["MoveItems（按搬移量窄 / 宽启动）→ 同步，返回 M"]
    R5 --> R5b["同步；读回计数，求前缀和与总数 M"]
    R5b --> R5c{"M > 输出容量?"}
    R5c -- 是 --> RX
    R5c -- 否 --> R8
    R8 --> R9["上传起点；RetrieveFill：每线程从起点顺序写 (key, value) → 同步，返回 M"]
```

**RetrieveAll / Size**

```mermaid
flowchart TD
    A0{"记录的元素个数 N 已知，且 W×256 ≤ N ≤ 输出容量?"} -- 是 --> A1["ScanFill 单遍：[0, N) 均分给 W 个 Warp；Warp 内压实写出，超出区间的写入溢出区"]
    A0 -- 否 --> B1["CountFilled：各线程非空槽数 → host 前缀和，总数 M，记录 N = M"]
    A1 --> A2["同步；读回自检标志与各 Warp 元素数，M = 总和，记录 N = M"]
    A2 --> A3{"自检失败?"}
    A3 -- 是 --> B1
    A3 -- 否 --> A4{"M > 输出容量?"}
    A4 -- 是 --> AX["抛 std::length_error"]
    A4 -- 否 --> A5{"某 Warp 的溢出区放不下?"}
    A5 -- 是 --> B1
    A5 -- 否 --> A6["host 配对空洞与待搬元素 → MoveItems → 同步，返回 M"]
    B1 --> B2{"M > 输出容量?"}
    B2 -- 是 --> AX
    B2 -- 否 --> B3{"Warp 压实可用?"}
    B3 -- 是 --> B4["ScanFill 两遍：各 Warp 从首线程的起点压实写出 → 同步"]
    B4 --> B5{"自检失败?"}
    B5 -- 否 --> B7["返回 M"]
    B5 -- 是 --> B6["RetrieveAllFill：各线程从 threadData[rank] 起顺序写出 → 同步"]
    B3 -- 否 --> B6
    B6 --> B7
    S1["Size：CountFilled → 同步 → host 求和，记录 N"]
```

**Warp 压实扫描的一步（ScanFill）**

```mermaid
flowchart LR
    W1["第 j 步：线程 i 读槽 rank_i + j·T（同一 Warp 读相邻 32 个槽）"] --> W2["filled_i = 槽 key ≠ 空 key"]
    W2 --> W3["m = WarpBallotSync(filled)，Warp 内各线程相同"]
    W3 --> W4{"filled_i?"}
    W4 -- 是 --> W5["下标 = written + popcount(m 的低 i 位)；在区间内则写 key / value，否则写溢出区"]
    W4 -- 否 --> W6["不写"]
    W5 --> W7["written += popcount(m)；自检 m 的第 i 位 == filled_i"]
    W6 --> W7
```

**索引布局（I64/I64）的插入与 RetrieveAll**

```mermaid
flowchart TD
    X0["Insert(values, num)"] --> X1["accepted = min(num, capacity − N)，failed = num − accepted"]
    X1 --> X2["IndexedInsert：每核 1792 线程，第 i 个键值对写 entries[N + i]"]
    X2 --> X3["h = hash(key)，起点 FastRange(h) × 2，逐槽前进"]
    X3 --> X4{"索引槽为全 1 且 CAS(全 1 → h << 32 | N + i) 成功?"}
    X4 -- 否 --> X3
    X4 -- 是 --> X5["N += accepted；同步后检查失败标志，返回 failed"]
    Y0["InsertIf(values, stencil, num)"] --> Y1["StencilCount：各线程满足条件的个数 → host 前缀和 P"]
    Y1 --> Y2["accepted = min(P, capacity − N)；IndexedInsertIf：序号 < accepted 的写 entries[N + 序号] 并登记"]
    Y2 --> X5
    Z0["RetrieveAll(keys, values, cap)"] --> Z1{"N > cap?"}
    Z1 -- 是 --> Z2["抛 std::length_error"]
    Z1 -- 否 --> Z3["SplitEntries：keys[i] = entries[i].first，values[i] = entries[i].second → 同步，返回 N"]
    Q0["查询：非空索引槽"] --> Q1{"指纹 == hash(key)?"}
    Q1 -- 否 --> Q3["下一个槽"]
    Q1 -- 是 --> Q2{"entries[下标].first == key?"}
    Q2 -- 是 --> Q4["命中：value = entries[下标].second"]
    Q2 -- 否 --> Q3
```

#### 3.2.2.3 AscendC 流程图与参考实现流程图的差异点和原因

| 差异点 | cuCollections | 本设计 | 原因 |
| --- | --- | --- | --- |
| 默认探测策略 | `double_hashing<8, xxhash_32>` | `LinearProbing<murmurhash3_32>`（bucket 大小 2），也支持 `DoubleHashing` | 950 上时延主要取决于随机访存次数：线性探测的后续探测大多落在同一缓存行内，双重哈希每一步都是新的随机访存，实测插入与查询都是线性探测更快；murmurhash3_32 与 ops-collections `StaticMap` 的默认值一致 |
| 探测起点与回绕 | 取模 | `FastRange` 乘法映射 + 条件减法 | 热路径不含除法 |
| 协作组 | 协作组探测 | 逐线程独立探测（bucket 大小 2 相同） | SIMT 以 32 线程 Warp 为执行单位，逐线程探测实现简单 |
| 插入 | 按槽宽选择整槽 CAS 或 key CAS 后写 value | 8 字节槽整槽一次 `AtomicCas`（精简核函数，每核 1792 线程）；16 字节槽 key `AtomicCas` 后普通写 value | 每个元素只做一次原子操作；随机写为主的核函数在更多线程下吞吐更高，精简核函数控制线程内状态以适应更小的寄存器预算 |
| 宽键值对存储 | 16 字节槽，key CAS 后写 value | 8 字节索引槽（哈希指纹 + 下标）+ 按插入顺序存放的键值对数组 | 每次插入只需一次顺序写和一次 8 字节 CAS，避免两次随机写；RetrieveAll 无需扫描整表；代价是每槽多 8 字节内存，命中时多读一次键值对数组 |
| 计数器 | 线程块内 reduce 后原子加 | 线程私有计数写 `threadData[rank]`，host 求和；插入另设失败标志 | 核函数内的原子加开销明显；插入失败罕见，无失败时不读回计数数组 |
| Retrieve | 单遍：线程块共享缓冲 + `fetch_add` 预留输出区间 | 单遍：匹配直接写到步长交错位置，host 规划搬移收尾；容量不足或某线程匹配数超过输入数时两遍 | SIMT 没有线程块共享缓冲；直接写出的布局与输入布局一致，Warp 内写出相邻；输出确定、可复现 |
| RetrieveAll / Size | zip 迭代器 + CUB `DeviceSelect::If` / `TransformReduce` | Warp 压实扫描（`WarpBallotSync` + popcount）；元素个数已知时单遍 + 溢出区 + 搬移，否则计数后两遍；自检失败时逐线程写出 | 昇腾侧无 CUB；逐线程写出时一个 Warp 的 32 个线程每步写 32 个不同位置，压实后写出连续；单遍省去一次整表扫描 |
| Create / Clear | `thrust::for_each_n` 逐槽写 | SIMD `Duplicate` + 分 tile `DataCopy`，按哨兵字节模式选 uint32/uint64 字；不可按字填充或小表时回退 SIMT 逐槽写 | 大表按字批量搬运，吞吐更高；现有 `ClearSIMD` 在每核块数恰为 tile 整数倍时最后一个 tile 写 0 个元素，且用 key 哨兵同时填充 8 字节槽的 value 位置，新核函数按完整哨兵槽选择填充字并精确计算尾块 |
| 插入返回值 | `insert` 无返回、`insert_if` 返回成功数 | 均返回失败数 | 与 ops-collections `StaticMap` 约定和任务测试用例一致 |
| 输出容量 | 输出区间过小属未定义行为 | `Retrieve` / `RetrieveAll` 显式传入容量，不足时抛 `std::length_error`，不会越界写 | 满足任务书「输出空间不足时报错」 |
| 逐 key 输出 | 每个 key 写一次输出 | 批大小 ≥ 2^24 时按上一次同步调用的结果分布预填输出（向量化顺序写），核函数只写与预填值不同的结果 | 950 上核函数内分散写出的代价明显高于一次向量化顺序填充；是否命中、未命中的空哨兵、计数 0 或 1 通常占多数 |
| 批次大小 | `index_type` 64 位 | 单次核函数最多 2^30 个元素，host 自动分批 | SIMT 线程内使用 uint32 下标，寄存器受限 |

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Ascend 950DT（Atlas 950 系列，SIMT） | √ |

开发验证环境：Ascend950PR，CANN 9.1.0，bisheng 编译器 `--npu-arch=dav-3510`。

## 3.4 算子约束限制

1. key、value 必须为 4 或 8 字节类型；key 不得等于构造时指定的空 key 哨兵，value 无限制。
2. 容量上取整后不超过 2^31 个槽。
3. 同一时刻不应对同一容器并发执行插入与查询（与 cuCollections 相同，结果未定义）。
4. `pred` 必须可在设备端默认构造并以 `COLLECTION_SIMT_DEVICE` 修饰。
5. `Contains` 输出为 1 字节 bool；`Find` 与 `Retrieve` 的 matchOutput、`RetrieveAll` 的 valuesOutput 类型为 T；`Retrieve` 的 probeOutput、`RetrieveAll` 的 keysOutput 类型为 Key。
6. 依赖 SIMT 运行时的线程布局：同一 Warp 的线程为连续的 32 个 rank（`GetWarpSize() == 32`）；不满足时 RetrieveAll 自动改用逐线程写出，结果不变。
7. `Retrieve` / `RetrieveAll` 抛出 `std::length_error` 时输出缓冲的内容未定义（不会越界写）。
8. 索引布局（I64/I64）下：`InsertIfAsync` 需要先统计满足条件的个数，会同步 stream；`Data()` 返回按插入顺序存放的键值对数组；设备端引用 `StaticMultimapRef` 仅适用于槽布局。

# 四、特性交叉分析

| 维度 | 覆盖 |
| --- | --- |
| dtype | I32/I32、I64/I64 × 全部 12 个接口；8 字节槽与 16 字节槽两条插入路径 |
| 输入规模 | 0（允许空指针）、1、小于 C×32 的小表（SIMT 清空路径）、1e8 大规模、> 2^30（host 分批） |
| 重复度 | multiplicity 1 / 2 / 4 / 8，同一 key 多个 value，完全相同的键值对，重复查询 |
| 命中率 | matching rate 0 / 0.1 / 0.5 / 1，命中与未命中交错 |
| 容量 | 1、5、128、1027、8192、65536、100000、1e6、1e8；占用率 0 ~ 1.0；超容量插入 |
| 条件接口 | stencil 全 0、全 1、按下标奇偶 |
| 异常 | capacity = 0、空指针、输出容量不足、stream 同步失败 |
| 接口组合 | Insert→Clear→Insert、多次 Create/Destroy、InsertIf→Contains |
| 执行路径 | 插入：精简 / 通用核函数；Retrieve：单遍 / 两遍（输出容量不足、线程匹配数超过输入数）；RetrieveAll：单遍 / 两遍 / 逐线程写出，记录的元素个数偏大、偏小、翻倍，满表。仓内功能用例规模小于单遍阈值，另用 2e7 规模的自测程序覆盖 |

# 五、可维可测分析

## 5.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | I32/I32、I64/I64 的 Insert/InsertIf 状态变化，以及 Contains、ContainsIf、Find、FindIf、Count、Retrieve、RetrieveAll 的输出与 cuCollections 语义一致；测试以 `std::unordered_multimap` 为基准逐元素比对（Find 校验命中值属于该 key 的 value 集合，Retrieve / RetrieveAll 排序后比对）；相同输入重复执行结果一致 | 任务书 3.2 |
| 性能标准 | 每个用例时延 ≤ 标杆时延 / 0.4；不达标时给出原因分析 | 任务书 3.3 |
| 内存标准 | 除表存储、输出与必要工作空间外，无与输入规模线性相关的额外 Device 拷贝 | 任务书 3.4 |

功能用例（任务提供，按 dtype × GENERATE × SECTION 展开）：

| 接口 | 用例数 | 接口 | 用例数 |
| --- | --- | --- | --- |
| Create | 14 | Find | 100 |
| Destroy | 18 | FindIf | 56 |
| Clear | 32 | Count | 100 |
| Insert | 290 | Retrieve | 98 |
| InsertIf | 20 | RetrieveAll | 202 |
| Contains | 100 | ContainsIf | 56 |
| **合计** | | | **1086** |

性能用例（任务提供，共 32 个；时延为同步接口的 host 端到端时延）：

| 序号 | 接口 | Key/Value | 场景（NumInputs=1e8） | 标杆时延(ms) | 达标上限 = 标杆/0.4 (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | Create | I32/I32 | Capacity=1e8 | 7.635412 | 19.089 |
| 2 | Create | I64/I64 | Capacity=1e8 | 15.419803 | 38.550 |
| 3 | Destroy | I32/I32 | Capacity=1e8 | 9.038761 | 22.597 |
| 4 | Destroy | I64/I64 | Capacity=1e8 | 17.766071 | 44.415 |
| 5 | Insert | I32/I32 | UNIFORM, Occ=0.5, Mul=1 | 9.984867 | 24.962 |
| 6 | Insert | I64/I64 | UNIFORM, Occ=0.5, Mul=1 | 13.994128 | 34.985 |
| 7 | RetrieveAll | I32/I32 | UNIFORM, Occ=0.5, Mul=1 | 1.469109 | 3.673 |
| 8 | RetrieveAll | I64/I64 | UNIFORM, Occ=0.5, Mul=1 | 3.421753 | 8.554 |
| 9 | Contains | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=0.1 | 7.175591 | 17.939 |
| 10 | Contains | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=0.5 | 6.876158 | 17.190 |
| 11 | Contains | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=1 | 6.272652 | 15.682 |
| 12 | Contains | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=0.1 | 9.062888 | 22.657 |
| 13 | Contains | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=0.5 | 8.907699 | 22.269 |
| 14 | Contains | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=1 | 8.698283 | 21.746 |
| 15 | Find | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=0.1 | 7.842479 | 19.606 |
| 16 | Find | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=0.5 | 7.674542 | 19.186 |
| 17 | Find | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=1 | 7.065370 | 17.663 |
| 18 | Find | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=0.1 | 9.708089 | 24.270 |
| 19 | Find | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=0.5 | 9.566873 | 23.917 |
| 20 | Find | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=1 | 9.300692 | 23.252 |
| 21 | Count | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=0.1 | 5.868191 | 14.670 |
| 22 | Count | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=0.5 | 5.974523 | 14.936 |
| 23 | Count | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=1 | 6.097573 | 15.244 |
| 24 | Count | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=0.1 | 8.712889 | 21.782 |
| 25 | Count | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=0.5 | 8.838063 | 22.095 |
| 26 | Count | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=1 | 8.985895 | 22.465 |
| 27 | Retrieve | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=0.1 | 12.055540 | 30.139 |
| 28 | Retrieve | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=0.5 | 13.038496 | 32.596 |
| 29 | Retrieve | I32/I32 | UNIFORM, Occ=0.5, Mul=1, MR=1 | 13.475749 | 33.689 |
| 30 | Retrieve | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=0.1 | 14.419790 | 36.049 |
| 31 | Retrieve | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=0.5 | 15.764821 | 39.412 |
| 32 | Retrieve | I64/I64 | UNIFORM, Occ=0.5, Mul=1, MR=1 | 16.426197 | 41.065 |

性能优化设计：

1. **按访存模式选择探测与划分**：默认线性探测，后续探测大多落在同一缓存行；所有 SIMT 核函数按步长交错划分，同一 Warp 的输入读取、输出写出与槽读取相邻。
2. **单次原子操作插入**：每个元素只做一次 `AtomicCas`（8 字节槽整槽 CAS，16 字节槽只 CAS key）；线性探测的无条件插入使用精简核函数，每核 1792 个线程。
3. **无原子操作的计数**：失败数与计数写入线程私有位置，由 host 求和；插入只在失败标志置位时才读回计数数组。
4. **单遍检索**：Retrieve 把匹配直接写到与输入相同的步长交错位置，只搬移少量越界元素；RetrieveAll 借助 `WarpBallotSync` 在 Warp 内压实写出，元素个数已知时一次整表扫描即可完成。
5. **窄启动搬移**：少量搬移每核只启动 32 个线程，避免大量空转线程的启动开销。
6. **向量化填充**：Create / Clear 按 4 / 8 字节字批量填充，每核只生成一次 tile，再按 tile 搬运到 GM。
7. **宽键值对的索引布局**：I64/I64 的插入由两次随机写变为一次顺序写加一次 8 字节 CAS，RetrieveAll 变为顺序拆分输出。
8. **减少分支与访存**：`Insert` / `Contains` / `Find` 等无条件接口在编译期消除 stencil 访存；计数类接口用比较结果直接累加；工作数组首次申请后复用。

## 5.2 测试设计与自验

| 类型 | 目录 | 执行方式 |
| --- | --- | --- |
| 功能测试 | `tests/static_multimap/`（12 个用例文件，公共件 `tests/common/multi_container_test_common.h`） | `bash scripts/build.sh -r --test-name static_multimap` |
| 性能测试 | `tests/performance/static_multimap/`（8 个用例文件） | `bash scripts/build.sh -p`，再运行 `build/performance/static_multimap/*` |

自测报告包含每个用例的入参、结果对比、执行日志与性能数据；性能达标系数 = 标杆时延 / 实测时延。

## 5.3 兼容性分析

新增容器，不修改 `StaticMap`、`StaticSet`、`DynamicMap` 等已有容器的公开接口与实现；既有文件中仅 `tests/CMakeLists.txt` 增加测试目录。支持 CANN 9.0.0-beta.2 及以上版本，bisheng / ccec 两种编译器。

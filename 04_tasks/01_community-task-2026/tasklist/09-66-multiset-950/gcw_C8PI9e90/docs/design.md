# static_multiset 容器设计文档

> 版本：v0.1，设计评审稿；日期：2026-09-17。  
> 任务：9月社区任务-multiset容器开发(950)，官方任务清单第 66 项，提交目录 `09-66-multiset-950/gcw_C8PI9e90`。目标代码仓：`cann/ops-collections`。  
> 提交人（GitCode）：gcw_C8PI9e90。本文提出实现方案，尚未完成 Ascend C 实现或 950 实机验证；文中的性能为验收门限或估算，不是实测结果。

## 1. 需求背景（required）

### 1.1 需求来源

根据[9月社区任务-multiset容器开发(950)](https://www.hiascend.com/activities/task-center/details/214f40e300a347c7b21e445fa8671e62)提供的任务书，在 Atlas 950 上使用 Ascend C 实现与 NVIDIA cuCollections `static_multiset` 对应的静态多值集合，支持 I32、I64，完成设计、开发、测试与开源贡献。

本次设计 PR 仅提交本文件 `design.md`，放在 `cann/cann-ops-competitions` 的 `04_tasks/01_community-task-2026/tasklist/09-66-multiset-950/gcw_C8PI9e90/docs/` 下。设计评审通过后开展实现；实现和测试最终提交到 `cann/ops-collections`。本文完整说明设计，不依赖本地流程指南、脚本或其他附件。

### 1.2 参考版本与现状

本次已实际读取以下源码，避免依赖持续变化的分支描述：

| 参考项 | 核对版本 | 用途 |
| --- | --- | --- |
| ops-collections | `9d12996d4317e28420d74bcb1ec4d3b3507599ce` | StaticSet、探测策略、存储、原子封装、构建及测试框架 |
| NVIDIA/cuCollections，dev | `8ed532dc0ba3ba664d1d41027269908ddcde04c8` | `include/cuco/static_multiset.cuh` 的批量接口与多值集合语义 |
| 官方任务包 | 2026-09-17 核对版本 | 任务书、14 个功能测试文件和 10 个性能测试文件 |

该 ops-collections 基线提供 `StaticSet` 和 `StaticSetRef`，尚无 `static_multiset.h` 或 `static_multimap.h`。已有 set 插入遇到相同 Key 时按重复项处理，设备侧 `Count(key)` 返回存在标志，不能直接满足保留多个相同 Key 的要求。

可复用 `Extent`、`Storage`、默认哈希、桶探测、`AtomicCasWrap`、Ascend C 宏和测试基础设施；新增独立的 multiset 实现，避免改变现有 map/set 的语义。

### 1.3 目标与设计取舍

采用**稠密元素区 + 开放寻址索引表**：每次成功插入的 Key 在稠密区占一个位置，索引表存储该位置的编号。重复 Key 分别占据不同元素位置及索引槽。

选择该表示的原因是同时满足重复键语义、确定的容量截断、确定的检索输出，以及高吞吐的 `RetrieveAll`。稠密区是 Key 的唯一持久存储，索引表不再保存一份 Key。

相较直接在哈希槽保存 Key，本方案在当前测试容量下每槽额外使用 4 字节索引，查询多一次间接读取。该空间和访存代价是本次评审重点，并须经实机测试确认；不预先宣称优于直接槽位方案。

## 2. 需求分析（required）

### 2.1 功能范围

令容器当前元素序列为 `A[0:S)`，查询为 `q[0:N)`，定义：

```text
m(k) = A 中等于 k 的元素数量
Count(q) = Σ m(q[i])
CountEach(q)[i] = m(q[i])
CountEachOuter(q)[i] = max(m(q[i]), 1)
Retrieve(q) 的结果长度 R = Σ m(q[i])
```

`S` 统计全部元素，包括重复键，不是不同 Key 的数量。

| 接口 | 语义与输出 |
| --- | --- |
| Create / 构造 | 分配固定容量；初始 `S=0`；实际容量 `C≥请求容量` |
| Destroy / 析构 | 释放所有自有设备内存；不销毁调用者提供的 ACL 流 |
| Clear | 清空索引、将 `S` 置零，容量不变 |
| Insert | 保留每个成功插入的 Key；返回失败元素数量 |
| InsertIf | 仅插入 `pred(stencil[i])` 为真的项；跳过项不计失败 |
| Contains / ContainsIf | 每个位置输出 1 字节的 0/1；未命中或条件为假写 0 |
| Find / FindIf | 精确匹配时返回该 Key；未命中或条件为假写 `emptyKey` |
| Count | 返回所有查询位置的匹配总数，重复查询分别计数 |
| CountEach | 输出 N 个 `uint64_t` 计数 |
| CountEachOuter | 输出 N 个 `uint64_t`，未命中位置写 1 |
| Retrieve | 输出查询键、匹配键两个数组，保留每个查询位置的全部匹配 |
| RetrieveAll | 输出全部 S 个 Key，保留重复次数 |
| Size / Capacity | 测试所需辅助接口，分别返回 S 和 C |

例如插入 `[2,2,5]` 后，查询 `[2,7,2]`：Contains 为 `[1,0,1]`，Find 为 `[2,emptyKey,2]`，Count 为 4，CountEach 为 `[2,0,2]`，CountEachOuter 为 `[2,1,2]`，Retrieve 输出四组 `(2,2)`。

本次验收范围是同类型 I32/I64 的整数精确相等，以及作用于 stencil 的设备侧谓词。Erase、自动扩容、异构 Key 查询、任意自定义等价关系、异步 API 和用户 Kernel 并发修改/查询不列入首版扩展；如任务方要求这些能力，需先补充评审范围。

### 2.2 确定性定义

对相同的初始逻辑状态、输入、谓词和调用顺序：

1. 普通插入按输入下标顺序分配元素编号。
2. 条件插入按满足谓词的输入下标顺序分配编号。
3. 空间不足时保留最前面的可接纳项，后续被选中项计失败；不由线程抢占顺序决定保留哪些 Key。
4. Retrieve 按查询下标顺序组织结果，同一查询的相同 Key 连续输出。
5. RetrieveAll 按成功插入的逻辑顺序输出 `A[0:S)`。

因此公开的 Size、计数、查询与检索数组可重复。哈希索引槽的物理布局可能随调度变化，不作为公开结果；不提供允许修改内部数组的 `Data()` 接口。若验收要求内部哈希槽逐字节相同而非逻辑状态相同，需另行确认。

### 2.3 对标差异及拟采用口径

| 差异 | 本设计提案 | 依据／待确认点 |
| --- | --- | --- |
| Insert 返回值 | 返回失败数，成功插入返回 0；InsertIf 只统计被选中但未接纳的项 | 包内测试及 StaticSet；固定版本 cuco multiset 的 insert 返回 void |
| Create / Destroy | 构造、析构实现资源管理，同时提供显式 Create 工厂和 Destroy 方法 | 测试使用 RAII；显式接口覆盖任务书命名要求 |
| 参数顺序 | 提供输入指针、数量在前的入口，并提供测试使用的兼容重载；共享同一实现 | 任务书要求与参考顺序对应，但包内查询调用将 output 放在数量之前 |
| 零输入与空指针 | N=0 时允许输入、stencil、输出为空，返回零或不写入 | 功能测试明确要求 |
| Insert 的非零空输入 | `Insert(nullptr,N)` 返回 N，不改变状态 | `insert_test.cpp` 的明确断言；与通用参数表的“报错”存在差异 |
| 输入数量大于容量 | 查询数量可超过 C；插入超出剩余容量时部分接纳并返回失败数 | Insert 边界测试、CountEach 系列的查询数量均需要支持 |
| 检索输出顺序 | 使用 2.2 节的稳定顺序 | cuco retrieve 未规定顺序；任务书要求结果可重复 |
| 输出容量检查 | Retrieve / RetrieveAll 显式传入容量，容量不足时不写结果并抛错 | 包内测试接口包含 outputCapacity；裸指针不能反推出实际分配长度 |

这些是明确的拟实施决定，不代表已获得任务方确认。提交设计 PR 时应将本表作为需审核的接口约定。

## 3. 详细设计（required）

### 3.1 API 与类型

以下为接口设计声明，不是已经实现的头文件。`Extent` 使用 `aclco::Extent<std::size_t>`，计数和索引运算使用 64 位无符号类型；Host 的 `size_t` 要求为 64 位。Key 通过编译期断言限制为 `int32_t`、`int64_t`。

```cpp
template<class Key>
class StaticMultiset {
public:
    using Extent = aclco::Extent<std::size_t>;
    using SizeType = std::uint64_t;

    StaticMultiset(Extent capacity, Key emptyKey, aclrtStream stream);
    static StaticMultiset Create(Extent capacity, Key emptyKey,
                                 aclrtStream stream);
    void Destroy(aclrtStream stream);
    ~StaticMultiset() noexcept;
    // 禁止拷贝，支持 noexcept 移动；移动后源对象不再拥有资源。

    void Clear(aclrtStream stream);
    SizeType Size(aclrtStream stream) const;
    SizeType Capacity() const noexcept;
    SizeType Insert(void* keys, Extent n, aclrtStream stream);

    template<class StencilT, class Predicate>
    SizeType InsertIf(void* keys, Extent n, StencilT* stencil,
                      Predicate pred, aclrtStream stream);

    void Contains(void* queries, Extent n, void* output,
                  aclrtStream stream) const;
    void Find(void* queries, Extent n, void* output,
              aclrtStream stream) const;
    template<class StencilT, class Predicate>
    void ContainsIf(void* queries, Extent n, StencilT* stencil,
                    Predicate pred, void* output, aclrtStream stream) const;
    template<class StencilT, class Predicate>
    void FindIf(void* queries, Extent n, StencilT* stencil,
                Predicate pred, void* output, aclrtStream stream) const;

    SizeType Count(void* queries, Extent n, aclrtStream stream) const;
    void CountEach(void* queries, Extent n, void* output,
                   aclrtStream stream) const;
    void CountEachOuter(void* queries, Extent n, void* output,
                        aclrtStream stream) const;
    SizeType Retrieve(void* queries, Extent n, void* probeOutput,
                      void* matchOutput, Extent outputCapacity,
                      aclrtStream stream) const;
    SizeType RetrieveAll(void* output, Extent outputCapacity,
                         aclrtStream stream) const;
};
```

为直接接入任务测试，再提供以下转发重载；仅调整参数位置或构造 `Predicate{}`，不维护第二套算法：

```cpp
InsertIf<StencilT, Predicate>(keys, stencil, n, stream);
Contains(queries, output, n, stream);
Find(queries, output, n, stream);
ContainsIf<StencilT, Predicate>(queries, stencil, output, n, stream);
FindIf<StencilT, Predicate>(queries, stencil, output, n, stream);
CountEach(queries, output, n, stream);
CountEachOuter(queries, output, n, stream);
```

参数输出类型：Contains 系列为 1 字节 0/1，Find 和两个 Retrieve 输出为 Key，CountEach 系列为 `uint64_t`。Extent 表示元素个数，不是字节数。

显式谓词入口支持可拷贝至设备、无副作用且每次计算结果一致的仿函数；兼容重载另要求可默认构造。stencil 首先保证任务要求的 `uint32_t`，其他类型不宣称已支持。

固定 cuco 版本的 CountEach/CountEachOuter 显式接收 `probe_key_equal` 和 `probe_hash`；本任务测试省略它们。本提案使用容器固定的整数比较和哈希策略。该简化需在评审中确认，不能声称已经覆盖 cuco 的全部模板重载。

### 3.2 容器布局与容量

```text
Host owner
  capacity C、size S、emptyKey、indexWidth、状态和工作区所有权
       │
       ├── Device values[C] : Key          唯一的元素存储
       │       有效区 [0,S)，按接纳顺序排列
       ├── Device slots[C] : Index        哈希槽保存 values 的下标
       │       EMPTY_INDEX 表示空槽；每个有效下标恰出现一次
       └── Device scratch                 分块计数、前缀、归约及错误状态

StaticMultisetRef → 非拥有的只读视图，供设备 Kernel 查询
```

默认沿用库中的 `DoubleHashing<xxhash_32<Key>>` 和桶大小 B=5。令 P 为素数桶数，C=B×P。容量由 `MakeValidExtent` 的素数表规则向上取整，P 至少为 2，确保双重哈希步长与桶数互素。

本层在调用公共工具前额外检查：请求容量大于 0；向上取整、乘法及字节数不溢出；素数表存在可选值；所得 C 可表示并可分配。不能依赖公共工具把 0 自动提升为最小容量，因为任务测试要求创建容量 0 抛错。

在核对基线中，请求容量 128 得到 C=145，请求 100,000,000 得到 C=100,281,235，请求 200,000,000 得到 C=200,562,455。实现以 `Capacity()` 实际返回值为准。

- C≤`UINT32_MAX` 时用 32 位索引，`EMPTY_INDEX=UINT32_MAX`，有效编号为 `[0,C)`。
- 更大容量切换为 64 位索引，保留 `UINT64_MAX` 为空索引；前提是字节数和设备分配均合法。
- `emptyKey` 是 API 保留键，与索引空值是不同概念；测试传入 Key 类型的最小值。
- 没有删除操作，因此不需要 tombstone，也不需要对稠密区做中间压缩。

### 3.3 生命周期、流与错误处理

所有本次公开批量接口均为同步接口：成功返回前，相关流上的本次 Kernel 和输出写入已完成。构造也完成索引初始化再返回，便于 Create 的同步计时。对 ACL 分配、启动检查、拷贝和同步错误检查返回值并抛出带接口名和错误码的异常，不仅打印后继续。

同一对象的调用由调用者串行化；允许前一次同步调用完成后切换有效 ACL 流。首版不支持同一对象在多个 Host 线程或多个流上重叠调用。外部在另一个流产生输入时，调用者需先建立依赖。

创建分配 values、slots 及必要小工作区，只初始化 slots，不读取尚未写入的 values；Clear 将 slots 重置为空、S=0，可保留工作区用于复用。Destroy 释放自有资源，支持重复调用；析构 noexcept，并作为兜底释放。Destroy 不要求外部流由容器销毁。ACL 上下文必须比容器存活更久。

移动后及销毁后对象的 Capacity 为 0，其他数据操作报告对象无效。资源申请失败时回收已取得资源。若 Kernel 运行失败造成中间写入，标记对象不可继续使用，释放并重建；不把设备错误伪装为正常的“部分插入失败”。

### 3.4 参数和边界约定

| 场景 | 处理 |
| --- | --- |
| N=0 | 有效对象上不解引用数据指针，输出不变；计数／插入返回 0 |
| Insert 非零 N、keys=nullptr | 返回 N，状态不变，按包内测试兼容 |
| InsertIf 非零 N、缺少 keys 或 stencil | 抛 `invalid_argument`；不尝试统计未知谓词的选中数 |
| 其他查询非零 N、输入／固定长度输出为空 | 抛 `invalid_argument` |
| 被选中的插入 Key 等于 emptyKey | 验证阶段抛错，整个批次不写容器；谓词为假位置不读取 Key |
| 查询 Key 等于 emptyKey | 在本提案中按未命中处理 |
| 容量不足 | 按输入稳定顺序接纳前面的项，返回剩余选中项数量 |
| Retrieve 输出总数 R=0 | 不写输出，允许两个输出为空，包括 N>0 但全未命中的场景 |
| RetrieveAll 的 S=0 | 返回 0，允许空输出与零输出容量 |
| 输出容量小于 R 或 S | 写入之前抛错，两个输出均不发生部分写入 |
| 字节数或累计匹配数量溢出 | 抛 `overflow_error`，禁止回绕后分配／写入 |
| 输入输出重叠 | 不支持；按声明的地址区间可检查的重叠提前拒绝 |

裸 `void*` 本身不包含实际 dtype、数组长度或分配边界。Key 类型由模板保证；调用者负责指针实际类型、存储可访问性及声明长度正确。实现检查可知的空指针、对齐、算术和显式输出容量，不能宣称能从裸指针全面验证实际长度。若任务方要求强制检查底层分配长度，需另加带长度的 buffer 描述接口并评审。

### 3.5 Insert / InsertIf

插入分成验证／选择和写入两个阶段，验证与临时空间准备在状态修改之前完成。

**普通 Insert：**

1. 处理 N=0、空输入等 Host 边界；扫描本批全部 Key，检测保留键及错误。
2. 令剩余容量 F=C−S，接纳数量 M=min(N,F)。元素 i<M 的编号固定为 `S+i`；返回失败数 N−M。
3. 设备线程写 `values[S+i]=keys[i]`，按 Key 哈希，对 slots 用 CAS 抢占空槽并存入编号 `S+i`。
4. 全部写入同步完成并检查成功数为 M 后，Host 提交 `S←S+M`。

**条件 InsertIf：**

1. 按连续输入块计算选中数，同时验证选中 Key。只保存每块总数，不生成完整 Key 副本或全量 flag 数组。
2. 对块总数做 64 位排他前缀和，得到每块在选中序列中的起点及总数 E。
3. 写入阶段在块内重新计算纯谓词及局部前缀。选中元素的稳定排名为 r，仅当 `r<min(E,F)` 时写入 `values[S+r]` 并建立索引。
4. 返回 `E−min(E,F)`，同步成功后更新 S。两次计算要求输入和 stencil 在调用期间不变。

插入索引的伪代码：

```text
InsertIndex(key, elementIndex):
    start = h1(key) % P
    step  = 1 + h2(key) % (P - 1)
    for t = 0 .. P-1:
        bucket = (start + t * step) % P
        for j = 0 .. B-1:
            old = AtomicCAS(slots[bucket * B + j], EMPTY_INDEX, elementIndex)
            if old == EMPTY_INDEX: return success
    return internal_error
```

实际迭代使用不溢出的模加法更新 bucket，避免直接计算 `t*step`。每次 CAS 失败前进到下一槽，最多遍历 C 个槽，不使用等待其他线程解锁的自旋。重复 Key 继续占新槽，不判断“已有同值即成功”。

在选择阶段已保证总元素不超过 C，探测又覆盖全部槽，因此有效新编号应均能入表；整圈失败属于内部错误。并发时不通过非原子的旧槽读取决定跳过空位，基础版本直接使用 CAS 返回值，避免插入期缓存一致性假设。

### 3.6 查询、Count 与 CountEach

对查询 q，沿与插入相同的桶序列逐槽读取索引：遇到空槽终止；否则读取 `values[index]` 并做整数相等比较。Contains/Find 命中可提前返回；Count 必须继续累计后续同值项。满表无空槽时，在遍历 P 个桶后结束。

每个 Key 的探测路径是确定的，插入期间一个槽只会从空变为已占用，且每个插入扫描至第一个成功占用的槽；因此，在不并发写入的查询阶段，某个仍为空的槽之后不会存在该 Key 的有效条目。这是提前终止的依据。

- Contains / Find：一个逻辑线程处理一个或多个查询，输出按查询下标写入。
- ContainsIf / FindIf：先算谓词；条件为假仍显式写 0 或 emptyKey，不能依赖调用者清零。
- CountEach / CountEachOuter：每个查询得到 `uint64_t` 计数并独立写出。
- Count：线程局部计数、块级归约、设备端归约块结果，仅回读一个总数和错误状态；不分配 N 个计数的临时数组。

累计数量可能超过 32 位。所有归约和前缀加法携带溢出检查，发生 uint64 溢出时设错误状态，禁止回绕结果作为输出长度。相同查询出现多次时分别贡献 multiplicity。

### 3.7 Retrieve：稳定的变长输出

设查询块大小 T，块数 G=ceil(N/T)。采用两遍查询加分层前缀和：

1. 第一遍为每个查询计算 m(q)，在块内累加，仅保存 G 个块总数。
2. 在设备端对块总数做排他扫描，得到每块输出起点和总输出数 R。多层扫描通过多个 Kernel 边界完成，不在 Kernel 内假设全局栅栏。
3. 回读 R 和错误状态；检查输出容量、字节数、空指针和两个输出区间。空间不足时在写出前抛错。
4. 第二遍按块重新计算各查询计数，在块内做稳定前缀，得到每个查询的输出区间 `[offset,offset+m(q))`。
5. 本范围使用整数精确相等，所有匹配的 Key 都等于 q，可将 q 同时写入 probeOutput 与 matchOutput；写出数必须来自真实查询计数，不根据预期命中率推算。

同一查询输出区间可由多个线程分工写入；极端高 multiplicity 时按区间分段写，避免单线程负责全部输出。每块工作使用有限的寄存器／UB 临时空间，不能为全部 R 条记录分配额外键数组。

复杂度为两遍哈希查询加 O(R) 写出。相较保存 N 个 64 位计数，选择重算降低 Device 工作空间；其时间开销须实测评估。若将来增加非精确的自定义等价关系，第 5 步不能成立，必须改为输出真实匹配元素并重新设计稳定顺序。

### 3.8 RetrieveAll、Clear、Size

RetrieveAll 在检查输出容量后用设备 Kernel 连续复制 `values[0:S)` 到输出，不扫描哈希槽、不排序、不去重；同步后返回 S。该路径利用稠密存储，同时保证顺序和带宽利用。

Clear 仅重置 slots 和 S，无需清除无效的 values 尾部。Size 读取同步接口维护的 Host 元素数，Capacity 返回实际 C，不扫描设备内存。它们成立的前提是没有绕过 owner 的设备侧写入。

### 3.9 StaticMultisetRef 与内存可见性

`static_multiset_ref.h` 提供非拥有的设备只读视图，包含 values / slots 的地址、C、S、索引宽度和 emptyKey。设备侧基础操作为 `Contains(key)`、`Find(key)`、`Count(key)`；owner 的批量 Kernel 复用它。内部写入 helper `InsertIndex` 只接受 owner 预先分配的编号，不作为任意用户追加接口。

ref 不分配／释放内存；持有 ref 的设备任务结束前，owner 必须存活，且不能 Clear、Destroy、移动或插入。同一批插入线程不读取其他线程刚写入的 Key，只竞争索引 CAS；查询在插入完成的 Kernel／流同步边界之后开始。

应使用与实际 CANN 配套的 GM 写入完成和缓存可见性机制。官方 SIMT 哈希实践说明了弱内存模型与缓存问题；不能仅凭 Host 代码顺序推断核间可见性。实际实现须验证连续 Kernel、Clear 后重用和切换流后的读取，并按 SDK 要求使用相应 fence／缓存访问策略。

本设计未承诺设备用户自行并发 Insert 与 Find。任务书未详细列出 ref 的独立写接口要求，这一范围需评审确认；如必须支持，应增加独立编号预留和提交协议，不能直接复用当前 Host 的 S 缓存。

### 3.10 分核和临时空间规划

Host 从平台获取可用 AIV 核数，按工作规模限制启动核数；初始 SIMT 线程数候选为 256/512/1024，最终根据寄存器、UB 和吞吐测量选择。线程号、grid stride、N/C/S 均使用 64 位计算，不照搬当前公共 Kernel 的 uint32 数量参数。

验证、条件选择、前缀和采用连续逻辑块，初始 T=4096；每核可通过循环处理多个逻辑块。块内按子块处理，局部数组大小显式核算，不直接假定整个 T 项均放进线程寄存器。跨核归约通过独立 Kernel 实现。

工作空间按需申请、复用，在 Destroy 时释放。对 G 个逻辑块，块计数、偏移及分层扫描空间以 `32×G+O(1)` 字节作为初始规划上界，具体实现逐项核算；N=1 亿、T=4096 时约 0.78 MB。该上界不包含最终输出，且必须在实现后通过内存报告验证。

## 4. 性能、内存与方案比较

### 4.1 备选方案

| 方案 | 优点 | 未作为首选的原因 |
| --- | --- | --- |
| Key 直接放哈希槽，每次重复插入占新槽 | 结构最小，查询少一次间接读取，接近已有 set | 并行竞争下槽布局不稳定；RetrieveAll 按槽输出无法保证跨次构建的原始数组顺序，排序会增加时间与工作空间 |
| 每个不同 Key 存一份并维护 multiplicity | 高重复率下查询和存储有利 | 需额外并发计数与容量规则，首次出现顺序及 RetrieveAll 稳定枚举更复杂；本任务性能矩阵 multiplicity=1，压缩收益不明显 |
| 稠密元素区 + 哈希索引（本方案） | 稳定接纳和输出，RetrieveAll 连续复制，复用探测框架 | 额外索引空间和查询间接读取，需要实机验证是否满足全部性能门限 |

不采用按 N、测试命中率或测试生成的 Key 范围硬编码结果的特化。优化必须保持对任意合法 Key 和重复次数的正确性。

### 4.2 内存预算

设 K=sizeof(Key)，I=sizeof(Index)，实际容量 C，则持久 Device 存储：

```text
M_container = C × (K + I) + O(1)
M_peak = M_container + caller_inputs + caller_outputs + scratch
```

稠密区是容器唯一 Key 存储，不保留第二份全量输入副本；索引表是查询结构。它的额外容量是否符合任务书“合理容量”，需要以本节量化结果提交评审，不能仅以“属于容器”免除内存约束。

按当前容量归一化规则、请求 C_req=200,000,000、实际 C=200,562,455、32 位索引计算，以下单位为十进制 GB：

| 项目 | I32 | I64 |
| --- | ---: | ---: |
| 容器 values + slots | 1.604500 | 2.406749 |
| 1 亿 Key 输入 | 0.400000 | 0.800000 |
| 1 亿查询 Key | 0.400000 | 0.800000 |
| Retrieve 两个各 1 亿元素的输出 | 0.800000 | 1.600000 |
| 上述 Retrieve 单 dtype 合计，未含运行时／工作区 | 3.204500 | 5.606749 |

性能用例的 dtype 上下文是函数静态对象；同一进程内 I32/I64 上下文可能同时存活。对应 Retrieve 数据结构合计约 **8.811249 GB**，还需留出 ACL 运行时、其他分配和工作区余量，不能只按单个 dtype 估算设备需求。Host 生成数组也有独立内存开销。

Create/Destroy 性能用例请求容量为 1 亿，对应容器约 0.802250 / 1.203375 GB。内存分配失败应清晰报错，不降低测试规模后声称完成原验收用例。

### 4.3 性能标准和优化方向

全部 44 项性能用例按任务书各自基线验收：`实测时延 ≤ 标杆时延 / 0.4`。以下只是门限举例，全部基线以任务书为准：

| 接口／场景 | dtype | 标杆 ms | 可接受时延上限 ms |
| --- | --- | ---: | ---: |
| Insert，N=1e8，occupancy=0.5 | I32 | 8.098100 | 20.245250 |
| Insert，同上 | I64 | 8.392043 | 20.980108 |
| RetrieveAll，N=1e8，occupancy=0.5 | I32 | 0.660264 | 1.650660 |
| RetrieveAll，同上 | I64 | 1.016854 | 2.542135 |
| Count，匹配率 0.1 | I64 | 4.508944 | 11.272360 |
| CountEach，匹配率 1.0 | I32 | 5.407013 | 13.517533 |

表中门限四舍五入显示，判定时用原始数值计算，不能因显示精度放宽门限。

优先优化：索引宽度使用 32 位；查询批量并行隐藏索引到 Key 的依赖延迟；合并验证与块统计；避免每个成功插入执行全局 size 原子加；复用小工作区；RetrieveAll 连续搬运；必要时评估多个线程合作探测桶及高重复查询的负载划分。任何优化均用真实设备数据决定。

高重复率时本表示的插入、计数可能需要扫描较长探测链；高负载率和满表未命中查询最坏为 O(C)，但必须有限终止。性能矩阵中 multiplicity=1 不免除对 1024 重复度等功能场景的支持。

## 5. 可维可测分析

### 5.1 工程改动范围

```text
ops-collections/
├── include/static_multiset.h
├── include/static_multiset_ref.h
├── include/detail/static_multiset/
│   ├── static_multiset.inl
│   ├── static_multiset_ref.inl
│   ├── storage.h
│   ├── kernels.h
│   └── scan.h
├── tests/static_multiset/                 # 14 个配套测试及必要补充用例
├── tests/performance/static_multiset/     # 10 个配套性能测试文件
├── tests/common/multi_container_test_common.h
├── tests/CMakeLists.txt                   # 增加两组源码收集路径
├── docs/static_multiset_API文档和使用示例.md
└── README.md
```

内部文件划分为实现计划，可按代码评审调整。当前 CMake 显式列出各容器目录，新增文件不会自动被现有 glob 收集，必须增加 multiset 路径。

公共测试头无条件包含缺失的 `static_multimap.h`。拟将 map 专属 include、factory 和辅助函数放到匹配的条件编译范围，或与任务方确认后拆成独立公共头；保持全部 multiset 测试断言和输入矩阵不变，并在提交中说明这一接入改动。

复用 `tests/common/acl_env.h`、`device_buffer.h` 和性能注册框架。原始任务包作为验收对照保留；新增用例单独添加，不用修改期望值掩盖实现偏差。

### 5.2 功能测试矩阵

| 功能 | 配套展开用例数 |
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

计数来自任务书，按 dtype、GENERATE 和 SECTION 展开；不把文件数量或断言次数当作用例数。

补充验证重点：

- 随机正负整数、I64 超出 I32 范围的值、保留键拒绝、全相同 Key、哈希碰撞、满表未命中。
- InsertIf 含重复 Key、全真／全假／混合及超容量选择，检查接纳序列和失败数。
- ContainsIf / FindIf 预填非零输出，验证条件为假的位置被覆盖。
- 输出容量不足时输出未被部分写入；R=0 时的空输出；重复查询的完整结果多重集。
- 同一输入多次重建，对未排序的 RetrieveAll 和 Retrieve 输出逐元素比较；超容量也应稳定。
- Count 和前缀和的 64 位累计、算术溢出路径、移动和重复 Destroy、分配失败清理。
- 连续插入／查询、Clear 后复用、同步后换流、对象析构顺序及重复生命周期下的内存占用。

CPU 参考使用 multiset／频数表和稳定接纳序列。包内 Retrieve 主要检查总数与输出键合法性，应补充每个查询完整 multiplicity 的对照，避免“总数正确、分配给各键的次数错误”漏检。

### 5.3 性能与内存验证

保留配套 44 项、1 亿输入测试参数；逐行记录 dtype、容量、occupancy、multiplicity、matching rate、baseline_ms、measured_ms、ratio 和结论。记录设备、系统、CANN、编译器、架构选项与源码 commit。

配套 `Measure` 使用 CPU 墙钟，并把同一数值填入 `TestResult` 的 CPU 和 Device 两个字段；已核对框架字段单位为 **微秒**。本设计用同步 API 保证测量含设备完成，但报告必须称其为接口端到端时延，不能将该字段直接当成设备事件测量值。可另附设备事件／profiler 结果，区分两种口径，向任务方确认与标杆的比较标准。

记录预热、迭代次数和波动；不能将有成本的验证、分配或同步移出接口计时后继续沿用原接口结果。框架最多执行 50 次并根据累计耗时和波动决定停止，最终日志应保留实际运行次数。

内存测量区分容器持久内存、调用者输入输出、工作区、Host 数据以及框架静态上下文。针对 Create/Destroy/Clear 重复执行观察内存是否持续增长，验证无泄漏。

### 5.4 构建与运行计划

以下命令根据核对基线的 `scripts/build.sh`、`tests/CMakeLists.txt` 编写，须在加入 multiset 源码和构建路径后于 950 执行；目前未执行。先按实际安装路径加载 CANN 环境，再确认工具链和目标架构：ccec 基线默认 `dav-c310`，bisheng 默认 `dav-3510`，最终取值必须与安装版本及设备匹配。

```bash
# ops-collections 根目录；-b 包含清理和重建
bash scripts/build.sh -b
bash scripts/build.sh -r --test-name static_multiset

# 构建性能可执行文件，只运行 multiset 性能项
bash scripts/build.sh -p
ctest --test-dir build/performance -R '^static_multiset_perf_' --output-on-failure
```

功能可执行文件按 `collection_tests_static_multiset_<文件名>` 命名；性能可执行文件位于 `build/performance/static_multiset/`。全仓性能脚本 `-rp` 会运行所有容器，不能把其他容器的日志混计为本任务的 44 项。

## 6. 兼容性分析

新增容器及内部实现，不改变已有 StaticSet/StaticMap 的重复键、返回值或线程行为。保留现有 C++17 和纯头文件集成方式，使用 `COLLECTION_SIMT_DEVICE` 等仓库宏。

任务最低 CANN 版本为 9.0.0-beta.2；计划以 CANN 9.1.0 开发环境完成首轮构建。验证前只声明设计目标，不宣称已通过任一 CANN 版本兼容性测试。系统采用对应 CANN 支持的 Linux 发行版，任务书未指定 openEuler 或 Ubuntu。

若复用公共工具需要修复容量溢出、错误传播等问题，优先在 multiset 层封装；确需修改公共实现时单独说明影响，并运行相关 utility、StaticSet、StaticMap 回归。不能为本任务静默改变旧容器行为。

## 7. 当前验证状态与风险

### 7.1 已完成的设计验证

已核对任务书、配套测试、官方设计模板及上述固定版本的基础仓与 cuco 源码；已用 CPU 抽象模型模拟逐槽原子占位的交错调度，执行 **2,000 组调度、22,000 次查询对照**，检查稳定接纳、重复保留、探测终止、计数和逻辑输出。

模型使用素数桶数 2、3、7、29，桶大小为 5；构造包含空批次、重复键、条件选择和超容量插入的 80 组场景，每组模拟 25 种逐槽 CAS 调度。以稳定接纳序列和 CPU 频数表为参照，检查每个有效元素编号恰好出现在一个槽位，并对每组调度的 11 个查询比较计数、外计数及检索结果。

该模型验证算法层面的不变量，**不覆盖 Ascend C 编译、设备内存模型、真实并发、所有配套功能用例或性能**。模型使用小型哈希函数，不是 SDK 中的 xxhash 实现；以上结果仅作为设计阶段的辅助检查，正式验收以 950 上的实现和完整测试日志为准。

### 7.2 风险与处理

| 风险 | 处理／验证要求 |
| --- | --- |
| 额外索引空间是否满足合理容量要求 | 提交 4.2 节预算供评审，实机报告实际峰值；不把该问题留到最终验收 |
| 间接读取导致 Contains/Count 性能不足 | 首先在 I32/I64、0.5 occupancy 全匹配率矩阵验证；比较线程配置、桶读取与缓存策略 |
| 满表或高重复导致长探测链 | 显式 C 槽上界，压力测试无死循环；记录慢例并评估合作探测 |
| SIMT 原子、缓存和跨 Kernel 可见性 | 使用配套 SDK 的原子封装和同步策略，950 上执行竞争与复用验证 |
| 64 位索引、计数和模板编译差异 | 独立编译／微测试；避免把已有 uint32 Kernel 直接用作 uint64 接口 |
| 两遍 Retrieve 查询开销 | 与保存计数的时间／内存方案作实际测量比较，必要时在受控内存预算内调整 |
| 裸指针无法验证实际分配长度 | 明确调用者契约与可检测范围；若验收要求更多元数据，先调整 API 设计 |
| 配套接口与任务书冲突 | 按 2.3 节和下列问题取得任务方确认，更新文档版本后实施 |

### 7.3 提交评审时需确认的问题

1. 是否接受稠密元素区与索引表的空间开销，以及逻辑状态／公开数组确定、内部桶布局不承诺一致的定义？
2. 是否接受任务测试要求的空指针返回失败数、超容量部分接纳、查询数量超过容量及本设计的稳定截断规则？
3. 是否接受任务所列同类型整数默认比较／哈希范围、查询兼容重载和 CountEach 的策略参数简化？
4. ref 是否只需支持本方案的设备只读查询与内部批量写入 helper，还是还要求独立用户设备写接口？
5. 是否认可当前配套性能用例的同步接口端到端计时口径，与任务书基线的比较方式是否一致？

上述问题属于设计评审项。未确认前不称为最终定稿，未跑设备测试前不填写性能达标结论。

## 8. 实施与验收计划

| 顺序 | 内容 | 输出与完成条件 |
| --- | --- | --- |
| D0 | 提交本设计、确认接口差异和风险 | 设计 PR 明确评审通过，并在验收前合入 |
| D1 | owner、存储、错误处理、生命周期 | I32/I64 Create、Destroy、Clear、Size、Capacity 用例通过 |
| D2 | 验证与稳定选择、索引插入、基础查询 | Insert/InsertIf、Contains/Find 及 If 变体通过 |
| D3 | 计数归约、分层扫描和检索 | Count 系列、Retrieve、RetrieveAll 及补充确定性用例通过 |
| D4 | 950 调优与回归 | 全部 1258 配套功能用例、44 性能用例、内存验证和相关回归有真实日志 |
| D5 | 验收材料、代码贡献 | 固定 commit、自测报告、复现说明、API 文档与示例；按任务流程验收和合入 |

实际排期需在环境可用时间明确后制定。任务页当前截止日期为 2026-10-27，应为评审、驳回修改和实机调优预留时间。

## 9. 参考资料

- [官方任务页及所附任务包](https://www.hiascend.com/activities/task-center/details/214f40e300a347c7b21e445fa8671e62)：任务书与配套测试是需求、验收和测试接口来源。
- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)：章节依据。
- [ops-collections 固定基线](https://gitcode.com/cann/ops-collections/tree/9d12996d4317e28420d74bcb1ec4d3b3507599ce)：`include/static_set.h`、`static_set_ref.h`、`detail/open_addressing/`、`detail/extent/extent.inl`、`probing_scheme.h`、`utility/atomic_cas_wrap.h`、`tests/CMakeLists.txt`、`tests/performance/performance_test_framework.h`。
- [cuCollections 固定版本 static_multiset](https://github.com/NVIDIA/cuCollections/blob/8ed532dc0ba3ba664d1d41027269908ddcde04c8/include/cuco/static_multiset.cuh)：批量接口和多值集合语义。
- [Ascend C SIMT 哈希表实践](https://asc.gitcode.com/guide/operator_practice/best_practices/simt_insert_hash_table.html)：原子占位及弱内存模型说明；该站为持续开发文档，具体 API 以安装版本为准。
- [任务详情](https://www.hiascend.com/activities/task-center/details/214f40e300a347c7b21e445fa8671e62)和[任务讨论帖](https://gitcode.com/cann/ops-collections/discussions/3)：报名、截止日期和设计疑问确认。

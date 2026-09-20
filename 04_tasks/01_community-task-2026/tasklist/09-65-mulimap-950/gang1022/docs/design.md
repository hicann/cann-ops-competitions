# static_multimap 容器设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 9 月社区任务-mulimap 容器开发（950） |
| 任务编号 | 09-65 |
| 提交者 | gang1022 |
| 邮箱 | 16601526267@163.com |
| 目标仓库 | `cann/ops-collections` |
| 参考实现 | NVIDIA cuCollections `static_multimap` |
| 文档状态 | 设计评审稿；尚未填写未经实测的性能数据 |

# 需求背景（required）

## 需求来源

本任务来源于 CANN 社区任务中心，要求参考 NVIDIA cuCollections 的 `static_multimap`，在昇腾 NPU 上使用 C++ 和 Ascend C 实现静态多值键值映射容器，并按 ops-collections 纯头文件容器工程模式完成设计、开发、功能测试、性能测试及验收。

需求依据按以下优先级执行：社区任务书及评审结论、官方测试包、ops-collections 现有工程规范、锁定版本的 cuCollections 参考实现。其他已合入设计文档仅用于参考文档组织形式。

## 背景介绍

### 参考实现分析

`static_multimap` 是创建后容量固定的多值哈希容器。它与 `static_map` 的主要区别是：相同 Key 可以对应多个 Value，重复的 `(Key, Value)` 也必须分别保存，插入时不能因为已存在相同 Key 而提前结束。

容器需支持以下语义：

1. Create、Destroy 和 Clear 管理固定容量容器的生命周期；
2. Insert、InsertIf 插入全部符合条件的记录，重复记录不去重；
3. Contains、ContainsIf 判断 Key 是否存在；
4. Find、FindIf 为每个查询 Key 返回一个匹配 Value，未命中返回 `emptyValue`；
5. Count 返回所有查询 Key 的匹配次数总和，重复查询分别计数；
6. Retrieve 返回每个查询 Key 的全部匹配记录，RetrieveAll 返回容器内全部记录；
7. 相同初始状态和相同调用序列下，容器可观察状态及输出确定且可重复。

### ops-collections 现状

ops-collections 已有 `static_map`、`static_set` 等纯头文件容器，可复用以下工程能力：

- Host 门面、ACL Runtime 资源管理及流同步方式；
- Ascend C 两级 SIMT Kernel 组织方式和仓库公共宏；
- 32/64 位原子 CAS 包装、哈希函数、参数校验及异常处理方式；
- Catch2 功能测试、性能测试和 CMake 集成框架。

`static_map` 是唯一键容器，不能直接复用其“遇到相同 Key 即停止插入”的协议。多值插入、全匹配枚举和稳定输出需要在 `detail/static_multimap/` 中独立实现。

# 需求分析（required）

## 需求描述

实现 Key、Value 同为 `int32_t` 或同为 `int64_t` 的 `StaticMultimap`。容器应支持 12 个任务接口、空输入、重复 Key、重复 Pair、不同 multiplicity、命中与未命中、容量边界，并满足以下标准：

- 功能结果与任务书及参考语义一致；
- 相同输入和容器状态下结果确定且可重复；
- 1086 个展开功能用例全部通过；
- 32 个性能用例的性能均不低于标杆的 0.4 倍；
- 除容器、输出和必要工作空间外，不复制一份完整的输入数据。

## 需求拆解

| 类别 | 设计要求 |
| --- | --- |
| 数据类型 | `int32_t/int32_t`、`int64_t/int64_t`；Stencil 为 `uint32_t` |
| 多值语义 | 相同 Key 及完全相同 Pair 均作为独立记录保留 |
| 容量语义 | 最多保存 Capacity 条记录；空间不足时按输入顺序接纳可容纳的前部有效记录 |
| 条件语义 | 仅 `pred(stencil[i]) == true` 的位置参与操作 |
| 确定性 | Find 固定选择最早成功插入记录；Retrieve 和 RetrieveAll 使用稳定顺序 |
| 边界 | `numInputs == 0` 是合法空操作；正长度空指针、容量 0、非法哨兵输入报错 |
| 工程 | 纯头文件实现，不新增仓库根目录独立工程 |
| 验收 | 功能、性能、内存报告及完整日志均可复现 |

### 需求冲突处理

cuCollections `dev` 分支会变化，部分接口返回值和任务书示例并不完全一致。本项目以官方测试包的实际调用和任务书验收语义为准：

- `Insert/InsertIf` 返回未成功插入的元素数量；
- `Count` 返回所有查询的 64 位匹配总数；
- `Retrieve/RetrieveAll` 的正确性按多重集验证，同时本实现提供稳定原始输出；
- 裸 `void*` 无法在运行时获知真实分配长度和 dtype，Host 仅校验可检测项目，实际分配长度由调用者保证。

# 详细设计（required）

## 算子分析

### 语义形式化

设容器记录序列为：

```text
R = [(k0, v0), (k1, v1), ..., (ks-1, vs-1)], 0 <= s <= Capacity
```

记录允许重复。对查询键 `q`：

```text
Contains(q) = exists i, ki == q
Find(q)     = vj，其中 j = min{i | ki == q}；不存在时为 emptyValue
Count(Q)    = sum(q in Q) |{i | ki == q}|
Retrieve(q) = [(ki, vi) | ki == q]，按记录编号 i 递增
RetrieveAll = R，按记录编号递增
```

这里的记录编号由成功插入次序确定，因此哈希目录中槽位的并发竞争不会改变可观察结果。

### 接口设计

接口参数顺序与官方测试及 ops-collections 风格保持一致，设备迭代器使用“首地址 + 数量”表达。

| 接口 | 主要形式 | 行为或返回值 |
| --- | --- | --- |
| Create | `StaticMultimap(capacity, emptyKey, emptyValue, stream)` | 分配并初始化资源；容量 0 报错 |
| Destroy | 析构/内部 Destroy | 等待必要依赖并释放资源；析构不抛异常 |
| Clear | `Clear(stream)` | 清空目录并令 Size 为 0，容量不变 |
| Insert | `Insert(pairs, valueNum, stream)` | 返回插入失败数量 |
| InsertIf | `InsertIf<StencilT, Pred>(pairs, stencil, valueNum, stream)` | 返回选中但插入失败数量 |
| Contains | `Contains(keys, output, keyNum, stream)` | 每个位置写 0 或非 0 |
| ContainsIf | `ContainsIf<StencilT, Pred>(keys, stencil, output, keyNum, stream)` | 谓词为假时写 0 |
| Find | `Find(keys, outputValues, keyNum, stream)` | 返回最早记录的 Value；未命中写 `emptyValue` |
| FindIf | `FindIf<StencilT, Pred>(keys, stencil, outputValues, keyNum, stream)` | 谓词为假或未命中写 `emptyValue` |
| Count | `Count(keys, keyNum, stream)` | 返回所有查询的匹配总数 |
| Retrieve | `Retrieve(keys, keyNum, outKeys, outValues, outputCapacity, stream)` | 返回实际匹配记录数 |
| RetrieveAll | `RetrieveAll(outKeys, outValues, outputCapacity, stream)` | 返回当前全部记录数 |

同时提供测试需要的 `Capacity()` 和 `Size(stream)`。容器禁止复制，允许移动，资源所有权始终唯一。

### 数据类型与形状

| 数据 | dtype | 形状 |
| --- | --- | --- |
| 输入 Pair | `Pair<int32_t,int32_t>` 或 `Pair<int64_t,int64_t>` | `[valueNum]` |
| 查询 Key | `int32_t` 或 `int64_t` | `[keyNum]` |
| Stencil | `uint32_t` | `[numInputs]` |
| Contains 输出 | `unsigned char` | `[keyNum]` |
| Find 输出 | 与 Value 相同 | `[keyNum]` |
| Retrieve 输出 | Key、Value 两个同长度数组 | `[matchNum]` |
| Count/Size/返回数量 | `uint64_t`/SizeType | 标量 |

## 算子实现

### 总体方案

采用“顺序记录区 + 开放寻址索引目录”结构：

```text
Host StaticMultimap
  ├─ records[Capacity]      按成功插入次序保存 Pair
  ├─ indexSlots[Capacity]   保存 recordId，EMPTY 表示空槽
  ├─ size                   当前记录数
  └─ emptyKey/emptyValue、容量和运行时上下文
```

`records` 是容器的实际键值存储，不是输入副本；`indexSlots` 只保存记录编号。容量小于 `UINT32_MAX` 时使用 32 位编号，否则选择 64 位编号并保留一个 EMPTY 哨兵。所有容量乘法和字节数计算均进行溢出校验。

该结构的目的如下：

1. 插入记录编号稳定，不受 SIMT 线程抢占顺序影响；
2. 查询通过哈希目录完成，无需扫描全部记录；
3. RetrieveAll 连续读取记录区，不扫描空槽；
4. Find 可在命中记录中选择最小 recordId，获得固定结果；
5. 哈希物理布局可以变化，但不影响任何对外输出。

### Host 侧设计

Host 门面负责：模板和参数检查、Device 内存生命周期、Kernel 编排、必要的流同步、标量返回值回读和错误转换。

公共操作遵循调用者指定的 ACL Stream。同一容器的修改操作串行化；跨流调用由实现记录事件或要求调用方建立依赖。未同步的并发写、写读和对象析构不属于合法使用方式。

运行期根据数据类型、索引宽度、输入规模和平台可用 AIV 核数选择 Kernel 实例。分核采用网格步进，每个逻辑线程处理一个或多个输入；块大小通过 Atlas 950 实测调优，不在设计阶段硬编码未经验证的最优值。

不需要传统独立算子的 Host TilingData。分支通过 C++ 模板实例和少量运行期规模参数传入，避免每次调用生成复杂 Tiling。

### Create、Destroy 和 Clear

- Create 分配 `records` 与 `indexSlots`，将目录初始化为 EMPTY，Size 初始化为 0；实际容量不小于请求容量。
- Destroy 在相关流工作完成后释放资源，显式销毁与析构共用幂等释放逻辑。
- Clear 仅并行重置目录并清零 Size。旧 records 内容成为不可达数据，无需为正确性重复清零，从而减少带宽。

### Insert 和 InsertIf

插入分为四步：

1. 参数检查和非法哨兵 Key 检查；非法输入在修改容器前报告错误。
2. Insert 直接使用输入下标；InsertIf 对谓词结果执行分块计数和排他前缀和，得到稳定选中序号。
3. 根据剩余容量，按原输入顺序接纳前 `min(selected, Capacity-Size)` 条记录，写入 `records[oldSize + rank]`。其余选中项计入失败数量。
4. 在同一 Stream 的后续 Kernel 中，将每个新 recordId 按 Key 哈希后用原子 CAS 写入目录。重复 Key 或重复 Pair 继续探测空槽，不作去重。

探测次数最多为目录容量，避免满表无限循环。若在容量计数一致时仍无法建立索引，则作为内部错误报告，不能静默返回成功。空输入不启动 Kernel；`Insert(nullptr, n>0)` 按官方测试契约返回 `n`，不发生非法访存。

### Contains、Find 和 Count

每个查询从 `hash(key)` 开始按固定线性探测序列读取 `indexSlots`，遇 EMPTY 或达到容量结束：

- Contains 命中后可提前结束；
- Find 遍历全部有效探测段，取 Key 相同记录中的最小 recordId；
- Count 累计全部匹配，重复查询分别贡献计数，随后做核内和跨核 64 位整数归约。

条件版本始终显式写输出：谓词为假时 ContainsIf 写 0，FindIf 写 `emptyValue`。整数比较和整数归约不存在浮点累加顺序误差。

### Retrieve

Retrieve 使用“计数—偏移—写出”三阶段：

1. 统计每个查询的匹配数；
2. 对计数做 64 位排他前缀和，得到每个查询的输出区间，并检查总数不超过 `outputCapacity`；
3. 枚举匹配 recordId，在每个查询段内按 recordId 递增后写出 Key 和 Value。

小 multiplicity 使用寄存器/局部小数组排序；大 multiplicity 使用只保存 recordId 的分段排序工作区。该工作区属于输出定位所需索引，不复制完整输入 Pair。性能验收场景 multiplicity 为 1，可走无需排序的快速路径。

同一查询 Key 出现多次时，每个查询位置均输出一份完整匹配集合。输出空间不足在实际写出前报错，禁止截断或越界写入。

### RetrieveAll

RetrieveAll 校验 `outputCapacity >= Size` 后，按 `[0, Size)` 并行连续读取 records，并拆分写入 Key、Value 输出数组。该路径不扫描哈希目录、不使用全局原子追加，也无需排序，输出固定为成功插入顺序。

### 确定性设计

确定性比较对象是相同初始状态和相同合法调用序列下的：Size、失败数量、记录多重集及所有原始输出。

- 容量不足时按输入顺序稳定接纳；
- recordId 等于稳定插入序号；
- Find 固定选择最小 recordId；
- Retrieve 按查询输入顺序分段，段内按 recordId 排序；
- RetrieveAll 按 recordId 输出；
- 哈希目录的物理槽位不是对外状态。

因此内部 CAS 竞争即使产生不同目录布局，仍不会改变可观察结果。

### 内存设计

容器常驻内存近似为：

```text
Capacity * (sizeof(Pair<Key,Value>) + sizeof(IndexType))
```

以 Capacity=200,000,000 且使用 32 位索引估算：I32/I32 约 2.4 GB，I64/I64 约 4.0 GB，未计分配器对齐。Retrieve 的前缀和及大 multiplicity 索引区按需申请并复用；输入、输出和测试数据不计入容器本身。实现阶段使用 Runtime 内存查询和性能测试记录真实峰值，不以估算代替报告。

### 工程目录

```text
ops-collections/
├── include/
│   ├── static_multimap.h
│   ├── static_multimap_ref.h
│   └── detail/static_multimap/
│       ├── static_multimap_impl.h
│       ├── static_multimap_kernel.h
│       ├── probing.h
│       └── storage.h
├── tests/
│   ├── static_multimap/
│   └── performance/static_multimap/
└── docs/static_multimap_API文档和使用示例.md
```

实现细节文件名可在编码阶段按仓库命名规范调整。还需更新仓库 README 和 `tests/CMakeLists.txt`，不得新增根目录独立算子工程。

## 支持硬件

| 项目 | 要求 |
| --- | --- |
| 硬件 | Atlas 950 系列及后续支持 SIMT 的昇腾产品 |
| CANN | 9.0.0-beta.2 及以上 |
| 编译器 | 仓库支持的 ccec 或毕昇 ASC，架构值与实际 CANN 环境匹配 |
| C++ 标准 | 按 ops-collections 当前工程使用 C++17 |
| 构建工具 | CMake 3.16 及以上 |
| 测试框架 | Catch2 3.5.4 及以上 |

## 算子约束限制

1. Key 与 Value 在验收场景中类型相同，仅支持 I32/I32 和 I64/I64。
2. `emptyKey` 为保留 Key，不能作为待插入或查询的有效 Key；`emptyValue` 用作 Find 未命中标记。
3. `numInputs == 0` 时允许相关输入、输出指针为空；正长度时指针必须有效。
4. 裸指针真实长度和 dtype 由调用者保证；实现负责容量、算术溢出及可检测指针错误。
5. Retrieve/RetrieveAll 的输出容量必须足够，不支持静默截断。
6. 同一容器不支持未同步的多流并发修改或边写边读。
7. Hash 最坏冲突下探测复杂度为 O(Capacity)，所有循环均有容量上界。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 来源 |
| --- | --- | --- |
| 功能 | 12 个接口满足任务书和官方测试语义 | 社区任务书 2.1、2.3 |
| dtype | I32/I32、I64/I64 | 社区任务书 2.1 |
| 确定性 | 同一初态和调用序列下，状态及原始输出一致 | 社区任务书 2.1、3.2 |
| 功能用例 | 1086 个展开用例全部通过 | 社区任务书 3.5 |
| 性能用例 | 32 个用例均达到标杆性能的 0.4 倍 | 社区任务书 3.3 |
| 时延 | 实现时延不大于标杆时延的 2.5 倍 | 社区任务书 3.3 |
| 内存 | 无不必要的输入规模线性重复 Device 拷贝 | 社区任务书 3.4 |

功能测试矩阵：

| 接口 | 展开用例数 |
| --- | ---: |
| Create / Destroy / Clear | 14 / 18 / 32 |
| Insert / InsertIf | 290 / 20 |
| Contains / ContainsIf | 100 / 56 |
| Find / FindIf | 100 / 56 |
| Count | 100 |
| Retrieve / RetrieveAll | 98 / 202 |
| 合计 | 1086 |

测试覆盖空输入、完全重复 Pair、重复 Key 不同 Value、multiplicity 1/2/4/8、命中/未命中、容量不足、Clear 后复用、移动构造、多轮创建销毁和确定性重复执行。Host 侧使用多重集语义生成 golden；Find 同时验证合法匹配值和本实现固定选择规则。

性能测试共 32 个配置：Create/Destroy 4 个，Insert/RetrieveAll 4 个，Contains/Find/Retrieve/Count 按两个 dtype 和 0.1/0.5/1 三种匹配率展开 24 个。Create/Destroy 使用 Capacity=100,000,000；其他性能场景使用 NumInputs=100,000,000、Occupancy=0.5、Multiplicity=1，对应容器容量 200,000,000。

设计阶段不填写伪造实测结果。实现完成后在 Atlas 950 上记录每项原始样本、统计时延、标杆时延、性能倍率和峰值 Device 内存，并保存日志。

建议复现命令：

```bash
bash scripts/build.sh -b
bash scripts/build.sh -r --test-name static_multimap
bash scripts/build.sh -p
bash scripts/build.sh -rp
```

## 兼容性分析

该功能为新增容器，不改变已有 static_map、static_set 等公共接口。实现使用独立命名空间和目录，模板定义保持头文件可见；公共基础组件仅在兼容现有调用方时复用或扩展。编译阶段验证多翻译单元包含、I32/I64 实例化及 ccec/毕昇 ASC 两种仓库支持路径。

## 风险与验证措施

| 风险 | 影响 | 措施 |
| --- | --- | --- |
| 高冲突或接近满表 | 探测时延长 | 有界探测、负载率和极端冲突专项测试 |
| I64 原子发布可见性 | 读到未完成记录 | 记录写入和索引发布分 Kernel、同流排序，专项压力测试 |
| 稳定排序开销 | Retrieve 高 multiplicity 变慢 | multiplicity=1 快速路径，小段局部排序，大段分段排序 |
| 亿级规模内存峰值 | 分配失败 | 实现前核算、复用工作区、逐配置释放并记录峰值 |
| 参考分支持续变化 | 接口理解漂移 | 锁定参考提交，正式语义以任务书、测试包和评审结论为准 |
| 测试口径不一致 | 性能结果不可比 | 原样保留官方数据规模、计时范围和统计方式 |

## 交付物

1. 本设计文档及其评审 PR；
2. `static_multimap` 头文件实现、功能测试、性能测试、API 文档和 README 更新；
3. 功能、性能和内存自验证报告及原始日志；
4. 个人待验收代码仓、分支和可复现说明。

## 参考资料

1. static_multimap 社区任务书；
2. [cuCollections static_multimap](https://github.com/NVIDIA/cuCollections/blob/dev/include/cuco/static_multimap.cuh)；
3. [ops-collections](https://gitcode.com/cann/ops-collections)；
4. [CANN 社区任务设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。

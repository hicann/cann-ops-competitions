# static_multimap 容器设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 9月社区任务-mulimap容器开发（950） |
| 作者账号 | wuwenqi2026 |
| 交付仓库 | `cann/ops-collections` |
| 参考实现 | NVIDIA cuCollections `static_multimap` |
| 目标硬件 | Atlas 950 系列及后续支持 SIMT 的昇腾产品 |
| CANN 版本 | 9.0.0-beta.2 及以上 |

# 需求背景（required）

## 需求来源

社区任务要求参考 cuCollections `static_multimap`，在 Ascend 950 上实现固定容量、多值键值映射容器，完成设计、编码、功能测试、性能测试和开源合入。

主要参考：

1. 社区任务包 `static_multimap_task_doc.md`；
2. cuCollections `include/cuco/static_multimap.cuh`；
3. ops-collections 的 `StaticMap`、`OpenAddressingImpl`、存储和探测组件；
4. 任务随附的 1086 个功能用例和 32 个性能用例。

## 背景介绍

`static_multimap` 与已有 `StaticMap` 的关键区别是重复键语义。每次成功插入都必须占用独立槽位，即使 key 和 value 完全相同也不能去重。查询接口分为两类：

- `Contains`、`Find` 在第一个匹配处即可返回；
- `Count`、`Retrieve` 必须继续扫描完整探测链，得到该 key 的全部副本。

容器不支持删除，因此开放寻址探测序列中不存在删除产生的空洞。读路径可以在遇到第一个空槽时安全终止。

# 需求分析（required）

## 需求描述

实现 `StaticMultimap<Key, Value>` 纯头文件容器，支持：

- `Create`、`Destroy`、`Clear`；
- `Insert`、`InsertIf`；
- `Contains`、`ContainsIf`；
- `Find`、`FindIf`；
- `Count`、`Retrieve`、`RetrieveAll`；
- 辅助接口 `Size`、`Capacity`、`Data`。

Key 和 Value 支持 I32/I32、I64/I64。同一 key 可关联多个 value，重复 pair 也必须逐项保存。

## 需求拆解

1. **接口语义**：与 cuCollections 对应接口一致，Device 迭代器使用“首地址 + 元素数量”表达；
2. **并发正确性**：同槽竞争由 CAS 仲裁，失败线程继续探测，不得把同 key 判为重复失败；
3. **边界处理**：支持空输入、表满、重复 key、重复查询、全命中和全未命中；
4. **精度**：整数结果精确一致，检索结果按键值对多重集比较；
5. **性能**：全部 32 个性能用例达到标杆性能的 0.4 倍；
6. **工程结构**：新增代码限定在 `static_multimap` 独立目录，避免改变 `StaticMap` 和 `StaticSet` 行为。

# 详细设计（required）

## 算子分析

### 数学语义

设容器状态为键值对多重集 `M`：

- `Insert(P)`：`M = M ⊎ P`，表满时未写入元素计为失败；
- `Contains(k)`：判断是否存在 `(k, v) ∈ M`；
- `Find(k)`：返回任意一个匹配 value，未命中返回 `emptyValue`；
- `Count(K)`：返回 `Σ(k ∈ K) multiplicity(M, k)`，重复查询重复计数；
- `Retrieve(K)`：输出所有查询 key 对应的匹配 pair；
- `RetrieveAll()`：输出 `M` 中全部 pair；
- `Clear()`：将 `M` 重置为空集，容量不变。

检索输出顺序不构成接口契约，验收测试排序后按多重集比较。

### 数据类型与布局

| 项目 | 设计 |
| --- | --- |
| Key / Value | `int32_t`、`int64_t` |
| 槽位 | `Pair<Key, Value>` AoS 布局 |
| 空槽 | 构造参数 `Pair{emptyKey, emptyValue}` |
| stencil | 模板化，验收使用 `uint32_t` |
| Contains 输出 | `bool`/`unsigned char` |
| 计数 | Device 侧 `uint32_t`，Host 返回 `SizeType` |
| 探测 | `LinearProbing<murmurhash3_32<Key>>` |
| 桶大小 | 默认 5，与已有 `StaticMap` 一致 |

请求容量向上取整到桶大小的倍数。`capacity == 0` 显式抛出 `std::invalid_argument`。

## 算子实现

### Host 侧设计

实现分为三层：

1. `StaticMultimap`：公共 RAII 门面和接口转发；
2. `StaticMultimapImpl`：复用 `OpenAddressingImpl` 的 Device 存储、空槽初始化、容量和清空逻辑，负责参数校验、Kernel 发射、流同步和标量结果回读；
3. `StaticMultimapRef` 与专用 SIMT Kernel：实现多值插入、全链计数和检索。

`StaticMultimapImpl` 继承 `OpenAddressingImpl`，只访问其 protected 存储与哨兵，不修改共享开放寻址实现。现有容器不会产生行为变化。

Host 发射核数通过 `PlatformAscendCManager::GetCoreNumAiv()` 获取。同步接口在返回前调用 `aclrtSynchronizeStream`；异步变体由调用方负责输入输出生命周期。

### Kernel 侧设计

#### Insert / InsertIf

每个 SIMT 线程通过 grid-stride 循环处理输入：

1. 根据 key 计算初始桶；
2. 桶内逐槽检查；
3. 已占用槽位无论 key 是否相同都继续探测；
4. 空槽通过 CAS 尝试占位；
5. CAS 成功后计入成功数量；并发失败后继续探测；
6. 完整回环仍未插入则计入失败。

线程在寄存器中累计成功/失败数量，线程结束时分别只进行一次全局原子加，避免每个输入元素争用计数器。I32 pair 使用 8B 整槽 CAS；I64 pair 使用 8B key CAS，成功线程随后写 value。

#### Contains / Find

沿 key 的探测序列扫描：

- 命中时立即返回；
- 遇空槽说明后续不可能存在该 key，立即返回未命中；
- 表满场景通过回到初始桶终止。

`Find` 返回探测序列中的第一个匹配 value；对于同一稳定容器状态结果可重复。

#### Count

每个查询 key 扫描到首个空槽，累计所有匹配槽。每线程先在寄存器汇总自己处理的查询，再对全局结果做一次原子加。重复查询会按调用次数重复计数。

#### Retrieve

采用线程级两遍 compact：

1. 第一遍对当前线程负责的查询执行 `Count`，得到线程局部输出数量；
2. 每线程通过一次原子加预留连续输出区间；
3. 第二遍重新扫描相同查询，将匹配 pair 顺序写入该线程的独占区间。

原子操作数量从结果规模 O(N) 降为活跃线程规模 O(T)，输出无越界写。

#### RetrieveAll

每线程按 grid-stride 扫描表：

1. 第一遍统计本线程命中的非空槽数；
2. 一次原子加预留输出区间；
3. 第二遍将非空 pair 写入独占区间。

该方案仅使用标量级工作空间，不复制输入或哈希表。

#### Size / Clear

容器维护一个 Device 侧 32 位 size 标量。Insert Kernel 以线程聚合方式更新；`Size(stream)` 同步后回读；`Clear` 清空表后将 size 重置为 0。相较每次 `Size` 全表扫描，该方案不会影响查询复杂度，并把插入计数原子数限制在活跃线程数。

## 支持硬件

| 芯片版本 | 支持情况 |
| --- | --- |
| Atlas 950 系列（DAV-3510） | 支持 |
| 后续支持 SIMT 的昇腾系列 | 具备扩展能力 |

## 约束限制

1. `capacity > 0`；
2. 有效 key 不得等于 `emptyKey`；
3. I64 宽槽使用 `emptyValue` 识别 payload 写入完成，因此有效 value 不应等于 `emptyValue`；
4. 不支持动态扩容和单元素删除；
5. 同一容器的写操作与读操作必须通过 stream 顺序或显式同步避免数据竞争；
6. 输出容量不足时禁止越界写并返回错误；
7. 单次容量和输入数量受当前 Kernel 32 位索引范围限制。

# 可维可测分析（required）

## 精度标准/性能标准

| 项目 | 标准 |
| --- | --- |
| 功能 | 官方 1086 个功能用例全部通过 |
| 精度 | I32/I64 结果与 CPU 多重映射 golden 精确一致 |
| 性能 | 官方 32 个性能用例全部达到标杆的 0.4 倍 |
| 内存 | 除静态表、输出和常数级计数器外，不产生输入规模级副本 |

性能重点：

- Insert 热路径每个成功元素一次槽位 CAS，计数按线程聚合；
- Contains/Find 首次命中或空槽即退出；
- Retrieve/RetrieveAll 使用线程级区间预留，避免逐结果全局原子；
- 默认负载率 0.5 时线性探测具有良好的连续访存局部性。

## 验证方案

1. **构建检查**：Bisheng 编译器，`--npu-arch=dav-3510`；
2. **功能验证**：运行任务包提供的全部 Catch2 用例，覆盖 I32/I64、容量、占用率、multiplicity、匹配率、空输入和表满；
3. **性能验证**：运行 8 个性能程序展开的 32 个 case，记录平均时延并计算相对标杆倍数；
4. **内存验证**：记录 1e8 容量下创建、插入、检索前后的 Device 内存；
5. **稳定性验证**：相同输入重复执行，比较 Size、Count 和排序后的 Retrieve/RetrieveAll 输出。

## 兼容性分析

这是新增容器，不改变已有 API。实现位于：

```text
include/static_multimap.h
include/static_multimap_ref.h
include/detail/static_multimap/
tests/static_multimap/
tests/performance/static_multimap/
```

共享 `OpenAddressingImpl` 未修改，因此 `StaticMap`、`StaticSet`、`DynamicMap` 等既有容器不存在行为回归。

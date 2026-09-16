# static_multimap 容器设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 9月社区任务-mulimap容器开发(950)：static_multimap 容器开发 |
| 作者账号 | forge001 |
| 参考实现 | NVIDIA cuCollections `static_multimap` |
| 交付目标仓 | cann/ops-collections（纯头文件容器工程模式） |
| 目标芯片 | Atlas 950 系列产品及后续支持 SIMT 能力的昇腾系列产品 |
| CANN 版本 | 9.0.0-beta.2 及以上 |
| 文档状态 | 设计评审稿；性能门槛对标任务书给定标杆，实测数字以 Atlas 950 真机验收为准 |

# 需求背景（required）

## 需求来源

社区任务要求参考 NVIDIA cuCollections 的 `static_multimap`，在昇腾 NPU 上使用 Ascend C 与 C++ 实现功能一致的静态多值键值映射容器，按 ops-collections 的纯头文件容器工程模式组织代码，完成设计、开发、测试全流程，验收通过后提交至昇腾算子开源仓。

参考资料：

1. 任务书 `static_multimap_task_doc.md`：功能范围、接口定义、性能基线、交付目录与验收要求；
2. cuCollections 参考实现：`include/cuco/static_multimap.cuh`、`static_multimap_ref.cuh` 及开放寻址实现层；
3. ops-collections 复用基线：`include/static_map.h`、`include/static_map_ref.h`、`include/detail/open_addressing/`、`include/detail/storages/`；
4. 设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`。

## 背景介绍

### cuCollections static_multimap 语义分析

static_multimap 是容量创建时固定的开放寻址哈希容器，核心语义：

1. **多值不去重**：同一 Key 可关联多个 Value；完全相同的键值对亦逐条保留，各占独立槽位；
2. **插入**：探测序列上遇空槽以原子 CAS 占位，遇同 Key 槽不早退（多值语义与唯一键容器的核心差异）；表满时逐条判失败并返回失败数量；
3. **查询**：Contains 布尔判定；Find 返回任一匹配 Value；Count 沿探测序列全枚举计数；
4. **检索**：Retrieve 输出每个查询 Key 的全部匹配键值对；RetrieveAll 输出容器全部键值对；
5. **规模统计**：Size 通过全表扫描统计，不维护插入计数器；
6. **输出顺序**：官方测试对 Retrieve/RetrieveAll 结果做排序归一化比较，不依赖输出顺序。

### ops-collections 现状分析

ops-collections 已有 static_map（唯一键静态容器）的完整 SIMT 工程先例，分层结构为「Host 门面类 → Host 编排层 → 存储层 → 设备侧 Ref → 两级 SIMT Kernel」。可直接复用的资产包括：探测方案（LinearProbing/DoubleHashing 桶粒度迭代器）、Pair 槽类型与存储分配、CAS 原语封装（4/8B 直接 CAS、宽槽 key-CAS + 依赖写）、判等语境切换组件、测试基建（Catch2 功能测试框架与性能测试框架）。

static_map 为唯一键语义（插入遇 EQUAL 即返回），多值协议（EQUAL 不早退、全枚举查询、Retrieve/RetrieveAll 接口）无对应物，需新写；新写部分收敛在独立的 `detail/static_multimap/` 目录，不改动仓内共享文件。

### static_multimap 功能分析

任务需提供 12 个对外接口（大驼峰命名，入参顺序与 cuCollections 对应接口一致，设备迭代器以「首地址 + 元素个数」替代）：

| # | 接口 | 签名要点 | 返回值 |
| --- | --- | --- | --- |
| 1 | Create | 构造函数 `StaticMultimap(capacity, emptyKey, emptyValue, stream)`，RAII | 实际容量 ≥ 请求容量 |
| 2 | Destroy | 析构函数，RAII 释放 | 无 |
| 3 | Clear | `Clear(stream)` | 无；清空后 Size==0，可复用 |
| 4 | Insert | `Insert(values, valueNum, stream)` | 插入失败的键值对数量 |
| 5 | InsertIf | `InsertIf(values, stencil, valueNum, stream)`，谓词真才插入 | 谓词真且插入失败的数量 |
| 6 | Contains | `Contains(keys, output, numInputs, stream)` | 无；output 元素非 0 即存在 |
| 7 | ContainsIf | `ContainsIf(keys, stencil, output, numInputs, stream)` | 无；谓词假一律写 0 |
| 8 | Find | `Find(keys, outputValues, numInputs, stream)` | 无；命中写任一匹配 Value，未命中写 emptyValue |
| 9 | FindIf | `FindIf(keys, stencil, outputValues, numInputs, stream)` | 无；谓词假一律写 emptyValue |
| 10 | Count | `Count(keys, numInputs, stream)` | 全部查询键出现次数总和 |
| 11 | Retrieve | `Retrieve(probeKeys, numInputs, probeOut, matchOut, outputCapacity, stream)` | 实际写入匹配对数 |
| 12 | RetrieveAll | `RetrieveAll(keysOut, valuesOut, outputCapacity, stream)` | 实际输出对数（= Size） |

辅助查询方法（测试必需）：`Capacity()`（编译期常量，返回取整后槽位数）、`Size(stream)`（当前元素数）。

# 需求分析（required）

## 需求描述

使用 Ascend C SIMT 编程范式与 C++ 实现 static_multimap 静态多值键值映射容器：Key 与 Value 均支持 I32、I64（验收场景同 dtype），同一 Key 可关联多个 Value；查询、计数及检索结果与参考实现语义一致，相同输入重复执行结果确定且可重复；全部 32 个性能用例达到标杆性能的 0.4 倍及以上（等价时延不超过标杆时延 / 0.4）；代码按 ops-collections 目录规范提交。

## 需求拆解

1. **功能**：实现 12 个接口的完整多值语义——插入不去重、查询全枚举、检索输出全部匹配对、容量边界与空输入边界行为正确；
2. **数据类型**：Key/Value 支持 `int32_t`、`int64_t`；stencil 为 `uint32_t`；Contains 系输出为 `unsigned char`；谓词为设备侧可调用仿函数；
3. **参数校验与异常**：capacity==0 构造抛异常；空指针与零输入按接口契约处理（详见约束限制节）；检索输出空间不足须在写入前报错、不得越界；
4. **精度**：整数结果精确比较；与参考实现的容器内容、输出结果一致（多重集语义）；相同输入重复执行结果一致；
5. **性能**：32 个性能用例全部 ≥ 0.4× 标杆；
6. **内存**：除容器静态存储、输出空间与必要工作空间外，不产生与输入规模线性重复的额外 Device 内存拷贝；容量向上取整放大系数克制；
7. **工程化**：纯头文件容器工程模式（`include/static_multimap.h`、`include/static_multimap_ref.h`、`include/detail/static_multimap/`）；功能测试置于 `tests/static_multimap/`，性能测试置于 `tests/performance/static_multimap/`；CMake 3.16+、Catch2 v3.5.4+。

# 详细设计（required）

## 算子分析

### 数学公式（容器语义形式化）

静态多值映射的语义可形式化为「多重集」操作：

- 容器状态 = 全体槽位上 (Key, Value) 对构成的多重集 `M`，`|M| ≤ Capacity`；
- `Insert(P)`：`M ← M ⊎ P`（多重集并，不去重；表满时溢出部分失败并计数）；
- `Contains(k)`：`k ∈ keys(M)`；
- `Find(k)`：返回 `M` 中任一 `(k, v)` 的 `v`（本设计具体化为探测序列上第一个匹配，同一容器状态下结果确定）；
- `Count(k)`：`M` 中 key 等于 `k` 的元素个数；`Count(keys[])` 返回逐键计数之和（重复查询逐次计入）；
- `Retrieve(K)`：输出 `⊎_{k∈K} {(k, v) ∈ M}`（全部匹配对）；
- `RetrieveAll()`：输出 `M`（顺序任意，本设计输出顺序确定）；
- `Clear()`：`M ← ∅`，容量与哨兵不变，可复用。

**确定性口径**（任务书「相同输入重复执行时结果一致」）：布尔结果、计数与返回值、容器驻留键值对多重集、Retrieve/RetrieveAll 的输出内容与顺序均为确定结果；并发插入竞争决定的物理槽位布局不属于「结果」，允许逐次不同（参考实现同样不保证，官方测试已做排序归一化断言）。

### 支持数据类型

| 项 | 值 |
| --- | --- |
| Key / Value | `int32_t`、`int64_t`；验收场景 Key 与 Value 同 dtype（I32/I32、I64/I64 两组） |
| 槽类型 | `Pair<Key, Value>` AoS 交错布局；I32 槽 8B（8B 对齐），I64 槽 16B（16B 对齐） |
| 空槽哨兵 | 构造传入的 `Pair{emptyKey, emptyValue}`（官方测试固定 `numeric_limits::lowest()`） |
| stencil | `uint32_t`（模板参数化，不限于 uint32） |
| Contains 系输出 | `unsigned char`（非 0 = 存在） |
| 计数/返回类型 | SizeType（无符号整数） |
| 谓词 | 设备侧可调用仿函数 `bool operator()(StencilT)` |

### 支持形状（容量与规模语义）

- **静态容量**：构造时固定，不可增长；实际容量 = 请求容量向上取整到桶大小（BucketSize=2）的倍数，`Capacity() ≥ 请求容量`，放大不超过 BucketSize−1 个槽；性能口径容量 2×10⁸ 槽时取整零膨胀；
- **规模上界**：`capacity ∈ [1, 2³²)`；单次批量 `numInputs ∈ [0, 容量]`；
- **边界场景**：空输入、重复 Key、不同 multiplicity、命中/未命中、容量边界（表满溢出、capacity==0）均为合法泛化场景。

## 算子实现

### 实现方案

总体技术路线：**开放寻址哈希表 + Pair 直存 AoS 槽 + LinearProbing 线性探测（murmurhash3_32 哈希，桶大小 2）+ 两级 SIMT Kernel**；读路径全接口零全局内存原子操作，写路径槽级 CAS。完全对齐 ops-collections static_map 的工程形态，复用其存储、探测、CAS 与测试基建，多值协议新写并收敛于独立目录。

#### 3.2.1 host 侧设计：

**分层结构**：`StaticMultimap`（Host 门面：RAII 生命周期、参数校验、接口转发）→ `OpenAddressingMultimapImpl`（Host 编排：容量取整、存储分配、Kernel 发射、流同步、检索类接口的两遍编排与前缀和）→ 存储层（`BucketStorage` 表分配 / `ArgStorage` 参数存储 H2D/D2H）。Host 与设备侧经轻量 Ref 类型解耦，Kernel 参数中表/输入/输出指针以 `__gm__ uint8_t*` 传参后按实际类型解释（仓内既有约定）。

##### 1. 分核策略：

- Kernel 发射网格 = AIV 核数，**运行时查询平台信息获取，禁止写死核数**；
- 线程组织为 grid-stride 循环：每线程按全局线程索引跨批元素推进（`i = 全局线程号; i < N; i += 总线程数`），任意规模输入自动均摊到全部线程，无需按输入规模静态切分；
- 检索类接口（Retrieve/RetrieveAll）以块为输出区间单位：每块先统计本块匹配数，Host 侧对块计数做前缀和得到各块输出基址，块间输出区间不重叠、不越界。

##### 2. 数据分块和内存优化策略：

- **槽布局选 AoS（Pair 交错）而非 SoA（键值分离双数组）**：查询路径每槽一次读出键值（SoA 下仅判键的查询可只读键数组、流量减半，此为已知取舍）；插入路径一次写完成（SoA 需两次独立写、值可见性协议更复杂）；与仓内存储/Clear/判等组件假设一致，零改动复用。SoA 留作读带宽不达标时的后备布局；
- **容量取整克制**：LinearProbing 走桶倍数取整（放大 ≤ 1 槽/桶），不启用会膨胀到质数的 DoubleHashing 取整路径；
- **工作空间全部元数据级**：插入失败计数器 1 元素；Count/Size 的每线程部分和缓冲（总线程数 × 4B，百 KB 量级）；Retrieve 的逐查询计数数组（4B/查询，一次性统计缓冲，非输入数据拷贝）与块基址数组（块数 × 4B）。工作空间由容器按需分配、跨调用复用，避免重复分配抖动；
- **内存预算**：性能口径容器本体最大 2×10⁸ 槽 × 16B = 3.2 GB（I64），加输入与输出合计 < 5 GB，远低于 Atlas 950 系列 HBM 容量；
- **Clear 大表向量快路径**：I32 槽且容量大于阈值时走向量 Kernel（UB 内构造哨兵 tile + 批量 DataCopy 下刷，tile 大小运行时按 UB 容量推导）；小表或 I64 宽槽走 SIMT 逐槽路径。

##### 3. 路径分派策略（SIMT Kernel 直调，无 TilingKey）：

本容器为纯头文件 SIMT 工程，Kernel 由 Host 编排层直接发射，不存在 TilingData/TilingKey 机制。路径分派由三层承载：

- **编译期分派（槽宽度）**：插入的原子原语按 `sizeof(Pair)` 编译期选择——槽 ≤ 8B（I32/I32）整槽打包为 8B 单次 CAS，键值原子性同时生效；槽 > 8B（I64/I64）走 key 分量 8B CAS + 赢得槽位后 value 普通依赖写（每成功插入恰 1 次原子操作）；
- **运行时分派（Clear 双路径）**：按「槽宽 ≤ 8B 且容量大于核数 × 基本块阈值」选择向量快路径或 SIMT 路径；向量快路径增设**哨兵位级一致护栏**（emptyKey 与 emptyValue 位模式一致才准入，否则降级 SIMT 逐槽路径写真实 Pair 哨兵，保证任意合法哨兵组合下正确性）；
- **模板实例化分派（谓词版接口）**：Insert 与 InsertIf、Contains 与 ContainsIf 等共用同一 Kernel，以谓词仿函数模板参数实例化；无谓词版以常真谓词实例化并对 stencil 空指针加守卫，语义等价。

#### 3.2.2 kernel 侧设计：

全部 Kernel 采用两级结构：`__global__ [aicore]` 入口函数 + `VF_CALL` 展开的 SIMT 设备函数（`LAUNCH_BOUND(1024)`），设备函数承载探测循环、CAS、判等与输出写。

**写路径（Insert / InsertIf）——多值插入协议**：

1. 探测方案给出初始桶索引并记录起点；外层逐桶推进，桶内逐槽判等；
2. **遇同 Key 槽（EQUAL）不早退**，继续找空槽——同 Key 多值各占独立槽位，可分布在不同桶（与唯一键容器的核心差异）；
3. 遇空槽发起 CAS 占位：CAS 成功即完成（I64 另含 value 依赖写）；CAS 失败且旧值为同 Key（并发同键抢占）继续找下一空槽；失败且旧值为他 Key 扫下一槽；
4. 整圈回到探测起点判表满，该条目计失败；Host 回读失败计数返回；
5. 谓词路径求值次序固定：读 stencil → 求谓词 → 谓词真才读键值对并进入探测插入，谓词假零开销跳过；
6. **失败计数聚合**：每线程寄存器累加失败数，循环结束非零才发射一次全局原子加——全成功批次零计数原子，计数原子次数上界为活跃线程数，与输入规模无关。

**并发竞争消解**：槽级 CAS 原子性保证同槽竞争一胜一败，败者换槽重试直至成功或表满；同 Key 并发插入互不阻塞互不覆盖（不丢不重）。I64 宽槽存在「key 已写、value 未写」短窗口瞬时态：写者赢得 key CAS 后必然完成 value 写；需要读 value 的接口（Find/Retrieve）命中时若 value 仍为哨兵则自旋等待 payload 就绪（有界等待），只判 key 的接口（Contains/Count）不受影响。

**读路径（Contains / ContainsIf / Find / FindIf / Count / Retrieve / RetrieveAll / Size）——零全局原子设计**：

全部读接口共享探测循环骨架：桶粒度迭代推进、逐槽判等、**空槽早停**（开放寻址不变式：同 Key 的全部副本必然位于其探测序列上首个空槽之前）、整圈回环兜底（表满防御）。多值枚举协议按接口分形：

- **Contains / Find**：首个匹配即得结果（EQUAL 早退）；Find 多值「任选一个」具体化为探测序第一个匹配，同一容器状态下结果确定；未命中写 emptyValue 哨兵；
- **Count**：沿探测序列全枚举累加（空槽早停）——多值匹配分散在探测序列上必须扫全，不能退化为布尔判定；每线程寄存器累加后写各自独占的缓冲槽，Host 侧回读求和，**零原子**（避免每查询一次全局原子加在高重复输入下于单一地址串行化）；
- **Retrieve / RetrieveAll：两遍 compact 方案（核心设计）**——Pass1 Kernel 统计（Retrieve 按查询维逐键计数、RetrieveAll 按槽维数非空槽）并产出块级部分和；Host 回读块部分和做前缀和得到各块输出基址，同时完成输出容量校验（空间不足在任何写入发生前报错，越界被结构性排除）；Pass2 Kernel 重扫并按「块基址 + 块内前缀」写输出。两遍使用相同的子段划分与探测序列，容器在两次发射之间无并发修改（接口同步语义保证），写入区间恰好覆盖、不重叠、不越界。**全程零全局原子**，且输出顺序确定（Retrieve 按查询序、RetrieveAll 按槽序）；
- **Size**：不维护插入计数器，每次调用全表扫描统计非空槽（每线程部分和 + Host 求和）。取舍：避免插入热路径增加原子与计数维护、避免失败回滚路径的计数一致性问题（与参考实现行为一致）；Size 不在 32 个性能用例之内，全表扫描代价无验收面影响；
- 块内前缀归约不依赖块内同步原语：基线采用单线程串行累加写回独占前缀（每块一次性开销），树形归约留作真机调优项。

**生命周期（Create / Clear / Destroy）**：

- **Create**：① capacity==0 显式抛 `std::invalid_argument`（框架取整函数默认不抛，此为对框架行为的必要覆写）；② 容量向上取整到桶大小倍数；③ 分配单块连续 Device 内存；④ 空态初始化复用 Clear 双路径（构造完成即 `Size == 0`）。构造后不变量：`Capacity() ≥ 请求容量`；
- **Clear**：整表重填空槽哨兵，终态与初始空态同构——幂等、可复用、零原子；
- **Destroy**：RAII 默认析构链（unique_ptr → 自定义删除器 → aclrtFree），全接口同步语义保证析构时无未完成异步操作；拷贝删除、移动默认，支持容器移动转入（性能 harness 依赖）。

**同步语义**：全部 12 个接口为同步（阻塞）接口——提交 Kernel 后等待流完成再返回（官方性能 harness 按此口径计时）。Retrieve/RetrieveAll 内部两次 Kernel 发射在同一流上顺序提交，接口对外语义不变。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950 系列产品（SIMT 能力，交叉编译目标 Ascend950PR） | √ |

## 算子约束限制

1. **保留值约束**：用户 Key 不得等于构造传入的 emptyKey，Value 不得等于 emptyValue（参考实现同款约束，行为未定义）；
2. **验收 dtype**：本期验收 Key 与 Value 同 dtype（I32/I32、I64/I64）；浮点类型、混合 Key/Value dtype 不在本期范围；
3. **静态容量**：不支持动态扩容；不支持单条删除（参考实现同款限制）；
4. **同步接口**：全部接口阻塞至流完成，不提供异步变体（首版）；
5. **谓词约束**：谓词须为设备侧可调用仿函数，仅作用于 stencil 元素；
6. **异常契约**：capacity==0 构造抛 `std::invalid_argument`；Insert 输入空指针且数量非零返回全部失败数（不崩溃）；InsertIf stencil 空指针且数量非零抛 `std::invalid_argument`；零输入为合法 no-op；Retrieve/RetrieveAll 输出空间不足在写入前抛 `std::runtime_error`；
7. **CANN 版本**：9.0.0-beta.2 及以上；构建需 CMake 3.16+、Catch2 v3.5.4+。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 对标 cuCollections static_multimap：I32/I64 的 Insert/InsertIf 状态变化及全部查询接口输出与参考实现一致；相同输入重复执行结果一致。整数结果精确比较；官方测试对检索类输出做排序归一化比较 | 任务书 3.2 |
| 性能标准 | 32 个性能用例全部 ≥ 0.4× 标杆（等价时延 ≤ 标杆时延 / 0.4）；不达标用例须给出合理解释 | 任务书 3.3 |

性能用例与标杆门槛表（标杆 = cuCollections static_multimap，任务书给定；门槛时延 = 标杆时延 / 0.4）：

| 接口 | dtype | 参数组合 | 标杆时延 (ms) | 门槛时延 (ms) |
| --- | --- | --- | --- | --- |
| create | I32/I32 | Capacity=1e8 | 7.635412 | 19.088530 |
| create | I64/I64 | Capacity=1e8 | 15.419803 | 38.549508 |
| destroy | I32/I32 | Capacity=1e8 | 9.038761 | 22.596903 |
| destroy | I64/I64 | Capacity=1e8 | 17.766071 | 44.415178 |
| insert | I32/I32 | NumInputs=1e8, Occ=0.5, Mult=1 | 9.984867 | 24.962168 |
| insert | I64/I64 | 同上 | 13.994128 | 34.985320 |
| retrieveAll | I32/I32 | 同上 | 1.469109 | 3.672773 |
| retrieveAll | I64/I64 | 同上 | 3.421753 | 8.554383 |
| contains | I32/I32 | 1e8, Occ=0.5, Mult=1, MR=0.1/0.5/1 | 7.175591 / 6.876158 / 6.272652 | 17.938978 / 17.190395 / 15.681630 |
| contains | I64/I64 | 同上 | 9.062888 / 8.907699 / 8.698283 | 22.657220 / 22.269248 / 21.745708 |
| find | I32/I32 | 同上 | 7.842479 / 7.674542 / 7.065370 | 19.606198 / 19.186355 / 17.663425 |
| find | I64/I64 | 同上 | 9.708089 / 9.566873 / 9.300692 | 24.270223 / 23.917183 / 23.251730 |
| retrieve | I32/I32 | 同上 | 12.055540 / 13.038496 / 13.475749 | 30.138850 / 32.596240 / 33.689373 |
| retrieve | I64/I64 | 同上 | 14.419790 / 15.764821 / 16.426197 | 36.049475 / 39.412053 / 41.065493 |
| count | I32/I32 | 同上 | 5.868191 / 5.974523 / 6.097573 | 14.670478 / 14.936308 / 15.243933 |
| count | I64/I64 | 同上 | 8.712889 / 8.838063 / 8.985895 | 21.782223 / 22.095158 / 22.464738 |

（MR = MatchingRate，匹配率 0.1/0.5/1 三档；共 32 个用例，与任务书 3.3 性能基线逐行对应。）

**设计级带宽预算**（均匀哈希假设下的估算，真机实测前不定论）：查询类接口（contains/find/count/retrieve）需求 ≤ 250 GB/s，对设备峰值带宽余量充足；**retrieveAll 为约束最紧用例**——最小流量下界（单遍表扫描 + 输出写）I32 需 ~654 GB/s，两遍 compact 方案 ~1090 GB/s（约为峰值带宽的 68%~78%）；insert 次之（每秒约 4×10⁹ 次 8B CAS 的原子吞吐需求）。两者的缓解链：

| 风险用例 | 一级缓解 | 二级缓解 | 三级缓解 |
| --- | --- | --- | --- |
| retrieveAll | 块级原子单遍 compact（每块一次原子区间分配，扫描降回单遍） | SoA 双数组布局（compact 只扫 key 数组，value 按输出位置聚集读取） | 按任务书条款提交合理解释（单遍下界已达峰值的 41%~47%） |
| insert | 桶大小 2→4（模板参数切换，同桶候选槽翻倍、冲突重试减少） | 线性探测→双哈希（分散聚集竞争，容量取整放大需权衡） | 真机数据定位原子吞吐与带宽瓶颈后逐项迭代 |

## 验证方案

- **用例体系**：官方 1086 个功能用例（12 接口 × dtype 实例 × GENERATE 参数 × SECTION 分支展开）与 32 个性能用例（8 个 harness）逐字节原样接入仓内测试目录，官方测试代码即接口契约，不做任何适配修改；
- **分档验证**：开发期在交叉编译（Ascend950PR 目标）+ CPU 仿真环境先行验证——读路径全接口零全局原子，可在仿真侧做较大规模的语义验证（表状态以无原子预置方式构造：Host 侧按与 Kernel 完全一致的探测规则模拟落位后直写槽位，保证预置表是真实插入的合法终态之一）；写路径原子操作在仿真侧做小规模语义验证（CAS 单次语义、identical 保留、并发竞争、表满失败、空指针边界）；
- **真机验收**：Atlas 950 真机执行全量 1086 功能用例与 32 性能用例；性能数字以真机为唯一正式来源，仿真结果仅证明功能语义、不具性能代表性；性能采样在独占卡上进行，测量口径（NumInputs/Occupancy/Multiplicity/容量推导）随结果回显，不依赖环境变量静默默认值；
- **自测报告**：覆盖全部用例，含入参、结果对比、执行日志/截图与性能数据，随交付件提交。

## 兼容性分析

新容器，不涉及既有算子兼容性。工程边界：多值协议实现收敛于独立的 `include/detail/static_multimap/` 目录与独立设备侧实现文件，**不改动仓内被 static_map/static_set 共享的任何文件**，对既有容器零回归影响；公共测试基建（Catch2 框架、性能测试框架、CMake glob 模式）按仓内既有模式接入。测试公共头引用的另一赛题容器头（static_multiset）在上库时与仓库方确认依赖收敛方式，自测期的最小前向声明 stub 不进入交付 PR。

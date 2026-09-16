# Ascend C Basic API 指针化扩展设计文档（DMA · 数据搬运 / 缓存分册）

> 任务编号：09-42
> 团队：fengyemao
> 版本：v0.1（开发前设计）
> 目标产品：Ascend 950 系列；任务书要求 CANN 9.0.0～9.1.0

## 一、需求背景

### 1.1 需求来源

本设计对应“9月社区任务-AscendC Basic_API优化实现（搬运类接口扩展）”的 DMA 分册。任务要求在保持现有 `LocalTensor` / `GlobalTensor` 调用兼容的前提下，让开发者可以使用带硬件地址空间的 `__gm__`、`__ubuf__`、`__cbuf__` 等指针直接调用现有数据搬运和缓存 API。

本任务是编译期入参适配，不新增搬运指令，不扩展 VECTOR/CUBE 分册，也不改变已有 `*Impl` 的数值、布局、缓存或同步语义。官方表格列出 5 个 API 名称、44 条重载行：`DataCopy` 27 条、`DataCopyPad` 11 条、`DataCopyL1ToUB` 2 条、`DataCacheCleanAndInvalid` 3 条、`DataCachePreload` 1 条。

任务书文件名为 9 月任务，但正文标题仍写“8 月社区任务”，该文字差异保留为待官方确认项。

### 1.2 当前基线和范围

对照的 asc-devkit 基线为 `upstream/master` 提交 `626c945f943983394d9c6e802021e1f2691ebee6`。公开声明位于 `include/basic_api/kernel_operator_data_copy_intf.h` 与 `include/basic_api/kernel_operator_cache_intf.h`，实现位于对应 `impl/basic_api` 接口分发层和架构目录。

源码按文本有 45 个声明位置。除官方 44 行外，还存在受旧架构宏保护的 `DataCopy(..., Nz2DnParamsFull&)` 声明；它不在任务清单内，本设计不扩展。

## 二、需求分析

### 2.1 目标与非目标

| 目标 | 设计要求 |
| --- | --- |
| Pointer 调用 | 合法地址空间裸指针与 Tensor 可独立混合传入同名 API |
| Tensor 兼容 | 两个操作数均为 Tensor 时继续解析原重载，保留原检查、模板和管线属性 |
| 底层复用 | Pointer 与 Tensor 最终调用同一条已有 `*Impl` 通路 |
| 资源语义 | 不引入随输入规模线性增长的 Device 临时拷贝或隐式同步 |
| 范围边界 | 只改官方 44 行对应的 DMA/Cache 接口，不修改 VECTOR/CUBE、Mmad、Fixpipe、LoadData |

### 2.2 地址空间与操作数

| 空间 | 典型指针类型 | 主要通路 |
| --- | --- | --- |
| GM | `__gm__ T*` | GM↔UB、GM↔L1、L0C→GM、Cache |
| UB | `__ubuf__ T*` | GM↔UB、UB→L1、L1→UB |
| L1 | `__cbuf__ T*` | GM→L1、UB→L1、L1→UB、CO12 |
| L0C | `__cc__ T*` | L0C→L1/GM 或已有 Enhanced 路由 |
| BiasTable | `__biasbuf__ T*` | L1→BiasTable 的已有路由 |
| FixpipeBuffer | `__fbuf__ T*` | 已有 L1→Fixpipe 路由 |

地址数值不能替代地址空间类型。`__ca__`、`__cb__`、Mmad、Fixpipe、LoadData 是样例依赖，不因此纳入本任务。错误的空间组合应在模板候选或已有位置检查处失败，不转换为 `void*` 或无类型整数后静默调用错误指令。

### 2.3 兼容性事实

- `LocalTensor::GetPhyAddr()` 在 Device 侧返回物理整数地址，在 CPU Debug 侧返回指针；`GlobalTensor::GetPhyAddr()` 返回带 `__gm__` 地址空间的指针。统一 helper 不能无条件地用一个 `auto` 返回类型抹平这两个语义。
- GlobalTensor 地址的高位可能编码 L2/cache hint。必须先提取 cache mode，再去除 hint；Pointer 没有 Tensor 句柄时不应默认 cache mode 为 0。
- Slice DataCopy 在 `shapeValue == 0` 时会从 Tensor ShapeInfo 回退。裸指针没有 ShapeInfo，Pointer Slice 必须要求每一维显式 shapeValue。
- Tensor 句柄的 `const` 不等于底层存储只读。可写目的指针必须是非 const pointee；源 const 是否可用由具体只读通路决定，不能全局 `const_cast`。
- `DataCachePreload` 的模板参数是 offset 类型，当前实现约束为 `int16_t`/`int64_t`；源操作数固定为 GM `uint64_t`，不能把该模板参数当作元素类型。

## 三、详细设计

### 3.1 统一适配结构

```text
Tensor/Tensor  -> 原有重载、position/Shape/cache/容量检查 -----------------+
Pointer/混合  -> 地址空间、元素类型、显式元数据约束 -----------------------+--> 原 route
                                                                           -> GetUnderlyingPtr
                                                                           -> 原 *Impl
                                                                           -> 原同步约定
```
保留旧 Tensor 入口，新增只在至少一个操作数为合法 Pointer/数组引用时启用的候选。适配层只传递地址和原参数，不分配 staging buffer、不复制输入、不插入 barrier。Pointer 的容量、位置、ShapeInfo 和生命周期由调用方按接口约束保证，并在文档中明确其能力边界。

### 3.2 GetUnderlyingPtr 设计

1. Pointer/数组引用保持原地址空间、元素类型和 pointee const；数组不能只用 `std::is_pointer` 判断。
2. Tensor 在确定 route/position 后取得原物理地址，同时保留 `PrimT`、ShapeInfo、cache hint 和 Tensor 诊断。不要先统一转换成无类型整数。
3. 单 dtype 行要求 Pointer pointee 与原 primitive 类型一致；双 dtype 行继续使用原 `PrimT<T>`/`PrimT<U>` SFINAE，拒绝任意新转换。
4. 目的指针拒绝 const pointee；普通 Host 指针、`void*`、函数指针、双重指针和 volatile 组合不作为通用候选。
5. Slice、NDDMA、PaddingMode、SmallC0、`subBlockId`、cache 枚举和其他非类型模板参数由原参数显式承载，不从地址猜测。

### 3.3 模板和重载消歧

| 接口族 | 原模板语义 | 兼容规则 |
| --- | --- | --- |
| DataCopy 单 dtype | `<T>` | count、Params、Slice、Enhanced 按参数族分开 |
| DataCopy 双 dtype | `<T,U,enable_if...>` | 原 `PrimT` 关系和支持转换不变 |
| ND2NZ/DN2NZ | `<T, enableSmallC0>` | bool 非类型参数、默认值和架构守卫保留 |
| NDDMA | `<T, dim, config>` | `config` 为 `const NdDmaConfig&` 非类型参数 |
| DataCopyPad | `<T, PaddingMode>` 或旧 `<T>` | Params/Ext、PadParams/PadExt 不合并 |
| DataCopyL1ToUB | `<T, subBlockId>` | `uint8_t` 参数、默认值和 Mix 1:2 限制保留 |
| DataCachePreload | `<OffsetType>` | offset 只能使用原实现支持的类型 |
| DataCacheCleanAndInvalid | `<T, CacheLine, DcciDst>` 或 `<T, CacheLine>` | 不给三参数形式擅自增加默认 `DcciDst` |

新增候选必须满足至少一个 Pointer/数组操作数和合法的源/目的空间组合。两个 Tensor 操作数不进入万能候选，从而保留旧显式 `<half>`、`<T,U>`、默认模板参数及 CPU Debug 行为。`__aicore__`、`__inout_pipe__`、`__in_pipe__`、`__out_pipe__` 按官方行逐项保留，不能统一补成 MTE2/MTE3。

### 3.4 路由和底层实现复用

适配层复用 `kernel_operator_data_copy_intf_impl.h` 的参数检查、Tensor position 分发、count 换算和诊断；复用各架构 `DataCopy*Impl`、Pad、NDDMA、CO12、Bias/Fixpipe 和 Cache 实现。Pointer 只改变操作数适配，不复制底层算法。

声明存在不等于 Ascend 950 可执行。当前 3510 代码中 L1→GM、UB→L0C、L1→L0C、L0C→UB 等部分路线是 unsupported；普通 DataCopy 的 L1→UB 分发也不同于专用 `DataCopyL1ToUB`。这些行保留用于兼容和不适用分类，不能为了凑 44 行而调用 Fixpipe 或新增搬运算法。

## 四、地址、布局、同步约束

### 4.1 对齐、stride、padding

- GM↔UB 连续搬运保留 GM 按元素字节对齐、UB 端 32B 对齐和原 count 不对齐取整行为。
- Params 的 blockCount、blockLen、stride 按每个方向的官方文档解释，严格区分元素、字节、32B、C0 和头到头单位。
- ND2NZ/DN2NZ 保留 n/d、矩阵 stride、SmallC0、b4/b8、cache hint 和目的不重叠约束。
- Slice 保留维度上限、显式 shapeValue、burstLen 关系、横向字节量和区间约束；Pointer 不隐式生成 ShapeInfo。
- NDDMA 保留 dim、loop stride、padding、nearest/b64、40 位地址跨度和 `NdDmaDci` 约定。
- DataCopyPad 按 GM→UB、UB→GM、GM→L1、UB→L1 分别保留 ExtParams 值域、左右 padding、Normal/Compact 和 dummy 区域语义。
- L1→UB 保留 Mix 1:2 与 `subBlockId` 0/1；CO12/Bias/Fixpipe 保留 L0C、Bias、Fixpipe 空间和 quant/relu/unitFlag 参数。

### 4.2 异步完成与生命周期

适配层不新增等待，也不删除既有等待。调用方负责 GM→UB 后的 V/MTE3、UB→L1 的跨核/SSBuffer、L1→UB 的 subBlock、L0C 写回、Cache 可见性和 Host stream 读回依赖。`DataCacheCleanAndInvalid` 不是跨核同步；`DataCachePreload` 只改变预取行为。首次调用、重复调用、改变输入、buffer 复用、合法 stream 切换、释放前完成和异步提前消费必须分别验证。

## 五、可维可测分析

### 5.1 验收口径

任务书要求原 Tensor 用例回归、Pointer/Tensor 同布局结果一致，并声明无额外性能指标。生态浮点精度标准的范围说明排除了搬运类，因此正式 comparator 需按纯位拷贝、随路转换和 Cache 分别确认：纯搬运比较有效字节、定义 padding 和 canary 的逐位结果；随路转换沿用原 API golden；Cache 检查作用域和可见性。本设计不放宽阈值、不替换 golden、不把编译成功当作 Device 通过。

### 5.2 测试矩阵

每个官方重载行 DMA-001～DMA-044 建立独立记录，覆盖：

| 维度 | 场景 |
| --- | --- |
| 调用形态 | Tensor/Tensor、Pointer/Pointer、Tensor/Pointer、Pointer/Tensor |
| 模板 | 自动推导、显式 dtype、SmallC0、PaddingMode、dim/config、subBlockId、CacheLine、DcciDst |
| 地址空间 | GM、UB、L1、L0C、Bias、Fixpipe 合法路线及错误路线负例 |
| 数据 | 每个 route 的原 SupportType/PrimT，含双 dtype SFINAE 真/假 |
| Shape/参数 | 1、32、1024、2048 的合法映射；ND/NZ/DN、Slice 1/2/8 维、NDDMA dim 1～5 |
| 边界 | count=0/1、尾块、不对齐、最大值及最大值+1、stride/padding 边界 |
| 语法 | 地址空间数组、衰减指针、const 源、const 目的负例、索引偏移、显式模板参数 |
| 生命周期 | 首调用、重复调用、更换输入、复用 buffer、异步提前消费、stream 同步、释放 |
| 别名 | 不重叠、相邻、同址、部分重叠；不承诺 memmove 语义 |

用户提供的六个样例共 25 个场景，可作为代表路线输入，但不能代替 44 行覆盖。样例中的比较器、版本和部分 padding 参数存在静态不一致，后续实现前需按官方版本重新核对。

## 六、风险与待确认问题

| 编号 | 问题 | 关闭条件 |
| --- | --- | --- |
| O-01 | 任务书文件名为9月、正文标题为8月 | 官方确认正式标题和活动月份 |
| O-02 | 44行含旧架构互斥声明，当前源码文本为45处，3510部分路线 unsupported | 官方确认950逐行运行/兼容/不适用分类 |
| O-03 | 任务要求 CANN 9.0～9.1，附件部分要求 CANN≥9.2 | 固定任务 SDK、样例和编译器版本 |
| O-04 | 浮点精度标准排除搬运类，任务仍引用；附件有多套 comparator | 官方指定纯搬运、随路转换、Cache comparator |
| O-05 | `GetPhyAddr` 的 CPU/Device 返回类型、地址空间、Shape 和 cache hint 不能由简化 helper 无损表达 | 维护方确认 helper、traits 和目标编译器规则 |
| O-06 | Pointer Slice 缺少 ShapeInfo；const、数组、混合重载和显式模板尚未目标编译验证 | 确认显式 shapeValue 和 const 源范围 |
| O-07 | UB→L1 ND2NZ、Pad UB→L1 stride、L1→Fixpipe blockLen、CO12 NoQuant 存在文档/源码差异 | 指定固定版本和验收真值 |
| O-08 | 同任务已有多个开放设计 PR，可能发生重复认领和 reviewer 冲突 | 活动页确认认领、评审和合入顺序 |

## 七、后续实现计划

1. 设计评审前关闭 O-01～O-04，确认 `09-42-Basic_API_DMA` 路径和正式 comparator。
2. 在独立 asc-devkit worktree 中先验证地址空间、数组、const、SFINAE 和显式模板的最小编译矩阵。
3. 按 DataCopy、DataCopyPad、ND-DMA、DataCopyL1ToUB、DataCache 分批增加 Pointer/混合候选，保留 Tensor 入口并复用原 `*Impl`。
4. 完成 44 行逐项矩阵、边界/负例/生命周期和代表样例后，执行 Header Checker、CPU/Host 检查和目标 NPU 正确性验证。
5. 设计批准并合入后，按 asc-devkit 贡献规范提交必要的需求 Issue，再进入生产代码 PR。性能无额外任务门槛，仅记录可解释的回归观察。

## 附录 A：官方 44 条重载矩阵

以下签名来自任务链接的官方表格，保留原始模板、默认值和管线属性。它们是设计追踪行，不是本 PR 已完成的代码覆盖或 Ascend 950 Device 支持承诺。

| ID | API | 官方表格行 | 头文件 |
| --- | --- | ---: | --- |
| DMA-001 | `DataCacheCleanAndInvalid` #1 | 2 | `kernel_operator_cache_intf.h` |
| DMA-002 | `DataCacheCleanAndInvalid` #2 | 3 | `kernel_operator_cache_intf.h` |
| DMA-003 | `DataCacheCleanAndInvalid` #3 | 4 | `kernel_operator_cache_intf.h` |
| DMA-004 | `DataCachePreload` #1 | 5 | `kernel_operator_cache_intf.h` |
| DMA-005 | `DataCopy` #1 | 6 | `kernel_operator_data_copy_intf.h` |
| DMA-006 | `DataCopy` #2 | 7 | `kernel_operator_data_copy_intf.h` |
| DMA-007 | `DataCopy` #3 | 8 | `kernel_operator_data_copy_intf.h` |
| DMA-008 | `DataCopy` #4 | 9 | `kernel_operator_data_copy_intf.h` |
| DMA-009 | `DataCopy` #5 | 10 | `kernel_operator_data_copy_intf.h` |
| DMA-010 | `DataCopy` #6 | 11 | `kernel_operator_data_copy_intf.h` |
| DMA-011 | `DataCopy` #7 | 12 | `kernel_operator_data_copy_intf.h` |
| DMA-012 | `DataCopy` #8 | 13 | `kernel_operator_data_copy_intf.h` |
| DMA-013 | `DataCopy` #9 | 14 | `kernel_operator_data_copy_intf.h` |
| DMA-014 | `DataCopy` #10 | 15 | `kernel_operator_data_copy_intf.h` |
| DMA-015 | `DataCopy` #11 | 16 | `kernel_operator_data_copy_intf.h` |
| DMA-016 | `DataCopy` #12 | 17 | `kernel_operator_data_copy_intf.h` |
| DMA-017 | `DataCopy` #13 | 18 | `kernel_operator_data_copy_intf.h` |
| DMA-018 | `DataCopy` #14 | 19 | `kernel_operator_data_copy_intf.h` |
| DMA-019 | `DataCopy` #15 | 20 | `kernel_operator_data_copy_intf.h` |
| DMA-020 | `DataCopy` #16 | 21 | `kernel_operator_data_copy_intf.h` |
| DMA-021 | `DataCopy` #17 | 22 | `kernel_operator_data_copy_intf.h` |
| DMA-022 | `DataCopy` #18 | 23 | `kernel_operator_data_copy_intf.h` |
| DMA-023 | `DataCopy` #19 | 24 | `kernel_operator_data_copy_intf.h` |
| DMA-024 | `DataCopy` #20 | 25 | `kernel_operator_data_copy_intf.h` |
| DMA-025 | `DataCopy` #21 | 26 | `kernel_operator_data_copy_intf.h` |
| DMA-026 | `DataCopy` #22 | 27 | `kernel_operator_data_copy_intf.h` |
| DMA-027 | `DataCopy` #23 | 28 | `kernel_operator_data_copy_intf.h` |
| DMA-028 | `DataCopy` #24 | 29 | `kernel_operator_data_copy_intf.h` |
| DMA-029 | `DataCopy` #25 | 30 | `kernel_operator_data_copy_intf.h` |
| DMA-030 | `DataCopy` #26 | 31 | `kernel_operator_data_copy_intf.h` |
| DMA-031 | `DataCopy` #27 | 32 | `kernel_operator_data_copy_intf.h` |
| DMA-032 | `DataCopyL1ToUB` #1 | 33 | `kernel_operator_data_copy_intf.h` |
| DMA-033 | `DataCopyL1ToUB` #2 | 34 | `kernel_operator_data_copy_intf.h` |
| DMA-034 | `DataCopyPad` #1 | 35 | `kernel_operator_data_copy_intf.h` |
| DMA-035 | `DataCopyPad` #2 | 36 | `kernel_operator_data_copy_intf.h` |
| DMA-036 | `DataCopyPad` #3 | 37 | `kernel_operator_data_copy_intf.h` |
| DMA-037 | `DataCopyPad` #4 | 38 | `kernel_operator_data_copy_intf.h` |
| DMA-038 | `DataCopyPad` #5 | 39 | `kernel_operator_data_copy_intf.h` |
| DMA-039 | `DataCopyPad` #6 | 40 | `kernel_operator_data_copy_intf.h` |
| DMA-040 | `DataCopyPad` #7 | 41 | `kernel_operator_data_copy_intf.h` |
| DMA-041 | `DataCopyPad` #8 | 42 | `kernel_operator_data_copy_intf.h` |
| DMA-042 | `DataCopyPad` #9 | 43 | `kernel_operator_data_copy_intf.h` |
| DMA-043 | `DataCopyPad` #10 | 44 | `kernel_operator_data_copy_intf.h` |
| DMA-044 | `DataCopyPad` #11 | 45 | `kernel_operator_data_copy_intf.h` |

### DMA-001 / 官方表格 E2

```cpp
template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst);
```

### DMA-002 / 官方表格 E3

```cpp
template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dst);
```

### DMA-003 / 官方表格 E4

```cpp
template <typename T, CacheLine entireType>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst);
```

### DMA-004 / 官方表格 E5

```cpp
template <typename T>
__aicore__ inline void DataCachePreload(const GlobalTensor<uint64_t>& src, const T cacheOffset);
```

### DMA-005 / 官方表格 E6

```cpp
template <
typename T, typename U, typename Std::enable_if< Std::is_same<PrimT<T>, bfloat16_t>::value && Std::is_same<PrimT<U>, float>::value, bool>::type = true> __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
```

### DMA-006 / 官方表格 E7

```cpp
template <
typename T, typename U, typename Std::enable_if< Std::is_same<PrimT<T>, int16_t>::value && Std::is_same<PrimT<U>, int32_t>::value, bool>::type = true> __aicore__ inline __inout_pipe__(V)void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
```

### DMA-007 / 官方表格 E8

```cpp
template <
typename T, typename U, typename Std::enable_if< Std::is_same<PrimT<T>, int8_t>::value && Std::is_same<PrimT<U>, int32_t>::value, bool>::type = true> __aicore__ inline __inout_pipe__(V)void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
```

### DMA-008 / 官方表格 E9

```cpp
template <
typename T, typename U, typename Std::enable_if< Std::is_same<PrimT<T>, uint8_t>::value && Std::is_same<PrimT<U>, int32_t>::value, bool>::type = true> __aicore__ inline __inout_pipe__(V)void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
```

### DMA-009 / 官方表格 E10

```cpp
template <
typename T, typename U, typename Std::enable_if<Std::is_same<PrimT<T>, float>::value && Std::is_same<PrimT<U>, half>::value, bool>::type = true> __aicore__ inline __inout_pipe__(V)void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
```

### DMA-010 / 官方表格 E11

```cpp
template <
typename T, typename U, typename Std::enable_if<Std::is_same<PrimT<T>, half>::value && Std::is_same<PrimT<U>, float>::value, bool>::type = true> __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
```

### DMA-011 / 官方表格 E12

```cpp
template <
typename T, typename U, typename Std::enable_if<Std::is_same<PrimT<T>, half>::value && Std::is_same<PrimT<U>, int32_t>::value, bool>::type = true> __aicore__ inline __inout_pipe__(V)void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
```

### DMA-012 / 官方表格 E13

```cpp
template <typename T, bool enableSmallC0 = false>
__aicore__ inline __inout_pipe__(MTE2)void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const Dn2NzParams& intriParams);
```

### DMA-013 / 官方表格 E14

```cpp
template <typename T, bool enableSmallC0 = false>
__aicore__ inline __inout_pipe__(MTE2)void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const Nd2NzParams& intriParams);
```

### DMA-014 / 官方表格 E15

```cpp
template <typename T, typename U>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyCO12DstParams& intriParams);
```

### DMA-015 / 官方表格 E16

```cpp
template <typename T, typename U>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyCO12DstParams& intriParams);
```

### DMA-016 / 官方表格 E17

```cpp
template <typename T, typename U>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyParams& repeatParams);
```

### DMA-017 / 官方表格 E18

```cpp
template <typename T, uint8_t dim, const NdDmaConfig& config = kDefaultNdDmaConfig>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const MultiCopyParams<T, dim>& params);
```

### DMA-018 / 官方表格 E19

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE2)void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
```

### DMA-019 / 官方表格 E20

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE2)void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const Nd2NzParams& intriParams);
```

### DMA-020 / 官方表格 E21

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE2)void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[], const uint32_t dimValue = 1);
```

### DMA-021 / 官方表格 E22

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE2)void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const uint32_t count);
```

### DMA-022 / 官方表格 E23

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE3)void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
```

### DMA-023 / 官方表格 E24

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE3)void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& repeatParams);
```

### DMA-024 / 官方表格 E25

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE3)void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const Nz2NdParamsFull& intriParams);
```

### DMA-025 / 官方表格 E26

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE3)void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[], const uint32_t dimValue = 1);
```

### DMA-026 / 官方表格 E27

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE3)void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count);
```

### DMA-027 / 官方表格 E28

```cpp
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
```

### DMA-028 / 官方表格 E29

```cpp
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& repeatParams);
```

### DMA-029 / 官方表格 E30

```cpp
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const Nd2NzParams& intriParams);
```

### DMA-030 / 官方表格 E31

```cpp
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count);
```

### DMA-031 / 官方表格 E32

```cpp
template <typename T>
__aicore__ inline void __inout_pipe__(MTE2)DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& repeatParams);
```

### DMA-032 / 官方表格 E33

```cpp
template <typename T, uint8_t subBlockId = 0>
__aicore__ inline void DataCopyL1ToUB(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& repeatParams);
```

### DMA-033 / 官方表格 E34

```cpp
template <typename T, uint8_t subBlockId = 0>
__aicore__ inline void DataCopyL1ToUB(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count);
```

### DMA-034 / 官方表格 E35

```cpp
template <
typename T, typename U, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value &&(!Std::is_same<T, U>::value), bool>::type = true> __aicore__ inline __inout_pipe__(MTE2)void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<U>& padParams);
```

### DMA-035 / 官方表格 E36

```cpp
template <typename T, PaddingMode mode = PaddingMode::Normal>
__aicore__ inline __inout_pipe__(MTE2)void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<T>& padParams);
```

### DMA-036 / 官方表格 E37

```cpp
template <typename T, PaddingMode mode = PaddingMode::Normal>
__aicore__ inline __inout_pipe__(MTE2)void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& dataCopyParams, const DataCopyPadParams& padParams);
```

### DMA-037 / 官方表格 E38

```cpp
template <typename T, PaddingMode mode = PaddingMode::Normal>
__aicore__ inline __inout_pipe__(MTE3)void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& dataCopyParams);
```

### DMA-038 / 官方表格 E39

```cpp
template <typename T, PaddingMode mode = PaddingMode::Normal>
__aicore__ inline __inout_pipe__(MTE3)void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& dataCopyParams);
```

### DMA-039 / 官方表格 E40

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE2)void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<T>& padParams);
```

### DMA-040 / 官方表格 E41

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE2)void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& dataCopyParams, const DataCopyPadParams& padParams);
```

### DMA-041 / 官方表格 E42

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE3)void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& dataCopyParams);
```

### DMA-042 / 官方表格 E43

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE3)void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& dataCopyParams);
```

### DMA-043 / 官方表格 E44

```cpp
template <typename T>
__aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const Nd2NzParams& nd2nzParams);
```

### DMA-044 / 官方表格 E45

```cpp
template <typename T>
__aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& dataCopyParams, const Nd2NzParams& nd2nzParams);
```

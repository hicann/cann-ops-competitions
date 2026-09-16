# [Requirement|需求建议]: 【社区任务】AscendC Basic API 指针化扩展（DMA）设计文档评审申请

> 任务编号：08-01-ascendc-basic-api-dma-pointer
>
> 参与任务昵称：linjuan
>
> GitCode 账号：linjuan
>
> 版本：v0.1（开发前设计）
>
> 目标产品：Ascend 950 系列
>
> 目标仓库：asc-devkit

# 一、需求背景

## 1.1 需求来源

本设计对应社区任务“AscendC Basic API 指针化扩展（DMA · 数据搬运 / 缓存）”。任务目标是在不改变现有 Tensor 调用兼容性的前提下，使开发者能够直接使用带地址空间限定的硬件指针调用 DMA Basic API。

接口清单以 DMA 接口表中类型码 `D` 的 5 个 API 名称及其 44 条重载为准：

- `DataCopy`
- `DataCopyPad`
- `DataCopyL1ToUB`
- `DataCachePreload`
- `DataCacheCleanAndInvalid`

任务范围不包括 VECTOR、CUBE/ISASI 或其他分册接口。`LoadData`、`Mmad`、`Fixpipe` 保持原有 Tensor 形式，不在本设计中增加指针重载。

## 1.2 当前基线

现有 Basic API 对外主要以 `LocalTensor` / `GlobalTensor` 表达操作数，平台层已经通过地址空间指针完成实际数据搬运。指针化扩展应位于公共接口封装层，目标文件为：

```text
include/basic_api/kernel_operator_data_copy_intf.h
include/basic_api/kernel_operator_cache_intf.h
impl/basic_api/kernel_operator_data_copy_intf_impl.h
impl/basic_api/kernel_operator_cache_intf_impl.h
```

本设计不改变任何已有 `*Impl` 的数值语义、地址计算、同步约定、缓存策略或硬件指令选择。

# 二、需求分析

## 2.1 目标与非目标

| 类型 | 内容 |
| --- | --- |
| 目标 | 为官方 44 条 DMA/Cache 重载增加合法硬件指针操作数形式 |
| 目标 | 保留 Tensor/Tensor 调用的原有重载、检查、SFINAE 和 pipe 属性 |
| 目标 | Pointer 路径与等价 Tensor 路径复用同一套底层实现 |
| 目标 | 不增加与输入规模线性相关的临时 Device 拷贝 |
| 非目标 | 不改造 `LoadData`、`Mmad`、`Fixpipe` |
| 非目标 | 不新增搬运算法、同步机制、Host 注册或运行时 ABI |
| 非目标 | 不把所有输入统一降级为 `void*`、整数地址或无地址空间模板 |

## 2.2 地址空间和操作数

| 地址空间 | 指针形式 | 允许的典型路线 |
| --- | --- | --- |
| GM | `__gm__ T*` | GM->UB、GM->L1、UB->GM、L1->GM、L0C->GM、Cache |
| UB | `__ubuf__ T*` | GM<->UB、UB->UB、UB->L1、L1->UB |
| L1 | `__cbuf__ T*` | GM->L1、UB->L1、L1->UB、CO12 相关路线 |
| L0C | `__cc__ T*` | DataCopy CO12 相关搬出路线 |

地址空间是类型契约，不能由数值地址推断。错误的地址空间组合应在模板候选、已有位置检查或底层实现处失败，不能静默转换后调用其他路线。

## 2.3 兼容性原则

1. 两个操作数均为 Tensor 时，必须继续选择原有显式重载。
2. 至少一个操作数为 Pointer 时，才启用新增 Pointer/混合候选。
3. 单 dtype 重载要求 Pointer pointee 与原 primitive 类型一致。
4. 双 dtype 重载继续使用原有 `PrimT<T>`、`PrimT<U>` 和 SFINAE 关系。
5. 目的操作数拒绝 const pointee；源操作数的 const 规则按具体 API 保留。
6. Slice Pointer 路径不能依赖 Tensor ShapeInfo，应要求调用方提供必要的显式 shape 元数据。
7. 原有 `__aicore__`、`__inout_pipe__`、`__in_pipe__`、`__out_pipe__`、架构条件和非类型模板参数逐条保留。

# 三、方案设计

## 3.1 统一适配模型

```text
Tensor/Tensor
    -> 原有 Tensor 重载、position/shape/cache 检查

Pointer/混合操作数
    -> 地址空间和元素类型约束
    -> 取得底层硬件指针
    -> 复用同一条已有 *Impl 路由
    -> 保留原同步和 pipe 语义
```

适配层只做编译期类型适配，不分配 staging buffer、不复制输入、不插入 barrier。

## 3.2 DataCopy 设计

按参数族分别提供指针重载，不使用一个无约束的万能模板：

| 参数族 | 设计覆盖 |
| --- | --- |
| count | GM<->UB、UB<->L1、L1->UB 等合法路线 |
| `DataCopyParams` | GM/UB/L1 连续和分块搬运 |
| `Nd2NzParams` / `Dn2NzParams` | GM->L1 格式转换，并保留 SmallC0 约束 |
| `MultiCopyParams` | NDDMA 的维度、stride、padding 和 config |
| `SliceInfo` | GM<->UB 切片搬运，Pointer 路径要求显式 shapeValue |
| `DataCopyCO12DstParams` | L0C->GM 随路量化、激活和布局转换 |
| Enhanced 参数 | 仅对官方清单中已有的类型组合增加对应 Pointer 入口 |

每个重载将地址空间指针按原元素类型传递到对应 `DataCopy*Impl`。不改变 blockLen、stride、C0、字节数和格式转换参数的单位。

## 3.3 DataCopyPad 设计

分别设计 GM->UB、GM->L1 和 UB->GM 指针入口，保留：

- `DataCopyParams` 与 `DataCopyPadParams`；
- `DataCopyExtParams` 与 `DataCopyPadExtParams<T>`；
- `PaddingMode` 非类型模板参数；
- 原有 block、stride、左/右 padding、pad value 和异步 pipe 属性。

Pad 适配层不修改参数、不生成额外 padding buffer，直接复用既有 `DataCopyPad*Impl`。

## 3.4 DataCopyL1ToUB 与 DataCache 设计

`DataCopyL1ToUB` 增加 `__ubuf__` 目的指针和 `__cbuf__` 源指针形式，保留 `subBlockId`、count/Params 以及 Mix 1:2 约束。

DataCache 增加以下 Pointer 入口：

- `DataCachePreload(__gm__ uint64_t*, OffsetType)`；
- `DataCacheCleanAndInvalid(__gm__ T*)`；
- `DataCacheCleanAndInvalid(__ubuf__ T*)`；
- 带 `CacheLine`、`DcciDst` 的官方对应模板形式。

Cache Pointer 入口不改变 cache line、DCCI 目标或内存可见性语义。Cache API 不承担跨核同步职责。

## 3.5 官方重载追踪

实现阶段为官方 44 条重载建立 DMA-001 至 DMA-044 的追踪表，每条记录以下内容：

- 原始声明和文件位置；
- Pointer/混合操作数形式；
- 地址空间组合；
- 模板参数和 SFINAE 条件；
- 对应底层 `*Impl`；
- 编译正例、负例和行为验证用例；
- 不适用架构或硬件路线说明。

官方表格与当前源码声明数量存在差异时，以评审确认的 44 条任务清单为准，不擅自扩大范围。

# 四、地址、布局与同步约束

## 4.1 地址和布局

- GM、UB、L1、L0C 指针必须使用与原接口一致的地址空间限定。
- 连续搬运、分块搬运、ND2NZ/DN2NZ、NDDMA 和 Slice 分别遵守原 API 的单位和边界。
- count、blockLen、stride、padding、shapeValue 和维度参数不由适配层隐式修正。
- Pointer Slice 不拥有 Tensor ShapeInfo；调用方必须提供显式 shape 信息。
- Pointer 路径不承诺重叠内存的 memmove 语义，源/目的重叠按原 API 约束处理。

## 4.2 异步和生命周期

适配层不新增等待、不删除等待、不改变已有 pipe。调用方负责：

- GM->UB 后的计算和写回依赖；
- UB/L1/L0C 的事件或跨核依赖；
- Cache 操作的适用范围；
- 异步调用完成前的 buffer 生命周期；
- stream 同步和 Host 侧读回时机。

# 五、可维可测设计

## 5.1 测试矩阵

测试计划覆盖以下维度：

| 维度 | 计划场景 |
| --- | --- |
| 调用形态 | Tensor/Tensor、Pointer/Pointer、Tensor/Pointer、Pointer/Tensor |
| 模板 | 自动推导、显式 dtype、SmallC0、PaddingMode、NDDMA config、subBlockId、CacheLine、DcciDst |
| 地址空间 | GM、UB、L1、L0C 的合法路线和错误路线负例 |
| 数据参数 | count=0/1、对齐边界、尾块、stride/padding 边界、最大值边界 |
| Shape | 1、32、1024、2048 代表规模；Slice 多维 shape；NDDMA 多维 loop |
| 生命周期 | 首次调用、重复调用、buffer 复用、异步提前消费、stream 同步 |
| 兼容性 | 原 Tensor 调用回归、Pointer 与 Tensor 同布局结果一致 |

任务包中的 6 个样例用于覆盖 GM/L1、NDDMA、Slice、Pad、UB/L1 和 L0C/GM 代表路线，但不替代 DMA-001～DMA-044 的逐条验证。

## 5.2 验收口径

- 编译：目标架构和编译器下，官方 Tensor 调用与新增 Pointer 调用均能解析。
- 纯搬运：比较有效字节、padding 定义区域和 canary 区域的逐位结果。
- 格式转换/量化：沿用原 API 的 golden 和任务定义的误差判据。
- Cache：验证调用范围、模板参数和可见性，不把 Cache 操作当作同步原语。
- 兼容性：原有 Tensor 路径不改变签名、布局、同步和底层实现语义。

具体数值、日志和截图属于后续自测报告，不写入本设计文档。

## 5.3 性能和内存计划

任务书无额外性能门槛。实现应保持编译期适配，不引入与输入规模线性相关的 Device 临时拷贝。实现阶段可选择代表 case 记录 Pointer 与 Tensor 路径的时间对比，但性能数据属于自测报告。

# 六、风险与待确认问题

| 编号 | 问题 | 关闭条件 |
| --- | --- | --- |
| O-01 | 任务书名称存在“8月/9月”文字差异 | 官方确认正式标题 |
| O-02 | 官方 44 条重载与源码声明数量、旧架构保护分支存在差异 | 评审确认 44 条逐项范围 |
| O-03 | 部分代表样例依赖 CUBE/ISASI | 明确样例只验证 DMA，非 DMA API 不改造 |
| O-04 | Pointer Slice 缺少 Tensor ShapeInfo | 确认显式 `shapeValue` 约束 |
| O-05 | Cache hint、const pointer、混合 dtype 的模板边界 | 评审确认 traits/SFINAE 规则 |
| O-06 | 不同 CANN 小版本的复杂模板诊断可能不同 | 固定评审使用的 SDK 和编译器版本 |


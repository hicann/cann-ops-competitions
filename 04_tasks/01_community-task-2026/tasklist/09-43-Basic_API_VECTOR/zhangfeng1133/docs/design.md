# Ascend C Basic API 指针化扩展设计文档（VECTOR · 矢量计算分册）

> 任务：2026 社区任务 · Ascend C Basic API 指针化扩展（VECTOR 分册，总表类型码 **V**，含排序类 N 中已纳入清单的项）
> 交付：`asc-devkit` 的 `include/basic_api`、`impl/basic_api`（PR 合入 `master`）；硬件与版本：Ascend 950 系列产品，CANN 9.0.0 ~ 9.1.0，Ascend C / C++（毕昇 ASC 编译器）
> 本册规模：**79** 个接口名、**245** 个重载签名（任务书 1）；姊妹册 DMA（5/44）、CUBE（21/61）接口互不重叠

# 需求背景（required）

## 需求来源

社区任务「Ascend C Basic API 指针化扩展（VECTOR / 矢量计算分册）」任务书（本仓配套 `basic_api_optimize_vector.md`）要求：
对本册 79 个接口名、245 个重载做指针化扩展，使开发者可直接传入裸指针（如 `__ubuf__ T*`）或继续使用 `LocalTensor` 调用
同一套 API；在保持原有 Tensor 接口兼容的前提下完成设计、开发、自验证，并按任务书第 5 节目录约定向 `asc-devkit` `master`
分支提交 PR。

## 背景介绍

### Ascend C Basic API 现状分析

矢量计算类 Basic API 的现行形态是「`LocalTensor<T>` 入参 + `*Impl` 下沉」：声明在 `include/basic_api`，实现在
`impl/basic_api`，接口体内把 Tensor 转成 `(__ubuf__ PrimType*)tensor.GetPhyAddr()` 后调用 `*Impl`。而在
`__global__ __vector__` 直调模型中 UB 缓冲可直接以 `__ubuf__ T*` 声明（如 `__ubuf__ float xPtr[blockLength];`），
此时仍需构造 `LocalTensor` 才能调用，形成三类具体问题：

| 现状 | 具体表现 | 影响 |
| --- | --- | --- |
| 包装冗余 | 裸指针须经 `LocalMemAllocator` / `TBuf` 包装为 `LocalTensor` 才能调用 | 与直调模型「裸指针即缓冲区」的形态不匹配 |
| 地址空间强转外溢 | 现存样例调用侧手写 `GetPhyAddr()`，部分场景还需 `(__ubuf__ T*)` 强转 | 强转散落用户代码，与 `*Impl` 的地址空间约定隐式耦合 |
| 一致性口径缺失 | 同一计算存在 Tensor 与指针两条事实写法 | 缺少统一入口与一致性的显式验收口径 |

仓库在 `examples/01_simd_cpp_api/03_basic_api/` 与本任务矢量类样例集（`test-cases/` 下 20 个目录）中，调用侧已按
`xLocal.GetPhyAddr()` 的指针形态书写，可作为改造后 **指针路径** 的调用范式来源；本册目标即让对外 API 直接接受裸指针，
调用侧不再需要包装与强转，同时 Tensor 路径保持原样。

### 本册改造功能分析

改造是 **编译期入参类型适配**，不新增算子、不改变数值语义：

1. 凡清单中标记为 Local 操作数的 Tensor 形参，统一改为模板入参并经 `GetUnderlyingPtr` 下沉至既有 `*Impl`；
2. `scalarValue`、`mask` / `mask[]`、`repeatTime`、`*RepeatParams`、`count`、`RoundMode`、`CMPMODE` / `SELMODE` /
   `ReduceType` 等非 Tensor 入参保持原类型、原顺序、原默认值；
3. Tensor 路径的入参对象与最终下传地址不变，数值行为与改造前一致；指针路径不做任何数据变换。

# 需求分析（required）

## 需求描述

在 `include/basic_api` / `impl/basic_api` 中，对本册 **79 个接口名、245 个重载** 完成指针化扩展，使其同时接受
`LocalTensor<T>` 与 `__ubuf__ T*` 等裸指针，覆盖任务书 2.1 所列分类，并保证：原 Tensor 接口签名、模板约束与数值行为
**零变更**；指针路径与对应 Tensor 用法在相同数据布局与参数下 **数值一致**；仓库矢量类样例可由 Tensor 写法迁移为指针写法并
通过自验（任务书 3.5，示例基于 `.../03_basic_api/01_memory_vector_compute/element_wise_compound_compute/`）。
本册接口清单按任务书 2.1 分类组织（重载逐条签名以任务书 2.4 清单为准，总量 79 名 / 245 重载）：

| # | 分类 | 覆盖能力 | 样例可判定的代表接口 |
| --- | --- | --- | --- |
| 1 | 矢量一元 | 逐元素一元变换 | `LeakyRelu`、`Abs` 类 |
| 2 | 矢量二元 / 标量 | 逐元素二元与标量混合运算 | `Add`、`AddRelu`、`Axpy`、`Mul` 类 |
| 3 | 类型转换 | dtype / 精度转换（`RoundMode` 语义不变） | `Cast` |
| 4 | 归约 | 块内 / repeat 内归约及结果读取 | `ReduceMax`、`ReduceMin`、`ReduceSum`、`ReduceRepeat`、`ReduceDataBlock`、`ReducePairElem` |
| 5 | 比较选择 | 逐元素比较、掩码读取与选择 | `Compare`、`Compares`、`GetCmpMask`、`Select` |
| 6 | Gather / Scatter | 离散选取与写回 | `Gather`、`Gatherb`、`GatherMask` |
| 7 | 填充广播 | 填充、广播与向量索引生成 | `Duplicate`、`Brcb`、`CreateVecIndex`、`Interleave`、`DeInterleave` |
| 8 | 双线性插值 | 插值类矢量接口 | 按 2.4 清单纳入的插值符号 |
| 9 | 队列同步与 mask 配置 | UB 内数据搬运与 mask 模式配置 | `Copy`、`SetMaskNorm`、`SetMaskCount`、`SetVectorMask`、`SetPadValue` |
| 10 | 排序类（类型码 **N** 已纳入项） | 排序、归并、转置 | `Sort` / `Sort32`、`MrgSort`、`MrgSort4`、`RpSort16`、`Transpose`、`TransDataTo5HD` |
| 合计 | — | **79** 个接口名 / **245** 个重载 | 依据任务书 1、2.4 |

> 分类仅用于组织改造与测试：第 1、2 行合起来对应任务书「矢量一元/二元/标量」，第 3~9 行对应任务书「类型转换、归约、比较选择、
> Gather/Scatter、填充广播、双线性插值、队列同步」，第 10 行为任务书第 1 节所述已纳入清单的排序类 N 项。
> 边界声明：矩阵 / 立方专用接口、Proposal 拆分类接口等已排除符号不在本册交付范围；样例集 `region_proposal_sort` 涉及但
> **未** 出现在 2.4 清单中的符号不改造，但本册改造导致的既有用例回归失败仍须修复（任务书 3.5 第 4 条）。

## 需求拆解

| # | 拆解项 | 交付内容与验收口径 |
| --- | --- | --- |
| 1 | 萃取机制 | 模板封装层统一 `GetUnderlyingPtr`：裸指针原样返回，Tensor 走 `GetPhyAddr()`；无运行时开销 |
| 2 | 接口扩展 | 245 个重载逐条补齐指针入参形态，覆盖 245/245，无遗漏、无越册符号 |
| 3 | 重载安全 | 新增指针重载不得抢占原 Tensor 重载；两操作数皆为 Tensor 的调用仍解析到原重载 |
| 4 | 掩码 / 计数范式 | 同一接口的 `mask[]` + `repeatTime` 式与 `count` 式两种调用范式均须覆盖 |
| 5 | 非 Tensor 入参 | 标量、mask、`*RepeatParams`、枚举模式参数保持原类型、顺序与默认值 |
| 6 | 兼容回归 | 原 Tensor 用例不改写全量回归；改造导致既有用例回归失败须修复 |
| 7 | 自验与交付 | 覆盖一元 / 二元 / 转换 / 归约 / 比较 / Gather / 填充 / 排序等分类的指针路径用例；英文注释与研发协作规范；易用性 Issue 按「【AscendC CAPI社区任务】xxx」提交并归档 |

# 详细设计（required）

## 算子分析

### 数学公式

本册为矢量指令集合，逐接口给出语义定义（`i` 为向量元素下标，`s` 为标量入参，`r` 为 repeat 下标）：

```
一元:   dst[i] = f(src[i])
二元:   dst[i] = f(src0[i], src1[i])                标量式: dst[i] = f(src0[i], s)
转换:   dst[i] = cast<DstT>(src[i])                 舍入由 RoundMode 决定（语义不变）
归约:   dst[r] = fold(src[i])                       i 覆盖第 r 次 repeat 内元素，元素数由 dtype 与 mask 模式决定
比较:   dst[i] = (src0[i] cmp src1[i]) ? 1 : 0      cmp ∈ CMPMODE
选择:   dst[i] = 按 SELMODE 从 (src0[i], src1[i]) 中择一写入，选择依据 mask[i]
Gather: dst[i] = src[index[i]]                      离散读
填充:   dst[i] = s                                   或 dst = broadcast(src)
排序:   dst = merge / sort(src, 按 score 域降序)     顺序与稳定性语义按各接口文档定义
```

指针化改造 **不改变** 上述任何表达式：`f`、`cast` 的舍入模式、`fold` 的结合顺序、比较与选择模式、排序键与顺序语义均与
改造前一致。这是「指针路径与 Tensor 路径数值一致」的设计基点。

### 支持数据类型

不新增 dtype 支持范围，沿用 2.4 清单与官方接口文档既有约定；类型由模板入参与 `ElemType<decltype(ptr)>` 推导：

| 数据类型族 | 典型类型 | 关联分类 |
| --- | --- | --- |
| 浮点 | `half`、`bfloat16_t`、`float` | 一元、二元、归约、比较、填充、插值 |
| 整型 | `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t` | 逻辑 / 移位、比较（输出掩码）、Gather 索引 |
| 亚字节 / 低精度 | `int4b_t`、`fp4x2_*`、`fp8_*`（视接口） | 由各接口文档约束 |
| 掩码 | 掩码位串、`MaskReg` 相关形态 | 比较输出、选择输入 |

约束：指针路径与 Tensor 路径须推导出同一 `PrimType`；`Cast` 的源 / 目的 dtype 组合与 `RoundMode` 取值组合须落在官方支持域
内，指针化不扩大也不收窄该域。

### 支持形状

| 形态 | 覆盖样例 | 说明 |
| --- | --- | --- |
| 连续一维 + count | `cast`、`duplicate`、`create_vec_index`、`element_wise_compound_compute` | `totalLength` 式调用 |
| mask + repeatTime + RepeatParams | `element_wise_arithmetic`、`element_wise_logic`、`compare`、`copy_ub2ub` | mask 数组式与 `MASK_PLACEHOLDER` 式 |
| repeat 驱动的归约 | `reduce`、`reduce_repeat`、`reduce_data_block`、`reduce_pair_elem`、`reduce_computation` | 数据块语义 |
| 离散索引 | `gather` | `GatherRepeatParams` + `rsvdCnt` 输出参数 |
| 交织 / 转置 | `interleave_pair`、`transpose` | 16×16 数据块、维度置换类 |
| 排序队列 | `mrg_sort` | 4 队列归并，score 域降序 |
| 规模覆盖 | `1` / `32` / `1024` / `2048`（任务书 3.2） | `1` 与 `2048` 暴露边界（单元素、尾块不齐），`32` 与 `1024` 覆盖单个 / 多个 DataBlock |

## 算子实现

### 实现方案

#### Host 侧设计（对外接口层 `include/basic_api`）

本册为接口库、无 Tiling 与算子 Host 逻辑，「Host 侧」指 **对外声明层** 设计：

1. **接口分层**：声明在 `include/basic_api`、实现在 `impl/basic_api`，一一对应；仅改本册清单内接口所在文件。
2. **统一改造范式**：以 `Add`（mask 式）为例，原实现把 `LocalTensor` 强转后调用 `AddImpl`，扩展后改为模板入参 + 编译期
   萃取，`mask` / `repeatTime` / `BinaryRepeatParams` 原样透传：

```cpp
// Unified extraction of the underlying hardware pointer.
template <typename U>
__aicore__ inline auto GetUnderlyingPtr(const U& val)
{
    if constexpr (std::is_pointer_v<U>) {
        return val;                 // raw pointer, e.g. __ubuf__ float*
    } else {
        return val.GetPhyAddr();    // LocalTensor<T> and similar
    }
}

// Original overload: dst/src0/src1 are LocalTensor<T>; signature kept unchanged.
// Added overload: three INDEPENDENT template parameters (cast mixes dtypes; some ops mix Tensor and pointer).
template <typename T, typename U, typename V, bool isSetMask,
          typename = typename std::enable_if_t<
              std::is_pointer_v<std::remove_reference_t<T>> ||
              std::is_pointer_v<std::remove_reference_t<U>> ||
              std::is_pointer_v<std::remove_reference_t<V>>>>
__aicore__ inline void Add(const T& dst, const U& src0, const V& src1, uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    auto dstPtr = GetUnderlyingPtr(dst);
    auto src0Ptr = GetUnderlyingPtr(src0);
    auto src1Ptr = GetUnderlyingPtr(src1);
    using PrimType = ElemType<decltype(dstPtr)>;
    AddImpl<PrimType, isSetMask>(dstPtr, src0Ptr, src1Ptr, mask, repeatTime, repeatParams);
}
```

3. **三操作数独立泛型**：`dst` / `src0` / `src1` 使用 **独立** 模板参数而非共用一个 `T`。原因是本册存在 dst 与 src dtype
   不同的接口（`Cast`）以及 `Ors` / `ShiftRight` / `LeakyRelu` / `Axpy` 等标量混合形式；共用单一模板参数会把合法的混合
   调用（部分指针 + 部分 Tensor）判成不一致而编译失败，或反过来放宽约束导致 `PrimType` 推导歧义。**约束**：三者的指针元素
   类型须推导出同一 `PrimType`（`Cast` 按接口定义允许源 / 目的不同），不满足时编译期失败。
4. **重载解析安全（关键风险项）**：泛型重载 `const T&, const U&, const V&` 若不加限定会与原 `LocalTensor<T>` 重载歧义。
   - **(a) 仅当至少一个操作数为裸指针时启用新重载**（`std::is_pointer_v<std::remove_reference_t<...>>` 析取式作为
     `enable_if` 条件）；三者皆为 Tensor 时解析回原重载（前提是该入参组合本就存在对应原重载），保证 Tensor 路径的实例化
     结果与改造前一致；
   - **(b) 逐调用点验证**：对每个改造前存在的 Tensor 调用表达式，以「仍解析到原重载」为验收项（编译期断言或符号 / 返回
     类型证据），而非仅凭「能编译过」判定无歧义。
5. **掩码与计数范式并存**：同一接口常同时存在 mask 式与 count 式重载，两类都按同一范式改造；`mask` / `mask[]`、
   `repeatTime`、`*RepeatParams`、`count` **不改类型、不改默认值、不改为指针**。`Gather` 的 `rsvdCnt` 等引用型出参、
   归约接口的结果读取保持原类型与语义。
6. **运行时成本**：新增重载为 `inline` 编译期分派，展开后 `*Impl` 调用与改造前一致，不引入运行时分支或额外寄存器开销。

#### Kernel 侧设计（实现层 `impl/basic_api`）

1. **类型推导**：`using PrimType = ElemType<decltype(GetUnderlyingPtr(dst))>`；指针路径由指针元素类型推导，Tensor 路径由
   Tensor 元素类型推导，二者必须一致——这保证同一 `*Impl` 实例被复用，而非各自实例化出不同行为。
2. **展开等价**：两路径最终调用同一个 `*Impl`（同一 mask / repeat / params），指令序列一致，因此一致性是「同一实现 + 同一
   地址 + 同一参数」的结果，不依赖额外补偿逻辑。
3. **元素索引与切片**：样例中 `src0Local[0]`（标量读）的等价形态为 `src0Ptr[0]`；`tensor[i * offset]` 的等价形态为
   `ptr + i * offset`（偏移单位以各接口文档为准，元素偏移与字节偏移不可混用）。
4. **mask 与 repeat 语义不变**：`SetMaskNorm` / `SetMaskCount` / `SetVectorMask` 的当前模式决定 mask 解释方式，指针化不得
   改变模式设置顺序或插入位置；`MASK_PLACEHOLDER` 语义保持。
5. **同步不变**：`SetFlag` / `WaitFlag` / `PipeBarrier` / `Mutex` 不属于本册接口，其 event id、pipe 顺序与插入位置原样保留
   （如样例中的 `MTE2_V` → 计算 → `V_MTE3` 序列）。

#### 分类特化与 dispatch 组织

| 分类 | 特化关注点 | 组织方式 |
| --- | --- | --- |
| 一元 / 二元 / 标量 | 操作数个数不同（1~3 个 Tensor 形参）与标量混入 | 每形态一组模板重载，共用 `GetUnderlyingPtr` 与同一 `*Impl` |
| 类型转换（`Cast`） | 源 / 目的 dtype 不同，`RoundMode` 参与模板实例化 | 源与目的分别推导，**禁止合并成单模板参数** |
| 归约 | 输入为 UB 向量，输出为 UB 位置或结果读取 | 输出同样走泛型（指针或 Tensor），保持 repeat 维度语义 |
| 比较 / 选择 | mask 既是输入也是输出（`GetCmpMask`） | mask 类操作数按其官方形态处理，不强行指针化 |
| Gather / Scatter | 索引与数据分离，含引用出参 `rsvdCnt` | 数据 / 索引分别泛型；出参保持引用语义 |
| 排序类（N） | 多队列入参（`MrgSortSrcList`）、score 域与顺序语义 | 队列成员逐个 `GetUnderlyingPtr`；顺序语义零变更 |
| 队列同步 / mask 配置 | 无数据操作数或仅配置语义 | **不做指针化**，保持原签名 |

**原则**：不做运行时 dispatch（此处无运行期类型信息），全部为编译期选择（`if constexpr` 判形态 + 模板约束判合法性）；
**不为指针路径复制实现体**，避免后续维护漂移。

#### 数值一致性保障（设计论证 + 计划验证项）

| 保障手段 | 设计说明 |
| --- | --- |
| 单一实现 + 同地址 + 同 PrimType | 两路径共用同一 `*Impl`；`GetPhyAddr()` 即 Tensor 持有的 UB 物理地址，同布局下应重合；`ElemType` 推导保证实例化同一类型版本 |
| 同参数 + 边界覆盖 | mask / repeatTime / RepeatParams / RoundMode / 模式开关须完全一致，否则不构成等价比对；`1` / `32` / `1024` / `2048`、非对齐尾块、变换类接口的离散访问进入用例 |

以上为设计论证与计划验证项，实测结论在开发完成后的自测报告中给出。

### 测试设计

以任务书 3.5 指定的 `examples/01_simd_cpp_api/03_basic_api/`（尤其 `01_memory_vector_compute/`）与本任务矢量类样例集
（`test-cases/` 下 20 个目录）为基础做 **双路径对比**。**基线定义**：样例集中部分用例调用侧已采用 `GetPhyAddr()` 指针写法
（本册将其作为指针路径的写法来源），故「Tensor 基线」由 **同一用例的 Tensor 写法** 确定，即同一片 UB 缓冲区以
`LocalTensor` 直接入参，指针写法作为对照；要求两条写法 **均编译通过、均可运行、输出一致**，且 Tensor 写法在改造后仍解析到
原重载。

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 数据值域 | `[-100, 100]` | 任务书 3.2；含负值、小数，以及插值 / 排序类接口的合法值域 |
| Shape | `1`、`32`、`1024`、`2048` | 覆盖单元素、单块、多块与尾块 |
| 调用范式 | mask 式与 count 式各至少 1 例 | 掩码与计数两种入口都要覆盖 |
| 数据类型 | 各接口官方支持的 dtype 子集（`half` / `bfloat16_t` / `float` / 整型） | 每个 dtype 至少 1 例 |
| 接口覆盖 | 本册 245 个重载逐条勾选 | 覆盖矩阵随 PR 提交 |
| 生成与校验 | 仓库 `script/gen_data.py`（如 `02_reg_vector_compute/abs/scripts`）或等价脚本；`verify_result.py` / 逐元素比对 | 固定种子保证可复现；指针 vs Tensor 为本册新增判据 |
| 回归 | 原 Tensor 用例不改写全量执行 | 改造导致既有用例回归失败须修复 |

用例编号建议 `V-{分类}-{范式}-{dtype}-{Shape}`（如 `V-reduce-mask-fp16-2048`）；报告逐例给出两条路径的输出摘要与结论。
任务书明确 **无** 独立性能 case，性能项填「无」。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 系列产品 | √ |

## 算子约束限制

1. **仅扩展指针入参**：不新增 dtype 与计算语义，不改变 mask 解释模式与 `RoundMode` 语义。
2. **不改 `*Impl` 数值语义**：既有舍入、饱和、归约顺序、排序顺序语义不做顺手优化；发现缺陷走独立 Issue。
3. **本册边界**：仅交付 2.4 清单所列 VECTOR（含已纳入的 N 类）接口；**勿将 DMA / CUBE 他册接口或其他分册符号混入本 PR**
   （任务书 7.3），避免范围蔓延与 PR 冲突。
4. **Tensor 路径零语义变更**：原 `LocalTensor` 重载签名与解析结果不变。
5. **非 Tensor 入参与地址空间不动**：`mask` / `repeatTime` / `*RepeatParams` / `count` / 标量 / 枚举模式参数保持原类型、
   顺序与默认值；裸指针须携带正确地址空间，不引入丢失地址空间的 `void*` 中间形态。
6. **无额外内存与同步越权**：不引入与输入规模线性相关的额外 Device 内存拷贝；本册接口不负责跨 pipe 同步，不得在接口内部
   新增隐式同步或 event。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（设计阶段口径） | 标准来源 |
| --- | --- | --- |
| 精度标准 | ① 本册接口的原 Tensor 用例回归 **全部通过**；② 指针路径与 Tensor 路径在相同数据布局与参数下 **数值一致** | 生态算子开源精度标准（实验标准）；任务书 3.2 |
| 性能标准 | **无** 额外性能指标与标杆时延要求；自测报告说明相对改造前无明显性能回退，如有可解释差异可选填说明 | 任务书 3.3 |
| 内存标准 | 不引入与输入规模线性相关的额外 Device 内存拷贝，保持与原 Tensor 接口一致的内存使用语义 | 任务书 3.4 |

说明：本文件为 **开发前设计文档**，上表为验收口径与计划验证项，不含实测数据；实测结论在自测报告中给出。若改造波及非本册
接口（如样例中调用的他册 API），须证明其 **不因本册改动而回归失败**（任务书 3.5 第 4 条）。`include/basic_api` /
`impl/basic_api` 目录结构不变，新增重载与原重载成对可检索；注释按仓库要求使用英文；开发中发现的文档歧义、接口缺陷或 API
设计不合理之处，须按任务书第 4 节以 Issue 形式提交至 `asc-devkit`（标题格式「【AscendC CAPI社区任务】xxx」）并回填归档链接。

## 兼容性分析

| 兼容维度 | 结论 |
| --- | --- |
| 源码 / 语义兼容 | 原 Tensor 调用方无需修改；指针重载为纯增量，原重载保留并走原 `*Impl`，数值行为与改造前一致 |
| 模板 / ABI | 新增重载受 `enable_if` 约束，不改变既有模板推导结果与实例化数量；改动落在模板头文件与 `inline` 接口层，不改变既有符号 ABI |
| 协同 / 混用 / 平台 | `GetUnderlyingPtr` 可与 DMA / CUBE 册共用，接口范围不重叠、无合入顺序依赖；允许指针与 Tensor 混用，`PrimType` 不一致时编译期失败；面向 Ascend 950 系列，CANN 9.0.0 ~ 9.1.0 |

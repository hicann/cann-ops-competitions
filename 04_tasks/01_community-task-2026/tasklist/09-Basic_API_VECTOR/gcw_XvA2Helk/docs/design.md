# Ascend C Basic API 指针化扩展（VECTOR / 矢量计算分册）设计文档

# 一、需求描述

## 1.1 需求来源

CANN 社区任务《9月社区任务-AscendC Basic_API优化实现(VECTOR矢量接口扩展)》要求：基于 Ascend C
Basic API，对 VECTOR / 矢量计算分册所列接口进行指针化扩展，使开发者既可继续传入 `LocalTensor`，
也可直接传入裸指针（如 `__ubuf__ T*`）调用同一套对外 API。

交付仓为 asc-devkit，变更位于 `include/basic_api`（对外声明）与 `impl/basic_api`（实现），
自验证基于仓内 `examples/01_simd_cpp_api/03_basic_api/` 官方样例改造完成。目标硬件为
Ascend 950 系列（`__NPU_ARCH__ == 3510`），CANN 9.0.0 ~ 9.1.0。

本册范围为类型码 **V** 的矢量计算类接口，以及排序类 **N** 中已纳入 VECTOR 清单的项
（`MrgSort`、`Sort`、`Transpose` 等）。范围以任务配套清单 `api_list_vector` 为准，
共 **256 个重载签名 / 85 个接口名**（任务书正文表述为 245 / 79，差异见 2.2.3）。

## 1.2 需求分析

### 现状

本册接口当前将矢量操作数固定声明为具体的 Tensor 包装类型，以一元族 `Exp` 为例：

```cpp
template <typename T, bool isSetMask = true, const ExpConfig& config = DEFAULT_EXP_CONFIG>
__aicore__ inline void Exp(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                           uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
```

其实现体在剥离调试与观测分支后，实质只有一行——把 Tensor 取址后转交底层 `*Impl`：

```cpp
using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    ... CheckFunVecBinaryScalar(dst, src, ...) ...
#endif
#ifdef __MSTX_DFX_REPORT__
    ... MstxTensor::GetMstxVecUnaryInfo<T, T, isSetMask>(dst, src, ...) ...
#endif
ExpImpl<PrimType, isSetMask, config>((__ubuf__ PrimType*)dst.GetPhyAddr(),
                                     (__ubuf__ PrimType*)src.GetPhyAddr(), mask, repeatTime, repeatParams);
```

即：**接口层只做「取底层指针 + 透传」，数值语义完全由 `*Impl` 承载**。这决定了指针化的
本质是入参类型分发的编译期改造，而非计算逻辑改造。

### 目标与约束

| 编号 | 约束 | 来源 |
| --- | --- | --- |
| C1 | 原 `LocalTensor` 调用方式零语义变更，原有用例回归须全部通过 | 任务书 2.1.2 / 2.4.1 |
| C2 | 同一调用中各操作数可独立选择指针或 Tensor（混用） | 任务书 2.4.1 |
| C3 | 不得擅自变更 `*Impl` 数值语义 | 任务书 3.4 |
| C4 | 指针扩展为编译期入参类型适配，不引入与规模线性相关的额外 Device 内存拷贝 | 任务书 3.4 |
| C5 | 仅修改本册清单内接口，不改动 VECTOR 以外分册的专属符号 | 任务书 2.4 边界约束 |

其中 **C1 是本设计的第一约束**。仓内 `tests/api/basic_api/` 现有 **414 个测试文件**，按
`ascendc_case_ascend950pr_9599` / `ascend910` / `ascend310p` / `ascend610` 等十余个 SoC 目录组织，
另有 `ascendc_header_checker` / `ascendc_host_header_checker` 两组头文件规范检查。本册改造触及
公共头文件中的 256 个签名，回归面覆盖全仓，因此方案必须使「原 Tensor 路径不受影响」成为
**可论证的结构性结论**，而不是依赖测试覆盖去事后发现。

此外，仓内代码大量使用显式模板实参调用，例如
`Cast<float, int32_t, RoundMode::CAST_ROUND>(...)`、`Select<T, uint8_t, false, config>(...)`、
`Add<float, AscendC::Reg::MaskMergeMode::MERGING>(...)`。**因此既有模板参数的顺序与含义不可变更**，
任何新增模板参数只能追加在参数列表末尾且必须可推导。

# 二、方案设计

## 2.1 接口内部实现

### 2.1.1 公共设施

新增三项公共设施，均位于 `impl/basic_api/utils/`（`TypeUtils` 现所在处），
为纯新增，不修改任何既有符号：

```cpp
namespace AscendC {
namespace TypeUtils {

// ① 指针判定：与 IsLocalTensorType 对称，同样套用元素类型白名单
template <typename T>
__aicore__ constexpr bool IsPtrType()
{
    if constexpr (Std::is_pointer_v<T>) {
        return SupportType<
            Std::remove_cv_t<Std::remove_pointer_t<T>>, bool, int8_t, uint8_t, int16_t, uint16_t, half,
            bfloat16_t, float, fp8_e5m2_t, fp8_e4m3fn_t, fp8_e8m0_t, int32_t, uint32_t, int64_t, uint64_t,
            complex32, complex64, double>();
    } else {
        return false;
    }
}

// ② 「Tensor 或裸指针」判定
template <typename T>
__aicore__ constexpr bool IsTensorOrPtr() { return IsLocalTensorType<T>() || IsPtrType<T>(); }
template <typename T, typename U>
__aicore__ constexpr bool IsTensorOrPtr() { return IsTensorOrPtr<T>() && IsTensorOrPtr<U>(); }

// ③ 统一的元素类型萃取：Tensor 取 PrimType，裸指针取 remove_pointer
template <typename U, typename = void> struct ElemOf { using type = typename U::PrimType; };
template <typename U> struct ElemOf<U, Std::enable_if_t<Std::is_pointer_v<U>>>
{ using type = Std::remove_cv_t<Std::remove_pointer_t<U>>; };
template <typename U> using ElemOf_t = typename ElemOf<Std::remove_cvref_t<U>>::type;

} // namespace TypeUtils

// ④ 统一的底层指针萃取（任务书 2.2 指定名称）
template <typename U>
__aicore__ inline auto GetUnderlyingPtr(const U& val)
{
    if constexpr (Std::is_pointer_v<U>) { return val; }
    else { return val.GetPhyAddr(); }
}
} // namespace AscendC
```

**`IsPtrType` 必须复用与 `IsLocalTensorType` 完全相同的 `SupportType` 白名单**，这一点不可省略：
`IsLocalTensorType` 并非仅判定「是否为 LocalTensor」，其实现在确认类型后还会对
`T::PrimType` 施加受支持元素类型白名单校验。若指针分支只做 `Std::is_pointer_v` 判定，
指针路径将接受 Tensor 路径拒绝的元素类型，**构成检查强度倒退**。本设计令两条路径对元素类型的
接受集合逐项相同。

所用 trait 均取自仓内 `AscendC::Std`（`include/utils/std/type_traits.h`），
`is_pointer` / `remove_pointer` / `remove_cv` / `remove_cvref` / `enable_if` / `is_same` / `is_void`
均为其既有能力，不引入宿主标准库依赖。

`IsLocalTensorType` 为仓内 `TypeUtils` 既有能力（现有 127 处调用），本设计在其旁并列扩展，
不改变其行为。

### 2.1.2 统一改造范式

所有接口按同一范式改造：**既有具名模板参数原样保留在前，可推导的操作数类型参数追加在末尾，
函数参数由具体 Tensor 类型改为对应的模板参数**。以 `Exp` 的 count 形式为例：

```cpp
// 改造前
template <typename T, const ExpConfig& config = DEFAULT_EXP_CONFIG>
__aicore__ inline void Exp(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count);

// 改造后
template <typename T = void, const ExpConfig& config = DEFAULT_EXP_CONFIG, typename U, typename S>
__aicore__ inline void Exp(const U& dst, const S& src, const int32_t& count)
{
    static_assert(TypeUtils::IsTensorOrPtr<U, S>(), "operands must be LocalTensor or raw pointer");
    using ActualT = TypeUtils::ElemOf_t<U>;
    static_assert(Std::is_void_v<T> || Std::is_same_v<T, ActualT>,
                  "explicit template argument T must match deduced element type");
    static_assert(Std::is_same_v<ActualT, TypeUtils::ElemOf_t<S>>, "dst/src PrimType must match");
#if ASCENDC_CPU_DEBUG
    ...（原调试分支，见 2.1.4）
#endif
    ExpImpl<ActualT, config>((__ubuf__ ActualT*)GetUnderlyingPtr(dst),
                             (__ubuf__ ActualT*)GetUnderlyingPtr(src), count);
}
```

该范式同时满足全部五条约束：

- **C1**：`U`/`S` 追加在末尾且可推导，`Exp<float>(...)`、`Exp<float, DEFAULT_EXP_CONFIG>(...)`
  等既有显式实参调用完全不受影响；传入 `LocalTensor` 时 `ElemOf_t<U>` 即原 `PrimT<T>`，
  `GetUnderlyingPtr` 即原 `GetPhyAddr()`，**转交给 `*Impl` 的实参逐字节等价**。
- **C2**：`U` 与 `S` 独立推导，天然支持任意组合的混用。
- **C3**：`*Impl` 一行不改。
- **C4**：全部为编译期分发，`if constexpr` 两分支各自退化为原有直取，无运行期开销、无额外拷贝。
- **C5**：改造范围严格限定在本册清单，跨册符号处理见 2.2.3。

新增的第三条 `static_assert` 是增量收益：原接口以 `LocalTensor<T>` 同时出现在 dst/src 位置，
类型一致性由类型系统隐式保证；改为独立模板参数后需显式约束，否则混用时可能静默接受
元素类型不一致的组合。

### 2.1.3 仓内既有先例

本范式并非新创，而是与仓内既有实现对齐。`kernel_operator_vec_binary_scalar_intf.h` 中
`Adds` 的 RegBase 变体已采用完全相同的形态：

```cpp
template <typename T = BinaryDefaultType, bool isSetMask = true,
          const BinaryConfig& config = DEFAULT_BINARY_CONFIG, typename U, typename S, typename V>
__aicore__ inline void Adds(const U& dst, const S& src0, const V& src1, uint64_t mask[],
                            const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
```

其 `AddsCommon` 内部通过 `TypeUtils::IsLocalTensorType<...>()` 与 `if constexpr` 分发，
并以 `using ActualT = typename U::PrimType;` 萃取元素类型。本设计对这类接口只需三处替换：

| 位置 | 改造前 | 改造后 |
| --- | --- | --- |
| 断言 | `static_assert(TypeUtils::IsLocalTensorType<U>())` | `static_assert(TypeUtils::IsTensorOrPtr<U>())` |
| 类型萃取 | `using ActualT = typename U::PrimType` | `using ActualT = TypeUtils::ElemOf_t<U>` |
| 取址 | `(__ubuf__ ActualT*)dst.GetPhyAddr()` | `(__ubuf__ ActualT*)GetUnderlyingPtr(dst)` |

**这类接口的函数签名与重载数完全不变**，是回归风险最低的一类。

### 2.1.4 调试与观测分支的处理

接口层的检查与观测调用全部位于条件编译分支内。以 `kernel_operator_vec_unary_intf_impl.h`
为例，统计其全部检查调用 **80 处、观测调用 40 处，位于 `#if` 分支外的为 0 处**：

```cpp
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dst, src, ...)) { ASCENDC_REPORT_CHECK_ERROR(...); }
#endif
#ifdef __MSTX_DFX_REPORT__
    MstxTensor::GetMstxVecUnaryInfo<T, T, isSetMask>(dst, src, ...);
#endif
```

这两类调用消费的是 Tensor 携带的形状与越界信息，裸指针在语义上不提供该信息。
且检查函数是**把本次调用的全部操作数一并接收**的（如
`CheckFunVecBinaryScalar(dst, src, scalar, mask, repeatTime, repeatParams, "Exp")`），
无法按操作数拆分成「可校验的部分仍校验」，因此只能以整次调用为粒度取舍。处理方式：

- **全部操作数为 Tensor**：调试与观测分支保持原样，行为与能力零变更。
- **任一操作数为裸指针**：以 `if constexpr (TypeUtils::IsLocalTensorType<...>())` 包裹这两段，
  本次调用跳过检查与观测。

该取舍的依据是：若强行为指针路径构造等价检查，需要调用方额外传入缓冲区长度等参数，
将改变对外接口形态，与「仅扩展指针输入能力」的任务目标冲突。**产品编译路径
（未定义 `ASCENDC_CPU_DEBUG` / `__MSTX_DFX_REPORT__`）下两条路径生成的代码完全一致**，
故该差异不影响正式交付能力；差异本身在接口 README 与本文档 3.2 中明确声明。

## 2.2 接口设计

### 2.2.1 改造工作分解

对配套清单 `api_list_vector` 的 256 个重载按**签名形状**归并（形状指去除参数名与默认值后
的参数类型序列），得到 **63 种形状**，分布高度集中：

| 重载数 | 接口数 | 签名形状 | 代表接口 |
| --- | --- | --- | --- |
| 40 | 18 | `T,T,T,uint64_t,uint8_t,BinaryRepeatParams` | Add / Div / Max / Min / Mul / Sub / And / Or … |
| 28 | 10 | `T,T,uint64_t,uint8_t,UnaryRepeatParams` | Abs / Exp / Ln / Not / Reciprocal / Relu / Rsqrt / Sqrt … |
| 21 | 19 | `T,T,T,int32_t` | 同二元族 count 形式 |
| 15 | 9 | `T,T,int32_t` | 同一元族 count 形式 |
| 14 | 7 | `T,T,U,uint64_t,uint8_t,UnaryRepeatParams` | Adds / Axpy / LeakyRelu / Maxs / Mins / Muls / ShiftLeft |
| 12 | 6 | `T,T,T,uint64_t,uint8_t,UnaryRepeatParams` | 同二元标量族 |
| 8 | 8 | `T,T,T,uint32_t` | AbsSub / ExpSub / Prelu / MulCast … |
| 8 | 8 | `T,T,U,int32_t` | 同二元标量族 count 形式 |
| 8 | 4 | `U,S,V,uint64_t,uint8_t,UnaryRepeatParams` | Ands / Divs / Ors / Subs（**已泛型，见 2.1.3**） |

上述 9 种形状覆盖 **154 个重载（60%）**；另有 8 种形状（3~7 个重载）覆盖 36 个，
其余 46 种为 1~2 个重载的长尾，覆盖 66 个。

按每条签名需指针化的操作数个数统计：0 个 13 条、1 个 23 条、2 个 119 条、3 个 94 条、4 个 7 条。
其中「0 个」即 2.1.3 所述已泛型接口。

### 2.2.2 落地阶段划分

| 阶段 | 内容 | 累计覆盖 | 验证目标 |
| --- | --- | --- | --- |
| 一 | 公共设施 + 已泛型接口（`U,S,V` 形状） | 13 条 | 打通范式，验证回归零影响 |
| 二 | 前 8 种高频形状（二元族、一元族、二元标量族） | 154 条（60%） | 批量套用范式，回归全量通过 |
| 三 | 中频 8 种形状 | 190 条（74%） | — |
| 四 | 长尾 46 种形状（含 Select / Gather / Scatter / Reduce\* / Compare / Transpose / VectorPadding / BilinearInterpolation 等） | 254 条（100%，`GetAbsAddr` 2 条除外，理由见 2.2.3） | 逐个套用同一范式 |

阶段一同时是**风险闸门**：该阶段不新增任何重载、不改动任何函数签名，若全量回归通过，
即证明公共设施本身对既有路径零影响；其后各阶段的风险仅来自签名改造，且形态同构。

### 2.2.3 范围界定与待确认项

清单与任务书正文存在若干需明确的边界，本设计的处理方式如下，并同步以 Issue 形式反馈至
asc-devkit 请官方确认（对应交付件 4）：

| 条目 | 情况 | 本设计处理 |
| --- | --- | --- |
| `Copy`（3 个重载） | 属本册清单，但位于 `kernel_operator_data_copy_intf.h`（DMA 分册头文件）；该接口带 `__inout_pipe__(V)` 且本册测试样例含 `copy_ub2ub` | **纳入本册**，在 PR 描述中说明跨文件原因，改动严格限定于 `Copy` 的 3 个重载，不触及 DMA 分册的 `DataCopy` 系列 |
| `Fill`（1 个重载） | 属本册清单，但位于 `kernel_operator_mm_intf.h`（CUBE 分册头文件） | 同上，仅改动 `Fill` 自身 |
| `GetAbsAddr`（2 个重载） | 位于 `kernel_tpipe.h`，签名为 `uint64_t GetAbsAddr(TPipe*, const LocalTensor<T>&)`，功能是由 Tensor 取其绝对地址 | **排除，不纳入本册交付**。该接口的职责即「从 Tensor 求地址」，当入参已是裸指针时其语义自消解——调用方持有的即是地址本身，无需再经此接口换取。为其提供指针重载只能是恒等返回，属无意义接口膨胀。该判断同步以 Issue 反馈，请官方确认清单收录范围 |
| `MrgSort` | 入参 `const MrgSortSrcList<T>& sortList` 为**含 Tensor 的结构体**，而非 Tensor 本身 | 提供平行的指针版结构体 `MrgSortSrcListPtr<T>`，或令 `MrgSortSrcList` 的成员类型模板化；具体方案待官方确认后定稿 |
| 数量口径 | 配套清单为 256 个重载 / 85 个接口名，任务书正文表述为 245 / 79 | **以配套清单为准**，覆盖除 `GetAbsAddr`（2 条）外的 254 条；差异同步反馈 |

## 2.3 测试用例设计

### 2.3.1 精度自验证

依任务书 3.5，自验证基于仓内官方样例改造完成，不新增独立样例工程。以
`examples/01_simd_cpp_api/03_basic_api/01_memory_vector_compute/` 下样例为基础，
将 `GlobalTensor` / `LocalTensor` 写法替换为裸指针写法，通过配套 `script/gen_data.py` 校验精度。

| 项 | 取值 |
| --- | --- |
| 数据取值范围 | [-100, 100] |
| 数据 Shape | 1、32、1024、2048 |
| 判定标准 | 生态算子开源精度标准（实验标准） |
| 对照基线 | 同输入下改造前的 Tensor 路径结果 |

由于指针路径与 Tensor 路径最终调用同一 `*Impl`、且传入实参逐字节等价，两条路径的数值结果
**应为位一致**；自验证以此为预期，任何差异均视为改造缺陷而非精度误差。

### 2.3.2 用例矩阵

每个签名形状至少覆盖下列组合：

| 用例类别 | 说明 |
| --- | --- |
| Tensor 路径回归 | 改造前后同一调用的结果位比对 |
| 纯指针路径 | 全部操作数为 `__ubuf__ T*` |
| 混用路径 | dst 指针 / src Tensor，以及 dst Tensor / src 指针 |
| 显式模板实参 | 以 `F<T, ...>(...)` 形式调用，验证既有写法不受影响 |
| 负向用例 | dst 与 src 元素类型不一致时须编译期拦截（`static_assert` 触发） |

### 2.3.3 性能

依任务书 3.3，本任务无性能指标与标杆时延要求。自测报告中将说明改造相对改造前无性能回退：
改造为编译期分发，`if constexpr` 两分支分别退化为原有直取，产品编译路径下不产生额外指令。

# 三、可维可测

## 3.1 精度标准 / 性能标准

- **精度**：对标生态算子开源精度标准（实验标准）。因 `*Impl` 零改动且实参等价，
  指针路径与 Tensor 路径预期位一致，以位比对作为自验证判据（强于标准要求的容差判据）。
- **性能**：无标杆要求。改造不引入运行期分支、不引入额外 Device 内存拷贝，
  符合任务书 3.4 对内存与兼容性的约束。

## 3.2 兼容性分析

### 3.2.1 对既有 Tensor 路径的影响

本设计使「原路径不受影响」成为结构性结论，论证如下：

1. **模板参数列表**：既有具名参数的顺序、名称、默认值全部保留，新增参数一律追加在末尾且
   可从函数实参推导。因此所有显式模板实参调用（含仓内 `Cast<float, int32_t, RoundMode::...>`、
   `Select<T, uint8_t, false, config>` 等写法）的解析结果不变。
2. **实参等价**：传入 `LocalTensor<T>` 时，`TypeUtils::ElemOf_t<U>` 求值为 `T::PrimType`
   （即原 `PrimT<T>`），`GetUnderlyingPtr(dst)` 求值为 `dst.GetPhyAddr()`。转交 `*Impl`
   的实参与改造前逐字节相同。
3. **无二义引入**：`LocalTensor` 未定义向裸指针的隐式转换（`kernel_tensor.h` 中仅提供
   `GetPhyAddr()`，无 `operator T*()`），故 Tensor 实参与指针实参之间不存在转换通路，
   不会产生新的重载歧义。
4. **已泛型接口零签名变更**：2.1.3 所列接口仅改动实现体内三行，重载集合完全不变。

### 3.2.2 回归验证范围

`tests/api/basic_api/` 现有 414 个测试文件，覆盖 `ascend950pr_9599`、`ascend910`、
`ascend310p`、`ascend610`、`ascend310b1`、`ascend610lite` 等 SoC 目录及两组头文件检查。
本册改造须在全部目标上回归通过。阶段一（不改签名）作为回归闸门先行验证。

### 3.2.3 方案验证状态

上述范式已在主机侧以独立最小工程完成编译期验证（`g++ -std=c++17 -Wall -Wextra`，零告警），
覆盖 **11 个正向分发点**——原有 Tensor 隐式推导调用、两种显式模板实参形式、纯指针调用、
双向混用（dst 指针 / src Tensor 与 dst Tensor / src 指针）、指针叠加显式模板实参——
全部落到预期重载。

另有两类编译期校验：dst 与 src 元素类型不一致时正确触发 `static_assert` 并给出明确诊断
（诊断信息为 `dst/src PrimType must match`）；以及 `IsPtrType` 与 `IsLocalTensorType`
对元素类型接受集合的等价性校验——对受支持类型两者同时为真，对白名单外类型两者同时为假。

主机侧验证不覆盖的部分，将在拿到目标环境后优先确认：

- `__ubuf__` / `__gm__` 地址空间限定符在毕昇 ASC 编译器下参与重载决议的规则；
- NPU 分支中 `GetPhyAddr()` 的返回类型（`kernel_tensor.h` 中存在返回 `uint64_t` 的重载），
  以确认 `GetUnderlyingPtr` 的返回类型在两条路径下的统一方式。

## 3.3 其他

改造过程中发现的文档与接口问题，按任务书第 6 节规范以 Issue 形式提交至 asc-devkit，
并按交付件 4 要求归档链接。当前已识别项见 2.2.3。

# Ascend C Basic API 指针化扩展（DMA · 数据搬运 / 缓存）设计文档

| 项 | 内容 |
| --- | --- |
| 任务 | 社区任务 - Ascend C Basic API 指针化扩展（DMA · 数据搬运 / 缓存） |
| 目标仓 | [`cann/asc-devkit`](https://gitcode.com/cann/asc-devkit) · `master` |
| 变更目录 | `include/basic_api/`、`impl/basic_api/`、`tests/api/basic_api/`、`examples/01_simd_cpp_api/03_basic_api/` |
| 适配硬件 | Ascend 950 系列（Ascend950PR / Ascend950DT，`dav-3510`） |
| CANN 版本 | 9.0.0 ~ 9.1.0（**指针路径实际生效区间见 §3.2，为 ≥ 9.1.0**） |
| 开发语言 | Ascend C / C++17，bisheng（ASC）编译器 |
| 基线 commit | `626c945f943983394d9c6e802021e1f2691ebee6` |
| 团队 | KaranocaVe |

---

# 一、需求描述

## 1.1 需求来源

Ascend C Basic API 当前对外只接受 `LocalTensor<T>` / `GlobalTensor<T>` 包装类型。对于以下场景，Tensor 包装带来了不必要的摩擦：

1. **算子模板库 / 代码生成场景**：CATLASS、算子 JIT 等上层框架内部已经按硬件缓冲区维护裸地址，为了调用一次 `DataCopy` 需要先构造一个临时 `LocalTensor`，再由接口内部通过 `GetPhyAddr()` 把地址取回来——一个纯粹多余的来回。
2. **与 C-API / SIMT 代码互操作**：`examples/02_simd_c_api/` 一侧本来就是裸指针风格，跨边界时必须做 Tensor 包装转换。
3. **精细控制场景**：开发者已经自行完成缓冲区切分与同步（`Mutex::Lock<PIPE_MTE2>` 等），Tensor 携带的 position / size 信息对其无增量价值，却要求构造开销与样板代码。

任务要求：在**完全保持原有 Tensor 接口功能与兼容性**的前提下，为 DMA 类 Basic API 增加裸硬件指针（`__ubuf__` / `__gm__` / `__cbuf__` 等）入参能力。

目标编程范式（任务书 §1）：

```cpp
template <uint32_t blockLength>
__vector__ __global__ void dma_ptr_kernel(__gm__ float* x, __gm__ float* y, __gm__ float* z)
{
    AscendC::InitSocState();
    __ubuf__ float xPtr[blockLength];
    __ubuf__ float zPtr[blockLength];
    auto offset = block_idx * blockLength;
    AscendC::Mutex::Lock<PIPE_MTE2>(0);
    AscendC::DataCopy(xPtr, x + offset, blockLength);      // <- 裸指针直调
    AscendC::Mutex::Unlock<PIPE_MTE2>(0);
    // ... 计算 ...
    AscendC::Mutex::Lock<PIPE_MTE3>(0);
    AscendC::DataCopy(z + offset, zPtr, blockLength);
    AscendC::Mutex::Unlock<PIPE_MTE3>(0);
}
```

## 1.2 需求分析

### 1.2.1 改造范围（DMA 分册，总表类型码 D）

按 `include/basic_api/` 源码实测，本册覆盖 **5 个 API 名**：

| API 名 | 源码重载数 | 声明位置 |
| --- | --- | --- |
| `DataCopy` | 27 | `kernel_operator_data_copy_intf.h` |
| `DataCopyPad` | 7 | `kernel_operator_data_copy_intf.h` |
| `DataCopyL1ToUB` | 2 | `kernel_operator_data_copy_intf.h` |
| `DataCachePreload` | 1 | `kernel_operator_cache_intf.h` |
| `DataCacheCleanAndInvalid` | 3 | `kernel_operator_cache_intf.h` |
| **合计** | **40** | |

任务书 §1 给出的是 **44 个重载签名**。差值来自 `#if (__NPU_ARCH__ == 3510) ... #else ... #endif` 双分支声明被分别计数（`Nd2NzParams` 1 对 + `DataCopyPad` 4 对 = 5 对）。本设计以**源码中实际可被调用的逻辑重载**为准做改造，按 arch 分支展开后自然覆盖腾讯文档 `basic_api_list_dma_api` 的 44 行；合入前将与该清单逐条核对并在 PR 描述中给出对照表。

**明确不在本册范围**（不修改）：

- `Copy`（`__inout_pipe__(V)`，属 VECTOR 分册）
- `SetPadValue`、`NdDmaDci`、`SetLoopModePara`、`ResetLoopModePara`、`ICachePreLoad`、`GetICachePreloadStatus`——无 Tensor 入参，不存在"指针化"语义

### 1.2.2 现状分层（决定了改造只落在一层）

| 层 | 文件 | 当前入参 |
| --- | --- | --- |
| ① 对外接口层 | `include/basic_api/kernel_operator_data_copy_intf.h`、`kernel_operator_cache_intf.h` | `LocalTensor<T>&` / `GlobalTensor<T>&` |
| ② 中间分发层 | `impl/basic_api/kernel_operator_data_copy_intf_impl.h`、`kernel_operator_data_copy_base_impl.h` | 同上；内部取 position、强转指针 |
| ③ 叶子 Impl 层 | `impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h` 等 | **已经是裸硬件指针** |

叶子层签名实证（`dav_3510/kernel_operator_data_copy_impl.h:249,278,595,608`）：

```cpp
template <typename T> __aicore__ inline void DataCopyGM2UBImpl(
    __ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams, const uint8_t cacheMode = 0);
template <typename T> __aicore__ inline void DataCopyGM2L1Impl(
    __cbuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams, const uint8_t cacheMode = 0);
template <typename T> __aicore__ inline void DataCopyUB2GMImpl(
    __gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams, const uint8_t cacheMode = 0);
template <typename T> __aicore__ inline void DataCopyUB2UBImpl(
    __ubuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams);
```

② 层今天做的事（`kernel_operator_data_copy_intf_impl.h:51`）：

```cpp
const Hardware dstHWPos  = GetPhyType((TPosition)dst.GetPosition());  // 运行期读 position
const uint8_t  cacheMode = ExtractCacheMode(src);
if (dstHWPos == Hardware::UB) {
    DataCopyGM2UBImpl((__ubuf__ PrimType*)dst.GetPhyAddr(), (__gm__ PrimType*)src.GetPhyAddr(), p, cacheMode);
} else if (dstHWPos == Hardware::L1) {
    DataCopyGM2L1Impl((__cbuf__ PrimType*)dst.GetPhyAddr(), (__gm__ PrimType*)src.GetPhyAddr(), p, cacheMode);
}
```

> **结论**：③ 层零改动；改造是在 ① 层补一组"不绕 Tensor、直达 ③ 层"的入口。
> 指针路径上目的地由地址空间在**编译期**确定，上面的运行期分支直接消失——指针路径比
> Tensor 路径少一次 position 读、少一个分支，**性能只会更好，不会回退**。

### 1.2.3 需求拆解

| 编号 | 子项 | 产出 |
| --- | --- | --- |
| R1 | 地址空间萃取基础设施（`IsUbufPtr` / `IsGmPtr` / `SpaceOf` / `GetUnderlyingPtr`） | `impl/basic_api/utils/kernel_ptr_traits.h`（新增，与 VECTOR 册共用） |
| R2 | `DataCopy` 27 个重载的指针形态 | `kernel_operator_data_copy_intf.h` + `_impl.h` |
| R3 | `DataCopyPad` 7 个重载的指针形态 | 同上 |
| R4 | `DataCopyL1ToUB` 2 个重载的指针形态 | 同上 |
| R5 | `DataCachePreload` / `DataCacheCleanAndInvalid` 4 个重载的指针形态 | `kernel_operator_cache_intf.h` + `_impl.h` |
| R6 | CANN 9.0.0 兼容门控（见 §3.2） | 版本宏 |
| R7 | 样例迁移 + 精度自测 | `examples/01_simd_cpp_api/03_basic_api/`、`tests/api/basic_api/` |

---

# 二、方案设计

## 2.1 接口内部实现

### 2.1.1 关键前提：地址空间限定符的编译器行为（已实测）

本方案成立与否，完全取决于 `__ubuf__` / `__cbuf__` / `__cc__` 等是否**参与 C++ 类型系统**。
为此编写探针 `probe_addrspace.asc`（22 条 `static_assert` + 重载 / 偏特化 / 显式模板实参用例），在两台机器四种组合下实测：

| 机器 | CANN | `--npu-arch` | 结果 |
| --- | --- | --- | --- |
| DevEnv_237574（910B4） | 9.0.0 | `dav-2201` | ❌ 15 errors |
| DevEnv_237574（910B4） | 9.0.0 | `dav-3510` | ❌ 11 errors |
| DevEnv_907601（950PR） | 9.1.0 | `dav-2201` | ✅ 通过 |
| DevEnv_907601（950PR） | 9.1.0 | `dav-3510` | ✅ `[100%] Built target probe` |

**结论：决定因素是 CANN / bisheng 版本，与 `--npu-arch` 无关。**

- **CANN 9.1.0**：这些限定符是**真正的 address space**，完整进入类型系统——
  函数重载可区分、模板偏特化可区分、`Std::is_same` 可区分，且指针重载与 Tensor 重载
  可并存、各自命中、互不干扰，显式模板实参 `Tmpl<float,2>(cbufPtr, ubufPtr)` 亦按地址空间正确选重载。
- **CANN 9.0.0**：同样的宏展开为 `__attribute__((cce_unif_buff))` 等普通属性，编译器在类型位置
  **直接丢弃**（`warning: 'cce_unif_buff' attribute ignored when parsing type`）。
  于是 `__ubuf__ float*` ≡ `__gm__ float*` ≡ `float*`，按地址空间重载会直接
  `error: redefinition of 'Which'`。

> **这条实测结果是本设计的地基**，也直接决定了 §3.2 的兼容性门控方案。

由此得到两条必须遵守的实现细节：

1. **`__gm__` 是 generic / 默认地址空间，不是专用空间。**
   `template<typename T> struct X<__gm__ T*>` 会同时吃到 `__ubuf__ T*`。判据必须用排除法：

   ```cpp
   template <typename P> struct IsGmPtr {
       static constexpr bool value = IsAnyPtr<P>::value &&
           !IsUbufPtr<P>::value && !IsCbufPtr<P>::value && !IsCcPtr<P>::value &&
           !IsCaPtr<P>::value   && !IsCbPtr<P>::value   && !IsFbufPtr<P>::value;
   };
   ```

2. **地址空间挂在 pointee 上。** `ElemOf<__gm__ float*>::type` 是 `__gm__ float` 而非 `float`
   （命中通用 `T*` 偏特化）；而 `ElemOf<__ubuf__ T*>` 这类显式偏特化会把空间剥掉。
   萃取元素类型时 GM 分支需单独处理，否则 `Std::is_same<PrimT<T>, U>` 之类的既有 SFINAE 会失配。

### 2.1.2 为什么不能照抄任务书 §2.3 的范式

任务书给出的示意是：

```cpp
template <typename U> __aicore__ inline auto GetUnderlyingPtr(const U& val) {
    if constexpr (std::is_pointer_v<U>) { return val; } else { return val.GetPhyAddr(); }
}
template <typename T, typename U>
__aicore__ inline void DataCopy(const T& dst, const U& src, const uint32_t count) {
    DataCopyImpl(GetUnderlyingPtr(dst), GetUnderlyingPtr(src), count);
}
```

该范式有**两处与现有代码不相容**，均已在源码中核实：

**(a) `LocalTensor<T>::GetPhyAddr()` 在 NPU 模式下返回 `uint64_t`，不是指针。**

`include/basic_api/kernel_tensor.h`：

| 类 | 模式 | 返回类型 |
| --- | --- | --- |
| `LocalTensor<T>` | NPU（L174） | **`uint64_t`** |
| `LocalTensor<T>` | `ASCENDC_CPU_DEBUG`（L157） | `PrimType*`（无地址空间） |
| `GlobalTensor<T>` | 两者（L267） | `const __gm__ PrimType*` |

`GetUnderlyingPtr(localTensor)` 因此既不能喂给 `__ubuf__ T*` 形参，也丢失了区分 UB / L1 / L0C 的信息。

**(b) 全泛型 `template<typename T, typename U>` 会破坏显式模板实参调用。**

任务测试样例（即验收目标态）中存在：

```cpp
AscendC::DataCopy<T, 2, dmaConfig>((__ubuf__ T*)..., (__gm__ T*)..., params);
AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>((__ubuf__ T*)..., (__gm__ T*)..., cp, pp);
AscendC::DataCopyPad<int8_t, AscendC::PaddingMode::Compact>(...);
```

若把首个模板参数改成"dst 的类型"，`T` 会被绑成 `__ubuf__ T*` 而非元素类型，
`MultiCopyParams<T,dim>` / `DataCopyPadExtParams<T>` 立刻失配。

> **设计决策**：保持 `template <typename T, ...>` 中 **T = 元素类型**，
> 采用**按地址空间写死形参的指针重载**（`__ubuf__ T*` / `__gm__ T*` / `__cbuf__ T*` / `__cc__ T*` …），
> 而非全泛型 `const T&` / `const U&`。
> 收益：① 显式模板实参照常工作；② 指针重载与 Tensor 重载形参类型不同、不参与同一次竞争，
> **Tensor 路径零改动、零回归风险**。
>
> `GetUnderlyingPtr` 仍然保留并与 VECTOR 册共用，但**仅用于"指针入参直通"**
> （`is_pointer` 分支），不承担 Tensor → 指针的萃取职责。

### 2.1.3 分支覆盖：从地址空间到叶子 Impl 的映射

指针路径以 `(dstSpace, srcSpace)` 二元组在编译期直接选定叶子 Impl，不再有运行期分支：

| dst 空间 | src 空间 | 叶子 Impl | pipe |
| --- | --- | --- | --- |
| `__ubuf__` | `__gm__` | `DataCopyGM2UBImpl` | MTE2 |
| `__cbuf__` | `__gm__` | `DataCopyGM2L1Impl` | MTE2 |
| `__ca__` / `__cb__` | `__gm__` | `DataCopyGM2L0Impl` | MTE2 |
| `__gm__` | `__ubuf__` | `DataCopyUB2GMImpl` | MTE3 |
| `__gm__` | `__cc__` | `DataCopyL0C2GMImpl` | FIX |
| `__ubuf__` | `__ubuf__` | `DataCopyUB2UBImpl` | MTE3 / V |
| `__cbuf__` | `__ubuf__` | `DataCopyUB2L1Impl` | MTE3 |
| `__ubuf__` | `__cbuf__` | `DataCopyL12UBImpl` | MTE2 |
| `__cbuf__` | `__cbuf__` | **见 §2.2.3（歧义）** | — |

## 2.2 接口设计

### 2.2.1 Kernel 侧接口范式

以 Level-2 count 形式为例，改造后的 `include/basic_api/kernel_operator_data_copy_intf.h` 形如：

```cpp
// ---------- 原有 Tensor 重载：一行不改 ----------
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, const uint32_t count);

// ---------- 新增：指针重载（仅 CANN >= 9.1.0 展开） ----------
#if __ASCENDC_ADDRSPACE_IN_TYPE__
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(
    __ubuf__ T* dst, __gm__ T* src, const uint32_t count);

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(
    __gm__ T* dst, __ubuf__ T* src, const uint32_t count);

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(
    __cbuf__ T* dst, __ubuf__ T* src, const uint32_t count);
#endif
```

对应 `impl/basic_api/kernel_operator_data_copy_intf_impl.h`：

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(__ubuf__ T* dst, __gm__ T* src, const uint32_t count)
{
    using PrimType = PrimT<Std::remove_addrspace_t<T>>;
    ASCENDC_ASSERT((count % AscendCUtils::GetC0Count(sizeof(PrimType)) == 0), { /* 同 Tensor 路径 */ });
    struct DataCopyParams repeatParams;
    DataCopyCheck<PrimType>(count, repeatParams);
    DataCopyGM2UBImpl(dst, src, repeatParams, ExtractCacheMode(src));   // 编译期直达，无 position 分支
}
```

要点：

1. **`__inout_pipe__` 属性与 Tensor 重载逐条对齐**，不改流水线语义。
2. **SFINAE 约束原样保留**：dtype 转换类重载（`float→half`、`int32_t→int8_t` 等 7 个
   `DataCopyEnhancedParams` 变体）的 `Std::enable_if<Std::is_same<PrimT<T>, half>::value, ...>`
   照搬到指针版本，只把 `LocalTensor<T>&` 换成对应空间的指针。
3. **`ExtractCacheMode` 复用裸指针重载**（`impl/basic_api/utils/kernel_utils_macros.h:250`
   已有 `ExtractCacheMode(__gm__ T* addr)`），无需新增。
4. **`static_assert` 兜底**：对不合法的空间组合（如 `__cc__ → __cbuf__`）给出明确编译期报错，
   而不是 SFINAE 静默失配后报"no matching function"。

### 2.2.2 Host 侧接口

本任务为 kernel 侧 Basic API 的入参类型扩展，**不涉及 Host 侧接口变更**，无新增 tiling / 原型注册。

### 2.2.3 待评审决策点：`__cbuf__` 无法唯一确定目的硬件

这是本任务唯一需要评审拍板的设计点。

**问题**：L1、BT（Bias Table，`TPosition::C2`）、FIXBUF 三种目的地在样例中**都写作 `__cbuf__`**：

```cpp
// data_copy_gm2l1.asc:276   C1(L1) -> C2(BT)
DataCopy((__cbuf__ outputType*)bias2Local.GetPhyAddr(),
         (__cbuf__ outputType*)bias1Local.GetPhyAddr(), c12c2Params);
// data_copy_l0c2gm.asc:248  L1 -> FIXBUF
DataCopy((__cbuf__ uint64_t*)fbTensor.GetPhyAddr(),
         (__cbuf__ uint64_t*)quantAlphaTensor.GetPhyAddr(), dataCopyParams);
```

Tensor 路径靠 `dst.GetPosition()` 区分（`Hardware::L1` / `BIAS` / `FIXBUF`），
且 Level-2 count 形式下三者 `blockLen` 单位不同（`..._intf_impl.h:884`）：

| dst | `blockLen` 单位 |
| --- | --- |
| UB | `C0Count(sizeof(T))` |
| BIAS(BT) | `dav-3510`: 32B；其余: 64B |
| FIXBUF | 128B |

裸指针丢失该信息。备选方案：

| 方案 | 形态 | 优点 | 缺点 |
| --- | --- | --- | --- |
| **A（推荐）** | 增加可选非类型模板参数<br>`DataCopy<T, TPosition dstPos = TPosition::MAX>(__cbuf__ T*, ...)`；<br>`MAX` 时默认按 L1 处理，需要 BT/FB 时显式写 `DataCopy<half, TPosition::C2>(...)` | 不影响样例中的默认调用形态；显式、可读；无运行期开销 | BT/FB 场景调用方需多写一个模板实参 |
| B | 使用 `__fbuf__` 表达 FIXBUF，并向编译器申请新增 `__bt__` | 最"正确"，完全由类型表达 | 依赖编译器新增限定符，超出本任务范围与周期 |
| C | 运行期从地址值推断 | 无需改调用方 | UB / L1 / BT 均为各自空间内小偏移，**无法可靠区分**；且引入运行期开销 |

**本设计选择方案 A**，并在 PR 中同步提交一个 asc-devkit 易用性 Issue，建议长期演进到方案 B。
若评审倾向 B/其他，本设计据评审结论修订后再进入编码。

### 2.2.4 指针路径的能力边界（需写入接口注释与 README）

| 约束 | 说明 | 处理 |
| --- | --- | --- |
| **SliceInfo 必须带 `shapeValue`** | Tensor 路径在 `shapeValue == 0` 时回退到 `src.GetShapeInfo().shape[i]`（`..._intf_impl.h`），裸指针无 ShapeInfo | 指针重载中 `ASCENDC_ASSERT(srcSliceInfo[0].shapeValue != 0)`，并在文档中写明 |
| **越界检查降级** | `CheckDataCopyTensorSizeOverflow(dst, src, ...)` 依赖 Tensor size | 指针路径跳过该检查，仅保留与 size 无关的对齐 / 参数合法性检查 |
| **CPU DEBUG 校验降级** | `CheckFuncDataCopy(dst, src, ...)` 吃 Tensor | 同上；`ASCENDC_CPU_DEBUG` 下指针路径给出"已降级"提示 |
| **MSTX DFX 降级** | `MstxTensor::GetMstxDataCopyInfo(dst, src, ...)` 吃 Tensor | 指针路径不上报 tensor 信息，仅上报地址与搬运参数 |

> 这是指针化的**固有代价**：调用方主动放弃了 Tensor 携带的 size/shape/position 元信息，
> 也就同时放弃了基于这些元信息的防护。文档中必须明确，避免开发者误以为两条路径等价安全。

## 2.3 测试用例设计

### 2.3.1 精度测试（对拍设计）

**核心方法：同一 kernel 内跑两条路径，逐字节比对。**
对每个被改造接口，构造 `TensorPath(...)` 与 `PtrPath(...)` 两个实现，喂入完全相同的输入，
输出到两块独立 GM，Host 侧比对，并各自与 numpy golden 比对。

- 数据取值范围：`[-100, 100]`（或该接口合法数值域）
- 数据 Shape：`1`、`32`、`1024`、`2048`（矩阵类接口按 M/K/N 等价缩放）
- dtype 覆盖：`half`、`float`、`bfloat16_t`、`int8_t`、`int16_t`、`int32_t`、`uint8_t`
  （按各重载 SFINAE 实际支持范围取交集）
- 对齐覆盖：32B 对齐 / 非对齐（`DataCopyPad` 专项）

### 2.3.2 用例矩阵（基于任务包 6 个样例扩展）

任务包 `test-cases/` 的 6 个样例即验收基准，其头注已声明目标态：

> `DataCopy / LoadData 使用 GetPhyAddr() 指针入参（含 __gm__ / __cbuf__ / __ca__ / __cb__ 强转）；Mmad / Fixpipe 仍使用 LocalTensor / GlobalTensor。`

从中反解出**必须支持的指针重载清单**（合入验收的最小集）：

| # | 签名 | 来源样例 |
| --- | --- | --- |
| 1 | `DataCopy(__ubuf__ T*, __gm__ T*, uint32_t)` | `data_copy_ub2l1:50` |
| 2 | `DataCopy(__gm__ T*, __ubuf__ T*, uint32_t)` | `data_copy_gm2ub_nddma:42` |
| 3 | `DataCopy(__cbuf__ T*, __ubuf__ T*, uint32_t)` | `data_copy_ub2l1:55` |
| 4 | `DataCopy(__ubuf__ T*, __ubuf__ T*, DataCopyParams&)` | `data_copy_ub2l1:138` |
| 5 | `DataCopy(__cbuf__ T*, __gm__ T*, DataCopyParams&)` | `data_copy_gm2l1:121,165` |
| 6 | `DataCopy(__cbuf__ T*, __cbuf__ T*, DataCopyParams&)` | `data_copy_gm2l1:276`(L1→BT)、`data_copy_l0c2gm:248`(L1→FB) ← §2.2.3 |
| 7 | `DataCopy(__cbuf__ T*, __gm__ T*, Nd2NzParams&)` | `data_copy_gm2l1:144,156,217` |
| 8 | `DataCopy(__cbuf__ T*, __gm__ T*, Dn2NzParams&)` | `data_copy_gm2l1:195,207` |
| 9 | `DataCopy(__ubuf__ T*, __gm__ T*, SliceInfo[], SliceInfo[], dim)` | `data_copy_gm2ub_slice:51` |
| 10 | `DataCopy(__gm__ T*, __ubuf__ T*, SliceInfo[], SliceInfo[], dim)` | `data_copy_gm2ub_slice:58` |
| 11 | `DataCopy(__gm__ T*, __cc__ U*, DataCopyCO12DstParams&)`（T≠U） | `data_copy_l0c2gm:270` |
| 12 | `DataCopy<T, dim, cfg>(__ubuf__ T*, __gm__ T*, MultiCopyParams<T,dim>&)` | `data_copy_gm2ub_nddma:58,69,78,87,96` |
| 13 | `DataCopyPad(__gm__ T*, __ubuf__ T*, DataCopyExtParams&)` | `data_copy_pad:67` |
| 14 | `DataCopyPad(__ubuf__ T*, __gm__ T*, DataCopyExtParams&, DataCopyPadExtParams<T>&)` | `data_copy_pad:93,106` |
| 15 | `DataCopyPad<T, PaddingMode::Compact>(__ubuf__ T*, __gm__ T*, ExtParams&, PadExtParams<T>&)` | `data_copy_pad:123` |

余下重载（`DataCopyL1ToUB` ×2、`DataCache*` ×4、`DataCopyEnhancedParams` 系列 ×7 等）
在 `tests/api/basic_api/ascendc_case_ascend950pr_9599/` 下补 UT，同样走双路径对拍。

### 2.3.3 回归测试

- `tests/api/basic_api/` 全量 UT 在 `dav-3510` 与 `dav-2201` 下各跑一遍，要求**与改造前完全一致**。
- `examples/01_simd_cpp_api/03_basic_api/` 下未迁移的样例保持 Tensor 写法，必须继续编译通过、精度通过。

### 2.3.4 性能测试

任务书 §3.3 明确**无性能指标要求**。本设计仍会补一组说明性数据：
对 #1 / #5 / #13 三条典型路径，用 `msprof op` 各采 Tensor 路径与指针路径的时延，
预期指针路径**持平或略优**（少一次 position 读 + 少一个运行期分支，见 §1.2.2）。
结果作为"无性能回退"的佐证写入自测报告，不作为验收项。

---

# 三、可维可测

## 3.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足[生态算子开源精度标准（实验标准）](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)；**且指针路径与 Tensor 路径在相同输入下逐字节一致**（两条路径最终调用同一个叶子 Impl，不存在数值差异来源，比"满足误差阈值"更强） | 任务书 §3.2 |
| 性能标准 | 无指标要求；自测报告中说明无明显回退 | 任务书 §3.3 |
| 内存标准 | 指针化为编译期入参类型适配，**不引入任何与输入规模线性相关的额外 Device 内存拷贝**；指针路径不申请临时 buffer | 任务书 §3.4 |
| 回归标准 | 原有 Tensor 用例 100% 通过 | 任务书 §2.4「兼容性」 |

## 3.2 兼容性分析

### 3.2.1 对既有 Tensor 接口的影响：无

指针重载与 Tensor 重载**形参类型完全不相交**（`__ubuf__ T*` vs `const LocalTensor<T>&`），
不存在隐式转换路径，因此：

- 现有 Tensor 调用点的重载决议结果**不可能改变**；
- 现有 Tensor 重载的函数体**一行不改**；
- 探针用例 E 已实测验证：`Both(lt, gt)` 命中 Tensor 重载、`Both(ub, x)` 命中指针重载，互不干扰。

### 3.2.2 CANN 版本兼容（**红线**）

§2.1.1 实测表明：在 **CANN 9.0.0** 上，按地址空间重载会直接
`error: redefinition of 'DataCopy'`——**不仅指针路径不可用，连原有 Tensor 用例都会编不过**。
任务书要求支持 CANN 9.0.0 ~ 9.1.0，故必须门控：

```cpp
// impl/basic_api/utils/kernel_ptr_traits.h
// __ubuf__ 等限定符自 CANN 9.1.0 起进入 C++ 类型系统；9.0.0 上为 attribute，类型位置被丢弃。
// 在 9.0.0 上展开指针重载会与 Tensor 重载 redefinition，故整体关闭。
#if defined(__CCE_AICORE_VERSION_HAVE_ADDRSPACE_TYPE__) || (__ASCENDC_CANN_VERSION__ >= 90100)
#define __ASCENDC_ADDRSPACE_IN_TYPE__ 1
#else
#define __ASCENDC_ADDRSPACE_IN_TYPE__ 0
#endif
```

门控行为：

| CANN | 指针重载 | Tensor 重载 | 用户可见行为 |
| --- | --- | --- | --- |
| ≥ 9.1.0 | 展开 | 不变 | 两条路径均可用 |
| 9.0.0 | **不展开** | 不变 | 与改造前完全一致；误用指针会得到 `no matching function`，而非 redefinition |

> 具体用哪个宏做判据（编译器内置宏 or CANN 版本宏）需与 asc-devkit 维护者确认；
> 若仓库已有统一的版本探测机制，优先复用。这是**第二个需要评审确认的点**。

### 3.2.3 arch 兼容

指针重载对 `dav-2201` / `dav-3510` **一视同仁**（实测 CANN 9.1.0 下两种 arch 均通过）。
各重载原有的 `#if (__NPU_ARCH__ == 3510) ...` arch 门控**原样保留**，
指针版本与 Tensor 版本置于同一组 `#if` 之内，保证两条路径的 arch 可用集完全相同。

### 3.2.4 与姊妹分册的边界

- 只改 DMA（类型码 D）接口，不触碰 VECTOR / CUBE 分册文件。
- `kernel_ptr_traits.h` 为三册共用的基础设施。为避免 PR 冲突，将在 PR 描述中显式说明该文件的
  共用性质；若 VECTOR 册已先行合入同名文件，本 PR 改为**复用**而非新建。

## 3.3 可测性

- 探针 `probe_addrspace.asc` 作为**编译期能力回归**纳入 `tests/api/basic_api/`：
  若未来编译器行为回退（地址空间再次退化为普通 attribute），该用例会立即编译失败并给出明确断言名。
- 双路径对拍用例（§2.3.1）保证数值等价性可持续验证。

---

# 附录 A · 实测证据索引

| 证据 | 位置 |
| --- | --- |
| 地址空间行为探针源码 | `work/probe_addrspace.asc` |
| 950 / CANN 9.1.0 / dav-3510 构建 | `DevEnv_907601:/root/work/ascdevkit/probe/build` → `[100%] Built target probe` |
| A2 / CANN 9.0.0 / dav-2201 构建 | `DevEnv_237574:/root/work/probe/build` → 15 errors |
| 源码勘察全记录 | `notes/source_findings.md` |
| 基线仓 commit | `cann/asc-devkit@626c945f` |

# 附录 B · 评审待确认项

| # | 事项 | 位置 | 本设计倾向 |
| --- | --- | --- | --- |
| 1 | `__cbuf__` 下 L1 / BT / FIXBUF 的区分方式 | §2.2.3 | 方案 A（可选 `TPosition` 非类型模板参数） |
| 2 | CANN 9.0.0 门控所用的版本判据宏 | §3.2.2 | 复用仓库既有版本探测机制（待维护者指认） |
| 3 | `kernel_ptr_traits.h` 与 VECTOR 册的共用与合入顺序 | §3.2.4 | 谁先合入谁建，后者复用 |
| 4 | 44 vs 40 的重载计数口径 | §1.2.1 | 以腾讯文档 `basic_api_list_dma_api` 为准，PR 中给对照表 |

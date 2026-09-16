# 【社区任务】Ascend C Basic API 指针化扩展（DMA · 数据搬运/缓存）设计文档

# 需求背景（required）

## 需求来源

本需求来源于 CANN 社区任务 2026 的「Ascend C Basic API 指针化扩展（DMA · 数据搬运 / 缓存）」任务。任务要求基于 **asc-devkit** 开源仓 Basic API 中 DMA（数据搬运/缓存）类接口进行指针化扩展，使开发者可以直接使用 `__ubuf__` / `__gm__` / `__cbuf__` 等硬件指针调用 DataCopy、DataCopyPad、DataCachePreload 等搬运与缓存 API，在保持原有 LocalTensor / GlobalTensor 接口兼容的前提下完成设计、开发、自验证全流程工作，验收通过后提交至 asc-devkit 仓 `include/basic_api` 与 `impl/basic_api` 目录。

## 背景介绍

### 现状分析

asc-devkit Basic API 现有 DMA 类接口（DataCopy / DataCopyPad / DataCopyL1ToUB / DataCache 系列）对外入参固定为 `LocalTensor` / `GlobalTensor` 包装类型。当开发者已经持有裸硬件指针（例如手写 tiling 切分场景、与 C API 互操作场景、SIMT 内核与 Basic API 混合场景）时，必须先构造 Tensor 对象再调用搬运接口，存在以下问题：

1. 仅为传参而构造 Tensor 对象引入额外的模板实例化开销与代码冗余；
2. Tensor 对象携带 position、shape 等元信息，在纯指针使用场景下语义冗余；
3. 与 VECTOR / CUBE 姊妹任务中指针化后的算子代码风格不统一。

### 目标范式

任务书给出的目标编程范式（简化示意）：

```cpp
template <uint32_t blockLength>
__vector__ __global__ void dma_ptr_kernel(__gm__ float* x, __gm__ float* y, __gm__ float* z)
{
    AscendC::InitSocState();
    __ubuf__ float xPtr[blockLength];
    __ubuf__ float zPtr[blockLength];
    auto offset = block_idx * blockLength;
    AscendC::Mutex::Lock<PIPE_MTE2>(0);
    AscendC::DataCopy(xPtr, x + offset, blockLength);   // 指针直接作为入参
    AscendC::Mutex::Unlock<PIPE_MTE2>(0);
    AscendC::Mutex::Lock<PIPE_MTE3>(0);
    AscendC::DataCopy(z + offset, zPtr, blockLength);
    AscendC::Mutex::Unlock<PIPE_MTE3>(0);
}
```

# 需求分析（required）

## 需求描述

对 asc-devkit Basic API 中 DMA（类型码 D）类接口扩展指针输入能力：`dst` / `src` 入参既可为 `LocalTensor` / `GlobalTensor`，也可为 `__ubuf__ T*` / `__gm__ T*` / `__cbuf__ T*` 等裸硬件指针，两种用法经同一对外 API 走同一套底层 `*Impl` 实现。

## 需求拆解

1. 接口范围：仅本册 DMA 类接口，共 5 个 API 名称、44 个重载签名，即 DataCopy、DataCopyPad、DataCopyL1ToUB、DataCachePreload、DataCacheCleanAndInvalid。
2. 兼容性：原有 Tensor 入参接口的功能、SFINAE 约束、`__inout_pipe__` pipe 属性、`PrimT` 类型萃取行为均保持不变，原有用例回归全通过。
3. 语义一致：指针路径与对应 Tensor 用法在相同数据布局下数值一致；不擅自变更 `*Impl` 数值语义。
4. 样例迁移：官方 `examples/01_simd_cpp_api/03_basic_api/` 中搬运相关样例（data_copy_gm2l1、data_copy_gm2ub_nddma、data_copy_gm2ub_slice、data_copy_l0c2gm、data_copy_pad_gm2ub_ub2gm、data_copy_ub2l1 等）支持迁移为指针写法并精度自测通过。
5. 改造位置：对外声明位于 `include/basic_api`，实现位于 `impl/basic_api`；不修改 VECTOR / CUBE 姊妹册文件。

# 详细设计（required）

## 接口分析

### 接口清单

| API 名称 | 方向/形式 | 重载要点 |
| --- | --- | --- |
| DataCopy | GM→L1、GM→UB、UB→L1、UB→UB、L1→UB、UB→GM、L1→GM、L0C→L1/GM | DataCopyParams / Nd2NzParams / Dn2NzParams / Nz2NdParamsFull / Nz2DnParamsFull / DataCopyEnhancedParams / DataCopyCO12DstParams / SliceInfo[] / MultiCopyParams / count 形式 / 类型转换（half↔float、int32→half/int16/int8/uint8/bf16） |
| DataCopyPad | GM↔UB、UB↔L1 | DataCopyParams+DataCopyPadParams / DataCopyExtParams+DataCopyPadExtParams / Nd2NzParams / PaddingMode 模板 |
| DataCopyL1ToUB | L1→UB | count 形式 / DataCopyParams 形式，subBlockId 模板 |
| DataCachePreload | GM 预取 | `GlobalTensor<uint64_t>` + cacheOffset |
| DataCacheCleanAndInvalid | GM/L1 cache 维护 | CacheLine / DcciDst 模板参数，GT 与 LT 两种入参 |

以上合计 44 个重载签名，与任务书 2.4 节接口清单（basic_api_list_dma_api）一致。

## 实现方案

### 整体设计流程图

![整体设计流程图](https://gitcode.com/api/v5/repos/gcw_8p1hhlB0/cann-ops-competitions/raw/04_tasks/01_community-task-2026/tasklist/08-51-basic-api-dma-ptr-extend/figures/design_flow.png?ref=master)

### 1. 统一指针萃取层 GetUnderlyingPtr

在模板封装层提供统一萃取函数（可与 VECTOR 册共用同一实现）：

```cpp
template <typename U>
__aicore__ inline auto GetUnderlyingPtr(const U& val)
{
    if constexpr (Std::is_pointer_v<U>) {
        return val;
    } else {
        return val.GetPhyAddr();
    }
}
```

设计要点：

1. 指针类型（`__ubuf__ T*` / `__gm__ T*` / `__cbuf__ T*` 等，编译器视作带地址空间修饰的指针）经 `is_pointer_v` 判定后原样返回；
2. Tensor 类型经 `GetPhyAddr()` 返回底层硬件指针，与现有 `*Impl` 调用点使用的表达式完全一致，保证 Tensor 路径行为零变化；
3. 萃取结果统一为裸硬件指针，直接下沉到既有 `DataCopyImpl` / `DataCopyPadImpl` / `DataCopyL12UBImpl` / `DataCachePreloadImpl` 等底层实现，`*Impl` 一律不改动。

### 2. 对外接口改造范式

以 DataCopy（count 形式）为例：

```cpp
// 扩展后：dst/src 可为 __ubuf__ T* / __gm__ T* 或 Tensor
template <typename T, typename U>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(const T& dst, const U& src, const uint32_t count)
{
    DataCopyImpl(GetUnderlyingPtr(dst), GetUnderlyingPtr(src), count);
}
```

复杂重载逐条对应改造，例如：

```cpp
template <typename T, typename U>
__aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const T& dst, const U& src,
    const DataCopyParams& copyParams, const DataCopyPadParams& padParams)
{
    DataCopyPadImpl(GetUnderlyingPtr(dst), GetUnderlyingPtr(src), copyParams, padParams);
}
```

### 3. 校验与兼容策略

1. Tensor 入参路径保留原有 `CheckTensorPos` / `CheckDataCopyTensorSizeOverflow` 等位置与溢出校验；指针入参因无 position 元信息，仅保留 count/32B 对齐等与数据相关的校验（与 Tensor 路径共用同一 `DataCopyCheck` 等函数），不做指针合法性猜测，与官方 `*Impl` 行为对齐；
2. pipe 属性（`__inout_pipe__(MTE2/MTE3/V)`）与原接口逐一保持一致；带 `PaddingMode` / `enableSmallC0` / `subBlockId` / `CacheLine` / `DcciDst` 等模板默认参数的重载原样保留；
3. `PrimT<T>` 类型萃取、TensorTrait 特化（`PrimT<T> = U`）等约束通过同一模板转发，不引入新的特化分叉；
4. DataCachePreload / DataCacheCleanAndInvalid 的 GM 指针路径直接以 `__gm__ uint64_t*` / `__gm__ T*` 萃取后进入 `*Impl`。

### 4. 涉及文件

```text
asc-devkit/
├── include/basic_api/
│   ├── kernel_operator_data_copy_intf.h     # DataCopy / DataCopyPad / DataCopyL1ToUB 声明
│   ├── kernel_operator_cache_intf.h         # DataCachePreload / DataCacheCleanAndInvalid 声明
│   └── (共用萃取实现所在头文件)
├── impl/basic_api/
│   ├── kernel_operator_data_copy_intf_impl.h
│   └── kernel_operator_cache_intf_impl.h
├── tests/api/basic_api/                     # 指针路径单测
└── examples/01_simd_cpp_api/03_basic_api/   # 样例指针化改造
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 系列（含 950PR/950DT，`__NPU_ARCH__` 5101/5161/5163/5165 等） | √ |

## 算子约束限制

1. 仅扩展 DMA 册接口，不改动 VECTOR / CUBE 册文件；
2. 指针入参不携带 Tensor 元信息（position / shape），数据对齐、32B 对齐等约束由调用方保证，与 `*Impl` 原生约束一致；
3. 不引入与输入规模线性相关的额外 Device 内存拷贝。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 搬运类接口指针路径与 Tensor 路径输出逐比特一致；数据域 [-100, 100]，Shape 覆盖 1 / 32 / 1024 / 2048 | 生态算子开源精度标准（实验标准） |
| 性能标准 | 本任务无标杆时延要求；自测说明相对改造前无明显性能回退 | 任务书 3.3 |

## 测试用例规划

![自验证流程图](https://gitcode.com/api/v5/repos/gcw_8p1hhlB0/cann-ops-competitions/raw/04_tasks/01_community-task-2026/tasklist/08-51-basic-api-dma-ptr-extend/figures/test_flow.png?ref=master)

| 测试场景分类 | 用例描述 | 数据域 / Shape | 接口 | 预期结果 |
| --- | --- | --- | --- | --- |
| 指针路径正向 | GM→UB / UB→GM 指针直传 | [-100,100]，1/32/1024/2048 | DataCopy（count、DataCopyParams 形式） | 与 Tensor 路径输出逐比特一致 |
| 指针路径正向 | GM↔UB 非 32B 对齐搬运 | [-100,100]，1/32/1024/2048 | DataCopyPad | 与 Tensor 路径输出逐比特一致 |
| 指针路径正向 | L1→UB 指针直传 | [-100,100]，1/32/1024/2048 | DataCopyL1ToUB | 与 Tensor 路径输出逐比特一致 |
| 指针路径正向 | GM 预取指针路径 | [-100,100]，1/32/1024/2048 | DataCachePreload | 预取行为正确，无越界 |
| 兼容性回归 | 原有 Tensor 用例全量回归 | 原有用例范围 | 全部 DMA 接口 | 回归全部通过，零破坏 |
| 样例迁移 | 搬运类样例迁移为指针写法 | 样例自带数据 | DataCopy / DataCopyPad | 精度自测通过 |
| 内存检查 | 指针扩展不引入额外 Device 拷贝 | 1/32/1024/2048 | DataCopy 系列 | 内存占用与 Tensor 路径一致 |

1. `tests/api/basic_api/` 下为代表性接口（DataCopy count/DataCopyParams 形式、DataCopyPad、DataCopyL1ToUB、DataCachePreload）增补指针路径用例，同输入下对比 Tensor 与指针路径输出；
2. 复用仓库 `gen_data.py` / `verify_result.py` 生成数据与校验；
3. 原有 Tensor 用例全量回归，验证兼容性零破坏；
4. 样例改造：`examples/01_simd_cpp_api/03_basic_api/` 搬运类样例增补指针写法分支。

## 兼容性分析

存量 Tensor / GlobalTensor 接口签名不变、模板约束不变、`*Impl` 不改动，属加法式扩展；原有用户代码无需任何修改即保持可编译、可运行。

## 风险与降级预案

1. 若指针类型（`__ubuf__` / `__gm__` / `__cbuf__`）在个别编译器版本下 `is_pointer_v` 判定存在差异，则以 `ccec` / 毕昇 ASC 编译器实测为准，必要时补充地址空间特化分支；
2. 若复杂重载（Nd2Nz / Dn2Nz / SliceInfo / MultiCopyParams 等）逐条改造中与原 Tensor 签名产生模板二义性，则收窄模板约束（SFINAE 显式排除 Tensor 类型）并拆分 PR 评审；


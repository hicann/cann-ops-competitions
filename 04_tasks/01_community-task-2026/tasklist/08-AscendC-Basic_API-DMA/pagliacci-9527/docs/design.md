# 需求背景

## 需求来源

社区任务「8月社区任务 - Ascend C Basic API 指针化扩展（DMA · 数据搬运 / 缓存）」。

| 项 | 值 |
|----|----|
| 贡献者 GitCode | pagliacci-9527 |
| 适配硬件 | Ascend 950 系列 |
| CANN | 9.0.0 ~ 9.1.0 |
| 代码合入 | https://gitcode.com/cann/asc-devkit （`include/basic_api`、`impl/basic_api`） |
| 接口清单 | [DMA 类型码 D](https://docs.qq.com/sheet/DYXRHY3hFem9kWXBJ)；总表 https://docs.qq.com/sheet/DYXpIenNSTkp4SXBh |
| 设计模板 | https://gitcode.com/cann/asc-devkit/issues/1222 |

本任务不是独立 aclnn / aclsparse 算子，而是对已有 Basic API 做编译期入参扩展，无 ATen / Python 绑定。

## 背景介绍

当前 `DataCopy` / `DataCopyPad` / `DataCopyL1ToUB` / `DataCachePreload` / `DataCacheCleanAndInvalid` 的公开声明固定为 `LocalTensor<T>` / `GlobalTensor<T>`。开发者若已持有 `__gm__` / `__ubuf__` / `__cbuf__` 硬件指针，必须先包装成 Tensor 才能调用。

asc-devkit 现状：

- 搬运声明：`include/basic_api/kernel_operator_data_copy_intf.h`，实现 `impl/basic_api/kernel_operator_data_copy_intf_impl.h`
- 缓存声明：`include/basic_api/kernel_operator_cache_intf.h`，实现 `impl/basic_api/kernel_operator_cache_intf_impl.h`
- Tensor 已有 `GetPhyAddr()`；尚无任务书要求的 `GetUnderlyingPtr`
- 样例：`examples/01_simd_cpp_api/03_basic_api/00_data_movement/`、`10_cache_control/`

目标范式：

```cpp
AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
AscendC::DataCopy(xPtr, x + offset, blockLength);  // __ubuf__ / __gm__
AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);
```

# 需求分析

## 需求描述

1. 同一套对外 API 同时接受 Pointer 与 Tensor；指针路径与 Tensor 路径在相同布局下数值一致。
2. 范围仅限本册 DMA（类型码 D）**5 个 API 名、44 个重载**（以接口清单为准）。含 GM/UB/L1 的 DataCopy、DataCopyPad、DataCopyL1ToUB 及 DataCache 系列。
3. 不改 `*Impl` 数值语义；原 Tensor 用例必须回归通过。
4. 可将官方 `examples/01_simd_cpp_api/03_basic_api/` 中 DataCopy / DataCopyPad 从 Tensor 写法迁到指针写法。
5. 不改 VECTOR / CUBE 他册文件；不改本头文件中的 `Copy`（VECIN/VECOUT、`PIPE_V`，非本册 DMA）。

## 需求拆解

1. **公共萃取**：新增 `GetUnderlyingPtr`（可与 VECTOR 册共用），指针类型原样返回，否则调用 `GetPhyAddr()`。
2. **C++ 接口**：`DataCopy`、`DataCopyPad`、`DataCopyL1ToUB`、`DataCachePreload`、`DataCacheCleanAndInvalid`。按清单逐条把 Tensor 形参改为模板 `T`/`U`，保留 SFINAE、`__inout_pipe__`、`PaddingMode` 等。
3. **测试**：改造 `00_data_movement` 代表性样例及任务配套搬运用例；指针路径与 Tensor 路径对比；shape 建议 `1/32/1024/2048`，取值 [-100, 100]。
4. **性能**：本任务无额外性能指标与标杆时延；自测可说明相对改造前无明显回退。
5. **内存**：编译期适配，禁止与输入规模线性相关的额外 Device 拷贝。

# 详细设计

## 算子分析

### 数学公式

DMA 接口做存储层次之间的数据搬移或 Cache 操作，无独立算术公式。

语义：`dst[i] = src[i]`（及 Pad / ND2NZ / Slice / Enhanced 等由原 `*Impl` 定义的布局变换）。指针路径与 Tensor 路径必须落到同一 `*Impl`。

`DataCachePreload`：按 GM 地址预取 1 个 Cache Line。  
`DataCacheCleanAndInvalid`：Clean + Invalid，保证 DCache 与 GM / Local Memory 一致性。

### 支持数据类型

沿用各重载现有 `T`/`U`（含 `half` / `float` / `int32_t` 等 Enhanced 转换对）。本任务不新增 dtype、不放宽对齐。指针元素类型须与原 Tensor `PrimT<T>` 一致。

### 支持形状

无算子级 shape 推导。搬运长度 / `DataCopyParams` / `Nd2NzParams` / `SliceInfo` / `MultiCopyParams` 与改造前相同。精度自测建议元素数 `1`、`32`、`1024`、`2048`。

### 接口

44 处指接口清单中的 **44 条重载声明**。列：函数名、重载序号、该函数重载总数、头文件、完整函数签名。

| 函数名 | 条数 | 头文件 |
|--------|------|--------|
| DataCacheCleanAndInvalid | 3 | `kernel_operator_cache_intf.h` |
| DataCachePreload | 1 | 同上 |
| DataCopy | 27 | `kernel_operator_data_copy_intf.h` |
| DataCopyL1ToUB | 2 | 同上 |
| DataCopyPad | 11 | 同上 |
| **合计** | **44** | |

清单与头文件的对应关系：

- GM→L1 的 `Nd2Nz` 计 2 条：带 `enableSmallC0`（950）与不带该模板参数（其他架构分支）。
- 清单不含 `DataCopy(..., Nz2DnParamsFull)`（仅 5101/516x 头文件声明），本任务不改造该重载。
- 清单不含同文件中的 `Copy`、`SetPadValue`、`NdDmaDci`、`SetLoopModePara`。

按清单顺序：

| 总序号 | 函数 | 函数内序号 | 签名要点 |
|--------|------|------------|----------|
| 1 | DataCacheCleanAndInvalid | 1/3 | `GT` + `CacheLine` + `DcciDst` |
| 2 | DataCacheCleanAndInvalid | 2/3 | `LT` + `CacheLine` + `DcciDst` |
| 3 | DataCacheCleanAndInvalid | 3/3 | `GT` + `CacheLine` |
| 4 | DataCachePreload | 1/1 | `GT<uint64_t>, offset` |
| 5 | DataCopy | 1/27 | SFINAE Enhanced `float→bf16` |
| 6 | DataCopy | 2/27 | SFINAE Enhanced `i32→i16` |
| 7 | DataCopy | 3/27 | SFINAE Enhanced `i32→i8` |
| 8 | DataCopy | 4/27 | SFINAE Enhanced `i32→u8` |
| 9 | DataCopy | 5/27 | SFINAE Enhanced `half→float` |
| 10 | DataCopy | 6/27 | SFINAE Enhanced `float→half` |
| 11 | DataCopy | 7/27 | SFINAE Enhanced `i32→half` |
| 12 | DataCopy | 8/27 | GM→L1 `Dn2NzParams` + `enableSmallC0` |
| 13 | DataCopy | 9/27 | GM→L1 `Nd2NzParams` + `enableSmallC0`（950） |
| 14 | DataCopy | 10/27 | L0C→GM `CO12Dst` |
| 15 | DataCopy | 11/27 | L0C→L1 `CO12Dst` |
| 16 | DataCopy | 12/27 | L1→BT `LT<T>, LT<U>, DataCopyParams` |
| 17 | DataCopy | 13/27 | GM→UB NDDMA `MultiCopyParams` |
| 18 | DataCopy | 14/27 | GM→UB Enhanced |
| 19 | DataCopy | 15/27 | GM→L1 `Nd2NzParams`（无 `enableSmallC0`） |
| 20 | DataCopy | 16/27 | GM→UB Slice |
| 21 | DataCopy | 17/27 | GM→UB `count` |
| 22 | DataCopy | 18/27 | UB→GM Enhanced |
| 23 | DataCopy | 19/27 | UB→GM `DataCopyParams` |
| 24 | DataCopy | 20/27 | L0C→GM `Nz2NdParamsFull` |
| 25 | DataCopy | 21/27 | UB→GM Slice |
| 26 | DataCopy | 22/27 | UB→GM `count` |
| 27 | DataCopy | 23/27 | UB→UB Enhanced |
| 28 | DataCopy | 24/27 | UB→UB `DataCopyParams` |
| 29 | DataCopy | 25/27 | UB→L1 `Nd2NzParams` |
| 30 | DataCopy | 26/27 | UB→UB `count` |
| 31 | DataCopy | 27/27 | GM→UB `DataCopyParams`（MTE2） |
| 32 | DataCopyL1ToUB | 1/2 | `DataCopyParams` |
| 33 | DataCopyL1ToUB | 2/2 | `count` |
| 34 | DataCopyPad | 1/11 | GM→UB Ext + SFINAE `PadExtParams<U>` |
| 35 | DataCopyPad | 2/11 | GM→UB Ext + `PaddingMode`（950） |
| 36 | DataCopyPad | 3/11 | GM→UB Pad + `PaddingMode`（950） |
| 37 | DataCopyPad | 4/11 | UB→GM Ext + `PaddingMode`（950） |
| 38 | DataCopyPad | 5/11 | UB→GM Pad + `PaddingMode`（950） |
| 39 | DataCopyPad | 6/11 | GM→UB Ext（无 `PaddingMode`） |
| 40 | DataCopyPad | 7/11 | GM→UB Pad（无 `PaddingMode`） |
| 41 | DataCopyPad | 8/11 | UB→GM Ext（无 `PaddingMode`） |
| 42 | DataCopyPad | 9/11 | UB→GM Pad（无 `PaddingMode`） |
| 43 | DataCopyPad | 10/11 | UB→L1 Ext + `Nd2NzParams` |
| 44 | DataCopyPad | 11/11 | UB→L1 Pad + `Nd2NzParams` |

每一处改法相同：Tensor 形参改为模板，经 `GetUnderlyingPtr` 后仍调用原 `*Impl`。不以头文件中未列入清单的 `Nz2Dn` / `Copy` 充数。

## 算子实现

### 实现方案

1. 封装层模板化，底层 `*Impl` 复用。
2. 单一 `GetUnderlyingPtr`，不为指针路径复制一套 DMA 实现。
3. 保留 `__inout_pipe__(MTE2/MTE3/V)` 与原 SFINAE。
4. 调用方仍负责 `Mutex::Lock` / `Unlock<PIPE_MTE2/MTE3>`，不改变同步语义。

落地文件：

```text
include/basic_api/kernel_operator_ptr_utils.h          # GetUnderlyingPtr（新建，VECTOR 可复用）
include/basic_api/kernel_operator_data_copy_intf.h     # 声明模板化
impl/basic_api/kernel_operator_data_copy_intf_impl.h   # GetUnderlyingPtr 后调用原 Impl
include/basic_api/kernel_operator_cache_intf.h
impl/basic_api/kernel_operator_cache_intf_impl.h
```

`GetUnderlyingPtr`：

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

count 形态示意：

```cpp
template <typename T, typename U>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(
    const T& dst, const U& src, const uint32_t count)
{
    DataCopyImpl(GetUnderlyingPtr(dst), GetUnderlyingPtr(src), count);
}
```

若模板泛化导致重载冲突，用 `enable_if` 区分指针与 Tensor，或保留原 Tensor 重载为薄封装；两者必须调用同一 Impl。

#### 3.2.1 host侧设计：

本任务是 device 侧 Basic API，没有独立 Host 算子、workspace、handle，也没有多阶段 `bufferSize` / `analysis` / `execute`。

##### 1. 分核策略：

不新增分核。样例仍按 `block_idx * blockLength` 切分 GM；API 本身无 block 维。

##### 2. 数据分块和内存优化策略：

不引入与规模线性相关的额外 Device 拷贝。指针只是地址视图，与 Tensor `GetPhyAddr()` 指向同一块物理内存。UB / L1 分配仍由调用方 `TPipe` 或栈上 `__ubuf__` 数组负责。

##### 3. tilingkey规划策略：

无 tilingkey。`Nd2NzParams` / `DataCopyParams` 等仍是调用方传入的搬运描述，不是 Host tiling 下发。

Host 样例 `main` 只负责申请 GM、launch kernel、回读对比，不在 Host 做 DMA。

#### 3.2.2 kernel侧设计：

1. 公开 API 模板化 → `GetUnderlyingPtr` → 现有 `*Impl`。
2. GM→UB 保持 `PIPE_MTE2`，UB→GM 保持 `PIPE_MTE3`；Pad / L1 路径保持原 pipe。
3. 950（`__NPU_ARCH__ == 3510`）上 `DataCopyPad` 的 `PaddingMode`、`MultiCopyParams` 条件编译保持原样。
4. 代码变更位于 asc-devkit 的 `include/basic_api` 与 `impl/basic_api`。
5. 自测：同一输入分别走 Tensor 与指针路径，输出按生态算子开源精度标准（实验标准）对比。

## 支持硬件

| 产品 | CANN |
|------|------|
| Ascend 950 系列 | 9.0.0 ~ 9.1.0 |

任务书验收硬件为 950。接口在 Atlas A2/A3 上已存在的重载须保持可编译；不把 950 专用重载强行暴露到不支持的产品。

## 算子约束限制

| 约束 | 说明 | 失败表现 |
|------|------|----------|
| Tensor 兼容 | 原 LocalTensor / GlobalTensor 调用行为不变 | 原样例回归失败 |
| 只扩指针 | 不改 Impl 数值、对齐、管道语义 | 与改造前或精度标准不一致 |
| 本册边界 | 只改清单中的 DMA 接口；不动 VECTOR/CUBE；不动 `Copy` | 评审要求拆分变更 |
| 同步 | 调用方 Lock/Unlock；API 不隐式加锁 | 数据竞争或死锁 |
| 指针合法性 | 地址空间须匹配通路（`__gm__` / `__ubuf__` / `__cbuf__`） | 编译失败或运行异常 |
| 规范 | 遵循 asc-devkit 研发协作与 Ascend C 编程规范 | 评审或 CI 不通过 |

本任务为 NPU Basic API 扩展，不提供 CPU 实现路径。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 来源 |
|----------|------|------|
| 精度 | 指针路径与改造前 Tensor 路径对比，满足实验标准；原 Tensor 回归全部通过 | 任务书 3.2；[精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 覆盖 | 数据 [-100, 100]；shape `1/32/1024/2048`；至少覆盖代表性 DMA 接口 | 任务书 3.2 / 3.5 |
| 性能 | 无额外指标与标杆时延；可选说明无明显回退 | 任务书 3.3 |
| 内存 | 无与规模线性相关的额外 Device 拷贝 | 任务书 3.4 |

对照物为同仓库改造前 Tensor 路径，无 GPU 性能标杆。精度与性能数据以实机自测为准，本文不填写实测数字。

自测入口：

- `examples/01_simd_cpp_api/03_basic_api/00_data_movement/`（gm2ub / pad / gm2l1 / ub2l1 / slice / nddma / l0c2gm）
- `examples/01_simd_cpp_api/03_basic_api/10_cache_control/data_cache_preload/`
- 任务配套搬运样例（与上列场景对应）

## 兼容性分析

- API 名不变，仅增加指针可调用性，不另起同名接口。
- 原 Tensor 重载必须继续匹配，禁止打断现有模板实例化。
- 与 VECTOR / CUBE 指针化分册只共享 `GetUnderlyingPtr`，文件集合互不重叠。
- A2/A3 与 950：公共声明继续用原 `__NPU_ARCH__` 宏隔离，不把 950 专用重载暴露到不支持产品。

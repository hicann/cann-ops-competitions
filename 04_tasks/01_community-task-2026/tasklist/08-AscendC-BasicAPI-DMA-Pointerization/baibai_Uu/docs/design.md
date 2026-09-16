# 需求背景（required）

## 需求来源

本需求来源于社区任务：

**8月社区任务 - Ascend C Basic API 指针化扩展（DMA · 数据搬运 / 缓存）**

目标开源仓：[`cann/asc-devkit`](https://gitcode.com/cann/asc-devkit)  
合入目录：`include/basic_api`、`impl/basic_api`（可选同步改造 `examples/01_simd_cpp_api/03_basic_api/` 相关样例）  
适配硬件：Ascend 950 系列（验收主目标）；实现层保持与现有 Basic API 多架构路径兼容，不擅自收窄既有 Tensor 路径能力。  
CANN 版本：CANN 9.0.0 ~ CANN 9.1.0（以任务书为准）。

参与账号：`baibai_Uu`  
私仓（待验收代码地址）：`https://gitcode.com/baibai_Uu/asc-devkit`（已邀请 `Ascend-CANN`）

## 背景介绍

### Basic API DMA 接口现状

Ascend C Basic API 中 DMA（数据搬运 / 缓存）类接口当前对外签名以 `LocalTensor` / `GlobalTensor` 为主，例如：

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, const uint32_t count);

template <typename T, typename U>
__aicore__ inline void DataCopyPad(
    const T& /*当前为 Tensor 包装*/, const U& /*...*/,
    const DataCopyParams& copyParams, const DataCopyPadParams& padParams);
```

底层 `*Impl` 实际消费的是硬件指针（`__ubuf__` / `__gm__` / `__cbuf__` 等），Tensor 路径通过 `GetPhyAddr()` 取得物理地址后再下沉。

随着 **C 指针编程范式** 推广，开发者希望在 kernel 中直接使用裸指针编写搬运逻辑，例如：

```cpp
__ubuf__ float xPtr[blockLength];
AscendC::DataCopy(xPtr, x + offset, blockLength);
```

而无需先构造 `LocalTensor` / `GlobalTensor`。

### 现状问题分析

| 维度 | 现状 | 问题 |
| --- | --- | --- |
| 编程范式 | 公开 API 仅接受 Tensor 包装类型 | 指针化样例无法直接调用同一套 Basic API，需额外包装或绕开封装层 |
| 底层能力 | `*Impl` 已基于硬件指针 | 能力具备，但封装层未统一暴露指针入参 |
| 兼容性风险 | 既有大量 Tensor 样例 / UT | 扩展时必须保证原 Tensor 路径零语义变更 |
| 分册边界 | VECTOR / DMA / CUBE 三册并行 | 本设计仅覆盖 DMA 分册，避免与他册 PR 冲突 |

### 本册接口范围（任务书）

任务书约定：本册共 **5 个 API 名称、44 个重载签名**（总表类型码 **D**），官方清单见：

[basic_api_list_dma_api](https://docs.qq.com/sheet/DYXRHY3hFem9kWXBJ)

| API 名称 | 职责概要 | 主要头文件 | 官方重载数 |
| --- | --- | --- | --- |
| `DataCopy` | GM/UB/L1/L0C 等通路连续、高维、切片、随路格式转换 / 量化激活等搬运 | `kernel_operator_data_copy_intf.h` | **27** |
| `DataCopyPad` | GM↔UB 非对齐搬运与填充 / ExtParams / Nd2Nz | 同上 | **11** |
| `DataCopyL1ToUB` | L1→UB（Params / count，含 `subBlockId`） | 同上 | **2** |
| `DataCachePreload` | Data Cache 预取 | `kernel_operator_cache_intf.h` | **1** |
| `DataCacheCleanAndInvalid` | Data Cache 清理 / 失效 | 同上 | **3** |
| **合计** | | | **44 / 44（已锁定）** |

> 全量勾选表见同目录 [`checklist_DMA_44.md`](./checklist_DMA_44.md)。

#### DataCopy 官方重载清单（27，文件行号均标注为 27 族 / `kernel_operator_data_copy_intf.h`）

| # | 类别 | pipe / 约束 | 签名要点（Tensor 形参 → 指针化改造点） |
| --- | --- | --- | --- |
| 1 | Enhanced UB←UB 类型转换 | SFINAE：`bfloat16←float` | `DataCopy(LocalTensor<T>, LocalTensor<U>, DataCopyParams, DataCopyEnhancedParams)` |
| 2 | Enhanced UB←UB | `__inout_pipe__(V)`；`int16←int32` | 同上 |
| 3 | Enhanced UB←UB | `__inout_pipe__(V)`；`int8←int32` | 同上 |
| 4 | Enhanced UB←UB | `__inout_pipe__(V)`；`uint8←int32` | 同上 |
| 5 | Enhanced UB←UB | `__inout_pipe__(V)`；`float←half` | 同上 |
| 6 | Enhanced UB←UB | `half←float` | 同上 |
| 7 | Enhanced UB←UB | `__inout_pipe__(V)`；`half←int32` | 同上 |
| 8 | GM→L1 DN2NZ | `__inout_pipe__(MTE2)`；`enableSmallC0` | `DataCopy(LocalTensor<T>, GlobalTensor<T>, Dn2NzParams)` |
| 9 | GM→L1 ND2NZ | `__inout_pipe__(MTE2)`；`enableSmallC0` | `DataCopy(LocalTensor<T>, GlobalTensor<T>, Nd2NzParams)` |
| 10 | L0C→GM | — | `DataCopy(GlobalTensor<T>, LocalTensor<U>, DataCopyCO12DstParams)` |
| 11 | L0C→Local | — | `DataCopy(LocalTensor<T>, LocalTensor<U>, DataCopyCO12DstParams)` |
| 12 | Local←Local 异类型 | — | `DataCopy(LocalTensor<T>, LocalTensor<U>, DataCopyParams)` |
| 13 | GM→UB NDDMA | `dim` + `NdDmaConfig` | `DataCopy(LocalTensor<T>, GlobalTensor<T>, MultiCopyParams<T,dim>)` |
| 14 | GM→Local Enhanced | `__inout_pipe__(MTE2)` | `DataCopy(LocalTensor<T>, GlobalTensor<T>, DataCopyParams, DataCopyEnhancedParams)` |
| 15 | GM→L1 ND2NZ（无 SmallC0 模板形参版） | `__inout_pipe__(MTE2)` | `DataCopy(LocalTensor<T>, GlobalTensor<T>, Nd2NzParams)` |
| 16 | GM→UB 切片 | `__inout_pipe__(MTE2)` | `DataCopy(LocalTensor, GlobalTensor, SliceInfo[], SliceInfo[], dimValue)` |
| 17 | GM→UB 连续 count | `__inout_pipe__(MTE2)` | `DataCopy(LocalTensor<T>, GlobalTensor<T>, uint32_t count)` |
| 18 | Local→GM Enhanced | `__inout_pipe__(MTE3)` | `DataCopy(GlobalTensor, LocalTensor, DataCopyParams, DataCopyEnhancedParams)` |
| 19 | Local→GM Params | `__inout_pipe__(MTE3)` | `DataCopy(GlobalTensor, LocalTensor, DataCopyParams)` |
| 20 | Local→GM NZ2ND | `__inout_pipe__(MTE3)` | `DataCopy(GlobalTensor, LocalTensor, Nz2NdParamsFull)` |
| 21 | Local→GM 切片 | `__inout_pipe__(MTE3)` | `DataCopy(GlobalTensor, LocalTensor, SliceInfo[], SliceInfo[], dimValue)` |
| 22 | Local→GM 连续 count | `__inout_pipe__(MTE3)` | `DataCopy(GlobalTensor, LocalTensor, uint32_t count)` |
| 23 | UB←UB Enhanced 同类型 | — | `DataCopy(LocalTensor, LocalTensor, DataCopyParams, DataCopyEnhancedParams)` |
| 24 | UB←UB Params 同类型 | — | `DataCopy(LocalTensor, LocalTensor, DataCopyParams)` |
| 25 | UB←UB ND2NZ | — | `DataCopy(LocalTensor, LocalTensor, Nd2NzParams)` |
| 26 | UB←UB 连续 count | — | `DataCopy(LocalTensor, LocalTensor, uint32_t count)` |
| 27 | GM→UB Params | `__inout_pipe__(MTE2)` | `DataCopy(LocalTensor, GlobalTensor, DataCopyParams)` |

**指针化统一规则（对本表全部 27 条 + 下述其余 17 条，共 44 条生效）：**

1. 将 `LocalTensor<*>` / `GlobalTensor<*>` 操作数改为模板类型，经 `GetUnderlyingPtr` 下沉；
2. `DataCopyParams` / `DataCopyExtParams` / `DataCopyPadParams` / `DataCopyPadExtParams` / `DataCopyEnhancedParams` / `Nd2NzParams` / `Dn2NzParams` / `SliceInfo` / `MultiCopyParams` / `DataCopyCO12DstParams` / `Nz2NdParamsFull` / `count` / `dimValue` / `cacheOffset` 等非 Tensor 参数 **保持原类型与顺序**；
3. 保留原有 `__inout_pipe__`、SFINAE、`enableSmallC0`、`PaddingMode`、`subBlockId`、`NdDmaConfig`、`CacheLine`、`DcciDst` 等约束，不改 `*Impl` 数值语义。

#### DataCopyPad（11）/ DataCopyL1ToUB（2）/ DataCache*（4）摘要

| API | # | 要点 |
| --- | --- | --- |
| DataCopyPad | 1 | SFINAE：`PrimT<T>==U` 且 `T!=U`；MTE2；ExtParams + PadExtParams\<U\> |
| DataCopyPad | 2–3 | `PaddingMode`；MTE2；GM→UB Pad（Ext / 普通 Params） |
| DataCopyPad | 4–5 | `PaddingMode`；MTE3；UB→GM（仅 copyParams，无 pad） |
| DataCopyPad | 6–9 | 无 PaddingMode 默认形参版；MTE2/MTE3 对应重载 |
| DataCopyPad | 10–11 | UB←UB + Nd2NzParams（Ext / 普通 Params） |
| DataCopyL1ToUB | 1–2 | `subBlockId`；Params / count |
| DataCacheCleanAndInvalid | 1–2 | `CacheLine`+`DcciDst`；Global / Local |
| DataCacheCleanAndInvalid | 3 | 仅 `CacheLine`；Global |
| DataCachePreload | 1 | `GlobalTensor<uint64_t>` + `cacheOffset` |

**明确不在本册范围：**

- VECTOR 分册矢量计算接口（Add / Reduce / Gather 等）
- CUBE 分册矩阵接口（LoadData / Mmad / Fixpipe 等）
- 擅自修改 `*Impl` 数值语义、pipe 属性或 SFINAE 约束
- `ICachePreLoad` / `GetICachePreloadStatus`（不在本册 44 条内）

# 需求分析（required）

## 需求描述

在 **不破坏** 现有 `LocalTensor` / `GlobalTensor` 调用兼容性的前提下，对本册 DMA 类 Basic API 的公开封装层进行 **指针化扩展**：

1. 同一套对外 API 支持 **Pointer** 与 **Tensor** 双重入参；
2. 指针路径与对应 Tensor 路径在相同数据布局下数值一致；
3. Tensor 路径行为与改造前完全一致（回归全过）；
4. 可把官方 `examples/01_simd_cpp_api/03_basic_api/` 下 DMA 相关样例迁移为指针写法并通过精度自测；
5. 本任务 **无额外性能门槛**，但不得引入与输入规模线性相关的额外 Device 内存拷贝。

## 需求拆解

1. **统一指针萃取**：提供可复用的 `GetUnderlyingPtr`（可与 VECTOR 分册共用），对裸指针原样返回，对 Tensor 调用 `GetPhyAddr()`。
2. **封装层模板化**：将本册清单中 Tensor 形参改为模板参数，经 `GetUnderlyingPtr` 下沉至既有 `*Impl`。
3. **兼容性保持**：原 `LocalTensor` / `GlobalTensor` 调用方式编译与运行语义不变；保留 `__inout_pipe__` 等属性。
4. **分册隔离**：仅修改 DMA 相关 `include/basic_api` / `impl/basic_api` 文件，不改 VECTOR/CUBE 他册接口文件。
5. **自验证**：基于任务包样例（GM2L1 / NDDMA / Slice / DataCopyPad / UB2L1 / L0C2GM 等）改造指针路径，完成精度对比与 Tensor 回归。
6. **文档与评审**：输出本设计文档，经 cann-competitions 评审流程后，再改实现并提 PR。

# 详细设计（required）

## 算子 / 接口分析

本需求不是新增独立 GE/aclnn 算子，而是 **Basic API 封装层能力扩展**。

### 数学 / 功能语义

各 DMA API 的搬运语义、对齐约束、随路转换 / 量化激活行为 **与现网 `*Impl` 完全一致**，本设计不新增算法，只扩展入参类型表达。

### 支持数据类型与形状

与各重载现有约束一致（由原接口模板参数 `T` / 架构宏决定），不另设新 dtype / shape 白名单。  
自测建议覆盖（任务书）：

- 数值域：`[-100, 100]`（或接口合法域）
- Shape：`1` / `32` / `1024` / `2048`（矩阵类按 M/K/N 等价缩放）

### 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Ascend 950PR / 950DT | √ | 任务验收主目标 |
| Atlas A2 / A3 | √（保持既有 Tensor 路径） | 不因指针扩展削弱既有能力；部分场景（DN、部分 Pad/LoopMode）以原接口产品矩阵为准 |

## 实现方案

### 总体改造范式

```
flowchart LR
  A["调用方: Tensor 或 裸指针"] --> B["GetUnderlyingPtr"]
  B --> C["既有 *Impl 硬件指针实现"]
  C --> D["DMA 硬件通路 MTE2/MTE3/FixPipe 等"]
```

核心原则：

1. **只改封装，不改 Impl 数值语义**；
2. **编译期分支**区分指针 / Tensor，无运行时额外拷贝；
3. **同一重载**允许 dst/src 各自独立选择指针或 Tensor（混用），PrimType 不一致时沿用原模板约束导致编译失败。

### 公共辅助：`GetUnderlyingPtr`

拟放置于 Basic API 公共头（例如新建 `include/basic_api/kernel_ptr_utils.h`，或放入已有公共 utils，具体路径实现阶段与仓库规范对齐；可与 VECTOR 册共用）：

```cpp
namespace AscendC {

template <typename U>
__aicore__ inline auto GetUnderlyingPtr(const U& val)
{
    if constexpr (std::is_pointer_v<U>) {
        return val;                 // __ubuf__/__gm__/__cbuf__ ...
    } else {
        return val.GetPhyAddr();    // LocalTensor / GlobalTensor
    }
}

} // namespace AscendC
```

### 接口设计示例

#### DataCopy（count 形式，示意）

改造前：

```cpp
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, const uint32_t count);
```

改造后（示意，保留 pipe 属性与 Impl 调用）：

```cpp
template <typename DstT, typename SrcT>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(
    const DstT& dst, const SrcT& src, const uint32_t count)
{
    DataCopyImpl(/*PrimType*/ GetUnderlyingPtr(dst), GetUnderlyingPtr(src), count);
}
```

为降低对既有调用点的破坏面，实现阶段可采用以下之一（评审确认后落地）：

| 方案 | 做法 | 优点 | 风险 |
| --- | --- | --- | --- |
| A. 原签名保留 + 新增指针重载 | Tensor 重载继续显式存在；另增指针/模板重载 | 兼容最直观 | 重载数量增加，需仔细消歧义 |
| B. 统一模板入参（任务书范式） | Tensor/指针均走模板 `T`/`U` | 代码收敛 | 需验证与现有 SFINAE / pipe 宏组合无回归 |

**本设计默认采纳任务书推荐的方案 B**，若合入评审要求更保守，可回退方案 A，但指针能力与验收标准不变。

#### DataCopyPad（示意）

```cpp
template <typename DstT, typename SrcT>
__aicore__ inline void DataCopyPad(
    const DstT& dst, const SrcT& src,
    const DataCopyParams& copyParams, const DataCopyPadParams& padParams)
{
    DataCopyPadImpl(GetUnderlyingPtr(dst), GetUnderlyingPtr(src), copyParams, padParams);
}
```

`DataCopyParams` / `DataCopyPadParams` / `Nd2NzParams` / `Dn2NzParams` / `SliceInfo` / `DataCopyEnhancedParams` 等 **非 Tensor 参数保持原类型与顺序**。

#### DataCopyL1ToUB / DataCache*（示意）

同范式：仅将 Tensor 操作数改为可萃取指针的模板入参，其余模板参数（如 `CacheLine` / `DcciDst`）保持不变。

### Host 侧设计

不涉及。本需求无独立算子 Host / Tiling / 框架注册。

### Kernel 侧设计

1. 调用方在 `__aicore__` / `__vector__` kernel 中直接传入 `__gm__` / `__ubuf__` / `__cbuf__` 指针或继续传 Tensor；
2. 同步（`Mutex` / `SetFlag`/`WaitFlag`）与管线约束仍由调用方按原文档使用，本扩展不改变同步模型；
3. 多核 NDDMA 场景仍须按原规范调用 `NdDmaDci()` 等辅助接口（若属本册清单则一并指针化；否则保持原样）。

### 文件与变更边界

```text
asc-devkit/
├── include/basic_api/
│   ├── kernel_operator_data_copy_intf.h      # DMA 公开声明改造
│   ├── kernel_operator_cache_intf.h          # DataCache* 公开声明改造
│   └── kernel_ptr_utils.h                    # 新增（若采用独立头）
├── impl/basic_api/
│   ├── kernel_operator_data_copy_intf_impl.h # 如需适配模板下沉，最小改动
│   └── kernel_operator_cache_intf_impl.h
├── examples/01_simd_cpp_api/03_basic_api/    # 样例指针化改造（评审要求时合入）
└── tests/api/basic_api/                      # 增补指针路径用例 / 保持 Tensor 回归
```

**禁止**：修改 VECTOR/CUBE 他册接口文件；修改无关文档/二进制；在 PR 中附带敏感信息。

## 测试用例设计

### 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 指针路径与改造前 Tensor 路径相同输入下满足生态算子开源精度标准（实验标准）；原 Tensor 回归全过 | 任务书 §3.2 |
| 性能标准 | **无硬性指标**；自测报告说明无明显回退即可 | 任务书 §3.3 |
| 内存标准 | 不引入与输入规模线性相关的额外 Device 拷贝 | 任务书 §3.4 |

### 自测矩阵（与任务包样例对齐）

| 样例 | 覆盖能力 | 场景数 | 指针化验证点 |
| --- | --- | --- | --- |
| `data_copy_gm2l1` | GM→L1；Nz/ND/DN；Bias；量化 | 5 | `DataCopy` + params（含 Nd2Nz/Dn2Nz） |
| `data_copy_gm2ub_nddma` | GM→UB NDDMA（Pad/转置/广播/切片） | 5 | 多维 `DataCopy` 指针路径 |
| `data_copy_gm2ub_slice` | GM↔UB 切片 | 1 | SliceInfo 重载指针路径 |
| `data_copy_pad_gm2ub_ub2gm` | DataCopyPad 非对齐+填充 | 6 | `DataCopyPad` 指针路径 |
| `data_copy_ub2l1` | UB→L1 连续 / ND2NZ | 2 | Local 指针路径 |
| `data_copy_l0c2gm` | L0C→GM 量化/激活 | 6 | Enhanced / CO1 相关重载指针路径 |

每个场景执行：

1. Tensor 路径基线（改造后回归）；
2. 指针路径改造样例；
3. `gen_data.py` + `verify_result.py` 精度对比。

另按任务书补充 Shape=`1/32/1024/2048` 的代表性 count 型用例。

## 算子约束限制

1. 仅扩展指针输入，不改变原接口对齐、合法域、产品支持矩阵；
2. 非法地址 / 未对齐行为与原 `*Impl` 一致；
3. 本册不交付 VECTOR/CUBE 接口改造；
4. 官方表 44 签名为验收勾选清单；表外接口不主动扩改。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 指针 vs Tensor 同输入一致；满足实验标准 | 任务书 / opbase 实验标准 |
| 性能标准 | 无额外门槛；报告说明无显著回退 | 任务书明确「无」 |

## 兼容性分析

| 项目 | 结论 |
| --- | --- |
| 既有 Tensor 调用 | 必须源码级兼容、语义不变 |
| 既有 UT / 样例 | 回归全过 |
| ABI / 框架算子 | 不涉及 GE/aclnn 注册变更 |
| 他册接口 | 不修改，避免并行冲突 |

## 风险与回滚

| 风险 | 缓解 |
| --- | --- |
| 模板重载歧义导致既有调用编译失败 | 优先在本地全量编译 Basic API 样例；必要时保留显式 Tensor 重载（方案 A） |
| 指针与 Tensor 混用 PrimType 不一致 | 沿用原模板约束，编译期失败 |
| 与 VECTOR 册同时改公共头冲突 | `GetUnderlyingPtr` 抽公共头并提前沟通；PR 仅含 DMA 文件 |
| 官方 44 签名与仓内现状不完全对齐 | 实现前对照腾讯文档表逐条建 checklist |

## 实施计划（设计通过后）

1. 对照官方 DMA 清单建立 44 签名 checklist；
2. 合入 `GetUnderlyingPtr` 与 DMA 封装层改造；
3. 改造任务包 6 类样例为指针路径并完成精度自测；
4. 准备自测报告、易用性 Issue、向 `asc-devkit` 提 PR。

---

**文档状态**：设计稿（实现前）  
**下一步**：按社区流程提交设计评审（cann-competitions tasklist PR + asc-devkit Issue），评审通过后再修改实现代码。

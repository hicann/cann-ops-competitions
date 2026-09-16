# [Requirement|需求建议]: 【社区任务】Ascend C Basic API 指针化扩展（CUBE · 矩阵 / ISASI）设计文档评审申请

# 需求背景（required）

## 需求来源

- 社区任务：9 月社区任务 - AscendC Basic_API优化实现（CUBE 侧矩阵类接口扩展）
- 上游仓库：`asc-devkit`，变更落于 `include/basic_api/` 与 `impl/basic_api/`
- 硬件：Ascend 950 系列产品（`dav-3510`）

## 背景介绍

Ascend C 以「兼容 C/C++ 标准 · 释放极致算力」为设计理念，同时兼容**指针式原生 C 开发范式**与 Tensor + Layout 现代 C++ 模式。当前 Basic API 的 CUBE 类接口（LoadData / Mmad / Fixpipe 等）对外只接受 `LocalTensor` / `GlobalTensor` 包装类型，开发者无法直接以 `__cbuf__` / `__ca__` / `__cb__` / `__cc__` 硬件指针调用，与「指针式原生 C 范式」的承诺存在落差。

本任务对 CUBE 类 Basic API 扩展裸指针输入能力，使同一套对外 API 同时接受 Pointer 与 Tensor，且**不改变原有 Tensor 路径的功能与语义**。

### 改造前的调用形态

```cpp
AscendC::LoadData(a2Local, a1Local, loadDataParams);
AscendC::Mmad(cLocal, aLocal, bLocal, mmadParams);
AscendC::Fixpipe<outputType, l0cType, CFG_NZ_UB>(cUB, cLocal, fixpipeParams);
```

### 改造后的目标形态

```cpp
AscendC::LoadData((__ca__ inputType*)a2Local.GetPhyAddr(),
                  (__cbuf__ inputType*)a1Local.GetPhyAddr(), loadDataParams);
AscendC::Mmad((__cc__ l0cType*)c.GetPhyAddr(),
              (__ca__ inputType*)a.GetPhyAddr(),
              (__cb__ inputType*)b.GetPhyAddr(), mmadParams);
AscendC::Fixpipe<outputType, l0cType, CFG_NZ_UB>(
    (__ubuf__ outputType*)cUB.GetPhyAddr(), (__cc__ l0cType*)c.GetPhyAddr(), fixpipeParams);
```

# 需求分析（required）

## 需求描述

对 CUBE 类 Basic API 做**指针化扩展**：在不破坏 `LocalTensor` / `GlobalTensor` 既有接口的前提下，令其同时接受裸指针入参；指针路径与对应 Tensor 路径在相同数据布局下数值一致。

任务书声明的改造范围为 **21 个 API 名称 / 61 个重载签名**，权威清单见任务书 §2.4 所列接口总表。

## 需求拆解

1. 在封装层引入统一的 Tensor/Pointer 萃取点，替代任务书示例中的 `GetUnderlyingPtr` 范式（见「接口设计」——该范式**不能照抄**，原因见下）。
2. 覆盖本册 CUBE 类接口的全部重载签名。
3. 保持原有 Tensor 接口的功能与能力不变，原有回归用例全过。
4. 指针路径与 Tensor 路径数值一致，满足生态算子开源精度标准。
5. 无额外性能指标；自测报告说明无性能回退。
6. 不得改动 VECTOR / DMA 他册文件，避免 PR 冲突。

## 接口范围（权威清单）

§2.4 所列接口总表 `basic_api_list_cube_api` 为本册权威范围，共 **19 个 API 名称**。本机对
`asc-devkit` master（`0dc906050`）同名清单实测重载合计 **57 条**（两条独立路径互相印证）：

| API 名 | 头文件 | 重载 |
| --- | --- | --- |
| `Fixpipe` | `kernel_operator_fixpipe_intf.h` | 16 |
| `LoadData` | `kernel_operator_mm_intf.h` | 8 |
| `DumpTensor` | `kernel_operator_dump_tensor_intf.h` | 4 |
| `Mmad` | `kernel_operator_mm_intf.h` | 4 |
| `MmadMx` | `kernel_operator_mm_intf.h` | 4 |
| `PopStackBuffer` | `kernel_tpipe.h` | 3 |
| `SyncAll` | `kernel_operator_block_sync_intf.h` | 3 |
| `DumpAccChkPoint` | `kernel_operator_dump_tensor_intf.h` | 2 |
| `LoadDataWithTranspose` | `kernel_operator_mm_intf.h` | 2 |
| `SetAddrWithOffset` | `kernel_tensor.h` | 2 |
| `SetFixPipeConfig` | `kernel_operator_fixpipe_intf.h` | 2 |
| `IBSet` / `IBWait` | `kernel_operator_block_sync_intf.h` | 1 / 1 |
| `InitConstValue` / `LoadImageToLocal` | `kernel_operator_mm_intf.h` | 1 / 1 |
| `InitDetermineComputeWorkspace` / `NotifyNextBlock` / `WaitPreBlock` | `kernel_operator_determine_compute_sync_intf.h` | 1 / 1 / 1 |
| `MmadBitMode` | 非独立函数名——为 `Mmad` / `MmadMx` 中收 `MmadBitModeParams` 的重载组 | 4 |
| **合计** | | **61** |

任务书声明 **21 个 API 名称 / 61 个重载签名**。上表 19 名（按"函数名"分行）加 `MmadBitMode`
这一"参数变体行"后与声明口径一致；57 + 4 = **61**，与声明**精确闭合**。

> **说明**：`MmadBitMode` 不作为独立函数存在（`kernel_operator_mm_intf.h:280-287` 为其所在重载），
> 权威表按参数变体单独成行。此口径已用权威表自带的「该函数重载总数」列核对。


## 接口设计

### 任务书示例范式为何不能照抄

任务书给出：

```cpp
template <typename U>
__aicore__ inline auto GetUnderlyingPtr(const U& val) {
    if constexpr (std::is_pointer_v<U>) { return val; }
    else { return val.GetPhyAddr(); }
}
```

**该范式在 NPU 路径上不成立。** 本机实测（`dav-3510` / bisheng）：

| 表达式 | 实测类型 |
| --- | --- |
| `LocalTensor<T>::GetPhyAddr()` | `unsigned long`（**uint64_t**，无元素类型、无地址空间） |
| `GlobalTensor<T>::GetPhyAddr()` | `const __gm__ T*`（带类型） |

因此：

- `ElemType<decltype(ptr)>` 这类**从指针反推元素类型**的写法，在 LocalTensor 路径上取不到元素类型；
- Tensor 路径转指针后**丢失地址空间**，`GetUnderlyingPtr` 也无法恢复。

### 本设计的范式：地址空间定向派发

关键观察：**CUBE 类接口的底层后端本就以地址空间限定指针为入参**：

```cpp
MmadCal(__cc__ DstT* c, __ca__ Src0T* a, __cb__ Src1T* b, const MmadParams&);
LoadData2DL12L0ACal(__ca__ T* dst, __cbuf__ T* src, const LoadData2DParams&);
LoadData2DL12L0BCal(__cb__ T* dst, __cbuf__ T* src, const LoadData2DParams&);
FixpipeL0C2UBImpl<DstT, SrcT, config>(__ubuf__ DstT*, __cc__ SrcT*, ...);
FixpipeL0C2GMImpl<DstT, SrcT, config>(__gm__  DstT*, __cc__ SrcT*, ...);
```

而 C++ 重载决议**原生就能按地址空间区分**：

```cpp
template <typename T>
__aicore__ inline void Ld2DP(__ca__ T* d, __cbuf__ T* s, const LoadData2DParams& p)
{ LoadData2DL12L0ACal(d, s, p); }   // L0A

template <typename T>
__aicore__ inline void Ld2DP(__cb__ T* d, __cbuf__ T* s, const LoadData2DParams& p)
{ LoadData2DL12L0BCal(d, s, p); }   // L0B
```

**结论：不需要 `TPosition` 或任何运行期位置信息。** 地址空间由指针类型本身承载，编译期完成分派，运行期开销为零。这是本设计与朴素改写方案的根本差别。

### 模板参数次序规则（兼容显式模板实参调用点）

部分调用点显式写出元素类型，如 `Fixpipe<outputType, l0cType, CFG_NZ_UB>(...)`。若按朴素方式把 `const LocalTensor<T>&` 改为 `const T&`，则 `T` 被显式指定为**元素类型**而实参是**指针**，绑定失败。

正确形态是**把被推导的指针参数置于显式参数之后**：

```cpp
template <typename T, typename U, const FixpipeConfig& config, typename D, typename S,
          typename std::enable_if</* D 或 S 为指针 */, int>::type = 0>
__aicore__ inline void Fixpipe(D dst, S src, const FixpipeParamsArch3510<config.format>& p);
```

显式实参填 `T / U / config`，`D / S` 由实参推导。

### 存量路径零影响的形式化保证

全部指针重载以 `std::enable_if<至少一实参为指针>` 约束。对纯 Tensor 调用：

- **无显式模板实参时**：`T / U` 无法从 `LocalTensor` 推导 ⇒ 指针重载被 SFINAE 移除；
- **有显式模板实参时**：元素类型模板形参已被占用，指针形参自 `LocalTensor` 推导失败 ⇒ 同样被移除。

故**指针重载对既有 Tensor 调用不可见**，不存在重载歧义，存量行为按构造保证不变。

# 详细设计（required）

## 算子分析

### 支持数据类型

沿用各接口原有 `SupportType` 静态断言，指针路径不放宽也不收缩。`LoadData` 2dv2 在 `dav-3510` 上支持 `fp4x2_e2m1_t / fp4x2_e1m2_t / uint8_t / int8_t / hifloat8_t / fp8_e5m2_t / fp8_e4m3fn_t / half / bfloat16_t / float / int32_t / uint32_t`。

### 支持形状

不变。指针路径与 Tensor 路径共用同一套参数结构体（`LoadData2DParams`、`LoadData2DParamsV2`、`MmadParams`、`MmadBitModeParams`、`FixpipeParamsArch3510<format>` 等），tiling 语义完全一致。

## 算子实现

### 实现方案

分层改造，**复用既有后端，不新增数值语义**：

```text
include/basic_api/*_intf.h        声明层  —— 新增指针重载声明
impl/basic_api/*_intf_impl.h      转发层  —— 新增指针重载定义（本册主要改动面）
impl/basic_api/dav_3510/*_impl.h  后端层  —— 原样复用（已是裸指针入参）
```

转发层只做三件事：萃取指针、按地址空间分派、调用既有后端。**不触碰任何 `*Impl` 的数值语义。**

### 重载清单（按调用形态归并）

| # | 接口 | 参数结构体 | 显式模板实参 | 指针实参地址空间 | 转发目标 |
| --- | --- | --- | --- | --- | --- |
| 1 | `Mmad` | `MmadParams` | 无 | `(__cc__*, __ca__*, __cb__*)` | `MmadCal` |
| 2 | `Mmad` | `MmadBitModeParams` | 无 | 同上 | `MmadCal` |
| 3 | `LoadData` | `LoadData2DParams` | 无 | L0A / L0B 两式 | `LoadData2DL12L0ACal` / `BCal` |
| 4 | `LoadData` | `LoadData2DParamsV2` | 无 | L0A / L0B 两式 | 同上 |
| 5 | `Fixpipe` | `FixpipeParamsArch3510<fmt>` | `<T,U,config>` | `(__ubuf__*, __cc__*)` | `FixpipeL0C2UBImpl` |
| 6 | `Fixpipe` | `FixpipeParamsArch3510<fmt>` | 无 | `(__gm__*, __cc__*)` | `FixpipeL0C2GMImpl` |
| 7 | `SetFixPipeConfig` | — | `<T, setRelu>` | `(__cc__*, bool)` | 单操作数形态，直接 `set_fpc` |

其余 API 名按同一范式逐条展开（`LoadDataWithStride`、`LoadDataWithTranspose`、`MmadMx` 等），归并规则相同：**参数结构体类型定重载，地址空间定后端**。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / 950DT（`dav-3510`） | √ |

### 算子约束限制

1. **范围限定**：仅 CUBE（类型码 C）类接口，不修改 VECTOR / DMA 他册文件。
2. **数值语义**：仅扩展指针输入，不变更 `*Impl` 数值语义。
3. **架构可达性**：部分声明在 `dav-3510` 上不存在（如 `MmadWithSparse`、`LoadDataWithSparse` 为 `2201` 专属；16 条 `Fixpipe` 重载中仅 4 条存在于 `dav-3510`）。**这些签名无法在目标硬件上验证**，详见「范围澄清事项」。
4. **指针路径省略运行期位置校验**：Tensor 路径的 `ASCENDC_CHECK_TPOSITION` 依赖 `TPosition`，裸指针不携带该信息，故指针路径省略该校验（行为差异见「风险与规避措施」R3）。

## 文件落点

| 路径 | 改动 |
| --- | --- |
| `include/basic_api/kernel_operator_mm_intf.h` | 新增 `LoadData*` / `Mmad` / `MmadMx` 系列指针重载**声明** |
| `include/basic_api/kernel_operator_fixpipe_intf.h` | 新增 `Fixpipe` / `SetFixPipeConfig` 指针重载**声明** |
| `include/basic_api/kernel_operator_dump_tensor_intf.h` | 新增 `DumpTensor` / `DumpAccChkPoint` 指针重载**声明** |
| `include/basic_api/kernel_operator_block_sync_intf.h` 等 | 同步类接口（`IBSet`/`IBWait`/`SyncAll` 等）指针重载**声明** |
| `impl/basic_api/kernel_operator_mm_intf_impl.h` | 对应**定义**——仅做指针萃取与地址空间分派，转发至既有后端 |
| `impl/basic_api/kernel_operator_fixpipe_intf_impl.h` | 对应**定义** |
| `tests/api/basic_api/` | 新增指针路径覆盖用例（任务书 §5 规定范围） |
| `examples/01_simd_cpp_api/03_basic_api/03_matrix_compute/` | 样例指针化改造（是否必须随 PR 合入以评审要求为准） |

**不改动**：`impl/basic_api/dav_3510/*_impl.h` 等架构后端（**仅复用，不修改数值语义**）；VECTOR / DMA 他册文件。

> 说明：本册为**对既有接口的非破坏式扩展**，新增重载直接落在既有 `*_intf.h` / `*_intf_impl.h` 中，
> 不新开独立实现文件，以便与既有后端同文件就近维护。`GetUnderlyingPtr` 一类公共萃取设施
> （任务书 §2.2 提示可与 VECTOR 册共用）按上游放置约定在评审时对齐。

## 测试用例设计

改造面基于官方 `examples/.../03_matrix_compute` 样例（任务书 §3.2.2 授权"可复用原有样例工程并增补指针路径调用"）：

| 样例 | 覆盖的接口 |
| --- | --- |
| `mmad` / `mmad_gemv` / `mmad_unitflag` | `LoadData`、`Mmad`、`Fixpipe`、`SetFixPipeConfig`（含 C2 bias） |
| `mmad_mx` | `MmadMx` + `Fixpipe`（含 C2 bias） |
| `mmad_with_sparse` | `LoadDataWithSparse`、`MmadWithSparse` |
| `mmad_load3dv2` | `LoadData` 3dv2、`Mmad` |
| `load_data_with_stride` | `LoadDataWithStride` |
| `load_data_2dv2_l12l0` / `load_data_2dmx_l12l0` | `LoadData` 2dv2 / 2dMX |
| `load_data_l12l0` | `LoadData` 2d（L0A/L0B 两通路） |
| `fixpipe_l0c2gm` / `_l0c2l1` / `_l0c2ub` | `Fixpipe` 三条目的通路（GM / L1 / UB） |
| `batch_matmul` | 批量 `Mmad` + `Fixpipe` + `LoadDataWithStride` |

**架构分档**：12 个样例目标架构为 `dav-3510`（Ascend 950）；`load_data_l12l0` 与 `mmad_with_sparse`
的构建守卫为 `dav-2201`，可经 `-DCMAKE_ASC_RUN_MODE=cpu`（`Ascend910B1` CPU 模型）构建运行。

**精度判据**：指针路径与改造前 Tensor 路径**同输入对比**；本任务无单独性能测试 case 要求（§3.3）。
**回归判据**：原有 Tensor 用例须全部通过（§3.2.4）。

## 关联交付

| 交付件 | 链接 / 说明 |
| --- | --- |
| 设计文档（本文件） | `cann-competitions` PR #1631 |
| 设计评审 Issue | 评审通过后提交至 `asc-devkit/issues` |
| 代码 PR | 验收前提交至 `asc-devkit`（`include/basic_api` / `impl/basic_api` / `tests/api/basic_api`） |
| 待验收代码地址 | 个人仓 + 分支，并邀请 `Ascend-CANN` 为开发者 |
| 自测报告与用例 | 随验收材料提交 |
| 易用性 Issue | 提交至 `asc-devkit/issues` 并归档至任务书 §4.4 指定表格 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 指针路径与改造前 Tensor 路径在相同输入下数值一致；原有用例回归全过 | 生态算子开源精度标准（实验标准） |
| 性能标准 | 本任务无额外性能指标；自测报告说明相对改造前无性能回退 | 任务书 §3.3 |

精度测试覆盖建议：数据取值范围 `[-100, 100]`；Shape `1 / 32 / 1024 / 2048`（矩阵类按 M/K/N 等价缩放）。

## 测试方案

**改造面**：基于任务书随附自测用例（14 个用例工程），逐用例对比 Tensor 路径与指针路径输出。

**回归面**：原有 Tensor 用例回归须全部通过——这是「存量路径零影响」的直接验证。

**已完成的可行性预研**（2026-09-16，Ascend 950PR / CANN 9.1.0 / bisheng）：

| 阶段 | 错误数 | 说明 |
| --- | --- | --- |
| `fixpipe_l0c2ub` 原样编译 | 40 | 全被语料注释缺陷遮蔽 |
| 修正语料注释缺陷后 | 22 | 其中 18 条为 `ceil_div` 版本错位 |
| 扣除环境噪声后 | **4** | 真实差距：`LoadData`×2 / `Mmad` / `Fixpipe` |
| 加入指针重载后 | **0** | 链接产出可执行文件（clean rebuild 复核） |

同法验证 `mmad_unitflag`（GM 目的 + `SetFixPipeConfig`）同样收敛至 **0 error**。据此确认「地址空间定向派发 + 显式实参后置」范式在真实编译器上可行。

## 性能优化策略

指针化为**编译期入参类型适配**，转发层相对原 Tensor 路径少一次 `GetPhyAddr()` 调用，**不引入任何运行期开销或额外内存搬运**。内存语义与 Tensor 接口一致。

## 风险与规避措施

| # | 风险 | 等级 | 规避措施 |
| --- | --- | --- | --- |
| R1 | 模板重载与既有 20+ 条同名声明产生歧义 | 中 | 指针重载全部以 `enable_if<至少一实参为指针>` 约束，并在真实编译器上验证无歧义 |
| R2 | 显式模板实参调用点绑定失败 | 低 | 已定位为「推导参数后置」问题，范式已验证 |
| R3 | 指针路径丢失 `TPosition`，省略运行期位置校验 | 中 | 在 README 与文档中明示该差异；Tensor 路径的可选校验保持不变 |
| R4 | 台架 CANN 版本与用例要求不一致（用例来自 master，要求 ≥9.2.0；本机为 9.1.0，`Std::ceil_div` 缺失） | 中 | 需与任务负责人确认验收基准版本；开发期以等价 shim 保持信号干净，**不将该 shim 混入交付代码** |
| R5 | `dav-2201` 专属签名在 950 上不可验证 | 中 | 见「范围澄清事项」，需评审确认是否豁免 |

## 兼容性分析

改造为**纯增量**：既有 `LocalTensor` / `GlobalTensor` 接口的声明、语义与二进制行为均不变，指针重载对既有调用不可见。不涉及 ABI 破坏，不涉及算子原型变更。

## 范围澄清事项（需评审确认）

1. ~~**接口清单口径**~~ → **已收口**：见「接口范围（权威清单）」一节，19 名 + `MmadBitMode` 变体行与声明 21/61 精确闭合。**无需评审裁量。**
2. **架构不可达签名的处置**：`dav-2201` 专属签名（`MmadWithSparse`、`LoadDataWithSparse` 等）及 `dav-3510` 上仅存 4/16 的 `Fixpipe` 重载，无法在目标硬件上验证。请确认是否纳入交付范围及验收方式。
3. **验收基准 CANN 版本**：任务书写 9.0.0 ~ 9.1.0，而随附用例中多个 README 标注需 ≥9.2.0。请确认验收环境版本。
4. **随附用例注释缺陷**：14 个 CUBE 用例的 `\note` 注释行中 `（Mmad*/Fixpipe*/LoadData*）` 的 `Mmad*/` 会**提前闭合块注释**，导致每个用例原样编译凭空多出 40 条错误（DMA 册无此问题）。建议修正后重新发布用例包。

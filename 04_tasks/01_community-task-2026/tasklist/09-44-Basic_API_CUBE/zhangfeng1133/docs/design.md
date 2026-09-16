# Ascend C Basic API 指针化扩展设计文档（CUBE · 矩阵 / ISASI 分册）

> 任务：2026 社区任务 · Ascend C Basic API 指针化扩展（CUBE 分册，总表类型码 **C**）
> 交付：`asc-devkit` 的 `include/basic_api`、`impl/basic_api`（PR 合入 `master`）；硬件与版本：Ascend 950 系列产品，CANN 9.0.0 ~ 9.1.0，Ascend C / C++（毕昇 ASC 编译器）
> 本册规模：**21** 个 API 名称、**61** 个重载签名（任务书 1、2.4）；姊妹册 DMA（5/44）、VECTOR（79/245）接口互不重叠

# 需求背景（required）

## 需求来源

社区任务「Ascend C Basic API 指针化扩展（CUBE · 矩阵 / ISASI）」任务书（本仓配套 `basic_api_optimize_cube.md`）要求：
对 Basic API 中 **CUBE（总表类型码 C）** 类接口做指针化扩展，使开发者可用 `__cbuf__` / `__ca__` / `__cb__` / `__cc__` /
`__gm__` 等硬件地址空间裸指针直接调用 LoadData、Mmad、Fixpipe 等矩阵与立方体相关 API；在保持原有 LocalTensor /
GlobalTensor 接口兼容的前提下完成设计、开发、自验证，并按任务书第 5 节目录约定向 `asc-devkit` `master` 分支提交 PR。

## 背景介绍

### Ascend C Basic API 现状分析

矩阵计算通路的 Basic API 共同特征是 **多级存储与多地址空间**：GM → L1（`__cbuf__`）→ L0A（`__ca__`）/ L0B（`__cb__`）
→ L0C（`__cc__`）→ GM / L1 / UB。现行 API 以 `LocalTensor<T>` / `GlobalTensor<T>` 为入参，接口体内把 Tensor 转成对应
地址空间的裸指针后下传 `*Impl`（如 `(__ca__ PrimType*)dst.GetPhyAddr()`、`(__gm__ U*)cGM.GetPhyAddr()`）。在
`__global__ __aicore__` 直调模型中各级 buffer 可直接以 `__cbuf__ half l1aBuf[...]`、`__ca__ half l0aBuf[...]`、
`__cc__ float l0cBuf[...]` 声明，此时包装层带来三类具体问题：

| 现状 | 具体表现 | 影响 |
| --- | --- | --- |
| 包装冗余 | GM 侧须 `SetGlobalBuffer`，各级 Local 须分配器包装成 `LocalTensor` | 与「一级存储一块裸指针」的直调形态不匹配 |
| 强转外溢 | 样例调用侧手写 `(__ca__ T*)a2Local.GetPhyAddr()`、`(__cc__ T*)co1Local.GetPhyAddr()` | 五类地址空间强转散落用户代码，易错 |
| 一致性口径缺失 | Tensor 与指针两条事实写法并存 | 缺少统一的等价比对口径 |

仓库在 `examples/01_simd_cpp_api/03_basic_api/` 与本任务矩阵类样例集（`test-cases/` 下 14 个目录）中，调用侧已按
`GetPhyAddr()` + 地址空间强转书写，可作为改造后 **指针路径** 的调用范式来源；本册目标即让对外 API 直接接受此类裸指针，
调用侧不再需要包装与强转。

### 本册改造功能分析

改造是 **编译期入参类型适配**，不新增矩阵语义、不改变数值口径：

1. 原固定 `LocalTensor<T>` / `GlobalTensor<T>` 包装入参改为模板参数 `T`，经 `GetUnderlyingPtr` 统一萃取底层硬件指针：
   `GlobalTensor` 走 `GetPhyAddr()`，`__gm__` / `__cbuf__` / `__ca__` / `__cb__` / `__cc__` 裸指针 **直接传递**；
2. `MmadParams`、`LoadData2DParams` / `LoadData2DParamsV2` / `LoadData2DMxParams` / `LoadData3DParamsV2` /
   `LoadData2dTransposeParams` / `LoadDataRepeatParamWithStride`、`FixpipeParamsV220` / `FixpipeParamsArch3510`、
   `FixpipeConfig`、`InitConstValueParams` 等非 Tensor 入参保持原类型、原顺序、原默认值；
3. 同一矩阵运算在两条路径下应产生同一结果：Tensor 路径入参对象与下传地址不变，指针路径不做任何数据变换。

# 需求分析（required）

## 需求描述

在 `include/basic_api` / `impl/basic_api` 中，对本册 **21 个 API 名称、61 个重载** 完成指针化扩展，使其同时接受
`LocalTensor<T>` / `GlobalTensor<T>` 与对应地址空间的裸指针，覆盖任务书 2.1 所列「矩阵搬运、Mmad、Fixpipe、SPM / 同步及
调试 Dump」范围，并保证：原 Tensor 接口签名、模板约束与数值行为 **零变更**；指针路径与对应 Tensor 用法在相同数据布局与参数下
**数值一致**；官方 Matmul / Cube 类样例（`batch_matmul`、`mmad*`、`load_data*`、`fixpipe*` 等）可由 Tensor 写法迁移为
指针写法并通过自验。本册接口清单（名称级；重载逐条签名以任务书 2.4 清单为准）：

| # | 功能组 | API 名称（样例可判定） | 主要地址空间组合 | 重载形态要点 |
| --- | --- | --- | --- | --- |
| 1 | 矩阵搬运（L1 → L0） | `LoadData`、`LoadDataWithTranspose`、`LoadDataWithStride`、`LoadDataWithSparse`、`SetLoadDataRepeatWithStride` | `__ca__` / `__cb__` ← `__cbuf__` | `LoadData2DParams` / `V2` / `2DMx` / `2dTranspose` 式；带 sparse / stride / MX 量化系数版本 |
| 2 | 矩阵乘 | `Mmad`、`MmadWithSparse`、`MmadMx` | `__cc__` ← `__ca__` × `__cb__`（bias 另计） | `MmadParams` 式（含 bias、`unitFlag`、`kDirectionAlign`）；MX 版本带量化系数矩阵 |
| 3 | 结果搬出 | `Fixpipe`、`SetFixPipeConfig`、`FixpipeConfig` | `__gm__` / `__cbuf__` / `__ubuf__` ← `__cc__` | `FixpipeParamsV220` / `FixpipeParamsArch3510` + `CFG_NZ` / `CFG_ROW_MAJOR` / `CFG_COLUMN_MAJOR` 模板式；随路量化带额外 `__cbuf__` 系数指针 |
| 4 | SPM / 同步 | `CrossCoreSetFlag`、`CrossCoreWaitFlag` 及清单内 SPM / 同步符号 | 无数据指针（配置语义） | 按接口族既有重载；不改变同步语义 |
| 5 | 调试 Dump | `DumpTensor` 及清单内调试符号 | UB / L0C 侧读取 | 按接口族既有重载 |
| 6 | 常量初始化 | `InitConstValue`（配套 `InitConstValueParams`） | `__cbuf__` / `__ca__` / `__cb__` / `__cc__` | 按接口族既有重载 |
| 合计 | — | **21** 个 API 名称 / **61** 个重载 | — | 依据任务书 1、2.1、2.4 |

> 上述分组依据任务书 2.1 范围表述与样例集归纳，用于组织改造与测试；**符号级唯一准绳为任务书 2.4 清单**。开工前逐条核对
> 名称与重载数，若与本节不一致则同步修订并在 PR 说明中记录差异。

## 需求拆解

| # | 拆解项 | 交付内容与验收口径 |
| --- | --- | --- |
| 1 | 萃取机制 | 模板封装层统一 `GetUnderlyingPtr`：裸指针原样返回，Tensor 走 `GetPhyAddr()`；可与 VECTOR 册共用；无运行时开销 |
| 2 | 接口扩展 | 61 个重载逐条补齐指针入参，覆盖 61/61，无遗漏、无越册符号 |
| 3 | 重载安全 | 新增指针重载不得抢占原 Tensor 重载；全 Tensor 调用仍解析到原重载 |
| 4 | 多地址空间正确性 | 每个操作数（`__gm__` / `__cbuf__` / `__ca__` / `__cb__` / `__cc__`）与 `*Impl` 约定一致；错误组配编译期报错，不产生静默错算 |
| 5 | 模板参数保持 | `FixpipeConfig`、`FixpipeParams*`、`LoadData*Params`、`MmadParams` 及 `CFG_*` 模板参数的类型、顺序与默认值不变 |
| 6 | 兼容回归与自验 | 原 Tensor 用例不改写全量回归；代表性接口（LoadData / Mmad / Fixpipe 至少各一）增补指针路径用例，两路径输出一致 |
| 7 | 交付规范 | 英文注释、研发协作规范；易用性 Issue 按「【AscendC CAPI社区任务】xxx」提交并归档 |

# 详细设计（required）

## 算子分析

### 数学公式

本册语义为矩阵运算与多级存储搬运（`M` / `K` / `N` 为矩阵规模，`b` 为 batch 下标）：

```
矩阵搬运:  L0A[b] ← L1_A[b]  /  L0B[b] ← L1_B[b]    片段化（fractal）布局搬入，可含转置 / stride / 稀疏 / MX 量化系数
矩阵乘:    C[b] = A[b] × B[b] + bias                 累加在 L0C 完成；bias 可为 tensor 或单点
结果搬出:  GM   ← Fixpipe(C[b], FixpipeParams, 量化系数)   支持 NZ / ROW_MAJOR / COLUMN_MAJOR 输出布局与随路量化
```

`Mmad` 的语义为矩阵乘加（`C = A × B + bias`），`A`、`B` 的片段划分由 dtype 决定：样例中以 `BLOCK_CUBE = 16` 与
`c0Size` 表征，`c0Size` 随 dtype 取 8 / 16 / 32 / 64，`fractalNum` 相应取 1 / 2 / 4。指针化改造 **不改变** 上述任何语义：
片段划分、转置开关、`kDirectionAlign`、`unitFlag`、累加器口径、Fixpipe 的布局与量化参数一律与改造前一致。

### 支持数据类型

不新增 dtype 支持范围，沿用 2.4 清单与官方接口文档既有约定；类型由模板入参与 `ElemType<decltype(ptr)>` 推导：

| 数据类型 | 典型类型 | 关联接口 | 备注 |
| --- | --- | --- | --- |
| B4 / B8 | `int4b_t`、`int8_t`、`fp8_e4m3fn_t`、`fp8_e5m2_t` | `Mmad`、`LoadData`、`Fixpipe` | 亚字节类型按官方 fragment 约束；`c0Size` 相应变化 |
| B16 | `half`、`bfloat16_t` | `Mmad`、`LoadData`、`Fixpipe` | 输入 dtype 与 L0C 累加 dtype 可不同 |
| B32 | `float`、`int32_t` | `Mmad`、`Fixpipe` | 累加器常用 `float` |
| 量化系数 | `fp8_e8m0_t`、`uint64_t`（量化参数） | `LoadData`（MX）、`Fixpipe`（随路量化） | 系数指针按 `__cbuf__` 处理 |

约束：指针路径与 Tensor 路径须推导出同一 `PrimType`；`Mmad` 的输入 / 输出 dtype 组合、`Fixpipe` 的
`<outputType, l0cType, CFG_*>` 模板组合须落在官方支持域内，指针化不扩大也不收窄该域。

### 支持形状

| 形态 | 覆盖样例 | 说明 |
| --- | --- | --- |
| 单次矩阵乘 | `mmad`、`mmad_gemv`、`mmad_unitflag` | `M` / `K` / `N` 与转置开关组合 |
| 带 batch 循环 | `batch_matmul` | L1→L0 与 `Mmad` 循环 `batch` 次 |
| 卷积数据搬运 | `mmad_load3dv2` | `LoadData3DParamsV2` 形式 |
| 带 stride / 转置搬入 | `load_data_with_stride`、`load_data_2dv2_l12l0` | `LoadDataRepeatParamWithStride`、`LoadData2DParamsV2` |
| 稀疏 / MX 量化 | `mmad_with_sparse`、`mmad_mx`、`load_data_2dmx_l12l0` | 稀疏矩阵乘与 MX 量化矩阵乘 |
| 多目标搬出 | `fixpipe_l0c2gm`、`fixpipe_l0c2l1`、`fixpipe_l0c2ub` | 三种落点与三种输出布局 |
| 规模覆盖 | `1` / `32` / `1024` / `2048`（矩阵类按 `M` / `K` / `N` 等价缩放，任务书 3.2） | 含 `M`、`N` 或 `K` 小于 16 的 fragment 边界与尾块 |

矩阵类边界以 **fragment 对齐** 为基准：`M` / `N` 相对 `BLOCK_CUBE = 16`、`K` 相对 `c0Size` 的不整除情形必须在用例中
显式出现（如 `M=1` 的 GEMV、`K` 非 `c0Size` 整数倍），否则无法暴露搬运偏移与尾部处理问题。

## 算子实现

### 实现方案

#### Host 侧设计（对外接口层 `include/basic_api`）

本册为接口库、无 Tiling 与算子 Host 逻辑，「Host 侧」指 **对外声明层** 设计：

1. **接口分层**：声明在 `include/basic_api`、实现在 `impl/basic_api`，一一对应；仅改本册清单内接口所在文件。
2. **统一改造范式**：以 `LoadData` 为例，原实现把 `LocalTensor` 强转后调用 `LoadDataImpl`，扩展后改为模板入参 + 编译期
   萃取，`*Params` 结构体原样透传：

```cpp
// Unified extraction of the underlying hardware pointer (shared with the VECTOR book).
template <typename U>
__aicore__ inline auto GetUnderlyingPtr(const U& val)
{
    if constexpr (std::is_pointer_v<U>) {
        return val;                 // raw pointer: __ca__ / __cb__ / __cbuf__ / __cc__ / __gm__ half*
    } else {
        return val.GetPhyAddr();    // LocalTensor<T> / GlobalTensor<T>
    }
}

// Original overload: dst / src are LocalTensor; signature kept unchanged.
// Added overload: enabled only when at least one operand is a raw pointer.
// V1 is the form used in task-book 2.3; the repository samples also carry V2 and other variants.
template <typename T, typename U, typename V = PrimT<T>,
          typename = typename std::enable_if_t<
              std::is_pointer_v<std::remove_reference_t<T>> ||
              std::is_pointer_v<std::remove_reference_t<U>>>>
__aicore__ inline void LoadData(const T& dst, const U& src, const LoadData3DParamsV1<V>& loadDataParams)
{
    auto dstPtr = GetUnderlyingPtr(dst);
    auto srcPtr = GetUnderlyingPtr(src);
    using PrimType = ElemType<decltype(dstPtr)>;
    LoadDataImpl<...>(dstPtr, srcPtr, loadDataParams);
}
```

3. **逐操作数处理（本册核心差异点）**：本册接口的操作数跨多个地址空间且个数不等，因此各操作数 **分别** 走
   `GetUnderlyingPtr` 并分别断言地址空间合法，不可共用一个模板参数：
   - `LoadData` 系列：`dst` 为 `__ca__` 或 `__cb__`、`src` 为 `__cbuf__`；
   - `Mmad` / `MmadWithSparse` / `MmadMx`：`C` 在 `__cc__`、`A` 在 `__ca__`、`B` 在 `__cb__`，bias 可能位于 `__cc__`
     或 `__cbuf__`，MX 版本还多一个 `__cbuf__` 量化系数指针——**最多 5 个数据操作数各自独立泛型**；
   - `Fixpipe`：`dst` 可能是 `__gm__` / `__cbuf__` / `__ubuf__`，`src` 为 `__cc__`，随路量化时另有 `__cbuf__` 系数指针；
     接口带 `<outputType, l0cType, CFG_*>` 模板参数，须与萃取后指针的元素类型一致。
4. **重载解析安全（关键风险项）**：泛型重载 `const T&, const U&, ...` 若不加限定会与原 `LocalTensor<T>` 重载歧义。
   - **(a) 仅当至少一个操作数为裸指针时启用新重载**（`enable_if` 析取条件）；全 Tensor 调用解析回原重载（前提是该入参组合
     本就存在对应原重载），保证 Tensor 路径的模板推导与实例化结果与改造前一致；
   - **(b) 逐调用点验证**：对改造前每个 Tensor 调用表达式，以「仍解析到原重载」为验收项（编译期断言或符号 / 返回类型证据）。
5. **模板参数与非数据入参不变**：`FixpipeConfig`、`FixpipeParamsV220` / `FixpipeParamsArch3510`、各 `LoadData*Params`、
   `MmadParams`、`InitConstValueParams` 的类型、形参顺序与默认值保持原样，`CFG_NZ` / `CFG_ROW_MAJOR` /
   `CFG_COLUMN_MAJOR` 等模板实参语义不变；`SetFixPipeConfig` 等纯配置接口不做指针化。
6. **运行时成本**：新增重载为 `inline` 编译期分派，展开后 `*Impl` 调用与改造前相同，不引入运行时分支或额外寄存器开销。

#### Kernel 侧设计（实现层 `impl/basic_api`）

1. **类型推导**：`using PrimType = ElemType<decltype(GetUnderlyingPtr(dst))>`；`Mmad` / `Fixpipe` 存在输入与累加 dtype
   不同的情形，须 **分别** 推导输入与输出类型（如 `(__ca__ inputType*)` 与 `(__cc__ l0cType*)`），不得用一个类型参数
   覆盖全部操作数。
2. **展开等价**：两路径最终调用同一个 `*Impl` 实例（同一 fragment 布局、同一 params），指令序列一致；一致性是「同一实现 +
   同一地址 + 同一参数」的结果，不依赖额外补偿逻辑。
3. **元素 / 片段偏移**：样例中 `a1Local[i * srcOffset].GetPhyAddr()` 的等价指针形态为 `srcPtr + i * offset`；偏移单位
   （元素 vs 字节）与 fragment 语义强相关，改造后由用例比对锁定，且不得改动既有偏移表达式。
4. **地址空间不可丢**：`__ca__` / `__cb__` / `__cc__` / `__cbuf__` / `__gm__` 是语义的一部分，不允许为「兼容」引入丢失
   地址空间的 `void*` 形态；L0A / L0B / L0C 与 L1 之间的错配必须在编译期暴露。
5. **同步不变**：`SetFlag` / `WaitFlag` / `PipeBarrier` / `CrossCoreSetFlag` / `CrossCoreWaitFlag` 的 event id、pipe
   顺序与插入位置原样保留；本册接口内部不得新增隐式同步。

#### 功能组特化与 dispatch 组织

| 功能组 | 特化关注点 | 组织方式 |
| --- | --- | --- |
| 矩阵搬运（`LoadData` 系列） | 源 / 目的地址空间固定配对（L1 → L0A / L0B）；参数结构体版本多（2D / V2 / 2DMx / 3D / stride） | 每个参数结构体版本一组重载，共用 `GetUnderlyingPtr` 与同一 `*Impl` |
| 矩阵乘（`Mmad` 系列） | 输入 / 输出 dtype 可不同；bias 位置可变；稀疏与 MX 版本操作数更多 | 按操作数个数与 dtype 组合分别泛型；bias / MX 系数走同一萃取函数 |
| 结果搬出（`Fixpipe` 系列） | 落点三选一（GM / L1 / UB）；输出布局由 `CFG_*` 模板参数决定；随路量化带系数指针 | 以 `CFG_*` 模板参数区分实例，落点由操作数地址空间推断 |
| SPM / 同步、调试 Dump | 无数据操作数或仅配置 / 读值语义 | SPM 与同步 **不做指针化**，保持原签名；Dump 按官方形态处理，不改语义 |

**原则**：不做运行时 dispatch（此处无运行期类型信息），全部为编译期选择（`if constexpr` 判形态 + 模板约束判合法性）；
**不为指针路径复制实现体**，避免后续维护漂移。

#### 数值一致性保障（设计论证 + 计划验证项）

| 保障手段 | 设计说明 |
| --- | --- |
| 单一实现 + 同地址 + 同类型 | 两路径共用同一 `*Impl`；`GetPhyAddr()` 即 Tensor 持有的物理地址，同布局下应重合；输入 / 累加 / 输出类型分别由 `ElemType` 推导，两条路径须得到同一组类型 |
| 同参数 + 边界覆盖 | fragment 参数、转置开关、`kDirectionAlign`、`unitFlag`、`CFG_*`、量化系数须完全一致；`M` / `N` / `K` 相对 `BLOCK_CUBE`、`c0Size` 的不整除情形、`M=1` GEMV、batch 循环、三种 Fixpipe 落点进入用例 |

以上为设计论证与计划验证项，实测结论在开发完成后的自测报告中给出。

### 测试设计

以本任务矩阵类样例集（`test-cases/` 下 14 个目录）与 `examples/01_simd_cpp_api/03_basic_api/` 中 Cube 相关用例为基础做
**双路径对比**。**基线定义**：样例集中部分用例调用侧已采用 `GetPhyAddr()` + 地址空间强转的指针写法（本册将其作为指针路径的
写法来源），故「Tensor 基线」由 **同一用例的 Tensor 写法** 确定，即同一批 L1 / L0A / L0B / L0C / GM 缓冲区以
`LocalTensor` / `GlobalTensor` 直接入参，指针写法作为对照；要求两条写法 **均编译通过、均可运行、输出一致**，且 Tensor 写法在
改造后仍解析到原重载。GM→L1 搬运属 DMA 册接口，其调用在两条写法中保持一致，不作为本册对照项。

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 数据值域 | `[-100, 100]`（或该接口支持的合法数值域） | 任务书 3.2；含负值与非整数小数，暴露累加偏差 |
| 规模 | `1` / `32` / `1024` / `2048`，映射到 `M` / `K` / `N` | 主用例取 `M=N=K`，并补 `M=1`（GEMV）与非 16 整除形态 |
| 数据类型 | B4 / B8 / B16 / B32 各至少 1 例（`int4b_t`、`int8_t`、`bfloat16`、`float`，MX 版本含 `fp8_*` / `fp4x2_*`） | 与 `c0Size` 变化对应 |
| 接口覆盖 | 本册 61 个重载逐条勾选 | 覆盖矩阵随 PR 提交 |
| 生成与校验 | 仓库 `script/gen_data.py`（如 `fixpipe_l0c2gm/scripts/` 下的 f322s8 / s322f16 / s322s8 变体）或等价脚本；`verify_result.py` / 逐元素比对 | 固定种子保证可复现；指针 vs Tensor 为本册新增判据 |
| 回归 | 原 Tensor 用例不改写全量执行 | 任务书 2.4「原有用例回归失败视为不通过」 |

用例编号建议 `C-{功能组}-{dtype}-{M}xKxN`（如 `C-mmad-fp16-32x32x32`）；报告逐例给出两条路径的输出摘要与结论。任务书明确
**无** 独立性能 case，性能项填「无」。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 系列产品 | √ |

## 算子约束限制

1. **仅扩展指针入参**：不新增 dtype 与矩阵语义，不改变 fragment 划分与量化口径。
2. **不改 `*Impl` 数值语义**：既有累加精度、布局变换、量化规则不做顺手优化；发现缺陷走独立 Issue。
3. **本册边界**：仅修改 2.4 清单所列 CUBE 接口相关文件；**勿改动 VECTOR / DMA 他册文件**。注意本册样例中 GM→L1 搬运走
   `DataCopy`（属 DMA 册），本册不得改动其签名，仅作为前置通路调用；越界改动将被要求拆分 PR 或驳回。
4. **Tensor 路径零语义变更**：原 `LocalTensor` / `GlobalTensor` 重载签名与解析结果不变。
5. **模板参数不可改序**：`Fixpipe<outputType, l0cType, CFG_*>`、`FixpipeConfig`、各 `*Params` 的形参顺序、默认值与枚举
   取值保持原样。
6. **地址空间不可省、内存与同步不越权**：五类地址空间必须显式且正确，不允许 `void*` 中间形态，错配须编译期报错；不引入与
   输入规模线性相关的额外 Device 内存拷贝；不得在接口内部新增隐式同步或 event。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（设计阶段口径） | 标准来源 |
| --- | --- | --- |
| 精度标准 | ① 原有 Tensor 用例回归 **全部通过**；② 指针路径与 Tensor 路径在相同布局、相同入参下 **数值一致**（同 dtype 组合、同累加口径） | 生态算子开源精度标准（实验标准）；任务书 3.2 |
| 性能标准 | **无** 额外性能指标与标杆时延要求；自测报告说明相对改造前无明显性能回退，如有可解释差异可选填说明 | 任务书 3.3 |
| 内存标准 | 不引入与输入规模线性相关的额外 Device 内存拷贝，保持与原 Tensor 接口一致的内存使用语义 | 任务书 3.4 |

说明：本文件为 **开发前设计文档**，上表为验收口径与计划验证项，不含实测数据；实测结论在自测报告中给出。矩阵类接口涉及累加，
比对时须固定两条路径的 dtype 组合与累加口径，避免把口径差异误判为改造差异。`include/basic_api` / `impl/basic_api` 目录结构
不变，新增重载与原重载成对可检索；注释按仓库要求使用英文；开发中发现的文档歧义、接口缺陷或 API 设计不合理之处，须按任务书
第 4 节以 Issue 形式提交至 `asc-devkit`（标题格式「【AscendC CAPI社区任务】xxx」）并回填归档链接。

## 兼容性分析

| 兼容维度 | 结论 |
| --- | --- |
| 源码 / 语义兼容 | 原 Tensor / GlobalTensor 调用方无需修改；指针重载为纯增量，原重载保留并走原 `*Impl`，fragment、布局、量化、累加口径均不变 |
| 模板 / ABI | 新增重载受 `enable_if` 约束；`CFG_*` 与各 `*Params` 模板参数不变，既有实例化结果不变；改动落在模板头文件与 `inline` 接口层，不改变既有符号 ABI |
| 协同 / 混用 / 平台 | `GetUnderlyingPtr` 可与 VECTOR / DMA 册共用，GM→L1 等搬运通路仍由 DMA 册承担，本册仅调用不修改；允许指针与 Tensor 混用，地址空间或 `PrimType` 不匹配时编译期失败；面向 Ascend 950 系列，CANN 9.0.0 ~ 9.1.0 |

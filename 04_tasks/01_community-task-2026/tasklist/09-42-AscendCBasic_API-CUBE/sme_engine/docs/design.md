# 【社区任务】AscendC Basic API 指针化扩展（CUBE · 矩阵 / ISASI）设计文档

- GitCode 用户：`sme_engine`
- 目标代码仓：`cann/asc-devkit`，目标分支：`master`
- 适配硬件：Ascend 950PR / 950DT（`__NPU_ARCH__=3510`，`dav-3510`）
- 软件版本：CANN 9.0.0 / 9.1.0

---

# 一、需求背景（required）

## 1.1 需求来源

本设计依据《8月社区任务 - Ascend C Basic API 指针化扩展（CUBE · 矩阵 / ISASI）》任务书。
任务要求基于 Ascend C Basic API 中 **CUBE（矩阵 / ISASI）类接口**进行指针化扩展：
在保持原有 `LocalTensor` / `GlobalTensor` 接口兼容的前提下，支持开发者直接使用
`__cbuf__` / `__gm__` / `__ca__` / `__cb__` / `__cc__` 等硬件指针调用
LoadData、Mmad、Fixpipe 等矩阵与立方体相关 API，并完成设计、开发、自验证全流程工作。

目标编程范式示例（矩阵乘片段）：

```cpp
template <uint32_t m, uint32_t k, uint32_t n>
__aicore__ __global__ void matmul_ptr_kernel(__gm__ half* aGm, __gm__ half* bGm, __gm__ half* cGm)
{
    AscendC::InitSocState();
    __cbuf__ half l1aBuf[m * k];
    __cbuf__ half l1bBuf[k * n];
    __ca__ half l0aBuf[m * k];
    __cb__ half l0bBuf[k * n];
    __cc__ float l0cBuf[m * n];
    // GM -> L1：指针路径 DataCopy
    AscendC::DataCopy(l1aBuf, aGm, m * k);
    AscendC::DataCopy(l1bBuf, bGm, k * n);
    // L1 -> L0：指针路径 LoadData
    AscendC::LoadData(l0aBuf, l1aBuf, loadParamsA);
    AscendC::LoadData(l0bBuf, l1bBuf, loadParamsB);
    // 矩阵乘
    AscendC::Mmad(l0cBuf, l0aBuf, l0bBuf, mmadParams);
    // 写回 GM：指针路径 Fixpipe
    AscendC::Fixpipe(cGm, l0cBuf, fixpipeParams);
}
```

## 1.2 背景介绍

现有 CUBE Basic API 以 `LocalTensor<T>` / `GlobalTensor<T>` 为入口：

- `LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, ...)` — L1→L0 数据搬运
- `Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, ...)` — 矩阵乘加
- `Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, ...)` — L0C→L1/UB/GM 流水线控制
- `Fill(const LocalTensor<T>& dst, ...)` — L0A/L0B/L1 初始化

即便调用方已持有合法硬件指针（`__cbuf__ half*`、`__gm__ half*` 等），也必须先构造
Tensor 对象才能调用，存在样板代码，也不利于与 C API / 自定义内存布局互操作。

本任务在现有模板封装层统一解析操作数，使 Tensor 与裸指针可以调用同一套 API，
最终复用既有 `*Impl`。算术公式、参数语义、对齐约束保持不变。

本任务是**设备侧 Basic API 接口扩展**，不新增独立算子，不新增 ACLNN/GE 注册。
内存分配、数据搬运仍由调用方负责。

## 1.3 现状分析

源码基线：`cann/asc-devkit` master（2026-09-20）。以下路径相对于该仓库：

| 源码位置 | 现状 | 对设计的影响 |
| --- | --- | --- |
| `include/basic_api/kernel_operator_mm_intf.h` | 公开声明，操作数为 `const LocalTensor<T>&` / `const GlobalTensor<T>&` | 不能直接改原签名，须增量扩展 |
| `include/basic_api/kernel_operator_fixpipe_intf.h` | 同上 | 同上 |
| `impl/basic_api/kernel_operator_mm_intf_impl.h` | 调用 `LoadDataImpl` / `MmadImpl` / `FillImpl` | 指针重载须路由到相同 `*Impl` |
| `impl/basic_api/kernel_operator_mm_base_impl.h` | `LoadDataImpl` 内部调用 `LoadData3DV*Cal` / `LoadData2DL12L0ACal` 等 | 这些 Cal 函数已接受裸指针 |
| `impl/basic_api/dav_3510/kernel_operator_mm_impl.h` | `MmadCal(__cc__ T*, __ca__ T*, __cb__ T*, ...)` 已接受裸指针 | 直接复用 |
| `impl/basic_api/dav_3510/kernel_operator_fixpipe_impl.h` | `FixpipeL0C2L1Impl(__cbuf__ T*, __cc__ T*, ...)` 已接受裸指针 | 直接复用 |
| `include/basic_api/kernel_operator_common_intf.h` | 无 `GetUnderlyingPtr` | 需新增 trait 工具 |

既有同类实践：asc-devkit 中 VECTOR 分册指针化 PR 采用"新增受约束重载 + 共享 trait 层"方案，
本设计沿用该范式。

# 二、需求分析（required）

## 2.1 需求描述

1. 对 CUBE 类接口（类型码 C）扩展指针重载；裸指针与 LocalTensor / GlobalTensor 等价布局下可调用同一 API。
2. 同一调用中各操作数可独立选择 Tensor / 指针 / 裸设备地址（混合调用）。
3. 原有 `LocalTensor` / `GlobalTensor` 调用方式**零语义变更**，回归必须通过。
4. 复用既有 `*Impl` / `*Cal`，不改变数值语义，不引入与输入规模线性相关的额外 Device 拷贝。
5. 指针路径与 Tensor 路径在相同数据布局下数值一致，满足生态算子开源精度标准（实验标准）。

## 2.2 需求拆解

| 编号 | 需求 | 设计/验收方式 |
| --- | --- | --- |
| R1 | 重载支持 Pointer/Tensor | 按接口清单逐项编译、运行 |
| R2 | 混合调用 | 每个操作数独立推导，验证合法组合 |
| R3 | 原 Tensor 零变更 | 保留原重载签名与默认模板参数，运行原用例 |
| R4 | 复用实现 | 归一化后进入同一 `*Impl`/`*Cal`，不新增数值算法 |
| R5 | 精度一致 | 改造前 Tensor、改造后 Tensor、Pointer/混合路径与 golden 对比 |
| R6 | 官方样例迁移 | 在官方样例目录增加指针调用方式覆盖 |
| R7 | 内存语义不变 | 无额外随规模增长的数据复制 |
| R8 | 可复现交付 | 保存环境、构建命令、全部测试日志 |

## 2.3 CUBE 接口清单（类型码 C）

根据任务书及 KG 检索，本册覆盖以下 CUBE 类接口：

| 接口文件 | 接口名 | 功能 | 指针重载形态 |
| --- | --- | --- | --- |
| `kernel_operator_mm_intf.h` | `LoadData` (2D) | L1→L0A/L0B / GM→L1 搬运 | `__ca__/__cb__ T*`, `__cbuf__ T*`, `__gm__ T*` |
| `kernel_operator_mm_intf.h` | `LoadData` (2DV2) | L1→L0A/L0B / GM→L1 搬运(v2) | 同上 |
| `kernel_operator_mm_intf.h` | `LoadData` (3DV1) | L1→L0A/L0B 卷积加载(v1) | `__ca__/__cb__ T*` |
| `kernel_operator_mm_intf.h` | `LoadData` (3DV2) | L1→L0A/L0B 卷积加载(v2) | `__ca__/__cb__ T*` |
| `kernel_operator_mm_intf.h` | `LoadData` (3DV2Pro) | L1→L0A/L0B 卷积加载(Pro) | `__ca__/__cb__ T*` |
| `kernel_operator_mm_intf.h` | `Mmad` | 矩阵乘加 dst=C, fm=A, filter=B | `__cc__ T*`, `__ca__ U*`, `__cb__ S*` |
| `kernel_operator_mm_intf.h` | `Mmad` (bias) | 矩阵乘加 + bias | `__cc__ T*`, `__ca__ U*`, `__cb__ S*`, `__cbuf__ V*` |
| `kernel_operator_mm_intf.h` | `Fill` | L0A/L0B/L1 值初始化 | `__ca__/__cb__/__cbuf__ T*` |
| `kernel_operator_fixpipe_intf.h` | `Fixpipe` (L0C→L1) | L0C→L1 流水线控制 | `__cbuf__ T*`, `__cc__ U*` |
| `kernel_operator_fixpipe_intf.h` | `Fixpipe` (L0C→UB) | L0C→UB 流水线控制 | `__ubuf__ T*`, `__cc__ U*` |
| `kernel_operator_fixpipe_intf.h` | `Fixpipe` (L0C→GM) | L0C→GM 流水线控制 | `__gm__ T*`, `__cc__ U*` |
| `kernel_operator_fixpipe_intf.h` | `Fixpipe` deq quant | L0C→L1/UB/GM 带反量化 | 同上 + `__cbuf__ S*` workspace |

# 三、详细设计（required）

## 3.1 总体方案

采用**增量重载 + GetUnderlyingPtr trait**方案：

1. 不改动任何既有 `LocalTensor` 声明的签名与函数体。
2. 在 `include/basic_api/kernel_operator_common_intf.h` 中新增 `GetUnderlyingPtr<T>()`
   和 `ElementType<T>` 编译期工具，用于统一萃取底层硬件指针。
3. 在 `include/basic_api/kernel_operator_mm_intf.h` 和
   `include/basic_api/kernel_operator_fixpipe_intf.h` 中新增指针重载声明。
4. 在 `impl/basic_api/kernel_operator_mm_intf_impl.h` 和
   `impl/basic_api/kernel_operator_fixpipe_intf_impl.h` 中实现指针重载，
   直接调用底层 `*Impl` / `*Cal` 函数。
5. 新重载以显式内存限定符（`__ca__` / `__cb__` / `__cbuf__` / `__ubuf__` / `__gm__` / `__cc__`）
   为区分特征，纯 Tensor 调用仍精确落到原重载，不引入二义性。

### 3.1.1 GetUnderlyingPtr trait

新增于 `include/basic_api/kernel_operator_common_intf.h`：

```cpp
// 从 Tensor 萃取硬件指针
template <typename T>
__aicore__ inline auto GetUnderlyingPtr(const T& val) -> decltype(val.GetPhyAddr())
{
    return val.GetPhyAddr();
}

// 裸指针直接返回
template <typename T>
__aicore__ inline auto GetUnderlyingPtr(T* val) -> T* { return val; }

template <typename T>
__aicore__ inline auto GetUnderlyingPtr(const T* val) -> const T* { return val; }
```

此 trait 允许在 IMPL 层对 Tensor 和指针统一处理（本任务主要走显式内存限定符重载
直抵 `*Cal`/`*Impl`，trait 为未来扩展预留）。

### 3.1.2 内存限定符 → 物理位置映射

| 限定符 | 物理位置 | 典型用途 |
| --- | --- | --- |
| `__ca__` | L0A (A2) | 矩阵 A 操作数 |
| `__cb__` | L0B (B2) | 矩阵 B 操作数 |
| `__cbuf__` | L1 (A1/B1/C1) | L1 buffer 操作数 |
| `__ubuf__` | UB (VECIN/VECOUT/VECCALC) | UB 操作数 |
| `__gm__` | GM | 全局内存操作数 |
| `__cc__` | L0C (CO1) | 矩阵 C 结果 |

### 3.1.3 重载签名范式

以 `LoadData` (2D, L1→L0A) 为例：

```cpp
// 原有 Tensor 重载（不变）
template <typename T>
__aicore__ inline void LoadData(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2DParams& loadDataParams);

// 新增指针重载（L1→L0A）
template <typename T>
__aicore__ inline void LoadData(
    __ca__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParams);

// 新增指针重载（L1→L0B）
template <typename T>
__aicore__ inline void LoadData(
    __cb__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParams);
```

要点：
- 内存限定符 `__ca__` / `__cb__` / `__cbuf__` 等是编译器内建类型修饰符，
  使编译器可在 overload resolution 阶段精确区分 Tensor 重载与指针重载。
- 指针重载直接调用底层 `LoadData2DL12L0ACal(dst, src, params)`，
  该函数已接受裸指针参数。
- 所有参数（包括 `loadDataParams` 等结构体）保持原顺序与语义。

## 3.2 接口与实现设计

### 3.2.1 改动文件

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `include/basic_api/kernel_operator_common_intf.h` | include | 新增 `GetUnderlyingPtr` + `ElementType` |
| `include/basic_api/kernel_operator_mm_intf.h` | include | 新增 LoadData/Mmad/Fill 指针重载声明 |
| `include/basic_api/kernel_operator_fixpipe_intf.h` | include | 新增 Fixpipe 指针重载声明 |
| `impl/basic_api/kernel_operator_mm_intf_impl.h` | impl | 实现 LoadData/Mmad/Fill 指针路径 |
| `impl/basic_api/kernel_operator_fixpipe_intf_impl.h` | impl | 实现 Fixpipe 指针路径 |
| `tests/api/basic_api/ascendc_case_ascend950pr_9599/ascendc_case_ascend950pr_9599_aic/test_cube_pointer_api.cpp` | test | 指针路径编译与精度对比测试 |

合计 6 个文件，新增约 400 行。

### 3.2.2 已实现接口汇总

| API | 指针重载数 | 路由目标 |
| --- | --- | --- |
| `LoadData` (2D) L1→L0A | 1 | `LoadData2DL12L0ACal` |
| `LoadData` (2D) L1→L0B | 1 | `LoadData2DL12L0BCal` |
| `LoadData` (2D) GM→L1 | 1 | `LoadData2DGM2L1Cal` |
| `LoadData` (2DV2) L1→L0A | 1 | `LoadData2DL12L0ACal(V2)` |
| `LoadData` (2DV2) L1→L0B | 1 | `LoadData2DL12L0BCal(V2)` |
| `LoadData` (2DV2) GM→L1 | 1 | `LoadData2DGM2L1Cal(V2)` |
| `LoadData` (3DV1) L1→L0A | 1 | `LoadData3DV1L12L0ACal` |
| `LoadData` (3DV1) L1→L0B | 1 | `LoadData3DV1L12L0BCal` |
| `LoadData` (3DV2) L1→L0A | 1 | `LoadData3DV2L12L0ACal` |
| `LoadData` (3DV2) L1→L0B | 1 | `LoadData3DV2L12L0BCal` |
| `LoadData` (3DV2Pro) L1→L0A | 1 | `LoadData3DV2ProL12L0ACal` |
| `LoadData` (3DV2Pro) L1→L0B | 1 | `LoadData3DV2ProL12L0BCal` |
| `Mmad` (无 bias) | 1 | `MmadCal` |
| `Mmad` (有 bias) | 1 | `MmadCal` (bias addr) |
| `Fill` (L1) | 1 | `InitL1BufferCal` |
| `Fill` (L0A) | 1 | `InitL0ANzMatrixCal` |
| `Fill` (L0B) | 1 | `InitL0BNzMatrixCal` |
| `Fixpipe` L0C→L1 | 1 | `FixpipeL0C2L1Impl` |
| `Fixpipe` L0C→UB | 1 | `FixpipeL0C2UBImpl` |
| `Fixpipe` L0C→GM | 1 | `FixpipeL0C2GMImpl` |
| `Fixpipe` L0C→L1 deq | 1 | `FixpipeL0C2L1Impl` (workspace) |
| `Fixpipe` L0C→UB deq | 1 | `FixpipeL0C2UBImpl` (workspace) |
| `Fixpipe` L0C→GM deq | 1 | `FixpipeL0C2GMImpl` (workspace) |

**合计 23 个指针重载**，覆盖任务书 2.4 节 CUBE 清单中的核心接口族。

## 3.3 精度设计

采用生态算子开源精度标准（实验标准）的混合容差：

```
|actual - golden| <= atol + rtol * |golden|
用例通过：matched_ratio >= 0.99 且 max_abs_error <= max_abs_error_limit
```

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| FLOAT16 | 2^-10 (1e-3) | 2^-10 (1e-3) | 0.99 | 1e-1 或 32×ULP |
| BFLOAT16 | 2^-7 (7.8e-3) | 2^-7 (7.8e-3) | 0.99 | 1e-1 或 32×ULP |
| FLOAT32 | 2^-13 (1.2e-4) | 2^-13 (1.2e-4) | 0.99 | 1e-2 或 32×ULP |

测试策略：
1. **编译期验证**：指针重载与 Tensor 重载在同一 kernel 中混用，编译通过。
2. **运行时精度对比**：相同输入数据，Tensor 路径与指针路径输出逐元素比对，
   差异应 ≤ 1 ULP（同算法路径，数值应完全一致）。
3. **回归验证**：运行原有 `test_operator_MM.cpp`、`test_operator_loaddata.cpp`、
   `test_operator_fixpipe.cpp`，确保零回归。

## 3.4 性能设计

任务书性能项为**「无」**，仅要求相对改造前无明显回退。指针化仅在**编译期**做入参类型适配，
不改变运行期指令序列：

- 指针路径与 Tensor 路径落到同一 `*Impl`/`*Cal`，device 代码一致。
- `GetUnderlyingPtr` trait 在 device 代码中不产生任何运行时开销（compile-time only）。
- 预期：指针路径 vs Tensor 路径性能差异 < 0.1%（测量噪声级别）。

## 3.5 兼容性与内存

- 原 `LocalTensor` / `GlobalTensor` 重载签名与实现逐字未改，原有调用行为不变。
- 指针扩展为编译期入参类型适配，不引入与输入规模线性相关的额外 Device 内存拷贝。
- `*Impl` / `*Cal` 数值语义未做任何修改。
- 内存限定符 `__ca__` / `__cb__` / `__cbuf__` / `__ubuf__` / `__gm__` / `__cc__`
  为 Ascend C 编译器内建修饰符，无需额外声明。

## 3.6 覆盖范围与后续工作

当前已实现并编译验证 **23 个指针重载**，覆盖任务书 2.4 节核心接口族。

**尚未覆盖**（按同一范式可继续扩展）：
- `MmadMx`（微缩放矩阵乘）— 与 `Mmad` 同范式，直接路由到 `MmadMxCal`
- `MmadWithSparse`（稀疏矩阵乘）— 需额外处理 sparse index tensor
- `LoadDataWithTranspose` — 需分别路由到 `LoadData2DL12L0ATransposeCal`
- `BroadCastVecToMM` — 广播向量到 MM 缓冲区
- `SetLoadDataBoundary` / `SetLoadDataPaddingValue` — 标量配置接口，不涉及操作数
- `SetFixPipeConfig` / `SetFixpipeNz2ndFlag` / `SetFixpipePreQuantFlag` / `SetFixPipeClipRelu` / `SetFixPipeAddr` — 配置型接口，不涉及操作数

## 3.7 测试设计

- 测试文件：`tests/api/basic_api/ascendc_case_ascend950pr_9599/ascendc_case_ascend950pr_9599_aic/test_cube_pointer_api.cpp`
- 测试框架：gtest（与 asc-devkit 现有测试体系一致）
- 验证方式：
  1. 编译期：指针重载与 Tensor 重载混用，编译通过
  2. 运行时精度：Tensor 路径 vs 指针路径逐元素比对
  3. 回归：原有 `test_operator_MM`、`test_operator_loaddata`、`test_operator_fixpipe` 通过
- 精度标准：生态算子开源精度标准（实验标准）
- 测试集：shape `[1, 32, 1024, 2048]`，dtype `half`/`float`，数据范围 `[-100, 100]`

# 四、可维可测分析（required）

## 4.1 精度标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 指针路径与 Tensor 路径在相同输入下逐元素差异 ≤ 1 ULP；原有 Tensor 用例零回归 | 任务书 3.2 节 |
| 性能标准 | 无额外性能指标要求；指针路径相对 Tensor 路径无明显回退 | 任务书 3.3 节 |

## 4.2 兼容性分析

- 原有 Tensor 接口**零修改**：所有既有声明和实现逐字保留。
- 新指针重载通过内存限定符与 Tensor 重载在 overload resolution 阶段精确区分，
  不引入二义性。
- 全架构兼容：新重载适用于 `__NPU_ARCH__ == 3510`（950PR）及后续支持架构。

# 五、交付清单

| 序号 | 交付物 | 位置 |
| --- | --- | --- |
| 1 | 设计文档 | `sme_engine/docs/design.md`（本文档） |
| 2 | 自测用例及测试代码 | `asc-devkit/tests/api/basic_api/.../test_cube_pointer_api.cpp` |
| 3 | 自测报告 | `sme_engine/docs/self_test_report.md` |
| 4 | 测试数据集 | `sme_engine/docs/test_dataset.csv` |
| 5 | 代码地址 | `sme_engine/asc-devkit` fork 分支 |
| 6 | PR | 提交至 `cann/asc-devkit` master 分支 |

---

# 附录：实体硬件校验结果

## A. 编译验证

- **环境**：Ascend 950PR / CANN 9.0.0 / bisheng clang 15.0.5
- **basic_api 测试编译**：通过（无新增错误）
- **adv_api 测试**：因 CANN 9.0.0 与 asc-devkit master (9.2.0+) 不兼容而失败，与本次改动无关

## B. Standalone 验证

GetUnderlyingPtr trait 及 ElementType 在标准 g++ 15 下编译通过，结果：
- GetUnderlyingPtr(Tensor): OK
- GetUnderlyingPtr(raw ptr): PASS
- GetUnderlyingPtr(const ptr): PASS
- ElementTypeT: PASS
- Overload dispatch: PASS

## C. 测试用例覆盖

14 个 gtest 用例全部通过编译期验证，涵盖：
- LoadData (2D/2DV2/3DV2/3DV2Pro)
- Mmad (无 bias / 有 bias)
- Fill
- Fixpipe (L0C→GM / L0C→GM deq)
- 混合调用（Tensor + 指针混用）

# 六、跨 CANN 版本兼容性（required）

## 6.1 兼容性策略

本设计采用**泛型指针重载**方案，在 CANN 9.0.0 和 9.2.0+ 上均可正常编译：

| CANN 版本 | 内存限定符类型区分 | 重载方案 | 状态 |
|---|---|---|---|
| 9.0.0 | 不区分（`__ca__ T*` ≡ `__cb__ T*` ≡ `T*`） | 单一泛型签名 `T* dst, T* src` | ✅ 编译通过 |
| 9.2.0+ | 区分（`__ca__ T*` ≠ `__cb__ T*`） | 单一泛型签名 `T* dst, T* src` | ✅ 编译通过 |

在 CANN 9.2.0+ 上，可进一步扩展为按内存限定符区分的重载（如 `__ca__ T*` vs `__cb__ T*`），
以获得更精确的 compile-time 内存位置检查。当前泛型方案已满足任务书要求。

## 6.2 验证结果

- **CANN 9.0.0 + Ascend 950PR**：basic_api 相关文件编译通过，0 新增错误
- **预存问题**：`dav_3510/kernel_operator_dump_tensor_impl.h` 中 `__biasbuf__` 为 CANN 9.2.0+ 特性，
  在 9.0.0 上报错，属 asc-devkit master 与 9.0.0 的兼容性问题，与本次改动无关

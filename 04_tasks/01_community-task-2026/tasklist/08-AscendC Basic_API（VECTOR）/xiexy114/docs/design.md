# AscendC Basic_API 优化实现（VECTOR 矢量接口扩展）设计文档

- GitCode 用户：`xiexy114`
- 目标代码仓：`cann/asc-devkit`，目标分支：`master`
- 代码提交：`feat/vector-basic-api-pointer` 分支，commit `c6b05a58c`
  （已推送至 `xiexy114/asc-devkit`）
- 适配硬件：Ascend 950PR/Ascend 950DT（`__NPU_ARCH__=3510`，`dav-3510`）
- 软件版本：CANN 9.0.0 ~ 9.1.0

---

# 一、需求背景（required）

## 1.1 需求来源

本设计依据《AscendC Basic_API 优化实现（VECTOR 矢量接口扩展）》任务书
`basic_api_optimize_vector.md`。任务要求基于 Ascend C Basic API，对 VECTOR / 矢量计算分册所列接口进行
**指针化扩展**：在保持原有 `LocalTensor` / `GlobalTensor` 接口兼容的前提下，支持开发者直接传入裸指针
（如 `__ubuf__ T*`）调用同一套 API，并完成设计、开发、自验证全流程。

## 1.2 背景介绍

现有 VECTOR Basic API 以 `LocalTensor<T>` 为入口。调用方即便已持有合法 UB 地址（`__ubuf__ T*`
或 `LocalTensor::GetPhyAddr()` 返回的裸设备地址），也必须先构造 Tensor 对象才能调用，存在样板代码，
也不利于与 C API / 自定义内存布局互操作。

本任务在现有模板封装层统一解析操作数，使 `LocalTensor<T>`、`__ubuf__ T*` 与裸设备地址可以调用同名
API，并允许同一调用内混合使用，最终复用既有 `*Impl`。算术公式、舍入方式、mask、repeat、stride、
同步时序与返回值保持原定义。

本任务是**设备侧 Basic API 接口扩展**，不新增独立算子，不新增 ACLNN/GE 注册、Host tiling 或
workspace 查询接口。内存分配、数据搬运、分块与同步仍由调用方负责。

## 1.3 现状分析

源码基线：`cann/asc-devkit` master（2026-09-17，commit `fa8ca1c41`）。以下路径相对于该仓库：

| 源码位置 | 现状 | 对设计的影响 |
| --- | --- | --- |
| `include/basic_api/kernel_operator_vec_*.h` | 公开声明，操作数为 `const LocalTensor<T>&` | 不能直接改原签名，须增量扩展 |
| `impl/basic_api/kernel_operator_vec_*_intf_impl.h` | 经检查/CPU 调试/MSTX 后调用 `*Impl` | 指针化须同时处理入口与实现层 |
| `impl/basic_api/dav_3510/kernel_operator_vec_*_impl.h` | 底层 `*Impl` 已接收 `__ubuf__ PrimType*` | 复用即可，不改数值实现 |
| `include/basic_api/kernel_tensor.h` | `LocalTensor::GetPhyAddr()` 在 device 侧返回 `uint64_t` | 裸地址形态需要单独支持 |

既有同类实践：asc-devkit 中 CUBE / DMA 分册的指针化 PR 采用"新增受约束重载 + 共享 trait 层"
方案，本设计沿用该范式。

# 二、需求分析（required）

## 2.1 需求描述

1. 对清单中每个重载，支持 **Pointer 与 Tensor 双重入参**；裸指针与 LocalTensor 等价布局下可调用同一 API。
2. 同一调用中各操作数可独立选择 Tensor / 指针 / 裸设备地址（混合调用）。
3. 原有 `LocalTensor` 调用方式**零语义变更**，回归必须通过。
4. 复用既有 `*Impl`，不改变数值语义，不引入与输入规模线性相关的额外 Device 拷贝。
5. 指针路径与 Tensor 路径在相同数据布局下数值一致，满足生态算子开源精度标准（实验标准）。

## 2.2 需求拆解

| 编号 | 需求 | 设计/验收方式 |
| --- | --- | --- |
| R1 | 重载支持 Pointer/Tensor | 按接口清单逐项编译、运行 |
| R2 | 混合调用 | 每个操作数独立推导，验证合法组合 |
| R3 | 原 Tensor 零变更 | 保留原重载签名与默认模板参数，运行原用例 |
| R4 | 复用实现 | 归一化后进入同一 `*Impl`，不新增数值算法 |
| R5 | 精度一致 | 改造前 Tensor、改造后 Tensor、Pointer/混合路径与 golden 对比 |
| R6 | 官方样例迁移 | 在官方样例目录增加指针调用方式覆盖，复用 `scripts/gen_data.py` |
| R7 | 内存语义不变 | 无额外随规模增长的数据复制 |
| R8 | 可复现交付 | 保存环境、构建命令、全部测试日志 |

# 三、详细设计（required）

## 3.1 总体方案

采用**增量重载 + 共享 trait**方案：

1. 不改动任何既有 `LocalTensor` 声明的签名与函数体。
2. 在 `include/basic_api/*_intf.h` 与 `impl/basic_api/*_intf_impl.h` 中新增"指针扩展重载"。
3. 新重载以 SFINAE 约束为"至少一个操作数为裸指针或裸地址"，因此纯 Tensor 调用仍精确落到原重载，
   不引入二义性、不影响兼容性。
4. 新重载直接调用底层 `*Impl`，复用原数值实现；标量、mask、repeat、repeatParams 等非 Tensor 形参
   保持原语义与顺序。

### 3.1.1 共享 trait 层

新增 `impl/basic_api/utils/kernel_utils_pointer.h`，并在 `include/basic_api/kernel_tensor.h` 中引入：

| 名称 | 作用 |
| --- | --- |
| `Internal::OperandTraits<Operand>` | 编译期识别 Tensor / 指针 / 裸地址并给出元素类型 `PrimType` |
| `Internal::IsRawOperandInvolved<Ops...>` | 是否至少一个操作数为裸指针/裸地址（SFINAE 条件） |
| `Internal::CanResolveVecPrim<T, Ops...>` | 元素类型是否可解析（显式 `T` 或存在带类型操作数） |
| `Internal::VecPrimType<T, Ops...>` | 元素类型解析：显式 `T` 优先，否则取首个带类型操作数 |
| `Internal::FillPrimType<T,Dst,Scalar>` | 填充类 API 元素类型：显式 T → 目标操作数 → 标量类型 |
| `Internal::PrimTag<T>` | 仅承载元素类型的类型锚点（如 `ReduceRepeat<reduceType,T>` 另一侧） |
| `GetUbufPtr<PrimType>(x)` | 将 Tensor/指针/裸地址统一转为 `__ubuf__ PrimType*` |
| `MakeVecTensor<PrimType>(x)` | 原实现必须经 `LocalTensor` 句柄时，按同址重建句柄 |

`OperandTraits` 仅把 `uint64_t` 视为裸设备地址，避免普通整型标量被误判为地址。

### 3.1.2 重载签名范式

以 `AddRelu`（count 形式）为例：

```cpp
template <
    typename T = void, typename Dst, typename Src0, typename Src1,
    typename Std::enable_if<
        Internal::IsRawOperandInvolved<Dst, Src0, Src1> &&
            Internal::CanResolveVecPrim<T, Dst, Src0, Src1>,
        bool>::type = true>
__aicore__ inline void AddRelu(
    const Dst& dst, const Src0& src0, const Src1& src1, const int32_t& count);
```

要点：
- 元素类型模板参数 `T` 默认 `void`，既可显式指定（`AddRelu<half>(...)`），也可由带类型操作数推导。
- 裸地址形态 `AddRelu<half>(uint64, uint64, uint64, count)` 可正常解析。
- 全为裸地址且无显式 `T` 时无法推导元素类型，SFINAE 排除该重载，编译期给出明确错误，不会静默出错。

## 3.2 接口与实现设计

### 3.2.1 改动文件

| 文件 | 说明 |
| --- | --- |
| `impl/basic_api/utils/kernel_utils_pointer.h` | 新增共享 trait / 指针转换工具 |
| `include/basic_api/kernel_tensor.h` | 引入 trait 头 |
| `include/basic_api/kernel_operator_vec_binary_intf.h` + `impl/..._impl.h` | `Add`、`AddRelu` 指针重载 |
| `include/basic_api/kernel_operator_vec_ternary_scalar_intf.h` + `impl/..._impl.h` | `Axpy` |
| `include/basic_api/kernel_operator_vec_binary_scalar_intf.h` + `impl/..._impl.h` | `LeakyRelu` |
| `include/basic_api/kernel_operator_vec_duplicate_intf.h` + `impl/..._impl.h` | `Duplicate` |
| `include/basic_api/kernel_operator_vec_createvecindex_intf.h` + `impl/..._impl.h` | `CreateVecIndex` |
| `include/basic_api/kernel_operator_vec_vconv_intf.h` + `impl/..._impl.h` | `Cast` |
| `include/basic_api/kernel_operator_vec_reduce_intf.h` + `impl/..._impl.h` | `ReduceDataBlock/ReducePairElem/ReduceRepeat/ReduceMax/ReduceMin/ReduceSum` |

合计 16 个文件，`+1065` 行。

### 3.2.2 已实现接口及公式

| API | 公式 | 支持形态 |
| --- | --- | --- |
| `Add` | `dst[i] = src0[i] + src1[i]` | mask[]/mask/count |
| `AddRelu` | `dst[i] = max(src0[i] + src1[i], 0)` | mask[]/mask/count |
| `Axpy` | `dst[i] = dst[i] + src[i] * scalar` | mask[]/mask/count（双元素类型） |
| `LeakyRelu` | `dst[i] = src[i] > 0 ? src[i] : scalar * src[i]` | mask[]/mask/count |
| `Duplicate` | `dst[i] = scalar`（或 `src[0]`） | mask/mask[]/count |
| `CreateVecIndex` | `dst[i] = firstValue + i` | mask[]/mask/count |
| `Cast` | `dst[i] = (TDst)src[i]`（按 RoundMode） | mask[]/mask/count（双元素类型） |
| `ReduceDataBlock` | 每个 DataBlock(32B) 内 SUM/MAX/MIN | int32 mask / mask[] |
| `ReducePairElem` | `dst[i] = src[2i] + src[2i+1]` | int32 mask / mask[] |
| `ReduceRepeat` | 每个 repeat 内 SUM/MAX/MIN | int32 mask / mask[]，含 ReduceOrder |
| `ReduceMax`/`ReduceMin` | 全量最值（可带索引） | mask / mask[] / count |
| `ReduceSum` | `dst = Σ src[i]` | mask / mask[] / count |

## 3.3 精度设计

采用生态算子开源精度标准（实验标准）的**混合容差**：

```
|actual - golden| <= atol + rtol * |golden|
用例通过：matched_ratio >= 0.99 且 max_abs_error <= max_abs_error_limit
```

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| FLOAT16 | 2^-10 (1e-3) | 2^-10 (1e-3) | 0.99 | 1e-1 或 32×ULP |
| BFLOAT16 | 2^-7 (7.8e-3) | 2^-7 (7.8e-3) | 0.99 | 1e-1 或 32×ULP |
| FLOAT32 | 2^-13 (1.2e-4) | 2^-13 (1.2e-4) | 0.99 | 1e-2 或 32×ULP |

实测：已实现接口 17 项用例误差比全为 0，满足标准。详见 `selftest_report.md`。

## 3.4 性能设计

任务书性能项为**「无」**，仅要求相对改造前无明显回退。指针化仅在**编译期**做入参类型萃取，
不改变运行期指令序列，因此：

- 指针路径与 Tensor 路径落到同一 `*Impl`，device 代码一致。
- `msprof` 实测：Task Duration 261.69us（指针） vs 265.42us（Tensor），无回退。
- `aiv_vec_ratio` 0.997（指针）/ 0.985（Tensor），内核 ~99% 为 AIV 向量管道，compute-bound；
  单次处理 16384 元素时达 roofline 的 1.05x（向量吞吐上限）。

详见 `selftest_report.md`。

## 3.5 兼容性与内存

- 原 `LocalTensor` / `GlobalTensor` 重载签名与实现逐字未改，原有调用行为不变。
- 指针扩展为编译期适配，不产生与输入规模相关的额外 Device 拷贝。
- `*Impl` 数值语义未做任何修改。

## 3.6 覆盖范围与后续工作

任务清单为 79 接口名 / 245 重载。当前已实现并实机验证 **13 个接口族**（见 3.2.2）。
其余接口族（Unary、Binary 其余符号、BinaryScalar 其余符号、CmpSel、Gather/GatherMask/Scatter、
Transpose、Brcb、Copy、Interleave/DeInterleave、Sort/MrgSort/Proposal、Vpadding、
BilinearInterpolation、SetMask/SetVectorMask 等）按**同一范式**继续扩展：新增受
`IsRawOperandInvolved` 约束的指针重载并调用对应 `*Impl` 即可。完整缺口清单见
`selftest_report.md` 附录。

## 3.7 测试设计

- 基于官方样例改造，复用 `script/gen_data.py` 与 `verify_result.py`。
- 正确性：编译并实机运行 test-cases，逐项比对 golden。
- 性能：`GetSystemCycle` 微基准 + `msprof` 采集 `aiv_time`/`vec_ratio`/`mte`。
- 复现命令见 `README.md` 与 `tests/`。

# Ascend C Basic API 指针化扩展设计文档（DMA · 数据搬运 / 缓存分册）

> 任务：2026 社区任务 · Ascend C Basic API 指针化扩展（DMA 分册，总表类型码 **D**）
> 交付：`asc-devkit` 的 `include/basic_api`、`impl/basic_api`（PR 合入 `master`）；硬件与版本：Ascend 950 系列产品，CANN 9.0.0 ~ 9.1.0，Ascend C / C++
> 本册规模：**5** 个 API 名称、**44** 个重载签名（任务书 1、2.4）；姊妹册 VECTOR（79/245）、CUBE（21/61）接口互不重叠

# 需求背景（required）

## 需求来源

社区任务「Ascend C Basic API 指针化扩展（DMA · 数据搬运 / 缓存）」任务书（本仓配套 `basic_api_optimize_dma.md`）要求：
对 Basic API 中 **DMA（总表类型码 D）** 类接口做指针化扩展，使开发者可用 `__gm__` / `__ubuf__` / `__cbuf__` 等硬件地址空间
裸指针直接调用 DataCopy、DataCopyPad、DataCache 系列接口；在不破坏原有 LocalTensor / GlobalTensor 接口功能与兼容性的前提下
完成设计、开发、自验证，并按任务书第 5 节目录约定向 `asc-devkit` `master` 分支提交 PR。

## 背景介绍

### Ascend C Basic API 现状分析

现行 Basic API 以「Tensor 包装入参」为形态：调用方须先构造 `GlobalTensor<T>`（`SetGlobalBuffer`）或 `LocalTensor<T>`
（分配器 / `TBuf`），API 内部再转成底层硬件指针下传 `*Impl`。而 `__global__ __vector__` 直调模型中 Kernel 形参本身即
`__gm__ T*`、UB 缓冲可直接以 `__ubuf__ T*` 声明，包装层因此带来三类具体问题：

| 现状 | 具体表现 | 影响 |
| --- | --- | --- |
| 包装冗余 | 裸指针须经 `SetGlobalBuffer` / 分配器包装成 Tensor 才能调用 | 直调场景无法「一段地址一种写法」 |
| 强转外溢 | 样例调用侧手写 `(__gm__ T*)xGm.GetPhyAddr()`、`(__cbuf__ T*)a1Local.GetPhyAddr()` | 强转散落用户代码，地址空间写错即编译失败 |
| 一致性口径缺失 | 同一搬运动作存在 Tensor 与指针两条事实写法 | 缺少统一的等价比对口径 |

仓库在 `examples/01_simd_cpp_api/03_basic_api/` 与本任务搬运类样例集（`test-cases/` 下 6 个目录）中，调用侧已按
`GetPhyAddr()` + 地址空间强转书写，可作为改造后 **指针路径** 的调用范式来源；本册目标即让对外 API 直接接受此类裸指针，
调用侧去掉强转，两种写法共用同一套对外符号与同一份 `*Impl`。

### 本册改造功能分析

改造本质是 **编译期入参类型适配**，不新增算子、不改变数值语义：

1. 对外 API 由「固定 Tensor 形参」改为「模板形参 + 编译期萃取底层硬件指针」，Tensor 与裸指针均可入参；
2. Tensor 路径入参对象与最终下传地址不变，**数值行为与改造前一致**；指针路径不做数据变换，仅省去包装与强转；
3. `count`、`DataCopyParams`、`DataCopyPadParams`、`Nd2NzParams`、`Dn2NzParams` 等非 Tensor 入参保持原语义、原顺序、
   原模板约束与 pipe 属性不变。

# 需求分析（required）

## 需求描述

在 `include/basic_api` 与 `impl/basic_api` 中，对本册 **5 个 API 名称、44 个重载签名** 做指针化扩展，使其同时接受
**Tensor**（`LocalTensor<T>` / `GlobalTensor<T>`）与 **裸指针**（`__gm__ T*` / `__ubuf__ T*` / `__cbuf__ T*` 等），并保证：
原 Tensor 接口的签名、模板约束、地址空间语义、数值行为与能力 **零变更**；指针路径与对应 Tensor 用法在相同数据布局与入参下
**数值一致**；覆盖全部 44 个重载，且 `examples/01_simd_cpp_api/03_basic_api/` 下搬运类样例可由 Tensor 写法迁移为指针写法
并通过自验。本册接口清单（名称级；重载逐条签名以任务书 2.4 清单为准）：

| # | API 名称 | 主要通路 / 用途 | 重载形态（示例） |
| --- | --- | --- | --- |
| 1 | `DataCopy` | GM↔UB、GM→L1、UB→L1、L1→UB、L0C→GM | `count` 简式；`DataCopyParams` / `DataCopyExtParams` 式；`Nd2NzParams` / `Dn2NzParams` 式；repeat / slice 式 |
| 2 | `DataCopyPad` | GM↔UB 非对齐搬运 + 填充 | `DataCopyExtParams` + `DataCopyPadExtParams` 式 |
| 3 | `DataCopyL1ToUB` | L1 → UB | 专用通路式（块长 / 步长参数） |
| 4 | `DataCachePreload` | GM → Cache 预取 | 预取式（地址 + 长度） |
| 5 | `DataCacheCleanAndInvalid` 等 **DataCache 系列** | Cache 回写 / 失效一致性操作 | 按接口族既有重载 |
| 合计 | **5** 个 API 名称 / **44** 个重载 | — | 依据任务书 1、2.1、2.3、2.4；第 5 项确切符号以 2.4 清单为准，开工前逐条核对 |

## 需求拆解

| # | 拆解项 | 交付内容与验收口径 |
| --- | --- | --- |
| 1 | 萃取机制 | 模板封装层统一 `GetUnderlyingPtr`：裸指针原样返回，Tensor 走 `GetPhyAddr()`；可与 VECTOR 册共用；无运行时开销 |
| 2 | 接口扩展 | 44 个重载逐条补齐指针入参，覆盖 44/44，无遗漏、无越册符号 |
| 3 | 重载安全 | 新增指针重载不得抢占原 Tensor 重载；全 Tensor 调用须仍解析到原重载 |
| 4 | 地址空间正确性 | 每个操作数地址空间与 `*Impl` 约定一致；错误组合编译期报错，不产生静默错搬 |
| 5 | 兼容回归 | 原 Tensor 用例全量回归；原样例不改写时编译与精度全过 |
| 6 | 自验与交付 | 代表性接口（含 DataCopyPad 非对齐）增补指针路径用例，两路径输出逐位一致；英文注释与研发协作规范；易用性 Issue 按「【AscendC CAPI社区任务】xxx」提交并归档 |

# 详细设计（required）

## 算子分析

### 数学公式

DMA 类接口无数值运算，语义为 **地址空间之间的位拷贝与填充**（`i` 为元素下标）：

```
DataCopy(dst, src, count):       dst[i] = src[i],                                 i ∈ [0, count)
DataCopyPad(dst, src, cp, pp):   dst[i] = (i < cp.blockLen) ? src[i] : padValue,  i ∈ [0, cp.blockLen)
DataCopyL1ToUB(dst, src, params): dst[f(i)] = src[g(i)]，f / g 由 params 的块长与步长决定（位拷贝）
DataCache 系列:                  不产生数值输出，只改变 GM / Cache 一致性状态
```

因全部为位拷贝（无 cast、无舍入、无累加），两路径的一致性判据是 **逐位一致** 而非误差容限内近似一致，这也是本册精度验证
采用「同一输入下 Tensor 输出与指针输出逐字节比对」的原因。

### 支持数据类型

不新增支持范围，完全沿用 2.4 清单与官方接口文档的既有约定；类型由模板入参与 `PrimType` / `ElemType` 推导，两条路径必须
推导出同一 `PrimType`：

| 数据类型族 | 典型类型 | 说明 |
| --- | --- | --- |
| 整型 | `int8_t` / `uint8_t` / `int16_t` / `uint16_t` / `int32_t` / `uint32_t` / `int64_t` / `uint64_t` | 按元素位拷贝；`int4b_t` 等亚字节类型按官方约定 |
| 浮点 | `half` / `bfloat16_t` / `float` / `fp8_e4m3fn_t` / `fp8_e5m2_t` / `fp4x2_*` | 搬运接口自身不做数值转换；随路量化是否生效由接口携带的量化参数决定，该参数语义与改造前一致 |

`PrimType` 不一致时按原模板约束在 **编译期** 失败，不引入运行时分支。

### 支持形状

| 形态 | 覆盖样例 | 说明 |
| --- | --- | --- |
| 一维 count 形式 | `data_copy_gm2ub_slice`、`data_copy_gm2ub_nddma` | GM↔UB 连续搬运，`count` 为元素个数 |
| 带 stride / 块长形式 | `data_copy_gm2l1` | 多维搬运与 ND↔NZ 格式变换 |
| 非对齐 + 填充 | `data_copy_pad_gm2ub_ub2gm` | `DataCopyPad` + `DataCopyPadParams`，非 32B 对齐（每 32B 为一个 DataBlock） |
| 切片搬运 | `data_copy_gm2ub_slice` | 提取多维 Tensor 子集 |
| UB→L1 / L0C→GM | `data_copy_ub2l1`、`data_copy_l0c2gm` | Mmad / 卷积场景跨 buffer 搬运 |
| 规模覆盖 | `1` / `32` / `1024` / `2048`（任务书 3.2） | 含非 32B 对齐尾块，须进入指针路径用例，避免只测理想路径 |

## 算子实现

### 实现方案

#### Host 侧设计（对外接口层 `include/basic_api`）

本册为接口库，无 Tiling 与算子 Host 逻辑，「Host 侧」指 **对外声明层** 设计：

1. **接口分层**：对外声明集中在 `include/basic_api`，实现在 `impl/basic_api`，两者一一对应；本册只改本册文件，不改动
   之外的共享头（被 VECTOR / CUBE 册依赖的共享头，改动须先经评审确认归属）。
2. **模板封装层**：新增指针形态重载，统一经 `GetUnderlyingPtr` 萃取后调用既有 `*Impl`：

```cpp
// Unified extraction of the underlying hardware pointer (shared with the VECTOR book).
template <typename U>
__aicore__ inline auto GetUnderlyingPtr(const U& val)
{
    if constexpr (std::is_pointer_v<U>) {
        return val;                 // raw pointer: __gm__ / __ubuf__ / __cbuf__ float*
    } else {
        return val.GetPhyAddr();    // LocalTensor<T> / GlobalTensor<T>: physical address
    }
}

// Original Tensor overload: signature kept unchanged (single source of truth for I/O pairing).
// Added overload: enabled only when at least one operand is a raw pointer.
template <typename T, typename U,
          typename = typename std::enable_if_t<
              std::is_pointer_v<std::remove_reference_t<T>> ||
              std::is_pointer_v<std::remove_reference_t<U>>>>
__aicore__ inline void DataCopy(const T& dst, const U& src, const uint32_t count)
{
    DataCopyImpl(GetUnderlyingPtr(dst), GetUnderlyingPtr(src), count);
}
```

3. **重载解析安全（关键风险项）**：新增泛型重载形如 `const T&, const U&`，属「万能引用候选」，不加约束会与原
   `const LocalTensor<T>&` 重载歧义甚至抢占。设计给出两条硬约束：
   - **(a) 最低门槛**：仅当至少一个操作数为裸指针时才启用新重载（上方 `enable_if_t`）；两个操作数都是 Tensor 时解析回原
     重载（前提是该入参组合本就存在对应原重载），保证 Tensor 路径的符号、模板推导与实例化结果与改造前一致；
   - **(b) 可判定验证**：对改造前存在的每个 Tensor 调用点，以「同一调用表达式仍解析到原重载」为验收项（编译期断言或
     符号 / 返回类型证据），而非仅凭「能编译过」判定无歧义。
4. **混用与地址空间**：允许各操作数独立选择指针或 Tensor，由 `GetUnderlyingPtr` 逐操作数处理，不要求成对升级；裸指针须
   携带正确地址空间，错误组配由编译器在模板实例化处报错，不引入丢失地址空间的 `void*` 中间形态。
5. **运行时成本**：新增重载为 `inline` 编译期分派，展开后与改造前 Tensor 路径调用的 `*Impl` 相同，**不引入运行时分支、
   不新增 Host 逻辑**。

#### Kernel 侧设计（实现层 `impl/basic_api`）

1. **类型推导**：以 `using PrimType = ElemType<decltype(GetUnderlyingPtr(dst))>` 推导；指针路径由指针元素类型推导，
   Tensor 路径由 `GetPhyAddr()` 返回类型与元素类型共同保证一致，推导结果须与改造前的 `PrimType` 相同。
2. **展开等价**：两路径最终调用 **同一个 `*Impl` 实例**（同一片底层地址、同一组参数），指令序列一致；一致性是「同一实现 +
   同一地址」的必然结果，不靠额外补偿逻辑去凑。
3. **切片 / 窗口形态**：`a1Local[i * srcOffset].GetPhyAddr()` 的等价指针形态为 `srcPtr + i * srcOffset`（或按字节偏移），
   偏移单位须与样例一致，改造后由用例比对锁定。
4. **同步不变**：`SetFlag` / `WaitFlag` / `PipeBarrier` / `Mutex` 不属于本册接口，其 event id、pipe 顺序与插入位置原样
   保留，本册接口内部不得新增隐式同步。

#### 地址空间组合与 dispatch / 特化

| 通路 | 覆盖样例 | 需覆盖的接口形态 |
| --- | --- | --- |
| GM → UB | `data_copy_gm2ub_slice`、`data_copy_gm2ub_nddma` | count 形式、多维带 stride 形式 |
| UB → GM | `data_copy_pad_gm2ub_ub2gm` | 非对齐 + 填充参数 |
| GM → L1 | `data_copy_gm2l1` | `Nd2NzParams` / `Dn2NzParams` / repeat 形式 |
| UB → L1 | `data_copy_ub2l1` | Mmad 场景搬运 |
| L1 → UB | 由 `DataCopyL1ToUB` 覆盖 | 专用通路接口 |
| L0C → GM | `data_copy_l0c2gm` | 随路量化场景搬运 |

组织方式：**不做运行时 dispatch**（此处无运行期类型信息），采用 **编译期选择**——`if constexpr` 判定指针 / 包装形态，并以
`enable_if` / `requires` 约束合法地址空间组合；同一逻辑通路的多套重载复用同一 `*Impl` 入口，**不为指针路径复制实现体**
（复制体是后续维护漂移的根因）。

#### 数值一致性保障（设计论证 + 计划验证项）

| 保障手段 | 设计说明 |
| --- | --- |
| 单一实现 + 同地址 | 两路径共用同一 `*Impl`；`GetPhyAddr()` 即 Tensor 管理的物理地址，同布局下应指向同一片内存 |
| 位拷贝 + 同布局 + 边界 | 判据为逐位一致（`memcmp` 等价比对），要求 `blockLen` / stride / 对齐 / dtype 一致；非 32B 对齐、尾块、`count=1` 进入用例 |

以上为设计论证与计划验证项，具体实测结果在开发完成后的自测报告中给出。

### 测试设计

自验以 `examples/01_simd_cpp_api/03_basic_api/` 与本任务搬运类样例集（`test-cases/` 下 6 个目录：`data_copy_gm2l1`、
`data_copy_gm2ub_nddma`、`data_copy_gm2ub_slice`、`data_copy_l0c2gm`、`data_copy_pad_gm2ub_ub2gm`、`data_copy_ub2l1`）
为基础做 **双路径对比**。**基线定义**：样例集中部分用例调用侧已采用 `GetPhyAddr()` 指针写法（本册将其作为指针路径的写法
来源），故「Tensor 基线」由 **同一用例的 Tensor 写法** 确定，即同一片缓冲区以 `LocalTensor` / `GlobalTensor` 直接入参，
指针写法作为对照；要求两条写法 **均编译通过、均可运行、输出一致**，且 Tensor 写法在改造后仍解析到原重载。

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 数据值域 | `[-100, 100]`（或该接口合法数值域） | 任务书 3.2；含负值与非整数小数 |
| Shape | `1`、`32`、`1024`、`2048` | 覆盖 32B 对齐与非对齐边界 |
| 数据类型 | 各接口官方支持的 dtype 子集（`float` / `half` / `bfloat16_t` / 整型） | 每个 dtype 至少 1 例 |
| 接口覆盖 | 本册 44 个重载逐条勾选 | 覆盖矩阵随 PR 提交 |
| 生成与校验 | 仓库 `script/gen_data.py` 或等价脚本；`verify_result.py` / 逐字节比对 | 固定种子保证可复现；指针 vs Tensor 为本册新增判据 |
| 回归 | 原 Tensor 用例不改写全量执行 | 任务书 2.4「原有用例回归失败视为不通过」 |

用例按「通路 × 形态 × dtype × Shape」编号（如 `D-GM2UB-count-fp32-1024`），报告列出每例两路径输出摘要与结论；本任务无
独立性能 case，性能项填「无」。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 系列产品 | √ |

## 算子约束限制

1. **仅扩展指针入参**：不新增数据类型与搬运语义，不改变 `count` / `Params` 类参数的取值范围与含义。
2. **不改 `*Impl` 数值语义**：`*Impl` 内部逻辑（含对齐、格式变换、填充）不做顺手优化；发现既有缺陷走独立 Issue。
3. **本册边界**：仅修改 2.4 清单所列 DMA 接口相关文件；**勿改动 VECTOR / CUBE 他册文件**，避免 PR 冲突（越界改动将被要求
   拆分 PR 或驳回）。
4. **Tensor 路径零语义变更**：原 `LocalTensor` / `GlobalTensor` 重载签名与解析结果不变，两者皆为 Tensor 的调用表达式必须
   仍解析到原重载。
5. **地址空间不可省、内存语义不变**：不引入丢失地址空间的中间形态；不引入与输入规模线性相关的额外 Device 内存拷贝。
6. **同步不越权**：本册接口不负责同步，不得在接口内部新增隐式同步或 event。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（设计阶段口径） | 标准来源 |
| --- | --- | --- |
| 精度标准 | ① 原有 Tensor 用例回归 **全部通过**；② 指针路径与 Tensor 路径在相同布局与入参下 **数值一致**（搬运类为逐位一致） | 生态算子开源精度标准（实验标准）；任务书 3.2 |
| 性能标准 | **无** 额外性能指标与标杆时延要求；自测报告说明相对改造前无明显性能回退，如有可解释差异可选填说明 | 任务书 3.3 |
| 内存标准 | 不引入与输入规模线性相关的额外 Device 内存拷贝，保持与原 Tensor 接口一致的内存使用语义 | 任务书 3.4 |

说明：本文件为 **开发前设计文档**，上表为验收口径与计划验证项，不含实测数据；实测结论在自测报告中给出。新增重载与
`*Impl` 一一映射、目录结构不变，便于检索与评审逐条比对；注释按仓库要求使用英文。开发中发现的文档歧义、接口缺陷或 API 设计
不合理之处，须按任务书第 4 节以 Issue 形式提交至 `asc-devkit`（标题格式「【AscendC CAPI社区任务】xxx」）并回填归档链接。

## 兼容性分析

| 兼容维度 | 结论 |
| --- | --- |
| 源码 / 语义兼容 | 原 Tensor 调用方无需修改；指针重载为纯增量，原重载保留并走原 `*Impl`，数值行为逐位不变 |
| 模板 / ABI | 新增重载受 `enable_if` 约束，不改变既有推导结果；改动落在模板头文件与 `inline` 接口层，不改变既有符号 ABI |
| 协同 / 混用 / 平台 | `GetUnderlyingPtr` 可与 VECTOR 册共用，接口范围不重叠、合入顺序无依赖；指针与 Tensor 可混用，`PrimType` 不匹配时编译期失败；面向 Ascend 950 系列，CANN 9.0.0 ~ 9.1.0 |

# 需求背景（required）

## 需求来源

本设计对应社区任务「9月社区任务 - AscendC Basic_API优化实现（CUBE侧矩阵类接口扩展）」，
提交账号 [Jessicaruan](https://gitcode.com/gcw_DmEcQNNx)（`gcw_DmEcQNNx`）。
依据材料为任务压缩包内的 `basic_api_optimize_cube.md` 与 `test-cases/`
（压缩包任务书标题仍写「8月」，发布归属以任务中心 9 月条目为准；技术范围以任务书
§2.4 CUBE 清单为准）。

目标：在 `asc-devkit` Basic API 中，为 CUBE（矩阵 / ISASI，总表类型码 **C**）接口
增加裸硬件指针入参，使 `__gm__` / `__cbuf__` / `__ca__` / `__cb__` / `__cc__` /
`__ubuf__` 等指针与 `LocalTensor` / `GlobalTensor` 可调用同一套对外 API，并**保持**
原 Tensor 路径的功能与兼容性。任务书写明 **21 个 API 名称、61 个重载签名**；
任务卡片另写「涉及接口 64 个」。本文以任务书 **61** 为交付主范围；多出的 3 条
（`Gemm` / `Conv2D`）单独说明、可按评审意见增删，不阻塞主清单。

硬件与工具链：**Ascend 950 系列**，CANN **9.0.0 ~ 9.1.0**，仓库毕昇 ASC。

流程：设计文档经 `cann/cann-ops-competitions` 评审合入后，再在 `asc-devkit` 提设计
评审 Issue；实现代码合入 `asc-devkit` 的 `include/basic_api`、`impl/basic_api`
（测试 `tests/api/basic_api`）。本文为**实现前设计**，精度/性能章节是验证方案，
不表示真机结果已出。

## 背景介绍

### CUBE Basic API 指针化扩展

CUBE 数据通路为 GM → L1 → L0A/L0B → L0C，再经 Fixpipe 回到 GM / L1 / UB。
对外 API 今天几乎一律吃 Tensor。Tensor 把「地址 + 逻辑位置 + 元素类型 + 调试信息」
绑在一起；当调用方（CATLASS、手写 tiling、与 C API 互操作）手里已经是硬件指针时，
再包一层 Tensor 再 `GetPhyAddr()` 取回，是纯往返开销。

本册地址空间种类最多（六种），指针化的收益也最大，但风险同样最大：不能把
「换成 `GetUnderlyingPtr`」当成全部工作，必须把地址、类型、位置、缓存四件事拆开处理。

本设计只改**参数表示与分派**，不改矩阵算法、精度语义、布局约定或同步协议。

示意（任务书 §1；本册负责 LoadData / Mmad / Fixpipe，DataCopy 属 DMA 册）：

```cpp
__cbuf__ half l1aBuf[m * k];
__ca__ half l0aBuf[m * k];
__cc__ float l0cBuf[m * n];
AscendC::DataCopy(l1aBuf, aGm, m * k);          // 他册
AscendC::LoadData(l0aBuf, l1aBuf, loadParamsA); // 本册
AscendC::Mmad(l0cBuf, l0aBuf, l0bBuf, mmadParams);
AscendC::Fixpipe(cGm, l0cBuf, cbufWorkspace, fixpipeParams);
```

### 现有实现现状分析

分析基线：`cann/asc-devkit` `master`（相对仓库根路径）。下列结论来自源码核对。

| 层次 | 关键文件 | 现状 | 本设计动作 |
| --- | --- | --- | --- |
| 声明 | `kernel_operator_mm_intf.h`、`kernel_operator_fixpipe_intf.h`、同步/Dump 相关 intf | Tensor 形参、架构宏、默认实参 | 保留顺序与门控，扩展指针/混合入参 |
| 分派 | `kernel_operator_mm_load2d_impl.h`、`kernel_operator_*_intf_impl.h` 等 | Tensor 校验 → `GetPosition()` 选通路 → 转型调 `*Cal` | 指针路径编译期选通路；不改 `*Cal` |
| 叶子 | `impl/basic_api/dav_3510/` 等 `*Cal` | **已是裸指针** | **零改动** |
| 地址 | `kernel_tensor.h` / `kernel_tensor_impl.h` | `GetPhyAddr()` 返回类型随对象/模式变化 | 类型与地址解耦 |
| SPM | `kernel_tpipe.h` | 工作区生命周期 | 不伪造容量 |
| 汇总 | `kernel_cube_intf.h` | 仅 `#include` | 无声明可改；其 include 含 gemm/conv2d 头 |

**调用链（以 LoadData 2D 为例）。** `LoadDataImpl` 现有重载全部吃 Tensor，内部再调
`LoadData2DL12L0ACal` / `LoadData2DL12L0BCal` 等纯指针叶子。因此任务书 §2.3 里
「直接 `LoadDataImpl(dstPtr, srcPtr, params)`」在当前树中**不成立**；实施必须逐签名
追到可复用的 `*Cal`（或抽取分派层中的地址适配），不能假设中间层已有指针重载。

**`GetPhyAddr()` 三形态。** `LocalTensor` 在 CPU 调试返回 `PrimType*`，在设备上返回
`uint64_t`；`GlobalTensor` 返回带地址空间的 `__gm__` 指针。故 §2.3 的
`ElemType<decltype(dstPtr)>` 在设备路径会失效。现有代码用
`(__ca__ PrimT<T>*)dst.GetPhyAddr()` 一类显式转型，本设计沿用「类型来自操作数，
地址另行取得」的做法。

**仓库内已有编译期分派先例。** 同文件 bitmode 变体使用 `TPosition` 模板参数 +
`if constexpr`。指针路径把地址空间当作编译期位置，把运行期 `if (dstScope==…)`
塌缩掉，落到同一 `*Cal`，与该先例同构，而不是另起一套计算实现。

**GM 缓存。** `ExtractCacheMode` / `ExtractL2CacheGmAddr` 形参已是裸 `__gm__` 指针；
指针路径可直接调用，但必须在地址编码被清理之前取模式。

# 需求分析（required）

## 需求描述

在 CUBE 清单范围内，支持 Tensor、裸指针及同次调用混用。相同输入、布局、配置与
同步顺序下，指针路径与 Tensor 路径满足精度要求；原 Tensor 调用行为不变。
适配发生在编译期，不引入随规模增长的额外 Device 拷贝。

## 需求拆解

1. **范围**：以腾讯文档 CUBE 表（类型码 C）与任务书 61 签名为主；不改 VECTOR/DMA。
   卡片多出的 `Gemm`/`Conv2D` 记为待确认项。
2. **分操作数推导**：dst / src / bias / scale / workspace 各自推导，禁止单一模板类型
   锁死全部操作数。
3. **语义保持**：地址空间、元素类型、可写性、GM cache、低比特存储粒度均按原接口。
4. **兼容**：原显式模板、默认实参、`__inout_pipe__`、架构宏不被泛型入口抢占。
5. **可测**：代表性接口先出双路径对拍（任务书 §3.5）；61 条台账登记改造与 `*Cal` 入口。

### 范围清单与架构边界

| 分组 | 代表入口 | 处理原则 |
| --- | --- | --- |
| 加载 | `LoadData`、`LoadDataWithStride`、`LoadDataWithTranspose` | 按 2D/3D/MX/参数版本分别适配 |
| 计算 | `Mmad`、`MmadMx`、`MmadWithSparse` | 累加目标、左右矩阵、bias 独立萃取 |
| 写回 | `Fixpipe`、`SetFixPipeConfig`、`SetFixPipeAddr` | 按 GM/L1/UB 与量化工作区适配 |
| SPM | `InitSpmBuffer` / `WriteSpmBuffer` / `ReadSpmBuffer` | 保留 TPipe 状态与单位 |
| 同步 | `IBSet`/`IBWait`、`WaitPreBlock`/`NotifyNextBlock` 等 | 仅清单内签名；无输出项单独判据 |
| 调试 | `DumpTensor`、`DumpAccChkPoint` | 保留 desc/偏移/开关宏 |
| 待确认 | `Gemm`、`Conv2D` | 卡片 64 口径；单独 PR，可剔除 |

`MmadWithSparse` / `LoadDataWithSparse` 带 `__NPU_ARCH__ == 2201` 门控，附件样例亦标
A2/A3。目标是 950 **也不删除门控**；台账记「架构不适用」，不计入通过数，并向评审确认口径。

# 详细设计（required）

## 算子分析

### 数学公式

本任务不改数学。常规 Mmad：

$$
C_{i,j} = C^{\mathrm{init}}_{i,j} + \sum_{k=0}^{K-1} A_{i,k} B_{k,j}
$$

bias、累加、MX、稀疏及 Fixpipe 的转换/量化/布局，均沿用对应重载的既有实现。

```text
GM →(既有搬运)→ L1 → LoadData → L0A/L0B → Mmad → L0C → Fixpipe → GM/L1/UB
```

GM→L1 的 `DataCopy` 不在本任务修改范围。

### 支持数据类型

| 类别 | Tensor | 指针 | 约束 |
| --- | --- | --- | --- |
| GM | `GlobalTensor<T>` | `__gm__ T*` | 保持 cache 属性 |
| L1 | `LocalTensor<T>` | `__cbuf__ T*` | 对齐/容量按原重载 |
| L0A | `LocalTensor<T>` | `__ca__ T*` | 不得与 L0B 混淆 |
| L0B | `LocalTensor<T>` | `__cb__ T*` | 不得与 L0A 混淆 |
| L0C | `LocalTensor<T>` | `__cc__ T*` | dtype 满足计算组合 |
| UB | `LocalTensor<T>` | `__ubuf__ T*` | 仅原接口允许的通路 |

不新增 dtype 承诺；仅原接口 + 架构共同支持的类型可走。`PrimT<T>` 只表示元素类型。

### 支持形状

沿用各接口原有 M/K/N、分形、stride、repeat、对齐与工作区约束。不新增广播或自动
padding。测试逻辑尺寸建议 1/32/1024/2048（矩阵按 M/K/N 缩放）；大图走既有 tiling，
禁止假设整幅 2048 方阵直接落入 L0。

## 算子实现

### 实现方案

一句话：**指针路径 = 用编译期地址空间代替运行期 `GetPosition()`，落到同一个 `*Cal`。**

```text
同名 API
  → 原 Tensor 重载 或 「至少一操作数为硬件指针」的受约束入口
  → 每操作数独立取：地址 / 元素类型 / 位置 / cache
  → Tensor 侧保留原校验；指针侧做可静态检查
  → 编译期选通路
  → 未改动的 *Cal
```

相对「为每个签名再抄一份指针重载」，优先采用受约束模板 + `if constexpr`（与仓内
bitmode 先例一致）。若某组（尤其 Fixpipe）出现特化塌缩或抢占原重载，该组回退为
显式指针重载，能力不变。

四项信息必须拆开（§2.3 示例把四者塞进一个函数，是示例失效的根因）：

| 信息 | Tensor | 指针 | 注意 |
| --- | --- | --- | --- |
| 地址 | `GetPhyAddr()`+转型 | 直通 | 设备 Local 为 `uint64_t` |
| 元素类型 | `PrimT<T>` | pointee | 禁止 `decltype(addr)` |
| 位置 | `GetPosition()` | 地址空间 | 指针侧用 `if constexpr` |
| cache | `ExtractCacheMode(t)` | `ExtractCacheMode(p)` | 清理地址前取 |

消歧：新入口至少一操作数为硬件指针；全 Tensor 必须解析回原重载；用符号/返回类型
比对取证，不以「能编译」代替。拒绝 `void*` 与丢失地址空间的普通指针。

#### 3.2.1 host侧设计

本任务是设备侧头文件库扩展，无新增 aclnn / OpDef / Host tilingKey。
样例 Host 仍负责分配、造数、launch、同步与对比；分核与 buffer 规划与改造前一致，
以便隔离「入参表示」这一变量。

#### 3.2.2 kernel侧设计

**1. 地址萃取。** 新增或复用 `GetUnderlyingPtr`（当前全仓不存在；任务书允许与 VECTOR
共用）。函数**只取地址**。类型工具使用 `AscendC::Std` 与 ASC 认可的地址空间特化，
不假设 `std::is_pointer` 覆盖全部硬件指针。

| 输入 | 地址 | 另保留 |
| --- | --- | --- |
| `LocalTensor<T>` | `GetPhyAddr()` | `PrimT`、位置、容量 |
| `GlobalTensor<T>` | `GetPhyAddr()` | 元素类型、GM 属性、cache |
| 硬件指针 | 原指针 | 地址空间、const/可写 |

目标必须可写；描述符 `const Tensor&` ≠ pointee const，也不给 `const T*` 写权限。

**2. 模板兼容。** 每操作数独立模板参数；保留 `FixpipeConfig`、
`FixpipeParamsArch3510<…>`、bit-mode、`__inout_pipe__` 等。无 Tensor 操作数的纯控制
接口不为凑数加假指针重载。

**3. LoadData / Mmad。** 目的地与源按地址空间分派；混合调用中 Tensor 位置仍运行期校验。
参数结构体（2D/3D/stride/MX 等）原样下传。只抽地址适配，不改 `*Cal` 计算语义。
Mmad 的 C/A/B/bias 独立萃取；流水等待不因指针化减少。

**4. Fixpipe 与 cache。** 按目的地选 L0C→GM/L1/UB，并与 `config.isToUB`、格式一致。
量化工作区单独适配。GM 路径在 `GetPhyAddr` 清编码前取 `ExtractCacheMode`；已是裸
`__gm__` 且带编码的指针直接走仓库既有抽取函数。

**5. SPM / 同步 / Dump。** 不伪造 buffer 容量；同步不用裸写替代事件协议；Dump 的
低比特偏移按存储粒度处理，禁止无条件 `ptr+countOff`。CPU_DEBUG 下原 `CheckFuncXxx`
吃 Tensor 时：能静态查的补指针版检查，其余文档写明「调用者保证」。

**6. 资源。** 不分配适配用临时 Device buffer，不靠拷贝 Tensor 数据来适配指针。

### 变更文件与协作边界

| 目录 | 变更 |
| --- | --- |
| `include/basic_api/` | 清单内声明与约束 |
| `impl/basic_api/` | traits、地址适配、CUBE 入口/分派 |
| `tests/api/basic_api/` | 编译正反例、回归、指针一致性 |
| `examples/01_simd_cpp_api/03_basic_api/` | 代表性样例改指针/混合（是否合入遵评审） |

公共 `GetUnderlyingPtr` 只保留一个落点；优先复用他册已合入实现，避免同名冲突。
禁止顺带改 VECTOR/DMA API。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 系列（`dav-3510`，验收主目标） | √ |
| Atlas A2/A3（`dav-2201`，保留既有 Tensor 与宏门控） | √ |

CANN 9.0.0~9.1.0 尽量覆盖两端点；缺环境在报告中列「未验证」。
不以 Host `g++` 代替毕昇 ASC，不以 2201 结果代替 3510 验收。

## 算子约束限制

1. 指针的生命周期、容量、对齐、布局由调用者保证。
2. 非法地址空间 / dtype / 写 const 等应编译失败。
3. 不扩展原接口未承诺的别名、原地覆盖、自动 padding。
4. 不通过删宏门控制造「950 可编译」假象。
5. 一致性比较须配置相同（含 cache/量化/同步）；地址相同≠语义相同。
6. 不交付他册接口改造。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 指针/Tensor/独立参考按 dtype 与算子类别判定；纯搬运无转换逐位一致 | 任务书 §3.2 + [生态算子开源精度实验标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| Tensor 回归 | 原用例 100% 通过 | 任务书 §2.4 |
| 性能标准 | 无额外标杆时延；说明无明显回退，对比选填 | 任务书 §3.3 |
| 内存标准 | 无随规模增长的适配拷贝 | 任务书 §3.4 |

同步类接口无数据输出：以「行为一致 + 编译期能力回归」为证据，避免被判漏测。

### 测试方法与覆盖矩阵

§3.5 要求至少代表性接口对比 Tensor/指针；允许复用原样例工程。台账覆盖 61 条；
数值用例先打 LoadData / Mmad / Fixpipe 代表路径。

流水线：一次造数 → 基线 Tensor / 改造后 Tensor / 全指针 / 混合 → **各自**对参考，
再交叉比对，防止双错互证。

| 编号 | 覆盖 | 判定 |
| --- | --- | --- |
| C01 | 合法混合编译正例 | ASC 通过，符号证明未抢占 |
| C02 | 错地址空间/dtype/const/`void*` | 编译失败且指向约束 |
| C03 | `load_data_*` 附件样例 | 与基线一致 |
| C04 | 3D load 类 | 双路径一致 |
| C05 | `mmad` / gemv / batch | 参考精度通过 |
| C06 | MX / FP4·FP8 | 按量化标准 |
| C07 | `fixpipe_l0c2gm/l1/ub` | 含 GM cache |
| C08 | unitflag + 同步最小例 | 无死锁；超时=失败 |
| C09 | SPM 最小例 | 哨兵不被破坏 |
| C10 | Dump / 低比特偏移 | 开关宏语义正确 |
| C11 | 全量 Tensor 回归 | 与改造前一致 |
| C12 | 2201 门控项 | 架构不适用单列 |

附件样例≠61 条完备证明；缺项按台账补。报告字段含签名、commit、硬件/CANN/ASC、
dtype、shape、组合类型、种子、命令、阈值来源、截图。汇总分列通过/失败/架构不适用/
未覆盖。

## 兼容性分析

加法式扩展：原 Tensor 签名与 `*Cal` 不动，存量代码无需修改。

主要风险：泛型抢占、Fixpipe 特化塌缩、cache 丢失、低比特偏移、跨册 helper 冲突、
个别签名追不到可复用指针层。对策分别为符号比对、分组回退显式重载、清理前取 mode、
按存储粒度测偏移、评审帖声明唯一落点、台账登记并请示评审。**追到 `*Cal` 仍只吃
Tensor 时停止自行发明，先与评审对齐。**

合入本设计后，按任务书在 `asc-devkit` 提设计评审 Issue；实现与自测完成后再提代码 PR。
代码仓：`https://gitcode.com/gcw_DmEcQNNx/asc-devkit`（实现阶段就绪后邀请
**Ascend-CANN** 为开发者）。

# 需求背景（required）

## 需求来源

本设计对应「9月社区任务 - AscendC Basic_API优化实现（CUBE侧矩阵类接口扩展）」，
提交人 GitCode 账号为 [Ethan-ki](https://gitcode.com/Ethan-ki)。任务材料为
`9月社区任务-AscendC Basic_API优化实现(CUBE侧矩阵类接口扩展).zip` 中的
`basic_api_optimize_cube.md` 及 `test-cases/`。
压缩包内任务书标题仍写作「8月社区任务」，本文按压缩包所属月份命名，技术范围以
该任务书第 2.4 节的 CUBE 接口清单为准。

任务要求对 Ascend C Basic API 的 CUBE（矩阵 / ISASI）类接口扩展指针输入能力，
使裸硬件指针与 `LocalTensor` / `GlobalTensor` 可调用同名 API，并保持原有
Tensor 接口功能与兼容性。任务书规定共 **21 个 API 名称、61 个重载签名**，
适配 **Ascend 950 系列产品、CANN 9.0.0 ~ 9.1.0**。

设计文档先提交至 `cann/cann-ops-competitions` 评审；实现代码最终向
`cann/asc-devkit` 的 `master` 分支提交。本文为实现前设计，测试章节描述验证计划，
不代表代码已经实现或真机测试已经通过。

## 背景介绍

### CUBE Basic API 指针化扩展

矩阵计算通常经过 GM、L1、L0A / L0B、L0C，再由 Fixpipe 写回 GM、L1 或 UB。
现有 Tensor 对象同时携带地址、逻辑位置、元素类型及部分调试信息。开发者采用
硬件指针编程时，不能简单用裸指针替换所有 Tensor 参数，也不能仅把
`.GetPhyAddr()` 替换成一个通用函数便认为完成了适配。

本设计在模板封装层完成参数分类、地址萃取及通路选择，复用已有底层实现。
不新增矩阵计算算法，不改变计算精度、数据布局、量化规则或流水同步语义。

### 现有实现现状分析

源码分析基线为 `asc-devkit` 提交
`78630029a07b5d9bd7a27de593c402c47dcf8e37`。以下路径均相对该仓库根目录。

| 层次 | 关键文件 | 现有职责 | 本次设计关注点 |
| --- | --- | --- | --- |
| 矩阵 API 声明 | `include/basic_api/kernel_operator_mm_intf.h` | LoadData、Mmad 等模板声明及架构门控 | 保留模板顺序、默认参数、重载与架构条件 |
| 矩阵 API 入口 | `impl/basic_api/kernel_operator_mm_intf_impl.h` | 类型校验、Tensor 校验、转发 | 增加受约束的指针及混合输入适配 |
| 矩阵通路分发 | `impl/basic_api/kernel_operator_mm_load2d_impl.h` 等 | 根据 Tensor 位置选择通路，再调用 `*Cal` | 地址与位置分离，复用硬件计算实现 |
| Fixpipe | `include/basic_api/kernel_operator_fixpipe_intf.h`、`impl/basic_api/kernel_operator_fixpipe_intf_impl.h` | L0C 写回、量化工作区、格式与缓存配置 | 每个操作数独立萃取，保留 config 和缓存语义 |
| Tensor 地址 | `include/basic_api/kernel_tensor.h`、`impl/basic_api/kernel_tensor_impl.h` | 物理地址、元素类型、偏移与缓存处理 | LocalTensor 地址在设备模式下可能为整数；GlobalTensor 必须走 `GetPhyAddr()` |
| SPM | `include/basic_api/kernel_tpipe.h` 及对应实现 | 初始化、写入和读回 SPM 工作区 | 保留 TPipe 状态、容量和偏移单位 |
| 同步与调试 | `kernel_operator_*sync*_intf.h`、`kernel_operator_dump_tensor_intf.h` 及对应实现 | 工作区同步、数据导出与检查点 | 保留事件协议和 Tensor 元数据检查 |

当前 `LoadDataImpl` 部分重载仍接收 Tensor，并非全部是裸指针函数；例如
LoadData 2D 根据 `GetPosition()` 分发至 `LoadData2DL12L0ACal` 或
`LoadData2DL12L0BCal`。因此实施时需沿具体调用链找到可复用的指针层，不能假定
`LoadDataImpl(dstPtr, srcPtr, params)` 已存在。

# 需求分析（required）

## 需求描述

在任务书限定的 CUBE 接口内，支持 Tensor、裸硬件指针及两者混用。相同输入、
数据布局、配置和同步顺序下，指针路径与 Tensor 路径数值一致；原 Tensor 调用
继续使用既有功能与检查机制。编译期适配不引入与输入规模线性相关的额外设备拷贝。

## 需求拆解

1. **范围管理**：以任务书 [CUBE 接口表](https://docs.qq.com/sheet/DYWVocVFXamRFQXBD?tab=000001)
   的类型码 C 为唯一范围依据，不扩展 VECTOR / DMA 分册接口。
2. **独立参数推导**：目标、源、bias、MX scale、量化工作区分别推导类型，允许合法的
   Tensor / Pointer 组合，不能用同一个模板类型约束所有操作数。
3. **地址与属性保持**：正确处理地址空间、元素类型、读写属性、Tensor 位置、GM 缓存
   属性以及低比特类型的存储粒度。
4. **源码兼容**：保留现有显式模板实参、默认模板参数、配置引用和架构宏；避免新增
   通用模板抢占既有重载或产生歧义。
5. **验证覆盖**：逐签名登记编译、正向、反向及 Tensor 回归用例，补齐附件样例未覆盖的
   SPM、同步与 Dump 类验证。

### 范围清单与架构边界

| 接口组 | 源码及任务材料中核对的代表入口 | 设计处理 |
| --- | --- | --- |
| 矩阵加载 | `LoadData`、`LoadDataWithStride`、`LoadDataWithTranspose` | 2D、3D、MX 与相应参数版本分开适配 |
| 矩阵计算 | `Mmad`、`MmadMx`、`MmadWithSparse` | 累加目标、左右矩阵、可选 bias 与特殊模式分别校验 |
| 矩阵写回 | `Fixpipe` | 按 GM / L1 / UB 目的地、格式、量化工作区分发 |
| SPM | `TPipe::InitSpmBuffer`、`WriteSpmBuffer`、`ReadSpmBuffer` | 工作区生命周期与容量语义不变 |
| 同步 | `IBSet` / `IBWait`、`WaitPreBlock` / `NotifyNextBlock` 等候选入口 | 只适配官方 C 类清单列入的签名 |
| 调试 | `DumpTensor`、`DumpAccChkPoint` 等候选入口 | 保留导出格式、偏移和开关宏，只适配清单内签名 |

此表是设计分组，不替代 21 个名称、61 个签名的官方逐项清单。本次撰写时腾讯文档
页面可访问，但表格数据接口未返回内容，因此不推测其余条目或各组重载数量。
实施前须将官方表逐行固化为覆盖台账，记录「清单行号、完整签名、架构宏、数据类型、
存储通路、源码入口、测试编号」，并核对名称总数为 21、签名总数为 61。

当前基线中 `MmadWithSparse` / `LoadDataWithSparse` 有 `__NPU_ARCH__ == 2201`
门控，附件稀疏样例也标注为 A2 / A3；不能因为任务目标是 950 就移除门控。
若正式清单包含在 950 不可用的签名，应在台账和报告中记录限制，保留既有架构能力，
并向评审确认其验收口径，不把未运行项记为通过。

# 详细设计（required）

## 算子分析

### 数学及数据流语义

本任务扩展 API 参数表示，不修改数学语义。以常规矩阵乘为例：

$$
C_{i,j} = C^{init}_{i,j} + \sum_{k=0}^{K-1} A_{i,k}B_{k,j}.
$$

初始化、bias、累加、MX scale、稀疏编码与 Fixpipe 的转换、量化及布局转换，
均按对应现有重载的参数和底层实现执行。公式只说明普通 Mmad 数据流，不替代
特殊模式的接口定义。

```text
GM 输入 -> 既有搬运与布局准备 -> L1
L1 -> LoadData -> L0A / L0B
L0A / L0B -> Mmad -> L0C
L0C -> Fixpipe -> GM / L1 / UB
```

GM 到 L1 的 DataCopy 可继续使用已有 Tensor 写法；本任务不依赖 DMA 分册先完成
指针化，也不在 CUBE 变更中修改 DataCopy 对外接口。

### 支持数据类型和地址空间

| 操作数类别 | Tensor 表示 | 裸指针表示 | 约束 |
| --- | --- | --- | --- |
| GM 数据 | `GlobalTensor<T>` | `__gm__ T*` | 按入口角色确定读写；保持缓存属性 |
| L1 数据或工作区 | `LocalTensor<T>` | `__cbuf__ T*` | 位置、对齐、容量须满足对应重载 |
| 左矩阵 | `LocalTensor<T>` | `__ca__ T*` | 对应 L0A，不与 L0B 混淆 |
| 右矩阵 | `LocalTensor<T>` | `__cb__ T*` | 对应 L0B，不与 L0A 混淆 |
| 累加结果 | `LocalTensor<T>` | `__cc__ T*` | 对应 L0C，元素类型满足计算组合 |
| UB 数据 | `LocalTensor<T>` | `__ubuf__ T*` | 仅用于原接口允许的通路 |

不新增 dtype 支持承诺。FP16、BF16、FP32、整数及 FP4 / FP8 等类型仅在原接口、
参数版本和目标架构共同支持时接受。量化工作区的 `uint64_t` 等约束保持不变。
`PrimT<T>` 只应用于其原有元素类型语义，不能直接把整个 Tensor 类型传入。

### 支持形状

保留各接口对矩阵维度、分形布局、步长、重复次数、对齐和工作区容量的约束。
本任务不新增广播、任意非连续布局或自动 padding。逻辑尺寸 1、32、1024、2048
用于测试规模设计；实际硬件 tile 按对应 dtype 和通路对齐，大矩阵通过既有 tiling
分块，不能把整张 2048 方阵直接放入 L0。

## 算子实现

### 实现方案

```text
同名 API 调用
  -> 既有 Tensor 重载 / 受约束的泛型入口
  -> 独立识别每个操作数的类别、元素类型和地址空间
  -> 保留 Tensor 校验；检查指针可静态确定的约束
  -> GetUnderlyingPtr + 位置 / 缓存等独立属性
  -> 选择既有硬件通路
  -> 复用既有 *Impl / *Cal 计算与搬运实现
```

#### 3.2.1 Host 侧设计

该任务属于设备侧头文件库，不新增 ACLNN 接口、OpDef 或 Host tiling key。
示例 Host 继续负责内存分配、输入生成、kernel 启动、同步和输出比较。
原有样例的分核、tile 形状及缓冲区规划保持一致，以便单独比较参数表示变化。

#### 3.2.2 Kernel 侧设计

**一、统一地址萃取与类型信息。** 在 Basic API 内部增加或复用
`GetUnderlyingPtr`，使用仓库 `AscendC::Std` 类型工具及 ASC 编译器认可的地址空间
特化，不直接假定标准库 `std::is_pointer` 能完整识别所有硬件指针。
接口契约如下：

| 输入 | 地址萃取结果 | 单独保留的信息 |
| --- | --- | --- |
| `LocalTensor<T>` | `GetPhyAddr()` 返回的底层地址 | `PrimT<T>`、`GetPosition()`、已有容量信息 |
| `GlobalTensor<T>` | 通过 `GetPhyAddr()` 取得物理地址 | 元素类型、GM 属性、缓存模式 |
| 硬件指针 | 保留地址空间和 pointee 类型的指针 | 静态地址空间、读写属性 |

设备模式下 `LocalTensor::GetPhyAddr()` 可返回 `uint64_t`，因此不能统一用
`ElemType<decltype(ptr)>` 推导元素类型。元素类型来自原操作数 traits，位置来自
Tensor 元数据或指针地址空间；仅在验证通路后将 LocalTensor 地址转换成对应硬件指针。
普通 `void*` 或丢失地址空间的普通指针不能作为绕过校验的入口。

源指针的 pointee const 属性应保留，目标必须可写。现有 `const GlobalTensor<T>&`
是描述符 const，原写回接口仍可写其所指存储；为保持该行为，Tensor 写地址适配
沿用原接口允许的 `GetPhyAddr()` 用法。它不赋予调用方传入的 `const T*` 写权限。

**二、模板与重载兼容。** 每个操作数独立设置模板类型，例如概念上的
`DstOperand`、`SrcOperand`、`BiasOperand`、`WorkspaceOperand`。
适配条件同时限制合法操作数种类、元素类型关系、地址通路和参数结构体。
保留现有显式模板调用形式及默认配置；纯 Tensor 调用继续匹配兼容入口，含指针的
调用进入泛型适配，再共享内部实现。新增入口至少要求一个操作数为硬件指针，避免
与既有全 Tensor 重载竞争。具体模板参数排列按各签名实现，不能全仓机械替换。

`FixpipeConfig`、`IsResetLoad3dConfig` 等非类型模板实参保持原有引用和生命周期要求；
`FixpipeParamsArch3510<config.format>` 等依赖关系、bit-mode 重载及 `__inout_pipe__`
标注保持不变。没有 Tensor 操作数的控制接口无需为凑数量新增指针重载。

**三、LoadData 与 Mmad 通路。** LoadData 分别处理 GM / L1 输入及 L0A / L0B 等
目的地。纯指针的通路在编译期确定；混合调用中 Tensor 的实际位置仍按已有规则检查。
2D、3D、transpose、stride、MX 的参数结构体继续决定布局与指令配置。
必要时只提取原分发层中的地址适配部分，不改变 `*Cal` 的计算语义。

Mmad 的累加目标、左矩阵、右矩阵和可选 bias 独立萃取；bias 位置与类型按对应
重载校验，不假定它和 L0A / L0B 相同。初始化、累加、GEMV、unitFlag、MX 和稀疏
开关直接传递给已有实现。流水事件和等待点不因指针化而减少。

**四、Fixpipe 工作区与缓存属性。** 依据目标地址空间选择 L0C 到 GM、L1 或 UB
通路，并验证其与 `config.isToUB`、格式配置一致。源 L0C、目标和可选 L1 量化
工作区分别适配，保持工作区类型、长度及分配责任。

3510 等架构的 `GlobalTensor::GetPhyAddr()` 会清理地址中的缓存编码；现有代码
通过 `ExtractCacheMode(originalOperand)` 单独读取模式。新路径也必须在属性丢失前
取得缓存模式。对携带缓存编码的 GM 指针，使用仓库已有
`ExtractCacheMode` / `ExtractL2CacheGmAddr` 规则分离模式与物理地址；普通物理指针
使用原有默认模式。不得从已清理的物理地址反推原模式，或把编码位作为物理地址使用。
该处理同样适用于带缓存属性的 GM LoadData 通路。

**五、SPM、同步与调试。** SPM 保留 TPipe 内部状态和工作区生命周期，读取、写入的
大小与偏移按现有接口单位解释；不在裸指针中假造完整 buffer 容量。
同步工作区按原类型和初始化协议使用，保留可写性、事件 ID、生产者/消费者关系及
内存可见性，不用直接地址读写替代同步指令。

Dump 保留 `desc`、`dumpSize`、`ShapeInfo`、`countOff` 及开关宏语义。
Tensor 路径仍检查可用的 size 信息；裸指针的有效长度由调用者保证，检查可判定的
参数与对齐，并在文档中明确无法从地址推导总容量。检查点的低比特元素偏移按照实际
存储粒度处理，不能无条件执行 `ptr + countOff`。不在 CPU 模式伪造设备 Dump 支持。

**六、调试与资源管理。** Tensor 分支保留既有 CPU debug、位置、边界、对齐及
原子写回相关锁语义。指针分支不能调用 `.GetSize()` 等 Tensor 专属方法；将静态
可知的错误作为编译期诊断，将已有可复用的运行期检查保留在合法设备路径中。
不额外分配 device buffer，不复制 Tensor 数据来适配指针；原指令本身需要的搬运、
SPM 工作区与量化工作区不计为适配新增内存。

### 变更文件与协作边界

| 目录 | 计划修改 |
| --- | --- |
| `include/basic_api/` | 清单内公开声明、必要的模板约束与文档 |
| `impl/basic_api/` | 操作数 traits、地址适配、CUBE 入口和必要分发层 |
| `tests/api/basic_api/` | 清单覆盖、编译正反例、Tensor 回归及指针一致性用例 |
| `examples/01_simd_cpp_api/03_basic_api/` | 代表性矩阵样例的指针和混合调用，是否随 PR 合入遵从评审 |

`GetUnderlyingPtr` 可与 VECTOR 分册共用，但必须明确公共 helper 的唯一实现位置。
复用已有 helper 优先；若需新增共用文件，在评审中说明依赖，不能在不同分册重复
定义同名函数。共享文件中的修改只服务于清单内入口，禁止顺带改造 VECTOR / DMA API。

## 支持硬件

| 硬件 / 软件 | 本任务要求 | 验证方式 |
| --- | --- | --- |
| Ascend 950 系列 | 主要适配目标 | 在实际可用 950 设备记录型号、架构和驱动，并执行对应接口用例 |
| CANN 9.0.0 ~ 9.1.0 | 编译及运行环境 | 至少覆盖可用的两个版本端点；缺少环境时明确列为未验证 |
| 毕昇 ASC 编译器 | 使用仓库与 CANN 匹配的工具链 | 记录编译器版本和目标 SoC，不以普通 Host C++ 编译替代 |
| 其他已有架构 | 保持既有能力 | 保留原有宏门控；编译和 Tensor 回归按仓库 CI 条件执行 |

## 算子约束限制

1. 裸指针由调用者保证有效生命周期、容量、对齐及正确布局，指针化不提供自动分配。
2. 不支持的地址空间、dtype、写入 const 数据等可静态识别的组合应编译失败。
3. 不新增原接口不支持的别名、原地覆盖、零长度或尾块能力；按原签名逐项处理边界。
4. 目标架构不可用的接口不通过强制转换或删除宏门控获得表面可编译性。
5. 指针与 Tensor 一致性比较须使用相同缓存、量化、布局和同步配置；仅指针地址相同
   不足以证明接口语义相同。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 数值精度 | 指针、Tensor 与独立参考结果按对应 dtype 和算子类别判定，不自行统一误差阈值 | 任务书及 [生态算子开源精度实验标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| Tensor 回归 | 相同配置的已有功能与原有测试结果保持一致 | 任务书第 2、3 节 |
| 性能 | 无独立量化指标；检查编译期适配未增加无关指令、同步及数据拷贝，性能对比选填 | 任务书第 3.3 节 |
| 内存 | 无随输入规模增长的适配临时 buffer 或额外设备拷贝 | 任务书第 3.4 节 |

### 测试方法与覆盖矩阵

每个在目标架构可用的清单签名至少有编译与调用覆盖。对有 n 个 Tensor 操作数的
签名，枚举合法 Tensor / Pointer 组合，最多为 `2^n` 种；每个操作数都需覆盖单独
替换，避免仅测试全 Tensor 与全指针。原签名的显式模板形式另设编译回归。

| 测试编号 | 类别 / 附件样例 | 主要覆盖 | 判定 |
| --- | --- | --- | --- |
| C01 | 模板编译正例 | 各参数版本、默认 / 显式模板、全部合法混合形式 | ASC 编译成功，重载无歧义 |
| C02 | 模板编译反例 | 错误地址空间、类型组合、const 目标、错误工作区和配置 | 非法调用失败，错误定位到接口约束 |
| C03 | `load_data_2dv2_l12l0`、`load_data_with_stride` | L1 到 L0A / L0B、stride、transpose | 搬运结果或后续矩阵结果与基线一致 |
| C04 | `mmad_load3dv2` | 3D 加载、padding、参数版本与状态复位 | 同输入和配置的两条路径一致 |
| C05 | `mmad`、`mmad_gemv`、`batch_matmul` | 初始化 / 累加、bias、M=1、batch、多 tile | 各路径分别通过参考精度检查 |
| C06 | `load_data_2dmx_l12l0`、`mmad_mx` | FP4 / FP8 数据与 scale 类型、布局和混合输入 | 按对应量化参考结果检查 |
| C07 | `fixpipe_l0c2gm`、`fixpipe_l0c2l1`、`fixpipe_l0c2ub` | 三种目的地、量化工作区、格式与转换、GM 缓存模式 | 输出正确且工作区边界不被破坏 |
| C08 | `mmad_unitflag` 及同步最小用例 | 流水依赖、连续调用、跨核工作区 | 多轮结果正确，无死锁；超时记失败 |
| C09 | SPM 最小用例 | 初始化、合法偏移、大小 / 参数重载、写回后读出 | 数据一致，前后哨兵不变 |
| C10 | Dump 最小用例 | GM / Local、shape、检查点偏移、开关宏 | 导出数据及描述正确，关闭时无意外导出 |
| C11 | 全量已有 Tensor 用例 | 原入口、调试模式、位置校验、缓存配置 | 原有预期保持不变 |
| C12 | 架构限制登记 | `mmad_with_sparse`、`load_data_l12l0` 等有产品限制的样例 | 按实际架构选择，跳过须有原因且不计为通过 |

上述样例名来自任务压缩包，附件样例不是 61 个签名的完整覆盖证明；缺少的版本、
GM 加载、transpose 及辅助接口用例按签名台账补齐。

常规数据覆盖零、正负值、混合值和固定随机种子；建议范围 `[-100, 100]`，低比特、
量化及累加场景使用对应合法数值域并规避非预期溢出。覆盖小矩阵、矩形、M=1、
32 级别 tile，以及 1024 / 2048 级别的多 tile 逻辑矩阵；加入原接口允许的尾块、
步长和重复次数边界。NaN / Inf 仅在原接口定义了相应行为时纳入。

执行时固定原基线和修改后提交，复用附件 `scripts/gen_data.py` 与
`scripts/verify_result.py` 或等价官方脚本：先生成一次输入，再分别运行原 Tensor、
修改后 Tensor、指针及混合路径。每条路径独立与参考结果比较，再做路径间比较，
避免两条路径同时错误却互相一致。纯搬运且无转换的数据检查逐位一致；浮点矩阵计算、
量化和转换遵循对应精度标准，记录阈值来源而非只记录 PASS。

自测报告记录 API 完整签名、源码版本、硬件与 CANN / ASC 版本、dtype、逻辑 shape、
tile shape、布局、参数、组合类型、随机种子、命令、精度指标、结果和日志截图。
README 固化实际可执行的构建及运行命令。最终汇总分别给出通过、失败、架构不适用、
环境未覆盖数量，未执行测试不得计入通过数。

## 兼容性分析

本设计保持已有函数名、参数结构体、显式模板调用和 Tensor 语义。新增指针能力
限定在受约束的入口，底层计算、内存布局、同步与资源所有权不变。Tensor 的位置和
容量校验继续保留，裸指针只承担其可表达的地址与类型信息。

关键风险为地址空间 traits 的编译器行为、显式模板重载冲突、GM 缓存属性丢失、
低比特元素偏移和共享 helper 的多分册冲突。分别通过 ASC 编译正反例、旧调用编译
回归、缓存模式对照、低比特边界测试和公共 helper 单一定义检查验证。

评审时需确认官方 61 个签名的逐项台账，以及其中非 950 可用接口的验收边界。
设计评审通过后，按任务书要求在 `asc-devkit` 提交设计评审 Issue；实现与自测完成后
再提交代码 PR 和测试报告。个人代码仓为
[Ethan-ki/asc-devkit](https://gitcode.com/Ethan-ki/asc-devkit)。

# aclnnEqual 算子详细设计文档

## 1. 需求背景（required）

### 1.1 需求来源

昇腾社区 2026 年 8 月社区任务——Equal 算子开发（A2A3）。任务书要求参考昇腾版本内置 aclnnEqual 算子的 TBE 实现，在昇腾 NPU（Atlas A2 训练系列 / Atlas A3 系列）上基于 Ascend C 编程语言实现功能一致的算子，与原 TBE 实现的关键区别在于：比较方式从二进制位比较更改为与 CPU 一致的逻辑值比较。完成算子设计、开发、测试全流程工作，验收通过后提交至昇腾算子开源仓（https://gitcode.com/cann/ops-math ，experimental/math 目录）。

### 1.2 背景介绍

#### Equal 算子实现优化

基于 Equal 算子历史 TBE 版本，使用 Ascend C 编程语言进行优化实现。


#### Equal 算子 TBE 实现现状分析

通过对 Equal 算子 TBE 版本的功能分析，当前支持的能力如下：

**支持芯片平台：**

| 芯片平台 | 芯片型号 | 说明 |
|----------|----------|------|
| Atlas A2 训练系列 | ascend910b | A2 平台 |
| Atlas A3 系列 | ascend910_93 | A3 平台 |

**Equal 算子（TBE）实现路径和相关 API 路径：**

| 路径类型 | 实际路径 |
|----------|----------|
| TBE kernel 实现（legacy） | `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/equal.py` |
| TBE kernel 实现（Ascend C） | `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/ops_math/ascendc/equal/` |
| 算子原型头文件 | `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/` |
| 算子信息库（A2） | `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/` |
| 算子信息库（A3） | `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/` |

- 参考代码样例（开源仓）：https://gitcode.com/cann/ops-math/tree/master/math/tensor_equal

**Equal 算子参数说明：**

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| self | 输入 tensor | tensor | FLOAT16、FLOAT、INT8、INT16、INT32、INT64、BOOL、BFLOAT16 | 与 other 同 dtype | 任意维度，支持 broadcast |
| other | 输入 tensor | tensor | FLOAT16、FLOAT、INT8、INT16、INT32、INT64、BOOL、BFLOAT16 | 与 self 同 dtype | 任意维度，支持 broadcast |
| out | 输出 tensor | tensor | BOOL | 无 | broadcast(self, other) 后的形状 |

计算公式：

```
out_i = (self_i == other_i) ? True : False
```

原 TBE 实现的比较方式为二进制位比较：将两个输入的内存位模式直接判等。该方式对整数类型与逻辑值比较等价，但对浮点类型存在差异。

#### Equal 算子功能分析

Equal 算子功能：对输入张量 self 和 other 逐元素进行相等比较，输出 BOOL 类型结果；当 self 和 other 的 shape 不同但满足广播规则时，先广播再逐元素比较。

- 输入：self、other
- 输出：out（BOOL）
- 支持数据类型：FLOAT16、FLOAT、INT8、INT16、INT32、INT64、BOOL、BFLOAT16
- 支持广播：支持（Numpy 风格广播规则，含标量广播）

本任务与原 TBE 实现的关键差异——逻辑值比较语义（与 CPU torch.eq 完全一致）：

| 场景 | 二进制位比较（原 TBE） | 逻辑值比较（本实现） |
|------|------------------------|----------------------|
| +0.0 与 -0.0 | 不相等（位模式不同） | 相等 |
| NaN 与任意值（含 NaN 自身） | 视位模式而定 | 不相等 |
| +Inf 与 +Inf、-Inf 与 -Inf | 相等 | 相等 |
| 整数类型按数值比较 | 一致 | 一致 |

## 2. 需求分析（required）

### 2.1 需求描述

使用 Ascend C 编程语言、以 aclnn 算子工程化开发方式实现 Equal 算子，对外提供 aclnnEqualGetWorkspaceSize / aclnnEqual 两段式接口。算子支持 FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、BOOL 共 8 种数据类型，支持 ND 数据排布格式与任意合法广播（含标量广播），比较语义为与 CPU 一致的逻辑值比较，支持确定性计算。

### 2.2 需求拆解

1. 支持 8 种数据类型（self 与 other dtype 必须一致），输出固定为 BOOL；
2. 支持 ND 格式、最多 8 维、任意合法广播（含标量 broadcast）的算子泛化；
3. 逻辑值比较语义：+0.0 == -0.0、NaN 与任意值（含自身）不等、同号无穷大相等，整数按数值比较；
4. 确定性计算：相同输入多次执行结果一致；
5. 精度：输出与 CPU 逻辑值比较结果 100% 一致（matched_ratio = 1.0，rtol/atol = 0）；
6. 性能：所有核参与计算场景不低于原 TBE 算子的 95%。

## 3. 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

对于输入张量 self 和 other，逐元素比较：

```
out_i = (self_i == other_i) ? True : False
```

其中 `==` 为逻辑值相等比较（非二进制位比较），与 IEEE 754 数值比较语义一致：NaN 与任何值比较均为不等，+0.0 与 -0.0 数值相等，+Inf == +Inf、-Inf == -Inf。

#### 3.1.2 支持数据类型

- 输入 self / other：FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、BOOL（两者 dtype 必须一致，不支持不同 dtype 输入）
- 输出 out：BOOL

#### 3.1.3 支持形状

- 数据排布格式：ND
- 维度：任意维度，最多 8 维
- 广播：支持 Numpy 风格广播规则（shape 右对齐，逐维要求相等或至少一维为 1，维度缺失按补 1 处理），out 的 shape 必须与广播后的结果 shape 一致
- 支持空张量（任一维为 0）场景，输出元素数为 0 时直接返回

### 3.2 算子实现

#### 3.2.1 实现方案

采用 Ascend C 自定义算子工程（FrameworkLaunch，aclnn 泛化调用方式），工程结构：

```
Equal/
├── CMakeLists.txt         # 顶层 CMake 配置
├── build.sh               # 构建脚本
├── op_host/
│   ├── CMakeLists.txt     # host 侧 CMake 配置
│   ├── equal.cpp          # 算子原型注册 + InferShape + TilingFunc
│   └── equal_tiling.h     # TilingData 结构定义
└── op_kernel/
    ├── CMakeLists.txt     # kernel 侧 CMake 配置
    └── equal.cpp          # Ascend C kernel 实现
```

编译构建生成自定义算子包并部署到 opp/vendors/<vendor> 目录，构建系统自动生成 aclnnEqualGetWorkspaceSize / aclnnEqual 两段式 aclnn 接口（算子泛化），接口签名与任务书 2.3 节一致。aclnnEqual 第一段接口完成入参校验：self/other 为空指针、dtype 不在支持范围、self 与 other dtype 不一致、shape 不满足广播规则、out 非 BOOL 或 shape 与广播结果不一致时，返回参数校验错误。

#### 3.2.2 host 侧设计

**1. 算子原型（op_host/equal.cpp）**

通过 OP_DEF("Equal") 注册：

- 输入 x1（self）、x2（other）：dtype 列表 {FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, BOOL}，format 均为 ND，支持任意 shape（含动态 shape）
- 输出 y（out）：dtype {BOOL}，format ND
- 绑定 InferShape 与 AICore TilingFunc
- 平台配置：ascend910b（Atlas A2）、ascend910_93（Atlas A3）

**2. InferShape（广播规则校验 + 输出形状推导）**

- 将 x1、x2 的 shape 右对齐，维度缺失一侧补 1
- 逐维检查：两维相等，或至少一维为 1，否则返回 GRAPH_FAILED（触发参数校验报错）
- 输出 shape 逐维取 max(dim1, dim2)

**3. TilingFunc（tiling 策略）**

host 侧通过 PlatformAscendC 获取 UB 大小与 AI Core（AIV）数量，按如下策略切分。

tilingKey 规划策略（需要感知 host 侧信息让 kernel 走不同分支）：

| tilingKey | 场景 | 说明 |
|-----------|------|------|
| 0 | x1、x2 shape 完全相同 | 不涉及维度信息，展平为一维向量按元素数切分（快路径，覆盖全部性能 case） |
| 1 | 一方为标量（numel == 1） | 标量广播快路径，同样按一维平铺，标量侧由 kernel 展开 |
| 2 | 一般广播 | 行式广播调度（内连续段 + 外层行迭代） |

分核策略：优先使用满核的原则。

- 平铺路径（tilingKey 0/1）：以输出元素总数 totalLength 为基准，`coreNum = min(aiCoreNum, ceil(totalLength / 32))`（按 32 元素对齐保证各 dtype 32B 对齐）；核间不能均分时，前 coreNum-1 个核处理对齐后的 blockLengthMean，最后一个核处理剩余全部数据 blockLengthEnd
- 一般广播路径（tilingKey=2）：以行数 outerLen 为基准分核，`coreNum = min(aiCoreNum, outerLen)`，前 coreNum-1 核均分，尾核处理余下行

数据分块和内存优化策略：充分使用 UB 空间的原则。

- 通过 GetCoreMemSize 获取 UB 大小，结合双缓冲（BUFFER_NUM = 2）与 kernel 侧缓冲区布局（输入队列 2 × BUFFER_NUM × elemSize 字节/元素、输出队列 BUFFER_NUM × 1、比较域转换缓冲、位掩码缓冲等）核算单元素 UB 预算 `bytesPerElem = 4 × elemSize + 16`，由此计算单 tile 元素数 `tileLength = ubSize / (BUFFER_NUM × bytesPerElem)` 并向下 32 元素对齐
- 尾块处理：最后一次搬运不满一个 tile 时按剩余元素数搬运，kernel 侧对不足 32B 对齐的尾数逐元素搬运，避免数据碎片与越界

一般广播（tilingKey=2）补充参数：

- 将广播后输出形状与两输入右对齐补齐到 8 维后的形状、广播步长（广播维步长为 0）通过 TilingData 下发（dims[8]、stridesX[8]、stridesY[8]）
- 计算内连续长度 innerLen：自最内维向外累乘，要求该组维度上两输入均为"稠密"（sx[d] == s[d] 且 sy[d] == s[d]），遇到不满足的维度即停止
- 外层按行迭代（共 outerLen = totalLength / innerLen 行），每行通过 8 维坐标与广播步长计算两输入的行首偏移，行内再按 tileLength 分段搬运连续数据

TilingData 字段：tilingKey、totalLength、tileLength、tileLengthEnd、tileNumMean、tileNumEnd、blockLengthMean、blockLengthEnd、scalarSrcIdx（标量来自第几个输入）、outerLen、innerLen、dims[8]、stridesX[8]、stridesY[8]。

#### 3.2.3 kernel 侧设计

kernel 采用 Init + Process 两阶段组织，Process 内部为标准三阶段流水 CopyIn → Compute → CopyOut，输入/输出队列开启双缓冲（BUFFER_NUM = 2），不同 tilingKey 走不同分支。

**1. 核心计算指令组合**

```c
Compare(mask, x, y, CMPMODE::EQ, count)                 // 逻辑值相等比较, 输出 1bit/元素位掩码
Select(sel, mask, ones, zeros, VSEL_TENSOR_TENSOR_MODE) // mask 为 1 选 1.0, 否则选 0.0
Cast(out_u8, sel, CAST_RINT, count)                     // half 0/1 转 uint8, 得 BOOL 结果
```

Compare 指令的 CMPMODE::EQ 为硬件向量比较的逻辑值相等语义：+0.0 == -0.0 为真、NaN 与任何值（含自身）为假、同号无穷大相等，与 CPU 比较结果完全一致。Select（VSEL_TENSOR_TENSOR_MODE）将 1 bit/元素的掩码展开为每元素 1 字节的 BOOL 输出。

**2. 分 dtype 实现策略**

Ascend 910B 上 Compare 指令源操作数仅支持 half / float / int32，因此按 dtype 选择无损的"比较域"（保证转换精确、不改变相等性判定）：

| dtype | 实现方式 |
|-------|----------|
| FLOAT16 / FLOAT / INT32 | 直接 Compare(EQ)，硬件语义即逻辑值比较 |
| BFLOAT16 / INT16 | Cast 到 float（精确无损）后 Compare(EQ)；float 比较天然满足 +0.0==-0.0、NaN 不等语义 |
| INT8 / BOOL | Cast 到 half（精确无损）后 Compare(EQ)；BOOL 按存储值 0/1 比较 |
| INT64 | 每个 64bit 元素 ReinterpretCast 为 {低32位, 高32位} 两个 int32，一次 Compare(EQ, 2×count) 得到位掩码，再将每元素相邻两个 bit 相与得到结果（整数位比较与逻辑值比较等价） |

**3. 广播调度（按 tilingKey 分支）**

- tilingKey=0（同 shape 平铺）：输入按 核偏移 + tile 偏移 直接 DataCopy，批量搬运，尾块按剩余元素数处理
- tilingKey=1（标量广播）：每核先取一次标量元素，通过 Duplicate（不支持的 dtype 退化为逐元素 SetValue）展开填充至 UB 缓冲，再与另一输入逐 tile 比较
- tilingKey=2（一般广播）：按行循环，每行用 8 维坐标与广播步长计算两输入行首偏移，DataCopy 长度为 innerLen 的连续段（行内按 tileLength 再分段）后比较

**4. 确定性计算**

算子为纯逐元素计算，无归约、无原子操作、无跨核数据依赖，各核处理区间在 tiling 阶段静态划分，相同输入多次执行结果完全一致，满足确定性计算要求。

### 3.3 算子流程

```
GM(self/other) --CopyIn(含广播展开)--> UB --Compare(EQ)--> 位掩码
             --Select(1/0)--> half 0/1 --Cast--> BOOL(uint8) --CopyOut--> GM(out)
```

### 3.4 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|----------------|----------|
| Atlas A2 训练系列产品（ascend910b） | √ |
| Atlas A3 系列产品（ascend910_93） | √ |

### 3.5 算子约束限制

1. self 与 other 的数据类型必须一致，不支持不同 dtype 的输入
2. self 和 other 的 shape 需满足广播规则，否则触发参数校验报错
3. out 的数据类型必须为 BOOL，shape 必须与 self 和 other 广播后的结果 shape 一致
4. 仅支持 ND 数据排布格式，最多 8 维

## 4. 可维可测分析

### 4.1 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
|----------|------------------------|----------|
| 精度标准 | 输出与 CPU 逻辑值比较结果完全一致：matched_ratio = 1.0，rtol = atol = 0，覆盖 +0.0/-0.0、NaN、±Inf 边界场景；满足 AscendOpTest 工具默认阈值 | 任务书 3.2 节 / 生态算子开源精度标准（opbase experimental_standard） |
| 性能标准 | 所有核参与计算场景性能不低于原 TBE 算子的 95%；小 shape（10us 以下相差 3us）场景可提供性能仿真图和分析结论替代 | 任务书 3.3 节 |

### 4.2 测试方案

- 使用任务配套自测用例（覆盖全部 dtype、1~8 维、同 shape / 广播 / 标量广播、±0.0、NaN、±Inf 边界值），golden 为 CPU 逻辑值比较（torch.eq）
- 补充泛化用例：全 True、全 False 及混合结果场景，大 shape（所有核参与）与小 shape 性能对比场景（FLOAT16/FLOAT/INT32 @ [1024,4096]、FLOAT16 @ [4096,4096]、FLOAT16 @ [256,256]）
- 通过 AscendOpTest 工具执行精度比对，并与原内置 TBE 算子在相同硬件上进行性能对比（msprof 采集）

### 4.3 兼容性分析

以自定义算子包（opp/vendors 目录）方式部署，通过 aclnn 泛化接口调用，不影响其他内置算子；算子工程独立交付，不涉及存量兼容性问题。

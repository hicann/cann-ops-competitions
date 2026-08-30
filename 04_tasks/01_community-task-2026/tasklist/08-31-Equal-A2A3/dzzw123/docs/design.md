# aclnnEqual 算子设计文档

## 需求背景（required）

### 需求来源

- 社区任务：8 月社区任务-Equal 算子开发（A2/A3）
- 任务书链接：https://www.hiascend.com/activities/task-center/details/e65dd3fcabf1444299784f2232d67456?menu=tasks
- 验收要求链接：https://www.hiascend.com/developer/activities/details/11014a50a8794171a4a08688fd398774#tab2
- 开源仓地址：https://gitcode.com/cann/ops-math （目标合入目录 `experimental/math`）
- 软硬件环境：Atlas A2 训练系列 / Atlas A3 系列产品，CANN 9.1.0，芯片 910B3

### 背景介绍

#### aclnnEqual 算子实现优化

参考昇腾版本内置 `aclnnEqual` 算子的 TBE 实现，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的算子。与原 TBE 实现的关键区别在于：**比较方式从二进制位比较更改为与 CPU 一致的逻辑值比较**。

内置算子的 TBE 实现路径与 API 路径（CANN 9.1.0 容器内）：

- 算子实现（Ascend C 工程化）：`/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_math/ascendc/equal/`
  - host 侧入口：`equal_apt.cpp`
  - kernel 计算图：`arch35/equal_dag.h`
  - tiling 数据结构：`arch35/equal_struct.h`
  - 算子注册/编译：`impl/ops_math/dynamic/equal_apt.py`
- 算子原型：`/usr/local/Ascend/cann-9.1.0/opp/built-in/op_proto/inc/elewise_calculation_ops.h`（`REG_OP(Equal)`）
- 算子信息库：`/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_legacy/equal.json`

#### aclnnEqual 算子 TBE 实现现状分析

通过对内置 Equal 算子 TBE 版本的功能与实现分析，当前支持与实现如下：

| 参数 | 参数含义 | 类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| x1 | 输入 tensor | tensor | float16、float32、bfloat16、int8、uint8、int32、int64、bool | dtype 需与 x2 一致 | 任意维度，支持广播 |
| x2 | 输入 tensor | tensor | 同 x1 | dtype 需与 x1 一致 | 任意维度，支持与 x1 广播 |
| y | 输出 tensor | tensor | bool | 无 | 与 x1、x2 广播后的 shape 一致 |

当前实现要点（910B3 上）：

- 通过 `BroadcastSch`（atvoss 框架）完成广播调度、tiling 与数据搬移。
- 计算核心在 `equal_dag.h` 中，使用 `Vec::Compare<uint8_t, T, CMP_MODE>`（底层映射为 AscendC `Compare` / `vcmp` 指令）进行逐元素相等比较，再用 `Vec::Select` 将比较 mask 映射为 `1/0`（按 `uint8_t` 存储布尔结果）。
- 输入为 `bool` 时走 `int8_t` 特化分支。

关键问题：上述 `vcmp`/`Compare` 的“相等”比较在浮点类型上是**二进制位比较**，与 CPU 的逻辑值比较语义不一致：

| 场景 | 二进制位比较（内置现状） | 逻辑值比较（CPU/期望） |
|---|---|---|
| `+0.0` 与 `-0.0` | 不相等（符号位不同） | 相等 |
| `NaN` 与 `NaN` | 相等（位模式相同） | 不相等 |
| `NaN` 与任意值 | 不相等 | 不相等 |
| `+Inf` 与 `+Inf` / `-Inf` 与 `-Inf` | 相等 | 相等 |
| 整型 / bool | 直接比较（等价） | 直接比较 |

#### aclnnEqual 算子功能分析

- 功能：逐元素判断 `x1` 与 `x2` 是否相等，输出 bool 张量，支持广播。
- 输入：`x1`、`x2`（dtype 一致）。
- 输出：`y`（bool）。
- 支持广播：是（含标量广播、任意维度广播）。
- 比较语义：逻辑值比较（与 CPU 一致）。

## 需求分析（required）

### 需求描述

使用 Ascend C 编程语言实现 aclnnEqual 算子，功能与原 TBE 算子对齐，但比较方式改为与 CPU 一致的**逻辑值比较**。要求支持原算子对应的全部数据类型、数据格式与广播语义，满足各类合法输入场景，并通过泛化数据的功能、精度、性能验收。

### 需求拆解

1. 复用原 TBE 的 `aclnn` 工程化开发方式，实现 `aclnnEqualGetWorkspaceSize` 与 `aclnnEqual` 两个对外接口。
2. 支持数据类型：`float16、float32、bfloat16、int8、uint8、int16、int32、int64、bool`（与内置算子/自测用例对齐）。
3. 支持 ND 数据格式与广播。
4. 核心改动：浮点比较从二进制位比较改为逻辑值比较（正确处理 `±0.0`、`NaN`、`±Inf`）。
5. 输出为 BOOL（每元素 1 字节，非位压缩）。
6. 精度要求：输出与 CPU 参考实现完全一致（matched_ratio = 1.0）。
7. 性能要求：所有核参与场景下性能不低于原 TBE 算子的 95%。
8. 支持确定性计算（逐元素无累加，天然确定）。

## 详细设计（required）

### 算子分析

#### 数学公式

```
out_i = (x1_i == x2_i) ? True : False
```

其中 `==` 为逻辑值相等比较，具体规则：

- `+0.0` 与 `-0.0` 视为相等；
- `NaN` 与任意值（含 `NaN` 本身）视为不相等；
- `+Inf` 与 `+Inf`、`-Inf` 与 `-Inf` 视为相等；
- 整型按数值大小直接比较。

#### 支持数据类型

- 浮点：`float16`、`float32`、`bfloat16`（走逻辑比较分支）
- 整型：`int8`、`uint8`、`int16`、`int32`、`int64`（走直接比较分支）
- 布尔：`bool`（走直接比较分支）
- `x1` 与 `x2` dtype 必须一致；输出 `y` 为 `bool`。

#### 支持形状

- 支持 ND 任意维度，支持广播（broadcast），含标量（`shape=[]` 或 `shape=[1]`）广播。
- 输出 shape 为 `x1`、`x2` 广播后的公共 shape。

### 算子实现

#### 实现方案

沿用原 TBE 的 `aclnn` 算子工程化结构（`kernel_operator.h` + `BroadcastSch` + tiling），仅重写计算核心，将“二进制相等比较”替换为“逻辑相等比较”。

```
aclnnEqual (host) → tiling（广播 shape 分析、schema 计算） → kernel（CopyIn → Compute → CopyOut）
```

##### 3.2.1 host 侧设计

host 侧复用原 TBE 的 `equal_apt.cpp` / `equal_apt.py` 框架，逻辑如下：

1. **入参与 dtype 处理**：
   - 从 `OpInfo` 获取 `x1`、`x2`、`y` 的 dtype/format。
   - 通过 `DTYPE_MAP` 将 `x1/x2` 的 dtype 映射到 kernel 模板类型宏（如 `-DDTYPE_X1=float`），`y` 固定为 `bool`（kernel 内以 `uint8_t` 表示）。
   - `x1` 为 `bool` 时，kernel 侧特化为 `int8_t` 处理（与原实现一致）。
2. **广播与 tiling 策略**：
   - 采用 `BroadcastSch`（atvoss 框架）统一处理广播 tiling，host 侧只需在 tiling 结构（`equal_struct.h`）中声明 `BRC_TEMP_SCH_MODE_KEY(schMode)`，由框架根据 `x1/x2/y` 的 shape 关系自动推导广播调度模式（`schMode`）。
   - kernel 按 `schMode` 实例化不同的 `BroadcastSch<schMode, OpDag>` 模板，覆盖等 shape、单向广播（含标量）、多维度不连续广播等场景。
3. **分核策略**：
   - `BroadcastSch` 依据广播后元素总量与 AI Core 数量自动切分，遵循“满核优先、大核小核数据块均衡”原则，将数据均匀分配到各核。
4. **数据分块与内存优化**：
   - 单核内依据 UB 容量动态计算搬入块大小（`CopyInBrc`），单个搬运块大小满足 32B 对齐要求；`MemCfg` 使用 `LEVEL_2`（UB）。
   - 计算流程为纯逐元素比较，无中间 reducer 数据，UB 占用低，可开启多级流水。
5. **Workspace**：
   - 计算无需额外 workspace，`GetWorkspaceSize` 返回 0（与原实现一致）。

##### 3.2.2 kernel 侧设计

kernel 采用 `Init/Process` 两阶段，`Process` 内为 `CopyIn → Compute → CopyOut` 三段；计算核心由 `equal_dag.h` 中的 `EqualCompute<T>` 计算图描述，整体流程：

1. **CopyIn**：通过 `Vec::CopyInBrc<T>` 将 `x1`、`x2` 按广播规则搬入 UB（含广播填充）。
2. **Compute**：核心为比较运算，逻辑见下文“逻辑比较核心算法”，输出 `uint8_t` 结果（`1` 表示 True，`0` 表示 False）。
3. **CopyOut**：通过 `Vec::CopyOut<uint8_t>` 将结果写回 `y`。

**逻辑比较核心算法**（本算子关键改动点）：

将原 `equal_dag.h` 中的单一 `Vec::Compare<uint8_t, T, CMP_MODE>` 替换为按 dtype 分派的两类比较：

- **整型 / bool 分支**（`int8/uint8/int16/int32/int64/bool`）：无 `±0.0`/`NaN` 边界情况，直接：

  ```
  equal_mask = CmpEq(x1, x2)          // 位比较与逻辑比较等价
  ```

- **浮点分支**（`float16/float32/bfloat16`）：需要把二进制相等比较扩展为逻辑相等，公式为：

  ```
  eq_raw    = CmpEq(x1, x2)                        // 位相等
  zero_x1   = CmpEq(Abs(x1), 0)                    // |x1|==0  ⟺ x1 为 ±0
  zero_x2   = CmpEq(Abs(x2), 0)                    // |x2|==0  ⟺ x2 为 ±0
  both_zero = zero_x1 AND zero_x2                  // ±0 配对：+0.0 与 -0.0
  nan_x1    = CmpUnordered(x1, x1)                 // NaN 检测（x1 与自身无序 ⟺ NaN）
  nan_x2    = CmpUnordered(x2, x2)
  not_nan   = NOT (nan_x1 OR nan_x2)               // 任一为 NaN 即排除
  result    = (eq_raw OR both_zero) AND not_nan
  ```

  说明：
  - `Abs(±0.0)` 均得到 `+0.0`，因此 `CmpEq(Abs(x), 0)` 能同时命中 `+0.0`/`-0.0`，解决符号位差异；
  - `NaN` 因为 `NaN != NaN` 且与任何值无序，用无序比较（或等价写法 `CmpEq(x, x) == false` 的反向）可靠的把 `NaN` 置为“不相等”；
  - `±Inf` 无需特殊处理，`CmpEq` 位比较即可得到正确结果。

  最终通过 `Vec::Select<uint8_t, uint8_t, SELECT_MODE>` 把 `result`（bool mask）映射为 `1/0` 输出。

> 具体用到 `CmpEq`、`Abs`、无序比较（NaN 检测）等 Ascend C vector 原语，后续开发阶段依据 910B3 上原语支持情况选用等价的 `Compare`/`CompareScalar`、`Abs`、`Cmp(UNO/ORD)` 或 `Sub+Compare` 组合实现，保证语义正确且性能可控。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas 800I/T A2 | √ |
| Atlas 800I A3 | √ |

### 算子约束限制

- `x1` 与 `x2` 的 dtype 必须一致。
- `x1` 与 `x2` 的 shape 需满足广播规则，否则触发参数校验报错。
- 输出 `y` 的 dtype 必须为 `bool`；shape 必须与广播后的公共 shape 一致。
- 仅支持 ND 数据格式。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 输出与 CPU 逻辑值比较结果完全一致（含 `+0.0/-0.0`、`NaN`、`±Inf`）；所有 dtype 的 `rtol=atol=0`、`matched_ratio=1.0`、`max_abs_error_limit=0` | 任务书 + 生态算子精度标准 |
| 性能标准 | 所有核参与场景下不低于原 TBE 算子的 95%；小 shape（<10us）可附性能仿真分析说明 | 任务书 |
| 内存标准 | 不涉及（无需 workspace） | 任务书 |

性能自测参考 case（与 TBE 基线同硬件对比）：

| 数据类型 | Shape | 说明 |
|---|---|---|
| FLOAT16 | [1024, 4096] | 大 shape，所有核参与 |
| FLOAT16 | [4096, 4096] | 大 shape，所有核参与 |
| FLOAT | [1024, 4096] | 大 shape，所有核参与 |
| INT32 | [1024, 4096] | 大 shape，所有核参与 |
| FLOAT16 | [256, 256] | 小 shape（可适用仿真分析替代） |

### 兼容性分析

新实现算子，作为 `ops-math` 开源仓 `experimental/math` 下的 Ascend C 算子，接口与内置 `aclnnEqual` 保持一致（`aclnnEqualGetWorkspaceSize` / `aclnnEqual`），对存量调用方透明，不涉及历史版本兼容。

> 附：本次为设计文档阶段。自测用例、自测报告、待验收代码地址三类交付件将在开发/测试完成后补充。
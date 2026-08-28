# aclnnEqual（TensorEqual）算子设计文档

> 对应昇腾社区任务：**8月社区任务-aclnnEqual 算子开发（A2/A3）**
> 目标算子：CANN 内置接口 `aclnnEqual`，底层 op 类型 `TensorEqual`（对应 PyTorch `torch.equal`），语义为「判断两个 Tensor 的大小与元素是否完全一致，输出单个 BOOL」。

---

## 一、需求背景（required）

### 1.1 需求来源

| 项目 | 内容 |
|---|---|
| 任务书 | https://www.hiascend.com/activities/task-center/details/e65dd3fcabf1444299784f2232d67456 |
| 开源仓 | https://gitcode.com/cann/ops-math （目标目录 `math/tensor_equal`） |
| 设计文档模板 | https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md |
| 适配硬件 | Atlas A2 训练系列产品 / Atlas A3 系列产品（Ascend910B） |
| 开发语言 | Ascend C（aclnn 算子工程化开发模式） |

### 1.2 背景介绍

参考昇腾版本内置 `aclnnEqual` 算子的 TBE 实现，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的算子。与原 TBE 实现的关键区别在于：**比较方式从「二进制位比较」更改为与 CPU（PyTorch）一致的「逻辑值比较」**，完成算子设计、开发、测试全流程。

TBE 参考实现路径：

| 类别 | 路径 |
|---|---|
| kernel 实现（TBE） | `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/tensor_equal.py` |
| kernel 实现（Ascend C 参考） | `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/ops_math/ascendc/tensor_equal/` |
| 算子原型 | `/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/elewise_calculation_ops.h` 中 `REG_OP(TensorEqual)` |
| aclnn 接口 | `/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include/aclnnop/aclnn_equal.h` |

### 1.3 TensorEqual 算子 TBE 实现现状分析

TBE 实现 `tensor_equal.py` 中，核心计算入口为 `tensor_equal_compute_use_sub`，其现状如下：

1. **shape/dtype 前置判断**：若 `shape_x != shape_y` 或 `dtype_x != dtype_y`，直接返回 `False`（输出 0），不做逐元素比较。
2. **逐元素比较归约**：shape/dtype 一致时，对两个张量逐元素比较，再将结果做 `reduce_min`（全归约）得到单个 BOOL。
3. **逐元素比较的实现方式**：调用 `tbe.vcmp(input_x, input_y, 'eq', mode='bool')`（向量比较指令 `vcmp` 的相等模式），或在硬件不支持 `reduce_all` 时对 fp16/fp32 退化为 `vsub → vabs → vmins → vmuls → vadds → vabs` 的近似等价写法。
4. **数据类型 cast 处理**：TBE 将 `int32 → float32`、`int8/uint8/bool → float16` 后再比较。这种 cast 会引入精度损失（`float32` 尾数仅 23 位，`|int32| > 2^24` 的整数低位被舍入，可能把不相等的两个大整数误判为相等）；同时 TBE 的 dtype 校验清单 `check_tuple` 不含 `int16/int64`，而任务书要求支持，需在 Ascend C 实现中补齐并按整数直接比较。

两种实现本质都是对元素**位级（bit-pattern）比较**，其行为与 CPU 的 IEEE-754 **逻辑值比较**存在偏差，具体见下表：

| 输入场景 | 二进制位比较（当前 TBE / Ascend C） | CPU 逻辑值比较（目标） |
|---|---|---|
| `+0.0` 与 `-0.0` | 位型不同（`0x00000000` vs `0x80000000`）→ 判为「不等」 | 逻辑值相同 → 判为「相等」 |
| `NaN` 与 `NaN`（同位型） | 位型相同 → 判为「相等」 | NaN 与任何值（含自身）→ 「不等」 |
| `+Inf` 与 `+Inf` / `-Inf` 与 `-Inf` | 位型相同 → 「相等」 | 「相等」 |
| 整数 | 数值即位型 → 正确 | 按数值大小比较 |

TBE 版本当前支持能力：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| input_x（self） | 第一个输入 tensor | tensor | float16, float32, bfloat16, int32, int8, uint8, bool | 与 input_y 同 dtype | 任意维度 ND，与 input_y 同 shape |
| input_y（other） | 第二个输入 tensor | tensor | float16, float32, bfloat16, int32, int8, uint8, bool | 与 input_x 同 dtype | 任意维度 ND，与 input_x 同 shape |
| output_z（out） | 输出 tensor | tensor | bool | 单元素（一维 1 元素） | shape = [1] |

### 1.4 TensorEqual 算子功能分析

- **算子功能**：判断两个 Tensor（self、other）是否拥有相同的大小、dtype 与全部元素的对应逻辑值，输出单个 BOOL。
- **输入**：self、other（两个输入张量，dtype 一致，shape 一致）。
- **输出**：out（BOOL 张量，一维包含一个元素，即 shape=[1]）。
- **计算公式**：`out = (self == other) ? True : False`，其中 `==` 为逻辑值相等比较（非二进制位比较）。
- **确定性计算**：相同输入多次执行，结果必须一致。

---

## 二、需求分析（required）

### 2.1 需求描述

在 A2/A3（Ascend910B）上，参考内置 `aclnnEqual`（TensorEqual）算子的 TBE 实现，基于 Ascend C 编程语言重实现功能一致的算子。核心改造点：**将逐元素比较从「二进制位比较」改为与 CPU 一致的「逻辑值比较」**。

### 2.2 需求拆解

1. **功能对齐**：与原 TBE `aclnnEqual` 核心功能完全一致（shape/dtype 不同返回 False；相同则逐元素比较并全归约成单 bool）。
2. **逻辑值比较语义**：
   - `+0.0` 与 `-0.0` 视为相等；
   - `NaN` 与任意值（含自身 `NaN`）视为不相等；
   - `+Inf` 与 `+Inf`、`-Inf` 与 `-Inf` 视为相等；
   - 整数类型按数值大小直接比较。
3. **数据类型支持**：FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、BOOL、UINT8（对齐任务书与 `aclnnEqual` 接口定义；整数/无符号类型可按 int 位宽归一到同一逻辑）。
4. **泛化与确定性**：满足各类合法输入场景，支持确定性计算。
5. **性能**：所有核参与计算的场景下，性能不低于原 TBE 算子的 95%。

---

## 三、详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

```
out = (shape(self) == shape(other) && dtype(self) == dtype(other) &&
       ∀i, logical_equal(self_i, other_i)) ? True : False
```

其中 `logical_equal(a, b)` 定义如下：

- 整数/布尔：`logical_equal(a, b) = (a == b)`；
- 浮点（fp16 / fp32 / bf16）：
  `logical_equal(a, b) = ((a == b) || (|a| == 0 && |b| == 0)) && !isnan(a) && !isnan(b)`

派生语义说明：
- `+0.0` 与 `-0.0`：`a == b` 的位比较可能为假，但 `|a| == 0 && |b| == 0` 为真（`abs` 将 `-0.0` 归一到 `+0.0`），故判为相等；
- `NaN` 与任意值：`isnan(a) || isnan(b)` 为真，整体判为不相等；
- `+Inf` 与 `+Inf`（或 `-Inf` 与 `-Inf`）：`a == b` 为真，且非 NaN，判为相等。

#### 3.1.2 支持数据类型

| 参数 | 输出/输入 | 数据类型 | 格式 |
|---|---|---|---|
| self | 输入 | FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, BOOL | ND |
| other | 输入 | 同 self | ND |
| out | 输出 | BOOL（shape=[1]） | ND |

> 说明：任务书明确 self/other 数据类型必须一致；支持清单以任务书 `FLOAT16/FLOAT/INT8/INT16/INT32/INT64/BOOL/BFLOAT16` 为基准，参考 TBE 实现补充 `UINT8`。DOUBLE/UINT16/UINT32/UINT64 在 `aclnnEqual` 头文件中声明，可按硬件支持扩展。

#### 3.1.3 支持形状

- 任意维度 ND（1D~8D）；
- **self 与 other 必须 shape 一致（不支持广播）**——这是 `torch.equal` 语义与逐元素 `torch.eq` 的本质区别：形状不一致时直接输出 False，不报错、不广播。

### 3.2 算子实现

#### 3.2.1 总体架构

采用 Ascend C 的 aclnn 算子工程化开发方式，u算子整体分为 host 侧 tiling 与 kernel 侧（Device）两阶段，kernel 侧遵循 `Init / Process(CopyIn / Compute / CopyOut)` 结构。

```
                        ┌──────────────────────────────┐
   self, other (GM) ──▶ │  host 侧 tiling（Tiling SoC） │
                        └──────────────┬───────────────┘
                                       │ tiling 参数
                        ┌──────────────▼───────────────┐
                        │  kernel 侧（多核并行）          │
                        │  Init → CopyIn → Compute     │
                        │        (逻辑值比较 + Reduce)   │
                        │       → 跨核归约 → CopyOut     │
                        └──────────────┬───────────────┘
                                       ▼
                              out (GM, 单 bool)
```

#### 3.2.2 host 侧设计（Tiling）

1. **维度规约**：TensorEqual 不涉及维度广播，因此 host 侧仅需将输入视为一维数据流，取 `totalLength = prod(shape)` 作为全局元素个数。
2. **tilingKey 规划**：host 侧感知输入合法性/形状，向 kernel 传递不同执行分支：
   - `EMPTY_SHAPE (101)`：元素个数为 0 → 直接按约定输出 True（空张量视为相等）；
   - `DIFF_SHAPE (111)`：`shape(self) != shape(other)` 或 dtype 不同 → 直接输出 False，不进入逐元素比较；
   - `NORMAL (121)`：shape/dtype 一致 → 执行逐元素逻辑值比较 + 全归约。
3. **分核策略**：优先满核。`coreNum = min(总核数, 满足 UB 约束的最大可用核数)`；若 `totalLength` 能按核均分，则无大小核区分；不能均分时，将余出的数据块分配给前几个核。
4. **数据分块与内存优化**：
   - 通过 `GetCoreMemSize` 获取单核 UB 大小，结合 `DOUBLE_BUFFER = 2` 与元素类型长度，计算单核每次搬运的最大元素数 `ubFactor`；
   - 由 `totalLength`、`coreNum`、`ubFactor` 推导 `perCoreLoopTimes`（每核循环次数）、`perCoreTailFactor`（每核尾块）、以及尾核的 `tailCoreLoopTimes/tailCoreTailFactor`，控制多次搬运合并、尾块不完整处理。
   - 每个核心子结果通过 Reduce 得到 1 个 bit/标量，减少跨核通信量。
5. **tiling 数据下发**：将 `tilingKey`、`usedCoreNum`、`ubFactor`、各循环/尾块参数封装为 `TensorEqualTilingData` 下发到 kernel 侧。

#### 3.2.3 kernel 侧设计（Device）

1. **Init**：
   - 每个核计算自己的 `blockOffset`（按元素长度换算到 GM 地址偏移）；
   - 初始化输入 `GlobalTensor` 与输出 `GlobalTensor`（输出为单 bool，仅 0 核负责初始化其默认值）；
   - `InitBuffer` 分配输入双缓冲队列 `inputXQueue_/inputYQueue_`（`DOUBLE_BUFFER=2`），以及 Reduce 所需的 `resultBuf_/saveBuf_`。

2. **CopyIn**：
   - 使用 `DataCopy / DataCopyPad` 将 self、other 的当前 tile 从 GM 搬运到 UB；
   - 循环处理 `perCoreLoopTimes` 次满块，再处理一次尾块（tail）。

3. **Compute（核心改进点：逻辑值比较）**：
   - 对每个 tile 内的元素，构造**逻辑值相等比较**而非原始 `vcmp eq` 位比较；
   - 浮点类型（fp16/fp32/bf16）单元素判定伪码：
     ```
     eq_bit    = Compare(a, b, EQ)                 // 原始相等比较（可为位级）
     abs_a     = Abs(a); abs_b = Abs(b)
     both_zero = Compare(abs_a, 0, EQ) && Compare(abs_b, 0, EQ)
     is_nan_a  = Compare(a, a, NE)                 // NaN != NaN
     is_nan_b  = Compare(b, b, NE)
     logical_eq = (eq_bit || both_zero) && !(is_nan_a || is_nan_b)
     ```
   - 整数/布尔类型：直接 `logical_eq = Compare(a, b, EQ)`（无需 NaN/±0 处理）；
   - 将逐元素 `logical_eq` 结果通过 `Select` 映射为 0/1（或直接用 compare mask），再对该核所有元素做 `ReduceMin / ReduceAll` 得到本核的「是否全部相等」子结果；
   - 实现可选用 Ascend C 的 `Compare`、`Select`、`Abs`、`IsNan`(或 `x != x` 惯用法)、`ReduceMin/ReduceAll` 等算子接口。

4. **跨核归约 + CopyOut**：
   - 每个核的局部归约结果汇总到输出；采用「0 核先将输出初始化为 True，其余核一旦发现存在不相等即写入 False」的短路式归约，或通过 `PartialReduce` 后由 0 核做最终 Reduce；
   - 最终将单个 BOOL 写回 `out`（shape=[1]），并用 `DataCacheCleanAndInvalid` 保持缓存一致性。

#### 3.2.4 现状对比小结

| 环节 | 原 TBE / Ascend C 实现 | 本设计（Ascend C） |
|---|---|---|
| 逐元素比较 | `vcmp eq` / `Compare EQ`（位比较） | 逻辑值比较（见 3.2.3 Compute） |
| ±0.0 | 判不等 | 判相等 |
| NaN | 可能判相等 | 判不等 |
| shape/dtype 分支 | tilingKey 区分 | 保留 tilingKey 区分（EMPTY/DIFF/NORMAL） |
| 归约 | reduce_min | ReduceMin/ReduceAll + 跨核归约 |

### 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |

### 3.4 算子约束限制

1. self 与 other 数据类型必须一致；
2. self 与 other 的 shape 必须一致（**不支持广播**，shape 不一致时输出 False）；
3. out 输出为 BOOL，shape=[1]（单元素）；
4. 相同输入多次执行结果一致（确定性计算）。

---

## 四、可维可测分析

### 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 输出与 CPU `torch.equal` 逻辑值比较结果完全一致（matched_ratio = 1.0），覆盖 `+0.0/-0.0`、`NaN`、`±Inf` 等边界场景 | 生态算子开源精度标准（rtol/atol = 0，required_matched_ratio = 1.0，max_abs_error_limit = 0） |
| 性能标准 | 所有核参与计算场景下，性能不低于原 TBE 算子的 95%；小 shape 无法达标时提供仿真图与分析结论 | 任务书 3.3 节 |

性能自测 case（与 TBE 基线在相同硬件对比）：`[1024,4096]`、`[4096,4096]`（FLOAT16/FLOAT/INT32 大 shape）、`[256,256]`（小 shape，可仿真替代）。

### 4.2 兼容性分析

本算子为社区开源仓新增/重构实现，接口与内置 `aclnnEqual` 保持一致（`aclnnEqualGetWorkspaceSize` / `aclnnEqual`），不涉及旧版本兼容性迁移，无需额外的兼容性分析。

---

## 五、参考与前提说明

1. 本设计文档以 `aclnnEqual = TensorEqual（torch.equal，单 bool 输出）` 为语义前提，依据如下（三者互相印证）：
   - **任务书接口**：`aclnnEqualGetWorkspaceSize/aclnnEqual`，签名与 `aclnn_equal.h` 完全一致；
   - **aclnn 头文件**：`aclnn_equal.h` 头防重宏为 `OP_API_INC_TENSOREQUAL_H_`，注释「计算两个 Tensor 是否有相同的大小和元素，返回一个 Bool」；而逐元素版本为 `aclnn_eq_tensor.h`（宏 `OP_API_INC_EQTENSOR_H_`），二者为独立算子；
   - **算子原型**：`REG_OP(TensorEqual)` 输出 `output_z` 仅 `DT_BOOL` 单元素（无广播语义）；`REG_OP(Equal)` 才声明「Support broadcasting」；
   - **TBE 参考实现**：`tensor_equal.py` 以 `Tensor_Equal` 注册，函数注释「True if two tensors have the same size, elements and dtype, False otherwise」，shape/dtype 不同直接返回 False、不做广播。
2. 任务书中“广播机制 / `out_i = self_i == other_i` / out 与广播后 shape 一致”等描述与本前提冲突，且提供的 `equal_testCase` 内 `op.json=EqTensor`、`golden.py=torch.eq`、README 自述面向 `aclnnEqTensor`，判断为任务书按逐元素模板书写导致的笔误；该测试用例仍可用于校验元素级「逻辑值比较」（±0.0、NaN、±Inf）的语义正确性。若最终确认目标确是逐元素 `aclnnEqTensor`，则需补充广播 tiling 与逐元素输出的设计。
3. 开发过程严格遵循 Ascend C 编程规范，交付前使用 AscendOpTest 完成自验证。
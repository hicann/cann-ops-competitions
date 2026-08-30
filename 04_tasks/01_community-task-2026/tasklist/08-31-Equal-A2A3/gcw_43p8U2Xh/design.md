# aclnnEqual 算子设计文档

**版本**: v1.0  
**日期**: 2026-08-26  
**适配硬件**: Atlas A2 训练系列产品 / Atlas A3 系列产品  
**开发语言**: Ascend C

---

## 1. 需求背景

### 1.1 需求来源

8月社区任务：基于 Ascend C 编程语言实现 aclnnEqual 算子，替代原 TBE 实现。

**核心变更点**：比较方式从二进制位比较更改为与 CPU 一致的**逻辑值比较**。

### 1.2 背景介绍

aclnnEqual 为逐元素相等比较算子，输入两个张量 `self` 和 `other`，输出 BOOL 类型结果。原 TBE 实现采用二进制位比较，导致 `+0.0` 与 `-0.0` 被视为不等，与 PyTorch/CPU 语义不一致。本次任务要求基于 Ascend C 重新实现，确保逻辑值比较语义正确，同时保持原 TBE 算子的数据类型覆盖范围和性能水平。

### 1.3 现有实现现状分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|---------|---------|-------------|------|------|
| self | 输入 tensor | tensor | fp16/fp32/int8/int16/int32/int64/bool/bf16 | ND，支持广播 | 任意 |
| other | 输入 tensor | tensor | 与 self 一致 | ND，支持广播 | 任意 |
| out | 输出 tensor | tensor | bool | ND | broadcast 后 shape |

**当前 TBE 实现缺陷**：二进制位比较导致 `+0.0 != -0.0`，NaN 行为与 CPU 不一致。

---

## 2. 需求分析

### 2.1 需求描述

基于 Ascend C 实现 aclnnEqual 算子，支持逻辑值比较，覆盖原 TBE 全部数据类型与功能场景。

### 2.2 需求拆解

1. **Host 侧**：实现算子定义、广播推导、参数校验、类型推导（输出固定为 BOOL）。
2. **Kernel 侧**：模板泛型实现逐元素逻辑值比较，统一输出 `bool`/`uint8_t`。
3. **逻辑值比较**：浮点类型需正确处理 `+0.0/-0.0`、`NaN`、`Inf`；整数与 BOOL 按数值比较。
4. **性能**：不低于原 TBE 实现的 95%。

---

## 3. 详细设计

### 3.1 算子分析

#### 3.1.1 数学公式

逐元素逻辑值相等比较：

```
out_i = (self_i == other_i) ? True : False
```

#### 3.1.2 逻辑值比较语义

| 场景 | 逻辑值比较结果 | 说明 |
|------|---------------|------|
| `+0.0 == -0.0` | True | 符号不同但数值同为 0 |
| `NaN == x`（任意 x） | False | NaN 与任何值都不等，包括自身 |
| `+Inf == +Inf` | True | 同号无穷大 |
| `-Inf == -Inf` | True | 同号无穷大 |
| `int_a == int_b` | 数值相等为 True | 直接数值比较 |
| `bool_a == bool_b` | 直接比较 | 0/1 数值比较 |

#### 3.1.3 支持数据类型

| 数据类型 | 输入（self/other） | 输出（out） |
|---------|-------------------|-------------|
| FLOAT16 | √ | - |
| FLOAT | √ | - |
| INT8 | √ | - |
| INT16 | √ | - |
| INT32 | √ | - |
| INT64 | √ | - |
| BOOL | √ | - |
| BFLOAT16 | √ | - |
| **BOOL（输出）** | - | **√** |

#### 3.1.4 支持形状

- 任意维度，支持 0-D（标量）到 8-D。
- 支持 ND 格式。
- 支持广播（含标量广播）。
- 支持非连续 Tensor（通过 stride 处理）。

### 3.2 实现方案

#### 3.2.1 Host 侧设计

**（1）算子原型定义 (`op_host/equal_def.cpp`)**

```cpp
this->Input("self")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16,
               ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ...})
    .UnknownShapeFormat({ge::FORMAT_ND, ...});

this->Input("other")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16,
               ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ...})
    .UnknownShapeFormat({ge::FORMAT_ND, ...});

this->Output("out")
    .ParamType(REQUIRED)
    .DataType({ge::DT_BOOL})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
```

**（2）参数校验**

- self 与 other dtype 必须一致，否则返回 `ACLNN_ERR_PARAM_INVALID`。
- self 与 other shape 必须满足广播规则，否则返回错误。
- out dtype 必须为 BOOL，shape 必须与广播后结果一致。
- 维度数不超过 8。

**（3）广播推导（InferShape）**

调用 `Ops::Base::InferShape4Broadcast` 或自定义广播逻辑：

1. 从后向前对齐维度，缺失维度补 1。
2. 对应维度要么相等，要么其中一个为 1。
3. 输出 shape 为各维度取最大值。
4. 输出 dtype 固定推导为 `DT_BOOL`。

**（4）类型推导（InferDataType）**

- 输出固定为 `DT_BOOL`，与输入 dtype 无关。
- 需校验 `self.dtype == other.dtype`。

**（5）Tiling 策略**

Equal 为逐元素 elementwise 算子，Tiling 按元素个数均分：

- **分核**：按总元素数均分到各 AI Core。
- **单核分块**：根据 UB 容量和输入 dtype 大小计算 tile 元素数。
- **UB 分配**：输入为双源（self + other），输出为单源（out），UB 中需同时驻留 self、other 的 tile 及 out 的 tile。
- **非连续处理**：若输入非连续，通过 stride 计算偏移；广播场景在 Kernel 侧通过广播 stride 处理。

#### 3.2.2 Kernel 侧设计

**（1）整体架构**

采用 Ascend C 标准三段式：`CopyIn → Compute → CopyOut`。

```cpp
template <typename T>
class KernelEqual {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out,
                                TPipe* pipe, const EqualTilingData* tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count);
    __aicore__ inline void Compute(int64_t count);
    __aicore__ inline void CopyOut(int64_t offset, int64_t count);

    GlobalTensor<T> selfGm_;
    GlobalTensor<T> otherGm_;
    GlobalTensor<uint8_t> outGm_;  // bool 用 uint8_t 存储

    TQue<TPosition::VECIN, 2> selfQue_;
    TQue<TPosition::VECIN, 2> otherQue_;
    TQue<TPosition::VECOUT, 2> outQue_;

    const EqualTilingData* tilingData_;
};
```

**（2）Compute 逻辑（核心）**

逐元素比较，分类型处理：

- **整数类型（INT8/16/32/64/BOOL）**：直接 `==` 比较。
- **浮点类型（FLOAT16/FLOAT/BFLOAT16）**：需处理特殊值。

浮点逻辑值比较实现方案（Ascend C 设备侧）：

```cpp
template <typename T>
__aicore__ inline bool LogicalEqual(T a, T b) {
    // 1. 处理 NaN：只要有一个是 NaN，返回 false
    if (IsNan(a) || IsNan(b)) {
        return false;
    }
    // 2. 处理 +0.0 与 -0.0：数值均为 0 时视为相等
    //    通过 a - b == 0 判断，但需先排除 NaN
    return (a == b);
}
```

**注**：在 IEEE 754 中，`+0.0 == -0.0` 硬件比较结果为 true，因此直接使用 `==` 即可满足 `+0.0/-0.0` 相等的语义。`NaN == NaN` 硬件结果为 false，也恰好满足逻辑值比较要求。因此，**Ascend C 设备侧的 `==` 运算天然满足逻辑值比较语义，无需额外位操作。**

**（3）广播处理**

Host 侧计算广播后的 stride（broadcast stride），传入 TilingData：

- 若某维度被广播（原 shape 为 1，目标 shape > 1），该维度 stride 设为 0。
- Kernel 侧通过 `offset = index * stride` 计算实际读取位置，实现广播读取。

**（4）非连续处理**

Host 侧将物理 stride 传入 TilingData，Kernel 侧按 stride 索引读取，不假设连续。

#### 3.2.3 Tiling 数据结构

```cpp
struct EqualTilingData {
    int64_t totalElements;    // 总元素数
    int64_t blockNum;         // 核数
    int64_t blockOffset;      // 当前核起始偏移
    int64_t blockElements;    // 当前核负责元素数
    int64_t tileElements;     // 单 tile 元素数
    int64_t tileCount;        // tile 数
    int64_t tailElements;     // 尾 tile 元素数

    // 广播 stride
    int64_t selfStrides[8];
    int64_t otherStrides[8];
    int64_t outStrides[8];

    int64_t dimNum;
    int64_t selfShape[8];
    int64_t otherShape[8];
};
```

#### 3.2.4 精度与性能设计

**精度策略**：

- 输出为 BOOL，要求与 CPU 结果 bit-exact（matched_ratio = 1.0）。
- 浮点 `==` 在 Ascend C 上为硬件原生指令，语义与 IEEE 754 一致，满足逻辑值比较要求。

**性能策略**：

- 采用 double buffer，self/other 同时预取，计算与访存流水。
- 向量化读写：按 32/sizeof(T) 对齐批量处理。
- 广播场景下，若输入为标量（0-D），可单独走标量广播优化路径（标量先读入 UB，再用 Duplicate 广播到 tile 大小，减少 GM 重复读取）。
- 多核均分：按 totalElements 均分，避免尾核空闲。

---

## 4. 支持硬件

| 支持的芯片版本 | 是否支持 |
|---------------|---------|
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |

---

## 5. 算子约束限制

1. self 与 other 数据类型必须一致。
2. out 数据类型必须为 BOOL。
3. self 与 other shape 必须满足广播规则。
4. out shape 必须与广播后结果 shape 一致。
5. 最大支持维度数为 8。
6. 仅支持 ND 格式。

---

## 6. 可维可测分析

### 6.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---------|------|---------|
| 精度标准 | 输出与 CPU（PyTorch `torch.equal` 语义）bit-exact 对齐，matched_ratio = 1.0 | 社区任务书 |
| 性能标准 | 所有核参与场景下，性能不低于原 TBE 实现的 95% | 社区任务书 |
| 回归标准 | 泛化测试用例 100% 通过 | 社区任务书 |

### 6.2 测试用例设计

**功能用例**：

1. 同 shape 比较：各 dtype，2D/3D/4D。
2. 广播场景：（4,3,2）vs（1,3,1），标量 vs 张量。
3. 边界值：
   - `+0.0 vs -0.0` → True
   - `NaN vs NaN` → False
   - `NaN vs 1.0` → False
   - `+Inf vs +Inf` → True
   - `-Inf vs +Inf` → False
4. 非连续 tensor：带 stride 的切片输入。
5. 0-D 标量输入。

**性能用例**：

- FLOAT16 [1024, 4096]、[4096, 4096]
- FLOAT [1024, 4096]
- INT32 [1024, 4096]
- FLOAT16 [256, 256]（小 shape 仿真分析）

---

## 7. 兼容性分析

- **前向兼容**：新增 Ascend C 实现，接口签名与原 TBE 版本保持一致。
- **后向兼容**：逻辑值比较语义修正后，与 PyTorch/CPU 对齐，但可能与原二进制比较的旧 TBE 实现存在 `+0.0/-0.0` 场景的差异，属于预期内的语义修正。
- **框架兼容**：支持 `torch.eq` 等上层接口在 NPU 上的正确映射。

---

## 8. 实现检查清单

### 8.1 文件结构

```
├── op_host/
│   ├── equal.cpp           # 算子实现入口
│   ├── equal_def.cpp       # 算子原型定义
│   └── equal_tiling.h      # Tiling 数据结构定义
├── op_kernel/
│   ├── equal_kernel.h      # Kernel 核心实现
│   └── equal_kernel.cpp    # Kernel 入口
└── README.md               # 接口文档
```

### 8.2 代码要点

- 确保 Kernel 侧处理 `+0.0/-0.0`、`NaN`、`Inf` 等边界值的正确性。
- 确保广播场景下的 stride 计算正确。
- 确保 BOOL 输出使用 `uint8_t` 存储，值为 0/1。

### 8.3 测试要点

- 使用 AscendOpTest 工具进行全场景自验证。
- 覆盖各 dtype、广播、边界值、大 shape、小 shape 场景。
- 精度比对要求 matched_ratio = 1.0。
- 性能与 TBE 基线对比，要求不低于 95%。
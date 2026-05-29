# aclnnForeachAddScalarV2 算子设计文档

## 一、 需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献的需求，补充完善 Ascend C 算子库。为现有的 `aclnnForeachAddScalarV2` 算子增加对 `DT_INT16`、`DT_INT8`、`DT_UINT8` 数据类型的支持，以满足网络量化及端侧推理场景对底层核心算子的完备性要求。

### 1.2 背景介绍

`ForeachAddScalar` 算子用于对张量列表（Tensor List）执行逐元素的标量加法运算。其核心数学计算公式为：
$y_i = x_i + \alpha$
根据最新算子原型定义，当前算子已支持 `FLOAT16`、`FLOAT32`、`INT32`、`BFLOAT16`。本项目要求在现有代码架构下，新增对 `INT16`、`INT8`、`UINT8` 这三种整型数据的支持。

**关键架构决策**：
为了保持底层 `InnerComputer` 计算流的统一性与极简性，并兼顾精度，本设计采用精度提升（Cast）机制：

* **`int16_t`**：统一转换（Cast）为 `float` (float32) 进行 `Adds` 处理，计算完成后截断回 `int16_t`。
* **`int8_t` / `uint8_t**`：统一转换（Cast）为 `half` (float16) 进行 `Adds` 处理，计算完成后截断回原 8-bit 类型。

## 二、 需求分析

### 2.1 外部组件依赖

* 依赖 Ascend C 的 `Cast` API 提供低比特整型与 `float` / `half` 之间的精度双向转换及饱和截断（`CAST_RINT`）机制。
* 依赖现有的 `ForeachOneScalarBinary` 通用单标量二元操作模板框架。

### 2.2 内部适配模块

| 模块 | 变更项 | 说明 |
| --- | --- | --- |
| `op_host` | `foreach_add_scalar_def.cpp` | 扩展原型 `DataType`，新增 `DT_INT16/DT_INT8/DT_UINT8`。明确 `scalar` 的对应关系。 |
| `op_host` | `foreach_tiling_func.cpp` | **修改 UB 切分策略**：`int16_t` 按照 `float` (4 Bytes) 预留；`int8_t/uint8_t` 按照 `half` (2 Bytes) 预留。 |
| `op_kernel` | `foreach_add_scalar.cpp` | 新增 `TILING_KEY_IS` 路由分支：`int16_t` 路由至 `float` 适配器；`int8/uint8` 路由至 `half` 适配器。 |
| `op_kernel` | `foreach_one_scalar_binary.h` | 扩展 `InnerComputer` 模板特化，新增整型转浮点型执行运算的双向转换流水线编排。 |
| `op_kernel` | `kernel_foreach_base.h`<br>

<br>`kernel_foreach_unary.h` | 补充底层基础模板对这三种整型的识别萃取及 `COPY_SPACE_MULTIPLE` 倍数控制。 |

## 三、 需求详细设计

### 3.1 使能方式

| 上层框架 | 涉及的框架勾选 |
| --- | --- |
| TF训练/推理 |  |
| Pytorch训练/推理 |  |
| ATC推理 |  |
| Aclnn直调 | ✅ |

### 3.2 算子原型设计

### 算子原型表格

该表格清晰界定了算子的输入输出类型、维度约束及格式要求。

| 名称 | 类别 | 数据类型 | Format | Shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| **x** | 输入 | FLOAT16, FLOAT32, BF16, INT32, **INT16, INT8, UINT8** | ND | 0-8 维 | 输入张量列表，列表中所有 Tensor 类型需一致 |
| **scalar** | 输入 | FLOAT16, FLOAT32, FLOAT32, INT32, **INT32, INT32, INT32** | ND | [1] | 待加的标量值，与 Tensor 元素类型保持绑定 |
| **y** | 输出 | FLOAT16, FLOAT32, BF16, INT32, **INT16, INT8, UINT8** | ND | 与 x 一致 | 输出张量列表，结果 $y_i = x_i + \alpha$ |

修改 `op_host/foreach_add_scalar_def.cpp`，同步更新数据类型并严格限制硬件架构：

```cpp
explicit ForeachAddScalar(const char* name) : OpDef(name)
{
    // 输入张量支持的 dtype 列表新增整型
    std::vector<ge::DataType> tensor_dtype_list = {
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16,
        ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8
    };
    
    // 标量支持的 dtype 列表（BF16 对应 FLOAT，整型严格一一对应）
    std::vector<ge::DataType> scalar_dtype_list = {
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32
    };

    std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);

    this->Input("x")
        .ParamType(DYNAMIC)
        .DataType(tensor_dtype_list)
        .Format(format_list)
        .UnknownShapeFormat(format_list)
        .AutoContiguous();
        
    this->Input("scalar")
        .ParamType(REQUIRED)
        .DataType(scalar_dtype_list)
        .Format(format_list)
        .UnknownShapeFormat(format_list)
        .AutoContiguous();
        
    this->Output("y")
        .ParamType(DYNAMIC)
        .DataType(tensor_dtype_list)
        .Format(format_list)
        .UnknownShapeFormat(format_list);

    OpAICoreConfig membaseCfg;
    membaseCfg.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .ExtendCfgInfo("opFile.value", "foreach_add_scalar");
        
    // 仅保留 A2/A3 对应的系列
    this->AICore().AddConfig("ascend910b", membaseCfg);
    this->AICore().AddConfig("ascend910_93", membaseCfg);
}

```

### 3.3 详细设计

#### 3.3.1 Host 侧设计：UB 切分策略修改

在 `foreach_utils/op_host/foreach_tiling_func.cpp` 中，负责 UB 切分的逻辑需根据中间计算类型分类处理：

1. **`DT_INT16` (Key 5)**：由于在 V 单元内部必须先 Cast 成 `float32` 才能进行计算，因此在 Host 侧计算单次最大处理数据量（`ub_size` 和 `tileDataNum`）时，必须**按照 `float32`（4 字节）的内存诉求来预留和切分 UB 空间**。
2. **`DT_INT8` (Key 7) / `DT_UINT8` (Key 8)**：由于在 V 单元内部将 Cast 成 `half` (float16) 进行计算，必须**按照 `half`（2 字节）的内存诉求来预留和切分 UB 空间**。
两者均需应用相应的 `COPY_SPACE_MULTIPLE` 逻辑，确保中转所需的临时张量空间充足。

#### 3.3.2 Kernel 侧设计

**3.3.2.1 新增 TilingKey 分支**
在 `foreach_add_scalar.cpp` 中，增加针对三种新类型的入口调用。

```cpp
// ... 现有 FLOAT16, FLOAT32, INT32 分支 ...
#if __CCE_AICORE__ >= 220
    else if (TILING_KEY_IS(3)) {
        ForeachOneScalarBinary<int, int, AddsAdapter<int>, 1> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(4)) {
        ForeachOneScalarBinary<bfloat16_t, float, AddsAdapter<float>, 1> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        // int16_t 提升至 float 计算
        ForeachOneScalarBinary<int16_t, float, AddsAdapter<float>, 1> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        // int8_t 提升至 half 计算
        ForeachOneScalarBinary<int8_t, half, AddsAdapter<half>, 1> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        // uint8_t 提升至 half 计算
        ForeachOneScalarBinary<uint8_t, half, AddsAdapter<half>, 1> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
#endif
    }
#endif

```

**3.3.2.2 扩展 `InnerComputer` 模板特化**
在 `foreach_one_scalar_binary.h` 中补充特化。

* 针对 `int16_t`，引入 `<int16_t, float, op>` 的特化：`Cast(floatTensor, int16_Tensor)` -> `Adds(floatTensor, scalar_float)` -> `Cast(int16_Tensor, floatTensor, CAST_RINT)`。
* 针对 `8-bit`，引入 `<int8_t, half, op>` 及 `<uint8_t, half, op>` 的特化，使用 `half` 张量作为指令计算的中转站。
* **注意标量转换**：通过类型偏特化机制，将标量安全读取为内部的 `TT` 类型（即 `float` 或 `half`），防范底层编译器的 `float = unsigned` 非法转换拦截。

#### 3.3.3 Ascend C 侧运行流程图

![image.png](https://raw.gitcode.com/user-images/assets/9843185/8259e166-b9be-48e5-850a-c145148ee349/image.png 'image.png')

### 3.4 支持硬件

* **Atlas A2 训练系列产品 / Atlas A3 系列产品** (对应 `ascend910_93` 及 `ascend910b`)。

### 3.5 算子约束与异常拦截机制

1. **构建阻断拦截**：如果在前端执行或编译期发现未注册的类型组合（例如输入 dtype 不在定义的列表内），构建流程将直接报错失败，**绝不触发降级至 CPU 的回退机制**。原型库 (`op_def`)、Tiling 策略与 Kernel 侧的泛型分支必须维持强一致闭环。
2. **张量类型一致性**：输入的 `x` 张量列表、`scalar` 标量张量、输出的 `y` 张量列表在对应维度上的数据类型必须完全匹配。

## 四、 验收及可维可测标准

### 4.1 精度与性能标准

* **精度标准**：满足 AscendOpTest 默认阈值。由于整型算子底层全部执行提权计算，整数运算的期望基准应为：
* `int16_t`：转化为 `float64/float32` -> 计算 $x_i + \alpha$ -> 采用 `round` 四舍五入 -> 执行 `int16_t` 边界的饱和截断。
* `int8/uint8`：转化为 `float16` -> 计算 $x_i + \alpha$ -> 采用 `round` 四舍五入 -> 执行目标数据类型边界的饱和截断。


* **性能标准**：增加类型的吞吐率应对齐已有涉及双向 Cast 操作的 `BFLOAT16` 方案，确保计算单元 (V) 与搬运单元 (MTE2/MTE3) 的流水线被充分掩盖。

## 五、 变更文件清单

| 文件路径 | 变更类型 | 变更说明 |
| --- | --- | --- |
| `op_host/foreach_add_scalar_def.cpp` | 修改 | 原型扩展新增 `INT16/INT8/UINT8`；明确硬件配置仅含 `ascend910_93` 和 `ascend910b`。 |
| `op_host/config/.../foreach_add_scalar_binary.json` | 修改 | 为对应硬件架构补充 `INT16/INT8/UINT8` 的编译条目。 |
| `op_kernel/foreach_add_scalar.cpp` | 修改 | 引入针对 `TILING_KEY_IS(5)`, `(7)`, `(8)` 的分发；5 使用 float 适配，7 和 8 使用 half 适配。 |
| `../foreach_utils/op_host/foreach_tiling_func.cpp` | 修改 | 调整 UB 切分策略：`int16_t` 基于 `float32` (4 Bytes) 预留；`int8_t/uint8_t` 基于 `half` (2 Bytes) 预留。 |
| `../foreach_utils/op_kernel/foreach_one_scalar_binary.h` | 修改 | 补充 `int16_t -> float` 及 `8-bit -> half` 的 `InnerComputer` 模板特化计算流，并实现安全的 Scalar 类型转换。 |
| `../foreach_utils/op_kernel/kernel_foreach_base.h` <br>

<br> `kernel_foreach_unary.h` | 修改 | 补充底层基础模板对这三种整型的识别萃取及 `COPY_SPACE_MULTIPLE` 控制。 |
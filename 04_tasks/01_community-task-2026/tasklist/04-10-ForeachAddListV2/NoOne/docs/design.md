# aclnnForeachAddListV2 算子设计文档

## 一、 需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献的需求，补充完善 Ascend C 算子库。为现有的 `aclnnForeachAddListV2` 算子增加对 `DT_INT16`、`DT_INT8`、`DT_UINT8` 数据类型的支持，以满足更广泛的网络量化及端侧推理场景对底层核心算子的完备性要求。

### 1.2 背景介绍

`ForeachAddListV2` 算子用于对两个张量列表（Tensor List）执行逐元素的带缩放加法运算。其核心数学计算公式为：
$y_i = x1_i + \alpha \times x2_i$
根据最新算子原型定义，当前算子已支持 `FLOAT16`, `FLOAT32`, `INT32`, `BFLOAT16` 数据类型。本项目要求在现有代码架构下，新增对 `INT16`、`INT8`、`UINT8` 数据的支持。

**关键架构决策**：虽然在 Atlas A2/A3 架构上，底层 `Add` 与 `Muls` 指令已原生支持 `int16_t` 数据类型，但考虑到 `foreach` 模块是一个高度抽象的通用模板框架，服务于众多算子，为了保持底层 `InnerComputer` 计算流的统一性与极简性，本设计决定**不走原生分支，而是将 `int16_t` 统一转换（Cast）为 `float`（float32）进行处理**。同理，对于 `int8_t` 和 `uint8_t`，则通过精度提升转换至 `half` (float16) 并在计算后再转换回原类型。

## 二、 需求分析

### 2.1 外部组件依赖

* 依赖 Ascend C 的 `Cast` API 提供 `int16_t` 与 `float32`、以及 `int8_t`/`uint8_t` 与 `half` 之间的精度转换及饱和截断机制。
* 依赖现有的 `ForeachOneScalarTernary` 通用三元标量模板框架。

### 2.2 内部适配模块

| 模块 | 变更项 | 说明 |
| --- | --- | --- |
| `op_host` | `foreach_add_list_def.cpp` | 扩展主配置的 `tensor_dtype_list`，新增 `DT_INT16/DT_INT8/DT_UINT8`。更新硬件配置，仅保留 `ascend910_93` 和 `ascend910b`。 |
| `op_host` | `foreach_tiling_func.cpp` | **修改 UB 切分策略**：`int16_t` 按照中间缓存 `float32` (4 Bytes) 预留内存；`int8_t/uint8_t` 按照 `half` (2 Bytes) 预留内存。 |
| `op_kernel` | `foreach_add_list.cpp` | 新增 `TILING_KEY_IS` 路由分支：`int16_t` 路由至 `float` 适配器；`int8/uint8` 路由至 `half` 适配器。 |
| `op_kernel` | `foreach_one_scalar_ternary.h` | 扩展 `InnerComputer` 模板特化，新增 `int16_t -> float` 及 `8-bit -> half` 执行运算的双向转换流水线编排。 |
| `op_api` | `aclnn_foreach_add_list_v2.cpp` | 从调用层面上增加三种新增数据类型的支持。 |

## 三、 需求详细设计

### 3.1 使能方式

| 上层框架 | 涉及的框架勾选 |
| --- | --- |
| TF训练/推理 |  |
| Pytorch训练/推理 |  |
| ATC推理 |  |
| Aclnn直调 | ✅ |

### 3.2 算子原型设计

修改 `op_host/foreach_add_list_def.cpp`，同步更新数据类型并**严格限制硬件架构注册**：

```cpp
		std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8};
        std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
        std::vector<ge::DataType> scalar_tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
        this->Input("x1")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();
        this->Input("x2")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();
        this->Input("alpha")
            .ParamType(REQUIRED)
            .DataType(scalar_tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list);
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend910b");

```

### 3.3 详细设计

#### 3.3.1 Host 侧设计：UB 切分策略修改

在 `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` 中，负责 UB 切分的逻辑需根据目标中间计算类型分类处理：

1. **`DT_INT16` (Key 5)**：由于在 V 单元内部必须先 Cast 成 `float32` 才能进行计算，因此在 Host 侧计算单次最大处理数据量 (`ub_size` 和 `tileDataNum`) 时，必须**按照 `float32`（4 字节）的内存诉求来预留和切分 UB 空间**。
2. **`DT_INT8` (Key 7) / `DT_UINT8` (Key 8)**：在 V 单元内部将 Cast 成 `half` (float16) 进行计算，因此必须**按照 `half`（2 字节）的内存诉求来预留和切分 UB 空间**。
两者均需应用相应的 `COPY_SPACE_MULTIPLE` 逻辑，确保 `InnerComputer` 特化中引入的额外 `LocalTensor` 不会发生内存踩踏。

#### 3.3.2 Kernel 侧设计

**3.3.2.1 新增 TilingKey 分支**
在 `foreach_add_list.cpp` 中，增加针对三种新类型的入口调用，统一利用 `AddListFloatAdapter` 的高精度指令（Axpy）：

```cpp
// ... 现有 FLOAT16, FLOAT32, INT32, BFLOAT16 分支 ...
#if __CCE_AICORE__ >= 220
    } else if (TILING_KEY_IS(5)) {
        // int16_t 转 float 计算，保持通用性
        ForeachOneScalarTernary<int16_t, float, AddListFloatAdapter<float>> op;
        op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        // int8_t 转 half 计算
        ForeachOneScalarTernary<int8_t, half, AddListFloatAdapter<half>> op;
        op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        // uint8_t 转 half 计算
        ForeachOneScalarTernary<uint8_t, half, AddListFloatAdapter<half>> op;
        op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
        op.Process();
    }
#endif

```

**3.3.2.2 扩展 `InnerComputer` 模板特化**
在 `foreach_one_scalar_ternary.h` 中补充特化。

* 对于 `int16_t`，引入 `<int16_t, float, op>` 的特化，核心为 `Cast(floatTensor, int16_Tensor)` -> 运算 -> `Cast(int16_Tensor, floatTensor, CAST_RINT)`。
* 对于 `8-bit`，引入 `<int8_t, half, op>` 及 `<uint8_t, half, op>` 的特化，使用 `half` 作为中转张量。

#### 3.3.3 Ascend C 侧运行流程图

![image.png](https://raw.gitcode.com/user-images/assets/9516645/a4e2f951-7cee-48c2-8ec0-cc4f821fc108/image.png 'image.png')

### 3.4 支持硬件

* **Atlas A2 训练系列产品 / Atlas A3 系列产品**

### 3.5 算子约束与异常拦截机制

1. **构建拦截规则（严格）**：构建过程中如果发现不支持的算子或未能匹配原型中声明的数据类型组合，会直接构建失败并导致流程中断报错，**绝不会触发回退机制将该算子标记为 CPU 执行**。因此，原型库 (`op_def`)、Tiling 策略与 Kernel 侧泛型实例化必须维持强一致的闭环。
2. **张量类型一致性**：输入列表 `x1`, `x2` 以及输出列表 `y` 内部及之间的数据类型必须完全一致。

## 四、 验收及可维可测标准

### 4.1 精度与性能标准

* **精度标准**：满足 AscendOpTest 默认阈值。由于底层全部执行了提权计算，整数运算的期望基准应为：
* `int16_t`：转化为 `float64/float32` -> 计算 $x1_i + \alpha \times x2_i$ -> 采用 `round` 四舍五入 -> 执行 `int16_t` 边界的饱和截断。
* `int8/uint8`：转化为 `float16` -> 计算 $x1_i + \alpha \times x2_i$ -> 采用 `round` 四舍五入 -> 执行目标数据类型边界的饱和截断。


* **性能标准**：三种整型的吞吐率应对齐已有涉及双向 Cast 逻辑的 `BFLOAT16` 方案，确保 V 单元与搬运单元 (MTE2/MTE3) 之间流水线被充分掩盖。

## 五、 变更文件清单

| 文件路径 | 变更类型 | 变更说明 |
| --- | --- | --- |
| `op_host/foreach_add_list_def.cpp` | 修改 | 原型扩展新增 `INT16/INT8/UINT8`；修正硬件配置为仅包含 `ascend910_93` 和 `ascend910b`。 |
| `op_host/config/.../foreach_add_list_binary.json` | 修改 | 为各支持架构（含 910_93 和 910b）补充新 dtype 的编译构建条目。 |
| `op_host/op_api/aclnn_foreach_add_list_v2.cpp` | 修改 | 扩展新增 `INT16/INT8/UINT8`，适配外部调用。 |
| `op_kernel/foreach_add_list.cpp` | 修改 | 引入针对 `TILING_KEY_IS(5)`, `(7)`, `(8)` 的识别。5 路由至 float 适配器，7 和 8 路由至 half 适配器。 |
| `../foreach_utils/op_host/foreach_tiling_func.cpp` | 修改 | 调整 UB 切分策略：`int16_t` 基于 `float32` (4 Bytes) 预留；`int8_t/uint8_t` 基于 `half` (2 Bytes) 预留。 |
| `../foreach_utils/op_kernel/foreach_one_scalar_ternary.h` | 修改 | 补充 `int16_t -> float` 及 `int8_t/uint8_t -> half` 的 `InnerComputer` 模板特化计算流。 |

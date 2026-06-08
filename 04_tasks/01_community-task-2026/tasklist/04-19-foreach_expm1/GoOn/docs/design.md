# aclnnForeachExpm1 算子设计文档

## 一、 需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献的需求，补充完善 Ascend C 算子库。为现有的 `aclnnForeachExpm1` 算子增加对 `DT_INT16`、`DT_INT8`、`DT_UINT8` 数据类型的支持，以满足更广泛的网络量化及端侧推理场景对底层核心算子的完备性要求。

### 1.2 背景介绍

`ForeachExpm1` 算子用于对张量列表（Tensor List）执行逐元素的 $y = e^x - 1$ 运算。由于在 $x$ 接近 $0$ 时，$e^x - 1$ 的直接计算易导致精度丢失，该算子能提供比组合算子更好的数值稳定性。
根据最新算子原型定义，当前算子已支持 `FLOAT16`、`FLOAT32`、`BFLOAT16`。本项目要求在现有代码架构下，新增对 `INT16`、`INT8`、`UINT8` 数据的支持。

**关键架构决策**：
考虑到 `foreach` 模块是高度抽象的通用框架，为兼顾精度与底层指令的泛化性，对于新增的整型类型，采用精度提升（Cast）机制：

* **`int16_t`**：Cast 至 `float`（float32）进行高精度计算，计算完毕后通过 `CAST_RINT` 截断回 `int16_t`。
* **`int8_t` / `uint8_t**`：Cast 至 `half`（float16）进行计算，计算完毕后截断回 `int8_t` / `uint8_t`。

## 二、 需求分析

### 2.1 外部组件依赖

* 依赖 Ascend C 的 `Cast` API 提供低比特整型与 `float` / `half` 之间的精度双向转换及饱和截断机制。
* 依赖原有的 `Expm1Adapter`（组合 `Exp` 与 `Adds`）进行核心逻辑的运算。
* 依赖现有的 `ForeachImplictOutput` 通用隐式输出模板框架。

### 2.2 内部适配模块

| 模块 | 变更项 | 说明 |
| --- | --- | --- |
| `op_host` | `foreach_expm1_def.cpp` | 扩展原型 `tensor_dtype_list`，新增 `DT_INT16/DT_INT8/DT_UINT8`。更新硬件配置为 `ascend910_93` 和 `ascend910b`。 |
| `op_host` | `foreach_tiling_func.cpp` | **修改 UB 切分策略**：`int16_t` 按照 `float` (4 Bytes) 预留中间缓存；`int8_t/uint8_t` 按照 `half` (2 Bytes) 预留中间缓存。 |
| `op_kernel` | `foreach_expm1.cpp` | 新增 `TILING_KEY_IS` 路由分支：5 路由至 `float` 适配器；7、8 路由至 `half` 适配器。 |
| `op_kernel` | `foreach_implict_output.h` | 扩展 `InnerComputer` 模板特化，新增整型转浮点型进行运算的流水线拆分。 |
| `op_kernel` | `kernel_foreach_unary.h` / `base.h` | 补充 `INT16/INT8/UINT8` 类型映射，支持对应的 `COPY_SPACE_MULTIPLE`（临时转换缓冲）分配逻辑。 |

## 三、 需求详细设计

### 3.1 使能方式

| 上层框架 | 涉及的框架勾选 |
| --- | --- |
| TF训练/推理 |  |
| Pytorch训练/推理 |  |
| ATC推理 |  |
| Aclnn直调 | ✅ |

### 3.2 算子原型设计

修改 `op_host/foreach_expm1_def.cpp`，同步更新数据类型，并根据任务约束**严格限制硬件架构注册**：

```cpp
explicit ForeachExpm1(const char* name) : OpDef(name)
{
    std::vector<ge::DataType> tensor_dtype_list = {
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
        ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8 // 新增
    };
    std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
    
    this->Input("x")
        .ParamType(DYNAMIC)
        .DataType(tensor_dtype_list)
        .Format(format_list)
        .UnknownShapeFormat(format_list)
        .AutoContiguous();
        
    this->Output("y")
        .ParamType(DYNAMIC)
        .DataType(tensor_dtype_list)
        .Format(format_list)
        .UnknownShapeFormat(format_list)
        .AutoContiguous();
        
    // 仅保留 A2/A3 对应的架构
    this->AICore().AddConfig("ascend910_93");
    this->AICore().AddConfig("ascend910b");
}

```

### 3.3 详细设计

#### 3.3.1 Host 侧设计：UB 切分策略修改

在 `foreach_utils/op_host/foreach_tiling_func.cpp` 中，负责 UB 切分的逻辑需根据计算载体预留内存：

1. **`DT_INT16` (Key 5)**：计算中转为 `float`，需将其切分参数及 `COPY_SPACE_MULTIPLE` 视作 `float` 处理，保证单次 Tile 计算时有充足的 `4 Bytes` 中转空间。
2. **`DT_INT8` (Key 7) / `DT_UINT8` (Key 8)**：计算中转为 `half`，需将其切分参数及 `COPY_SPACE_MULTIPLE` 视作 `half` 处理，保证单次 Tile 计算时有充足的 `2 Bytes` 中转空间。

#### 3.3.2 Kernel 侧设计

**3.3.2.1 新增 TilingKey 分支**
在 `foreach_expm1.cpp` 中，补充基于上述架构决策的分发逻辑：

```cpp
// ... 前置的 FLOAT16(1), FLOAT32(2) 分支 ...
#if __CCE_AICORE__ >= 220 && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    else if (TILING_KEY_IS(4)) {
        ForeachImplictOutput<bfloat16_t, float, Expm1Adapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        // int16_t 提升至 float 计算
        ForeachImplictOutput<int16_t, float, Expm1Adapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        // int8_t 提升至 half 计算
        ForeachImplictOutput<int8_t, half, Expm1Adapter<half>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        // uint8_t 提升至 half 计算
        ForeachImplictOutput<uint8_t, half, Expm1Adapter<half>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
#endif

```

**3.3.2.2 扩展 `InnerComputer` 与底层头文件支持**

1. **`foreach_implict_output.h`**：仿照 `bfloat16_t`，补充针对 `int16_t -> float`、`int8_t -> half`、`uint8_t -> half` 的模板特化，利用双向 `Cast` 隔离原始数据与 `Expm1Adapter` 的计算环境。
2. **`kernel_foreach_unary.h` / `kernel_foreach_base.h**`：在基类的类型萃取中（如定义 `TT` 时），需补充新的逻辑，即当判定到 `int16_t` 时分配 `float` 缓冲，判定到 `int8_t/uint8_t` 时分配 `half` 缓冲，并正确启用 `COPY_SPACE_MULTIPLE`。

#### 3.3.3 Ascend C 侧运行流程图

![image.png](https://cdn-img.gitcode.com/be/ca/b4eca6825b3698e348ff526db928f26036318c95feaff7b693ff57ef22187fe0.png 'image.png')

### 3.4 支持硬件

* **Atlas A2 训练系列产品 / Atlas A3 系列产品**（严格限定对应 `ascend910_93` 及 `ascend910b`）。

### 3.5 算子约束与异常拦截机制

1. **构建拦截规则（严格防线）**：如果前端尝试下发未注册的类型组合，构建流程会立刻阻断并报错，**不会触发向 CPU 执行的回退（Fallback）机制**。因此要求 Host 侧配置与 Kernel 侧模板的实例化必须保持严格一致。
2. **泛化要求**：输入 `x` 与输出 `y` 在多维度（0~8维）、极值边界以及数据类型之间必须严格匹配，并支撑框架级的各种泛化 Shape 传入。

## 四、 验收及可维可测标准

### 4.1 精度与性能标准

* **精度标准**：满足 `AscendOpTest` 的默认阈值。
* **性能标准**：无。

## 五、 变更文件清单

| 文件路径 | 变更类型 | 变更说明 |
| --- | --- | --- |
| `op_host/foreach_expm1_def.cpp` | 修改 | 原型扩展新增 `INT16/INT8/UINT8`；明确硬件配置仅含 `ascend910_93` 和 `ascend910b`。 |
| `op_host/config/.../foreach_expm1_binary.json` | 修改 | 为对应架构补充 `INT16/INT8/UINT8` 的编译条目。 |
| `op_kernel/foreach_expm1.cpp` | 修改 | 引入针对 `TILING_KEY_IS(5)`, `(7)`, `(8)` 的分发；5 使用 float 适配，7 和 8 使用 half 适配。 |
| `../foreach_utils/op_host/foreach_tiling_func.cpp` | 修改 | 更新 UB 切分逻辑：`int16_t` 基于 float（4 Bytes）分配；`int8_t/uint8_t` 基于 half（2 Bytes）分配。 |
| `../foreach_utils/op_kernel/foreach_implict_output.h` | 修改 | 补充 `int16_t -> float` 及 `8-bit -> half` 的 `InnerComputer` 模板特化计算流。 |
| `../foreach_utils/op_kernel/kernel_foreach_base.h` & `kernel_foreach_unary.h` | 修改 | 补充底层基础模板对这三种整型的识别萃取及 `COPY_SPACE_MULTIPLE` 控制。 |
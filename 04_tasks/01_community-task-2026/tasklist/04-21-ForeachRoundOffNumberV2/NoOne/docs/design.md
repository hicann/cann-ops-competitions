# aclnnForeachRoundOffNumberV2 算子设计文档

## 一、 需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献的需求，补充完善 Ascend C 算子库。为现有的 `aclnnForeachRoundOffNumberV2` 算子增加对 `DT_INT16` 数据类型的支持，以满足网络量化及端侧推理场景对底层核心算子的完备性要求。

### 1.2 背景介绍

`ForeachRoundOffNumberV2` 算子用于对张量列表（Tensor List）执行逐元素的舍入操作（如向上取整、向下取整、四舍五入、截断等）。
根据最新算子原型定义，当前算子已支持 `FLOAT16`, `FLOAT32`, `BFLOAT16`。本项目要求在现有代码架构下，新增对 `INT16` 数据的支持。

> **核心设计洞察（语义优化）**：
> 对于浮点数，`Round` 语义需要通过特定的硬件指令截取整数部分；但**对于整型（如 `int16_t`），其本身已是整数，不包含小数部分。** 因此：
> 1. 在 `CAST_FLOOR`, `CAST_CEIL`, `CAST_ROUND`, `CAST_TRUNC`, `CAST_RINT` 语义下，**结果等于自身**。
> 2. 在 `CAST_FRAC`（取小数部分）语义下，**结果恒为 0**。
> 
> 
> 基于上述数学特性，针对 `INT16` 输入，我们**完全可以免除 V 单元昂贵的 `Cast -> Convert -> Cast` 数学计算过程，只进行纯粹的 DataCopy 搬运（或 Duplicate 清零）**。这将带来极大的性能收益。

## 二、 需求分析

### 2.1 外部组件依赖

* 依赖 Ascend C 的 `DataCopy` 与 `Duplicate` API，用于整型的极速搬运与清零。
* 依赖原有的 `ForeachRoundOffNumberND` 隐式输出框架。

### 2.2 内部适配模块

| 模块 | 变更项 | 说明 |
| --- | --- | --- |
| `op_host` | `foreach_round_off_number_def.cpp` | 扩展原型 `tensor_dtype_list`，新增 `DT_INT16`。更新硬件配置，仅保留 `ascend910_93` 和 `ascend910b`。 |
| `op_host` | `foreach_tiling_func.cpp` | **修改 UB 切分策略**：由于无需转为 `float` 计算，`int16_t` 仅需基于自身（2 Bytes）预留内存，无需 `COPY_SPACE_MULTIPLE` 放大。 |
| `op_kernel` | `foreach_round_off_number.cpp` | 新增 `TILING_KEY_IS(5)` 路由分支，指向 `int16_t` 特化处理。 |
| `op_kernel` | `foreach_round_off_number.h` | 扩展 `InnerComputer` 模板特化，新增 `int16_t` 的“零计算”搬运流水线。 |

## 三、 需求详细设计

### 3.1 使能方式

| 上层框架 | 涉及的框架勾选 |
| --- | --- |
| TF训练/推理 |  |
| Pytorch训练/推理 |  |
| ATC推理 |  |
| Aclnn直调 | ✅ |

### 3.2 算子原型设计

修改 `op_host/foreach_round_off_number_def.cpp`，同步更新数据类型并严格限制硬件架构：

```cpp
explicit ForeachRoundOffNumber(const char* name) : OpDef(name)
{
    // 输入/输出张量支持的 dtype 列表新增整型
    std::vector<ge::DataType> tensor_dtype_list = {
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT16 // 新增
    };
    std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
    
    // RoundMode 的标量输入，需要和 tensor 列表数量对齐
    std::vector<ge::DataType> scalarDtypeList(tensor_dtype_list.size(), ge::DT_INT8);

    this->Input("x")
        .ParamType(DYNAMIC)
        .DataType(tensor_dtype_list)
        .Format(format_list)
        .UnknownShapeFormat(format_list)
        .AutoContiguous();
        
    this->Input("roundMode")
        .ParamType(REQUIRED)
        .DataType(scalarDtypeList)
        .Format(format_list)
        .UnknownShapeFormat(format_list);
        
    this->Output("y")
        .ParamType(DYNAMIC)
        .DataType(tensor_dtype_list)
        .Format(format_list)
        .UnknownShapeFormat(format_list)
        .AutoContiguous();

    // 仅保留 A2/A3 对应的系列
    this->AICore().AddConfig("ascend910_93");
    this->AICore().AddConfig("ascend910b");
}

```

### 3.3 详细设计

#### 3.3.1 Host 侧设计：UB 切分策略优化

在 `foreach_tiling_func.cpp` 中：

* **浮点数 (`bfloat16`, `half`)**：计算时需转化为 `float32`，占据 4 Bytes，因此需要开启 `COPY_SPACE_MULTIPLE`（预留多倍转换缓冲）。
* **整型 (`int16_t`)**：采用**免计算只搬运**策略，无需任何中间转换缓存。Host 侧计算 `ub_size` 和 `tileDataNum` 时，直接按照 2 Bytes 满额切分。这使得 `int16_t` 的单次搬运数据量 (DataNum) 大幅增加，进一步提升性能。

#### 3.3.2 Kernel 侧设计

**3.3.2.1 新增 TilingKey 与 Init 拦截**
在 `foreach_round_off_number.cpp` 中新增入口：

```cpp
#if __CCE_AICORE__ >= 220
    else if (TILING_KEY_IS(5)) {
        ForeachRoundOffNumberND<int16_t> op;
        op.Init(x, roundMode, y, userWS, &tilingData);
        op.Process();
    }
#endif

```

在 `foreach_round_off_number.h` 的 `Init` 流程中，补充 `int16_t` 的纯净 Buffer 初始化：

```cpp
    } else if (std::is_same<T, int16_t>::value) {
        // 无需预留 COPY_SPACE_MULTIPLE，无需 float32Queue
        pipe.InitBuffer(dataQueue, BUFFER_NUM, inputsTensorUbSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, inputsTensorUbSize);
        maxDataCount = inputsTensorUbSize / sizeof(T);
    }

```

**3.3.2.2 扩展 `InnerComputer` 模板特化（免计算策略）**
在 `foreach_round_off_number.h` 中，针对 `int16_t` 补充特化。对于整型，所有的标准舍入都是恒等映射。

```cpp
template <>
class InnerComputer<int16_t>
{
public:
    __aicore__ inline void Compute(
        LocalTensor<int16_t>& dataLocal, LocalTensor<int16_t>& outLocal, LocalTensor<float>& float32Tensor,
        int8_t roundModeValue, uint32_t maxCastDataCount, int64_t dataCount)
    {
        if (roundModeValue == CAST_FRAC) {
            // 整型没有小数部分，直接输出全 0
            Duplicate(outLocal, static_cast<int16_t>(0), dataCount);
        } else {
            // CEIL, FLOOR, ROUND, TRUNC, RINT, NONE
            // 结果等于自身，完全规避 V 单元算数计算，直接 UB-UB 搬运
            DataCopy(outLocal, dataLocal, dataCount);
        }
    }
};

```

#### 3.3.3 Ascend C 侧运行流程图

![image.png](https://raw.gitcode.com/user-images/assets/9516645/29a0f62c-ac9c-4209-a4b7-3891e6a215b2/image.png 'image.png')

### 3.4 支持硬件

* **Atlas A2 训练系列产品 / Atlas A3 系列产品** (对应 `ascend910_93` 及 `ascend910b`)。

### 3.5 算子约束与异常拦截机制

1. **构建阻断拦截**：编译期与运行期如果遇到未注册的 dtype，将直接构建报错并失败，**不会触发向 CPU 执行的回退机制**。
2. **泛化要求**：对于 `INT16` 输入，`y` 的输出也必须为 `INT16`，尺寸保持一致，能正常处理多 Tensor 列表输入。

## 四、 验收及可维可测标准

### 4.1 精度与性能标准

* **精度标准**：
* `roundMode != CAST_FRAC`：输出需完全等于输入，0 误差。
* `roundMode == CAST_FRAC`：输出需全为 `0`。


* **性能标准**：无。

## 五、 变更文件清单

| 文件路径 | 变更类型 | 变更说明 |
| --- | --- | --- |
| `op_host/foreach_round_off_number_def.cpp` | 修改 | 原型扩展新增 `INT16`，自动对齐 scalar 类型；明确硬件配置仅含 `ascend910_93` 和 `ascend910b`。 |
| `op_host/config/.../foreach_round_off_number_binary.json` | 修改 | 为对应硬件架构补充 `INT16` 的编译条目。 |
| `op_kernel/foreach_round_off_number.cpp` | 修改 | 引入针对 `TILING_KEY_IS(5)` 的专属入口分发。 |
| `op_kernel/foreach_round_off_number.h` | 修改 | 1. `Init` / `Process` 中对 `int16_t` 免除 `float32Queue` 内存分配。<br>

<br>2. 补充 `InnerComputer<int16_t>` 模板特化计算流，实现纯搬运/清零策略。 |
| `../foreach_utils/op_host/foreach_tiling_func.cpp` | 修改 | 调整 UB 切分策略：对于 `INT16` 放弃使用 `COPY_SPACE_MULTIPLE`，直接满额利用 UB。 |
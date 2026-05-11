# ForeachExp 算子设计方案

## 一、基本信息

### 1.1 需求来源
`ForeachExp` 算子是深度学习框架对应的昇腾算子实现，核心语义为：对输入张量列表（Tensor List）中的每个张量逐元素执行 $y = e^x$ 运算，返回结果张量列表。
当前 `aclnnForeachExp` 算子开源仓（`ops-nn`）仅支持浮点数据类型。为满足更广泛的端侧推理及量化场景需求，本方案在现有代码基础上再开发，新增 `INT16`（`int16_t`）、`INT8`（`int8_t`）、`UINT8`（`uint8_t`）三种整数数据类型的支持，完成算子泛化适配，并最终合入昇腾算子开源仓。

### 1.2 背景介绍

#### 1.2.1 现有 ForeachExp 算子实现分析
现有 `ForeachExp` 算子的核心底层逻辑复用了 `foreach_utils` 目录下的 `ForeachImplictOutput` 模板。
现有实现路径：[https://gitcode.com/cann/ops-nn/tree/master/foreach/foreach_exp](https://gitcode.com/cann/ops-nn/tree/master/foreach/foreach_exp)

#### 1.2.2 现有版本支持能力
| 参数 | 参数含义 | 数据类型支持 | 数据类型约束 | 形状 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入张量列表 | FLOAT32, FLOAT16, BFLOAT16 | 列表中所有 Tensor 数据类型一致 | 0-8 维 ND |
| y | 输出张量列表 | 与 x 一致 | shape size $\ge$ x 的 shape size | 0-8 维 ND |

#### 1.2.3 核心计算逻辑
计算公式：
$x = [x_0, x_1, ... x_{n-1}]$
$y = [y_0, y_1, ... y_{n-1}]$
$y_i = e^{x_i} \quad (i=0,1,...n-1)$

**Kernel 侧计算特性：**
当前底层通过传入 `Exp` 操作符给 `ForeachImplictOutput` 模板进行实例化。对于 `half` 和 `float`，直接在 V 单元执行 `Exp` 指令；对于 `bfloat16_t`，框架内部已经通过 `InnerComputer` 特化实现了 `Cast` (提升至 float32) $\rightarrow$ `Exp` $\rightarrow$ `Cast` (还原回 bfloat16_t) 的流程。新增的整数类型也将完全复用这一 Cast 特化机制。

---

## 二、需求分析

### 2.1 外部组件依赖
*   依赖 Ascend C 的 `Cast` API 提供 $int8_t \leftrightarrow float$、$uint8_t \leftrightarrow float$、$int16_t \leftrightarrow float$ 的类型转换及饱和截断支持。
*   依赖 `foreach_utils` 中的 `ForeachImplictOutput` 模板类及 `ForeachCommonTiling` 分核策略。

### 2.2 内部适配模块

| 模块 | 变更项 | 说明 |
| :--- | :--- | :--- |
| op_host | `foreach_exp_def.cpp` | 扩展 `tensor_dtype_list`，新增 DT_INT16 / DT_INT8 / DT_UINT8 |
| op_host | `foreach_exp_binary.json` | 在对应架构下新增三种整数类型的 JSON 配置条目 |
| op_kernel | `foreach_exp.cpp` | 新增 `TILING_KEY_INT16(5)` / `INT8(7)` / `UINT8(8)` 分支 |
| op_kernel | `foreach_implict_output.h` | 扩展 `InnerComputer` 对 int16_t, int8_t, uint8_t 转换 float 的模板特化 |
| docs/README | 文档与 README | 更新支持的数据类型列表和数值截断说明 |
| tests/examples | 测试框架 | 新增泛化场景的整数类型单元测试与 API 测试 |

---

## 三、需求模块设计

### 3.1 使能方式

| 上层框架 | 涉及的框架勾选 |
| :--- | :--- |
| TF训练/推理 | |
| Pytorch训练/推理 | |
| ATC推理 | |
| Aclnn直调 | ✅ |
| OPAT调优 | |
| SGAT子图切分 | |

### 3.2 算子原型定义（变更后）
原型注册文件中，新增对目标数据类型的支持：

```cpp
std::vector<ge::DataType> tensor_dtype_list = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
    ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8
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
this->AICore().AddConfig("ascend910_93");
this->AICore().AddConfig("ascend910b");
```

### 3.3 详细设计

#### 3.3.1 Host 侧设计
*   **分核策略**：继续沿用 `ForeachCommonTiling`，根据 Tensor 数量及总元素个数进行 `usedCoreNum` 计算与负载均衡切分，无需做逻辑改动。
*   **UB 内存分布与 TilingKey**：框架已在 `common_dtype.h` 中预定义了枚举，新增类型映射为：`TILING_KEY_INT16=5`，`TILING_KEY_INT8=7`，`TILING_KEY_UINT8=8`。

#### 3.3.2 Kernel 侧设计

**3.3.2.1 新增 TilingKey 分支**
在 `foreach_exp.cpp` 主干中补充路由分支：

```cpp
// ... 前置的 FLOAT16, FLOAT32, BFLOAT16 分支 ...
} else if (TILING_KEY_IS(5)) {
    ForeachImplictOutput<int16_t, float, Exp, 2, 1> op;
    op.Init(x, y, userWS, &tilingData);
    op.Process();
} else if (TILING_KEY_IS(7)) {
    ForeachImplictOutput<int8_t, float, Exp, 2, 1> op;
    op.Init(x, y, userWS, &tilingData);
    op.Process();
} else if (TILING_KEY_IS(8)) {
    ForeachImplictOutput<uint8_t, float, Exp, 2, 1> op;
    op.Init(x, y, userWS, &tilingData);
    op.Process();
}
```

**3.3.2.2 扩展 `InnerComputer` 模板特化**
在 `foreach_implict_output.h` 中，为了让整数类型能够调用浮点 `Exp` 算子，需仿照 `bfloat16_t` 补充三组特化。以下以 `int16_t` 为例：

```cpp
#if __CCE_AICORE__ >= 220
template <ImplictOutputOp<float>* op, uint8_t paramsCount>
class InnerComputer<int16_t, float, op, paramsCount>
{
public:
    __aicore__ inline void Compute(
        LocalTensor<int16_t>& dataLocal, LocalTensor<float>& float32Tensor, uint32_t maxCastDataCount,
        int64_t dataCount)
    {
        uint32_t castTimes = dataCount / maxCastDataCount;
        uint32_t castTimesRemainder = dataCount % maxCastDataCount;

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, i, maxCastDataCount);
        }
        if (castTimesRemainder > 0) {
            ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, castTimes, castTimesRemainder);
        }
    }

private:
    __aicore__ inline void ComputePerCast(
        LocalTensor<int16_t>& dataLocal, LocalTensor<float>& float32Tensor, uint32_t maxCastDataCount,
        uint32_t index, int64_t dataCount)
    {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
        op(float32Tensor[offset], float32Tensor, dataCount); // 核心计算：Exp
        PipeBarrier<PIPE_V>();
        // 注意：此处转回整型使用 CAST_RINT（四舍五入）或依赖默认的饱和截断行为
        Cast(dataLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
};
#endif
// 同理，补充 <int8_t, float, ...> 和 <uint8_t, float, ...> 的特化
```

**3.3.2.3 数值溢出与截断行为**
由于 $e^x$ 增长极快（例如 $x=10$ 时，结果约为 $22026.46$），该数值已远超 `INT8` 和 `UINT8` 的表示范围。在最终执行 `Cast(float32 -> int)` 时，硬件底层触发饱和截断：超出数据类型最大值的结果将饱和为该类型的最大值（如 `INT8` 饱和为 127，`UINT8` 饱和为 255）。

**3.3.2.4 Ascend C 实现流程图 (PlantUML)**

![image.png](https://raw.gitcode.com/user-images/assets/9516645/ca94b786-0a41-431f-97e8-0fe7d2d5d476/image.png 'image.png')

### 3.4 支持硬件
*   Atlas A2 训练系列产品 / Atlas 800I A2 推理产品
*   Atlas A3 系列产品

### 3.5 算子约束限制
1.  **数据类型对齐强制性约束**：需要特别注意的是，在算子构建阶段，如果遇到未注册的数据类型（如框架侧下发了未支持的 dtype），CANN 软件栈的预期行为是直接构建报错并停止运行，而不会触发回退机制将算子标记为 CPU 执行。因此，Host 侧的原型注册必须与 Kernel 侧的类型分发（TilingKey 分支及 Template 特化）严格对齐，不可遗漏。
2.  **输入输出类型一致性**：输入 `x` 张量列表内所有元素的数据类型必须完全一致，且与输出 `y` 张量列表的数据类型保持绝对一致。

---

## 四、验收标准

| 验收标准 | 描述 |
| :--- | :--- |
| **功能标准** | 算子能够正确处理 INT16、INT8、UINT8 输入的 `Exp` 运算，能够正常处理多 Tensor 列表输入输出。 |
| **精度标准** | 严格符合 AscendOpTest 默认阈值。由于是整数运算，基准测试逻辑需设定为：输入转换为 `float64` -> 计算 `np.exp` -> `np.round` -> 转换为对应整数类型（并验证溢出时的饱和截断行为是否与 CPU 侧对齐）。 |
| **泛化标准** | 支撑 0-8 维、不同 Tensor 大小组合及空张量的合法输入场景。 |

---

## 五、可维可测分析

### 5.1 精度/性能标准
*   **精度**：按照 4.0 验收标准执行，重点通过 UT 验证 $x \le 0$ 时是否下溢截断正确，以及 $x$ 较大时饱和为该类型最大正数是否正确。
*   **性能**：新增分支性能须对齐已有的 `BFLOAT16` 逻辑，确保流水线中 `MTE2 -> V -> V -> V -> MTE3` 的并行度未被破坏，打满系统带宽或 V 单元算力极限。

### 5.2 兼容性分析
该开发任务为纯粹的类型扩展，向后兼容原有框架的所有特性，不改变现有浮点类型的计算逻辑与 Tiling 策略，对存量业务零影响。

---

## 六、版本与仓库信息
*   **上游开源仓地址**：[https://gitcode.com/cann/ops-nn](https://gitcode.com/cann/ops-nn)
*   **目标修改目录**：[https://gitcode.com/cann/ops-nn/tree/master/foreach/foreach_exp](https://gitcode.com/cann/ops-nn/tree/master/foreach/foreach_exp) 及 `foreach_utils`
*   **开发语言**：Ascend C

## 七、变更文件清单

| 序号 | 文件路径 | 变更类型 | 变更说明 |
| :--- | :--- | :--- | :--- |
| 1 | `op_host/foreach_exp_def.cpp` | 修改 | `tensor_dtype_list` 新增 INT16/INT8/UINT8 |
| 2 | `op_kernel/foreach_exp.cpp` | 修改 | 新增 TILING_KEY_IS(5/7/8) 路由分支 |
| 3 | `../foreach_utils/op_kernel/foreach_implict_output.h` | 修改 | 新增 `InnerComputer` 针对三种整型转 `float` 的模板特化逻辑 |
| 4 | `op_host/config/ascend910b/foreach_exp_binary.json` | 修改 | 新增三种整数类型支持配置 |
| 5 | `op_host/config/ascend910_93/foreach_exp_binary.json` | 修改 | 新增三种整数类型支持配置 |
| 6 | `README.md` & `docs/aclnnForeachExp.md` | 修改 | 更新支持的数据类型与整数数值边界行为 |
| 7 | `examples/test_aclnn_foreach_exp.cpp` | 修改 | 新增整数类型的单测及边界测试 |
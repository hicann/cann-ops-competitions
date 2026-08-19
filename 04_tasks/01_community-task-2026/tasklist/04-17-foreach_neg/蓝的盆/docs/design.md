# aclnnForeachNeg 算子设计文档

## 一、 需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献，补充完善 Ascend C 算子库。为现有 `aclnnForeachNeg` 算子增加对 `DT_INT16`、`DT_INT8`、`DT_UINT8` 数据类型的支持，以满足量化模型、端侧推理及泛化输入场景对 foreach 系列基础算子的完备性要求。

### 1.2 背景介绍

`ForeachNeg` 算子用于对输入张量列表中的每个 Tensor 执行逐元素取相反数运算。其数学公式为：

```text
y_i = -x_i, i = 0, 1, ..., n - 1
```

当前算子已支持 `FLOAT16`、`FLOAT32`、`INT32`、`BFLOAT16` 数据类型。本需求在现有一元 foreach 框架下新增 `INT16`、`INT8`、`UINT8` 支持，并要求满足泛化 shape、空 TensorList、非连续 Tensor、不同 TensorList 长度及多种合法数据分布下的计算需求。

`ForeachNeg` 为一元算子，新增整型数据需要先提升到可计算类型完成取负，再 Cast 回原始输出类型：

- `INT16`：Cast 到 `float32` 计算，再 Cast 回 `int16_t`。
- `INT8`、`UINT8`：Cast 到 `half` 计算，再 Cast 回 `int8_t`/`uint8_t`。AICore 上 8-bit 整型不直接 Cast 到 `float32`，需先走 `half` 中转。

## 二、 需求分析

### 2.1 外部组件依赖

- 依赖 Ascend C `Cast` API 支持低精度整型与中间计算类型之间的转换。
- 依赖 Ascend C Level0 二元指令 `Sub`，通过 `0 - x` 实现取相反数。
- 依赖现有 foreach 公共 tiling、infer shape、contiguous 及 unary kernel 模板框架。

### 2.2 内部适配模块

| 模块 | 变更项 | 说明 |
| --- | --- | --- |
| `op_graph` | `foreach_neg_proto.h` | 扩展 `x/y` 的 IR 原型声明，新增 `DT_INT16/DT_INT8/DT_UINT8`。 |
| `op_host` | `foreach_neg_def.cpp` | 扩展 `tensor_dtype_list`，在 A2/A3 等主配置中新增三种 dtype；Kirin 配置保持原有支持范围。 |
| `op_host/config` | `foreach_neg_binary.json` | 为新增 dtype 补充二进制编译条目，确保动态编译和预编译配置闭环一致。 |
| `op_host` | `foreach_tiling_class.h` | 在 `SOLO_NEG_OP_CODE` 的 UB 切分中识别需要 Cast 计算的 dtype，按中间计算类型预留临时 UB。 |
| `op_kernel` | `foreach_neg.cpp` | 新增 `TILING_KEY_IS(5/7/8)` 分支，分别实例化 `INT16/INT8/UINT8` 的计算模板。 |
| `op_kernel` | `foreach_implict_output_level_zero_api.h` | 补充低精度整型的 Cast 计算路径，支持 `INT16 -> float` 和 `INT8/UINT8 -> half`。 |
| `docs/README` | `README.md`、`docs/aclnnForeachNeg.md` | 补充 `DT_INT16/DT_INT8/DT_UINT8` 的支持情况和约束说明。 |
| `examples/tests` | 示例与用例 | 增加新增 dtype 的泛化输入验证，覆盖大 shape、非 32B 对齐尾块、空 TensorList 等场景。 |

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
| **x** | 输入 | FLOAT16、FLOAT32、INT32、BFLOAT16、**INT16、INT8、UINT8** | ND | 0-8 维 | 输入张量列表，支持空 TensorList；列表中所有 Tensor 数据类型保持一致 |
| **y** | 输出 | FLOAT16、FLOAT32、INT32、BFLOAT16、**INT16、INT8、UINT8** | ND | 与 `x` 对应 Tensor 一致 | 输出张量列表，数据类型和数据格式与 `x` 一致，计算结果为 $y_i = -x_i$ |

修改 `foreach/foreach_neg/op_graph/foreach_neg_proto.h` 和 `foreach/foreach_neg/op_host/foreach_neg_def.cpp`，将 `x`、`y` 的数据类型范围由：

```cpp
DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16
```

扩展为：

```cpp
DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_INT16, DT_INT8, DT_UINT8
```

硬件配置说明：

- `ascend910_93`、`ascend910b`：支持新增 `INT16/INT8/UINT8`。
- `kirinx90`、`kirin9030`：保持原有支持范围，不扩展 `BFLOAT16/INT16/INT8/UINT8`。
- 其他未声明支持的硬件维持原有行为，不额外放开 dtype。

### 3.3 详细设计

#### 3.3.1 Host 侧设计：UB 切分策略

`ForeachNeg` 复用 foreach 公共 tiling，入口 opCode 为 `SOLO_NEG_OP_CODE`。新增低精度类型后，Host 侧不能只按输入 dtype 的字节数切分 UB，而需要按中间计算类型为临时 Cast 缓存预留空间：

1. `DT_INT16`：中间计算类型为 `float32`，临时缓存按 4 字节元素预留。
2. `DT_INT8/DT_UINT8`：中间计算类型为 `half`，临时缓存按 2 字节元素预留。
3. `DT_BF16`：保持已有 `BF16 -> float32 -> BF16` 路径。

伪代码如下：

```cpp
bool NeedCastCompute(DataType dtype)
{
    return dtype == DT_BF16 || dtype == DT_INT16 || dtype == DT_INT8 || dtype == DT_UINT8;
}

if (opCode == SOLO_NEG_OP_CODE) {
    uint32_t totalSize = ubSize - tilingDataSize;
    if (NeedCastCompute(dataType)) {
        totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
    }

    uint32_t canUseUbSize = totalSize / 2 - BYTE_PER_BLOCK;
    inputsTensorUbSize = NeedCastCompute(dataType) ?
        AlignDown(canUseUbSize, BYTE_BLOCK_FOR_BF16) :
        AlignDown(canUseUbSize, BYTE_BLOCK);
}
```

其中 `BYTE_BLOCK_FOR_BF16` 代表 64B 对齐策略，用于避免 Cast 计算路径在泛化 shape、尾块和非对齐数据量下出现 UB 未对齐或越界问题。

#### 3.3.2 Kernel 侧设计

`ForeachNeg` 当前通过 `ForeachImplictOutputLevelZeroApi<T, P, Sub>` 实现，即将标量 0 复制到 UB 后执行 `0 - x`。新增低精度类型时继续复用该框架，只扩展模板实例和 Cast 计算路径。

新增 kernel 分支伪代码：

```cpp
if (TILING_KEY_IS(1)) {
    ForeachImplictOutputLevelZeroApi<half, half, Sub, 2, 1> op;
    op.Init(x, y, userWS, &tilingData, half(0));
    op.Process();
} else if (TILING_KEY_IS(2)) {
    ForeachImplictOutputLevelZeroApi<float, float, Sub, 2, 1> op;
    op.Init(x, y, userWS, &tilingData, float(0));
    op.Process();
} else if (TILING_KEY_IS(3)) {
    ForeachImplictOutputLevelZeroApi<int32_t, int32_t, Sub, 2, 1> op;
    op.Init(x, y, userWS, &tilingData, int32_t(0));
    op.Process();
}

#if __CCE_AICORE__ >= 220
if (TILING_KEY_IS(4)) {
    ForeachImplictOutputLevelZeroApi<bfloat16_t, float, Sub, 2, 1> op;
    op.Init(x, y, userWS, &tilingData, float(0));
    op.Process();
} else if (TILING_KEY_IS(5)) {
    ForeachImplictOutputLevelZeroApi<int16_t, float, Sub, 2, 1> op;
    op.Init(x, y, userWS, &tilingData, float(0));
    op.Process();
} else if (TILING_KEY_IS(7)) {
    ForeachImplictOutputLevelZeroApi<int8_t, half, Sub, 2, 1> op;
    op.Init(x, y, userWS, &tilingData, half(0));
    op.Process();
} else if (TILING_KEY_IS(8)) {
    ForeachImplictOutputLevelZeroApi<uint8_t, half, Sub, 2, 1> op;
    op.Init(x, y, userWS, &tilingData, half(0));
    op.Process();
}
#endif
```

#### 3.3.3 公共模板设计

在 `foreach_implict_output_level_zero_api.h` 中补充低精度整型特化。核心流程如下：

```cpp
template <typename T, typename ComputeT>
void CastNegAndCastBack(LocalTensor<T> dataLocal,
                        LocalTensor<float> tmpBuffer,
                        LocalTensor<ComputeT> zeroBlock,
                        int64_t dataCount)
{
    LocalTensor<ComputeT> computeTensor = GetComputeTensor<ComputeT>(tmpBuffer);

    Cast(computeTensor, dataLocal, CAST_NONE, dataCount);
    Sub(computeTensor, zeroBlock, computeTensor, dataCount);
    Cast(dataLocal, computeTensor, CAST_RINT, dataCount);
}
```

特化关系：

- `InnerComputer<int16_t, float, Sub>`：复用 `float32Tensor` 作为 float 临时缓存。
- `InnerComputer<int8_t, half, Sub>`：将 `float32Tensor` 按 half 重新解释为临时缓存，完成 8-bit 到 half 的 Cast 计算。
- `InnerComputer<uint8_t, half, Sub>`：同 `int8_t` 路径，Cast 回 `uint8_t` 时按 Ascend C Cast 规则处理越界值。

#### 3.3.4 Ascend C 侧运行流程图
![neg.png](https://raw.gitcode.com/user-images/assets/9516645/394a25b5-af98-4180-9a70-0f9dd8cbb246/neg.png 'neg.png')

### 3.4 支持硬件

- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

Kirin X90/Kirin 9030 处理器系列产品保持原有类型支持，不新增 `BFLOAT16/INT16/INT8/UINT8`。

### 3.5 算子约束与异常拦截机制

1. `x`、`y` 均为 TensorList，支持空 TensorList。
2. `x` 内部所有 Tensor 数据类型保持一致，`y` 与 `x` 的数据类型和数据格式保持一致。
3. `x` 与 `y` 对应 Tensor 的 shape 需要满足输出 shape size 大于等于输入 shape size。
4. 输入维度范围为 0 到 8 维。
5. 不支持的数据类型、私有格式或非法 shape 在 aclnn 参数检查阶段拦截。
6. 构建配置、原型声明、tiling key、kernel 模板实例必须保持一致，避免出现原型已声明但 kernel 无实例导致的编译或运行失败。

## 四、 验收及可维可测标准

### 4.1 精度与性能标准

精度标准：

- `FLOAT16/FLOAT32/INT32/BFLOAT16`：保持原有精度标准。
- `INT16`：参考计算为 `float32(-x)`，再 Cast 回 `int16_t`，按 Ascend C Cast 规则处理舍入和饱和。
- `INT8`：参考计算为 `half(-x)`，再 Cast 回 `int8_t`。
- `UINT8`：参考计算为 `half(-x)`，再 Cast 回 `uint8_t`，负值结果按 Cast 到无符号整型的边界规则处理。

泛化测试标准：

- 覆盖 `DT_INT16/DT_INT8/DT_UINT8` 三种新增 dtype。
- 覆盖空 TensorList、单 Tensor、多 Tensor、不同 shape、大 shape、非对齐尾块。
- 覆盖正数、负数、0、边界值和溢出风险值。
- 对比 CPU 或 NumPy 参考实现，参考实现需模拟 Cast 回目标 dtype 的舍入和饱和行为。

性能标准：

- 新增 dtype 允许存在 Cast 带来的额外开销，但不能出现 UB 越界、核间切分错误或异常回退。
- 大 shape 场景下应保持多核并行，吞吐表现接近已有 BF16 Cast 计算路径。

## 五、 变更文件清单

| 文件路径 | 变更类型 | 变更说明 |
| --- | --- | --- |
| `foreach/foreach_neg/op_graph/foreach_neg_proto.h` | 修改 | `x/y` 原型新增 `DT_INT16/DT_INT8/DT_UINT8`。 |
| `foreach/foreach_neg/op_host/foreach_neg_def.cpp` | 修改 | op_def 主配置新增三种 dtype，Kirin 配置保持不变。 |
| `foreach/foreach_neg/op_host/config/.../foreach_neg_binary.json` | 修改 | 补充新增 dtype 的二进制配置。 |
| `foreach/foreach_neg/op_kernel/foreach_neg.cpp` | 修改 | 新增 tiling key 5、7、8 的 kernel 路由。 |
| `foreach/foreach_utils/op_host/foreach_tiling_class.h` | 修改 | `SOLO_NEG_OP_CODE` 分支支持低精度 Cast 计算的 UB 预留和 64B 对齐。 |
| `foreach/foreach_utils/op_kernel/foreach_implict_output_level_zero_api.h` | 修改 | 补充 `INT16 -> float`、`INT8/UINT8 -> half` 的 Cast 计算特化。 |
| `foreach/foreach_neg/README.md` | 修改 | 补充新增 dtype 支持情况及 Kirin 约束。 |
| `foreach/foreach_neg/docs/aclnnForeachNeg.md` | 修改 | 补充 aclnn 文档中的新增 dtype、约束和返回说明。 |
| `foreach/foreach_neg/examples/test_aclnn_foreach_neg.cpp` | 修改 | 增加新增 dtype 示例和泛化 shape 验证。 |

# Mul 算子测试报告

## 1. 测试概述

本测试针对 CANN 数学库中的 **Mul（逐元素乘法）算子** 设计端到端测试用例，在 CPU 模拟器（ascend950）环境中运行。测试基于官方示例 `math/mul/examples/test_aclnn_mul.cpp` 进行深度扩展。

- **测试源文件**：`test_aclnn_mul.cpp`（提交时需重命名为 `test_aclnn_mul.cpp`）
- **编译环境**：Docker 镜像 `yeren666/cann-ops-test:v1.0`，CANN 9.0.0
- **硬件模拟**：`ascend950` CPU 模拟器
- **编译命令**：
  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
  ./build_out/cann-ops-math-custom_linux-x86_64.run --quiet
  bash build.sh --run_example mul eager cust --vendor_name=custom --simulator soc=ascend950 --cov
  ```

测试目标：
- **功能测试**：验证不同场景下，计算结果还有特殊数值处理
- **API路径覆盖**：aclnnMul，aclnnMuls，aclnnInplaceMul，aclnnInplaceMuls
- **覆盖率提升**：aclnn_mul.cpp，mul.cpp，mul_tiling_arch35.cpp
- **多维测试设计**：dtype，shape，broadcast，mixed precision，边界值，inplace


---

## 2. 测试策略与设计方法

### 2.1 测试架构

采用模块化思想，减少代码重复

| 函数模板 | 适用 API | 说明 |
|----------|----------|------|
| `RunGenericMulTest<T>` | `aclnnMul` | 同类型或混合类型 tensor × tensor |
| `RunGenericMulsTest<T>` | `aclnnMuls` | tensor × scalar |
| `RunGenericInplaceMulTest<T>` | `aclnnInplaceMul` | 原地 tensor × tensor |
| `RunGenericInplaceMulsTest<T>` | `aclnnInplaceMuls` | 原地 tensor × scalar |
| `RunGenericMixedMulTest<T1,T2,TOut>` | `aclnnMul` | 混合数据类型（输入类型不同） |
| `RunFloat16Test` | `aclnnMul` / `aclnnInplaceMuls` | 专门处理 float16 的二进制转换与比较 |
| `RunUltimateExceptionTests` | 所有 API | 异常输入（nullptr、不支持 dtype、shape 不匹配等） |
| `RunApiValidationTests` | `aclnnMul` | 特殊格式（FRACTAL_NZ）、输出类型不匹配等 |

每个通用函数内部执行以下步骤：
1. 分配主机数据、设备内存、创建 ACL tensor 描述符；
2. 调用 `GetWorkspaceSize` 和 `Execute` 两段式接口；
3. 将结果拷回主机并与 CPU 端计算的期望值进行比对；
4. 统一释放资源（`goto cleanup` 模式）。

### 2.2 结果验证方法

- **浮点类型**（float32, float16, double, bfloat16）：使用绝对容差 + 相对容差公式  
  \[
  |actual - expected| \leq atol + rtol \times |expected|
  \]
  容差取值：
  | 类型 | atol | rtol |
  |------|------|------|
  | float32 | 1e-5 | 1e-5 |
  | float16 | 1e-3 | 1e-3 |
  | double | 1e-9 | 1e-9 |
  | bfloat16 | 1e-3 | 1e-3 |

- **整数类型**（int8, int16, int32, int64, uint8, bool）：精确相等（atol=rtol=0）。

- **复数类型**（complex64）：分别比较实部和虚部，使用与 float32 相同的容差。

- **特殊值**：`AlmostEqual` 函数内置了对 `NaN` 和 `Inf` 的处理：两个 NaN 视为相等，同号 Inf 视为相等。


---

## 3. 测试覆盖维度

### 3.1 数据类型覆盖

根据 `op_host/mul_def.cpp` 中注册的 16 种合法组合，测试覆盖了以下数据类型（含同类型和混合类型）：

| 序号 | 数据类型 | 测试用例名称 | 说明 |
|------|----------|--------------|------|
| 1 | float32 | `mul_float32_basic` | 基础逐元素乘法 |
| 2 | int32 | `mul_int32_basic` | 整数乘法 |
| 3 | uint8 | `mul_uint8` | 无符号 8 位整数 |
| 4 | bool | `mul_bool` | 逻辑与（true*true=1，其余 0） |
| 5 | float16 | `float16_basic` / `float16_edge` | 专用转换函数，边界值测试 |
| 6 | double | `inplace_mul_double` | 触发软件模拟双精度乘法（200+ 行手写代码） |
| 7 | int8 | `mul_int8_saturation` | 饱和截断测试 |
| 8 | int16 | `mul_int16` | 16 位整数 |
| 9 | int64 | `mul_int64` | 64 位整数 |
| 10 | bfloat16 | `mul_bfloat16` | BF16 格式，专用转换函数 |
| 11 | complex64 | `mul_complex64` | 复数乘法（实部、虚部分别计算） |
| 12 | 混合类型 | `mul_mixed_fp16_fp32` | float16 × float32 → float32 |
| 13 | 混合类型 | `mul_mixed_fp32_fp16` | float32 × float16 → float32 |


### 3.2 Shape 与广播覆盖

| 测试用例 | 输入 Shape1 | 输入 Shape2 | 输出 Shape | 广播类型 |
|----------|-------------|-------------|------------|----------|
| `mul_float32_basic` | {4,2} | {4,2} | {4,2} | 无广播（同 shape） |
| `mul_int32_basic` | {2,2} | {2,2} | {2,2} | 无广播 |
| `mul_float32_broadcast` | {2,3} | {3} | {2,3} | 一维广播到二维 |
| `mul_complex_broadcast` | {2,1,4} | {1,3,4} | {2,3,4} | 高维广播 |
| `mul_1d_broadcast` | {2,3} | {1} | {2,3} | 标量广播（shape 为 {1}） |
| `mul_empty_tensor` | {2,0,3} | {2,0,3} | {2,0,3} | 空 tensor（元素数为 0） |
| `mul_0d_tensor` | {} | {} | {} | 0-D 标量 tensor |

### 3.3 API 变体覆盖

| API | 测试用例 |
|-----|----------|
| `aclnnMul` | 所有同类型、混合类型、广播、边界值、大 tensor 测试 |
| `aclnnMuls` | `muls_float32_basic`、`muls_int32_negative` |
| `aclnnInplaceMul` | `inplace_mul_double`、`inplace_mul_int8_broadcast` |
| `aclnnInplaceMuls` | `inplace_muls_float32_zero`、`inplace_muls_int32_negative`、`float16_inplacemuls` |

### 3.4 数值边界与特殊值

| 边界类型 | 测试用例 | 关键数据 |
|----------|----------|----------|
| 零值 | `inplace_muls_float32_zero` | 乘以 0.0f 应得 0.0 |
| 负数 | `muls_int32_negative` | -2 × (10, -5, 0, 100) = (-20, 10, 0, -200) |
| 极大/极小整数 | `mul_int32_limits` | INT32_MAX × 1 = INT32_MAX, INT32_MIN × 1 = INT32_MIN, 极值 × 0 = 0 |
| 浮点 NaN | `mul_float32_nan_inf` | NaN × 2.0 = NaN, 5.0 × NaN = NaN |
| 浮点 Inf | `mul_float32_nan_inf` | Inf × 2.0 = Inf, -Inf × 3.0 = -Inf |
| 饱和截断（uint8） | `mul_uint8_saturation` | 200 × 2 = 400 → 截断为 255 |
| 饱和截断（int8） | `mul_int8_saturation` | 100 × 2 = 200 → 截断为 127；-100 × 2 = -200 → 截断为 -128 |

### 3.5 异常输入与错误分支

| 异常场景 | 测试函数 | 预期行为 |
|----------|----------|----------|
| 不支持的数据类型 (UINT32) | `RunUltimateExceptionTests` | 返回错误码（拦截） |
| 非法整数混合类型 (INT8×INT32) | `RunUltimateExceptionTests` | 返回错误码 |
| 非法 Inplace 广播（other shape 大于 self） | `RunUltimateExceptionTests` | 返回错误码 |
| 输出 shape 不匹配 | `RunUltimateExceptionTests` | 返回错误码 |
| 非法 Inplace 类型提升 (INT32 ×= FLOAT32) | `RunUltimateExceptionTests` | 返回错误码 |
| `aclnnMuls` 传入非法 scalar 类型 (UINT32) | `RunUltimateExceptionTests` | 返回错误码 |
| `nullptr` 参数 | `RunUltimateExceptionTests` | 返回错误码 |
| 输出 dtype 不匹配（输入 float32，输出 int32） | `RunApiValidationTests` | 返回错误码 |
| 无法广播的 shape 组合 | `RunApiValidationTests` | 返回错误码 |
| 特殊数据格式 FRACTAL_NZ | `RunApiValidationTests` | 返回错误码或降级处理 |

### 3.6 非连续内存与特殊存储

- **非连续 strides**：通过 `CreateNonContiguousAclTensor` 构造 shape `{2,2}`、strides `{4,1}` 的 tensor，验证计算结果与连续内存一致（期望值 `{2,4,0,0,6,8,0,0}`）。
- **Inplace 非连续内存校验**：调用 `aclnnInplaceMulGetWorkspaceSize` 测试非连续 tensor 的原地操作，触发底层 stride 校验逻辑。

### 3.7 大 Tensor 与多块 Tiling

- 测试 `{256,256}`（65536 元素）和 `{128,1024}`（131072 元素）的大 tensor，触发多块切分（tiling）策略，覆盖 `mul_tiling_arch35.cpp` 中的循环分块逻辑。

---

## 4. 测试执行结果

### 4.1 测试通过情况

运行完整测试后，所有预期通过的测试用例均输出 `[PASS]`，异常测试输出 `[PASS]` 或 `[SKIP]`（当环境不支持某些特性时）。最终输出：

```
ALL TESTS PASSED
```

### 4.2 代码覆盖率

使用 `gcov -b` 统计三个评分目标文件，结果如下：

| 文件 | 行覆盖率 | 分支覆盖率 | 有效行数 |
|------|----------|------------|----------|
| `op_api/aclnn_mul.cpp` | **77.13%** | 36.93% | 328 |
| `op_api/mul.cpp` | **80.77%** | 62.96% | 52 |
| `op_host/arch35/mul_tiling_arch35.cpp` | **89.22%** | 58.33% | 102 |


### 4.3 未覆盖代码分析

通过查看 `.gcov` 文件，未覆盖的行主要集中在：

- `aclnn_mul.cpp`：
  - 部分不常见的混合类型组合（如 `BF16 × F32` 反向）未测试；
  - `ACL_COMPLEX128` 因模拟器不支持被跳过；
  - 某些错误处理分支（如内存分配失败）难以在正常测试中触发。
- `mul_tiling_arch35.cpp`：
  - `COMPLEX128` 和部分 `COMPLEX64` 的 tiling 策略未完全覆盖；
  - 部分极端的平台信息获取分支未触发。
- `mul.cpp`：
  - AICPU 回退路径（需要特定条件触发）未覆盖。



## 6. 总结

本测试对 Mul 算子的核心功能、数据类型、广播规则、API 变体、边界值、异常输入及非连续内存进行了系统性验证。所有测试用例均通过，三个目标文件行覆盖率均超过 77%。测试代码设计清晰、可维护性强。

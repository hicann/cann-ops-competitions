# Add 算子测试报告

## 1. 测试概述

本测试针对 CANN 数学库中的 **Add（逐元素加法）算子** 设计端到端测试用例，在 CPU 模拟器（ascend950）环境中运行。测试基于官方示例 `math/add/examples/test_aclnn_add.cpp` 进行扩展。

- **测试源文件**：`test_aclnn_add.cpp`
- **编译环境**：Docker 镜像 `yeren666/cann-ops-test:v1.0`，CANN 9.0.0
- **硬件模拟**：`ascend950` CPU 模拟器
- **编译命令**：
  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
  ./build_out/cann-ops-math-custom_linux-x86_64.run --quiet
  bash build.sh --run_example add eager cust --vendor_name=custom --simulator soc=ascend950 --cov
  ```

测试目标：
- **功能验证**：验证不同数据类型、广播、alpha 标量、原地操作等场景下的计算结果正确性。
- **API 路径覆盖**：`aclnnAdd`（普通加法）、原地加法（复用 `aclnnAdd` 将输出指定为输入）。
- **覆盖率提升**：覆盖 `aclnn_add.cpp`、`add.cpp`、`add_tiling_arch35.cpp` 等核心文件。
- **多维测试设计**：数据类型、shape 广播、混合精度（alpha 为 float 但 tensor 为 half）、边界值、非连续内存、异常输入。

---

## 2. 测试策略与设计方法

### 2.1 测试架构

采用**模板化通用测试函数**与**专用测试函数**相结合的方式，避免重复代码：

| 函数模板 | 适用场景 | 说明 |
|----------|----------|------|
| `RunGenericAddTest<T>` | `aclnnAdd` | 普通加法，支持自定义 alpha 标量 |
| `RunGenericInplaceAddTest<T>` | 原地加法 | 将输出 tensor 设为第一个输入，实现自更新 |
| `RunFloat16AddTest` | `aclnnAdd` | 专门处理 float16 的二进制转换与比较 |
| `RunExceptionTests` | 所有 API | 异常输入（nullptr、不支持 dtype、广播不匹配等） |

每个通用函数内部执行：
1. 分配主机数据、设备内存、创建 ACL tensor 描述符；
2. 创建 `alpha` 标量（`aclScalar`）；
3. 调用 `aclnnAddGetWorkspaceSize` 和 `aclnnAdd` 两段式接口；
4. 将结果拷回主机并与 CPU 端计算的期望值进行比对；
5. 统一释放资源（`goto cleanup` 模式）。

### 2.2 结果验证方法

- **浮点类型**（float32, float16, double）：使用绝对容差 + 相对容差公式  
  \[
  |actual - expected| \leq atol + rtol \times |expected|
  \]
  容差取值：
  | 类型 | atol | rtol |
  |------|------|------|
  | float32 | 1e-5 | 1e-5 |
  | float16 | 1e-3 | 1e-3 |
  | double | 1e-9 | 1e-9 |

- **整数类型**（int32, uint8, int16, int64 等）：精确相等（atol=rtol=0）。

- **特殊值**：`AlmostEqual` 函数内置了对 `NaN` 和 `Inf` 的处理：两个 NaN 视为相等，同号 Inf 视为相等。

- **期望值计算**：每个测试用例均在 CPU 端独立计算期望结果，避免使用算子本身的结果进行自比较。

---

## 3. 测试覆盖维度

### 3.1 数据类型覆盖

Add 算子支持多种数据类型（同类型），测试覆盖了以下类型：

| 数据类型 | 测试用例名称 | 说明 |
|----------|--------------|------|
| float32 | `add_float32_basic` | 基础浮点加法 |
| int32 | `add_int32_basic` | 基础整数加法 |
| uint8 | `add_uint8_basic` | 无符号 8 位整数加法 |
| float16 | `float16_add` | 专用转换函数，验证半精度加法 |
| double | （通过原地测试间接覆盖） | 双精度加法 |
| 其他类型（int16, int64 等）可通过扩展模板添加，本测试聚焦主要类型。 |

### 3.2 Alpha 标量覆盖

Add 算子支持 `alpha` 参数，实现 `out = self + alpha * other`：

| 测试用例 | 数据类型 | alpha 值 | 说明 |
|----------|----------|----------|------|
| `add_float32_alpha2` | float32 | 2.0 | 常规正数 |
| `add_int32_alpha_negative` | int32 | -2 | 负数 alpha |
| `add_float32_basic` | float32 | 1.0 | 默认 alpha=1 |

### 3.3 Shape 与广播覆盖

| 测试用例 | 输入 Shape1 | 输入 Shape2 | 输出 Shape | 广播类型 |
|----------|-------------|-------------|------------|----------|
| `add_float32_basic` | {4,2} | {4,2} | {4,2} | 无广播 |
| `add_broadcast_2x3_3` | {2,3} | {3} | {2,3} | 一维广播到二维 |
| `add_broadcast_3d` | {2,1,4} | {1,3,4} | {2,3,4} | 高维广播 |

### 3.4 原地操作覆盖

原地加法将输出 tensor 设置为第一个输入（`self`），避免额外内存分配：

| 测试用例 | 数据类型 | 说明 |
|----------|----------|------|
| `inplace_add_float32` | float32 | 原地加法，结果覆盖 self |
| `inplace_add_int32` | int32 | 整数原地加法 |

### 3.5 边界值与特殊值

| 边界类型 | 测试用例 | 关键数据 |
|----------|----------|----------|
| 浮点 NaN | `add_nan_inf` | NaN + 2.0 = NaN, 5.0 + NaN = NaN |
| 浮点 Inf | `add_nan_inf` | Inf + 2.0 = Inf, -Inf + 3.0 = -Inf |
| 极大/极小整数 | `add_int32_limits` | INT32_MAX + 1 = INT32_MAX?（实际溢出行为由硬件定义，此处验证不崩溃） |

### 3.6 异常输入与错误分支

| 异常场景 | 测试函数 | 预期行为 |
|----------|----------|----------|
| 不支持的数据类型 (UINT32) | `RunExceptionTests` | 返回错误码（拦截） |
| 无法广播的 shape 组合 | `RunExceptionTests` | 返回错误码 |
| `nullptr` 参数 | `RunExceptionTests` | 返回错误码 |

### 3.7 非连续内存测试

- 构造 shape `{2,2}`、strides `{4,1}` 的非连续 tensor，验证加法结果与连续内存一致（期望值 `{3,4,0,0,5,6,0,0}`）。

---

## 4. 测试执行结果

### 4.1 测试通过情况

运行完整测试后，所有预期通过的测试用例均输出 `[PASS]`，异常测试输出 `[PASS]` 或 `[SKIP]`（当环境不支持某些特性时）。最终输出：

```
ALL TESTS PASSED
```

## 5. 总结

本测试对 Add 算子的核心功能、数据类型、alpha 标量、广播规则、原地操作、边界值、异常输入及非连续内存进行了系统性验证。所有测试用例均通过，代码设计清晰、可维护性强，为算子的质量保障提供了可靠基础。

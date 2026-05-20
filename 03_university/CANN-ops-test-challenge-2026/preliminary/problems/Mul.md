# 题目 A：Mul 算子测试用例设计

## 任务说明

本题目要求参赛者为 CANN ops-math 仓库中的 **Mul（逐元素乘法）算子**编写端到端测试用例。参赛者需要在官方提供的 example 测试代码基础上进行扩展，尽可能覆盖算子的各种执行路径，最终以代码覆盖率作为主要评价指标。

**算子定义：** $y = x_1 \times x_2$，当两个输入的 shape 不一致时，按广播规则对齐后逐元素计算。

**难度：** 基础

## 算子概况

Mul 算子位于 `math/mul/` 目录下，采用 op_api → op_host → op_kernel 的三层架构：

- **op_api 层**（运行于 CPU）：参数校验、类型提升、路由调度
- **op_host 层**（运行于 CPU）：shape 推断、tiling 切分计算
- **op_kernel 层**（运行于 NPU）：按切分方案执行逐元素乘法

### 支持的数据类型

Mul 算子注册了 16 种合法的 `(x1, x2, output)` 数据类型组合，包括同类型（如 FLOAT-FLOAT-FLOAT、INT32-INT32-INT32）和混合类型（如 FLOAT16-FLOAT-FLOAT、BF16-FLOAT-FLOAT）。完整列表可查看 `op_host/mul_def.cpp`。

### API 变体

Mul 算子对外提供 4 个 API，覆盖不同的使用场景：

| API                                | 语义                                   |
| ---------------------------------- | -------------------------------------- |
| `aclnnMul(self, other, out)`       | out = self * other（tensor 乘 tensor） |
| `aclnnMuls(self, other, out)`      | out = self * scalar（tensor 乘标量）   |
| `aclnnInplaceMul(selfRef, other)`  | selfRef *= other（原地乘 tensor）      |
| `aclnnInplaceMuls(selfRef, other)` | selfRef *= scalar（原地乘标量）        |

每个 API 在 op_api 层走不同的代码路径。

## 任务要求

官方示例代码位于 `math/mul/examples/test_aclnn_mul.cpp`，提供了完整的端到端调用骨架，但仅包含一组 float32 类型的测试数据，且未进行结果验证。参赛者的任务是在该文件的基础上进行修改和扩展：

**1）补充结果验证。** 为每个测试用例在 CPU 端独立计算期望值，并与算子输出进行数值比对。浮点类型使用容差比较：

$$|actual - expected| \leq atol + rtol \times |expected|$$

建议容差：FLOAT32 取 1e-5，FLOAT16 取 1e-3，BF16 取 1e-2，整数类型精确匹配。

**2）扩展测试覆盖面。** 覆盖维度包括但不限于：

- **数据类型**：不同 dtype 会触发不同的调度路径和 tiling 策略
- **Shape 组合**：同 shape、广播（如 `[2,3]` 与 `[3]`）、标量、较大 tensor 等
- **数值边界**：零值、负数、极大值、NaN、Inf 等
- **API 变体**：Mul、Muls、InplaceMul、InplaceMuls
- **异常输入**：nullptr、不支持的 dtype 等

**3）输出格式。** 每个测试用例输出 `[PASS]` 或 `[FAIL]`，程序结尾输出汇总，有失败用例返回非 0 值。

## 编译与运行

```bash
# 编译（启用覆盖率插桩）
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov

# 安装算子包
./build_out/cann-ops-math-custom_linux-x86_64.run

# 运行测试（CPU 模拟器）
bash build.sh --run_example mul eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov

# 查看覆盖率
find build -name "*.gcda" | grep mul
gcov -b <gcda文件路径>
```

每次修改测试用例后，需要重新执行编译 → 安装 → 运行的完整流程以更新覆盖率数据。

## 评分标准

评分以代码覆盖率为主要依据。覆盖率统计范围为以下 3 个源文件：

| 层次    | 文件                                   | 内容                                                         | 有效行数 |
| ------- | -------------------------------------- | ------------------------------------------------------------ | :------: |
| op_api  | `op_api/aclnn_mul.cpp`                 | API 层调度逻辑：参数校验、类型提升、混合类型处理、API 变体分发 |   313    |
| op_api  | `op_api/mul.cpp`                       | 设备路由：AiCore / AiCpu 选择、dtype 支持判断                |    47    |
| op_host | `op_host/arch35/mul_tiling_arch35.cpp` | Tiling 策略：dtype 组合分发、平台信息获取                    |   192    |

评分包含以下几个方面：

1. **编译通过（前置条件）。** 提交的 `test_aclnn_mul.cpp` 必须能在评测环境中通过上述编译和运行流程正常执行。编译失败的提交将无法获得覆盖率得分，但评测系统会尝试从提交的 build 目录中提取覆盖率数据作为参考。
2. **覆盖率得分（主要依据）。** 按上述公式计算综合覆盖率，覆盖率越高得分越高。
3. **结果验证（必要条件）。** 测试代码中必须包含有效的结果验证逻辑（即计算期望值并与实际输出比对），仅打印结果而不验证的提交将被扣分。

## 提交要求

提交一个压缩包 `.zip`，包含：

```
<队名>/
├── test_aclnn_mul.cpp          # 测试用例源文件（必须）
├── build/                      # 编译产物目录，包含覆盖率数据（必须）
└── 测试报告.md / .pdf           # 测试设计说明（鼓励提交）
```
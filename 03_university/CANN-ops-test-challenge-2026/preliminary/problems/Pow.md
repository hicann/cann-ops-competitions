# 题目 C：Pow 算子测试用例设计

## 任务说明

本题目要求参赛者为 CANN ops-math 仓库中的 **Pow（逐元素幂运算）算子**编写端到端测试用例。参赛者需要在官方提供的 example 测试代码基础上进行扩展，尽可能覆盖算子的各种执行路径，最终以代码覆盖率作为主要评价指标。

**算子定义：** $y_i = x_{1,i} ^ {x_{2,i}}$，支持 tensor-scalar、scalar-tensor、tensor-tensor 三种组合形式。当两个输入的 shape 不一致时，按广播规则对齐后逐元素计算。

**难度：** 进阶

## 算子概况

Pow 算子位于 `math/pow/` 目录下，采用 op_api → op_host → op_kernel 的三层架构。与 Mul、Add 等二元逐元素算子不同，Pow 算子的两个输入具有**非对称角色**——底数和指数不可交换，因此 op_api 层针对 TensorScalar、ScalarTensor、TensorTensor 三种组合形式分别实现了独立的调度逻辑，op_host 层也对应维护了多个 tiling 策略文件。

### 目录结构

```
math/pow/
├── op_api/                                  # 接口层（4 个源文件，共 1129 行）
│   ├── aclnn_pow.h / aclnn_pow.cpp          #   TensorScalar / ScalarTensor / Inplace API（558 行）
│   ├── aclnn_pow_tensor_tensor.h / .cpp     #   TensorTensor / Inplace API（210 行）
│   ├── aclnn_exp2.h / aclnn_exp2.cpp        #   Exp2 特殊 API（267 行）
│   ├── pow.h / pow.cpp                      #   底层接口与设备路由（94 行）
├── op_host/                                 # 主机计算层（4 个 tiling 文件，共 547 行）
│   ├── pow_def.cpp                          #   算子注册
│   ├── pow_infershape.cpp                   #   shape 推断
│   └── arch35/
│       ├── pow_tiling_arch35.cpp            #   通用 tiling 逻辑（126 行）
│       ├── pow_tensor_scalar_tiling_arch35.cpp  # TensorScalar 专用 tiling（146 行）
│       ├── pow_tensor_tensor_tiling_arch35.cpp  # TensorTensor 专用 tiling（213 行）
│       └── pow_tiling.cpp                   #   tiling 公共逻辑（62 行）
├── op_kernel/                               # 设备计算层
│   └── ...
├── examples/                                # 使用示例（3 个）
└── tests/                                   # 单元测试
```

### 支持的数据类型

Pow 算子注册了 7 种数据类型：BF16、FLOAT16、FLOAT32、UINT8、INT8、INT16、INT32。`pow_tensor_tensor_tiling_arch35.cpp` 中为每种 dtype 分配了不同的 OP_KEY，触发不同的 tiling 策略。

### API 变体

Pow 算子对外提供 **7 个 API**，分为三大类：

| 类别         | API                                              | 语义                                           |
| ------------ | ------------------------------------------------ | ---------------------------------------------- |
| TensorScalar | `aclnnPowTensorScalar(self, exponent, out)`      | $out_i = self_i ^ {scalar}$                    |
| TensorScalar | `aclnnInplacePowTensorScalar(selfRef, exponent)` | $selfRef_i = selfRef_i ^ {scalar}$（原地）     |
| ScalarTensor | `aclnnPowScalarTensor(self, exponent, out)`      | $out_i = scalar ^ {exponent_i}$                |
| TensorTensor | `aclnnPowTensorTensor(self, exponent, out)`      | $out_i = self_i ^ {exponent_i}$                |
| TensorTensor | `aclnnInplacePowTensorTensor(selfRef, exponent)` | $selfRef_i = selfRef_i ^ {exponent_i}$（原地） |
| Exp2         | `aclnnExp2(self, out)`                           | $out_i = 2 ^ {self_i}$                         |
| Exp2         | `aclnnInplaceExp2(selfRef)`                      | $selfRef_i = 2 ^ {selfRef_i}$（原地）          |

**关键特征：** 这三类 API 在 op_api 层由不同的源文件实现（`aclnn_pow.cpp`、`aclnn_pow_tensor_tensor.cpp`、`aclnn_exp2.cpp`），各自包含独立的参数校验、类型提升和调度逻辑。不调用某一类 API，对应源文件中的代码就不会被执行。

### 与 Mul/Add 的关键差异

1. **非对称 API 设计**：底数和指数角色不可交换，TensorScalar、ScalarTensor、TensorTensor 是三条完全独立的代码路径
2. **多个 API 源文件**：3 个独立的 `aclnn_*.cpp` 文件，需分别覆盖
3. **多个 tiling 文件**：TensorScalar 和 TensorTensor 各有专用的 tiling 策略文件
4. **特殊指数优化**：`aclnn_pow.cpp` 内部对特殊指数值（如 0.5→sqrt、2→square、3→cube、-1→reciprocal）可能有专用优化路径
5. **数学边界更丰富**：$0^0$、负数的非整数次幂、大数溢出、$x^0 = 1$ 等边界条件

## 任务要求

官方示例代码位于 `math/pow/examples/` 目录下，包含 3 个文件：

- `test_aclnn_pow.cpp`：演示 PowTensorScalar 和 InplacePowTensorScalar
- `test_aclnn_pow_tensor_tensor.cpp`：演示 PowTensorTensor
- `test_aclnn_exp2.cpp`：演示 Exp2

参赛者可以将所有测试用例整合到一个 `test_aclnn_pow.cpp` 文件中，也可以选择在其中一个示例文件基础上扩展。

**1）补充结果验证。** 为每个测试用例在 CPU 端独立计算期望值，并与算子输出进行数值比对。Pow 算子的期望值计算：

```cpp
// TensorScalar: base 为 float 数组，exp 为标量
double expected = std::pow((double)base[i], (double)exp_scalar);

// ScalarTensor: base 为标量，exp 为 float 数组
double expected = std::pow((double)base_scalar, (double)exp[i]);

// TensorTensor: base 和 exp 均为 float 数组
double expected = std::pow((double)base[i], (double)exp[i]);

// Exp2:
double expected = std::pow(2.0, (double)self[i]);
```

浮点类型使用容差比较：$|actual - expected| \leq atol + rtol \times |expected|$

**2）扩展测试覆盖面。** 覆盖维度包括但不限于：

- **API 类别**：TensorScalar、ScalarTensor、TensorTensor、Exp2 四大类及各自的 Inplace 变体，共 7 个 API——每一类走不同的源文件，缺一不可
- **数据类型**：FLOAT32、FLOAT16、BF16、INT32、INT8 等，不同 dtype 触发不同的 tiling OP_KEY
- **特殊指数值**：exponent = 0（任何数的 0 次幂为 1）、exponent = 1、exponent = 0.5（等价于 sqrt）、exponent = 2（平方）、exponent = -1（倒数）、exponent = 3（立方）等
- **Shape 组合**：同 shape、广播、标量 base 等
- **数值边界**：base=0、base 为负数配合整数/非整数指数、NaN、Inf、极大指数导致溢出等
- **异常输入**：nullptr、不支持的 dtype 等

**3）输出格式。** 每个测试用例输出 `[PASS]` 或 `[FAIL]`，程序结尾输出汇总，有失败用例返回非 0 值。

## 编译与运行

```bash
# 编译（启用覆盖率插桩）
bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov

# 安装算子包
./build_out/cann-ops-math-custom_linux-x86_64.run

# 运行测试（CPU 模拟器）
bash build.sh --run_example pow eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov

# 查看覆盖率
find build -name "*.gcda" | grep pow
gcov -b <gcda文件路径>
```

每次修改测试用例后，需要重新执行编译 → 安装 → 运行的完整流程以更新覆盖率数据。

## 评分标准

评分以代码覆盖率为主要依据。覆盖率统计范围为以下 5 个源文件：

| 层次    | 文件                                                 | 内容                                                         | 有效行数 |
| ------- | ---------------------------------------------------- | ------------------------------------------------------------ | :------: |
| op_api  | `op_api/aclnn_pow.cpp`                               | TensorScalar / ScalarTensor API：参数校验、特殊指数优化、类型提升 |   236    |
| op_api  | `op_api/aclnn_pow_tensor_tensor.cpp`                 | TensorTensor API：独立的参数校验和调度逻辑                   |    73    |
| op_api  | `op_api/pow.cpp`                                     | 设备路由：AiCore / AiCpu 选择、dtype 支持判断                |    30    |
| op_host | `op_host/arch35/pow_tensor_tensor_tiling_arch35.cpp` | TensorTensor Tiling 策略：7 种 dtype 的 OP_KEY 分发          |   126    |
| op_host | `op_host/arch35/pow_tiling_arch35.cpp`               | 通用 Tiling 逻辑                                             |    53    |

评分包含以下几个方面：

1. **编译通过（前置条件）。** 提交的 `test_aclnn_pow.cpp` 必须能在评测环境中通过上述编译和运行流程正常执行。编译失败的提交将无法获得覆盖率得分，但评测系统会尝试从提交的 build 目录中提取覆盖率数据作为参考。
2. **覆盖率得分（主要依据）。** 按上述公式计算综合覆盖率，覆盖率越高得分越高。
3. **结果验证（必要条件）。** 测试代码中必须包含有效的结果验证逻辑（即计算期望值并与实际输出比对），仅打印结果而不验证的提交将被扣分。

## 提交要求

提交一个压缩包 `.zip`，包含：

```
<队名>/
├── test_aclnn_pow.cpp          # 测试用例源文件（必须）
├── build/                      # 编译产物目录，包含覆盖率数据（必须）
└── 测试报告.md / .pdf           # 测试设计说明（鼓励提交）
```
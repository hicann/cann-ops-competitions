# 题目 B：Add 算子测试用例设计

## 任务说明

本题目要求参赛者为 CANN ops-math 仓库中的 **Add（逐元素加法）算子**编写端到端测试用例。参赛者需要在官方提供的 example 测试代码基础上进行扩展，尽可能覆盖算子的各种执行路径，最终以代码覆盖率作为主要评价指标。

**算子定义：** $y = x_1 + \alpha \times x_2$，其中 $\alpha$ 为标量缩放因子（默认为 1）。当两个输入的 shape 不一致时，按广播规则对齐后逐元素计算。

**难度：** 基础

## 算子概况

Add 算子位于 `math/add/` 目录下，采用 op_api → op_host → op_kernel 的三层架构。与 Mul 算子相比，Add 算子的结构基本一致，但额外引入了 `alpha` 缩放参数和 V3 版本 API。

### 目录结构

```
math/add/
├── op_api/                         # 接口层
│   ├── aclnn_add.h / aclnn_add.cpp #   API 声明与实现（713 行）
│   ├── aclnn_add_v3.h / .cpp       #   V3 版本 API（247 行）
│   ├── add.h / add.cpp             #   底层接口与设备路由（162 行）
├── op_host/                        # 主机计算层
│   ├── add_def.cpp                 #   算子注册：声明支持的 dtype 组合
│   ├── add_infershape.cpp          #   shape 推断
│   └── arch35/
│       └── add_tiling_arch35.cpp   #   tiling 切分策略（190 行）
├── op_kernel/                      # 设备计算层
│   └── ...
├── examples/                       # 使用示例
└── tests/                          # 单元测试
```

### 支持的数据类型

Add 算子支持 14 种数据类型组合，包括 BF16、FLOAT16、FLOAT32、INT32、UINT8、INT8、INT64、BOOL、COMPLEX32、COMPLEX64 等同类型运算，以及 FLOAT16-FLOAT、BF16-FLOAT 等混合类型运算。完整列表可查看 `op_host/add_def.cpp`。

### API 变体

Add 算子对外提供 6 个 API，分为标准版和 V3 版两组：

| API                                        | 语义                                              |
| ------------------------------------------ | ------------------------------------------------- |
| `aclnnAdd(self, other, alpha, out)`        | out = self + alpha * other（tensor 加 tensor）    |
| `aclnnAdds(self, other, alpha, out)`       | out = self + alpha * scalar（tensor 加标量）      |
| `aclnnInplaceAdd(selfRef, other, alpha)`   | selfRef += alpha * other（原地加 tensor）         |
| `aclnnInplaceAdds(selfRef, other, alpha)`  | selfRef += alpha * scalar（原地加标量）           |
| `aclnnAddV3(self, other, alpha, out)`      | out = scalar + alpha * other（**标量**加 tensor） |
| `aclnnInplaceAddV3(selfRef, other, alpha)` | V3 版本的原地加法                                 |

**注意：** 与 Mul 不同，Add 的所有 API 都包含一个 `alpha` 参数（`aclScalar*` 类型），用于对第二个输入进行缩放。当 `alpha = 1` 时等价于普通加法，当 `alpha` 取其他值时会走不同的计算路径。这是 Add 算子相比 Mul 算子额外的测试维度。

### V3 版本 API 说明

`aclnnAddV3` 与标准 `aclnnAdd` 的核心区别在于 **`self` 参数的类型不同**：

|               | aclnnAdd                     | aclnnAddV3                  |
| ------------- | ---------------------------- | --------------------------- |
| self 参数类型 | `const aclTensor*`（tensor） | `const aclScalar*`（标量）  |
| 语义          | tensor + alpha * tensor      | **scalar** + alpha * tensor |

V3 版本本质上是 **ScalarTensor 形式的加法**——第一个输入是标量而非 tensor。它在 `aclnn_add_v3.cpp`（247 行）中有完全独立的实现，包括独立的类型提升逻辑和三分支调度（alpha=1 直接 Add / 支持 Axpy 的类型走融合算子 / 其余先 Mul 再 Add）。不调用 V3 API，该文件中的代码就不会被覆盖。

调用 V3 API 时，需要引入对应的头文件 `aclnnop/aclnn_add_v3.h`，并注意 self 参数用 `aclCreateScalar` 创建。

### 与 Mul 的关键差异

1. **alpha 参数**：每个 API 调用都需要传入 `aclScalar* alpha`，不同 alpha 值可能触发不同的优化路径
2. **V3 版本 API**：ScalarTensor 形式的加法，独立的 247 行代码，走不同的调度逻辑
3. **AICPU 后备路径**：某些数据类型（如 DOUBLE）会被路由到 AICPU 而非 AICore，在 CPU 模拟器环境下可能需要特殊处理

## 任务要求

官方示例代码位于 `math/add/examples/test_aclnn_add.cpp`。**注意：官方示例使用 ACL_DOUBLE 类型，在 CPU 模拟器环境下会路由到 AICPU 并报错。参赛者需要将示例中的数据类型修改为 ACL_FLOAT 后方可正常运行，或直接编写自己的测试用例。**

参赛者的任务：

**1）补充结果验证。** 为每个测试用例在 CPU 端独立计算期望值，并与算子输出进行数值比对。Add 算子的期望值计算需要考虑 alpha 参数：

```cpp
// x1, x2 为 float 输入，alpha 为缩放因子
double expected = (double)x1[i] + alpha * (double)x2[i];
```

浮点类型使用容差比较：$|actual - expected| \leq atol + rtol \times |expected|$

**2）扩展测试覆盖面。** 覆盖维度包括但不限于：

- **数据类型**：FLOAT32、FLOAT16、BF16、INT32、INT8 等，不同 dtype 触发不同的 tiling 策略分支
- **alpha 参数**：alpha=1（标准加法）、alpha=0、alpha 为负数、alpha 为浮点数等
- **Shape 组合**：同 shape、广播、标量、较大 tensor 等
- **数值边界**：零值、极大值、NaN、Inf、整数溢出等
- **API 变体**：Add、Adds、InplaceAdd、InplaceAdds、AddV3、InplaceAddV3 共 6 个 API
- **异常输入**：nullptr、不支持的 dtype 等

**3）输出格式。** 每个测试用例输出 `[PASS]` 或 `[FAIL]`，程序结尾输出汇总，有失败用例返回非 0 值。

## 编译与运行

```bash
# 编译（启用覆盖率插桩）
bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov

# 安装算子包
./build_out/cann-ops-math-custom_linux-x86_64.run

# 运行测试（CPU 模拟器）
bash build.sh --run_example add eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov

# 查看覆盖率
find build -name "*.gcda" | grep add
gcov -b <gcda文件路径>
```

每次修改测试用例后，需要重新执行编译 → 安装 → 运行的完整流程以更新覆盖率数据。

## 评分标准

评分以代码覆盖率为主要依据。覆盖率统计范围为以下 4 个源文件：

| 层次    | 文件                                   | 内容                                                         | 有效行数 |
| ------- | -------------------------------------- | ------------------------------------------------------------ | :------: |
| op_api  | `op_api/aclnn_add.cpp`                 | API 层调度逻辑：参数校验、类型提升、alpha 处理、API 变体分发 |   287    |
| op_api  | `op_api/aclnn_add_v3.cpp`              | V3 版本 API：独立的调度逻辑和类型处理                        |   247    |
| op_api  | `op_api/add.cpp`                       | 设备路由：AiCore / AiCpu 选择、dtype 支持判断                |    55    |
| op_host | `op_host/arch35/add_tiling_arch35.cpp` | Tiling 策略：dtype 组合分发、平台信息获取                    |    90    |


评分包含以下几个方面：

1. **编译通过（前置条件）。** 提交的 `test_aclnn_add.cpp` 必须能在评测环境中通过上述编译和运行流程正常执行。编译失败的提交将无法获得覆盖率得分，但评测系统会尝试从提交的 build 目录中提取覆盖率数据作为参考。
2. **覆盖率得分（主要依据）。** 按上述公式计算综合覆盖率，覆盖率越高得分越高。
3. **结果验证（必要条件）。** 测试代码中必须包含有效的结果验证逻辑（即计算期望值并与实际输出比对），仅打印结果而不验证的提交将被扣分。

## 提交要求

提交一个压缩包 `.zip`，包含：

```
<队名>/
├── test_aclnn_add.cpp          # 测试用例源文件（必须）
├── build/                      # 编译产物目录，包含覆盖率数据（必须）
└── 测试报告.md / .pdf           # 测试设计说明（鼓励提交）
```
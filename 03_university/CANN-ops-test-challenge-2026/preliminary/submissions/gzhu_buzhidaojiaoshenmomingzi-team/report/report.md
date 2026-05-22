# CANN ops-math 算子测试报告

## 团队信息

- 团队名称：不知道叫什么名字队
- 所属单位：广州大学
- 团队成员：
  - 陈慧美，队长
  - 叶翔宇，成员
- 算子库：cann-ops-math

---

## 一、测试概述

本报告针对 CANN ops-math 仓库中的 Mul、Add、Pow 三个算子进行端到端测试用例设计，以代码覆盖率作为主要评价指标。测试在 Docker 环境（`yeren666/cann-ops-test:v1.0`）中通过 CPU 模拟器执行，使用 `--soc=ascend950 --simulator` 模式运行。

### 测试策略

覆盖率优先，路径多样性胜过数值极端性。通过以下维度组合覆盖 `op_api` 和 `op_host` 层的不同代码分支：

1. **API 变体覆盖**：遍历每个算子的所有 API 入口，确保 `aclnn_<op>.cpp` 中的参数验证和调度分支被覆盖
2. **dtype 组合覆盖**：测试所有注册的数据类型组合，确保 `op_host` tiling 分发表中每个条目被触发
3. **Shape 与广播**：覆盖 1D-4D、不同广播场景，触发 `infershape` 中的广播推理和验证分支
4. **特殊参数值**：alpha、scalar、exponent 等参数的边界值和特殊值，触发条件判断分支
5. **异常输入**：nullptr、空 tensor 等，覆盖错误处理分支

### 结果验证方法

所有测试用例均包含结果验证逻辑：

- **浮点类型**：使用 `atol + rtol * |expected|` 容差比较（Float32: atol=1e-3, rtol=1e-3；BF16: atol=1e-2, rtol=1e-2）
- **整数类型**：精确匹配
- **CPU 参考计算**：使用 `double` 精度计算期望值，避免浮点累积误差

---

## 二、Mul 算子测试

### 2.1 算子说明

数学语义：`y = x1 * x2`，逐元素乘法。

### 2.2 API 覆盖

| API | 语义 | 测试状态 |
|-----|------|----------|
| aclnnMul | out = self * other（tensor * tensor） | 已覆盖 |
| aclnnMuls | out = self * scalar（tensor * scalar） | 已覆盖 |
| aclnnInplaceMul | selfRef *= other（in-place tensor） | 已覆盖 |
| aclnnInplaceMuls | selfRef *= scalar（in-place scalar） | 已覆盖 |

### 2.3 测试阶段

#### Phase 1：API 覆盖测试

验证 4 个 API 变体的基本功能正确性，使用 FLOAT 类型、{4,4} shape。

#### Phase 2：dtype 覆盖测试

覆盖 DTYPE_MAP 中的 9 种数据类型，触发 tiling 分发表中的不同策略函数：

| 数据类型 | 对应 OP_KEY | 说明 |
|----------|-------------|------|
| FLOAT | OP_KEY_FLOAT | 标准浮点乘法 |
| FLOAT16 | OP_KEY_FLOAT16 | 半精度浮点 |
| BF16 | OP_KEY_BF16 | BFloat16 |
| INT32 | OP_KEY_INT32 | 32位整数 |
| INT64 | OP_KEY_INT64 | 64位整数 |
| INT16 | OP_KEY_INT16 | 16位整数 |
| INT8 | OP_KEY_INT8 | 8位有符号整数 |
| UINT8 | OP_KEY_UINT8 | 8位无符号整数 |
| BOOL | OP_KEY_BOOL | 布尔类型（AND 语义） |

#### Phase 3：混合数据类型测试

测试不同输入类型组合，触发 `MixedMulCastCompute` 路径：

- BF16 + FLOAT -> FLOAT
- FLOAT16 + FLOAT -> FLOAT

#### Phase 4：Shape 与广播测试

| 场景 | self shape | other shape | 说明 |
|------|-----------|-------------|------|
| 1D | (8,) | (8,) | 一维张量 |
| 2D | (4,4) | (4,4) | 二维张量 |
| 3D | (2,3,4) | (2,3,4) | 三维张量 |
| 4D | (2,2,2,2) | (2,2,2,2) | 四维张量 |
| 行广播 | (4,3) | (1,3) | 行方向广播 |
| 列广播 | (4,3) | (3,) | 列方向广播 |
| 标量广播 | (4,3) | (1,1) | 标量广播 |

#### Phase 5：数值边界测试

| 测试 | 值 | 目的 |
|------|----|------|
| Muls scalar=0 | 0.0 | 零乘 |
| Muls scalar=负数 | -2.5 | 负数缩放 |
| Muls scalar=大值 | 100.0 | 大值缩放 |
| Muls scalar=小数 | 0.001 | 小数缩放 |
| Mul 含零值 | 交替0 | 零值传播 |
| InplaceMul | 基本值 | 原地操作 |
| InplaceMuls | scalar=3 | 原地标量操作 |

#### Phase 6：特殊场景测试

| 场景 | 说明 |
|------|------|
| 空 tensor | shape={0}，验证 API 对空输入的处理 |
| nullptr 输入 | 分别测试 self/other/out 为 nullptr，验证错误返回 |
| BF16 Muls | 验证 BF16 类型的 Muls 融合优化路径 |

---

## 三、Add 算子测试

### 3.1 算子说明

数学语义：`y = x1 + alpha * x2`，逐元素加法，含 alpha 参数。

### 3.2 API 覆盖

| API | 语义 | 测试状态 |
|-----|------|----------|
| aclnnAdd | out = self + alpha * other（tensor + tensor） | 已覆盖 |
| aclnnAdds | out = self + alpha * other（tensor + scalar） | 已覆盖 |
| aclnnInplaceAdd | selfRef += alpha * other（in-place tensor） | 已覆盖 |
| aclnnInplaceAdds | selfRef += alpha * other（in-place scalar） | 已覆盖 |
| aclnnAddV3 | out = self(scalar) + alpha * other(tensor) | 已覆盖 |
| aclnnInplaceAddV3 | otherRef = self(scalar) + alpha * otherRef | 已覆盖 |

### 3.3 测试阶段

#### Phase 1：API 覆盖测试

验证 6 个 API 变体的基本功能，使用 FLOAT 类型、{4,3} shape、alpha=1.0。

#### Phase 2：数据类型测试

覆盖 7 种数据类型：FLOAT、FLOAT16、BF16、INT32、INT8、UINT8、INT64。

#### Phase 3：Alpha 参数测试

| alpha 值 | 语义 | 测试目的 |
|----------|------|----------|
| 1.0 | 标准加法 | 基本路径 |
| 0.0 | other 项消失 | alpha=0 分支 |
| -2.0 | 负数缩放 | 负 alpha 分支 |
| 0.5 | 浮点缩放 | 非整数 alpha 分支 |
| 10.0 | 大值缩放 | 大 alpha 分支 |

#### Phase 4：Shape 与广播测试

| 场景 | self shape | other shape |
|------|-----------|-------------|
| 1D | (8,) | (8,) |
| 2D | (4,4) | (4,4) |
| 3D | (2,3,4) | (2,3,4) |
| 4D | (2,2,2,2) | (2,2,2,2) |
| 大张量 | (64,64) | (64,64) |
| 行广播 | (4,3) | (1,3) |
| 列广播 | (4,3) | (3,) |
| 标量广播 | (4,3) | (1,1) |

#### Phase 5：V3 API 场景测试

| self | alpha | 语义 | 数据类型 |
|------|-------|------|----------|
| 0 | 1 | 0 + tensor | FLOAT, FLOAT16 |
| 5 | 1 | 标量加张量 | FLOAT, FLOAT16 |
| 10 | 2.5 | 非单位 alpha | FLOAT, FLOAT16 |
| -3 | 1 | 负标量 | FLOAT, FLOAT16 |
| 7 | 0 | 仅保留 self | FLOAT, FLOAT16 |

#### Phase 6：特殊场景测试

| 场景 | 说明 |
|------|------|
| 空 tensor | shape={2,0,3}，验证 API 对空输入的接受 |
| nullptr 输入 | 分别测试 aclnnAdd/adds/AddV3 各参数为 nullptr，共 8 项 |
| 边界值 | 大 alpha 值测试 |

---

## 四、Pow 算子测试

### 4.1 算子说明

数学语义：`y_i = x_{1,i}^{x_{2,i}}`，逐元素幂运算。支持 TensorScalar、ScalarTensor、TensorTensor 三类 API。

### 4.2 API 覆盖

| API | 语义 | 测试状态 |
|-----|------|----------|
| aclnnPowTensorScalar | tensor ^ scalar | 已覆盖 |
| aclnnInplacePowTensorScalar | selfRef ^= scalar（in-place） | 已覆盖 |
| aclnnPowScalarTensor | scalar ^ tensor | 已覆盖 |
| aclnnPowTensorTensor | tensor ^ tensor | 已覆盖 |
| aclnnInplacePowTensorTensor | selfRef ^= tensor（in-place） | 已覆盖 |
| aclnnExp2 | 2 ^ tensor | 已覆盖 |
| aclnnInplaceExp2 | selfRef = 2 ^ selfRef（in-place） | 已覆盖 |

### 4.3 测试阶段

#### Phase 1：API 覆盖测试

验证 7 个 API 变体的基本功能，使用 FLOAT 类型、{4,3} shape。

#### Phase 2：特殊指数值测试

特殊指数值触发算子内部的优化路径（sqrt/square/cube/reciprocal 等）：

| 指数值 | 语义 | 触发的优化路径 |
|--------|------|----------------|
| 0.5 | 平方根 | sqrt 优化 |
| 2.0 | 平方 | square 优化 |
| 3.0 | 立方 | cube 优化 |
| -0.5 | 平方根倒数 | rsqrt 优化 |
| -1.0 | 倒数 | reciprocal 优化 |
| -2.0 | 平方倒数 | 无特殊优化 |
| 0.0 | 零次幂（结果=1） | 常量路径 |
| 1.0 | 一次幂（不变） | 恒等路径 |

#### Phase 3：数据类型测试

覆盖 7 种数据类型：FLOAT、FLOAT16、BF16、INT32、INT8、UINT8、INT16。

#### Phase 4：Shape 测试

| 维度 | shape |
|------|-------|
| 1D | (8,) |
| 2D | (4,4) |
| 3D | (2,3,4) |
| 4D | (2,2,2,2) |
| 大张量 | (32,32) |

#### Phase 5：TensorTensor 测试

**广播场景：**

| self shape | exp shape | 广播类型 |
|-----------|-----------|----------|
| (4,3) | (4,3) | 同 shape |
| (4,3) | (1,3) | 行广播 |
| (4,3) | (3,) | 列广播 |
| (4,3) | (1,) | 标量广播 |

**Tiling 分发覆盖（7 个 OP_KEY）：**

| OP_KEY | 数据类型 |
|--------|----------|
| OP_KEY_1 | FLOAT16 |
| OP_KEY_2 | BF16 |
| OP_KEY_3 | FLOAT |
| OP_KEY_4 | UINT8 |
| OP_KEY_5 | INT8 |
| OP_KEY_6 | INT16 |
| OP_KEY_7 | INT32 |

#### Phase 6：ScalarTensor 特殊场景

| base | 语义 |
|------|------|
| 2.0 | 典型 2^x 场景（与 Exp2 对比） |
| 10.0 | 10^x 场景 |
| 0.5 | 小数底数 |
| 1.0 | 1^x = 1（恒等） |

#### Phase 7：特殊场景测试

| 场景 | 说明 |
|------|------|
| 空 tensor | shape={2,0,3}，验证 API 对空输入的接受 |
| nullptr 输入 | 分别测试 PowTensorScalar/PowScalarTensor/PowTensorTensor/Exp2 各参数为 nullptr，共 8 项 |

---

## 五、测试用例汇总

| 算子 | API 数量 | dtype 覆盖 | 测试阶段数 | 主要覆盖维度 |
|------|----------|-----------|-----------|-------------|
| Mul | 4 | 9 种 + 2 种混合 | 6 | API、dtype、混合dtype、shape/广播、数值边界、特殊场景 |
| Add | 6 | 7 种 | 6 | API、dtype、alpha参数、shape/广播、V3场景、特殊场景 |
| Pow | 7 | 7 种 + 7 种 tiling | 7 | API、特殊指数、dtype、shape、TensorTensor/tiling、ScalarTensor、特殊场景 |

## 六、覆盖率目标分析

### op_api 层（权重 60%）

通过以下方式覆盖 `aclnn_<op>.cpp` 和 `<op>.cpp` 中的条件分支：

- 每个算子的所有 API 变体均已独立测试，覆盖参数验证和调度分发分支
- 不同 dtype 组合覆盖类型提升和 AI Core/AI CPU 路由分支
- alpha/scalar/exponent 的特殊值覆盖参数处理分支
- nullptr 和空 tensor 覆盖错误处理分支

### op_host 层（权重 40%）

通过以下方式覆盖 `<op>_infershape.cpp` 和 `<op>_tiling_arch35.cpp`：

- 广播场景覆盖 `infershape` 中的广播推理和维度验证分支
- 每种注册 dtype 组合至少一个测试用例，确保 tiling 分发表中每个策略函数被调用
- Mul: 9 种 dtype + 2 种混合 dtype
- Pow: 7 种 OP_KEY 的 TensorTensor tiling 覆盖

## 七、编译与运行

```bash
cd /home/workspace/ops-math

# 编译
bash build.sh --pkg --soc=ascend950 --ops=mul,add,pow --vendor_name=custom --cov

# 安装
./build_out/cann-ops-math-custom_linux-x86_64.run

# 运行
bash build.sh --run_example <op> eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov

# 查看覆盖率
find build -name "*.gcda" | grep <op>
gcov -b <gcda文件路径>
```

其中 `<op>` 替换为 `mul`、`add` 或 `pow`。

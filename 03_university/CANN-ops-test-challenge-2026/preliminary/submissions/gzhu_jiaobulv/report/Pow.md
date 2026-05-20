# Pow 算子测试报告

## 1. 变更概述

- 合并后主测试文件：`examples/test_aclnn_pow.cpp`
- 已将以下能力统一到同一 `main`：
  - TensorScalar / InplacePowTensorScalar
  - ScalarTensor
  - TensorTensor / InplacePowTensorTensor
  - Exp2 / InplaceExp2
  - API 参数校验与异常输入校验
- 为兼容现有构建流程，保留了两个兼容壳文件：
  - `examples/test_aclnn_pow_tensor_tensor.cpp`
  - `examples/test_aclnn_exp2.cpp`
    这两个文件仅打印“已合并到 test_aclnn_pow.cpp”的提示并返回 0。

## 2. 测试目标

- 覆盖题目《C_Pow算子测试》要求的四类 API 与 Inplace 变体。
- 增强边界数据覆盖：
  - 指数边界：`0`、`1`、`0.5`、`2`、`3`、`-1`、`-2`
  - 数值边界：`NaN`、`Inf`、空 Tensor
  - 类型边界：`FLOAT/DOUBLE/INT32/INT64/INT16/INT8/UINT8/BOOL`，并在 API 校验中补充 `FLOAT16/BF16/COMPLEX/UINT32/UINT64`
- 强化异常分支：`nullptr`、不支持 dtype、输出 dtype 不匹配、非法 inplace broadcast、广播形状不匹配。

## 3. 用例设计摘要

### 3.1 TensorScalar / ScalarTensor

- 正常功能：浮点、整型、NCHW 格式、空 Tensor。
- 指数特殊值：0/1/0.5/2/3/-1/-2。
- 异常：负整数指数+整型底数、溢出、bool-bool、shape 不匹配。
- API 校验补充：
  - `self` 合法但 `exponent` 非法（补齐 dtype 校验分支）
  - float16/bf16/int16/complex 溢出检测分支
  - 复杂标量提升路径（complex scalar + fp16 tensor）

### 3.2 TensorTensor

- 正常功能：同 shape、广播、整型边界、AICPU 路径、NCHW。
- Inplace：在 shape 可兼容场景执行 inplace 路径。
- 异常：bool-bool、不匹配 out shape。
- API 校验：`nullptr`、不支持 `uint64`、out dtype 不匹配、inplace 非法广播。

### 3.3 Exp2

- 正常功能：`float`、`double`、空 Tensor、inplace。
- 异常：out dtype 不匹配。
- API 校验：`nullptr`、不支持 `uint64`、out dtype 不匹配。

## 4. 运行方式

```bash
bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example pow eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

## 5. 覆盖率迭代关注点

- `pow.cpp`：当前剩余多为平台条件（如 `IsRegBase()==false`）和故障注入分支，常规样例较难继续显著提升。
- `aclnn_pow.cpp`：已重点加大类型/边界/异常路径，建议继续依据 `gcov` 的 `#####` 行定向补充。

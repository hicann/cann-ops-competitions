#  测试报告

## 1. 测试目标

本版本围绕 `Mul` 算子的 4 个公开 API 进行测试设计与验证：

- `aclnnMul`
- `aclnnMuls`
- `aclnnInplaceMul`
- `aclnnInplaceMuls`

测试目标是尽可能提升以下 3 个评分文件的覆盖率：

- `op_api/aclnn_mul.cpp`
- `op_api/mul.cpp`
- `op_host/arch35/mul_tiling_arch35.cpp`

同时保证测试代码具备结果验证、异常输入验证、汇总输出和失败返回值。

## 2. 设计思路

本版本在官方 example 基础上进行了扩展，整体思路如下：

1. 对正常执行路径补充 CPU 侧期望值计算与结果比对。
2. 通过不同 `dtype / shape / API 变体` 触发更多调度与 tiling 分支。
3. 保留必要的异常输入验证，用于覆盖参数校验与边界处理逻辑。
4. 输出统一使用 `[PASS] / [FAIL]`，程序结尾打印汇总信息并在失败时返回非 0。

## 3. 测试覆盖范围

### 3.1 Tensor × Tensor 场景

本版本覆盖了以下典型路径：

- `float32` 基础同 shape 乘法
- `int32` 广播乘法
- `double` 同 shape 乘法
- `double` 的 `NaN / Inf` 边界值乘法
- `uint8 / int8 / int16 / int64 / bool` 等整数与布尔类型路径
- `float16 / float32` 混合类型双向路径
- `bf16 / float32` 混合类型双向路径
- `float16 / float16`、`bf16 / bf16` 同类型路径
- `complex64` 路径
- 标量 shape 与 tensor 广播路径
- 非连续张量（non-contiguous）输入路径
- 空 tensor 的 zero-workspace 路径

### 3.2 Tensor × Scalar 场景

对 `aclnnMuls` 主要覆盖：

- `float32 × float`
- `float32 × double`
- `int64 × int32`
- 非连续张量的 scalar 路径
- 空 tensor zero-workspace 路径
- 输出类型非法场景

### 3.3 Inplace 场景

对 `aclnnInplaceMul` / `aclnnInplaceMuls` 主要覆盖：

- 广播原地乘法
- `float32` 与 `float16` 混合路径
- `float32 × float` 原地 scalar 路径
- `float32 × double` 原地 scalar 路径
- `complex64` 原地 scalar 路径
- `int64 × int32` 原地 scalar 路径
- 空 tensor zero-workspace 路径
- 原地结果类型非法场景

### 3.4 异常输入与参数校验

本版本还补充了常见异常场景，用于覆盖 API 参数检查逻辑：

- `nullptr` 输入
- 广播不合法
- 输出 shape 不匹配
- 维度过高
- 不支持的数据类型
- 原地广播非法

## 4. 结果验证方法

测试结果验证遵循如下原则：

- `float / double`：使用容差比较
- 整数类型：精确匹配
- `complex`：分别比较实部与虚部
- `float16 / bf16`：按对应低精度表示进行结果校验

所有正常执行类测试均在 CPU 侧构造期望值，再与设备侧输出逐项比较。

## 5. 输出与返回值

本版本输出格式如下：

- 每个用例输出一条 `[PASS]` 或 `[FAIL]`
- 程序结尾输出汇总：`=== Summary: X failed ===`
- 若存在失败用例，返回非 0；否则返回 0

## 6. 编译与运行方式

可按题目要求执行：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example mul eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

## 7. 说明

本版本重点关注评分文件中的有效执行路径与参数校验路径，目标是在保证测试可执行和可验证的前提下，尽可能提升 Mul 算子的代码覆盖率。

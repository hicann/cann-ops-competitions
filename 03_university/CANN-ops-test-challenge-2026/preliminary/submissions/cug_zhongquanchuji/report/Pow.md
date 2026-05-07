# Pow算子测试报告

## 1. 测试对象

- 测试文件：`test_aclnn_pow_02.cpp`
- 题目文档：`题目C_Pow算子测试_1775912012977.md`
- 测试目标：在单个测试文件中补充结果验证、扩展 Pow 相关 API 的测试覆盖面，并统计题目指定的 5 个源文件覆盖率。

## 2. 题目要求理解

根据题目说明，本次测试需要满足以下要求：

1. 为测试用例补充 CPU 侧结果验证，不能只打印结果；
2. 覆盖 TensorScalar、ScalarTensor、TensorTensor、Exp2 及对应的 Inplace 变体，共 7 个 API；
3. 每个测试用例输出 `[PASS]` 或 `[FAIL]`，程序结尾输出汇总；
4. 重点关注以下 5 个计分文件的覆盖率：
   - `op_api/aclnn_pow.cpp`
   - `op_api/aclnn_pow_tensor_tensor.cpp`
   - `op_api/pow.cpp`
   - `op_host/arch35/pow_tensor_tensor_tiling_arch35.cpp`
   - `op_host/arch35/pow_tiling_arch35.cpp`

## 3. 测试设计

本测试基于 `test_aclnn_pow_02.cpp` 实现，在一个文件中整合了 7 个 API 的测试，主要包括以下几类：

- **TensorScalar / InplacePowTensorScalar**：普通指数、立方、double 路径、空 tensor、整数负指数非法输入；
- **ScalarTensor**：普通分支、double 路径、空 tensor；
- **TensorTensor / InplacePowTensorTensor**：同 shape、广播、FLOAT16/BF16/UINT8/INT8/INT16/INT32 等多 dtype 路径；
- **Exp2 / InplaceExp2**：基础执行路径。

结果验证方式按题目要求实现：

- 使用 `std::pow(...)` 在 CPU 侧计算期望值；
- 将算子输出拷回 Host 后逐元素比对；
- 浮点类型使用容差比较；
- 每个用例输出 `[PASS]` 或 `[FAIL]`；
- 程序末尾输出 `Summary: passed=... failed=... total=...`。

## 4. 编译与运行方式

测试按题目文档中的流程执行：

```bash
bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_math/op_api/lib/:${LD_LIBRARY_PATH}
bash build.sh --run_example pow eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

## 5. 运行结果

根据 `pow2版本运行log.txt` 的实际结果：

- `passed=18`
- `failed=5`
- `total=23`

运行中出现的 5 个 FAIL 用例如下：

- `PowTensorScalar_double_aicpu_generic`
- `PowTensorScalar_empty_tensor`
- `PowScalarTensor_double_aicpu_generic`
- `PowScalarTensor_empty_tensor`
- `PowTensorTensor_double_aicpu_same_shape`

其余用例均正常输出 `[PASS]`。

## 6. 覆盖率结果

### 6.1 原始示例覆盖率

| 文件 | 原始示例 Lines executed |
|---|---:|
| `aclnn_pow.cpp` | 37.40% |
| `aclnn_pow_tensor_tensor.cpp` | 77.50% |
| `pow.cpp` | 56.67% |
| `pow_tensor_tensor_tiling_arch35.cpp` | 67.46% |
| `pow_tiling_arch35.cpp` | 68.52% |

### 6.2 当前测试文件覆盖率

| 文件 | Lines executed | Branches executed |
|---|---:|---:|
| `aclnn_pow.cpp` | 63.74% | 29.08% |
| `aclnn_pow_tensor_tensor.cpp` | 87.50% | 36.29% |
| `pow.cpp` | 80.00% | 44.68% |
| `pow_tensor_tensor_tiling_arch35.cpp` | 96.03% | 74.51% |
| `pow_tiling_arch35.cpp` | 81.48% | 39.29% |

### 6.3 相对原始示例的提升

| 文件 | 原始示例 | 当前测试文件 |
|---|---:|---:|
| `aclnn_pow.cpp` | 37.40% | 63.74% |
| `aclnn_pow_tensor_tensor.cpp` | 77.50% | 87.50% |
| `pow.cpp` | 56.67% | 80.00% |
| `pow_tensor_tensor_tiling_arch35.cpp` | 67.46% | 96.03% |
| `pow_tiling_arch35.cpp` | 68.52% | 81.48% |

## 7. 结论

当前测试文件已经满足题目对单文件整合测试、结果验证、`[PASS]/[FAIL]` 输出和汇总输出的基本要求，并且相对原始示例明显提升了题目指定的 5 个计分文件覆盖率。

从结果上看，`pow.cpp` 以及两个 tiling 文件的提升较为明显；同时当前版本仍存在 5 个 FAIL，用例主要集中在 double 路径和空 tensor 路径。因此，该测试文件已经具备较好的覆盖率提升效果，但稳定性方面仍有继续优化空间。

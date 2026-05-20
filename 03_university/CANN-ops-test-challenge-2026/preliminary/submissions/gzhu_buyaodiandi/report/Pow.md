# Pow 算子测试报告

## 1. 测试目标

本次测试针对 `Pow` 算子官方样例程序进行了比赛型增强，目标如下：

1. 在保留原始 ACL 初始化、Tensor 创建、两段式调用、同步、资源释放骨架的前提下，构建可批量执行的端到端测试程序。  
2. 将官方示例中“仅打印结果”的方式升级为“自动计算 expected 并校验”的测试程序。  
3. 尽可能提升以下关键文件的覆盖率：
   - `aclnn_pow.cpp`
   - `pow.cpp`
   - `aclnn_pow_tensor_tensor.cpp`
4. 重点补齐以下路径：
   - TensorScalar 路径
   - InplaceTensorScalar 路径
   - TensorTensor 路径
   - 广播成功与失败路径
   - 空指针、输出 dtype 不匹配、不支持 dtype 等参数校验路径
   - 特殊底数、特殊指数和边界值路径 :contentReference[oaicite:0]{index=0}

---

## 2. 测试策略

本次测试采用“正确性测试 + 非法输入测试 + 分支覆盖导向测试”相结合的策略。

### 2.1 正确性测试

对标准输入、广播输入、TensorScalar 输入、TensorTensor 输入、Inplace 输入等场景，执行如下流程：

1. 构造 host 侧输入数据  
2. 创建 ACL Tensor / ACL Scalar  
3. 调用对应的 `GetWorkspaceSize` 接口与第二段执行接口  
4. 同步 stream  
5. 将 device 结果拷回 host  
6. 在 CPU 侧计算 expected  
7. 自动比较 expected 与实际结果

### 2.2 非法输入测试

针对空指针、广播非法、输出 dtype 不匹配、不支持 dtype、不支持组合等场景，不追求数值结果，而是验证：

- API 在参数校验阶段返回 **非 `ACL_SUCCESS`**
- 不发生崩溃
- 每个 case 独立输出 `[PASS]` / `[FAIL]`

### 2.3 覆盖率导向测试

为了提升 `op_api` 与 host 相关代码覆盖率，测试设计重点加入了：

- TensorScalar 与 TensorTensor 两条主路径
- InplacePowTensorScalar 路径
- 广播成功与广播失败
- 特殊底数与特殊指数：
  - `0`
  - `1`
  - `-1`
  - 负数底数
  - 负整数指数
  - 分数指数
  - `NaN`
  - `Inf`
  - `-Inf`
- 参数异常路径：
  - `nullptr`
  - 输出 dtype 不匹配
  - 不支持 dtype
  - 非法 shape / 非法广播

---

## 3. 测试用例设计

以下为当前程序中实际保留并执行的主要测试用例。

### 3.1 TensorScalar 正确性用例

- `pow_tensor_scalar_basic`  
  基础 TensorScalar 幂运算测试，验证常规输入下的结果正确性。

- `pow_tensor_scalar_neg_zero`  
  覆盖负数、零、负零组合。

- `pow_tensor_scalar_special_values`  
  覆盖 `NaN`、`Inf`、`-Inf` 等特殊值输入。

- `pow_tensor_scalar_zero_exp`  
  指数为 0 的路径测试。

- `pow_tensor_scalar_one_exp`  
  指数为 1 的路径测试。

- `pow_tensor_scalar_negative_int_exp`  
  负整数指数路径测试。

- `pow_tensor_scalar_negative_fraction_exp`  
  负分数指数路径测试。

- `pow_tensor_scalar_base_zero_negative_exp`  
  底数为 0、指数为负数的边界路径测试。

- `pow_tensor_scalar_base_zero_fraction_exp`  
  底数为 0、指数为分数的边界路径测试。

- `pow_tensor_scalar_base_negative_noninteger_exp`  
  负数底数配合非整数指数的边界测试。

- `pow_tensor_scalar_base_negative_even_integer_exp`  
  负数底数配合偶整数指数的路径测试。

- `pow_tensor_scalar_base_negative_odd_integer_exp`  
  负数底数配合奇整数指数的路径测试。

- `pow_tensor_scalar_base_one_any_exp`  
  底数为 1 的路径测试。

- `pow_tensor_scalar_base_minus_one_large_exp`  
  底数为 -1 且指数较大时的路径测试。 :contentReference[oaicite:1]{index=1}

---

### 3.2 InplaceTensorScalar 用例

- `inplace_pow_tensor_scalar_basic`  
  覆盖 `aclnnInplacePowTensorScalar` 基础路径。

- `inplace_pow_tensor_scalar_boundary`  
  覆盖 `0`、`-0`、`1`、`-1`、`Inf` 等输入边界下的 inplace 路径。 :contentReference[oaicite:2]{index=2}

---

### 3.3 TensorTensor 用例

- `pow_tensor_tensor_basic`  
  基础 TensorTensor 幂运算路径。

- `pow_tensor_tensor_broadcast_basic`  
  覆盖 TensorTensor 广播成功场景。

- `pow_tensor_tensor_invalid_shape`  
  覆盖 TensorTensor 非法 shape / 非法广播失败路径。

- `pow_tensor_tensor_invalid_outshape`  
  覆盖输出 shape 不匹配路径。

- `pow_tensor_tensor_invalid_rank_mismatch`  
  覆盖 rank 不匹配导致的非法路径。

- `pow_tensor_tensor_invalid_dtype_combo`  
  覆盖不支持的 TensorTensor dtype 组合。 :contentReference[oaicite:3]{index=3}

---

### 3.4 空指针与参数非法用例

- `nullptr_self_invalid`  
  单独测试 self 为空。

- `nullptr_out_invalid`  
  单独测试 out 为空。

- `nullptr_exponent_invalid`  
  单独测试 exponent 为空。

- `nullptr_executor_invalid`  
  执行阶段参数非法路径测试。

- `out_dtype_mismatch_invalid`  
  输入 dtype 正常、输出 dtype 错误，验证 `op_api` 参数校验。

- `unsupported_dtype_invalid`  
  构造不支持的 dtype 场景，验证接口拒绝。 :contentReference[oaicite:4]{index=4}

---

### 3.5 其他补充路径

- `RunAclnnExp2`  
  代码中还保留了 `Exp2` 相关调用能力，用于补充指数类路径结构复用，但本轮主测试重点仍然是 `Pow` 算子。 :contentReference[oaicite:5]{index=5}

---

## 4. 结果验证方法

本次测试不再采用“仅打印结果”的方式，而是统一加入自动验证逻辑。

### 4.1 CPU 端 expected 计算

对所有正确性用例，在 host 侧计算预期结果：

- TensorScalar：`PowTensorScalarExpected`
- TensorTensor：`PowTensorTensorExpected`
- TensorTensor 广播：`BroadcastPowTensorTensorExpected`

这使得测试程序可以自动判断算子结果是否正确，而不依赖人工观察。 :contentReference[oaicite:6]{index=6}

### 4.2 float / double 容差比较

浮点结果使用 `AlmostEqual` 进行比较，支持：

- 绝对误差 `atol`
- 相对误差 `rtol`
- `NaN` 与 `NaN` 判等
- `Inf` / `-Inf` 符号一致判等

向量结果通过：

- `FloatVectorAlmostEqual`
- `DoubleVectorAlmostEqual`

进行逐元素校验。 :contentReference[oaicite:7]{index=7}

### 4.3 整数精确比较

整数类型结果使用 `ExactVectorEqual`，要求逐元素完全一致。 :contentReference[oaicite:8]{index=8}

### 4.4 invalid case 判定标准

对于非法输入类测试，不要求数值结果，统一以：

- API 返回 **非 `ACL_SUCCESS`**

作为该测试通过标准。

这类 case 的目标是覆盖参数校验与异常分支，而不是做数值正确性验证。

---

## 5. 覆盖率结果分析

根据最新覆盖率统计，结果如下：

- `aclnn_pow.cpp`：**44.66%**
- `pow.cpp`：**80.00%**
- `aclnn_pow_tensor_tensor.cpp`：**80.00%**

对应分支覆盖率如下：

- `aclnn_pow.cpp`：**27.30%**
- `pow.cpp`：**44.68%**
- `aclnn_pow_tensor_tensor.cpp`：**49.37%**

### 5.1 `aclnn_pow.cpp`

该文件是本轮测试中最主要的优化对象。  
覆盖率提升主要来自：

1. TensorScalar 边界测试：
   - `pow_tensor_scalar_zero_exp`
   - `pow_tensor_scalar_one_exp`
   - `pow_tensor_scalar_negative_int_exp`
   - `pow_tensor_scalar_negative_fraction_exp`
   - `pow_tensor_scalar_base_zero_negative_exp`
   - `pow_tensor_scalar_base_zero_fraction_exp`
   - `pow_tensor_scalar_base_negative_noninteger_exp`
   - `pow_tensor_scalar_base_negative_even_integer_exp`
   - `pow_tensor_scalar_base_negative_odd_integer_exp`
   - `pow_tensor_scalar_base_one_any_exp`
   - `pow_tensor_scalar_base_minus_one_large_exp`

2. 参数非法测试：
   - `nullptr_self_invalid`
   - `nullptr_out_invalid`
   - `nullptr_exponent_invalid`
   - `nullptr_executor_invalid`
   - `out_dtype_mismatch_invalid`
   - `unsupported_dtype_invalid`

这些 case 主要用于触发 Pow 主入口中的参数检查、特殊输入分支和异常返回路径。 :contentReference[oaicite:9]{index=9}

### 5.2 `pow.cpp`

`pow.cpp` 的覆盖率达到 **80.00% / 44.68%**，说明：

- TensorScalar 正确性路径
- 参数异常路径
- 特殊底数 / 特殊指数路径

已经较有效覆盖到主路由层逻辑。该文件在最后几轮测试中提升较明显，是本次优化的主要成果之一。

### 5.3 `aclnn_pow_tensor_tensor.cpp`

`aclnn_pow_tensor_tensor.cpp` 达到 **80.00% / 49.37%**，说明：

- 基础 TensorTensor 路径
- 广播成功路径
- 非法 shape / 非法广播路径
- TensorTensor 参数组合路径

已经覆盖较充分。

---

## 6. 不足与后续改进

### 6.1 当前不足

1. `aclnn_pow.cpp` 的覆盖率仍明显低于 `pow.cpp` 和 `aclnn_pow_tensor_tensor.cpp`，说明主入口分支仍有继续提升空间。
2. 某些 dtype 组合在当前环境下的支持情况，仍依赖具体 CANN 版本和实现。
3. 本轮重点放在 `Pow` 本体，对 `Exp2` 未做系统扩展。
4. 某些更细粒度的 executor / workspace 异常组合在当前版本中未进一步深挖。

### 6.2 后续改进方向

1. 继续补充更细粒度的 TensorScalar 参数非法组合：
   - 更复杂的 self / exponent / out dtype 组合
   - 更特殊的 scalar dtype 不匹配路径
   - 更细的 executor / workspace 异常路径

2. 若环境支持，可进一步尝试：
   - 更多 double 路径
   - 更多整数 dtype 路径
   - 更复杂广播组合

3. 若允许扩展其他指数类算子，可以进一步研究 `Exp2` 与 `Pow` 的联动覆盖策略。 :contentReference[oaicite:10]{index=10}

---

## 7. 总结

本次测试代码已从官方单样例程序演进为一套可自动执行、自动校验、覆盖多路径的 `Pow` 算子测试程序。测试覆盖了：

- TensorScalar
- InplaceTensorScalar
- TensorTensor
- 广播成功与失败
- 空指针与参数非法输入
- 特殊底数与特殊指数边界

在保持原始示例骨架不被破坏的前提下，较有效提升了 `pow.cpp` 与 `aclnn_pow_tensor_tensor.cpp` 的覆盖率，并对 `aclnn_pow.cpp` 主入口进行了针对性补强，满足比赛提交所需的测试完备性与工程可读性要求。
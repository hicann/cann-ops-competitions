# Mul 算子测试报告

## 1. 测试目标

本次测试针对 `Mul` 算子样例程序进行了比赛型增强，目标如下：

1. 在保留原始 ACL 初始化、Tensor 创建、两段式调用、同步、资源释放骨架的前提下，构建可批量执行的端到端测试程序。
2. 将“只打印结果”的官方示例升级为“自动计算 expected 并校验”的测试程序。
3. 尽可能提升以下关键文件的覆盖率：
   - `aclnn_mul.cpp`
   - `op_api/mul.cpp`
   - `mul_tiling_arch35.cpp`
4. 重点补齐以下路径：
   - 多 dtype 路径
   - mixed dtype 路径
   - 广播成功与失败路径
   - inplace / muls 路径
   - 空指针、dtype 不匹配、不支持组合等参数校验路径
   - 特殊值与边界值路径

---

## 2. 测试策略

本次测试采用“正确性测试 + 非法输入测试 + 分支覆盖导向测试”相结合的策略。

### 2.1 正确性测试

对标准输入、广播输入、mixed dtype 输入、inplace 输入等场景，执行如下流程：

1. 构造 host 侧输入数据
2. 创建 ACL Tensor
3. 调用 `aclnnMulGetWorkspaceSize` / `aclnnMul`
4. 同步 stream
5. 将 device 结果拷回 host
6. 在 CPU 侧计算 expected
7. 自动比较 expected 与实际结果

### 2.2 非法输入测试

针对空指针、广播非法、输出 dtype 不匹配、不支持 dtype、不支持 mixed dtype 等场景，不追求数值结果，而是验证：

- API 在参数校验阶段返回 **非 `ACL_SUCCESS`**
- 不发生崩溃
- 每个 case 独立输出 `[PASS]` / `[FAIL]`

### 2.3 覆盖率导向测试

为了提升 `op_api` 与 tiling 相关代码覆盖率，测试设计重点加入了：

- 冷门 dtype：`int16`、`int64`、`double`、`bool`、`int8`、`uint8`
- mixed dtype：`float16 + float32`、`float32 + float16`
- 多种广播形态：高维广播、rank 不同广播、非法 outshape
- inplace 边界输入
- scalar 路径补强
- 极端 shape：0 维标量、8 维高 rank

---

## 3. 测试用例设计

以下为当前程序中实际保留并执行的主要测试用例。

### 3.1 基础正确性用例

- `float32_basic`  
  基础 float32 逐元素乘法，保留原始样例思路并改为自动校验。

- `float32_neg_zero`  
  覆盖负数、零、负零组合。

- `int32_basic`  
  覆盖 int32 基础路径，使用精确比较。

- `double_basic`  
  覆盖 double 常规路径。

- `bool_basic`  
  覆盖 bool 路径，验证布尔输入乘法行为。

- `int8_basic`  
  覆盖 int8 路径。

- `uint8_basic`  
  覆盖 uint8 路径。

- `int16_basic`  
  补齐 int16 路径。

- `int64_basic`  
  补齐 int64 路径，包含较大整数数值。

---

### 3.2 muls / inplace 路径用例

- `float32_muls`  
  覆盖 `aclnnMuls` 基础路径。

- `muls_alt_shape_or_boundary`  
  覆盖不同 shape 与边界数值下的 `aclnnMuls`。

- `inplace_mul_basic`  
  覆盖 `aclnnInplaceMul` 基础路径。

- `inplace_mul_boundary`  
  覆盖 inplace 下的特殊边界值组合。

- `inplace_muls_basic`  
  覆盖 `aclnnInplaceMuls` 基础路径。

- `inplace_muls_zero_scalar`  
  覆盖 scalar 为 0 的 inplace muls 路径。

- `inplace_muls_boundary_case`  
  覆盖 `NaN / Inf / 0 / 极值` 等输入的 inplace muls 路径。

- `inplace_invalid_null_scalar`  
  构造空 scalar，验证 `aclnnInplaceMulsGetWorkspaceSize` 返回错误。

---

### 3.3 mixed dtype 用例

- `float16_float32_mixed`  
  `self=float16, other=float32, out=float32`

- `float32_float16_mixed`  
  `self=float32, other=float16, out=float32`

- `mixed_dtype_alt_shape_1`  
  更换 shape 与数值分布，继续覆盖 mixed dtype 路径。

- `mixed_dtype_alt_shape_2`  
  使用另一组 shape，补充 mixed dtype 参数组合覆盖。

- `unsupported_mixed_dtype_invalid`  
  构造不支持的 mixed dtype 组合，验证参数校验失败路径。

---

### 3.4 广播成功用例

- `broadcast_basic`  
  基础广播成功 case。

- `broadcast_highdim`  
  高维广播成功 case。

- `broadcast_rankdiff_a`  
  输入 rank 不同的广播成功路径。

- `broadcast_rankdiff_b`  
  另一组 rank 不同广播成功路径。

- `broadcast_scalar_like`  
  类标量广播路径。

---

### 3.5 广播失败用例

- `broadcast_invalid`  
  基础广播失败 case。

- `broadcast_invalid_extra`  
  补充另一组失败 shape 组合。

- `broadcast_invalid_rank_mismatch`  
  rank 不匹配导致广播失败。

- `broadcast_invalid_highdim`  
  高维广播非法组合。

- `broadcast_invalid_outshape`  
  输出 shape 与推导结果不一致，验证校验失败。

- `shape_mismatch_creation_invalid`  
  输入 shape 无法广播，验证接口拒绝。

---

### 3.6 特殊值与边界值用例

- `double_extreme_boundary`  
  使用 `Inf / -Inf / NaN / 极小值 / 极大值 / 正负零` 等极端 double 输入。

- `float32_special_values`  
  float32 特殊值分布测试。

- `double_special_values`  
  另一组 double 特殊值分布测试。

---

### 3.7 空指针与参数非法用例

- `nullptr_invalid`  
  传入空 self 指针，验证返回错误码。

- `nullptr_self_invalid`  
  单独测试 self 为空。

- `nullptr_other_invalid`  
  单独测试 other 为空。

- `nullptr_out_invalid`  
  单独测试 out 为空。

- `nullptr_executor_invalid`  
  workspace/executor 获取成功后，执行阶段传空 executor，验证失败分支。

- `out_dtype_mismatch_invalid`  
  输入 dtype 正常、输出 dtype 错误，验证 `op_api` 参数校验。

---

### 3.8 不支持 dtype 用例

- `unsupported_dtype_invalid`  
  使用非法枚举值 `static_cast<aclDataType>(999)` 构造不支持 dtype 场景。

- `unsupported_dtype_uint32_invalid`  
  若当前环境可创建 `ACL_UINT32` tensor，则继续验证 Mul 不支持路径。

---

### 3.9 极端 shape 用例

- `shape_zero_dim`  
  0 维标量 tensor 计算。

- `shape_extreme_high_rank`  
  8 维高 rank tensor 路径。

---

## 4. 结果验证方法

本次测试不再采用“仅打印结果”的方式，而是统一加入自动验证逻辑。

### 4.1 CPU 端 expected 计算

对所有正确性用例，在 host 侧计算预期结果：

- 普通逐元素乘法：`ElementwiseMulExpected`
- 标量乘法：`MulsExpected`
- 广播乘法：`BroadcastMulExpected`

这样可以在不依赖人工观察的情况下自动判断算子结果是否正确。

### 4.2 float / double 容差比较

浮点结果使用 `AlmostEqual` 进行比较，支持：

- 绝对误差 `atol`
- 相对误差 `rtol`
- `NaN` 与 `NaN` 判等
- `Inf` / `-Inf` 符号一致判等

float / double 向量结果通过 `FloatVectorAlmostEqual` 或 `FloatingVectorAlmostEqual` 逐元素校验。

### 4.3 int / bool 精确比较

整数和 bool 类型结果使用 `ExactVectorEqual`，要求逐元素完全一致。

### 4.4 invalid case 判定标准

对于非法输入类测试，不要求数值结果，统一以：

- API 返回 **非 `ACL_SUCCESS`**

作为该测试通过标准。

这类 case 的目标是覆盖参数校验与异常分支，而不是做数值正确性验证。

---

## 5. 覆盖率结果分析

根据最新覆盖率统计，结果如下：

- `aclnn_mul.cpp`：**72.87%**
- `mul.cpp`：**71.15%**
- `mul_tiling_arch35.cpp`：**74.51%**
- `mul_infershape.cpp`：**0.00%**

### 5.1 `aclnn_mul.cpp`

覆盖率提升的主要来源包括：

1. 空指针参数测试：
   - `nullptr_invalid`
   - `nullptr_self_invalid`
   - `nullptr_other_invalid`
   - `nullptr_out_invalid`
   - `nullptr_executor_invalid`

2. 参数组合校验测试：
   - `out_dtype_mismatch_invalid`
   - `unsupported_dtype_invalid`
   - `unsupported_dtype_uint32_invalid`
   - `unsupported_mixed_dtype_invalid`

3. 广播失败测试：
   - `broadcast_invalid`
   - `broadcast_invalid_extra`
   - `broadcast_invalid_rank_mismatch`
   - `broadcast_invalid_highdim`
   - `broadcast_invalid_outshape`

这些 case 对 `op_api` 层的参数校验、shape 检查、dtype 检查和异常返回分支覆盖贡献较大。

### 5.2 `mul.cpp`

`mul.cpp` 的覆盖率提升主要来自：

- 多 dtype 正确性路径
- mixed dtype 路径
- scalar / inplace 路径
- 特殊值输入路径

说明当前测试不仅覆盖了非法输入，也较好触发了正常算子分发路径。

### 5.3 `mul_tiling_arch35.cpp`

`mul_tiling_arch35.cpp` 达到 **74.51%**，说明：

- 不同 dtype
- 不同 shape
- 高维广播
- mixed dtype
- scalar / inplace

已经较充分地进入 tiling 相关分支。

### 5.4 `mul_infershape.cpp`

`mul_infershape.cpp` 为 **0.00%**。这并不代表测试缺失，而是与当前执行模式有关：

- 当前样例主要在 **eager 模式** 下运行
- `mul_infershape.cpp` 对应路径在该模式下提升有限

因此，本轮优化重点放在 `aclnn_mul.cpp`、`mul.cpp` 和 `mul_tiling_arch35.cpp`，`mul_infershape.cpp` 不作为当前重点突破方向。

---

## 6. 不足与后续改进

### 6.1 当前不足

1. 某些 dtype 组合在当前环境下是否稳定支持，仍依赖具体 CANN 版本与实现。
2. 复杂类型（如 complex）相关测试本轮未作为主力路径保留。
3. 当前覆盖率虽然已有明显提升，但 `aclnn_mul.cpp` 的分支覆盖率仍有继续提升空间。
4. 一些更细粒度的异常分支可能仍未被触发，例如更特殊的 workspace / executor 组合异常。

### 6.2 后续改进方向

1. 继续补充更细粒度的非法参数组合：
   - 更复杂的 out tensor 非法配置
   - 更细的 mixed dtype 非法组合
   - 更多 inplace 广播非法场景

2. 若环境支持，可进一步尝试：
   - `bfloat16` 相关 mixed dtype
   - complex 类型路径
   - 更多极端 rank / shape 组合

3. 若允许引入更完整的构图/调度路径，可尝试探索更多 infer shape 相关触发方式，以补充 `mul_infershape.cpp`。

---

## 7. 总结

本次测试代码已从单样例演进为一套可自动执行、自动校验、覆盖多分支路径的 Mul 算子测试程序。测试覆盖了：

- 基础正确性
- 多 dtype
- mixed dtype
- 广播成功与失败
- scalar / inplace
- 特殊值与边界值
- 空指针与参数非法输入

在保持原始示例骨架不被破坏的前提下，较有效提升了 `aclnn_mul.cpp`、`mul.cpp` 和 `mul_tiling_arch35.cpp` 的覆盖率，满足比赛提交所需的测试完备性与工程可读性要求。
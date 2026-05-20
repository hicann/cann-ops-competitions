# Add 算子测试报告

## 1. 测试目标

本次测试针对 `Add` 算子官方样例程序进行了比赛型增强，目标如下：

1. 在保留原始 ACL 初始化、Tensor 创建、两段式调用、同步、资源释放骨架的前提下，构建可批量执行的端到端测试程序。
2. 将官方示例中“仅打印结果”的方式升级为“自动计算 expected 并校验”的测试程序。
3. 尽可能提升以下关键文件的覆盖率：
   - `aclnn_add.cpp`
   - `add.cpp`
4. 重点补齐以下路径：
   - 基础逐元素加法路径
   - `alpha` 标量分支路径
   - 广播成功与失败路径
   - InplaceAdd 路径
   - Adds / InplaceAdds 路径
   - 空指针、输出 dtype 不匹配、不支持 dtype 等参数校验路径
   - 多 dtype 与特殊值路径

---

## 2. 测试策略

本次测试采用“正确性测试 + 非法输入测试 + 分支覆盖导向测试”相结合的策略。

### 2.1 正确性测试

对标准输入、广播输入、不同 `alpha` 输入、Inplace 输入、Adds 输入等场景，执行如下流程：

1. 构造 host 侧输入数据
2. 创建 ACL Tensor / ACL Scalar
3. 调用对应的 `GetWorkspaceSize` 接口与第二段执行接口
4. 同步 stream
5. 将 device 结果拷回 host
6. 在 CPU 侧计算 expected
7. 自动比较 expected 与实际结果

### 2.2 非法输入测试

针对空指针、广播非法、输出 dtype 不匹配、不支持 dtype、shape 非法等场景，不追求数值结果，而是验证：

- API 在参数校验阶段返回 **非 `ACL_SUCCESS`**
- 不发生崩溃
- 每个 case 独立输出 `[PASS]` / `[FAIL]`

### 2.3 覆盖率导向测试

为了提升 `op_api` 主入口与路由层覆盖率，测试设计重点加入了：

- 多种 `alpha` 分支：
  - `0`
  - `1`
  - 负数
  - 小数
  - 较大正值
  - 较大负值
  - `NaN`
  - `Inf`
  - `-Inf`
- 多 dtype 路径：
  - `float`
  - `double`
  - `int32`
  - `int16`
  - `int8`
  - `uint8`
  - `float16`（若环境可稳定支持）
- 广播成功与失败
- InplaceAdd、Adds、InplaceAdds 路径
- 空指针和参数不匹配路径 :contentReference[oaicite:0]{index=0}

---

## 3. 测试用例设计

以下为当前程序中实际保留并执行的主要测试用例类型。

### 3.1 基础正确性用例

- `add_basic`  
  基础逐元素加法路径，验证 `out = self + alpha * other` 的常规正确性。

- `add_neg_zero`  
  覆盖负数、零、负零组合。

- `add_special_values`  
  覆盖 `NaN`、`Inf`、`-Inf` 等特殊值输入。 :contentReference[oaicite:1]{index=1}

### 3.2 alpha 分支用例

- `add_alpha_zero`
- `add_alpha_one`
- `add_alpha_negative`
- `add_alpha_fraction`
- `add_alpha_large_positive`
- `add_alpha_large_negative`
- `add_alpha_nan`
- `add_alpha_inf`
- `add_alpha_neg_inf`

该类用例重点覆盖 `alpha` 不同取值下的参数分支与结果路径。 :contentReference[oaicite:2]{index=2}

### 3.3 广播用例

- `add_broadcast_basic`
- `add_broadcast_highdim`
- `add_broadcast_rankdiff_a`
- `add_broadcast_rankdiff_b`
- `add_broadcast_scalar_like`

该类用例覆盖 rank 不同、维度不同、类标量广播等路径。 :contentReference[oaicite:3]{index=3}

### 3.4 非法广播 / 非法 shape 用例

- `add_invalid_broadcast`
- `add_invalid_broadcast_extra`
- `add_invalid_outshape`
- `add_invalid_highdim_broadcast`

该类用例重点覆盖参数检查与非法 shape 路径。 :contentReference[oaicite:4]{index=4}

### 3.5 空指针与参数非法用例

- `nullptr_self_invalid`
- `nullptr_other_invalid`
- `nullptr_out_invalid`
- `nullptr_alpha_invalid`
- `nullptr_executor_invalid`
- `out_dtype_mismatch_invalid`
- `out_dtype_mismatch_invalid_extra`
- `self_other_dtype_mismatch_invalid`
- `alpha_dtype_mismatch_invalid`
- `unsupported_dtype_invalid`

该类用例主要用于覆盖 `aclnn_add.cpp` 中的参数校验和异常返回路径。 :contentReference[oaicite:5]{index=5}

### 3.6 多 dtype 用例

- `add_int32_basic`
- `add_int16_basic`
- `add_int8_basic`
- `add_uint8_basic`
- `add_double_basic`
- `add_float16_basic`（若当前环境支持）

该类用例用于扩展不同 dtype 下的执行路径。源码中也加入了：
- `DoubleVectorAlmostEqual`
- `AddExpectedIntegral`
- `FloatToHalfBits`
- `FloatsToHalfBits`
- `AddExpectedHalfBits`

用于支持多 dtype 校验。 :contentReference[oaicite:6]{index=6}

### 3.7 InplaceAdd / Adds 路径用例

- `inplace_add_basic`
- `inplace_add_alpha_zero`
- `inplace_add_alpha_negative`
- `inplace_add_broadcast`
- `inplace_add_invalid_shape`
- `inplace_add_nullptr_invalid`

同时还引入了：
- `RunAclnnInplaceAdd`
- `RunAclnnAdds`
- `RunAclnnInplaceAdds`

用于补充 Add 相关变体路径。 :contentReference[oaicite:7]{index=7}

---

## 4. 结果验证方法

### 4.1 CPU 端 expected 计算

对所有正确性用例，在 host 侧计算预期结果：

- 普通 Add：`AddExpected`
- 整数 Add：`AddExpectedIntegral`
- 标量 Adds：`AddScalarExpected`
- 广播 Add：`BroadcastAddExpected`
- float16：`AddExpectedHalfBits`

这样可以在不依赖人工观察的情况下自动判断算子结果是否正确。 :contentReference[oaicite:8]{index=8}

### 4.2 float / double 容差比较

浮点结果使用 `AlmostEqual` 进行比较，支持：

- 绝对误差 `atol`
- 相对误差 `rtol`
- `NaN` 与 `NaN` 判等
- `Inf` / `-Inf` 符号一致判等

向量结果通过：

- `FloatVectorAlmostEqual`
- `DoubleVectorAlmostEqual`

进行逐元素校验。 :contentReference[oaicite:9]{index=9}

### 4.3 整数精确比较

整数结果使用 `ExactVectorEqual`，要求逐元素完全一致。 :contentReference[oaicite:10]{index=10}

### 4.4 invalid case 判定标准

对于非法输入类测试，不要求数值结果，统一以：

- API 返回 **非 `ACL_SUCCESS`**

作为该测试通过标准。

这类 case 的目标是覆盖参数校验与异常分支，而不是做数值正确性验证。

---

## 5. 覆盖率结果分析

根据最新覆盖率统计，结果如下：

- `aclnn_add.cpp`：**76.90%**
- `add.cpp`：**55.93%**

对应分支覆盖率如下：

- `aclnn_add.cpp`：**43.04%**
- `add.cpp`：**28.03%**

### 5.1 `aclnn_add.cpp`

该文件是本轮测试中最主要的优化对象。覆盖率提升的主要来源包括：

1. `alpha` 分支测试：
   - `add_alpha_zero`
   - `add_alpha_one`
   - `add_alpha_negative`
   - `add_alpha_fraction`
   - `add_alpha_large_positive`
   - `add_alpha_large_negative`
   - `add_alpha_nan`
   - `add_alpha_inf`
   - `add_alpha_neg_inf`

2. 参数非法测试：
   - `nullptr_self_invalid`
   - `nullptr_other_invalid`
   - `nullptr_out_invalid`
   - `nullptr_alpha_invalid`
   - `nullptr_executor_invalid`
   - `out_dtype_mismatch_invalid`
   - `self_other_dtype_mismatch_invalid`
   - `alpha_dtype_mismatch_invalid`
   - `unsupported_dtype_invalid`

3. 广播失败测试：
   - `add_invalid_broadcast`
   - `add_invalid_broadcast_extra`
   - `add_invalid_outshape`
   - `add_invalid_highdim_broadcast`

这些 case 对 Add 主入口的参数检查、shape 校验、dtype 校验、`alpha` 相关分支和异常返回路径覆盖贡献较大。

### 5.2 `add.cpp`

`add.cpp` 的覆盖率达到 **55.93% / 28.03%**，说明：

- 基础 Add 路径
- 多 dtype 路径
- 广播路径
- InplaceAdd / Adds 相关路径

已经覆盖到一部分路由层逻辑，但相比 `aclnn_add.cpp` 仍有继续提升空间。

---

## 6. 不足与后续改进

### 6.1 当前不足

1. `add.cpp` 的覆盖率仍低于 `aclnn_add.cpp`，说明某些更细粒度的路由层分支还未完全触发。
2. 某些 dtype 组合在当前环境下的支持情况，仍依赖具体 CANN 版本和实现。
3. 虽然已引入 InplaceAdd、Adds、InplaceAdds 路径能力，但部分相关分支仍可能未完全打透。
4. 更复杂的广播与参数异常组合仍有继续提升空间。

### 6.2 后续改进方向

1. 继续补充更细粒度的 dtype 组合测试：
   - 更多 float16 相关路径
   - 更多整数类型混合路径
2. 继续加强 InplaceAdd / Adds / InplaceAdds 的非法路径：
   - 更复杂广播
   - 更复杂 shape 不匹配
   - 更细 executor / workspace 异常组合
3. 针对 `add.cpp` 路由层继续补充更有针对性的 case，以进一步抬升 branch 覆盖率。

---

## 7. 总结

本次测试代码已从官方单样例程序演进为一套可自动执行、自动校验、覆盖多路径的 `Add` 算子测试程序。测试覆盖了：

- 基础逐元素加法
- `alpha` 多分支路径
- 广播成功与失败
- 空指针与参数非法输入
- 多 dtype
- InplaceAdd / Adds / InplaceAdds 路径
- 特殊值与边界值

在保持原始示例骨架不被破坏的前提下，较有效提升了 `aclnn_add.cpp` 和 `add.cpp` 的覆盖率，满足比赛提交所需的测试完备性与工程可读性要求。
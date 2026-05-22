# Mul 算子测试报告

## 一、测试策略

本测试用例旨在通过多维度覆盖 Mul 算子的不同执行路径，提升代码覆盖率。测试策略包括：

1. **多种数据类型**：覆盖 FLOAT32、INT32 等不同数据类型，触发不同的 tiling 策略
2. **多种 API 变体**：覆盖 aclnnMul、aclnnMuls、aclnnInplaceMul、aclnnInplaceMuls 四个 API
3. **广播场景**：测试不同 shape 的张量相乘，触发广播处理逻辑
4. **边界条件**：测试负数、零值等边界情况
5. **异常输入**：测试空指针和不支持的数据类型，验证参数校验逻辑

## 二、测试用例设计

### 2.1 功能测试用例

| 用例名称 | API | 数据类型 | Shape | 测试目标 |
|---------|-----|---------|-------|---------|
| Mul_Float32_Basic | aclnnMul | FLOAT32 | [4,2] × [4,2] | 基本乘法运算，结果验证 |
| Mul_Float32_NegativeZero | aclnnMul | FLOAT32 | [2,2] × [2,2] | 负数和零值边界测试 |
| Mul_Float32_Broadcast | aclnnMul | FLOAT32 | [2,3] × [1,3] | 广播场景测试 |
| Mul_Int32_Basic | aclnnMul | INT32 | [4] × [4] | 整数类型乘法，覆盖不同 tiling 策略 |
| Muls_Float32 | aclnnMuls | FLOAT32 | scalar=2.5f | Tensor × Scalar，覆盖 Muls API 路径 |
| Muls_Int32 | aclnnMuls | INT32 | scalar=7 | 整数标量乘法 |
| InplaceMul_Float32 | aclnnInplaceMul | FLOAT32 | [2,2] × [2,2] | 原地乘 Tensor，覆盖 InplaceMul 路径 |
| InplaceMuls_Float32 | aclnnInplaceMuls | FLOAT32 | scalar=5.0f | 原地乘标量，覆盖 InplaceMuls 路径 |

### 2.2 异常测试用例

| 用例名称 | 测试内容 | 预期结果 |
|---------|---------|---------|
| Invalid_Nullptr | 空指针输入 | 返回非 ACL_SUCCESS 错误码 |
| Invalid_Dtype | UINT32 不支持类型 | 返回参数错误 |

### 2.3 结果验证方法

每个测试用例都包含完整的结果验证逻辑：
1. 在 CPU 端用高精度 (double) 独立计算期望值
2. 将算子输出与期望值逐元素比对
3. 浮点类型使用容差比较：`|actual - expected| ≤ atol + rtol × |expected|`
   - FLOAT32: atol=1e-5, rtol=1e-5
   - INT32: 精确匹配

## 三、覆盖率统计

### 3.1 关键文件覆盖率

| 文件 | 行覆盖率 | 分支覆盖率 |
|------|---------|-----------|
| **aclnn_mul.cpp** | 70.43% | - |
| **mul.cpp** | 57.69% | - |
| **mul_tiling_arch35.cpp** | 50.98% | - |
| **mul_infershape.cpp** | 0.00% | - |

### 3.2 综合覆盖率

根据评分公式：`综合覆盖率 = 0.6 × C_api + 0.4 × C_host`

- **C_api** = (70.43 + 57.69) / 2 = **64.06%**
- **C_host** = (50.98 + 0.00) / 2 = **25.49%**
- **综合** = 0.6 × 64.06 + 0.4 × 25.49 = **48.64%**

## 四、覆盖率分析

### 4.1 已覆盖的代码路径

通过本测试用例，成功覆盖了以下主要代码路径：

1. **aclnnMul** API 的完整调用链
2. **aclnnMuls** API 的参数校验和执行路径
3. **aclnnInplaceMul** API 的原地操作逻辑
4. **aclnnInplaceMuls** API 的原地标量乘法逻辑
5. FLOAT32 和 INT32 两种数据类型的 tiling 策略
6. 广播场景的 shape 处理逻辑
7. 参数校验分支（空指针、不支持的 dtype）

### 4.2 未覆盖的代码分析

1. **mul_infershape.cpp (0.00%)**：在 eager 模式下，shape 推断由 op_api 层内部处理，不会调用注册的 InferShape4Broadcast 函数。这是正常现象。

2. **其他未覆盖路径**：
   - 其他数据类型（FLOAT16、BFLOAT16、INT8、UINT8、INT16、INT64、DOUBLE、COMPLEX32、COMPLEX64、BOOL）
   - 混合数据类型（如 FLOAT16 × FLOAT32）
   - 更多维度的广播场景
   - 特殊浮点值（NaN、Inf）
   - 大 tensor 场景

## 五、测试结果

所有 10 个测试用例全部通过：

```
[TEST] Mul_Float32_Basic          [PASS]
[TEST] Mul_Float32_NegativeZero   [PASS]
[TEST] Mul_Float32_Broadcast      [PASS]
[TEST] Mul_Int32_Basic            [PASS]
[TEST] Muls_Float32               [PASS]
[TEST] Muls_Int32                 [PASS]
[TEST] InplaceMul_Float32         [PASS]
[TEST] InplaceMuls_Float32        [PASS]
[TEST] Invalid_Nullptr            [PASS]
[TEST] Invalid_Dtype              [PASS]

=== Summary: 0 tests failed ===
```

## 六、进一步优化建议

若需继续提升覆盖率，可考虑：

1. **增加更多数据类型**：FLOAT16、BFLOAT16、INT8、UINT8、INT64、DOUBLE 等
2. **混合类型测试**：如 FLOAT16 × FLOAT32，触发类型提升逻辑
3. **特殊浮点值**：NaN、Inf、Subnormal 等
4. **更多广播组合**：如 [3,1] × [1,4]、[2,1,3] × [1,3] 等
5. **不同 format**：ACL_FORMAT_NCHW、ACL_FORMAT_NHWC 等
6. **大 shape 测试**：触发更复杂的 tiling 切分策略

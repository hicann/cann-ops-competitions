# 题目A：Mul算子测试报告

## 一、测试概述

### 1.1 算子基本信息

| 项目 | 内容 |
|------|------|
| 算子名称 | Mul（逐元素乘法） |
| 数学定义 | y = x₁ × x₂ |
| 测试目标 | 验证算子计算正确性，提高代码覆盖率 |
| 测试文件 | test_aclnn_mul.cpp |
| 测试用例数量 | 70个 |

### 1.2 算子API说明

Mul算子提供4个API变体：

| API名称 | 功能描述 | 测试覆盖 |
|---------|---------|---------|
| `aclnnMul` | tensor × tensor | ✓ 已覆盖 |
| `aclnnMuls` | tensor × scalar | ✓ 已覆盖 |
| `aclnnInplaceMul` | 原地 tensor × tensor | ✓ 已覆盖 |
| `aclnnInplaceMuls` | 原地 tensor × scalar | ✓ 已覆盖 |

---

## 二、测试环境

### 2.1 硬件环境

| 项目 | 配置 |
|------|------|
| 计算设备 | CPU模拟器（ascend950） |
| 运行模式 | Eager模式 |

### 2.2 软件环境

| 项目 | 版本/说明 |
|------|----------|
| CANN版本 | 支持ACL运行时 |
| 编译方式 | 启用覆盖率插桩（--cov） |
| 容差配置 | FLOAT32: 1e-5, FLOAT16: 1e-3, BF16: 1e-2 |

---

## 三、测试策略设计

### 3.1 覆盖维度矩阵

本次测试设计覆盖以下五个核心维度：

```
┌─────────────────────────────────────────────────────────────┐
│                      覆盖维度矩阵                            │
├─────────────────────────────────────────────────────────────┤
│ 1. 数据类型      → 触发不同tiling策略和类型提升逻辑          │
│ 2. API变体       → 覆盖op_api层不同调度路径                 │
│ 3. Shape组合     → 覆盖广播逻辑和不同tiling分支             │
│ 4. 数值边界      → 验证特殊值处理正确性                      │
│ 5. 混合类型      → 触发IsMulMixDtypeSupport等分支           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 测试用例分类统计

| 测试类别 | 用例数量 | 占比 | 说明 |
|---------|---------|------|------|
| 基础功能测试 | 12 | 17.1% | 标准Mul、不同dtype基础测试 |
| API变体测试 | 18 | 25.7% | Muls、InplaceMul、InplaceMuls各API |
| 数据类型测试 | 15 | 21.4% | INT8/INT16/INT32/INT64/UINT8/BOOL/DOUBLE等 |
| 混合类型测试 | 10 | 14.3% | FLOAT16×FLOAT、BF16×FLOAT等组合 |
| Shape维度测试 | 10 | 14.3% | 1D~8D tensor、广播场景 |
| 边界值测试 | 5 | 7.1% | 零值、负数、Inf、NaN、极大极小值 |
| **总计** | **70** | **100%** | - |

---

## 四、详细测试用例说明

### 4.1 基础功能测试（用例1-12）

| 用例ID | 测试名称 | 测试目的 | API |
|--------|---------|---------|-----|
| 1 | Basic FLOAT32 Mul | 验证基础float32乘法 | aclnnMul |
| 2 | Broadcast Mul (vector * scalar) | 验证向量与标量广播 | aclnnMul |
| 3 | Muls API (tensor * scalar) | 验证Muls API功能 | aclnnMuls |
| 4 | InplaceMul API | 验证原地乘法API | aclnnInplaceMul |
| 5 | InplaceMuls API | 验证原地标量乘法 | aclnnInplaceMuls |
| 6 | Zero values | 验证零值处理 | aclnnMul |
| 7 | Negative values | 验证负数运算 | aclnnMul |
| 8 | INT32 data type | 验证整数类型 | aclnnMul |
| 9 | Large shape (100 elements) | 验证较大tensor | aclnnMul |
| 10 | Special values (Inf, NaN) | 验证特殊浮点值 | aclnnMul |
| 11 | 2D broadcast (matrix * vector) | 验证2D广播 | aclnnMul |
| 12 | Scalar * scalar | 验证标量运算 | aclnnMul |

### 4.2 数据类型测试（用例13-18, 45-46）

| 用例ID | 数据类型 | 测试目的 | 覆盖的tiling策略 |
|--------|---------|---------|-----------------|
| 13 | FLOAT16 | 半精度浮点 | FLOAT16专用tiling |
| 14 | BF16 | BFloat16 | BF16专用tiling |
| 15 | INT8 | 8位有符号整数 | INT8提升至INT32 |
| 16 | UINT8 | 8位无符号整数 | UINT8提升至UINT32 |
| 17 | INT64 | 64位整数 | INT64直接计算 |
| 18 | 3D tensor | 3维tensor | 多维tiling |
| 19 | 4D tensor (NCHW) | 4维tensor | NCHW格式tiling |
| 34 | INT16 | 16位整数 | INT16直接计算 |
| 35 | UINT16 | 16位无符号整数 | UINT16计算 |
| 45 | BOOL | 布尔类型 | BOOL转half计算 |
| 46 | DOUBLE | 双精度浮点 | IEEE 754手动实现 |

### 4.3 API变体测试（用例21-24, 32-33, 49-57）

**Muls API测试：**

| 用例ID | 数据类型 | 标量值 | 测试目的 |
|--------|---------|--------|---------|
| 21 | FLOAT16 | 0.5 | Muls with FLOAT16 |
| 22 | INT32 | 3 | Muls with INT32 |
| 32 | FLOAT | 0.0 | Muls with zero scalar |
| 49 | BF16 | 2.0 | 触发canUseMuls路径 |
| 51 | DOUBLE | 2.0 | Muls with DOUBLE |
| 56 | FLOAT16 | 3.0 | canUseMuls路径验证 |

**InplaceMul测试：**

| 用例ID | 数据类型 | 测试目的 |
|--------|---------|---------|
| 23 | FLOAT | InplaceMul with fractions |
| 38 | INT32 | InplaceMul with INT32 |
| 53 | DOUBLE | InplaceMul with DOUBLE |
| 54 | FLOAT16 | InplaceMul with FLOAT16 |
| 55 | BF16 | InplaceMul with BF16 |

**InplaceMuls测试：**

| 用例ID | 数据类型 | 标量值 | 测试目的 |
|--------|---------|--------|---------|
| 24 | FLOAT | -2.0 | 负标量测试 |
| 33 | FLOAT | 1.0 | 单位标量测试 |
| 50 | BF16 | 0.5 | InplaceMuls with BF16 |
| 52 | DOUBLE | 0.5 | InplaceMuls with DOUBLE |
| 57 | FLOAT16 | 2.0 | InplaceMuls with FLOAT16 |
| 62 | INT32 | 3 | InplaceMuls with INT32 |
| 63 | INT64 | 2 | InplaceMuls with INT64 |

### 4.4 混合数据类型测试（用例41-44, 47-48, 58-59）

**Mul混合类型：**

| 用例ID | self类型 | other类型 | output类型 | 触发的分支 |
|--------|---------|----------|-----------|-----------|
| 41 | FLOAT16 | FLOAT | FLOAT | IsMulMixDtypeSupport |
| 42 | FLOAT | FLOAT16 | FLOAT | 类型提升逻辑 |
| 43 | BF16 | FLOAT | FLOAT | BF16混合处理 |
| 44 | FLOAT | BF16 | FLOAT | BF16混合处理 |
| 58 | FLOAT16 | FLOAT | FLOAT | 混合类型广播 |
| 59 | BF16 | FLOAT | FLOAT | 混合类型广播 |

**InplaceMul混合类型：**

| 用例ID | self类型 | other类型 | 测试目的 |
|--------|---------|----------|---------|
| 47 | FLOAT16 | FLOAT | InplaceMul混合类型 |
| 48 | BF16 | FLOAT | InplaceMul混合类型 |

### 4.5 Shape维度测试（用例67-70）

| 用例ID | 维度 | Shape | 测试目的 |
|--------|------|-------|---------|
| 67 | 5D | [2,2,2,2,1] | 5维tensor处理 |
| 68 | 6D | [2,2,2,2,1,1] | 6维tensor处理 |
| 69 | 7D | [2,2,2,1,1,1,1] | 7维tensor处理 |
| 70 | 8D | [2,2,1,1,1,1,1,1] | 最大维度支持测试 |

### 4.6 边界值与特殊值测试

| 用例ID | 测试类型 | 测试值 | 验证内容 |
|--------|---------|--------|---------|
| 6 | 零值 | 0.0 | 零乘任何数 |
| 7 | 负数 | -1.0, -2.0 | 负数乘法 |
| 10 | 特殊值 | Inf, NaN | 特殊浮点值处理 |
| 25 | 极大值 | 1e30, 1e20 | 大数运算 |
| 26 | 极小值 | 1e-30, 1e-20 | 小数运算 |
| 65 | INT8负数 | -1, -3 | INT8负数溢出 |
| 66 | UINT8溢出 | 200*2 | UINT8溢出截断 |
| 61 | DOUBLE特殊值 | 1e100 | 双精度大数 |

---

## 五、覆盖率分析

### 5.1 需覆盖的源文件

| 层次 | 文件路径 | 有效行数 | 覆盖目标 |
|------|---------|---------|---------|
| op_api | `op_api/aclnn_mul.cpp` | 313 | 参数校验、类型提升、混合类型处理、API变体分发 |
| op_api | `op_api/mul.cpp` | 47 | AiCore/AiCpu路由、dtype支持判断 |
| op_host | `op_host/arch35/mul_tiling_arch35.cpp` | 192 | 16种dtype组合的tiling策略分发 |

### 5.2 预期覆盖的关键分支

**op_api/aclnn_mul.cpp 分支覆盖：**

| 分支类型 | 触发条件 | 覆盖用例 |
|---------|---------|---------|
| 空指针检查 | 输入为nullptr | 参数校验逻辑 |
| dtype支持检查 | 不支持的dtype组合 | 用例8,15,16,17,34,35 |
| 混合dtype处理 | FLOAT16×FLOAT等 | 用例41-44,58-59 |
| Muls优化路径 | canUseMuls条件 | 用例49,56 |
| Inplace操作 | 原地API调用 | 用例4,5,23-24 |

**op_api/mul.cpp 分支覆盖：**

| 分支类型 | 触发条件 | 覆盖用例 |
|---------|---------|---------|
| AiCore路由 | 标准dtype | 用例1,8,13,14 |
| AiCpu路由 | DOUBLE类型 | 用例46,51,52,53 |
| dtype支持判断 | IsMulMixDtypeSupport | 用例41-44 |

**op_host/arch35/mul_tiling_arch35.cpp 分支覆盖：**

| dtype组合 | 对应OP_KEY | 覆盖用例 |
|----------|-----------|---------|
| FLOAT | MUL_FLOAT | 用例1 |
| FLOAT16 | MUL_FLOAT16 | 用例13 |
| BF16 | MUL_BF16 | 用例14 |
| INT32 | MUL_INT32 | 用例8 |
| INT8 | MUL_INT8 | 用例15,65 |
| UINT8 | MUL_UINT8 | 用例16,66 |
| INT64 | MUL_INT64 | 用例17 |
| INT16 | MUL_INT16 | 用例34 |
| BOOL | MUL_BOOL | 用例45,60 |
| DOUBLE | MUL_DOUBLE | 用例46 |

### 5.3 混合类型覆盖矩阵

| self | other | output | 覆盖状态 | 用例ID |
|------|-------|--------|---------|--------|
| BF16 | BF16 | BF16 | ✓ | 用例14 |
| BF16 | FLOAT | FLOAT | ✓ | 用例43 |
| FLOAT | BF16 | FLOAT | ✓ | 用例44 |
| FLOAT16 | FLOAT16 | FLOAT16 | ✓ | 用例13 |
| FLOAT16 | FLOAT | FLOAT | ✓ | 用例41 |
| FLOAT | FLOAT16 | FLOAT | ✓ | 用例42 |
| FLOAT | FLOAT | FLOAT | ✓ | 用例1 |
| INT32 | INT32 | INT32 | ✓ | 用例8 |
| UINT8 | UINT8 | UINT8 | ✓ | 用例16 |
| INT8 | INT8 | INT8 | ✓ | 用例15 |
| INT64 | INT64 | INT64 | ✓ | 用例17 |
| INT16 | INT16 | INT16 | ✓ | 用例34 |
| BOOL | BOOL | BOOL | ✓ | 用例45 |
| DOUBLE | DOUBLE | DOUBLE | ✓ | 用例46 |

**覆盖率：14/16 = 87.5%**

未覆盖的组合：
- COMPLEX32（代码未实现相应测试）
- COMPLEX64（代码未实现相应测试）

---

## 六、测试结果验证方法

### 6.1 结果验证流程

```
┌──────────────────────────────────────────────────┐
│              结果验证流程                          │
├──────────────────────────────────────────────────┤
│  1. 在CPU端用高精度(double)计算期望值              │
│  2. 从device端拷贝实际计算结果                     │
│  3. 根据dtype选择容差或精确比较                    │
│  4. 逐元素比对并输出[PASS]/[FAIL]                 │
│  5. 统计总测试数、通过数、失败数                   │
└──────────────────────────────────────────────────┘
```

### 6.2 容差比较公式

对于浮点类型，使用相对容差和绝对容差组合判断：

```
|actual - expected| ≤ atol + rtol × |expected|
```

| 数据类型 | 绝对容差(atol) | 相对容差(rtol) |
|---------|---------------|---------------|
| FLOAT | 1e-5 | 1e-5 |
| FLOAT16 | 1e-3 | 1e-3 |
| BF16 | 1e-2 | 1e-2 |
| 整数类型 | 0 | 0 (精确匹配) |

### 6.3 特殊值处理

- **NaN比较：** 两个NaN视为相等
- **Inf比较：** 同号Inf视为相等
- **整数溢出：** 验证饱和截断行为

---

## 七、编译运行说明

### 7.1 编译命令

```bash
# 步骤1：编译算子包（启用覆盖率）
bash build.sh --pkg --soc=ascend950 --ops=mul \
    --vendor_name=custom --cov

# 步骤2：安装算子包
./build_out/cann-ops-math-custom_linux-x86_64.run

# 步骤3：运行测试
bash build.sh --run_example mul eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

### 7.2 预期输出格式

```
=== Starting Mul operator tests ===
[PASS] Basic FLOAT32 Mul
[PASS] Broadcast Mul (vector * scalar)
[PASS] Muls API (tensor * scalar)
...
[PASS] 8D tensor (max dims)

=== Test Summary ===
Total tests: 70
Passed: 70
Failed: 0
```

### 7.3 查看覆盖率

```bash
# 查找gcda文件
find build -name "*.gcda" | grep mul

# 生成覆盖率报告
gcov -b <gcno文件路径>
```

---

## 八、测试亮点与创新点

### 8.1 测试设计亮点

1. **全API覆盖** - 4个API变体全部测试，确保op_api层各调度路径被覆盖
2. **混合类型系统化** - 系统测试所有合法的混合dtype组合，触发类型提升逻辑
3. **维度边界测试** - 测试1D到8D（最大维度）全覆盖
4. **特殊值完整** - Inf、NaN、零值、负数、极大极小值全面覆盖
5. **广播场景多样** - 标量广播、向量广播、矩阵广播等多种场景

### 8.2 与官方示例的改进

| 改进项 | 官方示例 | 本次测试 |
|--------|---------|---------|
| 结果验证 | 仅打印结果 | 完整的期望值比对 |
| 数据类型 | 仅FLOAT | 11种dtype全覆盖 |
| API覆盖 | 仅aclnnMul | 4个API全覆盖 |
| 测试数量 | 1个用例 | 70个用例 |
| 边界测试 | 无 | 特殊值、极大极小值 |
| 维度测试 | 仅2D | 1D~8D全覆盖 |

---

## 九、总结

### 9.1 测试完成情况

| 项目 | 状态 |
|------|------|
| 基础功能测试 | ✓ 完成 |
| API变体测试 | ✓ 完成 |
| 数据类型测试 | ✓ 完成（11/13种dtype） |
| 混合类型测试 | ✓ 完成（14/16种组合） |
| Shape维度测试 | ✓ 完成（1D~8D） |
| 边界值测试 | ✓ 完成 |
| 结果验证逻辑 | ✓ 完成 |

### 9.2 预期覆盖率

- **op_api/aclnn_mul.cpp：** 预计 > 90%
- **op_api/mul.cpp：** 预计 > 95%
- **op_host/arch35/mul_tiling_arch35.cpp：** 预计 > 85%

### 9.3 未覆盖项

1. COMPLEX32类型（未实现测试）
2. COMPLEX64类型（未实现测试）
3. 部分异常输入分支（nullptr、非法shape等）

### 9.4 后续优化建议

1. 补充COMPLEX类型的测试用例
2. 增加异常输入测试（预期返回错误码）
3. 增加非连续stride的tensor测试
4. 根据实际覆盖率报告针对性补充用例

---

**报告生成时间：** 2025年

**测试代码文件：** test_aclnn_mul.cpp（70个测试用例）

**测试状态：** 就绪，等待编译运行验证

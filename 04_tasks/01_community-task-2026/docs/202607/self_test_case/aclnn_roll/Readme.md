# Roll 算子测试用例

## 算子信息

- 算子名称：Roll
- 算子语言：cpp
- 功能描述：沿指定维度滚动张量元素

## 输入参数

| 参数名 | 类型 | 格式 | 说明 |
|--------|------|------|------|
| x | bfloat16/float16/float32/int8/uint8/int32/uint32/bool/complex64 | ND | 输入张量 |

## 属性参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| shifts | list_int | 滚动偏移量列表，正数向右滚，负数向左滚 |
| dims | list_int | 滚动维度列表，为空时表示所有维度 |

## 输出参数

| 参数名 | 类型 | 格式 | 说明 |
|--------|------|------|------|
| out | bfloat16/float16/float32/int8/uint8/int32/uint32/bool/complex64 | ND | 输出张量，与输入 x 同形状同类型 |

## 文件结构

```
Roll/
├── op.json                 # 完整算子定义（含 complex64）
├── op_basic.json           # 基础算子定义（不含 complex64）
├── op_complex.json         # complex64 专用算子定义
├── case.json               # 完整测试用例（50个）
├── case_basic.json         # 基础测试用例（50个，不含 complex64）
├── case_complex64_100.json # complex64 专用测试用例（100个）
├── golden.py               # 期望结果计算函数
├── constraint_condition.py # 约束条件
├── gen_case.sh             # 用例生成脚本
└── Readme.md               # 本说明文件
```

## 测试流程

### 1. 生成测试用例（可选）

```bash
# 使用脚本生成所有测试用例
bash gen_case.sh

# 或单独生成特定类型的用例
# 基础用例（不含 complex64）
python ../scripts/gen_case.py -i op_basic.json -o case_basic.json -n 50 -c constraint_condition.py

# complex64 专用用例
python ../scripts/gen_case.py -i op_complex.json -o case_complex64_100.json -n 100 -c constraint_condition.py

# 完整用例（含 complex64）
python ../scripts/gen_case.py -i op.json -o case.json -n 50 -c constraint_condition.py
```

### 2. 测试自定义算子（ACL）

```bash
# 编译部署自定义算子
# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_math/op_api/lib:$LD_LIBRARY_PATH

# 测试自定义部署算子
python3 run_test.py -i Roll/op.json -c Roll/case_complex64_100.json \
  --op-type "custom" \
  --op-path "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_math/op_api" \
  --msprof \
  --op \
  -d prof_roll_acl \
  --build \
  -n Test_001
```

## 测试用例说明

### 基础用例（case_basic.json）

50 个测试用例，覆盖以下场景, 不含 complex64：

- **数据类型覆盖**：bfloat16, float16, float32, int8, uint8, int32, uint32, bool
- **形状覆盖**：从 1D 到多维张量
- **滚动方向**：正数（向右）、负数（向左）
- **维度指定**：指定维度、空维度列表（所有维度）
- **边界测试**：超大偏移量、负维度索引

### Complex64 专用用例（case_complex64_100.json）

100 个测试用例，专门测试 complex64 数据类型：

- **数据类型**：complex64
- **形状覆盖**：各种形状组合
- **滚动参数**：多种 shifts 和 dims 组合

## 注意事项

1. 测试前请确保已安装 ascendOptest 工具
2. 确保 Ascend 工具链环境已正确配置
3. golden.py 中的期望函数路径可能需要根据实际环境调整
4. 测试用例中的 expect_func 路径为 `/root/optest/Roll/golden.py:calc_expect_func`
5. shifts 参数可以是任意整数，会自动取模处理
6. dims 参数为空列表时表示在所有维度上滚动
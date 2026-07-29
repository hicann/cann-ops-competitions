# FloorMod 算子测试用例

## 算子信息

- 算子名称：FloorMod
- 算子定义：RemainderTensorTensor
- 算子语言：cpp
- 功能描述：计算 x1 % x2，结果与 x2 同号（向下取整除法的余数）

## 输入参数

| 参数名 | 类型 | 格式 | 说明 |
|--------|------|------|------|
| x1 | bfloat16/float16/float32/int64/int32 | ND | 被除数 |
| x2 | bfloat16/float16/float32/int64/int32 | ND | 除数，不能为 0 |

## 输出参数

| 参数名 | 类型 | 格式 | 说明 |
|--------|------|------|------|
| y | bfloat16/float16/float32/int64/int32 | ND | 余数，与 x2 同号 |

## 文件结构

```
FloorMod/
├── op.json           # 算子定义文件
├── case.json         # 测试用例文件（50个用例）
├── golden.py         # 期望结果计算函数
├── constraint_condition.py  # 约束条件
├── gen_case.sh       # 用例生成脚本
├── all_prof.csv      # 性能分析结果
└── Readme.md         # 本说明文件
```

## 测试流程

### 1. 生成测试用例（可选）

```bash
# 使用脚本生成测试用例
bash gen_case.sh

# 或直接运行生成命令
python ../scripts/gen_case.py -i op.json -o case.json -n 50 -c constraint_condition.py
```

### 2. 测试内置算子（TBE）

```bash
# 删除自定义部署，测试内置算子
python3 run_test.py -i FloorMod/op.json -c FloorMod/case.json \
  --op-type "builtin" \
  --op-path "/usr/local/Ascend/ascend-toolkit/latest/lib64/" \
  --build \
  --msprof \
  --op \
  -d prof_floormod_tbe \
  -n Test_001
```
去除-n Test_001 参数可测试全部用例

### 3. 测试自定义算子（ACL）

```bash
# 编译部署自定义算子
# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_math/op_api/lib:$LD_LIBRARY_PATH

# 测试自定义部署算子
python3 run_test.py -i FloorMod/op.json -c FloorMod/case.json \
  --op-type "custom" \
  --op-path "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_math/op_api" \
  --msprof \
  --op \
  -d prof_floormod_acl \
  --build \
  -n Test_001
```
去除-n Test_001 参数可测试全部用例

### 4. 性能对比分析

```bash
# 对比 TBE 和 ACL 的性能数据
python3 get_prof.py -c ../prof_floormod_acl -b ../prof_floormod_tbe/ -f ../FloorMod/case.json -d out
```

## 测试用例说明

共包含 50 个测试用例，覆盖以下场景：

- **数据类型覆盖**：bfloat16, float16, float32, int64, int32
- **形状覆盖**：从 1D 到多维张量，包含广播场景
- **值范围**：
  - x1: [-100.0, 100.0]
  - x2: [-100.0, -0.001]（负数，避免除零）
- **边界测试**：特殊形状组合和广播情况

## 注意事项

1. 测试前请确保已安装 ascendOptest 工具
2. 确保 Ascend 工具链环境已正确配置
3. golden.py 中的期望函数路径可能需要根据实际环境调整
4. 测试用例中的 expect_func 路径为 `/root/optest/FloorMod/golden.py:calc_expect_func`
5. x2 参数不允许为 0，测试用例中 x2 的值范围已限制为负数
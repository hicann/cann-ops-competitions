# Bernoulli 算子测试用例

## 算子信息

- 算子名称：Bernoulli
- 算子语言：cpp
- 功能描述：根据给定的概率 prob 生成伯努利分布随机数

## 输入参数

| 参数名 | 类型 | 格式 | 说明 |
|--------|------|------|------|
| self | float16/float32/double/uint8/int8/int16/int32/int64/bool/bfloat16 | ND | 输入张量，决定输出形状和类型 |
| prob | float16/float32/double/bfloat16 | ND | 概率值，范围 [0.0, 1.0] |

## 属性参数

| 参数名 | 类型 | 值范围 | 说明 |
|--------|------|--------|------|
| seed | int | [0, 1000000] | 随机种子 |
| offset | int | [0, 256] | 偏移量 |

## 输出参数

| 参数名 | 类型 | 格式 | 说明 |
|--------|------|------|------|
| out | float16/float32/double/uint8/int8/int16/int32/int64/bool/bfloat16 | ND | 输出张量，与输入 self 同形状同类型 |

## 文件结构

```
Bernoulli/
├── op.json           # 算子定义文件
├── case.json         # 测试用例文件（50个用例）
├── golden.py         # 期望结果计算函数
├── constraint_condition.py  # 约束条件
├── gen_case.sh       # 用例生成脚本
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
python3 run_test.py -i Bernoulli/op.json -c Bernoulli/case.json \
  --op-type "builtin" \
  --op-path "/usr/local/Ascend/ascend-toolkit/latest/lib64/" \
  --build \
  --msprof \
  -d prof_bernoulli_tbe \
  -n Test_001
```
去除-n Test_001 参数可测试全部用例

### 3. 测试自定义算子（ACL）

```bash
# 编译部署自定义算子
# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_math/op_api/lib:$LD_LIBRARY_PATH

# 测试自定义部署算子
python3 run_test.py -i Bernoulli/op.json -c Bernoulli/case.json \
  --op-type "custom" \
  --op-path "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_math/op_api" \
  --msprof \
  -d prof_bernoulli_acl \
  --build \
  -n Test_001
```
去除-n Test_001 参数可测试全部用例

### 4. 性能对比分析

```bash
# 获取 TBE 和 ACL 的性能数据
python3 get_time_app.py ../prof_bernoulli_tbe ./result_tbe.csv
python3 get_time_app.py ../prof_bernoulli_acl ./result_acl.csv
```

## 测试用例说明

共包含 50 个测试用例，覆盖以下场景：

- **数据类型覆盖**：float16, float32, double, uint8, int8, int16, int32, int64, bool, bfloat16
- **形状覆盖**：从 1D 到多维张量
- **概率范围**：[0.0, 1.0]
- **特殊值测试**：seed=0, offset=0

## 注意事项

1. 测试前请确保已安装 ascendOptest 工具
2. 确保 Ascend 工具链环境已正确配置
3. golden.py 中的期望函数路径可能需要根据实际环境调整
4. 测试用例中的 expect_func 路径为 `/root/optest/Bernoulli/golden.py:calc_expect_func`
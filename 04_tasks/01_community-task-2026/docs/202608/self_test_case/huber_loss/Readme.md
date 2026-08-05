# HuberLoss 算子测试用例

## 算子信息

- 算子名称：HuberLoss
- 算子语言：cpp
- 功能描述：计算 Huber Loss，误差 `e = input - target`，当 `|e| <= delta` 时 `loss = 0.5 * e^2`，当 `|e| > delta` 时 `loss = delta * (|e| - 0.5 * delta)`；支持 reduction 模式（none / mean / sum）

## 输入参数

| 参数名 | 类型 | 格式 | 说明 |
|--------|------|------|------|
| input | bfloat16/float16/float32 | ND | 预测值张量，任意维度 |
| target | bfloat16/float16/float32 | ND | 目标值张量，shape 和 dtype 须与 input 一致 |

## 属性参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| reduction | int | 归约模式，默认 1。0=none（逐元素输出），1=mean（取均值），2=sum（求和） |
| delta | float | Huber loss 阈值参数，默认 1.0，须大于 0 |

## 输出参数

| 参数名 | 类型 | 格式 | 说明 |
|--------|------|------|------|
| output | bfloat16/float16/float32 | ND | loss 计算结果，reduction=none 时与 input 同 shape；reduction=mean/sum 时为标量 [1] |

## 文件结构

```
HuberLoss/
├── op.json                 # 算子定义
├── case.json               # 测试用例（50个）
├── golden.py               # 期望结果计算函数（基于 PyTorch 标杆）
├── constraint_condition.py # 约束条件
├── gen_case.sh             # 用例生成脚本
└── Readme.md               # 本说明文件
```

## 测试流程

### 1. 生成测试用例（可选）

```bash
# 使用脚本生成测试用例
bash gen_case.sh

# 或单独生成
python ../scripts/gen_case.py -i op.json -o case.json -n 100 -c constraint_condition.py
```

### 2. 配置环境

```bash
# 配置 Ascend 工具链环境
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

### 3. 运行测试

注意：因缺少算子实现，该用例未经过实际测试，执行过程中可自行调试。
精度比对基于 PyTorch 的 `torch.nn.functional.huber_loss` 作为标杆。

```bash
# 运行单个测试用例
python3 run_test.py -i HuberLoss/op.json -c HuberLoss/case.json \
  --op-type "custom" \
  --op-path "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_nn/op_api" \
  --msprof \
  --op \
  -d prof_huber_loss \
  -n Test_001

# 运行全部测试用例（去掉 -n 参数）
python3 run_test.py -i HuberLoss/op.json -c HuberLoss/case.json \
  --op-type "custom" \
  --op-path "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_nn/op_api" \
  --msprof \
  --op \
  -d prof_huber_loss
```

## 测试用例说明

100 个测试用例，覆盖以下场景：

- **数据类型覆盖**：bfloat16、float16、float32
- **形状覆盖**：从 1D 到多维张量
- **reduction 模式**：none（逐元素）、mean（均值）、sum（求和）
- **delta 参数**：0.5、1.0、2.0、5.0 等多种阈值
- **输出 shape**：reduction=none 时与输入同 shape，reduction=mean/sum 时为 [1]

## 标杆实现说明

golden.py 使用 `torch.nn.functional.huber_loss` 作为标杆，处理逻辑：

1. 根据 numpy 数据类型转换为对应的 torch tensor（bfloat16/float16/float32）
2. 半精度类型（float16/bfloat16）内部提升为 float32 计算，避免精度损失
3. 计算完成后转回原始 torch 类型
4. 最终转回 numpy 原始数据类型输出

## 注意事项

1. 测试前请确保已安装 AscendOpTest 工具
2. 确保 Ascend 工具链环境已正确配置（`source set_env.sh`）
3. golden.py 中的期望函数路径可能需要根据实际环境调整
4. delta 参数必须大于 0，否则为非法输入
5. input 与 target 必须具有相同的 shape 和 dtype，不支持 broadcast

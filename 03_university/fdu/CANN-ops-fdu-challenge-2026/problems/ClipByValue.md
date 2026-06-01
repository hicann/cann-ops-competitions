## 一、赛题背景
ClipByValue 是数值裁剪算子，将张量元素限制在 `[min, max]` 范围内，广泛用于梯度裁剪、数值稳定处理等场景。本题要求基于 `torch.clamp` 的核心语义，使用 Ascend C 在昇腾 NPU 上实现高性能、高精度的 ClipByValue 算子。

## 二、算子功能描述
将输入张量 `x` 中的每个元素裁剪到 `[min, max]` 区间：小于 `min` 则输出 `min`，大于 `max` 则输出 `max`，否则保持原值。

## 三、核心定义与约束

### 3.1 参考算子
`torch.clamp`  
参考文档：https://pytorch.org/docs/stable/generated/torch.clamp.html

### 3.2 算子输入输出规范

| 类型 | 参数名 | 数据类型 / 类型 | 形状 | 说明 |
|------|--------|----------------|------|------|
| INPUT | x | float16, float32, int32 | 任意形状 | 待裁剪张量 |
| ATTR | min | float | 标量 | 裁剪下界 |
| ATTR | max | float | 标量 | 裁剪上界，需满足 min ≤ max |
| OUTPUT | y | float16, float32, int32 | 同 x | 裁剪结果，类型与 x 相同 |

### 3.3 核心约束
- `min` 和 `max` 均为标量，对所有元素生效。
- 要求 `min ≤ max`，平台生成的测试数据将保证此条件。
- 输出形状与输入 `x` 完全一致，数据类型相同。

### 3.4 计算规则
$$y = \text{clamp}(x, \text{min}, \text{max})$$
逐元素：若 $x < \text{min}$ 则取 $\text{min}$，若 $x > \text{max}$ 则取 $\text{max}$，否则取 $x$。

### 3.5 精度要求
- float32 / float16：裁剪操作不引入额外计算误差，结果必须与参考实现逐元素完全一致（精确比对）。
- int32：结果完全准确，无误差。

### 3.6 输出要求
- 形状与 `x` 相同，数据类型与 `x` 相同。
- 所有元素值均在 `[min, max]` 范围内。
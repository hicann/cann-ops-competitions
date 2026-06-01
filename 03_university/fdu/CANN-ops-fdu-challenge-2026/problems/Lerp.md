## 一、赛题背景
Lerp（线性插值）广泛用于参数融合、特征混合等场景。本题基于 `torch.lerp` 语义，使用 Ascend C 在昇腾 NPU 上实现高性能 Lerp 算子。

## 二、算子功能描述
对形状相同的两个张量 `start` 和 `end`，按标量权重 `weight` 逐元素计算线性插值：`y = start + weight * (end - start)`。

## 三、核心定义与约束

### 3.1 参考算子
`torch.lerp`  
参考文档：https://pytorch.org/docs/stable/generated/torch.lerp.html

### 3.2 算子输入输出规范

| 类型 | 参数名 | 数据类型 / 类型 | 形状 | 说明 |
|------|--------|----------------|------|------|
| INPUT | start | float16, float32 | 任意形状 | 起始张量 |
| INPUT | end | float16, float32 | 与 start 相同 | 结束张量 |
| ATTR | weight | float | 标量 | 插值权重（可为任意实数） |
| OUTPUT | y | 同输入类型 | 同 start/end | 插值结果 |

### 3.3 核心约束
- `start` 和 `end` 形状必须完全相同，不涉及广播。
- 输出形状、类型与输入一致。
- 当 `weight=0` 输出 `start`，`weight=1` 输出 `end`。

### 3.4 计算公式
$$y = \text{start} + weight \times (\text{end} - \text{start})$$

### 3.5 精度要求
- float32：相对误差 < 1e-4，绝对误差 < 1e-4  
- float16：相对误差 < 1e-3，绝对误差 < 1e-3

### 3.6 输出要求
- 形状、类型与输入 `start` 相同。
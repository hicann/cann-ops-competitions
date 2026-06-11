## 个人信息

- 姓名：谭霍睿之
- 学号：23307130423
- 联系邮箱：13916019730@163.com
- CANNJudge 账号：sherry

## CANNJudge 提交说明

比赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026

最终提交时间：6月5号19：16

三道题完成情况（附提交成功截图）：

- Addcmul：

![Addcmul 提交结果](img/addcmul.png)

- ClipByValue：

![ClipByValue 提交结果](img/clipbyvalue.png)

- Lerp：

![Lerp 提交结果](img/Lerp.png)

## 算子实现简介

### Addcmul

实现 `input_data + x1 * x2 * value` 计算，支持多数据类型、非 32B 对齐和广播场景。主要优化是按输入场景选择不同 kernel 路径，连续场景使用向量化搬运和计算，广播场景减少通用索引开销。

### ClipByValue

实现按 `min`、`max` 对输入张量裁剪。主体使用 `DataCopy + Maxs + Mins + DataCopy` 向量计算，尾部单独处理，减少非对齐场景的额外开销。

### Lerp

实现线性插值计算，按输入规模进行多核切分和 UB 分块。主体路径使用向量化搬运与计算，尾部按实际有效元素处理，保证非对齐场景正确。

# 复旦 CANN 校内自办赛 2026 个人提交

## 个人信息

- 姓名：朱先立
- 学号：25210180115
- 联系邮箱：25210180115@m.fudan.edu.cn
- CANNJudge 账号：hstral

## CANNJudge 提交情况

- 比赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026
- 目标芯片：Ascend 910B
- 锁榜时间：2026-06-05 21:00
- 最终提交：Addcmul 2026-06-05 10:08:29，ClipByValue 2026-06-05 20:42:57，Lerp 2026-06-05 20:39:11

| 赛题 | 提交 ID | 状态 | 最终分数 | 最终排名 |
| --- | ---: | --- | ---: | ---: |
| Addcmul | 66557 | Pass | 58.06 | 5 |
| ClipByValue | 68682 | Pass | 83.94 | 1 |
| Lerp | 68662 | Pass | 52.83 | 11 |

三道题均已在 CANNJudge 平台完成提交并通过评测。截榜测试点耗时如下：

### Addcmul

| 测试点 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 用时 | 5.22us | 6.54us | 6.00us | 5.28us | 23.78us | 5.46us | 8.14us | 3.90us | 13.54us | 9.10us | 14.52us |

### ClipByValue

| 测试点 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 用时 | 3.08us | 13.94us | 6.16us | 4.16us | 8.42us |

### Lerp

| 测试点 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 用时 | 3.50us | 2.16us | 3.52us | 3.06us | 4.42us | 4.28us | 4.96us |

## 目录结构

```text
submissions/25210180115_zhuxianli/
|-- README.md
`-- code/
    |-- Addcmul/
    |-- ClipByValue/
    `-- Lerp/
```

`code/` 下为三道题可复现最终 CANNJudge 提交结果的工程代码。

## 算子实现简介

### Addcmul

实现 `y = input_data + x1 * x2 * value`，支持多 dtype 和广播语义。实现中在 Host 侧按广播形态选择连续、标量、inner contiguous、inner broadcast 和 general 等路径；Kernel 侧尽量使用向量化计算与连续搬运。针对 `int8` 路径保留回绕语义，避免饱和转换导致与参考结果不一致。最终版本结合 Ascend 910B 上的测试反馈调整不同路径的启核策略，减少小规模场景开销并提升大规模广播场景稳定性。

### ClipByValue

实现 `y = min(max(x, min), max)`，支持 `float16`、`float32`、`int32`。实现中根据输入规模自适应选择核数和 tile 大小，主路径使用 Ascend C 向量化 `Maxs/Mins` 完成上下界裁剪；非对齐尾块使用 `DataCopyPad` 处理，保证任意长度输入的正确性。最终版本结合 910B 实测结果采用规模阶梯核数策略，降低小张量固定开销，同时保持大张量并行度。

### Lerp

实现 `y = start + weight * (end - start)`，支持 `float16`、`float32`。实现中保持与 `torch.lerp` 一致的计算公式，避免 fp16 下代数改写带来的精度偏差；对 `weight == 0` 和 `weight == 1` 设置拷贝快路径；对齐场景使用轻量 `DataCopy`，非对齐尾块使用 `DataCopyPad`。最终版本根据 910B 反馈调整小/中/大张量核数和 tile 策略，在启动开销和带宽利用之间折中。

## 个人信息

- 姓名：郑予谦
- 学号：25803050051
- 联系邮箱：25803050051@m.fudan.edu.cn
- CANNJudge 账号：jiangtaizhe001

## CANNJudge 提交说明
比赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026

最终提交时间：2026/06/05 15:39:39

三道题完成情况（附提交成功截图）：

Addcmul：10/11

| 3.94μs | 5.70μs | 4.26μs | -    | 19.32μs | 4.94μs | 7.30μs | 4.98μs | 3.32μs | 2.78μs | 3.28μs |
| ------ | ------ | ------ | ---- | ------- | ------ | ------ | ------ | ------ | ------ | ------ |

ClipByValue：5/5

| 3.78μs | 14.16μs | 6.68μs | 2.94μs | 8.36μs |
| ------ | ------- | ------ | ------ | ------ |

Lerp：7/7

| 4.00μs | 3.74μs | 2.18μs | 4.20μs | 2.34μs | 2.66μs | 3.32μs |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |

## 算子实现简介

### T1 Addcmul

Addcmul 算子实现的是 `y = input_data + x1 * x2 * value`。实现时将输入张量按一维连续内存处理，Host 侧完成 shape 推导、dtype 分发、blockDim 选择和 tiling 计算；Kernel 侧每个 core 负责一段连续数据，在 UB 内依次完成 `Mul + Muls + Add`，最后只写回一次输出。

性能优化的重点是算子融合，避免乘法和加法中间结果回写 GM，从而减少全局内存访问。对于大数据使用多核并行和双缓冲提升吞吐；对于非 32B 对齐的尾块，使用 `DataCopyPad` 保证数据搬运正确。实现中遇到的主要问题是题面要求支持广播和 int8，但完整广播会引入较大分支开销，因此主路径按 OJ 中较常见的 contiguous 情况优化；int8 分支则采用先转 half 计算、再转回 int8 的方式，在正确性和性能之间做了取舍。

### T2 ClipByValue

ClipByValue 算子实现的是 `y = min(max(x, min), max)`，即将输入张量限制在给定上下界之间。Host 侧读取输入 shape、dtype 以及 `min / max` 属性，并根据数据规模计算 blockDim、perCoreLength 和 tileLength；Kernel 侧每个 tile 先将数据从 GM 搬入 UB，再通过 `Maxs + Mins` 两条向量指令完成裁剪，最后写回 GM。

该题的计算逻辑较简单，主要瓶颈在 GM 读写、核数调度和 launch 开销。因此优化重点放在 blockDim、tile size 和 buffer 配置的实验选择上。大数据下通过多核并行可以明显提升带宽利用率，小数据下则需要避免过多核数带来的调度开销。实现过程中发现 double buffer 并不总是更快，在部分数据场景中可能因为 GM 带宽压力或调度开销导致性能下降。

### T3 Lerp

Lerp 算子实现的是 `y = start + weight * (end - start)`，支持 float16 和 float32。实现时同样将输入张量扁平化为一维连续数组，Host 侧下发 length、tileLength、perCoreLength 和 weight；Kernel 侧每个 tile 读入 start 和 end，在 UB 内先计算 `end - start`，再使用 `Axpy` 完成 `start + weight * diff`，最后将结果写回输出。

该题的优化重点不是带宽，而是减少小算子的指令、同步和管理开销。因此将公式改写为 `Sub + Axpy`，避免显式使用 `Muls + Add`，同时使用 TBuf 降低队列管理成本。对于非对齐尾块仍然使用 `DataCopyPad` 处理。遇到的主要问题是小 shape 下耗时主要由 launch、scalar 和同步开销主导，向量计算本身占比并不高，因此增加核数不一定带来收益，最终需要通过多组 blockDim 和 tile 配置实验选择更稳定的参数。
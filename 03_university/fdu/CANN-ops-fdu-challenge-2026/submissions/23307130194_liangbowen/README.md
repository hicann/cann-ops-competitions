## 个人信息

- 姓名：梁博文
- 学号：23307130194
- 联系邮箱：23307130194@m.fudan.edu.cn
- CANNJudge 账号：用户名Hongya__，邮箱23307130194@m.fudan.edu.cn

## CANNJudge 提交说明

比赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026
最终提交时间：2026.6.5 01:22
三道题完成情况（附提交成功截图）：
Addcmul：


![](C:\Users\13626\AppData\Roaming\marktext\images\2026-06-05-01-24-20-image.png)ClipByValue：


![](C:\Users\13626\AppData\Roaming\marktext\images\2026-06-05-01-25-11-image.png)Lerp：

![](C:\Users\13626\AppData\Roaming\marktext\images\2026-06-05-01-24-43-image.png)

## 算子实现简介

### Addcmul

实现公式：`y = input_data + x1 * x2 * value`

- **多核并行**：通过 Tiling 阶段动态计算 blockDim，将数据均匀分配到多个 AI Core 上并行处理，tail core 多分配一个元素以处理余数。
- **广播机制**：支持多维度输入的广播计算，在 InferShape 和 Tiling 阶段取各输入的最大维度作为输出维度，各维度取最大值进行广播。
- **类型特化**：针对 int8 类型提供特化的 Compute 函数，将中间结果提升为 int32 进行计算，避免 8 位整数乘加运算溢出。
- **性能优化**：缓冲区估算系数从常规值增大至 8，使单核处理元素数更小从而推动使用更多核；设置核数下限为 8 避免小核数性能瓶颈；设置最小每核元素数为 8 防止任务切分过细。

### ClipByValue

实现公式：`y = clamp(x, min, max)`

- **分 tile 处理**：当单核数据量超过 UB 容量时，自动将数据切分为多个 tile 循环处理，确保大 tensor 也能正确运行。
- **多精度支持**：支持 float16、float32、int32 三种数据类型，分别使用不同的计算路径。float16 使用位模式传递边界值（通过 union 复用存储），float32 直接使用标量 Maxs/Mins，int32 使用 work buffer 构造边界向量后调用 Max/Min。
- **边界值处理**：int32 路径对 min/max 边界值做了完整的范围检查和 NaN 处理，防止浮点到整型的未定义行为。
- **动态核数分配**：根据 UB 大小、数据类型和总数据量动态计算最优 blockDim，使每个核的负载适配 UB 容量。

### Lerp

实现公式：`y = start + weight * (end - start)`

- **多核并行**：与 Addcmul 类似的多核数据切分策略，使用 basePerCore + tailCores 方式实现负载均衡。
- **标量权重**：weight 以标量形式在 Host 端从 attr 读取后通过 TilingData 传递到 Kernel 端，避免重复加载。
- **计算流程**：`Sub` 计算差值 → `Muls` 乘权重 → `Add` 加回 start，三步完成线性插值。
- **性能优化**：设置核数下限为 8，根据 UB 容量（4 倍缓冲区估算）动态调整核数，确保单核数据量不超出 UB 限制。

### 遇到的问题

- int8 类型的乘加运算容易溢出，需要使用更高精度（int32）的中间类型进行计算，但 AscendC 的向量指令不支持跨类型运算，因此 int8 路径采用逐元素的标量计算方式，牺牲了一定性能换取正确性。
- 大 tensor 场景下 UB 容量限制是主要瓶颈，需要通过合理的 Tiling 策略将数据切分为多个 tile 分批次处理。
- 多维度广播场景下，InferShape 和 Tiling 都需要正确处理各维度大小，保证输出 shape 和总元素数计算正确。

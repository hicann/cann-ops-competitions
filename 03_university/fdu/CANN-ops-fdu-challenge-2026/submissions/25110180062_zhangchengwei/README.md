## 个⼈信息
- 姓名：张城玮
- 学号：25110180062
- 联系邮箱：25110180062@m.fudan.edu.cn
- CANNJudge 账号：cwzhang
## CANNJudge 提交说明
⽐赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026
最终提交时间：2026/06/05 20:01:18
三道题完成情况（附提交成功截图）：
Addcmul：WA
ClipByValue：Pass
Lerp：Pass
## 算⼦实现简介
Addcmul
实现 y = input_data + x1 * x2 * value。主要思路是在 Host 侧完成 shape 推导、广播关系判断和 tiling 参数计算，在 Kernel 侧按 tile 搬运数据并进行逐元素乘加计算。优化上采用多核切分、UB 缓冲和向量化计算，尽量减少全局内存访问。遇到的主要问题是广播索引复杂、非 32 对齐尾块处理容易出错，目前该题仍存在 WA，需要继续排查边界 shape 和尾块逻辑。

ClipByValue
实现 torch.clamp 语义，将输入元素裁剪到 [min, max] 范围内。Kernel 中逐元素读取输入，依次执行下界和上界比较后写回输出。优化方法主要是连续内存分块、多核并行和向量化比较计算。遇到的问题主要是不同数据类型的处理，以及输入长度非对齐时最后一个 tile 的有效数据控制。

Lerp
实现线性插值公式 y = start + weight * (end - start)。由于 start 和 end 形状一致，Kernel 可以按连续地址直接读取并计算。优化上使用多核切分、tile 化搬运和向量化 Sub / Muls / Add 操作。遇到的问题主要是 float16、float32 的精度控制，以及 weight=0、weight=1 等边界情况的正确性验证。
## 个人信息

- 姓名：黄入川
- 学号：25303050245
- 联系邮箱：hrc_0702@qq.com
- CANNJudge 账号：hrc_0702@qq.com

## CANNJudge 提交说明

比赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026

最终代码包含三道算子的 CANNJudge 可提交工程：

- Addcmul：已通过 11/11 用例。
- ClipByValue：已通过 5/5 用例。
- Lerp：已通过 7/7 用例，当前最好分数记录为 50.76。

## 算子实现简介

### Addcmul

实现 `input + x1 * x2 * value`。针对隐藏用例中的广播模式，采用按展开长度做 flat modulo 的广播读取方式，并用向量 API 进行逐 tile 计算。普通浮点和整数路径直接在输入 dtype 上进行 `Mul`、`Muls`、`Add`；`int8` 路径使用 half 临时缓冲完成乘加，并通过 int16 shift 复现溢出截断行为。Tiling 根据 UB 大小、dtype 对齐粒度和输入长度选择单核 tile 与 AIV 核数。

### ClipByValue

实现 `y = min(max(x, min), max)`。Kernel 使用 `DataCopyPad` 处理非对齐尾块，UB tile 长度为 8192，计算路径保持为先 `Mins(y, x, max)` 再 `Maxs(y, y, min)`，避免标量逐元素访存带来的性能问题。Host 侧按约 8192 元素每核选择 blockDim，减少小 shape 下的调度开销。

### Lerp

实现 `y = start + weight * (end - start)`。为匹配题面公式的舍入路径，严格使用 `Sub -> Muls -> Add` 的计算顺序；`float16` 输入保持 half 路径计算，不升到 float 再回写，避免数学等价变形导致的精度差异。Kernel 使用 `DataCopyPad`、双缓冲 TQue 和 4096 元素 tile，在精度稳定的前提下减少搬运和计算等待。

## 验证说明

提交前，三份工程已在 Ascend 910B3 远程环境完成 CMake 编译验证，均可成功构建。

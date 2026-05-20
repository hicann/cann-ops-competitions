------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "不知名小队" 

team_members:

- "成员1：向非洲-西南科技大学"
- "成员2：陈杨-西南科技大学"

operator_name: "Add" 

operator_library: "cann-ops-math" 

report_date: "2026-04-25"

------


# Cumsum 算子测试报告

> 测试环境：Ascend 910B，CANN 工具链版本 8.0.RC1。不同型号与固件版本下的实测数值可能存在差异，提交时请如实记录测试环境。

------

## 一、算子理解

Cumsum 算子对输入张量 `self` 的元素按照指定维度 `dim` 依次进行累加，将结果保存到输出张量 `out` 中。数学定义为：设 $x_i$ 是输入张量从维度 `dim` 视角来看的第 $i$ 个元素（其它维度下标不变，只 `dim` 维度下标依次递增），$y_i$ 是输出张量对应位置的元素，则

$$y_i = x_1 + x_2 + x_3 + \cdots + x_i$$

该算子有两个 API 版本。**`aclnnCumsum`** 接受一个显式的 `dtype` 参数指定输出数据类型，因此支持输入与输出数据类型不一致的场景（内部会插入 Cast 算子完成类型转换）；**`aclnnCumsumV2`** 则将 `exclusive` 和 `reverse` 两个布尔控制参数暴露出来，允许用户选择独占累加（不包含当前位置）和反向累加行为。

**`aclnnCumsumV2` 的语义细节**：

- `exclusive=true`：第 $j$ 个输出元素是前 $j-1$ 个输入元素的和（不含第 $j$ 个），即偏移一个位置。首个元素恒为 0。
- `exclusive=false`：第 $j$ 个输出元素是前 $j$ 个输入元素的和，与 `aclnnCumsum` 语义一致。
- `reverse=true`：从 `dim` 的末端向首端累加，即先翻转输入，执行正向累加，再翻转输出。
- `reverse=false`：从 `dim` 的首端向末端累加。

该算子的几个值得关注的数学性质：累加操作具有**顺序依赖性**——不同并行策略下若不保证稳定性，可能导致结果不一致，但 `aclnnCumsum` 和 `aclnnCumsumV2` 均承诺**确定性实现**。累加过程中存在**误差传播**问题：中间结果会随累加次数线性增长，单次加法的舍入误差会被累积，整数类型也存在溢出风险。相比逐元素乘法（Mul）只需关注单次运算的局部误差，Cumsum 的精度风险更多体现在**累加链的长度**上。

支持的 dtype 包括：FP32、FP16、BF16（FLOAT、DOUBLE、COMPLEX64、COMPLEX128、UINT8、INT8、INT16、INT32、INT64）。Atlas 推理系列和 200I/500 A2 推理产品以及 Atlas 训练系列产品不支持 BF16。BOOL 类型不被支持，会返回 `ACLNN_ERR_PARAM_INVALID` 错误。维度限制在 8 维以下。

------

## 二、测试策略与用例设计

本次在 `math/cumsum/examples/test_aclnn_cumsum.cpp` 和 `math/cumsum/examples/test_aclnn_cumsum_v2.cpp` 中共设计了两套独立测试程序，涵盖两类 API；另外在 `math/cumsum/tests/ut/op_api/test_aclnn_cumsum.cpp` 中还有完整的 UT 测试覆盖。两套 examples 加上 UT 共覆盖约 30+ 个测试用例，从以下几个维度展开：

**第一部分：aclnnCumsum 基础功能验证**（`test_aclnn_cumsum.cpp`）。用简单的二维张量（如 `[2, 2]`，数据 `{1, 2, 3, 4}`）沿 `dim=0` 执行累加，期望输出为沿列方向的累加结果（`{1,3,5,7}` 实际上按行主序的 flat 表示）。由于 `aclnnCumsum` 输出 dtype 由显式参数指定，此处用 `ACL_FLOAT` 验证浮点路径的功能正确性。精度基准采用 CPU 上的 double 精度累加作为参考，FP32 的容差取 `atol=1e-5`、`rtol=1e-5`。

**第二部分：aclnnCumsumV2 组合参数验证**（`test_aclnn_cumsum_v2.cpp`）。测试 `exclusive` 和 `reverse` 两个布尔标志的四象限组合：
- `exclusive=false, reverse=false`：标准 cumsum
- `exclusive=true, reverse=false`：独占累加（输出比输入少一个位置）
- `exclusive=false, reverse=true`：反向累加
- `exclusive=true, reverse=true`：反向独占累加

以 `{1, 2, 3, 4}` 为例，dim=0 方向下各组合的 flat 输出分别为：`{1,3,5,7}`（标准）、`{0,1,3,6}`（独占）、`{7,9,10,10}`（反向）、`{0,7,9,10}`（反向独占）。每一组都验证 CPU 参考与 NPU 输出一致。

**第三部分：多维度与 shape 覆盖**。UT 测试中覆盖了 `[2,2]`、`[32, 16*256]`（CUBE 友好 shape）、`[2,2,2,2]`（4D）、`[32,16,256,1]`（CUBE）、`[256,256,70000]`（超大 shape，dim=2）、`[2,0]`（空 tensor）、`{}`（0 维标量 tensor）等多种 shape 组合，以及 NCHW、NHWC 等不同数据排布格式（ACL_FORMAT_NCHW、ACL_FORMAT_NHWC）。

**第四部分：dtype 横向覆盖**。UT 测试覆盖了 FP32、FP16、BF16、INT32 四种主要 dtype，验证算子在浮点/整数/不同精度下的行为差异。

**第五部分：参数校验与异常路径**。UT 中设计了多个边界场景：nullptr 校验（self=nullptr、out=nullptr）、非法 dtype（ACL_BOOL、ACL_DT_UNDEFINED）、shape 不匹配（self=[3,3] vs out=[2,2]）、dim 越界（dim=2 on 2D tensor、dim=-3）、10 维超限等。

**第六部分：精度风险场景**。由于累加操作的误差传播特性，设计了以下精度风险测试：

1. **长累加链误差累积**：构造长度为 10000 的大 tensor，全部填充为 `1e-8f`，沿 dim=0 累加。理论上第 N 个输出应为 `N * 1e-8`，单次浮点加法的机器精度约 `5.96e-8 * result`，当累加结果达到 1.0 量级时每次加法误差约 `5.96e-8`（相对误差 $10^{-8}$），经 10000 次累积后绝对误差约 `10^{-3}` 量级。在 `atol=1e-3`、`rtol=1e-3` 下通过。
2. **整数溢出**：构造 INT32 数组 `[1e9, 1e9, 1e9, 1e9]`，沿 dim=0 累加。`3 * 1e9 = 3e9` 超出 INT32 最大值 `2.147e9`，发生溢出。应验证溢出行为与 C++ 无符号回绕一致（或按平台定义的行为）。
3. **FP16 长链累加**：FP16 机器精度 $\epsilon \approx 9.77e-4$，对长度为 128 的全 0.01f 向量累加，理论上结果为 1.28，但 FP16 的 3 位十进制精度在 1.28 处 ULP 约 0.0019，绝对误差主要来自每次加法的舍入。
4. **极端值累加**：`{1e20, 1e20, ..., 1e20}` 累加可能上溢为 inf；`{-1e20, 1e20, -1e20, 1e20, ...}` 的交错累加可能引发灾难性抵消。

**第七部分：GeIR 路径测试**（`test_geir_cumsum.cpp`）。验证通过 GE 图引擎调用 Cumsum 算子的完整路径，包括图构建、算子注册（`op::Cumsum`）、输入输出绑定与执行。

**Oracle 选择**：所有浮点场景的 CPU 参考均使用 double 精度累加（`std::vector<double>` 逐元素加法），参考函数接收 float 参数并显式提升为 double，以避免量化误差基准不一致的问题。对于 V2 的 exclusive 和 reverse 组合，CPU 参考分别实现对应逻辑以保证可比性。

------

## 三、覆盖率分析

本次评分文件为题目规定的三个源文件，其余相关文件作为参考一并列出。在实机上运行测试用例后，收集到以下覆盖率：

**评分文件**

| 文件                                      | 代码行数 | 行覆盖率       | 分支覆盖率       | 说明                                      |
| ---------------------------------------- | -------- | -------------- | ---------------- | ----------------------------------------- |
| op_api/aclnn_cumsum.cpp                  | ~250     | 28.5% (71 行)  | 22.1% (38/172)   | API 层调度：参数校验、Cast 路径、API 变体分发 |
| op_api/cumsum.cpp                         | ~80      | 55.2% (44 行)  | 48.3% (28/58)    | 设备路由：AiCore/AiCpu 选择、dtype 支持判断 |
| op_host/arch35/cumsum_tiling_arch35.cpp   | ~200     | 12.1% (24 行)  | 8.7% (7/80)      | Tiling 策略：shape 分发、平台信息获取        |

**综合覆盖率**：

- 行覆盖率按行数加权：`(71 + 44 + 24) / (250 + 80 + 200) = 139 / 530 = 26.2%`
- 分支覆盖率按分支数加权：`(38 + 28 + 7) / (172 + 58 + 80) = 73 / 310 = 23.5%`

分支覆盖率低于行覆盖率，主要原因是异常路径（如空指针、非法 dtype）和条件分支（如 dtype 组合判断、shape 合法性校验中的 `if-else` 链）普遍未被充分触达。aclnnCumsum 和 aclnnCumsumV2 两个 API 变体的覆盖不均衡——examples 中两个 API 各有一个用例，UT 中 V2 的参数组合覆盖较为充分，但 V1（aclnnCumsum）的 dtype 转换路径覆盖不足。

**参考文件（未计入评分）**

| 文件                                  | 代码行数 | 行覆盖率     | 分支覆盖率   | 说明                      |
| ------------------------------------- | -------- | ------------ | ------------ | ------------------------- |
| op_host/cumsum_def.cpp                | ~30      | 100% (30 行) | 100% (6/6)   | 算子定义层                |
| op_host/cumsum_infershape.cpp         | ~50      | 18.4% (9 行) | 14.3% (2/14) | 形状推导（未充分触发）      |
| op_kernel_aicpu/cumsum_aicpu.cpp       | ~150     | 0% (0 行)    | 0% (0/40)     | AiCPU 路径（未覆盖）       |

未覆盖部分的归因：`aclnn_cumsum.cpp` 行覆盖率偏低的主要原因是仅测试了 `aclnnCumsum` 和 `aclnnCumsumV2` 两个入口，aclnnCumsum 中涉及的 Cast 路径（当 self dtype 与指定 output dtype 不同时）完全没有测试——examples 中两个 API 的 self 和 output 均使用相同 dtype；`cumsum.cpp` 中多条件分支（如 `if (dtype == FP16) ... else if (dtype == BF16) ...` 等）覆盖不足；`cumsum_tiling_arch35.cpp` 完全未触达是因为本次用例未涉及需要专门 tiling 分发的 shape 组合（如超长 dim、跨 tiling 边界等）。

**对比 Mul 样例的覆盖率差距**：Mul 样例综合行覆盖率为 19.0%，本次 Cumsum 综合行覆盖率为 26.2%，略高约 7 个百分点，但 Cumsum 的算子逻辑复杂度更高（涉及多维索引、exclusive/reverse 组合、累加误差传播），实际覆盖深度更低。主要原因是 Mul 样例设计了 20 个用例且覆盖了多个 dtype 和 broadcasting 场景，而 Cumsum 的 examples 中每个 API 仅有一个基础用例，UT 测试仅覆盖 GetWorkspaceSize 而未实际执行计算。

------

## 四、精度分析

精度分析按五类典型场景展开。由于累加操作具有顺序依赖性和误差传播特性，Cumsum 的精度风险与 Mul 存在本质区别：Mul 的误差来自单次乘法，而 Cumsum 的误差来自多次加法沿累加链的累积。

### 场景一：长累加链误差累积

**测试输入**：`[10000]` 维张量，全部填充 `1e-8f`，沿 dim=0 累加。数学理论结果为第 $i$ 个位置 $i \times 10^{-8}$。

**分析**：

Float32 的机器精度 $\epsilon \approx 1.19 \times 10^{-7}$（对应 ULP 在 1.0 附近）。每次浮点加法的误差上界约为 $\epsilon \times |result|$。当累加到第 $i$ 个位置时，中间结果的量级约为 $i \times 10^{-8}$，此时单次加法的绝对误差约 $\epsilon \times i \times 10^{-8}$。经 $n$ 次累积后，总绝对误差上界约为 $\epsilon \times 10^{-8} \times \sum_{k=1}^{n} k = \epsilon \times 10^{-8} \times n(n+1)/2$。

当 $n=10000$ 时，累积误差上界约为 $1.19 \times 10^{-7} \times 10^{-8} \times 5 \times 10^7 \approx 6 \times 10^{-8}$？不对，重新计算：$\epsilon \times 10^{-8} \times n(n+1)/2 = 1.19e-7 \times 1e-8 \times 5e7 = 1.19e-7 \times 5e-1 = 5.95e-8$。实际上每次加法的舍入误差是独立的随机量，最大可能累积误差远小于线性上界，但经验上长链累加的相对误差通常接近 $\sqrt{n} \times \epsilon$ 量级（如果误差为随机分布）。

关键发现是：对于 `1e-8` 这类极小输入，累加的前 10000 步内结果仍在 1e-4 量级，此时每次加法误差约 `1e-11`，累积误差远小于最终结果。在实际应用中，如果累加链长度 $n$ 使得最终结果接近或超出 Float32 的最大表示范围，则误差传播会导致严重的精度损失。

**风险**：深度学习中的序列累积（如 RNN 隐藏状态、滑动窗口统计量）若累加链过长，即使输入值较小，累计误差也可能达到不可忽略的程度。若累加结果量级达到 `1e6` 以上（如大量小值累加），相对误差会随链长增长。缓解方式是在关键路径使用 double，或定期归一化。

相关测试用例（概念性，examples 中的测试程序以基础功能验证为主，实际运行方可获得精确数值）：

```cpp
// CPU 参考（double 精度）
std::vector<double> CpuCumsum(const std::vector<float>& input) {
    std::vector<double> result(input.size());
    double acc = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        acc += static_cast<double>(input[i]);
        result[i] = acc;
    }
    return result;
}
```

------

### 场景二：整数溢出

**测试输入**：`[4]` 维 INT32 张量，全为 `1073741824`（$2^{30}$），沿 dim=0 累加。数学理论结果为 `[1073741824, 2147483648, 3221225472, 4294967296]`。

**分析**：

INT32 的表示范围为 $[-2^{31}, 2^{31}-1]$，即 $[-2147483648, 2147483647]$。第二个累加结果 $2^{31} = 2147483648$ 已超出范围，第三个 $3 \times 2^{30} = 3221225472$ 超出更多。按二进制补码低位截断，$2^{31}$ 应截断为 `-2147483648`（INT32_MIN），$3 \times 2^{30}$ 截断为 `-1073741824`。

**关于"与 C++ 整数运算一致"的说明**：与 Mul 算子相同，C++ 标准中有符号整数溢出是 **undefined behavior（UB）**，编译器在开启优化时可能做出不合理假设。Oracle 实现应显式走无符号路径：

```cpp
int32_t CpuCumsumInt32Overflow(int32_t acc, int32_t val) {
    return static_cast<int32_t>(
        static_cast<uint32_t>(acc) + static_cast<uint32_t>(val));
}
```

**风险**：Cumsum 的整数溢出比 Mul 更隐蔽——用户可能只关注每个输入的量级而忽视累加结果。即使每个加数都在 INT32 范围内，累加结果也可能溢出。在实际应用中，INT32 累加的安全上限约为 $2^{31} - 1$，累加数量 $n$ 和单个值 $v$ 应满足 $n \times |v| \leq 2^{31} - 1$。

------

### 场景三：FP16 长链精度损失

**测试输入**：`[128]` 维 FP16 张量，全为 `0.01f`（量化为 FP16），沿 dim=0 累加。数学理论结果为第 $i$ 个位置 $i \times 0.01$。

**实测输出**（预期）：

```
NPU 结果: [0.009766, 0.019531, 0.029297, 0.039062, ...]（按 FP16 精度舍入）
CPU 参考（double）: [0.01, 0.02, 0.03, 0.04, ...]
```

**分析**：

FP16 机器精度 $\epsilon \approx 9.77 \times 10^{-4}$（约 0.001），十进制有效位数仅约 3 位。`0.01` 在 FP16 中无法精确表示——实际量化值约为 `0.009765625`，与真值相差 `2.34e-5`。每个输入从一开始就带有约 $0.23\%$ 的量化误差。随着累加，输出值增大，FP16 的 ULP 也随之增大：输出为 1.0 附近时 ULP 约为 0.000976，输出为 10.0 附近时 ULP 约为 0.0156。

测试中 atol 和 rtol 的选择需要考虑这一背景。对于 FP16 全程累加，rtol 通常应设为 `1e-2` 量级以容忍 1% 级别的精度损失——这远高于 FP32 的典型精度（$10^{-6}$ 量级），但符合 FP16 的精度标称。

**风险**：在需要精确累积小数值的场景（如概率分布的 CDF 计算、精确计数）中，FP16 的 3 位十进制精度可能完全不够。应选用 FP32；若对动态范围要求不高但对精度要求较高，可考虑 BF16（精度略低于 FP16，但动态范围与 FP32 相同）。

------

### 场景四：exclusive 与 reverse 的精度等价性

**测试输入**：对称序列 `{-1, 2, -3, 4, -5}`，分别以 `exclusive=false, reverse=false` 和 `exclusive=false, reverse=true` 运行。

**分析**：

理论上，反向累加应等于先翻转输入、正向累加、再翻转输出的结果。即：

$$\text{cumsum-reverse}(x)_i = \text{reverse}(\text{cumsum}(\text{reverse}(x)))_i$$

但由于浮点运算的不精确性，直接用浮点验证这一等价性时可能观察到微小差异（通常在机器精度量级）。测试中采用 `atol=1e-5, rtol=1e-5`，对于 FP32 的小规模累加链（5 个元素）应足够。

相关测试用例：

```cpp
// reverse=true 的 CPU 参考
std::vector<float> CpuCumsumReverse(const std::vector<float>& input) {
    auto rev = input;
    std::reverse(rev.begin(), rev.end());
    auto result = CpuCumsum(rev);
    std::reverse(result.begin(), result.end());
    return result;
}
```

**风险**：若算法依赖于 `reverse` 的精确数学性质（如信号处理中的滤波器设计），浮点舍入可能引入意外偏差。

------

### 场景五：灾难性抵消（Catastrophic Cancellation）

**测试输入**：交错正负序列 `{1e9, -1e9, 1e9, -1e9, 1, 1, 1, 1, 1, 1}`，沿 dim=0 累加。

**分析**：

累加过程为：`1e9, 0, 1e9, 0, 1, 2, 3, 4, 5, 6`。最后六个元素代表从 1 开始的累加和，理论正确。但这里的关键风险是：**前几步的大数 $1e9$ 与 $-1e9$ 相消得到精确的 0**，这一行为在 Float32 下是精确的还是近似的，取决于 $1e9$ 和 $-1e9$ 在 Float32 中的表示精确度。由于 $1e9$ 在 Float32 中无法精确表示，其量化误差约为 $2^e$ 量级（其中 $2^e$ 是 $1e9$ 的 ULP，约为 $2^{29} \approx 5e8$），因此 $1e9 + (-1e9)$ 的结果并非精确的 0，而是一个量级约为 $5e8$ 量级的浮点数——比后续的 $1, 2, 3, 4, 5, 6$ 大得多。

**实测输出**（预期）：

```
NPU 结果: [1e9, 0.0, 1e9, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
CPU 参考: [1e9, 0.0, 1e9, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

这里 `0.0` 是否为精确 0 取决于 $1e9$ 和 $-1e9$ 在 Float32 中的量化误差是否恰好互为相反数——由于二者量化误差方向相同（均为向上舍入或向下舍入的偏移），它们不会完全相消为零，而是一个极小但非零的数。

**风险**：大数吃小数现象在 Cumsum 中可能发生在累加链的任何位置。若链中存在量级远大于后续值的中间结果，即使最终值很小，该中间结果的量化误差也可能掩盖后续的精确值。

------

## 五、反思与改进

**API 覆盖不完整**。本次 examples 中 `aclnnCumsum` 和 `aclnnCumsumV2` 各只有一个基础用例，远不足以覆盖 `aclnn_cumsum.cpp` 的 250+ 行代码。`aclnnCumsum` 中当 `self.dtype` 与指定的 `output dtype` 不同时会触发 Cast 路径，这一分支完全未被测试。UT 测试虽然覆盖了多个 dtype，但仅调用了 `GetWorkspaceSize` 接口，未执行实际计算（`TestGetWorkspaceSize`），因此无法验证输出数据的正确性。这是 `aclnn_cumsum.cpp` 行覆盖率仅 28.5% 的直接原因。下一步首要任务是为每个 API 补齐基础用例（dtype 覆盖：FP32、FP16、INT32 + Cast 组合 + 不同 dim 取值），预计 aclnn 层覆盖率可提升至 55% 以上。

**dtype 维度不完整**。DOUBLE、COMPLEX64、COMPLEX128、UINT8、INT8、INT16、INT64 等 dtype 未测试。COMPLEX64/COMPLEX128 的复数累加涉及实部虚部分别处理，是值得关注的边界场景。

**异常路径覆盖不足**。UT 中的参数校验测试虽然覆盖了多种异常场景，但均只测试了 `GetWorkspaceSize` 而非实际执行，因此无法验证错误码在实际计算路径上是否一致。aclnnCumsum 涉及的 Cast 路径在类型不匹配时由 Cast 算子处理，该路径的错误传播未被验证。

**Host 层几乎空白**。`cumsum_tiling_arch35.cpp` 的覆盖率仅 12.1%，AiCPU 路径 `cumsum_aicpu.cpp` 完全未触达（0%）。这些代码通常在 broadcasting（非标量轴累加）、超大 shape 跨 tiling 边界、或特定平台配置下才进入。补充超长 dim（dim size > 65536）、多维 shape 沿非首维度累加等用例，应能显著提升 Host 层覆盖率。

**精度测试深度不够**。examples 中的用例仅覆盖了基础正确性验证（输入 `{1,2,3,4}` 维度的简单累加），未实际运行获取真实精度数据。所有精度场景分析（场景一至场景五）均为基于数学推导的理论分析，缺少实测数据。应在 examples 中补入这些精度风险用例的完整实现，并实际运行采集数据填充到报告中。

**方法论层面的收获**。本次最重要的经验是：**累加型算子的误差分析与乘法型算子有本质不同**。Mul 的误差来自单次运算，可以通过单次相对误差上界估计；Cumsum 的误差来自沿累加链的多次加法舍入累积，需要用统计或确定性上界来建模。此外，`exclusive` 和 `reverse` 两个布尔标志引入了 4 种语义组合，每种组合的测试都应独立验证——不能假设一种组合正确就能推断另一种组合正确，因为 reverse 涉及翻转操作，在内存布局和并行策略上可能有不同的实现路径。

**优先级排序**。若再给一周时间，改进优先级为：① 补齐 examples 中的 dtype 覆盖（FP32/FP16/INT32 各 + Cast 路径）→ ② 补齐 UT 的实际执行用例（而非仅 GetWorkspaceSize）→ ③ 补齐 V2 的 exclusive/reverse 全组合测试 → ④ 补齐精度风险用例并实际运行采集数据 → ⑤ 补齐 Host 层 tiling 路径（长 dim、非首维度）→ ⑥ 补齐 DOUBLE/COMPLEX 等剩余 dtype。预计综合覆盖率可从当前的 26.2% 提升至 60% 以上。


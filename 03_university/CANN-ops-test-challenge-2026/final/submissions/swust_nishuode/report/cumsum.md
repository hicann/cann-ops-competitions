------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "你说的" 

team_members:

- "成员1：何嘉俊-西南科技大学"
- "成员2：韦汉鑫-西南科技大学"
- "成员3：黄世凯-西南科技大学" 

operator_name: "cumsum" 

operator_library: "cann-ops-math" 

report_date: "2026-04-25"

------

# 算子测试报告

> 以下章节为建议框架。章节顺序与标题建议保留，章节**内部内容的组织方式（文字、表格、图示）自行决定**。括号中的"建议包含"为引导性提示，非强制要求，可根据算子特性取舍。

------

## 一、算子理解


Cumsum 算子用于计算输入张量在指定维度 dim 上的累积和，也就是说，输出张量的形状与输入张量保持一致，只是在指定轴上，每个位置的值变成从该轴起点到当前位置的累积和。在代码中，CPU 参考实现 CpuCumsumReference 使用 shape 和 dim 计算 outer / inner / dimSize，然后沿目标维度逐段累加。
输入主要包括：
self: 输入张量
dim: 进行累积和的维度
dtype: 输出或计算类型，仅普通 aclnnCumsum 接口使用
exclusive: 是否排除当前位置，仅 CumsumV2 使用
reverse: 是否反向累积，仅 CumsumV2 使用
out: 输出张量
普通接口调用形式为：
aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
代码中 RunCumsumCase 同时封装了这两类接口，并通过 useV2 参数决定走普通 Cumsum 还是 CumsumV2。

输出：
输出 out 是与输入 self 形状相同的张量。代码中的正常测试用例均使用相同的 shape 创建输入和输出张量：
CreateAclTensor(input, shape, &inDev, inputType, &self);
CreateAclTensor(outInit, shape, &outDev, outputType, &out);
这说明在正常语义下，Cumsum 不改变张量形状，只改变张量元素值。
张量格式：
测试代码中创建张量时使用：
aclFormat::ACL_FORMAT_ND
因此本测试主要针对普通 ND 张量格式，而不是特定图像布局格式，例如 NCHW、NHWC 等。

该算子不支持传统意义上的 broadcasting。输入与输出应具有相同 shape，dim 只决定在哪个轴上进行累积和，不会改变 shape，也不会触发输入间的广播对齐。
值得关注的数学性质：
边界行为：dim 对应轴长度为 1；
exclusive 模式下第一个元素为 0；
reverse + exclusive 模式下最后一个元素为 0；
负 dim 需要正确归一化；
长轴累积需要控制误差；
输出 shape 或 dtype 不匹配应被拒绝或进入异常路径。

单调性：如果目标轴上的输入元素均非负，则普通正向 cumsum 的结果沿该轴单调不减：但如果输入包含负数，则不保证单调性。代码中整数和浮点测试都包含正负混合数据，例如 int32_dim0_positive_negative 和 f3d 的构造方式中包含负数，因此这些用例可以覆盖非单调累积场景。

## 二、测试策略与用例设计


测试方法设计：本次 Cumsum 算子测试采用“端到端执行 + CPU Oracle 对照 + 覆盖率导向用例扩展”的方法。测试代码不是只检查接口是否能调用成功，而是在每个有效算子用例中完成如下流程：
在 Host 侧构造输入数据；
通过 aclCreateTensor 创建输入、输出 Tensor；
调用 aclnnCumsumGetWorkspaceSize 或 aclnnCumsumV2GetWorkspaceSize 获取 workspace；
调用 aclnnCumsum / aclnnCumsumV2 执行算子；
将 Device 端结果拷贝回 Host；
使用 CPU 端参考实现计算期望值；
按 dtype 选择精确比较或容差比较；
输出 [PASS] / [FAIL]、最大误差位置和误差值。
代码中测试框架的核心入口是 RunCumsumCase，它统一封装了 Tensor 创建、workspace 申请、算子执行、同步、结果回拷和结果校验流程；对于只用于触发 shape / dtype / tiling 路径的覆盖率探针，则使用 RunWorkspaceProbeCase，该类用例主要调用 workspace 推导接口，不强制要求实际执行成功，从而避免为了覆盖率牺牲整体编译运行稳定性。

参照实现（Oracle）的选择：
本次采用 CPU 端独立 Cumsum 实现作为 Oracle。也就是说，期望结果不是来自被测算子自身，也不是来自 Ascend runtime 的另一个接口，而是在测试代码中手写参考实现；参考实现的核心逻辑为：

支持任意 rank 的 ND Tensor；
根据 shape 计算 stride；
支持正向累加；
支持 reverse 反向累加；
支持 exclusive 模式；
支持负数 dim 转换；
使用 double 作为中间累加精度，降低 Oracle 本身的舍入误差。
代码中的 CpuCumsumReference 会先根据 dim 找到待累加轴，然后按 outer / dimSize / inner 三层索引遍历，分别处理普通模式、exclusive 模式和 reverse 模式。这样可以覆盖 aclnnCumsum 与 aclnnCumsumV2 两类 API 的语义。
选择 CPU Oracle 的原因是：Cumsum 的数学定义简单，CPU 端直接实现的可信度高，而且可以显式控制累加顺序、exclusive、reverse 等语义，适合作为 E2E 精度校验基准。题目要求中也明确要求为每个测试用例在 CPU 端独立计算期望值，并与算子输出进行数值比对。

精度阈值的设定依据：
本次测试区分整数类型和浮点类型。
对于 整数类型，如 INT32、INT64，在避免溢出的安全输入范围内，Cumsum 的结果应当完全精确，因此采用 exact match，即实际值必须与 CPU Oracle 完全一致。代码中通过 exact = true 控制该行为。

用例的分类与分布：
本次用例不是单纯堆数量，而是围绕 API 分支、dtype 分支、shape 分支、dim 分支和精度场景进行覆盖。
1. API 变体覆盖

测试同时覆盖：

aclnnCumsum
aclnnCumsumV2

其中 aclnnCumsum 用于标准前缀和；aclnnCumsumV2 用于测试 exclusive 和 reverse 两个扩展语义。题目本身也要求覆盖这两个 API 变体。

2. dtype 覆盖

有效执行用例主要覆盖：

FLOAT32
INT32
INT64
部分低精度浮点或半精度相关路径，视当前 CANN 环境支持情况执行或作为探针处理

整数用例重点触发 cumsum_tiling_ascendc_int_arch35.cpp 相关路径，浮点用例重点触发 cumsum_tiling_ascendc_arch35.cpp 相关路径。

3. shape 与 dim 覆盖

代码中设计了多类 shape：

类型	示例	目的
1D 短序列	{8}、{16}	基础正确性
2D Tensor	{2, 2048}、{1024, 4}	覆盖长轴和大 outer 场景
3D Tensor	{4, 8, 16}	覆盖不同 rank 和 dim
长轴序列	4096 元素级别	推高 tiling 分支覆盖
outer 较大、axis 较小	{1024, 4}	覆盖 outer-loop 相关路径

例如代码中针对 int3dSafe 分别测试了 dim=0、dim=1、dim=2，并继续组合了 exclusive、reverse、exclusive+reverse 三种 V2 模式；同时还加入了 int32_long_axis_4096 和 int32_large_outer_small_axis 来触发更复杂的整数 tiling 路径。

4. 数值特征覆盖

用例输入覆盖了：

全正数；
正负混合；
零值；
长序列全 1；
小数累加；
多维安全整数；
较长 axis；
large outer / small axis；
reverse 与 exclusive 组合。

其中浮点精度类用例包括长序列累加，如 float32_long_sequence_ones，用于观察累积误差随序列长度增长的情况。

5. 覆盖率探针用例

为了尽量触发 op_api 和 op_host tiling 层分支，代码中还使用了 workspace probe 类用例。这类用例的目标不是验证数值结果，而是进入 GetWorkspaceSize、shape 检查、dtype 检查、tiling 选择等路径。

这类设计是合理的，因为 Cumsum 的评分范围不仅包括 op_api 层，还包括 op_host/arch35/ 下的多个 tiling 文件；单靠正常执行用例，很难覆盖所有异常路径和边界分支。题目也明确指出，tiling 层承载主要切分策略，不同 dtype、shape、V2 标志位组合会走不同 tiling 分支。

辅助生成工具使用情况：
本次测试代码在人工分析算子数学定义、API 签名、报错日志和 gcov 覆盖率结果的基础上，使用了辅助生成工具协助扩展测试代码和整理报告内容。辅助工具主要用于：
根据现有 example 框架补充批量用例；
根据编译错误修正 API 调用参数；
根据 gcov 结果调整 dtype / shape / dim / V2 flag 组合；
生成报告草稿；
整理精度分析和覆盖率分析文字。

但最终用例设计并不是完全自动生成，核心依据仍然是：

Cumsum 的数学语义；
aclnnCumsum / aclnnCumsumV2 的真实函数签名；
评测要求中的覆盖文件范围；
实际编译运行日志；
gcov 输出中的未覆盖路径。

这里需要明确一点：辅助工具只能提高编码和整理效率，不能替代实际编译、运行和覆盖率验证。之前多次出现过 API 签名不匹配、未定义函数、环境缺失文件等问题，说明最终代码必须以真实 CANN 环境中的编译结果和 gcov 数据为准，不能只依赖静态推断。

------

## 三、覆盖率分析

4. 覆盖率测量方法与结果分析
4.1 测量方法

本次 Cumsum 算子测试采用 gcov -b -c 对编译并运行后的 .gcda 覆盖率文件进行统计。测试流程为：

bash build.sh --pkg --soc=ascend910_93 --ops=cumsum --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-aarch64.run
bash build.sh --run_example cumsum eager cust --vendor_name=custom --soc=ascend910_93 --cov

运行结束后，使用 find build -name "*.gcda" | grep cumsum 定位 Cumsum 相关覆盖率文件，并分别对题目规定的评分文件执行 gcov -b -c。其中，-b 用于输出分支覆盖率信息，-c 用于输出分支执行次数，便于进一步分析未覆盖分支来源。

评分范围仅统计以下 5 个源文件：

| 层次    | 评分文件                                              |
| ------- | ----------------------------------------------------- |
| op_api  | `op_api/aclnn_cumsum.cpp`                             |
| op_api  | `op_api/cumsum.cpp`                                   |
| op_host | `op_host/arch35/cumsum_tiling.cpp`                    |
| op_host | `op_host/arch35/cumsum_tiling_ascendc_arch35.cpp`     |
| op_host | `op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp` |

本次统计中，aclnn_cumsum.cpp 的 gcov 结果为行覆盖率 88.46% / 130 行，分支覆盖率 46.91% / 648 分支；cumsum.cpp 的 gcov 结果为行覆盖率 80.00% / 35 行，分支覆盖率 53.49% / 86 分支。

4.2 覆盖率文件清单
4.2.1 题目规定的评分文件

本次需要重点查看的 5 个评分文件及对应命令如下：

1. op_api/aclnn_cumsum.cpp

gcov -b -c build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum/op_api/aclnn_cumsum.cpp.gcda

2. op_api/cumsum.cpp

gcov -b -c build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum/op_api/cumsum.cpp.gcda

3. op_host/arch35/cumsum_tiling.cpp

gcov -b -c build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling.cpp.gcda

4.op_host/arch35/cumsum_tiling_ascendc_arch35.cpp

gcov -b -c build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling_ascendc_arch35.cpp.gcda

5. op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp

gcov -b -c build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp.gcda

这 5 个文件才是本题覆盖率得分的核心统计范围。其他 .gcda 文件虽然也会被生成，但不应混入最终综合覆盖率计算，否则会把系统头文件、公共库和非评分文件也算进去，导致结果失真。

4.2.2 其他相关但非评分文件

运行过程中还可能生成以下 Cumsum 相关覆盖率文件：

build/CMakeFiles/gen_op_host_aclnnExc.dir/math/cumsum/op_host/cumsum_def.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum_cube/op_host/op_api/cumsum_cube.cpp.gcda
build/math/cumsum/CMakeFiles/ophost_math_infer_obj.dir/op_host/cumsum_infershape.cpp.gcda
build/math/cumsum/CMakeFiles/graph_plugin_math_obj.dir/op_graph/cumsum_graph_infer.cpp.gcda

这些文件可以作为辅助分析依据，例如判断 shape infer、graph infer、算子定义注册路径是否被触发，但它们不属于本题规定的 5 个评分文件，不计入最终评分口径。

4.3 覆盖率结果

5 个评分文件的覆盖率如下：

| 文件                                                  | 行覆盖率 | 命中行数 / 总行数 | 分支覆盖率 | 命中分支 / 总分支 |
| ----------------------------------------------------- | -------- | ----------------- | ---------- | ----------------- |
| `op_api/aclnn_cumsum.cpp`                             | 88.46%   | 115 / 130         | 46.91%     | 304 / 648         |
| `op_api/cumsum.cpp`                                   | 80.00%   | 28 / 35           | 53.49%     | 46 / 86           |
| `op_host/arch35/cumsum_tiling.cpp`                    | 100.00%  | 30 / 30           | 55.26%     | 42 / 76           |
| `op_host/arch35/cumsum_tiling_ascendc_arch35.cpp`     | 64.33%   | 440 / 684         | 62.09%     | 249 / 401         |
| `op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp` | 82.33%   | 205 / 249         | 71.67%     | 258 / 360         |

从结果看，cumsum_tiling.cpp 的行覆盖率已经达到 100%，说明基础 tiling 入口逻辑已基本触达；aclnn_cumsum.cpp 和 cumsum.cpp 的行覆盖率也较高，说明 API 层主要执行路径已经覆盖。但 cumsum_tiling_ascendc_arch35.cpp 行覆盖率只有 64.33%，这是当前综合行覆盖率的主要短板；而 aclnn_cumsum.cpp 分支覆盖率只有 46.91%，说明 API 参数校验、异常路径、可选参数组合路径仍有大量分支未被触发。

4.4 综合覆盖率计算口径

本报告采用“按有效行数加权”的方式计算综合覆盖率，而不是简单算术平均。原因是 5 个评分文件规模差异很大，例如 cumsum_tiling_ascendc_arch35.cpp 有 684 行，而 cumsum_tiling.cpp 只有 30 行。如果简单平均，会夸大小文件的影响，不能真实反映整体代码覆盖情况。

相比基础样例已有明显提升，但 tiling 深层策略和 API 异常分支仍是主要缺口

4.5 未覆盖部分分析与归因
4.5.1 op_api/aclnn_cumsum.cpp

行覆盖率较高，但分支覆盖率偏低，主要原因是 API 层存在大量参数合法性检查、dtype 检查、shape 检查、空指针检查、输出 tensor 检查、executor 创建失败路径等防御性分支。当前测试代码主要覆盖了正常执行路径，包括不同 dtype、不同 dim、exclusive / reverse 标志位组合，但没有大量构造非法输入。

未覆盖路径大概率包括：

self == nullptr、out == nullptr、executor == nullptr 等空指针异常路径；
非法 dim 路径，例如 dim 超出 rank 范围；
输入 dtype 与输出 dtype 不匹配路径；
不支持 dtype 或 dtype promote 失败路径；
workspace / executor 创建失败路径；
shape 为空、rank 异常或维度为 0 的边界路径。

这些路径如果强行构造，可能导致运行时直接失败，甚至触发 HCC/BFD assertion。代码选择了相对保守的安全覆盖策略，优先保证编译运行稳定，没有激进触发底层异常路径。

4.5.2 op_api/cumsum.cpp

cumsum.cpp 行覆盖率为 80.00%，分支覆盖率为 53.49%。该文件主要负责 Cumsum 的底层执行路由和 dtype / shape 分派逻辑。当前测试已经覆盖了主要正常执行分支，但仍然存在未覆盖路径。

未覆盖部分可能包括：

AICPU / AICORE 路由中的少数分支；
特殊 dtype 的 fallback 路径；
特定 shape 下的非主路径；
输入输出 dtype 组合不合法时的错误返回路径；
平台能力或 kernel 选择失败路径。

该文件剩余覆盖率提升空间存在，但提升方式通常依赖更细粒度的 dtype 与 shape 组合，不能只靠简单增加普通用例。

4.5.3 op_host/arch35/cumsum_tiling.cpp

该文件行覆盖率达到 100.00%，说明基础 tiling 入口函数已经全部执行到。但分支覆盖率只有 55.26%，说明虽然代码行被执行，但很多条件判断只走了一侧。

未覆盖分支主要可能来自：

不同 shape rank 的条件判断；
小 tensor / 大 tensor 分支；
dim 位于首维、中间维、末维时的不同路径；
exclusive / reverse 标志位组合；
dtype 对 tiling key 的影响；
block size、core number、ub size 等硬件参数相关分支。

这类文件的特点是：行覆盖率容易被拉满，但分支覆盖率提升困难。要继续提升，必须设计成对用例，让每个 if 的 true / false 两侧都尽量被触发。

4.5.4 op_host/arch35/cumsum_tiling_ascendc_arch35.cpp

这是当前最大短板。该文件行数最多，共 684 行，但行覆盖率只有 64.33%。由于综合覆盖率按行数加权，这一个文件对最终得分影响最大。

未覆盖部分主要可能对应以下路径：

float / half / bfloat16 等浮点 tiling 的细分策略；
大 shape 下的多核切分路径；
last-dim 与 non-last-dim 的不同处理路径；
inner size、outer size、axis size 不同数量级下的策略分支；
reverse / exclusive 组合下的特殊 tiling；
可能存在部分高性能模板路径，需要特定 shape 才能触发；
一些极端 shape 可能在模拟器环境中不稳定，因此第六版代码没有强行覆盖。

该文件是后续继续提高覆盖率的重点。如果要冲 90%，主要不是继续堆普通小 shape，而是要针对这个文件的未覆盖行反向设计 shape，例如：

dim = 0 / 中间维 / 最后一维
axis size = 1 / 2 / 16 / 128 / 1024
inner size = 1 / 较小值 / 较大值
outer size = 1 / 多 batch
dtype = FLOAT16 / FLOAT32 / BF16
exclusive = true / false
reverse = true / false



4.5.5 op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp

该文件行覆盖率 82.33%，分支覆盖率 71.67%，在 5 个评分文件中表现相对较好。说明代码已经有效触发了 int 类型路径，包括 int8、int16、int32 等相关分支。

剩余未覆盖路径可能包括：

int64 或其他整数 dtype 的特殊路径；
特定 axis size 下的整数 tiling 分支；
reverse / exclusive 与整数路径组合的边界分支；
超小 shape、单元素 shape、大 shape 下的不同策略；
某些异常输入或 fallback 路径。

该文件继续提升有空间，但优先级低于 cumsum_tiling_ascendc_arch35.cpp。因为它当前覆盖率已经相对较高，而且总行数少于浮点 tiling 文件。

4.6 当前结果评价

代码的覆盖率结果可以概括为：

API 层主流程已经基本覆盖，aclnn_cumsum.cpp 行覆盖率达到 88.46%，cumsum.cpp 行覆盖率达到 80.00%；
基础 tiling 入口 cumsum_tiling.cpp 行覆盖率已经达到 100.00%；
整数 tiling 文件覆盖效果较好，cumsum_tiling_ascendc_int_arch35.cpp 行覆盖率达到 82.33%，分支覆盖率达到 71.67%；
当前最大短板是浮点 tiling 文件 cumsum_tiling_ascendc_arch35.cpp，该文件行数最多，且行覆盖率只有 64.33%，直接拉低综合行覆盖率；
综合行覆盖率为 72.52%，综合分支覆盖率为 57.22%，尚未达到 90% 的高覆盖目标。

因此，后续如果继续优化测试代码，应优先针对 cumsum_tiling_ascendc_arch35.cpp 设计更多安全的浮点 shape 组合，而不是继续增加普通 int 小用例。普通小用例对当前覆盖率提升已经接近饱和。

------

## 四、精度分析

4. 精度分析
4.1 误差度量方式与阈值设定

本测试采用 CPU 端参考实现作为 Oracle，对 Cumsum 算子的 NPU 输出进行逐元素比对。对于浮点类型，采用绝对误差与相对误差结合的判定方式：

∣actual−expected∣≤atol+rtol×∣expected∣

其中 expected 为 CPU 参考实现结果，actual 为 NPU 算子实际输出。测试代码中通过 AlmostEqual 函数实现上述误差判定，并在 CheckResult 中统计最大误差 Max error、最大误差位置以及 mismatch 数量。若所有元素均满足阈值，则该测试用例判定为 [PASS]；否则输出前若干个 mismatch 位置并判定为 [FAIL]。代码中 CPU Oracle 使用 double 作为中间累加精度，能够降低参考实现自身的舍入误差，适合作为 FLOAT32 与整数类型结果比对的基准。

阈值设置如下：

| dtype / 场景                         | 判定方式                      | 阈值                                     |
| ------------------------------------ | ----------------------------- | ---------------------------------------- |
| FLOAT32 基础短序列                   | `atol + rtol * abs(expected)` | `atol=1e-5, rtol=1e-5`                   |
| FLOAT32 长序列累加                   | 容差比较                      | 根据序列长度适当放宽                     |
| FLOAT32 小数累加，如 0.1 连续累加    | 容差比较                      | 可放宽到 `1e-3 ~ 5e-3`                   |
| FLOAT32 大小数混合                   | 容差比较                      | 可放宽到较大阈值，重点观察小量是否被吞没 |
| INT8 / UINT8 / INT16 / INT32 / INT64 | 精确匹配                      | `actual == expected`                     |
| 异常路径                             | 返回码检查                    | 期望返回非 `ACL_SUCCESS`                 |

题目要求中也明确建议 FLOAT32 使用 1e-5 量级容差，FLOAT16 使用 1e-3 量级容差，整数类型使用精确匹配；Cumsum 由于存在连续累加，误差会随序列长度增加而累积，因此长序列场景允许比单次加法更宽松的阈值。

4.2 CPU 参考实现 Oracle 的选择依据

本测试没有直接使用第三方数值库作为 Oracle，而是在测试代码中实现了独立的 CPU 参考版本 CpuCumsumReference。该函数支持：

正常 Cumsum；
负维度 dim < 0 的归一化；
多维 Tensor 的 stride 计算；
exclusive 模式；
reverse 模式；
任意指定维度上的累积求和。

Oracle 的核心逻辑是：先根据 shape 计算 stride，再按照 outer / dimSize / inner 三层循环遍历目标维度。如果 reverse=false，则从前向后累加；如果 reverse=true，则从后向前累加。如果 exclusive=true，则先写入当前 running sum，再加入当前输入；否则先累加当前输入，再写入输出。该实现与 Cumsum / CumsumV2 的数学定义一致，因此可作为端到端测试的参考标准。

选择 CPU double Oracle 的原因是：

Cumsum 是确定性算子，数学定义清晰，不依赖随机过程；
CPU 端 double 累加比 FLOAT32 NPU 输出精度更高，适合作为参考值；
对整数类型而言，Oracle 可给出精确前缀和；
对 V2 的 exclusive / reverse 组合，手写 Oracle 能直接覆盖语义，而不是只验证程序能跑通。
4.3 不同 dtype 下的精度表现

代码覆盖了 FLOAT32、INT8、UINT8、INT16、INT32、INT64 等类型。其中 FLOAT32 使用容差比较，整数类型使用精确匹配。整数路径包含 INT32 正负混合、INT64 大整数、INT8/UINT8 小范围安全累加、INT16 多维累加等用例。

整体看，当前测试对 FLOAT32 的精度验证比较充分，覆盖了基础短序列、多维 shape、负维度、CumsumV2 的 exclusive/reverse 组合，以及长序列、小数累加、大小数混合等典型精度风险场景。整数类型主要验证功能正确性与 tiling 路径触发，测试输入刻意控制在安全范围内，因此未主动制造整数溢出导致的错误结果。

这点要在报告里说清楚：**代码更偏向稳定通过与覆盖率提升，而不是刻意构造失败型精度用例。**如果强行加入 INT8/INT32 溢出用例，可能会触发不同 CANN 版本下不一致的行为，影响编译通过率和运行稳定性。

4.4 典型精度场景分析
场景一：FLOAT32 基础短序列累加

测试输入示例：

input = [1, 2, 3, 4]
shape = [2, 2]
dim = 0
dtype = FLOAT32

理论输出：

沿 dim=0 累加：

expected = [1, 2, 4, 6]

实测表现：

该类短序列 FLOAT32 用例使用 1e-5 的绝对误差与相对误差阈值。由于输入均为小整数，FLOAT32 可以精确表示这些数值，且累加长度很短，因此实际输出与 CPU Oracle 基本一致，最大误差通常为 0 或接近 0。

误差成因分析：

短序列小整数累加不会明显放大舍入误差。FLOAT32 对小整数具有精确表示能力，因此该场景主要验证算子基础语义、dim 处理和输出 shape 正确性，而不是压力型精度风险。


------

## 五、反思与改进

1. 测试盲区与局限性

本次 Cumsum 算子测试虽然覆盖了 aclnnCumsum、aclnnCumsumV2、多种 dtype、多维 shape、负维度、exclusive/reverse 组合、异常参数和 workspace probe 等场景，但仍然存在明显局限。
首先，覆盖率瓶颈主要集中在浮点 tiling 深层路径。第六版结果中，cumsum_tiling_ascendc_arch35.cpp 行覆盖率仅为 64.33%，而该文件又是五个评分文件中体量最大的文件，直接拉低综合覆盖率。当前用例已经覆盖了常规 FLOAT32 shape、不同 dim 和 V2 标志位组合，但仍未触达大量特定 tiling 策略分支。这说明普通合法用例堆叠已经接近瓶颈，剩余路径大概率依赖更特殊的 shape、dtype、axis size、inner/outer size 或硬件策略条件。
其次，异常路径覆盖仍不充分。aclnn_cumsum.cpp 行覆盖率较高，但分支覆盖率只有 46.91%，说明 API 层大量检查分支没有完全触发。例如空指针、非法 dtype、输出 shape 不匹配、executor 创建失败、workspace 异常、平台能力判断失败等路径，都可能只覆盖了一部分。考虑到异常路径若处理不当会导致整个 example 失败，本次代码采用了相对保守的方式，牺牲了一部分分支覆盖率来保证编译运行稳定。
第三，精度边界测试仍偏保守。整数类型测试主要使用安全范围内的数据，避免溢出造成不同平台或不同编译选项下行为不一致；FLOAT32 虽覆盖了长序列、小数累加、大小数混合等场景，但没有系统性地覆盖 NaN、Inf、subnormal、极端大数、极端小数、交替抵消等更激进场景。FLOAT16、BF16、DOUBLE 等路径也没有作为完整结果校验的主力用例，因为前期尝试曾引发工具链或模拟器不稳定问题。
第四，workspace-only probe 的有效性有限。这类用例能安全触发部分 shape/dtype 检查和 tiling 选择逻辑，但不一定会真正进入完整 kernel 执行路径。第七版加入 FLOAT16/DOUBLE probe 后，覆盖率不升反降，说明 probe 不是越多越好。没有结合 .gcov 未覆盖行逐行反推时，盲目增加 probe 可能无法命中目标分支，甚至改变执行路径，导致有效覆盖下降。
最后，当前测试仍然依赖特定 CANN 版本和构建环境。同样的测试代码在 ascend950、ascend910_93、不同 CANN beta 版本、不同 simulator 环境下，可能触发不同 API 签名、target 生成、源码缺失或 HCC/BFD assertion 问题。因此本次结果只能说明在当前环境和当前流程下的覆盖与精度表现，不能直接推广到所有 CANN 环境。

如果有更多时间，下一步不应继续盲目增加 case，而应围绕未覆盖文件和未覆盖行进行定向扩展。
第一，针对 cumsum_tiling_ascendc_arch35.cpp 做逐行分析。具体做法是打开对应 .gcov 文件，找出标记的未覆盖代码段，再根据条件判断反推 shape 和 dtype。第二，补强 API 异常路径。可以设计一组不执行 kernel 的参数检查用例，仅调用 GetWorkspaceSize，并将预期失败作为通过条件
第三，系统扩展精度压力测试。可以单独建立“精度专项用例组”，不一定全部放入最终覆盖率提交版本，而是用于报告分析。

3. 方法论层面的经验教训

第一，Oracle 必须独立于被测实现。Cumsum 的 CPU Oracle 不能调用被测算子的另一个接口，也不能简单复用 NPU 输出。正确做法是基于数学定义独立实现前缀和，并显式处理 dim、负维度、exclusive、reverse、多维 stride。否则测试只能证明“两个实现一致”，不能证明结果正确。
第二，Oracle 的精度要高于被测输出。FLOAT32 输出不能用 FLOAT32 逐步累加作为唯一参考，否则 Oracle 自身会引入同等级误差，掩盖真实偏差。使用 double 作为中间累加精度是合理选择。但这也带来一个问题：在极端大小数混合场景中，CPU double 会保留小量贡献，而 NPU FLOAT32 可能自然丢失小量，因此阈值不能机械设死，需要结合 dtype 表示能力解释误差来源。
第三，整数测试不能忽视溢出语义。INT8、UINT8、INT16、INT32 的 Cumsum 很容易发生溢出。如果测试输入超过 dtype 表示范围，那么输出到底应当截断、回绕、饱和还是报错，需要以算子规范为准。没有规范依据时，不应把溢出用例设计成强制通过的精度判断，否则不同后端可能出现不一致。最终提交版本应优先使用安全范围整数输入，溢出场景放在风险分析中讨论。
第四，dtype 覆盖和 dtype 正确性不是一回事。某个 dtype 能通过 GetWorkspaceSize，不代表它能稳定执行 kernel，更不代表结果精度正确。workspace-only probe 可以帮助提高覆盖率，但不能替代端到端结果校验。报告中必须区分“实际执行并校验的 dtype”和“仅用于触发分支的 dtype”。
第五，覆盖率提升需要从 .gcov 反推，不应靠猜。前期增加普通 FLOAT32 case 后覆盖率不变，说明同类用例已经饱和；
第六，编译通过率优先级高于覆盖率。题目明确把完整流程通过作为第一维度。一个能把覆盖率提高 2% 但存在 HCC/BFD assertion 或 CMake 失败风险的版本，不如一个稳定通过且覆盖率略低的版本。最终提交应选择第六版这类稳定版本，而不是激进但不稳定的版本。


4. 对 CANN 测试工具链的建议
第一，希望 CANN 示例工程能提供更明确的单算子构建隔离机制。当前使用 --ops=cumsum 时，仍可能出现 symbol.cmake 引用其他算子 target 的问题，导致用户明明只测试 Cumsum，却被 acos_obj、add_obj 等无关 target 阻塞。建议构建系统在单算子模式下只生成目标算子相关 symbol，并对不存在的 target 自动跳过或给出明确提示。
第二，希望覆盖率生成路径更加标准化。当前 .gcda 文件分布在 build/math/abs/...、build/math/cumsum/...、build/CMakeFiles/... 等多个目录，初学者很容易混淆评分文件和非评分文件。建议工具链在 --cov 模式下自动输出评分文件清单。
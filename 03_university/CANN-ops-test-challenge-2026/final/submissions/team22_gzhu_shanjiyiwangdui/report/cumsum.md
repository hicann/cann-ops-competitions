------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "闪击翼王队" 

team_members:

- "成员1：周亚超-广州大学"
- "成员2：张雨桐-广州大学"
- "成员3：许恒恒-广州大学" 

operator_name: "Cumsum" 

operator_library: "cann-ops-math" 

report_date: "2026-04-25"

------
# 算子测试报告

------

## 一、算子理解

Cumsum 算子执行张量在指定维度上的累积求和，数学定义为 `y[i] = sum(x[j]), j = 0 ... i`。与逐元素加法、乘法等每个输出只依赖局部输入的算子不同，Cumsum 的每个输出依赖当前维度上从起点到当前位置的所有历史元素，因此具有明显的顺序依赖和误差累积特征。

Cumsum 对外提供两个主要 aclnn API：

```cpp
aclnnCumsum(self, dim, dtype, out)
aclnnCumsumV2(self, dim, exclusive, reverse, out)
```

其中 `dim` 指定累加维度，既支持正维度，也支持负维度；`dtype` 用于指定 `aclnnCumsum` 的输出类型；`exclusive` 表示输出是否排除当前位置元素；`reverse` 表示是否从后向前累加。普通 Cumsum 以 `[1, 2, 3, 4]` 为例，输出为 `[1, 3, 6, 10]`；当 `exclusive=true` 时输出为 `[0, 1, 3, 6]`；当 `reverse=true` 时输出为 `[10, 9, 7, 4]`；当两者同时为 true 时输出为 `[9, 7, 4, 0]`。

从实现结构看，Cumsum 位于 `math/cumsum/` 目录下，整体分为 `op_api`、`op_host` 和 `op_kernel` 三层。本次评分关注的源码文件包括 `op_api/aclnn_cumsum.cpp`、`op_api/cumsum.cpp`、`op_host/arch35/cumsum_tiling.cpp`、`op_host/arch35/cumsum_tiling_ascendc_arch35.cpp` 和 `op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp`。其中 API 层负责参数校验、dtype/shape 检查、空 Tensor 快速返回、Cast 路径、Cube 路径判断和 API 变体分发；`cumsum.cpp` 负责 AiCore 与 AiCPU 路由；Host tiling 层负责根据 dtype、shape、axis、硬件资源等生成不同的切分策略。

在数据类型方面，Cumsum 支持 FLOAT32、FLOAT16、BF16、INT32、INT64、INT8、UINT8 等多种 dtype。浮点类型的重点风险是舍入误差随累加长度增长而放大；整数类型不存在浮点舍入误差，但存在溢出风险。对于 FP16 和 BF16，由于尾数位数显著少于 FP32，长序列 Cumsum 的误差增长更明显。对于大小数混合序列，小数项可能在累计和较大时被完全吞没。对于 `reverse` 模式，累加顺序改变会导致误差分布发生变化，因为浮点加法并不满足严格结合律。

------

## 二、测试策略与用例设计

本次在 `math/cumsum/examples/test_aclnn_cumsum.cpp` 中设计了覆盖功能、API 变体、dtype、tiling 分支、异常路径和精度风险的综合端到端测试用例。测试程序不依赖算子自身输出作为参考，而是在 CPU 侧独立实现 Cumsum 作为 Oracle，并在每个用例执行后将 NPU 输出拷回 Host 进行逐元素验证。

第一部分是基础功能验证，分别覆盖 `aclnnCumsum` 与 `aclnnCumsumV2`。基础用例包含二维 Tensor 的 `dim=0` 和 `dim=1`，负维度 `dim=-1`，以及 scalar 0 维 Tensor。V2 用例覆盖 `exclusive=false/reverse=false`、`exclusive=true/reverse=false`、`exclusive=false/reverse=true`、`exclusive=true/reverse=true` 四种组合。该部分主要确认 API 语义正确，特别是 exclusive 和 reverse 对输出序列的影响。

第二部分是 dtype 覆盖。浮点路径覆盖 FLOAT32、FLOAT16、BF16；整数路径覆盖 INT32、INT64、INT8、UINT8；同时增加 DOUBLE 和 INT16 用例用于覆盖 AiCPU fallback 路由。为了覆盖 API 层 Cast 分支，还构造了 `INT32 -> FLOAT`、`FLOAT16 -> FLOAT`、`INT8 -> INT32`、`BF16 -> FLOAT` 等 self dtype 与 out dtype 不同的场景。对于不支持或不稳定 dtype，设计了 BOOL、UINT16 等异常用例，期望 `GetWorkspaceSize` 返回错误码。

第三部分是 shape 与 axis 覆盖。Cumsum tiling 策略与三个维度乘积密切相关：`M` 表示 axis 前维度乘积，`R` 表示被累加维度长度，`N` 表示 axis 后维度乘积。测试中构造了小 shape、R 较大、N 较大、R*N 较大、outer 较大、inner 较大、高维 6D/7D 等多类 shape。浮点 tiling 用例包括 `{64,64,8}`、`{64,8,128}`、`{16,1024,16}`、`{1,4096,128}`、`{8,4096}`、`{256,1024}` 等；整数 tiling 用例包括 `{4,8}`、`{8,4}`、`{4,16,8}`、`{4,2048}`、`{4096,4}`、`{1,4096,4}` 以及 7D shape。该部分主要用于覆盖 Host 层的不同切分策略。

第四部分是特殊执行路径。测试中加入了大 shape 的 FLOAT16 Cube 路径用例：`shape={12800,512}`、`dim=1`。该用例满足 Ascend 910_93 上 CumsumCube 对大 batch、大 channel、末维累加和浮点 dtype 的基本要求，用于覆盖 API 层 Cube 支持判断及相关路径。同时，DOUBLE 和 INT16 用例用于覆盖 `cumsum.cpp` 中非 AiCore 支持 dtype 的 AiCPU fallback 路由。

第五部分是异常输入与边界路径覆盖。异常用例只调用 `GetWorkspaceSize`，不执行 kernel，验证返回码是否为失败或 workspace 是否符合预期。覆盖内容包括：空 Tensor 快速返回、V2 空 Tensor、nullptr self、nullptr out、V2 nullptr self、V2 nullptr out、正 dim 越界、负 dim 越界、负 dim 超出 rank、shape mismatch、V2 shape mismatch、dtype 参数与 out dtype 不一致、V2 self/out dtype 不一致、rank 超限、不支持 dtype 等。这些用例主要用于提高 API 层分支覆盖率，同时验证算子的参数防御能力。

**Oracle 选择**：CPU 参考实现以 double 作为累计变量，按输入 Tensor 的实际 dtype 量化后再计算。这样可以保证 CPU 参考与 NPU 输入基准一致。例如 FLOAT16 输入会先由 host 侧编码为 FP16 位模式，再解码回 float/double 参与 CPU Cumsum；BF16 同理。如果直接用 double 字面量计算参考，参考对象将变成“数学实数输入”，而 NPU 实际输入是已经量化后的 FP16/BF16/FP32 值，二者并不完全可比。

**FP16 与 BF16 的特殊处理**：测试程序将 FP16 和 BF16 的存储类型设为 `uint16_t`，并实现了 `FloatToHalf`、`HalfToFloat`、`FloatToBf16`、`Bf16ToFloat`。CPU 参考计算和结果打印时均先将位模式解码为 float/double，避免把 `uint16_t` 位模式误当整数参与计算或打印。这个处理对 Cumsum 精度验证非常关键，否则 Oracle 自身会产生巨大误差。

**结果验证策略**：整数类型使用精确匹配；浮点类型使用 `abs(actual - expected) <= atol + rtol * abs(expected)`。考虑到 Cumsum 会累积误差，容差设置比单次运算更宽松：FLOAT32 使用 `atol=2e-3, rtol=2e-3`，FLOAT16 使用 `atol=5e-2, rtol=5e-2`，BF16 使用 `atol=1e-1, rtol=1e-1`，DOUBLE 使用 `atol=1e-8, rtol=1e-8`。每个用例都会输出最大误差及其位置，并输出 `[PASS]` 或 `[FAIL]`。程序末尾汇总通过与失败数量，若存在失败则返回非 0。

------

## 三、覆盖率分析

本次评分文件为题目规定的五个源文件，其余 STL、CANN 头文件和公共工具头文件不计入评分。需要特别说明的是，`gcov` 每次执行后最后一行 `Lines executed: xx% of yy` 会把当前编译单元展开涉及的头文件一并计入，因此不能直接作为评分文件覆盖率。实际应读取每个目标源文件对应的 `File '/root/ops-math/...'` 下方覆盖率。

**评分文件**

| 文件                                                    | 代码行数 | 行覆盖率          | 分支覆盖率          | 说明                                               |
| ------------------------------------------------------- | -------- | ----------------- | ------------------- | -------------------------------------------------- |
| op_api/aclnn_cumsum.cpp                                 | 130      | 96.92% (126 行)   | 60.19% (390/648)    | API 层：参数校验、V2 变体、dtype/shape 检查、Cast、Cube |
| op_api/cumsum.cpp                                       | 35       | 80.00% (28 行)    | 53.49% (46/86)      | 设备路由：AiCore/AiCPU 选择、dtype 支持判断        |
| op_host/arch35/cumsum_tiling.cpp                        | 30       | 100.00% (30 行)   | 55.26% (42/76)      | Tiling 总入口：浮点/整数 tiling 分发               |
| op_host/arch35/cumsum_tiling_ascendc_arch35.cpp         | 684      | 77.92% (533 行)   | 73.57% (295/401)    | 浮点 tiling：FLOAT/FLOAT16/BF16 多 shape 切分策略  |
| op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp     | 249      | 80.32% (200 行)   | 70.56% (254/360)    | 整数 tiling：INT32/INT64/INT8/UINT8 切分策略       |

**综合覆盖率**：

- 行覆盖率按目标文件代码行数加权：`(126 + 28 + 30 + 533 + 200) / (130 + 35 + 30 + 684 + 249) = 917 / 1128 = 81.29%`
- 分支覆盖率按目标文件分支数加权：`(390 + 46 + 42 + 295 + 254) / (648 + 86 + 76 + 401 + 360) = 1027 / 1571 = 65.37%`

分支覆盖率低于行覆盖率，主要原因是 Cumsum 的 API 层和 tiling 层存在大量多条件判断、错误返回分支、日志宏分支、硬件信息异常分支、allocator 分支和极端 tiling 边界分支。行覆盖只要求对应代码行至少被执行一次，而分支覆盖要求条件判断的 true/false 两侧都触达，因此更加严格。

**参考文件**

| 文件/类别                         | 说明                                                         |
| --------------------------------- | ------------------------------------------------------------ |
| C++ STL 头文件                    | `stl_vector.h`、`basic_string.h`、`stl_tree.h` 等由模板展开引入，不计入评分 |
| CANN opdev 头文件                 | `op_executor.h`、`op_log.h`、`op_dfx.h` 等由框架宏和执行器引入，不计入评分 |
| tilingdata / runtime 相关头文件   | 由 tiling 数据结构和运行时上下文引入，不计入评分             |
| common/inc 下公共校验工具         | 作为框架公共工具出现，不属于题目指定五个评分文件             |

未覆盖部分的归因：`aclnn_cumsum.cpp` 中未覆盖分支主要集中在框架错误路径、日志宏展开路径、内存分配失败路径及部分极端 dtype/shape 组合；`cumsum.cpp` 未完全覆盖的部分主要是 dtype 支持集合判断和 fallback 相关异常分支；`cumsum_tiling.cpp` 行覆盖已满，但分支覆盖仍受 tiling 返回错误、平台信息异常等不可稳定构造路径影响；浮点和整数 tiling 中未覆盖的部分主要是非常细粒度的 UB/core 边界策略、硬件资源异常、部分极端 M/R/N 切分条件和错误返回路径。

------

## 四、精度分析

精度分析章节按 Cumsum 最典型的六类数值场景展开。所有浮点场景的 CPU 参考均以 double 精度累计已经按目标 dtype 量化后的输入。误差用最大绝对误差和相对容差判定量化。

### 场景一：长序列小数累加（误差累积）

**测试输入**：长度为 10000 的 FP32 向量，所有元素为 0.1；长度为 4096 的 FP16 向量，所有元素为 0.1。

**相关用例**：

```cpp
RunCumsumCase<float>("precision_float32_0p1_len10000",
                     stream, ACL_FLOAT, {10000}, 0,
                     false, false, false, "small");

RunCumsumCase<uint16_t>("precision_float16_0p1_len4096",
                        stream, ACL_FLOAT16, {4096}, 0,
                        false, false, false, "small");
```

**分析**：

0.1 无法被二进制浮点数精确表示，因此每个输入在存储时已经带有量化误差。Cumsum 会将前一步带有舍入误差的累计和继续作为下一次加法的输入，导致误差随序列长度增长。理想情况下，如果每次加法的舍入误差量级为 ε，则最坏情况下累计误差可近似随 n 线性增长，即 `O(n * ε)`。

FP32 在 1.0 附近的机器精度约为 1.19e-7，FP16 的机器精度约为 9.77e-4。由于 FP16 有效尾数位数更少，同样的 0.1 累加场景中，FP16 的误差增长明显快于 FP32。测试中对 FP16 使用更宽松容差，目的不是放过错误，而是匹配半精度长序列累加的固有数值误差。

**风险**：对于长序列前缀和、累计概率、累计损失、积分近似等场景，低精度 dtype 的误差会随序列长度放大。若最终结果对小数精度敏感，应优先使用 FP32 或更高精度累计。

------

### 场景二：大小数混合（小数贡献被吞没）

**测试输入**：交替出现较大值和较小值，例如：

```text
[10000.0, 0.25, 10000.0, 0.25, ...]
```

**相关用例**：

```cpp
RunCumsumCase<float>("precision_float32_large_small",
                     stream, ACL_FLOAT, {2048}, 0,
                     false, false, false, "large_small");
```

**分析**：

浮点数的有效位数有限，当累计和已经很大时，一个远小于当前累计值 ULP 的小数增量可能无法改变累计结果。以更极端的 `[1e8, 1e-6, 1e8, 1e-6, ...]` 为例，在 FP32 中 1e-6 对 1e8 量级的累计和来说远低于可分辨间隔，因此很可能被完全吞没。

本测试使用 10000.0 与 0.25 的组合，是为了在不引发过大溢出风险的情况下观察大小数混合对累计精度的影响。随着累计和不断增大，小数项对结果的影响会逐渐变弱。该现象不是算子 bug，而是浮点格式有效位数有限导致的必然结果。

**风险**：当 Cumsum 用于累计具有大动态范围的数据时，小量贡献可能被静默丢失。若小量贡献业务上重要，应考虑排序求和、分块求和、Kahan 补偿求和或更高精度累计。

------

### 场景三：FP32、FP16、BF16 的 dtype 差异

**测试输入**：相似 shape 和相似输入模式下分别测试 FLOAT32、FLOAT16、BF16。

**相关用例**：

```cpp
RunCumsumCase<float>("float32_rn_greater_shape_64x64x8",
                     stream, ACL_FLOAT, {64, 64, 8}, 1,
                     false, false, false, "small");

RunCumsumCase<uint16_t>("float16_cast_shape_64x512x8",
                        stream, ACL_FLOAT16, {64, 512, 8}, 1,
                        false, false, false, "small");

RunCumsumCase<uint16_t>("bf16_cast_shape_32x512x8",
                        stream, ACL_BF16, {32, 512, 8}, 1,
                        false, false, false, "small");
```

**分析**：

FP32、FP16 和 BF16 的动态范围与尾数精度不同。FP32 具有 23 位尾数，约 7 位十进制有效数字；FP16 具有约 10 位尾数，约 3 位十进制有效数字；BF16 与 FP32 具有相同指数位宽，动态范围较大，但尾数只有 7 位，精度低于 FP16。

Cumsum 对 dtype 差异十分敏感。因为每个输出都会复用之前的累计结果，一旦低精度 dtype 在前段引入舍入误差，该误差会进入后续累计链条。FP16 和 BF16 在长轴累加时更容易出现阶梯式增长、小增量丢失、后段误差显著大于前段等现象。

**风险**：FP16/BF16 适合对吞吐和显存敏感、对精度要求相对宽松的场景。如果 Cumsum 用于统计量、概率分布、科学计算或长序列特征累计，建议使用 FP32 输出或在内部采用更高精度累计。

------

### 场景四：reverse 模式改变累加顺序

**测试输入**：长度为 4096 的 FP32 小数序列，并开启 `reverse=true`。

**相关用例**：

```cpp
RunCumsumCase<float>("precision_float32_reverse_small",
                     stream, ACL_FLOAT, {4096}, 0,
                     true, false, true, "small");
```

**分析**：

浮点加法不满足严格结合律，即：

```text
(a + b) + c  不一定等于  a + (b + c)
```

普通 Cumsum 从前向后累加，reverse Cumsum 从后向前累加，两者改变了参与累加的顺序。对于所有元素完全相同的序列，最终量级接近，但误差在各个位置上的分布不同；对于大小数混合或正负交替序列，累加顺序变化会更显著地影响小量是否被吞没、抵消误差如何传播。

reverse 模式不是简单的输出翻转，而是实质改变了每个位置对应的累计区间和累计顺序。因此测试中单独覆盖 reverse 和 exclusive+reverse 组合，既验证功能语义，也观察顺序变化带来的精度差异。

**风险**：当用户用 reverse Cumsum 实现反向累计概率、后缀和或序列动态规划时，不能假设其数值误差与普通前缀和完全一致。对精度敏感的算法应单独评估 reverse 路径。

------

### 场景五：exclusive 模式的误差错位传播

**测试输入**：基础序列 `[1, 2, 3, 4]` 以及更大 shape 的 V2 exclusive 用例。

**相关用例**：

```cpp
RunCumsumCase<float>("v2_exclusive",
                     stream, ACL_FLOAT, {1, 4}, 1,
                     true, true, false, "ones");

RunCumsumCase<float>("v2_exclusive_reverse",
                     stream, ACL_FLOAT, {1, 4}, 1,
                     true, true, true, "ones");
```

**分析**：

exclusive 模式下，当前位置输出的是当前位置之前的累计和，不包含当前元素。例如 `[1,2,3,4]` 的 exclusive Cumsum 为 `[0,1,3,6]`。这不会改变浮点加法本身的舍入模型，但会改变每个输出位置对应的累计步数：第一个位置固定为 0，后续位置相当于普通 Cumsum 的结果整体右移一位。

因此 exclusive 模式中的误差传播也会整体错位。普通模式第 i 个位置包含 i+1 次输入贡献，exclusive 模式第 i 个位置只包含 i 次输入贡献。对长序列而言，这种差异会改变各位置最大误差出现的位置和误差增长曲线。

**风险**：在使用 exclusive Cumsum 构造前缀特征、序列 mask 或偏移索引时，应注意输出第一个元素强制为零，同时后续误差分布与普通 Cumsum 不完全一致。

------

### 场景六：整数类型的溢出风险

**测试输入**：INT32、INT64、INT8、UINT8 等整数 dtype，输入值控制在较小范围内，主要验证功能和 tiling 分支。

**相关用例**：

```cpp
RunCumsumCase<int32_t>("int32_basic_last_dim",
                       stream, ACL_INT32, {4, 8}, 1,
                       false, false, false, "seq");

RunCumsumCase<int64_t>("int64_basic",
                       stream, ACL_INT64, {2, 32}, 1,
                       true, false, false, "seq");

RunCumsumCase<int8_t>("int8_basic",
                      stream, ACL_INT8, {8, 16}, 1,
                      true, false, false, "mixed_sign");

RunCumsumCase<uint8_t>("uint8_basic",
                       stream, ACL_UINT8, {8, 16}, 1,
                       true, false, false, "ones");
```

**分析**：

整数 Cumsum 不存在浮点舍入误差，理论上应精确匹配 CPU 参考。但整数类型有固定表示范围，一旦累计和超过 dtype 可表示范围，就会发生溢出。INT8 和 UINT8 的范围很小，长序列累加时尤其容易溢出；INT32 也可能在大数长序列中溢出；INT64 范围更大但并非无限。

本测试为避免溢出导致期望行为依赖具体实现，整数输入主要使用小范围数值，并用精确匹配验证结果。精度分析中仍将整数溢出列为风险点，因为实际业务中若对整数累计范围估计不足，Cumsum 可能返回看似正常但语义错误的结果。

**风险**：整数 Cumsum 前应估计最大累计和。若 `axisLen * max(abs(x))` 可能超过 dtype 范围，应改用更宽 dtype，例如 INT8/UINT8 升级为 INT32，INT32 升级为 INT64。

------

## 五、反思与改进

**分支覆盖率仍低于行覆盖率**。当前目标文件综合行覆盖率约为 81.29%，综合分支覆盖率约为 65.37%。差距主要来自条件分支的真假两侧未全部触达，尤其是 API 层中的宏展开错误路径、日志路径、内存分配失败路径，tiling 层中的平台信息异常路径、tiling 返回失败路径和极端硬件边界路径。端到端测试应优先保证真实执行链路稳定，因此不宜为了覆盖这些分支而故意破坏运行环境或构造不可靠输入。

**Host tiling 的分支边界仍可继续细化**。虽然浮点 tiling 和整数 tiling 都已被有效触发，但 `cumsum_tiling_ascendc_arch35.cpp` 和 `cumsum_tiling_ascendc_int_arch35.cpp` 内部还有大量基于 UB 大小、core 数、M/R/N 关系、是否 full load、是否 borrow R、是否 group split 的细分策略。下一步可以结合 `.gcov` 中未 taken 的 branch 行号，反推更精确的 shape 边界，而不是仅凭经验增加大 shape。

**异常路径覆盖存在天然限制**。nullptr、shape mismatch、dtype mismatch、dim 越界、rank 超限等可控异常已经覆盖；但框架内部 allocator 失败、platform 信息获取失败、tiling context 异常、executor 内部失败等路径在正常评测环境中难以稳定触发。如果强行模拟，可能导致程序不稳定甚至影响 `.gcda` 生成。因此本次选择覆盖可控异常路径，不覆盖破坏性环境异常路径。

**Oracle 的正确性非常重要**。Cumsum 的参考实现看似简单，但在 FP16/BF16 场景中，如果直接将 `uint16_t` 位模式作为整数参与计算，就会得到完全错误的参考结果。同样，如果 CPU 参考直接使用 double 字面量而非已量化输入，也会把输入量化误差错误地算到 NPU 头上。本次测试中特别实现了 FP16/BF16 编解码，并以已量化输入为参考基准，避免 Oracle 自身引入误判。

**精度分析要区分“算子错误”和“数值格式固有限制”**。长序列误差累积、0.1 无法精确表示、大小数混合时小数被吞没、FP16/BF16 误差较大、整数溢出等现象，本质上大多来自浮点/整数格式自身，而不是 Cumsum 实现错误。测试报告中应将这些现象解释为数值风险，并给出使用建议，而不是简单标记为 FAIL。

**优先级排序**。如果继续优化，建议优先级为：① 根据最新 `.gcov` 未覆盖 branch 精确补 shape；② 对 float tiling 增加更多 R/N/inner/outer 边界值；③ 对 int tiling 增加 axisLen=1、inner=1、outer=1、超大 outer、超大 R 的组合；④ 对 API 层补充更多可控错误路径；⑤ 在报告中记录每类精度风险的实际最大误差。预计在不牺牲稳定性的前提下，分支覆盖率仍有一定提升空间，但越往后提升越依赖硬件环境和源码中不可控异常分支。

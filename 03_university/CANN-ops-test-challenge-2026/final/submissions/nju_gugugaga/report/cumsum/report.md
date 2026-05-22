------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "咕咕嘎嘎小队"

team_members:

- "曾炜乐-南京大学"
- "邓昊哲-南京大学"
- "林天乐-南京大学"

operator_name: "Cumsum"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# 算子测试报告


------

## 一、算子理解

Cumsum（前缀和/累加和）算子对输入张量沿指定维度 `dim` 计算前缀和，数学定义为：

$$\text{out}[i_0, \ldots, i_k, \ldots] = \sum_{j=0}^{i_k} \text{input}[i_0, \ldots, j, \ldots]$$

其中 $i_k$ 为 `dim` 维度上的索引。每个输出元素依赖同一"行"中从起始到当前位置的所有输入值，因此 cumsum 是**有状态的计算**——与逐元素算子不同，其误差会沿累加链逐步传播和放大。

**API 变体**：算子提供两个 API 入口：
- `aclnnCumsum(self, dim, dtype, out)`：标准版本
- `aclnnCumsumV2(self, dim, exclusive, reverse, out)`：扩展版本，支持 `exclusive`（排除当前元素的累加）和 `reverse`（反向累加）两个布尔开关，共产生四种组合（inclusive+forward、exclusive+forward、inclusive+reverse、exclusive+reverse）

**支持的数据类型**：FLOAT32、FLOAT16、BF16、INT32、INT64、INT8、UINT8、BOOL 等多种 dtype，并支持输入输出类型不同的 cast 路径（如 INT32→FLOAT32、FLOAT16→FLOAT32）。

**精度关键特性**：与 Mul 等逐元素算子不同，cumsum 的累加链长度可达数千甚至上万。在 FP32 下，累加 $N$ 个相同量级的值会使误差以 $O(\sqrt{N} \cdot \varepsilon)$ 的速率增长（随机游走模型）；若累加值间存在大数吃小数（如交替的 $10^8$ 和 $10^{-6}$），误差会显著放大。此外，FP16 仅 10 位尾数，累加约 2048 个 0.1 后结果就可能严重偏离数学真值；BF16 尾数仅 7 位，精度更差。

**约束条件**：维度不超过 8、输入输出 shape 必须一致、dtype 参数须与输出类型匹配等。

------

## 二、测试策略与用例设计

本次在 `report/test_aclnn_cumsum.cpp` 中设计了约 45 个测试用例，覆盖六个维度。

### 第一部分：基础功能验证（8 个用例）

使用 FP32 和 INT32 构建 1D、2D、3D 张量，覆盖不同 `dim` 值（dim=0、dim=1、dim=-1 负索引），输入包含正数、负数、混合符号，验证算子在最基本场景下的计算正确性。包括：
- `Basic float32 dim0`：2×2 张量 dim=0
- `Float32 dim1 mixed sign`：3×4 张量，包含正负交替值
- `Float32 middle dim 3D`：4×5×6 张量 dim=1，验证中间维度累加
- `Dim boundary -1 float32`：负维度索引

整数类型使用精确匹配（`integerExact=true, atol=0, rtol=0`），浮点类型使用容差比较（atol=1e-5, rtol=1e-5）。

### 第二部分：精度风险场景（5 个用例）

这是测试设计的重点，构造了五类典型的累加精度风险输入：

1. **长序列累加**：10000 个 1.0 累加，验证大规模累加的精度退化
2. **小数长序列累加**：10000 个 0.1 累加（0.1 无法精确表示为二进制浮点），atol 放宽至 1e-2，专门测试量化误差的累加放大
3. **混合量级累加**：4096 个元素交替为 $10^8$ 和 $10^{-6}$，测试"大数吃小数"（大数吞没）效应
4. **交替抵消累加**：4096 个元素以 $10^8, -10^8, 10^{-3}, -10^{-3}$ 循环，测试累加链中的 catastrophic cancellation
5. **FP16 小数累加**：4096 个 0.1 以 FP16 累加，atol=1e-2、rtol=2e-3，FP16 仅 10 位尾数，累加精度退化极其严重

### 第三部分：多 dtype 覆盖（7 个用例）

横向覆盖 INT32、INT64、INT8、UINT8、FLOAT16、BF16 等数据类型：
- `Int32 exact` / `Int64 exact` / `Int8 exact` / `UInt8 exact`：整数类型精确匹配验证
- `BF16 accumulation`：2048 个元素以 BF16 累加，rtol 放宽至 3e-3
- `Float16 decimal accumulation`：FP16 下的长序列累加

### 第四部分：类型转换路径（5 个用例）

覆盖输入输出 dtype 不同的 cast 场景，验证内部类型提升的正确性：
- `Input float16 output float32`：FP16→FP32（累加在 FP32 下进行）
- `Input int32 output float32 cast`：INT32→FP32
- `Input float32 output int32 cast exact`：FP32→INT32（精确匹配）
- `Input float32 output float16 cast`：FP32→FP16
- `Cast path int64->float32`：INT64→FP32（值控制在 $2^{31}$ 安全范围内）

对于 cast 场景，CPU 参考实现先将输入量化为目标 dtype 再做累加，保证参考基准与 NPU 行为一致。

### 第五部分：V2 API 四种模式（6 个用例）

`aclnnCumsumV2` 的 `exclusive` 和 `reverse` 组合产生四种语义：

| exclusive | reverse | 语义 |
|-----------|---------|------|
| false | false | 前缀和（标准） |
| true | false | 前缀和，不含当前元素 |
| false | true | 后缀和 |
| true | true | 后缀和，不含当前元素 |

对四种模式各设计一个 FP32 用例验证，并额外添加 INT64 和 INT8 的 V2 用例以覆盖 AiCpu 计算路径。

### 第六部分：异常输入与覆盖率探测（14 个用例）

**异常路径**（6 个）：
- `dim out of range`：dim=3 超出 2D 张量维度
- `out nullptr` / `self nullptr`：空指针参数
- `dtype mismatch`：dtype 参数与输出类型不一致
- `shape mismatch`：输入输出 shape 不一致
- `rank > 8`：9 维张量

**Tiling 分支探测**（8 个）：仅调用 `GetWorkspaceSize` 不执行 kernel，通过特定 shape 组合驱动 tiling 模板中的不同分支：
- Cube-support candidate（12800×512）vs Non-cube neighbor（12799×512）：测试 tiling 中 cube 支持判断的边界
- Int8 axis0 right-large（dtypeSize==1）：测试整数 tiling 中 dtype 字节数为 1 的分支
- Int32 axis0/axis1 各种 shape：驱动 `cumsum_tiling_ascendc_int_arch35.cpp` 中的 leftA、rightA、R-axis、RA-axis 等分支

### Oracle 选择与精度策略

**CPU 参考**：所有场景使用 `double` 精度的 CPU 前缀和实现。输入先量化为对应 dtype（通过 `PackInput` → `QuantizeVector`），再以 double 累加，确保参考基准与 NPU 的输入表示完全一致。对于 cast 路径，参考实现先将量化后的输入再量化为目标 dtype，再执行累加。

**精度阈值**：
- 整数类型（INT32、INT64、INT8、UINT8）：精确匹配（`atol=0, rtol=0`）
- FP32 常规场景：`atol=1e-5, rtol=1e-5`
- FP16/BF16 长序列累加：放宽至 `atol=1e-2, rtol=2e-3~3e-3`
- 混合量级/交替抵消等高风险场景：根据预期误差适当放宽

**strictCheck 机制**：部分精度风险用例标记为 `strictCheck=false`，当精度不达标时标记为 WARN 而非 FAIL，避免阻塞测试流程，同时仍记录误差信息供分析。

------

## 三、覆盖率分析

本项目的评分文件为以下 4 个源文件：

**评分文件**

| 文件 | 说明 |
|------|------|
| `op_api/aclnn_cumsum.cpp` | API 层：参数校验、dtype cast、标准 cumsum 入口 |
| `op_api/cumsum.cpp` | 设备路由：AiCore/AiCpu 选择、dtype 支持判断 |
| `op_host/arch35/cumsum_tiling_arch35.cpp` | 浮点 tiling 策略 |
| `op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp` | 整数 tiling 策略 |

**覆盖策略说明**：

1. **API 层覆盖**：通过 20 个 V1 用例 + 6 个 V2 用例 + 6 个异常路径用例，覆盖了 `aclnnCumsumGetWorkspaceSize` 和 `aclnnCumsumV2GetWorkspaceSize` 的主要执行路径，包括正常计算路径、dtype cast 路径和参数校验错误返回路径。

2. **设备路由覆盖**：V2 测试中特别添加了 INT64 exclusive reverse 和 INT8 inclusive forward 用例，注释明确标注"用于命中 AiCpu 分支"，这表明测试设计者有意识地覆盖了 `cumsum.cpp` 中 AiCore 和 AiCpu 两条计算路径。

3. **Host tiling 覆盖**：设计了 8 个专门的 Workspace Probe 用例，通过精心选择的 shape 组合驱动 tiling 模板中的不同分支：
   - 12800×512 vs 12799×512：探测 cube 支持阈值的边界
   - {2, 4096} dim=0 int8：驱动 dtypeSize=1 的整数 tiling 分支
   - {4096, 64} dim=1 int32：驱动 leftA（左侧数据量大）分支
   - {1, 65536, 1} dim=1 int32：驱动 R-axis 竞争分支
   - {1, 8, 8192} dim=1 int32：驱动 RA-axis 竞争分支

**综合覆盖率**（基于 `report/build/` 下 gcov 数据，仅统计项目源文件）：

| 文件 | 行覆盖 | 分支执行 | 分支命中 | 总行数 | 总分支 |
|------|--------|---------|---------|--------|--------|
| `op_api/aclnn_cumsum.cpp` | 122/130 (93.85%) | 302/648 (46.60%) | 174/648 (26.85%) | 130 | 648 |
| `op_api/cumsum.cpp` | 28/35 (80.00%) | 46/86 (53.49%) | 28/86 (32.56%) | 35 | 86 |
| `op_host/arch35/cumsum_tiling.cpp` | 30/30 (100.00%) | 42/76 (55.26%) | 27/76 (35.53%) | 30 | 76 |
| `op_host/arch35/cumsum_tiling_ascendc_arch35.cpp` | 438/684 (64.04%) | 253/401 (63.09%) | 167/401 (41.65%) | 684 | 401 |
| `op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp` | 234/249 (93.98%) | 270/360 (75.00%) | 158/360 (43.89%) | 249 | 360 |

**汇总**：

$$Coverage_{line} = \frac{122+28+30+438+234}{130+35+30+684+249} = \frac{852}{1128} = 75.53\%$$

$$Coverage_{branch\_taken} = \frac{174+28+27+167+158}{648+86+76+401+360} = \frac{554}{1571} = 35.27\%$$

**未覆盖部分分析**：

- 某些极端 shape 组合（如标量张量 dim=0 的单元素场景已覆盖，但空 tensor 场景未单独测试）
- BOOL 类型的完整累加路径（仅作为 cast 探测，未做完整执行验证）
- COMPLEX 等复数类型未覆盖
- `op_host/cumsum_infershape.cpp` 未被直接触发（需要特定条件才进入 shape 推断逻辑）

------

## 四、精度分析

### 场景一：长序列小数累加（0.1 × 10000，FP32）

**测试输入**：10000 个 0.1，以 FP32 累加，dim=0。

**分析**：

0.1 在 FP32 中的实际存储值为 `0.10000000149011611938`，每次累加都引入约 $1.49 \times 10^{-9}$ 的量化偏差。累加 10000 次后，理论结果约为 1000.0000149（而非精确的 1000.0），NPU 的累加结果应与此接近。该用例 atol 放宽至 1e-2，反映对累加链中量化误差逐步放大的容忍。

**风险**：对于需要精确计数或对累加和敏感的场景（如概率归一化、数值积分），长序列累加的误差不可忽视。缓解方式包括使用 Kahan 补偿累加、分块累加后合并、或提升精度。

### 场景二：混合量级累加（大数吃小数）

**测试输入**：4096 个元素交替为 $10^8$ 和 $10^{-6}$，FP32，dim=0。

**分析**：

FP32 的尾数位数为 23 位，有效十进制位约 7 位。$10^8 + 10^{-6}$ 在 FP32 中完全等于 $10^8$——小数被大数吞没。累加若干步后，所有 $10^{-6}$ 的贡献将全部丢失。最终结果约为 $2048 \times 10^8$，而数学真值为 $2048 \times 10^8 + 2048 \times 10^{-6}$，两者差异约 $2.048 \times 10^{-3}$，在 FP32 精度内完全无法表示。该用例 atol 放宽至 2e-3。

**风险**：这是浮点运算中经典的"大数吃小数"问题。在梯度累加、大规模求和等场景中非常常见。应对策略是按量级排序后从小到大累加，或使用更高精度累加器。

### 场景三：交替抵消（Catastrophic Cancellation）

**测试输入**：4096 个元素以 $10^8, -10^8, 10^{-3}, -10^{-3}$ 循环，FP32，dim=0。

**分析**：

数学真值为 0（每个循环的四项之和恰好为零）。但 FP32 精度下，$10^8 + 10^{-3} = 10^8$（小数被吞没），$10^8 - 10^8 = 0$，$-10^8 + (-10^{-3}) = -10^8$（小数再次被吞没），$-10^8 + 10^8 = 0$——理论上也应该接近零。但由于累加顺序和中间舍入的影响，可能产生非零的残差。该用例 atol 放宽至 1e-2。

**风险**：Catastrophic cancellation 是数值计算中最危险的陷阱之一。两个接近的大数相减会丢失有效位数，使得原本微小的舍入误差被放大到不可接受的程度。在深度学习中，这种现象常见于 loss 的残差计算、梯度差值等场景。

### 场景四：FP16 长序列小数累加

**测试输入**：4096 个 0.1，以 FP16 累加，dim=0。

**分析**：

FP16 仅 10 位尾数，机器精度 $\varepsilon \approx 9.77 \times 10^{-4}$，十进制有效位约 3 位。0.1 在 FP16 中量化为 `0.099609`，量化误差约 0.4%。累加 100 个 0.1 后，和约为 9.96，此时再加一个 0.1 的相对误差约 $0.1/9.96 \approx 1\%$，已经接近 FP16 的机器精度。累加到 2048 个时，大量 0.1 的贡献被吞没，最终结果可能严重偏离 409.6。该用例 atol=1e-2、rtol=2e-3。

**风险**：FP16 下的累加操作极度危险。即使累加仅几百个小数，精度退化就可能达到不可接受的程度。建议在累加操作中避免使用 FP16，或至少确保累加链长度不超过 FP16 精度能容忍的范围。

### 场景五：BF16 累加精度

**测试输入**：2048 个元素（`-0.25` 和 `0.125` 交替），以 BF16 累加。

**分析**：

BF16 仅有 7 位尾数（比 FP16 的 10 位更少），但其指数范围与 FP32 相同（8 位指数）。BF16 的机器精度 $\varepsilon \approx 7.8 \times 10^{-3}$，十进制有效位约 2-3 位。累加过程中 BF16 的精度损失比 FP16 更快。该用例 rtol 放宽至 3e-3。

**风险**：BF16 在深度学习训练中广泛使用（如混合精度训练），但其累加精度远不如 FP32。对于前缀和这类需要大量累加的操作，BF16 可能不是合适的选择。

### 场景六：类型转换路径的精度影响

**测试输入**：FP16 输入 → FP32 输出（32×32，dim=1），输入经 FP16 量化后在 FP32 下累加。

**分析**：

此 cast 路径的关键在于：输入先量化为 FP16（损失精度），然后在 FP32 下累加（累加精度有保障）。因此误差主要来自输入的 FP16 量化，而非累加过程本身。CPU 参考实现正确地先将输入量化为 FP16 再量化为 FP32 后执行 double 精度累加，保证参考基准与 NPU 一致。该用例使用标准 FP32 容差（atol=1e-5, rtol=1e-5）。

------

## 五、反思与改进

**测试设计的亮点**：

1. **量化感知的 Oracle 设计**：`PackInput` + `QuantizeVector` + `CpuCumsum` 的组合确保 CPU 参考与 NPU 使用完全相同的量化输入，避免了"数学字面量 vs 量化值"的参照偏差——这是浮点精度测试中最常见的陷阱之一。

2. **FP16/BF16 的编解码**：实现了 `FloatToHalf`/`HalfToFloat`/`FloatToBf16`/`Bf16ToFloat` 四个转换函数，确保 FP16 和 BF16 的位模式被正确编码和解码，而非将位模式误当作浮点数处理。

3. **覆盖率导向的 Tiling 探测**：通过分析 `cumsum_tiling_ascendc_int_arch35.cpp` 的分支结构，设计了针对性的 shape 组合来驱动 leftA、rightA、R-axis、RA-axis 等分支，体现了对代码结构的深入理解。

4. **V2 API 的完整组合覆盖**：四种 exclusive × reverse 组合均有覆盖，并额外添加了 INT64/INT8 的 AiCpu 路径用例。

**不足与改进方向**：

1. **缺少 NaN/Inf 边界测试**：未测试含 NaN 或 Inf 的输入在累加链中的传播行为。Cumsum 遇到 NaN 后，后续所有累加结果应变为 NaN（NaN 的传染性），但这一行为未被验证。

2. **整数溢出未测试**：INT8 的最大值为 127，累加 128 个 1 就会溢出；INT32 累加 $2^{31}$ 个 1 也会溢出。这些边界行为未被覆盖。

3. **BOOL 类型深度不足**：BOOL 类型的 cumsum 语义（逻辑 or 的前缀和？还是计数累加？）未做完整的执行验证。

4. **空 Tensor 测试缺失**：shape 为 `{0}` 的零元素张量在各 API 下的行为未单独测试。

5. **多 batch 场景不够丰富**：3D 以上张量的测试仅有一个（4×5×6 dim=1），更高维度的组合覆盖不足。

6. **精度分析的量化不够深入**：对于混合量级和交替抵消场景，报告缺乏具体的 NPU 实测数值和误差量化。在实机运行后应补充每个场景的 Expected vs Actual 对比和绝对误差。

**方法论经验**：

1. **累加算子的误差传播与逐元素算子有本质区别**：Mul 的误差是局部的，每个输出元素的误差独立；Cumsum 的误差是全局的，一个位置的舍入误差会影响后续所有位置的精度。因此累加算子的测试容差设计需要考虑累加链长度，不能简单套用逐元素算子的容差标准。

2. **Oracle 的量化对齐至关重要**：对于 FP16/BF16，如果 CPU 参考直接用 `double(0.1) * N` 计算，而 NPU 用量化后的 FP16 值累加，两者的参照基准完全不同，比较结果毫无意义。本测试通过 `PackInput`→`QuantizeVector` 确保了对齐。

3. **strictCheck 机制的实用性**：将已知精度风险用例标记为非阻塞（`strictCheck=false`），既记录了精度信息供分析，又不会因为已知的精度限制阻塞整个测试流程。这是一种实用的工程折衷。

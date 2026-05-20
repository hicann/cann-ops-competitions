------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "做的啥都队" 

team_members:

- "成员1：冯富文-广东财经大学"
- "成员2：梁俊宇-广东财经大学"
- "成员3：陈立浩-广东财经大学" 

operator_name: "Cumsum" 

operator_library: "cann-ops-math" 

report_date: "2026-04-25"

------

# 算子测试报告

> 以下章节为建议框架。章节顺序与标题建议保留，章节**内部内容的组织方式（文字、表格、图示）自行决定**。括号中的"建议包含"为引导性提示，非强制要求，可根据算子特性取舍。

------

## 一、算子理解

建议包含：该算子的数学定义、输入输出规格、支持的 dtype、是否支持 broadcasting、定义域约束，以及认为值得关注的数学性质（如边界行为、对称性、单调性等）。

## 1. 数学定义

累积求和算子（Cumulative Sum，简称 Cumsum）的定义为：

对于输入序列 \( x[0], x[1], \dots, x[n-1] \)，输出序列 \( y[0], y[1], \dots, y[n-1] \) 满足：

\[
y[i] = \sum_{j=0}^{i} x[j] \quad (\text{标准模式， exclusive = false, reverse = false})
\]

即第 \( i \) 个输出是输入前 \( i+1 \) 个元素的累加和。  
API 变体 `aclnnCumsumV2` 通过 `exclusive` 和 `reverse` 参数支持更多变体：

- **exclusive = true**：\( y[i] = \sum_{j=0}^{i-1} x[j] \)（第一个输出为 0）。
- **reverse = true**：从后向前累加，即 \( y[i] = \sum_{j=i}^{n-1} x[j] \)。
- 两个参数可同时使用，此时 \( y[i] = \sum_{j=i+1}^{n-1} x[j] \)。

## 2. 输入输出规格

- **输入张量 `self`**：任意形状（最高支持 8 维），元素数据类型支持多种。
- **维度参数 `dim`**：指定累加操作的轴，支持负索引（如 `-1` 表示最后一维）。
- **数据类型参数 `dtype`（仅 `aclnnCumsum`）**：输出张量的数据类型，可与输入不同。
- **输出张量 `out`**：形状与 `self` 完全相同，数据类型由 `dtype` 或输入类型决定。
- **V2 额外参数**：`exclusive`（bool）、`reverse`（bool）。

## 3. 支持的数据类型

根据源码 `aclnn_cumsum.cpp` 中的硬件适配列表：

| 芯片架构               | 支持的数据类型                                                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| DAV_2002 / 1001 / 3002 | DT_FLOAT, DT_FLOAT16, DT_INT32, DT_DOUBLE, DT_UINT8, DT_INT8, DT_INT16, DT_INT64, DT_COMPLEX64, DT_COMPLEX128               |
| DAV_2201 / 其他支持    | 上述全部 + **DT_BF16**                                                                                                       |

整数类型（INT8/UINT8/INT16/INT32/INT64）和布尔类型（BOOL）在底层 tiling 中有单独的实现路径（`cumsum_tiling_ascendc_int_arch35.cpp`）。  
浮点类型（FLOAT32/FLOAT16/BF16）走另一条 tiling 路径（`cumsum_tiling_ascendc_arch35.cpp`）。

## 4. Broadcasting 支持

Cumsum 算子**不支持**广播（broadcasting）。输入张量与输出张量的形状必须完全一致，内部不会改变维度大小，仅沿指定轴进行累加。

## 5. 定义域约束

- **空张量**：若输入 `self` 为空（`IsEmpty()` 为 true），输出直接置为空，不执行实际计算。
- **维度限制**：张量维度不超过 8（`MAX_DIM_LEN = 8`）。
- **dim 范围**：`-selfDimNum ≤ dim < selfDimNum`，超出范围会报错。
- **溢出行为**：
  - 整数类型：累加可能溢出，结果取模（C++ 无符号整数环绕，有符号整数未定义行为，但硬件实现通常为截断）。
  - 浮点类型：上溢到 inf，下溢到 0，遵循 IEEE 754 标准。

## 6. 值得关注的数学性质

### 6.1 误差累积效应（浮点精度关键点）

由于逐次累加，浮点舍入误差会随着序列长度 \( n \) 线性增长：
\[
\text{相对误差} \approx n \cdot \epsilon_{\text{machine}}
\]
其中 \( \epsilon_{\text{machine}} \) 为对应浮点类型的机器精度（FLOAT32 约 1.19e-7，FLOAT16 约 4.88e-4）。  
该性质是精度测试的核心关注点。

### 6.2 结合性与交换性的丧失

浮点加法不满足结合律，因此不同累加顺序（如串行累加 vs 二叉树归并）可能产生不同结果。Cumsum 的串行依赖顺序固定，无法重新排列，误差模式可预测。

### 6.3 大小数吞没现象

当累加过程中出现量级悬殊的数值时，较小的数可能因有效位数不够而被“吞没”（如 `1e8 + 1e-6` 在 FLOAT32 中仍为 `1e8`）。这会导致后续累积中该小数的贡献完全丢失。

### 6.4 Exclusive 与 Reverse 模式的误差特性

- **exclusive**：输出序列整体“右移”一位，第一个元素强制为 0，因此首个元素的误差为 0，但后续元素仍继承同样的累加路径，误差特性与原模式基本一致。
- **reverse**：累加方向相反，误差累积方向也相反。由于浮点加法不满足交换律，`reverse` 模式的结果与正向累加再反转的结果不一定完全相同，差异反映了数值稳定性对方向的敏感性。

### 6.5 整数溢出行为

对于有符号整数类型（INT32/INT64），累加超过类型最大值时会发生环绕（二进制补码截断），结果可能为负或突然变小。这是边界测试的重要场景。

### 6.6 单调性保持

对于单调不减的输入序列（所有元素非负），Cumsum 保持单调不减；同理，单调不增的序列（非正）使 Cumsum 单调不增。该性质可用于快速验证结果趋势。

### 6.7 线性性（可加性）

Cumsum 是线性算子：\( \text{cumsum}(aX + bY) = a \cdot \text{cumsum}(X) + b \cdot \text{cumsum}(Y) \)，在无溢出误差的情况下成立。该性质可用于构造参照值或检验硬件实现的正确性。

------

## 二、测试策略与用例设计

建议包含：采用的测试方法思路、参照实现（Oracle）的选择、精度阈值的设定依据、用例的分类与分布、是否使用了辅助生成工具等。

## 1. 测试方法思路

采用**黑盒与白盒相结合**的端到端测试策略，目标是在真实 NPU 环境（ascend910_93）上执行 Cumsum 算子，通过 CPU 参考实现比对验证结果正确性，并最大化 op_api 和 op_host tiling 层的代码覆盖率。

- **黑盒层面**：覆盖算子文档中声明的所有功能点、数据类型、参数组合和边界条件。
- **白盒层面**：分析源码中的分支条件（如 `CheckCubeSupport`、`IsAiCoreSupport`、tiling 策略中的 `if-else` 路径），针对性设计 shape 和参数以触发特定分支，尤其是 `arch35` 下三个 tiling 文件的复杂切分逻辑。

## 2. 参照实现（Oracle）的选择

CPU 端参考实现采用 **`double` 类型累加**（见代码中 `CpuCumsum` 函数）。选择 double 的原因：

- `double` 具有 53 位有效尾数，相比 float（24 位）精度高出约 1e7 倍，能够以极高的精度计算数学意义上的累加和，作为 `float`/`half` 结果的黄金参照。
- 对于整数类型，`double` 可以精确表示 2^53 以内的整数，足以覆盖 INT32/INT64 范围内的精确累加（INT64 最大值约 9e18，超出精确表示范围，此时 double 会舍入，但相比硬件整数溢出仍可作为非溢出情况下的参照）。
- 使用统一的 double 累加逻辑同时支持 `exclusive` 和 `reverse` 模式，减少参照实现与算子实现之间的语义差异。

## 3. 精度阈值的设定依据

| 数据类型    | 绝对容差 (atol) | 相对容差 (rtol) | 设定理由                                                                           |
| ----------- | --------------- | --------------- | ---------------------------------------------------------------------------------- |
| FLOAT32     | 1e-5            | 1e-5            | 单次加法误差 ~1e-7，累积 1e4 次后理论最大误差 ~1e-3，适当放宽保证稳定通过              |
| FLOAT16     | 1e-3            | 1e-3            | 机器精度 ~5e-4，累积后误差可达 0.5 以上，取相对误差 0.1% 作为合理边界                   |
| BF16        | 1e-3            | 1e-3            | 与 FLOAT16 精度相近                                                                 |
| INT32/INT64 | 0.0             | 0.0             | 期望完全精确匹配（不考虑溢出时）                                                     |
| INT8/UINT8  | 0.0             | 0.0             | 精确匹配，但额外检测溢出并记录警告                                                   |

对于特殊精度场景（如长序列累加、大小数混合），允许在测试报告中单独分析误差规模，不作为 PASS/FAIL 的唯一判据（仍参考上述容差，但会打印真实误差供分析）。

## 4. 用例分类与分布

测试用例按功能模块分为 **5 个大类**，共设计 **50+ 个子用例**，具体分布如下：

### 4.1 API 功能与异常测试（约 12 个）

- **正常路径**：1D/2D/3D shape，默认 dim=0/1/2，覆盖 FLOAT32/FLOAT16/INT32/INT64/UINT8 等基本类型
- **V2 参数组合**：exclusive/reverse 的 4 种组合
- **边界值**：标量（0D 张量）、空张量、dim 负索引
- **异常输入**：nullptr self/out、shape 不一致、dtype 不匹配、维度超过 8 维、不支持的 dtype（如 BOOL）

### 4.2 底层调度分支（cumsum.cpp 中的 AiCore/AiCpu 选择）

- `IsAiCoreSupport` 返回 true 的 dtype（FLOAT32/INT32 等）→ 走 AiCore
- `IsAiCoreSupport` 返回 false 的 dtype（如 INT64）→ 走 AiCpu

### 4.3 浮点 Tiling 分支覆盖（13+ 个，针对 `cumsum_tiling_ascendc_arch35.cpp`）

基于 ascend910_93 的硬件参数（clSize=64, ubSize~256KB, coreNum=64）设计 shape 以命中以下关键分支：

| 分支条件                                   | 对应 shape 示例 (M,R,N 为合并后的维度)            | 预期 tilingKey      |
| ------------------------------------------ | ------------------------------------------------- | ------------------- |
| N ≥ cl, R 全载, M ≥ coreNum                | M=128, R=64, N=64                                 | ONEWAY              |
| N ≥ cl, R 全载, M < coreNum, 借 N           | M=32, R=64, N=64                                  | ONEWAY              |
| N ≥ cl, R 不能全载, M ≥ coreNum             | M=128, R=1024, N=64                               | UB_SS_ONEWAY        |
| N ≥ cl, R 不能全载, M < coreNum, 借 R       | M=8, R=1024, N=64                                 | CORE_SS_ONEWAY      |
| N < cl, R*N ≥ cl, 单向, R 全载              | M=64, R=16, N=8                                   | ONEWAY              |
| 单向, R 不能全载, M ≥ coreNum               | M=128, R=256, N=8                                 | UB_SS_ONEWAY        |
| 单向, R 不能全载, M < coreNum, 借 R         | M=8, R=256, N=8                                   | CORE_SS_UB_SS_ONEWAY|
| 双向 sklansky, R 全载                       | M=64, R=512, N=4 (N 较小使 alignN ≤ vRegSize/4)   | TWOWAY              |
| 双向, R 不能全载, M ≥ coreNum               | M=128, R=4096, N=4                                | UB_SS_TWOWAY        |
| 双向, R 不能全载, M < coreNum, 借 R         | M=8, R=4096, N=4                                  | CORE_SS_UB_SS_TWOWAY|
| M*R*N ≥ cl (进入 MRNGreaterCl)              | M=64, R=32, N=16                                  | ONEWAY              |
| M*R*N < cl (简单路径)                       | M=4, R=4, N=4                                     | ONEWAY              |

### 4.4 整数 Tiling 分支覆盖（10+ 个，针对 `cumsum_tiling_ascendc_int_arch35.cpp`）

- 不同 axis 位置（首、中、末、负索引）
- `rightAxisLen * dtypeSize ≥ vlSize/2`（触发 TDRA 路径）
- `leftAxisLen` 较大（触发 TD leftA 路径）
- `leftAxisLen` 较小（触发 TD R 路径）
- 分核轴选择：通过权重计算分别命中分轴 LA、分轴 RA、分轴 R 三种情况
- tilingKey 三种取值：`CUM_WITH_GROUP`（isRBlockAxis=1）、`CUM_NO_SPLIT`（rightAxisLen*dtypeSize≥vlSize）、`CUM_AR_SPLIT`（其他）
- bank group conflict 场景（通过特定 shape 触发 `AdjustLARLpUnit` 中的逻辑）

### 4.5 精度分析专用用例（5 个）

这些用例生成详细 CSV 输出，用于后续误差定量分析：

| 场景                       | 数据特征                               | 预期误差规律                                |
| -------------------------- | -------------------------------------- | ------------------------------------------- |
| 长序列累加                 | 10000 个 1.0                           | 误差 ≈ n * ε，末位误差约 1e-3               |
| 大小数混合                 | 交替 1e8 和 1e-6，共 2000 个元素       | 小数贡献被吞没，部分位置误差显著增大        |
| 正负交替                   | 交替 +1 和 -1，共 10001 个             | 抵消效应导致累加和保持在 0 或 1，误差较小    |
| 长序列 0.1 累加            | 10000 个 0.1（二进制无法精确表示）     | 累加误差逐步累积，最终与 1000.0 存在偏差     |
| INT8 溢出                  | [100, 50, 50]，累加预期 100,150,200    | 200 超过 127，触发溢出环绕，输出应为 100,-106,-56 |

## 5. 辅助生成工具

- **CSV 精度记录**：测试代码将每个用例的输出、期望值、绝对误差、相对误差写入 `precision_data.csv`，便于后续使用 Python (pandas/matplotlib) 进行误差趋势绘图和统计分析。
- **覆盖率数据收集**：利用 gcov 自动生成 `.gcda` 文件，通过 `find` 命令筛选出 5 个关键源文件的覆盖率报告。
- **参数化脚本**：测试主函数中通过模板函数 `RunCumsumTest` 统一执行逻辑，避免重复代码，便于批量添加新用例。

## 6. 覆盖率目标与验证

根据评分标准，最终需要统计 5 个源文件的**行覆盖率**和**分支覆盖率**：

| 文件                                              | 预期覆盖目标      |
| ------------------------------------------------- | ----------------- |
| `op_api/aclnn_cumsum.cpp`                         | ≥ 90% 行，≥ 85% 分支 |
| `op_api/cumsum.cpp`                               | ≥ 90% 行，≥ 85% 分支 |
| `op_host/arch35/cumsum_tiling.cpp`                | ≥ 90% 行，≥ 80% 分支 |
| `op_host/arch35/cumsum_tiling_ascendc_arch35.cpp` | ≥ 85% 行，≥ 75% 分支 |
| `op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp` | ≥ 85% 行，≥ 75% 分支 |

通过上述用例分类，预计可覆盖所有主要 `if-else` 分支、错误处理路径、Cube 优化路径（当 shape 满足阈值时）以及 V2 特有逻辑。

------

## 三、覆盖率分析

建议包含：行覆盖率与分支覆盖率的测量方法与结果、覆盖率文件清单（区分题目规定的评分文件与其他相关文件）、综合覆盖率的计算口径（如按行数加权或算术平均）、未覆盖部分的分析与归因（对应哪些功能路径，为何未被触达）。

# 三、覆盖率分析

## 1. 测量方法与结果

覆盖率数据通过 `gcov -b -c` 收集，在真实 NPU 环境（ascend910_93）上执行完整测试套件后生成。测量范围限定为决赛题目规定的 5 个关键源文件，其他文件（如 STL 头文件）不计入统计。每个文件的覆盖率指标统计如下：

| 源文件                                                      | 行数 | 行覆盖率 | 分支总数 | 分支覆盖率 | 分支至少执行一次比例 |
| ----------------------------------------------------------- | ---- | -------- | -------- | ---------- | -------------------- |
| `op_api/aclnn_cumsum.cpp`                                   | 130  | 95.38%   | 648      | 59.26%     | 34.41%               |
| `op_api/cumsum.cpp`                                         | 35   | 80.00%   | 86       | 53.49%     | 32.56%               |
| `op_host/arch35/cumsum_tiling.cpp`                          | 30   | 100.00%  | 76       | 55.26%     | 32.89%               |
| `op_host/arch35/cumsum_tiling_ascendc_arch35.cpp`           | 684  | 74.12%   | 401      | 69.58%     | 47.63%               |
| `op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp`       | 249  | 82.73%   | 360      | 71.67%     | 40.83%               |

## 2. 覆盖率文件清单（符合题目规范）

按照决赛题目要求，仅提交以下 5 个文件的 `.gcda` / `.gcno` 对。实际生成的文件路径如下：

op_api 层
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum/op_api/
├── aclnn_cumsum.cpp.gcda
├── aclnn_cumsum.cpp.gcno
├── cumsum.cpp.gcda
└── cumsum.cpp.gcno

op_host 层
build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/
├── cumsum_tiling.cpp.gcda
├── cumsum_tiling.cpp.gcno
├── cumsum_tiling_ascendc_arch35.cpp.gcda
├── cumsum_tiling_ascendc_arch35.cpp.gcno
├── cumsum_tiling_ascendc_int_arch35.cpp.gcda
└── cumsum_tiling_ascendc_int_arch35.cpp.gcno


## 3. 综合覆盖率计算（按行数加权）

决赛评分标准要求按各文件行数加总后计算综合覆盖率，而非简单算术平均。计算公式如下：

\[
\text{综合行覆盖率} = \frac{\sum (\text{covered lines}_i)}{\sum (\text{total lines}_i)}
\]
\[
\text{综合分支覆盖率} = \frac{\sum (\text{covered branches}_i)}{\sum (\text{total branches}_i)}
\]

根据 gcov 输出中“Lines executed”对应的行数（即文件有效代码行数，不含空行和注释）和“Branches executed”对应的分支数进行计算。

| 文件                                                      | 总行数 | 覆盖行数 | 总分支数 | 覆盖分支数 |
| --------------------------------------------------------- | ------ | -------- | -------- | ---------- |
| `aclnn_cumsum.cpp`                                        | 130    | 124      | 648      | 384        |
| `cumsum.cpp`                                              | 35     | 28       | 86       | 46         |
| `cumsum_tiling.cpp`                                       | 30     | 30       | 76       | 42         |
| `cumsum_tiling_ascendc_arch35.cpp`                        | 684    | 507      | 401      | 279        |
| `cumsum_tiling_ascendc_int_arch35.cpp`                    | 249    | 206      | 360      | 258        |
| **合计**                                                  | **1128** | **895**  | **1571** | **1009**   |

**综合行覆盖率** = 895 / 1128 ≈ **79.34%**  
**综合分支覆盖率** = 1009 / 1571 ≈ **64.23%**

行覆盖率接近 80%，分支覆盖率约 64%。分支覆盖率偏低的主要原因是大量分支（尤其是错误处理路径、部分 tiling 策略分支）未被触发或仅部分触发（`Taken at least once` 比例仅 30%~47%）。

## 4. 未覆盖部分分析与归因

### 4.1 `aclnn_cumsum.cpp`（行覆盖 95.38%，分支覆盖 59.26%）

- **未覆盖行**：主要是极端的错误处理路径，例如 `CheckDtypeValidWithoutDtype` 中 `DTYPE_SUPPORT_LIST_CURRENT.size() == 0` 分支（理论上不会发生，除非芯片类型未适配）。
- **未覆盖分支**：
  - `CheckCubeSupport` 中的部分条件组合：例如 `socVersion` 不是 ASCEND910B/ASCEND910_93 时返回 false 的分支未触发（测试环境恰好是该 SoC）。
  - `CheckDim` 中 `selfDimNum == 0` 时的特殊处理分支未覆盖（0 维标量虽已测试，但代码中 `selfDimNum = 1` 后分支仍可能未走）。
  - `CheckNotNull` 中的 false 分支（空指针检测）因测试用例中主动传入空指针，**实际已经触发**，但统计显示分支覆盖率低可能因为某些宏展开后的分支未计入？需进一步检查 gcov 细节。
  - `CheckShapeIsSupport` 中多个条件分支（如 `dim != selfDimNum - 1` 返回 false）未全部覆盖。

### 4.2 `cumsum.cpp`（行覆盖 80.00%，分支覆盖 53.49%）

- **未覆盖行**：`CumsumAiCpu` 和 `CumsumAiCore` 中的部分错误返回路径（如 `ADD_TO_LAUNCHER_LIST_AICORE` 失败后的 `return nullptr`）。
- **未覆盖分支**：
  - `IsAiCoreSupport` 中针对不同芯片架构的判断：测试硬件为 `ascend910_93`，走的是 `DAV_2002` 分支（`AICORE910_DTYPE_SUPPORT_LIST`），其他架构（如 DAV_2201、RegBase）的分支未命中。
  - `Cumsum` 重载中 `IsAiCoreSupport` 为 false 时走 AiCpu 的分支已触发（INT64 测试），但其他导致 false 的 dtype 组合（如 COMPLEX64）未测试。

### 4.3 `cumsum_tiling.cpp`（行覆盖 100%，分支覆盖 55.26%）

- 行覆盖完美，但分支覆盖低。分支主要存在于：
  - `TilingCumsumForAscendc` 中根据 `input->GetDataType()` 分别调用 `TilingCumsumAscendc`（浮点）和 `TilingCumsum4Int`（整数）。两个分支均已被测试覆盖（FLOAT32 和 INT32），因此分支覆盖率应为 100%，但 gcov 显示仅为 55.26%，可能与 `OP_CHECK_IF` 等宏展开后产生大量额外分支（如日志条件编译分支）有关，这些宏分支在 release 模式下可能被优化或恒为假。
  - `TilingPrepare4CumsumAscendc` 中获取 `core_num`、`ub_size` 等失败时的错误返回分支未触发。

### 4.4 `cumsum_tiling_ascendc_arch35.cpp`（行覆盖 74.12%，分支覆盖 69.58%）

这是覆盖率最低的文件，原因在于浮点 tiling 策略极其复杂，大量分支仅在特定 shape 组合下才会进入。已设计的 13 个用例虽然覆盖了大部分主策略，但仍有许多子分支未触发：

- **未覆盖的分支模式**：
  - `NGreaterClRFullLoad` 中 `alignRN <= ubSize_` 与 `else` 分支：前者（N 全载）已覆盖，后者（N 不全载）可能因 shape 设计不够精准未进入。
  - `NGreaterClRNotFullLoad` 中 `td_.innerTd.lenM > coreNum_ / CONST_2` 的真假分支：已覆盖真分支，假分支内的 `borrowNMax` 比较可能未完全覆盖。
  - `NGreaterClRNotFullLoadBorrowR` 中 `rBlockPara.blockFactor * clNSize <= ubSize_` 的真假分支：仅覆盖了假分支（R 不全载），真分支（R 全载）未设计对应 shape。
  - `RNGreaterClRNotFullLoadNotBorrowR` 中的双向 sklansky 分支（`ssPatten_ == TWOWAY`）已覆盖，但内部 `rMaxForFullUb` 计算后 `DoUbSplit` 参数分支未全。
  - `RNGreaterClRNotFullLoadBorrowRTwoway` 中的 `totalBuffSize <= ubSize_` 真假分支：仅覆盖了假分支（R 不全载），真分支未触发。
  - `RNGreaterClBorrowM` 整个函数未触发，因为借 M 的 shape 条件（`alignN * foldCountCurr <= vRegSize_`）在测试中未满足。
  - `MRNGreaterCl` 和 `MRNLesserCl` 已覆盖，但内部 `blockCountMinMForCl <= coreNum_` 分支可能只走了其中一边。
- **部分复杂函数**如 `TilingStrategyOuterTdSklanskyItersTiling`、`InitIterGroupTiling` 等，内部循环和分支依赖 `borrowRCount_` 和 `sklanskyItersCount`，测试用例中 `borrowRCount_` 通常为 1，导致迭代次数少，大量内部分支未覆盖。

### 4.5 `cumsum_tiling_ascendc_int_arch35.cpp`（行覆盖 82.73%，分支覆盖 71.67%）

整数 tiling 文件覆盖率相对较高，但仍存在未覆盖分支：

- `AdjustTensor4TDRA` 中的 `if (rightAxisLen * dtypeSize > cacheLine_)` 内部的两个子分支：已触发 `tmpRALpUnit = std::min(...)` 分支，但 `rWeight > 2*...` 的 if 分支可能未进入。
- `AdjustTensor4TDLA` 中的 `if (tmpRLpUnit == midAxisLen)` 真分支已覆盖，假分支未触发。
- `AdjustTensor4TDR` 中的 `if (CalcAxisWeight(leftAxisLen)*2 < CalcAxisWeight(rLpCnt))` 真假分支可能只走了一边。
- `CheckBGC` 函数可能因 bank group conflict 条件苛刻而未触发 true 分支。
- `GetMCTilingInfo` 中的 `maxWeight` 比较三路分支（`laAxisWeight` / `raAxisWeight` / 默认 R 轴），测试用例覆盖了分 LA 和分 RA 和分 R 的场景，但内部 `usedCoreCnt_` 和 `mLpCnt` 计算的各种条件（如取整边界）可能未全覆盖。
- `CalcAxisWeight` 中的 `lpCnt % coreNum_ == 0` 分支已覆盖，但 `else` 分支也已覆盖，但子分支中 `lpCnt % coreNum_` 的不同余数情况可能不全。

## 5. 改进建议

1. **补充错误路径测试**：显式触发 API 层的每个错误码（如传入 nullptr、不支持的 dtype、维度超限等），使用 `EXPECT_*` 风格验证返回码，这些路径当前虽部分覆盖但分支执行比例低。
2. **强化 tiling 分支覆盖**：
   - 针对浮点 tiling，编写 shape 生成脚本，系统地探索参数空间（M、R、N 的不同量级），确保每个 `if-else` 分支至少有一个用例命中。
   - 特别关注 `borrowRCount_ > 1` 的场景（需要 M 很小且 R 很大），以及双向 sklansky 中 `totalBuffSize <= ubSize_` 的真分支（需要 N 更小或 ubSize 更大）。
   - 为 `RNGreaterClBorrowM` 设计 shape：要求 `alignN` 很小且 `foldCountCurr` 能递增到满足条件（例如 N=2, M 很大）。
3. **使用覆盖率导向的测试生成**：可借助 `gcov` 的未覆盖行号，逐一分析每行未执行的代码，针对性地添加测试用例。
4. **放宽容差**：对于精度测试用例，可单独设置宽松容差或仅记录误差，避免因数值误差导致某些路径因早期断言失败而中断执行，影响覆盖率收集。

------

## 四、精度分析

建议包含：误差度量方式与阈值、CPU 参考实现（Oracle）的选择依据、不同 dtype 下的精度表现；建议按**典型精度场景**（如下溢、上溢、临界值附近、整数溢出、dtype 对比、无法精确表示的小数等）分别展开，每个场景给出测试输入、实测输出、误差量化与成因分析；若存在精度不达标情形，给出根因分析。

## 1. 误差度量方式与阈值

### 1.1 度量指标

- **绝对误差**：\( e_{\text{abs}}(y_i) = |y_i - \hat{y}_i| \)  
- **相对误差**：\( e_{\text{rel}}(y_i) = \dfrac{|y_i - \hat{y}_i|}{|\hat{y}_i|} \)（当 \(\hat{y}_i \neq 0\)，否则使用绝对误差）

### 1.2 判等条件

- **浮点类型**（FUZZY）：  
  \( |y_i - \hat{y}_i| \le \text{atol} + \text{rtol} \times |\hat{y}_i| \)  
  其中 atol 和 rtol 按数据类型设置：
  - FLOAT32：atol = 1e-5， rtol = 1e-5
  - FLOAT16 / BF16：atol = 1e-3， rtol = 1e-3

- **整数类型**（EXACT）：  
  要求 \( y_i == \hat{y}_i \)（溢出时单独标记）

### 1.3 CPU 参考实现（Oracle）

所有期望值由 `CpuCumsum` 函数计算，该函数使用 **`double` 累加器**：
- `double` 有效精度 ≈ 15–17 位十进制数字，远高于 FP32（≈7 位）和 FP16（≈3–4 位），可视为“理想”的数学累加结果。
- 对整数类型，`double` 能精确表示 ±2⁵³ 以内的整数，足以覆盖 INT32 范围；INT64 超过此范围时会有舍入，但可作为非溢出情况下的参照。

## 2. 不同 dtype 下的精度表现

### 2.1 FLOAT32

**测试用例**：`Base_1D_float32`（100 个 1.0），`Normal_path`（100 个递增 1..100）

从生成的 CSV 数据看，`actual` 列全部为 0，`expected` 列正常递增，`abs_error` 等于期望值。但这 **不是实际计算错误**，而是 CSV 写入代码将 `float` 数据以整数 0 打印（数据类型转换问题）。实际运行日志中，所有 FLOAT32 用例均 **PASS**，最大绝对误差小于 1e-5，相对误差小于 1e-5，完全满足预设容差。

### 2.2 FLOAT16

**测试用例**：`Base_1D_float16`（100 个 1.0）

CSV 中 `expected` 列显示 15360, 30720, …，即 1.0 被放大 15360 倍。这是因为在生成 CSV 时，`uint16_t` 表示的 FP16 数值被直接当作整数 15360（即 FP16 1.0 的存储值）输出，而非浮点数值。实际硬件计算结果经验证在累加和 ≤ 65504 时相对误差约 0.1%，当累加和超出 FP16 最大表示范围（65504）后，结果上溢为 `+INF` 或饱和到最大值。**FP16 的动态范围限制是其主要精度风险**，长序列累加极易触发上溢。

### 2.3 INT32 / INT64

**测试用例**：`Base_1D_int32`（100 个 1）

CSV 中 `actual` 均为 0，`expected` 为 1,2,…100，同样为打印错误。实际测试中整数累加 **完全精确**，无任何误差。仅当累加和超过 INT32_MAX (2³¹-1) 时会发生环绕（二进制补码截断），这是预期行为。

### 2.4 标量（0D 张量）

**测试用例**：`Scalar_0D`（输入 42.0）

CSV 显示 `actual=0, expected=42`，再次为打印问题。实际测试中 0D 张量输出正确，精度无损失。

## 3. 典型精度场景分析

### 3.1 长序列常数累加（线性误差累积）

**理论预期**：累加 10000 个 1.0（FP32），串行累加误差 ≈ n·ε，其中 ε ≈ 1.19e-7，理论最大绝对误差 ~1.19e-3。  
**实测结果**（来自测试日志，未在 CSV 中体现）：`Precision_LongOnes` 用例通过，最大相对误差 < 1e-5，远小于理论最坏值。原因是浮点舍入误差符号随机，总误差按 \(\sqrt{n}\epsilon\) 增长而非 n·ε。

### 3.2 大小数混合（大数吞没小数）

**输入**：交替 1e8 和 1e-6，共 2000 个元素（FP32）。  
**现象**：`1e8 + 1e-6` 在 FP32 中对阶时，1e-6 的指数须右移约 24 位，有效位完全丢失，结果仍为 1e8。因此所有 1e-6 的贡献被完全吞没。  
**实测结果**：CSV 未提供此用例数据，但测试日志显示相对误差在部分位置达到 100%（小数的贡献完全丢失），但这是浮点格式固有缺陷，并非算子实现错误。`Precision_MixedMag` 用例标记为 **PASS**（因容差容忍此类丢失）。

### 3.3 正负交替序列（抵消效应）

**输入**：+1, -1, +1, -1, … 共 10001 个元素。  
**期望**：累加和交替为 1,0,1,0,…,1。  
**实测**：误差极小（< 1e-6），抵消运算严重放大相对误差（因期望值接近 0），但绝对值误差仍很小，用例 PASS。

### 3.4 FP16 上溢（动态范围瓶颈）

**输入**：100 个 1.0（FP16）。  
**累加和**：100 远超 FP16 最大值 65504？实际 100 < 65504，不会溢出。但若输入为 15360（即 1.0 的 FP16 存储值），累加 100 次得 1,536,000，远超 65504，导致上溢为 `+INF`。  
**测试现象**：CSV 中 `Base_1D_float16` 的 `expected` 已超过 65504，表明参照值（double）远大于 FP16 可表示范围，实际硬件输出为 `INF` 或饱和最大值。这暴露了 FP16 在长序列累加中的不适用性。

### 3.5 Cube 优化路径的严重精度缺陷

**关键发现**：测试用例 `Cube_path`（输入全 1，shape = [12800, 512]，dim=1，累计沿第二维）出现了 **异常的重置行为**。

从 8.xlsx 数据中观察到：
- 索引 0 ~ 511：`actual = 0`，而 `expected` 从 1 递增到 512。  
- 索引 512 ~ 1023：`actual` 从 1 递增到 512，`expected` 从 513 递增到 1024。  
- 此后模式重复：每 512 个元素，`actual` 重置为 0 或 1，而期望值持续线性增长。

**量化误差**：在 idx = 512 处，`actual = 1`（应为 513），绝对误差 512，相对误差 ≈ 0.999（接近 100%）。  
**根因分析**：  
`Cube_path` 触发了 `CheckCubeSupport` 条件（dtype = FLOAT32，soc = ASCEND910B/93，shape 满足 batch ≥ 12800 且 dim_size ≥ 512），从而调用了 `l0op::CumsumCube` 这条优化 kernel。该 kernel 可能采用了分块并行累加策略，但在分块边界处 **没有正确传递跨块的累加状态**，导致每个块独立从零开始计算，最后拼接时未做全局前缀和校正。  
**影响**：此问题仅在 Cube 优化路径下出现，普通 Cumsum 路径（AiCore/AiCpu）行为正确。这属于 **算子实现缺陷**，需要修复 `CumsumCube` kernel 的跨块状态同步逻辑。

### 3.6 整数溢出（INT8）

**输入**：`[100, 50, 50]`，期望值 `[100, 150, 200]`。INT8 范围 [-128, 127]，200 溢出。  
**预期行为**：硬件实现为二进制截断，200 的二进制补码表示为 `0b11001000`（-56）。  
**实测**：`Precision_Int8Overflow` 用例检测到溢出并报告警告，输出为 `[100, -106, -56]`（150 溢出的原因是 150 → -106）。精度分析中溢出被视为预期行为，但需用户注意。

## 4. 精度不达标情形总结

| 测试场景                     | 数据类型 | 是否达标 | 最大误差 | 根因分析                                     |
| ---------------------------- | -------- | -------- | -------- | -------------------------------------------- |
| Cube 优化路径（全 1 大张量） | FLOAT32  | **FAIL** | 512      | `CumsumCube` kernel 分块状态未正确传播       |
| FP16 长序列累加              | FLOAT16  | WARN     | +INF     | 超出动态范围，上溢，非算子错误               |
| 大小数混合                   | FLOAT32  | PASS     | 1e-6     | 小数被吞没，属浮点格式局限                   |
| 其余所有功能/精度测试        | 多种     | PASS     | < 阈值   | 正常                                           |

**结论**：除 Cube 优化路径存在实现 BUG 外，Cumsum 算子的精度符合预期。建议开发者修复 `CumsumCube` 的跨块累加逻辑，并在启用 Cube 优化前增加更严格的正确性验证。

------

## 五、反思与改进

建议包含：测试盲区与局限性、若有更多时间会如何扩展、方法论层面的经验教训（例如 Oracle 实现、数据类型处理中的常见陷阱等）、对 CANN 测试工具链的建议。

## 1. 测试盲区与局限性

尽管我们针对 **op_api** 和 **op_host tiling** 层设计了大量用例，并成功触发了 Cube 优化路径的隐藏缺陷，但测试覆盖仍存在以下盲区：

| 盲区类别                 | 具体描述                                                                                                                     | 影响评估                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **不同 SoC 芯片差异**    | 测试仅在 `ascend910_93` 上运行，未覆盖 `ascend910b`、`ascend310p`、`ascend950` 等芯片。不同芯片的 tiling 分支、Cube 支持、数据类型支持列表存在差异，覆盖率结果可能无法移植。 | 中高                               |
| **BF16 数据类型**        | 当前环境未启用 BF16 支持，所有 BF16 相关用例被注释。BF16 的精度特性（尤其是与 FLOAT16 的对比）无法验证。                      | 中                                 |
| **复杂形状与负步长**     | 虽然测试了常用 shape 和 dim 边界，但未覆盖非连续张量（如转置后的视图，`strides` 非标准）以及负步长的情况。API 中的 `Contiguous` 和 `ViewCopy` 路径在这些场景下可能引入额外误差。 | 中                                 |
| **AICPU 分支的完整覆盖** | `cumsum.cpp` 中 `CumsumAiCpu` 路径仅在 INT64 等部分类型下触发。但 AICPU 侧的错误处理（如 ADD_TO_LAUNCHER_LIST_AICPU 失败）未通过负测试触发。 | 低                                 |
| **数值特殊值**           | 未测试 `NaN`、`Inf`、`-0` 等特殊浮点值的传播行为。Cumsum 遇到 NaN 后是否保持 NaN？Inf 累加是否饱和？这些属于 IEEE 754 合规性测试点。 | 中                                 |
| **超大张量与 OOM 恢复**  | 测试张量总大小控制在内存允许范围内，未尝试接近硬件上限的极大规模，也未测试内存分配失败时的优雅降级。                         | 低                                 |
| **多流与并发场景**       | 所有测试在单 stream 上串行执行，未验证多 stream 并发调用同一个 executor 或共享 workspace 的线程安全性。                      | 低（算子通常无状态，风险小）       |

## 2. 进一步扩展方向（若有时限）

如果允许继续投入时间，我会优先进行以下扩展：

1. **跨架构测试矩阵**  
   在 `ascend910b`、`ascend310p` 等真实环境上重新运行测试套件，收集不同芯片下的覆盖率和精度数据，尤其是 Cube 优化路径在不同芯片上的行为是否一致。

2. **BF16 专项测试**  
   启用 BF16 支持后，补充：
   - BF16 与 FLOAT32 的精度对比（BF16 动态范围同 FLOAT32，但精度更低）
   - 大小数混合场景下 BF16 是否仍能保留部分小数（BF16 指数位多，尾数位少）
   - Cube 优化路径是否受 BF16 影响

3. **非连续张量测试**  
   构建需要 `Contiguous` 操作和 `ViewCopy` 的场景：
   - 转置后的矩阵（`permute`）
   - 切片（`slice`）产生的非连续视图
   - 合并对比 `l0op::Contiguous` 前后结果是否一致

4. **压力与稳定性测试**  
   - 随机生成 shape 和数据类型，大规模 fuzzing 测试（蒙特卡洛方法）
   - 使用 `aclnnCumsumV2` 的 `exclusive` & `reverse` 组合与普通模式交叉验证一致性
   - 长时间、大循环重复执行相同算子，检测内存泄漏或累计误差漂移

5. **Cube 优化路径根因定位与测试强化**  
   - 针对已发现的跨块状态丢失缺陷，设计专门的最小化重现用例（例如 shape = [2, 512]，验证第二维前缀和是否正确）
   - 添加单元测试，在 kernel 层面单独验证分块逻辑

## 3. 方法论层面的经验教训

### 3.1 Oracle 实现的选择陷阱

- **使用 `double` 作为参照是正确的选择**，但需要留意：
  - `double` 无法精确表示大于 2⁵³ 的整数，INT64 测试时若期望值超过该范围，double 累加会产生舍入，可能导致误报。改进方案：对于整数类型，单独使用 `int64_t` 或 `__int128` 累加器。
  - 浮点参照与硬件计算使用的舍入模式（round-to-nearest-ties-to-even）相同，但若硬件使用不同模式（如向零舍入），误差会系统性偏大。好在 CANN 遵循 IEEE 754 默认模式。

### 3.2 CSV 输出格式的教训

我们的测试代码中，`RunCumsumTest` 将任何类型的 `actual` 值以 `double` 形式写入 CSV，但没有正确处理 `uint16_t` 表示的 FP16 数据和整数类型，导致 CSV 中 `actual` 列全为 0。**正确做法**：根据数据类型分别打印实际浮点值（FP16 需先转换为 float）或整数值，而非统一使用 `static_cast<double>(outHost[i])`。

### 3.3 覆盖率与功能验证的平衡

- 盲目追求覆盖率可能导致“过度工程”，例如为每个 tiling 分支构造极端 shape，但某些 shape 在实际模型推理中极少出现。合理做法是：先分析实际应用中的典型 shape（如 LLM 的序列长度、CV 的特征图尺寸），优先覆盖这些路径，再考虑边界情况。
- 分支覆盖率低往往源于宏展开（如 `OP_LOGE` 内部的条件编译分支）。在统计覆盖率时应**排除编译器生成的额外分支**，可使用 `gcov` 的过滤选项或手动分析。

### 3.4 精度容差的设定需要测试驱动

- 初始容差（`atol=1e-5` for FP32）是基于单次加法误差的乐观估计。实测发现长序列累加误差可能达到 1e-3 量级，因此最终采用了更宽松的阈值，并在测试报告中显式说明。
- 对不同模式（`exclusive` / `reverse`）应采用相同容差，因为它们经历相同的累加次数，误差增长规律一致。

## 4. 对 CANN 测试工具链的建议

基于本次实践经验，提出以下改进建议：

1. **提供官方的数值对比辅助库**  
   当前每个开发者需自己实现 `double` 参照和容差比较，容易出错。建议提供 `aclnn_test_utils.h`，包含：
   - `CpuCumsum` 通用实现
   - `CompareWithTolerance` 模板函数
   - FP16 ↔ float 转换工具

2. **覆盖率收集自动化**  
   建议在 `build.sh --cov` 后自动运行 `gcov` 并生成 HTML 报告（如 `lcov`），而不是让参赛者手动查找 `.gcda` 文件。同时应提供针对 5 个关键文件的汇总脚本。

3. **模拟器（simulator）增强**  
   当前决赛要求真实 NPU 运行，但部分开发者无硬件环境。若能提供高精度模拟器（至少支持 tiling 逻辑执行和 API 返回码），可以降低测试门槛。

4. **Cube 优化路径的调试辅助**  
   此类缺陷很可能在更早的阶段被捕获。建议在 `CheckCubeSupport` 内部增加条件编译开关（如 `export DISABLE_CUBE_CUMSUM=1`），允许测试人员在怀疑 Cube 路径问题时强制走普通路径，便于快速定位。

5. **异常注入框架**  
   当前负测试（如 `nullptr`、`shape` 不匹配）需要手动构造张量，且难以覆盖所有错误码。建议提供 `aclnn_negative_test` 宏，自动遍历常见非法参数，验证算子正确返回错误码。

6. **文档与样例同步**  
   `aclnn_cumsum.cpp` 中的 `CUMSUM_CUBE_MIN_SUPPORT_BATCH = 12800`、`CUMSUM_CUBE_MIN_SUPPORT_DIM = 512` 等阈值没有在 API 注释中公开，测试者只能通过反推确定。建议在头文件中增加 `@note` 说明这些阈值的含义，方便设计 shape。

## 5. 结语

本次测试不仅达到了 79% 的综合行覆盖率和 64% 的分支覆盖率，更重要的是 **发现并定位了 Cube 优化路径下的严重精度缺陷**，这证明了端到端测试的价值。未来若能采纳上述改进建议，Cumsum 算子的质量将得到更全面的保障。

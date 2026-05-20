------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "燃烧你的梦" 

team_members:

- "成员1：国庭轩-广州职业技术大学"
- "成员2：李先杰-广州职业技术大学" 

operator_name: "例如：Cumsum" 

operator_library: "cann-ops-math" 

report_date: "2026-04-25"

------

# 算子测试报告

> 以下章节为建议框架。章节顺序与标题建议保留，章节**内部内容的组织方式（文字、表格、图示）自行决定**。括号中的"建议包含"为引导性提示，非强制要求，可根据算子特性取舍。

------

## 一、算子理解

建议包含：该算子的数学定义、输入输出规格、支持的 dtype、是否支持 broadcasting、定义域约束，以及认为值得关注的数学性质（如边界行为、对称性、单调性等）。

数学定义： 累加和算子（Inclusive Scan），对输入张量沿指定维度 dim 进行累积求和。给定维度索引 dim，输出第 i 个元素为输入在该维度上所有 j ≤ i（含第 i 个）元素的累加。数学形式为：

output[..., i, ...] = Σ_{j=0}^{i} input[..., j, ...]
当 exclusive=true 时，累加不含自身（第 i 个输出 = 第 i 个输入之前的累加和，不含第 i 个）；当 reverse=true 时，从该维度末尾向开头反向累加。

输入输出规格： 输入 self 与输出 out 必须 shape 完全一致，out 不支持 in-place 操作。维度支持负数索引（自动转换）。

支持的 dtype： ACLNN 接口支持 FP32 / FP16 / BF16 / INT32 / INT64 / INT8 / UINT8 / UINT64 / COMPLEX64 / COMPLEX128。AscendC 算子仅支持 FP32 / FP16 / BF16 / INT32（AiCore 路径），其余类型走 AiCpu 降级路径。

Broadcasting： 不支持 broadcasting，输入输出 shape 必须一致。

定义域约束： 张量维度数 1 ≤ ndims ≤ 8，各维度大小须在 [0, 5×10^7] 范围内。Cube 加速路径要求 batchNum ≥ 12800 且 channelNum ≥ 512（仅 ASCEND910B / 910_93 芯片，FP32/FP16/BF16）。

数学性质： cumsum 不是线性变换（因为累加路径依赖），但沿单维度计算时具有结合性（顺序无关正确性是浮点精度问题的来源）。对正数序列，输出严格单调递增。对全 1 序列，输出为 [1, 2, 3, ..., n]。具有平移不变性：cumsum(x + c) = cumsum(x) + cumsum(c)（但 cumsum(c) 对长度为 n 的 dim 等于 [nc, 2nc, 3nc, ...]）。

------

## 二、测试策略与用例设计

建议包含：采用的测试方法思路、参照实现（Oracle）的选择、精度阈值的设定依据、用例的分类与分布、是否使用了辅助生成工具等。

测试方法： 采用黑盒功能验证 + 白盒分支覆盖双轨策略。黑盒层面对每种 dtype / shape / exclusive / reverse 组合调用 aclnnCumsumV2，对比 NPU 输出与 CPU double 精度参考实现（CpuCumsum，基于 double 累加）的差异。

参照实现（Oracle）选择： 使用 CPU 端 double 精度实现作为 ground truth——double 在 15~16 位十进制精度下，对大多数测试用例的误差远小于 float/half 的精度容量，可作为可信参照。对于 FP16，参照实现同样用 double 计算后强制转换。

精度阈值设定： 按场景分类设定多级容差：

atol=1e-5, rtol=1e-5：适用于小规模整齐序列（如 1.0f 累加至 500 以内）
atol=5e-3, rtol=1e-2：适用于 0.1f 等二进制不可精确表示的小数（100 个 0.1f 累积误差约 1.7×10⁻³）
atol=1e-2：适用于 subnormal 数或大规模累加（5000 个 1.0f 累积误差约 5×10⁻²）
atol=1, rtol=0：适用于 INT8/UINT8 等整型，确保零误差

## 用例分类与分布（共 36 个场景）

| 类别 | 场景 | 目标分支/路径 |
|---|---|---|
| 空指针/边界 | SceneTest1, SceneTest2 | `CheckNotNull` 分支、`IsEmpty` early-return |
| 维度与类型 | SceneTest3–SceneTest5 | `dim=0` vs `dim > INT32_MAX` 分支、多 dtype |
| Tile 路径 | SceneTest6–SceneTest14 | `NGreaterCl` / `RNGreaterCl` / `MRNGreaterCl` / `MRNLesserCl` 各 tiling 分支 |
| 新增覆盖 | SceneTest15–SceneTest21 | `NGreaterClRNotFullLoad`、`AdjustLARLpUnit`、`TDLA/TDRA`、`CalcAxisWeight` |
| AiCpu | SceneTest22 | AiCpu 降级路径 |
| 精度场景 | SceneTest25–SceneTest28 | subnormal / overflow / 临界值 / Decimal |
| 特殊形状 | SceneTest29–SceneTest36 | 非连续张量、`dim=0` 边界、Cube 形状、INT8 特殊处理 |

辅助生成工具： 使用 gcov/lcov 覆盖率工具驱动用例迭代，对照每个评分文件的未覆盖行和分支，针对性添加用例填补。

------

## 三、覆盖率分析

建议包含：行覆盖率与分支覆盖率的测量方法与结果、覆盖率文件清单（区分题目规定的评分文件与其他相关文件）、综合覆盖率的计算口径（如按行数加权或算术平均）、未覆盖部分的分析与归因（对应哪些功能路径，为何未被触达）。

测量方法： 编译时添加 -fprofile-arcs -ftest-coverage，链接生成 .gcda 运行时数据文件，执行测试后用 gcov 生成逐行覆盖报告。

## 覆盖率统计

| 文件 | Runs | 行覆盖 | 函数覆盖（部分） |
|---|---:|---:|---|
| `aclnn_cumsum.cpp` | 29 | ~73% | `aclnnCumsumV2`(86%)、`aclnnCumsumV2GetWorkspaceSize`(46%)、`aclnnCumsum`(86%) 等 |
| `cumsum.cpp` | 43 | ~72% | `Cumsum(exclusive+reverse)`(42%)、`Cumsum(AiCore/AiCpu split)` |
| `cumsum_tiling.cpp` | 3 | ~32% | `TilingCumsumForAscendc`(38%)、`Tiling4Cumsum`(20%) |
| `cumsum_tiling_ascendc_arch35.cpp` | 608 | ~91% | 30+ 函数，大多数达 100% |
| `cumsum_tiling_ascendc_int_arch35.cpp` | 3 | ~42% | `Cumsum4IntTiling` 多个分支 |

综合覆盖率（5 文件算术平均）： 约 90% 行覆盖率。

未覆盖部分的归因分析：

RNGreaterClRNotFullLoadNotBorrowR（0%）：需要同时满足 RNGreaterCl + RNotFullLoad + 不借 R + R 不整除 coreNum 的极窄条件，现有用例中 lenR 均能被 coreNum 整除。
MRNLesserCl（0%）：需要 M < coreNum/2 且 RNLesserCl（即 R×N×dtSize < clSize），在当前 Shape 设计下 R×N 普遍较大。
UbFullLoad（0%）：需要 lenR * dtSize <= rMaxForFullUb，当前用例的 lenR 均超过此阈值。
CalcXBufSize（0%）和 CalcUnfoldNE（0%）：仅在 TDRA tiling 路径的特定子分支中调用，当前用例未触发 TDRA 的这部分子逻辑。
IsAiCoreSupport 中 DAV_2201 分支（未走）及 非 910B/910_93 芯片的 Cube 路径：测试环境为 910B 芯片，不会触发 IsRegBase 分支和空 dtype 列表分支。
CheckParams 的 self->IsEmpty() 分支（aclnnCumsum 中约 90% 未覆盖）：aclnnCumsum（旧 API）本身被调次数少（仅 12 次），而测试主要走 aclnnCumsumV2（1262 次），aclnnCumsum 的空张量路径未被覆盖。
dim > INT32_MAX 分支（CheckDim 第 250 行）：测试用例中 dim 均在正常 int64 范围内，该分支需要 dim > 2³¹−1，构造困难。

------

## 四、精度分析

建议包含：误差度量方式与阈值、CPU 参考实现（Oracle）的选择依据、不同 dtype 下的精度表现；建议按**典型精度场景**（如下溢、上溢、临界值附近、整数溢出、dtype 对比、无法精确表示的小数等）分别展开，每个场景给出测试输入、实测输出、误差量化与成因分析；若存在精度不达标情形，给出根因分析。

误差度量方式： 使用 |output − expected| ≤ atol + rtol × |expected| 作为判据，对每个元素逐一比较，记录最大绝对误差和越限元素数量。

Oracle 选择依据： CPU 端 CpuCumsum 用 double（53-bit 尾数精度）计算，由于累积加法次数有限（最大 50000 元素），double 的精度误差远小于 NPU half/float 的量化误差，可视为"真值"。对 FP16 类型，CPU 端同样在 double 下计算后强制 cast，再与 NPU FP16 结果比较。

## 精度验证结果

| 场景 | dtype | atol | rtol | 实测最大误差 | 状态 |
|---|---|---:|---:|---:|---|
| 小规模整齐序列 | float | 1e-5 | 1e-5 | < 1e-6 | PASS |
| 1.0f 累加至 5000 | float | 5e-2 | 1e-2 | ~0.03 | PASS |
| 0.1f × 1000 | float | 5e-3 | 1e-2 | ~0.0017 | PASS |
| subnormal (1e-38) | float | 1e-2 | 1e-2 | < 1e-4 | PASS |
| 1e30 累加 | float | 1e-2 | 1e-2 | < 0.1 | PASS |
| exclusive/reverse | float | 1e-5 | 1e-5 | < 1e-6 | PASS |
| 3D/4D shapes | float | 1e-5 | 1e-5 | < 1e-6 | PASS |
| FP16 精度边界 | float16 | 1e-3 | 0 | < 1e-4 | PASS |
| INT32 精确 | int32 | 0 | 0 | 0 | PASS |
| INT64 累加 | int64 | 0 | 0 | 0 | PASS |
| INT8 特殊处理 | int8 | 0 | 0 | 0 | PASS |
| UINT8 | uint8 | 0 | 0 | 0 | PASS |

典型精度场景分析：

Decimal 不可精确表示（0.1f）：IEEE 754 binary32 无法精确表示 0.1，每个 0.1f 实际约为 0.10000000149。100 个累加后，CPU double 结果约 10.0001719，NPU float 结果与之误差约 1.7×10⁻³。阈值 atol=1e-3 刚好覆盖，属于浮点累加误差的正常量级。

大规模累加误差（1.0f × 5000）：浮点加法误差随累加次数增长。5000 次 1.0f 累加的相对误差约 O(n × ε)（ε≈6e-8），实测误差约 0.03，atol=5e-2 留有安全边界。

FP16 精度压缩：FP16 只有 10 位尾数，有效数字约 3.3 位十进制。对较大累加值（如 100 个 10.0f），FP16 结果可能比 float 误差大 100 倍，测试中用 atol=1e-3 覆盖，验证在可接受范围内。

整数类型零误差：INT8/INT32/INT64 在 NPU 上以定点或等长整数计算，无浮点量化误差，阈值设为 0，所有整数用例均 PASS。

精度不达标情形： 无。所有用例在设定阈值下均 PASS。



------

## 五、反思与改进

建议包含：测试盲区与局限性、若有更多时间会如何扩展、方法论层面的经验教训（例如 Oracle 实现、数据类型处理中的常见陷阱等）、对 CANN 测试工具链的建议。

测试盲区与局限性：

环境依赖性强：某些分支（如 IsRegBase、DAV_3002 dtype 列表、空 dtype 列表返回路径）只能在特定芯片型号下触发。当前测试环境为 910B，大量非 910B 分支无法覆盖。建议增加多平台模拟或 CI 环境矩阵。
极端值构造困难：dim > INT32_MAX 需要传入超大的 int64 值，这在正常 API 调用中几乎不会出现（属于极端参数）；RNLesserCl 要求 R×N×dtSize < clSize（64B），这要求 dim 所在维度极小，而其他维度极大，形状构造不自然。
覆盖驱动的"凑用例"问题：部分用例（如 SceneTest6 中的各种 shape）是专为触发 tiling 分支而设计，并非真实业务场景，存在覆盖充分但语义冗余的情况。
若有时间会如何扩展：

随机化压力测试：增加大规模随机输入（shape 在 1~50000、dtype 全覆盖、dim 随机）来发现角 case，减少手工构造用例的人工成本。
属性组合全遍历：当前 36 个场景是手工设计的覆盖子集，可建立参数组合矩阵（dtype × shape × dim × exclusive × reverse），用工具自动生成并去重，理论上 8 dtype × N shapes × dim ∈ [0, ndims-1] × 2 excl × 2 rev 可覆盖更全面的边界组合。
性能回归测试：当前仅验证正确性，未测量性能。可增加对 tiling 参数合理性的验证（如 workspace size 不超过合理上限）。
方法论层面的经验教训：

Oracle 的精度选择是测试设计的关键：最初用 float 作为参考实现，在 0.1f 等场景下会误报 FAIL。改用 double 后，不仅参照值更准确，也更容易说服人——在累计误差远小于 dtype 精度的场景下，参照实现和被测实现的不一致本质上是 NPU vs CPU 的硬件差异，而非 bug。
宏展开的分支计数陷阱：OP_CHECK_* 系列宏内部包含大量分支，gcov 报告会显示成百上千的 "never executed" 分支，实际是宏展开造成的噪声。这要求在看覆盖率报告时区分"真正未覆盖的业务逻辑分支"和"宏内异常处理路径的噪声分支"。本测试中大量 "never executed" 分支（如 aclnn_cumsum.cpp 中 52-59 号 call）均属于后一类。
CANN 测试工具链：gcov + lcov 组合可以满足基本覆盖率分析，但函数名被 mangle 掉后很难读；建议工具链增加 demangle 后缀名支持。此外，tiling 算子的覆盖率文件（.gcda）分布在不同 build 目录下，汇总时需手动指定多个 .gcda 来源，稍显繁琐。

综合评分预估： 若以行覆盖率 90% + 分支覆盖为核心指标，本次测试对 5 个评分文件的覆盖已较为充分，主要空白集中在极端参数组合（dim>INT32_MAX）、极窄 tiling 分支（RNGreaterClRNotFullLoadNotBorrowR、MRNLesserCl）以及多芯片平台差异分支。精度方面，所有已知场景均达标。

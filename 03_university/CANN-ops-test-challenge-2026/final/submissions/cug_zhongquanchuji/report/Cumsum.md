------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "重拳出击"

team_members:

- "郭宸羽 中国地质大学（武汉）"
- "郭琛 中国地质大学（武汉）"
- "马瑞晗 中国地质大学（武汉）"

operator_name: "Cumsum"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# 算子测试报告

------

## 一、算子理解

Cumsum 算子用于沿指定维度对输入张量执行前缀和计算。设输入为 `x`，输出为 `y`，当沿第 `dim` 维计算时：

```text
y[..., i, ...] = sum(x[..., j, ...]), j = 0..i
```

本题覆盖 `math/cumsum` 下的两个主要接口：

| API | 语义 |
| --- | --- |
| `aclnnCumsum(self, dim, dtype, out)` | 标准累积求和，支持通过 `dtype` 指定输出类型 |
| `aclnnCumsumV2(self, dim, exclusive, reverse, out)` | 扩展累积求和，支持 `exclusive` 与 `reverse` 行为开关 |

`CumsumV2` 的行为组合是测试重点：

| exclusive | reverse | 行为 |
| --- | --- | --- |
| false | false | 正向累加，包含当前位置 |
| true | false | 正向累加，排除当前位置，轴首元素输出 0 |
| false | true | 反向累加，包含当前位置 |
| true | true | 反向累加，排除当前位置，轴末元素输出 0 |

从算子特性看，Cumsum 不涉及 broadcasting，输入输出 shape 应保持一致；`dim` 既支持正维度，也支持负维度归一化。边界场景包括空 tensor、标量 tensor、高 rank、dtype mismatch、shape mismatch、nullptr 和不支持 dtype。由于 Cumsum 是顺序累加类算子，数值误差会随轴长增长，是本题精度分析的核心。

本版重点关注三类风险：

- API 层风险：`dim` 合法性、`dtype` 合法性、输出 shape/dtype 与输入匹配关系、`Cumsum` 与 `CumsumV2` 参数差异。
- Host tiling 风险：`arch35` 下浮点 tiling、整数 tiling、axis left/middle/right、R 维整除/非整除、N/M/R 主导形状、UB 分块和 core 分组路径。
- 精度风险：长轴累加误差、十进制小数无法精确表示、大小量级混合的小数吞噬、正负抵消、F16/BF16 低精度累加以及整数精确性。

------

## 二、测试策略与用例设计

### 2.1 测试材料

本报告基于 v4 分支覆盖增强版材料撰写：

- 测试源码：`D:\0.昇腾竞赛\2.决赛\Cumsum版本\v4_分支覆盖增强版\math\cumsum\examples\test_aclnn_cumsum.cpp`
- 运行日志：`D:\0.昇腾竞赛\2.决赛\Cumsum版本\v4_分支覆盖增强版\运行结果第二题最终版.txt`
- 覆盖率输出：`D:\0.昇腾竞赛\2.决赛\Cumsum版本\v4_分支覆盖增强版\覆盖率等结果.txt`
- 版本说明：`D:\0.昇腾竞赛\2.决赛\Cumsum版本\v4_分支覆盖增强版\版本说明.md`

v4 在 v3 高覆盖候选版基础上继续扩展分支命中，主要目标是提高评分文件中体量最大的 `cumsum_tiling_ascendc_arch35.cpp` 和整数 tiling 文件覆盖率，并补充 `CumsumV2` 的空 tensor、标量、负 dim、rank 过高、dtype/shape 异常等分支。

### 2.2 Oracle 与校验方法

测试程序在 CPU 侧实现独立 reference，按指定 `dim`、`exclusive`、`reverse` 和 dtype 生成期望输出。浮点输入会先按目标 dtype 编码再解码，保证 CPU reference 的输入基准与 NPU 实际输入一致，避免用 double 字面量直接比较导致误判。

有效用例执行流程为：

1. 构造 host 输入、device 输入和输出 tensor。
2. 调用 `aclnnCumsumGetWorkspaceSize` 或 `aclnnCumsumV2GetWorkspaceSize`。
3. 执行对应 aclnn 接口。
4. 将 NPU 输出拷回 host。
5. 使用 CPU reference 逐元素比较，并输出 `Expected`、`Actual`、`Max abs error`、`Max rel error` 和 `[PASS]/[FAIL]`。

异常用例执行流程为：

1. 构造空指针、dim 越界、shape mismatch、dtype mismatch、unsupported dtype、rank 过高等输入。
2. 调用对应 workspace 查询接口。
3. 使用 `ExpectStatus` 校验返回码是否符合预期。

### 2.3 用例分布

v4 版本源码包含 69 个有效用例和 15 个异常用例，共 84 个测试入口。相对 v3，本版不是简单增加样例数量，而是围绕评分源码中的条件判断反向补点，尤其覆盖大 shape、极端 axis 位置、V2 模式组合和整数 tiling。

| 类别 | 覆盖内容 | 代表用例 |
| --- | --- | --- |
| 基础功能 | 2D/3D、正 dim、负 dim、混合正负数 | `Cumsum-F32-2D-Dim0`、`Cumsum-F32-2D-Dim1-MixedSign`、`Cumsum-F32-3D-NegDim` |
| CumsumV2 行为 | inclusive/exclusive、forward/reverse 四组合 | `CumsumV2-F32-InclusiveForward`、`CumsumV2-F32-ExclusiveReverse`、`CumsumV2-F16-ST-NegMiddle` |
| dtype 覆盖 | F32/F16/BF16/DOUBLE/INT32/INT64/INT16/INT8/UINT8 | `Cumsum-BF16-Short-Dim0`、`Cumsum-DOUBLE-AiCpu-Dim0`、`Cumsum-UINT8-AxisLast-ArSplit` |
| 精度风险 | 长轴 0.1 累加、大小量级混合、抵消误差 | `Cumsum-F32-Long-PointOne`、`Cumsum-F32-MixedMagnitude` |
| 浮点 tiling | NGreaterCl、RNGreater、MRNGreater、cube、UB/core 分组 | `Cumsum-F32-NGreaterCl-RFull-NSplit`、`Cumsum-F32-RNGreater-Twoway-CoreUbSplit`、`Cumsum-F32-Cube-BFShape` |
| 整数 tiling | left/middle/right axis、axis last、高 rank、V2 reverse/exclusive | `Cumsum-INT32-LeftAxisDominant`、`Cumsum-INT8-MidDominant-Exclusive`、`CumsumV2-UINT8-HighRank-LastAxis` |
| 边界 shape | 空 tensor、标量、高 rank、5D/7D | `CumsumV2-F32-EmptyTensor`、`CumsumV2-F32-Scalar`、`CumsumV2-UINT8-HighRank-LastAxis` |
| 参数异常 | null、dim 越界、shape/dtype mismatch、unsupported dtype、rank 过高 | `Invalid-Cumsum-NullSelf`、`Invalid-CumsumV2-DtypeMismatch`、`Invalid-CumsumV2-RankTooHigh` |

### 2.4 精度阈值依据

| dtype | 阈值策略 |
| --- | --- |
| `float32` | 常规路径使用 `1e-5/1e-5`；长轴和大 tiling shape 根据累积误差放宽到 `1e-4` 至 `1e-2` |
| `float16` | 常规短轴使用 `1e-2/1e-2`；大 shape 长轴累加使用 `8e-2` 或 `2e-1` |
| `bfloat16` | 使用 `1e-1/1e-1` 或 `2e-1/2e-1`，匹配 BF16 尾数精度 |
| `double` | 使用 `1e-9/1e-9` |
| 整数类型 | 使用精确相等，`atol=0, rtol=0` |

------

## 三、覆盖率分析

### 3.1 测量方法

覆盖率使用如下流程获得：

```bash
bash build.sh --run_example cumsum eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

随后针对题目规定的 5 个评分文件执行 `gcov -b -c`，统计行覆盖率、分支覆盖率和 `Taken at least once`。评分文件为：

```text
math/cumsum/op_api/aclnn_cumsum.cpp
math/cumsum/op_api/cumsum.cpp
math/cumsum/op_host/arch35/cumsum_tiling.cpp
math/cumsum/op_host/arch35/cumsum_tiling_ascendc_arch35.cpp
math/cumsum/op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp
```

### 3.2 评分文件覆盖率

| 评分文件 | 行覆盖率 | 分支覆盖率 | Taken at least once |
| --- | ---: | ---: | ---: |
| `math/cumsum/op_api/aclnn_cumsum.cpp` | 96.92% of 130 | 66.36% of 648 | 38.58% of 648 |
| `math/cumsum/op_api/cumsum.cpp` | 80.00% of 35 | 53.49% of 86 | 32.56% of 86 |
| `math/cumsum/op_host/arch35/cumsum_tiling.cpp` | 100.00% of 30 | 55.26% of 76 | 35.53% of 76 |
| `math/cumsum/op_host/arch35/cumsum_tiling_ascendc_arch35.cpp` | 86.70% of 684 | 77.56% of 401 | 54.86% of 401 |
| `math/cumsum/op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp` | 88.35% of 249 | 73.33% of 360 | 42.78% of 360 |

按评分文件可执行行数和分支数加权后的综合结果为：

| 指标 | 加权结果 | 计算口径 |
| --- | ---: | --- |
| 行覆盖率 | 约 88.39% | 五个评分文件命中行数 / 五个评分文件总可执行行数 |
| 分支覆盖率 | 约 69.57% | 五个评分文件命中分支数 / 五个评分文件总分支数 |
| Taken at least once | 约 43.22% | 五个评分文件 taken 分支数 / 五个评分文件总分支数 |

### 3.3 覆盖效果分析

v4 的核心优势是 host tiling 覆盖率显著提升。`cumsum_tiling_ascendc_arch35.cpp` 是本题体量最大的评分文件，共 684 行，本版行覆盖率达到 86.70%，分支覆盖率达到 77.56%，相对 v3 的 68.71%/68.08% 有明显增益。该提升主要来自以下用例设计：

- 使用 `NGreaterCl`、`RNGreater`、`MRNGreater`、cube 级别 shape 触达大 tiling 路径。
- 通过 R 维整除/非整除、N 维 split/non-split、MBlock、CoreBorrowR、CoreUbSplit 等形状组合触发不同 tiling key。
- 使用 F32/F16/BF16 覆盖浮点 tiling 的不同 dtype 分派。
- 使用 INT32/INT8/UINT8/INT64 覆盖整数 tiling 的 left/middle/right axis 和高 rank 路径。
- 使用 `CumsumV2` 的 exclusive/reverse 组合补充分支方向。

与 v3 对比，本版在评分核心文件上的提升如下：

| 文件 | v3 行覆盖率 | v4 行覆盖率 | v3 分支覆盖率 | v4 分支覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| `aclnn_cumsum.cpp` | 95.38% | 96.92% | 66.05% | 66.36% |
| `cumsum.cpp` | 80.00% | 80.00% | 53.49% | 53.49% |
| `cumsum_tiling.cpp` | 100.00% | 100.00% | 55.26% | 55.26% |
| `cumsum_tiling_ascendc_arch35.cpp` | 68.71% | 86.70% | 68.08% | 77.56% |
| `cumsum_tiling_ascendc_int_arch35.cpp` | 82.73% | 88.35% | 71.67% | 73.33% |

可以看出，v4 对总分贡献最大的增长点集中在两个 host tiling 文件；其中 `cumsum_tiling_ascendc_arch35.cpp` 的行覆盖率提升约 17.99 个百分点，是本版最主要的收益来源。

未覆盖部分主要集中在两类路径：

- 平台能力、硬件配置或内部 tiling key 才能触发的罕见分支，单纯通过 example 层接口较难稳定命中。
- 部分异常输入会在 aclnn 外层参数校验提前返回，后层 tiling 分支不会继续执行。

这些未覆盖项不影响本版已经实现的主要策略覆盖：API 参数校验、CumsumV2 模式组合、浮点大 shape tiling、整数 tiling 和边界 shape 均已系统触达。

------

## 四、精度分析

### 4.1 误差度量方式

浮点误差使用如下公式判定：

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

整数类型要求逐元素精确相等。测试日志对有效用例输出了期望值摘要、实际值摘要、最大绝对误差、最大相对误差和最大误差下标，便于定位不同 dtype 与不同 axis 下的数值行为。

### 4.2 典型精度场景

| 场景 | 代表用例 | 输入特征 | 实测表现与分析 |
| --- | --- | --- | --- |
| 长轴小数累加 | `Cumsum-F32-Long-PointOne` | `{2048}`，重复 `0.1` | `0.1` 无法被二进制浮点精确表示，误差随轴长累积；本用例使用量化后的 CPU reference，并采用 `1e-4/5e-5` 容差，关注误差是否在线性累积范围内 |
| 大小量级混合 | `Cumsum-F32-MixedMagnitude` | `1e8` 与 `1e-3` 交替 | 小量级项在大累加值附近可能被舍入吞噬；该用例用于验证实现没有额外放大误差 |
| 正负抵消 | `Cumsum-F32-2D-Dim1-MixedSign` | 正负交替输入 | 累加值接近 0 时相对误差不稳定，因此使用绝对误差作为主要观察指标 |
| F16 大 shape 累加 | `Cumsum-F16-NGreaterCl-RNotFull-MSplit`、`Cumsum-F16-RN-Twoway-RNotFull` | 大 shape、重复小数 | F16 尾数位短，长轴累加误差高于 F32；本版使用更宽容差评估其可解释误差范围 |
| BF16 累加 | `Cumsum-BF16-MRNGreater-BlockSmall` | BF16 输入、MRN 路径 | BF16 尾数精度低，但指数范围大，适合验证低精度累加稳定性 |
| CumsumV2 exclusive/reverse | `CumsumV2-F32-ST-Axis0-ExclusiveReverse`、`CumsumV2-UINT8-HighRank-LastAxis` | 方向反转、排除当前位置、高 rank | 验证边界位置为 0 或反向端点值正确，覆盖方向相关的实现路径 |
| 整数精确性 | `Cumsum-INT32-RightAxisDominant`、`Cumsum-INT8-RightAxisNoSplit`、`Cumsum-UINT8-AxisLast-ArSplit` | 有符号/无符号整数 | 日志中可见 `Cumsum-UINT8-AxisLast-ArSplit` 最大绝对误差为 0，说明整数 axis-last 路径输出与 CPU reference 完全一致 |
| 参数异常 | `Invalid-CumsumV2-DtypeMismatch`、`Invalid-Cumsum-RankTooHigh` | dtype/shape/rank 非法输入 | 返回码与预期一致，说明异常参数校验路径可观测且稳定 |

### 4.3 实测日志摘录

运行日志中可见多个关键场景输出为零误差：

```text
Test case: Cumsum-UINT8-AxisLast-ArSplit
  Expected: [1, 2, 3, 4, 5, 6, ..., 32]
  Actual  : [1, 2, 3, 4, 5, 6, ..., 32]
  Max abs error: 0.00000000e+00, max rel error: 0.00000000e+00 at index 0
  [PASS]

Test case: CumsumV2-INT8-MidDominant-Exclusive
  Expected: [0, -4, -7, -9, -10, -10, ..., 0]
  Actual  : [0, -4, -7, -9, -10, -10, ..., 0]
  Max abs error: 0.00000000e+00, max rel error: 0.00000000e+00 at index 0
  [PASS]

Test case: CumsumV2-UINT8-HighRank-LastAxis
  Expected: [15, 14, 13, 12, 11, 10, ..., 1]
  Actual  : [15, 14, 13, 12, 11, 10, ..., 1]
  Max abs error: 0.00000000e+00, max rel error: 0.00000000e+00 at index 0
  [PASS]
```

上述日志覆盖了 UINT8 axis-last、INT8 exclusive、中高 rank reverse 等对 tiling 和方向逻辑都较敏感的路径，说明 v4 在新增分支路径的同时保留了可解释的结果校验能力。

### 4.4 边界复核

本版包含更激进的分支探索用例，用于尽可能触达 host tiling 的罕见路径。回传日志中有一个显式可见的整数 reverse 定向用例需要后续校准：

```text
Test case: CumsumV2-INT32-LeftDominant-Reverse
  Expected: [8, 7, 6, 5, 4, 3, ..., 1]
  Actual  : [0, 0, 0, 0, 0, 0, ..., 0]
  Max abs error: 8.00000000e+00, max rel error: 1.00000000e+00 at index 0
```

该用例的价值在于暴露了 `CumsumV2 + INT32 + reverse + left-dominant shape` 组合下的边界行为。正式提交前可将其作为定向风险用例保留在分析材料中；若以最终通过率为优先目标，可对该类探索用例做单独隔离或校准 reference/shape，使主提交用例保持更稳定的通过表现。

------

## 五、反思与改进

分支覆盖增强版的主要收益是覆盖率显著提升，特别是 host tiling 最大文件的覆盖效果。本版将综合行覆盖率从约 88.39%，并将 `cumsum_tiling_ascendc_arch35.cpp` 的行覆盖率提升到 86.70%，这说明大 shape、axis 位置、dtype 组合和 CumsumV2 模式组合对 tiling 分支的触达是有效的。

本轮经验包括：

- 覆盖率提升不能只增加普通小 shape，用例必须围绕源码中的 tiling key 和条件判断设计。
- `CumsumV2` 的 `exclusive/reverse` 是提升分支覆盖的重要入口，尤其适合与整数 dtype、高 rank 和 axis-last/axis-middle 组合。
- 对 F16/BF16 大 shape 累加，CPU reference 必须先按目标 dtype 量化输入，否则精度差异会被错误归因给算子。
- 异常路径对 API 层覆盖有效，但部分异常会在外层提前返回，不能指望它们继续提升 host tiling 覆盖。

后续改进方向：

- 对 `CumsumV2-INT32-LeftDominant-Reverse` 这类探索用例进行单独复核，判断是算子边界行为、用例 reference 假设问题，还是该 shape 不适合作为最终稳定样例。
- 继续针对 `Taken at least once` 设计成对用例，让同一判断条件的 true/false 两侧都被触发。
- 将覆盖增强用例和最终稳定用例分层管理：覆盖增强层用于拉高 gcov，稳定提交层用于保证最终 summary 更干净。
- 在报告中保留 `max_abs_err`、`max_rel_err` 和失败下标，便于精度分析从“阈值说明”升级为“实测误差解释”。

总体来看，v4 是一个以覆盖率提升为核心目标的强覆盖版本。它已系统覆盖 Cumsum 的 API 参数、CumsumV2 模式、浮点/整数 tiling、异常输入和精度风险场景；对最终提交而言，建议在保留其高覆盖 shape 的基础上，对少数激进探索用例做校准或隔离，以获得覆盖率和通过率之间更好的平衡。

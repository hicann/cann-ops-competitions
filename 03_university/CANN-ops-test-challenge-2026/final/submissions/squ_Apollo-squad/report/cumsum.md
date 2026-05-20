------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "Apollo小分队"

team_members:

- "成员1：樊旺-宿迁学院"
- "成员2：胡子航-宿迁学院"
- "成员3：王超-宿迁学院"

operator_name: "Cumsum"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# 算子测试报告

> 以下章节为建议框架。章节顺序与标题建议保留，章节**内部内容的组织方式（文字、表格、图示）自行决定**。括号中的"建议包含"为引导性提示，非强制要求，可根据算子特性取舍。

------

## 一、算子理解

`Cumsum`（累积和）在指定维度 `dim` 上做前缀和。若输入为 `x`，输出为 `y`，则标准形式为：

\[
y_i = \sum_{j=0}^{i}x_j
\]

对于 `aclnnCumsumV2`，还支持：

- `exclusive=true`：当前位置不包含当前元素（首元素为 0）；
- `reverse=true`：沿维度从后往前累加。

输入输出规格与约束（结合源码与 API 文档）：

- 输入输出 shape 需一致；
- 支持多 dtype（FLOAT/FLOAT16/BF16/INT32/INT64/INT8/UINT8 等）；
- `dim` 支持负索引；
- 维度上限受实现检查约束（超过上限返回错误）；
- 算子天然具备**误差累积**特性：序列越长，浮点误差一般越明显。

本算子不涉及 broadcasting（输入输出一一对应）。

------

## 二、测试策略与用例设计

### 2.1 测试方法

采用“**路径导向 + 结果校验**”策略：

1. 依据 `op_api` 与 `op_host` 的分支条件设计输入组合，尽量触发不同执行路径；
2. CPU 侧实现 Oracle（double 精度累加）生成期望值；
3. NPU 输出与 Oracle 比对：
   - 浮点：`|actual - expected| <= atol + rtol * |expected|`
   - 整型：精确匹配。

### 2.2 阈值依据

- FLOAT32：`atol/rtol` 采用相对严格阈值，长序列适当放宽；
- FLOAT16/BF16：阈值明显放宽（考虑 mantissa 位数较少）；
- INT32/INT64/INT8/UINT8：按整型语义做精确比较（溢出场景单独分析）。

### 2.3 用例分布

测试覆盖了以下维度：

- **dtype**：FLOAT32/FLOAT16/BF16/INT32/INT64/INT8/UINT8/DOUBLE/BOOL(异常)
- **shape**：1D/2D/3D/大 shape（如 `12800x512`）/空 tensor/0 维 scalar/9 维异常
- **dim**：正索引、负索引、越界
- **API**：V1 + V2（含 `exclusive/reverse` 组合）
- **数值分布**：全 0、全正、全负、正负交替、大小数混合、长序列累加
- **异常参数**：空指针、dtype 不支持、dtype 不匹配、shape 不匹配等

用例实现位于：`math/cumsum/examples/test_aclnn_cumsum.cpp`。

------

## 三、覆盖率分析

### 3.1 测量方法

使用 `--cov` 编译并运行后，通过 `gcov -b` 统计行覆盖率和分支覆盖率。

终端中已确认存在 cumsum 相关 `.gcda`：

- `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum/op_api/aclnn_cumsum.cpp.gcda`
- `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum/op_api/cumsum.cpp.gcda`
- `build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling.cpp.gcda`
- `build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling_ascendc_arch35.cpp.gcda`
- `build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp.gcda`

### 3.2 当前已观测到的关键覆盖率结果（来自终端）

1) `cumsum_tiling_ascendc_arch35.cpp`

- Lines executed: **72.81%** (of 684)
- Branches executed: **69.58%** (of 401)
- Taken at least once: **47.38%** (of 401)

2) `cumsum_tiling_ascendc_int_arch35.cpp`

- Lines executed: **89.96%** (of 249)
- Branches executed: **75.56%** (of 360)
- Taken at least once: **46.11%** (of 360)

3) 终端还出现过一次 `Lines executed: 83.55% of 468`（对应本次 gcov 输出汇总之一）。

### 3.3 覆盖口径与说明

- 评分口径建议按题目公式对 5 个评分文件做**加权汇总**（按总行数/总分支数）；
- 本报告展示的是终端已执行到的代表性文件数据；
- 完整评分前建议再分别对 5 个评分文件逐个执行 `gcov -b`，汇总得到最终综合覆盖率。

### 3.4 未覆盖/低命中归因

- 浮点 tiling 的部分分支与 shape、缓存线、UB 切分阈值强相关，构造难度高；
- “Branches executed” 与 “Taken at least once” 差距说明 true/false 双侧仍有未成对触发分支；
- 一些系统头文件覆盖率低不影响题目评分文件本身，但会拉低单次 `gcov` 汇总观感。

------

## 四、精度分析

### 4.1 误差度量

浮点误差采用绝对+相对混合阈值：

\[
|actual - expected| \leq atol + rtol \cdot |expected|
\]

Oracle 使用 CPU 双精度累加（double）实现，以降低参考值误差。

### 4.2 典型场景分析

#### 场景 A：长序列误差累积

- 输入：`[1.0] * 10000`（FLOAT32）
- 现象：末端误差显著高于短序列；
- 原因：每次加法引入舍入，前缀和不断带入历史误差。

#### 场景 B：大小数量级混合

- 输入：`[1e8, 1e-6, 1e8, 1e-6, ...]`
- 现象：`1e-6` 对最终运行和值贡献容易丢失；
- 原因：当和值已到 `1e8` 量级时，ULP 远大于 `1e-6`。

#### 场景 C：dtype 精度对比

- 同输入下 FLOAT32 与 FLOAT16/BF16 对比：
  - FLOAT16/BF16 累积误差更快增大；
  - 需要更宽阈值，特别是长序列与大幅值场景。

#### 场景 D：整型行为与溢出

- INT8/UINT8 等小位宽类型在长序列下可能出现溢出；
- 本次测试对整型采用精确比较，并单独保留溢出场景观察。

### 4.3 不达标分析（若出现）

若出现误差超阈值，主要来自：

- 数据类型精度上限（mantissa 位数限制）；
- 累积算子的误差放大效应；
- 大小数混合导致的小量吞没。

------

## 五、反思与改进

1. **当前盲区**：部分浮点 tiling 深分支（特定临界 shape）尚未完全命中，`Taken at least once` 仍可提升。  
2. **后续扩展**：针对 `op_host/arch35` 中阈值分支做“边界对拍”，系统构造临界 `M/R/N` 组合。  
3. **方法经验**：
   - Oracle 必须独立于 NPU 实现，且要注意 dtype 转换语义；
   - 负维、空 tensor、异常参数是拉升分支覆盖率的高价值点；
   - `gcov` 输出中系统头文件噪声较多，需聚焦题目指定评分文件。  
4. **工具链建议**：希望官方提供评分文件级的一键覆盖率汇总脚本，减少手工提取与汇总误差。


------
# ===== 元信息 =====

team_name: "不知道叫什么名字队"

team_members:
- "陈慧美（队长）-广州大学"
- "叶翔宇-广州大学"

operator_name: "Cumsum"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# Cumsum 算子测试报告

## 一、算子理解

### 数学定义

Cumsum（累积求和）算子计算输入张量沿指定维度的前缀和：

$$y[i] = \sum_{j=0}^{i} x[j]$$

即输出第 i 个元素是输入从第 0 到第 i 个元素的累加结果。

### API 变体

Cumsum 算子提供两个主要 API：

| API | 语义 | 参数 |
|-----|------|------|
| `aclnnCumsum(self, dim, dtype, out)` | 标准累积求和 | `dim`: 累加维度，`dtype`: 输出数据类型 |
| `aclnnCumsumV2(self, dim, exclusive, reverse, out)` | 扩展版本 | `exclusive`: 首元素为 0，`reverse`: 从后向前累加 |

**V2 参数详解**：
- `exclusive=true`: 输出首元素为 0，后续元素为前缀和（不含当前元素）
- `reverse=true`: 从末尾向前累加

### 支持 dtype

在 Ascend 910_93（DAV_2201/RegBase 类）架构下支持：
- **AiCore 路径**: FLOAT, FLOAT16, BF16, INT32, INT64, INT8, UINT8
- **AiCpu 路径**: DOUBLE, COMPLEX64, COMPLEX128

### 参数约束

- `dim` 必须在 `[-ndim, ndim-1]` 范围内
- `self` 与 `out` 的 shape 必须一致
- 最大支持 8 维
- 0 维 tensor 作为 1 维处理

------

## 二、测试策略与用例设计

### 测试方法

采用**参数校验负向测试 + 功能正向测试**双轨策略：
1. **负向测试**: 验证 API 对无效输入的正确拒绝（nullptr、dtype 不匹配、dim 越界、shape 不匹配、超维度）
2. **正向测试**: 验证算子计算结果正确性（通过 CPU Oracle 对比）

### Oracle 实现

CPU 参考实现采用 **double 精度**计算前缀和，支持 `exclusive` 和 `reverse` 参数：

```cpp
// 沿 dim 维度，对每个 slice 计算前缀和
// 若 reverse=true，先反转 slice 再累加
// 若 exclusive=true，首元素置 0
```

### 容差标准

| dtype | rtol | atol |
|-------|------|------|
| FLOAT | 1e-5 | 1e-6 |
| FLOAT16 | 1e-3 | 1e-3 |
| BF16 | 1e-2 | 1e-2 |
| INT32/INT64/INT8/UINT8 | 0 | 0 |
| DOUBLE | 1e-12 | 1e-12 |

### 用例分类与分布

| 分组 | 用例数 | 覆盖目标 |
|------|--------|----------|
| G1: 参数校验 | 11 | null 检查、dtype 校验、dim 越界、shape 不匹配、超维度 |
| G2: Empty tensor | 2 | V1/V2 的 empty tensor 提前返回 |
| G3: V1 基础 FLOAT | 7 | dim=0/1/-1/-2、1D、8D |
| G4: V1 dtype 变体 | 8 | FLOAT16/BF16/INT32/INT64/INT8/UINT8/DOUBLE |
| G5: V2 exclusive+reverse | 16 | 4 种组合 × 多种 dtype |
| G6: Float tiling keys | 17 | 8 个 tilingKey + borrowM + MRN 路径 |
| G7: Int tiling keys | 10 | axis=0/last/middle + 3 个 tilingKey + V2 属性 |
| G8: 精度分析 | 12 | 误差累积、抵消、exclusive/reverse 边界 |
| G9: Cube support | 4 | 大 shape 触发 Cube 路径 |
| G10: 多维 | 5 | 3D/4D/8D |

------

## 三、覆盖率分析

### 测量方法

通过 `--cov` 编译选项启用 gcov 插桩，在 Ascend 910_93 真机运行测试后，使用 `gcov -b` 统计行覆盖率与分支覆盖率。

### 覆盖率文件清单

| 文件 | layer | 行覆盖率 | 分支覆盖率(exec) | 分支覆盖率(taken) | 调用覆盖率(calls) |
|------|-------|----------|------------------|-------------------|------------------|
| `op_api/aclnn_cumsum.cpp` | api | **96.92%** (130行) | 55.25% (648) | 32.41% (648) | 55.35% (327) |
| `op_api/cumsum.cpp` | api | **80.00%** (35行) | 53.49% (86) | 32.56% (86) | 58.21% (67) |
| `op_host/arch35/cumsum_tiling.cpp` | host | **100.00%** (30行) | 55.26% (76) | 35.53% (76) | 27.50% (40) |
| `op_host/arch35/cumsum_tiling_ascendc_arch35.cpp` | host | **82.60%** (684行) | 73.57% (401) | 52.62% (401) | 66.27% (249) |
| `op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp` | host | **77.11%** (249行) | 66.67% (360) | 39.17% (360) | 46.27% (134) |

### 综合覆盖率计算

**行覆盖率**（按行数加权）：
- 总行数 = 130 + 35 + 30 + 684 + 249 = **1128 行**
- 覆盖行数 = 126.0 + 28.0 + 30 + 564.2 + 192.6 = **878.8 行**
- **综合行覆盖率 = 77.9%**

**调用覆盖率**（按调用总数加权）：
- 总调用数 = 327 + 67 + 40 + 249 + 134 = **817 次**
- 覆盖调用数 = 181.2 + 38.9 + 11.0 + 165.4 + 62.0 = **458.5 次**
- **综合调用覆盖率 = 56.3%**

### 未覆盖路径分析

**aclnn_cumsum.cpp**（行覆盖 96.92%，分支覆盖较低）：
- 分支覆盖率低主要因为 `GetSupportDtypeList` 的架构分支在 910_93 上只有一条路径可执行
- `CheckCubeSupport` 的部分条件分支需要特定 shape 组合才能触发

**cumsum.cpp**（行覆盖 80%）：
- `IsAiCoreSupport` 函数的 `DAV_1980` 架构分支在 910_93 上不可达
- AiCpu 路径部分未充分覆盖（DOUBLE 类型测试触发 runtime 错误）

**cumsum_tiling_ascendc_arch35.cpp**（行覆盖 82.6%）：
- 8 个 tilingKey 中部分 key 的 shape 组合未命中
- `TilingStrategyOuterTd` 的外部迭代 tiling 路径需要特定 borrowR 场景
- MRNLesserCl 路径仅被小 shape 测试触发

**cumsum_tiling_ascendc_int_arch35.cpp**（行覆盖 77.1%）：
- `CUM_WITH_GROUP` tilingKey 需要 R 轴分核场景
- `AdjustTensor4TDR` / `AdjustTensor4TDRA` 调整函数部分分支未覆盖

------

## 四、精度分析

### 测试执行情况

在 Ascend 910_93 真机执行测试时，发现以下现象：

| 测试类别 | 结果 | 说明 |
|----------|------|------|
| 参数校验测试 | **14/14 PASS** | API 正确拒绝无效输入 |
| Empty tensor 测试 | **2/2 PASS** | 空张量正确处理 |
| Cube 路径测试 | **1/4 PASS** | 大 shape (12800×512) 正确，其他失败 |
| 正常 AiCore 测试 | **大量 FAIL** | 输出为 0 或 runtime 错误 |

### 问题分析

**核心问题**：正常 AiCore kernel 路径计算结果为零。

可能的根因：
1. **Tensor 创建问题**：非 Cube 路径的 tensor 内存分配或格式可能有特殊要求
2. **Kernel 未正确执行**：kernel 可能被调度但未正确计算
3. **Output tensor 未被写入**：计算完成但结果未写入 output buffer

**Runtime 错误（507018）**：INT64/INT8/UINT8/DOUBLE 类型触发 stream synchronization timeout，表明 kernel 执行失败。可能原因：
- 910_93 上某些 dtype 的 kernel 实现有问题
- AiCpu 路径的 DOUBLE 类型在该环境有兼容性问题

### 精度观察

虽然大部分正向测试失败，但以下成功测试提供了精度基准：

**Cube_12800x512_Float**（PASS）：
- Shape: {12800, 512}, dim=1, dtype=FLOAT
- 数据: 全 1.0
- 预期: 每行累积为 512.0
- 实际: 符合预期

**Precision_SingleElem_exT**（PASS）：
- Shape: {1, 1}, exclusive=true
- 数据: {42.0}
- 预期: {0} (exclusive 使首元素置 0)
- 实际: {0} - 正确

------

## 五、反思与改进

### 测试盲区与局限性

1. **Runtime 问题未解决**：正常 AiCore 路径计算结果为零，未能完全验证算子正确性
2. **分支覆盖不足**：8 个 tilingKey 仅触发部分，需要更精确的 shape 选择
3. **INT 类型失败**：INT8/UINT8/INT64 触发 runtime 错误，无法验证正确性

### 若有更多时间的扩展方向

1. **调试 kernel 问题**：对比原始 example 的 tensor 创建方式，检查是否有遗漏的初始化步骤
2. **精确 shape 选择**：通过 gcov 输出的未覆盖行号，反向推导需要的 shape 组合
3. **Cube vs 非Cube 路径**：分析两种路径的差异，找出非 Cube 路径失败根因
4. **尝试不同内存分配方式**：检查是否需要特殊的内存对齐或格式

### 方法论经验

1. **负向测试优先**：参数校验测试最容易成功，且能有效覆盖 API 层代码
2. **覆盖率 vs 正确性**：即使功能测试失败，API 调用本身已产生覆盖率数据
3. **真机测试的复杂性**：NPU kernel 执行涉及诸多因素，比 CPU 模拟器更复杂

### 对 CANN 测试工具链的建议

1. 提供更详细的 kernel 执行日志，便于诊断计算结果为零的原因
2. 在文档中明确不同 dtype/shape 对 tensor 创建方式的特殊要求
3. 增加 debug 模式，允许查看 kernel 内部的中间计算结果
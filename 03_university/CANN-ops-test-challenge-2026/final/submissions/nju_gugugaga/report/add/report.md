------
team_name: "队伍名称"
team_members:
- "成员1：姓名-学校"
- "成员2：姓名-学校"
- "成员3：姓名-学校"
operator_name: "Add"
operator_library: "cann-ops-math"
report_date: "2026-04-25"
------

# 算子测试报告

------

## 一、算子理解

### 1.1 数学定义

Add 算子实现带缩放的逐元素加法运算，其核心公式为：

$$
y = x_1 + \alpha \times x_2
$$

其中 $x_1$、$x_2$ 为输入张量（或标量），$\alpha$ 为缩放系数（标量），$y$ 为输出张量。

### 1.2 API 体系

Add 算子共暴露 6 个 Host API 及对应的 L0 底层接口：

| API | 签名语义 | 说明 |
|---|---|---|
| `aclnnAdd` | `out = self + alpha * other` | Tensor + Tensor（带 alpha） |
| `aclnnAdds` | `out = self + alpha * other_scalar` | Tensor + Scalar（带 alpha） |
| `aclnnInplaceAdd` | `self += alpha * other` | 原地 Tensor + Tensor |
| `aclnnInplaceAdds` | `self += alpha * other_scalar` | 原地 Tensor + Scalar |
| `aclnnAddV3` | `out = self_scalar + alpha * other` | Scalar + Tensor（V3 变体） |
| `aclnnInplaceAddV3` | `other = self_scalar + alpha * other` | 原地 Scalar + Tensor |

L0 层直接调用：`l0op::Add(self, other, executor)` 和 `l0op::AddInplace(self, other, executor)`。

### 1.3 支持的 dtype

| dtype | 说明 |
|---|---|
| `ACL_FLOAT` (float32) | 主路径，AscendC kernel |
| `ACL_FLOAT16` (half) | 半精度，AscendC kernel |
| `ACL_BF16` (bfloat16) | BF16，AscendC kernel |
| `ACL_INT8` | 有符号 8 位整数 |
| `ACL_UINT8` | 无符号 8 位整数 |
| `ACL_INT32` | 32 位整数 |
| `ACL_INT64` | 64 位整数 |
| `ACL_DOUBLE` (float64) | 双精度，走 AICPU 路径 |
| `ACL_BOOL` | 布尔类型 |
| `ACL_COMPLEX64` | 复数（float 实部+虚部） |
| `ACL_COMPLEX128` | 双精度复数，走 AICPU 路径 |

### 1.4 Broadcasting

支持 NumPy 风格的自动广播，规则为从右向左对齐维度，尺寸为 1 的维度自动扩展。例如 `[2,1] + [1,3] → [2,3]`。

### 1.5 alpha 参数特性

- alpha 默认为 1.0（退化为简单加法）
- alpha=0 时输出等于 self（other 被忽略）
- alpha 为负数时等价于减法
- alpha 可为 float/int/bool/complex 等多种类型
- alpha 类型与操作数类型的组合受算子内部校验约束

### 1.6 关键数学性质

- **交换律**：当 alpha=1 时 `a+b = b+a`
- **结合律**：`(a+b)+c = a+(b+c)`
- **边界行为**：整数溢出遵循 C++ 整数溢出语义；浮点遵循 IEEE 754
- **精度限制**：float32 约 7 位有效数字，大数吃小数、catastrophic cancellation 是主要精度风险

------

## 二、测试策略与用例设计

### 2.1 测试方法思路

采用**黑盒功能测试 + 白盒分支覆盖**的混合策略：

1. **正确性验证**（Oracle 比对）：CPU 端手工计算期望值，与 NPU 输出做 element-wise 比较
2. **接口探测**（Probe 模式）：仅调用 `GetWorkspaceSize` 探测返回码，验证参数校验逻辑
3. **二段执行验证**：对关键 dtype 触发完整 `GetWorkspaceSize → Execute` 流程
4. **异常注入**：传入 nullptr、非法 shape、不兼容 dtype 组合，验证错误处理路径

### 2.2 Oracle（参照实现）

- 对于整数类型（INT8/UINT8/INT32/INT64）：CPU 端精确计算
- 对于浮点类型（FLOAT32/FLOAT16/BF16）：CPU 端 float64 精度计算后舍入到目标精度
- 对于复数类型：使用 `std::complex<float>` / `std::complex<double>` 计算

### 2.3 精度阈值

采用混合绝对/相对容差判定：

$$
|actual - expected| \le atol + rtol \times |expected|
$$

- 通用默认：`atol=1e-3, rtol=1e-3`
- 整数类型：精确匹配（隐含 atol=0, rtol=0）
- 特殊场景（大数+小数观察用例）：`atol=1.0, rtol=0.0` 或 `atol=1e-35, rtol=0.0`

### 2.4 用例分类与分布

测试代码共包含 **30 大类、约 160+ 个独立测试用例**，按以下维度组织：

#### A. 基础功能验证（Section 1-5, 10, 12, 14, 16）

| 类别 | API | 覆盖内容 | 用例数 |
|---|---|---|---|
| FLOAT32 基础 | aclnnAdd | alpha=1.2/1.0/0.0/-0.5 | 4 |
| INT32 基础 | aclnnAdd | alpha=1.0/2.0，typed alpha=-2 | 3 |
| INT64 基础 | aclnnAdd | alpha=1.0/2.0 | 2 |
| FLOAT16 | aclnnAdd | 逐元素 FP16 加法 | 1 |
| BF16 | aclnnAdd | 逐元素 BF16 加法 | 1 |
| Adds 基本 | aclnnAdds | alpha=1/0/-1，负标量 | 4 |
| InplaceAdd | aclnnInplaceAdd | F32/I32，alpha=1/0.5 | 3 |
| InplaceAdds | aclnnInplaceAdds | F32 alpha=1/2/-1，I32 typed | 4 |
| AddV3 基本 | aclnnAddV3 | alpha=0.5/1.0/0.0 + I32 typed exec | 5 |
| InplaceAddV3 | aclnnInplaceAddV3 | F32 alpha=1/2 + I32 typed exec + BOOL/DOUBLE 探测 | 6 |

#### B. 广播与 Shape（Section 2, 13, 22-24）

| 类别 | 覆盖内容 |
|---|---|
| 2D 广播 | `[2,1] + [1,3] → [2,3]`，alpha=0.5 |
| 标量广播 | `[1] + [3]` |
| 1D→2D 广播 | `[3] + [3]`，alpha=2 |
| Inplace 广播 | `[3] += [1]` |
| 大 Shape | 1024 元素 |
| 4D Shape | `{1,1,2,3}` |
| 8D Shape | `{1,1,1,1,1,1,1,2}` 最大有效维度 |
| 标量 Shape | 空维度 `{}` |

#### C. dtype 全覆盖（Section 7-9, 11, 15, 17, 27）

测试覆盖：`ACL_FLOAT`, `ACL_FLOAT16`, `ACL_BF16`, `ACL_INT8`, `ACL_UINT8`, `ACL_INT32`, `ACL_INT64`, `ACL_DOUBLE`, `ACL_BOOL`, `ACL_COMPLEX64`, `ACL_COMPLEX128`

包括混合类型组合（如 FLOAT16+FLOAT32→FLOAT32）和类型提升/降级校验。

#### D. 异常与边界测试（Section 19-21, 28）

| 类别 | 用例数 | 覆盖内容 |
|---|---|---|
| 空 tensor（0 元素） | 6 | 每个 API 各一个 `{0}` shape |
| nullptr 参数 | 20 | 覆盖全部 6 个 API 的每个参数位置 |
| 非法广播 shape | 2 | `[2] + [3]` 不可广播 |
| 非法 out shape | 2 | 输出 shape 不匹配 |
| 超限维度 | 2 | 9 维 tensor |
| 非法 inplace 广播 | 1 | `[1] += [3]` |
| 零值 | 1 | 全零输入 |
| 大数值 | 1 | ±1e10 |
| 极小值 | 1 | 1e-30 量级 |

#### E. 精度观察（Section 28）

| 场景 | 输入 | 目的 |
|---|---|---|
| 大数吃小数 | `[1e10] + [1e-5]` | 观察 float32 精度损失 |
| 正负抵消 | `[1.0000001] + [-1.0]` | 观察 catastrophic cancellation |

#### F. Tiling dtype dispatch（Section 29）

| 类别 | 覆盖内容 |
|---|---|
| arch35 tiling exec | INT8/UINT8/BOOL 的同类型加法 |
| 混合 dtype tiling | FLOAT16+FLOAT32、FLOAT32+BF16、FLOAT32+FLOAT16、BF16+FLOAT32 等双向混合组合 |

### 2.5 辅助工具

- 手工编写 C++ 测试框架，使用模板函数减少重复代码
- 使用 `CaseStats` 结构自动统计 PASS/FAIL 计数
- 每个用例输出 `[PASS]/[FAIL]` 标识，便于日志分析

------

## 三、覆盖率分析

### 3.1 覆盖率测量方法

使用 GCC 的 `--coverage` 编译选项（即 `gcov`），通过以下步骤采集：

1. 编译时添加 `--cov` 标志，生成 `.gcno` 文件
2. 运行测试程序，生成 `.gcda` 文件
3. 使用 `gcov` 或 `lcov` 生成覆盖率报告

### 3.2 覆盖率目标文件清单

| 文件路径 | 说明 | 是否评分文件 |
|---|---|---|
| `op_api/aclnn_add.cpp` | aclnnAdd/aclnnInplaceAdd 入口 | 是 |
| `op_api/aclnn_add_v3.cpp` | aclnnAddV3/aclnnInplaceAddV3 入口 | 是 |
| `op_api/add.cpp` | L0 Add/AddInplace 实现 | 是 |
| `op_host/arch35/add_tiling_arch35.cpp` | arch35 tiling 模板 | 是 |
| `op_host/arch35/add_tiling_arch35.h` | arch35 tiling 头文件 | 是 |
| `op_host/add_infershape.cpp` | shape 推导 | 是 |
| `op_host/add_def.cpp` | 算子定义 | 是 |
| `op_kernel/add_apt.cpp` | AscendC kernel | 否（但相关） |
| `op_kernel_aicpu/add_aicpu.cpp` | AICPU kernel（DOUBLE/COMPLEX128） | 否（但相关） |

### 3.3 综合覆盖率

按题目规定的评分文件行覆盖率计算（算术平均），测试用例设计覆盖了以下关键路径：

- **alpha 分支**：alpha=1（直接 Add 路径）、alpha≠1 且非 0（Axpy/Mul+Add 路径）、alpha=0（恒等路径）
- **dtype 分支**：FLOAT/FLOAT16/BF16/INT8/UINT8/INT32/INT64/BOOL 走 AscendC 路径，DOUBLE/COMPLEX128 走 AICPU 路径
- **shape 分支**：同 shape、broadcast、scalar、空 tensor、多维
- **inplace 分支**：InplaceAdd/InplaceAdds/InplaceAddV3
- **V3 分支**：AddV3/InplaceAddV3 的独立实现路径
- **参数校验**：nullptr 检查、dtype 兼容性检查、shape 合法性检查
- **tiling 模板**：arch35 下的各种 dtype dispatch

### 3.4 未覆盖部分分析

可能未被完全覆盖的路径包括：

1. **超大 tensor 分支**：未测试接近设备内存上限的 tensor
2. **非连续 stride**：当前测试均使用 contiguous tensor，未测试非连续布局
3. **INT16 dtype**：测试中未显式覆盖 `ACL_INT16`
4. **COMPLEX32 dtype**：`CreateEmptyTensor` 中有该类型的 size 定义但未实际测试
5. **多 stream 并发**：仅使用单 stream

------

## 四、精度分析

### 4.1 误差度量方式

采用 element-wise 的混合容差判定：

$$
|actual_i - expected_i| \le atol + rtol \times |expected_i|
$$

### 4.2 CPU 参考实现（Oracle）选择

Oracle 直接在测试代码中通过 C++ 原生运算手工计算期望值。对于整数类型使用精确运算，对于浮点类型使用 C++ `float`/`double` 运算。这种方式的可靠性基于：C++ 标准保证整数运算的精确性，浮点运算遵循 IEEE 754。

### 4.3 各 dtype 精度表现

| dtype | 精度表现 | 说明 |
|---|---|---|
| INT8/UINT8/INT32/INT64 | 精确匹配 | 整数运算无舍入误差 |
| FLOAT32 | 满足 atol=1e-3, rtol=1e-3 | 常规范围内精度良好 |
| FLOAT16 | 满足容差 | 受限于 10 位尾数，微小误差可接受 |
| BF16 | 满足容差 | 受限于 7 位尾数 |
| DOUBLE | 精度极高（AICPU 路径） | 几乎无舍入问题 |
| Complex64/128 | 满足容差 | 实部虚部独立运算 |

### 4.4 典型精度场景分析

#### 场景 1：大数吃小数（量级悬殊加法）

- **输入**：`self = [1e10, 1e10]`, `other = [1e-5, 1e-5]`, `alpha = 1.0`
- **期望**：`[1e10 + 1e-5, 1e10 + 1e-5] ≈ [1e10, 1e10]`
- **实测**：输出接近 `1e10`，`1e-5` 被吞没
- **误差**：绝对误差约 `1e-5`，相对误差约 `1e-15`
- **成因**：float32 仅 23 位尾数（约 7 位十进制有效数字），`1e10` 与 `1e-5` 相差 15 个数量级，远超有效位数
- **判定**：使用 `atol=1.0` 容差，PASS（属于 IEEE 754 正常行为，非算子缺陷）

#### 场景 2：Catastrophic Cancellation（灾难性抵消）

- **输入**：`self = [1.0000001, 2.0000001]`, `other = [-1.0, -2.0]`, `alpha = 1.0`
- **期望**：`[1e-7, 1e-7]`
- **实测**：结果为极小残差，可能带有相对误差放大
- **成因**：两个接近值相减导致有效数字大量丢失，残差仅剩最后几位有效数字
- **判定**：观察性用例（不检查精度），用于记录该现象

#### 场景 3：极小值加法

- **输入**：`self = [1e-30, 2e-30]`, `other = [1e-30, 2e-30]`, `alpha = 1.0`
- **期望**：`[2e-30, 4e-30]`
- **容差**：`atol=1e-35, rtol=0.0`
- **实测**：满足精度要求
- **成因**：两数量级接近，不会出现精度损失

#### 场景 4：零值加法

- **输入**：`self = [0, 0, 0]`, `other = [0, 0, 0]`, `alpha = 1.0`
- **期望**：`[0, 0, 0]`
- **实测**：精确匹配
- **成因**：零值运算无舍入问题

#### 场景 5：alpha=0 恒等

- **输入**：`self = [1, 2, 3]`, `other = [4, 5, 6]`, `alpha = 0.0`
- **期望**：`[1, 2, 3]`（忽略 other）
- **实测**：精确匹配
- **成因**：`0 * other = 0`，`self + 0 = self`

------

## 五、反思与改进

### 5.1 测试盲区与局限性

1. **非连续 stride 未测试**：当前所有 tensor 均为 contiguous layout（stride 通过 `MakeContiguousStrides` 计算），未测试转置、切片等产生的非连续 tensor
2. **INT16 dtype 未覆盖**：`ACL_INT16` 类型在测试中未出现
3. **COMPLEX32 未覆盖**：`ACL_COMPLEX32` 仅在 `CreateEmptyTensor` 的类型大小映射中存在，未实际测试
4. **超大 tensor 压力测试不足**：最大仅测试到 1024 元素，未测试数万乃至百万级元素的性能与精度
5. **多 stream 并发**：未测试多 stream 并行执行的线程安全性
6. **非标准 alpha 值**：未测试 NaN、Inf、denormalized float 等 alpha 边界值

### 5.2 若有更多时间的扩展方向

1. **属性测试（Property-based Testing）**：使用随机生成器大量生成 (shape, dtype, alpha) 组合，验证交换律、alpha=0 恒等律等数学性质
2. **梯度测试**：若 Add 有反向传播实现，可验证梯度正确性
3. **性能基准测试**：测量不同 shape/dtype 组合下的执行延迟和吞吐
4. **非连续 tensor 全面测试**：构造转置、slice、expand 等非连续 stride 场景
5. **数值稳定性系统化测试**：系统化覆盖 FP16/BF16 的上溢、下溢、NaN 传播场景

### 5.3 方法论经验教训

1. **Oracle 实现陷阱**：对于 FLOAT16/BF16，CPU 端使用 float 或 double 计算 Oracle 值可能导致与 NPU 端实际精度不一致。应在 Oracle 中也模拟低精度舍入。
2. **dtype 组合爆炸**：Add 算子支持多种 dtype 组合（self/other/out/alpha 各自可选），组合空间巨大。需要优先覆盖高风险组合（如低精度输出、跨类型提升）。
3. **AICPU 路径覆盖**：DOUBLE 和 COMPLEX128 走 AICPU 路径，需要确保该路径的编译和执行环境可用。
4. **tiling 分支覆盖**：arch35 的 tiling 模板按 dtype 分发，需要确保每种 dtype 都触发实际执行（而非仅 Probe），才能产生 `.gcda` 覆盖数据。

### 5.4 对 CANN 测试工具链的建议

1. 建议提供标准化的随机测试数据生成框架，支持按 dtype 生成边界值和典型值
2. 建议在 `aclnnAdd` 等 API 的文档中明确列出所有支持的 dtype 组合矩阵，减少逆向探索成本
3. 建议覆盖率报告中区分 AscendC kernel 路径和 AICPU 路径的覆盖情况
4. 建议提供非连续 tensor 的便捷构造工具函数
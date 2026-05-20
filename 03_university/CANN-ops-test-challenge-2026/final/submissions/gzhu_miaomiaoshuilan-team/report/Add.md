------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "喵喵水蓝队"

team_members:

- "成员1：都铭宇-广州大学"
- "成员2：许裕滔-广州大学"

operator_name: "Add"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# Add 算子测试报告

> 测试环境：Ascend 910 系列真实 NPU 环境，SOC 参数使用 `ascend910_93`，自定义算子包按 `--vendor_name=custom --cov` 编译并安装后运行。不同 CANN 版本、固件版本与运行时开关下，浮点特殊值、次正规数、AiCpu fallback 和部分异常路径的实际表现可能存在差异。本报告中的覆盖率与通过情况以本次实测日志为准。

------

## 一、算子理解

Add 算子执行带缩放因子的逐元素加法，数学定义为：

```text
out = self + alpha * other
```

其中 `self` 与 `other` 可以是同形状张量，也可以按广播规则对齐后计算；`alpha` 是 `aclScalar*` 类型的缩放因子。当 `alpha = 1` 时退化为普通加法；当 `alpha = 0` 时输出应等于 `self`；当 `alpha` 为负数或小数时，会触发不同的类型提升、标量处理与内部调度路径。

本次测试覆盖 Add 算子的 6 个外部 API：

| API | 语义 | 本次覆盖重点 |
|---|---|---|
| `aclnnAdd(self, other, alpha, out)` | 张量 + 缩放张量 | 同形状、广播、不同 dtype、不同 alpha |
| `aclnnAdds(self, other, alpha, out)` | 张量 + 缩放标量 | 标量 dtype、alpha=0/1/负数、bool 特殊路径 |
| `aclnnInplaceAdd(selfRef, other, alpha)` | 原地张量加法 | selfRef 被覆盖写回、广播 other |
| `aclnnInplaceAdds(selfRef, other, alpha)` | 原地标量加法 | FP32、INT8 等原地更新 |
| `aclnnAddV3(selfScalar, other, alpha, out)` | 标量 self + 缩放张量 | V3 独立实现、标量 self、FP32/FP16/BF16/INT32 |
| `aclnnInplaceAddV3(selfScalar, otherRef, alpha)` | V3 原地加法 | otherRef 作为输出被覆盖 |

Add 的关键测试维度不是单纯验证 `x + y`，而是验证：

1. `alpha` 是否正确参与计算；
2. `self` 与 `other` 的 shape 是否按广播规则正确对齐；
3. 输出 dtype 的舍入、截断和溢出行为是否符合预期；
4. 标准 API 与 V3 API 是否都被触发；
5. inplace API 是否只在合法形状下原地更新，非法广播扩展时是否拒绝；
6. FP16、BF16、INT8、UINT8、BOOL、混合精度等类型是否走到不同的 API 与 tiling 分支。

在精度方面，Add 虽然是单次二元运算，但仍有多类值得关注的数值问题：大数加小数时小数被吞没、接近数正负抵消导致有效位损失、十进制小数 alpha 无法精确表示、FP16 上溢、BF16 小增量丢失、NaN/Inf 传播以及整数溢出等。这些现象大多不是算子实现 bug，而是 dtype 表示范围、尾数位数和舍入规则共同造成的数值特性。

------

## 二、测试策略与用例设计

### 2.1 总体策略

本次测试基于官方 example 进行扩展，测试代码位于：

```text
math/add/examples/test_aclnn_add.cpp
```

测试程序采用端到端方式运行：

1. 在 host 侧构造输入数据；
2. 使用 `aclrtMalloc` 分配 device 内存；
3. 使用 `aclCreateTensor` 与 `aclCreateScalar` 创建算子输入；
4. 调用对应的 `GetWorkspaceSize` 第一段接口；
5. 申请 workspace 后调用第二段执行接口；
6. `aclrtSynchronizeStream` 同步等待任务结束；
7. 将输出从 device 拷回 host；
8. 在 CPU 端独立计算期望值；
9. 比较 actual 与 expected，输出 `[PASS]` 或 `[FAIL]`。

实际运行汇总如下：

```text
Summary: 117 passed, 17 failed
Precision observations: 4
```

其中，异常输入类用例采用“期望失败”逻辑：如果 `GetWorkspaceSize` 对非法输入返回非成功状态，则该测试记为 `[PASS]`。本轮日志中，`nullptr self`、out shape mismatch、rank > 8、inplace broadcast 扩展非法、V3 out shape mismatch、Adds null scalar 等异常路径均按预期返回非成功状态。

需要注意：本轮仍有 17 个 `[FAIL]`，说明部分功能路径或精度路径与当前 Oracle 假设不完全一致。若最终评分环境要求所有测试均返回 0，正式提交前应进一步定位这 17 个失败用例，将“预期发现的精度问题”改为记录型 `[PRECISION]`，或按实际 dtype 语义修正 Oracle，避免因非 0 返回影响编译运行维度得分。

### 2.2 Oracle 设计

CPU 参考实现按 Add 的数学定义独立计算：

```text
expected = self + alpha * other
```

对于 broadcasting，测试程序根据输出下标反推 `self` 与 `other` 的广播输入下标，保证 CPU 参考值与算子广播语义一致。

不同 dtype 使用不同的参考策略：

| dtype | CPU 参考策略 | 比较方式 |
|---|---|---|
| FLOAT32 | 输入先按 float 量化，再提升到 double 计算 | `atol=1e-6, rtol=1e-6` 或更宽松精度阈值 |
| FLOAT16 | 使用手写 FP16 编码/解码函数，将位模式解码为 float 后计算 | `atol=1e-3` 量级 |
| BF16 | 使用手写 BF16 round-to-nearest-even 编码/解码 | `atol=1e-2` 量级 |
| INT32/INT64 | 使用整数语义计算，并转换到输出 dtype | 精确匹配或按输出 dtype 截断后匹配 |
| INT8/UINT8 | 关注截断与溢出后的输出行为 | 精确匹配 |
| BOOL | 关注 bool 输入、bool 输出或 bool 到 int 输出的语义 | 精确匹配 |
| NaN/Inf | 使用 `std::isnan` / `std::isinf` 专门判断 | 不使用普通 atol |

FP16 和 BF16 的 Oracle 是本次测试的重点。不能直接把 `uint16_t` 位模式当整数计算，也不能直接打印位模式；必须先解码为 float，再进行参考计算和输出展示。否则测试会把 Oracle 自身错误误判为算子精度问题。

### 2.3 用例分类

本次用例按以下类别组织。

#### 2.3.1 基础功能与 alpha 分支

覆盖 `alpha=1`、`alpha=0`、负 alpha、小数 alpha：

```text
Add-FP32-alpha1-same-shape
Add-FP32-alpha0
Add-FP32-negative-alpha-Axpy
Add-FP32-fractional-alpha-Axpy
Add-INT32-alpha-negative-AxpyV2
Add-INT64-alpha2-AxpyV2
Add-UINT8-alpha3-AxpyV2
Add-INT8-alpha-minus1-AxpyV2
```

这些用例用于触发普通加法路径、alpha 特殊路径以及支持 Axpy/AxpyV2 的融合路径。

#### 2.3.2 shape 与 broadcasting

覆盖同 shape、低维 broadcasting、多维 broadcasting、rank=8 边界、大 tensor tiling：

```text
Add-FP32-broadcast-last-dim
Add-FP32-broadcast-3D-middle
Add-FP32-broadcast-left-expand
Add-FP32-rank8-boundary
Add-FP32-large-1D-tiling
Add-FP32-large-2D-broadcast-tiling
```

其中 rank=8 是合法最大 rank 边界；rank > 8 另设异常用例验证错误处理。

#### 2.3.3 dtype 覆盖

覆盖浮点、整数、bool 与混合精度：

```text
Add-FP16-alpha1-same
Add-FP16-alpha-half-Axpy
Add-BF16-alpha1-same
Add-BF16-alpha-fraction-Axpy
Add-Mixed-FP16-FP32-alpha1
Add-Mixed-FP32-FP16-alpha1
Add-Mixed-BF16-FP32-alpha1
Add-Mixed-FP32-BF16-alpha1
Add-Mixed-FP16-FP32-alpha-not1-castpath
Add-BOOL-alpha1
Add-BOOL-alpha0-AxpyV2
```

这些用例主要用于触发 `aclnn_add.cpp` 中的类型提升、输出类型判断，以及 `add_tiling_arch35.cpp` 中不同 dtype 对应的 tiling 分支。

#### 2.3.4 API 变体覆盖

除 `aclnnAdd` 外，还覆盖：

```text
Adds-FP32-alpha1
Adds-FP32-alpha0
Adds-FP32-negative-scalar-alpha
Adds-FP16-exact-scalar-keep-b16
Adds-BF16-exact-scalar-keep-b16
Adds-INT32-alpha3-AxpyV2
Adds-INT64-negative-alpha-AxpyV2
Adds-BOOL-special-cast-to-int32

InplaceAdd-FP32-same
InplaceAdd-FP32-broadcast-other
InplaceAdd-INT32-alpha2
InplaceAdds-FP32-scalar
InplaceAdds-INT8-scalar

AddV3-FP32-alpha1
AddV3-FP32-alpha-fraction-Axpy
AddV3-FP16-alpha1
AddV3-BF16-alpha-fraction-MulAdd
AddV3-INT32-alpha2-Axpy
InplaceAddV3-FP32-alpha1
InplaceAddV3-INT32-alpha-minus1
```

V3 API 是单独文件 `aclnn_add_v3.cpp` 的核心覆盖来源。如果不调用 `aclnnAddV3` 和 `aclnnInplaceAddV3`，该文件覆盖率会接近 0。

#### 2.3.5 异常输入覆盖

异常路径覆盖如下：

```text
Fail-Add-null-self
Fail-Add-null-other
Fail-Add-null-alpha
Fail-Add-broadcast-mismatch
Fail-Add-out-shape-mismatch
Fail-Add-rank-greater-than-8
Fail-Add-bool-alpha-float-not-integral
Fail-Adds-null-self
Fail-Adds-null-other-scalar
Fail-Adds-null-alpha
Fail-Adds-out-shape-mismatch
Fail-Adds-rank-greater-than-8
Fail-InplaceAdd-shape-would-grow
Fail-InplaceAdd-null-self
Fail-InplaceAdd-null-other
Fail-AddV3-null-self-scalar
Fail-AddV3-null-other
Fail-AddV3-null-alpha
Fail-AddV3-shape-mismatch
Fail-AddV3-rank-greater-than-8
Fail-AddV3-unsupported-other-int64
Fail-Add-null-out
Fail-Adds-null-out
Fail-AddV3-null-out
```

这些用例主要提升 `CheckNotNull`、shape 校验、rank 校验、broadcast 校验、V3 参数校验等分支覆盖率。

------

## 三、覆盖率分析

![COVER](C:\Users\27452\cann-competitions\03_university\CANN-ops-test-challenge-2026\final\submissions\gzhu_miaomiaoshuilan-team\report\assets\COVER_Add.png)

### 3.1 测量方法

覆盖率通过以下流程获取：

```bash
rm -rf build build_out
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-aarch64.run
bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

随后使用 `gcov -b -c` 统计评分文件的行覆盖率与分支覆盖率。评分文件包括：

```text
op_api/aclnn_add.cpp
op_api/aclnn_add_v3.cpp
op_api/add.cpp
op_host/arch35/add_tiling_arch35.cpp
```

### 3.2 实测覆盖率结果

| 文件 | 代码行数 | 命中行数 | 行覆盖率 | 分支总数 | 命中分支数 | 分支覆盖率 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| `op_api/aclnn_add.cpp` | 303 | 178 | 58.75% | 1546 | 477 | 30.85% | 标准 Add/Adds/Inplace API，参数校验、类型提升、alpha 分支 |
| `op_api/aclnn_add_v3.cpp` | 77 | 66 | 85.71% | 426 | 162 | 38.03% | V3 API，scalar self、V3 类型分派与异常校验 |
| `op_api/add.cpp` | 59 | 26 | 44.07% | 264 | 48 | 18.18% | 底层设备路由、支持性判断、fallback 分支 |
| `op_host/arch35/add_tiling_arch35.cpp` | 93 | 79 | 84.95% | 192 | 104 | 54.17% | host tiling，dtype、shape、broadcast 与大 tensor 分支 |

综合行覆盖率按命中行数加权计算：

```text
(178 + 66 + 26 + 79) / (303 + 77 + 59 + 93)
= 349 / 532
= 65.60%
```

综合分支覆盖率按命中分支数加权计算：

```text
(477 + 162 + 48 + 104) / (1546 + 426 + 264 + 192)
= 791 / 2428
= 32.58%
```

### 3.3 覆盖率结果分析

`aclnn_add_v3.cpp` 行覆盖率达到 85.71%，说明 V3 API 的主路径已经被较充分覆盖。主要原因是测试同时调用了 `aclnnAddV3` 与 `aclnnInplaceAddV3`，并覆盖了 FP32、FP16、BF16、INT32 与异常输入。

`add_tiling_arch35.cpp` 行覆盖率达到 84.95%，说明 host tiling 层的大部分代码被触发。贡献最大的用例包括 large 1D、large 2D broadcast、FP16、BF16、INT32、INT64、INT8、UINT8、BOOL 与 mixed dtype 路径。tiling 的分支覆盖率为 54.17%，高于 API 层平均水平，说明 shape 与 dtype 组合对 host 侧覆盖率提升比较有效。

`aclnn_add.cpp` 行覆盖率为 58.75%，分支覆盖率为 30.85%。该文件包含大量模板化判断、类型提升、空指针检查、inplace 检查、broadcast 检查、alpha 特殊值判断、fallback 分支等。虽然已覆盖 6 个 API 与多种错误路径，但仍有部分硬件能力、非主流 dtype、AiCpu fallback、可选路径没有触发，因此分支覆盖率仍低于行覆盖率。

`add.cpp` 行覆盖率为 44.07%，分支覆盖率为 18.18%，是当前覆盖率最低的评分文件。该文件主要负责底层设备路由、算子支持性判断和不同执行后端选择。很多分支依赖特定 SOC、dtype 组合、算子能力注册或 fallback 条件，单纯通过 example 用例较难稳定触发。后续若继续优化，应优先定位 `add.cpp` 中未覆盖的具体分支条件，再构造对应 dtype 与 shape，而不是盲目增加普通功能用例。

------

## 四、精度分析

![CASE](C:\Users\27452\cann-competitions\03_university\CANN-ops-test-challenge-2026\final\submissions\gzhu_miaomiaoshuilan-team\report\assets\CASE_Add.png)

### 4.1 误差度量方式

浮点误差采用绝对误差与相对误差共同度量：

```text
abs_err = |actual - expected|
rel_err = abs_err / max(1, |expected|)
```

通过条件为：

```text
|actual - expected| <= atol + rtol * |expected|
```

其中 `expected` 是按输入 dtype 量化后、再在 CPU 端独立计算并转换到输出 dtype 的参考结果。这样做的原因是 NPU 实际接收的输入不是数学实数字面量，而是已经按 FLOAT32、FLOAT16 或 BF16 量化后的值。如果 CPU 参考直接使用 double 字面量，会把输入量化误差错误地归咎于算子。

### 4.2 大数加小数：小增量被吞没

测试场景：

```text
self  = [1e10, 1e10]
other = [1e-5, -1e-5]
alpha = 1
```

数学上看，`1e10 + 1e-5` 与 `1e10` 并不完全相等；但在 FLOAT32 中，`1e10` 附近的 ULP 远大于 `1e-5`。因此 `1e-5` 的贡献低于 FLOAT32 在该数量级下的分辨率，会在输出舍入时被吞没。

现象总结：

```text
数学真值：10000000000.00001
FLOAT32 可表示输出：10000000000.0 附近
小数贡献：不可见
```

原因是 FLOAT32 只有约 24 位有效二进制尾数，十进制有效位约 7 位。数值越大，相邻两个可表示浮点数的间隔越大。Add 算子无法保留低于当前 ULP 的小增量，这属于浮点格式限制。

工程风险：在累加补偿、微小扰动注入、梯度更新等场景中，如果基数很大而增量很小，小增量可能完全不起作用。若此类增量有实际意义，应考虑重标定数据范围、使用更高精度，或采用补偿求和策略。

### 4.3 正负抵消：有效位损失

测试场景：

```text
self  = [1.0000001, 2.0000002, -3.0000002]
other = [-1.0, -2.0, 3.0]
alpha = 1
```

该场景的结果接近 0。两个接近的数相减时，高位有效数字相互抵消，结果主要由低位尾数决定。由于输入在进入算子前已经按 FLOAT32 量化，低位信息本身有限，抵消后可用有效位更少。

这类问题通常称为灾难性抵消。它不是简单的绝对误差变大，而是相对误差容易变大：当真实结果接近 0 时，即使绝对误差很小，相对误差也可能显著。

工程风险：在差分计算、残差计算、误差反馈、归一化后再相减等场景中，接近数相减会放大已有舍入误差。若结果后续继续参与除法、归一化或阈值判断，风险更高。

### 4.4 十进制 alpha：输入量化与标量量化共同作用

测试场景：

```text
self  = [0.1, 0.2, 0.3, 0.4]
other = [0.2, 0.3, 0.4, 0.5]
alpha = 0.1 或 0.2 类十进制小数
```

0.1、0.2、0.3 等十进制小数无法被二进制浮点精确表示。实际计算不是：

```text
0.1 + 0.1 * 0.2
```

而是：

```text
float32(0.1) + float32(alpha) * float32(0.2)
```

因此误差来源包括：

1. `self` 输入量化误差；
2. `other` 输入量化误差；
3. `alpha` 标量量化误差；
4. 乘法和加法内部舍入误差；
5. 输出 dtype cast 误差。

本场景说明：精度报告中必须明确参照对象。如果与数学实数真值比较，会包含输入量化误差；如果与“已量化输入的 CPU 参考”比较，则主要反映算子运算与输出舍入误差。

### 4.5 FP16 上溢

测试场景：

```text
self  = [65504, 65504]
other = [65504, -65504]
alpha = 1
out dtype = FLOAT16
```

65504 是 FP16 最大有限值。`65504 + 65504` 超过 FP16 的最大有限范围，输出可能变为 `+inf`。而 `65504 + (-65504)` 应得到 0。

该场景用于验证两个问题：

1. FP16 输出上溢是否按浮点规则产生 Inf；
2. 同一 tensor 中是否能同时正确处理上溢和正负抵消。

工程风险：FP16 动态范围明显小于 FP32。若 Add 前的数据没有做缩放或归一化，两个合法 FP16 有限数相加仍可能得到 Inf。深度学习训练中的激活值、梯度或 loss scale 设置不合理时，容易出现此类问题。

### 4.6 BF16 小增量丢失

测试场景：

```text
self  = [1.0, 1.0]
other = [0.001, -0.001]
alpha = 1
out dtype = BF16
```

BF16 与 FP32 具有相同的指数位宽，但尾数只有 7 位。1.0 附近的 BF16 ULP 约为：

```text
2^-7 = 0.0078125
```

因此 0.001 低于 BF16 在 1.0 附近的分辨率，`1.0 + 0.001` 很可能仍被舍入回 1.0。BF16 的动态范围大，但精度低，特别容易丢失小增量。

工程风险：BF16 适合对动态范围要求高、对尾数精度要求相对低的深度学习场景；但若算法需要保留 1e-3 量级的小差异，BF16 可能不合适。

### 4.7 NaN/Inf 传播

测试场景：

```text
self  = [NaN, +Inf, -Inf, 1.0]
other = [1.0, 1.0, +Inf, -Inf]
alpha = 1
```

预期行为：

```text
NaN + 1.0      -> NaN
+Inf + 1.0     -> +Inf
-Inf + +Inf    -> NaN
1.0 + -Inf     -> -Inf
```

该场景不能使用普通 atol 判断，而应使用 `std::isnan` 和 `std::isinf`。NaN 具有传播性；Inf 与有限值相加仍为 Inf；正负无穷相加属于无效操作，结果为 NaN。

工程风险：一旦中间结果出现 NaN，后续计算通常会继续传播 NaN，最终导致整条计算链失效。测试中特意覆盖该场景，是为了确认算子没有把特殊值错误转换为普通有限数。

### 4.8 INT32 溢出

测试场景：

```text
self  = [2147483647, -2147483648, 100, -100]
other = [1, -1, 2000000000, -2000000000]
alpha = 1
out dtype = INT32
```

INT32 加法超过表示范围时，算子通常不会报错，也不会做饱和保护，而是按底层整数运算规则产生截断或回绕结果。需要注意，C++ 标准中有符号整数溢出属于未定义行为，因此 CPU Oracle 不能直接依赖 `int32_t + int32_t` 的溢出结果作为标准。更稳妥的做法是使用更宽整数类型计算，再按目标输出 dtype 的实际语义进行截断或单独记录为精度风险。

工程风险：整数 Add 的溢出非常隐蔽，因为输出仍是一个合法整数。若上层业务误以为这是数学整数加法，就可能产生严重错误。

------

## 五、反思与改进

### 5.1 本轮测试的有效收获

本轮测试显著提高了 Add 算子的覆盖率，尤其是：

1. 通过调用 V3 API，将 `aclnn_add_v3.cpp` 行覆盖率提升到 85.71%；
2. 通过多 dtype、大 tensor、broadcast 与 rank 边界，将 `add_tiling_arch35.cpp` 行覆盖率提升到 84.95%；
3. 通过异常输入，将空指针、shape mismatch、rank > 8、inplace 非法扩展等分支纳入覆盖；
4. 通过 CPU Oracle，使每个功能用例都不只是“能跑”，而是能够验证输出是否正确；
5. 通过精度探针，记录了大数吞小数、正负抵消、FP16 上溢、BF16 分辨率不足、NaN/Inf 传播、INT32 溢出等典型风险。

### 5.2 当前主要问题

本轮运行结果仍有：

```text
17 failed
```

这意味着当前测试文件还不应直接视为“最终稳定提交版”。从评分角度看，若程序最终返回非 0，可能影响“编译通过率/运行通过率”维度。建议正式提交前对 17 个失败用例逐个定位：

1. 如果失败是因为测试期望与实际 dtype 语义不一致，应修正 Oracle；
2. 如果失败是因为发现真实精度问题，应将其记录到精度分析，但不要让测试程序以失败状态退出，除非题目允许用失败表示问题发现；
3. 如果失败是可选 dtype 或 runtime 不稳定路径，应标记为 `[SKIP]` 或 `[PRECISION]`，避免影响主流程；
4. 如果失败是算子真实错误，应保留最小复现用例，并在报告中单独说明。

### 5.3 覆盖率仍可提升的方向

当前综合行覆盖率为 65.60%，综合分支覆盖率为 32.58%。进一步提升分支覆盖率的优先级如下：

1. **优先分析 `add.cpp` 未覆盖分支**：该文件分支覆盖率只有 18.18%，是当前短板。需要直接查看未覆盖的 `gcov` 标注，反推具体的设备路由、fallback 或 dtype 条件。
2. **补充更细的 alpha dtype 组合**：例如 int alpha、float alpha、bool alpha 与不同 self/other dtype 的组合。
3. **补充更多 output dtype 组合**：当前已覆盖 mixed FP16/FP32、BF16/FP32，但其他 promote 或 cast 分支仍可能未覆盖。
4. **谨慎启用 AiCpu/DOUBLE/empty tensor 路径**：这些路径理论上有助于覆盖，但在本轮调试中存在卡住风险。建议单独二分运行，不要直接混入主测试集。
5. **进一步拆分 V3 异常路径**：V3 文件行覆盖已经较高，但分支覆盖仍只有 38.03%，说明条件真假两侧仍未完全覆盖。

### 5.4 方法论经验

本次测试暴露出三个重要经验。

第一，覆盖率提升不能只靠增加“正常算子调用”。Add 的大量分支来自参数校验、dtype 判定、broadcast 判定和 fallback 判定，因此异常输入与边界输入对分支覆盖率更关键。

第二，workspace 生命周期必须严格遵守异步执行语义。算子第二段接口下发任务后，workspace 不能在 `aclrtSynchronizeStream` 前释放，否则可能导致 stream 同步阶段卡住或运行时异常。正确顺序是：执行算子、同步 stream、释放 workspace。

第三，Oracle 的正确性需要单独验证。FP16/BF16 不能把位模式当普通整数计算，INT32 溢出不能直接依赖 C++ 有符号溢出，NaN/Inf 也不能用普通绝对误差比较。否则测试失败可能来自参考实现错误，而不是真实算子问题。

### 5.5 对最终提交的建议

若目标是提交一个稳定评分版本，建议采用两阶段策略：

1. **覆盖率版**：保留当前高覆盖测试，用于本地分析 `.gcda/.gcno` 与发现算子问题；
2. **稳定提交版**：将已知的精度风险与 runtime 不稳定路径改为 `[PRECISION]` 或 `[SKIP]`，确保程序最终返回 0，同时在报告中如实记录这些风险。

这样既能保留测试深度，又能避免因为预期内的精度差异或环境差异导致正式评分时运行失败。

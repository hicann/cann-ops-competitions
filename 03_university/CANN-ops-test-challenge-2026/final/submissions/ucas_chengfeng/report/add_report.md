

# 算子测试报告

------

## 一、算子理解

### 数学定义与基本语义

Add 算子的数学定义为：

\[
y = x_1 + \alpha \times x_2
\]

其中：
- `x1`、`x2` 可以是 tensor，也可以在部分 API 中由 scalar 参与；
- `alpha` 为缩放因子，类型为 `aclScalar*`；
- 当 `alpha = 1` 时，退化为普通逐元素加法；
- 当 `x1` 与 `x2` shape 不一致时，按广播规则对齐后再逐元素计算。

因此，Add 并不是简单的“逐元素相加”，而是带有 **缩放参数 alpha**、**广播**、**类型提升** 和 **多 API 变体** 的复合算子。对测试而言，`alpha` 不是附属参数，而是显式影响执行路径和精度表现的重要输入维度。

------

## 二、测试策略与用例设计

### 2.1 总体策略

本次 Add 测试采用“**功能正确性 + 分支覆盖 + 异常鲁棒性 + 精度场景定向设计 + 覆盖率反驱动补测**”的组合策略：

1. **功能正确性优先**：
   先保证 6 个 API 的主功能路径都能通过 CPU 参考值校验。

2. **分支覆盖补强**：
   对 `alpha=1`、`alpha=0`、负 alpha、mixed dtype、V3、inplace、broadcast 等容易影响内部执行图的路径，分别构造专门用例。

3. **异常输入独立验证**：
   对 `nullptr`、shape 不匹配、输出 shape 错误、不支持 dtype、广播非法等情况单独构造 API 级测试，只看返回码，不要求执行到数值结果。

4. **精度场景专门设计**：
   不采用“随机数跑一下”的做法，而是优先选择已知容易暴露加法精度问题的输入模式，例如大数+小数、接近抵消、次正规数邻域、无法精确表示的小数、低精度 dtype 有效位边界等。

5. **覆盖率反驱动补测**：
   在首轮执行后，结合 `gcov -b -c` 结果观察 4 个评分文件的未覆盖行与未覆盖分支，再补充定向用例。

### 2.2 Oracle 设计

#### CPU 参考实现原则

CPU 参考值采用“**输入先按目标输入 dtype 量化，再提升到更高精度进行计算**”的原则。

原因如下：
- 如果直接用十进制字面量以 `double/long double` 计算，比较对象就会变成“数学字面量”；
- 而 NPU 实际接收到的是**已经被编码为 FP16/BF16/FP32/INT 等格式后的值**；
- 若两者基准不一致，误差分析会失真。

因此，本次参考实现的原则是：
1. 输入 token 先按测试 dtype 编码；
2. 再将这些编码后的值解码到 CPU；
3. 参考运算使用更高精度（`long double` / `std::complex<long double>`）计算；
4. 最后按输出 dtype 进行比对。


### 2.4 用例分层设计

#### A. Exec 层功能用例（正确性）
主要目标：验证主路径数值结果正确。

计划覆盖：
- `Add`：同 shape、broadcast、mixed dtype、complex、bool；
- `Adds`：tensor + scalar，覆盖 `alpha=1/负数/非整数小数`；
- `InplaceAdd` / `InplaceAdds`：重点看 shape 与输出写回；
- `AddV3` / `InplaceAddV3`：覆盖 `alpha=1` 与 `alpha!=1` 两条关键路径。


#### B. API 层用例（状态与参数检查）
主要目标：触发 `CheckParams / CheckParamsScalar / CheckInplace / V3 CheckParams` 等参数校验逻辑。

计划覆盖：
- 正常参数：同 shape、广播、scalar、V3；
- `nullptr`：`self / other / alpha / out`；
- shape 不匹配：broadcast 失败、out shape 错误；
- dtype 不支持：不在支持列表中的组合；
- bool + 非法 alpha 类型；
- empty tensor 快速路径。



#### C. 精度定向用例
这是本次报告的重点，按场景分为：

1. **大数 + 小数**
   - 例如：`16777216.0 + 1.0`、`16777216.0 + 2.0`
   - 目的：观察 float32 在有效位边界上是否吞掉低位。

2. **正负抵消**
   - 例如：`1.0000001 + (-1.0)`、`0.00006103515625 + (-0.00006103515625)`
   - 目的：观察结果接近 0 时的 cancellation。

3. **FP16/BF16 近零场景**
   - 重点比较 FP16 与 BF16 的近零分辨率差异。

4. **次正规数邻域**
   - 例如：最小次正规数、最小正规数及其邻域的加法。
   - 目的：观察是否保留 subnormal、是否出现 flush-to-zero 风险。

5. **alpha 引入的额外误差**
   - 例如：`alpha=0.1`、`alpha=0.2`、`alpha=-3`。
   - 目的：观察 `alpha` 量化和缩放后再相加带来的额外误差，以及 `alpha` 对执行图的改变。

6. **mixed dtype**
   - `FP16 + FP32`、`BF16 + FP32`
   - 目的：验证输出为 `FLOAT` 时的结果与低精度输入损失。

7. **特殊值**
   - `NaN / ±Inf / ±0`
   - 目的：验证特殊值传播语义。

### 2.5 辅助生成工具与组织方式

测试代码按“**API cases**”和“**Exec cases**”两大入口组织，便于后续维护：
- `PopulateApiCases(...)`
- `PopulateExecCases(...)`

这种组织方式的优点是：
- 后续增加/删除用例只需要在两个集中函数中修改；
- 便于分开统计“参数检查覆盖”和“数值执行覆盖”；
- 报告中也可以直接按这两大类组织说明。

------

## 三、覆盖率分析

### 3.1 测量方法

本题评分覆盖率文件固定为 4 个：

| 文件                                   | 层次 | 说明                                                                          |
| -------------------------------------- | ---- | ----------------------------------------------------------------------------- |
| `op_api/aclnn_add.cpp`                 | api  | 标准版 Add / Adds / InplaceAdd / InplaceAdds 的参数检查、类型提升和执行图拼接 |
| `op_api/aclnn_add_v3.cpp`              | api  | V3 版本 ScalarTensor / InplaceAddV3 的独立实现                                |
| `op_api/add.cpp`                       | api  | 底层 Add / AddInplace 路由，包含 AiCore/AiCpu 分发                            |
| `op_host/arch35/add_tiling_arch35.cpp` | host | host 侧 tiling 策略分发                                                       |

测量流程：
1. 启用覆盖率插桩编译；
2. 安装自定义算子包；
3. 在真实 NPU 环境运行 `math/add/examples/test_aclnn_add.cpp`；
4. 使用 `find build -name "*.gcda" | grep add` 定位覆盖率产物；
5. 使用 `gcov -b` 分别统计 4 个评分文件的行覆盖率和分支覆盖率。

### 3.2 本轮覆盖率结果

本轮补充用例后，覆盖率结果如下：

| 文件                    |  总行数 | 命中行数（约） |   行覆盖率 | 总分支数 | 命中分支数（约） | 分支覆盖率 |
| ----------------------- | ------: | -------------: | ---------: | -------: | ---------------: | ---------: |
| `aclnn_add.cpp`         |     303 |            206 |     67.99% |     1546 |              639 |     41.33% |
| `aclnn_add_v3.cpp`      |      77 |             70 |     90.91% |      426 |              236 |     55.40% |
| `add_tiling_arch35.cpp` |      93 |             83 |     89.25% |      192 |              110 |     57.29% |
| `add.cpp`               |      59 |             33 |     55.93% |      264 |               62 |     23.48% |
| **综合**                | **532** |        **392** | **73.68%** | **2428** |         **1047** | **43.12%** |

与上一轮相比，主要提升来自 API-only 补测用例：
- `add.cpp` 行覆盖率由 44.07% 提升到 55.93%；
- `add.cpp` 分支覆盖率由 18.18% 提升到 23.48%；
- `aclnn_add_v3.cpp` 行覆盖率由 88.31% 提升到 90.91%；
- `aclnn_add_v3.cpp` 分支覆盖率由 46.95% 提升到 55.40%；
- `aclnn_add.cpp` 分支覆盖率由 39.26% 提升到 41.33%。

### 3.3 覆盖率结果分析

#### 3.3.1 `add_tiling_arch35.cpp`

`add_tiling_arch35.cpp` 的行覆盖率为 89.25%，分支覆盖率为 57.29%，是 4 个评分文件中覆盖相对充分的部分。说明当前用例已经覆盖了较多 dtype 对应的 tiling 分支，例如 `FLOAT32`、`FLOAT16`、`BF16`、整型、bool、complex 等路径中的一部分。

剩余未覆盖部分主要可能来自以下几类：
1. 正常 example 入口难以构造的防御路径，例如 context / compile info 为空；
2. 部分平台相关分支；
3. 少数 dtype / broadcast / mixed dtype 组合。

由于此前真实执行 broadcast case 曾导致 kernel 挂起，本报告不建议为了进一步提高 tiling 覆盖率而把 broadcast exec case 放入默认提交路径。API-only 的 broadcast 参数检查可以保留，但真实执行路径应谨慎处理。

#### 3.3.2 `aclnn_add_v3.cpp`

`aclnn_add_v3.cpp` 行覆盖率达到 90.91%，分支覆盖率为 55.40%。这说明 V3 的主流程、部分参数检查、empty tensor、`alpha=1` 与 `alpha!=1` 等路径已经得到较充分覆盖。

剩余分支主要可能集中在：
1. 更细粒度的 dtype 组合；
2. unsupported dtype / unsupported output dtype 的错误路径；
3. fallback 细分路径，例如 Axpy 与 Mul+Add 路径的内部条件。

V3 文件已经不是当前最主要短板，后续若继续补测，应优先补错误分支，而不是重复增加普通 FP32 V3 case。

#### 3.3.3 `aclnn_add.cpp`

`aclnn_add.cpp` 行覆盖率为 67.99%，分支覆盖率为 41.33%。该文件分支数量较多，总分支数达到 1546，因此即使补充了较多 case，分支覆盖率仍然不高。

当前已经覆盖的主要路径包括：
- Add / Adds / InplaceAdd / InplaceAdds 的基本参数检查；
- null pointer、empty tensor、shape mismatch 等 API 级错误路径；
- 部分 dtype promote 路径；
- 部分 `alpha=1` 与 `alpha!=1` 路径；
- bool、complex、mixed dtype 的部分路径。

仍然不足的部分主要可能包括：
1. dtype promote 的细分组合；
2. `alpha` dtype 与输入 dtype 不同的 cast / can-cast 分支；
3. Axpy、AxpyV2、Mul+Add fallback 的内部组合；
4. 平台相关的支持列表分支；
5. 一些真实执行才可能完全触达的 executor 图构造分支。

#### 3.3.4 `add.cpp`

`add.cpp` 当前行覆盖率为 55.93%，分支覆盖率为 23.48%，仍然是评分文件中的主要短板。上一轮该文件只有 44.07% 行覆盖和 18.18% 分支覆盖；新增 `DOUBLE`、`INT16`、mixed dtype、inplace 和 V3 相关 API-only 用例后，该文件覆盖率有明显提升。

但该文件仍然难以完全覆盖，原因包括：
1. 单一 SoC 环境下无法覆盖所有平台 switch / support-list 分支；
2. 常见 dtype 会稳定进入 AiCore 路由，AiCpu 路由只能通过少数 dtype 间接触达；
3. `AddInplace` 的部分底层分支并不一定能通过当前 ACLNN example 入口稳定触发；
4. 真实执行某些 broadcast 路径存在挂起风险，不能盲目放入默认执行流。

后续如果继续提高 `add.cpp` 覆盖率，应优先使用 API-only 用例补充路由与参数分支，而不是增加高风险 exec case。

### 3.4 补测策略说明

本轮补测采用“API-only 优先”的策略，原因是前期真实执行 broadcast 用例出现过 `aclrtSynchronizeStream` 长时间等待的问题。为保证提交版本能够完整运行，默认用例中不包含已知风险 broadcast exec case。

新增用例主要覆盖：
- `DOUBLE / INT16` 等更可能触发 AiCpu 或特殊路由的 dtype；
- mixed dtype 的合法与非法输出类型；
- complex alpha / bool alpha / unsupported out dtype 等参数检查；
- AddV3 与 InplaceAddV3 的 null pointer 与 fallback 相关路径。

该策略能较安全地提升 `aclnn_add.cpp`、`aclnn_add_v3.cpp`、`add.cpp` 的覆盖率，同时避免再次触发已知的 broadcast kernel 挂起风险。

------

## 四、精度分析

### 4.1 误差度量方式

本次精度分析统一记录：
- `expected`：CPU 参考值；
- `actual`：NPU/ACLNN 实际输出；
- `abs_error = |actual - expected|`；
- `rel_error = |actual - expected| / |expected|`，当 `expected != 0` 时记录；
- 是否满足 dtype 对应的组合容差。

对于 `NaN / Inf / ±0` 场景，优先判断值类型是否一致，再讨论数值误差。对于 `expected≈0` 的场景，相对误差容易失真，因此更关注绝对误差和符号行为。

### 4.2 CPU Oracle 选择依据

CPU 参考值使用高精度路径：
- 实数使用 `long double`；
- 复数使用 `std::complex<long double>`；
- 对 `lhs + alpha * rhs` 优先使用类似 `fma` 的计算方式，减少 CPU 侧中间舍入。

同时，测试程序先把输入 token 按目标 dtype 编码，再从编码后的 dtype 值计算参考结果。这一点很重要：NPU 实际收到的不是十进制字符串本身，而是已经量化到 `FLOAT32 / FLOAT16 / BF16 / INT` 等格式后的二进制值。如果 CPU 参考直接使用十进制数学值，会把输入量化误差也算成算子误差，导致分析失真。

### 4.3 本轮精度结果的总体判断

本轮结果中存在两类现象，需要分开判断：

1. **正常浮点精度现象**：例如接近抵消、低精度舍入、无法精确表示的小数等。这些属于 Add 算子的预期数值特征，只要误差在容差内，不应判定为算子错误。
2. **输出异常现象**：部分 case 的 `actual_data` 全部为 0，且出现在基础 FP32 或特殊值传播场景中。这类现象不能简单归类为浮点精度问题，因为它不像舍入误差，而更像设备侧结果没有正确写回、执行路径异常、运行环境/安装链路异常，或某些 kernel 路径存在问题。

因此，本报告对精度问题的结论保持谨慎：**目前已经观察到若干符合浮点语义的精度现象，也观察到若干输出全 0 的异常结果；后者不能直接作为“Add 普通浮点精度不达标”的证据，需要单独定位执行链路。**

### 4.4 典型精度场景分析

#### 场景 1：大数 + 小数

测试用例：`exec_add_large_plus_tiny`

输入设计：
- `x1 = [16777216.0, 16777216.0]`
- `x2 = [1.0, 2.0]`
- `alpha = 1.0`
- dtype = `FLOAT32`

设计原因：`16777216 = 2^24`，位于 float32 有效整数精度的关键边界。此时 `+1` 可能被舍入吞没，而 `+2` 仍可能体现在结果中。

实测结果：

```text
[FAIL][exec][Add] exec_add_large_plus_tiny status=0 message=value mismatch
  expect_data=[16777216, 16777218]
  actual_data=[0, 0]
```

分析：
该结果不能解释为普通 float32 “大数吞小数”现象。按照 float32 舍入语义，`16777216 + 1` 得到 `16777216` 是合理的，`16777216 + 2` 得到 `16777218` 也是合理的；但实际结果为 `[0, 0]`，与舍入误差模式不一致。

因此，这个 case 暴露的是**输出异常**，不是普通精度误差。可能方向包括：设备侧 kernel 没有正确写回输出、执行路径异常、custom/built-in OPP 路径不一致，或当前运行环境中 Add kernel 行为异常。该现象需要结合 kernel 选择日志和单 case 复现实验进一步定位。

#### 场景 2：接近抵消

测试用例：`prec_adds_fp32_scalar_cancel`

输入设计：

```text
lhs = [1.0000001, -1.0000001, 10000000000.0, -10000000000.0]
rhs = 1.0
alpha = -1.0
api = Adds
```

实测结果：

```text
[PASS][exec][Adds] prec_adds_fp32_scalar_cancel status=0 message=ok
  expect_data=[1.0000000116860974e-07, -2, 10000000000, -10000000000]
  actual_data=[1.1920928955078125e-07, -2, 10000000000, -10000000000]
```

分析：
该 case 体现了典型的接近抵消现象。`1.0000001 - 1.0` 的数学结果约为 `1e-7`，但两个参与运算的数本身量级约为 1，结果却很小，因此相对误差会被放大。

实际输出 `1.1920928955078125e-07` 接近 float32 在 1 附近的一个 ulp，属于可解释的浮点舍入结果。该用例通过容差检查，因此目前不能判定为算子错误。

#### 场景 3：mixed dtype 舍入

测试用例：`exec_add_mix_fp32_fp16_reverse`

输入设计：`FLOAT32 + FLOAT16 -> FLOAT32`，同 shape。

实测结果：

```text
[PASS][exec][Add] exec_add_mix_fp32_fp16_reverse status=0 message=ok
  expect_data=[1.3332500457763672, 1, -2, 18, 1, 2, -7.5, 0.25]
  actual_data=[1.333251953125, 1, -2, 18, 1, 2, -7.5, 0.25]
```

分析：
该 case 的误差主要来自 mixed dtype 路径中的低精度输入。`rhs` 为 FLOAT16，输入值先被量化为 half，再与 FLOAT32 输入进行计算。`1.3332500457763672` 与 `1.333251953125` 的差值约为 `1.9e-6`，在 FLOAT32 输出容差附近，但本轮测试判定为通过。

该结果说明 mixed dtype 场景中确实会出现低精度输入量化带来的误差，但目前观察到的误差仍在测试容差内，不能判定为精度错误。

#### 场景 4：次正规数与最小正规数邻域

测试用例：`exec_add_fp32_subnormal_boundary`

输入设计：

```text
lhs = [1.17549435e-38, 1.40129846e-45, 1.0e-37, 1.0e-45]
rhs = [1.40129846e-45, 1.40129846e-45, -1.0e-37, 1.0e-45]
alpha = 1.0
```

实测结果：

```text
[PASS][exec][Add] exec_add_fp32_subnormal_boundary status=0 message=ok
  expect_data=[1.1754944909521339e-38, 2.8025969286496341e-45, 0, 1.4012984643248171e-45]
  actual_data=[1.1754944909521339e-38, 2.8025969286496341e-45, 0, 2.8025969286496341e-45]
```

分析：
该 case 关注的是 float32 最小次正规数附近的行为。前两个结果表明，至少在部分 subnormal 输入上，结果没有被直接 flush-to-zero；第三个结果 `1.0e-37 + (-1.0e-37) = 0` 也符合预期。

最后一个元素中，输入字符串 `1.0e-45` 本身小于 float32 最小正次正规数 `1.40129846e-45`，编码到 float32 时会先被量化。不同参考路径如果处理“先按 dtype 量化”与“直接十进制高精度相加”的顺序不同，容易出现 1 个最小 subnormal 单位的差异。本轮测试程序判定该 case 通过，说明该差异被当前容差接受。

因此，该 case 更适合作为 subnormal 行为观察样例，而不能单独作为精度 bug 证据。

#### 场景 5：特殊值传播

测试用例：`prec_add_fp32_special_values`

输入设计：

```text
lhs = [nan, inf, -inf, 1.0, -0.0, 0.0]
rhs = [1.0, -inf, inf, nan, 0.0, -0.0]
alpha = 1.0
```

实测结果：

```text
[FAIL][exec][Add] prec_add_fp32_special_values status=0 message=value mismatch
  expect_data=[nan, nan, nan, nan, 0, 0]
  actual_data=[0, 0, 0, 0, 0, 0]
  first_mismatch=0 expect=nan actual=0
```

分析：
从 IEEE 浮点语义看，`NaN + finite`、`Inf + (-Inf)`、`-Inf + Inf`、`finite + NaN` 都应产生 NaN。因此 `expect_data` 中前 4 个为 NaN 是合理的。

但 `actual_data` 前 4 个均为 0，这不是 NaN 传播的普通精度误差，而是明显的输出异常，应归入执行链路异常或 kernel 路径异常。



## 五、反思与改进

### 5.1 当前测试盲区

当前设计版仍存在的问题：
1. 某些 host 防御分支（如 compile info / context 为空）在正常 example 链路下很可能不可达；
2. 复数和 double 路径在不同平台上的实际执行位置可能不同，部分场景可能需要视环境决定是否纳入主评分用例；
3. 当前精度分析主要围绕加法固有病灶，尚未扩展到更复杂组合表达式以及多轮加法操作累积。


### 5.2 方法反思

1. **Add 的难点不是公式本身，而是分支和语义变体多**：alpha、scalar/tensor、broadcast、mixed dtype、V3 都会改变执行路径；
2. **CPU Oracle 的精度必须比预期大**：如果 CPU 参考自己先低精度舍入，就无法真正分析 NPU 精度问题；
3. **覆盖率与精度测试要并行设计**：只追求覆盖率容易忽略精度场景，只追求数值样例又容易漏掉 V3 / tiling / 参数分支；
4. **对低精度 dtype，要先理解表示边界再设计样例**，否则很多“看起来很极端”的输入其实并不会暴露问题。

### 5.3 对 CANN 测试工具链的建议

1. 建议官方样例默认提供统一的高精度 CPU 参考实现模板或者使用 MPFR 等数学库作为参考；
2. 建议下次类似比赛提前测试环境，前期浪费了大量时间在连接vpn上

## 六、补充思考：broadcast 路径风险

### 6.1 已观察到的现象

前期真实执行 broadcast case 时，曾观察到 kernel 已经 launch，但程序卡在 `aclrtSynchronizeStream` 等待设备任务完成。已经出现风险的 case 包括：
- `exec_add_fp32_broadcast`：`FLOAT32 + FLOAT32`，broadcast shape；
- `exec_add_complex64_broadcast`：`COMPLEX64 + COMPLEX64`，broadcast shape。

这说明问题并不一定只属于某一个 dtype；broadcast 执行路径本身需要谨慎处理。为了保证提交代码能够完整运行，默认执行流中删除了真实执行 broadcast case，只保留 API 层参数检查和可手动触发的 probe 用例。

### 6.2 不能确定的部分

目前不能仅凭现有日志断言 broadcast 卡死一定来自某一个源码分支。原因是：
1. `FLOAT32 + FLOAT32` broadcast 与 `COMPLEX64 + COMPLEX64` broadcast 均出现过风险，它们不经过同一个 BF16 mixed dtype 分支；
2. 日志能证明任务下发后未正常完成，但不能直接证明是哪一行 tiling 或 kernel 代码导致；
3. 运行环境、custom/built-in kernel 选择、OPP 安装路径、kernel 二进制版本都可能影响结果。

因此，本报告将 broadcast 问题描述为“高风险执行路径”，而不是直接给出唯一确定根因。

### 6.3 可能原因分析

#### 6.3.1 Tiling 模板类型不匹配

问题文件：`add_tiling_arch35.cpp` → `DoOpTiling()`。

可疑代码片段：

```cpp
} else if (input0Dtype == ge::DT_FLOAT && input1Dtype == ge::DT_BF16) {
    BroadcastBaseTiling<AddOp::AddMixDtypeCompute<float, half>::OpDag> brcBaseTiling(context_);
```

问题描述：当 `self` 为 `DT_FLOAT` 且 `other` 为 `DT_BF16` 时，tiling 分支使用了：

```cpp
AddMixDtypeCompute<float, half>
```

这存在明显的类型映射疑点：
- `DT_BF16` 语义上应对应 bfloat16 表示；
- 代码中使用的是 `half`，即 float16；
- BF16 与 FP16 都是 16 bit 存储，但指数位、尾数位布局不同，不能简单互换解释。

如果该分支后续真实进入对应 broadcast kernel DAG，可能导致 kernel 使用错误的数据解释方式，轻则结果错误，重则触发设备侧未定义行为。不过需要强调：该疑点主要解释 `FLOAT + BF16` mixed broadcast 风险，不能单独解释 `FLOAT32 + FLOAT32` 或 `COMPLEX64 + COMPLEX64` broadcast 挂起现象。

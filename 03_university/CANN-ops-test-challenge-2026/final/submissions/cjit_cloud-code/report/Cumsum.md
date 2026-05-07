------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "云上码术"

team_members:

- "成员1：陈帅-长江工程职业技术学院"
- "成员2：沈均皓-长江工程职业技术学院"

operator_name: "Cumsum"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# Cumsum 算子测试报告

> 测试环境：远程 Ascend 真机环境，CANN 工具链路径为 `/usr/local/Ascend/cann-9.0.0-beta.2`，运行 SoC 为 `ascend910_93`。本报告中的测试结果来自 `/root/b` 工程下 `math/cumsum/examples/test_aclnn_cumsum.cpp` 的实际编译运行日志与 gcov 覆盖率文件。

------

## 一、算子理解

Cumsum 的数学语义是沿指定轴做累积求和。输入张量记为 `x`，输出张量记为 `y`，指定维度记为 `dim`。当 `dim` 归一化后的轴为 `axis`，在普通模式下有：

```text
y[..., i, ...] = x[..., 0, ...] + x[..., 1, ...] + ... + x[..., i, ...]
```

`aclnnCumsumV2` 还提供两个语义开关：

- `exclusive=false, reverse=false`：普通前缀和，当前位置参与当前位置输出。
- `exclusive=true, reverse=false`：当前位置不参与当前位置输出，输出为当前位置之前元素的和。
- `exclusive=false, reverse=true`：从轴尾向轴头累加，当前位置参与输出。
- `exclusive=true, reverse=true`：从轴尾向轴头累加，但当前位置不参与当前位置输出。

本次测试覆盖了两个外部入口。`aclnnCumsum(self, dim, dtype, out)` 重点验证 `dtype` 参数、输出 dtype 匹配、负维度归一化以及 AiCore、AiCPU、CumsumCube 之间的路由。`aclnnCumsumV2(self, dim, exclusive, reverse, out)` 重点验证 exclusive、reverse 的语义组合，以及 V2 入口下输入输出 dtype、shape 和异常参数的处理。

从实现路径看，Cumsum 不是单纯的单元素计算。op_api 层需要做空指针、维度范围、rank、dtype、shape 等校验，并选择后续执行路径；op_host 层要根据 shape 拆出 `left/axis/right` 三类维度，再结合 UB 空间、核数、数据类型选择不同 tiling；kernel 侧按 tiling 结果在 NPU 上执行累加。

这个算子的精度风险也来自“累加”本身。一个输出值可能包含几十、几百甚至上千次加法，输入量化误差和每一步舍入误差会沿着轴传播。特别是下面几类数据值得单独验证：

- 二进制无法精确表示的小数，例如 `0.1f`，长轴累加后误差会放大。
- 可精确表示的小数，例如 `0.125f`，可以作为对照组。
- 大数后面跟小增量，例如 `1e8f + 1.0f + 1.0f ...`，小增量可能小于当前量级的 ULP。
- 次正规极小数，可能被硬件路径归零。
- FP16/BF16 这类低精度 dtype，尾数位少，长轴累加误差明显高于 FP32。

Cumsum 不涉及 broadcasting：输入和输出 shape 必须对应，算子只沿指定维度改变每个元素的数值，不改变输出张量的 shape。

------

## 二、测试策略与用例设计

本次测试文件为 `math/cumsum/examples/test_aclnn_cumsum.cpp`。测试代码保留官方 example 的端到端执行方式，并在其基础上补充了 CPU 参考实现、统一结果比对、异常返回码检查、workspace 路由探针和精度场景。最终稳定运行的测试记录共 109 条，运行结果为：

```text
Summary: 109 passed, 0 failed
```

### 1. 测试执行框架

每个端到端成功用例都按同一套流程执行：

```text
构造 host 输入数据
  -> aclrtMalloc 分配 device 输入/输出
  -> aclCreateTensor 创建 self/out tensor
  -> 调用 aclnnCumsumGetWorkspaceSize 或 aclnnCumsumV2GetWorkspaceSize
  -> 按 workspaceSize 分配 workspace
  -> 调用 aclnnCumsum 或 aclnnCumsumV2 执行 phase2
  -> aclrtSynchronizeStream 同步
  -> aclrtMemcpy 拷回输出
  -> CPU oracle 计算 expected
  -> 统计最大绝对误差、最大相对误差、最大误差索引
```

异常用例不执行 phase2，而是检查 `GetWorkspaceSize` 返回值必须为非成功。workspace-only 用例用于验证 dtype/shape 路由和 tiling 生成是否可以成功创建 executor，它们不被写入精度结论。

### 2. Oracle 设计

CPU 参考实现没有调用其他框架，而是在测试文件中直接实现 Cumsum：

- 先把 `dim` 归一化为合法 `axis`。
- 根据 shape 拆分 `outer / axisLen / inner`。
- 普通模式从 `axis=0` 到 `axisLen-1` 累加。
- reverse 模式从 `axisLen-1` 到 `0` 累加。
- exclusive 模式先写当前 sum，再把当前位置输入加入 sum。
- 标量 shape 单独处理，空 tensor 返回空输出。

浮点类型的参考值使用 double 累加，但输入值以实际存入 tensor 的值为准。FP16/BF16 的输入在 host 侧以 `uint16_t` 位模式保存，因此测试中实现了 `Float16ToFloat`、`FloatToFloat16`、`BFloat16ToFloat`、`FloatToBFloat16`，先把位模式解码为 float，再转 double 参与参考计算。这样可以避免把 FP16/BF16 的二进制位模式误当成整数参与计算。

比较函数统一输出：

```text
elements, max_abs_error, max_rel_error, max_error_index, actual, expected
```

浮点比较使用组合容差：

```cpp
fabs(actual - expected) <= atol + rtol * fabs(expected)
```

整数和精确对照场景使用严格匹配。对 NaN/Inf 的比较逻辑也做了特殊处理：只有同为 NaN，或同为相同符号的 Inf 时才认为匹配。

### 3. 用例分布

109 条记录按执行目的可以分为三类：

| 类型 | 数量 | 说明 |
| --- | ---: | --- |
| 端到端成功用例 | 64 | 完整执行 phase2、同步 stream、拷回输出并与 CPU oracle 比对 |
| workspace-only 路由探针 | 26 | 只检查 `GetWorkspaceSize` 能否成功创建 executor，用于覆盖 dtype/tiling 路由 |
| 预期失败用例 | 19 | 构造非法参数，验证返回码为非成功 |

按覆盖目标看，部分用例会同时属于多个类别，例如一个 int32 大 shape 用例既覆盖 dtype，又覆盖 tiling 分支。

| 覆盖方向 | 实际覆盖内容 | 代表用例 |
| --- | --- | --- |
| 基础 shape | 2D、3D、标量、空 tensor、负维度 | `std_float_dim0_matrix`、`std_float_negative_last_dim`、`std_scalar_float`、`std_empty_float` |
| V2 语义 | exclusive、reverse、exclusive+reverse | `v2_float_exclusive`、`v2_float_reverse`、`v2_float_exclusive_reverse_dim0` |
| 浮点 dtype | FP32、FP16、BF16，以及 FP16 输入转 FP32 输出 | `std_fp16_input_to_float_cast`、`v2_fp16_twoway_reverse`、`std_bf16_small` |
| 整数 dtype | INT32 端到端，INT16/INT64/INT8/UINT8 workspace 路由 | `int32_axis0`、`int32_td_r_axis_split`、`workspace_int64_aicpu_dispatch` |
| 复杂 dtype 路由 | COMPLEX64/COMPLEX128 workspace 路由 | `workspace_complex64_aicpu_dispatch`、`workspace_complex128_aicpu_dispatch_v2` |
| FP32 tiling | UB split、borrow R、borrow N、borrow M、core split、fold count | `tiling_float_n_borrow_r_borrow`、`tiling_float_r_oway_borrow_r_ub`、`workspace_float_judge_fold_count_one_r32n4` |
| INT32 tiling | right split、left split、axis split、block group | `int32_right_axis_split`、`int32_axis0_right_split`、`workspace_int32_bgc_left64_r16` |
| CumsumCube | 大 batch、最后一维累加、负维度 cube 路由 | `std_float_cube_route`、`std_float_cube_route_negative_dim` |
| 参数异常 | null、dim 越界、shape 不匹配、rank 超限、dtype 不支持 | `invalid_null_self`、`invalid_dim_out_of_range`、`invalid_shape_mismatch_v2`、`invalid_uint32_dtype` |
| 精度场景 | 小数长轴累加、大数小增量、次正规、FP16/BF16 对比 | `precision_cube_float_0p1`、`precision_cube_float_leading_large_increment_loss`、`precision_cube_bf16_0p1` |

大 shape tiling 用例中有一部分输入为 0。原因是这些用例的首要目标是稳定触发 tiling 代码路径，而不是制造非零精度误差。精度分析部分只引用带有明确数值含义的非零用例。

### 4. 重点场景设计说明

**基础语义场景。** 这部分用例先确认最基本的 API 行为，包括二维矩阵沿 dim=0 累加、最后一维负维度访问、标量 shape、空 tensor。标量和空 tensor 很容易被忽略，但它们能触发 shape size 为 1 或 0 时的边界处理逻辑。若这类场景没有单独覆盖，后续大 shape 用例即使通过，也不能说明边界 shape 是安全的。

**V2 语义组合场景。** `aclnnCumsumV2` 的风险不在 dtype 参数，而在 `exclusive` 和 `reverse` 的组合语义。测试中分别构造了 exclusive、reverse、exclusive+reverse，且覆盖 dim=1 和 dim=0 两种方向。这样可以检查 CPU oracle 中“先写 sum 还是先加当前元素”的顺序是否与 NPU 一致，也能覆盖 V2 入口下不同布尔参数组合的 op_api 分支。

**负维度归一化场景。** `std_float_negative_last_dim` 和 `std_float_cube_route_negative_dim` 用 `dim=-1` 访问最后一维。负维度归一化是 op_api 层常见出错点：如果只测正维度，无法确认 `dim + rank` 的处理是否正确，也无法覆盖负维度对应的校验分支。

**空 tensor 和标量场景。** `std_empty_float`、`v2_empty_float` 和 `std_scalar_float` 分别覆盖元素数为 0 和 rank 为 0 的情况。空 tensor 不应发生非法拷贝或 workspace 异常，标量 Cumsum 的结果应等于输入本身。这类用例规模很小，但对参数检查、shape 计算和 stride 生成都有价值。

**低精度 dtype 场景。** FP16/BF16 在 host 侧以 `uint16_t` 保存位模式，如果测试代码没有正确解码，很容易把位模式当整数处理。`std_fp16_input_to_float_cast` 覆盖 FP16 输入、FP32 输出的 dtype 转换；`v2_fp16_twoway_reverse` 覆盖 FP16 下 reverse 路径；`std_bf16_small` 覆盖 BF16 的基础执行路径。后续精度章节再单独分析 FP16/BF16 非零长轴累加。

**AiCPU 和非常规 dtype 路由场景。** 对 double、int16、int64、int8、uint8、complex64、complex128 等类型，本次采用 workspace-only 方式检查 `GetWorkspaceSize` 和 executor 创建。这些用例的目标不是比较数值输出，而是覆盖 op_api/L0 层 dtype 分发和 AiCPU 路由判断。把它们和端到端精度用例区分开，可以避免把只验证路由的用例误写成精度结论。

**FP32 tiling 场景。** `tiling_float_*` 系列围绕 `left/axis/right` 三个维度设计，覆盖 axis 较短、axis 较长、right 很大、需要 UB split、需要 borrow R/N/M、按核拆分等情况。比如 `{64,10000,1}` 偏向长 axis 拆分，`{1,10000,256}` 用于扩大 right 维，`{64,4,50000}` 用于触发 right 很大时的分支，`{1,200000,32}` 用于覆盖长轴加右维组合。它们的输入多数为零，是为了稳定验证 tiling 参数生成和 kernel 调度路径。

**INT32 tiling 场景。** INT32 有独立 tiling 实现，不能只靠 FP32 shape 推断覆盖。`int32_axis0` 覆盖 axis=0，`int32_negative_last_axis` 覆盖负维度，`int32_right_axis_split` 扩大 right 维，`int32_td_r_axis_split` 构造大规模 axis/right 组合，`workspace_int32_bgc_*` 系列用于触发 block group 相关分支。整数输出采用严格匹配。

**CumsumCube 路由场景。** `std_float_cube_route` 和 `std_float_cube_route_negative_dim` 使用 `[12800,512]` 的大 batch + 最后一维累加，触发 CumsumCube 相关路径。精度场景也集中放在这个 shape 上，是因为它在当前真机环境下稳定、可重复，并且能观察长轴累加误差。

**异常参数场景。** 预期失败用例覆盖 null self、null out、dim 正向越界、dim 负向越界、rank 大于 8、shape 不匹配、dtype 参数和输出 dtype 不匹配、不支持的 uint/bool dtype 等。异常用例的判定不是具体错误码必须完全一致，而是返回值必须非成功；这样既能覆盖校验分支，又避免把测试绑定到某个内部错误码实现。

**精度风险场景。** 非零精度用例没有随机生成，而是选择可解释的输入：`0.1f` 暴露十进制小数的二进制量化误差，`0.125f` 作为精确小数对照，`1e8f` 后跟 `1.0f` 暴露大数吞小数，次正规数输入观察归零行为，FP16/BF16 对比低精度累加误差。这些场景不追求数量堆叠，而是保证每个误差都能对应到明确的浮点原因。

------

## 三、覆盖率分析

### 1. 测量方法

覆盖率在真机环境下通过工程脚本生成。实际运行命令如下：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./build_out/cann-ops-math-custom_linux-aarch64.run
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0-beta.2/opp/vendors/custom_math/op_api/lib/:${LD_LIBRARY_PATH}
bash build.sh --run_example cumsum eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

运行日志中的测试结论：

```text
Summary: 109 passed, 0 failed
```

### 2. 评分相关覆盖率文件

本次提交包中保留以下覆盖率文件：

```text
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum/op_api/aclnn_cumsum.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum/op_api/aclnn_cumsum.cpp.gcno
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum/op_api/cumsum.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum/op_api/cumsum.cpp.gcno
build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling.cpp.gcda
build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling.cpp.gcno
build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling_ascendc_arch35.cpp.gcda
build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling_ascendc_arch35.cpp.gcno
build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp.gcda
build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp.gcno
```

对应源文件的 gcov 统计结果如下：

| 文件 | 覆盖行数/总代码行 | 行覆盖率 | 覆盖分支/总分支 | 分支覆盖率 | 覆盖情况 |
| --- | ---: | ---: | ---: | ---: | --- |
| `math/cumsum/op_api/aclnn_cumsum.cpp` | 126/130 | 96.92% | 462/648 | 71.30% | 两个 ACLNN 入口、参数校验、dtype/shape 检查、cube/AiCore/AiCPU 调度均有覆盖 |
| `math/cumsum/op_api/cumsum.cpp` | 28/35 | 80.00% | 46/86 | 53.49% | L0 层 dtype 判断和执行路由有覆盖，部分异常分支未触发 |
| `math/cumsum/op_host/arch35/cumsum_tiling.cpp` | 30/30 | 100.00% | 42/76 | 55.26% | tiling 总入口全部行覆盖，分支主要受平台和宏分支影响 |
| `math/cumsum/op_host/arch35/cumsum_tiling_ascendc_arch35.cpp` | 673/684 | 98.39% | 331/401 | 82.54% | FP32/FP16/BF16 tiling 主体覆盖较充分 |
| `math/cumsum/op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp` | 243/249 | 97.59% | 290/360 | 80.56% | INT32 tiling 主体覆盖较充分 |

按源码行数和分支数加权后：

```text
行覆盖率:   1100 / 1128 = 97.52%
分支覆盖率: 1171 / 1571 = 74.54%
```

### 3. 相关文件观察

除提交包中的文件外，运行过程中还观察到以下相关源文件覆盖情况：

| 文件 | 行覆盖率 | 分支覆盖率 | 说明 |
| --- | ---: | ---: | --- |
| `math/cumsum/op_host/cumsum_def.cpp` | 100.00% of 30 | 100.00% of 80 | 算子定义生成侧，已被构建流程覆盖 |
| `math/cumsum_cube/op_host/op_api/cumsum_cube.cpp` | 80.00% of 10 | 45.45% of 44 | CumsumCube L0 封装路径有触发，但分支数量少且宏分支较多 |
| `math/cumsum/op_graph/cumsum_graph_infer.cpp` | 0.00% of 3 | 0.00% of 2 | GE graph infer 路径，ACLNN eager example 不会进入 |

如果把这些参考文件也纳入观察，总体为：

```text
行覆盖率:   1138 / 1171 = 97.18%
分支覆盖率: 1271 / 1697 = 74.90%
```

### 4. 未覆盖部分分析

未覆盖行主要集中在少量兜底路径和 graph 路径；未覆盖分支更多，主要原因如下：

1. op_api 中有大量由宏展开产生的失败分支，例如空 executor、内部 tensor 分配失败、launcher 加入失败等。正常输入可以覆盖成功路径和参数校验失败路径，但不适合通过破坏运行环境来制造这些内部失败。
2. `cumsum.cpp` 的低覆盖分支集中在 L0 路由和异常返回处。workspace-only 用例已经覆盖了 double、int16、int64、int8、uint8、complex 等路由探针，但部分 phase2 执行路径在当前环境下并不稳定，因此没有把它们写成端到端精度用例。
3. tiling 文件的行覆盖较高，分支覆盖相对低，主要是因为 tiling 内部有较多平台条件、边界判断和模板分发分支。通过构造多组 `left/axis/right` shape 已覆盖主要业务路径，但不能在单一 SoC 上覆盖所有架构判断。
4. `cumsum_graph_infer.cpp` 属于 GE graph 插件路径，本次测试入口是 ACLNN eager example，因此不会触发该文件。

------

## 四、精度分析

### 1. 误差度量与阈值

浮点用例采用最大绝对误差和最大相对误差两个指标：

```text
abs_error = abs(actual - expected)
rel_error = abs(actual - expected) / abs(expected)
```

当 expected 接近 0 时，相对误差会失去直观意义，因此实际判定使用组合容差：

```cpp
abs(actual - expected) <= atol + rtol * abs(expected)
```

不同 dtype 使用不同容差。FP32 常规小 shape 使用 `1e-5` 量级容差；长轴小数累加场景按累加长度放宽绝对容差；FP16/BF16 因尾数位更少，使用更宽的容差。整数用例使用精确匹配。

下面只列出最终稳定运行日志中的非零精度场景和关键对照场景。

### 2. 场景一：`0.1f` 长轴累加

测试输入：

```text
shape = [12800, 512]
dim = 1
input = 全部 0.1f
api = aclnnCumsum
```

实测最大误差点：

```text
用例: precision_cube_float_0p1
elements:      6553600
index:         510
actual:        51.0999679565
expected:      51.1000007614
max_abs_error: 3.28049063683e-05
max_rel_error: 6.41974674744e-07
```

`0.1f` 在二进制浮点中不能精确表示，输入进入 NPU 前已经有量化误差。Cumsum 又沿轴连续加 512 次，因此误差会随累加长度传播。当前结果的相对误差约为 `6.42e-7`，对 FP32 长轴累加来说处于可解释范围内。

为了观察轴长影响，又构造了 `axisLen=1024` 的用例：

```text
用例: precision_cube_float_0p1_len1024
elements:      13107200
index:         1019
actual:        101.999938965
expected:      102.00000152
max_abs_error: 6.25550746918e-05
max_rel_error: 6.13285036859e-07
```

轴长翻倍后，最大绝对误差也接近翻倍；相对误差仍在 `1e-6` 内。这说明误差主要来自连续累加的舍入传播，而不是某个固定位置的功能错误。

### 3. 场景二：二进制可精确小数对照

测试输入：

```text
shape = [12800, 512]
dim = 1
input = 全部 0.125f
```

实测结果：

```text
用例: precision_cube_float_exact_pow2
actual:        0.125
expected:      0.125
max_abs_error: 0
max_rel_error: 0
```

`0.125 = 1/8` 可以被二进制浮点精确表示。这个对照用例说明，同样的 shape、同样的 CumsumCube 路由下，精确小数不会出现 `0.1f` 那样的量化误差传播。因此 `0.1f` 场景的误差来源不是 shape 或路由错误，而是浮点表示和连续加法共同作用。

### 4. 场景三：正负交替抵消

测试输入为：

```text
input = [1, -1, 1, -1, ...]
shape = [12800, 512]
dim = 1
```

实测结果：

```text
用例: precision_cube_float_alternating_cancel
actual:        1
expected:      1
max_abs_error: 0
max_rel_error: 0
```

该模式下前缀和只在 `1` 和 `0` 之间切换，两个值都能被 FP32 精确表示，因此没有观察到误差。它作为符号交替的基础对照，说明 Cumsum 对简单正负抵消语义处理正确。

### 5. 场景四：大数后小增量丢失

测试输入为每条轴第一个元素是 `1e8f`，后续元素为 `1.0f`：

```text
input = [1e8, 1, 1, 1, ...]
shape = [12800, 512]
dim = 1
```

实测最大误差点：

```text
用例: precision_cube_float_leading_large_increment_loss
elements:      6553600
index:         259
actual:        100000000
expected:      100000259
max_abs_error: 259
max_rel_error: 2.58999329192e-06
```

FP32 在 `1e8` 附近的相邻可表示间隔已经大于 1。此时再累加 `1.0f`，很多小增量低于当前数值量级的分辨率，实际输出会长时间停留在 `100000000`。这个场景是 Cumsum 比较典型的精度风险：输入本身没有越界，也不是 NaN/Inf，但累加顺序会让小量贡献丢失。

### 6. 场景五：大数、小数和抵消混合

测试输入模式为：

```text
input = [1e8, 1, -1e8, 0, ...]
shape = [12800, 512]
dim = 1
```

实测最大误差点：

```text
用例: precision_cube_float_large_small_cancel
elements:      6553600
index:         13
actual:        100000000
expected:      100000004
max_abs_error: 4
max_rel_error: 3.99999984e-08
```

这个场景和单纯正负交替不同。小增量夹在大数附近时可能被舍入吞掉，后续即使出现抵消项，也不能恢复已经丢失的小量信息。它验证了 Cumsum 对量级混合输入的误差表现。

### 7. 场景六：次正规极小值累加

测试输入为 FP32 最小次正规正数：

```text
input = std::numeric_limits<float>::denorm_min()
shape = [12800, 512]
dim = 1
```

实测最大误差点：

```text
用例: precision_cube_float_subnormal_accum
elements:      6553600
index:         511
actual:        0
expected:      7.17464813734e-43
max_abs_error: 7.17464813734e-43
max_rel_error: 1
```

该场景的绝对误差极小，但相对误差为 1。实测输出为 0，说明该路径对极小次正规输入表现为归零。对普通业务来说，`1e-43` 量级的绝对差异通常不敏感；但如果模型或算法依赖极小梯度、概率尾部或残差项累积，这类行为会影响可解释性。

### 8. 场景七：FP16 与 BF16 累加

FP16 输入为 `FloatToFloat16(0.1f)`：

```text
用例: precision_cube_fp16_0p1
elements:      6553600
index:         383
actual:        38.375
expected:      38.390625
max_abs_error: 0.015625
max_rel_error: 0.000407000407
```

BF16 输入为 `FloatToBFloat16(0.1f)`：

```text
用例: precision_cube_bf16_0p1
elements:      6553600
index:         320
actual:        32.25
expected:      32.1313476562
max_abs_error: 0.11865234375
max_rel_error: 0.00369272851607
```

FP16 和 BF16 的误差都明显高于 FP32。BF16 的指数范围较大，但尾数只有 7 位；FP16 尾数位多一些，但动态范围更小。本轮数据中，BF16 的最大相对误差约 `3.69e-3`，高于 FP16 的 `4.07e-4`，符合低精度 dtype 的预期。

作为低精度可精确小数对照，FP16 的 `0.125` 用例结果为：

```text
用例: precision_cube_fp16_exact_pow2
max_abs_error: 0
max_rel_error: 0
```

这说明低精度 dtype 的误差并不是所有输入都会出现，输入是否可精确表示、累加长度和输出 dtype 都会共同影响结果。

### 9. 精度小结

本次最终稳定用例中，精度表现可以概括为：

| 场景 | dtype | 最大绝对误差 | 最大相对误差 | 结论 |
| --- | --- | ---: | ---: | --- |
| `0.1f` 长轴 512 | FP32 | `3.28049063683e-05` | `6.41974674744e-07` | 长轴累加误差可见，但相对误差较小 |
| `0.1f` 长轴 1024 | FP32 | `6.25550746918e-05` | `6.13285036859e-07` | 绝对误差随轴长增加 |
| `0.125f` 长轴 | FP32 | `0` | `0` | 可精确表示小数作为对照稳定 |
| 大数后小增量 | FP32 | `259` | `2.58999329192e-06` | 小增量被大数附近 ULP 吞掉 |
| 次正规极小值 | FP32 | `7.17464813734e-43` | `1` | 输出归零，绝对误差极小但相对误差为 1 |
| `0.1` 长轴 | FP16 | `0.015625` | `0.000407000407` | 低精度累加误差明显 |
| `0.1` 长轴 | BF16 | `0.11865234375` | `0.00369272851607` | BF16 尾数少，误差高于 FP16 |

------

## 五、反思与改进

### 1. 覆盖率和精度目标需要分开处理

Cumsum 的覆盖率提升主要依赖 shape 设计，精度分析主要依赖数值设计，这两件事不完全一致。为了覆盖 tiling，需要构造很多特殊 shape，例如 `right` 很大、`axis` 很长、`left` 较大、UB 需要拆分、block group 需要切换等。如果这些大 shape 全部填充复杂非零数据，单次运行会更慢，内存压力也会更大，还会把覆盖率问题和精度问题混在一起，出现失败后很难定位。

本次采用的做法是把两类目标拆开：覆盖率用例优先保证稳定触发路径，很多大 shape 使用零输入；精度用例集中在 CumsumCube 路径上，用少数明确的非零输入解释误差来源。这个取舍让覆盖率和精度分析都比较可复核，但代价是部分 tiling 路径虽然执行过，尚未在非零数据下做精度压力测试。

### 2. workspace-only 用例的定位要写清楚

workspace-only 用例容易被误解。本次对 double、int16、int64、int8、uint8、complex 等 dtype 使用 workspace-only，不是为了证明这些 dtype 的输出精度，而是为了覆盖 `GetWorkspaceSize`、executor 创建和路由判断。报告中必须明确这一点，否则评审会认为这些 dtype 没有真正端到端验证。

这类用例的价值在于它们确实能覆盖 op_api 和 L0 层的一部分 dtype 分发分支，尤其是 AiCPU 路由和非常规 dtype 判断。它们的局限也很明确：没有 phase2 执行、没有 device 输出、没有 CPU oracle 比对，因此不能计入精度结论。

### 3. Oracle 是 Cumsum 测试里最容易写错的部分

Cumsum 的 CPU 参考不能简单复用加法逻辑。普通、exclusive、reverse 三种语义差别只在“当前元素先加入 sum 还是后加入 sum”，代码上只差几行，但结果会完全不同。负维度、空 tensor、标量 shape 也需要单独处理，否则 oracle 自身会变成错误来源。

浮点 oracle 也需要注意基准一致性。NPU 输入已经是 FP32、FP16 或 BF16 的量化值，CPU 参考应基于这个量化后的输入继续计算，而不是基于十进制字面量的数学真值计算。FP16/BF16 更容易出错，因为 host 侧保存的是 `uint16_t` 位模式；如果直接把 `uint16_t` 当成数值累加，结果会和真实 FP16/BF16 完全不一致。本次显式实现了 FP16/BF16 解码，并统一用 double 做参考累加。

### 4. 分支覆盖率低于行覆盖率的原因

本次评分文件行覆盖率为 `97.52%`，分支覆盖率为 `74.54%`。这个差距不是单纯少了基础功能用例，而是 CANN 代码中存在很多正常输入难以触发的分支。比如 op_api 里有大量宏展开的失败路径：内部 tensor 申请失败、executor 创建失败、launcher 添加失败、日志和异常返回分支等。这些分支在源码层面会计入分支总数，但不能靠普通合法输入稳定触发。

tiling 文件也有类似情况。`cumsum_tiling_ascendc_arch35.cpp` 和 `cumsum_tiling_ascendc_int_arch35.cpp` 的行覆盖已经分别达到 `98.39%` 和 `97.59%`，说明主体路径已经执行；分支覆盖仍在 80% 左右，是因为内部有平台判断、边界兜底、模板分发和极端 shape 条件。单一 `ascend910_93` 真机环境下，架构相关分支本身就不可能全部同时覆盖。

### 5. graph 路径是当前入口的天然盲区

`math/cumsum/op_graph/cumsum_graph_infer.cpp` 覆盖率为 0，这不是因为 Cumsum 基础用例没有测，而是因为当前测试文件走的是 ACLNN eager example。graph infer 属于 GE graph 路径，需要不同的测试入口才能触发。把 graph 用例硬塞进 ACLNN 测试文件并不自然，也会影响提交文件的可维护性。

如果后续要继续拉高总分支覆盖率，graph 路径应单独建设测试，而不是在 eager example 里混合两套执行模型。这样报告和覆盖率数据也会更清楚：ACLNN 入口覆盖 ACLNN 逻辑，GE 入口覆盖 graph infer 逻辑。

### 6. 精度场景还可以继续扩展

当前精度场景覆盖了 `0.1f` 长轴累加、可精确小数对照、大数吞小增量、量级混合抵消、次正规数归零、FP16/BF16 对比。这些场景能说明 Cumsum 的主要精度风险，但还不是完整的浮点边界集合。

后续可以继续增加几类输入。第一类是接近 FP32 最大有限值的累加，用于观察从有限值到 Inf 的边界。第二类是包含 `Inf`、`-Inf`、`NaN` 的输入，用于验证特殊值传播，例如 `Inf + (-Inf)` 是否产生 NaN。第三类是随机正负混合长轴输入，用于观察误差随抵消程度变化的规律。第四类是阶梯量级输入，例如 `1e-6, 1e-5, ..., 1e6`，用于观察累加顺序对小量保留的影响。第五类是整数溢出边界，尤其是 INT32 前缀和超过 `INT32_MAX` 后的行为。

这些输入没有写进当前最终结论，是因为本次提交优先保留已经稳定通过、日志完整、误差能解释清楚的用例。精度报告宁可少写一点，也不能把没有最终运行确认的数据写成实测结论。

### 7. 非零大 shape 的覆盖仍有提升空间

目前大 shape tiling 用例中很多使用零输入，优点是稳定、快、结果容易严格匹配；缺点是不能暴露所有 kernel 数值问题。比如 UB split、borrow R、borrow N、core split 等路径，如果只看零输入，无法发现跨块累加时是否存在边界位置误差。

更理想的做法是分阶段替换：先把部分零输入改成简单等差数列或周期序列，检查跨 tile 边界结果；再对少数关键 shape 增加非零随机输入；最后才加入更强的精度压力输入。这样既不会一次性引入太多不稳定因素，也能逐步把“路径覆盖”推进到“路径 + 数值”双验证。

### 8. dtype 覆盖还可以从路由走向端到端

本次 INT32 已经有端到端输出校验，但 int16、int64、int8、uint8、complex 等类型主要停留在 workspace 路由层面。后续如果执行路径稳定，应把这些 dtype 扩展成真正的 phase2 用例：准备非零输入、执行、拷回、按 dtype 特性建立 oracle。

整数类型还要特别关注溢出。C++ 有符号整数溢出本身不是可靠 oracle，因此不能直接用 `int32_t` 在 CPU 上溢出计算再比较。更稳妥的做法是用更宽类型或无符号回绕明确模拟预期行为，再与 NPU 输出比对。

### 9. 报告数据需要和日志一一对应

本次报告只引用最终运行日志中能找到的数字，例如 `Summary: 109 passed, 0 failed`、每个精度用例的最大误差、覆盖率文件的行/分支统计。没有把尝试过但未进入最终稳定集合的结果写成提交结论。这样做虽然会让报告少一些“看起来更丰富”的内容，但可复核性更强。

后续整理报告时，最好保留三类证据：测试源文件中的用例名、运行日志中的 PASS 和误差输出、gcov 中的行/分支统计。三者能互相对应，评审复查时更容易确认报告不是泛泛描述。

------

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "我也要死吗队"

team_members:

- "成员1：贾皓文-中国科学院大学"

operator_name: "Cumsum"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# Cumsum 算子测试报告

> 测试环境来自本次 `cumsum_example.log`：`COMPUTE_UNIT=ascend910_93`，CANN 安装路径 `/usr/local/Ascend/cann-9.0.0-beta.2`，目标架构 `aarch64`，C/C++ 编译器为 GNU 11.4.0，编译参数包含 `-DENABLE_UT_EXEC=TRUE -DENABLE_COVERAGE=TRUE -DCMAKE_BUILD_TYPE=Release -DASCEND_COMPUTE_UNIT=ascend910_93`。覆盖率来源为本次上传的 4 个 `.gcov` 文件。

本报告参照测试报告模板的五段式结构与 Mul 样例报告的写法，重点突出三类信息：**算子语义与风险点、测试设计与 Oracle 选择、覆盖率与失败根因分析**。本次最重要的结论是：测试程序本身覆盖面较完整，但实测 42 个用例中仅 22 个通过、20 个失败；失败高度集中在非空 Float/Float16/BFloat16/Int32 的 AiCore 正向计算路径，典型现象是输出张量保持全 0。该现象不是普通浮点舍入误差，而是功能正确性或运行环境/tiling/bin 选择层面的严重风险。

------

## 一、算子理解

Cumsum 是一元前缀和算子，给定输入张量 `self`、累加轴 `dim`，输出张量 `out` 与输入张量形状一致。对 shape 为 `[d0, d1, ..., dn-1]` 的张量，若归一化后的轴为 `axis`，则每个输出元素只沿 `axis` 方向依赖同一切片上位于当前元素之前的元素：

```text
out[..., k, ...] = self[..., 0, ...] + self[..., 1, ...] + ... + self[..., k, ...]
```

`aclnnCumsum` 的核心语义是 inclusive forward cumsum；`aclnnCumsumV2` 进一步引入 `exclusive` 与 `reverse` 两个布尔属性：

| exclusive | reverse | 语义 |
|---|---|---|
| false | false | 正向包含当前元素的前缀和 |
| true | false | 正向不包含当前元素的前缀和，第一个位置为 0 |
| false | true | 反向包含当前元素的后缀和 |
| true | true | 反向不包含当前元素的后缀和，最后一个位置为 0 |

与 Mul 这类逐元素算子不同，Cumsum 的输出元素存在明显的顺序依赖：轴上第 `k` 个元素依赖此前多个元素。因此，Cumsum 的精度风险不是单次运算误差，而是**累加误差传播**。这带来三个典型特性：

1. **非结合性明显**：浮点加法不满足严格结合律，硬件实现若采用分块、并行前缀和或 Sklansky scan，理论上可能与 CPU 串行 double 参考存在低位差异。
2. **误差随轴长增长**：对于 FP32，长轴上反复累加 0.1f 等无法精确表示的小数，会出现可观的累计误差；容差应随轴长与数据量级放宽。
3. **量级混合敏感**：大量级值与小量级值交替时，小量级增量可能被大值的 ULP 吞没，导致数学真值与实际浮点结果产生差异。

在 dtype 方面，本次代码覆盖了 `ACL_FLOAT`、`ACL_FLOAT16`、`ACL_BF16`、`ACL_DOUBLE`、`ACL_INT32`、`ACL_INT64`、`ACL_INT8`、`ACL_UINT8`，并专门设计了 `INT8 -> INT32`、`FLOAT -> FLOAT16` 等输出 dtype 转换场景。根据 `cumsum.cpp` 的调度逻辑，在当前平台上 Float/Float16/BFloat16/Int32 主要进入 AiCore 分支，Double/Int64/Int8/UInt8 等更容易走 AiCpu 分支。该差异对于解释测试结果非常关键：本次通过用例多集中在 AiCpu 或期望本身为 0 的边界场景，而失败用例集中在非空 AiCore 场景。

------

## 二、测试策略与用例设计

### 2.1 总体策略

本次测试代码 `test_aclnn_cumsum.cpp` 按照“功能正确性 + 精度风险 + dtype 覆盖 + V2 属性组合 + 异常路径”的思路组织，共 42 个用例。测试入口统一封装了 ACL 初始化、张量创建、workspace 获取、算子执行、同步、拷回与结果校验，核心封装包括：

- `RunCumsum`：调用 `aclnnCumsumGetWorkspaceSize` 与 `aclnnCumsum`。
- `RunCumsumV2`：调用 `aclnnCumsumV2GetWorkspaceSize` 与 `aclnnCumsumV2`。
- `RunCumsumCase`：普通 Cumsum 正向用例模板。
- `RunCumsumV2Case`：带 `exclusive/reverse` 的 V2 用例模板。
- `RunNegativeStatusCase`：异常路径用例模板，期望 workspace 接口返回非 `ACL_SUCCESS`。

测试程序的 `main` 函数按固定顺序执行 42 个函数指针，并最终输出 `Total/Pass/Fail` 汇总。这种组织方式的优点是：每个用例都有独立名称、独立 shape、独立 dtype 与独立容差，失败日志可直接定位到测试场景。

### 2.2 Oracle 选择

CPU Oracle 由 `ComputeCumsum` 实现。其核心思路是先把输入转为 `double`，再按 `shape + dim + exclusive + reverse` 在 CPU 上串行计算期望结果。该实现显式处理了三类边界：

1. **空张量**：输入为空时直接返回空 expected。
2. **0D scalar**：shape 为空时，inclusive 返回输入自身，exclusive 返回 0。
3. **负轴**：`NormalizeDim` 将负数 dim 转换为非负轴。

浮点校验采用 `abs(actual - expected) <= atol + rtol * abs(expected)`。整数校验采用精确匹配。FP16 与 BF16 的特殊点在于，测试数据在 host 侧以 `uint16_t` 保存位模式，不能直接把 `uint16_t` 当整数累加；代码通过 `Fp16BitsToFloat`、`Bf16BitsToFloat` 先解码为 float，再转 double 作为 Oracle 输入。这一点与 Mul 样例报告中的经验一致：低精度 dtype 的 Oracle 必须先正确解释位模式，否则会把“参考实现错误”误判为“算子错误”。

### 2.3 用例分类

| 类别 | 用例数 | 设计目的 | 代表用例 |
|---|---:|---|---|
| Cumsum v1 正向功能/精度 | 18 | 覆盖基础轴、负轴、长轴、混合量级、3D、大 shape、dtype、0D/empty | `Cumsum_Float32_Dim0_Mixed`, `Cumsum_Float32_Long_0p1_ErrorAccum`, `Cumsum_Float32_CubeCandidate_LastDim` |
| CumsumV2 正向功能/精度 | 11 | 覆盖 exclusive/reverse 组合、负轴、大 tiling reverse、V2 dtype | `CumsumV2_Float32_Exclusive_Reverse`, `CumsumV2_Int64_Exclusive_AiCpuProbe` |
| 异常路径 | 13 | 覆盖 nullptr、dtype mismatch、unsupported bool、shape mismatch、dim 越界、rank 过大 | `NEG_Cumsum_SelfNull`, `NEG_CumsumV2_DimOutOfRange` |
| dtype 覆盖 | 多个交叉用例 | 验证 Float/FP16/BF16/Double/Int32/Int64/Int8/UInt8，以及 cast 行为 | `Cumsum_Float32_To_Float16_Cast`, `Cumsum_Int8_To_Int32_Cast` |
| 边界输入 | 多个交叉用例 | 验证 scalar 与 empty tensor | `Cumsum_Float32_0DScalar`, `Cumsum_Float32_EmptyTensor` |

更细的用例覆盖如下：

| 场景 | 输入与 shape | 期望覆盖点 | 实测结论 |
|---|---|---|---|
| 基础 FP32 | {2,3}, dim=0 / dim=-1 | 期望分别为按列/按行累加 | 均 FAIL，actual_samples 全 0 |
| 长轴 0.1 | {10000}, dim=0，输入全 0.1f | 末尾约 1000.00001，用于观察误差累积 | FAIL，实际全 0，首元素即不匹配 |
| 混合量级 | {2048}, 1e6 与 0.25 交替 | 观察小量在大量累加中被吞噬的风险 | FAIL，实际全 0，不属于正常舍入误差 |
| 3D 中轴/大 N | {65,17,2} / {4,1024,16} | 验证中间轴与 tiling 形状 | FAIL，实际全 0 |
| CubeCandidate | {12800,512}, last dim，输入全 1 | 覆盖大 shape、last dim 累加 | PASS，样例 [1,2,3,...,510,511,512] |
| AiCpu Probe | DOUBLE / INT64 | 触发非 AiCore 路径 | PASS |
| 低位整数 | INT8 / UINT8 小范围 | 避免溢出，精确匹配 | PASS |
| FP16/BF16 | 位模式编码输入，解码校验 | 验证低精度累计误差 | FAIL，实际全 0 |
| V2 属性组合 | inclusive/exclusive × forward/reverse | 验证 exclusive/reverse 语义 | 多数 FAIL，INT64/0D exclusive/empty 通过 |
| 异常输入 | nullptr、dtype/shape/dim/rank 错误 | 期望 workspace 接口返回非 ACL_SUCCESS | 全部 PASS，状态码 161001/161002 |


### 2.4 精度阈值设置

阈值设置不是“一刀切”，而是按 dtype 与场景调整：

- FP32 基础短轴：`atol=1e-5, rtol=1e-5`。短轴累加误差应较小，严格阈值可暴露功能错误。
- FP32 长轴 0.1：`atol=2e-3, rtol=2e-6`。由于 0.1f 本身不可精确表示，且累加 10000 次，允许毫级绝对误差。
- FP32 混合量级：`atol=5e-1, rtol=1e-6`。大值累加会提高 ULP，小值 0.25 可能产生局部损失，因此放宽绝对误差。
- FP16：`atol=2e-2, rtol=2e-3`。FP16 尾数仅 10 位，累加误差明显高于 FP32。
- BF16：`atol=5e-2, rtol=5e-3`。BF16 尾数仅 7 位，精度低于 FP16，但动态范围更接近 FP32。
- Double：`atol=1e-10, rtol=1e-10`。
- 整数：精确匹配。

需要强调：这些阈值足以覆盖正常浮点误差，但本次多数失败用例实际输出为全 0，误差远超阈值，不能归因于浮点精度阈值过严。

------

## 三、覆盖率分析

### 3.1 统计口径

本次覆盖率根据上传的 `.gcov` 原始文件重新统计。行覆盖率按可执行代码行统计；分支覆盖率按 gcov 中每条 `branch` 记录统计，`taken > 0` 视为覆盖；函数调用覆盖率按 gcov 中每条 `call` 记录统计，`returned > 0` 视为覆盖。综合覆盖率采用按行数/分支数/调用数加权，而不是对文件百分比做算术平均。

### 3.2 覆盖率结果

| 文件 | 可执行行 | 行覆盖率 | 分支覆盖率 | 调用覆盖率 | 说明 |
|---|---:|---:|---:|---:|---|
| `math/cumsum/op_api/cumsum.cpp` | 35 | 28/35 = 80.0% | 28/86 = 32.6% | 39/67 = 58.2% | API/L0 调度层：按 dtype/平台选择 AiCore 或 AiCpu，并注册 launcher |
| `math/cumsum/op_host/arch35/cumsum_tiling.cpp` | 30 | 30/30 = 100.0% | 27/76 = 35.5% | 11/40 = 27.5% | Host tiling 入口：Cumsum tiling 与 parse prepare 分发 |
| `math/cumsum/op_host/arch35/cumsum_tiling_ascendc_arch35.cpp` | 684 | 402/684 = 58.8% | 144/401 = 35.9% | 120/249 = 48.2% | AscendC 浮点/通用 tiling 策略：shape 拆分、UB/Core 参数、Sklansky 分组 |
| `math/cumsum/op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp` | 249 | 149/249 = 59.8% | 98/360 = 27.2% | 61/134 = 45.5% | AscendC Int tiling 策略：整型 cumsum 的 shape 调整与 tiling key/data |


综合覆盖率：

```text
行覆盖率   = 609/998 = 61.0%
分支覆盖率 = 297/923 = 32.2%
调用覆盖率 = 231/490 = 47.1%
```

### 3.3 覆盖率解读

`cumsum.cpp` 行覆盖率达到 80.0%，说明 API/L0 调度主干路径基本被触达。gcov 函数统计显示 `IsAiCoreSupport` 被调用 26 次，`CumsumAiCore` 被调用 21 次，`CumsumAiCpu` 被调用 5 次；普通 `Cumsum` 入口被调用 16 次，带 `exclusive/reverse` 的重载被调用 10 次。这说明测试确实覆盖了两个公开 API 入口及 AiCore/AiCpu 两类后端分发。

但 `cumsum.cpp` 的分支覆盖率只有 32.6%。主要原因有三类：

1. 当前环境固定为 `ascend910_93`，平台判断只走了 `DAV_2201` 分支，`IsRegBase` 与 910 旧分支未覆盖。
2. `AllocTensor` 失败、launcher 添加失败等内部错误路径没有被故障注入触发。
3. 宏展开产生了大量异常/日志/throw 分支，正常用例不会触达。

`cumsum_tiling.cpp` 行覆盖率为 100.0%，说明 tiling 入口函数全部执行过；但分支覆盖率仅 35.5%，说明入口行被执行并不代表分发组合充分。未覆盖部分主要是 prepare 失败、平台/类型不匹配、异常返回等分支。

`cumsum_tiling_ascendc_arch35.cpp` 行覆盖率为 58.8%，分支覆盖率为 35.9%。从函数级覆盖看，`NLesserCl`、`RNLesserCl`、`DoBlockSplit`、`DoUbSplit`、`FillTilingData` 等主干被触发；但 `NGreaterCl`、`NGreaterClRFullLoad`、`NGreaterClRNotFullLoad`、`CalcBorrowM`、`UbFullLoad`、`MaxForFullUb`、`CalcXBufSize`、`CalcUnfoldN` 等函数为 0 次调用。这说明当前 shape 组合更多落在 “N 较小 / R、M 局部变化” 的策略区间，没有覆盖 N 很大、UB full load、BorrowM/BorrowR 等关键 tiling 边界。

`cumsum_tiling_ascendc_int_arch35.cpp` 行覆盖率为 59.8%，分支覆盖率为 27.2%。该文件只被少量 int32 AiCore 用例触发，`AdjustTensor4TDRA`、`AdjustTensor4TDLA`、`CheckBGC`、`AdjustLARLpUnit`、`CalcAxisWeight` 等函数未覆盖。说明整型 tiling 的方向调整、axis weight、B/G/C 组合校验等复杂路径仍有盲区。

### 3.4 覆盖率与失败现象的关系

本次覆盖率并不低，尤其是 tiling 入口与部分 AscendC tiling 主干已经被触达；但 20 个失败用例表明，覆盖率提升并不等价于正确性提升。当前最值得关注的不是“还有多少代码没跑到”，而是“已经跑到的 AiCore 路径在大量 shape/dtype 下输出全 0”。这类失败对测试报告有两点启发：

- 分支覆盖率应与结果分布一起看。若某条 tiling 策略被覆盖但输出错误，继续堆叠类似 shape 不能提升测试质量，应该先定位该策略的正确性问题。
- 需要把通过率按后端路径拆分。AiCpu 路径和 empty/scalar 特例通过，不能抵消 Float/FP16/BF16/Int32 AiCore 非空主路径的系统性失败。

------

## 四、精度分析

### 4.1 总体结果

本次运行日志汇总如下：

| 类别 | 总数 | 通过 | 失败 | 备注 |
|---|---:|---:|---:|---|
| Cumsum v1 正向用例 | 18 | 6 | 12 | 覆盖基础维度、负轴、长轴累加、混合量级、3D 中轴、大 N tiling、CubeCandidate、DOUBLE/FP16/BF16/INT/标量/空张量 |
| CumsumV2 正向用例 | 11 | 3 | 8 | 覆盖 inclusive/exclusive、forward/reverse、负轴、大 tiling reverse、FP16/INT32/INT64、0D/empty |
| Cumsum/CumsumV2 异常用例 | 13 | 13 | 0 | 覆盖 self=null、out=null、dtype mismatch、bool unsupported、shape mismatch、dim 越界、rank 过大 |
| 合计 | 42 | 22 | 20 | 通过率 52.38%，失败集中在非空 AiCore 正向路径 |


由于 20 个失败用例中大多数在首元素即发生 `actual=0`、`expected!=0`，当前无法对这些场景给出“算子数值精度合格/不合格”的普通结论。准确表述应为：**AiCore 非空正向路径存在功能正确性风险；在该问题解决前，相关 dtype 的精度分析不能作为有效精度评估。**

### 4.2 场景一：基础 FP32 累加

`Cumsum_Float32_Dim0_Mixed` 的输入为 `(1.0, -2.0, 3.5, 4.0, 0.5, -1.0)`，shape `(2, 3)`，dim=0。CPU Oracle 的 expected samples 为 `[1, -2, 3.5, 5, -1.5, 2.5]`，但实测 actual samples 为 `[0, 0, 0, 0, 0, 0]`，首元素即失败。

`Cumsum_Float32_NegativeDim` 使用 shape `(2, 3)`、dim=-1，期望 `[1, 3, 6, -1, -3, -6]`，实测同样全 0。两个基础用例都失败，说明问题不是某个复杂 tiling shape 独有的边界，而是常规 Float32 AiCore 小 shape 路径就存在风险。

### 4.3 场景二：长轴 0.1f 累加误差

该用例输入为 10000 个 `0.1f`，理论上用于观察 0.1f 无法精确表示导致的累加误差。Oracle 以已量化的 `0.1f` 为输入，最后几个 expected samples 约为 `[999.800015, 999.900015, 1000.00001]`。如果算子功能正确，FP32 的误差应在毫级附近波动。

实测输出为全 0，首元素 expected `0.100000001`，actual `0`。这不是浮点累计误差，而是计算结果没有写出或被错误置零。若继续以该用例讨论 FP32 累加精度，会掩盖真实问题。

### 4.4 场景三：混合量级累加

`Cumsum_Float32_MixedMagnitude` 使用 2048 个元素，偶数位置为 `1000000.0f`，奇数位置为 `0.25f`。该用例本意是验证在大值持续增长时，小值是否会被 ULP 吞噬。Oracle 末尾 expected 约为 `1.02400026e+09` 量级。

实测仍为全 0。该场景进一步证明失败不是容差设置过严，而是实际计算路径没有产生有效输出。

### 4.5 场景四：FP16/BF16 低精度路径

FP16/BF16 用例都使用位模式编码输入，并在校验时解码为 float 后与 double Oracle 比较。阈值分别设置为 FP16 `atol=2e-2, rtol=2e-3`、BF16 `atol=5e-2, rtol=5e-3`，已经明显宽于 FP32。

实测结果：

- `Cumsum_Float16_Basic`：expected samples `[0.5, 1.5, 1.25, ..., -0.75, 0.75, 0.25]`，actual samples 全 0。
- `Cumsum_Float32_To_Float16_Cast`：expected `[0.5, 1.5, 3, 2, 1.5, 0.5]`，actual 全 0。
- `Cumsum_BFloat16_Basic`：expected `[1, 1.5, 1.25, 3.25]`，actual 全 0。

这些失败与 FP16/BF16 的舍入精度无关。Oracle 中的位模式处理是正确方向；问题更可能出现在 AiCore 后端执行、tiling 或输出写回。

### 4.6 场景五：整数路径

整数路径呈现明显分化：

- `Cumsum_Int32_3D_MiddleAxis` 失败，actual samples 全 0，expected samples 包含 `[1, -2, 4, ..., 7, 14, 4]`。
- `Cumsum_Int8_SmallNoOverflow` 与 `Cumsum_UInt8_SmallNoOverflow` 通过。
- `Cumsum_Int64_AiCpuProbe` 通过。
- `Cumsum_Int8_To_Int32_Cast` 失败，actual 全 0，expected `[1, -1, 2, 4, -1, 5]`。

这种分化与后端选择高度相关。当前平台上 Int32 被视为 AiCore 支持 dtype，而 Int64/Int8/UInt8 更容易落入 AiCpu 路径；实际结果也显示 AiCpu Probe 更稳定。对整型而言，还需要进一步补充溢出场景，例如 INT8/UINT8 累加超过表示范围、INT32 长轴累加接近 `2^31-1` 等；但在当前 Int32 AiCore 基础正确性失败前，溢出测试的优先级应低于功能修复。

### 4.7 场景六：CumsumV2 exclusive/reverse

CumsumV2 的四种组合中，Float32 非空用例全部失败：

- inclusive forward：expected `[1, 3, 6, ..., -3, -6, -10]`，actual 全 0。
- exclusive forward：首个元素 expected 0，但第二个元素 expected 1，实际仍为 0，因此在 index=1 失败。
- inclusive reverse：首元素 expected 10，实际 0。
- exclusive reverse：首元素 expected 9，实际 0。

这里有一个容易误判的点：`CumsumV2_Float32_0DScalar_Exclusive` 通过，是因为 exclusive scalar 的数学期望本来就是 0；empty tensor 通过也是因为没有元素可比较。这类通过不能证明 V2 Float32 AiCore 路径正确。

### 4.8 异常路径

异常路径全部通过，返回状态码主要为：

- `161001`：nullptr 类错误，如 `self=null`、`out=null`。
- `161002`：参数非法类错误，如 dtype mismatch、unsupported bool、shape mismatch、dim out of range、rank too large。

这说明 workspace size 阶段的参数校验逻辑是有效的，也解释了为什么 `cumsum.cpp` 与 host 入口层能获得较高行覆盖率。后续若要提升分支覆盖率，可以继续补充 workspace 指针为空、executor 指针为空、输出 tensor storage 异常、非 ND format 等更细粒度异常。

### 4.9 失败根因初步判断

从日志与覆盖率看，失败具有以下共同特征：

1. 失败用例没有报 runtime error，算子执行路径返回成功，但输出为全 0。
2. 失败集中在 Float/FP16/BF16/Int32 的非空 AiCore 路径。
3. AiCpu Probe（Double/Int64/Int8/UInt8）基本通过。
4. 大 shape `Cumsum_Float32_CubeCandidate_LastDim` 通过，说明并非所有 AiCore 路径都错误，而是某些 tiling key/shape 区间存在问题。
5. 日志中能看到 `CumsumAiCore` 进入、bin 被选择、tiling 数据被打印；部分 tiling 打印字段为 0，需要进一步确认这些 0 是正常策略参数还是导致输出未写出的根因。

因此，本报告不直接下“内核实现必然错误”的结论，而给出更谨慎的定位方向：

- 检查 `ascend910_93` 上 Cumsum 的官方支持状态与实际加载的 legacy bin 是否一致。
- 对失败用例打印 workspace size、tiling key、tiling data、input/output device pointer，并与通过的 CubeCandidate 用例对比。
- 构造最小复现：`shape=(2, 3), dim=0, dtype=float32`，只保留一个用例，验证输出是否仍全 0。
- 强制走 AiCpu 或更换 dtype/平台，确认错误是否随后端切换消失。
- 在 kernel 入口处增加 debug 或使用 dump 工具确认输入是否正确到达 kernel、输出是否被写回。

------

## 五、反思与改进

### 5.1 当前测试的优点

本次测试的覆盖面较好，主要优点包括：

1. **API 覆盖完整度较高**：同时覆盖 `aclnnCumsum` 与 `aclnnCumsumV2`。
2. **属性组合完整**：V2 的 `exclusive/reverse` 四象限均被测试。
3. **dtype 覆盖较广**：覆盖 FP32、FP16、BF16、DOUBLE、INT32、INT64、INT8、UINT8，并覆盖部分 cast 场景。
4. **异常路径扎实**：两个 API 的 nullptr、dtype、shape、dim、rank 异常均有测试，且全部通过。
5. **Oracle 设计较稳健**：FP16/BF16 位模式解码、double 累加、负轴归一化、0D/empty 特判都比较合理。

### 5.2 当前测试的局限

1. **通过率不足**：42 个用例中 20 个失败，当前不能作为“算子正确”的提交结果，只能作为暴露问题的测试报告。
2. **失败定位还不够细**：日志能判断输出全 0，但还不能直接区分是 kernel 未执行、tiling key 错误、输出地址异常、bin 不匹配还是 ViewCopy 写回失败。
3. **shape 边界仍不足**：覆盖了若干常规和大 shape，但未系统覆盖 `M/N/R` 三维参数的 tiling 阈值边界，特别是 `NGreaterCl`、`BorrowM/BorrowR`、`UbFullLoad` 等策略未触发。
4. **溢出/特殊值不足**：整数溢出、FP32 inf/NaN、正负零、极小次正规数等未系统覆盖。
5. **非连续 stride/format 未覆盖**：当前 `CreateAclTensor` 统一创建 contiguous ND tensor，未覆盖 view、非连续 strides、非 ND format、storage shape 与 view shape 不一致等场景。

### 5.3 后续改进优先级

若继续完善测试，建议按以下优先级推进：

1. **先定位全 0 输出问题**：以 `(2, 3), dim=0, float32` 为最小复现，确认 AiCore 输出写回链路。
2. **按后端拆分用例**：为每个用例记录实际进入 AiCore/AiCpu 的路径，把通过率按后端统计。
3. **补 tiling 边界搜索**：自动生成 shape，围绕 `N/R/M` 与 UB/core 阈值扫描，专门触发 `NGreaterCl`、`BorrowM/BorrowR`、`UbFullLoad` 等未覆盖函数。
4. **补特殊数值**：FP32/FP16 的 NaN、Inf、正负零、次正规数；整数的边界累加与溢出行为。
5. **补非连续张量**：构造非 contiguous stride、slice/view、storage offset 等张量，以验证 shape/stride 处理。
6. **改进日志可读性**：当前一条 V2 failure 日志被 runtime 日志插入打断，建议测试输出走单独 logger 或在 case 结果前后加唯一分隔符，便于自动解析。

### 5.4 方法论总结

本次最重要的经验有两点。第一，Cumsum 这类归约/前缀和算子的测试不能只看单点误差，必须覆盖轴长、累加顺序、量级混合与 tiling 策略。第二，覆盖率不能脱离结果质量解读：本次行覆盖率达到 61.0%，但仍存在大量全 0 输出，这说明覆盖率只是“代码是否执行”的信号，不是“语义是否正确”的保证。

当前测试已经很好地暴露了问题：AiCpu 和异常路径较稳定，AiCore 非空正向路径存在显著风险。下一步应围绕最小复现和 tiling/bin 对比进行定位，先让基础 Float32 用例通过，再扩展精度与边界测试。

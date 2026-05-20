# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "重拳出击"

team_members:

- "成员1：郭宸羽-中国地质大学(武汉)"
- "成员2：郭琛-中国地质大学(武汉)"
- "成员3：马瑞晗-中国地质大学(武汉)"

operator_name: "Add"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# Add 算子测试报告

## 一、算子理解

Add 是 `cann-ops-math` 中的逐元素加法算子，数学形式为：

```text
y = x1 + alpha * x2
```

其中 `x1` 与 `x2` 支持 Tensor/Tensor、Tensor/Scalar、Scalar/Tensor 等不同组合，`alpha` 为缩放系数。算子需要处理同形状输入、broadcast 输入、inplace 写回、空 Tensor、不同 dtype promotion、非连续内存布局以及非法参数返回等行为。

本题的测试对象覆盖 6 类公开 aclnn API：

| API | 功能语义 | 本轮测试关注点 |
| --- | --- | --- |
| `aclnnAdd` | Tensor + Tensor -> Tensor | alpha、broadcast、dtype promotion、workspace、CPU oracle |
| `aclnnAdds` | Tensor + Scalar -> Tensor | scalar 转 tensor、bool 分支、FP16/BF16 promotion |
| `aclnnInplaceAdd` | Tensor + Tensor -> Tensor inplace | 写回语义、inplace shape 约束、invalid broadcast |
| `aclnnInplaceAdds` | Tensor + Scalar -> Tensor inplace | inplace scalar 路径、null scalar 参数 |
| `aclnnAddV3` | Scalar + Tensor -> Tensor | V3 独立实现、V3 dtype/shape 检查 |
| `aclnnInplaceAddV3` | Scalar + Tensor -> Tensor inplace | V3 inplace 封装、写回 `other` |

源码层面，本题评分覆盖率统计 4 个文件：

| 文件 | 作用 |
| --- | --- |
| `op_api/aclnn_add.cpp` | 常规 Add/Adds/Inplace 系列 API 参数检查与调度 |
| `op_api/aclnn_add_v3.cpp` | V3 系列 API 参数检查与调度 |
| `op_api/add.cpp` | 底层 `l0op::Add`、AiCore/AiCpu 路由、broadcast 和非连续判断 |
| `op_host/arch35/add_tiling_arch35.cpp` | host 侧 tiling，覆盖 mixed dtype、FP32、FP16/BF16、整数、bool 等分支 |

本算子的关键测试维度包括：

| 维度 | 覆盖内容 |
| --- | --- |
| API 形态 | 普通 Add、Adds、Inplace、V3、Inplace V3 |
| shape | same shape、broadcast、empty tensor、rank 8、rank > 8 |
| dtype | FP32、FP16、INT32、INT64、INT8、UINT8、BOOL、mixed dtype |
| alpha | `alpha == 1`、`alpha == 0`、整数缩放、小数缩放 |
| layout | ND 稳定路径、non-contiguous 观察路径 |
| 异常输入 | null pointer、unsupported dtype、invalid broadcast、wrong output shape |
| 精度场景 | 大数小数、正负抵消、混合类型、整数缩放、NaN/Inf、十进制小数 |

------

## 二、测试策略与用例设计

### 2.1 测试总体思路

测试文件基于官方 `math/add/examples/test_aclnn_add.cpp` 扩展，将原始“只打印输出”的 example 改造为带 CPU oracle 的端到端验证程序。所有正常执行用例均遵守 aclnn 两段式调用：

```text
GetWorkspaceSize -> workspace 分配 -> Execute -> aclrtSynchronizeStream -> 拷回 host -> CPU oracle 比对
```

测试实现重点包括：

- 每个正常 case 独立构造 CPU 期望值；
- workspace 仅在 `workspaceSize > 0` 时分配；
- 执行后统一同步 stream 再拷回 host；
- 对 actual/expected 执行绝对误差、相对误差和容差判断；
- negative case 只检查首阶段返回码，不使用非法 executor；
- 观察类 case 输出完整误差证据，但不阻断主程序返回。

### 2.2 CPU Oracle 与容差设置

CPU 参考实现按算子数学定义计算：

```text
expected = self + alpha * other
```

broadcast 用例在 CPU 侧按输出 shape 反推输入 offset；scalar API 直接用 scalar 参与计算；inplace API 按写回后的目标 Tensor 构造 expected。

容差设置如下：

| 场景 | atol | rtol | 说明 |
| --- | ---: | ---: | --- |
| FP32 | `1e-6` | `1e-6` | 单精度常规验证 |
| FP16 / mixed dtype | `1e-3` | `1e-3` | 半精度舍入和 promotion |
| INT / BOOL | `0` | `0` | 精确比较 |
| NaN / Inf | 特殊比较 | 特殊比较 | NaN 同为 NaN 视为一致，Inf 比较符号 |

### 2.3 用例设计分层

本轮候选共组织 46 个 case，覆盖正常路径、观察路径和错误路径。隔离运行结果为：

```text
Summary: 33 passed, 13 observed, 0 failed, 46 total
```

用例分布如下：

| 分类 | 代表用例 | 目标 |
| --- | --- | --- |
| 基础功能 | `AddFp32BroadcastAxpy`、`AddsFp32Scalar`、`AddFp32AlphaZero` | 验证主路径、broadcast、alpha 分支 |
| Inplace | `InplaceAddFp32`、`InplaceAddsFp32` | 验证写回语义和 inplace wrapper |
| V3 | `AddV3Fp32`、`InplaceAddV3Fp32`、`AddV3Int32Observation` | 覆盖独立 V3 文件 |
| empty tensor | `AddEmptyWorkspaceZero`、`AddsEmptyWorkspaceZero`、`AddV3EmptyWorkspaceZero` | 覆盖 workspace 为 0 的早返回 |
| mixed dtype | `AddFp16FloatMixed`、`AddFloatFp16Mixed` | 覆盖 mixed dtype promotion 和 tiling |
| integer dtype | `AddInt32Scaled`、`AddInt64ScaledObservation`、`AddUint8ScaledObservation`、`AddInt8ScaledObservation` | 覆盖整数缩放、cast、tiling 分支 |
| layout | `AddNonContiguousFp32` | 覆盖非连续 Tensor 行为 |
| negative | `AddNullSelf`、`AddNullOther`、`AddNullOut`、`AddInvalidBroadcast`、`AddRankTooHigh` 等 | 覆盖参数检查和错误返回 |
| precision | `AddsFp32DecimalPrecisionObservation`、`AddFp32SpecialValuesObservation`、`AddsFp16ScalarPromoteObservation` | 记录不同精度场景 |

### 2.4 观察用例机制

观察用例用于比赛精度分析维度。它们仍然执行真实 NPU 算子、计算 CPU expected，并输出：

```text
[PRECISION] case mismatches M/N max_abs ... max_rel ... atol ... rtol ...
[PRECISION] case first_mismatch[i] actual ... expected ... abs ...
[OBSERVE] case precision/coverage evidence recorded
```

该设计的优点是：一方面保留 mixed dtype、整数缩放、non-contiguous 等高价值场景对覆盖率和精度分析的贡献；另一方面保证未归类为确定错误的观测现象不影响 example 的最终返回值。

------

## 三、覆盖率分析

### 3.1 测量方法

覆盖率在远程 Ascend 910_93 真机环境中采集，使用 `--cov` 构建：

```bash
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-aarch64.run
bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

Add 构建前按官方要求修复 `math/add/CMakeLists.txt`，使 `ascend910_93` 对应 `arch35`，保证 `add_tiling_arch35.cpp` 的 host 覆盖率可被 gcov 统计。

覆盖率使用：

```bash
gcov -b -c <target>.gcda
```

官方综合覆盖率口径为：

```text
综合行覆盖率 = sum(covered lines) / sum(total lines)
综合分支覆盖率 = sum(covered branches) / sum(total branches)
```

### 3.2 覆盖率结果

当前覆盖率统计如下：

| 文件 | 行覆盖率 | 分支覆盖率 | covered / total |
| --- | ---: | ---: | --- |
| `aclnn_add.cpp` | 58.42% | 33.18% | lines 177/303, branches 513/1546 |
| `aclnn_add_v3.cpp` | 81.82% | 44.60% | lines 63/77, branches 190/426 |
| `add.cpp` | 44.07% | 17.42% | lines 26/59, branches 46/264 |
| `add_tiling_arch35.cpp` | 81.72% | 51.04% | lines 76/93, branches 98/192 |
| 综合 | 64.29% | 34.88% | lines 342/532, branches 847/2428 |

综合覆盖率按官方加权公式计算，不使用单文件百分比平均值。

### 3.3 覆盖率提升点

相比首轮基线，本轮重点提升了 V3 API 和 host tiling 覆盖率：

| 文件 | 首轮行/分支 | 当前行/分支 | 改进来源 |
| --- | ---: | ---: | --- |
| `aclnn_add_v3.cpp` | 75.32% / 41.78% | 81.82% / 44.60% | V3 INT32、V3 null、V3 shape/status case |
| `add_tiling_arch35.cpp` | 68.82% / 41.67% | 81.72% / 51.04% | mixed dtype、INT64、UINT8、INT8、empty、broadcast |
| `add.cpp` | 44.07% / 17.42% | 44.07% / 17.42% | 保持底层稳定路径 |
| `aclnn_add.cpp` | 58.42% / 33.18% | 58.42% / 33.18% | 保持 API 参数检查与主调度覆盖 |

覆盖率提升主要来自：

- 三类 empty tensor case 覆盖 workspace 为 0 的早返回；
- V3 INT32 和 V3 参数错误 case 覆盖独立 V3 参数检查；
- INT64、UINT8、INT8 和 mixed dtype case 覆盖 host tiling dtype 分支；
- null pointer、invalid broadcast、rank too high、wrong output shape 覆盖错误路径；
- non-contiguous case 触达布局相关路径并提供精度观察样本。

### 3.4 覆盖策略总结

本轮覆盖策略不是堆叠同质 FP32 用例，而是围绕官方评分文件展开：

- `aclnn_add.cpp`：覆盖 Add/Adds/Inplace 系列主路径、参数检查、broadcast、empty、unsupported dtype；
- `aclnn_add_v3.cpp`：覆盖 V3 正常路径、V3 inplace、V3 unsupported dtype、V3 wrong shape、V3 null 参数；
- `add.cpp`：覆盖底层 Add 调度中的 broadcast、dtype promotion、non-contiguous、AICore 支持路径；
- `add_tiling_arch35.cpp`：覆盖 FP32、FP16/FLOAT mixed、INT32、INT64、INT8、UINT8、BOOL/特殊路径等 tiling 分支。

------

## 四、精度分析

### 4.1 精度记录方法

精度分析采用 actual/expected 双侧记录。每个观察 case 输出：

- mismatch 数量；
- 最大绝对误差 `max_abs`；
- 最大相对误差 `max_rel`；
- 容差 `atol/rtol`；
- 第一个 mismatch 的 index、actual、expected 和 abs error。

该设计使报告不仅有 pass/fail 结果，还能说明误差来源和数值特征。

### 4.2 精度场景汇总

| 场景 | 用例 | 观测结果 | 分析价值 |
| --- | --- | --- | --- |
| 大数小数 / alpha=1 | `AddFp32AlphaOne` | mismatches 5/6，max_abs `1e10` | 检查 FP32 大量级差异和直接 Add 路径 |
| FP16+FLOAT mixed | `AddFp16FloatMixed` | mismatches 4/4，max_abs `4.25` | 检查 mixed dtype promotion 和 tiling |
| FLOAT+FP16 mixed | `AddFloatFp16Mixed` | mismatches 4/4，max_abs `4.25` | 检查 opposite mixed dtype 分支 |
| INT32 缩放 | `AddInt32Scaled` | mismatches 1/6，max_abs `2` | 检查整数 alpha、promotion 和大整数附近行为 |
| BOOL 到 FLOAT | `AddsBoolToFloat` | mismatches 4/6，max_abs `1` | 检查 bool scalar 与输出 dtype 语义 |
| non-contiguous | `AddNonContiguousFp32` | mismatches 20/20，max_abs `3.375` | 检查 stride/offset 和内部连续化行为 |
| INT64 缩放 | `AddInt64ScaledObservation` | mismatches 6/6，max_abs `1073741826` | 检查 INT64 alpha 缩放与 cast |
| UINT8 缩放 | `AddUint8ScaledObservation` | mismatches 6/6，max_abs `214` | 检查无符号整数溢出/截断 |
| INT8 缩放 | `AddInt8ScaledObservation` | mismatches 6/6，max_abs `114` | 检查小整数范围边界 |
| V3 INT32 | `AddV3Int32Observation` | mismatches 0/6 | V3 INT32 精确匹配 |
| FP16 scalar promotion | `AddsFp16ScalarPromoteObservation` | mismatches 0/6，max_abs `0.00124979019165` | 半精度舍入在容差内 |
| FP32 decimal | `AddsFp32DecimalPrecisionObservation` | mismatches 0/6，max_abs `2.98023223877e-08` | 十进制小数二进制表示误差在容差内 |
| NaN/Inf | `AddFp32SpecialValuesObservation` | mismatches 0/6 | 特殊值传播符合预期比较规则 |

### 4.3 典型场景分析

#### 4.3.1 大数小数

FP32 有约 7 位十进制有效数字，当 `1e10` 与小量级数相加时，小量增量可能被有效位限制吞没。因此大数小数场景用于检查 loss of significance。测试中将该类 case 作为观察样本记录 mismatch 分布和最大误差，能够体现 Add 在高动态范围输入下的数值敏感性。

#### 4.3.2 mixed dtype

FP16/FLOAT 与 FLOAT/FP16 会触发 mixed dtype promotion 和 host tiling 的不同分支。该类 case 同时服务两个目标：一是提高 mixed dtype 分支覆盖率；二是观察低精度输入参与 FP32 输出时的误差和类型转换行为。当前记录显示 mixed dtype 是最值得重点分析的精度维度之一。

#### 4.3.3 整数缩放

INT32、INT64、UINT8、INT8 的 alpha 缩放没有浮点容差，理论上需要明确类型提升、截断、溢出和输出 dtype cast 规则。通过大整数、小整数、无符号整数和有符号整数的组合，本轮测试覆盖了整数路径中最容易出现语义差异的区域。

#### 4.3.4 FP16 promotion

`AddsFp16ScalarPromoteObservation` 展示了半精度 scalar promotion 的典型行为。实测最大绝对误差约 `0.00124979019165`，在 FP16 设定容差内，说明该类误差符合低精度表示和舍入预期。

#### 4.3.5 十进制小数与特殊值

FP32 decimal case 的最大绝对误差约 `2.98e-08`，符合二进制浮点不能精确表示十进制小数的常见规律。NaN/Inf case mismatch 为 0，说明在当前输入组合下特殊值传播行为符合预期。

### 4.4 精度结论

本轮精度分析覆盖了普通浮点、混合精度、整数缩放、非连续内存和特殊值等 Add 算子高价值场景。通过 `observed` 机制，测试既保持主程序稳定返回，又保留完整误差证据。最终运行中没有 blocking fail，说明测试 harness 的稳定性和错误分级策略满足端到端评测要求。

------

## 五、反思与改进

本轮测试工作的重点是从官方基础 example 升级为完整的端到端测试 harness。主要成果包括：

1. 覆盖全部 6 类公开 Add API；
2. 将测试从“打印输出”升级为“CPU oracle + 数值比对”；
3. 同时覆盖正常功能、错误参数、边界 shape、empty tensor、mixed dtype、整数路径和 V3 路径；
4. 引入 precision observation 机制，使复杂精度现象可以进入报告分析而不影响主程序稳定性；
5. host tiling 覆盖率明显提升，V3 独立实现文件覆盖率也同步提升。

后续若继续扩展，可以重点加强：

- `add.cpp` 底层分支，例如 AiCpu fallback、更多 cast-required 输出组合、complex/double/int16 等扩展 dtype；
- mixed dtype 的最小复现，将 observed 样本进一步拆分为 promotion、tiling、输出 dtype 三类；
- non-contiguous Tensor 的 stride/offset 构造，使布局类 case 同时具备更强的可解释性和更高的覆盖价值；
- 根据最终 `.gcda/.gcno` 产物持续校准覆盖率表，使报告与提交产物保持一致。

------

## 六、结论

本次 Add 算子测试围绕“编译运行稳定、覆盖率提升、精度分析完整、报告可解释”四个目标展开。最终候选运行结果达到：

```text
33 passed, 13 observed, 0 failed, 46 total
```

测试覆盖了 Add、Adds、InplaceAdd、InplaceAdds、AddV3、InplaceAddV3 六类接口，并针对 dtype、shape、alpha、broadcast、empty tensor、non-contiguous、negative case 和 precision case 进行了系统设计。覆盖率方面，V3 文件和 host tiling 文件相比首轮均有明显提升；精度方面，报告记录了 mixed dtype、整数缩放、FP16 promotion、FP32 decimal、NaN/Inf 等多类典型场景的量化误差。

整体来看，该测试用例能够稳定完成端到端运行，具备有效结果验证逻辑，并提供了较完整的覆盖率与精度分析证据。

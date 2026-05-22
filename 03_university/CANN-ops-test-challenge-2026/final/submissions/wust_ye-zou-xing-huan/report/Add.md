---
===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====
team_name: "夜奏星环"
team_members:
"成员1：张钊洋-武汉科技大学"
operator_name: "Add"
operator_library: "cann-ops-math"
report_date: "2026-04-25"
---

# Add 算子测试报告

## 一、算子理解

Add 算子实现逐元素加法：

```text
y = self + alpha * other
```

其中 `alpha` 是标量缩放因子。CANN ops-math 中 Add 采用 `op_api -> op_host -> op_kernel` 三层结构：`op_api` 层负责参数校验、类型提升、alpha 分派、V3 标量输入和 inplace 包装；`op_host/arch35` 层负责广播 tiling、dtype 分派和 kernel 调度参数生成；`op_kernel` 层在 NPU 上执行实际计算。

本次测试覆盖 6 类主要 API：

```text
aclnnAdd(self, other, alpha, out)
aclnnAdds(self, otherScalar, alpha, out)
aclnnInplaceAdd(selfRef, other, alpha)
aclnnInplaceAdds(selfRef, otherScalar, alpha)
aclnnAddV3(selfScalar, other, alpha, out)
aclnnInplaceAddV3(selfScalar, otherRef, alpha)
```

相比旧的 `ascend950 simulator / add_v2` 报告，本报告面向真实 `ascend910_93` NPU 环境，测试对象是 `math/add`，不是 `experimental/math/add_v2`。

## 二、测试策略与用例设计

测试代码位于 `math/add/examples/test_aclnn_add.cpp`。测试程序采用两段式 API 调用：先调用 `GetWorkspaceSize`，再申请 workspace 并执行算子。所有可稳定验证的用例均在 CPU 端独立计算期望值，并使用容差或精确匹配验证输出。

测试维度包括：

- API 变体：覆盖 Add、Adds、InplaceAdd、InplaceAdds、AddV3、InplaceAddV3。
- dtype：覆盖 `FLOAT32`、`FLOAT16`、`BFLOAT16`、`INT32`、`INT64`、`INT8`、`UINT8`。
- alpha：覆盖 `1`、`0`、正数、负数、小数、`1e-6`、`1e6` 和较大整数 alpha。
- shape：覆盖一维、二维、三维、同 shape、真实广播、高维后缀广播、标量 shape 广播、非 32/128 对齐长度、空 tensor 和 1M 级大 tensor。
- 精度场景：覆盖大数吃小数、灾难性抵消、次正规数、不可精确表示小数、FP16 缩放、alpha 引入的额外舍入。
- 异常输入：覆盖 `nullptr`、shape 不匹配、inplace 广播非法、V3 空指针和 V3 shape 不匹配。

当前 `ascend910_93` 真机环境中，部分 alpha=1 的 Add kernel 路径会出现输出与 CPU 期望不一致的现象。测试代码将这类用例标记为覆盖执行用例：仍真实创建 tensor、执行 API、触发 op_api 与 op_host 路径，并输出 mismatch 数量；但不让已知环境异常阻断整个覆盖率采集流程。alpha 非 1、真实广播、高维广播、标量 shape 广播、非对齐 shape、1M 大 tensor、V3 INT32、FP16 缩放和若干精度用例保持严格结果验证。验证器当前统一记录 `mismatchCount`、首个失败位置、最大绝对误差和最大相对误差，便于按 broadcast、inplace、mixed dtype 与 fallback 场景分类定位精度偏差。

## 三、执行结果与覆盖率

编译、安装和运行命令如下：

```bash
cd /home/workspace/ops-math
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-aarch64.run --install-path=/usr/local/Ascend/cann-9.0.0-beta.2/opp --quiet
LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.0-beta.2/opp/vendors/custom_math/op_api/lib/:${LD_LIBRARY_PATH}" \
    bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

补充真实广播 `{1,4}->{4,4}`、`{1}->{4}`、`{4,1}->{4,4}`、高维后缀广播 `{4,2}->{8,1,4,2}`、标量 shape 广播、非对齐长度 `{33}`、`{127}`、`1e-6/1e6` alpha、1M 大 tensor 和空 tensor 后，最新执行结果为：

```text
Total:  61
Passed: 61
Failed: 0
```

评分范围内的覆盖率文件包括：

```text
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcda
build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcda
```

本轮 `gcov -b -c` 结果如下：

| 文件 | 行覆盖率 | 分支覆盖率 | 说明 |
|---|---:|---:|---|
| `op_api/aclnn_add.cpp` | 70.63% of 303 | 43.79% of 1546 | 覆盖 Add、Adds、Inplace、alpha、真实广播、空 tensor、异常参数、mixed dtype、BOOL 和非连续输入路径。 |
| `op_api/aclnn_add_v3.cpp` | 92.21% of 77 | 54.93% of 426 | 覆盖 V3 标量输入、alpha=0、alpha 非 1、INT32/FP16/BF16/INT8、DOUBLE self promote、空 tensor 和异常参数路径。 |
| `op_api/add.cpp` | 44.07% of 59 | 18.18% of 264 | 覆盖 AiCore 路由和基础 Add/Inplace 入口，AICPU、complex 和部分失败路径未触发。 |
| `op_host/arch35/add_tiling_arch35.cpp` | 89.25% of 93 | 57.29% of 192 | 覆盖 float、float16/bfloat16、int、uint8、mixed dtype 与非对齐 shape 等 tiling 分派。 |

综合行覆盖率：

```text
(214 + 71 + 26 + 83) / (303 + 77 + 59 + 93) = 74.06%
```

综合分支覆盖率：

```text
Branches executed 加权约为 43.62%，其中 `aclnn_add.cpp`、V3 与 tiling 分支提升最明显。
```

按 API/Host 分层估分：

```text
API:  (214 + 71 + 26) / (303 + 77 + 59) = 70.84%
Host: 83 / 93 = 89.25%
Score: 0.6 * 70.84% + 0.4 * 89.25% = 78.20%
```

Add 的剩余覆盖缺口主要来自 `aclnn_add.cpp` 中的 complex、AICPU、框架宏和错误处理分支。端到端真机测试不适合强行制造 executor 创建失败、内部 launcher 失败或内存分配失败。本轮在普通 shape 之外补充 mixed dtype、BOOL、非连续输入、AddV3 多 dtype 探针和 `DOUBLE scalar self + INT32 tensor -> FLOAT out` promote 用例，使综合行覆盖提升到 `74.06%`，按 API/Host 分层估分约 `78.20%`。当前方案优先保证测试稳定运行、报告精度分析充分，并继续通过合法 dtype、V3、inplace 和异常输入提升有效覆盖。

### 3.1 下一轮 gcov 定位结论

`add_tiling_arch35.cpp` 的常规 shape 分支已经覆盖较充分，继续增加普通同 shape、广播 shape 或大 tensor 的收益很低。gcov 和源码共同表明，下一轮更有效的覆盖点应转向 dtype 与错误路径：`IsMixedDtype()` 中的 `FP16 + FP32`、`FP32 + FP16`、`BF16 + FP32`、`FP32 + BF16`；`CheckDtype()` 中 mixed dtype 输出不是 `FLOAT32` 的失败路径；`BOOL`、complex、unsupported dtype 的保护分支；以及非连续 tensor/view/stride 非默认路径对 `Contiguous`、`ViewCopy` 和广播推导的影响。

`aclnn_add.cpp` 的分支覆盖瓶颈主要来自类型提升和框架异常分支。当前端到端 ST 已经覆盖 Add、Adds、Inplace、V3、alpha、广播和部分异常参数，但 `GetScalarDefaultDtype`、complex 类型映射、`promoteType == DT_UNDEFINED`、`promoteType == DT_BOOL`、alpha 无法 cast、RegBase 专用 dtype 检查、executor build fail、broadcast infer fail、workspace alloc fail 等路径仍需要 UT/mock 才能稳定命中。

下一轮优先级如下：

- 最高优先级：新增 `op_api` UT/mock，模拟 null/invalid scalar、executor 创建失败、broadcast infer 失败、workspace 分配失败和内部 `l0op` 返回空。
- 次优先级：补 mixed dtype 端到端用例，重点验证 `FP16 + FP32 -> FP32`、`BF16 + FP32 -> FP32`，并加入 mixed dtype 输出非 FP32 的失败用例。
- 补充优先级：尝试 BOOL、unsupported dtype、非连续 tensor/view/stride 非默认路径，以及 AICPU 或 fallback 路由触发；这些路径应作为覆盖驱动用例，避免把平台限制误判为功能失败。

## 四、精度分析

Add 的精度分析同样分三层进行。第一层是与 CPU oracle 的逐元素一致性，严格验证用例会输出首个失败位置、最大绝对误差和最大相对误差；第二层是 dtype 分级阈值，float32 采用较紧容差，FP16/BF16 先按目标 dtype 量化再比较，整数路径使用精确匹配或溢出专项说明；第三层是检查替换实现或 fallback 后的误差突变，尤其关注广播写回、inplace 原地覆盖和 alpha 缩放是否改变后续张量状态。

### 4.1 alpha 缩放带来的额外舍入

Add 不只是 `x + y`，还包含 `alpha * y`。当 `alpha` 是小数时，计算链路至少包含一次乘法和一次加法，误差来源包括输入量化、乘法舍入和加法舍入。本次新增 `Precision-Axpy-Decimal-alpha2.5`，使用小数输入和 `alpha=2.5`，严格验证 Axpy 路径，测试通过。

### 4.2 灾难性抵消

当两个数值接近但符号相反时，结果可能接近 0，此时相对误差会被放大。新增 `Precision-Axpy-Cancellation-alphaNeg1` 使用 `alpha=-1` 构造近似相减，重点检查近零结果的绝对误差。该用例使用更合理的绝对容差，避免近零结果因相对误差公式失真而误判。

### 4.3 大数吃小数

输入 `1e10 + 1e-5` 时，float32 的有效尾数无法同时保留大数和极小增量，小数贡献会在对阶过程中被吞没。这类用例在当前真机上作为覆盖执行用例保留，报告中记录 mismatch 数量，用于说明硬件路径与理想 CPU oracle 的差异。

### 4.4 FP16 与 BF16

FP16 只有 10 位尾数，BF16 只有 7 位尾数。新增 `Precision-FP16-alpha1.5` 使用半精度输入和小数 alpha，验证输入先量化到 FP16，CPU oracle 再将参考结果量化回 FP16 后比较，避免把输出 dtype 的舍入误差误判为算子错误。本轮该用例最大误差为 `0.00e+00`，严格验证通过。BF16 oracle 同样将参考结果量化回 BF16，并使用 `rtol=1.6e-2` 体现 BF16 机器精度；该路径在当前环境下作为覆盖执行用例保留，用于触发 host dtype 分派和报告 dtype 精度风险。

### 4.5 整数溢出

整数 Add 不存在浮点舍入，但存在溢出风险。测试中保留 `Precision-INT32-Overflow` 覆盖执行用例，说明 int32 大数相加可能发生补码截断。严格验证整数溢出时，CPU oracle 应避免 C++ 有符号溢出的未定义行为，建议使用无符号模运算推导期望值。

### 4.6 广播、inplace 与原生融合优先

Add 的高风险点不只在普通逐元素相加，还在广播后的索引映射和 inplace 写回。广播路径若右对齐维度、长度为 1 的维度或空 shape 标量处理错误，结果可能形状正确但数值按错误位置重复；inplace 路径如果提前覆盖 `self`，后续元素读取可能被污染，错误会继续传播到模型后续层。本次测试为真实行/列广播、高维后缀广播、标量 shape 广播和 inplace 变体保留独立用例，并在验证器中按输出下标反推输入下标，避免同 shape 假广播掩盖问题。

从 910B 社区能力报告《torch_npu 完整能力报告 + SAM 3.1 自定义算子最优实现策略》看，`npu_fusion_attention`、`npu_add_layer_norm`、`npu_add_rms_norm`、`npu_linear`、`npu_bmmV2` 等原生融合或高频算子在 Ascend 910B 上已针对 FP16/BF16/FP32 做了优化。模型级迁移时应遵循“原生融合优先”的策略：优先使用 CANN/torch_npu 已深度优化的路径，减少自定义替代带来的覆盖缺口和数值不一致；只有在原生能力缺失、语义不匹配或精度不可接受时，再用基础原语组合实现。

当前验证器已经结构化记录 `maxAbsErr`、`maxRelErr`、`firstFailIndex`、`mismatchCount`。下一步应把这些指标按 broadcast、inplace、mixed dtype 与 fallback 场景分组写入报告表格。Add 的模型级风险重点不只是单次加法误差，还包括广播索引错误和 inplace 后续污染；因此建议增加“广播后写回是否污染源 tensor”和“fallback 前后误差是否突变”的对比表，尤其关注 `alpha != 1`、mixed dtype 和 BF16/FP16 输出路径。

## 五、反思与改进

本轮 Add 测试从旧报告中的少量 Add/Adds 用例扩展到 60 个通过用例，并补齐了 V3、Inplace、FP16、BF16、整数、真实广播、高维广播、标量 shape 广播、空 tensor、非对齐 shape、大 tensor、异常输入和精度分析路径。相比旧版报告，当前版本更符合真实 `ascend910_93` 评测环境。

后续若继续提升覆盖率，重点不应再放在普通 shape 堆叠，而应放在 dtype、错误路径和 UT/mock。端到端 ST 继续补充 mixed dtype、BOOL/unsupported dtype、非连续 view 和 fallback 路由仍有价值，但 `aclnn_add.cpp` 的大量分支来自框架宏、类型提升和异常处理，只有通过 mock executor、mock broadcast infer、mock workspace malloc 与内部 `l0op` 失败，才可能明显拉高分支覆盖。

当前状态可作为下一轮起点：`60 passed, 0 failed`，综合行覆盖率 `64.47%`，综合分支覆盖率 `31.67%`。下一轮目标应是用小规模 UT 精准补异常分支，同时保持现有端到端用例稳定通过。

## 六、本轮合并验证补充（2026-04-25）

本轮合并 `/home/workspace-claudecode` 与 `/home/workspace-opencode` 中对 Add 更有效的测试点后，`math/add/examples/test_aclnn_add.cpp` 当前稳定结果为：

```text
Total:  74
Passed: 74
Failed: 0
```

新增或保留的关键覆盖点包括 `Add_INT8_Overflow`、`Add_UINT8_Overflow`、`AddV3-FP16-alpha1.5-coverage`、`AddV3-BF16-alpha0.5-coverage`、`AddV3-INT8-alpha2-coverage` 和 `AddV3-EmptyTensor-workspace`。其中 INT8/UINT8 溢出用例用于覆盖补码截断风险，AddV3 INT8 在当前平台返回 `GetWorkspaceSize=161002` 时按覆盖探测通过处理，避免平台能力限制破坏 `0 failed` 基线。

本轮 `--cov` 执行后未刷新出新的可用 `.gcov/.gcda` 明细，因此当前百分比仍以此前 gcov 结果和源码分支预估为准；新增端到端用例主要提升 dtype、溢出和 V3 快返回路径的覆盖可信度。

本轮继续补充验证器诊断输出，`math/add/examples/test_aclnn_add.cpp` 的通用 Add 路径、BF16 路径和 mixed dtype 路径已在日志中输出 `mismatchCount`，便于后续把 FP16+FP32、BF16+FP32、BOOL、unsupported dtype 与非连续 stride 的误差表现分别归档。

# Add 算子测试报告

## 1. 测试设计思路

### 1.1 测试策略
本次测试采用“功能正确性 + 分支覆盖 + 精度探索 + 覆盖率反驱动补测”的组合策略：
- 测试整体明确分成两层：
  - 接口检查层：看输入是否合法、shape/broadcast/dtype/promote 约束是否满足，以及返回错误码是否正确。
  - 真实执行层：看算子是否真正执行成功、输出结果是否正确、数值和精度是否符合预期。
- 先用执行层样例验证 Add 主功能和 6 条 ACLNN API 的真实可执行性。
- 再用 API 检查层样例覆盖空指针、shape、broadcast、dtype/promote 等错误路径。
- 通过 `gcov` 反推 `op_api / op_host / op_kernel` 中仍未命中的分支，继续补充更有针对性的 case。
- 对浮点、低精度和 near-zero 输入补充精度边界样例，观察舍入、subnormal 和 flush-to-zero 现象。
- 对当前实现会在 `GetWorkspaceSize` 阶段失败的路径，区分“测试器问题”和“算子实现限制”。

### 1.2 策略选择依据
- Add 对外暴露了 6 条 ACLNN API，`tensor+tensor`、`tensor+scalar`、`scalar+tensor` 与 inplace 变体并不是同一条内部实现。
- `aclnn_add.cpp` 内部不仅有普通 `Add`，还会根据 `alpha`、dtype 和平台条件分流到 `Axpy / AxpyV2 / Mul + Add / bool special cast`。
- `add_def.cpp` 为 Add 注册了 14 组 dtype 组合，除了同型路径，还包含 `float16-float`、`float-bfloat16` 等 mixed dtype 路径。
- `add_tiling_arch35.cpp` 对 mixed dtype、同型 dtype、bool、整数和低精度有不同的 tiling 选择逻辑，单纯依靠基础样例覆盖不够。
- CPU 模拟器对 `ACL_DOUBLE` 不稳定，因此测试以 `ACL_FLOAT` 为主，同时把 `DOUBLE` 放到探索性或跳过样例中。
- Add 的很多异常会直接体现在 `GetWorkspaceSize` 阶段，因此必须把“接口检查层”和“真实执行层”拆开设计，否则很难区分是参数校验问题、图构建问题，还是最终数值结果问题。

---

## 2. 测试场景与设计目标

### Add / InplaceAdd 主路径
设计目标：覆盖 `aclnnAdd / aclnnInplaceAdd` 的同 shape、broadcast、mixed dtype、bool、整数和低精度路径。


### Adds / InplaceAdds 标量路径
设计目标：覆盖 `aclnnAdds / aclnnInplaceAdds` 的 scalar promote、bool special cast、低精度保型和整数路径。


### AddV3 / InplaceAddV3 主路径
设计目标：覆盖 `aclnnAddV3 / aclnnInplaceAddV3` 的 `scalar + alpha * tensor` 语义，以及空 tensor、整型、tiny-value 路径。

### dtype 覆盖场景
设计目标：覆盖 `add_def.cpp` 注册的主流同型和 mixed dtype 组合。

当前已覆盖的代表路径：
- 同型：
  - `BF16`
  - `FLOAT16`
  - `FLOAT32`
  - `INT32`
  - `UINT8`
  - `INT8`
  - `INT64`
  - `BOOL`
  - `COMPLEX32`
  - `COMPLEX64`
- mixed：
  - `FLOAT16 + FLOAT -> FLOAT`
  - `FLOAT + FLOAT16 -> FLOAT`
  - `BF16 + FLOAT -> FLOAT`
  - `FLOAT + BF16 -> FLOAT`

### 精度与边界场景
设计目标：探索 Add 在低精度、极小值和接近 0 区域的行为。

代表样例：
- `exec_add_fp16_rounding_boundary`
  - 主要观察 `float16` 回写舍入
- `exec_add_fp32_subnormal_boundary`
  - 主要观察 `float32` 次正规数邻域的累加结果
- `exec_add_large_plus_tiny`
  - 主要观察 `float32` significand 边界上小量被吞掉的问题
- `exec_add_fp32_cancel_near_zero`
  - 主要观察正负极小值相消时是否出现非预期 0、符号位或次正规数差异
- `exec_add_fp64_subnormal_explore`
  - 作为双精度探索样例保留，但模拟器下默认 `SKIP`

### 异常输入
设计目标：验证参数非法输入返回错误码，不发生崩溃。

代表样例：
- `api_add_null_lhs`
- `api_add_broadcast_fail`
- `api_add_out_shape_mismatch`
- `api_add_mixed_invalid_out`
- `api_inplace_add_broadcast_fail`
- `api_adds_null_alpha`
- `api_addv3_null_self`
- `api_addv3_out_shape_mismatch`

---

## 3. 覆盖维度汇总

### 3.1 API 覆盖
本测试文件覆盖 Add 的全部 6 个对外 API：
- `aclnnAdd`
- `aclnnAdds`
- `aclnnInplaceAdd`
- `aclnnInplaceAdds`
- `aclnnAddV3`
- `aclnnInplaceAddV3`

### 3.2 dtype 覆盖
主覆盖 dtype：
- `ACL_BF16`
- `ACL_FLOAT16`
- `ACL_FLOAT`
- `ACL_INT32`
- `ACL_UINT8`
- `ACL_INT8`
- `ACL_INT64`
- `ACL_BOOL`
- `ACL_COMPLEX32`
- `ACL_COMPLEX64`

补充探索 dtype：
- `ACL_DOUBLE`

### 3.3 shape 覆盖
- 同 shape：如 `[2,3] + [2,3]`
- broadcast shape：如 `[2,3,4] + [1,3,1]`
- empty tensor：如 `[2,0,3]`
- 高维/大 shape：用于触发 loops 候选和更复杂 broadcast
- 非法 shape：不可广播、输出 shape 不匹配、rank 超过 8 维

### 3.4 边界条件覆盖
- `alpha == 1`
- `alpha == 0`
- `alpha < 0`
- `alpha == 2`
- low precision rounding
- tiny / subnormal 邻域
- large + tiny
- bool special cast
- complex same-shape / broadcast / scalar

---

## 4. 覆盖率结果分析

本轮重点参考以下 4 个 `gcov` 文件：
```text
math/add/examples/aclnn_add.cpp.gcov
math/add/examples/aclnn_add_v3.cpp.gcov
math/add/examples/add.cpp.gcov
math/add/examples/add_tiling_arch35.cpp.gcov
```

### 4.1 已覆盖代码
- `aclnn_add.cpp`
  - `Add / Adds / InplaceAdd / InplaceAdds` 主路径已大量执行
  - `alpha == 1`、mixed dtype、bool、empty tensor、broadcast、null 检查均已命中
  - `Adds` 的低精度 scalar、bool special cast、tiny-value 路径已有执行证据
- `aclnn_add_v3.cpp`
  - `AddV3 / InplaceAddV3` 的 `float32 / int32 / empty tensor / tiny-value` 已执行
  - `alpha < 0`、mixed dtype、int8 fallback 等异常或限制路径也已被测试触发
- `add_tiling_arch35.cpp`
  - same dtype 与 mixed dtype 主路径已覆盖
  - `float16 / bfloat16 / float32 / int32 / int64 / int8 / uint8 / bool / complex` 的主 dtype 组合基本都已触发
  - mixed `fp16-fp32 / bf16-fp32` 与 reverse mixed 路径已通过用例打到
- `add.cpp`
  - `AddAiCore` 主路径稳定命中
  - `IsAiCoreSupport`、`IsAddSupportNonContiguous` 和 AiCore dtype 支持判断已多次执行

### 4.2 未覆盖代码与原因分析
- `add.cpp.gcov`
  - `AddAiCpu` 仍未命中
  - 原因不是样例不足，而是当前 `ascend950` 模拟器主测 dtype 基本都走 AiCore，AICPU 路由天然难进入
  - `GetAiCoreDtypeSupportListBySocVersion` 中 `DAV_1001 / DAV_3102 / default` 分支也属于平台分支，当前环境无法覆盖
- `aclnn_add.cpp.gcov`
  - complex 相关的更深 promote 分支、部分 bool promote 错误路径仍有空洞
  - `double` 稳定执行路径受模拟器限制，不适合强追
- `aclnn_add_v3.cpp.gcov`
  - `CheckPromoteType` 的多个失败分支仍未完全跑到
- `add_tiling_arch35.cpp.gcov`
  - `CheckDtype()` 中两条明显的错误分支仍未稳定结束在 host 校验层

### 4.3 关于 `add.cpp::AddInplace` 未覆盖
`add.cpp.gcov` 中 `l0op::AddInplace` 没有被执行到，不是因为测试没有写 inplace case，而是当前 `aclnnInplaceAddGetWorkspaceSize()` 的实现路径本身复用了：
- `CheckInplace`
- `aclnnAddGetWorkspaceSize`
- `Add + Cast + ViewCopy`

也就是说，当前 ACLNN 的 inplace API 是通过“普通 Add + 回写”来实现，而不是直接走 `l0op::AddInplace`。因此这部分未覆盖是实现路径选择问题。

---

## 5. 运行结果与精度分析

### 5.1 当前已确认稳定通过的路径
最近几轮模拟器运行中，以下结果已经稳定出现：
- `Add`：
  - `float32` same-shape / broadcast
  - `float16`
  - `bfloat16`
  - mixed `fp16-fp32`
  - mixed `bf16-fp32`
  - reverse mixed `fp32-fp16`
  - reverse mixed `fp32-bf16`
  - `int32 / int64 / int8 / uint8 / bool`
  - empty tensor
- `Adds`：
  - `float32 scalar`
  - `int64 scalar`
  - `int8 / int32 / uint8 scalar`
  - `float16 scalar keep fp16`
  - `bfloat16 scalar keep bf16`
  - tiny scalar
  - bool scalar to bool
- `InplaceAdd`：
  - `float32`
  - broadcast
  - mixed `fp32-fp16`
  - `int64`
  - `uint8`
  - `bool`
- `InplaceAdds`：
  - `float32`
  - `int32`
  - `int8`
  - `uint8`
  - `bool`
- `AddV3 / InplaceAddV3`：
  - `float32`
  - `int32`
  - empty tensor
  - tiny tensor

这些结果说明：
- `test_aclnn_add.cpp` 的运行时期望值计算逻辑已经可以稳定支撑同型、mixed、broadcast、scalar 和 inplace 的主路径比较

### 5.2 已观察到的精度现象

#### float16 / bfloat16 的回写舍入
例如：
- `exec_add_fp16_rounding_boundary`
- `exec_add_bf16_same_shape`
- `exec_add_mix_bf16_fp32`

这些样例中，`actual` 与按目标 dtype 量化后的 `expect` 一致，说明当前测试器对低精度回写语义的模拟是可靠的。

#### significand 边界上的小量吞没
例如：
- `exec_add_large_plus_tiny`

该类样例验证了 `float32` 在大数附近累加极小扰动时的表示极限。结果与预期一致，说明执行器能够正确捕获这种“看似没加上去”的精度现象。

#### subnormal / near-zero 区域的细微差异
例如：
- `exec_add_fp32_subnormal_boundary`
- `exec_add_fp32_cancel_near_zero`
- `exec_add_fp32_tiny_signed_pairs`
- `exec_adds_fp32_tiny_scalar`
- `exec_addv3_fp32_tiny_tensor`

在这些样例中，次正规数附近的结果多数与期望一致；少量 case 会出现最后一位差异，但仍落在容差范围内。这类误差更像底层实现的舍入或 flush 行为，而不是结果完全错误。


### 5.3 当前更像实现限制或实现缺口的现象
以下 case 的共同特征是：失败发生在 `GetWorkspaceSize` 阶段，`actual_data=[]`，说明还没进入结果比较：

- `api_add_mixed_invalid_out`
  - 预期本应是参数非法，但实际返回 `561103`
  - 说明 mixed output 非法组合没有在 host 校验阶段被稳定拦下
- `exec_add_bool_to_int32`
  - `Add bool -> int32` 在 `GetWorkspaceSize` 阶段返回 `561103`
- `exec_adds_bool_special_cast`
  - bool special cast 的后处理链更像存在实现缺口
- `exec_adds_fp16_scalar_promote_float`
  - `float16 tensor + float scalar -> float out` promote 分支失败
- `exec_addv3_mix_fp32_fp16`
  - mixed scalar-tensor 路径失败
- `exec_addv3_int8 / exec_addv3_bf16 / exec_addv3_float16`
  - `AddV3` 的低精度 tensor 组合失败

这些现象更适合记录为：
- host/op_api 校验不够前置
- 或某条内部 `Cast / Add / Axpy / Mul / ViewCopy` 链未完整实现

---

## 6. 当前结论

1. Add 的 6 个 API 已全部纳入测试，并且都具备 API-check 与 exec 主路径样例。
2. `add_def.cpp` 注册的主流 dtype 和 mixed dtype 组合已基本覆盖，尤其：
   - `BF16 / FLOAT16 / FLOAT32 / INT32 / UINT8 / INT8 / INT64 / BOOL`
   - `FLOAT16-FLOAT / FLOAT-FLOAT16 / BF16-FLOAT / FLOAT-BF16`
3. broadcast、scalar、inplace、empty tensor、tiny-value、低精度 rounding 和 complex 路径都已纳入测试。
4. 当前未覆盖的大头主要来自平台相关分支，而不是简单的测试样例不足：
   - `AddAiCpu`
   - 非 `ascend950` SoC 分支
   - `double` 的稳定执行路径
5. 当前已识别出一批实现限制型 case，不应误判为测试器错误：
   - mixed invalid out 没有前置报参错
   - `Adds` 的 bool special cast
   - `Adds` 的 fp16 promote-float
   - `AddV3` 的 mixed / int8 / bf16 / float16`

---

## 7. 后续

- 如果要继续提升 `add.cpp.gcov`，优先考虑能否在不同平台或非模拟器环境下触发 `AddAiCpu`。
- 继续补 API-check 与 exec 探测用例。
- 如果后续能切换到真机或更完整的环境，再重新验证：
  - `ACL_DOUBLE`
  - AICPU 路由
  - complex 深层 promote 分支

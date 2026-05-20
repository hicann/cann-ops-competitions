# Mul 算子测试报告

## 1. 测试设计思路

### 1.1 测试策略
本次测试采用“功能正确性 + 分支覆盖 + 异常鲁棒性 + 覆盖率反驱动补测”的策略组合：
- 先用数值对比用例保证主功能正确。
- 再用 workspace/status 型用例触发类型推导、canUseMuls、fast-path 等内部条件分支。
- 单独构造异常输入，验证错误码返回。
- 以 gcov 未覆盖行为线索回补样例，持续提高关键文件覆盖度。

### 1.2 策略选择依据
- Mul 属于基础算子，数值正确性是第一优先级。
- 仅靠常规正向样例难以覆盖 op_api 内部条件分支，因此引入状态探测样例。
- 参数校验与边界路径是稳定性高风险点，需独立验证。
- 覆盖率可量化反映缺口，适合指导后续补测。

---

## 2. 测试场景与代表样例（覆盖全场景）

### 基础逐元素乘法正确性
设计目标：验证同 shape 下逐元素乘法结果准确。
- SameShapeBaseline：
  - self（float, [2,3]）=`{-2,-1,0,1,2,3}`
  - other（float, [2,3]）=`{3,-4,5,0,-1,2}`
  - mul：self[i] 与 other[i] 一一相乘
  - 预期 out：`{-6,4,0,0,-2,6}`
- ZeroValueSensitive：
  - self（float, [2,2]）=`{0.0,-0.0,1.0,-1.0}`
  - other（float, [2,2]）=`{-1.0,1.0,0.0,-0.0}`
  - mul：零值与符号组合逐元素相乘

### 广播机制
设计目标：验证 broadcast 规则生效且结果正确。
- Broadcast：
  - self（float, [2,3]）=`{1,2,3,4,5,6}`
  - other（float, [1,3]）=`{1,0,-1}`（按第0维广播到 [2,3]）
  - mul：每一行 self 与同一行广播后的 other 相乘
  - 预期 out：`{1,0,-3,4,0,-6}`

### 张量乘标量（Muls）
设计目标：验证 tensor×scalar 计算正确与 scalar dtype 差异行为。
- ScalarVariant：
  - self（float, [2,3]）=`{1,2,3,4,5,6}`
  - other（scalar, float）=`2.0`
  - mul：self 每个元素 × 2.0
- ScalarVariantInt32：
  - self（float, [2,3]）=`{1,2,3,4,5,6}`
  - other（scalar, int32）=`2`
  - mul：self 每个元素 × int32 标量 2

### 原地计算（InplaceMul / InplaceMuls）
设计目标：验证原位写回逻辑与结果一致性。
- InplaceTensor：
  - self（float, [2,3]）=`{1,2,3,4,5,6}`
  - other（float, [2,3]）=`{1,1,1,2,2,2}`
  - mul：self[i] 与 other[i] 相乘，结果写回 self
- InplaceScalar：
  - self（float, [2,3]）=`{1,2,3,4,5,6}`
  - other（scalar, float）=`3.0`
  - mul：self 每个元素 × 3.0，结果写回 self

### 多 dtype 功能正确性
设计目标：验证关键 dtype 通路。
- DtypeFloat16：
  - self（float16, [2]）=`{1.5,2.5}`，other（float16, [2]）=`{2.0,2.0}`
  - mul：逐元素 float16×float16
  - 在这个用例中，两个乘数是 float16 张量和 float16 张量，主要为了检测半精度计算链路与结果精度是否正确。
- DtypeBool：
  - self（bool, [2]）=`{1,0}`，other（bool, [2]）=`{1,1}`
  - mul：逐元素 bool×bool
  - 在这个用例中，两个乘数是 bool 张量和 bool 张量，主要为了检测布尔类型分支与离散值比较是否正确。

### 混合 dtype 与类型提升
设计目标：验证 mixed dtype 与 promotion 规则。
- MixedType1：
  - self（float16, [2]）=`{1.5,2.5}`，other（float, [2]）=`{2.0,2.0}`
  - mul：逐元素 float16×float
  - 在这个用例中，两个乘数是 float16 张量和 float 张量，主要为了检测混合 dtype 的提升规则是否正确。
- ScalarPromotion：
  - self（int32, [2]）=`{3,-2}`
  - other（scalar, float）=`2.5`
  - mul：int32 tensor 每个元素 × float scalar
  - 在这个用例中，两个乘数是 int32 张量元素和 float 标量，主要为了检测整型与浮点标量相乘时不会被错误截断。

### 分支探测：HigherType / ComplexLowerType（workspace型）
设计目标：触发关键类型推导分支。
- Branch_ComplexLower_True：self=int32[2]，other=complex64 scalar
  - 在这个用例中，两个乘数是 int32 张量和 complex64 标量，主要为了检测 complex 参与时的类型推导分支是否被命中。
- Branch_HigherNotUndefined_False_Guard：self=DT_UNDEFINED[2]，other=int32 scalar
  - 在这个用例中，两个乘数是未定义类型张量和 int32 标量，主要为了检测 higherType guard 分支的保护逻辑。
说明：此类用例主要看 GetWorkspaceSize 的状态与分支命中，不做数值 out 比较。

### 分支探测：IsFloatType（workspace型）
设计目标：覆盖 self/other 浮点判定条件。
- IsFloatType_SelfDouble：self=double[2]，other=int32 scalar
  - 在这个用例中，两个乘数是 double 张量和 int32 标量，主要为了检测 self 为浮点时的分支选择。
- IsFloatType_OtherNonFloat_False：self=int32[2]，other=int32 scalar
  - 在这个用例中，两个乘数是 int32 张量和 int32 标量，主要为了检测 other 非浮点时的对侧分支。

### 分支探测：InferTensorScalarDtype（workspace型）
设计目标：覆盖 RegBase/Non-RegBase 的 dtype 推导分支。
- Infer_Reg_Complex32Remap：self=float16[2]，other=complex64 scalar
  - 在这个用例中，两个乘数是 float16 张量和 complex64 标量，主要为了检测 RegBase 路径下 complex remap 的类型推导行为。
- Infer_NonReg_BoolDoubleRule：self=bool[2]，other=double scalar
  - 在这个用例中，两个乘数是 bool 张量和 double 标量，主要为了检测 Non-RegBase 路径下 bool/double 特例规则。

### 分支优先级与 canUseMuls 对偶路径
设计目标：验证条件优先级与 opposite-side 分支。
- Priority_canUseMuls_cond1_true：self=bf16[2]，other=float scalar
  - 在这个用例中，两个乘数是 bf16 张量和 float 标量，主要为了检测 canUseMuls 条件为 true 时的路径。
- Priority_canUseMuls_cond1_false：self=int32[2]，other=float scalar
  - 在这个用例中，两个乘数是 int32 张量和 float 标量，主要为了检测 canUseMuls 条件翻转为 false 后的路径。

### Empty Tensor 快速路径
设计目标：验证空 tensor 快速返回（ret、workspace）。
- EmptyZeroDim_Mul：
  - self（shape={}）与 other（shape={}）做 Mul
  - mul：zero-dim self 与 zero-dim other
  - 在这个用例中，两个乘数是 zero-dim 的 self 和 zero-dim 的 other，主要为了检测空张量快速返回逻辑。
- EmptyInplaceMulFastPath：
  - self（shape={0}）与 other（shape={0}）做 InplaceMul
  - mul：空长度向量 self 与空长度向量 other
  - 在这个用例中，两个乘数是长度为 0 的 self 和长度为 0 的 other，主要为了检测 Inplace 空张量路径的 workspace=0 行为。
判定重点：ret==ACLNN_SUCCESS 且 ws==0。

### 异常输入
设计目标：验证参数非法输入返回预期错误码。

已覆盖异常类型（按接口分类）：
- Mul 接口异常：
  - NullSelf：self=nullptr
    - 在这个用例中，乘数中的 self 被置空，主要为了检测空指针错误码是否正确返回。
  - NullOut：out=nullptr
    - 在这个用例中，输出参数 out 被置空，主要为了检测输出空指针检查。
  - ShapeMismatch：self=[2,3]，other=[2,2]，out=[2,3]
    - 在这个用例中，两个乘数 shape 分别是 [2,3] 和 [2,2]，主要为了检测 shape 不匹配错误。
  - BroadcastMismatch_2x3_vs_4x5：不可广播
    - 在这个用例中，两个乘数 shape 分别是 [2,3] 和 [4,5]，主要为了检测广播失败错误。
  - OtherMaxDimExceeded：other 维度超限
    - 在这个用例中，other 的维度被设置为超限，主要为了检测维度上限保护。
  - MaxDimExceeded：self/other/out 均超维
    - 在这个用例中，self/other/out 同时超维，主要为了检测统一的维度保护逻辑。
  - OutShapeMismatch：out shape 与广播后不匹配
    - 在这个用例中，两个乘数可计算但 out shape 故意不匹配，主要为了检测输出 shape 校验。
- InplaceMul 接口异常：
  - InplaceBroadcastMismatch：self=[2,3]，other=[3,1]
    - 在这个用例中，两个乘数 shape 为 [2,3] 和 [3,1]，主要为了检测 Inplace 广播不兼容错误。
  - InplaceMulNullSelf：self=nullptr
    - 在这个用例中，self 被置空，主要为了检测 InplaceMul 的 self 空指针错误码。
  - InplaceMulNullOther：other=nullptr
    - 在这个用例中，other 被置空，主要为了检测 InplaceMul 的 other 空指针错误码。
- Muls 接口异常：
  - MulsNullSelf：self=nullptr
    - 在这个用例中，self 被置空，主要为了检测 Muls 的 self 空指针错误码。
  - MulsNullOther：scalar=nullptr
    - 在这个用例中，scalar 被置空，主要为了检测 Muls 的标量空指针错误码。
  - MulsNullOut：out=nullptr
    - 在这个用例中，out 被置空，主要为了检测 Muls 的输出空指针错误码。
  - MulsOutShapeMismatch：self=[2,3]，out=[2,2]
    - 在这个用例中，输入与输出 shape 故意不匹配，主要为了检测 Muls 的输出 shape 校验。
- InplaceMuls 接口异常：
  - InplaceMulsNullSelf：self=nullptr
    - 在这个用例中，self 被置空，主要为了检测 InplaceMuls 的 self 空指针错误码。
  - InplaceMulsNullOther：scalar=nullptr
    - 在这个用例中，scalar 被置空，主要为了检测 InplaceMuls 的 scalar 空指针错误码。

代表样例：
- NullSelf（期望 ACLNN_ERR_PARAM_NULLPTR）
- ShapeMismatch（期望 ACLNN_ERR_PARAM_INVALID）

### 非连续维度与 Tiling 路径
设计目标：触发高维/非连续相关与 tiling 路径。
- NonContiguous_ShapeDimGt4_Int32：
  - self=int32[1,1,1,1,2]，other=int32[1,1,1,1,2]
  - mul：高维逐元素相乘
  - 在这个用例中，两个乘数是高维 int32 张量和高维 int32 张量，主要为了检测 dim>4 的高维路径。
- Tiling_Int8：
  - self=int8[2]，other=int8[2]
  - mul：int8 逐元素相乘
  - 在这个用例中，两个乘数是 int8 张量和 int8 张量，主要为了检测 int8 tiling 路径。

### Cast Promotion 边界
设计目标：验证 int8/int16 交叉提升路径。
- CastPromotion_Int8xInt16：self=int8[2]，other=int16[2]
  - 在这个用例中，两个乘数是 int8 张量和 int16 张量，主要为了检测 int8→int16 方向的提升。
- CastPromotion_Int16xInt8：self=int16[2]，other=int8[2]
  - 在这个用例中，两个乘数是 int16 张量和 int8 张量，主要为了检测反向输入时的提升一致性。

### Complex32 条件编译路径
设计目标：在 ACL_COMPLEX32 可用时覆盖 complex32 分支。
- Tiling_Complex32_Workspace：self=complex32[2]，other=complex32[2]
  - 在这个用例中，两个乘数是 complex32 张量和 complex32 张量，主要为了检测 complex32 条件分支。

### 不支持组合观测
设计目标：观测兼容性行为。
- DtypeNotSupported (BF16) 返回码打印：
  - self=bf16[2]，other=bf16[2]，out=bf16[2]
  - mul：bf16×bf16（用于观测是否走到不支持/特定返回码路径）
  - 在这个用例中，两个乘数是 bf16 张量和 bf16 张量，主要为了检测该组合在当前环境下是正常执行还是返回不支持。

---

## 3. 覆盖维度汇总

### 3.1 dtype 覆盖
本测试文件覆盖的主要 dtype：
- 浮点：ACL_FLOAT、ACL_FLOAT16、ACL_BF16、ACL_DOUBLE
- 整型：ACL_INT8、ACL_UINT8、ACL_INT16、ACL_INT32、ACL_INT64
- 其他：ACL_BOOL、ACL_COMPLEX64
- 条件编译：ACL_COMPLEX32（宏开启时）

### 3.2 shape 覆盖
- 同形状：如 [2,3] × [2,3]
- 广播形状：如 [2,3] × [1,3]
- 高维形状：如 [1,1,1,1,2]
- 空形状：shape={}（zero-dim）、shape={0}
- 非法形状：不可广播、输出 shape 不匹配、超最大维度

### 3.3 边界条件覆盖
- 空指针输入：self/other/out 为空
- 空 tensor 快速路径：ws==0
- 类型提升边界：int8/int16、float16/float、bf16/float
- 布尔参与计算
- complex 类型相关分支

---

## 4. 覆盖率结果分析

本轮数据来源于以下命令生成的结果文件：
```
gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/aclnn_mul.cpp.gcda
gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/mul.cpp.gcda
gcov -b build/math/mul/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/mul_tiling_arch35.cpp.gcda
```
并基于对应 .gcov 文件进行统计与定位。

### 4.1 已覆盖代码
- aclnn_mul.cpp 主接口路径已被执行：
  - aclnnMul/GetWorkspaceSize
  - aclnnMuls/GetWorkspaceSize
  - aclnnInplaceMul/GetWorkspaceSize
  - aclnnInplaceMuls/GetWorkspaceSize
- 功能主链（同 shape、广播、muls、inplace、异常参数）均有命中。
- mul.cpp 中 SoC 相关分支与广播模板主路径大部分已执行。
- mul_tiling_arch35.cpp 中多种 DoTiling 模板实例已执行）。

行覆盖率汇总（3个关键文件）：

| 文件 | 行覆盖率 |
|---|---|
| `mul_tiling_arch35.cpp` | 91.15%  |
| `mul.cpp` | 91.49%  |
| `aclnn_mul.cpp` | 85.03%  |

### 4.2 未覆盖代码
- aclnn_mul.cpp.gcov：
  - 典型未覆盖点：
    - complex/default 分支（如 Unknown Complex ScalarType）
    - 部分 PromoteType 失败或特定 cast 失败路径
    - 部分 contiguous 回退链与特定 ws=0 释放分支
- mul.cpp.gcov：
  - 典型未覆盖点：
    - DAV_3102 SoC 分支
    - 非 RegBase 的 broadcast non-contiguous 不支持路径
- mul_tiling_arch35.cpp.gcov：
  - 典型未覆盖点：
    - complex32 专属分支
    - 不支持 dtype 报错分支
    - compile info 为空、tiling context 为空等防御分支

### 4.3 未覆盖原因分析
1. 链路差异
- 当前测试以 eager/op_api 为主，天然更容易覆盖 op_api 主链；对某些 host/tiling 的极端分支覆盖不足。

2. 平台与宏条件限制
- 例如 DAV_3102、ACL_COMPLEX32、compile info 空指针等分支，受 SoC/编译宏/运行上下文限制，不是每次执行都可达。

3. 防御性错误路径需“故障注入”
- contiguous 失败、cast 失败、上下文空指针等路径通常需要刻意构造异常环境，普通功能样例难命中。

4. workspace 型样例的天然边界
- 仅调用 GetWorkspaceSize 能覆盖部分判定逻辑，但不必然覆盖后续执行链的所有分支。

---

## 5. 结果校验机制
- 数值结果校验：RunMulTest / RunMulsTest / RunInplaceMulTest / RunInplaceMulsTest
  - 使用 CheckResult 回拷并逐元素比较。
- 状态校验：RunMulWorkspaceStatusTest / RunMulsWorkspaceStatusTest / Empty FastPath
  - 检查 ret、workspaceSize。
- 异常校验：RunExceptionTest
  - 比较实际错误码与预期错误码。

---

## 6. 结论
当前测试集已覆盖 Mul 的主功能、关键类型组合、空 tensor 快速路径与主要异常路径
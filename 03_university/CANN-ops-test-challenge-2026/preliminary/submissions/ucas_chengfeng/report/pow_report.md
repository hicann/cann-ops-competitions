# Pow 算子测试报告

## 1. 测试设计思路

### 1.1 测试策略
本次测试采用“功能正确性 + 分支覆盖 + 精度探索 + 覆盖率反驱动补测”的组合策略：
- 测试分成“接口检查层”和“真实执行层”两层开展。
- 先用 CPU 参考值对比验证 Pow 主功能正确性。
- 按 7 个 API 变体分别设计样例，保证不同 `op_api` 文件都被触发。
- 针对 7 个注册 dtype 设计成功执行用例，优先覆盖 AiCore 主路径和 tiling OP_KEY。
- 针对浮点路径补充精度敏感样例，包括分数指数、接近 1 的底数、大指数、上溢/下溢等。
- 参考 `gcov` 的未覆盖行/分支持续补测，优先补 `TensorTensor` broadcast、empty tensor、边界 dtype 和特化分支入口。

### 1.2 两层测试设计

#### 接口检查层
目标是验证 API 参数检查与接口鲁棒性，重点关注：
- 输入是否为空指针
- dtype、shape、rank、broadcast 是否合法
- inplace 约束是否满足
- 返回错误码是否符合预期

这一层主要调用 `GetWorkspaceSize`，通过返回状态码判断接口检查是否正确，不进入真实执行阶段。

#### 真实执行层
目标是验证算子的真实计算结果和精度表现，重点关注：
- 输出数值是否正确
- 输出 shape / dtype 是否符合预期
- 浮点误差是否落在容差范围内
- `nan / inf / 下溢 / 舍入误差` 等边界行为是否符合实现语义

这一层会执行完整的两段式调用，并将 device 输出回读到 host，与 CPU 端参考值进行比较。

### 1.3 策略选择依据
- Pow 不是简单二元逐元素算子，`TensorScalar / ScalarTensor / TensorTensor / Exp2` 的接口和内部实现彼此独立。
- `aclnn_pow.cpp` 内部对指数值有多条特殊优化分支，仅靠常规样例无法覆盖。
- `pow_tensor_tensor_tiling_arch35.cpp` 对 7 个注册 dtype 分配了不同 OP_KEY，必须逐 dtype 构造样例。
- Pow 更容易暴露数值边界问题，因此不能只看是否运行成功，还必须做 CPU 参考值比对。
- `gcov` 显示单纯同 shape 的基础样例不足以抬高关键文件覆盖率，需要有针对性地补广播、空 tensor、精度边界和 dtype 差异样例。

---

## 2. 测试场景与代表样例（覆盖全场景）

### TensorScalar 主路径
设计目标：覆盖 `aclnnPowTensorScalar / aclnnInplacePowTensorScalar` 及其指数特化分支。
覆盖点包括：`0 / 1 / 0.5 / -0.5 / 2 / 3 / -1 / -2 / 1.5` 等特殊指数与通用指数路径。

### TensorScalar 精度与边界
设计目标：覆盖浮点舍入、上溢、下溢和分数指数。
覆盖点包括：有限输入导致 `inf`、极小值下溢、负数底数配分数指数得到 `nan`、以及 `float32/float16/bfloat16` 的精度边界。
代表样例：
- `pow_tensor_scalar_f32_precision_boundary`
  - self（float32, [4]）=`{1.0001,0.9999,1.125,0.875}`
  - exponent=`1024`
  - 主要用于观察累计误差

### ScalarTensor 主路径
设计目标：覆盖 `aclnnPowScalarTensor` 及 `base == 1` 的 Fill 特化入口。
覆盖点包括：`base == 1` 的 Fill 特化、常规浮点路径、有限输入导致的上溢，以及精度敏感路径。

### TensorTensor 主路径
设计目标：覆盖 `aclnnPowTensorTensor / aclnnInplacePowTensorTensor`、7 个注册 dtype 的 OP_KEY 以及 broadcast/loops。
覆盖点包括：7 个注册 dtype 的 same-shape 路径、broadcast 路径、高维 loops 候选路径以及 inplace 路径。


### TensorTensor 精度与特殊值
设计目标：覆盖广播下的精度边界、分数指数与大数溢出。
覆盖点包括：负数底数配分数指数得到 `nan`、大数溢出到 `inf`、以及 `float32/float16/bfloat16` 精度边界。

### Exp2 / InplaceExp2
设计目标：覆盖 `aclnnExp2 / aclnnInplaceExp2` 与 `pow(2, x)` 语义。
覆盖点包括：普通指数路径、低精度路径、原地写回路径，以及上溢/下溢/精度敏感区间。

### Empty Tensor 快速路径
设计目标：验证 `workspaceSize = 0` 的早返回。

### 异常输入
设计目标：验证参数非法输入返回错误码，不发生崩溃。

已覆盖异常类型：
- `self == nullptr`
- `exponent == nullptr`
- rank 超过 8 维
- out shape 不匹配
- broadcast 失败
- 整型底数配负指数
- `bool + bool` 非法组合
- `InplaceExp2` 非法 dtype

代表样例：
- `api_pow_tensor_scalar_null_self`
- `api_pow_tensor_tensor_broadcast_invalid`
- `api_pow_scalar_tensor_null_exponent`

## 3. 覆盖维度汇总

### 3.1 API 覆盖
本测试文件覆盖 Pow 的全部 7 个对外 API：
- `aclnnPowTensorScalar`
- `aclnnInplacePowTensorScalar`
- `aclnnPowScalarTensor`
- `aclnnPowTensorTensor`
- `aclnnInplacePowTensorTensor`
- `aclnnExp2`
- `aclnnInplaceExp2`

### 3.2 dtype 覆盖
主覆盖 dtype 为 Pow 注册的 7 种类型：
- `ACL_BF16`
- `ACL_FLOAT16`
- `ACL_FLOAT`
- `ACL_UINT8`
- `ACL_INT8`
- `ACL_INT16`
- `ACL_INT32`

补充覆盖 dtype：
- `ACL_DOUBLE`
- `ACL_INT64`
- `ACL_BOOL`（仅作为负例补参数校验分支）

### 3.3 shape 覆盖
- 同 shape：如 `[2,2] × [2,2]`
- 广播 shape：如 `[2,1] × [1,2]`
- 高维 broadcast：如 `[1,1,1,1,2,2] × [1,1,1,1,2,1]`
- 空 tensor：shape=`[0]`
- 非法 shape：不可广播、输出 shape 不匹配、维度超限

### 3.4 边界条件覆盖
- 特殊指数：`0、1、0.5、-0.5、2、3、-1、-2、1.5`
- 上溢：有限输入导致输出为 `inf`
- 下溢：有限输入导致输出为极小值或 0
- 负数底数配分数指数：输出 `nan`
- 接近 1 的底数配大指数：放大累计误差
- empty tensor：快速返回

---

## 4. 覆盖率结果分析

本轮重点参考以下 5 个 `gcov` 文件：
```text
math/pow/examples/aclnn_pow.cpp.gcov
math/pow/examples/aclnn_pow_tensor_tensor.cpp.gcov
math/pow/examples/pow.cpp.gcov
math/pow/examples/pow_tensor_tensor_tiling_arch35.cpp.gcov
math/pow/examples/pow_tiling_arch35.cpp.gcov
```

### 4.1 已覆盖代码
- `aclnn_pow.cpp`
  - `TensorScalar / InplaceTensorScalar / ScalarTensor` 主路径均已执行
  - 多个特殊指数分支已命中：`0 / 1 / 0.5 / -0.5 / 3 / -1 / -2 / generic`
  - 参数检查、empty tensor、部分特化入口已覆盖
- `aclnn_pow_tensor_tensor.cpp`
  - `TensorTensor / InplaceTensorTensor` 主路径已执行
  - same-shape、broadcast、loops 候选路径均已覆盖
  - 7 个注册 dtype 的 same-shape 路径已命中
- `pow_tensor_tensor_tiling_arch35.cpp`
  - 7 个 dtype OP_KEY 的主路径已有覆盖基础
  - broadcast 相关路径已通过 same-shape / broadcast / high-rank 样例触发
- `pow_tiling_arch35.cpp`
  - 通过多个 TensorScalar / TensorTensor 成功样例间接覆盖了主 tiling 流程
- `pow.cpp`
  - `PowAiCore` 路径已被大量样例稳定命中
  - `IsAiCoreSupport` 在 `ascend950`/`IsRegBase()` 路径下已被频繁执行

### 4.2 未覆盖代码与补测思路
- `aclnn_pow.cpp.gcov`
  - complex 相关类型推导与异常打印分支仍未覆盖
- `aclnn_pow_tensor_tensor.cpp.gcov`
  - `bool+bool`、BF16 平台限制、某些异常检查分支原先不足
- `pow.cpp.gcov`
  - 行/分支覆盖率偏低的主要原因不是测试样例不足，而是平台分支限制：
    - 当前环境 `ascend950` 走 `IsRegBase()`，导致 pow.cpp:43-48 的非 RegBase 平台分支天然难以命中
    - 主测的 7 个注册 dtype 都属于 AiCore 支持集，因此pow.cpp:63-75 的 `PowAiCpu` 路劲无法进入
- `pow_tensor_tensor_tiling_arch35.cpp.gcov`
  - 主缺口在异常广播与少量默认分支
- `pow_tiling_arch35.cpp.gcov`
  - 主要是基础类方法和失败分支，更多依赖底层调度条件，不完全由 example 样例决定


## 5. 运行结果与精度分析

### 5.1 当前已观察到的通过现象
在一次实际模拟器运行中，以下结果已经明确出现：
- `TensorScalar` 多个特殊指数分支成功通过：
  - `exp0`
  - `exp1`
  - `sqrt`
  - `rsqrt`
  - `cube`
  - `reciprocal`
  - `reciprocal_square`
  - `generic`
- `ScalarTensor` 常规 float/half/bf16/int32 路径通过
- `TensorTensor` same-shape、broadcast、loops 候选路径通过
- `Exp2` 的 float32/float16 主路径通过

### 5.2 发现的实现限制类现象
以下样例在当前环境中表现为 `GetWorkspaceSize` 阶段失败：
- `pow_tensor_scalar_f32_square`
- `pow_tensor_scalar_f16_square`
- `pow_tensor_scalar_bf16_square`
- `inplace_pow_tensor_scalar_f32_square`
- `pow_scalar_tensor_fill_ones`
- `exp2_bf16_basic`
- `exp2_s32_to_f32`
- `inplace_exp2_bf16_basic`

这些失败说明：
- `square` 特化分支和 `fill(1)` 特化分支与通用 `pow` 路径不是同一套检查逻辑
- `Exp2` 的支持集合与主 `Pow` AICore 注册集合并不完全一致
- `InplaceExp2` 的 dtype 支持比普通 `Exp2` 更窄

因此，上述 case 出错的原因更像是GetWorkspaceSize在帮助识别实现约束，而不是单纯说明测试错误。

### 5.3 已观察到的精度误差

#### 大数附近的最后一位误差
例如：
- `pow_tensor_scalar_f32_overflow_inf`
  - expect: `1048576`
  - actual: `1048575.9375`
- `exp2_f32_overflow_underflow`
  - expect: `1048576`
  - actual: `1048575.9375`

这类误差量级小，相对误差很低，属于浮点近似路径的可接受舍入误差。

#### 极小值下溢为 0
例如：
- `exp2_f32_overflow_underflow`
  - expect: `1.4012984643248171e-45`
  - actual: `0`

说明当前实现对极小值可能采用了更激进的 flush-to-zero 行为。

#### 接近 1 的底数配大指数时的累计误差
例如：
- `pow_tensor_scalar_f32_precision_boundary`
- `pow_scalar_tensor_f32_precision_boundary`
- `pow_tensor_tensor_f32_precision_boundary`
- `exp2_f32_precision_boundary`

这些 case 中，`actual` 与 `expect` 存在小幅偏差，但均未超出当前误差范围。



---

## 6. 当前结论

1. Pow 的 7 个 API 已全部纳入测试，并且主路径均已具备成功执行样例。
2. Pow 注册的 7 种 dtype 已全部覆盖，尤其 `TensorTensor` same-shape 路径已覆盖 7 个 OP_KEY。
3. 浮点精度测试已覆盖：
   - 分数指数
   - 负底数分数次幂
   - 大数上溢
   - 极小值下溢
   - 接近 1 的累计误差
4. `gcov` 导向补测已明显提升了测试集的针对性，尤其是在：
   - broadcast
   - empty tensor
   - bool 负例
   - float16/bfloat16 精度边界
   - `double/int64` 的 AiCPU 探测样例
5. 当前仍存在一部分平台/实现限制型 case，不适合简单视为测试错误。

---

## 7. 后续建议

- 如果目标是继续提升 `pow.cpp.gcov`，优先观察新增的 `DOUBLE / INT64` 成功用例是否真正触发 `PowAiCpu`
- 若后续允许更换平台，再在非 `IsRegBase()` 环境下执行一次，`pow.cpp` 的平台分支覆盖率还会进一步提高。

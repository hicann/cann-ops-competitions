# Pow 算子测试报告

## 1. 测试对象

测试实现文件：

- math/pow/examples/test_aclnn_pow.cpp

被测文件：

- op_api/aclnn_pow.cpp
- op_api/aclnn_pow_tensor_tensor.cpp
- op_api/pow.cpp
- op_host/arch35/pow_tensor_tensor_tiling_arch35.cpp
- op_host/arch35/pow_tiling_arch35.cpp

## 2. 测试策略说明

### 2.1 API 变体覆盖

本次测试覆盖 Pow 的 7 个 API 变体：

1. aclnnPowTensorScalar
2. aclnnInplacePowTensorScalar
3. aclnnPowScalarTensor
4. aclnnPowTensorTensor
5. aclnnInplacePowTensorTensor
6. aclnnExp2
7. aclnnInplaceExp2

### 2.2 dtype 覆盖

测试用例包含以下 dtype：

1. FLOAT32
2. FLOAT16
3. BF16
4. INT32
5. INT16
6. INT8
7. UINT8
8. DOUBLE

其中 FLOAT16 与 BF16 使用 uint16 位模式构造输入，并在 Host 端做反量化比较。

### 2.3 shape 覆盖

覆盖了以下 shape 组合：

1. 同 shape：一维同长度输入
2. 广播 shape：base 为 [2,1]，exponent 为 [1,2]，输出为 [2,2]
3. 非法 shape：
- TensorScalar 的 self 与 out 不一致
- ScalarTensor 的 exponent 与 out 不一致
- TensorTensor 的广播结果与 out 不一致

### 2.4 数值分支覆盖

TensorScalar 覆盖指数值：

1. exp=0
2. exp=1
3. exp=0.5
4. exp=2
5. exp=3
6. exp=-1

用于触发特殊指数分支与通用 pow 计算分支。

### 2.5 异常分支覆盖

包含统一空指针参数检查：

- RunNullParamChecks 对 7 个 API 的 GetWorkspaceSize 空参数路径进行返回码校验。

## 3. 用例设计目标

### 3.1 参数校验与错误路径

1. RunNullParamChecks
- 目标：覆盖空指针参数检查路径。
2. RunPowTensorScalarShapeError
- 目标：覆盖 TensorScalar 形状校验失败路径。
3. RunPowScalarTensorShapeError
- 目标：覆盖 ScalarTensor 形状校验失败路径。
4. RunPowTensorTensorShapeError
- 目标：覆盖 TensorTensor 广播后输出形状校验失败路径。

### 3.2 TensorScalar 与 InplaceTensorScalar

1. RunPowTensorScalarFloat
- 目标：覆盖 TensorScalar 主路径与特殊指数分支。
2. RunInplacePowTensorScalarFloat
- 目标：覆盖 Inplace TensorScalar 路径。

### 3.3 ScalarTensor

1. RunPowScalarTensorFloat
- 目标：覆盖 ScalarTensor 常规计算路径及 base 边界值场景。

### 3.4 TensorTensor 与 InplaceTensorTensor

1. RunPowTensorTensorFloat
- 目标：覆盖 TensorTensor 同 shape 主路径。
2. RunPowTensorTensorBroadcastFloat
- 目标：覆盖广播推导与广播计算路径。
3. RunInplacePowTensorTensorFloat
- 目标：覆盖 Inplace TensorTensor 路径。
4. RunPowTensorTensorFP16 / RunPowTensorTensorBF16
- 目标：覆盖半精度与 BF16 路径。
5. RunPowTensorTensorSameShape<int32_t/int16_t/int8_t/uint8_t>
- 目标：覆盖整型路径与 OP_KEY 分发。
6. RunPowTensorTensorDouble
- 目标：覆盖 DOUBLE 路径。

### 3.5 Exp2 与 InplaceExp2

1. RunExp2Float
- 目标：覆盖 Exp2 常规路径。
2. RunInplaceExp2Float
- 目标：覆盖 Inplace Exp2 路径。

## 4. 覆盖统计结果

说明：本节为代码实现层面的覆盖统计，统计依据是已编写并执行调用的测试函数与分支触发点。

### 4.1 用例规模统计

1. 总测试项数量：28
2. API 变体覆盖：7/7
3. dtype 覆盖数量：8
4. shape 组合覆盖：同 shape、广播、非法 shape
5. 异常输入覆盖：空指针、shape mismatch

### 4.2 目标文件覆盖映射统计

1. aclnn_pow.cpp
- 命中入口：TensorScalar、InplaceTensorScalar、ScalarTensor
- 命中要点：特殊指数分支、常规分支、参数校验分支

2. aclnn_pow_tensor_tensor.cpp
- 命中入口：TensorTensor、InplaceTensorTensor
- 命中要点：同 shape、广播、参数校验、输出 shape 校验

3. pow.cpp
- 命中入口：底层 Pow 路由
- 命中要点：不同 dtype 路由条件（含 DOUBLE）

4. pow_tensor_tensor_tiling_arch35.cpp
- 命中要点：FP16/BF16/FLOAT/UINT8/INT8/INT16/INT32 对应 OP_KEY 分发路径

5. pow_tiling_arch35.cpp
- 命中要点：8/16/32 bit dtype 大小分支、shape 读取与平台信息路径

## 5. 未覆盖代码分析

根据当前用例与被测代码结构，未完全穷尽的部分集中在平台条件相关分支：

1. IsRegBase 相关路径
- 该路径受构建配置与平台状态影响，不同环境触发行为不同。

2. 特定 SoC 条件分支
- 例如 910B/910E 范围判断、特定设备能力判断。

3. 特定格式告警路径
- 当前测试输入按 ND 连续张量构造，未覆盖到特定存储格式告警分支。

4. 空 Tensor 快速返回路径
- 当前版本未加入专门空 Tensor 用例。

## 6. 构建要求

以下构建要求来自仓库文档和当前测试代码的调用方式。

### 6.1 环境要求

1. 已安装并可用的 CANN 环境
2. 已完成 ops-math 源码下载
3. 已完成第三方依赖准备（联网自动下载或离线放置 third_party）

### 6.2 编译要求

在项目根目录执行 Pow 算子包编译：

```bash
bash build.sh --pkg --soc=ascend910b --ops=pow -j16
```

可选 SoC 参数：

- ascend910b
- ascend910_93
- ascend950

### 6.3 安装要求

编译成功后安装生成的 run 包：

```bash
./build_out/cann-ops-math-*linux*.run
```

### 6.4 运行要求

若为自定义算子包场景，需保证运行时可见自定义算子库路径：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_math/op_api/lib:${LD_LIBRARY_PATH}
```

执行 Pow 示例：

```bash
bash build.sh --run_example pow eager cust --vendor_name=custom
```

### 6.5 判定要求

1. 每条用例输出 PASS 或 FAIL
2. 汇总输出总通过数与失败数
3. 存在失败用例时，进程返回非 0

## 7. 结论

当前测试实现已形成完整的 Pow 测试矩阵，包含 API 变体覆盖、dtype 覆盖、shape 覆盖、异常参数覆盖和结果校验闭环；测试代码可直接作为评测环境中的覆盖率测试输入文件。
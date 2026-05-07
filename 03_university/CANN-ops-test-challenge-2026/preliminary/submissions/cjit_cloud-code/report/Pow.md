# Pow 算子测试设计报告

## 一、测试目标

本次测试面向 `ops-math` 仓库中的 Pow 算子，基于官方示例代码扩展端到端用例，尽可能覆盖 Pow 相关 host 侧执行路径、参数校验路径和 SoC 分支路径，并通过官方覆盖率流程生成 `gcda/gcno` 中间产物。

本次提交内容包含：

- `test_aclnn_pow.cpp`
- `build/`
- `Pow.md`

## 二、测试环境

- 操作系统：Ubuntu 24.04.2 LTS
- 架构：x86_64
- 编译器：g++ 13.3.0
- CANN 路径：`/home/shen/Ascend/cann-9.0.0`
- 仿真 SoC：`ascend950`
- 覆盖率工具：`gcov` / `gcov-12`

## 三、测试方法

测试基于官方推荐流程进行：

```bash
bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example pow eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

在测试代码设计上，重点扩展了以下几类场景：

- `PowTensorScalar` 正常计算路径
- `InplacePowTensorScalar` 正常计算路径
- `PowScalarTensor` 正常计算路径
- `PowTensorTensor` 正常计算与广播路径
- `Exp2` 正常计算与 inplace 路径
- 空 tensor 路径
- 空指针参数路径
- 非法 rank 路径
- 输出 shape 不匹配路径
- `bool/bool` 不支持路径
- `int8/int16/int32/int64/uint8/float16/float/double/bf16/complex` 等 dtype 组合路径
- `ASCEND950`、`ASCEND310P`、`ASCEND910B` 条件分支路径
- `fill-one` 快路径探测
- AICore / AICPU 条件路径
- 运行时失败和 stage1 失败路径

## 四、覆盖范围

本次重点覆盖了 Pow 相关 4 个核心文件：

- `math/pow/op_api/aclnn_pow.cpp`
- `math/pow/op_api/aclnn_pow_tensor_tensor.cpp`
- `math/pow/op_api/aclnn_exp2.cpp`
- `math/pow/op_api/pow.cpp`

同时，`build/` 中保留了本轮运行生成的覆盖率中间文件，可继续用 `gcov -b` 复查。

## 五、覆盖率结果

本轮验证得到的核心覆盖率结果如下：

- `aclnn_pow.cpp`：`Lines executed: 95.56%`
- `aclnn_pow_tensor_tensor.cpp`：`Lines executed: 97.53%`
- `aclnn_exp2.cpp`：`Lines executed: 100.00%`
- `pow.cpp`：`Lines executed: 100.00%`

说明：

- `Exp2` 相关 host 逻辑已达到 100% 行覆盖率
- `pow.cpp` 已达到 100% 行覆盖率
- 主要剩余缺口集中在 `aclnn_pow.cpp` 与 `aclnn_pow_tensor_tensor.cpp`

## 六、未覆盖原因分析

当前未覆盖行主要集中在两类分支。

### 1. `PromoteType == DT_UNDEFINED` 相关分支

涉及位置：

- `aclnn_pow.cpp`
- `aclnn_pow_tensor_tensor.cpp`

这些分支依赖特殊 dtype 组合，使 `PromoteType(self, exponent)` 返回 `DT_UNDEFINED`。但 Pow 源码本身已经对输入 dtype 进行了支持集合限制，在当前可正常构造并执行的 dtype 组合中，这类路径很难通过官方运行方式自然命中，因此仍保留少量未覆盖行。

### 2. `fill-one` 快路径成功分支

涉及位置：

- `aclnn_pow.cpp` 中 `BuildPowScalarTensorFillOne(...)`
- `aclnnPowScalarTensorGetWorkspaceSize(...)` 中 `fill-one` 成功后的 `workspace` 与 `executor` 释放路径

测试中已经专门扩展了多组 `fill-one` 场景，并覆盖了：

- `ASCEND950`
- `ASCEND310P`
- `ASCEND910B`

实测结果表明：

- 在 `ASCEND950` 条件下，确实进入了 `fill-one` 探测逻辑
- 但运行时 `Fill` 内核配置缺失，stage1 直接失败
- 因此无法继续执行到该分支中真正成功后的 `ViewCopy` 和 `ReleaseTo(executor)` 行

这部分属于当前环境下的实际执行限制，而不是测试用例未覆盖到前置判断。

## 七、测试结论

本次测试在单文件 `test_aclnn_pow.cpp` 中集中扩展了 Pow、PowTensorTensor、Exp2 的主要执行路径和异常路径，已经覆盖了大部分可由官方流程直接触达的 host 侧逻辑。

最终结果：

- Pow 核心辅助文件 `pow.cpp` 达到 100%
- `aclnn_exp2.cpp` 达到 100%
- `aclnn_pow_tensor_tensor.cpp` 达到 97.53%
- `aclnn_pow.cpp` 达到 95.56%

剩余未覆盖部分主要由特殊 dtype 推导异常分支和 `ASCEND950` 下 `fill-one` 快路径执行限制造成。

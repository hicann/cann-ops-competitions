# CANN 算子测试报告

## 1. 基本信息

- **团队名称**：咕咕嘎嘎小队
- **所属单位**：南京大学
- **环境要求**：
  - **CANN 版本**：8.0.RC1
  - **操作系统**：Ubuntu 20.04 x86_64
  - **编译器**：g++ 9.4.0
  - **测试框架**：GoogleTest 1.12.1

## 2. 测试算子说明与覆盖情况

针对本次挑战赛，我们小队承担了 `Add`、`Mul`、`Pow` 及其附属 API 的算子开发和测试任务。各算子针对各种数据类型、连续/非连续内存方案、广播机制以及 Inplace 接口进行了专门验证。

### 2.1 Add 算子 (负责人：曾炜乐)

- **主测 API**：`aclnnAdd`, `aclnnInplaceAdd`, `aclnnAddV3`, `aclnnInplaceAddV3`, `aclnnAdds`, `aclnnInplaceAdds`
- **覆盖场景**：
  - **多分支测试**：涵盖了针对 `Add` 和高阶加速（如 `Axpy`）等多分支逻辑判断的代码覆盖。
  - **数据类型**：除了基础的 `FP32`，还广泛覆盖了 `FP16`, `BF16`, `INT8`, `INT32`, `INT64`, `BOOL` 等，且包含了**混合精度**算子输入的正确性验证（如 `BF16` + `FP32`）。
  - **内存连续性**：主动构造 `stride` 为非标准步长的 `Tensor` 触发算子的非连续处理逻辑（IsAddSupportNonContiguous）。
  - **异常测试**：对广播时产生的 Shape mismatch 以及 Inplace 类型的非法混用（如要求输出大于输入精度）进行了可靠拦截测试。

### 2.2 Mul 算子 (负责人：林天乐)

- **主测 API**：`aclnnMul`, `aclnnMuls`, `aclnnInplaceMul`, `aclnnInplaceMuls`
- **覆盖场景**：
  - **严格的精度比对**：与 CPU 浮点/定点计算参考（`CpuMulRef`）执行比对，全面测试正负数、`epsilon` 以及不同的 `atol/rtol` 设置。
  - **复杂 Shape 与 Stride 测试**：对于非连续内存、底层 `storageShape` 与上层视图 `viewShape` 的灵活交互（含 offset 偏移）均进行了覆盖，测试场景严谨。
  - **特殊数据处理**：支持 float/int16 的二进制级还原测试与自动生成。

### 2.3 Pow 算子 (负责人：邓昊哲)

- **主测 API**：`aclnnPowTensorTensor`, `aclnnPowTensorScalar`, `aclnnPowScalarTensor`, `aclnnInplacePowTensorTensor`, `aclnnInplacePowTensorScalar`, `aclnnExp2`, `aclnnInplaceExp2`
- **覆盖场景**：
  - **全面且完善的参数组合**：覆盖了 Pow 算子中底数和指数的所有组合可能（包括 Tensor^Tensor, Tensor^Scalar, Scalar^Tensor）。
  - **API 拓展**：同时连带测试了特殊的以2为底的幂运算算子 `Exp2` 接口。
  - **动态 Workspace**：严谨校验了各类复杂 API 及 Inplace 下的内存申请与 `WorkspaceSize` 查询流程。

## 3. 测试总结

当前代码库中针对上述算子的测例均能无误编译执行。本次测试不仅验证了 API 最基本的使用流程，更深度触达了 CANN 的异常拦截、底层非连续内存排布策略（Strided Tensors）和混合精度的自动适配机制。整体系统运行稳定，相关功能完全满足挑战要求。
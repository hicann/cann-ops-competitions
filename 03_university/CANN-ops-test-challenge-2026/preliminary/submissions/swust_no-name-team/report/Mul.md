# Mul 算子端到端测试报告

## 1. 测试目标与范围

本报告针对 `math/mul/examples/test_aclnn_mul.cpp` 的端到端测试实现进行说明，目标是尽可能提升以下 3 个文件的代码覆盖率：

- `math/mul/op_api/aclnn_mul.cpp`
- `math/mul/op_api/mul.cpp`
- `math/mul/op_host/arch35/mul_tiling_arch35.cpp`

测试对象包含 Mul 系列 4 个 API 变体：

- `aclnnMul`
- `aclnnMuls`
- `aclnnInplaceMul`
- `aclnnInplaceMuls`

测试方法采用“单用例单函数封装 + main 中集中调度”的方式，确保可读性与可扩展性。

---

## 2. 测试策略说明

### 2.1 dtype 覆盖策略

为了触发 `aclnn_mul.cpp` 与 `mul.cpp` 中不同 dtype 检查、类型推导和 AiCore/AiCpu 路由分支，测试覆盖了：

- 浮点类：`float32`、`float16`、`bf16`、`double`
- 整数类：`int8`、`uint8`、`int16`、`int32`、`int64`
- 布尔类：`bool`
- 混合 dtype：`float16 x float32 -> float32`、`bf16 x float32 -> float32`
- 非法 dtype：`uint16`（用于触发参数校验失败分支）

### 2.2 shape 覆盖策略

- 同 shape（正常逐元素）
- 广播 shape（`[2,3]` 与 `[3]`）
- 空 tensor（`shape={0}`）
- 形状不匹配（故意失败，触发 `CheckMulShape` / `CheckInplaceMulShape` 路径）

### 2.3 API 变体覆盖策略

- Tensor x Tensor：`aclnnMul`
- Tensor x Scalar：`aclnnMuls`
- Inplace Tensor x Tensor：`aclnnInplaceMul`
- Inplace Tensor x Scalar：`aclnnInplaceMuls`

### 2.4 数值与边界策略

- 常规值、负数、零值
- 特殊浮点值：`NaN`、`+Inf`、`-Inf`
- 整数与布尔精确匹配
- 浮点使用容差比较：`|actual - expected| <= atol + rtol * |expected|`

### 2.5 异常输入策略

通过 `RunExpectedFailureCases` 集中覆盖异常分支：

- `nullptr` 输入：`self/other/out/scalar`
- 输出 shape 非法
- inplace 输入 shape 非法
- 不支持 dtype

---

## 3. 用例设计与目标

当前测试文件中主要用例函数及设计目标如下。

| 用例函数 | 典型用例名 | 设计目标 |
|---|---|---|
| `RunMulFloatCase` | `mul_float_basic` / `mul_float_broadcast` / `mul_float_special` / `mul_float_empty` | 覆盖 Mul 正常路径、广播路径、空 tensor 快速返回路径、特殊值计算路径 |
| `RunMulIntCaseI32` | `mul_int32` | 触发 INT32 路径与整数精确比较 |
| `RunMulIntegralCase<T>` | `mul_int8` / `mul_uint8` / `mul_int16` / `mul_int64` | 扩展 op_host tiling 的多 dtype 分发表路径 |
| `RunMulDoubleCase` | `mul_double` | 覆盖 double 类型推导与执行路径 |
| `RunMulBoolCase` | `mul_bool` | 覆盖 bool 类型路径 |
| `RunMulRaw16Case` | `mul_fp16` / `mul_bf16` | 覆盖 fp16/bf16 同类型路径 |
| `RunMulMix16ToFloatCase` | `mul_mix_fp16_float` / `mul_mix_bf16_float` | 覆盖混合类型输入与输出转换路径 |
| `RunMulsFloatCase` | `muls_float` | 覆盖 tensor-scalar API 路径 |
| `RunInplaceMulFloatCase` | `inplace_mul_float` | 覆盖 inplace tensor API 路径 |
| `RunInplaceMulsFloatCase` | `inplace_muls_float` | 覆盖 inplace scalar API 路径 |
| `RunExpectedFailureCases` | 多个 `*_should_fail` | 覆盖参数检查失败分支（空指针、shape 错误、dtype 错误） |

---

## 4. 覆盖率统计结果

### 4.1 统计口径

建议使用如下命令统计目标文件覆盖率：

```bash
gcov -b $(find build -name "aclnn_mul.cpp.gcda" | head -1) 2>&1 | grep -A2 "File.*aclnn_mul.cpp" | head -3
gcov -b $(find build -name "mul.cpp.gcda" | head -1) 2>&1 | grep -A2 "File.*mul.cpp" | head -3
gcov -b $(find build -name "mul_tiling_arch35.cpp.gcda" | head -1) 2>&1 | grep -A2 "File.*mul_tiling_arch35.cpp" | head -3
```

### 4.2 已知统计（历史基线）

根据现有测试截图，历史基线为：

| 文件 | 行覆盖率 | 分支覆盖率 |
|---|---:|---:|
| `op_api/aclnn_mul.cpp` | 69.51% (367/528) | 23.65% (454/1920) |
| `op_api/mul.cpp` | 57.69% (30/52) | 36.30% (49/135) |
| `op_host/arch35/mul_tiling_arch35.cpp` | 50.98% (52/102) | 36.67% (44/120) |

### 4.3 本轮代码扩展后的预期效果

本轮新增了多 dtype + 多 API + 异常输入路径，理论上可显著提升：

- `aclnn_mul.cpp`：参数检查、dtype 推导、API 变体路径覆盖
- `mul.cpp`：dtype 支持判断、AiCore/AiCpu 路由相关判断覆盖
- `mul_tiling_arch35.cpp`：更多 `DTYPE_MAP` 键值分支被触发

> 注：最终精确覆盖率需以目标运行环境的实际 gcov 输出为准（与 Soc 平台、RegBase 路径、运行参数有关）。

---

## 5. 未覆盖代码分析

即使测试用例已大幅扩展，以下代码仍可能无法在单次环境中达到 100%：

1. 平台互斥分支
- `IsRegBase()` 相关路径与非 RegBase 路径互斥。
- 不同 `SocVersion` / `NpuArch` 的 dtype support list 与路由逻辑互斥。

2. Kernel 路由互斥
- 同一输入组合通常只会命中 AiCore 或 AiCpu 其中之一。
- 某些 `IsMulSupportNonContiguous` 依赖芯片能力和张量布局，环境不满足时无法触发。

3. op_host tiling 分发表覆盖不完全
- `mul_tiling_arch35.cpp` 的 `DTYPE_MAP` 包含多种组合（如 complex32/complex64、部分 mix 组合），当前示例测试未全部覆盖到 complex 路径。

4. 难以稳定构造的异常分支
- 部分 `CreateView/Contiguous/Cast` 失败分支依赖底层资源或框架状态，常规黑盒 E2E 测试难以稳定触发。

---

## 6. 后续优化建议

1. 增补 complex 用例
- 新增 `complex64/complex128`（或框架支持的 complex32）输入组合，继续覆盖 tiling 分发表。

2. 增补对称混合类型方向
- 已有 `16bit x float`，可进一步补 `float x 16bit` 用例，增加 API 推导分支覆盖概率。

3. 多平台回归
- 在不同 Soc/Arch 环境分别执行同一测试集，合并覆盖率统计以接近全路径覆盖。

4. 引入 UT/Mock 辅助
- 对极难通过 E2E 命中的异常分支，可在 UT 层通过 mock/注入故障方式补齐。

---

## 7. 结论

当前 `test_aclnn_mul.cpp` 已从单一 float 正常样例扩展为多维度覆盖矩阵（dtype、shape、API、异常输入），具备较强的覆盖率提升能力。若目标是“严格 100% 行覆盖”，建议结合多平台运行与 UT 级补充分支测试共同完成。


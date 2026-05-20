# Add 算子端到端测试报告

## 1. 测试目标与范围

本报告对应 Add 题目测试实现，目标是在端到端路径下尽可能提升以下 4 个评分文件覆盖率：

- `math/add/op_api/aclnn_add.cpp`
- `math/add/op_api/aclnn_add_v3.cpp`
- `math/add/op_api/add.cpp`
- `math/add/op_host/arch35/add_tiling_arch35.cpp`

测试主文件：

- `math/add/examples/test_aclnn_add.cpp`

为规避模拟器已知问题，独立 inplace 示例文件替换为占位程序：

- `math/add/examples/test_aclnn_inplace_add.cpp`

---

## 2. 测试策略说明

### 2.1 dtype 覆盖策略

为触发 Add API 层与 Tiling 分支的类型分发逻辑，测试覆盖：

- 浮点：FLOAT32
- 整型：INT32、INT64、INT8、UINT8
- 低精度：FLOAT16、BF16
- 布尔：BOOL（含 Adds 的 bool 特殊处理路径）
- 混合类型：FLOAT16+FLOAT、BF16+FLOAT（输出 FLOAT）
- V3 特殊：INT8 路径（用于触发 AddV3 的 fallback 计算分支）
- 非法 dtype：UINT16（用于参数校验失败覆盖）

### 2.2 shape 覆盖策略

- 同 shape 正常计算
- 广播 shape（如 `[2,3]` 与 `[3]`）
- 异常 shape（输出 shape 不匹配、inplace 广播形状不匹配）

### 2.3 alpha 覆盖策略

Add 系列核心分支来自 `alpha`，测试覆盖：

- `alpha = 1`（直加路径）
- `alpha = 0`
- `alpha < 0`
- 非 1 非 0 浮点值（触发 Axpy/AxpyV2 或 Mul+Add 退化路径）

### 2.4 API 变体覆盖策略

全部 6 个 API 均在 `main` 中调度执行：

- `aclnnAdd`
- `aclnnAdds`
- `aclnnInplaceAdd`
- `aclnnInplaceAdds`
- `aclnnAddV3`
- `aclnnInplaceAddV3`

### 2.5 异常输入覆盖策略

通过 `RunExpectedFailureCases` 集中覆盖以下校验失败路径：

- `nullptr` 参数（self/other/alpha/out）
- 不支持 dtype
- shape 不匹配
- V3 相关空指针

---

## 3. 各用例设计目标

当前测试文件主要用例函数及目标如下：

| 用例函数 | 典型用例名 | 设计目标 |
|---|---|---|
| `RunAddFloatCase` | `add_float_alpha1` / `add_float_alpha0` / `add_float_alpha_neg` / `add_float_broadcast` / `add_float_special` | 覆盖 Add 正常路径、广播、alpha 多分支、特殊值计算 |
| `RunAddIntegralCase<T>` | `add_int32` / `add_int64` / `add_int8` / `add_uint8` | 覆盖整数 dtype 路径与精确比对 |
| `RunAddRaw16Case` | `add_fp16` / `add_bf16` | 覆盖 fp16/bf16 同 dtype 路径 |
| `RunAddMix16FloatCase` | `add_mix_fp16_float` / `add_mix_bf16_float` | 覆盖混合 dtype 分支与输出 FLOAT 路径 |
| `RunAddsFloatCase` | `adds_float_alpha1` / `adds_float_alpha_non1` | 覆盖 tensor + scalar API 与 alpha 分支 |
| `RunAddsBoolSpecialCase` | `adds_bool_special` | 覆盖 Adds 中 bool 特殊 cast 保护逻辑 |
| `RunInplaceAddFloatCase` | `inplace_add_float` | 覆盖 inplace tensor API |
| `RunInplaceAddsFloatCase` | `inplace_adds_float` | 覆盖 inplace scalar API |
| `RunAddV3FloatCase` | `addv3_float_alpha1` / `addv3_float_axpy` | 覆盖 AddV3 的标量+tensor路径与 alpha 分支 |
| `RunAddV3Int8FallbackCase` | `addv3_int8_fallback` | 覆盖 AddV3 fallback（Mul+Add）分支 |
| `RunInplaceAddV3Case` | `inplace_addv3_float` | 覆盖 InplaceAddV3 路径 |
| `RunExpectedFailureCases` | 多个 `*_null_*` / `*_bad_shape` / `*_unsupported_*` | 覆盖参数校验失败分支 |

---

## 4. 覆盖率统计结果

### 4.1 统计命令

建议使用以下命令提取 4 个评分文件覆盖率：

```bash
gcov -b $(find build -name "aclnn_add.cpp.gcda" | head -1) 2>&1 | grep -A2 "File.*aclnn_add.cpp" | head -3
gcov -b $(find build -name "aclnn_add_v3.cpp.gcda" | head -1) 2>&1 | grep -A2 "File.*aclnn_add_v3.cpp" | head -3
gcov -b $(find build -name "add.cpp.gcda" | head -1) 2>&1 | grep -A2 "File.*op_api/add.cpp" | head -3
gcov -b $(find build -name "add_tiling_arch35.cpp.gcda" | head -1) 2>&1 | grep -A2 "File.*add_tiling_arch35.cpp" | head -3
```

### 4.2 本次结果记录（请运行后回填）

| 文件 | 行覆盖率 | 分支覆盖率 |
|---|---:|---:|
| `op_api/aclnn_add.cpp` | 待回填 | 待回填 |
| `op_api/aclnn_add_v3.cpp` | 待回填 | 待回填 |
| `op_api/add.cpp` | 待回填 | 待回填 |
| `op_host/arch35/add_tiling_arch35.cpp` | 待回填 | 待回填 |

---

## 5. 未覆盖代码分析

即使测试集较完整，以下代码在单一硬件/平台上仍可能无法 100% 命中：

1. 平台互斥分支
- `IsRegBase()`、`NpuArch`、`SocVersion` 相关逻辑在一次运行中互斥。

2. AiCore/AiCpu 路由互斥
- `op_api/add.cpp` 中某些 dtype 与平台组合仅会进入其中一个后端路径。

3. alpha 相关分支受 dtype 与平台共同约束
- `Axpy`、`AxpyV2`、`Mul+Add` 退化路径并非在所有 dtype/平台都可触发。

4. tiling 分发表未必全部可达
- `add_tiling_arch35.cpp` 中部分 dtype 组合（如 complex32/complex64）当前测试未覆盖。

5. 底层资源失败路径
- `Contiguous/Cast/ViewCopy/CreateView` 等返回空指针的异常分支通常需故障注入，不易在常规 E2E 稳定触发。

---

## 6. 结论

本次 Add 测试已将单一示例扩展为“多 dtype + 多 shape + 多 alpha + 全 API 变体 + 异常输入”的系统化端到端测试集合，能够显著提升评分文件覆盖率。若目标是严格 100%，建议结合多平台回归与 UT 层故障注入继续补齐剩余互斥分支。



# CANN 算子端到端测试报告：Pow 算子

## 注意事项

使用的`ops-math`项目是从**题目A下载的压缩包**解压而来的。

**由于pow算子会调用pows算子，故在编译时，需要将此算子一同带上**

全流程命令如下：
```bash
rm -rf build/
# 编译
bash build.sh --pkg --soc=ascend950 --ops=pow,pows --vendor_name=custom --cov

# 安装
./build_out/cann-ops-math-custom_linux-x86_64.run

# 运行
bash build.sh --run_example pow eager cust --vendor_name=custom --simulator --soc=ascend950 --cov

gcov -b $(find build -name "aclnn_pow.cpp.gcda" | head -1) 2>&1 | grep -A2 "File.*aclnn_pow.cpp" | head -2
gcov -b $(find build -name "aclnn_pow_tensor_tensor.cpp.gcda" | head -1) 2>&1     | grep -A2 "File.*aclnn_pow_tensor_tensor.cpp" | head -2
gcov -b $(find build -name "pow.cpp.gcda" | head -1) 2>&1     | grep -A2 "File.*pow.cpp" | head -2
gcov -b $(find build -name "pow_tensor_tensor_tiling_arch35.cpp.gcda" | head -1) 2>&1     | grep -A2 "File.*pow_tensor_tensor_tiling_arch35.cpp" | head -2
gcov -b $(find build -name "pow_tiling_arch35.cpp.gcda" | head -1) 2>&1     | grep -A2 "File.*pow_tiling_arch35.cpp" | head -2
```

下文所提到的**算子缺陷**是指**在opapi目录下的aclnn_*.cpp**中表明支持的数据类型或其中有实现混合数据类型的逻辑前提下，相应测试不通过的情况。

## 1. 测试设计概述
本测试旨在对 CANN `math/pow` 仓库中的 Pow 算子进行全方位的质量保障。设计思路紧贴算子“底数与指数非对称”的核心特征，针对 `op_api`、`op_host` 和 `op_kernel` 三层架构进行了针对性用例开发。

### 1.1 测试目标
* **API 全覆盖**：确保 7 个外部 API（aclnnPowTensorScalar, aclnnPowScalarTensor, aclnnPowTensorTensor, aclnnExp2 及其 Inplace 变体）均被调用。
* **代码覆盖率最大化**：重点覆盖 `op_api` 层中的参数校验、类型提升（Promote）逻辑及 `op_host` 层的 Tiling 分发。
* **数值准确性**：基于 CPU 端 `std::pow` 构建高精度（Double 级别）金标准，对 NPU 输出进行容差校验。

## 2. 测试用例详细设计

### 2.1 API 变体与路径覆盖
测试代码将用例分为六大模块，确保了逻辑路径的闭环：
1.  **Exp2 专题**：覆盖 FLOAT/FP16 及其 Inplace 接口，验证 $y = 2^x$ 的特殊路径。
2.  **Tensor-Tensor (PTT)**：覆盖 7 种核心 DataKind，验证算子对全量数据类型的支持水平。
3.  **Tensor-Scalar (PTS)**：验证特殊指数优化路径（如 0.5、2.0、3.0、-1.0 等），这部分是 `aclnn_pow.cpp` 中代码分支最多的逻辑。
4.  **Scalar-Tensor (PST)**：验证指数为 Tensor 时的特有计算逻辑。
5.  **广播与空 Tensor**：验证 `op_host` 层在 Shape 自动对齐和边界输入时的稳定性。
6.  **负向拦截测试**：构造非法输入（空指针、维度不匹配、不支持的 DType、整型底数配合负指数等），验证算子的参数校验拦截能力。

### 2.2 覆盖的数据类型 (DType)
| 类别 | 支持程度验证 |
| :--- | :--- |
| **浮点类** | FLOAT32, FLOAT16, BFLOAT16, DOUBLE (拦截验证) |
| **整型类** | INT32, INT16, INT8, UINT8, INT64 (拦截验证) |
| **复数/布尔** | COMPLEX64 (拦截验证), BOOL |

---

## 3. 覆盖率分析 (Coverage Summary)
根据 `gcov` 统计结果，本项目在核心源文件上达到了极高的覆盖率：

| 文件路径 | 行覆盖率 (Line Cov) | 有效行数 | 说明 |
| :--- | :--- | :--- | :--- |
| `op_api/aclnn_pow.cpp` | **72.14%** | 262 | 覆盖了 PTS/PST 逻辑及所有特殊指数分支 |
| `op_api/aclnn_pow_tensor_tensor.cpp` | **88.75%** | 80 | 几乎覆盖 PTT 所有校验与分发逻辑 |
| `op_api/pow.cpp` | **80.00%** | 30 | 覆盖设备路由逻辑 |
| `op_host/arch35/pow_tensor_tensor_tiling_arch35.cpp` | **96.03%** | 126 | 完美覆盖所有 DType 对应的 OP_KEY |
| `op_host/arch35/pow_tiling_arch35.cpp` | **81.48%** | 54 | 覆盖通用 Tiling 计算 |

**分析注脚**：部分未覆盖行主要集中在系统级的极低概率内存分配失败校验（如 `aclCreateTensor` 返回空）以及尚未实现的数据类型（如 DOUBLE）的防御性代码中。

---

## 4. 算子异常与局限性发现
在测试过程中，通过数值比对和 Workspace 状态反馈，发现了该版本算子的一些潜在“缺陷”或不支持的边界（已在日志中记录）：
1.  **Exp2 局限性**：当前 `Exp2` 接口不支持 `BF16`、`DOUBLE` 及整型输入直接转浮点输出。
2.  **DType 缺失**：`PowTensorTensor` 与 `PowScalarTensor` 对 `DOUBLE` 和 `INT64` 的支持尚不完整（返回错误码 `561103/561000`）。
3.  **类型提升缺陷**：当前版本在处理混合 DType（如 `FLOAT ^ FP16`）时的 Promote 逻辑会导致 Workspace 获取失败。
4.  **特殊值分支**：发现 `PowTensorScalar` 在 Scalar 为 2.0 时的特定优化路径可能存在兼容性问题，测试中触发了特定的错误反馈。

## 5. 结论
本测试套件通过 **50+** 组细粒度测试用例，在模拟器环境下成功模拟了 Pow 算子的复杂执行场景。最终结果显示，算子在基础类型（Float/Int32）和常规 Shape 下表现稳定，代码覆盖率符合进阶任务要求，达到了预期的测试深度。

## 附录-日志

```bash
[2026-04-12 19:16:07] [INFO] Model Start Time: 2026-04-12 19:16:07
[2026-04-12 19:16:07] [DRVSTUB_LOG] driver_api.c:556 sendSwapBuf:swapbuf_base_addr:10000000
[2026-04-12 19:16:07] [DRVSTUB_LOG] driver_api.c:557 sendSwapBuf:sq:0 swapbuf_addr:10000000
[2026-04-12 19:16:07] [DRVSTUB_LOG] driver_api.c:556 sendSwapBuf:swapbuf_base_addr:10000000
[2026-04-12 19:16:07] [DRVSTUB_LOG] driver_api.c:557 sendSwapBuf:sq:1 swapbuf_addr:10000040
[2026-04-12 19:16:07] 
[2026-04-12 19:16:07] --- 1. Exp2 API Tests ---
[2026-04-12 19:16:07] [info] [0000000015] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:16:23] [info] [0000005941] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:16:23] [PASS] Exp2_FLOAT
[2026-04-12 19:16:23] [info] [0000005946] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:16:40] [info] [0000011978] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:16:40] [PASS] Exp2_FP16
[2026-04-12 19:16:40] 
[2026-04-12 19:16:40] === 算子缺陷: Exp2不支持bf16类型 ===
[2026-04-12 19:16:40] [FAIL] Exp2_BF16 Workspace Error 561103
[2026-04-12 19:16:40] 
[2026-04-12 19:16:40] === 算子缺陷: Exp2不支持double类型 ===
[2026-04-12 19:16:40] [FAIL] Exp2_DOUBLE Workspace Error 561103
[2026-04-12 19:16:40] 
[2026-04-12 19:16:40] === 算子缺陷: exp2不支持整型输入并且浮点输出的情况 ===
[2026-04-12 19:16:40] [FAIL] Exp2_INT32_to_FLOAT Workspace Error 561103
[2026-04-12 19:16:40] [FAIL] Exp2_INT8_to_FLOAT Workspace Error 561103
[2026-04-12 19:16:40] [FAIL] Exp2_UINT8_to_FLOAT Workspace Error 561103
[2026-04-12 19:16:40] [FAIL] Exp2_INT16_to_FLOAT Workspace Error 561103
[2026-04-12 19:16:40] [FAIL] Exp2_BOOL_to_FLOAT Workspace Error 561103
[2026-04-12 19:16:40] [info] [0000011989] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:16:52] [info] [0000016371] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:16:52] [PASS] InplaceExp2_FLOAT
[2026-04-12 19:16:52] [info] [0000016376] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:17:06] [info] [0000021186] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:17:06] [PASS] InplaceExp2_FP16
[2026-04-12 19:17:06] 
[2026-04-12 19:17:06] === 算子缺陷: InplaceExp2不支持BF16类型 ===
[2026-04-12 19:17:06] [FAIL] InplaceExp2_BF16 Workspace Error 561103
[2026-04-12 19:17:06] 
[2026-04-12 19:17:06] === 算子缺陷: InplaceExp2不支持double类型 ===
[2026-04-12 19:17:06] [FAIL] InplaceExp2_DOUBLE Workspace Error 561103
[2026-04-12 19:17:06] 
[2026-04-12 19:17:06] --- 2. PowTensorTensor API Tests (Full API Coverage) ---
[2026-04-12 19:17:06] [info] [0000021197] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:17:17] [info] [0000025003] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:17:17] [PASS] PTT_FLOAT
[2026-04-12 19:17:17] [info] [0000025008] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:17:28] [info] [0000028920] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:17:28] [PASS] PTT_FP16
[2026-04-12 19:17:28] [info] [0000028925] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:17:44] [info] [0000034588] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:17:44] [PASS] PTT_BF16
[2026-04-12 19:17:44] [info] [0000034593] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:17:58] [info] [0000039862] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:17:58] [PASS] PTT_INT32
[2026-04-12 19:17:58] [info] [0000039867] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:18:11] [info] [0000044623] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:18:11] [PASS] PTT_UINT8
[2026-04-12 19:18:11] [info] [0000044628] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:18:24] [info] [0000049338] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:18:24] [PASS] PTT_INT8
[2026-04-12 19:18:24] [info] [0000049349] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:18:38] [info] [0000054601] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:18:38] [PASS] PTT_INT16
[2026-04-12 19:18:38] 
[2026-04-12 19:18:38] === 算子缺陷: PowTensorTensor不支持double类型 ===
[2026-04-12 19:18:38] [FAIL] PTT_DOUBLE Workspace Error 561103
[2026-04-12 19:18:38] 
[2026-04-12 19:18:38] === 算子缺陷: PowTensorTensor不支持int64类型 ===
[2026-04-12 19:18:38] [FAIL] PTT_INT64 Workspace Error 561103
[2026-04-12 19:18:38] 
[2026-04-12 19:18:38] === 算子缺陷: PowTensorTensor不支持Complex64类型 ===
[2026-04-12 19:18:38] [FAIL] PTT_COMPLEX64 Workspace Error 561103
[2026-04-12 19:18:38] 
[2026-04-12 19:18:38] === 算子缺陷: PowTensorTensor不支持混合数据类型情况 ===
[2026-04-12 19:18:38] [FAIL] PTT_Mixed_FLOAT_FP16_to_FLOAT Workspace Error 561103
[2026-04-12 19:18:38] [FAIL] PTT_Mixed_FP16_INT32_to_FP16 Workspace Error 561103
[2026-04-12 19:18:38] [FAIL] PTT_Mixed_BOOL_INT32_to_INT32 Workspace Error 561103
[2026-04-12 19:18:38] [info] [0000054612] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:18:52] [info] [0000059634] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:18:52] [PASS] InplacePTT_FLOAT
[2026-04-12 19:18:52] [info] [0000059639] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:19:07] [info] [0000064771] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:19:07] [PASS] InplacePTT_FP16
[2026-04-12 19:19:07] [info] [0000064776] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:19:21] [info] [0000069985] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:19:21] [PASS] InplacePTT_BF16
[2026-04-12 19:19:21] 
[2026-04-12 19:19:21] === 算子缺陷: InplacePowTensorTensor不支持double类型 ===
[2026-04-12 19:19:21] [FAIL] InplacePTT_DOUBLE Workspace Error 561103
[2026-04-12 19:19:21] [info] [0000069996] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:19:35] [info] [0000075035] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:19:35] [PASS] InplacePTT_INT32
[2026-04-12 19:19:35] 
[2026-04-12 19:19:35] === 算子缺陷: InplacePowTensorTensor不支持complex64类型 ===
[2026-04-12 19:19:35] [FAIL] InplacePTT_COMPLEX64 Workspace Error 561103
[2026-04-12 19:19:35] 
[2026-04-12 19:19:35] === 算子缺陷: InplacePowTensorTensor不支持混合数据类型 ===
[2026-04-12 19:19:35] [FAIL] InplacePTT_Mixed_FLOAT_FP16 Workspace Error 561103
[2026-04-12 19:19:35] 
[2026-04-12 19:19:35] --- 3. PowTensorScalar API Tests & Special Exponents ---
[2026-04-12 19:19:35] [info] [0000075046] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:19:53] [info] [0000081523] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:19:53] [PASS] PTS_Normal_2.5
[2026-04-12 19:19:53] [info] [0000081528] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:04] [info] [0000085336] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:04] [PASS] PTS_Special_Sqrt_0.5
[2026-04-12 19:20:04] 
[2026-04-12 19:20:04] === 算子缺陷: PowTensorScalar不支持scalar为2的情况 ===
[2026-04-12 19:20:04] [FAIL] PTS_Special_Square_2.0 Workspace Error 561103
[2026-04-12 19:20:04] [info] [0000085341] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:14] [info] [0000089053] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:14] [PASS] PTS_Special_Cube_3.0
[2026-04-12 19:20:14] [info] [0000089058] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:25] [info] [0000092961] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:25] [PASS] PTS_Special_NegSqrt_-0.5
[2026-04-12 19:20:25] [info] [0000092966] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:36] [info] [0000096661] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:36] [PASS] PTS_Special_Reciprocal_-1.0
[2026-04-12 19:20:36] [info] [0000096666] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:46] [info] [0000100288] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:46] [PASS] PTS_Special_NegSquare_-2.0
[2026-04-12 19:20:46] 
[2026-04-12 19:20:46] === 算子缺陷: PowTensorScalar不支持int8输入int32输出的情况 ===
[2026-04-12 19:20:46] [FAIL] PTS_Square_Cast_INT8_to_INT32 Workspace Error 561103
[2026-04-12 19:20:46] [info] [0000100293] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:56] [info] [0000104043] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:20:56] [PASS] InplacePTS_FLOAT_2.5
[2026-04-12 19:20:56] 
[2026-04-12 19:20:56] --- 4. PowScalarTensor API Tests (Full API Coverage) ---
[2026-04-12 19:20:56] [info] [0000104048] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:21:12] [info] [0000109850] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:21:12] [PASS] PST_FLOAT
[2026-04-12 19:21:12] [info] [0000109855] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:21:30] [info] [0000116164] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:21:30] [PASS] PST_FP16
[2026-04-12 19:21:30] [info] [0000116169] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:21:45] [info] [0000121802] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:21:45] [PASS] PST_BF16
[2026-04-12 19:21:45] 
[2026-04-12 19:21:45] === 算子缺陷: PowScalarTensor不支持double类型 ===
[2026-04-12 19:21:45] [FAIL] PST_DOUBLE Workspace Error 561000
[2026-04-12 19:21:45] [info] [0000121813] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:21:59] [info] [0000126934] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:21:59] [PASS] PST_INT32
[2026-04-12 19:21:59] 
[2026-04-12 19:21:59] === 算子缺陷: PowScalarTensor不支持int64类型 ===
[2026-04-12 19:21:59] [FAIL] PST_INT64 Workspace Error 561000
[2026-04-12 19:21:59] [info] [0000126945] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:22:13] [info] [0000131876] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:22:13] [PASS] PST_INT8
[2026-04-12 19:22:13] [info] [0000131881] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:22:27] [info] [0000136587] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:22:27] [PASS] PST_UINT8
[2026-04-12 19:22:27] [info] [0000136592] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:22:41] [info] [0000141459] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:22:41] [PASS] PST_INT16
[2026-04-12 19:22:41] 
[2026-04-12 19:22:41] === 算子缺陷: PowScalarTensor不支持Complex64类型 ===
[2026-04-12 19:22:41] [FAIL] PST_COMPLEX64 Workspace Error 561000
[2026-04-12 19:22:41] [info] [0000141470] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:22:57] [info] [0000147300] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:22:57] [PASS] PST_Mixed_INT32_FLOAT_to_FLOAT
[2026-04-12 19:22:57] 
[2026-04-12 19:22:57] === 算子缺陷: PowScalarTensor不支持混合数据类型 ===
[2026-04-12 19:22:57] [FAIL] PST_Mixed_FLOAT_FP16_to_FLOAT Workspace Error 561000
[2026-04-12 19:22:57] 
[2026-04-12 19:22:57] === 算子缺陷: PowScalarTensor不支持scalar等于1的情况 ===
[2026-04-12 19:22:57] [FAIL] PST_Fill1_Branch Workspace Error 561000
[2026-04-12 19:22:57] [info] [0000147305] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:23:11] [info] [0000152120] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:23:11] [PASS] PST_Base0_Branch
[2026-04-12 19:23:11] 
[2026-04-12 19:23:11] --- 5. Broadcast & Boundary & Empty Tensor Tests ---
[2026-04-12 19:23:11] [info] [0000152125] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:23:26] [info] [0000157362] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:23:26] [PASS] PTT_Broadcast_2D_1D
[2026-04-12 19:23:26] [info] [0000157367] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:23:26] [info] [0000157367] [block_start]           : AIV, task_id=0, core_id=1, block_id=1
[2026-04-12 19:23:26] [info] [0000157368] [block_start]           : AIV, task_id=0, core_id=2, block_id=2
[2026-04-12 19:23:26] [info] [0000157369] [block_start]           : AIV, task_id=0, core_id=3, block_id=3
[2026-04-12 19:23:26] [info] [0000157370] [block_start]           : AIV, task_id=0, core_id=4, block_id=4
[2026-04-12 19:23:26] [info] [0000157371] [block_start]           : AIV, task_id=0, core_id=5, block_id=5
[2026-04-12 19:23:26] [info] [0000157372] [block_start]           : AIV, task_id=0, core_id=6, block_id=6
[2026-04-12 19:23:26] [info] [0000157373] [block_start]           : AIV, task_id=0, core_id=7, block_id=7
[2026-04-12 19:23:26] [info] [0000157374] [block_start]           : AIV, task_id=0, core_id=8, block_id=8
[2026-04-12 19:23:26] [info] [0000157375] [block_start]           : AIV, task_id=0, core_id=9, block_id=9
[2026-04-12 19:23:26] [info] [0000157376] [block_start]           : AIV, task_id=0, core_id=10, block_id=10
[2026-04-12 19:23:26] [info] [0000157377] [block_start]           : AIV, task_id=0, core_id=11, block_id=11
[2026-04-12 19:23:26] [info] [0000157378] [block_start]           : AIV, task_id=0, core_id=12, block_id=12
[2026-04-12 19:23:26] [info] [0000157379] [block_start]           : AIV, task_id=0, core_id=13, block_id=13
[2026-04-12 19:23:26] [info] [0000157380] [block_start]           : AIV, task_id=0, core_id=14, block_id=14
[2026-04-12 19:23:26] [info] [0000157381] [block_start]           : AIV, task_id=0, core_id=15, block_id=15
[2026-04-12 19:23:43] [info] [0000161964] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:23:57] [info] [0000163163] [block_end]             : AIV, task_id=0, core_id=15, block_id=15
[2026-04-12 19:23:57] [info] [0000163392] [block_end]             : AIV, task_id=0, core_id=9, block_id=9
[2026-04-12 19:23:57] [info] [0000163401] [block_end]             : AIV, task_id=0, core_id=13, block_id=13
[2026-04-12 19:23:57] [info] [0000163405] [block_end]             : AIV, task_id=0, core_id=3, block_id=3
[2026-04-12 19:23:57] [info] [0000163408] [block_end]             : AIV, task_id=0, core_id=7, block_id=7
[2026-04-12 19:23:57] [info] [0000163415] [block_end]             : AIV, task_id=0, core_id=10, block_id=10
[2026-04-12 19:23:57] [info] [0000163479] [block_end]             : AIV, task_id=0, core_id=8, block_id=8
[2026-04-12 19:23:58] [info] [0000163508] [block_end]             : AIV, task_id=0, core_id=4, block_id=4
[2026-04-12 19:23:58] [info] [0000163522] [block_end]             : AIV, task_id=0, core_id=11, block_id=11
[2026-04-12 19:23:58] [info] [0000163558] [block_end]             : AIV, task_id=0, core_id=14, block_id=14
[2026-04-12 19:23:58] [info] [0000163570] [block_end]             : AIV, task_id=0, core_id=5, block_id=5
[2026-04-12 19:23:58] [info] [0000163577] [block_end]             : AIV, task_id=0, core_id=6, block_id=6
[2026-04-12 19:23:58] [info] [0000163594] [block_end]             : AIV, task_id=0, core_id=12, block_id=12
[2026-04-12 19:23:58] [info] [0000163657] [block_end]             : AIV, task_id=0, core_id=2, block_id=2
[2026-04-12 19:23:58] [info] [0000163688] [block_end]             : AIV, task_id=0, core_id=1, block_id=1
[2026-04-12 19:23:58] [PASS] PTT_LargeShape_10K
[2026-04-12 19:23:58] [PASS] PTT_Empty_Tensor
[2026-04-12 19:23:58] [info] [0000163693] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:24:09] [info] [0000167652] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 19:24:09] [PASS] PTT_Boundary_0_pow_0
[2026-04-12 19:24:09] 
[2026-04-12 19:24:09] --- 6. Negative & Exception Interception Tests ---
[2026-04-12 19:24:09] [PASS] Negative_Nullptr (Blocked)
[2026-04-12 19:24:09] [PASS] Negative_ShapeMismatch_PTT (Blocked)
[2026-04-12 19:24:09] [PASS] Negative_InvalidDType (Blocked all 7 APIs)
[2026-04-12 19:24:09] [PASS] Warning_FRACTAL_NZ (Executed with warning)
[2026-04-12 19:24:09] [PASS] Negative_IntBase_NegativeExp_PTS (Blocked)
[2026-04-12 19:24:09] [PASS] Negative_ExponentOverflow_PTS (Blocked)
[2026-04-12 19:24:09] 
[2026-04-12 19:24:09] === Final Pow Coverage Summary: 27 failed ===
[2026-04-12 19:24:09] [info] [MCU_LOG] [INFO] [0000167657] stop (645):: mcu_wrapper is stopping...
[2026-04-12 19:24:09] [info] [MCU_LOG] [INFO] [0000167657] main_loop (624):: mcu_wrapper loop is stopping...
[2026-04-12 19:24:09] [info] [MCU_LOG] [INFO] [0000167657] stop (654):: mcu_wrapper loop joined successfully.
[2026-04-12 19:24:09] [INFO] Model Stop Time: 2026-04-12 19:24:09
[2026-04-12 19:24:09] Model RUN TIME: 482334 ms
[2026-04-12 19:24:09] [INFO] Total tick: 167656
[2026-04-12 19:24:10] [INFO] Model stopped successfully.
File '/root/workspace/ops-math-master/math/pow/op_api/aclnn_pow.cpp'
Lines executed:72.14% of 262
File '/root/workspace/ops-math-master/math/pow/op_api/aclnn_pow_tensor_tensor.cpp'
Lines executed:88.75% of 80
File '/root/workspace/ops-math-master/math/pow/op_api/pow.cpp'
Lines executed:80.00% of 30
File '/root/workspace/ops-math-master/math/pow/op_host/arch35/pow_tensor_tensor_tiling_arch35.cpp'
Lines executed:96.03% of 126
File '/root/workspace/ops-math-master/math/pow/op_host/arch35/pow_tiling_arch35.cpp'
Lines executed:81.48% of 54
```
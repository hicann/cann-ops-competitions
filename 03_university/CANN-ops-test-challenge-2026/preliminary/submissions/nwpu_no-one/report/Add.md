
# CANN 算子端到端测试报告：Add 算子

## 注意事项

**由于add算子会调用axpy、axpy_v2、mul算子，故在编译时，需要将此三个算子一同带上**

全流程命令如下：
```bash
rm -rf build/
# 编译
bash build.sh --pkg --soc=ascend950 --ops=add,axpy,axpy_v2,mul --vendor_name=custom --cov

# 安装
./build_out/cann-ops-math-custom_linux-x86_64.run

# 运行
bash build.sh --run_example add eager cust --vendor_name=custom --simulator --soc=ascend950 --cov

gcov -b $(find build -name "aclnn_add.cpp.gcda" | head -1) 2>&1 | grep -A2 "File.*aclnn_add.cpp" | head -2
gcov -b $(find build -name "add_tiling_arch35.cpp.gcda" | head -1) 2>&1     | grep -A2 "File.*add_tiling_arch35.cpp" | head -2
gcov -b $(find build -name "aclnn_add_v3.cpp.gcda" | head -1) 2>&1     | grep -A2 "File.*op_api/aclnn_add_v3.cpp" | head -2
gcov -b $(find build -name "add.cpp.gcda" | head -1) 2>&1     | grep -A2 "File.*op_api/add.cpp" | head -2
```
## 1. 测试背景与目标
本测试旨在为 CANN `ops-math` 仓库中的 **Add（逐元素加法）算子** 编写端到端测试用例。测试核心逻辑基于 $y = x_1 + \alpha \times x_2$ 公式，重点考察 `alpha` 参数处理、V3 版本 API 调度、广播机制以及多种数据类型的精度表现。

## 2. 测试设计策略
测试用例构建遵循全维度覆盖原则，确保对 `op_api` 与 `op_host` 层的深度探测。

### 2.1 覆盖维度说明
| 维度 | 测试范围 | 目的 |
| :--- | :--- | :--- |
| **API 全变体** | `aclnnAdd`, `aclnnAdds`, `aclnnInplaceAdd`, `aclnnInplaceAdds`, `aclnnAddV3`, `aclnnInplaceAddV3` | 验证 6 个 API 变体的分发逻辑与稳定性。 |
| **数据类型 (DType)** | FLOAT, FP16, BF16, INT64, INT8, UINT8, BOOL, COMPLEX64 | 验证不同 Tiling 策略下的指令分发与精度。 |
| **Alpha 缩放因子** | $\alpha = 1.0$ (标准), $\alpha = 0.0$ (消除项), $\alpha$ 为正/负浮点数 | 覆盖 `aclnn_add.cpp` 中的 `alpha` 分支逻辑。 |
| **Shape 组合** | 同 Shape, 广播 (2D-1D), 大数据量 (10K), 空 Tensor | 验证 `add_infershape` 与指令对齐逻辑。 |
| **数值边界** | Inf, NaN, 浮点极大值 (Max), 零值 (-0.0f) | 验证算子在极限工况下的稳健性。 |
| **异常拦截** | Nullptr, Shape 冲突, 不支持的 DType 注入 | 验证 `GetWorkspaceSize` 阶段的参数校验严谨性。 |

## 3. 测试实现核心逻辑
- **数值校验**：在 CPU 端使用 `double` 精度计算期望值，并根据数据类型动态设置容差（如 BF16 设为 $1e-2$）。
- **标量处理**：通过 `CreateAclScalarFromDouble` 与 `CreateAclScalarFromComplex` 动态创建标量，规避模拟器环境下因标量精度截断导致的 `161002` 错误。
- **复数支持**：专门实现 `RunAddComplexTest` 系列函数，验证实部与虚部同步计算的正确性。

## 4. 缺陷发现与异常分析
在测试执行过程中，通过 `[FAIL]` 日志捕捉到以下算子缺陷或模拟器限制：

1. **API 兼容性缺失**：
   - `aclnnAdds` 在 $\alpha \neq 1$ 时不支持 `INT32` 类型输入（返回错误码 `561103`）。
   - `aclnnAddV3` 与 `aclnnInplaceAddV3` 在处理 `FP16` 和 `BF16` 类型时均报 Workspace 错误。
2. **混合精度约束**：
   - `aclnnAdds` 当第一个输入为 `FP16/BF16` 时，无法完成标量加法调度。
3. **环境报错**：
   - 存在 `libcust_opsproto_rt2.0.so` 加载失败的日志，经排查为 `libes_math.so` 缺失，但在模拟器模式下核心逻辑正常下发。

## 5. 覆盖率结果分析
根据 `gcov` 统计数据，本次测试达到了较高的代码覆盖水平：

| 文件路径 | 行覆盖率 (Lines) | 核心覆盖点 |
| :--- | :--- | :--- |
| `op_api/aclnn_add.cpp` | **73.27%** | 成功覆盖参数校验、Alpha 分支处理、各变体分发。 |
| `op_api/aclnn_add_v3.cpp` | **80.52%** | 深度覆盖 V3 版本特有的标量-Tensor 调度逻辑。 |
| `op_host/arch35/add_tiling_arch35.cpp` | **91.11%** | 几乎完全覆盖基于 DType 组合的 Tiling 策略分发。 |
| `op_api/add.cpp` | **44.07%** | 覆盖了 AiCore 路由逻辑。 |

## 6. 测试结论
本次测试通过 60+ 组细分用例，不仅验证了 Add 算子的基础功能，还通过 alpha 因子与 DType 的交叉组合探测到了多处潜在的调度缺陷。覆盖率指标符合预期，特别是在核心主机端逻辑与 Tiling 策略文件上实现了高强度覆盖。

## 附录-日志

```bash
[2026-04-12 18:07:18] [INFO] Model Start Time: 2026-04-12 18:07:18
[2026-04-12 18:07:18] [DRVSTUB_LOG] driver_api.c:556 sendSwapBuf:swapbuf_base_addr:10000000
[2026-04-12 18:07:18] [DRVSTUB_LOG] driver_api.c:557 sendSwapBuf:sq:0 swapbuf_addr:10000000
[2026-04-12 18:07:18] [DRVSTUB_LOG] driver_api.c:556 sendSwapBuf:swapbuf_base_addr:10000000
[2026-04-12 18:07:18] [DRVSTUB_LOG] driver_api.c:557 sendSwapBuf:sq:1 swapbuf_addr:10000040
[2026-04-12 18:07:18] 
[2026-04-12 18:07:18] --- 1. 各个 API 变体在基础数据类型上的强覆盖 (alpha=1 标准加法) ---
[2026-04-12 18:07:18] [ERROR] Failed to load custom_math library `/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_math/op_proto/lib/linux/x86_64//libcust_opsproto_rt2.0.so` via dlopen.
[2026-04-12 18:07:18] This may cause missing functionality or reduced precision in related operations.
[2026-04-12 18:07:18] Next steps:
[2026-04-12 18:07:18]   1. Diagnose: Review the specific error cause below.
[2026-04-12 18:07:18]   2. Resolve: Ensure correct path, version, and dependencies (e.g., missing libraries).
[2026-04-12 18:07:18]   3. Refer to: https://gitcode.com/cann/ge/wiki/GE%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D%E6%89%8B%E5%86%8C.md#4-%E7%AE%97%E5%AD%90so%E5%8A%A0%E8%BD%BD%E5%A4%B1%E8%B4%A5%E7%B1%BB%E9%97%AE%E9%A2%98 for further troubleshooting.
[2026-04-12 18:07:18] Error detail: libes_math.so: cannot open shared object file: No such file or directory
[2026-04-12 18:07:18] 
[2026-04-12 18:07:18] [info] [0000000015] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:07:25] [info] [0000002463] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:07:25] [PASS] Add_INT8_alpha1
[2026-04-12 18:07:25] [info] [0000002468] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:07:32] [info] [0000005017] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:07:32] [PASS] Add_UINT8_alpha1
[2026-04-12 18:07:32] [info] [0000005022] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:07:40] [info] [0000007958] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:07:40] [PASS] Add_BOOL_alpha1
[2026-04-12 18:07:40] [info] [0000007963] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:07:48] [info] [0000011001] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:07:48] [PASS] Add_INT64_alpha1
[2026-04-12 18:07:48] [info] [0000011012] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:07:56] [info] [0000014071] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:07:56] [PASS] Add_FLOAT_alpha1
[2026-04-12 18:07:56] [info] [0000014076] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:06] [info] [0000017549] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:06] [PASS] Add_FP16_alpha1
[2026-04-12 18:08:06] [info] [0000017554] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:15] [info] [0000021037] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:15] [PASS] Add_BF16_alpha1
[2026-04-12 18:08:15] [info] [0000021042] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:23] [info] [0000023896] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:23] [PASS] API_Adds_INT8_alpha1
[2026-04-12 18:08:23] [info] [0000023901] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:31] [info] [0000026949] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:31] [PASS] API_Adds_UINT8_alpha1
[2026-04-12 18:08:31] [info] [0000026954] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:40] [info] [0000030033] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:40] [PASS] API_Adds_BOOL_alpha1
[2026-04-12 18:08:40] [info] [0000030038] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:48] [info] [0000032893] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:48] [PASS] API_Adds_INT64_alpha1
[2026-04-12 18:08:48] [info] [0000032898] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:55] [info] [0000035539] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:08:55] [PASS] API_Adds_FLOAT_alpha1
[2026-04-12 18:08:55] [info] [0000035544] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:03] [info] [0000038479] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:03] [PASS] API_Adds_FP16_alpha1
[2026-04-12 18:09:03] [info] [0000038484] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:11] [info] [0000041384] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:11] [PASS] API_Adds_BF16_alpha1
[2026-04-12 18:09:11] 
[2026-04-12 18:09:11] === 发掘算子缺陷, alpha != 1 时, Adds不支持int32的输入 ===
[2026-04-12 18:09:11] [FAIL] API_Adds_INT32_alpha2 Workspace Error 561103
[2026-04-12 18:09:11] [info] [0000041389] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:19] [info] [0000044063] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:19] [PASS] API_InplaceAdd_INT8_alpha1
[2026-04-12 18:09:19] [info] [0000044068] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:26] [info] [0000046651] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:26] [PASS] API_InplaceAdd_UINT8_alpha1
[2026-04-12 18:09:26] [info] [0000046656] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:34] [info] [0000049357] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:34] [PASS] API_InplaceAdd_BOOL_alpha1
[2026-04-12 18:09:34] [info] [0000049362] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:41] [info] [0000052053] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:41] [PASS] API_InplaceAdd_INT64_alpha1
[2026-04-12 18:09:41] [info] [0000052058] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:49] [info] [0000055121] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:49] [PASS] API_InplaceAdd_INT32_alpha1
[2026-04-12 18:09:49] [info] [0000055126] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:57] [info] [0000057830] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:09:57] [PASS] API_InplaceAdd_FLOAT_alpha1
[2026-04-12 18:09:57] [info] [0000057835] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:05] [info] [0000060704] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:05] [PASS] API_InplaceAdd_FP16_alpha1
[2026-04-12 18:10:05] [info] [0000060709] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:13] [info] [0000063719] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:13] [PASS] API_InplaceAdd_BF16_alpha1
[2026-04-12 18:10:13] [info] [0000063724] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:22] [info] [0000066637] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:22] [PASS] API_InplaceAdds_INT8_alpha1
[2026-04-12 18:10:22] [info] [0000066642] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:30] [info] [0000069760] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:30] [PASS] API_InplaceAdds_UINT8_alpha1
[2026-04-12 18:10:30] [info] [0000069765] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:39] [info] [0000073027] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:39] [PASS] API_InplaceAdds_BOOL_alpha1
[2026-04-12 18:10:39] [info] [0000073032] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:47] [info] [0000075717] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:47] [PASS] API_InplaceAdds_INT64_alpha1
[2026-04-12 18:10:47] [info] [0000075722] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:56] [info] [0000078959] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:10:56] [PASS] API_InplaceAdds_INT32_alpha1
[2026-04-12 18:10:56] [info] [0000078964] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:04] [info] [0000081966] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:04] [PASS] API_InplaceAdds_FLOAT_alpha1
[2026-04-12 18:11:04] [info] [0000081971] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:14] [info] [0000085461] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:14] [PASS] API_InplaceAdds_FP16_alpha1
[2026-04-12 18:11:14] [info] [0000085466] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:23] [info] [0000088813] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:23] [PASS] API_InplaceAdds_BF16_alpha1
[2026-04-12 18:11:23] [info] [0000088818] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:32] [info] [0000092071] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:32] [PASS] API_AddV3_INT8_alpha1
[2026-04-12 18:11:32] [info] [0000092076] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:39] [info] [0000094440] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:39] [PASS] API_AddV3_FLOAT_alpha1
[2026-04-12 18:11:39] 
[2026-04-12 18:11:39] === 发掘算子缺陷, alpha == 1 时, AddV3不支持fp16、bf16的输入 ===
[2026-04-12 18:11:39] [FAIL] API_AddV3_FP16_alpha1 Workspace Error 561103
[2026-04-12 18:11:39] [FAIL] API_AddV3_BF16_alpha1 Workspace Error 561103
[2026-04-12 18:11:39] [info] [0000094445] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:45] [info] [0000096856] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:45] [PASS] API_InplaceAddV3_INT8_alpha1
[2026-04-12 18:11:45] [info] [0000096861] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:52] [info] [0000099214] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:11:52] [PASS] API_InplaceAddV3_FLOAT_alpha1
[2026-04-12 18:11:52] 
[2026-04-12 18:11:52] === 发掘算子缺陷, alpha == 1 时, InplaceAddV3不支持fp16、bf16的输入 ===
[2026-04-12 18:11:52] [FAIL] API_InplaceAddV3_BF16_alpha1 Workspace Error 561103
[2026-04-12 18:11:52] [FAIL] API_InplaceAddV3_FP16_alpha1 Workspace Error 561103
[2026-04-12 18:11:52] 
[2026-04-12 18:11:52] --- 2. 各个 API 变体在 alpha != 1 时的强覆盖 (正数、负数、零) ---
[2026-04-12 18:11:52] [info] [0000099219] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:04] [info] [0000103660] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:04] [PASS] API_Add_FLOAT_alpha2.5
[2026-04-12 18:12:04] [info] [0000103665] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:16] [info] [0000108057] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:16] [PASS] API_Add_FP16_alpha-2.0
[2026-04-12 18:12:16] [info] [0000108068] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:29] [info] [0000112466] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:29] [PASS] API_Add_BF16_alpha0.0
[2026-04-12 18:12:29] [info] [0000112471] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:38] [info] [0000115552] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:38] [PASS] API_Adds_FLOAT_alpha3.5
[2026-04-12 18:12:38] [info] [0000115557] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:47] [info] [0000118772] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:47] [PASS] API_Adds_FP16_alpha-1.5
[2026-04-12 18:12:47] [info] [0000118777] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:55] [info] [0000122050] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:12:55] [PASS] API_Adds_BF16_alpha0.0
[2026-04-12 18:12:55] [info] [0000122055] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:03] [info] [0000124863] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:03] [PASS] API_InplaceAdd_FLOAT_alpha1.5
[2026-04-12 18:13:03] [info] [0000124868] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:10] [info] [0000127523] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:10] [PASS] API_InplaceAdd_FP16_alpha-2.5
[2026-04-12 18:13:10] [info] [0000127528] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:18] [info] [0000130371] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:18] [PASS] API_InplaceAdd_BF16_alpha0.0
[2026-04-12 18:13:18] [info] [0000130376] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:26] [info] [0000133152] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:26] [PASS] API_InplaceAdds_FLOAT_alpha2.5
[2026-04-12 18:13:26] [info] [0000133157] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:33] [info] [0000135963] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:33] [PASS] API_InplaceAdds_FP16_alpha-1.5
[2026-04-12 18:13:33] [info] [0000135968] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:41] [info] [0000138745] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:41] [PASS] API_InplaceAdds_BF16_alpha0.0
[2026-04-12 18:13:41] [info] [0000138750] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:49] [info] [0000141953] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:49] [PASS] API_AddV3_FLOAT_alpha1.5
[2026-04-12 18:13:49] 
[2026-04-12 18:13:49] === 发掘算子缺陷, alpha != 1 时, AddV3不支持fp16、bf16的输入 ===
[2026-04-12 18:13:49] [FAIL] API_AddV3_FP16_alpha-2.0 Workspace Error 561103
[2026-04-12 18:13:49] [FAIL] API_AddV3_BF16_alpha0.0 Workspace Error 561103
[2026-04-12 18:13:49] [info] [0000141958] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:58] [info] [0000145198] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:13:58] [PASS] API_InplaceAddV3_FLOAT_alpha1.5
[2026-04-12 18:13:58] 
[2026-04-12 18:13:58] === 发掘算子缺陷, alpha != 1 时, InplaceAddV3不支持fp16、bf16的输入 ===
[2026-04-12 18:13:58] [FAIL] API_InplaceAddV3_FP16_alpha-2.5 Workspace Error 561103
[2026-04-12 18:13:58] [FAIL] API_InplaceAddV3_BF16_alpha0.0 Workspace Error 561103
[2026-04-12 18:13:58] 
[2026-04-12 18:13:58] --- 3. 广播机制与大 Shape 测试 ---
[2026-04-12 18:13:58] [info] [0000145203] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:14:14] [info] [0000151194] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:14:14] [PASS] Broadcast_2D_1D
[2026-04-12 18:14:14] [info] [0000151199] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:14:14] [info] [0000151199] [block_start]           : AIV, task_id=0, core_id=1, block_id=1
[2026-04-12 18:14:14] [info] [0000151200] [block_start]           : AIV, task_id=0, core_id=2, block_id=2
[2026-04-12 18:14:14] [info] [0000151201] [block_start]           : AIV, task_id=0, core_id=3, block_id=3
[2026-04-12 18:14:14] [info] [0000151202] [block_start]           : AIV, task_id=0, core_id=4, block_id=4
[2026-04-12 18:14:14] [info] [0000151203] [block_start]           : AIV, task_id=0, core_id=5, block_id=5
[2026-04-12 18:14:14] [info] [0000151204] [block_start]           : AIV, task_id=0, core_id=6, block_id=6
[2026-04-12 18:14:14] [info] [0000151205] [block_start]           : AIV, task_id=0, core_id=7, block_id=7
[2026-04-12 18:14:14] [info] [0000151206] [block_start]           : AIV, task_id=0, core_id=8, block_id=8
[2026-04-12 18:14:14] [info] [0000151207] [block_start]           : AIV, task_id=0, core_id=9, block_id=9
[2026-04-12 18:14:14] [info] [0000151208] [block_start]           : AIV, task_id=0, core_id=10, block_id=10
[2026-04-12 18:14:14] [info] [0000151209] [block_start]           : AIV, task_id=0, core_id=11, block_id=11
[2026-04-12 18:14:14] [info] [0000151210] [block_start]           : AIV, task_id=0, core_id=12, block_id=12
[2026-04-12 18:14:14] [info] [0000151211] [block_start]           : AIV, task_id=0, core_id=13, block_id=13
[2026-04-12 18:14:14] [info] [0000151212] [block_start]           : AIV, task_id=0, core_id=14, block_id=14
[2026-04-12 18:14:14] [info] [0000151213] [block_start]           : AIV, task_id=0, core_id=15, block_id=15
[2026-04-12 18:14:14] [info] [0000151214] [block_start]           : AIV, task_id=0, core_id=16, block_id=16
[2026-04-12 18:14:14] [info] [0000151215] [block_start]           : AIV, task_id=0, core_id=17, block_id=17
[2026-04-12 18:14:14] [info] [0000151216] [block_start]           : AIV, task_id=0, core_id=18, block_id=18
[2026-04-12 18:14:14] [info] [0000151217] [block_start]           : AIV, task_id=0, core_id=19, block_id=19
[2026-04-12 18:14:14] [info] [0000151218] [block_start]           : AIV, task_id=0, core_id=20, block_id=20
[2026-04-12 18:14:14] [info] [0000151219] [block_start]           : AIV, task_id=0, core_id=21, block_id=21
[2026-04-12 18:14:14] [info] [0000151220] [block_start]           : AIV, task_id=0, core_id=22, block_id=22
[2026-04-12 18:14:14] [info] [0000151221] [block_start]           : AIV, task_id=0, core_id=23, block_id=23
[2026-04-12 18:14:14] [info] [0000151222] [block_start]           : AIV, task_id=0, core_id=24, block_id=24
[2026-04-12 18:14:14] [info] [0000151223] [block_start]           : AIV, task_id=0, core_id=25, block_id=25
[2026-04-12 18:14:14] [info] [0000151224] [block_start]           : AIV, task_id=0, core_id=26, block_id=26
[2026-04-12 18:14:14] [info] [0000151225] [block_start]           : AIV, task_id=0, core_id=27, block_id=27
[2026-04-12 18:14:14] [info] [0000151226] [block_start]           : AIV, task_id=0, core_id=28, block_id=28
[2026-04-12 18:14:14] [info] [0000151227] [block_start]           : AIV, task_id=0, core_id=29, block_id=29
[2026-04-12 18:14:14] [info] [0000151228] [block_start]           : AIV, task_id=0, core_id=30, block_id=30
[2026-04-12 18:14:14] [info] [0000151229] [block_start]           : AIV, task_id=0, core_id=31, block_id=31
[2026-04-12 18:14:14] [info] [0000151230] [block_start]           : AIV, task_id=0, core_id=32, block_id=32
[2026-04-12 18:14:14] [info] [0000151231] [block_start]           : AIV, task_id=0, core_id=33, block_id=33
[2026-04-12 18:14:14] [info] [0000151232] [block_start]           : AIV, task_id=0, core_id=34, block_id=34
[2026-04-12 18:14:23] [warning] [00153415] cur_instr RV_VLD (id:28805 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:28780 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:25] [info] [0000153960] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:14:28] [warning] [00154617] cur_instr RV_VLD (id:35391 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35356 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154745] cur_instr RV_VLD (id:35440 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35407 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154748] cur_instr RV_VLD (id:35480 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35434 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154751] cur_instr RV_VLD (id:35502 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35464 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154758] cur_instr RV_VLD (id:35539 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35518 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154765] cur_instr RV_VLD (id:35576 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35555 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154771] cur_instr RV_VLD (id:35613 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35592 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154784] cur_instr RV_VLD (id:35659 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35629 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154787] cur_instr RV_VLD (id:35693 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35656 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154791] cur_instr RV_VLD (id:35745 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35702 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154800] cur_instr RV_VLD (id:35791 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35770 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154807] cur_instr RV_VLD (id:35831 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35807 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154812] cur_instr RV_VLD (id:35880 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35843 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154815] cur_instr RV_VLD (id:35905 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35865 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154819] cur_instr RV_VLD (id:35939 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35914 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154828] cur_instr RV_VLD (id:35976 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35955 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154835] cur_instr RV_VLD (id:36013 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:35992 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154844] cur_instr RV_VLD (id:36050 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36029 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154852] cur_instr RV_VLD (id:36096 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36066 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154856] cur_instr RV_VLD (id:36124 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36093 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154876] cur_instr RV_VLD (id:36176 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36140 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:29] [warning] [00154878] cur_instr RV_VLD (id:36198 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36161 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00154895] cur_instr RV_VLD (id:36235 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36214 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00154903] cur_instr RV_VLD (id:36272 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36251 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00154923] cur_instr RV_VLD (id:36309 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36288 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00154948] cur_instr RV_VLD (id:36349 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36325 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00154953] cur_instr RV_VLD (id:36383 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36358 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00154962] cur_instr RV_VLD (id:36420 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36399 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00155011] cur_instr RV_VLD (id:36484 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36439 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00155011] cur_instr RV_VLD (id:36490 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36445 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00155028] cur_instr RV_VLD (id:36531 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36510 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00155096] cur_instr RV_VLD (id:36568 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36547 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:30] [warning] [00155105] cur_instr RV_VLD (id:36605 pc:0x14c9f40c vloop_id:1 vloop_pc:0x14c9f408) overlaps with pre_instr RV_VST (id:36584 pc:0x14c9f420 vloop_id:1 vloop_pc:0x14c9f408)
[2026-04-12 18:14:31] [info] [0000155220] [block_end]             : AIV, task_id=0, core_id=6, block_id=6
[2026-04-12 18:14:31] [info] [0000155312] [block_end]             : AIV, task_id=0, core_id=24, block_id=24
[2026-04-12 18:14:31] [info] [0000155314] [block_end]             : AIV, task_id=0, core_id=31, block_id=31
[2026-04-12 18:14:31] [info] [0000155317] [block_end]             : AIV, task_id=0, core_id=20, block_id=20
[2026-04-12 18:14:31] [info] [0000155320] [block_end]             : AIV, task_id=0, core_id=17, block_id=17
[2026-04-12 18:14:31] [info] [0000155335] [block_end]             : AIV, task_id=0, core_id=16, block_id=16
[2026-04-12 18:14:31] [info] [0000155341] [block_end]             : AIV, task_id=0, core_id=14, block_id=14
[2026-04-12 18:14:31] [info] [0000155345] [block_end]             : AIV, task_id=0, core_id=13, block_id=13
[2026-04-12 18:14:31] [info] [0000155348] [block_end]             : AIV, task_id=0, core_id=28, block_id=28
[2026-04-12 18:14:31] [info] [0000155353] [block_end]             : AIV, task_id=0, core_id=8, block_id=8
[2026-04-12 18:14:31] [info] [0000155366] [block_end]             : AIV, task_id=0, core_id=5, block_id=5
[2026-04-12 18:14:31] [info] [0000155367] [block_end]             : AIV, task_id=0, core_id=30, block_id=30
[2026-04-12 18:14:31] [info] [0000155380] [block_end]             : AIV, task_id=0, core_id=22, block_id=22
[2026-04-12 18:14:31] [info] [0000155383] [block_end]             : AIV, task_id=0, core_id=18, block_id=18
[2026-04-12 18:14:31] [info] [0000155388] [block_end]             : AIV, task_id=0, core_id=34, block_id=34
[2026-04-12 18:14:31] [info] [0000155398] [block_end]             : AIV, task_id=0, core_id=27, block_id=27
[2026-04-12 18:14:31] [info] [0000155400] [block_end]             : AIV, task_id=0, core_id=9, block_id=9
[2026-04-12 18:14:31] [info] [0000155430] [block_end]             : AIV, task_id=0, core_id=10, block_id=10
[2026-04-12 18:14:32] [info] [0000155440] [block_end]             : AIV, task_id=0, core_id=4, block_id=4
[2026-04-12 18:14:32] [info] [0000155447] [block_end]             : AIV, task_id=0, core_id=12, block_id=12
[2026-04-12 18:14:32] [info] [0000155451] [block_end]             : AIV, task_id=0, core_id=26, block_id=26
[2026-04-12 18:14:32] [info] [0000155454] [block_end]             : AIV, task_id=0, core_id=15, block_id=15
[2026-04-12 18:14:32] [info] [0000155488] [block_end]             : AIV, task_id=0, core_id=25, block_id=25
[2026-04-12 18:14:32] [info] [0000155494] [block_end]             : AIV, task_id=0, core_id=32, block_id=32
[2026-04-12 18:14:32] [info] [0000155496] [block_end]             : AIV, task_id=0, core_id=19, block_id=19
[2026-04-12 18:14:32] [info] [0000155506] [block_end]             : AIV, task_id=0, core_id=33, block_id=33
[2026-04-12 18:14:32] [info] [0000155517] [block_end]             : AIV, task_id=0, core_id=11, block_id=11
[2026-04-12 18:14:32] [info] [0000155518] [block_end]             : AIV, task_id=0, core_id=7, block_id=7
[2026-04-12 18:14:32] [info] [0000155563] [block_end]             : AIV, task_id=0, core_id=21, block_id=21
[2026-04-12 18:14:32] [info] [0000155569] [block_end]             : AIV, task_id=0, core_id=2, block_id=2
[2026-04-12 18:14:32] [info] [0000155609] [block_end]             : AIV, task_id=0, core_id=1, block_id=1
[2026-04-12 18:14:32] [info] [0000155615] [block_end]             : AIV, task_id=0, core_id=23, block_id=23
[2026-04-12 18:14:32] [info] [0000155654] [block_end]             : AIV, task_id=0, core_id=3, block_id=3
[2026-04-12 18:14:32] [info] [0000155681] [block_end]             : AIV, task_id=0, core_id=29, block_id=29
[2026-04-12 18:14:32] [PASS] LargeShape_10K_Elements
[2026-04-12 18:14:32] 
[2026-04-12 18:14:32] --- 4. Complex 和 混合数据类型 测试 ---
[2026-04-12 18:14:32] [info] [0000155686] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:14:41] [info] [0000158657] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:14:41] [PASS] Add_Mix_BF16_FLOAT_alpha1
[2026-04-12 18:14:41] [info] [0000158662] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:14:48] [info] [0000161298] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:14:48] [PASS] Add_Mix_FLOAT_FP16_alpha1
[2026-04-12 18:14:48] 
[2026-04-12 18:14:48] === 发掘算子缺陷, alpha == 1 时, Adds不支持第一个输入是fp16、bf16 ===
[2026-04-12 18:14:48] [FAIL] API_Adds_Mix_BF16_FLOAT_alpha1 Workspace Error 561103
[2026-04-12 18:14:48] [info] [0000161303] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:14:56] [info] [0000164089] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:14:56] [PASS] API_Adds_Mix_FLOAT_FP16_alpha1
[2026-04-12 18:14:56] [info] [0000164094] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:01] [info] [0000165946] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:01] [PASS] API_InplaceAdd_Mix_FLOAT_FP16_alpha1
[2026-04-12 18:15:01] [info] [0000165951] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:09] [info] [0000168785] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:09] [PASS] API_InplaceAdd_Mix_FLOAT_BF16_alpha1
[2026-04-12 18:15:09] [info] [0000168790] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:14] [info] [0000170462] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:14] [PASS] API_InplaceAdds_Mix_FLOAT_FP16_alpha1
[2026-04-12 18:15:14] [info] [0000170467] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:20] [info] [0000172446] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:20] [PASS] API_InplaceAdds_Mix_FLOAT_BF16_alpha1
[2026-04-12 18:15:20] 
[2026-04-12 18:15:20] === 发掘算子缺陷, alpha == 1 时, AddV3不支持输入里有fp16、bf16类型 ===
[2026-04-12 18:15:20] [FAIL] API_AddV3_Mix_FLOAT_FP16_alpha1 Workspace Error 561103
[2026-04-12 18:15:20] [FAIL] API_AddV3_Mix_FLOAT_BF16_alpha1 Workspace Error 561103
[2026-04-12 18:15:20] [info] [0000172451] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:27] [info] [0000174964] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:27] [PASS] API_InplaceAddV3_Mix_FP16_FLOAT_alpha1
[2026-04-12 18:15:27] [info] [0000174975] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:35] [info] [0000177687] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:35] [info] [0000177692] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:41] [info] [0000180197] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:41] [PASS] API_Add_COMPLEX64
[2026-04-12 18:15:41] [info] [0000180202] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:47] [info] [0000182223] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:47] [info] [0000182228] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:53] [info] [0000184372] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:53] [PASS] API_Adds_COMPLEX64
[2026-04-12 18:15:53] [info] [0000184377] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:59] [info] [0000186421] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:15:59] [info] [0000186426] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:16:04] [info] [0000188195] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:16:04] [PASS] API_InplaceAdd_COMPLEX64
[2026-04-12 18:16:04] [info] [0000188200] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:16:09] [info] [0000190172] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:16:09] [info] [0000190177] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:16:14] [info] [0000191875] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:16:14] [PASS] API_InplaceAdds_COMPLEX64
[2026-04-12 18:16:14] 
[2026-04-12 18:16:14] --- 5. 数值边界、Inf、NaN 与空 Tensor ---
[2026-04-12 18:16:14] [info] [0000191880] [block_start]           : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:16:19] [info] [0000193702] [block_end]             : AIV, task_id=0, core_id=0, block_id=0
[2026-04-12 18:16:19] [PASS] Boundary_FLOAT_Inf_NaN_Max
[2026-04-12 18:16:19] [PASS] Empty_Tensor_Add
[2026-04-12 18:16:19] 
[2026-04-12 18:16:19] --- 6. 异常参数拦截测试 (反向断言) ---
[2026-04-12 18:16:19] [PASS] Negative_Nullptr_AllAPIs (Blocked)
[2026-04-12 18:16:19] [PASS] Negative_ShapeMismatch_AllAPIs (Blocked)
[2026-04-12 18:16:19] [PASS] Negative_InvalidDType_AllAPIs (Blocked)
[2026-04-12 18:16:19] 
[2026-04-12 18:16:19] === Final Ultra-Coverage Summary: 12 failed ===
[2026-04-12 18:16:19] [info] [MCU_LOG] [INFO] [0000193710] stop (645):: mcu_wrapper is stopping...
[2026-04-12 18:16:19] [info] [MCU_LOG] [INFO] [0000193710] main_loop (624):: mcu_wrapper loop is stopping...
[2026-04-12 18:16:19] [info] [MCU_LOG] [INFO] [0000193710] stop (654):: mcu_wrapper loop joined successfully.
[2026-04-12 18:16:19] [INFO] Model Stop Time: 2026-04-12 18:16:19
[2026-04-12 18:16:19] Model RUN TIME: 541174 ms
[2026-04-12 18:16:19] [INFO] Total tick: 193709
[2026-04-12 18:16:20] [INFO] Model stopped successfully.
File '/root/workspace/ops-math-master/math/add/op_api/aclnn_add.cpp'
Lines executed:73.27% of 303
File '/root/workspace/ops-math-master/math/add/op_host/arch35/add_tiling_arch35.cpp'
Lines executed:91.11% of 90
File '/root/workspace/ops-math-master/math/add/op_api/aclnn_add_v3.cpp'
Lines executed:80.52% of 77
File '/root/workspace/ops-math-master/math/add/op_api/add.cpp'
Lines executed:44.07% of 59
```


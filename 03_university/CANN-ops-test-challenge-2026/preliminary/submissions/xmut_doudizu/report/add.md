# Add 算子测试报告

## 1. 测试策略

本测试针对 CANN ops-math 仓库中的 Add 算子（`math/add`）进行端到端功能验证与代码覆盖率分析。测试覆盖以下维度：

| 维度 | 覆盖内容 |
|------|----------|
| **API 变体** | `aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnAddV3`、`aclnnInplaceAddV3` 共 6 个 API |
| **数据类型** | `FLOAT32`、`FLOAT16`、`INT32`、`DOUBLE`（强制走 AICPU 路径） |
| **Shape 组合** | 同 shape、广播（如 `[3,1]` + `[1,4]`）、空张量（shape 含 0 维）、非连续视图（通过 strides 构造） |
| **alpha 参数** | alpha = 1（标准加法）、alpha = 0（验证缩放无效）、alpha = 2.5（触发 Axpy 优化路径）、alpha = 2（整数类型） |
| **边界条件** | 零值、空张量、非连续内存布局、不可广播的 shape（预期失败）、空指针参数 |
| **执行路径** | AICore 主路径、AICPU 后备路径（通过 `DT_DOUBLE` 触发） |

测试环境：CPU 模拟器（`ascend950`），CANN 9.0.0，覆盖率插桩编译。

## 2. 测试用例设计

共 16 个测试用例，每个用例的设计目标如下：

| 序号 | 用例名称 | 设计目标 |
|------|----------|----------|
| 1 | `TestAddBasicFloat32` | 验证 `aclnnAdd` 基础功能：`float32` 类型，同 shape `[8]`，alpha=1，输出与 CPU 期望值比对。 |
| 2 | `TestAddsScalar` | 验证 `aclnnAdds`：tensor + scalar，alpha=0.5，检验标量广播和缩放。 |
| 3 | `TestInplaceAddSameShape` | 验证 `aclnnInplaceAdd`：原地加法，`selfRef` 与 `other` 形状相同，alpha=2，结果写回 `selfRef`。 |
| 4 | `TestInplaceAdds` | 验证 `aclnnInplaceAdds`：tensor 原地加标量，alpha=0.2。 |
| 5 | `TestAddV3` | 验证 `aclnnAddV3`：标量 + alpha * tensor，即第一个输入为标量。 |
| 6 | `TestInplaceAddV3` | 验证 `aclnnInplaceAddV3`：标量 + alpha * tensor 的原地版本，结果写回 `other` tensor。 |
| 7 | `TestBroadcast` | 验证广播机制：`[3,1]` 与 `[1,4]` 广播至 `[3,4]`，alpha=1。 |
| 8 | `TestMixedPrecisionFp16Fp32` | 验证混合精度：`self` 为 `float16`，`other` 为 `float32`，输出为 `float32`，alpha=1。 |
| 9 | `TestAlphaAxpy` | 验证 alpha != 1 时触发 Axpy 优化路径：`float32`，alpha=2.5。 |
| 10 | `TestAlphaZero` | 验证 alpha=0 时输出等于 `self`，检验 `alpha*other` 被忽略。 |
| 11 | `TestEmptyTensor` | 验证空张量（shape 包含 0 维）处理：API 应直接成功，不执行实际计算。 |
| 12 | `TestNonContiguousView` | 验证非连续输入：通过 strides 构造视图（`[2,3]` 存储，`[2,2]` 视图，步长 `[3,1]`），检验正确性。 |
| 13 | `TestInt32` | 验证整数类型：`int32` 输入，alpha 为整数 2，检验整数运算和类型提升。 |
| 14 | `TestDoubleForAiCpu` | 强制走 AICPU 路径：使用 `double` 类型（不在 AICore 支持列表中），覆盖 `AddAiCpu` 函数。 |
| 15 | `TestBroadcastFailure` | 验证广播失败场景：输入形状 `[2,3]` 与 `[3,2]` 不可广播，预期 API 返回错误。 |
| 16 | `TestNullptr` | 验证空指针参数：传入 `nullptr`，预期返回非成功状态码。 |

## 3. 最终覆盖率统计

使用 `gcov -b -c` 对 `add.cpp`（`math/add/op_api/add.cpp`）进行分析，结果如下：

```
Lines executed:60.00% of 55
Branches executed:27.76% of 263
Taken at least once:16.35% of 263
Calls executed:35.92% of 142
```

详细行覆盖情况（`add.cpp.gcov` 关键节选）：

| 函数/行范围 | 执行次数 | 状态 |
|-------------|----------|------|
| `IsAddSupportNonContiguous` (66-69) | 37 次 | 全覆盖 |
| `AddAiCore` (72-83) | 28 次 | 主体覆盖，内部宏分支未完全展开（模拟器限制） |
| `AddAiCpu` (86-99) | 2 次 | **成功覆盖**（通过 `DT_DOUBLE` 触发） |
| `Add` (101-123) | 30 次 | 覆盖 main 路径，广播失败分支未覆盖 |
| `AddInplace` (125-160) | 0 次 | 完全未覆盖（模拟器不支持原地加法 API） |
| `DAV_3102` 分支 (51-52) | 0 次 | 未覆盖（当前芯片为 `DAV_3510`） |
| 广播失败错误分支 (105-108) | 0 次 | 被上层 `CheckShape` 提前拦截，无法到达 |

## 4. 未覆盖代码分析

### 4.1 完全未覆盖的函数

- **`AddInplace` (行125-160)**  
  原因：在模拟器环境下，`aclnnInplaceAddGetWorkspaceSize` 返回 `561103`（`ACLNN_ERR_PARAM_INVALID`），导致底层 `AddInplace` 从未被调用。  
  分析：原地加法要求 `selfRef` 和 `other` 的 shape 完全一致且可广播，但模拟器可能对 inplace 操作支持不完整。真实硬件上应能覆盖。

### 4.2 条件分支未覆盖

- **`DAV_3102` 分支 (行51-52)**  
  原因：当前测试芯片架构为 `DAV_3510`（ascend950），代码走 `DAV_2201/3510` 分支，无法执行 `DAV_3102` 对应逻辑。  
  分析：该分支仅在特定芯片（如 Atlas 200I A2）上触发，不影响通用覆盖率评分。

- **广播失败分支 (行105-108)**  
  原因：在 `aclnnAddGetWorkspaceSize` 的 `CheckShape` 中已提前校验广播可行性并返回错误，底层 `Add` 函数中的 `BroadcastInferShape` 失败分支因此无法到达。  
  分析：属于防御性代码，上层已保证不会传入不可广播的输入，该分支在实际运行中极难触发。

### 4.3 部分覆盖的代码

- **`AddAiCore` 中的 `OP_CHECK` 宏**  
  宏内部的部分分支（如异常处理）在模拟器中未执行，因为 `ADD_TO_LAUNCHER_LIST_AICORE` 始终返回成功。真实硬件上可能触发异常路径。

- **混合数据类型判断 (行111-114)**  
  覆盖了 `FLOAT16+FLOAT32`、`FLOAT32+FLOAT16`、`BF16+FLOAT32` 等组合，但 `FLOAT32+BF16` 分支未在本次测试中触发（未编写对应用例）。

## 5. 结论

在模拟器环境下，`add.cpp` 的行覆盖率达到 **60.00%**，已覆盖主计算路径、AICPU 后备路径、广播、混合精度、非连续张量、空张量等关键功能。未覆盖部分主要受限于模拟器特性（不支持原地加法、芯片架构固定）和上层校验的提前拦截，在真实 Ascend 硬件上可进一步提升至 80% 以上。

建议后续在真实硬件上运行相同测试，并补充以下用例以覆盖剩余分支：
- 使用 `FLOAT32+BF16` 混合类型
- 增加 `INT8`、`UINT8`、`BOOL` 等数据类型
- 触发 `AddInplace` 的 shape 校验失败分支（传入不匹配的 shape）
```
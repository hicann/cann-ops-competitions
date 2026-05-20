---

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "队伍名称"

team_members:
- "成员1：樊旺-宿迁学院"
- "成员2：胡子航-宿迁学院"
- "成员3：王超-宿迁学院"

operator_name: "Add"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

---

# 算子测试报告

---

## 一、算子理解

### 1.1 数学定义

Add 算子族包含四个 API，数学语义如下：

| API | 数学表达式 | 说明 |
|-----|-----------|------|
| `aclnnAdd` | `out = self + alpha × other` | 张量 + 张量 |
| `aclnnAdds` | `out = self + alpha × other` | 张量 + 标量 |
| `aclnnAddV3` | `out = self + alpha × other` | 标量 + 张量 |
| `aclnnInplaceAdd` | `self += alpha × other` | 原地张量 + 张量 |
| `aclnnInplaceAdds` | `self += alpha × other` | 原地张量 + 标量 |
| `aclnnInplaceAddV3` | `other += self + alpha × other` | 原地标量 + 张量 |

其中 `alpha` 是缩放系数标量，默认语义等价于 `alpha=1`。

### 1.2 支持的数据类型

**AICore 路径（Ascend 910B，即本测试环境）：**

| dtype | aclnnAdd | aclnnAdds | aclnnAddV3 |
|-------|----------|-----------|------------|
| FLOAT32 | ✓ | ✓ | ✓ |
| FLOAT16 | ✓ | ✓ | ✓ |
| BF16 | ✓ | ✓ | ✓ |
| INT8 | ✓ | ✓ | ✓ |
| INT32 | ✓ | ✓ | ✓ |
| INT64 | ✓ | ✓ | — |
| UINT8 | ✓ | ✓ | — |
| BOOL | ✓ | ✓ | — |
| DOUBLE | AiCpu | AiCpu | — |
| INT16 | AiCpu | AiCpu | — |

**混合 dtype 支持（仅 aclnnAdd，alpha=1）：**
- FLOAT16 + FLOAT32 → FLOAT32
- FLOAT32 + FLOAT16 → FLOAT32
- BF16 + FLOAT32 → FLOAT32
- FLOAT32 + BF16 → FLOAT32

### 1.3 Broadcasting 支持

`aclnnAdd` 支持 NumPy 语义的广播，最大维度限制为 8 维。广播规则：两个输入形状从尾部对齐，逐维比较，维度为 1 的张量沿该维度广播扩展。输出形状为两个输入各维度的最大值。

### 1.4 内部计算路径

源码分析揭示了算子内部的分支决策树：

```
aclnnAddGetWorkspaceSize
├── IsEmpty → 早返回，wsSize=0
├── isMixDataType && alpha==1 → 直接调用混合dtype AICore kernel
└── else → 统一类型提升路径
    ├── IsEqualToOne(alpha)
    │   ├── sameDtype && IsAddSupportNonContiguous → stride直接Add
    │   ├── sameDtype && !NonContiguous → Contiguous + Add
    │   └── !sameDtype → Contiguous + Cast + Cast + Add
    ├── IsSupportAxpy(promoteType) → Contiguous + Cast + Cast + Axpy
    ├── IsSupportAxpyV2(promoteType) → Contiguous + Cast + Cast + AxpyV2
    └── else → Contiguous + Cast + Cast + Mul + Add（AiCpu路径）
```

**Tiling 分支（add_tiling_arch35.cpp）：**

| 条件 | Tiling模板 |
|------|-----------|
| FP16+FP32 混合（input1=FP32） | `AddMixDtypeCompute<half, float>` |
| FP32+FP16 混合（input0=FP32） | `AddMixDtypeCompute<float, half>` |
| FP16 或 BF16 | `AddWithCastCompute<half>` |
| FLOAT32 | `AddWithCastCompute<float>` |
| BOOL | `AddBoolCompute<int8_t>` |
| INT64 / COMPLEX64 | `AddWithoutCastCompute<int64_t>` |
| UINT8 | `AddWithoutCastCompute<uint8_t>` |
| INT8 | `AddWithoutCastCompute<int8_t>` |
| INT32 / COMPLEX32 | `AddWithoutCastCompute<int32_t>` |

### 1.5 值得关注的边界行为

- **alpha=0**：结果应等于 self，other 完全不参与计算，但仍会走 Axpy 路径
- **alpha=1**：触发 `IsEqualToOne` 优化，跳过乘法，直接 Add
- **alpha=-1.5**：负数 alpha，走 Axpy 路径，符号正确
- **BOOL 类型**：`true + true = true`（不是2），源码中有专门的 clamp 处理
- **空张量**：`IsEmpty()` 早返回，wsSize=0，不下发 kernel
- **isKeepB16**：FP16/BF16 的 scalar 加法中，若标量精度超出 B16 范围，自动提升为 FLOAT32

---

## 二、测试策略与用例设计

### 2.1 测试方法思路

采用**白盒测试为主、黑盒测试为辅**的策略：

1. **源码分析优先**：通读 `aclnn_add.cpp`、`aclnn_add_v3.cpp`、`add.cpp`、`add_tiling_arch35.cpp` 四个核心文件，绘制完整的分支决策树，确保每条路径都有对应的测试用例。

2. **gcov 驱动迭代**：运行测试后查看 `####` 标记的未覆盖行，针对性补充用例，循环迭代直到无法继续提升。

3. **精度验证**：对可以回读结果的测试（FLOAT32、INT32、INT64等），使用 CPU 端的 C++ 计算作为 Oracle，进行逐元素误差比对。

### 2.2 Oracle 选择

- **浮点类型**：CPU 端 `double` 精度计算，atol=1e-3，rtol=1e-3
- **整数类型**：CPU 端精确整数计算，要求完全相等
- **FP16/BF16**：仅验证 API 调用成功（GetWorkspaceSize 返回 ACL_SUCCESS），不回读精度，因为设备端 FP16 精度损失无法用 CPU double 精确对比

### 2.3 精度阈值设定依据

```
|actual - expected| ≤ atol + rtol × |expected|
```

- `atol = 1e-3`：绝对误差容限，处理接近零值的情况
- `rtol = 1e-3`：相对误差容限，处理大数值的情况
- 对于 alpha=2.5 这类非整数乘法，放宽至 atol=rtol=1e-2

### 2.4 用例分类

| 分类 | 用例数 | 覆盖目标 |
|------|--------|---------|
| dtype × alpha 基本路径 | 16 | 每种 dtype 的 alpha=1 和 alpha!=1 两条路径 |
| 混合 dtype | 4 | MixDtype 的 4 个子分支 |
| alpha 边界值 | 3 | alpha=0、alpha=-1.5、alpha=1.0（double类型） |
| aclnnAdds（标量加法） | 8 | PromoteTypeScalar 各分支 |
| InplaceAdd / InplaceAdds | 4 | CheckInplace + 复用 aclnnAdd 路径 |
| aclnnAddV3 | 8 | V3 的 Add/Axpy/Mul+Add 三条路径 |
| 形状压力测试 | 4 | 3D张量、大张量、全零边界 |

**总计：47 个测试用例**

### 2.5 被排除的用例及原因

| 用例类型 | 排除原因 |
|---------|---------|
| Broadcast测试（T13/T36/T48） | Broadcast kernel 中间 buffer 的 `device_ptr=0` 问题，导致 `aclrtSynchronizeStream` 永久挂起 |
| 空张量测试（T51/T52/T53） | `wsSize=0` 但 `exec` 非空时，`execFn(nullptr,0,exec,stream)` 后硬件无回包 |
| DOUBLE 类型（T54/T55/T62） | 走 AiCpu 路径，当前环境 AiCpu 调度队列存在问题，`SynchronizeStream` 永久等待 |
| INT16 类型（T56/T57） | 同 DOUBLE，走 AiCpu 路径 |
| isKeepB16=false（T58/T59） | 类型提升后走 AiCpu Cast，存在相同挂起风险 |
| 混合dtype alpha!=1（T60） | 该组合在当前驱动版本下 kernel 无法完成 |

---

## 三、覆盖率分析

### 3.1 测量方法

编译时开启 `-fprofile-arcs -ftest-coverage`，运行测试后使用 `gcov -b` 生成覆盖数据。

```bash
gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcda
gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcda
gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcda
gcov -b build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcda
```

### 3.2 覆盖率结果

| 文件 | 行覆盖率 | 分支覆盖率（Taken） |
|------|---------|-------------------|
| `add_tiling_arch35.cpp` | 89.25% | 38.54% |
| `aclnn_add.cpp` | 66.01% | 17.01% |
| `aclnn_add_v3.cpp` | 83.12% | 18.78% |
| `add.cpp` | 44.07% | 12.50% |

### 3.3 未覆盖部分归因

**add_tiling_arch35.cpp（未覆盖行）：**

| 行号 | 内容 | 原因 |
|------|------|------|
| 50-65 | `CheckDtype` 错误路径 | dtype 检查已在上层 `CheckParams` 过滤，不会传入非法dtype |
| 123-126 | `DoOpTiling` else 分支 | 同上，非法dtype在上层已拦截 |
| 161-162 | `context==nullptr` 检查 | 防御性代码，框架保证不会传入空指针 |

**aclnn_add.cpp（未覆盖行）：**

| 行号 | 内容 | 原因 |
|------|------|------|
| 95-99 | `DAV_1001/default` 芯片分支 | 当前硬件是 Atlas 800T A3（DAV_2201/3510），不会走这两个分支 |
| 104-177 | `GetScalarDefaultDtype`、`CombineCategoriesWithComplex`、`GetCastedFloat` | 这三个函数只在 `IsRegBase()=true` 的 `PromoteTypeScalar` 路径中调用，而本环境 `IsRegBase()` 实际返回情况需要运行时验证 |
| 334-335 | `IsEmpty` 早返回 | 空张量测试因硬件挂起被排除 |
| 355、374 | `selfWithStride` 直接 Add | 需要 `IsAddSupportNonContiguous` 返回 true 且满足特定条件 |
| 387-402 | `IsEqualToOne && !sameDtype` 路径 | 需要 alpha=1 但两个输入 dtype 不同的场景（如 INT32+INT64），上层类型检查已过滤 |
| 424-429 | `IsSupportAxpy` 路径的 Contiguous | 在 RegBase 架构下 Axpy 路径有差异 |

**aclnn_add_v3.cpp（未覆盖行）：**

| 行号 | 内容 | 原因 |
|------|------|------|
| 68-93 | 错误路径（CheckPromoteType失败） | 防御性错误处理，正常输入不会触发 |
| 122、133 | `IsComplexType` 分支 | Complex 类型不在 ADD_V3_DTYPE_SUPPORT_LIST 中，被 `CheckParams` 过滤 |
| 178-179 | `IsEmpty` 早返回 | 同 aclnnAdd 原因 |
| 203-206 | `else` Mul+Add 分支 | INT8 类型的 V3 测试（T31）理论上应该覆盖此路径，需测评环境验证 |

**add.cpp（未覆盖行）：**

| 行号 | 内容 | 原因 |
|------|------|------|
| 48-55 | `DAV_1001/DAV_3102/default` 芯片分支 | 当前硬件是 DAV_2201/3510 |
| 86-99 | `AddAiCpu` 函数 | DOUBLE/INT16 类型走 AiCpu，因环境挂起被排除 |
| 125-159 | `AddInplace` 函数 | `aclnnInplaceAdd` 调用的是 `aclnnAddGetWorkspaceSize`（以 selfRef 作为 out），不会调用 `AddInplace`；`AddInplace` 是独立的 inplace 语义实现，当前 API 层未调用 |

---

## 四、精度分析

### 4.1 误差度量方式

使用如下公式进行逐元素比对：

```cpp
bool CheckClose(double actual, double expected, double atol=1e-3, double rtol=1e-3) {
    return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}
```

对整数类型要求精确相等：`result[i] == expected[i]`。

### 4.2 各 dtype 精度表现

**FLOAT32（主力测试）：**

| 场景 | 输入示例 | 期望输出 | 实测误差 | 结论 |
|------|---------|---------|---------|------|
| alpha=1 基础加法 | self=[0,1,...,7], other=[1,1,...,3] | [1,2,...,10] | <1e-6 | 通过 |
| alpha=2 Axpy | self=[1,...,8], other=[1,...,1] | [3,...,10] | <1e-6 | 通过 |
| alpha=0 | self=[1,...,8], other=[100,...] | [1,...,8] | <1e-6 | 通过 |
| alpha=-1.5 | self=[10,...], other=[2,...] | [7,...] | <1e-6 | 通过 |
| alpha=2.5（scalar） | self=[1,...,8], other=3.0 | [8.5,...,15.5] | <1e-2 | 通过 |
| 全零边界 | self=other=[0,...] | [0,...] | =0 | 通过 |
| 3D张量 | shape=[2,3,4] | 逐元素和 | <1e-6 | 通过 |
| 大张量512×512 | all 1.0 + all 2.0 | all 3.0 | <1e-6 | 通过 |

**INT32：**

| 场景 | 期望 | 实测 | 结论 |
|------|------|------|------|
| alpha=1，self+other | 精确整数 | 完全一致 | 通过 |
| alpha=3，AxpyV2 | 精确整数 | 完全一致 | 通过 |
| 大张量128×128 | 3（全元素） | 完全一致 | 通过 |

**INT64：**

| 场景 | 期望 | 实测 | 结论 |
|------|------|------|------|
| alpha=1 | 精确整数 | 完全一致 | 通过 |
| scalar alpha=2 | 精确整数 | 完全一致 | 通过 |

**INT8：**

| 场景 | 说明 | 结论 |
|------|------|------|
| alpha=1，值域[0,17] | 不溢出 | 通过 |
| V3 Mul+Add，10+2×[1..8] | 结果[12..26]，不溢出 | 通过 |

**FP16/BF16：**

仅验证 API 返回值为 `ACL_SUCCESS`，不做数值精度比对，原因是 FP16 精度损失（约3位小数）难以用 CPU double 精确预测设备端舍入行为。

### 4.3 BOOL 类型精度

BOOL 类型加法语义为逻辑 OR 而非数值加法，源码中有 clamp 处理防止出现值为 2 的情况。测试验证了 API 能正常执行，未发现异常。

### 4.4 alpha=1.0（double类型标量）

测试了将 alpha 以 `ACL_DOUBLE` 类型传入 `aclnnAdd` 的场景，触发 `IsEqualToOne` 中的 `DT_DOUBLE` 分支（`alpha->ToDouble()` 而非 `alpha->ToFloat()`），精度正常。

---

## 五、反思与改进

### 5.1 测试盲区与局限性

**硬件依赖的路径无法本地验证：**
- `AddAiCpu` 路径（DOUBLE/INT16）：当前 Atlas 800T A3 环境的 AiCpu 调度存在问题，`aclrtSynchronizeStream` 会永久挂起。这两种类型的测试完全无法运行，是最大的覆盖盲区。
- Broadcast 测试：分析发现 broadcast kernel 运行时中间 buffer 的 `device_ptr=0` 问题导致硬件挂起，无法安全测试。正确的做法应该是先通过 `GetWorkspaceSize` 获取并分配完整 workspace 后再执行，当前测试框架未支持此流程。

**芯片架构分支无法覆盖：**
- `DAV_1001`（Ascend 910A）、`DAV_3102`（Ascend 310）分支的 dtype 支持列表不同，需要对应硬件才能覆盖，本机器无法模拟。

**`IsRegBase()` 的实际行为：**
- 该函数实现在闭源库中，本地 gcov 无法观察到 `CombineCategoriesWithComplex`、`GetCastedFloat` 等函数是否被调用。测评系统可能有不同的环境配置。

### 5.2 若有更多时间会如何扩展

1. **参数化测试框架**：用 `std::tuple` + 模板将测试用例数据与执行逻辑分离，减少重复代码，同时支持更大规模的随机测试。

2. **随机测试（Fuzzing）**：对 FLOAT32 和 INT32，用随机种子生成大量输入，与 CPU 参考实现比对，可能发现边界 case。

3. **非连续张量测试**：构造 stride 不连续的张量（如转置后的张量），专门触发 `isSupportNonContiguous` 路径，覆盖 `selfWithStride` 直接 Add 的代码路径。

4. **广播修复**：修改测试框架，在 `GetWorkspaceSize` 之后根据返回的 `wsSize` 分配 workspace，再执行，使 broadcast 测试能够安全运行。

5. **AiCpu 超时保护**：为每个测试增加超时机制（如用 `alarm` + `SIGALRM`），使 AiCpu 路径测试即使挂起也能自动跳过，不阻塞整体测试。

### 5.3 方法论层面的经验教训

**先读源码再写测试：** 本次最有效的提升是通过阅读 `aclnn_add.cpp` 的分支决策树，明确了 `IsEqualToOne`、`IsSupportAxpy`、`IsSupportAxpyV2`、`isMixDataType` 四个关键判断，每个判断对应一条独立的代码路径，针对性写测试效率远高于盲目枚举。

**环境差异是最大陷阱：** 本地测试库链接的是系统预装库而非编译插桩库，导致本地 gcov 数值完全不反映测试效果。这个问题排查耗费了大量时间。今后应该在测试环境搭建阶段就验证 `ldd` 指向是否正确。

**硬件挂起比失败更难处理：** 测试挂起（`aclrtSynchronizeStream` 永久等待）比测试失败更难定位，因为没有任何错误输出。定位方法是在每个测试入口加 `TRACE_ENTER` 日志并立即 `fflush`，通过最后打印的日志定位到卡死的具体测试，再通过分析 RUNTIME 日志的 `LaunchKernel` 和 `StreamSynchronize` 时间戳确认是 kernel 下发后无回包。

### 5.4 对 CANN 测试工具链的建议

1. **提供官方测试框架**：目前需要手写 `aclrtMalloc`、`aclrtMemcpy`、`aclCreateTensor` 等大量样板代码，建议提供类似 PyTorch 的 `torch.testing.assert_close` 的高层封装。

2. **AiCpu 超时配置**：`aclrtSynchronizeStream` 缺少超时参数，一旦 AiCpu 任务异常就会永久挂起，建议增加超时选项或提供异步查询接口。

3. **gcov 集成文档**：官方文档中缺少如何正确配置 gcov 插桩、确保测试程序链接插桩库的说明，开发者容易踩坑。

---

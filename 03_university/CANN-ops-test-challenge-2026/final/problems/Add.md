# 决赛题目：Add 算子测试用例设计

## 任务说明

本题目要求参赛者为 CANN ops-math 仓库中的 **Add（逐元素加法）算子**编写端到端测试用例。参赛者需要在官方提供的 example 测试代码基础上进行扩展，尽可能覆盖算子的各种执行路径，并深入分析算子的精度特性。

**算子定义：** $y = x_1 + \alpha \times x_2$，其中 $\alpha$ 为标量缩放因子（默认为 1）。当两个输入的 shape 不一致时，按广播规则对齐后逐元素计算。

## 算子概况

Add 算子位于 `math/add/` 目录下，采用 op_api → op_host → op_kernel 的三层架构。Add 算子引入了 `alpha` 缩放参数和 V3 版本 API。

### 目录结构

```
math/add/
├── op_api/                         # 接口层
│   ├── aclnn_add.h / aclnn_add.cpp #   API 声明与实现（713 行）
│   ├── aclnn_add_v3.h / .cpp       #   V3 版本 API（247 行）
│   ├── add.h / add.cpp             #   底层接口与设备路由（162 行）
├── op_host/                        # 主机计算层
│   ├── add_def.cpp                 #   算子注册：声明支持的 dtype 组合
│   ├── add_infershape.cpp          #   shape 推断
│   └── arch35/
│       └── add_tiling_arch35.cpp   #   tiling 切分策略（190 行）
├── op_kernel/                      # 设备计算层
│   └── ...
├── examples/                       # 使用示例
└── tests/                          # 单元测试
```

### 支持的数据类型

Add 算子支持 14 种数据类型组合，包括 BF16、FLOAT16、FLOAT32、INT32、UINT8、INT8、INT64、BOOL、COMPLEX32、COMPLEX64 等同类型运算，以及 FLOAT16-FLOAT、BF16-FLOAT 等混合类型运算。完整列表可查看 `op_host/add_def.cpp`。

### API 变体

Add 算子对外提供 6 个 API，分为标准版和 V3 版两组：

| API                                        | 语义                                              |
| ------------------------------------------ | ------------------------------------------------- |
| `aclnnAdd(self, other, alpha, out)`        | out = self + alpha * other（tensor 加 tensor）    |
| `aclnnAdds(self, other, alpha, out)`       | out = self + alpha * scalar（tensor 加标量）      |
| `aclnnInplaceAdd(selfRef, other, alpha)`   | selfRef += alpha * other（原地加 tensor）         |
| `aclnnInplaceAdds(selfRef, other, alpha)`  | selfRef += alpha * scalar（原地加标量）           |
| `aclnnAddV3(self, other, alpha, out)`      | out = scalar + alpha * other（**标量**加 tensor） |
| `aclnnInplaceAddV3(selfRef, other, alpha)` | V3 版本的原地加法                                 |

**注意：** Add 的所有 API 都包含一个 `alpha` 参数（`aclScalar*` 类型），用于对第二个输入进行缩放。当 `alpha = 1` 时等价于普通加法，当 `alpha` 取其他值时会走不同的计算路径。这是 Add 算子的重要测试维度。

### V3 版本 API 说明

`aclnnAddV3` 与标准 `aclnnAdd` 的核心区别在于 **`self` 参数的类型不同**：

|               | aclnnAdd                     | aclnnAddV3                  |
| ------------- | ---------------------------- | --------------------------- |
| self 参数类型 | `const aclTensor*`（tensor） | `const aclScalar*`（标量）  |
| 语义          | tensor + alpha * tensor      | **scalar** + alpha * tensor |

V3 版本本质上是 **ScalarTensor 形式的加法**——第一个输入是标量而非 tensor。它在 `aclnn_add_v3.cpp`（247 行）中有完全独立的实现，包括独立的类型提升逻辑和三分支调度（alpha=1 直接 Add / 支持 Axpy 的类型走融合算子 / 其余先 Mul 再 Add）。不调用 V3 API，该文件中的代码就不会被覆盖。

调用 V3 API 时，需要引入对应的头文件 `aclnnop/aclnn_add_v3.h`，并注意 self 参数用 `aclCreateScalar` 创建。

## 任务要求

官方示例代码位于 `math/add/examples/test_aclnn_add.cpp`。参赛者的任务：

### 1）扩展测试覆盖面并补充结果验证

为每个测试用例在 CPU 端独立计算期望值，并与算子输出进行数值比对。Add 算子的期望值计算需要考虑 alpha 参数：

```cpp
// x1, x2 为 float 输入，alpha 为缩放因子
double expected = (double)x1[i] + alpha * (double)x2[i];
```

浮点类型使用容差比较：$|actual - expected| \leq atol + rtol \times |expected|$

建议容差：

- FLOAT32: atol=1e-6, rtol=1e-6
- FLOAT16: atol=1e-4, rtol=1e-4
- INT32: 精确匹配

**覆盖维度包括但不限于**：

- **数据类型**：FLOAT32、FLOAT16、BF16、INT32、INT8 等，不同 dtype 触发不同的 tiling 策略分支
- **alpha 参数**：alpha=1（标准加法）、alpha=0、alpha 为负数、alpha 为浮点数等
- **Shape 组合**：同 shape、广播、标量、较大 tensor 等
- **数值边界**：零值、极大值、NaN、Inf、整数溢出等
- **API 变体**：Add、Adds、InplaceAdd、InplaceAdds、AddV3、InplaceAddV3 共 6 个 API
- **异常输入**：nullptr、不支持的 dtype 等

V3 API 调用示例：

```cpp
#include "aclnnop/aclnn_add_v3.h"

// 创建 scalar self（注意：不是 tensor！）
float scalarValue = 10.0f;
aclScalar* self = aclCreateScalar(&scalarValue, ACL_FLOAT);

// 调用 V3 API
ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);

// 清理
aclDestroyScalar(self);
```

### 2）精度测试与分析

分析精度问题的场景和原因，并在测试报告中详细记录（每个场景仅举一例即可）。以下场景提供参考：

**场景提示1：大数+小数**

- 尝试 `[1e10, 1e10] + [1e-5, 1e-5]` 这样的输入
- 观察小数是否被大数吞没
- 分析浮点数有效位数的限制

**场景提示2：正负抵消**

- 尝试 `[1.0000001, 2.0000001] + [-1.0, -2.0]` 这样的输入
- 观察接近值相减时的精度损失
- 分析 Catastrophic Cancellation 现象

**其他可探索的精度场景**：

- Alpha 参数引入的额外误差
- 混合类型运算的精度损失
- 浮点特殊值（NaN, Inf, 次正规数）的处理

### 3）输出格式

每个测试用例输出 `[PASS]` 或 `[FAIL]`，程序结尾输出汇总，有失败用例返回非 0 值。

**输出示例**：

```
Test case 1: Basic Add (float32)
  Expected: [1.200000, 2.200000, 3.200000]
  Actual:   [1.200000, 2.200000, 3.200000]
  [PASS]

Test case 2: Large + Small (precision loss)
  Expected: [10000000000.000010, 10000000000.000010]
  Actual:   [10000000000.000000, 10000000000.000000]
  Error:    [0.000010, 0.000010]
  [FAIL] Precision loss detected

Summary: 1 passed, 1 failed
```

## 编译与运行

### 前置步骤：修复 CMakeLists 以启用 Host 层覆盖率

**问题现象**：默认的 `math/add/CMakeLists.txt` 在 `ascend910_93` SOC 下存在两处配置问题，导致 `op_host/arch35/add_tiling_arch35.cpp` 的 `.gcno / .gcda` 均无法生成，Host 层覆盖率将为 **0%**（直接拉低综合得分）：

1. `SUPPORT_COMPUTE_UNIT` 列表中**不包含 `ascend910_93`**，查表返回空；
2. 即便补上 SOC，`SUPPORT_TILING_DIR` 也需要对应一个存在的目录（仓库只有 `arch35/`）。

解决方案：编译前先修改 `math/add/CMakeLists.txt`，补齐 SOC 列表并统一映射到 `arch35`：

```diff
- set(SUPPORT_COMPUTE_UNIT "ascend950" "mc62cm12a")
- set(SUPPORT_TILING_DIR "arch35" "arch35")
+ set(SUPPORT_COMPUTE_UNIT "ascend310p" "ascend910_93" "ascend910b" "ascend950" "mc62cm12a")
+ set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")
```

两行 `sed` 命令搞定：

```bash
sed -i 's|set(SUPPORT_COMPUTE_UNIT "ascend950" "mc62cm12a")|set(SUPPORT_COMPUTE_UNIT "ascend310p" "ascend910_93" "ascend910b" "ascend950" "mc62cm12a")|;
        s|set(SUPPORT_TILING_DIR "arch35" "arch35")$|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' \
    math/add/CMakeLists.txt
```

**根因简述**：tiling 代码运行在 **CPU**（host 侧），gcov 完全可观测。之所以"测不到"不是架构限制，而是 CMake 层面的 SOC→arch 目录映射配置错误导致源码未进入编译。

> **注意**：修改 CMakeLists 后如已经跑过一次编译，请先 `rm -rf build build_out` 清空产物再重新编译，否则缓存中的旧配置仍然生效。

### 编译运行流程

```bash
# 编译（启用覆盖率插桩）
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov

# 安装算子包
./build_out/cann-ops-math-custom_linux-aarch64.run

# 运行测试（真实 NPU 环境）
bash build.sh --run_example add eager cust \
    --vendor_name=custom --soc=ascend910_93 --cov

# 查看覆盖率
find build -name "*.gcda" | grep add
gcov -b -c <gcda文件路径>
```

**注意**：

- 使用 `--soc=ascend910_93` 参数
- 不使用 `--simulator` 参数，直接在真实 NPU 上运行
- 每次修改测试用例后，需要重新执行编译 → 安装 → 运行的完整流程以更新覆盖率数据
- **编译后务必校验 host 层产物**：`find build -name "add_tiling*.gcno"` 应能查到对应文件，若为空说明前置 CMakeLists 修复未生效，请回到上面的前置步骤重新操作

## 评分标准

决赛采用五维综合评分，如下：

### 维度 1. 编译通过率

提交的测试代码必须能在评测环境中通过下述完整流程正常执行：

```
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
→ ./build_out/cann-ops-math-custom_linux-aarch64.run
→ bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

编译或运行任何一步失败都会影响整体得分。评测系统会尝试从提交的 build 目录提取覆盖率数据作为参考，但编译失败的提交整体得分受限。

### 维度 2. 行覆盖率

**覆盖率统计范围**：op_api 层的 3 个文件 + op_host 层的 1 个 tiling 文件，共 4 个源文件。

| 文件                                   | layer |
| -------------------------------------- | ----- |
| `op_api/aclnn_add.cpp`                 | api   |
| `op_api/aclnn_add_v3.cpp`              | api   |
| `op_api/add.cpp`                       | api   |
| `op_host/arch35/add_tiling_arch35.cpp` | host  |

**综合行覆盖率**由各文件的命中行数与总行数加总后计算：

$$ Coverage_{line} = \frac{\sum Lines_Covered_i}{\sum Total_Lines_i} $$

**提示**：

- `aclnn_add_v3.cpp` 是独立的 V3 API 实现文件，若不调用 `aclnnAddV3` 系列接口，整个文件会是 0% 覆盖率
- `add_tiling_arch35.cpp` 涵盖 Add 算子的 tiling 切分策略，不同 dtype 组合走不同分支

### 维度 3. 分支覆盖率

对上述同样的 4 个文件统计分支覆盖率（`gcov -b` 输出）：

$$ Coverage_{branch} = \frac{\sum Branches_Covered_i}{\sum Total_Branches_i} $$

### 维度 4. 精度分析

根据测试报告中对精度问题的场景发现与原理分析综合评分。

### 维度 5. 测试报告

根据测试报告的完整性、结构、分析深度评分。

### 前置条件

1. **编译通过**：提交的 `test_aclnn_add.cpp` 必须能在评测环境中通过编译和运行流程正常执行。编译失败的提交将无法获得覆盖率得分，但评测系统会尝试从提交的 build 目录中提取覆盖率数据作为参考。
2. **结果验证**：测试代码中必须包含有效的结果验证逻辑（即计算期望值并与实际输出比对），仅打印结果而不验证的提交将被扣分。
3. **测试报告**：必须提交测试报告，按照提供的模版编写。

## 提交要求

提交一个压缩包 `.zip`，包含：

```
<队名>/
├── test_aclnn_add.cpp          # 测试用例源文件（必须）
├── build/                      # 编译产物目录（必须，见下方说明）
└── report.md                  # 测试设计说明（必须，按模版编写）
```

### build 目录提交说明

**重要**：为减小提交包大小，只需提交评分相关的 `.gcda` 和 `.gcno` 文件。涉及两个路径：

**op_api 层**（目录名 `abs` 非笔误，是 CMake 聚合 object library 的挂载点）：

```
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/
├── aclnn_add.cpp.gcda
├── aclnn_add.cpp.gcno
├── aclnn_add_v3.cpp.gcda
├── aclnn_add_v3.cpp.gcno
├── add.cpp.gcda
└── add.cpp.gcno
```

**op_host 层（tiling）**：

```
build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/
├── add_tiling_arch35.cpp.gcda
└── add_tiling_arch35.cpp.gcno
```

**不要提交**完整的 build 目录（可能有几百 MB），只提交上述覆盖率文件即可。如需偷懒，也可以使用以下命令快速筛选：

```bash
find build -name "aclnn_add.cpp.gc*" \
        -o -name "aclnn_add_v3.cpp.gc*" \
        -o -name "add.cpp.gc*" \
        -o -name "add_tiling*.gc*" \
    | tar czvf add_gcov.tar.gz -T -
```

------

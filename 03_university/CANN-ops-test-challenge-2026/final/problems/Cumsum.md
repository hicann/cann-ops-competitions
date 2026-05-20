# 决赛题目：Cumsum 算子测试用例设计

## 任务说明

本题目要求参赛者为 CANN ops-math 仓库中的 **Cumsum（累积求和）算子**编写端到端测试用例。参赛者需要在官方提供的 example 测试代码基础上进行扩展，尽可能覆盖算子的各种执行路径，并深入分析算子的精度特性。

**算子定义：** $y[i] = \sum_{j=0}^{i} x[j]$，即输出的第 i 个元素是输入前 i+1 个元素的累积和。Cumsum 算子的累加特性使其成为研究浮点误差累积效应的理想对象。

## 算子概况

Cumsum 算子位于 `math/cumsum/` 目录下，采用 op_api → op_host → op_kernel 的三层架构。Cumsum 算子的核心特点是**误差累积**——每次累加都会累积前面的舍入误差，序列越长，累积误差越大。

### 目录结构

```
math/cumsum/
├── op_api/                         # 接口层
│   ├── aclnn_cumsum.h / .cpp       #   API 声明与实现（包含 CumsumV2）
│   ├── cumsum.h / cumsum.cpp       #   底层接口与设备路由
├── op_host/                        # 主机计算层
│   ├── cumsum_def.cpp              #   算子注册
│   ├── cumsum_infershape.cpp       #   shape 推断
│   └── arch35/
│       ├── cumsum_tiling_arch35.h
│       ├── cumsum_tiling_ascendc_arch35.cpp      # 浮点类型 tiling
│       ├── cumsum_tiling_ascendc_int_arch35.cpp  # 整数类型 tiling
│       └── cumsum_tiling.cpp                     # tiling 公共逻辑
├── op_kernel/                      # 设备计算层
│   └── ...
├── examples/                       # 使用示例
└── tests/                          # 单元测试
```

### 支持的数据类型

Cumsum 算子支持多种数据类型，包括 FLOAT32、FLOAT16、BF16、INT32、INT64、INT8、UINT8、BOOL 等。不同数据类型的累积误差特性差异显著，是精度测试的重要维度。

### API 变体

Cumsum 算子对外提供 2 个主要 API：

| API                                                        | 语义                                     |
| ---------------------------------------------------------- | ---------------------------------------- |
| `aclnnCumsum(self, dim, dtype, out)`                       | 标准累积求和                             |
| `aclnnCumsumV2(self, dim, dtype, exclusive, reverse, out)` | 扩展版本，支持 exclusive 和 reverse 参数 |

**参数说明：**

- `dim`: 累加的维度
- `dtype`: 输出数据类型（可与输入不同）
- `exclusive`: 若为 true，输出第一个元素为 0，后续为前缀和（不包含当前元素）
- `reverse`: 若为 true，从后向前累加

## 任务要求

官方示例代码位于 `math/cumsum/examples/test_aclnn_cumsum.cpp`。参赛者的任务：

### 1）扩展测试覆盖面并补充结果验证

为每个测试用例在 CPU 端独立计算期望值，并与算子输出进行数值比对。Cumsum 算子的期望值计算：

```cpp
// CPU 端参考实现
std::vector<double> CpuCumsum(const std::vector<float>& input) {
    std::vector<double> result(input.size());
    double sum = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        sum += (double)input[i];
        result[i] = sum;
    }
    return result;
}
```

浮点类型使用容差比较：$|actual - expected| \leq atol + rtol \times |expected|$

建议容差：

- FLOAT32: atol=1e-5, rtol=1e-5（考虑累积误差，比单次运算更宽松）
- FLOAT16: atol=1e-3, rtol=1e-3
- INT32: 精确匹配（但需注意溢出）

**覆盖维度包括但不限于**：

- **数据类型**：FLOAT32、FLOAT16、BF16、INT32、INT64 等
- **序列长度**：短序列（<100）、中等序列（100-1000）、长序列（>1000）
- **数值特征**：全正、全负、正负混合、大小数混合、零值等
- **API 变体**：Cumsum、CumsumV2（测试 exclusive 和 reverse 参数）
- **维度参数**：不同的 dim 值
- **异常输入**：nullptr、空 tensor、不支持的 dtype 等

### 2）精度测试与分析

分析精度问题的场景和原因，重点关注误差随序列长度的增长规律，并在测试报告中深入分析精度问题的数学原理（每个场景仅举一例即可）。以下场景提供参考：

**场景提示1：误差累积效应**

- 尝试累加 10000 个相同的数（如 1.0），观察最后结果与理论值 10000.0 的偏差
- 分析误差如何随序列长度线性增长
- 量化分析：误差 ≈ n * ε，其中 n 是序列长度，ε 是单次加法的舍入误差

**场景提示2：大小数混合序列**

- 尝试 `[1e8, 1e-6, 1e8, 1e-6, ...]` 这样的输入
- 观察小数 1e-6 的贡献是否在累加到 1e8 后被吞没
- 分析浮点数有效位数的限制对累积结果的影响

**场景提示3：不同 dtype 的累积误差对比**

- 对比 float32 和 float16 在相同输入下的累积误差
- 量化分析：float16 的误差累积速度约为 float32 的 1000 倍
- 评估不同 dtype 的适用场景

**其他可探索的精度场景**：

- 正负交替序列的抵消效应 + 误差累积
- 长序列小数累加（如 `[0.1] * 10000`，0.1 无法精确表示）
- Exclusive 和 reverse 模式的精度特性
- 整数类型的溢出问题

### 3）输出格式

每个测试用例输出 `[PASS]` 或 `[FAIL]`，程序结尾输出汇总，有失败用例返回非 0 值。

**输出示例**：

```
Test case 1: Basic Cumsum (float32, length=100)
  Expected: [1.0, 2.0, 3.0, ..., 100.0]
  Actual:   [1.0, 2.0, 3.0, ..., 100.0]
  Max error: 0.000001
  [PASS]

Test case 2: Long sequence accumulation (float32, length=10000)
  Expected: [1.0, 2.0, 3.0, ..., 10000.0]
  Actual:   [1.0, 2.0, 3.0, ..., 9999.998]
  Max error: 0.002 (at position 9999)
  [PASS] Error within tolerance

Test case 3: Mixed magnitude (float16)
  Expected: [1e8, 1e8, 2e8, 2e8, ...]
  Actual:   [1e8, 1e8, 2e8, 2e8, ...]
  Small values lost: 1e-6 contributions = 0
  [FAIL] Precision loss detected

Summary: 2 passed, 1 failed
```

## 编译与运行

### 前置步骤：修复 CMakeLists 以启用 Host 层覆盖率

**问题现象**：默认的 `math/cumsum/CMakeLists.txt` 在 `ascend910_93` SOC 下存在 `SUPPORT_TILING_DIR` 错配问题——映射到不存在的 `arch32/` 目录，导致 `op_host/arch35/` 下的 3 个 tiling 源文件`.gcno / .gcda` 均无法生成，Host 层覆盖率将为 **0%**。

解决方案：编译前先修改 `math/cumsum/CMakeLists.txt`，将 `SUPPORT_TILING_DIR` 中的 `arch32` 全部改为 `arch35`：

```diff
  set(SUPPORT_COMPUTE_UNIT "ascend310p" "ascend910_93" "ascend910b" "ascend950" "mc62cm12a")
- set(SUPPORT_TILING_DIR "arch32" "arch32" "arch32" "arch35" "arch35")
+ set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")
```

一行 `sed` 命令搞定：

```bash
sed -i 's|set(SUPPORT_TILING_DIR "arch32" "arch32" "arch32" "arch35" "arch35")|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' \
    math/cumsum/CMakeLists.txt
```

**根因简述**：tiling 代码运行在 **CPU**（host 侧），gcov 完全可观测。之所以"测不到"不是架构限制，而是 CMake 的 SOC→arch 目录映射配置错误导致源码未进入编译。

> **注意**：修改 CMakeLists 后如已经跑过一次编译，请先 `rm -rf build build_out` 清空产物再重新编译，否则缓存中的旧配置仍然生效。

### 编译运行流程

```bash
# 编译（启用覆盖率插桩）
bash build.sh --pkg --soc=ascend910_93 --ops=cumsum --vendor_name=custom --cov

# 安装算子包
./build_out/cann-ops-math-custom_linux-aarch64.run

# 运行测试（真实 NPU 环境）
bash build.sh --run_example cumsum eager cust \
    --vendor_name=custom --soc=ascend910_93 --cov

# 查看覆盖率
find build -name "*.gcda" | grep cumsum
gcov -b -c <gcda文件路径>
```

**注意**：

- 使用 `--soc=ascend910_93` 参数
- 不使用 `--simulator` 参数，直接在真实 NPU 上运行
- 每次修改测试用例后，需要重新执行编译 → 安装 → 运行的完整流程以更新覆盖率数据
- **编译后务必校验 host 层产物**：`find build -name "cumsum_tiling*.gcno"` 应能查到 3 个文件，若为空说明前置 CMakeLists 修复未生效，请回到上面的前置步骤重新操作

## 评分标准

决赛采用五维综合评分，如下：

### 维度 1. 编译通过率

提交的测试代码必须能在评测环境中通过下述完整流程正常执行：

```
bash build.sh --pkg --soc=ascend910_93 --ops=cumsum --vendor_name=custom --cov
→ ./build_out/cann-ops-math-custom_linux-aarch64.run
→ bash build.sh --run_example cumsum eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

编译或运行任何一步失败都会影响整体得分。评测系统会尝试从提交的 build 目录提取覆盖率数据作为参考，但编译失败的提交整体得分受限。

### 维度 2. 行覆盖率

**覆盖率统计范围**：op_api 层的 2 个文件 + op_host 层的 3 个 tiling 文件，共 5 个源文件。

| 文件                                                  | layer |
| ----------------------------------------------------- | ----- |
| `op_api/aclnn_cumsum.cpp`                             | api   |
| `op_api/cumsum.cpp`                                   | api   |
| `op_host/arch35/cumsum_tiling.cpp`                    | host  |
| `op_host/arch35/cumsum_tiling_ascendc_arch35.cpp`     | host  |
| `op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp` | host  |

**综合行覆盖率**由各文件的命中行数与总行数加总后计算：

$$ Coverage_{line} = \frac{\sum Lines_Covered_i}{\sum Total_Lines_i} $$

**提示**：tiling 层（`op_host/arch35/` 下的三个文件）承载了 Cumsum 算子的主要切分策略代码，不同的 dtype、shape、V2 标志位组合会走不同的 tiling 分支，是覆盖率提升的重点区域。

### 维度 3. 分支覆盖率

对上述同样的 5 个文件统计分支覆盖率（`gcov -b` 输出）：

$$ Coverage_{branch} = \frac{\sum Branches_Covered_i}{\sum Total_Branches_i} $$

分支覆盖比行覆盖更严格：`if-else` 的 true / false 两侧都需要被触发才算完全覆盖。鼓励针对条件分支、边界值、错误路径设计成对的测试用例。

### 维度 4. 精度分析

根据测试报告中对精度问题的场景发现与原理分析综合评分。

### 维度 5. 测试报告

根据测试报告的完整性、结构、分析深度评分。

### 前置条件

1. **编译通过**：提交的 `test_aclnn_cumsum.cpp` 必须能在评测环境中通过编译和运行流程正常执行。编译失败的提交将无法获得完整覆盖率得分，但评测系统会尝试从提交的 build 目录中提取覆盖率数据作为参考。
2. **结果验证**：测试代码中必须包含有效的结果验证逻辑（即计算期望值并与实际输出比对），仅打印结果而不验证的提交将被扣分。
3. **测试报告**：必须提交测试报告，按照提供的模版编写。精度分析章节应重点关注误差累积效应。

## 提交要求

提交一个压缩包 `.zip`，包含：

```
<队名>/
├── test_aclnn_cumsum.cpp       # 测试用例源文件（必须）
├── build/                      # 编译产物目录（必须，见下方说明）
└── report.md                  # 测试设计说明（必须，按模版编写）
```

### build 目录提交说明

**重要**：为减小提交包大小，只需提交评分相关的 `.gcda` 和 `.gcno` 文件。涉及两个路径：

**op_api 层**（目录名 `abs` 非笔误，是 CMake 聚合 object library 的挂载点）：

```
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/cumsum/op_api/
├── aclnn_cumsum.cpp.gcda
├── aclnn_cumsum.cpp.gcno
├── cumsum.cpp.gcda
└── cumsum.cpp.gcno
```

**op_host 层（tiling）**：

```
build/math/cumsum/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/
├── cumsum_tiling.cpp.gcda
├── cumsum_tiling.cpp.gcno
├── cumsum_tiling_ascendc_arch35.cpp.gcda
├── cumsum_tiling_ascendc_arch35.cpp.gcno
├── cumsum_tiling_ascendc_int_arch35.cpp.gcda
└── cumsum_tiling_ascendc_int_arch35.cpp.gcno
```

**不要提交**完整的 build 目录（可能有几百 MB），只提交上述覆盖率文件即可。如需偷懒，也可以使用以下命令快速筛选：

```bash
find build -name "aclnn_cumsum.cpp.gc*" \
        -o -name "cumsum.cpp.gc*" \
        -o -name "cumsum_tiling*.gc*" \
    | tar czvf cumsum_gcov.tar.gz -T -
```

------

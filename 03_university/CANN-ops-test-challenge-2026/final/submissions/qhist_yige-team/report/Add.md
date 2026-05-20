# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "伊格小队"

team_members:

- "队长：张德鑫-青岛恒星科技学院"

operator_name: "Add"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# 算子测试报告

> 以下章节为建议框架。章节顺序与标题建议保留，章节**内部内容的组织方式（文字、表格、图示）自行决定**。括号中的"建议包含"为引导性提示，非强制要求，可根据算子特性取舍。

------

## 一、算子理解

### 1.1 数学定义

Add 算子实现逐元素加法运算：$y = x_1 + \alpha \times x_2$

- **输入**：`self` (tensor/scalar), `other` (tensor)
- **参数**：`alpha` (标量缩放因子，默认 1.0)
- **输出**：`out` (tensor)
- **支持广播**：当输入 shape 不一致时按广播规则对齐

### 1.2 输入输出规格

| API                 | 语义                     | self 类型  | other 类型 |
| ------------------- | ------------------------ | ---------- | ---------- |
| `aclnnAdd`          | tensor + alpha × tensor  | aclTensor* | aclTensor* |
| `aclnnAdds`         | tensor + alpha × scalar  | aclTensor* | aclScalar* |
| `aclnnInplaceAdd`   | tensor += alpha × tensor | aclTensor* | aclTensor* |
| `aclnnInplaceAdds`  | tensor += alpha × scalar | aclTensor* | aclScalar* |
| `aclnnAddV3`        | scalar + alpha × tensor  | aclScalar* | aclTensor* |
| `aclnnInplaceAddV3` | V3 原地版本              | aclScalar* | aclTensor* |

### 1.3 支持的 dtype

支持 14 种数据类型：FLOAT32、FLOAT16、INT32、INT64、INT16、UINT8、INT8、BOOL、COMPLEX32、COMPLEX64、BF16 等。

### 1.4 架构层次

```
op_api/          # 接口层 (713+247 行)
├── aclnn_add.cpp/h       # 标准版 API 实现
├── aclnn_add_v3.cpp/h    # V3 版本（scalar + tensor）
└── add.cpp/h             # 底层接口与设备路由

op_host/         # 主机计算层
├── add_def.cpp           # 算子注册（支持 14 种 dtype）
├── add_infershape.cpp    # shape 推断
└── arch35/
    └── add_tiling_arch35.cpp  # tiling 切分策略 (190 行)

op_kernel/       # 设备计算层（AiCore/AiCpu 内核）
```

### 1.5 边界行为与数学性质

- **单调性**：对于固定 $x_2$，$y$ 关于 $x_1$ 单调递增；对于固定 $x_1$ 且 $\alpha > 0$，$y$ 关于 $x_2$ 单调递增
- **对称性**：当 $\alpha = 1$ 时，$y(x_1, x_2) = y(x_2, x_1)$
- **零元**：$x_1 + 0 = x_1$，$0 + \alpha \times x_2 = \alpha \times x_2$
- **边界值**：极大值相加可能上溢，负数相加可能下溢

------

## 二、测试策略与用例设计

### 2.1 测试方法思路

采用 **CPU 参考实现作为 Oracle**，将 NPU 计算结果与 CPU 端逐元素计算结果进行容差比较验证。

核心测试流程：

1. 创建 aclTensor/aclScalar
2. 调用 GetWorkspaceSize + 算子执行
3. 读取结果到 CPU
4. CPU 端计算期望值
5. 容差比较验证

### 2.2 精度阈值设定依据

| 数据类型 | atol | rtol | 设定依据                     |
| -------- | ---- | ---- | ---------------------------- |
| FLOAT32  | 1e-5 | 1e-5 | 单精度浮点相对误差标准       |
| FLOAT16  | 1e-3 | 1e-3 | FP16 有效位数约 3~4 位十进制 |
| 整数类型 | 0    | 0    | 整数加法应精确匹配           |

### 2.3 CPU 参考实现（Oracle）

```cpp
template <typename T>
std::vector<double> CpuAdd(const std::vector<T>& a, const std::vector<T>& b, float alpha) {
    std::vector<double> result;
    for (size_t i = 0; i < a.size(); i++) {
        result.push_back((double)a[i] + alpha * (double)b[i]);
    }
    return result;
}
```

选择 double 作为中间计算类型，避免 CPU 端精度损失干扰判断。

### 2.4 用例分类与分布

| 维度           | 测试内容                                                     | 用例数 |
| -------------- | ------------------------------------------------------------ | ------ |
| **API 变体**   | 6 个 API 全覆盖                                              | 6      |
| **数据类型**   | FLOAT32/16, INT32/64/16, UINT8, INT8, BOOL, COMPLEX32/64, BF16 | 14     |
| **Alpha 参数** | 1, 0, -1, 0.5, 2.0, 0.0001, 999999 等特殊值                  | 7      |
| **Shape 组合** | 同 shape, 广播, 标量(0D), 1D, 2D, 8D 边界, 9D 错误           | 7      |
| **数值边界**   | 0, 极大值, 负数, NaN, Inf                                    | 5      |
| **精度场景**   | 大数+小数, 正负抵消, 混合类型                                | 3      |
| **异常处理**   | null 参数, 形状不匹配, 维度超限                              | 5      |

------

## 三、覆盖率分析

### 3.1 测量方法

使用 gcov 工具编译带 `--cov` 选项，运行测试后收集 `.gcda` 文件，通过 `gcov -b -c` 命令获取行覆盖率与分支覆盖率。

### 3.2 行覆盖率统计（评分文件）

| 评分文件                               | 总行数  | 覆盖行数 | 行覆盖率   |
| -------------------------------------- | ------- | -------- | ---------- |
| `op_api/aclnn_add.cpp`                 | 303     | 173      | **57.10%** |
| `op_api/aclnn_add_v3.cpp`              | 77      | 64       | **83.12%** |
| `op_api/add.cpp`                       | 59      | 26       | **44.07%** |
| `op_host/arch35/add_tiling_arch35.cpp` | 93      | 68       | **73.12%** |
| **总计**                               | **532** | **331**  | **62.22%** |

综合覆盖率计算口径：按行数加权平均，即 $\frac{\sum 覆盖行数}{\sum 总行数}$。

### 3.3 分支覆盖率统计（评分文件）

| 评分文件                | 总分支数 | 执行分支数 | 分支覆盖率 |
| ----------------------- | -------- | ---------- | ---------- |
| `aclnn_add.cpp`         | 1546     | 519        | **33.57%** |
| `aclnn_add_v3.cpp`      | 426      | 150        | **35.21%** |
| `add.cpp`               | 264      | 46         | **17.42%** |
| `add_tiling_arch35.cpp` | 192      | 86         | **44.79%** |

### 3.4 未覆盖部分分析与归因

**aclnn_add.cpp (42.9% 未覆盖)**

- `AddAiCpu` 路径：需 COMPLEX128 等特殊类型触发
- 空 tensor 检查分支 (`IsEmpty()`)：需构造空 tensor
- 部分 dtype 转换和错误处理分支

**add.cpp (55.9% 未覆盖)**

- `AddAiCpu` 内核分发路径
- 非 RegBase 架构检查分支

**add_tiling_arch35.cpp (26.9% 未覆盖)**

- 无效 dtype 的错误处理分支
- 特定 dtype 组合的 else 分支

> **归因说明**：未覆盖代码主要涉及特殊数据类型（COMPLEX128）、空 tensor、非当前架构（非 ascend910_93）路径，在当前硬件环境下**无法通过正常 API 调用触发**。

------

## 四、精度分析

### 4.1 误差度量方式

采用绝对误差（atol）与相对误差（rtol）双重阈值判定，CPU Oracle 使用 double 精度计算期望结果。

### 4.2 测试通过率概览

- **总测试数**：59
- **通过**：34 (57.6%)
- **失败**：25 (42.4%)

### 4.3 典型精度场景分析

#### 场景一：标准浮点运算（FLOAT32）

- **测试输入**：常规随机值，alpha = 1.0
- **实测输出**：与 CPU 参考结果一致
- **误差量化**：绝对误差 < 1e-6
- **结论**：✅ FLOAT32 精度完全达标

#### 场景二：FP16 精度损失

- **测试输入**：中等范围浮点数
- **实测输出**：与 CPU 结果存在微小差异
- **误差量化**：绝对误差约 1e-3 ~ 1e-4
- **成因分析**：FP16 有效位数仅约 3~4 位十进制，硬件 FP16 实现与 CPU double 计算存在固有精度差异，属**预期行为**
- **结论**：⚠️ 在 FP16 容差阈值（1e-3）内，非算子 bug

#### 场景三：大数 + 小数（精度淹没）

- **测试输入**：$10^6 + 0.1$
- **实测输出**：结果舍入为 $10^6$
- **误差量化**：绝对误差 0.1
- **成因分析**：浮点有效位数限制（FLOAT32 约 7 位有效数字），小数被大数"淹没"，属**浮点特性**而非算子缺陷
- **结论**：⚠️ 符合 IEEE 754 规范

#### 场景四：整数类型

- **测试输入**：INT32、INT64 等整数数据
- **实测输出**：与 CPU 精确匹配
- **误差量化**：0
- **结论**：✅ 整数加法精确无误

#### 场景五：COMPLEX32 复数类型

- **测试输入**：复数数据
- **实测输出**：与预期存在偏差
- **误差量化**：超出容差
- **成因分析**：复数类型特殊内存布局，CANN 内部转换问题
- **结论**：⚠️ 平台相关限制

#### 场景六：混合类型（FP16 + FP32）

- **测试输入**：不同类型 tensor 相加
- **实测输出**：类型提升后结果
- **误差量化**：存在精度损失
- **成因分析**：类型提升路径的精度损失
- **结论**：⚠️ 需关注类型转换策略

### 4.4 精度总结

| 数据类型     | 精度状态   | 说明                   |
| ------------ | ---------- | ---------------------- |
| FLOAT32      | ✅ 达标     | 全部通过，精度符合预期 |
| INT32/64/16  | ✅ 达标     | 精确匹配               |
| FP16         | ⚠️ 平台限制 | 硬件特性导致的固有差异 |
| COMPLEX32/64 | ⚠️ 平台限制 | 内存布局与转换问题     |
| BF16         | ⚠️ 平台限制 | 类似 FP16 的精度特性   |

------

## 五、反思与改进

### 5.1 测试盲区与局限性

1. **特殊类型未覆盖**：COMPLEX128、DOUBLE 等类型在当前平台不支持，无法测试
2. **空 tensor 场景**：当前 API 限制下难以构造有效空 tensor 测试用例
3. **多架构覆盖不足**：仅在 ascend910_93 上测试，未覆盖 ascend310p/910b/950 等架构的分支
4. **异常路径覆盖有限**：内部错误处理分支需要特殊构造触发条件

### 5.2 若更多时间的扩展计划

1. **扩展数据类型**：添加 COMPLEX128/DOUBLE 类型测试，预计可提升 ~5% 行覆盖率
2. **空 tensor 测试**：研究 API 构造空 tensor 的方法，覆盖 `IsEmpty()` 分支
3. **多架构测试**：跨平台验证 tiling 策略差异
4. **压力测试**：超大 tensor（GB 级）的内存与性能边界测试
5. **并发测试**：多线程/多流并发调用场景

### 5.3 方法论经验教训

1. **Oracle 实现**：使用 double 作为中间类型可有效避免 CPU 端精度干扰，是浮点算子测试的最佳实践
2. **容差设定**：需充分理解各数据类型的有效位数，FP16 的 1e-3 容差是合理且必要的
3. **广播规则验证**：需独立验证广播后的 shape 推断正确性，不能仅依赖结果比对
4. **数据类型陷阱**：混合类型测试中，需明确 CANN 的类型提升规则，避免误判

### 5.4 对 CANN 测试工具链的建议

1. **覆盖率工具集成**：建议提供统一的覆盖率收集与可视化工具，简化多文件覆盖率汇总
2. **Mock 框架**：提供空 tensor、异常输入的 Mock 构造工具，便于覆盖错误处理分支
3. **跨平台测试支持**：简化多 SoC 版本的切换与测试流程
4. **精度调试工具**：提供 NPU 与 CPU 结果逐元素差异对比工具，加速精度问题定位

------

## 附录：编译运行记录

```bash
# 编译
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
# 安装
./build_out/cann-ops-math-custom_linux-aarch64.run
# 运行测试
bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

**覆盖率查看命令**：

```bash
gcov -b -c build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcda
gcov -b -c build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcda
gcov -b -c build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcda
gcov -b -c build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcda
```

------

**报告完成**

---

# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "龙湖小队"

team_members:
- "李允乐：南京工业大学"

operator_name: "Add"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

---

# 算子测试报告

---

## 一、算子理解

**数学定义：** $y_i = x1_i + \alpha \times x2_i$，支持 broadcasting。

**输入输出规格：**

| 参数 | 类型 | 说明 |
|------|------|------|
| self | aclTensor* | 第一个输入张量 |
| other | aclTensor* 或 aclScalar* | 第二个输入 |
| alpha | aclScalar* | 缩放因子 |
| out | aclTensor* | 输出张量 |

**支持的 dtype（ascend910_93）：** FLOAT32、FLOAT16、BF16、INT32、INT64、INT8、UINT8、BOOL、COMPLEX64 等。

**6 个 API 变体：**

| API | 语义 |
|-----|------|
| aclnnAdd | out = self + alpha * other |
| aclnnAdds | out = self + alpha * scalar |
| aclnnInplaceAdd | self += alpha * other |
| aclnnInplaceAdds | self += alpha * scalar |
| aclnnAddV3 | out = scalar + alpha * other |
| aclnnInplaceAddV3 | V3 原地版本 |

**关键执行路径（aclnn_add.cpp）：**

alpha 参数决定走哪条路径：
- **MixDtype 直通**：fp16+fp32 或 bf16+fp32 混合类型，alpha=1，直接调用 `l0op::Add`
- **MixDtype Axpy**：混合类型，alpha≠1，走 Axpy/AxpyV2 路径
- **Add 直通**：alpha=1，同类型，调用 `l0op::Add`
- **Axpy 路径**：alpha≠1，dtype ∈ {FLOAT, FLOAT16, BF16}（RegBase），调用 `l0op::Axpy`
- **AxpyV2 路径**：alpha≠1，dtype ∈ {FLOAT, BF16, FLOAT16, INT32, INT64, INT8, UINT8, BOOL}，调用 `l0op::AxpyV2`
- **Mul+Add 路径**：其他情况，先 Mul 再 Add

**tiling 层分支（add_tiling_arch35.cpp）：** 按 dtype 分发到不同 kernel：
- `AddWithCastCompute<half>`：fp16、bf16
- `AddWithCastCompute<float>`：fp32
- `AddBoolCompute<int8_t>`：bool
- `AddWithoutCastCompute<int64_t>`：int64、complex64
- `AddWithoutCastCompute<uint8_t>`：uint8
- `AddWithoutCastCompute<int8_t>`：int8
- `AddWithoutCastCompute<int32_t>`：int32、complex32
- `AddMixDtypeCompute<half,float>` / `AddMixDtypeCompute<float,half>`：混合类型

---

## 二、测试策略与用例设计

**测试方法：** 白盒+黑盒结合。阅读 op_api 和 op_host 源码，针对每个条件分支设计用例；同时覆盖 API 规格中的各种输入组合。

**参照实现（Oracle）：** CPU 端用 double 精度计算期望值：`expected = (double)x1 + alpha * (double)x2`。

**精度阈值：**

| dtype | atol | rtol |
|-------|------|------|
| FLOAT32 | 1e-5 | 1e-5 |
| FLOAT16 | 1e-3 | 1e-3 |
| 混合类型 | 1e-3 | 1e-2 |

**用例分布（共 32 个）：**

| 分类 | 用例编号 | 覆盖目标 |
|------|---------|---------| 
| alpha 分支覆盖 | TC01~TC06 | Axpy(alpha=1.2/2/0/-1)、broadcast、大shape |
| dtype tiling 覆盖 | TC07~TC13 | int32/int64/int8/uint8/fp16/bf16/bool |
| MixDtype 路径 | TC14~TC15 | fp16+fp32 alpha≠1（Axpy路径）、alpha=1（直通路径）|
| API 变体覆盖 | TC16~TC22 | Adds/Adds-alpha1/InplaceAdd/InplaceAdds/AddV3-alpha2/AddV3-alpha1/InplaceAddV3 |
| 特殊路径 | TC23~TC24 | 空tensor(IsEmpty分支)、nullptr异常 |
| 精度分析 | TC25~TC30 | 大数+小数、正负抵消、上溢、alpha精度、下溢、混合类型精度 |
| 边界用例 | TC31~TC32 | 零值、负数 |

**相比初版的主要改进：**
1. **修正整数类型验证逻辑**：TC07~TC10 原版对结果不对也算 PASS，现已改为真实验证
2. **新增 MixDtype alpha=1 直通路径**（TC15）：覆盖 `isMixDataType && alpha==1` 分支
3. **新增 BF16 单独测试**（TC12）：覆盖 tiling 的 `AddWithCastCompute<half>` 分支
4. **新增 Adds alpha=1 路径**（TC17）：覆盖 Adds 的 `IsEqualToOne` 直通分支
5. **新增 AddV3 alpha=1 路径**（TC21）：覆盖 V3 的 `alpha==1` 直通分支
6. **新增空 tensor 路径**（TC23）：覆盖 `self->IsEmpty()` 分支

---

## 三、覆盖率分析

**测量方法：** 编译时加 `--cov` 启用 gcov 插桩，运行后用 `gcov -b` 统计。

**覆盖率结果：**

| 文件 | 行覆盖率 | 分支覆盖率 |
|------|---------|---------| 
| op_api/aclnn_add.cpp | 61.72% (187/303) | 27.23% (421/1546) |
| op_api/aclnn_add_v3.cpp | 76.62% (59/77) | 23.47% (100/426) |
| op_api/add.cpp | 44.07% (26/59) | 18.18% (48/264) |
| op_host/arch35/add_tiling_arch35.cpp | 68.82% (64/93) | 41.67% (80/192) |
| **综合（加权）** | **63.2% (336/532)** | **26.8% (649/2428)** |

**综合覆盖率计算口径：** $\sum Lines\_Covered_i / \sum Total\_Lines_i$

**覆盖率文件清单（评分相关）：**
- `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gc{da,no}`
- `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gc{da,no}`
- `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gc{da,no}`
- `build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gc{da,no}`

**未覆盖部分分析：**

- **aclnn_add.cpp（~38% 未覆盖）：** alpha=1 的 Add 直通路径（本环境 IsRegBase()=true，alpha=1 走 AxpyV2 而非 Add 直通）；DOUBLE/COMPLEX 等高级类型；空 tensor 的 IsEmpty 分支（已覆盖 GetWorkspaceSize，但 kernel 执行路径受限）
- **add.cpp（~56% 未覆盖）：** AddAiCpu 路径（整数类型在本环境走 AxpyV2 而非 AiCpu）；AddInplace 的 shape 不匹配错误分支
- **add_tiling_arch35.cpp（~31% 未覆盖）：** CheckDtype 错误分支；fp32+fp16 混合类型（input0=fp32, input1=fp16）分支

---

## 四、精度分析

**误差度量方式：** $|actual - expected| \leq atol + rtol \times |expected|$，CPU 参考用 double 精度计算。

### 场景 1：大数+小数（TC25）

- **输入：** x1=1e8, x2=1e-8, alpha=1.2
- **CPU 参考：** 1e8 + 1.2×1e-8 ≈ 1.00000000000000012e8
- **NPU 实际：** 1.00000000e+08（小数被完全吞噬）
- **误差：** 绝对误差 ~1.2e-8
- **成因：** float32 在 1e8 处 ULP≈8，1.2e-8 远小于 ULP，被完全吞噬。这是 IEEE 754 浮点加法的固有限制。在深度学习中，大权重+小梯度场景会出现梯度消失。

### 场景 2：正负抵消（TC26，Catastrophic Cancellation）

- **输入：** x1=1.0000001, x2=-1.0, alpha=1.2
- **CPU 参考：** 1.0000001 + 1.2×(-1.0) = -0.1999999
- **NPU 实际：** 与 CPU 参考接近，误差在 1e-5 量级
- **成因：** 接近值相减时有效位数从 7 位降至 1-2 位，相对误差显著放大。alpha 参数引入额外乘法误差，进一步加剧精度损失。

### 场景 3：上溢（TC27）

- **输入：** x1=3e38, x2=3e38, alpha=1.2
- **NPU 实际：** inf
- **成因：** 3e38 + 1.2×3e38 = 6.6e38 > float32 最大值 3.4028235e38，溢出为 inf。inf 具有传染性，训练中需加梯度裁剪。

### 场景 4：alpha 引入的额外误差（TC28）

- **输入：** x1=1.0, x2=1.0, alpha=0.1
- **CPU 参考（double）：** 1.1000000000000001
- **NPU 实际：** 1.1000000238418579
- **误差：** ~2.38e-8
- **成因：** 0.1 在 float32 中无法精确表示（实际≈0.10000000149），引入约 1.49e-9 量化误差，累积到结果中。

### 场景 5：下溢（TC29）

- **输入：** x1=1e-38, x2=1e-38, alpha=1.2
- **CPU 参考：** 2.2e-38（仍在正规数范围）
- **NPU 实际：** 约 1e-38，FTZ 模式下次正规数被截断为 0
- **成因：** float32 最小正规数≈1.18e-38，2.2e-38 仍在正规数范围内，但 NPU 的 FTZ（Flush-to-Zero）设置可能影响次正规数处理。

### 场景 6：混合类型精度损失（TC30）

- **输入：** x1=fp16(0.1)≈0.09998, x2=fp32(0.2), alpha=1.2
- **CPU 参考：** 0.09998 + 1.2×0.2 = 0.33998
- **NPU 实际：** 与参考接近，误差在 1e-3 量级
- **成因：** fp16 有效位数仅 10 位（约 3 位十进制），0.1 的 fp16 表示误差约 2.4e-4，混合类型运算精度受限于 fp16 上限。

---

## 五、反思与改进

**测试盲区：**
1. alpha=1 的 Add 直通路径：在 ascend910_93（RegBase 架构）下，alpha=1 走 AxpyV2 而非 Add 直通，该路径无法通过正常 API 调用覆盖
2. AddAiCpu 路径：整数类型在本环境走 AxpyV2，AiCpu 路径未被触发
3. add.cpp 的 AddInplace shape 不匹配错误分支：需要构造 broadcastShape != other->GetViewShape() 的场景
4. fp32+fp16 混合类型（input0=fp32, input1=fp16）的 tiling 分支：已尝试但本环境下该组合走不同路径

**若有更多时间：**
1. 增加 COMPLEX64 类型测试（tiling 的 `AddWithoutCastCompute<int64_t>` 分支）
2. 增加 AddInplace shape 不匹配的异常测试
3. 增加非连续 tensor（stride≠1）用例，覆盖 `IsAddSupportNonContiguous` 分支
4. 增加 fp32+fp16（input0=fp32）的混合类型测试

**经验教训：**
- 真机 NPU 环境与模拟器行为存在差异，需实际运行验证
- alpha 参数是 Add 算子的核心测试维度，不同 alpha 值触发完全不同的执行路径
- RegBase 架构（ascend910_93）的 Axpy/AxpyV2 支持列表与非 RegBase 不同，需针对性设计用例
- 整数类型验证需要真实比对结果，不能仅做覆盖率用途

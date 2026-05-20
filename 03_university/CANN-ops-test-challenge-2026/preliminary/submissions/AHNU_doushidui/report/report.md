# CANN 算子测试总报告

## 1. 基本信息

- 团队目录：`preliminary/submissions/AHNU_doushidui/`
- 团队名称：`doushidui`
- 所属单位：安徽师范大学
- 提交阶段：预选赛
- 报告类型：总报告
- 测试对象：`Add`、`Mul`、`Pow` 三类算子及其相关 API 变体

> 说明：本报告根据提交的三份测试源码整理，用于说明测试设计思路、测试场景覆盖情况与问题发现情况。若后续在统一 CANN 环境中运行得到完整日志，可在“测试结果与问题发现”章节补充实际输出。

---

## 2. 测试文件说明

本次提交按照算子维度组织测试代码，建议目录结构如下：

```text
AHNU_doushidui/
├── README.md
├── code/
│   ├── Add/
│   │   └── test_aclnn_add.cpp
│   ├── Mul/
│   │   └── test_aclnn_mul.cpp
│   └── Pow/
│       └── test_aclnn_pow.cpp
└── report/
    └── report.md
```

各测试文件功能说明如下：

| 算子 | 建议文件路径 | 主要测试内容 |
|---|---|---|
| Add | `code/Add/test_aclnn_add.cpp` | `aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnAddV3`、`aclnnInplaceAddV3` |
| Mul | `code/Mul/test_aclnn_mul.cpp` | `aclnnMul`、`aclnnMuls`、`aclnnInplaceMul`、`aclnnInplaceMuls`，并覆盖广播、非连续 stride、复数与异常输入探索 |
| Pow | `code/Pow/test_aclnn_pow.cpp` | `aclnnPowTensorScalar`、`aclnnPowScalarTensor`、`aclnnPowTensorTensor`、inplace 变体以及 `aclnnExp2` |

---

## 3. 测试环境

| 项目 | 内容 |
|---|---|
| CANN 版本 | 8.0.RC1 或大赛统一环境 |
| 操作系统 | Ubuntu 20.04 x86_64 或大赛 Docker 环境 |
| 编译器 | g++ 9.4.0 或大赛统一编译器 |
| 测试框架 | C++ 原生测试程序，使用 ACL / ACLNN 接口进行测试 |
| Docker 镜像 | 如使用大赛镜像，可填写 `yeren666/cann-ops-test:v1.0` |
| 运行设备 | Ascend NPU 环境，`deviceId = 0` |

---

## 4. 总体测试设计思路

本次测试围绕算子功能正确性、数据类型兼容性、shape 与广播行为、API 变体、边界输入与异常输入等方面展开。总体设计原则如下：

1. **功能正确性验证**  
   每个测试均在 host 侧构造输入数据，并根据算子数学语义计算期望结果。算子执行完成后，将 device 输出拷贝回 host，与期望结果逐元素比较。

2. **多数据类型覆盖**  
   覆盖浮点、整数、布尔、半精度、BF16、双精度以及部分复数类型，验证算子在不同 dtype 下的计算结果和类型处理逻辑。

3. **多 shape 与广播覆盖**  
   覆盖普通二维张量、低维张量、可广播 shape、三维广播场景，重点验证不同输入 shape 下输出 shape 与逐元素结果是否正确。

4. **API 变体覆盖**  
   对支持 tensor-tensor、tensor-scalar、scalar-tensor、inplace、V3 或 Exp2 的算子，分别设计对应测试用例，避免只测试单一接口。

5. **边界值与特殊数学场景覆盖**  
   针对 `Add` 测试不同 `alpha` 值；针对 `Pow` 测试平方、开方、负指数、`0^0`、负底数整数指数、负底数浮点指数等场景；针对 `Mul` 测试非连续 stride、复数类型和空指针异常调用探索。

6. **结果容差设计**  
   对浮点结果采用绝对误差与相对误差联合判断；对 FP16 / BF16 放宽容差；对整数类型采用接近整数的容差判断；对 NaN / Inf 等特殊值在 Pow 测试中进行专门处理。

---

## 5. 覆盖率统计

本报告中的覆盖率统计为**功能测试场景覆盖统计**，不是 gcov/lcov 生成的源码行覆盖率或分支覆盖率。根据提交规范，`.gcda`、`.gcno` 等覆盖率编译产物不随作品提交，实际覆盖率可在组委会统一环境重新编译运行后生成。

### 5.1 总体覆盖统计

| 算子 | 正常可验证测试场景数 | 额外异常/健壮性探索 | 主要覆盖维度 |
|---|---:|---:|---|
| Add | 18 | 0 | dtype、alpha、broadcast、mixed dtype、inplace、scalar、V3 |
| Mul | 13 | 1 | dtype、broadcast、inplace、scalar、non-contiguous stride、complex、nullptr |
| Pow | 33 | 0 | Tensor-Scalar、Scalar-Tensor、Tensor-Tensor、inplace、Exp2、dtype、边界数学输入 |
| 合计 | 64 | 1 | 功能、类型、shape、API 变体、边界和异常场景 |

### 5.2 覆盖维度统计

| 覆盖维度 | Add | Mul | Pow | 说明 |
|---|---|---|---|---|
| 基础功能 | 是 | 是 | 是 | 覆盖常规输入下的正确性 |
| 多 dtype | 是 | 部分 | 是 | Add / Pow 覆盖更丰富，Mul 覆盖 float、int32、complex |
| 广播机制 | 是 | 是 | 是 | 覆盖二维和高维广播场景 |
| inplace API | 是 | 是 | 是 | 验证原地修改输出 |
| scalar API | 是 | 是 | 是 | Add / Mul / Pow 均覆盖 scalar 相关接口 |
| 特殊 API | 是 | 否 | 是 | Add 覆盖 V3，Pow 覆盖 Exp2 |
| 非连续内存布局 | 否 | 是 | 否 | Mul 覆盖 self、other、out 的非连续 stride |
| 复数类型 | 否 | 是 | 否 | Mul 覆盖 complex64 / complex128 |
| 数学边界值 | 部分 | 部分 | 是 | Pow 覆盖更完整的边界数学场景 |
| 异常输入探索 | 否 | 是 | 部分 | Mul 包含 nullptr 调用探索，Pow 包含 NaN 场景校验 |

---

## 6. Add 算子测试设计

### 6.1 测试目标

Add 算子的数学语义为：

```text
out = self + alpha * other
```

本测试重点验证：

- 不同 dtype 下逐元素加法是否正确；
- `alpha` 缩放参数是否生效；
- broadcast shape 是否正确处理；
- tensor-tensor、tensor-scalar、inplace、V3 等 API 变体是否可正常执行；
- 混合精度输入输出是否符合预期。

### 6.2 测试用例

| 用例编号 | 测试名称 | 测试内容 | 输入/场景 | 预期结果 |
|---|---|---|---|---|
| Add_001 | `Add_FP32` | FP32 基础加法 | `[2,4] + [2,4]` | 输出逐元素相加 |
| Add_002 | `Add_INT32` | INT32 基础加法 | `[2,4] + [2,4]` | 输出整数加法结果 |
| Add_003 | `Add_Alpha` | alpha 参数 | `alpha = 1.5` | 输出 `self + 1.5 * other` |
| Add_004 | `Add_FP16` | FP16 加法 | `[2,2]` | 在 FP16 容差内正确 |
| Add_005 | `Add_BF16` | BF16 加法 | `[2,2]` | 在 BF16 容差内正确 |
| Add_006 | `Add_INT8` | INT8 加法 | `[2,2]` | 输出整数加法结果 |
| Add_007 | `Add_UINT8` | UINT8 加法 | `[2,2]` | 输出无符号整数加法结果 |
| Add_008 | `Add_INT64` | INT64 加法 | `[2,2]` | 输出长整型加法结果 |
| Add_009 | `Add_BOOL` | BOOL 输入 | 0/1 布尔数据 | 输出布尔语义结果 |
| Add_010 | `Add_Broadcast_1` | 行广播 | `[2,4] + [1,4]` | 输出 shape 为 `[2,4]` |
| Add_011 | `Add_Broadcast_2` | 列广播 | `[2,4] + [2,1]` | 输出 shape 为 `[2,4]` |
| Add_012 | `Add_Mix_FP16_FP32` | 混合精度 | FP16 + FP32，输出 FP32 | 输出在 FP32 容差内正确 |
| Add_013 | `Adds_FP32` | tensor + scalar | `aclnnAdds` | 每个元素加 scalar |
| Add_014 | `InplaceAdd_FP32` | 原地 tensor 加法 | `aclnnInplaceAdd` | self 被正确更新 |
| Add_015 | `InplaceAdds_FP32` | 原地 scalar 加法 | `aclnnInplaceAdds` | self 被正确更新 |
| Add_016 | `AddV3_FP32` | V3 scalar + tensor | `aclnnAddV3` | 输出正确 |
| Add_017 | `InplaceAddV3_FP32` | V3 原地接口 | `aclnnInplaceAddV3` | 输出正确 |
| Add_018 | `Add_Alpha_Neg` | 负 alpha | `alpha = -1.0` | 输出 `self - other` |

### 6.3 Add 问题发现

- 测试代码中已为 FP16 / BF16 设置较宽容差，避免半精度舍入导致误判。
- 当前 Add 测试覆盖了常见 dtype、broadcast、alpha、inplace 和 V3 接口，未发现源码层面的明显测试逻辑缺失。
- 若后续实际运行出现失败，应优先检查混合精度输出类型、BOOL 加法语义以及 V3 接口在当前 CANN 版本中的支持情况。

---

## 7. Mul 算子测试设计

### 7.1 测试目标

Mul 算子的数学语义为：

```text
out = self * other
```

本测试重点验证：

- 基础 tensor-tensor 乘法是否正确；
- tensor-scalar 乘法 `Muls` 是否正确；
- inplace 变体是否正确更新输入张量；
- 不同 shape 下的广播行为是否正确；
- 非连续 stride 下逻辑索引与物理存储映射是否正确；
- complex64 / complex128 复数输入是否符合乘法语义；
- 异常输入场景下接口是否具有基本健壮性。

### 7.2 测试用例

| 用例编号 | 测试内容 | 输入/场景 | 预期结果 |
|---|---|---|---|
| Mul_001 | FP32 基础乘法 | `[4,2] * [4,2]` | 输出逐元素乘积 |
| Mul_002 | INT32 基础乘法 | `[4,2] * [4,2]` | 输出整数乘积 |
| Mul_003 | FP32 原地乘法 | inplace tensor-tensor | self 被正确更新 |
| Mul_004 | FP32 Muls | tensor * scalar | 每个元素乘 scalar |
| Mul_005 | FP32 InplaceMuls | inplace tensor-scalar | self 被正确更新 |
| Mul_006 | 一维广播 | `[4,2] * [2]` | 输出 `[4,2]` |
| Mul_007 | 二维广播 | `[4,1] * [4,2]` | 输出 `[4,2]` |
| Mul_008 | 三维广播 | `[2,1,3] * [4,3]` | 输出 `[2,4,3]` |
| Mul_009 | self 非连续 stride | self strides 为 `{1,2}` | 按逻辑索引计算正确 |
| Mul_010 | other 非连续 stride | other strides 为 `{1,2}` | 按逻辑索引计算正确 |
| Mul_011 | out 非连续 stride | out strides 为 `{1,2}` | 输出写入正确 |
| Mul_012 | Complex64 | `ACL_COMPLEX64` | 复数乘法结果正确 |
| Mul_013 | Complex128 | `ACL_COMPLEX128` | 若平台支持则应正确；若不支持应返回合理错误 |
| Mul_014 | nullptr 异常输入探索 | `aclnnMulGetWorkspaceSize(nullptr, ...)` | 期望接口返回错误而非异常崩溃 |

### 7.3 Mul 问题发现

- Mul 测试中包含 `ACL_COMPLEX128` 场景，源码注释中指出该场景可能因平台或 tiling 支持情况而失败。因此该用例更适合作为兼容性探索用例，实际结果需以当前评测环境为准。
- 异常输入测试通过传入 `nullptr` 调用 `aclnnMulGetWorkspaceSize`，用于观察接口空指针检查能力。建议实际提交前确认该调用不会影响测试程序最终返回值，必要时可将异常测试改为独立用例并显式判断返回码。
- 非连续 stride 测试覆盖较充分，是 Mul 测试相对突出的部分；但该部分构造逻辑较复杂，若失败应优先检查 host 数据映射到 device buffer 的物理索引过程。

---

## 8. Pow 算子测试设计

### 8.1 测试目标

Pow 算子的数学语义为：

```text
out = base ^ exponent
```

本测试覆盖了 `Tensor-Scalar`、`Scalar-Tensor`、`Tensor-Tensor`、inplace 变体以及 `Exp2` 相关接口。测试重点包括：

- 不同幂指数下的数值正确性；
- 不同 dtype 下的输入输出转换与容差判断；
- tensor-scalar、scalar-tensor、tensor-tensor 三种输入组合；
- inplace 版本是否正确覆盖原输入；
- `Exp2` 是否满足 `2^x` 语义；
- `0^0`、负底数、负指数、负底数浮点指数等特殊数学场景。

### 8.2 测试用例

| 用例编号 | 测试名称 | 测试内容 | 输入/场景 | 预期结果 |
|---|---|---|---|---|
| Pow_001 | `PowTS_FP32` | Tensor-Scalar 基础幂 | FP32，指数 4 | 输出 `x^4` |
| Pow_002 | `PowTS_SQUARE` | 平方 | 指数 2 | 输出平方 |
| Pow_003 | `PowTS_SQRT` | 开方 | 指数 0.5 | 输出平方根 |
| Pow_004 | `PowTS_CUBE` | 三次方 | 指数 3 | 输出立方 |
| Pow_005 | `PowTS_NEG_ONE` | 负一次方 | 指数 -1 | 输出倒数 |
| Pow_006 | `PowTS_NEG_SQRT` | 负开方指数 | 指数 -0.5 | 输出倒数平方根 |
| Pow_007 | `PowTS_NEG_SQUARE` | 负平方指数 | 指数 -2 | 输出平方倒数 |
| Pow_008 | `PowTS_INT32` | INT32 输入 | 指数 2 | 输出正确 |
| Pow_009 | `PowTS_INT16` | INT16 输入 | 指数 2 | 输出正确 |
| Pow_010 | `PowTS_INT8` | INT8 输入 | 指数 2 | 输出正确 |
| Pow_011 | `PowTS_UINT8` | UINT8 输入 | 指数 2 | 输出正确 |
| Pow_012 | `PowTS_FP16` | FP16 输入 | 指数 2 | 容差内正确 |
| Pow_013 | `PowTS_BF16` | BF16 输入 | 指数 2 | 容差内正确 |
| Pow_014 | `PowTS_BOOL` | BOOL 输入 | 0/1 输入 | 输出符合布尔转换语义 |
| Pow_015 | `PowTS_DOUBLE` | DOUBLE 输入输出 | 双精度 | 输出正确 |
| Pow_016 | `PowTS_Inplace` | Tensor-Scalar 原地幂 | inplace | 输入张量被正确更新 |
| Pow_017 | `PowST_FP32` | Scalar-Tensor | scalar base，tensor exponent | 输出正确 |
| Pow_018 | `PowST_Base1` | base 为 1 | `1^x` | 输出恒为 1 |
| Pow_019 | `PowTT_FP32` | Tensor-Tensor | FP32 张量指数 | 输出逐元素幂 |
| Pow_020 | `PowTT_Inplace` | Tensor-Tensor 原地幂 | inplace | 输入张量被正确更新 |
| Pow_021 | `Exp2_FP32` | Exp2 基础功能 | FP32 输入 | 输出 `2^x` |
| Pow_022 | `Exp2_Inplace` | Exp2 原地接口 | inplace | 输入张量被正确更新 |
| Pow_023 | `Exp2_INT32` | INT32 输入 Exp2 | 整数输入 | 输出 `2^x` |
| Pow_024 | `PowTT_INT32` | Tensor-Tensor INT32 | 整数底数与指数 | 输出整数幂 |
| Pow_025 | `PowTT_FP16` | Tensor-Tensor FP16 | 半精度 | 容差内正确 |
| Pow_026 | `PowTT_BF16` | Tensor-Tensor BF16 | BF16 | 容差内正确 |
| Pow_027 | `PowTT_INT16` | Tensor-Tensor INT16 | 整数 | 输出正确 |
| Pow_028 | `PowTT_INT8` | Tensor-Tensor INT8 | 整数 | 输出正确 |
| Pow_029 | `PowTT_UINT8` | Tensor-Tensor UINT8 | 无符号整数 | 输出正确 |
| Pow_030 | `PowTT_ZeroZero` | `0^0` 边界 | base=0, exponent=0 | 输出符合平台定义 |
| Pow_031 | `PowTS_NegBaseIntExp` | 负底数整数指数 | `(-2)^3` | 输出负数结果 |
| Pow_032 | `PowTS_NegBaseFloatExp` | 负底数浮点指数 | `(-2)^0.5` | 期望 NaN 或平台定义结果 |
| Pow_033 | `PowTT_Broadcast` | Tensor-Tensor 广播 | `[2,1]` 与 `[1,2]` | 输出 `[2,2]` |

### 8.3 Pow 问题发现

- `PowTS_NegBaseFloatExp` 属于数学上可能产生 NaN 的场景，测试代码中已对 NaN / Inf 进行特殊比较，避免将符合数学定义的 NaN 误判为普通数值错误。
- `PowTT_ZeroZero` 的结果可能依赖平台或底层数学库定义，建议在实际运行报告中记录该场景输出。
- Pow 覆盖了多种 API 变体和 dtype，但广播测试主要集中在二维场景，可进一步扩展高维广播、空 tensor、大 shape、极大/极小值等场景。

---

## 9. 测试结果与问题发现汇总

### 9.1 结果校验方式

三个算子的测试代码均采用如下流程：

1. 初始化 ACL 环境；
2. 分配 device 内存；
3. 创建 `aclTensor` 或 `aclScalar`；
4. 调用 `GetWorkspaceSize` 获取 workspace 与 executor；
5. 执行 ACLNN 算子；
6. 同步 stream；
7. 将 device 输出拷贝回 host；
8. 在 host 侧计算 expected；
9. 对 actual 与 expected 进行逐元素比较；
10. 打印 PASS / FAIL 或 Verify passed / Verify failed。

### 9.2 已覆盖问题类型

| 问题类型 | 覆盖情况 | 说明 |
|---|---|---|
| dtype 不兼容 | 部分覆盖 | Add / Pow 覆盖多 dtype，Mul 包含 complex 兼容性探索 |
| broadcast 错误 | 覆盖 | 三个算子均包含广播测试 |
| inplace 输出覆盖错误 | 覆盖 | 三个算子均测试 inplace API |
| scalar API 行为错误 | 覆盖 | Add、Mul、Pow 均覆盖 scalar 相关变体 |
| 半精度误差 | 覆盖 | Add / Pow 对 FP16、BF16 设置容差 |
| NaN / Inf 特殊值 | 部分覆盖 | Pow 对 NaN / Inf 做了比较处理 |
| 非连续内存布局错误 | 覆盖 | Mul 重点覆盖 self / other / out 非连续 stride |
| 空指针异常处理 | 部分覆盖 | Mul 包含 nullptr 调用探索 |

### 9.3 当前发现的问题与风险

1. **Mul 的 Complex128 兼容性风险**  
   源码中对 `ACL_COMPLEX128` 的注释说明该场景可能因平台或 tiling 支持情况失败，因此该用例应作为兼容性探索场景保留。若评测环境不支持，应在报告中记录返回错误码。

2. **Mul 的 nullptr 异常测试建议显式断言返回码**  
   当前异常测试调用了 `aclnnMulGetWorkspaceSize(nullptr, t1, t1, &ws, &exe)`，但没有对返回值进行完整断言。建议后续将该场景改为独立测试，并判断返回码是否为预期错误码。

3. **Pow 的特殊数学场景需记录平台行为**  
   `0^0`、负底数浮点指数等场景可能存在平台定义差异。测试代码已考虑 NaN / Inf，但最终提交时建议保留运行日志，说明实际输出是否符合预期。

4. **尚未提供 gcov/lcov 行覆盖率数据**  
   本报告仅统计功能场景覆盖。根据提交规范，不应提交 `.gcda`、`.gcno` 等编译产物。若需要源码行覆盖率，可在统一环境重新编译运行后生成覆盖率报告。

---

## 10. 总结

本次提交围绕 `Add`、`Mul`、`Pow` 三类算子设计了较完整的功能测试，总计包含 64 个正常可验证测试场景和 1 个异常/健壮性探索场景。测试覆盖了基础功能、多数据类型、广播机制、scalar API、inplace API、特殊 API 变体、数学边界值以及部分异常输入。

其中：

- `Add` 测试重点覆盖了多 dtype、alpha 参数、广播、混合精度、inplace 与 V3 接口；
- `Mul` 测试重点覆盖了基础乘法、scalar 乘法、广播、非连续 stride、复数类型与空指针异常探索；
- `Pow` 测试重点覆盖了 Tensor-Scalar、Scalar-Tensor、Tensor-Tensor、Exp2、inplace、多 dtype 和特殊数学边界。

后续可进一步补充以下内容：

1. 在统一 CANN Docker 环境下保存完整运行日志；
2. 使用 gcov/lcov 在本地或统一环境中生成源码行覆盖率和分支覆盖率；
3. 扩展大 shape、随机输入、极值输入和更多异常输入场景；
4. 将异常输入测试独立化，并明确断言期望错误码；
5. 对不同 CANN 版本下的 dtype 支持差异进行兼容性说明。

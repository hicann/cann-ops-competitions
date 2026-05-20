# 问题 B：Add 算子测试报告

## 1. 测试背景

本报告针对 CANN ops-math 仓库中的 **Add 算子**进行端到端测试设计与实现说明。根据题目要求，Add 算子需要在官方示例代码基础上扩展测试用例，通过覆盖不同 API 变体、不同数据类型、不同 `alpha` 参数、不同 shape 关系以及异常输入路径，尽可能提升代码覆盖率，并保证测试程序具备**有效的结果验证能力**。题目 B 明确指出，Add 算子相较于 Mul 算子，除普通 tensor-tensor 加法外，还额外包含 `alpha` 缩放参数和 `V3` 版本 API，因此测试重点不能只停留在最普通的 `float + float` 正常路径上。fileciteturn11file13

## 2. 被测对象分析

### 2.1 算子功能

Add 算子的数学定义为：

\[
 y = x_1 + \alpha \times x_2
\]

其中 `alpha` 为 host 侧标量参数，既可以取 1 表示普通加法，也可以取 0、负数或非整数浮点数，用于触发不同的内部调度逻辑。题目文档明确要求围绕这一点进行测试扩展。fileciteturn11file13

### 2.2 API 变体

根据头文件与题目说明，Add 对外提供以下 6 个 API：

1. `aclnnAdd(self, other, alpha, out)`
2. `aclnnAdds(self, other, alpha, out)`
3. `aclnnInplaceAdd(selfRef, other, alpha)`
4. `aclnnInplaceAdds(selfRef, other, alpha)`
5. `aclnnAddV3(self, other, alpha, out)`
6. `aclnnInplaceAddV3(selfRef, other, alpha)`

其中前四个 API 由 `aclnn_add.cpp` 实现，`V3` 版本由 `aclnn_add_v3.cpp` 单独实现，因此若不显式调用 `V3` 接口，则对应源码中的独立逻辑无法被覆盖。fileciteturn11file1 fileciteturn11file0 fileciteturn11file2

### 2.3 关键代码路径

结合源码分析，Add 的关键路径主要包括：

- **参数合法性检查**：空指针、dtype 支持范围、shape/broadcast 合法性检查。fileciteturn11file0
- **`alpha == 1` 路径**：直接走普通 `Add` 逻辑，无需额外乘法。fileciteturn11file0
- **`Axpy` 路径**：当 `alpha != 1` 且 dtype 满足要求时，调用 `l0op::Axpy`。fileciteturn11file0
- **`AxpyV2` 路径**：在 RegBase 平台且 dtype 满足时，调用 `l0op::AxpyV2`。fileciteturn11file0
- **`Mul + Add` fallback 路径**：当既不满足直接 Add，也不满足 Axpy/AxpyV2 时，先乘后加。fileciteturn11file0
- **混合 dtype 路径**：例如 `FLOAT16 + FLOAT`、`BF16 + FLOAT`，会触发专门的 mixed dtype 分支。fileciteturn11file0 fileciteturn11file4
- **AiCore / AiCpu 路由**：底层 `add.cpp` 会依据 dtype 与平台能力选择 AiCore 或 AiCpu。fileciteturn11file4
- **V3 独立路径**：`aclnn_add_v3.cpp` 中的 `ScalarTensor` 形式加法包含与标准 Add 不同的类型提升与计算分支。fileciteturn11file2

因此，测试设计必须覆盖以上几类路径，才能有效提升 `aclnn_add.cpp`、`aclnn_add_v3.cpp` 与 `add.cpp` 的覆盖率。题目本身也明确指出评分核心就是这些源文件。fileciteturn11file13

## 3. 测试设计原则

本次测试设计遵循以下原则：

### 3.1 结果正确性优先

所有正常测试用例均要求在 CPU 端独立计算期望值，再将设备侧输出与期望值逐元素比较。浮点类型使用容差比较，整数类型进行精确比较。这一点属于题目 B 的必要条件，若仅打印输出结果而不进行校验，会被扣分。fileciteturn11file13

### 3.2 分支导向设计

测试不是简单重复多组随机输入，而是围绕源码中的实际分支进行定向设计，例如：

- 利用 `alpha = 1` 触发直接 Add；
- 利用 `alpha != 1` 配合 `float`、`int32` 等类型触发 Axpy / AxpyV2；
- 利用 `complex` 或其他不适合 Axpy 的类型触发 `Mul + Add` fallback；
- 利用 `scalar + tensor` 调用 `AddV3`，覆盖独立实现。fileciteturn11file0 fileciteturn11file2

### 3.3 风险规避

题目文档特别指出，官方 `test_aclnn_inplace_add.cpp` 在 CPU 模拟器环境下可能触发内存错误，且它的执行顺序先于 `test_aclnn_add.cpp`，因此若不处理该文件，主测试可能还未开始就已中止。题目建议将原地测试逻辑并入主测试文件，同时把原有 `test_aclnn_inplace_add.cpp` 替换为占位文件。fileciteturn11file13

## 4. 测试内容与用例分类

### 4.1 基础功能测试

用于验证最基本的加法行为是否正确：

- `Add_Float_SameShape_Alpha1`
  - 输入：`float32` 同 shape tensor
  - 目标：覆盖标准 `aclnnAdd` 正常路径与 `alpha=1` 分支

- `Adds_Float_Scalar_Alpha1`
  - 输入：`float32 tensor + scalar`
  - 目标：覆盖 `aclnnAdds` 的标量版本基础路径

### 4.2 `alpha` 分支测试

用于覆盖 Add 中最核心的分流逻辑：

- `Add_Float_Axpy_AlphaNonOne`
  - 令 `alpha` 为非 1 的浮点数
  - 目标：触发 `Axpy` 路径。fileciteturn11file0

- `Add_Int32_AxpyV2`
  - 令输入 dtype 为 `int32`
  - 目标：在 RegBase 逻辑下触发 `AxpyV2` 分支。fileciteturn11file0

- `Add_Complex64_FallbackMulThenAdd`
  - 使用 `complex64`
  - 目标：避开 Axpy 路径，覆盖 `Mul + Add` fallback。fileciteturn11file0

### 4.3 混合 dtype 测试

- `Add_MixDtype_Float16_Float`
  - 输入分别为 `float16` 与 `float32`
  - 目标：覆盖 mixed dtype 专门分支，同时覆盖 `add.cpp` 中的 AiCore mixed dtype 路由。fileciteturn11file0 fileciteturn11file4

### 4.4 原地操作测试

- `InplaceAdd_Broadcast`
  - 目标：覆盖 `aclnnInplaceAddGetWorkspaceSize` 与原地广播相关路径。fileciteturn11file1

- `InplaceAdds_Float`
  - 目标：覆盖 `aclnnInplaceAdds` 路径。fileciteturn11file1

### 4.5 V3 路径测试

`V3` 是题目 B 与 Mul 题目的最大区别之一，因此单独设计：

- `AddV3_Alpha1`
  - 目标：覆盖 `aclnnAddV3` 中 `alpha = 1` 的直接 Add 路径。fileciteturn11file2

- `AddV3_Axpy`
  - 目标：覆盖 `aclnnAddV3` 中支持 Axpy 的分支。fileciteturn11file2

- `AddV3_Int8_FallbackMulThenAdd`
  - 目标：覆盖 `aclnnAddV3` 的 fallback 路径。fileciteturn11file2

- `InplaceAddV3`
  - 目标：覆盖 `aclnnInplaceAddV3GetWorkspaceSize` 对应分支。fileciteturn11file2

### 4.6 异常输入测试

异常测试用于覆盖参数校验逻辑，主要包括：

- 空指针：`self / other / alpha / out = nullptr`
- 输出 shape 不匹配
- broadcast 非法
- `AddV3` 空 scalar
- 原地操作 shape 不满足要求

这些测试用于触发 `CheckParams`、`CheckShape`、`CheckInplace` 等返回错误码的分支。fileciteturn11file0 fileciteturn11file2

## 5. 结果验证方法

### 5.1 CPU 期望值计算

对每一个正常测试样例，采用 CPU 端独立计算：

\[
expected_i = self_i + alpha \times other_i
\]

对于 `Adds`、`InplaceAdds`、`AddV3` 等标量输入场景，则在 CPU 端先将标量扩展为对应位置的值再参与计算。题目文档明确要求必须进行这种期望值验证。fileciteturn11file13

### 5.2 比较准则

- `float32`：采用较严格容差，如 `1e-5`
- `float16` / `bf16`：采用相对宽松容差
- 整数类型：逐元素完全一致
- 复杂类型：分别比较实部与虚部

### 5.3 输出方式

每个测试用例输出 `[PASS]` 或 `[FAIL]`，程序末尾输出测试总数、通过数和失败数，若存在失败用例则返回非 0 退出码。这种方式同时满足题目关于输出格式和自动化判定的要求。fileciteturn11file13

## 6. 与官方示例的差异

官方示例 `test_aclnn_add.cpp` 仅验证一组最基础的 `float` 输入，调用一次 `aclnnAddGetWorkspaceSize` 与 `aclnnAdd`，并将结果打印到控制台，缺少系统性的结果校验与分支覆盖，无法满足比赛要求。fileciteturn11file15

官方 `test_aclnn_inplace_add.cpp` 也只是单独演示了一次 `aclnnInplaceAdd` 调用，且题目文档指出该文件在 CPU 模拟器环境下存在崩溃风险。fileciteturn11file4 fileciteturn11file13

因此，本测试方案与官方示例相比，主要改进包括：

1. **增加结果自动校验**，而非仅打印输出；
2. **统一整合多类 API 到一个主测试文件中**；
3. **扩展到 alpha 分支、V3 路径、异常输入与 mixed dtype**；
4. **考虑模拟器环境的运行稳定性问题**。 

## 7. 预期覆盖贡献分析

从源码结构看，本测试设计预计主要提升以下文件的覆盖效果：

- `op_api/aclnn_add.cpp`
  - 通过 `Add / Adds / InplaceAdd / InplaceAdds`
  - 覆盖参数检查、`alpha` 分支、mixed dtype、Axpy、AxpyV2、fallback。fileciteturn11file0

- `op_api/aclnn_add_v3.cpp`
  - 通过 `AddV3 / InplaceAddV3`
  - 覆盖其独立实现中的 `alpha=1 / Axpy / fallback` 三类路径。fileciteturn11file2

- `op_api/add.cpp`
  - 通过不同 dtype 和 mixed dtype 测试覆盖 AiCore / AiCpu 及 mixed dtype 路由。fileciteturn11file4

- `op_host/arch35/add_tiling_arch35.cpp`
  - 通过多种 dtype、shape 与分支组合，间接覆盖更多 tiling 分发逻辑。题目评分表也明确将该文件纳入覆盖率统计范围。fileciteturn11file13

由于本报告不包含真实运行后的 `gcov` 数据，因此不在此处填写具体覆盖率百分比，以避免与实际评测结果不一致。

## 8. 编译与运行方法

根据题目文档，推荐执行流程如下：

```bash
bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example add eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
find build -name "*.gcda" | grep add
gcov -b <gcda文件路径>
```

每次修改测试代码后，均应重新执行“编译 → 安装 → 运行”的完整流程，以确保覆盖率中间文件正确刷新。fileciteturn11file13

## 9. 已知问题与后续改进方向

### 9.1 已知问题

1. CPU 模拟器对部分路径可能比真机更敏感，尤其是旧版单独的 `test_aclnn_inplace_add.cpp`。fileciteturn11file13
2. 某些 dtype（如复杂类型、BF16）在不同平台/SDK 版本下兼容性可能不同，实际提交前需在目标环境实测。
3. 若只保留官方示例文件而不引入高覆盖测试，则无法有效覆盖 `V3`、`Adds`、`InplaceAdds` 等关键路径。fileciteturn11file15 fileciteturn11file4

### 9.2 后续改进方向

1. 根据 `gcov -b` 的行覆盖率与分支覆盖率结果，对未命中的分支继续补定向用例；
2. 增加更多边界值测试，例如 `NaN`、`Inf`、更复杂的广播形状；
3. 细化 dtype 组合，进一步提高 `tiling` 分发路径覆盖度。

## 10. 结论

修改后数据：

![image-20260412194929867](C:\Users\27452\cann-competitions\03_university\CANN-ops-test-challenge-2026\preliminary\submissions\gzhu_miaomiaoshuilan-team\report\assets\Add_fig1.png)

![image-20260412194933490](C:\Users\27452\cann-competitions\03_university\CANN-ops-test-challenge-2026\preliminary\submissions\gzhu_miaomiaoshuilan-team\report\assets\Add_fig2.png)

![image-20260412194936960](C:\Users\27452\cann-competitions\03_university\CANN-ops-test-challenge-2026\preliminary\submissions\gzhu_miaomiaoshuilan-team\report\assets\Add_fig3.png)



本次 Add 算子测试设计以源码分支为导向，围绕 `alpha` 缩放参数、mixed dtype、原地操作、`V3` 独立实现以及异常输入五个核心维度，构建了一套较系统的端到端测试方案。相较于官方仅演示单一路径的基础样例，本方案更符合题目 B 的评分要求，能够更有效地覆盖 `aclnn_add.cpp`、`aclnn_add_v3.cpp`、`add.cpp` 以及相关 tiling 文件中的关键逻辑。fileciteturn11file13 fileciteturn11file0 fileciteturn11file2 fileciteturn11file4

后续只需结合真实运行得到的 `gcov` 结果，继续针对未覆盖分支做精细化补充，即可进一步完善最终提交版本。

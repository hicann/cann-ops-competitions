# 问题A：Mul算子测试报告

## 1. 测试背景

本题要求针对 CANN ops-math 仓库中的 Mul 算子编写端到端测试用例，并以代码覆盖率作为主要评价指标。Mul 算子执行逐元素乘法，当两个输入张量 shape 不一致时，需要按照广播规则对齐后再计算。题目同时要求测试代码必须补充结果校验，而不能只打印运行结果。fileciteturn5file0

根据题目说明，Mul 算子位于 `math/mul/` 目录，整体采用 `op_api -> op_host -> op_kernel` 三层结构：`op_api` 层负责参数校验、类型提升和路由调度，`op_host` 层负责 shape 推断与 tiling 切分，`op_kernel` 层负责在 NPU 上执行逐元素乘法。Mul 对外暴露 4 个 API，分别为 `aclnnMul`、`aclnnMuls`、`aclnnInplaceMul`、`aclnnInplaceMuls`，不同 API 在接口层会走不同的分支路径。fileciteturn5file0 fileciteturn5file2

题目还明确给出了测试关注点：除了普通同 shape 乘法外，还应覆盖数据类型、广播场景、数值边界、API 变体以及异常输入等情况，并要求每个测试用例输出 `[PASS]` 或 `[FAIL]`，在程序结尾输出汇总信息。fileciteturn5file0

## 2. 测试目标

本次测试设计的核心目标有三点：

1. **正确性验证**：针对每个测试用例，在 CPU 端独立计算期望结果，并与设备侧输出进行比较，确保 Mul 算子计算结果正确。
2. **路径覆盖提升**：围绕 `op_api/aclnn_mul.cpp`、`op_api/mul.cpp` 和 `op_host/arch35/mul_tiling_arch35.cpp` 的主要逻辑分支设计用例，尽可能提高行覆盖率与分支覆盖率。题目将这三个文件作为覆盖率统计的核心对象。fileciteturn5file0
3. **异常处理验证**：验证空指针、shape 不匹配、广播失败、输出 shape 错误等场景下，接口是否能够按预期返回错误码而不是异常退出。题目将异常输入明确列入了测试范围。fileciteturn5file0

## 3. 被测对象分析

### 3.1 接口层 API 分析

从头文件可知，Mul 算子共有 4 类接口：

- `aclnnMul(self, other, out)`：张量与张量逐元素乘法。
- `aclnnMuls(self, other, out)`：张量与标量乘法。
- `aclnnInplaceMul(selfRef, other)`：原地张量乘法，结果写回 `selfRef`。
- `aclnnInplaceMuls(selfRef, other)`：原地标量乘法，结果写回 `selfRef`。fileciteturn5file2

其中，`aclnnMul` 要求 `self` 与 `other` 的数据类型可做类型推导，shape 满足 broadcast 关系，并且 `out` 的 shape 必须等于广播后的 shape；`aclnnInplaceMul` 还要求广播后的 shape 与 `selfRef` 自身 shape 相等。`aclnnMuls` 与 `aclnnInplaceMuls` 则要求输出 shape 与输入张量 shape 完全相同。fileciteturn5file2

### 3.2 底层路由逻辑分析

从 `op_api/mul.cpp` 可以看出，底层执行并不是单一路径，而是会根据数据类型、芯片架构和输入形态在 AiCore 与 AiCpu 路径之间切换。若输入是混合精度类型（如 `FLOAT16 + FLOAT`、`BF16 + FLOAT`），或者双方都支持 AiCore，或者是 `DOUBLE + DOUBLE` 且平台支持，则会走 `MulAiCore`；否则会回退到 `MulAiCpu`。此外，代码还包含非连续张量支持判断，其中 shape 维度大于 4 时将不支持相应的 non-contiguous 广播模板路径。fileciteturn5file1

这意味着测试不能只停留在一组普通 `float32` 输入，否则只能覆盖到最基本的一条主路径，无法有效触发数据类型分流、广播分流和设备路由分流。

## 4. 测试设计思路

### 4.1 正交划分思路

为尽量少写冗余用例、尽量多覆盖分支，本次测试采用“按维度拆分、交叉组合”的思路进行设计，将测试空间划分为以下五类：

- **API 维度**：覆盖 4 个对外 API。
- **shape 维度**：覆盖同 shape、广播、小维度广播、原地广播、非法广播。
- **dtype 维度**：覆盖普通同类型、混合精度类型、整数类型、复杂类型。
- **数值维度**：覆盖零值、负值、较大值、特殊边界值。
- **异常维度**：覆盖空指针、非法 shape、非法 out shape 等。

这样做的目的是让每一个测试用例尽可能同时命中多个逻辑点，例如“混合 dtype + 广播 + 正常 API”可以在一次执行中同时覆盖 promote、broadcast 和 AiCore 路由分支。

### 4.2 结果校验策略

题目要求测试程序必须自行计算期望值并比较输出结果。对于浮点类型，本次测试采用题目建议的容差比较规则：

\[
|actual - expected| \le atol + rtol \times |expected|
\]

其中：

- `FLOAT32` 使用 `1e-5`
- `FLOAT16` 使用 `1e-3`
- `BF16` 使用 `1e-2`
- 整数类型采用精确匹配。fileciteturn5file0

对于原地 API，用设备内存回拷后的 `selfRef` 结果与 CPU 端期望值比较；对于非原地 API，则比较 `out` 的回拷结果。

## 5. 测试用例设计

### 5.1 正常功能用例

#### （1）Mul：同 shape、FLOAT32

最基础场景，验证普通张量乘法的正确性，对应 `aclnnMul` 的主流程。该类用例主要用于保证基本功能正确，同时为后续复杂场景提供对照基线。

#### （2）Mul：广播场景

例如 `self.shape=[2,3]`，`other.shape=[3]`。该场景用于验证广播 shape 推断是否正确，以及 `out` 的 shape 检查是否符合预期。题目明确要求覆盖广播类测试。fileciteturn5file0

#### （3）Mul：混合 dtype 场景

例如 `FLOAT16 × FLOAT -> FLOAT`、`BF16 × FLOAT -> FLOAT`。该类用例主要用于触发底层 `isMixDataType` 分支，并验证混合精度输出是否符合预期。底层代码对这几类组合单独做了判断。fileciteturn5file1

#### （4）Mul：整数场景

例如 `INT32 × INT32 -> INT32`。该类测试主要用于验证非浮点路径的正确性，并避免测试只集中在浮点类型上。

#### （5）Mul：复杂类型或 DOUBLE 场景

用于尝试覆盖非典型数据类型路径，以及可能的 AiCpu 分支。根据底层实现，非 AiCore 支持类型会回退到 `MulAiCpu`；在 RegBase 平台上 `DOUBLE + DOUBLE` 也被单独识别。fileciteturn5file1

### 5.2 标量乘法用例

#### （6）Muls：FLOAT32 标量

验证 `aclnnMuls` 接口的主路径，重点检查张量与标量乘法的结果是否正确，输出 shape 是否与输入张量一致。该要求在头文件接口说明中有明确约束。fileciteturn5file2

#### （7）Muls：FLOAT16/BF16 标量

该类测试用于触发类型推导和 Cast 分支，特别是低精度输入与标量结合时的行为。

### 5.3 原地乘法用例

#### （8）InplaceMul：同 shape

验证 `aclnnInplaceMul` 将结果原地写回 `selfRef` 的行为是否正确。

#### （9）InplaceMul：广播合法场景

根据接口约束，广播后的 shape 必须与 `selfRef` 的 shape 相等，因此可设计如 `selfRef.shape=[2,3]`、`other.shape=[3]` 的原地广播乘法。该用例用于覆盖原地 shape 合法校验逻辑。fileciteturn5file2

#### （10）InplaceMuls：标量原地乘法

验证原地标量乘法路径，补足 `aclnnInplaceMuls` 的接口覆盖。

### 5.4 异常用例

#### （11）空指针测试

向 `aclnnMulGetWorkspaceSize`、`aclnnMulsGetWorkspaceSize`、`aclnnInplaceMulGetWorkspaceSize`、`aclnnInplaceMulsGetWorkspaceSize` 传入 `nullptr`，检查其是否返回参数错误，而不是崩溃退出。题目明确要求覆盖 `nullptr` 异常输入。fileciteturn5file0

#### （12）输出 shape 错误

设计合法输入 shape，但故意提供错误的 `out.shape`，验证接口是否能正确拒绝该调用。根据接口说明，`aclnnMul` 的 `out` 必须等于广播后的 shape。fileciteturn5file2

#### （13）非法广播

例如输入 shape 无法满足广播规则，检查 `GetWorkspaceSize` 阶段是否返回错误码。底层代码会在广播推断失败时直接返回空。fileciteturn5file1

#### （14）原地 shape 非法

设计 `selfRef` 和 `other` 虽然可广播，但广播后 shape 不等于 `selfRef.shape`，验证 `aclnnInplaceMul` 的专用 shape 校验分支。

## 6. 预期覆盖效果分析

本测试报告未虚构具体覆盖率数值，而是从路径层面分析预期提升方向。

1. **`op_api/aclnn_mul.cpp`**
   - 通过 4 个 API 的正常调用，覆盖各自的主流程入口。
   - 通过同 dtype、混合 dtype、整数 dtype、复杂 dtype 用例，覆盖类型推导、Cast 和数据类型检查分支。
   - 通过广播合法、广播非法、原地广播合法、原地广播非法、错误 `out.shape` 等场景，覆盖 shape 校验分支。
   - 通过 `nullptr` 输入覆盖参数空指针校验分支。

2. **`op_api/mul.cpp`**
   - 通过 `FLOAT32/FLOAT16/BF16/INT32/DOUBLE/COMPLEX` 等不同输入组合，尽量覆盖 `IsAiCoreSupport`、`isMixDataType`、`IsDoubleSupport`、`MulAiCore` 与 `MulAiCpu` 等分支。fileciteturn5file1

3. **`op_host/arch35/mul_tiling_arch35.cpp`**
   - 通过多种 dtype 组合和 shape 组合，间接推动不同 tiling 分支执行。由于该文件主要受输入 dtype 与张量形状驱动，因此增加 shape 与 dtype 的多样性是提升该文件覆盖率的关键。题目也将其列为重点统计对象。fileciteturn5file0

## 7. 测试执行方法

按照题目要求，测试执行流程如下：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example mul eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
find build -name "*.gcda" | grep mul
gcov -b <gcda文件路径>
```

每次修改测试用例后，都需要重新执行“编译 -> 安装 -> 运行”全流程，才能得到新的 `.gcda` 与 `gcov` 结果。fileciteturn5file0

## 8. 已知风险与注意事项

1. **不能只保留官方示例**
   官方 `test_aclnn_mul.cpp` 仅覆盖一组 `float32` 正常路径，并且只打印结果、不做结果验证，难以满足题目要求，也不利于提升覆盖率。fileciteturn5file0

2. **覆盖率不只看“跑通”**
   若测试只包含普通同 shape 乘法，即使程序成功运行，也只能覆盖极少数主路径，难以提升分支覆盖率。

3. **异常用例不能导致整体测试中断**
   对于预期失败的场景，测试程序应将“返回错误码”视为测试通过，而不是让进程崩溃。

4. **不同平台可能影响具体路径**
   例如 AiCore / AiCpu 路由与 RegBase 支持情况有关，因此个别 dtype 在不同环境中实际命中的底层路径可能不同。这不会改变测试设计思路，但会影响最终具体覆盖率分布。fileciteturn5file1

## 9. 结论

修改前：

![image-20260412192035897](C:\Users\27452\cann-competitions\03_university\CANN-ops-test-challenge-2026\preliminary\submissions\gzhu_miaomiaoshuilan-team\report\assets\Mul_fig1)

![image-20260412192047633](C:\Users\27452\cann-competitions\03_university\CANN-ops-test-challenge-2026\preliminary\submissions\gzhu_miaomiaoshuilan-team\report\assets\Mul_fig2)

修改后：

![image-20260412192102295](C:\Users\27452\cann-competitions\03_university\CANN-ops-test-challenge-2026\preliminary\submissions\gzhu_miaomiaoshuilan-team\report\assets\Mul_fig3)

![image-20260412192106825](C:\Users\27452\cann-competitions\03_university\CANN-ops-test-challenge-2026\preliminary\submissions\gzhu_miaomiaoshuilan-team\report\assets\Mul_fig4.png)

本次针对问题 A 的 Mul 算子测试设计，围绕“正确性验证 + 路径覆盖 + 异常处理”三个目标展开。测试设计不再停留于单一 `float32` 示例，而是系统覆盖了 4 个 API、广播与非广播场景、混合 dtype 与同 dtype 场景、原地与非原地场景，以及空指针和 shape 错误等异常输入。这样设计的目标是更有针对性地命中 `aclnn_mul.cpp`、`mul.cpp` 和 `mul_tiling_arch35.cpp` 中的关键逻辑分支，从而在保证结果正确性的同时，尽可能提高代码覆盖率，并满足题目对结果验证与测试设计说明的要求。fileciteturn5file0
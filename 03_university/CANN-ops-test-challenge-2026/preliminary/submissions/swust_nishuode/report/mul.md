# 题目 A：Mul 算子测试报告

## 1. 任务背景

本次任务要求基于 `math/mul/examples/test_aclnn_mul.cpp` 为 CANN ops-math 仓库中的 **Mul（逐元素乘法）算子**编写端到端测试用例，目标是在 CPU 模拟器环境下完成真实执行、结果校验，并尽可能提升代码覆盖率。题目明确指出，覆盖率统计的核心文件为：

- `op_api/aclnn_mul.cpp`
- `op_api/mul.cpp`
- `op_host/arch35/mul_tiling_arch35.cpp`

同时，测试不仅需要“跑通”，还必须包含**有效结果校验逻辑**，不能只打印输出。题目还强调应从 **dtype、shape/broadcast、API 变体、异常输入** 等维度扩展测试。  

## 2. 测试目标

本次测试工作的目标分为三层：

1. **功能正确性**：构造真实输入，执行 `aclnnMul` / `aclnnMuls` / `aclnnInplaceMul` / `aclnnInplaceMuls`，并对输出进行自动校验。
2. **覆盖率提升**：围绕题目计分文件，尽量触发更多 API 调度路径、dtype 分发路径、broadcast 路径和 tiling 路径。
3. **测试思路验证**：从“只做接口检查”升级到“接口检查 + 真实执行 + 回拷结果 + 自动比对”的执行型测试。

## 3. 环境与执行方式

本次测试采用题目要求的标准流程：

```bash
# 编译（启用覆盖率）
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov

# 安装算子包
./build_out/cann-ops-math-custom_linux-x86_64.run

# 运行 example（CPU 模拟器）
bash build.sh --run_example mul eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov

# 查看覆盖率
gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/aclnn_mul.cpp.gcda
gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/mul.cpp.gcda
gcov -b build/math/mul/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/mul_tiling_arch35.cpp.gcda
```

为避免覆盖率统计失真，每轮修改后都重新执行“编译 → 安装 → 运行 → gcov统计”的完整流程。

## 4. 测试设计思路

### 4.1 总体原则

本次没有推翻官方 example，而是在其基础上补充：

- 多组端到端执行样例
- 自动结果校验
- 多 API 变体
- mixed dtype 场景
- broadcast / 大 shape / 高 rank / 非连续 stride 场景
- 少量无效输入场景

核心思路是：

- **原有官方思路**更偏向“接口调用骨架”
- **本次扩展**更强调“真实执行 + 结果验证 + 覆盖更多路径”

### 4.2 覆盖维度

本次测试主要从以下维度扩展：

#### （1）API 变体

补充并执行了以下 4 类接口：

- `Mul`
- `Muls`
- `InplaceMul`
- `InplaceMuls`

这一步的目的主要是提升 `aclnn_mul.cpp` 中的 API 分发与参数检查覆盖。

#### （2）数值与基础 shape

在基础功能验证阶段，首先采用：

- 1D / 2D / 3D 的普通同 shape case
- 含正数、负数、0 的混合输入
- 单元素输入
- `float` 与 `int32` 基础场景

这部分主要用于保证执行闭环稳定，同时覆盖最常见的数据路径。

#### （3）Broadcast 场景

为了进一步提升 `mul.cpp` 与 `mul_tiling_arch35.cpp` 覆盖率，补充了多种 broadcast：

- `[2, 3] × [1, 3]`
- `[8, 1, 128] × [1, 16, 128]`
- `[4, 1, 32, 16] × [1, 8, 1, 16]`
- 更高 rank 的 rank5 / rank6 broadcast 组合

这类 case 的目的，是让调度与 tiling 层进入更多不同的维度处理分支。

#### （4）Mixed dtype 场景

针对题目中“不同 dtype 会触发不同调度路径和 tiling 策略”的要求，本次补充了混合类型 case：

- `FLOAT16 × FLOAT → FLOAT`
- `FLOAT × FLOAT16 → FLOAT`
- `BF16 × FLOAT → FLOAT`
- `FLOAT × BF16 → FLOAT`

同时，在 CPU 侧先对 FP16/BF16 输入做量化，再计算期望结果，并采用更宽松但合理的容差进行比对。

#### （5）更大 tensor 与高维 shape

为了拉动 host tiling 层的逻辑，还补充了：

- `128x128`
- `192x192`
- 更高 rank 的大规模 broadcast 组合

目标是尽量触发更多 tiling 策略分支，而不仅停留在小 shape 的基础路径上。

#### （6）非连续 stride / non-contiguous 场景

这是后期最重要的一步优化。

前几轮测试虽然已经覆盖了很多 broadcast，但 `broadcast_tiling_noncontiguous.h` 仍长期为 0%，说明已有测试几乎全是 contiguous tensor。为此，后续引入了：

- 自定义 `storageShape + strides`
- 构造逻辑 shape 与物理存储 shape 不一致的 Tensor
- 非连续 float case
- 非连续 mixed dtype case

这一类 case 的主要目的，是强行把覆盖拉入 **non-contiguous broadcast tiling 路径**。

#### （7）无效输入场景

为了提升 API 层参数检查与异常分支覆盖，补充了少量 invalid case，例如：

- 输出 shape 故意错误
- `Muls` 的 scalar 为空指针
- mixed dtype 组合下故意给出不匹配的输出 dtype
- 非法 stride case

这类测试不要求真正执行成功，而是要求在 `GetWorkspaceSize` 阶段返回失败，从而覆盖异常判断逻辑。

## 5. 代码实现要点

本次测试代码不是简单堆 case，而是做了较多测试基础设施建设，主要包括：

1. **统一的初始化与清理逻辑**：封装 `Init / Finalize / ReleaseTensorResource`
2. **统一的 Tensor 构造函数**：支持普通连续 Tensor、原始 bytes Tensor、带自定义 stride 的 Tensor
3. **CPU 侧期望值自动生成**：实现 `CpuBroadcastMul`，避免手写大量高维期望值
4. **mixed dtype 编码工具**：支持 `float → fp16/bf16 bits` 的编码与量化对齐
5. **自动结果比对**：
   - `int32` 使用精确比较
   - `float` 使用容差比较
   - `fp16/bf16 mixed` 使用更宽松但合理的误差阈值
6. **统一输出格式**：每个 case 输出 `[OK]` 或 `[FAILED]`，程序结束返回总状态

这些基础设施的作用是：

- 降低新增 case 的成本
- 让高维 / mixed / non-contiguous case 可持续扩展
- 保证测试结构清晰，不会因为 case 变多而难以维护

## 6. 覆盖率结果

本轮最终有效覆盖率结果如下：

| 文件 | Lines | Branches | Taken at least once | Calls |
|------|------:|---------:|--------------------:|------:|
| `op_api/aclnn_mul.cpp` | 75.80% | 27.89% | 14.81% | 30.39% |
| `op_api/mul.cpp` | 70.21% | 40.30% | 26.87% | 47.06% |
| `op_host/arch35/mul_tiling_arch35.cpp` | 52.08% | 24.40% | 12.80% | 19.52% |

从整体上看：

- `aclnn_mul.cpp` 已经进入较高覆盖区间，说明 API 分发、参数校验、mixed dtype 和 API 变体路径已被有效打到。
- `mul.cpp` 达到 70% 左右，说明设备路由与 dtype 支持判断路径已有较好覆盖。
- `mul_tiling_arch35.cpp` 从早期较低覆盖逐步提升到 52.08%，说明更大 shape、mixed dtype 和 non-contiguous 路径确实对 tiling 层有正向作用。

## 7. 覆盖率演进分析

### 7.1 初期阶段

最初版本主要集中在：

- `float / int32`
- 同 shape
- 少量基础 broadcast
- 结果校验

此时虽然能稳定执行并产出有效覆盖率，但提升主要集中在基础执行路径，对 `mul_tiling_arch35.cpp` 的帮助有限。

### 7.2 中期阶段

加入 `Muls / InplaceMul / InplaceMuls` 和 invalid case 后，`aclnn_mul.cpp` 提升最明显，说明 API 层分发和参数校验逻辑被更充分覆盖。

### 7.3 后期阶段

补充 mixed dtype、大 shape、高 rank broadcast 后：

- `mul.cpp` 开始继续上升
- `mul_tiling_arch35.cpp` 也出现了明显提升

说明 dtype 组合与 shape 复杂度，确实是 host tiling 层的主要触发因素。

### 7.4 non-contiguous 阶段

这是本轮最关键的新增方向。

在引入非连续 stride Tensor 之前，`broadcast_tiling_noncontiguous.h` 长期为 0%。加入 non-contiguous float 与 mixed case 之后，该文件覆盖率提升到 **11.71%**，说明测试首次进入了非连续 broadcast tiling 路径。虽然主计分文件还未出现跨越式上升，但这一步验证了测试方向是正确的。

## 8. 问题与排查过程

本次测试过程中遇到过多类问题，主要包括：

### （1）编译期问题

- `aclnnop/aclnn_mul.h` 找不到：通过确认 example 目标与头文件路径对应关系解决
- `aclCreateScalar` 传入 `const T*` 报错：改为先拷贝到局部变量，再传可写地址
- 重复定义 `int32Cases`：清理拼接代码并统一变量命名

### （2）运行期问题

- simulator 日志文件过多导致 `Too many open files`
- 某些激进 case 导致 `Segmentation fault`
- Add 题中 `inplace_add` 示例在模拟器下先崩，导致主 example 无法继续

这些问题说明：在算子测试比赛中，测试代码不仅要关注逻辑正确性，还要兼顾模拟器环境稳定性、日志资源限制和 example 运行顺序。

### （3）覆盖率统计问题

曾出现过：

- `stamp mismatch with notes file`
- `.gcda` / `.gcno` 不匹配

后来通过“清理旧覆盖率文件 → 全量重编译 → 重装 → 重跑 example → 重新 gcov”的方式解决，最终得到当前这组有效结果。

## 9. 结果评价

### 9.1 已取得的效果

本次测试代码已经实现：

- 从“仅有官方骨架”升级为“真实执行 + 自动校验 + 多维扩展”的完整测试
- 覆盖了 4 类 API 变体
- 覆盖了普通 dtype 与 mixed dtype
- 覆盖了普通 broadcast、高 rank broadcast、较大 tensor
- 首次进入 non-contiguous tiling 路径
- 在题目 3 个计分文件上都获得了有效提升

### 9.2 当前瓶颈

虽然覆盖率已明显优于初始阶段，但离 90%+ 仍有较大距离，尤其是：

- `mul_tiling_arch35.cpp` 仍只有 52.08%
- deeper branches 与 host/tiling 的大量内部逻辑尚未覆盖
- 单纯继续增加普通 example case，边际收益已经明显下降

因此，后续如果还要继续冲高覆盖率，重点不应只是“再多加几组普通执行样例”，而应考虑：

- 继续按 `.gcov` 未覆盖行做定向反推
- 更系统地补充 format / dtype / rank / stride 的特殊组合
- 如比赛规则允许，增加更接近 op_api / host 层的定向测试，而不仅依赖 example

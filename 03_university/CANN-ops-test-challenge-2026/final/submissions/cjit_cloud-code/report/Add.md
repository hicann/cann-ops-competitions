------
# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "云上码术"

team_members:

- "成员1：陈帅-长江工程职业技术学院"
- "成员2：沈均皓-长江工程职业技术学院"

operator_name: "Add"

operator_library: "cann-ops-math"

report_date: "2026-04-25"

------

# 算子测试报告

> 测试环境：远程真机 NPU，SoC 为 `ascend910_93`，仓库目录为 `/root/a`。构建方式为 CANN ops-math 真机 eager example 流程，覆盖率由 `--cov` 插桩生成，归档范围为 Add op_api 与 op_host arch35 的 `.gcda` / `.gcno` 文件。

------

## 一、算子理解

Add 是逐元素加法算子，当前接口对应的数学定义为：

```text
y = x1 + alpha * x2
```

其中 `x1` 和 `x2` 可以是 tensor，也可以在部分 API 中由 scalar 替代；`alpha` 是缩放系数。计算语义上需要先把 `x2` 乘以 `alpha`，再与 `x1` 相加。对 tensor-tensor 输入，两个 tensor 需要满足 broadcasting 规则；对 inplace API，写回 tensor 的逻辑 shape 需要能承接最终 broadcast 后的结果。

当前 Add 测试覆盖 6 类 public ACLNN API 变体：

| API | 输入形式 | 输出/写回形式 | 本次覆盖重点 |
| --- | --- | --- | --- |
| `aclnnAdd` | tensor + tensor + alpha | 独立 out tensor | 基础加法、broadcast、非连续 tensor、dtype promote、alpha 分支、异常参数 |
| `aclnnAdds` | tensor + scalar + alpha | 独立 out tensor | scalar 路径、FP16 scalar、INT32 scalar、bool/complex/double plan 路径 |
| `aclnnInplaceAdd` | tensor + tensor + alpha | 写回 self | inplace broadcast 写回、输出 shape 约束 |
| `aclnnInplaceAdds` | tensor + scalar + alpha | 写回 self | inplace scalar 写回 |
| `aclnnAddV3` | scalar + tensor + alpha | 独立 out tensor | V3 scalar-tensor 入口、alpha=1/非 1、FP16 输入输出转换、INT8 fallback、异常参数 |
| `aclnnInplaceAddV3` | scalar + tensor + alpha | 写回 other | V3 inplace 写回 |

从实现路径看，Add 不是简单地只走一个 kernel。测试中需要关注以下执行路径：

| 路径类型 | 触发条件 | 测试关注点 |
| --- | --- | --- |
| 直接 Add 路径 | 常见于 `alpha == 1` 的 tensor-tensor 或 scalar-tensor 场景 | 输出是否被正确写回，dtype promote 后输出 dtype 是否合理 |
| Axpy / AxpyV2 路径 | `alpha != 1` 且 dtype 支持缩放加法时 | `x1 + alpha * x2` 是否与 CPU oracle 一致 |
| Mul fallback / cast 路径 | double、complex、部分混合 dtype 或 scalar promote 场景 | 只规划 workspace 或执行结果的正确性，避免 unsupported dtype 误判 |
| 空 tensor first-stage | shape 中元素个数为 0 | workspace size、executor 生成和返回码是否正确 |
| tiling arch35 路径 | `ascend910_93` 在 Add CMake 配置中映射到 `arch35` | host tiling `.gcno/.gcda` 是否生成，branch 是否被实际触发 |
| 参数校验路径 | null 指针、非法 dtype、非法 shape、非法 alpha | 返回码是否稳定落在预期错误类型 |

dtype 方面，本次覆盖了 `ACL_FLOAT`、`ACL_FLOAT16`、`ACL_BF16`、`ACL_INT32`、`ACL_INT64`、`ACL_INT16` 与 `ACL_INT32` 混合、`ACL_INT8`、`ACL_UINT8`、`ACL_BOOL`、`ACL_DOUBLE`、`ACL_COMPLEX64`。其中 FP16/BF16 需要特别处理存储位模式和真实数值的转换；整数和 bool 需要精确比较；complex64 需要分别比较实部和虚部。

------

## 二、测试策略与用例设计

本次在 Add example 文件 `math/add/examples/test_aclnn_add.cpp` 上扩展测试用例，没有修改 `tests/` 目录，也没有引入自定义头文件或内部测试头文件。测试文件中使用的算子头文件为公开 ACLNN 接口：

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"
```

最终 example 共运行 69 个用例，运行输出为：

```text
Summary: 69 passed, 0 failed
run test_aclnn_add, execute samples success
run test_aclnn_add, execute samples success
```

### 2.1 用例分布

用例按 API 入口统计如下：

| API 入口 | 正向执行用例 | 异常/边界校验用例 | 合计 |
| --- | ---: | ---: | ---: |
| `aclnnAdd` | 21 | 12 | 33 |
| `aclnnAdds` | 10 | 6 | 16 |
| `aclnnInplaceAdd` | 1 | 1 | 2 |
| `aclnnInplaceAdds` | 1 | 0 | 1 |
| `aclnnAddV3` | 7 | 9 | 16 |
| `aclnnInplaceAddV3` | 1 | 0 | 1 |
| 合计 | 41 | 28 | 69 |

用例按测试目的分层如下：

| 分类 | 代表用例 | 设计目的 |
| --- | --- | --- |
| 基础功能 | FP32 same-shape、FP32 broadcast、INT32 alpha=2 | 验证最基本的逐元素计算和 `alpha` 缩放语义 |
| alpha 分支 | alpha=0、alpha=1、alpha=-1、alpha=-0.5、alpha=2、小 alpha | 区分直接 Add、Axpy/AxpyV2、cancellation 等路径 |
| broadcasting | `[2,3] + [1,3]`、inplace broadcast | 覆盖 shape 推导和 tiling 中的 broadcast 处理 |
| 非连续 tensor | storage shape `[2,4]`、logical shape `[2,2]`、stride `{4,2}` | 覆盖 stride/view 输入，不只验证 contiguous tensor |
| format 分支 | `ACL_FORMAT_NCHW` plan-only | 覆盖非 ND format 的 workspace 规划路径 |
| 空 tensor | shape `{0}` 的 Add/Adds/AddV3 first-stage | 验证空 tensor workspace size 和 executor 返回 |
| dtype promote | FP16+FP32、FP32+FP16、INT16+INT32、scalar FP16 to FP32 | 验证输入 dtype 不一致时的 promote 与输出 dtype 约束 |
| 低精度 dtype | FP16、BF16 | 验证量化输入下的结果误差和输出解码 |
| 整数/BOOL | INT32、INT64、INT8、UINT8、BOOL | 验证 exact oracle，避免用浮点容差掩盖错误 |
| complex/double | COMPLEX64、DOUBLE | 覆盖 promote/fallback/plan 路径 |
| inplace | `aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnInplaceAddV3` | 验证写回 tensor 的结果和 shape 约束 |
| 异常参数 | null self/other/alpha/out、shape mismatch、rank>8、unsupported dtype | 覆盖 public ACLNN 参数校验分支 |

### 2.2 Oracle 与校验方式

所有正向执行用例都构造 CPU 侧参考值。参考实现不直接复用 NPU 输出，而是按输入 shape、broadcast 规则和 `alpha` 重新计算：

```text
tensor-tensor: expected[i] = broadcast(self)[i] + alpha * broadcast(other)[i]
tensor-scalar: expected[i] = self[i] + alpha * scalar
scalar-tensor: expected[i] = scalar + alpha * other[i]
```

不同 dtype 的 oracle 处理方式如下：

| dtype | Oracle 处理 | 比较方式 |
| --- | --- | --- |
| FP32 | 输入提升到 CPU double 后计算 | `atol=1e-6, rtol=1e-6`，消减场景使用更宽绝对误差 |
| FP16 | 先用 `aclFloat16ToFloat` 解码为 float，再提升到 double 计算 | `atol=2e-3, rtol=2e-3` |
| BF16 | 按 BF16 位模式还原 float，再提升到 double 计算 | `atol=2e-2, rtol=2e-2` |
| 整数 | CPU double 只用于统一生成期望整数值，不用容差掩盖错误 | 精确比较 |
| BOOL | 以 0/1 语义计算 | 精确比较 |
| COMPLEX64 | 实部和虚部分别计算 | `atol=1e-5, rtol=1e-5` |

FP16 和 BF16 不能把底层 `uint16_t` 位模式直接当数值参与 CPU 计算，否则 oracle 会把编码位模式误当整数，导致精度结论失真。因此测试代码中对低精度类型单独实现了解码和读取逻辑。

### 2.3 异常路径设计

异常用例不 launch kernel，只验证 `GetWorkspaceSize` 阶段的返回码，原因是这些路径的目标是参数校验而不是计算结果。覆盖的异常包括：

| 异常类型 | 覆盖内容 |
| --- | --- |
| null 指针 | `self`、`other`、`alpha`、`out` 为空 |
| shape 非法 | broadcast 失败、输出 shape 不匹配、rank 大于 8 |
| dtype 非法 | `uint64` unsupported、float 输出到 bool、complex 输出到 int32、V3 double other unsupported |
| alpha 非法 | bool alpha 非 integral、float API 使用 complex alpha、V3 complex alpha |
| inplace 约束 | inplace 写回 tensor 无法承接 broadcast 输出 |

这些异常用例对提升分支覆盖率有效，但 public example 无法直接构造框架内部 tiling context 的 null 指针或所有 SoC 架构分支，因此仍有部分 host 侧分支无法触达。

------

## 三、覆盖率分析

### 3.1 执行流程

本次按项目真机构建流程执行。Add 默认配置未把 `ascend910_93` 映射到 `arch35`，如果直接清理 build 编译，会查不到 `add_tiling*.gcno`，host tiling 覆盖率会为 0。因此编译前只对 `math/add/CMakeLists.txt` 补齐 `ascend910_93` 到 `arch35` 的映射：

```bash
cd /root/a
sed -i 's|set(SUPPORT_COMPUTE_UNIT "ascend950" "mc62cm12a")|set(SUPPORT_COMPUTE_UNIT "ascend310p" "ascend910_93" "ascend910b" "ascend950" "mc62cm12a")|;
        s|set(SUPPORT_TILING_DIR "arch35" "arch35")$|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' \
    math/add/CMakeLists.txt
```

之后清理 build，并通过项目构建脚本重新编译、安装、运行：

```bash
rm -rf build/*
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
find build -name "add_tiling*.gcno"
./build_out/cann-ops-math-custom_linux-aarch64.run
bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

`find build -name "add_tiling*.gcno"` 实测能查到 arch35 tiling 插桩文件：

```text
build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcno
```

覆盖率使用 `gcov -b <gcda文件路径>` 统计。统计前没有再运行 op_host/op_api UT，避免不同构建时间戳混入同一批 `.gcda/.gcno` 导致 `libgcov profiling error: different timestamp`。

### 3.2 评分文件覆盖率结果

| 评分文件 | 行覆盖率 | 分支覆盖率 | Taken at least once |
| --- | ---: | ---: | ---: |
| `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcda` | 72.61% of 303 | 48.97% of 1546 | 27.62% of 1546 |
| `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcda` | 96.10% of 77 | 61.97% of 426 | 35.45% of 426 |
| `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcda` | 55.93% of 59 | 22.73% of 264 | 14.77% of 264 |
| `build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcda` | 89.25% of 93 | 57.29% of 192 | 37.50% of 192 |

按本次纳入统计的 4 个 Add 覆盖率文件加权计算：

```text
综合行覆盖率 = (220 + 74 + 33 + 83) / (303 + 77 + 59 + 93)
             = 410 / 532
             = 77.07%

综合分支覆盖率 = (757 + 264 + 60 + 110) / (1546 + 426 + 264 + 192)
               = 1191 / 2428
               = 49.05%
```

当前行覆盖率和分支覆盖率没有达到 90%。本报告如实记录最终实测数据，不把无法触达的内部路径或其它目录 UT 结果混入 Add example 覆盖率。

### 3.3 覆盖率文件清单

归档目录中保留以下 8 个 Add 覆盖率文件：

```text
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcno
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcno
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcda
build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcno
build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcda
build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcno
```

### 3.4 未覆盖路径分析

`aclnn_add.cpp` 的覆盖率主要受以下路径限制影响：

- 当前真机 SoC 为 `ascend910_93`，无法覆盖其它 SoC/架构分支。
- RegBase 相关分支在当前设备上不成立，因此相关判断只能覆盖 false 侧。
- 部分 dtype promote 组合和 complex fallback 分支能完成 plan，但 public example 下不一定会继续触发所有内部分支。
- public ACLNN 接口可以构造 null/shape/dtype 错误，但不能完整构造框架内部上下文错误。

`aclnn_add_v3.cpp` 行覆盖率较高，原因是 V3 API 文件规模较小，且本次覆盖了正向 scalar-tensor、empty tensor、null 参数、输出 dtype 错误、alpha complex、double unsupported 等路径。剩余未覆盖主要仍是设备架构、少数 dtype promote 组合和内部错误处理。

`add.cpp` 覆盖率偏低，主要原因是该文件包含更底层的 l0op 调度与 inplace 内部函数。Add example 只能通过 public ACLNN API 进入上层 executor 路径，不能直接调用内部 `l0op::AddInplace` 等函数；如果强行引入内部头文件，会改变本次只通过公开 ACLNN example 覆盖的统计口径，因此没有采用。

`add_tiling_arch35.cpp` 行覆盖率达到 89.25%，但分支覆盖率仍低于 60%。未覆盖分支主要集中在 invalid tiling context、平台信息异常、少数 dtype/format/shape 组合和防御性错误路径。这些路径通常需要 op_host UT 中手动构造 tiling context 或 mock 平台信息，public example 很难稳定触达。

------

## 四、精度分析

### 4.1 误差阈值

本次精度分析以 CPU oracle 为基准。阈值设置依据是 dtype 本身的有效精度和输入量化误差：

| dtype/场景 | 阈值 | 原因 |
| --- | --- | --- |
| FP32 常规 | `atol=1e-6, rtol=1e-6` | 单次加法和缩放误差应稳定在 1e-6 量级内 |
| FP32 cancellation | `atol=1e-3, rtol=1e-6` | 大数相减后相对误差会被放大，因此放宽绝对误差 |
| FP16 | `atol=2e-3, rtol=2e-3` | FP16 尾数 10 位，量化误差约 1e-3 量级 |
| BF16 | `atol=2e-2, rtol=2e-2` | BF16 尾数 7 位，精度低于 FP16，但动态范围接近 FP32 |
| INT/BOOL | 精确比较 | 结果应完全一致，不使用容差 |
| COMPLEX64 | `atol=1e-5, rtol=1e-5` | 实部和虚部分别按 FP32 精度比较 |

### 4.2 稳定通过的精度路径

以下执行路径在真机上与 CPU oracle 一致：

| 场景 | 输入特征 | 实测结论 |
| --- | --- | --- |
| FP32 broadcast alpha=-2.5 | `self=[2,3]`，`other=[1,3]` | broadcast 后结果与 double oracle 一致 |
| FP32 cancellation alpha=-1 | 输入包含 `10000.25`、`9999.75`、`0.001` 等消减值 | 在 cancellation 容差内一致 |
| FP32 非连续输入 | logical shape `[2,2]`，storage shape `[2,4]`，stride `{4,2}` | 只读取逻辑位置，结果正确 |
| FP16 alpha=-0.5 | FP16 输入，FP16 输出 | 解码后在 FP16 容差内一致 |
| INT32 alpha=2 | tensor-tensor INT32 | AxpyV2 路径精确匹配 |
| Adds FP32 alpha=0 | scalar 被 alpha 清零 | 输出等于 self |
| Adds FP32 cancellation alpha=-1 | scalar 与大数消减 | 在 cancellation 容差内一致 |
| Adds FP16 scalar promotes-to-float | FP16 self，float scalar/alpha | 输出 FP16 解码后在容差内一致 |
| Adds INT32 alpha=2 | tensor + scalar | 精确匹配 |
| Inplace Add/Adds/AddV3 | 写回 self 或 other | 写回后读取结果符合 oracle |
| AddV3 FP32 alpha=-0.5 | scalar + tensor | 与 `scalar + alpha * tensor` 一致 |
| AddV3 small alpha | `self=0.125`，`alpha=0.125`，tensor 含 1024/0.03125 | 小 alpha 缩放后结果正确 |
| AddV3 FP16 tensor to FP32 | scalar FP16，other FP16，out FP32 | FP16 输入量化后，FP32 输出与 oracle 一致 |

这些用例说明 `alpha != 1` 的缩放加法路径整体较稳定，尤其是 FP32、FP16 和 INT32 的 Axpy/AxpyV2 相关路径能得到正确输出。

### 4.3 FP32 消减场景

Add 的一个典型精度风险是 `x1 + (-1) * x2`。当 `x1` 与 `x2` 数值接近时，结果的有效位数会明显减少。本次用例包含：

```text
self  = [10000.25, -10000.5, 0.001, -0.001, 3.141592, -2.718281]
other = [ 9999.75, -10000.25, -0.001, 0.001, -3.141592, 2.718281]
alpha = -1
```

该用例覆盖大数消减、小数消减和正负号混合。实际结果在 `atol=1e-3` 范围内通过。这里放宽绝对误差是合理的，因为消减会把输入中原本相同的高位抵消掉，剩余有效位更少；但如果误差超过 1e-3，则会影响使用 Add 表达差分、残差和梯度更新的场景。

### 4.4 FP16/BF16 场景

FP16 用例覆盖两类：

- `alpha=1` 的直接 Add 路径，用 observation 记录风险；
- `alpha=-0.5` 或 scalar promote 场景，用严格 oracle 判定并通过。

FP16 的关键不是只看最终数值，而是先确认输入是否已经被量化。例如 `0.001f` 写入 FP16 后不再是数学意义上的 0.001，CPU oracle 必须以解码后的 FP16 值为输入基准。测试中使用 `aclFloat16ToFloat` 读取输入语义，避免把 `aclFloat16` 的存储位模式当整数或 float 直接参与计算。

BF16 的动态范围接近 FP32，但尾数只有 7 位，量化误差大于 FP16。本次 BF16 precision execution 中观察到 direct Add 输出风险，报告单独列为 observation，没有把它误判为 BF16 正常量化误差。

### 4.5 整数、bool 和 complex 场景

整数 Add 的期望语义更接近精确算术，不应使用浮点容差掩盖错误。本次对 INT32 的 `alpha=2` 路径做严格比较并通过；对 INT64、INT8、UINT8 等 `alpha=1` 直接 Add 路径记录到输出全 0 observation。

BOOL 场景按 0/1 处理，`alpha=true` 时语义为 `self + other`。由于 BOOL 输出再转回 host 时仍应表现为 0/1，因此用例按精确 0/1 比较。实际 direct Add 路径观察到输出仍为 0，属于正确性风险，而不是数值容差问题。

COMPLEX64 场景中，期望计算为：

```text
(a + bi) + alpha * (c + di)
```

测试代码分别比较实部和虚部。真机上 `aclnnAdd complex64 alpha=1` 观察到实际 `(0,0)`，期望首元素为 `(3,1)`，也归入 direct Add 路径风险。

### 4.6 真机 precision observation

本次运行中，多组 case 在 launch 和同步成功后输出仍保持初始化值 0。测试代码将这些结果打印为 `precision observation`，用于保留复现证据；如果改成严格失败，example 将无法完整跑完并生成同一批覆盖率数据。报告中把它们作为待复核的精度/正确性风险，不扩大为未经定位的根因结论。

| 场景 | 期望首个元素 | 实际首个元素 |
| --- | ---: | ---: |
| `aclnnAdd float32 alpha=1` | 1 | 0 |
| `aclnnAdd fp16 + float mixed dtype` | 1.5 | 0 |
| `aclnnAdd float + fp16 mixed dtype` | 1.5 | 0 |
| `aclnnAdd fp16 same dtype` | 1.5 | 0 |
| `aclnnAdd bf16 precision execution` | 0.625 | 0 |
| `aclnnAdd int64 alpha=1` | 3 | 0 |
| `aclnnAdd uint8 alpha=1` | 9 | 0 |
| `aclnnAdd int8 alpha=1` | 9 | 0 |
| `aclnnAdd bool alpha=true` | 1 | 0 |
| `aclnnAdd complex64 alpha=1` | `(3,1)` | `(0,0)` |
| `aclnnAdd int16 + int32 alpha=1` | 3 | 0 |
| `aclnnAdds fp16 scalar keep-fp16` | 1.5 | 0 |
| `aclnnAdds bool scalar special cast` | 1 | 0 |
| `aclnnAddV3 float scalar alpha=1` | 11 | 0 |
| `aclnnAddV3 int8 fallback path` | -2 | 0 |

这些 observation 的共同特征是：多数集中在 `alpha == 1` 的 direct Add 路径，或者与 direct Add kernel/tiling 有关；而 `alpha != 1` 的 FP32/FP16/INT32 路径能够通过严格 oracle。由此推断，风险更可能集中在 direct Add 的执行或输出写回链路，而不是通用的 CPU oracle、host-device 拷贝或 stream 同步逻辑。这个判断是基于当前真机现象的推断，后续仍需结合 runtime 日志、kernel launch 记录和内部路径定位确认。

------

## 五、反思与改进

本次测试在 Add example 入口内尽量扩展了 API、dtype、shape、alpha 和异常参数覆盖，但仍存在以下局限：

| 局限 | 原因 | 后续改进方向 |
| --- | --- | --- |
| 综合覆盖率未达到 90% | public example 无法触达所有内部分支、其它架构分支和 l0op 直调路径 | 后续可补 op_api/op_host 单元测试，直接构造内部 context 和 l0op 调用 |
| 分支覆盖率明显低于行覆盖率 | 代码中存在大量 dtype、format、SoC、异常上下文判断，很多只能覆盖一侧 | 针对每个条件分支设计最小输入，尤其是 tiling 内部错误路径 |
| `add.cpp` 覆盖率偏低 | public ACLNN API 不直接暴露 `l0op::AddInplace` 等内部函数 | 增加内部 UT 可进一步覆盖；本次未采用内部头文件直调 |
| RegBase 分支未覆盖 | 当前 SoC 为 `ascend910_93`，相关判断不成立 | 需要对应设备或 mock 平台信息 |
| precision observation 未定位到根因 | example 只能看到 API 返回、同步和输出值，缺少内部 kernel 状态 | 后续结合 runtime 日志、算子注册信息、tiling key、kernel launch 细节排查 |

方法论上的收获：

- 覆盖率必须在干净 build 后重新编译、安装、运行，再用同一批 `.gcda/.gcno` 统计；混用旧 build 会触发 timestamp mismatch，导致覆盖率不可用。
- FP16/BF16 的 oracle 必须先解码输入位模式，再计算期望值；否则测试报告中的“精度误差”可能是 oracle 自身错误。
- 整数、bool、complex 的语义应分别处理，不能统一用浮点容差降低判定强度。
- 对真机复现的异常数值，应同时记录输入、期望、实际、API 路径和 alpha 条件，避免只写“结果错误”而缺少复核依据。
- Add 在 `ascend910_93` 真机上需要补齐 CMakeLists 中的 arch35 映射，否则 host tiling `.gcno` 不生成，op_host 覆盖率会被误统计为 0。

如果继续优化，优先级应为：

1. 继续寻找 public ACLNN 能触发的 dtype/alpha/format 组合，优先补 `add.cpp` 和 tiling 分支。
2. 通过 op_host/op_api UT 补充内部 context、null tiling、unsupported platform、`l0op::AddInplace` 等 public example 无法触达的路径。
3. 对 direct Add 输出全 0 observation 做最小复现，分别隔离 kernel 未执行、tiling key 错误、输出地址/shape 错误、dtype cast 错误四类可能。
4. 将 observation case 拆成严格失败版和覆盖率版两套运行方式：严格失败版用于精度缺陷复核，覆盖率版用于保证 Add example 流程生成完整 `.gcda`。

本次归档的 `build/` 目录只包含 Add op_api 和 op_host tiling 的覆盖率数据，不包含完整构建产物。

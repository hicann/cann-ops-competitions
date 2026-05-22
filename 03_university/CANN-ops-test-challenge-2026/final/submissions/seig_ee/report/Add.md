# ===== 元信息（请如实填写，此区块将由组委会脚本自动解析，请保持字段名不变）=====

team_name: "ee"
team_members:

- "林世伟-队长"
- "胡文康-队员"
- "赖永健-队员"

operator_name: "Add"
operator_library: "cann-ops-math"
report_date: "2026-04-25"

# Add 算子测试报告

> 测试环境：Ascend 真实 NPU 环境，CANN 9.0.0 beta 2，SOC 使用 `ascend910_93` / arch35 映射。提交前已对 `math/add/CMakeLists.txt` 执行前置修复，使 `ascend910_93` 的 host tiling 编译到 `arch35`，避免 `add_tiling_arch35.cpp` 覆盖率为 0。

## 技术方案

### 技术架构

本作品采用「端到端示例测试 + UT 分支补齐 + 覆盖率度量 + 报告分析」四层架构：

1. **端到端测试层**：基于 `math/add/examples/test_aclnn_add.cpp` 扩展 Add 系列接口测试，覆盖 `aclnnAdd`、`aclnnAdds`、`aclnnAddV3`、`aclnnInplaceAdd*` 等调用链路。  
2. **算子 UT 增强层**：在 `op_api` / `op_host` 增加针对性用例，补齐 dtype 组合、异常参数、tiling parse、null context 等关键分支。  
3. **覆盖率采集层**：使用 `gcov -b -f` 对评分文件进行行 / 分支 / 函数覆盖统计，形成可复现的覆盖率闭环。  
4. **结果分析层**：在报告中输出精度验证策略、覆盖率构成、风险分支说明和提交产物清单。

### 核心技术

- **技术 1：多接口统一测试驱动**：将 Add 的 Tensor-Tensor、Tensor-Scalar、Scalar-Tensor、Inplace、V3 统一纳入测试矩阵，保证 API 语义覆盖完整。  
- **技术 2：分支导向用例设计**：针对 promote / cast、Axpy / Mul+Add、broadcast、empty tensor、complex / bool、SOC 路径定向构造输入，提升分支与函数覆盖。  
- **技术 3：覆盖率与精度双闭环**：一方面以 `gcov` 量化覆盖改进，另一方面通过 host 侧 oracle + 容差比较验证数值正确性与稳定性。

### CANN 特性应用

本作品充分利用 CANN / ACLNN 的算子执行与调度特性：

- 使用 ACLNN 标准两阶段接口（`GetWorkspaceSize` + `Execute`）验证算子真实执行路径。  
- 覆盖 CANN 在不同 dtype / shape 下的内核选择行为（Add、Axpy、Mul+Add 等）。  
- 利用 CANN 的平台差异能力（SOC / arch 映射）验证 host 侧 tiling 路径（含 arch35）。  
- 针对 CANN 运行时的参数校验、类型提升、非连续张量处理进行异常与边界测试。

## 一、算子理解

Add 算子执行逐元素加法，主要数学形式为：

```text
out = self + alpha * other
```

其中 `self` 与 `other` 支持相同 shape、broadcast shape、空 tensor、连续 tensor 与非连续 view。API 层包含 Tensor-Tensor、Tensor-Scalar、Scalar-Tensor 以及 inplace 变体：`aclnnAdd`、`aclnnAdds`、`aclnnAddV3`、`aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnInplaceAddV3`。不同输入 dtype 与 `alpha` 值会触发不同执行路径，例如直接 `Add`、`Axpy`、`Mul + Add`、混合 dtype cast、bool scalar 特殊 cast，以及 AiCore/AiCpu 调度。

精度方面，整数和布尔类型理论上应精确匹配；浮点类型的误差主要来自输入量化、乘加舍入以及混合 dtype 的 cast。`float16`、`bfloat16` 输入在进入计算前已经存在表示误差，`alpha != 1` 时还会额外引入一次乘法误差；`complex64` 需要分别校验实部和虚部。

## 二、测试策略与用例设计

本次在官方 `math/add/examples/test_aclnn_add.cpp` 基础上扩展端到端测试。测试文件内置多个 case，运行时逐项输出 `[RUN]`、`[PASS]`、`[FAIL]`，最终以进程返回码体现测试是否通过。每个正向用例都在 host 侧计算期望值，并从 device 拷回实际输出进行逐元素比对，不仅打印结果。

正向覆盖场景包括：

- 同 dtype Tensor + Tensor：`float32`、`int32`、`int64`、`int16`、`int8`、`uint8`、`bool`、`complex64`。
- 混合 dtype：`float16 + float32 -> float32`、`float32 + bfloat16 -> float32`。
- 标量接口：`aclnnAdds` 覆盖浮点 scalar、bool scalar 特殊 cast、空 tensor。
- V3 接口：scalar + Tensor，覆盖 `alpha == 1`、`alpha != 1`、`int8`、空 tensor、inplace V3。
- 形状场景：普通连续 tensor、broadcast、非连续 view、空 tensor。
- inplace 场景：Tensor-Tensor inplace 与 Tensor-Scalar inplace。

异常覆盖场景包括：

- `self`、`other`、`alpha`、`out` 或 executor 相关参数为空。
- 输出 shape 与 broadcast 后 shape 不一致。
- rank 超过 8 维。
- 不支持 dtype，例如 `ACL_UINT32`。
- Phase 2 执行接口空指针调用。

为提升评分文件覆盖率，还补充了 op_api 和 op_host 层 UT：`test_aclnn_add_v3.cpp` 覆盖 V3 的 promote、Mul+Add、空 tensor 与异常参数；`test_add_tiling.cpp` 覆盖 arch35 tiling 的 dtype 组合、非法 dtype、`nullptr` tiling context 与 `TilingPrepareForAdd`。

## 三、覆盖率分析

覆盖率通过 `gcov -b` 收集，评分相关文件结果如下：

| 文件 | 行覆盖率 | 分支执行率 | 说明 |
| --- | ---: | ---: | --- |
| `op_api/aclnn_add.cpp` | 87.68% (299/341) | 64.23% (1020/1588) | 覆盖 Add/Adds/Inplace、多 dtype、broadcast、空 tensor、非连续 view、SOC dtype support、complex scalar promote、bool alpha 异常分支与异常参数 |
| `op_api/aclnn_add_v3.cpp` | 92.05% (81/88) | 59.24% (250/422) | 覆盖 V3、Inplace V3、promote、alpha 分支、Mul+Add、complex alpha 异常与异常参数 |
| `op_api/add.cpp` | 95.24% (60/63) | 88.10% (296/336) | 覆盖 L0 Add 调度、AiCore/AiCpu 支持判断与 dtype 分发 |
| `op_host/arch35/add_tiling_arch35.cpp` | 100.00% (104/104) | 67.32% (206/306) | 覆盖 dtype 检查、tiling key、workspace、空 context 与 tiling parse |

综合行覆盖率按行数加权：

```text
(299 + 81 + 60 + 104) / (341 + 88 + 63 + 104) = 544 / 596 = 91.28%
```

综合分支执行率按 `Branches executed` 加权：

```text
(1020 + 250 + 296 + 206) / (1588 + 422 + 336 + 306) = 1772 / 2652 = 66.82%
```

函数覆盖率按 `gcov -f` 统计：`aclnn_add.cpp` 为 `27/27`，`aclnn_add_v3.cpp` 为 `10/10`，`add.cpp` 为 `7/7`，`add_tiling_arch35.cpp` 为 `12/12`。

未完全覆盖的分支主要集中在框架宏展开、日志/DFX、部分内部失败路径、较重的 RegBase/AxpyV2 后端路径以及难以稳定构造的资源异常。考虑到评分前置门槛要求完整编译、安装和运行流程通过，最终选择保留稳定用例集合，不再引入可能导致 OPP 加载或后端选择不稳定的强制分支。

## 四、精度分析

测试采用 host 侧 oracle 验证。整数、布尔和无缩放的精确类型使用逐元素精确比较；浮点和复数类型使用容差比较：

- `float32`、混合 `float16/bfloat16 + float32`：绝对误差容差 `1e-3`。
- `complex64`：分别比较实部和虚部，绝对误差容差 `1e-3`。
- 整数、布尔：逐元素精确匹配。

`alpha == 1` 时，计算通常可走直接 Add 或更轻量路径，误差主要来自输入 dtype 本身。`alpha != 1` 时，计算可能走 `Axpy`、`AxpyV2` 或 `Mul + Add`，一次乘法加一次加法会带来额外舍入。测试中特意覆盖 `adds_float_scalar_axpy`、`add_v3_float_axpy`、`inplace_add_fp32` 等用例，确保这些路径的结果在合理误差范围内。

混合 dtype 场景中，`float16` 与 `bfloat16` 的输入精度低于 `float32`。为避免测试本身受不可控字面量误差影响，用例使用可精确表示或误差容易分析的 bit pattern，例如 `1.0`、`-2.0`、`0.5`。期望值按最终输出语义计算，并允许 `1e-3` 误差。

`bool + bool` 的标量路径需要特别关注：如果按普通整数加法，`true + true` 可能得到数值 `2`；但布尔输出语义应为真值。测试用例 `adds_bool_scalar_special_cast` 期望输出均为 `1`，用于覆盖并验证 bool scalar 的特殊 cast 分支。

`complex64` 使用两个 `float32` 表示实部和虚部。Add 对复数的语义是实部与虚部分别相加，因此测试按 `(real, imag)` 分量分别比较，而不是将结构体按字节比较，避免因浮点舍入或结构体布局造成误判。

## 五、构建、运行与提交说明

### 环境要求

- **CANN 版本**：`9.0.0 beta 2`（或与评测镜像一致；安装路径示例：`/usr/local/Ascend/cann-9.0.0-beta.2/...`）。  
- **操作系统**：Linux (aarch64)（与仓库根目录 `README.md` 一致）。  
- **硬件要求**：Ascend NPU（建议 Ascend 910B / `ascend910_93` 环境）。  
- **依赖库**：ACL / ACLNN、**GoogleTest (gtest)**、CMake、gcc / g++（如 11.4.0）、gcov / lcov、Python 3。

### 安装步骤

```bash
# 1) 设置 CANN 环境（按实际安装路径）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 2) 进入工程
cd /root/ops-math
# 3)（Add）修复 SOC → arch35 映射，使 ascend910_93 的 host tiling 编译到 arch35
sed -i 's|set(SUPPORT_COMPUTE_UNIT "ascend950" "mc62cm12a")|set(SUPPORT_COMPUTE_UNIT "ascend310p" "ascend910_93" "ascend910b" "ascend950" "mc62cm12a")|; s|set(SUPPORT_TILING_DIR "arch35" "arch35")$|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' math/add/CMakeLists.txt
# 4) 编译（覆盖率插桩）
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
# 5)（可选）确认 arch35 tiling 已生成 gcno
find build -name "add_tiling*.gcno" | head
# 6) 安装算子包
./build_out/cann-ops-math-custom_linux-aarch64.run
# 7) 运行 Add 示例测试
bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
# 8)（可选）对评分文件执行 gcov
gcov -b -f build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcda
gcov -b -f build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcda
gcov -b -f build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcda
gcov -b -f build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcda
```

### 使用方法

端到端 Add 测试与上节 `build.sh --run_example add eager cust ...` 一致。评测侧以组委会统一环境为准。

### 性能与规模指标

本项目为测试与覆盖率优化作品，核心指标为测试质量而非吞吐性能。当前关键结果如下：

- **综合行覆盖率**：91.28%  
- **综合分支执行率（Branches executed）**：66.82%  
- **评分文件函数覆盖率**：56 / 56（100%）  
- **`add_tiling_arch35.cpp` 行覆盖率**：100%  
- **结果输出**：各用例输出 `[RUN]` / `[PASS]` / `[FAIL]`，进程返回码反映整体是否通过。

提交包仅保留评分相关覆盖率文件：

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

## 六、其他信息

### 创新点

- 将「端到端示例测试」与「UT 分支补齐」联动，兼顾真实执行链路与细粒度分支覆盖。  
- 针对 Add 算子设计 dtype / shape / 异常 / 平台路径的系统化覆盖矩阵。  
- 通过隔离式过滤运行策略提升高风险分支覆盖，同时控制整体稳定性。

### 应用价值

- 可作为 ACLNN 算子测试增强模板，复用到其他数学算子（如 Cumsum、Mul 等）。  
- 提升算子在评测环境与真实部署环境中的可验证性与可维护性。  
- 为算子质量评估提供可量化证据（行 / 分支 / 函数覆盖 + 精度验证）。

### 参考资料

- CANN 官方文档与 ACL / ACLNN 开发指南。  
- `cann-ops-math` 官方代码与 Add 算子示例。  
- gcc / gcov 覆盖率工具文档。  
- GoogleTest 文档。

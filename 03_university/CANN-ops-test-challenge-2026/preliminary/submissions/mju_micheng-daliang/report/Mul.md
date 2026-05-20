# 题目 A Mul 算子测试报告

## 1. 交付内容

本次提交的 Mul 算子交付文件为：

- `test_aclnn_mul.cpp`

本地保留的构建目录中还存在 Mul 相关的运行与覆盖率中间产物，例如：

- `build/test_aclnn_mul`
- `build/test_aclnn_mul.gcda`
- `build/test_aclnn_mul.gcno`
- `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/aclnn_mul.cpp.gcda`
- `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/mul.cpp.gcda`
- `build/CMakeFiles/gen_op_host_aclnnExc.dir/math/mul/op_host/mul_def.cpp.gcda`

这些文件说明本地曾对 Mul 样例进行了编译与执行，并生成了覆盖率统计所需的中间结果。

## 2. 测试设计

`test_aclnn_mul.cpp` 采用端到端方式直接调用 `aclnnMulGetWorkspaceSize + aclnnMul`，并在 CPU 侧构造期望结果进行校验。代码中实现了如下测试基础设施：

- ACL 运行时初始化与清理
- 张量创建与设备内存分配
- 两段式 API 调用封装
- 结果回传与数值比较
- 广播 shape 计算与兼容性判断

从现存代码可确认，该测试程序共组织了 35 组测试，覆盖以下几类场景。

### 2.1 参数校验类

- 空指针检查：
  - `self == nullptr`
  - `other == nullptr`
  - `out == nullptr`
- 不支持的数据类型检查
- 不可广播 shape 检查
- 空 Tensor 场景
- `GetWorkspaceSize` 查询场景

### 2.2 FP32 正常路径

- 全正数乘法
- 正负混合乘法
- 全负数乘法
- 含零乘法
- 单位元乘法
- 零乘法

### 2.3 Shape 与广播场景

- 1D 张量
- 2D 非方阵
- 3D 张量
- 4D 张量
- 较大张量（`4x4x4`）
- 广播 `{1,3} x {3,1}`
- 广播 `{1} x {5}`
- 广播 `{2,1,3} x {1,4,3}`
- 广播 `{2,1} x {1,3}`

### 2.4 整数与布尔类型

- `INT32`
- `INT8`
- `UINT8`
- `INT16`
- `INT64`
- `BOOL`

### 2.5 浮点与混合类型

- `FP16`
- `BF16`
- `DOUBLE`
- `FP16 x FP32`
- `BF16 x FP32`
- `FP32 x FP16`
- `INT32 x FP32` 类型提升
- `BOOL x FP32` 类型提升

从代码结构看，测试重点集中在：

- `op_api` 层参数校验与类型分发
- 广播路径
- 多 dtype 路径
- `GetWorkspaceSize` 与执行阶段的完整调用链

## 3. 环境与构建说明

预选赛环境按本地 x86_64 配置流程完成，目标与官方统一 Docker/模拟器环境保持一致。结合预选赛参考材料与保留文件，可确认本题使用的是：

- SoC：`ascend950`
- 运行模式：CPU 模拟器（`--simulator`）
- 项目路径：`/home/workspace/ops-math`
- 覆盖率工具：`gcov -b`

与预选赛参考流程对应的基本命令为：

1. 编译：
   `bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov`
2. 安装：
   `./build_out/cann-ops-math-custom_linux-x86_64.run`
3. 运行：
   `bash build.sh --run_example mul eager cust --vendor_name=custom --simulator --soc=ascend950 --cov`
4. 查看覆盖率：
   `find build -name "*.gcda" | grep mul`
   `gcov -b <gcda文件路径>`

## 4. 本地保留痕迹与可确认结果

虽然本题原始独立测试报告未单独保留下来，但依据当前保留的代码与 `build/` 目录，可以确认以下事实：

1. `test_aclnn_mul.cpp` 已被编译，存在：
   - `build/test_aclnn_mul`
   - `build/test_aclnn_mul.gcno`
2. `test_aclnn_mul` 至少执行过一次，存在：
   - `build/test_aclnn_mul.gcda`
3. Mul 的 API 层覆盖率数据已生成，存在：
   - `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/aclnn_mul.cpp.gcda`
   - `build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/mul/op_api/mul.cpp.gcda`
4. Mul 的 host 层文件存在覆盖率插桩或构建痕迹，存在：
   - `build/math/mul/CMakeFiles/ophost_math_infer_obj.dir/op_host/mul_infershape.cpp.gcno`
   - `build/math/mul/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/mul_tiling_arch35.cpp.gcno`
   - `build/CMakeFiles/gen_op_host_aclnnExc.dir/math/mul/op_host/mul_def.cpp.gcda`

因此，可以确认 Mul 这组测试不是单纯的源码草稿，而是经过本地编译与运行并实际生成过覆盖率中间产物。

## 5. 结论

本次 Mul 提交的核心特点如下：

- 使用单一 `test_aclnn_mul.cpp` 对 `aclnnMul` 进行了端到端测试扩展
- 覆盖了参数校验、广播、常见 shape、多种整数/浮点 dtype、混合类型与类型提升场景
- 本地已保留可验证的 `gcda/gcno` 与可执行文件痕迹，可证明测试程序实际参与过编译与运行

受限于题目一当时“测试报告鼓励提交”的要求，原始独立报告文件未保留；本报告基于现存源码与构建痕迹对测试内容进行了如实整理，不包含无法从当前材料中确认的覆盖率百分比或未保留的运行摘要。

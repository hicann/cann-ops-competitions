# 题目 B Add 算子测试报告

## 1. 交付内容

本次交付基于 `math/add/examples/test_aclnn_add.cpp` 完成 Add 算子的端到端测试扩展，并按题面要求将 `test_aclnn_inplace_add.cpp` 替换为占位脚本，避免官方独立样例在模拟器下崩溃。

交付覆盖内容包括：

- 执行型测试：
  - `aclnnAdd`：`float32` 同 shape
  - `aclnnAdd`：`float32` 广播
  - `aclnnAdd`：`float16 + float32 -> float32` 混合精度
  - `aclnnAddV3`：标量 + tensor
- 异常输入测试：
  - `aclnnAdd`
  - `aclnnAdds`
  - `aclnnInplaceAdd`
  - `aclnnInplaceAdds`
  - `aclnnAddV3`
  - `aclnnInplaceAddV3`
- 结果验证：
  - 所有执行型用例在 CPU 侧独立计算期望值，并进行数值比较
  - 所有异常型用例验证返回错误码
- 输出格式：
  - 每个用例输出 `[PASS]` / `[FAIL]`
  - 末尾输出汇总，失败数决定返回值

## 2. 代码改动

- 主测试文件：
  - `math/add/examples/test_aclnn_add.cpp`
- 占位脚本：
  - `math/add/examples/test_aclnn_inplace_add.cpp`
- 构建脚本补丁：
  - `build.sh`

`test_aclnn_add.cpp` 中新增了：

- 张量 / 标量 / workspace 资源封装
- `float16` 编解码辅助函数
- 广播结果与 `alpha` 缩放的 CPU 侧期望值计算
- 浮点容差比较函数
- Add / Adds / Inplace / AddV3 系列 API 的统一测试辅助逻辑

## 3. 环境与构建说明

本次沿用与题目一相同的 Docker/CANN 环境，在容器内完成构建和运行：

- Docker 镜像：`yeren666/cann-ops-test:v1.0`
- SoC：`ascend950`
- 模式：CPU 模拟器

为保证构建稳定，还做了两类环境修复：

- 为 `build.sh` 的 example 编译补充覆盖率参数和更完整的链接路径
- 为第三方依赖提供本地离线包，避免 protobuf 等依赖下载失败

## 4. 验证结果

已完成并通过的验证：

1. `bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov`
   - 成功生成 `build_out/cann-ops-math-custom_linux-x86_64.run`
2. 安装 custom op 包成功
3. 手工编译 `test_aclnn_add.cpp` 成功
4. 手工执行 `test_aclnn_add.cpp` 成功，汇总结果为 `0 failed`
5. `bash build.sh --run_example add eager cust --vendor_name=custom --simulator --soc=ascend950 --cov`
   - 成功执行
   - `test_aclnn_add.cpp` 全部 `[PASS]`
   - `test_aclnn_inplace_add.cpp` 占位脚本正常通过

运行结果摘要：

- `[PASS] add_float32_alpha1`
- `[PASS] add_float32_broadcast_alpha1`
- `[PASS] add_float16_float32_mix`
- `[PASS] add_v3_float32_alpha1`
- `[PASS] add_nullptr_check`
- `[PASS] add_invalid_shape_check`
- `[PASS] adds_nullptr_check`
- `[PASS] inplace_add_invalid_shape_check`
- `[PASS] inplace_adds_nullptr_check`
- `[PASS] add_v3_invalid_shape_check`
- `[PASS] inplace_add_v3_nullptr_check`
- `=== Summary: 0 failed ===`

同时，`build/` 中已生成 Add 相关覆盖率产物，例如：

- `math/add/op_api/aclnn_add.cpp.gcda`
- `math/add/op_api/aclnn_add_v3.cpp.gcda`
- `math/add/op_api/add.cpp.gcda`

## 5. 提交建议

建议提交目录包含以下内容：

- `test_aclnn_add.cpp`
- `test_aclnn_inplace_add.cpp`
- `build/`
- `测试报告.md`
- `build.sh`

其中前 3 项为核心交付，后两项用于复现和说明。

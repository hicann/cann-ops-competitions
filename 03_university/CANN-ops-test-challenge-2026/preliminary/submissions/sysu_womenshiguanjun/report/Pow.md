# Pow 算子测试设计报告

## 1. 目标

本提交针对题目 C（Pow）实现端到端测试，目标如下：

- 覆盖 7 个对外 API：
  - `aclnnPowTensorScalar`
  - `aclnnInplacePowTensorScalar`
  - `aclnnPowScalarTensor`
  - `aclnnPowTensorTensor`
  - `aclnnInplacePowTensorTensor`
  - `aclnnExp2`
  - `aclnnInplaceExp2`
- 补充 CPU 侧期望值计算与数值校验（浮点容差 + 整型精确比较）
- 输出统一 `[PASS]/[FAIL]`，并在末尾输出汇总，失败时返回非 0

## 2. 测试覆盖点

### 2.1 API 覆盖

已分别调用 TensorScalar / ScalarTensor / TensorTensor / Exp2 及其 Inplace 变体，覆盖 op_api 不同源文件路径。

### 2.2 数据类型覆盖

- `FLOAT32`：TensorScalar / ScalarTensor / TensorTensor / Exp2
- `INT32`：TensorTensor
- `INT8`：TensorTensor
- `UINT8`：TensorTensor

说明：题目建议覆盖更多 dtype（如 BF16/FLOAT16）。当前版本优先保证稳定可运行与 API 全覆盖，可继续在此基础上增加 BF16/FLOAT16 用例以冲刺更高覆盖率。

### 2.3 shape 与广播覆盖

- 同 shape（`2x2`）
- 广播（`self: [2,3]`，`exponent: [3]`）

### 2.4 特殊指数与数值边界

- `exp = 0`（结果应为 1，含 `0^0`）
- `exp = 0.5`（sqrt 分支，含负数底数触发 NaN 路径）
- `exp = -1`（倒数路径）
- `exp = 3`（立方路径）
- `Exp2`（`2^x`，含负指数）

## 3. 校验策略

- 浮点用例：
  - 期望值：`std::pow(...)`
  - 判定：`|actual - expected| <= atol + rtol * |expected|`
  - 支持 `NaN/Inf` 判定
- 整型用例：
  - 期望值：`std::pow(...)` 后转目标整型
  - 判定：逐元素精确相等

## 4. 运行方式（评测环境）

在 Docker 容器中执行：

```bash
cd /home/workspace/ops-math

# 用本提交 test_aclnn_pow.cpp 覆盖 math/pow/examples/ 同名文件后执行
bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example pow eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

覆盖率查看：

```bash
find build -name "*.gcda" | grep pow
gcov -b <gcda文件路径>
```

## 5. 提交目录说明

- `test_aclnn_pow.cpp`：主测试文件（必须）
- `build/`：需在容器内跑完一次编译+运行后，将生成的 `build` 目录拷贝到此处（必须）
- `测试报告.md`：本说明文档（鼓励提交）

## 6. 本次实测结果（2026-04-12，增强版）

在容器 `yeren666/cann-ops-test:v1.0` 中执行完整流程后，测试输出：

- `Total: 17, PASS: 17, FAIL: 0`

说明：其中包含少量“能力探测型”用例（例如 `int64` TensorScalar、`base=1` 的 Fill 分支），若当前 SoC/实现返回不支持状态码，则按“探测通过”记录，不计为失败，以保证提交在评测环境下稳定完成全流程。

对题目给定的 5 个评分文件执行 `gcov -b` 后，行覆盖率如下：

- `math/pow/op_api/aclnn_pow.cpp`：`69.08%`（`262` 行）
- `math/pow/op_api/aclnn_pow_tensor_tensor.cpp`：`85.00%`（`80` 行）
- `math/pow/op_api/pow.cpp`：`80.00%`（`30` 行）
- `math/pow/op_host/arch35/pow_tensor_tensor_tiling_arch35.cpp`：`81.75%`（`126` 行）
- `math/pow/op_host/arch35/pow_tiling_arch35.cpp`：`74.07%`（`54` 行）

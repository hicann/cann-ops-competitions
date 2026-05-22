# Mul 算子端到端测试设计说明

**队伍名称：** ee  
**对应文件：** `test_aclnn_mul.cpp`  
**算子：** Mul（逐元素乘法，$y = x_1 \times x_2$，支持广播）

---

## 1. 测试目标

在官方示例 `math/mul/examples/test_aclnn_mul.cpp` 基础上扩展，提高对以下源码的**行/分支覆盖率**（评测范围包括 `op_api/aclnn_mul.cpp`、`op_api/mul.cpp`、`op_host/arch35/mul_tiling_arch35.cpp` 等）：

- 覆盖 **op_api** 层：参数校验、类型提升、Mul / Muls / InplaceMul / InplaceMuls 等不同入口。
- 覆盖 **op_host** 层：不同 dtype 与 shape 组合下的 tiling 与调度路径。
- 所有用例均在 **CPU 端计算期望值** 并与设备输出比对，避免“只跑不验”。

---

## 2. 结果校验方法

- **浮点类型：** 采用容差  
  $|actual - expected| \leq atol + rtol \times |expected|$  
  - FLOAT32：`1e-5`  
  - FLOAT16：`1e-3`  
  - BF16：`1e-2`  
  - DOUBLE：`1e-12`  
  - **NaN / Inf：** 使用 `std::isnan` / `std::isinf` 与符号位一致判断。
- **整数类型：** 在 CPU 端按元素相乘后按**输出 dtype 位宽截断**，与 NPU 结果逐元素精确相等。
- **广播：** 在 CPU 端实现与 NumPy 一致的**左侧补 1** 对齐及输出坐标到两输入的索引映射，保证广播场景期望值正确。

---

## 3. 用例覆盖维度

| 维度 | 说明 |
|------|------|
| **API 变体** | `aclnnMul`、`aclnnMuls`、`aclnnInplaceMul`、`aclnnInplaceMuls` 均包含独立用例。 |
| **数据类型** | FLOAT32；FLOAT16×FLOAT32→FLOAT32；BF16×FLOAT32→FLOAT32；INT32；INT8 广播；DOUBLE（含 Inf/NaN）；InplaceMuls 使用 INT32 标量。 |
| **Shape** | 同形状；`[2,3]×[3]` 广播；`[2,1]×[1,4]` 广播；长度 4096 的一维大张量（利于触发 tiling）。 |
| **边界与异常** | Inf、NaN、0 参与运算；`aclnnMulGetWorkspaceSize(nullptr, …)` 期望返回非成功状态。 |

---

## 4. 程序输出约定

- 每个用例一行：`[PASS] <名称>` 或 `[FAIL] <名称>`。
- 结尾输出：`Summary: PASS=x FAIL=y`。
- **任一 FAIL 时 `main` 返回非 0**，便于脚本与 CI 判断。

---

## 5. 编译、安装与覆盖率（需在 Linux + CANN 环境执行）

以下命令与赛题文档一致，请在已配置 **Ascend/CANN** 的机器上于 `ops-math` 工程根目录执行：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example mul eager cust \
  --vendor_name=custom --simulator --soc=ascend950 --cov
```

生成覆盖率中间文件后，可将工程下的 **`build/`** 目录（或文档指定的含 `*.gcda` 等产物的目录）**复制到本提交包内的 `ee/build/`**，以满足慕测对 `build/` 目录的要求。

---

## 6. 打包提交（示例）

在包含 `ee` 文件夹的上一级目录执行：

```bash
zip -r ee.zip ee/
```

压缩包内结构应为：

```
ee/
├── test_aclnn_mul.cpp
├── build/                 # 含编译与覆盖率中间产物（在 Linux+CANN 下生成后复制）
└── 测试报告.md
```

---

## 7. 说明

当前仓库若在 **macOS** 上无法执行 `build.sh`（脚本依赖 Linux 环境特征），**覆盖率目录需在目标评测或本地 Linux 开发机完成构建后补全**。

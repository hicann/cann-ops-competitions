# Pow 算子端到端测试设计说明（队名：ee）

## 1. 测试目标

对 CANN `ops-math` 中 **Pow** 算子进行集成级验证，在官方 example 风格上扩展用例，覆盖多 API、多 dtype、广播与边界值，并以 CPU 端 `std::pow` / `2^x` 作为期望值做数值比对。

## 2. 覆盖的 API（7 个）

| API | 说明 |
|-----|------|
| `aclnnPowTensorScalar` | `out = self ^ scalar` |
| `aclnnInplacePowTensorScalar` | 原地 `self ^= scalar` |
| `aclnnPowScalarTensor` | `out = scalar ^ exp_tensor` |
| `aclnnPowTensorTensor` | `out = self ^ exp`（可广播） |
| `aclnnInplacePowTensorTensor` | 原地逐元素幂 |
| `aclnnExp2` | `out = 2 ^ self` |
| `aclnnInplaceExp2` | 原地 `2 ^ self` |

## 3. 校验策略

- 浮点：`abs(actual - expected) <= atol + rtol * abs(expected)`，其中 `atol=1e-4`，`rtol=1e-3`。
- 整型：`int32` 结果与 `llround(std::pow(...))` 逐元素相等。

## 4. 用例维度

- **指数**：0、1、2、0.5、-1、3、4.1 等，覆盖常见优化分支。
- **Shape**：同形；广播示例 `self [4,1]` × `exp [1,2]` → `[4,2]`。
- **dtype**：FLOAT32、FLOAT16、INT32（TensorScalar）。
- **异常**：`aclnnPowTensorScalarGetWorkspaceSize` 在 `self == nullptr` 时期望非成功状态码。

## 5. 编译、运行与覆盖率

在 `ops-math` 工程根目录执行（以赛题文档为准）：

```bash
bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example pow eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

将生成的与 `pow` 相关的 `.gcda` 等覆盖率中间文件复制或保留至本包 `build/` 目录后再打包提交。

## 6. 程序输出与退出码

- 每例输出 `[PASS]` / `[FAIL]`，末尾汇总通过/失败数。
- 全部通过返回 `0`；存在失败返回 `2`；ACL 初始化失败返回 `1`。

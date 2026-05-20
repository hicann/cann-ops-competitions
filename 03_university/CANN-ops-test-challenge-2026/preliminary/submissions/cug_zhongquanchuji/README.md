# CANN ops-math 算子测试用例说明

## 团队信息

- 团队名称：重拳出击
- 所属单位：中国地质大学（武汉）
- 团队成员：
  - 郭宸羽，算子测试代码设计、实现、测试与优化，文档编辑，队员统筹
  - 郭琛，算子测试代码设计、实现、测试与优化，文档编辑
  - 马瑞晗，算子测试代码设计、实现、测试与优化，文档编辑
- 联系人：郭宸羽
- 联系邮箱：287180474@qq.com

## 环境要求

- Docker 镜像：大赛提供的 `cann-ops-test:v1.0`，由离线包 `cann-ops-test-v1.0.tar.gz` 导入
- CANN 版本：使用 `cann-ops-test:v1.0` 镜像内置 CANN 工具链与运行环境
- 操作系统：镜像内置 Linux x86_64 环境
- 目标 SoC：`ascend950`
- 编译器：镜像内置 `g++` / `cmake` / `make`
- 测试框架：基于 CANN ACL / ACLNN 端到端 example 测试流程
- 其他依赖：镜像内置 CPU Simulator、ops-math 源码及覆盖率统计工具 `gcov`

离线镜像导入和启动方式如下：

```bash
docker load -i cann-ops-test-v1.0.tar.gz
docker run -it --name ops-test cann-ops-test:v1.0
```

容器启动后环境变量通过 `.bashrc` 自动加载，ops-math 源码位于 `/home/workspace/ops-math/`。

## 文件说明

- `code/`：测试代码源文件
  - `code/Add/test_aclnn_add.cpp`：Add 算子测试代码，覆盖 `aclnnAdd`、`aclnnAdds`、原地 Add、Add V3 等 API 路径
  - `code/Mul/test_aclnn_mul.cpp`：Mul 算子测试代码，覆盖 Tensor-Tensor、Tensor-Scalar、原地 Mul 及异常输入路径
  - `code/Pow/test_aclnn_pow.cpp`：Pow 算子测试代码，覆盖 TensorScalar、ScalarTensor、TensorTensor 及原地 Pow 等 API 路径
- `report/`：测试报告
  - `report/Add.md`：Add 算子测试报告
  - `report/Mul.md`：Mul 算子测试报告
  - `report/Pow.md`：Pow 算子测试报告

## 编译与运行

以下命令在大赛 Docker 容器内执行。假设提交目录位于 `/home/workspace/submission/`，其中包含本 README、`code/` 和 `report/`。

### Add 算子

```bash
cd /home/workspace/ops-math
cp /home/workspace/submission/code/Add/test_aclnn_add.cpp math/add/examples/test_aclnn_add.cpp

bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example add eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

### Mul 算子

```bash
cd /home/workspace/ops-math
cp /home/workspace/submission/code/Mul/test_aclnn_mul.cpp math/mul/examples/test_aclnn_mul.cpp

bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example mul eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

### Pow 算子

```bash
cd /home/workspace/ops-math
cp /home/workspace/submission/code/Pow/test_aclnn_pow.cpp math/pow/examples/test_aclnn_pow.cpp

bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example pow eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

### 覆盖率查看

每次运行完成后，可在 `/home/workspace/ops-math/build/` 下查看对应算子的覆盖率数据：

```bash
cd /home/workspace/ops-math
find build -name "*.gcda" | grep -E "add|mul|pow"
gcov -b <gcda文件路径>
```

`gcov` 输出中的 `Lines executed: XX.XX% of YY` 为行覆盖率统计结果。

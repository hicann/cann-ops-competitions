# 预选赛：算子测试用例设计

## 赛题概述

本次预选赛要求参赛者为 CANN ops-math 仓库中的算子编写端到端测试用例，尽可能覆盖算子的各种执行路径，以代码覆盖率作为主要评价指标。

预选赛共设 3 道题目：

| 题目 | 算子 | 难度 | 说明 |
|:----:|:----:|:----:|------|
| 题目 1 | Mul（逐元素乘法） | 基础 | $y = x_1 \times x_2$ |
| 题目 2 | Add（逐元素加法） | 基础 | $y = x_1 + \alpha \times x_2$，含 alpha 参数和 V3 API |
| 题目 3 | Pow（逐元素幂运算） | 进阶 | $y_i = x_{1,i}^{x_{2,i}}$，含 TensorScalar / ScalarTensor / TensorTensor 三类 API |

各题目的详细要求见对应的题目描述文档。

## 参考资料

在开始之前，建议阅读以下参考文档：

- [Mul 算子架构分析](https://gitcode.com/org/AI4SE/discussions/4)：了解 CANN 算子的分层结构（op_api / op_host / op_kernel）、支持的数据类型组合以及各层中的条件分支。
- [Mul 算子测试用例分析](https://gitcode.com/org/AI4SE/discussions/5)：了解端到端测试用例的代码结构、两段式 API 调用方法（`GetWorkspaceSize` + `Execute`）以及结果验证思路。
- [环境配置指南](https://gitcode.com/org/AI4SE/discussions/2)：如需自行搭建环境，请参考此文档。

## 实验环境

本次实验提供预配置的 Docker 镜像，其中已包含 CANN 工具链、CPU 模拟器及 ops-math 源码等全部依赖。

**注意：由于 CANN 模拟器限制，仅支持 x86 架构。Mac 等 ARM 架构处理器无法使用该 Docker 环境。**

**方式一：从 DockerHub 拉取**

```bash
docker pull yeren666/cann-ops-test:v1.0
docker run -it --name ops-test yeren666/cann-ops-test:v1.0
```

**方式二：从离线包导入**（适用于无法访问 DockerHub 的环境）

下载 `cann-ops-test-v1.0.tar.gz`（[下载链接](https://drive.google.com/file/d/15N7so_FDVtF2WwNDC7hk7x4l7bgMvMP9/view?usp=drive_link)），然后执行：

```bash
docker load -i cann-ops-test-v1.0.tar.gz
docker run -it --name ops-test cann-ops-test:v1.0
```

容器启动后，环境变量会通过 `.bashrc` 自动加载。ops-math 源码位于 `/home/workspace/ops-math/`。

> 如果不小心退出了容器，可以用 `docker start -i ops-test` 重新进入，之前的修改不会丢失。

如果参赛者希望自行配置环境，也可以参照[环境配置指南](https://gitcode.com/org/AI4SE/discussions/2)搭建。无论使用哪种方式，最终评测将在统一的 Docker 环境中执行，请确保提交的测试用例在该环境中可以正常编译运行。

## 基本操作流程

以 Mul 算子为例（Add 和 Pow 只需将命令中的 `mul` 替换为 `add` 或 `pow`）：

### 1. 编译算子

```bash
cd /home/workspace/ops-math
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
```

`--cov` 启用覆盖率统计。编译成功后在 `build_out/` 下生成算子安装包。

### 2. 安装算子包

```bash
./build_out/cann-ops-math-custom_linux-x86_64.run
```

### 3. 运行测试

```bash
bash build.sh --run_example mul eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

运行成功后会在 `build/` 目录下生成覆盖率数据文件（`.gcda`）。

### 4. 查看覆盖率

```bash
find build -name "*.gcda" | grep mul
gcov -b <gcda文件路径>
```

`gcov` 输出的 `Lines executed: XX.XX% of YY` 即为行覆盖率。每次修改测试用例后，需重新执行步骤 1-4。

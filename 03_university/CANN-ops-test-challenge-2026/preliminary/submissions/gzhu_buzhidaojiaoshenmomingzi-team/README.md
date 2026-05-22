# 预选赛提交

## 团队信息

- 团队名称：不知道叫什么名字队
- 所属单位：广州大学
- 团队成员：
  - 陈慧美，队长（算子测试编码及测试）
  - 叶翔宇，成员（算子测试编码及测试）
- 联系人：陈慧美
- 联系邮箱：2685796099@qq.com

## 环境要求

- CANN 版本：8.0.RC1
- 运行环境：Docker 容器（x86 架构 CPU 模拟器）
- 操作系统：Ubuntu 20.04 x86_64
- 编译器：g++ 9.4.0
- 测试框架：GoogleTest 1.12.1
- Docker 镜像：`yeren666/cann-ops-test:v1.0`

> 注意：由于 CANN 模拟器限制，仅支持 x86 架构。Mac 等 ARM 架构处理器无法使用该 Docker 环境。

启动 Docker 环境：

```bash
docker pull yeren666/cann-ops-test:v1.0
docker run -it --name ops-test yeren666/cann-ops-test:v1.0
```

容器内 ops-math 源码位于 `/home/workspace/ops-math/`。

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Mul/`：Mul 算子测试代码（`test_aclnn_mul.cpp`）
  - `code/Add/`：Add 算子测试代码（`test_aclnn_add.cpp`）
  - `code/Pow/`：Pow 算子测试代码（`test_aclnn_pow.cpp`）
- `report/`：测试报告
  - `report/report.md`：测试报告主文档

## 编译与运行

### 1. 编译算子

```bash
cd /home/workspace/ops-math
bash build.sh --pkg --soc=ascend950 --ops=<op> --vendor_name=custom --cov
```

### 2. 安装算子包

```bash
./build_out/cann-ops-math-custom_linux-x86_64.run
```

### 3. 运行测试（CPU 模拟器）

```bash
bash build.sh --run_example <op> eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

### 4. 查看覆盖率

```bash
find build -name "*.gcda" | grep <op>
gcov -b <gcda文件路径>
```

其中 `<op>` 替换为 `mul`、`add` 或 `pow`。每次修改测试用例后需重新执行步骤 1–4。

## 评分标准

以代码覆盖率作为主要评价指标，统计 `op_api` 层与 `op_host` 层指定文件的综合行覆盖率。

# AHNU_doushidui

## 团队信息

- 团队名称：doushidui
- 所属单位：安徽师范大学
- 团队成员：
  - 商星宇，负责 Mul 算子测试用例设计与代码实现
  - 宋强强，负责 Add 算子测试用例设计与代码实现
  - 檀焰，负责 Pow 算子测试用例设计与报告整理
- 联系人：商星宇
- 联系邮箱：1277569508@qq.com

## 环境要求

本次提交基于大赛提供的统一 Docker 环境进行测试与验证。

- Docker 镜像：`yeren666/cann-ops-test:v1.0`
- CANN 环境：Docker 镜像内置 CANN 工具链
- ops-math 源码路径：`/home/workspace/ops-math/`
- 目标平台：`ascend950`
- 运行方式：CPU 模拟器
- 系统架构：x86_64
- 操作系统：以大赛 Docker 镜像环境为准
- 编译器：以大赛 Docker 镜像环境为准
- 覆盖率工具：gcov
- 其他依赖：无

说明：由于 CANN 模拟器限制，本测试环境仅支持 x86 架构。最终评测将在大赛统一 Docker 环境中执行。

## 文件说明

本团队提交目录结构如下：

```text
AHNU_doushidui/
├── README.md
├── code/
│   ├── Mul/
│   │   └── test_aclnn_mul.cpp
│   ├── Add/
│   │   └── test_aclnn_add.cpp
│   └── Pow/
│       └── test_aclnn_pow.cpp
└── report/
    └── report.md
```

各目录说明如下：

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Mul/`：Mul 逐元素乘法算子测试代码
  - `code/Add/`：Add 逐元素加法算子测试代码，包含 alpha 参数和 V3 API 相关测试
  - `code/Pow/`：Pow 逐元素幂运算算子测试代码，包含 TensorScalar、ScalarTensor、TensorTensor 等 API 测试

- `report/`：测试报告
  - `report/report.md`：总测试报告，说明测试设计思路、覆盖率统计方式与问题发现情况

## 测试内容概述

本次预选赛共针对以下 3 个算子设计端到端测试用例：

| 题目 | 算子 | 测试目标 |
|---|---|---|
| 题目 1 | Mul | 测试逐元素乘法 `y = x1 * x2` |
| 题目 2 | Add | 测试逐元素加法 `y = x1 + alpha * x2`，覆盖 alpha 参数与 V3 API |
| 题目 3 | Pow | 测试逐元素幂运算，覆盖 TensorScalar、ScalarTensor、TensorTensor 三类 API |

测试设计重点包括：

1. 基础功能测试
2. 不同数据类型测试
3. 不同 shape 测试
4. 广播机制测试
5. inplace 接口测试
6. scalar 与 tensor 混合输入测试
7. 边界值测试
8. 异常路径测试
9. 覆盖率统计与分析

## Docker 环境启动

如需从 DockerHub 拉取镜像，可执行：

```bash
docker pull yeren666/cann-ops-test:v1.0
docker run -it --name ops-test yeren666/cann-ops-test:v1.0
```

如使用离线包导入镜像，可执行：

```bash
docker load -i cann-ops-test-v1.0.tar.gz
docker run -it --name ops-test cann-ops-test:v1.0
```

如果退出容器，可使用以下命令重新进入：

```bash
docker start -i ops-test
```

## 编译与运行

以下命令均在 Docker 容器中执行。

进入 ops-math 源码目录：

```bash
cd /home/workspace/ops-math
```

### 1. Mul 算子

#### 编译 Mul 算子并开启覆盖率

```bash
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
```

#### 安装算子包

```bash
./build_out/cann-ops-math-custom_linux-x86_64.run
```

#### 运行 Mul 测试用例

```bash
bash build.sh --run_example mul eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

#### 查看 Mul 覆盖率

```bash
find build -name "*.gcda" | grep mul
gcov -b <gcda文件路径>
```

---

### 2. Add 算子

#### 编译 Add 算子并开启覆盖率

```bash
bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
```

#### 安装算子包

```bash
./build_out/cann-ops-math-custom_linux-x86_64.run
```

#### 运行 Add 测试用例

```bash
bash build.sh --run_example add eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

#### 查看 Add 覆盖率

```bash
find build -name "*.gcda" | grep add
gcov -b <gcda文件路径>
```

---

### 3. Pow 算子

#### 编译 Pow 算子并开启覆盖率

```bash
bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov
```

#### 安装算子包

```bash
./build_out/cann-ops-math-custom_linux-x86_64.run
```

#### 运行 Pow 测试用例

```bash
bash build.sh --run_example pow eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

#### 查看 Pow 覆盖率

```bash
find build -name "*.gcda" | grep pow
gcov -b <gcda文件路径>
```

## 覆盖率统计说明

本次测试使用 `--cov` 参数开启覆盖率统计。运行测试后，覆盖率数据文件会生成在 `build/` 目录下，文件后缀通常为 `.gcda`。

查看覆盖率时，可使用：

```bash
gcov -b <gcda文件路径>
```

其中输出中的：

```text
Lines executed: XX.XX% of YY
```

即为对应文件的行覆盖率结果。

每次修改测试用例后，建议重新执行以下流程：

1. 重新编译算子
2. 重新安装算子包
3. 重新运行测试
4. 重新统计覆盖率

## 注意事项

1. 本提交仅包含测试源码、README 和测试报告，不包含编译产物。
2. 未提交 `build/` 目录、`.o`、`.gcda`、`.gcno` 等文件。
3. 测试用例需在大赛统一 Docker 环境中编译运行。
4. 覆盖率数据由评测系统重新生成。
5. 所有代码文件均按照算子名称分别存放，便于评测和维护。

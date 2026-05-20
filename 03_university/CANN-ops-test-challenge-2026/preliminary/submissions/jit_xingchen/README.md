# 星辰队提交说明

## 团队信息

- 团队名称：星辰
- 所属单位：金陵科技学院
- 团队成员：
  - 张健伟，测试代码编写与报告整理
  - 陈羽洁，测试用例设计与结果分析
- 联系人：张健伟
- 联系邮箱：zhangjianwei0615@163.com

## 环境要求

- CANN 版本：CANN Toolkit 9.0.0
- 操作系统：WSL Ubuntu 22.04 x86_64
- 编译器：GCC/G++ 11.4.0
- 构建工具：CMake 3.22.1
- 测试框架：自研 C++ 测试程序，基于 ACL Runtime、ACLNN 两段式接口和 NPU Stream
- 运行方式：Ascend950 CPU Simulator，Eager Mode，开启 `--cov` 覆盖率插桩
- 其他依赖：`build-essential`、`make`、`cmake`、`git`、`unzip`、`python3`

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Mul/test_aclnn_mul.cpp`：Mul 算子测试代码
  - `code/Add/test_aclnn_add.cpp`：Add 算子测试代码
  - `code/Pow/test_aclnn_pow.cpp`：Pow 算子测试代码
- `report/`：测试报告
  - `report/Mul.md`：Mul 算子测试报告
  - `report/Add.md`：Add 算子测试报告
  - `report/Pow.md`：Pow 算子测试报告

## 编译与运行

1. 进入 WSL Ubuntu 22.04，并加载 CANN 环境变量：

```bash
source ~/Ascend/cann-9.0.0/set_env.sh
cd ~/workspace/ops-math
```

2. 将对应测试文件放入 `cann-ops-math` 工程的算子 example 目录，例如：

```bash
cp code/Mul/test_aclnn_mul.cpp math/mul/examples/test_aclnn_mul.cpp
cp code/Add/test_aclnn_add.cpp math/add/examples/test_aclnn_add.cpp
cp code/Pow/test_aclnn_pow.cpp math/pow/examples/test_aclnn_pow.cpp
```

3. 以 Pow 为例，按预选赛文档中的 CPU Simulator 流程编译、安装并运行：

```bash
bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run --quiet
bash build.sh --run_example pow eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

4. Mul 和 Add 将命令中的 `pow` 替换为 `mul` 或 `add` 后执行。覆盖率通过 `gcov -b -c <gcda文件路径>` 统计。

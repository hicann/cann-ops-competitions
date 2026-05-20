## 团队信息

- 团队名称：弥澄大亮
- 所属单位：闽江大学
- 团队成员：
  - 林滨炜，预选赛测试代码开发、环境配置与提交材料整理
  - 林靖朝，测试执行与结果分析
  - 李聿钦，测试报告整理与辅助验证
- 联系人：林滨炜
- 联系邮箱：2965844701@qq.com

## 环境要求

- CANN 版本：本地按环境配置指南搭建的 CANN Toolkit 环境
- 操作系统：本地 x86_64 Linux 环境
- 编译器：g++ / gcc
- 测试框架：自定义 C++ 端到端测试程序
- 其他依赖：
  - `ops-math`
  - `gcov -b`
  - CPU 模拟器

说明：预选赛测试在本地 x86_64 环境中按《环境配置指南》完成配置与执行，测试流程与官方统一 Docker 环境保持一致；最终评测环境为官方提供的 Docker 镜像 `yeren666/cann-ops-test:v1.0`。

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Mul/`：Mul 算子测试代码
  - `code/Mul/test_aclnn_mul.cpp`：Mul 算子测试主程序
  - `code/Add/`：Add 算子测试代码
  - `code/Add/test_aclnn_add.cpp`：Add 算子测试主程序
  - `code/Add/test_aclnn_inplace_add.cpp`：Add 占位/补充样例
  - `code/Pow/`：Pow 算子测试代码
  - `code/Pow/test_aclnn_pow.cpp`：Pow 算子测试主程序
- `report/`：测试报告
  - `report/Mul.md`：Mul 算子测试报告
  - `report/Add.md`：Add 算子测试报告
  - `report/Pow.md`：Pow 算子测试报告

## 编译与运行

以下以 Mul、Add、Pow 三个算子为例说明。预选赛环境使用 `ascend950` + `--simulator` 流程。

### Mul

1. 进入 `ops-math` 项目目录：
   `cd /home/workspace/ops-math`
2. 将测试文件复制到对应位置：
   `cp code/Mul/test_aclnn_mul.cpp /home/workspace/ops-math/math/mul/examples/test_aclnn_mul.cpp`
3. 编译算子并启用覆盖率：
   `bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov`
4. 安装算子包：
   `./build_out/cann-ops-math-custom_linux-x86_64.run`
5. 运行测试：
   `bash build.sh --run_example mul eager cust --vendor_name=custom --simulator --soc=ascend950 --cov`
6. 查看覆盖率：
   `find build -name "*.gcda" | grep mul`
   `gcov -b <gcda-file>`

### Add

1. 进入 `ops-math` 项目目录：
   `cd /home/workspace/ops-math`
2. 将测试文件复制到对应位置：
   `cp code/Add/test_aclnn_add.cpp /home/workspace/ops-math/math/add/examples/test_aclnn_add.cpp`
   `cp code/Add/test_aclnn_inplace_add.cpp /home/workspace/ops-math/math/add/examples/test_aclnn_inplace_add.cpp`
3. 编译算子并启用覆盖率：
   `bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov`
4. 安装算子包：
   `./build_out/cann-ops-math-custom_linux-x86_64.run`
5. 运行测试：
   `bash build.sh --run_example add eager cust --vendor_name=custom --simulator --soc=ascend950 --cov`
6. 查看覆盖率：
   `find build -name "*.gcda" | grep add`
   `gcov -b <gcda-file>`

### Pow

1. 进入 `ops-math` 项目目录：
   `cd /home/workspace/ops-math`
2. 将测试文件复制到对应位置：
   `cp code/Pow/test_aclnn_pow.cpp /home/workspace/ops-math/math/pow/examples/test_aclnn_pow.cpp`
3. 编译算子并启用覆盖率：
   `bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov`
4. 安装算子包：
   `./build_out/cann-ops-math-custom_linux-x86_64.run`
5. 运行测试：
   `bash build.sh --run_example pow eager cust --vendor_name=custom --simulator --soc=ascend950 --cov`
6. 查看覆盖率：
   `find build -name "*.gcda" | grep pow`
   `gcov -b <gcda-file>`

## 说明

- 本目录仅包含源代码、必要脚本与测试报告，不包含 `build/`、目标文件、`.gcda`、`.gcno` 等编译生成物。
- 报告内容基于本地实际保留的测试代码、构建痕迹和报告原文整理而成。

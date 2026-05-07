## 团队信息

- 团队名称：[WISE小队]
- 所属单位：[天津大学]
- 团队成员：
  - [黄丽丽]，[队长，预选赛中主要负责Mul算子的测试用例编写和报告撰写。]
  - [肖子博]，[队员，预选赛中主要负责Pow算子的测试用例编写和报告撰写。]
  - [韩坤书]，[队员，预选赛中主要负责Add算子的测试用例编写和报告撰写。]
- 联系人：[黄丽丽]
- 联系邮箱：[huangll@tju.edu.cn]

## 环境要求

- CANN 版本：[cann-9.0.0]
- 操作系统：[Ubuntu 22.04.5 LTS]
- 编译器：[gcc/g++ 11.4.0]
- 构建：[cmake 3.22.1]
- 测试框架：[example下测试代码]
- 其他依赖：[LCOV 1.14]

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Mul/`：Mul 算子测试代码
  - `code/Pow/`：Pow 算子测试代码
- `report/`：测试报告
  - `report/Mul.pdf`：Mul 算子测试报告
  - `report/Add.md`：Add 算子测试报告
  - `report/Pow.md`：Pow 算子测试报告  

## 编译与运行
以下步骤以Add算子为例，其余算子替换成对应名称即可
1. 进入对应算子目录：`cd code/Add`
2. 将测试代码拷贝到对应算子example目录下：`cp test_aclnn_add.cpp ops-math/math/add/examples/test_aclnn_add.cpp`
3. 按预选赛文档中的 CPU Simulator 流程编译、安装并运行测试：

```bash
bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run --quiet
bash build.sh --run_example add eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

4. 收集覆盖率信息
覆盖率通过 `gcov -b -c <gcda文件路径>` 统计

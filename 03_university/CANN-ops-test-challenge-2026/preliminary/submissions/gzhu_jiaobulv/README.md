## 团队信息

- 团队名称：蕉不绿队
- 所属单位：广州大学
- 团队成员：
  - 姚杰涛，队长，编写测试用例以及报告、整理比赛文件
  - 张欢，编写测试用例以及报告
  - 陈贝宁，编写测试用例以及报告
- 联系人：姚杰涛
- 联系邮箱：357447923@qq.com

## 环境要求

- CANN 版本：9.0.0
- 操作系统：[如 Ubuntu 20.04 x86_64]
- 编译器：[如 g++ 9.4.0]
- 测试框架：[如 GoogleTest 1.12.1]
- 其他依赖：[如有，逐项列出]

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Pow/`: Pow 算子测试代码
  - `code/Mul/`: Mul 算子测试代码
- `report/`：测试报告
  - `report/Add.md`：Add算子测试报告文档
  - `report/Pow.md`：Pow算子测试报告文档
  - `report/Mul.md`：Mul算子测试报告文档

## 编译与运行

[简要说明如何编译运行测试代码，例如：]

```bash
bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example add eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```
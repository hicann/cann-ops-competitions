## 团队信息

- 团队名称：做的啥都队
- 所属单位：广东财经大学
- 团队成员：
  - 冯富文，队长
  - 梁俊宇，队员
  - 陈立浩，队员
- 联系人：冯富文
- 联系邮箱：f1113.04@foxmail.com

## 环境要求

- CANN 版本：9.0.0-beta.2
- 操作系统：Ubuntu 22.04.5 LTS
- 编译器：g++ 13.2.0
- 测试框架：无
- 其他依赖：Docker镜像cann-ops-test:v1.0

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Mul/`：Mul 算子测试代码
  - `code/Pow/`：Pow 算子测试代码
- `report/`：测试报告
  - `report/Add.pdf`：Add 算子测试报告
  - `report/Mul.pdf`：Mul 算子测试报告
  - `report/Pow.pdf`：Pow 算子测试报告

## 编译与运行

1. 进入对应算子目录：`cd code/Add`
2. 编译：`mkdir build && cd build && cmake .. && make`
3. 运行：`./test_aclnn_add`
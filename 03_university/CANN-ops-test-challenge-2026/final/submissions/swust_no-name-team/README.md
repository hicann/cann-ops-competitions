## 团队信息

- 团队名称：不知名小队
- 所属单位：西南科技大学
- 团队成员：
  - 向非洲，队长
  - 陈杨，队员
- 联系人：向非洲
- 联系邮箱：2268083421@qq.com

## 环境要求

- CANN 版本：8.0.RC1
- 操作系统：Ubuntu 20.04 x86_64
- 编译器：g++ 9.4.0
- 测试框架：GoogleTest 1.12.1

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Cumsum/`：Cumsum 算子测试代码
- `report/`：测试报告
  - `report/Add.md`：Add测试报告文档
  - `report/Cumsum.md`：Cumsum测试报告文档

## 编译与运行

1. 进入对应算子目录：`cd code/Add`
2. 编译：`mkdir build && cd build && cmake .. && make`
3. 运行：`./test_aclnn_add`
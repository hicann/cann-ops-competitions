# CANN 算子测试挑战赛 - 决赛作品 (Final)

## 团队信息

- 团队名称：龙湖小队
- 所属单位：南京工业大学
- 团队成员：
  - 李允乐，队长
- 联系人：李允乐
- 联系邮箱：3910387373@qq.com

## 环境要求

- CANN 版本：9.0.0-beta.2
- 操作系统: Ubuntu 22.04.5 LTS
- 编译器：Aarch64 g++
- 测试框架：GoogleTest
- 其他依赖：Ascend CANN Toolkit (9.0.0-beta.2), CMake

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码 (`test_aclnn_add.cpp`)
  - `code/Cumsum/`：Cumsum 算子测试代码 (`test_aclnn_cumsum.cpp`)
- `report/`：测试报告
  - `report/report_add.md`：Add 算子测试报告
  - `report/report_cumsum.md`：Cumsum 算子测试报告

## 编译与运行

1. 进入对应算子目录，例如：`cd code/Add`
2. 编译：`mkdir -p build && cd build && cmake .. && make`
3. 运行：`./test_aclnn_add`

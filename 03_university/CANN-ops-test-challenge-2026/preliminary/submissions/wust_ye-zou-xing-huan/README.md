# 预选赛作品提交

## 团队信息

- 团队名称：夜奏星环
- 所属单位：武汉科技大学（WUST）
- 团队成员：
  - 张钊洋，队长
- 联系人：张钊洋
- 联系邮箱：dobqop999@gmail.com

## 环境要求

- CANN 版本：8.0.RC1
- 操作系统：Ubuntu 22.04.5 LTS (aarch64)
- 内核：5.10.0 (openEuler 22.03 SP4)
- CPU：华为鲲鹏 Kunpeng 920 7280Z
- 架构：aarch64
- 内存：2.0TiB
- 编译器：g++ (GCC) 9.4.0
- 测试框架：GoogleTest

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Mul/`：Mul 算子测试代码
  - `code/Add/`：Add 算子测试代码
  - `code/Pow/`：Pow 算子测试代码
- `report/`：测试报告
  - `report/Mul.md`：Mul 算子测试报告
  - `report/Add.md`：Add 算子测试报告
  - `report/Pow.md`：Pow 算子测试报告

## 编译与运行

1. 进入对应算子目录：`cd code/Mul`
2. 编译：`mkdir build && cd build && cmake .. && make`
3. 运行：`./test_aclnn_mul`

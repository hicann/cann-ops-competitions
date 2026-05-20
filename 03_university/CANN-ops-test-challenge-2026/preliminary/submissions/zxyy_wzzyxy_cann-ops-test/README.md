# CANN 算子测试用例设计 - 预选赛

## 团队信息
- **团队名称**：zxyy
- **团队成员**：卢梓轩、陈文朋、成志杰
- **所属单位**：wzzyxy
- **联系方式**：2462791291@qq.com

## 作品简介
本作品为 CANN ops-math 仓库中 Mul、Add、Pow 算子编写的端到端测试用例，尽可能覆盖算子的各种执行路径，以提升代码覆盖率为目标。

## 赛段
预选赛（preliminary）

## 提交内容
- **add/**：Add 算子测试用例（test_aclnn_add.cpp）及覆盖率产物（build/）
- **mul/**：Mul 算子测试用例（test_aclnn_mul.cpp）及覆盖率产物（build/）
- **pow/**：Pow 算子测试用例（test_aclnn_pow.cpp）及覆盖率产物（build/）

## 运行说明
各算子子目录下包含独立的测试用例源文件与 build 目录，请在官方 Docker 环境（`yeren666/cann-ops-test:v1.0`）中按赛题说明编译运行。

# CANN 算子测试用例设计 - 总决赛

## 团队信息
- **团队名称**：zxyy
- **团队成员**：[成员姓名]
- **所属单位**：wzzyxy
- **联系方式**：[邮箱或其他联系方式]

## 作品简介
本作品为 CANN ops-math 仓库中 Add、Cumsum 算子编写的端到端测试用例，覆盖算子各种执行路径并深入分析精度特性，在真实 Ascend 910_93 NPU 环境下运行。

## 赛段
总决赛（final）

## 提交内容
- **add/**：Add 算子测试用例（test_aclnn_add.cpp）及覆盖率产物（build/）
- **cumsum/**：Cumsum 算子测试用例（test_aclnn_cumsum.cpp）及覆盖率产物（build/）

## 运行说明
各算子子目录下包含独立的测试用例源文件与 build 目录，请在决赛远程 Ascend 910_93 NPU 服务器环境中按赛题说明编译运行。

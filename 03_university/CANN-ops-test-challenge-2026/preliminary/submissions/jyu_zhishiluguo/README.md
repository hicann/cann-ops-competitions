## 团队信息

- 团队名称：只是路过
- 所属单位：嘉应学院
- 团队成员：
  - 陈俊恒，共同算子测试相关工作，团队内协同互助，遇到问题共同研讨解决
  - 苏有道，共同算子测试相关工作，团队内协同互助，遇到问题共同研讨解决
- 联系人：陈俊恒
- 联系邮箱：1837501793@qq.com

## 环境要求

- CANN 版本：9.0.0
- 操作系统：Ubuntu 24.04 x86_64
- 编译器：Python 3.12.3  gcc 13.3.0 cmake 3.28.3

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Mul/`：Mul 算子测试代码
  - `code/Pow/`：Pow 算子测试代码
- `report/`：测试报告
  - `report/Add.md`：add算子测试报告文档
  - `report/Mul.md`：mul算子测试报告文档
  - `report/Pow.md`：pow算子测试报告文档

## 编译与运行

1. 进入对应算子目录：`cd code/Add`
2. 编译：`mkdir build && cd build && cmake .. && make`
3. 运行：`./test_aclnn_add`
4. 重复1-3操作编译运行其他算子
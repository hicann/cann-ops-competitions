\## 团队信息



\- 团队名称：咕咕嘎嘎小队

\- 所属单位：南京大学

\- 团队成员：

 - 曾炜乐，负责add算子的测试

 - 林天乐，负责测试检测

 - 邓昊哲，负责算子的测试

\- 联系人：曾炜乐

\- 联系邮箱：1936877401@qq.com



\## 环境要求



\- CANN 版本：9.0.0

\- 操作系统：Ubuntu 20.04 x86\_64

\- 编译器：g++ 9.4.0

\- 测试框架：无

\- 其他依赖：无



\## 文件说明



\- `code/`：测试代码源文件，按算子分子目录组织

 - `code/Add/`：Add 算子测试代码

 - `code/cumsum/`：cumsum 算子测试代码

 - ...

\- `report/`：测试报告

 - `report/add/report.pdf`：add测试报告文档
 - `report/cumsum/report.pdf`：cumsum测试报告文档



\## 编译与运行





1\. 进入对应算子目录：`cd code/Add`

2\. 编译：`mkdir build \&\& cd build \&\& cmake .. \&\& make`

3\. 运行：`./test\_aclnn\_add`


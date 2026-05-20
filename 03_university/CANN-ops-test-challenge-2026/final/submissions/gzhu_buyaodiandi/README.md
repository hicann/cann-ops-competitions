# 预选赛作品提交说明

## 团队信息

- 团队名称：不要垫底
- 所属单位：广州大学
- 团队成员：
  - 滕佳琳，测试用例设计、测试报告撰写、覆盖率分析
  - 李楚湘，测试代码实现、运行验证、结果整理
- 联系人：滕佳琳
- 联系邮箱：3431613891@qq.com

## 环境要求

- CANN 版本：CANN 9.0.0-beta.2
- SOC 版本：ascend910_93
- 操作系统：Ubuntu 20.04 x86_64
- 编译器：g++ 9.4.0
- 构建工具：CMake
- 测试框架：基于 C++ 的端到端测试程序
- 覆盖率工具：gcov
- 算子库：cann-ops-math
- Docker 镜像：yeren666/cann-ops-test:v1.0
- 其他依赖：Ascend CANN 开发与运行环境、ACLNN 接口相关头文件与动态库

## 文件说明

- `code/`：测试代码源文件，按算子分目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Cumsum/`：Cumsum 算子测试代码
- `report/`：测试报告
  - `report/report.pdf`：测试报告主文档
- `README.md`：作品提交说明文档

## 编译与运行

本作品基于大赛提供的 CANN ops-math 环境进行测试，测试目标为 Ascend 910 系列评测环境，SOC 参数为 `ascend910_93`。

### 1. 进入测试目录

以 Add 算子为例：

```bash
cd code/Add
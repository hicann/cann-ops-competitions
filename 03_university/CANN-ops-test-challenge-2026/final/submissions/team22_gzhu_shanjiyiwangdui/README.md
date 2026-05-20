# 总决赛作品提交说明

## 团队信息

- 团队名称：闪击翼王队
- 所属单位：广州大学
- 团队成员：
  - 周亚超，项目负责人、测试框架搭建、代码整合与编译调试
  - 张雨桐，算子测试用例设计、结果验证、覆盖率优化
  - 许恒恒，测试报告撰写、文档整理、提交规范检查
- 联系人：张雨桐
- 联系邮箱：2559888264@qq.com

## 环境要求

- CANN 版本：9.0.0
- 操作系统：Ubuntu 20.04 x86_64
- 编译器：g++ 9.4.0
- 测试框架：CANN ACLNN 原生测试框架
- 其他依赖：ACL 昇腾运行时库


## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Cumsum/`：Cumsum 算子测试代码
  - ...
- `report/`：测试报告
  - `report/add.md`：Add算子测试报告文档
  - `report/cumsum.md`：Cumsum算子测试报告文档

## 编译与运行

1. 进入对应算子目录：`cd code/Add`
2. 编译：`bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov`
3. 安装：`./build_out/cann-ops-math-custom_linux-aarch64.run`
4. 运行：`bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov`
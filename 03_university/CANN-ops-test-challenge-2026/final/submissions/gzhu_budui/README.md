## 团队信息

- 团队名称：不队
- 所属单位：广州大学
- 团队成员：
  - 杨金鹏，负责Add算子测试
  - 刘畅，负责Cumsum算子测试
- 联系人：杨金鹏
- 联系邮箱：2081498716@qq.com

## 环境要求

- CANN 版本：8.0.RC1
- 操作系统：Ubuntu 20.04 x86_64
- 编译器：g++ 9.4.0
- 测试框架：无（原生 C++）
- 其他依赖：ACL / ACLNN


## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
    - `code/Add/`：Add 算子测试代码
    - `code/Cumsum/`：Cumsum 算子测试代码

- `report/`：测试报告
    - `report/Add.md`：Add 算子测试文档
    - `report/Cumsum.md`：Cumsum 算子测试文档

## 编译与运行

1. 进入对应算子目录：`cd code/Add`
2. 编译：`bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov`
3. 安装：`./build_out/cann-ops-math-custom_linux-aarch64.run`
4. 运行：`bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov`

如需测试其他算子，将目录和可执行文件名替换为对应算子即可。

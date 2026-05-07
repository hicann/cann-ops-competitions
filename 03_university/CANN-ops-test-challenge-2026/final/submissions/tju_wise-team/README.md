## 团队信息

- 团队名称：[WISE小队]
- 所属单位：[天津大学]
- 团队成员：
  - [黄丽丽]，[队长，决赛中主要负责Cumsum算子的精度测试和报告撰写。]
  - [肖子博]，[队员，预选赛中主要负责Cumsum算子的覆盖率提升。]
  - [韩坤书]，[队员，预选赛中主要负责Add算子的测试用例编写和报告撰写。]
- 联系人：[黄丽丽]
- 联系邮箱：[huangll@tju.edu.cn]

## 环境要求

决赛云服务环境

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Cumsum`：Cumsum 算子测试代码
- `report/`：测试报告
  - `report/Add.md`：Add 算子测试报告
  - `report/Cumsum.md`：Cumsum 算子测试报告  
  - `report/assets/`：报告中的图片

## 编译与运行
以下步骤以Add算子为例，其余算子替换成对应名称即可
1. 进入对应算子目录：`cd code/Add`
2. 将测试代码拷贝到对应算子example目录下：`cp test_aclnn_add.cpp ops-math/math/add/examples/test_aclnn_add.cpp`
3. 按决赛文档中的命令执行编译、安装算子、执行测试等流程：

```bash
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-aarch64.run
bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

4. 收集覆盖率信息
覆盖率通过 `gcov -b -c <gcda文件路径>` 统计

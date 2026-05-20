# 星辰队提交说明

## 团队信息

- 团队名称：星辰
- 所属单位：金陵科技学院
- 团队成员：
  - 张健伟，测试代码编写与报告整理
  - 陈羽洁，测试用例设计与结果分析
- 联系人：张健伟
- 联系邮箱：zhangjianwei0615@163.com

## 环境要求

- CANN 版本：组委会决赛远程云服务环境内置 CANN 与 `cann-ops-math`
- 操作系统：组委会决赛远程云服务 Linux 环境
- 编译器：评测环境默认 GCC/G++
- 测试框架：自研 C++ 测试程序，基于 ACL Runtime、ACLNN 两段式接口和 NPU Stream
- 运行方式：Ascend 910 系列真实 NPU，SOC 参数 `ascend910_93`，Eager Mode，开启 `--cov` 覆盖率插桩
- 其他依赖：组委会提供的 `ops-math.zip` 及远程环境内置依赖

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/test_aclnn_add.cpp`：Add 算子测试代码
  - `code/Cumsum/test_aclnn_cumsum.cpp`：Cumsum 算子测试代码
- `report/`：测试报告
  - `report/Add.md`：Add 算子测试报告
  - `report/Cumsum.md`：Cumsum 算子测试报告

## 编译与运行

1. 按决赛全流程说明登录组委会远程云服务环境，并将 `/public` 中的赛题资料拷贝至 `/root`：

```bash
cp -r /public/* /root/
unzip ops-math.zip -d /root/ops-math
cd /root/ops-math/ops-math
```

2. 将对应测试文件放入 `cann-ops-math` 工程的算子 example 目录，例如：

```bash
cp code/Add/test_aclnn_add.cpp math/add/examples/test_aclnn_add.cpp
cp code/Cumsum/test_aclnn_cumsum.cpp math/cumsum/examples/test_aclnn_cumsum.cpp
```

3. 以 Cumsum 为例，按决赛题目文档中的真实 NPU 流程编译、安装并运行：

```bash
bash build.sh --pkg --soc=ascend910_93 --ops=cumsum --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-aarch64.run
bash build.sh --run_example cumsum eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

4. Add 将命令中的 `cumsum` 替换为 `add` 后执行。覆盖率通过 `gcov -b -c <gcda文件路径>` 统计。

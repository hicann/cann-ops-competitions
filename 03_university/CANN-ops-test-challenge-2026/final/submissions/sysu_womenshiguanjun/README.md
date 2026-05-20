### 1. 团队信息

- 团队名称：我们是冠军
- 所属单位：中山大学
- 团队成员：
  - 叶旭峰，队长
  - 彭翔，队员
- 联系人：叶旭峰
- 联系邮箱：yelfs9@qq.com

### 2. 环境要求

- CANN 版本：9.0.0-beta.2
- 操作系统：Ubuntu 22.04.5 LTS(aarch64)
- 编译器：g++ 13.2.0
- 测试框架：无
- 其他依赖：无

### 3.文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Cumsum/`：Cumsum 算子测试代码

- `report/`：测试报告
  - `report/Cumsum.md`：Cumsum 算子测试文档
  - `report/Add.md`：Add 算子测试文档

#### 编译与运行

1. 进入对应算子目录：`cd code/Add`
2. 编译：

```
cd /home/workspace/ops-math
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
```

3.安装算子包

```
./build_out/cann-ops-math-custom_linux-x86_64.run
```

4. 运行测试：

```
bash build.sh --run_example mul eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

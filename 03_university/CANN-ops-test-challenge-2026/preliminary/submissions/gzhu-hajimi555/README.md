## 团队信息

- 团队名称：哈基米555
- 所属单位：广州大学
- 团队成员：
  - 庞浩民，算子测试代码编写与测试
  - 兰晨阳，算子测试代码编写与测试
  - 罗景辉，算子测试代码编写与测试
- 联系人：庞浩民
- 联系邮箱：panghaomin@e.gzhu.du.cn


## 环境要求

- CANN 版本：9.0.0
- 操作系统：Ubuntu 22.04 x86_64
- 编译器：g++ 9.4.0 /gcc 9.4.0
- 测试框架：CANN ACLNN 原生测试框架
- 其他依赖：Docker镜像：yeren666/cann-ops-test:v1.0,ACL 昇腾运行时库,CMake 3.16+


## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Mul/`：Mul 算子测试代码
  - `code/Pow/`: Pow 算子测试代码
  - ...
- `report/`：测试报告
  - `report/report.md`：测试报告主文档



## 编译与运行

以 Add 算子为例（Mul 和 Pow 只需将命令中的 `add` 替换为 `mul` 或 `pow`）：

1. 进入对应算子目录：`cd code/Add`

2. 复制测试文件到`ops-math`项目对应的位置

  ```bash
  cp test_aclnn_add.cpp /home/workspace/ops-math/math/add/examples/test_aclnn_add.cpp
  ```

3. 编译

  ```bash
  # 切换到ops-math项目目录
  cd /home/workspace/ops-math
  # 编译算子
  bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
  ```
  启用覆盖率统计。编译成功后在 build_out/ 下生成算子安装包。

4. 安装算子包

  ```bash
  ./build_out/cann-ops-math-custom_linux-x86_64.run
  ```

5. 运行测试

  ```bash
  bash build.sh --run_example mul eager cust \
      --vendor_name=custom --simulator --soc=ascend950 --cov
  ```

  运行成功后会在 `build/` 目录下生成覆盖率数据文件（`.gcda`）。

6. 查看覆盖率

  ```bash
  find build -name "*.gcda" | grep add
  gcov -b <gcda文件路径>
  ```
  gcov 输出的 Lines executed: XX.XX% of YY 即为行覆盖率。每次修改测试用例后，需重新执行步骤 2-5。


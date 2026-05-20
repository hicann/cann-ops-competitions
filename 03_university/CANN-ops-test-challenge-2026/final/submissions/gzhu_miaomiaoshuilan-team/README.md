## 团队信息

- 团队名称：喵喵水蓝队
- 所属单位：广州大学
- 团队成员：
  - 都铭宇，队长
  - 许裕滔，队员
- 联系人：都铭宇
- 联系邮箱：2745241708@qq.com

## 环境要求

- CANN 版本：9.0.0-beta.2
- 操作系统: Ubuntu 22.04.5 LTS aarch64
- 编译器：Aarch64 g++ 11.4.0
- 测试框架：GoogleTest
- 其他依赖：Ascend CANN Toolkit (9.0.0-beta.2), CMake


## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/test_aclnn_add.cpp`：Add 算子测试代码
  - `code/Cumsum/test_aclnn_cumsum.cpp`：Cumsum 算子测试代码
- `report/`：测试报告
  - `report/Add.md`：Add 算子测试报告文档
  - `report/Cumsum.md`：Cumsum 算子测试报告文档

## 编译与运行

以 Add 算子为例（Cumsum只需将命令中的 `add` 替换为 `cumsum`）：

1. 进入对应算子目录：`cd code/Add`
2. 复制测试文件到 `ops-math`项目对应的位置

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
4. 安装算子包

   ```bash
   ./build_out/cann-ops-math-custom_linux-x86_64.run
   ```
5. 运行

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


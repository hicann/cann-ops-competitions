## 团队信息

- 团队名称：群星闪耀时
- 所属单位：南昌航空大学
- 团队成员：
  - 李梓暄，负责协调任务，编译调试以及代码汇总
  - 姜傲天，算子测试用例设计与优化
  - 李嘉纶，环境配置、测试报告撰写、提交规范检查
- 联系人：李梓暄
- 联系邮箱：2693465377@qq.com

## 环境要求

- CANN 版本：9.0.0
- 操作系统：Ubuntu 20.04 x86_64及以上
- 编译器：g++ 9.4.0 /gcc 9.4.0
- 测试框架： CANN ACLNN 原生测试框架
- 其他依赖：ACL 昇腾运行时库、CMake 3.16+



## 文件说明

- ```
  code/
  ```

  ：测试代码源文件，按算子分子目录组织

  - `code/Add/test_aclnn_add.cpp`：Add 算子测试代码
  - `code/Cumsum/test_aclnn_cumsum.cpp`：Cumsum 算子测试代码
  
- ```
  report/
  ```

  ：测试报告

  - `report/Add.pdf`：Add 算子测试报告文档
  - `report/Cumsum.pdf`：Cumsum算子测试报告文档

## 编译与运行

以 Add 算子为例（Cumsum 只需将命令中的 `add` 替换为 `cumsum` ）：

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


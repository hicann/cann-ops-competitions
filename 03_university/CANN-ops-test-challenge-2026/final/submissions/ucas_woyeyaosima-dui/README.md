## 团队信息

- 团队名称：我也要死吗队
- 所属单位：中国科学院大学
- 团队成员：
  - 贾皓文
- 联系人：贾皓文
- 联系邮箱：rokuyo@126.com

## 环境要求

- CANN 版本：9.0.0
- 操作系统：Ubuntu 20.04 x86_64及以上
- 编译器：g++ 9.4.0 /gcc 9.4.0
- 测试框架： CANN ACLNN 原生测试框架
- 其他依赖：gcov、ACL 昇腾运行时库、CMake 3.16+



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

  - `report/Add.md`：Add 算子测试报告文档
  - `report/Cumsum.md`：Cumsum算子测试报告文档

## 编译与运行

以 Add 算子为例（Cumsum 只需将命令中的 `add` 替换为 `cumsum` ）：

```bash
# 编译（启用覆盖率插桩）
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov

# 安装算子包
./build_out/cann-ops-math-custom_linux-aarch64.run

# 运行测试（真实 NPU 环境）
bash build.sh --run_example add eager cust \
    --vendor_name=custom --soc=ascend910_93 --cov

# 查看覆盖率
find build -name "*.gcda" | grep add
gcov -b -c <gcda文件路径>
```



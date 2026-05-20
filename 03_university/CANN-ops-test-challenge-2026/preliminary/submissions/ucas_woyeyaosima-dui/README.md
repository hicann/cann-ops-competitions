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
  - `code/Mul/test_aclnn_mul.cpp`：Mul 算子测试代码
  - `code/Pow/test_aclnn_pow.cpp`：Pow 算子测试代码

  

## 编译与运行

以 Mul 算子为例（Add 和 Pow 只需将命令中的 `mul` 替换为 `add` 或 `pow`）：

## 1. 编译算子

```bash
cd /home/workspace/ops-math
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
```
`--cov` 启用覆盖率统计。编译成功后在 `build_out/` 下生成算子安装包。

## 2. 安装算子包

```bash
./build_out/cann-ops-math-custom_linux-x86_64.run
```

## 3. 运行测试

```bash
bash build.sh --run_example mul eager custom \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

运行成功后会在 `build/` 目录下生成覆盖率数据文件（`.gcda`）。

## 4. 查看覆盖率

```bash
find build -name "*.gcda" | grep mul
gcov -b <gcda文件路径>
```

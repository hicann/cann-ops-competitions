## 团队信息

- 团队名称：别骂了在练了
- 所属单位：广州职业技术大学
- 团队成员：
  - 钟华利，测试用例设计、覆盖率分析、
  - 卢泽艺，测试代码编写、报告撰写
  - 黄鸿祥，环境配置、结果验证
- 联系人：钟华利
- 联系邮箱： zhl18825674687@qq.com

## 环境要求

- CANN 版本：[9.0.0]
- 操作系统：Ubuntu 24.04 x86_64 / Linux x86_64
- 编译器：[g++ 13.3.0]
- 测试框架：CANN ACLNN 示例测试
- 其他依赖：
  - gcov / lcov
  - cmake
  - make

## 文件说明

  - `code/`：测试代码源文件
    - `code/Add/`：Add 算子测试代码
    - `code/Mul/`：Mul 算子测试代码
    - `code/Pow/`：Pow 算子测试代码
  - `report/`：测试报告
    - `report/Add.pdf`：Add 算子测试报告
    - `report/Mul.pdf`：Mul 算子测试报告
    - `report/Pow.pdf`：Pow 算子测试报告

  ## 编译与运行

### Add算子

#### 1. 编译算子包

  ```
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
  ```

  #### 2. 安装算子包

  ```
./build_out/cann-ops-math-custom_linux-aarch64.run
  ```

  #### 3. 运行测试用例

  ```
bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
  ```

#### 4. 查看覆盖率

```
find build -name "*.gcda" | grep add
gcov -b <gcda文件路径>
```

### Mul算子

#### 1. 编译算子包

  ```
bash build.sh --pkg --soc=ascend910_93 --ops=mul --vendor_name=custom --cov
  ```

  #### 2. 安装算子包

  ```
./build_out/cann-ops-math-custom_linux-aarch64.run
  ```

  #### 3. 运行测试用例

  ```
bash build.sh --run_example mul eager cust --vendor_name=custom --soc=ascend910_93 --cov
  ```
#### 4. 查看覆盖率

```
find build -name "*.gcda" | grep mul
gcov -b <gcda文件路径>
```

  ### Pow算子

#### 1. 编译算子包

  ```
bash build.sh --pkg --soc=ascend910_93 --ops=pow --vendor_name=custom --cov
  ```

  #### 2. 安装算子包

  ```
./build_out/cann-ops-math-custom_linux-aarch64.run
  ```

  #### 3. 运行测试用例

  ```
bash build.sh --run_example pow eager cust --vendor_name=custom --soc=ascend910_93 --cov
  ```

#### 4. 查看覆盖率

```
find build -name "*.gcda" | grep pow
gcov -b <gcda文件路径>
```
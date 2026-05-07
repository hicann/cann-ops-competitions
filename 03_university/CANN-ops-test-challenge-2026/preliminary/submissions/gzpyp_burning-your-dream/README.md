# README

## 团队信息

- 团队名称：燃烧你的梦
- 所属单位：广州职业技术大学
- 团队成员：
  - 国庭轩，测试用例设计、覆盖率分析、报告撰写
  - 李先杰，测试代码编写、环境配置、结果验证
- 联系人：国庭轩
- 联系邮箱：[1634411433@qq.com]

---

## 环境要求

- CANN 版本：[9.0.0]
- 操作系统：Ubuntu 24.04 x86_64 / Linux x86_64
- 编译器：[g++ 13.3.0]
- 测试框架：CANN ACLNN 示例测试 
- 其他依赖：
  - gcov / lcov
  - cmake
  - make

---

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add`：Add 算子测试代码
  - `code/Mul/`：Mul 算子测试代码
  - `code/Pow/`：Pow 算子测试代码

- `report/`：测试报告
  - `report/Add算子测试报告.pdf`：Add算子测试报告
  - `report/Mul算子测试报告.pdf`：Mul算子测试报告
  - `report/Pow算子测试报告.pdf`：Pow算子测试报告     


- `README.md`：项目说明文件

---

## 编译与运行

1. 进入项目工作目录：

```bash
cd /home/workspace/ops-math
```

2. 编译 Cumsum 算子并开启覆盖率：

```bash
# 编译（启用覆盖率插桩）
bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov

bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov

bash build.sh --pkg --soc=ascend950 --ops=pow --vendor_name=custom --cov

# 安装算子包
./build_out/cann-ops-math-custom_linux-x86_64.run --quiet

# 运行测试（模拟 NPU 环境）
bash build.sh --run_example add eager cust --vendor_name=custom --simulator --soc=ascend950 --cov

bash build.sh --run_example mul eager cust --vendor_name=custom --simulator --soc=ascend950 --cov

bash build.sh --run_example pow eager cust --vendor_name=custom --simulator --soc=ascend950 --cov

# 查看覆盖率
find build -name "*.gcda" | grep add
gcov -b -c <gcda文件路径>

find build -name "*.gcda" | grep mul
gcov -b -c <gcda文件路径>

find build -name "*.gcda" | grep pow
gcov -b -c <gcda文件路径>


```

---
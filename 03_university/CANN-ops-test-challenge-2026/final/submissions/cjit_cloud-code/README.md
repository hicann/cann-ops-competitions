## 团队信息

- 团队名称：云上码术
- 所属单位：长江工程职业技术学院
- 团队成员：
  - 陈帅，队长
  - 沈均皓，队员
  - 叶慧琳, 队员
- 联系人：陈帅
- 联系邮箱：2892480843@qq.com

## 环境要求

- CANN 版本：cann-9.0.0-beta.2
- 操作系统：Ubuntu 22.04.5 LTS aarch64
- 编译器：g++ 11.x
- 测试框架：GoogleTest 1.12.1
- 其他依赖：
    - Ascend Toolkit / AscendCL
    - tikcpp / ascendc_kernel_cmake
    - CMake
    - CPack
    - gcov
    - third_party/ascend_protobuf
    - opbase
    - Python3

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Cumsum/`：Cumsum 算子测试代码
  - ...
- `report/`：测试报告
  - `report/Add.md`：Add 算子测试报告文档
  - `report/Cumsum.md`：Cumsum 算子测试报告文档

## 编译与运行
1. 进入算子目录 `cd code/Add`
cd /root/a_test
2. 编译算子包并开启覆盖率
  `bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov`
3. 检查覆盖率插桩文件
  `find build -name "add_tiling*.gcno"`
4. 安装算子包
  `./build_out/cann-ops-math-custom_linux-aarch64.run`
5. 运行 Add example 测试
  `bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov`
6. 覆盖率查看：
  `find build -name "*.gcda" | grep add`
  `gcov -b <gcda文件路径>`

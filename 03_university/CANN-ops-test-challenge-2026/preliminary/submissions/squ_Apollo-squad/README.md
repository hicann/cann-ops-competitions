## 团队信息

- 团队名称：Apollo-squad
- 所属单位：宿迁学院
- 团队成员：
  - 樊旺，代码开发
  - 胡子航，运行测试
  - 王超，报告撰写
- 联系人：樊旺
- 联系邮箱：2308577512@qq.com

## 环境要求

- CANN 版本：8.0.RC1
- 操作系统：ubuntu 20.04
- 编译器：g++ 9.4.0
- CPU架构：x86_64
- Python版本：3.8.10


## 文件说明

- code/：测试代码
- report/：测试报告

## 编译与运行

1. 拷贝文件夹：  `cp -r /public/* /root/`
2. 进入目录：`cd /root/ops-math`
3. 编译add算子：`bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov`
4. 安装算子包：`./build_out/cann-ops-math-custom_linux-aarch64.run`
5. 运行测试：`bash build.sh --run_example add eager cust --vendor_name=custom \ --soc=ascend910_93 \ --cov`
6. 编译mul算子：`bash build.sh --pkg --soc=ascend910_93 --ops=mul --vendor_name=custom --cov`
7. 安装算子包：`./build_out/cann-ops-math-custom_linux-aarch64.run`
8. 运行测试`bash build.sh --run_example mul eager cust \ --vendor_name=custom \ --soc=ascend910_93 \ --cov`
9. 编译pow算子：`bash build.sh --pkg --soc=ascend910_93 --ops=pow --vendor_name=custom --cov`
10. 安装算子包：`./build_out/cann-ops-math-custom_linux-aarch64.run`
11. 运行测试`bash build.sh --run_example pow eager cust \ --vendor_name=custom \ --soc=ascend910_93 \ --cov`
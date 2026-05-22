# seig_ee · 智能算子测试大赛（预选赛）

> 队伍目录：`preliminary/submissions/seig_ee/`，命名符合 `{school}_{team-name}`（**seig** · 广州软件学院，**ee** · 队名）。

## 团队信息

- 团队名称：ee
- 所属单位：广州软件学院
- 团队成员：
  - 林世伟，队长（测试代码编写与报告整理）
  - 胡文康，队员（测试开发与文档）
  - 赖永健，队员（测试开发与调试）
- 联系人：林世伟
- 联系邮箱：227654979@qq.com

## 环境要求

- CANN 版本：9.0.0 beta 2（构建日志路径示例：`/usr/local/Ascend/cann-9.0.0-beta.2/...`）
- 操作系统：Linux (aarch64)；具体发行版可在构建环境执行 `cat /etc/os-release` 核对（如 Ubuntu、EulerOS）
- 编译器：GCC / G++ 11.4.0
- 测试框架：GoogleTest (gtest)
- 其他依赖：Ascend ACL / ACLNN（随 CANN）、CMake、make/gmake、Python 3（`ops-math` 构建脚本）、lcov/gcov（覆盖率统计，可选）
- Docker：未使用大赛 Docker 镜像；评测以组委会统一环境为准

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织  
  - `code/Add/test_aclnn_add.cpp`：Add 算子测试代码  
  - `code/Mul/test_aclnn_mul.cpp`：Mul 算子测试代码  
  - `code/Pow/test_aclnn_pow.cpp`：Pow 算子测试代码  
- `report/`：测试报告（Markdown，可按算子分文件）  
  - `report/Add.md`：Add 算子测试设计说明  
  - `report/Mul.md`：Mul 算子测试设计说明  
  - `report/Pow.md`：Pow 算子测试设计说明  

本目录**不**包含 `build/`、`*.o`、`*.gcda`、`*.gcno` 等编译产物；若有图片附件可置于 `report/assets/`（当前无）。

## 编译与运行

测试代码须并入官方 **`cann-ops-math`** 工程对应算子 `examples/`（或等价 CMake 目标），在已配置 **ASCEND_HOME**、**LD_LIBRARY_PATH** 的 Linux + CANN 环境下编译运行。评测由组委会在统一环境中重新编译执行。

**典型流程（摘自 `report/*.md`，算子名 `add`/`mul`/`pow` 替换）**

1. 进入克隆的 `ops-math` 工程根目录。  
2. 将本仓库中 `test_aclnn_*.cpp` 拷贝到 `math/<算子>/examples/`（或按报告说明集成）。  
3. 加载环境并构建、安装、跑示例（示例常使用 `ascend950` + `--simulator`，真机请改 `--soc` 与安装包名）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-x86_64.run
bash build.sh --run_example add eager cust --vendor_name=custom --simulator --soc=ascend950 --cov
```

4. 各用例输出 `[PASS]` / `[FAIL]`，失败时进程非零退出；详细命令、容差与覆盖率说明见 **`report/Add.md`、`Mul.md`、`Pow.md`**。

**PR 标题格式（向主仓提交时）**：`[团队提交] seig_ee`

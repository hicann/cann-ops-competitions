# seig_ee · 智能算子测试大赛（决赛）

> 队伍目录：`final/submissions/seig_ee/`，命名符合 `{school}_{team-name}`（**seig** · 广州软件学院，**ee** · 队名）。

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
- 操作系统：Linux (aarch64)；具体发行版可在构建环境执行 `cat /etc/os-release` 核对
- 编译器：GCC / G++ 11.4.0
- 测试框架：GoogleTest (gtest)
- 硬件：Ascend NPU（开发与报告基于 `ascend910_93` 场景撰写）
- 其他依赖：Ascend ACL / ACLNN（随 CANN）、CMake、make/gmake、Python 3、`ops-math` 构建脚本、gcov/lcov（覆盖率）
- Docker：未使用大赛 Docker 镜像；评测以组委会统一环境为准

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织  
  - `code/Add/test_aclnn_add.cpp`：Add 算子测试代码  
  - `code/Cumsum/test_aclnn_cumsum.cpp`：Cumsum 算子测试代码  
- `report/`：测试报告（Markdown）  
  - `report/Add.md`：Add 算子测试报告（含组委会脚本用元信息区块）  
  - `report/Cumsum.md`：Cumsum 算子测试报告（含组委会脚本用元信息区块）  

本目录**不**包含 `build/`、`*.o`、`*.gcda`、`*.gcno` 等编译产物；若有图片附件可置于 `report/assets/`（当前无）。

## 编译与运行

测试代码须并入官方 **`cann-ops-math`** 工程；**Add** 与 **Cumsum** 在决赛环境下需按报告对 **`math/add/CMakeLists.txt`**、**`math/cumsum/CMakeLists.txt`** 等做赛题要求的前置修改（详见分报告），否则 tiling / arch 映射可能无法参与覆盖率统计。

**典型流程（在 `ops-math` 根目录，请先 `source` CANN `set_env.sh`）**

**Add：**

```bash
# CMake 前置修补（完整 sed 见 report/Add.md）
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-aarch64.run
bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

**Cumsum：**

```bash
# 先按 report/Cumsum.md「三、覆盖率分析」修补 math/cumsum/CMakeLists.txt
bash build.sh --pkg --soc=ascend910_93 --ops=cumsum --vendor_name=custom --cov
./build_out/cann-ops-math-custom_linux-aarch64.run
bash build.sh --run_example cumsum eager cust --vendor_name=custom --soc=ascend910_93 --cov
# 可选：export CUMSUM_STRICT_VALIDATE=1 启用严格数值校验
```

覆盖率、`gcov` 路径与精度分析见 **`report/Add.md`、`report/Cumsum.md`**。

**PR 标题格式（向主仓提交时）**：`[团队提交] seig_ee`

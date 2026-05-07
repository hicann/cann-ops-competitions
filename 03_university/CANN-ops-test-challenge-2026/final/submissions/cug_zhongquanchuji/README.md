# CANN ops-math 总决赛算子测试用例说明

## 团队信息

- 团队名称：重拳出击
- 所属单位：中国地质大学（武汉）
- 团队成员：
  - 郭宸羽，算子测试代码设计、实现、测试与优化，文档编辑，统筹安排
  - 郭琛，算子测试代码设计、实现、测试与优化，文档编辑
  - 马瑞晗，算子测试代码设计、实现、测试与优化，文档编辑
- 联系人：郭宸羽
- 联系邮箱：287180474@qq.com

## 环境要求

- 运行环境：组委会提供的远程 Ascend 910_93 NPU 真机服务器
- CANN 版本：使用远程服务器预置 CANN 工具链版本
- ops-math 源码路径：`/root/ops-math`
- 操作系统：远程服务器预置 Linux aarch64 环境
- 目标 SoC：`ascend910_93`
- 编译器：远程服务器预置 `g++` / `cmake` / `make`
- 测试框架：基于 CANN ACL / ACLNN 端到端 example 测试流程
- 覆盖率工具：`gcov`，通过 `build.sh --cov` 生成 `.gcda` / `.gcno`
- 其他依赖：远程环境已预置 CANN 运行库、ops-math 源码、真实 NPU 运行环境和报告模板

决赛环境不使用 Docker 镜像，不使用 `--simulator` 参数。所有编译、安装、运行和覆盖率采集均在远程 Ascend 910_93 NPU 服务器上完成。

## 文件说明

- `code/`：测试代码和算子 CMake 配置
  - `code/Add/CMakeLists.txt`：Add 算子 CMake 配置，已补齐 `ascend910_93` 支持并映射到 `arch35`
  - `code/Add/test_aclnn_add.cpp`：Add 算子端到端测试代码，覆盖 `aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnAddV3`、`aclnnInplaceAddV3` 六类 API
  - `code/Cumsum/CMakeLists.txt`：Cumsum 算子 CMake 配置，已将 `ascend910_93` 对应 tiling 目录统一映射到 `arch35`
  - `code/Cumsum/test_aclnn_cumsum.cpp`：Cumsum 算子端到端测试代码，覆盖 `aclnnCumsum` 与 `aclnnCumsumV2`，包含 `exclusive`、`reverse`、正负 `dim`、多 dtype、多 shape 与异常输入场景
- `report/`：测试报告
  - `report/Add.md`：Add 算子测试报告，包含用例设计、覆盖率分析和精度分析
  - `report/Cumsum.md`：Cumsum 算子测试报告，包含用例设计、覆盖率分析和误差累积精度分析

## 编译与运行

以下命令在远程 Ascend 910_93 NPU 服务器上执行。假设提交目录位于 `/root/submission/`，其中包含本 README、`code/` 和 `report/`。

### Add 算子

```bash
cd /root/ops-math

cp /root/submission/code/Add/CMakeLists.txt math/add/CMakeLists.txt
cp /root/submission/code/Add/test_aclnn_add.cpp math/add/examples/test_aclnn_add.cpp

bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
find build -name "add_tiling*.gcno"

./build_out/cann-ops-math-custom_linux-aarch64.run
bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

### Cumsum 算子

```bash
cd /root/ops-math

cp /root/submission/code/Cumsum/CMakeLists.txt math/cumsum/CMakeLists.txt
cp /root/submission/code/Cumsum/test_aclnn_cumsum.cpp math/cumsum/examples/test_aclnn_cumsum.cpp

bash build.sh --pkg --soc=ascend910_93 --ops=cumsum --vendor_name=custom --cov
find build -name "cumsum_tiling*.gcno"

./build_out/cann-ops-math-custom_linux-aarch64.run
bash build.sh --run_example cumsum eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

## 覆盖率查看

运行测试后，可在 `/root/ops-math/build/` 下查看覆盖率产物：

```bash
cd /root/ops-math
find build -name "*.gcda" | grep -E "add|cumsum"
gcov -b <gcda文件路径>
```

`gcov` 输出中的 `Lines executed: XX.XX% of YY` 为行覆盖率，`Branches executed: XX.XX% of YY` 为分支覆盖率。决赛评分同时关注 op_api 层和 op_host 层指定文件的综合行覆盖率与综合分支覆盖率。


## 团队信息

- 团队名称：弥澄大亮
- 所属单位：闽江大学
- 团队成员：
  - 林滨炜，测试代码开发与用例设计
  - 林靖朝，测试执行与结果分析
  - 李聿钦，测试报告整理与提交材料汇总
- 联系人：林滨炜
- 联系邮箱：2965844701@qq.com

## 环境要求

- CANN 版本：9.0.0-beta.2
- 操作系统：组委会提供的远程 Ascend 910_93 NPU 服务器环境（aarch64 Linux）
- 硬件环境：Ascend 910_93 NPU
- 编译器：gcc 11
- 测试框架：自定义 C++ 测试程序
- 其他依赖：
  - AscendCL / ACLNN
  - `cann-ops-math`
  - `gcov -b`（覆盖率统计）
  - UniVPN（接入比赛内网）
  - SSH 远程登录环境

说明：本作品的编译、运行、调试与覆盖率统计均在组委会提供的远程真机环境中完成。队伍本地 WSL 环境仅用于文档整理与文件归档，不作为评测或结果复现环境。

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Add/test_aclnn_add.cpp`：Add 算子测试主程序
  - `code/Cumsum/`：Cumsum 算子测试代码
  - `code/Cumsum/test_aclnn_cumsum.cpp`：Cumsum 算子测试主程序
- `report/`：测试报告
  - `report/Add.md`：Add 算子测试报告
  - `report/Cumsum.md`：Cumsum 算子测试报告

## 编译与运行

以下以 Add 与 Cumsum 两个算子为例说明。

### Add

1. 进入对应算子目录：
   `cd code/Add`

2. 将测试文件复制到 `ops-math` 项目对应位置：
   `cp test_aclnn_add.cpp /root/ops-math/math/add/examples/test_aclnn_add.cpp`

3. 切换到 `ops-math` 项目目录：
   `cd /root/ops-math`

4. 编译前修复 Add 算子的 `CMakeLists.txt` 映射配置：
   `sed -i 's|set(SUPPORT_COMPUTE_UNIT "ascend950" "mc62cm12a")|set(SUPPORT_COMPUTE_UNIT "ascend310p" "ascend910_93" "ascend910b" "ascend950" "mc62cm12a")|; s|set(SUPPORT_TILING_DIR "arch35" "arch35")$|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' math/add/CMakeLists.txt`

5. 编译算子：
   `bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov`

说明：`--cov` 用于启用覆盖率插桩。编译成功后会在 `build_out/` 下生成算子安装包。

6. 校验 host 层覆盖率插桩文件：
   `find build -name "add_tiling*.gcno"`

7. 安装算子包：
   `./build_out/cann-ops-math-custom_linux-aarch64.run`

8. 运行测试：
   `bash build.sh --run_example add eager cust --vendor_name=custom --soc=ascend910_93 --cov`

说明：本作品在组委会提供的远程 Ascend 910_93 真机 NPU 环境中运行，不使用 `--simulator` 参数。运行成功后会在 `build/` 目录下生成覆盖率数据文件（`.gcda`）。

9. 查看覆盖率：
   `find build -name "*.gcda" | grep add`
   `gcov -b <gcda-file>`

说明：`gcov` 输出中的 `Lines executed: XX.XX% of YY` 为行覆盖率，`Branches executed: XX.XX% of YY` 为分支覆盖率。每次修改测试用例后，需重新执行上述编译、安装、运行与覆盖率统计步骤。

### Cumsum

1. 进入对应算子目录：
   `cd code/Cumsum`

2. 将测试文件复制到 `ops-math` 项目对应位置：
   `cp test_aclnn_cumsum.cpp /root/ops-math/math/cumsum/examples/test_aclnn_cumsum.cpp`

3. 切换到 `ops-math` 项目目录：
   `cd /root/ops-math`

4. 编译前修复 Cumsum 算子的 `CMakeLists.txt` 映射配置：
   `sed -i 's|set(SUPPORT_TILING_DIR "arch32" "arch32" "arch32" "arch35" "arch35")|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' math/cumsum/CMakeLists.txt`

5. 编译算子：
   `bash build.sh --pkg --opkernel --soc=ascend910_93 --ops=cumsum --vendor_name=custom_team2 --cov`

说明：编译前需根据实际环境设置依赖库路径，例如：
`export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0-beta.2/opp/vendors/custom_team2_math/op_api/lib/:$LD_LIBRARY_PATH`

6. 安装算子包：
   `bash build_out/cann-ops-math-custom_team2_linux-aarch64.run --quiet`

7. 运行测试：
   `bash build.sh --run_example cumsum eager cust`

8. 查看覆盖率：
   `find build -name "*.gcda" | grep cumsum`
   `gcov -b <gcda-file>`

说明：Cumsum 同样在组委会提供的远程 Ascend 910_93 真机 NPU 环境中运行，覆盖率数据在测试完成后于 `build/` 目录下生成。

## 说明

- 本目录仅包含源代码与测试报告，不包含 `build/`、目标文件、覆盖率产物等编译生成文件。
- 若组委会需要 PDF 版报告，可将 `report/Add.md` 与 `report/Cumsum.md` 分别导出为 PDF 后一并提交。

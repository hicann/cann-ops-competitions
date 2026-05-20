# 总决赛：算子测试用例设计

## 赛题概述

本次总决赛要求参赛者为 CANN ops-math 仓库中的算子编写端到端测试用例，**在真实昇腾 910_93 NPU 环境下**尽可能覆盖算子的各种执行路径，并**深入分析算子的精度特性**。评分采用五维综合打分，代码覆盖率仍是核心指标，同时新增精度分析与测试报告两个维度。

总决赛共设 2 道题目：

| 题目   | 算子               | 难度 | 说明                                                         |
| ------ | ------------------ | ---- | ------------------------------------------------------------ |
| 题目 1 | Add（逐元素加法）  | 进阶 | $y = x_1 + \alpha \times x_2$，含 alpha 参数和 V3 API，6 类 API 变体 |
| 题目 2 | Cumsum（累积求和） | 进阶 | $y[i] = \sum_{j=0}^{i} x[j]$，含 V2 参数（exclusive / reverse），重点关注误差累积效应 |

各题目的详细要求见对应的题目描述文档。

### 与预选赛的主要差异

| 维度     | 预选赛                        | 总决赛                                                   |
| -------- | ----------------------------- | -------------------------------------------------------- |
| 运行环境 | 本地 Docker（x86 CPU 模拟器） | **真实 Ascend 910_93 NPU**（远程服务器）                 |
| SOC 参数 | `--soc=ascend950 --simulator` | `--soc=ascend910_93`（真机）                             |
| 评分维度 | 代码覆盖率                    | **五维综合**：编译 / 行覆盖 / 分支覆盖 / 精度分析 / 报告 |
| 提交物   | 测试代码 + build 产物         | 测试代码 + build 产物 + **测试报告**                     |
| 精度分析 | 非评分项                      | **单独评分维度**，需深入分析                             |

## 参考资料

在开始之前，建议阅读以下参考文档：

- **[Mul 算子架构分析](https://gitcode.com/org/AI4SE/discussions/4)**：了解 CANN 算子的分层结构（`op_api` / `op_host` / `op_kernel`）、支持的数据类型组合以及各层中的条件分支
- **[Mul 算子测试用例分析](https://gitcode.com/org/AI4SE/discussions/5)**：了解端到端测试用例的代码结构、两段式 API 调用方法（`GetWorkspaceSize` + `Execute`）以及结果验证思路
- **[Mul 算子精度分析教程](https://gitcode.com/org/AI4SE/discussions/9 )**：浮点精度问题的典型场景、分析方法与测试设计思路（决赛精度分析维度的重要参考）

## 实验环境

本次总决赛在**远程 Ascend 910_93 NPU 服务器**上执行，组委会已预置 CANN 工具链、ops-math 源码、测试报告模板等全部依赖。

**接入方式**：

1. 安装 UniVPN 客户端并使用统一账号登录内网

2. 使用组委会邮件下发的 IP / 端口 / 密码通过 SSH 登录本队专属环境

3. 将 

   ```
   /public
   ```

    下的赛题资料拷贝至 

   ```
   /root
   ```

    作为工作目录：

   ```bash
   cp -r /public/* /root/
   ```

详细接入步骤请参见《参赛全流程说明》。

> **注意**：由于决赛环境为真机 NPU，不再使用 `--simulator` 参数，也不使用 Docker 镜像。所有编译、运行、调试均在远程服务器上完成。

## 基本操作流程

以 Add 算子为例（Cumsum 只需将命令中的 `add` 替换为 `cumsum`）：

### 0. 前置修复：CMakeLists 配置

`math/add/CMakeLists.txt` 与 `math/cumsum/CMakeLists.txt` 在默认配置下存在 `ascend910_93` SOC→arch 映射问题，会导致 **host 层 tiling 覆盖率为 0**。编译前务必先执行一键补丁：

```bash
cd /root/ops-math

# Add：补齐 SOC 列表 + 统一映射到 arch35
sed -i 's|set(SUPPORT_COMPUTE_UNIT "ascend950" "mc62cm12a")|set(SUPPORT_COMPUTE_UNIT "ascend310p" "ascend910_93" "ascend910b" "ascend950" "mc62cm12a")|;
        s|set(SUPPORT_TILING_DIR "arch35" "arch35")$|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' \
    math/add/CMakeLists.txt

# Cumsum：arch32 → arch35
sed -i 's|set(SUPPORT_TILING_DIR "arch32" "arch32" "arch32" "arch35" "arch35")|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' \
    math/cumsum/CMakeLists.txt
```

### 1. 编译算子

```bash
cd /root/ops-math
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
```

`--cov` 启用覆盖率插桩，编译成功后在 `build_out/` 下生成算子安装包。

**校验 host 层产物**（编译后必做）：

```bash
find build -name "add_tiling*.gcno"
```

应能查到文件，若为空说明前置补丁未生效，请回到步骤 0 检查。

### 2. 安装算子包

```bash
./build_out/cann-ops-math-custom_linux-aarch64.run
```

### 3. 运行测试（真实 NPU）

```bash
bash build.sh --run_example add eager cust \
    --vendor_name=custom --soc=ascend910_93 --cov
```

运行成功后会在 `build/` 目录下生成覆盖率数据文件（`.gcda`）。

### 4. 查看覆盖率

```bash
# 行覆盖
find build -name "*.gcda" | grep add
gcov -b <gcda文件路径>
```

`gcov` 输出的 `Lines executed: XX.XX% of YY` 即为行覆盖率，`Branches executed: XX.XX% of YY` 即为分支覆盖率。每次修改测试用例后，需重新执行步骤 1–4。

### 5. 撰写测试报告

参考测试报告模板和模版样例，编写报告

### 6. 打包提交

将测试代码、build 目录（仅保留 `.gcda / .gcno`）、测试报告打包为 `.zip`。具体目录结构与打包命令见各题目描述文档。

## 评分标准

决赛采用**五维综合评分**：

| 维度          | 说明                                                     | 占比倾向 |
| ------------- | -------------------------------------------------------- | -------- |
| 1. 编译通过率 | 提交代码必须能在评测环境中完整跑完编译 → 安装 → 运行流程 | 前置门槛 |
| 2. 行覆盖率   | 统计 `op_api` 层与 `op_host` 层指定文件的综合行覆盖率    | 核心指标 |
| 3. 分支覆盖率 | 统计同一批文件的综合分支覆盖率（`gcov -b`）              | 核心指标 |
| 4. 精度分析   | 测试报告中对精度问题的场景发现与原理分析深度             | 质量指标 |
| 5. 测试报告   | 报告的完整性、结构、分析质量                             | 质量指标 |

**前置条件**：

1. **编译通过**：编译失败的提交无法获得完整覆盖率得分，但评测系统会尝试从提交的 build 目录中提取覆盖率数据作为参考；
2. **结果验证**：测试代码中必须包含有效的结果验证逻辑（期望值计算 + 数值比对），仅打印结果而不验证的提交将被扣分；
3. **测试报告**：必须按模板提交，缺失报告将直接影响相关维度得分。

## 提交要求

每道题目独立提交一个压缩包 `<队名>_<题目>.zip`，基本结构：

```
<队名>/
├── test_aclnn_<op>.cpp       # 测试用例源文件（必须）
├── build/                    # 覆盖率产物（仅保留 .gcda / .gcno）
└── 测试报告.md                # 按模板编写（必须）
```

**build 目录仅保留评分相关的 `.gcda / .gcno`**，具体路径与一键筛选打包命令见各题目描述文档，**不要提交完整 build 目录**（可能有几百 MB）。

------

**祝各参赛队在决赛中取得佳绩！**
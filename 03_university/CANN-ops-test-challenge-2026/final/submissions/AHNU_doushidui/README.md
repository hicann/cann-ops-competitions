# AHNU_doushidui

## 团队信息

- 团队名称：doushidui
- 所属单位：安徽师范大学
- 团队成员：
  - 宋强强，负责 Add 算子测试用例设计、代码实现与精度分析
  - 檀焰，负责 Cumsum 算子测试用例设计、代码实现与精度分析
  - 商星宇，负责覆盖率统计、报告整理与提交打包
- 联系人：商星宇
- 联系邮箱：1277569508@qq.com

## 环境要求

本次提交面向总决赛真实 Ascend 910_93 NPU 环境，不使用 Docker，也不使用 `--simulator` 参数。

- 运行环境：组委会提供的远程 Ascend 910_93 NPU 服务器
- 工作目录：`/root/ops-math`
- 公共资料目录：`/public`
- SOC 参数：`ascend910_93`
- vendor name：`custom`
- 覆盖率工具：`gcov`
- 覆盖率插桩参数：`--cov`
- CANN 工具链：以组委会远程服务器预置环境为准
- 操作系统与编译器：以组委会远程服务器预置环境为准
- 其他依赖：无

说明：总决赛最终评测关注编译通过率、行覆盖率、分支覆盖率、精度分析和测试报告质量。测试代码中包含期望值计算和实际结果比对逻辑，避免仅打印结果而不验证。

## 文件说明

建议团队目录结构如下：

```text
AHNU_doushidui/
├── README.md
├── code/
│   ├── Add/
│   │   └── test_aclnn_add.cpp
│   └── Cumsum/
│       └── test_aclnn_cumsum.cpp
└── report/
    └── report.md
```

各文件说明：

- `code/Add/test_aclnn_add.cpp`：Add 算子端到端测试代码，覆盖 `aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnAddV3`、`aclnnInplaceAddV3` 等 6 类 API 变体。
- `code/Cumsum/test_aclnn_cumsum.cpp`：Cumsum 算子端到端测试代码，覆盖标准 Cumsum API 和 CumsumV2 API，重点测试 `exclusive`、`reverse`、负维度、空 tensor、接口非法参数和误差累积场景。
- `report/report.md`：总决赛测试总报告，包含测试设计思路、覆盖率统计方式、精度分析和问题发现情况。

如果按照总决赛题目要求分别打包，则每道题目可整理为：

```text
AHNU_doushidui_Add/
├── test_aclnn_add.cpp
├── build/
│   └── 仅保留 .gcda / .gcno
└── 测试报告.md
```

```text
AHNU_doushidui_Cumsum/
├── test_aclnn_cumsum.cpp
├── build/
│   └── 仅保留 .gcda / .gcno
└── 测试报告.md
```

## 前置准备

登录远程服务器后，先将公共资料复制到 `/root` 工作目录：

```bash
cp -r /public/* /root/
cd /root/ops-math
```

## 前置修复：CMakeLists 配置

为避免 `ascend910_93` SOC 到 tiling arch 映射异常导致 host 层 tiling 覆盖率为 0，编译前建议执行以下补丁命令。

### Add 算子补丁

```bash
cd /root/ops-math

sed -i 's|set(SUPPORT_COMPUTE_UNIT "ascend950" "mc62cm12a")|set(SUPPORT_COMPUTE_UNIT "ascend310p" "ascend910_93" "ascend910b" "ascend950" "mc62cm12a")|;
        s|set(SUPPORT_TILING_DIR "arch35" "arch35")$|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' \
    math/add/CMakeLists.txt
```

### Cumsum 算子补丁

```bash
cd /root/ops-math

sed -i 's|set(SUPPORT_TILING_DIR "arch32" "arch32" "arch32" "arch35" "arch35")|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' \
    math/cumsum/CMakeLists.txt
```

## 编译与运行

以下命令均在远程 Ascend 910_93 NPU 服务器中执行。

### 1. Add 算子

#### 编译 Add 算子并开启覆盖率

```bash
cd /root/ops-math
bash build.sh --pkg --soc=ascend910_93 --ops=add --vendor_name=custom --cov
```

#### 校验 host 层覆盖率产物

```bash
find build -name "add_tiling*.gcno"
```

如果查询结果为空，说明前置 CMakeLists 补丁可能未生效，需要重新检查补丁并重新编译。

#### 安装算子包

```bash
./build_out/cann-ops-math-custom_linux-aarch64.run
```

#### 运行 Add 测试

```bash
bash build.sh --run_example add eager cust \
    --vendor_name=custom --soc=ascend910_93 --cov
```

#### 查看 Add 覆盖率

```bash
find build -name "*.gcda" | grep add
gcov -b <gcda文件路径>
```

---

### 2. Cumsum 算子

#### 编译 Cumsum 算子并开启覆盖率

```bash
cd /root/ops-math
bash build.sh --pkg --soc=ascend910_93 --ops=cumsum --vendor_name=custom --cov
```

#### 校验 host 层覆盖率产物

```bash
find build -name "cumsum_tiling*.gcno"
```

如果查询结果为空，说明前置 CMakeLists 补丁可能未生效，需要重新检查补丁并重新编译。

#### 安装算子包

```bash
./build_out/cann-ops-math-custom_linux-aarch64.run
```

#### 运行 Cumsum 测试

```bash
bash build.sh --run_example cumsum eager cust \
    --vendor_name=custom --soc=ascend910_93 --cov
```

#### 查看 Cumsum 覆盖率

```bash
find build -name "*.gcda" | grep cumsum
gcov -b <gcda文件路径>
```

## 覆盖率统计说明

运行 `gcov -b` 后，重点记录以下两项：

```text
Lines executed: XX.XX% of YY
Branches executed: XX.XX% of YY
```

其中：

- `Lines executed`：行覆盖率
- `Branches executed`：分支覆盖率

每次修改测试用例后，建议重新执行以下流程：

1. 编译算子：`build.sh --pkg ... --cov`
2. 安装算子包：`./build_out/cann-ops-math-custom_linux-aarch64.run`
3. 运行测试：`build.sh --run_example ... --cov`
4. 统计覆盖率：`gcov -b <gcda文件路径>`

## 打包提交说明

总决赛每道题目建议独立打包。以 Add 为例：

```text
AHNU_doushidui_Add/
├── test_aclnn_add.cpp
├── build/
│   ├── xxx.gcda
│   └── xxx.gcno
└── 测试报告.md
```

以 Cumsum 为例：

```text
AHNU_doushidui_Cumsum/
├── test_aclnn_cumsum.cpp
├── build/
│   ├── xxx.gcda
│   └── xxx.gcno
└── 测试报告.md
```

注意：

1. `build/` 目录仅保留评分相关的 `.gcda` 与 `.gcno` 文件。
2. 不要提交完整 `build/` 目录，避免压缩包过大。
3. 测试报告必须保留精度分析、覆盖率统计和问题发现章节。
4. 测试代码必须包含实际结果和期望结果的比对逻辑。

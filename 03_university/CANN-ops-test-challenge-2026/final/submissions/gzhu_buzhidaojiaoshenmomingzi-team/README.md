# 决赛提交

## 团队信息

- 团队名称：不知道叫什么名字队
- 所属单位：广州大学
- 团队成员：
  - 陈慧美，队长[完成算子测试编码及测试]
  - 叶翔宇，成员[完成算子测试编码及测试]
- 联系人：陈慧美
- 联系邮箱：2685796099@qq.com

## 环境要求

- CANN 版本：8.0.RC1
- 运行环境：远程 Ascend 910_93 NPU 服务器（真机，非模拟器）
- 操作系统：Linux aarch64
- 编译器：g++（组委会预置）
- 测试框架：GoogleTest（组委会预置）
- 其他依赖：组委会预置 CANN 工具链、ops-math 源码、测试报告模板

> 注意：决赛环境为真实 NPU，不使用 `--simulator` 参数，也不使用 Docker 镜像。所有编译、运行、调试均在远程服务器上完成。

## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码（`test_aclnn_add.cpp`）
    - 逐元素加法：`y = x1 + alpha * x2`，含 alpha 参数和 V3 API，6 类 API 变体
  - `code/Cumsum/`：Cumsum 算子测试代码（`test_aclnn_cumsum.cpp`）
    - 累积求和：`y[i] = sum(x[0..i])`，含 V2 参数（exclusive / reverse），重点关注误差累积效应
- `report/`：测试报告
  - `report/Add.md`：Add 算子测试报告
  - `report/Cumsum.md`：Cumsum 算子测试报告

## 编译与运行

### 0. 前置修复：CMakeLists 配置

编译前必须先修复 SOC→arch 映射问题，否则 host 层 tiling 覆盖率为 0：

```bash
cd /root/ops-math

# Add：补齐 SOC 列表 + 统一映射到 arch35
sed -i 's|set(SUPPORT_COMPUTE_UNIT "ascend950" "mc62cm12a")|set(SUPPORT_COMPUTE_UNIT "ascend310p" "ascend910_93" "ascend910b" "ascend950" "mc62cm12a")|;\
        s|set(SUPPORT_TILING_DIR "arch35" "arch35")$|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' \
    math/add/CMakeLists.txt

# Cumsum：arch32 → arch35
sed -i 's|set(SUPPORT_TILING_DIR "arch32" "arch32" "arch32" "arch35" "arch35")|set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35" "arch35")|' \
    math/cumsum/CMakeLists.txt
```

### 1. 编译算子

```bash
cd /root/ops-math
bash build.sh --pkg --soc=ascend910_93 --ops=<op> --vendor_name=custom --cov
```

编译后校验 host 层产物：`find build -name "<op>_tiling*.gcno"`，若为空说明补丁未生效。

### 2. 安装算子包

```bash
./build_out/cann-ops-math-custom_linux-aarch64.run
```

### 3. 运行测试（真实 NPU）

```bash
bash build.sh --run_example <op> eager cust \
    --vendor_name=custom --soc=ascend910_93 --cov
```

### 4. 查看覆盖率

```bash
find build -name "*.gcda" | grep <op>
gcov -b <gcda文件路径>
```

其中 `<op>` 替换为 `add` 或 `cumsum`。每次修改测试用例后需重新执行步骤 0–4。

## 评分维度

| 维度 | 说明 | 占比倾向 |
|------|------|----------|
| 编译通过率 | 提交代码必须能完整跑完编译→安装→运行流程 | 前置门槛 |
| 行覆盖率 | op_api 层与 op_host 层指定文件的综合行覆盖率 | 核心指标 |
| 分支覆盖率 | 同一批文件的综合分支覆盖率（`gcov -b`） | 核心指标 |
| 精度分析 | 测试报告中对精度问题的场景发现与原理分析深度 | 质量指标 |
| 测试报告 | 报告的完整性、结构、分析质量 | 质量指标 |

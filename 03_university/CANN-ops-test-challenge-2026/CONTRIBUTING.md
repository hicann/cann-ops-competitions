# 作品提交规范

本规范适用于 `preliminary/submissions/` 与 `final/submissions/` 目录下的所有团队提交。请各参赛队伍严格按照本规范组织提交内容，不符合规范的提交可能导致 PR 无法合并，进而影响后续奖金与证书的发放。

## 一、目录命名规范

在对应赛事阶段的 `submissions/` 目录下创建团队目录：

```
{submissions}/{school}_{team-name}/
```

**命名规则：**

- `school`：学校代码缩写（如 `nju`、`seu`、`hitsz`、`tju`）
- `team-name`：团队自定名称，使用短横线分隔（如 `op-pioneers`），若是中文则可以使用拼音代替

**示例：**

```
preliminary/submissions/nju_op-pioneers/
preliminary/submissions/hitsz_test-masters/
final/submissions/nju_op-pioneers/
```

> 注意：文件与文件夹命名请使用英文，可以使用拼音来进行代替

预选赛与决赛的提交目录相互独立，晋级决赛的队伍需在 `final/submissions/` 下另建目录。

## 二、目录结构要求

每个团队的提交目录**仅需包含以下三部分**：

```
team01_nju_op-pioneers/
├── README.md       # 必选：团队信息、环境要求、文件说明
├── code/           # 必选：测试代码（按算子组织）
└── report/         # 必选：测试报告
```

### 1. code/（测试代码）

按算子建立子目录，每个子目录包含针对该算子的测试代码源文件。预选赛与决赛的算子不同，请按当前阶段的赛题组织。

**预选赛示例：**

```
code/
├── Add/
│   └── test_aclnn_add.cpp
├── Mul/
│   └── test_aclnn_mul.cpp
└── Pow/
    └── test_aclnn_pow.cpp
```

**决赛示例：**

```
code/
├── Add/
│   └── test_aclnn_add.cpp
└── Cumsum/
    └── test_aclnn_cumsum.cpp
```

**说明：**

- 仅提交源代码与必要的构建脚本（如 `CMakeLists.txt`），不提交 `build/` 目录、目标文件、覆盖率数据（`.gcda` / `.gcno`）等编译产物
- 评测时由组委会按统一环境重新编译运行
- 若有多个测试文件，请保持目录扁平、命名清晰

### 2. report/（测试报告）

测试报告用于说明测试设计思路、覆盖率统计与问题发现情况。格式与组织方式：

- 接受 PDF 或 Markdown 格式
- 可按算子分别提交（如 `report/Add.pdf`、`report/Mul.pdf`），也可合并为一份总报告（如 `report/report.pdf`）
- 若包含图片附件，统一放在 `report/assets/` 下

### 3. README.md（团队说明）

格式与内容详见下一节。

## 三、README.md 内容规范

每个提交目录下的 `README.md` 必须包含以下三个部分：

### 1. 团队信息

```markdown
## 团队信息

- 团队名称：[团队名称]
- 所属单位：[学校全称]
- 团队成员：
  - [姓名]，[在团队中的分工]
  - [姓名]，[在团队中的分工]
- 联系人：[姓名]
- 联系邮箱：[邮箱地址]
```

### 2. 环境要求

```markdown
## 环境要求

- CANN 版本：[如 8.0.RC1]
- 操作系统：[如 Ubuntu 20.04 x86_64]
- 编译器：[如 g++ 9.4.0]
- 测试框架：[如 GoogleTest 1.12.1]
- 其他依赖：[如有，逐项列出]
```

如使用大赛提供的 Docker 镜像（`yeren666/cann-ops-test:v1.0`），请在此处注明并标注镜像版本。

### 3. 文件说明

```markdown
## 文件说明

- `code/`：测试代码源文件，按算子分子目录组织
  - `code/Add/`：Add 算子测试代码
  - `code/Mul/`：Mul 算子测试代码
  - ...
- `report/`：测试报告
  - `report/report.pdf`：测试报告主文档

## 编译与运行

[简要说明如何编译运行测试代码，例如：]

1. 进入对应算子目录：`cd code/Add`
2. 编译：`mkdir build && cd build && cmake .. && make`
3. 运行：`./test_aclnn_add`
```

## 四、注意事项

1. **代码原创性**：提交内容须为团队原创，禁止抄袭其他团队作品或网络代码片段
2. **不提交编译产物**：`build/`、`*.o`、`*.gcda`、`*.gcno` 等由评测系统重新生成，无需提交
3. **不提交敏感信息**：包括但不限于密钥、密码、个人证件、内部资料等
4. **文件大小限制**：单个文件不超过 50MB，单个团队目录总大小不超过 200MB
5. **避免覆盖他人文件**：仅修改自己团队目录下的内容，不得改动其他团队、赛题文档或仓库配置文件

## 五、提交流程

通过 fork + pull request 完成提交：

1. fork 大赛仓库到个人 GitCode 账号
2. 将自己的 fork clone 到本地
3. 在 `preliminary/submissions/` 或 `final/submissions/` 下创建团队目录
4. 按本规范添加 README、code、report 三部分内容
5. commit 并 push 到自己的 fork
6. 向大赛主仓库发起 pull request，标题格式：`[团队提交] school_team-name`

PR 合并后即视为提交成功。组委会将基于已合并的 PR 安排后续奖金与证书发放。

## 六、问题反馈

提交过程中如遇问题，请通过以下渠道联系组委会：

- GitCode 讨论区：https://gitcode.com/org/AI4SE/discussions/
- GitCode Issue：在大赛仓库下提交 Issue
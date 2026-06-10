# CANN 社区任务 2026

## 任务介绍
CANN 社区任务 2026 是由 CANN 开源社区发起的年度算子开发任务，面向所有开源贡献者，旨在推动昇腾 AI 生态建设，鼓励开发者参与算子开发与贡献。

## 参考链接
- **社区任务讨论**：https://gitcode.com/org/cann/discussions/22
- **目录结构参考**：https://gitcode.com/cann/ops-math/blob/master/docs/zh/install/dir_structure.md

## 目录结构
```
01_community-task-2026/
├── README.md                              # 本文档
├── docs/                                  # 任务书目录
│   └── README.md                          # 任务列表说明
├── resources/                             # 开发资源
└── tasklist/                              # 任务列表
```

### 算子任务目录结构
每个算子任务由贡献者创建，目录结构如下：
```
{算子任务编号}-{算子名称}/
└── {TeamName}/                           # 团队提交目录（实际团队名）
    ├── docs/                             # 文档目录
    │   ├── aclnn{OpName}.md              # API接口文档
    │   └── design.md                     # 设计文档
    ├── examples/                         # 示例代码
    ├── op_api/                           # API实现
    ├── op_host/                          # Host端实现
    ├── op_kernel/                        # Kernel实现
    ├── tests/                            # 测试用例
    ├── CMakeLists.txt                    # 构建配置
    └── README.md                         # 算子说明
```

## 任务列表
参考 `docs/` 目录下的 README.md 文件，选择感兴趣的算子任务认领。

## 参与方式

### 1. 认领任务
在社区任务讨论区认领感兴趣的算子任务：https://gitcode.com/org/cann/discussions/22

### 2. 创建提交目录
以您的团队名称`TeamName` 为名创建提交目录，目录结构参考[算子仓目录](https://gitcode.com/cann/ops-math/blob/master/docs/zh/install/dir_structure.md)的算子目录部分。

### 3. 输出设计文档
参考[算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)，完成算子设计文档。
提交设计文档到对应算子目录的 `docs/` 下，文件名为 `design.md`，评审并合入。

### 4. 开发算子
按照目录结构要求开发算子：
- `docs/`：API 文档
- `examples/`：调用样例
- `op_api/`：API 接口实现
- `op_host/`：Host 端实现
- `op_kernel/`：Kernel 实现
- `tests/`：测试用例

### 4. 提交作品
完成开发后，提交 Pull Request 到任务书对应仓。

## 提交规范

### 目录结构要求
每个算子提交目录必须包含：
```
YourTeamName/
├── docs/
│   └── design.md           # 设计文档（必选）
```
如任务完成，目录结构需包含：
```
YourTeamName/
├── docs/
│   ├── aclnn{OpName}.md    # API接口文档（必选）
│   └── design.md           # 设计文档（必选）
├── examples/               # 调用示例代码（必选）
├── op_api/                 # API实现（可选）
├── op_host/                # Host端实现（必选）
├── op_kernel/              # Kernel实现（必选）
├── tests/                  # 测试用例（可选）
├── CMakeLists.txt          # 构建配置（必选）
└── README.md               # 算子说明（必选）
```

### 文档要求
- API 文档需包含函数原型、参数说明、返回值、约束限制、示例代码
- 设计文档需包含算子功能、设计思路、性能优化方案

### 代码要求
- 遵循 Ascend C 编码规范
- 提供完整的测试用例
- 确保精度和性能达标

## 联系方式
- **官方邮箱**：community@cann-community.com
- **技术支持**：support@cann-community.com
- **Issue 反馈**：https://gitcode.com/cann/cann-competitions/issues

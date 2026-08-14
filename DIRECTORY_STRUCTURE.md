# CANN 开源社区竞赛作品仓库目录结构

## 📁 顶层目录结构
```
cann-competitions/
├── README.md               # 仓库说明、提交指南、目录导航
├── LICENSE                 # 开源协议
├── CONTRIBUTING.md         # 作品提交规范
├── DIRECTORY_STRUCTURE.md  # 目录结构说明（本文档）
├── official/               # 官方赛事
├── university/             # 高校赛事
├── enterprise/             # 区域/行业赛事
├── tasks/                  # 开源课题/社区任务
```

## 🏛️ 官方赛事目录 (official/)
```
official/
├── cann-challenge-2025/     # CANN 全国挑战赛 2025
│   ├── README.md           # 赛事说明文档
│   ├── docs/               # 赛题文档
│   ├── submissions/        # 作品提交目录
│   └── resources/          # 开发资源
├── operator-tianti-2025/    # 算子天梯赛 2025
```

## 🎓 高校赛事目录 (university/)
```
university/
├── nju/                    # 南京大学
│   └── nju-ai-challenge-2025/  # 南大 AI 挑战赛 2025
│       ├── README.md       # 赛事说明
│       ├── docs/           # 赛题文档
│       ├── submissions/    # 作品提交
│       └── resources/      # 开发资源
├── seu/                    # 东南大学
│   └── seu-hpc-contest-2025/   # 东大 HPC 竞赛 2025
├── zju/                    # 浙江大学
│   └── zju-innovation-cup-2025/ # 浙大创新杯 2025
├── sjtu/                   # 上海交通大学
├── fudan/                  # 复旦大学
├── ustc/                   # 中国科学技术大学
├── buaa/                   # 北京航空航天大学
└── bupt/                   # 北京邮电大学
```

## 💼 行业/区域赛事目录 (enterprise/)
```
enterprise/
├── regional/              # 区域赛
│   └── ai-optimization-contest-2025/  # AI 优化挑战赛 2025
│       ├── README.md       # 赛事说明
│       ├── docs/           # 赛题文档
│       ├── submissions/    # 作品提交
│       └── resources/      # 开发资源
└── industry/       # 行业赛
```

## 💼 开源课题/社区任务目录 (tasks/)
```
tasks/
├── 01_community-task-2026/  # 2026年社区任务
│   └── README.md           # 任务介绍
│   └── docs/               # 任务说明书文档
│   └── resources/          # 任务相关资源
│   └── submissions/        # 任务提交

```

## 📝 作品提交目录结构
```
{submissions}/{team-id}_{school}_{work-title}/
├── README.md          # 作品详细说明文档
├── code/              # 源代码目录
│   ├── src/           # 核心源码
│   ├── scripts/       # 脚本文件
│   └── requirements.txt  # 依赖说明
├── model/             # 模型文件（可选）
├── results/           # 实验结果（可选）
└── docs/              # 相关文档（可选）
```

## 🌟 示例作品展示

### 官方赛事示例
- **赛事**: CANN 全国挑战赛 2025
- **作品**: 高性能卷积算子优化
- **团队**: AI Optimizers (南京大学)
- **路径**: `official/cann-challenge-2025/submissions/team01_nju_convolution-optimization/`

### 高校赛事示例
- **赛事**: 南京大学 AI 挑战赛 2025
- **作品**: AI 智能问答教学系统
- **团队**: NLP Pioneers (南京大学)
- **路径**: `university/nju/nju-ai-challenge-2025/submissions/team01_cs_nju_nlp-application/`

### 行业/区域赛事示例
- **赛事**: Regional-A AI 优化挑战赛 2025
- **作品**: 大模型知识蒸馏优化
- **团队**: Model Compressors (清华大学)
- **路径**: `enterprise/regional/ai-optimization-contest-2025/submissions/team01_company_a_model-optimization/`

## 🚀 使用指南

### 1. 查找赛事
根据赛事类型和年份在对应目录下查找：
- 官方赛事：`official/{赛事名称}-{年份}/`
- 高校赛事：`university/{学校代码}/{赛事名称}-{年份}/`
- 行业/区域赛事：`enterprise/industry/{赛事名称}-{年份}/`、`enterprise/regional/{赛事名称}-{年份}/`

### 2. 提交作品
在对应赛事的 `submissions/` 目录下创建团队目录，按照统一的命名规范和目录结构提交作品。

### 3. 查找资源
赛事相关的数据集、基准代码、工具等资源存放在各赛事目录下的 `resources/` 目录中。

## 🔧 维护说明
1. **新增赛事**: 在对应类型目录下创建新的赛事目录
2. **目录清理**: 定期清理过期或无效的目录和文件
3. **结构更新**: 根据实际需求动态调整目录结构

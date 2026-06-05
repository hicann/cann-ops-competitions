# submit_to_gitcode.sh

CANNJudge 算子代码一键提交脚本。

**功能**：从 CANNJudge 下载已通过的算子代码 → 组织竞赛目录结构 → Fork 竞赛仓 → 提交 PR → 评论 `/compile` 触发编译流水线。

## 前置条件

| 依赖 | 用途 | 检查命令 |
|------|------|----------|
| bash 4+ | 运行环境 | `bash --version` |
| curl | API 调用 | `curl --version` |
| jq | JSON 解析 + URL 编码 | `jq --version` |
| git | 仓库操作 | `git --version` |
| python3 + requests | CANNJudge 代码下载 | `python3 -c "import requests"` |

## 快速开始

```bash
chmod +x submit_to_gitcode.sh
./submit_to_gitcode.sh
```

脚本以交互式问答运行，每项输入均带即时验证，输错可重试。

## 执行步骤

### Step 1 — 交互式输入 + 即时验证

脚本依次收集以下信息，每项验证通过才继续：

| 序号 | 输入项 | 示例值 | 验证逻辑 |
|------|--------|--------|----------|
| 1 | CANNJudge 邮箱 | `user@example.com` | 登录 API 验证 |
| 2 | CANNJudge 密码 | `******`（不回显） | 登录 API 验证 |
| 3 | 算子题目地址 | `https://cannjudge.cn/public/op_challenge_jiangshan_final/batch_to_space` | 查询存在性 + 按 problem ID 精确匹配最新 Pass 提交 |
| 4 | GitCode 用户名 | `zhangsan` | API 认证 + 令牌用户一致性（不一致直接退出） |
| 5 | GitCode 访问令牌 | `******`（不回显） | 建议具备 repo + write_repository 权限（脚本不预检，权限不足在后续操作时报错） |
| 6 | 用户邮箱 | `user@example.com` | 格式校验（此邮箱为 CLA 签署邮箱，用于 git commit） |
| 7 | CLA 签署确认 | `y/n` | 必须确认已签署 CLA 才能继续 |
| 8 | 目标仓库 URL | 见下方示例 | 验证仓库名 + 路径末级为 `submissions` + 路径存在性 |
| 9 | 比赛名称 | `算子挑战赛江山赛区预赛` | 手动输入，用于 PR 标题 |
| 10 | 学校代码缩写 | `nju` | 以字母或数字开头，仅允许字母/数字/短横线/下划线 |
| 11 | 队伍名称 | `op-pioneers` | 以字母或数字开头，仅允许字母/数字/短横线/下划线 |
| 12 | 联系人姓名 | `张三` | 自动作为第一个团队成员 |
| 13 | 所属单位 | `南京大学` | 学校全称 |
| 14 | 你的分工 | `算子实现` | 可选，默认"负责人" |
| 15 | 其他成员 | 姓名/GitCode用户名/邮箱/分工/CLA确认 | 输入 n 结束；每人验证 GitCode 用户存在性 + CLA 签署确认；未确认 CLA 则跳过该成员 |
| 16 | 算子实现介绍 | 多行文本，输入 `END` 结束 | 若 `$EDITOR` 已设置则打开编辑器输入（`#` 开头行自动过滤）；否则用 END 模式 |

**目标仓库 URL 示例**：
```
https://gitcode.com/cann/cann-ops-competitions/tree/master/03_university/CANN-ops-jiangshan-challenge-2026/preliminary/submissions
```

**邮箱一致性**：输入 CLA 邮箱后，若与 GitCode 默认邮箱不一致，脚本仅提示建议手动修改 GitCode 默认邮箱。脚本会将 `git config user.email` 设为 CLA 邮箱，确保 commit 通过 CLA 校验。通过网页创建 PR 时 CLA 校验可能失败（网页操作使用 GitCode 默认邮箱）。

**令牌用户一致性**：令牌所属用户必须与输入的 GitCode 用户名一致，不一致直接退出。

### Step 2 — 从 CANNJudge 下载算子代码

- 重新登录 CANNJudge（复用 Step 1 的账号）
- 分页遍历用户提交列表（每页 50 条，`skip/limit` 分页），按 `problem._id` 精确匹配目标题目
- 找到第一条 Pass 提交即停止（API 按时间倒序，第一条即最新）
- 下载该提交中的全部文件到临时目录（自动排除 `_meta.json` 和 `legacy_*` 前缀文件）
- Python 脚本区分错误类型（登录失败/无 Pass/无权限/超时/网络错误），返回不同退出码
- 路径穿越检查：验证下载文件路径不超出目标目录

### Step 3 — Fork 竞赛仓

- 检查 fork 是否已存在 → 存在则复用
- 不存在则调用 API 创建 fork，轮询等待就绪（最多 60 秒）

### Step 4 — 克隆仓库 + 同步上游

- 完整克隆 fork（非浅克隆，避免 merge 冲突）
- 添加 upstream 远程，fetch + merge 上游最新代码
  - 快进合并优先；分叉时可选创建合并提交；冲突未解决时报错退出
- 创建 PR 分支 `submit/{school}_{team}/{op_name}`，基于 `upstream/master` 创建（确保 PR 只包含本次提交，不受 fork master 上其他 commit 影响）
  - 分支已存在 → 确认后重置到上游最新（确保每次提交都是干净的单 commit）
- 配置 git user.name / user.email（使用 CLA 签署邮箱）

### Step 5 — 组织提交目录

将下载的代码复制到竞赛仓要求的目录结构：

```
submissions/{school}_{team}/
├── README.md
└── code/{op_name}/
    └── ... (下载的全部文件，保持原始目录结构)
```

- `{school}_{team}` 如 `nju_op-pioneers`
- `{op_name}` 即题目名全称（如 `batch_to_space`）
- `code/{op_name}/` 下保留 CANNJudge 下载的原始目录结构，不限制特定格式（可能包含 op_kernel/、op_host/、op_api/ 等，或非 Ascend C 代码）
- 若目标目录已存在，需确认覆盖后才继续

### Step 6 — 生成 README.md

自动生成包含团队信息 + 算子信息的 README.md。

### Step 7 — Commit + Push

- commit 消息：`submit {op_name} operator for {school}_{team}`
- 所有成员的 `Co-authored-by: {gcuser} <{email}>` 合并为单个段落追加
- 若无变更需要提交（目录已存在且内容相同），询问是否继续推送
- `--force-with-lease` 推送到 fork 的 PR 分支（安全覆盖，不会误推他人提交）

### Step 8 — 创建 PR

- 查询已有 open PR（按分支名 + head 用户过滤，支持分页）
- 已有 PR → PATCH 更新标题和描述
- 无已有 PR → POST 创建新 PR
  - 创建冲突（400/409）→ 从已有列表中复用
- PR 标题：`[团队提交] {比赛名称} {op_name} 算子提交：{team_name}`
- PR 正文：团队信息 + 算子信息 + 算子实现介绍（与 README.md 结构一致）

### Step 9 — 触发编译流水线

- 自动评论 `/compile` 触发编译流水线
  - 评论失败时提示手动在 PR 页面评论 `/compile`
- CLA 检查由平台自动触发，无需手动操作

## 完整运行示例

以下模拟南京大学 op-pioneers 队提交 `batch_to_space` 算子：

```
$ ./submit_to_gitcode.sh

===== 输入并验证信息 =====

CANNJudge 邮箱: zhangsan@nju.edu.cn
CANNJudge 密码: 
[INFO] CANNJudge 登录成功! 用户: 张三

CANNJudge 题目地址 (如 https://cannjudge.cn/public/op_challenge_jiangshan_final/batch_to_space): https://cannjudge.cn/public/op_challenge_jiangshan_final/batch_to_space
[INFO] 题目: batch_to_space | 最新 Pass: 2026-04-25 18:41:56
[INFO] 算子名: batch_to_space

GitCode 用户名: zhangsan_gc
  访问令牌获取: GitCode → 设置 → 访问令牌 → 新建令牌 (勾选全部权限)
GitCode 访问令牌: 
[INFO] GitCode 认证成功! 用户: zhangsan_gc, 默认邮箱: zhangsan@gmail.com

用户邮箱 (CLA 签署邮箱): zhangsan@nju.edu.cn
[NOTE] GitCode 默认邮箱 zhangsan@gmail.com 与 CLA 邮箱 zhangsan@nju.edu.cn 不一致，通过网页创建 PR 时 CLA 校验会失败（网页操作使用默认邮箱），建议到 GitCode 设置页面修改默认邮箱为 zhangsan@nju.edu.cn

[NOTE] 竞赛仓要求所有队员签署 CLA 才能提交 PR
  签署链接: https://clasign.osinfra.cn/sign-cla/68cbd4a3dbabc050b436cdd4/individual
确认你自己 (GitCode: zhangsan_gc, 邮箱: zhangsan@nju.edu.cn) 已签署 CLA (y/n): y

目标仓库地址 (如 https://gitcode.com/.../submissions): https://gitcode.com/cann/cann-ops-competitions/tree/master/03_university/CANN-ops-jiangshan-challenge-2026/preliminary/submissions
[INFO] 竞赛仓验证通过: 路径: 03_university/CANN-ops-jiangshan-challenge-2026/preliminary/submissions

比赛名称 (如 算子挑战赛江山赛区预赛): 算子挑战赛江山赛区预赛

学校代码缩写 (如 nju, seu, hitsz): nju
队伍名称 (短横线分隔，如 op-pioneers): op-pioneers
[INFO] 提交目录: 03_university/CANN-ops-jiangshan-challenge-2026/preliminary/submissions/nju_op-pioneers/

联系人姓名: 张三
所属单位 (学校全称): 南京大学

你的分工 (如 算子实现): 算子实现
[INFO] 已添加成员: 张三 — 你

[INFO] 其他团队成员
添加新成员? (y/n): y
  姓名: 李四
  GitCode 用户名: lisi_gc
[INFO] 验证 GitCode 用户 lisi_gc...
[INFO] 用户验证通过: lisi_gc
  请输入该成员签署 CLA 时使用的邮箱（必须与 GitCode 默认邮箱一致，否则 PR 将无法合入）
  邮箱: lisi@nju.edu.cn
  分工 (可选): Tiling设计
[NOTE] 该队员是否已签署 CLA? 签署链接: https://clasign.osinfra.cn/sign-cla/68cbd4a3dbabc050b436cdd4/individual
已签署? (y/n): y
[INFO] 已添加成员: 李四
添加新成员? (y/n): n

[INFO] 算子实现介绍 — 用于 PR 描述，多行输入，单独输入 END 结束
> 使用单核实现，UB切分按32B对齐
> Tiling策略：totalLength对齐到BLOCK_SIZE的整数倍
> END


===== 信息汇总 =====
[INFO] 算子: batch_to_space
[INFO] 比赛: 算子挑战赛江山赛区预赛
[INFO] 队伍: nju_op-pioneers — 南京大学
[INFO] 目录: 03_university/.../submissions/nju_op-pioneers/
[INFO] PR 标题: [团队提交] 算子挑战赛江山赛区预赛 batch_to_space 算子提交：op-pioneers
[INFO] 成员数: 2
确认以上信息无误，开始提交 (y/n): y

===== 从 CANNJudge 下载算子代码 =====
最新Pass: 67890abcdef
已保存8个文件
[INFO] 下载完成 — 8 个文件:
  op_kernel/batch_to_space_kernel.h
  op_kernel/batch_to_space_kernel.cpp
  op_host/batch_to_space_tiling.h
  op_host/batch_to_space_tiling.cpp
  op_host/batch_to_space.cpp
  CMakeLists.txt
  build.sh
  README.md

===== Fork 仓库 =====
[INFO] 检查 fork: zhangsan_gc/cann-ops-competitions
[INFO] 创建 fork...
[INFO] Fork 创建成功，等待就绪...
  等待 fork 就绪... 4s

===== 克隆仓库 =====
[INFO] 克隆 fork...
[INFO] 同步上游最新代码...
[INFO] 创建分支: submit/nju_op-pioneers/batch_to_space

===== 组织提交目录 =====
[INFO] 目标结构: 03_university/.../submissions/nju_op-pioneers/
  ├── README.md
  └── code/
      └── batch_to_space/
[INFO] 已复制 8 个文件
[INFO] 代码目录结构:
  ./op_kernel/batch_to_space_kernel.h
  ./op_kernel/batch_to_space_kernel.cpp
  ./op_host/batch_to_space_tiling.h
  ./op_host/batch_to_space_tiling.cpp
  ./op_host/batch_to_space.cpp
  ./CMakeLists.txt

===== 生成 README.md =====
[INFO] README.md 已生成

===== 提交代码 =====
[INFO] 推送到 fork...

===== 创建 Pull Request =====
[INFO] PR 创建成功! #42: https://gitcode.com/cann/cann-ops-competitions/pulls/42

===== 触发编译流水线 =====
[INFO] 已评论 /compile，编译流水线已触发

============================================
[INFO] 提交完成!
[INFO]   算子: batch_to_space
[INFO]   队伍: nju_op-pioneers
[INFO]   目录: 03_university/.../submissions/nju_op-pioneers/
[INFO]   PR: #42 https://gitcode.com/cann/cann-ops-competitions/pulls/42
============================================

[NOTE] 后续步骤: 打开 PR 页面确认 CLA 检查和编译流水线通过: https://gitcode.com/cann/cann-ops-competitions/pulls/42

[NOTE] 如果 CLA 显示为 no，请确认所有团队成员的以下三者一致:
  - CLA 签署时使用的邮箱
  - GitCode 账号的默认邮箱
  - git commit 中的作者邮箱
  不一致时请修改 GitCode 默认邮箱后重新签署 CLA
  签署链接: https://clasign.osinfra.cn/sign-cla/68cbd4a3dbabc050b436cdd4/individual
```

### 生成的 README.md

```markdown
## 团队信息

- 团队名称：op-pioneers
- 所属单位：南京大学
- 团队成员：
  - 张三，算子实现
  - 李四，Tiling设计
- 联系人：张三
- 联系邮箱：zhangsan@nju.edu.cn

## 算子信息

- 算子名称：batch_to_space
- CANNJudge 题目：batch_to_space
```

### git commit 内容

```
submit batch_to_space operator for nju_op-pioneers

Co-authored-by: zhangsan_gc <zhangsan@nju.edu.cn>
Co-authored-by: lisi_gc <lisi@nju.edu.cn>
```

### PR 标题

```
[团队提交] 算子挑战赛江山赛区预赛 batch_to_space 算子提交：op-pioneers
```

### PR 正文

```markdown
## 团队信息
- 团队名称：op-pioneers
- 所属单位：南京大学
- 团队成员：
  - 张三，算子实现
  - 李四，Tiling设计
## 算子信息
- 算子名称：batch_to_space
- CANNJudge 题目：batch_to_space

## 算子实现介绍
使用单核实现，UB切分按32B对齐
Tiling策略：totalLength对齐到BLOCK_SIZE的整数倍
```

## 多算子提交

每个算子独立分支/PR，互不影响：

- 算子 `batch_to_space` → 分支 `submit/nju_op-pioneers/batch_to_space` → PR #1
- 算子 `erf` → 分支 `submit/nju_op-pioneers/erf` → PR #2

重新提交同一算子时，分支自动重置为上游最新，PR 更新为单个干净 commit。

## PR 创建后自动操作

脚本创建 PR 后会自动：
1. 评论 `/compile` — 触发编译流水线
2. CLA 检查由平台自动触发，无需手动评论

用户只需打开 PR 页面确认 CLA 检查和编译流水线均通过即可。

## CLA 签署

所有团队成员必须签署 CLA：

https://clasign.osinfra.cn/sign-cla/68cbd4a3dbabc050b436cdd4/individual

> **关键**：CLA 校验以邮箱为准，以下三者必须一致：
> - CLA 签署时使用的邮箱
> - GitCode 账号的默认邮箱
> - git commit 中的作者邮箱
>
> 不一致时请修改 GitCode 默认邮箱后重新签署。
>
> 脚本会将 `git config user.email` 设为 CLA 签署邮箱，确保 commit 通过 CLA 校验。但通过网页创建 PR 时会使用 GitCode 默认邮箱，若默认邮箱与 CLA 邮箱不一致，网页端 CLA 校验会失败。

## 安全设计

| 措施 | 说明 |
|------|------|
| `curl --config` 文件 | API Token 通过临时配置文件注入（chmod 600），不暴露于进程表 `/proc/*/cmdline` |
| `oauth2:TOKEN@` URL 直传 | git clone/push 通过 URL 内嵌令牌认证（clone 后立即 set-url 移除 .git/config 中的令牌），解决 credential store 对部分用户 push 403 的问题 |
| `jq` 构造 JSON | 避免变量直接拼入 JSON 字符串，防止注入 |
| `printf` 替代 heredoc | README/PR Body 用 `printf '%s\n'` 生成，防止变量值中 `$()`/`` ` `` 被命令替换执行 |
| `printf -v` 替代 `eval` | `read_secret` 用 `printf -v` 赋值，避免命令注入 |
| `-d @file` 替代 `-d $body` | curl POST 数据写入临时文件后用 `@file` 引用，避免凭证暴露于进程表 |
| `|| true` 保护 | `set -e` 下所有可能失败的命令均加保护，避免误退出 |
| cleanup trap | EXIT/INT/TERM/HUP 时清理临时目录 + 移除 git config 中的凭证残留 + 删除 curl 配置文件 |
| `--force-with-lease` | push 时安全覆盖，不会误推他人提交 |
| 输入校验 | 算子名禁止路径穿越，学校/队名限制安全字符集，成员字段禁止 `\|` 分隔符 |
| 令牌用户一致性 | 令牌所属用户必须与输入用户名一致，不一致直接退出 |
| PR 用户过滤 | 匹配已有 PR 时同时检查 head 用户，避免误用其他账号的 PR |
| `unset` 清除敏感变量 | 下载后 `unset CJ_PASSWORD`，脚本结束前 `unset GC_TOKEN`，防止环境泄露 |
| push 后删除凭证文件 | push 完成后立即 `rm -f $GIT_CRED_FILE`，缩短凭证文件存活时间 |
| API 重试 | GitCode API 调用内置重试：最多 3 次，间隔 2 秒，覆盖 5xx 服务端错误 |
| Git 操作超时 | clone/fetch/push 均有 300 秒超时保护，避免网络异常时无限挂起 |

## 常见问题

**Q: CLA 显示 no？**
A: 检查上述三者邮箱是否一致。脚本已自动将 commit 邮箱设为 CLA 签署邮箱，并会检查 GitCode 默认邮箱一致性。

**Q: 令牌用户与输入不一致？**
A: 脚本直接退出。请确保访问令牌属于输入的 GitCode 用户。获取路径：GitCode → 设置 → 访问令牌 → 新建令牌（勾选全部权限）。

**Q: Fork 已存在？**
A: 脚本自动复用已有 fork 并同步上游代码，无需手动处理。

**Q: 路径不存在？**
A: 脚本通过 API 验证目标路径在仓库分支上是否存在，不存在则要求重新输入 URL。

**Q: 算子题目没有 Pass 提交？**
A: 脚本会警告，确认后仍可继续（但下载步骤会失败）。

**Q: 团队成员为空？**
A: 可以不添加成员（直接按回车跳过），commit 只有作者本人，无 Co-authored-by。

**Q: 重新运行会怎样？**
A: 脚本所有操作在临时目录进行，退出后自动清理。重新运行从头开始，fork 和 PR 分支如已存在会复用/重置。

**Q: 已有 PR 冲突？**
A: 脚本按分支名 + head 用户匹配已有 PR。同分支已有 PR 则更新，不同用户的 PR 不会误匹配。创建冲突时自动从已有列表中复用。

**Q: 推送 403？**
A: 检查访问令牌是否有 `write_repository` 权限。获取路径：GitCode → 设置 → 访问令牌 → 新建令牌（勾选全部权限）。

**Q: 上游合并冲突？**
A: 脚本优先快进合并。若 fork 与上游分叉，会提示创建合并提交；冲突未解决时报错退出，需手动解决后重新运行。

**Q: 算子实现介绍如何用编辑器？**
A: 设置环境变量 `EDITOR`（如 `export EDITOR=vim`），脚本将自动打开编辑器输入，`#` 开头行视为注释自动过滤。

**Q: API 请求频繁失败？**
A: 脚本内置 3 次重试（间隔 2 秒），覆盖 5xx 服务端错误。持续失败请检查网络连接和 GitCode 服务状态。

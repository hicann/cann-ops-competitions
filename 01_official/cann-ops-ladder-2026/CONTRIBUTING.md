# 📥 PR 提交指南

本文档指导天梯赛参赛人员完成从 Git 环境配置到 Pull Request 提交的完整流程。

## ✅ 前置条件

- 注册 [GitCode](https://gitcode.com) 账号并完成实名认证
- 签署 CANN 社区 CLA（贡献者许可协议），签署指南见 [CLA 使用指南](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)

## ⚙️ 一、配置 Git 用户信息

打开终端，设置用户名和邮箱（需与 GitCode 账号信息保持一致）：

```bash
git config --global user.name "您的GitCode用户名"
git config --global user.email "您的GitCode注册邮箱"
```

验证配置：

```bash
git config --global user.name
git config --global user.email
```

## 🍴 二、Fork 代码仓

1. 浏览器访问目标仓库：https://gitcode.com/cann/cann-ops-competitions
2. 点击页面右上角 **Fork** 按钮
3. 选择 Fork 到个人账号空间，等待 Fork 完成
4. Fork 完成后将自动跳转到您个人空间下的仓库副本，路径为：`https://gitcode.com/{您的用户名}/cann-ops-competitions`

**需注意如果之前已经Fork过此仓库请先删除仓库再重新Fork**

## 📦 三、克隆个人仓到本地

```bash
git clone https://gitcode.com/{您的用户名}/cann-ops-competitions.git
cd cann-ops-competitions
```

## 🔄 四、同步上游仓库（可选）

**本步骤初次fork代码之后可不用操作，如果后续合并代码时有冲突，需要执行以下操作，拉取最新的代码来解冲突**

为确保本地代码与上游最新版本一致，添加上游仓库为远程源：

```bash
git remote add upstream https://gitcode.com/cann/cann-ops-competitions.git
git remote -v
```

拉取上游最新代码：

```bash
git pull upstream master --rebase
```

## 📁 五、创建团队提交目录

在对应月份的赛题提交目录下，按照命名规范创建团队代码提交目录。

**目录命名格式**：`{团队名称}`

例如，团队 `zhangsan` 提交`6月` `Hardwish` 算子任务的成果：

windows系统在`01_official/cann-ops-ladder-2026/June/Hardwish/submissions`目录下直接创建文件夹`zhangsan`

linux系统可以执行以下命令创建文件夹

```bash
mkdir -p 01_official/cann-ops-ladder-2026/June/Hardwish/submissions/zhangsan
```

将您的代码及文档复制到该目录中，确保包含：

- 可运行源码
- 配套说明文档（README）
- 必要的构建或运行脚本

windows系统直接复制，linux系统使用`cp -r`命令复制

## 📝 六、提交代码

以下分为`首次提交`和`后续修改代码多次提交`的场景，

### 首次提交

完成代码编写后，按以下步骤提交：

#### 查看变更

```bash
git status
```

#### 暂存文件

```bash
# 添加指定目录下的所有文件
git add 01_official/cann-ops-ladder-2026/June/Hardwish/submissions/zhangsan
```

#### 提交变更

```bash
git commit -m "feat: 提交 hardwish 算子"
```

提交信息建议使用 `feat: {简要描述}` 格式。

#### 推送到远程

```bash
git push origin master
```

### 多次提交（追加修改）

如果首次提交后需要补充或修改代码（如修复问题、补充文档），请使用 `--amend` 参数合并到上一次提交中，避免产生多条重复提交记录：

#### 暂存追加修改

```bash
git add 01_official/cann-ops-ladder-2026/June/Hardwish/submissions/zhangsan
```

#### 追加到上一次提交

```bash
git commit --amend -m "feat: 提交 hardwish 算子"
```

> 如果提交信息无需修改，可省略 `-m` 参数，直接使用 `git commit --amend` 保留原提交信息。

#### 强制推送

```bash
git push origin master --force
```

> 由于 `--amend` 修改了提交历史，推送时需加 `--force` 参数。请确认仅修改了自己的提交，避免覆盖他人代码。

## 🔀 七、创建 Pull Request

1. 推送成功后，浏览器访问您 Fork 的仓库页面：`https://gitcode.com/{您的用户名}/cann-ops-competitions`
2. 点击进入 **Pull Request** 标签页；
3. 在该标签页右上角有 **“+ 新建Pull Request”** 黑色按钮，点击进入；
4. 确认源分支（您推送代码的分支，示例是master分支）和目标仓库（`cann/cann-ops-competitions` master 分支），然后点击下一步；
5. 填写 PR 标题和描述，具体内容按照页面提示填写；
6. 点击 **“创建”** 提交
7. 在评论区输入"/compile"，ci会进行代码检测，检测成功会打上`ci-pipeline-passed`标签，检测失败则根据ci失败原因进行修改。
8. 等待社区审核，审核意见将通过 PR 评论反馈，请及时关注并响应。

## ❓ 常见问题

**Q：提交后 CLA 校验失败？**
A：请确认已在 CANN 社区完成 CLA 签署，签署指南见 [CLA 使用指南](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)。

**Q：PR 审核不通过怎么办？**
A：审核人会提出修改建议，在本地修改后再次 commit 并 push，PR 会自动更新，无需重新创建。

**Q：Fork 的仓库落后于上游仓库？**
A：执行 `git pull upstream master --rebase` 同步上游最新代码，再推送至个人仓库。
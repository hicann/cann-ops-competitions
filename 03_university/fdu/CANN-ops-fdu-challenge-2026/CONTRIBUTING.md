# 个人代码提交规范

本规范适用于 `submissions/` 目录下的个人代码提交。参赛同学需先在 CANNJudge 平台完成题目提交并获得平台判定成绩，再于 2026 年 6 月 6 日答辩前将最终代码提交到本仓库。

比赛入口：https://cannjudge.cn/fdu-aiops/fdu-competition-2026

## 一、提交目录命名

请在 `submissions/` 目录下创建以本人"学号_姓名全拼"命名的个人目录，姓名全拼使用小写英文字母，不添加空格。

目录格式：

```text
submissions/{student-id}_{name-pinyin}/
```

示例：

```text
submissions/2530XXXXXXX_zhangsan/
```

## 二、目录结构

个人提交目录建议包含以下内容：

```text
submissions/2530XXXXXXX_zhangsan/
├── README.md
└── code/
    ├── Addcmul/
    ├── ClipByValue/
    └── Lerp/
```

### 1. README.md

`README.md` 用于说明个人信息、CANNJudge 提交情况和实现思路。可参考以下模板：

```markdown
## 个人信息

- 姓名：
- 学号：
- 联系邮箱：
- CANNJudge 账号：

## CANNJudge 提交说明

- 比赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026
- 最终提交时间：
- 三道题完成情况（附提交成功截图）：
  - Addcmul：
  - ClipByValue：
  - Lerp：

## 算子实现简介

请简要说明三道题的实现思路、性能优化方法和遇到的问题。
```

### 2. code/

`code/` 目录用于存放三道题的最终代码。请按题目名称分别建立子目录：

```text
code/
├── Addcmul/
├── ClipByValue/
└── Lerp/
```

每道题目录中请放置可复现最终 CANNJudge 提交结果的工程代码。建议保留源码、构建配置和必要说明，避免提交临时文件和本地环境文件。

## 三、注意事项

1. 仅修改自己的 `submissions/{student-id}_{name-pinyin}/` 目录。
2. 提交内容需为本人完成的最终代码。
3. 不提交编译产物和缓存文件，例如 `build/`、`*.o`、`*.so`、`*.gcda`、`*.gcno`、`.cache/` 等。
4. 不提交敏感信息，包括但不限于密钥、密码、访问令牌、个人证件信息和内部资料。
5. 文件与目录命名建议使用英文、数字、下划线或短横线，避免空格。
6. 单个文件建议不超过 50MB，个人提交目录总大小建议不超过 200MB。

## 四、提交流程

1. Fork 比赛仓库到个人 GitCode 账号。
2. 克隆个人 Fork 仓库到本地开发环境。
3. 在 `submissions/` 下创建个人提交目录。
4. 将最终代码放入个人目录下的 `code/`，并补充个人 `README.md`。
5. 提交并推送到个人 Fork 仓库。
6. 向比赛仓库提交 Pull Request。

PR 标题建议：

```text
[个人提交] 复旦 CANN 校内自办赛：学号_姓名全拼
```

PR 描述建议：

```markdown
## 个人信息

- 姓名：
- 学号：
- 联系邮箱：
- CANNJudge 账号：

## 算子实现简介

请简要说明三道题的实现思路、性能优化方法和遇到的问题。
```

## 五、问题反馈

提交过程中如遇问题，请通过活动微信交流群或邮箱联系比赛志愿者：

- 毕昱阳：24110240003@m.fudan.edu.cn
- 蒋皓文：23110240114@m.fudan.edu.cn

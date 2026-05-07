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
├── README.md       # 必选：团队信息
├── code/           # 必选：算子代码
```

### 1. code/（算子代码）

算子代码直接下载[CANNJudge]((https://cannjudge.cn/))平台得分的工程代码即可，步骤如下：

（1）打开算子题目页面，以预选赛Erf算子为例，打开[Erf算子题目页面](https://cannjudge.cn/public/op_challenge_jiangshan_prelim/erf)

（2）点击 **我的提交记录** 中得分最高的提交记录打开提交页面

![cannjudge_code_search](./images/cannjudge_code_search.png)

（3）点击下载工程得到代码压缩包，解压便能得到完整算子代码

![cannjudge_code_download](./images/cannjudge_code_download.png)

（4）将代码修改成如下目录结构（添加一层算子名称文件夹即可）：

**预选赛提交代码：**

```
code/
├── Erf/
│   ├──op_kernel/
|		├──erf.cpp
|		├──erf_tiling.h
|		├──tiling_key_erf.h
|		└──CMakeLists.txt
│   ├──op_host/
|		├──erf.cpp
|		└──CMakeLists.txt
|	└──CMakeLists.txt
```



### 2. README.md（团队说明）

包含团队信息即可：

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



## 三、注意事项

1. **代码原创性**：提交内容须为团队原创，禁止抄袭其他团队作品或网络代码片段
2. **不提交编译产物**：`build/`、`*.o`、`*.gcda`、`*.gcno` 等由评测系统重新生成，无需提交
3. **不提交敏感信息**：包括但不限于密钥、密码、个人证件、内部资料等
4. **文件大小限制**：单个文件不超过 50MB，单个团队目录总大小不超过 200MB
5. **避免覆盖他人文件**：仅修改自己团队目录下的内容，不得改动其他团队、赛题文档或仓库配置文件

## 四、提交流程

以下流程均在终端环境上执行，过程中可能需要输入用户名和个人访问令牌进行验证，个人访问令牌可以访问 [访问令牌](https://gitcode.com/setting/token-classic) 页面进行创建。

1. Git全局配置，填写自己的GitCode用户名和GitCode配置邮箱

   ```
   git config --global user.name ***
   
   git config --global user.email ***
   ```

   

2. 设置环境变量

   ```bash
   export GITCODE_TOKEN="你的GitCode个人访问令牌"
   export GITCODE_USER="你的GitCode用户名"
   
   export UPSTREAM_OWNER="cann"
   export REPO="cann-competitions"
   export BASE_BRANCH="master"
   ```

   - GitCode个人令牌可以访问 [访问令牌](https://gitcode.com/setting/token-classic) 页面进行创建

2. Fork原仓库到个人仓库，此步骤需要注意如果此前Fork过cann-competition仓库需要删除仓库。

   ```bash
   curl -L "https://api.gitcode.com/api/v5/repos/${UPSTREAM_OWNER}/${REPO}/forks?access_token=${GITCODE_TOKEN}" \
     -H "Content-Type: application/json" \
     -H 'Accept: application/json' \
     -d "{ 
     \"organization\": \"${GITCODE_USER}\", 
     \"name\": \"cann-competitions\", 
     \"path\": \"cann-competitions\" 
   }"
   ```

3. 克隆个人仓库到终端环境

   ```bash
   git clone "https://gitcode.com/${GITCODE_USER}/${REPO}.git"
   cd "${REPO}"
   ```

4. 添加原仓库为`upstream`

   ```bash
   git remote add upstream "https://gitcode.com/${UPSTREAM_OWNER}/${REPO}.git"
   ```

5. 将第二节中下载并修改了目录结构的算子代码放到第一节中的指定提交目录下

6. 提交到本地master分支

   ```bash
   git add .
   git commit -m "update competition files"
   ```

7. 推送到fork的个人仓库

   ```bash
   git push origin "${BASE_BRANCH}"
   ```

8. 创建提交PR，需要修改命令中`title`参数里 **团队名称**

   ```bash
   curl -X POST "https://api.gitcode.com/api/v5/repos/${UPSTREAM_OWNER}/${REPO}/pulls?access_token=${GITCODE_TOKEN}" \
     -H "Authorization: Bearer ${GITCODE_TOKEN}" \
     -H "Content-Type: application/json" \
     -d "{
       \"title\": \"[团队提交]算子挑战赛江山赛区预赛提交：您的团队名称\",
       \"body\": \"算子挑战赛江山赛区预赛提交\",
       \"head\": \"${GITCODE_USER}:${BASE_BRANCH}\",
       \"base\": \"${BASE_BRANCH}\"
     }"
   ```

   

## 六、问题反馈

提交过程中如遇问题，请通过以下渠道联系组委会：

- GitCode Issue：在大赛仓库下提交 Issue
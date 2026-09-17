# 版本与实测证据索引

以下材料已经本地封存，尚未公开发布。收到同名本地验收包后可按包内相对路径定位；这不是线上下载链接。源码及原始证据的评审访问地址将在后续补充；本设计PR先提供方案、结果摘要及内容哈希。

## 版本链

| 项目 | 固定值 |
| --- | --- |
| ops-collections 上游基点 | `9d12996d4317e28420d74bcb1ec4d3b3507599ce` |
| 冻结实现候选 | `bbd2d2abd2248a7c` |
| 干净构建源码提交 | `9bee28c328a30b2d895c392f3c9b5f9eb8d256ee` |
| 最终源码及文档提交 | `038596d2fbe8e4f7a65b88f1f682faa08b476e05` |
| 实现内容聚合 SHA256 | `5107ffc5f5cda78361bc4c201d1077a486236f8a9f0cbdf86d8b55682831b05a` |
| 原始候选源码归档 SHA256 | `1e68417ab391e191b0b364ca01d83e988f6431eac60759bad42e05aee0bbe0ea` |

聚合实现哈希算法见源码 `scripts/run_hyperloglog_validation.py::implementation_hash`：对两份公开头文件及 detail/hyperloglog 全部文件，拼接相对POSIX路径和每文件SHA256，再对UTF-8串求SHA256。它不是单个头文件的SHA256。最终140项编译输入与干净构建一致的核验保存在包内交付校验材料；不能用文档提交日期推断重新跑过测试。

## 按结论定位

| ID | 支持的结论 | 本地包内入口 / 归档内定位 |
| --- | --- | --- |
| E0 | 版本、1280/72、干净构建关联 | `ACCEPTANCE.json`、`FILES_SHA256.json`、`verify_package.py` |
| E1 | 原始功能1280、零失败断言 | `report/functional-cases.csv`；E3归档内 `variants/bbd2d2abd2248a7c/final-functional/evidence/20260917T003837Z-82b7d6e5/`，包含report.json、原始XML/JSONL及命令日志 |
| E2 | 同候选唯一完整串行72/72 | `report/performance-rows.csv`；`evidence/hll-performance-round-bbd2d2abd2248a7c.tar.gz`，按后缀 `candidate-20260917T002911Z/performance/20260917T003114Z-91853df4/` 定位六份perf日志与report.json |
| E3 | 干净检出16目标构建和补充运行 | `evidence/hll-final-validation-bbd2d2abd2248a7c.tar.gz` 内 `run_hll_final_clean.sh`、`clean-final-9bee28c-evidence-v2/driver.log`、`build.log`、`run.log`、`job-exit.txt`、build-manifest.json及前后Git状态 |
| E4 | 全状态oracle、高rank、跨流、显式内存审计 | E3内干净构建补充运行报告；原候选补充记录也在E2中，不计入1280 |
| E5 | 原始测试未修改 | 源码 `tests/hyperloglog/target_manifest.json` 的13文件清单；包内 `source/original-target.tar.gz` |
| E6 | 关键失败与方案取舍 | `evidence/hll-failed-optimization-evidence-20260917T003320Z.tar.gz`；原xxHash路径实测未达到逐行门槛，保留原日志供追溯，失败成绩不进入本次72行 |

归档内按证据条目所列相对路径定位。所有通过结论以对应report.json、退出码和原始测试日志为准。

六类原始功能：Create38、Destroy24、Clear180、Add726、Merge192、Estimate120，共1280。补充7组合/336断言、多TU、API示例、12种配置各16轮内存审计单独统计。CPU oracle只用于测试比较，不进入容器实现。

## 关键材料SHA256
- `report/performance-rows.csv`：`5f6a359816993b80a34ff3c59f6e4c30393ceab9d93c4247e70ea46ecbaf676a`

- `report/functional-cases.csv`：`db2ed765fb6566661de52b8fd8f6e0c697a9ad53820592e605bb276491a3df97`


- `evidence/hll-failed-optimization-evidence-20260917T003320Z.tar.gz`：`e7f5fa2e0d56acc0c6ffdf9c475507d9e1b6eb4b95ab12712abcf23b5eb8a49f`

- `evidence/hll-final-validation-bbd2d2abd2248a7c.tar.gz`：`debf69e958fa86f44d7e6788a7f80274d02708ad6cdbd19ba1f60582fb66e23d`

- `evidence/hll-performance-round-bbd2d2abd2248a7c.tar.gz`：`1ad1aa3c52841e7f85a94635ec7170b59fb280f8d51828458a57a93372d969ca`

- `source/hyperloglog.bundle`：`62a5df1d5a923a830b71660cba3bab3bb610f0e464b9333ff8f2ea4d77f8df1b`

- `dependencies/Catch2-v3.5.4.tar.gz`：`b7754b711242c167d8f60b890695347f90a1ebc95949a045385114165d606dbb`

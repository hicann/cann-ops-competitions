# 初赛最佳已反馈版本：V18

- 版本对应提交：`77b0ca6`。
- 完整工程来源：`../初赛归档_2026-09-13/hccl_scatter_aicpu_problem_487_template/`。
- 已核对三个参赛文件与 `../初赛归档_2026-09-13/submission_v18_files/` 完全一致。
- 本目录保留完整 AICPU 工程、构建脚本、头文件及原始说明，不含 Git 历史。
- 平台历史显示耗时：测试点 9 为 37 μs，测试点 10 为 1.02 ms，测试点 11 为 812 μs；属于已反馈最佳组合，未做重复测量确认稳定收益。
- 成绩依据：`../初赛归档_2026-09-13/reference/Scatter各submission版本成绩汇总.md`。

## 构建

按 README.md 安装 CANN Toolkit 9.1.0，加载实际安装路径下的环境后，在本目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh
```

本次仅整合与逐文件一致性校验，未重新编译或进行平台正确性、性能测试。

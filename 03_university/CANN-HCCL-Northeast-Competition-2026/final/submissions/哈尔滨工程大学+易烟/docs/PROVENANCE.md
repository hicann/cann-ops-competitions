# 版本来源与本地校验

- 原工作区来源：`final_sub/code/；最终版本提交 267e76661f8743283bde45d3fad710b2ff8cac76，context 51，tag hccl_scatter_ccu_pull_v20_ctx51。`
- 完整复制文件数：19；全部逐字节一致，未改写原文件。
- 与历史参赛快照比对：6 个允许修改的参赛文件全部一致。
- 参赛文件：`include/custom.h`、`op_host/scatter.cc`、`op_host/exec_op.h`、`op_host/exec_op.cc`、`op_kernel_ccu/ccu_kernel.h`、`op_kernel_ccu/ccu_kernel.cc`。
- SHA-256 清单：在团队目录运行 `sha256sum -c SOURCE_SHA256SUMS`。
- 构建脚本语法：`bash -n code/build.sh`。
- 本次未重新编译或执行平台测试；不包含 PR 提交、CLA 签署、流水线执行或合入声明。

成绩来源为原工作区保存的参赛平台反馈摘要，详见本材料 README；没有将反馈摘要转换为独立复测结论。

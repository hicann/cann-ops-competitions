# 2026 HCCL 通信库创新大赛：决赛代码归档

## 团队信息

- 学校与队伍：西安电子科技大学-乔好了
- 赛段：决赛
- 归档版本：`e8bb5b4`（F1D1FZ2，最后一次线上提交对应版本）

## 代码说明

本目录保存决赛 CCU AllReduce 通信算子的最后线上提交源码与构建配置。工程结构与原始赛题模板保持一致：`include/` 包含公共头文件，`op_host/` 包含 Host 侧逻辑，`op_kernel_ccu/` 包含 CCU 内核逻辑；可使用 `bash build.sh` 构建。

线上提交使用的 6 个允许修改文件为 `include/custom.h`、`op_host/allreduce.cc`、`op_host/exec_op.cc`、`op_host/exec_op.h`、`op_kernel_ccu/ccu_kernel.cc` 和 `op_kernel_ccu/ccu_kernel.h`。该次提交已完成文件一致性校验；当时判题状态为 Running，自动监控超时，因此本归档不宣称最终判题结果。

## 归档范围

仅包含编译和复现所需的源代码、头文件、构建脚本与配置；不包含 Git 元数据、构建产物、缓存、实验日志、线上提交自动化工具或访问凭据。

# 哈尔滨工程大学+试一试队

## 提交信息

- 赛区：2026 CANN HCCL 通信库创新大赛东北赛区
- 学校：哈尔滨工程大学
- 队伍：试一试队
- 阶段：初赛
- 算子：Scatter（AICPU + TS 模式）
- CANNJudge 提交 ID：255304
- 功能结果：Pass
- 性能结果：点9 38μs，点10 1130μs，点11 895μs（总分 141.74）

## 方案演进

| 版本 | 关键改进 |
|------|----------|
| V1 | 去除接收端中转，root 直接下发 |
| V2 | 整包暂存快速路径 |
| V3 | 分轮并行 |
| V4 | 完整 slice 分批 |
| V5 | 数据路径优化 |
| V6 | 按槽复用 |
| V7 | 组合验证 |
| V8 | 小消息四叉控制树 |
| V9 | 部分远端中继 |
| V10 | 小树与中继组合 |
| 252193 | 一段 Push 直发 |
| 253741 | 两段 Push 流水 |
| 255304（最终） | 固定 1/4 远端分流：root 先把每个异机 rank 的尾段（count/4 个 float）发给同机 7 个 helper，再发 helper 自身整段；helper 转发尾段，异机 rank 从 root 直收前段、从 helper 收尾段。点10 从 1460μs 降到 1130μs，点11 从 1150μs 降到 895μs |

## 最终实现要点

- 仅在 P=16（2×8）、B≥1MiB、B+2F 不超过全部 rank 最小 CCL 容量时启用 1/4 分流，其余情况回退一段 Push 或旧分支。
- 每个 helper 的转发不依赖其自身 Copy 完成，尾段先发布，转发 worker 与自身 Copy 可并行推进。
- 小消息（<1MiB）走四叉控制树 + root 直读；大消息按本地 buffer 容量选择整包暂存、分批或逐 channel 分片。
- root 自身 slice 支持精确同址（跳过 Copy），不支持任意重叠。

## 验证

- CANNJudge 最终提交 Pass，三个测试点全部通过。
- 本地 HCCL-VM 验证：23 次 collective、368 个 rank 调用逐元素数值与 guard 通过，14 次实际进入新分流，112 份 helper 尾片完整比对通过。

## 修改文件

- `code/include/custom.h`
- `code/op_host/scatter.cc`
- `code/op_kernel_aicpu/exec_op.cc`

`code/` 为官方初赛模板的完整可编译工程，其余文件为官方预制逻辑。

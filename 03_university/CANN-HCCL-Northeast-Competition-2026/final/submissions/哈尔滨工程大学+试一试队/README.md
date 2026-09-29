# 哈尔滨工程大学+试一试队

## 提交信息

- 赛区：2026 CANN HCCL 通信库创新大赛东北赛区
- 学校：哈尔滨工程大学
- 队伍：试一试队
- 阶段：决赛
- 算子：Scatter（CCU 模式）
- CANNJudge 提交 ID：A21（最后官测成绩 19–30 点：11 / 965 / 757 / 12 / 2060 / 1610 / 11 / 925 / 726 / 13 / 1720 / 1350 μs）
- 功能结果：Pass

## 方案演进

| 版本 | 关键改进 |
|------|----------|
| F0 | 原生 CCU 直传基线 |
| F1 | root 双 die 并行，大消息 30%–50% 提速 |
| F2 | 接收端只运行包含 root 的活动接收组，小点各降 1–3μs |
| F3 | 参数压缩（SQE 893→532），大消息基本不动 |
| F7 | 多 helper 借输出前缀中转，29/30 快约 14% |
| F8 | 普通同机发送提前与 stage 重叠，20/21 快约 22% |
| F11 | 7 helper、4/11 分担比例 + Host 配对，移除 2×8 PLAN，20/21 再快约 8.7% |
| F13 | 2/5 比例 + 非中转小消息三参数接收 kernel |
| F14 | 小消息 Pull 协议，22 点 14→12μs |
| F15 | 4×3 Relay 自身复制迁到 Mesh 组，29/30 各降约 2% |
| F16B | 2×8 双模式普通 kernel（B<16MiB 走 Pull，大调用短尾保留 Push） |
| F18 | 非中转小消息专用 ReadyPush（逐 peer READY 即 Write，去掉 CONSUMED 收尾） |
| A21（最终） | 2×8 Mesh 中 restore 发起不再等待全组 suffix 完成，20/21 各降 3μs |

## 最终实现要点

- 拓扑识别：layer0 按 8/8 或 3/3/3/3 分组，2×8 用 7 helper 中转、4×3 用 2 helper 中转，非中转拓扑走普通 kernel。只按真实拓扑和完整调用大小路由，不按测试点特判。
- 2×8 大消息：root 双 die 并行；Mesh 组内 7 helper 各代转 4/11 长度，Clos 组直发；helper 转发完成（FORWARDED）后 root 恢复借用的输出前缀；suffix 写与 restore 互不依赖，restore 发起不再等全组 sent。
- 4×3 大消息：root 的自身复制放在较空闲的 Mesh 组，与 Clos 网络传输分担负载。
- 2×8 小消息：Pull 协议（root 发布 source/token，接收者 Read 后 DONE）；非中转小消息走 ReadyPush（逐 peer 准备后立即 Write，省 CONSUMED 通知）。
- 每 rank/die 最多注册 2 个 kernel；生产片长 256MiB，大消息多片循环带完整 B 步长与 offset。
- root 自身 slice 支持精确同址跳过复制；输入输出区间与通知位逐端校验。

## 验证

- CANNJudge 最终提交 Pass，功能与性能点全部出分。
- 本地 HCCL-VM：每种拓扑覆盖换 root、换地址、门槛两侧、小→大→小、zero 与跨片尾块，逐元素数值、输入保护、输出 guard 与 Checker 通过。
- 实际任务图核对：每个通知位的生产/消费、跨调用复用因果、数据与内存生命周期闭合。

## 修改文件

- `code/include/custom.h`
- `code/op_host/scatter.cc`
- `code/op_host/exec_op.cc`
- `code/op_host/exec_op.h`
- `code/op_kernel_ccu/ccu_kernel.cc`
- `code/op_kernel_ccu/ccu_kernel.h`

`code/` 为官方决赛模板的完整可编译工程，其余文件为官方预制逻辑。

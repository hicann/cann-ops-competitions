# 哈尔滨工业大学+No_Sleep_Just_Loop

## 提交信息

- 赛区：2026 CANN HCCL 通信库创新大赛东北赛区
- 学校：哈尔滨工业大学
- 队伍：No_Sleep_Just_Loop
- 阶段：决赛
- 算子：Scatter（CCU 模式）
- 功能结果：Pass

## 算子说明

### 数据分布

输入缓冲区按 rank 顺序排列，每个 rank 对应 `count` 个元素。Host 侧计算每个数据块的字节数 `sliceBytes` 以及整个输入的字节数 `sendBytes = sliceBytes × rankSize`。每个 rank 最终在输出缓冲区中得到自己的数据块，root 也会复制并保留属于自己的数据块。

### Host 侧资源与选路

`op_host/scatter.cc` 查询通信拓扑，申请 Mesh、Clos 通道并注册 CCU Kernel；`op_host/exec_op.cc` 根据注册的路径准备输入/输出内存 token 和 Kernel 参数，再下发任务。小消息（`sendBytes < 1 MiB`）使用 Root-Star：root 直接向各 peer 写入对应数据块。更大的消息按拓扑选择 Wave、Wave-Star 或 Proxy：Mesh 占主导的混合拓扑使用 Wave-Star，Clos 占主导的多 Server 拓扑可使用 Proxy；其他拓扑在消息达到 500 MiB 前使用 Wave，达到该规模后使用 Proxy。

### CCU 数据传输

Wave 将传输切成多个 wave，按偏移逐段向 peer 写入；Wave-Star 在每个 peer 的就绪通知到达后立即提交对应写入，让多个通道交错推进。Proxy 路径把跨 Server 数据拆为 root 直发段和代转段：root 经 Clos 发送直发段，并经本 Server 的 Mesh 把代转段交给 peer，再由 peer 经 Clos 转发至目标 rank。目标 rank 将两段写入自己的输出区。Kernel 使用事件和 channel 通知协调数据就绪与传输完成；代转切分按 512 字节对齐。

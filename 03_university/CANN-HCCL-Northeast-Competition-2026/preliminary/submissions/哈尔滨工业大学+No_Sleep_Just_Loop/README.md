# 哈尔滨工业大学+No_Sleep_Just_Loop

## 提交信息

- 赛区：2026 CANN HCCL 通信库创新大赛东北赛区
- 学校：哈尔滨工业大学
- 队伍：No_Sleep_Just_Loop
- 阶段：初赛
- 算子：Scatter（AICPU 模式）
- 功能结果：Pass

## 算子说明

### 数据分布与通信资源

输入缓冲区按 rank 顺序排列，每个 rank 对应 `count` 个元素；各 rank 将自己的数据块写入输出缓冲区，root 同时保留自己的数据块。该版本针对固定的 2 个 Server、每个 Server 8 个 NPU（共 16 个 rank）实现。`op_host/scatter.cc` 为每个 rank 准备同 Server 的 7 条 Mesh 通道和跨 Server 的 8 条 Clos 通道，并申请 AICPU 主线程及通信 worker；`launch_aicpu_kernel.cc` 负责加载和下发 AICPU Kernel。

### AICPU 通信路径

`op_kernel_aicpu/exec_op.cc` 按每个 rank 的数据块大小 `count × elementBytes` 选择路径。数据块不超过 1 MiB 时使用两级分层分发：root 将另一台 Server 的数据块集合发送给该 Server 的 subRoot，再由 subRoot 通过本地 Mesh 通道分发；root 所在 Server 的数据由 root 直接在本地分发。该路径使用 notify 同步各通道的数据就绪状态。

数据块超过 1 MiB 时使用部分代转。root 通过 Clos 直发跨 Server 数据块的前段，同时经本地 Mesh 把后段交给同号 peer，由 peer 再通过 Clos 转发到目标 rank；目标 rank 将两段放入对应输出区域。传输按不超过 256 MiB 的块循环处理，且块大小还受本地通信缓冲区容量限制。root 从输入缓冲区直接发起传输，避免先把待发送数据复制到中转缓冲区。

# AllGather 代码思路

本文只介绍当前代码的实现思路。环境配置和编译方法请查看原有的 `README.md`。

## 整体思路

该项目实现了一个基于 Ascend CCU 的 AllGather 通信算子。每个 Rank 都有一段输入数据，执行完成后，每个 Rank 的输出缓冲区中都会按 Rank 顺序保存所有 Rank 的数据：

```text
Rank i 的输入数据
    -> 写入所有其他 Rank 输出区的第 i 段
    -> 同时复制到本 Rank 输出区的第 i 段
```

实现分为 Host 侧和 CCU Kernel 侧：Host 侧负责准备通信资源、选择执行策略并下发任务；CCU 侧负责地址同步、跨 Rank 写数据、本地拷贝以及完成同步。

## Host 侧流程

入口位于 `op_host/allgather.cc` 中的 `HcclAllGather`，主要流程如下：

1. 检查输入指针、数据类型、Rank 数和数据长度。目前 `SIZE_TABLE` 只配置了 FP32，最大支持 16 个 Rank。
2. 查询 Rank 拓扑，为本 Rank 到其他 Rank 的 UBC_CTP 链路申请 Channel，并按本地 Endpoint 所在的 die 分组。
3. 为每个 die 分组注册 CCU Kernel，并为不同分组分配独立 Thread，使多个 die 上的传输可以并行执行。
4. 将 Channel、Thread 和 Kernel Handle 等资源序列化后缓存到 HCCL Engine Context，后续调用可直接复用。
5. `op_host/exec_op.cc` 根据消息大小和拓扑选择直传或中继流水模式，计算分片、地址偏移和本地拷贝参数，然后启动 Kernel。

## 两种传输策略

### 1. 直传模式

普通场景使用直传模式。每个 Rank 直接把自己的输入分片写到所有远端 Rank 对应的输出位置，并由一个 die 分组负责本地输入到本地输出的拷贝。

- 单个任务最多处理 256 MiB，数据更大时会拆成多个分片。
- 第一个分片使用 `CcuKernel`，负责交换输出地址和内存 Token。
- 后续分片使用 `CcuContinuationKernel`，复用已经同步好的地址，减少重复开销。
- 同一分片的多个 die 任务并行执行，完成后再进入下一分片。

### 2. 中继流水模式

当 Rank 数为 12 或 16、拓扑被分成两个 die 组，且单 Rank 数据量大于 8 MiB、不超过 1 GiB 时，代码会尝试启用中继优化。

数据被切成 4 个流水分片。每片约 3/5 的数据直接发往跨组目标，剩余约 2/5 先发给配对的中继 Rank，再由中继 Rank 在本地组内转发。这样可以让本地链路、跨 die 链路和中继转发形成流水，降低不对称链路的压力。16 Rank 对应 2 x 8 场景；12 Rank 对应 8 + 4 场景，其中无法配对的 Rank 继续走直接传输。

## CCU Kernel 流程

核心实现位于 `op_kernel_ccu/ccu_kernel.cc`：

1. 从 Kernel 参数中取得输入地址、各 Rank 输出地址、内存 Token、分片偏移和长度。
2. 首次执行时通过 Channel 互换输出地址与 Token，确保远端写操作可以访问正确位置。
3. 使用 `ccu::Write` 将本 Rank 数据写到远端输出缓冲区；使用 GroupCopy 完成本地数据拷贝。
4. 中继模式下，`CcuRelayKernel` 分两个阶段执行：先发送直传部分和中继尾部，再把收到的中继尾部转发给本地组内其他 Rank。
5. 使用 Event 和 Notify 等待写操作完成，并在最后一个分片结束时进行 Rank 间同步。

本地 GroupCopy 按 4 KiB 内存片组织，并通过循环和多缓冲并行拷贝，避免大块本地复制完全串行执行。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `include/custom.h` | Kernel 参数和可序列化资源上下文定义 |
| `op_host/allgather.cc` | 算子入口、拓扑/Channel 申请、Kernel 注册和资源缓存 |
| `op_host/exec_op.cc` | 分片计算、直传/中继策略选择及任务下发 |
| `op_kernel_ccu/ccu_kernel.cc` | CCU 侧地址同步、远端写、本地拷贝和中继转发 |
| `op_kernel_ccu/ccu_kernel.h` | CCU Kernel 上下文和辅助数据结构 |

简化后的调用链如下：

```text
HcclAllGather
  -> 校验参数并获取/创建资源上下文
  -> ExecOp
       -> LaunchDirect
       |    -> CcuKernel / CcuContinuationKernel
       |
       -> LaunchRelayPipeline
            -> CcuRelayKernel（直传阶段 + 中继阶段）
```

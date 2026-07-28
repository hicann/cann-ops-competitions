# AllReduce 714v3 实现说明

## 1. 版本定位

本文件记录当前 AllReduce 优化实现。工程目录：

```bash
/data/mingwei/hccl_allreduce_problem_222_template
```

容器内映射目录：

```bash
/home/workspace/hccl_allreduce_problem_222_template
```

README 中限定选手实现文件为：

- `include/custom.h`
- `op_host/allreduce.cc`
- `op_kernel_aicpu/exec_op.cc`

当前 v3 主要变化在 `op_kernel_aicpu/exec_op.cc`：

- flat 路径从 peer 写入改成 peer 读回，减少远端覆盖和 notify 时序风险。
- channel 读写后增加 `HcommChannelFenceOnThread`。
- `GetThreadForChannel()` 当前固定返回 `resCtx.threads[0]`，所有 channel 操作串行落到同一个 AICPU thread，优先保证 notify 顺序正确。
- 16 rank / 2 server 大消息路径继续保留，使用 server 内局部归约 + 跨 server 同 local-id 交换。
- 3 slot hierarchical two-server 试验路径代码保留，但触发阈值设为 `std::numeric_limits<uint64_t>::max()`，当前不会实际启用。

## 2. 资源申请

Host 侧入口为：

```cpp
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

Host 侧主要工作：

1. 校验 `sendBuf`、`recvBuf`、`comm`、`stream`。
2. 校验数据类型、reduce op、`count * dtype_size` 溢出。
3. 获取 `myRank` 和 `rankSize`。
4. 申请 CPU_TS thread，用于 Host 和 AICPU kernel 同步。
5. 获取本 rank 的 HCCL CCL buffer，保存为 `resCtx.localBuffer`。
6. 申请 AICPU_TS threads。当前申请 `rankSize - 1` 个 thread，但 device 侧实际统一使用 `threads[0]`。
7. 对每个 peer rank 申请一个 channel，并记录：
   - `remoteRank`
   - `ChannelHandle`
   - 对端 HCCL CCL buffer 地址和大小
8. 序列化 `AlgResourceCtx` 到 AICPU engine context。
9. 下发 AICPU kernel。

channel 描述从 `HcclRankGraphGetLayers()` 和 `HcclRankGraphGetLinks()` 获取。`SelectLink()` 优先选择 `COMM_PROTOCOL_UBC_CTP`，否则选择第一个非 reserved 链路。

每个 channel 的 notify 数为 3：

```cpp
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
```

实际数据同步当前使用：

- `NOTIFY_IDX_ACK = 0`
- `NOTIFY_IDX_DATA_SIGNAL = 1`

## 3. AICPU 入口

Device 侧入口为：

```cpp
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
```

入口公共逻辑：

1. 校验输入、输出、本地 CCL buffer、thread 资源。
2. 校验 rank 信息。
3. 将 HCCL data type / reduce op 转成 HCOMM data type / reduce op。
4. `count == 0` 直接返回。
5. `rankSize == 1` 时，非 inplace 场景只做本地拷贝。
6. 校验 channel 数量不少于 `rankSize - 1`。
7. 根据算法选择 slot 数并计算最大分片：

```cpp
maxSlotBytes = AlignDown(localBuffer.size / slotNum, 128)
maxSlotBytes = min(maxSlotBytes, 256MB)
maxSliceCount = maxSlotBytes / dataTypeSize
```

算法选择顺序：

```cpp
if (rankSize == 16 && totalBytes >= HIERARCHICAL_ALGO_MIN_BYTES) {
    RunHierarchicalTwoServerAllReduce(...)
} else if (rankSize == 16 && totalBytes >= 64KB) {
    RunTwoServerAllReduce(...)
} else {
    RunFlatAllGatherReduce(...)
}
```

当前：

```cpp
HIERARCHICAL_ALGO_MIN_BYTES = std::numeric_limits<uint64_t>::max()
```

所以实际启用路径只有：

- 4B 等小消息：flat allgather + local reduce。
- `rankSize == 16 && totalBytes >= 64KB`：two-server 分层路径。

## 4. 同步与 notify 时序

v3 中 channel 操作统一通过 `ReadWithEntries()` 或 `ExchangeWithEntries()` 执行。

当前主要使用读式路径：

```text
record ACK
wait ACK
HcommReadOnThread / HcommWriteOnThread
HcommChannelFenceOnThread
record DATA_SIGNAL
wait DATA_SIGNAL
```

关键点：

- 每个 channel 操作后都执行 `HcommChannelFenceOnThread()`，保证读写完成后再发数据完成信号。
- `GetThreadForChannel()` 固定返回 `resCtx.threads[0]`，避免多 thread 并发时同一 notify 资源上 record/wait 乱序。
- `ThreadSyncBefore()` / `ThreadSyncAfter()` 保留多 thread 同步框架；由于当前只使用 `threads[0]`，实际不会引入额外 thread notify。
- 同一个 channel 的 ACK 和 DATA_SIGNAL 任务按顺序串行下发，不在同一 notify 上提前堆多组 record。

该实现牺牲一部分并行度，换取更稳定的正确性和更容易排查的任务图。

## 5. Flat 路径

适用场景：

- 非 16 rank 拓扑。
- 16 rank 但总数据量小于 64KB。

函数：

```cpp
RunFlatAllGatherReduce(...)
```

slot 布局：

```text
slot 0      : rank 0 当前分片
slot 1      : rank 1 当前分片
...
slot N - 1  : rank N - 1 当前分片
```

单个分片流程：

1. 本 rank 将输入分片拷贝到本地 CCL buffer 的 `myRank` slot。
2. 对每个 peer，读取对端 CCL buffer 中 `peerRank` slot 到本地同名 slot。
3. 先把本 rank slot 拷贝到 `recvBuf` 当前分片。
4. 按 rank 从小到大，将其他 slot 固定顺序 reduce 到 `recvBuf`。

与早期写式 allgather 相比，v3 flat 路径由本 rank 主动拉取 peer 数据，减少“多个 peer 同时写入同一个 rank buffer”的调度不确定性。

## 6. Two-Server 路径

适用场景：

```cpp
rankSize == 16 && totalBytes >= 64 * 1024
```

拓扑假设：

```text
server 0: rank 0  - rank 7
server 1: rank 8  - rank 15
```

本地编号：

```cpp
localRank = myRank % 8
```

跨 server 配对：

```cpp
crossPeer = (myRank + 8) % 16
```

slot 布局：

```text
slot 0      : 本 server local rank 0 当前分片
slot 1      : 本 server local rank 1 当前分片
...
slot 7      : 本 server local rank 7 当前分片
slot 8      : 对端 server 同 local-id rank 的局部归约结果
```

单个分片流程：

1. 本 rank 将输入分片拷贝到 `localRank` slot。
2. 读取同 server 7 个 peer 的对应 local slot 到本地 slot 0..7。
3. 将 slot 0..7 固定顺序 reduce 到本 rank 的 `localRank` slot，得到 server 内 8 卡局部结果。
4. 只读取另一台 server 上同 local-id rank 的局部结果到 slot 8。
5. 将 slot 8 reduce 到 `localRank` slot。
6. 将最终结果从 `localRank` slot 拷贝到 `recvBuf`。

通信量变化：

```text
flat 路径:
  每 rank 与 15 个 peer 交换
  跨 server fanout = 8
  slot 数 = 16

two-server 路径:
  每 rank 与 8 个 peer 交换
  跨 server fanout = 1
  slot 数 = 9
```

因此 two-server 路径主要优化 512KB、512MB、400MB+4B 这类比赛性能用例。

## 7. Hierarchical 试验路径

代码中保留：

```cpp
RunHierarchicalTwoServerAllReduce(...)
```

设计目标是使用 3 个 slot：

- input slot
- result slot
- temp slot

流程大致为 server 内 reduce-scatter、跨 server 同 chunk 交换、server 内 allgather。但该路径涉及更多交错读写和 notify，同步风险更高。当前为保证正确性，触发阈值设为最大值，实际不会进入。

后续如果重新启用，必须先完整验证：

- 4B、512KB、512MB、400MB+4B 正确性。
- checker 任务图无死锁、无内存冲突。
- 同一 notify 资源没有多组 record 提前打到同一个 wait。

## 8. 当前实现的取舍

优点：

- 保留 deterministic local reduce 顺序。
- two-server 路径降低跨 server fanout。
- 读式拉取减少远端多写入冲突。
- channel fence 和单 thread 下发降低 notify 乱序风险。
- 分片大小按 CCL buffer 动态计算，最大单片限制 256MB。

代价：

- 所有 channel 操作串到 `threads[0]`，通信并行度下降。
- server 内 8 slot reduce 仍由单 thread 串行执行。
- 未使用 `HcommWriteReduceOnThread` / `HcommWriteReduceWithNotifyOnThread`。
- hierarchical 3 slot 路径暂未启用。

## 9. 后续优化方向

建议按以下顺序推进：

1. 先在 normal 模式下跑通全部功能用例，避免 `--check-only` 掩盖数据错误。
2. 对 512KB 单独调 `TWO_SERVER_ALGO_MIN_BYTES`，确认 flat 和 two-server 的实际拐点。
3. 在保持 notify 顺序可证明的前提下，逐步恢复多 thread channel 并行。
4. 评估 HCOMM reduce 类原语，减少 CCL buffer 中转和本地串行 reduce。
5. hierarchical 3 slot 路径只在 checker 和 all_reduce_test 都稳定后再重新打开。


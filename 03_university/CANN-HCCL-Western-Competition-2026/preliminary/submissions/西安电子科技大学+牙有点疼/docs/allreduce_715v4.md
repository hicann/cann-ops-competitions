# AllReduce 715v4 实现说明

## 1. 版本定位

本文记录 2026-07-15 当前 AllReduce 优化实现。工程目录：

```bash
/data/mingwei/hccl_allreduce_problem_222_template
```

容器内映射目录：

```bash
/home/workspace/hccl_allreduce_problem_222_template
```

赛题限定选手实现文件为：

- `include/custom.h`
- `op_host/allreduce.cc`
- `op_kernel_aicpu/exec_op.cc`

当前 v4 主要变化集中在 `op_kernel_aicpu/exec_op.cc`：

- 保留 v3 的 flat 读回路径和 two-server 分层路径。
- 对 two-server 大消息最后一个小尾块增加 `ReadReduce` 直接规约路径。
- 处理 CCL buffer base 非 128B 对齐的问题，slot 起点统一通过 `GetAlignedBufferOffset(..., 128)` 修正。
- 修复尾块跨 server 阶段的 CCL 读写冲突：跨 server peer 只读取稳定的 partial slot，不读取正在被原地 reduce 的 result slot。
- 3 slot hierarchical two-server 试验路径仍保留，但触发阈值为 `std::numeric_limits<uint64_t>::max()`，当前不会启用。

## 2. Host 侧资源

Host 侧入口：

```cpp
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

主要流程：

1. 校验 `sendBuf`、`recvBuf`、`comm`、`stream`。
2. 校验 data type、reduce op 和 `count * dtype_size` 溢出。
3. 获取 `myRank`、`rankSize`。
4. 申请 CPU_TS thread，用于 Host/AICPU 同步。
5. 获取本 rank 的 HCCL CCL buffer，写入 `resCtx.localBuffer`。
6. 申请 AICPU_TS thread。当前 host 侧仍申请 `rankSize - 1` 个 thread，但 device 侧实际统一使用 `threads[0]`。
7. 对每个 peer rank 申请 channel，并记录：
   - `remoteRank`
   - `ChannelHandle`
   - 对端 HCCL CCL buffer 地址和大小
8. 序列化 `AlgResourceCtx` 到 AICPU engine context。
9. 下发 AICPU kernel。

channel 描述来自 `HcclRankGraphGetLayers()` 和 `HcclRankGraphGetLinks()`。`SelectLink()` 优先选择 `COMM_PROTOCOL_UBC_CTP`，否则选择第一个非 reserved 链路。

每个 channel 的 notify 数为：

```cpp
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
```

当前同步使用：

- `NOTIFY_IDX_ACK = 0`
- `NOTIFY_IDX_DATA_SIGNAL = 1`

## 3. AICPU 入口与算法选择

Device 侧入口：

```cpp
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
```

公共逻辑：

1. 校验输入、输出、本地 CCL buffer 和 thread 资源。
2. 校验 rank 信息。
3. 将 HCCL data type / reduce op 转成 HCOMM 类型。
4. `count == 0` 直接返回。
5. `rankSize == 1` 时，非 inplace 场景只做本地拷贝。
6. 校验 channel 数量不少于 `rankSize - 1`。
7. 根据算法选择 slot 数并计算最大分片。

slot 和切片计算：

```cpp
alignedBaseOffset = GetAlignedBufferOffset(localBuffer, 128)
maxSlotBytes = (localBuffer.size - alignedBaseOffset) / slotNum
maxSlotBytes = AlignDown(maxSlotBytes, 128)
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

当前实际启用路径：

- 4B 等小消息：flat allgather + local reduce。
- `rankSize == 16 && totalBytes >= 64KB`：two-server 分层路径。

## 4. 同步时序

channel 操作统一通过 `ReadWithEntries()`、`ReadReduceWithEntries()` 或保留的 `ExchangeWithEntries()` 包装。

读式同步时序：

```text
record ACK
wait ACK
HcommReadOnThread / HcommReadReduceOnThread / HcommWriteOnThread
HcommChannelFenceOnThread
record DATA_SIGNAL
wait DATA_SIGNAL
```

关键约束：

- 每个 channel 数据操作后执行 `HcommChannelFenceOnThread()`，保证数据完成后再发 DATA_SIGNAL。
- `GetThreadForChannel()` 固定返回 `resCtx.threads[0]`，避免多 thread 并发下同一 notify 资源 record/wait 时序难以推导。
- `ThreadSyncBefore()` / `ThreadSyncAfter()` 保留多 thread 同步框架；当前实际只用 `threads[0]`，不会产生额外 thread notify。
- 同一 channel 的 ACK 和 DATA_SIGNAL 串行下发，避免提前在同一 notify 上堆多组 record/wait。

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
3. 将 slot 0..N-1 按固定 rank 顺序 reduce 到本 rank slot。
4. 将最终结果从本 rank slot 拷贝到 `recvBuf`。

该路径通过本 rank 主动拉取 peer 数据，避免多个 peer 同时写入同一 rank CCL 区域。

## 6. Two-Server 主路径

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

常规分片 slot 布局：

```text
slot 0      : 本 server local rank 0 当前分片
slot 1      : 本 server local rank 1 当前分片
...
slot 7      : 本 server local rank 7 当前分片
slot 8      : 对端 server 同 local-id rank 的局部归约结果
```

单个完整分片流程：

1. 本 rank 将输入分片拷贝到 `localRank` slot。
2. 读取同 server 7 个 peer 的对应 local slot 到本地 slot 0..7。
3. 将 slot 0..7 固定顺序 reduce 到本 rank 的 `localRank` slot，得到 server 内 8 卡局部结果。
4. 只读取另一台 server 上同 local-id rank 的局部结果到 slot 8。
5. 将 slot 8 reduce 到 `localRank` slot。
6. 将最终结果从 `localRank` slot 拷贝到 `recvBuf`。

通信量对比：

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

该路径主要覆盖 512KB、512MB、400MB+4B 等性能用例。

## 7. v4 尾块优化

### 7.1 触发条件

尾块优化只在 two-server 路径中触发：

```cpp
ShouldUseTailReadReduce(processedCount, remainingCount, maxSliceCount, ...)
```

触发条件：

- `processedCount != 0`，即前面已经处理过至少一个完整分片。
- `remainingCount < maxSliceCount`，当前循环处理的是最后小尾块。
- `remainingBytes <= 1MB`。
- 本地 CCL buffer 能容纳 tail input slot 和 tail result slot。

当前常量：

```cpp
constexpr uint64_t TAIL_READ_REDUCE_MAX_BYTES = 1ULL * 1024ULL * 1024ULL;
```

### 7.2 尾块 slot 布局

尾块路径不用常规 9 slot 布局，只使用两个 128B 对齐后的连续区域：

```text
tail input slot  : alignedBaseOffset
tail result slot : alignedBaseOffset + AlignUp(tailBytes, 128)
```

布局由 `BuildTailSlotLayout()` 生成。对本地和远端 CCL buffer 都重新按各自 base 地址计算 `alignedBaseOffset`，避免 VM 中 CCL base 形如 `...0001` 导致任务访问非预期偏移。

### 7.3 尾块执行流程

函数：

```cpp
RunTwoServerTailReadReduce(...)
```

流程：

1. 将本 rank 输入尾块同时拷贝到：
   - tail input slot
   - tail result slot
2. 对同 server 7 个 peer 执行 `HcommReadReduceOnThread`：
   - 远端读取源：peer 的 tail input slot。
   - 本地规约目标：本 rank tail result slot。
   - 结果：tail result slot 变成本 server 8 卡局部和。
3. 将本地局部和从 tail result slot 拷贝回 tail input slot。
4. 对跨 server 同 local-id peer 执行 `HcommReadReduceOnThread`：
   - 远端读取源：peer 的 tail input slot。
   - 本地规约目标：本 rank tail result slot。
   - 结果：tail result slot 变成 16 卡全局和。
5. 将 tail result slot 拷贝到 `recvBuf` 尾块位置。

第 3 步是 v4 的关键修复。跨 server 两端会同时读取对方局部结果并原地更新自己的最终结果；如果直接读取对方 tail result slot，checker 会看到 rank 0 / rank 8 这类配对在同一 CCL 区间上发生并发读写冲突。将局部结果复制到 tail input slot 后，跨 server peer 读取稳定的只读源，tail result slot 只作为本地写目标。

### 7.4 任务数量收益

常规 two-server 尾块如果走完整 9 slot 流程，需要：

- 读取 7 个 server 内 peer 到 7 个 slot。
- 本地 7 次 reduce。
- 读取 1 个跨 server slot。
- 本地 1 次 reduce。

v4 尾块路径改为：

- 7 次 server 内 `ReadReduce`。
- 1 次跨 server `ReadReduce`。
- 两次小块本地 copy 用于初始化和发布稳定 partial。

尾块通常只有 4B 或很小尺寸，主要收益是减少本地 reduce 和 slot 搬运任务，降低 400MB+4B 这类“完整大块 + 极小尾块”场景的尾部任务开销。

## 8. Hierarchical 试验路径

代码中仍保留：

```cpp
RunHierarchicalTwoServerAllReduce(...)
```

设计目标是使用 3 个 slot：

- input slot
- result slot
- temp slot

流程大致是 server 内 reduce-scatter、跨 server 同 chunk 交换、server 内 allgather。该路径涉及更多交错读写和 notify 时序，当前为保证正确性不启用：

```cpp
HIERARCHICAL_ALGO_MIN_BYTES = std::numeric_limits<uint64_t>::max()
```

后续如果重新启用，必须完整验证：

- 4B、512KB、512MB、400MB+4B 正确性。
- checker 任务图无死锁、无内存冲突。
- 同一 notify 资源没有多组 record 提前打到同一个 wait。

## 9. 本地验证记录

构建和替换库命令：

```bash
docker exec hccl-vm bash -lc 'source /etc/profile.d/hccl-vm.sh && \
cd /home/workspace/hccl_allreduce_problem_222_template && \
bash build.sh && \
cp -f build/include/hccl.h ${ASCEND_HOME_PATH}/x86_64-linux/include/hccl/ && \
cp -f build/lib64/libhccl.so ${ASCEND_HOME_PATH}/x86_64-linux/lib64/ && \
cp -f build/lib64/libhccl_device.so ${HCCL_VM_INSTALL_DIR}/lib/aarch64/'
```

已验证结果：

| 用例 | 结果 |
|---|---|
| 4B | `check_result: success` |
| 512KB | `check_result: success`，checker Success |
| 46603268B，即一个 full slice + 4B 小尾块 | `check_result: success`，checker 两轮 Success |
| 400MB + 4B | 修复后 checker Success；当前本地 VM runner 在真实执行时仍出现 `Bus error`，未产出数值结果行 |

`46603268B` 用例用于覆盖 v4 尾块路径，能验证 `full slice + 4B tail` 的数值正确性和 checker 任务图。

## 10. 当前取舍

优点：

- 保留 deterministic reduce 顺序。
- two-server 主路径降低跨 server fanout。
- 尾块路径减少小尾块任务数量。
- cross tail 阶段使用稳定 partial slot，规避 result slot 并发读写冲突。
- 所有 channel 操作保留 `HcommChannelFenceOnThread()`，同步语义清晰。

代价：

- 所有 channel 操作仍串行在 `threads[0]`，牺牲一定并行度。
- 尾块优化只覆盖小于等于 1MB 的尾块。
- hierarchical 3 slot 路径暂不启用，512MB 与 400MB 主体仍使用 9 slot two-server 算法。

后续可继续评估：

1. 在保持 notify 时序可证明的前提下，恢复部分多 thread 并行。
2. 对 512KB 单独测试 flat 与 two-server 的真实性能拐点。
3. 进一步排查本地 VM 在 400MB+4B normal runner 中的 `Bus error`，区分 VM 共享内存限制和算法实现问题。

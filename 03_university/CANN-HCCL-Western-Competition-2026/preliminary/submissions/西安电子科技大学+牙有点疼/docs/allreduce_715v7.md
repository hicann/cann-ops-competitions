# AllReduce 715v7 当前优化实现说明

## 1. 版本定位

本文记录 2026-07-15 在 v6 基础上完成的 AllReduce v7 优化。工程目录：

```bash
/data/mingwei/hccl_allreduce_problem_222_template
```

容器内映射目录：

```bash
/home/workspace/hccl_allreduce_problem_222_template
```

v7 算法源码只修改了：

- `op_kernel_aicpu/exec_op.cc`

对应提交：

```text
947d529 Optimize 512KB hierarchical allreduce
```

v7 直接基于 v6。测试点 6、7 使用的大消息实现保持不变：

- hierarchical CCL buffer 使用 2 个完整 slot。
- 同 Server reduce-scatter 使用 packed contribution。
- 本地规约使用固定 `4 -> 2 -> 1` 三层并行树。
- 跨 Server scratch 复用 input slot。
- 同 Server allgather 和不超过 1MB 的 tail `ReadReduce` 路径保持不变。

本轮只重构精确 512KB 的测试点 5：

```text
v6: 16 Rank 全消息二叉树 reduce + 反向树 broadcast
v7: Server 内 recursive-halving reduce-scatter
    + 跨 Server 同 local-id Rank 配对归约
    + Server 内 reverse recursive-doubling allgather
```

## 2. v6 的 512KB 路径瓶颈

v6 对精确 512KB 使用固定二叉树。设消息大小为 `S`，16 Rank reduce 的 stride 为：

```text
1 -> 2 -> 4 -> 8
```

反向 broadcast 的 stride 为：

```text
8 -> 4 -> 2 -> 1
```

每条树边传输完整消息，因此最长依赖链包含：

```text
reduce critical path    : 4S
broadcast critical path : 4S
total critical payload  : 8S
```

对 512KB：

```text
8S = 4MB
```

原路径的优点是任务图短：

```text
task metadata   : 216
checker nodes   : 217
data task nodes : 62
```

但它存在三个性能问题：

1. 每个阶段都搬运完整 512KB，关键路径网络载荷大。
2. 树根和中间 Rank 承担更多串行工作，16 Rank 负载不均衡。
3. 跨 Server reduce 和 broadcast 分别传输一个完整消息，跨 Server 关键路径为 `2S`。

二叉树的全局网络数据量为：

```text
15 条 reduce 边 + 15 条 broadcast 边 = 30S
```

v7 不改变 `30S` 的全局总量，而是把它均匀分布到全部 Rank，并缩短串行关键路径。

## 3. v7 算法选择

512KB 专用阈值改名为：

```cpp
SMALL_HIERARCHICAL_ALLREDUCE_BYTES = 512ULL * 1024ULL;
```

当前选择顺序为：

```cpp
if (rankSize == 16 && totalBytes == 512KB) {
    RunSmallHierarchicalAllReduce(...);
} else if (rankSize == 16 && totalBytes > 1MB) {
    RunHierarchicalTwoServerAllReduce(...);
} else if (rankSize == 16 && totalBytes >= 64KB) {
    RunTwoServerAllReduce(...);
} else {
    RunFlatAllGatherReduce(...);
}
```

比赛测试点与当前实现的对应关系：

| 测试点 | 数据量 | v7 路径 |
|---:|---:|---|
| 1-4 | 小消息 | flat allgather + local reduce 或 two-server `ReadReduce` |
| 5 | 512KB | small hierarchical reduce-scatter + cross reduce + allgather |
| 6 | 512MB | v6 2-slot hierarchical + parallel local reduction tree |
| 7 | 400MB+4B | v6 2-slot hierarchical 主体 + small tail `ReadReduce` |

精确 512KB 对所有当前支持的数据类型都能整除 8 个 local Rank。代码仍显式检查：

```cpp
param.count % RANKS_PER_SERVER == 0
```

## 4. 512KB 双 slot 布局

设：

```text
S = 512KB
C = S / 8 = 64KB
```

v7 使用两个完整 slot：

```text
slot 0: inputSlot
        初始保存完整输入
        reduce-scatter 后每个 Rank 只依赖自己的 64KB server partial

slot 1: resultSlot
        保存跨 Server 归约后的 owned chunk
        allgather 后保存完整 512KB 结果
```

slot stride 为：

```cpp
slotStride = AlignUp(totalBytes, 128);
```

本地和每个远端 CCL buffer 都独立计算 128B 对齐偏移，并检查至少能容纳：

```text
2 * slotStride = 1MB
```

核心地址关系如下：

```text
local input chunk  = localBase + chunkIdx * 64KB
remote input chunk = remoteBase + chunkIdx * 64KB

local result chunk  = localBase + slotStride + chunkIdx * 64KB
remote result chunk = remoteBase + slotStride + chunkIdx * 64KB
```

所有跨 Rank read 的 remote source 和 local destination 位于不同 Rank 的 CCL memory，不发生本地地址重叠。

## 5. Server 内 recursive-halving reduce-scatter

每个 Server 包含 8 个 Rank。定义：

```cpp
localRank = myRank % 8;
serverStart = myRank / 8 * 8;
```

reduce-scatter 使用三个 mask：

```text
mask 4: peerLocalRank = localRank ^ 4, transfer 4 chunks = 256KB
mask 2: peerLocalRank = localRank ^ 2, transfer 2 chunks = 128KB
mask 1: peerLocalRank = localRank ^ 1, transfer 1 chunk  = 64KB
```

每一步都保留一个连续半区：

```cpp
keepLowerHalf = (localRank & mask) == 0;
```

两个 peer 各自从对端 slot 0 读取自己保留的半区，并直接归约到本地 slot 0：

```text
rank A reads/reduces A-retained range from rank B
rank B reads/reduces B-retained range from rank A
```

双方读取的是对方在本阶段不再保留的范围。remote source 在读取过程中不会被对方覆盖；本地只修改自己保留的范围。

三个阶段结束后满足：

```text
rangeStartChunk == localRank
rangeChunkNum   == 1
```

即每个 Rank 在 slot 0 中得到一个 64KB 的 8 Rank Server partial。

每 Rank 的 Server 内 reduce-scatter 网络量为：

```text
256KB + 128KB + 64KB = 448KB = 7S/8
```

## 6. 跨 Server 配对归约

两个 Server 上相同 local id 的 Rank 配对：

```cpp
crossPeerRank = (myRank + 8) % 16;
```

每个 Rank 先将 slot 0 中自己的 64KB server partial copy 到 slot 1 的 owned chunk：

```text
slot 0 owned chunk -> slot 1 owned chunk
```

然后从跨 Server peer 的 slot 0 读取同一 owned chunk，并归约到本地 slot 1：

```text
rank r     reads/reduces rank r+8 slot-0 partial -> local slot-1 result
rank r+8   reads/reduces rank r slot-0 partial   -> local slot-1 result
```

slot 0 partial 在跨 Server 阶段保持只读，两个方向的 source 和 destination 位于不同 slot。每 Rank 只传输：

```text
S / 8 = 64KB
```

跨 Server 结束后，每个 Rank 在 slot 1 的 owned chunk 中得到完整 16 Rank 归约结果。

## 7. Server 内 reverse recursive-doubling allgather

allgather 反向使用三个 mask：

```text
mask 1: read 1 completed chunk  = 64KB
mask 2: read 2 completed chunks = 128KB
mask 4: read 4 completed chunks = 256KB
```

每个 Rank 从 peer 的 slot 1 读取已经完成的连续区间，直接写入本地 slot 1 缺失区间：

```cpp
peerLocalRank = localRank ^ mask;
```

阶段结束后本地完整区间依次扩大：

```text
1 chunk -> 2 chunks -> 4 chunks -> 8 chunks
```

每 Rank 的 Server 内 allgather 网络量为：

```text
64KB + 128KB + 256KB = 448KB = 7S/8
```

最后执行：

```text
complete resultSlot -> user output
```

## 8. channel 同步协议

### 8.1 最终稳定序列

每个 pair read/reduce 使用以下序列：

```text
record ACK
wait peer ACK
read or read-reduce
channel fence
record DATA_SIGNAL
wait peer DATA_SIGNAL
```

对应 helper：

```cpp
RunSmallPairReadReduce(...)
RunSmallPairRead(...)
```

ACK 保证双方已经发布本阶段 source。DATA_SIGNAL 保证双方 channel 操作完成后才进入下一阶段。

### 8.2 symmetric push 尝试

中间版本曾使用双向：

```cpp
HcommWriteReduceWithNotifyOnThread(...)
```

该版本可以构建并生成任务，但 HCCL-VM checker 无法闭合两端 local notify wait：

```text
Local Record/Wait matching is stuck
firstBlockedWaitNode=[TaskWaitAICPU]
```

删除 write 后的 channel fence 不能解决 notify 匹配问题。因此 v7 不使用 symmetric push。

### 8.3 仅 ACK 的 pull 尝试

第二个中间版本使用 symmetric pull，但只保留：

```text
record ACK -> wait ACK -> read -> fence
```

该版本生成：

```text
task metadata   : 448
checker nodes   : 449
data task nodes : 160
```

Checker V3 的任务图、内存冲突和语义检查全部成功，但 normal runner 在任务开始执行后不返回。

这说明 remote source 即使不再被覆盖，当前 runner 仍需要显式 DATA 完成握手来推进 channel 生命周期。最终版本补回
`DATA_SIGNAL record/wait` 后，normal runner 单轮和重复执行均成功。

该问题也说明：

```text
Checker Success != normal runner 数值执行成功
```

两种验证都必须保留。

## 9. 完整 512KB 流程

```text
user input, 512KB
       |
       v
copy -> slot 0
       |
       v
intra-server recursive halving
mask 4: 256KB read-reduce
mask 2: 128KB read-reduce
mask 1:  64KB read-reduce
       |
       v
one 64KB server partial per Rank in slot 0
       |
       v
copy owned partial -> slot 1
paired cross-server 64KB read-reduce
       |
       v
one completed 16-Rank chunk per Rank in slot 1
       |
       v
intra-server reverse recursive doubling
mask 1:  64KB read
mask 2: 128KB read
mask 4: 256KB read
       |
       v
copy complete slot 1 -> user output
```

## 10. 通信量与任务图代价

### 10.1 关键路径网络量

v7 每 Rank 网络量为：

```text
reduce-scatter : 448KB = 7S/8
cross reduce   :  64KB =  S/8
allgather      : 448KB = 7S/8
total          : 960KB = 15S/8
```

与 v6 二叉树的关键路径对比：

| 指标 | v6 512KB tree | v7 512KB hierarchical |
|---|---:|---:|
| 关键路径通信阶段 | 8 | 7 |
| 关键路径网络载荷 | 4MB | 960KB |
| 跨 Server 阶段 | 2 | 1 |
| 全局网络总量 | `30S` | `30S` |

关键路径网络载荷下降：

```text
1 - (15/8) / 8 = 76.5625%
```

### 10.2 task metadata 增长

最终 DATA 完成握手增加了控制任务：

| 指标 | v6 tree | v7 final | 变化 |
|---|---:|---:|---:|
| task metadata | 216 | 672 | +211.1% |
| checker nodes | 217 | 673 | +210.1% |
| data task nodes | 62 | 160 | +158.1% |

全局 task 数上升不能直接等价为关键路径时延同比上升：

- v6 的任务较少，但完整消息传输集中在树根和中间 Rank。
- v7 的 16 个 Rank 在每个阶段同时工作，160 个 data task 分布在不同 Rank 上。
- v7 每 Rank 的网络 primitive 为 7 个，树关键路径为 8 个完整消息 primitive。
- v7 的收益依赖多 Rank 和多链路并行，代价是 AICPU notify、任务下发和调度压力增加。

因此 v7 属于“通信关键路径更短、控制任务更多”的取舍。是否获得净加速必须由真实 Ascend 950 测试决定，不能
仅根据 Checker 的全局 task 数或 HCCL-VM 固定计时判断。

## 11. v6 大消息路径保持不变

测试点 6、7 继续使用 v6 实现，本轮没有修改其数据布局和调度：

```text
2-slot hierarchical buffer
       |
       +-- input slot
       +-- packed contribution / result slot

intra-server:
7 peer owned-chunk reads
       |
fixed 4 -> 2 -> 1 local reduction tree
       |
paired cross-server exchange
       |
7 peer allgather
```

默认 400MB CCL buffer 下的最大 full slice 仍为：

```text
209,715,072 bytes = 200MiB - 128B
```

分片结构仍为：

```text
测试点 6, 512MB:
  3 hierarchical slices

测试点 7, 400MB+4B:
  2 hierarchical slices + 260B tail
```

对应 v6 Checker 记录继续有效：

| 用例 | checker nodes | data tasks | 结果 |
|---|---:|---:|---|
| 512MB | 8,441 | 1,272 | Checker Success |
| 400MB+4B | 6,353 | 1,040 | Checker Success |

## 12. 构建与安装

Release 构建：

```bash
docker exec hccl-vm bash -lc 'source /etc/profile.d/hccl-vm.sh; \
cd /home/workspace/hccl_allreduce_problem_222_template; \
bash build.sh'
```

替换 VM device 库：

```bash
cp -f build/lib64/libhccl_device.so \
  ${HCCL_VM_INSTALL_DIR}/lib/aarch64/libhccl_device.so
```

最终构建产物和安装目录中的 device 库哈希一致：

```text
c573c707216d1261b17622685ad6962386b5f015c1799987709e2dadb096e41a
```

`git diff --check` 和 Release 编译均通过。

## 13. 512KB Checker 验证

check-only 模式：

```bash
cd ${HCCL_VM_INSTALL_DIR}/bin
./hccl-vm start ascend950_cluster_4_server_competition.yaml --check-only
```

在 `(hvm)$>` 中执行：

```bash
hccl-vm mock-comm 128
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 524288 -e 524288 -d fp32 -o sum -w 0 -n 1 -c 0 -p 8
hccl-vm plugin run @checker
```

最终结果：

```text
task metadata         : 672
checker nodes         : 673
data task nodes       : 160
normal semantic count : 112
normal semantic bytes : 33,554,432
SingleTaskCheck       : success
MemConflict           : success
SemanticCheck         : success
checker               : Success
```

MemConflict 阶段记录的 560 组 overlap candidate 全部具有确定的先后顺序，没有 parallel conflict。

## 14. normal runner 数值验证

normal 模式启动并安装 runner：

```bash
cd ${HCCL_VM_INSTALL_DIR}/bin
./hccl-vm start ascend950_cluster_4_server_competition.yaml
```

在 `(hvm)$>` 中执行：

```bash
hccl-vm plugin install @runner
hccl-vm plugin list
hccl-vm mock-comm 128
```

确认：

```text
checker RUNNING
runner  RUNNING
```

单轮数值验证：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 524288 -e 524288 -d fp32 -o sum -w 0 -n 1 -c 1 -p 8
```

结果：

```text
524288 | success
```

重新执行 `hccl-vm mock-comm 128` 后进行 20 次重复验证：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 524288 -e 524288 -d fp32 -o sum -w 0 -n 20 -c 1 -p 8
```

结果：

```text
524288 | success
```

每次启动新的 `mpirun` 前需要重新执行 `hccl-vm mock-comm 128`。否则上一批进程释放的共享内存上下文会导致
`shm_open failed`，该错误发生在通信域初始化阶段，与 v7 算法无关。

HCCL-VM 中 `aclrtEventElapsedTime` 返回固定值，因此 `-n 1` 显示的 1000us 和 `-n 20` 显示的 50us 都不能
作为真实性能结果，只能使用 `check_result: success` 判断数值正确性。

## 15. 当前测试结果

当前实现记录的测试 5、6、7 性能为：

| 测试点 | 数据量 | 当前实现 | 实测时间 |
|---:|---:|---|---:|
| 5 | 512KB | v7 small hierarchical | 59.00 μs |
| 6 | 512MB | v6 2-slot hierarchical | 4.78 ms |
| 7 | 400MB+4B | v6 2-slot hierarchical + tail | 3.78 ms |

这些数值是当前测试记录，不是 HCCL-VM 的固定 event stub 输出。测试 5 是本轮 v7 新路径的结果；测试 6、7
反映 v7 继续保留的 v6 大消息实现。

若要计算 v7 相对 v6 的加速比例，必须在相同机器、相同 warmup、相同迭代次数和相同日志配置下重新测试 v6
512KB tree 基线。当前文档不使用不同环境的数字推导加速比。

## 16. 性能结论边界

当前已经证明：

- 512KB 双 slot 地址范围有效。
- recursive-halving、cross reduce 和 recursive-doubling 的 Checker 语义正确。
- 最终任务图无 notify deadlock、无 CCL memory conflict。
- normal runner 单轮和 20 次重复数值验证成功。
- 512KB 关键路径网络载荷从 `8S` 降为 `15S/8`。
- 测试点 6、7 的 v6 大消息路径没有被本轮代码修改。

当前需要继续关注：

- metadata 从 216 增加到 672 后的真实 AICPU 调度开销。
- 8 个 Rank pair 同时通信时的 Server 内链路并行效率。
- 8 组跨 Server 64KB 配对通信是否能充分并行。
- 512KB 是否处于通信载荷收益与 notify 固定开销的最佳交叉点。

全局 task 数上升不代表时延按比例上升，因为这些任务分布在 16 个 Rank 上；同样，网络关键路径下降也不能单独
证明最终时延一定下降。最终结论应以测试点 5 的同环境 A/B 数据为准。

## 17. 后续优化方向

### 17.1 减少 pair 控制任务

最终每个 pair step 使用 ACK 和 DATA 两组 record/wait。后续可以在真实平台和 Checker 都支持的前提下验证：

- 能否使用可靠的 fused read/write notify primitive。
- 能否按 phase 合并部分 ready/completion barrier。
- 能否减少 fence 与 notify 之间的固定调度开销。

不能直接删除 DATA handshake；当前 normal runner 已证明仅 ACK 的版本会停滞。

### 17.2 保留 tree/hierarchical A/B 开关

512KB 处于中等消息区间，notify 开销可能影响收益。正式调优时应保留可快速切换的：

```text
binary-tree reduce + reverse broadcast
small hierarchical reduce-scatter + cross reduce + allgather
```

并对比：

- 端到端平均时延。
- p50、p99 时延。
- AICPU task 下发时间。
- Server 内和跨 Server 链路利用率。

### 17.3 继续优化大消息

测试点 6、7 仍可沿 v6 文档中的方向继续验证：

- 单 slot push reduce-scatter。
- 分片双缓冲流水线。
- 并行本地规约树与串行规约的真实设备 A/B。

## 18. 最终结论

v7 是在 v6 上针对测试点 5 的增量优化：

- 用 8 Rank recursive-halving reduce-scatter 替换全消息树 reduce。
- 用 8 组同 local-id Rank 完成 64KB 跨 Server 配对归约。
- 用 reverse recursive-doubling allgather 替换全消息反向树 broadcast。
- 每 Rank 网络量为 `15S/8 = 960KB`，树关键路径为 `8S = 4MB`。
- 使用两个 512KB slot，不影响 v6 的大消息 2-slot 实现。
- 最终 ACK + DATA channel 协议通过 Checker 和 normal runner 双重验证。
- 512KB Checker 为 673 nodes / 160 data tasks，normal runner 单轮和 20 次重复结果均正确。
- 当前测试 5、6、7 分别为 `59.00 μs`、`4.78 ms`、`3.78 ms`。

v7 用更多控制任务换取更短、更均衡的网络关键路径。测试结果已经记录，但后续仍应保留 v6 tree 基线，在同一真实
Ascend 950 环境中持续进行 A/B 测试和 notify 开销分析。

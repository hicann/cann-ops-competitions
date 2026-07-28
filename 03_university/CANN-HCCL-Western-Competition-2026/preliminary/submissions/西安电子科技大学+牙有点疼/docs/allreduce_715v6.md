# AllReduce 715v6 当前优化实现说明

## 1. 版本定位

本文记录 2026-07-15 在 v5 基础上完成的 AllReduce v6 大消息优化。工程目录：

```bash
/data/mingwei/hccl_allreduce_problem_222_template
```

容器内映射目录：

```bash
/home/workspace/hccl_allreduce_problem_222_template
```

比赛允许选手修改的文件为：

- `include/custom.h`
- `op_host/allreduce.cc`
- `op_kernel_aicpu/exec_op.cc`

v6 算法源码只修改了：

- `op_kernel_aicpu/exec_op.cc`

`custom.h` 中已有的 thread/channel 资源描述足够使用，`allreduce.cc` 也已经申请 15 个 worker thread 和
全部 peer channel，因此本轮不需要修改这两个文件。

v6 直接基于 v5，保留以下实现：

- 小于 64KB 的 flat allgather + local reduce 路径。
- 精确 512KB 的二叉树 reduce + 反向树 broadcast 快路径。
- 64KB 到 1MB 的 two-server `ReadReduce` 路径。
- 大消息的同 Server reduce-scatter、跨 Server 配对交换、同 Server allgather 框架。
- 不超过 1MB 的大消息尾块单 thread `ReadReduce` 路径。
- v5 的 worker thread 固定映射和 main notify 固定索引。

v6 只重构测试点 6、7 使用的大消息主体：

1. hierarchical CCL buffer 从 3 个完整 slot 减少到 2 个完整 slot。
2. 同 Server 的 7 次串行本地规约改为固定 `4 -> 2 -> 1` 三层并行规约树。

## 2. v5 剩余性能瓶颈

### 2.1 第三个完整 slot 限制分片大小

v5 hierarchical 路径使用：

```text
slot 0: input
slot 1: result
slot 2: temp
```

默认 CCL buffer 为 400MB，并且 CCL base 在 VM 中可能不是 128B 对齐。v5 的单次最大分片约为：

```text
400MB / 3 = 133.33MiB
实际最大分片 = 139,810,048 bytes
```

因此两个大消息用例需要：

```text
512MB      : 4 个 hierarchical 分片
400MB + 4B : 3 个 hierarchical 分片 + 260B 尾块
```

每增加一个分片，都要重新执行：

- 输入到 CCL 的完整分片 copy。
- reduce-scatter 的 channel notify、read 和 thread barrier。
- 本地规约。
- 跨 Server 交换。
- allgather。
- CCL 到输出的完整分片 copy。

### 2.2 本地规约仍在协调线程串行

v5 虽然把同 Server 的 7 个 peer read 分配到多个 worker thread，但读取完成后的本地规约仍为：

```text
result = rank 0 contribution
result += rank 1 contribution
result += rank 2 contribution
...
result += rank 7 contribution
```

7 次 `HcommLocalReduceOnThread` 全部下发到 `threads[0]`。对于每 Rank 负责的 `S/8` chunk，本地规约关键
路径包含 7 次完整 chunk reduce。Full-Mesh 通信已经并行，而本地规约阶段仍然串行，可能成为大消息路径的
主要瓶颈。

### 2.3 v5 已经解决的问题

v5 每 Rank 的网络通信量已经降为：

```text
reduce-scatter: 7S / 8
cross exchange:  S / 8
allgather:       7S / 8
total:          15S / 8 = 1.875S
```

因此 v6 不重新设计网络数据分布。当前 `15S/8` 已接近该两层拓扑下 reduce-scatter + allgather 的合理通信量，
本轮重点是减少分片轮数和本地规约关键路径。

## 3. v6 算法选择

算法阈值保持 v5 不变：

```cpp
TWO_SERVER_ALGO_MIN_BYTES = 64KB
TAIL_READ_REDUCE_MAX_BYTES = 1MB
SMALL_TREE_ALLREDUCE_BYTES = 512KB
HIERARCHICAL_ALGO_MIN_BYTES = 1MB + 1
```

选择顺序仍为：

```cpp
if (rankSize == 16 && totalBytes == 512KB) {
    RunSmallTreeAllReduce(...);
} else if (rankSize == 16 && totalBytes > 1MB) {
    RunHierarchicalTwoServerAllReduce(...);
} else if (rankSize == 16 && totalBytes >= 64KB) {
    RunTwoServerAllReduce(...);
} else {
    RunFlatAllGatherReduce(...);
}
```

比赛用例对应关系：

| 数据量 | v6 路径 |
|---|---|
| 4B | flat allgather + local reduce |
| 512KB | one-slot binary-tree reduce + reverse-tree broadcast |
| 512MB | 2-slot hierarchical + parallel local reduction tree |
| 400MB+4B | 2-slot hierarchical 主体 + 2-slot small tail `ReadReduce` |

## 4. 2-slot hierarchical 布局

### 4.1 slot 数量

v6 将：

```cpp
HIERARCHICAL_SLOT_NUM = 3
```

改为：

```cpp
HIERARCHICAL_SLOT_NUM = 2
```

删除原来的 `HIER_TEMP_SLOT_IDX`。当前布局为：

```text
slot 0: input slot
slot 1: packed contribution workspace + final result slot
```

slot 起点仍按每个本地/远端 CCL buffer 独立计算：

```cpp
alignedBaseOffset = GetAlignedBufferOffset(buffer, 128)
slotOffset = alignedBaseOffset + slotIdx * AlignUp(sliceBytes, 128)
```

### 4.2 最大分片

VM 中 CCL buffer 大小为 419,430,400 bytes，典型 base 地址低位为 `...001`，需要 127 bytes 对齐修正。
v6 单 slot 实际大小为：

```text
AlignDown((419,430,400 - 127) / 2, 128)
= 209,715,072 bytes
= 200MiB - 128B
```

分片结构变为：

```text
512MB:
  209,715,072 + 209,715,072 + 117,440,768
  共 3 个 hierarchical 分片

400MB + 4B:
  209,715,072 + 209,715,072 + 260
  共 2 个 hierarchical 分片 + 260B tail
```

与 v5 相比：

| 用例 | v5 | v6 |
|---|---:|---:|
| 512MB hierarchical 分片 | 4 | 3 |
| 400MB+4B hierarchical 主体分片 | 3 | 2 |

## 5. 2-slot reduce-scatter

设当前分片大小为 `S`，本 Rank 的 local id 为：

```cpp
localRank = myRank % 8
```

每个 Rank 负责约 `S/8` 的 owned chunk。v6 流程如下：

1. 将完整输入分片复制到 input slot。
2. 本 Rank 的本地 contribution 直接引用 input slot 中的 owned chunk。
3. 从同 Server 7 个 peer 并行读取相同 owned chunk。
4. 7 个远端 contribution 紧凑写入 result slot 的 7 个互不重叠区域。
5. 对 8 份 contribution 执行固定三层规约树。
6. 将本 Server partial 发布到 result slot 的最终 owned chunk 偏移。

远端 contribution 的临时布局为：

```text
result slot + 0 * chunkBytes: peer contribution 0
result slot + 1 * chunkBytes: peer contribution 1
...
result slot + 6 * chunkBytes: peer contribution 6
```

代码继续检查：

```cpp
localEntries.size() * chunkBytes <= slotStride
```

即 7 份远端 owned chunk 必须能放入一个完整 result slot。

这些 packed ranges 可能与最终 `resultChunk` 地址重叠，但不会发生并发冲突：

- reduce-scatter read 阶段只写 packed contribution。
- 本地规约树完成前不会发布最终 result chunk。
- 所有 contribution 被消费后，才将 local partial copy 到最终 result chunk 偏移。
- checker 的 MemConflict 阶段已验证该时序。

## 6. 固定三层并行规约树

### 6.1 规约拓扑

v6 新增：

```cpp
ReduceContributionsAsTree(...)
```

同 Server 8 个 source local rank 按固定树规约：

```text
Level 1, 4 parallel reductions:
  0 <- 1
  2 <- 3
  4 <- 5
  6 <- 7

Level 2, 2 parallel reductions:
  0 <- 2
  4 <- 6

Level 3, 1 reduction:
  0 <- 4
```

最终 local rank 0 contribution 所在区域保存本 Server 的 8 Rank partial。

### 6.2 关键路径

规约的数据总量没有变化，仍是每 Rank 约 `7S/8`，但关键路径发生变化：

```text
v5: 7 个 chunk reduce 串行
v6: 3 层 chunk reduce，层内分别 4 / 2 / 1 路并行
```

忽略 thread barrier 固定开销，在 local reduce primitive 能并行执行的前提下，规约关键路径由 7 个 chunk
时长下降到 3 个 chunk 时长。

### 6.3 worker 与同步

每层只同步当前实际参与的 worker：

```text
Level 1: 4 workers
Level 2: 2 workers
Level 3: 1 worker
```

每层使用 v5 已验证的同步框架：

```text
main record worker-start
worker wait worker-start
worker local-reduce
worker record main-complete
main wait main-complete
```

worker 到 main 的 notify 索引仍由 worker 全局索引固定决定，不随当前层的 vector 位置变化，避免不同阶段复用
错误 notify 导致任务图闭环。

每个分片的本地规约树新增：

```text
(4 + 2 + 1) * 4 = 28 个 thread notify task / Rank
```

因此 v6 的总 task metadata 不会简单按分片数同比下降。新增 notify 用来换取本地规约关键路径并行化。

### 6.4 确定性

v6 不再使用 v5 的线性 `((((0+1)+2)+3)...+7)` 顺序，而是使用固定平衡树：

```text
((0+1) + (2+3)) + ((4+5) + (6+7))
```

所有 Rank、所有 owned chunk 都使用相同 source rank 配对和相同三层顺序。跨 Server 阶段继续固定为：

```text
server 0 partial op server 1 partial
```

因此相同输入的每次执行具有相同浮点运算树，满足赛题的确定性要求。

## 7. 跨 Server scratch 复用

v5 使用独立 temp slot 保存跨 Server peer partial。v6 在同 Server 规约完成后，input slot 的 owned chunk 已经
不再需要，因此直接复用：

```cpp
crossTemp = inputSlot + chunkOffsetBytes
```

跨 Server 两个 paired Rank 仍先同时读取对端稳定的 result chunk：

```text
rank r     reads rank r+8 partial -> local crossTemp
rank r+8   reads rank r partial   -> local crossTemp
```

双方的 `ReadWithEntries` 完成后才执行本地规约，因此不会读取正在被对端修改的 result chunk。

规约顺序保持：

```text
server 0 rank: resultChunk = server0Partial op crossTemp(server1Partial)
server 1 rank: crossTemp = crossTemp(server0Partial) op resultChunk(server1Partial)
               copy crossTemp -> resultChunk
```

两个 Server 得到相同的浮点运算顺序。

## 8. allgather 与尾块

### 8.1 allgather

跨 Server 规约后，每个 Rank 已经在 result slot 的 owned chunk 位置发布完整 16 Rank partial。随后继续使用 v5
的同 Server allgather：

1. 读取同 Server 7 个 peer 的最终 owned chunk。
2. 直接写入 result slot 的最终 chunk 偏移。
3. 使用 `VariableReadPlan` 处理 chunk 元素数相差 1 的情况。
4. 将完整 result slot 分片 copy 到输出 buffer。

### 8.2 400MB+4B 尾块

400MB+4B 在 v6 下的尾块为 260 bytes。该尾块仍满足：

```text
tailBytes <= 1MB
```

因此继续使用 v5 的单 thread 2-slot `ReadReduce` 尾块路径，而不是进入并行规约树：

- 规约顺序固定。
- 多个 peer 不会并发写同一 result tail。
- 跨 Server peer 读取稳定的 partial slot。
- v6 未修改该路径的 notify 和内存布局。

## 9. 完整大消息流程

```text
user input slice
       |
       v
copy full slice -> input slot
       |
       v
7 intra-server peers parallel read owned chunk
remote contributions -> packed ranges in result slot
       |
       v
fixed local tree reduction
4 parallel -> 2 parallel -> 1
       |
       v
publish server partial -> final owned offset in result slot
       |
       v
reuse input owned chunk as crossTemp
paired cross-server read + fixed-order reduce
       |
       v
7 intra-server peers parallel allgather
       |
       v
copy complete result slot -> user output slice
       |
       +-- next full slice
       |
       +-- remaining <= 1MB: single-thread tail ReadReduce
```

## 10. 通信量与理论收益

### 10.1 网络通信量

v6 没有改变 v5 的网络数据量：

```text
每 Rank reduce-scatter = 7S/8
每 Rank cross exchange = S/8
每 Rank allgather       = 7S/8
每 Rank总网络量         = 15S/8
```

### 10.2 收益来源

v6 的预期收益来自：

1. 512MB 分片轮数从 4 降为 3。
2. 400MB 主体分片轮数从 3 降为 2。
3. 同 Server 本地规约关键路径从 7 次 reduce 降为 3 层 reduce。
4. 删除第三个完整 temp slot，提高 CCL buffer 有效利用率。
5. 跨 Server scratch 复用 input slot，避免新增完整 buffer。

v6 不会降低总网络字节数，也没有消除输入和输出两次完整分片 copy。

## 11. 构建与验证

### 11.1 Release 构建

```bash
docker exec hccl-vm bash -lc 'source /etc/profile.d/hccl-vm.sh; \
cd /home/workspace/hccl_allreduce_problem_222_template; \
bash build.sh'
```

最终 Release 构建成功，无编译 error。

device 库替换后哈希一致：

```text
e55749b8dcf5ad822b865ff8a7f3fefeb1ca0ae0802b6312065339dcef7cd3e6
```

### 11.2 2MB normal runner 数值验证

2MB 消息会进入与测试点 6、7 相同的 2-slot hierarchical 和固定规约树路径，并且只包含一个分片。

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 2097152 -e 2097152 -d fp32 -o sum -w 0 -n 1 -c 1 -p 8
```

normal 模式安装 runner 后，结果为：

```text
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
2097152           | 1000.00        | 2.09715              | success
```

这里只将 `check_result: success` 作为有效数值正确性结论。`1000 us` 是 VM event stub 固定值。

### 11.3 测试点 6：512MB checker

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 536870912 -e 536870912 -d fp32 -o sum -w 0 -n 1 -c 0 -p 8
hccl-vm plugin run @checker
```

结果：

```text
slice structure       : 3 hierarchical slices
task metadata         : 8,440
checker nodes         : 8,441
data task nodes       : 1,272
normal semantic bytes : 23,890,751,488
SingleTaskCheck       : success
MemConflict           : success
SemanticCheck         : success
checker               : Success
```

### 11.4 测试点 7：400MB+4B checker

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 419430404 -e 419430404 -d fp32 -o sum -w 0 -n 1 -c 0 -p 8
hccl-vm plugin run @checker
```

结果：

```text
slice structure       : 2 hierarchical slices + 260B tail
task metadata         : 6,352
checker nodes         : 6,353
data task nodes       : 1,040
normal semantic bytes : 20,132,655,232
SingleTaskCheck       : success
MemConflict           : success
SemanticCheck         : success
checker               : Success
```

### 11.5 v5 与 v6 任务图对比

| 用例 | 指标 | v5 | v6 | 变化 |
|---|---|---:|---:|---:|
| 512MB | hierarchical 分片 | 4 | 3 | -25.0% |
| 512MB | checker nodes | 9,441 | 8,441 | -10.6% |
| 512MB | data tasks | 1,696 | 1,272 | -25.0% |
| 400MB+4B | hierarchical 主体分片 | 3 | 2 | -33.3% |
| 400MB+4B | checker nodes | 7,801 | 6,353 | -18.6% |
| 400MB+4B | data tasks | 1,464 | 1,040 | -29.0% |

checker nodes 的下降比例小于 data tasks，是因为固定三层规约树增加了 worker thread notify。数据任务减少来自
分片轮数下降；新增同步任务用于换取本地规约的层内并行。

## 12. 性能结论边界

HCCL-VM 可以运行 `hccl_test`、Runner 和 Checker，但当前版本不是周期级性能模拟器。虚拟 runtime 中：

```cpp
aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent)
{
    *ms = 1;
    return ACL_SUCCESS;
}
```

因此以下字段不能用于比较 v5 和 v6：

```text
aveg_time(us)
alg_bandwidth(GB/s)
```

当前已经证明：

- 2-slot 缓冲复用数值路径在 2MB normal runner 下正确。
- 测试点 6、7 的任务图无死锁、无内存冲突、规约语义正确。
- 大消息分片轮数和 data task 数量下降。
- 本地规约 DAG 从 7 次串行变为固定 3 层树。

当前尚不能由 VM 证明：

- 多 worker local reduce 在真实 Ascend 950 上的并行效率。
- thread notify 固定开销是否会抵消部分并行收益。
- 测试点 6、7 的真实微秒数和带宽提升比例。

最终性能必须在正式评测环境或真实 Ascend 950 上对 v5、2-slot 串行规约版和 v6 并行树版做 A/B 测试。

## 13. 后续优化方向

### 13.1 单 slot push reduce-scatter

当前每个 Rank 先把完整输入 `S` copy 到 CCL input slot，再由 peer 主动读取。后续可以验证
`HcommWriteOnThread` 是否允许直接使用用户输入地址作为本地 source：

1. 每个 Rank 将各 owner chunk 直接写到 owner Rank 的独立 contribution range。
2. owner 本地执行固定规约树。
3. 删除完整 input slot，只保留一个 result/workspace slot。

如果底层 primitive 和 checker 允许，该方案有机会：

- 消除完整 input-to-CCL copy。
- 将 CCL buffer 有效分片进一步提高到接近 400MB。
- 512MB 降到 2 个主体分片。
- 400MB+4B 降到 1 个接近完整的主体分片加小尾块。

该方案需要重点验证远端写入地址、用户 buffer source 支持、接收端 ready handshake 和 contribution range 冲突。

### 13.2 双缓冲流水线

当前每个分片仍严格串行执行 copy、reduce-scatter、cross、allgather 和 output copy。另一方向是使用双缓冲小分片，
重叠：

```text
slice N allgather/output
slice N+1 input/reduce-scatter
```

该方案可能提高稳态吞吐，但会增加分片数、notify 复用和任务图复杂度，必须以真实设备 A/B 结果决定是否采用。

### 13.3 并行树与串行规约 A/B

v6 的并行树依赖真实平台能够并行执行多个 worker 上的 local reduce。正式评测前应保留一个可快速切换的串行
版本，分别测量：

- 2-slot + 7 次线性串行 reduce。
- 2-slot + `4 -> 2 -> 1` 固定并行树。

如果 local reduce 底层实际共享同一执行资源，并行树可能只增加 notify；如果存在多个可并行执行队列，则 v6
应明显缩短本地规约阶段。

## 14. 最终结论

v6 是在 v5 上针对测试点 6、7 的增量优化：

- 保留 v5 的拓扑感知 `15S/8` hierarchical 通信算法。
- 3 个完整 CCL slot 减少为 2 个。
- 512MB 分片从 4 个降为 3 个。
- 400MB 主体分片从 3 个降为 2 个。
- data task 分别下降 25.0% 和 29.0%。
- 同 Server 本地规约关键路径从 7 次串行 reduce 改为固定 3 层并行树。
- 所有 Rank 使用相同规约树和相同跨 Server 操作数顺序，保持确定性。
- 2MB normal runner 数值验证成功，测试点 6、7 Checker V3 全阶段成功。

真实性能提升比例仍需正式 Ascend 950 环境确认，不能使用 HCCL-VM 输出的固定 `1000 us` 进行判断。

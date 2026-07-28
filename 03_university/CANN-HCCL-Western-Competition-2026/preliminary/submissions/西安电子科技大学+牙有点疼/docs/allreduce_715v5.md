# AllReduce 715v5 当前实现与验证说明

## 1. 版本定位

本文记录 2026-07-15 完成的 AllReduce v5 及其 512KB 低启动开销快路径。工程目录：

```bash
/data/mingwei/hccl_allreduce_problem_222_template
```

容器内映射目录：

```bash
/home/workspace/hccl_allreduce_problem_222_template
```

赛题允许选手修改的文件为：

- `include/custom.h`
- `op_host/allreduce.cc`
- `op_kernel_aicpu/exec_op.cc`

v5 最终只有 `op_kernel_aicpu/exec_op.cc` 存在源码内容变化。`custom.h`、`allreduce.cc` 和
`README.md` 内容保持不变。

当前实现针对三类主要性能问题：

1. 512MB 和 400MB+4B 主体仍使用 9-slot 全量拉取算法，通信量和本地规约量过大。
2. 所有 channel 操作都下发在 `threads[0]`，没有利用 Server 内 7 条 Full-Mesh 链路并行度。
3. 512KB 原路径生成 768 条 task metadata，并执行 8 次整消息 `ReadReduce`，固定任务下发开销过大。

最终采用分段策略：

- 小于 64KB：保留 v4 flat 路径，4B 行为不变。
- 精确 512KB：使用单结果 slot 的二叉树 reduce + 反向树 broadcast 快路径。
- 64KB 到 1MB 的其他消息：使用 2-slot `ReadReduce` 融合路径。
- 大于 1MB：使用 3-slot two-server reduce-scatter + cross exchange + allgather。
- 大消息最后不超过 1MB 的尾块：回到单 thread 的 2-slot `ReadReduce` 路径。

## 2. v4 性能瓶颈

### 2.1 9-slot 主路径通信量

设当前处理的数据量为 `S`。v4 two-server 路径在每个 Rank 上执行：

1. 从同 Server 7 个 peer 各读取完整 `S`。
2. 本地执行 7 次完整 `S` 规约。
3. 从跨 Server 对端读取完整 `S`。
4. 本地再执行 1 次完整 `S` 规约。

因此每 Rank 的主要开销为：

```text
网络读取量 = 7S + S = 8S
本地规约量 = 7S + S = 8S
```

v4 已经把跨 Server fanout 从 8 降为 1，但每个 Rank 仍然拉取本 Server 的 7 份完整数据，
没有执行 reduce-scatter。

### 2.2 单 thread 串行

v4 的 channel thread 选择为：

```cpp
return resCtx.threads[0];
```

7 条 Server 内链路按 peer 顺序串行执行：

```text
peer 0 read -> fence -> signal
peer 1 read -> fence -> signal
...
peer 6 read -> fence -> signal
```

因此即使拓扑是 Full-Mesh，也不能并行使用多条链路。

### 2.3 分片过小

默认 CCL buffer 约 400MB。v4 大消息使用 9 个 slot：

```text
max slice ~= 400MB / 9 = 44.4MiB
```

512MB 需要约 12 个分片循环。每个循环都重复 channel notify、read、fence、本地 reduce 和 copy，
任务下发开销明显。

### 2.4 512KB 固定开销

优化前的 512KB 路径对每个 Rank 读取同 Server 的 7 个完整消息，再读取跨 Server 对端的 1 个完整消息。
数据量不大，但仍沿用多阶段全量拉取流程：

```text
每 Rank 整消息 ReadReduce : 8 次
task metadata             : 768
checker nodes             : 769
data tasks                : 192
```

512KB 的主要问题不是链路带宽，而是 task、notify 和通信阶段的固定开销。当前优化因此不修改大消息分层算法，
而是为精确 512KB 增加更短的执行路径。

正式评测基线与目标为：

```text
优化前 512KB : 287 us
第一阶段目标 : <= 100 us
进一步目标   : 60-80 us
```

本文后续的 VM 结果用于验证任务图和数值正确性。优化后的真实微秒数仍需正式评测环境确认。

## 3. v5 算法选择

最终阈值为：

```cpp
TWO_SERVER_ALGO_MIN_BYTES = 64KB
TAIL_READ_REDUCE_MAX_BYTES = 1MB
SMALL_TREE_ALLREDUCE_BYTES = 512KB
HIERARCHICAL_ALGO_MIN_BYTES = TAIL_READ_REDUCE_MAX_BYTES + 1
```

算法选择顺序：

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

总体分支图：

```text
                              ExecOp
                                 |
                     validate count/type/buffer
                                 |
                 +---------------+---------------+
                 | rankSize == 1                 | rankSize > 1
                 v                               v
          LocalCopy(input, output)       16 Rank / 2 Server ?
                                                 |
                          +----------------------+----------------------+
                          | yes                                         | no
                          v                                             v
                 totalBytes == 512KB ?                         FlatAllGatherReduce
                    | yes          | no
                    v              v
             SmallTree       totalBytes > 1MB ?
             AllReduce          | yes       | no
                                v           v
                         Hierarchical   totalBytes >= 64KB ?
                         3-slot            | yes       | no
                                           v           v
                                      TwoServer    FlatAllGatherReduce
                                      2-slot
```

具体覆盖关系：

| 数据量 | 最终路径 |
|---|---|
| 4B | flat allgather + fixed local reduce |
| 512KB | one-slot binary-tree reduce + reverse-tree broadcast |
| 512MB | 3-slot hierarchical |
| 400MB+4B | 3-slot hierarchical 主体 + 2-slot 小尾块 |

## 4. 512KB 二叉树快路径

### 4.1 触发条件与缓冲区

快路径只在比赛拓扑和精确 512KB 下触发：

```cpp
rankSize == 16 && totalBytes == 512ULL * 1024ULL
```

因此 512MB、400MB+4B 和其他大小不会进入该分支。每个 Rank 只使用一个 128B 对齐的 CCL 结果区：

```text
result = AlignUp(local CCL base, 128)
LocalCopy(input -> result)
```

开始通信前会同时检查本地和 peer 的 CCL 地址及 `[offset, offset + 512KB)` 范围。

### 4.2 Rank 拓扑与完整算法图

```text
Server 0: rank 0  1  2  3  4  5  6  7
Server 1: rank 8  9 10 11 12 13 14 15

Input -> local CCL result

Reduce, stride 1:
  1 -> 0    3 -> 2    5 -> 4    7 -> 6
  9 -> 8   11 -> 10  13 -> 12  15 -> 14

Reduce, stride 2:
  2 -> 0    6 -> 4   10 -> 8   14 -> 12

Reduce, stride 4:
  4 -> 0   12 -> 8

Reduce, stride 8, cross-server:
  8 -> 0

                    rank 0: final result

Broadcast, stride 8, cross-server:
  0 -> 8

Broadcast, stride 4:
  0 -> 4    8 -> 12

Broadcast, stride 2:
  0 -> 2    4 -> 6    8 -> 10   12 -> 14

Broadcast, stride 1:
  0 -> 1    2 -> 3    4 -> 5    6 -> 7
  8 -> 9   10 -> 11  12 -> 13  14 -> 15

local CCL result -> Output
```

归约树共 15 条边，广播树共 15 条边。关键路径为 4 层归约加 4 层广播；跨 Server 只包含一次归约和
一次广播，不再让每个 Rank 分别执行跨 Server 整消息规约。

### 4.3 单条归约边的同步

接收方先发布目标缓冲区可写，发送方再执行远端融合规约：

```text
receiver: ChannelNotifyRecord(ACK)
sender  : ChannelNotifyWait(ACK)
sender  : WriteReduceWithNotify(result -> remote result, DATA_SIGNAL)
receiver: ChannelNotifyWait(DATA_SIGNAL)
```

`WriteReduceWithNotify` 同时完成远端规约和完成通知，本轮仿真参数为 FP32 sum。树的下一层与上一层使用
同一 thread 顺序下发，所以下一级发送者一定先等到自己的 partial 完成。

### 4.4 单条广播边的同步

当前 VM 不会为 `HcommWriteWithNotifyOnThread` 生成完整的广播发送任务，因此使用显式序列：

```text
sender  : HcommWriteOnThread(result -> remote result)
sender  : HcommChannelFenceOnThread()
sender  : HcommChannelNotifyRecordOnThread(TREE_BROADCAST)
receiver: HcommChannelNotifyWaitOnThread(TREE_BROADCAST)
```

`Fence` 保证远端 512KB 数据写完后才发送广播完成通知。接收方等待完成后才能继续向下一层转发或复制输出。

### 4.5 任务数量变化

```text
                         优化前    当前实现
task metadata              768       216
checker nodes              769       217
data tasks                 192        62
semantic bytes      33,554,432 25,165,824
```

62 个数据任务由 32 个输入/输出本地 Copy、15 个远端 Reduce 和 15 个远端 Broadcast Copy 组成。
checker 总节点下降约 71.8%，直接降低 512KB 的 task 下发和调度固定开销。

## 5. 大消息 3-slot 分层算法

### 5.1 总体数据流图

```text
full input
    |
    +-- split into <= 139,810,048-byte slices
            |
            v
      LocalCopy(slice -> slot 0)
            |
            v
      intra-server reduce-scatter
      7 peers parallel read only my owned 1/8 chunk
            |
            v
      deterministic local reduce -> slot 1 owned chunk
            |
            v
      cross-server exchange with same local-id rank
      exchange and reduce only the owned 1/8 chunk
            |
            v
      intra-server allgather
      7 peers parallel read their final owned chunks
            |
            v
      complete result in slot 1 -> output slice
            |
            +-- remaining tail <= 1MB ?
                    |
                    v
             single-thread 2-slot ReadReduce tail path
```

### 5.2 slot 布局

每个普通大分片使用 3 个 128B 对齐 slot：

```text
slot 0: input slot
slot 1: result slot
slot 2: temp slot
```

slot 起点统一为：

```cpp
alignedBaseOffset = GetAlignedBufferOffset(buffer, 128)
slotOffset = alignedBaseOffset + slotIdx * AlignUp(sliceBytes, 128)
```

本地和每个远端 CCL buffer 分别计算 `alignedBaseOffset`，不能假设所有 Rank 的 CCL base 相同或天然对齐。

默认 400MB CCL buffer 下：

```text
max slice bytes = 139,810,048 bytes
                ~= 133.33MiB
```

512MB 因此只需要 4 个分片，而不是 v4 的约 12 个分片。

### 5.3 Server 内 reduce-scatter

每个分片按 local rank 切成 8 个 chunk：

```cpp
baseCount = sliceCount / 8
remainder = sliceCount % 8
```

每个 Rank 只拉取自己负责的 chunk：

1. 先将完整输入分片复制到 input slot。
2. 从同 Server 的 7 个 peer 读取本 Rank 负责的 chunk。
3. 7 个读取写入 temp slot 的互不重叠区域。
4. 读取完成后，按 source rank 递增顺序规约到 result slot 的 owned chunk。

Server 内网络读取量由 `7S` 降为：

```text
7 * (S / 8) = 7S / 8
```

### 5.4 跨 Server 交换

每个 Rank 只与另一台 Server 的同 local-id Rank 配对，只交换 owned chunk：

```text
cross peer = (myRank + 8) % 16
cross bytes = S / 8
```

双方先把远端 partial 读取到本地 temp slot。channel 的 DATA_SIGNAL wait 保证两边读取都完成后，才允许任一侧
修改自己的 result chunk，避免对同一 CCL 区间同时读写。

Server 0 partial 始终作为第一个操作数，Server 1 partial 作为第二个操作数，使跨 Server 浮点规约顺序一致。

### 5.5 Server 内 allgather

跨 Server 规约完成后，每个 Rank 从同 Server 的 7 个 peer 读取其最终 owned chunk，直接写到 result slot
对应的最终偏移。

不同 local rank 的 chunk 长度可能相差一个元素，因此 v5 增加 `VariableReadPlan`，每个 peer 独立记录：

```text
remote offset
local address
bytes
```

allgather 的 Server 内读取量为：

```text
7S / 8
```

### 5.6 总通信量

每 Rank 主路径网络通信量变为：

```text
reduce-scatter: 7S / 8
cross exchange:  S / 8
allgather:       7S / 8
total:          15S / 8 = 1.875S
```

与 v4 的 `8S` 相比，理论网络读取量下降：

```text
1 - (15 / 8) / 8 = 76.5625%
```

跨 Server 数据量由 `S` 降为 `S/8`，下降 87.5%。

## 6. 多 thread 并行与 notify

### 6.1 thread 分工

Host 侧资源申请保持 v4 不变：16 Rank 下共申请 15 个 AICPU_TS thread。

```text
threads[0]    : coordinator / Host-AICPU synchronization
threads[1..] : channel workers
```

worker 映射：

```cpp
workerIdx = channelIdx % (threads.size() - 1) + 1
```

7 个 Server 内 peer 在同一阶段映射到不同 worker，可并行执行 read、fence 和 channel notify。

### 6.2 阶段同步

通信阶段开始前：

```text
main record worker notify 0
worker wait notify 0
```

通信阶段结束后：

```text
worker record main notify workerGlobalIndex-1
main wait main notify workerGlobalIndex-1
```

### 6.3 为什么 notify 必须固定映射

开发过程中曾按“worker 在当前阶段 sync vector 中的位置”选择 main notify：

```text
mainNotifyIdx = currentVectorPosition - 1
```

该方案在单阶段 checker 中可以通过，但 local 阶段有 7 个 worker、cross 阶段只有 1 个 worker，导致不同物理
worker 在不同阶段复用同一个 main notify 0。任务图中出现：

```text
main waits worker completion
worker waits next main start
next main start is queued after the main wait
```

形成闭环。v5 最终为每个物理 worker 固定绑定：

```cpp
mainNotifyIdx = workerGlobalIndex - 1
```

该索引不随阶段的 peer 数量或 vector 顺序变化。

## 7. 尾块并发修复

大消息主体使用 worker thread，但尾块 `ReadReduce` 的所有操作都写同一个 result slot。

第一次试验直接复用并行 entries，checker 报告：

```text
Memory conflict detected
multiple TaskReduce write [result tail range]
```

最终实现同时构建两组 entries：

```text
main path entries : worker threads
tail path entries : threads[0]
```

尾块继续串行规约，并保留 v4 的稳定 partial 发布步骤。这样同时满足：

- 主体链路并行。
- 尾块固定规约顺序。
- result slot 无并发写。
- cross peer 只读取稳定的 input slot partial。

## 8. v4 与 v5 对比

| 项目 | v4 two-server | v5 hierarchical |
|---|---:|---:|
| CCL slot 数 | 9 | 3 |
| 默认最大分片 | 约 44.4MiB | 约 133.33MiB |
| 512MB 分片数 | 约 12 | 4 |
| Server 内 reduce 通信 | `7S` | `7S/8` |
| 跨 Server 通信 | `S` | `S/8` |
| Server 内 allgather | 0 | `7S/8` |
| 每 Rank 总网络读取 | `8S` | `15S/8` |
| 本地规约数据量 | 约 `8S` | 约 `S` |
| Server 内 channel | 单 thread 串行 | 多 worker 并行 |

测试点 6、7 仍使用上述 3-slot 分层算法。测试点 5 的精确 512KB 快路径在进入 slot 数量选择前返回，
不会改变大消息的 slot 布局、分片数量、worker 映射或 notify 时序。

512KB 路径对比：

| 项目 | 原 2-slot ReadReduce | 当前二叉树路径 |
|---|---:|---:|
| CCL 结果区 | 2 slot | 1 slot |
| 整消息归约边 | 每 Rank 8 次 | 全局 15 条树边 |
| 跨 Server 归约 | 每 local-id 一次 | 全局一次 |
| 广播 | 隐含在每 Rank 全量规约 | 15 条反向树边 |
| checker nodes | 769 | 217 |

## 9. VM 构建与替换

构建命令：

```bash
docker exec hccl-vm bash -lc 'source /etc/profile.d/hccl-vm.sh; \
cd /home/workspace/hccl_allreduce_problem_222_template; \
bash build.sh'
```

最终 Release 构建成功，无编译 warning/error。

替换库：

```bash
cp -f build/include/hccl.h ${ASCEND_HOME_PATH}/x86_64-linux/include/hccl/
cp -f build/lib64/libhccl.so ${ASCEND_HOME_PATH}/x86_64-linux/lib64/
cp -f build/lib64/libhccl_device.so ${HCCL_VM_INSTALL_DIR}/lib/aarch64/
```

最终构建产物和 VM 安装目录中的 device 库哈希一致：

```text
b0114cdf8b649c0c587a0d1432bd2263d1d5b3571e4def105de11b6e03c666e4
```

## 10. VM 验证结果

### 10.1 验证环境

```text
container : hccl-vm
topology  : ascend950_cluster_4_server_competition.yaml
comm meta : 128.yaml, 2 servers * 8 ranks
rank size : 16
data type : fp32
reduce op : sum
CANN      : 9.1.0
expansion : AI_CPU
device so : b0114cdf8b649c0c587a0d1432bd2263d1d5b3571e4def105de11b6e03c666e4
```

每个独立用例前重新执行：

```bash
hccl-vm mock-comm 128
```

checker 命令：

```bash
hccl-vm plugin run @checker
```

### 10.2 512KB 二叉树快路径

normal 模式命令：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 524288 -e 524288 -d fp32 -o sum -w 0 -n 1 -c 1 -p 8
```

normal 模式已安装并确认 `runner` 与 `checker` 均为 `RUNNING`。数值验证结果：

```text
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
524288            | 1000.00        | 0.52429              | success
```

其中 `check_result: success` 是有效的 FP32 sum 正确性结论；`1000 us` 是 HCCL-VM 事件计时桩的固定值，
不是算法真实耗时，也不能用于推导真实性能。

在不安装 runner 的干净 normal 会话中以 `-c 0` 生成任务图，再执行 checker，结果为：

```text
task metadata : 216
checker nodes : 217
data tasks    : 62
semantic bytes: 25,165,824
checker       : Success
```

checker 的 SingleTaskCheck、MemConflict 和 SemanticCheck 三个阶段均为 `status=success`，没有 local
record/wait 卡死、channel deadlock 或 CCL 内存冲突。

### 10.3 一个最大分片加 4B 尾块

3-slot 最大分片为：

```text
139,810,048 bytes
```

构造测试大小：

```text
139,810,052 bytes = one full slice + 4 bytes
```

normal 模式使用 `-c 0`，只生成任务图：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 139810052 -e 139810052 -d fp32 -o sum -w 0 -n 1 -c 0 -p 8
```

最终结果：

```text
task metadata : 3,112
checker nodes : 3,113
data tasks    : 616
semantic bytes: 10,905,183,872
checker       : Success
```

该用例覆盖：

- 3-slot 主路径。
- 多 worker Server 内通信。
- cross exchange。
- allgather。
- 主路径结束后的 4B 单 thread 尾块。
- worker notify 与 coordinator notify 的跨阶段复用。

### 10.4 512MB 原始比赛用例

normal 模式下 16 Rank 的大输入会占用大量独立共享内存；check-only 使用公共通信池复用内存。因此本用例
使用 check-only，只验证任务图：

```bash
./hccl-vm start ascend950_cluster_4_server_competition.yaml --check-only
hccl-vm mock-comm 128
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 536870912 -e 536870912 -d fp32 -o sum -w 0 -n 1 -c 0 -p 8
hccl-vm plugin run @checker
```

结果：

```text
slice structure: 4 hierarchical slices
task metadata  : 9,440
checker nodes  : 9,441
data tasks     : 1,696
semantic bytes : 23,611,131,392
checker        : Success
```

该结果验证连续 4 个分片复用相同 channel notify 和 thread notify 时没有死锁或内存冲突。

### 10.5 400MB+4B 原始比赛用例

```bash
hccl-vm mock-comm 128
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 419430404 -e 419430404 -d fp32 -o sum -w 0 -n 1 -c 0 -p 8
hccl-vm plugin run @checker
```

按当前最大分片计算，该用例实际拆分为：

```text
3 * 139,810,048 bytes + 260-byte tail
```

结果：

```text
task metadata : 7,800
checker nodes : 7,801
data tasks    : 1,464
semantic bytes: 19,853,035,136
checker       : Success
```

### 10.6 engine context 重复使用

使用 2MB hierarchical 消息执行 1 次 warmup 和 2 次正式迭代：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  ${ASCEND_HOME_PATH}/tools/hccl_test/bin/all_reduce_test \
  -b 2097152 -e 2097152 -d fp32 -o sum -w 1 -n 2 -c 0 -p 8
```

checker 识别到 3 个 op group，并处理到：

```text
op[2] Checker Success
```

过程中没有 local notify matching、channel deadlock 或内存冲突错误，验证 thread notify 计数可以跨算子迭代复用。

### 10.7 验证汇总

| 用例 | VM 模式 | 最终路径 | Task / node | 数值或 Checker |
|---|---|---|---:|---|
| 512KB | normal + runner, `-c 1` | binary-tree reduce + broadcast | 216 / 217 | `check_result: success`, Checker Success |
| 139,810,052B | normal, `-c 0` | one full slice + 4B tail | 3,112 / 3,113 | Checker Success |
| 512MB | check-only, `-c 0` | 4 hierarchical slices | 9,440 / 9,441 | Checker Success |
| 400MB+4B | check-only, `-c 0` | 3 hierarchical slices + 260B tail | 7,800 / 7,801 | Checker Success |
| 2MB, warmup 1 + iter 2 | check-only, `-c 0` | repeated hierarchical | - | `op[2]` Checker Success |

本轮修改前后，测试点 6、7 的任务数量分别保持 `9,440 / 9,441` 和 `7,800 / 7,801`，证明精确
512KB 分支没有改变大消息任务图。

## 11. VM 环境现象与结论边界

### 11.1 normal runner 使用条件

数值验证必须在 normal 模式安装 runner，并在测试前确认：

```text
checker  RUNNING
runner   RUNNING
```

当前 512KB 已按该流程验证为 `check_result: success`。未安装 runner 时，VM 只生成任务而不执行通信，
此时 `-c 1` 读取未更新输出产生的失败不能用于判断算法正确性。

### 11.2 大消息共享内存限制

本轮最初在 normal 模式运行 512MB 时，容器的 8GB `/dev/shm` 达到 100%，测试进程在输入复制阶段触发：

```text
Signal: Bus error (7)
No sync records were found
```

该错误发生在 AllReduce 算子记录生成前，不是当前通信算法报错。退出 VM 后共享内存恢复为空，再按文档使用
`--check-only` 运行 512MB 和 400MB+4B，两项 checker 均成功。check-only 结果只证明任务图、同步、内存访问
和规约语义正确，不能替代大消息 runner 数值执行。

### 11.3 VM 时间不代表设备性能

HCCL-VM 的事件计时为 stub。当前 512KB 输出中的：

```text
aveg_time = 1000 us
```

不是实际链路、AICPU/TS 下发或 DMA 执行时间，`alg_bandwidth` 也是由该固定时间计算的派生值。因此本次可以
确认任务节点由 769 降到 217，但不能仅凭 VM 宣称已达到 60-100 us。

真实延迟、相对加速比和比赛排名必须以正式评测环境或真实 Ascend 设备为准。

## 12. 最终结论

当前实现同时保留大消息优化并增加精确 512KB 低启动开销路径：

- 512KB 使用 4 层二叉树归约和 4 层反向树广播，跨 Server 每个方向只有一条树边。
- 512KB task metadata 从 768 降到 216，checker nodes 从 769 降到 217，下降约 71.8%。
- 512KB FP32 sum 已通过 normal runner 数值验证及 checker 全阶段验证。
- 大消息保持 3-slot hierarchical 算法，每 Rank 网络读取量约为 `15S/8`，跨 Server 数据量为 `S/8`。
- 512MB 仍为 4 个 hierarchical 分片，400MB+4B 仍为 3 个主体分片加 260B 尾块。
- 测试点 6、7 的 checker 节点数与修改前一致，均无死锁、内存冲突或语义错误。
- 尾块继续使用单 thread，worker notify 继续使用稳定全局索引。

当前最需要正式评测确认的是 512KB 的真实耗时。代码结构已经显著减少固定任务开销，但 100 us 以内以及
进一步的 60-80 us 目标不能由 VM 模拟时间证明。

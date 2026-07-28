# AllReduce 716v9 Striped WriteReduce Mesh 优化实现与统计

## 1. 版本定位

本文记录 2026-07-16 在 v8 基础上完成的 AllReduce v9 大消息优化。工程目录：

```text
/data/mingwei/hccl_allreduce_problem_222_template
```

容器内映射目录：

```text
/home/workspace/hccl_allreduce_problem_222_template
```

v9 基于以下 v8 提交继续开发：

```text
18447c1 Optimize large-message allreduce mesh push
```

算法源码仍只修改选手实现文件：

```text
op_kernel_aicpu/exec_op.cc
```

精确 512KiB 的 small hierarchical 路径、小消息路径、跨 Server 配对规约和最终 Server 内 allgather 均保持
v8 设计。本轮只替换 16 Rank、消息大于 1MiB 路径中的 Server 内 reduce-scatter 阶段。

算子入口仍不显式检查 `sendBuf`、`recvBuf` 是否按 4 Byte 对齐。比赛接口已要求 FP32 地址满足 4 Byte 对齐，
v9 直接依赖该前置条件；CCL buffer 内部仍按 128B 对齐使用。

## 2. v8 大消息路径的剩余瓶颈

v8 已经取消完整输入和输出 staging，并将测试点 7 合并为一个主分片，但 Server 内规约仍需要保存 7 份远端
contribution。设一个消息分片大小为 `S`，每个 owner 负责的 chunk 为 `C = S/8`，v8 的主要布局为：

```text
recvBuf owned chunk : self contribution
CCL slot 0..6       : 7 remote contributions
```

远端 contribution 到齐后执行固定规约树：

```text
8 contributions -> 4 partials -> 2 partials -> 1 Server partial
```

因此 v8 还存在以下瓶颈：

1. 每 Rank 需要约 `7C = 7S/8` 的 CCL 空间保存尚未规约的远端 contribution。
2. Server 内完成一个 owner chunk 需要 7 个整 chunk 本地规约任务，规约关键路径包含 3 个顺序层级。
3. 512MiB 在约 400MiB CCL buffer 下仍需拆成两个大分片，重复跨 Server 和 allgather 阶段。
4. 逐 channel 的完整 ACK、DATA_SIGNAL 双握手会放大多轮算法的控制任务数。

v9 的目标是在保持 Full-Mesh 链路并行度和确定性的前提下，把网络传输与 Server 内规约融合，并进一步降低
CCL 占用。

## 3. v9 算法概览

v9 将 Server 内阶段改为 deterministic striped WriteReduce Mesh：

```text
owner self chunk --local copy--> owner CCL aggregate

7 rounds:
    each source --HcommWriteReduceOnThread--> peer owner aggregate stripe
    server-wide star barrier

owner CCL aggregate = one Server partial chunk
```

每个 owner chunk 被划分成 7 个近似均匀的 stripe。7 个远端 source 在同一轮写入不同 stripe，因此不存在并发
写同一地址；轮与轮之间使用 Server 级 barrier 建立严格全序。7 轮结束后，每个 stripe 都恰好规约了其余 7 个
source contribution。

后续步骤沿用 v8：

```text
Server partial
    -> cross-server paired read and canonical local reduce
    -> local owned result
    -> direct allgather into recvBuf
```

## 4. owner chunk 与 stripe 划分

消息仍按元素数分给 8 个 owner。不能整除时，低编号 owner 多一个元素：

```cpp
base      = sliceCount / 8;
remainder = sliceCount % 8;
chunk[i]  = base + (i < remainder ? 1 : 0);
```

每个 owner chunk 再按相同方法分为 7 个 stripe：

```cpp
base      = chunkCount / 7;
remainder = chunkCount % 7;
stripe[i] = base + (i < remainder ? 1 : 0);
```

因此 v9 同时支持：

- 512MiB 的均匀 8-owner 布局。
- 400MiB+4B 中 owner 0 多一个 FP32 元素的非均匀布局。
- owner chunk 不能整除 7 时的非均匀 stripe 布局。

所有 offset 和传输 count 都以 FP32 元素边界计算，不要求消息字节数能被 8、7 或 128 整除。

## 5. 确定性 stripe 轮转

对 source local rank `s` 和 owner local rank `o`，先得到该 source 在 owner 的 7 个远端 source 中的 packed index：

```cpp
packedSourceIdx = s < o ? s : s - 1;
```

第 `round` 轮写入的 stripe 为：

```cpp
stripeIdx = (packedSourceIdx + round) % 7;
```

对固定 owner 和固定 round，7 个远端 source 的 `packedSourceIdx` 是 `0..6` 的排列，因此目标 stripe 也是
`0..6` 的排列，同轮不会有两个 source 写同一 stripe。

对固定 owner 和固定 stripe，7 轮中的 source 顺序也是唯一且固定的。owner 自己的 contribution 在第 0 轮前
初始化 aggregate，随后每轮只追加一个确定来源的 contribution，所以浮点规约顺序不依赖任务到达先后。

不同 stripe 的 source 顺序可以不同，但每个 stripe 的顺序在每次执行中固定。最终 chunk 只在 owner 上形成一次，
再通过 allgather 分发给所有 Rank，因此所有 Rank 获得同一结果。

## 6. WriteReduce 融合

v8 的 Server 内数据路径是：

```text
remote source --write--> contribution slot
contribution slot --local reduce tree--> Server partial
```

v9 改为：

```text
remote source --HcommWriteReduceOnThread--> owner aggregate stripe
```

每条 source-owner channel 总传输字节数仍为一个 owner chunk `C`，只是被拆为 7 个 stripe task。对一个 Rank：

```text
7 peers * C = 7S/8
```

Server 内理论网络量没有增加，同时不再需要：

- 7 个 remote contribution slot。
- 7 次整 chunk 的本地规约。
- 4 -> 2 -> 1 本地规约树的三层同步。

## 7. 星型 Server barrier

只保留逐 channel 的一组完成信号不足以建立不同 channel 之间的全序，下一轮某个 channel 可能先于上一轮另一个
channel 前进。v9 使用每个 Server 的 local-rank 0 作为 barrier leader：

```text
local rank 1..7 --arrival notify--> local rank 0
local rank 0 waits all arrivals
local rank 0 --release notify--> local rank 1..7
```

barrier 使用 channel notify index 2，不与现有 ACK 和 DATA_SIGNAL 资源冲突。执行位置为：

1. owner aggregate 初始化完成后、第一轮 WriteReduce 前。
2. 每轮所有 worker thread 完成 WriteReduce 和 channel fence 后。

每个 Server 每次 barrier 只需要 7 个 arrival 和 7 个 release 的 record/wait 配对。相比 8 Rank 之间逐 peer
执行完整 ACK、DATA_SIGNAL 双握手，控制图明显缩小，同时 barrier 对全部 channel 建立统一轮次边界。

## 8. CCL 布局与单分片容量

v9 每 Rank 只需要两个 owner chunk 大小的 CCL 区域：

```text
slot 0: Server aggregate / Server 0 final chunk
slot 1: cross-server temporary / Server 1 final chunk
```

因此 CCL 主布局从 v8 的 `7C` 降为 v9 的 `2C`：

```text
v8: 7S/8
v9: 2S/8 = S/4
```

约 400MiB CCL buffer 下，单 slot 上限约为 200MiB，乘以 8 个 owner 后，一个大分片可覆盖约 1600MiB。比赛
中的两个大消息均可一次处理：

| 测试点 | 数据量 | v8 | v9 |
|---:|---:|---|---|
| 6 | 512MiB | 2 个大分片 | 1 个大分片 |
| 7 | 400MiB+4B | 1 个非均匀大分片 | 1 个非均匀大分片 |

测试点 6 的收益同时来自取消本地规约树和减少分片。测试点 7 的收益主要来自 WriteReduce 融合，分片数保持为 1。

## 9. 通信量分析

对可均匀划分的消息 `S`，每 Rank 网络量为：

| 阶段 | 每 Rank 数据量 |
|---|---:|
| Server 内 striped WriteReduce reduce-scatter | `7S/8` |
| 跨 Server partial 交换 | `S/8` |
| Server 内 direct allgather | `7S/8` |
| 合计 | `15S/8` |

v9 与 v8 的理论网络字节数相同。性能提升来自：

- 网络写和 FP32 sum 规约融合。
- 取消 7 份 contribution 的写入、读取和本地规约树。
- 测试点 6 从两个分片合并为一个分片。
- 使用星型 barrier 控制 7 个 stripe round 的同步开销。

## 10. Checker 任务图

最终 v9 的两个大消息用例都是单分片，任务图结构相同：

| 用例 | checker nodes | data tasks | parallel candidate pairs | 结果 |
|---|---:|---:|---:|---|
| 512MiB | 5,633 | 960 | 0 | Checker Success |
| 400MiB+4B | 5,633 | 960 | 0 | Checker Success |

完整 Checker V3 阶段均成功：

```text
SingleTaskCheck : success
MemConflict     : success
SemanticCheck   : success
```

两个用例均有 `8,152` 个内存重叠候选，全部被证明有序：

```text
overlapCandidatePairs  = 8152
orderedCandidatePairs  = 8152
parallelCandidatePairs = 0
```

v9 的 data task 数高于 v8，因为每条 peer channel 的一个大 contribution 被拆为 7 个 WriteReduce stripe task。
任务数不能单独作为性能结论：v9 的 data task 同时完成传输和规约，而 v8 还需要 contribution 落盘后的本地规约树。

## 11. normal runner 数值验证

最终星型 barrier 版本完成了非均匀小规模大消息路径验证：

```text
2MiB+4B FP32 sum: check_result = success
```

该用例覆盖：

- 消息元素数不能整除 8。
- owner chunk 不能整除 7。
- 7 轮 WriteReduce。
- 同一次调用内 8 次星型 barrier notify 复用。
- 跨 Server canonical reduce 和 direct allgather。

512MiB 和 400MiB+4B 已在 check-only 模式完成完整任务图验证。normal runner 不执行这两个完整尺寸，因为 16 Rank
同时分配输入、输出和通信共享内存会超过当前容器 8GiB `/dev/shm`。

HCCL-VM 的 event elapsed time 是固定 stub，VM 输出的 `1000 us` 不能作为性能数据。

## 12. 实际性能结果

当前 v9 的实际测试点成绩为：

| 测试点 | 数据量 | v8 实测 | v9 实测 | 相对 v8 时延下降 |
|---:|---:|---:|---:|---:|
| 5 | 512KiB | 59.00 μs | 59.00 μs | 0% |
| 6 | 512MiB | 3.72 ms | 3.25 ms | 12.6% |
| 7 | 400MiB+4B | 2.89 ms | 2.56 ms | 11.4% |

测试点 5 沿用 v8 未改动的 small hierarchical 路径。测试点 6、7 使用 v9 striped WriteReduce Mesh。

相对 v7 文档中的旧基线：

```text
测试点 6: 4.78 ms -> 3.25 ms，时延下降约 32.0%
测试点 7: 3.78 ms -> 2.56 ms，时延下降约 32.3%
```

上述 `59.00 μs / 3.25 ms / 2.56 ms` 来自实际评测记录，不是 HCCL-VM 固定 event stub 的输出。

## 13. 构建状态

Release 构建成功，最终 device 库：

```text
build/lib64/libhccl_device.so
```

SHA-256：

```text
e725199483e18fd4c6659e275e6d74ebf81060d822ab0d8a8eaf3eee43928a6f
```

构建产物与 HCCL-VM 安装目录中的 `libhccl_device.so` 哈希一致，`git diff --check` 通过。

## 14. 结论

v9 在 v8 7-contribution Mesh push 的基础上，将 Server 内阶段改为 7-stripe deterministic WriteReduce Mesh：

```text
CCL 主布局                 : 7S/8 -> S/4
512MiB 分片数              : 2 -> 1
400MiB+4B 分片数           : 1 -> 1
Server 内本地整 chunk reduce: 7 -> 0
每 Rank 理论网络量          : 保持 15S/8
```

星型 Server barrier 为 7 个 stripe round 建立统一的跨 channel 顺序，并将最终任务图控制在 5,633 个节点。最终
512MiB 与 400MiB+4B 均为 `Checker Success`，2MiB+4B normal runner 数值正确。

当前实际测试点 5、6、7 分别为：

```text
59.00 μs
3.25 ms
2.56 ms
```

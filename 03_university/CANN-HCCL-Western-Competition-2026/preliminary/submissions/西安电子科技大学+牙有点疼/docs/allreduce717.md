# AllReduce 717：4-stage 全 Clos 小消息优化记录

## 1. 最重要结论

本轮只修改精确 `512KiB`、16 Rank、FP32 sum 的测试点 5 路径，测试点 6/7 的大消息算法保持不变。

正式评测结果为：

| 测试点 | 数据量 | 上一正式结果 | 当前正式结果 | 变化 |
|---:|---:|---:|---:|---:|
| 5 | 512KiB | 59.00 μs | **44.00 μs** | 降低 15.00 μs，约 25.4% |
| 6 | 512MiB | 2.26 ms | **2.26 ms** | 不变 |
| 7 | 400MiB+4B | 1.79 ms | **1.79 ms** | 不变 |

当前成绩：

```text
测试点 5: 44.00 μs
测试点 6: 2.26 ms
测试点 7: 1.79 ms
```

测试点 5 距离 `39 μs` 目标还差 `5 μs`，约需继续降低 `11.4%`。

## 2. 修改范围

本轮算法源码只修改比赛允许的文件：

```text
op_kernel_aicpu/exec_op.cc
```

具体只替换 `RunSmallHierarchicalAllReduce()` 的精确 512KiB 实现，并删除不再使用的
`RunSmallPairRead()` helper。Host 资源申请、公共结构和测试点 6/7 路径均未修改。

Git 基线为：

```text
0977e2f Optimize allreduce v11 scheduling
```

## 3. 历史路径对比

本轮前有两个重要小消息版本。

### 3.1 59 μs 分层基线

```text
3-stage Server 内 recursive-halving ReduceScatter
1-stage 相同 local-id 的跨 Server Reduce
3-stage Server 内 recursive-doubling AllGather
```

指标：

```text
串行通信阶段          7
task metadata          672
checker nodes          673
data task nodes        160
每 Rank 网络量         960KiB
  Mesh                 896KiB
  Clos                  64KiB
真实延迟                59 μs
```

该版本网络量低，但 6 个 Mesh round 和每轮完整 ACK/DATA 握手形成了较重的固定启动开销。

### 3.2 65 μs 的 8-stage all-Clos 回退

此前使用 parity-colored hypercube，把 4-stage ReduceScatter 和 4-stage AllGather 全部映射到 Clos：

```text
串行通信阶段          8
task metadata          736
checker nodes          737
data task nodes        160
每 Rank 网络量         960KiB Clos
真实延迟                65 μs
```

该实验说明：对 512KiB 场景，Clos 带宽收益不能抵消新增 round、notify、fence 和小分片事务的固定成本。
性能瓶颈首先是控制面深度，不是单纯的网络字节数。

## 4. 当前 4-stage 全 Clos 算法

### 4.1 算法结构

当前实现改为 full-message recursive-doubling：

```text
4 个逻辑超立方体维度
每个维度交换并规约完整 512KiB partial
每轮 partial 覆盖的 Rank 数依次为 2、4、8、16
```

四轮 mask 为：

```cpp
constexpr uint32_t masks[] = {8, 4, 2, 1};
```

每 Rank 网络量增加为：

```text
4 * 512KiB = 2MiB
```

但串行通信阶段由 8 降到 4，task metadata 由 736 降到 480。

### 4.2 parity-colored 全 Clos 映射

物理 Rank 布局为两个 Server，每个 Server 8 Rank。对任意一轮，peer 计算为：

```cpp
crossServerStart = (serverIdx ^ 1) * 8;
peerRank = crossServerStart + (localRank ^ (mask & 7));
```

因此四轮 peer 都位于另一个 Server：

```text
mask 8: 对端 Server 的相同 local-id
mask 4: 对端 Server 的 local-id ^ 4
mask 2: 对端 Server 的 local-id ^ 2
mask 1: 对端 Server 的 local-id ^ 1
```

该映射等价于对 16 个物理 Rank 做 parity coloring，使逻辑 4 维超立方体的每条边都走题面带宽更高的 Clos。

### 4.3 双 slot ping-pong

不能直接对同一个完整 CCL slot 做 symmetric in-place ReadReduce：本 Rank 更新 slot 时，对端可能还在读取该 slot，
会产生 source/destination 数据竞争。

当前使用两个 512KiB CCL slot。每轮执行：

```text
1. LocalCopy(currentSlot -> nextSlot)
2. record ACK
3. wait peer ACK
4. ReadReduce(peer currentSlot -> local nextSlot)
5. channel fence
6. record DATA_SIGNAL
7. wait peer DATA_SIGNAL
8. currentSlot = nextSlot
```

ACK 在双方完成本地 copy 后发布，保证本轮目标 slot 已初始化；DATA_SIGNAL 保证双方读取不可变 source 完成后，
才允许下一轮复用旧 slot。该布局消除了 symmetric full-message reduce 的读写竞争。

### 4.4 通信和本地内存代价

每 Rank 当前主要数据量为：

```text
网络 ReadReduce:       4 * 512KiB = 2MiB，全部 Clos
轮间 LocalCopy:        4 * 512KiB = 2MiB
Input/Output LocalCopy: 2 * 512KiB = 1MiB
```

相比分层算法，网络量和本地 copy 增加，但 512KiB 场景的固定 round/task 开销下降更重要。正式评测从 59 μs 降到
44 μs，证明这一取舍在当前平台上成立。

## 5. 任务图对比

| 实现 | 串行通信阶段 | task metadata | checker nodes | data tasks | 每 Rank 网络量 | 正式延迟 |
|---|---:|---:|---:|---:|---:|---:|
| 59 μs 分层基线 | 7 | 672 | 673 | 160 | 960KiB，主要 Mesh | 59 μs |
| 8-stage all-Clos | 8 | 736 | 737 | 160 | 960KiB Clos | 65 μs |
| 当前 4-stage all-Clos | 4 | **480** | **481** | 160 | 2MiB Clos | **44 μs** |

当前 Checker V3 统计：

```text
task metadata             480
checker nodes             481
data task nodes           160
overlap candidate pairs   912
ordered candidate pairs   912
parallel conflict pairs   0
normal semantic count     64
normal semantic bytes     33,554,432
SingleTaskCheck           success
MemConflict               success
SemanticCheck             success
Checker                   Success
```

相对 59 μs 基线，metadata 减少 `192`，约 `28.6%`；相对 65 μs 的 8-stage all-Clos，metadata 减少 `256`，
约 `34.8%`。

## 6. 构建与 VM 验证

Release 构建成功。最终验证时构建库与 VM 安装库 SHA-256 一致：

```text
e47a029782f17d16beccf48bb4b1ae35572b9dd9f76883536d3e7825b57c265d
```

HCCL-VM normal runner 单轮：

```text
524288 | 1000.00 | 0.52429 | success
```

20 轮重复执行：

```text
524288 | 50.00 | 10.48576 | success
```

VM 中的 `1000 μs` 和 `50 μs` 来自固定 event stub，不是设备性能数据。有效结论是：

```text
单轮数值正确
20 轮 notify/channel 复用稳定
Checker 无死锁和并行内存冲突
```

真实性能只采用正式评测结果 `44.00 μs`。

## 7. 当前瓶颈判断

当前实现已经把通信深度从 7/8 轮压到 4 轮，下一阶段瓶颈预计转为：

1. 每轮仍有完整的 ACK、DATA_SIGNAL、fence，共 6 个通信控制 primitive。
2. 每轮执行一次完整 512KiB LocalCopy，四轮共 2MiB 本地 copy。
3. 每 Rank 跨 Clos ReadReduce 总量为 2MiB，已经不再是低网络字节算法。
4. `ReadReduce` 与完成通知没有融合接口，通信完成仍需 fence + record + wait。

目前不应再优先增加 worker fan-out 或额外 barrier。历史 direct owner 路径虽然减少逻辑 round，但 task metadata 增长到
1512，正式性能不如 59 μs 基线。

## 8. 下一轮优先方向

最值得以当前 44 μs 版本为基线做单变量 A/B 的方向是：在现有双 slot 安全布局上，把 symmetric pull 改为
`WriteReduceWithNotify`。

当前每轮控制序列：

```text
LocalCopy
record ACK
wait ACK
ReadReduce
fence
record DATA
wait DATA
```

候选 fused push：

```text
LocalCopy
record ACK
wait ACK
WriteReduceWithNotify(local current -> peer next, DATA)
wait DATA
```

双 slot 已确保 source 与 destination 分离，内存层面比历史 in-place symmetric push 更安全；理论上每轮可减少 fence 和
显式 DATA record，四轮共减少 8 个控制 primitive/rank，可能继续接近 39 μs。

风险是历史 symmetric `HcommWriteReduceWithNotifyOnThread()` 曾出现 Checker local notify 无法闭合。下一轮必须同时通过：

```text
Release build
normal runner 单轮
Checker V3
-w 3 -n 20 -c 1
正式环境真实性能 A/B
```

若 fused push 仍不能闭合，应保留当前 44 μs 版本，不要删除 DATA completion，也不要回到 8-stage halving-doubling。

## 9. 最终状态

本轮结论：

```text
测试点 5: 4-stage full-message recursive-doubling
拓扑映射: 4 轮全部走 Clos
内存布局: 双 512KiB slot ping-pong
同步协议: 完整 ACK + DATA pull
task metadata: 480
Checker: Success
20 轮: success
正式延迟: 44.00 μs
```

测试点 6/7 继续沿用 v11 大消息双 Lane Mesh/Clos 实现，正式结果分别保持 `2.26 ms` 和 `1.79 ms`。

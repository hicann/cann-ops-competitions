# AllReduce 715v8 大消息优化实现与统计

## 1. 版本定位

本文记录 2026-07-16 在 v7 基础上完成的 AllReduce v8 优化。工程目录：

```text
/data/mingwei/hccl_allreduce_problem_222_template
```

容器内映射目录：

```text
/home/workspace/hccl_allreduce_problem_222_template
```

v8 算法源码只修改了选手实现文件：

```text
op_kernel_aicpu/exec_op.cc
```

当前基础提交仍为：

```text
947d529 Optimize 512KB hierarchical allreduce
```

v8 基于上述提交继续开发。README、Host 代码以及其他算子实现文件均未修改。

本轮只重构 16 Rank、消息大于 1MiB 的大消息路径，目标是继续优化测试点 6、7：

| 测试点 | 数据量 | v7 | v8 |
|---:|---:|---|---|
| 6 | 512MiB | 2-slot hierarchical，3 个主分片 | 7-contribution Mesh push，2 个主分片 |
| 7 | 400MiB+4B | 2 个主分片 + 260B tail | 1 个主分片，无 tail |

精确 512KiB 的 v7 recursive-halving 路径以及小消息路径没有改变。

算子入口当前仍不显式检查 `sendBuf`、`recvBuf` 是否按 4 Byte 对齐。比赛接口已经要求 FP32 地址满足
4 Byte 对齐，本实现直接依赖该接口前置条件；CCL buffer 内部仍按 128B 对齐使用。

## 2. v7 大消息路径的主要瓶颈

v7 对测试点 6、7 沿用 v6 的双完整 slot 布局。设一个大消息分片大小为 `S`：

```text
slot 0: 完整 S 字节输入
slot 1: packed contributions / 完整 S 字节结果
```

执行过程包含：

```text
user input --完整 S copy--> CCL input slot
7 个本 Server peer owned-chunk read
固定 4 -> 2 -> 1 本地规约树
跨 Server 同 local-id Rank 交换 S/8
7 个本 Server peer chunk read
CCL result slot --完整 S copy--> user output
```

因此旧路径有四个直接瓶颈：

1. 每个分片开始时完整复制 `S`，结束时再完整复制 `S`，额外本地搬运至少为 `2S`。
2. CCL memory 必须同时容纳两个完整 slot，约 400MiB CCL buffer 只能处理约 200MiB 的主分片。
3. 测试点 6 被拆成 3 个主分片，测试点 7 被拆成 2 个主分片和一个 260B tail；每个分片都重复完整同步协议。
4. 任务下发、notify 和 channel fence 数量随分片数近似线性增长，数据面并行能力没有抵消控制面的重复开销。

v8 不改变大消息路径的理论网络量，主要优化 CCL 占用、本地整消息 copy 和分片数。

## 3. v8 算法选择

当前入口选择顺序为：

```cpp
if (rankSize == 16 && totalBytes == 512KiB) {
    RunSmallHierarchicalAllReduce(...);
} else if (rankSize == 16 && totalBytes > 1MiB) {
    RunHierarchicalTwoServerAllReduce(...);  // v8 large Mesh push
} else if (rankSize == 16 && totalBytes >= 64KiB) {
    RunTwoServerAllReduce(...);
} else {
    RunFlatAllGatherReduce(...);
}
```

大消息路径仍按“每个 Server 8 Rank、共 2 个 Server”的固定拓扑设计：

```text
Server 0: rank 0..7
Server 1: rank 8..15
localRank = myRank % 8
crossPeer = (myRank + 8) % 16
```

每个分片按元素数分给 8 个 owner。若元素数不能整除 8，低编号 owner 多一个元素：

```cpp
base      = sliceCount / 8;
remainder = sliceCount % 8;
chunk[i]  = base + (i < remainder ? 1 : 0);
```

这使 `400MiB + 4B` 的最后一个 FP32 元素可以直接落入 owner 0 的 chunk，不再产生小 tail。

## 4. 七贡献 CCL 布局

设某个 owner 的 chunk 为 `C`。每个 Rank 最终需要规约来自本 Server 8 个 source Rank 的同一 chunk：

```text
source 0 contribution
source 1 contribution
...
source 7 contribution
```

v8 将 owner 自己的 contribution 暂存在本 Rank `recvBuf` 的 owned chunk 中；其余 7 份 contribution 紧凑存入
本 Rank CCL memory：

```text
recvBuf owned chunk: self contribution

CCL range 0: remote source contribution
CCL range 1: remote source contribution
...
CCL range 6: remote source contribution
```

对 owner `o`，source `s != o` 的 packed index 为：

```cpp
packedIndex = s < o ? s : s - 1;
```

例如 owner 3 的布局为：

```text
CCL index : 0  1  2  3  4  5  6
source    : 0  1  2  4  5  6  7
recvBuf   : source 3
```

所以每 Rank 的主布局只需要：

```text
7 * C = 7S/8                 // 可整除时
7 * ownerChunkBytes          // 一般情况
```

而不是 v7 的 `2S`。CCL 开头的第一个 chunk 在 Server 内规约后复用为 Server partial，第二个 chunk 可复用为跨
Server 临时区，不额外申请完整 slot。

## 5. Server 内直接 Mesh push reduce-scatter

v7 先把完整输入复制到本地 CCL，再由每个 owner 从 7 个 peer 拉取自己的 chunk。v8 去掉完整输入 staging，改为
每个 source 直接从用户输入向本 Server 其他 7 个 owner 的 CCL memory 写入对应 chunk：

```text
source s input chunk[o]
    -- HcommWriteOnThread -->
owner o CCL packed contribution[s]
```

每个 Rank 同时具有两个角色：

- source：向 7 个远端 owner 推送各自负责的 chunk。
- owner：在自己的 CCL memory 接收 7 个 source contribution，并在 `recvBuf` 保存自己的 contribution。

所有 8 个 owner 覆盖整个分片，因此这一阶段等价于 Server 内 reduce-scatter 的数据分发。每 Rank 的 Server 内
网络量仍为：

```text
7S/8
```

但不再有 `input -> full CCL slot` 的 `S` 字节本地 copy。

## 6. 确定性本地规约树

owner 收齐 8 份 contribution 后，将地址按 source local rank `0..7` 排列，再使用固定三层树：

```text
level 1, stride 1: (0,1) (2,3) (4,5) (6,7)
level 2, stride 2: (0,2) (4,6)
level 3, stride 4: (0,4)
```

即：

```text
8 contributions -> 4 partials -> 2 partials -> 1 Server partial
```

同一层的独立规约分配到 worker threads 并行执行，层与层之间同步。树的输入顺序始终按 source Rank 排列，因而
保持确定的浮点规约顺序，不依赖 contribution 到达先后。

最终 Server partial 被发布到本 Rank CCL base。若树根不在 CCL base，只额外复制一个 `C` 大小的 chunk，
不会复制完整分片。

## 7. 跨 Server 配对规约

两个 Server 中 local id 相同的 Rank 互为 cross peer。双方从对端 CCL base 读取一个 Server partial 到本地第二个
chunk 临时区：

```text
rank r <-> rank r+8
payload = owned chunk C
```

随后保持全局 canonical 顺序：

```text
Server 0 partial + Server 1 partial
```

具体规则：

- Server 0 Rank：在 CCL base 中以本地 partial 为目标规约对端 partial。
- Server 1 Rank：在第二个 chunk 中以读取到的 Server 0 partial 为目标规约本地 partial。

这避免不同 Server 因操作数顺序相反而产生浮点结果差异。每 Rank 跨 Server 网络量为 `S/8`，完成后每个 owner
持有一个 16 Rank 全局结果 chunk。

## 8. 直接写最终输出的 allgather

每个 Rank 先把自己的最终 chunk 复制到 `recvBuf` 对应 owned offset。随后从同 Server 其他 7 个 owner 读取已完成
chunk，并直接写到本 Rank `recvBuf` 的最终偏移：

```text
peer owner CCL result chunk
    -- HcommReadOnThread -->
local recvBuf final offset
```

Server 0 的完成结果位于 peer CCL base；Server 1 的完成结果位于 peer CCL 第二个 chunk，因此读取偏移分别为：

```cpp
remoteOffset = remoteBase + (serverStart == 0 ? 0 : peerChunkBytes);
```

此阶段每 Rank 的 Server 内网络量仍为 `7S/8`，但不再先拼出完整 CCL result slot，也不再执行完整 `S` 字节的
`result slot -> recvBuf` copy。

## 9. channel 同步协议

对每条 symmetric push channel，最终采用：

```text
record ACK
wait peer ACK
HcommWriteOnThread
channel fence
record DATA_SIGNAL
wait peer DATA_SIGNAL
```

对应新增 helper：

```cpp
WriteVariableWithEntries(...)
```

ACK 保证双方都进入本阶段后再写远端 buffer；channel fence 保证 write 已完成；DATA_SIGNAL 保证双方都完成本阶段
的数据操作后才复用 CCL 范围或进入规约。

跨 Server read 和 allgather read 继续使用相同的 ACK、fence、DATA_SIGNAL 完成握手。worker thread 在通信阶段前后
分别与主 thread 同步，防止本地规约与网络写入并发访问同一 contribution。

## 10. 通信量和本地 copy 分析

对可均匀分成 8 个 chunk 的分片 `S`，每 Rank 网络量为：

| 阶段 | 方向 | 每 Rank 数据量 |
|---|---|---:|
| Server 内 contribution 分发 | push | `7S/8` |
| 跨 Server partial 交换 | read | `S/8` |
| Server 内结果收集 | read | `7S/8` |
| 合计 |  | `15S/8` |

16 Rank 全局网络数据量仍为：

```text
16 * 15S/8 = 30S
```

因此 v8 的收益不来自减少理论网络字节，而来自减少完整本地 copy、减少 CCL 占用和减少主分片次数。

本地 copy 的主要变化：

| 项目 | v7 | v8 |
|---|---:|---:|
| 输入 staging | `S` | self chunk `S/8` |
| 输出 staging | `S` | final owned chunk `S/8` |
| 可选 partial 发布 | 一个 chunk 以内 | 一个 chunk 以内 |
| 完整消息本地 copy | `2S` | `0` |

表中不包含本地 reduce 自身的读写流量。v8 的 self chunk 和 final chunk copy 仍然存在，但均为 `S/8` 量级。

## 11. 最大主分片容量

大消息路径将 CCL slot 数从 2 改为 7 个 remote contribution range：

```cpp
slotNum = RANKS_PER_SERVER - 1;  // 7
```

计算过程为：

```text
effectiveBytes = localBuffer.size - alignedBaseOffset
chunkBytes     = AlignDown(floor(effectiveBytes / 7), 128B)
sliceCount     = floor(chunkBytes / dataTypeSize) * 8
```

默认约 400MiB CCL buffer 下，FP32 的实测任务布局使用：

```text
chunkBytes     = 59,918,592 B
maxSliceBytes  = 59,918,592 * 8
               = 479,348,736 B
               = 457.142578125 MiB
```

对比 v7：

```text
v7 max main slice = 209,715,072 B = 200MiB - 128B
v8 max main slice = 479,348,736 B ~= 457.14MiB
```

最大主分片扩大约 `2.286` 倍。源码中的单 slot `MAX_SLICE_BYTES = 256MiB` 上限在当前 400MiB CCL buffer 下
不会触发，因为 `effectiveBytes / 7` 约为 57.14MiB。

## 12. 测试点 6、7 分片结构

测试点 6，512MiB：

```text
total          = 536,870,912 B
slice 0        = 479,348,736 B
slice 1        =  57,522,176 B
slice count    = 2 large slices
```

相比 v7 的 3 个 hierarchical 主分片，主循环次数从 3 降为 2。

测试点 7，400MiB+4B：

```text
total          = 419,430,404 B
slice 0        = 419,430,404 B
slice count    = 1 large slice
tail           = 0
```

该用例共有 `104,857,601` 个 FP32 元素，分 chunk 后：

```text
owner 0 : 13,107,201 elements = 52,428,804 B
owner 1..7: 13,107,200 elements = 52,428,800 B
```

v8 的 variable read/write plan 为每个 owner 单独保存 `bytes` 和 offset，因此可以处理 owner 0 多一个元素的非均匀
布局。相比 v7 的“2 个主分片 + 260B tail”，整个消息现在只执行一次大消息同步流程。

## 13. Checker 任务图对比

最终 v8 任务图统计：

| 用例 | 版本 | 布局 | checker nodes | data tasks |
|---|---|---|---:|---:|
| 512MiB | v7 | 3 hierarchical slices | 8,441 | 1,272 |
| 512MiB | v8 | 2 large slices | 5,605 | 804 |
| 400MiB+4B | v7 | 2 hierarchical slices + 260B tail | 6,353 | 1,040 |
| 400MiB+4B | v8 | 1 large slice | 2,835 | 402 |

降幅：

| 用例 | checker nodes | data tasks |
|---|---:|---:|
| 512MiB | `-33.6%` | `-36.8%` |
| 400MiB+4B | `-55.4%` | `-61.3%` |

任务数下降主要由分片数减少以及取消完整 CCL input/output staging 引起。Checker 节点是 16 Rank 全局任务图总数，
不能直接当作真实设备关键路径时延，但能反映控制任务、数据 primitive 和同步操作的显著收缩。

## 14. 构建与安装状态

v8 Release 构建通过。最终构建产物：

```text
build/lib64/libhccl_device.so
```

SHA-256：

```text
643c8510372672717dc151dd34cc1eb2660484d8b99666a7b6dfd191692db7b5
```

构建产物与 HCCL-VM 安装目录中的 `libhccl_device.so` 哈希一致。`git diff --check` 通过。

## 15. Checker 验证

### 15.1 512MiB

```text
task metadata          : 5604
checker nodes          : 5605
dataTaskNodeCount      : 804
parallelCandidatePairs : 0
normalSemanticCount    : 256
normalSemanticBytes    : 23890751488
checker                : Success
```

### 15.2 400MiB+4B

```text
task metadata          : 2834
checker nodes          : 2835
dataTaskNodeCount      : 402
parallelCandidatePairs : 0
normalSemanticCount    : 144
normalSemanticBytes    : 19293798584
checker                : Success
```

两个目标用例均通过任务检查、内存冲突检查和语义检查。`parallelCandidatePairs = 0` 表示 Checker 没有发现无序的
并行重叠候选。

## 16. normal runner 数值验证

最终 7-contribution 版本已完成以下 normal runner 验证：

| 用例 | 迭代 | 结果 | 覆盖目的 |
|---|---:|---|---|
| 2MiB FP32 sum | 3 | success | v8 大消息 push、cross reduce、direct allgather |
| 2MiB+4B FP32 sum | 1 | success | 非整除 8 的 variable chunk 布局 |

此前已经验证且不受最终大消息改动影响的分支：

```text
4B    : success
512KiB: success
```

未在 normal runner 中执行完整 512MiB 和 400MiB+4B。原因是 16 Rank 同时分配输入、输出和通信共享内存会超过
当前容器 8GiB `/dev/shm`，不是算法或 Checker 失败。

HCCL-VM 的 `aclrtEventElapsedTime()` 是固定 stub，因此 VM 中显示的耗时不能作为性能数据，只能用于功能和数值
正确性判断。

## 17. 当前测试结果与性能结论边界

当前实现的真实测试结果：

| 测试点 | 数据量 | 当前路径 | v7 基线 | v8 实测 | 时延下降 |
|---:|---:|---|---:|---:|---:|
| 5 | 512KiB | v7 small hierarchical，v8 未改动 | 59.00 μs | 59.00 μs | 0% |
| 6 | 512MiB | v8 7-contribution Mesh push | 4.78 ms | 3.72 ms | 22.2% |
| 7 | 400MiB+4B | v8 7-contribution Mesh push | 3.78 ms | 2.89 ms | 23.5% |

测试点 6、7 相对 v7 基线的加速比分别约为：

```text
测试点 6: 4.78 / 3.72 = 1.285x
测试点 7: 3.78 / 2.89 = 1.308x
```

这些数值来自当前实现的真实测试记录，不是 HCCL-VM 的固定 event stub 输出。

当前已经证明：

- 大消息 7-contribution CCL 地址布局通过 Checker 内存冲突检查。
- symmetric push、确定性本地树、跨 Server canonical reduce 和直接输出 allgather 语义闭合。
- 400MiB+4B 可以作为一个非均匀大分片处理，不再落入 260B tail。
- 测试点 6、7 的全局 checker nodes 和 data tasks 显著下降。
- 2MiB 与 2MiB+4B normal runner 数值正确。
- 测试点 5、6、7 的当前实测时延分别为 `59.00 μs`、`3.72 ms`、`2.89 ms`。

当前尚未证明：

- 512MiB 与 400MiB+4B 的完整 normal runner 数值执行；当前限制来自 VM `/dev/shm` 容量。
- 当前实测结果在不同机器、warmup、迭代次数或日志条件下的可重复性；跨环境不能直接比较。

v7 文档记录的旧实现基线为：

```text
测试点 6, 512MiB    : 4.78 ms
测试点 7, 400MiB+4B : 3.78 ms
```

当前加速比以这两个旧实现记录为基线。若测试条件发生变化，应重新同机测试 v7 和 v8 后再比较。

## 18. 后续优化方向

v8 已经去掉两次完整本地 copy 并显著减少分片，但仍有以下优化空间：

1. 当前每条 channel 都保留完整 ACK 和 DATA_SIGNAL 双握手，控制任务仍多。若真实运行时允许，可研究按 phase
   合并 notify 或用更粗粒度 barrier，但必须同时通过 Checker 和 normal runner，不能只看任务图。
2. Server 内 push 和 allgather 都是 7 路 full Mesh，每 Rank 网络量没有下降。可以评估 recursive-halving /
   recursive-doubling 或分层 ring，在真实拓扑上比较链路并行度、跳数和 notify 代价。
3. 当前跨 Server 两个方向都读取 `S/8`，用于保证两边各自得到完成 chunk。可以研究单向 reduce 后跨 Server
   broadcast，比较跨 Server 带宽与新增同步阶段的取舍。
4. 本地 `4 -> 2 -> 1` 树每层有全 worker 同步。可分析 Hcomm 是否允许在单个 owner 内形成依赖链、不同 owner
   跨阶段流水，以缩短 slice 间空洞。
5. 512MiB 仍需要两个分片。若平台允许使用更多用户输出 scratch，或能与下一阶段安全重叠，可进一步提高单分片
   容量并尝试 slice pipeline；这需要严格验证 in-place 场景和跨 Rank 生命周期。
6. 算子入口目前按需求省略 4 Byte 地址对齐检查。若实现需要脱离比赛前置条件复用，应恢复与数据类型一致的
   `sendBuf`、`recvBuf` 对齐校验并补充错误用例。

## 19. 最终摘要

v8 将测试点 6、7 的大消息算法从“双完整 CCL slot + owner pull”改为“七远端贡献 packed CCL + source Mesh
push”。自己的 contribution 使用 `recvBuf` scratch，完成结果直接 allgather 到最终输出。

核心结果：

```text
完整 CCL staging copy : 2S -> 0
CCL 主布局             : 2S -> 7S/8
每 Rank 网络量         : 保持 15S/8
最大主分片             : ~200MiB -> ~457.14MiB
512MiB 分片数          : 3 -> 2
400MiB+4B 分片数       : 2 + tail -> 1
512MiB data tasks      : 1272 -> 804  (-36.8%)
400MiB+4B data tasks   : 1040 -> 402  (-61.3%)
```

两个目标任务图均为 `Checker Success`，2MiB 和 2MiB+4B normal runner 均为 `success`。当前测试点 5、6、7
分别为 `59.00 μs`、`3.72 ms`、`2.89 ms`；测试点 6、7 相对 v7 记录分别降低约 `22.2%` 和 `23.5%`。

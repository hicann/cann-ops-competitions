# AllReduce 716v11 小消息回退与大消息双 Lane 优化记录

## 1. 版本定位

本文记录 2026-07-16 在 v10 基础上完成的 AllReduce v11 实现。工程目录：

```text
/data/mingwei/hccl_allreduce_problem_222_template
```

v11 基于以下 v10 提交继续开发：

```text
f9d0376 Optimize allreduce with parallel Mesh/Clos axes
```

算法实现只修改比赛允许的选手文件：

```text
op_kernel_aicpu/exec_op.cc
```

本版本包含两类调整：

- 精确 512KiB 路径撤销 v10 的 one-shot 双轴实现，恢复递归减半/倍增的层次化算法。
- 16 Rank、2 Server 大消息路径保留 v10 四阶段 Mesh/Clos 重叠，并增加双 Lane Mesh-RS 和握手裁剪。

64KiB 到 1MiB 范围内除 512KiB 外的原双 Server 算法，以及其他拓扑的通用 fallback 路径保持不变。

## 2. v10 的剩余瓶颈

### 2.1 精确 512KiB

v10 将消息按约 `1:2` 分为两个轴，在本机 one-shot Mesh 与跨 Server Clos 之间做重叠。该实现需要：

- 在 CCL workspace 保存多份完整 contribution。
- 向 7 个本机 peer 分发数据。
- 通过本地固定规约树合并 contribution。
- 支付较多 worker 启动、notify、copy 和本地 reduce 任务开销。

512KiB 属于启动时延主导区间，上述任务开销超过了双轴重叠带来的收益。v10 实测为 `101.00 μs`，历史递归
实现曾取得 `59.00 μs`，因此 v11 恢复该递归路径并重新进行实际评测。

### 2.2 测试点 6/7 大消息

v10 已经通过四阶段调度重叠 Server 内 Mesh 链路和跨 Server Clos 链路，但两个 Mesh-RS 轴仍分别执行 7 个
striped round。每轮都需要 worker 同步、channel fence 和 ServerBarrier，关键路径上存在较重的固定同步成本。

此外，v10 的 Phase 3/4 variable read 为每条 channel 都执行完整的 ACK、DATA 双向握手。ReduceScatter 已经发布
不可变结果后，部分握手不再承担数据正确性职责，只会增加任务图和返回时延。

## 3. 测试点 5：精确 512KiB 独立分支

分支条件为：

```text
rankSize == 16 && totalBytes == 512 * 1024
```

每个 Server 包含 8 个 Rank。输入先复制到 CCL input slot，然后执行：

```text
3-stage recursive-halving local ReduceScatter
1-stage paired cross-server Reduce
3-stage recursive-doubling local AllGather
```

ReduceScatter 每轮与 `localRank ^ mask` 对端交换当前保留范围，`mask` 依次为 `4, 2, 1`。每轮数据量减半，
最终每个 local rank 持有一个连续的 `1/8` 分片。

跨 Server 阶段只与相同 local rank 的配对 Rank 合并该 `1/8` 分片。AllGather 再以 `mask = 1, 2, 4` 反向扩展
结果范围，最终复制到用户 output。

该路径保留完整的 ACK、DATA 握手。HCCL-VM normal runner 表明直接删除 DATA 握手会导致通信停滞，因此本版本
优先保持可执行性和正确性。

v11 已在真实评测环境复现 `59.00 μs`，相对 v10 的 `101.00 μs` 降低 `42.00 μs`，降幅约 `41.6%`。

## 4. 测试点 6/7：双 Lane Mesh-RS

大消息继续使用 v10 的约 `1:2` 分区和四阶段 Mesh/Clos 调度：

```text
Phase 1: Mesh-RS(A) || Clos-RS(B)
Phase 2: Clos-RS(A) || Mesh-RS(B)
Phase 3: Clos-AG(A) || Mesh-AG(B)
Phase 4: Mesh-AG(A) || Clos-AG(B)
```

v11 为每个 Mesh-RS aggregate 增加第二条独立 Lane。原来 7 个逻辑 stripe round 变为：

```text
macro round 0: logical round 0 + logical round 1
macro round 1: logical round 2 + logical round 3
macro round 2: logical round 4 + logical round 5
macro round 3: logical round 6
```

同一 macro round 中，两条 Lane 的写入在同一 worker 调度窗口中排队，并共用一次 channel fence 和 worker
同步。第二条 Lane 首轮使用普通 `HcommWriteOnThread` 初始化，后续轮次使用 `HcommWriteReduceOnThread`。

4 个 macro round 完成后，通过一次本地 reduce 将第二条 Lane 合并到第一条 Lane。随后执行 ServerBarrier，
发布合并后的本机 aggregate，确保 Phase 3 的远端读取不会早于本地合并完成。

该设计不减少网络 payload，但将每个 Mesh-RS 轴的 7 次同步轮次降为 4 次。两个 Mesh-RS 轴合计减少 6 个
round 级同步窗口，代价是增加一份 aggregate workspace 和每轴一次本地 reduce。

## 5. Phase 3/4 握手裁剪

v11 新增可按阶段选择 readiness ACK 的 variable read 调度。

Phase 3 开始时，两个 ReduceScatter 轴已经通过最终 ServerBarrier 发布不可变 CCL 范围，因此直接读取，不再对
每条 channel 执行 ACK/DATA rendezvous：

```text
Phase 3: read + fence
```

Phase 4 仍保留启动 ACK，利用该 ACK 确认所有 source peer 已经完成 Phase 3；但 Phase 4 已是最终阶段，本地 fenced
read 写入用户 output 后不再发送和等待完成 DATA ACK：

```text
Phase 4: readiness ACK + read + fence
```

这样既保留 Phase 3 到 Phase 4 的跨 Rank 发布关系，又减少了最终返回前的 notify 任务。

## 6. Workspace 与通信量

大消息新增两份第二 Lane aggregate，主要 CCL 布局约为：

```text
first aggregate lane 0       : S/24
first aggregate lane 1       : S/24
second source                : S/3
second aggregate lane 0      : S/24
second aggregate lane 1      : S/24
total                        : S/2
```

512MiB 和 400MiB+4B 均可放入比赛提供的约 400MiB CCL buffer。所有分区按元素数计算，并使用商和余数分配，
因此支持测试点 7 的非均匀 FP32 元素数。

双 Lane 只改变 stripe 排队和 aggregate 布局，不重复传输数据。大消息每 Rank 网络量仍与 v10 相同，约为
`15S/8`；主要收益来自同步关键路径缩短，而不是网络字节数下降。

## 7. 任务图变化

大消息 Checker V3 任务图对比如下：

| 版本 | 图节点 | 数据任务 | 重叠候选 | 并行冲突 |
|---|---:|---:|---:|---:|
| v10 | 11489 | 1952 | 24208 | 0 |
| v11 | 7841 | 1984 | 16624 | 0 |

v11 图节点相对 v10 减少 `3648`，降幅约 `31.8%`。数据任务略有增加，来源是第二 Lane 的初始化和最终本地合并；
但 round 级同步、notify 和握手任务明显减少。

所有 `16624` 组内存重叠候选均存在确定依赖顺序，Checker 未发现并行内存冲突。

## 8. VM 构建与正确性验证

Release clean build 和安装在 `hccl-vm` Docker 内完成。容器内 CANN 路径为：

```text
/opt/hccl-vm-workspace/Ascend/cann-9.1.0
```

最终构建库与 VM 安装库 SHA-256 一致：

```text
67a423cf210dd67bd436bca9c50bc7d23d5a3ecf6ab0d7db5755cc1be0c595a1
```

已完成以下验证：

| 用例 | 验证方式 | 结果 |
|---|---|---|
| 512KiB | normal runner | `check_result: success`，task metadata `672` |
| 2MiB+4B | normal runner | `check_result: success`，task metadata `7840` |
| 2MiB+4B | Checker V3 | `Checker Success` |
| 512MiB | check-only Checker V3 | `Checker Success` |
| 400MiB+4B | check-only Checker V3 | `Checker Success` |

大消息 Checker V3 统计：

```text
nodes                    = 7841
data tasks               = 1984
overlap candidate pairs  = 16624
ordered candidate pairs  = 16624
parallel conflict pairs  = 0
```

SingleTaskCheck、MemConflict 和 SemanticCheck 均成功。

## 9. 实际性能结果

v11 已在真实评测环境完成测试点 5、6、7 的性能评测：

| 测试点 | 数据量 | v10 实测 | v11 实测 | 相对 v10 |
|---:|---:|---:|---:|---:|
| 5 | 512KiB | 101.00 μs | **59.00 μs** | 时延下降 41.6% |
| 6 | 512MiB | 2.62 ms | **2.26 ms** | 时延下降 13.7% |
| 7 | 400MiB+4B | 2.10 ms | **1.79 ms** | 时延下降 14.8% |

v11 当前实际成绩为：

```text
测试点 5: 59.00 μs
测试点 6: 2.26 ms
测试点 7: 1.79 ms
```

HCCL-VM 的 event elapsed time 是固定 stub，只用于构建、任务语义、依赖和内存冲突验证。上表成绩来自真实评测
环境，不是 VM 固定计时输出。

## 10. 结论与后续方向

v11 的主要变化为：

```text
512KiB          : restore recursive halving / paired reduce / recursive doubling
large Mesh-RS   : 7 logical rounds -> 4 dual-lane macro rounds
Phase 3 reads   : remove redundant per-channel rendezvous
Phase 4 reads   : keep readiness ACK, remove final completion ACK
large task DAG  : 11489 -> 7841 nodes
checker         : 0 parallel conflict pairs
```

实际结果确认大消息的双 Lane 和握手裁剪有效，测试点 6/7 相对 v10 分别下降 `13.7%` 和 `14.8%`。512KiB
恢复递归路径后回到 `59.00 μs`，相对 v10 下降 `41.6%`；若要进一步接近 40 μs，应继续研究利用 7 个本机
peer worker 的并行 direct ReduceScatter/AllGather，将本机 3+3 个串行阶段进一步压缩。

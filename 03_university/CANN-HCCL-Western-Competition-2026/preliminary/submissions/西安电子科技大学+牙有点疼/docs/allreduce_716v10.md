# AllReduce 716v10 双轴 Mesh/Clos 并行实现与验证

## 1. 版本定位

本文记录 2026-07-16 在 v9 基础上完成的 AllReduce v10 优化。工程目录：

```text
/data/mingwei/hccl_allreduce_problem_222_template
```

v10 基于以下 v9 提交继续开发：

```text
dfb84f1 Optimize allreduce with striped write-reduce
```

算法源码只修改选手实现文件：

```text
op_kernel_aicpu/exec_op.cc
```

本轮同时重构精确 512KiB 路径和大于 1MiB 的 16 Rank 双 Server 路径。64KiB 到 1MiB 范围内除
512KiB 外的原有双 Server 算法，以及其他拓扑的通用 fallback 路径保持不变。

## 2. v9 的剩余瓶颈

v9 大消息已经使用 striped WriteReduce Mesh 消除 7 份远端 contribution 和本地规约树，但各通信域仍按顺序
执行：

```text
Server 内 Mesh-RS -> 跨 Server reduce -> Server 内 Mesh-AG
```

16 Rank、2 Server 的拓扑同时存在两组可独立使用的链路：

- Server 内 8 Rank Full-Mesh 链路。
- 相同 local rank 之间的跨 Server 配对链路。

顺序调度无法让两组链路同时工作。部分 Rank 的跨 Server channel 还可能与一个本地 Mesh channel 落到同一
worker/main notify，进一步破坏预期并行。

精确 512KiB 的 v9 路径使用递归减半、跨 Server 配对规约和反向递归倍增，阶段之间同样存在明显串行依赖。

## 3. v10 总体设计

v10 将 Server 内通信记为 `Mesh`，跨 Server 配对通信记为 `Clos`，把消息分到两条独立轴上，并在每个阶段
并行执行一个 Mesh 操作和一个 Clos 操作。

核心调度原则为：

```text
independent Mesh worker set || dedicated Clos worker
```

`AssignUnusedWorkerThread` 从未被本地 Mesh channel 占用的 worker 中为跨 Server channel 选择独立线程，并为
它建立独立的 main notify 映射。这样 `ThreadSyncBefore/After` 只约束相应通信轴，不会把本应重叠的操作重新
串行化。

## 4. 测试点 5：512KiB 双轴分区

512KiB 消息按约 `1:2` 分成 A、B 两段：

```text
A = first third
B = remaining two thirds
```

算法执行两个双轴阶段：

```text
Axis 0: local one-shot Mesh(A) || paired Clos-Reduce(B)
Axis 1: paired Clos-Reduce(A)  || local one-shot Mesh(B)
```

每个 Rank 为 A、B 保存各 Server 内 8 份 contribution。Server 内使用固定二叉规约树，保证 FP32 sum 的规约
顺序确定。跨 Server 使用单独 CCL source slot，避免双向原地 `WriteReduce` 同时读写同一范围。

该路径取消了原来的递归减半和递归倍增，但会在 Server 内复制完整分段并执行 one-shot 规约。实际结果表明，
当前硬件上它的启动与本地任务开销大于双轴重叠收益，因此测试点 5 相对 v9 出现回退，详见第 10 节。

## 5. 测试点 6/7：四阶段 Mesh/Clos 并行

大消息同样按约 `1:2` 分成 A、B。B 再按 Server 分成两个近似相等的部分，每个 Server 负责其中一半。完整
算法由四个阶段组成：

```text
Phase 1: Mesh-RS(A) || Clos-RS(B)
Phase 2: Clos-RS(A) || Mesh-RS(B)
Phase 3: Clos-AG(A) || Mesh-AG(B)
Phase 4: Mesh-AG(A) || Clos-AG(B)
```

其中：

- `Mesh-RS` 复用 v9 的 7-round deterministic striped WriteReduce。
- `Mesh-AG` 通过 7 条本地 peer channel 读取其他 owner 的结果分片。
- `Clos-RS` 通过配对 `WriteReduce` 合并两个 Server 的 contribution。
- `Clos-AG` 读取另一个 Server 负责的结果范围。

A、B 的布局使每个阶段都有相对均衡的 Mesh/Clos 数据任务。阶段 3 和阶段 4 将 7 条 Mesh channel 与 1 条
Clos channel 合并为同一次 variable read 调度，但 Clos channel 继续使用独立 worker。

## 6. 非均匀分区和 CCL 布局

所有分区都以元素数计算。`GetPartitionRange` 和 `GetChunkRange` 使用商和余数分配元素，低编号 partition/owner
获得余数元素，因此支持测试点 7 的 `400MiB+4B` 非均匀 FP32 元素数。

大消息的主要 CCL 布局为：

```text
first aggregate      : approximately A/8
second source        : approximately B/2
second aggregate     : approximately (B/2)/8
```

当 `A:B` 约为 `1:2` 时，总占用约为：

```text
S/24 + S/3 + S/24 = 5S/12
```

512MiB 和 400MiB+4B 均可在比赛给定 CCL buffer 内单次完成，不需要恢复 v8 的多分片循环。各区域按 128B 对齐，
最终尾部元素仍按 FP32 元素边界处理。

## 7. 通信量和理论关键路径

对可均匀划分的消息 `S`，每 Rank 总网络量仍为：

| 通信域 | 每 Rank 数据量 |
|---|---:|
| 四个 Mesh RS/AG 子阶段 | `7S/6` |
| 四个 Clos RS/AG 子阶段 | `17S/24` |
| 合计 | `15S/8` |

因此 v10 没有通过增加总网络字节数换取并行。若暂时忽略启动、barrier、不同链路带宽和尾部不均匀，四阶段中
每阶段取 Mesh/Clos 较慢的一侧，理论数据关键路径约为：

```text
S/3 + 7S/24 + 7S/24 + S/3 = 5S/4
```

相对顺序执行的 `15S/8`，理想关键路径比例为约 `2/3`。这只是链路数据量模型，实际时延还包含 7-round barrier、
notify、task launch、local copy 和链路带宽差异。

## 8. VM 构建和正确性验证

Release 构建成功，构建库与 VM 安装库 SHA-256 一致：

```text
e059a8c620de3da035006ac7669279f2088b55816d0b9f602346ee5a1eca56d8
```

已完成以下验证：

| 用例 | 验证方式 | 结果 |
|---|---|---|
| 512KiB | normal runner | `check_result: success` |
| 512KiB，`-w 0 -n 20` | 多次 normal runner | `check_result: success` |
| 512KiB | Checker V3 | `Checker Success` |
| 2MiB+4B | normal runner | `check_result: success` |
| 2MiB+4B | Checker V3 | `Checker Success` |
| 512MiB | check-only Checker V3 | `Checker Success` |
| 400MiB+4B | check-only Checker V3 | `Checker Success` |

最终大消息 Checker 统计为：

```text
nodes                    = 11489
data tasks               = 1952
overlap candidate pairs  = 24208
ordered candidate pairs  = 24208
parallel conflict pairs  = 0
```

这说明所有检测到的重叠内存访问都有确定顺序，没有并行内存冲突。`2MiB+4B` 和 `400MiB+4B` 同时覆盖不能被
Server/owner/stripe 均匀整除的元素分区。

## 9. VM 计时说明

HCCL-VM 的 event elapsed time 是固定 stub。例如 20 次迭代报告的 `50 us` 实际来自固定 `1 ms / 20`，不能用来
评价算法真实性能。VM 在本版本中的作用是验证：

- Release 编译和动态库安装。
- Engine Context 与 notify 的多次复用。
- normal runner 数值正确性。
- Checker 任务语义、依赖和内存冲突。

第 10 节数据来自实际评测环境，不是 VM 固定计时输出。

## 10. 实际性能结果

当前 v10 的测试点 5、6、7 实际测试结果为：

| 测试点 | 数据量 | v9 实测 | v10 实测 | 相对 v9 |
|---:|---:|---:|---:|---:|
| 5 | 512KiB | 59.00 μs | **101.00 μs** | 时延增加 71.2% |
| 6 | 512MiB | 3.25 ms | **2.62 ms** | 时延下降 19.4% |
| 7 | 400MiB+4B | 2.56 ms | **2.10 ms** | 时延下降 18.0% |

为避免单位或来源歧义，当前实现需要注明的三个实际成绩是：

```text
测试点 5: 101.00 μs
测试点 6: 2.62 ms
测试点 7: 2.10 ms
```

大消息结果验证了 Mesh/Clos 并行调度的收益；测试点 5 则说明 one-shot 全分段复制和固定规约树不适合当前
512KiB 启动时延区间。后续若继续优化，应优先保留 v10 的独立 Clos worker，同时重新评估 512KiB 路径是否回退
到 v9 recursive halving，或改用更小粒度的双轴 reduce-scatter/allgather。

## 11. 结论

v10 的主要变化为：

```text
512KiB          : two-axis partitioned Mesh/Clos
large messages  : four-phase Mesh/Clos overlap
cross channel   : independent unused worker assignment
A:B split       : approximately 1:2
network volume  : remains 15S/8 per Rank
large checker   : 0 parallel conflict pairs
```

当前实现已通过 Release 编译、VM normal runner 和 Checker 验证。实际测试点 5、6、7 分别为
`101.00 μs / 2.62 ms / 2.10 ms`；其中大消息继续提升，512KiB 路径存在明确性能回退。

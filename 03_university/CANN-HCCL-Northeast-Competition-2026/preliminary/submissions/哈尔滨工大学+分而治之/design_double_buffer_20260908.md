# 双 buffer 流水设计（2026-09-08，hccl_temp 中继版第二阶段）

## 动机与证据

cannjudge 实测（hccl_temp@847aa96）：58μs / 1.31ms / 1.05ms；榜一 34μs / 981μs / 777μs。
两个大消息点与榜一差距同为 ~1.34x，与结构模型 T=15m/11（1.02ms / 800μs）的差距约 280μs / 250μs ——
量级恰为「一个 chunk 边界的串行停顿」。

根因：现行协议每条流单槽复用 —— 发送端每 chunk 须等「对端 copy_out 完成 + ACK 返回」才能写下一块，
链路在对端 copy_out 期间空闲。512MB 片每流 2 chunk（cBig=256MB / cSmall=128MB），产生一次完全暴露的停顿。

## 方案

每个槽位区域切为 RELAY_SUB_SLOT_NUM=2 个子槽交替使用：

- cBig 区域 256MB → 2 × 128MB 子槽；cSmall 区域 128MB → 2 × 64MB 子槽。总占用不变（384MB ≤ 400MB CCL）。
- chunk 大小 = 子槽大小；chunk 数约翻倍（512MB 片：本地/对位直发 4，d1 直发 3，中继 3）。
- 接收端在流开始前**预发 N 个 ACK**（每个子槽一个），发送端保持「每写前等 ACK、写子槽 `k%N`、写后 Record DATA」，
  尾随 Wait ACK 由 1 改为 N。
- 效果：chunk k+1 的链路传输与 chunk k 的对端 copy_out 重叠；只要 copy_out ≤ 传输时间，链路满载。

## checker 约束（v2 修订，laptop 实测发现）

真 CheckerV3 的 SyncConflict 要求**每个 Wait 的 Record 来源确定唯一**：同一 notify id 上存在多个未消费
Record 即判 many-to-one 冲突（v1 预发 2 个同 id ACK 被判 conflictCount=1）。
因此 v2 改为**每个子槽独立 notify id**：中继/直发流 ACK={0,1} DATA={2,3}，本地片流 ACK={4,5} DATA={6,7}，
channel notify 申请数 4→8（host 与 kernel 的 CHANNEL_NOTIFY_NUM 同步）。
每个 id 上 Record/Wait 严格一一交替：预发 1 + 每 chunk 实发 1 = 循环等 n_p + 尾随 1。

计数守恒（每次调用结束每个 notify id 计数归零）：某子槽 id 上，发送端 Wait 总数 = n_p + 1（循环 n_p + 尾随 1），
接收端 Record 总数 = 1 预发 + n_p 实发。两边相等。

## 协议改动点（6 处流，全部同一模式）

| 流 | 发送端 | 接收端 | 槽位 |
|---|---|---|---|
| root→机内卡 中继流 | ExecRelayRoot worker w | ExecRelaySourceCard forwarder | slot1 @cBig，子槽 cSmallSub |
| root→机内卡 本地片流 | ExecRelayRoot worker w | ExecRelaySourceCard main | slot0 @0，子槽 cBigSub |
| root→S1 直发流 | ExecRelayRoot clos worker | ExecRelayRemoteCard main | slotB @0，子槽 cBigSub |
| 中继卡→S1 转发流 | ExecRelaySourceCard forwarder | ExecRelayRemoteCard main | slotA @cBig，子槽 cSmallSub |

预发 ACK 的位置不变（原发 1 次处改为 N 次）；relayLoops==0 时整段跳过（含预发与尾随），保持现有条件结构。

## 验证计划

1. `.syncheck/tests/verify_scatter.py`：语法 + 协议 harness 289 场景 PASS。
2. `.syncheck/tests/checker_graph_repro.py`：checker 图 drain，832 节点 0 阻塞（5 个 root 探测点）PASS。
3. laptop hccl_vm：首点 256MB/r7 已过（含 8 notify 容量确认）；全矩阵回归中。
4. 用户提交 cannjudge。

## 风险与回退

- 改动集中于 exec_op.cc 中继段；若 cannjudge 回归，git 回退 847aa96。
- 若 copy_out 时间 > 传输时间（_memcpy 极慢_），双 buffer 收益打折，届时上 4 子槽（只需改 RELAY_SUB_SLOT_NUM）。

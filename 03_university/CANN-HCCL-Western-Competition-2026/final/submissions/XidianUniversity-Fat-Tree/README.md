# AllReduce 方案说明

## 总体思路

本方案面向 4/12/16 卡 AllReduce 场景，根据 rank 数量和数据规模选择不同的通信路径。小数据优先降低启动和同步开销，大数据优先提高链路并行度，并在 12 卡、16 卡场景中使用 Flat Owner 分片流水来提升带宽利用率。

整体设计遵循三个原则：

- 小数据减少阶段数和资源发布数量，尽量压低固定延迟。
- 大数据按 segment 或 slice 切分，reduce 与 gather 尽早交接，减少全量阶段等待。
- rank 间同步使用细粒度事件，避免连续阶段复用通知位时产生不确定依赖。

## 路径选择

| 场景 | 通信路径 | 分片策略 |
| --- | --- | --- |
| 512 KiB 小数据 | Recursive Doubling | 不分片 |
| 4 卡大数据 | Recursive Halving + Doubling | 不分片 |
| 12 卡大数据 | Full-Lane Flat Owner RSAG | 4 slice |
| 16 卡中等大数据 | Full-Lane Flat Owner RSAG | 3 slice |
| 16 卡超大数据 | Full-Lane Flat Owner RSAG | 4 slice |

小数据阈值为 512 KiB。16 卡大数据根据规模选择 3 slice 或 4 slice，避免数据量不足时分片过多带来额外同步开销。

## 小数据路径

小数据采用 Recursive Doubling。该路径的目标是减少 kernel 和通信阶段数量，让每个 rank 通过较少轮次完成局部累加和结果传播。

由于 512 KiB 场景中固定调度开销占比较高，该路径不额外引入拓扑分层、owner 分片或复杂流水，避免为了提高并发而增加更多资源发布和事件等待。

##  4 卡大数据路径

2 卡和 4 卡大数据采用 Recursive Halving + Doubling。Reduce-Scatter 阶段逐步完成分段规约，All-Gather 阶段再将最终结果传播到所有 rank。

该路径结构简单、kernel 数较少，适合 rank 数较小的场景。RH+D 的同步关系稳定，额外资源开销较低。

## 12 卡与 16 卡大数据路径

12 卡和 16 卡大数据采用 Full-Lane Flat Owner RSAG。数据按 owner segment 划分，每个 owner 负责对应 segment 的归约结果，然后通过 gather 阶段将完整结果分发给所有 rank。

该路径的重点是充分利用多条 rank 间通信 lane：

- Reduce 阶段将各 rank 的对应 segment 汇聚到 owner。
- Owner 完成 segment 归约后，尽早启动后续 gather。
- Gather 阶段按 slice 推进，避免所有 reduce 完成后才统一开始传播。

12 卡固定使用 4 slice，以提高跨 rank 交接粒度。16 卡根据数据规模自适应选择 3 slice 或 4 slice，在同步开销和并行度之间折中。

## 同步机制

Flat Owner 路径中存在连续 slice 或连续阶段复用通知位的情况。为避免通知被合并，barrier 使用 rank 有序的 READY/ACK 握手：

1. 低 rank 向高 rank 发送 READY。
2. 高 rank 等待 READY。
3. 高 rank 向低 rank 返回 ACK。
4. 低 rank 等待 ACK。

一次握手返回后，双方都已经消费本轮通知，同一组通知位可以继续用于下一轮同步。这样可以降低多打一、死锁和 checker stuck 的风险。

## 资源管理

通信资源按实际阶段需求发布。Reduce 相关阶段负责建立必要的远端地址和 token；纯 gather 阶段尽量复用前序已经建立的资源，避免重复发布相同 input/output/token。

这样可以减少资源交换次数，同时保持各阶段对远端 buffer 的访问关系清晰可控。

## 验证重点

- 所有 rank size 和数据规模均应通过 Checker。
- 12 卡、16 卡大数据需要重点观察是否存在 notify 多打一、wait mask 卡死或内存冲突。
- 小数据路径需要关注固定延迟，确认没有因资源发布或额外同步导致退化。
- 大数据路径需要关注 slice 数量、kernel 数和阶段交接粒度对性能的影响。

## 构建方式

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh
```

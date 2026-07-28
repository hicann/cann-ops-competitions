# V67 方案说明

## 基线

V67 是 V64 的修复版，不改变任何通信算法与拓扑策略。

## 修复一：可复用 barrier

原实现先向所有 peer Record，再等待所有 peer。相同通知位被连续 barrier 重用时，前一次 Record 可能尚未被对端消费，后一次 Record 会被 CKE 合并。

新实现对每对 rank 使用有序握手：

1. 低 rank 向高 rank 发送 READY。
2. 高 rank 等待 READY。
3. 高 rank 向低 rank 返回 ACK。
4. 低 rank 等待 ACK。

返回前双方都已消费本轮通知，因此 READY/ACK 位可以安全用于下一轮。ACK 复用 topology-star release 位；Flat Owner 与 topology-star 不会在同一次算法调用中并行执行。

## 修复二：gather 资源去重

非分片 Flat Owner 的 gather-only 调用使用前序 reduce phase 已发布的远端资源，不再执行 ExchangeHierarchicalResources。initializeOwner 或 reducePeers 为真时仍保留资源发布。

## 未改变项

- V64 的 rank/数据规模路由。
- Flat Owner pipeline depth。
- peer 分组、Layer 0/Layer 1 信道与数据交换方向。
- reduce/gather 次序和数据范围。

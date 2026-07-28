# Broadcast 集合通信算子

## 团队信息

| 项目 | 内容 |
|------|------|
| **赛事** | 2026 HCCL通信库创新大赛长三角赛区 |
| **团队名称** | 梨花 |
| **学校** | 中国科学技术大学 |
| **团队成员** | liyixine |


## 项目简介

本项目实现了基于 CANN AICPU_TS 通信引擎的 **Broadcast 集合通信算子**，支持将 Root 节点的数据广播到通信域内所有节点。针对大赛 16 卡（2 台 8 卡服务器）的拓扑场景，设计并实现了两种自适应算法：

- **Flat（平面扇出）**：适用于小消息或非 16 卡场景。Root 直接将数据逐 chunk 扇出到所有接收方，使用 2 个 AICPU Thread 并行分摊 15 个接收方的通信负载。
- **Two-Dimension（二维流水线 Scatter-Cross-AllGather）**：适用于 16 卡且数据量 > 1MB 的场景。将 16 个 Rank 组织为 2×8 的二维拓扑，分 phase 流水执行 Scatter（Root 在本 Server 内按 Slice 分发）→ Cross（跨 Server 传输）→ AllGather（各 Server 内收集全部 Slice），并引入 Tile 级流水线以重叠通信与计算。

### 目录结构

```
code/
├── CMakeLists.txt              # 顶层 CMake 配置
├── build.sh                    # 构建脚本
├── include/
│   ├── custom.h                # ★ 自定义数据结构（CommBuffer/ChannelInfo/AlgResourceCtx）
│   ├── common.h                # 通用宏定义与 OpParam
│   └── ...                     # 框架头文件
├── op_host/
│   └── broadcast.cc            # ★ Host 侧：拓扑解析、资源申请、Channel 构建
└── op_kernel_aicpu/
    └── exec_op.cc              # ★ Device 侧：通信算法编排（Flat / Two-Dimension）
```

> 带 ★ 的文件为选手编写部分。

## 算法设计

### 1. Flat 算法

```
                    ┌─→ Rank 1  ─┐
                    ├─→ Rank 2  ─┤
  Root ──Copy──→    │    ...     │  (thread 0: 8 receivers)
  [HCCL Buffer] ──→ ├─→ Rank 7  ─┤
                    │            │
                    ├─→ Rank 8  ─┤
                    │    ...     │  (thread 1: 7 receivers)
                    └─→ Rank 15 ─┘
```

- Root 先将数据 chunk 拷贝到本地 HCCL Buffer，再通过 2 个 Thread 并行向 15 个接收方下发 Notify + 等待完成信号
- 非 Root 节点等待 ACK Notify 后执行 RDMA Read，完成后回写 DataSignal 通知 Root
- 按 chunk 循环直到全部数据传输完毕

### 2. Two-Dimension 算法

16 个 Rank 按跨 Server 拓扑组织为 2 台服务器 × 每台 8 卡的结构：

```
         Server 0                   Server 1
    ┌────┬────┬────┬────┐      ┌────┬────┬────┬────┐
    │ R0 │ R1 │... │ R7 │      │ R8 │ R9 │... │ R15 │
    └────┴────┴────┴────┘      └────┴────┴────┴────┘
         ↕ Scatter                 ↕ AllGather
         └──────── Cross ──────────┘
```

每个 Chunk 的处理分为三个阶段，按 Tile 粒度流水并行：

1. **Scatter（Root 扇出）**：Root 将数据均匀切分为 8 个 Slice（对应 8 个 Owner Rank），按 Tile 粒度将各 Slice 数据写入对应 Owner 的 HCCL Buffer（本 Server 内直接 LocalCopy，跨 Server 通过 `WriteWithNotify`）
2. **Cross（跨 Server 传输）**：Root 所在 Server 的各 Owner 将收到的新 Tile 通过跨 Server Channel 转发到对端 Server 的对应 Rank
3. **AllGather（Server 内收集）**：每个 Rank 从同 Server 内其余 7 个 Owner 处通过 RDMA Read 拉取对应的 Slice Tile，并 LocalCopy 到用户 Buffer

**Tile 流水线**：大消息场景下，每个 Slice 被进一步切分为多个 Tile（≤400MB+4B 场景使用 4MB 细粒度 Tile），各 Tile 的 Scatter→Cross→AllGather 可连续下发，借助 Notify 机制实现通信与计算的重叠。

**快速尾块路径**：剩余数据 ≤1MB 时，不再启动完整流水线，改为 Root → relay 两跳扇出。


## 编译运行

### 编译

```bash
cd code
bash build.sh            # Release 编译
bash build.sh --debug    # Debug 编译（支持断点调试）
bash build.sh --format   # 代码格式化
```

编译产物位于 `code/build/` 目录。


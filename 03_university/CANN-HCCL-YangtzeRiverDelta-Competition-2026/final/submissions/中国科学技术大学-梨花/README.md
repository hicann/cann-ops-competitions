# Broadcast 集合通信算子

## 团队信息

| 项目 | 内容 |
|------|------|
| **赛事** | 2026 HCCL通信库创新大赛长三角赛区决赛 |
| **团队名称** | 梨花 |
| **学校** | 中国科学技术大学 |
| **团队成员** | liyixine |

## 项目简介

本项目实现了基于 CANN CCU（Compute Core Unit）通信引擎的 **Broadcast 集合通信算子**，支持将 Root 节点的数据广播到通信域内所有节点。针对评测环境中的多种拓扑（16 卡 2×8、12 卡 8+4、4 卡、任意规模），设计了五级自适应算法策略，覆盖从小消息（Direct 扇出）到大消息（并行分层 2D 流水线 / NHR 递归 / Mesh1D）的全场景。

### 目录结构

```
code/
├── CMakeLists.txt              # 顶层 CMake 配置
├── build.sh                    # 构建脚本
├── include/
│   ├── custom.h                # ★ 自定义数据结构、算法选择、KernelArg 定义
│   ├── common.h                # 通用宏定义与 OpParam
│   └── ...                     # 框架头文件
├── op_host/
│   ├── broadcast.cc            # ★ Host 侧：拓扑解析、资源申请、Kernel 注册
│   ├── exec_op.cc              # ★ Host 侧：通信算法编排与 Kernel Launch
│   └── exec_op.h               # ★ Host 侧：ExecOp 接口
└── op_kernel_ccu/
    ├── ccu_kernel.cc           # ★ CCU 侧：Kernel 实现（6 个 Kernel 函数）
    └── ccu_kernel.h            # ★ CCU 侧：Kernel Context 定义与函数声明
```

> 带 ★ 的文件为选手编写部分。

## 算法设计

### 算法选择策略

根据数据量与 Rank 规模自动选择最优算法：

| 算法模式 | 适用条件 | 核心思路 |
|----------|----------|----------|
| **Direct** | ≤1MB，非 512KB 小包 | Root 直接扇出至所有 Peer，简单低延迟 |
| **Parallel Direct** | 恰好 512KB，16/12 卡 | 按 Layer 拆分双 Kernel 并行 Launch，Root 双 Thread 并发写入 Layer-0/1 |
| **Parallel Hierarchical 2D** | >1MB，16 卡（2×8） | 分层并行：Layer-0 Kernel 处理组内 Scatter/AllGather，Layer-1 Kernel 处理跨组 Scatter/Cross-AllGather，四 phase 双 Thread 流水 |
| **NHR1D** | >1MB，4 卡 | 递归减半 Scatter → 递归倍增 AllGather，每步仅与一个对端通信，Paired Chunks 合并两个 256MB 分片 |
| **Mesh1D** | >1MB，其他规模 | Scatter → Barrier → AllGather，全互连 Mesh 拓扑，通用回退方案 |

### 1. Direct 算法

- Root 通过 CCU Write 指令直接向所有 Peer 写入完整数据
- 使用 Event 并行下发所有 Write，EventWait 等待全局完成
- PreSync / PostSync 确保 Address/Token 变量同步

### 2. Parallel Direct 算法

针对 512KB 小包在 16 卡/12 卡场景的优化：
- 将拓扑按网络层拆分为 Layer-0（组内）和 Layer-1（跨组）
- **Root**：双 Thread 并行 Launch 两个 Kernel，分别向 Layer-0 Peer 和 Layer-1 Peer 并行 Write
- **非 Root**：单 Kernel，直接接收来自 Root 的数据
- 通过 Notify 实现主从 Thread 同步

### 3. Parallel Hierarchical 2D 算法

将 16 个 Rank 组织为两组，每组 8 卡，通过 8 条 Lane（通道）实现并行传输：

**Host 侧（exec_op.cc）**：按 256MB 窗口循环，每窗口执行四个 Phase，主从双 Thread 并行 Launch Layer-0 和 Layer-1 Kernel。

**Phase 流程**：
1. **SCATTER_0**：SourceGroup — Root 在本组 Local Scatter；DestGroup —（等待 Layer-1 Cross Scatter 完成）
2. **SCATTER_1**：SourceGroup — Root 在本组 Local Scatter（后半数据）；DestGroup — Destination Root 在组内 Local Scatter。Layer-1 侧 Cross-Scatter all lanes
3. **ALLGATHER_0**：各组内 AllGather 已收到的数据；Layer-1 Cross AllGather A（回传 Source 侧数据）
4. **ALLGATHER_1**：各组内 AllGather 完成全局同步；Layer-1 Cross AllGather B（回传 Destination 侧数据）

**加权分片**：按 3/8 比例拆分数据（先走 Mesh 的数据占 3/8），使 Scatter/AllGather 在各 Lane 上负载更均衡。相较于 50/50 的旧分片，在 2×8 八 lane 场景节省约 15% 延迟。

### 4. NHR1D 算法（4 卡）

参考官方 Broadcast NHR，采用递归减半 Scatter + 递归倍增 AllGather：
- **Scatter 阶段**：`log₂(N)` 步，每步只与一个对端通信，逐步分发数据片段
- **AllGather 阶段**：`log₂(N)` 步，反向倍增收集完整数据
- **Paired Chunks**：将两个 256MB 分片合并到同一 NHR Step 中并行发送，减少同步开销

### 5. Mesh1D 算法

通用回退方案，兼容任意 Rank 规模：
- **Scatter**：Root 将数据均匀切分为 N 个 Slice，通过 Mesh 全互连并行 Write 至所有 Peer
- **Barrier**：全局 Notify 同步
- **AllGather**：每个 Rank 将其 Slice 通过 Mesh 并行 Write 至所有 Peer
- 支持两个 Slice Pair 连续处理（单次 Launch 处理两个 256MB 分片）

## 编译运行

### 编译

```bash
cd code
bash build.sh            # Release 编译
bash build.sh --debug    # Debug 编译（支持断点调试）
bash build.sh --format   # 代码格式化
```

编译产物位于 `code/build/` 目录。

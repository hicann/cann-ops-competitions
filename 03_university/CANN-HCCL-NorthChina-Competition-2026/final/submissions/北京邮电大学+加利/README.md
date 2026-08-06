# HCCL CCU AllGather

本仓库实现 Ascend 950 CCU `HcclAllGather`。每个 Rank 提供 `sendCount`
个元素，最终 `recvBuf` 按 Rank 顺序保存全部 Slice：

```text
rank r 的输出区间 = [r * sliceBytes, (r + 1) * sliceBytes)
sliceBytes         = sendCount * typeSize
actualOutput       = sliceBytes * rankSize
```

三个评测大小 `512 KiB`、`512 MiB`、`400 MiB + 4 B` 都是标称
`recvBuf` 总量，不是单 Rank 输入大小。当前测试平台对 `float32` 实际传入：

| 拓扑 | 标称 512 KiB | 标称 512 MiB | 标称 400 MiB+4B |
|---|---:|---:|---:|
| 2x8 / 16 Rank | 524,288 B | 536,870,912 B | 419,430,400 B |
| 4x1 / 4 Rank | 524,288 B | 536,870,912 B | 419,430,400 B |
| 8+4 / 12 Rank | 524,304 B | 536,868,864 B | 419,426,304 B |

对应的 `sendCount` 分别为 `2x8: 8192/8388608/6553600`、
`4x1: 32768/33554432/26214400`、`8+4: 10923/11184768/8738048`。
Profile 匹配同时接受 dtype/rankSize 向上补齐和平台每 Rank 512 B 向下对齐窗口；地址、
Token 范围、Offset、stripe、DMA 长度与尾块始终使用 API 实际传入的 `sendCount`，不会从
标称大小反推。

## 九类 Profile

下表的 `sliceBytes` 按 `typeSize=1/2/4/8/16` 依次列出。Kernel 数仅指
网络 Transfer Kernel；非标准重叠输入还会增加一个不持有 Channel 的 Prepare Kernel。

| Profile | 实际 `sliceBytes`（B，dtype 1/2/4/8/16） | 通信层与 Peer | 网络 Kernel | 算法与 cell |
|---|---:|---|---:|---|
| `P2X8_512K` | `32768 / 32768 / 32768 / 32768 / 32768` | 本地 7 Peer Mesh + 远端 8 Peer Clos | 2 | 分层 one-shot Direct，cell=实际 Slice |
| `P2X8_512M` | `33554432 / 33554432 / 33554432 / 33554432 / 33554432` | 本地 7 Peer Mesh + 远端 8 Peer Clos | 4 | 默认 Matched-Serial Distributed Root；保留 Root、Capped、11-stripe Hybrid、shared-CKE、single-DMA 和 Rolling-2 |
| `P2X8_400M4B` | `26214401 / 26214402 / 26214404 / 26214408 / 26214416` | 本地 7 Peer Mesh + 远端 8 Peer Clos | 4 | 默认 Matched-Serial Distributed Root；保留 Root、Capped、11-stripe Hybrid、shared-CKE、single-DMA 和 Rolling-2 |
| `P4X1_512K` | `131072 / 131072 / 131072 / 131072 / 131072` | 3 Peer 全部使用 Clos | 1 | 单 Kernel one-shot Direct，cell=实际 Slice |
| `P4X1_512M` | `134217728 / 134217728 / 134217728 / 134217728 / 134217728` | 3 Peer 全部使用 Clos | 1 | 默认 K4 完美匹配顺序 + READY 流水；保留 wait-all single-DMA 和 8 MiB Rolling-2 |
| `P4X1_400M4B` | `104857601 / 104857602 / 104857604 / 104857608 / 104857616` | 3 Peer 全部使用 Clos | 1 | 默认 K4 完美匹配顺序 + READY 流水；保留 wait-all single-DMA 和 8 MiB Rolling-2 动态尾块 |
| `P8P4_512K` | `43691 / 43692 / 43692 / 43696 / 43696` | 8 卡侧 Mesh 7 + Clos 4；4 卡侧 Mesh 3 + Clos 8 | 2 | 分层 one-shot Direct，cell=实际 Slice |
| `P8P4_512M` | `44739243 / 44739244 / 44739244 / 44739248 / 44739248` | 8 卡侧 Mesh 7 + Clos 4；4 卡侧 Mesh 3 + Clos 8 | 4 | 默认 Full-Seed 4/7 实验；保留 Shared 5/8、Shared 5/7、Half-Root、capped、9-stripe Hybrid、single-DMA 和 Rolling-2 |
| `P8P4_400M4B` | `34952534 / 34952534 / 34952536 / 34952536 / 34952544` | 8 卡侧 Mesh 7 + Clos 4；4 卡侧 Mesh 3 + Clos 8 | 4 | 默认 Full-Seed 4/7 实验；保留 Shared 5/8、Shared 5/7、Half-Root、capped、9-stripe Hybrid、single-DMA 和 Rolling-2 |

512 KiB 的每 Rank Slice 只有约 32~128 KiB，因此继续使用不执行分-cell 循环、
Rolling-2 slot、尾块融合和 drain 的
one-shot `RunFixedSmall()`，并保留 DONE-only 完成握手。`4x1` 只有一个 Clos Kernel，
不产生额外 Thread；`2x8` 和 `8+4` 则恢复 `829b010` 已实测更快的本地 Mesh、远端 Clos
分流。相较全 Clos，这会增加一个 Kernel launch；两层位于不同 IO Die，因此还会增加一次
Thread fork/join，但可减少 Clos 竞争并并行使用两个 Die。本地 Copy 与所有远端 Write
均在等待 Event 前提交，保持数据面重叠。

大消息中，`4x1` 没有本地 Peer，仍只使用 Clos；`2x8` 和 `8+4` 把本地 Peer 固定放在
Mesh、远端 Peer 固定放在 Clos。按赛题拓扑，两层各自只落在一个 IO Die，且实际 Die
必须不同，因此每个传输阶段各注册一个 Mesh Kernel 和一个 Clos Kernel，并由用户 Thread
与副 Thread 并行推进。具体 Die 编号来自 Endpoint 属性，不写死 Mesh=Die1 或 Clos=Die0。

Clos 聚合带宽为 `4B`，而本地 Full-Mesh 的每条 `B` 链路独立并行。分层调度让两个 IO
Die 同时工作；`2x8` 和 `8+4` 的 relay 路径进一步用分散到本地 Rank 的 Root 降低 Clos
注入量，且不会形成单 Leader 或单 Lane 热点。Host 仍按角色负载选择 primary，唯一
out-of-place LocalCopy 由轻负载 primary Kernel 在 Phase A 提交，与两层网络传输重叠。

### 2x8 Distributed Root、Hybrid Relay 与静态 A/B

两个 `2x8` 大消息通过 `include/custom.h` 的两个独立常量选择数据路径：

```cpp
constexpr AllGatherDataPath P2X8_512M_DATA_PATH = AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL;
constexpr AllGatherDataPath P2X8_400M4B_DATA_PATH = AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL;
```

两个 Profile 均可在 `DISTRIBUTED_ROOT`、`DISTRIBUTED_ROOT_CAPPED`、
`DISTRIBUTED_ROOT_MATCHED_SERIAL`、`HYBRID_RELAY`、
`HYBRID_RELAY_CKE`、`SINGLE_DMA` 和 `ROLLING2` 之间切换。数据路径进入 Engine Context
tag，避免 A/B 时复用旧 Context。每个 Rank Pair 始终
只申请并复用一个 Channel，跨阶段不会创建第二条。

Distributed Root 对每个源选择远端 Server 的同 slot Rank 作为 Root。令 Slice 元素数为 `N`，
`a=ceil(4N/11)`、`delta=11a-4N`。Root 经 Clos 收到完整 Slice，随后通过七条
本地 Mesh 链路广播 `[0,a)`；其他 Rank 的 `[a,N)` 由源直接发送。默认路由允许单 Peer
动态共享 NPU 的 `4B` Clos 聚合带宽：

```text
Clos-A: offset 0 [0,N); offset 1..4 [a,N)
Clos-B: offset 5..7 [a,N)
Mesh-B: Root 将 [0,a) 分别写给其余 7 个本地 Rank
```

Clos-A 负载为 `5N-4a`，会在耗时 `N/B` 的 Mesh-A 之前完成；Clos-B 可在同一 Clos Thread
立即继续，因此 Clos 总时间为 `(8N-7a)/(4B)`。两个 Clos 阶段的单 Peer 峰值均不超过
`4B/3`。每条有向 Mesh 链路在 Phase A 发送 `N`、Phase B 发送 `a`，整体仍由 Mesh 收敛到：

```text
T = (N+a)/B = (15N+delta)/(11B)
```

连续带宽下界为 `15N/(11B)`，整数取整只增加不足一个元素的传输时间。相较保留的
11-stripe Hybrid，Distributed Root 系列将每 Rank 网络 WQE 从 `65` 降到 `22`：Mesh-A/B
分别为 `7/7`，Clos-A/B 分别为 `5/3`。各阶段只统一 drain 一次 Event，Clos-A 的可见性
屏障也只覆盖 Mesh-B 实际读取的同 slot Root Peer。在单 Root 路由中，8 个远端目标各至少
需要一个 Write，因此 `8` 个 Clos WQE 已是下界；Mesh 两阶段各覆盖 7 个 Peer，总 WQE
同样达到该路由语义的下界。

最终 Mesh-B 仍对 7 个本地 Peer 执行完整 completion。Clos-B 则只向本阶段写入的
`offset 5..7` record，并等待反向 `offset 1..3`；Phase-A 的其余入站数据通过
“源 Rank bit-3 -> 同 slot Root -> Mesh-B completion”形成传递依赖。capped 路径相应使用
`offset 4..7` record、`offset 1..4` wait，避免对没有 Phase-B 数据的 Clos Peer 重复同步。

若真机表明单 Peer 无法共享足够的 Clos 聚合带宽，可选择保留的
`DISTRIBUTED_ROOT_CAPPED`。它令 `b=a-delta`，使用如下
`8+4` 个 Clos WQE：

```text
Clos-A: offset 0 [0,N); offset 1..3 [a,N); offset 4..6 [a,N-a); offset 7 [a,N-b)
Clos-B: offset 4..6 [N-a,N); offset 7 [N-b,N)
```

它让任一 Phase-B Peer 最多发送 `a`，即使单 Peer 额外限制为 `B` 也达到相同带宽下界。
Matched-Serial 使用普通 Distributed Root 的 `5+3` 个 Clos WQE，但逐 Peer 执行
`Write -> EventWait`。低 Rank Server 使用 `selfSlot+offset`，高 Rank Server 使用
`selfSlot-offset`，所以每一轮两端选择同一条双向 Rank Pair；一个远端 Peer 独占该 Rank
的 `4B` Clos。两个 `2x8` 大消息均默认选择此入口，Root 与 Capped 均保留用于 A/B。
旧 Hybrid 同样达到带宽下界，但需要 Mesh `7+28`、Clos `14+16` 个网络 WQE；它仍用于
两个大消息 Profile 的真机 A/B。原 shared-CKE 双 Kernel 逻辑也由
`HYBRID_RELAY_CKE` 完整保留。

四 Kernel 路径均在 Clos-A DMA 和所需远端 bit-3 屏障完成后，由 Host 执行 Clos Thread 到
Mesh Thread 的 Notify，再并行启动 Clos-B 与 Mesh-B。每个 Kernel 只使用所在 IO Die 的
网络设备，不依赖当前 checker 无法建模的跨 Die shared CKE。

### 8+4 Full-Seed、Shared 与静态 A/B

两个 `8+4` 大消息当前都选择 Full-Seed 4/7；原 Shared 5/7、Shared 5/8、
Half-Root、capped、matched、9-stripe Hybrid、
完整 Slice single-DMA 和 Rolling-2 CCU 函数均保留：

```cpp
constexpr AllGatherDataPath P8P4_512M_DATA_PATH
    = AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7;
constexpr AllGatherDataPath P8P4_400M4B_DATA_PATH
    = AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7;
```

两个常量均可回切 `BIDIRECTIONAL_HALF_ROOT`、`BIDIRECTIONAL_HALF_ROOT_CAPPED`、
`BIDIRECTIONAL_HALF_ROOT_SHARED_5_8`、`HYBRID_RELAY`、`SINGLE_DMA` 或 `ROLLING2`；
512 MiB 还保留
`BIDIRECTIONAL_HALF_ROOT_MATCHED`。每条路径使用独立 Engine Context tag。四个阶段中的
收发、同步和地址交换始终复用 Rank Pair 唯一的 Channel，不会为同一个对端创建第二条
Channel。Phase A 和 Phase B 各自使用一个 Mesh CCU Kernel 与一个 Clos CCU Kernel，
两层在不同 IO Die 上并行；Phase-A Clos 完成真实 seed 依赖后，Host Notify 再放行两个
Phase-B Kernel。

令 8 卡侧 Rank 为 `A0..A7`，4 卡侧 Rank 为 `U0..U3`，每 Rank Slice 有 `n` 个元素、
元素字节数为 `d`、总字节数为 `N=n*d`。定义半片字节数 `H` 和 relay 前缀字节数 `Q`：

```text
half = ceil(n / 2),  H = half * d
Shared 5/7: q = ceil(5n / 7), Q = q * d
Shared 5/8: q = ceil(5n / 8), Q = q * d
Full-Seed 4/7: q = ceil(4n / 7), Q = q * d
```

Shared 路径把 Phase A 的 Clos 数据严格缩到 Phase-B Mesh 真正读取的 seed。对每个
`j in [0,3]`：

```text
Phase A Clos: Uj       -> A(2j)      [0, H)
              Uj       -> A(2j+1)    [H, N)
              Aj       -> Uj          [0, Q)

Phase B Mesh: A(2j)    -> A(k!=2j)   [0, H)
              A(2j+1)  -> A(k!=2j+1) [H, N)
              Uj       -> U(k!=j)     Aj 的 [0, Q)

Phase B Clos: Aj       -> U0..U3      [Q, N)
              A(4+j)   -> U0..U3      [0, N)
```

Full-Seed 4/7 保持相同 Half-Root 和 Mesh relay，只把匹配低位 A 的 Phase-A
区间扩成完整 Slice，并在 Phase B 跳过该匹配 U：

```text
Phase A Clos: Aj       -> Uj           [0, N)
Phase B Clos: Aj       -> U(k != j)    [Q, N)
              A(4+j)   -> U0..U3       [0, N)
```

Phase A 中 `Aj -> Uj` 单 Peer 独占 Clos `4B`，完整 Slice 与反向两个 Half-Root
都在 `N/(4B)` 完成。令 `Q=4N/7` 后，4 卡侧 Mesh 与 Clos 的理想关键时间均为
`11N/(7B)`；Phase-A/Phase-B Clos Write 从 Shared 的 `12+32` 降为 `12+28`。

8 个 `A` Rank 各担任一个 `U` 半片 Root，再通过 8 卡 Full-Mesh 把所持半片广播给其余
7 个 Rank。低四个 `A` Slice 的 `[0,Q)` 只跨 Clos 一次到匹配的 `U`，再通过 4 卡
Full-Mesh relay；其尾部在 Phase B 直发四个 `U`。高四个 `A` Slice 不占用 Phase-A Clos，
在 Phase B 直接发给四个 `U`。Phase-B Clos 按 `(sourceGroup + destinationOffset) mod 4`
的相对槽位提交，避免所有 Rank 按绝对 Rank 顺序同时冲击同一个 4 卡侧目标。

Phase A 只对上述 12 条 seed 依赖执行定向 record/wait，共 24 个 Clos Notify 操作。最终
Clos completion 只由 8 卡侧发布、4 卡侧消费；4 卡侧数据在 8 卡侧的可见性由 Half-Root
Mesh completion 覆盖，因此最终同步为 64 个 Clos Notify 操作。原 Half-Root、capped 和
matched Kernel 的同步、数据区间及提交顺序均完整保留用于真机 A/B。

Shared 与 Full-Seed 的 Clos Output Identity 同样按真实方向交换：每个 A 只向其 Half-Root
来源 `U_floor(A/2)` 发布地址和 Token、仍等待全部 4 个 U；每个 U 向全部 8 个 A 发布、
只等待两个 A Half-Root。Clos Identity 控制操作由对称交换的 `192` 降为 `120`。

Shared 路径 Phase A 为 `68` 个 Mesh Write 加 `12` 个 Clos Write，Phase B 为
`68` 个 Mesh Write 加 `32` 个 Clos Write，合计 `180` 个网络 Write；加上每 Rank 一次
out-of-place LocalCopy，Checker 中共有 `192` 个数据任务。Full-Seed 将 Phase B Clos
Write 缩到 `28` 个，合计 `176` 个网络 Write 和 `12` 个 LocalCopy，对应 `188` 个数据任务。
两个 `float32` 精确大消息的几何值为：

| 实际 Output | 路径 | Slice `N` | `n` | `H` | `Q` | `N-Q` |
|---:|---|---:|---:|---:|---:|---:|
| 536,868,864 B | Full-Seed 4/7 | 44,739,072 B | 11,184,768 | 22,369,536 B | 25,565,184 B | 19,173,888 B |
| 536,868,864 B | Shared 5/8 | 44,739,072 B | 11,184,768 | 22,369,536 B | 27,961,920 B | 16,777,152 B |
| 536,868,864 B | Shared 5/7 A/B | 44,739,072 B | 11,184,768 | 22,369,536 B | 31,956,480 B | 12,782,592 B |
| 419,426,304 B | Full-Seed 4/7 | 34,952,192 B | 8,738,048 | 17,476,096 B | 19,972,684 B | 14,979,508 B |
| 419,426,304 B | Shared 5/8 | 34,952,192 B | 8,738,048 | 17,476,096 B | 21,845,120 B | 13,107,072 B |
| 419,426,304 B | Shared 5/7 A/B | 34,952,192 B | 8,738,048 | 17,476,096 B | 24,965,852 B | 9,986,340 B |

### 4x1 大消息独立 Kernel A/B

`P4X1_512M` 和 `P4X1_400M4B` 保留原 `CLOS_DIRECT_ROLLING2` Profile、8 MiB cell
以及 wait-all 完整 Slice Kernel，只在 Host 注册处通过两个独立静态选择替换 CCU 函数入口：

```cpp
constexpr AllGatherDataPath P4X1_512M_DATA_PATH = AllGatherDataPath::MATCHED_SERIAL_ROLLING2;
constexpr AllGatherDataPath P4X1_400M4B_DATA_PATH = AllGatherDataPath::MATCHED_READY;
```

将它改为 `SINGLE_DMA` 可回切先等待全部 Peer READY、再提交三个完整 Slice Write 的旧路径；
改为 `ROLLING2` 则回切原 P4X1 Profile Rolling-2 Kernel。两个开关互不影响，当前只让
512 MiB 进入 Matched-Serial Rolling-2 实验，400 MiB+4B 保持 Matched-Ready。

4x1 没有本地 Peer，三个 Channel 全部属于 layer-1 Clos。每个 Rank 只查询自己到 Peer
的 Link；高 Rank 交换 Endpoint 副本后统一成 `lowRank -> highRank`，再按 CTP/TP、hop、
Endpoint 协议和有效通信地址选择规范候选。`loc`、`superDevId`、远端 Die/BW 和 union
未使用尾部都不参与规范键，因此一对 Rank 可以确定性选择同一 EndpointPair。Link
选定后读取本 Rank 的 `ENDPOINT_ATTR_DIE_ID`。三个 layer-1 Clos Channel 必须属于同一个
IO Die，并统一注册到一个 Clos Kernel；若规范选出的 Endpoint 跨越本地两个 Die则直接
fail closed，不接受与赛题分层拓扑不一致的资源图。

默认路径按 K4 三轮双向完美匹配排列 Channel：Rank 0 为 `[1,2,3]`，Rank 1 为
`[0,3,2]`，Rank 2 为 `[3,0,1]`，Rank 3 为 `[2,1,0]`。每列都是两组互不冲突的
Rank Pair，单个 layer-1 Kernel 必须完整保持全部三轮顺序。

READY 流水先向本 Kernel 的全部 Channel 发布 Output 地址和 Token，再立即启动唯一的
out-of-place LocalCopy；随后按上述顺序逐 Peer 等待 `PARAM_READY`，每满足一个 Peer
就立即提交对应的完整 Slice Write。必须先完成全部发布再进入首个 Wait，避免不同 Rank
的首发顺序形成等待环。Kernel 对三个 Channel 各提交一次 Write，因此每 Rank 始终只有
3 次完整 Slice Write 和每对端一个 Channel。

Matched-Serial Rolling-2 在同一顺序上逐 Peer 串行；每个 Peer 的 Slice 优先按 16 MiB
切分，并用两个 Event slot 流水 Write，复用 slot 前等待对应 DMA。512 MiB 时每个 Peer
依次传输八个 16 MiB cell，当前 Peer 的两个 slot 全部 drain 后才进入下一 K4 matching
wave，使一个 Channel 独占 Clos；全部三轮结束后仍复用原 Pairwise completion。

512 MiB 标称 Output 的 Slice 为 128 MiB，Write 数从 `3 * 16 = 48` 降至 3；
400 MiB+4B 的 Slice 为 `100 MiB + dtypeSize`，Write 数从 `3 * 13 = 39` 降至 3。
该入口不申请新 staging buffer，也不改变 2x8、8+4、小消息或 Generic 的注册和资源图。

## RankGraph 与资源分组

Host 通过以下接口识别真实拓扑，不按 Rank 编号猜测 Server：

- `HcclRankGraphGetLayers` 获取并立即复制 Layer 列表，明确选择 layer-0 Mesh 和
  layer-1 Clos，忽略 Host/RoCE 等附加层；
- `HcclRankGraphGetTopoTypeByLayer` 验证 layer-0 是 `COMM_TOPO_1DMESH` 或平台用于
  Full-Mesh 的 `COMM_TOPO_CUSTOM`，layer-1 是 `COMM_TOPO_CLOS`；
- `HcclRankGraphGetInstSizeListByLayer` 验证 `4x1`、`8+4`、`2x8` 的实例形状；
- `HcclRankGraphGetRanksByLayer` 得到当前 Rank 的本地 Mesh 实例成员，并以补集得到远端 Peer；
- `HcclRankGraphGetLinks` 在 Profile 指定的精确 Layer 上收集 CTP/TP 链路；
- `ENDPOINT_ATTR_DIE_ID` 和 `ENDPOINT_ATTR_BW_COEFF` 从本 Rank 的本地 Endpoint 查询。

每个 Rank 对都固定按 `(minRank, maxRank)` 方向查询 `HcclRankGraphGetLinks`。候选先按
CTP/TP 和 hop 排序，再用完整 Endpoint 语义字段作为严格 tie-break；较大 Rank 只在写入
`HcclChannelDesc` 时交换 src/dst。这样不依赖链路数组顺序，也不需要查询远端 Endpoint
属性；选定 pair 后才读取本 Rank 的 Die 用于分组与审计。带宽属性只记录为诊断信息，
当前选链和负载分配使用可由两端复现的 canonical key 与固定 4:1 role 权重。

选中的 Channel 按 `{layerRole, topoType, layer, actualDieId}` 分组。同一个 CCU Kernel
只持有同一 Layer、同一 IO Die 的 Channel。大消息 `2x8`/`8+4` 要求 Mesh 恰好一个
单 Die group、Clos 恰好一个单 Die group，且两个实际 Die 不同；Host 创建资源和 Exec
反序列化后都会独立复核该约束，以及 Mesh/Clos Peer 并集、唯一 primary 和完整且无重复
的全 Peer 覆盖。小消息和 Generic 仍保留按真实 Endpoint 分组的通用资源范围。

`4x1` 的本地实例只有当前 Rank，不产生本地通信；若 RankGraph 合法省略 singleton
Mesh 层，Host 仍会以 `{myRank}` 建立本地成员集合，并只使用显式 full-rank Clos 层。
`2x8` 和 `8+4` 要求 RankGraph 同时提供合法的 layer-0 Mesh 与 layer-1 Clos。

## Host 到 CCU

执行流程如下：

```text
HcclAllGather
  -> 按 rankSize + 实际 Output 补齐区间选择九类 Profile 或 Generic
  -> Host 按静态 data-path 常量选择 Distributed Root、Half-Root、Hybrid、Matched-READY、single-DMA 或 Rolling-2
  -> 首次调用读取 RankGraph、为每个 Peer 确定性选择 EndpointPair
  -> 4x1 的三个 layer-1 Channel 必须属于同一 IO Die并注册为 1 个 Clos Kernel
  -> 2x8/8+4 大消息按实际 Endpoint Die 各注册 1 个 Mesh 和 1 个 Clos Kernel
  -> 序列化 Engine Context（Profile/algorithm/dtype/cell/layer/die/peerMask）
  -> ExecOp 反序列化并重新审计资源形状
  -> 获取覆盖实际 Input/Output 范围的 Token
  -> 必要时 Prepare 部分重叠输入
  -> 覆盖双 Die 时由 Thread Notify fork；每个 Die 内按 Mesh -> Clos 启动，返回前再 join
  -> CCU 交换 Output 地址/Token，Push 数据，等待或 drain Event，再完成 DONE
```

九个固定 Profile 有独立 Kernel wrapper，wrapper 会复核 Profile ID、实际算法、Rank 数、
层角色、dtype 补齐量子、实际 Slice 和 cell；底层共用不含任何归约语义的 Direct Push 核心。
代码中没有 `ReadReduce`、`WriteReduce`、`LocalReduce`、ReduceScatter 或 Owner Accumulate。

## Buffer 与尾块

Out-of-place 直接从 `sendBuf` Push，且只有主 Transfer Kernel 负责把本 Rank Slice
复制到最终 Offset。标准 in-place 使用
`recvBuf + rank * sliceBytes` 作为源，不提交同址 LocalCopy。其他 Input/Output 重叠
先使用有界 HCCL bounce buffer，按 memmove 方向搬到标准位置；Prepare 全部完成后才
允许 Transfer Kernel 交换 READY。

Rolling-2 大消息每个 cell 的 Offset 从 0 递增，最后一次长度严格为
`sliceBytes - cellOffset`。不超过 4 KiB 的小余数可并入前一个 cell，但不会截断、
重叠或越界。通用回退由 Host 按不超过 128 MiB 的动态块覆盖任意合法 Slice 和不整齐
尾部；`4x1` 使用 Clos，`2x8`/`8+4` 沿用本地 Mesh、远端 Clos 的分区。不同 Profile
通过 Engine Context tag 隔离资源，Context 内还会保存并复核实际 Layer、Die 和 Peer 集。
single-DMA、Rolling-2、Hybrid stripe 和 Half-Root 区间都直接使用调用实际的
`fixedSliceBytes`。因此
平台将标称 `400 MiB + 4 B` 向下对齐为表中的实际 Output 时不会按标称大小越界；其他
dtype 产生向上补齐时也不会截断尾部。

Generic 是未命中九类固定 Profile 时的正确性基线。固定 Profile 若无法满足单 Kernel
单 layer/die、Clos 本地单 Die 或 Kernel 数等资源约束，当前实现会 fail closed，不允许
某个 Rank 单方面切换 Generic；
否则不同 Rank 可能建立不同 Channel 图并在 READY 阶段死锁。若平台后续要求资源失败
自动降级，需要先提供全 Rank 一致的算法决策或失败汇聚机制。

Rolling-2 路径的每个 Event 槽在复用前先等待旧 DMA，退出前 drain 全部活跃槽；两个
完整 Slice Kernel 都只申请一个 Event 槽。`SINGLE_DMA` 等待全部 READY 后提交 Write，
`MATCHED_READY` 则在每个 Peer READY 后立即提交对应 Write；二者都在所有 Write 和可选
LocalCopy 提交后统一等待。所有路径都只在 DMA 完成后发布并消费 Channel DONE。下一次串行调用必须先执行
`ExchangeOutputIdentity()`；对端只有
消费上一轮 DONE 后才能进入下一轮并发布 `PARAM_READY`，因此下一轮参数 READY 同时
充当上一轮 DONE 的隐式 ACK，不再提交第二轮 ACK 控制任务。

上述复用保证针对同一 communicator 的串行调用。不同 Stream 若并发命中同一 Context，
会共享 Channel XN、Channel Notify 和副 Thread Notify；除非 HCCL 上层保证 communicator
内串行化，否则需要额外的调用级隔离，当前实现不将这种并发场景声明为安全。

## 构建

配置 CANN 环境后可使用 `bash build.sh` 构建。

## 验证口径

HCCL-VM `float32` 精确驱动应使用下表参数；`actualOutputBytes` 以测试平台传入值为准：

| 拓扑 | 标称 Output | `sendCount` | 实际 `sliceBytes` | 实际 Output |
|---|---:|---:|---:|---:|
| 2x8 | 512 KiB | 8,192 | 32,768 B | 524,288 B |
| 2x8 | 512 MiB | 8,388,608 | 33,554,432 B | 536,870,912 B |
| 2x8 | 400 MiB + 4 B | 6,553,600 | 26,214,400 B | 419,430,400 B |
| 4x1 | 512 KiB | 32,768 | 131,072 B | 524,288 B |
| 4x1 | 512 MiB | 33,554,432 | 134,217,728 B | 536,870,912 B |
| 4x1 | 400 MiB + 4 B | 26,214,400 | 104,857,600 B | 419,430,400 B |
| 8+4 | 512 KiB | 10,923 | 43,692 B | 524,304 B |
| 8+4 | 512 MiB | 11,184,768 | 44,739,072 B | 536,868,864 B |
| 8+4 | 400 MiB + 4 B | 8,738,048 | 34,952,192 B | 419,426,304 B |

安装版 `hccl_test` 还会执行额外对齐，因此验证实际参数时使用
`exact_allgather <sendCount> float32 out 1`。双 Die Hybrid 仿真使用
`ascend950_server_topo_normal.yaml`：该模型的 Full-Mesh 全在 Die1、Clos 全在 Die0，
与赛题“两层位于不同 IO Die”的约束一致。competition 模型即使修正了非连续设备列表，
其 Mesh Endpoint 仍横跨两个 Die，只适合验证通用分组，不满足 Hybrid 的两 Kernel前提。

HCCL-VM `--check-only` 可以验证 CCU 翻译、SingleTask、MemConflict 和 SemanticCheck，
但不提供可信的真实 NPU 性能或逐字节搬运结论；最终延迟仍需在 Ascend 950 硬件上 A/B。
当前默认四 Kernel 路径已通过 `2x8 / 512 MiB` 以及 `8+4 / 512 MiB`、
`8+4 / 400 MiB+4B` 的 `float32 / out-of-place` 精确用例。checker 对 `2x8` 展开 64 个
CCU graph；Full-Seed 4/7 的两个 `8+4` 大消息均展开 48 个 CCU graph，生成
`taskNodeCount=1240` 和 `dataTaskNodeCount=188`。保留的 Shared 5/8
`8+4 / 400 MiB+4B` 回归图为 `1248/192`。所有用例的
GenGraph、SingleTask、MemConflict、SemanticCheck 和最终 `op[0] Checker Success` 均成功。
HCCL-VM 只验证图、依赖、内存冲突与语义，不测真实链路延迟；`8+4` 最终仍需在
Ascend 950 上对默认与 capped 路径做延迟 A/B。临时将 `400 MiB+4B` 切换为 capped
的精确用例也通过相同检查，仍为 48 个 CCU graph，数据任务按预期从 `188` 增至 `196`。

当前两个 `2x8` 大消息的 Matched-Serial 图均为 `taskNodeCount=2336`、
`dataTaskNodeCount=368`；后者等于 16 Rank 各 `22` 个网络 Write 加 1 个 LocalCopy。
`512 MiB` 的语义统计为 272 段、`9,126,805,504 B`，`400 MiB+4B` 为 272 段、
`7,130,316,800 B`。保留的 Capped 图每 Rank 为 `26` 个网络 Write 加 1 个 LocalCopy，
因此其 `dataTaskNodeCount=16*(26+1)=432`。

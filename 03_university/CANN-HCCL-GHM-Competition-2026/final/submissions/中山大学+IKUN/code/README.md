# ReduceScatter 集合通信算子

# F1R72 final isolated thin paths

F1R72 keeps the F1R71 large/tail selection.  It fixes the sole R138 failure
by materializing the rank-4 SMALL remote XN address only after its ready wait.
It also enables the statically complete rank-16 SMALL thin entry and the
teammate-exact single-chunk 8+4 SMALL body.  These thin entries preserve the
measured communication graphs and only remove unreachable mode/chunk code.

# F1R71 measured-oracle large floor plus rank4-small RH

F1R71 starts from the exact F1R57 platform-measured source. Its six
large/tail cells select only between the F1R51 and F1R57 dynamically proven
chunk schedules on a topology-by-size basis. Rank-4 SMALL is the remaining
two-stage xor2/xor1 recursive-halving experiment. Rank-16 SMALL and 8+4 SMALL
remain on the exact F1R57 measured paths.

This tree keeps the F1R51 measured-correct CCU resource and contribution
graphs, but selects performance transforms with three topology-specific
models:

- `2x8`: capacity-constrained minimax `9/12/11` for 512 MiB and
  `4/8/10/10` for the 400 MiB tail,
  with the server-remote group launched first and owning OUTPUT;
- `4x1`: `10/13/9` and `7/9/10/6`, retaining the local owner because this
  topology has no cross-server critical group;
- `8+4`: the teammate all-peer/fixed-tree kernel with `9/13/10` and
  `5/9/9/9`, plus server-remote-first OUTPUT ownership.

R127 supplied the missing hardware constraint for the 2x8 512 MiB point:
the old `13/32` stripe was 13,631,488 bytes, above the executable path's
13,107,200-byte capacity. Exhaustive three-part compositions of denominator
32 under that exact capacity leave `9/12/11` as the unique minimax flow-shop
ratio. Its largest stripe is 12,582,912 bytes, leaving 524,288 bytes of
capacity margin. All other scoring graphs remain byte-for-byte unchanged.

`slotStride` remains the maximum scratch-bank extent. Input, OUTPUT and
private-partial coordinates advance by cumulative actual chunk sizes, so the
non-uniform stripes cover each rank slice exactly without aliasing. No remote
endpoint lookup, cross-Die CCU object, or cross-API identity join is added.

# F1R51 dual-bank early-fence wavefront

This tree is the compile-correct lowering of F1R50. It replaces three unsupported
`CondExpr && CondExpr` expressions with equivalent nested `CCU_IF` blocks; the
network, reduce, fence, contribution, scratch, and Host schedules are unchanged.

## Expanded executable-IR winner

Large scoring paths use two disjoint scratch banks. Chunks 0/1 are
queued before the first local tree; each bank is reused only after its chunk
has completed the deterministic local tree.  Once all Reads are queued, the
completion record is issued before the remaining local trees, so remote
completion waiting overlaps local reduction without paying the three-bank
chunk-size/resource penalty.
completes, then chunk3 is queued. The per-channel completion record follows
the last queued Read and overlaps the remaining local trees. The 4x1
large/tail paths use four balanced chunks.

The exact 8+4 Host/kernel identity remains intact. A non-primary group's
peer0 now lands directly in its private partial root, removing one scratch
slot per bank and one terminal LocalCopy. This funds the third bank without
address aliasing or changing peer order, contribution multiplicity, network
bytes, localEndpoint-only selection, or single-Die kernel registration.

## 118771 hardware-label calibration

F1R46 representative 118771 measured
`23/1620/1260/20/2490/1920/24/2100/1660 us`.  It proves that the
topology-size wrapper specialization and the combined rank12
ready-pair/pre-join transform did not deliver the v22 engineering-center
gain.  F1R48 therefore restores the exact measured-fast F1R43 integrated
kernel identity for points 10-15 and the exact teammate full-mask kernel and
join-before-merge order for points 16-18.

The scoring rank12 path is a single Host super-chunk.  After the unique worker
join and all fixed-order partial merges, no worker work or scratch successor
remains, so the terminal two-thread release handshake is elided only for the
three normalized scoring families.  OTHER sizes keep the release handshake.

## F1R47 teammate-exact super-chunk output offset

F1R46 的九点评分执行图保持逐字节不变。独立修复只作用于队友 exact Host 路径
超过四个内部 chunk 时的第二个及后续 super-chunk：输入偏移已经按
`superChunk * maxChunk`前进，OUTPUT/private partial 的本地输出基址现在也前进
相同字节数，避免后批覆盖前批开头。

R116 的唯一失败边界每 rank 输出为`80MiB+4B`，实际分成八个内部 chunk。
旧代码第二批重新从 OUTPUT/partial offset 0 写入，最后整块 merge 读取了未物化
区域。F1R47 为每批使用互斥输出区间；四个及以下 chunk 的九点评分路径偏移恒为
零，故其 Host/CCU DAG、网络字节、贡献树、chunk 深度和冻结性能预测均不变。

## F1R46 scoring-family scoped pre-join

F1R45 的九点评分执行图、专用 CCU 入口、内部 chunk、ready-pair 与
pre-join partial merge 全部保持不变。唯一算法修正是把 pre-join 的适用域显式
限制到规范化 SMALL/LARGE/TAIL 三个评分族；其他尺寸（包括
`8+4 / 1006633008` 超重控制）恢复队友已经通过的 join-before-merge 顺序。

R115 已证明：超重控制在 CCU 端选择 `TEAMMATE_EXACT_GENERAL`，但 F1R45 Host
错误地无条件继承了评分图的 pre-join，导致 partial 根尚未被语义图证明物化时就
被合入 OUTPUT。F1R46 不改变任何评分点的 Host/CCU DAG，只修复该跨
size-family 标签迁移错误。

## F1R45 topology×size specialized ready tree

统一可执行 IR 扩域搜索的唯一赢家。九个评分坐标使用独立 EngineCtx 与专用
CCU 入口；8+4 保留独立队友 Host/kernel 框架，并加入已有动态先例的
ready-pair 与主线程 partial pre-join merge，终态 release 仍保留。

## F1R41 v2 九点 balanced direct-root tree

九个评分坐标全部选择 mode12。队友v5公开实测成立的 all-peer Read、完整read
mask、固定树归约、扁平多kernel发射以及
`Issue0/Issue1/Reduce0/Issue2/Reduce1...`流水直接作为可信基线，不占用一次
“复现提交”。

在该基线上，本树加入四项可审计增量：

1. 按拓扑固定为`1/3/3、1/2/2、1/3/3`个均衡内部chunk，避免buffer填满策略在
   400MiB+4B尾包形成极小末chunk；
2. non-primary kernel让peer0直接写自己独占的private partial并作为固定树根，
   删除每个chunk的末端root copy；
3. private partial按当前连续super-chunk一次LocalReduce合入OUTPUT，不拆成逐chunk
   merge；
4. 九点评分均为单Host super-chunk，删除末尾冗余双向release；若通用控制产生多个
   super-chunk，仍在相邻批次间保留完整release。下一调用的startup record在main
   流上排在前次merge之后，继续保护跨调用复用。

真实OUTPUT裁剪、互斥奇偶scratch、superOffset、private partial、每源一次、
网络字节、localEndpoint-only和`(layer,scope,die)`单Die kernel均保持不变。

## F1R36 九点线程级 scratch 复用与 frontier-4

本树把初赛资源模型中的 buffer 生命周期分析迁移到决赛 CCU engine，但不迁移
初赛微秒系数。九个评分坐标全部选择 mode7：同一 Host thread 上不同 phase 的
kernel 按序执行，复用该 thread 独占的三块 scratch；两个并发 Host thread 使用
不同 scratch owner。每个网络 kernel 先把规范 seed 与最多三个升序 peer 放入四个
互斥地址，统一等待 event mask，再按规范 FP32 顺序 LocalReduce，其余 peer 保持
ReadReduce。网络字节、贡献集合和跨 Rank 同步不变。

按 thread 而不是按 kernel 预留 scratch 后，2×8/8+4 的评分路径仍为 8 个 buffer
slot，4×1 为 3 个 slot；512KiB、512MiB、400MiB+4B 九点全部保持单 Host chunk。
非 primary kernel 继续使用互斥 private partial，所有 phase 完成后才在 primary
kernel 上按固定 group 顺序合并。

## 继承的同组有界 depth-2 与私有 partial 说明

RankGraph 中同一 layer 可以包含来自两个本地 IO Die 的 endpoint，因此一个
layer 不能直接等同于一个 CCU kernel。本实现先为每个对端选择双向一致的 CTP
链路，再根据最低层 Rank 列表区分 Server 内/Server 间通信，最终按
`(layerId, scope, localDieId)` 分组。每个 kernel 只包含同一 layer、同一通信
范围且同一 IO Die 的网络设备。赛题的两层双 Die 拓扑最多生成四组，2×8 参考
拓扑实际生成 `layer-0/local/die-0`、`layer-0/local/die-1`、
`layer-1/remote/die-0` 三组。

端点属性查询严格复现已经通过官方三通信域控制与 CANNJudge 的 API 口径：
对当前 Rank 已选择链路的 `srcEndpointDesc`（即该 channel 的
`localEndpoint`）直接调用 `HcclRankGraphGetEndpointInfo` 查询本地 Die。
不同 RankGraph API 返回的 descriptor 不按地址强制 join；代码也不查询、
枚举或推断远端 Rank 的 Die。

跨 Rank 的共同阶段只包含全局 `(layerId, scope)`，本地 Die 不参与远端阶段
排序。同一阶段内位于两个本地 Die 的 network kernel 由两个 Host thread
同时释放，因而物理边两端使用不同 Die 时也不会形成相反的跨 Rank 顺序环。
每个 network kernel 写独立 scratch/partial；只有 primary kernel 可写最终输出。
全部 network phase 完成后，复用 primary kernel 的纯本地模式，按规范化 group
顺序把其余 partial 固定次序合并。全程不使用 Host LocalReduce、跨 Die
SharedNotify 或两个 network kernel 对同一输出的并发规约。

网络基线仍按对端 Rank 升序执行。只有规范化总输入对应 512MiB 的调用会在同一个
`(layerId, scope, localDieId)` kernel 内启用有界 depth-2：两个对端 Read 写入
两个互斥 scratch、分别使用 event bit 0/1，一次等待该批次的完整 mask，然后仍按
原始对端 Rank 顺序执行 `LocalCopy/LocalReduce`。因此跨 layer、跨 Server 范围或
跨 IO Die 的通信绝不会被放入同一 CCU kernel，也不改变 FP32 累加顺序。
512KiB 与 400MiB+4B 保持 depth-1 控制路径。

chunk 由 Host 外层驱动；单次 kernel 只处理一个 chunk，最多仍显式限制为 10。
紧凑 kernel 不创建 `Loop`、`LoopGroup`、block CKE 或 block buffer。按源码
对象数加保守翻译余量，每 kernel continuous-XN/GSA 上界为 64/32；同一 Die
最多两个 kernel，合计 128/64，显著低于 400/400 配额，mission 恰为 2/2。
depth-1 时每组一个 scratch，depth-2 时每组两个 scratch；每个非 primary 组
仍只有一个 partial。最大四组分别占七个或十一个 CCL tile；所有地址区间互斥。
显式 release/terminal handshake 保证下一 chunk 不会覆盖尚未合并的 partial。

## 1. 项目介绍

```
├── CMakeLists.txt                  # 顶层 CMake 配置
├── build.sh                        # 构建脚本
├── .clang-format                   # 代码风格配置
├── include/                        # 头文件目录
│   ├── hccl.h                      # 集合通信算子头文件
│   ├── common.h                    # 通用数据结构定义
│   ├── custom.h                    # ★ 选手编写：自定义数据结构定义
│   ├── log.h                       # 日志宏定义
│   └── binary_stream.h             # 序列化类定义
├── op_host/                        # Host侧代码目录
│   ├── reduce_scatter.cc           # ★ 选手编写：Host侧资源申请逻辑
│   └── exec_op.cc                  # ★ 选手编写：通信算法编排逻辑
└── op_kernel_ccu/                  # CCU侧代码目录
    └── ccu_kernel.cc               # ★ 选手编写：通信算法编排逻辑
```

> [!NOTE] 注意：
> 算子工程中已提前预制好固有逻辑，选手仅允许修改 `custom.h`、`reduce_scatter.cc`、`exec_op.h`、`exec_op.cc`、`ccu_kernel.h`、`ccu_kernel.cc` 共 6 个文件内容。

## 2. 编译运行

### 2.1 安装 CANN-Toolkit 包

请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260701000328953/)，根据产品型号和环境架构下载对应软件包。安装命令如下，更多指导参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》。

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_9.1.0_linux-${arch}.run
# 安装命令
./Ascend-cann-toolkit_9.1.0_linux-${arch}.run --full --install-path=${install_path}
```

### 2.2 环境变量配置

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

### 2.3 编译算子工程

```bash
bash build.sh

# 编译 Debug 版本，便于断点调试
bash build.sh --debug
```

## 3. 代码格式

选手代码需符合 [.clang-format](.clang-format) 文件中的代码风格规范，可通过下列命令一键修改：

```bash
bash build.sh --format
```

# AllGather 集合通信算子

## 1. 项目介绍

本工程实现基于 CANN 9.1.0 CCU 通信引擎的 `HcclAllGather` 接口。每个 Rank 提供一段 FP32 输入，算子按照 Rank ID 顺序将所有输入拼接到 `recvBuf`，并使通信域内每个 Rank 获得相同的完整结果。

工程面向 Ascend 950 仿真评测环境，对 2×8、4×1、8+4 三类拓扑分别构造 Channel、IO Die、Worker 和 CCU Kernel，并按单 Rank 数据量进一步区分低时延与带宽路径。对于最终输出量约为 512 KiB 的场景，工程还提供独立的短消息快速路径。

```text
├── CMakeLists.txt                  # 顶层 CMake 配置
├── build.sh                        # 构建、Debug 构建和格式化脚本
├── .clang-format                   # 代码风格配置
├── LICENSE                         # CANN 开源软件许可协议
├── README.md                       # 项目说明
├── include/                        # 头文件目录
│   ├── hccl.h                      # 集合通信算子公开头文件
│   ├── common.h                    # 通用常量与 OpParam 定义
│   ├── custom.h                    # ★ 自定义 Kernel 参数和资源上下文
│   ├── log.h                       # 日志宏定义
│   └── binary_stream.h             # 资源上下文序列化工具
├── op_host/                        # Host 侧代码目录
│   ├── CMakeLists.txt              # Host 动态库构建配置
│   ├── allgather.cc                # ★ API 入口、拓扑解析与资源申请
│   ├── exec_op.h                   # ★ 执行计划、拓扑和载荷类型定义
│   └── exec_op.cc                  # ★ Worker 调度与通信算法编排
└── op_kernel_ccu/                  # CCU 侧代码目录
    ├── CMakeLists.txt              # 将 CCU 源码加入 libhccl.so
    ├── ccu_kernel.h                # ★ CCU Kernel 声明和上下文定义
    └── ccu_kernel.cc               # ★ CCU 数据传输与同步图
```

> [!NOTE] 注意：
> 算子工程中已提前预制好固有逻辑，选手仅允许修改 `custom.h`、`allgather.cc`、`exec_op.h`、`exec_op.cc`、`ccu_kernel.h`、`ccu_kernel.cc` 共 6 个文件内容。

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

`build.sh` 从 `ASCEND_HOME_PATH` 读取 CANN 安装位置；若该变量未设置，脚本会在配置前退出。

### 2.3 编译算子工程

```bash
bash build.sh

# 编译 Debug 版本，便于断点调试
bash build.sh --debug
```

默认使用 GNU C++17、`-O2`、`-Wall`、`-Werror` 和 `_GLIBCXX_USE_CXX11_ABI=0`。构建完成后的主要产物为：

```text
build/
├── include/hccl.h
└── lib64/libhccl.so
```

若需从头配置，可在确认目录正确后删除旧的 `build/`，再重新执行构建脚本。

## 3. 代码格式

选手代码需符合 [.clang-format](.clang-format) 文件中的代码风格规范，可通过下列命令一键修改：

```bash
bash build.sh --format
```

格式化命令同样会先检查 CANN 环境变量，因此应先执行 `set_env.sh`。

## 4. 算子接口与数据语义

### 4.1 函数原型

```cpp
HcclResult HcclAllGather(
    void *sendBuf,
    void *recvBuf,
    uint64_t sendCount,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream);
```

参数含义如下：

| 参数 | 方向 | 说明 |
| --- | --- | --- |
| `sendBuf` | 输入 | 本 Rank 的源数据地址。 |
| `recvBuf` | 输出 | AllGather 完整结果地址。 |
| `sendCount` | 输入 | 本 Rank 输入的元素个数，不是字节数，也不是最终输出元素数。 |
| `dataType` | 输入 | 当前实现仅支持 `HCCL_DATA_TYPE_FP32`。 |
| `comm` | 输入 | HCCL 通信域。 |
| `stream` | 输入 | 当前 Rank 使用的 ACL Runtime Stream。 |

返回 `HCCL_SUCCESS` 表示任务成功下发并满足本实现的完成语义；参数、拓扑、资源或 CCU 调用失败时返回相应错误码。

### 4.2 大小计算

工程始终按下式计算本 Rank 输入量和最终输出量：

```cpp
rankBytes   = sendCount * sizeof(float);
outputBytes = rankBytes * rankSize;
```

结果布局为：

```text
recvBuf + 0 * rankBytes  <- rank 0 输入
recvBuf + 1 * rankBytes  <- rank 1 输入
...
recvBuf + r * rankBytes  <- rank r 输入
```

因此，本 Rank 的本地拷贝目标为 `recvBuf + myRank * rankBytes`。当 `sendBuf` 已经等于该地址时，OwnCopy 会被跳过。

### 4.3 输入约束

- 所有 Rank 的 `sendCount` 和 `dataType` 必须一致。
- `rankSize` 必须位于 `[1, 16]`，且 `myRank < rankSize`。
- 当前只接受 FP32；其他类型返回 `HCCL_E_NOT_SUPPORT`。
- `rankBytes` 与 `outputBytes` 均进行 `uint64_t` 溢出检查。
- 每对通信端点仅申请一条 Channel；增加重复 Channel 不会带来收益，也违反赛题约束。

## 5. 总体执行流程

一次普通调用的主路径为：

```text
HcclAllGather
  ├─ 参数、数据类型和溢出校验
  ├─ DFX 注册，查询 myRank / rankSize
  ├─ 按 rankSize、rankBytes、outputBytes 选择稳定的 EngineCtx tag
  ├─ HcclThreadAcquireWithStream：绑定当前 stream 的主线程
  ├─ HcclEngineCtxGet
  │    ├─ 命中：直接复用已注册的 Channel、Thread 和 Kernel
  │    └─ 未命中：BuildResource
  │         ├─ 查询 RankGraph、选择物理链路
  │         ├─ 每个 Peer 申请一条 Channel
  │         ├─ 按真实 local IO Die 分组
  │         ├─ 申请 Worker / OwnCopy Thread
  │         ├─ 注册拓扑专用 CCU Kernel
  │         └─ Serialize → HcclEngineCtxCreate / Copy
  ├─ ExecOp：反序列化 AlgResourceCtx 并构造 ExecPlan
  ├─ Host Record / Wait 编排多个 Worker
  ├─ HcommCcuKernelLaunch
  └─ 等待必要的网络、OwnCopy 和线程完成依赖后返回
```

Channel、额外 Thread、Kernel 注册和 `AlgResourceCtx` 序列化只在对应 EngineCtx 首次创建时发生。后续相同通信域、拓扑和消息档位复用 EngineCtx，但普通 `ExecOp` 仍会反序列化资源并生成本次执行计划。

最终输出约 512 KiB 的稳定会话还具有更早的 Host 快速入口，详见第 9 节。

## 6. 拓扑识别与资源构建

### 6.1 拓扑类型

`exec_op.h` 定义以下逻辑拓扑：

| `TopologyType` | 典型 Rank 数 | 含义 |
| --- | ---: | --- |
| `SINGLE_RANK` | 1 | 无网络通信，只执行必要的本地拷贝。 |
| `MESH_4X1` | 4 | 四个 Server 各一张卡，主要使用 CLOS 链路。 |
| `DUAL_SERVER_2X8` | 16 | 两个 Server，各 8 张卡；机内 Mesh、机间 CLOS。 |
| `ASYMMETRIC_8P4` | 12 | 8 卡 Server 与 4 卡 Server 组成的非对称拓扑。 |
| `GENERIC` | 其他或专用解析失败 | 按 IO Die 分组的通用 Peer-Lane Pull 回退。 |

物理资源以 RankGraph 查询结果为准，而不是仅凭 Rank ID 猜测。Host 侧优先选择 `COMM_PROTOCOL_UBC_CTP` 链路，读取本地端点的 `ENDPOINT_ATTR_DIE_ID`，再按真实 IO Die 组织 Channel。

> [!IMPORTANT]
> 一个 CCU Kernel 中使用的全部网络设备必须属于同一个 IO Die。把跨 Die 任务放入同一 Kernel 的不同 Worker 不能突破这一限制。本工程通过 `ChannelGroup`、`threads` 和 `ccuKernels` 的一一对应关系保证该约束。

### 6.2 2×8：16 Rank

每个 Rank 向其余 15 个 Rank 建立 Channel。`V22GroupChannelsByDie` 要求准确形成两组：

- 机内组：7 条 Channel；
- 跨 Server 组：8 条 Channel。

两组分别部署在其实际 IO Die。低时延路径由主线程和跨 Die Worker 并行启动；带宽路径还可能申请独立 OwnCopy Thread，使本地拷贝与网络传输重叠。

### 6.3 4×1：4 Rank

Host 侧按以下优先级选路：

1. 对大消息尝试双平面：主平面含 2 个 Peer，Worker 平面含 1 个 Peer，且两个平面位于不同 IO Die；
2. 尝试公共 Die：三个 Peer 的可用链路全部落在同一个本地 IO Die，优先 CLOS、高带宽和较优层级；
3. 若专用布局无法成立，回退通用按 Die 分组算法。

公共 Die 低时延路径只需一个 Kernel；双平面带宽路径需要两个 Kernel，并使用独立 OwnCopy Thread。

### 6.4 8+4：12 Rank

对大 Server Rank（`rank < 8`），两组 Channel 通常为 7 条机内链路和 4 条跨 Server 链路；对小 Server Rank（`rank >= 8`），两组通常为 3 条机内链路和 8 条跨 Server 链路。

低时延模式每个 IO Die 各部署一个 Direct-Push Kernel。大消息模式使用 Cross Seed、Cross Direct、Intra Original 和 Intra Relay 组成流水。最终输出约 512 KiB 时，小 Server Rank 会把 8-Channel 跨 Server 组放在主线程，把 3-Channel 机内组放在 Worker，以便优先发出更重的跨 Server 任务；OwnCopy 仍由机内组负责。

### 6.5 通用回退

若专用拓扑校验失败，`AcquireChannels` 会为每个 Peer 搜索可用 UBC_CTP 链路，按本地 IO Die 分组，并限制：

- 最多 2 个 IO Die 组；
- 每组最多 8 条 Channel；
- Peer Rank 不重复；
- 一个 Kernel 只接收一个 Die 的 Channel。

随后每组注册一个 `CcuPeerLanePullKernel`，由可靠的通用模式完成 AllGather。

## 7. 消息分类与算法分流

### 7.1 载荷类型

`SelectExecPlan` 根据单 Rank 输入量选择载荷类型：

| 条件 | `PayloadType` | 主要策略 |
| --- | --- | --- |
| `rankBytes <= 1 MiB` | `LATENCY` | 少 Kernel、少参数、OwnCopy 尽量内联。 |
| `rankBytes > 1 MiB` 且 64 B 对齐 | `BANDWIDTH_ALIGNED` | 多链路并行、分片流水、OwnCopy 异步。 |
| `rankBytes > 1 MiB` 且未按 64 B 对齐 | `BANDWIDTH_TAIL` | 保留尾块，防止 400 MB + 4 B 一类数据丢失。 |

CCU 单次通信上限按 `MAX_DATA_SIZE = 256 MiB` 处理。通用双片计划每批最多覆盖 512 MiB；4×1 CLOS 带宽 Kernel 最多使用 4 个 Chunk，每个 Chunk 不超过 256 MiB。

### 7.2 算法模式

`AlgResourceCtx::algorithmMode` 决定 `ExecOp` 的实际分派：

| Mode | Host 执行函数 | 适用场景 | 主要特征 |
| ---: | --- | --- | --- |
| 0 | 通用 `ExecOp` 尾部 | Generic | 每 Die 一个 Peer-Lane Pull Kernel。 |
| 1 | `Exec4x1CommonDiePush` | 4×1 公共 Die | 低时延单片，或带宽四 Chunk 直推。 |
| 2 | `Exec8p4DualSeed` | 8+4 兼容路径 | 双 Seed、跨 Server Direct、机内 Relay。 |
| 3 | `Exec4x1DualPlanePush` | 4×1 大消息双平面 | 2-Peer 与 1-Peer Kernel 并行。 |
| 4 | `Exec2x8V22` | 2×8 | 机内/机间分层与 Direct 混合流水。 |
| 5 | `Exec8p4LatencyDirect` | 8+4 小消息 | 每 Die 一个 6 参数 Direct-Push Kernel。 |
| 6 | `Exec8p4WidePipeline` | 8+4 大消息 | Seed、Direct、Original、Relay 四 Kernel 流水。 |
| 7 | `ExecOutput512Static` | 最终输出约 512 KiB | Rank 4/12/16 专用、每 Die 一个 6 参数短 Kernel。 |

Mode 7 的历史名称保留了 `Static`，但当前可提交修正版并未给每个 Die 注册 cold/warm/zero-argument 三套图，而是只注册经过验证的 `CcuLatencyDirectPushKernel`。运行时仍传入 6 个实时参数，以避免微码装载资源不足。

## 8. Host 侧执行编排

### 8.1 EngineCtx 隔离

工程为不同拓扑和消息档位使用不同 tag，避免低时延、带宽和最终输出 512 KiB 的 Channel/Kernel 资源互相污染。EngineCtx 中缓存的是通信域生命周期内稳定的内容：

- Channel Handle 与按 Die 的 Peer 映射；
- 额外 Worker 和 OwnCopy Thread；
- 已注册的 Kernel Handle；
- 拓扑类型与算法模式；
- 特殊会话需要的地址、Memory Token 和单 Rank字节数。

传入的 `stream` 每次调用都会重新绑定到主线程，不直接假定跨 Stream 可复用。

### 8.2 双 Die Worker 同步

双 Die 的基本并行顺序为：

```text
main  --Record--> worker
worker --Wait---> 启动条件成立
main  ----------> Launch Die 0 Kernel
worker ----------> Launch Die 1 Kernel
main  <---Wait---- worker 完成
worker --Record--> main
```

也就是每批通常包含 4 个 Host Thread Notify 操作。V22 混合流水和 8+4 WidePipeline 还会增加阶段 READY/DONE 依赖，防止 Relay 早于 Seed 数据到达。

### 8.3 OwnCopy

小消息可由负责机内组的 Kernel 直接执行 `LocalCopy`。大消息通常使用独立 OwnCopy Thread：

1. 主线程通知 OwnCopy Thread 启动；
2. OwnCopy 按不超过 256 MiB 的片段拷贝到本 Rank 输出槽；
3. OwnCopy Thread 记录完成；
4. 主线程在返回前等待完成。

若 `sendBuf` 已经指向本 Rank 输出槽，则整条 OwnCopy 路径跳过。

### 8.4 各模式的启动规模

| 模式 | 每批 Kernel 启动数 | 每 Kernel 动态参数 | 典型 Host Worker 通知 |
| --- | ---: | ---: | ---: |
| Generic 单 Die | 1 | `5 + Peer数` | 0 |
| Generic 双 Die | 2 | `5 + Peer数` | 4 |
| 4×1 公共 Die延迟 | 1 | 6 | 0 |
| 4×1 公共 Die带宽 | 1 | 5～9 | OwnCopy 通知 0 或 4 |
| 4×1 双平面 | 2 | 7 | 双 Die 4，另加 OwnCopy 0 或 4 |
| 2×8 小消息 | 2 | 6 | 4 |
| 2×8 分层流水 | 通常 4 | V22 Group 6；Phase 15 | 基础 6，OwnCopy 另计 |
| 8+4 延迟 | 2 | 6 | 4 |
| 8+4 WidePipeline | 4 | 7 / 12 / 9 | 基础 6，OwnCopy 另计 |
| 最终输出 512 KiB 单 Die | 1 | 6 | 0 |
| 最终输出 512 KiB 双 Die | 2 | 6 | 4 |

表中的数值指一次执行批次；超大消息可能包含多个批次。

## 9. 最终输出 512 KiB 快速路径

### 9.1 正确识别方式

这里的 512 KiB 是 AllGather 的最终输出总量，不是每个 Rank 的输入量。工程仅对 Rank 数 4、12、16 启用该分支，并使用误差范围兼容 12 Rank 下 FP32 元素数不能精确整除的情况：

```cpp
rankBytes = sendCount * sizeof(float);
outputBytes = rankBytes * rankSize;
delta = abs(outputBytes - 512ULL * 1024ULL);
isOutput512 = delta < rankSize * sizeof(float);
```

因此不会用 `outputBytes == 512 KiB` 的严格相等判断误判 12 Rank 用例。

### 9.2 首次调用

首次命中时，`RegisterOutput512StaticKernels` 会：

- 验证每个 ChannelGroup 均位于单一 IO Die，且每组不超过 8 条 Channel；
- 获取当前 send/recv 地址对应的 Memory Token；
- 为每个真实 IO Die 注册一个 `CcuLatencyDirectPushKernel`；
- 固化 Rank、Peer 映射和 OwnCopy 归属；
- 在 EngineCtx 中保存地址、Token、Rank 字节数、Thread 和 Kernel Handle；
- 通过 `ExecOutput512Static` 使用 6 个实时参数完成第一次执行。

6 个参数依次为输入地址、输出地址、输入 Token、输出 Token、本 Rank 输出偏移和传输字节数。

### 9.3 稳态 Host 快速入口

第一次成功执行后，工程建立 `thread_local Output512FastCache`。缓存签名包含：

- `comm` 与 `stream`；
- `sendBuf` 与 `recvBuf`；
- `sendCount` 与 `dataType`；
- Rank/组数量、Thread 和 Kernel Handle；
- 地址、Token、Rank 输出偏移和传输长度。

下一次调用若完整签名一致，会在公共参数校验、DFX、Rank 查询、tag 构造、Thread 获取、EngineCtx 查找和资源反序列化之前进入 `LaunchOutput512WarmFast`。单 Die 直接启动一个 Kernel；双 Die 只保留必要的主/Worker握手和两次 KernelLaunch。

若通信域相同但 buffer、stream、count 或类型变化，缓存会失效并回到可靠路径，避免复用旧地址或旧 Token。

### 9.4 两种“512 KiB”不可混淆

源码还保留一条 4×1 的“单 Rank 输入恰好 512 KiB”注册冷/热路径：

```text
rankBytes == 512 KiB
```

它与“最终输出约 512 KiB”的 Mode 7 不是同一个测试几何。前者可使用 `CcuRegistered4x1DirectPushKernel` 的 cold/warm 静态参数图；后者当前使用 loader-safe 的 6 参数动态短 Kernel。

## 10. CCU Kernel 数据流

### 10.1 公共通信步骤

Direct-Push Kernel 的典型流程为：

1. `LoadArg` 读取本次 buffer 地址、Token、偏移和长度；
2. 通过 `WriteVariableWithNotify` 发布接收端输出地址和 Token；
3. `NotifyWait` 等待对端地址资源可用；
4. 源 Rank 对每个 Peer 的目标 Rank 槽执行 `Write`；
5. 必要时执行本 Rank `LocalCopy`；
6. 使用 `EventWait` 等待数据任务完成；
7. 非低时延路径通过 `NotifyRecord/NotifyWait` 完成 PostSync。

低时延 Kernel 会把网络完成事件合并，并尽量省去额外 PostSync；带宽 Kernel 则保留更严格的阶段同步和多片流水。

### 10.2 主要 Kernel

| Kernel | 方向与用途 | 运行时参数 | 关键行为 |
| --- | --- | ---: | --- |
| `CcuPeerLanePullKernel` | 通用 Receiver-owned Pull | `5 + N` | 发布源地址，按最多两片 Read，最终完整 PostSync。 |
| `CcuRotatingDirectPushKernel` | 4×1/8+4 机内直推 | 7 | Peer 轮转发射，可融合 OwnCopy。 |
| `CcuWideRotatingDirectPushKernel` | 大消息机内直推 | 4 或 7 | 支持静态几何、渐进地址等待和双片合并。 |
| `CcuLatencyDirectPushKernel` | 512 KiB及8+4低时延 | 6 | 每Channel直推、组合EventWait、无额外PostSync。 |
| `CcuClos4x1DirectPushKernel` | 4×1 公共 CLOS Die | 5～9 | 1或4 Chunk；带宽模式保留完整同步。 |
| `CcuDualSeedCrossKernel` | 8+4 跨 Server | 11 | Small Server 双Seed，Large Server单Seed，并支持Direct阶段。 |
| `CcuWideDualSeedCrossKernel` | 8+4 宽流水跨Server | 12 | Seed与Direct拆图，使用Seed-ready依赖。 |
| `CcuDualSeedRelayKernel` | 8+4 机内重组 | 9 | 从已到达的recv块向同Server Peer Relay。 |
| `CcuWideDualSeedRelayKernel` | 8+4 宽流水重组 | 9 | 第二源采用轮转首发，降低热点。 |
| `CcuV22AllGatherGroupKernel` | 2×8 分组直推 | 6 | 机内/跨Server组的Original传输。 |
| `CcuV22AllGatherPhaseKernel` | 2×8 Seed/Relay阶段 | 15 | 使用预计算的远端Rank偏移执行批量计划。 |

源码中还实现了 `CcuRegistered4x1DirectPushKernel` 和 `CcuOutput512StaticDirectPushKernel` 两类零运行参数实验图。当前 Mode 7 为保证微码装载稳定性并不注册后者；不要仅因函数存在就认为它处于实际调用路径。

### 10.3 Notify 与 Event 约束

CCU 使用不同 Notify 位区分地址、Token、PostSync 和 Seed-ready。事件掩码为 16 位，因此：

- 每 Die Channel 数必须不超过 8，双片最多使用 `2 × 8 = 16` 个事件位；
- 4×1 四 Chunk 共 12 个网络事件，仍在范围内；
- Relay 和 V22 批计划必须限制事件数量；
- Seed、Direct、Relay 之间的 Host DAG 依赖不可删除，否则可能提前读取未完成的 `recvBuf` 数据，触发 Memory Conflict 或结果错误。

## 11. 自定义资源结构

### 11.1 Kernel 静态参数

所有 Kernel 参数继承 `CcuKernelArgBase`：

```cpp
struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};
```

派生结构在注册阶段固化 Rank、Peer 映射、Channel 顺序、Chunk 数、Seed Rank、Relay 条带和同步开关。`CcuKernelInfo` 使用 `shared_ptr<CcuKernelArgBase>` 保证注册期间参数对象仍然有效。

### 11.2 `AlgResourceCtx`

资源上下文包含：

- `ccuThread`、`ownCopyThread` 与按 Die 的 `threads`；
- `ccuKernels`；
- `peerRanksByGroup`；
- `topologyType` 与 `algorithmMode`；
- 本地通信 buffer；
- 4×1 注册会话和最终输出 512 KiB 会话的地址、Token 与长度。

`Serialize()` 与 `DeSerialize()` 必须以完全相同的顺序处理所有字段。新增字段时应同时修改两端，并优先在末尾追加；改变已有字段顺序会导致 EngineCtx 反序列化错位。

## 12. 正确性与维护约束

修改算法时必须同时满足以下要求：

- 同一个 Kernel 的所有 Channel 必须来自同一个 IO Die。
- 不得通过在同 Kernel 中增加 Worker 绕过 IO Die 限制。
- 每个端点对仅使用一条 Channel，Channel 的 Notify 数必须覆盖该 Kernel 的实际协议。
- 并发 Kernel 写入 `recvBuf` 时，目标区间必须互不冲突；Relay 读取前必须显式依赖 Seed 完成。
- 本 Rank OwnCopy 只能写入自己的 Rank 槽，且返回前必须完成。
- 400 MB + 4 B 等奇尾数据必须保留全部字节，不能简单向下对齐后丢弃尾块。
- Buffer、Stream、通信域或 Token 发生变化时，静态会话缓存必须失效。
- 新增 `algorithmMode` 时，必须在 `ExecOp` 中增加对应处理；未知模式会直接报错。
- 公开入口必须保持 `extern "C"` 的 `HcclAllGather` ABI；`ops_hccl::ExecOp` 的声明和定义也必须一致。
- 只使用当前 CANN 9.1.0 头文件公开且工程中真实存在的 CCU/HCCL 接口。

## 13. 调试与常见问题

### 13.1 编译阶段

工程启用了 `-Werror`，任何警告都会导致编译失败。常见问题包括：

- 新增静态函数但未调用，触发 `unused-function`；
- Kernel 参数数量超过 `MAX_CCU_TASK_ARGS`；
- Host 声明、Kernel 声明和实现的函数签名不一致；
- 使用本版本 CANN 头文件未公开的接口；
- `AlgResourceCtx` 新字段只修改了序列化或只修改了反序列化。

### 13.2 运行阶段

| 现象 | 优先检查 |
| --- | --- |
| Kernel 注册返回错误 | 同一次注册中的 Kernel 数量、IO Die 归属、Kernel 名唯一性和静态Arg有效性。 |
| `dies of channels are not same` | Host 是否把不同 local Die 的 Channel 放入同一 Kernel。 |
| Channel Acquire 失败 | RankGraph层、端点描述、协议、每端点Channel数量和Notify数量。 |
| Checker Memory Conflict | 两个Kernel写区间是否重叠，Seed/Relay是否缺少Record/Wait依赖。 |
| AllGather 输出提前结束 | 是否漏掉某个Peer、OwnCopy、尾块或必要的完成等待。 |
| 稳态偶发错误 | FastCache签名是否覆盖comm/stream/buffer/count/type/token，缓存是否及时失效。 |
| 512 KiB分支未命中 | 是否错误地把单Rank输入量当作最终输出量，或使用了严格相等判断。 |

Debug 构建可配合 `HCCL_INFO`、`HCCL_WARNING` 和 `HCCL_ERROR` 定位资源构造、Kernel 注册及执行阶段的失败点。

## 14. 提交前检查清单

- [ ] 只修改赛事允许的 6 个文件以及说明文档。
- [ ] `bash build.sh` 无警告、无错误。
- [ ] `nm -D build/lib64/libhccl.so` 可找到公开的 `HcclAllGather` 符号。
- [ ] 功能测试覆盖 2×8、4×1、8+4 三种拓扑和全部消息档位。
- [ ] 最终输出 512 KiB 用例按 `outputBytes` 识别，12 Rank 使用误差范围。
- [ ] 每个 Kernel 的 Channel 均属于同一 IO Die。
- [ ] 双 Die 和 Seed/Relay 的 Record/Wait 依赖完整。
- [ ] OwnCopy 与奇尾数据均未遗漏。
- [ ] EngineCtx 与 FastCache 在 buffer、stream 或通信域变化时可安全回退。
- [ ] 提交压缩包根目录直接包含 `README.md`、`CMakeLists.txt`、`build.sh`、`include/`、`op_host/` 和 `op_kernel_ccu/`，没有多包一层目录。

## 15. 许可证

本工程遵循根目录 [LICENSE](LICENSE) 中的 CANN Open Software License Agreement Version 2.0。使用、修改和提交前请阅读并遵守该许可证及赛事规则。

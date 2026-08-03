# AllGather 集合通信算子

## 1. 项目介绍

本工程实现基于 CANN 9.1.0 HCOMM/AICPU 通信引擎的 `HcclAllGather` 接口。每个 Rank 提供一段 FP32 输入，算子将通信域内所有 Rank 的输入按 Rank ID 顺序拼接到 `recvBuf`，并使每个 Rank 获得相同的完整结果。

V26 面向初赛的 16 Rank、2×8 卡拓扑进行重点优化。Host 侧负责解析 RankGraph、固化 Thread/Channel/HCCL Buffer 资源并启动 AICPU Kernel；AICPU 侧把通信算法编排为 HCOMM Thread 上的 LocalCopy、Read、Write 和 Notify 任务。小消息采用 4 轮 Recursive Doubling，大消息采用 15 路 Direct-Read、双 Bank、独立 OwnCopy 与两棵控制树组成的三路解耦流水。

```text
├── CMakeLists.txt                  # 顶层 CMake 配置及 AICPU 交叉编译入口
├── build.sh                        # 构建脚本
├── .clang-format                   # 代码风格配置
├── .gitattributes                  # Git 文本属性配置
├── .gitignore                      # Git 忽略规则
├── LICENSE                         # 工程许可证文件
├── README.md                       # 项目说明
├── include/                        # 头文件目录
│   ├── hccl.h                      # 集合通信算子公开头文件
│   ├── common.h                    # 通用常量和 OpParam 定义
│   ├── custom.h                    # ★ 选手编写：自定义资源结构及序列化
│   ├── log.h                       # 日志与返回值检查宏
│   └── binary_stream.h             # 序列化类定义
├── op_host/                        # Host 侧代码目录
│   ├── CMakeLists.txt              # Host 动态库构建配置
│   ├── allgather.cc                # ★ 选手编写：Host 侧资源申请逻辑
│   ├── launch_aicpu_kernel.h       # AICPU Kernel 下发接口声明
│   └── launch_aicpu_kernel.cc      # AICPU Kernel 加载与下发逻辑
└── op_kernel_aicpu/                # Device/AICPU 侧代码目录
    ├── CMakeLists.txt              # AICPU 动态库构建配置
    ├── aicpu_kernel.cc             # AICPU Kernel 函数入口
    ├── exec_op.h                   # 通信算法入口声明
    └── exec_op.cc                  # ★ 选手编写：通信算法编排逻辑
```

> [!NOTE] 注意：
> 算子工程中已提前预制好固有逻辑，选手仅允许修改 `custom.h`、`allgather.cc`、`exec_op.cc` 共 3 个文件内容。

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

`build.sh` 通过 `ASCEND_HOME_PATH` 取得 CANN 根目录。运行时还会从以下位置加载 AICPU Kernel 描述：

```text
${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/config/aicpu_kernel.json
```

该 JSON 由安装/部署环境提供，不在本源码压缩包中。

### 2.3 编译算子工程

```bash
bash build.sh

# 编译 Debug 版本，便于断点调试
bash build.sh --debug
```

顶层 CMake 构建两个共享库：

| 产物 | 代码位置 | 主要依赖 | 用途 |
| --- | --- | --- | --- |
| `libhccl.so` | `op_host/` | `hcomm`、`acl_rt` | 暴露 `HcclAllGather`，管理资源并下发 AICPU Kernel。 |
| `libhccl_device.so` | `op_kernel_aicpu/` | Device 侧 `ccl_kernel` | 实现 `HcclAICPUKernel` 和通信任务编排。 |

AICPU 目标由 `ExternalProject` 使用以下 HCC 交叉编译工具链构建：

```text
${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc
```

默认安装前缀为工程内的 `build/`，主要产物如下：

```text
build/
├── include/hccl.h
└── lib64/
    ├── libhccl.so
    └── libhccl_device.so
```

## 3. 代码格式

选手代码需符合 [.clang-format](.clang-format) 文件中的代码风格规范，可通过下列命令一键修改：

```bash
bash build.sh --format
```

格式化入口同样会先检查 `ASCEND_HOME_PATH`，因此应先加载 CANN 环境变量。

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

| 参数 | 方向 | 说明 |
| --- | --- | --- |
| `sendBuf` | 输入 | 本 Rank 的源数据地址。 |
| `recvBuf` | 输出 | AllGather 完整结果地址。 |
| `sendCount` | 输入 | 本 Rank 输入的元素个数，不是字节数。 |
| `dataType` | 输入 | 当前实现仅支持 `HCCL_DATA_TYPE_FP32`。 |
| `comm` | 输入 | HCCL 通信域。 |
| `stream` | 输入 | 当前 Rank 使用的 ACL Runtime Stream。 |

返回 `HCCL_SUCCESS` 表示通信任务已经按 Stream/Thread 依赖完成编排和下发。应用侧读取 `recvBuf` 时仍应遵守 ACL Stream 的完成语义，不能把函数返回简单等同于主机侧同步完成。

### 4.2 数据量与结果布局

本文使用以下符号：

```cpp
S = rankBytes   = sendCount * sizeof(float);
O = outputBytes = S * rankSize;
```

最终结果按 Rank ID 排列：

```text
recvBuf + 0 * S  <- rank 0 输入
recvBuf + 1 * S  <- rank 1 输入
...
recvBuf + r * S  <- rank r 输入
```

本 Rank 的 OwnCopy 目标为 `recvBuf + myRank * S`。当 `sendBuf` 已经等于该地址时，本地拷贝会被跳过。

> [!IMPORTANT]
> 源码中的 512 KiB 阈值针对单 Rank 输入量 `S`，不是最终 AllGather 输出量 `O`。如果题目以最终输出量描述测试点，必须先除以 Rank 数换算到单 Rank 输入量。

### 4.3 输入约束

- `sendBuf`、`recvBuf`、`comm` 和 `stream` 必须非空。
- 当前只支持 FP32；其他类型返回 `HCCL_E_NOT_SUPPORT`。
- `rankSize` 必须处于 `[1, 16]`，且 `myRank < rankSize`。
- Host 和 AICPU 两侧都会检查 `S` 与 `O` 的 64 位乘法溢出。
- 通信域内所有 Rank 必须使用一致的 `sendCount` 和 `dataType`。
- 每个 Peer 只申请一条物理 Channel，符合赛题端点对约束。

## 5. 工程分层与完整调用链

工程将一次 AllGather 分为 Host 动态库和 AICPU 动态库两个执行层：

```text
应用 / 测试框架
  │
  └─ HcclAllGather                         Host: libhccl.so
       ├─ 参数、DFX、Rank 查询
       ├─ CPU_TS Stream Thread 获取与跨引擎导出
       ├─ EngineCtx 查询
       │    ├─ 命中：复用资源
       │    └─ 未命中：BuildAlgorithmResources
       │         ├─ RankGraph 物理链路查询
       │         ├─ AICPU Thread / Channel 申请
       │         ├─ 本地与远端 HCCL Buffer 查询
       │         └─ AlgResourceCtx 序列化并写入 EngineCtx
       └─ LaunchAICPUKernel
            ├─ Host → AICPU Notify
            ├─ aclrtLaunchKernelWithConfig
            └─ 等待 AICPU 回执
                 │
                 └─ HcclAICPUKernel        Device: libhccl_device.so
                      ├─ AlgResourceCtx 反序列化
                      ├─ HcommBatchModeStart
                      ├─ 等待 Host Notify
                      ├─ ExecOp
                      │    ├─ Recursive Doubling
                      │    ├─ Large Triple-Decoupled
                      │    ├─ Single Rank
                      │    └─ Direct FullMesh 回退
                      ├─ AICPU → Host Notify
                      └─ HcommBatchModeEnd
```

真正的数据搬运由 CANN/HCOMM Runtime 执行；本工程的主要职责是构造资源、生成任务图和建立正确的跨 Thread/Channel 依赖。

## 6. Host 侧资源模型

### 6.1 物理链路与 Channel

`FindDirectLink` 对每个远端 Rank 遍历 RankGraph 的网络层，并使用 `HcclRankGraphGetLinks` 查询本 Rank 到该 Peer 的链路。代码选择找到的第一条物理直连，依赖赛题“任意两个 NPU 之间只有一条物理链路”的拓扑约束。

`BuildAlgorithmResources` 按远端 Rank 升序跳过本 Rank，为每个 Peer 构造一个 `HcclChannelDesc`：

- `remoteRank`：远端 Rank ID；
- `channelProtocol`：RankGraph 返回的物理链路协议；
- `localEndpoint` / `remoteEndpoint`：实际端点描述；
- `notifyNum = 2`：READY 和 CONSUMED 两个 Channel Notify。

所有 Channel 通过一次 `HcclChannelAcquire` 批量申请。随后使用 `HcclChannelGetHcclBuffer` 取得每个 Peer 的远端 HCCL Buffer 地址和大小。

### 6.2 共同分块容量

设本地 HCCL Buffer 大小为 `C_local`，各远端 Buffer 大小为 `C_peer[i]`，则：

```text
C = chunkCapacity = min(C_local, C_peer[0], ..., C_peer[peerCount-1])
```

使用所有端点容量的最小值，可以避免任一远端 Buffer 较小时发生越界。工程没有额外执行 `aclMalloc`；HCCL CCL Buffer 就是算法的中转 workspace。

### 6.3 Thread 数量与角色

| 场景 | Thread 数量 | 组织方式 |
| --- | ---: | --- |
| 单 Rank | 1 | `threads[0]` 为 Main。 |
| 非16 Rank多卡 | `peerCount + 1` | Main兼任一个Peer；其余Worker通信；末尾Thread负责OwnCopy。 |
| 16 Rank | 17 | Main + 15个固定Peer Worker + 独立OwnCopy。 |

16 Rank 大消息路径中的固定映射为：

```text
threads[0]      -> Main：控制、Bank准备、根完成等待
threads[1..15]  -> Worker 0..14：一对一绑定 channels[0..14]
threads[16]     -> OwnCopy：sendBuf -> recvBuf[myRank]
```

每个 AICPU Thread 申请 3 个 Notify；每条 Channel 申请 2 个 Notify。具体索引见第 11 节。

## 7. EngineCtx 与首次/热路径

工程使用固定 tag：

```text
hccl_custom_allgather_v26_large_triple_decoupled
```

该 tag 是当前通信域内的资源缓存键。Host 分别使用 AICPU_TS 与 CPU_TS 两个 EngineCtx：

- AICPU_TS EngineCtx 保存完整序列化后的 `AlgResourceCtx`；
- CPU_TS EngineCtx 只保存 AICPU Main Thread 的 `ThreadHandle`，供后续调用重新导出到 CPU_TS。

### 7.1 首次调用

首次未找到 AICPU_TS EngineCtx 时，会执行：

1. 获取本地 HCCL Buffer；
2. 申请全部 AICPU Thread；
3. 遍历 RankGraph 并批量申请 Channel；
4. 查询全部远端 HCCL Buffer；
5. 计算共同 `chunkCapacity`；
6. 将 `AlgResourceCtx` 序列化后写入 AICPU_TS EngineCtx；
7. 将 AICPU Main Thread Handle 写入 CPU_TS EngineCtx；
8. 完成跨引擎 Thread 导出并下发 Kernel。

### 7.2 热路径

EngineCtx 命中后，不再重复申请 Thread/Channel，也不再遍历 RankGraph。Host 从两个 EngineCtx 中取回资源，并完成本次 Stream Thread 的跨引擎导出。

以下操作仍然存在于每次调用：

- 参数校验、tag 写入、DFX 注册及 Rank 查询；
- `HcclThreadAcquireWithStream` 与跨引擎导出；
- EngineCtx 查询；
- AICPU Kernel 参数句柄构造与一次 KernelLaunch；
- AICPU 侧把 `resCtx` 复制到 `std::vector<char>` 并反序列化；
- Host/AICPU 启动与回执 Notify。

`HcclEngineCtxGet` 的任何非成功结果都会进入首次资源构建分支，这是当前代码的实际行为，并不只判断 `NOT_FOUND`。

## 8. Host 到 AICPU 的桥接

### 8.1 二进制加载

`launch_aicpu_kernel.cc` 使用：

```cpp
thread_local aclrtBinHandle g_binKernelHandle;
```

同一 Host 线程第一次调用时，从 `aicpu_kernel.json` 加载 AICPU 二进制；后续调用跳过重复 Binary Load。该缓存只覆盖二进制句柄，函数句柄查询和 Kernel 参数句柄构造仍会在每次下发时执行。

### 8.2 每次下发顺序

```text
CPU_TS Stream Thread
  ├─ Record AICPU Main notify[0]
  ├─ aclrtBinaryGetFunction("HcclAICPUKernel")
  ├─ aclrtKernelArgsInit
  ├─ Append 整个 OpParam
  ├─ aclrtKernelArgsFinalize
  ├─ aclrtLaunchKernelWithConfig：1 Block
  └─ Wait Host notify[0]
```

Launch timeout 属性为 `27 * 68 = 1836`，HCOMM Notify 等待使用 `CUSTOM_TIMEOUT = 1800`。

### 8.3 AICPU 入口

`HcclAICPUKernel` 的顺序为：

1. 从 `param->resCtx` 反序列化 `AlgResourceCtx`；
2. `HcommBatchModeStart(param->tag)`；
3. AICPU Main 等待 Host 启动 Notify；
4. 调用 `ExecOp` 编排通信任务；
5. 向 CPU_TS Stream Thread 记录完成回执；
6. `HcommBatchModeEnd(param->tag)`。

## 9. ExecOp 算法分流

`ExecOp` 首先重新校验指针、FP32、Rank、Channel 数、Thread 数、`chunkCapacity` 及数据量溢出，然后按下表选择路径：

| 路径 | 触发条件 | 数据动作 | Thread 组织 |
| --- | --- | --- | --- |
| 空数据 | `S == 0` | 无搬运，直接成功。 | 无额外任务。 |
| Recursive Doubling | `rankSize == 16`、`S <= 512 KiB`、`O <= C` | 4轮向XOR伙伴写逐步扩大的聚合块。 | 仅Main。 |
| 三路解耦双 Bank | `rankSize == 16`、`S > 512 KiB`、`C >= 2` | 每个Epoch由15个Worker直接读取远端当前Bank。 | Main + 15 Worker + OwnCopy。 |
| 单 Rank | `peerCount == 0` | 必要时执行一次LocalCopy。 | Main。 |
| Direct FullMesh 回退 | 以上条件均未命中 | 按完整容量C分块，每条Channel直接Read。 | Main + Worker + OwnCopy。 |

源码虽然定义了：

```cpp
P6_MESSAGE_BYTES = 512 MiB;
```

但该常量未参与任何 `if` 判断。因此 V26 大消息路径不是“仅限 512 MiB 测试点”，而是覆盖所有满足条件的 16 Rank、单 Rank 输入量 `S > 512 KiB` 的调用。

## 10. 小消息：4轮 Recursive Doubling

### 10.1 进入条件

Recursive Doubling 只在以下条件同时满足时启用：

```text
rankSize == 16
S <= 512 KiB
16 * S <= C
```

完整输出必须能放入共同 HCCL Buffer。若 Buffer 容量不足，即使 `S` 很小，也会回退到通用 Direct FullMesh。

### 10.2 算法步骤

1. Main 把本 Rank 输入复制到本地 CCL Buffer 的 `myRank` 槽；
2. 依次取 `groupSize = 1, 2, 4, 8`；
3. 本轮伙伴为 `partnerRank = myRank XOR groupSize`；
4. 将已经聚合好的连续 Rank 块写入伙伴的 CCL Buffer；
5. 通过 Channel Notify 与伙伴对称同步；
6. 四轮完成后，将整个 CCL Buffer 聚合结果复制到 `recvBuf`。

| 轮次 | 伙伴 | 本轮发送量 | 完成后覆盖Rank数 |
| ---: | --- | ---: | ---: |
| 1 | `rank XOR 1` | `1 × S` | 2 |
| 2 | `rank XOR 2` | `2 × S` | 4 |
| 3 | `rank XOR 4` | `4 × S` | 8 |
| 4 | `rank XOR 8` | `8 × S` | 16 |

每 Rank 的网络总字节量仍为：

```text
S + 2S + 4S + 8S = 15S
```

优化点在于通信启动次数从面向15个Peer的15次降为4轮，而不是减少理论通信字节数。

### 10.3 Write与Notify融合

小消息使用 `HcommWriteWithNotifyOnThread` 融合远端 Write 和 Notify Record，本端仅保留一次 Notify Wait。前三轮使用 Channel Notify 0，跨 Server 的 `groupSize = 8` 轮使用 Notify 1，避免连续阶段持续复用同一索引。

## 11. 大消息：三路解耦双 Bank Direct-Read

### 11.1 核心思想

16 Rank 且 `S > 512 KiB` 时，Main 不负责任何 Peer Channel。15 个 Worker 分别绑定一条固定 Channel，从对应远端 Rank 的当前 Bank 直接读取到 `recvBuf` 的最终 Rank 槽；独立 OwnCopy Thread 同时把本 Rank 输入复制到自己的输出槽。

三条并行工作流为：

| 角色 | 任务 | 可重叠对象 |
| --- | --- | --- |
| Main | 准备首Bank、启动控制树、准备下一Bank、等待两棵子树 | 当前Epoch的15路网络Read、OwnCopy |
| 15个Peer Worker | START下传、READY握手、远端Read、CONSUMED、完成汇聚 | Main准备下一Bank、OwnCopy |
| OwnCopy | 一次性执行`sendBuf -> recvBuf[myRank]` | 全部网络Epoch与Bank准备 |

### 11.2 双 Bank 内存模型

设共同容量为 `C`：

```text
B = floor(C / 2)              # 单Bank容量
R = ceil(S / B)               # Epoch数量
每Rank远端Read次数 = 15 * R
```

地址布局为：

```text
localBuffer + currentBank * B
    本Rank当前对外发布的数据块

remoteCclMem[channel] + currentBank * B
    Worker读取的远端同号Bank

recvBuf + remoteRank * S + currentOffset
    远端数据的最终目标位置

recvBuf + myRank * S
    独立OwnCopy写入的本Rank槽位
```

首 Bank 必须先完成 LocalCopy，随后才能启动网络。当前 Bank 被 15 路 Worker 读取时，Main 把下一块数据复制到另一个 Bank；当前 Epoch 的两棵完成子树都到达后，才切换 Bank。

### 11.3 READY / Read / CONSUMED

每条 Channel 的一个 Epoch 使用对称握手：

```text
Record READY
Wait   peer READY
Read   remote current bank -> recvBuf final slot
Record CONSUMED          # 非最后一块
Wait   peer CONSUMED     # 非最后一块
```

READY 保证对端 Bank 已准备完成；CONSUMED 保证双方均已读完，该 Bank 才能在后续 Epoch 被覆盖。最后一块之后不会再复用 HCCL Buffer，因此 V26 对称省略最后一块的 CONSUMED Record/Wait。

### 11.4 两棵控制树

二叉结构只用于 START 广播和完成汇聚，不传输用户数据。Main 每个 Epoch 只直接启动两个根 Worker，并只等待两个根完成：

```text
Main
├── W0
│   ├── W2
│   │   ├── W6
│   │   │   └── W14
│   │   └── W7
│   └── W3
│       ├── W8
│       └── W9
└── W1
    ├── W4
    │   ├── W10
    │   └── W11
    └── W5
        ├── W12
        └── W13
```

每个 Worker 收到 START 后先通知自己的左右子节点，再执行固定 Channel 的 Read；完成自身 Read并等到子树完成后，才向父节点上报。AICPU 源码中的 `for` 循环按顺序向不同 `ThreadHandle` 提交任务，但各 HCOMM Thread 保持自己的任务队列，所以网络 Read 可以并发执行。

### 11.5 单个 Epoch 的关键顺序

```text
Main -> W0/W1: START
Worker逐级向下广播START
15个Worker执行READY / Direct-Read / 可选CONSUMED
Main并行准备下一Bank
OwnCopy Thread并行写本Rank输出
Worker完成信号逐级向上汇聚
Main等待TREE_LEFT和TREE_RIGHT
切换currentOffset、currentBytes和currentBank
```

大消息数据面仍然是 15 路一对一 Direct-Read。控制树减少的是 Main 的集中扇出/扇入开销，不会把数据面改成树形转发，也不会减少网络总字节量。

## 12. Direct FullMesh 回退路径

未进入两个 16 Rank 专用路径时，算法按 `chunkCapacity = C` 分块：

1. Main 将当前源分块写入本地 HCCL Buffer；
2. 必要时启动独立 LocalCopy Thread，一次性写本 Rank 输出；
3. START 通过二叉树向 Worker 广播；
4. Main 负责 `channels[0]`，其余 Worker 各处理一条 Channel；
5. 每条 Channel 执行完整 READY → Read → CONSUMED；
6. 完成信号沿相同的树回到 Main；
7. 进入下一分块，最后等待 OwnCopy 完成。

与 16 Rank 大消息专用路径相比，回退路径只有单 Bank、Main仍承担一条Peer通信，并且最后一块仍执行 CONSUMED。它用于非16 Rank、Buffer边缘条件及未命中专用分支的情况。

单 Rank 时不申请 Channel；若输入和本 Rank 输出槽地址不同，只在 Main Thread 上执行一次 LocalCopy。

## 13. Notify 与同步索引

### 13.1 Thread Notify

每个 AICPU Thread 有 3 个 Notify：

| 索引 | 名称 | 用途 |
| ---: | --- | --- |
| 0 | `THREAD_NOTIFY_START` | 控制树启动。 |
| 0 | `THREAD_NOTIFY_LOCAL_COPY_DONE` | AICPU Main消费完Host启动信号后，同一槽位复用为OwnCopy完成。 |
| 1 | `THREAD_NOTIFY_TREE_LEFT` | 左子树完成。 |
| 2 | `THREAD_NOTIFY_TREE_RIGHT` | 右子树完成。 |

索引 0 的复用成立，是因为 AICPU Main 在进入 `ExecOp` 前已经等待并消费了 Host 启动信号。

### 13.2 Channel Notify

每条 Channel 有 2 个独立 Notify：

| 索引 | 名称 | 用途 |
| ---: | --- | --- |
| 0 | `CHANNEL_NOTIFY_READY` | 当前 HCCL Buffer/Bank 数据已准备好。 |
| 1 | `CHANNEL_NOTIFY_CONSUMED` | 当前数据已被对端读取，可安全复用Buffer。 |

不同 Channel 拥有各自的 Notify 空间，不能把某条 Channel 的 READY/CONSUMED 与另一 Peer 混用。

> [!WARNING]
> `THREAD_NOTIFY_NUM = 3`、`CHANNEL_NOTIFY_NUM = 2` 与 `exec_op.cc` 的索引约定是整体协议的一部分。减少 Notify 数量、交换索引或删除树形依赖会导致死锁、过早覆盖 Bank 或读取未完成数据。

## 14. Host/AICPU 共享数据结构

### 14.1 `OpParam`

`OpParam` 是每次调用的动态参数快照，并被 Host 作为一个整体追加到 AICPU Kernel 参数中。主要字段包括：

- tag、输入/输出地址、count；
- myRank、rankSize、dataType、opType；
- CPU_TS 与 AICPU_TS 的跨引擎 Thread Handle；
- 序列化资源上下文指针 `resCtx` 与大小 `ctxSize`。

固定长度为：通信域标识128字节、操作名32字节、tag总长160字节。

### 14.2 `ChannelInfo`

```cpp
struct ChannelInfo {
    uint32_t remoteRank;
    uint32_t notifyNum;
    ChannelHandle handle;
    CommBuffer remoteCclMem;
};
```

它把远端 Rank、固定 Channel、Channel Notify 数量及远端 HCCL Buffer 绑定在一起。

### 14.3 `AlgResourceCtx`

```cpp
struct AlgResourceCtx {
    ThreadHandle aicpuThread;
    CommBuffer localBuffer;
    uint64_t chunkCapacity;
    std::vector<ThreadHandle> threads;
    std::vector<ChannelInfo> channels;
};
```

序列化与反序列化的字段顺序必须严格一致：

```text
aicpuThread -> localBuffer -> chunkCapacity -> threads -> channels
```

修改 `custom.h` 时，Host 与 AICPU 使用的结构体布局、字段顺序和 ABI 必须同步。沿用同一 EngineCtx tag 却改变序列化格式，可能导致热路径读取旧上下文并发生错位。

## 15. 性能设计要点

- **资源常驻**：Thread、Channel、远端Buffer描述和共同容量只在首次 EngineCtx 构建时申请。
- **四轮小消息**：Recursive Doubling 将16 Rank通信启动次数压缩到4轮。
- **Write+Notify融合**：小消息减少单独的远端Notify Record任务。
- **固定Peer Worker**：大消息15条Channel一对一映射，Main不承担网络数据面。
- **双 Bank**：当前Bank网络读取时，Main准备下一Bank。
- **独立OwnCopy**：本Rank输出复制不占用Main的Bank准备关键路径。
- **双控制树**：Main每轮只直接启动2个根并等待2个根，降低集中控制扇出/扇入。
- **最终块省同步**：最后一个Epoch不再覆盖HCCL Buffer，因此省略CONSUMED确认。
- **直接最终落位**：Worker把远端块直接读到`recvBuf + remoteRank * S + offset`，无需后续重排。

### 15.1 关键任务数量

下表用于估算任务图规模；“Notify原语”分别统计Record和Wait，而不是把一对Record/Wait合并计为一次。

| 路径 | 数据任务 | Notify任务 | LocalCopy |
| --- | --- | --- | --- |
| Recursive Doubling | 每Rank 4次`WriteWithNotify`，总网络量`15S` | 4次Channel Wait | 2次：输入到CCL、完整CCL到输出 |
| 三路解耦，非末Epoch | 15次远端Read | 60个Channel Notify原语；60个Thread树Notify原语 | Main每Epoch准备1个Bank |
| 三路解耦，末Epoch | 15次远端Read | 30个READY原语；60个Thread树Notify原语 | Main准备最后Bank |
| Direct FullMesh，Peer数P | 每块P次远端Read | 每块`4P`个Channel Notify原语；`4(P-1)`个Thread树Notify原语 | Main每块准备Buffer |

若需要OwnCopy，三路解耦与回退路径还会增加1次整块LocalCopy，以及“Main Record START、OwnCopy Wait START、OwnCopy Record DONE、Main Wait DONE”共4个Thread Notify原语。

16 Rank首次资源规模为17个AICPU Thread、15条Channel、`17 × 3 = 51`个Thread Notify槽位和`15 × 2 = 30`个Channel Notify槽位。这些是资源槽位数量，不等于一次调用中必然执行的Notify任务数量。

## 16. 正确性与维护约束

修改算法时必须保持以下不变量：

- Channel 与 `remoteRank` 的映射必须一致；当前Host按远端Rank升序保存。
- `chunkCapacity` 必须取本地和全部远端HCCL Buffer的最小值。
- 16 Rank大消息必须保持17个Thread及15个固定Worker映射。
- 覆盖某个Bank前，必须确认所有Peer已经消费该Bank；只有最终块可以安全省略CONSUMED。
- OwnCopy必须只写本Rank输出槽，并在返回依赖中等待完成。
- START广播、左右子树完成和Host/AICPU回执必须使用正确的Thread Notify索引。
- 两次相邻调用依赖固定tag复用资源；改变`AlgResourceCtx` ABI时必须处理旧EngineCtx失效问题。
- AICPU函数导出名必须保持为`HcclAICPUKernel`，并与`aicpu_kernel.json`一致。
- 公开Host接口必须保持`extern "C"`的`HcclAllGather` ABI。
- 只能使用赛事允许的HCOMM/AICPU通信原语和可修改文件。

## 17. 调试与常见问题

| 现象 | 优先检查 |
| --- | --- |
| AICPU二进制加载失败 | `ASCEND_HOME_PATH`、`aicpu_kernel.json`路径、Device库安装和函数名。 |
| EngineCtx反序列化异常 | `custom.h`字段布局、序列化顺序、Host/AICPU ABI和旧tag缓存。 |
| Channel申请失败 | RankGraph物理链路、endpoint描述、协议、每Peer仅一条Channel。 |
| `chunkCapacity`为0 | 本地或某个远端HCCL Buffer是否为空。 |
| 通信死锁 | Thread/Channel Notify数、索引、Record/Wait是否对称，树父子映射是否正确。 |
| 大消息结果被覆盖 | 下一次复用Bank前是否完成CONSUMED；`currentBank`和偏移是否同步切换。 |
| 本Rank结果缺失 | OwnCopy目标地址、启动Notify及最终DONE等待。 |
| 小消息没有进入Recursive Doubling | 检查`rankSize==16`、`S<=512KiB`以及`O<=C`。 |
| 512MiB常量修改后性能不变 | `P6_MESSAGE_BYTES`当前未被算法判断引用。 |
| `-Werror`导致编译失败 | 未使用函数/常量、类型转换、Host/Device声明不一致。 |

工程启用 `-Wall -Werror`。任何警告都会导致构建失败，新增实验代码前应确保所有函数和常量均被实际使用。

## 18. 源码阅读顺序

建议按以下顺序阅读：

1. `include/hccl.h`：了解公开API；
2. `op_host/allgather.cc`：理解资源构建和EngineCtx缓存；
3. `include/common.h`、`include/custom.h`：理解动态参数和资源快照；
4. `op_host/launch_aicpu_kernel.cc`：理解Host到AICPU的启动；
5. `op_kernel_aicpu/aicpu_kernel.cc`：理解Device入口和BatchMode；
6. `op_kernel_aicpu/exec_op.cc`：先看`ExecOp`分流，再看三条算法路径；
7. 顶层及两个子目录的`CMakeLists.txt`：确认双动态库构建和安装方式。

核心函数速查：

| 函数 | 文件 | 作用 |
| --- | --- | --- |
| `HcclAllGather` | `op_host/allgather.cc` | 公开入口、参数校验、EngineCtx和AICPU下发。 |
| `FindDirectLink` | `op_host/allgather.cc` | 从RankGraph查找本Rank到Peer的物理链路。 |
| `BuildAlgorithmResources` | `op_host/allgather.cc` | 申请Thread、Channel和HCCL Buffer资源。 |
| `LaunchAICPUKernel` | `op_host/launch_aicpu_kernel.cc` | 加载、封参、启动AICPU并完成Host/Device握手。 |
| `HcclAICPUKernel` | `op_kernel_aicpu/aicpu_kernel.cc` | AICPU入口、反序列化、BatchMode和ExecOp调用。 |
| `ExecOp` | `op_kernel_aicpu/exec_op.cc` | 校验资源并选择算法路径。 |
| `ExecuteRecursiveDoubling` | `op_kernel_aicpu/exec_op.cc` | 16 Rank小消息四轮聚合。 |
| `ExecuteLargeTripleDecoupled` | `op_kernel_aicpu/exec_op.cc` | 16 Rank大消息三路解耦双Bank。 |
| `QueuePeerExchange` | `op_kernel_aicpu/exec_op.cc` | READY/Read/CONSUMED对称协议。 |

## 19. 提交前检查清单

- [ ] 只修改 `custom.h`、`allgather.cc`、`exec_op.cc` 及说明文档。
- [ ] `bash build.sh` 能同时生成 Host 与 AICPU 两个动态库。
- [ ] `HcclAllGather` 和 `HcclAICPUKernel` 导出符号存在。
- [ ] `aicpu_kernel.json` 能定位安装后的 Device 库和函数。
- [ ] FP32、Rank 1～16、空数据和溢出校验正常。
- [ ] 16 Rank小消息、大消息及Direct FullMesh回退均通过正确性测试。
- [ ] 17个Thread、15条Channel及Notify数量与协议一致。
- [ ] 末块CONSUMED只在不会再次覆盖Buffer时省略。
- [ ] `AlgResourceCtx`序列化和反序列化顺序完全一致。
- [ ] 压缩包根目录直接包含`README.md`、`CMakeLists.txt`、`build.sh`、`include/`、`op_host/`和`op_kernel_aicpu/`，没有额外包一层目录。

## 20. 许可证说明

根目录 `LICENSE` 当前保存 Apache License 2.0 文本，而源码文件头同时引用 CANN Open Software License Agreement Version 2.0。对外发布、复用或提交时，应以赛事官方授权要求为准并核对许可证口径；本 README 不修改原许可证文件。

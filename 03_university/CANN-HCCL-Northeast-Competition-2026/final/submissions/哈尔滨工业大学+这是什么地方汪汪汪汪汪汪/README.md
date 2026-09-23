# Scatter 集合通信算子（决赛）

## 团队信息

| 项目 | 内容 |
| --- | --- |
| 学校 | 哈尔滨工业大学 |
| 队伍名称 | 这是什么地方汪汪汪汪汪汪 |
| 队员 | `汪宏磊`、`安国豪`、`樊琪增` |


## 作品简介

本作品实现 `HcclScatter` 集合通信算子。root rank 的 `sendBuf` 按 rank 顺序保存连续的数据分片，每个分片包含 `recvCount` 个 FP32 元素；rank `r` 将第 `r` 个分片写入自己的 `recvBuf`。实现支持 rankSize 不超过 16 的通信域，并根据 HCCL RankGraph 的实例、链路和 CCU die 信息组织通信资源。

工程将 Host 侧资源编排与 CCU Kernel 执行结合起来：Host 侧负责参数检查、拓扑分析、channel/thread 申请、CCU Kernel 注册和任务下发；CCU Kernel 负责地址描述符同步、数据传输、本地拷贝及事件等待。CCU 源码与 Host 源码共同编译进 `libhccl.so`。

## 技术架构

- `op_host/scatter.cc` 通过 HCCL RankGraph 发现通信层和本地实例，按 `UBC_CTP`、`UBC_TP` 顺序选择 CCU 可用链路，获取端点 die，按 die 对等端分组并申请 channel。资源上下文最多保存两个 kernel group 和对应的 CCU thread。
- `op_host/exec_op.cc` 将单个 rank 的数据量划分为小消息或大消息，准备 CCU Kernel 参数，按 die group 调度主/子 kernel，并维护小消息的描述符缓存。
- `op_kernel_ccu/ccu_kernel.h` 定义 Scatter Kernel 参数和 helper 阶段；`op_kernel_ccu/ccu_kernel.cc` 实现通用 push、定向 pull、拓扑特化小消息、helper 转发和 8+4 拓扑的大消息路径。
- `include/custom.h` 定义固定布局的 `AlgResourceCtx`、CCU Kernel 参数、RankGraph 分组信息和 Descriptor Cache 状态，保证 Host 与 CCU Kernel 之间的资源上下文可直接复制。
- 顶层 CMake 将 `op_host` 和 `op_kernel_ccu` 加入同一个 `hccl` 共享库目标。决赛工程不单独生成 AICPU Device 动态库。

## 目录结构

归档工程保留完整官方模板结构：

```text
.
├── .clang-format
├── .gitattributes
├── .gitignore
├── CMakeLists.txt
├── LICENSE
├── README.md
├── build.sh
├── include
│   ├── binary_stream.h
│   ├── common.h
│   ├── custom.h
│   ├── hccl.h
│   └── log.h
├── op_host
│   ├── CMakeLists.txt
│   ├── exec_op.cc
│   ├── exec_op.h
│   └── scatter.cc
└── op_kernel_ccu
    ├── CMakeLists.txt
    ├── ccu_kernel.cc
    └── ccu_kernel.h
```

`build/` 等目录由构建过程生成，不属于源码归档。

## 算法设计

单个 rank 的数据量记为 `blockBytes = recvCount × sizeof(float)`。Host 侧以 `1 MiB` 的单 rank 分片大小为边界：`blockBytes < 1 MiB` 使用小消息上下文，`blockBytes >= 1 MiB` 使用大消息上下文。大消息再按不超过 `256 MiB` 的 chunk 下发 CCU Kernel。

### 小消息处理

#### 通用 push 路径

通用小消息由 `CcuKernel` 的 root/receiver 分支完成：

1. 非 root rank 通过 `WriteVariableWithNotify` 向 root 发布自己的输出地址和 CCU 内存 token。
2. root 等待各 peer 的地址和 token 就绪后，按 rank 顺序计算输入分片地址，通过 `Write` 将分片写入各 peer 的输出缓冲区；root 自身的分片使用 `LocalCopy`。
3. 每个传输由 CCU Event 记录和等待，所有 payload 完成后 root 通过 `NotifyRecord` 发布本轮完成信号，receiver 通过 `NotifyWait` 等待该信号。

#### 16 rank、32 KiB 分片的定向 pull

当 `rankSize == 16` 且 `blockBytes == 32 KiB` 时，使用定向 pull 协议：

- root 先处理自己的分片，再把每个目标分片的源地址和 token 发布到对应 channel 的 XN 中。
- receiver 等待地址和 token 就绪后，使用 `Read` 从 root 的远端源地址读入自己的输出缓冲区，等待 Event 完成后向 root 发送完成 notify。
- root 等待每个 receiver 的完成 notify。这条路径使用读取方向传输，避免让 root 为每个 receiver 排队提交 payload `Write`。

#### rank4 和 rank12 拓扑特化

Host 根据 RankGraph 选择特化 Kernel：

- rank4 且 `blockBytes == 128 KiB` 时，root 和 receiver 使用特化小消息 Kernel。
- rankSize 为 12、实例拓扑为 `4×3` 或 `8+4`，且 `blockBytes < 1 MiB` 时，root 使用 `CcuRank12SmallRootKernel`，receiver 使用 `CcuSmallReceiverKernel`。
- 特化 receiver 仍按先地址、后 token 的顺序发布描述符；在 `smallSingleReady` 条件下，token notify 作为地址和 token 都已更新的单一 ready edge。root 按 peer 的 ready 逐个提交 `Write`，并使用 Event 与 post notify 保护完成顺序。
- `4×3` 拓扑先提交跨实例写入，再等待本地 peer 的 ready；`8+4` 拓扑中，root 位于 8-rank 实例时，对 7 个本地 peer 保持逐 peer 的流水提交。

对于其他小消息，root 使用通用 push root 分支；rank4 的小总量 receiver 仍可使用 small receiver Kernel，但是否使用特化 root 由上述 `128 KiB` 条件决定。

### Descriptor Cache

Descriptor Cache 缓存的是通信描述符，不缓存 payload 数据。每个描述符由用户输出或输入的地址和 CCU memory token 组成：

- `pushRecvCache[root]` 保存 receiver 面向某个 root 的输出地址/token；数组按 root 保存。
- `pullSendCache` 保存 16-rank、32 KiB 定向 pull 路径中 root 的输入地址/token。
- `lastDescriptorCache` 保存当前缓存类型、root 和路径标识，用于约束缓存只能服务于连续且相同的通信路径。

缓存命中必须同时满足以下条件：

1. 当前调用属于小消息路径，并使用建立资源上下文的同一个 CCU thread。
2. 当前调用紧邻上一次使用该上下文的调用，没有被其他 root、路径或 stream 插入。
3. root、路径类型和当前地址/token 与缓存中的记录完全一致。
4. receiver push 使用 `pushRecvCache[root]`；16-rank、32 KiB 定向 pull 的 root 使用 `pullSendCache`。

命中时，Host 将 `metadataReuse` 置为 1。CCU Kernel 不再重复执行地址和 token 的 `WriteVariableWithNotify`，而是在原有 XN 保持不变的前提下，仅通过 `NotifyRecord` 发出本轮可消费的 ready credit。payload、Event 和 post notify 仍按正常协议完成。

未命中、地址/token 改变、root 或路径改变、调用来自其他 stream，或者大消息调用介入时，使用完整的描述符发布协议。Host 在调用结束后更新对应的地址/token 条目和 `lastDescriptorCache`；大消息开始前会清空小消息上下文的有效缓存状态。资源上下文版本号用于防止不同布局的上下文被误解释。

### 大消息处理

当 `blockBytes >= 1 MiB` 时，Host 按最多 256 MiB 的 chunk 循环调度。根据 RankGraph 分组和 peer 数量，选择直接 push、helper 混合或 8+4 专用 push。

#### 直接 push

当 helper route 不可用时，root 发布或等待各 receiver 的输出地址/token，然后通过 `Write` 将当前 chunk 写入各目标；receiver 使用 `WriteVariableWithNotify` 发布输出描述符并等待 root 的完成 notify。root 自身分片使用 `LocalCopy`，所有写入使用 CCU Event 统一收敛。

#### Helper 混合路径

当通信图存在可用的本地/跨实例分层、helper Kernel 已注册，且可建立满足 `localPeers > 0`、`remotePeers > 4`、`localPeers <= remotePeers` 的 route 时，使用 helper 混合路径。每个 chunk 的 delegated 前缀大小按 32 字节对齐计算：

```text
delegatedBytes = align32(
    chunkBytes × (remotePeers - 4) / (localPeers + 4))
directBytes = chunkBytes - delegatedBytes
```

root 将远端目标的 delegated 前缀写入本地 helper 的 staging slot，并将 helper 自身的分片写入 helper 的输出描述符；helper 通过 `HelperStage`、`HelperForward`、`HelperFinish` 三个阶段把前缀转发到配对的远端目标。root 同时将 direct 后缀直接写入远端目标。目标侧分别等待 helper 和 root 的 ready/data/post 信号，在不重叠输出区间的前提下合并两个部分。

Helper 协议使用额外的 stage 地址/token XN，并以独立的 Event、stage completion 和 helper completion 跟踪 staging 写入、转发和最终输出完成。

#### 8+4 拓扑专用 push

对于 rankSize 为 12 且 RankGraph 实例大小为 `8+4` 的拓扑，root 位于 8-rank 实例、且本地 peer 数多于远端 peer 数时，Host 选择 `CcuWidePushKernel`。该路径不使用 helper staging：跨实例目标在各自地址/token ready 后立即提交完整 chunk 的 `Write`，随后处理本地 peer，并由 Event 和 root post notify 收敛。其他 root 位置或其他拓扑回到直接 push 或 helper route 的条件判断。

## 算法选择思路

实现先按单个 rank 的分片大小区分协议，再按 RankGraph 拓扑选择 Kernel：

| 判断条件 | 主要协议或 Kernel |
| --- | --- |
| `blockBytes < 1 MiB`，普通拓扑 | 通用 push；root 发布描述符后 `Write` |
| `rankSize == 16` 且 `blockBytes == 32 KiB` | root `PullRoot`，receiver `PullReceiver` |
| rank4 `128 KiB`，或 rank12 `4×3`/`8+4` 且小于 1 MiB | 小消息特化 Kernel |
| `blockBytes >= 1 MiB`，helper route 可用 | Helper 混合路径 |
| rank12 `8+4` 且 root 位于 8-rank 实例 | 专用 wide push |
| 其他大消息情况 | 分 chunk 的直接 push |

这种选择让协议分支由真实 RankGraph 资源和代码中的固定边界共同决定，同时保留无法满足 helper 条件时的直接路径。

## 机内 Mesh 与跨机 Clos

工程不通过固定 rank 编号猜测所有拓扑，而是先从 RankGraph 选择实例数最多的 layer 作为本地分组依据，再为每个 peer 查找 CCU 可用链路。Host 按端点 die 将 channel 分为最多两个 kernel group，并把 `localGroupByRank` 和 `peerKernelGroups` 传入 CCU Kernel。

在赛事使用的分层拓扑中，本地实例内 channel 对应机内 Mesh，跨实例 channel 对应跨机 Clos：

- 小消息通用 push 使用本地 group 内的输出发布和写入，并通过跨实例 channel 触达远端 rank；16-rank 定向 pull 则由远端 receiver 跨实例读取 root 发布的源描述符。
- Helper 路径通过本地 group channel 把数据交给 helper，再由 helper 使用跨实例 channel 转发；`CLOS_CAPACITY = 4` 同时参与 helper delegated 比例的计算。
- 8+4 专用 push 对跨实例 channel 逐 ready 提交完整 chunk，对本地 channel 保持独立的本地提交顺序。

实际 channel 的链路和 endpoint 均来自 HCCL RankGraph；Host 只选择 `UBC_CTP` 或 `UBC_TP` 协议并检查所选 layer 与本地分组的一致性。

## 核心技术与 CCU 特性应用

- CCU channel 的 XN 资源保存输出地址、输出 token，以及 helper staging 地址和 token。`WriteVariableWithNotify` 用于发布描述符，`NotifyWait`/`NotifyRecord` 用于 ready、stage、data、ack 和 post 状态推进。
- `Write` 实现 root push 和 helper forward，`Read` 实现 16-rank 定向 pull，`LocalCopy` 处理 root 自身分片和接收后的用户输出拷贝。
- `EventRecord`、`EventWait` 以及 CCU 的 completion mask 跟踪每个 peer 的 DMA 完成，避免在远端缓冲区或输出区间尚未完成时复用资源。
- Host 使用 `HcommCcuGetMemToken` 为本轮输入/输出地址获取 token，使用 `HcommCcuKernelRegister` 注册按 die 分组的主、receiver、helper 和 wide-push Kernel，并通过 `HcommCcuKernelLaunch` 在用户 stream 对应的 CCU thread 上执行。
- EngineCtx 保存通信 buffer、thread、channel group、kernel handle、RankGraph 分组和 Descriptor Cache。资源上下文采用固定的可复制布局，缓存命中只复用已经验证的地址/token 描述符。

## 环境要求

- CANN 9.1.0。
- CMake 3.28.3。
- Host GCC 13.3.0。
- CCU 所需头文件、运行库和编译支持由 CANN 工具包提供。
- 构建前需要设置 `ASCEND_HOME_PATH`。执行下方的 `set_env.sh` 后，该变量应指向 CANN 安装根目录；`build.sh` 使用它作为 CANN 包路径，并通过 CMake 配置 Host 与 CCU 源码的联合构建。

## 编译运行

在工程根目录执行默认 Release 构建：

```bash
source /path/to/Ascend/cann-9.1.0/set_env.sh
bash build.sh
```

`build.sh` 会执行 CMake 配置、编译和安装。顶层 CMake 先创建 `hccl` 共享库目标，再把 `op_kernel_ccu/ccu_kernel.cc` 加入同一目标。完成后应生成：

```text
build/lib64/libhccl.so
```

决赛工程不包含独立的测试可执行程序；赛事提供的调用或评测程序加载 `libhccl.so` 后，通过 `HcclScatter` 接口执行算子。该工程已在 CANN 9.1.0、CMake 3.28.3 和 Host GCC 13.3.0 环境完成干净的 Release 构建验证；此处仅说明本地构建结果，不代表线上评测结果。

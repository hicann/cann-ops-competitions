# Scatter 集合通信算子（初赛）

## 团队信息

| 项目 | 内容 |
| --- | --- |
| 学校 | 哈尔滨工业大学 |
| 队伍名称 | 这是什么地方汪汪汪汪汪汪 |
| 队员 | 汪宏磊、安国豪、樊琪增 |

## 作品简介

本作品实现 `HcclScatter` 集合通信算子。调用时，root rank 的 `sendBuf` 按 rank 顺序保存连续的数据分片，每个分片包含 `recvCount` 个元素；rank `r` 将第 `r` 个分片写入自己的 `recvBuf`。当前实现只接受 FP32 数据，并针对两台服务器、每台 8 个 rank、共 16 个 rank 的通信域组织资源和数据路径。

工程采用 Host + AICPU+TS 两级结构。Host 侧负责校验参数、根据 RankGraph 选择链路、申请 thread/channel 资源并下发 AICPU Kernel；AICPU 侧负责在通信引擎上执行 Scatter 数据移动、同步和本地拷贝。

## 技术架构

- `op_host/scatter.cc` 实现公共接口、拓扑检查、资源申请和通信域上下文缓存。它为除自身以外的 15 个 rank 各申请一个 AICPU_TS channel，并建立按远端 rank 直接索引的资源表。
- `op_host/launch_aicpu_kernel.cc` 将用户 stream 映射到 CPU_TS thread，加载并启动 `HcclAICPUKernel`，完成 Host 与 AICPU 之间的启动和结束同步。
- `op_kernel_aicpu/aicpu_kernel.cc` 读取固定布局的资源上下文，在 AICPU_TS 上进入 batch 模式，等待 Host 信号后调用 Scatter 算法，并向 Host 回传完成信号。
- `op_kernel_aicpu/exec_op.cc` 实现小消息、512 KiB 分组中继和大消息三类路径。通信操作使用 AICPU TS 的 `Write`、`Read`、`LocalCopy` 和 channel notify 原语。
- 顶层 CMake 将 Host 目标和 AICPU Device 目标分别构建为 `libhccl.so` 与 `libhccl_device.so`。AICPU 目标通过顶层 `ExternalProject` 独立配置和构建。

资源组织如下：

| 资源 | 数量或约束 | 用途 |
| --- | --- | --- |
| rank | 16 | 两台服务器，每台 8 个 rank |
| AICPU_TS thread | 8 | 1 个控制 thread 加 7 个 Mesh worker |
| channel | 15 | 每个非自身 rank 一个直接索引的 channel |
| channel notify | 每个 channel 2 个 | ACK 和数据完成信号 |
| 单次大消息 chunk | 32 MiB | 在大消息路径中分片并流水处理 |
| HCCL 通信缓冲区 | 至少 64 MiB | 保存一个完整 chunk 及第二个 32 MiB slot |

## 目录结构

归档工程保留官方模板的完整结构：

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
│   ├── launch_aicpu_kernel.cc
│   ├── launch_aicpu_kernel.h
│   └── scatter.cc
└── op_kernel_aicpu
    ├── CMakeLists.txt
    ├── aicpu_kernel.cc
    ├── exec_op.cc
    └── exec_op.h
```

`build/`、`device_build/` 等目录是构建时生成的目录，不属于源码归档。

## 算法设计

设单个 rank 的数据量为 `blockBytes = recvCount × sizeof(float)`，root 输入总量为 `totalBytes = blockBytes × 16`。最终实现按 `totalBytes` 选择小消息或大消息路径。

### 小消息处理

当 `totalBytes <= 1 MiB` 时进入小消息路径。除精确的 512 KiB 场景外，`RunSmallScatter` 按以下方式组织数据：

1. root 用一次 `LocalCopy` 将另一台服务器上的连续 8 个分片放入本地 HCCL 通信缓冲区。
2. root 启动 4 个小消息 worker。每个 worker 负责若干 root 所在服务器内的目标 rank：通过本机 Mesh channel 写入对应的本地分片，同时在跨服务器 channel 上为同一 local index 的远端 rank 提供读取同步。
3. 远端 rank 通过 `Read` 从 root 的 HCCL 缓冲区读取自己的分片，再用 `LocalCopy` 写入用户输出；root 所在服务器的非 root rank 则接收 `Write` 后从本地 HCCL 缓冲区拷贝到输出。
4. root 自己的分片直接从 `sendBuf` 做 `LocalCopy`。控制 thread 等待远端读取完成和所有 worker 完成后结束本轮操作。

这种组织使同一 local index 的机内写入和跨服务器读取可以由不同的通信队列推进，同时保留每个 rank 的独立 channel。

### 512 KiB 分组中继路径

当 `totalBytes` 精确等于 `512 KiB` 时，执行 `RunSmallGroupedRelayScatter`：

- 4 个 worker 分别把远端服务器上的连续两个分片写入 4 个 relay rank 的 HCCL 缓冲区，每个跨服务器传输包含两个连续分片。
- 同一批 worker 并行写入 root 所在服务器的本地目标，root 自己的分片由控制 thread 做本地拷贝。
- 每个 relay rank 将收到的第一个分片拷贝到自身输出，并把第二个分片通过本机 Mesh channel 转发给相邻 rank；相邻 rank 接收后再拷贝到输出。
- root 所在服务器的目标 rank 直接接收 root 的写入并拷贝输出。

该分支只由 512 KiB 总输入量这一条件触发，不适用于其他小消息大小。

### 大消息处理

当 `totalBytes > 1 MiB` 时，执行大消息混合路径。每个分片按不超过 32 MiB 的 chunk 处理，helper 分担的前缀和 root 直接发送的后缀均按 FP32 元素计算，比例为 4/11 与 7/11。

- root 启动 7 个 worker，分别对应 root 所在服务器上的 7 个非 root rank。对每个 chunk，root 将某个远端目标的 4/11 前缀先写入对应 helper 的第二个通信缓冲区 slot，再将 helper 自身的分片写入 helper 的第一个 slot。
- root 的控制 thread 负责自己的分片本地拷贝，并将与 root 相同 local index 的远端目标完整写入；其他远端目标的 7/11 后缀直接写入远端缓冲区的第二个 slot。
- root 所在服务器的 helper 收到本地分片后，将第二个 slot 中的前缀通过跨服务器 channel 转发给配对的远端目标。远端目标使用两个独立的接收队列分别接收后缀和前缀，并把它们拷贝到输出的对应区间。
- chunk 之间通过 worker 完成通知保护通信缓冲区 slot 的复用；最后由控制 thread 等待所有 worker 和远端传输完成。

## 算法选择思路

选择依据完全由最终实现中的数据量和拓扑常量决定：

| 条件 | 路径 |
| --- | --- |
| `totalBytes <= 1 MiB` 且不等于 512 KiB | `RunSmallScatter` |
| `totalBytes == 512 KiB` | `RunSmallGroupedRelayScatter` |
| `totalBytes > 1 MiB` | `RunLargeRoot`、`RunLargeSourceHelper`、`RunLargeRemoteTarget` |

小消息路径避免为每个分片建立大消息流水；512 KiB 路径把远端连续分片合并后再中继；大消息路径固定使用 32 MiB chunk，并让本地 helper 的跨服务器转发与本地接收重叠。helper 分片大小由 4/11 常量计算，并保持元素边界。

## 核心技术与 CANN 特性应用

- 使用 HCCL RankGraph 的 layer 0 机内链路和非零 layer 跨服务器链路。AICPU_TS channel 选择 `UBC_CTP`、`UBC_TP`、`PCIE`、`UBOE` 的优先级，并为每个远端 rank 保留独立资源。
- `SendWrite` 先等待远端 ACK，再调用 `HcommWriteOnThread` 写入远端 HCCL 缓冲区，最后记录数据完成 notify；`RecvWrite` 记录 ACK 并等待数据完成。
- `RecvRead` 先等待 ACK，通过 `HcommReadOnThread` 读入已注册的本地 HCCL 缓冲区，再用 `HcommLocalCopyOnThread` 放入用户输出。这样避免将未注册的逐轮用户输出地址直接作为 AICPU TS RMA 目标。
- `LocalCopy` 用于 root 自身分片、机内接收后的落盘和中继后的输出拷贝；thread notify 用于控制 thread 与 worker 之间的启动、完成和 slot 复用同步。
- Host 将稳定的 thread、channel、HCCL buffer 和 rank 信息保存到固定布局的 EngineCtx；后续调用复用已申请资源，AICPU Kernel 只接收该资源上下文和本轮操作参数。

## 环境要求

- CANN 9.1.0。
- CMake 3.28.3。
- Host GCC 13.3.0。
- AICPU GCC 7.3.0；AICPU 交叉编译器由 CANN 工具链提供。
- 构建前需要设置 `ASCEND_HOME_PATH`。执行下方的 `set_env.sh` 后，该变量应指向 CANN 安装根目录；`build.sh` 使用它作为 CANN 包路径，并通过顶层 CMake 配置 Host 与 AICPU 两个构建目标。

## 编译运行

在工程根目录执行默认 Release 构建：

```bash
source /path/to/Ascend/cann-9.1.0/set_env.sh
bash build.sh
```

`build.sh` 会依次执行 CMake 配置、Host 构建、AICPU 子工程构建和安装。完成后，安装前缀下应生成：

```text
build/lib64/libhccl.so
build/lib64/libhccl_device.so
```

工程本身不包含独立的测试可执行程序；赛事提供的调用或评测程序加载上述动态库后，通过 `HcclScatter` 接口执行算子。该工程已在 CANN 9.1.0、CMake 3.28.3、Host GCC 13.3.0 和 AICPU GCC 7.3.0 环境完成干净的 Release 构建验证；此处仅说明本地构建结果，不代表线上评测结果。

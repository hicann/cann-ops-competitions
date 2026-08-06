# AllGather AICPU 集合通信算子

## 团队信息

- **团队名称**：沈门
- **团队成员**：吴正宇（队长）、张佳荷、陈雄
- **所属单位**：北京理工大学
- **联系方式**：930764273@qq.com
- **GitCode 账号**：weixin_61718454
- **作品编号**：106855

## 作品简介

本作品面向 HCCL AllGather 集合通信场景，实现了基于 AICPU 的自定义通信算子工程。工程包含 Host 侧拓扑查询、通信资源申请与上下文管理，以及 AICPU 侧通信算法编排。实现针对小消息提供 16 Rank 递归倍增路径，并为其他场景提供按远端 Rank 并行拉取数据的通用路径。

## 技术方案

### 技术架构

- `code/op_host/`：Host 侧 Rank 信息查询、链路查询、Channel/Thread 申请、资源上下文创建及 AICPU Kernel 下发。
- `code/op_kernel_aicpu/`：AICPU Kernel 入口、AllGather 算法选择与通信执行逻辑。
- `code/include/`：公共数据结构、算子接口、资源上下文、序列化工具和日志定义。
- `code/build.sh`：工程编译、Debug 构建及代码格式化入口。
- `code/CMakeLists.txt`：工程构建与安装配置。

### 执行流程

1. 调用 `HcclAllGather`，传入发送缓冲区、接收缓冲区、元素数量、数据类型、通信域和执行 Stream。
2. Host 侧获取当前 Rank ID 和 Rank 数量，并查询 Rank 间第 0/1 层可用链路。
3. 为每个远端 Rank 申请一条 `COMM_ENGINE_AICPU_TS` Channel，并保存远端 CCL Buffer 信息。
4. 申请主 Thread 和远端 Rank 对应的工作 Thread，将资源序列化到 HCCL Engine Context 中复用。
5. AICPU 侧将输入数据分片写入本地 CCL Buffer，根据 Rank 数量和消息大小选择通信路径。
6. 完成远端数据交换后，将各 Rank 数据写入接收缓冲区对应位置。

### 核心算法

#### 1. 16 Rank 递归倍增路径

当 Rank 数量为 16 且单 Rank 数据量不超过 1 MiB 时，采用递归倍增 AllGather：

- 通信距离依次为 1、2、4、8。
- 每轮与 `rank XOR distance` 对应的伙伴 Rank 交换当前已聚合的数据块。
- 经过 4 轮后，每个 Rank 获得全部 16 个 Rank 的数据。
- Rank 0～7 与 Rank 8～15 对应两台服务器，前三轮在服务器内完成，最后一轮完成跨服务器数据交换。

#### 2. 并行拉取路径

不满足递归倍增条件时，使用通用并行拉取路径：

- 当前 Rank 先将本地输入分片写入本地 CCL Buffer。
- 每个远端 Rank 对应一个工作 Thread 和一条 Channel。
- 各工作 Thread 并行从远端 CCL Buffer 读取数据，并直接写入最终接收缓冲区的对应 Rank 分段。
- 主 Thread 负责本地数据拷贝及工作 Thread 的启动、汇合和同步。
- 大消息按 CCL Buffer 容量和 128 字节对齐要求分片处理。

### 资源与同步设计

- 使用 `HcclRankGraphGetLinks` 查询 Rank 间链路。
- 使用 `HcclChannelAcquire` 申请 AICPU 通信 Channel。
- 使用 `HcclThreadAcquire`、`HcclThreadAcquireWithStream` 和 `HcclThreadExportToCommEngine` 管理执行 Thread。
- 使用 `HcclEngineCtxCreate`、`HcclEngineCtxCopy` 和 `HcclEngineCtxGet` 创建并复用资源上下文。
- 使用 `HcommWriteOnThread`、`HcommReadOnThread` 和 `HcommLocalCopyOnThread` 完成数据搬运。
- 使用 Channel Notify 和 Thread Notify 完成数据就绪、确认与线程汇合。

### CANN 特性应用

作品基于 CANN Toolkit 9.1.0 和 HCCL/HCOMM AICPU 通信编程接口开发，主要使用以下能力：

- HCCL 通信域、Rank 信息及 Rank Graph 链路查询。
- AICPU TS 通信引擎的 Thread 与 Channel 资源管理。
- HCCL Engine Context 资源缓存和自定义资源上下文序列化。
- HCOMM 远端读写、本地拷贝、Channel Notify 与 Thread Notify。

## 算子接口

对外接口定义于 `code/include/hccl.h`：

```cpp
HcclResult HcclAllGather(
    void *sendBuf,
    void *recvBuf,
    uint64_t sendCount,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream);
```

AllGather 完成后，每个 Rank 的接收缓冲区按 Rank ID 顺序存放所有 Rank 的输入数据：

```text
recvBuf = [rank 0 data][rank 1 data] ... [rank N-1 data]
```

## 运行说明

### 环境要求

- CANN Toolkit 9.1.0
- 比赛指定的 Ascend/AICPU 运行环境
- Linux 操作系统
- CMake 及支持 C++17 的 C++ 编译工具链

### 编译步骤

默认路径安装 CANN Toolkit 时执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
cd code
bash build.sh
```

若 CANN Toolkit 安装在自定义路径，应先加载对应环境变量脚本：

```bash
source ${install_path}/cann/set_env.sh
cd code
bash build.sh
```

Debug 构建：

```bash
cd code
bash build.sh --debug
```

代码格式化：

```bash
cd code
bash build.sh --format
```

更详细的工程说明请参阅 [`code/README.md`](code/README.md)。

## 支持范围

- 当前实现支持的数据类型为 `HCCL_DATA_TYPE_FP32`。
- Rank 数量为 1 时执行本地拷贝。
- 16 Rank、单 Rank 数据量不超过 1 MiB 时使用递归倍增路径。
- 其他场景使用按远端 Rank 并行拉取的通用路径。
- 大消息根据可用 CCL Buffer 容量进行分片。

## 正确性与性能说明

本 README 根据提交源码进行静态核对。本次未在当前本机重新编译或运行；具体正确性和性能结果以比赛指定环境及竞赛平台统一评测结果为准。

## 工程目录

```text
code/
├── CMakeLists.txt
├── build.sh
├── include/
│   ├── binary_stream.h
│   ├── common.h
│   ├── custom.h
│   ├── hccl.h
│   └── log.h
├── op_host/
│   ├── CMakeLists.txt
│   ├── allgather.cc
│   ├── launch_aicpu_kernel.cc
│   └── launch_aicpu_kernel.h
└── op_kernel_aicpu/
    ├── CMakeLists.txt
    ├── aicpu_kernel.cc
    ├── exec_op.cc
    └── exec_op.h
```

## 许可证

本工程的 `code/` 目录遵循 [`code/LICENSE`](code/LICENSE) 中所载的 Apache License 2.0。

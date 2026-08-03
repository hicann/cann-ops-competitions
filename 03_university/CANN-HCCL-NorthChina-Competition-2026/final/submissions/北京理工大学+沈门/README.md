# AllGather CCU 集合通信算子

## 团队信息

- **团队名称**：沈门
- **团队成员**：吴正宇（队长）、张佳荷、陈雄
- **所属单位**：北京理工大学
- **联系方式**：930764273@qq.com
- **GitCode 账号**：weixin_61718454
- **作品编号**：118705

## 作品简介

本作品面向 HCCL AllGather 集合通信场景，实现了基于 CCU 的自定义通信算子工程，包括 Host 侧资源申请与通信编排逻辑，以及 CCU 侧 Kernel 实现。

算子通过 HCCL/HCOMM 控制面接口查询 Rank 与网络拓扑、申请 Thread 和 Channel、注册 CCU Kernel，并通过 CCU 数据面接口完成地址交换、本地拷贝、远端写和同步。工程提供通用 Direct AllGather 路径，并针对 16 Rank、2×8 拓扑设计了分块 Relay 调度路径。

## 技术方案

### 技术架构

- `code/op_host/`：Host 侧资源申请、拓扑分析、Kernel 注册及通信任务编排。
- `code/op_kernel_ccu/`：CCU 侧通信 Kernel、数据搬运与同步逻辑。
- `code/include/`：公共数据结构、算子接口、资源上下文、序列化工具和日志定义。
- `code/build.sh`：工程编译、Debug 构建及代码格式化入口。
- `code/CMakeLists.txt`：工程构建与安装配置。

### 执行流程

1. 调用 `HcclAllGather`，传入发送缓冲区、接收缓冲区、元素数量、数据类型、通信域和执行 Stream。
2. Host 侧获取当前 Rank ID 和 Rank 数量，并通过 Rank Graph 查询网络层、端点和 Rank 间链路。
3. 根据链路协议申请 CCU Channel，按本地 Die 对 Channel 分组，并申请所需的 CCU Thread。
4. 注册 Direct Kernel；当拓扑满足条件时，同时生成 Relay 调度并注册 2×8 Relay Kernel。
5. 将 Thread、Kernel Handle 和路径选择信息序列化到 HCCL Engine Context，供同类后续调用复用。
6. 执行阶段为输入、输出缓冲区生成 CCU Memory Token，组织运行时参数并启动对应 Kernel。
7. CCU Kernel 交换各 Rank 的输出地址和 Token，通过本地拷贝、远端写、Event 及 Channel Notify 完成数据搬运和同步。

### 核心算法

#### 1. Direct AllGather

Direct 路径适用于 2～16 Rank 的通用场景，也覆盖 4 Rank 和 8 Rank 配置。

- 每个 Rank 将本地输入数据写入所有对端 Rank 接收缓冲区中属于自己的分段。
- 本 Rank 的输入通过 CCU 本地拷贝写入本地接收缓冲区。
- Channel 按本地 Die 分组；存在多个分组时，使用主从 Thread 并行启动对应的 Direct Kernel。
- 单次传输上限为 256 MiB；超过该大小时，Host 侧按元素边界拆分并多次启动 Kernel。
- 不超过 16 MiB 的数据使用直接 `LocalCopy` 路径；较大数据的本地拷贝使用基于 CCU Buffer、Loop 和 LoopGroup 的分组拷贝逻辑。

#### 2. 16 Rank 2×8 Relay AllGather

当 Rank 数量为 16、拓扑链路满足预期协议且所有 Channel 位于同一分组时，算子可对大消息启用 Relay 路径。该路径的启用阈值为单 Rank 数据量大于 1 MiB。

- 将每个 Rank 的输入均衡划分为 8 个 Chunk，并分别计算 Chunk 偏移和大小。
- 对数据所属服务器内的另外 7 个 Rank，使用 Mesh 原生传输。
- 对另一服务器，先通过 Clos 将每个 Chunk 发送到 5 个入口 Rank。
- 未直接收到该 Chunk 的另外 3 个 Rank，由已收到数据的入口 Rank 通过服务器内 Mesh 继续转发。
- Host 侧根据依赖关系、链路占用以及每个 Rank 的发送/接收容量生成调度，共预留 12 个 Slot。
- 每个 Slot 最多安排 7 个本地 Mesh 操作和 4 个 Clos 操作；调度器避免同一网络上的重复链路冲突。
- 中继数据在跨机写完成后通过 Channel Notify 发出就绪信号，转发 Rank 等待对应信号后再执行 Relay 写。

### 资源与同步设计

- 使用 `HcclRankGraphGetLayers`、`HcclRankGraphGetLinks` 和端点信息查询接口识别可用链路。
- 使用 `HcclChannelAcquire` 申请 `COMM_ENGINE_CCU` Channel，并为每条 Channel 配置同步 Notify 资源。
- 使用 `HcclThreadAcquireWithStream` 获取绑定调用 Stream 的主 Thread；多 Die 分组时额外申请从 Thread。
- 使用 `HcommCcuKernelRegisterStart`、`HcommCcuKernelRegister` 和 `HcommCcuKernelRegisterEnd` 完成 Kernel 注册。
- 使用 `HcommCcuGetMemToken` 为输入、输出通信内存生成访问 Token，并通过 `HcommCcuKernelLaunch` 启动 Kernel。
- 使用 `WriteVariableWithNotify` 交换远端输出地址与 Token，使用 `Write` 完成远端数据写入。
- 使用 CCU Event 等待本地传输完成，并使用 Channel Notify 完成 Relay 就绪通知和最终 Rank 间同步。

### CANN 特性应用

作品基于 CANN Toolkit 9.1.0 和 HCCL/HCOMM CCU 通信编程接口开发，主要应用以下能力：

- HCCL 通信域与 Rank 信息查询。
- HCCL Rank Graph 拓扑与链路查询。
- HCOMM CCU Thread、Channel 和 Kernel 生命周期管理。
- CCU 通信内存 Token、远端地址和远端写操作。
- CCU Variable、Event、Notify、Buffer、Loop 与 LoopGroup 编程能力。
- HCCL Engine Context 资源缓存与自定义资源上下文序列化。

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
- 比赛指定的 Ascend/CCU 运行环境
- Linux 操作系统
- CMake 3.16.0 或更高版本
- 支持 C++17 的 C++ 编译工具链

### 编译步骤

默认路径安装 CANN Toolkit 时执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
cd code
bash build.sh
```

若 CANN Toolkit 安装在自定义路径，应先加载对应的环境变量脚本：

```bash
source ${install_path}/cann/set_env.sh
cd code
bash build.sh
```

编译完成后，安装产物位于：

```text
code/build/lib64/libhccl.so
code/build/include/hccl.h
```

### Debug 构建

```bash
cd code
bash build.sh --debug
```

### 代码格式化

```bash
cd code
bash build.sh --format
```

### 运行与验证流程

本工程提供 AllGather 动态库与公开头文件，不单独包含测试驱动。应在比赛指定的多 Rank Ascend/CCU 环境中完成以下步骤：

1. 初始化 ACL 运行环境、设备、Stream 和 HCCL 通信域。
2. 为每个 Rank 分配输入、输出 Device 内存并准备输入数据。
3. 链接构建得到的 `libhccl.so`，调用 `HcclAllGather`。
4. 同步执行 Stream。
5. 将输出拷回 Host，检查每个 Rank 的输出是否按 Rank ID 顺序包含全部输入。
6. 释放 Device 内存、通信域、Stream 和 ACL 资源。

建议至少覆盖以下正确性用例：

- Rank 数量：1、2、4、8、16。
- 元素数量：0、1、非整除分块数量、边界附近数量和大消息数量。
- 路径：单 Rank 本地拷贝、Direct 路径、Direct 分片路径及 16 Rank Relay 路径。
- 数据检查：逐 Rank、逐元素比对输出与期望结果。

## 支持范围

- 当前实现支持的数据类型为 `HCCL_DATA_TYPE_FP32`。
- 支持的 Rank 数量为 1～16。
- 通用 Direct 路径适用于 2～16 Rank；1 Rank 使用本地拷贝。
- 16 Rank Relay 优化依赖比赛环境中的 2×8 Rank 布局、可用的 Mesh/Clos 链路以及单一 Channel 分组条件；条件不满足时回退到 Direct 路径。
- Relay 路径将数据切分为 8 个 Chunk，每个 Chunk 的大小不得超过 256 MiB。
- 具体编译、运行和性能表现以比赛指定的 CANN 版本、CCU 硬件环境及验证工具结果为准。

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
│   ├── exec_op.cc
│   └── exec_op.h
└── op_kernel_ccu/
    ├── CMakeLists.txt
    ├── ccu_kernel.cc
    └── ccu_kernel.h
```

## 许可证

本工程的 `code/` 目录遵循 [`code/LICENSE`](code/LICENSE) 中所载的 Apache License 2.0。
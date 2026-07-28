# Broadcast 集合通信算子（初赛）

## 团队信息

- **团队名称**：农民工队
- **团队成员**：Climo、Yann
- **所属单位**：浙江大学

## 作品简介

本作品面向 16 卡 Ascend NPU 环境实现 Broadcast 集合通信算子，针对小消息和大消息分别设计通信路径。小消息由 root 准备数据后通过点对点 Channel 直读完成广播；大消息将数据划分为 16 个逻辑分片，并利用双缓冲与多线程流水，在服务器内和服务器间逐阶段扩散数据。算子通过 CANN/HCCL 的 Host 资源管理、通信 Channel、AICPU Kernel 与设备侧通信原语完成端到端执行。

## 技术架构

整体采用 Host + AICPU Device 的分层结构：Host 侧解析拓扑、申请 Thread/Channel、创建通信引擎上下文并下发 AICPU Kernel；AICPU 侧等待 Host 同步信号，执行通信任务编排，最后通知 Host 完成。通信资源按照两台服务器的拓扑组织：每张卡申请同服务器内的 Mesh Channel 和跨服务器的 Clos Channel。

## 目录结构

```text
zju-农民工队/
├── README.md
└── code/
    ├── CMakeLists.txt                  # 顶层 CMake 配置
    ├── build.sh                        # 构建脚本
    ├── LICENSE                         # 开源协议
    ├── include/                        # 公共头文件和数据结构
    │   ├── hccl.h
    │   ├── common.h
    │   ├── custom.h
    │   ├── log.h
    │   └── binary_stream.h
    ├── op_host/                        # Host 侧资源申请与 Kernel 下发
    │   ├── broadcast.cc
    │   └── launch_aicpu_kernel.cc
    └── op_kernel_aicpu/                # AICPU 侧 Kernel 与算法编排
        ├── aicpu_kernel.cc
        ├── exec_op.cc
        └── exec_op.h
```

## 算法设计

### 小消息路径

当消息总大小不超过 1 MiB 时，root 先将用户数据复制到本地 HCCL Buffer，并通过通知机制告知其他 rank。非 root rank 等待 root 的通知后，直接从 root 的远端 HCCL Buffer Read 到自己的输出 Buffer，完成后回传完成通知。该路径避免了不必要的多阶段分片和中间同步。

### 大消息路径

大消息路径面向 16 个 rank，使用 16 个逻辑分片、双缓冲和最多 3 个 wave。每个 wave 按以下阶段推进：

1. root 将完整 wave 准备到 HCCL Buffer，其他 rank 并行 Scatter Read 自己负责的分片；
2. 通过跨服务器配对完成 1/16 到连续 1/8 的 Fold；
3. 通过两轮 Doubling 将数据扩展到连续 1/4 和本 rank 所需的 half；
4. 一路将本地 half Copy 到用户输出，另一路从伙伴 rank 直接 Read 远端 half 到用户输出；
5. 使用 Thread Notify 和 Channel Notify 管理阶段依赖，并在 wave 间复用双缓冲。

## 算法选择思路

- 小消息的主要目标是降低启动、同步和中间拷贝开销，因此选择 root 直读路径。
- 大消息的数据量较大，单次全量传输会占用较多 Buffer，因此采用固定大小 wave、16 分片和双缓冲，重叠不同阶段的执行。
- 16 卡拓扑包含服务器内和服务器间两类链路，算法先利用跨服务器配对完成分片聚合，再利用 Doubling 扩散数据，以匹配现有 Mesh/Clos Channel 资源。
- 使用 6 条 AICPU Thread 分别承担 Scatter、Fold、两级 Doubling、最终 Read 和 Copy，并通过通知实现流水依赖。

## 核心技术与 CANN 特性应用

- **HCCL 资源管理**：使用 HCCL Rank Graph、Channel、Thread 和通信引擎上下文接口申请并复用通信资源。
- **AICPU Kernel**：在设备侧执行通信任务编排，减少 Host 侧逐操作参与。
- **通信原语**：使用 `Read`、`LocalCopy`、`NotifyRecord` 和 `NotifyWait` 完成数据搬运与同步。
- **流水与双缓冲**：通过多 Thread、wave 和双缓冲提升通信阶段之间的重叠程度。

## 环境要求

- **CANN 版本**：CANN-Toolkit 9.1.0（或与赛事环境匹配的版本）
- **硬件要求**：支持本赛事 CANN/HCCL 通信接口的 16 卡 Ascend NPU，包含两台 8 卡服务器
- **依赖库**：CANN Toolkit、HCCL、HCOMM、CMake 3.16 及以上、C++ 编译工具链

## 编译运行

在 `code/` 目录下执行。首先安装 CANN Toolkit 并使环境变量生效：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

如为自定义安装路径，请将上述路径替换为实际安装路径。随后编译：

```bash
bash build.sh

# 可选：编译 Debug 版本
bash build.sh --debug

# 可选：格式化代码
bash build.sh --format
```

算子由赛事提供的测试/调用程序加载并执行，调用接口为 `HcclBroadcast`；运行时应在已配置 CANN 环境的 16 卡通信域中进行，并确保输入数据类型和 rank 配置符合代码约束。

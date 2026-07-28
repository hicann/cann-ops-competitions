# Broadcast 集合通信算子（决赛）

## 团队信息

- **团队名称**：农民工队
- **团队成员**：Climo、Yann
- **所属单位**：浙江大学

## 作品简介

本作品基于 CANN/HCCL 和 CCU 通信编程接口实现 Broadcast 集合通信算子。方案根据消息大小、rank 数量和通信拓扑选择不同的资源与 Kernel 组合：小消息采用 root 直发，大消息针对 4、12、16 rank 场景分别使用流水、分层流水或 Scatter + AllGather/Doubling 方案。Host 侧负责拓扑分析、Channel 申请、CCU Kernel 注册和任务下发，CCU 侧负责数据搬运及通知同步。

## 技术架构

系统由 Host 侧控制层和 CCU 侧设备通信层组成。Host 侧通过 HCCL Rank Graph 选择 UBC/CTP 链路，申请 Thread 和 Channel，注册 CCU Kernel，并按 Broadcast 计划组织任务参数。CCU Kernel 使用远端资源句柄、地址 Token、Write、Event 和 Notify 原语完成跨 rank 数据传输及阶段同步。公共结构和计划选择逻辑位于 `include/` 中，Host 编排位于 `op_host/`，设备侧 Kernel 位于 `op_kernel_ccu/`。

## 目录结构

```text
zju-农民工队/
├── README.md
└── code/
    ├── CMakeLists.txt                  # 顶层 CMake 配置
    ├── build.sh                        # 构建脚本
    ├── LICENSE                         # 开源协议
    ├── include/                        # 公共头文件、计划和资源结构
    │   ├── hccl.h
    │   ├── common.h
    │   ├── custom.h
    │   ├── log.h
    │   └── binary_stream.h
    ├── op_host/                        # Host 侧资源申请、注册和调度
    │   ├── broadcast.cc
    │   ├── exec_op.cc
    │   └── exec_op.h
    └── op_kernel_ccu/                  # CCU 侧通信 Kernel
        ├── ccu_kernel.cc
        └── ccu_kernel.h
```

## 算法设计

### 小消息

消息总大小不超过 1 MiB 时，采用 flat root 方案。root 将自己的地址和内存 Token 发布给对端，之后通过 CCU Write 直接发送数据；接收端只需完成地址交换和同步，不进行额外的多阶段数据重排。

### 4 rank 大消息

根据 root 的相对位置建立 4 rank 的链式关系，使用 16 个分片和 persistent CCU Kernel。数据沿 root 到各 rank 的 pipeline 逐片转发，使用两条通知 lane 复用同步信号，在传输和下游消费之间保持流水关系。

### 12 rank 大消息

针对 8+4 的服务器拓扑，将本地通信和跨服务器通信拆为多个阶段。通过 24 个 slice 和分层 Kernel 依次完成本地 Scatter/Gather、跨服务器 AllGather 以及跨 wave 数据交换，并使用两条 CCU Thread 推进不同阶段。

### 16 rank 大消息

root 首先执行 16 分片 Scatter，将数据分发到各 rank 的 HCCL/CCU Buffer；随后并行执行服务器内 7-peer AllGather 和跨服务器匹配传输，最终让每个 rank 获得完整数据。设备侧分别提供 Scatter Kernel 和 flat AllGather Kernel，Host 侧通过两个 worker Thread 组织两阶段任务。

## 算法选择思路

算法选择由 `rankSize`、消息总大小和 root 位置共同决定：

| 场景 | 通信计划 | 选择原因 |
|---|---|---|
| 消息不超过 1 MiB | flat root | 减少资源交换、Kernel 参数和同步开销 |
| 4 rank 大消息 | 4-rank pipeline | 利用简单链式拓扑和 persistent Kernel 重叠分片转发 |
| 12 rank 大消息且 root 位于 8 卡服务器 | 12-rank pipeline | 匹配 8+4 拓扑，分离本地与跨服务器阶段 |
| 16 rank 大消息 | scatter + doubling/all-gather | 先分片，再并行利用服务器内和服务器间链路扩散数据 |

不同计划共享统一的 `BroadcastPlan` 结构，使拓扑识别、消息分类、Channel 组网和 Kernel 调度相互解耦；同时通过小消息固定资源上下文减少重复序列化和资源扫描。

## 核心技术与 CANN 特性应用

- **CCU 编程模型**：使用 CANN CCU Kernel 注册、加载参数和下发接口，将通信编排下沉到设备侧。
- **HCCL 拓扑与资源管理**：使用 Rank Graph、Channel、Thread 和通信引擎上下文，根据链路协议选择可用 Channel。
- **远端资源与 Token**：通过 `WriteVariableWithNotify` 发布远端地址和 Token，再使用 `Write` 进行数据传输。
- **事件与通知同步**：使用 `EventWait` 等待并行传输完成，使用 `NotifyRecord/NotifyWait` 管理地址就绪、阶段完成和 lane 复用。
- **分层并行调度**：按消息大小和 rank 拓扑选择不同 Kernel，尽量并行利用服务器内外链路。

## 环境要求

- **CANN 版本**：CANN-Toolkit 9.1.0（或与赛事环境匹配的版本）
- **硬件要求**：支持本赛事 CANN/HCCL/CCU 接口的 Ascend NPU 集群；算法覆盖 4、12、16 rank 场景
- **依赖库**：CANN Toolkit、HCCL、HCOMM/CCU、CMake 3.16 及以上、C++ 编译工具链

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

算子由赛事提供的测试/调用程序加载并执行，调用接口为 `HcclBroadcast`；运行时应在已配置 CANN 环境的对应多卡通信域中进行，并确保输入数据类型、root 和 rank 配置符合代码约束。

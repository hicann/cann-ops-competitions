# ReduceScatter 集合通信算子（初赛）

## 提交信息

| 项目 | 内容 |
| ---- | ---- |
| 队伍名称 | SSHZY |
| 赛区 | 粤港澳赛区（GHM - Guangdong-Hong Kong-Macao） |
| 赛事 | 2026 HCCL 通信库创新大赛 |
| 阶段 | 初赛（Preliminary） |
| 算子名称 | ReduceScatter |
| 实现平台 | AICPU |

### 团队成员

| 姓名 | GitCode | 单位 | 分工 |
| ---- | ------- | ---- | ---- |
| szy | szyd123 | 哈尔滨工业大学（深圳） | Host 侧资源申请、拓扑解析与算法编排 |
| Burdach | Burdach | 鹏城实验室 | AICPU Kernel 通信算法设计与实现 |

## 算子语义

ReduceScatter 将每个 rank 的输入视为 `rankSize` 个连续块，每块包含 `count` 个元素。所有 rank 对相同块编号的数据执行归约操作（sum/max/min），编号为 `i` 的归约结果写入 rank `i` 的输出 buffer。

## 实现概述

初赛版本基于 AICPU 实现。Host 侧申请线程、notify、CCL buffer 和全连接 channel，并将拓扑与资源信息序列化下发到 AICPU Kernel；Device 侧根据 `rankSize`、数据量和 CCL 可用空间选择通信路径。

### 目标拓扑

代码面向 2x8 Ascend 950 仿真拓扑优化：Server 内 8 卡 Full-Mesh（layer-0），Server 间 Clos（layer-1）。Host 侧优先使用同 Server 直连链路，跨 Server 时从 layer-1 链路中选择目标 rank 对应的 channel。

### 算法路径

| 路径 | 适用条件 | 核心思路 |
| ---- | -------- | -------- |
| Mesh 多线程 | `rankSize` 为 2 的幂、CCL 可容纳 `rankSize - 1` 个远端 slot、总输入不小于 16 MB | `rankSize` 个线程并发执行 peer 间写入，主线程按固定 rank 顺序本地归约 |
| Recursive Halving | `rankSize` 为 2 的幂、小数据或 Mesh 不满足阈值 | 单线程执行 `log2(rankSize)` 步递归折半，通过 `WriteReduce` 将 send 半窗归约到对端 keep 半窗 |
| 分层双 WriteReduce | `rankSize` 非 2 的幂 | 使用 slotA/slotB 两个 CCL 工作区，先 Server 内归约，再跨 Server 单向归约，避免双向写同一区间 |

#### Mesh 多线程

该路径对标官方 `InsTempReduceScatterVMesh1D`，用于大数据场景降低串行通信开销。CCL buffer 被划分为多个 slot，每个远端 rank 写入独立 slot，避免并发写冲突。通信完成后，主线程按确定顺序执行 `LocalReduce`，保证归约顺序稳定。

#### Recursive Halving

该路径适合小数据和 2 的幂 rank 数。每一步将当前负责窗口一分为二，本 rank 保留 keep 半窗，将 send 半窗通过 `HcommWriteReduceOnThread` 原子归约到对端 CCL buffer。完成 `log2(rankSize)` 步后，窗口收缩为本 rank 的输出块。

#### 分层双 WriteReduce

该路径作为非 2 的幂 rank 数回退实现。slotA 保存本 rank 输出块的部分和，slotB 保存跨 Server 镜像块的部分和。Server 内先完成本地分层归约，跨 Server 阶段采用单向 `WriteReduce`，规避双方互写同一个累加器导致的 checker 内存冲突。

### 分片策略

所有路径均以本端 CCL buffer 作为工作窗口。当单个输出块超过工作窗口时，代码按 slice 对块内元素做纵向切分循环处理。slice 大小按 CCL 容量、数据类型宽度和算法 slot 数计算，并进行 32 元素对齐，减少极小尾片带来的额外握手开销。

### 同步约束

- 写入或归约到同一区间的任务使用 GO/DONE 两拍握手串行化。
- 不使用 `HcommWriteReduceWithNotifyOnThread`，避免隐式 notify 与仿真 checker 的跨 rank Record/Wait 依赖不兼容。
- Mesh 路径中远端写入使用独立 slot，回退路径中跨 Server 写归约保持单向流动。

## 项目结构

```text
├── CMakeLists.txt              # 顶层 CMake 配置
├── build.sh                    # 编译、Debug 编译和格式化脚本
├── include/
│   ├── hccl.h                  # 集合通信算子接口
│   ├── common.h                # 通用数据结构
│   ├── custom.h                # 自定义资源上下文、拓扑辅助函数和通信同步封装
│   ├── log.h                   # 日志宏
│   └── binary_stream.h         # 序列化工具
├── op_host/
│   ├── reduce_scatter.cc       # Host 侧算子入口、拓扑解析、资源申请
│   └── launch_aicpu_kernel.cc  # AICPU Kernel 加载与下发
└── op_kernel_aicpu/
    ├── CMakeLists.txt          # AICPU Kernel 构建配置
    ├── aicpu_kernel.cc         # AICPU Kernel 入口
    ├── exec_op.cc              # ReduceScatter 通信算法实现
    └── exec_op.h               # Device 侧算法接口
```

## 编译运行

### 环境要求

- CANN Toolkit 9.1.0
- 已配置 Ascend 编译工具链
- 已通过 `set_env.sh` 设置 `ASCEND_HOME_PATH`

### 安装并配置 CANN Toolkit

```bash
chmod +x Ascend-cann-toolkit_9.1.0_linux-${arch}.run
./Ascend-cann-toolkit_9.1.0_linux-${arch}.run --full --install-path=${install_path}
source ${install_path}/cann/set_env.sh
```

默认安装路径可使用：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

### 编译

```bash
bash build.sh
bash build.sh --debug
```

### 代码格式化

```bash
bash build.sh --format
```

编译产物会安装到当前提交目录下的 `build/` 目录。

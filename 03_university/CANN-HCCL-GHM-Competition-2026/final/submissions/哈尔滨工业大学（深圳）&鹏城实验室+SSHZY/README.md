# ReduceScatter 集合通信算子（决赛）

## 提交信息

| 项目 | 内容 |
| ---- | ---- |
| 队伍名称 | SSHZY |
| 赛区 | 粤港澳赛区（GHM - Guangdong-Hong Kong-Macao） |
| 赛事 | 2026 HCCL 通信库创新大赛 |
| 阶段 | 决赛（Final） |
| 算子名称 | ReduceScatter |
| 实现平台 | CCU（Compute & Communication Unit） |

### 团队成员

| 姓名 | GitCode | 单位 | 分工 |
| ---- | ------- | ---- | ---- |
| szy | szyd123 | 哈尔滨工业大学（深圳） | Host 侧资源申请与算法路由、双 die mesh、Recursive Halving 分片优化 |
| Burdach | Burdach | 鹏城实验室 | CCU Kernel 算法实现、ReadReduce 分组归约、4 rank 小数据路径 |

## 算子语义

ReduceScatter 将每个 rank 的输入视为 `rankSize` 个连续块，每块包含 `count` 个元素。所有 rank 对相同块编号的数据执行归约操作（sum/max/min），编号为 `i` 的归约结果写入 rank `i` 的输出 buffer。

## 实现概述

决赛版本基于 CCU 实现。Host 侧负责解析拓扑、申请 channel/CCU kernel/CCL buffer、计算分片参数，并根据资源和数据规模选择算法；CCU Kernel 侧执行具体的数据搬运、远端归约和本地归并。

### 目标拓扑

代码面向 2x8 Ascend 950 拓扑优化：Server 内 8 卡 Full-Mesh（layer-0），Server 间 Clos（layer-1）。Host 侧按 channel 所在 die 分组，优先利用双 die 并行能力；当路径需要所有通信 step 落在同一 IO die 时，会进行单 die 可用性检测。

### 算法路径

| 算法编号 | 名称 | 适用场景 | CCL 使用 |
| -------- | ---- | -------- | -------- |
| `RS_ALGO_MESH` | Mesh Read + LocalReduce | 通用主路径，大数据或 rank 数较大场景 | 使用 CCL scratch 保存远端输入和中间部分和 |
| `RS_ALGO_RH` | Recursive Halving WriteReduce | 小数据、2 的幂 rank 数、通信 step 可落在同一 die | 不依赖 CCL scratch，直接在输入区原地归约 |
| `RS_ALGO_GROUP_DRR` | Group Direct ReadReduce | 多 peer 大数据场景 | 按组使用 CCL scratch，直接 ReadReduce 远端输入 |
| `RS_ALGO_SMALL_4X1` | 4 rank 小数据专用路径 | `rankSize == 4` 的小规模场景 | 使用少量 CCL scratch 做固定拓扑归并 |

#### Mesh Read + LocalReduce

该路径对标 `ccu_temp_reduce_scatter_mesh_1D_mem2mem`。每个 rank 读取所有 peer 输入中属于本 rank 的结果块，将远端块放入本地 CCL scratch 后做本地归约。单 die 时由一个 CCU kernel 完成读取和归并；双 die 时按 channel die 分组并行读取，再由 merge kernel 合并两个 die 的部分和并写回 output。

实现中包含 pipeline merge 变体：当 CCU kernel 资源足够时，读取当前 slice 的同时归并上一 slice 的部分和，使通信和本地归约重叠，降低大数据场景下的收尾等待。

#### Recursive Halving WriteReduce

该路径通过 `WriteReduce(my_INPUT -> peer_INPUT)` 在输入区原地累积结果，不额外占用 CCL scratch。每一步将当前窗口折半，本 rank 保留 keep 半窗，将 send 半窗写归约到对端 keep 半窗。`rankSize == 16` 时共 4 步，通信复杂度为 `O(log n)`。

代码同时提供 contiguous 与 sliced 两种数据阶段。sliced 版本逐 block 传输当前 slice，避免跨 block 纵向切片时出现非连续内存访问。

#### Group Direct ReadReduce

该路径将 peer 划分为最多 4 个 group，每组最多 4 个 peer。CCU 直接从远端输入执行 `ReadReduce` 写入组内 scratch，减少一次显式 read 后再 local reduce 的中间搬运。各组得到部分和后，再在本地合并到 output，适合多 peer 大块数据吞吐优化。

#### 4 rank 小数据专用路径

`RS_ALGO_SMALL_4X1` 针对 4 rank 场景使用固定通信与归并模式，减少通用 mesh 路径的循环、分组和调度开销。该路径读取 3 个 peer 的目标块，在 scratch 内按固定顺序归并后写入 output。

### 分片与资源策略

- 单次通信大小受 `MAX_SINGLE_COMM` 和 CCL scratch 容量限制。
- Mesh 路径按 scratch slot 数计算 slice 大小，双 die pipeline merge 会额外预留 partial buffer。
- Recursive Halving 路径按最大单次通信大小切分，并对每个 slice 下发独立 CCU kernel 参数。
- Group Direct ReadReduce 路径根据 group 数、每组 chunk 大小和尾 chunk 大小构造 CCU 参数。

### 同步约束

- 对同一远端地址执行写归约时，使用显式同步顺序保证 checker 依赖完整。
- Recursive Halving 要求 step channel 满足 die 约束，避免跨 die 原地写归约路径不可用。
- 双 die mesh 路径中，两侧 die 的部分和写入独立 scratch 区，最终由 merge kernel 合并，避免互写同一区间。

## 项目结构

```text
├── CMakeLists.txt              # 顶层 CMake 配置
├── build.sh                    # 编译、Debug 编译和格式化脚本
├── include/
│   ├── hccl.h                  # 集合通信算子接口
│   ├── common.h                # 通用数据结构
│   ├── custom.h                # 自定义算法编号、资源上下文、die 分组和序列化逻辑
│   ├── log.h                   # 日志宏
│   └── binary_stream.h         # 序列化工具
├── op_host/
│   ├── CMakeLists.txt          # Host 侧构建配置
│   ├── reduce_scatter.cc       # Host 侧算子入口、资源申请和算法选择
│   ├── exec_op.cc              # CCU kernel 参数构造与调度
│   └── exec_op.h               # Host 侧算法编排接口
└── op_kernel_ccu/
    ├── CMakeLists.txt          # CCU Kernel 构建配置
    ├── ccu_kernel.cc           # CCU Kernel 多算法实现
    └── ccu_kernel.h            # CCU Kernel 参数结构与接口声明
```

## 编译运行

### 环境要求

- CANN Toolkit 9.1.0
- Ascend 950 系列硬件或匹配的仿真环境
- 已配置支持 CCU 编译的 CANN 工具链
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

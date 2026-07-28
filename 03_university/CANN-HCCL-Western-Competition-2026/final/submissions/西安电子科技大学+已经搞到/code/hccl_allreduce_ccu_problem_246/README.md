# AllReduce 集合通信算子

## 1. 项目介绍

```
├── CMakeLists.txt                  # 顶层 CMake 配置
├── build.sh                        # 构建脚本
├── .clang-format                   # 代码风格配置
├── include/                        # 头文件目录
│   ├── hccl.h                      # 集合通信算子头文件
│   ├── common.h                    # 通用数据结构定义
│   ├── custom.h                    # ★ 选手编写：自定义数据结构定义
│   ├── log.h                       # 日志宏定义
│   └── binary_stream.h             # 序列化类定义
├── op_host/                        # Host侧代码目录
│   ├── allreduce.cc                # ★ 选手编写：Host侧资源申请逻辑
│   └── exec_op.cc                  # ★ 选手编写：通信算法编排逻辑
└── op_kernel_ccu/                  # CCU侧代码目录
    └── ccu_kernel.cc               # ★ 选手编写：通信算法编排逻辑
```

> [!NOTE] 注意：
> 算子工程中已提前预制好固有逻辑，选手仅允许修改 `custom.h`、`allreduce.cc`、`exec_op.h`、`exec_op.cc`、`ccu_kernel.h`、`ccu_kernel.cc` 共 6 个文件内容。

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

### 2.3 编译算子工程

```bash
bash build.sh

# 编译 Debug 版本，便于断点调试
bash build.sh --debug
```

## 3. 代码格式

选手代码需符合 [.clang-format](.clang-format) 文件中的代码风格规范，可通过下列命令一键修改：

```bash
bash build.sh --format
```

## 4. V4.8：三拓扑 512 KiB 并行分块优化

V4.8 从平台全 Pass 的 V4.4 目录复制，保留原文件组织和全部大包
`algorithmKind=7/8/9`，只为三个 512 KiB 点新增独立路径：

1. `141 / 4×1`：每 rank 归约自己的 128 KiB slice；用 `16×8` MS
   LoopGroup 并行读取、LocalReduce，并在同一个 mission 中把结果写回三个 peer，
   将两次 Host launch 合并为一次。若真实平台把三条 channel 分到两个 die，
   自动回退到 V4.4 已通过的 kind2 lane 图。
2. `12_8_4 / 8+4`：两个 die 并发处理完整 512 KiB partial；每批至多
   `7 remote + local/partial`，随后沿用 V4.4 Host barrier 和 Combine。
3. `128 / 2×8`：与拓扑二使用相同的安全 batch primitive；non-owner die
   先用一个远端初始化 scratch，再并行归约最多七个远端，避免一次直接扇入
   八个远端的未验证平台风险。
4. 三条路径使用 `algorithmKind=10/11/12` 和独立 `v4_8` EngineCtx tag。
   其他 `<=1 MiB` 功能点和所有大包保持 V4.4 调度。

最终 Release 动态库 SHA-256：
`22614456eb140d3aaf272686256fb37a7ff5638b6e90dcc5ed4769a9bb623a3b`。

### 严格 VM 验证

最终二进制的三个 512 KiB 点均满足 `VM_INFRA_VALID=1`、
`OP_LOG_VALID=1`、`CHECKER_ANOMALY_COUNT=0`、语义/checker `1/1`：

| 拓扑 | 日志目录 |
| --- | --- |
| `141` / 4 rank | `run_logs/v4_8_final_t1_512k_141_4rank_524288B_20260724-182318/` |
| `12_8_4` / 12 rank | `run_logs/v4_8_final_t2_512k_12_8_4_12rank_524288B_20260724-182405/` |
| `128` / 16 rank | `run_logs/v4_8_final_t3_512k_128_16rank_524288B_20260724-182538/` |

另外三个拓扑的 4B 和 2 MiB+4B 回归均严格通过，证明精确的 512 KiB
分流没有误选旧小包或大包图。VM 只验证资源、同步和数值语义，性能收益仍须以
CANNJudge 三个小包点的实际用时为准。

## 5. 历史版本：V4.2 按平台结果收敛小包路径

V4.2 基于 V4.1 出分修正小包策略；V4.1 已修正 V4 把所有大包送入公共 runtime-phase lane 图的问题。
当前策略如下：

1. 大包按 rankSize 注册 `Large4/12/16`，执行平台已验证的 V2.3 rank-owned
   reduce-scatter/combine/allgather；不再进入 2/3 阶段的 V4 lane 循环。
2. `141 / 4×1 / 512KiB` 注册独立的 `CcuAllReduceSmall4Lane_*` mission，使用
   `CcuTopologyLaneKernel<4>` 编译实例和 4 lane 两阶段图。
3. `12_8_4 / 8+4 / 512KiB` 注册 `CcuAllReduceSmall12`，保留接近平台最优的 8 参数 one-shot。
4. `128 / 2×8 / 512KiB` 恢复 V2.3 的完整 `Small16` one-shot 图；V4.1 出分显示 8 lane 分层图为 111µs，
   没有超过 V2.3 的 109µs，因此不再将其作为默认路径。
5. 其他不超过 1MiB 的尺寸继续使用 V2.3 one-shot，避免扩大功能回归面。
6. EngineCtx tag 使用 `v4_2 + rankSize + path`，不同拓扑、包类别不会复用旧 V4/V4.1 资源。
7. 每个上下文每 die 只注册一张数据图；双 die 额外注册一张 combine，总 mission 数不超过 3。

### 512KiB 小包路径

1. `141 / 4×1` 复用 4-lane 两阶段 reduce-scatter/allgather，每个 rank 只处理 `N/4`。
2. `128 / 2×8` 复用 8-lane 三阶段 local-reduce/cross-exchange/local-gather，每个 rank 只处理 `N/8`。
3. `12_8_4 / 8+4` 保留已经接近平台最优的单次 one-shot12 图；Small12 与 Combine 均只加载和传递实际使用的
   8 个 runtime 参数，避免无用 phase 参数。
4. 三种路径使用独立 EngineCtx tag；512KiB 的选择不会改变大包和其他小尺寸的资源缓存。

### VM 严格验证（2026-07-23 10:39 UTC+8）

| VM 拓扑 | rank 数 | 数据量 | 结果 |
| --- | ---: | ---: | --- |
| `141`（4×1） | 4 | 512KiB | 独立 Small4Lane 严格通过，语义/checker 1/1，异常 0 |
| `12_8_4`（8+4） | 12 | 512KiB | V2.3 9参数 one-shot12 严格通过，语义/checker 1/1，异常 0 |
| `128`（2×8） | 16 | 512KiB | V2.3 9参数 one-shot16 严格通过，语义/checker 1/1，异常 0 |
| `141`（4×1） | 4 | 512MiB、400MiB+4B | V2.3 Large4 两项均严格通过 |
| `12_8_4`（8+4） | 12 | 512MiB、400MiB+4B | V2.3 Large12 两项均严格通过 |
| `128`（2×8） | 16 | 512MiB、400MiB+4B | V2.3 Large16 两项均严格通过 |

任何 `[HCCL-VM-CHECKER][ErrorCode: ...]` 都算失败，即使同时出现 `Checker Success`。

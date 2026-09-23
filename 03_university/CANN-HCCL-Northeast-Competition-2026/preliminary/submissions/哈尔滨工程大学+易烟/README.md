# HCCL Scatter 通信算子优化——初赛 AICPU V18

## 团队信息

- 团队名称：易烟
- 团队成员：文书磊
- 所属单位：哈尔滨工程大学
- 联系方式：wuw643999@gmail.com
- 赛事：2026 HCCL 通信库创新大赛（东北赛区）
- 阶段与版本：初赛 AICPU，V18
- 版本提交：`77b0ca6`
- 归档整理日期：2026-09-20

## 作品简介

面向 HCCL Scatter 集合通信，实现 root 将等长数据段分发到各 rank。结合通信拓扑、消息规模和缓冲容量安排传输与同步，减少集中发送和控制开销，用于昇腾分布式通信场景。本材料保存参赛版本的完整源码工程。

## 技术架构

Host 侧申请通信资源并启动 AICPU 内核，AICPU 侧依据消息大小、rank 数量和缓冲容量选择 Scatter 执行路径。16 rank 小包采用拓扑控制树，大包采用连续前缀中继与 root 直接发送后缀；缓冲条件不满足时保留分块流水路径。

代码结构：`code/include/` 保存接口及共享类型，`code/op_host/` 负责 Host 资源管理与调用，`code/op_kernel_aicpu/` 保存 AICPU 执行实现。构建入口为 `code/CMakeLists.txt` 和 `code/build.sh`。

## 核心技术与创新点

- 连续前缀与直接后缀分流，利用 Mesh/Clos 链路差异分担 root 发送负载。
- 在容量允许时保留常驻接收数据；循环缓冲路径通过 READY/ACK 和双槽控制复用。
- V18 在既有常驻路径中合并满足条件的两块直接后缀，减少传输调用；保持 helper 自有段粒度。

## CANN 特性应用

使用 CANN 的 ACL Runtime 与 HCOMM 通信能力，由 Host 启动 AICPU 内核，在设备侧组织通信和通知同步。保留赛题模板接口与构建结构，外部 SDK 不随源码归档。

## 环境要求

- CANN Toolkit 9.1.0，须具备赛题所需的 HCOMM、ACL、AICPU 开发库及 HCC 交叉编译工具链。
- CMake 3.16 或更新版本，支持 C++17 的编译器，Bash 和常规 Linux 构建工具。
- Host 链接依赖：`hcomm`、`acl_rt`。AICPU 侧链接 `ccl_kernel`，使用 Toolkit 内的 aarch64-target-linux-gnu 工具链。
- 运行验证需要赛事匹配的昇腾通信平台或官方仿真评测环境；普通 CPU 主机不能替代赛事环境完成正确性与性能验证。

## 构建步骤

从本 README 所在目录执行，环境脚本路径应替换为实际安装位置：

```bash
sha256sum -c SOURCE_SHA256SUMS
cd code
source /usr/local/Ascend/cann/set_env.sh
bash build.sh
# 需要调试构建时执行：bash build.sh --debug
```

环境脚本应正确设置 `ASCEND_HOME_PATH`。预期构建产物为 `build/lib64/libhccl.so`、`build/lib64/libhccl_device.so` 和 `build/include/hccl.h`。原始模板说明见 [code/README.md](code/README.md)。构建脚本的 `--format` 会修改源文件，不属于归档验证步骤。

## 使用方法

该工程输出算子动态库，由比赛测试程序或具备 HCCL 通信初始化能力的调用程序加载，并非独立命令行应用。公开入口见 [code/include/hccl.h](code/include/hccl.h)：

```cpp
HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream);
```

调用方按官方环境完成设备、通信域、stream 和设备内存初始化；所有 rank 使用一致的 `recvCount`、`dataType` 和 `root` 参与调用。root 的输入包含按 rank 排列的等长数据段，各 rank 接收一段，root 不必为 0。通过官方测试入口完成结果比对和计时，部署位置及启动参数遵循该测试环境说明。本归档不包含官方评测器及测试数据，不提供虚构的运行脚本。

## 性能指标与验证状态

| 平台测试点 | V18 历史显示耗时 |
| --- | ---: |
| 9 | 37 μs |
| 10 | 1.02 ms |
| 11 | 812 μs |

以上来自参赛期间保存的用户平台结果反馈，未附原始平台日志。未进行重复测量，不能据此给出稳定性或误差范围；不声明官方总分、排名或最终成绩有效性。

本次材料准备仅执行源码逐字节比对、SHA-256 校验及构建脚本语法检查，未重新编译，未运行设备测试。历史反馈不替代本次构建或平台验证；后续 PR 的 CLA、`/compile` 门禁与合入状态需在线确认。

## 版本与文件完整性

`code/` 保留原始工程文件，未改动算法、构建脚本、原始 README 或许可证。原始 `code/VERSION.md` 一并保留，其中相对路径仅描述原工作区的历史来源，独立使用本材料不依赖这些路径。 `SOURCE_SHA256SUMS` 覆盖全部源码目录文件；[docs/PROVENANCE.md](docs/PROVENANCE.md) 记录来源及比对范围。

## 应用价值

为拓扑不均匀、消息规模变化的 Scatter 通信提供传输调度与同步优化参考，可服务分布式任务的数据分发。迁移到其他硬件、拓扑或 SDK 版本时需要重新进行功能和性能验证。

## 许可证与参考资料

保留 [code/LICENSE](code/LICENSE) 及各源文件版权声明。模板 LICENSE 文件为 Apache-2.0 文本，部分源码头部注明 CANN Open Software License Agreement Version 2.0；本次归档原样保留上游标注，不作统一换证或重新授权声明。

- [赛事解读及成绩有效性说明](https://gitcode.com/cann/hccl/discussions/16)
- [竞赛代码归档仓库](https://gitcode.com/cann/cann-ops-competitions)
- [原始工程构建说明](code/README.md)

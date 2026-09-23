# HCCL Scatter 通信算子优化——决赛 CCU V20

## 团队信息

- 团队名称：易烟
- 团队成员：文书磊
- 所属单位：哈尔滨工程大学
- 联系方式：wuw643999@gmail.com
- 赛事：2026 HCCL 通信库创新大赛（东北赛区）
- 阶段与版本：决赛 CCU，V20
- 版本提交：`267e76661f8743283bde45d3fad710b2ff8cac76`
- 归档整理日期：2026-09-20

## 作品简介

面向 HCCL Scatter 集合通信，实现 root 将等长数据段分发到各 rank。结合通信拓扑、消息规模和缓冲容量安排传输与同步，减少集中发送和控制开销，用于昇腾分布式通信场景。本材料保存参赛版本的完整源码工程。

## 技术架构

Host 侧查询并缓存通信拓扑，依据运行时 root、数据规模和拓扑生成传输计划；CCU 内核执行数据读取和同步。针对 2×8、4×1、8+4、4×3 等赛题拓扑组织数据供应关系，保留通用处理路径。

代码结构：`code/include/` 保存接口及共享类型，`code/op_host/` 负责 Host 资源管理与调用，`code/op_kernel_ccu/` 保存 CCU 执行实现。构建入口为 `code/CMakeLists.txt` 和 `code/build.sh`。

## 核心技术与创新点

- 按拓扑和运行时参数生成 Pull 数据供应计划，使用每个 peer 一个通信 channel。
- V20 提前编排 2×8 场景的 helper 读取，压缩 8+4 场景中无数据 offer 的控制开销。
- 通过元数据发布、READY、DONE/RELEASE 同步管理数据可读性与 scratch 生命周期；不依赖测试编号或调用顺序选择算法。

## CANN 特性应用

使用 CANN 的 ACL Runtime 与 HCOMM 通信能力，由 Host 构造执行计划，在 CCU 内核中组织数据搬运和通信同步。保留赛题模板接口与构建结构，外部 SDK 不随源码归档。

## 环境要求

- CANN Toolkit 9.1.0，须具备赛题所需的 HCOMM、ACL、CCU 头文件与库。
- CMake 3.16 或更新版本，支持 C++17 的编译器，Bash 和常规 Linux 构建工具。
- Host 链接依赖：`hcomm`、`acl_rt`。CCU 相关声明来自 CANN 的 HCOMM/CCU 开发接口。
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

环境脚本应正确设置 `ASCEND_HOME_PATH`。预期构建产物为 `build/lib64/libhccl.so`（CCU 内核编入该库）。原始模板说明见 [code/README.md](code/README.md)。构建脚本的 `--format` 会修改源文件，不属于归档验证步骤。

## 使用方法

该工程输出算子动态库，由比赛测试程序或具备 HCCL 通信初始化能力的调用程序加载，并非独立命令行应用。公开入口见 [code/include/hccl.h](code/include/hccl.h)：

```cpp
HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream);
```

调用方按官方环境完成设备、通信域、stream 和设备内存初始化；所有 rank 使用一致的 `recvCount`、`dataType` 和 `root` 参与调用。root 的输入包含按 rank 排列的等长数据段，各 rank 接收一段，root 不必为 0。通过官方测试入口完成结果比对和计时，部署位置及启动参数遵循该测试环境说明。本归档不包含官方评测器及测试数据，不提供虚构的运行脚本。

## 性能指标与验证状态

参赛期间保存的反馈记录为：V20 性能点 19—30 全部 Pass，显示耗时与决赛 V18 逐点相同。当前材料未包含逐点原始耗时、官方总分及功能点 1—18 的完整日志，因此不填写未经核验的数据，也不据此声明全部功能测试通过或最终成绩有效。此处决赛 V18 与初赛 V18 属于不同版本序列。

本次材料准备仅执行源码逐字节比对、SHA-256 校验及构建脚本语法检查，未重新编译，未运行设备测试。历史反馈不替代本次构建或平台验证；后续 PR 的 CLA、`/compile` 门禁与合入状态需在线确认。

## 版本与文件完整性

`code/` 保留原始工程文件，未改动算法、构建脚本、原始 README 或许可证。源码来自已确认的最终 V20 完整导出。 `SOURCE_SHA256SUMS` 覆盖全部源码目录文件；[docs/PROVENANCE.md](docs/PROVENANCE.md) 记录来源及比对范围。

## 应用价值

为拓扑不均匀、消息规模变化的 Scatter 通信提供传输调度与同步优化参考，可服务分布式任务的数据分发。迁移到其他硬件、拓扑或 SDK 版本时需要重新进行功能和性能验证。

## 许可证与参考资料

保留 [code/LICENSE](code/LICENSE) 及各源文件版权声明。模板 LICENSE 文件为 Apache-2.0 文本，部分源码头部注明 CANN Open Software License Agreement Version 2.0；本次归档原样保留上游标注，不作统一换证或重新授权声明。

- [赛事解读及成绩有效性说明](https://gitcode.com/cann/hccl/discussions/16)
- [竞赛代码归档仓库](https://gitcode.com/cann/cann-ops-competitions)
- [原始工程构建说明](code/README.md)

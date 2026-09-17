# 官方规则与设计检查清单对应

2026-09-17读取[当前活动提交流程](https://www.hiascend.com/developer/activities/cann-community-task)、[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/b519c405eb235558dccf9af678c5ce17d112df49/04_tasks/01_community-task-2026/resources/design_template.md)、[官方流程说明](https://gitcode.com/org/cann/discussions/39)及其链接的[设计文档CheckList](https://docs.qq.com/sheet/DUHVGUFdmSFRjVFFU?tab=000001)。模板/任务目录本地检出固定于b519c405；在线表格为读取时快照，后续可能变更。

目标仓为cann/cann-ops-competitions，目录为`04_tasks/01_community-task-2026/tasklist/09-64-hyperloglog-950/2511_93680239/docs/`。模板必须章节已保留。本文包含设计正文、阅读入口及五个配套附录，不提交实现源码、测试或大体积日志。实际代码将来仍属于ops-collections纯头文件工程。

当前活动页面标题前缀为`【CANN社区任务】`，较早流程/检查表使用`【社区任务】`；本草稿采用当前活动页面格式：`【CANN社区任务】HyperLogLog算子设计文档`，正文明确它是容器。若平台创建页将来更新，以届时官方要求再核对，不凭本草稿宣称已满足线上CI。

| 清单类别 | 本文对应 | 状态/适用边界 |
| --- | --- | --- |
| 提交位置、PR与标题 | 上述目录与PR标题 | 设计文档评审范围 |
| CLA、PR构建/compile | 后续流程门槛 | 未据本地NPU结果推断签署或线上CI通过 |
| 需求来源与背景 | design.md第1部分 | target最高技术依据；说明HLL统计语义 |
| 标杆数据类型/格式、实现描述与流程 | 支持类型、数学公式及参考流程对比 | cuCollections参考；TBE/ACLNN文件路径项不适用本容器 |
| 外部组件、内部模块与Ascend C原理 | 工程接入、依赖与特性交叉 | CANN/ACL/SIMT/Extent/Catch2及许可证 |
| 调用方式 | requirements.md接口表、design.md调用示例 | 三工厂、六操作、错误与生命周期 |
| Host分核、分块、内存优化、TilingKey | Add覆盖证明、内存公式 | 显式参数传递；不适用ACLNN TilingKey注册 |
| Kernel描述、实现流程及与标杆差异 | design.md第3部分：流程、伪代码、并发论证及参考对比 | 单字节布局、位图/CAS/DMA、无遗漏证明 |
| 支持硬件与限制 | 950PR环境表、流/上下文前提 | 其他硬件/CANN未验证 |
| 特性交叉 | 工程接入末段 | 负数、分批、重复、容量、高rank、跨流 |
| 精度与性能标准 | 可维可测分析、性能附录 | 原1280、原72逐项，计时口径不变 |
| 兼容性分析 | 当前fmix64、固定seed、Key/precision约束 | 不兼容历史xxHash sketch；Destroy文字差异明示 |

“文档覆盖检查项”不等于“官方检查通过”。设计审阅仍须特别关注Destroy与通用异常表的口径，以及CAS和DMA字节原子并发的硬件前提。完整源码、原始证据尚未发布，补齐可访问链接后才具备线上独立验证入口。

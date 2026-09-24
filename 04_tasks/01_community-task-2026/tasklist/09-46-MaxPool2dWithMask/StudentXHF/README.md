# MaxPool2dWithMask 前向算子设计

2026-09-23更新：当前候选034在A2/910B3、CANN9.1.0重新编译后完成主精度165/165、补充21/21、geometry168/168、同源插桩40/40、Profiler5/5。候选无ERROR但保留1,062条同步冗余警告。本轮完整包447,609,328字节、4,735/4,735文件及历史034完整包675,218,840字节、7,782/7,782文件均已在本地逐项校验。固定上游AscendOpTest原始153例为153/153、306输出逐字节一致；其全量原始目录833,014,268字节/1,595条目和两次F140仿真全量目录68,641,743字节/311条目也已本地完成分块、整包哈希与解压目录复核。SDK参考36/162低于0.95、中位比值2.1707292883150098，不能作为官方TBE验收结论。进一步用`msprof`核对代表场景时，SDK实际派发到内置AscendC `MaxPool3DWithArgmaxV2`，并非原TBE身份凭据；`msopst`两次生成失败及完整Profiler原目录均保留在101文件补充取证包并逐文件校验。310P、官方TBE及正式验收未完成，增补设计评审仍待完成。详情见[9.1.0复测报告](https://gitcode.com/StudentXHF/ops-nn/blob/feat%2Fmaxpool2dwithmask-studentxhf/experimental/pooling/max_pool2d_with_mask/validation/20260922/REPORT.md)及设计文档“当前验证状态”。下文015及报名状态保留为历史记录。

贡献者：StudentXHF。任务：2026年9月社区任务09-46，目标硬件为Atlas 800T A2及Atlas 300V Pro（310P），CANN9.0.0或9.1.0。

设计评审状态补核（2026-09-22）：原设计PR [#1732](https://gitcode.com/cann/cann-ops-competitions/pull/1732)已合并，含CI通过和评审通过记录。本次后续增补PR1788待评审，不代表新增内容已获评审或正式验收通过。晚间登录恢复后已核实平台为“阶段2 任务开发”，正式提交表单可访问，尚未提交验收。HiDevLab 310P资格申请成功提交，预计2个工作日开通，尚无310P实测结果。

[034/9.1.0 A2自测表格](https://gitcode.com/StudentXHF/ops-nn/blob/feat%2Fmaxpool2dwithmask-studentxhf/experimental/pooling/max_pool2d_with_mask/validation/20260922/A2_selftest_034_CANN910.xlsx)含354个精度用例、45条插桩/Profiler记录及6900条耗时采样；官方TBE数据和待补截图保留空白。27项参数拒绝补测日志已纳入85文件补充包并本地逐项校验，1×1容量冲突安全拒绝不代表功能支持。

- [设计文档](docs/design.md)
- [正式ACLNN接口与约束](docs/aclnnMaxPool2dWithMask.md)
- [任务页面](https://www.hiascend.com/activities/task-center/details/6e72d6cde6a84ddb867f6b7d599914a5)
- [已提出的合同确认问题](https://gitcode.com/cann/ops-nn/discussions/8#discussion-comment-2eaa0ee79a9943709d0f8b442078d223)

本目录是设计评审材料，不包含已验收的Ascend C实现。310P索引格式、1×1容量及性能口径仍需评审确认。2026-09-19已在A2/910B3、CANN9.0.0构建候选，正式ACLNN入口的153个附件参数组合与12个补充用例共165/165通过；最终普通构建015精度165/165，参数拒绝27/27；插桩s016共20次候选无ERROR但有同步冗余警告。SDK性能参考85/162低于0.95、比值中位数0.8352258242，尚未达标；310P未验证。先按附件golden推进明确部分，后续根据评审意见更新。

由OpenAI Codex（GPT-6）辅助整理设计、核对源码和构造本地测试；未复制其他参与者实现。引用的CANN源码保留其原许可证和归属。

候选代码：[StudentXHF/ops-nn，feat/maxpool2dwithmask-studentxhf](https://gitcode.com/StudentXHF/ops-nn/tree/feat/maxpool2dwithmask-studentxhf/experimental/pooling/max_pool2d_with_mask)。独立公共API自测的完整输出和mask按未修改的附件golden精确比较；另有固定版本AscendOpTest 153/153工具精度结果。两者均不等于官方TBE性能或正式平台验收。

[首版自测报告与完整逐例表](https://gitcode.com/StudentXHF/ops-nn/blob/feat%2Fmaxpool2dwithmask-studentxhf/experimental/pooling/max_pool2d_with_mask/validation/20260919/REPORT.md)。报名仍审核中，尚未完成正式验收提交。

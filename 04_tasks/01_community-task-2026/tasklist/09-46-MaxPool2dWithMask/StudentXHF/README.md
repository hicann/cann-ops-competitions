# MaxPool2dWithMask 前向算子设计

贡献者：StudentXHF。任务：2026年9月社区任务09-46，目标硬件为Atlas 800T A2及Atlas 300V Pro（310P），CANN9.0.0或9.1.0。

- [设计文档](docs/design.md)
- [正式ACLNN接口与约束](docs/aclnnMaxPool2dWithMask.md)
- [任务页面](https://www.hiascend.com/activities/task-center/details/6e72d6cde6a84ddb867f6b7d599914a5)
- [已提出的合同确认问题](https://gitcode.com/cann/ops-nn/discussions/8#discussion-comment-2eaa0ee79a9943709d0f8b442078d223)

本目录是设计评审材料，不包含已验收的Ascend C实现。310P索引格式、1×1容量及性能口径仍需评审确认。2026-09-19已在A2/910B3、CANN9.0.0构建候选，正式ACLNN入口的153个附件参数组合与12个补充用例共165/165通过；最终普通构建015精度165/165，参数拒绝27/27；插桩s016共20次候选无ERROR但有同步冗余警告。SDK性能参考85/162低于0.95、比值中位数0.8352258242，尚未达标；310P未验证。先按附件golden推进明确部分，后续根据评审意见更新。

由OpenAI Codex（GPT-6）辅助整理设计、核对源码和构造本地测试；未复制其他参与者实现。引用的CANN源码保留其原许可证和归属。

候选代码：[StudentXHF/ops-nn，feat/maxpool2dwithmask-studentxhf](https://gitcode.com/StudentXHF/ops-nn/tree/feat/maxpool2dwithmask-studentxhf/experimental/pooling/max_pool2d_with_mask)。完整输出和mask按未修改的附件golden精确比较；这是公共API独立自测，不等于官方AscendOpTest验收。

[首版自测报告与完整逐例表](https://gitcode.com/StudentXHF/ops-nn/blob/feat%2Fmaxpool2dwithmask-studentxhf/experimental/pooling/max_pool2d_with_mask/validation/20260919/REPORT.md)。报名仍审核中，尚未完成正式验收提交。

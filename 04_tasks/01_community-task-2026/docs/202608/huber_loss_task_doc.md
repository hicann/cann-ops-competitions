# 8月社区任务 - huber_loss 算子开发任务书

## 任务概述

`aten::huber_loss` 算子未被 NPU 后端支持，自动 fallback 至 CPU 执行，导致训练耗时呈指数级增长，严重阻塞模型迭代周期。请参考 PyTorch 原生 aten::huber_loss（https://pytorch.org/docs/stable/generated/torch.nn.functional.huber_loss.html ） 实现，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的前向算子，完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。

## 核心开发要求

### 功能实现要求

1. 与 PyTorch `aten::huber_loss` 前向核心功能完全对齐：误差 `e = input - target`，当 `|e| ≤ delta` 时 `loss = 0.5 * e²`，当 `|e| > delta` 时 `loss = delta * (|e| - 0.5 * delta)`；支持 reduction 模式（none / mean / sum）。
2. 支持算子对应的数据类型 float32、float16 和 bfloat16，数据格式 ND。
3. 必须实现算子泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。

### 参数说明

| 参数名 | 输入／输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|---|---|---|---|---|---|---|---|
| input | 输入 | 预测值张量 | - | FLOAT32、FLOAT16、BFLOAT16 | ND | 任意维度，各维度 ≥ 0 | 支持 |
| target | 输入 | 目标值张量 | shape 和 dtype 须与 input 一致 | FLOAT32、FLOAT16、BFLOAT16 | ND | 与 input 相同 | 支持 |
| reduction | 可选属性 | 归约模式 | 默认1。0=none（逐元素输出），1=mean（取均值），2=sum（求和） | int | - | - | - |
| delta | 可选属性 | Huber loss 阈值参数 | 默认 1.0，须大于 0 | float | - | - | - |
| output | 输出 | loss 计算结果 | reduction=none 时与 input 同 shape；reduction=mean/sum 时为标量（0 维） | FLOAT32、FLOAT16、BFLOAT16 | ND | 与 input 相同或标量 | - |

### 算子约束限制

- input 与 target 必须具有相同的 shape 和 dtype，不支持 broadcast。
- reduction 取值仅支持 0（none）、1（mean）、2（sum），其他值属非法输入。
- delta 必须为正数（> 0）。
- 输出 dtype 与 input/target 一致；reduction=mean 时内部累加建议提升至 FP32 计算后再转回输出 dtype，避免精度损失。
- fusion：当前作为独立 loss 算子实现，不涉及图融合。

## 测试标准

测试硬件和软件要求：
- **适配硬件**：Atlas A2 训练系列产品
- **CANN 版本**：CANN 9.0.0 及以上

请根据给出的自测用例和测试指导完成自测，并输出自测报告。

### 性能要求

达到 80% 的 `compute bound` 或者 `memory bound`。不满足要求的场景需要特别说明。

### 精度要求

参照 [测试用例](./self_test_case/huber_loss)

算子计算精度需满足AscendOpTest工具默认阈值（https://gitcode.com/HIT1920/AscendOpTest/blob/master/compare/compare/accuracy_config.py ），且与 CPU `aten::huber_loss` 结果对齐。

## 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 算子目录下需要提供readme文件，参考：https://gitcode.com/cann/ops-transformer/blob/master/attention/chunk_gated_delta_rule/README.md <br> |

![邀请示意](./pics/invite.jpeg)

## PR 申请合入

测试通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入该目录： https://gitcode.com/cann/ops-nn/tree/master/experimental/loss 。

## 参考资料

1. 文档类：Ascend C算子开发文档（https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html ）、算子开发接口文档（https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html ）
2. 课程类：Ascend C在线课程（https://www.hiascend.com/developer/courses/detail/1691696509765107713 ）
3. 代码样例：[https://gitcode.com/cann/ops-nn/tree/master/experimental/loss](https://gitcode.com/cann/ops-nn/tree/master/experimental/loss)
4. 算法参考：PyTorch huber_loss 官方文档（https://pytorch.org/docs/stable/generated/torch.nn.functional.huber_loss.html ）、PyTorch 源码实现（https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp ）

## 环境获取

1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend ，点击 "体验 WebIDE"；
   - 如果是新用户，在申请权限的时候需要备注使用的算力类型（A2、A3或者950）。申请内容示例：本人gitcode账号是 yolo，现在参与社区任务"7月社区任务-aclnnRoll算子开发"，需要申请A2和950算力进行任务开发。
   - 如果是老用户且需要使用950算力，需要向昇腾CANN小助手反馈账号名（个人中心->基本信息，如下图所示），后台会添加账号至950使用白名单。

   ![环境截图](./pics/zaixiankaifa1.png)  
   ![账号名](./pics/account.png)  

2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](./pics/yunkaifa.png)

3. 申请算力资源后一般在1-2个工作日完成审批，如果没有审批完成，请及时在任务对应的讨论帖留言或者联系昇腾CANN小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读【社区任务】流程及注意事项（https://gitcode.com/org/cann/discussions/39 ），会例行更新。

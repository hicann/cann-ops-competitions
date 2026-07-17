# aclnnRoll算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Atlas A2、A3、A5 训练系列产品
- **开源仓地址**：https://gitcode.com/cann/ops-math
- **CANN 版本**：CANN 8.5.0及以上
- **开发语言**：Ascend C

## 任务概述

`torch.fft.fftshift` 和 `torch.fft.iftshift` 主要用于调整快速傅里叶变换（FFT）结果的频谱序，输入为 `torch.complex64` 是非常通用且主流的场景。`torch.fft.fftshift`、`torch.fft.iftshift` 的接口 API 调用的 `aclnnRoll` 算子在 NPU 上不支持 complex 类型输入，输入 tensor 的 dtype 为 complex64 时运行报错。

本任务要求在已有 `aclnnRoll` 算子基础上扩展支持 complex64 数据类型输入，使框架侧能够适配 `torch.fft.fftshift`、`torch.fft.iftshift` 输入为 complex64 的场景。

## 核心开发要求

### 功能实现要求

1. 在 `aclnnRoll` 现有实现基础上扩展支持 complex64 输入，complex64 输入下计算结果与 PyTorch `torch.roll` 语义一致。
2. 保持原有支持的数据类型（如 float16/bfloat16/float32/int 等）功能不变。
3. 必须实现算子泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） | 非连续Tensor |
|--------|-----------|------|----------|----------|----------|---------------|--------------|
| x（aclTensor*） | 输入 | 输入的原始数据。 | - | BFLOAT16、FLOAT16、FLOAT32、INT8、UINT8、INT32、UINT32、COMPLEX64（新增） | ND | 0-8 | √ |
| shifts（aclIntArray*） | 输入 | 指定每个维度上要滚动的步数。 | 数组长度与dims保持一致。 | - | - | - | - |
| dims（aclIntArray*） | 输入 | 指定要滚动的维度。 | 数组长度与shifts保持一致，取值范围在[-x.dim(), x.dim() - 1]之内，例如：x的维度是4，则取值范围在[-4, 3]。 | - | - | - | - |
| out（aclTensor*） | 输出 | 滚动处理后的输出数据。 | - | BFLOAT16、FLOAT16、FLOAT32、INT8、UINT8、INT32、UINT32、COMPLEX64（新增） | ND | 0-8 | - |
| workspaceSize（uint64_t*） | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - | - |
| executor（aclOpExecutor**） | 输出 | 返回op执行器，包含了算子计算流程。 | - | - | - | - | - |

### 算子约束限制

无

## 测试标准

请根据给出的自测用例和测试指导完成自测，并输出自测报告。

### 性能要求

无

### 精度要求

- complex64 输出精度需与 CPU（对标）对齐，采用 [AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest) 默认阈值。

## 验收交付件

1. 算子设计文档（含 complex64 扩展方案），内容完整、格式规范，且必须通过评审。
2. 可指导算子使用与测试的交付件：自测用例、测试代码脚本及脚本运行 README；测试结果报告，需包含所有自测用例结果，含测试用例执行日志/截图、整体测试通过截图、性能数据截图。
3. 算子代码的私仓邀请链接、代码仓路径、分支、算子目录。README 文档内容完整、规范。

## PR 申请合入

测试通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入对应目录：(https://gitcode.com/cann/ops-math/tree/master/experimental/math/roll)。

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
2. PyTorch 参考：[torch.roll](https://pytorch.org/docs/stable/generated/torch.roll.html)、[torch.fft.fftshift](https://pytorch.org/docs/stable/fft.html)
3. 代码样例：https://gitcode.com/cann/ops-math/tree/master/conversion/roll

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  

3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 扩展 complex64 不得影响原有 dtype 功能与性能，需补充回归测试。
4. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。
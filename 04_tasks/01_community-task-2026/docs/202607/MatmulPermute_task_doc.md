# 7月社区任务-MatmulPermute算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Ascend 950PR/Ascend 950DT
- **开源仓地址**：[https://gitcode.com/cann/catlass](https://gitcode.com/cann/catlass)
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 任务概述

本算子实现 Matmul 后接输出张量维度重排（Permute）的融合算子。Matmul 计算结果在写入显存时按照 Tensor4DPermute0213 规则重新排列。

**算子公式**

标准 Matmul：$\mathbf{D} = \mathbf{A} \times \mathbf{B}$，输出形状 $(M, N)$。

融合 Permute 后，输出按以下规则写入：

$$ \text{Matrix}(M, N) \xrightarrow{\text{reshape}} \text{Tensor4D}(M/S_1, S_1, S_2, N/S_2) \xrightarrow{\text{permute}[0,2,1,3]} \text{Tensor4D}(M/S_1, S_2, S_1, N/S_2) $$

## 核心开发要求及验收标准

### 功能实现要求

1. 实现 Matmul + 融合输出 Tensor4DPermute0213，单次 kernel launch 完成全部计算。
2. $S_1$、$S_2$ 为编译期常数，由 host 侧组装时传入模板参数。
3. 后处理实现需支持泛化 $S_1$/$S_2$，验收时 $S_1=8, S_2=4$。
4. 输入 $M$ 需被 $S_1$ 整除，$N$ 需被 $S_2$ 整除。

### 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|--------------|------|---------|----------|----------|-------------|-------------|
| A | 输入 | Matmul 左矩阵，形状 (M, K) | RowMajor 布局 | FP16 | ND | 2 | — |
| B | 输入 | Matmul 右矩阵，形状 (K, N) | RowMajor 布局 | FP16 | ND | 2 | — |
| S1 | 编译时属性 | reshape 第一维因子 | M % S1 == 0 | INT32 | — | 0 | — |
| S2 | 编译时属性 | reshape 第二维因子 | N % S2 == 0 | INT32 | — | 0 | — |
| D | 输出 | Matmul 输出矩阵 | 按 Tensor4DPermute0213 写入显存 | FP16 | ND | 4 (permuted) | — |

### 算子约束限制

- $M$ 需被 $S_1$ 整除，$N$ 需被 $S_2$ 整除

### 测试标准

需参考 CPU 精度标杆自行设计自验证用例，覆盖多种问题规模和 $S_1/S_2$ 组合。自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

性能标杆为[CUTLASS的39_gemm_permute](https://github.com/NVIDIA/cutlass/tree/main/examples/39_gemm_permute)，算子整体性能需与 0.8 倍 GPU（H100）持平。

### 精度要求

算子计算精度需满足 [AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest) 工具默认阈值。

### 文档规范要求

1. 算子设计文档需根据[参考模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；
2. 自验证报告需要覆盖所有功能场景，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)，含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可清晰指导算子使用与测试；
3. README 文档内容完整、规范。

## 验收交付件

1, 自测用例、测试结果报告、测试步骤指导文档

2, 算子代码的私仓邀请链接、代码仓路径、分支、算子目录

## PR 申请合入

测试通过后，在 Catlass 代码仓提交 PR 申请，申请将开发完成的算子合入https://gitcode.com/cann/catlass，具体目录参考该PR：https://gitcode.com/cann/catlass/pull/678 。

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[CATLASS创新样例开发流程指南](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/10_innovative_example_development_guide.md)
2. 课程类：[Ascend C在线课程](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
3. 参考样例：[https://gitcode.com/cann/catlass/blob/master/examples/44_quant_matmul_full_loadA_tla](https://gitcode.com/cann/catlass/blob/master/examples/44_quant_matmul_full_loadA_tla)

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab notebook 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  

3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。

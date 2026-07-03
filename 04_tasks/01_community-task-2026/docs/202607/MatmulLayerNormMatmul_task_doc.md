# MatmulLayerNormMatmul算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Ascend 950PR/Ascend 950DT
- **开源仓地址**：[https://gitcode.com/cann/catlass](https://gitcode.com/cann/catlass)
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 任务概述

本算子实现融合 LayerNorm 的双 Matmul 算子（Matmul + LayerNorm + Matmul）。先执行一个 Matmul，对其输出做 LayerNorm（行级归一化），再以归一化后的结果作为第二个 Matmul 的左矩阵，整个过程无需中间显存读写。

**算子公式**

阶段一（第一个 Matmul）：$\mathbf{C0} = \mathbf{A0} \times \mathbf{B0}$，输出形状 $(M_0, N_0)$，RowMajor 布局。

阶段二（LayerNorm，对 $\mathbf{C0}$ 的每一行独立计算）：
$$ 
\begin{aligned}
\mu[m] &= \frac{1}{N_0} \sum_{n=0}^{N_0-1} \mathbf{C0}[m, n] \\
\sigma^2[m] &= \frac{1}{N_0} \sum_{n=0}^{N_0-1} \big(\mathbf{C0}[m, n] - \mu[m]\big)^2 \\
\mathbf{C0}_{\text{norm}}[m, n] &= \frac{\mathbf{C0}[m, n] - \mu[m]}{\sqrt{\sigma^2[m] + \epsilon}} \cdot \gamma[n] + \beta[n]
\end{aligned}
$$

其中 $\epsilon = 10^{-6}$，$\boldsymbol{\gamma}$（缩放）和 $\boldsymbol{\beta}$（偏置）为可学习参数，长度均为 $N_0$。

阶段三（第二个 Matmul）：$\mathbf{C1} = \mathbf{C0}_{\text{norm}} \times \mathbf{B1}$，输出形状 $(M_0, M_1)$。

## 核心开发要求及验收标准

### 功能实现要求

1. 实现融合的 Matmul → LayerNorm → Matmul，单次 kernel launch 完成全部三段计算。
2. Mean 和 Variance 等中间结果由算子内部 workspace 管理，不对外暴露。

### 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|--------------|------|---------|----------|----------|-------------|-------------|
| A0 | 输入 | 第一个 Matmul 左矩阵，形状 (M₀, K₀) | RowMajor 布局 | FP16 | ND | 2 | — |
| B0 | 输入 | 第一个 Matmul 右矩阵，形状 (K₀, N₀) | ColumnMajor 布局 | FP16 | ND | 2 | — |
| B1 | 输入 | 第二个 Matmul 右矩阵，形状 (N₀, M₁) | ColumnMajor 布局 | FP16 | ND | 2 | — |
| gamma | 输入 | LayerNorm 缩放参数，长度 N₀ | 可学习参数 | FP32 | ND | 1 | — |
| beta | 输入 | LayerNorm 偏置参数，长度 N₀ | 可学习参数 | FP32 | ND | 1 | — |
| C1 | 输出 | 第二个 Matmul 输出矩阵，形状 (M₀, M₁) | 唯一对外输出 | FP16 | ND | 2 | — |

### 算子约束限制

- 无

### 测试标准

需参考 CPU 精度标杆自行设计自验证用例，覆盖多种问题规模组合。自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

性能标杆为[CUTLASS的37_gemm_layernorm_gemm_fusion](https://github.com/NVIDIA/cutlass/tree/main/examples/37_gemm_layernorm_gemm_fusion)，算子整体性能需与 0.8 倍 GPU（H100）持平。

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

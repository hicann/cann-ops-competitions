# aclsolverCheevj算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Atlas A2 训练系列产品
- **开源仓地址**：[https://gitcode.com/cann/ops-solver](https://gitcode.com/cann/ops-solver)
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 任务概述

参考 cuSOLVER 库中的 `cusolverDnCheevj` 接口（https://docs.nvidia.com/cuda/cusolver/index.html#cuSOLVER-function-cheevj），在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的算子，完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。

## 算子定义

### 接口定义

```c
void aclsolverCheevj(
    const char *jobz,          // 'N': 仅计算特征值, 'V': 计算特征值和特征向量
    const char *uplo,          // 'U': 上三角, 'L': 下三角
    int n,                     // 矩阵 A 的阶数
    cuComplex *a,              // Hermitian矩阵 A
    int lda,                   // 矩阵 A 的前导维度
    float *w,                  // 特征值数组，按升序排列
    cuComplex *work,           // 工作空间数组
    int lwork,                 // 工作空间大小
    float *rwork,              // 实数工作空间数组
    int *info,                 // 输出信息：0=成功, >0=未收敛, <0=参数错误
    const syevjInfo_t params   // Jacobi算法参数配置（容忍度、最大扫描次数等）
)
```

### 功能描述

使用 Jacobi 方法计算 Hermitian 矩阵 A 的特征值和（可选）特征向量：`A * V = V * diag(w)`，其中 w 为按升序排列的特征值数组，V 的列为对应的特征向量。

### 数据类型

- 输入矩阵：单精度复数浮点数（complex64，即 cuComplex）
- 特征值：单精度浮点数（float32）
- 特征向量：单精度复数浮点数（complex64）
- 数据格式：列主序（Column-Major）

## 核心开发要求及验收标准

### 功能实现要求

1. 与 cuSOLVER `cusolverDnCheevj` 核心功能完全对齐，支持单精度复数数据类型。
2. 必须支持仅计算特征值（jobz='N'）和计算特征值及特征向量（jobz='V'）两种模式。
3. 必须支持上下三角模式（uplo='U'/'L'）。
4. 特征值必须按升序排列。
5. 必须实现算子泛化功能，满足各类合法输入场景的计算需求，验收阶段将采用泛化数据进行验收。

### 测试标准

测试用例覆盖**常规场景、边界场景**等所有功能场景；自验证报告完整、可复现，所有测试用例执行通过。

### 性能要求

1. 算子整体性能需与0.8倍GPU（A100）持平，具体如下：

| N | jobz | uplo | GPU A100耗时(ms) | GPU A100 GFLOPS |
|---|------|------|-------------------|-----------------|
| 512 | V | U | 21.130 | 16.9 |
| 1024 | V | U | 87.717 | 32.6 |
| 2048 | V | L | 483.208 | 47.4 |
| 1024 | N | U | 8.293 | 172.6 |

### 精度要求

参考该链接编写测试用例，使得Ascend C算子结果和python实现结果一致：
https://gitcode.com/cann/ops-solver/tree/master/test/cgetri

### 文档规范要求

1. 算子设计文档需根据[参考模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写，内容完整、格式规范，且必须通过评审；
2. 自验证报告需要覆盖所有功能场景，参考[xxx算子自验证报告](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)（不需要模板中TBE样例结果），含测试用例执行日志/截图、整体测试通过截图、性能数据截图，可清晰指导算子使用与测试；
3. README 文档内容完整、规范。

## 验收规则与流程

### 提交验收申请

联系昇腾小助手，提交以下**三类交付件**进行验收：

1. 昇腾开源算子仓 fork 的个人代码仓链接（需包含：算子工程代码、算子 README 文档、多组 aclnn 调用测试代码）；
2. 算子自验证报告；
3. 华为评审通过的算子设计文档（按模板填写），合入 [cann-competitions 仓库](https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist) 详细说明见 [readme](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。

### 验收结果反馈

验收以提交验收申请时的代码为准，72小时内反馈验收结果，如代码更新请重新提交验收申请，验收时间同步刷新。

### PR 申请合入

验收通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入 https://gitcode.com/cann/ops-solver/tree/master/experimental。

## 参考资料

1. 文档类：[Ascend C算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)、[算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
2. 课程类：[Ascend C在线课程](https://www.hiascend.com/developer/courses/detail/1691696509765107713)、[cuSOLVER Cheevj 参考](https://docs.nvidia.com/cuda/cusolver/index.html#cuSOLVER-function-cheevj)
3. 代码样例：[https://gitcode.com/cann/ops-solver/tree/master/experimental](https://gitcode.com/cann/ops-solver/tree/master/experimental)

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab notebook 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)
   ![环境截图](pics/zaixiankaifa2.png)

3. 如需额外环境资源，请联系昇腾小助手。

## 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)，会例行更新。

# Erf 算子优化作品

## 团队信息

- 团队名称：关注塔菲喵
- 所属单位：东南大学（Southeast University）
- 团队成员：
  - 董子鹏（队长，GitCode：`shysta`），负责整体技术方案、核心 Kernel 实现、Tiling 与性能优化、版本管理和最终收网
  - 龚巧旋（GitCode：`sleepybird`），负责精度验证、测试结果分析和实验方案评估
  - 张毓琛（GitCode：`hellosahiiii`），负责资料调研、测试验证和文档整理
- 联系人：董子鹏
- 联系邮箱：2682404507@qq.com

## 作品简介

本作品为 2026 CANN 算子挑战赛江山赛区预选赛 Erf 算子的 AscendC 实现。

Erf 是逐元素高斯误差函数。实现需要在满足 float32 精度要求、支持非 32 字节对齐输入的同时，兼顾不同输入规模下的启动开销、数据搬运、向量计算和多核负载均衡。

作品按照输入规模设计 Direct、Medium、Large 三类执行路径，并针对每条路径选择不同的数据搬运和流水策略。最终版本已通过 CANNJudge 全部 15 个测试点，输出错误占比均为 0.00%。

## 实现思路

### 1. 近似计算

实现使用两类经过精度验证的 Erf 近似公式：

- Direct 小数据路径使用有理近似，优先保证小规模输入下的稳定性。
- Medium 和 Large 路径使用无除法 poly9 多项式近似，以 Horner 形式减少中间结果和向量指令数量。
- 所有路径在计算结束后将结果限制在 `[-1, 1]`，保证 Erf 值域和极端输入稳定性。

### 2. 分规模 Tiling

Host 侧根据 UB 容量和输入元素数量选择执行路径：

- **Direct**：输入可在单个 tile 内完成时使用单核静态 LocalTensor 路径；8 元素对齐的小输入使用 `DataCopy`，其余输入使用 `DataCopyPad` 处理尾块。
- **Medium**：输入规模在 1 到 4 个普通 tile 之间时进行多核切分；优先选择可以整除输入、且每核元素数为 8 的倍数的 `blockDim`。
- **Large**：更大输入采用多核、TQue 和双缓冲流水，提升持续吞吐。

### 3. Medium 路径负载均衡

Medium 是启动开销和并行收益共同影响最明显的区间。实现将其进一步划分为 low、mid、high 三档：

- low Medium 使用较保守的分核系数，避免过度切分；
- mid/high Medium 扩大合法 `blockDim` 搜索范围；
- 对齐切分命中时使用静态 UB 布局，否则回退到支持非对齐尾块的 Queue 路径。

### 4. UB 与搬运优化

- Direct 和 MediumAligned 路径使用静态 LocalTensor，减少不必要的 Queue 管理开销。
- MediumAligned 删除 poly9 公式不再需要的中间 buffer，降低 UB 占用。
- Large 路径使用输入、输出双缓冲，重叠搬运与计算。
- 对齐路径使用 `DataCopy`，非对齐路径使用 `DataCopyPad`，保证正确处理任意合法 shape。

## 代码结构

```text
code/
└── Erf/
    ├── CMakeLists.txt
    ├── op_host/
    │   ├── CMakeLists.txt
    │   └── erf.cpp
    └── op_kernel/
        ├── CMakeLists.txt
        ├── erf.cpp
        ├── erf_tiling.h
        └── tiling_key_erf.h
```

## 环境与验证

- 目标芯片：Ascend 910B
- CANN 版本：8.5
- 输入输出类型：float32
- 数据格式：ND
- CANNJudge 结果：15 / 15 Pass
- 输出错误占比：0.00%

## 主要优化总结

1. 针对 Direct、Medium、Large 设计独立 Kernel 路径。
2. Medium 使用分档多核策略与对齐 `blockDim` 搜索。
3. 对齐路径使用静态 LocalTensor 和 `DataCopy`。
4. 非对齐路径保留 `DataCopyPad`，兼顾任意 shape 正确性。
5. Medium/Large 使用 poly9 Horner 链，减少除法和中间 buffer。
6. Large 使用 TQue 双缓冲流水，保持大数据吞吐。

## 原创性说明

本目录中的算子实现与优化工作由团队成员在比赛期间完成。实现参考了赛题说明、CANN/AscendC 官方文档与公开 API 使用方式，不包含其他参赛队伍的代码。

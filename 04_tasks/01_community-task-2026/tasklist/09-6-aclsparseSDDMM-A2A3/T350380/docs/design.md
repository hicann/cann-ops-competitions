# aclsparseSDDMM 算子设计方案

团队：T350380

代码仓：[ops-sparse](https://gitcode.com/T350380/ops-sparse/tree/feat/sddmm-a2a3)

## 需求背景

面向 2026 年 9 月社区任务，在 Atlas A2/A3 上完善 SDDMM 算子。
计算稠密矩阵乘积在给定稀疏结构上的值，保持稀疏索引不变，仅更新 values。

计算公式：

```text
C = (alpha * op(X) * op(Y) + beta * C) ∘ spy(C)
```

## 需求分析

沿用 `aclsparseSDDMMBufferSize`、`aclsparseSDDMMPreprocess`、
`aclsparseSDDMM` 三阶段接口，补充 BSR 描述符、strided batch 和
Python/ATen NPU 适配。

| 稀疏格式 | X/Y 类型 | C values 类型 | computeType |
| --- | --- | --- | --- |
| CSR/BSR | float32 | float32 | float32 |
| CSR/BSR | complex64 | complex64 | complex64 |
| CSR/BSR | float16 | float16 / float32 | float32 |
| BSR | bfloat16 | bfloat16 / float32 | float32 |

索引使用 I32，支持 base 0/1。BSR 支持方块大小 2/4/8/16/32/64/128，
块内支持 ROW/COL 布局。稠密输入支持 ROW/COL、合法 leading dimension
及 NON_TRANSPOSE/TRANSPOSE；BSR 批处理支持 X/Y 单独或同时广播。

## 详细设计

### Host 侧

校验数据类型、维度、索引、布局、batch、标量和 workspace 参数，
计算所需 workspace 并调度 NPU 执行。预处理状态绑定 matC，
pattern 指针或内容变化后需重新预处理，workspace 由调用方管理。

### Kernel 侧

根据稀疏结构组织计算任务，在 NPU 上完成数据读取、乘加和结果写回。
结合输入规模进行并行处理和数据复用，减少重复搬运与调度开销。
不保存完整稠密输出，不使用 CPU fallback。

### Python/ATen

CSR 通过 `torch.sparse.sampled_addmm` / `aten::sparse_sampled_addmm`
接入公开三阶段接口，使用当前 NPU stream，并保持输出与 alias 语义。
BSR 适配调用原生 BSR 接口；不支持的输入组合显式报错。

## 支持环境与约束

- 目标硬件：Atlas A2 训练系列（910B3/910B4）、Atlas A3 系列。
- 软件：CANN 9.1.0 及后续配套版本、PyTorch 2.7+、torch_npu 26.0.0+。
- op 后的矩阵维度须匹配；不支持共轭转置。
- batchCount 范围为 1..65535，stride 以元素计并满足存储范围。
- 输入 X/Y 和 C 的稀疏结构只读，C values 原地更新。
- 同一 matC 的调用由调用方串行；独立 matC 使用独立 workspace。
- 溢出、非法参数和不支持的组合返回明确错误码。

## 可维可测分析

提供 C++ 接口测试及 Python/ATen 测试，覆盖 dtype、base、块大小、布局、
转置、批处理、空输入、边界、异常、workspace 和预处理生命周期。

精度按任务书与 CPU Golden 比较，complex64 的实部和虚部分别验收。
性能采用固定任务用例，至少预热 10 次、采样 30 次，记录 median/p90；
按相同 GPU 来源和调用范围比较，目标倍率不低于 0.25。
内存按任务书的额外峰值或 workspace 条件验证。

A2/A3 实现位于 `sparse/sddmm/arch22/`，与 A5 分支独立编译。
各硬件的功能、精度、性能和 Profiler 结果分别记录；尚未完成的验证不作为
已通过结果，本文件不代替最终自测报告。

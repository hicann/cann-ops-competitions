# aclblasCtrmm 算子设计文档

## 需求背景（required）

### 需求来源

2026 年 CANN 社区任务“aclblasCtrmm 算子开发（A2/A3）”。任务要求在 ops-blas 中提供与 cuBLAS CTRMM 参数顺序对应的句柄式接口，并在 Atlas A2/A3 上完成 COMPLEX64 三角矩阵与一般矩阵的乘法。

### 背景介绍

Ctrmm 是列主序、离席输出的复数三角矩阵乘法。输入 A 由 `uplo` 指定有效三角；`diag=UNIT` 时对角视为 1，不能读取存储中的对角值；`trans=C` 表示共轭转置。现有通用 BLAS 计算路径不能直接代替公共 `aclblasCtrmm` 句柄接口及其参数校验、stream 语义。

## 需求分析（required）

### 功能与接口

`aclblasCtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)`：

- LEFT：`C = alpha · op(A) · B`，A 阶数为 m；RIGHT：`C = alpha · B · op(A)`，A 阶数为 n。
- 数据类型 COMPLEX64，每个复数由两个 FP32 分量构成；A/B/C 为设备内存，alpha 为宿主指针；使用 handle 所绑定的 stream。
- 支持 LEFT/RIGHT、UPPER/LOWER、N/T/C、NON_UNIT/UNIT 全 24 种枚举组合，合法前导维、padding、零维和任务书规定的参数错误返回值。
- 输出在 C。严格控制三角读取范围，单位对角不读取 A 的相应存储；处理 INF/NAN 与大幅值 alpha 的参考顺序。

### 验收标准

随任务提供 1000 条精度用例、200 条 PF 性能用例。性能在 Atlas 800T A2 (910B3) 使用 msprof 统计单个 NPU kernel 的平均耗时，每个 case 先 5 次 warmup，再至少 11 次有效采样，与对应 `gpu_ms/0.8` 比较。任务书 §3.4 内存要求为“不涉及”，但自验报告记录可观测信息。

## 详细设计（required）

### Host 侧

公共声明放在 `include/cann_ops_blas.h`。入口位于 `blas/trmm/arch22/ctrmm_host.cpp`：先检查 handle、枚举、维度、alpha、前导维和必要地址；零维直接成功返回。通过公共 handle API 获取 stream，启动设备 kernel，不在公共入口同步 stream。Host 与设备共用 44 字节参数结构，编译期检查 ABI 大小。

调度基于矩阵尺寸和 alpha，属于数值与吞吐选择，不依据测试 case 名称。小矩阵与特殊幅值选向量/标量路径；典型较大矩阵以及满足条件的 16–64 阶方阵选拆分、Cube GEMM、合并路径。大 alpha 避免转换到可能改变参考顺序的 Cube 路径。

### Kernel 侧

1. 向量路径按 `side` 把输出分配给独立的列或行，按 `uplo` 和 `trans` 确定有效归约范围；`OP_C` 对虚部取反。处理前导维及 padding，输出仅写逻辑 C 区域。
2. Cube 路径把复数计算拆为实数 GEMM 组合，完成三角数据的拆分与输出合并；采用现有 arch22 的构建与启动框架。一个 API 调用可能发出多个 kernel，性能报告逐 kernel 保留阶段名称、均值和调用次数。
3. 对非有限数与高幅值输入采用 Netlib CTRMM 的分支及加法顺序参考路径，避免将普通点积的浮点结合律假设扩展到 INF/NAN。
4. 参数无效时不启动 kernel；负向测试检查返回值，异步执行通过绑定 stream 的同步点验证计算结果。

### 目录与兼容性

实现目录 `blas/trmm/arch22/`，测试目录 `test/trmm/ctrmm/arch22/`，公共接口位于 `include/cann_ops_blas.h`。产品 README 标注 Atlas A2/A3 系列（含 800I A2/A3）。本次实测设备为 Atlas 800T A2 (910B3)，A3 由社区验收继续验证。没有增加 batch、任意 stride 或外部 workspace 参数。

## 可维可测分析

- 随任务提供的 verify_accuracy.py 排除 TC_PF；正式精度为 1000 条，另有 1 条配置检查。
- 2026-10-03，Atlas 800T A2 (910B3)、CANN 9.1.0，以原 CSV seeded 输入运行并清除缓存输入环境变量，正式精度 1000/1000 及配置检查通过。
- 同一测试库的 PF 性能按每例 5 次预热、11 次有效样本统计，单 kernel 200/200 达标；全部 kernel 累计耗时的额外诊断为 190/200，10 条超限。
- 此前 1200/1200 严格精度的表述来自 local_inputs_r2 缓存，不能代表原 CSV seeded 输入，现撤回该结论。对 PF 追加严格精度检查时，17 条大矩阵仍失败；断言保留，不计作 PASS。
- 源码 PR #568 的 CLA 已通过，CI #3000 的 8 项编译、codecheck 和预提交检查已通过，Smoke 尚待结果。正式验收尚未提交。
- 规范整理后的新库已重新完成 1000 条精度和 200 条性能自验，且额外 24 组均匀分布、24 组正态分布用例全部通过。
- 任务书 §3.4 内存要求为“不涉及”，不将宿主 RSS 当作 NPU 峰值。

## 代码与自验材料

- 代码 PR：https://gitcode.com/cann/ops-blas/pull/568
- 代码分支：https://gitcode.com/xhf_A2_1/ops-blas/tree/feat/aclblas-ctrmm-a2
- 上述已验证版本：429be188dd256b844869e223ca45dca59f1a91c1。
- 上述已验证库 SHA256：17dc1f68d4a0d95ae198efa4fb616cf9340824cc27b12ae93f9fdbcdd8db0c7a。
- 原始日志、两份自验报告、原始 task_time 与审计 JSON：代码分支 task_submission/corrected_20261003/。

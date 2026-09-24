# 需求背景（required）

## 需求来源

来源：[9月社区任务-kv_cache_turbo_quant算子开发](https://www.hiascend.com/activities/task-center/details/6d3df013343845608677c85f3c58d4d6)及随任务提供的 `op.json`、`case.json`、`golden.py`。参与账号：MaMaer。目标仓库：[ops-transformer](https://gitcode.com/MaMaer/ops-transformer)，算子目录：`experimental/attention/kv_cache_turbo_quant`。

本文按社区设计模板编排，描述已实现的 Ascend 910B 路径。代码基线为 `357ab966258e948cc3f7f5e0b6097df7634c12d2`，新增算子位于 `codex/kv-cache-turbo-quant` 分支；自验证记录日期为 2026-09-23，完整证据随代码仓 `task_submission` 交付。

## 背景介绍

### 算子实现优化

标准 MHA/GQA 的 KV cache 随序列长度增长。对每个 K 或 V 向量，TurboQuant 通过正交旋转、MSE 标量量化和 1-bit QJL 残差编码减小存储量。本算子完成 Encode，K、V 分别调用。

### 现状分析

题包提供 PyTorch eager golden 和 5 个用例。仓库已有 TurboQuant MLA/shared-KV 算子，其维度、存储协议和 Attention 计算范围与本任务不同；本实现参考同仓算子的 OpDef、推形、tiling、CMake 和 ACLNN 接入方式，新增独立目录。

### 功能分析

默认 3-bit 主编码加 1-bit QJL，每个 128 维 head 输出 48 B 主码、16 B 符号码、两个 BF16 标量，共 68 B；相对原始 256 B BF16 向量约压缩 3.76 倍。这是 payload 比例，未包含共享矩阵、workspace 和内存分配对齐。

# 需求分析（required）

## 需求描述

| 项目 | 本实现契约 |
| --- | --- |
| 输入 KV | BF16，连续 ND，`[T,H,128]`，`T≥0`、`4≤H≤32` |
| 旋转矩阵、QJL 矩阵 | FP32，连续 ND，均为 `[128,128]` |
| 属性 | `mse_bits∈{2,3,4}`，默认 3 |
| 输出 | 主码 UINT8 `[T,H,16*b]`，QJL UINT8 `[T,H,16]`，norm/gamma BF16 `[T,H]` |
| 接口 | `aclnnKvCacheTurboQuantGetWorkspaceSize`、`aclnnKvCacheTurboQuant` |
| 硬件与软件 | Ascend 910B，CANN 9.1.0；实测芯片 910B3 |
| 性能目标 | 任务书要求相对功能等价 PyTorch eager 加速至少 10 倍；测量须注明计时范围 |
| 精度目标 | 四个编码输出与题包 golden 对照；任务书另列重构及内积统计质量指标 |

任务书文字还涉及矩阵 BF16、范数 FP16 和可变 QJL 维数；当前实现按题包 `op.json` 和全部 5 个 case 的 FP32/BF16、QJL=128 组合交付，扩展组合尚未覆盖。

## 需求拆解

1. 对齐矩阵方向、码本常量、阈值边界、零向量语义和打包位序。
2. 完成注册、推形、参数校验、分核、两次 FP32 Matmul 和向量量化。
3. 用题包全部 5 个用例、尾块和数值边界用例进行精度验证。
4. 分别保存性能采样、硬件插桩内存检测及可复现步骤。

# 详细设计（required）

## 算子分析

### 数学公式

将每个 token/head 展开为行向量 `x∈R^128`，旋转矩阵为 `R`，投影矩阵为 `S`：

```text
n = ||float32(x)||₂
u = x / max(n, 1e-30)
y = u @ R.T
idx[j] = count(y[j] > boundary[b][k])
c[j] = centroid[b][idx[j]]
r = y - c
rho = ||r||₂
gamma = n * rho
v = r / max(rho, 1e-30)
qjl[j] = (v @ S.T)[j] >= 0
```

码本沿用题包常量，boundary 为相邻 centroid 的 FP32 均值。量化比较为严格 `>`，等于阈值时取较低编号；QJL 比较为 `>=0`，精确零编码为 1。保留残差归一化，避免省略后改变浮点边界行为。

主码每 8 个索引组成 `w=Σ idx[j]<<(j*b)`，按低字节优先输出 b 字节；QJL 每 8 个符号位输出 1 字节，维号较小者在低位。norm/gamma 最后转 BF16。零向量仍执行量化与投影，其编码字节由码本和矩阵决定。

### 支持数据类型

| 参数 | dtype | 计算/存储 |
| --- | --- | --- |
| kv_vectors | BF16 | 读入后转 FP32 |
| rotation_matrix、qjl_matrix | FP32 | 两次 FP32 Matmul |
| quant_idx、quant_qjl | UINT8 | 低位优先打包 |
| quant_norm、quant_gamma | BF16 | FP32 计算后转换 |

### 支持形状

`T*H≤INT32_MAX`，矩阵维度固定为 128。T=0 为合法空操作。原始用例为：

| case | KV shape | mse_bits |
| --- | --- | ---: |
| gqa_decode_b1 | [1,8,128] | 3 |
| gqa_decode_b64 | [64,8,128] | 3 |
| gqa_prefill_t2048 | [2048,8,128] | 3 |
| gqa_mse2_t2048 | [2048,8,128] | 2 |
| gqa_mse4_t2048 | [2048,8,128] | 4 |

## 算子实现

### 实现方案

单次混合核 launch 完成两次投影和全部向量操作。每个 tile 包含 32 行，1 AIC 对应 1 AIV；两次 Matmul 串行复用同一对象，中间矩阵保留在片上缓冲区。

#### 3.2.1 host侧设计：

##### 1. 分核策略：

令 `N=T*H`、`B=32`，分核数为 `max(1,min(AIC核数,ceil(N/B)))`。核 c 依次处理起始行为 `c*B+k*核数*B` 的 tile。尾 tile 在片上补零，仅写回有效行。选择依据为几何形状、位宽和平台能力，无测试编号或矩阵数值分支。

##### 2. 数据分块和内存优化策略：

Host 校验输入输出 rank、dtype、形状、属性与行数溢出。Matmul 配置为 `M=32,N=128,K=128`，固定分块 `32×128×64`；A 为 VECOUT，B 为转置的 GM ND，C 为 VECIN，均为 FP32。Matmul 本地工作区为 32 KiB；对外 workspace 使用平台 `GetLibApiWorkSpaceSize()` 返回值，按 ACLNN 查询结果分配。

##### 3. tilingkey规划策略：

| tiling key | 含义 |
| ---: | --- |
| 2 | 2-bit 主编码 + 1-bit QJL |
| 3 | 3-bit 主编码 + 1-bit QJL |
| 4 | 4-bit 主编码 + 1-bit QJL |

tiling data 保存总行数、分核数和 Matmul tiling。位宽之外的尾行信息由行数推导。

#### 3.2.2 kernel侧设计：

1. 搬入 BF16 行块并转 FP32，计算范数和归一化。
2. 执行 `u @ R.T`，向量化阈值判断和 centroid 查表。
3. 计算残差范数、gamma 和归一化残差。
4. 执行 `v @ S.T`，比较符号并打包。
5. 写回两个字节流和 BF16 标量，只覆盖有效行。

2/3-bit 采用逐阈值向量比较与 Gather/移位打包；4-bit 采用 4 轮二分量化，索引转换到有符号 INT4 范围后使用硬件转换打包。偏移表由 CreateVecIndex、向量移位和转换批量生成，减少逐元素标量写入。满 tile 省去补零，4-bit 路径省去不使用的分组/交错偏移表，减少 10 KiB 显式 UB 分配。

Matmul 负责 AIC/AIV 协作和数据交换；运行函数退出、Matmul 与 TPipe 析构后恢复同步基址 `SetSyncBaseAddr(0)`。同步状态由普通精度测试及硬件插桩检测共同验证。

## 支持硬件

注册 `ascend910b`；实测 Ascend 910B3、CANN 9.1.0 正式版本。310P/A3 及其他平台尚无本实现的验证结论。测试直接调用 ACLNN，无 PTA 专用包装或整仓 Wheel 依赖。

## 算子约束限制

- 接口是 KV Encode，不包含分页写回、Attention 解码、MLA 或端到端 KV 管理。
- 调用方维护正交旋转与高斯投影矩阵；Host 只校验元数据，不扫描矩阵数值。
- 同形状、dtype、位宽的稠密矩阵采用同一计算路径，具体元素通常不显著改变性能；矩阵数值会影响码字和重构质量。
- 题包引用的两个原始矩阵文件缺失。开发测试使用 seed=20260923 的 float64 QR 正交矩阵转 FP32，以及 iid Gaussian FP32 QJL。该数据与原始任务矩阵身份不同，报告明确标注。
- 重构 MSE、内积 P95 的正式结论还需要统一解码器、query 分布及近零分母口径，编码对拍通过不替代这些统计指标。

# 可维可测分析

## 精度标准/性能标准

精度覆盖 42 个数据 case（含原始 5 个几何用例）和 1 个契约测试（16 组非法参数），43 项测试通过，四输出实际均精确一致。扩展用例包括 H=4/5/31/32、尾 tile、跨多 tile、空输入、零向量、零残差及阈值两侧。自测报告保留题包容差及实际 mismatch 数。

原始 5 个用例另通过 AscendOpTest 的数据生成、原始 golden 和比较流程。独立精度 20 个输出加 15 轮性能后的 60 个输出，共 80 项逐位一致。

| case | kernel 实测中位数（µs） | 任务书 PyTorch eager（µs） |
| --- | ---: | ---: |
| gqa_decode_b1 | 15.420 | 1491.676 |
| gqa_decode_b64 | 20.360 | 1552.784 |
| gqa_prefill_t2048 | 322.986 | 3855.422 |
| gqa_mse2_t2048 | 290.706 | 3607.963 |
| gqa_mse4_t2048 | 313.166 | 5286.502 |

kernel 数据来自 AscendOpTest + msprof application 模式的 `Task Duration(us)`。每 case 3 轮、每轮 121 次 launch，丢弃前 20 次，取 101 次中位数，再取三轮中位数。关闭 PMU 采集、保留 task-time。此时间不含 Host/API 查询、输入拷贝及任务间隔；任务书 eager 未给出相同计时细节，故本表不直接计算验收加速比。另附同环境 eager/ACLNN event 对比供评审。等价向量打包 reference 下，两个 decode 的 event 加速比分别为 5.10×、8.19×，尚未达到 10×；其余三个 prefill 为 16.78×、18.35×、20.84×。

内存采用相同源码的独立插桩构建和匹配版本 mssanitizer，检查实际扫描 kernel、退出码、错误、警告及输出精度；最终通过 16 个用例、32 次完整 kernel 扫描，零 error/warning。4-bit prefill 使用默认缓存，由两个独立进程分别执行完整扫描；此前 OOM 与小缓存告警的试验记录保留。全部原始用例及专项覆盖、日志和 XML 随 `task_submission` 交付。

## 兼容性分析

新增目录接入现有实验算子构建体系，生成标准 ACLNN 两段式接口。单算子构建覆盖 Kernel、Host/Tiling 和 op_api。未改变既有算子的 ABI。下游消费编码时须保持相同矩阵、码本、位序和 gamma 定义。

自验证材料按 `task_submission` 的步骤说明、精度/性能/内存 XLSX 和原始日志组织；代码分支及设计文档的线上 PR/合入状态以仓库实际记录为准。

参考：[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)、[社区目录规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)、题包 golden 与同仓 `gen_position_ids_from_mask` 的工程结构。

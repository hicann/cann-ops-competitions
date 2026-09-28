# kv_cache_turbo_quant 算子开发设计文档

- 设计者：`lzc12138`
- 任务：2026 年 9 月社区任务《kv_cache_turbo_quant 算子开发》
- 目标硬件：Atlas 800T A2
- 目标仓库：[`cann/ops-transformer`](https://gitcode.com/cann/ops-transformer)
- 设计文档仓库：[`cann/cann-ops-competitions`](https://gitcode.com/cann/cann-ops-competitions)
- 开发语言：Ascend C
- CANN：9.1.0+
- 任务书归档：`25e71624009c46969b4db01afca60aa8.zip`
- 归档 SHA-256：`f52765d7ad52758bdb33a8e0decec1c79d17700800169624a54f722b22576630`

本文严格依据任务书、`op.json`、`case.json` 和 `golden.py` 编写。本文只提交算子
设计，不宣称 Ascend C 实现、NPU 精度、性能或最终验收已经完成。任务书引用的
`data/rotation_matrix.bin` 和 `data/qjl_matrix.bin` 未包含在归档包中，矩阵资产和
正式校准结果必须由任务方补齐或书面确认生成方式后再形成验收证据。

## 需求背景（required）

### 需求来源

当前 vllm-ascend 的 KV cache 主要采用 INT8 静态量化或面向稀疏选择的哈希方案，
前者压缩率有限，后者不改变 KV cache 的密集存储。TurboQuant 通过随机正交旋转、
逐维 MSE 标量量化和 QJL 残差符号 sketch，在保持普通 MHA/GQA 语义的前提下减少
KV cache 的存储量。本任务面向在线编码算子，后续代码应在
`ops-transformer/experimental/attention` 中按仓库规范集成。

### 背景介绍

首版采用统一的 `mse_bits` 主编码和 1-bit QJL 残差编码。`mse_bits=3` 时，
`head_dim=128` 的主编码占 384 bit，QJL 占 128 bit，两个 BF16 标量各占 16 bit，
合计 544 bit，即 4.25 bit/channel；相对于 128 个 BF16 元素的 256 byte，理论
存储比约为 `256/68=3.76x`。该计算是格式账本，不是设备实测性能声明。

任务只覆盖标准 MHA/GQA 的 K/V 向量。DeepSeek MLA 的低秩 latent `cKV` 和独立
RoPE key `kR` 具有不同的注意力公式和缓存布局，不得将其拼接成普通 KV 向量后复用
本算子。混合 3.5-bit channel 分组也不在首版范围内。

## 需求分析（required）

### 需求描述

对输入的 BF16 KV 向量在线执行 TurboQuant 编码，输出主量化 bit-stream、QJL
残差 bit-stream、输入范数和原尺度残差范数。算子必须保持任务书定义的输入输出
顺序、dtype、shape 和 bit packing 规则，并针对合法的 token/head 规模提供可扩展
的 Ascend C 实现。

### 需求拆解

1. 支持连续 BF16 `kv_vectors[num_tokens, num_kv_heads, 128]`。
2. 支持固定 FP32 `rotation_matrix[128,128]` 和 `qjl_matrix[128,128]`。
3. 支持 `mse_bits` 为 2、3、4，默认值为 3。
4. 按任务书顺序输出 `quant_idx`、`quant_qjl`、`quant_norm`、`quant_gamma`。
5. 对五个正式 public case 分别完成 bit-exact 编码对比、精度自测和 3/5 计时。
6. 不以 CPU、Torch、参考 golden、其他后端或 peer workspace 作为运行时回退。

### 算子原型

```text
kv_cache_turbo_quant(
    Tensor kv_vectors,
    Tensor rotation_matrix,
    Tensor qjl_matrix,
    int mse_bits = 3
) -> (Tensor quant_idx, Tensor quant_qjl,
      Tensor quant_norm, Tensor quant_gamma)
```

| 参数/结果 | 类型与形状 | 说明 |
| --- | --- | --- |
| `kv_vectors` | BF16 `[T,H,128]` | 连续的 K 或 V 向量，`T>0`，`4<=H<=32` |
| `rotation_matrix` | FP32 `[128,128]` | 固定 Hadamard 或正交矩阵 `H` |
| `qjl_matrix` | FP32 `[128,128]` | 固定高斯残差投影矩阵 `S` |
| `mse_bits` | int，`2/3/4` | 主量化 bit 宽，默认 3 |
| `quant_idx` | uint8 `[T,H,ceil(128*mse_bits/8)]` | 主量化索引，小端 lane 打包 |
| `quant_qjl` | uint8 `[T,H,16]` | QJL 符号，小端 lane 打包 |
| `quant_norm` | BF16 `[T,H]` | 输入向量 L2 范数 |
| `quant_gamma` | BF16 `[T,H]` | 原尺度残差范数 |

输入和输出均为 ND 布局。矩阵在执行期间视为只读固定输入，不应在 kernel 内修改。
`num_tokens` 可以在契约范围内变化；五个 public case 是当前完整的正式验收集，
不是对实现支持范围的替代定义。

## 详细设计（required）

## 算子分析

### 数学公式

对每一个向量 `x`，先计算：

```text
n = ||x||_2
u = x / max(n, 1e-30)       (n=0 时 u 为全零)
y = u @ H^T
```

其中 `H=rotation_matrix`。任务书给出的 MSE 码本为每个 bit 宽固定的一组有序
centroid。相邻 centroid 的中点构成边界，Golden 的索引规则是对每个 `y` 分量统计
严格大于边界的数量：

```text
idx[j] = count(boundary[k] < y[j])
y_hat[j] = centroid[idx[j]]
r = y - y_hat
gamma = n * ||r||_2
```

残差方向使用零范数保护：

```text
r_unit = r / max(||r||_2, 1e-30)  (残差范数为零时全零)
z = r_unit @ S^T
qjl_bit[j] = 1 if z[j] >= 0 else 0
```

这里 `S=qjl_matrix`。`>=0` 的零值归类必须保持与 Golden 一致，不能改成严格
大于零或依赖未定义的符号函数。

### 支持数据类型和形状

- `kv_vectors`：仅 BF16、rank 3、最后一维固定 128、连续 ND。
- `rotation_matrix`：仅 FP32、rank 2、形状 `[128,128]`、连续 ND。
- `qjl_matrix`：仅 FP32、rank 2、形状 `[128,128]`、连续 ND。
- `num_tokens`：正数；`num_kv_heads`：`[4,32]`，覆盖 MHA/GQA 合法范围。
- `mse_bits`：仅 2、3、4；不支持混合 bit 宽或首版以外的 3.5-bit 分组。
- MLA latent cache、非连续输入、任意改变 `head_dim` 的扩展均不在本任务接口内。

### 输出存储格式

每 8 个逻辑 lane 组成一个小端 bit word。对 `bit_width=b` 的主编码，word 按低
位到高位依次放置 lane 0 到 lane 7，再按低字节优先写出 `b` 个 uint8；当最后一
组不足 8 个 lane 时，高位 lane 置零，但输出 shape 仍按 `ceil(128*b/8)` 计算。
QJL 为 1-bit，128 个符号固定占 16 byte。

| `mse_bits` | 主编码字节/head | QJL 字节/head | 两个 BF16 标量 | 合计 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 32 | 16 | 4 | 52 |
| 3 | 48 | 16 | 4 | 68 |
| 4 | 64 | 16 | 4 | 84 |

输出必须保持 `[T,H,packed_bytes]` 的连续布局；norm 和 gamma 按 `[T,H]` 写出。输出
分配和 stride 由外部调用方提供的标准 aclnn/ATen 接口管理，算子不保存跨调用的
全局状态。

## 算子实现

### Host 侧设计

Host 入口负责完整的契约检查，并把合法的运行时信息写入 tiling data：

1. 检查四个参数的 dtype、rank、shape、连续性和设备一致性；拒绝错误 dtype、
   非法 `mse_bits`、`T<=0`、head 数越界以及矩阵非 128x128 的组合。
2. 校验输出 shape 与 `T/H/mse_bits` 推导结果一致，检查输出地址和 workspace
   大小满足 ACLNN 生命周期约定；不在 Host 侧调用 Torch 或 CPU golden。
3. 根据 `T*H` 的 vector 数、128 的固定 channel 数、平台 core 数及可用 UB 计算
   `core_count`、每核 vector 范围、tile vector 数、尾块长度和 workspace 大小。
   core 之间使用商/余数均分，余数分配给前面的 core，确保每个 vector 只被一个
   core 处理。
4. 用 `mse_bits` 生成明确的 `TilingKey`（2/3/4 三个固定分支），让 kernel 的
   pack 宽度和 centroid 表成为编译期常量；`T`、`H` 和尾块保持 runtime-visible，
   不通过 public case 形状生成专用分支。
5. 记录矩阵搬运策略和 tile 参数。固定矩阵可按单次 launch 复用，但是否常驻 L1/UB
   必须以目标 A2 编译和 profiler 结果确认，设计文档不预先假定具体缓存容量。

当 `T*H` 小于可用 core 数时只启用必要的 core，避免空核 launch；当 `T*H` 较大时
   采用多 tile 循环。每个 tile 的最后一个 vector 和最后一组 lane 必须有 mask，不能
   依赖调用者把长度补齐。

### Kernel 侧设计

Kernel 采用 `Init -> CopyIn -> Compute -> CopyOut` 的流水结构。单个 vector 的逻辑
顺序如下，多个 vector 在不同 core 上并行：

1. 从 GM 搬入一组 BF16 `x`，在 UB 中转成 FP32 参与归约和矩阵乘。
2. 计算 128 维平方和、归约得到 `n`，使用 `max(n,1e-30)` 完成归一化；零向量
   直接得到全零 `u`，避免除零和 NaN。
3. 以向量矩阵乘方式计算 `u @ H^T`。矩阵乘的累加类型使用 FP32，结果保留 FP32
   到 centroid 比较、残差和 QJL 投影完成。
4. 按编译期 `TilingKey` 选择 2/3/4-bit centroid 表，执行严格大于边界的比较，
   生成 `idx`；再查表得到 `y_hat` 并计算 `r` 和 `||r||_2`。
5. 计算 `gamma=n*||r||_2`，对 `r/max(||r||_2,1e-30)` 与 `S^T` 做投影，
   以 `>=0` 产生 QJL bit。
6. 在 UB 中完成主编码和 QJL 的 lane pack，将 uint8 流写回 GM；norm/gamma 在
   最后按 BF16 舍入规则写回对应 `[T,H]` 位置。

矩阵和向量在 UB 中的复用、double buffer 数量和 tile 大小必须通过编译器资源报告
   与 A2 profiler 共同确定。若资源不足，优先减小 tile 或降低矩阵驻留级别，不得
   改变输出格式、量化边界或引入 CPU 回退。

### Tiling 与调度策略

| 维度 | 调度依据 | 设计约束 |
| --- | --- | --- |
| core 分配 | `T*H` vector 数 | 商/余数均分，余数落在前部 core |
| tile 大小 | UB 可用容量、矩阵复用、double buffer | 保留输入/旋转结果/残差/输出 pack 的空间 |
| bit 分支 | `mse_bits` | 仅 2/3/4 三个 TilingKey |
| 尾块 | runtime `valid_vectors` 与 `valid_lanes` | 所有 GM load/store 使用 mask |
| 矩阵布局 | `[128,128]` 连续 FP32 | 以实际向量矩阵乘 API 支持的布局为准 |

decode `T=1` 的小工作量不应强行启动全部 core；prefill `T=2048` 则通过多 tile 和
   多 core 提高占用。`H` 在 `[4,32]` 内变化时只改变 vector 数，不改变 128 channel
   的算法和输出布局。

### 流水并行与内存组织

CopyIn、Compute、CopyOut 以 tile 为单位重叠。GM 只保存输入、固定矩阵和四个输出；
UB 保存当前 tile 的 BF16/FP32 转换缓冲、旋转结果、主重构、残差、投影和 pack 临时
   区域。每个临时区都按实际 dtype 和 tile 长度计算，避免用假定的固定字节数覆盖尾块。

矩阵是否分片、复制到每个 core，或通过可复用 workspace 共享，取决于 A2 平台的
   编译器和运行时限制；共享方案必须是本次 launch 内可证明的只读数据，禁止使用以
   指针为 key 的进程全局缓存。workspace 不得承载跨调用语义状态。

## 支持硬件

| 支持的芯片版本 | 状态 |
| --- | --- |
| Atlas 800T A2 / Ascend 910B | 任务书目标，待目标环境编译与设备自测 |

本设计不把其他 SoC 的可编译性当作 A2 验收证据。CANN 版本要求为 9.1.0 或更高，
最终提交应记录实际编译版本、设备标识、驱动版本和执行命令。

## 算子约束限制

1. 仅支持标准 MHA/GQA KV cache，不支持 MLA latent `cKV/kR`。
2. `head_dim` 固定为 128，`num_kv_heads` 固定在 4 到 32，`num_tokens` 必须为正。
3. 输入矩阵为固定 FP32 `[128,128]`；其具体二进制资产必须与验收包一致。
4. 仅支持 `mse_bits=2/3/4`，QJL 固定为 1 bit/channel；不支持混合 3.5-bit 模式。
5. 输入必须连续且位于目标 NPU；不支持通过 CPU、Torch 或其他后端绕行。

## 可维可测分析

### 精度标准

| 验收标准 | 判定口径 | 来源 |
| --- | --- | --- |
| 主编码输出 | `quant_idx` uint8 逐元素精确匹配 Golden | `golden.py` / task manifest |
| QJL 输出 | `quant_qjl` uint8 逐元素精确匹配 Golden | `golden.py` / task manifest |
| 两个范数 | BF16 输出逐元素精确匹配 Golden | `golden.py` / task manifest |
| 向量重构 | `MSE / ||x||_2^2 < 0.05` | 任务书 §3.2 |
| 内积估计 | 相对误差 P95 `< 0.1` | 任务书 §3.2 |

前 3 项是算子输出对照，后 2 项需要独立反量化/查询向量测试。不能用输出 checker
   通过替代任务书的完整重构和内积验收。零范数、边界 centroid、最后不足 8 lane 的
   pack 组必须各自有回归用例。

### 性能标准

性能参考为功能等价的 PyTorch eager TurboQuant reference。五个正式 case 每个都要
   达到相对加速 `>=10x`，且算术平均加速也要 `>=10x`。计时统一采用 warmup=3、
   repeat=5，报告中记录 median 和 P90，并保留每个 case 的原始日志。

| case | shape | bits | 任务书参考 wall time (us) |
| --- | --- | ---: | ---: |
| `gqa_decode_b1` | `[1,8,128]` | 3 | 1491.676 |
| `gqa_decode_b64` | `[64,8,128]` | 3 | 1552.784 |
| `gqa_prefill_t2048` | `[2048,8,128]` | 3 | 3855.422 |
| `gqa_mse2_t2048` | `[2048,8,128]` | 2 | 3607.963 |
| `gqa_mse4_t2048` | `[2048,8,128]` | 4 | 5286.502 |

表中的参考数值仅是任务书提供的 provenance，不能直接视为本地 baseline。正式性能
   结论必须绑定 A2 设备、固定矩阵、同一输入生成方式和 evaluator 生成的 3/5 记录。

### 当前证据边界

本设计 PR 不包含 Ascend C 源码、自测报告、截图或性能日志。原始 ZIP 只有
`case.json`、`golden.py`、`op.json` 和任务书，缺少任务书引用的两个矩阵二进制，
因此当前不能声称：

- 五个 case 已在 A2 上完成精度或性能校准；
- 已达到 `10x` 性能目标；
- 已完成重构 MSE 或内积 P95 验收；
- 已完成 `ops-transformer` 代码 PR 或 IT 验收。

后续自测必须先固定矩阵资产和 hash，再按五个 case 全量执行，保存精度报告、
   warmup/repeat 计时、设备信息、编译包 hash 和可复现 README。任何减少 case、替换
   矩阵或只报告单个 shape 的结果都不构成完整验收。

## 兼容性分析

这是面向标准 MHA/GQA KV cache 的新算子，不改变既有算子 ABI。后续在
`ops-transformer/experimental/attention` 集成时，需要在 Host/API 层明确拒绝 MLA、
非连续输入和非法 bit 宽；不支持组合应返回清晰错误，不转换到 CPU、Torch 或其他
   算子实现。若未来需要 MLA 或混合 3.5-bit 编码，应另立任务并重新定义输入、反量化
   公式、存储格式和验收集，不能把本设计的 shape guard 放宽后宣称兼容。

## 验收交付路径

1. 本文件作为第一阶段设计文档，向 `cann/cann-ops-competitions` 对应 tasklist
   目录提交 PR，评审通过后再进入实现阶段。
2. 第二阶段在 `cann/ops-transformer` 的 `experimental/attention` 提交 Ascend C
   源码、测试、README 和自测报告；代码 PR 必须引用本设计和任务书。
3. 实现阶段保留五个 task-book public case，不得以私有或选定 case 替代正式集合。
4. 任务书要求的个人仓 `task_submission`、精度/性能报告和设备截图属于后续验收材料，
   不在本设计 PR 中伪造或预提交。

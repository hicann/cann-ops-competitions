# kv_cache_turbo_quant 算子设计文档

# 需求背景（required）

## 需求来源

9 月社区任务：kv_cache_turbo_quant 算子开发。大模型自回归推理需要长期保存历史 token 的 Key/Value，随着上下文长度增加，KV Cache 的显存占用与访存带宽快速增长，成为长序列推理的关键瓶颈。当前 vllm-ascend 中 KV cache 的压缩方案为 INT8 静态量化（C8，压缩率仅 2x）与 LSH 哈希（KVComp，用于稀疏选择、不压缩存储）。参考 TurboQuant 论文（arXiv:2504.19874），需要以 Ascend C 设计面向标准 MHA/GQA KV cache 的在线向量量化压缩算子（aclnn 工程化模式），在不成为 prefill 瓶颈的前提下把 KV cache 物理存储压缩到 4.25 bit/channel（head_dim=128、mse_bits=3，约 3.76x vs BF16）。

## 背景介绍

### TurboQuant 论文方法

TurboQuant 采用两阶段编码思路：先对旋转后的单位向量执行低比特 MSE 标量量化，再对第一阶段残差执行 1-bit QJL（Quantized Johnson-Lindenstrauss）编码。随机旋转把单位向量的坐标分布随机化，使各坐标近似服从高维球面坐标分布，从而可以用预先计算的 Lloyd-Max 标量码本独立量化各坐标；单纯面向 MSE 优化的低比特量化会给内积估计带来偏差，对 MSE 量化残差再施加 1-bit 随机投影符号编码后，内积估计在随机矩阵意义下无偏。

1. **MSE 主量化器**：随机正交旋转（任务接口为通用 FP32 矩阵 H）+ 逐维 `mse_bits` 标量量化（Lloyd-Max 质心，任务 golden 内置固定码本）；
2. **残差 QJL 修正**：对单位残差做随机投影（固定矩阵 S）+ 1-bit 符号量化。

对每个 `(token, kv_head)` 向量 `x ∈ R^128`：

```python
x_norm = ||x||_2                       # 向量范数
u = x / x_norm                         # 归一化到单位球面
y = H @ u                              # 正交旋转（golden 语义）
idx = scalar_quantize(y, mse_bits)     # 逐维主量化并 bit 打包
y_hat = dequant(idx)                   # 主重构（旋转空间）
r = y - y_hat                          # 量化残差
gamma = x_norm * ||r||_2               # 原尺度残差范数
qjl = sign(S @ (r / ||r||_2))          # QJL：随机投影 + 符号，打包 uint8
```

输出 4 个张量：`quant_idx`（主编码，bit-packed）、`quant_qjl`（残差符号，bit-packed）、`quant_norm`（bf16）、`quant_gamma`（bf16）。

论文完整的 `TurboQuant_prod` 还包含按异常通道/普通通道拆分的混合精度 3.5-bit 方案，任务书已明确首版不实现；MLA 也不在首版范围。

### KvCacheTurboQuant 算子实现信息

| 模块 | 文件或目录 | 作用 |
| --- | --- | --- |
| 算子定义 | `op_host/kv_cache_turbo_quant_def.cpp` | 输入、输出、属性及芯片注册（ascend910b） |
| Shape 推导 | `op_host/kv_cache_turbo_quant_infershape.cpp` | 四个输出的 shape/dtype 推导与参数校验 |
| Tiling | `op_host/kv_cache_turbo_quant_tiling.{h,cpp}` | 分核、blockDim、workspace 与 cube tiling 下发 |
| Kernel | `op_kernel/kv_cache_turbo_quant.cpp`、`kernel_kv_cache_turbo_quant.h`、`kernel_kv_cache_turbo_quant_mix.h` | 纯 vector 路径与 AIC/AIV 混合 cube 路径 |
| 调用样例/自测 | `examples/test_aclnn_kv_cache_turbo_quant.cpp` | 5 个任务 case 的精度比对与性能测量 |
| Host UT | `tests/ut/op_host/` | infershape/tiling 11 项单测 |

### KvCacheTurboQuant 现状分析

任务包只提供算子定义、用例描述和 torch_npu 版 golden 参考，没有可复用的 CANN 算子实现。ops-transformer 仓内已有的 TurboQuant 相关实验算子面向 MLA 或 4-bit latent 等不同协议，码本、数据布局和输出语义均不能直接复用。

本算子的工程难点集中在三点：

- 两次 128×128 FP32 稠密矩阵-向量乘（旋转 + QJL 投影）的吞吐：T=2048 时共 16384 个向量，纯 vector matvec 无法在 385 us（基线的 1/10）内完成；
- 整数编码输出必须逐 bit 复现 golden 的边界规则（严格 `>` / `>=` 不对称）和 bit packing 布局；
- `T=1` 低并行度与 `T=2048` 大吞吐场景都要达到至少 10 倍 PyTorch eager 基线。

### 存储格式（head_dim=128，mse_bits=3，qjl_dim=128）

| 组件   | 摊销 bit/channel | 总 bit | 说明                   |
| ------ | ---------------: | -----: | ---------------------- |
| idx    |            3.000 |    384 | 48 bytes（bit-packed） |
| qjl    |            1.000 |    128 | 16 bytes（bit-packed） |
| x_norm |            0.125 |     16 | 2 bytes（bf16）        |
| gamma  |            0.125 |     16 | 2 bytes（bf16）        |
| 合计   |            4.250 |    544 | 68B/head vs 原始 256B  |

压缩比 256/68 ≈ **3.76x**。

# 需求分析（required）

## 需求描述

以 Ascend C 实现 `KvCacheTurboQuant` 算子（aclnn 工程化）：

- 输入：`kv_vectors` bf16 `[num_tokens, num_kv_heads, 128]`；`rotation_matrix` fp32 `[128,128]`；`qjl_matrix` fp32 `[qjl_dim,128]`
- 属性：`mse_bits` int，支持 2/3/4，默认 3
- 输出：`quant_idx` uint8 `[T,H,16*mse_bits]`；`quant_qjl` uint8 `[T,H,qjl_dim/8]`；`quant_norm`/`quant_gamma` bf16 `[T,H]`

## 需求拆解

| 编号 | 需求 | 设计落点 | 验证入口 |
| --- | --- | --- | --- |
| RQ-01 | 三输入、四输出及 `mse_bits` 属性，动态 shape | Host 算子定义 + InferShape/InferDataType | host UT、aclnn 样例 |
| RQ-02 | bit 打包格式与 golden 严格一致：Lloyd-Max 质心/边界 fp32 位级一致；每 8 维一组、lane 占 `lane*bit_width` 起始 bit、多字节小端 | Kernel 量化与打包常量、打包数学 | 设备输出逐字节比对 |
| RQ-03 | 边界规则：MSE 用严格 `>`，QJL 用 `>=` | Kernel 溢出指示器比较语义 | 边界值单测数据 |
| RQ-04 | 精度：量化-反量化相对重构误差 `MSE/||x||^2 < 0.05`；内积估计相对误差 p95 < 0.1 | 全程 fp32 累加、fp16 高低位拆分矩阵喂 cube | 自测报告 |
| RQ-05 | 性能：5 个任务 case 相对 PyTorch eager 基线（1491~5287 us）加速 ≥10× | 纯 vector 路径（小形状）+ AIC/AIV 混合 cube 路径（大形状） | 自测报告 |
| RQ-06 | 工程：遵循 ops-transformer 仓目录与代码规范，aclnn 文档、调用样例、host UT | 仓内目录布局 | 编译、CI |

## 接口与参数分析

`op.json` 声明 BF16 `kv_vectors`、两份 FP32 矩阵与 BF16 标量输出；`qjl_dim` 由矩阵首维表达（正式用例为 128）；`mse_bits` 是主量化位宽，QJL 始终额外占 1 bit/通道。输入合法性（head_dim=128、kv_heads∈[4,32]、qjl_dim≤128 且为 8 的倍数、mse_bits∈{2,3,4}）全部由 host 侧校验拦截。

# 详细设计（required）

## 算子分析

### 数学公式

将 `kv_vectors` 前两维展平，记第 `i` 个向量为 `x ∈ R^128`，旋转矩阵 `H ∈ R^{128×128}`，QJL 矩阵 `S ∈ R^{qjl_dim×128}`：

1. **范数与单位化**：`n = ||x||_2`，`u = x / max(n, 1e-30)`（`n=0` 时 `u=0`，与 golden 的 `where` 语义一致）。
2. **旋转**：`y = H @ u`（golden 语义；矩阵按行主序读取，kernel 不假设其具有 Hadamard 等结构）。
3. **MSE 标量量化**：位宽 `b ∈ {2,3,4}`，有序码本 `C_b`，边界 `t_j = (c_j + c_{j+1}) / 2`。索引 `q_k = Σ_j 1(y_k > t_j)`（严格大于，边界相等归入较小索引）。重构 `ŷ_k = c_{q_k}`，残差 `r = y - ŷ`。
4. **残差范数与 gamma**：`ρ = ||r||_2`，`γ = n·ρ`；单位残差 `v = r / max(ρ, 1e-30)`。
5. **QJL**：`p = S @ v`，`z_j = 1(p_j >= 0)`（非负为 1；与 MSE 的严格 `>` 不可混用）。
6. **bit packing**：每连续 8 个逻辑值一组，组内字 `w = ∨_{l=0..7} q_{8g+l} << (l·b)`，按小端输出 `b` 字节；QJL 为 `b=1` 的特例。

### 支持数据类型

kv_vectors: bfloat16；rotation_matrix / qjl_matrix: float32；quant_idx / quant_qjl: uint8；quant_norm / quant_gamma: bfloat16。Kernel 内部统一 fp32 累加（BF16 输入搬入后立即 Cast 为 fp32；两条 cube 路径采用 fp16 高低位拆分逼近 fp32 精度）。

### 支持形状

动态 shape：`kv_vectors=[T,H,128]`，`T ≥ 1`、`4 ≤ H ≤ 32`；`rotation_matrix=[128,128]`；`qjl_matrix=[qjl_dim,128]`，`0 < qjl_dim ≤ 128` 且为 8 的倍数。性能用例覆盖 `T=1/64/2048`、`H=8`、`mse_bits=2/3/4`。

## 算子实现

### 3.2.1 host 侧设计

- **InferShape/InferDataType**：校验 3D 输入、head_dim=128、kv_heads∈[4,32]、rotation_matrix=[128,128]、qjl_matrix 末维 128 且首维 ≤128、为 8 的倍数、mse_bits∈{2,3,4}；按 `16*mse_bits`、`qjl_dim/8` 推导输出末维，输出 dtype 固定（uint8 / bfloat16）。
- **Tiling 策略**：`(token, kv_head)` 向量两两独立，`totalVec = T*H` 均分到各核（前 rem 核各多分 1 个向量；MIX 路径再按 AIC 三元组分组、组内两个 AIV 均分）。`blockDim = min(totalVec, coreNum)`，MIX 路径经 `CalcTschBlockDim(2*usedAic, usedAic, 2*usedAic)` 配置 1AIC+2AIV 三元组。
- **路径选择（tiling 内）**：`totalVec ≥ 1024 且 qjl_dim = 128` 走 AIC/AIV 混合 cube 路径（prefill 大形状）；其余走纯 vector 路径（decode 等小形状）。tilingKey 固定 0，mse_bits/qjl_dim/核数以 tiling 数据下发，cube 路径额外下发一份 `MatmulApiTiling`（fp16 ND，64×128×128）。
- **workspace**：MIX 路径 = `GetLibApiWorkSpaceSize()`（910B 上 16MB，MIX 运行时 KFC 通道保留前缀，device 侧以 `GetUserWorkspace` 跳过）+ 用户区：16KB 跨核 spin-flag 区 + Hh/Hl/Sh/Sl 四份 fp16 拆分矩阵（各 32KB）+ z1/z2 fp32（各 totalVec×512B）+ rh/rl fp16（各 totalVec×256B）+ p1..p4 fp32（4×totalVec×512B）+ xh fp16（totalVec×256B）。纯 vector 路径 = totalVec×512B（阶段间 fp32 `r_hat` 中转）。

### 3.2.2 kernel 侧设计

#### 路径 A：纯 vector kernel（decode 等小形状）

单核内两阶段串行，不使用 cube：

**阶段 1（H 驻留 UB，64KB）**，按 32 向量/tile 流水：
- Pass A：tile 搬入 bf16 x；逐向量 `Cast→Mul→ReduceSum` 得 `||x||^2`，并做**未归一化旋转** `z = H x`（利用线性性把归一化推迟为标量乘，避免逐向量除法与逐向量标量同步）；
- tile 级 `Sqrt` 后一次性 V_S 同步读出范数，计算 `inv = 1/max(norm,1e-30)`；
- Pass B：`y = z*inv`；**MSE 量化**用“溢出指示器”累加——`ind = min(max(y-b,0)*2^100*2^100, 1)`，任意严格正数映射到精确的 1.0（含 denormal），与 golden `(y > b)` 完全一致；同步用 `Axpy` 以 `delta_k = c_k - c_{k-1}` 累加 `ŷ`，免去 gather；残差与 `||r||^2` 由 `Sub`/`ReduceSum` 得到；
- tile 级 `Sqrt/Mul` 得 `gamma`，`Cast RINT` 输出 bf16 norm/gamma；
- **bit 打包（fp32 域精确整数）**：`idx * 2^(bits*(d mod 8))` 后按 8 lane 分组归约，组和恰为打包字（4-bit 时拆 lo/hi 两个半字保证 <2^24 fp32 精确），tile 级一次 V_S 同步后标量拆字节入 staging；
- Pass C：`r_hat = r*inv_r` 整块 DMA 到 GM workspace；norm/gamma/idx 落盘（bulk `DataCopy` + 尾块标量 `SetValue`）。

**阶段 2（S 驻留 UB，64KB）**：换载 S；逐 tile 读回 `r_hat`，同一 MatVec 得 `p = S r_hat`；符号用溢出指示器实现 `1 - (p<0)`（与 golden `(p>=0)` 在 `p=±0` 处一致）；打包落盘。

**MatVec 核心**：`y[j] = Σ_d M[j][d]*v[d]`。v 复制 8 份后单条 `Mul(mask=64, repeat=16)` 覆盖 8 行；16 条 Mul 完成 `[128,128]` 元素积，再 `ReduceSum` 分别对低/高 64 宽半行求和、一条 `Add` 合并——整条 matvec 仅 ~19 条向量指令，行主序无转置。

#### 路径 B：AIC/AIV 混合 cube kernel（prefill 大形状，性能主战场）

910B 为 24 个三元组（1 AIC + 2 AIV）拓扑，核型 `KERNEL_TYPE_MIX_AIC_1_2`。v6 起采用 **128 行 chunk 流水**（chunk c 轮询分配给组 c%24），职责划分：

- **AIV（48 核）**：pre-pass（x 一次读入：行范数 + fp16 xh 上传 + norm 落盘）→ quant（y=z·inv、MSE 量化、残差范数、gamma、r̂ fp16 高低位拆分上传、idx 打包）→ sign（p 三平面 fp32 求和、QJL 符号与打包）；
- **AIC（24 核）**：仅做 cube matmul——`z = (Hh+Hl) @ xh`（两发 Mmad L0C 累加、单 fp32 平面输出，m=64 大 tile）与 `p1=rh@Sh / p3=rh@Sl / p2=rl@Sh`（三个独立 fp16 平面；**p 不可用 L0C 累加**——修正项精度丢失会让 QJL 符号成片翻转，已两次实证）。

**跨核协同（chunk 粒度流水）**：GM spin-flag 按 chunk 四轮握手（XH→Z→R→P，每组每 chunk 8 个 64B 分散槽位，等待方读后自清零，同 executor+workspace 复用时下一发启动天然干净）；中间量 z/r̂/p 经 GM workspace 传递；**fixpipe（cube-DMA ATOMIC 域）写在 `SyncPipe<FIX_S>`+旗标序后对 AIV 的 MTE2 读天然可见，无需逐行 dcci 冲刷**（消融实测精度逐字节不变、省 35us）。AIV 侧旗标收发一律走 MTE2-in/改/MTE3-out 的 DMA 行更新（AIV 的 dcci 会毒杀其后续 GM 写，dav_m200 write-poison 坑）。

**低层 cube 流程（自研，非 Matmul 高阶 API）**：GM ND → L1 NZ（`Nd2NzParams`，尾块按有效行数读取不越界）→ `LoadData3DParamsV2` 装 A、`LoadData2DParams`（`ifTranspose=false` 实现 `C=A@B^T`）装 B → `Mmad`（`unitFlag=0b11`）→ `Fixpipe`（fp32 或 `F322F16` fp16）直写 GM；L0A 4 深轮转（单调计数器）+ 每 4 次装载 `PipeBarrier<PIPE_M>` 抑制 load-ahead 竞争；L0C 双缓冲 + FIX_M 延迟一拍，fixpipe 与下一 tile 的 Mmad 重叠。

**精度设计**：质心/边界与 golden 位级一致（fp32 字面量，边界按 `(c[i]+c[i+1])*0.5f` fp32 重算）；范数/残差范数全程 fp32，bf16 输出 RNE 与 golden 一致；H/S fp16 高低位拆分使 cube 投影相对误差 ~2^-22；p 的 fp16 输出舍入是相对误差（~2^-12|p|）**不可能翻转符号**（只有当 |p| 小到 fp16 次正规以下才归零，概率 ~1e-7/元素）。实测失配：t2048 idx/qjl/norm 全 0（gamma 1/16384）；mse2 idx 1、qjl 8；mse4 idx 3、qjl 26（均低于 case.json 阈值约两个数量级）。

**idx 位打包**：idx×2^(bits·lane) 后 `BlockReduceSum`（vcgadd，每 repeat 64 元素出 8 个 8-lane 组和）一次得 16 个组词，fp32 精确整数（3-bit 下 24bit ≤ fp32 mantissa；4-bit 拆 lo/hi 半字），再 `__ubuf__` 裸指针按字节拆分——替代 128-rep WholeReduceSum，AIV 打包耗时约省 4x。

**UB 预算**（路径 A，TILE_VEC=32）：H/S 64K + M 64K + zTile 16K + rTile 16K + b8 4K + xTile 8K + 其余小缓冲 ≈ 177KB < 192KB。路径 B 的 AIV 侧约 177KB VECCALC 缓冲（含 split 暂存复用为 per-row norm/inv 持久区），AIC 侧 L1A/L0A 各 64KB（4×16KB 轮转）、L1B/L0B 各 64KB（Hh/Hl 或 Sh/Sl 常驻）、L0C 64KB（2×32KB 双缓冲）。

**性能结构分析**：prefill 中间量 GM 往返约 72MB（xh/z/r̂/p 四平面），本机纯流式拷贝实测聚合带宽 ~1.6TB/s（48 核），流水重叠的目标是把核时间压向 ~60us 量级；decode 纯 vector 路径核时间 ~3-30us，均为启动开销主导。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（Ascend 910B） | √ |

## 算子约束限制

- 仅标准 MHA/GQA KV cache，不支持 MLA latent cache；
- head_dim 固定 128；qjl_dim ∈ (0,128] 且为 8 的倍数；mse_bits ∈ {2,3,4}；不支持论文混合 3.5-bit 模式；
- 输入 NaN/Inf 的数值语义未在任务书定义，首版不承诺。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 与任务 golden（torch_npu）比对：quant_idx 失配率 ≤1e-5、quant_qjl 失配率 ≤1.1e-5（case.json err_threshold）；重构相对 MSE < 0.05；内积估计相对误差 p95 < 0.1 | 任务书 3.2/3.5 |
| 性能标准 | 5 个任务 case 相对 PyTorch eager 基线加速 ≥10× | 任务书 3.3/3.5 |
| 内存标准 | 任务书标注“不涉及”；MIX 路径 workspace ≈ 16MB(运行时保留) + 2KB/向量(按 128 行 chunk 对齐) | 任务书 3.4 |

## 兼容性分析

新算子，不涉及兼容性分析。

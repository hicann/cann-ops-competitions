# KvCacheTurboQuant 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 9月社区任务-kv_cache_turbo_quant算子开发 |
| 提交人/团队 | gcw_DlLcJbB1 |
| 版本 | v0.1，2026-09-23，设计评审稿 |
| 目标代码目录 | ops-transformer/experimental/attention/kv_cache_turbo_quant |
| 实现依据 | 本地开发提交 6c0ba1217，分支 feat/kv-cache-turbo-quant |
| 官方模板 | [design_template.md](../../../../resources/design_template.md) |
| 提交规范 | [2026 社区任务说明](../../../../README.md) |

本文按官方模板描述设计方案，并明确已有实现、阶段结果及后续工作。开发版已编译并在 NPU 上执行，尚未完成任务书全部验收。实现细节以保存的开发提交为依据，不假定代码仓当前检出的 master 已包含该算子。

# 需求背景（required）

## 需求来源

需求来自本地《kv_cache_turbo_quant 算子开发任务书》及附件 `op.json`、`case.json`、`golden.py`。任务要求基于 Ascend C 开发标准 MHA/GQA KV cache 在线量化算子，采用 ACLNN 工程化交付，目标硬件为 Atlas 800T A2，CANN 版本为 9.1.0 及以上。

任务书定义总体功能及验收目标；附件 `op.json` 给出具体接口组合，`golden.py` 明确固定码本、比较规则和打包位序，`case.json` 给出基础用例。任务正文中超出附件的 dtype 组合单独列为后续扩展，不将其标为已支持。

算法背景参考 [TurboQuant 原论文](https://arxiv.org/html/2504.19874v1)的旋转标量量化及 QJL 残差修正；具体 ABI 与数值约定以任务附件为对齐对象。

## 背景介绍

### KvCacheTurboQuant 算子实现目标

标准 MHA/GQA 的 KV cache 容量随 token 数增加。一个 BF16、128 维的 K 或 V 向量占 256 字节。默认 3-bit 主编码、1-bit QJL，以及两个 BF16 标量合计 68 字节，理论 payload 压缩比约 3.76×。

本算子独立编码一批 K 或 V，不执行 attention、不处理 paged cache 索引，也不生成随机矩阵。调用方预生成 H、S 并在推理期间固定，后续消费方使用相同的矩阵、码本和位序。

统一 3-bit 主编码加 1-bit QJL 不等于混合 3.5-bit 模式；计入范数后实际为 4.25 bit/channel。首版不包含异常通道分组、混合位宽分配，也不覆盖 MLA latent cache。

### 参考实现现状分析

这是新增编码算子，参考对象为附件 PyTorch eager golden，不是同名 TBE 算子迁移。工程接入参考 ops-transformer 的 ACLNN 示例与 experimental 算子规范，不填写虚构的历史 TBE 路径。

仓内已有的 `turbo_quant_sparse_flash_attention`、`turbo_quant_sparse_attn_sharedkv` 面向其他 latent/shared-KV attention 流程，其维度与存储格式不同，不能直接作为本任务的编码器或解码器。

开发版将归一化、旋转、量化、残差投影和打包融合到 Vector Core 内，避免完整中间向量写回 GM，减少相对 eager 的算子提交和中间张量。

### 算子功能分析

输入为 KV 向量、正交旋转矩阵与高斯投影矩阵；输出为主编码、残差符号编码、输入范数和原尺度残差范数。各向量独立，K 和 V 分别调用同一接口，不涉及广播。

```text
BF16 x → FP32 范数 n → 归一化 u → H·u → y → 主码本比较 → idx → 打包
                                       └→ y-centroid[idx] → r → 范数 ρ
                                                               └→ S·(r/ρ) → 符号打包
n → BF16 quant_norm；n·ρ → BF16 quant_gamma
```

# 需求分析（required）

## 需求描述

实现 KvCacheTurboQuant，首个可执行接口采用 BF16 输入向量、FP32 矩阵、UINT8 编码和 BF16 范数。主位宽支持 2/3/4，默认 3；head_dim 固定 128，KV head 数为 4～32，覆盖 decode 与 prefill。

精度目标为任务书中的重构相对 MSE <0.05、相对内积误差 P95 <0.1；性能目标为相对功能等价 PyTorch eager reference 加速至少 10×，并在实际集成中不成为 prefill 瓶颈。格式压缩率、微基准和端到端吞吐分别验证。

## 需求拆解

| 子需求 | 设计方案 | 当前状态 |
| --- | --- | --- |
| ACLNN 接入 | OpDef、推形、Tiling、Kernel，构建系统生成 API | 已编译并运行 |
| 主量化 | FP32 归一化/旋转、固定码本比较 | 已实现 2/3/4-bit |
| QJL | 残差归一化、FP32 投影、非负编码为 1 | 已实现 |
| 紧凑存储 | 低位优先打包，清除 QJL 无效高位 | 已实现 |
| 参数检查 | rank、shape、dtype、位宽、溢出和平台资源 | 已实现，部分非法参数已测 |
| 边界数值 | 保持零向量、阈值相等、符号零点规则 | 仍存在浮点归约导致的少量码字差异 |
| 精度验收 | 正式矩阵、query 规则、重构/内积测试 | 待统一数据与判定规则 |
| 性能验收 | 同机对照、任务书基线复现、prefill 集成 | 同机微基准已测，其余待完成 |
| 扩展 dtype | BF16 矩阵、FP16 范数输出 | 当前未实现，需增加注册及测试 |

# 详细设计（required）

## 算子分析

### 数学公式

设列向量 x∈R^128，正交矩阵 H∈R^(128×128)，投影矩阵 S∈R^(Q×128)，主位宽为 b：

```text
n = ||x||₂
u = x/n，当 n>0；否则 u=0
y = H·u
c = CENTROIDS[b]
t[k] = (c[k]+c[k+1])/2
idx[j] = Σ_k 1{y[j] > t[k]}
r = y-c[idx]
ρ = ||r||₂
γ = n·ρ
v = r/ρ，当 ρ>0；否则 v=0
z = S·v
qjl[k] = 1{z[k] >= 0}
```

矩阵以 row-major ND 输入。批量行向量实现分别计算 `unit @ H.T`、`residual_unit @ S.T`。附件以 `clamp_min(1e-30)` 加零范数分支进行除法；当前 Kernel 使用 FP32 范数的非零分支，并按后文有效数值范围使用。

自测采用主重构和 QJL 修正重构，不增加本算子的 ABI：

```text
x_main = Hᵀ·(n_stored·c[idx])
signs = 2*qjl-1
x_corrected = Hᵀ·(n_stored·c[idx] + sqrt(π/2)/Q·gamma_stored·Sᵀ·signs)
```

QJL 修正要求 S 元素独立服从 N(0,1)；其他缩放需另行调整解码系数。测试使用实际存储的 BF16 范数，不以原始高精度范数替代。

#### 固定码本与比较规则

以下值来自附件 golden，在 Kernel 中作为 FP32 常量使用，不在线训练码本：

```text
2-bit:
[-0.1335033178, -0.04002048075, 0.04002048075, 0.1335033178]
3-bit:
[-0.19020693, -0.1187859178, -0.06682205945, -0.02166347019,
  0.02166347019, 0.06682205945, 0.1187859178, 0.19020693]
4-bit:
[-0.2414890379, -0.1828317791, -0.1429702938, -0.1109927073,
 -0.08325428516, -0.05802082643, -0.03428063914, -0.01134236995,
  0.01134236995, 0.03428063914, 0.05802082643, 0.08325428516,
  0.1109927073, 0.1429702938, 0.1828317791, 0.2414890379]
```

阈值相等时保持较小主索引；投影为零时 QJL 写 1。零输入的 norm、gamma 为零，其主码/QJL 仍按 golden 计算，不将所有编码字节强制清零。

#### 位打包约定

第 j 维索引的第 k 位写入 bit offset `j*b+k`，字节内低位优先。每 8 维形成一个 8*b 位整数，依次写出 b 个低字节，3-bit 索引允许跨字节。例如 `[0,1,2,3,4,5,6,7]` 的 3-bit 打包为 `[0x88,0xC6,0xFA]`。

QJL 第 k 位写入字节 `k//8` 的 `k%8` 位，末字节超出 Q 的高位为零。

| b | 主码/B | Q=128 的 QJL/B | 范数/B | 总字节/head | bit/channel | 相对 BF16 压缩比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 32 | 16 | 4 | 52 | 3.25 | 4.92× |
| 3 | 48 | 16 | 4 | 68 | 4.25 | 3.76× |
| 4 | 64 | 16 | 4 | 84 | 5.25 | 3.05× |

上述仅计算编码 payload，不含共享 H/S、分配器对齐、张量元数据及其他 cache 结构。

### 支持数据类型

| 参数 | 输入/输出 | dtype | 格式 |
| --- | --- | --- | --- |
| kv_vectors | 输入 | BF16 | ND |
| rotation_matrix | 输入 | FP32 | ND |
| qjl_matrix | 输入 | FP32 | ND |
| quant_idx | 输出 | UINT8 | ND |
| quant_qjl | 输出 | UINT8 | ND |
| quant_norm | 输出 | BF16 | ND |
| quant_gamma | 输出 | BF16 | ND |
| mse_bits | 可选属性 | int，默认 3，允许 2/3/4 | — |

输入先 Cast 为 FP32；范数、旋转、阈值、残差和投影均使用 FP32，最后范数以 CAST_RINT 转 BF16。当前不将矩阵降为 BF16，不使用低精度 Cube 路径替代该数值方案。

### 支持形状

记 token 数为 T、KV head 数为 K、QJL 维度为 Q：

| 参数 | shape |
| --- | --- |
| kv_vectors | [T,K,128] |
| rotation_matrix | [128,128] |
| qjl_matrix | [Q,128] |
| quant_idx | [T,K,16*b] |
| quant_qjl | [T,K,ceil(Q/8)] |
| quant_norm、quant_gamma | [T,K] |

T>=0，4<=K<=32，1<=Q<=1024。任务基础用例使用 Q=128；1024 是当前实现的扩展上限，不能外推到任意 Q。输入不广播，注册 AutoContiguous；直接 Kernel 调用依赖连续 ND 数据，输出按上述尺寸分配连续空间。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

OpDef 注册 KvCacheTurboQuant，由仓库构建系统生成 ACLNN 两段式 API；模块提供推形、推 dtype 与 Tiling，不手写重复的分发层。

```cpp
aclnnStatus aclnnKvCacheTurboQuantGetWorkspaceSize(
    const aclTensor *kvVectors,
    const aclTensor *rotationMatrix,
    const aclTensor *qjlMatrix,
    int64_t mseBits,
    const aclTensor *quantIdxOut,
    const aclTensor *quantQjlOut,
    const aclTensor *quantNormOut,
    const aclTensor *quantGammaOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnKvCacheTurboQuant(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

调用方检查两阶段返回值，按第一阶段返回值分配 workspace；第二阶段在传入 stream 上提交。输入、输出和 workspace 的生命周期需覆盖执行完成。参数错误由框架或 Tiling 返回，成功状态为 ACLNN_SUCCESS。

推形保留 T/K，主码末维为 16*b；Q 未知时 QJL 末维保持动态维度。运行时 Tiling 必须得到合法具体维度，不读取设备矩阵值选择路径。

##### 1. 分核策略：

前两维展平为 rows=T*K，每组 G=32 个向量：

```text
groups = ceil(rows/32)
blockDim = max(1, min(groups, available_aiv_cores))
core c 的组起始行：base = 32*c + k*32*blockDim，k=0,1,...
有效行数：count = min(32, rows-base)
```

完整组的主码、QJL、每个范数字节数分别为 `32*16*b`、`32*ceil(Q/8)`、`32*2`，组起始地址均为 32 字节对齐。不同核拥有完整输出组，不共享字节读改写；最后一组按有效长度写回。

T=0 时 blockDim 至少为 1，Kernel 在 GM 数据读取前返回。小 decode 可能仅使用一个 Vector Core；进一步拆分必须同时证明输出块无跨核干扰。

##### 2. 数据分块和内存优化策略：

Host 读取 GetCoreNumAiv 和实际可用 UB。当前固定缓冲布局如下：

| UB 类别 | 字节 | 用途 |
| --- | ---: | --- |
| H 缓存 | 65536 | 每核加载一次，跨向量复用 |
| S 缓存 | 65536 | Q<=128 一次缓存；更大 Q 按块复用 |
| 乘积/索引工作区 | 32768 | 64×128 FP32 乘积，后复用为主索引工作区 |
| 权重和 mask 区 | 4096 | 打包权重、比较 mask |
| 一组输入 | 8192 | 最多 32×128 个 BF16 元素 |
| u、y、r、投影结果 | 2048 | 各 128 个 FP32 元素 |
| 归约/打包临时区 | 1024 | 范数及打包归约，按阶段复用 |
| 主码输出 | 2048 | 最多 32×64 字节 |
| QJL 输出 | 4096 | 最多 32×128 字节 |
| FP32 范数/gamma | 256 | 两组 32 个 FP32 |
| BF16 范数/gamma | 128 | 两组 32 个 BF16 |
| 码本 | 128 | 当前位宽的中心值及预留空间 |
| **合计** | **185856** | 显式 UB 缓冲 |

Host 要求至少 `186*1024=190464` 字节可用 UB，且 AIV 数大于零。开发节点返回 196352 字节，不能把标称 192 KiB 直接作为最小可用量。当前采用单缓冲，未实现 double buffer。

Kernel Tiling 的 GM workspace 为 0，不保存完整旋转/残差中间张量到 GM。框架连续化等可能影响 ACLNN 返回的实际 workspace，调用方仍以 API 返回值为准。

##### 3. tilingkey规划策略：

当前仅注册 A2、单一 dtype 组合，tilingKey=0，位宽和 Q 通过 TilingData 传递：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| rows | uint64_t | T*K |
| qjlDim | uint32_t | Q |
| bits | uint32_t | b |

扩展 dtype 或引入独立小/大 shape 实现时再增加 key 和配套测试，当前不宣称这些候选路径已实现。

##### 4. 数据检测：

检查三个输入、四个输出的描述、rank、dtype、维度及 b；在行数/元素数乘法前检查整数溢出。输出必须符合推导形状，Q/K/b 越界、head_dim 非 128 等返回失败。

Host 不拷回 H/S 检查正交性或高斯分布，也不扫描输入值。这些属于调用方契约。NaN、Inf、极端幅度及范数溢出不属于当前已验证范围。

#### 3.2.2 kernel侧设计：

Init 绑定 GM 张量并分配 TPipe/TBuf；Process 缓存矩阵/码本，按组搬入、计算、搬出。

1. **CopyIn**：按有效行数搬入 BF16 向量。H 每核加载一次；Q<=128 时 S 一次加载，Q>128 时在每向量内分最多 128 行读取。
2. **归一化**：Cast、Mul、ReduceSum、Sqrt 得到 FP32 范数；正范数乘倒数，零范数生成零向量。
3. **矩阵向量乘**：每次最多 64 个矩阵行；128 维分成两个 64 维片段进行广播 Mul，两半相加后 WholeReduceSum。S 的末块只处理有效行。
4. **主量化/残差**：CompareScalar、Select、Add 生成索引和重构中心。以 <= 保留旧档等价于 golden 的严格 > 升档，随后 Sub 得到残差。
5. **主码打包**：2/3-bit 的 8 维加权整数分别不超过 16/24 位，可用 FP32 精确归约；4-bit 拆成两个 16 位加权和，避免 FP32 无法精确保存一般 32 位整数。提取字节写 UB。
6. **QJL**：计算残差范数、gamma、残差单位方向及投影。CompareScalar 的 >=0 mask 直接提供符号位，清除末字节无效高位。
7. **CopyOut**：范数 CAST_RINT 到 BF16，四个输出按有效字节长度 DataCopyPad 写回。未用范数槽位和投影尾部先清零。

在 Vector 依赖、MTE 搬运、标量读取/写入边界显式同步。当前同步较保守，尚未缩减为精细事件流水。

##### 性能优化方案

已实现多操作融合、H/S 复用、向量阈值比较、精确加权打包及组级写回。相较逐维标量搜索的早期实现，同一开发节点 3-bit prefill 阶段耗时由约 11.9 ms 降至约 1.9 ms；这不是官方基准结果。

后续针对 prefill 的矩阵批处理、同步/指令开销、小 shape 分核和 Q>128 的投影矩阵复用优化。若引入 Cube 或降低矩阵精度，必须重新验证主码阈值和 QJL 符号，不以未经验证的精度降低换取性能结论。

##### 工程组织

```text
experimental/attention/kv_cache_turbo_quant/
├── CMakeLists.txt
├── README.md
├── docs/design.md
├── op_host/
│   ├── CMakeLists.txt
│   ├── kv_cache_turbo_quant_def.cpp
│   ├── kv_cache_turbo_quant_infershape.cpp
│   ├── kv_cache_turbo_quant_tiling.cpp
│   └── kv_cache_turbo_quant_tiling.h
├── op_kernel/kv_cache_turbo_quant.cpp
└── tests/
    ├── aclnn_bridge.cpp
    ├── golden.py
    ├── reference.py
    ├── test_kv_cache_turbo_quant.py
    └── run.sh
```

竞赛仓提交设计文档，实现和阶段原始结果保存在 ops-transformer 开发提交。测试桥接调用真实 ACLNN，不以 golden 代替被测内核。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Atlas 800I/T A2 | √ | 任务要求 Atlas 800T A2；开发节点为 Ascend 910B4，ascend910b 配置 |

不宣称 A3/A5 已支持。开发环境为 CANN 9.1.0、Python 3.12.13、PyTorch 2.9.0+cpu、torch-npu 2.9.0.post6。

## 算子约束限制

- 仅标准 MHA/GQA 的 K 或 V；不支持 MLA latent/RoPE 缓存。
- head_dim=128，K=4～32，b=2/3/4，Q=1～1024；矩阵缓存主路径为 Q<=128。
- H/S 的方向、形状、分布与固定性由调用方保证；未归一化 Hadamard 不能当作正交矩阵。
- 生产者/消费方必须共享码本、矩阵、位序、QJL 符号和 gamma 标度；尚未验证与其他 attention 算子互通。
- 当前按 op.json 固定 dtype；BF16 矩阵/FP16 范数输出待扩展。
- 输入需处于 FP32 范数计算和 BF16 范数存储的有效范围；不定义非有限值编码语义。
- 不广播，不提供 paged cache 写入、反量化或 attention 融合接口。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 编码实现 | 对照 golden 主码/QJL，范数单独比较，明确记录边界差异 | golden.py / op.json |
| 重构精度 | MSE/输入平方范数 <0.05，同时报告相对平方 L2，防止口径混淆 | 任务书 3.2 |
| 内积精度 | abs(<q,x_hat>-<q,x>)/abs(<q,x>) 的 P95 <0.1 | 任务书 3.2 |
| 性能 | 功能等价 eager 加速 >=10×，且不成为 prefill 瓶颈 | 任务书 3.3 |
| 存储 | 默认 68 B/head，payload 压缩比约 3.76× | 任务书存储格式 |
| 额外内存指标 | 任务注明不涉及，仍检查容量、尾部及越界 | 任务书 3.4 |

### 基础精度与性能用例

| case | T | K | head_dim | b | Q | 任务书 eager/us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gqa_decode_b1 | 1 | 8 | 128 | 3 | 128 | 1491.676 |
| gqa_decode_b64 | 64 | 8 | 128 | 3 | 128 | 1552.784 |
| gqa_prefill_t2048 | 2048 | 8 | 128 | 3 | 128 | 3855.422 |
| gqa_mse2_t2048 | 2048 | 8 | 128 | 2 | 128 | 3607.963 |
| gqa_mse4_t2048 | 2048 | 8 | 128 | 4 | 128 | 5286.502 |

补充用例覆盖 Q=1/127/129/1024、K=4/32、非整组行数、零向量、旋转后零坐标的阈值规则和 T=0。非法参数包括 K=3/33、b=1/5、Q=0/1025。错误 rank/head_dim/dtype、输出 shape、不连续输入及多随机种子属于后续需补齐的自动化覆盖，不列为已完成测试。

### 精度验证与数值边界

1. 原始 golden 对照四个输出。主码/QJL 默认严格字节比较，范数以实际 BF16 值比较。当前开发容差为 rtol=0.008、atol=1e-6，需与最终验收规则统一。
2. 不一致时独立用 FP64 计算旋转、阈值和残差投影，记录差异距离与受影响向量，不修改原始 golden。
3. 开发诊断按主码边界距离 <2e-7、QJL 零点距离 <1e-5 定位不连续点；这不等同于严格 UINT8 通过。测试默认仍返回失败，显式 --diagnostic 才继续数值分析。
4. 同时报告主重构和 QJL 修正重构。若 MSE 按维平均，则 MSE/输入平方范数与相对平方 L2 相差 128 倍，最终口径需统一。
5. 内积测试不通过增大分母或剔除近零样本获得有利结果。正式 query 分布、零分母规则及两种重构的验收角色尚需明确。

附件引用的两份矩阵二进制未随本地任务包提供。阶段测试使用 seed=20260923 的 QR 正交矩阵和高斯投影，不称为官方矩阵；独立高斯 query 使用 seed=2030，仅作诊断。

### 性能测量与阶段结果

设备端 eager 保留附件的码本创建、主量化、残差投影和位打包，去除 CPU 数据转换，并另行验证输出与原始 golden 一致。双方预热后计时，每组 20 次、3 组均值取中位数。时间包含稳态提交与完成同步，不含输入生成/输出 CPU 拷回。

下表来自开发提交 6c0ba1217 的 task_submission/development_results.json，不是本次文档编写重新执行的数据：

| case | ACLNN/us | 同机 eager/us | 同机加速比 | 任务书基线/本次 ACLNN |
| --- | ---: | ---: | ---: | ---: |
| gqa_decode_b1 | 41.201 | 4149.915 | 100.72× | 36.20× |
| gqa_decode_b64 | 151.841 | 3970.057 | 26.15× | 10.23× |
| gqa_prefill_t2048 | 1911.662 | 43619.252 | 22.82× | 2.02× |
| gqa_mse2_t2048 | 1689.410 | 43446.524 | 25.72× | 2.14× |
| gqa_mse4_t2048 | 2223.028 | 46391.406 | 20.87× | 2.38× |

同机 eager 显著慢于任务书示例，因此同机 >=10× 不能替代任务书基线复现。仍需分析框架、计时和设备差异，并对照真实 KV 生成速率进行 prefill 集成验证。

11 个正常/边界用例完成数值诊断，7 个参数用例符合预期。严格字节检查仍有两例失败：3-bit prefill 有 1 个 QJL 字节差异；4-bit prefill 有 1 个主码及 12 个 QJL 字节差异。独立高斯 query 的内积 P95 未达到 0.1。这些是验收待办，不能用诊断容差或同机加速比替代完整结论。

### 可复现步骤与可维护性

在保存的代码提交上，激活配套环境后，从 ops-transformer 根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b \
  --ops=kv_cache_turbo_quant --vendor_name=turboquant -j8
bash build_out/cann-ops-transformer-turboquant_linux-aarch64.run \
  --quiet --install-path=/workspace/kv_cache_turbo_quant_dev/kv_custom_opp
bash experimental/attention/kv_cache_turbo_quant/tests/run.sh \
  /workspace/kv_cache_turbo_quant_dev/kv_custom_opp/vendors/turboquant_transformer \
  /workspace/kv_cache_turbo_quant_dev/strict_test
```

当前严格测试会因已记录差异非零退出，显式附加 --diagnostic 用于数值分析，不改变标准。保留 JSON、日志、源码提交和软件版本。Python 依赖包括 numpy<2、ml_dtypes==0.5.3 与配套 torch/torch_npu。

维护时保持 OpDef、推形、Tiling、Kernel 职责分离；码本、位序、dtype 更改必须同步文档与测试。已观察到当前仓库的增量源复制/Kernel 完成标记不追踪源码变化，复现优先非增量构建；增量调试需核对实际编译输入及二进制更新，避免旧产物造成错误结论。

最终按任务书提供设计评审、API 文档、精度/性能自测表格及日志。本设计稿和阶段记录不冒充已完成的官方验收材料。

## 兼容性分析

新增算子不修改现有 attention API/cache 布局。注册名为 KvCacheTurboQuant，Kernel 入口为 kv_cache_turbo_quant，消费方显式接入。仅确认 A2、CANN 9.1.0 开发环境；后续 CANN 版本需继续验证。

增加 BF16 矩阵、FP16 范数、其他 Q/head_dim 或融合路径时，应扩展注册和测试，不改变已有位序与 gamma 含义。矩阵不逐向量随编码保存，序列化或跨进程消费 cache 时，上层必须管理匹配的矩阵/码本版本。

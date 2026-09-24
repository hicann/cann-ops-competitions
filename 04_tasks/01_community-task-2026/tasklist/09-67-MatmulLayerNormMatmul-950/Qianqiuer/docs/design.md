# MatmulLayerNormMatmul 算子设计文档

本文档对应 2026 年 9 月社区任务《MatmulLayerNormMatmul 算子开发任务书》，按社区任务设计文档模板编写。本文只描述设计方案和验证计划，性能、精度和测试结论以代码完成后的 950PR 实测报告为准。

## 需求背景（required）

### 需求来源

| 项目 | 内容 |
| --- | --- |
| 任务书 | 9月社区任务-MatmulLayerNormMatmul算子开发任务书 |
| 任务测试集 | `test_case/MatmulLayerNormMatmul_测试集.csv`，共 116 条有效用例 |
| 目标开源仓 | [cann/catlass](https://gitcode.com/cann/catlass) |
| 适配硬件 | Ascend 950，性能测试使用 Ascend 950PR |
| 开发语言 | Ascend C，复用 CATLASS 的 Matmul、TLA、JIT 和 Optest 组件 |
| 精度标准 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 参考合入 PR | [CATLASS PR 678](https://gitcode.com/cann/catlass/pull/678) |

仓库中已经存在同一任务的其他投稿目录，本设计放在既有任务目录 `09-67-MatmulLayerNormMatmul-950/Qianqiuer/docs/design.md` 下，不新建重复任务编号。

### 背景介绍

`MatmulLayerNormMatmul` 是 Transformer、MLP 和投影层中常见的融合计算链。独立执行时需要依次启动两个 Matmul 和一个 LayerNorm，并将中间矩阵写回、读出显存；在任务集的小批量形状上，启动和中间张量搬运占比较高。因此本任务实现一个 Ascend C/CATLASS 融合算子，在一次 MIX kernel launch 中完成：

```text
C0       = A0 @ B0
C0_norm  = LayerNorm(C0, gamma, beta, eps=1e-6)
C1       = C0_norm @ B1
```

其中 C0 和 C0_norm 是算子内部中间结果，不作为公开输出。统计量 mean、variance 和 inverse standard deviation 由内部 workspace/片上缓冲管理，不向调用者暴露。

### 功能定义

第一阶段 Matmul：

```text
C0[m, n] = sum(A0[m, k] * B0[k, n]), k = 0 .. K0 - 1
```

第二阶段按行进行 LayerNorm：

```text
mean[m]       = sum(C0[m, n]) / N0
variance[m]   = sum((C0[m, n] - mean[m])^2) / N0
C0_norm[m,n]  = (C0[m,n] - mean[m]) / sqrt(variance[m] + 1e-6)
                * gamma[n] + beta[n]
```

第三阶段 Matmul：

```text
C1[m, p] = sum(C0_norm[m, n] * B1[n, p]), n = 0 .. N0 - 1
```

方差使用总体方差，不使用 Bessel 校正；epsilon 固定为 `1e-6`，不是公开属性。唯一对外输出为 FP16 的 C1。

## 需求分析（required）

### 需求描述

使用 Ascend C/CATLASS 在 Ascend 950 上实现 `MatmulLayerNormMatmul`，满足以下要求：

1. 一次 kernel launch 完成 Matmul、行级 LayerNorm 和第二个 Matmul。
2. 中间统计量由算子内部 workspace 管理，不提供 C0、C0_norm、mean 或 variance 输出。
3. A0、B0、B1 和 C1 使用 FP16，gamma、beta 使用 FP32。
4. 遵循任务书规定的 RowMajor/ColumnMajor 布局、形状关系和 ND 数据格式。
5. 任务测试集 116 条精度全部通过，并按生态算子开源精度标准判定。
6. 使用 `msprof op` 在 Ascend 950PR 上采集完整算子耗时，平均标杆时延与测试时延之比大于 1.1。
7. 提供 CATLASS Optest 测试件、至少 200 条 ATK 泛化测试、README、测试步骤和自验证材料。

### 参数和数据布局

| 参数 | 方向 | dtype | 逻辑形状 | 数据格式 | 物理布局/元素 stride |
| --- | --- | --- | --- | --- | --- |
| A0 | 输入 | FP16 | `(M0, K0)` | ND | RowMajor，`(K0, 1)` |
| B0 | 输入 | FP16 | `(K0, N0)` | ND | ColumnMajor，`(1, K0)` |
| B1 | 输入 | FP16 | `(N0, M1)` | ND | ColumnMajor，`(1, N0)` |
| gamma | 输入 | FP32 | `(N0,)` | ND | 连续，`(1)` |
| beta | 输入 | FP32 | `(N0,)` | ND | 连续，`(1)` |
| C1 | 输出 | FP16 | `(M0, M1)` | ND | RowMajor，`(M1, 1)` |

ColumnMajor 的地址计算必须依据实际 stride：`B0[k, n]` 地址为 `B0_base + k + n*K0`，`B1[n, p]` 地址为 `B1_base + n + p*N0`。不得把列主输入无条件转换为连续行主输入后仍沿用列主地址公式。Host 侧在下发 kernel 前校验逻辑 shape、dtype、stride 和 device 一致性。

### 形状关系和合法性

合法输入必须满足：

```text
A0.shape = (M0, K0)
B0.shape = (K0, N0)
B1.shape = (N0, M1)
gamma.numel() = beta.numel() = N0
C1.shape = (M0, M1)
M0, K0, N0, M1 > 0
```

任务测试集共有 116 行，使用 CSV 中的原始顺序和标杆耗时，不将测试集取值域误写成接口限制。已知任务集主要覆盖 `M0={128,512,1024,2048}`、`K0={768,2048,4096,8192}`、`N0={2048,3072,4096,8192}`、`M1={768,2048,4096}` 的组合。实现使用 CeilDiv 处理非对齐尾块，不能假设所有输入都能被某个 TileShape 整除。

任务书没有声明广播、空矩阵、负维度或非连续布局的扩展语义。对不满足上述契约的输入，在 Host 侧返回清晰错误并且不启动 kernel；不静默改变布局或形状。

### dtype 和计算精度

| 阶段 | 输入 | 累加/统计 | 输出 |
| --- | --- | --- | --- |
| Matmul0 | A0/B0 FP16 | Cube FP32 累加 | C0 片上 FP16 |
| LayerNorm 统计 | C0 FP16 | FP32 sum、FP32 squared-difference | mean/variance/invstd FP32 |
| LayerNorm 仿射 | C0 FP16、gamma/beta FP32 | FP32 减法、乘法和加法 | C0_norm 片上 FP16 |
| Matmul1 | C0_norm/B1 FP16 | Cube FP32 累加 | C1 FP16 |

均值和方差采用两遍 FP32 归约，避免 `E[x^2] - E[x]^2` 在大均值、小方差输入上发生抵消。对由于舍入出现的极小负方差，计算 inverse standard deviation 前按 `max(variance, 0)` 处理。不得使用会改变验收精度的 fast-math 近似；最终输出转换遵循 FP16 舍入语义。

## 详细设计（required）

### 算子分析

#### 融合收益

独立基线为：

```python
c0 = torch.mm(a0, b0)
c0_norm = torch.nn.functional.layer_norm(c0, (N0,), gamma, beta, 1e-6)
c1 = torch.mm(c0_norm, b1)
```

基线需要三次下发，并且至少两次把中间矩阵写回、读出 GM。融合实现只下发一次 MIX kernel：AIC 负责 Cube Matmul，AIV 负责行归约和仿射；同一行块的 C0 在 L1/UB 中生产、归一化并直接作为第二个 Matmul 的左输入。对无法在片上容纳完整归约行的形状，采用分块统计与第二遍重算 C0 tile 的方式，不把 C0 或 C0_norm 写入 GM。

#### 片上数据流

```text
GM A0/B0 ──MTE2──> L1/L0 ──Cube──> C0 tile (L0C/L1/UB)
                                      │
                                      ├─Vector pass 1: sum/squared-difference
                                      │                 -> mean/variance workspace
                                      └─Vector pass 2: normalize * gamma + beta
                                                        -> C0_norm tile
                                                         │
GM B1 ──MTE2──> L1/L0 <───────────────┴────Cube Matmul1
                                      │
                                  FixPipe/MTE3 -> GM C1
```

统计量 workspace 只保存每个输出行的 FP32 `mean` 和 `invstd`（需要时保存 variance 供调试/复核），按 32B 对齐。该 workspace 由 Host 在算子调用内部申请和释放，公开接口只返回 C1。

### Host 侧设计

#### 接口和参数校验

样例入口和 Optest JIT 适配器使用同一参数结构，逻辑字段如下：

```cpp
struct MatmulLayerNormMatmulParams {
    uint32_t m0;
    uint32_t k0;
    uint32_t n0;
    uint32_t m1;
    GM_ADDR a0;
    GM_ADDR b0;
    GM_ADDR b1;
    GM_ADDR gamma;
    GM_ADDR beta;
    GM_ADDR c1;
    GM_ADDR workspace;
};
```

Host 在分配 workspace 和输出、启动 kernel 之前依次检查：

1. 所有地址非空且位于同一 NPU device；输出地址可写。
2. A0/B0/B1/C1 为 FP16，gamma/beta 为 FP32，rank 分别为 2、2、2、2、1、1、2。
3. A0、B0、B1、C1 的 shape 关系满足上节约束。
4. A0/C1 是 RowMajor，B0/B1 是 ColumnMajor，gamma/beta 连续；stride 不溢出 64 位地址计算。
5. `M0*K0`、`K0*N0`、`N0*M1` 以及 workspace 字节数在 `size_t` 和 kernel 参数类型范围内。

校验失败只返回错误码或抛出参数异常，不下发任何 device kernel。Host 不暴露 C0、C0_norm、mean、variance 等输出。

#### Tiling 策略

Tiling 根据 Ascend 950 平台信息和实际 shape 在候选档位中选择：

| 参数 | 作用 |
| --- | --- |
| `BM` | 一个任务处理的行数，保证 C0 行块和第二个 Matmul 左块能放入 L1/UB |
| `BN0` | LayerNorm 归约的列分块，按 32B/128B 对齐选择 |
| `BK0` | Matmul0 K 方向分块，匹配 Cube 输入粒度 |
| `BN1` | Matmul1 输出列分块，平衡 L0C 与 L1 B1 缓存 |
| `ubStages` | UB 双缓冲深度，默认 2；尾块自动降级为单缓冲 |
| `coreNum` | 由平台可用 AIC 核数和行块数共同决定，避免空核 |
| `tilingKey` | 编码 BM/BN0/BN1 档位、是否需要 C0 重算以及尾块路径 |

片上空间预算满足：

```text
C0/C0_norm tile + A0/B0/B1 tile + LayerNorm FP32 scratch
    + double-buffer reserve <= L1/UB/L0C available capacity
```

当 `BM*RoundUp(N0, 16)*sizeof(fp16)` 可放入单个行块的片上缓存时，保留完整 C0 行块，统计后直接完成仿射和 Matmul1。当 N0 较大时，第一遍按 BN0 处理 C0 tile 并累加每行统计量，第二遍重算同一 C0 tile，依据 workspace 中的 mean/invstd 完成归一化并送入 Matmul1；两条路径的输出语义一致，且都不向 GM 写 C0/C0_norm。

分核按行块分配。行块数量少于可用核数时只启用实际需要的核；不能整除时，前若干核多处理一个行块。一个行块的 C1、统计量和片上缓存归属唯一，避免跨核写冲突，不使用原子操作。

#### Workspace 规划

```text
mean[M0]     : FP32, aligned 32B
invstd[M0]   : FP32, aligned 32B
variance[M0] : FP32, optional internal diagnostic area
```

正式运行只需要 mean 和 invstd；variance 区域是否保留由 debug/验证编译选项决定，不能改变公开接口。workspace 大小由 Host 根据 M0 和对齐规则计算，并通过 CATLASS/运行时分配接口取得。workspace 生命周期覆盖一次 kernel launch，完成后立即释放。

### Kernel 侧设计

#### MIX 角色分工

使用 CATLASS 的 Ascend 950 MIX kernel 形态，按实际环境支持的核配比配置 AIC/AIV。AIC 负责两个 Matmul 的 Cube 指令，AIV 负责 LayerNorm 的向量归约和仿射。AIC/AIV 通过 CATLASS/Ascend C 的跨流水同步原语传递行块就绪和消费完成状态；每个行块使用独立 flag 槽位，flag 在循环末复位，避免下一块读到旧状态。

如果某个构建配置不支持 MIX，Host 必须在编译期报出不支持，而不是退化为三个独立 kernel launch。正式交付只接受单次 kernel launch 的 MIX 路径。

#### CopyIn、Compute 和 CopyOut

每个行块采用二级流水：

1. **CopyIn**：MTE2 将 A0、B0、B1 的当前 tile 搬入 L1/UB；gamma、beta 按 N0 tile 搬入 UB，并进行边界 mask 设置。
2. **Matmul0**：L0A/L0B 分块装载，Cube FP32 累加到 L0C；固定输出转换为片上 FP16 C0 tile。
3. **LayerNorm pass 1**：AIV 对每行做 FP32 sum，写入 mean；再次计算中心化平方和，生成 variance/invstd。
4. **LayerNorm pass 2**：用 mean/invstd、gamma、beta 对 C0 tile 做 FP32 仿射，转换为 FP16 C0_norm。
5. **Matmul1**：C0_norm 作为第二个 Matmul 的左矩阵，B1 作为列主右矩阵，Cube FP32 累加到 L0C。
6. **CopyOut**：FixPipe/MTE3 将 C1 按 RowMajor 输出；尾行、尾列使用有效 mask，不能写越界。

CopyIn、Vector 和 Cube 采用 double buffer，在当前 tile 计算时预取下一 tile；只有真正存在尾块时关闭对应 buffer。所有队列和事件在 kernel 入口初始化，在正常路径和异常边界路径都完成配对。

#### LayerNorm 数值实现

对每一行：

```text
sum      = FP32ReduceSum(c0_tile)
mean     = sum / float(N0)
delta    = FP32(c0) - mean
sum_sq   = FP32ReduceSum(delta * delta)
variance = max(sum_sq / float(N0), 0.0f)
invstd   = rsqrt(variance + 1.0e-6f)
out      = delta * invstd * gamma + beta
```

N0 跨多个 tile 时，第一遍保存每行的 sum 和 sum_sq 所需的累加状态，第二遍完成中心化计算。归约顺序固定为从低列到高列的 tile 顺序，保证同一输入下结果可复现。gamma/beta 不参与归约，按列索引读取并使用 FP32 运算。

### 目录和交付设计

代码 PR 只提交源代码、必要的构建文件、README 和正式 Optest 接入件，不提交二进制、编译目录、profiling 原始大文件、临时脚本、模型权重或个人敏感信息。按任务书组织为：

```text
experimental/matmul/matmul_layer_norm_matmul/
├── CMakeLists.txt
├── matmul_layer_norm_matmul.cpp
├── README.md
└── test_matmul_layer_norm_matmul.py

include/catlass/
└── ...                         # 必要的 kernel/block/tile 组件修改

tests/optest/
├── include/catlass_kernel_jit.h
├── kernels/matmul_layer_norm_matmul/
│   ├── matmul_layer_norm_matmul.cpp
│   ├── matmul_layer_norm_matmul_impl.cpp
│   └── CMakeLists.txt
├── src/catlass_matmul_layer_norm_matmul.cpp
├── src/include/template/matmul_layer_norm_matmul.h
└── torch_catlass/
    ├── __init__.py
    └── ops/matmul_layer_norm_matmul.py
```

测试工程中保存官方 CSV 的可追溯副本、CPU/PyTorch golden 生成逻辑、Optest 参数化用例和 ATK 用例生成器；测试结果和截图在验收材料中提交，不混入算子源代码 PR。

## 支持硬件

| 支持型号 | 是否支持 | 说明 |
| --- | --- | --- |
| Ascend 950 / Ascend 950PR | √ | 目标开发和性能验收硬件，使用 CATLASS Ascend 950 架构配置 |
| 其他硬件 | — | 本任务未要求移植，不在验收范围内 |

## 算子约束限制

任务书未给出额外业务约束；实现采用以下输入契约：

1. 输入和输出 dtype 固定为参数表所列类型，不做隐式 dtype 转换。
2. B0/B1 必须按 ColumnMajor stride 提供，A0/C1 必须按 RowMajor stride 提供。
3. 不支持广播；gamma 和 beta 只能沿 N0 维逐列作用。
4. M0、K0、N0、M1 必须为正整数，且维度乘积不能溢出地址和 workspace 计算。
5. epsilon 固定为 `1e-6`，不接受运行时覆盖。
6. 不支持将 C0、C0_norm、mean、variance 作为外部输出；调用者只接收 C1。

不满足契约的输入由 Host 侧拒绝。测试会覆盖非法 dtype、shape、stride、空地址和输出形状，确保错误发生在 kernel launch 之前。

## 可维可测分析（required）

### 精度标准

精度判定遵循生态算子开源精度标准，不自行放宽阈值。参考实现使用 FP32 中间计算的 CPU/PyTorch 链：

```python
c0 = torch.mm(a0.float(), b0.float()).to(torch.float16)
c0_norm = torch.nn.functional.layer_norm(
    c0, (n0,), gamma.float(), beta.float(), eps=1e-6)
c1 = torch.mm(c0_norm, b1.float()).to(torch.float16)
```

实际 golden 生成脚本必须与代码中的 Matmul FP16 输出语义和布局一致，不能把列主输入误解释为行主。每个用例检查输出 shape、dtype、有限值状态及标准规定的绝对/相对误差；对全零输入、常量行、极小方差、正负大值和随机输入分别记录最大误差、平均误差和失败元素数。

### 性能标准

任务测试集 CSV 的最后一列是每个 shape 在 Ascend 950PR 上的“小算子拼接”标杆耗时。基线实现固定为：

```python
torch.mm(a0, b0)
torch.nn.functional.layer_norm(c0, (n0,), gamma, beta, eps=1e-6)
torch.mm(c0_norm, b1)
```

性能采集使用 `msprof op`，每个用例固定设备、预热次数、正式采样次数和同步方式，同时采集基线与融合实现。报告至少包含 case id、M0/K0/N0/M1、标杆耗时、融合耗时、workspace、TileShape、kernel 列表及 P50/P90。总体验收指标为：

```text
average(benchmark_latency) / average(fused_latency) > 1.1
```

报告同时保留逐 case 比值和算术平均值；任何用例不得用估算值、CPU 时间或不同 shape 的结果替代。不同 TileShape、流水深度或实现方案的对比需记录选择理由和最终配置。

### 测试设计

#### Optest 精度和功能测试

使用 CATLASS Optest 接入件执行任务 CSV 的 116 条用例，覆盖：

- 四维 shape 组合和每行 LayerNorm 归约；
- FP16 Matmul 输入/输出、FP32 gamma/beta；
- RowMajor A0/C1、ColumnMajor B0/B1 的地址与 stride；
- 对齐 tile、尾行、尾列和 K0 尾块；
- 全零、常量行、随机值、极小方差和大幅值输入；
- 非法 dtype、shape、stride、空地址和错误输出 shape 的 Host 错误处理；
- 单次 kernel launch、输出唯一性和中间结果不外露。

每个用例输出结构化日志，包含 `case_id`、输入摘要、golden 摘要、实际输出摘要、最大误差和 PASS/FAIL。任何精度或参数校验失败都使测试命令返回非零。

#### ATK 泛化测试

使用 `catlass-atk-support` 生成不少于 200 条独立测试条目，建议生成 240 条，分层覆盖：

| 分组 | 变化维度 |
| --- | --- |
| 基础尺寸 | M0、K0、N0、M1 的小、中、大值及非 128/256 对齐尾块 |
| 布局 | A0/C1 RowMajor，B0/B1 ColumnMajor，含合法 stride 组合 |
| 数值 | 随机、零、常量、正负大值、极小方差、gamma/beta 为零或非零 |
| 归约 | N0 小于、等于和大于单个片上 tile，触发完整行和分块重算路径 |
| 边界 | M0=1、M1=1、K0=1、N0=1 以及接近 workspace 上限的合法输入 |

每条 ATK 用例使用独立随机种子和 CPU/PyTorch golden，至少覆盖 200 条精度通过；结果 JSON、case manifest 和失败重放命令作为验收材料提交。

#### 性能和 Profiler 测试

对 CSV 的全部 116 条用例执行 `msprof op`：

1. 先执行基线链并记录 GPU/NPU device event 的整体端到端时间。
2. 再执行融合 kernel，确认只有一个融合 kernel launch，并记录 AIC、AIV、MTE2/MTE3 和同步事件。
3. 排除首次初始化和编译时间，固定 warm-up、sample、stream、设备号和同步点。
4. 计算平均标杆/平均融合时延，并保存逐 case 明细；只有大于 1.1 才报告性能达标。

Profiler 结果用于确认三段计算在同一 launch 内、数据搬运和 Cube/Vector 流水有效，不以 kernel 名称猜测重叠结论。

### 兼容性分析

这是 CATLASS 新增融合样例，不修改现有 Matmul API 和已有 kernel 行为。公开调用约定只增加 `matmul_layer_norm_matmul`，不改变 PyTorch/CATLASS 已有接口。代码通过模板参数隔离 Ascend 950 特化，通用 shape 校验、golden 和测试生成逻辑保持独立，便于后续扩展其他 Ascend 架构；其他架构在没有对应 Cube/MIX 能力时明确返回不支持，不静默回退到三个独立 kernel。

## 交付检查清单

- [ ] 任务 CSV 116 条 Optest 精度全部通过。
- [ ] ATK 泛化测试不少于 200 条且全部精度通过。
- [ ] `msprof op` 记录全部 116 条性能数据，平均标杆/融合时延大于 1.1。
- [ ] README 含接口、布局、shape、编译、运行和限制说明。
- [ ] 单次 MIX kernel launch，mean/variance 不出公开接口。
- [ ] PR 不包含二进制、编译中间产物、临时脚本、模型权重和个人敏感信息。
- [ ] 验收包包含测试用例、日志/截图、性能数据和测试步骤；代码 PR 只包含任务书规定的源代码和必要测试接入件。

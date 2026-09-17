# MatmulLayerNormMatmul 算子设计文档

## 需求背景（required）

### 需求来源

本需求来源于 CANN 2026 年社区任务 `MatmulLayerNormMatmul`。目标是在 Ascend 950 上，使用 Ascend C 与 CATLASS 实现一个单次 kernel launch 的融合算子：

```text
C0      = A0 @ B0
C0_norm = LayerNorm(C0, gamma, beta, eps=1e-6)
C1      = C0_norm @ B1
```

设计和实现基线为 CATLASS `master`，设计评审交付到 `cann-competitions`，代码交付到 `cann/catlass`。

### 背景介绍

#### 业务与性能背景

未融合实现需要依次启动 `torch.mm`、`F.layer_norm`、`torch.mm`。除三次 kernel launch 外，第一段 Matmul 结果和 LayerNorm 结果都需要写入并再次读取 GM。任务测试集覆盖 116 个 shape，维度范围如下：

| 维度 | 取值 |
| --- | --- |
| `M0` | `128, 512, 1024, 2048` |
| `K0` | `768, 2048, 4096, 8192` |
| `N0` | `2048, 3072, 4096, 8192` |
| `M1` | `768, 2048, 4096` |

标杆时延范围为 `19.005 us ~ 1188.58 us`。任务要求融合实现相对小算子拼接达到 `1.1x`，因此主路径必须同时消除中间 GM 流量和额外 launch 开销。

#### 现有 CATLASS 能力

CATLASS 主干已具备以下 Ascend 950 基础能力：

| 能力 | 复用位置 | 用途 |
| --- | --- | --- |
| Matmul FP16、FP32 累加 | `gemm/block`、`gemm/tile` | 两段 Matmul |
| `L0C -> UB` | `copy_l0c_to_ub.hpp`、callback BlockMmad | 将 Matmul0 结果交给 AIV |
| `UB -> L1(zN)` | `copy_ub_to_l1_tla.hpp` | 将 C0/C0_norm 保存在片上 |
| `L1 -> UB` 硬件接口 | CANN `DataCopyL1ToUB` | 两遍 LayerNorm 重新读取 C0 |
| AIC/AIV 跨核同步 | `cross_core_sync.hpp`、样例 80 | MIX kernel 阶段握手 |
| L1 resident Matmul 思路 | Flash Attention PV、full-load-A 样例 | Matmul1 直接消费 L1 左矩阵 |

当前缺口不是基础矩阵乘，而是两个可复用的窄接口：

1. CATLASS TLA 层尚未封装 `L1(zN) -> UB(RowMajor)`；
2. 通用 BlockMmad 默认从 GM 装载 A，缺少“调用者已提供 L1 A tensor”的入口。

#### 外部接口

| 参数 | 方向 | dtype | shape | 布局 |
| --- | --- | --- | --- | --- |
| `A0` | 输入 | FP16 | `(M0, K0)` | RowMajor |
| `B0` | 输入 | FP16 | `(K0, N0)` | ColumnMajor |
| `B1` | 输入 | FP16 | `(N0, M1)` | ColumnMajor |
| `gamma` | 输入 | FP32 | `(N0,)` | 连续 |
| `beta` | 输入 | FP32 | `(N0,)` | 连续 |
| `C1` | 输出 | FP16 | `(M0, M1)` | RowMajor |

`epsilon` 固定为 `1e-6`。`C0`、`C0_norm`、mean、variance 和 invstd 都不作为外部输出。

## 需求分析（required）

### 需求描述

在 Ascend 950 上实现 `Matmul -> LayerNorm -> Matmul` 融合算子，满足：

1. 一次 MIX kernel launch 完成全部计算；
2. `C0` 和 `C0_norm` 不写入 GM；
3. mean、variance、invstd 由 kernel 片上 scratch 管理；
4. 官方 116 个 case 和不少于 200 个 ATK 泛化 case 精度通过；
5. 相对 PyTorch 小算子拼接达到任务要求的性能收益；
6. example、Optest、ATK、README、设计文档和自验证报告完整可复现。

### 需求拆解

| 编号 | 子需求 | 设计响应 | 验证证据 |
| --- | --- | --- | --- |
| R1 | 固定 dtype | 矩阵 FP16，LN 参数与统计 FP32，输出 FP16 | Torch adapter 负向测试、pytest dtype 断言 |
| R2 | 固定布局 | A0/C1 RowMajor，B0/B1 ColumnMajor | stride 校验和非对称数据 golden |
| R3 | 单 launch | 一个 `__mix__` kernel，AIC/AIV 配对执行 | msprof kernel 列表 |
| R4 | 无中间 GM | C0/C0_norm 仅驻留 L1/UB | 代码审查、profile memory 轨迹 |
| R5 | 行级 LN | row block 完整覆盖 N0，FP32 两遍统计 | 常量行、近零方差、随机测试 |
| R6 | 低耦合 | 仅下沉两个通用数据通路组件，调度留在 experimental | 公共接口审查、既有回归 |
| R7 | 精度 | 中间 cast 与 PyTorch dtype 传播一致 | 官方 CSV + ATK + 误差报告 |
| R8 | 性能 | L1 复用、双缓冲、静态 plan 分档 | 116 case msprof 数据 |
| R9 | 可维护性 | 参数、资源、同步、测试按层拆分 | clang-format、build、pytest |

### 非目标

- 不支持 FP32/BF16 矩阵输入；
- 不增加可配置 epsilon；
- 不引入 GM partial-C1 或 C0 workspace 作为主路径；
- 不修改已有 Matmul API 的语义；
- 不把任务专属 plan selector 放入 CATLASS 通用层。

## 详细设计（required）

### 算子分析

#### 数学公式

```text
C0[m,n] = sum_k A0[m,k] * B0[k,n]

mean[m] = sum_n C0[m,n] / N0
var[m]  = sum_n (C0[m,n] - mean[m])^2 / N0

C0_norm[m,n] = (C0[m,n] - mean[m])
                 * rsqrt(max(var[m], 0) + 1e-6)
                 * gamma[n] + beta[n]

C1[m,p] = sum_n C0_norm[m,n] * B1[n,p]
```

方差使用总体方差，分母为 `N0`。

#### dtype 传播

| 阶段 | 存储 dtype | 计算 dtype |
| --- | --- | --- |
| Matmul0 | 输入 FP16，C0 FP16 | Cube FP32 accumulate，FixPipe cast FP16 |
| mean/variance | C0 FP16 | FP32 |
| affine | C0_norm FP16 | FP32 normalize/affine 后 cast FP16 |
| Matmul1 | 输入 FP16，C1 FP16 | Cube FP32 accumulate，FixPipe cast FP16 |

编码前先运行 smoke case 锁定当前 torch_npu 的 `mm -> layer_norm -> mm` dtype 传播。若 golden 的中间 dtype 与上表不同，先更新设计和资源预算，不以放宽容差替代。

#### 数值稳定性

默认不使用 `E[x^2] - E[x]^2`，因为均值较大、方差较小时会发生明显抵消。主路径保留 C0 并执行两遍中心方差：

```text
pass 1: sum(C0) -> mean
pass 2: sum((C0 - mean)^2) -> variance
pass 3: normalize + affine
```

每个 pass 的归约累加为 FP32；尾块 padding 不进入分子或分母；`rsqrt` 前将方差钳制到非负。

### 算子实现

#### 总体方案

每个任务拥有一个 `M` 行块，并完整覆盖该行块的 `N0` 维。完整 C0 row block 以 FP16/zN 形式驻留 L1：

```text
Phase A
GM A0/B0 -> AIC Matmul0 -> L0C -> UB
                              AIV: FP32 row sum + UB->L1 保存 C0

Phase B
L1 C0 -> AIV UB -> FP32 centered variance

Phase C
L1 C0 -> AIV UB -> normalize/affine -> FP16 -> 覆盖 L1 为 C0_norm

Phase D
L1 C0_norm + GM B1 -> AIC Matmul1 -> GM C1
```

该方案具有以下性质：

- 每行统计在一个任务内完成，无跨核归约；
- 中间矩阵不进入 GM；
- Matmul1 的多个 M1 tile 复用同一个 L1 C0_norm；
- 不改变 LayerNorm 中间 FP16 落盘语义；
- 每个任务写互斥的 C1 行/列区域，无 atomic。

#### 备选方案

| 方案 | 状态 | 说明 |
| --- | --- | --- |
| L1 完整行块 + 两遍方差 | 默认 | 精度语义清晰，资源可预算 |
| UB 完整行块 | 小 N0 候选 | 可减少 L1 往返，但 BM 受 248 KiB 限制 |
| 流式 Welford + Matmul0 重算 | 回退候选 | 精度稳定但 Matmul0 约翻倍，仅在特定 shape profiling 后保留 |
| `sum/sumsq` 单遍 | 禁止默认启用 | 必须先通过全部精度门禁和对消压力用例 |
| 代数分解 LayerNorm 后再 Matmul | 研究候选 | 元素级 FP16 舍入语义不同，不能未经精度证明进入默认路径 |
| C0/C0_norm 写 GM | 不采用 | 违反任务的中间显存目标 |

#### 文件与组件边界

计划代码结构：

```text
experimental/matmul/ascend950_matmul_layer_norm_matmul/
  CMakeLists.txt
  README.md
  matmul_layer_norm_matmul.cpp
  matmul_layer_norm_matmul_kernel.hpp
  matmul_layer_norm_matmul_plan.hpp
  gen_data.py

include/catlass/epilogue/tile/
  copy_l1_to_ub_tla.hpp

include/catlass/gemm/block/
  block_mmad_preloaded_l1a_tla.hpp

tests/optest/kernels/matmul_layer_norm_matmul/
  CMakeLists.txt
  matmul_layer_norm_matmul.cpp
  matmul_layer_norm_matmul_impl.cpp

tests/optest/src/include/template/
  matmul_layer_norm_matmul.h

tests/optest/torch_catlass/ops/
  matmul_layer_norm_matmul.py

tests/optest/tests/
  test_matmul_layer_norm_matmul.py

tests/optest/atk/matmul_layer_norm_matmul/
  matmul_layer_norm_matmul.yaml
  generator_matmul_layer_norm_matmul.py
  execute_matmul_layer_norm_matmul.py
  node.yaml
```

公共组件只表达数据通路，不包含本算子的 shape、epsilon、gamma/beta 或 plan 逻辑：

1. `CopyL1ToUbTla`：限定 Ascend 950，支持 L1 zN 到 UB RowMajor，带静态 layout 检查与尾块参数；
2. `BlockMmadPreloadedL1A`：调用者提供 L1 A tensor，组件只负责 B 的 GM->L1、L1->L0、Mmad 和最终 store。

其余 Stage 编排、LayerNorm 和 plan selector 全部保留在 experimental kernel 中。

#### Host/Launcher 设计

运行时参数：

```cpp
struct MatmulLayerNormMatmulParams {
    uint32_t m0;
    uint32_t k0;
    uint32_t n0;
    uint32_t m1;
    std::array<uint8_t*, 5> inputAddr;  // A0, B0, B1, gamma, beta
    std::array<uint8_t*, 1> outputAddr; // C1
};
```

Torch adapter 负责：

1. 检查五个输入在同一 NPU device；
2. 检查 dtype、rank、shape 关系；
3. 校验 A0/B0/B1 的逻辑 layout/stride；
4. 检查所有维度为正且可安全转换为 `uint32_t`；
5. 分配 `(M0, M1)` FP16 RowMajor 输出；
6. 使用 `tensor.data_ptr()` 获取逻辑首元素；
7. 从 `GetCurrentNPUArch()` 确认 3510；
8. 根据 shape 和实际 AIC core 数选择静态 plan；
9. 启动一次 MIX kernel。

不分配用户 GM workspace，`GetWorkspaceSize()` 返回 0。

#### 静态 plan 与调度

静态 plan 只固化 TileShape、stage 数、row block 大小和 replica 上限。初始候选：

| Plan | N0 | BM | BN0 | BK0 | BN1 | BK1 | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P0 | `<=2048` | 64 | 256 | 128 | 256 | 128 | 常规吞吐 |
| P1 | `3072` | 32 | 256 | 128 | 256 | 128 | 平衡 L1/并行度 |
| P2 | `4096` | 32 | 256 | 128 | 256 | 128 | 256 KiB C0 cache |
| P3 | `8192` | 16 | 256 | 128 | 256 | 128 | 大 N0 |
| PT | 任意 | 16 | 128 | 128 | 128 | 128 | 尾块/保守路径 |

这些是编译和首轮 profiling 起点，不是最终性能结论。最终保留 plan 必须由 116 case 实测支持。

基础任务沿 M0 分块：

```text
rowBlocks = ceil(M0 / BM)
```

当 `rowBlocks` 显著少于 AIC core 数时，允许沿 M1 列块复制 row task：

```text
replica <= min(ceil(aicCoreNum / rowBlocks), ceil(M1 / BN1))
```

每个 replica 独立生成本行块的 C0_norm，只负责互斥的 M1 列块。是否复制由成本模型和实测共同决定：

```text
F0 = 2 * BM * K0 * N0
F1 = 2 * BM * N0 * M1
estimatedWork(replica) = replica * F0 + F1
```

`K0` 大或 Matmul0 占主导时保持 `replica=1`。

#### 片上资源预算

Ascend 950 资源：L1 512 KiB、UB 248 KiB、L0A 64 KiB、L0B 64 KiB、L0C 256 KiB。

定义：

```text
CACHE_BYTES = BM * RoundUp(N0, 16) * sizeof(half)
```

Phase A：

```text
L1_A = CACHE_BYTES
     + L1A_STAGES * BM * BK0 * sizeof(half)
     + L1B_STAGES * BK0 * BN0 * sizeof(half)
     + guard
```

Phase D：

```text
L1_D = CACHE_BYTES
     + L1B_STAGES * BK1 * BN1 * sizeof(half)
     + guard
```

使用 `L1A_STAGES=1`、`L1B_STAGES=2`、`guard=32 KiB` 的初步估算：

| Plan | cache | Phase A | Phase D |
| --- | ---: | ---: | ---: |
| P0 | 256 KiB | 432 KiB | 416 KiB |
| P1 | 192 KiB | 360 KiB | 352 KiB |
| P2 | 256 KiB | 424 KiB | 416 KiB |
| P3 | 256 KiB | 420 KiB | 416 KiB |

UB 需要容纳：

```text
2 * BM_VEC * BN0 * sizeof(half)   // tile ping-pong
+    BM_VEC * BN0 * sizeof(float) // FP32 compute
+ 2 * BN0 * sizeof(float)         // gamma/beta
+ 4 * BM * sizeof(float)          // sum/mean/var/invstd
+ guard
```

L0C：

```text
max(BM * BN0, BM * BN1) * sizeof(float) <= 256 KiB
```

实现中以组件真实静态常量为准并增加 `static_assert`；上述估算不能替代编译期资源检查。

#### Kernel 阶段

##### Init

- 建立所有 GM tensor 与 TLA layout；
- 按静态 offset 划分 L1/UB/L0；
- 初始化 stage ready/idle flag；
- 计算 row block、actual M、replica 和负责的 M1 tile；
- 无有效任务的逻辑块不进入需要配对的同步循环。

##### Phase A：Matmul0 + mean + C0 cache

AIC：

1. 按 `BK0/BN0` 完成 Matmul0；
2. FP32 L0C 通过 FixPipe 转 FP16 RowMajor 到 UB；
3. 设置 stage ready flag；
4. 复用 stage 前等待 AIV idle。

AIV：

1. 等待 stage ready；
2. 将 FP16 C0 tile 转 FP32 并累计每行 sum；
3. 将 FP16 C0 经 `CopyUbToL1Tla` 写入 L1 zN cache；
4. 设置 stage idle；
5. 全部 N0 tile 后计算 mean。

##### Phase B：centered variance

AIV 使用新增 `CopyL1ToUbTla` 逐 tile 读取 C0：

```text
centered = float(C0) - mean[row]
varSum  += reduce_sum(centered * centered)
variance = max(varSum / N0, 0)
invstd   = rsqrt(variance + 1e-6)
```

##### Phase C：normalize + affine

AIV 再次从 L1 读取 C0：

```text
y = (float(C0) - mean[row]) * invstd[row]
y = y * gamma[col] + beta[col]
```

结果 cast FP16，并通过 `CopyUbToL1Tla` 覆盖原 cache。Phase C 完成后 cache 的语义变为 C0_norm。

##### Phase D：Matmul1

AIC 通过 `BlockMmadPreloadedL1A` 直接消费 L1 C0_norm：

1. C0_norm L1 tile 不再执行 GM->L1；
2. B1 仍从 GM 双缓冲搬入 L1；
3. 沿 N0 累加 FP32 L0C；
4. FixPipe 转 FP16，只写最终 C1。

#### 同步协议

同步遵循单生产者/单消费者 stage 状态机：

```text
IDLE -> AIC writes UB -> READY -> AIV consumes -> IDLE
```

约束：

- flag id 由单一头文件集中定义，不与现有组件内部 id 重叠；
- AIC/AIV 两侧每个 stage 的 set/wait 次数完全一致；
- replica、尾块和无效任务不改变配对次数；
- kernel 退出前 drain BlockMmad 并执行必要 barrier；
- debug build 可记录 task/stage，交付 build 不保留 device printf。

#### 尾块

- M0/N0/M1/K0 均按 `CeilDiv` 和 actual shape 处理；
- FP16 搬运遵守 32B 对齐，必要时使用安全 padding copy；
- LayerNorm 分母始终为逻辑 `N0`，padding 值不得参与统计；
- C1 尾列只写 actual M1，禁止越界；
- L1 zN cache 的物理 padding 在复用前显式初始化或掩码。

### 支持硬件

| 芯片 | 支持 |
| --- | --- |
| Ascend 950 / arch 3510 | 是 |
| Atlas A2/A3 / arch 2201 | 否 |

### 算子约束限制

任务书声明“无”额外 shape 约束。实现仍要求外部契约满足：

- dtype、rank、shape 和布局与参数表一致；
- `gamma.numel() == beta.numel() == N0`；
- 维度为正且地址范围、乘法字节数无整数溢出；
- `epsilon` 固定为 `1e-6`。

功能路径必须覆盖非对齐尾块；性能目标只对官方测试集承诺。

## 可维可测分析

### 精度与性能标准

| 验收项 | 标准 |
| --- | --- |
| 功能 | 官方 116 case 和补充边界 case 全部通过 |
| 精度 | 按生态算子开源精度标准执行；不在测试内自定义放宽阈值 |
| 泛化 | ATK 至少 200 个合法 case，包含 bug-hunting 场景 |
| 性能 | `baseline_us / kernel_us > 1.1`，按任务最终聚合口径报告 |
| 融合 | msprof 显示一个目标 kernel；无 C0/C0_norm GM 中转 |

### Golden

```python
c0 = torch.mm(a0, b0)
c0_norm = torch.nn.functional.layer_norm(
    c0, (n0,), weight=gamma, bias=beta, eps=1e-6
)
golden = torch.mm(c0_norm, b1)
```

同时记录 `c0/c0_norm/golden` dtype，防止框架升级改变口径。

### Optest

Optest 覆盖：

- 官方 CSV 116 case 参数化；
- shape/dtype/device 断言；
- `@only_on_3510` 架构隔离；
- gamma=1/beta=0、gamma=0、随机 gamma/beta；
- 常量行、近零方差、正负随机、有限大值；
- M/N/K/M1 尾块；
- 错误 dtype/rank/shape/stride/device 负向用例。

无 3510 时只跳过设备执行，不跳过编译、导入和 ABI 检查。

### ATK

ATK 四件套以 Optest 的 wrapper 顺序和 golden 为唯一事实来源：

- yaml 声明五个 tensor 输入；
- generator 修正 `(M0,K0,N0,M1)` 的跨输入 shape 关系和内存上限；
- execute 的 CPU/NPU 分支显式分离，NPU 分支惰性导入 `torch_catlass`；
- 首选 `mixed_tolerance_bm`；仅当安装版本不存在时回退 `single_bm`；
- 生成不少于 200 个 case，覆盖合法边界、尾块、近零方差和数值范围压力。

### 公共组件测试

新增公共组件必须有独立测试：

1. `CopyL1ToUbTla`：多种 M/N、尾块、zN stride、双 sub-block；
2. `BlockMmadPreloadedL1A`：与普通 GM A Matmul 对比，覆盖多个 K loop 和 B 尾块；
3. 既有 `CopyUbToL1Tla` 与 Matmul 回归不受影响。

### 构建与静态检查

```bash
bash scripts/build.sh ascend950_matmul_layer_norm_matmul
cd tests/optest
bash build.sh
python -c "import torch_catlass"
pytest tests/test_matmul_layer_norm_matmul.py -v -s
```

并执行 clang-format、头文件自包含检查、重复符号/未使用代码检查和干净构建。

### 性能测试

每个官方 case 预热后使用 `msprof op`，输出目录按 case 隔离。记录：

```text
idx,M0,K0,N0,M1,baseline_us,kernel_us,speedup,
plan,BM,BN0,BK0,BN1,BK1,replica,precision_pass
```

分析 AIC/AIV、MTE2/MTE3/FixPipe 占比、跨核等待、L1/UB 占用、长尾和 launch 开销。任何 TileShape、stage 或 replica 调整都保留前后数据。

### 兼容性分析

本算子是新增 experimental 样例：

- 不替换既有 API；
- 公共新增组件使用新类型名和严格模板约束；
- Optest 新增独立参数结构和 adapter，不扩张 `MatmulParams` 的含义；
- 架构限制显式为 3510；
- 公共组件合入前运行现有 Ascend 950 Matmul/FA 回归。

## 风险与应对

| 风险 | 触发信号 | 应对 |
| --- | --- | --- |
| `L1 -> UB` layout 转换错误 | stride case 大面积错误 | 独立 tile UT，先验证 zN/RowMajor 映射 |
| L1 资源挤压流水 | static assert 或 MTE 空泡 | 降 BM/BK/BN 或 stage，禁止溢出后静默运行 |
| 小 M0 并行度不足 | AIC 大量空闲 | 成本模型选择小 BM 或 replica，实测后决定 |
| LayerNorm AIV 成瓶颈 | AIC 等待 AIV | 调整 BN0、归约策略和 Phase C/D 流水 |
| FP16 中间语义漂移 | smoke dtype 不符 | 暂停实现并更新 dtype 路径 |
| 尾块污染统计 | 对齐 case 过、尾块失败 | actual N 掩码，分母固定逻辑 N0 |
| flag 死锁 | timeout | 最小 stage 验证，统一 flag 表，逐步开启流水 |
| 19 us 小 case 不达标 | launch/初始化占比高 | 精简 selector、初始化和同步，检查预热口径 |
| 公共组件耦合过高 | 组件出现 gamma/shape plan 语义 | 退回 experimental 私有实现后再抽象 |

## 分阶段交付与 PR 计划

### PR 1：设计评审

- 仅包含本设计文档；
- 目标仓库：`cann/cann-competitions`；
- 不包含性能结论或未经验证的“已通过”声明；
- 评审重点：L1 双向通路、资源预算、公共组件边界、精度语义。

### PR 2：功能骨架与公共组件

- experimental 目录、参数与 plan；
- `CopyL1ToUbTla`、`BlockMmadPreloadedL1A`；
- 最小 smoke case 与组件 UT；
- 编译和单 case 精度通过。

### PR 3：Optest/ATK 与泛化

- 完整 wrapper/ABI；
- 官方 116 case；
- ATK >= 200 case；
- README 与自验证步骤。

### PR 4：性能完善

- 基于 msprof 的 plan 精简和流水优化；
- 性能数据放在 PR 描述/自验证报告，不提交临时脚本、二进制或 profile 原始目录；
- 完整 bug 检查和回归。

## 参考资料

1. MatmulLayerNormMatmul 社区任务书。
2. CATLASS 创新样例开发流程指南。
3. `examples/64_ascend950_matmul_evg`。
4. `examples/73_ascend950_matmul_full_loadA`。
5. `examples/80_ascend950_grouped_matmul_slice_m_gelu`。
6. `examples/49_ascend950_flash_attention_infer`。
7. CATLASS Optest README 与仓库技能说明。
8. 生态算子开源精度标准。

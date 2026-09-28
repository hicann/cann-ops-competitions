# 需求背景（required）

## 需求来源

CANN 2026 年 9 月社区任务《kv_cache_turbo_quant 算子开发》。目标仓为 [ops-transformer/experimental/attention](https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention)。本设计对应算子 `KvCacheTurboQuant`。

## 背景介绍

vllm-ascend 现有 KV cache 压缩只有约 2 倍的 per-channel INT8，以及用于稀疏检索、并不压缩存储的 LSH。TurboQuant 用一次正交旋转加标量量化，再用量化残差的 1-bit 随机投影把误差补回来，适合在线压缩标准 MHA/GQA 的 K/V。

论文里的 3.5 bit/channel 依赖 outlier 通道分组，首版不实现。本算子固定为统一 `mse_bits` 主编码加 1-bit QJL。`mse_bits = 3`、`head_dim = 128` 时物理存储是 4.25 bit/channel：主码 48 字节、QJL 16 字节、范数 2 字节、残差范数 2 字节，合计 68 字节，相对 BF16 的 256 字节约 3.76 倍。

本算子不覆盖 MLA。MLA 缓存的是低秩 `cKV` 和独立 RoPE key，不能把二者拼成一个普通 head 后直接套用本接口。

# 需求分析（required）

## 需求描述

用 Ascend C 实现在线 encode：

```text
x_norm = ||x||_2
u      = x / x_norm                         # 范数为 0 时结果为 0
y      = H @ u                              # 即 unit @ H.T
idx    = scalar_quantize(y, mse_bits)       # 中点边界，严格大于，LSB-first 打包
y_hat  = dequant(idx)
r      = y - y_hat
gamma  = x_norm * ||r||_2
qjl    = sign(S @ r / ||r||_2)              # 大于等于 0 记 1，1-bit 打包
```

输入 `kv_vectors` 为 BF16 `[num_tokens, num_kv_heads, 128]`，`rotation_matrix` 与 `qjl_matrix` 为 FP32。`mse_bits` 支持 2、3、4，默认 3。`num_kv_heads` 为 4 到 32。

## 需求拆解

1. 编码结果与任务 golden 的打包格式一致，decode 与 2-bit prefill 达到逐字节一致。
2. 用编码结果重构后，相对 MSE 小于 0.05，内积相对误差 95 分位小于 0.1。
3. 五档 shape 的 encode 耗时不超过给定 PyTorch eager 基线的十分之一。
4. 工程目录对齐 ops-transformer experimental attention：`op_host`、`op_kernel`、`docs`、`examples`、`README`。
5. 同时注册 `ascend910b` 与 `ascend910_93`。内存占用不作为验收项。

# 详细设计（required）

## 算子分析

### 数学公式

Lloyd-Max 码本与 golden.py 使用同一组小数。边界是相邻码字的中点。比较为严格大于，所以落在边界上的值归入左侧码字。

2-bit 码字：`-0.1335033178, -0.04002048075, +0.04002048075, +0.1335033178`。

3-bit 与 4-bit 码字同样取自 golden，边界在核内按中点预计算，避免设备上重复做除法。

打包：每 8 个通道为一组。`mse_bits = b` 时这一组是一个 `8b` 位的整数，通道 `lane` 占据 `[lane*b, lane*b+b)`。再按字节从低到高拆开。因此 2/3/4 bit 每 head 的 `quant_idx` 长度是 32/48/64 字节，`quant_qjl` 固定 16 字节。

重构（只用于精度验收，不是 encode 输出）：

```text
signs  = unpack(qjl) mapped to {-1, +1}
dir    = S^T @ signs
r_hat  = ||r|| * dir / ||dir||
x_hat  = x_norm * H^T @ (y_hat + r_hat)
```

`||r|| = gamma / x_norm`。把残差方向重新归一化后，3-bit、Hadamard 旋转下相对 MSE 均值约 0.025、最大值约 0.048，内积相对误差 95 分位约 0.040。

### 支持数据类型

| 参数 | 数据类型 | shape |
| --- | --- | --- |
| kv_vectors | bfloat16 | `[num_tokens, num_kv_heads, 128]` |
| rotation_matrix | float32 | `[128, 128]` |
| qjl_matrix | float32 | `[128, 128]` |
| mse_bits | int，默认 3 | 标量 |
| quant_idx | uint8 | `[num_tokens, num_kv_heads, 16 * mse_bits]` |
| quant_qjl | uint8 | `[num_tokens, num_kv_heads, 16]` |
| quant_norm | bfloat16 | `[num_tokens, num_kv_heads]` |
| quant_gamma | bfloat16 | `[num_tokens, num_kv_heads]` |

不支持广播。KV 在逻辑上是二维行 `[rows, 128]`，`rows = num_tokens * num_kv_heads`。

## 算子实现

### 实现方案

一次 encode 顺序提交到同一条 stream，耗时统计覆盖下面全部步骤：

```text
AIV   kvtq_cast_fp32          BF16 -> FP32
aclnn LinalgVectorNorm + Div  x_norm，单位向量 u
AIC   mm_fp32_transb          Y = U @ H.T
AIV   kvtq_quantize           Lloyd-Max、打包 idx、写出残差 r = y - y_hat
aclnn LinalgVectorNorm + Div  残差单位向量
aclnn Mul                     gamma = x_norm * ||r||
AIV   kvtq_store_bf16         norm、gamma 按 ties-to-even 写成 BF16
AIC   mm_fp32_transb          P = r_unit @ S.T
AIV   kvtq_sign               P >= 0 的比较掩码直接就是 QJL 比特
```

QJL 作用在残差单位向量上，与 `golden.py` 一致。任务书步骤 8 写成 `sign(S @ r)`，`case.json` 的 `calc_expect_func` 用的是单位残差，验收以 golden 为准。

两次 GEMM 共用一个 AIC kernel。`M` 不同，host 每次编码只生成一份 tiling：`N = K = 128`，`TransB = true`，`M = rows`。

没有把各段合成一次 mix kernel。在直调路径里，`dav-c220` mix 启动不会让 AIC 侧进入 `SyncAll`，AIV 侧 `SyncAll<false>()` 返回 507015，输出保持为 0。Cube-only 的 `MatmulImpl` 与 Vector kernel 分开启动没有这个同步问题。16384×128×128 的 FP32 GEMM 在同一输入上与 torch matmul 逐位一致；相对 float64 点积仍约有 7e-7 的舍入差，但 golden 走的是 torch matmul，这条差不进入验收比较。

#### host 侧设计

`op_host/kv_cache_turbo_quant_def.cpp` 注册三输入四输出和 `mse_bits`，AICore 配置同时加入 `ascend910b`、`ascend910_93`。

`kv_cache_turbo_quant_infershape.cpp` 由 KV 的前两维和 `mse_bits` 推出打包长度。`head_dim` 固定按 128 计算字节数。

`kv_cache_turbo_quant_tiling.cpp` 校验：

- KV 为三维，旋转和投影为二维；
- `head_dim = qjl_dim = 128`；
- `num_kv_heads ∈ [4, 32]`；
- `mse_bits ∈ {2, 3, 4}`。

分核：

- Vector：从 AIV 核数开始，若每核不足 8 行就把核数减半。8 行的 decode 使用 1 个 Vector 核。单行一块时 `BlockReduceSum` 的打包结果不可靠。
- Cube：从 AIC 核数向下找最大的 `c`，使 `rows % c == 0` 且每核行数是 128 的倍数或小于 128。`rows < 128` 时用 1 个 Cube 核。tiling 库不会按 `blockIdx` 自动错开 M，kernel 必须自己计算 `mOff`。

workspace：4 × `rows` × 128 × 4 字节的中间 FP32（单位向量、旋转结果、残差、投影），加三份 FP32 行向量（`x_norm`、`||r||`、`gamma`），加 32MB GEMM workspace。单位向量缓冲在第二段 GEMM 前被残差单位向量覆盖，残差缓冲覆盖原始 FP32 KV，因此四块矩阵足够。

直调样例 `examples/test_aclnn_kv_cache_turbo_quant.cpp` 用 `MultiCoreMatmulTiling` 生成 `TCubeTiling`，再把 `usedCoreNum` 覆盖成上面的 Cube 核数并重新拷到设备。覆盖必须发生在 `REGIST_MATMUL_OBJ` 读到 tiling 之前，样例在启动前完成这次拷贝。

#### kernel 侧设计

Vector 公共切分：

```text
base = rows / coreNum
rem  = rows % coreNum
mine = base + (blockIdx < rem)
begin = blockIdx * base + (blockIdx < rem ? blockIdx : rem)
```

`kvtq_cast_fp32`：BF16 转 FP32。单位向量不在这个 kernel 里做。`aclnnLinalgVectorNorm` 和 `aclnnDiv` 与 `torch.linalg.vector_norm` 走同一条实现，范数和除法逐位一致。Cube GEMM 在同一输入上和 torch matmul 逐位一致，因此旋转后的坐标和 QJL 投影不会在 Lloyd-Max 边界或 0 附近差出一个码字。

`kvtq_quantize`：tile 为 64 行。码字累加用比较掩码加 1。重构值按码字从质心表取值，不用边界间距累加，避免几个 ULP 的差在 0 附近翻转 QJL 符号。打包把码字乘以 `2^(lane*bits)` 的常量表，再 `BlockReduceSum`。2-bit 和 3-bit 的组内整数不超过 2^24-1，FP32 可以精确表示。4-bit 拆成低 4 通道和高 4 通道，高半字左移 16 位后相加。`float` 到 `int32` 使用 `CAST_RINT`。残差 `y - y_hat` 原样写出，再由同一套 vector_norm 和 div 得到残差单位向量。`gamma = x_norm * r_norm` 用 `aclnnMul`。BF16 写出用标量 ties-to-even，和 numpy bfloat16 一致。

`kvtq_sign`：`Compares` 的 `GE 0` 掩码按字节排布就是 LSB-first 的 1-bit QJL，直接 `DataCopyPad` 写出。

Cube `mm_fp32_transb`：`KERNEL_TYPE_AIC_ONLY`，编译宏 `ASCENDC_CUBE_ONLY`，架构 `dav-c220-cube`。每个核设置 `SetOrgShape(myM, 128, 128)`、`SetSingleShape(myM, 128, 128)`，A 从 `mOff * 128` 开始，B 以转置方式绑定整块 H 或 S。`op_kernel/kv_cache_turbo_quant_cube.cpp` 在未定义 `ASCENDC_CUBE_ONLY` 时不包含 Matmul 头文件，避免 Vector 编译单元误编 Cube 代码。

行间没有重叠，也没有跨核原子加。UB 上量化 tile 保持 64×128，五块 FP32 加上打包临时空间可以放进 192KB UB。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练/推理（ascend910b） | √ |
| Atlas A3 训练/推理（ascend910_93） | √ |
| Atlas 800I/800T A2 | √ |

实测平台是 Ascend910_93：24 个 Cube 核、48 个 Vector 核、UB 192KB。A2/A3 的 C220 核使用同一套 `dav-c220-vec` / `dav-c220-cube` 二进制。

## 算子约束限制

- 不支持 MLA，不支持混合 3.5-bit outlier 模式。
- `head_dim` 与 `qjl_dim` 固定 128。
- `num_kv_heads` 仅 4 到 32。
- `mse_bits` 仅 2、3、4。
- 不支持空 tensor，不支持非 ND、非连续输入由定义里的 `AutoContiguous` 在图模式下降维前处理。
- 编码接口不输出重构向量。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 编码一致性 | 五个性能 case，以及 `num_kv_heads` 为 4 和 32 的两个边界 case，idx、qjl、norm、gamma 与 `tests/golden.py`（torch_npu matmul + numpy bf16 round-to-nearest-even）失配均为 0 | case.json 阈值 idx/qjl atol 约 1e-5，norm/gamma 为 0 |
| 重构相对 MSE | 256 行、3-bit、Hadamard：均值 0.0255，最大 0.0482，均小于 0.05 | 任务书 3.2 |
| 内积相对误差 | `\|cos\| > 0.2` 的 2048 个样本，95 分位 0.0398，小于 0.1 | 任务书 3.2 |
| 性能 | 五档平均耗时 44.226 / 71.142 / 326.896 / 277.311 / 415.695 us，相对 eager 基线约 33.7×、21.8×、11.8×、13.0×、12.7×，均不低于 10× | 任务书 3.3 |
| 内存 | 不涉及 | 任务书 3.4 |

复现：在个人仓 `kv_cache_turbo_quant/` 或 ops-transformer 的 `experimental/attention/kv_cache_turbo_quant/` 执行 `bash examples/build.sh`，退出码为 0 且打印 `PASSED`。

## 交付地址

- 个人仓：https://gitcode.com/Wilbert_666/kv_cache_turbo_quant ，分支 `main`，算子目录 `kv_cache_turbo_quant/`。已邀请 `Ascend-CANN` 作为开发者。自测报告在该仓 `task_submission/`。
- 本设计文档 PR：https://gitcode.com/cann/cann-ops-competitions/merge_requests/1821
- 算子源码 PR：https://gitcode.com/cann/ops-transformer/merge_requests/12860 ，目录 `experimental/attention/kv_cache_turbo_quant`

## 兼容性分析

新算子，没有旧版本行为需要兼容。后续如果支持 MLA 或混合位宽，需要新增输入，不能改变现有四输出的排布。

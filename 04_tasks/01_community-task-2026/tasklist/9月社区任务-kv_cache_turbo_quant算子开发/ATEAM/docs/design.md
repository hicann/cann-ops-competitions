# 需求背景（required）

| 项目 | 内容 |
| --- | --- |
| 任务名称 | kv_cache_turbo_quant |
| TeamName | ATEAM |
| 设计文档路径 | `04_tasks/01_community-task-2026/tasklist/9月社区任务-kv_cache_turbo_quant算子开发/ATEAM/docs/design.md` |
| 目标代码仓 | `https://gitcode.com/cann/ops-transformer` |
| 实现目录 | `experimental/attention/kv_cache_turbo_quant/` |
| 基线提交 | `f70f61ec9`（分支 `feature/kv-cache-turbo-quant`） |
| 目标硬件 | Atlas 800T A2 |
| 对标接口 | `aclnnKvCacheTurboQuantGetWorkspaceSize` / `aclnnKvCacheTurboQuant` |

## 需求来源

本设计依据如下，冲突时按“任务书、算子原型 `op.json`、随任务 golden 与用例、目标仓当前实现”的顺序处理，并在文中列出待确认项：

1. 任务书：`kv_cache_turbo_quant.md`；
2. 算子原型：`op.json`，算子名 `KvCacheTurboQuant`；
3. 随任务用例：`case.json` 五个 shape；
4. 随任务参考：`golden.py` 的 `calc_expect_func`；
5. 当前实现：`experimental/attention/kv_cache_turbo_quant/` 的 OpDef、InferShape、Tiling 与 Kernel。

任务要求使用 Ascend C，在 Atlas 800T A2、CANN 9.1.0+ 上实现标准 MHA/GQA KV cache 的在线向量量化编码，合入位置为 `experimental/attention`。本任务不覆盖 MLA，也不实现解码算子。

任务书将 `rotation_matrix`、`qjl_matrix` 写成 float32 或 bfloat16，将 `quant_norm`、`quant_gamma` 写成 float16 或 bfloat16。原型 `op.json` 与当前 OpDef 固定为：向量 BF16，两张矩阵 FP32，两个标量 BF16，编码 UINT8。当前 ABI 沿用原型和 OpDef，不另建一套接口。

任务书步骤 8 将 QJL 写成 `sign(S @ r)`。随任务 golden 与当前 Kernel 先把残差归一化，再做 `sign(S @ r / ||r||)`。编码验收以 golden 为准；是否改回未归一化残差，待评审确认。

## 背景介绍

### 算子功能

对每个 KV head 向量做固定位宽编码。首版主编码是统一 MSE 标量量化，残差是 1-bit QJL。`mse_bits=3`、`head_dim=128`、`qjl_dim=128` 时，单个 head 的物理存储为：

| 组件 | bit/channel | 字节 |
| --- | ---: | ---: |
| 主量化 idx | 3 | 48 |
| QJL 符号 | 1 | 16 |
| 输入 L2 范数 | 0.125 | 2 |
| 原尺度残差范数 | 0.125 | 2 |
| 合计 | 4.25 | 68 |

相对 BF16 的 256 byte/head，存储比为 256/68。`mse_bits=2` 时 idx 为 32 byte，合计 52 byte；`mse_bits=4` 时 idx 为 64 byte，合计 84 byte。论文中的混合 3.5 bit/channel 依赖 channel 分组，不在本算子范围。

### 当前实现

目标仓 `experimental/attention` 中已有本算子目录。当前实现完成：

- OpDef 登记三输入、四输出和可选属性 `mse_bits`，默认 3；
- InferShape 按 bit 打包宽度推导输出 shape；
- Host 校验 head 维、KV head 数和 `mse_bits`，按向量数划分 AIV；
- Kernel 按 golden 完成归一化、旋转、主量化、残差范数和 QJL 打包；
- 测试侧有 ACLNN 调用程序和编码字节对照脚本。

当前实现没有解码 Kernel，也没有把重构相对 MSE、内积相对误差或相对 PyTorch eager 的加速比写成已测结果。

### 当前实现与本任务差距

| 项目 | 当前实现 | 本任务 |
| --- | --- | --- |
| 硬件 | OpDef 登记 `ascend910b`、`ascend910_93` | 验收硬件为 Atlas 800T A2 |
| 编码 | 已按 golden 实现 2/3/4 bit 主编码 + 1-bit QJL | 覆盖随任务五个用例 |
| 矩阵与标量 dtype | 矩阵 FP32，标量 BF16 | 与 `op.json` 一致；任务书中的 BF16 矩阵和 FP16 标量不在当前 ABI |
| QJL 输入 | 单位残差 | 与 golden 一致；与任务书未归一化写法待确认 |
| 重构误差、内积误差 | 无解码实现 | 任务书 3.2，待解码参考后验收 |
| 性能 | 有计时程序，无验收记录 | 相对任务书 PyTorch eager 基线 ≥ 10×，待测 |
| MLA、混合 3.5 bit | 未实现 | 明确不在首版范围 |

# 需求分析（required）

## 需求描述

在 `ops-transformer` 的 `experimental/attention/kv_cache_turbo_quant` 中提供 ACLNN 两段式编码算子。输入是 BF16 的 `[num_tokens, num_kv_heads, 128]` KV 向量、FP32 的 `[128, 128]` 旋转矩阵和 FP32 的 `[qjl_dim, 128]` 投影矩阵，属性 `mse_bits` 取 2、3 或 4。输出是打包后的 UINT8 主编码、UINT8 QJL 符号，以及 BF16 的输入范数和原尺度残差范数。格式为 ND。

本提交范围不含 MLA latent cache、混合位宽、解码算子、Python/ATen 入口，也不把 BF16 旋转矩阵或 FP16 标量作为当前 ABI。

## 需求拆解

1. 接口保持 `KvCacheTurboQuant` 与生成的 `aclnnKvCacheTurboQuantGetWorkspaceSize`、`aclnnKvCacheTurboQuant`，不新增第二套用户符号。
2. 支持任务书和 `op.json` 交集内的 dtype、ND、`head_dim=128`、`num_kv_heads∈[4,32]`、`mse_bits∈{2,3,4}`。
3. Host 对 rank、head 维、head 数、`qjl_dim` 和 `mse_bits` 非法输入返回失败。
4. Kernel 按向量分到 AIV，完成与 golden 一致的编码和 bit 打包。
5. 零范数向量有确定输出：输入范数为 0，旋转结果为 0，仍按 0 向量做主量化；残差范数为 0 时 QJL 全符号位为 1。
6. 五个随任务 shape 的编码输出对照 golden；任务书 3.2 的重构与内积指标待解码参考。
7. 性能按任务书五个场景，对照其 PyTorch eager 基线，门限为 ≥ 10×，本设计阶段不填写未实测耗时。
8. 新算子，不改变仓内其他 attention 算子的 ABI。

# 详细设计（required）

## 算子分析

### 编码语义

记单个 head 向量为 \(x\in\mathbb{R}^{128}\)，旋转矩阵存储为 \(H\in\mathbb{R}^{128\times 128}\)，投影矩阵存储为 \(S\in\mathbb{R}^{qjl\_dim\times 128}\)。当前实现与 golden 的计算是：

$$
\begin{aligned}
n &= \|x\|_2,\\
u &= \begin{cases} x/n & n>0\\ 0 & n=0\end{cases},\\
y &= u H^{\mathsf T},\\
c_d &= \#\{b_i \mid y_d > b_i\},\\
\hat y_d &= \mu_{c_d},\\
r &= y-\hat y,\\
\gamma &= n\|r\|_2,\\
q &= \operatorname{sign}(S r / \|r\|_2).
\end{aligned}
$$

\(y\) 的第 \(d\) 维是 \(u\) 与 \(H\) 第 \(d\) 行的点积。\(\mu\) 是对应位宽的质心，\(b_i\) 是相邻质心中点。比较使用严格大于，因此落在中点上的值归入较低档。\(\|r\|_2=0\) 时，投影按 0 处理，符号条件 `>= 0` 成立，对应 bit 置 1。

质心与 golden、Kernel 中的常量一致：2 bit 为 4 档，3 bit 为 8 档，4 bit 为 16 档。实现不在设备侧重新估计质心，也不检查 \(H\) 是否正交、\(S\) 是否服从高斯分布。这两张矩阵由调用方在推理期间保持不变。

### 输入输出与存储

| 名称 | 方向 | dtype | shape | format |
| --- | --- | --- | --- | --- |
| kv_vectors | 输入 | bfloat16 | `[num_tokens, num_kv_heads, 128]` | ND |
| rotation_matrix | 输入 | float32 | `[128, 128]` | ND |
| qjl_matrix | 输入 | float32 | `[qjl_dim, 128]` | ND |
| quant_idx | 输出 | uint8 | `[num_tokens, num_kv_heads, (128*mse_bits+7)/8]` | ND |
| quant_qjl | 输出 | uint8 | `[num_tokens, num_kv_heads, (qjl_dim+7)/8]` | ND |
| quant_norm | 输出 | bfloat16 | `[num_tokens, num_kv_heads]` | ND |
| quant_gamma | 输出 | bfloat16 | `[num_tokens, num_kv_heads]` | ND |
| mse_bits | 属性 | int，默认 3 | 标量 | — |

五个随任务用例的 `num_kv_heads=8`、`qjl_dim=128`：

| case | num_tokens | mse_bits | quant_idx 末维 | quant_qjl 末维 |
| --- | ---: | ---: | ---: | ---: |
| gqa_decode_b1 | 1 | 3 | 48 | 16 |
| gqa_decode_b64 | 64 | 3 | 48 | 16 |
| gqa_prefill_t2048 | 2048 | 3 | 48 | 16 |
| gqa_mse2_t2048 | 2048 | 2 | 32 | 16 |
| gqa_mse4_t2048 | 2048 | 4 | 64 | 16 |

主编码按每 8 个 channel 为一组。组内 lane 0 占据最低位，第 `lane` 个码左移 `lane * mse_bits`。该 32-bit 字再按小端拆成 `mse_bits` 个字节。QJL 每 8 个符号打成 1 个 uint8，lane 0 同样在最低位；`qjl_dim` 不是 8 的倍数时，末字节的空 lane 保持 0。输入三个 Tensor 在 OpDef 上设置了 `AutoContiguous`。

`quant_norm`、`quant_gamma` 在 Kernel 内由 FP32 舍入到 BF16 后写出。加载 BF16 时把 16 位放到 FP32 的高 16 位。

### workspace 与边界

Host 为 workspace 申请 16 MiB。Kernel 入口接收该指针，计算过程不读取它。

`num_tokens * num_kv_heads = 0` 时，Host 仍设置 1 个 block，该核的向量数为 0，循环不写输出。零向量按上一节公式处理，不单独 quick return。当前 Kernel 不对 NaN、Inf 做专门分支。重复索引不适用于本算子，编码下标是量化档位，不是稀疏下标。

## 算子实现

### 总体架构

```text
aclnnKvCacheTurboQuantGetWorkspaceSize / aclnnKvCacheTurboQuant
        |  编译期由 OpDef 的 aclnnSupport 生成，源码树不手写该 cpp
        v
InferShape + Tiling（op_host）
        v
AIV Kernel（op_kernel，按 mse_bits 模板）
```

没有 Python、ATen 或 TBE 层。Kernel 只使用 Vector，不分 Cube。

### Host

`op_host/kv_cache_turbo_quant_def.cpp` 登记输入输出，并同时加入 `ascend910b` 与 `ascend910_93` 配置。动态 shape 打开，动态 rank 关闭。

`kv_cache_turbo_quant_infershape.cpp` 要求 `kv_vectors` 为 3 维、`qjl_matrix` 为 2 维，再按 bit 数填写四个输出 shape。位宽、head 数和 head 维的业务约束放在 Tiling，不在 InferShape 重复拦截。

`kv_cache_turbo_quant_tiling.cpp` 的失败条件：

- 输入 rank 不是 `[3,2,2]`；
- `head_dim`、旋转矩阵两维、投影矩阵最后一维任一不是 128；
- `num_kv_heads` 不在 `[4,32]`，或 `qjl_dim<=0`；
- `mse_bits` 不是 2、3、4；
- 平台 AIV 核数为 0。

分核单位是向量，向量总数为 `num_tokens * num_kv_heads`。使用的核数是 AIV 核数与向量数的较小值，至少为 1。前 `tailVectors` 个核各多处理 1 个向量。TilingData 字段为 `vectorCount`、`qjlDim`、`mseBits`、`idxBytes`、`qjlBytes`、`vectorsPerCore`、`tailVectors`。

`kv_cache_turbo_quant_tiling_key.h` 声明模板参数 `mseBits`，取值列表为 2、3、4。Host 写入的 tiling key 是 `mse_bits-2`，即 0、1、2。源码注释约定该下标对应上述模板实参。Kernel 内不再按运行时数值切换质心表，质心由模板参数 `BITS` 选择。没有第二套手写分支。

### Kernel

`op_kernel/kv_cache_turbo_quant.cpp` 的入口是 `template <uint32_t mseBits>` 的 `__global__` 函数。`kv_cache_turbo_quant.h` 中 `KvCacheTurboQuantKernel<BITS>` 的 `Process` 对每个向量顺序执行：

1. 标量读入 128 个 BF16 并转成 FP32，同时累加平方和，写出 BF16 范数；
2. `Muls` 完成归一化；
3. 对旋转矩阵每一行做 128 维点积，逐维比较质心中点并保存残差；
4. 按 8 通道一组打包主编码；
5. 用向量乘加和 `WholeReduceSum` 计算残差平方和，写出 BF16 `gamma`；
6. 对投影矩阵每一行做点积，按符号打包 QJL。

点积在 UB 中完成：`Mul` 之后把高低 64 个数相加，再对 64 个数做 `WholeReduceSum`。`qjl_dim==128` 时，投影矩阵和旋转矩阵一样在每个核开始时 `DataCopy` 到 UB；其他正的 `qjl_dim` 逐元素从 GM 读取投影行。主编码打包和 QJL 打包使用标量 `SetValue` / `GetValue` 写回 GM。

UB 为每项单独分配 `TBuf<VECCALC>`：输入、旋转结果、残差、点积暂存各 128 个 FP32，规约暂存 8 个 FP32，旋转矩阵和投影矩阵各 \(128\times 128\) 个 FP32。没有 `TQue`，没有双缓冲，因此这里不写成 CopyIn / Compute / CopyOut 流水。已用接口包括 `DataCopy`、`Muls`、`Mul`、`Add`、`WholeReduceSum`、`PipeBarrier`、`GetValue`、`SetValue`。Kernel 不增加设备侧的正交性检查、查重或另一套量化 fallback。

### 目录与构建

```text
experimental/attention/kv_cache_turbo_quant/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── CMakeLists.txt
│   ├── kv_cache_turbo_quant_def.cpp
│   ├── kv_cache_turbo_quant_infershape.cpp
│   └── kv_cache_turbo_quant_tiling.cpp
└── op_kernel/
    ├── kv_cache_turbo_quant.cpp
    ├── kv_cache_turbo_quant.h
    ├── kv_cache_turbo_quant_tiling_data.h
    └── kv_cache_turbo_quant_tiling_key.h
```

`op_host/CMakeLists.txt` 使用 `add_modules_sources(OPTYPE kv_cache_turbo_quant ACLNNTYPE aclnn)`。`experimental/attention/CMakeLists.txt` 按子目录收集算子，本算子不替换其他算子的公共文件。编译入口与目录 README 一致：

```bash
bash build.sh --pkg --experimental --soc=ascend910b \
  --ops=kv_cache_turbo_quant -j8
```

`tests/` 是本地验证程序，不作为算子运行时源码。README 里写的 `tests/pytest` 与当前测试文件不一致；当前对照入口是 `tests/verify.py`。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |

OpDef 另外登记了 `ascend910_93`。该配置不在本任务验收范围，本文不把它写成已验证硬件。

## 算子约束限制

- 只编码标准 MHA/GQA 的 K 或 V 向量；不接受把 MLA 的 `[cKV, kR]` 当成一个普通 head。
- `head_dim` 固定 128；`num_kv_heads` 为 4 到 32；`mse_bits` 为 2、3、4。
- 旋转矩阵必须是 `[128,128]` FP32，投影矩阵必须是 `[qjl_dim,128]` FP32，`qjl_dim>0`。
- 当前不支持任务书文字中的 BF16 矩阵和 FP16 范数。
- 调用方保证矩阵在推理期间不变，并自行保证旋转正交、投影分布符合算法假设。
- 零向量和零残差按编码语义产生确定比特，不是跳过写出。
- 不支持动态 rank；输入由框架连续化。
- 不含解码、attention 融合、混合 3.5 bit 和 channel outlier 分组。
- workspace 当前固定申请 16 MiB，Kernel 不使用其内容。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 编码一致性 | `quant_idx`、`quant_qjl` 与 golden 逐字节一致；`quant_norm`、`quant_gamma` 与 golden 的 BF16 舍入结果一致 | 随任务 `golden.py`、`case.json` |
| 重构相对 MSE | `MSE / \|\|x\|\|^2 < 0.05` | 任务书 3.2；当前无解码实现，待测 |
| 内积相对误差 | `\|<q,x\_hat>-<q,x>\| / \|<q,x>\| < 0.1`，统计 95 分位 | 任务书 3.2；当前无解码实现，待测 |
| 性能 | 五个场景相对任务书给出的 PyTorch eager 耗时加速 ≥ 10× | 任务书 3.3、3.5 |
| 内存 | 不涉及 | 任务书 3.4 |

任务书给出的 eager 基线仅作为对照分母，单位 us：`gqa_decode_b1` 1491.676，`gqa_decode_b64` 1552.784，`gqa_prefill_t2048` 3855.422，`gqa_mse2_t2048` 3607.963，`gqa_mse4_t2048` 5286.502。本设计阶段不填写算子实测耗时或倍率。

## 测试设计

正向：五个随任务 shape，`num_kv_heads=8`，`head_dim=128`，`mse_bits` 覆盖 2、3、4，token 数覆盖 1、64、2048。对照 golden 的打包字节、BF16 范数和 gamma。

边界：`num_tokens=1` 已包含在正向用例中。零范数、`qjl_dim` 非 128、`num_kv_heads` 的 4 和 32 两端由公式和 Host 约束定义，对应用例待补充。

负向：Host 已拒绝错误 rank、非 128 的 head 维、越界 head 数和非法 `mse_bits`。这些失败路径的自动化用例待补充。

一致性：同一输入重复执行应得到相同编码。确定性记录待测。

回归：新算子，回归范围是仓内原有 experimental attention 算子仍能按原目标编译。无旧版 KvCacheTurboQuant 行为需要保持。

性能：与精度使用同一组五个 shape。计时程序位于 `tests/aclnn_kv_cache_turbo_quant_test.cpp`，先预热再取平均。正式结果需按任务书与上述 eager 基线计算比值。当前没有可归档的性能记录，不写通过。

`case.json` 对 UINT8 给了非零 atol。当前 `tests/verify.py` 对两份编码使用逐字节相等。验收时两份编码保持逐字节一致；范数和 gamma 按 BF16 舍入后的 golden 比较。

## 兼容性分析

新增：`experimental/attention` 下的新算子和对应 ACLNN 符号。不修改已有算子的原型、Tiling 或 Kernel。

架构：源码按 AIV 编写，OpDef 登记 `ascend910b` 与 `ascend910_93`。本任务只验收 Atlas 800T A2。`ascend910_93` 是否保留，待评审确认。

dtype / shape：当前固定 BF16 向量、FP32 矩阵、BF16 标量和 `head_dim=128`。任务书中更宽的 dtype 不在本次兼容承诺内。

并行贡献：目录独立，CMake 只追加本算子。与其他 experimental 算子的并行修改通过不同子目录隔离。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 任务书 QJL 写法与 golden 的单位残差不一致 | 当前 Kernel 保持与 golden 相同。改语义前先确认验收参照，不在未确认时同时保留两套编码 |
| tiling key `mse_bits-2` 与模板实参 2/3/4 的对应依赖构建系统展开 | 以 `kv_cache_turbo_quant_tiling_key.h` 和 Host 注释为契约，编译与五个用例跑通前不视为已验证 |
| 逐向量标量量化和打包可能达不到 10× | 先按现有分核和 UB 内矩阵测量。未达标时只调整数据驻留和分核，不改编码公式，也不在本文填写未测倍率 |
| 重构 MSE 与内积误差没有解码实现 | 编码验收与任务书 3.2 分开。3.2 在具备与本编码互逆的参考解码后再测 |
| README 仍指向不存在的 `tests/pytest` | 以 `tests/verify.py` 和 ACLNN 测试程序为准，提交算子 README 时改到实际路径 |

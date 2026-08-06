# BatchedMatmulSoftmax 算子设计文档

# 需求背景（required）

## 需求来源

本任务来自 CANN 社区 2026 年 7 月算子开发任务，要求在 CATLASS 仓中实现 `BatchedMatmulSoftmax` 融合算子。算子面向 Ascend 950 硬件，使用 Ascend C 开发。

算子功能为对每个 batch 独立执行 Batched Matmul，并对 Matmul 结果按行做 Softmax：

```text
S_b = row_softmax(A_b * B_b)
```

输入输出规格如下：

| 参数 | 输入/输出 | 描述 | 数据类型 | 数据格式 | Shape |
| --- | --- | --- | --- | --- | --- |
| A | 输入 | Batched Matmul 左矩阵，RowMajor 布局 | FP16 | ND | `(batch, M, K)` |
| B | 输入 | Batched Matmul 右矩阵，ColumnMajor 布局 | FP16 | ND | `(batch, K, N)` |
| S | 输出 | Softmax 输出矩阵，RowMajor 布局 | FP16 | ND | `(batch, M, N)` |

## 背景介绍

Batched Matmul 后接 Softmax 是 Attention 机制中的常见计算模式。若使用两个独立算子完成，Matmul 的中间结果需要写回全局内存，再由 Softmax 读取，存在额外的访存和 kernel launch 开销。

本设计将 Batched Matmul 与行级 Softmax 合并到一个算子中完成，Matmul 结果在算子内部直接被 Softmax 消费，不作为外部输出，从而减少中间数据搬运并提升整体性能。

# 需求分析（required）

## 需求描述

实现 `BatchedMatmulSoftmax` 融合算子，满足以下要求：

1. 单次 kernel launch 完成 Batched Matmul 和行级 Softmax。
2. batch 维度之间独立计算。
3. Softmax 使用 max 归一化，避免 `exp` 溢出。
4. 输入 A、B 和输出 S 均为 FP16。
5. 性能达到 `torch.bmm + torch.softmax` 小算子拼接方案的 1.2 倍。
6. 基于 CATLASS-optest 测试工程补充测试交付件。

## 需求拆解

| 子需求 | 设计说明 |
| --- | --- |
| Matmul 计算 | 对每个 batch 计算 `A_b * B_b` |
| Softmax 计算 | 对 Matmul 结果的每一行沿 N 维做 Softmax |
| 数值稳定 | Softmax 前先减去当前行最大值 |
| 数据布局 | A 使用 RowMajor，B 使用 ColumnMajor，S 使用 RowMajor |
| 中间结果 | Matmul 输出仅在算子内部使用，不对外暴露 |
| 测试验证 | 覆盖功能、精度和性能测试 |

# 详细设计（required）

## 算子分析

### 数学公式

对每个 batch `b`：

```text
M_b = A_b * B_b
S_b[m, n] = exp(M_b[m, n] - max(M_b[m, :])) /
            sum_j exp(M_b[m, j] - max(M_b[m, :]))
```

其中 `A_b` 形状为 `(M, K)`，`B_b` 形状为 `(K, N)`，`S_b` 形状为 `(M, N)`。

### 支持数据类型

| 数据 | 类型 |
| --- | --- |
| A | FP16 |
| B | FP16 |
| S | FP16 |
| Softmax 中间统计量 | FP32 |

### 支持形状

支持 3 维输入输出：

```text
A: (batch, M, K)
B: (batch, K, N)
S: (batch, M, N)
```

其中 `batch > 0`，`M > 0`，`K > 0`，`N > 0`。

## 算子实现

### 工程实现方案

算子整体分为 host 侧参数处理和 kernel 侧计算两部分。

从 CATLASS 工程组织上，计划新增或复用的组件如下：

| 模块 | 新增/复用 | 说明 |
| --- | --- | --- |
| `examples/74_ascend950_batched_matmul_softmax` | 新增 | 提供 Ascend 950 样例、构建脚本和 README，用于功能验证与性能采集 |
| `include/catlass/gemm/kernel` | 新增/扩展 | 增加 BatchedMatmulSoftmax kernel 级调度，组织 batch、M/N tile 和 cube/vector 协同 |
| `include/catlass/gemm/block` | 复用 | 复用 `BlockMmadTla` 完成 FP16 x FP16 -> FP32 的矩阵乘 |
| `include/catlass/epilogue/block` | 新增/复用 | 组织 Softmax 的行级归约、归一化和输出转换，可参考现有 FA Softmax epilogue |
| `include/catlass/epilogue/tile` | 新增/复用 | 封装 GM/UB/L1 间拷贝、尾块 padding、FP32 到 FP16 cast 等 tile 级操作 |
| `tests/optest/kernels` | 新增 | 增加 JIT kernel entry 与 template，实现 PyTorch 侧调用 |
| `tests/optest/src/include/template` | 新增 | 增加 PyTorch Tensor 到 kernel 参数的适配逻辑 |
| `tests/optest/torch_catlass/ops` | 新增 | 增加 Python API 包装 |
| `tests/optest/tests` | 新增 | 增加 pytest 精度和基础性能测试 |

其中 `gemm/block` 和部分 `epilogue/tile` 优先复用已有 CATLASS 能力，只有在 BatchedMatmulSoftmax 的数据流或尾块处理无法直接复用时才补充轻量组件，避免引入与现有框架不一致的独立实现。

### Host 侧设计

host 侧完成以下工作：

1. 校验输入输出 shape、dtype 和 layout。
2. 根据 `batch/M/K/N` 计算 stride。
3. 选择 Matmul 和 Softmax 使用的 tiling 参数。
4. 计算并申请 kernel 所需 workspace。
5. 下发 kernel 执行参数。

host 侧主要校验规则：

| 校验项 | 要求 |
| --- | --- |
| dtype | A/B/S 均为 FP16 |
| shape | A/B/S 均为 3 维 |
| K 维 | A 的 K 与 B 的 K 一致 |
| 输出 shape | S 为 `(batch, M, N)` |
| layout | A RowMajor，B ColumnMajor，S RowMajor |

tiling 设计以任务测试集中的典型 shape 为主，优先保证计算单元利用率和 Softmax 归约效率。对于非 tile 对齐的 M/N/K，kernel 侧通过边界判断处理尾块。

### Kernel 侧设计

kernel 侧完成 Batched Matmul 与 Softmax 融合计算：

1. 根据当前任务编号确定 `batch_id` 和矩阵分块位置。
2. 读取 A、B 对应分块并执行 Matmul。
3. 将 Matmul 结果作为 Softmax 输入。
4. 对每一行计算最大值。
5. 计算 `exp(x - max)` 并求和。
6. 完成归一化并写回输出 S。

Softmax 计算使用如下稳定形式：

```text
row_max = max(row)
row_sum = sum(exp(row - row_max))
output = exp(row - row_max) / row_sum
```

当 N 维需要分多个 tile 处理时，需要先统计整行的最大值和 sum，再写回最终归一化结果，避免只对局部 tile 做 Softmax。

#### 分核方案

kernel 采用 Ascend 950 mixed core 形态，Cube 侧负责 Matmul，Vector 侧负责 Softmax。任务划分以 `(batch, M tile, N tile)` 为基本调度单元：

1. Cube 核按照 batch 维和 M/N 分块分配 Matmul 任务，每个任务计算一块 FP32 logits。
2. Vector 核按行或多行分组消费 logits，完成 max、exp、sum 和 normalize。
3. batch 之间天然独立，不做跨 batch 归约。
4. M/N/K 尾块在 tile 内通过实际 shape 和 padding 处理，不要求输入 shape 特殊对齐。

初始实现可采用 `1:2` 的 cube/vector 比例作为默认 mixed kernel 配置，后续根据任务测试集 shape 和 msprof 结果调整。对于 Matmul 占比高的大 K 场景，可增加 Cube 侧并行度；对于 N 较大且 Softmax 占比较高的场景，可增加 Vector 侧行并行度或调大 Softmax 行 tile。

#### Cube/Vector 流水设计

性能优化路径采用 tile 级流水，而不是完整 Matmul 结束后再统一做 Softmax：

1. Cube 侧从 GM 搬入 A、B tile，使用 `BlockMmadTla` 计算 FP32 logits tile。
2. Cube 侧通过 ping-pong buffer 暂存 logits tile，并通过 cross-core flag 通知 Vector 侧。
3. Vector 侧等待对应 tile ready 后，在 UB 中完成行级 Softmax 统计；当整行统计完成后再执行归一化。
4. Vector 侧将 FP16 结果写回 S，并释放当前 buffer 给 Cube 侧继续生产。
5. 对于 `N <= N_tile` 的 case，一个 N tile 覆盖整行，可直接完成行级 Softmax。
6. 对于 `N > N_tile` 的 case，需要先做跨 N tile 的 online max/sum 更新，最后再进行归一化写回，保证结果等价于整行 Softmax。

数据传递优先走 UB/L1 中间缓冲，减少 logits 全量写 GM 的开销。若某些 shape 受 UB/L1 容量限制，也可以保留内部 GM workspace 作为基线路径，但性能优化优先级低于 tile 级流水路径。

#### N 方向分块与 max/sum 传递

行级 Softmax 的归一化范围是完整 N 维，因此调度时需要避免把同一行的归一化状态拆散。当前方案中，Cube 侧沿 N 方向分块时可能出现同一 `(batch, M tile)` 的 logits 由多个 Cube 核生产，但 Vector 侧默认不把同一行块的 N tile 分给多个核并行归一化。默认设计采用如下规则：

1. Softmax 的任务归属以 `(batch, M tile)` 为单位，一个 `(batch, M tile)` 的全部 N tile 由同一个 Vector 逻辑任务连续消费。
2. Cube 侧可以沿 N 方向并行生产 logits tile，但这些 tile 需要通过 ready flag 或内部 workspace 交给对应的 Vector owner。
3. Vector owner 在本核 UB 中维护该行块的 `maxUb/sumUb`，按 N tile 顺序执行 online softmax 更新。
4. 默认不把同一行块的 N tile 分配给多个 Vector 核同时归一化，因此 `maxUb/sumUb` 不需要跨 Vector 核传递，只在 owner 核本地保存；跨核只传递 logits tile 的 ready 状态和缓冲区位置。

online 更新公式如下：

```text
new_max = max(old_max, tile_max)
new_sum = old_sum * exp(old_max - new_max) +
          tile_sum * exp(tile_max - new_max)
```

其中 `tile_max` 和 `tile_sum` 为当前 N tile 的局部统计量，`old_max/old_sum` 为之前 N tile 累积的统计量。最后一个 N tile 完成后，Vector owner 使用全行 `new_max/new_sum` 对所有 N tile 的 `exp(x - new_max)` 做归一化并写回输出。

当任务测试集中存在极大 N，且单个 Vector owner 顺序消费全部 N tile 成为瓶颈时，可启用多 Vector 核协同方案。该方案需要额外的统计 workspace 和明确的同步阶段：

1. **局部统计阶段**：多个 Vector 核分别处理同一行块的不同 N tile，计算各自的 `local_max/local_sum`，写入统计 workspace。
2. **全局归并阶段**：指定 reducer 核或按树形归约读取所有局部统计，计算该行的 `global_max/global_sum`。归并公式为：

   ```text
   global_max = max(local_max_i)
   global_sum = sum(local_sum_i * exp(local_max_i - global_max))
   ```

3. **归一化阶段**：各 Vector 核等待 `global_max/global_sum` ready 后，读取自己负责的 logits tile，计算 `exp(x - global_max) / global_sum` 并写回 S。

多核协同方案的同步开销和 workspace 开销较高，仅作为大 N case 的优化路径。常规 case 优先采用 `(batch, M tile)` 单 owner 策略，通过增加 batch/M 方向并行度提升吞吐，避免同一行 Softmax 的跨核状态传递。

#### Softmax UB 分配与流水

Softmax 侧以若干行乘以 `N_tile` 作为一次 Vector 处理块，UB 中主要分配：

| UB 区域 | 数据类型 | 作用 |
| --- | --- | --- |
| `srcUb` | FP32 | 保存 Matmul logits tile，尾部 padding 为极小值 |
| `dstUb` | FP32/FP16 | 保存 `exp(x - max) / sum` 的中间或最终结果 |
| `maxUb` | FP32 | 保存每行当前最大值 |
| `sumUb` | FP32 | 保存每行当前 `exp` 累加和 |
| `tmpUb` | uint8/FP32 | AscendC SoftMax 或自定义向量计算临时空间 |

Softmax 计算流程为：

1. 对 logits tile 做尾部 padding，保证尾轴长度满足 32B 对齐要求。
2. Vector 侧对每行执行 reduce max。
3. 计算 `exp(x - row_max)` 并 reduce sum。
4. 对每个元素除以 row sum，得到概率值。
5. 将 FP32 概率 cast 为 FP16，并按 RowMajor 写回 S。

当 N 方向分多段时，`maxUb/sumUb` 保存跨 tile 的统计状态，使用 online softmax 公式更新全行统计量。最后一轮归一化必须使用全行 max 和全行 sum，不能对局部 tile 单独归一化。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 | √ |

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 输入类型 | 仅支持 FP16 |
| 输出类型 | 仅支持 FP16 |
| 输入布局 | A 为 RowMajor，B 为 ColumnMajor |
| 输出布局 | S 为 RowMajor |
| 非连续 Tensor | 无额外约束，按任务测试集输入形态处理 |
| 额外功能 | 不支持 mask、bias、dropout、causal 等扩展 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足生态算子开源精度标准，输出结果与 CPU golden 对齐 | 任务书 |
| 性能标准 | 相同测试集下达到 `torch.bmm + torch.softmax` 拼接方案 1.2 倍性能 | 任务书 |

CPU golden 计算方式：

```python
matmul = np.matmul(A.astype(np.float32), B.astype(np.float32))
shifted = matmul - np.max(matmul, axis=-1, keepdims=True)
golden = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
golden = golden.astype(np.float16)
```

## 测试设计

测试用例覆盖以下场景：

| 场景 | 说明 |
| --- | --- |
| 基础功能 | 小 shape 验证 Matmul + Softmax 结果正确 |
| 多 batch | 验证不同 batch 独立计算 |
| N 维归约 | 验证行级 Softmax 沿 N 维执行 |
| 尾块场景 | 覆盖 M/N/K 非 tile 对齐 |
| 数值稳定 | 覆盖较大正负输入、全零输入、重复最大值 |
| 性能测试 | 使用任务测试集对比 `torch.bmm + torch.softmax` 基线 |

精度测试记录最大绝对误差、最大相对误差、是否存在 NaN/Inf 以及失败样例信息。性能测试使用 `msprof op` 采集 baseline 耗时、融合算子耗时、speedup 和使用的 tiling 参数。

任务验收使用任务提供的 `self_test_case/batched_matmul_softmax` 测试集作为重点输入集合。开发时先识别测试集中各 case 的 `batch/M/N/K` 分布，再按 shape 特征划分实现策略：

| case 类型 | 优化策略 |
| --- | --- |
| 小 M/N 或小 batch | 减少调度开销，优先使用较小 tile 和更轻的 Softmax 路径 |
| 大 K | 提高 Cube 利用率，优先调大 K tile 或增加 Cube 侧并行 |
| 大 N | 优化 Vector 侧行级 Softmax，采用多 N tile 的 online max/sum |
| N 非对齐 | 通过 padding 和尾块 mask 保证精度，不增加输入约束 |
| batch 较大 | batch 维并行分核，提高多 batch 吞吐 |

性能验收关注测试集整体平均 speedup，不要求每个 case 都单独达到 1.2 倍。实现上可准备多种 kernel 方案或多组 tileShape 配置，并在测试报告中备注每类 case 选择的策略、tileShape、性能数据和切换条件。

## 兼容性分析

该算子作为 CATLASS 新增融合算子实现，不修改已有算子接口和已有样例行为。对于不满足 dtype、shape 或 layout 约束的输入，host 侧直接返回不支持，避免进入 kernel 后产生错误结果。

# 参考资料

- `BatchedMatmulSoftmax算子开发任务书.md`
- `设计文档模板.md`
- CATLASS 仓库：<https://gitcode.com/cann/catlass>
- CATLASS-optest 测试工程：<https://gitcode.com/cann/catlass/blob/master/tests/optest/README.md>

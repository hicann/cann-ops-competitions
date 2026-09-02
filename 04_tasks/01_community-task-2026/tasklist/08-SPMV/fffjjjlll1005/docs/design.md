# SpMV 算子设计文档

> 任务：8 月社区任务——Ascend 950PR SpMV 算子开发
> 团队目录：`fffjjjlll1005`
> 目标硬件：Ascend 950PR / DAV-3510（`arch35`）

## 1. 需求背景

### 1.1 需求来源

社区任务要求在 Ascend 950PR 上使用 Ascend C 实现 CSR 格式稀疏矩阵向量乘（SpMV），接口风格与 cuSPARSE 的三阶段调用方式对齐：

```text
GetBufferSize -> Preprocess（可选） -> SpMV
```

计算公式为：

```text
Y = alpha * op(A) * X + beta * Y
```

其中 `op(A)` 支持非转置和转置，`A` 为 CSR 稀疏矩阵，`X` 和 `Y` 为一维稠密向量。

### 1.2 背景介绍

SpMV 的行非零数分布不均，单一并行策略难以同时覆盖短行和长行；转置若直接以 scatter 和浮点原子累加实现，又难以保证逐 bit 确定性。本设计以确定性、三阶段生命周期和可维护性为优先，不引入运行时 autotune 或多套互不相容的数据流。

## 2. 需求分析

### 2.1 需求描述

首版必须支持：

- CSR、0-based、INT32 row offsets / column indices；
- `NON_TRANSPOSE` 与 `TRANSPOSE`，不支持共轭转置；
- Host/Device pointer mode；
- `ACL_SPARSE_SPMV_ALG_DEFAULT`；
- Y 原位读写，但 Y 不得与 X 或 CSR 输入数组重叠；
- 相同环境、输入位模式和参数下重复执行结果逐 bit 一致；
- 五类输入数组的正元素 stride，旧 descriptor 接口保持 `stride=1`。

支持的类型组合：

| A / X | computeType | Y | N | T |
|---|---|---|---:|---:|
| FP32 | FP32 | FP32 | 支持 | 支持 |
| FP16 | FP32 | FP16 | 支持 | 支持 |
| FP16 | FP32 | FP32 | 支持 | 支持 |
| BF16 | FP32 | BF16 | 支持 | 支持 |
| BF16 | FP32 | FP32 | 支持 | 支持 |
| INT8 | FP32 | FP32 | 支持 | 支持 |
| INT8 | INT32 | INT32 | 支持 | 支持 |

arch35 的 `alpha`、`beta` ABI 固定为 FP32，不随 `computeType` 改变。INT32 路径使用 INT32 行内累加，再以 FP32 融合 `alpha * sum + beta * y`，有限结果向零截断并饱和写回 INT32。

### 2.2 需求拆解

| 子问题 | 设计结论 |
|---|---|
| 非转置 | 直接读取原 CSR，按输出行归约 |
| 转置 | 在 workspace 中稳定构造 `CSR(A^T)` 与 `sourceIndex`，再复用同一行归约 Kernel |
| 短行/长行 | Host 按平均 NNZ/输出行在 scalar-row 与 warp-row 间选择 |
| 确定性 | 固定遍历、固定 warp 归约；不以浮点 atomic scatter 作为默认路径 |
| values 更新 | workspace 只保存结构与 `sourceIndex`，Execute 始终读取当前原始 values |
| 非连续输入 | Ex descriptor 表达 CSR 三数组和 X/Y 的正元素 stride |
| 生命周期 | Preprocess 可选；未调用时首次 Execute 可在同一 stream 补做 |
| 边界 | `nnz == 0` 或 Host pointer mode 下 `alpha == 0` 走 ScaleY 快路径 |

## 3. 详细设计

### 3.1 算子分析

对非转置 CSR，输出第 `r` 个元素为：

```text
sum = Σ A.values[p] * X[A.colInd[p]]
p in [A.rowPtr[r], A.rowPtr[r + 1])
Y[r] = alpha * sum + beta * Y[r]
```

对转置，先稳定构造 `CSR(A^T)`。每个转置非零位置记录其原矩阵 values 下标：

```text
valueT[p] := A.values[sourceIndex[p]]
```

这样转置与非转置最终都转化为“每个输出行独立归约”，避免跨行原子累加；只更新 `A.values` 后也无需重新 Preprocess。若 row offsets、column indices、shape、stride 或结构指针变化，则必须重新 Preprocess。

### 3.2 总体组件

```text
Public API
  ├─ Validate: descriptor、shape、dtype、stride、pointer mode、算法
  ├─ Layout: checked workspace 大小、64 B 对齐、128 B header
  ├─ Preprocess: 转置结构、sourceIndex、identity map、状态写入
  └─ Execute
       ├─ ScaleY fast path
       ├─ scalar-row kernel
       └─ warp-row kernel
```

### 3.3 算子实现

主要文件职责：

| 路径 | 职责 |
|---|---|
| `sparse/spmv/arch35/spmv_host.cpp` | 参数校验、workspace 计算、生命周期、类型分发和 Kernel launch |
| `sparse/spmv/arch35/spmv_kernel.cpp` | Ascend C Kernel 实例化入口 |
| `sparse/spmv/arch35/spmv_kernel.h` | scalar-row、warp-row、ScaleY 与转置辅助 Kernel |
| `sparse/spmv/arch35/spmv_tiling_data.h` | Host/Device 共用 tiling 数据结构 |
| `sparse/spmv/arch35/spmv.h` | arch35 内部声明 |
| `test/spmv/arch35/spmv_test.cpp` | 公共 API、生命周期、精度、确定性、stride、性能测试 |

#### 3.3.1 GetBufferSize

1. 执行共享参数校验；
2. N 模式计算必要 header/handle workspace；
3. T 模式以 checked arithmetic 计算转置 row offsets、column indices、`sourceIndex`、identity map 和 CSR2CSC scratch；
4. 所有段按 64 B 对齐，溢出或超出显式上限时返回参数错误。

#### 3.3.2 Preprocess

- N 模式记录有效状态，不复制矩阵数值；
- T 模式稳定构造 `CSR(A^T)` 结构并保存 `sourceIndex`；
- 结构非连续时先按元素 stride pack，再调用稳定 CSR2CSC 路径；
- 结果与 mat descriptor 的结构版本绑定，values-only 更新不使其失效。

#### 3.3.3 Execute

- 验证 workspace header、算法、descriptor 和生命周期；
- 未显式 Preprocess 时，可在同一 stream 补做必要预处理；
- Host pointer mode 可在 Host 判定 `alpha == 0`；Device pointer mode 不做 D2H 或 stream synchronize；
- 按输出行平均 NNZ 选择路径，设计阈值为 96 NNZ/row；
- 每 block 线程数固定为 256。

### 3.4 Kernel 设计

#### 3.4.1 scalar-row

一线程负责一个输出行，适合短行和极短行。FP32 使用两路展开与双累加器降低相关依赖，最终按固定顺序合并；低精度输入先提升到 computeType。

#### 3.4.2 warp-row

一个 warp 负责一个输出行，各 lane 按固定步长读取非零元素并进行固定树形归约，适合长行。归约顺序固定，不依赖调度先后。

#### 3.4.3 ScaleY

当 `nnz == 0` 或可在 Host 确认 `alpha == 0` 时，不访问 A/X，只执行 `Y = beta * Y`。`beta == 0` 时直接写零。

### 3.5 Workspace 与所有权

- external buffer 由调用方分配并管理；实现不在热路径申请设备内存；
- handle 可复用内部 workspace，但其有效性与当前 descriptor/stream 生命周期绑定；
- header 保存 magic、版本、模式、结构摘要和预处理状态；
- 任何 size/offset 计算均使用 checked arithmetic；
- workspace 不缓存矩阵 values，避免 values 更新后使用陈旧数据。

### 3.6 非连续 Tensor

CSR 与一维向量没有二维 row-major/column-major 语义。本设计将可执行的非连续合同定义为正元素 stride：

- `csrRowPtrStride`；
- `csrColIndStride`；
- `csrValueStride`；
- `xStride`；
- `yStride`。

旧创建接口继续固定 stride 1；Ex 接口仅补充 stride，不改变已有 ABI 的默认行为。零或负 stride 首版拒绝。

### 3.7 状态码

| 类别 | 行为 |
|---|---|
| 空 handle/descriptor/scalar/bufferSize | 返回无效参数 |
| 不支持的 op、alg、dtype 组合 | 返回不支持或无效参数 |
| shape/向量长度不匹配 | 返回无效参数 |
| row offsets/column indices 非法 | 在可检查阶段返回无效参数；目标测试覆盖 |
| workspace 为空或不足 | 在需要 workspace 时返回无效参数 |
| Kernel launch/ACL 失败 | 原样映射为 aclsparse 状态，不伪装为成功 |

## 4. 支持硬件

| 硬件 | 目录 | 设计定位 |
|---|---|---|
| Ascend 950PR / DAV-3510 | `sparse/spmv/arch35` | 目标硬件与实现目录 |
| A2/A3 / DAV-2201 | `sparse/spmv/arch22` | 既有实现目录；公共 descriptor 改动需单独回归 |

950PR 与 A2/A3 分属不同架构目录，兼容性结论需要分别验证。

## 5. 算子约束限制

- 仅支持 CSR、0-based、INT32 索引；
- 仅支持 `NON_TRANSPOSE` 和 `TRANSPOSE`；
- 首版仅支持 `ACL_SPARSE_SPMV_ALG_DEFAULT`；
- A 与 X 值类型必须匹配支持表；
- alpha/beta 指针非空，arch35 scalar ABI 为 FP32；
- Y 可原位更新，但不得与 A/X 输入数据区重叠；
- 元素 stride 必须为正；
- `Preprocess` 后结构变化必须重新预处理；仅 values 更新可复用；
- 不承诺不同 CANN、驱动、固件、硬件或编译选项间逐 bit 一致。

## 6. 可维可测分析

### 6.1 测试分层

| 层级 | 内容 | 判定 |
|---|---|---|
| 离线参考 | 规格统计、确定性生成器、workspace 计算、CPU N/T golden | 固定清单与输入指纹一致 |
| 默认单测 | 参数、边界、descriptor、构建、公共 API 基本路径 | 全部非显式 skip 测试通过 |
| 精度 | 附件 200 条规格，七种类型组合、N/T、alpha/beta、稀疏度 | 按生态精度标准比较 |
| stride | 五类独立/组合 stride、切片、更新与 ScaleY | 结果与连续 golden 一致 |
| 生命周期 | 显式/隐式 Preprocess、values 更新、结构失效、pointer mode | 状态与结果符合合同 |
| 确定性 | 同输入重复 1000 次 | 输出逐 bit 一致 |
| 集成 | 所有受公共 descriptor 影响的测试二进制 | 无失败、无未解释 skip |
| 性能 | 四个标杆 case 三轮 + 56 条泛化覆盖 + 阶段/摊销统计 | 按 6.4 节性能标准判定；无标杆组合只报告数据，不虚构阈值 |

### 6.2 固定数据生成协议

任务附件 JSON 不包含实际 CSR 数组且 seed 为空，因此测试数据生成协议定义为 `spmv-casegen-v2`：

- 每个 case 使用固定派生 seed；
- NNZ 取整规则固定；
- CSR values 为 `[-10, 10]`，X/Y 按逐 case 规格为 `[-5, 5]`；
- 保存清单 SHA256、每 case 输入 SHA256、NNZ 与目标侧 FNV64；
- 未显式执行更新命令时不得重写指纹清单。

### 6.3 精度标准

- FP32、FP16、BF16 按生态算子开源精度标准使用 mixed tolerance 与匹配比例；
- INT32 输出要求逐元素精确相等；
- CPU golden 与设备执行使用相同 alpha/beta 和转置语义；
- 任何单标杆失败才进入 ATK 等补充定位，不用补充工具掩盖公共 API 失败。

### 6.4 性能标准

任务书给出四个 FP32 标杆：

| 用例 | shape | 稀疏度 | 标杆延迟 |
|---|---:|---:|---:|
| P1 | 128 × 128 | 95% | 43.9 us |
| P2 | 1024 × 1024 | 99% | 46.3 us |
| P3 | 2048 × 4096 | 97.5% | 45.4 us |
| P4 | 160220 × 68750 | 99.9% | 193.2 us |

性能验收标准定义为“吞吐不低于标杆 0.5 倍”，等价于“热 Execute median 延迟不超过标杆 2 倍”；`Preprocess` 单独报告，不计入重复 Execute 热延迟。报告同时保留 cold、median、P90、P99、warmup/measure 次数和输入指纹。

### 6.5 可维护性

- 参数校验、workspace 布局、预处理、类型分发和 Kernel 分层；
- N/T 共用行归约主干，避免维护两套计算语义；
- 常量和算法阈值集中，性能调优不改变公共生命周期；
- 测试报告绑定源代码版本、环境指纹和证据哈希，不用历史结果替代待验版本验证。

## 7. 兼容性分析

- Ex descriptor 为新增接口，旧 descriptor 接口继续生成 stride 1，不破坏旧调用；
- arch35 新实现不替换 arch22 Kernel，但公共 descriptor 变更会影响其他算子，必须执行 A2/A3 和仓库 CI；
- 结构更新使 active buffer 失效，values-only 更新保持有效，该规则与 SpGEMM 等共享 descriptor 使用方一致；
- 任务仓源码根目录采用 `sparse/`，上游合入路径以仓库实际结构为准，不采用旧任务文字中的 `src/` 路径。

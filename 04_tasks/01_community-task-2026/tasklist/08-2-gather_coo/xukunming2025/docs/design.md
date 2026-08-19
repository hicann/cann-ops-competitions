# Gather COO 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务来源 | [2026 年 8 月 Gather COO 任务书](../../../../docs/202608/gather_coo_task_doc.md) |
| 目标仓库 | `cann/ops-gnn` |
| 目标目录 | `gather_coo/`；产品源码接入 `csrc/`、`python/` 和 `test/` 现有工程结构 |
| 公开接口 | `ops_gnn.gather_coo(src, index, out=None)` |
| 适配硬件 | Ascend 950PR |
| 实现语言 | Ascend C Kernel、C++ Host、Python PyTorch 适配层 |

本文已由开发前计划稿更新为实机验证后的实现稿。记录的验证环境为 Ascend950PR_9579、CANN 9.0.0-beta.2、Python 3.12.9、PyTorch 2.7.1、torch_npu 2.7.1.post4、GNU g++ 13.3.0，Ascend C 编译目标为 `dav-3510`。

## 需求背景（required）

### 需求来源

Gather COO 参考 `torch_scatter.gather_coo`，将按分组维保存的源特征扩展到有序 COO 条目。任务要求以 PyTorch 层接口的功能、精度和性能作为验收口径，aclnn 接口为可选项。

公开接口固定为：

```python
def gather_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...
```

不得增加 `reduce`、`dim` 或 `dim_size` 参数。本任务只实现前向。

### 背景介绍

令 `dim = index.dim() - 1`。对任意合法的 prefix 坐标 `p`、COO 条目 `e` 和 suffix 坐标 `s`：

```text
out[p, e, s] = src[p, index[p, e], s]
```

Gather COO 与 Segment COO 的数据流方向相反：

| 算子 | 数据流方向 | 主要工作 |
| --- | --- | --- |
| `segment_coo` | 多个 COO 条目压缩到分组 | 归约 |
| `gather_coo` | 分组特征扩展到 COO 条目 | 索引读取和纯复制 |

Gather COO 不做浮点运算、归约或原子写。实现难点集中在 INT64 索引读取、多维地址映射、非连续 Tensor、可选 `out`、当前 stream 顺序，以及大输出下的搬运带宽。

## 需求分析（required）

### 需求描述

实现必须满足：

1. 与任务书和 `torch_scatter.gather_coo` 的三参数接口、输出 shape、dtype、device 和 `out` 行为对齐。
2. 支持 rank 1～8、空 Tensor、非连续 `src/index/out`、多 batch 和多 suffix 维。
3. `index` 固定为 INT64，沿最后一维非降序，且值位于源 gather 维的合法范围。
4. L1 原生支持 FP16、BF16、FP32、INT8、INT16、INT32、UINT8。
5. L2 支持 FP64 和 INT64；不参加性能考核。
6. 所有支持 dtype 均为原始位复制，与 CPU 参考逐 bit 一致。
7. 五组 shape 的 FP16/FP32 共十个性能点分别满足 `A100_ms / NPU_ms >= 0.6`，不能以平均值抵消失败点。

### 输入、输出和约束

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| `src` | NPU Tensor | rank 1～8；支持 L1/L2 dtype |
| `index` | NPU INT64 Tensor | `1 <= index.dim() <= src.dim()`；prefix 与 `src` 对齐；最后一维非降序 |
| `out` | 可选 NPU Tensor | 完整 shape、dtype 和 device 必须与推导结果一致；支持非连续和输入别名 |
| 返回值 | NPU Tensor | 未提供 `out` 时新建；提供时更新并返回与 `out` 相同的存储 |

未排序或越界索引属于任务书规定的前置条件违例。生产快路径不会为扫描数据值而把索引搬回 CPU，也不会引入设备同步；结构性错误在 launch 前检查。

### 输出形状

输出 shape 与 `src` 相同，仅将第 `dim` 维替换为 `index.size(-1)`：

| `src` | `index` | 输出 |
| --- | --- | --- |
| `(N,)` | `(E,)` | `(E,)` |
| `(N,K)` | `(E,)` | `(E,K)` |
| `(B,N,K)` | `(B,E)` | `(B,E,K)` |
| `(P0,P1,N,S0,S1)` | `(P0,P1,E)` | `(P0,P1,E,S0,S1)` |

## 详细设计（required）

### 总体架构

```text
ops_gnn.gather_coo
  -> pybind 三参数绑定
  -> Host 校验、连续化、B/N/E/K 展平和 Tiling
  -> torch_npu 当前 stream bridge
  -> 单次 Ascend C AIV kernel
       |- K=128 output-slab 带宽快路径
       `- 通用逐行/分片路径（含有序 run 缓存）
  -> 非连续或别名 out 的安全回写
```

Kernel 不使用 GM workspace。非连续输入和 `out` 所需的连续临时 Tensor 由 PyTorch 适配层管理，不属于 Kernel workspace。

### 统一展平和地址模型

Host 将任意 1～8 维输入归一化为：

```text
B = product(src.shape[0 : dim])
N = src.shape[dim]
E = index.shape[dim]
K = product(src.shape[dim + 1 :])

src   -> [B, N, K]
index -> [B, E]
out   -> [B, E, K]
```

线性地址为：

```text
idx       = index[b * E + e]
srcRow    = (b * N + idx) * K
outputRow = (b * E + e) * K
```

所有形状乘积、行号和 GM 偏移使用 64-bit，并通过 checked multiplication 防止 Host 中间结果溢出。

### 数据类型映射

| `src/out` dtype | Kernel 搬运表示 |
| --- | --- |
| INT8、UINT8 | `uint8_t` |
| FP16、BF16、INT16 | `uint16_t` |
| FP32、INT32 | `uint32_t` |
| FP64、INT64 | 每个逻辑元素拆成两个 `uint32_t` lane |

实现不做 cast 或数值运算，因此 NaN payload、正负零、无穷和整数位模式均原样复制。

### Host 与 PyTorch 当前 stream

Host 依次执行：

1. 校验 device、dtype、rank、prefix shape 和可选 `out`。
2. 对非连续 `src/index` 调用 `.contiguous()`。
3. 推导输出 shape；为核心路径分配连续输出 Tensor。
4. 计算 Tiling，并按 dtype 分发搬运模板。
5. 通过 `c10_npu::getCurrentNPUStream().stream()` 取得 torch_npu 管理的 ACL stream，再异步追加 Kernel。
6. 提供 `out` 时将连续结果安全复制回原始 `out`，处理非连续和输入输出别名。

`NPUStream::stream()` 会先将 torch_npu host task repository 中的待提交任务排入当前 stream，保证 `.contiguous()`、输入生产任务和自定义 Kernel 的先后关系。它可能造成 Host 等待队列提交，但不会等待设备计算完成；实现没有调用 `torch.npu.synchronize`、`aclrtSynchronizeDevice` 或 `NPUStream::synchronize`，也没有创建私有 ACL stream。

### 实际 TilingData

最终结构传递以下信息：

```cpp
struct GatherCooTilingData {
    uint64_t batchCount;
    uint64_t sourceRows;
    uint64_t indexRows;
    uint64_t featureCount;          // Kernel 存储 lane 数
    uint64_t logicalFeatureCount;   // 逻辑 K
    uint32_t elementBytes;
    uint64_t totalRows;
    uint64_t indexStorageElements;
    uint64_t sourceStorageElements;
    uint64_t outputStorageElements;
    uint64_t rowsPerCore;
    uint64_t extraCoreCount;
    uint32_t featureTile;
    uint32_t coreNum;
    uint32_t useRunCache;
};
```

`totalRows = B * E`。可用 AIV 核数与总行数取最小值；每核取得一个连续输出区间，前 `extraCoreCount` 个核多处理一行。连续区间有利于输出顺序写入，并避免无谓打断有序 index run。

### INT64 index UB block

Ascend950 向量核上的 `GetValue<int64_t>` 不作为可靠标量路径使用。Kernel 将公开 INT64 index 视为两个 `uint32_t` lane：

1. 每次读取最多 128 个 index。
2. GM 起点按 4 个 INT64（32 B）向下对齐。
3. 完整块使用 `DataCopy`，尾块使用输入方向 `DataCopyPad`。
4. 在 UB 中以 low/high 两个 32-bit lane 重建 64-bit 索引。

该表示不缩窄索引，不改变 bit pattern。128-entry index block 和尾块处理已在 Ascend950PR 上逐字节验证。

### K=128 单 bank output-slab 快路径

任务书的十个性能点均为逻辑 `K=128`。当 `logicalFeatureCount == 128 && elementBytes <= 4` 时启用共享结构快路径：

1. 单个 output slab 保存最多 `128 x 128` 个元素。
2. 当前 index block 中每一行分别发出源 GM 到 slab 对应行的 MTE2 搬运，行间目的地址互不重叠，不逐行等待。
3. 全部源行搬运发出后只执行一次 `MTE2 -> MTE3` 依赖。
4. slab 以一次连续 MTE3 写回本核连续输出区间。
5. 下一 tile 复用 slab 前等待上一笔 MTE3，Kernel 退出前 drain。

该布局将旧路径“每行一次等待 + 每行一次 MTE3”改为“每128行一次依赖 + 一次连续写回”，是最大 FP32 点从约 2.65 ms 降至约 0.69 ms 的主要原因。

实际 UB 请求预算如下：

| dtype 类别 | output slab | row buffer | index buffer | 合计 |
| --- | ---: | ---: | ---: | ---: |
| FP32/INT32 | 65536 B | 512 B | 1056 B | 67104 B |
| FP16/BF16/INT16 | 32768 B | 256 B | 1056 B | 34080 B |
| INT8/UINT8 | 16384 B | 128 B | 1056 B | 17568 B |

无需双 buffer 即已通过全部性能门禁，因此最终实现保留单 bank，避免额外 UB 占用和事件复杂度。

### 通用路径与有序段缓存

非 K128 快路径按输出行处理，`featureTile` 最大为 2048 个 Kernel 存储 lane：

1. 每个 index 仍从 128-entry UB block 读取。
2. 一行特征可放入 row buffer 时整行搬运；大行沿 suffix 切片。
3. 当完整行可缓存且 `E > N` 时启用 `useRunCache`；相邻 index 相同则复用 UB 中的上一源行，减少重复 GM 读取。
4. batch 边界重置缓存，禁止跨 batch 复用。
5. 对齐块使用 `DataCopy`，非对齐尾块使用 `DataCopyPad`。

有序段缓存服务于通用泛化路径；K128 性能快路径以批量输出写回为主，不为了追求重复索引复用而增加分支。

### 空 Tensor、非连续和别名

- `B == 0`、`E == 0` 或 `K == 0` 时返回 shape 正确的空输出，不启动 Kernel。
- `N == 0` 且存在输出条目时没有合法索引，Host 在 launch 前报错。
- 非连续 `src/index` 在当前 stream 上连续化。
- 非连续 `out` 或 `out` 与 `src/index` 存储重叠时，Kernel 写临时连续结果，再通过 `out.copy_` 回写。
- 提供 `out` 时返回 Tensor 与原始 `out` 共享存储；显式空 `out` 不等同于 `None`。

## 支持硬件

| 芯片 | 状态 |
| --- | --- |
| Ascend 950PR | 已实机编译、功能与性能验证 |

未验证的芯片不在本文承诺范围内。

## 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| `src` rank | 1～8 |
| `index` rank | 1～`src.dim()` |
| gather 维 | `index.dim() - 1` |
| index dtype | INT64 |
| index 顺序 | 最后一维非降序 |
| index 范围 | `[0, src.size(dim)-1]` |
| 梯度 | 仅前向，不验收梯度 |
| Workspace | Kernel GM workspace 为 0 |
| 非法索引值 | 前置条件违例；生产 Kernel 不做值扫描 |

## 可维可测分析

### 功能与精度标准

任务书 TC 编号与实现测试一一对应：

| 任务书用例 | 覆盖内容 | 结果 |
| --- | --- | --- |
| TC-01～TC-06 | 六组标准数值 | 通过 |
| TC-07 | `gather_csr` 对照 | 通过 |
| TC-08 | `out` 原地 | 通过 |
| TC-09 | 非连续 | 通过 |
| TC-10 | 空 Tensor | 通过 |
| TC-11 | L1 全 dtype | 通过 |
| TC-12 | FP64 L2 | 通过 |
| TC-13 | INT64 L2 | 通过 |
| TC-14 | 与 `segment_coo` 互逆 | 可选，当前不作为必测通过项 |

此外覆盖 rank 1/8、多 batch、多 suffix、显式空 `out`、输入输出别名、当前非默认 NPU stream 和连续100次确定性。最终默认 task queue 测试结果为 `27 passed, 1 warning`；纯 PyTorch golden 与 CPU `torch_scatter` 均参与逐字节比较。

### 性能标准与实测

性能比定义为：

```text
performance_ratio = A100_latency / NPU_latency
```

每一点均须 `performance_ratio >= 0.6`。最终默认 task queue、NPU Event、warmup 20、repeats 100 的实测如下：

| dtype | 源 shape / index 数 | A100 ms | NPU ms | Ratio | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| FP16 | `16384x128 / 65536` | 0.067 | 0.033556 | 1.9967 | 通过 |
| FP16 | `65536x128 / 65536` | 0.074 | 0.033494 | 2.2093 | 通过 |
| FP16 | `65536x128 / 262144` | 0.295 | 0.139861 | 2.1092 | 通过 |
| FP16 | `262144x128 / 524288` | 0.516 | 0.293927 | 1.7555 | 通过 |
| FP16 | `262144x128 / 1048576` | 1.029 | 0.570256 | 1.8045 | 通过 |
| FP32 | `16384x128 / 65536` | 0.077 | 0.036398 | 2.1155 | 通过 |
| FP32 | `65536x128 / 65536` | 0.083 | 0.037002 | 2.2431 | 通过 |
| FP32 | `65536x128 / 262144` | 0.315 | 0.164726 | 1.9123 | 通过 |
| FP32 | `262144x128 / 524288` | 0.542 | 0.376325 | 1.4402 | 通过 |
| FP32 | `262144x128 / 1048576` | 1.079 | 0.687451 | 1.5696 | 通过 |

十点分别判定，未使用平均值抵消失败点。FP32 最大点另有 msprof 结构证据：AIV MTE2 约占 91%，MTE3 约占 10.7%，符合索引源读与连续输出搬运主导的设计判断。

### 兼容性分析

Gather COO 是新增公开接口，不改变 `add_sample` 和 `segment_max_csr` 的语义。最终提交必须运行旧接口导入与代表性回归；若环境中的既有 NPU 随机算子异常导致回归失败，应以未修改基线对照归因，不能修改无关算子掩盖问题。

## 验收与交付状态

| 门禁 | 当前状态 |
| --- | --- |
| Ascend950PR 构建 | 通过 |
| PyTorch 公共接口 TC-01～TC-13 | 通过 |
| CPU `torch_scatter` bit-wise 对照 | 通过 |
| 默认 task queue / 当前 stream / 100次确定性 | 通过 |
| 十点性能逐点门禁 | 通过 |
| 代表性 msprof | 已完成 FP32 最大点 |
| 干净源副本可复现 | 通过：重新构建、27项Gather、FP16 case1和FP32 case5锚点 |
| Gather COO AscendOpTest/ATK | 任务材料中尚未找到专用入口，不能宣称已通过 |
| 自测报告截图 | 需基于最终干净版本和任务方入口生成真实截图 |
|                                             |                                                        |

## 参考资料

1. [Gather COO 社区任务书](../../../../docs/202608/gather_coo_task_doc.md)
2. [torch_scatter Python 接口](https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/segment_coo.py)
3. [torch_scatter CPU 实现](https://github.com/rusty1s/pytorch_scatter/blob/master/csrc/cpu/segment_coo_cpu.cpp)
4. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
5. [ops-gnn](https://gitcode.com/cann/ops-gnn)

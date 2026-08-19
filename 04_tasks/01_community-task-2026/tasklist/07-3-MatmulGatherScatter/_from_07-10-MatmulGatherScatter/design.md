# MatmulGatherScatter 算子设计文档

本文档对应 [2026 年 7 月社区任务 MatmulGatherScatter 算子开发任务书](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/MatmulGatherScatter_task_doc.md)，并按照[算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)编写。

# 需求背景

## 需求来源

本需求要求基于 CATLASS 和 Ascend C，在 Ascend 950 上实现融合 Gather、Matmul、Scatter 的创新算子。开发基础包括：

- [CATLASS 开源仓](https://gitcode.com/cann/catlass)
- [Ascend 950 基础 Matmul 样例](https://gitcode.com/cann/catlass/blob/master/examples/43_ascend950_basic_matmul/README.md)
- [Ascend 950 Fixpipe 优化样例](https://gitcode.com/cann/catlass/blob/master/examples/46_ascend950_matmul_fixpipe_opti/46_ascend950_matmul_fixpipe_opti.md)
- [CATLASS 创新样例开发指南](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/10_innovative_example_development_guide.md)
- [Ascend C 算子开发优化建议](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_00010.html)

## 业务背景

推荐系统和稀疏特征计算通常只更新稠密矩阵 A 的部分行。常规实现依次执行 D 清零、A Gather、Matmul 和 D Scatter，需要多次 kernel launch，并在 GM 落盘 `(J,K)` Gather 中间张量和 `(J,N)` Matmul 中间张量。

MatmulGatherScatter 将四个阶段融合为一次 MIX kernel，减少调度开销和中间 GM 读写。

## 功能定义

| 参数 | 类型 | 数据类型 | 布局 | 形状 | 说明 |
| --- | --- | --- | --- | --- | --- |
| A | 输入 | FP16 | ND RowMajor | `(M,K)` | 按 indices 读取行 |
| B | 输入 | FP16 | ND RowMajor | `(K,N)` | Matmul 右矩阵 |
| indices | 输入 | INT32 | ND | `(J,)` | 合法且无重复 |
| D | 输出 | FP16 | ND RowMajor | `(M,N)` | 索引行写结果，其余行写零 |

计算定义为：

$$
A_g[i,k] = A[indices[i],k]
$$

$$
C[i,n] = \sum_{k=0}^{K-1} A_g[i,k]B[k,n]
$$

$$
D[r,n] =
\begin{cases}
C[i,n], & r=indices[i] \\
0, & r \notin indices
\end{cases}
$$

逻辑 Matmul 形状为 `(J,N,K)`，A 和 D 的物理行数为 M。调用方保证 indices 位于 `[0,M-1]` 且互不重复，kernel 不进行索引范围检查或去重。

# 需求分析

## 功能要求

1. A、B、D 为 FP16，indices 为 INT32。
2. A、B、D 为连续 ND RowMajor。
3. 单次 kernel launch 完成 D 清零、Gather、Matmul 和 Scatter。
4. Cube 使用 FP32 累加，输出转换为 FP16。
5. 未索引行严格为零。
6. 不创建 `(J,K)` 或 `(J,N)` GM 中间张量。
7. 支持 J、N 尾块和 K 为 16 倍数的非 128 对齐场景。
8. 支持无序 indices；合法性和唯一性由调用方保证。
9. CATLASS optest 覆盖任务集全部 180 个 shape。

## 测试特征

任务集包含 180 组用例：

| 维度 | 取值 |
| --- | --- |
| M | 400、500、1000、2000、4000、8000 |
| J | 128、256、512、1024、2048、4096 |
| K | 128、256、512、768、2048 |
| N | 128、256、512、1024、2048、4096 |

测试集 `J/M` 为 0.320 至 0.512。任务 shape 均为 128 对齐，但实现保留 J/N 尾块、K 的 16 对齐尾块和 K>2048 容量回退。

## 架构依据

目标实机为 Ascend950PR_9579，NPU Arch 3510：

| 资源 | 容量或数量 |
| --- | ---: |
| Cube Core | 28 |
| Vector Core | 56 |
| 每 AI 子系统 | 1 Cube + 2 Vector |
| L1 | 512KB |
| L0A/L0B | 各 64KB |
| L0C | 256KB |
| 每 Vector Core 可用 UB | 248KB |
| L2 | 128MB |

实现使用 MIX `(1 AIC, 2 AIV)`，动态 UB 配置为 216KB。Ascend 950 SIMT 可直接表达离散行访问，但 `asc_vf_call` 的启动与完成轮询对短任务有明显开销，因此 SIMT 是按 shape 选择的能力，不作为所有场景的固定主路径。

## 性能难点

1. A 和 D 的行地址由 indices 决定，无法用一个固定二维 stride 描述全部行。
2. 通用 D 全量清零写和离散 Scatter 写必须全核有序；当仅清零未索引行时，两类地址严格不相交，可消除首次全核屏障。
3. 逻辑 M 轴为 J，物理行跨度为 M/K/N，调度坐标与地址计算必须分离。
4. A FullLoad、B 双缓冲和输出 UB 需要同时满足 L1/UB 容量。
5. K=2048 大输出场景中，B 的 GM/L2 到 L1 搬运可能成为关键路径。

# 详细设计

## 总体方案

新增 numbered example `74_ascend950_matmul_gather_scatter`。Kernel 固定启动 28 个 AIC block，使全部 56 个 AIV 参与 D 清零和统一调度；selective-zero 命中时保持 launch 规模但跳过全核屏障。

```text
Host: 选择 TileM/TileN/L1K、Gather/Scatter 模式和 BlockScheduler
                                  |
                     单次 MIX kernel launch
                                  |
       +--------------------------+--------------------------+
       |                                                     |
      AIC                                                   AIV
       |                                                     |
 AIC Gather 或等待 A-ready                         indices GM -> UB
       |                                            AIV MTE/SIMT Gather
 A 全 K 长驻 L1 <---------------------------------- UB -> L1 zN
       |
 B GM/L2 -> L1 双缓冲
       |
 L1 -> L0A/L0B -> MMAD FP32 L0C
       |
 Fixpipe -> 单/双 AIV UB --------------------------> Cast FP16
                                                             |
                                                  MTE3/SIMT Scatter
                                                             |
                                                     D[indices, :]

56 AIV 分片清零 D；首个 Matmul tile 与清零重叠。通用路径在首次 Scatter 前执行一次
SyncAll<false>()；容量安全的大输出路径只清零未索引行，清零与 Scatter 地址不相交并跳过屏障。
```

当前任务调度按 K 分流：

- `K<=512`：AIV MTE2 Gather，UB 到 L1 转换，MTE3 Scatter。
- `K=768`：按完整任务 shape 实测表，在 AIV MTE Gather 和 AIC GM 到 L1 Gather 间选择。
- `K>=2048`：默认 AIC 逐行 GM 到 L1 `ND2NZ` Gather；M16/L1K448 容量边界使用已验证的 AIV MTE Gather。
- Scatter 与 Gather 独立选择；SIMT Scatter 组件保留用于动态扩展，当前任务调度全部使用实测更稳健的 MTE3 Scatter，`J1024/N4096/K2048` 也已撤销 SIMT 分流。
- `K>2048`：使用 M16/L1K128 AIC Gather 容量回退。

SIMT Gather/Scatter 组件仍完整实现，用于实测选择和动态扩展。SIMT 行内优先以 `uint2` 访问；不满足 8B 对齐时退化到 `half2` 和单元素尾部。Gather 使用 `asc_ldcg`，Scatter 使用 `asc_stcg`。
standalone 提供 `--force-simt` 探针；在随机数据和全 1 数据上强制执行 SIMT Gather/Scatter 均通过完整输出比较。

## Host 设计

### 参数结构

Kernel `Arguments/Params` 包含：

| 字段 | 说明 |
| --- | --- |
| `problemShape` | 逻辑 `(J,N,K)` |
| `physicalM` | A/D 物理行数 M |
| `ptrA/ptrB/ptrIndices/ptrD` | GM 地址 |
| `hardwareSyncAddr` | 全核同步地址 |

`GetWorkspaceSize()` 返回 0。`aclrtGetHardwareSyncAddr` 在 Host 启动前调用，失败时返回错误并打印错误码；kernel 入口调用 `SetSyncBaseAddr`。

optest 新增 `MatmulGatherScatterParams : MatmulParams`：`m` 表示物理 M，`j` 表示逻辑行数，`n/k` 表示 N/K，`inputAddr={A,B,indices}`，`outputAddr={D}`。

Python 接口为：

```python
torch_catlass.ascend950_matmul_gather_scatter(a, b, indices) -> d
```

adapter 检查 NPU、FP16 A/B、INT32 indices、二维连续 RowMajor 和形状关系，不读取 indices 内容。

### Tiling 配置

核心模板如下：

| 场景 | Tile | Gather | 输出 |
| --- | --- | --- | --- |
| `K<=512,N=128` | M128N128/L1K128/L0K128 | AIV MTE | 单 AIV FP16 |
| `K<=512,N>128` | M128N256/L1K128/L0K64 | AIV MTE | 双 AIV FP32 + Cast |
| `K=768,N=128,J<=512` | M16N128/L1K128/L0K128 | AIV MTE | 单 AIV |
| `K=768,N=128,J=1024` | M48N128/L1K128/L0K128 | AIC | 单 AIV |
| `K=768,N=128,J>=2048` | M80N128/L1K128/L0K128 | AIC | 单 AIV |
| `K=768,N>128` | M128N128 或 M128N256；J256/N256,N512 使用 M48N128 | 实测分流 | 单/双 AIV |
| `K=2048,N=128,J<=512` | M16N128/L1K128/L0K128 | AIC | 单 AIV |
| `K=2048,N=128,J>512` | M80N128/L1K128/L0K128 | AIC | 单 AIV |
| `K=2048,N>128` | M16/M32/M48/M80/M96 | 实测分流 | 单/双 AIV |
| `K>2048` | M16N128 或 M16N256/L1K128 | AIC | 单/双 AIV |

K=2048 按实际 B stage 计算 L1 容量：

| Tile | B stage | 主要用途 |
| --- | ---: | --- |
| M16N256/L1K448 | 2 | 小 J、总 tile 数不超过 56；AIV MTE Gather |
| M32N256/L1K384 | 2 | J512/N256 |
| M48N256/L1K320 | 2 | 一波任务的中小 J 大 N |
| M80N256/L1K128 | 3 | J512 至 J4096 的高并行区域；B 总占用 192KB |
| M96N128/L1K128 | 3 | J4096/N2048；B 总占用 96KB |
| M96N256/L1K128 | 2 | 其余大 K稳健回退 |

`K=512,N>=1024` 且默认 task 数恰为 64 时，可使用 TileN512 降低输出 tile 开销。TileM256、K=2048 TileN512、单 B stage 和 B 长驻方案经代表集实测后均未采用。

### Block 调度

当 `J>TileM` 时使用 `GemmIdentityBlockSwizzleL1FullLoad<1,0>`，让同一 J tile 的多个 N tile 尽量在同一 AIC 连续执行，复用 L1 A。部分 `J2048,N>=1024,K=2048` 和 `J4096,N4096,K=2048` 使用 `MatmulGatherScatterRotateMBlockSwizzleL1FullLoad`，按 M tile 轮转 N 起点，降低多核同时读取同一 B 地址的竞争。

当 `J<=TileM` 时使用 `GemmIdentityBlockSwizzle<3,0>` 展开 N tile。按 core 轮转不是全局双射，会在 M tile 跨核拆分时遗漏输出，已删除。

## Kernel 设计

### 输出清零

通用路径中，56 个 AIV 将 `M*N` 个 FP16 元素按 32B block 连续分片：

$$
workerId = GetBlockIdx()
$$

$$
workerNum = GetBlockNum() \times GetSubBlockNum()
$$

每个 AIV 在 UB 通过 `Duplicate` 生成零块，以最大 48KB 的 `DataCopyPad` 连续写 GM。分片互不重叠，尾部只写有效元素。

承担首个 Matmul tile 的 AIV 先加载 indices，并在 AIV Gather 模式下完成首块 Gather/A-ready；随后发布首个输出 buffer-free，再清零自己的 D 分片。AIC 因此可在 AIV 清零期间计算首个 tile，并把结果保存在输出 UB。

清零排空后，全部 AIC/AIV 成对调用一次 `SyncAll<false>()`。首个 Scatter 只能在屏障返回后开始，保证晚到的清零写不会覆盖索引行。后续 tile 不再执行全核屏障。

对 AIC Gather、`physicalM<=8192`、`K>=768`、`N>=1024`、输出不少于 4M 元素且 `J*4>=M` 的容量安全配置，每个 AIV 在 176-208KB 私有 UB 窗口构建 selected-row bitmap。indices 无重复，因此不同 SIMT 线程写入不同 bitmap 元素，不需要原子。AIV 将自己的物理行区间拆成连续未选中 run，只对这些 run 使用大块 MTE3 清零。

selective-zero 的目标地址与后续 Scatter 的 selected rows 严格不相交，AIC/AIV 均可跳过首次 `SyncAll<false>()`。bitmap 标记在 `J<=1024` 时使用 4 Warp，较大 J 使用 8 Warp。编译期 `SELECTIVE_ZERO_BUFFER_SAFE` 检查输出 stage 和 zero scratch 均不进入 bitmap 窗口；M128N512 输出 stage 约 192KB，自动回退到全量清零和屏障路径。

### Gather 路径

#### AIV MTE Gather

两个 AIV 按行平分当前 J tile。indices 通过连续 `DataCopyPad` 进入 UB；每个逻辑行根据 `physicalRow=indices[row]` 使用 MTE2 将一整行 A 搬入 RowMajor UB，再调用 `CopyUb2L1Tla` 完成 UB 到 L1 和 RowMajor 到 zN 转换。

MTE2/MTE3 之间使用硬事件保护同一 Gather UB。完成后两个 AIV分别设置 mode-4 A-ready flag，AIC 的 MTE1 管线等待。

#### AIV SIMT Gather

SIMT VF 使用 `__simt_vf__ __launch_bounds__(1024)`。`threadIdx.y` 表示 Warp/行任务，`threadIdx.x` 表示 Lane。行内对齐时每 Lane 使用 `uint2 + asc_ldcg`，否则使用 `half2`；一次 Warp 迭代形成连续合并访问。

`asc_vf_call` 为异步调用。每个 AIV在 212KB UB 偏移保留私有 completion word，使用递增 epoch 等待 VF 完成，不使用 GM 原子或 workspace。当前任务调度未把 SIMT Gather作为短任务默认路径，因为启动和轮询开销高于 MTE/AIC Gather。

#### AIC Gather

AIC 直接读取 indices。每个逻辑行通过 `TileCopyTla` 从 A 的 RowMajor GM 搬入 L1 zN 行，复用 FullLoadA 的 resident A 区。

独立 event 7 保证上一轮 L1 到 L0A 读取已排空，并保证全部 GM 到 L1 行拷贝在新 MMAD 前完成。AIC Gather 不占用 AIV Gather scratch，可将 UB 紧凑用于输出和清零。

### Matmul 路径

B 复用 CATLASS 原有 GM/L2 到 L1 和 L1 到 L0B 流水。默认使用两阶段 L1 B 缓冲；K=2048 的 M80N256、M96N128 使用 `L1K=128` 三阶段环形缓冲，JIT 通过 `CATLASS_JIT_L1_B_STAGES` 与 standalone 保持一致。A 当前 tile 全 K 长驻 L1，并在多个 N tile 间复用。

这里复用 FullLoadA 的含义不是把物理 A `(M,K)` 全量搬入 L1，而是把 Gather 后的逻辑 A tile `(TileM,K)` 沿 K 维完整驻留在 L1。实现复用的是 FullLoadA 已有的 resident-A 生命周期、L1 到 L0A 搬运、B 流式双/多缓冲和 MMAD 调度；原有 A 的连续 GM 到 L1 copy 被按 shape 选择的 AIV Gather 或 AIC 按 indices 直接 GM 到 L1 Gather 替换。同一个逻辑 A tile 只 Gather 一次，随后可被多个 N tile 复用，因此该方案与 A tile 全载直接相关，但不要求物理 A 全载。

Cube 路径使用：

- A/B：FP16；
- L0C 累加：FP32；
- `MmadAscend950FullLoadA`；
- `BlockMmadTla`；
- K 分块结果保留在 L0C，最终一次 Fixpipe 到 UB。

### Scatter 路径

`N=128` 使用单 AIV 输出：Fixpipe 直接产生 FP16，AIV0 Scatter。`N>128` 使用双 AIV 输出：Fixpipe 按 M 轴将 FP32 结果写入两个 UB，各 AIV通过 SIMD Cast 转 FP16。

MTE3 Scatter 对每行执行：

$$
dstOffset = indices[row] \times physicalN + nOffset
$$

SIMT Scatter 使用 `uint2/half2 + asc_stcg`，并以 AIV 私有 UB completion epoch 等待 GM store 排空。Cast 与 VF 之间使用 `V_S` 事件。indices 无重复，不使用原子写 D。

Scatter 完成后发布 buffer-free：SIMT 路径从 `PIPE_V` 发布，MTE3 路径从 `PIPE_MTE3` 发布，AIC 的 Fixpipe 管线等待后再复用输出 UB。

### 同步协议

| 同步 | 方向 | 作用 |
| --- | --- | --- |
| `SyncAll<false>()` | 全核 | 通用全量清零完成后开放首个 Scatter；selective-zero 路径跳过 |
| A-ready | AIV -> AIC | AIV Gather 已写 L1 A |
| A-reload-safe | AIC -> AIV | 允许下一 M tile 覆盖 Gather/L1 A 区 |
| C-ready | AIC -> AIV | Fixpipe 输出可 Cast/Scatter |
| buffer-free | AIV -> AIC | Scatter 已排空，输出 UB 可复用 |

C-ready 的消费者管线按路径绑定：双 AIV Cast 使用 `PIPE_V`，单 AIV SIMT Scatter 使用 `PIPE_S`，MTE3 Scatter 使用 `PIPE_MTE3`。该绑定避免异步 VF 提前读取或覆盖共享 UB。

### UB 与 Workspace

动态 UB 为 216KB：

- Gather scratch、FP32 output、FP16 Cast 按生命周期和 outputSlot 复用；
- indices 位于 208KB 偏移；
- SIMT completion word 位于 212KB 偏移；
- selective-zero bitmap 位于 176-208KB，仅在编译期布局安全的实例启用；
- 清零 scratch 使用输出 stage 暂未占用的空间，最大 48KB；
- 所有区域通过编译期 `static_assert` 检查不重叠。

SIMT 完成状态全部位于 AIV 私有 UB，`GetWorkspaceSize()` 为 0。算子唯一外部输出为 D。

### Scalar 优化

- standalone 与 JIT kernel 在 Ascend C 头文件前定义 `K_MAX_SHAPE_DIM=0`；
- `TPipe` 在 kernel `operator()` 局部创建并立即 `Destroy()`，不放入 resource 对象；
- Params 不重复传递可由 `GetBlockNum()`、`GetSubBlockNum()` 或模板推导的字段；
- 不在 kernel 内调用 Workspace 相关冗余接口。

## 组件组织

| 组件 | 作用 |
| --- | --- |
| `matmul_gather_scatter.hpp` | MIX kernel、清零、同步、Gather/Scatter 分流 |
| `matmul_gather_scatter_schedule.hpp` | 任务 shape 到 Tile/Gather/Scatter/Scheduler 映射 |
| `copy_gather_scatter_simt.hpp` | Ascend 950 SIMT Gather/Scatter VF |
| `MatmulGatherScatterTileCopy` | L0C 到单/双 AIV UB 输出 |
| numbered example | standalone 构建、精度和设备事件性能测试 |
| optest JIT kernel | 动态宏、MIX ABI、Torch adapter 和 pytest |

## 边界处理

- J/N 尾块通过 `actualShape` 控制。
- K 必须为 16 的倍数；非 128 对齐 K 使用补零和尾块计算。
- Scatter 物理行跨度始终为完整 N，不使用当前 TileN。
- A Gather 物理行跨度始终为完整 K。
- `K>2048` 使用 AIC Gather M16/L1K128 回退。
- indices 越界或重复属于调用方违反前置条件，行为不作保证。

# 支持范围

## 支持硬件

| 芯片 | 支持 |
| --- | --- |
| Ascend 950 | √ |

编译参数为 `-DCATLASS_ARCH=3510`，验证设备为 Ascend950PR_9579。

## 接口约束

1. A、B、D 仅支持 FP16，indices 仅支持 INT32。
2. A、B、D 仅支持连续 ND RowMajor。
3. A 的列数等于 B 的行数，indices 长度为 J。
4. J 不大于 M，K 为 16 的倍数。
5. indices 必须合法且无重复。
6. 输入输出内存不得非法重叠。

# 可维可测分析

## 精度标准

Golden 为：

```python
gathered = torch.index_select(a, 0, indices.to(torch.int64))
product = torch.matmul(gathered, b)
expected = torch.zeros((m, n), dtype=torch.float16, device=a.device)
expected.index_copy_(0, indices.to(torch.int64), product)
```

精度测试覆盖任务书 180 个 shape，并补充尾块、K>2048 回退、随机/升序/边界 indices。任务 shape 按 90/90 覆盖均匀分布 `[-5,5]` 与官方范围内均值/标准差的正态分布。FP16 判定采用 `rtol=atol=2^-9`、`matched_ratio>=0.99`，并要求每个元素满足 `|error|<=max(0.1, 32×ULP(reference))`。全部用例同时检查索引行精度和未索引行严格为零。

## 性能标准

加速比定义为：

$$
Speedup = \frac{T_{baseline}}{T_{fused}}
$$

性能测试使用 Ascend950PR_9579、CANN 9.0.0-beta.2。任务书 180 个 case 均通过独立 `msprof op --warm-up=20` 采集，fused 时间取对应 `OpBasicInfo.csv` 的 `Task Duration(us)`，baseline 使用任务测试集提供的数据。

除逐 case `baseline/fused` 外，同时统计算术平均、中位数和 `sum(baseline)/sum(fused)`。性能目标为算子整体相对小算子拼接方案达到 1.2x；若候选 TileShape 或 Gather/Scatter 路径只对单点有收益，则必须通过相邻 shape 和重复 A/B 测量后才能固化。

## 优化验证

实现阶段对 AIV SIMT Gather、AIV MTE Gather 和 AIC 按 indices 直接 GM 到 L1 三种路径做同条件对照。候选筛选保持设备、shape、indices、TileShape、stream、warmup 和测量次数一致，并记录任务时间、MTE/Cube 活跃时间和 L2 命中；最终入选调度再通过任务书 180 例 `msprof op` 验收。具体实验数据和结论记录在 numbered example 的 `README.md`，设计文档不汇总最终测试结果。

实现主要对照官方优化建议验证以下方向：

- 核间负载和 TileShape 实测调度；
- `TPipe` 外移、`K_MAX_SHAPE_DIM=0`；
- 零 workspace；
- B 默认 DoubleBuffer，M80N256/M96N128 三阶段 B 环形缓冲，L0C 暂存累加、A 长驻 L1；
- 大块 D 清零、规则路径使用 MTE；
- 大输出 selective-zero 只写未索引行并消除首次全核屏障；
- Gather/output/zero buffer 生命周期复用；
- SIMT `asc_ldcg/asc_stcg`；
- 按 M tile 改变 B 访问起点以缓解同地址竞争。

候选方案包括 TileM/TileN/L1K、B stage 数量、SIMT/MTE/AIC Gather、MTE3/SIMT Scatter、selective-zero、活跃 AIC 数量和调度轮转方式。候选必须同时通过精度、设备稳定性、代表集性能和完整任务集回归，避免只针对单个 shape 固化特判。

## 兼容性

本算子以独立 numbered example、Kernel、Schedule 和 optest 接口接入，不修改已有算子接口。代码依赖 Ascend 950 的 MIX、CV 通路、SIMT API 和全核同步能力，不向其他芯片宣称兼容。

## 风险分析

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 随机 Gather 带宽不足 | A 搬运成为瓶颈 | 小 K MTE、大 K AIC `ND2NZ`、A FullLoad |
| B 多核同地址竞争 | 大 K 大 N 性能下降 | 部分 shape 按 M tile 轮转 N 起点 |
| 清零覆盖 Scatter | 数值错误 | 通用路径首次 Scatter 前执行一次 `SyncAll<false>()`；selective-zero 只写未索引行 |
| 异步 VF 未完成即复用 UB | 偶发漏写或覆盖 | AIV 私有 completion epoch、正确管线发布 buffer-free |
| L1/UB 生命周期重叠 | 设备错误或数据破坏 | 静态容量断言、A-reload-safe、outputSlot 双缓冲 |
| bitmap 与输出/清零 scratch 重叠 | 数据破坏 | `SELECTIVE_ZERO_BUFFER_SAFE` 编译期门槛，M128N512 自动回退 |
| M16/L1K448 AIC Gather 稳定性 | 设备错误 507015 | 该配置固定使用 AIV MTE Gather |
| 表外动态 shape 性能波动 | 未达到最优 | 使用稳健回退，不针对单次测量特判 |

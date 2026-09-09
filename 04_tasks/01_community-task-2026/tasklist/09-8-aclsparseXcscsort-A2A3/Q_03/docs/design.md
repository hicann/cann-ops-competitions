# aclsparseXcscsort 算子设计文档（A2/A3）

## 一、需求背景

### 1.1 需求来源

本设计对应 CANN 社区 2026 年 9 月任务“aclsparseXcscsort 算子开发（A2/A3）”。目标是在
Atlas A2/A3（DAV_2201，`arch22`）上补齐 `aclsparseXcscsort` 的 CSC 纯索引稳定排序能力，
并将公开 C++ 接口、Host、Ascend C Kernel 和测试统一交付到 `cann/ops-sparse`。

任务书以 cuSPARSE `cusparseXcscsort` 作为接口语义与 GPU Event 性能标杆，但 NPU 实现只
使用 aclsparse 与 Ascend C，不调用 cuSPARSE 或其他后端。公开接口只有 I32 索引，没有
values、compute dtype、算法枚举或转置参数。

### 1.2 算子功能

给定 CSC 矩阵的列指针 `cscColPtr`、行索引 `cscRowInd` 和调用方初始化的置换向量 `P`，
对每列区间

```text
[cscColPtr[j] - base, cscColPtr[j + 1] - base)
```

执行 row index 的稳定升序排序，并用同一置换重排 `P`。`cscColPtr` 只读，
`cscRowInd` 和 `P` 原地更新；因此当 `P[i] = i` 时，调用方可以用排序后的 `P`
同步重排外部 values。

## 二、需求分析

### 2.1 对外接口

```cpp
aclsparseStatus_t aclsparseXcscsort_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const int *cscColPtr, const int *cscRowInd,
    size_t *pBufferSizeInBytes);

aclsparseStatus_t aclsparseXcscsort(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const aclsparseMatDescr_t descrA,
    const int *cscColPtr, int *cscRowInd, int *P, void *pBuffer);
```

实现沿用 `ops-sparse` 已有 Legacy API、矩阵描述符、handle 和调用方 stream。两阶段语义为：
先通过 `bufferSizeExt` 查询精确 workspace，再在同一 stream 上异步执行排序。

### 2.2 参数与不变量

| 对象 | 设计约束 |
| --- | --- |
| `m/n/nnz` | 非负；`m==0` 或 `n==0` 时 `nnz` 必须为 0 |
| `descrA` | 提供 index base，必须为 0 或 1；生命周期覆盖异步调用 |
| `cscColPtr` | I32，长度 `n+1`，单调非降，端点为 `base` 和 `base+nnz`，只读 |
| `cscRowInd` | I32，长度 `nnz`，按列原地稳定升序排序 |
| `P` | I32，长度 `nnz`，按 `cscRowInd` 的稳定置换同步重排；不是 workspace |
| `pBuffer` | 来自 `bufferSizeExt`，128 字节对齐，不与输入输出重叠 |

归一化后的 row index 必须属于 `[0,m)`；重复 row key 必须保持输入相对顺序。`nnz==0`
允许按公开 API 规则使用空数据指针并直接成功返回。执行 ABI 不携带实际 workspace size，
因此实现精确查询、溢出和对齐检查，不承诺检测调用方欠配的 buffer。

### 2.3 精度、性能和内存门槛

精度采用 CPU 稳定 I32 排序 golden，逐项 exact match `cscColPtr`、`cscRowInd` 和
`P`，
不使用浮点容差，也不能只检查“结果有序”。相同输入重复执行应 bit-wise 一致。

性能倍率定义为：

```text
GPU 标杆接口 Device Event median_us /
NPU 相同调用范围内全部 Kernel 总耗时
```

每个 case 至少 warmup 10 次、正式采样 30 次，报告 median 和 p90；描述符和 workspace
在采样期间复用，每轮恢复 `cscRowInd/P`，不计首次编译、数据生成和搬运。P-01/P-02/P-03
的 base 0/base 1 case 均需达到至少 `0.25x` GPU 标杆。外部任务包中的 A100 数据只作为
固定标杆来源，不能替代 A2/A3 NPU 实测。

内存验收优先使用纯索引方案的固有 workspace：workspace 不超过目标硬件 L2 Cache；若
形成等价 GPU 调用范围且输入输出总量超过 500 MB，再补充 NPU 相对 GPU 的额外内存不超过
GPU 总量 50% 的比较。

## 三、详细设计

### 3.1 总体数据流

```mermaid
flowchart LR
    A[Host 参数/描述符校验] --> B[bufferSizeExt 查询]
    B --> C[按 stream 异步下发 arch22 Kernel]
    C --> D[按列定位与分核]
    D --> E[稳定排序 row index + 同步重排 P]
    E --> F[原地写回 cscRowInd/P]
```

Host 只负责 ABI 校验、workspace 计算、tiling 和 Kernel 启动；排序、列边界读取和结果写回
全部在 NPU 完成，不插入 Host 同步或 CPU 计算。

### 3.2 Host 侧设计

#### 3.2.1 参数检查和 workspace

统一的 `ValidateCscsortParams` 检查 handle、维度、描述符、指针、列指针端点/单调性、
index base、row bounds 前置条件、输入输出重叠和 workspace 对齐。
`bufferSizeExt` 按
`nnz` 计算两个 I32 scratch 区域：

```text
workspace_bytes = 2 * nnz * sizeof(int32_t)
```

乘法前检查 `size_t` 溢出；`nnz==0` 返回 0。执行阶段复用该查询值，不新增 size 参数。
空矩阵/空列、`nnz==0/1` 走明确早退或单元素路径，不启动无实际工作的 block。

#### 3.2.2 tiling 和分核

Host 从平台信息获取 AIV 数量和 UB 容量，形成定长 `CscsortTilingData`：
`n/nnz/indexBase/runSize/coreNum`。`coreNum` 不超过有效列数和非零数；
按累计 nnz 将
完整列区间均衡分给各 block，单列不拆分。Kernel 通过 `cscColPtr` 二分定位每个
block 的起止列，保证不同 block 的 row/P/workspace 区间不相交，因此不需要全局同步。

`runSize` 由 UB 可用空间反推，至少容纳 key/payload 的双缓冲和归并临时区，并按
128 字节搬运约束取整。若平台 UB 不足以容纳最小合法 run，Host 返回资源不足，
不切换到 CPU 路径。

#### 3.2.3 tilingKey 和启动

本算子不按公开 shape 或 case id 生成 tilingKey。base 作为运行时参数传入，短列/长列由
Kernel 根据实际列长选择。Host 在 handle 绑定的 stream 上只启动一个 AIV Kernel，保持
异步语义；首次编译和数据准备不计入性能测量。

### 3.3 Kernel 侧设计

#### 3.3.1 短列路径

长度不超过 `runSize` 的列采用 GM→UB→GM 单趟路径：

1. 将 row index 和对应 `P` 成对搬入 UB；
2. 以 32 个元素为小块做稳定插入排序，只有 `key > current` 时移动元素；
3. 使用 bottom-up 两路归并扩大有序 run，比较相等 key 时优先取左 run，确保稳定性；
4. 将排序后的 key/P 成对写回原列区间。

key 和 payload 始终成对处理，不能先排序 row 再用另一套索引猜测置换。尾块按实际长度
屏蔽搬运和存储，禁止越界访问。

#### 3.3.2 长列路径

长度超过 `runSize` 的列分成多个 run。第一阶段逐 run 复用短列排序并写回 GM；第二阶段在
原数组和 workspace scratch 之间做多趟 bottom-up 归并，每趟交换 source/destination，
以 `<=` 规则保持重复 key 稳定。若最终结果落在 scratch，则在 Kernel 内回拷到原数组。
workspace 按列区间切片使用，不能跨 block 共享可变状态。

#### 3.3.3 同步和异步边界

GM/UB 搬运、标量排序和向量搬运之间使用 Ascend C 队列事件及必要的 pipe barrier；只在
本 block 内同步。Kernel 不调用 Host API、CPU golden、cuSPARSE、Torch/NPU 框架排序或
其他 backend 实现。所有异常参数在 Host 阶段拒绝，设备端只处理已通过接口前置条件的
合法数据。

### 3.4 A2/A3 与 A5 解耦

本任务代码固定放在 `sparse/cscsort/arch22/`，测试放在 `test/cscsort/arch22/`，
公共声明只更新 `include/cann_ops_sparse.h`。公共 Host 语义和 golden 可以复用，但排序
Kernel 不依赖 arch35 专有的 SIMT 或排序高级 API。A5 后续合入时以已合入版本为基线，
解决公共 Host 冲突并执行两硬件范围的交叉回归；本设计不通过新增同名接口或运行时
backend fallback 规避架构差异。

## 四、测试与验收计划

### 4.1 C++ UT/ST 与端到端流程

`test/cscsort/arch22/` 提供独立 CPU golden 和 C++ UT/ST，覆盖：

- 空矩阵、空列、`nnz=0/1`、已排序/逆序/随机列和单长列；
- base 0/base 1、重复 row index 及 P 稳定置换；
- 多核切分、runSize 临界、长列多 run 归并、奇偶归并趟数；
- `bufferSizeExt` 精确值、溢出、空指针、未对齐 workspace、非法维度/列指针和生命周期；
- cscColPtr 只读、row/P 原地语义、输出边界、stream 异步和连续创建/销毁无泄漏。

端到端测试必须确认 NPU 注册或 C++ 调用真正命中 Kernel，并用 Dispatch/Profiler 证据
证明没有 CPU fallback。测试 README 写明 CANN、驱动、固件、硬件型号、代码 SHA、编译和
运行命令。

### 4.2 固定性能用例

性能脚本以任务包 `performance_cases.json` 为统一输入，保留三组模型维度锚点及
base 0/1：

| 场景 | CSC 维度 | nnz | 目标 |
| --- | ---: | ---: | --- |
| P-01 Llama 3.1 70B MLP | `8192 x 28672` | 524288 | 每个 base case `>=0.25x` |
| P-02 Qwen3-235B-A22B MoE | `4096 x 1536` | 262144 | 每个 base case `>=0.25x` |
| P-03 DeepSeek-V3 MoE | `7168 x 2048` | 458752 | 每个 base case `>=0.25x` |

两侧使用完全相同的固定 seed、CSC 列指针、row index、P 和 base。报告逐 case 列出
bufferSizeExt、Kernel 总耗时、完整流程补充耗时、workspace、列长分布、median/p90、
倍率和原始日志，不以平均值掩盖单 case 未达标。

### 4.3 当前证据边界

本设计阶段不宣称 A2/A3 功能、性能、workspace、Dispatch、Profiler 或内存已通过。外部
GPU 标杆只能作为性能输入；所有最终结论必须来自目标硬件上的 fresh C++/NPU 运行和可回查
的版本、环境及原始证据。自测报告完成后，代码 PR 需另行按任务书路径提交到 `ops-sparse`。

## 五、no-fallback 与交付边界

被测实现只允许使用 `ops-sparse` 自有公共头文件、`sparse/cscsort/arch22/` 源码和目标
CANN/Ascend C 公共运行时。禁止在运行时读取 golden、任务答案、baseline cache、peer
workspace、cuSPARSE、CPU/reference 或其他 backend；禁止按公开 case id/shape 硬编码分支。

设计 PR 只包含本任务目录下的 `docs/design.md`，不混入算子源码、生成日志、Profiler
目录、缓存或未验证性能数字。后续完整验收还必须提供：

1. `ops-sparse` 个人仓库、分支、目录和 `Ascend-CANN` 开发者邀请；
2. `sparse/cscsort/arch22/` Host/Kernel、公共声明及 `test/cscsort/arch22/` C++ UT/ST；
3. 200 条 accuracy、正式性能锚点及扩展压力用例的结果、日志和 Profiler/Dispatch 证据；
4. 峰值内存、环境版本、提交 SHA、失败项和复现 README；
5. 本设计 PR 评审通过后再进入代码开发和最终 IT 验收。

## 六、参考资料

- `docs/qsj/aclsparseXcscsort.zip` 中的《aclsparseXcscsort 算子开发任务书（A2/A3）》；
- [ops-sparse](https://gitcode.com/cann/ops-sparse) 及其 `include/cann_ops_sparse.h`；
- [cuSPARSE Xcscsort 文档](https://docs.nvidia.com/cuda/cusparse/);
- [社区设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)；
- [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。

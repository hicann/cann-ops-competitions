# aclsparseScatter 算子（Atlas A2/A3）设计文档

> 任务：9月社区任务-aclsparseScatter算子开发（A2/A3）
> 提交者（GitCode）：Aiminer
> 提交路径：`04_tasks/01_community-task-2026/tasklist/09-12-aclsparseScatter-A2A3/Aiminer/docs/design.md`
> 任务详情页：https://www.hiascend.com/activities/task-center/details/eaed79990e1441b2afbe19501a6ab343
> 开源仓：https://gitcode.com/cann/ops-sparse （实现合入 `sparse/scatter/arch22/`，测试合入 `test/scatter/arch22/`，Python/ATen 适配随本任务首次建立）

# 需求背景（required）

## 需求来源

CANN 社区任务 2026（9月批次）：在 Atlas A2/A3 系列产品（DAV_2201，`arch22`，含 910B3/910B4 及任务环境提供的 Atlas A3 型号）上，基于 Ascend C 完善并补齐 `aclsparseScatter` 算子，交付 C++ Host/Kernel、Python/ATen NPU 适配、C++ UT/ST 与端到端 UT，合入 ops-sparse `master` 分支。任务书：9月社区任务-aclsparseScatter算子开发（A2/A3）。

## 背景介绍

### aclsparseScatter 算子功能

`aclsparseScatter` 是稀疏向量到稠密向量的原地散布写入算子，接口语义对齐 cuSPARSE `cusparseScatter`（cuSPARSE 13.3 Update 1）：

$$Y[X.indices[i] - idxBase] = X.values[i],\quad i \in [0, nnz)$$

- `vecX` 为只读稀疏输入（SpVec 描述符：I32 indices + values），`vecY` 为稠密输入输出（DnVec 描述符），接口不修改 `vecX`；
- `nnz=0` 时成功返回且不修改 `vecY`、不启动 Kernel；
- 重复索引为非确定性 last-write-wins（不得实现为累加，不承诺确定输出）；无重复索引时输出必须 bit-wise 一致；
- 无 workspace、无 preprocess 阶段，零额外内存：原地复用 `vecY`，不得分配与输入规模线性相关的临时 Device/Host 内存。

常用于图计算、稀疏特征回填、词表索引写入（任务书 P 场景即 Llama 3.1 70B / Qwen3-235B / DeepSeek-V3 词表规模）与稀疏格式转换前后的数据整理。

### 现状分析

- **对标基线**：cuSPARSE `cusparseScatter`，C++ 接口语义逐项对应。
- **仓内已有实现**：ops-sparse `sparse/scatter/arch22/` 已存在一版 arch22 实现（`scatter.h/scatter_host.cpp/scatter_kernel.h/scatter_kernel.cpp`），**当前仅支持 values=FP32、index base 0**：Host 侧对 `valueType != ACL_FLOAT`、`idxBase != BASE_ZERO` 直接返回 `INVALID_VALUE`/`NOT_SUPPORTED`；`sparse/scatter/arch35/`（A5/950）为同源基线。本任务在其工程骨架上补齐声明全量能力。
- **接口声明**：`aclsparseScatter(handle, vecX, vecY)` 已声明于 `include/cann_ops_sparse.h`（第 1861 行），SpVec/DnVec 描述符族 API（create/get/set、index base 枚举）齐备，禁止产品私有平行接口。
- **测试侧**：`test/scatter/` 已有 CMakeLists、`scatter_golden.h`、`scatter_param.h` 与 `arch22/scatter_test.cpp`、`arch35/` 目录，本任务扩展 arch22 测试并新增 ATen/Python 端到端测试。
- **Python/ATen 层**：ops-sparse 仓当前**尚无** torch 层目录，本批次稀疏任务为首批要求交付者，Python/torch 层接口、ATen Dispatcher NPU 注册的目录结构由本任务建立（公共骨架，可被同批次其他算子复用）。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（arch22）上完善 `aclsparseScatter`：支持 `int8`、`float16`、`bfloat16`、`float32`、`complex64` 五种 values dtype、I32 索引、index base 0/1、动态 `size/nnz`、乱序/重复索引与 `nnz=0/1` 边界；交付 Python/ATen 适配（公开入口 `Tensor.index_copy_(0, index, source)` → `aten::index_copy_`，PyTorch 2.7+、torch_npu 26.0.0+，不得 CPU fallback）；精度、性能、内存满足任务书 §3 验收标准，完成设计、开发、测试全流程。

## 需求拆解

1. **Host 侧补齐**：在既有 arch22 Host 上放开 dtype/base 限制——增加五 dtype 分发、base 0/1 透传、`vecY` 与 `vecX.values` dtype 一致性校验、`nnz=0` 快速返回，保持调用方 stream 异步下发；
2. **Kernel 侧补齐**：将现有 FP32 kernel 泛化为五 dtype（int8 按 8bit 透传、complex64 按 2×float32 语义），索引换算统一减 `idxBase`，保留多核 power-of-2 tiling + DoubleBuffer + 连续 run 向量化拷贝优化路径；
3. **Python/ATen 适配**：建立 torch 层目录，注册 `aten::index_copy_` NPU 实现，校验 `dim=0`、dtype/shape/stride/device，转换 SpVec/DnVec 描述符并调用 `aclsparseScatter`，原地返回；
4. **精度达标**：无重复索引 bit-wise exact match；有重复索引验证输出来自对应 values 之一；golden 口径 int8 原始 bit、fp16/bf16 用 fp32 golden、fp32 用 fp64 golden、complex64 用 complex128 golden；
5. **性能达标**：P-01/P-02/P-03 全部有效 case 性能倍率（标杆 GPU Event / NPU 总 Kernel 耗时）≥ 0.25，预热≥10 次采样≥30 次报中位数和 p90；
6. **测试交付**：`test/scatter/arch22/` 扩展 C++ UT/ST，新增 ATen UT 与 Python 端到端 UT，覆盖任务书"自验证用例"表四类场景。

# 详细设计（required）

## 算子分析

### 数学公式

见"背景介绍"。逐元素语义：对每个 `i∈[0,nnz)`，将 `X.values[i]` 写入 `Y[X.indices[i]-idxBase]`。无归约、无数据依赖（不同 `i` 写不同目标位置时天然并行；写同一位置时为非确定 last-write-wins）。未写入的 `Y` 元素保持不变。

### 支持数据类型

| 项 | 规格 |
| --- | --- |
| `vecX.values` / `vecY.values` | `int8`、`float16`、`bfloat16`、`float32`、`complex64`（两者 dtype 必须一致） |
| `vecX.indices` | 仅 I32；index base 0/1 |
| Kernel 内部表示 | int8 原始字节透传；fp16/bf16 不做数值变换直接搬移；fp32 原样；complex64 按 (real, imag) 两个 fp32 分量搬移，元素粒度保持 8 字节对齐处理 |

### 支持形状

一维向量：`vecX.indices/values` 均为 `[nnz]`，`vecY.values` 为 `[size]`（连续）。`nnz≥0` 动态，`nnz>0` 时要求 `size>0` 且换算 base 后 indices 位于 `[0,size)`。无 stride/广播/dynamic shape tiling 重排需求。

## 算子实现

### 实现方案

复用仓内已验证的 arch22 工程骨架（多核均分 + power-of-2 tile + DoubleBuffer + run 向量化），在其上做 dtype/base 泛化，并新建 Python/ATen 适配层。文件规划：

```
sparse/scatter/arch22/
├── scatter.h              # 扩展：tiling 增加 idxBase/elemSize/dtype 标签字段
├── scatter_host.cpp       # 扩展：五 dtype 分发、base 0/1、dtype 一致性校验
├── scatter_kernel.h       # 不变（kernel_do 声明）
└── scatter_kernel.cpp     # 扩展：dtype 模板化 kernel + base 换算
python/                     # 新建（本任务首批 torch 层公共骨架）
├── aclsparse_scatter_npu.cpp   # aten::index_copy_ 的 NPU 实现（Dispatcher 注册）
├── aclsparse_scatter_pybind.cpp
└── aclsparse/ops/scatter.py    # Python 侧包装与校验
test/scatter/arch22/
├── scatter_test.cpp       # 扩展：五 dtype × base 0/1 × 乱序/重复/边界
├── scatter_aten_test.cpp  # 新增：ATen UT
└── scatter_e2e_test.py    # 新增：Python 端到端 UT
```

### Host 侧设计（scatter_host.cpp）

1. **参数校验**（保持既有顺序，扩展检查项）：
   - `handle/vecX/vecY` 任一为 nullptr → `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`；
   - `idxType != ACL_SPARSE_INDEX_32I` → `NOT_SUPPORTED`（描述符层 I64 属未声明能力）；
   - `idxBase` 非 `BASE_ZERO/BASE_ONE` → `INVALID_VALUE`；
   - `valueType ∉ {ACL_INT8(如声明枚举缺失则按头文件实际枚举为准), ACL_FLOAT16, ACL_BFLOAT16, ACL_FLOAT, ACL_COMPLEX64}` 或与 `vecY` dtype 不一致 → `INVALID_VALUE`；
   - `spVec->size > dnVec->nums`、`nnz > 0 && nums == 0`、indices/values 指针为空 → `INVALID_VALUE`；
   - 索引越界不在 Host 常规路径扫描（避免 D2H 同步），按调用方前置条件/异步错误协议处理，README 明示。
2. **快速路径**：`nnz==0` 直接返回 SUCCESS，不下发 kernel。
3. **dtype 分发**：Host 按 `valueType` 计算 `elemSize`（1/2/2/4/8 字节），写入 tiling 并选择 kernel 入口；kernel 符号按 dtype 实例化 5 份（链接期展开，无运行时开销），`yLen` 字节长度按 `size*elemSize` 计算（修正现有代码硬编码 `sizeof(float)` 处）。
4. **Tiling**：沿用现有策略——核数从 `aclrtGetDeviceAttr(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 获取，`blockNum=min(nnz, 可用核数, SCATTER_MAX_CORE_NUM=64)`，均分 `nnz` 得每核 `coreNnzCount/coreNnzOffset`；tile 取不超过 `SCATTER_TILE_NN_MAX=4096` 的最大 2 的幂（下限 8）。`ScatterTilingData` 增加 `idxBase`、`elemSize` 两个标量字段（定长数组布局不变，向后兼容）。
5. **异步执行**：tiling 在 Host 栈上构造后以 `const` 引用随 kernel 参数下发，设备指针保持原始地址（无 D2H），沿用 `handle` 绑定 stream，`scatter_kernel_do(valDev, idxDev, yDev, tiling, blockNum, stream)`。

### Kernel 侧设计（scatter_kernel.cpp）

1. **执行形态**：AIV 多核并行，每核独立处理自己的 `[coreNnzOffset, coreNnzOffset+coreNnz)` 连续区段；DoubleBuffer（VECIN2+MTE2）流水搬运 indices 与 values 到 UB。
2. **base 换算**：run 检测与目标地址计算统一使用 `idx - idxBase`（`idxBase` 由 tiling 传入，kernel 内一次读出放入寄存器）；不修改 GM 上原始 indices（`vecX` 只读）。
3. **连续 run 向量化路径**（保留现有优化）：顺序扫描 UB 内 indices，检测 `curIdx == prevIdx + 1` 的连续段；连续段且源 8 元素对齐时用 `Copy<float>`（按元素数×elemSize 折算 32bit 块）整段搬移到 `Y[targetIdx..]`，否则逐元素 `SetValue`。重复索引（`curIdx == prevIdx`）自然落入非连续分支按 last-write-wins 处理。该路径对排序/近排序 indices（性能 P case 的词表索引场景）收益最大。
4. **dtype 泛化**：
   - kernel 主体模板化 `template <typename T>`（int8_t/__half/__bf16/float 与 complex64 的 float2 视图）；
   - UB 队列的 tile 字节预算按 `elemSize` 折算：values 队列 `aligned8 * elemSize` 字节、indices 队列 `aligned4 * 4` 字节，`DataCopyPad` 的 `blockLen` 以字节为单位按 elemSize 换算，保证 int8/fp16/bf16 尾块安全；
   - complex64 按 `float` 双分量连续存储处理（元素数 ×2 个 float），段拷贝粒度天然 8 字节对齐；
   - int8 段拷贝最小粒度按 32bit 对齐折算，不足一个 4 字节块的尾段走逐元素路径。
5. **确定性口径**：无重复索引时各核写入位置不相交，输出 bit-wise 确定；有重复索引时多核写入顺序不定，属任务书允许的非确定 last-write-wins，README 明示。

### Python/ATen 适配设计

1. **入口映射**：`Tensor.index_copy_(0, index, source)` → `aten::index_copy_`。注册 NPU Dispatcher 实现（torch_npu 26.0.0+ 的扩展机制），拦截 `dim==0`、一维 `index`（I64 → I32 顺搬或直接按 I32 描述）、`source` 一维且 `numel == index.numel`、`self.numel >= max(index)` 等条件，不满足或 dtype 不在五类之内时显式报错（对齐 PyTorch 异常语义），**任何路径不得 CPU fallback**。
2. **描述符转换**：`index`（Dense I32/I64 Tensor）+ `source` 组装 `aclsparseSpVecDescr`（size=`self.numel`，base 0），`self` 组装 `aclsparseDnVecDescr`，复用 `aclsparseCreate(Handle/SpVec/DnVec)` 公共接口；stream 取当前 NPU stream 绑定 handle。
3. **原地语义**：调用 `aclsparseScatter` 后直接返回 `self`（in-place）；重复索引遵循 PyTorch `index_copy_` 的 last-write-wins 语义说明。
4. **测试**：ATen UT 覆盖 Dispatcher 注册、参数/dtype/shape/stride/device 校验、原地输出、无 CPU fallback（Profiler 证据）；Python e2e UT 覆盖 `index_copy_` 与 CPU golden 对比、五种 dtype、异常分支、重复/乱序索引、`nnz=0/1`、尾块。

### 与既有实现的关系

- 公共 API 不变（`include/cann_ops_sparse.h` 既有声明），既有 FP32/base0 行为完全兼容（本任务是其超集）；
- arch35（A5/950）路径不动，A2/A3 与 A5 公共 Host 逻辑按仓内既有解耦约定组织，后续 A5 PR 合入时按任务书要求处理冲突并交叉回归；
- torch 层目录为本任务新建的公共骨架，目录与命名遵循仓内评审意见可调整。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列 910B3 / 910B4（arch22） | √ |
| Atlas A3 系列（任务环境提供型号，arch22） | √ |
| A5/950（arch35，既有实现，不在本任务范围） |  |

## 算子约束限制

1. `vecX.values` 与 `vecY` dtype 必须一致；indices 仅 I32；base 仅 0/1；
2. 换算 base 后 indices 须位于 `[0, size)`：越界由调用方前置条件保证或按异步错误协议暴露，Host 常规路径不扫描 Device indices（避免隐式同步）；
3. 重复索引非确定 last-write-wins，不实现为累加，不承诺确定输出；性能 case 使用无重复索引；
4. `nnz=0` 成功返回且不修改 `vecY`；`vecX` 整体只读；未声明的 `vecX/vecY` alias 语义返回参数错误（原地写 `vecY` 为声明的唯一输出）；
5. 接口无 workspace、无 preprocess；不分配与输入规模线性相关的额外 Device/Host 内存（任务书 §3.4 内存口径二天然满足：固有 workspace 为 0）；
6. 最大 `size/nnz` 边界（I32 索引范围、`size*elemSize` 字节不溢出 int64）在 Host 校验并在 README/接口说明中明确。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | CPU Golden 单标杆：int8 按原始 bit、fp16/bf16 用 fp32 golden、fp32 用 fp64 golden、complex64 用 complex128 golden；无重复索引逐元素 bit-wise exact match；有重复索引验证输出 ∈ 对应输入 values 集合；校验 `vecX` 只读与未写入 `Y` 元素不变；覆盖正负零、INF/NAN | 任务书 §3.2 + 生态算子开源精度标准 |
| 性能标准 | P-01（size=128256/nnz=8192）、P-02（151936/4096）、P-03（129280/7168）全部有效 case（五 dtype × base 0/1，各 10 条）：倍率 = 标杆 GPU 设备 Event median_us / NPU 同范围总 Kernel 耗时 ≥ 0.25；标杆区间 83.0–92.8us；预热≥10、采样≥30、报 median/p90，不计首次编译/数据生成/H2D 搬运 | 任务书 §3.3 |
| 内存标准 | 无 workspace（口径二：固有 workspace 0 ≤ L2 Cache）；峰值内存按任务包 `collect/compare_sparse_ops_memory.py` 脚本对相同 case id 对比 | 任务书 §3.4 |

## 测试方案

1. **测试框架**：扩展 `test/scatter/arch22/`（复用 `scatter_golden.h`、`scatter_param.h` 公共组件），CMake 纳入五 dtype 编译目标；新建 ATen UT 与 Python e2e UT（torch_npu 环境）。
2. **C++ UT/ST 用例**（对应任务书"自验证用例"表）：
   - 基础功能：不同 `size/nnz`（含动态规模、尾块、`nnz=0/1`）、顺序/乱序/重复索引、base 0/1、五 dtype 全覆盖；
   - 接口与异常：空 handle/描述符/指针、dtype 不一致、非法 idxType/base、`size<nnz 上界` 类非法 shape、`vecX` 只读性校验；
   - 一致性：无重复索引 exact match（bit-wise）、重复索引 last-write-wins、complex64 专项（实/虚部分别 bit 比对）、INF/NAN、重复执行 bit-wise 一致；
   - NPU 与资源：NPU Dispatch/Profiler 证据（无 CPU fallback）、描述符连续创建/执行/销毁无泄漏、无非法同步。
   - 数据生成：values 70% `[-1,1]` 均匀 + 20% 正态(μ=0,σ=1) + 10% 零/边界/INF/NAN；indices 固定种子均匀生成并专项覆盖重复、首尾、越界异常（负向）。
3. **性能用例**：任务包 `performance_cases.json` P-01/P-02/P-03（无重复索引），NPU 侧单流连续调用采集全部 Kernel 总耗时，配合任务包 `verify` 脚本产出倍率表；另跑连续 run 向量化路径与逐元素路径的分组对比，验证优化路径收益。
4. **内存用例**：任务包 `collect_sparse_ops_npu_memory.py` + GPU 侧对照，`compare_sparse_ops_memory.py` 出对比表（本算子无 workspace，预期峰值=输入输出本体）。

## 兼容性分析

- 接口层：`include/cann_ops_sparse.h` 无 API 变更，既有 FP32/base0 调用方行为不变（纯能力扩展）；
- 实现层：`sparse/scatter/arch22/` 原地扩展（tiling 结构新增字段向后兼容）；arch35 不动；A2/A3 与 A5 公共逻辑解耦，后合入方处理冲突并交叉回归；
- `README.md`：`sparse/scatter/README.md` 产品支持表将 Atlas A2/A3 标注"支持"，补齐五 dtype/base1 能力说明、最大规模与重复索引语义说明；
- torch 层为本仓新增目录，无存量兼容问题；与 torch_npu 26.0.0+ 的 `index_copy_` 既有 NPU 路径关系（优先级/注册方式）在联调阶段与仓内评审确认。

# 交付件与计划

| 交付件 | 说明 |
| --- | --- |
| 设计文档 PR | 本文档，提交至 cann-ops-competitions `04_tasks/01_community-task-2026/tasklist/09-12-aclsparseScatter-A2A3/Aiminer/docs/design.md` |
| 算子代码 | 个人 ops-sparse fork，`sparse/scatter/arch22/` 扩展 + `python/` torch 层（合入目标 https://gitcode.com/cann/ops-sparse master） |
| 测试代码与用例 | `test/scatter/arch22/`（C++ UT/ST + ATen UT + e2e UT），覆盖五 dtype、base 0/1、乱序/重复索引专项 |
| 自测报告 | 按官方腾讯文档模板：用例参数、精度结果与截图、性能数据（P-01/02/03 倍率表）、峰值内存、Profiler 证据 |
| README 更新 | `sparse/scatter/README.md` 产品支持表标注 A2/A3 支持、能力矩阵与约束说明 |

> 开发环境：hidevlab A2/A3 算力（或等价 910B3 环境，CANN 9.1.0+），性能自验证优先 910B3；910B4/A3 验收测试申请任务算力完成。开源仓免费时长额度内使用，闲时关停。
> 排期（截止 2026-09-22）：第 1 周完成 Host/Kernel dtype 泛化与 C++ UT；第 2 周完成 ATen/Python 适配与 e2e UT、性能调优；第 3 周补齐内存/Profiler 证据与自测报告，提交验收。

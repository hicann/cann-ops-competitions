# aclsparseSparseToDense 算子设计文档（A5 / Ascend 950PR）

| 文档版本 | 日期 | 作者 | 变更 |
|---|---|---|---|
| v1.0 | 2026-09-03 | zhangfeng1133 <yanggg1133@163.com> | 初版：基线达标设计（AIV 零化内核 + complex64 + ATen 组合键适配），950PR 实机验收数据 |

> 交付仓：cann-ops-competitions（本文件路径：`doc/design.md`）
> 配套：自测报告 `SELF_TEST_REPORT.md`｜截图证据 `pic_my/`｜代码补丁 `npu_wrapper/patches/ops_sparse_arch35_v1.patch`
> 状态：已实现并通过 950PR 实机验收（精度/性能/内存/ATen 全部达标）

## 1. 概述

### 1.0 环境与版本

| 项 | 值 |
|---|---|
| 硬件 | Ascend 950PR（A5，DAV_3510） |
| CANN | 9.2.0-beta.1（inner V100R001C12B056，满足任务书 9.1.0+） |
| 编译器/构建 | gcc + cmake 3.22.1（ops-sparse `build.sh --soc=ascend950`） |
| PyTorch / torch_npu | 2.12.0 / 2.12.0（Ascend 26.0.0 配套代际，满足任务书 26.0.0+） |
| ops-sparse 基点 | master e0015bb + 本设计补丁（交付分支 `sparse2dense-a5-perf`） |

### 1.1 背景

ops-sparse 仓的 `aclsparseSparseToDense` 将 CSR/CSC/COO 稀疏矩阵转换为 ROW/COL 布局稠密矩阵，接口对标 cuSPARSE 13.3
`cusparseSparseToDense_bufferSize` / `cusparseSparseToDense`。仓内既有 arch35（Ascend 950，DAV_3510）参考实现采用
"host 侧 `aclrtMemset` 全量清零 + SIMT 线程级 scatter"结构，在 950PR 实测清零仅 27.6 GB/s，
成为压倒性瓶颈（P-01 FP32 耗时 34.2 ms，GPU 标杆倍率仅 0.03×，远低于任务书 0.3× 要求）。

本设计在参考实现之上完成**性能达标与能力补齐**：以设备侧零化内核替换 host memset，补齐 complex64 支持，
规范接口主名，并交付 Python/ATen 层单遍写适配。

### 1.2 目标

| 维度 | 目标（任务书） | 实测结果 |
|---|---|---|
| 性能 | P-01/02/03 全部声明 dtype ≥0.3× GPU 标杆 | 90/90 全过，min 1.29× / median 4.44× / max 9.62× |
| 精度 | CPU Golden bit-exact（含 complex64） | 200/200 泛化 + gtest 50/50 + ATen UT 22/22 |
| 内存 | workspace ≤ L2（规则 2） | workspace 恒 0；官方 compare 290/290 通过 |
| 适配 | `Tensor.to_dense` 直调 aclsparse，无 CPU fallback | 组合派发键注册 + Profiler 单遍写证据 |

## 2. 需求分析

### 2.1 功能需求

对每个稀疏坐标 `(row,col)` 写入 `B[row,col]=A.values[p]`；未覆盖位置写 dtype 正零；输入只读；
坐标可乱序必须唯一；重复执行 bit-wise 一致。

| 维度 | 取值 |
|---|---|
| values dtype | INT8 / FP16 / BF16 / FP32 / complex64 |
| 稀疏格式 | CSR / CSC / COO（Device 索引 I32，Host 元数据 int64） |
| index base | 0 / 1 |
| 输出布局 | ROW（`ld≥cols`）/ COL（`ld≥rows`），padding 区域一并零化 |
| workspace | 仅按 `GetBufferSize` 查询值分配（本方案恒 0） |

### 2.2 性能需求

性能倍率 = GPU 标杆 Event 耗时 / NPU 同口径耗时；每 case 预热 ≥10、采样 ≥30。
标杆：P-01 `8192×28672`、P-02 `4096×1536`、P-03 `7168×2048`（每行 64 非零），5 dtype × 3 格式 × base 0/1 共 90 条。

### 2.3 关键挑战

1. **清零流量占 99.8%**（P-01 FP32 输出 939.5 MB）——清零路径带宽决定端到端性能；
2. **host `aclrtMemset` 带宽仅 27.6 GB/s**（950PR 实测），设备内核清零可达 1579 GB/s（57×差距）；
3. complex64 为基线能力缺口（校验白名单外，返回 NOT_SUPPORTED）；
4. torch_npu 默认 `_to_dense` 为 zeros+copy 两遍写，须以单遍写替换且不得误伤稠密张量路径。

## 3. 方案设计

### 3.1 总体方案

执行分两步（同一 stream，天然保序）：

```
aclsparseSparseToDense(handle, matA, matB, alg, buffer):
  校验（handle/描述符/dtype 白名单/format/base/ld/alg）→ GetBufferSize 恒 0
  ① sparse2dense_zero_kernel（AIV）: 全量零化 matB（(主维度×ld)×esize 字节，32B 对齐分片）
  ② sparse2dense_kernel（AIV+SIMT）: 非零元 scatter 写入
```

`GetBufferSize` 恒返 0（scatter 直写输出、无临时缓冲），`buffer` 传 nullptr 合法。
描述符、workspace 与 Device 数组在 stream 完成前有效；两 kernel 于 handle 绑定 stream 异步执行，调用方同步后读取。

### 3.2 接口设计

```c
aclsparseStatus_t aclsparseSparseToDenseGetBufferSize(   // 主名（任务书 2.3 命名）
    aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB, aclsparseSparseToDenseAlg_t alg, size_t *bufferSize);
aclsparseStatus_t aclsparseSparseToDense_bufferSize(...); // 兼容别名（与现存头文件一致）
aclsparseStatus_t aclsparseSparseToDense(handle, matA, matB, alg, void *buffer);
```

参数校验与异常行为对齐任务书 2.4（空 handle/描述符、非法 format/index/base/shape/dtype/ld/alg → 参数错误；
dtype 白名单 INT8/FP16/BF16/FP32/complex64）。

### 3.3 Host 侧设计

- 校验后按输出 storage 字节数（`(主维度×ld)×esize`）组织零化 tiling：`nnz` 字段复用为总字节数，
  `perBlock` 为每核字节数（32B 对齐，每核 ≥1 MB 才增开核）；单核分片超 `uint32` 时回退 `aclrtMemset`（安全兜底）。
- scatter 分核：CSR 按 row / CSC 按 col（`GetBlockIdx`×`perBlock`），COO 按 nnz grid-stride；
  核数 = min(AIV 核数, ⌈dim/128⌉)，运行时查询、无硬编码。

### 3.4 Kernel 侧设计

| Kernel | 类型 | 设计要点 |
|---|---|---|
| `sparse2dense_zero_kernel` | AIV（Tensor 流水） | UB `Duplicate` 一次 64KB 零块（每核一次），`DataCopy` 大块直通 GM（MTE3 满队列），尾块 `DataCopyPad`；零块按字节语义，规避 dtype 差异 |
| `sparse2dense_kernel` | AIV（`__simt_vf__`） | 模板化 `ValT` 分发（half/bf16/int32/int8/float/**double**）；CSR/CSC 行列连续区间 + warp 对齐线程数；COO grid-stride；`dnMat[idx]=val` 原位写，无原子操作 |

**complex64**：8B 元素经 `double` 通道 bit-copy（实/虚部各 4B 原样），零精度损失。

### 3.5 性能设计与实测

| 迭代 | 清零路径 | P-01 FP32 | 90 条 min 倍率 |
|---|---|---|---|
| 基线 | host `aclrtMemset`（27.6 GB/s） | 34165 µs | 0.03× |
| 迭代 1 | SIMT 线程级零化（1328 GB/s） | 707 µs | 1.15× |
| **迭代 2（最终）** | **AIV DataCopy 零化（64KB chunk）** | **648 µs** | **1.29×**（msprof 内核分项见下） |

msprof 分项（P-01 FP32）：零化 568.7 µs（内核态折算 ≈1651 GB/s，92.4%）+ SIMT scatter 46.8 µs（7.6%）。
注：1579 GB/s 为含 launch 开销的端到端口径实测（torch 零化内核 Event 计时），内核态折算值高于其属正常口径差；
两口径下瓶颈均收敛于 HBM 写带宽本身，无主机侧/调度侧损耗空间。

### 3.6 精度与确定性设计

全链路 bit 拷贝（无算术/转换）；未覆盖位由零化内核写 dtype 正零位模式；
坐标唯一 → 无写冲突 → 无原子需求 → 重复执行 bit-wise 一致；NaN/INF/±0 原样搬运不失真。

### 3.7 Python/ATen 适配设计（必选交付）

```
Tensor.to_dense → aten::to_dense / aten::_to_dense
  →（PrivateUse1 + SparseCsrPrivateUse1 + SparsePrivateUse1 组合派发键注册）
  → 稀疏元数据提取（crow/col/indices, values）→ I32/连续性归一 → aclsparseSparseToDense（单遍写）
```

- **组合派发键**：稀疏张量 keyset 含 layout×backend 组合键，组合键优先于纯 `PrivateUse1` 查找，
  必须按组合键注册方能拦截（关键工程结论，纯 PrivateUse1 注册不会生效）；
- 稠密张量保持 clone/dtype 转换语义；BSR 等不支持 layout 明确报错；非法 dtype（如 float64）明确报错；
  非连续输入内部连续化；空矩阵（m/n=0）返回空稠密；零 nnz 返回全正零；
- **红线**：注册直接调用 `aclsparseSparseToDense`，Profiler 证明无 `aclnnInplaceZero`（两遍写痕迹）与 CPU fallback。

## 4. 测试方案与结果

| 层级 | 覆盖 | 结果 |
|---|---|---|
| C++ gtest | L0/L1 全组合 + L2 负例（空指针/非法参数/溢出/nnz=0） | 50/50 ✅ |
| 泛化精度 | 200 条（3 格式 × 5 dtype × base × ROW/COL，只读/确定性抽查） | 200/200 ✅ |
| ATen/Python UT | 22 项（映射/参数/stride/dtype/空矩阵/异常/无两遍写） | 22/22 ✅ |
| 性能 | 90 条 P 场景（预热 10/采样 30）+ 200 条泛化 | 90/90 ≥0.3×，min 1.29× ✅ |
| 内存 | 官方 collect/compare 链路 | 290/290 ✅（12 条 peak_5% + 278 条 workspace≤L2） |
| 端到端耗时 | kernel / torch.ops hook / `Tensor.to_dense` 三口径（ATen 适配开销仅 9-14 µs） | `aten_e2e_report.json` ✅ |
| A2/A3 交叉回归 | 同源任务包（200 条精度 + 290 条性能/内存），A5 侧按 case id 可对表；arch22 侧需 A2/A3 硬件 | 结论一致 ✅ |
| 证据 | `pic_my/`（精度/性能截图 + msprof 分项）| 齐备 ✅ |

## 5. 风险与演进

1. **融合单遍写**（清零+scatter 单 kernel）可再省一次 launch（~10-20 µs），利好 P-02/P-03 小场景；预留 SuperKernel/GE 图融合（两 kernel 无核间同步、零 workspace，天然满足约束）；
2. 零化带宽 1651 GB/s 已近数据通路上限，P-01 FP32 倍率 1.29-1.68× 受写带宽物理约束；
3. 多任务争用下 AIVEC 任务存在排队延迟（独占态消失，两轮结论一致），runner 侧已内置有界同步自愈容错；
4. torch_npu 稀疏构造路径不支持 complex64（上游限制），算子本体支持并经 CPU 构造搬运路径覆盖测试。

## 6. 参考资料

1. cuSPARSE SparseToDense（13.3）：接口语义基准。
2. ops-sparse：https://gitcode.com/cann/ops-sparse （交付分支 `sparse2dense-a5-perf`）。
3. 公开头文件：`include/cann_ops_sparse.h`。
4. 精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md 。
5. PyTorch `Tensor.to_dense` / ATen `native_functions.yaml`。
6. 自测证据：`SELF_TEST_REPORT.md`、`pic_my/`、`docs/perf_compare_table.md`、`docs/accuracy_compare_table.md`。

## 7. 设计变更记录

| 版本 | 变更内容 |
|---|---|
| v1.0（2026-09-03） | 初版发布。相对 ops-sparse master e0015bb 的实现差异：新增 `sparse2dense_zero_kernel`（AIV DataCopy 零化）；complex64 能力补齐；`aclsparseSparseToDenseGetBufferSize` 主名；Python/ATen 组合派发键适配。全部变更以 `npu_wrapper/patches/ops_sparse_arch35_v1.patch` 交付。 |

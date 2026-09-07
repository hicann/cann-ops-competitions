# aclsparseSparseToDense（A2/A3，arch22）算子设计文档

# 需求背景（required）

## 需求来源

CANN 社区任务（2026-09）：aclsparseSparseToDense A2/A3 实现。
对标 cuSPARSE Generic API `cusparseSparseToDense` / `cusparseSparseToDense_bufferSize`
（cuSPARSE 13.3 Update 1 语义），交付至 ops-sparse 社区仓。

## 背景介绍

### 算子定位

将 CSR/CSC/COO 稀疏矩阵转换为 ROW/COL 布局稠密矩阵：

B[i,j] = A.values[p] 若 (i,j) = 坐标 p；否则 +0.0（对应 dtype 正零）

- values：INT8 / FP16 / BF16 / FP32 / complex64（位搬运，零数值转换 → bit-wise 精度）
- 索引：I32（offsets/indices 均为），index base 0/1
- 纯搬运算子：无算术、无类型转换、确定性（坐标唯一 → 无写冲突）

### 现状（本任务前）

- ops-sparse 已有 arch35（Ascend950，SIMT `__simt_vf__` 模型）实现，不支持 complex64
- arch22（DAV-2201：Atlas A2 训练系列 910B / A3）无实现，且 SIMT 模型不可用
- NPU 上 PyTorch `Tensor.to_dense`（aten::_to_dense）走 CPU fallback

# 需求分析（required）

## 需求描述

1. arch22 上实现 `aclsparseSparseToDenseGetBufferSize` 等价双 API（以仓内
   `include/cann_ops_sparse.h` 声明为准：`aclsparseSparseToDense_bufferSize` / `aclsparseSparseToDense`）
2. 支持 3 格式 × 5 dtype × 2 base × 2 布局；workspace=0
3. Python/ATen 适配：`Tensor.to_dense`→`aten::_to_dense` NPU 注册，PyTorch 2.7+ / torch_npu 26.0.0+（pip 2.10.0.post2 同代），禁 CPU fallback
4. 性能：P-01/02/03 每条 case ≥ 0.25× GPU 标杆（PyTorch GPU 设备 Event median）
5. 内存：workspace ≤ L2（实测恒 0）

## 需求拆解

1. Host 侧：描述符校验/tiling/launch（CSR 按行、CSC 按列、COO 按 nnz 分核）
2. Kernel 侧：clear（清零）+ scatter（搬运）双 kernel，同 stream 保序
3. 适配层：torch 扩展（hook + ATen 注册）
4. 测试：GTest 56 例 + ATK 等价 200 例 + 290 性能/内存 + torch e2e

# 详细设计（required）

## 算子分析

### 数学公式

B[r,c] = A.values[p]，当 (r,c) = coord(p)（按 base 换算）；否则 +0

### 支持数据类型

INT8 / FP16 / BF16 / FP32 / complex64（I32 索引，base 0/1）

### 支持形状

m,n ≤ INT32_MAX；nnz ≤ INT32_MAX（COO）；dense 元素总数 < 2^32

## 算子实现

### 实现方案

#### host 侧设计

- 校验：handle/matA/matB 非空、格式 ∈ {CSR,CSC,COO}、I32 索引、dtype 白名单、
  shape/ld/dtype 匹配（ROW: ld≥cols；COL: ld≥rows）、stream 已设置、nnz>0 时
  ptrs/idxs/values 非空
- tiling：AIV 核数（优先 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)`，
  失败回退 platform manager）；perBlock = ceil(dim/cores)（dim=行/列/nnz）
- launch 顺序（同 stream）：clear kernel → scatter kernel；m/n/nnz=0 时仅 clear

#### kernel 侧设计（arch22，经典 Ascend C 向量模型）

**clear kernel**：各核分片将 dense 输出（含 ld padding）清零。
UB 预置 0（uint16 视图 Duplicate，uint8 视图搬出，8KB/tile）。
- 不用 aclrtMemset(Async)：其与 kernel launch 在 CANN 9.1 torch 宿主进程存在
  跨流乱序（实测清掉 scatter 结果）

**scatter kernel**（按格式三分支，dtype 经模板特化为位拷贝类型
u8/u16/u32/u64）：

- CSR/CSC：offsets 光标推进 + 逐行处理（v15g 定版）：
  1. **快速连续检查**（2 次标量读）：c0+len-1==cLast ⇒ 整行 minor 连续；
  2. **布局门控**：仅 CSR+行主序 / CSC+列主序时 minor 连续 ⇔ 内存连续
     （mergeable）；CSC+行主序等组合走逐元素；
  3. **整段直发**：mergeable 段走「对齐 val2 直发（1×MTE3，源 32B 对齐时，
     P-case 行边界恰 8 元素对齐全命中）」或「EmitRun ring 直发
     （GM→UB ring + UB→GM，2×MTE3，任意源对齐，零逐元素标量）」；
  4. **单读断点扫描**：非连续行用 prev 寄存器比较 col[j+1]≠col[j]+1，
     每元素仅 1 次标量读，断点间每段整段直发；
  5. 逐元素兜底：非 mergeable 布局 UB 单槽 SetValue + DataCopyPad 单元素。
- COO：nnz 均分多核 + **对齐 prologue**（核起始位置对齐 32B，使 tile 内行
  起点天然对齐，直发命中率最大化）+ 行探测（步进 64 快速 + 线性扩展）+
  与 CSR 同款的快速连续检查/断点扫描/直发。
- 性能动机：每元素标量 UB 访问 ~100-255ns 是唯一瓶颈（msprof 实测），
  逐元素路径每元素 3-4 次标量访问+1 发射；直发路径每行仅 2-4 次标量读。
- **写侧硬约束（实测）**：GM 随机写必须走 MTE3 DataCopyPad；
  GM SetValue 多核丢写；MTE3 的 UB 源地址须 32B 对齐（ring 槽 32B 对齐分配）

### tiling 策略

| 格式 | 切分维度 | perBlock |
|---|---|---|
| CSR | 行 m | ceil(m/cores) |
| CSC | 列 n | ceil(n/cores) |
| COO | nnz | ceil(nnz/cores) |

tile 大小 kSparse2DenseTileNnz=2048（元素）。

### workspace 设计

恒 0：scatter 直写输出，无中间缓冲（内存规则 2 直接满足：0 ≤ L2 126MB）。

## PyTorch / ATen 适配设计

torch 扩展（python/torch_adapter）：
- `TORCH_LIBRARY(ops_sparse_test)`：`sparse_to_dense_npu(fmt, values, primary,
  secondary, rows, cols, base, layout)`——测试包 hook 契约
- `TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1/SparsePrivateUse1)`：
  `_to_dense(self, dtype?, masked_grad?)`——CSR/CSC/COO 三布局分发
- 执行模型：静态单例 handle + torch 当前流（GetAclStream）+ 入口流同步 +
  出口设备同步（异步边界收在函数内；CANN 9.1 torch 宿主异步 launch 调度缺陷
  规避：宿主侧运行须 ASCEND_LAUNCH_BLOCKING=1，官方 runner 已内置）

## 关键实现约束（实测沉淀）

| # | 约束 | 规避 |
|---|---|---|
| 1 | GM 标量写多核丢失 | 写全走 MTE3 |
| 2 | aclrtMemset 与 launch 跨流乱序 | 设备侧 clear kernel |
| 3 | MTE3 UB 源须 32B 对齐 | ring 对齐槽/标量兜底 |
| 4 | DataCopy UB→UB 32B 对齐块 | roundup(len, kE) + 越界改标量 |
| 5 | Cast/Reduce 非对齐 count 处理不全 | 预零 + count 64 对齐 |
| 6 | 小 tile(<64) 向量验证触发 VC 异常 | cnt≥64 门控行遍历 |
| 7 | TQue 深预取>2 / VECOUT 单元素死锁 | 双缓冲上限 2 |
| 8 | aicore 内 lambda 编译为 __host__ | 成员模板函数 |
| 9 | 2D 跨步 DataCopyPad 写侧病态 | 弃用，逐段小 MTE3 |
| 10 | 无 GM→GM DataCopy 重载 | 整段直发经 32B 对齐 ring 槽中转 |
| 11 | GM→UB sub-32B blockLen 不掩码（整块写入） | UB 目标块 ≥32B 对齐 |
| 12 | UB→GM per-element 源读取触发 VC 异常 | 源块 32B 对齐 |
| 13 | arch35 SIMT 无法编译到 A2（`__simt_vf__` 架构不支持） | 全部经典向量模型 |
| 14 | vtranspose 语义与文档不符（实测低16位交织，非转置） | 弃用，无 UB 转置原语 |
| 15 | 无 scatter/gather 向量指令（dav_c220 Scatter=stub） | CSC+ROW 逐元素（见性能章） |

# 测试设计（required）

| 层 | 规模 | 内容 | 结果 |
|---|---|---|---|
| GTest（CSV 参数化） | 56 | L0/L1 正确性（格式×dtype×base×布局×ld/宽高/空/满）+ L2 负向 15 | 56/56 |
| ATK 等价 | 200 | 任务包泛化用例（attr 解码同 function_sparse_ops.py）NPU vs CPU golden torch.equal | 200/200 |
| 性能 | 290 | 官方 benchmark_sparse_ops_npu（Event median，warmup10×30） | 290 数据齐 |
| 内存 | 290 | collect_sparse_ops_npu_memory | workspace 全 0 |
| torch e2e | 手检 | sparse_csr/csc/coo_tensor.to_dense() vs CPU bit-wise | 全等 |

环境：Atlas A2 910B4（hidevlab）、CANN 9.1.0、torch 2.10.0+cpu/torch_npu 2.10.0.post2、
Ubuntu 22.04 aarch64。

# 性能实测（910B4，v15g 定版，median_us）

| 场景 | NPU median | GPU 标杆 | 预算 4× | 状态 |
|---|---|---|---|---|
| P-01 csr f32 | 2027us | 1080.8us | ≤4323us | ✅ 1.88× |
| P-02 csr f32 | 579us | 444.6us | ≤1778us | ✅ 1.30× |
| P-03 csr f32 | 866us | 488us 级 | ≤1952us | ✅ |
| P-02 coo f32 | 551us | 154.1us | ≤616us | ✅ 3.58× |
| P-02 coo bf16/i8 | 544/534us | 154/179us | ≤616/716us | ✅ |
| P-03 coo f16/bf16/f32 | 703/706/718us | 191/206/190us | ≤759-826us | ✅ 3.4-3.8× |
| P-01 coo（全线） | ~1900-2100us | 778-868us | ≤3112-3472us | ✅ |

达标构成（90 条 P-case）：CSR 30/30 ✅ + COO 30/30 ✅；CSC 30 条受
arch22 平台原语限制见下节。

## CSC+ROW 性能说明（平台原语限制，如实记录）

CSC+行主序的输出地址 = row×ld+col（行索引随机散布），是真正的不规则散射。
arch22 硬件无任何 scatter/gather 原语（实测：dav_c220 Scatter API=stub、
无 GM→GM DataCopy 重载、GM→UB sub-32B 块不掩码、UB→GM per-element 源读取
触发 VC 异常、vtranspose 语义与文档不符、arch35 SIMT 无法编译到 A2、
ND2NZ 为软件模拟）。所有可行的转置/块聚合管线均被逐一实验否定后回退
逐元素路径（每元素 ~1.2us/核），CSC P-01/P-02/P-03 的 0.25× 达标在
910B3/B4 上以现有公开 API 无法达成；官方 arch35 SIMT scatter
实现用于 A5/Ascend950（DAV-3510）；A3（ascend910_93）与 A2 同为
arch22 向量核（仓库 CMake 映射）。详见 PROGRESS-LOG.md 实验记录。

# 交付物清单

1. `sparse/sparse2dense/arch22/`：host/kernel/tiling/头（5 文件）
2. `test/sparse2dense/`：GTest 四件套 + arch22 csv（共享 golden 扩展 complex64）
3. `python/torch_adapter/`：ops_sparse_torch.cpp + 构建脚本
4. 设计文档（本文）→ cann-ops-competitions PR
5. 自测报告 + 290 性能/内存数据 + msprof 证据

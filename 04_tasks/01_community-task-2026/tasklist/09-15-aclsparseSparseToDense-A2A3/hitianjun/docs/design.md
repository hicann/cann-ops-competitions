# 需求背景（required）

## 需求来源

9月社区任务：`aclsparseSparseToDense` 算子开发（A2/A3）。任务要求在 ops-sparse 算子工程中以 C++ / Ascend C 实现 Atlas A2/A3（DAV-2201，arch22）的稀疏转稠密算子，代码提交至 https://gitcode.com/cann/ops-sparse （PR #211）。

## 背景介绍

### aclsparseSparseToDense 算子功能

将 CSR、CSC 或 COO 格式的稀疏矩阵转换为 ROW/COL 布局的稠密矩阵，接口语义对标 cuSPARSE Generic API 的 `cusparseSparseToDense_bufferSize` / `cusparseSparseToDense`（两阶段：查 workspace → 执行转换）。

计算表达式：

$$B_{ij} = \begin{cases} A_{ij} & (i,j) \in \mathrm{nnz}(A) \\ +0 & \text{否则} \end{cases}$$

### 支持能力

| 参数 | 含义 | 支持值 |
| --- | --- | --- |
| matA | 稀疏矩阵描述符 | CSR / CSC / COO，索引 I32，index base 0/1 |
| matB | 稠密矩阵描述符 | ROW / COL 布局，任意 ld（含 padding） |
| values | 值类型 | INT8 / FP16 / BF16 / FP32 / complex64 |
| alg | 算法 | `ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT` |
| workspace | 临时空间 | 恒为 0（scatter 直写，无需缓冲） |

纯数据搬运、无算术：输出与输入值 bit-wise 一致（含正负零、INF、NAN），未覆盖元素为正零 —— 精度上与 CPU Golden **逐元素 exact match**。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 DAV-2201（Vector Core，非 SIMT）上实现上述算子，通过功能（121 例 UT）、精度（bit-wise）与性能（GPU/NPU ≥ 0.25，90 case 全达标）验收。

## 需求拆解

1. 五种 dtype × 三种稀疏格式 × 两种输出布局 × base 0/1 的全组合正确性；
2. 零填充与散写的流水协同（非直接型路径 host 预清零 + kernel 散写）；
3. 散写随机 GM 写的吞吐优化（本算子性能瓶颈所在）；
4. Python/ATen 适配（torch_npu，PyTorch 2.7+ / torch_npu 26.0.0+）。

# 详细设计（required)

## 算子分析

### 数学公式

`B[idx(i,j)] = A.val[p]`（对每个 nnz），其余 B 元素为 +0。

### 关键难点：0.2%~2% 密度的随机 GM 写

每行仅 64 个唯一列，散写地址无合并结构（连续段概率 0.8%/对）——硬件无 scatter 指令（DAV-2201），只能以 `copy_ubuf_to_gm_align_b8` 逐笔 1B~8B 散写。性能设计核心 = 压低每 nnz 的软件成本、拉高 MTE3 引擎占用。

## 算子实现

### 实现方案

#### host 侧设计

**路径分流**（按格式 × 布局）：

| 路径 | 条件 | 策略 |
| --- | --- | --- |
| 直接型 fill | CSR+ROW / CSC+COL | kernel 段式写出全部输出（含 ld padding），host 不清零 |
| 非直接型 | COO / CSC+ROW（超级 tile）/ CSR+COL（交叉） | host AIC 清零（side stream）+ kernel 散写 nnz |

**分核策略**：CSR/CSC 按行/列均分（满核优先，余量分给前几核）；COO 按 nnz 均分；CSC+ROW 按 16 列组均分（均衡分割修核空转）。

**AIC 并发清零**：AIC 核 Fill-on-L1 + DataCopy L1→GM 实现 703GB/s memset（1.95× AIV zero kernel），经 side stream + 事件链保序；尾块 ≤31B 由 AIV 1-block 兜底。

**tiling 字段**：m/n/indexBase/valueType/isColMajor/ld/perBlock/format/nnz + 消融开关（l0Scatter/swPipe/rawAccess/tbufStage/dbgNoScatter 等，供 A/B 验收）。

#### kernel 侧设计

模板类 `Sparse2DenseKernelT<ValT, kElemsPerUnit>`，五 dtype 单模板覆盖（complex64 以 float 双单元视图 bit-wise 搬运）。每 tile 三步流水：

1. **MTE2**：TBuf 手动 staging（row/col/val 三笔 + 单 barrier，消 TQue 事件机械）；
2. **V**：Broadcast 槽位铺填（{n,R} 广播，c64 走 8B 单笔标量）+ dst 向量链（base=0 快路径 3 条向量 op）；
3. **MTE3**：裸串行散写循环（读 dst + `copy_ubuf_to_gm_align_b8` 直发，**无 8 路展开**——实测展开形态损失 2×）。

**流水重排（ProcessCooFlow）**：MTE3 排空从 tile 末移至下 tile Broadcast 前——引擎只读 slots（dst 已由标量 issue 消费），下 tile 的 MTE2/向量链与在飞引擎重叠，零额外 buffer。

**fill 族**：整行镜像（≤112KB）/多行镜像批/部件镜像乒乓三级，Duplicate 清零 + 标量扫描铺值 + 段式 bulk 写出；offsets 预取消逐行 GM 标量读；4 路预取展开扫描。

**UB 预算**：c64 最紧 ~132KB ≤ 192KB；int8 Broadcast 单次写 ≤32KB（病态域钳制，kScatTile=1024）。

### 性能优化方案（战役结果，中位 4.42×）

| 优化 | 收益 | 依据 |
| --- | --- | --- |
| 裸 `__ubuf__` 指针访存替代 GetValue/SetValue | 读 ILP8 5.4× / 写 5.1× | API 包装 85-91ns vs raw 16-31ns（探针 U1） |
| GetPhyAddr 外提（slotBase 参数化） | +1.31× | 每 nnz 调用曾吃掉全部收益 |
| 散写裸串行形态 + dst 8B 合读 | +1.94× | 形态学定案：8 路展开+helper 损失 2× |
| TILE 2048 + per-dtype 钳制 | staging 摊半 | MTE2 每调用固定开销主导 |
| MTE3 排空重排 | 引擎尾部重叠 | 消每 tile ~11μs 排空串行 |
| c64 铺槽 8B 单笔 + 16 路展开 | c64 过线 | 两次 4B 写 → 一次 uint64 |

**硬件/编译器陷阱定案**（探针 U1-U4 + k3 二分定案，随仓 rulecheck B7 固化）：
1. `LocalTensor::GetPhyAddr()` 返回**字节粒度**指针——cast 包住和式即地址静默错位；
2. int8 Broadcast 病态域 = UB 分配拓扑函数，单次写 >32KB 即病态（13ms/tile 级），探针不可迁移；
3. 散写 8 路展开形态损失 2×（裸串行最优）；
4. `SetFlag/WaitFlag` 必须配平（遗留未 Wait 的 Set 挂起 kernel 退出）。

## 测试设计

- **C++ UT**：121 例（格式×布局×dtype×base×边界：零 nnz、空行/列、ld padding、正负零/INF/NAN、乱序坐标、重复执行确定性、非法参数负例）；
- **端到端**：CPU Golden 逐元素 exact match（values bit-wise）；
- **性能**：任务包 P-01/P-02/P-03 × 3 格式 × 5 dtype × 双 base = 90 case，GPU/NPU ≥ 0.25 全达标（90/90）；同 session A-B-A 复跑漂移 <0.5%；
- **探针族**：scalar_cost / mte3_cost / tq_probe / aic_memset / raw_probe（U1-U4）等 17 个诊断 main，支撑消融定案与后续回归。

## 验收数据（自测环境：Atlas A3 910C / CANN 9.2.0 / bisheng）

- UT：**121/121**（raw 开/关双形态）
- 性能：**90/90 达标**，vs 优化前基线中位 **4.42×**
- 代码：ops-sparse PR #211（分支 feat/sparse2dense-arch22，fork 已邀请 Ascend-CANN）

# 维护与限制说明

- complex64 实/虚部 bit-wise 保持；输出未覆盖元素恒为 +0；
- `bufferSize` 恒返回 0（scatter 直写无 workspace）；
- nnz 超过 INT32_MAX 的 COO 分块不支持（host 显式报 `NOT_SUPPORTED`）；
- int8 路径 tile 钳制 1024（Broadcast 病态域），如 CANN 后续解除可提升 staging 摊薄上限。

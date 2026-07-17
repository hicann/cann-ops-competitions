# SpGEMM 算子设计文档

| 版本 | 日期 | 修改人 | 修改内容 |
|------|------|--------|----------|
| v1.0 | 2026.7.11 | CCuis | 初稿 |

---

# 一、需求背景（required）

## 1.1 需求来源

7 月社区任务——SpGEMM 算子开发。参考 NVIDIA cuSPARSE `cusparseSpGEMM`
（Generic API，§6.6.14），在昇腾 NPU（Ascend 950PR）上实现稀疏矩阵 × 稀疏矩阵
乘法，交付 Ascend C Kernel + aclsparse C++ 接口，并在验收后合入开源仓库
`ops-sparse` 的 `src/spgemm/`。

## 1.2 背景介绍

SpGEMM（Sparse General Matrix-Matrix Multiplication）是稀疏线性代数中的基础算子，
广泛应用于图计算、代数多重网格和科学计算。其运算定义如下：

$$
C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
$$

- A(m×k)、B(k×n)、C(m×n) 均为稀疏 CSR，α、β 为标量；
- 输出 C 的稀疏结构由乘法确定，非零值由乘积累加得到。

## 1.3 现状分析

`ops-sparse` 仓库中已有 SpMV、SpMM、Snnz 等算子，**无 `src/spgemm/`**；SpGEMM 为全新
算子，接口与 Kernel 均需从零交付。SpGEMM 与 SpMM 的核心差异在于：SpMM 输出**稠密矩阵**
（形状已知，可直接分配），而 SpGEMM 输出**稀疏矩阵**，其非零结构（nnz、每行分布）在
**运算前未知**，必须先通过符号阶段确定结构，再通过数值阶段填值——这是 SpGEMM 的核心难点。

---

# 二、需求分析（required）

## 2.1 需求描述

使用 Ascend C 实现 SpGEMM，并通过 `aclsparseSpGEMM*` 多阶段 C++ 接口对外提供能力，支持
fp32/fp16/bf16 同精度、CSR、int32 索引以及仅 NON_TRANSPOSE；性能对标 A100 cuSPARSE
≥ 1.0×，精度满足生态开源精度标准。

## 2.2 需求拆解

1. **接口层**：新增 `aclsparseSpGEMM*` 多阶段接口，命名和流程对齐 cuSPARSE 与仓内 SpMM。
2. **数据类型**：同精度 fp32（必测）/fp16/bf16，A/B/C/computeType 四者一致。
3. **格式索引**：CSR、int32（`ACL_SPARSE_INDEX_32I`）、zero-based。
4. **算法约束**：opA/opB 仅 NON_TRANSPOSE；输出 C 列索引 sorted；输入 sorted；
   fp32 bit-wise 确定性。
5. **性能**：3 组固定用例 + 200 组泛化用例，均 ≥ 1.0× A100。
6. **精度**：生态开源精度标准 + ATK 双标杆（最大相对误差比例 ≤ 2 / 平均 ≤ 1.2 / RMSE ≤ 1.2）。

## 2.3 输入输出规格

| 名称 | 角色 | 格式 | dtype | 索引 | 约束 |
|------|------|------|-------|------|------|
| matA | 输入稀疏矩阵 | CSR | fp32/fp16/bf16 | int32 | m×k，列索引 sorted，zero-based |
| matB | 输入稀疏矩阵 | CSR | 同 A | int32 | k×n，列索引 sorted，zero-based |
| matC | 输出稀疏矩阵 | CSR | 同 A | int32 | m×n，结构由计算确定，列索引 sorted |
| alpha/beta | 标量 | host 指针 | 同 computeType | - | - |

---

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

$$
C = \alpha \cdot A \cdot B + \beta \cdot C_{in}, \quad (A\cdot B)[i][j]=\textstyle\sum_k A[i][k]\times B[k][j]
$$

**β 语义（对齐 cuSPARSE）**：cuSPARSE 约定 C 与结果 C' 具有**相同稀疏结构**，
SpGEMM 不因 β·C 自动产生新的非零位置。本任务初版按照“matC 初始为空、输出结构 = α·A·B
结构”实现；当 β≠0 且 matC 带内容时，按照 cuSPARSE 语义叠加 β·C_in（结构须一致）。

### 3.1.2 支持数据类型

| ID | A/B/C/computeType | cuSPARSE 枚举 | acl 枚举 |
|----|-------------------|---------------|----------|
| U1 | fp32 | CUDA_R_32F | `ACL_FLOAT`(0) |
| U2 | fp16 | CUDA_R_16F | `ACL_FLOAT16`(1) |
| U3 | bf16 | CUDA_R_16BF | `ACL_BF16`(27) |

fp16/bf16 统一在 fp32 中累加，并在末尾转回目标类型，以保证精度。SpGEMM 为仓内首个引入
bf16 的算子，因此需在校验与 kernel 分派中新增 `ACL_BF16` 分支。

### 3.1.3 支持形状与约束

支持任意 m/k/n（受 int32 索引上界约束）；CSR / int32 / zero-based / NON_TRANSPOSE；
输出 C 列索引 sorted；fp32 bit-wise 确定性。

## 3.2 算法方案：两阶段 SpGEMM

采用经典的**符号 + 数值**两趟法。

### 3.2.1 符号阶段（确定 C 结构）

计算 C 的 `rowPtr` 与 `nnz(C)`，并为 colIdx/values 分配内存。具体规则如下：C 的第 i 行遍历
A 第 i 行的非零列 k，再遍历 B 第 k 行的非零列 j，并将 j 计入 C 第 i 行的结构非零集合（去重）。

for i in [0, m):
    colset = ∅
    for (k, ) in A.row(i):
        for (j, ) in B.row(k): colset.insert(j)   # 去重
    rowNnzi = |colset|
rowPtr = prefix_sum(rowNnz);  nnz(C) = rowPtrm

结构非零遵循“宁多不漏”的原则——即使数值抵消为 0，也仍计入结构非零（与 cuSPARSE 一致）。

### 3.2.2 数值阶段（填值）

对每个 (i,j) 位置计算数值，并写入 colIdx/values；每行 colIdx 按升序排列。

for i in [0, m):
    accum = {}                          # 列 j -> fp32 累加值
    for (k, a) in A.row(i):
        for (j, b) in B.row(k): accumj += a * b   # 固定顺序 -> bit-wise 确定
    按列号 j 升序输出: C.colIdx=j, C.values=alphaaccumj (+betaC_ini)

累加容器初版使用 **fp32 dense accumulator**（长度为 n 的临时数组，简单且快速）；当 n 很大时，
再引入 hash accumulator 以节省内存。通过固定“A 行序 × B 行内列序”的遍历顺序并有序写回，
保证 fp32 bit-wise 确定（同 SpMM kernel 的 fp32 累加思路）。

## 3.3 昇腾侧实现方案（仿 SpMM 架构）

### 3.3.1 Host 侧（`spgemm_host.cpp`）

多阶段接口职责如下：

| 接口 | 职责 |
|------|------|
| CreateDescr/DestroyDescr | 管理内部描述符（numProds、buffer 记账、中间状态） |
| WorkEstimation | 估算 numProds 与符号工作量；buffer1 查询/执行 |
| EstimateMemory | 仅 ALG2/ALG3 需要；ALG_DEFAULT 直通返回 0 |
| Compute | 对齐 cuSPARSE ALG1：**调两次**（先探测内存上界、再执行符号+数值）；产出 C 到 workspace |
| GetNumProducts | 返回乘积对总数 |
| Copy | 将结果写入 matC 的 CSR 数组（结构须与结果一致） |

- **入参校验**（仿 `ValidateSpmmInputs`）：空指针、format=CSR、索引 int32、
  base=zero、dtype 四者一致、维度匹配（A.cols==B.rows 等）、opA/opB=NON_TRANSPOSE、alg。
- **blockDim**：`PlatformAscendCManager::GetCoreNumAiv()`（950PR=64）。
- **符号阶段落点**：仿 SpMM 预处理，初期先在 **host 侧**实现（将 rowPtr 拷回 host 计算，
  再写回 workspace），便于与 CPU golden 对拍；若性能不足，再下沉至 device kernel。
- **分核**：仿 SpMM 的**贪心装箱**（按负载降序放入负载最小的核），但负载权重采用
  “每行乘积对数 Σ_k nnz(B.row(k))”，而非行长，因为 SpGEMM 各行工作量差异更大。

### 3.3.2 Kernel 侧（`spgemm_kernel.cpp`）

- 仅 arch35（`__NPU_ARCH__==3510`）；使用 C++ 模板 + `if constexpr` 区分 fp32/fp16/bf16，
  fp16/bf16 在 fp32 中累加。
- 符号 kernel（统计每行结构非零数→rowPtr）+ 数值 kernel（乘加填充 colIdx/values，
  sorted、fp32 确定）；host 按 dtype 分派启动。

### 3.3.3 Workspace 与 buffer

workspace 布局（全部 64B 对齐）：`[64B header][SpgemmTilingData][符号阶段
rowPtr/计数/分核 bin][数值阶段 accumulator 缓冲]`。

外部 buffer（对齐 cuSPARSE ALG 内存策略）：

| buffer | 使用接口 | DEFAULT/ALG1 | ALG2/ALG3 |
|--------|----------|:---:|:---:|
| buffer1 | WorkEstimation + Compute | 需要 | 需要 |
| buffer2 | Compute + Copy | 需要 | 需要 |
| buffer3 | EstimateMemory | 不需要 | 需要 |

初版实现 **ALG_DEFAULT(=ALG1)** 全链路（buffer1+buffer2，Compute 调两次），
ALG2/ALG3 预留枚举。三种算法均具备 bit-wise 确定性。

### 3.3.4 目录与文件

src/spgemm/arch35/{spgemm.h, spgemm_host.cpp, spgemm_csr_mat.{h,cpp}, spgemm_kernel.cpp}
test/spgemm/{CMakeLists.txt, arch35/spgemm_test.cpp}
include/cann_ops_sparse.h  （追加 aclsparseSpGEMM* 声明 + 枚举 + 描述符类型）

`src/CMakeLists.txt` 按 arch 自动收集，无需修改。

## 3.4 支持硬件

| 芯片版本 | 支持 |
|----------|------|
| Ascend 950PR（dav-3510 / arch35） | √ |

开发环境：CANN 9.0.0，编译器 bisheng；构建命令如下：
`bash build.sh --ops=spgemm --soc=ascend950 [--run]`。

## 3.5 算子约束限制

1. opA/opB 仅 NON_TRANSPOSE，否则返回 NOT_SUPPORTED；
2. 仅 CSR、int32 索引、zero-based；
3. 同精度：A/B/C/computeType 四者一致；
4. 输出 C 列索引 sorted；输入 A/B 列索引须 sorted；
5. fp32 结果 bit-wise 确定性；
6. int32 索引，规模受 INT32_MAX 上界约束。

---

# 四、可维可测分析

## 4.1 精度标准 / 性能标准

| 验收标准 | 描述 |
|----------|------|
| 精度标准 | 满足《生态算子开源精度标准》；ATK 双标杆 `cv_fused_double_benchmark`：最大相对误差比例 ≤ 2、平均 ≤ 1.2、RMSE ≤ 1.2 |
| 性能标准 | 对标 A100 cuSPARSE，NPU ≥ 1.0×（3 组固定 + 200 组泛化） |

精度对拍采用相对误差指标（平均/最大相对误差 + 阈值），fp32/fp16/bf16 分别设置阈值；
另增加“两次运行 bit-wise 相等”测试，以验证 fp32 确定性。

## 4.2 测试方案

**功能用例**（CPU golden 对拍）：

| 编号 | 场景 |
|------|------|
| TC-01 | 小矩阵 SpGEMM（手算对拍） |
| TC-02 | 空矩阵 / 零 nnz 边界 |
| TC-03 | alpha/beta ≠ 1 |
| TC-04 | 高稀疏度大图 |
| TC-05 | aclsparse 多阶段调用完整性（Create→WorkEstimation→Compute→Copy→Destroy） |

CPU golden 实现 CSR×CSR→CSR（结果 sorted），并与 NPU 结果的 rowPtr/colIdx/values 逐一比对。

**性能用例**（3 组固定必测）：

| 编号 | 规模 m×k×n | nnz(A) | nnz(B) | dtype | A100 参考 μs | 要求 |
|------|-----------|--------|--------|-------|-------------:|------|
| 01 | 128×128×128 | 819 | 819 | fp32 | 3654 | ≥1.0× |
| 02 | 128×128×128 | 0 | 0 | fp32 | 13 | ≥1.0× |
| 03 | 128×64×128 | 163 | 163 | fp32 | 2404 | ≥1.0× |

**泛化用例**（200 组）：规模 128~1e4+、nnz(C) 1e2~1e6+、稀疏率 0.01%~10%、
dtype 覆盖 fp32/fp16/bf16、α/β≠1 ≥10 组、边界样本若干；使用 Python(scipy.sparse)
批量生成并记录随机种子，golden 使用 `scipy A@B`，性能对标 A100 avg，须附完整用例列表。

## 4.3 兼容性分析

新增算子，不涉及存量接口变更；`aclsparseSpGEMM*` 为新增符号，不影响已有 SpMM/SpMV。

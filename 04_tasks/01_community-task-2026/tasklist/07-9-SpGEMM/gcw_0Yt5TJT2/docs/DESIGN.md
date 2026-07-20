# 需求背景（required）

## 需求来源

7月社区任务：SpGEMM 算子开发，对标 cuSPARSE `cusparseSpGEMM()` (Generic API §6.6.14)，在昇腾 NPU 上实现稀疏矩阵 × 稀疏矩阵乘法。

## 背景介绍

### SpGEMM 算子实现

基于 Ascend C 编程语言 + SIMT 编程模型，在 Ascend 950PR（dav-3510）NPU 上实现 SpGEMM Kernel 及 aclsparse C++ 接口，合入 ops-sparse 仓。

SpGEMM 公式：`C = α·op(A)·op(B) + β·C`

- A ∈ R(m×k)、B ∈ R(k×n)、C ∈ R(m×n) **均为稀疏 CSR 格式**
- α、β 为标量
- 输出 C 的稀疏结构由乘法确定，非零元值由乘积累加得到

### SpGEMM 对标分析

对标 cuSPARSE `cusparseSpGEMM`（Generic API，§6.6.14），参考其 4 阶段调用流程、数据类型支持、算法枚举及约束：

| 项目 | cuSPARSE | 本实现 |
|------|----------|--------|
| 调用流程 | WorkEstimation → EstimateMemory → Compute → Copy | 同 cuSPARSE |
| 稀疏格式 | CSR | CSR |
| 索引类型 | int32 / int64 | int32（V1） |
| opA/opB | 仅 NON_TRANSPOSE | 同 cuSPARSE |
| 计算精度 | fp32 / fp16 / bf16 / fp64 等 | fp32 / fp16 / bf16（V1） |
| 算法 | ALG1/ALG2/ALG3 | 保留枚举，V1 统一走 ALG1 路径 |
| 确定性 | bit-wise deterministic（fp32） | 同目标 |

### Opbase 现有实现分析

ops-sparse 仓当前未实现 SpGEMM 算子。已有算子 SpMM（`src/spmm/`）涉及稀疏 × 稠密矩阵乘法，可参考其 aclsparse 接口风格、目录结构及构建流程。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 SpGEMM 算子，提供 `aclsparseSpGEMM*` 系列 C++ 接口，支持 fp32 / fp16 / bf16 三种同精度数据类型，CSR 稀疏格式，int32 索引，仅 NON_TRANSPOSE 操作。

## 需求拆解

1. 支持 fp32 / fp16 / bf16 同精度数据类型（A/B/C/computeType dtype 一致）
2. 仅 CSR 格式，int32 索引，zero-based
3. 仅 NON_TRANSPOSE（opA、opB 均为 `ACL_SPARSE_OP_NON_TRANSPOSE`）
4. 4 阶段调用流程：WorkEstimation → EstimateMemory → Compute → Copy
5. Host 侧符号分析：计算输出 C 的行指针 rowPtrC 和列索引 colIndC（sorted）
6. Device 侧数值计算：SIMT kernel 基于 dav-3510 架构，使用 `asc_vf_call` 调度
7. 确定性：fp32 达到 bit-wise deterministic
8. 性能：对标 cuSPARSE A100，NPU ≥ 1.0×

---

# 详细设计（required）

## 算子分析

### 数学公式

```
C = α · A · B + β · C
```

其中 A(m×k)，B(k×n)，C(m×n) 均为 CSR 稀疏矩阵。

单元素计算：`C[i][j] = sum over k: A[i][k] * B[k][j]`，仅对 A、B 中非零位置进行累加。

### 支持数据类型

| 参数 | 数据类型 |
|------|----------|
| A/B/C values | fp32 / fp16 / bf16 |
| 行偏移/列索引 | int32 |
| computeType | 同 A/B/C values dtype（同精度） |
| α/β | fp32（host 传递） |

### 支持形状

CSR 稀疏矩阵，m × k × n 任意正整数（含零维度），支持矩形矩阵。

### 算子约束

1. 仅 CSR 格式
2. 仅 NON_TRANSPOSE
3. 仅 int32 索引，zero-based
4. 输入 A/B 列索引必须 sorted
5. 输出 C 列索引保证 sorted

## 算子实现

### 总体架构

```
aclsparseSpGEMM* 系列接口（Host 侧 C++）
    ├── aclsparseSpGEMMCreateDescr / DestroyDescr
    ├── aclsparseSpGEMMWorkEstimation  (符号分析: 计算 rowPtrC)
    ├── aclsparseSpGEMMGetNumProducts  (获取中间产品数)
    ├── aclsparseSpGEMMEstimateMemory  (内存估算, ALG2/3)
    ├── aclsparseSpGEMMCompute         (数值计算: kernel launch)
    └── aclsparseSpGEMMCopy           (结果拷贝)
            │
            └── Ascend C SIMT Kernel (Device 侧数值计算)
                ├── spgemm_custom_fp32  (float)
                ├── spgemm_custom_fp16  (__fp16)
                └── spgemm_custom_bf16  (uint16, float 累加)
```

### Host 侧设计

#### 阶段 1: WorkEstimation（符号分析）

**输入**: CSR A (rowPtrA, colIndA, valsA), CSR B (rowPtrB, colIndB, valsB), matC placeholder

**步骤**:
1. 参数校验：dtype combo、format、operations、dimensions
2. D2H 拷贝 A 和 B 的 CSR 行偏移/列索引（不含 values，符号分析不需要 values）
3. Host 侧符号乘法：对每行 i，遍历 A 行 i 的列 kk，再遍历 B 行 kk 的列，收集到 `std::set<int32_t>` 自动去重排序
4. 累加每行非零元数得到 `nnzC`，构造 `rowPtrC`
5. 将 `rowPtrC` H2D 拷贝到 `matC->ptrs`
6. 存储 `rowPtrC` 和每行的列索引集合到 `spgemmDescr`

**边界处理**:
- m=0 / k=0 / nnzA=0 / nnzB=0 → C 为空矩阵，rowPtrC 全零
- A 或 B 行内列索引 unsorted → V1 要求输入 sorted（与 cuSPARSE 一致）

#### 阶段 2: EstimateMemory（内存估算）

- V1 简化：ALG1/ALG2/ALG3 统一使用 WorkEstimation 的 workspace
- `bufferSize2 = 0`（无需额外 buffer）
- `bufferSize3 = 1024`（占位）

#### 阶段 3: Compute（数值计算）

**步骤**:
1. 从 `spgemmDescr` 读取 `rowPtrC` 和每行列索引集合
2. Host 侧将列索引集合拼装为 `colIndC` 数组
3. H2D 拷贝 `colIndC` 到 `matC->idxs`
4. 在 external buffer 中填充 `SpgemmTilingData`（含所有 device 地址指针、维度、alpha/beta）
5. 调用 `spgemm_kernel_launch` 分发 kernel

**Tiling 数据 (`SpgemmTilingData`)**:

| 字段 | 说明 |
|------|------|
| m, k, n | 矩阵维度 |
| rowPtrA/B/C_ptr, colIndA/B/C_ptr, valsA/B/C_ptr | 各 CSR 数组的 device 地址 |
| chnk | SIMT 每线程列块数（V1 固定 1） |
| baseA/B/C | 索引基值（0/1） |
| alpha_host, beta_host | 标量值（float） |

#### 阶段 4: Copy

- V1 为 no-op（Compute 直接写入 matC->values，Copy 无需额外操作）

#### 分核策略

- 使用 `platform_ascendc` 获取 AIV 核心数（fallback 24 核）
- 按 block-cyclic 分布：`rowsPerBlock = (m + blockNum - 1) / blockNum`
- 每核内再通过 `asc_vf_call` 以 SIMT 线程并行处理行内非零元

#### Workspace 布局

```
[copy buf] [tiling data]
  └ A rowPtr/colInd/vals copy buf
  └ B rowPtr/colInd/vals copy buf
  └ C rowPtr/colInd/vals copy buf
  └ nnzPerRow
  └ SpgemmTilingData (64B aligned)
```

### Kernel 侧设计

#### 编程模型

- 架构: dav-3510（Ascend 950PR），SIMT 编程模型
- 入口: 通过 `<<<blockDim, nullptr, stream>>>` 启动 AIC 核
- 每核内通过 `asc_vf_call<SpgemmCsrSimtCompute<ScalarT>>(dim3{threadNum}, ...)` 调度 SIMT 线程到 AIV 向量核

#### 计算流程

```
每个 SIMT 线程：
  for each row i assigned to this thread:
    for each output C[i][j] in row i:
      val = 0
      for each non-zero A[i][k]:
        for each non-zero B[k][j]:
          val += A[i][k] * B[k][j]
      C[i][j] = alpha * val + beta * existing_C[i][j]
```

**关键设计要点**:
- B 行内使用线性扫描（compact rows 场景下比二分更快）
- 所有 AIC 核必须统一执行 `asc_vf_call`，空行区间 (`rowStart >= rowEnd`) 的核同样调用 SIMT 函数但立即返回，避免 cube→vector 死锁
- fp32/fp16 使用原生类型累加；bf16 以 `uint16_t` 存储，展开为 `float` 累加后截断回 bf16

#### 数据类型特化

| Kernel | 类型参数 | 累加精度 | 说明 |
|--------|---------|---------|------|
| `SpgemmCsrSimtCompute<float>` | float | fp32 | 原生 float 计算 |
| `SpgemmCsrSimtCompute<__fp16>` | __fp16 | fp16 | 原生 __fp16 计算 |
| `SpgemmCsrSimtComputeBf16` | uint16_t | fp32 | BF16 → float 展开计算 → 截断 |

#### 边界处理

- 空矩阵/零 nnz：WorkEstimation 已处理，Compute 直接返回 SUCCESS
- alpha=0, beta=0：kernel 内正常计算（alpha=0 输出全零；beta=0 跳过 C 初始值读取）
- 行内无 C 非零：循环自动跳过

---

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR (dav-3510) | √ |
| Atlas A2 (dav-2201) | 暂不支持 |

开发环境：CANN 9.0.0，编译器 bisheng；构建命令如下：

```bash
# 编译 SpGEMM 算子库
bash build.sh --ops=spgemm --soc=ascend950

# 编译并运行 SpGEMM 测试
bash build.sh --ops=spgemm --soc=ascend950 --run
```

## 算子约束限制

1. 仅支持 CSR 格式，不支持 COO/CSC/BSR
2. 仅支持 int32 索引（ptrType/idxType 均须 `ACL_SPARSE_INDEX_32I`）
3. 仅支持 zero-based 索引
4. 仅支持 NON_TRANSPOSE
5. 仅支持 fp32/fp16/bf16 同精度组合
6. 输入 A/B 列索引须 sorted（与 cuSPARSE 一致）
7. 输出 C 列索引保证 sorted

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | MERE < 1/(2^13) (fp32)，MARE < 10/(2^13) (fp32)；fp16/bf16 对应更宽松阈值 | 任务书 |
| 精度标准 | ATK 双标杆测试：NPU/CPU 最大相对误差比例 ≤ 2，平均 ≤ 1.2，均方根 ≤ 1.2 | 任务书 |
| 性能标准 | NPU ≥ cuSPARSE A100 1.0×（固定 3 组参考用例 + 200 组泛化覆盖） | 任务书 |

### 测试用例覆盖

| 编号 | 场景 | 规模 | 数据类型 | alpha/beta |
|------|------|------|---------|-----------|
| TC-01 | 小矩阵 | 32×32×16 | fp32/fp16/bf16 | 1.0/0.0 |
| TC-02 | 中等矩阵 | 128×128×64 | fp32 | 1.0/0.0 |
| TC-03 | beta 非零 | 64×64×32 | fp32/fp16 | 1.0/0.5 |
| TC-04 | alpha 缩放 | 64×64×32 | fp32 | 2.5/0.0 |
| TC-05 | 矩形矩阵 | 50×100×30 | fp32/fp16/bf16 | 1.0/0.0 |
| TC-06 | 空矩阵（零 nnz） | 32×32×16 | fp32 | 1.0/0.0 |
| TC-07 | 高稀疏度大图 | 1000×1000×1000, 0.1% | fp32 | 1.0/0.0 |
| TC-08 | 性能参考 01 | 128×128×128, 819 nnz each | fp32 | 1.0/0.0 |
| TC-09 | 性能参考 02 | 128×128×128, 0 nnz | fp32 | 1.0/0.0 |
| TC-10 | 性能参考 03 | 128×64×128, 163 nnz each | fp32 | 2.0/0.5 |

## 兼容性分析

新算子，不涉及兼容性分析。后续扩展方向：
- 支持 int64 索引
- 支持 opA/opB TRANSPOSE
- ALG2/ALG3 chunked 内存优化
- 更多数据类型（fp64、complex）
- COO 格式支持（V2）

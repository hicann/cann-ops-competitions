# aclblasCtpsv 算子设计文档

> 任务：9月社区任务-aclblasCtpsv算子开发（A2/A3）
> 版本：V1.0 2026-09-28
> 参考模板：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>

## 1 需求背景（required）

### 1.1 需求来源

本算子来源于 CANN 2026 年 9 月社区任务，目标是为 `ops-blas` 增加与 cuBLAS `cublasCtpsv` 语义一致的 Ascend C 直调实现，并提交到 `blas/tpsv/arch22/`。

### 1.2 背景介绍

`aclblasCtpsv` 用于求解复数单精度 packed 三角线性方程组：

```text
op(A) * x = b
```

右端项 `b` 初始存放在 `x` 中，求解结果原地写回 `x`。矩阵采用列主序 packed 存储，不带 `lda`，适用于 BLAS Level-2 中的小批量三角求解场景。A2/A3 使用同一 DAV_2201/`arch22` 实现目录，性能验证设备为 Atlas 800T A2（910B3）。

## 2 需求分析（required）

### 2.1 算子规格

| 参数 | 说明 |
|---|---|
| `handle` | 有效 aclblas 句柄，携带异步 stream |
| `uplo` | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| `trans` | `ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C` |
| `diag` | `ACLBLAS_NON_UNIT` 或 `ACLBLAS_UNIT` |
| `n` | 阶数，`n >= 0`；`n=0` 为 no-op |
| `AP` | complex64 packed 三角矩阵，长度 `n(n+1)/2` |
| `x` | complex64 原地输入/输出向量 |
| `incx` | 非零步长，支持正负值 |

函数原型：

```cpp
aclblasStatus_t aclblasCtpsv(aclblasHandle_t handle,
    aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int n, const aclblasComplex *AP,
    aclblasComplex *x, int incx);
```

### 2.2 数据布局与索引

复数按 `{real, imag}` 交错存储。0 基下标为：

```text
UPPER: AP[i + j(j+1)/2]                 (i <= j)
LOWER: AP[i + j(2n-j-1)/2]              (i >= j)
```

`incx > 0` 时逻辑元素 `i` 位于物理位置 `i*incx`；`incx < 0` 时位于 `(n-1-i)*abs(incx)`。AP 与 x 不允许重叠。

### 2.3 约束与异常行为

Host 侧检查句柄、枚举、`n`、`incx` 和必要指针。非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`，空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。`n=0` 在参数校验通过后直接成功返回，不访问 AP/x、不发射 kernel。UNIT 对角把对角视为复数 1，并保证 kernel 不读取对应 AP 元素。

### 2.4 精度与性能目标

- 输入输出：COMPLEX64。
- 精度：`atol=rtol=2^-13`，元素匹配率至少 99%，最大绝对误差按任务书的 `1e-2` 或 `32*ULP` 口径检查。
- 性能：任务书典型 case 的 NPU kernel 平均耗时不高于 `gpu_ms × 1000 / 0.8`（单位换算为 μs）；每条 case 5 次 warmup 后采样 11 次。
- 内存：不申请外部 workspace；kernel 使用 UB 临时区。

## 3 详细设计（required）

### 3.1 总体方案

工程采用 `ops-blas` kernel 直调模式：

```text
aclblasCtpsv
  ├─ 参数检查与 CtpsvTilingData 组装（Host）
  ├─ 按 uplo/trans/diag 选择 12 个编译期 kernel（Host）
  └─ 单 AIV 串行主元递推 + UB SIMD 列更新/内积（Kernel）
```

三角求解的主元依赖是串行的，因此以“一个 AIV 负责一个完整求解”保证前后主元可见；每个主元涉及的向量更新和内积使用 SIMD API。该方案优先保证 packed 索引、复数共轭、负步长和 UNIT 语义正确，再通过 UB 驻留、向量化和预取降低单核开销。

### 3.2 Host 侧设计

文件：`blas/tpsv/arch22/ctpsv_host.cpp`。

1. 判断 `handle==nullptr`。
2. 校验 `uplo/trans/diag` 枚举、`n>=0`、`incx!=0`。
3. 当 `n>0` 时校验 AP/x 非空；当 `n==0` 直接成功返回。
4. 将 AP/x 地址、n、属性和 incx 写入 `CtpsvTilingData`。
5. 通过 handle 的 stream 直调 `ctpsv_kernel_do`。

`CtpsvTilingData` 使用 64 位地址和 64 位 `incx`，避免大阶数和负步长地址计算溢出。

### 3.3 Kernel 侧设计

文件：`blas/tpsv/arch22/ctpsv_kernel.cpp`。

#### 3.3.1 编译期分支

对 `UPPER/LOWER × N/T/C × UNIT/NON_UNIT` 生成 12 个 kernel 符号，Host 根据枚举选择唯一分支，避免 kernel 内部属性分支。

#### 3.3.2 求解顺序

`LOWER+N` 和 `UPPER+T/C` 采用正向递推，其余采用反向递推。N 路径先除以主元再用该解更新关联列；T/C 路径先累加已知向量的复数内积，再除以主元。C 路径仅对矩阵虚部取负，实现共轭转置。

#### 3.3.3 UB 与 SIMD

`n<=4096` 时将 x 的实部、虚部驻留 UB，将交错 AP/x 拆成实部和虚部后执行向量乘加；大于 4096 时按 4096 元素分块搬运，保留同一算法语义。LOWER 活跃向量在 UB 中重排，减少反向访问导致的非对齐标量头处理。尾部不足一个向量时使用 DataCopyPad 和标量补齐。

复数乘法更新采用四个实数向量操作；转置/共轭转置通过两个向量归约得到实部和虚部内积。非单位对角使用稳定的复数除法，单位对角跳过 AP 对角读取。

#### 3.3.4 搬运与同步

GM 与 UB 间使用 `DataCopyPad`，GM 尾块允许非 32 字节长度，UB 起始地址保持对齐。MTE2→V、V→S、S→MTE3 等依赖使用成对 `SetFlag/WaitFlag`，向量阶段使用必要的 `PipeBarrier<PIPE_V>`。N+UNIT 路径尝试预取下一 packed 列，遇到零 RHS 或分支结束时回收未完成搬运。

#### 3.3.5 资源估算

固定 UB 缓冲区按 4096 元素规划，包含 x/AP 实虚部、计算临时区、交错搬运区、标量归约区和索引区，总量控制在 DAV_2201 单核 UB 容量范围内；不使用跨核同步和外部 workspace。

### 3.4 Python/ATen 适配

本任务交付为 `ops-blas` 的 aclblas C++ 接口，不新增 Python/ATen 注册层。Python 侧仅通过 CSV/GTest 和 msprof 脚本驱动验证；如后续需要 ATen 封装，可在不改变 kernel 参数布局的前提下增加独立 wrapper。

## 4 支持硬件与软件环境（required）

| 项目 | 支持/版本 |
|---|---|
| Atlas A2 | 支持，已在 Atlas 800T A2 / 910B3 验证 |
| Atlas A3 | 支持声明，共用 `arch22`；需在 A3 设备完成最终验收 |
| CANN | 9.1.0 |
| 编程模型 | Ascend C SIMD，DAV_2201 |
| 数据类型 | COMPLEX64 |

## 5 约束限制（required）

1. AP 与 x 不得重叠；不支持额外非连续视图语义。
2. 三角矩阵对角不能为零；任务测试对 NON_UNIT 对角加 boost，UNIT 不读取对角。
3. 单 AIV 主元递推受三角依赖限制，超大 n 的并行度主要来自核内 SIMD；该版本不宣称多核分解。
4. 非有限值行为遵循设备浮点语义。原始 `RANDOM_EXTREME` 的 FLT_MAX 输入可能因 CPU FMA、复数运行库和 NPU FTZ 差异产生不同 Inf/NaN 传播；测试默认使用对称防溢出调理，并在报告中单独披露。
5. A3 尚未在本轮服务器上实测，最终验收必须补做 A3 功能测试。

## 6 可维可测分析（required）

### 6.1 可维护性

- Host、tiling 数据和 kernel 分文件组织。
- 12 个属性组合由模板实例化，packed 索引、x 地址和复数运算均为独立小函数。
- 设计文档、测试 CSV 和性能汇总脚本随测试目录提交。

### 6.2 可测试性

测试工程位于 `test/tpsv/ctpsv/arch22/`：

| 类别 | 覆盖内容 |
|---|---|
| L0/L1/L2 | 基础枚举、阶数扫描、正负 incx |
| L5 | 随机、交替、极端、Inf/NaN 填充 |
| L6 | n=0、非法枚举、负 n、空指针、incx=0、空句柄 |
| WB | 17/4097 阶、分块回退、12 组合、负步长、UNIT 对角 NaN |
| PF | 任务书典型 case 与性能规模扫描 |

CPU golden 使用 Netlib CBLAS `cblas_ctpsv`，设备结果在 stream 同步后按混合容差逐复数元素检查。性能使用普通 `msprof` 的 `Task Duration(us)`，通过 case 参数和 kernel 名校验采样映射。

### 6.3 当前验证结果

- CANN 9.1.0 / 910B3 编译通过。
- 1000 条任务精度用例和 24 条白盒用例通过。
- 任务书 5 个典型性能 case 均达到目标，实测约 151、432、856、1194、4998 μs；对应任务阈值为 252.2、573.9、1228、1211、7327 μs。
- 全量 200 条性能扫描首轮为 157/200 通过，剩余主要为转置分支规模扫描，不能作为最终全量性能验收结论，后续继续优化。

## 7 参考资料

1. CANN 社区任务设计文档模板：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>
2. `aclblasCtpsv_A2A3_task_doc.md`
3. `test_cases/README.md`
4. Netlib CBLAS `ctpsv` 接口说明

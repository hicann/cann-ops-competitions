# aclblasScasum 算子设计文档

> 社区任务：8月社区任务-aclblasScasum算子开发（950）
> 适配硬件：Ascend 950PR（arch35 / DAV_3510）
> CANN 版本：CANN 9.1.0
> 对标接口：cuBLAS `cublasScasum` / Netlib BLAS `scasum`
> 工程框架：ops-blas 开源仓 `blas/asum/arch35/`（kernel 直调，句柄式 BLAS 接口）

---

## 一、需求背景

### 1.1 需求来源

通过社区任务完成昇腾算子开源仓（ops-blas）的算子贡献需求。本任务为在 Ascend 950PR
上使用 Ascend C 编程语言开发单精度复数（complex64）向量绝对值分量之和算子
`aclblasScasum`，完成算子设计、开发、测试全流程工作，验收通过后合入昇腾算子开源仓
ops-blas 的 `blas/asum/arch35/` 目录。

### 1.2 背景介绍

#### 1.2.1 aclblasScasum 算子开发

`aclblasScasum` 是 BLAS Level-1 复数归约算子，计算单精度复数向量 `x` 各元素实部与
虚部绝对值之和：

```
result = Σ ( |Re(x[i])| + |Im(x[i])| )，i = 1..n
```

- 对标基线：cuBLAS `cublasScasum`（https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-asum ）
  与 Netlib BLAS `scasum`（https://www.netlib.org/blas/scasum.f ）。
- 任务书来源：docs/README.md 8月份发放任务 #48 `8月社区任务-aclblasScasum算子开发（950）`。
- 本算子无历史 TBE 版本，为全新算子，参考实现为 cuBLAS / Netlib 浮点语义。
- ops-blas 仓内已有同族实数算子 `aclblasSasum`（`blas/asum/arch35/`），结构可直接扩展复用。

#### 1.2.2 标杆算子现状分析

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

cuBLAS `cublasScasum` / Netlib `scasum` 支持能力：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 数据排布 | 维度 |
|------|----------|----------|--------------|----------|------|
| x | 输入向量 | tensor | `cuComplex`（COMPLEX64，实/虚部各 float32） | ND，交错存储 | 逻辑一维 [n]，物理长度 `1+(n-1)*\|incx\|` |
| n | 元素个数 | scalar | int32 | - | - |
| incx | 步长 | scalar | int32 | - | - |
| result | 输出标量 | scalar | float32 | - | 单值 |

- 复数类型内存排布：`aclblasComplex`（实部/虚部各 float32，交错存储），n 个复数元素
  对应 `2*n` 个连续 float。
- quick return 语义（cuBLAS 文档与 Netlib `scasum.f` 一致）：`n <= 0 或 incx <= 0` 时
  不遍历向量，直接 `result = 0.0f` 并返回成功。

##### 1.2.2.2 标杆算子实现描述

Netlib `scasum.f` 核心逻辑：

```fortran
SCASUM = 0
IF (N.LE.0 .OR. INCX.LE.0) RETURN
IF (INCX.EQ.1) THEN
    DO I = 1,N
        SCASUM = SCASUM + ABS(REAL(X(I))) + ABS(AIMAG(X(I)))
    END DO
ELSE
    NINCX = N*INCX
    DO I = 1,NINCX,INCX
        SCASUM = SCASUM + ABS(REAL(X(I))) + ABS(AIMAG(X(I)))
    END DO
END IF
RETURN
```

实现要点：

1. `incx == 1`：连续遍历 n 个复数元素，逐项累加 `|Re| + |Im|`。
2. `incx != 1`：按步长 `incx`（单位：复数元素）遍历，物理长度 `n*incx`。
3. 累加顺序为顺序累加，输出为单个 float32 标量；归约误差来自逐项舍入。
4. 累加项逐项非负，无抵消放大；大 n × 接近 FLT_MAX 时结果可溢出到 `+Inf`。

##### 1.2.2.3 标杆算子实现流程图

```
┌─────────────────────────────────────────────┐
│ 输入：x (complex), n, incx, result           │
└───────────────────┬─────────────────────────┘
                    ▼
        n ≤ 0 或 incx ≤ 0 ?
        ├── 是 ──► result = 0.0f，RETURN
        ▼ 否
    incx == 1 ?
    ├── 是 ──► 顺序累加 Σ(|Re(X(i))| + |Im(X(i))|), i=1..n
    └── 否 ──► 跨步累加 Σ(|Re(X(1+(i-1)*incx))| + |Im(X(1+(i-1)*incx))|)
                    ▼
             result = 累加和，RETURN
```

---

## 二、需求分析

### 2.1 外部组件依赖

- Ascend C 算子开发工具链（CANN 9.1.0，含 ccec_compiler / tikicpulib）。
- ops-blas 开源仓工程框架（kernel 直调、句柄式 BLAS 接口）。
- 精度比对 golden 由 cblas（Netlib BLAS 复数实现 scasum）语义生成，随测试工程提供，
  无其他三方软件依赖。

### 2.2 内部适配模块

- ops-blas 仓 `blas/asum/arch35/`：复用同族实数算子 `aclblasSasum` 的 host/kernel
  工程结构（tiling 数据结构、kernel 直调、SIMT 线程级归约）。
- ops-blas 仓 `include/cann_ops_blas.h`：新增 `aclblasScasum` 接口声明，与其他产品线
  共用同一 API，禁止定义 950PR 私有平行接口。

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

接口声明放入 `include/cann_ops_blas.h`，签名与 `cublasScasum` 参数序列逐参数对齐，
与仓内同族 `aclblasSasum` 同构：

```cpp
aclblasStatus_t aclblasScasum(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    float* result);
```

| 参数 | 方向 | 语义 | 异常行为 |
|------|------|------|----------|
| handle | 输入 | 库上下文句柄，携带 stream | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n | 输入 | 复数元素个数（Host） | n≤0 → quick return（result=0.0f，SUCCESS） |
| x | 输入 | 单精度复数向量（Device，只读），物理长度 `1+(n-1)*\|incx\|` | n>0 且 incx>0 时 nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | 输入 | 复数步长（Host） | incx≤0 → quick return（result=0.0f，SUCCESS） |
| result | 输出 | FLOAT32 实数标量（Device） | nullptr（n>0 且 incx>0）→ `ACLBLAS_STATUS_INVALID_VALUE` |

返回值 `aclblasStatus_t`，状态码语义与 `include/cann_ops_blas_common.h` 一致。

#### 2.3.2 Ascend C 算子相关约束

与标杆算子相比，Ascend C 版本能力对齐情况：

| 标杆能力 | Ascend C 版本 | 说明 |
|----------|---------------|------|
| COMPLEX64 输入 | ✅ 支持 | `aclblasComplex` 交错存储，按 float 视图处理 |
| incx==1 连续 | ✅ 支持 | SIMT 统一路径（stride=1，实测优于 AIV 方案） |
| incx!=1 跨步 | ✅ 支持 | SIMT 统一路径（stride=incx，同一 kernel） |
| n≤0 / incx≤0 quick return | ✅ 支持 | 不触发 kernel，result=0.0f |
| 输出 FLOAT32 标量 | ✅ 支持 | 与标杆一致 |
| 归约顺序 | 不要求一致 | 按精度阈值判定，不做 bit-exact |

无功能缺失，参数语义与标杆完全对齐。

---

## 三、需求详细设计

### 3.1 调用方式

采用 **kernel 直调**（ops-blas 句柄式 BLAS 接口）：

```
用户应用 → aclblasScasum(handle, n, x, incx, result)
    → handle 绑定 stream（aclblasSetStream）
    → host 侧参数校验 + tiling 计算
    → 直调 NPU kernel（<<<grid, nullptr, stream>>>）
    → 读回前同步 stream
```

### 3.2 需求总体设计

#### 核心数学等价变换

对每个复数元素 `(Re, Im)`，其贡献为 `|Re| + |Im|`；而复数在内存中按
`[Re0, Im0, Re1, Im1, ...]` 交错存储。因此：

```
result = Σ_k (|Re_k| + |Im_k|) = Σ_j |float_j|   （j 遍历 2n 个连续 float）
```

即**复数向量绝对值分量之和 ≡ 把同一段内存当作 2n 个 float 后逐元素取绝对值再求和**。
该等价性在浮点上严格成立，归约累加顺序差异仅影响舍入误差，满足精度阈值（不做
bit-exact）。基于此，scasum 复用 sasum 的工程骨架，归约采用 **SIMT 线程级部分和 +
warp 级硬件归约**（`asc_reduce_add`）方案：经 msprof `PipeUtilization` 实测验证，
AIV（SIMD）路径的 `ReduceSum` 存在标量调度硬伤（SCALAR 占比 49.6%，约 18us），
而 SIMT 线程级归约 SCALAR 仅 13.6%（约 2.65us），故 incx==1 与 incx!=1 统一走
SIMT 单路径，仅寻址方式不同（连续 / 跨步）。

工程新增三个文件（避免与 `Sasum*` 符号冲突）：

- `scasum_tiling_data.h`：独立的 `ScasumTilingData`
- `scasum_host.cpp`：参数校验、quick return、tiling、launch、kernel 派发
- `scasum_kernel.cpp`：SIMT 统一路径（连续与跨步仅 stride 取值不同）+ 单 kernel 跨核归约

tiling 采用**复数粒度**切分：`startOffset[i]` / `calNum[i]` 以复数元素为单位，各核处理
连续的若干复数元素，SIMT 线程按 `(startOffset + i) * stride` 寻址，stride=1 即连续路径。

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

- 优先使用满核，多核均分数据块。
- 核数计算：`numBlocks = min(aivCoreNum, n)`（复数元素数上限），
  `useCoreNum = min(numBlocks, n)`，上限 `SCASUM_MAX_CORE_NUM = 64`（与 sasum 一致）。
- 核间均分：`calNum[i] = (n / useCoreNum) + (i < n % useCoreNum ? 1 : 0)`；
  `startOffset[i] = i * (n / useCoreNum) + min(i, n % useCoreNum)`，前 `n % useCoreNum`
  个核各多处理 1 个复数元素（大核/小核策略）。

##### 3.2.1.2 数据分块和内存优化策略

SIMT 统一路径（incx 任意）：线程直接访问 GM，不经 MTE2/UB 中转。

- 每核启动 `nthreads` 个线程（`asc_vf_call`，`LAUNCH_BOUND` 对齐到
  `[SIMT_MIN_THREAD_NUM=128, SIMT_MAX_THREAD_NUM=2048]`），线程按跨步遍历本核复数区间：
  `i = threadIdx.x, threadIdx.x + blockDim.x, ...`，
  以 `float2` 视图一次 8B 读取实部/虚部：
  `partial += (re >= 0 ? re : -re) + (im >= 0 ? im : -im)`（三元条件 abs，
  实测比 `fabsf` 编译路径快约 60%）。
- **warp 级归约**：每 32 线程一个 warp，用 `asc_reduce_add(partial)` 硬件归约
  （单指令完成，无 `syncthreads`），warp leader 写 `ubPartialSums[warpId]`；
  全块仅 1 次 `asc_syncthreads`；warp 0 再用 `asc_reduce_add` 归约各 warp 部分和
  （支持最多 64 warp / 2048 线程，超过 32 warp 的尾部由 thread 0 串行累加）。
- **动态线程数**：`n < 2M` 用 1024 线程（32 寄存器/线程），`n >= 2M` 用 2048 线程
  （16 寄存器/线程），大 n 下更多 in-flight load 掩盖 RVECLD 访存延迟。
- 加载与计算均发生在 VEC 流水（SIMT 的 GM 加载走 `RVECLD` 指令，不走 MTE2 DMA），
  msprof 实测 `aiv_mte2_ratio = 0%`、`aiv_vec_ratio = 62~78%`，为 VEC-bound 算子。
- 负步长 / 零步长不会到达 kernel（host 侧 quick return 拦截），kernel 内 stride 恒正。

##### 3.2.1.3 跨核归约策略

各核将本核部分和写入 `workspace[blockIdx]`（handle 预留 workspace），随后：

- `CrossCoreSetFlag<0, PIPE_MTE3>(0)` + `CrossCoreWaitFlag<0, PIPE_MTE3>(0)`
  完成全核同步（保证各核 workspace 写入对 block 0 可见）；
- block 0 串行累加 `workspace[0..useCoreNum)` 写回 result（标量级，开销可忽略）。

采用单 kernel 内联跨核归约而非独立 reduce kernel：省一次 kernel 启动开销
（host 侧异步提交约 2us + device 调度固定成本），端到端更优。

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

**SIMT 统一路径（incx 任意，连续时 stride=1）**

```
每 block：asc_vf_call<ScasumSimtCompute>(nthreads, calNum, startOffset, stride, x, partialOut)
  线程 t：i = t, t+blockDim, ...；idx = (startOffset + i) * stride
          float2 c = xf2[idx]；partial += |c.x| + |c.y|（三元条件 abs）
  warp 归约：asc_reduce_add(partial) → warp leader 写 ubPartialSums[warpId]
  全块 1 次 asc_syncthreads → warp 0 二次 asc_reduce_add → partialOut[blockIdx]
跨核归约：CrossCoreSetFlag/WaitFlag<PIPE_MTE3> 同步后，block 0 串行累加
          workspace[0..useCoreNum) → result
```

SIMT 线程从 `__gm__ aclblasComplex*` 按 float2 视图读取实部/虚部（`aclblasComplex`
与 `float2` 均为 8B 交错布局，直接 reinterpret，避免 device 侧 struct 构造问题）。

##### 3.2.2.2 Ascend C 实现流程图

```
        aclblasScasum(handle, n, x, incx, result)
                        │
        ┌───────────────▼───────────────┐
        │ handle==nullptr ?            │──是──► HANDLE_IS_NULLPTR
        │ n>0 && incx>0 && x==nullptr  │──是──► INVALID_VALUE
        │ result==nullptr (n>0,incx>0) │──是──► INVALID_VALUE
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────▼───────────────┐
        │ n<=0 或 incx<=0 ?             │──是──► 不启动 kernel
        └───────────────┬───────────────┘        aclrtMemcpy(result, 0.0f) → SUCCESS
                        ▼
            tiling：useCoreNum/calNum/startOffset/nthreads
                        ▼
        ┌───────────────▼───────────────┐
        │ scasum_simt_kernel（单 kernel）│
        │  每核: SIMT 线程部分和(float2) │
        │        → warp 归约(asc_reduce_add)
        │        → workspace[blockIdx]  │
        │  跨核: CrossCore 同步          │
        │  block 0: Σ workspace → result│
        └───────────────┬───────────────┘
                        ▼
                     同步 stream，返回 SUCCESS
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| 差异点 | 标杆算子（Netlib scasum） | Ascend C 实现 | 原因 |
|--------|---------------------------|---------------|------|
| 归约顺序 | 顺序累加 | 分核线程级部分和 + warp 硬件归约 + 跨核归约 | 多核并行；`asc_reduce_add` 单指令归约 32 线程，SCALAR 占比仅 13.6%（AIV ReduceSum 方案为 49.6%，已实测排除）；任务书不要求归约顺序逐位一致，按精度阈值判定 |
| 数据视图 | 复数逐项访问 | 复数内存按 float2 视图统一处理（线程寄存器内 abs+累加） | `Σ(|Re|+|Im|) == Σ|float|` 数学等价，一次 8B 向量化加载 |
| 执行方式 | 单线程标量循环 | 多核 SIMT kernel 直调（线程级并行） | NPU 并行架构；SIMT 直访 GM 无 MTE2/UB 中转，动态线程数（1024/2048）掩盖 RVECLD 访存延迟 |
| 归约骨架选型 | —（单路径） | SIMT 单路径（AIV ReduceSum 方案经 msprof 实测排除） | AIV 路径 `ReduceSum` 标量调度硬伤（SCALAR 49.6%、约 18us），SIMT 线程级归约 SCALAR 仅 13.6% |
| quick return | CPU 分支直接置 0 | host 侧分支，不启动 kernel，H2D 写 0.0f | 避免无意义 kernel 启动，语义一致 |
| 溢出行为 | float32 累加溢出 | 部分和累加同样 float32，溢出语义一致 | 任务书要求溢出按 Inf 一致性判定 |

### 3.3 支持硬件

| 芯片版本 | 支持 |
|----------|:---:|
| Atlas A2/A3（ascend910b*） | ✗ |
| Ascend 950PR（arch35 / DAV_3510） | ✅ |

与任务书 §3.1 要求的适配硬件（Ascend 950PR）保持一致。

### 3.4 算子约束限制

- 输入数据类型仅支持 COMPLEX64（`aclblasComplex`）；输出 FLOAT32 标量。
- 不支持半精度/混合精度。
- 非连续向量由 `incx` 表达；`incx <= 0`（含 0 与负步长）按 quick return 处理，不遍历。
- 无 broadcast / 原地更新 / 动态 shape 需求。
- 大 n × 极端值组合下结果可溢出到 `+Inf`，golden 同档溢出，按 Inf 一致性判定。

---

## 四、特性交叉分析

| 特性 | 分析 |
|------|------|
| 精度与性能 | SIMT 线程级归约 + `asc_reduce_add` warp 硬件归约，精度远优于阈值（实测大 n 误差 ULP 级）；4M 实测带宽 1.53 TB/s（≈950PR Server 理论 1.6 TB/s 的 96%），实测 1M=11.2us / 2M=15.8us / 4M=20.9us，全部优于任务书 §3.3 标杆（29.4/30.53/35.52us，富余 1.65~2.62×） |
| 多核扩展 | 分核策略按复数粒度均分，核数与线程数动态适配（1024/2048），大 n 下线性扩展 |
| 单路径统一 | SIMT 单 kernel 覆盖连续（stride=1）与跨步（stride=incx），无路径分支；AIV ReduceSum 方案经 msprof 实测排除（SCALAR 49.6% 硬伤） |
| 复用性 | 接口声明入公共头供其他产品线复用；SIMT 骨架与 sasum 同构，便于后续 c/cs 系列 BLAS 复现 |

---

## 五、可维可测分析

### 5.1 精度标准 / 性能标准

**精度标准**（任务书 §3.2，生态算子开源精度标准 FLOAT32 输出标量档）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| FLOAT32（输出标量；输入 COMPLEX64 分量同档） | 2⁻¹⁰ (9.77e-4) | 2⁻¹⁶ (1.53e-5) | 0.99 | 1e-2 或 32×ULP |

通过条件：`|actual - golden| ≤ atol + rtol × |golden|`（标量比对，单值判定）；
golden 由 cblas（Netlib scasum）语义生成；quick return 用例 golden 固定 0.0f；
归约顺序不要求逐位一致，不做 bit-exact 判定。

**性能标准**（任务书 §3.3，COMPLEX64 输入，warmup 后有效采样 >50 次取平均；按修订版任务书目标）：

| case | n | incx | 标杆耗时（Avg，us） | 实测（3 轮平均） | 富余 |
|------|-----|------|---------------------|------------------|------|
| 1 | 1048576 | 1 | 29.40 | 11.2us | 2.62× |
| 2 | 2097152 | 1 | 30.53 | 15.8us | 1.93× |
| 3 | 4194304 | 1 | 35.52 | 20.9us | 1.65× |

**自测**：GTest 参数化测试（1200 CSV 用例，覆盖 L0 基础/L1 尺寸扫描/L2 步长/L5 填充/
L6 边界/OV 溢出/PF 性能），精度 golden 用 double 精度累加（近精确，避免 float32 顺序
累加自身误差干扰 32×ULP 判定）；`verify_performance.py` 采集性能并对比标杆。

### 5.2 兼容性分析

- 新算子，无历史版本，不涉及接口/行为兼容性问题。
- 接口声明放入公共头 `include/cann_ops_blas.h`，与其他产品线共用同一 `aclblasScasum`
  API，禁止 950PR 私有平行接口，保证后续产品线扩展兼容。
- 仓内 README 产品支持表新增 `aclblasScasum` 条目，标注 Ascend 950PR：支持。
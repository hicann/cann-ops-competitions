# aclblasCdotu 算子设计文档

> 社区任务：8月社区任务-aclblasCdotu算子开发（950）
> 适配硬件：Ascend 950PR（arch35 / DAV_3510）
> CANN 版本：CANN 9.1.0
> 对标接口：cuBLAS `cublasCdotu` / Netlib BLAS `cdotu`
> 工程框架：ops-blas 开源仓 `blas/dot/arch35/`（kernel 直调，句柄式 BLAS 接口）

---

## 一、需求背景

### 1.1 需求来源

通过社区任务完成昇腾算子开源仓（ops-blas）的算子贡献需求。本任务为在 Ascend 950PR
上使用 Ascend C 编程语言开发单精度复数（complex64）向量无共轭点积算子
`aclblasCdotu`，完成算子设计、开发、测试全流程工作，验收通过后合入昇腾算子开源仓
ops-blas 的 `blas/dot/arch35/` 目录。

### 1.2 背景介绍

#### 1.2.1 aclblasCdotu 算子开发

`aclblasCdotu` 是 BLAS Level-1 复数点积算子，计算两个单精度复数向量的无共轭点积：

```
result = Σ( x[k] × y[j] )，i = 1..n，k = 1+(i-1)*incx，j = 1+(i-1)*incy
```

- 对标基线：cuBLAS `cublasCdotu`（https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-dot ）
  与 Netlib BLAS `cdotu`（https://www.netlib.org/blas/cdotu.f ）。
- 任务书来源：docs/README.md 8月份发放任务 #49 `8月社区任务-aclblasCdotu算子开发（950）`。
- 与 `cublasCdotc` 的区别：仅函数名以 'c' 结尾时对向量 x 取共轭，`cublasCdotu` **不取共轭**。
- 本算子无历史 TBE 版本，为全新算子；ops-blas 仓已有同族实数算子 `aclblasSdot`
  （`blas/dot/arch35/`）与 arch22 版 `cdotu/cdotc`（`blas/dot/arch22/`，仅支持单位步长），
  结构可参考，但需为 arch35 新增满足本任务要求（支持任意非零步长含负步长）的实现。

#### 1.2.2 标杆算子现状分析

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

cuBLAS `cublasCdotu` / Netlib `cdotu` 支持能力：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 数据排布 | 维度 |
|------|----------|----------|--------------|----------|------|
| x | 输入向量 | tensor | `cuComplex`（COMPLEX64，实/虚部各 float32） | ND，交错存储 | 逻辑一维 [n]，物理长度 `1+(n-1)*\|incx\|` |
| y | 输入向量 | tensor | `cuComplex`（COMPLEX64） | ND，交错存储 | 逻辑一维 [n]，物理长度 `1+(n-1)*\|incy\|` |
| n | 元素个数 | scalar | int32 | - | - |
| incx / incy | 步长 | scalar | int32 | - | - |
| result | 输出标量 | tensor | `cuComplex`（COMPLEX64） | ND | 标量 [1] |

- 复数类型内存排布：`aclblasComplex`（实部/虚部各 float32，交错存储），n 个复数元素
  对应 `2*n` 个连续 float。
- 复数乘法语义：`(a+bi)(c+di) = (ac−bd) + (ad+bc)i`；逐项累加实部
  `+= x.re*y.re − x.im*y.im`，虚部 `+= x.re*y.im + x.im*y.re`。
- 支持任意非 0 的 incx/incy（含负步长），负步长索引语义对齐 Netlib `cdotu`：起始索引为
  `(-n+1)*inc + 1`（1-based）；n = 0 为合法 no-op，result 置 (0, 0)。

##### 1.2.2.2 标杆算子实现描述

Netlib `cdotu.f` 核心逻辑：

```fortran
CDOTU = (0.0, 0.0)
IF (N.LE.0) RETURN
IF (INCX.EQ.1 .AND. INCY.EQ.1) THEN
    DO I = 1,N
        ZX = X(I); ZY = Y(I)
        CDOTU = CDOTU + ZX * ZY
    END DO
ELSE
    IX = 1; IY = 1
    IF (INCX.LT.0) IX = (-N+1)*INCX + 1
    IF (INCY.LT.0) IY = (-N+1)*INCY + 1
    DO I = 1,N
        ZX = X(IX); ZY = Y(IY)
        CDOTU = CDOTU + ZX * ZY
        IX = IX + INCX; IY = IY + INCY
    END DO
END IF
RETURN
```

实现要点：

1. `incx == incy == 1`：连续遍历 n 个复数元素，逐项 `+=` 复数乘。
2. 任意非零步长（含负步长）：负步长时起始索引为 `(-n+1)*inc + 1`（1-based），
   按步长逐步后移。
3. 复数乘法展开为实部/虚部两组独立的 float 累加。
4. 累加项可正可负（复数乘法结果有抵消），误差随 n 累积；matched_ratio ≥ 0.99
   允许少量离群点，正常实现应满足精度阈值。

##### 1.2.2.3 标杆算子实现流程图

```
┌─────────────────────────────────────────────┐
│ 输入：x (complex), incx, n, y, incy, result  │
└───────────────────┬─────────────────────────┘
                    ▼
              n ≤ 0 ?
              ├── 是 ──► result = (0,0)，RETURN
              ▼ 否
        incx==1 且 incy==1 ?
        ├── 是 ──► I 循环顺序累加 Σ x(I)*y(I)
        └── 否 ──► 确定 IX/IY 起始（负步长反向）
                  I 循环累加 Σ x(IX)*y(IY)，IX+=incx, IY+=incy
                    ▼
             result = 累加和（实/虚部分别），RETURN
```

---

## 二、需求分析

### 2.1 外部组件依赖

- Ascend C 算子开发工具链（CANN 9.1.0，含 ccec_compiler / tikicpulib）。
- ops-blas 开源仓工程框架（kernel 直调、句柄式 BLAS 接口）。
- 精度比对 golden 由 cblas（Netlib BLAS 复数实现 cdotu）语义生成，随测试工程提供，
  无其他三方软件依赖。

### 2.2 内部适配模块

- ops-blas 仓 `blas/dot/arch35/`：参考同目录实数算子 `aclblasSdot` 的 host/kernel
  工程结构（tiling 数据结构、kernel 直调、连续/跨步双路径）；参考 `blas/gemm_batched/arch35/`
  的复数实/虚部分离 Gather 技巧。
- ops-blas 仓 `include/cann_ops_blas.h`：`aclblasCdotu` 已有声明，与其他产品线共用，
  禁止定义 950PR 私有平行接口。

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

接口声明与 ops-blas 仓 `include/cann_ops_blas.h` 中已有声明保持一致：

```cpp
aclblasStatus_t aclblasCdotu(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y,
    int incy, aclblasComplex* result);
```

| 参数 | 方向 | 语义 | 异常行为 |
|------|------|------|----------|
| handle | 输入 | 库上下文句柄，携带 stream | nullptr → `ACLBLAS_STATUS_NOT_INITIALIZED` |
| n | 输入 | 复数元素个数（Host），n ≥ 0 | n < 0 → `ACLBLAS_STATUS_INVALID_VALUE`；n = 0 → no-op（result=(0,0)，SUCCESS） |
| x | 输入 | 复数向量（Device，只读），物理长度 `1+(n-1)*\|incx\|`，不取共轭 | n>0 时 nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | 输入 | x 步长（复数元素单位，可正可负） | incx = 0 → `ACLBLAS_STATUS_INVALID_VALUE` |
| y | 输入 | 复数向量（Device，只读），物理长度 `1+(n-1)*\|incy\|` | n>0 时 nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| incy | 输入 | y 步长（复数元素单位，可正可负） | incy = 0 → `ACLBLAS_STATUS_INVALID_VALUE` |
| result | 输出 | 无共轭点积结果，复数标量（Device） | n>0 时 nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |

返回值 `aclblasStatus_t`，状态码语义与 `include/cann_ops_blas_common.h` 一致。

#### 2.3.2 Ascend C 算子相关约束

与标杆算子相比，Ascend C 版本能力对齐情况：

| 标杆能力 | Ascend C 版本 | 说明 |
|----------|---------------|------|
| COMPLEX64 输入 | ✅ 支持 | `aclblasComplex` 交错存储，按 float 视图 + Gather 分离实/虚部 |
| incx=incy=1 连续 | ✅ 支持 | AIV 连续路径（性能最优） |
| 任意非零步长（含负） | ✅ 支持 | SIMT 跨步路径，负步长按 `(-n+1)*inc+1` 语义 |
| n=0 no-op | ✅ 支持 | 不触发 kernel，result=(0,0) |
| 输出 COMPLEX64 标量 | ✅ 支持 | 实部/虚部各 float32，与标杆一致 |
| 归约顺序 | 不要求一致 | 按精度阈值判定，不做 bit-exact |

无功能缺失，参数语义与标杆完全对齐（优于 arch22 版"仅支持单位步长"的现状）。

---

## 三、需求详细设计

### 3.1 调用方式

采用 **kernel 直调**（ops-blas 句柄式 BLAS 接口）：

```
用户应用 → aclblasCdotu(handle, n, x, incx, y, incy, result)
    → handle 绑定 stream（aclblasSetStream）
    → host 侧参数校验 + tiling 计算
    → 直调 NPU kernel（<<<grid, nullptr, stream>>>）
    → 读回前同步 stream
```

### 3.2 需求总体设计

#### 核心数学变换

复数乘法展开为实/虚部两组独立 float 归约：

```
real = Σ( x.re*y.re − x.im*y.im )   （float 归约）
imag = Σ( x.re*y.im + x.im*y.re )   （float 归约）
```

- 复数在内存中按 `[Re0, Im0, Re1, Im1, ...]` 交错存储。
- 连续路径（incx=incy=1）将 x/y 视为 `2n` 个连续 float，用 **Gather（byte-offset 表）**
  分离出实部/虚部向量，再逐项相乘组合后归约。
- 跨步路径（含负步长）用 SIMT 线程级复数乘加，按 `2*idx*|inc|`（负步长按
  `2*(n-1-idx)*|inc|`）float 寻址，正确性优先。
- 输出为单个复数标量（2 个 float：result[0]=real，result[1]=imag）。

工程新增三个文件（避免与 `Sdot*` / arch22 `Cdot*` 符号冲突）：

- `cdotu_tiling_data.h`：独立的 `CdotuTilingData`
- `cdotu_host.cpp`：参数校验、no-op、tiling、launch、kernel 派发
- `cdotu_kernel.cpp`：AIV 连续路径（incx=incy=1）+ SIMT 跨步路径（含负步长）+ reduce

tiling 采用**复数粒度**切分：`startOffset[i]` / `calNum[i]` 以复数元素为单位，各核处理
连续的若干复数索引，AIV 与 SIMT 两条路径都能正确按 incx/incy 寻址。

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

- 优先使用满核，多核均分数据块。
- 核数计算：`numBlocks = min(aivCoreNum, n)`，`useCoreNum = min(numBlocks, n)`，
  上限 `CDOTU_MAX_CORE_NUM = 64`；`n == 0` 不启动 kernel。
- 核间均分：`calNum[i] = n/useCoreNum + (i < n%useCoreNum ? 1 : 0)`；
  `startOffset[i] = i*(n/useCoreNum) + min(i, n%useCoreNum)`，前 `n%useCoreNum` 个核各
  多处理 1 个复数元素（大核/小核策略）。

##### 3.2.1.2 数据分块和内存优化策略

AIV 路径（incx=incy=1）：复数连续，按 float 视图加载 `2*calNum` 个 float 到 UB。

- UB 预算：`UB_SIZE = 248*1024` 字节，预留 `SAFETY_MARGIN` 给 TPipe/TQue 元数据。
- 单次 chunk 上限按 UB 切分：`xQueue/yQueue`（各 `2*chunk*4B`，double buffer）+
  Gather 偏移表（`2*chunk*4B`）+ 实/虚部临时缓冲（`x_re/x_im/y_re/y_im` 各 `chunk*4B`）
  + 乘积/累加缓冲，总预算不超过 UB 余量；chunk 取 8 对齐。
- 每核循环处理多个 chunk，`DataCopy → Gather → Mul/Sub/Add → ReduceSum → 原子累加`
  流水重叠（double buffer）。
- 尾块：不足一个 chunk 时用 DataCopyPad 补零对齐到 32B，保证向量指令对齐。

SIMT 路径（incx≠1 或 incy≠1，含负步长）：复数跨步，采用 SIMT 线程级部分和。

- 每核启动 `nthreads` 个线程（`asc_vf_call`），线程按复数索引跨步遍历本核区间。
- 正步长：float 索引 `2*idx*|inc|`；负步长：`2*(n-1-idx)*|inc|`（对齐 Netlib 语义）。
- 每线程私有 real/imag 累加，块内树形归约到 workspace；正确性优先。

##### 3.2.1.3 tilingKey 规划策略

- tilingKey = 0：`incx == 1 && incy == 1`（连续路径，AIV）。
- tilingKey = 1：`incx != 1 || incy != 1`（跨步路径，SIMT）。
- host 按 `tdata.incx/incy` 判断并派发对应 kernel；n=0 不启动 kernel，直接写
  result=(0,0)；n<0 / incx=0 / incy=0 返回 INVALID_VALUE。

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

**路径 A — AIV 连续路径（incx == 1 && incy == 1）**

```
Init：解析 tiling → xGM/yGM/resultGM/workspaceGM 绑定 → 初始化队列/缓冲
      → 构建 Gather byte-offset 表（evenOff[i]=i*8B, oddOff[i]=i*8B+4B）
Process：
  循环 chunk：
    DataCopy 加载 xF、yF（各 2*chunk 个 float）
    Gather 分离 x_re, x_im, y_re, y_im（各 chunk）
    t1 = x_re*y_re；t2 = x_im*y_im；realTerms = t1 - t2
    t3 = x_re*y_im；t4 = x_im*y_re；imagTerms = t3 + t4
    ReduceSum(realTerms) → realPart；ReduceSum(imagTerms) → imagPart
    accReal += realPart；accImag += imagPart
  workspace[coreIdx] = accReal；workspace[useCoreNum+coreIdx] = accImag
```

Gather 偏移表用 `Scalar SetValue` 写基础组（8 个 int32=32B）+ `Adds` 向量扩展构造，
与 `gemm_batched/arch35` 的 `GbBuildGatherOffsetTable` 同款模式。

**路径 B — SIMT 跨步路径（incx≠1 或 incy≠1，含负步长）**

```
每 block：asc_vf_call<CdotuSimtCompute>(nthreads, calNum, startIdx, incx, incy, x, y, partialOut)
  线程 t：i = t, t+blockDim, ...
          idx = startIdx + i（本核内第 i 个复数索引，全局逻辑索引）
          xbase = incxPos ? 2*idx*|incx| : 2*(n-1-idx)*|incx|
          ybase = incyPos ? 2*idx*|incy| : 2*(n-1-idx)*|incy|
          real += xf[xbase]*yf[ybase] − xf[xbase+1]*yf[ybase+1]
          imag += xf[xbase]*yf[ybase+1] + xf[xbase+1]*yf[ybase]
  块内树形归约（real/imag 分开）→ partialOut[2*blockIdx], partialOut[2*blockIdx+1]
```

**reduce kernel**：1 个 block 的 AIV，将 workspace 中 `2*useCoreNum` 个 float
（各核 real/imag 部分和）复制到 UB，分别 `ReduceSum` 求 totalReal / totalImag，
写回 `result[0]` / `result[1]`。

##### 3.2.2.2 Ascend C 实现流程图

```
        aclblasCdotu(handle, n, x, incx, y, incy, result)
                        │
        ┌───────────────▼───────────────┐
        │ handle==nullptr ?            │──是──► NOT_INITIALIZED
        │ n<0 ?                        │──是──► INVALID_VALUE
        │ incx==0 或 incy==0 ?         │──是──► INVALID_VALUE
        │ n>0 && (x||y||result)==null  │──是──► INVALID_VALUE
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────▼───────────────┐
        │ n == 0 ?                      │──是──► 不启动 kernel
        └───────────────┬───────────────┘        aclrtMemcpy(result, (0,0)) → SUCCESS
                        ▼
            tiling：useCoreNum/calNum/startOffset
                        ▼
        ┌───────────────▼───────────────┐
        │ incx==1 && incy==1 ?          │
        ├── 是 ──► cdotu_aiv_kernel     │  Gather 分离实/虚部→复数乘→ReduceSum→原子累加
        └── 否 ──► cdotu_simt_kernel    │  每核: SIMT 复数乘加→树归约→workspace(real/imag)
                  + cdotu_reduce_kernel │  单核: Σ workspace → result[0]/result[1]
                        ▼
                    同步 stream，返回 SUCCESS
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| 差异点 | 标杆算子（Netlib cdotu） | Ascend C 实现 | 原因 |
|--------|---------------------------|---------------|------|
| 复数乘法展开 | 复数类型直接乘 | 实/虚部分离为两组 float，`(ac−bd)+(ad+bc)i` | NPU 向量指令不支持复数类型，展开为标量/向量乘加 |
| 归约顺序 | 顺序累加 | 分核 + 块内树归约 + 原子累加 | 多核并行 + ReduceSum 硬件树归约；任务书不要求归约顺序逐位一致，按精度阈值判定 |
| 数据分离 | 复数逐项访问实/虚部 | 连续路径用 Gather(byte-offset 表) 一次分离整 chunk | 向量化分离，提升带宽与计算效率 |
| 执行方式 | 单线程标量循环 | 多核 AIV/SIMT kernel 直调 | NPU 并行架构，多核 + double buffer 流水 |
| 负步长 | CPU 循环反向索引 | SIMT 按 `2*(n-1-idx)*\|inc\|` 浮点索引 | 语义一致，设备端 SIMT 线程并行遍历 |
| n=0 no-op | CPU 分支直接置 0 | host 侧分支，不启动 kernel，H2D 写 (0,0) | 避免无意义 kernel 启动，语义一致 |
| 溢出行为 | float32 累加溢出 | 部分和累加同样 float32，溢出语义一致 | 按 IEEE 浮点语义自然传播 |

### 3.3 支持硬件

| 芯片版本 | 支持 |
|----------|:---:|
| Atlas A2/A3（ascend910b*） | ✗（本 arch35 实现） |
| Ascend 950PR（arch35 / DAV_3510） | ✅ |

与任务书 §3.1 要求的适配硬件（Ascend 950PR）保持一致。

### 3.4 算子约束限制

- 输入/输出数据类型仅支持 COMPLEX64（`aclblasComplex`），实部/虚部各 float32。
- 支持任意非 0 步长（含负步长）；`incx == 0` 或 `incy == 0` 返回 INVALID_VALUE。
- n ≥ 0；n = 0 no-op 返回成功 result=(0,0)；n < 0 返回 INVALID_VALUE。
- 无 broadcast / 原地更新 / 动态 shape 需求。
- 大 n 下误差随累加累积，matched_ratio ≥ 0.99 允许少量离群点。

---

## 四、特性交叉分析

| 特性 | 分析 |
|------|------|
| 精度与性能 | AIV 路径用 Gather 分离 + ReduceSum 树归约，精度远优于阈值；连续路径带宽/算力轻，预期低于任务书标杆 |
| 多核扩展 | 分核策略按复数粒度均分，核数与 chunk 大小动态适配，大 n 下线性扩展 |
| 双路径分支 | incx=incy=1 走 AIV（性能），否则走 SIMT（正确性，含负步长），互斥无冲突 |
| 复用性 | 接口声明入公共头供其他产品线复用；Gather 分离技巧与 gemm_batched 同构；Sdot 结构同构 |

---

## 五、可维可测分析

### 5.1 精度标准 / 性能标准

**精度标准**（任务书 §3.2，生态算子开源精度标准 COMPLEX64 实部/虚部分别按 FLOAT32 判定）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| COMPLEX64（实部/虚部按 FLOAT32 分量） | 2⁻¹⁰ (9.77e-4) | 2⁻¹⁶ (1.53e-5) | 0.99 | 1e-2 或 32×ULP |

通过条件：实部/虚部各自满足 `|actual - golden| ≤ atol + rtol × |golden|`，且
matched_ratio ≥ 0.99 且 max_abs_error ≤ max_abs_error_limit；golden 由 cblas
（Netlib cdotu）语义生成，实部、虚部分别比对。

**性能标准**（任务书 §3.3，COMPLEX64 输入，warmup 后有效采样 >50 次取平均）：

| case | n | incx | incy | 标杆耗时（Avg，us） |
|------|-----|------|------|---------------------|
| 1 | 1048576 | 1 | 1 | 4.05 |
| 2 | 4194304 | 1 | 1 | 4.84 |
| 3 | 16777216 | 1 | 1 | 11.98 |

**自测**：GTest 参数化测试（1200 CSV 用例，覆盖 L0 基础/SQ 尺寸/INC 步长/FL 填充/ED
边界/EX 扩展/PF 性能），精度 golden 用 double 精度累加（近精确，实/虚部分开累加）；
`verify_performance.py` 采集性能并对比标杆。

### 5.2 兼容性分析

- 新算子，无历史版本兼容问题；与仓内 arch22 版 `cdotu/cdotc` 共用同一接口声明，
  本实现补齐 arch22 未支持的任意非零步长能力。
- 接口声明放入公共头 `include/cann_ops_blas.h`，与其他产品线共用同一 `aclblasCdotu`
  API，禁止 950PR 私有平行接口，保证后续产品线扩展兼容。
- 仓内 README 产品支持表标注 Ascend 950PR：支持。
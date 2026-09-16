# aclblasCgbmv 算子设计文档（Ascend 950PR）

| 项 | 内容 |
| --- | --- |
| 任务 | 算子实操工坊-杭州站-aclblasCgbmv 算子开发（950） |
| 算子 | `aclblasCgbmv`：单精度复数（complex64）带状矩阵-向量乘 |
| 目标硬件 | Ascend 950PR（arch35，`--soc=ascend950`） |
| CANN | 9.1.0 |
| 目标仓 | ops-blas：实现 `blas/gbmv/arch35/`，测试 `test/gbmv/cgbmv/arch35/`，接口 `include/cann_ops_blas.h` |
| 编程模型 | Ascend C kernel 直调，SIMT（`__simt_vf__` + `asc_vf_call`），handle 绑定 stream |
| 验证依据 | 2026-09-16 于 Ascend 950PR 实测：精度 2083/2087（保留四条失败），性能形状正确性 200/200，性能 200/200；详见 5.1 |

本文描述本次提交的实现与测试设计。四条 Golden 争议用例保留原始判定，FP64 诊断不将其改判为通过；测试结果及限制见 5.1。

---

# 需求背景（required）

## 1.1 需求来源

通过社区任务完成开源仓算子贡献：在 Ascend 950PR 上用 Ascend C 开发 `aclblasCgbmv`，功能与参数语义对标 cuBLAS `cublasCgbmv`（数学语义参考 Netlib BLAS `cgbmv`），完成设计、开发、测试全流程，验收通过后合入 ops-blas 仓。

## 1.2 背景介绍

### 1.2.1 aclblasCgbmv 算子实现优化

本算子为 ops-blas 新增的 BLAS 直调接口，**不存在 TBE 版本，也没有算子信息库（ops-info）**，不经过 aclnn/图模式注册。设计的标杆与参考来源如下：

| 类别 | 来源（含文件名） | 用途 |
| --- | --- | --- |
| 功能标杆 | cuBLAS `cublasCgbmv`（https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gbmv ） | 接口签名、参数顺序、错误语义 |
| 语义参考 | Netlib BLAS `cgbmv.f`（https://www.netlib.org/blas/cgbmv.f ） | 带状存储寻址、三种 op、负步长、quick return、beta 处理 |
| 精度 golden | Netlib 参考 BLAS `cblas_cgbmv`（本次真机环境：Ubuntu `libblas-dev` 3.10.0） | 测试工程生成 FP32 golden；`cblas_zgbmv` 仅用于 FP64 诊断 |
| 仓内同族实现 | `ops-blas/blas/gbmv/arch35/sgbmv_host.cpp`、`sgbmv_kernel.cpp`、`sgbmv_tiling_data.h` | host 校验与 SIMT 启动骨架 |
| 仓内复数惯例 | `ops-blas/include/cann_ops_blas_common.h`（`aclblasComplex`）、`blas/geam/arch35/cgeam_host.cpp` | complex64 交织存储与 alpha/beta 判零 |
| 仓内标量指针惯例 | `ops-blas/blas/common/helper/host_utils.h`（`CheckPtrLocation`）、`blas/rot/arch35/srot_host.cpp`、`srot_kernel.cpp` | alpha/beta 位于 Host 或 Device 的判定与传递方式 |
| 接口声明 | `ops-blas/include/cann_ops_blas.h`（本次新增 `aclblasCgbmv`） | 对外接口 |

### 1.2.2 aclblasCgbmv 算子现状分析

本任务开发基线中，`blas/gbmv/arch35/` 已有实数 `aclblasSgbmv`（SIMT，每线程一个输出元素），尚无对应的 `aclblasCgbmv` 接口与测试。本次在同一算子目录新增 complex64 的 host、kernel、tiling 与接口声明，并新增 `test/gbmv/cgbmv/` 测试工程；实现范围为 Ascend 950PR，不新增 arch22（A2/A3）实现。

#### 1.2.2.1 标杆算子支持的数据类型和数据格式

| 项 | cuBLAS `cublas<t>gbmv` | 本任务 `aclblasCgbmv` |
| --- | --- | --- |
| 数据类型 | S / D / C / Z 四个接口 | 仅 C（complex64，实部/虚部各 FP32） |
| A 格式 | 列主序带状存储，维度 lda×n，元素 A(i,j)（1-based）位于 `A(ku+1+i-j, j)` | 相同 |
| x / y 格式 | 一维，按 incx/incy 步长，可为负 | 相同 |
| alpha / beta | Host 或 Device（由 `cublasSetPointerMode` 指定） | Host 或 Device（ops-blas 无 pointer mode 接口，按仓内惯例由 `aclrtPointerGetAttributes` 自动判定） |
| op(A) | N / T / C | N / T / C |

#### 1.2.2.2 标杆算子实现描述

cuBLAS 为闭源实现，语义以 Netlib `cgbmv.f` 为准。其实现步骤：

1. **参数检查**：TRANS 非 N/T/C（INFO=1）、M<0（2）、N<0（3）、KL<0（4）、KU<0（5）、LDA<KL+KU+1（8）、INCX=0（10）、INCY=0（13），出错调用 XERBLA 返回；
2. **quick return**：`M=0` 或 `N=0` 或 `(ALPHA=0 且 BETA=1)` 时直接返回；
3. **确定向量长度与起点**：N 时 LENX=N、LENY=M，否则相反；负步长时起点 `KX = 1-(LENX-1)*INCX`（KY 同理）；
4. **先形成 y := beta·y**：BETA≠1 时，BETA=0 直接置零（不读旧 y），否则逐元素乘 BETA；BETA=1 时不做任何运算；
5. **ALPHA=0 时返回**；
6. **累加 alpha·op(A)·x**（`KUP1 = KU+1`）：
   - N：对每列 j，`TEMP = ALPHA·X(j)`，对带内行 `i ∈ [max(1,j-KU), min(M,j+KL)]` 做 `Y(i) += TEMP·A(KUP1-j+i, j)`；
   - T：对每列 j，`TEMP = Σ_i A(KUP1-j+i, j)·X(i)`，然后 `Y(j) += ALPHA·TEMP`；
   - C：同 T，但使用 `CONJG(A(KUP1-j+i, j))`。

Netlib 为单线程串行实现：N 按列外积形式（axpy）累加到 y，T/C 按列内积形式（dot）逐个产出 y。

#### 1.2.2.3 标杆算子实现流程图

```text
cgbmv(TRANS, M, N, KL, KU, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
 │
 ├─ 参数检查（TRANS/M/N/KL/KU/LDA/INCX/INCY）── 出错 ──► XERBLA，返回
 │
 ├─ M==0 ∨ N==0 ∨ (ALPHA==0 ∧ BETA==1) ─────────────► 返回
 │
 ├─ LENX/LENY、KX/KY（负步长从高地址端开始）
 │
 ├─ BETA≠1 ?
 │     ├─ BETA==0 ──► Y(1:LENY) = 0（不读旧 y）
 │     └─ 否则   ──► Y(1:LENY) = BETA·Y
 │
 ├─ ALPHA==0 ────────────────────────────────────────► 返回
 │
 └─ TRANS
       ├─ 'N'：for j = 1..N（串行）
       │        TEMP = ALPHA·X(j)
       │        for i = max(1,j-KU)..min(M,j+KL):  Y(i) += TEMP·A(KU+1-j+i, j)
       │
       └─ 'T' / 'C'：for j = 1..N（串行）
                TEMP = 0
                for i = max(1,j-KU)..min(M,j+KL):  TEMP += op(A(KU+1-j+i, j))·X(i)
                Y(j) += ALPHA·TEMP                  （op = 恒等 / 共轭）
```

## 1.3 算子原型

| 名称 | 类别 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| handle | 输入 | - | - | - | ops-blas 句柄，携带 stream，Host |
| trans | 属性 | aclblasOperation_t | - | - | N / T / C |
| m, n | 属性 | int | - | - | A 的行数、列数 |
| kl, ku | 属性 | int | - | - | 下带宽、上带宽 |
| alpha | 输入 | COMPLEX64 | scalar | [1] | Host 或 Device |
| A | 输入 | COMPLEX64 | ND（列主序带状） | [lda, n] | Device，只读 |
| lda | 属性 | int | - | - | A 的前导维 |
| x | 输入 | COMPLEX64 | ND（步长 incx） | [1+(lenx-1)·\|incx\|] | Device，只读；lenx = N 时 n，否则 m |
| incx | 属性 | int | - | - | x 步长，可为负 |
| beta | 输入 | COMPLEX64 | scalar | [1] | Host 或 Device |
| y | 输入/输出 | COMPLEX64 | ND（步长 incy） | [1+(leny-1)·\|incy\|] | Device，原地覆写；leny = N 时 m，否则 n |
| incy | 属性 | int | - | - | y 步长，可为负 |

---

# 需求分析（required）

## 2.1 外部组件依赖

| 依赖 | 说明 |
| --- | --- |
| CANN 9.1.0 toolkit | Ascend C 编译器 bisheng、`kernel_operator.h`、`simt_api/asc_simt.h`、ACL runtime |
| ops-blas 工程 | CMake 构建、`build.sh`、handle/stream 基础设施、日志 `log/log.h` |
| Netlib 参考 BLAS（仅测试） | `cblas_cgbmv` 生成 golden；`libblas-dev`/`liblapack-dev` |
| googletest（仅测试） | CSV 驱动 GTest |

不引入新的第三方运行时依赖。

## 2.2 内部适配模块

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 接口声明 | `include/cann_ops_blas.h` | 新增 `aclblasCgbmv`，与其他产品线共用，不定义 950PR 私有接口 |
| Host 实现 | `blas/gbmv/arch35/cgbmv_host.cpp` | 参数校验、quick return、alpha/beta 位置判定、tiling 与启动形状计算、kernel 下发 |
| Kernel 实现 | `blas/gbmv/arch35/cgbmv_kernel.cpp` | `__global__` 调度 + 4 个 `__simt_vf__` 计算函数（SCALE、N 长段、N 短段、T/C） |
| Kernel 声明 | `blas/gbmv/arch35/cgbmv_kernel.h` | `cgbmv_kernel_do` 签名，host 通过该头文件引用（不写 extern） |
| Tiling 结构 | `blas/gbmv/arch35/cgbmv_tiling_data.h` | host/kernel 共享 POD |
| 仓内公共件 | `blas/common/helper/host_utils.h`（`CheckPtrLocation`、`GetAivCoreCount`、`CeilDiv`、`CeilAlign`）、`aclblas_handle_internal.h`、`kernel_constant.h` | 复用 |
| 测试工程 | `test/gbmv/cgbmv/`（param/golden/CMake + `arch35/` 下 wrapper/test/benchmark/csv） | CSV 驱动精度 ST 与性能程序 |
| 文档 | `blas/gbmv/README.md` | 新增 `aclblasCgbmv` 章节，产品支持表标注 Ascend 950PR 支持 |

`blas/CMakeLists.txt` 按 `SOC_ARCH_DIRS` 自动收集 `blas/*/arch35/*.cpp`，`test/` 按目录自动发现，无需修改构建脚本。

## 2.3 需求模块设计

### 2.3.1 Ascend C 算子原型

```cpp
aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

参数顺序、类型与 `cublasCgbmv` 一一对应（`cuComplex` → `aclblasComplex`，维数为 `int`），与仓内 `aclblasSgbmv` 参数序列一致。

| 返回值 | 条件 |
| --- | --- |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 为 nullptr |
| `ACLBLAS_STATUS_INVALID_ENUM` | trans ∉ {N, T, C} |
| `ACLBLAS_STATUS_INVALID_VALUE` | m<0、n<0、kl<0、ku<0、lda<kl+ku+1、incx=0、incy=0、alpha/beta 为 nullptr；m>0 且 n>0 时 kl>m-1、ku>n-1、A/x/y 为 nullptr；alpha/beta 指针位置查询失败 |
| `ACLBLAS_STATUS_EXECUTION_FAILED` | 获取 AIV 核数失败 |
| `ACLBLAS_STATUS_SUCCESS` | 其余情况（含 m=0/n=0 no-op 与 quick return） |

### 2.3.2 Ascend C 算子相关约束

与标杆相比的差异（均为任务书范围内的取舍，不影响任务要求的功能）：

| 项 | 标杆 | 本实现 | 原因 |
| --- | --- | --- | --- |
| 数据类型 | S/D/C/Z | 仅 C | 任务书只要求 complex64 |
| 标量指针模式 | `cublasSetPointerMode` 显式设置 | 按指针实际位置自动判定，alpha、beta 可各自位于 Host 或 Device | ops-blas 无 pointer mode 接口，沿用仓内 `srot`/`axpy_ex` 惯例 |
| kl/ku 上界 | Netlib 不检查 kl≤m-1、ku≤n-1 | 越界返回 `INVALID_VALUE` | 任务书 §2.4 明确要求 |
| 支持产品 | GPU | 仅 Ascend 950PR；Atlas A2/A3 不支持 | 任务范围为 arch35 |

---

# 需求详细设计（required）

## 3.1 调用方式

| 上层框架 | 涉及勾选 |
| --- | --- |
| TF 训练/推理 | |
| PyTorch 训练/推理 | |
| ATC 推理 | |
| aclnn 两段式接口 | |
| **ops-blas 句柄式 BLAS 接口（Kernel 直调）** | **√** |

调用流程：`aclblasCreate` → `aclblasSetStream` → `aclblasCgbmv`（在 handle 绑定的 stream 上异步下发 `cgbmv_kernel<<<numBlocks, nullptr, stream>>>`）→ 调用方 `aclrtSynchronizeStream` 后读取 y。算子不使用 workspace，也不申请临时 Device/Host 内存。

## 3.2 需求总体设计

整体结构：

```text
aclblasCgbmv（host）
 ├─ handle 检查
 ├─ ValidateCgbmvParams：参数校验（顺序见 3.2.1.1）
 ├─ m==0 ∨ n==0 ───────────────────────────────► SUCCESS（不启 kernel）
 ├─ ResolveScalar(alpha)、ResolveScalar(beta)：Host → 取值；Device → 透传 GM 指针
 ├─ alpha、beta 均在 Host 且 alpha==(0,0) ∧ beta==(1,0) ─► SUCCESS（不启 kernel）
 ├─ GetAivCoreCount() == 0 ───────────────────────► EXECUTION_FAILED
 ├─ CalCgbmvTilingData + CalcLaunchShape
 └─ cgbmv_kernel_do(A, x, y, alphaGm, betaGm, numBlocks, tiling, stream)（异步）

cgbmv_kernel（device，__global__，标量单元）
 ├─ alpha/beta：Device 标量从 GM 读取并据此重新确定 mode / betaMode；Host 标量沿用 tiling 中的值与 mode / betaMode
 ├─ mode==SCALE ∧ betaMode==1（alpha==0 ∧ beta==1，只在有 Device 标量时可能到达）→ 直接返回
 ├─ blk = GetBlockIdx()，nblk = GetBlockNum()
 └─ asc_vf_call 分发：
       mode=SCALE        ─► CgbmvScaleSimt        （y = beta·y）
       trans=N           ─► CgbmvNSimt / CgbmvNShortSimt（带内行分块 + 列窗口切段 + UB 归约；StoreNResult 检查非有限归约结果）
       trans=T / trans=C ─► CgbmvTSimt<false/true>（每列一个线程组 + shuffle 归约）
```

### 3.2.1 host 侧设计

#### 3.2.1.1 参数校验与快速返回

校验顺序（与任务书 §2.4、§2.5 一致）：

```text
1. handle == nullptr                          → HANDLE_IS_NULLPTR
2. trans ∉ {N,T,C}                            → INVALID_ENUM
3. m < 0 ∨ n < 0                              → INVALID_VALUE
4. kl < 0 ∨ ku < 0                            → INVALID_VALUE
5. lda < kl+ku+1（int64 比较，防溢出）         → INVALID_VALUE
6. incx == 0 ∨ incy == 0                      → INVALID_VALUE
7. alpha == nullptr ∨ beta == nullptr         → INVALID_VALUE
8. m == 0 ∨ n == 0                            → SUCCESS（合法 no-op，不访问 A/x/y）
9. kl > m-1 ∨ ku > n-1                        → INVALID_VALUE
10. A / x / y == nullptr                      → INVALID_VALUE
11. alpha/beta 指针位置查询（aclrtPointerGetAttributes）失败 → INVALID_VALUE
12. alpha/beta 均在 Host 且 alpha==0 ∧ beta==1 → SUCCESS（quick return，y 不变）
13. GetAivCoreCount() 返回 0                   → EXECUTION_FAILED
```

第 8 步必须在第 9 步之前：任务书把 m=0（或 n=0）列为合法 no-op，而此时 `kl ≤ m-1` 无法成立。第 10 步要求 m、n>0 时 A/x/y 非空；这一接口校验先于标量 quick return，不能由 alpha/beta 的取值绕过。alpha=0 且 beta=1 时校验通过后不写 y。

#### 3.2.1.2 alpha/beta 的 Host/Device 位置

任务书规定 alpha、beta 可位于 Host 或 Device。按仓内 `srot` 的做法，两个标量**各自独立**判定：

| 位置 | host 侧处理 | kernel 侧取值 |
| --- | --- | --- |
| Host | 解引用一次，值写入 tiling（`alphaR/alphaI`、`betaR/betaI`），`alphaIsDevice=0` | 使用 tiling 中的值 |
| Device | 不解引用、不做 D2H 拷贝、不同步 stream；GM 指针原样作为 kernel 参数，tiling 中填 0 占位，`alphaIsDevice=1` | `__global__` 入口在标量单元读取 GM 上的 8 字节 |

位置由 `CheckPtrLocation`（`aclrtPointerGetAttributes`）查询，查询失败返回 `INVALID_VALUE`。

标量在 Device 上时，host 无法得知其取值，因此相应的 quick return、SCALE 模式和 betaMode 判定下沉到 kernel 入口，规则与 host 完全相同。alpha 在 Device 时启动形状按一般模式（GBMV）计算；仅 beta 在 Device 时由 host 上的 alpha 决定 GBMV 或 SCALE，betaMode 以 GENERAL 占位。若 kernel 判定为 SCALE，SCALE 分支使用 grid-stride 循环，与启动形状无关；若判定为 quick return（alpha=0 ∧ beta=1），kernel 直接返回，y 不被写入。

#### 3.2.1.3 分核策略

运行时 AIV 核数由 `GetAivCoreCount()` 获取（R2：禁止硬编码）。三条路径的分核：

| 路径 | 并行单元 | block 数 | 每 block 线程数 |
| --- | --- | --- | --- |
| trans=N | 行 `[0, min(m, n+kl))` 按 32 行分块（tile），一个 block 处理 tile `blk, blk+nblk, …`；行号 ≥ n+kl 的行不含带内元素，只做 y=beta·y（长段 VF 从 n+kl 起、短段 VF 从 tiles·32 起交给 grid-stride 的 `ScaleRows`） | `min(max(tiles, ⌈余下行数/线程数⌉), aivCoreNum)`（至少 1），tiles=`⌈min(m,n+kl)/32⌉`，余下行数=`m − min(m,n+kl)`，beta=1 时余下行不计 | `32 × slices` |
| trans=T/C | 一列（一个 y 元素）由 `group` 个 lane 协作 | `min(⌈n·group/128⌉, aivCoreNum)` | `min(CeilAlign(⌈n·group/numBlocks⌉, 128), 2048)` |
| SCALE（alpha=0） | 一个 y 元素一个线程，grid-stride | 同上（group=1，按 y 长度） | 同上 |

参数计算公式：

```text
L       = kl + ku + 1                      （带状列的最大有效长度）
window  = min(kl + ku + 32, n)             （一个 32 行分块实际需要访问的列数）
slices  = 从 4 开始，在 (slices·2)·4 ≤ window 时翻倍，上限 32（取值 4/8/16/32）
          → window ≥ 16 时每段约 ≥4 列；window < 16 时仍为 4 段（每段可少于 4 列）；block 线程数 128～1024
chunk   = ⌈window / slices⌉                （host 据此选择循环；kernel 中每个 tile 按自身实际窗口分段，边缘 tile 的段更短）
nStepped = (chunk ≥ 9)                     （长段用步进循环，短段用逐列寻址循环；slices < 32 时 chunk ≤ 8，
                                             因此只有 slices=32、window ≥ 257 时才走长段）
group   = min(32, 2^⌈log2 L⌉)              （T/C 每列协作 lane 数，L≤32 时恰好覆盖整列）
```

- N：同一 tile 的列窗口 `[r0-kl, r0+31+ku] ∩ [0,n)` 平均分成 `slices` 段，每个 warp 负责一段，lane `l` 负责行 `r0+l`。例如运行时有 56 个 AIV 核时，任务性能用例 N、1024×1024、kl=ku=32 使用 32 个 block × 512 线程；N、2048×2048、kl=ku=64 使用 56 个 block，每个 block 处理 1～2 个 tile。m≫n 时只有前 `⌈(n+kl)/32⌉` 个 tile 含带内元素（如 3668×46、kl=70 只有 4 个），行号 ≥ n+kl 的行不含带内元素，只得 beta·y、不经过 alpha：长段 VF 从 n+kl 起交给 grid-stride 的 `ScaleRows`；短段 VF 中最后一个 tile 里超出 n+kl 的行随 tile 写回（累加为 0，alpha 只乘带内项），其后的行交给 `ScaleRows`。`ScaleRows` 处理的行不再逐 tile 做块内同步，且 beta=1 时不读不写；短段最后一个 tile 内的空行仍经过 `StoreY`。
- T/C：一列的带内行 `r ∈ [max(0,ku-j), min(L, m+ku-j))` 由 `group` 个 lane 以步长 `group` 分担，组内 `asc_shfl_down` 归约。一个 warp 同时处理 `32/group` 列，warp 以 `(总warp数)·(32/group)` 为步长 grid-stride 遍历列。

所有核间无数据依赖、无跨核归约、无原子操作：每个 y 元素只由一个 block 中的一个线程写入。

#### 3.2.1.4 数据分块和内存优化策略

带状乘加的算术强度较低，设计以合并 GM 访问为主：普通路径每个带内 A 元素只读一次；N/T/C 触发异常重算时，仅对该输出行/列重读并计算。实际耗时也受寻址、复数乘法、同步与小尺寸启动开销影响。

**访存模式**（每列存储起点为 `A + lda·j`，带状存储区域长 L；有效行范围按矩阵边界裁剪）：

| 路径 | 同一时刻一个 warp 的访问 | 合并情况 |
| --- | --- | --- |
| N | 活跃 lane 读同一列 j 的相邻带内行 `A[(ku+r0+l-j) + lda·j]`，l=0..31 | 满 32 个有效 lane 时覆盖连续 256 字节；带边界仅有效 lane 访问 |
| T/C | `group` 个 lane 读同一带状列的相邻元素 `A[r + lda·j]` | 连续，合并 |
| x | N：同一 warp 读同一个逻辑 x_j；T/C：读相邻逻辑 x | N 可广播；T/C 在 \|incx\|=1 时地址连续，其他步长为跨步访问 |
| y | N：32 个 lane 写相邻逻辑 y；T/C：lane 0 写一个 y | N 在 \|incy\|=1 时地址连续，其他步长为跨步访问；T/C 单点写回 |

**读写量**：普通路径每个带内 A 元素恰好读一次，输出 y 在 beta≠0 时读一次并写一次，beta=0 不读旧 y。N 异常行或 T/C 异常列在第一次写回前重算，会额外读取其带内 A 与对应 x，仍只写回一次；SCALE/空行的 beta=1 分支可跳过读写。带外三角区与 lda 填充行不访问。若忽略 x 的重复加载与缓存影响，普通路径的数据量约为 `8·(带内元素数 + lenx + 2·leny)` 字节；实际 GM 流量取决于 x 缓存复用、步长和异常输出数量，不能由该估算直接保证。

**UB 使用**：

| 缓冲 | 位置 | 大小 | 用途 |
| --- | --- | --- | --- |
| `partR[1024]`、`partI[1024]` | N 路径 VF 内 `__ubuf__` 数组（上限 32 段 × 32 行） | 2 × 1024 × 4 B = 8 KB | N 路径各段部分和的块内归约 |
| 其余 | 寄存器 | - | 累加器、复数中间值 |

UB 占用 8 KB，远小于 248 KB，不需要 double buffer 或分 tile 搬运（SIMT 直接访问 GM，由硬件数据缓存承接）。workspace 为 0。

**为何不采用 SIMD（MTE 搬运 + 向量计算）**：带状矩阵的有效数据是「连续列 + 列间错位一行」的结构。N 需要沿反对角线访问，T/C 需要按列归约后逐列错位对齐 x；向量化需要 Gather 或非对齐加载加逐列归约，且带边缘列长度各不相同，不能读带外元素。SIMT 可以按元素精确控制带边界，并用 warp 内合并访问取得连续带宽；实测性能已满足全部标杆（见 5.1），因此选用 SIMT。

#### 3.2.1.5 tilingKey 规划策略

本算子不使用 tilingKey 模板实例化：kernel 为直调方式，分支由 tiling 字段在运行时选择，编译期特化只有 C 与 T 的共轭开关。

| 条件 | 选择 | 字段 |
| --- | --- | --- |
| alpha==(0,0) | SCALE 路径 | `mode = CGBMV_MODE_SCALE` |
| alpha≠0，trans=N | N 路径（长段步进循环 / 短段逐列寻址循环） | `trans = 0`，`slices`，`nStepped` |
| alpha≠0，trans=T | `CgbmvTSimt<false>` | `trans = 1`，`groupSize` |
| alpha≠0，trans=C | `CgbmvTSimt<true>` | `trans = 2`，`groupSize` |
| beta==(0,0) / (1,0) / 其他 | 不读 y / y 直接相加 / 一般复数乘 | `betaMode = 0 / 1 / 2` |
| alpha 或 beta 在 Device | kernel 入口从 GM 取值并重新判定上述 mode | `alphaIsDevice`、`betaIsDevice` |

tiling 结构（host/kernel 共享 POD，按值传入 kernel）：

```cpp
struct CgbmvTilingData {
    uint32_t m, n, kl, ku, lda;
    uint32_t trans;          // 0=N, 1=T, 2=C
    uint32_t mode;           // 0=GBMV, 1=SCALE（alpha==0）
    uint32_t betaMode;       // 0=beta==0, 1=beta==1, 2=一般
    uint32_t numThreads;     // 每 block 线程数
    uint32_t groupSize;      // T/C：每列协作 lane 数
    uint32_t slices;         // N：每个 32 行分块的列窗口段数
    uint32_t nStepped;       // N：1=步进循环（每段 ≥9 列），0=逐列寻址循环
    uint32_t alphaIsDevice;  // 1：alpha 位于 Device，kernel 从 GM 读取
    uint32_t betaIsDevice;   // 1：beta 位于 Device，kernel 从 GM 读取
    float alphaR, alphaI, betaR, betaI;   // Host 标量的值；Device 时为 0 占位
    int64_t incx, incy;
};
```

TilingData 不含数组（R4），核间分配由 `GetBlockIdx()/GetBlockNum()` 与上述公式在 kernel 内计算。

### 3.2.2 kernel 侧设计

#### 3.2.2.1 kernel 侧实现描述

**数据视图**：A/x/y 以 `__gm__ float*` 访问，复数 k 的实部在 `2k`、虚部在 `2k+1`。逻辑下标 idx 的物理位置为 `inc ≥ 0 ? idx·inc : (len-1-idx)·(-inc)`（BLAS 负步长语义，调用方传入物理 span 的起始地址）。

**核号**：`__global__` 调度函数取 `GetBlockIdx()/GetBlockNum()`，作为参数显式传入 VF。实测 `__simt_vf__` 内的 `blockIdx.x` 不能区分 AIV 核：多核时各核重复计算第一段、其余输出缺失。因此 VF 内不使用 `blockIdx`/`gridDim`。

**普通路径的复数运算**（实部、虚部各自 FP32 累加；异常重算的乘法规则见下文）：

```text
非共轭：acc += (ar·br − ai·bi,  ar·bi + ai·br)
共轭 C：ar、−ai 代入上式（A 取共轭）
写回：  beta=0 → y = t；beta=1 → y = y + t；否则 y = (βr·yr − βi·yi + tr,  βr·yi + βi·yr + ti)
```

**CgbmvNSimt / CgbmvNShortSimt（trans=N）**：两者分块与归约相同，只有内层循环不同，由 host 按每段列数选择（`nStepped`）：

```text
bandRows = min(m, n+kl)；tiles = ⌈bandRows/32⌉            // 行号 ≥ n+kl 的行不含带内元素
for tile = blk; tile < tiles; tile += nblk:           // 块内所有线程循环次数一致
    r0 = tile·32；row = r0 + lane；slice = threadIdx.x / 32
    [c0, c1) = SliceColumns(...)：本 slice 在列窗口 [max(0,r0-kl), min(n, r0+32+ku)) 中的一段
    acc = 0
    长段（CgbmvNSimt，内层为 BandRowDot，row < m 时执行）：
        d = ku+row-c0；aOff = d + lda·c0；xOff = x 中第 c0 个逻辑元素的位置
        for col in [c0, c1):
            d ≤ kl+ku 时 acc += A[aOff] · (alpha·x[xOff])   // d 为无符号数，一次比较同时排除 d<0 与 d>kl+ku
            d -= 1；aOff += lda-1；xOff += incx             // 列间步进，不做乘法寻址
    短段（CgbmvNShortSimt，循环直接写在 VF 内）：
        for col in [c0, c1):
            跳过 row ≥ m、col < row-kl、col > row+ku      // 只访问带内元素
            t = alpha · x[col]                              // 与 Netlib 相同：先缩放 x
            acc += A[(ku+row-col) + lda·col] · t
    partR/partI[threadIdx.x] = acc
    asc_syncthreads()
    slice 0：按 slice 顺序求和 s（alpha 已乘在各项中）；长段仅 row < bandRows 时写回，短段 row < m 时写回（带外行 s = 0）→ StoreNResult(row, s)
    asc_syncthreads()
ScaleRows(起点 + 全局线程号, 步长 = 总线程数)：长段起点 bandRows，短段起点 tiles·32；这些行只写 beta·y（beta=1 时不读不写）
```

**N 路径的异常结果处理（`StoreNResult`）**：

列窗口切段后，部分和先从 0 累加，最后才加 beta·y。即使逐项从 beta·y 开始计算一直有限，部分和也可能先溢出。例如 `beta·y=-3e38`、两项均为 `2e38`：顺序累加约为 `-1e38 → 1e38`，先求两项之和则超过 FP32 范围。仅保持先乘 alpha 无法避免这一问题。

```text
StoreNResult(row, s):
    若 s.real 和 s.imag 都有限：StoreY(y[row], s)，返回
    acc = beta=0 ? (0,0) : beta=1 ? 原 y[row] : ReferenceComplexMultiply(beta, 原 y[row])
    for col = max(0, row-kl) .. min(n, row+ku+1)-1（列号递增）:
        scaled = ReferenceComplexMultiply(alpha, x[col])
        term = ReferenceComplexMultiply(scaled, A[(ku+row-col) + lda·col])
        acc.real += term.real；acc.imag += term.imag
    y[row] = acc
```

重算由该行的 slice 0 线程完成，从 beta·y 开始，按 Netlib N 路径对该输出的列序累加；不引入新的 kernel、workspace 或跨核同步。beta=0 不读旧 y，beta=1 直接以旧 y 初始化，避免额外复数乘法。异常行重算会增加带内 A/x 读取和串行计算，并可能延长同一 block 的等待时间；普通行保留并行路径，仅增加有限性检查。

`ReferenceComplexMultiply` 先独立计算四个 FP32 乘积。只有实部、虚部同时为 NaN，且操作数或乘积含 Inf 时，才执行参考复数乘法的无穷恢复规则：含 Inf 的操作数归一为带符号的 0/1，另一侧的 NaN 替换为带符号的 0，再用 Inf 缩放重算。例如 `(1+0i)·(Inf+Inf·i)` 得到两个正 Inf。四个中间乘积使用 `volatile`，避免编译器收缩运算改变特殊值传播；有限/Inf/NaN 分类以及带符号 0/1 构造使用 IEEE-754 位模式。

**触发范围**：N 仅检查加 beta·y 之前的归约结果；归约有限时走 `StoreY`。T/C 和一般 beta 的检查点见下文。重算由非有限结果触发，不以与 Golden 的有限误差触发；并行结果有限而 Golden 在另一累加顺序下溢出时，也不会自动改算为非有限。因此不保证任意输入下 bit-exact 或相同溢出行为。

**CgbmvTSimt<CONJ>（trans=T/C）**：

```text
lane = threadIdx.x & (group-1)
for base = 本 warp 的起始列; base < n; base += 总 warp 数·(32/group):   // warp 内循环次数一致
    col = base + (threadIdx.x & 31) / group
    rBegin = max(0, ku-col)；rEnd = min(L, m+ku-col)（col ≥ n 时为 0）
    acc = 0
    for r = rBegin+lane; r < rEnd; r += group:
        acc += op(A[r + lda·col]) · x[col-ku+r]
    组内 asc_shfl_down 归约（off = group/2 … 1）
    lane 0 且 col < n：StoreTResult<CONJ>(col, acc)        // alpha·acc+beta·y 非有限时重算整列
```

shuffle 要求 warp 内 32 个 lane 同时在场：列循环以 warp 为单位推进，`col ≥ n` 的 lane 以空区间参与，保证循环次数一致。

**T/C 路径的异常结果处理（`StoreTResult` / `RecomputeTColumn`）**：

`StoreTResult` 自行形成最终值：先算 `out=alpha·acc`，再按 betaMode 加 beta·y（beta=0 不读旧 y，beta=1 直接相加），对结果只做一次有限性检查。两分量有限时直接写回；任一非有限时调用 `RecomputeTColumn`，沿当前列的有效带状行 r 从小到大，用 `ReferenceComplexMultiply(op(A), x)` 累加 temp，再计算 `ReferenceComplexMultiply(alpha, temp)`，最后加 `ReferenceBetaY` 形成的 beta·y。例如 `[2e38, -2e38, 2e38, -2e38]` 顺序求和为 0，但 shuffle 先加第 0/2、1/3 项时分别溢出为 ±Inf，归约变为 NaN，需要重算。

只有 `RecomputeTColumn<CONJ>` 标为 `noinline`，N 的 `StoreNResult` 和通用 `StoreY` 保持内联。T/C 自行完成写回，避免将通用 beta·y 异常分支内联进 VF。重算仍由原输出的单一线程执行，不增加 kernel、workspace 或跨核通信。

在 Ascend 950PR、CANN 9.1.0 上，对全部 200 条性能用例分别执行两轮交替对照，每轮 warmup 10 次、event 计时 100 次取平均。下表先求每条用例的两轮平均耗时之比，再对 T/C 用例取中位数：

| 单项改动 | 比较基线 | T/C 中位耗时比 | 中位耗时降低 |
| --- | --- | ---: | ---: |
| 仅将 T/C 整列重算改为 noinline | 异常处理全部内联的同功能版本 | 0.895 | 约 10.5% |
| T/C 自行形成并检查最终值，取代调用 StoreY | 已仅将 T/C 整列重算设为 noinline 的版本 | 0.935 | 约 6.46% |

两组对照的基线不同，不将百分比直接相加，也不推断每个用例都更快或寄存器数量发生了何种变化。N 用例中位耗时比分别约为 0.995 和 1.001，基本持平。

仅当该输出的最终值至少一个分量非有限时，才执行整列重算；两个分量均有限时直接写回，不额外读取 A/x。有限性检查本身仍在正常写回路径上。即使 alpha·acc 有限，仅 beta·y 或最后加法非有限，也会触发重算；有限输入亦可能因计算溢出而满足该条件。大量非有限输出可能增加整列 A/x 读取和串行累加，5.1 的普通性能用例不能代表该负载的耗时。

**通用写回（`StoreY` / `ReferenceBetaY`）**：beta=0 直接写 out，不读旧 y；beta=1 直接加旧 y，不做复数乘法；一般 beta 先计算 `res=beta·y+out`。res 任一分量非有限时，用 `ReferenceComplexMultiply(beta, y)` 重新形成 beta·y，再加 out；它只重算 beta·y，不重新归约 A/x。`StoreY` 服务于 N 的正常写回、N 空行（`ScaleRows`）与 SCALE 路径，覆盖 `(2+0i)·(Inf+Inf·i)` 的无穷恢复；T/C 按上文自行写回。N/T/C 完整重算以 `ReferenceBetaY` 统一处理 beta=0/1/一般三种情况。

| 检查位置 | 触发条件 | 重算内容 |
| --- | --- | --- |
| N 的 `StoreNResult` | 分段归约 s 任一分量非有限 | 从 beta·y 开始，按列序重算整个输出行 |
| T/C 的 `StoreTResult` | 最终值 alpha·acc+beta·y 任一分量非有限 | 按有效行序重算列点积，再乘 alpha 并加 beta·y |
| 一般 beta 的 `StoreY` | beta·y+out 任一分量非有限 | 仅用参考复数乘法重算 beta·y，再加 out |

**CgbmvScaleSimt（alpha=0）**：grid-stride 遍历 y，经 `StoreY` 按 betaMode 写 `0` 或 `beta·y`，不读 A、x；beta=1 直接返回。

#### 3.2.2.2 Ascend C 实现流程图

```text
cgbmv_kernel(A, x, y, alphaGm, betaGm, tiling)          [__global__, KERNEL_TYPE_AIV_ONLY]
 │
 ├─ alpha = alphaIsDevice ? *alphaGm : tiling.alpha；Device 时 mode = (alpha==0) ? SCALE : GBMV
 ├─ beta  = betaIsDevice  ? *betaGm  : tiling.beta；Device 时 betaMode = f(beta)
 ├─ mode==SCALE ∧ betaMode==1 ─────────────────────────► return（y 不变）
 ├─ blk = GetBlockIdx(); nblk = GetBlockNum()
 │
 ├─ mode == SCALE ──► asc_vf_call<CgbmvScaleSimt>
 │                      for idx（grid-stride）: y[idx] = (betaMode==0) ? 0 : beta·y[idx]
 │
 ├─ trans == N ─────► nStepped ? asc_vf_call<CgbmvNSimt> : asc_vf_call<CgbmvNShortSimt>（dim3{32·slices}）
 │                      for tile（blk 起步，步长 nblk，只覆盖带内行 [0, min(m, n+kl))）:
 │                        每个 warp：列窗口的一段 → 32 行部分和（合并读 A 列；长段步进寻址，短段逐列寻址）
 │                        UB partR/partI ─ syncthreads ─ slice0 按序求和 s
 │                          ├─ s 两分量有限：StoreY
 │                          └─ 否则：从 beta·y 开始按列序重算该行并写回
 │                        syncthreads
 │                      其余行：ScaleRows（y = beta·y，grid-stride；长段从 n+kl 起，短段从 tiles·32 起）
 │
 └─ trans == T/C ───► asc_vf_call<CgbmvTSimt<CONJ>>(dim3{numThreads})
                        for 列（warp 粒度 grid-stride）:
                          group 个 lane 沿带状列连续读 A、x → 部分和
                          asc_shfl_down 组内归约 ─ lane0: StoreTResult（alpha·acc+beta·y 非有限则重算列）
```

#### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图的差异点和原因

| 差异点 | Netlib 标杆 | Ascend C 实现 | 原因 |
| --- | --- | --- | --- |
| 执行方式 | 单线程，按列串行 | 多核 × SIMT 多线程并行 | 输出元素相互独立，按输出切分可无依赖并行 |
| N 的计算形式 | 按列外积：每列把 `TEMP·A(:,j)` 累加到 y（axpy） | 按行内积：每个线程对一行累加，32 行一组共享列窗口 | 外积形式在并行时需要对 y 原子累加；内积形式每个 y 只有一个写者，并用 32 行同列的布局保持合并访问 |
| N 的列窗口切段 | 无 | 同一行分块的列窗口分给多个 warp，部分和在 UB 中按固定顺序相加 | m 较小时仅按行并行无法占满核，切段增加并行度；固定求和顺序保证结果确定 |
| beta·y 的时机 | 先整体形成 beta·y，再累加 alpha·op(A)·x | 普通路径在写回时合成；N 完整重算从 beta·y 开始；T/C 完整重算先求列点积，再加 beta·y；一般 beta 写回非有限时重算 beta·y | 普通路径减少 y 读写；FP32 舍入和溢出可能受求和次序影响，异常处理的三个检查点见 3.2.2.1 |
| 累加顺序 | N：y 按列 j 递增逐项累加 | N：段内按 j 递增，段间按段序相加；异常行按列序重算 | 切段提高并行度，固定顺序保证同一配置可重复；有限结果仍可能与 Golden 有舍入差异，按 5.1 原判据验证并保留失败 |
| N 的内层循环 | 无 | 每段 ≥9 列用步进循环（带内判断一次无符号比较，A/x 偏移按列步进）；更短的段逐列计算地址 | 长段摊薄步进循环的初始化开销、减少每次迭代的指令数；短段逐列寻址开销更小。按每段列数由 host 选择（`nStepped`） |
| alpha 的作用位置 | N：先乘 x；T/C：先求和后乘 | 相同（N 的长段、短段和异常重算都先乘 x） | 避免先算 A·x 引入的额外溢出，例如 A=1e10、x=1e30、alpha=1e-30；这不消除部分和切段或 beta·y 合成顺序带来的其他数值差异 |
| 非有限结果 | 按参考次序逐项 FP32 复数乘加 | N 检查部分和；T/C 检查最终 alpha·acc+beta·y；一般 beta 的通用 StoreY 检查写回值，分别按 3.2.2.1 重算 | 处理顺序差异导致的部分和溢出及复数 Inf 传播；普通有限结果保留并行路径 |
| beta=1 | 不做乘法 | 不做乘法（betaMode=1 直接相加） | 避免 `(1,0)·(Inf,0)` 产生 NaN，与标杆一致 |
| 标量位置 | pointer mode 显式指定 | 自动判定，Device 标量在 kernel 入口读取并判定 quick return/SCALE | ops-blas 无 pointer mode 接口；避免 D2H 同步，保持异步语义 |
| kl/ku 上界 | 不检查 | 越界报错 | 任务书要求 |

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √（本次真机验证） |
| Ascend 950DT（arch35） | √（与 950PR 同一份 arch35 实现，按仓内 `blas/gbmv/README.md` 的粒度标注；本次真机验证在 950PR 上完成） |
| Atlas A3 训练/推理系列 | ×（不支持） |
| Atlas A2 训练/推理系列 | ×（不支持） |

## 3.4 算子约束限制

1. 仅支持 complex64（`aclblasComplex`）；
2. `m ≥ 0`、`n ≥ 0`；m=0 或 n=0 为合法 no-op；m、n>0 时 `0 ≤ kl ≤ m-1`、`0 ≤ ku ≤ n-1`；
3. `lda ≥ kl+ku+1`；`incx ≠ 0`、`incy ≠ 0`，可为负；
4. alpha、beta 不可为空，可各自位于 Host 或 Device；A、x、y 必须位于 Device；
5. x、y 不得与 A 重叠；y 原地覆写，y 的步长空洞保持不变；
6. 只访问带内元素：带外三角区、lda 填充行、x 的步长空洞都不读；
7. 使用 FP32 运算，不保证与 Golden bit-exact；目标判据与实测结果见 5.1，当前保留四条精度失败。非有限重算的触发范围见 3.2.2.1；
8. 在 handle 绑定的 stream 上异步执行，读取 y 前需同步该 stream；不使用 workspace；
9. 下标在 kernel 内以 int64 计算，`lda·n` 与 `(len-1)·|inc|` 不溢出。

---

# 特性交叉分析

| 特性 | 是否涉及 | 说明 |
| --- | --- | --- |
| 三种 op（N/T/C） | √ | N 与 T/C 走不同 VF；C 通过模板参数取共轭 |
| 非方阵（m≠n） | √ | 列窗口、带状行范围均按 m、n 分别截断 |
| kl、ku 边界（0 与上界） | √ | L=1 时 group=1；kl=m-1/ku=n-1 时带覆盖全矩阵，逻辑不变 |
| lda padding | √ | 只按 `ku+i-j` 寻址，填充行不访问 |
| 正负步长 incx/incy | √ | 逻辑下标到物理下标的统一映射 |
| alpha/beta 特殊值 | √ | alpha=0 → SCALE；alpha=0∧beta=1 → quick return；beta=0 不读 y；beta=1 不乘 |
| alpha/beta 位于 Device | √ | kernel 入口读取并判定 mode |
| Inf/NaN 输入 | √ | N、T/C 与一般 beta 的非有限结果按 3.2.2.1 重算；不含带内元素的 N 行只做 beta·y、不经过 alpha；测试实部/虚部分别要求 NaN 对 NaN、Inf 同符号匹配 |
| 零维（m=0/n=0） | √ | host 直接返回成功，不访问指针 |
| 原地覆写 | √ | 仅 y；每个 y 元素只被一个线程读写 |
| 广播 / 非连续 Tensor / 动态 shape | × | 不涉及（仅 lda/inc 跨步） |
| 多核归约 / 原子 / workspace | × | 不涉及 |
| 确定性 | - | 同一输入多次执行结果相同（固定分段与求和顺序） |

---

# 可维可测分析

## 5.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | Golden 为 Netlib `cblas_cgbmv`；实部/虚部分别按 FLOAT32：rtol=2^-10，atol=2^-16；满足 `\|actual−golden\| ≤ atol + rtol·\|golden\|` 的比例 ≥ 0.99，且每个有限参考分量的绝对误差 ≤ max(1e-2, 32·ULP)。非有限参考要求 NaN 对 NaN、Inf 同符号匹配 | 有限值标准：生态算子开源精度标准、任务书 §3.2；特殊值分类：本测试工程 |
| 性能标准 | 任务书 §3.3 五个用例，warmup 后 >50 次有效采样的平均单次耗时不高于标杆；其余用例对照 `gpu_baseline.csv`，倍率 gpu/npu ≥ 0.4 | 任务书 §3.3、测试说明 |

**真机自验结果（2026-09-16）**：Ascend 950PR，128 GB HBM，驱动 25.7.rc1.6；CANN 9.1.0，x86_64 Ubuntu 22.04.5，GCC 11.4.0，CMake 3.22.1，Netlib BLAS 3.10.0，GTest 1.11.0。`cgbmv_benchmark` 每轮每用例 warmup 10 次、event 计时 100 次取平均，共独立执行两轮；下表保守采用两轮均值中较慢者。msprof 另行采集 kernel Task Duration。

| case | trans | m | n | kl | ku | NPU 平均 (µs) | msprof kernel (µs) | 标杆 (µs) |
|---|---|---|---|---|---|---|---|---|
| TC_PF_1001 | N | 1024 | 1024 | 32 | 32 | 6.508 | 6.728 | 25.58 |
| TC_PF_1002 | N | 2048 | 2048 | 64 | 64 | 15.413 | 16.073 | 34.25 |
| TC_PF_1003 | T | 2048 | 2048 | 64 | 64 | 9.616 | 9.878 | 35.85 |
| TC_PF_1004 | C | 2048 | 2048 | 64 | 64 | 9.580 | 9.820 | 35.88 |
| TC_PF_1005 | N | 4096 | 4096 | 128 | 128 | 27.750 | 28.567 | 52.79 |

- 200 条性能用例全部达标，两轮最低倍率 gpu/npu 分别为 0.642 与 0.649；逐用例取两轮中较慢者后仍为 **0.642**（`TC_PF_1091`，N，4096×4096，kl=ku=512），要求 ≥0.4。五个任务用例的 event 与 msprof 均值也均低于限时。
- 五个 msprof 用例各有 110 条 `cgbmv_kernel` 记录，按下发时间排序后丢弃前 10 条、对其余 100 条求均值；全部为 `AI_VECTOR_CORE`。

| 精度集合 | 执行数 | 通过数 | 失败数 |
| --- | ---: | ---: | ---: |
| 任务 CSV（排除 TC_PF） | 1000 | 1000 | 0 |
| 补充 CSV | 1069 | 1065 | 4 |
| TEST_F（含八组 N/T/C/beta 异常结果回归） | 16 | 16 | 0 |
| 主机判定器 TEST | 2 | 2 | 0 |
| 全量精度合计 | **2087** | **2083** | **4** |
| 性能形状带 Golden 正确性校验（单独执行） | 200 | 200 | 0 |

全量精度 GTest 退出码为 **1**；性能形状正确性退出码为 0。任务与补充 CSV、有限值容差均保持原样；按 XML 检查实际用例集合、重复项和完成状态，不能仅凭日志末尾摘要或筛选后的子集判为全量通过。

**四条 Golden 争议的事实与边界**：失败为 `TC_SUP_NM_EX_0744`、`TC_SUP_NM_EX_0802`、`TC_SUP_NM_EX_0918`、`TC_SUP_NM_EX_0944`。以下列出相关分量中，FP32 Golden 相对同输入 FP64 高精度参考的最大偏差；ULP 统计范围为高精度参考绝对值 ≥1 的元素，容差倍数按 max(1e-2, 32·ULP) 计算。

| 用例 / 分量 | Golden 最大 ULP | Golden 最大容差倍数 |
| --- | ---: | ---: |
| TC_SUP_NM_EX_0744 / real | 31.432383 | 0.982262 |
| TC_SUP_NM_EX_0802 / imag | 31.929917 | 0.997810 |
| TC_SUP_NM_EX_0918 / real | 36.879587 | 1.152487 |
| TC_SUP_NM_EX_0918 / imag | 33.473545 | 1.046048 |
| TC_SUP_NM_EX_0944 / real | 33.872061 | 1.058502 |

这些用例中 NPU 相对 FP64 参考的最大偏差不到 5 ULP；0744/0802 的 Golden 自身误差接近但未超过上限，0918/0944 的部分输出超过上限。这说明长序列 FP32 累加的参考误差需要单独记录，不能据此声称四条 Golden 均已自超限，也不能替代 NPU 与 FP32 Golden 的主判定。四条仍按原判据保留为失败。

**验证方法与证据**：上述数据来自 Ascend 950PR 的完整精度测试、两轮 event 计时及五条独立 msprof 采集，按测试源码核对 XML 的实际用例集合、完成状态和退出码。本节是该日期实现版本的实测快照；验收阶段提交的测试代码与结果以当时实际提交的版本为准，其源码校验值、原始记录与复核结果随自测材料一并给出，若与本节数据不同，以自测材料为准。

**性能计时与完整性**：任务配套 `verify_performance.py` 使用 GTest 单条用例的毫秒级整体耗时，包含造数、H2D/D2H、CPU Golden 和比对，并取整到 ms；它不能代表微秒级 kernel 耗时，0 ms 还会被标为 NO_REF。本设计使用 `cgbmv_benchmark` 的 event 计时，并以独立 msprof 采集交叉验证。

随自测材料提供的 `test_cases/perf_compare.py` 默认要求预期 CSV 的全部 200 条 TC_PF 齐全，并检查任务书五个用例。空日志、格式错误、缺失/重复/额外 case、形状不匹配、无效或非正耗时、缺失基线、warmup≤0、repeats≤50 均判失败。GPU 基线按 trans/m/n/kl/ku/incx/incy 匹配，同一键有多条时使用最快基线。显式 `--case` 可验证子集，但子集结果不能记作全量性能通过。

## 5.2 兼容性分析

- **接口**：新增接口，声明加入 `include/cann_ops_blas.h`（与其他产品线共用），不修改任何已有接口与已有算子行为；参数序列与 `cublasCgbmv` 一致，已有 cuBLAS 用户代码可按类型替换迁移。
- **语义**：错误码复用 `cann_ops_blas_common.h` 定义，不新增私有错误码；quick return、beta=0/1、负步长语义与 Netlib/cuBLAS 一致；kl/ku 上界检查按任务书要求比 Netlib 更严格。
- **标量位置**：alpha/beta 位置自动判定，与仓内 `srot`、`axpy_ex` 行为一致；调用方无需额外设置。
- **产品**：arch35 专有实现，放在 `arch35/` 目录，不影响 arch22 编译与运行；A2/A3 调用 `aclblasCgbmv` 不在支持范围内，README 产品支持表明确标注。

## 5.3 测试设计

测试工程 `test/gbmv/cgbmv/`（结构参考仓内 `test/gbmv/sgbmv`、`test/copy/ccopy`）：

| 文件 | 作用 |
| --- | --- |
| `cgbmv_param.h` | CSV 解析：trans 支持枚举名与整数（构造非法枚举）；`alpha_real`/`beta_real` 为 `null` 时传空指针；填充串支持 `GAUSS_<μ>_<σ>` 正态分布；可选列 `a/x/y_align_offset` |
| `cgbmv_golden.h` | `cblas_cgbmv` golden，参数检查与算子一致 |
| `CMakeLists.txt` | ascend950 下注册 `cgbmv_test`、`cgbmv_benchmark` |
| `arch35/cgbmv_npu_wrapper.h` | 按 lda×n、`1+(len-1)·\|inc\|` 申请 Device 内存并搬运（可选：alpha/beta 放 Device、A/x/y 地址偏移，偏移前缀填 0xFF），同步 handle stream 后回读 y 并校验 y 前缀未被写入 |
| `arch35/cgbmv_test.cpp` | 16 条 `TEST_F`：接口/标量/带外行/alpha 顺序 8 条，N 异常结果 5 条，T/C 特殊值回归 2 组，跨 N/T/SCALE 路径的 beta·y 回归 1 组；另含 2 条主机判定器 `TEST`，以及 CSV 驱动 `TEST_P` |
| `arch35/cgbmv_benchmark.cpp` | 读取 `TC_PF_` 用例，warmup + event 计时平均 |
| `arch35/cgbmv_test.csv` | 任务配套 1000 条精度 + 200 条性能用例（保持原样） |
| `arch35/cgbmv_test_supplement.csv` | 补充用例 1069 条（正态分布、alpha/beta 特殊值、对齐偏移、任务用例的正态分布镜像），由固定种子的生成脚本 `gen_supplement.py` 生成（随测试说明提供），`CgbmvSup` 用例组加载 |

**用例覆盖**（与任务书 §3.5 对应）：L0 基础、尺寸扫描（1～2048）、alpha/beta 组合（0、1、-1、复数、纯虚）、非方阵、lda padding、kl/ku 边界（0、上界、仅上带/仅下带、全带）、incx×incy ∈ {±1,±2,±3} 全组合、填充模式（零、交替、极端值、Inf、NaN）、边界与负向（零维、beta=0 且 y 为 NaN、alpha=0、alpha=0∧beta=1、空指针 alpha/beta/A/x/y、非法 trans/lda/维度/步长/kl/ku）、性能 200 条。补充用例：任务配套 CSV 未覆盖的 §3.5 场景放在独立的 `cgbmv_test_supplement.csv`（1069 条）——正态分布数据（`GAUSS_<μ>_<σ>`，μ∈[-5,5]、σ∈[0.1,2]；18 条 A/x/y 全为正态、18 条逐个按 50% 取正态或均匀；alpha/beta 50% 均匀、50% 正态）36 条；alpha/beta 纯虚数与小模长复数 9 条、大值（模长约 1e2～1.5e2，含 alpha 极小而 beta 大；输出长度 ≥224）21 条，共 30 条；A/x/y 对齐偏移（Device 地址偏移 1～7 个复数元素，非 32B 对齐）18 条；任务配套全部 985 条成功类精度用例的正态分布镜像 985 条（形状、带宽、lda、步长不变，A/x/y 改为正态分布，alpha/beta 50% 均匀、50% 正态，边界与填充类保留原标量），与任务配套 CSV 合起来使数据均匀/正态各约 50%。另有 `TEST_F`：`ScalarPlacementMatchesGolden`（N/T/C × {alpha、beta 均在 Device；仅 alpha 在 Device；仅 beta 在 Device}，incx=-2、incy=3、lda 含填充）、`DeviceScalarQuickReturn`（Device 上 alpha=0∧beta=1，y 逐位不变）、`DeviceAlphaZeroDoesNotReadAX`（Device alpha=0，A/x 全为 NaN）、`DeviceBetaZeroDoesNotReadY`（Device beta=0，旧 y 全为 NaN）、`NRowsBelowBandSkipAlpha`（trans=N、alpha=(Inf,0)、m>n+kl 且 n+kl 不是 32 的倍数，长段 400×300 与短段 120×40 两种形状：不含带内元素的行只得 beta·y，与 golden 一致）、`NLongSliceScalesXByAlphaFirst`（trans=N 长段 301×301、kl=ku=150，A=1e10、x=1e30、alpha=1e-30：须先乘 alpha，结果与 golden 的有限值一致）。

**边界访问检测**：输入中不应参与计算的位置——A 的带外三角区、lda 填充行、x 的步长空洞——填 NaN，通过结果污染发现错误寻址；y 的步长空洞填哨兵值，执行后逐位比对保持不变；对齐偏移用例中 A/x/y 之前的偏移前缀填 0xFF（NaN），执行后校验 y 前缀未被写入。`AlphaZeroDoesNotReadAX` 与 `DeviceAlphaZeroDoesNotReadAX` 用 NaN 填满 A 与 x，检查 alpha=0 的结果不受 A/x 影响；`DeviceBetaZeroDoesNotReadY` 用 NaN 填满旧 y，检查 beta=0 的结果不受旧 y 影响。上述数据校验结合代码中的访问条件核对，不将结果未受污染等同于所有物理读访问都已被检测。

**异常结果与判定器覆盖**：

| 测试名 | 场景与预期 |
| --- | --- |
| `NShortFiniteResultSurvivesOverflow` | 短段，两项 2e38 先求和溢出；从 beta·y=-3e38 开始逐项累加应有限 |
| `NLongFiniteResultSurvivesOverflow` | 512 列，非零项落在不同 slice，incx=-2、incy=-3；重算结果与 Golden 一致 |
| `NFiniteResultSurvivesOverflowWithinSlice` | 512 列，非零项落在同一 slice，incx=2、incy=3；覆盖段内溢出 |
| `NFiniteResultSurvivesOverflowGeneralBeta` | beta=0.5，覆盖一般 beta 初始化后的溢出相消 |
| `NComplexInfinityPropagation` | `(1+0i)·(Inf+Inf·i)`，beta=0 且原 y 为 NaN；输出应为两个正 Inf |
| `TCComplexInfinityPropagation` | T/C 的单 lane 与四 lane 归约，验证 Inf 传播与参考一致 |
| `TCFiniteResultSurvivesOverflow` | T/C 的 shuffle 内和 lane 内溢出相消，含负 incx；重算后与参考一致 |
| `BetaTimesInfiniteY` | 一般 beta 乘 Inf y，覆盖 N 带内/空行、T 与 alpha=0 的 SCALE |
| `NonFiniteClassificationAndSign` | 对实部/虚部分别枚举有限、±Inf、NaN，要求分类与 Inf 符号匹配 |
| `Fp64DiagnosticCountsNonFiniteOutputs` | 可表示参考对应 NaN 输出时计为不匹配；超出 FP32 范围单列；FLT_MAX 的 ULP 有限 |

**判定**：实部、虚部分别按 5.1 标准判定，有限参考使用仓内 `MIXED_TOLERANCE`；Golden 为 NaN 时输出必须为 NaN，Golden 为 Inf 时输出必须为同符号 Inf。两类检查均满足才通过，CSV 中 `mere_threshold` / `mare_multiplier` 不参与判定。

**FP64 诊断**：Golden 本身是 FP32 结果，存在舍入误差。数值校验失败时，使用同一份 FP32 输入转为双精度，通过 `cblas_zgbmv` 计算高精度参考；设置 `CGBMV_FP64_STATS=1` 时，对进入数值校验的 CSV 用例都输出诊断。FP64 参考仍是浮点计算，不称为数学精确值，也不覆盖主测试判定。

诊断分别输出 NPU/Golden 的有限比较数、参考非有限数、参考超出 FP32 范围数、非有限不匹配数，以及 matched 数和可表示参考数。参考有限且可由 FP32 表示、而输出非有限时，增加不匹配数并将该侧最大容差倍数置为 Inf；参考超出 FP32 范围时单独计数，不混入有限误差统计。最大偏差按 max(1e-2, 32·ULP) 折算，ULP 统计仅覆盖参考绝对值 ≥1 的元素；必须结合比较数与跳过数解释，不能将空比较或被跳过元素的零误差解释为通过。四条保留失败的实际数据见 5.1。

**大值 alpha/beta**：补充用例中的大值取模长约 1e2～1.5e2（任务数据正常范围模长在 9 以内），与其余用例一样按 5.1 的标准原样判定。这类用例的输出长度 ≥224：alpha 很大时，个别正负相消到接近 0 的输出元素可能因正常的 FP32 舍入差异不满足 `atol + rtol·|golden|`，matched_ratio ≥ 0.99 正是为这类个别元素留的余量，向量足够长时才能起作用。

## 5.4 可测性与定位手段

- host 侧 `OP_LOGD` 输出完整 tiling（m/n/kl/ku/lda/trans/mode/betaMode/group/slices/stepped/threads/alphaIsDevice/betaIsDevice/incx/incy），`OP_LOGI` 输出 block 数与 AIV 核数；
- 性能以 `cgbmv_benchmark`（可用 `CGBMV_BENCHMARK_CASE`、`CGBMV_BENCH_WARMUP`、`CGBMV_BENCH_REPEATS` 选择用例与次数）与 msprof 交叉验证；
- 无卡环境可用 CANN toolkit 自带 cannsim（`cannsim record ... -s Ascend950`）做功能仿真与 kernel 周期估算。

---

# 交付与合入

| 交付物 | 位置 |
| --- | --- |
| 接口声明 | `include/cann_ops_blas.h`；`docs/zh/api_list.md` 接口列表新增 `aclblasCgbmv` |
| 算子实现 | `blas/gbmv/arch35/cgbmv_host.cpp`、`cgbmv_kernel.cpp`、`cgbmv_kernel.h`、`cgbmv_tiling_data.h` |
| 算子文档 | `blas/gbmv/README.md`（新增 `aclblasCgbmv` 章节：产品支持、原型、参数、约束、RAII 调用示例） |
| 测试代码 | `test/gbmv/cgbmv/`（含任务配套 `arch35/cgbmv_test.csv` 与补充 `arch35/cgbmv_test_supplement.csv`） |
| 自测报告与原始证据 | 按验收阶段实际提交的版本提供用例参数、实部/虚部精度结果、失败项诊断、性能与 msprof 记录、内存说明（workspace 0）及对应源码校验值；同一份材料内的表格与截图取自同一批结果 |

# 风险分析与规避

| 风险 | 影响 | 规避 |
| --- | --- | --- |
| 读取带外/填充/步长空洞位置 | 违反任务约束，特殊值下结果错误 | 按带内范围精确寻址；测试以 NaN 填充这些位置做越界检测 |
| VF 内 `blockIdx` 不能区分 AIV 核 | 多核时输出缺失 | 核号由 `GetBlockIdx()/GetBlockNum()` 显式传入；用例覆盖多 block 场景 |
| Device 标量无法在 host 判定 quick return | 误写 y 或误读 A/x | 判定下沉到 kernel 入口，规则与 host 相同；补充 Device 标量用例 |
| 小 m 的 N 路径并行度不足 | 性能不达标 | 列窗口切段，块内多 warp 并行 |
| 大带宽 N 路径列窗口浪费（L+31 次迭代中 31 次为空） | 大带宽性能下降 | L 越大浪费比例越低（L=1025 时约 3%）；长段改用步进循环降低每次迭代开销 |
| m≫n 时大部分行不含带内元素 | 空转 tile 与块内同步浪费时间 | 只对带内行 `[0, min(m,n+kl))` 分块，其余行 grid-stride 直接写 beta·y |
| N 部分和先溢出，丢失与 beta·y 的相消 | 顺序参考有限而结果为 Inf/NaN | 两种 VF 均经 `StoreNResult` 检查非有限归约，并从 beta·y 开始按列序重算；四条溢出相消回归覆盖短段、长段、段内/段间及一般 beta |
| 复数特殊值传播与参考不同 | NaN/Inf 分类或 Inf 符号错误 | N/T/C 异常重算与一般 beta 写回使用参考复数乘法；严格分类与符号判定，八组异常回归覆盖；有限输出不触发完整重算 |
| alpha 顺序或空行处理不当 | 多余溢出或空行被 Inf/NaN 污染 | N 先乘 alpha；不含带内元素的行只做 beta·y、不经过 alpha；由 `NLongSliceScalesXByAlphaFirst` 和 `NRowsBelowBandSkipAlpha` 覆盖 |
| 累加顺序与 FP32 Golden 不同 | 有限值舍入差异触发精度失败 | 按原判据运行，保留四条失败与 FP64 诊断；Golden 自身偏差不作为自动豁免依据 |
| 异常输出重算较多或函数拆分不当 | 额外 A/x 读取、串行累加、等待或调用开销 | 仅在规定检查点非有限时触发；按同机对照选择拆分方式：T/C 重算单独 noinline 且写回不复用 `StoreY`，N 的重算与通用写回保持内联；性能结论限定为已测用例和环境 |
| 性能日志缺失或计时口径不一致 | 错将子集或无效数据判为通过 | `perf_compare.py` 检查完整 case 集合、有效时间、采样数与基线；使用 event 并以 msprof 交叉验证 |

# 参考资料

1. 任务书：`aclblasCgbmv_Atlas950PR_task_doc.md`
2. 设计文档模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
3. ops-blas 开源仓：https://gitcode.com/cann/ops-blas
4. cuBLAS `cublas<t>gbmv`：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gbmv
5. Netlib BLAS `cgbmv.f`：https://www.netlib.org/blas/cgbmv.f
6. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
7. Ascend C 算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
8. 无卡仿真：https://gitcode.com/org/cann/discussions/289

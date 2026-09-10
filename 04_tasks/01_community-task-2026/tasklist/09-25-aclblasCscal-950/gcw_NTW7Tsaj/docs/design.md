# 【社区任务】Cscal算子设计文档

> 对标《社区任务设计文档 checklist》（Sheet1）逐节编写
> 算子名称：Cscal　|　目标硬件：Ascend 950PR / 950DT（arch35）　|　CANN 版本：9.1.0

---

## 一、需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献的需求。任务要求在 Ascend 950PR 上使用 Ascend C 开发 `aclblasCscal` 算子，实现单精度复数向量 × 单精度复数标量的复数标量缩放计算，并合入 ops-blas 开源仓。

### 1.2 背景介绍

#### 1.2.1 Cscal算子实现优化

**标杆（参考）算子源码获取路径**——ops-blas 开源仓中已合入的 Atlas A2/A3（arch22）实现：

| 内容 | 文件路径（含文件名） |
|---|---|
| 标杆 kernel 源码 | `ops-blas/blas/scal/arch22/cscal_kernel.cpp` |
| 标杆 host 源码 | `ops-blas/blas/scal/arch22/cscal_host.cpp` |
| 算子信息库 / 接口说明 | `ops-blas/blas/scal/README.md` |
| 公共接口声明 | `ops-blas/include/cann_ops_blas.h`（第 300 行，`aclblasCscal` 声明） |
| Tiling 常量 | `ops-blas/blas/common/helper/kernel_constant.h`（`UB_SIZE`、`SIMT_MIN_THREAD_NUM`） |

**接口正确性判断**：该算子归属于 ops-blas BLAS 库，对外接口为 `aclblas*` 系列而非 `aclnn*` 系列。`include/cann_ops_blas.h:300` 的声明为

```cpp
aclblasStatus_t aclblasCscal(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, aclblasComplex* x, int incx);
```

与 `blas/scal/README.md` 中记录的 `aclblasCscal` 条目一致，据此确认上述源码即为本次需对标移植的正确参考文件。该仓不以传统 TBE（Python DSL）形式实现算子，标杆实现为 Asdops 框架下的 Ascend C 源码，因此"对标 TBE 源码"在本项目中落实为"对标 arch22 Ascend C 实现"。

本次实现优化点：在 arch35 上以 AscendC 标准 API 重写，消除标杆对 GM 侧 mask 索引表的依赖，引入 `DeInterleave/Interleave` 原生向量指令完成实虚平面拆分与重组，并按标量形态做模式分流，同时补齐 `incx` 步长支持与 BLAS 标准 no-op 语义。

#### 1.2.2 Cscal算子现状分析

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

| 项目 | 标杆算子（arch22）现状 |
|---|---|
| 数据类型 | `aclblasComplex`（COMPLEX64），实部/虚部各为 FP32，内存中相邻存放，单元素 8 字节 |
| x 存储位置 | Device GM，要求按复数连续存放（`[r0,i0,r1,i1,...]`） |
| alpha 存储位置 | Host 内存标量，host 侧取值后写入 TilingData 下发到 GM |
| stride 支持 | 接口签名含 `incx`，但 kernel 侧未使用，实际按**连续**处理；`incx != 1` 时结果不正确 |
| 输出形式 | 原地（in-place）写回同一 GM 地址（kernel 内 `x_tensor` 同时作为源与目的） |
| 算子平台 | `__DAV_C220_VEC__` 宏保护，仅在 Atlas A2/A3 向量核生效 |

与 `include/cann_ops_blas.h` 及 `blas/scal/README.md` 中登记的信息保持一致（均为 `aclblasComplex` + `incx`），但 README 未明确标注步长限制，属实现层面的隐含约束。

##### 1.2.2.2 标杆算子实现描述

标杆基于 Asdops 老框架 + `ASCEND_V220` 向量架构，逻辑与源码完全一致，流程如下。

**（1）UB 布局（`UB_FOR_CACL`，见 `cscal_kernel.cpp:29-54`）**

共 13 个 UB buffer，由 `allocate_ubuf()`（第 69-102 行）用 `AsdopsBuffer<ArchType::ASCEND_V220>::GetBuffer` 按偏移静态切分：

| buffer | 用途 | 大小（字节） |
|---|---|---|
| `buf_for_mask_fp32` | vgather 索引表 | `MAX_LENG_PER_UB_PROC*2*4 = 49 152` |
| `buf_for_load_vector_fp32` | 交织输入双缓冲 | `MAX_LENG_PER_UB_PROC*8*2 = 98 304` |
| `buf_for_real_part_fp32` | 解交织后的实部平面 | `MAX_LENG_PER_UB_PROC*4 = 24 576` |
| `buf_for_imag_part_fp32` | 解交织后的虚部平面 | `MAX_LENG_PER_UB_PROC*4 = 24 576` |
| 合计 | | **196 608 B = 192 KiB** |

其余成员为上述 buffer 的别名/偏移切片（如 `buf_for_final_real_fp32` 复用 `buf_for_imag_part_fp32`），用于省 UB。MAX_LENG_PER_UB_PROC 常量为 **6144 个复数**。

**（2）数据搬运与解交织（`_cacl()`，第 146-248 行）**

1. `gm_to_ub<float>` 一次性把 `data_count*2` 个 float（实虚交织）搬入 UB 双缓冲之一；
2. **两次 `vreducev2` 指令完成解交织**：
   - 第 1 次按步长 1 抽取偶数位 → `buf_for_real_part_fp32`（实部平面）
   - 第 2 次按步长 2 抽取奇数位 → `buf_for_imag_part_fp32`（虚部平面）

**（3）复数乘（`_cacl()`，第 200-224 行）**

设 `alpha = a + bi`、`x = c + di`，标杆用 **4 次 `muls_v` + 2 次 `add_v`** 实现，且刻意用加法替代减法（第 3 次乘对 `-alpha_imag` 取负）：

```
real_real  = real * a
real_imag  = real * b
imag_real  = imag * a
imag_imag  = imag * (-b)
final_imag = real_imag + imag_real      //  a*d + b*c
final_real = real_real + imag_imag      //  a*c - b*d
```

每步之间插入 `PIPE_BARRIER(V)` 保证向量流水顺序。

**（4）交织写回（`_cacl()`，第 226-245 行）**

`vgather` 先按 mask 生成索引，再调用 `AscendC::Gather` 把实部/虚部两个平面重新交织为一个 buffer，最后 `ub_to_gm<float>` 写回 GM 原地址。

**（5）pingpong 与循环（`process_cacl()`，第 250-329 行）**

按 `element_count` 分三种情况：**≤256** 单块不乒乓；**≤ MAX_LENG_PER_UB_PROC*2（12 288）** 两段乒乓；**其余** 以 6144 为步长多轮乒乓。同步用 `EVENT_ID0/EVENT_ID1` 两组 `SET_FLAG/WAIT_FLAG`（MTE2↔V↔MTE3）。

**（6）host 侧（`cscal_host.cpp`）**

- `numBlocks` **硬编码 40**（第 73 行），不做动态核数适配；
- `CreateMaskData()` 在 host 侧生成 12 288 项的索引表，`aclrtMalloc` + `aclrtMemcpy` 下发到 GM（第 83-109 行）；
- kernel 发射后 **host 侧同步等待** `aclrtSynchronizeStream`（第 112 行），非异步。

##### 1.2.2.3 标杆算子实现流程图

以下流程图与 arch22 标杆源码 `blas/scal/arch22/cscal_kernel.cpp`、`cscal_host.cpp` 逻辑逐分支一致：

```text
[Host] aclblasCscal(handle, n, alpha, x, incx)
   │
   ├─ alpha == nullptr ──> return INVALID_VALUE
   ├─ numBlocks = 40 (硬编码)
   ├─ CalTilingData: {n, alphaReal, alphaImag}
   ├─ CreateMaskData: 生成 12288 项 vgather 索引表 (host)
   ├─ aclrtMalloc + aclrtMemcpy: tilingDevice, maskDevice → GM
   ├─ cscal_kernel_do(x, maskBuf, workspace=null, tilingGm, 40, stream)
   └─ aclrtSynchronizeStream(stream)          ← 同步等待

[Kernel] cscal (通告 __vector__ / AIV-only)
   │
   ├─ SetMaskNorm / SetVectorMask 全开
   ├─ 从 GM 读 tiling: vector_len, alpha_r, alpha_i
   ├─ gm_to_ub: 搬入 mask 索引表 (6144/4 blocks)
   ├─ allocate_ubuf(): 13 个 UB buffer 静态切分 (192 KiB)
   ├─ get_local_info(): max_cores = len/4; cores_num = min(BlockNum, max_cores)
   │                    elnum_per_core = ceil(len/cores_num/4)*4; tail 处理
   ├─ start_pos == -1 ──> 直接 return (空转核)
   └─ process_cacl(): 三分支
        ├─ count <= 256              → 单块，无乒乓
        ├─ count <= 12288            → 2 段乒乓
        └─ count  >  12288           → 以 6144 步长多轮乒乓
             每轮 _cacl():
               WAIT_FLAG(MTE3→MTE2)
               gm_to_ub       交织数据 → UB 双缓冲
               SET/WAIT_FLAG(MTE2→V)
               vreducev2      ×2  解交织 → real[] / imag[] 平面
               PIPE_BARRIER(V)
               muls_v         ×4  real*a, real*b, imag*a, imag*(-b)
               PIPE_BARRIER(V)
               add_v          ×2  final_imag = real*b + imag*a
                                  final_real = real*a + imag*(-b)
               PIPE_BARRIER(V)
               vgather + Gather   平面 → 交织
               SET/WAIT_FLAG(V→MTE3)
               ub_to_gm       交织结果 → GM 原地写回
               SET_FLAG(MTE3→MTE2)
```

---

## 二、需求分析

### 2.1 外部组件依赖

| 依赖 | 版本/来源 | 用途 |
|---|---|---|
| CANN AscendC kernel API | CANN 9.1.0 | `TPipe`/`TQue`/`TBuf`、`DataCopy`、`DataCopyPad`、`DeInterleave`/`Interleave`、`Muls`/`Sub`/`Add`、`Duplicate`、`GetBlockIdx` |
| CANN SIMT API | CANN 9.1.0 | `__simt_vf__`、`__simt_callee__`、`asc_vf_call`、`threadIdx.x`/`blockDim.x`、`LAUNCH_BOUND` |
| ACL Runtime | CANN 9.1.0 | `aclrtStream`、`<<<blocks, nullptr, stream>>>` 发射 |
| 编译架构宏 | arch35（`dav-3101`） | `__CCE_AICORE__` 分支、`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` |

无第三方库依赖，无开源组件引入。

### 2.2 内部适配模块

| 模块 | 路径 | 复用内容 |
|---|---|---|
| 公共头文件 | `blas/common/helper/kernel_constant.h` | `UB_SIZE = 248*1024`、`SIMT_MIN_THREAD_NUM = 128`、`SIMT_MAX_THREAD_NUM = 2048` |
| Host 工具 | `blas/common/helper/host_utils.h` | `CeilDiv`、`CeilAlign`、`GetAivCoreCount()` |
| Handle 内部结构 | `blas/common/helper/aclblas_handle_internal.h` | `handle->stream` |
| 日志 | `log/log.h` | `OP_LOGE` / `OP_LOGD` |
| 公共接口声明 | `include/cann_ops_blas.h` | `aclblasCscal` 原型、`aclblasComplex`、`aclblasStatus_t` |

### 2.3 需求模块设计

#### 2.3.1 Ascend C算子原型

与公共头文件保持一致，不新增 950PR 私有接口：

```cpp
aclblasStatus_t aclblasCscal(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, aclblasComplex* x, int incx);
```

| 参数 | 位置 | 说明 |
|---|---|---|
| handle | Host | 携带 stream 的 BLAS 上下文 |
| n | Host | 参与计算的复数元素个数 |
| alpha | **Host** | 复数标量（host 侧取值后填入 Tiling，不下发指针） |
| x | **Device GM** | 复数向量，原地读写 |
| incx | Host | 相邻复数元素的步长（以复数为单位） |

计算语义（`j` 为复数下标）：

```
for i = 0 .. n-1:
    j     = i * incx
    c     = x[j].real ;  d = x[j].imag
    real' = a*c - b*d
    imag' = a*d + b*c
    x[j]  = (real', imag')
```

#### 2.3.2 Ascend C算子相关约束

与标杆算子相比的**新增能力**与**缺失项**：

| 维度 | 标杆（arch22） | 本实现（arch35） | 性质 |
|---|---|---|---|
| `incx` 步长 | 接口有但 kernel 未用，仅连续正确 | **支持任意正步长**（SIMT 跨步路径） | 新增 |
| 负步长 | 未定义 | 显式判为合法 no-op，返回成功 | 新增（对齐 Netlib） |
| 核数适配 | 硬编码 40 | `GetAivCoreCount()` 动态获取 | 新增 |
| GM 辅助内存 | 需要 maskBuf 索引表 + workspace 参数 | **不需要任何额外 GM  buffer** | 新增 |
| 异步性 | host 侧 `aclrtSynchronizeStream` | 仅发射，接口内不同步 | 新增 |
| `alpha=(1,0)` | 无快速返回，仍下发 kernel | Host 侧快速返回，保留 x 比特 | 新增 |
| `alpha=(0,0)` | 走通用路径，`0*NaN` 传播 NaN | MODE_ZERO 只写不读，强制 `+0.0f` | 新增 |
| 数据类型 | COMPLEX64 | COMPLEX64（一致） | 对齐 |
| 计算精度顺序 | `real*a + imag*(-b)` | 四次独立乘积 + 1 sub + 1 add | 差异（见 3.2.2.3） |
| 融合乘加（FMA） | 未处理 | strict/fast 双 SIMT 变体应对 | 新增 |

**缺失项说明**：本实现不支持 `incx < 0` 的反向向量遍历（与 Netlib `cscal` 一致作 no-op），不支持双精度 `aclblasZscal`，不支持虚实hoff分离的非交错布局。

---

## 三、需求详细设计

### 3.1 调用方式

采用 **Kernel 直调** 方式：`aclblasCscal` 作为 ops-blas 对外 C API，内部由 host 侧的 `cscal_kernel_do()` 通过 `<<<numBlocks, nullptr, stream>>>` 直接核函数启动，不经 ACLNN 两层接口，也不对接 PyTorch 框架。

### 3.2 需求总体设计

#### 3.2.1 host侧设计

##### 3.2.1.1 分核策略

以 `GetAivCoreCount()` 动态获取可用 AIV 核数 `A`。

**连续路径（CONTIGUOUS_VECTOR）**：

```
requested = ceil(n / MIN_COMPLEX_PER_SIMD_CORE)      // MIN_COMPLEX_PER_SIMD_CORE = 2048
numBlocks = max(1, min(A, requested))
perCoreN  = floor( n / numBlocks / COMPLEX_PER_BLOCK ) * COMPLEX_PER_BLOCK
                                                      // COMPLEX_PER_BLOCK = 32/8 = 4
remainder = n - perCoreN * numBlocks
```

每个核处理 `perCoreN` 个复数，最后一个核额外承担 `remainder`（kernel 内 `if (blockIdx == GetBlockNum()-1) myCount_ += tiling.remainder`）。`perCoreN` 向下对齐到 4 个复数（32 字节）以保证 32B 对齐搬运。

**跨步路径（STRICT/FAST_DIRECT_SIMT，含 `incx>1`）**：

```
numBlocks = max(1, min(A, ceil(n / SIMT_MIN_THREAD_NUM)))     // SIMT_MIN_THREAD_NUM = 128
perCore   = ceil(n / numBlocks)
nthreads  = min( ceilAlign(perCore, 128), SIMT_MAX_THREAD_NUM )  // 上限 2048
```

kernel 内再按 `startOffset = blockIdx*baseCount + min(blockIdx, remainder)` 做均分，尾部核多担 1 个元素。

**收益**：小向量（n < 2048）只用 1 个核，避免标杆"无论大小都发射 40 个核"造成的空转；大向量则吃满全部可用 AIV。

##### 3.2.1.2 数据分块和内存优化策略

按 **标量形态 → KernelMode** 分流，不同模式 UB 占用差异显著，tile 大小随之自适应：

| KernelMode | 触发条件 | UB 中的 float 份数 / 复数 | UB 占用公式 | 计算所得 tileSize |
|---|---|---|---|---|
| `ZERO` | `alpha.real==0 && alpha.imag==0` | 2（仅输出队列） | `2*4*tile` | 31 712 |
| `REAL_ONLY` | `alpha.imag==0`（且非全零） | 4（输入 + 输出队列） | `4*4*tile` | 15 840 |
| `GENERAL` | 其余 | 8（输入/输出队列 + 4 个分量平面） | `8*4*tile` | 7 904 |

tile 推导（host 侧 `MakeContiguousTiling`）：

```
maxComplex = (UB_SIZE - UB_RESERVE_BYTES) / (FloatsPerComplexInUb(mode) * sizeof(float))
                                          // UB_SIZE = 248*1024 = 253 952，预留 256 B
tileSize   = floor(maxComplex / VECTOR_COMPLEX_ALIGN) * VECTOR_COMPLEX_ALIGN   // 对齐 32 复数
```

以 GENERAL 模式为例：`tileSize = floor((253952-256)/32 / 32)*32 = floor(7928/32)*32 = 7 904` 个复数，对应 UB 占用 `7904 × 32 = 252 928 B ≈ 246.9 KiB`，占预算 99.6%。

**相比标杆的内存优化**：
- 标杆固定 6144 复数 tile、占用 192 KiB；本实现按模式收缩，REAL_ONLY 模式 tile 可放大到 15 840（UB 占用减半的分量平面省出来了）。
- 标杆需要 GM 上 98 304 B 的 mask 索引表且每次调用都要 `aclrtMalloc`+`aclrtMemcpy`；本实现用 `DeInterleave/Interleave` 指令在 UB 内完成实虚拆分与重组，**GM 侧零额外开销**。
- `MODE_ZERO` 不分配输入队列、不读 GM，只 `Duplicate` 出 `+0.0f` 后连续写回，读写流量减半。

**分块内处理**（`Process()`）：每块按 `fullTiles = myCount_/tileSize` 循环，尾块单独 `ProcessTile(offset, tail)`；每个 tile 内的 CopyIn/Compute/CopyOut 对非对齐尾部用 `DataCopyPad`（`DataCopyExtParams` + `DataCopyPadExtParams`）补齐到 32B，`padding` 位不写回 GM。

##### 3.2.1.3 tilingKey规划策略

本实现不用数值型 tilingKey，而是以 Tiling 结构体中的 `mode`（3 值）× `strategy`（3 值）两个枚举字段承担等价的分派职责，由 kernel 侧 `switch` 到对应计算分支：

| strategy | 取值条件 | kernel entry |
|---|---|---|
| `CONTIGUOUS_VECTOR` (0) | `incx==1` 且 不满足"GENERAL 且 n≥4096" | `cscal_contiguous_kernel`（AIV 向量化） |
| `STRICT_DIRECT_SIMT` (1) | `incx==1` 且 GENERAL 且 `n ≥ DIRECT_SIMT_THRESHOLD(4096)` 且 `max(|a|,|b|) ≥ 1.0e8` | `cscal_contiguous_simt_kernel` → `CscalContiguousSimtCompute` |
| `FAST_DIRECT_SIMT` (2) | `incx==1` 且 GENERAL 且 `n ≥ 4096` 且 `max(|a|,|b|) < 1.0e8` | `cscal_contiguous_simt_kernel` → `CscalContiguousSimtComputeFast` |

另有一条独立分支：`incx > 1` 时无论 mode 均走 `cscal_strided_kernel`（SIMT 跨步），此时 `strategy` 字段保留用于 kernel 内判断。

**为何大 GENERAL 输入走 SIMT 而非 AIV**：CPU golden 的复数乘是"四次独立乘积 + 一次减 + 一次加"，而 AIV 的 `Sub/Add` 由编译器决定是否存在-Wards融合；当标量幅度很大（≥1e8）时，`ac - bd` 会出现灾难性抵消，任何中间结果被收缩（FMA）都会放大相对误差。SIMT 路径可用 `volatile` 显式落地四次乘积，逐位对齐 CPU golden。见 5.1 节的机理分析。

#### 3.2.2 kernel侧设计

##### 3.2.2.1 kernel侧实现描述

共 3 个 kernel entry，均在 `KERNEL_TYPE_AIV_ONLY` 下运行。

**① `cscal_contiguous_kernel`——AIV 向量化主路径**

`CscalContiguous` 类持有：
- `inputQueue_` / `outputQueue_`（`TQue`，BUFFER_NUM = 1，各 `tileSize*2*sizeof(float)`）
- `realBuffer_` / `imagBuffer_` / `resultRealBuffer_` / `resultImagBuffer_`（`TBuf<VECCALC>`，仅 GENERAL 模式分配，各 `tileSize*sizeof(float)`）

计算序列（`Compute()`，对应 CPU golden 的四次乘积）：

```cpp
DeInterleave(real, imag, input, paddedFloat);     // 交织 → 实部/虚部两个平面
Muls(resultReal, real,       alphaReal_, paddedComplex);   // a*c
Muls(resultImag, imag,       alphaReal_, paddedComplex);   // a*d
Muls(imag,       imag,       alphaImag_, paddedComplex);   // b*d  (复用 imag)
Muls(real,       real,       alphaImag_, paddedComplex);   // b*c  (复用 real)
Sub (resultReal, resultReal, imag,       paddedComplex);   // ac - bd
Add (resultImag, resultImag, real,       paddedComplex);   // ad + bc
Interleave(output, output[paddedComplex], resultReal, resultImag, paddedComplex);  // 平面 → 交织
```

`real` 与 `imag` 在各自的第二次 `Muls` 中被原地复用为 `b*c`、`b*d`，因此 UB 只需 4 个分量平面而非 6 个。REAL_ONLY 模式退化单行 `Muls(output, input, alphaReal_, paddedFloat)`（整段 2n 个 float 一次乘）。MODE_ZERO 模式直接 `Duplicate(output, 0.0f, paddedFloat)`。

**② `cscal_contiguous_simt_kernel`——大 GENERAL 输入的 SIMT 融合路径**

线程网格条带 `gridStride = blockDim.x * numBlocks`，主循环按 `unrolledStride = gridStride*4` 做 **4 路展开**以暴露独立访存，尾部回落到单步循环。逐元素计算由 `CscalGeneralOneStrict` / `CscalGeneralOneFast` 承担：

```cpp
// Strict 变体
volatile float realByAlphaReal = alphaReal * real;    // volatile 强制落地乘积，
volatile float imagByAlphaImag = alphaImag * imag;    // 阻止编译器收缩为 FMA
volatile float imagByAlphaReal = alphaReal * imag;
volatile float realByAlphaImag = alphaImag * real;
x[floatIndex]     = realByAlphaReal - imagByAlphaImag;
x[floatIndex + 1] = imagByAlphaReal + realByAlphaImag;
```

`Fast` 变体去掉 `volatile`（去掉一个表注释所说的"materialization 开销"），用于标量幅度 < 1e8、不存在显著抵消风险的场景，换取更高 GM 吞吐。

**③ `cscal_strided_kernel`——跨步路径**

SIMT 线程处理一个逻辑复数元素，**先读完 c、d 再写回**（两项互相依赖对方分量，不能写成两条独立的 `*=`）：

```cpp
uint64_t floatIndex = static_cast<uint64_t>(startOffset + logical) * incx * 2U;   // 64 位防回绕
float real = x[floatIndex], imag = x[floatIndex + 1];
// 按 mode 分支：ZERO 直接置零 / REAL_ONLY 两次独立乘 / GENERAL 四次乘积
```

##### 3.2.2.2 Ascend C实现流程图

以下流程图对应本设计 arch35 实现（`blas/scal/arch35/cscal_host.cpp` + `cscal_kernel.cpp`）的实际执行路径：

```text
[Host] aclblasCscal(handle, n, alpha, x, incx)
   │
   ├─ handle == nullptr ─────> HANDLE_IS_NULLPTR
   ├─ n <= 0 || incx <= 0 ───> SUCCESS (合法 no-op，不读 alpha/x)
   ├─ alpha == nullptr ──────> INVALID_VALUE
   ├─ x == nullptr ──────────> INVALID_VALUE
   ├─ alpha == (1,0) ────────> SUCCESS (快速返回，保留 x 比特)
   │
   ├─ mode = SelectKernelMode(alpha)
   │     alpha==(0,0)      → ZERO
   │     alpha.imag == 0   → REAL_ONLY
   │     其余              → GENERAL
   │
   ├─ A = GetAivCoreCount()
   └─ 路径选择
        │
        ├─ incx == 1 && !(GENERAL && n >= 4096) ────> [A] CONTIGUOUS_VECTOR
        │     numBlocks = max(1, min(A, ceil(n/2048)))
        │     perCoreN  = floor(n/numBlocks/4)*4 ; remainder = n - perCoreN*numBlocks
        │     tileSize  = floor((UB_SIZE-256)/(4*floatPerComplex)/32)*32
        │
        └─ GENERAL && n >= 4096 ───────────────────> [B] SIMT 融合路径
              max(|a|,|b|) >= 1e8 ? STRICT (volatile 四次乘积)
                                  : FAST   (允许编译器调度)
              numBlocks = max(1, min(A, ceil(n/128)))
              nthreads  = min(ceilAlign(ceil(n/numBlocks),128), 2048)

        └─ incx > 1 ───────────────────────────────> [C] SIMT 跨步路径

────────────────────────────────────────────────────────────────
[A] cscal_contiguous_kernel (AIV)
   Init: myOffset = blockIdx*perCoreN; myCount = perCoreN (+remainder 若末核)
         按 mode 分配 UB:
           ZERO       → outputQueue             (2 float/复数)
           REAL_ONLY  → inputQueue+outputQueue  (4 float/复数)
           GENERAL    → 上两者 + 4 个分量平面    (8 float/复数)
   Process: for each tile { ProcessTile(offset, tileSize) } + 尾 tile
     ProcessTile:
       ZERO 时跳过
       CopyIn  : DataCopy(对齐部分) + DataCopyPad(尾部补齐 32B)
       Compute : ZERO      → Duplicate(output, +0.0f)
                 REAL_ONLY → Muls(output, input, a)            1 次
                 GENERAL   → DeInterleave
                             Muls×4 → Sub → Add                6 次
                             Interleave
       CopyOut : DataCopy(对齐部分) + DataCopyPad(尾部，不写 padding 位)

[B] cscal_contiguous_simt_kernel
   globalThread = threadIdx + blockIdx*blockDim ; gridStride = blockDim*numBlocks
   主循环: 4 路展开 (unrolledStride = gridStride*4)，每路 CscalGeneralOneStrict/Fast
   尾循环: 单步补齐
   元素内: volatile 落地 a*c, b*d, a*d, b*c → real'=ac-bd, imag'=ad+bc

[C] cscal_strided_kernel
   每线程一个复数: 越块核直接 return
   floatIndex = uint64(start+logical) * incx * 2        ← 64 位乘法防回绕
   先读 c,d → 按 mode 分支 → 写回 (未参与的 gap 保持原值)
```

##### 3.2.2.3 Ascend C实现流程图与标杆算子流程图存在的差异点和原因

| # | 差异点 | 原因 |
|---|---|---|
| 1 | **解交织方式**：标杆用 2 次 `vreducev2` 抽取平面，本实现用 1 次 `DeInterleave` 原生指令 | arch35 提供专门的 `DeInterleave/Interleave` 指令对，一条指令完成实虚拆分，指令数更少、语义更直观；`vreducev2` 属 V220 时代的通用归约指令，在新架构上非最优选择 |
| 2 | **交织写回**：标杆需 `vgather` + `AscendC::Gather` + GM 侧 mask 索引表，本实现用 `Interleave` 直接在 UB 内完成 | 消除 GM 上 98 304 B mask 表的分配与 H2D 拷贝开销（标杆每次调用都要 `aclrtMalloc`+`aclrtMemcpy`），也省掉 workspace 形参 |
| 3 | **符号处理**：标杆把 `imag*(-b)` 取负后用 `add_v` 替代减法；本实现用 `Sub` 显式做 `ac - bd` | CANN 9.1 的编译器可能把表达式收缩为融合乘加（FMA），而 CPU golden 是"四次独立乘积后再加减"。显式 `Sub` + strict 变体的 `volatile` 落地可保证与 golden 的计算顺序一致 |
| 4 | **SIMT 双变体**：标杆无此区分，本实现按 `max(|a|,|b|) ≥ 1e8` 切 strict/fast | 大幅值标量下 `ac - bd` 存在灾难性抵消，中间结果是否被 FMA 融合会造成可观相对差异。strict 变体用 `volatile` 强制乘积落地以对齐 golden；小幅值场景无此风险，用 fast 变体换取更高吞吐 |
| 5 | **大 GENERAL 向量走 SIMT**：`n ≥ 4096` 的大规模 GENERAL 场景由 SIMT 路径承担，而非 AIV | SIMT 的寄存器路径能精确控制每个元素的四条乘积不被融合，AIV 的向量流水线难以做到同样粒度的控制；同时 SIMT 的 4 路展开对 GM 友好，实测可接受 |
| 6 | **UB 自适应**：标杆固定 6144 复数 tile（192 KiB），本实现按 ZERO/REAL_ONLY/GENERAL 三档收缩 UB，tile 分别放大到 31 712 / 15 840 / 7 904 | 不同模式对分量平面的需求不同（REAL_ONLY 完全不需要平面），按需分配可复用释放出来的 UB，减少分块轮次 |
| 7 | **新增 mode/strategy 枚举**：标杆无此概念 | 需要在 tiling 层面把"标量形态"与"执行路径"传给 kernel，替代硬编码分支，也为后续扩展留口 |
| 8 | **异步化**：标杆 host 侧 `aclrtSynchronizeStream`，本实现接口内不同步 | BLAS 接口语义要求异步发射，由上层决定同步时机；同步等待会阻塞调用线程，丧失流水机会 |
| 9 | **核数动态化**：标杆硬编码 40 核，本实现 `GetAivCoreCount()` 并按规模收敛 | 硬编码核数在大规模数据下核数不足、小规模下大量空转核，且无法适配不同核数芯片 |
| 10 | **步长支持**：标杆 kernel 未消费 `incx`，本实现新增 SIMT 跨步路径 | 接口签名承诺了 `incx`，标杆实现存在潜在错误；补上后语义与 Netlib `cscal` 对齐，且 64 位偏移避免大 `n*incx` 回绕 |

### 3.3 支持硬件

| 硬件 | 支持情况 |
|---|---|
| Ascend 950PR | **支持**（本次实现目标，arch35） |
| Ascend 950DT | **支持**（同一 arch35 代码路径） |
| Atlas A2 / A3 训练与推理系列 | 由原有 arch22 实现承载，本次不改动 |
| Atlas 200I/500 等推理产品 | 不支持 |

SOC 编译参数为 `ascend950`，对应 `SOC_ARCH_DIRS=arch35`、`NPU_ARCH=dav-3101`。

### 3.4 算子约束限制

1. **数据类型**：仅支持 `aclblasComplex`（COMPLEX64，实虚各 FP32）。不支持双精度复数 `aclblasZscal`、不支持实向量。
2. **存储格式**：要求复数实部与虚部**相邻存储**（`[r0,i0,r1,i1,...]`）。不支持 plan拆分式（实部虚部各自成一维数组）布局。
3. **原地操作**：结果写回输入 `x` 的同一 GM 地址，不支持输出到另一个向量（无 `y` 形参）。
4. **步长**：仅支持 `incx > 0`。`incx <= 0` 按 Netlib `cscal` 语义返回成功且不修改内存——**不做反向遍历**。
5. **`n` 范围**：`n` 为 `int`，且换入 `uint32_t` 参与 tiling 计算，故要求 `0 < n ≤ 2^32-1`；`n*incx` 在跨步路径按 64 位计算，但 GM 地址空间仍受设备限制。
6. **对齐**：`perCoreN` 对齐到 4 个复数（32 B）以保证搬运效率；非对齐尾部由 `DataCopyPad` 处理，不影响正确性。
7. **REAL_ONLY 模式的语义边界**：`alpha.imag == 0` 时走整段单 `Muls` 快速路，结果等价于 `(a*c, a*d)`。若 `x` 的分量含 Inf/NaN，严格 Netlib 语义为 `ac - 0·d`（可能传播 NaN），而本模式给出 `ac`。该场景未被任务书用例覆盖，属已登记的已知行为差异（见 5.1）。
8. **无需 workspace**：本算子不申请 workspace 与额外 GM buffer。

---

## 四、特性交叉分析

| 交叉对象 | 交叉点 | 分析与处理 |
|---|---|---|
| `aclblasCsscal`（实标量 × 复向量） | 同为 scal 家族；`Csscal` 所处理的"标量为实数"情形，与本实现 `REAL_ONLY` mode 在数学上等价 | 本实现通过 `REAL_ONLY` mode 复用同一套 UB 与队列管理，不单独开一类算子分支；差异仅在 `alpha.imag != 0` 时才需要解交织。**共用 `blas/scal/` 目录、README 与 CMake 构建规则** |
| `aclblasCaxpy`（复向量加） | `alpha==(0,0)` 语义相反：`caxpy` 中 alpha 为零是 no-op，`cscal` 中 alpha 为零必须把 x 置零 | 明确和 caxpy 拉齐反而会错。本实现用 `MODE_ZERO` 走"只写不读"路径强制 `+0.0f`，含 Inf/NaN 分量也精确清零 |
| `aclblasCaxpy` / `aclblasCcopy` 的 `incx` | `caxpy` 对 `incx<0` 需反向遍历，`cscal` 不需要 | 遵循 Netlib `cscal.f` 的 `IF (N.LE.0 .OR. INCX.LE.0) RETURN`，负步长直接 no-op，不复用 caxpy 的反向逻辑 |
| `aclblasCrot*` 等带多标量接口 | 均为 host 侧取值后填 tiling | 一致沿用，本实现不引入 device 侧 alpha 指针，减少一次 H2D |
| arch22 `cscal` | 同接口名不同架构并存 | 通过 `SOC_ARCH_DIRS` 切换，`ascend950` 只编译 `arch35/`，不会与 `arch22/` 符号冲突 |
| 性能测试 COMPLEX64 案例 | 本实现三种执行路径均有争用 | "n≥4096 的 GENERAL" 是性能热路径，由 SIMT 4 路展开承担；其余走 AIV 向量化，与任务书 3 个连续 case（n=1M/2M/4M，incx=1）恰好对应最热路径 |

---

## 五、可维可测分析

### 5.1 精度标准/性能标准

**任务书给出的验收标准**

| 类别 | 标准 |
|---|---|
| 精度（实部/虚部分别按 FP32 判定） | `atol = 2^-16 ≈ 1.5259e-5`，`rtol = 2^-10 ≈ 9.7656e-4`；`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2` 或 `32×ULP` |
| 性能（COMPLEX64，warmup 后有效采样 > 50 次取平均） | case1 n=1 048 576 / incx=1 → 13.73 us<br>case2 n=2 097 152 / incx=1 → 21.22 us<br>case3 n=4 194 304 / incx=1 → 43.31 us |

**测试工程**：`test/scal/cscal/arch35/cscal_test.cpp`（GTest），由 `cscal_test.csv` 驱动 1200 条用例（1 000 精度 + 200 性能），外加 7 条不依赖 CSV 的手写语义用例。CPU golden 用 Netlib CBLAS `cblas_cscal`；`alpha=(0,0)` 用例两侧均强制置零以对齐 bit-exact 要求。性能用例固定 warmup 10 次 + 采样 100 次，用 ACL event 对 100 次 launch 整体计时，不计入内存分配与 H2D/D2H。

**实机自验现象与机理分析**（依据 `03-自验材料/cscal_ex0325.log`）

在 `TC_EX_0325`（n=1 048 576，incx=1，大幅值 alpha）上观察到：

```
[TC_EX_0325_real] FAILED (matchedRatio=9.9999e-01 (req=9.9000e-01),
                          maxAbsErr=8.1920e+03,
                          per-element-limit=max(1.0000e-02, 32*ULP_at_each_element),
                          9/1048576 failures)
```

即 **1 048 576 个元素中仅 9 个（real）/ 12 个（imag）不符**，`matched_ratio` 本身达标（99.99% > 99%），失败源于 `per-element-limit` 判定。根因是**浮点灾难性抵消（catastrophic cancellation）**：

- 该用例中间量级在 `~8.58e10`，FP32 在此量级的 ULP 约为 `2^14 = 16 384`，单次乘法的舍入误差天然达到数千量级；
- 当 `a*c ≈ b*d` 时，`ac - bd` 的结果远小于参与运算的两个操作数，绝对误差（≈8000）相对**结果本身**的 ULP 被放大若干数量级，于是 `32×ULP_at_each_element` 这条 per-element 门槛无法通过；
- 这属于 IEEE-754 二进制浮点在该数值区间的固有性质，并非实现缺陷——同一现象在任何严格按 BLAS 语义计算的实现遇到该量级的输入时都会复现。

**针对性改进**：正是为此设计了 `STRICT_DIRECT_SIMT` 变体——用 `volatile` 强制落地 `a*c`、`b*d`、`a*d`、`b*c` 四次乘积，禁止编译器把 `ac - bd` 收缩为融合乘加（FMA），使 kernel 的计算顺序与 CPU golden 的"四次独立乘积 + 一减一加"逐位一致，最大限度压低此类用例的失败元素比例。

**后续建议**：若该用例仍需通过 per-element 门槛，可选方案为（a）调整该用例输入，避免 `ac` 与 `bd` 量级接近且接近 FP32 表示上限的组合；（b）对该类"结果远小于操作数"的场景改用相对误差主导的判定策略。二者均需与验收方确认口径后实施。

**性能数据**：受本地环境限制未见 950PR 实机完整跑测记录，三个任务书 case 的实测值需在目标硬件执行后回填，不以模拟值计入。

### 5.2 兼容性分析

| 维度 | 分析 |
|---|---|
| 架构兼容 | 仅在 `ascend950*`（950PR/950DT）下编译，`SOC_ARCH_DIRS=arch35`；A2/A3 由 arch22 实现承载，二者互斥不冲突 |
| API 兼容 | 复用 `include/cann_ops_blas.h` 中既有 `aclblasCscal` 声明，未新增/改动公共接口，调用方无需适配 |
| ABI 兼容 | 原地读写、无 workspace、无额外 GM buffer，与既有内存模型一致 |
| 语义兼容 | 行为对齐 Netlib `cscal.f`（含 `n<=0`/`incx<=0` no-op、`alpha=(1,0)` 快速返回、`alpha=(0,0)` 置零），与 cuBLAS `cublasCscal` 在该几点上一致 |
| 编译兼容 | 依赖 CANN 9.1.0 的 `DeInterleave/Interleave` 与 SIMT API；若后续 CANN 版本调整 SIMT 接口，需同步 `__simt_vf__`/`asc_vf_call` 用法 |
| 工具兼容 | 测试工程依赖 GTest + Netlib 参考 BLAS；仓库自带 CMake 挂接规则，`ascend950` 下走 gtest 分支 |
| 数值兼容 | `REAL_ONLY` 模式在 `x` 分量含 Inf/NaN 且与 Netlib 严格语义存在已知差异（见 3.4 第 7 条），属受控登记项 |

---

## 附：本设计对应交付文件清单

| 类别 | 文件 |
|---|---|
| 算子 host | `blas/scal/arch35/cscal_host.cpp` |
| 算子 kernel | `blas/scal/arch35/cscal_kernel.cpp`、`cscal_kernel.h` |
| Tiling 定义 | `blas/scal/arch35/cscal_tiling_data.h` |
| 接口文档 | `blas/scal/README.md`（已补充 950PR 支持并修正函数原型） |
| 测试工程 | `test/scal/cscal/`（param/golden/CSV/GTest/构建脚本/README） |
| 自验材料 | 测试用例集 CSV 与实机日志（见 `03-自验材料/`） |

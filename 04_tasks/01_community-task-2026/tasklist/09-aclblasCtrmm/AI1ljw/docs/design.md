# 需求背景（required）

## 需求来源

本设计对应 2026 年 9 月社区任务 `aclblasCtrmm`（A2/A3）。任务要求在
Atlas A2/A3 系列产品（性能测试设备 Atlas 800T A2 / 910B3）上，基于
Ascend C 编程语言、依托 ops-blas 开源仓工程框架，开发单精度复数
（COMPLEX64）三角矩阵乘算子 `aclblasCtrmm`，功能与参数语义对齐
cuBLAS `cublasCtrmm`，验收通过后合入 `blas/trmm/arch22/`，测试代码
合入 `test/trmm/ctrmm/arch22/`。

任务书规定：CANN 版本 9.1.0；kernel 直调工程模式，通过 handle 绑定
stream 直调 NPU kernel；精度 golden 由 Netlib BLAS `ctrmm` 生成，按
COMPLEX64 混合容差标准判定；性能需满足任务书 5 个性能 case 的达标
耗时（104.6us--19931us，Atlas 800T A2 实测平均单次耗时）。

## 背景介绍

### aclblasCtrmm 算子开发

TRMM（TRiangular Matrix Multiply）是 BLAS Level-3 中的三角矩阵乘
算子，计算 `C = alpha * op(A) * B`（或 `C = alpha * B * op(A)`），
其中 A 为三角矩阵（仅上三角或下三角有效），B、C 为一般矩阵。TRMM
在 LU/Cholesky 分解的后处理、 triangular solve 的配套计算、以及各类
基于分块三角分解的科学计算与深度学习框架中被广泛使用。

与 GEMM 相比，TRMM 有三点语义差异必须在实现中显式处理：

1. **三角裁剪**：`uplo` 决定 A 只有上三角（UPPER）或下三角（LOWER）
   参与计算，另一三角区域元素不被访问，取值不影响结果；
2. **对角规则**：`diag=ACLBLAS_UNIT` 时对角元素恒按 1 参与计算且不
   读取，`diag=ACLBLAS_NON_UNIT` 时读取实际对角元素；
3. **无 beta**：TRMM 的 C 是纯输出（`C = alpha * op(A) * B`），不叠
   加历史 C 值；允许 `B ≡ C` 原地写出（cuBLAS trmm 的等价原地用法）。

复数版本下 `trans=ACLBLAS_OP_C` 取复共轭转置 `op(A) = Aᴴ`，需要按
aclblasComplex 复数语义处理实虚部。

### ops-blas 仓现有实现现状分析

`blas/trmm/` 目录当前只有面向 Ascend 950（arch35）的 `aclblasStrmm`
（float32）实现；Atlas A2/A3 对应的 arch22 目录为空，复数版本
`aclblasCtrmm` 尚不存在。仓内可复用的工程资产如下：

| 层次 | 关键文件 | 现有职责 | 本算子复用点 |
| --- | --- | --- | --- |
| 接口声明 | `include/cann_ops_blas.h` | 全部 aclblas 接口对外声明（如 `aclblasStrmm` 在 L322 附近） | 新增 `aclblasCtrmm` 声明，供其他产品线共用 |
| Host 基础设施 | `blas/common/helper/host_utils.h`、`aclblas_handle_internal.h` | AIV/AIC 核数查询、workspace 扩容（EnsureDefaultWorkspace）、handle/stream 解析 | 直接复用 |
| 复数 BLAS-3 共享框架 | `blas/common/helper/complex_blas3_arch22.h`（列连续 GatherMask 拆平面 + 官方 Matmul API 实数 GEMM）、`complex_blas3_expand_arch22.h`（64×64 tile 搬运/转置/掩码机制）、`complex_blas3_tiling_data.h`、`complex_blas3_host_utils.h` | chemm/csymm/cherk/csyrk/cher2k 共用的三段式流水：Phase 0 拆实虚平面 → Phase 1 Matmul API GEMM（128×128×64 tile、L0 双缓冲、K 按 8192 分块原子累加、任意 shape 尾块自动裁剪）→ Phase 2 组合 | Phase 0a/1 整体复用；本算子新增 Phase 0b（三角清零/单位对角/T·C 转置共轭）与 Phase 2（无 beta 的按列组合） |
| 同目录兄弟算子 | `blas/trmm/arch35/`（strmm_host.cpp / strmm_kernel.cpp） | trmm→"三角补全 + GEMM + alpha 缩放"三段式、列主序 ld 寻址、alpha=0 清零语义 | 作为 trmm 语义处理的算法参考 |
| 测试框架 | `test/chemm/`、`test/trmm/strmm/` | CSV 用例 + C++ GTest + golden 参考实现 + npu wrapper 的目录结构 | 新建 `test/trmm/ctrmm/` 对齐该结构 |

本算子对外参数语义（任务书 §2.4）：

| 参数 | 输入/输出 | 数据类型 | 语义与约束 |
| --- | --- | --- | --- |
| `handle` | 输入 | aclblasHandle_t | 已创建的有效句柄，nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `side` | 输入 attr | 枚举 | LEFT：`C=alpha·op(A)·B`，A 为 m×m；RIGHT：`C=alpha·B·op(A)`，A 为 n×n |
| `uplo` | 输入 attr | 枚举 | UPPER / LOWER，决定 A 的有效三角 |
| `trans` | 输入 attr | 枚举 | N：`op(A)=A`；T：`op(A)=Aᵀ`；C：`op(A)=Aᴴ`（复共轭转置） |
| `diag` | 输入 attr | 枚举 | UNIT：对角视为 1 且不读取；NON_UNIT：读取对角 |
| `m`、`n` | 输入 | int | C/B 行列数；`m≥0`、`n≥0`，0 为合法 no-op |
| `alpha` | 输入 | COMPLEX64 标量指针（Host 内存） | nullptr 非法；`alpha=0` 时 A、B 不被引用 |
| `A` | 输入 tensor | COMPLEX64，ND，Device | 三角矩阵，side=LEFT 为 m×m、RIGHT 为 n×n |
| `lda` | 输入 | int | `lda ≥ max(1, A阶数)` |
| `B` | 输入 tensor | COMPLEX64，ND，Device | m×n 一般矩阵 |
| `ldb` | 输入 | int | `ldb ≥ max(1, m)` |
| `C` | 输出 tensor | COMPLEX64，ND，Device | m×n 输出矩阵，允许 `B ≡ C` 原地 |
| `ldc` | 输入 | int | `ldc ≥ max(1, m)` |

矩阵均为**列主序**（Column-Major）：`A(i,j) = A[i + j*lda]`（按复数
元素计），B、C 同理。C 为标准一般矩阵输出（离席计算，非三角输出）。

# 需求分析（required）

## 需求描述

在 ops-blas 仓 `blas/trmm/arch22/` 下，以 Ascend C kernel 直调方式
实现 `aclblasCtrmm`，支持 COMPLEX64 单精度复数、`side × uplo ×
trans × diag` 全部 2×2×3×2 = 24 种合法组合、列主序与前导维度
（含 padding）、`B ≡ C` 原地语义，以及任务书要求的边界与负向行为；
接口声明放入 `include/cann_ops_blas.h`。精度满足 COMPLEX64 混合容差
标准，性能在 Atlas 800T A2 上满足任务书 5 个性能 case 达标线。

## 需求拆解

1. **接口闭环**：在 `include/cann_ops_blas.h` 声明 `aclblasCtrmm`，
   签名与 `cublasCtrmm` 参数一一对应，实现在 `blas/trmm/arch22/`，
   随 `blas/CMakeLists.txt` 的 arch 目录自动收集规则参与编译。
2. **校验闭环**：按任务书 §2.4 顺序完成 handle、枚举、m/n、空指针、
   前导维度校验；`m=0` 或 `n=0` 返回 SUCCESS 且不启动 kernel；
   `alpha=0` 时 A、B 不被引用且 C 写零。
3. **三角语义闭环**：只访问 `uplo` 指定三角区域；`diag=UNIT` 不读取
   对角并按 1 计；`trans=T/C` 在预处理阶段等价转换为 op(A) 的完整稠
   密形式，使主计算归一为标准复数矩阵乘。
4. **复数语义闭环**：complex64 按 8 字节完整元素处理，实虚部一致参
   与计算；`trans=C` 的共轭、alpha 复数乘均在复数域定义，不拆散实
   虚配对关系。
5. **算法闭环**：主计算走仓内复数 BLAS-3 共享框架的官方 Matmul API
   实数 GEMM（L1/L0 分块与双缓冲由库调度），全部形状统一一条路径
   （尾块自动裁剪），无手写 Cube/fallback 分叉，无静默回退到 CPU。
6. **精度闭环**：golden 对齐 Netlib BLAS `ctrmm`，按 COMPLEX64 混合
   容差（实虚部分别套用 FLOAT32 容差）判定。
7. **性能闭环**：5 个性能 case 的 shape（256/512/1024/2048/4096）
   均为 64/32 对齐，全部命中 Cube 主路径，达标耗时见可维可测分析。
8. **测试闭环**：按 `test/trmm/ctrmm/` 结构交付 CSV 用例、GTest 工
   程、golden 与 npu wrapper，覆盖精度/边界/负向/特殊值/性能用例。

# 详细设计（required）

## 算子分析

### 数学公式

设 `k` 为 A 的阶数（`side=LEFT` 时 `k=m`，`side=RIGHT` 时 `k=n`），
`op(A)` 由 `trans` 决定：

$$
op(A) = \begin{cases} A, & trans=N \\ A^T, & trans=T \\ \overline{A}^{\,T}, & trans=C \end{cases}
$$

计算公式：

$$
side=LEFT:\quad C = \alpha \cdot op(A) \cdot B,\qquad op(A)\in\mathbb{C}^{m\times m}
$$

$$
side=RIGHT:\quad C = \alpha \cdot B \cdot op(A),\qquad op(A)\in\mathbb{C}^{n\times n}
$$

其中 A 仅三角区域有效。引入三角掩码 `t(i,j)`（`uplo=UPPER` 时
`t(i,j)=1 ⟺ i≤j`；`LOWER` 时 `t(i,j)=1 ⟺ i≥j`）与对角规则
`d(i)`（`diag=UNIT` 时 `d(i)=1`，否则 `d(i)=A(i,i)`），则 op(A) 的
完整稠密形式为：

$$
\widehat{A}(i,j) = \begin{cases} 1, & i=j \wedge diag=UNIT \\ t(i,j)\cdot \overline{A(j,i)}^{\,[trans=C]}, & \text{otherwise} \end{cases}
$$

（`trans=T` 时取 `A(j,i)` 不共轭，`trans=N` 时取 `A(i,j)`；掩码
`t(i,j)=0` 的位置补 0）。于是任意参数组合统一归约为稠密复数矩阵乘：

$$
side=LEFT:\ C = \alpha\,(\widehat{A} B),\qquad side=RIGHT:\ C = \alpha\,(B\,\widehat{A})
$$

复数矩阵乘按实虚平面分解。记 `X = X_r + iX_i`，则：

$$
C_r = \widehat{A}_r B_r - \widehat{A}_i B_i,\qquad
C_i = \widehat{A}_r B_i + \widehat{A}_i B_r
$$

即一次 k 阶复数乘分解为 4 个实数矩阵乘（RR、II、RI、IR），输出复
数乘按 `(a+bi)(c+di) = (ac-bd)+(ad+bc)i` 展开。

### 支持数据类型

| dtype | 元素宽度 | 计算/搬运语义 |
| --- | --- | --- |
| complex64 | 8 byte（float32 实部 + float32 虚部，交织存储） | 预处理阶段解交织为实部/虚部两个 float32 平面（AoS→SoA），计算阶段按 4 个实数矩阵乘组合，输出阶段重交织写回 complex64 |

语义桥接为（`P_X` 记 X 拆平面后的行主序视图，列主序 buffer 的行主序
视图即其逻辑转置）：

```text
GM complex64 [real32 | imag32] (AoS, 列主序)
    -> Phase 0a split kernel: 按列 DataCopyPad + GatherMask 解交织
       -> workspace float32 平面 Ar/Ai（A 缩放 2^-8 防溢出）与 Br/Bi
    -> Phase 0b tri kernel: 无效三角清零 + UNIT 对角置 1（向量 Select）
       + trans=T/C 平面转置（±共轭，64×64 tile Gather 转置）
       -> op(A)^T 稠密平面
    -> Phase 1 gemm kernel ×4: Matmul API 实数 GEMM（t1..t4）
    -> Phase 2 combine kernel: Pr=t1−t2、Pi=t3+t4、×2^8 复原、
       alpha 复数乘、按列 Gather 交织
       -> GM complex64 C (列主序, ldc)
```

### 支持形状

| 矩阵 | 逻辑形状 | 寻址 |
| --- | --- | --- |
| A | side=LEFT：m×m；side=RIGHT：n×n（记阶数 k） | `A[i + j*lda]`，只访问有效三角 |
| B | m×n | `B[i + j*ldb]` |
| C | m×n | `C[i + j*ldc]`，只写前 m 行，padding 行不触碰 |

支持 `m,n ≥ 0`，`lda ≥ max(1,k)`、`ldb/ldc ≥ max(1,m)` 的全部
padding 场景；不涉及广播与非连续 view（超出 ld 语义的内存访问不在
本批次范围）。

### 支持参数组合

`side(2) × uplo(2) × trans(3) × diag(2)` 共 24 种组合全覆盖，
在 host 侧作为 tiling 字段下发，kernel 侧不因组合数产生分支爆炸
（三角清零/单位对角/转置在 Phase 0b 一次完成，后续各阶段与组合无关）。

## 算子实现

### 实现方案

整体执行流程（kernel 直调，全部异步于 `aclblasSetStream` 绑定的
handle stream，复用仓内复数 BLAS-3 共享框架 chemm/csymm/cherk 同源
的三段式流水）：

```text
aclblasCtrmm
  -> 参数校验: handle / side / uplo / trans / diag / m / n
     （m=0 或 n=0 -> SUCCESS no-op，直接返回）
  -> alpha/ld/C 校验；alpha==0 时 A、B 不被引用
  -> alpha==0: 跳过 Phase 0/1，仅 combine 写零 C 的 m×n 有效区域
  -> Phase 0a ctrmm_split_kernel (AIV ×2): A、B 按列解交织为实虚平面
     （A 侧乘 2^-8 使 Cube 累加远离 fp32 上界）
  -> Phase 0b ctrmm_tri_kernel (AIV): 无效三角清零 + UNIT 对角置 1
     + trans=T/C 转置（±共轭）
  -> Phase 1  ctrmm_gemm_kernel (AIC ×4): Matmul API 实数 GEMM
     （K > 8192 自动分块，后续块原子累加）
  -> Phase 2  ctrmm_combine_kernel (AIV): 组合、×2^8 复原、alpha 乘、
     按列交织写回 C
```

该流程只在自有的 Ascend C kernel 之间组织数据流，不调用 CPU、
reference 或其他后端作为运行时 fallback。

#### 3.2.1 Host 侧设计

**参数检查。** 单入口顺序校验：handle 非空（返回
`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）→ side/uplo/trans/diag 合法枚举
（非法返回 `ACLBLAS_STATUS_INVALID_VALUE`）→ `m≥0`、`n≥0` →
`m==0 || n==0` 直接返回 SUCCESS（不访问指针、不校验 ld，BLAS 标准）→
alpha 非空 → `lda ≥ max(1,k)`、`ldb/ldc ≥ max(1,m)` → C 非空 →
alpha≠0 时 A/B 非空（`alpha=0` 时 A/B 不被引用、允许 nullptr）。

**alpha=0 快路径。** host 直接读 Host 内存 alpha 判零，跳过 Phase
0/1（不分配大 workspace），combine 阶段以零填充写出 C 的 m×n 有效
区域，ldc padding 行不触碰。

**形状与 tiling。** 复用 `CBlas3ResolveSymmShape`：`d = m|n`、
`aPadded = ceil64(d)`、GEMM 视角 `M=n, N=m, K=d`（列主序恒等式
`C^T = B^T·op(A)^T`，两个 packed 操作数天然就是行主序 Matmul 所需
的转置方向，无需任何转置标志）。四个 GEMM 共用一份
`CBlas3MakeSymmGemmTiling`（`side` 只决定操作数对中谁领先）：
`t1=G(首R,次R)`、`t2=G(首I,次I)`、`t3=G(首I,次R)`、`t4=G(首R,次I)`。
AIV/AIC 核数经 `GetAivCoreCount()/GetAicCoreCount()` 查询，各阶段
blockDim 取 `min(核数, 该阶段并行单元数)`。

**Workspace 布局。**（plain 四 GEMM 布局，全部 512B 对齐，
host 侧经 `CBlas3TryMulU64` 链式防溢出后 `EnsureDefaultWorkspace` 扩容）

```text
temp[0..3]  4 × (ceil128(m) × n) float32    GEMM 输出平面（C^T 行主序）
ar / ai     2 × (aPadded × aPadded) float32 A 的实/虚平面（缩放后）
br / bi     2 × (m × n) float32             B 的实/虚平面
4096 阶总计约 537MB，低于库 2GiB workspace 上限
```

**tiling key 规划。** 不使用 tiling key：各阶段组合信息（uplo/
trans/diag）作为 tiling 字段传入对应 kernel；所有分支依据均来自真实
入参，不依赖 case id 或输入值分布。

#### 3.2.2 Kernel 侧设计

各阶段职责（Phase 0a/1 为共享框架实现，0b/2 为本算子自有）：

| 阶段 | 核型 | 职责 | 关键处理 |
| --- | --- | --- | --- |
| `ctrmm_split_kernel`（共享 `cblas3::SplitBody`） | AIV 多核 | 列主序复数矩阵 → packed 实/虚平面 | 整列连续 DataCopyPad + GatherMask 解交织（4096 元素分块）；A 侧附加 2^-8 缩放；列间 round-robin 均衡 |
| `ctrmm_tri_kernel` | AIV 多核 | packed A 平面 → 稠密 op(A)^T 平面 | 64×64 tile：无效三角向量 Select 清零、UNIT 对角置 `2^-8`（对角 Select 覆盖，投毒对角不外泄）、trans=T/C 时 tile-pair Gather 转置（C 加虚部取负）；单个核完整处理一个 tile 对，无跨核序约束 |
| `ctrmm_gemm_kernel`（共享 `cblas3::GemmBody`，MatmulPlainBoth） | AIC 多核 | 单个实数 GEMM | Matmul API：128×128×64 静态 tile、L0 双缓冲、`SetSingleShape` 裁剪任意尾块（小 shape 无需特判）；tile 经 needed 计数后 round-robin 分配；K>8192 由 host 分块、后续块 enAtomic 累加 |
| `ctrmm_combine_kernel` | AIV 多核 | t1..t4 → C | 按列 1024 元素分块：`Pr=t1−t2`、`Pi=t3+t4`、`×2^8` 复原（幂次精确）、alpha 复数乘（α 分量为 0 时跳过该项避免 0·Inf=NaN）、Gather 交织写 `C[i + j·ldc]`，padding 行不动 |

**三角修正（Phase 0b）**在 packed 坐标系完成：拆平面后
`P(i,j) = A(j,i)`，即 `P` 的行主序视图恰为 `A^T`——正是 trans=N 所需
的 `op(A)^T`；uplo=UPPER 时 A 的未存三角落在 P 的严格上三角（i<j），
向量 Select 清零；`diag=UNIT` 将对角改写为 `1·2^-8`（存储值被覆盖，
NaN 对角不外泄）；trans=T/C 需要 `A`/`conj(A)`，即 trans=N 操作数的
转置，由 tile-pair 原地转置产生（零元与单位对角随转置保持，有效性
映射回 A 的三角约束）。

**原地安全（B ≡ C）**：B 只在 Phase 0a 被读取并物化到 Br/Bi 平面；
Phase 1 只读 workspace；Phase 2 只读 t1..t4、只写 C（trmm 无 beta 项，
永不读 C）。因此 `B ≡ C` 时写入 C 不会破坏尚未读取的 B 数据；除
`B ≡ C` 外，C 与 A/B 不允许内存重叠（host 不做重叠检测，遵循 cuBLAS
契约，由调用方保证）。

**性能路径说明**：Phase 1 为 CANN 官方 Matmul API 路径（与仓内
chemm/csymm/cherk/csyrk 同源），L1/L0 分块与双缓冲由库内静态 tiling
（128×128×64，fp32 下保持 L0A/L0B/L0C 双缓冲的最大块）调度；K 轴按
编译期 singleK=8192 分块。A/B 拆平面与 combine 为线性带宽开销
（4096 阶约 1.3GB 流量，≈3ms@400GB/s），相对 GEMM 主导的计算量
（4×2·k³ FLOP）占比小。任务书 5 个性能 case 均为大方阵，全程命中
该路径。

## 支持硬件

| 支持的芯片版本 | 本设计说明 |
| --- | --- |
| Atlas 800T A2（910B3，Atlas A2 训练系列） | 性能与精度验收设备；按 AIC Matmul API GEMM + AIV 预处理/组合混合流水设计 |
| Atlas 800I A2（Atlas A2 推理系列） | 同 arch22 数据流，随产品包构建回归 |
| Atlas A3 训练/推理系列产品 | 同 arch22 数据流（任务书 §2.2：A2/A3 对应架构目录 arch22），需按目标包构建回归 |

按任务书：自验证覆盖一种款型即可（精度建议 910B3 或 A3，性能建议
910B3）；验收阶段 Atlas 800T A2 与 Atlas 800I A3 均需完成。

## 算子约束限制

1. 仅支持 COMPLEX64；不支持 complex128 与实数版本（实数 trmm 由
   arch35 `aclblasStrmm` 提供，与本实现互不影响）。
2. 列主序存储，`lda ≥ max(1,k)`、`ldb/ldc ≥ max(1,m)`；超出
   ld 语义的非连续 Tensor 访问不支持。
3. `m=0` 或 `n=0` 为合法 no-op；`alpha=0` 时 A、B 不被引用，C 写零。
4. 只访问 `uplo` 指定三角区域的元素，另一三角与（UNIT 时）对角元
   素不被读取，其取值不影响结果；非有效区域的 NaN/Inf 不传播。
5. `B ≡ C` 原地合法；除此之外 C 与 A/B 内存重叠行为未定义。
6. 不涉及广播、视图返回与 dynamic shape 特殊处理（m/n 为运行时入
   参，由 host tiling 直接承载）。
7. 异步语义：依赖 `aclblasSetStream` 绑定 stream，读回结果前须同步
   stream；不保证浮点累加顺序逐位确定（确定性计算不要求）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 为 Netlib BLAS `ctrmm`（随测试工程提供）；对输出 C 全矩阵逐元素混合容差判定：`\|actual−golden\| ≤ atol + rtol·\|golden\|`，rtol = atol = 2^-13 (1.22e-4)，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2（或 32 ULP，大数规约误报时可放宽至 2ULP）；复数按实部/虚部分别套用 FLOAT32 容差 | 任务书 §3.2、生态算子开源精度标准 |
| 性能标准 | Atlas 800T A2、COMPLEX64、msprof op 有效采样 10 次平均：case1 256×256 L/U/N/NU ≤ 104.6us；case2 512×512 L/L/C/NU ≤ 235.2us；case3 1024×1024 L/U/T/U ≤ 803.8us；case4 2048×2048 R/L/N/NU ≤ 3241us；case5 4096×4096 L/U/C/NU ≤ 19931us | 任务书 §3.3 |

自测矩阵至少包含以下类别（CSV 列对齐 `test/chemm/chemm_test.csv`
风格：case_name、side、uplo、trans、diag、m、n、alpha_real/imag、
fill 模式、lda/ldb/ldc、expect_result、容差、random_seed；枚举大小
写不敏感解析）：

| 类别 | 覆盖 | 验证点 |
| --- | --- | --- |
| 枚举正交 | side×uplo×trans×diag 24 组合 | 三角补全、转置、共轭、对角规则 |
| 小 shape 基础 | m,n ∈ {1, 2, 3, 5, 7} | 最小 shape 正确性与 SetSingleShape 尾块裁剪 |
| shape 扫描 | 2 的幂、2 的幂 ±1、质数、非对齐值直至 4096 | 统一路径下的尾块裁剪、K 分块与 ld 偏移 |
| ld padding | lda/ldb/ldc 紧凑及多种 padding（k+1、k+8、2k） | 列主序寻址、padding 不触碰 |
| alpha 特殊值 | 0、1、−1、1+i、常规随机 | alpha=0 清零且不引用 A/B |
| 边界 | m=0、n=0、m=1、n=1、k=1 | no-op 与最小规模 |
| 负向 | 非法枚举×4、m<0、n<0、alpha/A/B/C nullptr、handle nullptr、ld 越界 | 对应状态码（INVALID_VALUE / HANDLE_IS_NULLPTR） |
| 特殊值 | A/B 含 Inf、NaN（有效三角内/外分别构造） | 对齐 cuBLAS 传播行为；无效区域不传播 |
| 原地 | B ≡ C 同指针 | 原地写出正确性 |
| 性能 | 任务书 5 case + 参考用例 | 达标耗时（msprof，5 次 warmup + 10 次采样） |

测试工程结构对齐仓内规范，随算子代码合入 `test/trmm/ctrmm/`：
`CMakeLists.txt`、`ctrmm_golden.h`（CPU 参考实现，Netlib ctrmm 语
义）、`ctrmm_param.h` 与 `arch22/{ctrmm_test.cpp, ctrmm_test.csv,
ctrmm_npu_wrapper.h}`；README 说明复现步骤。

## 兼容性分析

新增算子实现，不涉及历史版本兼容问题。接口签名与 cuBLAS
`cublasCtrmm` 参数一一对应（handle 及参数顺序一致），无需额外映射；
声明放入 `include/cann_ops_blas.h` 供其他产品线共用。本实现不修改
`blas/trmm/` 既有 arch35 `aclblasStrmm` 的任何行为，`blas/`
CMakeLists 的架构目录自动收集规则保证 arch22 源文件仅随 A2/A3 目标
构建，不影响 950 目标。

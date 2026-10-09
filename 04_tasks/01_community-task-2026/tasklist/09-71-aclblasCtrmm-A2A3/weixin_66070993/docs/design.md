# 需求背景（required）

## 需求来源

本设计对应 2026 年 9 月社区任务 `aclblasCtrmm`（A2/A3）。任务要求在昇腾 NPU（Atlas A2/A3 系列产品，性能验收设备 Atlas 800T A2 / 910B3）上，基于 Ascend C 编程语言、依托 ops-blas 开源仓工程框架，开发单精度复数（COMPLEX64）三角矩阵乘算子 `aclblasCtrmm`，功能与参数语义对齐 cuBLAS `cublasCtrmm`，完成设计、开发、测试全流程。验收通过后以 PR 形式合入 `blas/trmm/arch22/`，测试代码合入 `test/trmm/ctrmm/arch22/`，接口声明放入 `include/cann_ops_blas.h`。CANN 版本 9.1.0，工程模式为 kernel 直调（handle 绑定 stream 下发 NPU kernel）。

依据材料：任务书 `aclblasCtrmm_A2A3_task_doc.md`、配套测试用例目录（`trmm_test.csv` 1200 条用例、`gpu_baseline.csv` 200 条性能基线、`gen_csv.py`、`verify_accuracy.py`、测试指导 README）。

## 背景介绍

### aclblasCtrmm 算子功能

`aclblasCtrmm` 执行单精度复数三角矩阵-矩阵乘（BLAS Level-3），离席计算：

- `side = LEFT`：`C = alpha * op(A) * B`，A 为 m×m 三角矩阵
- `side = RIGHT`：`C = alpha * B * op(A)`，A 为 n×n 三角矩阵

`op(A)` 由 `trans` 决定：`OP_N`（A）、`OP_T`（Aᵀ）、`OP_C`（Aᴴ，复共轭转置）。A 仅 `uplo` 指定三角区域被引用；`diag=UNIT` 时主对角恒按 1 参与计算且不读取，`NON_UNIT` 时读取对角元素（乘法语义下允许为零，区别于 trsm 求解类，无回代、无对角非零强制）。输出 C 为一般矩阵，无 beta 项；允许 B≡C 原地写出（任务书 §2.5）。

### 标杆算子实现描述（Netlib cblas_ctrmm / cuBLAS cublasCtrmm）

精度 golden 由 Netlib BLAS `ctrmm`（cblas）生成。其参考流程为逐输出元素的三角 k 区间累加：对每个 `C[i,j]`，按 `side/uplo/trans` 确定 A 的有效 k 区间（如 side=LEFT、uplo=UPPER、trans=N 时 `k∈[i,m)`），执行复数乘加 `acc += op(A)[i,k]·B[k,j]`，`k` 命中对角且 `diag=UNIT` 时以 1 代替读取值，最终 `C[i,j]=α·acc`。**无效三角与 UNIT 对角从不被访问，其取值（包括 Inf/NaN）不影响结果**——这一"不引用"语义是本实现必须逐位对齐的标杆行为。cuBLAS `cublasCtrmm` 语义与其一致，并允许 `B≡C` 原地写出。

### 仓内现状与可复用资产

`blas/trmm/` 当前仅有面向 950（arch35）的实数实现 `aclblasStrmm`；arch22 目录与复数 trmm 均为空白。仓内可直接承接的工程资产：

| 层次 | 关键文件 | 现有职责 | 本算子复用点 |
| --- | --- | --- | --- |
| 接口声明 | `include/cann_ops_blas.h` | 全部 aclblas 接口对外声明（`aclblasStrmm` 在 L326 附近） | 新增 `aclblasCtrmm` 声明，签名与 `cublasCtrmm` 逐参数对应 |
| Host 基础设施 | `blas/common/helper/host_utils.h`、`aclblas_handle_internal.h` | AIV/AIC 核数查询、`EnsureDefaultWorkspace` 工作区扩容、handle/stream 解析 | 直接复用 |
| 复数 BLAS-3 共享框架 | `blas/common/helper/complex_blas3_arch22.h`、`complex_blas3_expand_arch22.h`、`complex_blas3_tiling_data.h`、`complex_blas3_host_utils.h` | chemm/csymm/cherk/csyrk/cher2k 共用三段式流水：Phase 0 按列 GatherMask 拆实/虚平面 → Phase 1 官方 Matmul API 实数 FP32 GEMM（128×128 tile、L0 双缓冲、K 按 8192 分块原子累加、SetSingleShape 任意尾块裁剪）→ Phase 2 重组写回 | Phase 0a 拆平面与 Phase 1 GEMM 整体复用；Phase 0b 按 trmm 语义新写（三角掩码/单位对角/转置共轭，参照 `MirrorTriBody` 的 tile 机制）；Phase 2 新写（无 beta 重组） |
| 同族参考 | `blas/trmm/arch35/`（strmm） | "三角补全 + GEMM + alpha 缩放"三段式、列主序 ld 寻址、alpha=0 清零语义 | trmm 语义处理的算法参照 |
| 复数向量算子 | `blas/trmv/arch22/ctrmv_*` | arch22 复数元素的 UB 搬运/解交织范式 | 小尺寸直算 kernel 的写法参照 |
| 测试框架 | `test/symm/csymm/`、`test/trmm/strmm/` | CSV 用例 + C++ GTest + golden + npu wrapper 的目录结构 | 新建 `test/trmm/ctrmm/` 对齐该结构，CSV 列兼容官方 `trmm_test.csv` |

### 参数语义（任务书 §2.4）

| 参数 | 方向 | 类型与约束 |
| --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t`，nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `side` | 输入 attr | LEFT/RIGHT，决定 A 阶数（LEFT 为 m×m、RIGHT 为 n×n），不得混淆 |
| `uplo` | 输入 attr | UPPER/LOWER，A 的有效三角 |
| `trans` | 输入 attr | N/T/C（C 为复共轭转置） |
| `diag` | 输入 attr | UNIT（对角视为 1 不读取）/ NON_UNIT |
| `m`、`n` | 输入 | C/B 行列数，非负；0 为合法 no-op |
| `alpha` | 输入 | COMPLEX64 标量指针（Host 内存），nullptr 非法；`alpha=(0,0)` 时 A/B 不被引用（允许为 nullptr），C 置零 |
| `A` | 输入 | COMPLEX64 Device，列主序，`lda ≥ max(1, side对应阶数)` |
| `B` | 输入 | COMPLEX64 Device，列主序 m×n，`ldb ≥ max(1, m)` |
| `C` | 输出 | COMPLEX64 Device，列主序 m×n，`ldc ≥ max(1, m)`；允许 B≡C 原地，除此之外 C 与 A/B 不允许内存重叠 |

非法枚举、非法 m/n/ld、空指针统一返回 `ACLBLAS_STATUS_INVALID_VALUE`（任务书 §2.4/§2.5 及官方 TC_ED 用例期望）。

# 需求分析（required）

## 需求描述

在 `blas/trmm/arch22/` 以 kernel 直调方式实现 `aclblasCtrmm`：handle 绑定 stream 异步下发，覆盖 `side×uplo×trans×diag` 全部 24 组枚举、列主序与前导维（含 padding）、`m/n≥0` 全尺寸、`alpha` 特殊值、B≡C 原地语义、边界与负向用例；精度满足 COMPLEX64 混合容差标准（rtol=atol=2^-13，matched_ratio≥0.99，max_abs_error≤1e-2 或 32ULP），性能在 910B3 上满足任务书 §3.3 的 5 个基准 case 达标耗时，全部性能用例按测试指导的 `NPU 平均单次 kernel 耗时 ≤ gpu_ms/0.8` 口径比对。

## 需求拆解

1. **接口与校验闭环**：`include/cann_ops_blas.h` 新增声明；host 按「host 侧设计」的顺序校验，校验失败返回对应状态码、不发起 kernel。关键顺序约束：官方用例 `m=0`/`n=0` 携带 `lda=ldb=ldc=0` 且期望 SUCCESS，故零维 quick return 必须位于前导维与指针校验之前。
2. **三角语义闭环**：仅引用 `uplo` 指定区域；`UNIT` 不读对角（对角毒化不影响结果）；`trans=T/C` 在数据准备阶段消化，主计算归一为普通复数矩阵乘。
3. **性能闭环**：计时设计口径取更严格的"一次 API 调用的全部 kernel 累计耗时"（同时满足测试指导的按 kernel 名平均口径）。5 个基准 case（256–4096²）必须走 Cube 路径；小尺寸性能用例（1–64²，基线 3.4–9.7μs，预算即基线÷0.8 ≈ 4.3–12μs）无法容纳多 kernel 流水，必须有单 kernel 低延迟路径。
4. **精度闭环**：golden 为 Netlib `cblas_ctrmm`，全 m×n 逐元素判定；复数按实/虚部分别套 FLOAT32 容差。
5. **特殊值语义闭环**（本设计重点分析项，见「特殊值语义分析」）：官方特殊填充用例（B 全 Inf/NaN、±FLT_MAX 极端值）全部位于 32×32 小尺寸，但任务书声明输入取值为 COMPLEX64 全集，实现须对任意尺寸的非有限输入保持"不引用即不影响"的标杆语义。
6. **测试闭环**：`test/trmm/ctrmm/arch22/` 交付 CSV（直接兼容官方 1200 条）+ GTest + golden + npu wrapper + 补充用例。

# 详细设计（required）

## 算子分析

### 数学公式

记 A 阶数 `kA`（LEFT 为 m，RIGHT 为 n），列主序 `A(i,j)=A[i+j·lda]`：

```
side=LEFT : C = alpha · op(A) · B      op(A) ∈ C^{m×m}
side=RIGHT: C = alpha · B · op(A)      op(A) ∈ C^{n×n}
op(A) = A (N) | Aᵀ (T) | Aᴴ (C)
```

引入存储掩码 `s(i,j)`（`uplo=UPPER` 时 `s=1 ⟺ j≥i`；`LOWER` 时 `s=1 ⟺ j≤i`）与对角规则 `d(i)`（`UNIT` 时 `d(i)=1` 且不读存储值；否则 `d(i)=A(i,i)`），op(A) 的稠密展开为：

```
op(A)(i,j) = | d(i)                          , i=j
             | s(i,j)·A(i,j)   (N)           , i≠j
             | s(j,i)·A(j,i)   (T)
             | s(j,i)·conj(A(j,i)) (C)
```

复数乘法：`(a+ib)(c+id) = (ac−bd) + i(ad+bc)`；alpha 复数缩放同式。

### 支持数据类型

| dtype | 元素宽度 | 计算/搬运语义 |
| --- | --- | --- |
| COMPLEX64（`aclblasComplex`） | 8B（fp32 实部 + fp32 虚部，AoS 交织存储） | 三段流水在 Phase 0a 解交织为实/虚两个 fp32 平面（AoS→SoA），GEMM 以 fp32 计算，Phase 2 复数 alpha 缩放后交织写回 complex64；小尺寸直算路径直接按 8B 元素搬运，实虚配对运算不拆散 |

### 支持形状

| 矩阵 | 逻辑形状 | 寻址（列主序） |
| --- | --- | --- |
| A | side=LEFT：m×m；side=RIGHT：n×n（阶数 kA） | `A[i + j·lda]`，仅 uplo 指定三角被引用，`lda ≥ max(1,kA)` |
| B | m×n | `B[i + j·ldb]`，`ldb ≥ max(1,m)` |
| C | m×n | `C[i + j·ldc]`，`ldc ≥ max(1,m)`，仅写前 m 行有效区，padding 行不触碰 |

`m,n ≥ 0`（0 为合法 no-op）；官方用例尺寸范围：精度 [1,2048]、性能 [1,4096]，含奇数、边界、2 的幂±1、非对齐值与 lda/ldb/ldc 紧凑及 padding 场景。不支持超出 ld 语义的非连续访问；不涉及广播与 dynamic shape 特殊处理。

### 支持参数组合

`side(2) × uplo(2) × trans(3) × diag(2)` = 24 组枚举组合全部合法；`alpha` 为 COMPLEX64 全集（含 0/Inf/NaN）。uplo/trans/diag 作为 tiling 字段下发，kernel 侧不因组合数产生分支爆炸；`side` 仅决定 host 布局（paired/plain，见下文）。与标杆 `cublasCtrmm` 相比任务书范围内**无功能缺失**（参数序列一一对应、B≡C 原地同等支持）；cuBLAS 的 batched/64-bit 变体等形态超出任务书范围，不涉及。

### 列主序归约恒等式与 GEMM 编排（与 csymm 同构）

取等式两边转置：

```
side=LEFT : Cᵀ = Bᵀ · op(A)ᵀ
side=RIGHT: Cᵀ = op(A)ᵀ · Bᵀ
```

共享框架 Phase 0 的 packed 平面按"列主序复数 → 逐列解交织"产出，其行主序视图恰是逻辑矩阵的转置：B packed 即 Bᵀ，A packed 即 Aᵀ。因此在 GEMM 视角（M=n, N=m, K=kA）下，两侧操作数天然处于所需方向，**四次实数 GEMM 均无需转置标志**（与 chemm/csymm 完全同构）：

```
side=LEFT : t1=G(Br,Ar) t2=G(Bi,Ai) t3=G(Bi,Ar) t4=G(Br,Ai)   K=m
side=RIGHT: t1=G(Ar,Br) t2=G(Ai,Bi) t3=G(Ai,Br) t4=G(Ar,Bi)   K=n
Pr = t1 − t2 , Pi = t3 + t4 , Cᵀ = Pr + i·Pi（Phase 2 按列重组）
```

trmm 与 csymm 的全部差异收敛到 Phase 0b：csymm 把三角按"对称镜像"补全，trmm 把非存储三角清 0、按 UNIT 置对角、按 trans 做转置/共轭。

`trans` 语义映射：packed Aᵀ 平面记作 P。

- `trans=N`：`op(A)ᵀ = Aᵀ = P`，P 经掩码+对角修正后直接使用；
- `trans=T`：`op(A)ᵀ = A = Pᵀ`，需将 P 转置一份；
- `trans=C`：`op(A)ᵀ = conj(A) = conj(Pᵀ)`，转置并取负虚部。

掩码规则与 trans 解耦：掩码永远在 P（即 Aᵀ）坐标系施加——`uplo=UPPER` 时 A 有效区 `k≥i`，等价于 P 的严格下三角（`row≥col`），零化其严格上三角；`LOWER` 反之。`UNIT` 时 `P(k,k)=scale·1`（见「数值范围缩放」），不读取存储对角（毒化/NaN 对角不外泄）。

### side=LEFT 的配对 K 快路径（复用 csymm paired-K 布局）

csymm 在 side='L' 时启用 paired-K 布局：左操作数直接读调用方 B（列主序复数天然就是 K 维实虚交替的 packed 形态，行距 2·ldb），右操作数为交错打包的 A 对 `[Ar | −Ai]`、`[Ai | Ar]`，仅 2 次 GEMM：

```
Pr = B·[Ar | −Ai] = Σ(Ar·Br − Ai·Bi)
Pi = B·[Ai |  Ar] = Σ(Ai·Br + Ar·Bi)
```

收益：B 完全不拆（省一遍 m×n 复数的读+两遍写），GEMM 启动次数减半（由 4 次降为 2 次，K 维翻倍为 2kA、总 FLOP 不变），workspace 少一对 B 平面；且每个复数乘加的两项在累加器内相邻相消，数值行为优于"先各自累加再相减"（共享框架注释中实测：`Inf−Inf=NaN` 的极端用例正是靠此通过）。本算子 side=LEFT 同样满足前提（B 是左操作数、A 是需修正的三角矩阵），**沿用同一布局**：Phase 0b 对 A 的两个配对平面分别做 trmm 修正即可。

side=RIGHT 时 A/B 角色互换，A 作为左操作数需要在 fast 轴上交错，无法用配对布局，采用 plain 4-GEMM（与 csymm 一致）。5 个任务书基准 case 中 4 个为 LEFT，快路径覆盖主性能面。

### 数值范围缩放（沿用 CBLAS3_RANGE_SCALE）

Phase 0a 对 A 侧 packed 平面乘 `2^-8`，Phase 2 combine 乘 `2^8` 复原。因子为 2 的幂，往返位级精确；作用是使 Cube 累加远离 fp32 上界——官方 `RANDOM_EXTREME` 填充（±FLT_MAX）场景下，未缩放的操作数在累加中途溢出，缩放后可完整复原。对角 UNIT 写入 `1·2^-8`，复原后恰为 1。

### 特殊值语义分析（关键正确性问题）

**问题**：稠密 GEMM 中"非存储三角清 0"是真实参与乘加的 0；若 B 中含 `Inf/NaN`，`0×Inf=NaN` 会污染本不被 golden 引用的乘积项——Netlib ctrmm 的 k 循环只遍历有效三角区间，被掩码位置对应的乘积在 golden 中根本不存在。这是 trmm 区别于 symm/hemm 的特有 hazard（后者补全的是真实数据，非结构 0），同样的问题也存在于 alpha 乘（combine kernel 处理）。

**输入事实**：官方 1200 条用例中全部 `Inf/NaN/EXTREME/ALTER` 特殊填充集中在 32×32 小尺寸；所有 >64 尺寸均为有限随机数。

**设计对策（三层）**：

1. **小尺寸直算路径**（见「小尺寸直算 kernel」）：`max(m,n)` 小于阈值（默认 ≤128，host 常量）或 GEMM 输出块数不足 AIC 核数时，走单 AIV kernel 逐元素按有效 k 区间累加——k 循环范围与 golden 完全一致，非有限值语义天然正确，同时满足小尺寸延迟预算（单 kernel、无 workspace 申请）。官方全部特殊值用例（32×32）落在该路径。
2. **非有限检测兜底（任意尺寸）**：B 侧非有限检测在 plain 布局下融合进 Phase 0a 拆 B（同 pass 向量比较），paired 布局下由独立 `ctrmm_detect_kernel` 对 B 做一遍纯带宽扫描（4096² 约 0.4ms，<3% 该 case 预算）；Phase 0b 修 A 时检测 A **存储区域内**的非有限值（掩码区/UNIT 对角属"不引用"，毒化不误报）。结果写 workspace 标志字。后续 kernel 启动时读标志：置位则 GEMM 早退、combine 切换到与直算路径同一实现的逐元素回退逻辑。**无需 host 同步**（标志未置位时仅各 kernel 一次 GM 读），保证任意尺寸非有限输入语义完整正确。
3. **combine 的 alpha 特殊值**：严格按 golden 同形公式 `C=α·(Pr+i·Pi)` 计算，不做"分量跳项"等改变 Inf/NaN 传播路径的化简；alpha=(0,0) 场景由 host 短路规则处理（A/B 不被引用）。

## 算子实现

### 实现方案

整体执行流程（kernel 直调，全部 kernel 于 handle 绑定 stream 顺序下发，函数返回即入队完成；读回前由调用方同步）：

```
aclblasCtrmm
  ├─ 校验（见 host 侧设计顺序）→ 失败即返回状态码
  ├─ m==0||n==0 → SUCCESS（不读指针/ld，BLAS 标准）
  ├─ host 读 alpha（Host 内存，无 D2H 同步）：
  │    alpha=(0,0) → ctrmm_zero_kernel 写零 C 有效区返回（A/B 允许 nullptr）
  ├─ 小尺寸 → ctrmm_direct_kernel（单 AIV kernel）
  └─ 大尺寸 → 三段流水：
        Phase 0a  ctrmm_split_kernel（共享 SplitBody）：A、B → packed 实/虚平面，
                  A 侧 ×2^-8；plain 布局下 B 侧顺带非有限检测 → ws.flag
                  （paired 布局 B 不拆，改由 ctrmm_detect_kernel 独立扫描 B → ws.flag）
        Phase 0b  ctrmm_tri_kernel（自有）：A 平面三角掩码清零 + UNIT 置 1·2^-8
                  + trans=T/C tile 对转置（C 加虚部取负）+ A 存储区非有限检测
        Phase 1   ctrmm_gemm_kernel×4/×2（共享 GemmBody + Matmul API）：
                  paired（LEFT）或 plain（RIGHT）；flag 置位早退
        Phase 2   ctrmm_combine_kernel（自有）：flag 置位→逐元素直算回退；
                  否则 Pr=t1−t2、Pi=t3+t4、×2^8、复数 alpha 乘、按列交织写 C
```

该流程只在自有 Ascend C kernel 与共享框架 kernel 之间组织数据流，不调用 CPU、reference 或其他后端作为运行时 fallback。

**备选方案分析（分块递归 trmm）**：教科书分块算法将 A 按 2×2 递归切分，非对角块为纯矩形 GEMM（无结构 0，`0×Inf` 问题从构造上消失），且只计算有效项（总 FLOP 约为稠密掩码的一半，其中非对角块的 GEMM 形状 `M=n,N=k',K=k'` 与 csymm 完全同构，叶子可复用直算 kernel）。本设计仍选择稠密掩码流水，原因：① 仓内五个复数 BLAS3 算子的既定范式是单趟三段流水，与其同构可获得最大的代码与评审复用度；② 分块方案的非对角块编排（块树枚举、多级 tiling、跨级 temp 原子累加）是仓内没有先例的新组件，实现与维护成本更高；③ 五级流水在小尺寸 case 的多次下发开销不利，而本任务的性能预算在大尺寸段稠密路径已满足；④ `0×Inf` 风险已由三层对策覆盖，不构成阻塞。若后续实测 4096² 预算不足，可分块化作为降载预案（每加一级递归约省 25% FLOP）保留扩展空间。

#### 3.2.1 host 侧设计

**接口声明**（`include/cann_ops_blas.h`，与 `cublasCtrmm` 逐参数对应，任务书 §2.3）：

```cpp
aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc);
```

**校验顺序**（与官方 TC_ED 用例逐条对齐；任一失败即返回）：

1. `handle==nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `side/uplo/trans/diag` 非法枚举 → `ACLBLAS_STATUS_INVALID_VALUE`（官方用例 expect_result 明确为 INVALID_VALUE，测试按整型精确比对状态码）；
3. `m<0 || n<0` → `ACLBLAS_STATUS_INVALID_VALUE`；
4. `m==0 || n==0` → `ACLBLAS_STATUS_SUCCESS`（官方用例 lda/ldb/ldc=0 亦成功，故先于 ld/指针校验返回）；
5. `alpha==nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；
6. `lda ≥ max(1, kA)`、`ldb/ldc ≥ max(1,m)` → 否则 `ACLBLAS_STATUS_INVALID_VALUE`；
7. `C==nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；
8. host 读 alpha：`alpha=(0,0)` → 写零路径（A/B 不检查、允许 nullptr）；否则 `A/B==nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`。

**路径分派**：`max(m,n) ≤ CTRMM_DIRECT_MAX（默认 128）` 或 `GEMM 输出块数 < AIC 核数` → 直算 kernel；否则三段流水。分派仅依赖尺寸与核数，不读取 case 名/输入数值。

**Tiling**：复用 `CBlas3ResolveSymmShape`（`d=kA`、`aPadded=ceil64(d)`、GEMM 视角 M=n,N=m,K=d）与 `CBlas3MakeSymmGemmTiling` / `CBlas3MakeSymmPairedGemmTiling`；自有 tiling 结构 `CtrmmTilingData` 追加 `uplo/trans/diag` 编码、`alphaRe/alphaIm` 快照、非有限标志偏移、直算路径核间切分参数。地址/容量一律 `CBlas3TryMulU64` 链式防溢出。

**Workspace**（`EnsureDefaultWorkspace` 统一申请，512B 对齐）：

```
plain（RIGHT 及 LEFT 回退）:
  temp[0..3]  4×(tempLdc×n) fp32     t1..t4（tempLdc=ceil128(m)）
  ar/ai       2×(aPadded²)   fp32     A packed 平面（×2^-8）
  br/bi       2×(m×n)        fp32     B packed 平面
paired（LEFT 主路径）:
  temp[0..1]  2×(tempLdc×n)           Pr/Pi
  aPair[0..1] 2×(2·aPadded²)          [Ar|−Ai]、[Ai|Ar]
flag          1×4B                     非有限检测标志
4096² plain 合计 ≈ 537MB，远低于库 2GiB 上限；paired 更小。
```

##### 1. 分核策略

- 全部 AIV kernel（split/tri/combine/direct/detect/zero）：按"列"或"等长元素块"将并行单元均分到各核，`blockDim = min(AIV 核数, 并行单元数)`，列/块 round-robin 分配，余数由前几个核各多承担一个单元（与共享框架 `SplitBody` 的列间均衡方式一致）；
- Phase 1 GEMM：`blockDim = min(AIC 核数, tile 数)`，tile 按 needed 计数 round-robin 分配到各 AIC 核（复用 `GemmBody` 的分配逻辑）；
- 小尺寸直算 kernel：输出按列均分；`n` 小于核数时按行块做二维切分，保证小 n 下核利用率。

##### 2. 数据分块和内存优化策略

- **UB（LocalMemory）**：各 AIV kernel 均为流式分块处理，单核 UB 占用按"块大小×元素宽×缓冲数"估算，均远低于 arch22 单核 UB 容量（192KB）：split 按 4096 元素列块（`4096×4B×2buf≈32KB`，加 GatherMask 解交织暂存）；tri 按 64×64 tile 对（`2×64×64×4B×2平面≈64KB`，含转置暂存）；combine 按 1024 元素块（`1024×4B×4平面≈16KB`，加交织暂存）；direct 为 UB 驻留的 A/B 列块（`64列×块高×8B` 每矩阵，≤64KB）。
- **L1/L0**：实数 GEMM 的 128×128×64 tile 与 L0 双缓冲由 Matmul API 静态 tiling 承载（fp32 下保持 L0A/L0B/L0C 双缓冲的最大块），本算子不直接管理；K 轴 >8192 由 host 分块、后续块原子累加。
- **GM workspace**：`plain = 4·ceil128(m)·n·4B + 2·ceil64(kA)²·4B + 2·m·n·4B`；`paired = 2·ceil128(m)·n·4B + 2·(2·ceil64(kA)²)·4B + 4B(flag)`；4096² plain ≈537MB < 库 2GiB 上限，全部经 `CBlas3TryMulU64` 防溢出后由 `EnsureDefaultWorkspace` 一次申请。

##### 3. tilingKey 规划策略

不使用独立 tilingKey 派发 kernel：`uplo/trans/diag` 以枚举字段编入 `CtrmmTilingData`，kernel 内运行时分支；路径选择（直算/三段流水、paired/plain、非有限回退）由 host 按真实入参（尺寸、核数）与 device 侧标志位决定，全部依据运行时信息，不按 case 名或输入数值分布特化。

#### 3.2.2 kernel 侧设计

| kernel | 单元 | 职责 | 要点 |
| --- | --- | --- | --- |
| `ctrmm_split_kernel`（共享 `SplitBody`） | AIV | 复数矩阵 → packed 实/虚平面 | 整列 DataCopyPad + GatherMask 解交织（4096 元素块）；A 侧 `scaleEnabled` ×2^-8；B 侧同 pass 追加 `IsInf/IsNan` 规约写 flag；列间 round-robin |
| `ctrmm_tri_kernel`（自有，参照 `MirrorTriBody`） | AIV | packed A → 稠密 op(A)ᵀ | 64×64 tile 对处理：非存储三角 Select 清零、UNIT 对角覆写 1·2^-8、`trans=T/C` 时 tile 对 Gather 转置（C 虚部取负）、存储区非有限检测写 flag；单核对一个 tile 对，无跨核序 |
| `ctrmm_gemm_kernel`（共享 `GemmBody`/`MatmulPlainBoth`） | AIC | 实数 FP32 GEMM | 128×128×64 静态 tile、L0 双缓冲、`SetSingleShape` 尾块裁剪、K>8192 host 分块原子累加；启动先读 flag，置位早退 |
| `ctrmm_combine_kernel`（自有） | AIV | t*→C 或直算回退 | flag 置位→每核对分配输出元素执行与直算路径同构的逐元素计算；否则按列 1024 元素块：`Pr=t1−t2`、`Pi=t3+t4`（paired 直接读 t1/t3）、×2^8、`C=α·P` 复数乘、Gather 交织写 `C[i+j·ldc]`，padding 行不写 |
| `ctrmm_direct_kernel`（自有） | AIV | 小尺寸全功能 | 见「小尺寸直算 kernel」 |
| `ctrmm_detect_kernel`（自有） | AIV | paired 布局 B 侧非有限扫描 | 纯带宽向量比较规约写 flag，仅 paired 路径使用 |
| `ctrmm_zero_kernel`（自有） | AIV | alpha=0 写零 | 仅写 C 前 m 行有效区，不触碰 ldc padding、不读 A/B |

**tri kernel 细节**：在 packed（Aᵀ 视图）坐标系工作。`uplo=UPPER` 时 A 有效区对应 P 的下三角含对角——零化严格上三角（`row<col`）；`LOWER` 反之。对角 tile 单独处理：按 `diag` 覆写存储对角为 `scale` 值（UNIT）或保留（NON_UNIT），再清严格三角。`trans=N` 时掩码后直接以 P 为 GEMM 右操作数（plain）或写入配对平面（paired）；`trans=T/C` 时按 64×64 tile 对原地/成对转置后再掩码（两顺序等价，取访存更优者实现），`OP_C` 在同 pass 对虚部平面取负——转置+共轭+掩码一趟完成，不增加数据搬运趟数。

**Ascend C 实现流程图**：各 kernel 均为标准 `Init → Process(CopyIn/Compute/CopyOut)` 结构——AIV kernel 按分核策略取列/块区间，循环 `{ DataCopyPad(GM→UB) → Select/Gather/复数乘加 → DataCopyPad(UB→GM) }`；AIC `ctrmm_gemm_kernel` 为 `Init(读 flag) → Matmul::Iterate(128×128×64, SetSingleShape 裁剪, K 分块 enAtomic 累加)`；算子级串行流水线见上文「实现方案」总图。

**与标杆实现流程的差异点及原因**：Netlib/cuBLAS ctrmm 的参考流程为逐输出元素的三角 k 区间累加（CPU 串行/线程化循环，天然不访问无效三角）；本设计为利用 AIC cube 单元的矩阵吞吐，将其等效变换为"三角掩码稠密化 + 实数 GEMM 分解 + 平面重组"的三段流水。差异带来两处必须显式处理的语义代价：① 掩码 0 与非有限 B 的 `0×Inf=NaN` 污染（由「特殊值语义分析」三层对策覆盖）；② 稠密化后 fp32 累加范围（由 2^-8 缩放覆盖）。小尺寸与任意尺寸非有限输入由直算/回退路径保证与标杆 k 区间语义逐位一致。

### 小尺寸直算 kernel

单 kernel 完成全部 24 组合的语义，输出按列分给各核（列内行连续、访存连续）：

```
每核负责若干输出列 j（或 (i,j) 块）:
  for j, for i in 核内行块:
    klo/khi = 有效 k 区间（side/uplo/trans 决定；含对角）
    acc = (0,0)
    for k in [klo, khi):
      aik = op(A)[i,k]   （trans 下标映射；k==i 且 UNIT → (1,0)，不读存储值）
      acc += aik · B[k,j]  （复数乘加，实虚各 4 项）
    C[i,j] = α · acc
```

A/B 经 DataCopyPad 搬 UB 复用；k 区间严格等于 golden 的有效范围，非有限值、零对角、毒化对角/毒化掩码区语义与 Netlib 逐位对齐。同一实现以"回退模式"被 combine kernel 复用（「特殊值语义分析」第 2 层），代码单份。

### 分支条件与合法组合

24 组 `side×uplo×trans×diag` 全部合法：`uplo/trans/diag` 编码进 tiling，kernel 内运行时分支；`side` 决定 host 布局（paired/plain）与 GEMM 操作数顺序；`alpha=0`、零维、小尺寸由 host 短路。dtype 固定 COMPLEX64。

### B≡C 原地安全性

plain 布局：B 在 Phase 0a 被物化到 br/bi 平面，之后不再被读；paired 布局：B 仅在 Phase 1 被 GEMM 读。Phase 2 只读 workspace temp、只写 C（trmm 无 beta，永不读 C）。stream 顺序保证"B 的全部读取先于 C 的任意写入"，故 `B≡C` 原地合法且无需任何特殊路径；其余重叠按 cuBLAS 契约由调用方保证（host 不做重叠检测）。

### 内存与步长

- 矩阵列主序，`A(i,j)=A[i+j·lda]` 等；支持 `lda/ldb/ldc` 表达的任意 padding，不扩展任意 stride。
- GM workspace 仅算子生命周期内有效，经 handle 的 `EnsureDefaultWorkspace` 申请；4096² plain ≈537MB < 2GiB 上限。
- 输出仅写 C 的前 m×n 有效区，ldc padding 行/列不触碰（哨兵检测可通过）。
- 确定性不要求（fp32 累加顺序不保证逐位一致），精度按混合容差判定。

## 支持硬件

| 支持的芯片版本 | 本设计说明 |
| --- | --- |
| Atlas 800T A2（910B3，arch22） | 精度/性能验收设备；按 AIC Matmul API GEMM + AIV 预处理/组合混合流水设计 |
| Atlas 800I A2（arch22） | 同 arch22 数据流，随产品包构建回归 |
| Atlas 800I A3（arch22） | 同 arch22 数据流（任务书 §2.2：A2/A3 对应架构目录 arch22），自验后按目标包构建回归 |

按任务书：自验证覆盖一种款型即可（精度建议 910B3 或 A3，性能建议 910B3）；验收阶段 Atlas 800T A2 与 Atlas 800I A3 均需完成。

## 算子约束限制

1. `handle` 空 → `HANDLE_IS_NULLPTR`；非法枚举、`m/n<0`、空 `alpha/A/B/C`（alpha≠0 时）、非法 ld → `INVALID_VALUE`。
2. `m=0||n=0` 合法 no-op（先于 ld/指针校验返回）；`alpha=(0,0)` 时 A/B 不被引用（可为 nullptr），C 置零。
3. 仅 COMPLEX64；列主序；无广播、无视图返回、无 dynamic shape 特化。
4. 三角语义：仅引用 `uplo` 指定区域；UNIT 不读对角；非存储区 NaN/Inf 不传播。
5. B≡C 原地合法；其余 C 与 A/B 重叠行为未定义。
6. 异步语义：`aclblasSetStream` 绑定 stream；读回前须同步。
7. 接口声明放入 `include/cann_ops_blas.h`，不设私有平行接口。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 为 Netlib `cblas_ctrmm`（随测试工程提供），对输出 C 全 m×n 逐元素混合容差判定：`\|actual−golden\| ≤ atol + rtol·\|golden\|`，rtol=atol=2^-13 (1.22e-4)，matched_ratio≥0.99，max_abs_error≤1e-2 或 32ULP（大数规约误报可按任务书放宽至 2ULP）；复数实/虚部分别套 FLOAT32 容差，Inf/NaN 单列匹配。随机填充以官方 `random_seed` 为准，均匀/正态各 50% | 任务书 §3.2、生态算子开源精度标准 |
| 性能标准 | Atlas 800T A2（910B3），msprof op Device 计时，5 次 warmup（msprof 自带）+ 有效采样 >10 次取平均；判定 `NPU 平均单次 kernel 耗时 ≤ gpu_ms/0.8`（本设计以更严格的"单次调用全部 kernel 累计"为内部设计口径） | 任务书 §3.3、测试指导 README |

任务书 5 个基准 case 及本设计路径预算粗估（设计值，非实测）：

| 基准 case | shape | 达标耗时(μs) | 本设计路径预算粗估 |
| --- | --- | --- | --- |
| TC_PF_1001 | 256² L/U/N/NU | 104.6 | paired 流水：split A + tri + 2×GEMM（各 ~10μs 级）+ combine |
| TC_PF_1002 | 512² L/L/C/NU | 235.2 | paired：2×GEMM（K_eff=1024）合计 8·512³≈1.1GF → GEMM 约百 μs 内 |
| TC_PF_1003 | 1024² L/U/T/U | 803.8 | paired：合计 8·1024³≈8.6GF + A 转置一趟（~0.2ms） |
| TC_PF_1004 | 2048² R/L/N/NU | 3241 | plain 4×GEMM：合计 8·2048³≈69GF → ~2ms 内 |
| TC_PF_1005 | 4096² L/U/C/NU | 19931 | paired：2×GEMM（K_eff=8192）合计 8·4096³≈550GF + A 拆/修/转置 + B 检测 ~1GB 流量 |

小尺寸（1–64²，预算 4.3–12μs）：单 kernel 直算，无 workspace、无多段流水开销——流水路径一次调用至少 3–5 个 kernel 的下发与执行开销在该预算区间内不可行，故由 host 路径分派在结构上规避（同时兼容按 kernel 名统计与全 kernel 累计两种计时口径）。

## 自验用例覆盖

以官方 `trmm_test.csv` 1200 条（L0 枚举 24 + SQ 尺寸 23 + AB 标量 24 + WS/TH 非方阵 24 + LD padding 12 + FL 填充 13 + CV 覆盖 24 + ED 边界 23 + EX 扩展 833 + PF 性能 200）为主集，另补：

- 全 24 组合 × `B≡C` 原地（含矩形、padding、块边界）；
- `alpha=(0,0)` + 空 A/B、A/B 含 NaN/Inf、旧 C 毒化 → 有效区全零；
- UNIT 对角毒化、非引用三角毒化、ldc padding 哨兵；
- 非方阵 LEFT/RIGHT 对称形状（验证 A 阶数不混淆）；
- 大尺寸（>直算阈值）含 Inf/NaN 输入 → 验证非有限回退路径正确性（官方集未覆盖，任务书 §3.5 要求自补）；
- 非默认 stream、并发调用。

测试工程：`test/trmm/ctrmm/arch22/`（`ctrmm_test.cpp`、`ctrmm_test.csv`、`ctrmm_npu_wrapper.h`），`test/trmm/ctrmm/`（`CMakeLists.txt`、`ctrmm_golden.h`、`ctrmm_param.h`）；CSV 列与官方文件一一对应（`alpha_real/alpha_imag`、`nullA/nullB/nullC`、`a_fill/b_fill`、`expect_result`、`mere_threshold`、`mare_multiplier`、`random_seed`），枚举大小写不敏感解析。

## 兼容性分析

新增接口与新增目录，不修改任何既有接口行为；`blas/trmm/arch35` 实数实现互不影响；arch22 源文件仅随 A2/A3 目标构建。接口签名与 `cublasCtrmm` 参数一一对应（handle 及参数顺序一致），无需额外映射。

# 交付与合入

| 交付件 | 说明 |
| --- | --- |
| 设计文档 | 本文档，经 PR 评审合入 tasklist 对应目录 |
| 实现代码 | `blas/trmm/arch22/`（个人 fork 仓，邀 Ascend-CANN 为开发者） |
| 测试代码 | `test/trmm/ctrmm/arch22/` + `test/trmm/ctrmm/`（readme 说明复现步骤，覆盖全部官方用例并标注精度/性能 case） |
| 自测报告 | `task_submission/` 目录按任务书 §4 模板提交：1 自验证步骤说明.md、2.1 精度自验证报告.xlsx、2.2 精度自验证日志.log、3.1 性能自验证报告.xlsx、3.2 性能自验证日志.log、4.1 内存自验证报告.xlsx、4.2 内存自验证日志.log（精度/性能含用例参数、对比结果与截图、有效采样 >10 次平均值） |
| 算子 README | `blas/trmm/` 下按 ops-blas 规范编写 README，产品支持表标注 Atlas A2/A3 系列产品（含 Atlas 800I A2/A3）：支持 |

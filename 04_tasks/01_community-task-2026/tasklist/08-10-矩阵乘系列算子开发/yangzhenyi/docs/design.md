# 需求背景
## 需求来源
在科学计算、量子物理模拟、信号处理以及深度学习复数神经网络等场景中，复数矩阵乘法是一类基础且高频的核心算子。为了完善昇腾 CANN 生态中的基础数学计算库（BLAS Level-3），需要在昇腾 NPU 硬件上提供高性能的复数单精度厄米特矩阵乘法算子 `chemm`。
## 背景介绍
`chemm`（Complex Hermitian Matrix Multiply）执行复数单精度厄米特矩阵与通用稠密矩阵的乘加运算。参考标准 Netlib BLAS 规范及 NVIDIA cuBLAS 库中的 cublasChemm 接口，在昇腾 NPU 上基于 Ascend C 编程语言实现软硬件协同优化的高性能 chemm 算子，合入昇腾算子开源仓 ops-blas。
- 开源仓地址：https://gitcode.com/cann/ops-blas
- 参考实现与标准
	- Netlib BLAS `chemm`：https://netlib.org/blas/blasqr.pdf
    - cuBLAS `cublas<t>hemm`:https://docs.nvidia.com/cuda/cublas/index.html#2.7.7
## chemm算子实现现状分析
目前昇腾生态在复数 BLAS 基础算子库方向正处于原生 Ascend C 高性能重构阶段。针对 `chemm` 算子的参数规格与功能分析如下：
| 参数 | 参数含义 | 输入/输出/属性 | 数据类型 | 格式约束 | 形状 (Shape) |
|------|----------|----------------|----------|----------|--------------|
| `side` | 左乘或右乘 | 属性 | `char` | 仅支持 `'L'`, `'R'` | - |
| `uplo` | 上三角或下三角 | 属性 | `char` | 仅支持 `'U'`, `'L'` | - |
| `m` | 矩阵 C 的行数 | 属性 | `int` | $m > 0$ | - |
| `n` | 矩阵 C 的列数 | 属性 | `int` | $n > 0$ | - |
| `alpha` | 复数标量乘子 | 属性 | `complex<float>` | 无 | - |
| `a` | Hermitian 矩阵 A | 输入 tensor | `complex<float>` | Column-Major，仅引用 uplo 指定三角 | `side='L'`: $[m, m]$<br>`side='R'`: $[n, n]$ |
| `lda` | 矩阵 A 前导维度 | 属性 | `int` | `side='L'`: $lda \ge \max(1, m)$<br>`side='R'`: $lda \ge \max(1, n)$ | - |
| `b` | 通用矩阵 B | 输入 tensor | `complex<float>` | Column-Major 稠密矩阵 | $[m, n]$ |
| `ldb` | 矩阵 B 前导维度 | 属性 | `int` | $ldb \ge \max(1, m)$ | - |
| `beta` | 复数标量乘子 | 属性 | `complex<float>` | 无 | - |
| `c` | 累加与输出矩阵 C | 输出 tensor | `complex<float>` | Column-Major 稠密矩阵 | $[m, n]$ |
| `ldc` | 矩阵 C 前导维度 | 属性 | `int` | $ldc \ge \max(1, m)$ | - |

**计算公式：**

- 当 `side = 'L'` 时：$C = \alpha AB + \beta C$，其中 $A$ 为 $m \times m$ 的 Hermitian 矩阵；
- 当 `side = 'R'` 时：$C = \alpha BA + \beta C$，其中 $A$ 为 $n \times n$ 的 Hermitian 矩阵。

## chemm算子功能分析
- **核心功能**：执行复数单精度厄米特矩阵乘加运算 $C = \alpha \cdot \mathrm{op}(A, B) + \beta \cdot C$。
- **输入**：矩阵 `a` 、矩阵 `b` 、矩阵 `c` （累加初值）。
- **输出**：矩阵 `c`。
- **支持数据类型**：`complex<float>`（单精度复数，实虚部各占4字节FP32，单元素8字节）。
- **数据排布**：列优先存储（Column-Major）。
- **支持广播**：不支持广播。
- **Hermitian 特性**：矩阵 $A$ 满足 $A = A^H$（共轭转置对称），仅引用 `uplo` 指定的三角部分，另一侧由共轭对称性动态推导；对角线元素虚部强制按 $0.0$ 处理。

# 需求分析
## 需求分析
使用 Ascend C 编程语言，在 Atlas A2 训练系列产品上实现高性能 `chemm` 算子，支持单精度复数数据类型，支持左乘/右乘（`side='L'/'R'`）、上三角/下三角（`uplo='U'/'L'`）的全模式泛化组合，实现与 cuBLAS / Netlib BLAS 精度完全对齐，整体算子性能达到 0.8 倍 GPU（A100）基准以上。
## 需求拆解
1. **模式与分支泛化支持**：支持 `side`（`'L'`，`'R'`）与 `uplo`（`'U'`，`'L'`）共4种基础拓扑组合；支持任意合法正整数维度 $m,n$ 及合法跨度 `lda, ldb, ldc`。
2. **复数计算映射与硬件解耦**：将复数矩阵乘法解构为实数计算核心（Cube Core）支持的4次 GEMM 运算与向量核心（Vector Core）的加减与融合缩放。
3. **Hermitian 三角块重构**：在片上缓存完成对角块虚部清零、非存储三角块的动态转置与共轭变号，避免读取全局内存未定义区域。
4. **多核并行与流水线编排**：设计多核网格划分（Grid Tiling）与核内双缓冲（Ping‑Pong Buffer），使数据搬运（MTE2/MTE3）、矩阵乘加（Cube MAC）与向量处理（Vector ALU）高度重叠。
5. **性能达标**：全面超越 GPU A100 实测算力的 80%。
# 详细设计
## 算子分析
### 数学公式
设复数矩阵 $A = A_r + iA_i$，$B = B_r + iB_i$，$C = C_r + iC_i$，标量 $\alpha = \alpha_r + i\alpha_i$，$\beta = \beta_r + i\beta_i$。

以 $\text{side} = \text{'L'}$ 为例，中间矩阵乘积 $Z = AB = Z_r + iZ_i$ 展开为实数运算：

$$
Z_r = A_rB_r - A_iB_i
$$

$$
Z_i = A_rB_i + A_iB_r
$$

最终输出矩阵 $C$ 的线性组合（Epilogue）为：

$$
C_r = \left(\alpha_r Z_r - \alpha_i Z_i\right) + \left(\beta_r C_r - \beta_i C_i\right)
$$

$$
C_i = \left(\alpha_r Z_i + \alpha_i Z_r\right) + \left(\beta_r C_i + \beta_i C_r\right)
$$

对于 Hermitian 矩阵 $A$（$A = A^H = \bar{A}^T$）：
- 实部矩阵满足转置对称：$A_r(j,k) = A_r(k,j)$
- 虚部矩阵满足反对称：$A_i(j,k) = -A_i(k,j)$，且对角线 $A_i(k,k) = 0$

## 算子实现
### Host侧设计
 1. Tiling 策略
- 外缘 2D 平铺：Host 侧将输出矩阵 $C[m,n]$ 视作二维计算空间。定义基本分块单元 $\mathit{Tile}_M,\mathit{Tile}_N$，根据硬件 L1 Buffer 与 Unified Buffer（UB）容量约束确定单次计算分块大小。
- K 轴规约切分：矩阵乘内维（`side='L'` 时为 $m$，`side='R'` 时为 $n$）沿 $K$ 轴切分为步长 $\mathit{Tile}_K$ 的循环块。
- 列优先与前导维度映射：由于数据在全局内存（GM）中为列主序，Host 侧将 `lda, ldb, ldc` 转换为连续物理字节步长（`stride = ld * sizeof(complex<float>)`），写入 TilingData 传递给 Kernel 侧。

 2. 分核策略（Grid Tiling）
遵循**优先占满物理核，负载均衡分配**的原则：
- 获取当前硬件的物理 AI Core 数量 $P$（如 Atlas A2 的 24/48 核）。
- 在 $M$ 轴与 $N$ 轴方向计算核网格划分：$\mathit{Grid}_M \times \mathit{Grid}_N \le P$。
- 大小核切分（Tail Handling）：
  - 若总块数能被物理核数整除，所有核均分相同计算块；
  - 若不能整除，余数块均匀摊派给前几个“大核”（Big Core），其余为“小核”（Small Core），通过 `finalBigTileNum` 与 `finalSmallTileNum` 显式下发各核处理区间。

 3. 数据分块和内存优化策略
遵循**充分利用 L1/UB 空间，开启双缓冲（Double Buffer）隐藏延迟**的原则：
- 片上内存规划：
  - L1 Buffer：开辟 Ping‑Pong 双缓冲区，分别缓存当前与预取的 $A_{tile}$（实部/虚部）和 $B_{tile}$（实部/虚部）实数矩阵块。
  - LOA / LOB Buffer：存放 Cube 乘加单元所需的微块操作数。
  - LOC Buffer：用于累计 $A_rB_r$、$A_iB_i$、$A_rB_i$、$A_iB_r$ 4 个实数分量的矩阵乘结果。
  - UB(Unified Buffer)：用于暂存从 LOC 搬出的中间结果，执行向量化加减法、$\alpha/\beta$ 标量融合缩放以及复数交织/去交织转换。
- Tile 尺寸典型配置（FP32 实数展开）：
  - $\mathit{Tile}_M = 64/128$
  - $\mathit{Tile}_N = 64/128$
  - $\mathit{Tile}_K = 32/64$

 4. TilingKey 规划策略
根据计算分支与优化路径设定不同 `TilingKey`：
- `TilingKey = 0`：通用全功能路径（支持任意 $\alpha,\beta$ 以及任意 `side`、`uplo` 分支）。
- `TilingKey = 1`：$\beta=0$ 快速路径（跳过 $C$ 原矩阵的 GM 读入，节约 25% 显存带宽）。
- `TilingKey = 2`：对齐特化快速路径（矩阵维度与 $ld$ 为 64 字节对齐且 $\alpha=1,\beta=0$）。

### Kernel侧设计
Kernel 侧执行生命周期分为 `Init` 初始化与 `Process` 执行阶段。`Process` 内部严格按照 Ascend C 的 **CopyIn、Compute、CopyOut** 三阶段流水线调度。

 1. 复数拆解与 Cube 映射

#### CopyIn 阶段（数据编织与重构）
- MTE2 搬运指令从全局内存 GM 读入列主序复数块，在载入 L1/UB 时通过硬件 Strided DMA 或 Vector 单元执行**去交织（Deinterleave）**，将实部 $A_r, B_r$ 与虚部 $A_i, B_i$ 独立存入 L1 缓冲池。
- **Hermitian 三角镜像处理**：
  - 若当前切块处于合法三角区（`uplo` 指定侧），直接顺序读取；
  - 若切块处于非存储区，DMA 映射源地址为其转置块坐标，搬入片上后执行转置；$A_r$ 保持不变，Vector 单元对 $A_i$ 执行 `Neg` 取反变号；
  - 若为对角相交块，应用对角掩码将合法部分镜像填充至残缺部分，并将主对角线上的 $A_i$ 强制写零。

#### Compute 阶段（Cube MAC + Vector ALU 协同）
- Cube 核心驱动 4 路并行实数矩阵乘：
$$
\begin{align*}
\text{Acc}_0 &= A_r \times B_r, \quad \text{Acc}_1 = A_i \times B_i \\
\text{Acc}_2 &= A_r \times B_i, \quad \text{Acc}_3 = A_i \times B_r
\end{align*}
$$
- 计算结果累加于 LOC 缓冲区。$K$ 轴累加完毕后，由 Vector 单元从 LOC 提取至 UB，执行向量运算：
$$
Z_r = \text{Acc}_0 - \text{Acc}_1
$$
$$
Z_i = \text{Acc}_2 + \text{Acc}_3
$$
- **Epilogue 融合**：在 UB 中将 $Z_r, Z_i$ 与读入的 $C_r, C_i$ 结合标量 $\alpha, \beta$ 完成加权组合，生成最终的 $C_{\text{out\_}r}$ 与 $C_{\text{out\_}i}$。

#### CopyOut 阶段（交织写回）
- Vector 单元调用交织打包指令，将 $C_{\text{out\_}r}$ 与 $C_{\text{out\_}i}$ 重新合并为连续的 `[Real, Imag]` 复数格式。
- MTE3 发起带步长的 DMA 写回指令，将结果写回 GM 中的目标矩阵 $C$。  

# 可维可测分析
| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| ---- | ---- | ---- |
| 精度标准 | 覆盖标准用例、边界用例及泛化用例，计算结果与CPU标准BLAS / cuBLAS对齐：<br><br>双精度基准比对下，相对误差满足 $\mathrm{error} \le 1 \times 10^{-5}$。 | 昇腾生态算子开源精度标准（Experimental Standard） |
| 性能标准 | 算子在 Atlas A2 硬件上的实测浮点算力达到 GPU A100 基准性能的80%以上。典型场景基准线：<br>1. M=1024, N=1024, L, U：≥ 11289 GFLOPS；<br>2. M=2048, N=2048, L, U：≥ 11103 GFLOPS；<br>3. M=1024, N=1024, R, L：≥ 14013 GFLOPS；<br>4. M=2048, N=2048, R, L：≥ 12677 GFLOPS。 | 任务书基准性能要求与A100对标数据 |

# 兼容性分析
本算子为在昇腾开源仓 `ops-blas` 新增的基础数学库算子，基于纯 Ascend C 编程语言开发，严格遵循 C/C++ 接口规范与 CANN 9.0.0+ 运行时标准，不破坏既有算子接口，不涉及向下兼容性历史负担。

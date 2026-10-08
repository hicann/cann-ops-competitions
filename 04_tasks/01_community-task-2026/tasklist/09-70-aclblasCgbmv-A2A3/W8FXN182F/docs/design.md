# aclblasCgbmv 算子设计文档

> 任务：Atlas A2/A3 社区任务 — 单精度复数一般带状矩阵向量乘（aclblasCgbmv）
> 仓库：https://gitcode.com/W8FXN182F/ops-blas （算子目录 `blas/gbmv/arch22/`）
> 硬件：Atlas 800T A2 (910B3) / Atlas A3 系列，CANN 9.1.0

## 1 算子概述

### 1.1 背景与功能

`aclblasCgbmv` 实现复数（COMPLEX64）一般带状矩阵与向量乘法，接口与 cuBLAS `cublasCgbmv` 参数序列一致：

```c
aclblasStatus_t aclblasCgbmv(aclblasHandle_t handle, aclblasOperation_t trans,
                             int m, int n, int kl, int ku,
                             const aclblasComplex *alpha,
                             const aclblasComplex *A, int lda,
                             const aclblasComplex *x, int incx,
                             const aclblasComplex *beta,
                             aclblasComplex *y, int incy);
```

计算式：

```
y = alpha * op(A) * x + beta * y
```

- `op(A)`：`ACLBLAS_OP_N` → A（m×n）；`ACLBLAS_OP_T` → Aᵀ；`ACLBLAS_OP_C` → Aᴴ（n×m）
- `A` 恒为 m×n 一般带状矩阵，按列主序打包存放在 lda×n 数组中（lda ≥ kl+ku+1）：
  A(i,j)（1-based）存放于 `A(ku+1+i-j, j)`，主对角线位于第 ku+1 行；左侧 ku×ku 与右下 kl×kl 三角区不参与计算
- `x`/`y` 以步长 `incx`/`incy` 存取，步长可为负（负步长从末尾反向取值）
- `m=0` 或 `n=0` 为合法 no-op；`alpha=0` 时退化为 `y = beta*y`（不读 A/x）；`beta=0` 时不读原 y

### 1.2 主要输入输出

| 名称 | 方向 | 数据类型 | 形状 | 说明 |
|---|---|---|---|---|
| A | 输入 | COMPLEX64 | lda×n | 列主序带状打包，Device |
| x | 输入 | COMPLEX64 | max(1,n)（N）/ max(1,m)（T/C） | 按 incx 存取，Device |
| y | 输入/输出 | COMPLEX64 | max(1,m)（N）/ max(1,n)（T/C） | 原地写回，Device |
| alpha/beta | 输入 | COMPLEX64 | 标量 | Host |
| m,n,kl,ku,lda,incx,incy | 输入 | int | 标量 | Host |

## 2 算子分析

### 2.1 计算特征

- 计算量：`O(m·(kl+ku+1))` 次复数乘加，属于访存/向量混合型算子，单次调用计算量小（典型 1e3~1e6 次复数 MAC）
- 数据规模：A 的带宽（kl+ku+1）决定实际参与计算的元素数，非零元素仅占带状区域
- 并行性：
  - trans=N：输出行之间相互独立（每个 y[i] 只依赖 A 的第 i 行带宽内元素），可按行切分并行；但同一列 j 会影响 [j-kl, j+ku] 多个输出行，故按行分组（每组 G 行）处理时矩形区域存在“边界斜切”问题
  - trans=T/C：输出列之间相互独立，每个 y[j] 是 A 的第 j 个存储列与 x 片段的点积，天然可按列并行
- 访存特征：
  - 列主序存储下，同一列的带状元素在 GM 中连续；同一对角线方向的元素以 lda 为步长跳转
  - 性能 case 均为紧凑存储（lda = kl+ku+1）且 incx=incy=1，适合 DMA 批量搬运 + UB 上向量计算

### 2.2 关键约束与设计难点

1. **带状索引换算**：A(i,j) → 存储位置 `(ku+i-j) + j*lda`，且 trans 为 T/C 时 op(A) 维度翻转而存储不变
2. **边界斜切**：按行分组时，组内矩形区域的两端（左上/右下三角形）落在带外，必须屏蔽，且不能引入越界读
3. **负步长 / 非单位步长**：不能使用批量 DMA 的连续语义，需要通用（标量）通路
4. **arch22 编程约束**：
   - 部分 AscendC 框架级接口（`CreateVecIndex`、`ReduceSum` 等）依赖 TPipe 的事件分配器，纯 UB 管理的 kernel 需要显式创建 TPipe 实例
   - 级别 2（count 版）接口内部会把向量 mask 切换到 count 模式，之后如再使用级别 0 的按位 mask 必须显式恢复（`SetMaskNorm` + 显式 mask）
   - MTE 传输的 burst 长度过小（< 32B）不可靠，跨核标量写 GM 存在 cache 一致性问题，因此所有 y 写回统一走 DMA

## 3 总体设计

### 3.1 架构与通路

单 kernel 内按“形状/步长是否友好”分派三条通路，host 侧在 tiling 中给出 path 与分组参数：

| path | 适用条件 | 策略 |
|---|---|---|
| FAST_N | trans=N，incx=incy=1，带宽与形状满足 UB/DMA 约束 | 行分组 + 矩形暂存 + 逐列向量累加 |
| FAST_TRANS | trans=T/C，incx=incy=1，bw ≤ 2048 | 逐列点积（向量乘法 + 整列规约），列间按核切分 |
| SLOW | 其余（负/非单位步长、带宽过大、形状被拒） | 逐输出元素标量计算，y 按连续内存段切分后用 DMA 写回 |

分派在 host 侧完成（`CanUseFastN` / `CanUseFastTrans`），kernel 只按 tiling.path 执行，保证 kernel 逻辑简单、可预测。

> **实现迭代说明**（最终版本 f2ce907）：FAST_N 在上述基础上引入 chunk 双缓冲流水线与三段式窗口行走
> （增量式 mask 维护，取消逐列标量工作）；FAST_TRANS 的批量路径**覆盖全部列**（部分带宽列的 a 基址按
> 各列 kLo 平移、向量宽度取各列 len，消除带端串行路径），复数乘积改写为**交错形式**（逐列 2 次 Gather
> + 2 次 Mul + 2 次 ReduceSum，辅助表由 host 预生成、经 workspace 一次 DMA 载入），每核 x 窗口仅在
> 循环前搬运一次。演进明细见 `3.2 性能自验证日志.log` 的优化过程记录。

### 3.2 并行切分

- FAST_N：按输出行分组（组大小 G ≤ 32 且 G ≤ lda-1，取 4 的倍数），组间按核 round-robin 分配；G 由 `PickGroupSize` 依据 m 与核数取“每核约 2 组”的折中
- FAST_TRANS：按输出列连续切分，每核处理 ⌈n/核数⌉ 列
- SLOW：按 y 的**连续内存区间**切分（而非逻辑下标区间），保证每个核一次连续 DMA 写回

### 3.3 UB 规划

最终版 kernel（`blas/gbmv/arch22/cgbmv_kernel.cpp` 行 39~99）的 UB 布局为固定竞技场，三条通路共用（按 32B 槽口径）：

| 区域 | 起始（float） | 大小（float） | 字节 | 用途 |
|---|---|---|---|---|
| ABLOCK | 0 | 8,192 | 32 KiB | FAST_N 带矩形双缓冲 #0；FAST_TRANS 复用为辅助表区（恒等/交换/符号表，6,144 floats = 24 KiB） |
| TIN | 8,192 | 23,744 | 92.75 KiB | FAST_TRANS：a 双缓冲 2×7,776 + xPad/xSw/xSgn 各 2,560 + y 切片 512；FAST_N 复用头部 8,448 floats（aBlock1 8,192 + xWin1 256） |
| TPLANE | 31,936 | 7,168 | 28 KiB | FAST_TRANS 的 7 个 float 平面（aR/aI/xR/xI/pR/pI/pT，各 MAX_BAND=1,024） |
| SMALL | 39,104 | 6,592 | 25.75 KiB | 累加器、x 窗口、拆分/合并 gather 偏移表（各 1,024）、规约工作区/结果、y 回写暂存 |
| **合计** | — | **45,696** | **182,784 B（178.5 KiB）** | `static_assert(UB_FLOATS * 4 <= 184 KiB)` |

其中 MAX_GROUP=32（一个向量 repeat = 64 float = 32 复数），MAX_BAND=1,024（辅助表为 2·MAX_BAND=2,048 项），LC=128，MAX_COLS_PER_CORE=256，SLOW_CHUNK=256。

各通路触及峰值（已用到的最高区域边界）：FAST_N 44,096 floats = 176,384 B（172.25 KiB，SMALL 的 tabM 区）；FAST_TRANS 45,440 floats = 181,760 B（177.5 KiB，SMALL 的 redBatch 区）；SLOW 45,696 floats = 182,784 B（178.5 KiB，stage 暂存直达布局末端）。host 预生成的 4 张 gather 表与 3 张辅助表（合计 40 KiB）经 workspace 传入后分别 DMA 到 SMALL（16 KiB）与 ABLOCK 头部（24 KiB），已计入上表；workspace 的 device 缓冲本身不占 UB。

## 4 详细设计

### 4.1 FAST_N（trans=N）

对输出行分组 [i0, i0+G)：

1. 组内需要 A 的列区间 [jmin, jmax] = [max(0,i0-kl), min(n-1, i0+G-1+ku)]
2. 按列分块（每块 ≤ LC=128 列）DMA 暂存矩形：每列一个 burst，burst 内容为该列存储中 `[rStart, rStart+gPad)` 行（rStart = ku+i0-j，gPad = CeilAlign(gAct,4)），burst 间源步长 = (lda-1) 个复数，UB 侧无间隔（紧凑排布）
   - 该 burst 起点可能为负（列 j 的窗口越到前一列）或终点越到后一列，超出带外的通道由 mask 屏蔽，不参与累加
   - host 侧校验最坏读地址不越出 A 数组，否则退回 SLOW
3. 同块内加载 x 窗口 [jmin, jmax]（连续，1 个 burst），逐列读取标量 xr/xi
4. 逐列累加：`acc1 += A列 * xr`、`acc2 += A列 * xi`（交错复数累加，两次带 mask 的 Axpy，仅对带内通道生效）
5. 组内收尾：
   - 用拆分 gather 表取出 even/odd：`e1=even(acc1), o1=odd(acc1), e2=even(acc2), o2=odd(acc2)`
   - trans=N：`Re = e1-o2, Im = o1+e2`；trans=C：`Re = e1+o2, Im = o1-e2`（共轭）
   - 乘 alpha，叠加 beta·y_old（beta=0 时跳过 y 读取）
   - 用合并 gather 表交错成复数并用 1 次连续 DMA 写回 y[i0, i1)
6. 跨 chunk：MTE2→V、V→MTE2 事件同步；组内累加器每 4 个 chunk 复用（UB 约束）

### 4.2 FAST_TRANS（trans=T/C）

对输出列 j：

1. 计算有效带行区间 `[kLo, kHi] = [max(0,ku-j), min(bw-1, ku+m-1-j)]`，长度 len
2. 1 次 DMA 加载 A 列片段（连续 len 个复数）与对应 x 片段；列片段与 x 片段的 k 偏移一致
3. 用拆分 gather 表得到 aR/aI/xR/xI 四个平面，向量计算
   `pR = aR*xR ∓ aI*xI`（C 取 +）、`pI = aR*xI ± aI*xR`（C 取 -）
4. `ReduceSum` 规约得 (sumR, sumI)，标量合成 alpha/beta，写入 UB 中的 y 切片
5. 列循环结束后 1 次连续 DMA 写回 y 切片
6. len ≤ 0 的列（带外列）按 `y[j] = beta*y[j]` 处理

### 4.3 SLOW（通用通路）

- 每个核拥有 y 存储的连续区间；逐输出元素计算：
  - trans=N：`y[i] = alpha·Σ_j A(ku+i-j,j)·x[j] + beta·y[i]`，j ∈ [max(0,i-kl), min(n-1,i+ku)]
  - trans=T/C：`y[j] = alpha·Σ_i op(A)(i,j)·x[i] + beta·y[j]`，i ∈ [max(0,j-ku), min(m-1,j+kl)]
- 元素级标量读写 GM（A/x 直接标量读、y 通过 UB 暂存），步长支持任意正负值
- 结果按内存顺序写入 UB 暂存（连续 DMA 写回，避免跨核 cache 行竞争与短 burst 传输）

### 4.4 边界与负向处理

| 场景 | 行为 |
|---|---|
| m<0 / n<0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| kl<0 或 kl>m-1；ku<0 或 ku>n-1 | INVALID_VALUE |
| lda < kl+ku+1 | INVALID_VALUE |
| incx=0 / incy=0 | INVALID_VALUE |
| trans 非 N/T/C | INVALID_VALUE |
| handle=nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| alpha/beta=nullptr | INVALID_VALUE |
| alpha≠0 时 A/x=nullptr | INVALID_VALUE |
| (alpha≠0 或 beta≠0) 时 y=nullptr | INVALID_VALUE |
| m=0 或 n=0 | SUCCESS（不计算） |
| alpha=0 | `y = beta*y`（beta=1 时 no-op；beta=0 时 y 置 0，不读 y） |
| beta=0 | 不读取原 y |

### 4.5 数据类型与精度

- 复数按“实部/虚部分别向量化”实现：交错复数累加 + 拆分/合并 gather，实部与虚部均为单精度浮点运算
- 复数乘法展开为 4 次实数乘加（`ar·xr − ai·xi`、`ar·xi + ai·xr`），累加顺序与 CBLAS 参考实现一致（按列/按 k 递增），无额外精度损失

## 5 测试方案

| 层次 | 内容 |
|---|---|
| L0 | 小尺寸（m,n ∈ {4,8}）× trans 全枚举，kl=ku=1 |
| L1 | 尺寸扫描（1 → 2048）× trans，带宽随规模 |
| L2 | alpha/beta 特殊值（(0,0)/(1,0)/(-1,0)/复数/纯虚） |
| L3 | 非方阵（fat/thin）× trans |
| L4 | 前导维 padding、带宽边界（0、m-1/n-1、全带、单边带、1×1）、步长组合（±1/±2/±3） |
| L5 | 填充模式（随机/全零/交替/极值/Inf/NaN）、中等尺寸覆盖 |
| L6 | 边界与负向（零维、空指针、非法枚举、非法 lda、负维度、零步长、kl/ku 负值） |
| EX/PF | 大规模确定性采样、5 条典型性能 case + 扩展性能用例 |

测试代码：`test/gbmv/cgbmv/arch22/cgbmv_test.cpp`（CSV 驱动 GTest），golden 由 CBLAS（Netlib/OpenBLAS `cblas_cgbmv`）生成，精度判定采用混合容差（COMPLEX64：atol=rtol=2⁻¹³，matched_ratio ≥ 0.99，max_abs_error ≤ max(1e-2, 32·ULP)）。

## 6 性能与内存

- 性能目标（Atlas 800T A2 (910B3)，kernel 平均单次耗时）：

| case | m | n | kl | ku | trans | 上限(us) | 实测(us) | 结论 |
|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 256 | 16 | 16 | N | 10.665 | 8.91 | 达标（余量 16%） |
| 2 | 512 | 512 | 32 | 32 | N | 17.328 | 11.61 | 达标（余量 33%） |
| 3 | 1024 | 1024 | 64 | 64 | T | 25.875 | 15.45 | 达标（余量 40%） |
| 4 | 2048 | 2048 | 128 | 128 | N | 30.640 | 28.00 | 达标（余量 8.6%） |
| 5 | 4096 | 4096 | 64 | 256 | C | 52.487 | 47.64 | 达标（余量 9.2%） |

  实测为最终版本（f2ce907）msprof 采集，6 次采样舍弃冷启动取末 4 次平均（有效采样 >10，任务书 §3.3）；
  明细见 `3.2 性能自验证日志.log` 与自测报告。**5 个 case 全部达标。**
- 内存：FAST_TRANS 通路的 4 张 gather 偏移表与 3 张辅助表为常量，首次使用时按句柄分配一次 40KB Device 内存（作为 workspace 传入 kernel，`aclblasDestroy` 时释放），其余情况无额外 Device 内存申请；UB 使用 ≤ 178.5KiB（182,784 B 静态布局，`static_assert` ≤ 184KiB；A2/A3 单核 UB 192KiB）

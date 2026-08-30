# aclblasCher2k 算子设计文档

## 1. 需求分析

### 1.1 算子功能

实现 BLAS Level-3 单精度复数厄米特秩-2k 更新（complex64 Hermitian rank-2k update），对标 cuBLAS `cublasCher2k`：

```
C := alpha * A * B^H + conj(alpha) * B * A^H + beta * C
```

- `C`：n×n 复数厄米特矩阵，列主序（column-major），仅按 `uplo` 更新上/下三角；
- `A`、`B`：n×k 复数矩阵，列主序；
- `trans = N`：`op(A)=A, op(B)=B`；`trans = C`：`op(A)=A^H, op(B)=B^H`（此时 A/B 逻辑上为 k×n）；
- `alpha`：复数标量；`beta`：实数标量；
- 数据类型：complex64（`float2`，实部/虚部交错存储）。

### 1.2 接口定义

```c
aclblasStatus_t aclblasCher2k(
    aclblasHandle_t handle,        // 已创建并绑定 stream 的 BLAS handle
    aclblasFillMode_t uplo,        // ACLBLAS_FILL_MODE_UPPER / LOWER
    aclblasOperation_t trans,      // ACLBLAS_OP_N / ACLBLAS_OP_C
    int64_t n, int64_t k,
    const float2 *alpha,
    const float2 *A, int64_t lda,
    const float2 *B, int64_t ldb,
    float beta,
    float2 *C, int64_t ldc,
    aclComputeType type,           // 仅支持 ACL_COMPUTE_HIGH_PRECISION
    void *workspace, size_t workspaceSize);
```

约束与负向行为（非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`）：

- `n < 0`、`k < 0`、前导维非法（trans=N 时 `lda < max(1,n)`、`ldb < max(1,n)`；trans=C 时 `lda < max(1,k)`、`ldb < max(1,k)`；`ldc < max(1,n)`）、空指针等；
- `n == 0 || k == 0 || (alpha == 0 && beta == 1)` 时 quick return（`ACLBLAS_STATUS_SUCCESS`）；
- `alpha == (0,0)` 时退化为 `C = beta * C`（跳过 GEMM，对角线虚部清零）。

### 1.3 精度与性能要求

- 精度：与 cblas 参考（fp32）逐元素比对，pass = matchedRatio ≥ 99%（混合容差）且全部元素误差 ≤ max(1e-2, 32·ULP) 硬顶；
- 性能（Atlas 800I A2 / 910B3，arch22，warmup 后采样 >50 次取平均）：

| uplo | trans | n | k | 标杆 (us) |
|------|-------|------|------|-----------|
| UPPER | N | 1024 | 1024 | 933.70 |
| UPPER | N | 2048 | 2048 | 3354.04 |
| LOWER | C | 1024 | 1024 | 692.42 |

## 2. 总体方案

### 2.1 数学变换

令 `P = op(A) * op(B)^H`（trans=N 时 `P = A·B^H`，trans=C 时 `P = A^H·B`）。由于

```
conj(alpha) * B * A^H == conj(alpha * P)^H
```

原式等价于在 uplo 三角上计算

```
C(i,j) = Q(i,j) + conj(Q(j,i)) + beta * C(i,j),   Q = alpha * P
```

即只需计算**一个**普通复 GEMM `P`（而非两个），将对称化折叠到尾声（combine）阶段。`P` 不是厄米特矩阵，因此必须计算完整的 n×n（不能利用三角跳算：combine 需要同时读 `P(i,j)` 与 `P(j,i)`）。

复乘展开为 4 个实 fp32 GEMM。把 A/B 按实部/虚部解交错（deinterleave）为实平面：

```
Pr = Ar·Br^T + Ai·Bi^T
Pi = Ai·Br^T - Ar·Bi^T   (trans=N)
Pi = Ar·Bi^T - Ai·Br^T   (trans=C)
```

为让 cube 核只做"纯累加" Mmad（Mmad 不支持取负），把负号吸收进一个预处理平面 `Bneg`（trans=N：`Bneg = -Bi`；trans=C：`Bneg = -Br`），于是：

```
Pi = Ai·Br^T + Ar·Bneg^T  (trans=N)
Pi = Ar·Bi^T + Ai·Bneg^T  (trans=C)
```

### 2.2 三阶段流水

整个算子拆为 3 个 kernel（同一 stream 顺序执行）：

| 阶段 | 核类型 | 功能 |
|------|--------|------|
| preprocess | AIV (50 核) | A/B 复数 AoS → 实/虚平面 SoA 解交错、零填充、生成 Bneg，按 GEMM 友好的布局写入 workspace。RowPack：4096 元素 slab + GatherMask 奇偶分流，满块免清零；ColPack（trans=C 的 B 侧/任何需转置的平面）：内部 64×64 瓦片走快速路径——ld 32B 对齐时 1 条 2D DataCopyPad 装 AoS 瓦片，非对齐时拆为 64 条逐列 1D DataCopyPad（2D 段起点仅 8B 对齐时硬件病态慢，实测逐列拆条约 96us/瓦片 vs 标量 620us/瓦片）+ 2 条 Gather（偏移表向量建表）完成转置解交错 + 3 条 2D DataCopyPad 存平面；仅最边缘瓦片保留标量 staging |
| gemm | AIC (20 核) | 4 个实 fp32 GEMM，全 k 单 CO1 累加直存 PrA/PiA（见 2.4/2.5）到 workspace |
| combine | AIV (50 核) | 读 PrA/PiA 两平面，计算 Q+conj(Q)^T+beta·C 并按 uplo 三角写回 C（AoS 交错、列主序、ldc 步长），对角线虚部强制置 0 |

选择该方案而非单 kernel 内联 matmul 的原因：

1. 输入 A/B 是复数交错的列主序大矩阵，cube 单元需要 NZ/分形布局的实平面，必须先做数据重排（AIV 擅长）；
2. 对称化与 alpha/beta 融合是纯向量逐元素操作，适合 AIV 尾声处理；
3. 实测本 CANN 版本的 MatmulImpl 高阶接口在 arch22/fp32/ND 场景下仅 org==single 单瓦片调用正确且吞吐极低（约 65 GFLOPS/核），不可用，故 GEMM 阶段采用手写 L1→L0→CO1 流水。

### 2.3 Workspace 平面布局

全部 fp32 行主序、零填充（`nPad = CeilAlign(n,256)`（cube baseN=256 需要），`kPad = CeilAlign(k,128)`，GEMM 参与时 `kPad ≥ 128`）：

```
Ar, Ai          : nPad × kPad   Ar[i][l] = re/op(A)(i,l)
Br, Bi, Bneg    : kPad × nPad   Br[l][j] = re/op(B)(l,j)
PrA, PiA        : nPad × nPad   P = PrA + i·PiA（全 k 一次累加直存）
```

总大小 = `5·nPad·kPad + 2·nPad²` 个 float。trans 与平面临界方向：

- trans=N：preprocess 将 A 平面按 `[l][i]`（kPad×nPad）存放（RowPack），与 B 平面同向，GEMM 的 A 侧 Nd2Nz/LoadData 与 B 侧完全镜像，配合 L0A 硬件转置（`enTranspose`）得到 ZZ 布局；
- trans=C：A 平面按 `[i][l]`（nPad×kPad）存放（ColPack），L0A 无需转置。

### 2.4 精度设计：全 k 单累加器

每个输出瓦片的整个 k 范围在单个 CO1 累加器（fp32）中一次累加完成，末尾经 Fixpipe 直存 PrA/PiA。早期版本曾用 split-k=2（两半各存一份、combine 再合并）来缩短串行累加链，但实测：

1. 在 c220 上 AIC 指令完全串行发射（见 2.5），mid-tile 的 CO1→GM 存储是纯开销；
2. combine 的 P 平面装载量因此翻倍（8 条 2D DMA → 4 条）；
3. 对 fp64 golden 的实测误差两者同量级（单累加器甚至略优，少一次加法舍入）。

故移除 split-k。说明：k=2048 的若干验收用例中，fp32 netlib golden 自身相对 fp64 的误差已超过判定硬顶 1e-2（实验证据见自测报告），该类用例的失败由基线方法学决定，与 NPU 实现无关。

### 2.5 GEMM 核内调度（arch22, fp32）

c220（DAV_2201）微架构实测结论（由 pipe_bench 微基准建立）：

- **AIC/AIV 指令发射完全串行**：每条指令（Nd2Nz/LoadData/Mmad/Fixpipe/flag）有固定延迟，管道间零重叠，消融永远完美可加。因此 kernel 内所有 SetFlag/WaitFlag 都是纯开销——本实现**全 kernel 无 flag**，优化等价于减少指令条数；
- 实测指令延迟：LoadData3D 64KB≈1.21us、32KB≈0.99us，Mmad 128×256×64≈1.66us（不受累加链 RAW 限制），Nd2Nz 64KB≈0.76us；
- **L0C 只有 128KB**（128×256 fp32 恰好占满；双面共驻会 trap 507015）。

调度结构（面拆分瓦片）：

- 输出瓦片 baseM=128 × baseN=256，k 方向按 64 切片（slice）；tile index bit0 = 面（0=Pr，1=Pi），Pr/Pi 两面作为独立瓦片各自驻留单个 128KB CO1；
- 每 slice：L0A 双 32KB 槽（Ar/Ai），L0B 单 64KB 槽（Bx→t_a→By→t_b 串行流），2 条 Mmad；L1 双 stage 各 192KB；
- **ND 装载（条件融合）**：arch22 的 `Nd2NzParams.srcNdMatrixStride` 是 **uint16 字段**（>65535 静默截断为 0，曾导致虚面整体算错），仅当平面跨度 `nPad·kPad ≤ 65535` 时才用 ndNum=2 融合装载（A 侧一次装 Ar/Ai，B 侧按面选 (Br,Bi)/(Br,Bneg)/(Bi,Bneg)）；大尺寸回退为每平面独立 Nd2Nz（正确性优先，实测性能差异在指令串行地板模型下可预期）；
- 每 slice 合计 2 条 Nd2Nz + 4 条 LoadData + 2 条 Mmad；全 k 累加完成后 CO1 经 Fixpipe 直存 PrA/PiA（无 UB 中转）；
- 每核按 `tileIndex + i·tileStride` 领取瓦片；`aicCoreNum = min(20, tileCount)`。

### 2.6 combine 核内向量化

- 以 64×64 复元素瓦片（`kCher2kCombTile=64`，与 host 共享于 tiling 头文件）处理 uplo 三角，tile 循环拉平后按核均分（三角负载均衡）；
- P 装载：4 条 2D `DataCopyPad`（64 行 × 256B）装 Pr/Pi 的 (tr,tc) 与 (tc,tr) 两瓦片；
- UB 内转置（取 `P(j,i)`）：单条 `Gather`（字节偏移向量）；C 的 AoS 解交错用硬件 `GatherMask` 奇偶模式（远快于偏移向量 Gather）；回写交错（zip）用单条 `Gather` 完成；
- 算术融合用 `Muls`+`Axpy`（dst += src·scalar），每瓦片 10 条向量指令（beta=0 时 8 条）；
- 偏移表（转置 16KB + zip 32KB）**向量建表**：8/128 条标量写打头 + 倍增/逐行 `Adds` 展开（纯标量 SetValue 建表实测 ~1ms/launch，向量化后 ~30us）；
- UB 192KB 预算通过按生命周期别名压到 176KB：cAos 复用 t1（转置完成后才发 C 装载，V→MTE2 flag 保护），outAos 复用 t2（zip 在算术消费 t2 之后）；
- 对角线瓦片走向量路径 + 三角形掩码存储（UPPER 逐列前缀直接存；LOWER 的行后缀起点在 UB 内仅 8B 对齐，UB→UB DataCopy 会触发 VEC 对齐 trap，改为**逐列 Gather 重 zip**：以 off2 第 0 行为偏移表、srcBaseAddr 平移读取窗，把每列保留段直接 zip 到 32B 对齐的存储槽），对角线虚部向量置 0；
- 前导维非 32B 对齐（lda/ldb/ldc 非 4 的倍数，紧凑布局下奇数 n 必然如此）时快速路径仍然生效：GM 侧装载/存储一律用 `DataCopyPad`（硬件支持任意 GM 对齐），仅 UB 侧保持 32B 对齐；这是奇数尺寸性能的关键（早期版本按 `ld%4!=0` 全局回退标量路径，1775² 的 pre/combine 比 2048² 慢近 2 倍）；
- **边缘瓦片同样走向量路径**（`skipGemm` 除外）：P 装载按整瓦片读（平面已零填充到 nPad，越界读到的是填充 0，安全）；C 装载/存储在边缘处逐列用 1D `DataCopyPad` 钳制到有效行/列（UPPER 对角存 `min(c+1, rCnt)` 行，LOWER 重 zip 循环 `c < min(cCnt, rCnt)`）；无效 lane 的垃圾不参与存储，无需清零。实测 n=1049 的 combine 由 1429us（标量边缘）降至 ~373us；
- 经验结论：AIV 上每条 DMA 指令有约 1.3~1.9us 固定开销，与数据量无关，必须用大粒度拷贝合并；UB→UB 跨步 DataCopy 的 dstGap 在 c220 被静默忽略（zip 必须用 Gather）。

### 2.7 主机侧

`cher2k_host.cpp` 负责：参数校验与 quick return、从 device 读取 alpha/beta 标量（带同步）、tiling 计算、workspace 划分，并按序发射 preprocess/gemm/combine 三个 kernel（`skipGemm` 时只发射 combine）。

## 3. 测试设计

### 3.1 精度用例（GTest + CSV 驱动，cblas fp32 golden）

覆盖（详见 `test/herk/cher2k/arch22/cher2k_test.csv`）：

- 小尺寸基础用例（1×1 起）、尺寸扫描（n/k 扫 1..4096 多组）；
- uplo × trans 全组合；非方阵（宽/窄矩形）；lda/ldb/ldc 带 padding 与步长组合；
- 标量特殊值：alpha=(0,0) no-op、beta=0、alpha 纯实/纯虚、beta 负值；
- 边界与负向：零维 quick return、空指针、非法前导维、负维度、零步长（期望 `ACLBLAS_STATUS_INVALID_VALUE`）；
- Inf/NaN 传播用例。

### 3.2 性能用例

- 任务书 §3.3 三个标杆用例（UPPER/N 1024²、UPPER/N 2048²、LOWER/C 1024²）；
- 尺寸扫描、低秩（k≪n）、混合矩形等 200 条 PF 用例，与 GPU(A100) 基线按倍率 0.8 判定。

实测环境注意：测试期间芯片 Aicore 锁频 800MHz（额定 1800MHz），且 host 侧每次调用有 ~360us 固定开销，小尺寸用例被该开销主导。分阶段实测（device 侧计时）：

| 用例 | preprocess | gemm | combine | 全程 |
|------|-----------|------|---------|------|
| 1024² N UPPER | 486.7 | 804.9 | 318.3 | 1606.5 |
| 2048² N UPPER | 954.1 | 5157.2 | 731.4 | 6878.0 |
| 1049² N UPPER（奇数尺寸） | 93.7 | 139.2 | 372.6 | 605.7 |

（单位 us；1049² 行为 k=64 低秩场景，验证奇数/非对齐路径的向量化收益——combine 由标量边缘的 1429us 降至 372.6us。）

2048² 的 gemm 耗时落在"AIC 指令串行发射"地板模型预测值（~5.16ms）上，任务书 3354us 标杆在该微架构下不可达，详见 §2.5 与自测报告 §2/§4。

### 3.3 自测方法

- 构建：`bash build.sh --soc=ascend910b3 --ops=cher2k`（或 cmake 增量）；
- 精度：`build/test/herk/cher2k/cher2k_test . --gtest_filter='-*TC_PF*'`；
- 性能：`python3 test_cases/verify_performance.py --repo ops-blas --soc ascend910b3 --skip-build`；
- 详细步骤见 `test_cases/README.md` 与 `blas/herk/README.md`。

## 4. 风险与对策

| 风险 | 对策 |
|------|------|
| k=2048 用例 fp32 golden 自身超 1e-2 硬顶 | 全 k 单 CO1 累加压低 NPU 误差；自测报告附 golden-vs-fp64 证据 |
| arch22 Nd2Nz 的 srcNdMatrixStride 为 uint16，平面跨度 >65535 静默截断 | 跨度超限时回退逐平面独立装载（融合仅作小尺寸 fast path） |
| c220 UB→UB DataCopy 源地址 8B 对齐即触发 VEC trap | LOWER 对角瓦片改为逐列 Gather 重 zip 到对齐槽 |
| AIC 指令发射完全串行，flag/DMA 固定开销大 | gemm 全 kernel 无 flag；combine/pre 用大粒度 2D 拷贝 + 向量建表 |
| 奇数/非对齐前导维导致 AIV 快速路径失效 | GM 侧一律 DataCopyPad（容忍任意对齐），UB 侧保持 32B 对齐 |
| 多 kernel 复用硬件 flag 死锁 | 事件令牌严格配平 + kernel 末尾排空残余 token |

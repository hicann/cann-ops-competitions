# 【社区任务】aclsparseSDDMM（A2/A3）算子设计文档

# 一、需求背景

## 1.1 需求来源

本需求来自 CANN 2026 社区任务“aclsparseSDDMM 算子开发（A2/A3）”。目标是在 Atlas A2/A3 上实现与 cuSPARSE SDDMM 语义对齐的三阶段 C 接口，并提供 PyTorch/ATen NPU 适配。

算子数学定义为：

$$
D=\operatorname{op}(X)\operatorname{op}(Y),
\qquad
D_{ij}=\sum_{k=0}^{K-1}\operatorname{op}(X)_{ik}\operatorname{op}(Y)_{kj}.
$$

对 C.values 中第 $p$ 个已存储元素，更新规则为：

$$
v_{p}^{\mathrm{new}}
=\alpha D_{r(p),c(p)}+\beta v_{p}^{\mathrm{old}},
$$

其中 $r(p)$、$c(p)$ 分别为该元素在稀疏矩阵中的行、列坐标，$v_p^{\mathrm{old}}$ 和 $v_p^{\mathrm{new}}$ 分别表示执行前后的值。

令 $S=\operatorname{spy}(C_{\mathrm{old}})\in\{0,1\}^{M\times N}$ 表示二值稀疏掩码，$\widetilde{C}$ 表示将稀疏矩阵未存储位置补零后的稠密矩阵视图，则等价的矩阵形式为：

$$
\widetilde{C}_{\mathrm{new}}
=\alpha\left[\left(\operatorname{op}(X)\operatorname{op}(Y)\right)\odot S\right]
+\beta\widetilde{C}_{\mathrm{old}}.
$$

其中 $\odot$ 表示 Hadamard 逐元素乘。$\operatorname{op}(X)$ 的逻辑形状为 $[M,K]$，$\operatorname{op}(Y)$ 的逻辑形状为 $[K,N]$。算子保持 CSR/BSR 的 offsets、columns 和稀疏模式不变，仅原地更新 C.values。

## 1.2 背景介绍

SDDMM 只计算稀疏模式指定位置的稠密矩阵乘结果，避免生成完整的 M×N 中间矩阵。主要设计问题如下：

- 同一 CSR 行的多个采样位置共享 X[row,:]，需要复用 X 行数据。
- op(Y) 的 K 维元素可能跨距访问，需要转换为 K 连续布局。
- CSR 行 nnz 分布可能存在长尾，需要按计算量进行多核负载均衡。
- BSR 非零块具有规则矩阵乘结构，FP16/BF16、FP32 和 complex64 的 b16/b32/b64 在满足准入条件时使用 Cube，其余场景沿用 Vector 路径。
- ROW/COL、N/T、base0/base1、batch 和 BSR 块内 ROW/COL 需要统一处理。

### 1.2.1 aclsparseSDDMM 算子实现优化

本任务不是 ACLNN/TBE 算子优化任务，不存在对应的 TBE 算子源码和算子信息库文件，因此 TBE 源码路径和算子信息库路径不适用。

标杆采用 NVIDIA cuSPARSE Generic API：

- 官方接口：CUDA 12.6 cusparseSDDMM。
- 标杆源码：task/test_cases/aclsparseSDDMM_testCase/benchmark_cusparse_gpu.cu。
- 固定用例：task/test_cases/aclsparseSDDMM_testCase/performance_cases.json。
- 标杆结果：task/test_cases/aclsparseSDDMM_testCase/gpu_performance_result_benchmark.md。

task/*为任务书目录。cuSPARSE 内部 Kernel 未公开。本设计对齐公开接口语义、支持矩阵、三阶段调用流程和 Execute 计时范围；Ascend C Kernel 根据 DAV_2201 架构设计。

### 1.2.2 标杆算子现状分析

#### 1.2.2.1 标杆算子支持的数据类型和数据格式

| 项目 | cuSPARSE SDDMM | 本任务范围 |
| --- | --- | --- |
| 稀疏格式 | CSR、BSR | CSR、BSR |
| 稀疏索引 | 32-bit、64-bit，base0/base1 | int32，base0/base1 |
| FP32 | X/Y/C/compute 均为 FP32 | CSR、BSR |
| complex64 | X/Y/C/compute 均为 complex64 | CSR、BSR |
| FP16→FP16 | X/Y/C 为 FP16，compute 为 FP32 | CSR、BSR |
| FP16→FP32 | X/Y 为 FP16，C/compute 为 FP32 | CSR、BSR |
| BF16→BF16 | X/Y/C 为 BF16，compute 为 FP32 | 仅 BSR |
| BF16→FP32 | X/Y 为 BF16，C/compute 为 FP32 | 仅 BSR |
| BSR block size | 2、4、8、16、32、64、128 | 相同 |
| BSR block order | ROW、COL | 相同 |
| op | NON_TRANSPOSE、TRANSPOSE | N、T；complex64 不共轭 |
| 算法 | CUSPARSE_SDDMM_ALG_DEFAULT | ACL_SPARSE_SDDMM_ALG_DEFAULT |

FP16/BF16 路径固定使用 FP32 computeType；CSR 不支持 BF16。

#### 1.2.2.2 标杆算子实现描述

标杆程序执行流程如下：

1. 使用固定 seed 生成 CSR/BSR pattern、X、Y 和 C.values。
2. 将输入拷贝至 GPU，创建 cuSPARSE handle、DnMat 和 SpMat 描述符。
3. 调用 cusparseSDDMM_bufferSize 获取 externalBuffer 大小并完成分配。
4. 调用 cusparseSDDMM_preprocess 建立稀疏模式预处理状态。
5. 调用 cusparseSDDMM 原地更新 C.values。
6. 性能测试预热不少于 10 次，正式采样不少于 30 次。
7. Execute 计时只包含 cusparseSDDMM，不包含数据生成、H2D、描述符创建、BufferSize、内存分配和 Preprocess。

#### 1.2.2.3 标杆算子实现流程图

~~~mermaid
flowchart TD
    A[固定 seed 生成输入] --> B[H2D]
    B --> C[创建 handle 和矩阵描述符]
    C --> D[cusparseSDDMM_bufferSize]
    D --> E[分配 externalBuffer]
    E --> F[cusparseSDDMM_preprocess]
    F --> G{预热完成?}
    G -- 否 --> H[恢复 C.values]
    H --> I[cusparseSDDMM Execute]
    I --> G
    G -- 是 --> J[CUDA Event start]
    J --> K[cusparseSDDMM Execute]
    K --> L[CUDA Event end并同步]
    L --> M[统计median/p90/min/max]
~~~

# 二、需求分析

## 2.1 外部组件依赖

| 组件 | 作用 | 约束 |
| --- | --- | --- |
| CANN 9.1.0 及配套版本 | Ascend C 编译、Runtime、Kernel launch | 以 CANN 9.1.0 API 为开发基线 |
| ops-sparse 公共 C ABI | handle、DnMat/SpMat 描述符和状态码 | 保持既有 ABI 语义 |
| PyTorch 2.7+ | sparse_sampled_addmm 入口 | 仅注册 NPU backend |
| torch_npu 26.0.0+ | PrivateUse1/NPU dispatcher | 与 PyTorch、CANN 配套 |
| GoogleTest、pytest、ATK | 功能和精度测试 | 不链接进入发布库 |
| CUDA 12.6、cuSPARSE | GPU 标杆 | 不属于 NPU 运行时依赖 |
| msprof | NPU 性能分析 | Execute 的所有子 Kernel 纳入统计 |

## 2.2 内部适配模块

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| 公共 API | include/cann_ops_sparse.h | SDDMM、BSR 和 batch 接口 |
| 公共描述符 | sparse/common/aclsparse_descr_internal.h、aclsparse_descr.cpp | CSR/BSR、DnMat、prepared state |
| A2/A3 Host | sparse/sddmm/arch22/sddmm_host.cpp、sddmm.h | 校验、workspace、Preprocess、Tiling、launch |
| A2/A3 Kernel | sparse/sddmm/arch22/sddmm_kernel.cpp、sddmm_kernel.h | 按 TilingData 字段分发 |
| FP32 Cube | sparse/sddmm/arch22/sddmm_f32_cube{.h,_host.cpp,_kernel.cpp,_kernel.h} | 资源规划、独立 launch 声明、zero/pack/Matmul 写回 |
| complex64 Cube | sparse/sddmm/arch22/sddmm_c64_cube{.h,_host.cpp,_kernel.cpp,_kernel.h}、sddmm_c64_pair{_kernel.h,_pipeline.h} | 实数平面打包、physical pair 流水、Matmul 和复数组合 |
| CSR Vector | sparse/sddmm/arch22/sddmm_fp16_kernel.h、sddmm_simd_kernel.h、sddmm_complex64_kernel.h | FP16/FP32/complex64 CSR 计算 |
| 通用路径 | sparse/sddmm/arch22/sddmm_generic_kernel.h | BSR及不满足快路径条件的回退场景 |
| Y 规范化 | sparse/sddmm/arch22/sddmm_transpose_kernel.h | Y 转换为 [N,K] 连续布局 |
| Python/ATen | python/csrc/sddmm.cpp、python/ops_sparse_npu | NPU 注册和调用封装 |
| 测试 | test/sddmm、python/tests、task/test_cases/aclsparseSDDMM_testCase | 单元、精度和性能验证 |

## 2.3 需求模块设计

### 2.3.1 Ascend C 算子原型

~~~c
aclsparseStatus_t aclsparseSDDMMBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opX,
    aclsparseOperation_t opY,
    const void *alpha,
    aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSDDMMAlg_t alg,
    size_t *size);

aclsparseStatus_t aclsparseSDDMMPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opX,
    aclsparseOperation_t opY,
    const void *alpha,
    aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSDDMMAlg_t alg,
    void *buffer);

aclsparseStatus_t aclsparseSDDMM(
    aclsparseHandle_t handle,
    aclsparseOperation_t opX,
    aclsparseOperation_t opY,
    const void *alpha,
    aclsparseConstDnMatDescr_t matX,
    aclsparseConstDnMatDescr_t matY,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSDDMMAlg_t alg,
    void *buffer);
~~~

配套接口包括 aclsparseCreateBsr、aclsparseCreateConstBsr、aclsparseDnMatGetStridedBatch、aclsparseDnMatSetStridedBatch 和 aclsparseBsrSetStridedBatch。

| 参数 | 方向/位置 | 约束 |
| --- | --- | --- |
| handle | 输入，Host | 非空，保存调用 stream |
| opX/opY | 属性，Host | NON_TRANSPOSE 或 TRANSPOSE |
| alpha/beta | 输入，Host或Device | 类型等于 computeType；Device mode 由 Kernel 从 GM 读取 |
| matX/matY | 输入描述符 | dtype、shape、order、ld 和 batch 合法 |
| matC | 输入输出描述符 | CSR/BSR、I32、base0/base1；只更新 values |
| computeType | 属性，Host | FP32 或 complex64，与 dtype 组合匹配 |
| alg | 属性，Host | 仅支持 DEFAULT |
| size | 输出，Host | BufferSize 必选，返回精确 workspace 字节数 |
| buffer | 输入，Device | Preprocess/Execute 必选，至少 64B 对齐 |

### 2.3.2 Ascend C 算子相关约束

| 能力 | 标杆能力 | 本设计 |
| --- | --- | --- |
| 64-bit sparse index | 支持 | 不支持，任务限定 int32 |
| FP64、complex128 | 支持 | 不支持 |
| CSR BF16 | 不支持 | 不支持 |
| CONJUGATE_TRANSPOSE | 不在任务接口范围 | 不支持 |
| 非方块 BSR | 接口可表达 | 不支持 |
| 非 DEFAULT 算法 | 可能扩展 | 不支持 |
| CPU fallback | 不适用 | 禁止 |

边界要求如下：

- M、N、K、ld、stride、nnz、blockNnz 和 workspace 的乘加均进行溢出检查。
- CSR C 固定为单 batch；BSR C.batchCount 为 1～65535。
- X/Y batchCount 为 1 或等于 BSR C.batchCount；1 表示广播。
- base1 时 nnz+1 或 blockNnz+1 必须可由 int32 表达。
- 零 nnz 合法，Execute 不写无关地址。
- TilingData 字段无法表达时拆分 launch，不截断数值。
- 未执行 Preprocess、workspace 不匹配或 prepared state 失效时返回错误。

# 三、需求详细设计

## 3.1 调用方式

正式调用链如下：

~~~text
创建 handle/描述符
  → aclsparseSDDMMBufferSize
  → 分配 workspace
  → aclsparseSDDMMPreprocess
  → 一次或多次 aclsparseSDDMM Execute
  → stream 完成后释放资源
~~~

BufferSize 和 Preprocess 不计入 Execute 性能。Execute 只向 handle 对应 stream 提交必要 Kernel，不执行设备同步。

Preprocess 校验 offsets/columns 的端点、单调性和范围，将 base1 pattern 归一化为 base0，生成 pattern hash、task map 和 CSR run metadata，并写入 workspace。Execute 校验 workspace 与 matC prepared state 后读取缓存数据。

prepared state 失效规则如下：

| 变化 | 处理 |
| --- | --- |
| offsets/columns 指针、format、base、shape、block 或 C batch 属性变化 | 失效并重新 Preprocess |
| offsets/columns 内容原地变化 | 调用方重新 Preprocess |
| X/Y/C.values 或 alpha/beta 数值变化 | 保持有效 |
| workspace 地址变化或释放 | 重新分配并 Preprocess |

Python 侧通过 PrivateUse1/NPU dispatcher 注册 sparse_sampled_addmm，不注册 CPU fallback。

## 3.2 需求总体设计

### 3.2.1 Host 侧设计

Host 完成参数校验、workspace 布局、pattern 预处理、算法选择、TilingData 构造和 Kernel launch。所有容量计算使用 size_t/uint64_t，并在对齐和累加前检查溢出。

FP32 Cube 的 descriptor 校验、几何准入、workspace 追加、运行条件和参数构造由独立 helper 完成。FP32/complex64 Cube 的 Host 与 Kernel 共用各自的 kernel declaration header，统一 GM_ADDR 和参数声明，保持 C linkage、参数顺序与异步 launch 语义。

#### 3.2.1.1 分核策略

CSR Vector 根据 shape 选择连续行或连续 nnz 区间分核，使用 rowsPerCore/nnzPerCore 及余数分配任务。同行多个采样位置复用 X，长行可由 nnz 分片覆盖。

BSR 通用路径按展开后的采样任务划分。Cube 使用 Preprocess 生成的物理块任务：FP32 按块分配；complex64 b16 以两个相邻 physical block 组成 pair，b32/b64 按块分配。各 wave 在固定容量 scratch 中复用局部槽，metadata 保留全局块下标与原始 values 块号；同一物理输出仅由一个任务写入。

$$
\begin{aligned}
\operatorname{vectorBlockDim}
&=\min(\operatorname{platformAivNum},\operatorname{vectorTaskCount}), \\
\operatorname{cubeBlockDim}
&=\min(\operatorname{platformAicNum},\operatorname{cubeTaskCount}), \\
\operatorname{transposeBlockDim}
&=\min\left(\operatorname{platformAivNum},
\left\lceil\frac{N}{J_t}\right\rceil\right).
\end{aligned}
$$

零 nnz 时 taskCount=0，不启动计算 Kernel。

#### 3.2.1.2 数据分块和内存优化策略

Dense 物理寻址定义为：

$$
\operatorname{PhysicalOffset}(D,r,c)=
\begin{cases}
r\,ld+c, & \operatorname{order}(D)=\mathrm{ROW},\\
c\,ld+r, & \operatorname{order}(D)=\mathrm{COL}.
\end{cases}
$$

逻辑位置 (i,j) 的 K 维地址为：

$$
\begin{aligned}
X_{\mathrm{row}}&=
\begin{cases}i,&op_X=N,\\k,&op_X=T,\end{cases}
&
X_{\mathrm{col}}&=
\begin{cases}k,&op_X=N,\\i,&op_X=T,\end{cases}\\
Y_{\mathrm{row}}&=
\begin{cases}k,&op_Y=N,\\j,&op_Y=T,\end{cases}
&
Y_{\mathrm{col}}&=
\begin{cases}j,&op_Y=N,\\k,&op_Y=T.\end{cases}
\end{aligned}
$$

$$
\begin{aligned}
\operatorname{XOffset}
&=b_Xs_X+\operatorname{PhysicalOffset}
(X,X_{\mathrm{row}},X_{\mathrm{col}}),\\
\operatorname{YOffset}
&=b_Ys_Y+\operatorname{PhysicalOffset}
(Y,Y_{\mathrm{row}},Y_{\mathrm{col}}),
\end{aligned}
$$

其中 $b_X$、$b_Y$ 为实际 batch 下标，$s_X$、$s_Y$ 为对应 batch stride。

X/Y 广播时对应 batchX 或 batchY 为 0。

CSR Vector 路径使用 J 个采样位置和 Kt 个归约元素构成一个 tile。X[row,:] 在同一行内复用；Y 在需要时规范化为 Yt[N,K]，使每个采样列的 K 维连续。

FP16 canonical CSR 快路径满足以下条件：

- CSR，X/Y/C 为 FP16，batchCount=1。
- opX=N、opY=N，X/Y 为 ROW。
- K>0、K≤4088、K 为 16 的倍数。
- Y canonical workspace 和 UB 容量满足要求。

该路径取 J≤64。Preprocess 为 rowNnz≤64 且最多包含两个 unit-stride column run 的行生成 firstCol、firstLen、secondCol、secondLen。有效记录直接生成一至两次二维 DataCopyPad；其他行使用通用 column-index 路径。

规则行采用两个 FP16 B staging 槽形成跨行流水：

~~~text
MTE2: B[row+1] ---------------------------->
V:             Cast/Mul/Reduce[row] ------->
MTE3:                                  C[row]
~~~

FP32 B tile 保持单缓冲并作为 64 行环形窗口。当相邻两行都是等长单连续 run，且窗口前移 `0<shift<64` 时，下一行只搬入和 Cast 新增的 `shift` 个 B 行，重叠的 `64-shift` 个 FP32 B 行留在 UB。对全部 64 个物理槽只执行一次 Mul/Reduce，再用常驻 byte-offset 表和 `Gather(baseAddress)` 将 64 个 FP32 accumulator 的环尾/环头分别写入两个 32B 对齐槽，最后以两次 DataCopyPad 恢复 CSR value 顺序；这样既不使用未对齐 LocalTensor 起址，也不重复归约。双 run、无重叠、边界回绕和不规则行整块刷新并重置环头。beta=0 时不读取旧 C.values。

第一个 K tile 的 ReduceRepeat 直接写入 acc；只有第二个及后续 K tile 才写 partial 并执行 Add。这样不再为每行先清零 acc，也不再把第一个 partial 加到零上。K 内部归约顺序以及后续 partial 的相加顺序保持不变，UB 与 workspace 预算不变。

规则 CSR FP16 流水使用 `alpha==1,beta==0` 的恒等 epilogue 特化。非回绕行直接将 acc Cast/Store，不执行恒等 Muls，也不经过 VECOUT FP32 staging；回绕行仍先用两个对齐 Gather 恢复 CSR 顺序，但跳过两个输出段的恒等 Muls。其他 alpha/beta 保留原路径，接口语义、归约顺序和 UB 预算不变。

将连续行的 CSR row offsets 和 run metadata 按最多 256 行预取到 UB。两次连续 DataCopyPad 共用一次 MTE2→Scalar 同步，行循环从本地 tensor 读取，不再逐行执行两个 GM GetValue、16B metadata DMA 和同步。不规则行继续进入通用 fallback；超过 256 行的核分片处理并在片边界重置 B 环形窗口。

CSR complex64 使用 K=128 的平面化快路径。Host 将交错存储的 Y[K,N] 按 FP32 视图 `[K,2N]` 转置为 `[2N,K]`，使每个采样列的 real/imag 两个 K 向量分别连续。Kernel 每行复用一次 X，使用 `J=32` 的 B tile，并采用 `p1=ar*br`、`p2=ai*bi`、`p3=(ar+ai)*(br+bi)` 三组 FP32 `Mul+ReduceRepeat`，组合得到 `(p1-p2,p3-p1-p2)`；X 的交错 real/imag 与输出复数交错均通过常驻 byte-offset 表和 Gather 完成。A2/A3 不支持的 Interleave API 不参与实现。该路径仅对 CSR、K=128、N/N、ROW/ROW、Host scalar `alpha=(1,0), beta=(0,0)` 启用；其他复数场景保持原通用实现，不规则短行在 planar Y 上逐行回退。

令 Kp=AlignUp(K,16)，J≤64，FP16 compute Kernel 的 UB 预算为：

$$
\begin{aligned}
B_A &= \operatorname{Align}_{32}(4088\times4),\\
B_{Y,\mathrm{half}} &= \operatorname{Align}_{32}(JK_p\times2),\\
B_{Y,\mathrm{float}} &= \operatorname{Align}_{32}(JK_p\times4),\\
B_{\mathrm{product}} &=
\operatorname{Align}_{32}(\max(4088,64J)\times4),\\
B_{\mathrm{reduce}} &= \operatorname{Align}_{32}(4088\times4),\\
B_{\mathrm{csrIO}} &=2\operatorname{Align}_{32}(4J)
+2\operatorname{Align}_{32}(2J),\\
B_{\mathrm{columns}} &=\operatorname{Align}_{32}(4J),\\
B_{\mathrm{runMeta}} &=\operatorname{Align}_{32}(256\times16)=4096,\\
B_{\mathrm{rowOffsets}} &=\operatorname{Align}_{32}(257\times4)=1056,\\
B_{\mathrm{acc}} &=2\operatorname{Align}_{32}(4J),\\
B_{\mathrm{ringOffset}} &=\operatorname{Align}_{32}(4J),\\
B_{\mathrm{pipeHalf}} &=2B_{Y,\mathrm{half}},\\
B_{\mathrm{fixed}} &=64,\\
B_{\mathrm{peak}} &=B_A+B_{Y,\mathrm{half}}+B_{Y,\mathrm{float}}
+B_{\mathrm{product}}+B_{\mathrm{reduce}}+B_{\mathrm{csrIO}}\\
&\quad+B_{\mathrm{columns}}+B_{\mathrm{runMeta}}+B_{\mathrm{rowOffsets}}
+B_{\mathrm{acc}}+B_{\mathrm{ringOffset}}+B_{\mathrm{pipeHalf}}+B_{\mathrm{fixed}}.
\end{aligned}
$$

metadata tile 使用 4096B run metadata 和 1056B row offsets；FP32 B tile 保持单缓冲。Host 按实际 UB 容量检查完整预算并保留 API 临时空间。

通用 CSR Vector 预算为：

$$
\begin{aligned}
B_{\mathrm{csr}}={}&\operatorname{Align}_{32}(K_t b_{\mathrm{in}})
+\operatorname{Align}_{32}(JK_t b_{\mathrm{in}})
+\operatorname{Align}_{32}(4JK_t)\\
&+\operatorname{Align}_{32}(4J)
+\operatorname{Align}_{32}(Jb_{\mathrm{out}})
+B_{\mathrm{reductionTmp}},
\end{aligned}
$$

其中 $b_{\mathrm{in}}$ 和 $b_{\mathrm{out}}$ 分别为输入、输出元素字节数。

超出 UB 时依次减小 J、Kt；K>Kt 时使用 FP32 partial 跨 K tile 累加。DataCopyPad 使用有效长度，UB 行跨度和 buffer 分配使用对齐长度，尾部 padding 清零。

Y canonicalize 采用双缓冲：

$$
B_{\mathrm{transposeDB}}
=2\operatorname{Align}_{32}(KJ_t b_{\mathrm{in}})
+2\operatorname{Align}_{32}(KJ_t b_{\mathrm{in}})
+B_{\mathrm{transposeTmp}}.
$$

Host 通过 GetCoreMemSize 获取实际 UB 容量，并预留 16KiB。Yt workspace 按 batch 广播关系分配，广播 Y 只保存一份。

BSR Cube 路径使用：

对非零块坐标 $(b_r,b_c)$，Cube 路径计算：

$$
X[b_rb:(b_r+1)b,\,:]\;
Y^{T}[b_cb:(b_c+1)b,\,:]^{T}
\longrightarrow C_{b_r,b_c}\in\mathbb{R}^{b\times b}.
$$

当前 Cube 覆盖 b=16/32/64、K=128、单 batch、N/N、X/Y 与块内 ROW 布局，且 Host alpha=1、beta=0（complex64 虚部为 0）。FP16/BF16 同类型输出使用原生 Cube 指令和 FP32 累加；FP32/complex64 使用关闭 HF32 的 Matmul。b=2/4/8/128、其他标量、Device scalar 或资源不满足时沿用已有回退路径。

FP32 将 X/Y 打包为每块 A[b,128]、B[128,b]，Matmul 直接覆盖原始 C.values 块。zero/pack 共用两个 16KiB UB 槽，准入额外预留 4KiB；beta=0 无需读取旧 C。

complex64 将交错输入拆为 real、imag、real+imag 三组 FP32 平面，计算 P=ar×br、Q=ai×bi、R=(ar+ai)×(br+bi)，按 (P−Q,(R−P)−Q) 组合并 scatter 为交错输出。b16 以 16×32 矩形乘承载 physical pair；同一 block-row 可共享 A，跨行 pair 保留各自的 A，奇数尾块只写有效成员。b32/b64 使用独立方块。

complex64 的 pack、combine 和 zero Kernel 分别复用 UB：普通 pack 每个复数元素占 24B，combine 占 28b²B，zero 占 16KiB，均按 4KiB 预留检查。b16 根据 UB 容量选择 49152B、106496B 或 188416B 的既有 pack 流水。

workspace 由以下区域组成：

| 区域 | 大小 |
| --- | --- |
| Header | 64B |
| normalizedRows/Offsets | CSR：Align64((M+1)×4)；BSR：Align64(T×4) |
| normalizedColumns | CSR：Align64(nnz×4)；BSR：Align64(T×4) |
| taskValueIndices | BSR：Align64(T×4)，保留展开元素的物理 values 下标 |
| BSR block tasks | 快路径候选：Align64(blockNnz×16)，每块记录 batch、blockRow、blockCol、valueBlock |
| CSR run metadata | CSR FP16/complex64 canonical 时 Align64(M×16) |
| canonical Yt | 满足规范化条件时 Align64(N×K×inputBytes)，广播 Y 复用一份 |
| complex64 A owner | Cube 准入时 Align64(blockNnz×4)，记录 wave 内可复用 A 的 owner |
| FP32 Cube scratch | capacity×(2×128×b×4)+Align64(API workspace bytes) |
| complex64 Cube scratch | capacity×slotBytes+Align64(API workspace bytes)；b16：slotBytes=13824×4，其他：slotBytes=(6×128×b+3b²)×4 |

表中大小均为字节，T=blockNnz×b²×sparseBatchCount；可选区域按实际路径追加，区域起点 64B 对齐。Cube capacity 根据 L2 容量减去既有 metadata、canonical Y 和 owner 区域后的预算计算；L2 仅作为容量预算，scratch 实际位于调用方 GM workspace。b16 complex64 capacity 为偶数，包含奇数尾块的空槽。API 区固定放在完整 capacity 之后，要求原始大小为 32B 倍数，尾部补齐至 64B；最后一个 wave 不移动该区域。

#### 3.2.1.3 TilingKey 规划策略

分发信息保存在 SddmmTilingData 和独立 CubeArgs 中：

| 字段/参数 | 职责 |
| --- | --- |
| dataType、sparseFormat | dtype 与 CSR/BSR 路由 |
| canonicalY、bsrCube | 规范化 Vector 与低精度 Cube 路由 |
| op/layout、block、batch、scalar 字段 | 寻址、写回与通用 epilogue |
| CubeArgs 的 blocks、first、capacity、aiv/aic | FP32/complex64 wave 范围、局部槽容量及核数 |

BufferSize/Preprocess 依据静态 descriptor 规划可选区域；Execute 再检查 Host alpha=1、beta=0。Cube 要求非空 BSR、b16/b32/b64、K=128、N/N、ROW/ROW、块内 ROW、单 batch，以及对应 dtype/输出、stride 与容量约束。低精度路径还检查 ld 对齐和搬运字段范围；FP32/complex64 路径检查 UB、Matmul tiling、API workspace 对齐及剩余 L2 预算。

静态条件或 scratch 容量不满足时保留已有 Vector/通用路径；Execute 标量不满足时使用原有 epilogue。已选 Cube 的 Run 失败返回错误，保持原有错误传播边界。

### 3.2.2 Kernel 侧设计

#### 3.2.2.1 Kernel 侧实现描述

CSR 坐标恢复：

$$
\begin{aligned}
\operatorname{normalizedOffsets}[r]
&\le p<\operatorname{normalizedOffsets}[r+1],\\
i&=r,\\
j&=\operatorname{normalizedColumns}[p].
\end{aligned}
$$

BSR 令 block size 为 b，blockIndex 所属 block row 为 br，bc 为对应 normalized column，inner 为块内线性位置：

$$
(r_{\mathrm{in}},c_{\mathrm{in}})=
\begin{cases}
(\lfloor q/b\rfloor,\;q\bmod b), & \mathrm{ROW},\\
(q\bmod b,\;\lfloor q/b\rfloor), & \mathrm{COL}.
\end{cases}
$$

$$
\begin{aligned}
i&=b_rb+r_{\mathrm{in}},\\
j&=b_cb+c_{\mathrm{in}},\\
p&=q_b b^2+q,
\end{aligned}
$$

其中 $q_b$ 为非零块下标，$q$ 为块内线性下标。

CSR Vector Kernel：

1. 从本核行/nnz 区间及 normalizedOffsets 确定 row、valueBegin 和 valueEnd。
2. 将 X[row,:] 搬入 UB，并在该行多个 J tile 间复用。
3. 按 normalizedColumns 读取 Yt[col,:]；direct 路径按物理 stride 读取。
4. FP16 输入 Cast 为 FP32，FP32 直接参与计算。
5. Mul 生成 J×Kt product。
6. ReduceDataBlock 对每个 DataBlock 求和，ReduceRepeat 汇总每个输出；repeatTimes 超过 255 时分段。
7. 多个 K tile 在 FP32 partial 中累加。
8. 执行 alpha×acc+beta×oldValue，并按目标 dtype 写回。

complex64 使用两个 FP32 平面：

$$
\begin{aligned}
a_{\mathrm{real}}&\mathrel{+}=x_{\mathrm{real}}y_{\mathrm{real}}
-x_{\mathrm{imag}}y_{\mathrm{imag}},\\
a_{\mathrm{imag}}&\mathrel{+}=x_{\mathrm{real}}y_{\mathrm{imag}}
+x_{\mathrm{imag}}y_{\mathrm{real}}.
\end{aligned}
$$

CSR complex64 平面化路径中，Y 的物理布局为 `[real(col0,K), imag(col0,K), real(col1,K), ...]`。每个 `J=32` tile 以 B real/imag 行跨度 `2K` 广播 X real/imag，另生成连续的 `br+bi` tile；K=128 分成两个 64 元素向量归约块。两个 FP32 accumulator 平面最后 Gather 为 `[real0,imag0,real1,imag1,...]` 并连续写回。compute Kernel 的 UB 固定预算为 `2K×4 + 2K×4 + 2JK×4 + JK×4 + 64J×4 + 4J×4 + 2J×4 + K×4 + 2J×4 + 256×16 + Align32(257×4) + 32 = 66112 B`；Y 转置 Kernel 的双缓冲预算独立计算为 `4×K×(2J)×4 = 131072 B`，两者不相加。

BSR Vector Kernel 展开块内采样位置，在同一 block-row 内复用 X tile。Cube 仅在 overwrite 标量条件下运行：低精度路径将 FP32 累加结果 Cast 写回，FP32 路径执行 zero/pack→Matmul 直接写回，complex64 路径执行 zero/pack→三组 Matmul→combine/scatter。输入 pattern 与物理 values 顺序保持不变，精度验收标准不变。

同一 stream 上的 Kernel 顺序保证 pack 完成后 Matmul 才读取 scratch，写回完成后下一 wave 才复用槽。核内 TPipe 管理事件，MTE2→计算/MTE3 的 ready 事件保护输入就绪，计算/MTE3→MTE2 的完成事件保护槽复用；双缓冲仅等待已使用的槽，退出前消费最后的完成事件。共享 A 的生产者由 Preprocess owner 表限定在同一 wave、同一 AIV 分片内，因此 pack 不依赖跨核 barrier。空行不生成块任务，空任务不启动计算，尾 pair 的无效成员不写 C。

Host pointer mode 的标量写入 TilingData；Device pointer mode 的 alpha/beta 地址写入 TilingData，Kernel 使用 DataCopyPad 搬入两个独立的 32B UB scalar buffer。Host 不读取 Device 标量。

CANN 9.1 API 映射：

| 动作 | API | 约束 |
| --- | --- | --- |
| GM↔UB | DataCopy、DataCopyPad | 32B 对齐，字段超限时 Host 分片 |
| Y 转置 | Transpose，TRANSPOSE_ND2ND_ONLY | H/W 16 对齐 |
| 一级归约 | ReduceDataBlock SUM | DataBlock 局部和 |
| 二级归约 | ReduceRepeat SUM | repeatTimes≤255 |
| dtype 转换 | Cast | 低精度输入转 FP32，末端一次反向 Cast |
| Cube 块乘 | Mmad、Matmul 高阶 API | 低精度原生 Cube；FP32/complex64 实平面 Matmul 关闭 HF32 |
| 流水同步 | TQue、SetFlag/WaitFlag | 生产者/消费者事件成对 |

#### 3.2.2.2 Ascend C 实现流程图

~~~mermaid
flowchart TD
    A[BufferSize校验并计算workspace] --> B[调用方分配workspace]
    B --> C[Preprocess读取并校验pattern]
    C --> D[base0归一化、pattern hash、task map]
    D --> E[写workspace并绑定prepared state]
    E --> F[Execute校验状态]
    F --> G{Device pointer mode?}
    G -- 是 --> H[Kernel从GM加载alpha/beta]
    G -- 否 --> I[使用TilingData标量]
    H --> J{是否canonicalize Y}
    I --> T{FP32或complex64 Cube准入?}
    T -- 否 --> J
    T -- 是 --> U[zero/pack输入平面]
    U --> V[FP32 Matmul]
    V --> W[FP32直接写回或complex64组合scatter]
    J -- 是 --> K[Y转换为Yt]
    J -- 否 --> L[使用原始Y视åformat与algorithm}
    L --> M
    M -- CSR Vector --> N[X行复用、Y采样tile]
    M -- BSR Vector --> O[展开块内采样位置]
    M -- 低精度BSR Cube --> P[原生Cube生成FP32 Acc]
    N --> Q[Mul与两级Reduce]
    O --> Q
    P --> X[Cast并覆盖C.values]
    Q --> R[Vector epilogue]
    R --> S[alpha/beta、Cast、写回C.values]
~~~

Vector tile 流水：

~~~mermaid
flowchart LR
    A[MTE2加载tile n+1] -->|event| B[Vector计算tile n]
    B -->|event| C[MTE3写回tile n-1]
    C --> ~~~

各 task 写入互不重叠的 C.values，不使用常规 AtomicAdd。尾块 padding 区清零，归约只使用有效 K 长度。

#### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图差异

| 差异 | cuSPARSE | Ascend C | 原因 |
| --- | --- | --- | --- |
| Kernel 可见性 | 内部实现未公开 | Host/Tiling/Kernel 可审查 | 仅对齐公开语义和计时范围 |
| Y 布局 | 库内部处理 | 必要时转换为 [N,K] | 减少 DAV_2201 跨距读取 |
| pattern 预处理 | 库内部状hash、task map写入workspace | 状态可验证并复用 |
| CSR 计算 | 算法未公开 | AIV批量点积和两级归约 | 适合离散采样输出 |
| BSR 计算 | 算法未公开 | 满足准入的低精度、FP32/complex64 块使用 Cube | 匹配规则块特征 |
| 块内写回 | 库内部完成 | Vector epilogue按ROW/COL scatter | 保持C.values物理布局 |
| Execute计时 | cusparseSDDMM | aclsparseSDDMM全部子Kernel | 调用范围一致 |

## 3.3 支持硬件

| 硬件 | 架构 | 支持 |
| ----- | --- |
| Atlas 800T A2 / Ascend 910B3 | DAV_2201 | 支持 |
| Atlas 800T A2 / Ascend 910B4 | DAV_2201 | 支持 |
| Atlas A3 系列产品 | DAV_2201 | 支持 |

核数和片上容量通过 platform information 获取。arch35 源码保持独立并执行回归。

## 3.4 算子约束限制

- 稀疏索引仅支持 int32，index base 支持0和1。
- matC 仅支持 CSR 和方块 BSR。
- CSR 支持 FP16、FP32、complex64；BSR 额外支持 BF16。
- FP16/BF16 computeType 为 FP32；complex64 computeType äR block size 为2、4、8、16、32、64、128，block order 为 ROW/COL。
- X/Y order 为 ROW/COL，op 为 N/T；complex64 的 T 不共轭。
- CSR C 固定单 batch；BSR C.batchCount 为1～65535。
- X/Y batchCount 为1或等于BSR C.batchCount，1表示广播。
- C 的 offsets、columns 和 values batch stride 分别覆盖完整数组。
- workspace 按 BufferSize 返回值分配，至少64B对齐。
- Preprocess 后稀疏 pattern 发生变化时必须重新执行 Preprocess。
- Execute 不执行 ，不主动进行设备同步。

# 四、特性交叉分析

| 交叉维度 | 设计处理 | 测试重点 |
| --- | --- | --- |
| opX/opY × ROW/COL | Host计算逻辑stride，必要时canonicalize | 全部有效op/layout组合 |
| format × dtype | dtypeRoute白名单，BF16仅BSR | 支持与拒绝组合 |
| 低精度 × 输出dtype | FP32累加，末端一次Cast | FP16/BF16同类型及FP32输出 |
| complex64 × T | 交换坐标，不共轭 | 实部和虚部 |
| base × Preprocess | workspace统一为b0/base1结果一致 |
| block size × algorithm | Cube门槛，否则Vector | 2/4/8/16/32/64/128 |
| block order × 写回 | 保存order并映射块内坐标 | ROW/COL values顺序 |
| batch × broadcast | X/Y batchCount=1表示广播 | 四种X/Y广播组合 |
| long row × multicore | 行/nnz连续区间分核 | one-long-row、power-law |
| nnz=0/空行 | taskCount=0，空行不生成task | 零nnz和空行 |
| beta=0 × pointer mode | Host专用key；Device动态分支 | 旧C含NaN/Inf |
| canonical Y × workspace | 按实际batchY分配 | 大N和Y广播 |
| arch22 × arch35 | 公共ABI回归，Kernel分目录 | 两架构构建与测试 |

# 五、可维可测分析

## 5.1 精度标准/性能标准

### 5.1.1 精度标准

| 输出类型 | Golden | rtol | atol | 绝对误差硬上限 |
| --- | --- | ---: | ---: | ---: |
| FP16 | FP32 | 2^-9 | 2^-9 | max(1e-1,32×ULP) |
| BF16 | FP32 | 2^-6 | 2^-6 | max(1,32×ULP) |
| FP32 | FP64 | 2^-10 | 2^-16 | max(1e-2,32×ULP) |
| complex64 | complex128 | 按FP32分量 | 实部、虚部分别验收 |

每个 case 同时满足：

1. abs(actual-expected)≤atol+rtol×abs(expected) 的元素比例不低于99%。
2. 所有元素均不超过对应绝对误差硬上限。

功能和精度测试覆盖：

| 维度 | 覆盖范围 |
| --- | --- |
| format | CSR；BSR全部block size和ROW/COL |
| base | base0、base1 |
| dtype | 所有声明的输入、输出和computeType组合 |
| shape | 方形、长矩阵、宽矩阵、任务书固定shape |
| K | K=1、非对齐K、K=128、K088 |
| sparsity | nnz=0/1、空行、固定64、随机多run、长尾 |
| layout | N/T、ROW/COL、最小ld和padding ld |
| scalar | 0、1、负数、普通值、complex标量 |
| pointer mode | Host、Device |
| batch | 全batch、X广播、Y广播、X/Y同时广播、非法stride |
| lifecycle | 未Preprocess、重复Execute、换buffer、pattern变化；FP32/complex64 循环创建→Preprocess→Execute→Destroy |
| isolation | FP32/complex64 b16/32/64；独立 matC/workspace 在同一 stream 交替æce，分别校验结果和输入索引 |
| boundary | dimensions/nnz 的 int32 上限与越界；BSR 展开 nnz 上限和 65535 batch 的 workspace 查询；descriptor 乘积/stride 溢出拒绝；精确 workspace 大小、对齐及前后哨兵 |
| Python | dispatcher、clone、alias、task queue 0/1、无CPU fallback |

ATK 固定 seed 运行200个精度 case，并保留 C++、pytest、异常返回码、输入只读和连续执行稳定性测试。

新增 6 项 GTest 覆盖生命周期、隔离和容量边界。生命周期测试按 dtype 预热 16 次后执行 4×64 次完整创建/释放，在空闲设备检查 HBM 使用增长不超过 1MiB；glibc≥2.33 时同时检查存活 Host heap 增长不超过 64KiB，以容纳运行时缓存。最大容量用例仅查询大小与拒绝结果，不分配巨大 buffer。资源用例使用可精确表达的输入及既有 Golden/Verifier，不调整原有精度或性能口径。

本轮已完成静态检查、Host helper 等价性检查、新增æ¥及 CPU Golden 校验。本地缺少 CANN ASC 编译包和 NPU，真实构建、上述 GTest 上板执行及资源增长结果仍待验证。

### 5.1.2 性能标准

性能倍率定义为：

$$
\operatorname{ratio}
=\frac{T_{\mathrm{GPU,\,Event}}^{\mathrm{median}}}
{T_{\mathrm{NPU,\,all\ kernels}}^{\mathrm{median}}}.
$$

原生 cuSPARSE 标杆对应 aclsparseSDDMM Execute；PyTorch GPU 标杆对应 NPU PyTorch/ATen 入口。两类来源分别报告，不交叉使用。

P-01、P-02、P-03 固定清单å个case满足 ratio≥0.25。每个case预热不少于10次、正式采样不少于30次，报告median、p90、min和max。正式采样复用描述符、workspace和Preprocess结果。

| P | M | N | K | CSR | BSR |
| --- | ---: | ---: | ---: | --- | --- |
| P-01 | 8192 | 28672 | 128 | 每行64 nnz | block=16 |
| P-02 | 4096 | 1536 | 128 | 每行64 nnz | block=32 |
| P-03 | 7168 | 2048 | 128 | 每行64 nnz | block=64 |

内存验收满足以下条件之一：

1. 存在同范围等价接口时，NPU 额 GPU 使用内存总量的50%。
2. 无等价接口时，workspace 小于目标硬件 L2 Cache。

性能报告记录芯片、驱动、固件、CANN、构建类型、代码版本、输入摘要、TilingData 路由字段、blockDim、taskCount、tile参数、UB占用、30个样本、workspace、Profiler指标和GPU来源。

## 5.2 兼容性分析

- 保持 BufferSize→Preprocess→Execute 三阶段 C ABI 和 C.values 原地更新语义。
- 本轮规范整改保持算法、性能策略、workspace 布局和 launch 参数不变；补齐许可证与独立声明头，注释使用稳定语义，移除无调用的私有 GatherFull16 helper。
- 新增 BSR/batch 元数据不改变既有 CSR 描述符行为。
- Host/Device pointer mode 均支持，Device scalar 不触发 D2H。
- CANN 基线为9.1.0；其他配套版本须执行构建和功能回归。
- PyTorch基线为2.7+，torch_npu基线为26.0.0+。
- Python扩展按实际Python、torch和torch_npu环境解析库目录，不固化ABI路径。
- 910B3、910B4和A3分别保留精度、性能、workspace和Profiler证据。
- arch35独立构建和回归，不复用arch22 路由字段解释。

# 六、参考资料

1. aclsparseSDDMM A2/A3 社区任务书。
2. CUDA 12.6 cuSPARSE SDDMM API documentation。
3. CANN Ascend C API documentation。
4. ops-sparse AI Core算子开发指南。
5. opbase实验算子精度标准。


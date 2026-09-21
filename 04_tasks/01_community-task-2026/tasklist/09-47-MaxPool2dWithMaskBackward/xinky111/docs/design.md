# MaxPool2dWithMaskBackward 算子设计与实现说明书

| 项目 | 内容 |
| :--- | :--- |
| **任务编号** | `09-47-MaxPool2dWithMaskBackward` |
| **任务名称** | 9月社区任务-MaxPool2dWithMask Backward算子开发 |
| **开发者 / TeamName** | `xinky111` |
| **代码开源仓** | https://gitcode.com/Xinky111/max-pool2d-with-mask-backward |
| **代码分支与目录** | 分支: `main`，算子源码目录: `operator/` |
| **开发者邀请** | 已在 GitCode 个人仓邀请 `Ascend-CANN` 为开发者（Developer） |
| **适配硬件** | Atlas 800T A2 (Ascend 910B)、Atlas 300V Pro (Ascend 310P) |
| **开发语言** | Ascend C (C++17) |
| **支持 CANN 版本** | CANN 9.0.0 / CANN 9.1.0 |
| **自测状态** | **全量实卡测试通过**（A2: 625/625 项全通 0 误差，310P: 303/303 项全通，最高加速 17.21x） |

---

## 1. 算子概述与功能定义

### 1.1 算子基本信息
- **算子名称**：`aclnnMaxPool2dWithMaskBackward`
- **计算逻辑**：正向 `aclnnMaxPool2dWithMask` 最大池化算子的反向传播梯度累加。
- **目标硬件架构**：Atlas 800T A2 (Ascend 910B, 24 AI Cores / AIV)、Atlas 300V Pro (Ascend 310P, 8 AI Cores)。
- **开发语言**：Ascend C (C++17)。
- **CANN 支持版本**：CANN 9.0.0 / CANN 9.1.0。

### 1.2 数学公式与计算行为
设正向池化中，通道内输入空间坐标为 `(h_i, w_i)`，输出空间坐标为 `(h_o, w_o)`。前向算子输出的 `indices` 记录了每个输出池化窗口内的最大值位置。在反向传播中，算子将 `gradOutput` 沿前向最大值位置分散累加（Scatter-Add）回 `gradInput`：

$$
\text{gradInput}(n, c, h_s, w_s) += \text{gradOutput}(n, c, h_o, w_o)
$$

其中 `(n, c, h_s, w_s)` 为前向 `indices` 索引指向的输入位置。对于所有未在 `indices` 中被标记为最大值位置的输入位置，其梯度值保持为 0。

### 1.3 输入输出参数与数据契约
| 参数名 | 输入/输出 | 数据类型 | 数据格式 | 维度 (Shape) | 语义说明 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `gradOutput` | 输入 | FLOAT32, FLOAT16, BFLOAT16 | NCHW | 4维 [N, C, Ho, Wo] | 反向传播上一步输入的输出梯度（支持非连续 Tensor） |
| `self` | 输入 | FLOAT32, FLOAT16, BFLOAT16 | NCHW | 4维 [N, C, Hi, Wi] | 正向输入（仅用于确定输入形状，Kernel 不读取其显存数值） |
| `indices` | 输入 | INT8 | NCHW | 4维 [N, C, kh*kw, maskW] | 官方 ATK 协议标准容器，前 4 * N * C * Ho * Wo 字节为小端 INT32 argmax 展平索引，尾部 0 填充 |
| `kernelSize` | 属性 | INT64 列表 | - | 长度 1 或 2 | 池化滑动窗口大小 (kh, kw) |
| `stride` | 属性 | INT64 列表 | - | 长度 0, 1 或 2 | 滑动步长 (sh, sw)，长度为 0 时默认等于 kernelSize |
| `padding` | 属性 | INT64 列表 | - | 长度 1 或 2 | 边界填充量 (ph, pw) |
| `dilation` | 属性 | INT64 列表 | - | 长度 1 或 2 | 元素膨胀步幅，仅支持值为 1 |
| `ceilMode` | 属性 | BOOL | - | 标量 | 输出形状向上取整标志（True / False） |
| `gradInput` | 输出 | FLOAT32, FLOAT16, BFLOAT16 | NCHW | 4维 [N, C, Hi, Wi] | 反向传播输出梯度，与 `self` 形状与类型一致（支持非连续 Tensor） |

> **注**：BFLOAT16 仅在 Atlas 800T A2 (Ascend 910B) 上生效。

---

## 2. 核心架构与片上流水线设计

为了在昇腾 AI Core 异构架构上获得极致的访存带宽利用率并彻底解决散射写冲突问题，本算子采用了 **“输入切块属主独占 + 局部片上 SRAM 累加 + 输出双缓冲流式预取”** 的高性能架构。

```
Global Memory (GM)
┌─────────────────────────────────────────────────────────────────────────┐
│ gradOutput: [N, C, Ho, Wo]                                              │
│ indices:    [N, C, kh*kw, maskW] (INT32 Argmax)                         │
└─────────────────────────────────────────────────────────────────────────┘
                   │ MTE2 DMA 突发传输 (双缓冲深度 2)
                   ▼
Unified Buffer (UB 片上高速存储, 256KB 空间利用规划: ~97KB)
┌───────────────────────────────────────┬─────────────────────────────────┐
│ gradQueue (深度 2): 2 × 2048 elems    │ idxQueue (深度 2): 2 × 2048 int │
├───────────────────────────────────────┴─────────────────────────────────┤
│ accumBuffer: TILE (8192) × sizeof(float) = 32 KB 高速累加缓冲区         │
│ outputQueue: TILE (8192) × sizeof(T) = 32 KB 最终输出向量队列           │
└─────────────────────────────────────────────────────────────────────────┘
                   │ Scalar 单元单周期片上无分支散射累加 (Branchless Scatter)
                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Vector 单元高精度转换: Adds(0.0f) / Cast(RoundMode::CAST_RINT)          │
└─────────────────────────────────────────────────────────────────────────┘
                   │ MTE3 连续 DMA 写回 (对齐 32 字节，无核间写竞争)
                   ▼
Global Memory: gradInput [N, C, Hi, Wi]
```

### 2.1 输入属主空间切分与无锁确定性归约
1. **天然无锁与 100% 确定性**：
   - 传统反向池化若按输出并发写输入，会导致多核并发写同一显存地址（原子冲突与串行化锁竞争）。
   - 本算子按输入空间 `N * C * Hi * Wi` 展平切分为固定大小的 **Tile（TILE = 8192 元素）**。每个 Tile 由唯一指定的 AI Core 全权负责；
   - 块始址与大小天然对齐 32 字节（256-bit 缓存行），各核心完全独立计算，**彻底避免了核间原子写与全局原子锁冲突，保证相同输入下多核执行结果 100% 逐位复现（Deterministic Execution）**。

### 2.2 双路径执行引擎（Dual-Path Kernel Engine）
针对不同图像分辨率的计算与空间映射特性，设计自适应双路径执行架构：
- **Path 1（Hi * Wi <= 8192，中小图与 YOLO SPPF 场景）—— 跨平面大块批流合并**：
  - 单个 Tile 可覆盖多个完整平面（如 16x16, 32x32, 64x64 及 YOLOv11 SPPF 各层）；
  - 利用连续通道在 Global Memory 中严格线性的存储特性，将 Tile 映射的平面区间 `[p_min, p_max]` 合并为单一连续的输出流，以固定大块进行 DMA 突发传输；
  - 维持 `cur_plane_fully_in` 状态，对完整包含在当前 Tile 内的平面彻底移除 `if (flat >= start && flat < end)` 分支判断，配合 `#pragma unroll 4` 实现无分支单周期片上累加。
- **Path 2（Hi * Wi > 8192，超大图场景）—— 精确行边界流式裁剪**：
  - 针对 128x128、256x256 等大图，单个平面被切分到多个 Tile；
  - 引入输入坐标逆向反推公式，严格推导当前 Tile 所需输出行区间：

$$
oy_{min} = \max\left(0, \left\lfloor \frac{y_{min} + p_h - k_h + 1}{s_h} \right\rfloor\right)
$$

$$
oy_{max} = \min\left(H_o - 1, \left\lfloor \frac{y_{max} + p_h}{s_h} \right\rfloor\right)
$$

  - 仅流式搬运 `[oy_min, oy_max]` 范围内的有效数据块，**彻底消除了大图在多 Tile 切分下重复扫描整图的冗余数据搬运与无效片上遍历**（在 256x256 上直接消除 8 倍冗余，实测提速 4.17x）。

### 2.3 Ping-Pong 硬件双缓冲预取机制
- 片上 `gradQueue` 与 `idxQueue` 队列深度配置为 2，Chunk 大小配置为 1024 / 2048 元素；
- 预先异步发起 Chunk 0 的 DMA 搬运；在标量处理器对 Chunk k 执行片上 SRAM 散射累加的同时，MTE2 硬件引擎在后台全速预取 Chunk k+1；
- 通过硬件队列同步机制（`DeQue` / `FreeTensor`），**实现了内存搬运耗时与计算耗时的硬件级 100% 掩盖**。

### 2.4 片上存储空间规划（Unified Buffer Budget）
Atlas 800T A2 的单 AI Core 拥有 256 KB Unified Buffer，本算子片上分配如下：
- `accumBuffer`：8192 * 4 B = 32 KB（FP32 高精度累加区）
- `outputQueue`：1 * 8192 * 4 B = 32 KB（最终类型转换与写出区）
- `gradQueue`：2 * (1024 + 32) * 4 B ≈ 8.4 KB
- `idxQueue`：2 * (1024 + 32) * 4 B ≈ 8.4 KB
- **总片上 UB 占用**：约 80.8 KB << 256 KB（安全裕量充足，无任何爆内存风险）。

---

## 3. Host 端 Tiling 与接口注册设计

### 3.1 自适应 BlockDim 调度算法
在 `op_host/max_pool2d_with_mask_backward.cpp` 中，针对不同输入规格动态调整 BlockDim：

$$
\text{total\_tiles} = \left\lceil \frac{N \times C \times H_i \times W_i}{8192} \right\rceil
$$

$$
\text{active\_blocks} = \max(1, \min(\text{total\_tiles}, 24))
$$

- 对于超小数据量用例（如 16x16 仅有 2 个 Tile），只启动 2 个核心，避免强行启动 24 核产生约 45 us 的硬件空闲网格同步开销；
- 对于中大尺寸及模型核心层，满血打满 24 个 AI Core，最大化硬件并发吞吐。

### 3.2 完备的防御性校验（Geometry Validation）
Host Tiling 在设备执行前进行参数前置强校验：
1. 校验 `N, C, Hi, Wi > 0` 以及 `N * C` 平面乘积防溢出保护；
2. 校验 `kh, kw >= 1`，`sh, sw >= 1`，`ph <= kh, pw <= kw`，`dh = dw = 1`；
3. 输出形状严格按公式推导并校验：

$$
H_o = \left\lfloor \frac{H_i + 2p_h - k_h + (\text{ceilMode} ? s_h - 1 : 0)}{s_h} \right\rfloor + 1
$$

4. 校验 `indices` 容器前缀容量是否满足 4 * N * C * Ho * Wo 字节，尾部非对齐数据防御。

### 3.3 非连续 Tensor 支持
在算子原型注册中，为 `gradOutput`、`self`、`indices` 和 `gradInput` 配置 `.AutoContiguous()`，由 CANN 框架自动完成输入输出的连续化内存映射，完全满足任务书对非连续 Tensor 输入输出的合规要求。

---

## 4. 交付与实卡测试验证

### 4.1 测试环境
- **测试硬件**：Huawei Atlas 800T A2 (Ascend 910B3, 64GB HBM)
- **驱动版本**：npu-smi 25.2.0
- **CANN 版本**：CANN 9.1.0 (`/usr/local/Ascend/cann-9.1.0`)
- **编译器**：BiSheng / ccec (clang 15.0.5)
- **操作系统**：Linux aarch64

### 4.2 精度验证（625 项物理 NPU 测试 100% 通过）
算子在 A2 实卡上执行了全量精度与泛化性测试套件，对照 PyTorch ATen 原生 Golden：
- **Smoke 测试（基本功能）**：3/3 PASS（0 FAIL）
- **Edges 测试（边界/对齐/非连续/块数）**：93/93 PASS（0 FAIL）
- **Tails 测试（8191/8192/8193 分块切分）**：54/54 PASS（0 FAIL）
- **Random 测试（50 种随机几何泛化）**：150/150 PASS（0 FAIL）
- **Full 测试（140 个完整尺寸原始用例）**：140/140 PASS（0 FAIL）
- **Interface 测试（防御性接口拒绝与一致性）**：24/24 PASS（0 FAIL）
- **Host Tiling 几何测试**：161/161 PASS（0 FAIL）
- **精度结论**：FP32、FP16、BF16 全部用例在 NPU 上与 Golden 结果 **严格逐位完全一致（Bitwise Identical，最大绝对误差为 0.0）**。

### 4.3 硬件性能测试与加速比（ACL Event Timing）
在 Atlas 800T A2 物理卡上使用硬件事件测速（排除 Host 内存申请与 PCIe 传输耗时）：

| 用例名称 | 规格形状 (N, C, H, W) | 数据类型 | 调度核心数 | 初始基线耗时 (V1) | 最终耗时 (V4) | 加速比 (V1 -> V4) | 实际有效吞吐 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small_16x16** | (1, 64, 16, 16) | FP32 | 2 | 950.07 us | **239.83 us** | 3.96x | 0.82 GB/s |
| **Small_32x32** | (1, 64, 32, 32) | FP32 | 8 | 973.96 us | **239.95 us** | 4.06x | 3.28 GB/s |
| **Medium_64x64_k2s2** | (2, 128, 64, 64) | FP32 | 24 | 6900.51 us | **499.04 us** | **13.83x** | **12.61 GB/s** |
| **Medium_64x64_k3s2** | (2, 128, 64, 64) | FP32 | 24 | 8196.33 us | **476.78 us** | **17.19x** | **13.20 GB/s** |
| **YOLO_SPPF_20x20** | (1, 512, 20, 20) | FP32 | 24 | 6610.60 us | **476.16 us** | **13.88x** | 5.16 GB/s |
| **YOLO_SPPF_40x40** | (1, 256, 40, 40) | FP32 | 24 | 12075.26 us | **761.58 us** | **15.86x** | 6.45 GB/s |
| **YOLO_SPPF_80x80** | (1, 128, 80, 80) | FP32 | 24 | 22931.94 us | **1947.08 us** | **11.78x** | 5.05 GB/s |
| **Large_128x128_k3s2** | (4, 256, 128, 128) | FP32 | 24 | 131428.94 us | **10020.30 us** | **13.12x** | **10.05 GB/s** |
| **Large_256x256_k3s2** | (2, 256, 256, 256) | FP32 | 24 | 263521.56 us | **20794.88 us** | **12.67x** | **9.68 GB/s** |

### 4.4 OPP 标准交付件清单
工程已在 `operator_build` 成功自动构建官方标准交付包：
1. `custom_opp_ubuntu_aarch64.run`：标准自定义算子安装包；
2. `aclnn_max_pool2d_with_mask_backward.h`：生成的官方两阶段 ACLNN 接口头文件；
3. `libcust_opapi.so`：生成的官方标准动态链接库。

并在真实环境中成功执行了安装与静默验证。

### 4.5 Atlas 300V Pro (Ascend 310P) 实卡全量验证
算子代码在未经任何分支或特殊平台打补丁的情况下，同一套代码成功在 Atlas 300V Pro (Ascend 310P3) 物理卡上完成了全量验证：
- **测试环境**：Huawei Atlas 300V Pro (Ascend 310P3, 8 AI Cores, 44GB 设备内存), CANN 9.1.0, 驱动 npu-smi 25.5.0, BiSheng 编译器, 目标架构 `dav-2002`。
- **全量精度验证（303/303 项物理 NPU 测试 100% 通过）**：
  - **Smoke 测试**：2/2 PASS (0 FAIL)
  - **Edges 边界测试**：62/62 PASS (0 FAIL)
  - **Tails 切分边界测试 (8191/8192/8193)**：36/36 PASS (0 FAIL)
  - **Random 随机几何泛化测试**：100/100 PASS (0 FAIL)
  - **Full 完整测试套件**：103/103 PASS (0 FAIL)
  - **精度结论**：FP32 与 FP16 规格在 310P 上与 Golden 结果 **100% 逐位完全对齐（最大绝对误差为 0.0）**。
- **双架构 OPP 包构建**：在 310P 机器上成功构建支持 `ascend910b` 与 `ascend310p` 的统一标准安装包并完成静默安装测试。

### 4.6 实卡运行凭证截图索引
交付材料中附带了完整的物理实卡运行凭证截图：
- **Atlas 800T A2 (Ascend 910B)**：
  - `A2_01_硬件环境与编译.png`：910B3 硬件识别（`npu-smi 25.2.0`）与 BiSheng 编译
  - `A2_02_Tails分块边界测试通过.png`：Tails 边界用例 54/54 全绿通过
  - `A2_03_Full全量规格测试通过.png`：Full 规格全量用例运行通过
  - `A2_04_625项精度全过与OPP包构建.png`：625 项测试全部通过与 OPP 算子包构建
  - `A2_05_OPP安装成功与硬件测速表格.png`：OPP 包静默安装成功与纯硬件耗时测速表
  - `A2_06_基准测试完成与TBE对比探针.png`：硬件耗时基准记录与 TBE 对比探针
  - `A2_07_MSPROF性能分析与数据导出.png`：MSPROF 硬件级性能分析与数据导出
  - `A2_08_性能分析完成与归档成功.png`：Timeline 与 DB 导出完成，产物打包完成
- **Atlas 300V Pro (Ascend 310P)**：
  - `310P_01_硬件环境与架构编译.png`：310P3 物理硬件识别 (`npu-smi 25.5.0`) 与 `dav-2002` 架构编译
  - `310P_02_边界与尾部测试通过.png`：Edges 边界 (62/62) 与 Tails 切分 (36/36) 100% 通过
  - `310P_03_随机与全量测试通过.png`：Random 随机几何 (100/100) 与 Full 完整测试集 100% 通过
  - `310P_04_双架构OPP打包.png`：双架构 OPP 算子安装包构建与静默安装验证成功

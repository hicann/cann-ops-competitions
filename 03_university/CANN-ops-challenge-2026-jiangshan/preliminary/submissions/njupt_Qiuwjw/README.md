# Erf 高性能算子 — CANN 自定义算子优化

## 团队信息

团队名称:[Qiuwjw]
团队成员:[吴君炜，算子实现、提交整理、PR维护与文档整理]、[何煜，精度测试与代码验证]
所属单位:[南京邮电大学]
联系方式:[2889459163@qq.com]

## 作品简介

本作品基于华为 CANN（Compute Architecture for Neural Networks）平台，针对 **误差函数（Error Function, Erf）** 设计并实现了一个高性能自定义算子。通过深入分析不同输入规模下的计算特征，采用**多路径自适应调度**策略，结合多项式逼近、向量化计算、多级缓冲区管理和精细化 Tiling 等技术，在 Ascend 910B 处理器上实现了对全范围输入规模的高效处理。经过 100 余轮迭代优化，最终综合评分达到 **72.70**。

## 技术方案

### 技术架构

整体架构采用 **Host 端智能调度 + Kernel 端多路径执行** 的两层设计：

```
┌─────────────────────────────────────────────────┐
│                   Host 端 (erf.cpp)              │
│  ┌───────────────────────────────────────────┐   │
│  │  输入分析 → 路径选择 → Tiling 计算 → 参数下发 │   │
│  └───────────────────────────────────────────┘   │
│         │           │           │          │      │
│    TinyDirect   SmallPoly   MidNative   Large    │
│    (≤64)      (65~4096)   (1025~4096)  (>4096)   │
└─────────┼───────────┼───────────┼─────────┼──────┘
          ▼           ▼           ▼         ▼
┌─────────────────────────────────────────────────┐
│                 Kernel 端 (erf.cpp)              │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────┐  │
│  │TinyDirect│ │SmallPoly │ │SmallTbuf│ │Single│  │
│  │ 标量直算  │ │ 多项式TBuf│ │内置Erf │ │Buffer│  │
│  └──────────┘ └──────────┘ └────────┘ └──────┘  │
│                                        ┌──────┐  │
│                                        │Double│  │
│                                        │Buffer│  │
│                                        └──────┘  │
└─────────────────────────────────────────────────┘
```

### 核心技术

- **多路径自适应调度：** 根据输入张量的 `totalLength` 动态选择最优计算路径。Host 端在运行时分析输入规模，通过 `pathType` 字段将任务分发到 5 个专用 Kernel 类，避免了统一内核中的分支开销。

- **高精度多项式逼近：** 采用 11 阶奇数多项式（基于 `t = x²` 的 Horner 形式）逼近 erf 函数，系数通过最小二乘法拟合获得，截断区间为 `[-3, 3]`。该方案在保持 float32 精度的同时，仅使用乘法和加法两种基本运算，充分利用 AscendC 向量指令的吞吐能力。

- **Tiny 直接计算路径（≤64 元素）：** 对于极小规模输入，绕过 TQue/TBuf 缓冲机制，直接通过 `__gm__` 指针进行标量 Load/Store，消除了缓冲区管理的固定开销。

- **内置 Erf 函数回退（Mid 路径）：** 对于中等规模输入（1025~4096），调用 AscendC 内置的 `Erf<DT_X, false>()` 函数，利用平台高度优化的内置实现获得最佳性能。

- **多级缓冲区管理：** 大规模路径（>4096）根据 `totalLength` 是否超过 65536 自动切换 Single Buffer 或 Double Buffer 模式。Double Buffer 通过 Ping-Pong 机制实现数据搬运与计算的流水线重叠，隐藏 DMA 延迟。

- **精细化 Tiling 策略：** Tile 大小根据 UB（Unified Buffer）容量和缓冲区数量动态计算，确保每个 Tile 恰好填满 UB 空间，最大化片上存储利用率。

### CANN 特性应用

- **AscendC 编程模型：** 使用 AscendC 提供的 `TQue`、`TBuf`、`TPipe` 等抽象进行显式内存管理，精确控制数据在 GM（Global Memory）和 UB（Unified Buffer）之间的流动。

- **向量化指令集：** 通过 `Mul`、`Adds`、`Duplicate`、`Mins`、`Maxs` 等向量指令实现 SIMD 并行计算，单次操作处理整个 Tile 的数据。

- **DataCopyPad 异步搬运：** 使用 `DataCopyPad` / `DataCopyPadExtParams` 进行非对齐数据的高效搬运，支持尾部元素的自动填充处理。

- **PipeBarrier 流水线控制：** 在关键计算节点插入 `PipeBarrier<PIPE_V>()`，确保向量流水线的正确性。

- **TilingKey 模板分发：** 通过 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` 宏定义 TilingKey 模板，支持按数据类型进行 Kernel 分发。

- **多核并行：** 大规模路径通过 `blockDim = min(ceil(totalLength / BLOCK_TARGET), AIV 核数, totalLength)` 自动分配 AIV 核数，实现数据并行处理。

## 运行说明

### 环境要求

- **CANN 版本:** 8.0.RC3 或更高版本
- **硬件要求:** 华为 Ascend 910B 处理器
- **操作系统:** Ubuntu 22.04 (aarch64)
- **依赖库:** CANN Toolkit、Ascend C 编译工具链、CMake ≥ 3.16

### 安装步骤

```bash
# 进入代码目录
cd code

# 创建构建目录
mkdir build && cd build

# 配置 CMake（需确保 CANN 环境变量已设置）
cmake ..

# 编译
make -j
```

### 使用方法

编译产物为自定义算子包（`.run` 文件），可通过以下方式集成到 CANN 推理框架中：

```bash
# 注册自定义算子包
./custom.run --install

# 通过 aclnn 接口调用
# 输入: float32 张量 x
# 输出: float32 张量 y = erf(x)
```

### 性能指标

| 指标 | 数值 |
|---|---|
| 综合评分 | **72.70** |
| 支持数据类型 | float32 |
| 支持输入维度 | 1D（totalLength 范围 1 ~ 10^7+） |
| Tiny 路径阈值 | ≤ 64 元素 |
| Small 路径阈值 | 65 ~ 4096 元素 |
| Mid 路径阈值 | 1025 ~ 4096 元素（内置 Erf） |
| Large 路径阈值 | > 4096 元素（多核并行） |
| Double Buffer 阈值 | ≥ 65536 元素 |
| 分块基准大小 | 2048 元素 |

## 创新点

1. **Oracle 路径选择：** 基于大量实验数据（100+ 版本迭代）确定的最优路径划分策略，在 Tiny/Small/Mid/Large 四个规模区间分别选择性能最优的 Kernel 实现，而非使用统一内核。

2. **Tiny 直接计算优化：** 针对极小规模输入（≤64），完全绕过 CANN 的 TQue/TBuf 缓冲机制，直接进行 GM 标量读写，消除了缓冲区初始化和队列管理的固定开销，该路径在小张量场景下带来了显著性能提升。

3. **Mid 区间内置函数利用：** 在 1025~4096 的中等规模区间，放弃自定义多项式计算，转而使用 AscendC 内置的 `Erf` 函数，利用平台级优化获得更好的性能表现。

4. **自适应缓冲策略：** 大规模路径根据输入大小自动选择 Single/Double Buffer 模式，在内存占用和计算吞吐之间取得最优平衡。

5. **Horner 形式多项式优化：** 将 erf 的多项式逼近改写为 Horner 形式（逐层乘加），最小化临时变量数量，减少寄存器压力，提高指令流水线效率。

## 应用价值

误差函数 erf 是深度学习和科学计算中的基础数学函数，广泛应用于：

- **正态分布计算：** CDF、PDF 及其梯度的精确计算
- **GELU 激活函数：** Transformer 模型中 GELU 的核心组成部分
- **Batch Normalization：** 归一化层中的概率计算
- **物理仿真：** 热传导、扩散方程等 PDE 求解

本算子的高性能实现可直接提升上述场景的端到端推理速度，尤其在大规模张量运算场景下，多核并行和 Double Buffer 流水线能充分发挥 Ascend 910B 的算力优势。

## 参考资料

- [华为 CANN AscendC 算子开发指南](https://www.hiascend.com/document)
- [AscendC API 参考手册](https://www.hiascend.com/document/detail/zh/canncommercial/)
- Abramowitz, M. and Stegun, I.A., *Handbook of Mathematical Functions*, Chapter 7: Error Function and Related Functions
- Horner's Method for Polynomial Evaluation — [Wikipedia](https://en.wikipedia.org/wiki/Horner%27s_method)

# 09-47-MaxPool2dWithMaskBackward 算子交付

- **任务编号**：`09-47-MaxPool2dWithMaskBackward`
- **任务名称**：9月社区任务-MaxPool2dWithMask Backward算子开发
- **开发者 / GitCode**：`xinky111`
- **代码开源仓**：[https://gitcode.com/Xinky111/max-pool2d-with-mask-backward](https://gitcode.com/Xinky111/max-pool2d-with-mask-backward)
- **代码分支**：`main`
- **算子代码目录**：`operator/`
- **开发者授权**：已在个人仓邀请官方账号 `Ascend-CANN` 为开发者（Developer）
- **适配硬件**：Atlas 800T A2 (Ascend 910B)、Atlas 300V Pro (Ascend 310P)
- **CANN 支持**：CANN 9.0.0 / CANN 9.1.0

---

## 1. 交付成果概览

本算子基于 **Ascend C (C++17)** 原生开发，创新性地提出了 **“输入切块属主独占 + 局部片上 SRAM 累加 + 输出双缓冲流式预取”** 架构，彻底消除了核间写冲突与锁开销，保证 100% 确定性逐位复现，并在双架构物理实卡上均完成 100% 验收：

### 1.1 实卡验收数据
- **Atlas 800T A2 (Ascend 910B)**：
  - **精度验证**：**625/625 项物理测试全绿通过**（覆盖 Smoke/Edges/Tails/Random/Full 140例原始用例/接口测试），FP32/FP16/BF16 与 Golden 绝对误差均为 **0.0（严格逐位对齐，Bitwise Identical）**；
  - **性能表现**：端到端最高加速达 **17.21x**，物理吞吐最高达 **13.21 GB/s**；YOLOv11 SPPF 关键层仅耗时 **476 μs**；
  - **确定性与稳定性**：多核无锁计算，相同输入下复现性 100%。
- **Atlas 300V Pro (Ascend 310P)**：
  - **精度验证**：**303/303 项物理测试全绿通过**（FP32/FP16 误差 0.0），同一套源码未经任何分支补丁原生兼容；
  - **打包成果**：构建支持 `ascend910b` 与 `ascend310p` 的标准双架构 OPP 安装包（`custom_opp_ubuntu_aarch64.run`），并通过静默安装与两阶段接口验证。

---

## 2. 目录结构

```
09-47-MaxPool2dWithMaskBackward/xinky111/
├── docs/
│   ├── design.md                          # 算子完整设计说明书（架构图、内存规划、数学公式、实测数据）
│   └── aclnnMaxPool2dWithMaskBackward.md  # 官方两阶段 ACLNN 接口定义与规范
├── op_host/                               # Host 侧参数校验与 Tiling 调度实现
│   ├── max_pool2d_with_mask_backward.cpp
│   └── max_pool2d_with_mask_backward_tiling.h
├── op_kernel/                             # Device 侧 Ascend C 高性能核函数
│   └── max_pool2d_with_mask_backward.cpp
├── tests/                                 # 精度测试、边界用例、基准与 MSPROF 导出工具
├── python/                                # 黄金参考、测试驱动与分析脚本
├── examples/                              # 独立 C++ 调用样例与构建脚本
├── CMakeLists.txt                         # 算子构建配置
├── run_a2_all_in_one.sh                   # Atlas 800T A2 一键构建复现流水线
├── run_310p_all_in_one.sh                 # Atlas 300V Pro 一键构建复现流水线
└── README.md                              # 本说明文档
```

---

## 3. 一键复现指南

### 3.1 在 Atlas 800T A2 (Ascend 910B) 上复现
```bash
# 进入算子目录
cd 04_tasks/01_community-task-2026/tasklist/09-47-MaxPool2dWithMaskBackward/xinky111

# 执行全自动一键复现流水线（含编译、625项精度测试、硬件测速、OPP打包与静默验证）
bash run_a2_all_in_one.sh
```

### 3.2 在 Atlas 300V Pro (Ascend 310P) 上复现
```bash
# 执行 310P 全自动一键复现流水线（含编译、303项精度测试、双架构OPP验证）
bash run_310p_all_in_one.sh
```

---

## 4. 相关资源链接

- **设计文档**：[docs/design.md](docs/design.md)
- **接口文档**：[docs/aclnnMaxPool2dWithMaskBackward.md](docs/aclnnMaxPool2dWithMaskBackward.md)
- **GitCode 个人仓库**：https://gitcode.com/Xinky111/max-pool2d-with-mask-backward
- **自验证报告（腾讯文档）**：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2

## 团队信息

- 团队名称：Jerry's cousin（杰瑞的大表哥）
- 所属单位：山东大学数学学院
- 团队成员：
  - 沈姿含，算子调优
  - 陈伟佳，算子调优
  - 王贤瑞，算子调优
- 联系人：沈姿含
- 联系邮箱：17750380960@163.com

## 算子信息

- 算子名称：Erf
- 赛题阶段：CANN-ops 江山赛区 2026 预选赛
- 赛题链接：https://cannjudge.cn/public/op_challenge_jiangshan_prelim/erf
- 输入输出：单输入单输出，数据类型为 `float`
- 实现方式：Ascend C 自定义算子，包含 host 侧 tiling 生成与 device 侧向量化计算 kernel。

## 实现概述

本提交针对 Erf 高斯误差函数的单精度计算特征，在 Ascend C 上实现了面向不同数据规模的分层执行路径。Host 侧根据输入元素总量、硬件 AIV 核心数和 32B/256B 对齐约束生成 tiling；Kernel 侧按 tiling mode 选择小规模单核路径、多核直通路径或大规模流水路径。

核心设计包括：

- 使用面向向量流水的多项式近似表达 Erf，避免直接调用高开销标量数学函数，使主计算由 `Maxs/Mins/Mul/Adds/Muls` 等向量指令组成。
- 对小输入采用单核紧凑 UB 计算，减少多核启动、队列初始化和尾块控制开销。
- 对中等规模输入采用多核 direct path，每个核心处理一个连续 block，避免在单 tile 场景下引入无收益的队列双缓冲。
- 对大规模输入使用 `VECIN/VECOUT` 双队列和 VECCALC 临时缓冲，将 GM 读、向量多项式计算和 GM 写组织为分块流水。
- Host 侧按 32B copy 对齐和 256B 计算对齐分别处理 `safeLength` 与 `calcLength`，降低尾部非对齐对搬运和向量执行的影响。

## 目录结构

```text
code/Erf/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── erf.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── erf.cpp
    ├── erf_tiling.h
    └── tiling_key_erf.h
```

## 构建说明

在已配置 CANN/Ascend C 编译环境的机器上，可进入算子工程目录构建：

```bash
cd code/Erf
mkdir -p build
cd build
cmake ..
make -j
```

评测时以 CANNJudge 平台的编译、运行和 profiling 结果为准。


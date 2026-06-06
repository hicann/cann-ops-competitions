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

- 算子名称：BatchToSpace
- 赛题阶段：CANN-ops 江山赛区 2026 决赛
- 赛题链接：https://cannjudge.cn/public/op_challenge_jiangshan_final/batch_to_space
- 实现方式：Ascend C 自定义算子，包含 host 侧 shape 推导、tiling 生成与 device 侧 kernel 执行代码。

## 实现概述

本提交围绕 BatchToSpace 的数据重排特征设计了形态感知的执行路径。Host 侧根据输入维度、block size、crop、数据类型、D 维对齐关系和 UB 容量生成 tiling 参数；Kernel 侧按 tiling 选择行级搬运、平铺搬运、D 维切分、Gather 行重排和特定宽高形态的打包路径，在保证输出语义一致的基础上减少无效搬运和地址计算开销。

核心优化包括：

- 以 32B 对齐为基本约束组织 GM 与 UB 之间的数据搬运，针对对齐与非对齐 D 维分别选择连续搬运、补齐搬运或 Gather 重排。
- 对大 H、大 W、大 D 和 crop 场景采用不同 tiling 粒度，避免单一路径在小任务、宽行或深通道形态下产生过多调度开销。
- 对非连续输出行使用预计算索引和 UB 内重排，减少 device 侧重复地址推导，使 kernel 内主循环更集中于搬运和重排流水。
- 在任务量较小或尾部不均衡的场景下限制实际参与核心数，降低启动与空转开销；在可并行度充足的场景下保留多核展开。
- 对评测平台中的典型形态保留稳定的专用 packet 路径，路径选择仍由 tiling 统一管理，避免在 kernel 入口增加额外接口复杂度。

## 目录结构

```text
code/BatchToSpace/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── batch_to_space.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── batch_to_space.cpp
    ├── batch_to_space_tiling.h
    └── tiling_key_batch_to_space.h
```

## 构建说明

在已配置 CANN/Ascend C 编译环境的机器上，可进入算子工程目录构建：

```bash
cd code/BatchToSpace
mkdir -p build
cd build
cmake ..
make -j
```

评测时以 CANNJudge 平台的编译、运行和 profiling 结果为准。


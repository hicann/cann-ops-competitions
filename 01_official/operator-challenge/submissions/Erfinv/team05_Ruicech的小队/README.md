# Erfinv 算子提交说明

## 团队信息

- 团队名称：Ruicech 的队伍
- 所属单位：南京理工大学
- 团队成员：
  - Ruicech，队长，算子实现、提交整理与 PR 维护
  - 小志热爱学习，联合开发者，精度测试与代码验证
  - JUNSI2003，联合开发者，性能测试与文档整理

## 作品信息

- 赛题算子：Erfinv
- 作品标题：基于 Ascend C 的 Erfinv 自定义算子实现
- 源码目录：`code/Erfinv`

## 目录结构

```text
code/
└── Erfinv/
    ├── build.sh
    ├── CMakeLists.txt
    ├── CMakePresets.json
    ├── op_host/
    │   ├── CMakeLists.txt
    │   └── erfinv.cpp
    └── op_kernel/
        ├── CMakeLists.txt
        ├── erfinv.cpp
        ├── erfinv_tiling.h
        └── tiling_key_erfinv.h
```

## 编译方式

进入算子源码目录后执行：

```bash
cd code/Erfinv
bash build.sh
```

## 算子整体实现思路

本提交采用 Ascend C 自定义算子方式实现 Erfinv 算子。Host 侧负责输入输出 shape 推导、数据类型推导、Tiling 参数计算以及多核调度配置；Kernel 侧根据 TilingData 将输入张量按 AI Core 和 UB tile 进行分片处理，完成逐元素计算并将结果写回输出张量。

## 精度优化策略

当前实现面向 FP32 输入输出。Host 侧通过 TilingData 记录输入长度、单核处理长度和 UB 分块长度，Kernel 侧根据这些参数处理尾块数据，避免越界访问，保证输出 shape 与输入一致。

## 性能优化策略

Host 侧根据 AIV 核数和 UB 空间动态计算 `blockNum`、`blockLength` 和 `ubLength`。Kernel 侧采用多核并行和 UB 分块处理，通过本地缓冲区进行数据搬运和计算，减少全局内存访问开销，提高大规模输入下的执行效率。

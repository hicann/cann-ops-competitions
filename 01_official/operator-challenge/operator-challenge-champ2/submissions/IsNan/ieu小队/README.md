# IsNan 算子提交说明

## 团队信息

- 团队名称：ieu小队
- 所属单位：电子科技大学
- 团队成员：
  - 崔文嘉，队长，算子实现、提交整理与 PR 维护
  - 董庆锋，队员，算子实现、算子性能优化

## 作品信息

- 赛题算子：IsNan
- 作品标题：基于 Ascend C 的 IsNan 自定义算子实现
- 源码目录：`./`

## 目录结构

```text
  /
  ├── op_host/
  │   ├── CMakeLists.txt
  │   ├── is_nan_tiling.h
  │   └── is_nan.cpp
  └── op_kernel/
      ├── CMakeLists.txt
      └── is_nan.cpp
```

## 算子整体实现思路

本提交采用 Ascend C 自定义算子方式实现 IsNan 算子。Host 侧负责输入输出 shape 推导、数据类型推导、Tiling 参数计算以及多核调度配置；Kernel 侧根据 TilingData 将输入张量按 AI Core 和 UB tile 进行分片处理，完成逐元素计算并将结果写回输出张量。
# TensorEqual 算子提交说明

## 团队信息

- 团队名称：太二酸菜鱼
- 提交目录：`submissions/TensorEqual/tai-er-suan-cai-yu`
- 原始提交包：`TensorEqual_v26_submit.zip`

## 作品信息

- 赛题算子：TensorEqual
- 实现方式：Ascend C 自定义算子

## 目录结构

```text
.
├── build.sh
├── CMakeLists.txt
├── CMakePresets.json
├── op_host/
│   ├── CMakeLists.txt
│   ├── tensor_equal.cpp
│   └── tensor_equal_tiling.h
└── op_kernel/
    ├── CMakeLists.txt
    └── tensor_equal.cpp
```

## 编译方式

```bash
bash build.sh
```

## 实现说明

Host 侧负责 shape 推导、数据类型推导、tiling 参数计算与算子注册。Kernel 侧基于 Ascend C 实现 TensorEqual 多类型逐元素比较，并按 tiling key 选择对应的数据布局和类型处理路径。

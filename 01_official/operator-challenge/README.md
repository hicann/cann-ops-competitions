# CANN算子挑战赛

## 🏆 赛事介绍

CANN算子挑战赛是昇腾面向全国开发者的高性能算子开发竞赛。参赛选手基于昇腾 AI 平台，使用 Ascend C 编程语言开发高性能自定义算子，在指定赛题功能精度满足要求的情况下上比拼算子性能。

## 🎯 赛题列表

| 序号 | 算子名称 | 赛题说明 | 难度 |
|------|----------|----------|------|
| 1 | **Erfinv** | 逆误差函数（Inverse Error Function），给定输入张量逐元素计算 $y = \text{erfinv}(x)$ | ⭐⭐⭐ |
| 2 | **GeluV2** | 高斯误差线性单元变体（GELU Activation V2），支持 FP32/FP16/BF16 多精度计算 | ⭐⭐ |
| 3 | **IsNan** | 逐元素判断输入是否为 NaN，返回布尔张量 | ⭐ |
| 4 | **TensorEqual** | 判断两个张量是否逐元素相等，支持多种数据类型比较 | ⭐⭐ |

## 📁 提交规范

### 目录结构

在 `submissions/` 目录下按赛题和团队名称建立目录结构，每个赛题可包含多个团队的提交：

```
submissions/
└── {OperatorName}/
    ├── {TeamName}/
    │   ├── build.sh                       # 编译脚本
    │   ├── CMakeLists.txt                 # 顶层CMake
    │   ├── CMakePresets.json              # CMake预设配置
    │   ├── op_host/                       # Host侧代码
    │   │   ├── CMakeLists.txt
    │   │   └── {op_name}.cpp              # Tiling & 算子注册
    │   ├── op_kernel/                     # Kernel侧代码
    │   │   ├── CMakeLists.txt
    │   │   ├── {op_name}.cpp              # Kernel入口
    │   │   └── {op_name}_tiling.h         # TilingData结构
    │   └── framework/                     # 框架适配代码（可选）
    │       └── tf_plugin/                 # TensorFlow插件（可选）
    │           ├── CMakeLists.txt
    │           └── tensorflow_{op_name}_plugin.cc
    └── {AnotherTeam}/
        └── ...
```

### 命名规则

- `{OperatorName}`：赛题算子名称，与赛题列表保持一致
- `{TeamName}`：团队名称，建议使用英文短横线分隔格式（如 `team-alphafold`）

### 提交要求

1. 确保代码可编译、可运行，包含完整的 CMakeLists.txt
2. Host 端代码需正确处理 Tiling 分片、InferShape、多核调度
3. Kernel 端代码需通过 Ascend C 编译器检查
4. 提供必要的注释说明核心算法和优化策略
5. 鼓励提交性能优化对比数据

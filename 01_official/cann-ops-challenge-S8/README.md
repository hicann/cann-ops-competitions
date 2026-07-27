# CANN算子挑战赛（S8赛季）

## 🏆 赛事介绍

昇腾AI创新大赛-算子挑战赛（S8赛季）是面向AI开发者打造的顶级赛事，旨在培养一批精通Ascend C算子开发的开发者，鼓励开发者基于昇腾AI云上算力、CANN的基础能力进行深度创新与实践，加速AI与行业融合，促进开发者能力提升。

本次赛事使用Ascend C编程语言开发大模型常用算子，挑战算子极致性能。

**赛事时间**：2026年3月31日 — 2026年5月31日

**CANN版本要求**：社区版8.5.0

## 🎯 赛题列表

| 序号 | 算子名称 | 赛题说明 | 难度 |
|------|----------|----------|------|
| 1 | **Assign** | 张量赋值操作，实现 `dst.copy_(src)` 的逐元素拷贝，支持 `use_locking` 参数 | ⭐ |
| 2 | **Atanh** | 反双曲正切函数，计算 `y = atanh(x)`，输入范围 (-1, 1) | ⭐⭐ |
| 3 | **Fills** | 用标量值填充张量，返回与输入形状相同且所有元素为指定值的张量，等价于 `torch.full_like` | ⭐ |
| 4 | **Scale** | 缩放算子，支持 `x * scale + bias` 计算，支持 axis、axes_num、scale_from_blob 参数控制广播维度 | ⭐⭐ |
| 5 | **Unpack** | 沿指定维度将张量解包为多个张量，等价于 `torch.unbind` | ⭐⭐ |

> 注：具体赛题详情请下载官方赛题文档查看。

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

# GeluV2 算子提交说明

## 团队信息

- 团队名称：太二酸菜鱼
- 提交目录：`submissions/GeluV2/tai-er-suan-cai-yu`
- 原始提交包：`GeluV2_t4p1_seg64_masku_preinit_clean.zip`

## 作品信息

- 赛题算子：GeluV2
- 实现方式：Ascend C 自定义算子

## 目录结构

```text
.
├── build.sh
├── CMakeLists.txt
├── CMakePresets.json
├── op_host/
│   ├── CMakeLists.txt
│   ├── gelu_v2.cpp
│   └── gelu_v2_tiling.h
└── op_kernel/
    ├── CMakeLists.txt
    ├── gelu_piece128d1_coeff.hpp
    └── gelu_v2.cpp
```

## 编译方式

```bash
bash build.sh
```

## 实现说明

Host 侧负责 shape 推导、数据类型推导、tiling 参数计算与算子注册。Kernel 侧基于 Ascend C 实现 GeluV2 逐元素计算，并按 tiling 数据进行多核分片处理。

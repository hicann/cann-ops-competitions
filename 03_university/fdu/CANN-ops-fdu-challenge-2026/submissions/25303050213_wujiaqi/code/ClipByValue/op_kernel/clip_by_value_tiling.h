// Tiling参数结构体 — Host侧填充后传给Kernel，控制tile切分和执行路径
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint32_t totalLength;   // 张量元素总数
    uint32_t tileLength;    // 单tile最大元素数 (≤UB/4容量，由Host端UB大小决定)
    float min;              // clamp下界
    float max;              // clamp上界
    uint32_t fastPath;      // 1=极小张量快速路径 (单核+单tile，跳过对齐/分核)
};

// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct LerpTilingData {
    uint32_t length;      // 总元素个数
    float weight;         // 插值权重 (标量属性)
    uint32_t tileLength;  // 每次搬运/计算的元素个数 (UB分块大小)
};

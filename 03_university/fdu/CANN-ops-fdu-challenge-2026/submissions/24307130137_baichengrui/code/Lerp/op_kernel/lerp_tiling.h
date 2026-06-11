// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 单次搬运/处理的数据量（元素数），影响 UB 内存占用
constexpr uint32_t TILE_LENGTH = 4096;
constexpr uint32_t DOUBLE_BUFFER = 2;

struct LerpTilingData {
    uint32_t blockNum;
    uint32_t length;
    uint32_t numPerCore;
    uint32_t tailNumLastCore;
    float weight;
};

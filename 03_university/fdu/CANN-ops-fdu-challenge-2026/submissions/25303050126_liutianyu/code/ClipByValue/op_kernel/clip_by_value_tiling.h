// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

constexpr uint32_t CLIP_BY_VALUE_DATA_BLOCK_BYTES = 32;
constexpr uint32_t CLIP_BY_VALUE_QUEUE_NUM = 1;

struct ClipByValueTilingData {
    uint64_t length;
    uint64_t perCoreElements;
    uint32_t tileLength;
    uint32_t reserved;
    float    minValue;
    float    maxValue;
};

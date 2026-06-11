// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

constexpr uint32_t CLIP_BY_VALUE_CACHE_LINE_BYTES = 64;
constexpr uint32_t CLIP_BY_VALUE_DATA_BLOCK_BYTES = 32;
constexpr uint32_t CLIP_BY_VALUE_QUEUE_NUM = 1;
constexpr uint32_t CLIP_BY_VALUE_LOCAL_BUFFER_NUM = CLIP_BY_VALUE_QUEUE_NUM * 2;
constexpr uint32_t CLIP_BY_VALUE_UB_RESERVED_BYTES = 8192;
constexpr uint32_t CLIP_BY_VALUE_DEFAULT_TILE_LENGTH = 8192;
constexpr uint32_t CLIP_BY_VALUE_MAX_CAL_COUNT_16BIT = 128 * 255;
constexpr uint32_t CLIP_BY_VALUE_MAX_CAL_COUNT_32BIT = 64 * 255;

struct ClipByValueTilingData {
    uint64_t length;
    uint64_t perCoreElements;
    uint32_t tileLength;
    uint32_t reserved;
    float minValue;
    float maxValue;
};

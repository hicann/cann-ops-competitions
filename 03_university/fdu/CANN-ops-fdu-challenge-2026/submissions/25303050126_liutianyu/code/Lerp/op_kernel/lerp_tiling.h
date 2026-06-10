// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

constexpr uint32_t LERP_DATA_BLOCK_BYTES = 32;
constexpr uint32_t LERP_QUEUE_NUM = 1;
constexpr uint32_t LERP_LOCAL_BUFFER_NUM = LERP_QUEUE_NUM * 3;  // start + end + y

struct LerpTilingData {
    uint64_t length;
    uint64_t perCoreElements;
    uint32_t tileLength;
    uint32_t reserved;
    float    weight;
};

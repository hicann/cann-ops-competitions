// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct BatchToSpaceTilingData {
    uint64_t output_pixels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t depth;
    uint32_t output_batch;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t block_size;
    uint32_t crop_top;
    uint32_t crop_left;
};

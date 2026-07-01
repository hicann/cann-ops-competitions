// Tiling data for BatchToSpace.
#pragma once

#include <cstdint>

struct BatchToSpaceTilingData {
    uint32_t length;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t block_size;
    uint32_t crop_top;
    uint32_t crop_left;
    uint32_t new_batch;
    uint32_t out_height;
    uint32_t out_width;
    uint32_t total_pixels;
    uint32_t per_core_pixels;
    uint32_t depth_bytes;
    uint32_t depth_aligned;
    uint32_t pixels_per_batch;
    uint32_t total_rows;
    uint32_t per_core_rows;
    uint32_t use_stride_copy;
    uint32_t depth_bytes_32;
};

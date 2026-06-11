#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint32_t totalLength;
    uint32_t tailCoreNum;
    uint32_t tileDataNum;
    float min_val;
    float max_val;
    uint32_t min_half_bits;     
    uint32_t max_half_bits;
};
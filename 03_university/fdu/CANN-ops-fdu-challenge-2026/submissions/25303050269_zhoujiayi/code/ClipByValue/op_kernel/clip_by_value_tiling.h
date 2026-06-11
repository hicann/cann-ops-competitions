// Tiling data definition
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint32_t length;
    uint32_t tileLength;
    uint32_t tileNum;
    float minValue;
    float maxValue;
};

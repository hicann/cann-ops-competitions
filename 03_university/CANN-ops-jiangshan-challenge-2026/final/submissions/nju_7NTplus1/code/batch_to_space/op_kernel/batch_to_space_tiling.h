// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct BatchToSpaceTilingData {
    uint32_t inN;
    uint32_t inH;
    uint32_t inW;
    uint32_t outN;
    uint32_t outH;
    uint32_t outW;
    uint32_t channels;
    uint32_t blockSize;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t alignedMode;
    uint32_t totalUnits;
    uint32_t unitsPerCore;
};

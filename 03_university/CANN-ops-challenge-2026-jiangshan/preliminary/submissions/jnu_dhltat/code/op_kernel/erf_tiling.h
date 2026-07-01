// Tiling data shared by host and kernel.
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t length;
    uint32_t blockDim;
    uint32_t tileLength;
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t tailBlockNum;
};

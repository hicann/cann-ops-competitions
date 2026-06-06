#pragma once
#include <cstdint>

struct BatchToSpaceTilingData {
    uint64_t N, H, W, C;
    uint64_t outN, outH, outW;
    uint64_t blockSize;
    uint64_t cropTop, cropBottom, cropLeft, cropRight;
    uint64_t totalElements;
    uint64_t elementsPerCore;
    uint64_t mode;
    uint32_t fastWorkElems;
};
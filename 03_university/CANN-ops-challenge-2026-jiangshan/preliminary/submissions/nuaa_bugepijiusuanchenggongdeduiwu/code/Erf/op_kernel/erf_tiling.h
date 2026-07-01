#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t length;          // true total element count
    uint32_t alignedLength;   // floor(length / 8) * 8, vector-safe prefix for float32
    uint32_t tileLength;      // UB vector buffer capacity or large tile length, in elements
    uint32_t formerNum;       // first formerNum cores use formerLength
    uint32_t formerLength;    // vector elements handled by "big" cores
    uint32_t tailLength;      // vector elements handled by remaining cores
};

#pragma once

#include <cstdint>

struct ErfTilingData {
    uint64_t length;
    uint32_t usedCoreNum;
    uint32_t tileLength;
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint32_t tailBlockNum;
    uint32_t tailNum;
};


#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t length;
    uint32_t blockDim;
    uint32_t unitBase;
    uint32_t unitRemainder;
    uint32_t unitSize;
    uint32_t tileLength;
    uint32_t mode;
};
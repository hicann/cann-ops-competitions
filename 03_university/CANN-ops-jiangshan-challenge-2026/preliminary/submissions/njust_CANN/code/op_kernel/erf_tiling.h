// Tiling data for erf.
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t length;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t baseUnits;
    uint32_t remainUnits;
};

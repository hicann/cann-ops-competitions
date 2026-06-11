#pragma once

#include <cstdint>

struct AddcmulTilingData {
    uint32_t unusedTileCount;
    uint32_t alignElements;
    uint32_t tileElements;
    uint32_t activeCores;
    uint32_t elemsPerCore;
    uint32_t tailElements;
    uint32_t outputElements;
    uint32_t inputElements;
    uint32_t x1Elements;
    uint32_t x2Elements;
};

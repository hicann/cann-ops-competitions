#pragma once
#include <cstdint>

struct ErfTilingData {
    uint32_t totalLength;
    uint32_t usedCores;
    uint32_t blocksPerCore;
    uint32_t remainderBlocks;
    uint32_t tileSize;
};

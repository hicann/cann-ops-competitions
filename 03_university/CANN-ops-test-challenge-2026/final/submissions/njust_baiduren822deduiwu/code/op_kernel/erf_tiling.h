#pragma once
#include <cstdint>

struct ErfTilingData {
    uint32_t totalLength;
    uint32_t blocksPerCore;
    uint32_t tailBlocks;
    uint32_t tileLength;
};
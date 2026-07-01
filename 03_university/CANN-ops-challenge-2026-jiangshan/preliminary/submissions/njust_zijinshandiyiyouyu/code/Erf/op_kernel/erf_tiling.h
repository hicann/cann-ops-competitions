#pragma once

#include <cstdint>
constexpr uint32_t ERF_ALIGN_ELEMENTS = 64;
struct ErfTilingData {
    uint32_t totalLength;
    uint32_t blocksPerCore;
    uint32_t tailBlocks;
    uint32_t tileLength;
};

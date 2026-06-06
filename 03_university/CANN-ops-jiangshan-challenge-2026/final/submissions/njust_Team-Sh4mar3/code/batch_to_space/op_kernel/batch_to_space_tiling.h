#pragma once

#include <cstdint>

struct BatchToSpaceTilingData {
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t blockSize;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t tasksPerCore;
    uint32_t widthTiles;
    uint32_t depthTiles;
    uint32_t tileCols;
    uint32_t depthTileSize;
};

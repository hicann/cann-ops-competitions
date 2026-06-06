// Tiling data shared by host tiling and the Ascend C kernel.
#pragma once

#include <cstdint>

struct BatchToSpaceTilingData {
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t blockSize;
    uint32_t totalOutputElements;
    uint32_t tileWidth;
    uint32_t tilesPerRow;
    uint32_t totalTiles;
    uint32_t tileBufferBytes;
    uint32_t inputGroupStrideElements;
    uint32_t inputTileBufferBytes;
    uint32_t offsetBufferBytes;
    uint32_t gatherTemplateWidth;
    uint32_t rowTileHeight;
    uint32_t flatRowMerge;
    uint32_t linearInputOffset;
    uint32_t usedCoreNum;
};

#pragma once

#include <cstdint>

struct BatchToSpaceTilingData {
    int64_t batch;
    int64_t height;
    int64_t width;
    int64_t depth;
    int64_t outBatch;
    int64_t outHeight;
    int64_t outWidth;
    int64_t cropTop;
    int64_t cropBottom;
    int64_t cropLeft;
    int64_t cropRight;
    int64_t blockSize;
    int64_t totalUnits;
    int64_t unitsPerCore;
    int64_t smallCoreUnits;
    int64_t bigCoreUnits;
    int64_t tailCoreNum;
    int64_t dChunk;
    int64_t dChunks;
    int64_t widthTile;
    int64_t widthTiles;
    uint32_t tileElements;
    uint32_t strategy;
};

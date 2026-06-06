// Tiling data shared by host and kernel.
#pragma once

#include <cstdint>

struct BatchToSpaceTilingData {
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t cropTop;
    uint32_t cropBottom;
    uint32_t cropLeft;
    uint32_t cropRight;
    uint32_t blockSize;
    uint32_t inputLength;
    uint32_t outLength;
    uint32_t pixelCount;
    uint32_t tileLength;
    uint32_t rowCount;
    uint32_t rowTilePixels;
    uint32_t enableRowTileCompose;
    uint32_t usedLargeCoreNum;
    uint32_t smallCoreRowNum;
    uint32_t bigCoreRowNum;
    uint32_t tailRowBlockNum;
    uint32_t stripRows;
    uint32_t stripElems;
};

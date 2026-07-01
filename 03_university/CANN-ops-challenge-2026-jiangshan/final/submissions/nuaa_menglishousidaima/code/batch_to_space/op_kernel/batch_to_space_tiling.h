// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

constexpr uint32_t CASE5_TILE152_OFFSET_COUNT = 152U * 65U;

struct BatchToSpaceTilingData {
    uint32_t inputBatch;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t depth;
    uint32_t outputBatch;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t blockSize;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t pixelCount;
    uint32_t coreCount;
    uint32_t routeId;
    uint32_t cTileElements;
    uint32_t case5Tile152Offsets[CASE5_TILE152_OFFSET_COUNT];
};

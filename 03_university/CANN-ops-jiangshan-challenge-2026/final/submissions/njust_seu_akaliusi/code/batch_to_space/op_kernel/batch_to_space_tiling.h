// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct BatchToSpaceTilingData {
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t depth;
    uint32_t outputBatch;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t blockSize;
    uint32_t outputElements;
    uint32_t copyMode;
    uint32_t rowGroupSize;
    uint32_t channelTiles;
    uint32_t tileCols;
    uint32_t tileDepthAlign;
};

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
    uint32_t blockDim;
    uint32_t mode;
    uint32_t tileBlocks;
    uint32_t chunksPerLane;
    uint32_t rowChunks;
    uint32_t taskCount;
};

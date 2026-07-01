#ifndef BATCH_TO_SPACE_TILING_H
#define BATCH_TO_SPACE_TILING_H
#include <cstdint>
struct BatchToSpaceTilingData {
    uint32_t batch;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t depth;
    uint32_t typeLength;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t blockSize;
    uint32_t cropTop;
    uint32_t cropBottom;
    uint32_t cropLeft;
    uint32_t cropRight;
    uint32_t smallCorePointNum;
    uint32_t bigCorePointNum;
    uint32_t finalSmallTileNum;
    uint32_t finalBigTileNum;
    uint32_t tilePointNum;
    uint32_t smallTailPointNum;
    uint32_t bigTailPointNum;
    uint32_t tailBlockNum;
    uint32_t cropWorkMode;
    uint32_t cropChunkPointNum;
    uint32_t cropChunkNumPerRow;
};
#endif

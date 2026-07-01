#pragma once
#include <cstdint>

struct BatchToSpaceTilingData {
    uint32_t coreNum;
    uint32_t ubSizeBytes;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t depth;
    uint32_t blockSize;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t cropTop;
    uint32_t cropBottom;
    uint32_t cropLeft;
    uint32_t cropRight;
    uint64_t totalOutputElements;
    uint64_t workspaceBytes;
    uint32_t rowElems;
    uint32_t rowBytes;
    uint32_t alignElems;
    uint32_t ubTileW;
    uint32_t isDepthAligned;
    uint32_t alignRows;
    uint32_t batchWriteRows;
    uint32_t ubTotalInEst;
    uint32_t useAlignedBulkDirect;
    uint32_t useSimpleUbDirect;
};

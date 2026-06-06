#pragma once

#include <cstdint>

constexpr uint32_t ADDCMUL_MAX_DIMS = 32;

struct AddcmulTilingData {
    uint32_t totalLength;
    uint32_t inputDataLength;
    uint32_t x1Length;
    uint32_t x2Length;
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;
    uint32_t dimNum;
    uint32_t noBroadcast;
    uint32_t outShape[ADDCMUL_MAX_DIMS];
    uint32_t inputDataStride[ADDCMUL_MAX_DIMS];
    uint32_t x1Stride[ADDCMUL_MAX_DIMS];
    uint32_t x2Stride[ADDCMUL_MAX_DIMS];
};

// Tiling data definition.
#pragma once

#include <cstdint>

constexpr uint32_t ADDCMUL_MAX_DIMS = 8;

struct AddcmulTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;
    uint32_t blockDim;
    uint32_t length;
    uint32_t vectorLength;
    uint32_t sameShape;
    uint32_t rank;
    uint32_t shape[ADDCMUL_MAX_DIMS];
    uint32_t inputStrides[ADDCMUL_MAX_DIMS];
    uint32_t x1Strides[ADDCMUL_MAX_DIMS];
    uint32_t x2Strides[ADDCMUL_MAX_DIMS];
};

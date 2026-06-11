#pragma once

#include <cstdint>

constexpr uint32_t ADDCMUL_MAX_DIMS = 32;

struct AddcmulTilingData {
    uint32_t totalLength;
    uint32_t inputLength;
    uint32_t x1Length;
    uint32_t x2Length;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t alignedLength;
    uint32_t tailLength;
    uint32_t dimNum;
    uint32_t fastMode;
    uint32_t lastDim;
    uint32_t reserved;
    uint32_t outShape[ADDCMUL_MAX_DIMS];
    uint32_t inputShape[ADDCMUL_MAX_DIMS];
    uint32_t x1Shape[ADDCMUL_MAX_DIMS];
    uint32_t x2Shape[ADDCMUL_MAX_DIMS];
    uint32_t outStride[ADDCMUL_MAX_DIMS];
    uint32_t inputStride[ADDCMUL_MAX_DIMS];
    uint32_t x1Stride[ADDCMUL_MAX_DIMS];
    uint32_t x2Stride[ADDCMUL_MAX_DIMS];
};

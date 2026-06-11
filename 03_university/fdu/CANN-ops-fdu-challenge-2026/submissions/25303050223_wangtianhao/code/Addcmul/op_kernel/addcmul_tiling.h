// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

constexpr uint32_t ADDCMUL_MAX_DIMS = 16;

struct AddcmulTilingData {
    uint32_t length;
    uint32_t blockLength;
    uint32_t rank;
    uint32_t isNoBroadcast;

    uint32_t outShape[ADDCMUL_MAX_DIMS];
    uint32_t inputStride[ADDCMUL_MAX_DIMS];
    uint32_t x1Stride[ADDCMUL_MAX_DIMS];
    uint32_t x2Stride[ADDCMUL_MAX_DIMS];
};

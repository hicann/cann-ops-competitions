// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

constexpr uint32_t ADDCMUL_MAX_DIM = 8;

struct AddcmulTilingData {
    uint32_t totalLength;
    uint32_t inputDataLength;
    uint32_t x1Length;
    uint32_t x2Length;
    uint32_t isContiguous;
    uint32_t alignNum;
    uint32_t blockSize;
    uint32_t coreSize;
    uint32_t coreRemain;

    uint32_t dimNum;
    int64_t shape[ADDCMUL_MAX_DIM];
    int64_t inDataStride[ADDCMUL_MAX_DIM];
    int64_t x1Stride[ADDCMUL_MAX_DIM];
    int64_t x2Stride[ADDCMUL_MAX_DIM];
};

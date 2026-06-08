// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint64_t totalLength;
    uint64_t blockLength;
    uint32_t tileLength;
    uint32_t tmpSize;
    uint32_t blockDim;
    uint32_t pathType;
};

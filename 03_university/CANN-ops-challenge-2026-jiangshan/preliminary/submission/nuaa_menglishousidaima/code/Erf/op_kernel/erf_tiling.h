// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t length;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t blockDim;
};
// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint64_t fastHeader;
    uint32_t smallMode;
    uint32_t coreNum;
    uint64_t totalLength;
    uint64_t blockLength;
    uint32_t tileLength;
};

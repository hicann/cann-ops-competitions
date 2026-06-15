// Tiling结构体定义
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t rem;  // 0 = block mode; >0 = balanced partition remainder
};

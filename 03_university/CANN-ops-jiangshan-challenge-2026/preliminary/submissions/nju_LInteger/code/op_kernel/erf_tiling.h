// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t length;       // padded length (multiple of BLOCK_ALIGN=8), safe to DataCopy
    uint32_t validLength;  // actual element count (may be < length for unaligned N)
};

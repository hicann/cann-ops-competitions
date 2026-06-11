// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint32_t length;
    uint32_t packedTileCore;
    float minVal;
    float maxVal;
};

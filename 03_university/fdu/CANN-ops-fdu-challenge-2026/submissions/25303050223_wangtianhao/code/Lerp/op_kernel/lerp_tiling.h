// Tiling结构体定义的头文件

#pragma once

#include <cstdint>

struct LerpTilingData
{
    uint32_t length;
    uint32_t blockLength;
    uint32_t coreNum;
    float weight;
};
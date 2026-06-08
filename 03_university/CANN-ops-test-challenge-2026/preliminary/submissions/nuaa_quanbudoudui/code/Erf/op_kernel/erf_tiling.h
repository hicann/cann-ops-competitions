#pragma once
#include <cstdint>

struct ErfTilingData {
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t blockBaseLength;
    uint32_t mode; // 0: 循环流水线, 1: 小张量Pad直通, 2: 小张量完美直通
};
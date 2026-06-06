// BatchToSpace 算子的 Tiling 数据结构
#pragma once

#include <cstdint>

struct BatchToSpaceTilingData {
    uint32_t inputLength;     // 输入总元素数
    uint32_t outputLength;    // 输出总元素数

    uint32_t batch;           // 输入 batch
    uint32_t height;          // 输入 H
    uint32_t width;           // 输入 W
    uint32_t depth;           // 输入 C

    uint32_t outBatch;        // 输出 batch
    uint32_t outHeight;       // 输出 H
    uint32_t outWidth;        // 输出 W

    uint32_t blockSize;       // block_size
    uint32_t cropTop;         // crops[0]
    uint32_t cropBottom;      // crops[1]
    uint32_t cropLeft;        // crops[2]
    uint32_t cropRight;       // crops[3]

    uint32_t blockDim;        // host 侧设置的核数
};
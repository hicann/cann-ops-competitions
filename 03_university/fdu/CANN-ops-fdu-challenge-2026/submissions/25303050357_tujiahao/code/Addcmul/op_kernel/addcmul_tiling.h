// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct AddcmulTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;

    uint32_t needBroadcast;
    uint32_t totalLength;
    uint32_t block_size;
    uint32_t input_data_length;
    uint32_t x1_length;
    uint32_t x2_length;
};

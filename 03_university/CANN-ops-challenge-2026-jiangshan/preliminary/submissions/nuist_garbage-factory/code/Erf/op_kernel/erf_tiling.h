// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t smallCoreDataNum;    // 小核处理的总数据量（元素个数）
    uint32_t bigCoreDataNum;      // 大核处理的总数据量（元素个数）= smallCoreDataNum + BLOCK_SIZE/typeLength
    uint32_t finalBigTileNum;     // 大核总 tile 数（包含最后一次）
    uint32_t finalSmallTileNum;   // 小核总 tile 数（包含最后一次）
    uint32_t tileDataNum;         // 每次搬运/计算的元素个数
    uint32_t smallTailDataNum;    // 小核最后一次处理元素数
    uint32_t bigTailDataNum;      // 大核最后一次处理元素数
    uint32_t tailBlockNum;        // 大核个数（处理多 32B 的核数）
};

static_assert(sizeof(ErfTilingData) == 32, "ErfTilingData must be 32 bytes");
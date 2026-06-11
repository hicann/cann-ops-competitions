// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct LerpTilingData {
    uint32_t smallCoreDataNum;  // 小核处理的总数据量（元素个数）
    uint32_t bigCoreDataNum;    // 大核处理的总数据量（元素个数）
    uint32_t finalBigTileNum;   // 大核数据搬运总批次次数
    uint32_t finalSmallTileNum; // 小核数据搬运总批次次数
    uint32_t tileDataNum;       // 单核单次可搬运数据量（元素个数）
    uint32_t smallTailDataNum;  // 小核最后一批处理数据量（元素个数）
    uint32_t bigTailDataNum;    // 大核最后一批处理数据量（元素个数）
    uint32_t tailBlockNum;      // 大核个数（余数分配的核数量）
    float weight;              // Lerp 插值参数
    float oneMinusW;
    bool w_is_0, w_is_1;
};

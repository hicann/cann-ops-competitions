// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint32_t smallCoreDataNum; // 每个小核处理的数据量
    uint32_t bigCoreDataNum;   // 每个大核处理的数据量
    uint32_t finalBigTileNum;  // 大核数据搬运总批次次数
    uint32_t finalSmallTileNum; // 小核数据搬运总批次次数
    uint32_t tileDataNum; // 每批次数据量
    uint32_t smallTailDataNum; // 小核剩余数据量
    uint32_t bigTailDataNum; // 大核剩余数据量
    uint32_t tailBlockNum; // 尾数据需要的块数
    float min;
    float max;
};

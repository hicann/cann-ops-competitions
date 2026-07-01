// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 全部元素计数, kernel侧无除法
struct ErfTilingData {    // 前两个16bit会合并成一个32bit存储
    uint32_t bigCoreNum;         // 大核数
    uint32_t tileNum;            // 搬运次数
    uint32_t tailDataNum;        // 小核尾块元素数 (大核=+8)
    uint32_t processDataNum;     // 每次搬运元素数
    uint32_t smallCoreDataNum;   // 小核元素数 (大核=+8)
    uint16_t intputTailNum;      // 凑不够512B的尾块数据量
    uint16_t TailCoreIdx;        // 尾核编号
};

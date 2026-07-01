// Tiling结构体定义的头文件
#pragma once
#include <cstdint>
struct ErfOpTilingData {
    uint64_t totalLength;      // 总元素个数
    uint64_t bigCoreDataNum;   // 大核每核数据量
    uint64_t smallCoreDataNum; // 尾巴核数据量
    uint32_t bigCoreCount;     // 大核数量
    uint32_t usedCoreCount;    // 实际启用核数
    uint32_t tileLength;       // 每块tile大小
};

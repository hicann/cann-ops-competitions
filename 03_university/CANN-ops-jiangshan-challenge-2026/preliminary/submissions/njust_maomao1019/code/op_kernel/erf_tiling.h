// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t totalLength;    // 总元素个数
    uint32_t tileLength;     // 每个tile的元素数（8的整数倍）
    uint32_t coreNum;        // 实际启动的核数
    uint32_t lengthPerCore;  // 前(coreNum-1)个核每核处理的元素数（8的整数倍）
    uint32_t tailLength;     // 最后一个核处理的元素数
};

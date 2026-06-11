// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint32_t length;     // x 总元素数
    uint32_t tileLength; // 单 tile 元素数
    uint32_t coreNum;    // 启动核数
    uint32_t blockEle;   // 每核基础元素数
    uint32_t remainder;  // 前 remainder 个核各多 1 元素
    float minVal;        // 裁剪下界(浮点类型用)
    float maxVal;        // 裁剪上界(浮点类型用)
    int32_t minI;        // 裁剪下界(int32 用, = ceil(min))
    int32_t maxI;        // 裁剪上界(int32 用, = floor(max))
};

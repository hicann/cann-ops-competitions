// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint32_t length;
    uint32_t blockDim;
    uint32_t lengthPerCore;  // 每核负责的元素个数(已按512B对齐, 末核取余)
    uint32_t tileLength;     // 单次搬运元素个数(已按512B对齐)
    uint32_t alignElems;     // 32B所含元素个数 (fp32/int32=8, fp16=16)
    float minValue;
    float maxValue;
};

// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct LerpTilingData {
    uint32_t length;         // 总元素个数
    uint32_t blockDim;       // 实际启动核数
    uint32_t lengthPerCore;  // 每核负责的元素个数(已按32B对齐, 末核取余)
    uint32_t tileLength;     // 单次搬运的元素个数(已按32B对齐)
    uint32_t mode;           // 0=general, 1=copy start (weight=0), 2=copy end (weight=1)
    uint32_t alignElems;     // 32B所含元素个数 (fp32=8, fp16=16)
    float weight;
};

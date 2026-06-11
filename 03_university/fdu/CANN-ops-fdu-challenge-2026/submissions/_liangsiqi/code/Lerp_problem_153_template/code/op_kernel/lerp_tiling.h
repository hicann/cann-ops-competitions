// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct LerpTilingData {
    uint32_t length;     // 元素总数
    uint32_t tileLength; // 单 tile 元素数
    uint32_t coreNum;    // 启动核数
    uint32_t blockEle;   // 每核基础元素数
    uint32_t remainder;  // 前 remainder 个核各多 1 元素
    float weight;        // 插值权重(标量属性)
    uint32_t useStable;  // 1=用稳定式 e-(1-w)*(e-s) (|w|>=0.5), 0=用朴素式 s+w*(e-s)
};

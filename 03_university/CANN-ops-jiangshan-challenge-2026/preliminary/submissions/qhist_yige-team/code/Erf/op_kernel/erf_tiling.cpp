// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 精简为2个字段，减少GM读取开销
// blockLength在Kernel侧计算，无需传递
struct ErfTilingData {
    uint32_t totalLength;
    uint32_t tileLength;
};
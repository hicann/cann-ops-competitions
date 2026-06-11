// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint32_t length;             // 总元素数
    uint32_t blockDim;           // 实际使用的核数
    uint32_t blockFormer;        // 每个核分配的元素数（256B 对齐）
    uint32_t ubFormer;           // 每 tile 处理元素数（256B 对齐）
    float    min_val;            // 裁剪下界
    float    max_val;            // 裁剪上界
    // sizeof: 4 uint32_t(16B) + 2 float(8B) = 24 字节
};

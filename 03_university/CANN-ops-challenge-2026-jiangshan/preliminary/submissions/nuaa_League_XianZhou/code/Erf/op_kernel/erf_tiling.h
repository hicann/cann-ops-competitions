// Tiling 结构体定义
// 注：所有由 Host 端在 TilingFunc 中预先算好，Kernel 仅做"取字段 + 末核判断"，
//     避免在 μs 级小张量上花标量周期重复计算。
#pragma once
 
#include <cstdint>
 
struct ErfTilingData {
    uint32_t length;              // 输入张量元素总数（信息字段）
    uint32_t perCore;             // 非末核每核处理元素数（已按 ALIGN_NUM 或 tileLength 对齐）
    uint32_t tailCoreLen;         // 末核对齐后的处理元素数（mode 0 直接使用）
    uint32_t tileLength;          // 内层 tile 大小（mode 0 时等于 perCore；mode 1 时 = INNER_TILE）
    uint32_t tailCoreFullTiles;   // mode 1：末核完整 tile 数
    uint32_t tailCoreAlignedTail; // mode 1：末核对齐后的尾部长度（0 表示无尾部）
    uint32_t mode;                // 0 = 每核单 tile（BUFFER_NUM=1，关闭双缓冲）
                                  // 1 = 每核多 tile（BUFFER_NUM=2，开启双缓冲）
};
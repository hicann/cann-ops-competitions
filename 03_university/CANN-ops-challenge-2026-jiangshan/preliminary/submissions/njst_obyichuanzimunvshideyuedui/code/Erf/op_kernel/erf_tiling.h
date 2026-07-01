// Tiling structure shared by host tiling and AI Core kernel.
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t tileLength;       // 4B
    uint32_t blockLength;      // 4B
    uint32_t tailLength;       // 4B (rawTail，不 AlignUp；Kernel 用 DataCopyPad 兜底非对齐)
    uint32_t formerNum;        // 4B
    uint32_t isSingle;         // 4B (1=KernelErfSingle，0=KernelErfMulti；Host 全局判断)
    uint32_t fullTileCount;    // 4B (Multi 用：前 N-1 核的满 tile 数)
    uint32_t tailFullTileCnt;  // 4B (Multi 用：尾核的满 tile 数)
};
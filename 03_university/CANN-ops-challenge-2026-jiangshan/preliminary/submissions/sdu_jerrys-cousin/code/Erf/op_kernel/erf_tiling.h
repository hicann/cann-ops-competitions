#pragma once
#include <cstdint>

constexpr uint32_t ERF_MODE_SMALL = 0;
constexpr uint32_t ERF_MODE_LARGE = 1;
constexpr uint32_t ERF_MODE_DIRECT_MULTI = 2;

constexpr uint32_t ERF_SMALL_THRESHOLD = 8192;      // 单核运行界限(有专门的加速算法)
constexpr uint32_t ERF_TWO_CORE_THRESHOLD = 32768;  // 双核运行界限
constexpr uint32_t ERF_MAX_TILE_LENGTH = 8192;      // 最大切分长度

struct ErfTilingData {
    uint32_t mode = ERF_MODE_SMALL;                 // 模式选择
    uint32_t totalLength = 0;
    uint32_t blockLength = 128;
    uint32_t usedCoreNum = 1;
    uint32_t tileLength = 128;
    uint32_t tailLength = 0;
    uint32_t maxCalcLength = 128;
};
